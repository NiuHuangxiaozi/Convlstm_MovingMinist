import torch
import torch.nn as nn

class ConvLstmCell(nn.Module):
    '''
     input_vector:
                [ Batch_size  , input_channel , height , width]
    '''
    def __init__(self,input_channel,hidden_channel,kernel_size):
        super(ConvLstmCell, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_channel=input_channel
        self.hidden_channel=hidden_channel
        self.kernel_size=kernel_size
        self.padding=kernel_size // 2

        self.conv=nn.Conv2d(self.input_channel+self.hidden_channel,
                                    self.hidden_channel*4,
                                    kernel_size=self.kernel_size,
                                    padding=self.padding,
                                    bias=True)
    def init_hidden(self,x):
        batchsize,channel,height,width=x.shape
        return torch.zeros(batchsize,self.hidden_channel,height,width,device=self.conv.weight.device),\
               torch.zeros(batchsize, self.hidden_channel, height, width,device=self.conv.weight.device)
    def forward(self,x,hn=None,cn=None):
        '''
        :param x:[batch,channel,height,width]
        :param hn:
        :param cn:
        :return:
        '''
        assert (hn==None)^(cn==None)==0
        if hn==None:
            hn ,cn =self.init_hidden(x)
        #先将输入和hn拼在一起

        input_and_hn=torch.cat((x,hn),dim=1)
        temp=self.conv(input_and_hn)
        temp_f,temp_i,temp_g,temp_o=torch.split(temp,self.hidden_channel,dim=1)


        f=nn.Sigmoid()(temp_f)
        i=nn.Sigmoid()(temp_i)
        g=nn.Tanh()(temp_g)
        o=nn.Sigmoid()(temp_o)

        C_t=f*cn+i*g
        h_t=o*nn.Tanh()(C_t)
        return h_t,C_t


class ConvLstm(nn.Module):
    def __init__(self,input_channel,hidden_channel,kernel_size,num_layers=1,
                 batch_first=True):
        super(ConvLstm, self).__init__()
        '''
        input_vector:
                [ Batch_size , time_step , input_channel , height,width]
                
        input_channel:such as rgb
        hidden_channel: inner representations
        kernel_size:kernel size
        num_layers：the lay of lstm,default=1
        batch_first:if TRUE, the first dimension is batch_size.
        bias: bias
        dropout:whether use dropout to fight back over-fitting.   
        '''
        self.input_channel=input_channel
        self.hidden_channel=hidden_channel
        self.kernel_size=kernel_size
        self.num_layers=num_layers
        self.batch_first=batch_first
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        #construct the network
        #[ cell1 , cell2 , cell3 ,  cell4 , ...... ]
        self.layers = []
        for lay_index in range(self.num_layers):
            input_channel=self.input_channel if lay_index==0 else self.hidden_channel
            Cell = ConvLstmCell(input_channel,self.hidden_channel,self.kernel_size).to(self.device)
            self.layers.append(Cell)


    def forward(self,x,initial_hn=None,initial_cn=None):
        '''
        input : [B , S, C=input_channel , H , W]  or   [ S , B , C , H , W]
        output : [ B , S , C = hidden_channel , H , W]
        '''
        if self.batch_first ==True:
                x=x.transpose(0,1)#exchange 0 and 1 dimension
        #the current input shape is  [ S , B , C , H , W]

        #inite the hn ,cn
        '''
        self.hn_cns's shape:
          [(hn_0,cn_0) , (hn_1,cn_1) , (hn_2,cn_2) , ...... , (hn_num_layers,cn_num_layers]        
        '''
        assert (initial_hn==None)^(initial_hn==None)==0
        hn_0=None
        cn_0=None
        if initial_hn==None:
            hn_0,cn_0=self.Initial_hidden(x)
        else:
            hn_0, cn_0=initial_hn,initial_cn

        hn_cns = []
        for lay in range(self.num_layers):
                hn_cns.append([hn_0,cn_0])

        #if the lay is zero the cell inputtensor is x or else the upper hn
        #lay_hn is used to preserve the upper hn
        lay_hn=None
        out_put_hns=[]

        time_step = x.shape[0]
        for step in range(time_step):#iterator the step
            for lay in range(self.num_layers):#every step has many layers
                input_tensor=None

                if lay==0:#if the lay is 1,input tensor is x
                    input_tensor=x[step]
                else:#else the iput is the upper floor 's hn
                    input_tensor = lay_hn

                #run convlstmcell
                hn_cns[lay][0], hn_cns[lay][1] = \
                        self.layers[lay](input_tensor, hn_cns[lay][0],hn_cns[lay][1])
                lay_hn=hn_cns[lay][0].data#reserve the upper hn

                if lay==self.num_layers-1:#the final lay ,so reserve the final hns as output
                    out_put_hns.append(lay_hn)

        #gather evey step of hn
        out_put=torch.stack([hn for hn in out_put_hns],dim=0)

        last_hn=out_put_hns[-1]

        return out_put,last_hn

    def Initial_hidden(self,x):
        '''
        h_n: [batch_size, inner_channel(=hidden_dim),height,width]
        c_n:[batch_size, inner_channel(=hidden_dim),height,width]
        '''
        time_steps, batch_size, input_channel, height, width = x.shape
        return  \
            (
            torch.zeros(batch_size,self.hidden_channel,height,width).to(self.device),
            torch.zeros(batch_size, self.hidden_channel, height,width).to(self.device)
            )





#moving minist encoder-forcasting model


class encoder_forcasting(nn.Module):
    def __init__(self,input_channel,hidden_channel,out_put):
        super(encoder_forcasting, self).__init__()
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_channel=input_channel
        self.hidden_channel=hidden_channel
        self.output_channel=out_put
        self.kernel_size=3

        self.encoder_lay_one = ConvLstmCell(self.input_channel,
                                                                  self.hidden_channel,
                                                                  kernel_size=self.kernel_size
                                                                )

        self.encoder_lay_two = ConvLstmCell(self.hidden_channel , self.hidden_channel,
                                                                 kernel_size=self.kernel_size
                                                                 )

        self.decoder_lay_one=ConvLstmCell(self.hidden_channel,
                                                                self.hidden_channel,
                                                                kernel_size=self.kernel_size
                                                                )

        self.decoder_lay_two=ConvLstmCell(self.hidden_channel,
                                                                 self.hidden_channel,
                                                                 kernel_size=self.kernel_size
                                                                 )

        self.conv3d=nn.Conv3d\
                (
                              self.hidden_channel,
                              self.output_channel,
                              kernel_size=(1,3,3),
                              padding=(0,1,1)
                                    )
    def forward(self,x,future_step):
        b,seq_len, _, h, w = x.size()
        # initialize hidden states
        h_t, c_t = self.encoder_lay_one.init_hidden(x[:,0,:,:])
        h_t2, c_t2 = self.encoder_lay_two.init_hidden(x[:,0,:,:])
        h_t3, c_t3 = self.decoder_lay_one.init_hidden(x[:,0,:,:])
        h_t4, c_t4 = self.decoder_lay_two.init_hidden(x[:,0,:,:])

        #encoder
        for predict_index in range(seq_len):
            h_t, c_t=self.encoder_lay_one(x[:,predict_index,:,:],h_t,c_t)

            h_t2, c_t2 = self.encoder_lay_two(h_t,h_t2,c_t2)

        #####
        encoder_vector=h_t2
        outputs=[]

        # decoder
        for t in range(future_step):
            h_t3, c_t3 = self.decoder_lay_one(x=encoder_vector,hn=h_t3,cn=c_t3)
            h_t4, c_t4 = self.decoder_lay_two(x=h_t3,hn=h_t4,cn= c_t4)
            encoder_vector = h_t4
            outputs += [h_t4]  # predictions
        outputs = torch.stack(outputs, 1)
        outputs = outputs.permute(0, 2, 1, 3, 4)
        outputs = self.conv3d(outputs)
        outputs = torch.nn.Sigmoid()(outputs)

        return outputs



if __name__=="__main__":
    input_channel=1
    hidden_channel=128
    kernel_size=3
    num_layers=1
    predict_steps=10
    x = torch.Tensor(32, 10, 1, 64, 64).cuda()
    model=encoder_forcasting(input_channel,hidden_channel,1).cuda()

    answer=model(x,predict_steps)


