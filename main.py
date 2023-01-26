
from torch.utils.data import Dataset, DataLoader
from my_model import  encoder_forcasting
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import math


print(torch.cuda.is_available())
print(torch.cuda.device_count())

def save_as_image(flag,data,batchindex):
    data=np.reshape(data,(data.shape[0],data.shape[1],64,64))
    data.astype(np.uint8)
    for index in range(len(data[:, 0, 0, 0])):
        count = 0
        for j in data[index, ...]:
            img = np.expand_dims(j, -1)
            filename = '%s/test%d/video_%d/frame%d.jpg' % (
                flag, batchindex,index,count)
            count += 1
            savepath1 = './%s/test%d' % (flag,batchindex)
            savepath2 = './%s/test%d/video_%d' % (flag, batchindex, index)
            if not os.path.exists(savepath1):
                os.mkdir(savepath1)
            if not os.path.exists(savepath2):
                os.mkdir(savepath2)
            print(img)
            cv2.imwrite(filename, img*255)

def  show_result(epoches , losses):
    plt.title('Result Analysis:loss')
    plt.plot(epoches,losses, color='black', label='Loss')
    plt.legend()  # 显示图例
    plt.xlabel('epoches')
    plt.ylabel('loss')
    plt.show()


train_batchsize=100
test_batchsize=16
#dataset
from MovingMNIST import MovingMNIST
train_set = MovingMNIST(root='./', train=True, download=True)
train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=train_batchsize,
                 shuffle=True)
test_set = MovingMNIST(root='./', train=False, download=True)
test_loader = torch.utils.data.DataLoader(
                 dataset=test_set,
                 batch_size=test_batchsize,
                 shuffle=False)



#模型训练一些参数的配置
input_dim=1
hidden_dim=64
output_dim=1
pre_length=10
epoch= 100


#创建存储原来图片和预测图片的文件夹
raw_photo="original"
answer_photo="final"
if not os.path.exists(raw_photo):
    os.mkdir(raw_photo)
if not os.path.exists(answer_photo):
    os.mkdir(answer_photo)



#in main function we train and test
if __name__=="__main__":
    #define device
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #define model
    model = encoder_forcasting(input_dim,hidden_dim,output_dim).to(device)

    #并行化模块，如果只有1块gpu，注释下面这一行
    model = torch.nn.DataParallel(model).to(device)

    #define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    #define loss function
    criterion=nn.MSELoss().cuda()

    losses=[]
    epoches=[i for i in range(1,epoch+1)]

    #train
    model.train()
    for epc in range(epoch):
        epoch_loss=[]
        for batchidx, (data, label) in enumerate(train_loader):
            data=data/255
            label=label/255
            '''
            data:[batch,timestep=10,height=64,weight=64]
            经过unsqueeze之后变为
            data:[batch,timestep=10,channel=1,height=64,weight=64]
            '''
            data = data.unsqueeze(dim=2).to(device).float()
            label = label.unsqueeze(dim=2).to(device).float()

            answer_pred = model(data, pre_length)

            answer_pred=answer_pred.squeeze()
            label=label.squeeze()
            '''
            经过squeeze变化：
            before:[batch,timestep=10,channel=1,height=64,weight=64]
            after:[batch,timestep=10,height=64,weight=64]
            '''
            loss = criterion(answer_pred, label)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=10.0)
            optimizer.step()

            print(f'\r batchs [{epc + 1}/{epoch + 1}] Loss: {loss.item() / label.shape[0]:.10f}', end='')
            epoch_loss.append(loss.detach().item()*label.shape[0])
        losses.append(sum(epoch_loss)/len(train_set))
        #打印这一轮的错误
        print("Epoch", epc + 1,"/",epoch+1 ,"Loss:", sum(epoch_loss)/len(train_set))

    #保存模型
    torch.save(model.state_dict(), "my_model_100_1.pt")

    #展示loss-epoch的变化
    show_result(epoches, losses)

    # 读取已经训练好的模型
    '''
    #model = EncoderDecoderConvLSTM(hidden_dim, input_dim).to(device)
    #model = torch.nn.DataParallel(model).to(device)
    model = encoder_forcasting(input_dim, hidden_dim, output_dim).to(device)
    model = torch.nn.DataParallel(model).to(device)
    model.load_state_dict(torch.load("my_model_100.pt"))
    criterion = nn.MSELoss().cuda()
    '''
    #模型的测试阶段
    model.eval()
    test_loss=[]
    for batchidx, (test_data, test_label) in enumerate(test_loader):

        test_data=test_data/255
        test_label=test_label/255

        save_as_image(raw_photo,test_label.numpy(), batchidx)

        test_data = test_data.unsqueeze(dim=2).to(device).float()
        test_label = test_label.unsqueeze(dim=2).to(device).float()

        test_data_pred = model(test_data,pre_length)
        test_data_pred= test_data_pred.transpose(1, 2)
        loss = criterion(test_data_pred, test_label)

        save_as_image(answer_photo,test_data_pred.cpu().detach().numpy(), batchidx)
        break

    print("train_test_over")





