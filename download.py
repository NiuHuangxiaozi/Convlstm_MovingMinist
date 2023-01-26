
from tqdm import tqdm
import zipfile
def unzip_file(zip_src, dst_dir):
    """
    解压缩
    :param zip_src:
    :param dst_dir:
    :return:
    """
    r = zipfile.is_zipfile(zip_src)
    if r:
        fz = zipfile.ZipFile(zip_src, 'r')
        bar = tqdm(fz.namelist())
        bar.set_description("unzip  " + zip_src)
        for file in bar:
            fz.extract(file, dst_dir)
    else:
        print('This is not zip')

unzip_file("./mnist_test_seq.zip", "./raw")