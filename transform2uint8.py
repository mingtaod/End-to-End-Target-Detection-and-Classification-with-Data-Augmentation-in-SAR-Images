from PIL import Image
import numpy as np
import math
import os

path = 'C:\\Users\\dongm\\Desktop\\Dataset\\mask' # the directory need to be changed according to the exact location of the dataset
newpath = 'C:\\Users\\dongm\\Desktop\\Dataset\\mask' # need to be changed according to the exact location of the dataset


def toeight():
    filelist = os.listdir(path)  # 该文件夹下所有的文件（包括文件夹）
    for file in filelist:
        whole_path = os.path.join(path, file)
        img = Image.open(whole_path)  # 打开图片img = Image.open(dir)#打开图片
        img = np.array(img)
        # img = Image.fromarray(np.uint8(img / float(math.pow(2, 16) - 1) * 255))
        img = Image.fromarray(np.uint8(img)*40)
        img.save(newpath + file)


toeight()
