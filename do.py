import tensorflow as tf
import numpy as np
from cut_image import *
from nn import *
#'''''


#给路径，返回判断出的列表
def shibie(path):
    im = readIMAGE(path)

    spilt_picture = cut_image_better(im)
    model = Mymodel()
    y = []
    for pic in spilt_picture:
        pre = prediect(img=pic, model=model)
        y.append(pre)
    return y

#'''''