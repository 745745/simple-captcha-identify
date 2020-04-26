import numpy as np
from PIL import Image
from queue import Queue
import os


#把图片读入并适度去除噪点,有笔画255，没笔画0
def readIMAGE(path):
    IMAGE=Image.open(path).convert('L')
    IMAGE_value=np.array(IMAGE)

    for i in range(len(IMAGE_value)):
        for j in range(len(IMAGE_value[0])):
            if(IMAGE_value[i][j]<110):
                IMAGE_value[i][j]=255
            else:
                IMAGE_value[i][j] = 0
    return IMAGE_value

#检测周围点是否有连通，有连通则入队
def connected(im,im_2,list,que):

    x=list[0]
    y=list[1]
    y_max = 0
    y_min=y
    #因为小写的j是断开的，所以要多弄点范围
    a = np.arange(-1, 2)
    b = np.arange(-1,2)
    de = []
    for i in a:
        for j in b:
            if (i != 0 or j != 0):
                de.append([i, j])

    for i in de:
        if(x+i[0]<len(im) and y+i[1]<len(im[0]) and im[x+i[0],y+i[1]]==255 and (not im_2[x+i[0],y+i[1]])):
            que.put([x+i[0],y+i[1]])
            im_2[x+i[0],y+i[1]]=1
            if(y+i[1]>y_max):
                y_max=y+i[1]
            if(y+i[1]<y_min):
                y_max=y+i[1]
    return y_max,y_min

#连通图法分割，适用于不粘连的验证码
def cut_image_better(im):
    #最大小值
    y_max=[]
    y_min=[]
    #临时储存最大小值
    p_max=[]
    p_min=[]
    #im_2用来查看该点是否被访问
    im_2=np.zeros((len(im),len(im[0])))
    a=Queue(im.size)
    #开始求连通里面x坐标最大的
    for i in range(len(im[0])):
        for j in range(len(im)):
            if(im[j,i]==255 and(not im_2[j,i])):
                a.put([j,i])
                im_2[j,i]=0
                while(not a.empty()):
                    x = a.get()
                    d,b=connected(im, im_2, x, a)
                    p_max.append(d)
                    p_min.append(b)
                y_max.append(max(p_max))
                y_min.append(min(p_min))
                p_max.clear()
                p_min.clear()
    spilt_picture=[]
    spilt_picture_value=[]

    for i in range(len(y_max)):
        im1=Image.fromarray(im[:,y_min[i]-2:y_max[i]+2])
        #im1.show()
        spilt_picture.append(im1)

    for img in spilt_picture:
        img = img.resize((32, 32), Image.ANTIALIAS)
        ima = np.array(img)
        for i in range(len(ima)):
            for j in range(len(ima[0])):
                if (ima[i][j] < 150):
                    ima[i][j] = 0
                else:
                    ima[i][j] = 255
        #ima=ima/255.0
        spilt_picture_value.append(ima)
    return spilt_picture_value


'''''
path="C:/Users/17251/Desktop/06.jpg"
im=readIMAGE(path)

pc=cut_image_better(im)
for img in pc:
    im=Image.fromarray(img)
    im=im.resize((32,32),Image.ANTIALIAS)
    im.show()
    ima=np.array(im)
'''''