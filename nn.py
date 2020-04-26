import tensorflow as tf
from tensorflow.keras import Model
from tensorflow import keras
from PIL import  Image
import numpy as np
import os
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
import cut_image as ct

#标签对应的字符
label=[]
for letter in range(47,58):
    label.append(chr(letter))
for letter in range(65,91):
    label.append(chr(letter))
for letter in range(97,123):
    label.append(chr(letter))


#读取数据集
def load_data(path):
    file=open(path,'r')
    contents=file.readlines()
    file.close()
    np.random.seed(5120)
    np.random.shuffle(contents)
    x,y=[],[]
    for content in range(30000):
        value=contents[content].split()
        pic=value[0]
        img=Image.open(pic)
        img=np.array(img.convert('L'))
        for i in range(len(img)):
            for j in range(len(img[0])):
                if(img[i][j]<200):
                    img[i][j]=255
                else:
                    img[i][j] = 0
        #im=Image.fromarray(img)
        #im.show()
        img=Image.fromarray(img)
        img = img.resize((32, 32), Image.ANTIALIAS)
        #img.show()
        img=np.array(img.convert('L'))
        img=img/255.0
        #img=tf.image.convert_image_dtype(img,tf.float32)
        img=np.resize(img,(32,32,1))
        x.append(img)
        y.append(int(value[1]))
    x = np.array(x)
    y = np.array(y)
    return x,y

#设置数据集
def data_set(x,y):
    np.random.seed(22220)
    np.random.shuffle(x)
    np.random.seed(22220)
    np.random.shuffle(y)

    test_number=25000
    x_train=x[:test_number]
    y_train=y[:test_number]
    x_test=x[test_number:]
    y_test=y[test_number:]
    return x_train,y_train,x_test,y_test

#定义网络
class Mymodel(Model):
    def __init__(self):
        super(Mymodel,self).__init__()
        self.flatten = Flatten()
        self.f1 = Dense(1024, activation='relu')
        self.d6 = Dropout(0.1)
        self.f2 = Dense(512, activation='relu')
        self.d7 = Dropout(0.2)
        self.f3=Dense(128,activation='tanh')
        self.d8=Dropout(0.2)
        self.f4 = Dense(63, activation='softmax')

    def call(self, x):
        x = self.flatten(x)
        x = self.f1(x)
        x = self.d6(x)
        x = self.f2(x)
        x = self.d7(x)
        x = self.f3(x)
        x = self.d8(x)
        y=self.f4(x)
        return y


#训练模型
def train(model,x_train,y_train,x_test,y_test):
    model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.005),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['sparse_categorical_accuracy'])

    checkpoint_save_path = "./checkpoint/yzm.ckpt"
    if os.path.exists(checkpoint_save_path + '.index'):
        model.load_weights(checkpoint_save_path)

    image_gen_train=tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.2,
        horizontal_flip=False,
        zoom_range=0.5,
    )
    call_back = [
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path, save_best_only=True, save_weights_only=True)
    ]
    history = model.fit(image_gen_train.flow(x_train, y_train, batch_size=256,shuffle=True), epochs=100, validation_data=(x_test, y_test),
                        validation_freq=1,
                        callbacks=call_back)
    model.summary()
    return history


#预测结果
def prediect(img,model):
    checkpoint_save_path = "./checkpoint/yzm.ckpt"
    if os.path.exists(checkpoint_save_path + '.index'):
        model.load_weights(checkpoint_save_path)
    img = tf.image.convert_image_dtype(img, tf.float32)
    x_prediect=img[tf.newaxis,...,tf.newaxis]
    result=model.predict(x_prediect)
    pred=tf.argmax(result,axis=1)
    pre=pred.numpy()
    return label[pre[0]]

def draw_loss_prc(history):
    acc = history.history['sparse_categorical_accuracy']
    val_acc = history.history['val_sparse_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()






'''''
x,y=load_data('data.txt')
x_train,y_train,x_test,y_test=data_set(x,y)
model=Mymodel()
history=train(model,x_train,y_train,x_test,y_test)
draw_loss_prc(history)
'''''

'''''
model=Mymodel()
path="C:/Users/17251/Desktop/character&number/Sample006/img006-00004.png"
img=ct.readIMAGE(path)
img=(Image.fromarray(img)).resize((32,32),Image.ANTIALIAS)
img=np.array(img.convert('L'))
y=prediect(img=img,model=model)
print(y)
'''''

'''''
def shibie(path):
    im = ct.readIMAGE(path)
    spilt_picture = ct.cut_image_better(im)
    model = Mymodel()
    y = []
    for pic in spilt_picture:
        pc=Image.fromarray(pic)
        pre = prediect(img=pic, model=model)
        y.append(pre)
    return y

z=shibie("c:/Users/17251/Desktop/2/10.jpg")
print(z)
'''''