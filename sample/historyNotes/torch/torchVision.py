# -*- coding: utf-8 -*-
#*****************************************************************
#   Copyright (C) 2022 IEucd Inc. All rights reserved.
#   
#   @Author: weiyutao
#   @Created Time : 2022/8/10 11:25:52
#   @File Name : torchVision.py
#   @Description :构建第一个ETL过程
#
#*****************************************************************
import torch#顶级torch包和张量库
import torch.nn as nn#用于构建神经网络的模块和可扩展类的子包
import torch.optim as optim#包含标准优化库比如SGD和Adan的子包
import torch.nn.functional as F#一个功能接口，包含用于构建神经网络的典型操作，如损失函数和卷积

import torchvision#一个包，提供对流行数据集、模型架构和计算机视觉图像转换的访问
import torchvision.transforms as transforms#包含用于图像处理的常用转换的接口

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow
from sklearn.metrics import confusion_matrix

torch.set_printoptions(linewidth = 120)



#下载数据集到当前文件夹下，如果有的话检查覆盖，没有的话创建data文件夹然后将数据下载进去,注意这个下载是固定格式.，每次下载之前检查数据，不用担心会重复下载
train_set = torchvision.datasets.FashionMNIST(
    root = './data',
    train = True,
    download = True,
    transform = transforms. Compose([transforms.ToTensor()])
)

#创建一个DataLoader包装器来加载数据，train_set只是作为一个参数来进行传递的,我们可以想象将train_set数据源传入加载器就可以得到根据batch_size分组好的数据集
trian_loader = torch.utils.data.DataLoader(train_set,batch_size = 1000,shuffle = True)
#到这我们其实已经完成了ETL的所有任务，我们使用了train_set对象保存了下载的网络资源，在下载的时候就已经转换为自己需要的格式了。然后使用加载器train_loader将下载好的数据进行分组存储,保存在train_loader中方便存取

print(len(train_set))#计算train_set也就是数据源的训练集的样本量
print(train_set.targets.shape)#targets是label也就是标签，也就是y,是一阶张量，轴长是60000
print(train_set.targets.bincount())#统计每个label出现的次数，注意此处bincount的用法其实就是统计求和的结果,可以发现每个label都出现了6000次，一共有10个label。那么对于这样的每个label有着相同的数量，我们称该数据集是平衡的，否则我们称之为不平衡的数据集。当然对于我们的日常项目来说，我们的数据集大多是不平衡的，而我们需要做的就是尽量减少数据的不平衡。所以平衡的数据集对于我们的训练是近乎完美的

#上面我们已经得到数据集了，那么在pytorch框架下我们如何从数据集中获取对应的数据呢，这里我们需要使用item和next方法
sample = next(iter(train_set))#next可以获取到第数据集中的第一个图像，注意next取到的数量伴随着数据集是否定义batchsize而变，如果batchsize = 10，那么next取到的就是10个图像，因为这里的sample是c从源数据也就是trainset中next的并没有定义batchsize，所以只会取到一个。
print(len(sample))#sample中存储的是训练集数据和标签数据，对应的长度是2，其中包含了特征数据和label标签
image,label = sample#使用image保存特征数据也就是图像的数据
print(image.shape,'\n',label)#lable是标签，所以label的类型是int型，而image保存的是一张图片，所以image是一个三阶张量，对应的轴长分别为1*28*28

#下面我们可以将image画出来
#注意因为我们的颜色通道的轴长是1，所以我们在画图的时候需要将该维度进行挤压。可以使用squeeze方法挤压掉轴长为1的轴
#plt.imshow(image.squeeze(),cmap = "gray")
#plt.show()


#以上我们已经可以取出一个图像了，下面我们可以取出一批，需要使用加载器
display_loader = torch.utils.data.DataLoader(train_set,batch_size = 10)#取出10个图像，放在display_loader对象中，然后使用next和iter函数取出来，注意这里的shuffle参数，默认的是关闭状态，如果开启的话，则每次调用next的时候批次都会不同
batch = next(iter(display_loader))#batch对象返回的是图像集合，返回多少张图像得看batchsize是多大，本案例中batch返回的批次是10，因为我们是从batchsize为10的display_loader中next的，而对应的获取的时候的shuffle是否开启会导致next方法获取数据集的时候是否是从每一个批次中获取到第一个数据。如果shuffle=false默认情况，那么会返回第一批次的数据集。
print("len:",len(batch))#此处的batch的尺寸还是2，因为它保存了数据集和标签
images,labels = batch
print(images.shape,labels.shape)#images是一个四阶张量，对应的轴长分别为10*1*28*28，labels是一个一阶张量，对应的轴长是10


"""
绘制批量图像1
#根据以前学习到的张量的基本操作，如果我们想要获取某一个张量中的某一个元素我们需要使用索引的方式获取。images[0]获取到的是第一个图像
#我们可以使用torvision中的创建grid的方法创建栅栏以使得我们可以画出批量的图像
grid = torchvision.utils.make_grid(images,nrow = 10)
plt.figure(figsize = (15,15))
plt.imshow(np.transpose(grid,(1,2,0)))
plt.show()
#plt.imshow(grid.permute(1,2,0))该用法类似于上步的代码
"""

#下面我们使用另一种办法批量绘制图像，可以更加智能的绘制出想要的图片的数量
#我们的绘制方法是先定义batch_size = 1,这样我们将原始数据集分成了等数量的份，本案例是分成了60000份，我们将shuffle参数定义为true然后开始批量绘制
#shuffle参数为true意味着会从每一个批次中取一个，如果直接使用next，会取60000次，但是我们可以通过限定循环条件来取出想要的数量，如果shuffle为false，那么next只会从第一批次取数据
how_many_to_plot = 20#首先定义需要绘制的图像数量
train_loader = torch.utils.data.DataLoader(train_set,batch_size = 1,shuffle = True)
plt.figure(figsize = (50,50))
for i,batch in enumerate(train_loader,start = 1):#enumerate会返回当前数据和对应的索引
    image,label = batch
    plt.subplot(10,10,i)#三个参数分别为批量图显示的总行数 总列数 和第几个图
    plt.imshow(image.reshape(28,28),cmap = 'gray')#此处的reshape等价于squeeze挤压，都是将轴长为1的轴挤压掉
    plt.axis('off')
    plt.title(train_set.classes[label.item()],fontsize = 14)#此处是将每张图片的label作为title、我们也可以直接显示出每个label对应的名称，在train_Set中有
    if(i >= how_many_to_plot):#判断条件以在打印出想要的数量以后停止
        break
plt.show()


#以上我们已经完成了我们训练的第一步，也即准备数据的过程，下来我们将进行模型的构建过程，此时我们将使用torch中的工具，在torch.nn.Moudle类
#需要构建神经网络，我们需要对CNN的工作原理以及用于构建CNN的组件有一个大致的了解。
#paddingtype:valid nopadding;same zeros around the edges.前者是不使用零填充，后者是使用零填充，前者是卷积层会减少输入的轴长，使得输出的轴长小于输入的。后者是在卷积操作之前零填充输入，保证在卷积操作之后轴长等于原始输入的轴长
#working with code in keras
import keras
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense,Flatten
from keras.layers.convolutional import *

#now we will create a completely arbitrary CNN
model_valid = Sequential([
    Dense(16,input_shape = (20,20,3),activation = 'relu'),
    Conv2D(32,kernel_size = (3,3),activation = 'relu',padding = 'valid'),
    Conv2D(64,kernel_size = (5,5),activation = 'relu',padding = 'valid'),
    Conv2D(128,kernel_size = (7,7),activation = 'relu',padding = 'valid'),
    Flatten(),
    Dense(2,activation = 'softmax')
])
#remember from earlier that,valid means no padding.
#the Dense,the first code means that input shape is 20*20,and the convolutional layers numbers is 3.the code rows mean the layers
#第一行代码Dense代表输入密集层，输入的数据维度是20*20，总共有三个卷积层，然后第二行代码表示的是第一个卷积层的过滤器的轴长3*3，第二个是5*5，第三个是7*7，valid表示无令填充。activation表示该层使用的激活函数是relu，最后一行Dense表示的是输出层。每一层对应的如果存在卷积行为，也就是存在过滤器，那么该层会被卷积，该层的输出会被减少（因为该案例是无零填充的情况，所以会造成维度的下降），我们可以推断，输入层的输入是20*20，然后输入层没有过滤器，所以输入层不会被执行卷积操作，输入层的输出也就是第一隐藏层的输入还是20*20，而第一隐藏层存在卷积，轴长为3*3，所以第一隐藏层的输入为20*20，输出为20-3+1*20-3+1 = 18*18，第一隐藏层的输出对应的就是第二层的输入，第二层的卷积轴长是5*5，多以第二层的输出轴长为14*14，第三层的输入是14*14，卷积轴长是7*7，所以第三层的输出是8*8，对应的输出层的输入就是8*8.输出层案例中没有卷积操作。那么前面的数字代表的是什么呢

#对应的，我们也可以建立same模式的模型，也就是加入零填充。只需要将padding的属性值更改为same即可，该模式下每层的输入和输出的轴长固定，均为20*20



#下面我们来考虑建立最大池化的代码，这个将使用keras库
from keras.layers.pooling import *

#下面我们可以建立一个完全任意的卷积神经网络模型，。其中有某一层会考虑加入最大池化
model_valid1 = Sequential([
    Dense(16,input_shape(20,20,3),activation = 'relu'),
    Conv2D(32,kernel_size = (3,3),activation = 'relu',padding = 'same'),
    MaxPooling2D(pool_size(2,2),strides = 2,padding = 'valid'),
    Conv2D(64,kernel_size = (5,5),activation = 'relu',padding = 'same'),
    Flatten(),
    Dense(2,activation = 'softmax')
])

#注意以上我们在卷积神经网络的第一个卷积层之后加入了池化层，也就是说这个池化层会影响到第一隐藏层也就是第一卷积层的输出，在第一隐藏层进行卷积操作后，会对卷及操作的输出使用2*2大小，步幅为2的过滤器进行池化，然后将其作为第一隐藏层的输出，作为第二隐藏层的输入，然后第二隐藏层会进相应的卷及操作，注意这里的卷积操作使用的是same参数也就是使用了零填充。最后是输出层.注意池化层并不都会使得输入的轴长减半，只有当池化层的过滤器的轴长是2*2并且步伐是2的时候才会发生这样的情况，至于其他情况如何我们需要进行计算。
























    






























