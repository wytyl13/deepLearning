# -*- coding: utf-8 -*-
#*****************************************************************
#   Copyright (C) 2022 IEucd Inc. All rights reserved.
#   
#   @Author: weiyutao
#   @Created Time : 2022/8/13 8:38:43
#   @File Name : CnnCreate.py
#   @Description : 
#
#*****************************************************************
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import tensorflow

torch.set_printoptions(linewidth = 120)


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 1,out_channels = 6,kernel_size = 5)#第一隐藏层，卷积层，输入通道是1，输出通道是6也就是第一隐藏层过滤器的数量是6.因为第一隐藏层的上一层是输入层，也就是没有卷积，所以输入层的通道是颜色通道的数量，灰度图像的输入通道是1，而RGB的输入通道是3.kernel_size对应的是过滤器的宽高，也就是第一隐藏层是一个5*5的过滤器，拥有6个这样的过滤器
        self.conv2 = nn.Conv2d(in_channels = 6,out_channels = 12,kernel_size = 5)#第二隐藏层，卷积层，输入通道是6，输出通道是12，也就是第二隐藏层有12个大小为5*5的过滤器

        self.fc1 = nn.Linear(in_features = 12*4*4,out_features = 120)#第一隐藏层的线性层，输入是12*4*4，输出张量是120
        self.fc2 = nn.Linear(in_features = 120,out_features = 60)#第二隐藏层的线性层，上面定义了其卷积参数，输入是120，输入层的输出，输出是60
        self.out = nn.Linear(in_features = 60,out_features = 10)#输出层，不含卷积，输入张量是第二隐藏层的输出60，输出张量是10

#注意每一个神经网络层都可以定义线性层和卷积层，卷积层是在线性层的基础上操作的，其目的是为了识别特征。如果没有卷积层学习参数可以做到图像识别，但是如果加上卷积层，可以进行图像检测，可以对图像的每个像素的图形情况进行检测。
#一般只对隐藏层和输出层进行定义，不低输入层定义，而且神经网络的层数也不包含输入层。一般的输出层没有卷积。而隐藏层一般含有卷积，卷积定义的参数单位是输入通道和输出通道和kernel_size,上一层过滤器的数量，本卷积层过滤器的数量，和过滤器的大小，特殊的对于输入层和输出层，输入层的输出通道(也就是第一隐藏层的输入通道)由输入层的颜色通道决定，一般为1或者3，而输出层的输出通道一般等于输出层的张量大小。本案例中就是10


    def forward(self,t):
        return t
"""
    #重写tostring方法，在python中是__repr__方法，注意我们这里继承了python的Module类中的方法，打印实例化对象会自动输入对应的网络的字符串形式输出，我们可以在我们的自定义类中重写对应的该方法，python中是__rept__(self)方法，而java中是toString方法

    def __repr__(self):
        return "weiyutao"
"""



"""
kernel_size:sets the height and width of the filter
out_channels:sets depth of  the filter,this is the number of kernels inside the filter.one kernel produces one output channel.
out_features:sets the size of the output tensor.
in_features:the length of the flattend output from previous layer
注意内核和过滤器的异同，在神经网络中，我们经常互换这两个次，但是这两个词语还是有差距的。内核是2D张量，过滤器是包含内核集合的3D张量，我们将内核应用于单个通道，将过滤器应用于多个通道。比如某一层的过滤器的数量是3个，而其中某一个内核的维度是什么？也就是说单个过滤器称为内核，多个过滤器称为过滤器。也就是说内核的维度是3D，而内核的维度是2D。

当我们由卷积层切换到线性层的时候需要展开张量


以上就是我们对于神经网络建立的参数的总结：
    首先，我们有两种类型的参数：超参数和数据相关的超参数
        超参数是由我们手动定义的，而数据层的超参数是由数据决定的。比如上面的5个参数，kernel_size是由我们手动定义的，属于超参数，而第一个卷积层的输入通道in_channel就是由输入数据的颜色通道决定的，属于数据相关的超参数，而输出层的out_feature取决于我们训练集中存在的类的数量，比如在Fashion-MNIST数据集中有服装类10个，所以该输出属于数据相关的超参数。对应的非输出层的输出out_channels，比如隐藏层的输入完全是由过滤器的数量决定的，所以它是超参数，而对于隐藏层线性层的输出out_features，也不完全是由数据决定的，所以它是超参数。
综上：超参数是由用户自定义的，而数据相关的超参数则完全是由数据决定。
所以，以上两个最基本的数据相关的超参数是第一卷积层的input_channels超参数和输出层的out_features超参数

"""



if __name__ == "__main__":
    network = Network()
    print(network)
    print(network.conv1)
    print(network.conv1.weight.shape)#卷积层的权重张量，该张量的形状反映了卷积层的所有信息，6*1*5*5，是一个rank-4张量，第一个轴长6表示过滤器的数量，第二个轴长1表示颜色通道数量，第三个第四个轴长分别表示过滤器的大小,其实权重张量就是很好的反映了每一个卷积层过滤器的所有信息。
    
    print(network.fc1.weight.shape) 
    
    #张量的相乘A.matmul(B)
    #访问网络参数
    for param in network.parameters():
        print(param.shape)

    
    #还有一个方法可以返回两个值，name和对应的参数，使用户的是named_parameters
    for name,param in network.named_parameters():
        print(name,'\t\t',param.shape)







































