# -*- coding: utf-8 -*-
#*****************************************************************
#   Copyright (C) 2022 IEucd Inc. All rights reserved.
#   
#   @Author: weiyutao
#   @Created Time : 2022/8/8 16:37:58
#   @File Name : linregTorch.py
#   @Description :使用现成的torch框架实现线性回归 
#
#*****************************************************************
import numpy as np
import pandas as pd
from torch import nn
from torch.utils import data
import torch

#根据传入的W和b参数维度生成例题的线性回归模型y = X@W+b
def synthenic_data(W,b,m):
    X = torch.normal(0,1,(m,len(W)))
    y = torch.matmul(X,W) + b
    y += torch.normal(0,0.01,y.shape)
    return X,y.reshape((-1,1))

#分小组进行梯度下降，生成对应的mini_batch
def load_array(data_arrays,batch_size,is_train = True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset,batch_size,shuffle = is_train)

#真正的参数，因为例题是我们出的，我们最后的结论就是判断拟合的参数和真正参数的误差
true_W = torch.tensor([2,-3.4])
true_b = 4.2

#根据真正的参数的维度随机生成自己的特征值和标签数据，生成的数据符合正太分布
features,labels = synthenic_data(true_W,true_b,1000)

#得到训练特征值和标签数据以后，我们对其进行分组得到mini_batch
batch_size = 10
data_iter = load_array((features,labels),batch_size)
next(iter(data_iter))



#定义模型的结构，本案例是输入2输出1单层神经网络也就是简单线性回归
#sequential相当于一个列表容器，将神经网络每层的参数和单元数还有一些更细节的东西储存起来，储存到一个net对象中
net = nn.Sequential(nn.Linear(2,1))#单层网络，输入是2输出是1


#初始化参数
#所有的细节都保存在net第0个索引值的位置上，weight是W，bias是b参数
#下面的操作是初始化W和b参数，W符合均值为0，偏差为0.01的正态分布，b为0
net[0].weight.data.normal_(0,0.01)#net中存储的是每层神经网络的层数和W和b，对应的w是net[0].weight对应的b是net[0].bias，想要获取数据.data
net[0].bias.data.fill_(0)

loss = nn.MSELoss()#计算均方误差，也就是线性回归的损失函数也称为平方范数
#实例化sgd，也就是训练函数
trainer = torch.optim.SGD(net.parameters(),lr = 0.03)

iterators = 3
for epoch in range(iterators):
    for X,y in data_iter:
        l = loss(net(X),y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features),labels)
    print(loss(1:1))































