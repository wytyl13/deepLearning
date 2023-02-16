# -*- coding: utf-8 -*-
#*****************************************************************
#   Copyright (C) 2022 IEucd Inc. All rights reserved.
#   
#   @Author: weiyutao
#   @Created Time : 2022/8/12 9:28:20
#   @File Name : ConvNuetualNetWork.py
#   @Description :卷积神经网络
#
#*****************************************************************
import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



class NetWork(nn.Module):
    def _init_(self):
        super()._init_()
        #注意这里的参数不要和可学习参数搞混淆，此处不同于每层的可学习参数的推导，但是每层的输入和输出和可学习参数是对应的，只不过可学习参数的公式是每层的输入*输出+偏差（偏差数量就是该层的过滤器数量）
        self.conv1 = nn.conv2d(in_channels = 1,out_channels = 6,kernel_size = 5)#第一个卷积层，因为输入层没有卷积，所以没有过滤器，所以输出是1，对应的第一个卷积层的输入就是1，第一个卷积层的输出等于第一个卷积层的过滤器的数量，本案例是6，所以可以判断出第一个卷积层的过滤器数量是6
        self.conv2 = nn.conv2d(in_channels = 6,out_channels = 12,kernel_size = 5)#第二个卷积层的输入是第一个卷积层的输出，为6，第二个卷积层的输出就是第二个卷积层的过滤器，为12.
        
        self.fc1 = nn.Linear(in_features = 12 * 4 * 4,out_features = 120)#输入层，三阶张量，轴长分别为12*4*4，
        self.fc2 = nn.Linear(in_features = 120,out_features = 60)#第一个隐藏层的线性输入就是输入层的输出120，输出特征是60，可以发现减少了一半
        self.out = nn.Linear(in_features = 60,out_features = 10)#输出层的输入是第二层的输出60，输出层的输出是10.可以推断是多分类的问题，需要分类20个数
    
    def forward(self,t):
        return t

















