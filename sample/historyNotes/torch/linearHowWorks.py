# -*- coding: utf-8 -*-
#*****************************************************************
#   Copyright (C) 2022 IEucd Inc. All rights reserved.
#   
#   @Author: weiyutao
#   @Created Time : 2022/8/14 13:42:50
#   @File Name : linearHowWork.py
#   @Description : 
#
#*****************************************************************
import numpy as np
import torch
import torch.nn as nn





in_features = torch.tensor([1,2,3,4],dtype = torch.float32)#注意定义的基础张量是向量，前面只有一个中括号是一维列向量，两个括号是二维数组，也就是矩阵，注意这个区别，张量只要是1阶的都是向量，注意这个重要特点。4*1
weight_matrix = torch.tensor([
    [1,2,3,4],
    [2,3,4,5],
    [3,4,5,6]
],dtype = torch.float32)
#3*4



result1 = weight_matrix.matmul(in_features)#3*4 @ 4*1 = 3*1，向量
print(result1)
#通常权重矩阵定义了一个线性函数，它将具有四个元素的一维张量映射到具有三个元素的一维张量，我们将此函数视为从4维欧几里得空间到3维欧几里得空间的映射。这也就是线性层工作的方式，他们使用权重矩阵，将将空间infeature映射到outfeature空间。
#也就是我们定义了一个4个元素的向量，然后使用权重矩阵，将该4个元素的向量降维到三个元素的向量。那么对应的权重矩阵的维度就是输出维度*输入维度，也就是3*4，对应的pytorch源码也是这样定义的。使用权重矩阵的乘法来进行降维。



fc = nn.Linear(in_features = 4,out_features = 3,bias = False)#定义一个线性层，根据输入特征和输出特征torch就可以初始化权重矩阵

#然后使用fc对象作为线性层去降低输入特征的维度
result2 = fc(in_features)
print(result2)
#可以发现已经达到了降维的效果，而这个值和前面的值不同的原因是因为torch使用的是权重矩阵随机初始化的方法，所以会不同
#我们可以明确的设置linear方法中的权重矩阵的值，这样我们就可以得到和reault1完全一样的结果
fc.weight = nn.Parameter(weight_matrix)#将自定义的权重矩阵的变量赋值给fc的权重矩阵，使用的是nn.paramter方法传递
result3 = fc(in_features)
print(result3)













































