'''================================================================
*   Copyright (C) 2022 IEucd Inc. All rights reserved.
*   
*   FileName:autoDerivatation.py
*   Author:weiyutao
*   CreateTime:2022-03-02
*   Describe:自动求导的实现
*
================================================================'''

import torch

x = torch.arange(4.0)#注意这块可以使用dtype来定义数值类型，但是要存储梯度一定要使用浮点类型的数据，使用int类型的数据是不行的；输出的x的结果是[0,1,2,3]
x.requires_grad_(True)#开辟一块地方来存储梯度，使用x.grad来访问，默认是0

#下面我们来计算y，y其实就是向量x的点积，但是在计算y之前我们需要开辟空间来存储我们即将计算的梯度，梯度其实就是求导，求y对x的导数
y = 2 * torch.dot(x,x)#求x的点积，其实就是2x^2

y.backward()#反向传播函数来自动计算y关于x每个分量的梯度

xx = x.grad#使用x.grad可以获取求得的结果,其实就是对y也就是2x^2对x求导的结果就是4x,可以进行验证


#以上，在默认情况下torch会累积每次计算的梯度，所以我们在计算下一个独立的梯度之前需要对之前的grad存储进行清零，方法是grad.zero_()
x.grad.zero_()
print(x.shape)
print(x.T)

y = x.sum()#求和，默认的axis是0
print(y)

y.backward()
print(x.grad)#tensor([1., 1., 1., 1.])]),注意这个结果需要记忆，向量的sum，梯度为全1，维度是向量的维度

print(xx == 4 * x)#输出的结果是一个和x一样维度的布尔值的张量，对应的每个元素的值都是True



