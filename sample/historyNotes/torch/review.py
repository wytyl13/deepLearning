# -*- coding: utf-8 -*-
#*****************************************************************
#   Copyright (C) 2022 IEucd Inc. All rights reserved.
#   
#   @Author: weiyutao
#   @Created Time : 2022/8/8 7:36:19
#   @File Name : review.py
#   @Description :复习torch 
#
#*****************************************************************
import numpy as np
import matplotlib.pyplot as plt
import torch
import site



a = torch.arange(12)
b = a.reshape((3,4))
b[:] = 2
#print(a,"\n",b)
#print(id(a),id(b))

a = torch.tensor([3.5])
#将大小为1的张量转换为python标量
#print(float(a))


X = torch.arange(24).reshape(2,3,4)
#print(id(X))


#多维
#比如上面的维度是2，3，4  对应的2是第三维度，3是行，4是列，可以分别按照对应索引的维度进行求和
#0索引对应的是第三维度，1对应的是行维度，4对应的是列维度。sum:axis = 1  shape=2,1,4
#sum:axis = 2  shape=2,3,1   sum:axis = 1,2   shape=2,1,1   keepdims = true  的情况下是将对应的维度置为1而不是去除维度
X = X.sum([1,2],keepdims = True)
#print(X.shape)



#线性回归

#首先生成线性回归的模型，然后进行回归拟合参数，最后比对拟合的正确程度
def synthetic_data(W,b,m):
    X = torch.normal(0,1,(m,len(W)))#生成均值为0的随机数，维度是n*m
    #W的维度是nl * nl-1,在线性回归中就是1,n  len(W) = n,而X的维度m*n
    #W的维度是n*1维度为n的列向量。b的维度就是m*1，使用python的广播机制
    y = torch.matmul(X,W) + b
    #在y的基础上随机加点噪声
    y += torch.normal(0,0.01,y.shape)#注意这里使用+=可以减少内存的占用，在原始内存的基础上操作的
    return X,y.reshape((-1,1))#返回的y的维度是列为1，而行自动推导，也就是保证返回的y是一个向量


def plot_X(X):
    fig,ax = plt.subplots()
    ax.scatter(X[0],X[1],c = 'black',lw = 2)
    ax.set_xlabel('x1',fontsize=20)
    ax.set_ylabel('x2',fontsize=20)
    ax.set_title(label = 'original data',fontsize = 25)
    plt.show()


def data_iter(batch_size,features,labels):
    m = len(features)
    indices = list(range(m))#生成一个list，单位是0-999
    np.random.shuffle(indices)#将生成的list随机打乱

    for i in range(0,m,batch_size):
        batch_indices = torch.tensor(indices[i:min(i + batch_size,m)])
        yield features[batch_indices],labels[batch_indices]


#定义线性回归模型
def linreg(X,w,b):
    return torch.matmul(X,w) + b

#定义损失函数
def compute_error(y_hat,y):
    return (y_hat - y.reshape(y_hat.shape)) **2 / 2

#定义优化算法
def sgd(params,lr,batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()












if __name__ == "__main__":
    true_W = torch.tensor([2,-3.4])#定义一个张量，注意默认的都是向量，维度为2，1
    true_b = 4.2#此处使用广播机制，把true_b参数从1，1维度广播为m,1维度
    features,labels = synthetic_data(true_W,true_b,1000)#生成1000个样本
    #生成的X的维度是1000*2，y的维度是1000*1)])    
    batch_size = 10
    for X,y in data_iter(batch_size,features,labels):
        print(X.shape,'\n',y.shape)
    
    #定义初始化模型参数
    W = torch.normal(0,0.01,size = (2,1),requires_grad = True)
    b = torch.zeros(1,requires_grad = True)

    print(W,'\n',b)
    #进行模型的训练
    lr = 0.03
    iterators = 3
    net = linreg
    loss = compute_error

    batch_size = 10
    for epoch in range(iterators):
        for X,y in data_iter(batch_size,features,labels):
            l = loss(net(X,W,b),y)
            l.sum().backward()
            sgd([W,b],lr,batch_size)
        with torch.no_grad():
            train_l = loss(net(features,W,b),labels)
            print("epoch %d,loss %f"%(epoch + 1,float(train_l.mean())))














	
	
