'''================================================================
*   Copyright (C) 2022 IEucd Inc. All rights reserved.
*   
*   FileName:gradientDescent.py
*   Author:weiyutao
*   CreateTime:2022-03-06
*   Describe:简单线性回归的实现
*
================================================================'''

import matplotlib.pyplot as plt
import random
import torch

def synthetic_data(w,b,num_example):

	X = torch.normal(0,1,(num_example,len(w)))
	y = torch.matmul(X,w) + b
	y += torch.normal(0,0.01,y.shape)

	return X,y.reshape((-1,1))#返回的y以向量的形式输出

true_w = torch.tensor([2,-3.4])
true_b = 4.2

features,labels = synthetic_data(true_w,true_b,1000)

#定义一个data_iter函数，接受批量的大小特征矩阵和标签向量作为输入，生成大小为batch——size的小批量
def data_iter(batch_size,features,labels):#传入我们的训练集数据，其中包含X和y也就是对应的特征值和标签
	
	num_examples = len(features)
	indices = list(range(num_examples))#生成每个样本对应的索引

	#将这些生成索引顺序打乱，然后我们顺序抽取就可以得到一个打乱的小批量索引
	random.shuffle(indices)

	for i in range(0,num_examples,batch_size):#每次跳跃batch_size个大小去抽取索引
		batch_indices = torch.tensor(indices[i:min(i + batch_size,num_examples)])
		yield features[batch_indices],labels[batch_indices]

for X,y in data_iter(10,features,labels):
	print(X,'\n',y)
	break#注意这里如果没有break那么会输出全部生成的内容，但是有了break就只输出一个就结束了


#定义初始化模型参数
w = torch.normal(0,0.01,size = (2,1),requires_grad = True)
b = torch.zeros(1,requires_grad = True)

#定义模型
def linreg(X,w,b):
	return torch.matmul(X,w) + b

#定义损失函数
def squared_loss(y_hat,y):
	return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

#定义优化算法，注意其中的参数params其实就是w和b参数的组合，我们将其组和在一块形成一个列表然后分别在sgd方法中使用grad去调用反向求导的结果，每迭代一次就清空一次grad，params中存储的两个参数分别是w和b，分别对w和b进行迭代梯度下降，这个batch_size参数其实就是随机梯度下降的所挑选的样本的个数
def sgd(params,lr,batch_size):
	with torch.no_grad():
		#小批量随机梯度下降
		for param in params:
			param -= lr * param.grad / batch_size
			param.grad.zero_()
#以上就是关于模型的所有方法的定义，下来我们进行模型的训练
lr = 0.03#定义学习率
num_epochs = 3#定义梯度下降的迭代次数
net = linreg
loss = squared_loss

batch_size = 10

for epoch in range(num_epochs):
	for X,y in data_iter(batch_size,features,labels):
		l = loss(net(X,w,b),y)
		l.sum().backward()#反向求导
		sgd([w,b],lr,batch_size)
	with torch.no_grad():
		train_l = loss(net(features,w,b),labels)
		print("epoch %d,loss %f"%(epoch + 1,float(train_l.mean())))

#比较真实参数和通国训练学到的参数来评估训练的成功程度
print("w的估计误差%s"%(true_w - w.reshape(true_w.shape)))
print("b的估计误差%s"%(true_b - b.reshape(b.shape)))
#以上就是使用一些基本的语法手动实现的简单线性回归其实也就是单层神经蛙网络的实现，我们可以调整对应的超参数来看看超参数的不同给回归结果带来的不同的影响情况，下面我们使用pytorch框架来简单实现下以上步骤

#fig,ax = plt.subplots()
#ax.scatter(features[:,1],labels,1)
#plt.show()





