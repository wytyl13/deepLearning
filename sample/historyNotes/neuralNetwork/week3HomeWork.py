'''================================================================
*   Copyright (C) 2022 IEucd Inc. All rights reserved.
*   
*   FileName:week3HomeWork.py
*   Author:weiyutao
*   CreateTime:2022-03-07
*   Describe:第三周编程作业，构建具有但隐藏层的2类分类神经网络
*
================================================================'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary,sigmoid,load_planar_dataset,load_extra_datasets


#%matplotlib inline

np.random.seed(1)#设置一个固定的随机种子


X,Y = load_planar_dataset()

fig,ax = plt.subplots()
ax.scatter(X[0,:],X[1,:],c = Y,s = 40,cmap = plt.cm.Spectral)
#plt.show()

#首先查看简单的逻辑回归的分类效果，我们使用sklearn的内置函数来进行估计
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T,Y.T)

plot_decision_boundary(lambda x: clf.predict(x),X,Y)#注意Z = model.predict(np.c_[xx.flatten(),yy.flatten()]),我们这里使用了lambda构造简单的函数

plt.title("Logistic Regression")
predictionAccuracy = clf.predict(X.T)#预测结果

#下面打印下预测准确率
print("逻辑回归的准确性：",np.mean(predictionAccuracy == Y)*100,"%")
#返回的准确率为47%，可以发现这类问题使用普通的逻辑回归是不可取的，这是因为我们所要处理的数据不是线性可分的，所以此时使用逻辑回归是不可取的

#下面我们来搭建神经网络
#首先定义神经网络的结构
def layer_sizes(X,Y):

	n_x = X.shape[0]#输入层的数量，其实也就是特征数量
	n_h = 4#因为我们设计的隐藏层的数量是4，所以我们硬编码隐藏层数量为4
	n_y = Y.shape[0]#输出层的数量

	return n_x,n_h,n_y

#初始化模型的参数
def initialize_parameters(n_x,n_h,n_y):
	#已知三个维度，输入层和隐藏层和输出层的个数
	#需要初始化输出的参数是w1,b1,w2,b2分别对应的维度如下：
	#w1(n_h,n_x)  b1(n_h,1)  w2(n_y,n_h)  b2(n_y,1)

	np.random.seed(2)#指定一个随机种子，以便我们的输出和教材输出一致
	W1 = np.random.randn(n_h,n_x) * 0.01#注意还需要在随机数的基础上进一步缩放w参数
	b1 = np.zeros((n_h,1))
	W2 = np.random.randn(n_y,n_h) * 0.01
	b2 = np.zeros((n_y,1))

	parameters = {
	
		"W1":W1,
		"b1":b1,
		"W2":W2,
		"b2":b2
	
	}

	return parameters

#定义前向传播计算各个神经元的参数的值
def forward_propagation(X,parameters):

	#获取前向传播需要的我们之前初始化的各个神经层的参数	
	W1,b1,W2,b2 = parameters['W1'],parameters['b1'],parameters['W2'],parameters['b2']

	#前向传播计算z和a
	Z1 = np.dot(W1,X) + b1
	A1 = np.tanh(Z1)
	Z2 = np.dot(W2,A1) + b2
	A2 = sigmoid(Z2)
	#保存我们的前向传播的计算结果，后面反向传播的时候需要使用，注意torch中默认会保存计算结果，所以我们如果不想保存计算结果的话需要声明在torch中
	cache = {
	
		"Z1":Z1,
		"A1":A1,
		"Z2":Z2,
		"A2":A2
	
	}
	
	return A2,cache


#计算损失函数和梯度下降方法，代价函数的计算是为了计算损失值，正向传播是计算初始化参数w和b得到的损失值，而反向传播中则利用了梯度下降法进行参数的优化，就是因为存在隐藏层，所以此处使用的参数优化是在反向传播中进行的，反之如果不存在隐藏层的话我们会直接在正向传播中进行参数的优化，而不需要进行反向传播
def costFunctionVsGradient(X,Y,parameters,A2):
	m = X.shape[1]
	#J = -1 / m * np.sum(np.multiply(Y , np.log(A2)) + np.multiply((1 - Y) , np.log(1 - A2)),axis = 1)
	#注意Y和A的维度的定义，首先如果同维度，可以点乘也可以矩阵的相乘，注意sum的维度
	J = -1 / m * np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2))
	#注意上面的定义方法较麻烦，所以我们使用下面简单的办法，所以般情况下只要可以手动定义的一定不要使用第三方库的，因为第三方库的格式化和书写都很规整，而使用python原生的书写更加高效和简洁一点，但是python的语法可能没有那么清晰明了，因为出现一个结果并不是我们可以分析出来的
	#以上是计算成本函数，也就是计算正向传播的损失值，下面需要使用梯度下降法去估计w和b参数


	return J

#定义反向传播
def backward_propagation(parameters,cache,X,Y):
	m = X.shape[1]

	W1 = parameters['W1']
	W2 = parameters['W2']

	A1 = cache['A1']
	A2 = cache['A2']

	dZ2 = A2 - Y
	dW2 = (1 / m) * (dZ2 @ A1.T)
	db2 = (1 / m) * np.sum(dZ2,axis = 1,keepdims = True)
	dZ1 = (W2.T @ dZ2) * (1 - np.power(A1,2))
	dW1 = (1 / m) * (dZ1 @ X.T)
	db1 = (1 / m) * np.sum(dZ1,axis = 1,keepdims = True)

	grads = {
		
		"dW1":dW1,
		"db1":db1,
		"dW2":dW2,
		"db2":db2
	
	}

	return grads


#以上分别定义了前向传播和反向传播，下面需要使用梯度下降法进行估计参数
def update_parameters(parameters,grads,learning_rate = 1.2):
	
	W1,W2 = parameters['W1'],parameters['W2']
	b1,b2 = parameters['b1'],parameters['b2']

	dW1,dW2 = grads['dW1'],grads['dW2']
	db1,db2 = grads['db1'],grads['db2']	

	W1 = W1 - learning_rate * dW1
	b1 = b1 - learning_rate * db1
	W2 = W2 - learning_rate * dW2
	b2 = b2 - learning_rate * db2

	parameters = {
	
		"W1":W1,
		"b1":b1,
		"W2":W2,
		"b2":b2
	
	}
	
	return parameters

#现在把以上的方法整合到模型中，实现参数的迭代估计
def nn_model(X,Y,n_h,num_iterations,isPrint_cost = False):
	np.random.seed(3)
	n_x = layer_sizes(X,Y)[0]
	n_y = layer_sizes(X,Y)[2]

	parameters = initialize_parameters(n_x,n_h,n_y)
	W1 = parameters['W1']
	b1 = parameters['b1']	
	W2 = parameters['W2']
	b2 = parameters['b2']
	
	for i in range(num_iterations):
		A2,cache = forward_propagation(X,parameters)
		cost = costFunctionVsGradient(X,Y,parameters,A2)
		grads = backward_propagation(parameters,cache,X,Y)
		parameters = update_parameters(parameters,grads)

		if isPrint_cost:
			if i % 1000 == 0:
				print("第",i,"次循环，成本为：",cost)
	return parameters

#下面定义预测方法,需要传入的参数是上一步计算的最终的parameters和样本数据X
def predict(X,parameters):

	A2,cache = forward_propagation(X,parameters)
	predictions = (A2 > 0.5)

	return predictions





if __name__ == "__main__":

	n_x,n_h,n_y = layer_sizes(X,Y)
	
	#调整超参数n_h也就是调整隐藏层的个数，看对成本值和预测准确率的影响
	hidden_layer_sizes = [1,2,3,4,5,20,50]

	for i in hidden_layer_sizes:
		parameters = nn_model(X,Y,n_h = i,num_iterations = 10000,isPrint_cost = True)
		predictions = predict(X,parameters)
		print(predictions.mean())
		


