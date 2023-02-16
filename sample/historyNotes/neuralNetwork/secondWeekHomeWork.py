'''================================================================
*   Copyright (C) 2022 IEucd Inc. All rights reserved.
*   
*   FileName:secondWeekHomeWork.py
*   Author:weiyutao
*   CreateTime:2022-02-26
*   Describe:深度学习第二周的编程作业，使用带有神经网络的逻辑回归进行图片识别
*
================================================================'''

import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
#可以实时显示图片，不用再使用plt.show()了
import h5py
import pandas as pd
import scipy.io as sio
from skimage import transform
pathTrain = "D:/网盘下载/deepLearning.ai.programingData/深度学习编程作业数据/datasets/train_catvnoncat.h5"

pathTest = "D:/网盘下载/deepLearning.ai.programingData/深度学习编程作业数据/datasets/test_catvnoncat.h5"

trainData = h5py.File(pathTrain,'r')

testData = h5py.File(pathTest,'r')

trainDataOrg = trainData['train_set_x'][:]

trainLabelsOrg = trainData['train_set_y'][:]

testDataOrg = testData['test_set_x'][:]

testLabelsOrg = testData['test_set_y'][:]

#注意这块的trainDataOrg直接取属于dataSet数据类型，那么dataSet数据类型的数据不能使用reshape方法，所以我们这块需要取出对应的数据，使用[:]取出数据,其实这个方法可以推广，对于特殊类型的高等数据类型，我们可以使用中括号取出对应的数值，然后就可以操作reshape方法了,这就是为什么上步中的所有赋值操作后面都有中括号了

#plt.imshow(trainDataOrg[176])

#plt.show()

#下面我们来操作数据维度的转换
mTrain = trainDataOrg.shape[0]#其实就是样本数量m
mTest = testDataOrg.shape[0]#其实就是测试集的样本数量m

#我们对X的操作是将其转换为(n,m)维度的数组，分别对训练集和测试集进行数据维度的转换
trainDataTrans = trainDataOrg.reshape(mTrain,-1).T#首先将四维数据转换为二维，然后再进行转置，最后三个维度的数据就是一张图片，将其转换为1维数据
testDataTrans = testDataOrg.reshape(mTest,-1).T#方法和上步一样

#下面我们对label也就是标签数据进行数据维度的转换，标签维度的数据维度是(ni,m),ni对应的是当前层神经元的个数，而m是样本的个数,注意这块的数学对各个变量维度的推导；只要数学分析正确了，代码的书写很简单而且很灵活，注意结合有道笔记上的数学推导进行一起学习
trainLabelsTrans = trainLabelsOrg.reshape(1,mTrain)
testLabelsTrans = testLabelsOrg.reshape(1,mTest)


#标准化数据，为什么要标准化呢？因为数据跨度很大，也就是说很离散，我们作标准化的目的是将训练数据标准化到相近的范围内，可以提升算法的运行效率，标准化方法有很多，比如：去均值化或者对原始数据中的每一个元素除以数据的最大跨度，前者可以使得原始数据的均值为0，而后者可以将原始数据限定在0-1的范围内,将原始data数据进行标准化，也就是对应的训练数据集和测试数据集的特征值
trainDataSta = trainDataTrans / 255
testDataSta = testDataTrans / 255

#定义sigmod函数
def sigmoid(z):
	a = 1 / (1 + np.exp(-z))
	return a

#初始化参数
n_dim = trainDataSta.shape[0]#其实就是特征值的个数，也就是输入x的个数，也就是12288
w = np.zeros((n_dim,1))#注意小w的维度是(n,1),因为W是由小w.T组成的数组
b = 0#注意，因为传播机制，所以b=0即可

#定义前向传播函数，代价函数以及梯度下降
def propagate(w,b,X,y):
	
	#前向传播
	Z = np.dot(w.T,X) + b#注意因为本案例只涉及到一个隐藏层并且对应一个神经单元，所以w就是1个
	A = sigmoid(Z)

	#2 代价函数
	m = X.shape[1]
	J = (-1 / m) * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A))
	#注意这里的A和y的维度是一样的，所以我们不需要采用矩阵的乘法，直接点乘即可
	#并且需要注意的是sum是默认是按照列进行求和的，所以如果是按照行进行求和的话需要制定axis参数

	#3 梯度下降
	dw = (1 / m) * np.dot(X,(A - y).T)
	db = (1 / m) * np.sum(A - y)
	
	grands = {
		"dw" : dw,
		"db" : db
	}

	return grands,J


#优化部分
def optimize(w,b,X,y,alpha,iters,isPrint = False):
	costs = []
	for i in range(iters):
	
		grands,J = propagate(w,b,X,y)
		dw = grands['dw']
		db = grands['db']

		w = w - alpha * dw
		b = b - alpha * db
		
		if i % 100 == 0:
			costs.append(J)
			if(isPrint):
				print("iters is",i,"cost is",J)
	grands = {"dw":dw,"db":db}
	params = {"w":w,"b":b}

	return grands,params,costs

#定义预测函数，就是使用上步骤中优化好的参数w和b带到新的测试集中进行预测
def predict(w,b,X_test):
	
	Z =	w.T @ X_test + b
	A = sigmoid(Z)

	m = X_test.shape[1]
	y_pred = np.zeros((1,m))

	for i in range(m):
		
		if(A[:,i] > 0.5):
			y_pred[:,i] = 1
		else:
			y_pred[:,i] = 0

	return y_pred

#模型的整合
def model(w,b,X_train,y_train,X_test,y_test,alpha,iters):

	grands,params,costs = optimize(w,b,X_train,y_train,alpha,iters)
	w = params['w']
	b = params['b']

	y_pred_train = predict(w,b,X_train)
	y_pred_test = predict(w,b,X_test)

	print("the train acc is",np.mean(y_pred_train == y_train) * 100,'%')
	print("the test acc is",np.mean(y_pred_test == y_test) * 100,'%')

	data = {
		"w" : w,
		"b" : b,
		"y_pred_train":y_pred_train,
		"y_pred_test":y_pred_test,
		"alpha":alpha,
		"costs":costs
	}

	return data



if __name__ == "__main__":
	data = model(w,b,trainDataSta,trainLabelsTrans,testDataSta,testLabelsTrans,alpha = 0.005,iters = 2000)
	#注意这块的alpha是0.05的话会出现问题，所以我们设置的alpha尽量小一点
	#_,_,_ = optimize(w,b,trainDataSta,trainLabelsTrans,alpha = 0.005,iters = 2000)
	
	#以上已经测试除了准确率我们再随便照一张图片看看准确率
	index = 20
	pred = data['y_pred_test'][:,index]
	real = testLabelsOrg[index]

	print(pred == real)

	plt.imshow(testDataOrg[index,:])
	plt.show()

	plt.plot(data['costs'])
	plt.xlabel("per hundred iters")
	plt.ylabel("cost")
	plt.show()
	
	#不同的alpha的比较
	alphas = [0.01,0.001,0.0001]

	for i in alphas:
		data = model(w,b,trainDataSta,trainLabelsTrans,testDataSta,testLabelsTrans,alpha = i,iters = 2000)
		plt.plot(data['costs'],label = "alpha = %f"%(i))
	plt.xlabel("per hundred iters")
	plt.ylabel("cost")
	plt.legend()
	plt.show()

	#新图片的预测
	fname = "c:/users/80521/desktop/猫.jfif"
	image = plt.imread(fname)
	plt.imshow(image)
	plt.show()
	
	image_trans = transform.resize(image,(64,64,3)).reshape(64*64*3,1)
	y = predict(data['w'],data['b'],image_trans)

	print(y)
