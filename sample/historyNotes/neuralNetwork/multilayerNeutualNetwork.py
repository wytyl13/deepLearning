'''================================================================
*   Copyright (C) 2022 IEucd Inc. All rights reserved.
*   
*   FileName:multilayerNeutualNetwork.py
*   Author:weiyutao
*   CreateTime:2022-03-11
*   Describe:以构建单层神经网络为基础向多层次神经网络拓展
*
================================================================'''

import numpy as np
import h5py
import matplotlib.pyplot as plt
import testCases
from dnn_utils import sigmoid,sigmoid_backward,relu,relu_backward
import lr_utils

np.random.seed(1)

def initializeParameters(n_x,n_h,n_y):
	
	W1 = np.random.randn(n_h,n_x) * 0.01
	b1 = np.zeros((n_h,1)) 
	W2 = np.random.randn(n_y,n_h) * 0.01
	b2 = np.zeros((n_y,1))

	parameters = {
	
		"W1" : W1,
		"b1" : b1,
		"W2" : W2,
		"b2" : b2
	
	}
	
	return parameters

#以上是我们构建的对于两层神经网络的初始化参数的函数，那么如何构建多层的呢？
def multilayerInitializeParameters(layerDims):#layerDims是神经网络每个图层的节点数量的列表
	np.random.seed(3)
	parameters = {}
	L = len(layerDims)

	for l in range(1,L):#注意是从1到L，不包含0，这是因为第0层是输入层，由训练集数据决定，不是我们定义神经网络层级的范畴
		parameters["W" + str(l)] = np.random.randn(layerDims[l],layerDims[l - 1]) / np.sqrt(layerDims[l - 1])
		parameters["b" + str(l)] = np.zeros((layerDims[l],1))
		#注意在使用字典的键值的时候传入的是字符串，然而拼接的时候需要将字符串和字符串进行拼接，所以需要将int类型的变量强制转换为字符串变量然后拼接，注意randon里面传入的参数不能带括号，只能传入两个int类型的数据，而zeros里面需要传入的维度需要带上括号，注意这个细节	

	return parameters
		
#前向传播函数
def linearForward(A,W,b):
	Z = W @ A + b
	#其中W指的是本层神经网络的参数，A是上一层的激活值b是本层神经网络的参数,当然在第一层神经网络的计算中，A对应的就是输入样本的特征值X
	cache = {
	
		"W" : W,
		"A" : A,
		"b" : b
	
	}

	return Z,cache
#定义完整的前向传播，包括求Z和A，Z的求解就是假设函数的求解，A的求解就是加入激活函数,在多层神经网络中我们使用的激活函数包括RELU和sigmoid函数，前者主要用在隐藏层的激活函数中，而后者作为输出层的激活函数中，前者的表达式为：RELU = MAX(0,Z)，后者的表达式为SIGMOID = 1 / (1 + np.exp(-Z))
def linearVsActivationForward(APre,W,b,activation):
	if activation == "sigmoid":
		Z,linearCache = linearForward(APre,W,b)
		A,activationCache = sigmoid(Z)
	elif activation == "relu":
		Z,linearCache = linearForward(APre,W,b)
		A,activationCache = relu(Z)
	#注意以上返回的activationCache其实就是Z

	cache = {
	
		"linearCache" : linearCache,
		"activationCache" : activationCache

	}

	return A,cache

#定义前向传播的模型，也就是将上述方法整合在一块前向计算到最后的A
def multilayerModelForward(X,parameters):
	#前向传播所需要的参数就是输入样本和对应的估计参数，注意这里面的细节
	#有多少层隐藏层就需要计算多少个cache，cache中保存的每一层计算到的A,W,b和Z
	caches = []#使用列表存储cache,每次存储的cache是一个字典
	reluA = X#第一次前向传播传入的参数就是X样本数据，然后后面的每次前向传播传入的参数都是上一步计算到的激活函数的值A，这个在下面的循环迭代中有体现
	L = len(parameters) // 2#parameters是我们初始化的参数，类型是一个字典，对应的字典的长度整除以2就是我们定义的神经网络的层数，因为每层对应的两个参数一个w一个b,注意得到的是真正的神经网络的层数，而对应的如果使用range(L)就会有L个数值，但是我们的神经网络没有第0层的，因为第0层其实就是输入层，而我们在计算迭代的时候是直接从第一层也就是隐藏层开始的，所以我们使用1-L，那么就会从1遍历到L - 1层，这正符合我们的要求，因为我们对第1到L -1 层的激活函数使用的是relu，而对最后一层也就是第L层的激活函数使用的是sigmoid函数，所以这个正好符合我们的循环遍历，先对前1到L-1层进行for循环，然后对第L层单独进行计算，注意每层计算到的cache和A激活函数值都需要缓存下来以便于后期的反向传播使用，而对应cache中有w，b参数
	for l in range(1,L):
		APre = reluA
		reluA,reluCache = linearVsActivationForward(APre,parameters['W' + str(l)],parameters['b' + str(l)],activation = "relu")
		caches.append(reluCache)
	
	sigmoidA,sigmoidCache = linearVsActivationForward(reluA,parameters['W' + str(L)],parameters['b' + str(L)],activation = "sigmoid")

	#注意以上for循环中和for循环外计算第L层所使用的参数APre和reluA的区别，for循环中使用的参数A都是前一个计算出来的A也称为APre，而对应的第一层计算所需要的A也就是APre也就是X，所以我们在刚开始定义了APre = A，而对应的for循环的刚开始我们需要将定义好的A其实也就是reluA赋值给APre，其实这个APre使用什么名称都可以，使用A也可以，只要做到前后一致即可，注意这块的循环中技巧，要不很容易出错
	caches.append(sigmoidCache)

	return sigmoidA,caches#注意sigmoidA就是计算的输出层的A值，其实也就是最后的值

#前向传播定义完成以后我们会利用其得到最终的A，然后要使得估计的A去逼近真实的Y，在这个过程中就可以达到参数最优，那么就需要定义每一个估计值A和对应的Y的损失值，所有的样本加在一块就是成本值，所以下一步我们需要定义成本函数
def computeCost(sigmoidA,Y):
	m = Y.shape[1]
	cost = -1 / m * np.sum(Y * np.log(sigmoidA) + (1 - Y) * np.log(1 - sigmoidA))
	#注意如果两个数组的维度是相同的，那么相乘的目的其实就是让他们的各个元素进行点乘，那么我们不要使用@进行矩阵的相乘，直接使用*表示点乘即可
	return cost

#接下来是反向传播方法的定义：
def linearBackward(dZ,cache):#注意dZ是对激活函数包括sigmoid和relu函数的求导
	W,APre,b = cache['W'],cache['A'],cache['b']#这里的是cache是前向传播中计算的

	m = APre.shape[1]
	dW = dZ @ APre.T / m
	db = np.sum(dZ,axis = 1,keepdims = True) / m#注意这块的keepdims = True这个参数很重要，这个参数可以保持原来对维度不变，不论全面的求和方向是什么，保持和求和之前的维度一样，这块如果没有这个参数的话后面的反向传播中db的维度就会出现错误
	dAPre = W.T @ dZ#relu(Z) = max(0,Z)

	return dW,dAPre,db

#工具包中已经定义了激活函数的倒数的计算方法，总共两个激活函数对应的有两个激活函数的倒数，对应传递的参数是dA和Z值，返回的是dZ

#下来我们直接正式开始定义反向线性激活
def linearVsActicationBackward(dA,cache,activation = "relu"):
	linearCache,activationCache = cache['linearCache'],cache['activationCache']
	if activation == "relu":
		dZ = relu_backward(dA,activationCache)
		dW,dAPre,db = linearBackward(dZ,linearCache)
	elif activation == "sigmoid":
		dZ = sigmoid_backward(dA,activationCache)
		dW,dAPre,db = linearBackward(dZ,linearCache)
	return dW,dAPre,db

#以上我们定义了所有的倒数，但是维度没有计算输出层的sigmoidA的倒数
def getdSigmoidA(sigmoidA,Y):
	dSigmoidA = - (Y / sigmoidA) - ((1 - Y) / (1 - sigmoidA))
	return dSigmoidA

#以上是关于后向传播的各种方法，下面我们使用以上函数去定义构建多层模型的后向传播,注意这里的参数dA，在反向传播的时候，可以传入dSigmoidA,也可以传入dReluA,这个是通用的方法，在后面的if判断中可以区分，因为我们的后向激活函数的传播需要有两个激活函数，所以这里的参数是可以传入任何一种激活函数的.而参数caches是在前向传播的时候保存的计算结果，包括W,B,A,Z，当时我们的保存是分类进行保存的，W,b,A是放在caches中的linearCache,而Z是放在activationCache中，注意总共有L层神经网络，对应的也就有L个cache，也就是说caches的长度应该是L，这个我们在之前定义前向传播模型的时候已经定义好了。

def multilayerModelBackward(sigmoidA,Y,caches):
	grads = {}
	L = len(caches)
	m = sigmoidA.shape[1]
	Y = Y.reshape(sigmoidA.shape)
	dSigmoidA = getdSigmoidA(sigmoidA,Y)
	

	#反向传播最后一层也就是输出层，需要输入的参数是dSigmoidA,sigmoidCache,而sigmoidCache是caches中的最后一个，这里需要注意一下，因为caches的长度是L，而对应的如果我们想要根据列表索引去索引对应的值的话，因为索引是从0开始的，而我们有L个值，所以对应的索引应当是从0-(L-1),所以这里最后一个cache也就是sigmoidCache也就是输出层的cache就是caches[L - 1]
	currentCache = caches[L - 1]
	grads["dW" + str(L)],grads["dA" + str(L)],grads["db" + str(L)] = linearVsActicationBackward(dSigmoidA,currentCache,activation = "sigmoid")
	#注意这里使用caches中的最后一个索引位置的数据也就是第L-1个位置的数据求得的是第L层的结论，所以我们这里命名为L

	#上面我们对输出层进行了反向传播，下来我们对隐藏层进行反向传播
	for l in reversed(range(L - 1)):
		#从后往前遍历，从L-2一直遍历到0
		currentCache = caches[l]
		grads["dW" + str(l + 1)],grads["dA" + str(l + 1)],grads["db" + str(l + 1)] = linearVsActicationBackward(grads["dA" + str(l + 2)],currentCache,activation = "relu")
		#注意以上的细节，在存入grads中的时候按照第几层作为字典中的键值进行存储，而在从caches中取对应层数的数据进行反向传播的时候是按照列表对应的索引也就是从0开始的索引开始，所以这里就会有一个索引位置的偏差，比如一共有L层索引，那么对应的caches就是有L-1的长度，因为列表是从0开始的。而对应的最后一个输出层对应的列表索引就是L-1，而进行反向传播计算后返回的值进行存储命名的时候是按照真实的层数来命名的，所以在命名的时候需要加上1.而我们在进行反向传播的时候，特别是在对隐藏层进行反向传播的时候，我们每次传递的参数都需要使用前一步骤计算存入grads中的结果，而这个结果我们是按照实际层数作为键值进行存储的，所以在取数据的时候一定要注意键值名称和循环变量之间的关系，这块一定要搞清楚。

	return grads


#前面已经定义好了正向和反向传播中使用的所有方法，并且将所有的方法整合成了反向和正向传播两个模型，下面我们来定义更新参数的方法
def updateParameters(parameters,grads,learningRate):

	L = len(parameters) // 2
	for l in range(L):
		parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learningRate * grads["dW" + str(l + 1)]
		parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learningRate * grads["db" + str(l + 1)]
		
	return parameters


#建立两层神经网络
def twoLayerModel(X,Y,layerDims,learningRate = 0.0075,numIterations = 3000,printCost = False,isPlot = True):
	
	np.random.seed(1)
	grads = {}
	costs = []
	n_x,n_h,n_y = layerDims
	
	#初始化参数
	parameters = initializeParameters(n_x,n_h,n_y)

	W1 = parameters["W1"]
	b1 = parameters["b1"]
	W2 = parameters["W2"]
	b2 = parameters["b2"]

	#开始迭代
	for i in range(0,numIterations):
		#前向传播
		reluA,cache1 = linearVsActivationForward(X,W1,b1,activation = "relu")
		sigmoidA,cache2 = linearVsActivationForward(reluA,W2,b2,activation = "sigmoid")
		
		#损失函数
		cost = computeCost(sigmoidA,Y)

		#后向传播
		dSigmoidA = getdSigmoidA(sigmoidA,Y)		
		#开始后向传播隐藏层，倒数第一层是sigmoid函数，其余的都是relu函数,注意后向传播的时候输入的是下一层的A，然后输出的是上一层的A，输出的是本层的w和b

		dW2,dA1,db2 = linearVsActicationBackward(dSigmoidA,cache2,activation = "sigmoid")
		dW1,dA0,db1 = linearVsActicationBackward(reluA,cache1,activation = "relu")
		
		#保存后向传播的结果
		grads["dW1"] = dW1
		grads["db1"] = db1
		grads["dW2"] = dW2
		grads["db2"] = db2
		
		#更新参数
		parameters = updateParameters(parameters,grads,learningRate)
		W1 = parameters["W1"]
		b1 = parameters["b1"]
		W2 = parameters["W2"]
		b2 = parameters["b2"]
		#打印成本值
		if i % 100 == 0:
			costs.append(cost)
			if printCost:
				print("第" , i , "次迭代，成本值为：" , np.squeeze(cost))
		
	#可视化成本函数的迭代过程
	if isPlot:
		fig,ax = plt.subplots()
		ax.plot(np.squeeze(costs),label = "cost")
		ax.set_xlabel("iter",fontsize = 15)
		ax.set_ylabel("cost",fontsize = 15)
		ax.set_title("costs",fontsize = 20)
		plt.show()


	return parameters


#------------------建立测试数据的方法------------------------------------------------------------------------------------
def getTestModelBackwardData():
	np.random.seed(3)
	sigmoidA = np.random.randn(1,2)
	Y = np.array([[1,0]])

	W1 = np.random.randn(3,4)
	A1 = np.random.randn(4,2)
	b1 = np.random.randn(3,1)
	Z1 = np.random.randn(3,2)
	linearCache = ((W1,A1,b1),Z1)

	W2 = np.random.randn(1,3)
	A2 = np.random.randn(3,2)
	b2 = np.random.randn(1,1)
	Z2 = np.random.randn(1,2)
	activationCache = ((W2,A2,b2),Z2)

	caches = (linearCache,activationCache)

	return sigmoidA,Y,caches





if __name__ == "__main__":
	'''
	parameters,grads = testCases.update_parameters_test_case()
	parameters = updateParameters(parameters,grads,learningRate = 0.1)

	print("W1 = " + str(parameters["W1"]))
	print("b1 = " + str(parameters["b1"]))
	print("W2 = " + str(parameters["W2"]))
	print("b2 = " + str(parameters["b2"]))
	'''

	#测试前向和损失函数
	'''
	np.random.seed(1)
	X = np.random.rand(4,2)
	W1 = np.random.randn(3,4)
	b1 = np.zeros((3,1))
	W2 = np.random.randn(1,3)
	b2 = np.zeros((1,1))
	parameters = {
	
		"W1" : W1,
		"b1" : b1,
		"W2" : W2,
		"b2" : b2
	
	}

	sigmoidA,caches = multilayerModelForward(X,parameters)

	print("sigmoidA:" + str(sigmoidA))
	print("caches的长度为：" + str(len(caches)))

	Y = np.array([[1,1,1]])#定义一个二维数组，注意如何看是几维的，看最前面的方括号的数量，最前面有几个方括号数据就是几维的，数据维度的压缩方法在numpy中是np.squeeze,该方法可以将维度是1的挤压掉，比如A.shape = (1,2,3) np.squeeze(A).shape = (2,3),但是该方法对数据维度不是一维的没有用，而且使用该方法的时候还可以加入axis参数，axis后面的参数设置的对应的就是数据维度的索引，比如axis = 0,那么对应的就是将维度从左开始第一个为1的维度压缩掉。而对应的12，对应的分别是第二个和第三个，而对应的如果维度不是1，那么就不起作用，注意这个用法；
	sigmoidA = np.array([[0.8,0.9,0.4]])

	cost = computeCost(sigmoidA,Y)
	print("cost = " + str(cost))
	'''
	
	'''
	#测试反向传播激活函数的求解
	sigmoidA,linearActivationCache = testCases.linear_activation_backward_test_case()
	dW,dAPre,db = linearVsActicationBackward(sigmoidA,linearActivationCache,activation = "sigmoid")
	print("sigmoid:")
	print("dAPre:" + str(dAPre))
	print("dW:" + str(dW))
	print("db:" + str(db))
	
	dW,dAPre,db = linearVsActicationBackward(sigmoidA,linearActivationCache,activation = "relu")
	print("relu:")
	print("dAPre:" + str(dAPre))
	print("dW:" + str(dW))
	print("db:" + str(db))
	'''

	'''
	#测试向后传播模型
	sigmoidA,Y,caches = getTestModelBackwardData()
	grads = multilayerModelBackward(sigmoidA,Y,caches)

	print("dW1" + str(grads["dW1"]))
	print("dA1" + str(grads["dA1"]))
	print("db1" + str(grads["db1"]))
	'''



	#开始读取数据集
	pathTrain = "D:/网盘下载/deepLearning.ai.programingData/深度学习编程作业数据/datasets/train_catvnoncat.h5"
	pathTest = "D:/网盘下载/deepLearning.ai.programingData/深度学习编程作业数据/datasets/test_catvnoncat.h5"

	trainData = h5py.File(pathTrain,'r')
	testData = h5py.File(pathTest,'r')

	trainDataOrg = np.array(trainData["train_set_x"][:])#取value值,并且转换为数组格式
	trainLabelsOrg = np.array(trainData["train_set_y"][:])

	testDataOrg = np.array(testData["test_set_x"][:])
	testLabelsOrg = np.array(testData["test_set_y"][:])

	#转换y的维度
	trainLabelsOrg = trainLabelsOrg.reshape(1,trainLabelsOrg.shape[0])
	testLabelsOrg = testLabelsOrg.reshape(1,testLabelsOrg.shape[0])

	#操作数据维度的变化，样本是四维数据，第一个维度是样本的个数编号，第二三个维度是每一个样本对应的数据集的横纵个数维度，最后一个维度是共有几个这样的横纵坐标。也就是说一副图片对应的是RGB三个板块，每一个板块对应的维度都是64*64，而总共有209个样本，所以对应的维度是(209,64,64,3)
	mTrain = trainDataOrg.shape[0]
	mTest = testDataOrg.shape[0]
	
	#我们对x额操作是将其转换为(n,m)维度的数组，分别对训练集和测试集进行数据维度的转换
	trainDataTrans = trainDataOrg.reshape(mTrain,-1).T#先将数据按照纵轴的维度的拉直，也就是说将64*64*3拉直，对应的拉直后的长度是12288，对应的维度是m,12288也就是m,n;然后对其进行转置后的为n,m
	testDataTrans = testDataOrg.reshape(mTest,-1).T#同上

	trainDataSta = trainDataTrans / 255
	testDataSta = testDataTrans / 255#因为数据的取值范围是从0-255，所以除以255可以将数据进行缩小标准化，也就是可以将数据标准化到0-1之间，可以提升算法的执行效率，去均值化也可以，去均值化的目的是将数据集中的数据的均值为0，而我们这里使用的方法是将数据集中的数据取值范围定义为0-1

	n_x = 12288
	n_h = 7
	n_y = 1
	layerDims = (n_x,n_h,n_y) 


	parameters = twoLayerModel(trainDataSta,trainLabelsOrg,learningRate = 0.0025,layerDims = layerDims,numIterations = 2500,printCost = True,isPlot = True)




	





