'''================================================================
*   Copyright (C) 2022 IEucd Inc. All rights reserved.
*   
*   FileName:torchHello.py
*   Author:weiyutao
*   CreateTime:2022-03-02
*   Describe:
*
================================================================'''


import torch
import pandas as pd
import os





x = torch.arange(12)

X = x.reshape(3,4)

y = torch.zeros(3,3,3)

z = torch.ones(3,3,3)

n = torch.tensor([[1,2,3],[2,3,4],[3,4,5]])
#列表嵌套列表可以构造多维数组，看前面的中括号数，前面有一个的是一维数组，两个的是二维数组，三个的是三维数组

Y = torch.arange(12,dtype = torch.float32).reshape((3,4))
Z = torch.tensor([[2.0,1,4,3],[1.0,2,3,4],[0.0,1,3.0,6.0]])

A,B = torch.cat((Y,Z),dim = 0),torch.cat((Y,Z),dim = 1)

C = X == Y

os.makedirs(os.path.join('..','data'),exist_ok = True)
dataFile = os.path.join('..','data','houseTiny.csv')
with open(dataFile,'w') as f:
	f.write("NumRooms,Alley,price\n")
	f.write('NA,Pave,127500\n')
	f.write('2,NA,106000\n')
	f.write('4,NA,178100\n')
	f.write('NA,NA,140000\n')


data = pd.read_csv(dataFile)

inputs,outputs = data.iloc[:,0:2],data.iloc[:,2]

inputs = inputs.fillna(inputs.mean())

inputs = pd.get_dummies(inputs,dummy_na = True)#将缺失值作为一个特征加入，也就是01，比如一个非数字格式的字段，那么如果去掉缺失值我们就不能使用大概率赋值的情况了，比如使用平均值去代替缺失值，此时我们就可以使用dummies方法将缺失值作为一个字段，非缺失值也就是说该字段中的非缺失值也作为一个字段去进行统计，是的就是1，不是的就是0，命令方式以原来的字段名称加下划线缺失值或者是非缺失值，所以会多出来很多的字段，不过这个方法非常简便并且管用

X,y = torch.tensor(inputs.values),torch.tensor(outputs.values)

print(X,"\n",y)



