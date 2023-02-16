import numpy as np
import matplotlib.pyplot as plt


"""
# @Author: weiyutao
# @Date: 2023-02-10 14:50:54
# @Parameters: 
# @Return: 
# @Description: we will implement the function about imshow picture, scatter and 
# line used one function. we will use rewrite method. you can use cv2 or matplotlib
# in order to unified the data type, so we will use matplotlib. the image is the picture
# data you have read in memory.
"""

class Imshow:
    def __init__(self, name) -> None:
        self.name = name

    def imshow(self, image):
        plt.title(self.name)
        plt.axis('off')
        plt.imshow(image)
        plt.show()

class Scatter(Imshow):
    def __init__(self, name) -> None:
        super().__init__(name)
    
    """
    # @Author: weiyutao
    # @Date: 2023-02-10 15:06:27
    # @Parameters: X, inputData(m, n), y inputLabel(m, 1), n = 2, because this function is scatter.
    # of course, the X should be to be classified here. if the label of X has not been
    # classified, you should add some judge conditions used y. of course, this imshow is not generally used.
    # @Return: 
    # @Description: but if you want to show image in plt, you should use array datatype, because
    # sometimes the matrix data type can be error. but if you use numpy, you should transform the tuple
    # to an two dimension array. or you will get error when you operated matrix multiplication
    """
    def imshow(self, inputData, x_label_decision_boundary=['none'], y_label_decision_boundary=['none']):
        if(type(inputData).__name__ == 'ndarray'):
            X = inputData
        else:
            X = inputData.getA()
        fig, ax = plt.subplots()
        plt.scatter(X[:50, :1], X[:50, 1:], color='red', marker='o', label='setosa')
        plt.scatter(X[50:100, :1], X[50:100, 1:], color='blue', marker='x', label='versicolor')
        # plt.scatter(X[100:, :1], X[100:, 1:],color='green', marker='+', label='Virginica')
        if((x_label_decision_boundary[0] != 'none') & (y_label_decision_boundary[0] != 'none')):
            plt.plot(x_label_decision_boundary, y_label_decision_boundary, color='red', linestyle='--', label='predict value')
        ax.set_xlabel(xlabel='feature 1', fontsize=18)
        ax.set_ylabel(ylabel='feature 2', fontsize=18)
        ax.set_title(label=self.name, fontsize=18)
        plt.legend(loc = 2)
        plt.show()


class Line(Scatter):
    def __init__(self, name) -> None:
        super().__init__(name)
    

    """
    # @Author: weiyutao
    # @Date: 2023-02-10 15:06:27
    # @Parameters: X, inputData(m, n), y inputLabel(m, 1), because this function will imshow line
    # figure, so you can imshow many line, so n will be not limited. but in order to distinguish the
    # different line, we set the limit as 3. line figure can show the different informations about the 
    # same axis. just like you want to show the difference between one feature and other features for
    # all samples. so you can just pass X. the axis should be divided uniform.
    # @Return: 
    # @Description: 
    """
    def imshow(self, X):
        # define the line type
        type = [['red', 'o', 'setosa'], ['blue', 'x', 'versicolor'], ['green', '+', 'Virginica']]
        x_label = np.linspace(0, 30, 150)
        fig, ax = plt.subplots()
        for i in range(X.shape[1]):
            plt.plot(x_label, X[:, i:i+1], color=type[i]
                     [0], marker=type[i][1], label=type[i][2])
        ax.set_xlabel(xlabel='x label', fontsize=18)
        ax.set_ylabel(ylabel='feature i', fontsize=18)
        ax.set_title(label=self.name, fontsize=18)
        plt.legend(loc = 2)
        plt.show()

