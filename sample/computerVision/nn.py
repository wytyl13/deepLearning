import numpy as np
import matplotlib.pyplot as plt
from data_utils import load_CIFAR10

class NearestNeighbor(object):
    def __init__(self) -> None:
        pass
    

    '''
     * @Author: weiyutao
     * @Date: 2023-05-06 11:48:23
     * @Parameters: the dimension X is m*n, each row is one sample.
     the dimension of y is 1*m. n is the feature numbers of each sample.
     * @Return: 
     * @Description: 
     '''
    def train(self, X, y):
        self.Xtr = X
        self.ytr = y
    
    '''
     * @Author: weiyutao
     * @Date: 2023-05-06 11:51:37
     * @Parameters: X is the test_data. the dimension of X is 
     similar to the former function X. it is m*n, m is the testdata sample 
     numbers. n is the feature numbers for each sample. y is the label of test data.
     * @Return: 
     * @Description: 
     '''
    def predict(self, X, y):
        
        num_test = X.shape[0]
        yPred = np.zeros(num_test, dtype = self.ytr.dtype)
        # loop over all test rows
        # calculate the L1 or L2 distance about the each test sample
        # with each train sample. so the amount of calculation will
        # be m*n if you have m samples in train_data and n samples
        # in test_data.
        for i in range(num_test):
            distances = np.sum(np.abs(self.Xtr - X[i, :]), axis = 1)
            # get the index of the min distance in the distance list.
            min_index = np.argmin(distances)
            # get the lable for the index that in self.ytr what we have 
            # and set the labels we have got inside of the yPred variable.
            yPred[i] = self.ytr[min_index]
            print(yPred[i], y[i])
        return yPred

    # notice, you just need to pass two parameters, and you should add the self param
    # if you want to define the showImage function in one class. or you will pass
    # the error number parameters.
    def showImage(self, X_train, y_train):
        classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        num_classes = len(classes)
        samples_per_class = 7
        for y, cls in enumerate(classes):
            # get a 1 dimension array stored the indicies of y_train
            # that y_train != y.
            # np.flatnonzero(arr): return the indicies of non zero element in arr.
            # np.flatnonzero(arr == 1): return the indicies that arr == 1.
            # notice the condition is not the arr is equal to zero.
            idxs = np.flatnonzero(y_train == y)
            # random generate the 1 dimension array from idxs and the numbers
            # is smaples_per_class. the number means the numbers of image what 
            # you want to show in one column. the image numbers what you want to show in one
            # row is eqaul to the len of classes.
            # show the title in the first row.
            idxs = np.random.choice(idxs, samples_per_class, replace = False)
            for i, idx in enumerate(idxs):
                plt_idx = i * num_classes + y + 1
                plt.subplot(samples_per_class, num_classes, plt_idx)
                plt.imshow(X_train[idx].astype('uint8'))
                plt.axis('off')
                if i == 0:
                    plt.title(cls)
        plt.show()


    def subSample(self, num_train, num_test, X_train, X_test, y_train, y_test):
        mask = list(range(num_train))
        X_train = X_train[mask]
        y_train = y_train[mask]

        mask = list(range(num_test))
        X_test = X_test[mask]
        y_test = y_test[mask]

        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

        return X_train, X_test, y_train, y_test



if __name__ == "__main__":
    cifar10_dir = 'data/cifar-10-batches-py'
    try:
        del X_train, y_train
        del X_test, y_test
        print('Clear previously loaded data.')
    except:
        pass

    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    """ 
    you can use reshape, ravel, flatten to change the dimension.
    the reshape function is provided in numpy.
    transpose is provided in python.
    the difference between them is the former will not change the data order.
    the last will change the data order. the same for theses two function
    is they can both change the data dimension.
    transpose is dedicated to change the data shaft, just like two dimension matrix.
    the efficient of transpose is transpose.
    a = 
    [
        [1, 2, 3],
        [3, 4, 5]
    ]
    b = transpose(a, 0, 1) = 
    b = 
    [
        [1, 3],
        [2, 4],
        [3, 5]
    ]
    c = a.reshape(3, 2)
    c = 
    [
        [1, 2],
        [3, 3],
        [4, 5]
    ]
    you can find the reshape function will not change the data order.
    so it is dedicated to using for the image data.
    you can find the auuracy is very low if you used nn algorithm what means
    nearest neighbor.
    then we will test the knn algorithm what means k-nearest neighbor.
    k means to find the the nearest k neighbor, what can also named it as
    the k min distances. nn just find the smallest distance and find the index
    in X_train as the predict label, knn will find k smallest distance and find
    all the k index in X_train and use the max numbers labels as the predict label.
    just like the 7 smallest distance is 1, 10, 23, 24, 56, 87, 88.
    the find the label in X_train data based on the 7 index.
    and return the label that in most times.
    just like the laebls is as follow that the 7 index in X_train data.
    1, 1, 2, 1, 3, 1, 1.
    then we can get the predict label is 1. because it occurs the most times.
    """
    nn = NearestNeighbor()
    nn.train(X_train.reshape(50000, -1), y_train)
    yPred = nn.predict(X_test.reshape(10000, -1), y_test)
    