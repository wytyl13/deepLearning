'''**********************************************************************
 * Copyright (C) 2023. IEucd Inc. All rights reserved.
 * @Author: weiyutao
 * @Date: 2023-05-10 14:29:39
 * @Last Modified by: weiyutao
 * @Last Modified time: 2023-05-10 14:29:39
 * @Description: 
 then, we will test the knn algorithm, because the efficient of nearest neighbor is
 very low, so we will get part of the samples.

 ok, we have implemented the knn algorithm based on the L2 distance,
 and define three method to implement the L2 distance.
 we have found it will be most efficient if we drop the for loop in the function.
 but one problem is we have defined one super parameter k. so we should
 define one function that can find the best parameter k used cross
 validation method.

 but generally, the linear classification is no much turned over for
 some special problems. just like the cases as follow.
 1 1 0 0
 1 1 0 0
 0 0 1 1
 0 0 1 1
 this case need to two line to distinguish it at least.
***********************************************************************'''
from nn import *
from data_utils import load_CIFAR10
from data_utils import imshowMatrix

class KNearestNeighbor(NearestNeighbor):

    def __init__(self) -> None:
        super().__init__()
    
    # you need not to explicit statement the attribution in the current class.
    # you can define it when you need to.
    def train(self, X, y):
        self.X_train = X
        self.y_train = y
    '''
     * @Author: weiyutao
     * @Date: 2023-05-10 15:32:58
     * @Parameters: 
        X: X_train data.
        k: the number of nearest neighbors that vote for the predicted labels.
        num_loops: the L2 distance algorithm parameters invovled zero loop, one loop and two loops.
        of course, you can also use L1 distance algorithm here.
     * @Return: the predict labels.
     * @Description: 
     '''
    def predict(self, X, k = 1, num_loops = 0):
        if num_loops == 0:
            # no explicit loops.
            dists = self.L2NOLOOPS(X)
        elif num_loops == 1:
            dists = self.L2ONELOOPS(X)
        elif num_loops == 2:
            dists = self.L2TWOLOOPS(X)
        else:
            raise ValueError("Invalid value %d for num_loops" % num_loops)
        """ 
        of course, we can visualize the distance matrix.
        imshowMatrix(dists)
        """
        return self.predict_labels(dists, k = k)


    '''
     * @Author: weiyutao
     * @Date: 2023-05-10 16:15:17
     * @Parameters: 
        X: X_test
     * @Return: 
        dists: A numpy array that shape is (num_test, num_train) where dists[i, j].
        notice, it is similar as L1 distance, because you need to compare each test samples
        with each samples. so you will get num_test, num_train dimension array distance result.
     * @Description: computer the L2 distance based on each point in X what is the test samples
     and each training point in self.X_train using no explicit loops. it is one conventional algorithm
     one loops means we used one for loop.
     '''
    def L2ONELOOPS(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            # L2: d2(I1, I2) = [xigema((I1 - I2)^2)]^(1/2)
            dists[i, :] = np.sqrt(np.sum(np.square(self.X_train - X[i, :]), axis = 1))
        return dists

    # TWOLOOPS means to not use the matrix attribution to calculate.
    # just used two for loops. so it will be lowest efficient.
    # but you should notice, these calculation method will have the same result dists.
    def L2TWOLOOPS(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                dists[i, j] = np.sqrt(np.sum(np.square(self.X_train[j, :] - X[i, :])))
        return dists

    """ 
    NO LOOPS means we will not use the for loop but used the matrix multi operations.
    L2 = np.sqrt(Σ(np.square(I1 - I2))) = np.sqrt(Σ(I1^2 -2*I1*I2 + I2^2))
    mul = 2*I1*I2    dimension is (num_test, features) @ (features, num_train) = (num_test, num_train)
    X1 = I1^2 dimension is (num_test, 1)
    Y1 = I2^2 dimension is (1, num_train)
    """

    def L2NOLOOPS(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        # matrix multi operation
        mul = np.dot(X, self.X_train.T)
        X1 = np.sum(np.square(X), axis = 1)
        X1 = X1.reshape(X.shape[0], 1)
        Y1 = np.sum(np.square(self.X_train), axis = 1)
        Y1 = Y1.reshape(1, self.X_train.shape[0])
        dists = np.sqrt(X1 - 2 * mul + Y1)
        return dists

    '''
     * @Author: weiyutao
     * @Date: 2023-05-10 17:01:41
     * @Parameters: 
     * @Return: 
     * @Description: 
     '''
    def predict_labels(self, dists, k = 1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            closest_y = []
            # return the indicies idx sorted based on ascending.
            # and get the first k indicies.
            idx = np.argsort(dists[i])[:k]

            # get the k labels from y_train based on the idx we have got.
            closest_y = self.y_train[idx]

            # bincount: return the number of occurrences of each element in closest_y
            # argmax: return the index of the max number of occurences. of course
            # you can alse return the row or column indicies by adding the axis
            # if the parameters is two dimension data.
            # find the label that max number of occurrences in closest_y what got
            # from y_train based on the idx.
            y_pred[i] = np.argmax(np.bincount(closest_y))
        return y_pred
    
    def time_function(self, f, *args):
        # calculate the wasted time that one function with args
        import time
        tic = time.time()
        f(*args)
        toc = time.time()
        return toc - tic

    def getAccuracy(self, X, y_test, k = 1, num_loops = 0):
        y_predict = self.predict(X = X, k = k, num_loops = num_loops)
        num_test = y_test.shape[0]
        num_correct = np.sum(y_predict == y_test)
        accuracy = float(num_correct) / num_test
        # print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))
        return accuracy

    """ 
    what is cross validation? split the train samples into n copies, use each copies
    as the test data and use all the samples as train data to predict the test samples.
    then, you should predict n times.
    then you will get n accuracy for each copies.
    then for loop one k superparameter list. excute the former code.
    you should get the best superparameter in the result.
    1...50...100...150...200
    train samples is form 1 to 200.
    predict 1...50, 51...100, 101...150, 151...200 use one k and get the accuracy. you should get one accuracy list.
    then test another k used the former method.
    you should get each accuracy based on each k, select the best k at last.
    notice, if your class instancement has set train funcitono out of the function crossValidation,
    you should reset it into this function. but if you have defined the same samples in your case, 
    you need not to reset it, because the training samples are similar.
    """
    def crossValidation(self, X_train, y_train, k_choices):
        # define the batch
        num_folds = 5
        X_train_folds = []
        y_train_folds = []
        # split X_train and y_train into num_folds copies.
        X_train_folds = np.array_split(X_train, num_folds)
        y_train_folds = np.array_split(y_train, num_folds)

        k_to_accuracies = {}
        for k in k_choices:
            k_to_accuracies[k] = []
            for i in range(num_folds):
                X_val = X_train_folds[i]
                y_val = y_train_folds[i]
                # tranform the X_train_folds from a list to numpy.ndarray.
                X_tr = np.concatenate([X_train_folds[j] for j in range(num_folds) if j != i])
                y_tr = np.concatenate([y_train_folds[j] for j in range(num_folds) if j != i])
                self.train(X_tr, y_tr)
                accuracy = self.getAccuracy(X_val, y_val, k)
                k_to_accuracies[k].append(accuracy)
        return k_to_accuracies

    def printDict(self, dict):
        for key in sorted(dict):
            for value in dict[key]:
                print("key = %d, value = %f" % (key, value))


    def imshowCrossValidation(self, dict, k_choices):
        for k in k_choices:
            accuracies = dict[k]
            plt.scatter([k] * len(accuracies), accuracies)
        # plot the trend line. with error bars that correspond to standard deviation.
        accuracies_mean = np.array([np.mean(v) for k, v in sorted(dict.items())])
        accuracies_std = np.array([np.std(v) for k, v in sorted(dict.items())])
        plt.errorbar(k_choices, accuracies_mean, yerr = accuracies_std)
        plt.title("cross validation on k")
        plt.xlabel("k")
        plt.ylabel("cross validation accuracy")
        plt.show()

if __name__ == "__main__":
    cifar10_dir = 'data/cifar-10-batches-py'
    try:
        del X_train, y_train
        del X_test, y_test
        print('Clear previously loaded data.')
    except:
        pass

    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    nn = NearestNeighbor()

    """ 
    imshow the train samples image.
    nn.showImage(X_train, y_train)
    """

    X_train, X_test, y_train, y_test = nn.subSample(5000, 500, X_train, X_test, y_train, y_test)
    knn = KNearestNeighbor()

    """  
    you should set the train function when you use the KNearestNeighbor class instancement
    to predict the test samples.
    """
    knn.train(X_train, y_train)

    """ 
    predict used knn
    y_predict = knn.predict(X_test, k = 1, num_loops = 0)
    num_test = y_test.shape[0]
    num_correct = np.sum(y_predict == y_test)
    accuracy = float(num_correct) / num_test
    print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))
    """

    """ 
    test the difference between no, one and two loops for knn algorithm.
    dists = knn.L2NOLOOPS(X_test)
    dists_one = knn.L2ONELOOPS(X_test)
    difference = np.linalg.norm(dists - dists_one, ord = 'fro')
    print('the difference between one loops and no loops was: %f' %(difference, ))
    """

    """ 
    calculate the difference of time wasted between no, one and two loops method for L2 distance.
    no_loop_time = knn.time_function(knn.L2NOLOOPS, X_test)
    one_loop_time = knn.time_function(knn.L2ONELOOPS, X_test)
    two_loop_time = knn.time_function(knn.L2TWOLOOPS, X_test)

    print('NO loop version took %f seconds' % no_loop_time)
    print('ONE loop version took %f seconds' % one_loop_time)
    print('TWO loop version took %f seconds' % two_loop_time)

    NO loop version took 0.390990 seconds
    ONE loop version took 77.160246 seconds
    TWO loop version took 74.091275 seconds
    you can find that the NOLOOP METHOD IS the biggest efficient.
    knn.getAccuracy(X_test, y_test, k = 5, num_loops = 0)
    """

    """ 
    test to find the best hiperparameter k used cross validation method.
    """
    k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]
    k_to_accuracy = knn.crossValidation(X_train, y_train, k_choices)
    knn.imshowCrossValidation(k_to_accuracy, k_choices)
    

