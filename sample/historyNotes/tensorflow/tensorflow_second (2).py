"""
this file we will learn single neurons, why named it as neurons?
this is because they are similar in biology. the simple things each
neurons will do is accept input and return the output.
we will use xi represent the input, x is range from 1 to n.
so the n is the feature for one input.
we will use weight, bias, x, f symbol. w is weight what we want to 
optimize, b is the bias, x is the input, f is the activation function.
the f will be wide variety because of you want to achieve different effect.
z = w1x1 + w2x2 + w3x3 + ... + wnxn + b
y = f(z)
the dimension of x is m*n, m is sample number, n is the feature numbers of each sample.
we will consider the simple condition no hidden layer. and the number of output layer is 1.
just like the figure as follow

x1      z    
x2      (output layer, w, b)
x..
xn

x1, x2, x3, ..., xn is X, the dimension of X is m*n
m is the sample number, n is the feature numbers of one sample.
X is the set of one sample.
z = w @ x + b
1*m = 1*n @ n*m + 1*1
y = f(z)
the y is the active function result of output z. z is the current layer output.

then we can formal each dimension, just like this case
we just have one output layer and has no the hidden layer.
so we can define the x current layer number is n_h, just like
the output layer numbers in this case is n_h. and the feature of the 
input sample is n_x. and the sample numbers is m.\
each layer has their exclusive weight and bias.
just like the layer from the first hidden layer to output layer.
the symbol will start with 1, and increasly, just like this case, 
it just has one output layer, so the output layer has w1, b1, z1, A1.
w1 is the weight for this current layer, b1 is the bias for this current layer,
z1 is equal to w1@x+b1, A1 = sigmoid(z1).
z2 = w2 @ A1 + b2. A2 = sigmoid(z2).
just because this example only has 1 layer.
the total layer numbers is hidden layer numbers add output layer.
notice, each layer can has multiple neurons. the neurons numbers of hidden layer
is represented by n_h, and the neurons numbers of output layer is represented
by n_y.
so we can use the expression to represent this forward propagation
z1 = w1 @ x + b1
n_h*m = n_h*n_x @ n_x*m + n_h*1
n_h is the nuerons numbers of the current hidden layer.
n_y is the nuerons numbers of the output layer.
m is the sample numbers of input.
n_x is the feature numbers of the input.

this case has not the hidden layer, and just has one neurons for output layer.
then we will consider the one hidden layer, and the hidden layer has four neurons.

x1      z1 = w1 @ X + b1
x2      A1 = relu(z1)            
x3      hidden1               z2 = w2 @ A1 + b2
x4      hidden2               A2 = sigmoid(z2)
x..     hidden3               output1
xn      hidden4

input   hidden layer        output layer
the neurons numbers of the current hidden layer is n_h = 4.
X = n_x, m 
n_x = n
w1 = n_h, n_x
w2 = n_y, n_h
b1 = n_h, 1
b2 = n_y, 1
z1 = w1 @ X + b1
A1 = n_h*m
A2 = n_y*m
n_h*m = n_h*n_x @ n_x*m + n_h*1
z2 = w2 @ A1 + b2
n_y*m = n_y*n_h @ n_h*m + n_y*1
the rule is 
    wi
        (the first dimension, neurons numbers of current layer, involved hidden layer and output layer)
        (the second dimension, the neurons numbers of the former layer)
    bi
        (the first dimension, current numbers of current layer, involved hidden layer and output layer)
        (the second dimension is 1, a const number)
    the first w and b, just like w1 and b1, the dimension is,
    the second dimension of w1 is the input neurons numbers that is the feature numbers for sample.
    the first dimension of w1 is the neurons numbers of the current layer.



then we will learn the actively function. just like sigmoid, relu, tanh and so on.
1 identity function
    just like as follow
    def identity(z):
        return z
2 sigmoid function
    it will return the value range from 0 to 1 if you give the input from -∞ to +∞
    def sigmoid(z):
        return 1 / (1 + e**(-z))
        or you can return
        return np.divide(1.0, np.add(1.0, np.exp(-z)))
        return 1 / (1 + np.exp(-z))
3 tanh function: the hyperbplic tangent function.
    numpy has this funcion tanh.
    this function will return the value range from -1 to 1.
4 relu function: Rectification of linear unit function.
    def relu(z):
        return max(0, z)
    this applicaiton of relu function have many methods.
        return np.maximun(x, 0)
        return x * (x > 0)
        return (abs(x) + x) / 2
        return np.maxmium(x, 0, x)
        np.maxmium(x, 0, x) is faster four times than np.maxmium(x, 0)
        the first method will not create the new array, and the second method will create the new array.
        (x > 0) will return 0 or 1.
5 leaky relu: parameterized rectifier linear unit.
    the param alpha usually is 0.01
    f(z) = alpha * z, if z < 0
    f(z) = z, if z >= 0
    you can use relu function to define lrelu function.
6 swish actively function
    this function is more accuracy than relu actively function
7 other actively function: 
    arcTan: tan(z)**-1
    elu, softplus.
        f(z) = ln(1 + e**z)
8 sigmoid and relu is the most common form of actively function.
    you can also use these two actively function to achieve any nonlinear function.
    and member that tensorflow has defined all actively function, you can use them directly.
    but you should know the concept of these actively function so that you can use them flexible.

then, we will learn the cost function and the gradient descent method that can minimize the cost function.
our pupose is to make the parameters of the cost function minimum.
if the parameter is small, you can use the method of calculus to get the minimize,
but the prameter is large in neural network, so you can not use these method in calculus, 
then the gradient descent method has been proposed!
the gradient descent algorithm is one of the best algorithm in machine learning.

give the cost function about the w
J(w), the prameter w is weight. its type is vector.
    just like abovw, we have two method to calculate the parameters w
    first, we can calculate using mathmatics method directly.
        we will use the matrix operation, and derivative.
    second, we can give a random initialization w.
        then, we can give the learning rate alpha.
        just like this cost funciton is a u type function.
        what we want to do is find the w parameter that made the cost function minimum.
        the gradient descent method used the iteration concept.
        if the prameter w is just one. just like the w is a element not a vector.
        so this problem is simple. we can calculate it by ourselves.
        but we can also used iteration method to calculate it.
        imagine the cost function type is a u. give a random initialization w.
        image we will get the max cost, just like we had on both sides of u.
        then, our purpose is minimize the cost, just like the bottom of u.
        so we can calculate the partial derivative of the cost function. 
        the value we calculated is the slope, the direction of the slope is up.
        but we want to get the bottom, so we should use the opposite of the slope.
        and the w parameters is corresponding to the cost function value, so the direction
        of w is equal to the cost funciton value. so we just need to get the same direction
        w and cost funciton, we can use the slope direction to update the parameters w.
        so it will be this, w(n+1) = w(n) - alpha * J(w)'
        J(w)' means the partial derivative of cost function about w.
        if the w parameters is a vector, so you should iteration for each w.
        this means the next w is updated by the former w. and alpha is the leaning rate.
        it means we will use the negative of glope. because we want to arrive the bottom of the u.
    then, the learning rate alpha and iteration numbers, we should give through experience and practice.
        the learning rate generally is 0.01, and the iteration numbers generally is 100~1000.
        we can give a condition iteration numbers, just like, while the little change happend in current cost value and former cost value,
        we should stop the iteration. or you can dedine a threshold value. just when |J(wn+1) - J(w)| < epsilon.
    last, we should know, the better learning rate will result in faster speed.
        the learning rate means the step size in the direction of the negative of the slope. so it is very important for the learning speed.
        and the slope will determin the accuracy of learning result.
        a good rule of thumb is start from 0.05, and test the cost value changes in different iteration numbers.
        and you should test smaller learning rate gradually.
        if you find that the cost value changed little with the iteration numbers increasely, so you 
        should consider the smaller learning rate. and if you find that the cost value is falling, you should 
        add the iteration numbers.
        notice, the absolute value of learning rate is meaningless, the important is 
        the falling speed and behavior of cost value.

then, we will learn the logistic regression.
    the cost function is cross entropy in logistic regression.
    L(y_ - y) = -(y * log(y_) + (1 - y) * log(1 - y_)
    J(w, b) = 1/m * ΣL(y_ - y)
    and, we want to the discrete output, so we should use the actively function.
    sigmoid(z) = 1 / (1 + e^(-z))
    then, we will implementation logistic regression used tensorflow. we can also use the python languege implementation it directly.
    but it is meaningless, you can try to do a time if you want to do it. i have done many times used it, but every time will not
    bring the ascension.
    notice, the logistic regression will use sigmoid as actively function,
    and sigmoid funciton can not accept too big input. so we usually used 
    sigmoid as the actively function of output layer
    notice, the normalization method is different between picture and other data, because the 
    picture data elements is range from 0 to 255, and other data elements are ruless. so we can use
    255 to normalize the picture elements. it is so simple that you can just calculate each element divided by 255.
"""

import tensorflow as tf
import random
import time
import numpy as np
from sklearn.datasets import load_boston
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
tf.compat.v1.disable_eager_execution()

def relu(z):
    return np.maxmium(z, 0, z)

def lrelu(z, alpha):
    return relu(z) - alpha * relu(-z)

# the currengt element minus the mean, and divided by standard deviation.
# the benefits of features normalization is we will ignore the effects of abnormal
# characteristic value of results.  
def normalize(X):
    mean = np.mean(X, axis = 0) # mean each feature for all samples, so you should use the dimension 0, it means the longitudinal.
    sigma = np.std(X, axis = 0)
    return (X - mean) / sigma

"""
then we defined the function to optimize the linear regression used tensorflow.
it will be easy. you should notice that, it is independent for graph and calculation in tensorflow.
you should explicit declaretion the graph element, just like W, b, X and Y.
define the W and b used Variable, define the X and Y used placeholder.
the none means we want to implicitly declared. notice, all variable in graph are param
we can change them for each case.
cost = 1/m * Σ(y_ - y)^2
y_ = w.T @ X + b
"""
def linear_regression_graph(training_samples_n):
    tf.compat.v1.reset_default_graph()
    X = tf.compat.v1.placeholder(tf.float32, [training_samples_n, None]) # n*m we need not to define the sample numbers, because we can pass any sample numbers.    
    Y = tf.compat.v1.placeholder(tf.float32, [1, None]) # 1*m
    learning_rate = tf.compat.v1.placeholder(tf.float32, shape = ()) # none shape.
    # init the w used 1 and init b used 0
    W = tf.Variable(tf.ones([training_samples_n, 1])) # n*1, notice the []
    b = tf.Variable(tf.zeros(1)) # 1*1
    init = tf.compat.v1.global_variables_initializer()
    y_ = tf.matmul(tf.transpose(W), X) + b # 1*n @ n*m + 1*1 = 1*m
    cost = tf.reduce_mean(tf.square(y_ - Y)) # 1*1 value
    training_step = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    return X, Y, learning_rate, W, b, y_, init, cost, training_step

# then define the function that run the graph we have created used the former function
# notice, you should use the different variable name to accept the arguments or parameter. 
# becaus python language is loosely, so you should use the strict code. or you will get the erro during the running.
def run_linear_regression(training_samples_n, learning_r, training_epochs, train_obs, train_labels, debug = False):
    X, Y, learning_rate, W, b, y_, init, cost, training_step = linear_regression_graph(training_samples_n)
    sess = tf.compat.v1.Session()
    sess.run(init)
    cost_history = np.empty(shape = [0], dtype = float) # init a float type variable to store all the costs. we can also define a list directly.
    for epoch in range(training_epochs + 1):
        sess.run(training_step, feed_dict = {X: train_obs, Y: train_labels, learning_rate: learning_r})
        cost_ = sess.run(cost, feed_dict = {X: train_x, Y: train_y, learning_rate: learning_r})
        cost_history = np.append(cost_history, cost_)
        if(epoch % 1000 == 0) & debug:
            print("reached epoch", epoch, "cost J = ", str.format('{0:6f}', cost_))
    return sess, cost_history

# define a function that can show the picture used a list elements
# you should pass a row in datasets, we will show the picture.
def plot_matrix(elements):
    matrix = elements.reshape(28, 28)
    plt.imshow(matrix, cmap=matplotlib.cm.binary, interpolation="nearest")
    plt.axis("off")
    plt.show()

def normalize_picture(datasets):
    return datasets / 255.0


# define the logistic placeholder, it is roughly equal to the linear regression.
# the different is the defined about y_ and cost function
def logistic_regression_graph(training_samples_n):
    tf.compat.v1.reset_default_graph()
    X = tf.compat.v1.placeholder(tf.float32, [training_samples_n, None]) # x(n, m)
    Y = tf.compat.v1.placeholder(tf.float32, [1, None]) # y(1, m)
    learning_rate = tf.compat.v1.placeholder(tf.float32, shape = ())
    W = tf.compat.v1.Variable(tf.zeros([1, training_samples_n]))
    b = tf.compat.v1.Variable(tf.zeros(1))
    init = tf.compat.v1.global_variables_initializer()
    y_ = tf.sigmoid(tf.matmul(W, X) + b)
    cost = -tf.reduce_mean(Y * tf.compat.v1.log(y_) + (1 - Y) * tf.compat.v1.log(1 - y_))
    training_step = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    return X, Y, learning_rate, W, b, y_, init, cost, training_step


# we have defined the placeholder to create logistic graph, then we should define the 
# method to calculate it used tensorflow. but you should notice log in this function,
# because log(0) is meaningless, so it will be meaningless if y_ equal to 0 or 1.
# and this condition will happen when leanring_rate is big, cost_ will return nan.
# if it happend, you should test the smaller learning_rate.
# so you should notice the reason of normalize the data is to reduce the original data, so much so that w@x+b will be not
# particularly large or very small. then the return value of sigmoid fucntion will be range from 0 to 1, not 0. or the cost_
# will be nan, because the log(0) is meaningless.
def run_logistic_regression(training_samples_n, learning_r, training_epochs, train_obs, train_labels, debug = False):
    X, Y, learning_rate, W, b, y_, init, cost, training_step = logistic_regression_graph(training_samples_n)
    print(y_, cost)
    sess = tf.compat.v1.Session()
    sess.run(init)
    cost_history = np.empty(shape=[0], dtype = float)
    for epoch in range(training_epochs + 1):
        sess.run(training_step, feed_dict = {X: train_obs, Y: train_labels, learning_rate: learning_r})
        cost_ = sess.run(cost, feed_dict = {X: train_obs, Y: train_labels, learning_rate: learning_r})
        cost_history = np.append(cost_history, cost_)
        if(epoch % 500 == 0) & debug:
            print("reached epoch", epoch, "cost J = ", str.format('{0:.6f}', cost_))
        
    return sess, cost_history


if __name__ == "__main__":
    
    # test tensorflow basic content.
    """
    list1 = random.sample(range(1, 1000), 100)
    list2 = random.sample(range(1, 1000), 100)
    # test the time consumption used circle method
    start = time.time()
    a = [list1[i] * list2[i] for i in range(len(list1))]
    end = time.time()
    print("程序运行的时间为：{}".format(end-start))

    # the test the time consumption used numpy
    # you should cast the object from list to array.
    list1_np = np.array(list1)
    list2_np = np.array(list2)
    start = time.time()
    b = np.multiply(list1_np, list2_np)
    end = time.time()
    print("程序运行的时间为：{}".format(end-start))
    
    #the you can find that the time consumption used circle is large than used numpy.
    # because numpy used matrix and the circle used for circle.
    # another reason is numpy used c language. and the python circle is original python code.
    # although the python made by c language, but the first method used python language circle keyword.

    start = time.time()
    x = np.random.random(10**8)
    end = time.time()
    print("程序的运行时间为：{}".format(end - start))
    # generate a list involved 10**8 random numbers.
    # this virtual machine will run 3.5s and the local machine will run 0.72s
    # and the different time you will consumption if you used the different machine.

    #then, we will test the time consumption used the different relu method.
    """


    # test linear_regression created by tensorflow
    """
    # created the linear regression used tensorflow.
    # m is the sample numbers, and the n is the feature numbers for one sample.
    # notice, we should as far as possible avoid the use of circukation, as far as possible
    # use numpy matric.
    # we should explicity declare matrix dimension in tensorflow, only in this way can we good control over them.
    # we will use the boston dataset.
    boston = load_boston()
    features = np.array(boston.data) # X, m*n
    lables = np.array(boston.target) # y, m*1

    # print(boston["DESCR"])
    training_samples_m, training_samples_n = features.shape
    print(training_samples_m, training_samples_n)
    # you can find, this datasets have 506 samples and each sample has 13 features.
    # we have got the datasets, and then we should normalized datasets. defined the function above.
    features_norm = normalize(features)
    print(features)

    # this case, we will use X: n*m, y: 1*m, b: 1*m, w: n*1
    # so we should transpose first.
    train_x = np.transpose(features_norm)
    # notice, you should reshape from 506 to 1*506. the 506 is dedicated to numpy, we need to use the matrix
    train_y = np.transpose(lables).reshape(1, len(lables))

    print(train_x.shape)
    print(train_y.shape)

    sess, cost_history = run_linear_regression(training_samples_n, 0.01, 10000, train_x, train_y, debug = True)
    # as is known to all, the smaller learning rate, the faster computational efficiency.
    """

    # test logistic_regression created by tensorflow
    mint = tf.keras.datasets.mnist
    (train_x, train_y), (test_x, test_y) = mint.load_data()
    print(train_x.shape, train_y.shape) # train data, 60000, 28*28
    print(test_x.shape, test_y.shape) # test data, 10000
    # the data sets have 70000 samples.
    # plot_matrix(train_x[0])

    # first, this datasets involved 10 numbers, from 0 to 9.
    # we can simple this problem. it means we will find 0 and 1 number samples,
    # then create the logical model for these samples.
    train_x_01 = train_x[np.any([train_y == 1, train_y == 2], axis = 0)]
    train_y_01 = train_y[np.any([train_y == 1, train_y == 2], axis = 0)]

    print(train_x_01.shape, train_y_01.shape)

    train_x_01_normalize = normalize_picture(train_x_01) #x(m, n)
    train_y_01_last = train_y_01 - 1

    # just like the dimension above for linear regression.
    # x(n, m) y(1, m) b(1, m) w(n*1), it is different from neural network.
    # logistic regression has not the hidden layer, and just has a neurons.
    # so the dimension of w is (n, 1); and the dimension for neurons is w(n, m)
    # the other is same, but this case just has the output layer. the neurons network
    # will consider multi hidden layer and multi neurons for each hidden layer.
    # you should tensile each sample first. it means you should cast the dimension
    # from 28*28 to 1*784, you can use flatten or reshape
    train_x_01_flatten = train_x_01_normalize.reshape(train_x_01.shape[0], -1) # from(m, 28, 28) to (m, 784)
    n_dim = train_x_01_flatten.shape[1]
    # then you should adjust the dimension for each variable.
    train_x_01_normalize_tr = train_x_01_flatten.transpose() # x(n, m)
    train_y_01_last_tr = train_y_01_last.reshape(1, train_y_01_last.shape[0]) # y(1, m)
    print(train_x_01_normalize_tr.shape, train_y_01_last_tr.shape)
    print(train_x_01_normalize_tr[:,:1])
    # we have finished the data processing, then we should create the logictic
    # model used tensorflow, it is same to linear regression, we should define the
    # placeholer and variable to form an image structure. then calculate it.
    sess, cost_history = run_logistic_regression(n_dim, 0.005, 5000, train_x_01_normalize_tr, train_y_01_last_tr, debug = True)

    