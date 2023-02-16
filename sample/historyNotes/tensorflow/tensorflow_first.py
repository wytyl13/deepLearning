"""
we will start with tensorflow. we assume that we have certain basis about neural network.
tensorflow will create a calculation chart, but it will not automatic calculation this calculation chart.
tensor is the basic unit for tensorflow.
1 is a scalar
[1,2,3] is a vector
[[1,2,3],[4,5,6]] is a matrix or an array.
the tensor has the static type and the dynamic dimension.
it means you can change its dimension but can not change its type.
TensorFlow.tensor is the object about tensor. it is a class, that
has two attributes, just like data type(float32) and shape([1,2]).
each element in a tensor will have the same data type, and you need not
to define the shape about tensor.
in addition to the basic tensor that the class tensor, it has also
Varibale, constant, placeholder three tensor type. they are dedicated to 
one condition, just like variable class is dedicated to weight, constant
is dedicated to the fix tensor, and the palceholder class is dedicated to 
the dataset.
"""

import tensorflow as tf
# because we used the version 2.6, and the session object is version 1.
# so you should code the follow code to make two version can be conpatible.
# this session object is also the version 1.
tf.compat.v1.disable_eager_execution()

x1 = tf.constant(1)
x2 = tf.constant(3)
z = tf.add(x1, x2)
# util here, the three code above has created the calculation chart used tensorflow
# then we will calculate it. we will use the calculation object.
# you should use the run function in session object that dedicated to calculating the calculation chart.
# then, you should pass the tensor object z as a param into the run function.
# this function will return the calculation value.
sess = tf.compat.v1.Session()
print(sess.run(z))
# you can also run one tensor. it will return the value.
# it is also useful in one special condition.
print(sess.run(x1))

# we can also use variable attribute to create the tensor.
x1 = tf.Variable(1)
x2 = tf.Variable(2)
z = tf.add(x1, x2)
# you should init the tensor variable first.
# you can use the code as follow to init the tensor variable.
# but this attribute variable has deleted in version 2 or greater version.
# it is different from variable and constant, so you should
# init the variable first, you can use the initializer function to init
# it, or you can use the global_variables_initializer to init all variable.
# you can use the tensor variable you have defined directly.
# sess.run(x2.initializer)
# sess.run(x1.initializer)
# but this function global_variables_initializer is dedicated to version 1,
# so you should add compat.v1 before used this function
# this code can init all the variable.
init = tf.compat.v1.global_variables_initializer()
sess.run(init)
print(sess.run(z))

# then we will learn how to use the placeholder attribute to create a tensor.
# it is different from the constant and variable attribute, 
# you should statement the placeholder first, then you can just calculate this chart you have statemented.
# the first param is the element type you want to store, the second 
# param is the dimension of the tensor you want to statement.
# because this placeholder attribute is dedicated to v2 or greater version
# so you should add the compat.v1 before you used it.
x1 = tf.compat.v1.placeholder(tf.float32, 1)
x2 = tf.compat.v1.placeholder(tf.float32, 1)
z = tf.add(x1, x2)

# then, if you has not init the value for x1 and x2, you will return a type details about z.
# notice, the structure of the figure or chart and calculation were independent of each other.
# we can assigment value for these two placeholder tensor used dict
feed_dict = {x1 : [1], x2 : [2]}
# you should pass this dict into the run function as the second param.
# the first param is calculate figure, the second param is the assigment.
# the run function is dedicated to the current session that you have created.
print(sess.run(z, feed_dict))

# then, we can temp to define a vector tensor. then calculate it.
x1 = tf.compat.v1.placeholder(tf.float32, [2])
x2 = tf.compat.v1.placeholder(tf.float32, [2])
z = tf.add(x1, x2)
feed_dict = {x1 : [1, 1], x2 : [2, 2]}
print(sess.run([x1, x2, z], feed_dict))

# you should close the session last.
sess.close()
# then, to summarize
"""
constant is dedicated to those entity that will be never changed.
variable is dedicated to those entity that will be changed. just like the weight.
placeholder is dedicated to those entity that will be changed during the calculation.
because the tensor you created used placeholder attribute is statementation first, then you can 
just define the dict to assigment.
notice, you can also use the sess.run function to calculate a list tensor objetc, just like
sess.run([x1, x2, z]), this function will return the calculation value of x1, x2 and z.
notice, x1, x2, z are just the calculation figure, run function can calculate these figure.
x1 x2 z are the structure of objective existence, run will compute the structure.
"""

"""
then we will learn the dependencies between nodes.
it means it will automatically determine the node that
all the content of the node your requirements necessary to rely on.
it means the multiple conputations will may be happend on each node.
just like x = c + 1, y = x + 1, z = x + 2.
if you want to calculate x, y, z independent.
just like sess.run(x), sess.run(y), sess.run(z).
the x figure will be calculated three times. tensorflow will not store the first calculation result
and if you calculate them used list, the figure x will be calculated 1 time.
just like xx, yy, zz = sess.run([x, y, z]) . or you can also
used yy, zz = sess.run([y, z]). this is a general skill you can imporve your efficient.
"""
c = tf.compat.v1.placeholder(tf.float32, [2])
x = tf.add(c, 1)
y = tf.add(x, 1)
z = tf.add(x, 2)

feed_dict = {c : [1, 2]}
sess = tf.compat.v1.Session()
"""
we want to calculate x, y, z figure. the tensorflow will
automatically identity what node should calculate first, just like
this case, the second param feed_dict is the assignment param. and 
tensorflow has known it should pass what node, it is c node.
because c node is the independency for this calculation about x, y, z.
and this figure x will just calculate 1 time. the tensor flow will store the calculation result.
"""
xx, yy, zz = sess.run([x, y, z], feed_dict)
print(xx, yy, zz)
sess.close()



# then we will consider the session
# we have used the explicit declaration session.
# then we will learn how to implicitly declare session.
# you will need not to close this session if you used implicitly declare session.
# because the system automatically close it.
with tf.compat.v1.Session() as sess:
    xx = sess.run(x, feed_dict)
    print(xx)

