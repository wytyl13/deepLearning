/**********************************************************************
 * Copyright (C) 2023. IEucd Inc. All rights reserved.
 * @Author: weiyutao
 * @Date: 2023-05-29 10:27:58
 * @Last Modified by: weiyutao
 * @Last Modified time: 2023-05-29 10:27:58
 * @Description: reference the deep learning book.
 * the application of machine learning including but not limited to all the scenario as follow.
 * classification, regression, machine translation, structed output, anomaly detection, 
 * the synthesis and sampling, misssing value fill, denoising.
 * 
 * unsupervised and supervised learning algorithm.
 *      supervised: p(x)
 *      unsupervised: p(y|x)
 * generally, there are not the strict boundaries between supervised and unsupervised.
 * we can transform from the unsupervised to supervised learning algorithm.
 * p(x) => p(y|x) = π(i=1_n)p(xi|x1, x2, x3, ..., xi-1) 
 * p(y|x) = p(x, y)/Σp_y'(x, y') = p(x, y) / p(x)
 * 
 * we should notice that how to get the probabilty of one two dimension random variables.
 * you should distinguish the continune and discrete variables.
 * continune: the probabilty desity function.
 *      P(X, Y) = ∫∫f(x, y)dxdy = 1.
 * discrete: the probabilty distribution function.
 *      P(X = x_i, Y = y_j) = p_ij, and P(X, Y) = 1.
 * these are joint probabilty density function above,  we can define the edge density function used it.
 * edge desity function:
 *      f_X(x) = ∫f(x, y)dy
 *      f_Y(y) = ∫f(x, y)dx
 * edge probabilty distribution function:
 *      P(X = x_i) = Σ_j p(X = x_i, Y = y_j)
 *      P(Y = y_j) = Σ_i p(X = x_i, Y = y_j)
 * 
 * traditionally, the regression, classification and structed output are called as surpervised learning problem.
 * other tasks of density estimation are called as unsurpervised learning problem.
 * there is a scene, the training data is changed with the environment.
 * these algorithm are called as the reinforcement learning. it is more complex.
 * most of the machine learning algorithm running in the fixed data. the data are the set of all the samples. 
 * and the samples are the set of all the features. we will store it used the design matrix.
 * but this method is not always effective, just like you want to store the different size images.
 * more size image has more pixel. so it will has the different columns for the design matrix if you used the original
 * image, and do nothing. we called this problem as heterogeneous data.
 * in order to handle these problems. we generally will not express the data used m rows, but the combined of 
 * m element: {x(1), x(2), x(3), ..., x(n)}. this expression means each element x(i) could have the different size.
 * 
 * how to define the machine learning algorithm?
 *      through experience on certain tasks in order to improve the computer program performance of the algorithm.
 * the linear regression:
 *      y^ = w.T @ x, w is the parameter vector. you can also call it as weight. it will be produce very big effect
 *      for the prediction if the weight is big. and it will be no effect on the prediction.
 * so we can define the tasks based on above: 
 *      task: y^ = w.T @ x
 *      performance measurement: we can use the simple method OLS(MEAN SQUARED ERROR). it can also be named as 
 *      euclidean distance.
 *          MSE_test = 1/mΣ(y^_test - y_test)^2
 *      we can test the performance for the algorithm used the performance measurement MSE_test. but how to train
 *      the weight? we can use the derivative of MSE in the train data.
 *          MSE_train = 1/mΣ(y^_train - y_train)^2 => MSE_train'_w = 0
 *          [1/mΣ(X_train @ w - y_train)^2]'_w = 0 => [(X_train @ w - y_train).T @ (X_train @ w - y_train)]'_w = 0
 *          => [(w.T @ X_train.T - y_train.T) @ (X_train @ w - y_train)]'_w = 0 => 
 *          [(w.T @ X_train.T @ X_train @ w) - (w.T @ X_train.T @ y_train) - (y_train.T @ X_train) @ w + (y_train.T @ y_train)]'_w = 0
 *          => [(w.T @ X_train.T @ X_train @ w) - 2w.T @ X_train.T @ y_train + y_train.T @ y_train]'_w = 0
 *          => 2X_train.T @ X_train @ w - 2X_train @ y_train = 0
 *          => w = (X_train.T @ X_train)_-1 @ X_train.T @ y_train
 * so we can get MSE_train'_w = 0 => w = (X_train.T @ X_train)_-1 @ X_train.T @ y_train. this is normal equation method.
 * linear regression model: y^ = w.T @ x + b, the nature is the mapping from x to y. it can be also named as affine
 * function. in order to add the bias variable b, we can add one row for the train data x used 1 as the value of pixel.
 * of course, the initialization of weight should be has the same size as the x.
 * 
 * error: we can use the training error and generalization error. the former is to train the expected weight. and
 * the last is order to get the bigger accuracy. so we can use error_train and error_test to express the error.
 * but the test error is the expected result what we should treat it as the key consideration.
 * but we could just observed the training set, how to influence the performance of the test set?
 * we can do nothing in the design algorithm if the training set and testing set arbitrary data collection, 
 * but we can do something in the design algorithm if the training set and testing set have the conditional.
 * generally, we will do independent identically distributed hypothesis what means all the samples in each data set
 * are independent of each other. and the training set and testing set are with the same distribution.
 * this can be named as the data generating distribution. expressed as P_data. this assuming allow us to mathmatically
 * studies the relationship between the training error and testing error. the direct link about these two error we can
 * observe is the expect about these two error is same. assuming the probability distribution p(x, y), repeated sampling
 * to generate the training set and testing set from the original data. for one fixed weight, these two error has the same
 * expect. this is because the calculation about these two expect used the same data generating process. so we can name 
 * this used data generating process. it is one sample assuming and statistic method. and this distribution can be named as
 * the data generating distribution.
 * of course, the difference is we will not fix the weight first, but sampling to get the training set, then select the parameters
 * to reduce the training error to get the weight. then, sampling to get the testing data. at last, use the trained weight
 * to get the testing error, you can find that the expect of testing error is greate or equal to the training error.
 * so we can get two factors that can influence the performance of machine learning algorithm.
 * first, we should reduce the training error.
 * second, we should close the gab between training error and testing error.
 * if the first error is big, the model result will be underfitting.
 * if the second error is big, the model result will be overfitting.
 * these two factors are the major challenge for the machine learning.
 * the capacity for the model can control the model to prefer to the undefitting or the overfitting.
 * the capacity of the model is the ability for the model to fitting any function.
 * the lower capacity model could be poor performance for the training set. and the high capacity model could
 * be poor performance for the testing set. so the former will be result to underfitting and the last will be result to the overfitting.
 * why, because the high capacity model could remember the feature that do not apply to the test set of the training set.
 * 
 * one method could control the training set algorithm is to select the hypothesis space.
 * just like, the linear regression function will select all linear function of the input as the hypothesis space.
 * the hypothesis space of generalized linear regression involved polynomial function and linear function. it will
 * improve the capacity of the model. so you can find that nonlinear regression will be most likely has a fitting, 
 * because the capacity of the nonlinear regression model is bigger.
 * just like y^ = wx + b, y^ = w1x + w2x^2 + b, y^ = Σ_i=1_9 wix^i.
 * the last has bigger capacity, so it will result to the higher probabilty of overfitting.
 * 
 * then, we can consider the generalization ability for one model, just like one regression.
 * one polynomial model, just like linear, secondary and nine times model.
 * you can find one simple nonlinear model. linear model can not show the curvature functions. so it will be
 * owe fitting, nine times function can show the correct function, but it will be badly for generalization ability.
 * because the training parameters is more than the training samples. so it will be overfitting.
 * 
 * so far, we have learned two method involved change the input feature numbers and add the corresponding parameters
 * of these features to change the capacity of the model.just like reduce the numbers of weight or reduce
 * the polynomial numbers. in face, there are many othere method to change the capacity of the model.
 * the model specifies we can select the model from which functions when we implemented the training target.
 * this can be named as representational capacity. you can image it as the capacity of one model.
 * because the representational capacity of one model means the capacity to fit different type data for one model.
 * the smaller representational capacity will unable to capture the advanced features of the data.
 * but the learning algorithm will not always find the optimal function but find the function that can reduce greatly
 * the training error. so it means the effective capacity of the learning algorithm will be smaller than the representational
 * capacity.
 * 
 * so this method to get the optimal function is similar to the occam's razor what said that among hypothesis that can 
 * explain known obeserved phenomena equally well, we should choose the simplest one. just like the overfitting and 
 * owe fitting, we can not select them, we should select the simplest function but the fitting result is the best.
 * because the simplest function has the greater generalization capacity.
 * then, how to quantitative the capacity of one model? there are many method, one of the most famous method is
 * vapnik-chervonenkis dimension. but notice, it is very difficult to quantitative the capacity of one model.
 * 
 * we have learned the concept of parameter model, then we will learn the non-parametric model.
 * the paramtric model has the limited and fixed feature. non-parametric model without the limit.
 * just like the nearest neighbor regression, it is one non-parametric model. it has not set the parameters weight.
 * the model is simplest, just like when you want to classifier the test point x, model will select the k point that
 * the distance is nearest to x point. so we have not set any parameters for the model. the different method will
 * express used the different distance function, just like L1 or L2.
 * of course, we can also define one non-paramtric model used parametric model and non-parametric model.
 * just like we can define one model based on two layer. the out layer is to test the different polynomial times.
 * it is non-parametric, and the inside layer is to do the linear regression.
 * 
 * the idea model assuming that we know the probability distribution of the generated data in advace. because the training
 * set and the test set has the same distribution, this is the premise that we can do regression.
 * because these two set are both random sampling from one original data. and the distribution of the original data is
 * fixed and known.
 * 
 * just like the knn algorithm. it is the non paramters. the k nearest neighborhood.
 * you should notice the relationship between the capacity and error.
 * the optimal capacity, underfitting, overfitting, training error, generalization error, generalization gap.
 * first, we consider these condition as follow.
 * the underfitting will happend when the capacity of the model is low. just like the linear regression.
 * the training error and generalization error will both be big. accompanied by the increase of the capacity of the model.
 * the trainging error and generalization error will be both reduce but the generalization gab will be increase.
 * it is overfitting regime.
 * 
 * the ideal model assuming that we have konwn the real probabilty distribution for the original data.
 * but because of the exists of the noise. the predict error based on the known real distribution can be also
 * named as bayes error.
 * 
 * the training error and generalization error will be changed as the size of the training data.
 * but the expect of the generalization will not be increased as the training data increase.
 * 
 * the optimal capacity of the model will be increased as the size of the training data increase.
 * 当训练集增加时，训练误差也随之增大。这是由于越大的数据集越难以拟合。但是同时会导致测试误差减小，因为关于训练数据的
 * 不正确的假设越来越少。
 * 最优容量点处的测试误差接近于贝叶斯误差。训练误差可以低于贝叶斯误差，因为训练算法有能力记住训练集中特定的样本。
 * 但是当训练集趋向于无穷大时候，任何固定容量的模型的训练误差都至少增至贝叶斯误差。
 * 当训练集大小增大时，最优容量也会随之增大，最优容量在达到足够捕捉模型复杂度之后就不再增长了。
 * 
 * 也就是说最优容量会随着训练集大小的增大而增大，意味着我们可以通过扩大样本规模来增大模型的最优容量。
 * 同时，训练集的增大会导致训练误差的增大，但同时会导致测试误差的减小。
 * 训练集趋向于无穷大时候，任何固定容量的模型的训练误差都至少增至贝叶斯误差，训练误差可以低于贝叶斯误差。
 * 
 * 没有一个机器学习算法总是比其他的要好。这意味着机器学习研究的目标不是找一个通用学习算法或是绝对最好的学习算法，
 * 而是理解什么样的分布于人工智能获取经验的真实世界相关，以及什么样的学习算法在我们关注的数据生成分布上效果最好。
 * 

 * then, until here we just have one method to adjust the degree of fitting for one model, what is change the capacity 
 * of the model by increasing or reducing the number of the polynomial for the regression model. it is the orginal
 * and necessary method. but we should have the other method to adjust the degree of fitting for one model based on 
 * the orginal method. then, we will learn the regularization.
 * then, we should start with the preference function what means to compared with the nonpreference functions, 
 * we will select the preference function unless the nonpreference function has the better effect.
 * for example, we can consider to add the weight decay into the cost fucntion for one model. just like the linear
 * regression model, we can add the regularization into the cost function of the model when you trained the model.
 * J(w) = MSEtrain + λL2， L2 is the weight decay what is the parameter w^2. why add it into the cost function.
 * if you add the regularization, minimize the cost function J will be as the balance between fitting the training
 * data and the trade-off the small weight norm. as the increaseing of the weight λ, the slope of fitting the training
 * data will be smaller.
 * 
 * why regularization item can prevent the over fitting?
 * first, we should know what time that over fitting will happen.
 * first, the model is complex that means the model has the big capacity. second, the training data set is 
 * small. the second factor is objective existance. so we usually prevent the over fitting by optimizing the
 * first factor.
 * then, how to reduce the complexity of the model?
 * first, you can reduce the number of the highest order term. second, you can reduce the numbers of the weight.
 * so the former means you can reduce the number of the highest order term of the model. the last means
 * you can reduce the number of the unknown parameter of the model that can also named as the weight.
 * these two method can both reduce the complexity of the model.
 * then, how to reduce the numbers of the weight?
 * you can add the regularization item. then, why the regularization can reduce the numbers of the weight?
 * the regularization involved L1 and L2 regularization. L1 means ||Wn||1, is equal to W1 + W2 + W3 + ... + Wn, 
 * L2 means ||Wn||2, is equal to W1^2 + W2^2 + W3^2 + ... + Wn^2.
 * then, we should known the mathematical meaning of the regularization. how to come the regularization?
 * we want to add the limit condition about the weight, so we should add the limit condition
 * based on the original loss function.
 * the original loss function is equal to L(W) = 1/n * Σ(f(xi) - yi)^2. this is the most simple
 * loss function that is based on the least square method.
 * then we add the limit condition that limit the value range of the unknown parameters.
 * L1 is W1 + W2 + W3 + ... + Wn ≤ m
 * L2 is W1^2 + W2^2 + W3^2 + ... + Wn^2 ≤ m
 * then, you can use lagrange multiplier method to calculate the loss function that 
 * consider the limit condition.
 * it is equal to L(W) = 1/n * Σ(f(xi) - yi)^2 + λ(L1 - m OR L2 - m)
 * and calculate the partial derivatives that σL(W)/σW and σL(W)/σλ
 * and make σL(W)/σW = 0 and σL(W)/σλ = 0. then you can calculate the W* and λ*.
 * notice, the m is a constant number, it can be ignored when you calculate the partial
 * derivatives of the L(W). so the regularization can be expressed used
 * L1 or L2. and notice,  the L1 and L2 is the different limit value range for the weight.
 * the L1 is the diamod limit condition and the L2 is the circle limit condition.
 * and the L1 is more sparse sex than L2, why? 
 * first, you should know what is the sparse sex. it means the weight w has the big probabilty to acheive zero value.
 * when the wi is zero, the model will drop the corresponding feature xi. so this is the feature selected function
 * about the L1. because the weight has the big probabilty to acheive zero value, so the L1 can implement the function
 * that feature selected for one image. in order to understand this problem. we should consider the loss function image first.
 * the solution space of the loss function is consists of multiple contour. why the L2 is more sparse sex that L2?
 * we can consider it based on the solution space shape of L1 and L2. the solution space shape of L1 is one diamod, 
 * and L2 is circle. and the four angle of the diamod all on the coordinate. and the intersection point of the limit 
 * condition and solution space of the loss function is more likely to be the intersection point for the diamod limit
 * condition, and the y value is zero if the intersection point dropped on the coordinate. so the weight is more likely
 * to be zero if the limit condition shape is diamod. that means the L1 has the more sparse sex that L2, because the 
 * limit condition shape of L2 is circle, and circle has the smaller probability that dropped on the coordinate, so 
 * it will has the smaller sparse sex.
 * 
 * so it means if we have limited the range value of Wi, just like used L1 and L2, we can make more weight value to zero
 * that means we can drop some weight parameters so that we can reduce the complexity of the model based on reduce
 * the number of the weight parameters. and it is worth to mentioned that L2 is more sparse sex than L2. that means
 * L1 has the better efficient than L2 to reduce the complexity of the model. because it will make more zero value for
 * the weight paramters.
 * 
 * then, we can also consider this problem based on the probability distribution.
 * L1 is equal to the laplace distribution assuming. and L2 is euqal to the normal distribution assuming.
 * how to deep understanding it?
 * we can start from the bayesian maximum a posteriori probabilty estimation.
 * then, we should compare the bayesian estimation and the maximum likelihood estimation what is equal to 
 * P(D|θ), D is the sample set from x1, x2, ... xn. P(D|θ) = P(x1|θ) * P(x2|θ) * P(x3|θ) * ... * P(xn|θ)
 * the maximun likelihood estimation is the prior probability. the bayesian maximum posteriori probability estimation
 * is the posteriori probability. P(θ|D) = [P(D|θ)*P(θ)]/P(D). then, we can calculate the max value of P(θ|D).
 * because the P(D) is the constant value, so the max value of P(θ|D) is equal to the max value of P(D|θ)*P(θ)
 * make the exponential of P(D|θ)*P(θ) is equal to logP(D|θ) + logP(θ). converted the max value problem to the minimum value
 * problem. minimum value of the -logP(D|θ) - logP(θ) = -Σn=1_n logP(xi|θ) - logP(θ). because the P(θ) is the prior 
 * probability, so we can assume the probabilty distribution as the normal distribution and laplace distribution.
 * laplace distribution is equal to 1/（2λ）*e^(-|θ|/λ) and the normal distribution is equal to 1/[(2π)^(1/2)*σ]*e^(-θ^2/[2*σ^(1/2)])
 * log(1/2λ*e^(-|θ|/λ)) = log(1/2λ) + log(e^(-|θ|/λ)) = log(1/2λ) + 1/λ(-|θ|)
 * -log(1/2λ*e^-|θ|) = -log(1/2λ) + 1/λ*|θ|, |θ| is the L1 regularization expression. so the laplance distribution assuming
 * is equal to the L1 regularization. similar, log(1/[(2π)^(1/2)*σ]*e^(-θ^2/[2*σ^(1/2)])) = log(1/[(2π)^(1/2)*σ]) + 1/(2σ^2) * Σθi^2
 * Σθi^2 is the L2 limit condition. so we L2 is equal to the normal distribution.
 * then we can analysis the image of laplance distribution and normal distribution. the former is more sharp than the last.
 * so the value probability more likely to be zero than normal distribution. because the normal distribution is more
 * gentle than laplance distribution.
 * 
 * we should deep learning the difference between prior probability and posteriori probability.
 * prior probability is equal to P(A) what means the probability of A event. P(B|A) means the probability of
 * the event A occurs the probability of event B were obeserved. it is the likelihood function what can calculated
 * based on the maximum likelihood estimation what is equal to π(i=1_n)P(Bi|A) = P(B1|A) * P(B2|A) * ... * P(Bn|A).
 * P(A|B) is posteriori probability what means to calculate the P(A|B) based on the prior probability P(A) and the
 * likelihood estimation P(B|A). it can also consider as calculate the latest probability of event by considering
 * the last event B. P(A|B) = P(B|A) * P(A) / P(B), P(A) and P(B) are all the prior probability. it is objective.
 * P(B|A) is calculated based on the likelihood estimation function(MLE).
 * the former are all to handle the overfitting by using the weight decay method. the generally method is to add
 * the regularization.
 * 
 * 
 * then, we will system to learn how to define the loss function based on the different scenery.
 * before that, we should learn the last steps about how to train one model used the trian dataset.
 * we can consider the super parameter and cross validation.
 * the value of super parameter are not by learning algorithm itself. just like the number of the polynomial.
 * it is the super parameter about the model capacity. but how to select the super parameter value? we can not
 * verify it based on the testing dataset. because it will always be super result if you selected the verify dataset 
 * from testing dataset. because the verify dataset is order to select the most optimal algorithm and super parameters.
 * the testing dataset is to verify the accuracy of the model. so the verify dataset and testing dataset should be independent.
 * so we should verify the super parameters based on the dataset that out of the testing dataset, generally, we can select it
 * based on the training dataset. because it is easy to get. how to select the verify dataset based on the training dataset?
 * we selected 80% data from the original data used for training. and 20% data used for verifying. they are independent
 * to the testing dataset. of course, we can use cross validation based on the simple method.
 * 
 * cross validation.
 * why cross validation? if the training dataset and verify dataset is fixed. just like 80% and 20%. the prediction result
 * based on the testing dataset will be problematic. it is because it will be problematic even if the error of the 
 * testing dataset is small. because if the sample numbers of training dataset is much larger than testing dataset,
 * it is normal phenomenon that the error of testing dataset is always small. so we can not judge to select the better
 * model by selecting the smaller error of the testing dataset. then, how to handle this problem? of course, it will not
 * be one problem if the number of testing dataset is large enough. so we just consider when the testing data is very small.
 * we can use the cross validation method. it is order to add the number of testing dataset on the basis of the existing
 * data.
 * K-cross validation. we can divide into k based on the original samples. ki is independent and don't repeat.
 * first iteration, we can select k1 as the testing dataset, and the other k2-kk as the training dataset.
 * second iteration, we can select k2 as the testing dataset, and k1, k3-kn as the training dataset. 
 * repeat util select kk as the testing dataset.
 * last, calculate the mean of all error k. then we will estimate the generalization error of the algorithm.
 * 
 * the another estimation method is point estimation, it is different from the parameter estimation.
 * the paramter estimation generally is to estimate the vector parameter. and the estimation result is one function. 
 * just like the linear regression, we always try to describe the relationship that x and y based on one function.
 * and the point estimation is the statistic estimation just like mean, variance and standard derivation. it just gave
 * one single estimation value. 
 * 
 * 
 * 
 * 
 * 
 * 
 * 
 * 
 * 
 *      
 * 
 * 
 * 
 *      
***********************************************************************/
