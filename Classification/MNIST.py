import math
import warnings
from scipy.stats import logistic

import numpy as np
import pdb
import sys
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import theano.tensor as T
import theano
from scipy.spatial.distance import pdist, squareform
import random
import read_data
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import argparse

'''
    Sample code to reproduce our results for the Classification of MNIST.
    Our settings are almost the same as Blundell, Charles, et al. "Weight uncertainty in neural networks." (ICML15)
    Our implementation is based on SVGD code: https://github.com/DartML/Stein-Variational-Gradient-Descent

    p(y | W, X, \gamma) = \prod_i^N  N(y_i | f(x_i; W), \gamma^{-1})
    p(W | \lambda) = \prod_i N(w_i | 0, \lambda^{-1})
    p(\gamma) = Gamma(\gamma | a0, b0)
    p(\lambda) = Gamma(\lambda | a0, b0)

    The posterior distribution is as follows:
    p(W, \gamma, \lambda) = p(y | W, X, \gamma) p(W | \lambda) p(\gamma) p(\lambda)
    To avoid negative values of \gamma and \lambda, we update loggamma and loglambda instead.

    Copyright (c) 2016,  Ruiyi Zhang
    All rights reserved.
'''

rng = np.random.RandomState(1)
srng = RandomStreams(rng.randint(520))

def dropout(X, trng = srng, p = 0.25):
    if p != 0:
        retain_prob = 1 - p
        X = X / retain_prob * trng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
    return X

def batchnorm(X, g=None, b=None, u=None, s=None, a=1., e=1e-8):
    """
    batchnorm with support for not using scale and shift parameters
    as well as inference values (u and s) and partial batchnorm (via a)
    will detect and use convolutional or fully connected version
    """
    if X.ndim == 4:
        if u is not None and s is not None:
            b_u = u.dimshuffle('x', 0, 'x', 'x')
            b_s = s.dimshuffle('x', 0, 'x', 'x')
        else:
            b_u = T.mean(X, axis=[0, 2, 3]).dimshuffle('x', 0, 'x', 'x')
            b_s = T.mean(T.sqr(X - b_u), axis=[0, 2, 3]).dimshuffle('x', 0, 'x', 'x')
        if a != 1:
            b_u = (1. - a)*0. + a*b_u
            b_s = (1. - a)*1. + a*b_s
        X = (X - b_u) / T.sqrt(b_s + e)
        if g is not None and b is not None:
            X = X*g.dimshuffle('x', 0, 'x', 'x') + b.dimshuffle('x', 0, 'x', 'x')
    elif X.ndim == 2:
        if u is None and s is None:
            u = T.mean(X, axis=0)
            s = T.mean(T.sqr(X - u), axis=0)
        if a != 1:
            u = (1. - a)*0. + a*u
            s = (1. - a)*1. + a*s
        X = (X - u) / T.sqrt(s + e)
        if g is not None and b is not None:
            X = X*g + b
    else:
        raise NotImplementedError
    return X

class svgd_bayesnn:

    '''
        We define a one-hidden-layer-neural-network specifically. We leave extension of deep neural network as our future work.

        Input
            -- X_train: training dataset, features
            -- y_train: training labels
            -- batch_size: sub-sampling batch size
            -- max_iter: maximum iterations for the training procedure
            -- M: number of particles are used to fit the posterior distribution
            -- n_hidden: number of hidden units
            -- a0, b0: hyper-parameters of Gamma distribution
            -- master_stepsize, auto_corr: parameters of adgrad
    '''
    def __init__(self, X_train, y_train, X_test, y_test, num_class, batch_size = 100, max_iter = 1000, M = 20, n_hidden = 50, a0 = 1, b0 = 1, master_stepsize = 5e-4, auto_corr = 0.99):
        self.n_hidden = n_hidden
        self.d = X_train.shape[1]   # number of data, dimension
        self.M = M
        self.num_class = num_class
        self.batch_size = batch_size
        self.stepsize = master_stepsize
        self.epoch = int(max_iter * batch_size / 60000) # For Mnist
        num_vars = self.d * n_hidden + n_hidden + n_hidden * num_class + num_class + 2 + n_hidden * (n_hidden + 1) + 4 * self.n_hidden + self.num_class + self.d # w1: d*n_hidden; b1: n_hidden; w3 = n_hidden; b3 = 1; 2 variances
        self.theta = np.zeros([self.M, num_vars])  # particles, will be initialized later
        '''
            The data sets are normalized so that the input features and the targets have zero mean and unit variance
        '''
        self.std_X_train = np.std(X_train, 0)
        self.std_X_train[ self.std_X_train == 0 ] = 1
        self.mean_X_train = np.mean(X_train, 0)

        self.mean_y_train = np.mean(y_train)
        self.std_y_train = np.std(y_train)
        self.history = np.zeros([self.epoch, 2])

        self.learningRateBlock = int(self.epoch * 0.2 * 60000/self.batch_size)
        self.learningRateBlockDecay = 0.5
        '''
            Theano symbolic variables
            Define the neural network here
        '''
        X = T.matrix('X') # Feature matrix
        y = T.matrix('y') # labels

        w_1 = T.matrix('w_1') # weights between input layer and hidden layer
        v_11 = T.vector('v_11')
        v_12 = T.vector('v_12') # Transform Vector between input layer and hidden layer
        b_1 = T.vector('b_1') # bias vector of hidden layer

        w_2 = T.matrix('w_2') # weights between hidden layer and hidden layer
        v_21 = T.vector('v_21')
        v_22 = T.vector('v_22')
        b_2 = T.vector('b_2') # bias of output

        w_3 = T.matrix('w_3') # weights between hidden layer and output layer
        v_31 = T.vector('v_31')
        v_32 = T.vector('v_32') # Transform Vector between output layer and hidden layer
        b_3 = T.vector('b_3') # bias of output

        N = T.scalar('N') # number of observations

        p_1 =  T.eye(self.d) - 2 * T.outer(v_11, v_11) / T.sum(v_11**2)
        q_1 =  T.eye(self.n_hidden) - 2 * T.outer(v_12, v_12) / T.sum(v_12**2)

        p_2 =  T.eye(self.n_hidden) - 2 * T.outer(v_21, v_21) / T.sum(v_21**2)
        q_2 =  T.eye(self.n_hidden) - 2 * T.outer(v_22, v_22) / T.sum(v_22**2)

        p_3 =  T.eye(self.n_hidden) - 2 * T.outer(v_31, v_31) / T.sum(v_31**2)
        q_3 =  T.eye(self.num_class) - 2 * T.outer(v_32, v_32) / T.sum(v_32**2)

        wf_1 = T.dot(T.dot(p_1, w_1), q_1)
        wf_2 = T.dot(T.dot(p_2, w_2), q_2)
        wf_3 = T.dot(T.dot(p_3, w_3), q_3)

        log_gamma = T.scalar('log_gamma')   # variances related parameters
        log_lambda = T.scalar('log_lambda')

        ###
        #prediction = (T.nnet.nnet.softmax(T.dot( T.nnet.relu(T.dot(T.nnet.relu(T.dot(X, wf_1)+b_1), wf_2) + b_2) , wf_3) + b_3))
        prediction = (T.nnet.nnet.softmax(T.dot(T.nnet.relu( batchnorm(T.dot(T.nnet.relu( batchnorm(T.dot(X, wf_1)+b_1)), wf_2) + b_2) ), wf_3) + b_3))
        ''' define the log posterior distribution '''
        priorprec = T.log(b0/a0)
        log_lik_data = T.sum(T.sum(y * T.log(prediction)))
        log_prior_w = -0.5 * (num_vars-2) * (T.log(2*np.pi)-priorprec) - (T.exp(priorprec)/2)*((w_1**2).sum() + (w_2**2).sum() + (w_3**2).sum() + (b_1**2).sum() + (b_2**2).sum() + (b_3**2).sum()) + 1e-9*log_gamma + 1e-9*log_lambda

        # sub-sampling mini-batches of data, where (X, y) is the batch data, and N is the number of whole observations
        log_posterior = (log_lik_data * N / X.shape[0] + log_prior_w)
        dw_1, db_1, dw_2, db_2, dw_3, db_3, dv_11, dv_12, dv_21, dv_22, dv_31, dv_32, d_log_gamma, d_log_lambda = T.grad(log_posterior, [w_1, b_1, w_2, b_2, w_3, b_3, v_11, v_12, v_21, v_22, v_31, v_32, log_gamma, log_lambda])

        # automatic gradient
        logp_gradient = theano.function(
             inputs = [X, y, w_1, b_1, w_2, b_2, w_3, b_3, v_11, v_12, v_21, v_22, v_31, v_32, log_gamma, log_lambda, N],
             outputs = [dw_1, db_1, dw_2, db_2, dw_3, db_3, dv_11, dv_12, dv_21, dv_22, dv_31, dv_32, d_log_gamma, d_log_lambda]
        )

        # prediction function
        self.nn_predict = theano.function(inputs = [X, w_1, b_1, w_2, b_2, w_3, b_3, v_11, v_12, v_21, v_22, v_31, v_32], outputs = prediction)

        '''
            Training with SVGD
        '''
        # normalization
        X_train = self.normalization(X_train)
        N0 = X_train.shape[0]  # number of observations

        ''' initializing all particles '''
        for i in range(self.M):
            w1, b1, w2, b2, w3, b3, v11, v12, v21, v22, v31, v32, loggamma, loglambda = self.init_weights(a0, b0)
            # use better initialization for gamma
            ridx = np.random.choice(range(X_train.shape[0]), \
                                           np.min([X_train.shape[0], 1000]), replace = False)
            y_hat = self.nn_predict(X_train[ridx,:], w1, b1, w2, b2, w3, b3, v11, v12, v21, v22, v31, v32)
            loggamma = -np.log(np.mean(np.power(y_hat - y_train[ridx], 2)))
            self.theta[i,:] = self.pack_weights(w1, b1, w2, b2, w3, b3, v11, v12, v21, v22, v31, v32, loggamma, loglambda)
            #w1_, b1_, w2_, b2_, w3_, b3_, v11_, v12_, v21_, v22_, v31_, v32_, loggamma_, loglambda_ = self.unpack_weights(self.theta[i,:])
            #print(np.sum((v31_-v31)**2))
            #pdb.set_trace()


        grad_theta = np.zeros([self.M, num_vars])  # gradient
        # adagrad with momentum
        fudge_factor = 1e-5
        historical_grad = 0
        for iter in range(max_iter):
            # sub-sampling
            batch = [ i % N0 for i in range(iter * batch_size, (iter + 1) * batch_size) ]
            for i in range(self.M):
                w1, b1, w2, b2, w3, b3, v11, v12, v21, v22, v31, v32, loggamma, loglambda = self.unpack_weights(self.theta[i,:])
                dw1, db1, dw2, db2, dw3, db3, dv11, dv12, dv21, dv22, dv31, dv32, dloggamma, dloglambda = logp_gradient(X_train[batch,:], y_train[batch], w1, b1, w2, b2, w3, b3, v11, v12, v21, v22, v31, v32, loggamma, loglambda, N0)
                grad_theta[i,:] = self.pack_weights(dw1, db1, dw2, db2, dw3, db3, dv11, dv12, dv21, dv22, dv31, dv32, dloggamma, dloglambda)

            # calculating the kernel matrix
            if(self.M > 1):
                kxy, dxkxy = self.svgd_kernel(h=-1)
                grad_theta = (np.matmul(kxy, grad_theta) + dxkxy) / self.M   # \Phi(x)

            # adagrad
            if iter == 0:
                historical_grad = historical_grad + np.multiply(grad_theta, grad_theta)
            else:
                historical_grad = auto_corr * historical_grad + (1 - auto_corr) * np.multiply(grad_theta, grad_theta)
            adj_grad = np.divide(grad_theta, fudge_factor+np.sqrt(historical_grad))

            if((iter+1) % self.learningRateBlock == 0):
                master_stepsize = master_stepsize * self.learningRateBlockDecay
                print(master_stepsize)

            self.theta = self.theta + master_stepsize * adj_grad

            if(iter * self.batch_size % (X_train.shape[0]) == 0):
                epoch_index = int(iter * self.batch_size / X_train.shape[0])
                pred = self.predict(X_test)
                self.history[epoch_index, 0] = self.evluation(X_train, y_train, iter)
                self.history[epoch_index, 1] = sum(pred == y_test)*1.0/X_test.shape[0]
                print('Epoch ', iter * self.batch_size / X_train.shape[0], ' Iter:', iter,' Cost: ', self.history[epoch_index, 0])
                print('Precision: ', self.history[epoch_index, 1])
                if(epoch_index % 10 == 0):
                    np.savez('structure' + np.str(epoch_index) + '.npz', v11=v11,v12=v12,v21=v21,v22=v22,v31=v31,v32=v32)
                    self.savemodel()

    def normalization(self, X, y = None):
        X = (X - np.full(X.shape, self.mean_X_train)) / \
            np.full(X.shape, self.std_X_train)

        if y is not None:
            y = (y - self.mean_y_train) / self.std_y_train
            return (X, y)
        else:
            return X

    '''
        Initialize all particles
    '''
    def init_weights(self, a0, b0):
        w1 = 2.0 / np.sqrt(self.d + 1) * np.random.randn(self.d, self.n_hidden)
        b1 = np.zeros((self.n_hidden,))

        w2 = 2.0 / np.sqrt(self.n_hidden + 1) * np.random.randn(self.n_hidden, self.n_hidden)
        b2 = np.zeros((self.n_hidden,))

        w3 = 2.0 / np.sqrt(self.n_hidden + 1) * np.random.randn(self.n_hidden, self.num_class)
        b3 = np.zeros((self.num_class,))

        v11 = 1e-8 / np.sqrt(self.d + 1) * np.random.randn(self.d,)
        v12 = 1e-8 / np.sqrt(self.n_hidden + 1) * np.random.randn(self.n_hidden,)

        v21 = 1e-8 / np.sqrt(self.n_hidden + 1) * np.random.randn(self.n_hidden,)
        v22 = 1e-8 / np.sqrt(self.num_class + 1) * np.random.randn(self.n_hidden,)

        v31 = 1e-8 / np.sqrt(self.n_hidden + 1) * np.random.randn(self.n_hidden,)
        v32 = 1e-8 / np.sqrt(self.num_class + 1) * np.random.randn(self.num_class,)

        loglambda = np.log(np.random.gamma(a0, b0))
        loggamma = np.log(np.random.gamma(a0, b0))
        return (w1, b1, w2, b2, w3, b3, v11, v12, v21, v22, v31, v32, loggamma, loglambda)
    '''
        Calculate kernel matrix and its gradient: K, \nabla_x k
    '''
    def svgd_kernel(self, h = -1):
        sq_dist = pdist(self.theta)
        pairwise_dists = squareform(sq_dist)**2
        if h < 0: # if h < 0, using median trick
            h = np.median(pairwise_dists)
            h = np.sqrt(0.5 * h / np.log(self.theta.shape[0]+1))

        # compute the rbf kernel

        Kxy = np.exp( -pairwise_dists / h**2 / 2)

        dxkxy = -np.matmul(Kxy, self.theta)
        sumkxy = np.sum(Kxy, axis=1)
        for i in range(self.theta.shape[1]):
            dxkxy[:, i] = dxkxy[:,i] + np.multiply(self.theta[:,i],sumkxy)
        dxkxy = dxkxy / (h**2)
        return (Kxy, dxkxy)


    '''
        Pack all parameters in our model
    '''
    def pack_weights(self, w1, b1, w2, b2, w3, b3, v11, v12, v21, v22, v31, v32, loggamma, loglambda):
        params = np.concatenate([w1.flatten(), b1, w2.flatten(), b2, w3.flatten(), b3, v11.flatten(), v12, v21.flatten(), v22.flatten(), v31.flatten(), v32.flatten(), [loggamma], [loglambda]])
        #pdb.set_trace()
        return params

    '''
        Unpack all parameters in our model
    '''
    def unpack_weights(self, z):
        w = z
        w1 = np.reshape(w[:self.d*self.n_hidden], [self.d, self.n_hidden])
        b1 = w[self.d*self.n_hidden:(self.d+1)*self.n_hidden]

        w = w[(self.d+1)*self.n_hidden:]
        w2 = np.reshape(w[:self.n_hidden * self.n_hidden], [self.n_hidden, self.n_hidden])
        b2 = w[self.n_hidden * self.n_hidden : (self.n_hidden + 1) * self.n_hidden]

        w = w[(self.n_hidden+1)*self.n_hidden:]
        w3 = np.reshape(w[:self.n_hidden * self.num_class], [self.n_hidden, self.num_class])
        b3 = w[self.n_hidden * self.num_class : (self.n_hidden + 1) * self.num_class] ## or self.n_hidden + 1

        w = w[(self.n_hidden + 1) * self.num_class:]
        v11 = np.reshape(w[:self.d], [self.d, ])
        v12 = np.reshape(w[self.d:self.d + self.n_hidden], [self.n_hidden,])

        w = w[self.d + self.n_hidden:]
        v21 = np.reshape(w[:self.n_hidden], [self.n_hidden,])
        v22 = np.reshape(w[self.n_hidden : 2 * self.n_hidden], [self.n_hidden,])

        w = w[2 * self.n_hidden:]
        v31 = np.reshape(w[:self.n_hidden], [self.n_hidden, ])
        v32 = np.reshape(w[-(2 + self.num_class): -2], [self.num_class,])

        # the last two parameters are log variance
        loggamma, loglambda = w[-2], w[-1]

        return (w1, b1, w2, b2, w3, b3, v11, v12, v21, v22, v31, v32, loggamma, loglambda)


    def evluation(self, X_test, y_test, iter):
        pred_y_test = np.zeros([self.M, X_test.shape[0], self.num_class])
        '''
            Since we have M particles, we use a Bayesian view to calculate rmse and log-likelihood
        '''
        for i in range(self.M):
            w1, b1, w2, b2, w3, b3, v11, v12, v21, v22, v31, v32, log_gamma, loglambda = self.unpack_weights(self.theta[i, :])
            pred_y_test[i, :] = self.nn_predict(X_test, w1, b1, w2, b2, w3, b3, v11, v12, v21, v22, v31, v32)
        #pdb.set_trace()
        pred = np.mean(pred_y_test, axis=0)
        cost = np.mean(np.sum(y_test * np.log(pred), axis = 1), axis = 0)
        return cost

    def predict(self, X_test):
        # normalization
        X_test = self.normalization(X_test)
        pred_y_test = np.zeros([self.M, X_test.shape[0], self.num_class])
        '''
            Since we have M particles, we use a Bayesian view to calculate rmse and log-likelihood
        '''


        for i in range(self.M):
            w1, b1, w2, b2, w3, b3, v11, v12, v21, v22, v31, v32, log_gamma, loglambda = self.unpack_weights(self.theta[i, :])
            pred_y_test[i, :] = self.nn_predict(X_test, w1, b1, w2, b2, w3, b3, v11, v12, v21, v22, v31, v32)

        pred = np.mean(pred_y_test, axis=0)
        pred_class1 = np.zeros((X_test.shape[0],))
        for i in range(X_test.shape[0]):
            maxline = np.max(pred[i,:])
            for j in range(self.num_class):
                if (np.abs(pred[i, j] - maxline) < 1e-6):
                    pred_class1[i] = j
        pred_class2 = np.zeros((X_test.shape[0],self.num_class))
        for i in range(X_test.shape[0]):
            maxline = np.max(pred[i,:])
            for j in range(self.num_class):
                if (np.abs(pred[i, j] - maxline) < 1e-6):
                    pred_class2[i, j] = 1
        #pdb.set_trace()
        #pred = logistic.cdf(pred)
        return pred_class1
    def savemodel(self):
        np.savez('SSVGD' + str(self.n_hidden) +'_'+ str(self.M) + '_' + str(self.batch_size) + '_' + str(self.stepsize) + '.npz', history = self.history)

def savefinalmodel(self):
    np.savez('SSVGD_full' + str(self.n_hidden) +'_'+ str(self.M) + '_' + str(self.batch_size) + '_' + str(self.stepsize) + '.npz', history = self.history)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-batch", type = int, default = 200)
    parser.add_argument("-step", type = float, default = 5e-4)
    parser.add_argument("-epoch", type = int, default = 100)
    parser.add_argument("-hidden", type = int, default = 400)
    parser.add_argument("-b", type = float, default = 1)
    parser.add_argument("-M", type = int, default = 20)
    parser.add_argument("-adam", type = float, default = 0.99)

    global args
    args = parser.parse_args()

    X_train = read_data.extract_images('train-images-idx3-ubyte.gz')
    y_train = read_data.extract_labels('train-labels-idx1-ubyte.gz', True)

    X_test = read_data.extract_images('t10k-images-idx3-ubyte.gz')
    y_test = read_data.extract_labels('t10k-labels-idx1-ubyte.gz', False)

    X_test = np.array(X_test, ndmin = 2)
    Class_Num = 10

    batch_size = args.batch
    n_hidden = args.hidden
    stepsize = args.step

    max_iter = int(args.epoch * 60000 / batch_size)
    print('Setting: ')
    print('Batch_size: ', batch_size)
    print('Step size: ', stepsize)
    print('Particles: ', args.M)
    print('Hidden: ', n_hidden)
    print('Covariance: ', 1/args.b)
    print('RMSprop: ', args.adam)
    print('Epochs: ', args.epoch)

    svgd = svgd_bayesnn(X_train, y_train, X_test, y_test, master_stepsize = stepsize, batch_size = batch_size, n_hidden = n_hidden, max_iter = max_iter, num_class = Class_Num, b0 = args.b ,M = args.M, auto_corr = args.adam)
    savefinalmodel(svgd)
