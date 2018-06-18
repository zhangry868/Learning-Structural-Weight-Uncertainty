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

'''
    Sample code to reproduce our results for the toy classifiction example.
    Our implementation is also based on SVGD code. Thanks for Qiang Liu & Dilin Wang making their code public.
    
    Copyright (c) 2018,  Ruiyi Zhang
    All rights reserved.
'''

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
    def __init__(self, X_train, y_train,  batch_size = 100, max_iter = 500, M = 20, n_hidden = 50, a0 = 1, b0 = 0.1, master_stepsize = 1e-3, auto_corr = 0.9):
        self.n_hidden = n_hidden
        self.d = X_train.shape[1]   # number of data, dimension 
        self.M = M
        
        num_vars = self.d * n_hidden + n_hidden * 2 + 4 + self.d + 2 * n_hidden + 1  # w1: d*n_hidden; b1: n_hidden; w2 = n_hidden; b2 = 1; 2 + 2 variances; 
                                                                                     #v_11: d; v_12: hidden; v_21:hidden v_22: 1
        self.theta = np.zeros([self.M, num_vars])  # particles, will be initialized later
        
        '''
            We keep the last 10% (maximum 500) of training data points for model developing
        '''
        size_dev = min(int(np.round(0.1 * X_train.shape[0])), 500)
        X_dev, y_dev = X_train[-size_dev:], y_train[-size_dev:]
        X_train, y_train = X_train[:-size_dev], y_train[:-size_dev]

        '''
            The data sets are normalized so that the input features and the targets have zero mean and unit variance
        '''
        self.std_X_train = np.std(X_train, 0)
        self.std_X_train[ self.std_X_train == 0 ] = 1
        self.mean_X_train = np.mean(X_train, 0)
        
        '''
            Theano symbolic variables
            Define the neural network here
        '''
        X = T.matrix('X') # Feature matrix
        y = T.vector('y') # labels
        
        w_1 = T.matrix('w_1') # weights between input layer and hidden layer
        v_11 = T.matrix('v_11')
        v_12 = T.matrix('v_12') # Transform Vector between input layer and hidden layer
        b_1 = T.vector('b_1') # bias vector of hidden layer
        
        w_2 = T.matrix('w_2') # weights between hidden layer and output layer
        v_21 = T.matrix('v_21')
        v_22 = T.matrix('v_22') # Transform Vector between output layer and hidden layer
        b_2 = T.scalar('b_2') # bias of output
        
        N = T.scalar('N') # number of observations
        
          # variances related parameters
        log_lambda = T.scalar('log_lambda')
        log_phi = T.scalar('log_phi')
        log_psi = T.scalar('log_psi')
        
        ###
        p_1 =  T.eye(self.d) - 2 * T.dot(v_11, T.transpose(v_11)) / T.sum(v_11**2)
        q_1 =  T.eye(self.n_hidden) - 2 * T.dot(v_12, T.transpose(v_12)) / T.sum(v_12**2)
        p_2 =  T.eye(self.n_hidden) - 2 * T.dot(v_21, T.transpose(v_21)) / T.sum(v_21**2)
        q_2 =  T.eye(1) - 2 * T.dot(v_22, T.transpose(v_22)) / T.sum(v_22**2)

        wf_1 = T.dot(T.dot(p_1, w_1), q_1)
        wf_2 = T.dot(T.dot(p_2, w_2), q_2)
        prediction = (T.nnet.nnet.sigmoid(T.dot(T.nnet.relu(T.dot(X, wf_1)+b_1), wf_2) + b_2)).flatten()
        #prediction = T.dot(T.nnet.relu(T.dot(X, wf_1)+b_1), wf_2) + b_2

        ''' define the log posterior distribution '''
        #log_lik_data = -T.sum(T.log(prediction)[T.arange(y.shape[0]), y])
        log_lik_data = - ( T.dot(y, T.log(prediction.flatten())) + T.dot((1 - y), (T.log(1 - prediction.flatten()))))

        #log_lik_data = -0.5 * X.shape[0] * (T.log(2*np.pi) - log_gamma) - (T.exp(log_gamma)/2) * T.sum(T.power(prediction.flatten() - y, 2))
        log_prior_data = (a0 - 1) * (log_lambda + log_phi + log_psi) - b0 * (T.exp(log_lambda) + T.exp(log_phi) + T.exp(log_psi)) + log_lambda + log_phi + log_psi 
        log_prior_w = -0.5 * (num_vars-2) * (T.log(2*np.pi)-log_lambda) - (T.exp(log_lambda)/2)*((w_1**2).sum() + (w_2**2).sum() + (b_1**2).sum() + b_2**2) - (T.exp(log_phi)/2) * ((v_11**2).sum() + (v_21**2).sum()) - (T.exp(log_psi)/2) * ((v_12**2).sum() + (v_22**2).sum())
        
        # sub-sampling mini-batches of data, where (X, y) is the batch data, and N is the number of whole observations
        log_posterior = (log_lik_data * N / X.shape[0] + log_prior_data + log_prior_w)

        dw_1, db_1, dw_2, db_2, dv_11, dv_12, dv_21, dv_22, d_log_lambda, d_log_phi, d_log_psi \
        = T.grad(log_posterior, [w_1, b_1, w_2, b_2, v_11, v_12, v_21, v_22, log_lambda, log_phi, log_psi])
        
        # automatic gradient
        logp_gradient = theano.function(
            inputs = [X, y, w_1, b_1, w_2, b_2, v_11, v_12, v_21, v_22, log_lambda, log_phi, log_psi, N],
            outputs = [dw_1, db_1, dw_2, db_2, dv_11, dv_12, dv_21, dv_22, d_log_lambda, d_log_phi, d_log_psi]
        )
        
        # prediction function
        self.nn_predict = theano.function(inputs = [X, w_1, b_1, w_2, b_2, v_11, v_12, v_21, v_22], outputs = prediction)
        
        '''
            Training with SVGD
        '''
        # normalization
        X_train, y_train = self.normalization(X_train, y_train)
        N0 = X_train.shape[0]  # number of observations
        
        ''' initializing all particles '''
        for i in range(self.M):
            w1, b1, w2, b2, v11, v12, v21, v22, loglambda, logphi, logpsi = self.init_weights(a0, b0)
            #############
            # use better initialization for gamma
            ridx = np.random.choice(range(X_train.shape[0]), \
                                           np.min([X_train.shape[0], 1000]), replace = False)
            y_hat = self.nn_predict(X_train[ridx,:], w1, b1, w2, b2, v11, v12, v21, v22)
            loggamma = -np.log(np.mean(np.power(y_hat - y_train[ridx], 2)))
            self.theta[i,:] = self.pack_weights(w1, b1, w2, b2, v11, v12, v21, v22, loglambda, logphi, logpsi)
            # w1_, b1_, w2_, b2_, v11_, v12_, v21_, v22_, loggamma_, loglambda_, logphi_, logpsi_ = self.unpack_weights(self.theta[i,:])
            # print(np.sum((v21_-v21)**2))
        grad_theta = np.zeros([self.M, num_vars])  # gradient 
        # adagrad with momentum
        fudge_factor = 1e-6
        historical_grad = 0
        for iter in range(max_iter):
            print(iter)
            print(w1)
            # sub-sampling
            batch = [ i % N0 for i in range(iter * batch_size, (iter + 1) * batch_size) ]
            for i in range(self.M):
                w1, b1, w2, b2, v11, v12, v21, v22, loglambda, logphi, logpsi = self.unpack_weights(self.theta[i,:])
                dw1, db1, dw2, db2, dv11, dv12, dv21, dv22, dloglambda, dlogphi, dlogpsi = logp_gradient(X_train[batch,:], y_train[batch], w1, b1, w2, b2, v11, v12, v21, v22, loglambda, logphi, logpsi, N0)
                grad_theta[i,:] = self.pack_weights(dw1, db1, dw2, db2, dv11, dv12, dv21, dv22, dloglambda, dlogphi, dlogpsi)
                #print('grad:')
                #print(grad_theta[i,:])
                #pdb.set_trace()
            # calculating the kernel matrix
            if(iter >= 278):
                pdb.set_trace()
            #if(iter > 100):
            #	pdb.set_trace()

            kxy, dxkxy = self.svgd_kernel(h=-1)  
            grad_theta2 = grad_theta
            grad_theta = (np.matmul(kxy, grad_theta) + dxkxy) / self.M   # \Phi(x)
            #print('w1:')
            #print(w1)
           
            # adagrad 
            if iter == 0:
                historical_grad = historical_grad + np.multiply(grad_theta, grad_theta)
            else:
                historical_grad = auto_corr * historical_grad + (1 - auto_corr) * np.multiply(grad_theta, grad_theta)
            adj_grad = np.divide(grad_theta, fudge_factor+np.sqrt(historical_grad))
            self.theta = self.theta + master_stepsize * adj_grad 

            if(iter % 10 == 0):
                X_test = np.dstack(np.meshgrid(np.arange(-6, 6, 0.1), np.arange(-6, 6, 0.1))).reshape((-1, 2))
                y_predict = self.predict(X_test)
                y_predict = np.array(y_predict).reshape((-1, 1))
                draw(y_predict, iter)
        '''
            Model selection by using a development set
        
        X_dev = self.normalization(X_dev) 
        for i in range(self.M):
            w1, b1, w2, b2, v11, v12, v21, v22, loglambda, logphi, logpsi = self.unpack_weights(self.theta[i, :])
            pred_y_dev = self.nn_predict(X_dev, w1, b1, w2, b2, v11, v12, v21, v22) * self.std_y_train + self.mean_y_train
            # likelihood
            def f_log_lik(loggamma): return np.sum(  np.log(np.sqrt(np.exp(loggamma)) /np.sqrt(2*np.pi) * np.exp( -1 * (np.power(pred_y_dev - y_dev, 2) / 2) * np.exp(loggamma) )) )
            # The higher probability is better    
            lik1 = f_log_lik(loggamma)
            # one heuristic setting
            loggamma = -np.log(np.mean(np.power(pred_y_dev - y_dev, 2)))
            lik2 = f_log_lik(loggamma)
            if lik2 > lik1:
                self.theta[i,-2] = loggamma  # update loggamma
		'''

    def normalization(self, X, y = None):
        X = (X - np.full(X.shape, self.mean_X_train)) / \
            np.full(X.shape, self.std_X_train)
            
        if y is not None:
            return (X, y)  
        else:
            return X
    
    '''
        Initialize all particles
    '''
    def init_weights(self, a0, b0):
        w1 = 1.0 / np.sqrt(self.d + 1) * np.random.randn(self.d, self.n_hidden)
        b1 = np.zeros((self.n_hidden,))
        w2 = 1.0 / np.sqrt(self.n_hidden + 1) * np.random.randn(self.n_hidden, 1)
        b2 = 0.

        v11 = 1.0 / np.sqrt(self.d + 1) * np.random.randn(self.d, 1)
        v12 = 1.0 / np.sqrt(self.n_hidden + 1) * np.random.randn(self.n_hidden, 1)
        v21 = 1.0 / np.sqrt(self.n_hidden + 1) * np.random.randn(self.n_hidden, 1)
        v22 = 1.0 / np.sqrt(1 + 1) * np.random.randn(1, 1)

        loglambda = np.log(np.random.gamma(a0, b0))
        logphi = np.log(np.random.gamma(a0, b0))
        logpsi = np.log(np.random.gamma(a0, b0))
        return (w1, b1, w2, b2, v11, v12, v21, v22, loglambda, logphi, logpsi)
    
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
    def pack_weights(self, w1, b1, w2, b2, v11, v12, v21, v22, loglambda, logphi, logpsi):
        params = np.concatenate([w1.flatten(), b1, w2.flatten(), [b2], v11.flatten(), v12.flatten(), v21.flatten(), v22.flatten(), [loglambda], [logphi], [logpsi]])
        return params
    
    '''
        Unpack all parameters in our model
    '''
    def unpack_weights(self, z):
        w = z
        w1 = np.reshape(w[:self.d*self.n_hidden], [self.d, self.n_hidden])
        b1 = w[self.d*self.n_hidden:(self.d+1)*self.n_hidden]
    
        w = w[(self.d+1)*self.n_hidden:]
        w2 = np.reshape(w[:self.n_hidden], [self.n_hidden, 1])
        b2 = w[self.n_hidden] ## or self.n_hidden + 1

        w = w[self.n_hidden + 1:]
        v11 = np.reshape(w[:self.d], [self.d, 1])
        v12 = np.reshape(w[self.d:self.d + self.n_hidden], [self.n_hidden, 1])
        v21 = np.reshape(w[self.d + self.n_hidden : self.d + 2 * self.n_hidden], [self.n_hidden, 1])
        v22 = np.reshape(w[-4], [1, 1])
        
        # the last two parameters are log variance
        loglambda, logphi, logpsi = w[-3], w[-2], w[-1]
        
        return (w1, b1, w2, b2, v11, v12, v21, v22, loglambda, logphi, logpsi)

    def predict(self, X_test):
        # normalization
        X_test = self.normalization(X_test)
        pred_y_test = np.zeros([self.M, X_test.shape[0]])
        '''
            Since we have M particles, we use a Bayesian view to calculate rmse and log-likelihood
        '''
        

        for i in range(self.M):
            w1, b1, w2, b2, v11, v12, v21, v22, loglambda, logphi, logpsi = self.unpack_weights(self.theta[i, :])
            pred_y_test[i, :] = self.nn_predict(X_test, w1, b1, w2, b2, v11, v12, v21, v22).flatten()# * self.std_y_train + self.mean_y_train).flatten()
        
        pred = np.mean(pred_y_test, axis=0)
        #pdb.set_trace()
        #pred = logistic.cdf(pred)
        return pred

np.random.seed(1)

# We create the train and test sets with 90% and 10% of the data

def draw(y, iter):
	plt.cla()
	plt.axis([-6,6,-6,6])
	X_train = np.asarray([[-2.16595599, -2.16161097],[-1.55935101, -1.629561],[-2.99977125, -2.5910955 ],[-2.39533485, -1.24376513],[-2.70648822, -2.94522481],[-2.81532281, -1.65906498],[-2.62747958, -2.1653904 ],[-2.30887855, -1.88262034],[-2.20646505, -2.71922612],[-1.92236653, -2.60379702],[ 2.60148914,  1.19669367],[ 2.93652315,  1.84221525],[ 1.62684836,  2.91577906],[ 2.38464523,  2.06633057],[ 2.7527783,   2.38375423],[ 2.78921333,  1.63103126],[ 1.17008842,  2.37300186],[ 1.07810957,  2.66925134],[ 1.33966084,  1.03657655], [ 2.75628501,  2.50028863]])
	x0 = X_train[:10,0]
	y0 = X_train[:10,1]
	x1 = X_train[10:,0]
	y1 = X_train[10:,1]

	plt.scatter(x0, y0, c='b')
	plt.scatter(x1, y1, c='r')

	X,Y=np.meshgrid(np.arange(-6, 6, 0.1), np.arange(-6, 6, 0.1))
	Z = y.reshape(X.shape)
	plt.contourf(X,Y,Z,alpha=0.5, cmap=cm.RdYlBu)

	plt.draw()
	plt.title(iter)
	plt.pause(0.1)
	pdb.set_trace()
	plt.savefig('SSVGD.pdf')
	plt.savefig('SSVGD.png')

if __name__ == '__main__':
	#warnings.simplefilter('error', RuntimeWarning)
	
	X_train = np.asarray([[-2.16595599, -2.16161097],[-1.55935101, -1.629561],[-2.99977125, -2.5910955 ],[-2.39533485, -1.24376513],[-2.70648822, -2.94522481],[-2.81532281, -1.65906498],[-2.62747958, -2.1653904 ],[-2.30887855, -1.88262034],[-2.20646505, -2.71922612],[-1.92236653, -2.60379702],[ 2.60148914,  1.19669367],[ 2.93652315,  1.84221525],[ 1.62684836,  2.91577906],[ 2.38464523,  2.06633057],[ 2.7527783,   2.38375423],[ 2.78921333,  1.63103126],[ 1.17008842,  2.37300186],[ 1.07810957,  2.66925134],[ 1.33966084,  1.03657655], [ 2.75628501,  2.50028863]])
	X_train = X_train.reshape((-1, 2))
	y_train = np.zeros((X_train.shape[0],), dtype='float32')
	y_train[10:] = 1.
	y_train[:10] = 0.

	X_test = np.dstack(np.meshgrid(np.arange(-6, 6, 0.1), np.arange(-6, 6, 0.1))).reshape((-1, 2))

	batch_size = 20
	n_hidden = 30
	max_iter = 2000

	print(y_train.shape)

	svgd = svgd_bayesnn(X_train, y_train, batch_size = batch_size, n_hidden = n_hidden, max_iter = max_iter, M = 20)

	plt.figure()
	y_predict = svgd.predict(X_test)
	y_predict = np.array(y_predict).reshape((-1, 1))
	#draw(y_predict, X_train)
