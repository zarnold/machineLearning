#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pylab as plt
import sklearn.metrics as skm
from sklearn import datasets


class NeuralNetwork:

  def __init__(self,verbose=False, size_hidden=3,n_pass=20000, epsilon = 0.01):
    self.W1=[]
    self.b1=[]
    self.W2=[]
    self.b2=[]
    self.a1=[]
    self.n_samp=0
    self.in_s=0
    self.out_s=0
    self.size_hidden=size_hidden
    self.epsilon=epsilon
    self.n_pass=n_pass
    self.verbose=verbose

  def consoleLog(self,m):
    if self.verbose :
      print '######### '+str(m)

  def plot_decision_boundary(self,X,Y):
      # Set min and max values and give it some padding
      x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
      y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
      h = 0.01
      # Generate a grid of points with distance h between them
      xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
      # Predict the function value for the whole gid
      Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
      Z = Z.reshape(xx.shape)
      # Plot the contour and training examples
      plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
      plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Spectral)
      plt.show()

  # given a neural network with its weights
  # Compute output from input x
  def predict_proba(self,x):
    # simple neural with tanh as activation function
    z1 = x.dot(self.W1) + self.b1
    self.a1 = np.tanh(z1)
    z2 = self.a1.dot(self.W2) + self.b2
    exp_scores = np.exp(z2)
    #softmax
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probs

  def loss(self,X,Y):
    p=self.predict_proba(X)
    #the normalize does not seem to work :3
    #loss=skm.log_loss(Y,p,normalize=True)
    loss=skm.log_loss(Y,p)
    return loss/len(p)

  # just compute an ouput and take the most probable
  def predict(self,x):
    p=self.predict_proba(x)
    return np.argmax(p,axis=1)

  # backprop is computing
  # of the derivative in backward
  def backprop(self,X,Y,output):
    delta3 = output
    delta3[range(self.n_samp), Y] -= 1
    dW2 = (self.a1.T).dot(delta3)
    db2 = np.sum(delta3, axis=0, keepdims=True)
    # note that is because derivative of tanh is 1-tanh2
    # with activation function, you should recompute 
    delta2 = delta3.dot(self.W2.T) * (1 - np.power(self.a1, 2))
    dW1 = np.dot(X.T, delta2)
    db1 = np.sum(delta2, axis=0)
    # Gradient descent parameter update
    self.W1 += -self.epsilon * dW1
    self.b1 += -self.epsilon * db1
    self.W2 += -self.epsilon * dW2
    self.b2 += -self.epsilon * db2
    

  # train the network
  # On X to Y
  def fit(self,X,Y):
    # init params
    np.random.seed(0)
    self.n_samp=len(X)
    self.in_s=len(X[0])
    self.out_s=len(set(Y))
    self.W1 = np.random.randn(self.in_s, self.size_hidden) / np.sqrt(self.in_s)
    self.b1 = np.zeros((1, self.size_hidden))
    self.W2 = np.random.randn(self.size_hidden, self.out_s) / np.sqrt(self.size_hidden)
    self.b2 = np.zeros((1, self.out_s))
    print '%d samples with %d input size and %d output size'%(self.n_samp,self.in_s,self.out_s)
    for i in xrange(0,self.n_pass):
      p=self.predict_proba(X)
      self.backprop(X,Y,p)
      if i%1000==0 & self.verbose:
        p=self.loss(X,Y)
        print "Loss after iteration %i: %f" %(i, p)
    return True
      

