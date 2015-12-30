#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pylab as plt
from sklearn import datasets

X,Y=datasets.make_moons(n_samples=500,noise=.2)

n_samples=len(X)
input_size=len(X[0])
output_size=len(set(Y))

print '%d samples with %d input size and %d output size'%(n_samples,input_size,output_size)

epsilon = 0.01 # learning rate for gradient descent
reg_lambda = 0.01 # regularization strength

def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Spectral)
    plt.show()

# given a neural network with its weights
# Compute output from input x
def compute_output(model,x):
  # simple neural with tanh as activation function
  W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
  z1 = x.dot(W1) + b1
  a1 = np.tanh(z1)
  z2 = a1.dot(W2) + b2
  exp_scores = np.exp(z2)
  #softmax
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
  # do not forget to update the model
  model = { 'W1': W1, 'a1':a1, 'b1': b1, 'W2': W2, 'b2': b2}
  return probs,model

# Loss function for softmax is entropy
def compute_cross_entropy(y,y_hat):
  lp = -np.log(y_hat[range(n_samples), y])
  data_loss = np.sum(lp)
  return data_loss

def compute_loss(model):
  W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
  p,m=compute_output(model,X)
  loss=compute_cross_entropy(Y,p)
  return 1./n_samples*loss

def predict_model(model,x):
  p,m=compute_output(model,x)
  return np.argmax(p,axis=1)

def backprop(output,model):
  W1, b1, W2, b2, a1 = model['W1'], model['b1'], model['W2'], model['b2'], model['a1']
  delta3 = output
  delta3[range(n_samples), Y] -= 1
  dW2 = (a1.T).dot(delta3)
  db2 = np.sum(delta3, axis=0, keepdims=True)
  delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
  dW1 = np.dot(X.T, delta2)
  db1 = np.sum(delta2, axis=0)
  # Gradient descent parameter update
  W1 += -epsilon * dW1
  b1 += -epsilon * db1
  W2 += -epsilon * dW2
  b2 += -epsilon * db2
  # Assign new parameters to the model
  model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
  return model       
  

def build_model(size_hidden=3,n_pass=40000):
  np.random.seed(0)
  W1 = np.random.randn(input_size, size_hidden) / np.sqrt(input_size)
  b1 = np.zeros((1, size_hidden))
  W2 = np.random.randn(size_hidden, output_size) / np.sqrt(size_hidden)
  b2 = np.zeros((1, output_size))
  model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
  for i in xrange(0,n_pass):
    p,model=compute_output(model,X)
    model=backprop(p,model)
    if i%1000==0:
      print "Loss after iteration %i: %f" %(i, compute_loss(model))
  return model
    
model=build_model(size_hidden=5)
plot_decision_boundary(lambda x: predict_model(model, x))

    



