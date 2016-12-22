#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pylab as plt
from sklearn import  linear_model
from sklearn.metrics import mean_squared_error
import math

plt.style.use('fivethirtyeight')

# We wanna model a ballistic throw
# from a canon whom angle is theta
# How far will the cannonball go ?

# Lets see how a strict polynomial model perform
# given only n points
# this is an illustration for trajectoires #2
# See https://soundcloud.com/dequaliter/trajectoires-2-decembre-2016


# First create a fake dataset 
# the Complete formula is v0².sin(2.theta) / g
# but we do not really care about constant 
# so remove g because we stay on Earth 
# and v0 because we alway use same powder quantity
# so d =  sin(2.theta)
# cf newton

theta = np.linspace(0,math.pi,100)
deg = 360*theta/(2*math.pi)
d=np.sin(2*theta)


# Plot nicely
plt.scatter(deg,d,label="Distance parcourue en fonction de l angle du fut")
plt.legend()
plt.show()


## If you really want it vs the initial speed
## and a complete model
## here it is
from mpl_toolkits.mplot3d import Axes3D
g=9.8

# Use two meshgrid, one for computing, one for displaying
v = np.linspace(0,300,100)
X,Y = np.meshgrid(v,theta)
u,v = np.meshgrid(v,deg)
Z = ( (X**2) * np.sin(Y) / g ) 

ax = plt.subplot(111, projection='3d')

ax.plot_wireframe(u,v, Z, rstride = 10, cstride=10)
ax.set_xlabel('Vitesse (m/s)', fontsize=12)
ax.set_ylabel('Angle (degree)', fontsize=12)
ax.set_zlabel('Distance (m)', fontsize=12)
plt.legend()
plt.show()



## Whatever, for purpose of 
## Readability, keep model simple


## So, how well a model perform with 1,2,3,..n points
## and 1,2,3,...n degree

## for plotting

theta = np.linspace(0,math.pi/2,100)
deg = 360*theta/(2*math.pi)
d=np.sin(2*theta)

lr=linear_model.LinearRegression()
n_trial=7
for n in np.arange(1,1+n_trial):
  idx = np.random.randint(1,100,n)
  print idx
  observation_x1 = theta[idx]
  observation_Y = d[idx]
  observation_X = observation_x1
  real_X = theta
  # F stand for freedom aka, how many dimension is the model enable to use
  # here we only use polynomial but could add log or sin or whatever
  # build the array of all polynomial features
  for f in np.arange(2,n_trial+1):
    observation_X = np.vstack([observation_X,observation_x1**f])
    real_X=np.vstack([real_X,theta**f])
  observation_X = observation_X.T
  real_X = real_X.T
  lr.fit(observation_X,observation_Y)
  y_pred = lr.predict(real_X) 
  rmse = mean_squared_error(d,y_pred)
  plt.subplot(1,n_trial,n)
  plt.scatter(deg[idx],observation_Y,color='green',s=150,label='Observations')
  plt.plot(deg,y_pred,color='red',label='model polynomial')
  plt.plot(deg,d,color='blue',label='modele newtonien')
  plt.title(str(n)+' observations.\nError = '+str(round(rmse,3)))
  plt.legend(prop={'size':10})

plt.show()



