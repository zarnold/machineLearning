#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pylab as plt
from sklearn import datasets

import NeuralNetwork as nn
import sklearn.preprocessing as ppr

# Exemple of use
I,O=datasets.make_moons(n_samples=500,noise=.01)
X=I[:400]
Y=O[:400]
X_valid=I[400:]
Y_valid=O[400:]
clf=nn.NeuralNetwork(size_hidden=4,n_pass=30000)
clf.fit(X,Y)
print 'Validation score is %f'%(clf.loss(X_valid,Y_valid))

# now you can predict and cross valid, even 
# do hyperparameter search
# by using the score function


# Example with more output classes
I,O=datasets.make_classification(n_samples=5000,n_features=25,n_classes=5,n_informative=12,n_repeated=2)
hn=[]
sc=[]
for n in range(20):
  X=I[:4000]
  Y=O[:4000]
  X_valid=I[4000:]
  Y_valid=O[4000:]
  clf=nn.NeuralNetwork(size_hidden=n,n_pass=30000)
  clf.fit(X,Y)
  X_v=I[4000:]
  Y_v=O[4000:]
  Y_hat=clf.predict(X_v)
  s=map(lambda x: int(x),Y_hat==Y_v)
  score=np.mean(s)
  hn.append(n)
  sc.append(score)
  print "So for 5 classes and %d hidden neurones, we got : %f"%(n,score) 

plt.scatter(hn,sc)
plt.show()

## Example with the titanic dataset (does not work well )
# Filter
df_train=pd.read_csv('train.csv')
df=df_train[['Sex','Age','SibSp','Parch','Pclass','Fare','Embarked','Survived']].fillna(0)
df['Embarked_vec']=map(lambda x:list(set(df.Embarked)).index(x),df.Embarked)
df['F']=map(lambda x:list(set(df.Sex)).index(x),df.Sex)

# Balance
smallest=(len(df[df.Survived==1]),len(df[df.Survived==0]))[len(df[df.Survived==0])<len(df[df.Survived==1])]
t=df[df.Survived==1].sample(smallest)
f=df[df.Survived==0].sample(smallest)
df_balanced=pd.concat([t,f])

# Make dataset
X=np.array(df_balanced[['Age','SibSp','Parch','Pclass','Fare','F','Embarked_vec']])
Y=np.array(df_balanced.Survived)
X_n=ppr.normalize(X,norm='l2')

clf=nn.NeuralNetwork(size_hidden=6,n_pass=40000)
clf.fit(X_n,Y)
h=clf.predict(X_n)
s=1.*sum(map(lambda x,y: 1-abs(x-y),h,Y))/len(h)
print s

'''
sc=[]
for n in range(20):
  print '======= Pass %d ========='%n
  clf=nn.NeuralNetwork(size_hidden=n,n_pass=30000)
  clf.fit(X_n,Y)
  h=clf.predict(X_n)
  s=1.*sum(map(lambda x,y: 1-abs(x-y),h,Y))/len(h)
  print s
  sc.append(s)
'''

