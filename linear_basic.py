# In machine Learning, this
# are standard library but you
# may use anothers
import pandas as pd
import numpy as np

# The linear model is one of the most fundamentals
# It  just is a fit of a line in a cloud of points
from sklearn import linear_model

print "First,  A very simple use case"

#Generate Some X
print "*"*50
U=np.random.randn(15000)
V=np.random.randn(15000)
W=np.random.randn(15000)
X=np.array([U,V,W])
X=X.T

# Build some Y
a=2
b=45
c=-7
d=-4
e=28
f=12

Y=a*U+b*V+c*W

# Now see how the lienar regression is able to find 
# your build

lr = linear_model.LinearRegression()
lr.fit(X,Y)

print "Your coefs were : "
print "%d,%d and %d"%(a,b,c)
print "And the machine found : "
print lr.coef_
print "Very good"

print "*"*50
print " now let's see how the algo is impacted by noise"
Y=Y+np.random.uniform(-4,4,15000)
lr.fit(X,Y)

print "Your coefs were : "
print "%d,%d and %d"%(a,b,c)
print "plus some noise"
print "And the machine found : "
print lr.coef_
print "Not that bad"

print "*"*50
print " OK. now let's talk about colinearity"
print "What does happen when some features ares strictly colinears ?"

# First : add some independant features
A=np.random.randn(15000)
B=np.random.randn(15000)
C=np.random.randn(15000)

X=np.array([U,V,W,A,B,C])
X=X.T

Y=a*U+b*V+c*W+d*A+e*B+f*C
lr.fit(X,Y)

print "Your coefs were : "
print "%d,%d,%d,%d,%d and %d"%(a,b,c,d,e,f)
print "And the machine found : "
print lr.coef_
print "it still works fine"


print "*"*20
print " But what if there are highly collinear features ?"
A=4*U
B=-60*U+22*V
C=10*W+U

X=np.array([U,V,W,A,B,C])
X=X.T


Y=a*U+b*V+c*W+d*A+e*B+f*C
lr.fit(X,Y)

print "Your coefs were : "
print "%d,%d,%d,%d,%d and %d"%(a,b,c,d,e,f)
print "And the machine found : "
print lr.coef_
print "It does not work fine anymore. It's pretty bad"
print "*"*50
