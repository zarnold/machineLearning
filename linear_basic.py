# In machine Learning, this
# are standard library but you
# may use anothers
import pandas as pd
import numpy as np

# The linear model is one of the most fundamentals
# It  just is a fit of a line in a cloud of points
from sklearn import linear_model



# A very simple use case

#Generate Some X
U=np.random.randn(15000)
V=np.random.randn(15000)
W=np.random.randn(15000)
X=np.array([U,V,W])
X=X.T

# Build some Y
a=2
b=45
c=-7
Y=a*U+b*V+c*W

# Now see how the lienar regression is able to find 
# your build

lr = linear_model.LinearRegression()
lr.fit(X,Y)

print "Your coefs were : "
print "%d,%d and %d"%(a,b,c)
print "And the machine found : "
print lr.coef_

# now see how the algo is impacted by noise
Y=Y+np.random.uniform(-4,4,15000)
lr.fit(X,Y)

print "Your coefs were : "
print "%d,%d and %d"%(a,b,c)
print "plus some noise"
print "And the machine found : "
print lr.coef_


