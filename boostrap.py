import pandas as pd
import numpy as np

from sklearn import linear_model



# A very simple use case

#Generate Some X
U=np.random.randn(15000)
V=np.random.randn(15000)
W=np.random.randn(15000)
X=np.array([U,V,W])
X=X.T

# Build some Y
Y=2*U+45*V-7*W

# Now see how the lienar regression is able to find 
# your build
lr = linear_model.LinearRegression()
lr.fit(X,Y)

print "Your coefs were : "
print lr.coef_

#congratulation. this was your first machine learning algorithm with python !
