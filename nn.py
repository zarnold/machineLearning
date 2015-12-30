import numpy as np
import pylab as plt
from sklearn import datasets

X,Y=datasets.make_moons(n_samples=300,noise=.08)
plt.scatter(X[:,0],X[:,1],c=Y,s=70)
plt.show()

