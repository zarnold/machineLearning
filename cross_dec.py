import pandas as pd
import numpy as np
import pylab as plt

## Create a Fake dataset of workers
n_sample = 50000

# Age of employed
# Beta distrib of course
age = 14+80*np.random.beta(2,5,50000)
plt.hist(age,bins=200)
plt.show()

# Sexe
sex = np.random.choice([0,1],n_sample)

# diplome (bac + x )
grade = 1+np.random.exponential(1,n_sample).astype('int')
plt.hist(grade)
plt.show()

## Now, wages
# Let s say something like dat
s = map(lambda g: 10000+np.random.normal(10000*g,5000),grade)

