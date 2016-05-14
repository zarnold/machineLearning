import pandas as pd
import numpy as np
import pylab as plt

## Create a Fake dataset of workers

# Age
age = 14+80*np.random.beta(2,5,50000)
plt.hist(age,bins=200)
plt.show()

