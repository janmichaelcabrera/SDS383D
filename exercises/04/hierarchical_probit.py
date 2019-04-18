from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append('../../scripts/')
from samplers import Trace
import scipy.stats as stats
from numpy.linalg import inv

# np.random.seed(3)

# Import data
df = pd.read_csv('../../data/polls.csv', delimiter=',')

# Create an array of unique state names
states = np.unique(df.state)

# Instantiate lists for storing sorted data
data = []
y = []
X = []

# Sort data by state
for s, state in enumerate(states):
    data.append(df[df.state==state])
    y.append(np.nan_to_num(data[s].bush.values))

    # ['Bacc' 'HS' 'NoHS' 'SomeColl']
    ed = pd.get_dummies(data[s].edu, drop_first=False)

    # ['18to29' '30to44' '45to64' '65plus']
    age = pd.get_dummies(data[s].age, drop_first=False)

    x = np.column_stack([np.ones(len(y[s])), ed.values, age.values, data[s].female, data[s].black, data[s].weight])

    X.append(x)

# print(X[0])

Z = np.array([stats.halfnorm.rvs() for i in range(len(y[0]))])

print(Z)

# print(-stats.halfnorm.rvs(-1))
# print(stats.halfnorm.rvs(loc=[1,0]))
# help(stats.halfnorm)
# help(stats.halfnorm.rvs)