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

    x = np.column_stack([np.ones(len(y[s]), dtype=np.int8), ed.values, age.values, data[s].female, data[s].black, data[s].weight])

    X.append(x)

# r = stats.truncnorm.rvs(-np.inf, np.inf, loc = 5, size=100000)
# print(X[0].shape)
beta = np.ones(X[0].shape[1])
beta_star = np.ones(X[0].shape[1])
B_star = np.eye(X[0].shape[1])*10**6

Z = np.zeros(y[0].shape[0])

beta_trace = []
Z_trace = []

iterations = 1000
burn = 0 

for p in range(iterations):
    for i in range(len(Z)):
        if y[0][i] == 1.0:
            a = 0
            b = np.inf
        else:
            a = -np.inf
            b = 0

        Z[i] = stats.truncnorm.rvs(a, b, loc=X[0][i,:] @ beta)

    beta_cov = inv(inv(B_star) + X[0].T @ X[0])
    beta_mean = beta_cov @ (inv(B_star) @ beta_star + X[0].T @ Z)

    beta = stats.multivariate_normal.rvs(mean=beta_mean, cov=beta_cov)
    Z_trace.append(Z)
    beta_trace.append(beta)

beta_trace = np.asarray(beta_trace)
beta_trace_mean = np.mean(beta_trace[burn:], axis=0)

plt.figure()
plt.plot(beta_trace[:,6])
plt.show()