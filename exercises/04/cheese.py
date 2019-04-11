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
df = pd.read_csv('../../data/cheese.csv', delimiter=',')

# Create an array of unique store names
stores = np.unique(df.store)

# Number of data rows
n = 5555
p = 4
d = p + 1

# Instantiate lists for storing sorted data
data = []
y = []
X = []

# Sort data by store
for s, store in enumerate(stores):
    data.append(df[df.store==store])
    y.append(np.log(data[s].vol))
    data[s].vol = np.log(data[s].vol)
    data[s].price = np.log(data[s].price)
    X.append(np.array([np.ones(data[s].shape[0]), data[s].price, data[s].disp, data[s].price*data[s].disp]).T)

d = 10
s = 88
sigma_sq = 1

theta = np.zeros(4)
V = np.eye(4)*10**6

beta = np.zeros((s,4))
sigma = np.zeros(s)
# beta[0] = stats.multivariate_normal.rvs(mean=theta, cov=V)

beta_trace = []
sigma_trace = []
V_trace = []

#### Iterations
iterations = 100
burn = 10
for j in range(iterations):
    B = 0
    for store in range(s):
        n = len(y[store])

        beta_cov = inv(inv(V) + X[store].T @ ((1/sigma_sq)*np.eye(n)) @ X[store])
        beta_mean = beta_cov @ (inv(V) @ theta + (1/sigma_sq) * (X[store] @ y[store]))

        beta[store] = stats.multivariate_normal.rvs(mean=beta_mean, cov=beta_cov)

        a = n/2
        b = (y[store] - X[store] @ beta[store]) @ (y[store] - X[store] @ beta[store])/2

        sigma[store] = stats.invgamma.rvs(a, 1/b)
    
        B = B + np.tensordot((beta[store] - theta),(beta[store] - theta).T, axes=0)
    # print(B)
    V = stats.invwishart.rvs(d+s, np.eye(4) + B)
    # print(V)
    beta_trace.append(beta)
    sigma_trace.append(sigma)

beta_trace = np.asarray(beta_trace)
beta_trace_mean = np.mean(beta_trace[burn:], axis=0)


fig, ax = plt.subplots(11, 8, figsize=(10,12))
fig.subplots_adjust(hspace=0.6)
n = 0
for i in range(11):
    for j in range(8):
        x_hat = X[n][X[n][:,1].argsort()]
        ax[i,j].plot(data[n][data[n].disp==0].price, data[n][data[n].disp==0].vol, '.k', linewidth=0.1)
        ax[i,j].plot(data[n][data[n].disp==1].price, data[n][data[n].disp==1].vol, '.r', linewidth=0.1)
        ax[i,j].plot(x_hat[:,1], beta_trace_mean[n][0]+beta_trace_mean[n][1]*x_hat[:,1], '-k')
        ax[i,j].plot(x_hat[:,1], (beta_trace_mean[n][0]+beta_trace_mean[n][2])+(beta_trace_mean[n][1]+beta_trace_mean[n][3])*x_hat[:,1], '-r')
        n+=1

plt.show()