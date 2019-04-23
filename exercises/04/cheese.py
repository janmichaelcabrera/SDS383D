from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append('../../scripts/')
from samplers import Trace
import scipy.stats as stats
from numpy.linalg import inv

np.random.seed(3)

# Import data
df = pd.read_csv('../../data/cheese.csv', delimiter=',')

# Create an array of unique store names
stores = np.unique(df.store)

# Number of data rows
n = 5555
p = 4
d = 10

# Instantiate lists for storing sorted data
data = []
y = []
X = []

# Sort data by store
for s, store in enumerate(stores):
    data.append(df[df.store==store])

    # \bar{y}_i
    y.append(np.log(data[s].vol))
    data[s].vol = np.log(data[s].vol)
    data[s].price = np.log(data[s].price)

    # X_i^T
    X.append(np.array([np.ones(data[s].shape[0]), data[s].price, data[s].disp, data[s].price*data[s].disp]).T)

# Number of stores
s = 88

### Instantiate priors and traces
sigma_sq = 1

theta = np.zeros(p)
V = np.eye(p)*10**6

beta = np.zeros((s,p))
sigma = np.zeros(s)

beta_trace = []
sigma_trace = []
V_trace = []

#### Iterations
iterations = 5000
burn = 500
for j in range(iterations):
    # Store variable for inverse-Wishart
    B = 0
    # Iterate over stores to calculate \beta_i's
    for store in range(s):

        n = len(y[store])
        # beta_cov = [V^{-1} + X_i \frac{1}{\sigma_i^2} I_{n_i} X_i^T]^{-1}
        beta_cov = inv(inv(V) + X[store].T @ ((1/sigma_sq)*np.eye(n)) @ X[store])

        # beta_mean = beta_cov [V^{-1} \theta + \frac{1}{\sigma_i^2}X_i \bar{y}_i]
        beta_mean = beta_cov @ (inv(V) @ theta + (1/sigma_sq) * (X[store] @ y[store]))

        # Sample from multivariate normal for each store
        beta[store] = stats.multivariate_normal.rvs(mean=beta_mean, cov=beta_cov)

        # Shape and scale parameters for sigma
        a = n/2
        b = (y[store] - X[store] @ beta[store]) @ (y[store] - X[store] @ beta[store])/2

        # Sample from inverse-gamma distribution
        sigma[store] = stats.invgamma.rvs(a, 1/b)
    
        # Sum variable for inverse-Wishart
        B += np.tensordot((beta[store] - theta),(beta[store] - theta).T, axes=0)

    # Sample from inverse-Wishart distribution
    V = stats.invwishart.rvs(d+s, np.eye(p) + B)

    # Append samples to trace
    beta_trace.append(beta)
    sigma_trace.append(sigma)

# Reduce trace
beta_trace = np.asarray(beta_trace)
beta_trace_mean = np.mean(beta_trace[burn:], axis=0)

# Plot all stores together into one plot
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

plt.savefig('figures/cheese_all_plots.pdf')

# Plot selected stores
i = 10, 41, 55
for p, plots in enumerate(i):
    plt.figure()
    x_hat = X[plots][X[plots][:,1].argsort()]
    plt.plot(data[plots][data[plots].disp==0].price, data[plots][data[plots].disp==0].vol, '.k', linewidth=2)
    plt.plot(data[plots][data[plots].disp==1].price, data[plots][data[plots].disp==1].vol, '.r', linewidth=2)
    plt.plot(x_hat[:,1], beta_trace_mean[plots][0]+beta_trace_mean[plots][1]*x_hat[:,1], '-k', linewidth=2)
    plt.plot(x_hat[:,1], (beta_trace_mean[plots][0]+beta_trace_mean[plots][2])+(beta_trace_mean[plots][1]+beta_trace_mean[plots][3])*x_hat[:,1], '-r', linewidth=2)

    plt.savefig('figures/cheese_plots_'+str(plots)+'.pdf')