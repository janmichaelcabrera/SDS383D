from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append('../../scripts/')
# from samplers import Trace
import scipy.stats as stats
from numpy.linalg import inv

burn=000

beta_trace = np.load('traces/hierarchical_probit/beta_trace.npy')
mu_trace = np.load('traces/hierarchical_probit/mu_trace.npy')
# Z_trace = np.load('traces/hierarchical_probit/Z_trace.npy')

mu_trace_mean = np.mean(mu_trace[burn:], axis=0)
beta_trace_mean = np.mean(beta_trace[burn:], axis=0)

X = np.load('traces/hierarchical_probit/X.npy')
y = np.load('traces/hierarchical_probit/y.npy')

y_hat = y.copy()

n_states = y_hat.shape[0]

print((stats.norm.cdf(x=X[2] @ beta_trace_mean[2])>=0.5).astype(int))
pred = np.zeros(n_states)

for s in range(n_states):
    y_hat[s] = stats.norm.cdf(mu_trace_mean[s] + X[s] @ beta_trace_mean[s])
    p =((y_hat[s])>=0.5).astype(int)
    pred[s] = np.sum((p - y[s]) == 0)/y[s].shape[0]

print(pred)
print(pred.mean())

plt.figure()
plt.plot(mu_trace[burn:])
plt.show()

plt.figure()
for i in range(n_states):
    plt.plot(beta_trace[burn:,i])
plt.show()

# plt.figure()
# plt.plot(Z_trace[burn:])
# plt.show()

# y = 0

# loc = 2

# if y == 1:
#     a = 0
#     b = np.inf

# else:
#     a = -np.inf
#     b = 0

# x = stats.truncnorm.rvs(a - loc, b - loc, loc = loc, size=1000000)

# plt.figure()
# plt.hist(x, bins=50)
# plt.show()