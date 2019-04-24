from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append('../../scripts/')
# from samplers import Trace
import scipy.stats as stats
from numpy.linalg import inv

burn=2000

beta_trace = np.load('traces/hierarchical_probit/beta_trace.npy')
mu_trace = np.load('traces/hierarchical_probit/mu_trace.npy')
Z_trace = np.load('traces/hierarchical_probit/Z_trace.npy')

mu_trace_mean = np.mean(mu_trace[burn:], axis=0)
beta_trace_mean = np.mean(beta_trace[burn:], axis=0)

print(Z_trace[burn:].shape)
print(Z_trace[burn:].shape)



# print(beta_trace_mean)

# plt.figure()
# plt.plot(mu_trace[burn:])
# plt.show()

plt.figure()
plt.plot(beta_trace[burn:,0])
plt.show()

# plt.figure()
# plt.plot(Z_trace[burn:])
# plt.show()