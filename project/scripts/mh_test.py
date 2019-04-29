#!/packages/python/anaconda3/bin python

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os, sys, time
import pandas as pd
from dft_esm import energy_storage
from scipy.signal import savgol_filter
from scipy.optimize import minimize
import scipy.stats as stats

np.random.seed(3)

def func(x, exponent=3):
    return x**exponent

x = np.linspace(0,10, num=50)

y_obs = func(x) + stats.norm.rvs(scale=50, size=len(x))

# Initialize traces
alpha_trace = []
sigma_trace = []

alpha_accepted = []
alpha_rejected = []

# Set initial guess for alpha in the M-H Algorithm
alpha = 1

# Set initial predicted values for heat flux given initial guess in thermal conductivity
y_hat = func(x, exponent=alpha)

# Set initial tuning parameter for proposal distribution
var_epsilon = 10

# Set initial guess for variance for data
sigma_sq = 1

# Initialize acceptance cound for determining the acceptance probability
acceptance_count = 0

# Optimal acceptance probability
p_optimal = 0.45


samples = 20000
# Begin sampling
for i in range(samples):
    # Sample from proposal distribution given var_epsilon
    epsilon = stats.norm.rvs(scale=var_epsilon)

    # Propose new value for alpha given epsilon
    alpha_star = alpha + epsilon
    
    # Predicted heat flux at proposed value
    y_hat_star = func(x, exponent=alpha_star)

    # Log ratio of posteriors, to make computation tractable
    log_beta = -(1/(2*sigma_sq))*(((y_obs - y_hat_star)**2).sum() - ((y_obs - y_hat)**2).sum())

    # Ratio of posteriors, \beta = \frac{p(\alpha^{star}|data)}{p(\alpha | data)}
    beta = np.exp(log_beta)
    
    # Determine acceptance of proposed value
    if np.random.uniform() < np.minimum(1, beta):
        # Set proposed values
        alpha = alpha_star
        y_hat = y_hat_star
        # Iterate acceptance count
        acceptance_count += 1
        alpha_accepted.append(alpha)
    else:
        alpha = alpha
        y_hat = y_hat
        alpha_rejected.append(alpha)

    # Tune variance of proposal distribution every 100 steps
    if (i+1) % 100 == 0:
        # Calculates the current acceptance probability
        p_accept = acceptance_count/i

        # New var_epsilon = var_epsilon \frac{\Phi^{-1}(p_{opt}/2)}{\Phi^{-1}(p_{cur}/2)}
        var_epsilon = var_epsilon * (stats.norm.ppf(p_optimal/2)/stats.norm.ppf(p_accept/2))

    # Perform Gibbs sampling step on \sigma^2
    # sigma_sq | data \sim IG(N/2, \frac{1}{2} \sum_{i=1}^N (q_{inc,i} - \hat{q}_{inc,i})^2)
    sigma_sq = stats.invgamma.rvs(len(y_obs)/2, 1/(0.5*((y_obs - y_hat)**2).sum()))

    # Append traces
    alpha_trace.append(alpha)
    sigma_trace.append(sigma_sq.copy())

print(var_epsilon, p_optimal, p_accept)

burn = 1000

alpha_trace = alpha_trace[burn:]
sigma_trace = sigma_trace[burn:]
alpha_hat = np.mean(alpha_trace)

print(alpha_hat)
print(np.mean(sigma_trace))

plt.figure()
plt.plot(alpha_accepted[burn:])
plt.show()

# plt.figure()
# plt.plot(alpha_accepted, '.b')
# plt.plot(alpha_rejected, 'xr')
# plt.show()

plt.figure()
plt.plot(sigma_trace)
plt.show()

plt.figure()
# for i in range(len(alpha_trace)):
#     plt.plot(x, func(x, exponent=alpha_trace[i]), color='grey')
plt.plot(x, y_obs, '.k', label='Data')
plt.plot(x, func(x, exponent=alpha_hat), label='Predicted')
plt.plot(x, func(x), '--k', label='True')
plt.legend(loc=0)
plt.show()