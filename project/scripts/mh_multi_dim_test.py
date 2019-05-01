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

def func(x, params=[20, 2, 0.5]):
    beta_0, beta_1, beta_2 = params
    return beta_0 + x*beta_1 + beta_2*x**2

x = np.linspace(0,10, num=50)

y_true = func(x)
y_obs = func(x) + stats.norm.rvs(scale=5, size=len(x))

# Initialize traces
alpha_trace = []
sigma_trace = []

# Set initial guess for alpha in the M-H Algorithm
alpha = [1, 1, 1]

# Number of dimensions
d = len(alpha)

# Set initial guess for variance for data
sigma_sq = 1

# Set initial predicted values for heat flux given initial guess in thermal conductivity
y_hat = func(x, params=alpha)

# Set initial tuning parameter for proposal distribution
epsilon_cov = np.eye(d)

# Initialize acceptance count for determining the acceptance probability
acceptance_count = 0

# Optimal acceptance probability
p_optimal = 0.45

samples = 4000
tune_every = 10 # samples
times_tune = 100 # number of times to tune
tune_total = tune_every*times_tune
# Begin sampling
i = 0
t = 0
while acceptance_count < samples+tune_total:
    # Sample from proposal distribution given var_epsilon
    epsilon = stats.multivariate_normal.rvs(cov=epsilon_cov)

    # Propose new value for alpha given epsilon
    alpha_star = alpha + epsilon
    
    # Predicted at proposed value
    y_hat_star = func(x, params=alpha_star)

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
        alpha_trace.append(alpha.copy())

    # Tune variance of proposal distribution
    if (acceptance_count+1) % tune_every == 0 and t < tune_total:
        # Calculates the current acceptance probability
        p_accept = acceptance_count/i

        # New epsilon_cov = 2.4^2 S_b / d
        S = np.var(alpha_trace[-tune_every:], axis=0)
        epsilon_cov = 2.4**2 * np.diag(S)/d
        t+=1

    # Perform Gibbs sampling step on \sigma^2
    # sigma_sq | data \sim IG(N/2, \frac{1}{2} \sum_{i=1}^N (q_{inc,i} - \hat{q}_{inc,i})^2)
    sigma_sq = stats.invgamma.rvs(len(y_obs)/2, scale=(0.5*((y_obs - y_hat)**2).sum()))

    # Append traces
    sigma_trace.append(sigma_sq.copy())
    i += 1

print('acceptance: {:2.4f}'.format(acceptance_count/i))
burn = 1000

alpha_trace = alpha_trace[burn:]
sigma_trace = sigma_trace[burn:]
alpha_hat = np.mean(alpha_trace, axis=0)

print(alpha_hat)

# 95% credible interval
y_bar = func(x, params=alpha_hat)
y_bar_lower = y_bar - np.sqrt(np.mean(sigma_trace))*1.96
y_bar_upper = y_bar + np.sqrt(np.mean(sigma_trace))*1.96

plt.figure()
plt.plot(alpha_trace)
plt.show()

plt.figure()
plt.hist(sigma_trace)
plt.show()

plt.figure()
plt.plot(x, y_obs, '.k', label='Data')
plt.plot(x, y_bar, label='Predicted')
plt.plot(x, y_true, '--k', label='True')
plt.fill_between(x=x, y1=y_bar_lower, y2=y_bar_upper, color='grey')
plt.legend(loc=0)
plt.show()