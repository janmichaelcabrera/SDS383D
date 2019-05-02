#!/packages/python/anaconda3/bin python

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os, sys, time
import pandas as pd
from scipy.signal import savgol_filter
from scipy.optimize import minimize
import scipy.stats as stats

alpha_obs = stats.norm.rvs(size=1000)

print(np.var(alpha_obs))

# Set initial guess for alpha
alpha = 2

# Assume we don't know sigma_sq
sigma_sq = 5

# Set initial tuning parameter for proposal distribution
var_epsilon = 4

# Initialize acceptance count for determining the acceptance probability
acceptance_count = 0

# Optimal acceptance probability
p_optimal = 0.45

samples = 10000

alpha_accepted = []
sigma_trace = []

# Begin Sampling
for i in range(samples):
    # Sample from proposal distribution given var_epsilon
    epsilon = stats.norm.rvs(scale=var_epsilon)

    # Propose new value for alpha given epsilon
    alpha_star = alpha + epsilon

    log_beta = -(1/(2*sigma_sq))*(((alpha_obs - alpha_star)**2).sum() - ((alpha_obs - alpha)**2).sum())

    # Ratio of posteriors
    # beta = np.exp(-(1/(2*sigma_sq))*(alpha_obs - alpha_star)**2) / np.exp(-(1/(2*sigma_sq))*(alpha_obs - alpha)**2) 
    beta = np.exp(log_beta)

    # Determine acceptance of proposed value
    if np.random.uniform() < np.minimum(1, beta):
        # Set proposed values
        alpha = alpha_star

        acceptance_count += 1
        alpha_accepted.append(alpha)

    # Tune variance of proposal distribution every 100 steps
    if (i+1) % 100 == 0:
        p_accept = acceptance_count/i
        var_epsilon = var_epsilon * (stats.norm.ppf(p_optimal/2)/stats.norm.ppf(p_accept/2))

    sigma_sq = stats.invgamma.rvs(len(alpha_obs)/2, scale = (0.5*((alpha_obs - alpha)**2).sum()))

print(var_epsilon, p_accept)

print(np.mean(alpha_accepted), np.var(alpha_accepted))

plt.figure()
plt.hist(alpha_accepted, bins=30)
plt.show()