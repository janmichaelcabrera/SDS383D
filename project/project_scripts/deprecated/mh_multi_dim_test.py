#!/packages/python/anaconda3/bin python

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import os, sys, time
from scipy.optimize import minimize
import scipy.stats as stats
sys.path.append('../../../scripts/')
from metropolis_hastings import Models

np.random.seed(3)

def func(x, params=[20, 2, 0.5]):
    beta_0, beta_1, beta_2 = params
    return beta_0 + x*beta_1 + x**beta_2

def func1(x, params=2):
    beta_0 = params
    return x**beta_0

x = np.linspace(0,10, num=50)

y_true = func(x)
y_obs = func(x) + stats.norm.rvs(scale=1, size=len(x))

# params = [1]
params = [1, 1, 1]

model = Models('test', func, x, y_obs, params)

print(model.mle())

# alpha_trace, sigma_trace = model.metropolis_random_walk(samples=1000)

# alpha_trace.save_trace()
# sigma_trace.save_trace()

# print('acceptance: {:2.4f}'.format(model.p_accept))

burn = 1000

alpha_trace = np.load('traces/test_alpha_trace.npy')[burn:]
sigma_trace = np.load('traces/test_sigma_trace.npy')[burn:]

alpha_hat = np.mean(alpha_trace, axis=0)

print(alpha_hat)

# 95% credible interval
y_bar = model.y_hat
y_bar_lower = y_bar - np.sqrt(np.mean(sigma_trace))*1.96
y_bar_upper = y_bar + np.sqrt(np.mean(sigma_trace))*1.96

# plt.figure()
# plt.plot(alpha_trace)
# plt.show()

# plt.figure()
# plt.hist(sigma_trace, bins=30, color='black')
# plt.show()

plt.figure()
for i in range(len(alpha_trace)):
    plt.plot(x, func(x, params=alpha_trace[i]), color='grey')
plt.plot(x, y_obs, '.k', label='Data')
plt.plot(x, y_bar, label='Predicted')
plt.plot(x, y_true, '--k', label='True')
# plt.fill_between(x=x, y1=y_bar_lower, y2=y_bar_upper, color='grey')
plt.legend(loc=0)
plt.show()