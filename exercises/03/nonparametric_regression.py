from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import sys
sys.path.append('../../scripts/')
from gaussian_process import gaussian_process
import pandas as pd

np.random.seed(3)

# Import data
data = pd.read_csv('../../data/utilities.csv', delimiter=',')

# Parse data
X = data['temp']
Y = np.log(data['gasbill']/data['billingdays'])

# Sort X data for plotting purposes
x_star = X.drop_duplicates().sort_values().values

# test value
# x_star = np.array([x_star[int(np.round(x_star.shape[0]/2))]])

b = 10
tau_1_squared = 5
tau_2_squared = 10**-6

hyperparams = b, tau_1_squared, tau_2_squared

GP = gaussian_process(X, hyperparams, y=Y, x_star=x_star)
sigma_hat = GP.approx_sigma()
y_star = GP.smoother(sigma_hat = sigma_hat)

plt.figure()
plt.plot(X, Y, '.k')
plt.plot(x_star, y_star, '-b')
plt.show()
plt.close()