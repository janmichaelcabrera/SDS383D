from __future__ import division
import numpy as np 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.stats import multivariate_normal
import sys
sys.path.append('../../scripts/')
from gaussian_process import gaussian_process
from gaussian_process import covariance_functions
import pandas as pd

# Import data
data = pd.read_csv('../../data/utilities.csv', delimiter=',')

# Parse data
X = data['temp']
Y = data['gasbill']/data['billingdays']

# Sort X data for plotting purposes
x_star = X.drop_duplicates().sort_values().values

# Set hyperparameters
b = 20
tau_1_squared = 10
tau_2_squared = 10**-6

# Pack hyperparameters for passing to model
hyperparams = b, tau_1_squared, tau_2_squared

# Create a guassian process object from data and prediction vector
GP = gaussian_process(X, hyperparams, y=Y, x_star=x_star, cov='squared_exponential')

# Create arrays for evaluating and saving the hyperparameters
b = np.linspace(50, 80, num=10)
tau_1_squared = np.linspace(30, 60, num=10)
Z = np.zeros((len(b), len(tau_1_squared)))

# Approximate the variance by fitting a GP once and evaluating the residual sum of squared errors
variance = GP.approx_var()

# Loops over each b and tau_1_squared and evaluates the log of the marginal likelihood
for i in range(len(b)):
    for j in range(len(tau_1_squared)):
        hyperparams = b[i], tau_1_squared[j], tau_2_squared
        Z[i][j] = GP.log_marginal_likelihood(hyperparams, variance=variance)

# Chooses the maximum value from the array of caclulated log marginal likelihoods
ind = np.unravel_index(np.argmax(Z, axis=None), Z.shape)

# Optimal values from the above
b_hat = b[ind[0]]
tau_1_hat = tau_1_squared[ind[1]]

# Prints the optimal values (these are evaluated on the dataset using a different script)
print(b_hat, tau_1_hat)

# For plotting purposes
b, tau_1_squared = np.meshgrid(b, tau_1_squared)

# Plots contour of log marginal likelihood vs b and tau_1_squared
plt.figure()
plt.contourf(b, tau_1_squared, Z, 100, cmap='jet')
plt.plot(b_hat, tau_1_hat, '.k', label='Optimal $b$ and $\\tau_1^2$')
plt.xlabel('b')
plt.ylabel('$\\tau_1^2$')
plt.colorbar()
plt.legend(loc=0)
# plt.show()
plt.savefig('figures/optimal_b_tau_squared_exponential.pdf')