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
b = 64
tau_1_squared = 30
tau_2_squared = 10**-6

# Pack hyperparameters for passing to model
hyperparams = b, tau_1_squared, tau_2_squared

# Create a guassian process object from data and prediction vector
GP = gaussian_process(X, hyperparams, y=Y, x_star=x_star, cov='squared_exponential')

GP.optimize_lml()

var = GP.approx_var()

# Run the GP smoother with the approximated variance
y_star, variance = GP.smoother(variance = var)

# Calculate credible interval
upper = y_star + np.sqrt(variance)*1.96
lower = y_star - np.sqrt(variance)*1.96

# Plot fit and credible interval
plt.figure()
plt.title('b={:.2f}'.format(GP.hyperparams[0])+'; $\\tau_1^2$={:.2f}'.format(GP.hyperparams[1]))
plt.plot(X, Y, '.k')
plt.plot(x_star, y_star, '-b')
plt.plot(x_star, upper, '-g')
plt.plot(x_star, lower, '-g')
plt.xlabel('temperature ($^{\circ}$F)')
plt.ylabel('normalized gassbill')
# plt.show()
plt.savefig('figures/utilities_fit_gp_optimal_squared_exponential.pdf')
plt.close()