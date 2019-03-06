from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt
import sys
sys.path.append('../../scripts/')
from smoothers import kernel_smoother
import pandas as pd

# Import data
data = pd.read_csv('../../data/utilities.csv', delimiter=',')

# Parse data
X = data['temp']
Y = np.log(data['gasbill']/data['billingdays'])

# Sort X data for plotting purposes
x_star = X.drop_duplicates().sort_values().values

# Instantiate smoother object (Default is a Gaussian kernel, order one)
model = kernel_smoother(X, Y, X, h=2)
# Run smoother on the data 
model.local_general()
# Optimize the bandwidth using LOOCV
h = model.LOOCV_optimization()
print(h)

# Instantiate second model with sorted x-data
optimal = kernel_smoother(X, Y, x_star, h=h)
# Run model
optimal.local_general()

# Establish intervel from model
interval = 1.96*model.sigma

# Sort the interval data for plotting purposes
I = pd.DataFrame(np.transpose([X.values, interval]))
I = I.sort_values(0).drop_duplicates()

# Establish upper and lower bounds for plotting
upper = optimal.y_star + I[1]
lower = optimal.y_star - I[1]

# Plot residuals
plt.figure()
plt.plot(X, model.residuals, '.k')
plt.xlabel('temperature ($^{\circ}$ F)')
plt.ylabel('log(residuals)')
# plt.show()
plt.savefig('figures/utilities_residuals_log.pdf')
plt.close()

# Plot fit and confidence interval
plt.figure()
plt.plot(X, Y, '.k')
plt.plot(x_star, optimal.y_star, '-b')
plt.plot(x_star, upper, '-g')
plt.plot(x_star, lower, '-g')
plt.xlabel('temperature ($^{\circ}$ F)')
plt.ylabel('log(normalized gassbill)')
# plt.show()
plt.savefig('figures/utilities_fit_log.pdf')
plt.close()