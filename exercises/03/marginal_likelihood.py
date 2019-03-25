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
import pandas as pd

# Import data
data = pd.read_csv('../../data/utilities.csv', delimiter=',')

# Parse data
X = data['temp']
Y = data['gasbill']/data['billingdays']

# Sort X data for plotting purposes
x_star = X.drop_duplicates().sort_values().values

# Set hyperparameters
b = 61
tau_1_squared = 39
tau_2_squared = 10**-6

# Pack hyperparameters for passing to model
hyperparams = b, tau_1_squared, tau_2_squared

# Create a guassian process object from data and prediction vector
GP = gaussian_process(X, hyperparams, y=Y, x_star=x_star, cov='matern_52')

# b = np.linspace(55, 65, num=4)
# tau_1_squared = np.linspace(35, 45, num=4)
b = np.arange(40, 90, 1)
tau_1_squared = np.arange(15, 90, 1)
Z = np.zeros((len(b), len(tau_1_squared)))

# variance = GP.approx_var()
variance = 1

for i in range(len(b)):
    for j in range(len(tau_1_squared)):
        hyperparams = b[i], tau_1_squared[j], tau_2_squared
        Z[i][j] = GP.log_marginal_likelihood(hyperparams, variance=variance)

b, tau_1_squared = np.meshgrid(b, tau_1_squared)

print(Z.max())
print(Z)

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# surf = ax.plot_surface(b, tau_1_squared, Z, cmap=cm.coolwarm, linewidth=0)
# fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.show()



# fig, ax = plt.subplots()
# CS = ax.contourf(b, tau_1_squared, Z, cmap='jet')
contours = plt.contour(b, tau_1_squared, Z, 3, colors='black')
plt.clabel(contours, inline=True, fontsize=8)
plt.show()

# plt.figure()
# levels = np.linspace(0.0001,0.1446,50)
# # levels = np.linspace(min(temp),max(temp),50)
# plt.contourf(b, tau_1_squared, Z, levels=levels, cmap = 'jet')
# plt.colorbar(format = "%.2f", label = '$(u_{rms}/U_{\infty})^2$')
# # quiver(x_temp[::m], y_temp[::n], T_ux[::n,::m], T_uy[::n,::m])#, scale=150, units = 'xy')
# # plt.axvline(l, c='k', lw=1)
# plt.show()

# print(GP.log_marginal_likelihood(hyperparams))

# var = 1

# # Run the GP smoother with the approximated variance
# y_star, variance = GP.smoother(variance = var)

# # Calculate credible interval
# upper = y_star + np.sqrt(variance)*1.96
# lower = y_star - np.sqrt(variance)*1.96

# # Plot fit and credible interval
# plt.figure()
# plt.plot(X, Y, '.k')
# plt.plot(x_star, y_star, '-b')
# plt.plot(x_star, upper, '-g')
# plt.plot(x_star, lower, '-g')
# plt.xlabel('temperature ($^{\circ}$F)')
# plt.ylabel('normalized gassbill')
# # plt.show()
# plt.savefig('figures/utilities_fit_gp_squared_exponential.pdf')
# plt.close()