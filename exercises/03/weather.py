from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import sys
from mpl_toolkits import mplot3d
sys.path.append('../../scripts/')
from gaussian_process import gaussian_process
from gaussian_process import covariance_functions
import pandas as pd
from matplotlib.mlab import griddata

# Import data
data = pd.read_csv('../../data/weather.csv', delimiter=',')

pressure = data['pressure']
temperature = data['temperature']

longitude = data['lon']
latitude = data['lat']


lon = np.linspace(longitude.min(), longitude.max(), num=50)
lat = np.linspace(latitude.min(), latitude.max(), num=50)

## Linear interp of data
# pi = griddata(longitude, latitude, pressure, lon, lat, interp='linear')
# ti = griddata(longitude, latitude, temperature, lon, lat, interp='linear')

X = np.transpose(np.array((longitude, latitude)))
x_star = np.transpose(np.array((lon, lat)))

hyperparams = 10, 10, 10**-6

# P = gaussian_process(X, hyperparams, y=pressure, x_star=x_star)
# P_star, P_var = P.smoother()

plt.figure()
plt.plot(lon, lat)
plt.show()

# T = gaussian_process(X, hyperparams, y=temperature, x_star=x_star)
# T_star, T_var = T.smoother()

# plt.figure()
# plt.plot(latitude, pressure, '.k')
# plt.plot(lat, P_star)
# plt.show()

# plt.figure()
# plt.plot(latitude, temperature, '.k')
# plt.plot(lat, T_star)
# plt.show()

# # Pressure
# fig = plt.figure()
# # CS = plt.contour(lon, lat, p_star, 10, linewidths=0.5, colors='k')
# # CS = plt.contourf(lon, lat, P_star, 100, cmap='jet', vmax=pressure.max(), vmin=pressure.min())
# ax = plt.axes(projection='3d')
# ax.contour3D(lon, lat, P_star, 50)
# # plt.plot(longitude, latitude, '.k')
# plt.colorbar()
# plt.show()

# ## Temperature
# plt.figure()
# CS = plt.contour(lon, lat, ti, 10, linewidths=0.5, colors='k')
# CS = plt.contourf(lon, lat, ti, 100, cmap='jet', vmax=temperature.max(), vmin=temperature.min())
# plt.plot(longitude, latitude, '.k')
# plt.colorbar()
# plt.show()

# # Parse data
# X = data['temp']
# Y = data['gasbill']/data['billingdays']

# # Sort X data for plotting purposes
# x_star = X.drop_duplicates().sort_values().values

# # Set hyperparameters
# b = 20
# tau_1_squared = 10
# tau_2_squared = 10**-6

# # Pack hyperparameters for passing to model
# hyperparams = b, tau_1_squared, tau_2_squared

# # Create a guassian process object from data and prediction vector
# GP = gaussian_process(X, hyperparams, y=Y, x_star=x_star, cov='squared_exponential')

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