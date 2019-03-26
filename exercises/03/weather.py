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
X = np.transpose(np.array((longitude, latitude)))


lon = np.linspace(longitude.min(), longitude.max(), num=10)
lat = np.linspace(latitude.min(), latitude.max(), num=10)

lon_1, lat_1 = np.meshgrid(lon, lat)

X1 = np.ravel(lon_1)
X2 = np.ravel(lat_1)

x_star = np.array((X1, X2)).T

p_hyperparams = 5, 51836, 10**-6
P = gaussian_process(X, p_hyperparams, y=pressure, x_star=x_star)
P_star, P_var = P.smoother()

P_star = P_star.reshape(lon_1.shape)

t_hyperparams = 0.8, 6.37, 10**-6
T = gaussian_process(X, t_hyperparams, y=temperature, x_star=x_star)
T_star, T_var = T.smoother()

T_star = T_star.reshape(lon_1.shape)

# # Pressure
plt.figure()
CS = plt.contourf(lon, lat, P_star, 50, cmap='jet', vmax=pressure.max(), vmin=pressure.min())
# plt.plot(longitude, latitude, '.k')
plt.colorbar()
plt.show()

## Temperature
plt.figure()
CS = plt.contourf(lon, lat, T_star, 100, cmap='jet', vmax=temperature.max(), vmin=temperature.min())
# plt.plot(longitude, latitude, '.k')
plt.colorbar()
plt.show()