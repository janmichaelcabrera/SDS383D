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

# Parse data into Y and X components
pressure = data['pressure'] # Y_1
temperature = data['temperature'] # Y_2
longitude = data['lon']
latitude = data['lat']
X = np.transpose(np.array((longitude, latitude))) # X

### Build up array to evaluate accross a grid of values
# Specifies bounds and number of points to divide the bounds into
num = 30
lon = np.linspace(longitude.min(), longitude.max(), num=num)
lat = np.linspace(latitude.min(), latitude.max(), num=num)

# Builds the grid from the previous lines (num x num Array)
lon_1, lat_1 = np.meshgrid(lon, lat)

# Converts the arrays into vectors (num^2 x 1)
X1 = np.ravel(lon_1)
X2 = np.ravel(lat_1)

# Builds an array from unraveled vectors (num^2 x 2)
x_star = np.array((X1, X2)).T

#### Pressure GP
# Initialize hyperparameters and fit a GP to observed data 
p_hyperparams = 0.23, 50127.18, 10**-6
P = gaussian_process(X, p_hyperparams, y=pressure, x_star=x_star)
P.optimize_lml()
P_star, P_var = P.smoother()

# Reshapes outputs of smoother for plotting purposes
P_star = P_star.reshape(lon_1.shape)
P_var = P_var.reshape(lon_1.shape)

#### Temperature GP
# Initialize hyperparameters and fit a GP to observed data 
t_hyperparams = 0.99, 6.07, 10**-6
T = gaussian_process(X, t_hyperparams, y=temperature, x_star=x_star)
T.optimize_lml()
T_star, T_var = T.smoother()

# Reshapes outputs of smoother for plotting purposes
T_star = T_star.reshape(lon_1.shape)
T_var = T_var.reshape(lon_1.shape)

levels = 50

#### Pressure
plt.figure()
plt.title('Pressure Difference: b={:.2f}'.format(P.hyperparams[0])+'; $\\tau_1^2$={:.2f}'.format(P.hyperparams[1]))
CS = plt.contourf(lon, lat, P_star, levels, cmap='jet') #, vmax=pressure.max(), vmin=pressure.min())
plt.colorbar(label='Pressure Difference ($Pa$)')
# plt.show()
plt.savefig('figures/weather_pressure.pdf')

#### Pressure variance
plt.figure()
plt.title('b={:.2f}'.format(P.hyperparams[0])+'; $\\tau_1^2$={:.2f}'.format(P.hyperparams[1]))
CS = plt.contourf(lon, lat, P_var, levels, cmap='jet', vmax=P_var.max(), vmin=P_var.min())
plt.plot(longitude, latitude, '.k')
plt.colorbar()
# plt.show()
plt.savefig('figures/weather_pressure_var.pdf')

#### Temperature
plt.figure()
plt.title('Temperature Difference: b={:.2f}'.format(T.hyperparams[0])+'; $\\tau_1^2$={:.2f}'.format(T.hyperparams[1]))
CS = plt.contourf(lon, lat, T_star, levels, cmap='jet') #, vmax=temperature.max(), vmin=temperature.min())
plt.colorbar(label='Temperature Difference ($^{\circ}$C)')
# plt.show()
plt.savefig('figures/weather_temperature.pdf')

#### Temperature variance
plt.figure()
plt.title('b={:.2f}'.format(T.hyperparams[0])+'; $\\tau_1^2$={:.2f}'.format(T.hyperparams[1]))
CS = plt.contourf(lon, lat, T_var, levels, cmap='jet', vmax=T_var.max(), vmin=T_var.min())
plt.plot(longitude, latitude, '.k')
plt.colorbar()
# plt.show()
plt.savefig('figures/weather_temperature_var.pdf')