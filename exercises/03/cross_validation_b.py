from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt
import sys
sys.path.append('../../scripts/')
from smoothers import kernel_smoother

# Function for testing smoothers
def func(x, period = 1):
	return np.sin(x*2*np.pi*period)

# Initialize random seed
np.random.seed(3)

# Initialize x-vector
x = np.linspace(0, 1, num=100)

# Noise level array, low and high
noise = np.array([0.05, 0.25])
noise_label = ['Low', 'High']

# Period of function, high period (wiggly), low period (smooth)
period = np.array([3, 0.5])
period_label = ['Wiggly', 'Smooth']

### Initialize lists for storing the different responses
# Training responses
Y_training = []
# Testing responses
Y_test = []
# True response
T = []

# Iterates over noise and period arrays to assign data to lists
for n in range(len(noise)):
	for p in range(len(period)):
		Y_training.append(func(x, period=period[p]) + np.random.normal(scale=noise[n], size=x.shape))
		Y_test.append(func(x, period=period[p]) + np.random.normal(scale=noise[n], size=x.shape))
		T.append(func(x, period=period[p]))

# Generates plots 
i = 0
for n in range(len(noise)):
	for p in range(len(period)):

		# Instantiate kernel_smoother object
		y_smooth = kernel_smoother(x, Y_training[i], x)
		# Perform initial curve fit
		y_smooth.predictor()

		# Optimize bandwidth given the test data
		y_smooth.optimize_h(Y_test[i])

		# Perform curve fit with optimized bandwidth
		y_smooth.predictor()

		# Predicted values for opimized bandwidth
		y_pred = y_smooth.y_star
		h_star = np.abs(y_smooth.h)

		plt.figure()
		plt.title(noise_label[n]+' Noise, '+ period_label[p] + ': h={:.2f}'.format(h_star[0]))
		plt.plot(x, Y_training[i], '.k', label='Training Data')
		plt.plot(x, Y_test[i], '.r', label='Test Data')
		plt.plot(x, T[i], '--k', label='True Response')
		plt.plot(x, y_pred, '-b', label='Predicted Response')
		plt.ylim([-1.5, 1.5])
		plt.legend(loc=0)
		# plt.show()
		plt.savefig('figures/cross_validation_'+noise_label[n]+'_'+period_label[p]+'.pdf')
		plt.close()
		i += 1