from __future__ import division
import numpy as np 
import sys
sys.path.append('../../scripts/')
from smoothers import kernel_smoother

# Function for testing smoothers
def func(x):
	return np.sin(x)

# Initialize random seed
np.random.seed(3)

# Initialize x-vector
x = np.linspace(0, 2*np.pi, num=20)

# Training data of the form, y_i = f(x_i) + \epsilon_i
y_training = func(x) + np.random.normal(scale=0.2, size=x.shape)

# Testing data of the form, y_i = f(x_i) + \epsilon_i
y_test = func(x) + np.random.normal(scale=0.2, size=x.shape)

# Initialize list for storing kernel_smoother objects
H =[]

# Array of differing bandwidths to evaluate
h = np.array([0.2, 0.3, 0.4, 0.5, 0.6])

# Initialize MSE vector
mse = np.zeros(len(h))

# Iterates over bandwidth array and finds approximate MSE for each
for i in range(len(h)):
	H.append(kernel_smoother(x, y_training, x, h=h[i]))
	H[i].local_constant()
	mse[i] = H[i].MSE(y_test)

print(mse)