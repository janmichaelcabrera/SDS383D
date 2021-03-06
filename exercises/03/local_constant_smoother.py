from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt
import sys
sys.path.append('../../scripts/')
from smoothers import kernel_smoother

# Create noisy data, y_i = sin(x_i) + e_i
x = np.linspace(0, 2*np.pi, num = 20)
y = np.sin(x) + np.random.normal(scale=0.2, size=x.shape)

# Create vector for fitting purposes
x_star = np.linspace(0, 2*np.pi)

# Instantiate array of bandwidths
h = np.array([0.25, 1.0])

# Instantiate a list to append kernel_smoother objects
U = []
G = []

# Iterates through array of bandwidths and passes the feature vector, response vector, and bandwidth to kernel_smoother object
for i in range(len(h)):
    U.append(kernel_smoother(x, y, x_star, kernel='uniform', h=h[i], D=0))
    G.append(kernel_smoother(x, y, x_star, h=h[i], D=0))

# Plots data 
plt.figure()
plt.plot(x, y, '.k', label='Noisy response')
plt.plot(x, np.sin(x), '--k', label='True function')
# Iterates over the smoother objects and plots functions for the uniform and gaussian kernels
for i in range(len(h)):
    U[i].local_general()
    G[i].local_general()
    plt.plot(x_star, U[i].y_star, label='Uniform Kernel, h='+str(h[i]))
    plt.plot(x_star, G[i].y_star, label='Gaussian Kernel, h='+str(h[i]))
plt.legend(loc=0)
# plt.show()
plt.savefig('figures/kernel_smoother.pdf')
plt.close()