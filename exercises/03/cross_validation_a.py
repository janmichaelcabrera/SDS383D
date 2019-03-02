from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt
import sys
from scipy.optimize import minimize
sys.path.append('../../scripts/')
from smoothers import kernel_smoother


def func(x):
	return np.sin(x)
np.random.seed(3)
x = np.linspace(0, 2*np.pi, num=20)
y_training = func(x) + np.random.normal(scale=0.2, size=x.shape)
y_test = func(x) + np.random.normal(scale=0.2, size=x.shape)

H =[]
h = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
mse = np.zeros(len(h))

for i in range(len(h)):
	H.append(kernel_smoother(x, y_training, x, h=h[i]))
	H[i].predictor()
	mse[i] = H[i].MSE(y_test)

print(mse)