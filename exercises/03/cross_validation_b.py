from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt
import sys
from scipy.optimize import minimize
sys.path.append('../../scripts/')
from smoothers import kernel_smoother


def func(x, period = 1):
	return np.sin(x*2*np.pi*period)

np.random.seed(3)

x = np.linspace(0, 1, num=50)

noise = np.array([0.05, 0.25])
period = np.array([3, 0.5])

Y_training = []
Y_test = []
T = []

for n in range(len(noise)):
	for p in range(len(period)):
		Y_training.append(func(x, period=period[p]) + np.random.normal(scale=noise[n], size=x.shape))
		Y_test.append(func(x, period=period[p]) + np.random.normal(scale=noise[n], size=x.shape))
		T.append(func(x, period=period[p]))

for i in range(len(Y_training)):

	y_smooth = kernel_smoother(x, Y_training[i], x)
	y_smooth.predictor()
	y_smooth.optimize_h(Y_test[i])
	y_smooth.predictor()
	y_pred = y_smooth.y_star

	plt.figure()
	plt.plot(x, Y_training[i], '.k')
	plt.plot(x, Y_test[i], '.r')
	plt.plot(x, T[i], '--k')
	plt.plot(x, y_pred)
	plt.show()
	plt.close()