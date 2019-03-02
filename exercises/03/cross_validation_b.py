from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt
import sys
from scipy.optimize import minimize
sys.path.append('../../scripts/')
from smoothers import kernel_smoother


def func(x, period = 1):
	return np.sin(x*2*np.pi*period)



# np.random.seed(3)
# x = np.linspace(0, 1, num=20)
# y_training = func(x) + np.random.normal(scale=0.2, size=x.shape)
# y_test = func(x) + np.random.normal(scale=0.2, size=x.shape)

# y_smooth = kernel_smoother(x, y_training, x)
# y_smooth.predictor()
# y_smooth.optimize_h(y_test)
# y_smooth.predictor()
# y_pred = y_smooth.y_star

# print(y_smooth.h)

# plt.figure()
# plt.plot(x, func(x), '--k')
# plt.plot(x, y_training, '.r')
# plt.plot(x, y_test, '.k')
# plt.plot(x, y_pred)
# plt.show()