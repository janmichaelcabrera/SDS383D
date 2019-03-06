from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt
import sys
sys.path.append('../../scripts/')
from smoothers import kernel_smoother
import pandas as pd

data = pd.read_csv('../../data/utilities.csv', delimiter=',')

X = data['temp']
# Y = data['gasbill']/data['billingdays']
Y = data['gasbill']
# Y = data['average']

# sorted_data = data.sort_values('temp')
# sorted_X = sorted_data['temp']
# sorted_Y = sorted_data['gasbill']

x_star = X.drop_duplicates().sort_values().values

model = kernel_smoother(X, Y, X, h=5.5)
model.local_linear()
h = model.LOOCV_optimization()
# residuals = model.Residuals()

print(h)



# h = np.logspace(0, 1, num=30)
# loocv = np.zeros(len(h))

# # print(h)
# for i in range(len(h)):
#     loocv[i] = model.LOOCV(x=h[i])

# n=0

# for i in range(len(h)):
#     if loocv[i] == loocv.min():
#         n=i
# # print(loocv.min())

# print(loocv)
# plt.figure()
# plt.plot(h, loocv)
# plt.show()
# plt.close()

# h = h[n]
# print(h)
optimal = kernel_smoother(X, Y, x_star, h=h)
optimal.local_linear()
# model = kernel_smoother(X, Y, X, h=h)
# model.local_linear()

# residuals = model.Residuals()

# plt.figure()
# plt.plot(X, residuals, '.k')
# plt.show()
# plt.close()

# # # plt.figure()
# # # plt.plot(sorted_X, residuals, '.k')
# # # plt.show()
# # # plt.close()

plt.figure()
plt.plot(X, Y, '.k')
plt.plot(x_star, optimal.y_star)
plt.show()
plt.close()