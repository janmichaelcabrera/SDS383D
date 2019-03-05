from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt
import sys
sys.path.append('../../scripts/')
from smoothers import kernel_smoother
import pandas as pd
import seaborn as sns

data = pd.read_csv('../../data/utilities.csv', delimiter=',')

X = data['temp']
# Y = data['gasbill']/data['billingdays']
Y = data['gasbill']
B = data['billingdays']
Y = Y/B

# sorted_data = data.sort_values('temp')
# sorted_X = sorted_data['temp']
# sorted_Y = sorted_data['gasbill']

x_star = X.drop_duplicates().sort_values().values

model = kernel_smoother(X, Y, X)
model.local_linear()
# h = model.LOOCV()
h = np.linspace(0.5, 10, num=20)
loocv = np.zeros(len(h))

# print(h)
for i in range(len(h)):
    loocv[i] = model.LOOCV(x=h[i])

print(loocv)
plt.figure()
plt.plot(h, loocv)
plt.show()
plt.close()

# h = 5.38666509
# print(h)
# optimal = kernel_smoother(X, Y, x_star, h=h)
# optimal.local_linear()

# # residuals = np.zeros(sorted_X.shape[0])

# # for i in range(len(residuals)):
# #     for j in range(len(x_star)):
# #         if x_star[j] == sorted_X[i]:
# #             residuals[i] = sorted_Y[i] - optimal.y_star[j]

# # plt.figure()
# # plt.plot(sorted_X, residuals, '.k')
# # plt.show()
# # plt.close()

# plt.figure()
# plt.plot(X, Y, '.k')
# plt.plot(x_star, optimal.y_star)
# plt.show()
# plt.close()