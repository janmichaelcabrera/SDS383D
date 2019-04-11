from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append('../../scripts/')
from samplers import Trace
import scipy.stats as stats
from numpy.linalg import inv

# np.random.seed(3)

# Import data
df = pd.read_csv('../../data/cheese.csv', delimiter=',')

# Create an array of unique store names
stores = np.unique(df.store)

# Number of data rows
n = 5555
p = 4
d = p + 1

# Instantiate lists for storing sorted data
data = []
y = []
X = []

# Sort data by store
for s, store in enumerate(stores):
    data.append(df[df.store==store])
    y.append(np.log(data[s].vol))
    data[s].price = np.log(data[s].price)
    X.append(np.array([np.ones(data[s].shape[0]), data[s].price, data[s].disp, data[s].price*data[s].disp]).T)



sigma_sq = 1

# theta = np.zeros(4)
theta = np.array([-5, 12, -5, 12])
V = np.eye(4)*10**6
# V = np.random.rand(4,4)
# V = V @ V.T

beta = stats.multivariate_normal.rvs(mean=theta, cov=V)

n = len(y[0])

beta_trace = []
sigma_trace = []


### Iterations
iterations = 1000
burn = 0
for j in range(iterations):
    beta_cov = inv(inv(V) + X[0].T @ ((1/sigma_sq)*np.eye(n)) @ X[0])
    beta_mean = beta_cov @ (inv(V) @ theta + (1/sigma_sq) * (X[0] @ y[0]))

    beta = stats.multivariate_normal.rvs(mean=beta_mean, cov=beta_cov)
    beta_trace.append(beta)

    a = n/2
    b = (y[0] - X[0] @ beta) @ (y[0] - X[0] @ beta)/2

    sigma = stats.invgamma.rvs(a, 1/b)
    sigma_trace.append(sigma)

beta_trace = np.asarray(beta_trace)
beta_trace_mean = np.mean(beta_trace[burn:], axis=0)

print(beta_trace_mean)

x_hat = X[0][X[0][:,1].argsort()]

plt.figure()
plt.plot(data[0][data[0].disp==0].price, np.log(data[0][data[0].disp==0].vol), '.k')
plt.plot(data[0][data[0].disp==1].price, np.log(data[0][data[0].disp==1].vol), '.r')
plt.plot(x_hat[:,1], beta_mean[0]+beta_mean[1]*x_hat[:,1], '-k')
plt.plot(x_hat[:,1], (beta_mean[0]+beta_mean[2])+(beta_mean[1]+beta_mean[3])*x_hat[:,1], '-r')
plt.show()

# plt.figure()
# plt.plot(beta_trace[burn:,0], label='Beta 0')
# plt.plot(beta_trace[burn:,1], label='Beta 1')
# plt.plot(beta_trace[burn:,2], label='Beta 2')
# plt.plot(beta_trace[burn:,3], label='Beta 3')
# plt.legend(loc=0)
# plt.show()


# # Cashe computation for \sum_{i=1}^88 X_i^T X 
# sum_XX = np.zeros((4,4))
# for i in range(88):
#     sum_XX = sum_XX + X[i].T @ X[i]

# b = np.zeros((88,4))

# sigma_sq = 1

# Sigma = np.random.rand(4,4)
# Sigma = Sigma @ Sigma.T

# # Sigma = np.array(([1, 0.8], [0.8, 1]))

# mu_beta = np.zeros(4)
# # mu_beta = np.array([-5, 12, -5, 12])
# cov_beta = Sigma

# C = Sigma

# beta = stats.multivariate_normal.rvs(mean=mu_beta, cov = inv(cov_beta)).T

# ### Instantiate traces
# b_trace = []
# beta_trace = []
# sigma_sq_trace = []
# Sigma_trace = []

# ### Iterations start here
# iterations = 100
# burn = 10
# for j in range(iterations):
#     ## b_i | y
#     for i in range(88):
#         b_cov = inv(inv(Sigma) + (1/sigma_sq)*W[i].T @ W[i])
#         b_mean = (1/sigma_sq)*W[i] @ (y[i] - X[i] @ beta) @ b_cov
#         b[i] = stats.multivariate_normal.rvs(mean=b_mean, cov=b_cov)

#     b_trace.append(b)

#     ## beta | y
#     beta_cov = inv(inv(Sigma) + (1/sigma_sq)*sum_XX)

#     beta_xy_sum = np.zeros(4)

#     for i in range(88):
#         beta_xy_sum = beta_xy_sum + X[i] @ (y[i] - W[i] @ b[i])

#     beta_mean = (inv(cov_beta) @ mu_beta + (1/sigma_sq) * beta_xy_sum) @ beta_cov
#     beta = stats.multivariate_normal.rvs(mean=beta_mean, cov=beta_cov)

#     beta_trace.append(beta)

#     ## sigma_sq | y
#     sigma_xy_sum = 0

#     for i in range(88):
#         sigma_xy_sum = sigma_xy_sum + (y[i] - X[i] @ beta - W[i] @ b[i]).T @ (y[i] - X[i] @ beta - W[i] @ b[i])

#     sigma_sq = stats.invgamma.rvs(n/2, 2/sigma_xy_sum)

#     sigma_sq_trace.append(sigma_sq)

#     ## Sigma | b
#     Sigma_bb_sum = np.zeros((4,4))

#     for i in range(88):
#         Sigma_bb_sum = Sigma_bb_sum + np.tensordot(b[i], b[i].T, axes=0)

#     Sigma = stats.invwishart.rvs(df=d+88, scale=C+Sigma_bb_sum)
#     Sigma_trace.append(Sigma)

# beta_trace = np.asarray(beta_trace)
# beta_trace_mean = np.mean(beta_trace[burn:], axis=0)

# b_trace = np.asarray(b_trace)
# b_trace_mean = np.mean(b_trace[burn:], axis=0)

# print(np.cov(beta_trace))

# plt.figure()
# plt.plot(beta_trace[burn:,0])
# plt.plot(beta_trace[burn:,1])
# plt.plot(beta_trace[burn:,2])
# plt.plot(beta_trace[burn:,3])
# plt.show()

# x = data[0].price
# slope, intercept, r_value, p_value, std_err = stats.linregress(x,y[0])

# print(slope, intercept, r_value, p_value, std_err)

# x_hat = X[0][X[0][:,1].argsort()]

# intercept_1 = beta_trace_mean[2]
# slope_1 = beta_trace_mean[3]

# y_hat = intercept_1 + slope_1*x_hat[:,1]

# print(y_hat)
# print(x_hat[:,1])

# plt.figure()
# plt.plot(data[0][data[0].disp==0].price, np.log(data[0][data[0].disp==0].vol), '.k')
# plt.plot(data[0][data[0].disp==1].price, np.log(data[0][data[0].disp==1].vol), '.r')
# plt.plot(x_hat[:,1], slope*x_hat[:,1] + intercept, '-k')
# plt.show()

# plt.figure()
# plt.plot(data[0][data[0].disp==1].price, data[0][data[0].disp==1].vol, '.k')
# plt.plot(data[0][data[0].disp==0].price, data[0][data[0].disp==0].vol, '.r')
# plt.show()


# fig, ax = plt.subplots(88)
# # fig.subplots_adjust(hspace=0.3, wspace=0.2)
# # fig.suptitle('$\\tau_2^2$={:.6f}'.format(tau_2_squared[0]))
# fig.set_size_inches(8,4.5)
# for s, store in enumerate(stores):
#     ax[s].plot(data[s][data[s].disp==1].price, data[s][data[s].disp==1].vol, '.k')
#     ax[s].plot(data[s][data[s].disp==1].price, data[s][data[s].disp==1].vol, '.r')
# # for i in range(b.shape[0]):
# #     for j in range(tau_1_squared.shape[0]):
# #         ax[i, j].set_title('b={:.2f}'.format(b[i]) + '; $\\tau_1^2$={:.2f}'.format(tau_1_squared[j]), fontsize=8)
# #         hyperparams = b[i], tau_1_squared[j], tau_2_squared[0]
# #         # Calculates covariance given x and current hyperparameters
# #         cov = matern_52(x, hyperparams)
# #         # Generates random sample from a multivariate normal
# #         fx = multivariate_normal.rvs(mean=np.zeros(x.shape[0]), cov=cov)
# #         ax[i, j].plot(x, fx, '-k')
# plt.show()
