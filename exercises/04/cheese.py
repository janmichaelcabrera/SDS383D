from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append('../../scripts/')
from samplers import Trace
import scipy.stats as stats
from numpy.linalg import inv

np.random.seed(3)

# Import data
df = pd.read_csv('../../data/cheese.csv', delimiter=',')

stores = np.unique(df.store)
n = 5555
p = 4
d = p + 1

data = []
y = []
X = []

for s, store in enumerate(stores):
    data.append(df[df.store==store])
    y.append(np.log(data[s].vol))
    data[s].price = np.log(data[s].price)
    X.append(np.array([np.ones(data[s].shape[0]), data[s].price, data[s].disp, data[s].price*data[s].disp]).T)

W = X

sum_XX = np.zeros((4,4))
for i in range(88):
    sum_XX = sum_XX + X[i].T @ X[i]

b = np.zeros((88,4))

sigma_sq = 1

Sigma = np.random.rand(4,4)
Sigma = Sigma @ Sigma.T

mu_beta = np.zeros(4)
cov_beta = Sigma

C = Sigma

beta = stats.multivariate_normal.rvs(mean=mu_beta, cov = inv(cov_beta)).T

### Instantiate traces
b_trace = []
beta_trace = []
sigma_sq_trace = []
Sigma_trace = []

### Iterations start here
iterations = 1000
burn = 0
for j in range(iterations):
    ## b_i | y
    for i in range(88):
        b_cov = inv(inv(Sigma) + (1/sigma_sq)*W[i].T @ W[i])
        b_mean = (1/sigma_sq)*W[i] @ (y[i] - X[i] @ beta) @ b_cov
        b[i] = stats.multivariate_normal.rvs(mean=b_mean, cov=b_cov)

    b_trace.append(b)

    ## beta | y
    beta_cov = inv(inv(Sigma) + (1/sigma_sq)*sum_XX)

    beta_xy_sum = np.zeros(4)

    for i in range(88):
        beta_xy_sum = beta_xy_sum + X[i] @ (y[i] - W[i] @ b[i])

    beta_mean = (inv(cov_beta) @ mu_beta + (1/sigma_sq) * beta_xy_sum) @ beta_cov
    beta = stats.multivariate_normal.rvs(mean=beta_mean, cov=beta_cov)

    beta_trace.append(beta)

    ## sigma_sq | y
    sigma_xy_sum = 0

    for i in range(88):
        sigma_xy_sum = sigma_xy_sum + (y[i] - X[i] @ beta - W[i] @ b[i]).T @ (y[i] - X[i] @ beta - W[i] @ b[i])

    sigma_sq = stats.invgamma.rvs(n/2, 2/sigma_xy_sum)

    sigma_sq_trace.append(sigma_sq)

    ## Sigma | b
    Sigma_bb_sum = np.zeros((4,4))

    for i in range(88):
        Sigma_bb_sum = Sigma_bb_sum + np.tensordot(b[i], b[i].T, axes=0)

    Sigma = stats.invwishart.rvs(df=d+88, scale=C+Sigma_bb_sum)
    Sigma_trace.append(Sigma)

b_trace_mean = np.mean(b_trace, axis=0)
beta_trace_mean = np.mean(beta_trace, axis=0)

# print(beta_trace)

print(beta_trace_mean)

plt.figure()
for i in range(iterations):
    plt.plot(i, beta_trace[i][0], '.k')
for i in range(iterations):
    plt.plot(i, beta_trace[i][1], '.r')
for i in range(iterations):
    plt.plot(i, beta_trace[i][2], '.b')
for i in range(iterations):
    plt.plot(i, beta_trace[i][3], '.g')
plt.show()


# x_hat = X[0][X[0][:,1].argsort()]
# y_hat = x_hat @ beta_trace_mean + W[0] @ b[0]

# print(y_hat)
# print(x_hat[:,1])

# plt.figure()
# plt.plot(data[0][data[0].disp==0].price, np.log(data[0][data[0].disp==0].vol), '.b')
# plt.plot(data[0][data[0].disp==1].price, np.log(data[0][data[0].disp==1].vol), '.r')
# plt.plot(x_hat[:,1], y_hat, '.k')
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
