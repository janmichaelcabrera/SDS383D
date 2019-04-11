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

d = 10
s = 88
sigma_sq = 1

theta = np.zeros(4)
V = np.eye(4)*10**6

beta = np.zeros((s,4))
sigma = np.zeros(s)
# beta[0] = stats.multivariate_normal.rvs(mean=theta, cov=V)

beta_trace = []
sigma_trace = []
V_trace = []

#### Iterations
iterations = 1000
burn = 0
for j in range(iterations):
    B = 0
    for store in range(s):
        n = len(y[store])

        beta_cov = inv(inv(V) + X[store].T @ ((1/sigma_sq)*np.eye(n)) @ X[store])
        beta_mean = beta_cov @ (inv(V) @ theta + (1/sigma_sq) * (X[store] @ y[store]))

        beta[store] = stats.multivariate_normal.rvs(mean=beta_mean, cov=beta_cov)

        a = n/2
        b = (y[store] - X[store] @ beta[store]) @ (y[store] - X[store] @ beta[store])/2

        sigma[store] = stats.invgamma.rvs(a, 1/b)
    
        B = B + np.tensordot((beta[store] - theta),(beta[store] - theta).T, axes=0)
    # print(B)
    V = stats.invwishart.rvs(d+s, np.eye(4) + B)
    # print(V)
    beta_trace.append(beta)
    sigma_trace.append(sigma)

beta_trace = np.asarray(beta_trace)
beta_trace_mean = np.mean(beta_trace[burn:], axis=0)
# print(beta_trace_mean)

store_index = 1

x_hat = X[store_index][X[store_index][:,1].argsort()]

plt.figure()
plt.plot(data[store_index][data[store_index].disp==0].price, np.log(data[store_index][data[store_index].disp==0].vol), '.k')
plt.plot(data[store_index][data[store_index].disp==1].price, np.log(data[store_index][data[store_index].disp==1].vol), '.r')
plt.plot(x_hat[:,1], beta_trace_mean[store_index][0]+beta_trace_mean[store_index][1]*x_hat[:,1], '-k')
plt.plot(x_hat[:,1], (beta_trace_mean[store_index][0]+beta_trace_mean[store_index][2])+(beta_trace_mean[store_index][1]+beta_trace_mean[store_index][3])*x_hat[:,1], '-r')
plt.show()


# plt.figure()
# plt.plot(beta_trace[burn:,0], label='Beta 0')
# plt.plot(beta_trace[burn:,1], label='Beta 1')
# plt.plot(beta_trace[burn:,2], label='Beta 2')
# plt.plot(beta_trace[burn:,3], label='Beta 3')
# plt.legend(loc=0)
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
