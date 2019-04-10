from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append('../../scripts/')
from samplers import Trace
import scipy.stats as stats

np.random.seed(3)

# Import data
df = pd.read_csv('../../data/cheese.csv', delimiter=',')

stores = np.unique(df.store)

data = []
y = []
X = []

for s, store in enumerate(stores):
    data.append(df[df.store==store])
    y.append(np.log(data[s].vol))
    data[s].price = np.log(data[s].price)
    X.append(np.array([np.ones(data[s].shape[0]), data[s].price, data[s].disp, data[s].price*data[s].disp]).T)

print(X[0])

# X_1 = 

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

# scores = []
# means = np.zeros(100)

# for i in range(100):
#     scores.append(data[data.school==i+1]['mathscore'].values)

# m = len(scores)
# N = len(data)

# for i in range(m):
#     means[i] = scores[i].mean()

# iterations = 500
# burn = 100

# tau_sq_trace = Trace('tau_sq', iterations, burn=burn)
# sigma_sq_trace = Trace('sigma_sq', iterations, burn=burn)
# mu_trace = Trace('mu', iterations, burn=burn)
# theta_trace = Trace('theta', iterations, shape=m, burn=burn)

# tau_sq = 0.5
# sigma_sq = 1
# mu = 50
# theta = np.ones(m)

# for p in range(iterations):
#     a_1 = m/2
#     b_1 = 1/(2*sigma_sq)*((theta - mu)**2).sum()

#     tau_sq = stats.invgamma.rvs(a_1, 1/b_1)

#     a_2 = (m+N)/2
#     b_2 = 0

#     for l in range(m):
#         for j in range(len(scores[l])-1):
#             b_2 = b_2 + (theta[j] - scores[l][j])**2

#     b_2 = 1/(2*tau_sq)*((theta - mu)**2).sum() + 0.5 * b_2

#     sigma_sq = stats.invgamma.rvs(a_2, 1/b_2)

#     theta_bar = theta.mean()
    
#     mu = stats.norm.rvs(theta_bar, scale=(sigma_sq*tau_sq)/m)

#     for i in range(len(theta)):
#         y_i = scores[i].mean()
#         n_i = len(scores[i])
#         v_i = (n_i/sigma_sq + 1/(tau_sq*sigma_sq))**-1
#         theta_i_star = ((n_i/sigma_sq)*y_i + 1/(tau_sq*sigma_sq)*mu)*v_i

#         theta[i] = stats.norm.rvs(theta_i_star, scale=v_i)

#     tau_sq_trace.update_trace(p, tau_sq)
#     sigma_sq_trace.update_trace(p, sigma_sq)
#     mu_trace.update_trace(p, mu)
#     theta_trace.update_trace(p, theta)


# plt.figure()
# plt.plot(n, k, '.k')
# plt.show()