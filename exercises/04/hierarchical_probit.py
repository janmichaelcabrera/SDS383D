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
df = pd.read_csv('../../data/polls.csv', delimiter=',')

# Create an array of unique state names
states = np.unique(df.state)
S = len(states)

# Instantiate lists for storing sorted data
data = []
y = []
X = []
Z = []
w = []

# ['Bacc' 'HS' 'NoHS' 'SomeColl']
ed = pd.get_dummies(df.edu, drop_first=False)

# ['18to29' '30to44' '45to64' '65plus']
age = pd.get_dummies(df.age, drop_first=False)

# Join data frames
df = df.join(ed).join(age)

# Sort data by state
for s, state in enumerate(states):
    data.append(df[df.state==state])
    y.append(np.nan_to_num(data[s].bush.values))

    # x = np.column_stack([data[s].HS, data[s].NoHS, data[s].SomeColl, data[s]['30to44'], data[s]['45to64'], data[s]['65plus'], data[s].female, data[s].black])
    x = np.column_stack([data[s].Bacc, data[s].HS, data[s].NoHS, data[s].SomeColl, data[s]['18to29'], data[s]['30to44'], data[s]['45to64'], data[s]['65plus'], data[s].female, data[s].black])
    X.append(x)
    Z.append(np.zeros(len(y[s])))
    w.append(np.ones(len(y[s])))

cols = X[0].shape[1]

sum_XX = np.array([X[i].T @ X[i] for i in range(S)]).sum(axis=0)

mu = np.ones(S)

theta = np.zeros(cols)
V = np.eye(cols)*10**6

beta = stats.multivariate_normal.rvs(mean=theta, cov=V)

sigma_sq = stats.invgamma.rvs(S/2, (0.5*(mu**2).sum())**-1)


beta_trace = []
mu_trace = []
sigma_trace = []

## Iterations
iterations = 1000
burn = 0

for p in range(iterations):
    for i in range(S):
        n_i = y[i].shape[0]
        for j in range(n_i):
            if y[i][j] == 1.0:
                a = 0
                b = np.inf
            else:
                a = -np.inf
                b = 0

            Z[i][j] = stats.truncnorm.rvs(a, b, loc=X[i][j,:] @ beta)

        mu_var = (1/sigma_sq + n_i)**-1
        mu_mean = w[i].T @ (Z[i] - X[i] @ beta)*mu_var
        mu[i] = stats.norm.rvs(loc=mu_mean, scale=mu_var)
    
    beta_cov = inv(inv(V) + sum_XX)
    sum_ZX = np.array([X[i].T @ (Z[i] - w[i] * mu[i]) for i in range(S)]).sum(axis=0)
    beta_mean = beta_cov @ (inv(V) @ theta + sum_ZX)
    beta = stats.multivariate_normal.rvs(mean=beta_mean, cov=beta_cov)

    sigma_sq = stats.invgamma.rvs(S/2-1, (0.5*(mu**2).sum())**-1)

    beta_trace.append(beta)
    mu_trace.append(mu.copy())
    sigma_trace.append(sigma_sq)


beta_trace = np.asarray(beta_trace)
mu_trace = np.asarray(mu_trace)
sigma_trace = np.asarray(sigma_trace)

plt.figure()
plt.plot(beta_trace[burn:])
plt.show()


# for p in range(iterations):
#     for i in range(len(Z)):
#         if y[0][i] == 1.0:
#             a = 0
#             b = np.inf
#         else:
#             a = -np.inf
#             b = 0

#         Z[i] = stats.truncnorm.rvs(a, b, loc=X[0][i,:] @ beta)

#     beta_cov = inv(inv(B_star) + X[0].T @ X[0])
#     beta_mean = beta_cov @ (inv(B_star) @ beta_star + X[0].T @ Z)

#     beta = stats.multivariate_normal.rvs(mean=beta_mean, cov=beta_cov)
#     Z_trace.append(Z)
#     beta_trace.append(beta)

# beta_trace = np.asarray(beta_trace)
# beta_trace_mean = np.mean(beta_trace[burn:], axis=0)
