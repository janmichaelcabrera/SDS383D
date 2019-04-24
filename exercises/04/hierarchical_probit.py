from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append('../../scripts/')
# from samplers import Trace
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

X = np.asarray(X)
w = np.asarray(w)

cols = X[0].shape[1]
d = 1

theta = np.zeros(cols)
V = np.eye(cols)
beta = stats.multivariate_normal.rvs(mean=theta, cov=V, size=S)

m = 0
v = 1
mu = stats.norm.rvs(loc=m, scale=v, size=S)

beta_trace = []
Z_trace = []
mu_trace = []
sigma_trace = []

## Iterations
iterations = 500

for p in range(iterations):
    B = np.zeros((cols,cols))
    for i in range(S):
        n_i = y[i].shape[0]
        for j in range(n_i):
            if y[i][j] == 1.0:
                a = 0 - X[i][j,:] @ beta[i]
                b = np.inf
            else:
                a = -np.inf
                b = 0 - X[i][j,:] @ beta[i]
            Z[i][j] = stats.truncnorm.rvs(a, b, loc=X[i][j,:] @ beta[i])

        beta_cov = inv(inv(V) + X[i].T @ X[i])
        beta_mean = beta_cov @ (inv(V) @ theta + X[i].T @ (Z[i] - w[i]*mu[i]))
        beta[i] = stats.multivariate_normal.rvs(mean=beta_mean, cov=beta_cov)

        mu_var = (1/v + w[i].T @ w[i])**-1
        mu_mean = mu_var*(m/v + w[i].T @ (Z[i] - X[i] @ beta[i]))
        mu[i] = stats.norm.rvs(loc=mu_mean, scale=mu_var)

        B += np.tensordot((beta[i] - theta).T, (beta[i] - theta), axes=0)

    theta = stats.multivariate_normal.rvs(mean=(1/S)*beta.sum(axis=0), cov=V/S)

    V = stats.invwishart.rvs(d+S, np.eye(cols) + B)

    v = stats.invgamma.rvs(S/2 - 1, (0.5*((mu - m)**2).sum())**-1)

    m = stats.norm.rvs(loc=mu.mean(), scale=(S/v)**-1)

    beta_trace.append(beta.copy())
    mu_trace.append(mu.copy())
    Z_trace.append(Z.copy())


beta_trace = np.asarray(beta_trace)
mu_trace = np.asarray(mu_trace)
Z_trace = np.asarray(Z_trace)

np.save('traces/hierarchical_probit/beta_trace', beta_trace)
np.save('traces/hierarchical_probit/mu_trace', mu_trace)
np.save('traces/hierarchical_probit/Z_trace', Z_trace)