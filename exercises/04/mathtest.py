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
data = pd.read_csv('../../data/mathtest.csv', delimiter=',')

scores = []
means = np.zeros(100)

for i in range(100):
    scores.append(data[data.school==i+1]['mathscore'].values)

m = len(scores)
N = len(data)

for i in range(m):
    means[i] = scores[i].mean()

iterations = 500
burn = 100

tau_sq_trace = Trace('tau_sq', iterations, burn=burn)
sigma_sq_trace = Trace('sigma_sq', iterations, burn=burn)
mu_trace = Trace('mu', iterations, burn=burn)
theta_trace = Trace('theta', iterations, shape=m, burn=burn)

tau_sq = 0.5
sigma_sq = 1
mu = 50
theta = np.ones(m)

for p in range(iterations):
    a_1 = m/2
    b_1 = 1/(2*sigma_sq)*((theta - mu)**2).sum()

    tau_sq = stats.invgamma.rvs(a_1, 1/b_1)

    a_2 = (m+N)/2
    b_2 = 0

    for l in range(m):
        for j in range(len(scores[l])-1):
            b_2 = b_2 + (theta[j] - scores[l][j])**2

    b_2 = 1/(2*tau_sq)*((theta - mu)**2).sum() + 0.5 * b_2

    sigma_sq = stats.invgamma.rvs(a_2, 1/b_2)

    theta_bar = theta.mean()
    
    mu = stats.norm.rvs(theta_bar, scale=(sigma_sq*tau_sq)/m)

    for i in range(len(theta)):
        y_i = scores[i].mean()
        n_i = len(scores[i])
        v_i = (n_i/sigma_sq + 1/(tau_sq*sigma_sq))**-1
        theta_i_star = ((n_i/sigma_sq)*y_i + 1/(tau_sq*sigma_sq)*mu)*v_i

        theta[i] = stats.norm.rvs(theta_i_star, scale=v_i)

    tau_sq_trace.update_trace(p, tau_sq)
    sigma_sq_trace.update_trace(p, sigma_sq)
    mu_trace.update_trace(p, mu)
    theta_trace.update_trace(p, theta)

# mu_trace.plot(figures_directory='figures/')
# mu_trace.histogram(figures_directory='figures/')

# tau_sq_trace.plot(figures_directory='figures/')
# tau_sq_trace.histogram(figures_directory='figures/')

# sigma_sq_trace.plot(figures_directory='figures/')
# sigma_sq_trace.histogram(figures_directory='figures/')

# theta_trace.plot(figures_directory='figures/theta_trace/')
# theta_trace.histogram(figures_directory='figures/theta_trace/')

mean = theta_trace.mean()

k = np.zeros((m, 2))

for i in range(m):
    k[i] = [np.abs((scores[i].mean() - mean[i])/scores[i].mean()), len(scores[i])]

k = np.sort(k, axis=0)

n = k[:,1]
k = k[:,0]


plt.figure()
plt.plot(n, k, '.k')
plt.show()