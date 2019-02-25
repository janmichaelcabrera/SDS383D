from __future__ import division
import numpy as np 
from numpy.linalg import inv
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

data = pd.read_csv('../../data/gdpgrowth.csv', delimiter=',')

Y = data['GR6096'] # X without intercept feature
X = data['DEF60'] # Y 

intercept = np.ones(len(Y)) # create intercept feature column
X = np.transpose(np.array((intercept, X))) # build matrix from intercept and X features



def gibbs_sampler(X, Y, iterations=5000):
	beta_trace = np.zeros((iterations, 2))
	omega_trace = np.ones(iterations)
	Lambda_trace = np.zeros((iterations, len(Y)))

	d = 1
	eta = 1
	n = len(X)
	k = np.ones(2)*0.1
	K = np.diag(k) # K
	h = 1

	m = np.zeros(2)

	for i in range(iterations):
	
		lambda_diag = np.ones(len(X))
		Lambda = np.diag(lambda_diag)

		m_star = inv(K + np.transpose(X) @ Lambda @ X) @ (K @ m + np.transpose(X) @ (Lambda @ Y))
		K_star = K + np.transpose(X) @ Lambda @ X
		d_star = d + n
		eta_star = np.transpose(m) @ K @ m + np.transpose(Y) @ Lambda @ Y + eta + np.transpose(K @ m + np.transpose(X) @ (Lambda @ Y)) @ inv(K_star) @ (K @ m + np.transpose(X) @ (Lambda @ Y))

		beta_trace[i] = stats.multivariate_normal.rvs(mean = m_star, cov=inv(omega_trace[i]* K_star))
		omega_trace[i] = stats.gamma.rvs(d_star/2, (2/eta_star))

		for j in range(len(lambda_diag)):
			lambda_diag[j] = stats.gamma.rvs((h+1)/2, (2/(h+omega_trace[i]*(Y[j] - np.transpose(X[j])@beta_trace[i])**2)))

		Lambda_trace[i] = lambda_diag

	return beta_trace, omega_trace, Lambda_trace

b, o, L, = gibbs_sampler(X, Y)

print(b[:,0].mean(), b[:,1].mean())

# plt.figure()
# plt.plot(b[:,0])
# plt.show()

# plt.figure()
# plt.plot(b[:,1])
# plt.show()