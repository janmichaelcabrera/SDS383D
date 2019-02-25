from __future__ import division
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from numpy.linalg import inv

class Trace:
	# This class will instantiate the traces for a problem
    def __init__(self, name, iterations, shape=1):
        self.name = name
        self.trace = np.zeros((iterations, shape))

    def plot(self, figures_directory=''):
        if len(figures_directory) == 0:
            plt.show()
        else:
            for i in range(self.trace.shape[1]):
                plt.figure()
                plt.plot(self.trace[:,i], '-b', label=self.name+'_'+str(i))
                plt.savefig(figures_directory+self.name+'_'+str(i)+'_trace.png')

    def update_trace(self, index, value):
        self.trace[index] = value

    def mean(self):
        mean = np.zeros(self.trace.shape[1])
        for i in range(len(mean)):
            mean[i] = self.trace[:,i].mean()
        return mean

class Gibbs:
	# This class will use a Gibbs sampling method to draw posteriors
    def __init__(self, X, Y, samples=100):
        self.X = X
        self.Y = Y
        self.samples = samples

    def heavy_tailed(self):
        iterations = self.samples
        beta_trace = Trace('beta', iterations = iterations, shape=self.X.shape[1])
        omega_trace = Trace('omega', iterations = iterations)
        Lambda_trace = Trace('Lambda', iterations=iterations, shape=len(self.Y))

        d = 1
        eta = 1
        n = self.X.shape[0]
        k = np.ones(self.X.shape[1])*0.1
        K = np.diag(k)
        h = 1
        omega = 1

        m = np.zeros(self.X.shape[1])

        lambda_diag = np.ones(self.X.shape[0])
        

        for i in range(iterations):

            Lambda = np.diag(lambda_diag)

            m_star = inv(K + np.transpose(self.X) @ Lambda @ self.X) @ (K @ m + np.transpose(self.X) @ (Lambda @ self.Y))
            K_star = K + np.transpose(self.X) @ Lambda @ self.X
            d_star = d + n
            eta_star = np.transpose(m) @ K @ m + np.transpose(self.Y) @ Lambda @ self.Y + eta + np.transpose(K @ m + np.transpose(self.X) @ (Lambda @ self.Y)) @ inv(K_star) @ (K @ m + np.transpose(self.X) @ (Lambda @ self.Y))

            beta = stats.multivariate_normal.rvs(mean = m_star, cov=inv(omega* K_star))

            omega = stats.gamma.rvs(d_star/2, (2/eta_star))

            for j in range(len(lambda_diag)):
                lambda_diag[j] = stats.gamma.rvs((h+1)/2, (2/(h+omega*(self.Y[j] - np.transpose(self.X[j])@beta)**2)))

            beta_trace.update_trace(i, beta)
            omega_trace.update_trace(i, omega)
            Lambda_trace.update_trace(i, lambda_diag)

        return beta_trace, omega_trace, Lambda_trace