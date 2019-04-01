from __future__ import division
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from numpy.linalg import inv

class Trace:
    """
    This class instantiates traces
    """
    def __init__(self, name, iterations, shape=1, burn=100):
        """
        Parameters
        ----------
        name: str
            name of the trace for file i/o purposes

        iterations: int
            number of iterations to be ran for a given trace

        shape: int
            sets size of trace, default=1

        """
        self.name = name
        self.trace = np.zeros((iterations, shape))
        self.burn = burn

        if burn > iterations:
            raise ValueError('Burn length %i is greater than total iterations %i'%(burn, iterations))

    def plot(self, figures_directory='', plot_index=0):
        """
        Parameters
        ----------
            figures_directory: str
                sets the directory for which figures are to be saved in
                default behavior is to show the figure rather than saving

        """
        if len(figures_directory) == 0:
            plt.plot(self.trace[self.burn:], '-k')
            plt.ticklabel_format(useOffset=False)
            plt.show()
        else:
            for i in range(self.trace.shape[1]):
                plt.figure()
                plt.plot(self.trace[self.burn:,i], '-k', label=self.name+'_'+str(i))
                plt.ticklabel_format(useOffset=False)
                plt.savefig(figures_directory+self.name+'_'+str(i)+'_trace.png')
                plt.close()

    def histogram(self, figures_directory='', plot_index=0):
        plt.figure()
        if len(figures_directory) == 0:
            plt.hist(self.trace[self.burn:], color='k')
            plt.ticklabel_format(useOffset=False)
            plt.locator_params(axis='x', nbins=5)
            plt.show()
        else:
            for i in range(self.trace.shape[1]):
                plt.figure()
                plt.hist(self.trace[self.burn:, i], color='k')
                plt.ticklabel_format(useOffset=False)
                plt.locator_params(axis='x', nbins=5)
                plt.savefig(figures_directory+self.name+'_'+str(i)+'_hist.png')
                plt.close()
        return 0

    def update_trace(self, index, value):
        """
        Parameters
        ----------
            index: int
                point in trace to update
            value: float
                the value passed to trace at index

        """
        self.trace[index] = value

    def mean(self):
        """
        Returns
        ----------
            mean: float
                calculates the mean for each column in a given trace

        """
        mean = np.zeros(self.trace.shape[1])
        for i in range(len(mean)):
            mean[i] = self.trace[self.burn:,i].mean()
        return mean

class Gibbs:
    """
	This class will use a Gibbs sampling methods to draw posterior distributions
    """
    def __init__(self, X, Y, samples=100):
        """
        Parameters
        ----------
            X:  float
                feature matrix

            Y:  float
                response vector

            samples: int
                number of samples to be drawn from posteriors, default is 100 for debugging purposes

        """
        self.X = X
        self.Y = Y
        self.samples = samples

    def heavy_tailed(self):
        """
        Returns
        -------
            beta_trace: float
                samples from posterior of the form
                .. math:: \\beta | \\bar{y} \\omega ~ N(m^*, (\\omega K^*)^{-1})

            omega_trace: float
                samples from the posterior of the form
                .. math:: \\omega | \\bar{y} ~ Gamma(d^*/2, \\eta^*/2)

            Lambda_trace: float
                samples from the posterior of the form
                .. math:: \\lambda_i | \\bar{y}, \\beta, \\omega ~ Gamma(\\frac{h+1}{2}, \\frac{h + \\omega(y_i + x_i^T \\beta)^2}{2})
        """

        #### Instantiate Traces
        iterations = self.samples
        beta_trace = Trace('beta', iterations = iterations, shape=self.X.shape[1])
        omega_trace = Trace('omega', iterations = iterations)
        Lambda_trace = Trace('Lambda', iterations=iterations, shape=len(self.Y))

        #### Instantiate Prior values
        d = 1
        eta = 1

        n = self.X.shape[0]

        k = np.ones(self.X.shape[1])*0.1
        K = np.diag(k)

        h = 1
        omega = 1

        m = np.zeros(self.X.shape[1])

        lambda_diag = np.ones(self.X.shape[0])

        #### Iteratively sample conditional distributions
        for i in range(iterations):
            Lambda = np.diag(lambda_diag)

            # m^* = (K + X^T \Lambda X)^{-1}(K m + X^T \Lambda y)            
            m_star = inv(K + np.transpose(self.X) @ Lambda @ self.X) @ (K @ m + np.transpose(self.X) @ (Lambda @ self.Y))

            # K^* = (K+ X^T \Lambda X)
            K_star = K + np.transpose(self.X) @ Lambda @ self.X

            # d^* = d + n
            d_star = d + n

            # \eta^* = m^T K m + y^T \Lambda y + \eta - (K m + X^T \Lambda y)^T  {K^*}^{-1} (K m + X^T \Lambda y)
            eta_star = np.transpose(m) @ K @ m + np.transpose(self.Y) @ Lambda @ self.Y + eta - np.transpose(K @ m + np.transpose(self.X) @ (Lambda @ self.Y)) @ inv(K_star) @ (K @ m + np.transpose(self.X) @ (Lambda @ self.Y))

            ### Sample posterior values
            beta = stats.multivariate_normal.rvs(mean = m_star, cov=inv(omega*K_star))
            omega = stats.gamma.rvs(d_star/2, (2/eta_star))
            for j in range(len(lambda_diag)):
                lambda_diag[j] = stats.gamma.rvs((h+1)/2, (2/(h+omega*(self.Y[j] - np.transpose(self.X[j])@beta)**2)))

            ### Update traces with posterior values
            beta_trace.update_trace(i, beta)
            omega_trace.update_trace(i, omega)
            Lambda_trace.update_trace(i, lambda_diag)

        return beta_trace, omega_trace, Lambda_trace