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
    def __init__(self, name):
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
        self.trace = []
        # self.burn = burn

        # if burn > iterations:
        #     raise ValueError('Burn length %i is greater than total iterations %i'%(burn, iterations))

    # def plot(self, figures_directory='', plot_index=0):
    #     """
    #     Parameters
    #     ----------
    #         figures_directory: str
    #             sets the directory for which figures are to be saved in
    #             default behavior is to show the figure rather than saving

    #     """
    #     if len(figures_directory) == 0:
    #         plt.plot(self.trace[self.burn:], '-k')
    #         plt.ticklabel_format(useOffset=False)
    #         plt.show()
    #     else:
    #         for i in range(self.trace.shape[1]):
    #             plt.figure()
    #             plt.plot(self.trace[self.burn:,i], '-k', label=self.name+'_'+str(i))
    #             plt.ticklabel_format(useOffset=False)
    #             plt.savefig(figures_directory+self.name+'_'+str(i)+'_trace.png')
    #             plt.close()

    # def histogram(self, figures_directory='', plot_index=0):
    #     plt.figure()
    #     if len(figures_directory) == 0:
    #         plt.hist(self.trace[self.burn:], color='k')
    #         plt.ticklabel_format(useOffset=False)
    #         plt.locator_params(axis='x', nbins=5)
    #         plt.show()
    #     else:
    #         for i in range(self.trace.shape[1]):
    #             plt.figure()
    #             plt.hist(self.trace[self.burn:, i], color='k')
    #             plt.ticklabel_format(useOffset=False)
    #             plt.locator_params(axis='x', nbins=5)
    #             plt.savefig(figures_directory+self.name+'_'+str(i)+'_hist.png')
    #             plt.close()
    #     return 0

    def update_trace(self, value):
        """
        Parameters
        ----------
            index: int
                point in trace to update
            value: float
                the value passed to trace at index

        """
        self.trace.append(value)

    def mean(self):
        """
        Returns
        ----------
            mean: float
                calculates the mean for each column in a given trace

        """
        mean = np.mean(np.asarray(self.trace), axis=0)
        return mean

    def save_trace(self, out_directory=''):
        np.save(out_directory+self.name+'_trace', np.asarray(self.trace))

class Gibbs:
    """
	This class will use a Gibbs sampling methods to draw full conditional distributions
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

    def cheese(self):
        """
        Attributes
        ----------
            beta_trace: array
                samples from conditional distribution for beta

            mu_trace: array
                samples from conditional distribution for beta
        """
        out_directory = 'traces/cheese/'
        iterations = self.samples

        # Instantiate traces
        beta_trace = Trace('beta')
        sigma_trace = Trace('sigma')
        V_trace = Trace('V')

        # Number of data rows
        n = 5555
        p = 4
        d = 10

        # Number of stores
        s = 88

        ### Instantiate priors
        theta = np.zeros(p)
        V = np.eye(p)*10**6

        beta = np.zeros((s,p))
        sigma_sq = np.ones(s)

        #### Iterations
        # iterations = 5000
        # burn = 500
        for j in range(iterations):
            # Store variable for inverse-Wishart
            B = 0
            # Iterate over stores to calculate \beta_i's
            for store in range(s):

                n = len(self.Y[store])
                # beta_cov = [V^{-1} + X_i \frac{1}{\sigma_i^2} I_{n_i} X_i^T]^{-1}
                beta_cov = inv(inv(V) + (1/sigma_sq[store]) * self.X[store].T @ self.X[store])

                # beta_mean = beta_cov [V^{-1} \theta + \frac{1}{\sigma_i^2}X_i \bar{y}_i]
                beta_mean = beta_cov @ (inv(V) @ theta + (1/sigma_sq[store]) * (self.X[store] @ self.Y[store]))

                # Sample from multivariate normal for each store
                beta[store] = stats.multivariate_normal.rvs(mean=beta_mean, cov=beta_cov)

                # Shape and scale parameters for sigma
                a = n/2
                b = (self.Y[store] - self.X[store] @ beta[store]).T @ (self.Y[store] - self.X[store] @ beta[store])/2

                # Sample from inverse-gamma distribution
                sigma_sq[store] = stats.invgamma.rvs(a, 1/b)
            
                # Sum variable for inverse-Wishart
                B += np.tensordot((beta[store] - theta),(beta[store] - theta).T, axes=0)

            # Sample from multivarate normal, theta_mean = 1/s \sum_{i=1}^s \beta_i, theta_cov = V/s
            theta = stats.multivariate_normal.rvs(mean=(1/s)*beta.sum(axis=0), cov=V/s)

            # Sample from inverse-Wishart distribution
            V = stats.invwishart.rvs(d+s, np.eye(p) + B)

            # Append samples to trace
            beta_trace.update_trace(beta.copy())
            sigma_trace.update_trace(sigma_sq.copy())
            V_trace.update_trace(V.copy())

        # Save traces for later use
        beta_trace.save_trace(out_directory=out_directory)
        sigma_trace.save_trace(out_directory=out_directory)
        V_trace.save_trace(out_directory=out_directory)

        return 0