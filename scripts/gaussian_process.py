from __future__ import division
import numpy as np
import warnings
from scipy.stats import multivariate_normal
from numpy.linalg import inv

def kronecker_delta(x_1, x_2):
    """
    Parameters
    ----------
        x_1, x_2: floats (scalars)
    
    Returns
    ----------
        delta: 0 or 1
            .. math:: \\delta(x_1, x_2)
    """
    if x_1 == x_2:
        delta = 1
    else:
        delta = 0
    return delta

class covariance_functions:
    """
    This class contains kernel functions for kernel smoothing
    """
    def __init__(self):
        pass

    def matern_52(x_1, x_2, hyperparams):
        """
        Parameters
        ----------
            x: float (vector)
                Vector of points

        Returns
        ----------
            C: float (matrix)
                Returns a Matern (5,2) square covariance matrix of size(x)
                .. math:: C_{5,2}(x_1, x_2) = \\tau_1^2 [1 + \\sqrt{5} d / b + (5/3) (d/b)^2 ] e^{-\\sqrt{5} (d/b)} + \\tau_2^2 \\delta(x_1, x_2)
        """

        # Unpack hypereparameters
        b, tau_1_squared, tau_2_squared = hyperparams

        # Initialize covariance matrix
        C = np.zeros((x_1.shape[0], x_2.shape[0]))

        # Evaluate (i,j) components of covariance matrix
        for i in range(x_1.shape[0]):
            for j in range(x_2.shape[0]):
                d = np.abs(x_1[i] - x_2[j])
                C[i][j] = tau_1_squared*(1 + np.sqrt(5)*(d/b) + (5/3)*(d/b)**2)*np.exp(-np.sqrt(5)*(d/b)) + tau_2_squared*kronecker_delta(x_1[i], x_2[j])

        return C

class gaussian_process:
    """
    This class returns a vector of smoothed values given feature and response vectors
    """
    def __init__(self, x, hyperparams, y = [], x_star = [], cov='matern_52'):
        """
        Parameters
        ----------
            x: float
                Feature vector

            y: float
                Response vector

            x_star: float
                Scalar or vector to be evaluated

            kernel: str (optional)
                Kernel type to be used: Available kernels are gaussian,
                
            h: float (optional)
                Bandwidth

            D: int (optional)
                Order of polynomial smoother
        """
        self.x = x
        self.hyperparams = hyperparams
        self.cov = cov
        self.y = y
        self.x_star = x_star

    def residuals(self):
        return 0

    def approx_sigma(self):
        C_xx_star = getattr(covariance_functions, self.cov)(self.x, self.x, self.hyperparams)
        C_xx = getattr(covariance_functions, self.cov)(self.x, self.x, self.hyperparams)
        sigma_hat = 1
        weights = C_xx_star @ inv(C_xx + sigma_hat*np.eye(self.x.shape[0]))

        y_star = np.transpose(weights) @ self.y
        self.residuals = self.y - y_star
        # Residual sum of squared errors
        rss = (self.residuals**2).sum()
        # Approximate standard error of fit
        return rss/(len(y_star)-1)

    def smoother(self, sigma_hat=[]):
        if not sigma_hat:
            sigma_hat = 1
        C_xx_star = getattr(covariance_functions, self.cov)(self.x_star, self.x, self.hyperparams)
        C_xx = getattr(covariance_functions, self.cov)(self.x, self.x, self.hyperparams)
        weights = C_xx_star @ inv(C_xx + sigma_hat*np.eye(self.x.shape[0]))

        y_star = np.transpose(weights) @ self.y

        return y_star

    def generate_random_samples(self, mean=[]):
        covariance = getattr(covariance_functions, self.cov)(self.x, self.x, self.hyperparams)
        self.fx = multivariate_normal.rvs(mean=mean, cov=covariance)
        return self.fx