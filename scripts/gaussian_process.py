from __future__ import division
import numpy as np
import warnings
from scipy.stats import multivariate_normal

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

    def matern_52(x, hyperparams):
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
        C = np.zeros((x.shape[0], x.shape[0]))

        # Evaluate (i,j) components of covariance matrix
        for i in range(x.shape[0]):
            for j in range(x.shape[0]):
                d = np.abs(x[i] - x[j])
                C[i][j] = tau_1_squared*(1 + np.sqrt(5)*(d/b) + (5/3)*(d/b)**2)*np.exp(-np.sqrt(5)*(d/b)) + tau_2_squared*kronecker_delta(x[i], x[j])

        return C

class gaussian_process:
    """
    This class returns a vector of smoothed values given feature and response vectors
    """
    def __init__(self, x, hyperparams, cov='matern_52'):
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

    def generate_random_samples(self, mean=np.zeros(self.x.shape[0])):
        covariance = getattr(covariance_functions, self.cov)(self.x, self.hyperparams)
        self.fx = multivariate_normal.rvs(mean=mean, cov=covariance)
        return self.fx