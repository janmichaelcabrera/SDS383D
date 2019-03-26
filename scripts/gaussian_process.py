from __future__ import division
import numpy as np
import warnings
from scipy.stats import multivariate_normal
from scipy.spatial.distance import euclidean
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
    if np.array_equal(x_1, x_2) == True:
        delta = 1
    else:
        delta = 0
    return delta

class covariance_functions:
    """
    This class contains covariance function for Gaussian process smoothing
    """
    def __init__(self):
        pass

    def squared_exponential(x_1, x_2, hyperparams):
        """
        Parameters
        ----------
            x: float (vector)
                Vector of points

        Returns
        ----------
            C: float (matrix)
                Returns a Matern (5,2) square covariance matrix of size(x)
                .. math:: C_{SE}(x_1, x_2) = \\tau_1^2 e^{-\\frac{1}{2} (d/b)^2} + \\tau_2^2 \\delta(x_1, x_2)
        """

        # Unpack hypereparameters
        b, tau_1_squared, tau_2_squared = hyperparams

        # Initialize covariance matrix
        C = np.zeros((x_1.shape[0], x_2.shape[0]))

        # Evaluate (i,j) components of covariance matrix
        for i in range(x_1.shape[0]):
            for j in range(x_2.shape[0]):
                d = euclidean(x_1[i], x_2[j])
                C[i][j] = tau_1_squared*np.exp(-(1/2)*(d/b)**2) + tau_2_squared*kronecker_delta(x_1[i], x_2[j])
        return C

    def matern_32(x_1, x_2, hyperparams):
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
                d = euclidean(x_1[i], x_2[j])
                C[i][j] = tau_1_squared*(1 + np.sqrt(3)*(d/b))*np.exp(-np.sqrt(3)*(d/b)) + tau_2_squared*kronecker_delta(x_1[i], x_2[j])

        return C

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
                d = euclidean(x_1[i], x_2[j])
                C[i][j] = tau_1_squared*(1 + np.sqrt(5)*(d/b) + (5/3)*(d/b)**2)*np.exp(-np.sqrt(5)*(d/b)) + tau_2_squared*kronecker_delta(x_1[i], x_2[j])

        return C

class svi_covariance_functions:
    """
    Single value input covariance functions. Same as above except takes one parameter, x. These functions are slightly more computationally efficient than the above. For a single value input function, the resultant covariance matrix is symmetric. These functions calculate the upper diagonal portion only and then mirrors ther results.
    """
    def __init__(self):
        pass

    def squared_exponential(x, hyperparams):
        """
        Parameters
        ----------
            x: float (vector)
                Vector of points

        Returns
        ----------
            C: float (matrix)
                Returns a Matern (5,2) square covariance matrix of size(x)
                .. math:: C_{SE}(x_1, x_2) = \\tau_1^2 e^{-\\frac{1}{2} (d/b)^2} + \\tau_2^2 \\delta(x_1, x_2)
        """

        # Unpack hypereparameters
        b, tau_1_squared, tau_2_squared = hyperparams

        # Initialize covariance matrix
        C = np.zeros((x.shape[0], x.shape[0]))

        # Evaluate (i,j) components of covariance matrix
        for i in range(x.shape[0]):
            for j in range(x.shape[0]):
                if j >= i:
                    d = euclidean(x[i], x[j])
                    C[i][j] = tau_1_squared*np.exp(-(1/2)*(d/b)**2) + tau_2_squared*kronecker_delta(x[i], x[j])
                else:
                    C[i][j] = C[j][i]
        return C

    def matern_32(x, hyperparams):
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
                if j >= i:
                    d = euclidean(x[i], x[j])
                    C[i][j] = tau_1_squared*(1 + np.sqrt(3)*(d/b))*np.exp(-np.sqrt(3)*(d/b)) + tau_2_squared*kronecker_delta(x[i], x[j])
                else:
                    C[i][j] = C[j][i]
        return C

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
                if j>= i:
                    d = euclidean(x[i], x[j])
                    C[i][j] = tau_1_squared*(1 + np.sqrt(5)*(d/b) + (5/3)*(d/b)**2)*np.exp(-np.sqrt(5)*(d/b)) + tau_2_squared*kronecker_delta(x[i], x[j])
                else:
                    C[i][j] = C[j][i]
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

            hyperparams: float (tuple)
                Hyperparameters for a Guassian process

            y: float (Not required for generating random samples)
                Response vector

            x_star: float (Not required for generating random samples)
                Scalar or vector to be evaluated

            cov: str (optional)
                Covariance function to be used. Available functions are:
                    Squared exponential = 'squared_exponential'
                    Matern 5/2 = 'matern_52'
                    Matern 3/2 = 'matern_32'

        """
        self.x = x
        self.hyperparams = hyperparams
        self.cov = cov
        self.y = y
        self.x_star = x_star

    def approx_var(self):
        """
        Returns
        ----------
            variance: float(scalar)
                Approximates the variance for a given problem using residual sum of squared errors. The function runs a Gaussian Process once for the approximation

        Attributes
        ----------
            residuals: float, len(y)
                residuals of prediction from data
                .. math:: r_i = y_i - \\hat{y}_i
        """

        # Evaluate covariance
        C_xx = getattr(covariance_functions, self.cov)(self.x, self.x, self.hyperparams)

        # Assume initial estimate for variance
        variance = 1

        # Calculate weights
        weights = C_xx @ inv(C_xx + variance*np.eye(self.x.shape[0]))

        # Calculate predictions from GP
        y_star = np.transpose(weights) @ self.y

        # Calculate residuals from the above prediction
        self.residuals = self.y - y_star

        # Residual sum of squared errors
        rss = (self.residuals**2).sum()

        # Approximate standard error of fit
        return rss/(len(y_star)-1)

    def smoother(self, variance=[]):
        """
            Parameters
            ----------
            variance: float (scalar; optional)
                If not specified, the sample variance is assumed to be one

            Returns
            ----------
            y_star: float, len(x_star)
                Predictor for x_star, 
                .. math:: y^* = W^T y
                .. math:: W = C(x^*, x) (C(x, x) + \\sigma^2 I)^{-1}

            post_var: float, len(x_star)
                Posterior variance given the observed data
                .. math:: var[f(x^*)| y] = C(x^*, x^*) - C(x^*, x) ( C(x, x) + \\sigma^2 I)^{-1} C(x^*, x)^T
        """
        # If a sample variance is not spcified, assume it is 1
        if not variance:
            variance = 1

        # Evaluate C(x^*, x)
        C_x_star_x = getattr(covariance_functions, self.cov)(self.x_star, self.x, self.hyperparams)
        # Evaluate C(x, x)
        C_xx = getattr(svi_covariance_functions, self.cov)(self.x, self.hyperparams)
        # Evaluate C(x^*, x^*)
        C_star_star = getattr(svi_covariance_functions, self.cov)(self.x_star, self.hyperparams)

        # Calculate weights matrix,  W = C(x^*, x) (C(x, x) + \\sigma^2 I)^{-1}
        weights = C_x_star_x @ inv(C_xx + variance*np.eye(self.x.shape[0]))
        
        # Calculate y_star, y^* = W^T y
        y_star = np.transpose(weights) @ self.y

        # Calculates posterior variance, var[f(x^*)| y] = C(x^*, x^*) - C(x^*, x) ( C(x, x) + \\sigma^2 I)^{-1} C(x^*, x)^T
        post_var = np.diag(C_star_star - C_x_star_x @ inv(C_xx + variance*np.eye(self.x.shape[0])) @ np.transpose(C_x_star_x))

        return y_star, post_var

    def log_marginal_likelihood(self, hyperparams, variance=[]):

        if not variance:
            variance = 1

        # Unpack hypereparameters
        b, tau_1_squared, tau_2_squared = hyperparams

        # Evaluate C(x, x)
        C_xx = getattr(svi_covariance_functions, self.cov)(self.x, hyperparams)

        covariance = variance*np.eye(self.x.shape[0]) + C_xx
        
        return multivariate_normal.logpdf(self.y, cov=covariance)
        # return -self.x.shape[0]/2 * np.log(np.linalg.det(covariance)) - 0.5 * np.transpose(self.y) @ np.linalg.inv(covariance) @ self.y

    def generate_random_samples(self, mean=[]):
        """
        Parameters
        ----------
            mean: float, vector
                Mean vector for multivariate normal, should have same shape as self.x. If not specified, assumed to be mean zero multivariate normal.

        Returns
        ----------
            fx: float, len(x)
                Returns random samples from a multivariate with specified mean

        Raises
        ----------
            ValueError
                If mean vector shape and x shape do not match
        """

        if not mean:
            mean = np.zeros(self.x.shape[0])

        if mean.shape[0] != self.x.shape[0]:
            raise ValueError('Mean shape, %i  and x shape, %i do not match'%(mean.shape[0], self.x.shape[0]))

        covariance = getattr(covariance_functions, self.cov)(self.x, self.x, self.hyperparams)
        fx = multivariate_normal.rvs(mean=mean, cov=covariance)
        return fx