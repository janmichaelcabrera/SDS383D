from __future__ import division
import numpy as np
from scipy.optimize import minimize
from numpy.linalg import inv

class kernels:
    """
    This class contains kernel functions for kernel smoothing
    """
    def __init__(self):
        pass

    def uniform(x):
        """
        Parameters
        ----------
            x: float
                argument to be evaluated

        Returns
        ----------
            k: float
                uniform kernel evaluation at x
                .. math:: k(x) = 1/2 I(x), with I(x) = 1, if |x| \\leq 1, 0 otherwise
        """
        if np.abs(x) <= 1:
            k = 0.5
        else:
            k = 0
        return k

    def gaussian(x):
        """
        Parameters
        ----------
            x: float
                argument to be evaluated

        Returns
        ----------
            k: float
                gaussian kernel evaluation at x
                .. math:: k(x) = \\frac{1}{\\sqrt{2 \\pi}} e^{-x^2/2}
        """
        return 1/np.sqrt(2*np.pi)*np.exp(-x**2/2)

class kernel_smoother:
    """
    This class returns a vector of smoothed values given feature and response vectors
    """
    def __init__(self, x, y, x_star, kernel='gaussian', h=1, D=1):
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
        self.y = y
        self.x_star = x_star
        self.h = h
        self.D = D
        self.kernel=kernel

        if x_star.shape != x:
            raise Warning('Feature vector and evaluation vector are not the same length. Residuals will not be calculated.')

    def local_general(self):
        """
        Returns
        ----------
            y_star: float, len(x_star)
                Predictor for x_star, 
                .. math:: y_i^* = [1 0 ...] H_i y
                .. math:: w(x_i, x^*) = \\frac{1}{2}K( \\frac{x_i - x^*}{h}), \\sum_{i=1}^n w(x_i, x^*) = 1

        Attributes
        ----------
            sigma: float, len(x_star)
                Standard error at x_star

            residuals: float, len(x_star)
                residuals of prediction from data (only calculated if len(y_star) == (len(y)))
                .. math:: r_i = y_i - \\hat{y}_i

            Hat_matrix: list
                first elements in hat matrix for a given x_star
        """
        # Instantiate vectors and matrices for storing calculations
        self.y_star = np.zeros(self.x_star.shape)
        self.sigma = np.zeros(self.x_star.shape)
        self.residuals = np.zeros(self.x_star.shape)
        self.Hat_matrix = []

        # Vector for determining y_star
        e_1 = np.append(1, np.zeros(self.D))

        # Iterate through each value in y_star
        for i in range(self.y_star.shape[0]):
            
            # Instantiate a weights vector
            weights = np.zeros(self.x.shape)

            # Instantiates X matrix 
            X = np.transpose(np.ones((self.x.shape[0], self.D+1)))

            # Populates X matrix given the order of the smoother, D
            for d in range(self.D):
                X[d+1] = (self.x - self.x_star[i])**(d+1)
            X = np.transpose(X)

            # Iterates through feature/response vectors
            for j in range(self.x.shape[0]):
                weights[j] = 1/self.h * getattr(kernels, self.kernel)((self.x[j] - self.x_star[i])/self.h)

            # Normalizes weights such that \sum_{i=1}^n w(x_i, x^*) = 1 
            weights = weights/weights.sum()

            # Construct weights matrix
            W = np.diag(weights)

            # Calculate hat matrix; H = (X^T W X)^{-1} X^T W
            H = (inv(np.transpose(X) @ W @ X) @ np.transpose(X) @ W)
            
            # Append hat matrix with current H matrix (needed for LOOCV)
            self.Hat_matrix.append(e_1 @ H)

            # \hat{f}(x^*) = e_1 H y
            self.y_star[i] = e_1 @ H @ self.y

            # Calculates residuals and confidence interval if y_star and y shapes are equivalent
            if self.y_star.shape == self.y.shape:
                self.residuals[i] = self.y[i] - self.y_star[i]

        # Residual sum of squared errors
        rss = (self.residuals**2).sum()
        # Approximate standard error of fit
        sigma_hat = rss/(len(self.y_star)-1)

        # Calculates approximate standard error 
        for i in range(self.sigma.shape[0]):
            self.sigma[i] = np.sqrt(np.transpose(self.Hat_matrix[i]) @ (self.Hat_matrix[i])*sigma_hat)

        return self.y_star

    def MSE(self, y_test):
        """
        Parameters
        ----------
            y_test: float
                Response vector

        Returns
        ----------
            MSE: float
                Average squared error for y_test given y
                .. math:: MSE \\approx \\frac{1}{n}\\sum(y_test - y_star)**2
        """
        self.y_test = y_test
        return np.mean((y_test - self.y_star)**2)

    def MSE_optimization(self, y_test):
        """
        Parameters
        ----------
            y_test: float
                Response vector

        Returns
        ----------
            h_star: float
                Uses BFGS minimization method to minimize the MSE by varying h_star
        """
        self.y_test = y_test
        def func(X):
            Y = kernel_smoother(self.x, self.y, self.x_star, kernel=self.kernel, h=X, D=self.D)
            Y.local_general()
            return Y.MSE(self.y_test)
        h_star = minimize(func, 1)
        self.h = h_star.x
        return h_star.x

    def LOOCV(self):
        """
        Returns
        ----------
            loocv: float
                Leave one out cross validation statistic (LOOCV)
                .. math: loocv = \\sum_{i=1}^n (\\frac{y_i - \\hat{y}_i}{1-H_{ii}})^2
        """
        loocv = np.zeros(len(self.y))
        for i in range(len(loocv)):
            loocv[i] = ((self.y[i] - self.y_star[i])/(1 - self.Hat_matrix[i][i]))**2
        return loocv.sum()

    def LOOCV_optimization(self):
        """
        Returns
        ----------
            h_star: float
                Uses BFGS minimization method to minimize the LOOCV by varying h_star
        """
        def func(X):
            Y = kernel_smoother(self.x, self.y, self.x_star, kernel=self.kernel, h=X, D=self.D)
            Y.local_general()
            return Y.LOOCV()
        h_star = minimize(func, 0.7)
        self.h = h_star.x
        return self.h