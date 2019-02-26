from __future__ import division
import numpy as np

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
    def __init__(self, x, y, x_star, h=1):
        """
        Parameters
        ----------
            x: float
                Feature vector

            y: float
                Response vector

            x_star: float
                Scalar or vector to be evaluated

            h: float (optional)
                Bandwidth
        """
        self.x = x
        self.y = y
        self.x_star = x_star
        self.h = h

    def predictor(self, kernel='uniform'):
        """
        Parameters
        ----------
            kernel: str
                Kernel type to be used: Available kernels are uniform, gaussian, 

        Returns
        ----------
            y_star: float, len(x_star)
                Predictor for x_star
                .. math:: y_i^* = \\sum_{i=1}^n w(x_i, x^*) y_i
                .. math:: w(x_i, x^*) = \\frac{1}{2}K( \\frac{x_i - x^*}{h}), \\sum_{i=1}^n w(x_i, x^*) = 1
        """
        # Instantiate y_star
        y_star = np.zeros(self.x_star.shape)

        # Iterate through each value in y_star
        for i in range(y_star.shape[0]):
            
            # Instantiate a weights vector
            weights = np.zeros(self.x.shape)
            
            # Iterates through feature/response vectors
            for j in range(self.x.shape[0]):

                # X = \frac{(x_i - x^*)}{h}
                X = (self.x[j] - self.x_star[i])/self.h

                # w(x_i, x^*) = \frac{1}{h}K(X)
                weights[j] = 1/self.h*getattr(kernels, kernel)(X)

            # Normalizes weights such that \sum_{i=1}^n w(x_i, x^*) = 1    
            weights = weights/weights.sum()

            # y_i^* = \sum_{i=1}^n w(x_i, x^*) y_i
            y_star[i] = (weights*self.y).sum()

        return y_star