from __future__ import division
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from numpy.linalg import inv

class linear_model:
    """
    This class fits linear models using bayesian and frequentist methods to data sets
    """
    def __init__(self, X, Y, K=[]):
        """
        Parameters
        ----------
            X:  float
                feature matrix

            Y:  float
                response vector

            K:  float, optional
                precision matrix for multivariate normal prior on beta
                if not specified, sets a low precision, approximating the frequentist fit
        
        Raises
        ------
            ValueError
                If precision matrix shape does not match shape of feature matrix

        """
        self.X = X
        self.Y = Y
        self.K = np.array(K)

        if self.K.shape[0]==0:
            k = np.ones(self.X.shape[1])*0.0001
            self.K = np.diag(k)
        else:
            if self.K.shape[0] != self.X.shape[1]:
                raise ValueError('Precision matrix shape, %i does not match feature matrix shape, %i'%(self.K.shape[0], self.X.shape[1]))

    def bayesian(self):
        """
        Returns
        -------
            m_star: float
                Represents the slope and intercept coefficients to a linear model
                .. math:: m^* = (K + X^T \\Lambda X)^{-1} (K m + X^T \\Lambda Y)
                .. math:: y = m^*[0] + x m^*[1]
        """
        m = np.zeros(self.X.shape[1])

        Lambda = np.eye(len(self.X))

        m_star = inv(self.K + np.transpose(self.X) @ Lambda @ self.X) @ (self.K @ m + np.transpose(self.X) @ (Lambda @ self.Y))

        return m_star

    def frequentist(self):
        """
        Returns
        -------
            b_0: float
                intercept to an ordinary least squares linear model

            b_1: float
                slope to an ordinary least squares linear model

            .. math:: y = b_0 + x b_1

        [1] Adapted from https://www.geeksforgeeks.org/linear-regression-python-implementation/ 

        """

        n = np.size(self.X[:,1]) 
      
        m_x, m_y = np.mean(self.X[:,1]), np.mean(self.Y) 
      
        SS_xy = np.sum(self.Y*self.X[:,1]) - n*m_y*m_x 
        SS_xx = np.sum(self.X[:,1]*self.X[:,1]) - n*m_x*m_x 
      
        b_1 = SS_xy / SS_xx 
        b_0 = m_y - b_1*m_x 
      
        return b_0, b_1

