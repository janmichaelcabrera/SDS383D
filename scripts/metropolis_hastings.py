#!/packages/python/anaconda3/bin python

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import os, sys, time
import pandas as pd
from scipy.optimize import minimize
import scipy.stats as stats

class Traces:
    """
    This class instantiates traces
    """
    def __init__(self, name, model):
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
        self.model = model
        self.trace = []

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

    def save_trace(self, out_directory='traces/'):
        """
        Parameters
        ----------
            out_directory: str (optional)
                directory to save trace object as npy binary
        """
        if os.path.isdir(out_directory) == True: pass
        else: os.mkdir(out_directory)
        np.save(out_directory+self.model+'_'+self.name+'_trace', np.asarray(self.trace))


class Models:
    """
    This class runs various models for calibrating parameters
    """
    def __init__(self, model, func, X, y_obs, params):
        """
        Parameters
        ----------
            model: str
                Model name for file i/o purposes

            func: function object
                The function to be evaluated when optimizing the parameters of interest. The assumed functional form of y_obs.

            X: array-like
                Inputs that describe the model
                .. math: y_obs = f(X) + e

            y_obs: array-like
                Observed values with which to fit the function, func, with inputs, X characterized by parameters, param.

            params: list
                The starting values for evaluating the statistical models

        Attributes
        ----------
            y_hat: array-like
                The function, func, evaluated at the inputs, X, given the initial parameters, params
        """
        self.model = model
        self.func = func
        self.X = X
        self.y_obs = y_obs
        self.params = params
        self.y_hat = func(X, params)

    def mle(self):
        """
        Returns
        ----------
            Optimized parameters given the observed data
        """

        # Wrapper function for minimizing parameter(s) of interest
        def temp(params):
            """
            Parameters
            ----------
                params: scalar or array-like


            Returns
            ----------
                squared error loss
                    .. math: loss = \\sum_{time} (q_pred - q_obs)^2
            """
            self.y_hat = self.func(self.X, params)
            return (self.y_obs - self.y_hat).T @ (self.y_obs - self.y_hat)

        # Minimize loss function
        res = minimize(temp, self.params)
        self.params = res.x
        # Optimized parameter
        return self.params

    def metropolis_random_walk(self, samples=100, cov_scale=2.38, tune_every=10, times_tune=100):
        """
        Parameters
        ----------
            samples: int (optional)
                Number of accepted samples to collect from the Metropolis-Hastings algorithm

            tune_every: int (optional)
                The frequency with which to update the covariance of the proposal distribtuion. Currently set to 10, the covariance will be updated every 10 accepted samples.

            times_tune: int (optional)
                The number of times to perform the tuning step. Currently set to 100, this will update the covariance of the proposal distribution 100 times.

        Returns
        ----------
            Parameter traces: Traces objects
                Traces of parameters sampled. See Traces class for details.

        Attributes
        ----------
            p_accept: scalar
                Acceptance probablity
        """

        # Initialize traces
        alpha_trace = Traces('alpha', self.model)
        sigma_trace = Traces('sigma', self.model)

        # Total number of accepted samples from tuning
        tune_total = tune_every*times_tune
        alpha = self.params

        # Set initial guess for variance for data
        sigma_sq = 1

        # Number of dimensions
        d = len(alpha)

        # Initialize proposal covariance
        epsilon_cov = cov_scale*np.eye(d)/np.sqrt(d)
        # epsilon_cov = np.diag([1.66579242e-02, 1.37417513e-07])

        # Initialize the acceptance_count
        acceptance_count = 0

        i = 0
        t = 0
        # Begin sampling
        while acceptance_count < samples+tune_total:
            # Perform Gibbs sampling step on \sigma^2
            # sigma_sq | data \sim IG(N/2, \frac{1}{2} \sum_{i=1}^N (y_{obs,i} - \hat{y}_i)^2)
            sigma_sq = stats.invgamma.rvs(len(self.y_obs)/2, scale=(0.5*((self.y_obs - self.y_hat)**2).sum()))

            # Sample from proposal distribution given var_epsilon
            epsilon = stats.multivariate_normal.rvs(cov=epsilon_cov)
            
            # Propose new value for alpha given epsilon
            alpha_star = alpha + epsilon
            # Predicted at proposed value
            y_hat_star = self.func(self.X, alpha_star)

            # Log ratio of posteriors, to make computation tractable
            log_beta = -(1/(2*sigma_sq))*(((self.y_obs - y_hat_star)**2).sum() - ((self.y_obs - self.y_hat)**2).sum())

            # Ratio of posteriors, \beta = \frac{p(\alpha^{star}|data)}{p(\alpha | data)}
            beta = np.exp(log_beta)
            
            # Determine acceptance of proposed value
            if np.random.uniform() < np.minimum(1, beta):
                # Set proposed values
                alpha = alpha_star
                self.y_hat = y_hat_star
                # Iterate acceptance count
                acceptance_count += 1

                # Append alpha trace
                alpha_trace.update_trace(alpha.copy())
                # Append sigma trace
                sigma_trace.update_trace(sigma_sq.copy())

            # Tune variance of proposal distribution
            if (acceptance_count+1) % tune_every == 0 and t < tune_total:
                # New epsilon_cov = 2.4^2 S_b / d
                S = np.var(alpha_trace.trace[-tune_every:], axis=0)
                epsilon_cov = 2.4**2 * np.diag(S)/d
                t+=1

            i += 1

        # Calculates the acceptance probability
        self.p_accept = acceptance_count/i

        return alpha_trace, sigma_trace