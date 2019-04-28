#!/packages/python/anaconda3/bin python

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os, sys, time
import pandas as pd
from dft_esm import energy_storage
from scipy.signal import savgol_filter
from scipy.optimize import minimize
import scipy.stats as stats

class Traces:
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
        """
        Parameters
        ----------
            out_directory: str
                directory to save trace object as npy binary
        """
        np.save(out_directory+self.name+'_trace', np.asarray(self.trace))

class Models:
    """
    This class runs various models for calibrating parameters
    """
    def __init__(self, T_f, T_r, time, q_obs):
        """
        Parameters
        ----------
            T_f: vector
                measured temperature on front of device

            T_r: vector
                measured temperature at rear of device

            time: vector
                experimental times

            q_obs: vector
                observed heat flux kW/m^2
        """
        self.T_f = T_f
        self.T_r = T_r
        self.time = time
        self.q_obs = q_obs


    def mle(self, k_init):
        """
        Parameters
        ----------
            k_init: scalar
                Initial guess for thermal conductivity in W/m-K. 

        Returns
        ----------
            Optimized thermal conductivity given the observed data
        """

        # Wrapper function for minimizing parameter of interest, k
        def func(k, q_obs, time, T_f, T_r):
            """
            Inputs
            ----------
                k: scalar
                    thermal conductivity W/m-k

                q_obs: vector
                    observed heat flux kW/m^2

                time: vector
                    experimental times

                T_f: vector
                    measured temperature on front of device

                T_r: vector
                    measured temperature at rear of device

            Returns
            ----------
                squared error loss
                    .. math: loss = \\sum_{time} (q_pred - q_obs)^2
            """
            q_pred = energy_storage(T_f, T_r, time, alpha=k)
            return (q_pred - q_obs).T @ (q_pred - q_obs)

        # Minimize loss function
        res = minimize(func, k_init, args=(self.q_obs, self.time, self.T_f, self.T_r))

        # Optimized parameter
        return res.x

    def metropolis(self, k_init, samples):
        """
        Parameters
        ----------
            k_init: scalar
                Initial guess for thermal conductivity in W/m-K.

            samples: scalar
                Number of samples to run the Metropolis-Hastings algorithm 

        Returns
        ----------
            Parameter traces for thermal conductivity and standard deviation
        """
        # Set trace directory
        trace_directory = 'traces'

        # Initialize traces
        alpha_trace = Traces('alpha')
        sigma_trace = Traces('sigma')

        # Set initial guess for alpha in the M-H Algorithm
        alpha = k_init

        # Set initial predicted values for heat flux given initial guess in thermal conductivity
        q_hat = energy_storage(self.T_f, self.T_r, self.time, alpha=alpha)

        # Set initial tuning parameter for proposal distribution
        var_epsilon = 0.01

        # Set initial guess for variance for data
        sigma_sq = 1

        # Initialize acceptance cound for determining the acceptance probability
        acceptance_count = 0

        # Optimal acceptance probability
        p_optimal = 0.45

        # Begin sampling
        for i in range(samples):
            # Sample from proposal distribution given var_epsilon
            epsilon = stats.norm.rvs(scale=var_epsilon)

            # Propose new value for alpha given epsilon
            alpha_star = alpha + epsilon
            
            # Predicted heat flux at proposed value
            q_hat_star = energy_storage(self.T_f, self.T_r, self.time, alpha=alpha_star)

            # Log ratio of posteriors, to make computation tractable
            log_beta = -(1/(2*sigma_sq))*(((self.q_obs - q_hat_star)**2).sum() - ((self.q_obs - q_hat)**2).sum())

            # Ratio of posteriors, \beta = \frac{p(\alpha^{star}|data)}{p(\alpha | data)}
            beta = np.exp(log_beta)
            
            # Determine acceptance of proposed value
            if np.random.uniform() < np.minimum(1, beta):
                # Set proposed values
                alpha = alpha_star
                q_hat = q_hat_star
                # Iterate acceptance count
                acceptance_count += 1
            else:
                alpha = alpha
                q_hat = q_hat

            # Tune variance of proposal distribution every 100 steps
            if (i+1) % 100 == 0:
                # Calculates the current acceptance probability
                p_accept = acceptance_count/i

                # New var_epsilon = var_epsilon \frac{\Phi^{-1}(p_{opt}/2)}{\Phi^{-1}(p_{cur}/2)}
                var_epsilon = var_epsilon * (stats.norm.ppf(p_optimal/2)/stats.norm.ppf(p_accept/2))

            # Perform Gibbs sampling step on \sigma^2
            # sigma_sq | data \sim IG(N/2, \frac{1}{2} \sum_{i=1}^N (q_{inc,i} - \hat{q}_{inc,i})^2)
            sigma_sq = stats.invgamma.rvs(len(self.q_obs)/2, 1/(0.5*((self.q_obs - q_hat)**2).sum()))

            # Append traces
            alpha_trace.update_trace(alpha)
            sigma_trace.update_trace(sigma_sq.copy())

        # Save traces
        alpha_trace.save_trace(out_directory=trace_directory)
        sigma_trace.save_trace(out_directory=trace_directory)

        return alpha_trace, sigma_trace
