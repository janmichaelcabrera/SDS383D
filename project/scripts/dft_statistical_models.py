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
    def __init__(self, T_f, T_r, time, q_obs):
        self.T_f = T_f
        self.T_r = T_r
        self.time = time
        self.q_obs = q_obs


    def mle(self, k_init):
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
            return ((q_pred - q_obs)**2).sum()

        # Minimize loss function
        res = minimize(func, k_init, args=(self.q_obs, self..time, self.T_f, self.T_r))

        # Optimized parameter
        return res.x
