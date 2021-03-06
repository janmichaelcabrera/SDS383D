#!/packages/python/anaconda3/bin python

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import os, sys, time
import pandas as pd
from dft_esm import energy_storage
from scipy.optimize import minimize
sys.path.append('../../scripts/')
from metropolis_hastings import Models

# Directories
input_directory = '../data/smoothed/5_kw_m2/'
in_file = '1903-02_05.csv', '1903-03_05.csv', '1903-04_05.csv', '1903-05_05.csv', '1903-06_05.csv', '1903-07_05.csv', '1903-08_05.csv', '1903-09_05.csv', '1903-10_05.csv', '1903-12_05.csv', '1903-13_05.csv', '1903-15_05.csv', '1903-16_05.csv', '1903-17_05.csv', '1903-18_05.csv', '1903-19_05.csv'

figures_directory = '../figures/'

# Set the data to read in
dat_index = 0

# Read in one file from list of data set, keep only first 420 seconds
data = pd.read_csv(input_directory+in_file[dat_index])[:420]

# Instantiate an array for the "assumed, observed" heat flux
q_obs = np.zeros(len(data.time))

# Set elements in array from minute 1 to minute 6 at 5 kW/m^2
q_obs[60:360] =5

# Wrapper for data inputs to model
X = [data.tc_1, data.tc_2, data.time]

k_init = [-4.1962,1]
model_name = 'dft_5_kwm2_2b'
# Initialize stats models
DFT = Models(model_name, energy_storage, X, q_obs, k_init)

# Get MLE
k_hat = DFT.mle()

# Run Metropolis algorithm
alpha_trace, sigma_trace = DFT.metropolis_random_walk(samples=200, tune_every=5, times_tune=200, cov_scale=10**-2)
print('Acceptance: ', DFT.p_accept)
alpha_trace.save_trace()
sigma_trace.save_trace()

burn = 1000
# Load traces
alpha_trace = np.load('traces/'+model_name+'_alpha_trace.npy')[burn:]
sigma_trace = np.load('traces/'+model_name+'_sigma_trace.npy')[burn:]

# print(alpha_trace[:,1])

np.set_printoptions(precision=3)
print('Thermal conductivity - mean: \n', np.mean(alpha_trace, axis=0))

print('std: \n', np.std(alpha_trace, axis=0))

print('Variance - mean: {:2.2f}, std: {:2.4f}'.format(np.mean(sigma_trace), np.std(sigma_trace)))

k_hat_mh = np.mean(alpha_trace, axis=0)
k_hat_lower = np.min(alpha_trace) # k_hat_mh - np.std(alpha_trace)*3.3
k_hat_upper = np.max(alpha_trace) # k_hat_mh + np.std(alpha_trace)*3.3

### Evaluate model at optimized parameter
# MLE
q_hat_mle = energy_storage(X, alpha=k_hat)

# Metropolis Random Walk and 95% credible interval
q_hat_mh = energy_storage(X, alpha=k_hat_mh)
q_hat_lower = energy_storage(X, alpha=[k_hat_lower])
q_hat_upper = energy_storage(X, alpha=[k_hat_upper])

plot = True
if plot == True:
    # Plot results
    plt.figure()
    for i in range(len(alpha_trace)):
        plt.plot(data.time, energy_storage(X, alpha_trace[i]), color='grey')
    # plt.fill_between(x=data.time, y1=q_hat_lower, y2=q_hat_upper, color='grey')
    plt.plot(data.time, q_obs, '-k', linewidth=1.5, label='Observed')
    plt.plot(data.time, q_hat_mh, '-b', linewidth=1.5, label='MH '+str(dat_index))
    plt.plot(data.time, energy_storage(X), '-r', linewidth=1.5, label='Uncalibrated')
    plt.plot(data.time, q_hat_mle,'--g', linewidth=1.5, label='MLE '+str(dat_index))
    plt.xlim((0,420))
    plt.xlabel('Time (s)')
    plt.ylabel('Heat Flux (kW/m$^2$)')
    plt.legend(loc=0)
    # plt.show()
    plt.savefig(figures_directory+'calibrated_results_'+model_name+'.png', dpi=300)

    # Plot traces and histograms
    for p in range(len(k_init)):
        plt.figure()
        plt.plot(alpha_trace[:,p], color='black')
        # plt.show()
        plt.savefig(figures_directory+'alpha_trace_'+str(p)+'_'+model_name+'.pdf')
        plt.close()

        plt.figure()
        plt.hist(alpha_trace[:,p], color='black', bins=30)
        # plt.show()
        plt.savefig(figures_directory+'alpha_hist_'+str(p)+'_'+model_name+'.pdf')

    plt.figure()
    plt.plot(sigma_trace, color='black')
    # plt.show()
    plt.savefig(figures_directory+'sigma_trace_'+model_name+'.pdf')

    plt.figure()
    plt.hist(sigma_trace, color='black', bins=30)
    # plt.show()
    plt.savefig(figures_directory+'sigma_hist_'+model_name+'.pdf')