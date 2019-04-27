#!/packages/python/anaconda3/bin python

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os, sys, time
import pandas as pd
from dft_esm import energy_storage
from dft_statistical_models import Models
from scipy.signal import savgol_filter
from scipy.optimize import minimize

input_directory = '../data/smoothed/5_kw_m2/'
in_file = '1903-02_05.csv', '1903-03_05.csv', '1903-04_05.csv', '1903-05_05.csv', '1903-06_05.csv', '1903-07_05.csv', '1903-08_05.csv', '1903-09_05.csv', '1903-10_05.csv', '1903-12_05.csv', '1903-13_05.csv', '1903-15_05.csv', '1903-16_05.csv', '1903-17_05.csv', '1903-18_05.csv', '1903-19_05.csv'

# Read in all data
all_data = []

for i, item in enumerate(in_file):
    all_data.append(pd.read_csv(input_directory+item)[:420])

# Set time array
all_time = all_data[0].time

### Take the mean of the temperatures across all the data
# Front mean temperature
Tfm = np.mean([all_data[i].tc_1 for i in range(len(all_data))], axis=0)
# Rear mean temperature
Trm = np.mean([all_data[i].tc_2 for i in range(len(all_data))], axis=0)

dat_index = 0
# Read in one file from list of data set, keep only first 420 seconds
data = pd.read_csv(input_directory+in_file[dat_index])[:420]

# Instantiate an array for the "assumed, observed" heat flux
q_obs = np.zeros(len(data.time))
# Set elements in array from minute 1 to minute 6 at 5 kW/m^2
q_obs[60:360] =5

# Initialize stats models
DFT = Models(data.tc_1, data.tc_2, data.time, q_obs)
DFT_all = Models(Tfm, Trm, data.time, q_obs)

DFT.metropolis(0.4, 1)

# # Get MLE
# k_hat = DFT.mle(0.01)
# all_k_hat = DFT_all.mle(0.01)

# # Evaluate model at optimized parameter
# q_hat = energy_storage(data.tc_1, data.tc_2, data.time, alpha=k_hat)
# all_q_hat = energy_storage(Tfm, Trm, all_time, alpha=all_k_hat)

# # Plot results
# plt.figure()
# plt.plot(data.time, q_obs, label='Observed')
# plt.plot(data.time, q_hat, label='Predicted '+str(dat_index))
# plt.plot(data.time, energy_storage(data.tc_1, data.tc_2, data.time), label='uncalibrated')
# plt.plot(data.time, all_q_hat, label='Mean Predicted')
# plt.xlim((0,420))
# plt.xlabel('Time (s)')
# plt.ylabel('Heat Flux (kW/m$^2$)')
# plt.legend(loc=0)
# plt.show()