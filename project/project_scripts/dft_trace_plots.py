#!/packages/python/anaconda3/bin python

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import os, sys, time
import pandas as pd
from dft_esm import energy_storage
from scipy.optimize import minimize
import scipy.stats as stats
sys.path.append('../../scripts/')
from metropolis_hastings import Models

model_name = 'dft_5_kwm2_2'

burn = 1000
# Load traces
alpha_trace = np.load('traces/'+model_name+'_alpha_trace.npy')[burn:]
sigma_trace = np.load('traces/'+model_name+'_sigma_trace.npy')[burn:]

rho = stats.pearsonr(alpha_trace[:,0], alpha_trace[:,1])
print(rho)

plt.figure()
plt.scatter(alpha_trace[:,0], alpha_trace[:,1], color='black')
plt.xlabel(r'$k^{(2)}_0$')
plt.ylabel(r'$k^{(2)}_1$')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# plt.show()
plt.savefig('../figures/alpha_correlation.pdf')