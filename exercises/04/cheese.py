from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append('../../scripts/')
from samplers import Trace
from samplers import Gibbs
import scipy.stats as stats
from numpy.linalg import inv

np.random.seed(3)

# Import data
df = pd.read_csv('../../data/cheese.csv', delimiter=',')

# Create an array of unique store names
stores = np.unique(df.store)

# Number of data rows
n = 5555
p = 4
d = 10

# Instantiate lists for storing sorted data
data = []
y = []
X = []

# Sort data by store
for s, store in enumerate(stores):
    data.append(df[df.store==store])

    # \bar{y}_i
    y.append(np.log(data[s].vol))
    data[s].vol = np.log(data[s].vol)
    data[s].price = np.log(data[s].price)

    # X_i^T
    X.append(np.array([np.ones(data[s].shape[0]), data[s].price, data[s].disp, data[s].price*data[s].disp]).T)

iterations = 5000

# Wrapper for running the model
def run_model(X, y, iterations):
    model = Gibbs(X, y, samples=iterations)
    model.cheese()

run_model(X, y, iterations)

# Load trace
burn=100
beta_trace = np.load('traces/cheese/beta_trace.npy')
beta_trace_mean = np.mean(beta_trace[burn:], axis=0)

# Plot all stores together into one plot
fig, ax = plt.subplots(11, 8, figsize=(10,12))
fig.subplots_adjust(hspace=0.6)
n = 0
for i in range(11):
    for j in range(8):
        x_hat = X[n][X[n][:,1].argsort()]
        ax[i,j].plot(data[n][data[n].disp==0].price, data[n][data[n].disp==0].vol, '.k', linewidth=0.1)
        ax[i,j].plot(data[n][data[n].disp==1].price, data[n][data[n].disp==1].vol, '.r', linewidth=0.1)
        ax[i,j].plot(x_hat[:,1], beta_trace_mean[n][0]+beta_trace_mean[n][1]*x_hat[:,1], '-k')
        ax[i,j].plot(x_hat[:,1], (beta_trace_mean[n][0]+beta_trace_mean[n][2])+(beta_trace_mean[n][1]+beta_trace_mean[n][3])*x_hat[:,1], '-r')
        n+=1

# plt.show()
plt.savefig('figures/cheese_all_plots.pdf')

# Plot selected stores
i = 10, 41, 55
for p, plots in enumerate(i):
    plt.figure()
    x_hat = X[plots][X[plots][:,1].argsort()]
    plt.plot(data[plots][data[plots].disp==0].price, data[plots][data[plots].disp==0].vol, '.k', linewidth=2)
    plt.plot(data[plots][data[plots].disp==1].price, data[plots][data[plots].disp==1].vol, '.r', linewidth=2)
    plt.plot(x_hat[:,1], beta_trace_mean[plots][0]+beta_trace_mean[plots][1]*x_hat[:,1], '-k', linewidth=2)
    plt.plot(x_hat[:,1], (beta_trace_mean[plots][0]+beta_trace_mean[plots][2])+(beta_trace_mean[plots][1]+beta_trace_mean[plots][3])*x_hat[:,1], '-r', linewidth=2)
    # plt.show()
    plt.savefig('figures/cheese_plots_'+str(plots)+'.pdf')