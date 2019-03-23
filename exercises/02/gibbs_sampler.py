from __future__ import division
import numpy as np 
from numpy.linalg import inv
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys
sys.path.append('../../scripts/')
from linear_models import linear_model
from samplers import Gibbs, Trace

data = pd.read_csv('../../data/gdpgrowth.csv', delimiter=',')

Y = data['GR6096'] # Y 
X = data['DEF60'] # X without intercept feature

intercept = np.ones(len(Y)) # create intercept feature column
X = np.transpose(np.array((intercept, X))) # build matrix from intercept and X features

#### Heavy Tailed Error Model
model = Gibbs(X, Y, samples=5000) # instantiate the model from the Gibbs class

beta_trace, omega_trace, Lambda_trace = model.heavy_tailed() # run the heavy tailed error model

omega_trace.plot('figures/') # plots omega trace
beta_trace.plot('figures/') # plots beta traces

b_0, b_1 = beta_trace.mean() # returns the means from each beta trace

#### Informed Bayesian Model
k = np.ones(X.shape[1])*0.1
K = np.diag(k) # K, precision matrix for multivariate normal prior on beta

informed = linear_model(X, Y, K) # Pass feature matrix, response vector, and precision matrix to create linear model object
m_star_1 = informed.bayesian() # Calculate linear model intercept and slope using the bayesian method


#### Model responses
x = np.linspace(X[:,1].min(), X[:,1].max()) # vector for plotting purposes
y1 = m_star_1[0] + x * m_star_1[1] # Informed Bayesian Linear Model
y2 = b_0 + x * b_1 # Heavy Tailed Error Model

plt.figure()
plt.plot(X[:,1], Y, '.k', label='GDP Growth Rate Vs. Defense Spending')
plt.plot(x, y1, '-b', label='Informed Bayesian Linear Model')
plt.plot(x, y2, '-r', label='Heavy Tailed Error Model')
plt.xlabel('Defense Spending')
plt.ylabel('GDP Growth Rate')
plt.legend(loc=0)
plt.savefig('figures/heavy_tailed.pdf')