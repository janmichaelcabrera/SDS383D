from __future__ import division
import numpy as np 
from numpy.linalg import inv
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('../../scripts/')
from linear_models import linear_model

data = pd.read_csv('../../data/gdpgrowth.csv', delimiter=',') # Import data

Y = data['GR6096'] # Y
X = data['DEF60'] # X without intercept feature 

intercept = np.ones(len(X)) # create intercept feature column
X = np.transpose(np.array((intercept, X))) # build matrix from intercept and X features

k = np.ones(X.shape[1])*0.1
K = np.diag(k) # K, precision matrix for multivariate normal prior on beta

#### Informed Bayesian Model
informed = linear_model(X, Y, K) # Pass feature matrix, response vector, and precision matrix to create linear model object
m_star_1 = informed.bayesian() # Calculate linear model intercept and slope using the bayesian method


#### Uninformed Bayesian Model
uninformed = linear_model(X, Y) # Pass feature matrix and response vector to create linear_model object
m_star_2 = uninformed.bayesian() # Calculate linear model intercept and slope using the bayesian method
beta_hat = uninformed.frequentist() # Calculate linear model intercept and slope using an ordinary least squared method

x = np.linspace(X[:,1].min(), X[:,1].max()) # vector for plotting purposes

#### Model responses
y1 = m_star_1[0] + x * m_star_1[1] # Informed Bayesian Linear Model
y2 = m_star_2[0] + x * m_star_2[1] # Uninformed Bayesian Linear Model
y3 = beta_hat[0] + x * beta_hat[1] # Ordinary Least Squares Linear Model

#### Plot models
plt.figure()
plt.plot(X[:,1], Y, '.k', label='GDP Growth Rate Vs. Defense Spending')
plt.plot(x, y1, '-b', label='Informed Bayesian Linear Model')
plt.plot(x, y2, '-r', label='Uninformed Bayesian Linear Model')
plt.plot(x[0:-1:2], y3[0:-1:2], '*g', label='Ordinary Least Squares Linear Model')
plt.xlabel('Defense Spending')
plt.ylabel('GDP Growth Rate')
plt.legend(loc=0)
plt.savefig('figures/bayesian_linear_model.pdf')