from __future__ import division
import numpy as np 
from numpy.linalg import inv
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('../../data/gdpgrowth.csv', delimiter=',')

Y = data['GR6096'] # X without intercept feature
X = data['DEF60'] # Y 

intercept = np.ones(len(Y)) # create intercept feature column
X = np.transpose(np.array((intercept, X))) # build matrix from intercept and X features

k = np.ones(2)*0.1
K = np.diag(k) # K

m = np.zeros(2)

Lambda = np.eye(len(X))

pre = inv(K + np.matmul(np.matmul(np.transpose(X), Lambda),X))

m_star = np.matmul(pre,(np.matmul(K, m) + np.matmul(np.transpose(X), Lambda).dot(Y)))

x = np.linspace(X[:,1].min(), X[:,1].max())

y = m_star[0] + x * m_star[1]

plt.figure()
plt.plot(X[:,1], Y, '.')
plt.plot(x, y)
plt.show()