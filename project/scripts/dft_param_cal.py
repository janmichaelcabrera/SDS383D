#!/packages/python/anaconda3/bin python

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os, sys, time
import pandas as pd
from dft_esm import energy_storage
from scipy.signal import savgol_filter
from smooth import gp_smooth
import pymc3 as pm

input_directory = '../calibration/redux/smoothed/5_kw_m2/'
in_file = '1903-02_05.csv', '1903-03_05.csv', '1903-04_05.csv', '1903-05_05.csv', '1903-06_05.csv', '1903-07_05.csv', '1903-08_05.csv', '1903-09_05.csv', '1903-10_05.csv', '1903-12_05.csv', '1903-13_05.csv', '1903-15_05.csv', '1903-16_05.csv', '1903-17_05.csv', '1903-18_05.csv', '1903-19_05.csv'

data = []
Tf = []
Tr = []

for i, item in enumerate(in_file):
    data.append(pd.read_csv(input_directory+item,index_col=None))
    Tf.append(data[i].tc_1.values)
    Tr.append(data[i].tc_2.values)

time = data[0].time.values[:420]
q = np.zeros(len(time))
q[60:360] = 5

num_data = len(data)

alpha, beta = None, None


q_obs = np.array([q for n in range(10)])

Tfm = np.mean([Tf[n][:420] for n in range(num_data)], axis=0)
Trm = np.mean([Tr[n][:420] for n in range(num_data)], axis=0)

with pm.Model() as model:
    # Priors
    alpha = pm.Uniform('alpha', lower=-1, upper=1)
    beta = pm.Uniform('beta', lower=-5, upper=5)
    sigma = pm.HalfNormal('sigma', sd=1000)

    q_pred = energy_storage(Tfm, Trm, time, alpha=alpha, beta=beta)

    y_obs = pm.Normal('y_obs', mu=q_pred, sd=sigma, observed=q_obs)

    trace = pm.sample(2000)
    pm.traceplot(trace)
    plt.show()

alpha_low = pm.hpd(trace['alpha'])[0]
alpha_high = pm.hpd(trace['alpha'])[1]

beta_low = pm.hpd(trace['beta'])[0]
beta_high = pm.hpd(trace['beta'])[1]

alpha, beta = trace['alpha'].mean(), trace['beta'].mean()
print(alpha, beta)

q_low = energy_storage(Tfm, Trm, time, alpha=alpha_low, beta=beta_low)
q_mean = energy_storage(Tfm, Trm, time, alpha=alpha, beta=beta)
q_high = energy_storage(Tfm, Trm, time, alpha=alpha_high, beta=beta_high)



plt.figure()
plt.plot(time, q, '-g', label='Calibration Set Point')
plt.plot(time, q_mean, '-b', label='Mean')
plt.fill_between(x=time, y1=q_low, y2=q_high, alpha=0.4, color='b')
plt.xlabel('Time (s)')
plt.ylabel('Heat Flux (kW/m$^2$)')
plt.legend(loc=0)
plt.show()



# q_pred = np.array([energy_storage(Tf[n], Tr[n], data[n].time.values, alpha=alpha, beta=beta) for n in range(num_data)]).T

# print(q_pred)

# with pm.Model() as model:
#     alpha = pm.Uniform('alpha', lower=-1, upper=1)
#     beta = pm.Uniform('beta', lower=-5, upper=5)
#     sigma = pm.HalfNormal('sigma', sd=1000)

#     q_pred = np.array([energy_storage(Tf[0], Tr[0], data[0].time.values, alpha=alpha, beta=beta)])

#     y_obs = pm.Normal('y_obs', mu=q_pred, sd=sigma, observed=q)

#     trace = pm.sample(100)
#     pm.traceplot(trace)
#     plt.show()


# plt.figure()
# plt.plot(data[0].time, energy_storage(Tf[0], Tr[0], data[0].time))
# plt.plot(data[1].time, energy_storage(Tf[1], Tr[1], data[1].time))
# plt.plot(data[0].time, q)
# plt.show()