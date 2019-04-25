#!/packages/python/anaconda3/bin python

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os, sys, time
import pandas as pd
from scipy.signal import savgol_filter

###############################
########## Functions ##########
###############################

def h_p(x):
    # Heat transfer coefficient (W/m^2K) evaluated at the film temperature
    y = 4.753E-12*x**5 - 1.303E-08*x**4 + 1.418E-05*x**3 - 7.675E-03*x**2 + 2.076E+00*x - 2.145E+02
    return y

#### DFT Thermo-physical properties

def rCp_s(x):
    # rho * cp steel (J/m^3K)
    y = 5.535E-09*x**5 - 2.671E-05*x**4 + 4.978E-02*x**3 - 4.469E+01*x**2 + 2.041E+04*x + 5.125E+05
    return y

def rCp_c(x):
    # rho * cp cerablanket (J/m^3K)
    y = -9.517E-08*x**4 + 3.309E-04*x**3 - 4.394E-01*x**2 + 3.211E+02*x + 8.151E+04
    return y
    
def k_c(x, alpha=None, beta=None):
    if (alpha or beta) == None:
        # conductivity of cerablanket (W/mK)
        y = 7.36E-17*x**5 - 3.02E-13*x**4 + 4.87E-10*x**3 - 2.35E-07*x**2 + 1.43E-04*x + 3.11E-03
        # INSWOOL
        # y = -2E-10*x**3 + 6E-07*x**2 - 0.0002*x + 0.0755
    else:
        y = alpha*x + beta
    return y

def energy_storage(Tf, Tr, my_times, alpha=None, beta=None):

    # Geometry
    L_p = 0.001905         #thickness of plate (m)
    l_i = (7./8.) * 0.0254 #thickness of insulation (m)
    absorp = 0.9
    epsilon = 0.9
    # Stefan-Boltzman constant
    sigma = 5.6704E-8 # W/m2K4

    T_ins = (Tf + Tr)/2

    q_net = rCp_s(Tf+273)*L_p*np.gradient(Tf, my_times) + k_c(T_ins+273, alpha=alpha, beta=beta)*(Tf-Tr)/l_i+rCp_c(T_ins+273)*l_i*np.gradient(T_ins, my_times)

    q_ref = (1 - absorp)*sigma*((Tf+273)**4 - (Tf[0]+273)**4)
    q_emit = epsilon*sigma*((Tf+273)**4 - (Tf[0]+273)**4) + epsilon*sigma*((Tr+273)**4 - (Tr[0]+273)**4)
    q_conv = h_p(T_ins+273)*(Tf - Tf[0]) + h_p(T_ins+273)*(Tr - Tr[0])

    q_inc = q_net + q_ref + q_emit + q_conv

    return q_inc/1000