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

#### Air Thermo-physical properties

def rho_air(T):
    a_0 = 357.3970321025
    b = -1.0038568629

    return a_0*T**b

def mu_air(T):
    a_0 = 3.2575*10**-5
    a_1 = -7.56163*10**-7
    a_2 = 7.47253*10**-9
    a_3 = -2.86507*10**-11
    a_4 = 3.89067*10**-14

    return a_0 + a_1*T + a_2*T**2 + a_3*T**3 + a_4*T**4

def k_air(T):
    a_0 = -3.06*10**-4
    a_1 = 9.89089*10**-5
    a_2 = -3.46571*10**-8
    a_3 = 0
    a_4 = 0

    return a_0 + a_1*T + a_2*T**2 + a_3*T**3 + a_4*T**4

def cp_air(T):
    a_0 = 1.1788
    a_1 = -2.8765*10**-3
    a_2 = 1.8105*10**-5
    a_3 = -5.1*10**-8
    a_4 = 5.4*10**-11

    return a_0 + a_1*T + a_2*T**2 + a_3*T**3 + a_4*T**4

def h_nat(Ts, T_inf, L=7.5*10**(-2)):
    g = 9.81 # m/s^2
    alpha = k_air(Ts+273)/(rho_air(Ts+273)*cp_air(Ts+273)) # m**2 /s
    nu = mu_air(Ts+273)/rho_air(Ts+273)
    beta = 1/(Ts+273)
    Ra = (g*beta*(Ts - T_inf)*L**3)/(nu*alpha)
    Nu = 0.54*Ra**(1/4)
    h = (k_air(Ts+273)/L)*Nu
    return h

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
    # q_conv = h_nat(Tf, Tf[0])*(Tf - Tf[0]) + h_nat(Tr, Tr[0])*(Tr - Tr[0])

    # q_inc = q_net + q_ref + q_emit + q_conv
    q_inc = q_net + q_ref + q_emit + q_conv

    return q_inc/1000