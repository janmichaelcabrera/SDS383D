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
    """
    Inputs
    ----------
        x: scalar, or vector
            temperature in K

    Returns
    ----------
        Heat transfer coefficient (W/m^2K) evaluated at the film temperature

    """
    y = 4.753E-12*x**5 - 1.303E-08*x**4 + 1.418E-05*x**3 - 7.675E-03*x**2 + 2.076E+00*x - 2.145E+02
    return y

#### DFT Thermo-physical properties

def rCp_s(x):
    """
    Inputs
    ----------
        x: scalar, or vector
            temperature in K

    Returns
    ----------
        Rho times specific heat of steel (kg J/m^3-K)

    """
    y = 5.535E-09*x**5 - 2.671E-05*x**4 + 4.978E-02*x**3 - 4.469E+01*x**2 + 2.041E+04*x + 5.125E+05
    return y

def rCp_c(x):
    """
    Inputs
    ----------
        x: scalar, or vector
            temperature in K

    Returns
    ----------
        Rho times specific heat of ceramic blanket (kg J/m^3-K)

    """
    y = -9.517E-08*x**4 + 3.309E-04*x**3 - 4.394E-01*x**2 + 3.211E+02*x + 8.151E+04
    return y
    
def k_c(x, alpha=None):
    """
    Inputs
    ----------
        x: scalar, or vector
            temperature in K

    Returns
    ----------
        Thermal conductivity of insulation (W/mK)

    """    
    if alpha[0] == None:
        # Cerablanket
        y = 7.36E-17*x**5 - 3.02E-13*x**4 + 4.87E-10*x**3 - 2.35E-07*x**2 + 1.43E-04*x + 3.11E-03
        # INSWOOL
        # y = -2E-10*x**3 + 6E-07*x**2 - 0.0002*x + 0.0755
    else:
        # alpha = np.asarray(alpha)
        # Temp = np.zeros((len(alpha), len(x.copy())))
        # for p in range(len(alpha)):
        #     Temp[p] = x.copy()**p
        # y = alpha @ Temp
        y = alpha[0] + alpha[1]*x
    return y

def energy_storage(X, alpha=[None]):
    """
    Inputs
    ----------
        X: list
            Wrapper for Tf, Tr, my_times

        Tf: vector
            Front temperature of DFT in deg C

        Tr: vector
            Rear temperature of DFT in deg C

        my_times: vector
            Experimental times corresponding to temperatures in seconds

        alpha: scalar (optional)
            Thermal conductivity of cerablanket (W/m-K). Default is None, so that the model uses the curve fit above to fit the model. The value can be changed and optimized over if this variable is used.

    Returns
    ----------
        q_inc: vector
            Incident heat flux to DFT using Energy Storage Method (ASTM E3057) in kW/m^2

    """
    ### Unpack model inputs
    Tf, Tr, my_times = X

    ### DFT geometry

    # Plate thickness (m)
    L_p = 0.001905
    # Insulation thickness (m_)
    l_i = (7./8.) * 0.0254

    # Absorptivity of plate
    absorp = 0.9
    # Emissivity of plate
    epsilon = 0.9
    # Stefan-Boltzman constant
    sigma = 5.6704E-8 # W/m2K4

    # Calcs insulation temperature as average of front and back temperatures
    T_ins = (Tf + Tr)/2

    # Film Temperatures
    T_ff = (Tf + Tf[0])/2 # film temp on front plate
    T_fr = (Tr + Tr[0])/2 # film temp on rear plate

    # Evalutes ESM at Tf and Tr
    q_net = rCp_s(Tf+273)*L_p*np.gradient(Tf, my_times) + k_c(T_ins+273, alpha=alpha)*(Tf-Tr)/l_i+rCp_c(T_ins+273)*l_i*np.gradient(T_ins, my_times)

    # Reflected heat flux loss
    q_ref = (1 - absorp)*sigma*((Tf+273)**4 - (Tf[0]+273)**4)
    # Emitted heat flux loss
    q_emit = epsilon*sigma*((Tf+273)**4 - (Tf[0]+273)**4) + epsilon*sigma*((Tr+273)**4 - (Tr[0]+273)**4)
    # Convective heat flux loss
    q_conv = h_p(T_ff+273)*(Tf - Tf[0]) + h_p(T_fr+273)*(Tr - Tr[0])
    # Incident heat flux
    q_inc = q_net/epsilon + q_emit/epsilon + q_conv/epsilon

    return q_inc/1000