from __future__ import division
import numpy as np 
import pandas as pd 
import time, os
from datetime import datetime
import matplotlib.pyplot as plt
from dft_esm import energy_storage
import sys
sys.path.append('../../scripts/')
from gaussian_process import gaussian_process

def smooth_unsmoothed(input_directory, output_directory):
    """
    Example
    ----------
    input_directory = '../data/unsmoothed/15_kw_m2/'
    output_directory = '../data/smoothed/15_kw_m2/'

    smooth_unsmoothed(input_directory, output_directory)
    """
    for in_file in os.listdir(input_directory):
        f = in_file.split('.')
        if 'csv' == f[1]:
            if in_file not in os.listdir(output_directory):
                data = pd.read_csv(input_directory+in_file)
                hyperparams = 10, 10, 10**-6
                TC1 = gaussian_process(data.time.values, hyperparams, y=data.tc_1.values, x_star=data.time.values)
                var = TC1.approx_var()
                TC1.optimize_lml()

                tc1_star, tc1_var = TC1.smoother(variance=var)

                TC2 = gaussian_process(data.time.values, hyperparams, y=data.tc_2.values, x_star=data.time.values)
                var = TC2.approx_var()
                TC2.optimize_lml()

                tc2_star, tc2_var = TC2.smoother(variance=var)

                data['tc_1'] = tc1_star
                data['tc_2'] = tc2_star

                data.to_csv(output_directory+in_file)
            else:
                print('File already exists')

def plot_residuals(input_directories, output_directory, show=False):

    """
    Example
    ----------
        unsmoothed_directory = '../data/unsmoothed/15_kw_m2/'
        smoothed_directory = '../data/smoothed/15_kw_m2/'

        plotting_directory = '../data/residuals/15_kw_m2/'

        input_directories = unsmoothed_directory, smoothed_directory

        plot_residuals(input_directories, plotting_directory, show=False)
    """

    unsmoothed, smoothed = input_directories

    for file in os.listdir(unsmoothed):    
        unsmoothed_data = pd.read_csv(unsmoothed+file)
        smoothed_data = pd.read_csv(smoothed+file)
        residuals_1 = unsmoothed_data['tc_1'] - smoothed_data['tc_1']
        residuals_2 = unsmoothed_data['tc_2'] - smoothed_data['tc_2']
        name = file.split('.')[0]

        plt.figure()
        plt.plot(unsmoothed_data['time'], residuals_1, '.k', label='TC1')
        plt.plot(unsmoothed_data['time'], residuals_2, '.b', label='TC2')
        plt.xlabel('time (s)')
        plt.ylabel('Residuals ($^\\circ$C)')
        plt.legend(loc=0)
        if show == False:
            plt.savefig(output_directory+name+'.pdf')
        else:
            plt.show()
        plt.close()

def plot(input_directory, output_directory, show=False):

    """
    Example
    ----------
        input_directory = '../data/smoothed/15_kw_m2/'
        output_directory = '../data/smoothed/15_kw_m2/plots/'
        plot(input_directory, output_directory, show=False)
    """

    for file in os.listdir(input_directory):

        if file != 'plots':

            data = pd.read_csv(input_directory+file)
            name = file.split('.')[0]

            plt.figure()
            plt.plot(data['time'], data['tc_1'], '-k', label='TC1')
            plt.plot(data['time'], data['tc_2'], '-b', label='TC2')
            plt.xlabel('time (s)')
            plt.ylabel('Temperature ($^\\circ$ C)')
            plt.legend(loc=0)
            if show == False:
                plt.savefig(output_directory+name+'.pdf')
            else:
                plt.show()
            plt.close()

def plot_together(input_directory, output_directory, out_name, show=False):
    """
    Example
    ----------
        input_directory = '../data/smoothed/15_kw_m2/'
        output_directory = '../data/smoothed/15_kw_m2/plots/'
        out_name = '15_kw_m2_together.pdf'
        plot_together(input_directory, output_directory, out_name, show=False)
    """

    plt.figure()
    for file in os.listdir(input_directory):
        if file != 'plots':

            data = pd.read_csv(input_directory+file)
            name = file.split('.')[0]

            plt.plot(data['time'], data['tc_1'], '-k')
            plt.plot(data['time'], data['tc_2'], '-b')
    plt.xlabel('time (s)')
    plt.ylabel('Temperature ($^\\circ$ C)')
    if show == False:
        plt.savefig(output_directory+out_name)
    else:
        plt.show()
    plt.close()