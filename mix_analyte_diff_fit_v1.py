# %%
from itertools import islice

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert, find_peaks
import pylab as p
import scipy.signal as sci
import scipy.stats as stats
from scipy.optimize import minimize
from scipy.optimize import curve_fit
import math
from math import pi
from scipy.interpolate import interp1d, UnivariateSpline
from sklearn.metrics import mean_squared_error
from collections import namedtuple
import itertools


# import ste_model_spectrum.py

from ste_model_spectrum import *

res = 964
dim_s = 100

# ------------------------------------------------------------------------------
import shelve

filename = 'shelve_save_data_analyte.out'

my_shelf = shelve.open(filename)
for key in my_shelf:
    globals()[key] = my_shelf[key]
my_shelf.close()

# ------------------------------------------------------------------------------
# import the clean ACE spectrum

i_file = open('data/Acephate_Exported.dat', "r")
temp = i_file.readlines()
temp = [i.strip('\n') for i in temp]
dataACE = np.array([(row.split('\t')) for row in temp], dtype=np.float32)
f_ACE = dataACE[:, 0]
dataACE = dataACE[:, 1]

# ------------------------------------------------------------------------------
f_sup = f_sup[:-1]
data_mean = np.mean(data, axis=2)
# ------------------------------------------------------------------------------
# align the data

#  what's the shift?
align_init = np.argmin((f_ACE - f_sup[0]) ** 2)
align_end = np.argmin((f_ACE - f_sup[-1]) ** 2)

# f_ACE[47]
# f_ACE[48]
# f_sup[0]
# f_sup[1]

len_temp = f_sup.shape[0]
data_ace_temp = np.zeros(len_temp)
n_feq = align_init

s1 = f_sup.copy()

for i in range(0, len_temp - 1):
    pos = np.where((f_ACE >= f_sup[i]) & (f_ACE <= f_sup[i + 1]))
    data_ace_temp[i] = np.mean(dataACE[pos])

# last point just guess
data_ace_temp[-1] = dataACE[align_end]

# # just align at the index 47
# dataACE[align_init:align_end]

# f_ACE[align_init:align_end].shape
# f_sup.shape

# update the data mean

data_mean[0] = data_ace_temp

# ------------------------------------------------------------------------------

#plt.figure('raw data')
#labels = ['ACE', 'MG', 'mix1_1', 'mix1_2', 'mix2_1', 'mix2_2']

#for data_mean, label in zip(data_mean, labels):
#    plt.plot(f_sup, data_mean, label=label)

#plt.legend()
#plt.show()

data_mean_ace = np.copy(data_mean[0])
data_mean_mg = np.copy(data_mean[1])
data_mean_ace_mix_1 = np.copy(data_mean[2])
data_mean_ace_mix_2 = np.copy(data_mean[3])

data_mean_ace_smoothed = sci.savgol_filter(data_mean_ace, 101, 2)
data_mean_ace_mix_1_smoothed = sci.savgol_filter(data_mean_ace_mix_1, 101, 2)
data_mean_ace_mix_2_smoothed = sci.savgol_filter(data_mean_ace_mix_2, 101, 2)
data_mean_mg_smoothed = sci.savgol_filter(data_mean_mg, 101, 2)

###<--------------Calculate neccessary erros for comparison-----------------------------
# error_ace is responsible for determening the position of the point (below or above predicted curve)
# arror_ace_abs is used alongside rmse_ace(root_mean_square_error of ace) to determine if the peak is strong enough

#ACE errors
error_ace = data_mean_ace - data_mean_ace_smoothed
error_ace_abs = np.abs(error_ace)
rmse_ace = np.sqrt(mean_squared_error(data_mean_ace, data_mean_ace_smoothed))
#ACE mix1 & mix2 errors
error_ace_mix_v1 = data_mean_ace_mix_1 - data_mean_ace_mix_1_smoothed
error_ace_abs_mix_v1 = np.abs(error_ace_mix_v1)
rmse_ace_mix_v1 = np.sqrt(mean_squared_error(data_mean_ace_mix_1, data_mean_ace_mix_1_smoothed))

error_ace_mix_v2 = data_mean_ace_mix_2 - data_mean_ace_mix_2_smoothed
error_ace_abs_mix_v2 = np.abs(error_ace_mix_v2)
rmse_ace_mix_v2 = np.sqrt(mean_squared_error(data_mean_ace_mix_2, data_mean_ace_mix_2_smoothed))
#MG errors
error_mg = data_mean_mg - data_mean_mg_smoothed
error_mg_abs = np.abs(error_mg)
rmse_mg = np.sqrt(mean_squared_error(data_mean_mg, data_mean_mg_smoothed))
###<------------------------------------------------------------------------------------

#<-----------------Detecting points that are above interpolated savgol_filter curve-----
Peak = namedtuple('Peak', ['index_pos', 'value'])
temp_list = []  # Create temprorary namedtuple to store position of the peak and value of the peak
for index, (value, value_error, value_error_abs) in enumerate(zip(data_mean_ace_mix_1, error_ace_mix_v1, error_ace_abs_mix_v1)):
    if (value_error_abs > rmse_ace_mix_v1) and value_error > 0:
        temp_peak = Peak(index, value)
        temp_list.append(temp_peak)

peaks_list = []
for index, peak in enumerate(islice(temp_list, len(temp_list) - 1)):
    if (peak.index_pos - temp_list[index + 1].index_pos) == -1 and peak.value > temp_list[index + 1].value:
        if peak.value < temp_list[index - 1].value:
            pass
        else:
            peaks_list.append(peak)
#<-------------------------------------------------------------------------------------

#<-----------------Converting peaks indeces to frequency values-------------------------
peak_frequency = np.zeros(len(peaks_list))
peak_values = np.zeros(len(peaks_list))
for index, (value_peak, frequency) in enumerate(zip(peaks_list, f_sup)):
    peak_frequency[index] = f_sup[value_peak.index_pos]
    peak_values[index] = value_peak.value
#<--------------------------------------------------------------------------------------
#<-----------------Detecting points that are above interpolated savgol_filter curve-----
Peak_1 = namedtuple('Peak', ['index_pos', 'value'])
temp_list = []  # Create temproray namedtuple to store position of the peak and value of the peak
for index, (value, value_error, value_error_abs) in enumerate(zip(data_mean_ace, error_ace, error_ace_abs)):
    if (value_error_abs > rmse_ace) and value_error > 0:
        temp_peak = Peak_1(index, value)
        temp_list.append(temp_peak)

peaks_list_1 = []
for index, peak in enumerate(islice(temp_list, len(temp_list) - 1)):
    if (peak.index_pos - temp_list[index + 1].index_pos) == -1 and peak.value > temp_list[index + 1].value:
        if peak.value < temp_list[index - 1].value:
            pass
        else:
            peaks_list_1.append(peak)
#<-------------------------------------------------------------------------------------

#<-----------------Converting peaks indeces to frequency values-------------------------
peak_frequency_1 = np.zeros(len(peaks_list_1))
peak_values_1 = np.zeros(len(peaks_list_1))
for index, (value_peak, frequency) in enumerate(zip(peaks_list_1, f_sup)):
    peak_frequency_1[index] = f_sup[value_peak.index_pos]
    peak_values_1[index] = value_peak.value
#<--------------------------------------------------------------------------------------

#<-----------------Detecting points that are above interpolated savgol_filter curve-----
Peak_2 = namedtuple('Peak', ['index_pos', 'value'])
temp_list = []  # Create temproray namedtuple to store position of the peak and value of the peak
for index, (value, value_error, value_error_abs) in enumerate(zip(data_mean_mg, error_mg, error_mg_abs)):
    if (value_error_abs > rmse_mg) and value_error > 0:
        temp_peak = Peak_2(index, value)
        temp_list.append(temp_peak)

peaks_list_2 = []
for index, peak in enumerate(islice(temp_list, len(temp_list) - 1)):
    if (peak.index_pos - temp_list[index + 1].index_pos) == -1 and peak.value > temp_list[index + 1].value:
        if peak.value < temp_list[index - 1].value:
            pass
        else:
            peaks_list_2.append(peak)
#<-------------------------------------------------------------------------------------

#<-----------------Converting peaks indeces to frequency values-------------------------
peak_frequency_2 = np.zeros(len(peaks_list_2))
peak_values_2 = np.zeros(len(peaks_list_2))
for index, (value_peak, frequency) in enumerate(zip(peaks_list_2, f_sup)):
    peak_frequency_2[index] = f_sup[value_peak.index_pos]
    peak_values_2[index] = value_peak.value
#<--------------------------------------------------------------------------------------

plt.figure('ACE & mix1')
plt.plot(f_sup, data_mean_ace, '-', label='ACE')
plt.plot(f_sup, data_mean_mg, '-', label='MG')
plt.plot(f_sup, data_mean_ace_smoothed, '-', label='ACE-smooth')
plt.plot(f_sup, data_mean_mg_smoothed, '-', label='MG-smooth')
plt.plot(peak_frequency, peak_values, '*', label='Peaks ace clear')
plt.plot(f_sup, data_mean_ace_mix_1, '-', label='ACE mix1')
plt.plot(f_sup, data_mean_ace_mix_1_smoothed, '-', label='ACE-smooth  mix')
plt.plot(peak_frequency_1, peak_values_1, '*', label='Peaks mix1')
plt.plot(peak_frequency_2, peak_values_2, '*', label='Peaks MG')
plt.legend()
interpolation_of_mg = interp1d(f_sup, data_mean, axis=0, fill_value='extrapolate')
x_new = np.arange(0, len_temp, 0.6025)
y_new = interpolation_of_mg(x_new)
plt.figure('MG and Interpolation')
plt.plot(f_sup, data_mean[1])#, '-', x_new, y_new, '*')
plt.show()
