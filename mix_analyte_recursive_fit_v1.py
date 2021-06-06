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
data_mean_smoothed = np.zeros((data_mean.shape[0], data_mean.shape[1]))

#<----------------------------------------Smoothe the mean of all data-----------------------------------------
for i in range(0, data_mean.shape[0]):
    data_mean_smoothed[i] = sci.savgol_filter(data_mean[i], 101, 2)
#<=============================================================================================================

#<------------------------Initialize errors for the data and Peak object to store peaks------------------------
Peak = namedtuple('Peak', ['index_pos', 'value'])
errors = data_mean - data_mean_smoothed
errors_abs = np.abs(errors)
rmse = np.zeros(data_mean.shape[0])
for index, (mean, mean_smoothed) in enumerate(zip(data_mean, data_mean_smoothed)):
    rmse[index] = np.sqrt(mean_squared_error(mean, mean_smoothed))
#<=============================================================================================================

#--------------------------Find the peaks----------------------------------------------------------------------
temp_list = [[] for i in range(data_mean.shape[0])]
peaks_list = [[] for i in range(data_mean.shape[0])]
peak_frequencies = [[] for i in range(data_mean.shape[0])]
peak_values = [[] for i in range(data_mean.shape[0])]
for main_index in range(0, data_mean.shape[0]):
    for index, (value, value_error, value_error_abs) in enumerate(zip(data_mean[main_index], errors[main_index], errors_abs[main_index])):
        if (value_error_abs > rmse[main_index]) and value_error > 0:
            temp_peak = Peak(index, value)
            temp_list[main_index].append(temp_peak)

    for index, peak in enumerate(islice(temp_list[main_index], len(temp_list[main_index]) - 1)):
        if (peak.index_pos - temp_list[main_index][index + 1].index_pos) == -1 and peak.value > temp_list[main_index][index + 1].value:
            if peak.value < temp_list[main_index][index - 1].value:
                pass
            else:
                peaks_list[main_index].append(peak)

    for index, (value_peak, frequency) in enumerate(zip(peaks_list[main_index], f_sup)):
        peak_frequencies[main_index].append(f_sup[value_peak.index_pos])
        peak_values[main_index].append(value_peak.value)
#<=============================================================================================================

plt.figure('raw data and peaks')
labels = ['ACE', 'MG', 'mix1_1', 'mix1_2', 'mix2_1', 'mix2_2']
peak_labels = ['ACE_peak', 'MG_peak', 'mix1_1_peak', 'mix1_2_peak', 'mix2_1_peak', 'mix2_2_peak']

# for data_mean, data_mean_smoothed, label, peak_f, peak_val, peak_label in zip(data_mean, data_mean_smoothed, labels, peak_frequencies, peak_values, peak_labels):
index_ace = 0
index_mg = 1
index_mix = 2
plt.plot(f_sup, data_mean[index_ace]/max(data_mean[index_ace]), '-', label=labels[index_ace])
plt.plot(f_sup, data_mean[index_mg]/max(data_mean[index_mg]), '-', label=labels[index_mg])
# plt.plot(f_sup, data_mean_smoothed[index_ace]/max(data_mean_smoothed[index_ace]), '-', label=labels[index_ace])
# plt.plot(f_sup, data_mean_smoothed[index_mg]/max(data_mean_smoothed[index_mg]), '-', label=labels[index_mg])
plt.plot(peak_frequencies[index_ace], peak_values[index_ace]/max(data_mean[index_ace]), '*', label=peak_labels[index_ace])
plt.plot(peak_frequencies[index_mg], peak_values[index_mg]/max(data_mean[index_mg]), '*', label=peak_labels[index_mg])
plt.plot(f_sup, data_mean[index_mix]/max(data_mean[index_mix]), '-', label=labels[index_mix])
plt.plot(peak_frequencies[index_mix], peak_values[index_mix]/max(data_mean[index_mix]), '*', label=peak_labels[index_mix])


plt.legend()
plt.show()
