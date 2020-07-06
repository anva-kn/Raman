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


plt.figure('ACE')
s = UnivariateSpline(f_sup, data_mean[0], s=5)
xs = np.linspace(0, 963, 200)
ys = s(xs)

plt.plot(f_sup, data_mean[0])
#plt.plot(xs, ys, 'o')
plt.show()