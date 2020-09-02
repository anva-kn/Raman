#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 02:33:27 2020

@author: akula
"""

import numpy as np
import pandas as pd
import shelve
import keras_experiment
import matplotlib.pyplot as plt

filename='shelve_save_data.out'

my_shelf = shelve.open(filename)
for key in my_shelf:
    globals()[key]=my_shelf[key]
my_shelf.close()

one_spectrum = data_15000[0][0]
print(data_15000.shape, f_sup.shape)

whole_data = np.zeros((10000, 1600))
print(whole_data.shape)
whole_data[0:1250] = np.copy(data_15000[0, 0:1250]/np.max(data_15000[0]))
whole_data[1250:2500] = np.copy(data_15000[1, 0:1250]/np.max(data_15000[1]))
whole_data[2500:3750] = np.copy(data_15000[2, 0:1250]/np.max(data_15000[2]))
whole_data[3750:5000] = np.copy(data_15000[3, 0:1250]/np.max(data_15000[3]))
whole_data[5000:6250] = np.copy(data_1500[0, 0:1250]/np.max(data_1500[0]))
whole_data[6250:7500] = np.copy(data_1500[1, 0:1250]/np.max(data_1500[1]))
whole_data[7500:8750] = np.copy(data_1500[2, 0:1250]/np.max(data_1500[2]))
whole_data[8750:10000] = np.copy(data_1500[3, 0:1250]/np.max(data_1500[3]))
labels = np.zeros(10000)
labels[0:5000] = 0
print(labels[5000])
labels[5000:10000] = 1
print(labels[5000])
plt.plot(f_sup, whole_data[1299], '-')#, label='one spectrum')
# plt.legend()
plt.show()
print(whole_data[9999])
X_fn = './data/X_finetune.npy'
y_fn = './data/y_finetune.npy'
X = np.copy(whole_data)
y = np.copy(labels)
print(y)
print(X.shape, y.shape)
checksumbitch = np.where(whole_data == 0)
indeces = np.unique(checksumbitch[0])