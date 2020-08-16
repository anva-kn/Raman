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

for i in range(data_15000.shape[0][1]):
    plt.plot(f_sup, data_15000[0][i], '-')#, label='one spectrum')
#plt.legend()
plt.show()