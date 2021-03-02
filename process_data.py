#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 19:38:59 2021

@author: anvarkunanbaev
"""

import sys
from os import path
import numpy as np
import pandas as pd
import shelve
import scipy.interpolate as si

n = len(sys.argv)

if n > 2:
    print("Max amount of arguments exceeded")
    break
else:
    filename = str(sys.argv[1])
    try:    
        data = pd.read_csv(filename, sep="\s+", header=None)
    except FileNotFoundError:
        print("Wrong filename or path.")
    else:
        break
    
def read_data(data):
    length = data.shape[1] - 1
    f_sup = np.array(data.iloc[:, 0], dtype=np.float64)
    data_15000_area1 = np.zeros((length, f_sup.shape[0]))
    
    j = 0
    for i in range(1, length):
    
        data_15000_area1[j] = np.array(mix_15000_area_1.iloc[:, i], dtype=np.float64)
        data_15000_area2[j] = np.array(mix_15000_area_2.iloc[:, i], dtype=np.float64)
        data_15000_area3[j] = np.array(mix_15000_area_3.iloc[:, i], dtype=np.float64)
        data_15000_area4[j] = np.array(mix_15000_area_4.iloc[:, i], dtype=np.float64)
    
    # #-------------------------------    1500 PPB    -----------------------------------------------
        data_1500_area1[j] = np.array(mix_1500_area_1.iloc[:, i], dtype=np.float64)
        data_1500_area2[j] = np.array(mix_1500_area_2.iloc[:, i], dtype=np.float64)
        data_1500_area3[j] = np.array(mix_1500_area_3.iloc[:, i], dtype=np.float64)
        data_1500_area4[j] = np.array(mix_1500_area_4.iloc[:, i], dtype=np.float64)
    # #-------------------------------    150 PPB    ------------------------------------------------
        data_150_area1[j] = np.array(mix_150_area_1.iloc[:, i], dtype=np.float64)
        # data_150_area2 = np.array(mix_150_area_2.iloc[:, 1:], dtype=np.float64)
        # data_150_area3 = np.array(mix_150_area_3.iloc[:, 1:], dtype=np.float64)
        # data_150_area4 = np.array(mix_150_area_4.iloc[:, 1:], dtype=np.float64)
        j += 1
    #
    #
    # ################################ SAVE THE DATA ###########################################
    save_to_file=1
    
    data_15000 = np.zeros([4, length, f_sup.shape[0]])
    data_1500 = np.zeros([4, length, f_sup.shape[0]])
    data_150 = np.zeros([4, length, f_sup.shape[0]])
    
    data_15000[0], data_15000[1], data_15000[2], data_15000[3] = data_15000_area1, data_15000_area2, data_15000_area3, data_15000_area4
    data_1500[0], data_1500[1], data_1500[2], data_1500[3] = data_1500_area1, data_1500_area2, data_1500_area3, data_1500_area4
    #  data_150[0], data_150[1], data_150[2], data_150[3] = data_150_area1, data_150_area2, data_150_area3, data_150_area4
    
    
    if save_to_file:
        filename = 'shelve_save_data.out'
    
        my_shelf = shelve.open(filename)
    
        my_shelf["data_15000"] = data_15000
        my_shelf["data_1500"] = data_1500
        my_shelf["data_150"] = data_150
        my_shelf["f_sup"] = f_sup
    
        my_shelf.close()
    

    


res = 1600
num_th = 5
peak_pos = []
interpol_mse = []
slide_win = 5
nu = 0.01
interpol_mse = interpol_mse + [1000, 1000, 1000]
