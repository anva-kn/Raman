#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 13:32:03 2020

@author: akula
"""

import numpy as np
import pandas as pd
import shelve

################################ READ THE DATA ################################################

#-------------------------------    15000 PPB    ----------------------------------------------
mix_15000_area_1 = pd.read_csv("data/MG-3_15000_ppb/area1_Exported.dat", sep="\s+", header=None)
mix_15000_area_2 = pd.read_csv("data/MG-3_15000_ppb/area2_Exported.dat", sep="\s+", header=None)
mix_15000_area_3 = pd.read_csv("data/MG-3_15000_ppb/area3_Exported.dat", sep="\s+", header=None)
mix_15000_area_4 = pd.read_csv("data/MG-3_15000_ppb/area4_Exported.dat", sep="\s+", header=None)
#-------------------------------    1500 PPB    ----------------------------------------------
mix_1500_area_1 = pd.read_csv("data/MG-4_1500_ppb/area1_Exported.dat", sep="\s+", header=None)
mix_1500_area_2 = pd.read_csv("data/MG-4_1500_ppb/area2_Exported.dat", sep="\s+", header=None)
mix_1500_area_3 = pd.read_csv("data/MG-4_1500_ppb/area3_Exported.dat", sep="\s+", header=None)
mix_1500_area_4 = pd.read_csv("data/MG-4_1500_ppb/area4_Exported.dat", sep="\s+", header=None)
#-------------------------------    150 PPB    ----------------------------------------------
mix_150_area_1 = pd.read_csv("data/MG-5_150_ppb/area1_Exported.dat", sep="\s+", header=None)
mix_150_area_2 = pd.read_csv("data/MG-5_150_ppb/area2_Exported.dat", sep="\s+", header=None)
mix_150_area_3 = pd.read_csv("data/MG-5_150_ppb/area3_Exported.dat", sep="\s+", header=None)
mix_150_area_4 = pd.read_csv("data/MG-5_150_ppb/area4_Exported.dat", sep="\s+", header=None)

################################ PREPROCESS THE DATA ###########################################

#-------------------------------    15000 PPB    ----------------------------------------------
f_sup = np.array(mix_15000_area_1.iloc[:, 0], dtype=np.float64)
data_15000_area1 = np.array(mix_15000_area_1.iloc[:, 1:], dtype=np.float64)
data_15000_area2 = np.array(mix_15000_area_2.iloc[:, 1:], dtype=np.float64)
data_15000_area3 = np.array(mix_15000_area_3.iloc[:, 1:], dtype=np.float64)
data_15000_area4 = np.array(mix_15000_area_4.iloc[:, 1:], dtype=np.float64)

#-------------------------------    1500 PPB    -----------------------------------------------
data_1500_area1 = np.array(mix_1500_area_1.iloc[:, 1:], dtype=np.float64)
data_1500_area2 = np.array(mix_1500_area_2.iloc[:, 1:], dtype=np.float64)
data_1500_area3 = np.array(mix_1500_area_3.iloc[:, 1:], dtype=np.float64)
data_1500_area4 = np.array(mix_1500_area_4.iloc[:, 1:], dtype=np.float64)

#-------------------------------    150 PPB    ------------------------------------------------
data_150_area1 = np.array(mix_150_area_1.iloc[:, 1:], dtype=np.float64)
data_150_area2 = np.array(mix_150_area_2.iloc[:, 1:], dtype=np.float64)
data_150_area3 = np.array(mix_150_area_3.iloc[:, 1:], dtype=np.float64)
data_150_area4 = np.array(mix_150_area_4.iloc[:, 1:], dtype=np.float64)


################################ SAVE THE DATA ###########################################
save_to_file=1

data_15000 = np.zeros([4, f_sup.shape[0], data_15000_area1.shape[1]])
data_1500 = np.zeros([4, f_sup.shape[0], data_15000_area1.shape[1]])
data_150 = np.zeros([4, f_sup.shape[0], data_15000_area1.shape[1]])

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