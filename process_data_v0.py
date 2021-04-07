#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 14:12:57 2021

@author: anvarkunanbaev
"""
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
from skimage import io

class ProcessData():
    
    def __init__(self, filename, number_of_arguments, shelve_name):
        self.filename = filename
        self.number_of_arguments = number_of_arguments
        self.shelve_name = shelve_name
    
    def read_dat_file(self):
        if self.number_of_arguments > 3:
            print("Max amount of arguments exceeded")
            return
        elif self.number_of_arguments < 3:
            print("Please provide a name for db")
            return
        else:
            try:    
                self.data = pd.read_csv(self.filename, sep="\s+", header=None)
                self.f_sup = np.array(self.data.iloc[:, 0], dtype=np.float64)
                self.data_array = self.data.loc[:, 1:].to_numpy(dtype=np.float64)
                self.data_array = self.data_array.T
                if self.data_array.shape[0] == 1:
                    self.data_array = np.squeeze(self.data_array)
                np.ascontiguousarray(self.data_array, dtype=np.float64)
                return self.f_sup, self.data_array
            except FileNotFoundError:
                print("Wrong filename or path.")
            
    def read_tiff_file(self):
        if self.number_of_arguments > 3:
            print("Max amount of arguments exceeded")
            return
        elif self.number_of_arguments < 3:
            print("Please provide a name for db")
            return
        else:
            try:    
                self.data = io.imread(self.filename)
                if self.data.ndim == 3:
                    shape = self.data.shape[0]
                    self.data_reshaped = self.data.transpose(2,1,0).reshape(-1, shape)
                    return np.ascontiguousarray(self.data_reshaped, dtype=np.float64)
                return self.data
            except FileNotFoundError:
                print("Wrong filename or path.")
            

    def store_data(self):
        self.read_dat_data()
        temp_shelf = shelve.open(self.shelve_name)
        temp_shelf[self.shelve_name] = self.data_array
        temp_shelf["f_sup" + self.shelve_name] = self.f_sup
        temp_shelf.close()
        
    def recursive_merge(inter, start_index=0):
       for i in range(start_index, len(inter) - 1):
           if inter[i][1] > inter[i + 1][0]:
               new_start = inter[i][0]
               new_end = inter[i + 1][1]
               inter[i] = [new_start, new_end]
               del inter[i + 1]
               return recursive_merge(inter.copy(), start_index=i)
       return inter
   
    
        
        
    # def read_tiff_area_file():
        
        
# number_of_arguments = len(sys.argv)
# if number_of_arguments <= 1:
#     print("Not enough input arguments")
#     sys.exit()

# filename = str(sys.argv[1])
# # filename = "data/MG-3_15000_ppb/area1_Exported.dat"
# frequency_support, data = ProcessData(filename, number_of_arguments).read_data()

# shelve_name = str(sys.argv[2])# + ".db"
# # shelve_name = "check"
# my_shelf = shelve.open(shelve_name, flag="c")
# my_shelf[shelve_name] = data
# my_shelf["f_sup_" + shelve_name] = frequency_support
# my_shelf.close()