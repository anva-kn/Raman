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

class ProcessData():
    
    def __init__(self, filename, number_of_arguments):
        self.filename = filename
        self.number_of_arguments = number_of_arguments
    
    def read_file(self):
        if self.number_of_arguments > 3:
            print("Max amount of arguments exceeded")
            return
        elif self.number_of_arguments < 3:
            print("Please provide a name for db")
            return
        else:
            try:    
                self.data = pd.read_csv(filename, sep="\s+", header=None)
            except FileNotFoundError:
                print("Wrong filename or path.")
            
            
    def read_data(self):
        self.read_file()
        length = self.data.shape[1] - 1
        f_sup = np.array(self.data.iloc[:, 0], dtype=np.float64)
        data_array = self.data.loc[:, 1:].to_numpy(dtype=np.float64)
        data_array = data_array.T
        return f_sup, data_array

    
number_of_arguments = len(sys.argv)
if number_of_arguments <= 1:
    print("Not enough input arguments")
    sys.exit()

filename = str(sys.argv[1])
frequency_support, data = ProcessData(filename, number_of_arguments).read_data()

shelve_name = str(sys.argv[2])

my_shelf = shelve.open(shelve_name)
my_shelf[shelve_name] = data
my_shelf["f_sup_" + shelve_name] = frequency_support
my_shelf.close()