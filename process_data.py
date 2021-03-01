#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 19:38:59 2021

@author: anvarkunanbaev
"""

from os import path
import numpy as np
import pandas as pd
import shelve
import scipy.interpolate as si

class ProcessData(filename, ):
    
    def recursive_merge(inter, start_index=0):
    for i in range(start_index, len(inter) - 1):
        if inter[i][1] > inter[i + 1][0]:
            new_start = inter[i][0]
            new_end = inter[i + 1][1]
            inter[i] = [new_start, new_end]
            del inter[i + 1]
            return recursive_merge(inter.copy(), start_index=i)
    return inter

    try:    
        mix = pd.read_csv(filename, sep="\s+", header=None)
    except FileNotFoundError:
        print("Wrong filename or path.")
    else:
        break
    
    