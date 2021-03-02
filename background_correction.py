#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 10:04:42 2021

@author: anvarkunanbaev
"""

import numpy as np
import pandas as pd
import shelve
import scipy.interpolate as si


class BackgroundCorrection(x_data, y_data, number_of_iteration: Optional):
    
    def recursive_merge(inter, start_index=0):
        for i in range(start_index, len(inter) - 1):
            if inter[i][1] > inter[i + 1][0]:
                new_start = inter[i][0]
                new_end = inter[i + 1][1]
                inter[i] = [new_start, new_end]
                del inter[i + 1]
                return recursive_merge(inter.copy(), start_index=i)
        return inter
    
    