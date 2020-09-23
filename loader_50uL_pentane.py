   # -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 22:26:03 2020

@author: Stefano rini
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as p
import scipy.signal as sci
import scipy.stats as stats
#import     
from math import pi

#------------------------------------------------

import shelve



plt.close('all')

#resolusiton of the spectrum
res=1024
dim=100

#from lmfit.models import LorentzianModel, QuadraticModel
from scipy.optimize import minimize


#----------------------------------------------------------------------
# Load my dark+glass+h2o data
#----------------------------------------------------------------------

# filename='tmp/shelve.out'

# my_shelf = shelve.open(filename)
# for key in my_shelf:
#     globals()[key]=my_shelf[key]
# my_shelf.close()

#----------------------------------------------------------------------
# Suck up the data
#----------------------------------------------------------------------

data=np.zeros([4,dim,res])

file_name=np.chararray(4,dim,res)

file_name[0]="D:/GitHub/SLA/data/D2O and pentane 50uL/50uL pentane/dark_0.2s1a_1_V78500241_"
file_name[1]="D:/GitHub/SLA/data/D2O and pentane 50uL/50uL pentane/gLass_0.2s1a_1_"
file_name[2]="D:/GitHub/SLA/data/D2O and pentane 50uL/50uL pentane/laser_0.2s1a_1_"
file_name[3]="D:/GitHub/SLA/data/D2O and pentane 50uL/50uL pentane/5C_50uL_0.2s1a_2_V78500241_"

for k in range(4):
    for i_file in range(1,dim+1):        
        file_name_k=file_name[k]+"%d.txt" % i_file        
        with open(file_name_k) as f:
            temp = f.readlines()[14:]
            temp=[i.strip('\n') for i in temp]
            data[k,i_file-1]=[float(row.split('\t')[1]) for row in temp]

#such the frequencies from the last set of data
f_sup_px  =np.array([float(row.split('\t')[0]) for row in temp])
# Mapping between pixels and frequencies

f_sup=-13719+f_sup_px*21.42 -0.0048456*f_sup_px**2





save_to_file=True

if save_to_file:
    filename='shelve_save_data_50uL_pentane.out'
    
    my_shelf = shelve.open(filename)
    
    my_shelf["data"]=data 
    my_shelf["f_sup"]=f_sup
    my_shelf["res"]=res
    my_shelf["num_exp"]=dim
    
    my_shelf.close()


