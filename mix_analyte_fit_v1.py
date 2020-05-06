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
import scipy
from math import pi


#------------------------------------------------------------------------------
import shelve

filename='shelve_save_data_analyte.out'

my_shelf = shelve.open(filename)
for key in my_shelf:
    globals()[key]=my_shelf[key]
my_shelf.close()

#------------------------------------------------------------------------------

f_sup=f_sup[:-1]
data_mean=np.mean(data,axis=2)

labels = ['ACE', 'MG', 'mix1_1','mix1_2','mix2_1','mix2_2']

for data_mean, label in zip(data_mean, labels):
    plt.plot(f_sup, data_mean, label=label)

plt.legend()
plt.show()




















# """
# Created on Wed May  6 11:40:53 2020

# @author: rinis
# """

# str_ACE='data/SERS with mix analyte/Acephate 10-2M 5715minutes/Acephate-SERS_1500ms_Exported.dat'
# str_MG='data/SERS with mix analyte/MG 10-5M 930minutes\MG-SERS_1500ms 1avg 2box_Exported.dat'
# str_mix1_1='data/SERS with mix analyte/mix/1115min/Mix-SERS-A-(1)-1115min_532nm_Newton_105um _ND1-51_20x-NA0.75_1500ms 1avg 2box (S-S-H-C)_Exported.dat'
# str_mix1_2='data/SERS with mix analyte/mix/1115min/Mix-SERS-A-(2)-1115min_532nm_Newton_105um _ND1-51_20x-NA0.75_1500ms 1avg 2box (S-S-H-C)_Exported.dat'
# str_mix2_1='data/SERS with mix analyte/mix/4027min\Mix-SERS-A-(1)-4027min_532nm_Newton_105um _ND1-51_20x-NA0.75_1500ms 1avg 2box (S-S-H-C)_Exported.dat'
# str_mix2_2='data/SERS with mix analyte/mix/4027min\Mix-SERS-A-(2)-4027min_532nm_Newton_105um _ND1-51_20x-NA0.75_1500ms 1avg 2box (S-S-H-C)_Exported.dat'

# loader_on=1
# #all spectrums
# if loader_on:
    
#     #-----------------ACE-----------------
#     i_file= open(str_ACE, "r")
#     temp = i_file.readlines()
#     temp=[i.strip('\n') for i in temp]
#     data_ace=np.array([(row.split('\t')) for row in temp], dtype=np.float32)
    
#     f_sup=data_ace[:,0]
#     data_ace=data_ace[:,1:101]
    
#     #-----------------MG-----------------
#     i_file= open(str_MG, "r")
#     temp = i_file.readlines()
#     temp=[i.strip('\n') for i in temp]
#     data_mg=np.array([(row.split('\t')) for row in temp], dtype=np.float32)
    
#     f_sup_mg=data_mg[:,0]    
#     data_mg=data_mg[:,1:101]
    
#     #-----------------mix 1-----------------
#     i_file= open(str_mix1_1, "r")
#     temp = i_file.readlines()
#     temp=[i.strip('\n') for i in temp]
    
#     data_mix1_1=np.array([(row.split('\t')) for row in temp], dtype=np.float32)        
#     f_sup_m1=data_mix1_1[:,0]    
#     data_mix1_1=data_mix1_1[:,1:101]
    
#     i_file= open(str_mix1_2, "r")
#     temp = i_file.readlines()
#     temp=[i.strip('\n') for i in temp]
        
#     data_mix1_2=np.array([(row.split('\t')) for row in temp], dtype=np.float32)        
#     data_mix1_2=data_mix1_2[:,1:101]
    
#     #-----------------mix 2-----------------
    
#     i_file= open(str_mix2_1, "r")
#     temp = i_file.readlines()
#     temp=[i.strip('\n') for i in temp]
    
#     data_mix2_1=np.array([(row.split('\t')) for row in temp], dtype=np.float32)        
#     f_sup_m2=data_mix2_1[:,0]    
#     data_mix2_1=data_mix2_1[:,1:101]
    
#     i_file= open(str_mix2_2, "r")
#     temp = i_file.readlines()
#     temp=[i.strip('\n') for i in temp]
        
#     data_mix2_2=np.array([(row.split('\t')) for row in temp], dtype=np.float32)        
#     f_sup_m22=data_mix2_2[:,0]
#     data_mix2_2=data_mix1_2[:,1:101]
        
    
    
#     init_plot=0
    
#     if init_plot:
#         plt.figure('Data available recap')                
#         plt.plot(f_sup,np.mean(data_ace,axis=1),label='ACE')
#         plt.plot(f_sup_mg,np.mean(data_mg,axis=1),label='MG')
#         plt.plot(f_sup_mg,np.mean(data_mix1_1,axis=1),'--',label='MIX1_1')
#         plt.plot(f_sup_mg,np.mean(data_mix1_2,axis=1),'-.',label='MIX1_2')
#         plt.plot(f_sup,np.mean(data_mix2_1,axis=1),'-*',label='MIX2_1')
#         plt.plot(f_sup,np.mean(data_mix2_2,axis=1),'-|',label='MIX2_2')
#         plt.legend()

# #-----------------------------------------------------------
    
# # downsample everything down to the same frequencies between ~142 to ~~1742

# # min f 151.281
# # max f 1728.141


# data_ace_temp=data_ace[120:alg2,:]
# data_mix2_1_temp=data_mix2_1[120:alg2,:]
# data_mix2_2_temp=data_mix2_1[120:alg2,:]

# len_temp=alg2-120

# # the rest of the data is avareged 
# data_mg_temp     =np.zeros([len_temp,dim])
# data_mix1_1_temp =np.zeros([len_temp,dim])
# data_mix1_2_temp =np.zeros([len_temp,dim])



# # f_sup[120]
# # max(f_sup_mg)

# # do frequencies line up? NO
    
# # find the 
# alg1=np.argmin((f_sup_mg-f_sup[120])**2)
# alg2=np.argmin((f_sup-max(f_sup_mg))**2)


# s1=f_sup[120:alg2]
# s2=f_sup_mg[alg1:-1]

# # s1 is short, s2 is long
# plt.plot(s1-s2[:s1.size])
# plt.plot(s1,'*',label='s1')
# plt.plot(s2,'.',label='s2')
# plt.legend()
# # s1 is always larger than s2

# diff_vec=s1-s2[:s1.size]



# #----------------------------------------------------------------------
# # Subsample
# #----------------------------------------------------------------------




# while i<len_temp-1:    
#     delta_f_int=s1[i+1]-s1[i]
#     pos=np.where((s2>=n_feq) & (s2<=n_feq+delta_f_int))
    
#     # 
#     data_mg_temp[i]     =np.mean(data_mg[alg1+pos,:],axis=1)
#     data_mix1_1_temp[i] =np.mean(data_mix1_1[alg1+pos,:],axis=1)
#     data_mix1_2_temp[i] =np.mean(data_mix1_2[alg1+pos,:],axis=1)
#     #data_mix2_2_temp[i] =np.mean(data_mix2_2[alg1+pos,:],axis=1)
    
#     # update counter
#     i=i+1
#     n_feq=n_feq+delta_f_int

# sub_plot=1


# if sub_plot:
#     plt.figure('Data subsampling recap')                
#     plt.plot(s1,np.mean(data_ace_temp,axis=1),label='ACE')    
#     plt.plot(s1,np.mean(data_mg_temp,axis=1),label='MG')
#     plt.plot(s1,np.mean(data_mix1_1_temp,axis=1),'--',label='MIX1_1')
#     plt.plot(s1,np.mean(data_mix1_2_temp,axis=1),'-.',label='MIX1_2')
#     plt.plot(s1,np.mean(data_mix2_1_temp,axis=1),'.-',label='MIX2_1')
#     plt.plot(s1,np.mean(data_mix2_2_temp,axis=1),'--',label='MIX2_2')
#     plt.legend()
    

# #----------------------------------------------------------------------
# # Save everything
# #----------------------------------------------------------------------



# # # what do we keep? 
# # # the data that are already sparse we just cut down

# # data_ace_temp=data_ace[120:alg2,:]
# # data_mix2_1_temp=data_mix2_1[120:alg2,:]



# # n_feq=min(s1[0],s2[0])
# # # s1 has fewer elements, so you have to merge s2 into s1
# # n_feq_last=min(s1[-1],s2[-1])
# # cont=0
    
# # while n_feq<=n_feq_last:    
# #     delta_f_int=s1[cont+1]-s1[cont]
# #     pos=np.where((s2>=n_feq) & (s2<=n_feq+delta_f_int))
    
    
# #     data[i,:]=np.amax(data[pos[0],:],0)
# #     data_st[i]=np.amax(data_st[pos[0]],0)    
# #     # update counter
# #     i=i+1
# #     n_feq=n_feq+delta_f_int
        
    


# # f_sup_int=np.array(f_sup).astype(int)

# # # the lowest resolution is 2 wavenumbers 
# # delta_f_int=max(np.diff(f_sup_int))

# # #starting frequency-1
# # n_feq=min(f_sup_int)
# # n_feq_last=max(f_sup_int)

# # data_sub=np.zeros(((int((n_feq_last-n_feq)/delta_f_int)+1,dim)))
# # #
# # f_sup_sub=np.array(range(min(f_sup_int),max(f_sup_int)+1,delta_f_int))
# # i=0;

# # while n_feq<=n_feq_last:    
# #     # pos=np.where(f_sup_int==n_feq | f_sup_int==n_feq)    
# #     pos=np.where((f_sup_int>=n_feq) & (f_sup_int<=n_feq+delta_f_int))
# #     data[i,:]=np.amax(data[pos[0],:],0)
# #     data_st[i]=np.amax(data_st[pos[0]],0)    
# #     # update counter
# #     i=i+1
# #     n_feq=n_feq+delta_f_int

# # data=data[1:i+1,:]
# # data_st=data_st[1:i+1]

# # dim_sub=i

# # np.where(f_sup_mg==f_sup[120])

# # f_sup_temp=

# # np.find()
# # plt.plot(f_sup,'--')
# # plt.plot(f_sup_mg,'-.')
# # plt.plot(f_sup_m1,'-.')
# # plt.plot(f_sup_m2,'-.')
        

#     # #normalize the area to one
#     # #
#     # i_file= open('data/MG set 1/1500ppb_all_spectrum.dat', "r")
#     # temp = i_file.readlines()
#     # temp=[i.strip('\n') for i in temp]
#     # data1500=np.array([(row.split('\t')) for row in temp], dtype=np.float32)
#     # f_1500=data1500[:,0]
#     # data1500=data1500[:,1:101]
#     # I=[np.trapz(data1500[:,i],f_1500) for i in range(100)]
#     # data1500=data1500/I
#     # data1500=data1500.reshape([1600,10,10])
    
    
#     # i_file= open('data/MG set 1/15ppb_all_spectrum.dat', "r")
#     # temp = i_file.readlines()
#     # temp=[i.strip('\n') for i in temp]
#     # data15=np.array([(row.split('\t')) for row in temp], dtype=np.float32)
#     # f_15=data15[:,0]
#     # data15=data15[:,1:101]
#     # I=[np.trapz(data15[:,i],f_15) for i in range(100)]
#     # data15=data15/I
#     # data15=data15.reshape([1600,10,10])