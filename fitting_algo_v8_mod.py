#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 21:11:49 2020

@author: Stefano Rini
"""

#-----------------------------------------------
# fit the peaks now
#-----------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as p
import scipy.signal as sci
import scipy.stats as stats
import pymf3
from math import pi
from scipy.optimize import leastsq
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

# from main_ste_v1 import data1500

#------------------------------------------------------------------------------

def smooth_data(data,win):
    return np.array([np.mean(data[int(np.max([j-win,0])):int(np.min([j+win,data.size]))]) for j in  range(data.size)])

# generalize gaussian with any amplitude
# for the bias, we will remove it from the fitting
def gauss_amp(x_data,beta):
    # beta: 
        # a
        # c
        # sgs    
        M=beta[0]
        c=beta[1]
        sgs=beta[2]
        return [M*math.exp(-(x_data[i]-c)**2/ (2*abs(sgs))) for i in range(x_data.size)]

def gen_lor_amp(x_data,beta):
    # beta: 
        # a
        # c
        # sgs    
        M=beta[0]
        c=beta[1]
        sgs=beta[2]
        d=beta[3]
        return [M/(1+(abs((x_data[i]-c)/sgs))**d) for i in range(x_data.size)]    

def positive_mse(y_pred,y_true):                
    loss=np.dot((10**8*(y_pred-y_true>0)+np.ones(y_true.size)).T,(y_pred-y_true)**2)
    return sum(loss)/y_true.size

def pos_mse_loss(beta, X, Y,function):                
    return positive_mse(function(X,beta), Y)

def mse_loss(beta,X, Y,function):                
    return sum((function(X,beta)-Y)**2)/Y.size

def find_peak_sym_supp(peak,l_pos,r_pos, win_len_mul, smooth_win, tol, Y):
    # win_len_mul= how much do we extend the window length 
    # smooth_win= how much do we smooth the function
    # tol: tollerance on the level of asymmetry as a fraction of the peak hight
    #               we take the asymmetry factor as in http://files.alfresco.mjh.group/alfresco_images/pharma//2014/08/22/d9fbf22f-a33b-4eff-93a4-c25afbc47fa2/article-45057.pdf
        
    Y_smooth=smooth_data(Y,smooth_win)
    Y_max=np.max(Y_smooth[l_pos-1:r_pos+1])
    Y_min=np.mean(np.sort(Y_smooth)[::int(Y_smooth.size/10)])
    Y_max_idx=l_pos+np.argmax(Y[l_pos:r_pos])    
    # make the window symmetric around the max
    # MAX OR MIN HERE!?
    win_len=2*max([Y_max_idx-l_pos,r_pos-Y_max_idx])
    #win_len=2*min(Y_max_idx-l_pos,r_pos-Y_max_idx)
    #plt.plot(Y_smooth[l_pos-20:r_pos+20])
    #plt.plot(Y[l_pos-20:r_pos+20])
    win_len_max=2*win_len_mul*win_len    
    
    C_decreasing=True
    C_in_window=True
    C_same_level=True
    
    l_pos_temp=Y_max_idx-1
    r_pos_temp=Y_max_idx+1
    
    while  C_decreasing and C_in_window and C_same_level: 
        #plt.plot(Y_smooth[l_pos-10:r_pos+10])
        C_decreasing = (Y_smooth[l_pos_temp-1]<=Y_smooth[l_pos_temp]) and (Y_smooth[r_pos+1]<=Y_smooth[r_pos]) 
        C_in_window  = (win_len<=win_len_max)
                
        peak_hight_l=Y_max-Y_smooth[l_pos_temp-1]        
        peak_hight_r=Y_max-Y_smooth[r_pos_temp+1]
        # peak drop as a ration of the minimum
        rel_height=(np.mean([Y_smooth[l_pos_temp-1],Y_smooth[r_pos_temp+1]])-Y_min)/(Y_max-Y_min)
        
        angle_l=np.degrees(np.arctan(peak_hight_l/(Y_max_idx-l_pos_temp)))
        angle_r=np.degrees(np.arctan(peak_hight_r/(r_pos_temp-Y_max_idx)))
        asymmetry=1-(angle_l+angle_r)/(2*np.max([angle_l,angle_r]))
        
        C_same_level=(1-rel_height)<=tol or asymmetry<=tol
        
        l_pos_temp=l_pos_temp-1    
        r_pos_temp=r_pos_temp+1            
        
    return [l_pos_temp,r_pos_temp]

def cos_window(t,M,o,d,al,be):
    
    if o<=t & t<o+al:
        return M/2*(1-np.cos(2*pi*(t-o)/(2*al)))
    elif o+al<=t & t<o+al+d: 
        return M
    elif o+al+d<= t & t<o+al+d+be:
        return M/2*(1-np.cos(2*pi*(be-(t-d-o-al))/(2*be)))
    else:
        return 0

def cos_win(x_data,beta):    
    M=beta[0]
    o=beta[1]
    d=beta[2]
    al=beta[3]
    be=beta[4]

    return [cos_window(t,M,o,d,al,be)  for t in range(x_data.size)]

# init functions

def init_lor(x_data,y_data):
    
    beta_init_lor=np.zeros(4)
    
    # M=beta[0]
    # c=beta[1]
    # sgs=beta[2]
    # d=beta[3]
        
    beta_init_lor[0]=np.max(y_data)
    #center
    beta_init_lor[1]=x_data[np.argmax(y_data)]
    # gamma
    beta_init_lor[2]=(x_data[-1]-x_data[0])/16
    # 
    beta_init_lor[3]=2
    
    return beta_init_lor

def init_cos(x_data,y_data,win):
    # order of arguments: M,o,d,al,be
    y_smoth=smooth_data(y_data,win)
    power=np.dot(y_data,y_data.T)/dim_amm

    floor_th=0.1*np.sqrt(power)
    # how to define the ceiling of the function
    ceil_th=0.9*np.sqrt(power)    

    #  while extending 0 l_win_low /  win_up / l_win_high-r_win_high / win_down / r_win_low-dim_amm

    l_win_low  = np.argmax(floor_th<y_smoth)
    r_win_low =dim_amm- np.argmax(floor_th<np.flip(y_smoth) )

    l_win_high =  np.argmax(ceil_th<y_smoth)
    r_win_high =dim_amm- np.argmax(ceil_th<np.flip(y_smoth) )    

    return [
        ceil_th,
        l_win_low,
        r_win_high - l_win_high,
        l_win_high - l_win_low,
        r_win_low - r_win_high,
    ]

def is_solvent(y_data):
    # check the time we are under the mean minus the standard deviation
    perc=np.sum(y_data<np.mean(y_data)-1/2*np.sqrt(np.var(y_data)))/y_data.size<0.5
    start_up=np.mean(y_data[:int(y_data.size*0.1)])>np.mean(y_data)
    return perc and start_up
    
#y_data>np.mean(np.sort(y_data)[0:int(y_data.size/25)])
   
#------------------------------------------------------------------------------
res=1600
dim_s=100

# poly_min_hat = quick_remove_bias(data1500)

data_sparse = np.reshape(data1500, [res, dim_s]) - np.reshape(poly_min_hat,[res,1])
data_sparse = data_sparse - np.min(data_sparse)

#------------------------------------------------------------------------------
# NORMALIZE THE SPECTRUM SO THAT THE AVERAGE AMPLITUDE IS ONE
#------------------------------------------------------------------------------

specMean=np.mean(data_sparse)

data_sparse=data_sparse/specMean
data_mean_sparse=np.mean(data_sparse,axis=0)

# FIND THE PEAKS 

peaks, properties = sci.find_peaks(smooth_data(data_sparse,2),prominence=0.05,width=1) 

peaks_v2, properties_v2 = sci.find_peaks(smooth_data(test3_data_sparse,2),prominence=0.05,width=1) 

# plt.plot(f_vec, test3_data_sparse)
# plt.plot(f_vec[peaks],test3_data_sparse[peaks],'*')
# form a metrix using prominence+width

peak_prom=properties['prominences']
peak_prom=peak_prom/np.sqrt(np.var(peak_prom))
peak_width=properties['widths']
peak_width=peak_width/np.sqrt(np.var(peak_width))
peak_high=(data_sparse[peaks])/np.sqrt(np.var(data_sparse[peaks]))
peak_high = peak_high.reshape((peak_high.shape[0], -1)).mean(axis=1)
# ANY CHOICE OF THE METRIC HERE!?

metric=10*peak_high+5*peak_prom+peak_width

idx_peaks=np.flip(np.argsort(metric))
idx=peaks[idx_peaks]
rough_peak_positions = idx

# variables we store
#------------------------------------------------------------------------------

# backup of the data
data_sparse_back=np.zeros([rough_peak_positions.size+1,data_sparse.shape[0],data_sparse.shape[1]])
# inital boundaries and boudaries after expansion
comp_range=np.zeros([rough_peak_positions.size,4])
# lorenzian fit
comp_beta_lor=np.zeros([rough_peak_positions.size,4])
# cos window fit fit
comp_beta_cos=np.zeros([rough_peak_positions.size,5])
# bias t & bias f
comp_bias=np.zeros([rough_peak_positions.size,2])
# the water peaks are inverse
comp_up_down=np.zeros([rough_peak_positions.size])
# start with the original data
data_sparse_back[0,:,:]=data_sparse

# MAIN LOOP OVER THE PEAKS
#------------------------------------------------------------------------------

plot_f=0
plot_t=1

for i in range(5):    
    print('Step: ',i,' -peak position: ',rough_peak_positions[i])

    p=peaks[i]

    l_ips=int(np.floor(properties["left_ips"][p]))
    r_ips=int(np.ceil(properties["right_ips"][p]))

    # rough_peak_positions[i] is in position idx_peaks[i]        
    # tolerance is 10%
    #                find_peak_sym_supp(peak,    l_pos,r_pos, win_len_mul, smooth_win, tol, Y):
    [l_win, r_win] = find_peak_sym_supp(peaks[p],l_ips,r_ips, 5, 5,0.0, data_sparse)

    x_data = f_vec[l_win:r_win]
    y_data = np.mean(data_sparse_back[i,l_win:r_win,:],axis=1)

    # this is my guess for a good strategy
    bias_f = min(y_data)/2
    # random choice of bias
    #bias_f = np.mean(np.sort(y_data)[0:int(y_data.size/25)])
    y_data = y_data -bias_f 

    beta_init_lor=init_lor(x_data,y_data)

    #result = minimize(pos_mse_loss, beta_init_lor, args=(x_data,y_data,gen_lor_amp), method='Nelder-Mead', tol=1e-8)
    result = minimize(mse_loss, beta_init_lor, args=(x_data,y_data,gen_lor_amp), method='Nelder-Mead', tol=1e-12)
    beta_hat_lor=result.x    

    if plot_f:
        plt.figure(f'peak number {int(i)} frequency')
        plt.plot(x_data,y_data)
        plt.plot(x_data,gen_lor_amp(x_data,beta_init_lor),'--')
        plt.plot(x_data,gen_lor_amp(x_data,beta_hat_lor),'-*')

    # pick  a single point to fix
    # y_data=data_sparse_back[i,int((r_win+l_win)/2),:]    
    # x_data=np.array(range(dim_amm))

    # #------------------------------------------------------------------------------
    # # is this a water peak? check the mean!

    # solve=is_solvent(y_data)       

    # if  solve:
    #     bias_t=np.mean(np.sort(y_data)[::-int(y_data.size/50)])
    #     #bias_t=np.max(y_data)
    #     y_data=bias_t-y_data
    #     y_data=y_data+np.min(y_data)
    # else:
    #     #bias_t=np.min(y_data)
    #     bias_t=np.mean(np.sort(y_data)[::int(y_data.size/50)])
    #     y_data=y_data-bias_t
    #     y_data=y_data-np.min(y_data)    

    # bounds_cos=np.array([np.zeros(5) , np.inf*np.ones(5)]).T    
    # bounds_cos=tuple(map(tuple, bounds))

    # beta_init_cos=np.array(init_cos(x_data,y_data,5))
    # #result = minimize(pos_mse_loss, beta_init_cos, args=(x_data,y_data,cos_win), tol=1e-8,bounds=bounds_cos)    
    # result = minimize(mse_loss, beta_init_cos, args=(x_data,y_data,cos_win), tol=1e-12,bounds=bounds_cos)    

    # beta_hat_cos=result.x

    # if plot_t:
    #     plt.figure('peak number '+str(int(i))+' time')
    #     plt.plot(x_data,(1-2*solve)*y_data)            
    #     #plt.plot(x_data,(1-2*solve)*np.array(cos_win(x_data,beta_init_cos)),'--')        
    #     plt.plot(x_data,(1-2*solve)*np.array(cos_win(x_data,beta_hat_cos)),'-*')        

    # # refit the lorenzian in the smaller interval        
    # # use the previous estiamate to start
    # y_data=np.mean(data_sparse_back[i,l_win:r_win,int(beta_hat_cos[1]):int(sum(beta_hat_cos[2:5]))],axis=1)        
    # x_data = f_sup[l_win:r_win]    
    # bias_f2 = 0
    # #bias_f2 = np.mean(np.sort(y_data)[0:max(int(y_data.size/25)])
    # y_data = y_data -bias_f2 
    # y_data=y_data-np.min(y_data)    

    # beta_init_lor2=init_lor(x_data,y_data)
    # #result = minimize(pos_mse_loss, beta_init_lor, args=(x_data,y_data,gen_lor_amp), method='Nelder-Mead', tol=1e-8)        
    # #result = minimize(mse_loss, beta_init_lor, args=(x_data,y_data,gen_lor_amp), method='Nelder-Mead', tol=1e-8)
    # result = minimize(mse_loss, beta_hat_lor, args=(x_data,y_data,gen_lor_amp), method='Nelder-Mead', tol=1e-12)
    # beta_hat_lor2=result.x

    # if plot_f:
    #     plt.figure('peak number '+str(int(i))+' frequency')
    #     plt.plot(x_data,y_data)    
    #     #plt.plot(x_data,gen_lor_amp(x_data,beta_hat_lor),'--')
    #     plt.plot(x_data,gen_lor_amp(x_data,beta_hat_lor2),'-*')

    # store everything     
    comp_range[i]=[l_win,r_win,l_ips,r_ips]
    comp_beta_lor[i]=beta_hat_lor2
    # comp_beta_cos[i]=beta_hat_cos
    comp_bias[i]=[bias_f2]#, bias_t]
    # comp_up_down[i]=solve
    # don't know what to subtract....
    data_sparse_back[i+1]=data_sparse_back[i]

        
    
        
        
    
    
    

# np.min(data_sparse_back[i+1])
    
# plt.plot(np.mean(data_sparse_back[i],axis=1))
# plt.plot(np.mean(data_sparse_back[i+1],axis=1))
  
            
# plt.pcolormesh(data_sub.T)
# plt.pcolormesh(data_sparse_back[i+1])    
    
    
# data_sparse_back[i+1]=
    
    
# X, Y = np.meshgrid(f_sup, np.array(range(dim_amm)))
# Z = data_sub
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.contour3D(X, Y, Z, 50, cmap='binary')
# Z1 = data_sparse_back[i].T
# ax.contour3D(X, Y, Z1, 50, cmap='binary')
    
    
    