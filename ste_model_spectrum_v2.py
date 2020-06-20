# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:52:29 2020

@author: Stefano rini
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert, find_peaks
import pylab as p
import scipy.signal as sci
import scipy.stats as stats
import pymf3
from scipy.optimize import minimize
from scipy.optimize import curve_fit
import math
import scipy
from math import pi

#----------------------------------------------------------------------------
#       FUNCTION TO BE FITTED
#----------------------------------------------------------------------------

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


def lor_amp(x_data,beta):
    # beta: 
        # a
        # c
        # sgs    
        M=beta[0]
        c=beta[1]
        sgs=beta[2]
        d=2
        return [M/(1+(abs((x_data[i]-c)/sgs))**d) for i in range(x_data.size)]    


def cos_window(t,L,M,R,o,d,al,be):
    
    if t<o:
        out=L    
    elif o<=t & t<o+al:
        out=L+(M-L)/2*(1-np.cos(2*pi*(t-o)/(2*al)))
    elif o+al<=t & t<o+al+d: 
        out=M
    elif o+al+d<= t & t<o+al+d+be:
        out=R+(M-R)/2*(1-np.cos(2*pi*(be-(t-d-o-al))/(2*be)))
    elif t>=o+al+d+be:
        out=R
    else:
        out=0
    return  out


def cos_window2(t,L,M,R,o,al,be):
    
    if t<o:
        out=L    
    elif o<=t & t<o+al:
        out=L+(M-L)/2*(1-np.cos(2*pi*(t-o)/(2*al)))
    elif o+al<= t & t<o+al+be:
        out=R+(M-R)/2*(1-np.cos(2*pi*(be-(t-o-al))/(2*be)))
    elif t>=o+al+be:
        out=R
    else:
        out=0
    return  out

def cos_win(x_data,beta):    
    L=beta[0]
    M=beta[1]
    R=beta[2]
    o=beta[3]
    d=beta[4]
    al=beta[5]
    be=beta[6]
    
    yvec=[cos_window(t,L,M,R,o,d,al,be)  for t in range(x_data.size)]
    return yvec

def cos_win2(x_data,beta):    
    L=beta[0]
    M=beta[1]
    R=beta[2]
    o=beta[3]    
    al=beta[4]
    be=beta[5]
    
    yvec=[cos_window2(t,L,M,R,o,al,be)  for t in range(x_data.size)]
    return yvec



#-----------------------------------------------------------------------------
# init functions

# def init_cos3(x_data,y_data,mean_level_win):
#     # order of arguments: L,M,R,o,al,be
    
#     # fix the smoothing window to 3
#     dim_amm=y_data.shape[0]
#     y_smoth=smooth_data(y_data,3)
#     power=np.dot(y_data,y_data.T)/dim_amm
        
#     floor_th=max(mean_level_win)
    
#     l_win_low  = np.argmax(floor_th<y_smoth) 
#     r_win_low =x_data.size- np.argmax(floor_th<np.flip(y_smoth) )
    
    
#     M=np.max(y_data[l_win_low:r_win_low])
#     peak_pos=l_win_low+np.argmax(y_data[l_win_low:r_win_low])
                     
#     #l_win_high =  np.argmax(ceil_th<y_smoth) 
#     #r_win_high =x_data.size- np.argmax(ceil_th<np.flip(y_smoth) )    
    
#     # L_init=0
#     # M_init =ceil_th
#     # R_init=0
#     # o_init = l_win_low  
#     # al_init= peak_pos-l_win_low  
#     # be_init= r_win_low- peak_pos
    
#     #L_int=y_smoth[0]-mean_level_win[0]
#     #R_int=y_smoth[-1]-mean_level_win[-1]
    
#     L_int=y_smoth[0]
#     R_int=y_smoth[-1]
    
    
#     beta_cos2_int=[L_int, M, R_int, l_win_low, peak_pos-l_win_low, r_win_low- peak_pos]
    
#     return beta_cos2_int   


#-----------------------------------------------------------------------------

def positive_mse(y_pred,y_true):                
    loss=np.dot((10**8*(y_pred-y_true>0)+np.ones(y_true.size)).T,(y_pred-y_true)**2)
    return sum(loss)/y_true.size

def pos_mse_loss(beta, X, Y,function):                
    return positive_mse(function(X,beta), Y)


def positive_mse_loss(beta, X, Y):
    p=np.poly1d(beta)
    error = sum([positive_mse(p(X[i]), Y[i]) for i in range(X.size)])/X.size
    return(error)

def reg_positive_mse(y_pred,y_true):                
    loss_pos=(y_pred-y_true)**2
    loss_neg=np.dot((y_pred-y_true<0),(y_pred-y_true)**8)
    loss=loss_pos+loss_neg
    return np.sum(loss)/y_true.size

def reg_pos_loss(beta, X, Y,function):                
    return positive_mse(function(X,beta), Y)


def mse_loss(beta,X, Y,function):
    return sum((function(X,beta)-Y)**2)/Y.size

def find_peak_sym_supp(peak,l_pos,r_pos, win_len_mul, smooth_win, Y):
    # win_len_mul= how much do we extend the window length 
    # smooth_win= how much do we smooth the function
    res=Y.shape[0]
    Y_smooth=smooth_data(Y,smooth_win)

    
    if l_pos==r_pos:
        y_data=Y_smooth[l_pos]
        Y_max=y_data
        Y_max_idx=l_pos
        Y_min=Y_max
    else:
        y_data=Y_smooth[l_pos:r_pos]
        Y_max=np.max(y_data)
        Y_max_idx=l_pos+np.argmax(Y[l_pos:r_pos])        
        Y_min=np.mean(np.sort(y_data)[0:int(y_data.size/10)+1])
            
    l_pos_temp=Y_max_idx-1
    r_pos_temp=Y_max_idx+1
        
    while  Y_smooth[l_pos_temp-1]<=Y_smooth[l_pos_temp]  and l_pos_temp>0:
        l_pos_temp=l_pos_temp-1
    
    while  Y_smooth[r_pos_temp+1]<=Y_smooth[r_pos_temp] and r_pos_temp<res-1: 
        r_pos_temp=r_pos_temp+1
        
    return [l_pos_temp,r_pos_temp]

#-----------------------------------------------------------------------------
# Initialize parameters
#-----------------------------------------------------------------------------

def init_cos(x_data,y_data):
    # order of arguments: M,o,d,al,be
    
    # fix the smoothing window to 3
    y_smoth=smooth_data(y_data,3)
    power=np.dot(y_data,y_data.T)/dim_s
        
    floor_th=0.1*np.sqrt(power)
    # how to define the ceiling of the function
    ceil_th=0.9*np.sqrt(power)    
    
    #  while extending 0 l_win_low /  win_up / l_win_high-r_win_high / win_down / r_win_low-dim_s
    
    l_win_low  = np.argmax(floor_th<y_smoth) 
    r_win_low =x_data.size- np.argmax(floor_th<np.flip(y_smoth) )
    
    
    l_win_high =  np.argmax(ceil_th<y_smoth) 
    r_win_high =x_data.size- np.argmax(ceil_th<np.flip(y_smoth) )    
    
    # L_init=0
    # M_init =ceil_th
    # R_init=0
    # o_init = l_win_low  
    # d_init = r_win_high -l_win_high 
    # al_init= l_win_high-l_win_low  
    # be_init= r_win_low- r_win_high
    
    beta_cos_int=[0, ceil_th, 0, l_win_low, r_win_high -l_win_high, l_win_high-l_win_low, r_win_low- r_win_high]
    
    return beta_cos_int

def init_cos2(x_data,y_data):
    # order of arguments: L,M,R,o,al,be
    
    # fix the smoothing window to 3
    y_smoth=smooth_data(y_data,3)
    power=np.dot(y_data,y_data.T)/y_data.shape[0]
        
    floor_th=0.1*np.sqrt(power)
    
    l_win_low  = np.argmax(floor_th<y_smoth) 
    r_win_low =x_data.size- np.argmax(floor_th<np.flip(y_smoth) )
    
    
    M=np.max(y_data[l_win_low:r_win_low])
    peak_pos=l_win_low+np.argmax(y_data[l_win_low:r_win_low])
                     
    #l_win_high =  np.argmax(ceil_th<y_smoth) 
    #r_win_high =x_data.size- np.argmax(ceil_th<np.flip(y_smoth) )    
    
    # L_init=0
    # M_init =ceil_th
    # R_init=0
    # o_init = l_win_low  
    # al_init= peak_pos-l_win_low  
    # be_init= r_win_low- peak_pos
    
    beta_cos2_int=[0, M, 0, l_win_low, peak_pos-l_win_low, r_win_low- peak_pos]
    
    return beta_cos2_int   

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

#------------------------------------------------------------------------------

# PEAK FINDING

def peaks_cwt_1(data_mean):
    peaks = sci.find_peaks_cwt(data_mean, np.arange(0.001,10)) 
    [ prominences , l_ips, r_ips]=sci.peak_prominences(data_mean, peaks ,wlen=5)
    results_eighty = sci.peak_widths(data_mean, peaks, rel_height=0.8)
    peak_width=results_eighty[0]
    l_ips=(np.floor(results_eighty[2]))
    r_ips=(np.ceil(results_eighty[3]))
    l_ips.astype(int)
    r_ips.astype(int)
    return [peaks, prominences , l_ips, r_ips, peak_width]
    

def peaks_standard(data_mean):
# want to 
    peaks, properties = sci.find_peaks(smooth_data(data_mean,2),prominence=0.00001) 
    prominences=properties['prominences']    
    peak_width=properties['right_bases']-properties['left_bases']
    l_ips=(properties['left_bases'])    
    r_ips=(properties['right_bases'])
    #
    #l_ips=int(properties['left_bases'])    
    #r_ips=int(properties['right_bases'])
    #
    #l_ips.astype(int)
    #r_ips.astype(int)
    return [peaks, prominences , l_ips, r_ips, peak_width]

#----------------------------------------------------------------------
# Remove Estimated florescence
#----------------------------------------------------------------------

def recursive_merge(inter, start_index = 0):
    for i in range(start_index, len(inter) - 1):
        if inter[i][1] > inter[i+1][0]:
            new_start = inter[i][0]
            new_end = inter[i+1][1]
            inter[i] = [new_start, new_end]
            del inter[i+1]
            return recursive_merge(inter.copy(), start_index=i)
    return inter    

def poly4(x_data,beta):
        p=np.poly1d(beta)
        return p(x_data)
        
def positive_mse(y_pred,y_true):                
    loss=np.dot((10**8*(y_pred-y_true>0)+np.ones(y_true.size)).T,(y_pred-y_true)**2)
    return np.sum(loss)/y_true.size

def pos_mse_loss(beta, X, Y,function):                
    return positive_mse(function(X,beta), Y)

def reg_positive_mse(y_pred,y_true):                
    loss_pos=(y_pred-y_true)**2
    loss_neg=np.dot((y_pred-y_true>0),np.abs((y_pred-y_true)))
    loss=loss_pos+loss_neg
    return np.sum(loss)/y_true.size

def reg_pos_mse_loss(beta, X, Y,function):                
    return reg_positive_mse(function(X,beta), Y)

def remove_est_florescence(f_sup,data_sub):
    
    #min_data_amm=np.min(data_sub,axis=1)
    if data_sub.ndim>1:
        min_data_amm=np.mean(data_sub,axis=1)
    else:
        min_data_amm=data_sub
    
    poly_min=np.poly1d(np.polyfit(f_sup,min_data_amm,3))(f_sup)
    poly_min_pos=poly_min+min(min_data_amm-poly_min)
    
    beta_init=np.polyfit(f_sup,min_data_amm,4)
    beta_init[4]=beta_init[4]+min(min_data_amm-poly_min)
    
    #result = minimize(pos_mse_loss, beta_init, args=(f_sup,min_data_amm,poly4), method='Nelder-Mead', tol=1e-12)
    #result = minimize(mse_loss, beta_init, args=(f_sup,min_data_amm,poly4), method='Nelder-Mead', tol=1e-12)
    result = minimize(reg_pos_mse_loss, beta_init, args=(f_sup,min_data_amm,poly4), method='Nelder-Mead', tol=1e-12)
    
    beta_hat = result.x                 
        
    plt_back=1
        
    if plt_back:
            plt.figure('Pos MSE fit')
            #plt.plot(f_sup,np.mean(data_sub,axis=1),'--',label='original data')    
            plt.plot(f_sup,min_data_amm,label='original data')    
            #plt.plot(f_sup,np.mean(data_sub,axis=1),'-.',label='data minus bias')
            plt.plot(f_sup,poly4(f_sup,beta_init),'-.',label='poly 4 init')
            plt.plot(f_sup,poly4(f_sup,beta_hat),'-.',label='poly 4 hat')
            plt.legend()
            
        
    # subtract mean
    
    if data_sub.ndim>1:
        data_sub2=np.subtract(data_sub,poly4(f_sup,beta_hat).reshape([data_sub.shape[0],1]))
    else:
        data_sub2=data_sub-poly4(f_sup,beta_hat)
            
    plt_final=1
    
    if plt_final:
        plt.figure('Final recap')        
        plt.plot(f_sup,poly4(f_sup,beta_hat),'--',label='poly 4 hat')
        plt.plot(f_sup,min_data_amm,label='original data-back')    
        if data_sub.ndim>1:
            plt.plot(f_sup,np.mean(data_sub2,axis=1),label='original data-back-pos_MSE')    
        else:
            plt.plot(f_sup,data_sub2,label='original data-back-pos_MSE')    
        
        plt.legend()
        
    return data_sub2
    
#------------------------------------------------------------------------------
# main function here

def model_spectrum_ste(f_sup,data_mean_sparse,number_of_peaks):
        
    # we assume that the data is normalized, that is, the mean is equal to one
    res=data_mean_sparse.shape[0]
    method=1
    
    #------------------------------------------
    if method==0:
        [peaks, prominences , l_ips, r_ips, peak_width]=peaks_cwt_1(data_mean_sparse)
    elif method==1:
        [peaks, prominences , l_ips, r_ips, peak_width]=peaks_standard(data_mean_sparse)
    
    peak_high=data_mean_sparse[peaks]
    
    r_ips.astype(int)
    l_ips.astype(int)
    
    metric=1*peak_high+5*prominences
    
    idx_peaks=np.flip(np.argsort(metric))
    idx=peaks[idx_peaks]
    
    #############################33
    ##
    ##   TAKE ONLY THE STRONGEST ~number_of_peaks~ PEAKS
    ##
    #############################33
    rough_peak_positions = idx[0:number_of_peaks]
    
    #for  i in range(int(rough_peak_positions.size)):    
    if False:
        for  i in range(5):        
            p=idx_peaks[i]
            l_ips=int(properties['left_bases'][p])
            r_ips=int(properties['right_bases'][p])
            
            plt.plot(f_sup,data_mean_sparse)        
            plt.plot(f_sup[peaks[p]],data_mean_sparse[peaks][p],'*r')
            plt.plot(f_sup[l_ips:r_ips],data_mean_sparse[l_ips:r_ips]+0.00005,'--k')
        
        for  i in range(5):            
            p=idx_peaks[i]
            plt.plot(data_mean_sparse)
            plt.plot(peaks[p],data_mean_sparse[peaks[p]],'*')
            plt.plot(range(l_ips[p],r_ips[p]),data_mean_sparse[l_ips[p]:r_ips[p]],'--')            
            #plt.plot(range(l_win,r_win),data_mean_sparse[l_win:r_win],'--')      
    
    #----------------------------------------------------------
    # let's say that if the peak prominance is in the 75%  quantile, than we use the expand peak method only
    # if we are in the 25% quantile we worry about symmetry
    # also, the smoothing window depends on the 
    peak_prom_mean=np.mean(prominences)
    peak_width_mean=np.mean(peak_width)    
    
    # variables we store
    #------------------------------------------------------------------------------
    
    comp_range=np.zeros([rough_peak_positions.size,2])
    
    comp_beta_gauss=np.zeros([rough_peak_positions.size,4])
    comp_beta_lor=np.zeros([rough_peak_positions.size,3])
    comp_beta_gen_lor=np.zeros([rough_peak_positions.size,4])
    comp_beta_cos=np.zeros([rough_peak_positions.size,6])
    comp_MSE=np.zeros([rough_peak_positions.size,4])
    comp_bias=np.zeros([rough_peak_positions.size,1])
    
    #data_sparse_back=np.zeros([rough_peak_positions.size+1,data_sparse.shape[0],data_mean_sparse.shape[1]])
    #data_sparse_back[0,:,:]=data_sparse
        
    plot_f=0
    check_sup_fig=0
    #for  i in range(5):        
    #   for  i in range(int(rough_peak_positions.size)):
         
    for  i in range(int(rough_peak_positions.size)):
            print('Step: ',i,' - peak position: ',rough_peak_positions[i])
    
            p=idx_peaks[i]
            
            [l_win, r_win] = find_peak_sym_supp(peaks[p],int(l_ips[p]),int(r_ips[p]), 2, 3, data_mean_sparse)
            
            #[l_win, r_win],[l_ips,r_ips]
            
            if check_sup_fig:
                plt.figure('peak number '+str(int(i))+' frequency')
                plt.plot(data_mean_sparse)
                plt.plot(peaks[p],data_mean_sparse[peaks[p]],'*')
                plt.plot(range(l_ips[p],r_ips[p]),data_mean_sparse[l_ips[p]:r_ips[p]],'--')            
                plt.plot(range(l_win,r_win),data_mean_sparse[l_win:r_win],'--')            
                        
            #[l_win, r_win]=[l_ips,r_ips]
            x_data = f_sup[l_win:r_win]
            y_data = data_mean_sparse[l_win:r_win]
            #y_data = np.mean(data_sparse_back[i,l_win:r_win,:],axis=1)
            
            # this is my guess for a good strategy
            #bias_f = min(y_data)/2
            # random choice of bias
            bias_f = np.mean(np.sort(y_data)[0:int(y_data.size/5)])
            y_data = y_data -bias_f 
            
            # you can use the same intialization for gaus, lor, gen_lor        
            beta_init_lor=init_lor(x_data,y_data)
            
            
            # best gaussian fit
            result = minimize(mse_loss, beta_init_lor, args=(x_data,y_data,gauss_amp), method='Nelder-Mead', tol=1e-12)        
            beta_hat_gauss=result.x            
            MSE_gauss=result.fun
            
            # best generalized lorenzian         
            result = minimize(mse_loss, beta_init_lor, args=(x_data,y_data,gen_lor_amp), method='Nelder-Mead', tol=1e-12)        
            beta_hat_gen_lor=result.x    
            MSE_gen_lor=result.fun
            
            # best lorenzian 
            beta_init_lor2=beta_init_lor[:3]
            result = minimize(mse_loss, beta_init_lor2, args=(x_data,y_data,lor_amp), method='Nelder-Mead', tol=1e-12)        
            beta_hat_lor=result.x    
            MSE_lor=result.fun
            
            # best cosine window 
            
            beta_int_cos2=init_cos2(x_data,y_data)
            result = minimize(mse_loss, beta_int_cos2, args=(x_data,y_data,cos_win2), method='Powell', tol=1e-12)        
            beta_hat_cos2=result.x    
            MSE_cos2=result.fun
                    
            #plt.plot(x_data,y_data,label='original data') 
            #plt.plot(x_data,cos_win2(x_data,beta_int_cos2),'--',label='cos in')  
            #plt.plot(x_data,cos_win2(x_data,beta_hat_cos2),'--',label='cos')  
            
            if plot_f:
                plt.figure('peak number '+str(int(i))+' frequency')
                plt.plot(x_data,y_data,label='original data') 
                plt.plot(x_data[peaks[p]-l_win],y_data[peaks[p]-l_win],'*',label='peak')
                            
                plt.plot(x_data,gen_lor_amp(x_data,beta_hat_gen_lor),'-.',label='gen lor')
                plt.plot(x_data,lor_amp(x_data,beta_hat_lor),'-*',label='lor')
                plt.plot(x_data,cos_win2(x_data,beta_hat_cos2),'--',label='cos')  
                plt.plot(x_data,gauss_amp(x_data,beta_hat_gauss),'-|',label='gauss')  
                plt.legend()
                plt.show()
        
                    
            # store everything
            # function order is 
            # gaus
            # lor 
            # gen_lor
            # cos
            
            # store everything     
            comp_range[i]=[l_win,r_win]    
            comp_beta_gauss[i]=beta_hat_gauss
            comp_beta_lor[i]=beta_hat_lor
            comp_beta_gen_lor[i]=beta_hat_gen_lor
            comp_beta_cos[i]=beta_hat_cos2
            comp_MSE[i]=[MSE_gauss, MSE_lor, MSE_gen_lor, MSE_cos2]
            comp_bias[i]=[bias_f]
                        
        
    return [comp_range, comp_beta_gauss, comp_beta_lor, comp_beta_gen_lor, comp_beta_cos, comp_MSE, comp_bias]


def recap_spectrum(f_sup,data_mean_sparse,number_of_peaks,comp_range, comp_beta_gauss, comp_beta_lor, comp_beta_gen_lor, comp_beta_cos, comp_MSE, comp_bias):
    
    method=1
    
    #------------------------------------------
    if method==0:
        [peaks, prominences , l_ips, r_ips, peak_width]=peaks_cwt_1(data_mean_sparse)
    elif method==1:
        [peaks, prominences , l_ips, r_ips, peak_width]=peaks_standard(data_mean_sparse)
    
    peak_high=data_mean_sparse[peaks]
    
    r_ips.astype(int)
    l_ips.astype(int)
    
    metric=1*peak_high+5*prominences
    
    idx_peaks=np.flip(np.argsort(metric))
    idx=peaks[idx_peaks]
    
    #############################33
    ##
    ##   TAKE ONLY THE STRONGEST ~number_of_peaks~ PEAKS
    ##
    #############################33
    rough_peak_positions = idx[0:number_of_peaks]
    
    res=f_sup.shape[0]
    data_hat=np.zeros(res)
    
    count=np.zeros(4)
    
    for i in range(int(rough_peak_positions.size)):
        
        min_MSE_pos=int(np.argmin(comp_MSE[i,:]))
        l_win=int(comp_range[i,0])
        r_win=int(comp_range[i,1])
        x_data = f_sup[l_win:r_win]
        
        # gaus
        # lor 
        # gen_lor
        # cos
        count[min_MSE_pos]=count[min_MSE_pos]+1
        
        if    min_MSE_pos==0:     
            data_hat[l_win:r_win]=comp_bias[i]+gauss_amp(x_data,comp_beta_gauss[i])           
        elif  min_MSE_pos==1:
            data_hat[l_win:r_win]=comp_bias[i]+lor_amp(x_data,comp_beta_lor[i])           
        elif  min_MSE_pos==2:
            data_hat[l_win:r_win]=comp_bias[i]+gen_lor_amp(x_data,comp_beta_gen_lor[i])           
        else:
            data_hat[l_win:r_win]=comp_bias[i]+cos_win2(x_data,comp_beta_cos[i])           
                
    recap_peak_fit=1
    

    plt.figure('Reconstructed spectrum')
    plt.plot(f_sup,data_mean_sparse,label='original data') 
    plt.plot(f_sup[rough_peak_positions],data_mean_sparse[rough_peak_positions],'*',label='matched peaks')
    plt.plot(f_sup,data_hat,label='spectrum estimate') 
    
    for  i in range(int(rough_peak_positions.size)):    
        p=idx_peaks[i]    
        plt.annotate(str(i), (f_sup[peaks[p]],data_mean_sparse[peaks[p]]+10 ))
    
    #plt.annotate(str(i), (600,-10))
    
    plt.legend()
    plt.show()
    
    return data_hat


#---------------------------------------------------------------------------
# partition the time series

def find_win_supp(peak,l_pos,r_pos, y_up):
    # calling line
    # peaks[i],l_ips[i],r_ips[i], mean_level, y_up,2
    # peak=peaks[i]
    # l_pos=l_ips[i]
    # r_pos=r_ips[i]
    # th_win=10
    
    res=y_up.shape[0]
    
    # no smoothing here
    Y_smooth=y_up
    
    if l_pos==r_pos:        
        Y_max_idx=l_pos

    else:
        Y_max_idx=l_pos+np.argmax(Y_smooth[l_pos:r_pos])
    
    
    # go left first 
    l_pos_temp=Y_max_idx-1
    
    # win_left=np.mean(Y_smooth[l_pos_temp-th_win:l_pos_temp-1])
    win_left=np.mean(Y_smooth[l_pos:l_pos_temp-1])
            
    while win_left<=Y_smooth[l_pos_temp]  and  y_up[l_pos_temp-1]>0  and l_pos_temp>0:        
        l_pos_temp=l_pos_temp-1
        win_left=np.mean(Y_smooth[l_pos:l_pos_temp-1])
    
    # then go right
    r_pos_temp=Y_max_idx+1
    win_right=np.mean(Y_smooth[r_pos_temp+1: ])
    
    while  win_right<=Y_smooth[r_pos_temp] and y_up[r_pos_temp+1]>0 and r_pos_temp<res-1:     
        r_pos_temp=r_pos_temp+1
        win_right=np.mean(Y_smooth[r_pos_temp+1:r_pos])
        
        
    if False:
        plt.plot(range(l_pos,r_pos),y_up[l_pos:r_pos],'k--')
        plt.plot(range(l_pos,r_pos),Y_smooth[l_pos:r_pos],'y--')
        plt.plot(range(l_pos_temp,r_pos_temp),Y_smooth[l_pos_temp:r_pos_temp],'r')
                

    return [l_pos_temp,r_pos_temp]

    
#---------------------------------------------------------------------------


def data_pre_process(f_sup,data):
    # expect a 1600 times 10 times 10 file
    res=data.shape[0]
    dim_s=data.shape[1]
    data_sub=data.reshape(res,dim_s)
    
    
    min_data_amm=np.min(data_sub,axis=1)
    
    poly_min=np.poly1d(np.polyfit(f_sup,min_data_amm,3))(f_sup)
    poly_min_pos=poly_min+min(min_data_amm-poly_min)
    
    beta_init=np.polyfit(f_sup,min_data_amm,4)
    beta_init[4]=beta_init[4]+min(min_data_amm-poly_min)
    
    result = minimize(pos_mse_loss, beta_init, args=(f_sup,min_data_amm,poly4), method='Powell', tol=1e-18)
    beta_hat = result.x                 

    data_sub2=np.subtract(data_sub,poly4(f_sup,beta_hat).reshape([1,res,1])).reshape(res,10,10)
    
    fig_on=0

    if  fig_on:
        plt.figure('Check subtracting florescence')
        #plt.plot(f_sup,np.mean(data,axis=(1,2)),label)
        plt.plot(f_sup,min_data_amm,label='original data')    
        plt.plot(f_sup,poly4(f_sup,beta_hat),'--',label='Estimated florescence')
        plt.plot(f_sup,np.mean(data_sub2,axis=(1,2)),label='Sparsified data')
        plt.legend()
        
    return data_sub2


def identify_fitting_win_up(f_sup,space_mean,slide_win=100,fit_fun=gen_lor_amp, init_fit_fun=init_lor):
    # slide the window, identify the block with small high  amplitude, fit 
    
    # loop over all windows of the data of size win 
    
    # order the data, form intervals of background and peaks
    # in the backgrount interpolate a poly 
    # int the rest interpoloted the fun, fitting optimized started by init_fit_fun    

    # data_mean=np.mean(data[2],axis=1)
    
    # make even
    slide_win=int(2*int(slide_win/2))
    win_small=int(slide_win/25)

    data_temp=space_mean.copy()
    
    # smoothing here
    #-------------------------------------------------------
    
    y_data=sci.savgol_filter(np.squeeze(data_temp), window_length = 2*int(win_small)+1, polyorder = 5)
    x_data=f_sup
    
    win_poly = np.empty(2*int(data_temp.size/slide_win), dtype=np.object)
    fit_gen_lor = np.empty(2*int(data_temp.size/slide_win), dtype=np.object)

    fitting_data=np.empty((2*int(data_temp.size/slide_win),10,),dtype=object)

    
    for i in range(2*int(data_temp.size/slide_win)):
        win_poly [i] = []        
        fit_gen_lor [i] = []
    
    for i in range(int(y_data.shape[0]/slide_win)):
        
        print('Window iteration',i)
        
        # slide of half of the window at each time        
        y_data_win=y_data[int(i*(slide_win/2)):int(i*(slide_win/2)+(slide_win))].copy()
        x_data_win=x_data[int(i*(slide_win/2)):int(i*(slide_win/2)+(slide_win))].copy()

        #--------------------------------------------------------------------
        # start with the mean, then????
        #
        # check this
        #           I update the  mean level in thethreshold calculations
        #
        #
        #
        #
        mean_level_win=np.mean(y_data_win)*np.ones(y_data_win.shape[0])

        # threshold for amplitude
        # DIP DOWN ONLY VERSION: WE PUT NO ABSOLUTE VAUE
        
        th_10=np.linspace(-np.sort(-(y_data_win-mean_level_win))[int(win_small)],0,11)
    
        # total MSE of interpolation plus fitting        
        mse10=100*np.ones(th_10.shape[0])
        
        # plt.plot(x_data,y_data)        
                
        mse_th=np.zeros(10)
        min_mse_th=100
        min_j= 0
        
        for j in range(1,th_10.shape[0]-1):
            
            print('Threshold iteration:',j)
            # both high and low
            # DIP DOWN ONLY VERSION: WE PUT NO ABSOLUTE VAUE
            
            # should we update the threshold?
            pos=np.array(np.where((y_data_win-mean_level_win)<th_10[j])[0])
                        
            if False:
                plt.plot(x_data_win,y_data_win)
                plt.plot(x_data_win[pos],y_data_win[pos])
                plt.plot(x_data_win,mean_level_win)
                
            if pos.size==0:
                # what to do here? restart with the mean value 
                mean_level=np.mean(y_data)
                pos=np.array(np.where((y_data-mean_level)<th_10[j])[0])
            
            #--------------------------------------------------------------------
            #   merge the points
            #--------------------------------------------------------------------
            
            diff_pos=pos[1:]-pos[:-1]-1            
            jumps=np.where(diff_pos>0)[0]
            
            #if the final element is res, the add a jump an the end
            if pos[-1]==f_sup.shape[0]-1:
                jumps=np.append(jumps,pos.shape[0]-1)
            
            final_lb=[]
            final_rb=[]
            
            if jumps.size==0:            
                    final_lb.append(pos[0])
                    final_rb.append(pos[-1])
            else:                    
                    # check the first jumps
                    final_lb.append(pos[0])
                    final_rb.append(pos[jumps[0]])
                    
                    k=0
                    while k<jumps.shape[0]-1:
                        # 
                        final_lb.append(pos[jumps[k]+1])
                        # go to the next gap    
                        k=k+1
                        final_rb.append(pos[jumps[k]])              
                    
                    # check the last jump
                    final_lb.append(pos[jumps[k]+1])
                    final_rb.append(pos[-1]) 
            
            if False: 
                # add the first and the last intervals     
                final_lb.insert(0,int(i*(slide_win/2)))
                final_rb.insert(0,int(i*(slide_win/2))+win_small)
                
                final_lb.append(int(i*(slide_win/2)+(slide_win))-win_small)
                final_rb.append(int(i*(slide_win/2)+(slide_win)))
                            
            idx_lr=np.zeros([2,len(final_rb)])
            idx_lr[0]=np.array(final_lb)
            idx_lr[1]=np.array(final_rb)
            idx_lr.astype(int)
            idx_lr=idx_lr.T
            
            # merge intervals  that are to within 
            
            # minimum length of the intervals here
            # random choice            

            if False:
                idx_lr =np.array(recursive_merge(idx_lr.tolist()))
                
            
            idx_lr = np.array(idx_lr).astype(int)
            # remove the one intervals
            idx_lr=idx_lr[np.where(idx_lr[:,1]-idx_lr[:,0]>0)[0],:] 
            
            idx_lr_poly=idx_lr
            
            #-------------------------------------------------------------------
            # Update the fitting
            #-------------------------------------------------------------------
                
            if idx_lr.size==0:
                # weird case when the whole interval is to be fittened
                # mean_level=np.zeros(x_data.shape)
                # don't do anything
                mean_level=mean_level
            else:                
                ind_poly=[]        
                for k in range(idx_lr.shape[0]):
                    left=int(idx_lr[k,0])
                    right=int(idx_lr[k,1])  
                    ind_poly=ind_poly+list(range(left,right-1))    
            
                ind_poly=np.array(ind_poly)
                
                mean_level_win=np.poly1d(np.polyfit(x_data_win[ind_poly],y_data_win[ind_poly],2))(x_data_win)
                mse_poly= np.var(y_data_win[ind_poly]-mean_level_win[ind_poly])
                
                if False:
                    plt.plot(x_data_win,y_data_win) 
                    plt.plot(x_data_win[pos],y_data_win[pos])                    
                    plt.plot(x_data_win,mean_level_win)  
                    plt.plot(x_data_win[ind_poly],y_data_win[ind_poly],'-*')
                    
            #----------------------------------
            # fit the function with recursive
            #----------------------------------

            # first obtain the complement of the intervals
            
            if idx_lr.size==0:
                # idx_lr_comp=np.array(range(i,i+int(slide_win)-1))
                idx_lr_comp=np.array([0,slide_win-1])
                idx_lr_comp=idx_lr_comp.reshape([1,2])
            else:
                idx_lr_comp=[] 
                
                # include the first interval?
                if  idx_lr[0,0]>0:
                    idx_lr_comp.append([0,idx_lr[0,0]])
            
                for k in range(1,idx_lr.shape[0]):
                    idx_lr_comp.append([idx_lr[k-1,1],idx_lr[k,0]])
                
                # include the last interval
                if  idx_lr[-1,1]<int(slide_win)-1:
                    idx_lr_comp.append([idx_lr[-1,1],int(slide_win)])
                        
            idx_lr_comp=np.array(idx_lr_comp).reshape(int(size(idx_lr_comp)/2),2)
            
            # remove the intervals that are shorter than the number of parameters, 
            # which is SIX
            
            # this variable is updated at each threshold window
            mse_comp = []
            range_comp = []
            beta_comp = []
    
            idx_lr_comp=np.array(idx_lr_comp[np.where(idx_lr_comp[:,1]-idx_lr_comp[:,0]>6)[0],:])
            
            if idx_lr_comp.size>0:
                for k in range(idx_lr_comp.shape[0]):
                    idx_comp = range(idx_lr_comp[k,0],idx_lr_comp[k,1])
                    #x_rec = x_data_win[idx_comp]
                    x_rec = np.array(idx_comp)
                    y_rec = y_data_win[idx_comp]
                    mean_rec =  mean_level_win[idx_comp]
                    
                    # remember this 
                    
                    print('Now processing the range',idx_lr_comp[k,0],'--',idx_lr_comp[k,1])
                    
                    recursive_merge_up(x_rec, y_rec, mean_rec, th_10[j], mse_comp, range_comp, beta_comp, fit_fun, init_fit_fun)
                    
                # let's check the minimum fitting per threshold value
                mse_th = np.sum(np.array(mse_comp)[0])
                
                # MSE POLY IS TOO LARGE: WE NEED TO USE ANOTHER MEASURE
                
                if mse_th + mse_poly<min_mse_th:
                    min_mse_th = mse_th + mse_poly
                    min_j=j
                
                if False :
                print('mse_th',mse_th)
                plt.figure("Thershold"+str(int(j)))
                plt.plot(x_data_win,y_data_win)
                plt.plot(x_data_win,mean_level_win)
                
            # store everything
            
            fitting_data[i,j]=[idx_lr_poly, mse_comp, range_comp, beta_comp]
            
            
            # recap plot at the threshold level
            
            if False :
                print('mse_th',mse_th)
                plt.figure(print("Thershold", i, sep="---"))
                plt.plot(x_data_win,y_data_win)
                plt.plot(x_data_win,mean_level_win)
            
            
        # recap plot
        
        
        
    # now merge the different windows 
        
        #fitting_data[i,j]=[idx_lr_poly, mse_comp, range_comp, beta_comp]
            
                        
    return    [MSE_comp, range_comp_rec, beta_comp_rec ]




def recursive_merge_up(x_sub_win, y_sub_win,mean_level_sub, thr_sub, mse_up, range_up, beta_up, fit_fun, init_fit_fun):
    
        # x_sub_win = x_rec  
        # y_sub_win = y_rec
        # mean_level_sub = mean_rec
        # thr_sub=th_10[j]
        #
        # fit_fun = gen_lor_amp
        # init_fit_fun = init_lor
        #
        # fit_fun = cos_win2
        # init_fit_fun = init_cos3
        
        
        # fit UP            
        #----------------------------------               
        y_bias=y_sub_win-mean_level_sub
        
        y_up_sub=np.max([y_bias-np.mean(y_bias),np.zeros(y_sub_win.shape[0])],axis=0)
    
        # IMPORTANT PARAMETER HERE
        # let's ask for the prominence to be at lesat twice the threshold
        
        prom= max([3*thr_sub,min(y_up_sub[np.where(y_up_sub>0)])])
        peaks, properties = sci.find_peaks(y_up_sub,prominence=prom)     
        
        count=0
        while peaks.size==0 and count<10:
            prom=0.5*prom
            count=count+1
            peaks, properties = sci.find_peaks(y_up_sub,prominence=prom)
    
        if peaks.size==0:
            peaks, properties = sci.find_peaks(y_up_sub)
        
        if peaks.size==0:
            return
             
        prominences = properties['prominences']    
        peak_width = properties['right_bases']-properties['left_bases']
        l_ips = properties['left_bases'].astype(int)
        r_ips = properties['right_bases'].astype(int)
        l_ips.astype(int)
        r_ips.astype(int)
        
        peak_l_r_final=[]
        
        peak_l_r_win=np.zeros([peaks.size,2])
        
        for i in range(peaks.size):     
            [l_win, r_win]=find_win_supp(peaks[i],l_ips[i],r_ips[i], y_up_sub)
            peak_l_r_win[i]=[l_win, r_win]
    
        # merge intervals                       
        peak_l_r_win = np.array(recursive_merge(peak_l_r_win.tolist())).astype(int)
        
        peak_l_r_final.append(list(peak_l_r_win))
        
        # check fitting
        if False:
            plt.plot(x_sub_win,y_sub_win)
            plt.plot(x_sub_win,y_up_sub+mean_level_sub)
            plt.plot(x_sub_win,(thr_sub+mean_level_sub)*np.ones(x_sub_win.shape[0]))
        
            for i in range(peaks.size):
                #plt.plot(x_data[l_ips[i]:r_ips[i]],y_data[l_ips[i]:r_ips[i]])
                plt.plot(x_sub_win[int(peak_l_r_win[i,0]):int(peak_l_r_win[i,1])],y_sub_win[int(peak_l_r_win[i,0]):int(peak_l_r_win[i,1])],'*r')
            plt.show()            
        
        
        beta_hat_fit=np.empty((peak_l_r_win.shape[0],),dtype=object)
        mse_fit=np.empty((peak_l_r_win.shape[0],),dtype=object)
        
        # fit each window up, fit then append to the range_up, beta_up, 
        for i in range(peak_l_r_win.shape[0]):
            idx_temp=range(int(peak_l_r_win[i,0]),int(peak_l_r_win[i,1]))
            
            x_data_temp = x_sub_win[idx_temp]
            y_data_temp = y_up_sub[idx_temp]
            mean_level_temp = mean_level_sub[idx_temp]
            
            # init_fit_fun_up=init_cos3
            # fit_fun_up=cos_win2        
            
            # fit_fun_up, init_fit_fun_up,
            #beta_init=init_fit_fun(x_data_temp,y_data_temp,mean_level_temp)
            beta_init=init_fit_fun(x_data_temp,y_data_temp)
            
            #----------------------------------------------------------------------
            # how do we deal with this in general?
            #----------------------------------------------------------------------
            
            # bounds_cos2_vec=np.array([np.zeros(6) , np.inf*np.ones(6)]).T    
            # bounds_cos2=tuple(map(tuple,bounds_cos2_vec))
                
            #result = minimize(mse_loss,beta_init, args=(x_data_temp,y_data_temp,fit_fun), tol=1e-12,bounds=bounds_cos2)    
            result = minimize(mse_loss,beta_init, args=(x_data_temp,y_data_temp,fit_fun), tol=1e-12)    
            beta_hat_fit[i]=result.x
            mse_fit[i]=result.fun
            
            if False:
                plt.plot(x_data_temp,y_data_temp+mean_level_temp)            
                plt.plot(x_data_temp,mean_level_temp)
                plt.plot(x_data_temp,fit_fun(x_data_temp,beta_init)+mean_level_temp)
                plt.plot(x_data_temp,fit_fun(x_data_temp,beta_hat_fit[i])+mean_level_temp)
        
            
        # mse_up=[] 
        # range_up=[]
        # beta_up=[]
        
        
        #------------------------------------------------------
        # important bit here: the position should be ABSOLUTE, not RELATIVE        
        # append: need anything else before appending?         
        mse_up.append(mse_fit)
        #range_up.append(peak_l_r_win)
        range_up.append(x_sub_win[peak_l_r_win])
        beta_up.append(beta_hat_fit)
        
        
        #peak_l_r_final=np.array(peak_l_r_final)
        
        # find the complementary intervas
        # idx_lr=np.array(peak_l_r_final)
        idx_lr=np.array(np.squeeze(np.array(peak_l_r_final))).reshape(int(np.size(peak_l_r_final)/2),2)
        #.reshape(int(peak_l_r_final.size/2),2)
        
        slide_win=x_sub_win.shape[0]
        
        # print(" idx_lr[0,0] -- value is",  idx_lr[0,0])
        
        # [MSE_comp_rec, range_comp_rec, beta_comp_rec ]  = recursive_merge_up(x_rec, y_rec, mean_rec, thr_sub, mse_comp, range_comp, range_comp, fit_fun, init_fit_fun)
        if idx_lr.size==0:
            # idx_lr_comp=np.array(range(i,i+int(slide_win)-1))
            idx_lr_comp=np.array([0,slide_win-1])
            idx_lr_comp=idx_lr_comp.reshape([1,2])
        else:
            idx_lr_comp=[] 
            
            # include the first interval?
            if  idx_lr[0,0]>0:
                idx_lr_comp.append([0,idx_lr[0,0]])
        
            for k in range(1,idx_lr.shape[0]):
                idx_lr_comp.append([idx_lr[k-1,1],idx_lr[k,0]])
            
            # include the last interval
            if  idx_lr[-1,1]<int(slide_win)-1:
                idx_lr_comp.append([idx_lr[-1,1],int(slide_win)])    
    
        #--------------------------------------------------------------            
        
        idx_lr_comp=np.array(idx_lr_comp)     
            
        # remove the intervals that are shorter than the number of parameters, 
        # which is SIX
        
        idx_lr_comp=np.array(idx_lr_comp[np.where(idx_lr_comp[:,1]-idx_lr_comp[:,0]>6)[0],:])
    
        if idx_lr_comp.size>0:
            
            # mse_up_rec  = []
            # range_up_rec = [] 
            # beta_up_rec = []
            
            # mse_up = []
            # range_up = [] 
            # beta_up = []
                    
            
            for k in range(idx_lr_comp.shape[0]):
                idx_comp = range(idx_lr_comp[k,0],idx_lr_comp[k,1])
                x_rec = x_sub_win[idx_comp]
                y_rec = y_sub_win[idx_comp]
                mean_rec =  mean_level_sub[idx_comp]
                
                                                                       # (x_sub_win, y_sub_win,mean_level_sub, thr_sub, mse_up, range_up, beta_up, fit_fun, init_fit_fun):
                #[mse_up_rec ,range_up_rec, beta_up_rec] = 
                recursive_merge_up(x_rec, y_rec, mean_rec, thr_sub, mse_up, range_up, beta_up, fit_fun, init_fit_fun)
                
                
        return None

#[mse_up,range_up,beta_up]
        


##########--------------------------------------------------
##########--------------------------------------------------
##########--------------------------------------------------



    




#-----------------------------------------------------------------------
# FIT DOWN IS STILL A POSITIVE DEFINED FUNCTION!

def recursive_merge_up_down(x_sub_win, y_sub_win,mean_level_sub, thr_sub, mse_up, range_up, beta_up, fit_fun, init_fit_fun):
    
    # x_sub_win = x_rec  
    # y_sub_win = y_rec
    # mean_level_sub = mean_rec
    
    # fit_fun = cos_win2
    # init_fit_fun = init_cos3
    
    # fit UP            
    #----------------------------------               
    
    y_up_sub=np.max([y_sub_win-mean_level_sub,np.zeros(y_sub_win.shape[0])],axis=0)

    # IMPORTANT PARAMETER HERE
    # let's ask for the prominence to be at lesat twice the threshold
    
    prom= 3*thr_sub
    peaks, properties = sci.find_peaks(y_up_sub,prominence=prom)     
    
    while peaks.size==0:
        prom=0.9*prom
        peaks, properties = sci.find_peaks(y_up_sub,prominence=prom)     
    
    prominences = properties['prominences']    
    peak_width = properties['right_bases']-properties['left_bases']
    l_ips = properties['left_bases'].astype(int)
    r_ips = properties['right_bases'].astype(int)
    l_ips.astype(int)
    r_ips.astype(int)
    
    peak_l_r_final=[]
    
    peak_l_r_win=np.zeros([peaks.size,2])
    
    for i in range(peaks.size):     
        [l_win, r_win]=find_win_supp(peaks[i],l_ips[i],r_ips[i], y_up_sub,1)
        peak_l_r_win[i]=[l_win, r_win]

    # merge intervals                       
    peak_l_r_win = np.array(recursive_merge(peak_l_r_win.tolist())).astype(int)
    
    peak_l_r_final.append(list(peak_l_r_win))
    
    # check fitting
    if False:
        plt.plot(x_sub_win,y_sub_win)
        plt.plot(x_sub_win,y_up_sub+mean_level_sub)
        plt.plot(x_sub_win,(thr_sub+mean_level_sub)*np.ones(x_sub_win.shape[0]))
    
        for i in range(peaks.size):
            #plt.plot(x_data[l_ips[i]:r_ips[i]],y_data[l_ips[i]:r_ips[i]])
            plt.plot(x_sub_win[int(peak_l_r_win[i,0]):int(peak_l_r_win[i,1])],y_sub_win[int(peak_l_r_win[i,0]):int(peak_l_r_win[i,1])],'*r')
        plt.show()            
    
    
    beta_hat_fit=np.empty((peak_l_r_win.shape[0],),dtype=object)
    
    # fit each window up, fit then append to the range_up, beta_up, 
    for i in range(peak_l_r_win.shape[0]):
        idx_temp=range(int(peak_l_r_win[i,0]),int(peak_l_r_win[i,1]))
        
        x_data_temp = x_sub_win[idx_temp]
        y_data_temp = y_up_sub[idx_temp]
        mean_level_temp = mean_level_sub[idx_temp]
        
        # init_fit_fun_up=init_cos3
        # fit_fun_up=cos_win2        
        
        # fit_fun_up, init_fit_fun_up,
        #beta_init=init_fit_fun(x_data_temp,y_data_temp,mean_level_temp)
        beta_init=init_fit_fun(x_data_temp,y_data_temp)
        
        #----------------------------------------------------------------------
        # how do we deal with this in general?
        #----------------------------------------------------------------------
        
        # bounds_cos2_vec=np.array([np.zeros(6) , np.inf*np.ones(6)]).T    
        # bounds_cos2=tuple(map(tuple,bounds_cos2_vec))
            
        #result = minimize(mse_loss,beta_init, args=(x_data_temp,y_data_temp,fit_fun), tol=1e-12,bounds=bounds_cos2)    
        result = minimize(mse_loss,beta_init, args=(x_data_temp,y_data_temp,fit_fun), tol=1e-12)    
        beta_hat_fit[i]=result.x
            
        if False:
            plt.plot(x_data_temp,y_data_temp+mean_level_temp)            
            plt.plot(x_data_temp,mean_level_temp)
            plt.plot(x_data_temp,fit_fun(x_data_temp,beta_hat_fit[i])+mean_level_temp)
            plt.plot(x_data_temp,fit_fun(x_data_temp,beta_hat_fit[i])+mean_level_temp)
        
    # append: need anything else before appending?         
    
    range_up.append(peak_l_r_win)
    beta_up.append(beta_hat_fit)
    
    
    # fit down
    #----------------------------------               
    
    y_down_sub=np.max([-y_sub_win+mean_level_sub,np.zeros(y_sub_win.shape[0])],axis=0)
    
    # gets smootheed at higher level
    #y_down_sub=sci.savgol_filter(y_down_sub, window_length = 11, polyorder = 5)       
    
    prom= 3*thr_sub
    peaks, properties = sci.find_peaks(y_down_sub,prominence=prom)     
                  
    prominences = properties['prominences']    
    peak_width = properties['right_bases']-properties['left_bases']
    l_ips = properties['left_bases'].astype(int)
    r_ips = properties['right_bases'].astype(int)
    l_ips.astype(int)
    r_ips.astype(int)
    
    peak_l_r_win=np.zeros([peaks.size,2])
    
    for i in range(peaks.size):        
        [l_win, r_win]=find_win_supp(peaks[i],l_ips[i],r_ips[i], y_down_sub,3)
        peak_l_r_win[i]=[l_win, r_win]
                
    # merge intervals                       
    peak_l_r_win = np.array(recursive_merge(peak_l_r_win.tolist())).astype(int)
    
    peak_l_r_final.append(list(peak_l_r_win)) 
    
    # check fitting
    if False:
        # plt.plot(x_sub_win,y_sub_win)
        plt.plot(x_sub_win,y_down_sub+mean_level_sub)
        # plt.plot(x_sub_win,mean_level_sub)
        plt.plot(x_sub_win,(thr_sub+mean_level_sub)*np.ones(x_sub_win.shape[0]))
    
        for i in range(peaks.size):
            #plt.plot(x_data[l_ips[i]:r_ips[i]],y_data[l_ips[i]:r_ips[i]])
            plt.plot(x_sub_win[int(peak_l_r_win[i,0]):int(peak_l_r_win[i,1])],y_down_sub[int(peak_l_r_win[i,0]):int(peak_l_r_win[i,1])] ,'*r')
        plt.show()            

    
    beta_hat_fit=np.empty((peak_l_r_win.shape[0],),dtype=object)
    mse_fit=np.empty((peak_l_r_win.shape[0],),dtype=object)
    # fit each window down, fit then append to the range_up, beta_up, 
    
    for i in range(peak_l_r_win.shape[0]):
        idx_temp=range(int(peak_l_r_win[i,0]),int(peak_l_r_win[i,1]))
        
        x_data_temp = x_sub_win[idx_temp]
        y_data_temp = y_up_sub[idx_temp]
        mean_level_temp = mean_level_sub[idx_temp]
        
        # init_fit_fun_down=init_cos3
        # fit_fun_down=cos_win2        
        
        # fit_fun_up, init_fit_fun_up,
        beta_init=init_fit_fun_down(x_data_temp,y_data_temp,mean_level_temp)
        
        #----------------------------------------------------------------------
        # how do we deal with this in general? bounds on the fitting function?
        #----------------------------------------------------------------------
        
        bounds_cos2_vec=np.array([np.zeros(6) , np.inf*np.ones(6)]).T    
        bounds_cos2=tuple(map(tuple,bounds_cos2_vec))
            
        result = minimize(mse_loss,beta_init, args=(x_data_temp,y_data_temp,fit_fun_down), tol=1e-12,bounds=bounds_cos2)    
        beta_hat_fit[i]=result.x
        mse_fit[i]=result.fun
        
        if False:
            plt.plot(x_data_temp,y_data_temp+mean_level_temp)            
            plt.plot(x_data_temp,mean_level_temp)
            plt.plot(x_data_temp,fit_fun(x_data_temp,beta_hat_fit[i])+mean_level_temp)
            #plt.plot(x_data,fit_fun(x_data,beta_hat_fit[i]))
        
        
    mse_up.append(mse_fit)
    range_up.append(peak_l_r_win)
    beta_up.append(beta_hat_fit)
        
    peak_l_r_final=np.array(peak_l_r_final)
    
    # find the complementary intervas
    # idx_lr=np.array(peak_l_r_final)
    idx_lr=np.array(np.squeeze(np.array(peak_l_r_final))).reshape(int(peak_l_r_final.size/2),2)
    #.reshape(int(peak_l_r_final.size/2),2)
    
    slide_win=x_sub_win.shape[0]
    
    print(" idx_lr[0,0] -- value is",  idx_lr[0,0])
    
    # [MSE_comp_rec, range_comp_rec, beta_comp_rec ]  = recursive_merge_up(x_rec, y_rec, mean_rec, thr_sub, mse_comp, range_comp, range_comp, fit_fun, init_fit_fun)
    if idx_lr.size==0:
        # idx_lr_comp=np.array(range(i,i+int(slide_win)-1))
        idx_lr_comp=np.array([0,slide_win-1])
        idx_lr_comp=idx_lr_comp.reshape([1,2])
    else:
        idx_lr_comp=[] 
        
        # include the first interval?
        if  idx_lr[0,0]>0:
            idx_lr_comp.append([0,idx_lr[0,0]])
    
        for k in range(1,idx_lr.shape[0]):
            idx_lr_comp.append([idx_lr[k-1,1],idx_lr[k,0]])
        
        # include the last interval
        if  idx_lr[-1,1]<int(slide_win)-1:
            idx_lr_comp.append([idx_lr[-1,1],int(slide_win)])    

    #--------------------------------------------------------------            
    
    idx_lr_comp=np.array(idx_lr_comp)     
        
    # remove the intervals that are shorter than the number of parameters, 
    # which is SIX
    
    idx_lr_comp=np.array(idx_lr_comp[np.where(idx_lr_comp[:,1]-idx_lr_comp[:,0]>6)[0],:])

    if idx_lr_comp.size>0:
        
        for k in range(idx_lr_comp.shape[0]):        
            idx_comp = range(idx_lr_comp[k,0],idx_lr_comp[k,1])
            x_rec = x_data_win[idx_comp]
            y_rec = y_data_win[idx_comp]
            mean_rec =  mean_level_win[idx_comp]
            
            # print('Step: ',i,' - peak position: ',rough_peak_positions[i])                       
 
            print('Now processing the range',idx_lr_comp[k,0],'--',idx_lr_comp[k,1])
        
            [mse_rec ,range_up_rec, beta_up_rec ] = recursive_merge_up(x_rec, y_rec,mean_rec, thr_sub, range_up, beta_up, fit_fun, init_fit_fun)
            
            mse_up.append(mse_up)
            range_up.append(range_up_rec)
            beta_up.append(beta_up_rec)
        
    return [mse_up,range_up,beta_up]
        
