from itertools import islice

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

#------------------------------------------------------------------------------

res=1600
dim_s=100

# turn on the loader

loader_on=1

if loader_on:
    
    #all spectrums
    i_file= open('data/MG set 1/original_MG_solution.dat', "r")
    temp = i_file.readlines()
    temp=[i.strip('\n') for i in temp]
    data_o=np.array([(row.split('\t')) for row in temp], dtype=np.float32)
    
    f_sup=data_o[:,0]
    data_o=data_o[:,1]
    
    #plt.plot(f_sup,data_o)
    
    I=np.trapz(data_o,f_sup)
    data_o=data_o/I
    
    
    #normalize the area to one
    #
    i_file= open('data/MG set 1/1500ppb_all_spectrum.dat', "r")
    temp = i_file.readlines()
    temp=[i.strip('\n') for i in temp]
    data1500=np.array([(row.split('\t')) for row in temp], dtype=np.float32)
    f_1500=data1500[:,0]
    data1500=data1500[:,1:101]
    I=[np.trapz(data1500[:,i],f_1500) for i in range(100)]
    data1500=data1500/I
    data1500=data1500.reshape([1600,10,10])
    
    
    i_file= open('data/MG set 1/15ppb_all_spectrum.dat', "r")
    temp = i_file.readlines()
    temp=[i.strip('\n') for i in temp]
    data15=np.array([(row.split('\t')) for row in temp], dtype=np.float32)
    f_15=data15[:,0]
    data15=data15[:,1:101]
    I=[np.trapz(data15[:,i],f_15) for i in range(100)]
    data15=data15/I
    data15=data15.reshape([1600,10,10])
    
    
    i_file= open('data/MG set 1/1.5ppb_all_spectrum.dat', "r")
    temp = i_file.readlines()
    temp=[i.strip('\n') for i in temp]
    data1p5=np.array([(row.split('\t')) for row in temp], dtype=np.float32)
    f_1p5=data1p5[:,0]
    data1p5=data1p5[:,1:101]
    I=[np.trapz(data1p5[:,i],f_1p5) for i in range(100)]
    data1p5=data1p5/I
    data1p5=data1p5.reshape([1600,10,10])
    
    init_plot=0
    
    if init_plot:
        plt.figure('Data available recap')
        plt.plot(f_sup,data_o,label='clean data')
        plt.plot(f_sup,np.mean(data1500,axis=(1,2))+0.0003,label='1500ppb')
        plt.plot(f_sup,np.mean(data15,axis=(1,2))+0.0006,label='15ppb')
        plt.plot(f_sup,np.mean(data1p5,axis=(1,2))+0.0009,label='1.500ppb')
        plt.legend()
        
#data1500[0,:]-data1p5[0,:] 
#data1p5[0,:] 

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

def positive_mse(y_pred,y_true):                
    loss=np.dot((10**8*(y_pred-y_true>0)+np.ones(y_true.size)).T,(y_pred-y_true)**2)
    return sum(loss)/y_true.size

def pos_mse_loss(beta, X, Y,function):                
    return positive_mse(function(X,beta), Y)


def positive_mse_loss(beta, X, Y):
    p=np.poly1d(beta)
    error = sum([positive_mse(p(X[i]), Y[i]) for i in range(X.size)])/X.size
    return(error)


def mse_loss(beta,X, Y,function):
    return sum((function(X,beta)-Y)**2)/Y.size

def find_peak_sym_supp(peak,l_pos,r_pos, win_len_mul, smooth_win, Y):
    # win_len_mul= how much do we extend the window length 
    # smooth_win= how much do we smooth the function
            
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
    power=np.dot(y_data,y_data.T)/dim_s
        
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
# Start  processing data

# data_MG=data1500
# data_MG=data_o
#mean_data_MG=np.mean(data1500.reshape(res,dim_s),axis=1)

mean_data_MG=data_o

mean_wo_peaks=smooth_data(mean_data_MG.copy(),5)
peaks, properties = sci.find_peaks(mean_data_MG,width=1/10000)

mean_wo_peaks=mean_data_MG.copy()
for p in range(peaks.shape[0]):
    # find the alpha & beta coefficient between the edges    
    lin_int=properties["width_heights"][p]
    mean_wo_peaks[int(peaks[p]-properties["widths"][p]):int(peaks[p]+properties["widths"][p])]=lin_int
    

for p in range(peaks.shape[0]):
    # find the alpha & beta coefficient between the edges
        
    l_pos = int(np.floor(properties["left_ips"][p]))
    r_pos = int(np.ceil(properties["right_ips"][p]))
    
    win_len=r_pos-l_pos
    
    l_max = max(0,l_pos-10*win_len)
    r_max = min(res-2,r_pos+10*win_len)
    
    # l_max = 0
    # r_max = res-2
    # extend the window to make sure that the spectrum is decreasing arond the window
    
    while  (mean_wo_peaks[l_pos-1]<=mean_wo_peaks[l_pos]) and (l_pos>l_max):
        l_pos=l_pos-1
    
    while  (mean_wo_peaks[r_pos+1]<=mean_wo_peaks[r_pos]) and (r_pos<r_max):
        r_pos=r_pos+1
    
    # plt.plot(l_pos,mean_wo_peaks[l_pos],'*')
    # plt.plot(r_pos,mean_wo_peaks[r_pos],'+')
    
    win = range(l_pos,r_pos)
    win_len=r_pos-l_pos
    min_win=min(mean_wo_peaks[win]).copy()
        
    mean_wo_peaks[l_pos:r_pos]=min_win
  

poly_min=np.poly1d(np.polyfit(f_sup,mean_wo_peaks,3))(f_sup)

poly_min_pos=poly_min+min(mean_wo_peaks-poly_min)


beta_init=np.poly1d(np.polyfit(f_sup,mean_wo_peaks,3))(f_sup)
beta_init[3]=beta_init[3]+min(mean_wo_peaks-poly_min)

beta_init=np.polyfit(f_sup,mean_wo_peaks,3)
beta_init[3]=beta_init[3]+min(mean_wo_peaks-poly_min)

result = minimize(positive_mse_loss, beta_init, args=(f_sup,mean_data_MG), method='Nelder-Mead', tol=1e-3)
#minimze(zero_one_loss, [0 1],args=(x_vec,y_vec))
#
#zero_one_loss(a_and_b,X,Y )
#loss( square_win(a_and_b,X),Y)/X.size

beta_init=np.polyfit(f_sup,mean_wo_peaks,4)
beta_init[4]=beta_init[4]+min(mean_wo_peaks-poly_min)

result = minimize(positive_mse_loss, beta_init, args=(f_sup,mean_data_MG), method='Nelder-Mead', tol=1e-3)

beta_hat = result.x                 
beta_hat -beta_init
poly_min_hat=np.poly1d(beta_hat)(f_sup)


#data_sparse=data_MG.reshape(res,dim_s)-np.reshape(poly_min_hat,[res,1])
#data_sparse=data_sparse-np.min(data_sparse)

data_sparse=mean_data_MG.reshape(res,1)-np.reshape(poly_min_hat,[res,1])

data_mean_sparse=np.mean(data_sparse,axis=1)


# algorthim: 
    # find peaks
    # expand symmetrically
    # fint gen_lorenz
    # remove data
    # repeat!
    
# FIND THE PEAKS 

# peaks, properties = sci.find_peaks(smooth_data(data_mean_sparse,2),prominence=0.0005,width=2) 
#peaks, properties = sci.find_peaks(smooth_data(data_mean_sparse,2),prominence=0.0005) 

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
##   TAKE ONLY THE STRONGEST 10 PEAKS
##
#############################33
rough_peak_positions = idx[0:10]

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

data_sparse_back=np.zeros([rough_peak_positions.size+1,data_sparse.shape[0],data_sparse.shape[1]])
data_sparse_back[0,:,:]=data_sparse

# MAIN LOOP OVER THE PEAKS
#------------------------------------------------------------------------------

plot_f=0
check_sup_fig=0
#for  i in range(5):        
 #   for  i in range(int(rough_peak_positions.size)):
     
for  i in range(int(rough_peak_positions.size)):
        print('Step: ',i,' -peak position: ',rough_peak_positions[i])

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
        y_data = np.mean(data_sparse_back[i,l_win:r_win,:],axis=1)
        
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
        
        
        # don't know what to subtract....
        data_sparse_back[i+1]=data_sparse_back[i]
        

#----------------------------------------------------------------------------
#recap plot after the fitting
#----------------------------------------------------------------------------

recap_peak_fit=1

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

if recap_peak_fit:
    plt.figure('Reconstructed spectrum')
    plt.plot(f_sup,data_mean_sparse,label='original data') 
    plt.plot(f_sup[rough_peak_positions],data_mean_sparse[rough_peak_positions],'*',label='matched peaks')
    plt.plot(f_sup,data_hat,label='spectrum estimate') 
    plt.legend()
    plt.show()
    
# CORRELATE SPATIALLY

#------------------------------------------------------------------------------
# pre-precess pipeline


def poly4(x_data,beta):
        p=np.poly1d(beta)
        return p(x_data)
        

def positive_mse(y_pred,y_true):                
    loss=np.dot((10**8*(y_pred-y_true>0)+np.ones(y_true.size)).T,(y_pred-y_true)**2)
    return np.sum(loss)/y_true.size

def pos_mse_loss(beta, X, Y,function):                
    return positive_mse(function(X,beta), Y)

def data_pre_process(data):
    # expect a 1600 times 10 times 10 file
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


# start with data 15
data1500p=data_pre_process(data1500)
data15p=data_pre_process(data15)
data1p5p=data_pre_process(data1p5)


# total correlation plot

corr_1500= np.dot(data_hat,data1500p.reshape(res,dim_s)).reshape(10,10)
corr_15= np.dot(data_hat,data15p.reshape(res,dim_s)).reshape(10,10)
corr_1p5= np.dot(data_hat,data1p5p.reshape(res,dim_s)).reshape(10,10)

plot_cor_space=1

if plot_cor_space:
    plt.figure('Total correlation in space')
    plt.plot(corr_1500.reshape(dim_s),label='1500ppb')
    plt.plot(corr_15.reshape(dim_s),label='15ppb')
    plt.plot(corr_1p5.reshape(dim_s),label='1.5ppb')
    plt.legend()
    plt.show()


# per peak correlation

mean_corr_1500=np.zeros(int(rough_peak_positions.size))
mean_corr_15=np.zeros(int(rough_peak_positions.size))
mean_corr_1p5=np.zeros(int(rough_peak_positions.size))



for i in range(int(rough_peak_positions.size)):
               
    l_win=int(comp_range[i,0])
    r_win=int(comp_range[i,1])
    x_data = f_sup[l_win:r_win]
    
    #np.dot(data_hat,data1p5p.reshape(res,dim_s)).reshape(10,10)
     
    mean_corr_1500[i]=np.mean(np.dot(data_hat[l_win:r_win],data1500p[l_win:r_win].reshape([r_win-l_win,dim_s])))
    mean_corr_15[i]=np.mean(np.dot(data_hat[l_win:r_win],data15p[l_win:r_win].reshape([r_win-l_win,dim_s])))
    mean_corr_1p5[i]=np.mean(np.dot(data_hat[l_win:r_win],data1p5p[l_win:r_win].reshape([r_win-l_win,dim_s])))                         
    # gaus
    # lor 
    # gen_lor
    # cos

plot_corr_peaks=1

if plot_corr_peaks:
    plt.figure('Total correlation per peak')
    plt.plot(mean_corr_1500,'-*',label='1500ppb')
    plt.plot(mean_corr_15,'-.',label='15ppb')
    plt.plot(mean_corr_1p5,'-|',label='1.5ppb')
    plt.xticks(np.arange(int(rough_peak_positions.size)), f_sup[rough_peak_positions])
    plt.legend()
    plt.show()
    