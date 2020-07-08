#-----------------------------------------------
# fit the peaks now
#-----------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as p
import scipy.signal as sci
import scipy.stats as stats
# import pymf3
from math import pi

from scipy.optimize import leastsq
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize


res=1600
dim_s=100


#all spectrums
i_file= open('original_MG_solution.dat', "r")
temp = i_file.readlines()
temp=[i.strip('\n') for i in temp]
data_o=np.array([(row.split('\t')) for row in temp], dtype=np.float32)

f_vec=data_o[:,0]
data_o=data_o[:,1]

#plt.plot(f_vec,data_o)

I=np.trapz(data_o,f_vec)
data_o=data_o/I


#normalize the area to one
#
i_file= open('1500ppb_all_spectrum.dat', "r")
temp = i_file.readlines()
temp=[i.strip('\n') for i in temp]
data1500=np.array([(row.split('\t')) for row in temp], dtype=np.float32)
f_1500=data1500[:,0]
data1500=data1500[:,1:101]
I=[np.trapz(data1500[:,i],f_1500) for i in range(100)]
data1500=data1500/I
data1500=data1500.reshape([1600,10,10])


i_file= open('15ppb_all_spectrum.dat', "r")
temp = i_file.readlines()
temp=[i.strip('\n') for i in temp]
data15=np.array([(row.split('\t')) for row in temp], dtype=np.float32)
f_15=data15[:,0]
data15=data15[:,1:101]
I=[np.trapz(data15[:,i],f_15) for i in range(100)]
data15=data15/I
data15=data15.reshape([1600,10,10])


i_file= open('1.5ppb_all_spectrum.dat', "r")
temp = i_file.readlines()
temp=[i.strip('\n') for i in temp]
data1p5=np.array([(row.split('\t')) for row in temp], dtype=np.float32)
f_1p5=data1p5[:,0]
data1p5=data1p5[:,1:101]
I=[np.trapz(data1p5[:,i],f_1p5) for i in range(100)]
data1p5=data1p5/I
data1p5=data1p5.reshape([1600,10,10])



# from loader import *

# ----------------------------------------------------------------------------
# Reading file without loader

# i_file= open('1500ppb_all_spectrum.dat', "r")
# temp = i_file.readlines()
# temp=[i.strip('\n') for i in temp]
# data_1500ppb=np.array([(row.split('\t')) for row in temp], dtype=np.float32)
# f_1500=data_1500ppb[:,0]
# data_1500ppb=data_1500ppb[:,1:101]
# I=[np.trapz(data_1500ppb[:,i],f_1500) for i in range(100)]
# data_1500ppb=data_1500ppb/I
# data_1500ppb=data_1500ppb.reshape([1600,10,10])

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
    return np.sum(loss)/y_true.size

def pos_mse_loss(beta, X, Y,function):                
    return positive_mse(function(X,beta), Y)

def mse_loss(beta,X, Y,function):
    return sum((function(X,beta)-Y)**2)/Y.size

def find_peak_sym_supp(peak,l_pos,r_pos, win_len_mul, smooth_win, Y):
    # win_len_mul= how much do we extend the window length 
    # smooth_win= how much do we smooth the function
            
    Y_smooth=smooth_data(Y,smooth_win)
    
    
    l_pos
    r_pos
    
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
        
    # plt.plot(Y_smooth)
    # plt.plot(range(l_pos,r_pos),y_data+1,'--')    
    
    l_pos_temp=Y_max_idx-1
    r_pos_temp=Y_max_idx+1
        
    while  Y_smooth[l_pos_temp-1]<=Y_smooth[l_pos_temp]  and l_pos_temp>0:
        l_pos_temp=l_pos_temp-1
    
    while  Y_smooth[r_pos_temp+1]<=Y_smooth[r_pos_temp] and r_pos_temp<res-2: 
        r_pos_temp=r_pos_temp+1
        
    return [l_pos_temp,r_pos_temp]


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

def is_solvent(y_data):
    # check the time we are under the mean minus the standard deviation
    perc=np.sum(y_data<np.mean(y_data)-1/2*np.sqrt(np.var(y_data)))/y_data.size<0.5
    start_up=np.mean(y_data[:int(y_data.size*0.1)])>np.mean(y_data)
    return perc and start_up
    
#y_data>np.mean(np.sort(y_data)[0:int(y_data.size/25)])
   
#------------------------------------------------------------------------------

INF=10**4

def one_side_MSE (y_pred,y_true):        
    if  y_pred-y_true>0:
        loss=INF*(y_pred-y_true)**2
    else:
        loss=(y_pred-y_true)**2            
    return(loss)


def obj_one_side_MSE(beta, X, Y):
    p=np.poly1d(beta)
    error = sum([one_side_MSE(p(X[i]), Y[i]) for i in range(X.size)])/X.size
    return(error)

def quick_remove_bias(data):    
    # find the minimum over the spacial dimension
    min_spec=np.min(data,axis=(1,2))    
        
    mean = np.mean(min_spec)
    min_spec -= np.mean(min_spec)
    autocorr_f = np.correlate(min_spec, min_spec, mode='full')
    mid=int(np.where(autocorr_f==max(autocorr_f ))[0])
    temp = autocorr_f[mid:]/autocorr_f[mid]    
    # plt.plot(temp)
   
    min_spec=np.min(data,axis=(1,2))          
    beta_init=np.polyfit(f_vec,min_spec,3)
    poly_min=np.poly1d(np.polyfit(f_vec,min_spec,3))(f_vec)
    beta_init[3]=beta_init[3]+min(min_spec-poly_min)

    result = minimize(obj_one_side_MSE, beta_init, args=(f_vec,min_spec), method='Nelder-Mead', tol=1e-9)   

    beta_hat = result.x                 
    beta_hat -beta_init
    poly_min_hat=np.poly1d(beta_hat)(f_vec)
    
    poly_min_hat=poly_min_hat+min(min_spec-poly_min_hat)
    
    plt.plot(poly_min_hat)
    plt.plot(min_spec)
    
    return poly_min_hat
    

def smooth_data(data,win):
    return np.array([np.mean(data[int(np.max([j-win,0])):int(np.min([j+win,data.size]))]) for j in  range(data.size)])


#------------------------------------------------------------------------------

res=1600
dim_s=100

poly_min_hat = quick_remove_bias(data1500)

data_sparse = np.reshape(data1500, [res, dim_s]) - np.reshape(poly_min_hat,[res,1])
data_sparse = data_sparse - np.min(data_sparse)
#------------------------------------------------------------------------------
# NORMALIZE THE SPECTRUM SO THAT THE AVERAGE AMPLITUDE IS ONE
#------------------------------------------------------------------------------

specMean=np.mean(data_sparse)

data_sparse=data_sparse/specMean
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
    [ prominences , l_ips, r_ips]=sci.peak_prominences(data_mean, peaks ,wlen=10)
    results_eighty = sci.peak_widths(data_mean, peaks, rel_height=0.8)
    peak_width=results_eighty[0]
    l_ips=(np.floor(results_eighty[2])-1)
    r_ips=(np.ceil(results_eighty[3])+1)
    l_ips.astype(int)
    r_ips.astype(int)
    return [peaks, prominences , l_ips, r_ips, peak_width]    

def peaks_standard(data_mean):
# want to 
    peaks, properties = sci.find_peaks(data_mean,prominence=0.01) 
    prominences=properties['prominences']    
    peak_width=properties['right_bases']-properties['left_bases']
    l_ips=np.floor(properties['left_bases']) 
    l_ips=l_ips-5
    r_ips=np.ceil(properties['right_bases'])
    r_ips=r_ips+5
    
    l_ips.astype(int)
    r_ips.astype(int)
    return [peaks, prominences , l_ips, r_ips, peak_width]

method=1

#------------------------------------------
if method==0:
    [peaks, peak_prom, l_ips, r_ips, peak_width]=peaks_cwt_1(data_mean_sparse)
elif method==1:
    [peaks, peak_prom , l_ips, r_ips, peak_width]=peaks_standard(data_mean_sparse)

peak_high=data_mean_sparse[peaks]


# # ANY CHOICE OF THE METRIC HERE!?

# metric=10*peak_high+5*peak_prom+peak_width
metric=10*peak_high + 5*peak_prom

idx_peaks=np.flip(np.argsort(metric))
idx=peaks[idx_peaks]
rough_peak_positions = idx

#for  i in range(int(rough_peak_positions.size)):    
if False:
    for  i in range(5):        
        p=idx_peaks[i]

        l_ips.astype(int)
        r_ips.astype(int)
        
        plt.plot(f_vec,data_mean_sparse)        
        plt.plot(f_vec[peaks[p]],data_mean_sparse[peaks][p],'*')
        plt.plot(f_vec[int(l_ips[p]):int(r_ips[p])],data_mean_sparse[int(l_ips[p]):int(r_ips[p])]+1,'--')
    
#----------------------------------------------------------
# let's say that if the peak prominance is in the 75%  quantile, than we use the expand peak method only
# if we are in the 25% quantile we worry about symmetry
# also, the smoothing window depends on the 
peak_prom_mean=np.mean(prominences)
peak_width_mean=np.mean(peak_width)

# variables we store
#------------------------------------------------------------------------------
    
# backup of the data
data_sparse_back=np.zeros([rough_peak_positions.size+1,data_sparse.shape[0],data_sparse.shape[1]])
# inital boundaries and boudaries after expansion
comp_range=np.zeros([rough_peak_positions.size,2])
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
plot_t=0
check_sup_fig=0

#for  i in range(5):        
 #   for  i in range(int(rough_peak_positions.size)):
     
for  i in range(int(rough_peak_positions.size)):
        print('Step: ',i,' -peak position: ',rough_peak_positions[i])

        p=idx_peaks[i]
        
        l_ips = l_ips.astype(int)
        r_ips = r_ips.astype(int)
        
        [l_win, r_win] = find_peak_sym_supp(peaks[p],int(l_ips[p]),int(r_ips[p]), 2, 3, data_mean_sparse)
        
        #[l_win, r_win],[l_ips,r_ips] 
        if check_sup_fig:
            plt.plot(data_mean_sparse)
            plt.plot(peaks[p],data_mean_sparse[peaks[p]],'*')
            plt.plot(range(l_ips[p],r_ips[p]),data_mean_sparse[l_ips[p]:r_ips[p]]+2,'--')            
            plt.plot(range(l_win,r_win),data_mean_sparse[l_win:r_win]+4,'--')            
                    
        #[l_win, r_win]=[l_ips,r_ips]
        x_data = f_vec[l_win:r_win]
        y_data = np.mean(data_sparse_back[i,l_win:r_win,:],axis=1)
        
        # this is my guess for a good strategy
        #bias_f = min(y_data)/2
        # random choice of bias
        bias_f = np.mean(np.sort(y_data)[0:int(y_data.size/5)])
        y_data = y_data -bias_f 
        
        beta_init_lor=init_lor(x_data,y_data)
               
        result = minimize(pos_mse_loss, beta_init_lor, args=(x_data,y_data,gen_lor_amp), method='Nelder-Mead', tol=1e-8)
        #result = minimize(mse_loss, beta_init_lor, args=(x_data,y_data,gen_lor_amp), method='Nelder-Mead', tol=1e-12)
        beta_hat_lor=result.x    
        
        if plot_f:
            plt.figure('peak number '+str(int(i))+' frequency')
            plt.plot(x_data,y_data)            
            plt.plot(x_data[peaks[p]-l_win],y_data[peaks[p]-l_win],'*')                        
            plt.plot(x_data,gen_lor_amp(x_data,beta_init_lor),'--')
            plt.plot(x_data,gen_lor_amp(x_data,beta_hat_lor),'-*')
    
        # pick  a single point to fix
        # y_data=data_sparse_back[i,int((r_win+l_win)/2),:]    
        # x_data=np.array(range(dim_s))
    
        # #------------------------------------------------------------------------------
        # # is this a water peak? check the mean!
        
        # solve=is_solvent(y_data)
        
        # if  solve:
        #     bias_t=np.mean(np.sort(y_data)[::-int(y_data.size/10)+1])
        #     #bias_t=np.max(y_data)
        #     y_data=bias_t-y_data
        #     y_data=y_data-np.min(y_data)
        # else:
        #     #bias_t=np.min(y_data)
        #     bias_t=np.mean(np.sort(y_data)[::int(y_data.size/10)+1])
        #     y_data=y_data-bias_t
        #     y_data=y_data-np.min(y_data)    
        
        # bounds_cos_vec=np.array([np.zeros(5) , np.inf*np.ones(5)]).T    
        # bounds_cos=tuple(map(tuple, bounds_cos_vec))
        
        # beta_init_cos=np.array(init_cos(x_data,y_data,5))
        # #result = minimize(pos_mse_loss, beta_init_cos, args=(x_data,y_data,cos_win), tol=1e-8,bounds=bounds_cos)    
        # result = minimize(mse_loss, beta_init_cos, args=(x_data,y_data,cos_win), tol=1e-12,bounds=bounds_cos)    
        
        # beta_hat_cos=result.x
            
        # if plot_t:
        #     plt.figure('peak number '+str(int(i))+' time')
        #     plt.plot(x_data,(1-2*solve)*y_data,'*')            
        #     plt.plot(x_data,(1-2*solve)*np.array(cos_win(x_data,beta_init_cos)),'--')        
        #     plt.plot(x_data,(1-2*solve)*np.array(cos_win(x_data,beta_hat_cos)),'--')        
        
        # # refit the lorenzian in the smaller interval        
        # # use the previous estiamate to start
        # y_data=np.mean(data_sparse_back[i,l_win:r_win,int(beta_hat_cos[1]):int(sum(beta_hat_cos[2:5]))],axis=1)        
        # x_data = f_vec[l_win:r_win]    
        # #bias_f2 = 0
        # bias_f2 = np.mean(np.sort(y_data)[0:int(y_data.size/10)+1])
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
        #     plt.plot(x_data,gen_lor_amp(x_data,beta_hat_lor2),'--')
        
        # store everything     
        comp_range[i]=[l_win,r_win]    
        comp_beta_lor[i]=beta_init_lor
        # comp_beta_cos[i]=beta_hat_cos
        comp_bias[i]=[bias_f]
        # comp_up_down[i]=solve
        # don't know what to subtract....
        data_sparse_back[i+1]=data_sparse_back[i]


#---------------------------------------------------------
# -> clustering 

# plot the distirbution of oringin and the duration 
# cos_window(t,M,o,d,al,be)
#              0,1,2,3 ,4

plt.figure('Time fitting summary')

# start finish dotted line
idx_d_l=[1]
idx_d_r=[1,2,3,4]
# start finish solid line
idx_s_l=[1,3]
idx_s_r=[1,3,2]



for  i in range(int(rough_peak_positions.size)):
#for  i in range(5):           
    p=idx_peaks[i]
    if comp_beta_cos[i,1]>=1:
        if comp_up_down[i]:        
            c='k'
        else:
            c='r'            
    else:
            c='b'
            
    plt.plot(f_vec[peaks[p]],np.sum(comp_beta_cos[i,idx_s_l]),color=c,marker='*')            
    plt.plot(f_vec[peaks[p]],np.sum(comp_beta_cos[i,idx_s_r]),color=c,marker='*')            
    plt.plot(f_vec[peaks[p]],comp_beta_cos[i,1],color=c,marker='o')
    plt.plot(f_vec[peaks[p]],sum(comp_beta_cos[i,idx_d_r]),color=c,marker='o')
    plt.plot([f_vec[peaks[p]],f_vec[peaks[p]]],[np.sum(comp_beta_cos[i,idx_d_l]),sum(comp_beta_cos[i,idx_d_r])],'--'+c)
    plt.plot([f_vec[peaks[p]],f_vec[peaks[p]]],[np.sum(comp_beta_cos[i,idx_s_l]),sum(comp_beta_cos[i,idx_s_r])],c)

    # add the peak number 
    #plt.annotate(str(i), ([f_vec[peaks[p]],sum(comp_beta_cos[i,idx_s_r])))
        
    plt.annotate(str(i), (f_vec[peaks[p]],sum(comp_beta_cos[i,idx_d_r])+20))
                                                  
                                                  
plt.plot(f_vec,data_mean_sparse*100)
#plt.plot(f_vec[peaks],100*data_mean_sparse[peaks],'*r')
             

X=comp_beta_cos[:,1]
Y=comp_beta_cos[:,2]

idx_sol=np.where(comp_up_down==1)
Xs=comp_beta_cos[idx_sol,1]
Ys=comp_beta_cos[idx_sol,2]


plt.plot(X,Y,'*r')
plt.plot(Xs,Ys,'*r')
plt.xlabel('origin')
plt.ylabel('duration')
# X is the origin
# Y is the duration

# np.min(data_sparse_back[i+1])
    
# plt.plot(np.mean(data_sparse_back[i],axis=1))
# plt.plot(np.mean(data_sparse_back[i+1],axis=1))
  
            
# plt.pcolormesh(data_sub.T)
# plt.pcolormesh(data_sparse_back[i+1])    
    
    
# data_sparse_back[i+1]=
    
    
# X, Y = np.meshgrid(f_vec, np.array(range(dim_s)))
# Z = data_sub
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.contour3D(X, Y, Z, 50, cmap='binary')
# Z1 = data_sparse_back[i].T
# ax.contour3D(X, Y, Z1, 50, cmap='binary')
    
    
    