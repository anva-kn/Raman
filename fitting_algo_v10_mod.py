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
    return sum(loss)/y_true.size

def pos_mse_loss(beta, X, Y,function):                
    return positive_mse(function(X,beta), Y)

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
        
    # plt.plot(Y_smooth)
    # plt.plot(range(l_pos,r_pos),y_data+1,'--')    
    
    
    

    l_pos_temp=Y_max_idx-1
    r_pos_temp=Y_max_idx+1
        
    while  Y_smooth[l_pos_temp-1]<=Y_smooth[l_pos_temp]  and l_pos_temp>0:
        l_pos_temp=l_pos_temp-1
    
    while  Y_smooth[r_pos_temp+1]<=Y_smooth[r_pos_temp] and r_pos_temp<dim_s-1: 
        r_pos_temp=r_pos_temp+1
        
    return [l_pos_temp,r_pos_temp]

def cos_window(t,M,o,d,al,be):
    
    if o<=t & t<o+al:
        out=M/2*(1-np.cos(2*pi*(t-o)/(2*al)))
    elif o+al<=t & t<o+al+d: 
        out=M
    elif o+al+d<= t & t<o+al+d+be:
        out=M/2*(1-np.cos(2*pi*(be-(t-d-o-al))/(2*be)))
    else:
        out=0
    return  out

def cos_win(x_data,beta):    
    M=beta[0]
    o=beta[1]
    d=beta[2]
    al=beta[3]
    be=beta[4]
    
    yvec=[cos_window(t,M,o,d,al,be)  for t in range(x_data.size)]
    return yvec

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
    power=np.dot(y_data,y_data.T)/dim_s
        
    floor_th=0.1*np.sqrt(power)
    # how to define the ceiling of the function
    ceil_th=0.9*np.sqrt(power)    
    
    #  while extending 0 l_win_low /  win_up / l_win_high-r_win_high / win_down / r_win_low-dim_s
    
    l_win_low  = np.argmax(floor_th<y_smoth) 
    r_win_low =dim_s- np.argmax(floor_th<np.flip(y_smoth) )
    
    l_win_high =  np.argmax(ceil_th<y_smoth) 
    r_win_high =dim_s- np.argmax(ceil_th<np.flip(y_smoth) )    
    
    # M_init =ceil_th
    # o_init = l_win_low  
    # d_init = r_win_high -l_win_high 
    # al_init= l_win_high-l_win_low  
    # be_init= r_win_low- r_win_high
    
    beta_cos_int=[ceil_th, l_win_low, r_win_high -l_win_high, l_win_high-l_win_low, r_win_low- r_win_high]
    
    return beta_cos_int

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
    [ prominences , l_ips, r_ips]=sci.peak_prominences(data_mean, peaks ,wlen=50)
    results_eighty = sci.peak_widths(data_mean, peaks, rel_height=0.8)
    peak_width=results_eighty[0]
    l_ips=(np.floor(results_eighty[2]))
    r_ips=(np.ceil(results_eighty[3]))
    l_ips.astype(int)
    r_ips.astype(int)
    return [peaks, prominences , l_ips, r_ips, peak_width]
    

def peaks_standard(data_mean):
# want to 
    peaks, properties = sci.find_peaks(smooth_data(data_mean_sparse,2),prominence=0.01) 
    prominences=properties['prominences']    
    peak_width=properties['right_bases']-properties['left_bases']
    l_ips=int(properties['left_bases'])    
    r_ips=int(properties['right_bases'])
    l_ips.astype(int)
    r_ips.astype(int)
    return [peaks, prominences , l_ips, r_ips, peak_width]

method=0

#------------------------------------------
if method==0:
    [peaks, prominences , l_ips, r_ips, peak_width]=peaks_cwt_1(data_mean_sparse)
elif method==1:
    [peaks, peak_prom , l_ips, r_ips, peak_width]=peaks_standard(data_mean_sparse)

peak_high=data_mean_sparse[peaks]

r_ips.astype(int)
l_ips.astype(int)
# #for  i in range(int(rough_peak_positions.size)):    
# if False:
#     plt.plot(f_sup,data_mean_sparse)        
#     plt.plot(f_sup[peaks],data_mean_sparse[peaks],'*')
#     for  i in range(peaks.size):                                
#         plt.plot(f_sup[l_ips[i]:r_ips[i]],data_mean_sparse[l_ips[i]:r_ips[i]]+1/2,'--')
        

# plt.plot(data_mean_sparse)
# plt.plot(peaks,data_mean_sparse[peaks],'*')

# #------------------------------------------

# peaks, properties = sci.find_peaks(smooth_data(data_mean_sparse,2),prominence=0.01) 

# plt.plot(data_mean_sparse)
# plt.plot(peaks,data_mean_sparse[peaks],'*')



# peak_prom=peak_prom/np.sqrt(np.var(peak_prom))
# #peak_width=properties['widths']
# #peak_width=peak_width/np.sqrt(np.var(peak_width))
# #peak_width=properties['widths']
# #
# peak_width=properties['right_bases']-properties['left_bases']
# peak_width=peak_width/np.sqrt(np.var(peak_width))
# peak_high=(data_mean_sparse[peaks])/np.sqrt(np.var(data_mean_sparse[peaks]))
# # ANY CHOICE OF THE METRIC HERE!?

# metric=10*peak_high+5*peak_prom+peak_width
metric=10*peak_high + 5*prominences

idx_peaks=np.flip(np.argsort(metric))
idx=peaks[idx_peaks]
rough_peak_positions = idx

#for  i in range(int(rough_peak_positions.size)):    
if False:
    for  i in range(5):        
        p=idx_peaks[i]
        l_ips=int(properties['left_bases'][p])
        r_ips=int(properties['right_bases'][p])
        
        plt.plot(f_vec,data_mean_sparse)
        
        plt.plot(f_vec[peaks[p]],data_mean_sparse[peaks][p],'*')
        plt.plot(f_vec[l_ips:r_ips],data_mean_sparse[l_ips:r_ips]+5,'--')
    
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

plot_f=1
plot_t=0
check_sup_fig=0

#for  i in range(5):        
 #   for  i in range(int(rough_peak_positions.size)):
     
for  i in range(int(rough_peak_positions.size)):
        print('Step: ',i,' -peak position: ',rough_peak_positions[i])

        p=idx_peaks[i]
        
        [l_win, r_win] = find_peak_sym_supp(peaks[p],int(l_ips[p]),int(r_ips[p]), 2, 3, data_mean_sparse)
        
        #[l_win, r_win],[l_ips,r_ips]
        l_ips = l_ips.astype(int)
        r_ips = r_ips.astype(int)
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
               
        #result = minimize(pos_mse_loss, beta_init_lor, args=(x_data,y_data,gen_lor_amp), method='Nelder-Mead', tol=1e-8)
        result = minimize(mse_loss, beta_init_lor, args=(x_data,y_data,gen_lor_amp), method='Nelder-Mead', tol=1e-12)
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
        # x_data = f_sup[l_win:r_win]    
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
            
    plt.plot(f_sup[peaks[p]],np.sum(comp_beta_cos[i,idx_s_l]),color=c,marker='*')            
    plt.plot(f_sup[peaks[p]],np.sum(comp_beta_cos[i,idx_s_r]),color=c,marker='*')            
    plt.plot(f_sup[peaks[p]],comp_beta_cos[i,1],color=c,marker='o')
    plt.plot(f_sup[peaks[p]],sum(comp_beta_cos[i,idx_d_r]),color=c,marker='o')
    plt.plot([f_sup[peaks[p]],f_sup[peaks[p]]],[np.sum(comp_beta_cos[i,idx_d_l]),sum(comp_beta_cos[i,idx_d_r])],'--'+c)
    plt.plot([f_sup[peaks[p]],f_sup[peaks[p]]],[np.sum(comp_beta_cos[i,idx_s_l]),sum(comp_beta_cos[i,idx_s_r])],c)

    # add the peak number 
    #plt.annotate(str(i), ([f_sup[peaks[p]],sum(comp_beta_cos[i,idx_s_r])))
        
    plt.annotate(str(i), (f_sup[peaks[p]],sum(comp_beta_cos[i,idx_d_r])+20))
                                                  
                                                  
plt.plot(f_sup,data_mean_sparse*100)
#plt.plot(f_sup[peaks],100*data_mean_sparse[peaks],'*r')
             

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
    
    
# X, Y = np.meshgrid(f_sup, np.array(range(dim_s)))
# Z = data_sub
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.contour3D(X, Y, Z, 50, cmap='binary')
# Z1 = data_sparse_back[i].T
# ax.contour3D(X, Y, Z1, 50, cmap='binary')
    
    
    