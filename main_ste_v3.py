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
#------------------------------------------------------------------------------


res=1600
dim_s=100


#all spectrums
i_file= open('data/MG set 1/original_MG_solution.dat', "r")
temp = i_file.readlines()
temp=[i.strip('\n') for i in temp]
data_o=np.array([(row.split('\t')) for row in temp], dtype=np.float32)

f_sup=data_o[:,0]
data_o=data_o[:,1]

plt.plot(f_sup,data_o)

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
    plt.plot(f_sup,data_o,label='clean data')
    plt.plot(f_sup,np.mean(data1500,axis=(1,2))+0.0003,label='1500ppb')
    plt.plot(f_sup,np.mean(data15,axis=(1,2))+0.0006,label='15ppb')
    plt.plot(f_sup,np.mean(data1p5,axis=(1,2))+0.0009,label='1.500ppb')
    plt.legend()
    
#data1500[0,:]-data1p5[0,:] 
#data1p5[0,:] 

##-----------------------------------------------
# fit with lorenz to double check things 
##-----------------------------------------------

#----------------------------------------------------------------------------
#       FUNCTION LOADER
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
    
    while  Y_smooth[r_pos_temp+1]<=Y_smooth[r_pos_temp] and r_pos_temp<dim_s-1: 
        r_pos_temp=r_pos_temp+1
        
    return [l_pos_temp,r_pos_temp]


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

data_MG=data1500
mean_data_MG=np.mean(data1500.reshape(res,dim_s),axis=1)

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


min_spec=np.min(data_MG,axis=(1,2))    

data_sparse=data_MG.reshape(res,dim_s)-np.reshape(poly_min_hat,[res,1])
data_sparse=data_sparse-np.min(data_sparse)

   

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

metric=10*peak_high+15*prominences

idx_peaks=np.flip(np.argsort(metric))
idx=peaks[idx_peaks]

#############################33
##
##   TAKE ONLY THE STRONGEST 10 PEAKS
##
#############################33
rough_peak_positions = idx[1:10]


# plt.plot(f_sup[rough_peak_positions ],data_mean_sparse[rough_peak_positions ],'*r')
# plt.plot(f_sup[peaks],data_mean_sparse[peaks],'.')
# plt.plot(f_sup,data_mean_sparse)


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
        
        [l_win, r_win] = find_peak_sym_supp(peaks[p],int(l_ips[p]),int(r_ips[p]), 2, 3, data_mean_sparse)
        
        #[l_win, r_win],[l_ips,r_ips]
        
        if check_sup_fig:
            plt.plot(data_mean_sparse)
            plt.plot(peaks[p],data_mean_sparse[peaks[p]],'*')
            plt.plot(range(l_ips[p],r_ips[p]),data_mean_sparse[l_ips[p]:r_ips[p]]+2,'--')            
            plt.plot(range(l_win,r_win),data_mean_sparse[l_win:r_win]+4,'--')            
                    
        #[l_win, r_win]=[l_ips,r_ips]
        x_data = f_sup[l_win:r_win]
        y_data = np.mean(data_sparse_back[i,l_win:r_win,:],axis=1)
        
        # this is my guess for a good strategy
        #bias_f = min(y_data)/2
        # random choice of bias
        bias_f = np.mean(np.sort(y_data)[0:int(y_data.size/5)])
        y_data = y_data -bias_f 
        
        beta_init_lor=init_lor(x_data,y_data)
               
        #result = minimize(pos_mse_loss, beta_init_lor, args=(x_data,y_data,gen_lor_amp), method='Nelder-Mead', tol=1e-8)
        result = minimize(mse_loss, beta_init_lor, args=(x_data,y_data,gen_lor_amp), method='Nelder-Mead', tol=1e-12)
        result = minimize(mse_loss, beta_init_lor, args=(x_data,y_data,gauss_amp), method='Nelder-Mead', tol=1e-12)
        beta_hat_lor=result.x    
        
        if plot_f:
            plt.figure('peak number '+str(int(i))+' frequency')
            plt.plot(x_data,y_data)            
            plt.plot(x_data[peaks[p]-l_win],y_data[peaks[p]-l_win],'*')                        
            plt.plot(x_data,gen_lor_amp(x_data,beta_init_lor),'--')
            plt.plot(x_data,gen_lor_amp(x_data,beta_hat_lor),'-*')
    
    
    
    
# CORRELATE SPATIALLY

#----------------------------------------------------------------------------

    # plt.plot(f_sup,np.mean(data1500,axis=(1,2))+0.0003,label='1500ppb')
    # plt.plot(f_sup,np.mean(data15,axis=(1,2))+0.0006,label='15ppb')
    # plt.plot(f_sup,np.mean(data1p5,axis=(1,2))+0.0009,label='1.500ppb')

# plot_on=1
# # plot some mean data

# data_mean_f=np.mean(data,axis=0)
# data_mean_s=np.mean(data,axis=(1,2))


# if plot_on:
#     plt.plot(data_mean_s)
#     plt.pcolormesh(data_mean_f)

# # remove the lower bound by:
# #    
# #   -finding the min
# #   -windowing
# #   -interpolating
 
# data_min_s=np.array([np.min(data[f,:,:])  for f in range(1600)]) 

# #   plt.plot(f_sup,data_mean_s)
# #   plt.plot(f_sup,data_min_s)

# # smooth the min


# # check NMF 

# def  quick_NMF(data,k=3):

#     model = NMF(n_components=k, init='random', random_state=0)
#     W = model.fit_transform(data)
#     H = model.components_
    
#     plot_NMF=1
    
#     if plot_NMF:
#         plt.figure("Model")
#         plt.plot(W)
#         plt.show()
#         plt.figure("Component")
#         plt.plot(np.transpose(H))
#         plt.show()

#     return W,H


# quick_NMF(data1500.reshape(res,dim_s),3)

# quick_NMF(data15.reshape(res,dim_s),2)

# quick_NMF(data1p5.reshape(res,dim_s),3)


# # remove the bias 

# INF=10**4

# def one_side_MSE (y_pred,y_true):        
#     if  y_pred-y_true>0:
#         loss=INF*(y_pred-y_true)**2
#     else:
#         loss=(y_pred-y_true)**2            
#     return(loss)


# def obj_one_side_MSE(beta, X, Y):
#     p=np.poly1d(beta)
#     error = sum([one_side_MSE(p(X[i]), Y[i]) for i in range(X.size)])/X.size
#     return(error)

# def quick_remove_bias(data):    
#     # find the minimum over the spacial dimension
#     min_spec=np.min(data,axis=(1,2))    
        
#     mean = np.mean(min_spec)
#     min_spec -= np.mean(min_spec)
#     autocorr_f = np.correlate(min_spec, min_spec, mode='full')
#     mid=int(np.where(autocorr_f==max(autocorr_f ))[0])
#     temp = autocorr_f[mid:]/autocorr_f[mid]    
#     # plt.plot(temp)
   
#     min_spec=np.min(data,axis=(1,2))          
#     beta_init=np.polyfit(f_sup,min_spec,3)
#     poly_min=np.poly1d(np.polyfit(f_sup,min_spec,3))(f_sup)
#     beta_init[3]=beta_init[3]+min(min_spec-poly_min)

#     result = minimize(obj_one_side_MSE, beta_init, args=(f_sup,min_spec), method='Nelder-Mead', tol=1e-9)   

#     beta_hat = result.x                 
#     beta_hat -beta_init
#     poly_min_hat=np.poly1d(beta_hat)(f_sup)
    
#     poly_min_hat=poly_min_hat+min(min_spec-poly_min_hat)
    
#     plt.plot(poly_min_hat)
#     plt.plot(min_spec)
    
#     return poly_min_hat
    

# def smooth_data(data,win):
#     return np.array([np.mean(data[int(np.max([j-win,0])):int(np.min([j+win,data.size]))]) for j in  range(data.size)])

# #------------------------------------------------------------------------------

# bias1500=quick_remove_bias(data1500)

# plt.figure('Data 1500ppb')
# plt.plot(f_sup,np.mean(data1500,axis=(1,2)),label='spatial mean')
# plt.plot(f_sup,np.min(data1500,axis=(1,2)),label='spatial minimum')
# plt.plot(f_sup,bias1500,label='bias spatial minimum')
# plt.legend()


# bias15=quick_remove_bias(data15)
# plt.figure('Data 15ppb')
# plt.plot(f_sup,np.mean(data15,axis=(1,2)),label='spatial mean')
# plt.plot(f_sup,np.min(data15,axis=(1,2)),label='spatial minimum')
# plt.plot(f_sup,bias15,label='bias spatial minimum')
# plt.legend()


# bias1p5=quick_remove_bias(data1p5)
# plt.figure('Data 1.5ppb')
# plt.plot(f_sup,np.mean(data1p5,axis=(1,2)),label='spatial mean')
# plt.plot(f_sup,np.min(data1p5,axis=(1,2)),label='spatial minimum')
# plt.plot(f_sup,bias1p5,label='bias spatial minimum')
# plt.legend()

# #------------------------------------------------------------------------------
# #interpolate the original spectrum with lorenzian


# beta_init=np.polyfit(f_sup,data_o,3)
# poly_min=np.poly1d(np.polyfit(f_sup,min_spec,3))(f_sup)
# beta_init[3]=beta_init[3]+min(min_spec-poly_min)

# result = minimize(obj_one_side_MSE, beta_init, args=(f_sup,data_o), method='Nelder-Mead', tol=1e-9)   

# beta_hat = result.x                 
# beta_hat -beta_init
# poly_min_hat=np.poly1d(beta_hat)(f_sup)
    
# poly_min_hat=poly_min_hat+min(data_o-poly_min_hat)
    
# data_o_nb=data_o-poly_min_hat

# plt.plot(f_sup,data_o)
# plt.plot(f_sup,poly_min_hat)
# plt.plot(f_sup,data_o_nb)


# mean = np.mean(data_o_nb)
# data_o_nb-= np.mean(data_o_nb)
# autocorr_f = np.correlate(data_o_nb, data_o_nb, mode='full')
# mid=int(np.where(autocorr_f==max(autocorr_f ))[0])
# temp = autocorr_f[mid:]/autocorr_f[mid]    
# win_t=np.where(temp>0.9)[0][-1]

# from lmfit.models import LorentzianModel, QuadraticModel

# mod = LorentzianModel()




# data_temp=data_o-poly_min_hat
# data_temp_mean=smooth_data(data_o_nb,win_t)

# #plt.plot(data_temp)
# #peaks, properties = sci.find_peaks(data_temp_mean,prominence=3,width=2)
# peaks, properties = sci.find_peaks(data_temp_mean,width=win_t)

# #idx=peaks[np.flip(np.argsort(data_temp[peaks]))]

# idx=np.flip(np.argsort(data_temp[peaks],kind='quicksort'))
# #idx=peaks[idx[0:7]]
# idx=peaks[idx[0:15]]

# plt.plot(f_sup[peaks],data_temp[peaks],'*')
# plt.plot(f_sup[idx], data_temp[idx],'.')
# plt.plot(f_sup, data_temp)


# rough_peak_positions = f_sup[idx]

# def add_peak(prefix, center, amplitude=0.1, sigma=0.1):
#     peak = LorentzianModel(prefix=prefix)
#     pars = peak.make_params()
#     pars[prefix + 'center'].set(center)
#     pars[prefix + 'amplitude'].set(amplitude)
#     pars[prefix + 'sigma'].set(sigma, min=0)
#     return peak, pars

# model = QuadraticModel(prefix='bkg_')
# params = model.make_params(a=0, b=0, c=0)

# for i, cen in enumerate(rough_peak_positions):
#     peak, pars = add_peak('lz%d_' % (i+1), cen)
#     model = model + peak
#     params.update(pars)
    
# init = model.eval(params, x=f_sup)
# result = model.fit(data_temp, params, x=f_sup)
# comps = result.eval_components()

# plt.plot(f_sup, data_temp, label='data')
# plt.plot(f_sup, result.best_fit, label='best fit')
# for name, comp in comps.items():
#     plt.plot(f_sup, comp, '--', label=name)

# #

# plt.legend(loc='upper right')
# plt.show()

# ##############################################################################
# ############################################################################################################################################################
# ##############################################################################
# ##############################################################################
# ##############################################################################











#     # window is when we go below 0.9
# win_t=np.where(temp>0.9)[0][-1]
# min_mean=smooth_data(min_spec,win_t)
    
# peaks, properties = sci.find_peaks(min_mean,width=2*win_t)
        
#     #peaks = sci.find_peaks_cwt(min_mean, np.arange(1,10))
    
# plt.plot(min_mean)
# plt.plot(peaks,min_mean[peaks],'o')
    
#     # find when we are under 0.5 correlation   
# win_t=np.where(temp>0.5)[0][-1]

#  #------------------------------------------------------------------------------
 
 
# pp=np.correlate(min_spec-np.mean(min_spec),min_spec-np.mean(min_spec), mode='same')/(np.var(min_spec))
# #    pp=pp[int(min_spec.size/2)]
     
    
# plt.plot(pp)
     
     
# min_corr=np.correlate(min_spec)
#     # set the window to the point at which the correlation fall under 0.5
# avg_corr= diag(min_corr,k)
# M=[np.mean(diag(min_corr,i)) for i in  range(min_spec.size)]
    
# peaks, properties = sci.find_peaks(data_min_s,prominence=5,width=5)


# #return bias_spect

# data=data15

# data=data15[:,].reshape(res,dim_s)


# win=10

# data_min_s=np.array([np.mean(data_min_s[int(np.max([j-win,0])):int(np.min([j+win,1600]))]) for j in  range(1600)])

# peaks, properties = sci.find_peaks(data_min_s,prominence=5,width=5)
# #                                   #,width=10,wlen=5)

# data_peak = data_min_s[peaks]
# data_peak_width=np.array([np.mean(data_min_s[int(properties["left_ips"][j]):int(properties["right_ips"][j])],axis=0) for j in  range(peaks.shape[0])] )
# data_peak_env=np.array([np.min(data_min_s[int(properties["left_ips"][j]):int(properties["right_ips"][j])],axis=0) for j in  range(peaks.shape[0])] )

# win=500    

# min_wo_peaks=data_min_s.copy()

# for p in range(peaks.shape[0]):
#     # find the alpha & beta coefficient between the edges
        
#     l_pos = int(np.floor(properties["left_ips"][p]))
#     r_pos = int(np.ceil(properties["right_ips"][p]))
    
#     plt.plot(l_pos,min_wo_peaks[l_pos],'*')
#     plt.plot(r_pos,min_wo_peaks[r_pos],'o')
    
#     win = range(l_pos,r_pos)
#     win_len=r_pos-l_pos
    
    
#     #l_max=l_pos-1000*win_len
#     #r_max=r_pos+1000*win_len
    
#     #min_win=min(min_wo_peaks[win]).copy()
#     # extend the window to make sure that the spectrum is decreasing arond the window
    
    
#     # while  (min_wo_peaks[l_pos-1]>=min_win) and (l_pos>l_max):
#     #     l_pos=l_pos-1
    
#     # while  (min_wo_peaks[r_pos+1]>=min_win) and (r_pos<r_max):
#     #     r_pos=r_pos+1
    
    
#     while  (min_wo_peaks[l_pos-1]<=min_wo_peaks[l_pos]):
#         l_pos=l_pos-1
    
#     while  (min_wo_peaks[r_pos+1]<=min_wo_peaks[r_pos]):
#         r_pos=r_pos+1
    
#     min_wo_peaks[l_pos:r_pos]=np.min([min_wo_peaks[l_pos],min_wo_peaks[r_pos]])
    
    
# poly_min=np.poly1d(np.polyfit(f_sup,min_wo_peaks,3))(f_sup)

# poly_min_pos=poly_min+min(min_wo_peaks-poly_min)

# peak_fig=1


# #----------------------------------------------------------------------------
# #
# # NON NEGATIVE ERROR POLY FIT OF DEGREE 3
# #
# #----------------------------------------------------------------------------

# INF=10**4

# def loss_function (y_pred,y_true):        
#     if  y_pred-y_true>0:
#         loss=INF*(y_pred-y_true)**2
#     else:
#         loss=(y_pred-y_true)**2            
#     return(loss)


# def objective_function(beta, X, Y):
#     p=np.poly1d(beta)
#     error = sum([loss_function(p(X[i]), Y[i]) for i in range(X.size)])/X.size
#     return(error)

# #
# # p=np.poly1d(beta_init)
# # sum([loss_function(p(f_sup[i]), min_wo_peaks[i]) for i in range(f_sup.size)])/f_sup.size
# #

# beta_init=np.polyfit(f_sup,min_wo_peaks,3)
# beta_init[3]=beta_init[3]+min(min_wo_peaks-poly_min)


# #result = minimize(objective_function, beta_init, args=(f_sup,data_min_s), method='BFGS', options={'gtol': 1e-8, 'disp': True})
# result = minimize(objective_function, beta_init, args=(f_sup,data_min_s), method='Nelder-Mead', tol=1e-9)

# result = minimize(objective_function, beta_init, args=(f_sup,data_min_s), method='Powell', tol=1e-9)

# beta_hat = result.x                 
# beta_hat -beta_init
# poly_min_hat=np.poly1d(beta_hat)(f_sup)

# # NOT WORKING : ask Anvar to do this

# objective_function(beta_init,f_sup,data_min_s)
# objective_function(beta_hat,f_sup,data_min_s)
# objective_function([0, 0, 0, 0],f_sup,data_min_s)


# if peak_fig:
#     x=data_min_s
#     plt.plot(f_sup,data_min_s)
#     plt.plot(f_sup,min_wo_peaks)
#     plt.plot(f_sup,poly_min_pos)
#     plt.plot(f_sup,poly_min_hat)
#     plt.plot(f_sup[peaks], x[peaks], "o")    
#     #plt.plot(f_sup,poly_min)
#     #plt.vlines(x=peaks, ymin=x[peaks] - properties["prominences"],ymax = x[peaks], color = "c1")
#     #plt.hlines(y=properties["width_heights"], xmin=properties["left_ips"],xmax=properties["right_ips"], color = "c1")    
#     plt.show()




# #------------------------------------------------------------------------------
# # CHECK NMF first 
# #------------------------------------------------------------------------------

# from sklearn.decomposition import NMF

# model = NMF(n_components=3, init='random', random_state=0)
# W = model.fit_transform(data.reshape([1600,100]))
# H = model.components_

# plot_NMF=1

# if plot_NMF:
#     plt.figure("Model 3 componenets")
#     plt.plot(W)
#     plt.show()
#     plt.figure("Component 3 componenets")
#     plt.plot(np.transpose(H))
#     plt.show()
    


# #------------------------------------------------------------------------------
# # Cauchy fitting of the peaks
# #------------------------------------------------------------------------------

# import scipy.optimize.curve_fit as curve_fit

# # pipeline
# data_temp=data[:,1,1]
# data_temp=data_temp-poly_min_hat
# data_temp_mean=smooth_data(data_temp,5)


# def lorentzian( f, c, A, gam ):
#     return A / (gam*pi) * 1/ ( 1 + ( f - c)**2)


# def gaussian( f, c,  A, sgs ):
#     return A / (sqrt(2*pi*sgs))*math.exp(- 1/(2*sgs) (f-c)**2)


# #order the peaks by magnitude and fit each of them in an interval around the peak

# peaks, properties = sci.find_peaks(data_temp_mean,prominence=10,width=10)
# idx_ord=peaks[np.flip(np.argsort(data_temp[peaks]))]
# rough_peak_positions = f_sup[idx_ord]

# num_fit_peaks=10


# for i in range(num_fit_peaks):
#     # fit with Gaussian or Lourenz: just pick the best fit    
            
#     l_pos = int(np.floor(properties["left_ips"][idx_ord[i]]))
#     r_pos = int(np.ceil(properties["right_ips"][idx_ord[i]]))
    
#     plt.plot(l_pos,min_wo_peaks[l_pos],'*')
#     plt.plot(r_pos,min_wo_peaks[r_pos],'o')
    
#     c_temp=rough_peak_positions[idx_ord[i]]
    
#     win = range(l_pos,r_pos)
#     win_len=r_pos-l_pos
    
#     popt, pcov = curve_fit(gaussian, xdata, ydata, bounds=(0, [3., 1., 0.5]))    
    

# # pick a spectrum
# # remove the poly minimum
# # find peaks

# from lmfit.models import LorentzianModel, QuadraticModel

# mod = LorentzianModel()

# data_temp=data[:,1,1]
# data_temp=data_temp-poly_min_hat
# data_temp_mean=smooth_data(data_temp,5)

# #peaks, properties = sci.find_peaks(data_temp_mean,prominence=3,width=2)
# peaks, properties = sci.find_peaks(data_temp_mean,prominence=10,width=10)
# idx=peaks[np.flip(np.argsort(data_temp[peaks]))]

# rough_peak_positions = f_sup[idx[1:10]]

# plt.plot(f_sup,data_temp_mean)
# plt.plot(f_sup,peaks,data_temp_mean[peaks],'o')
# plt.plot(f_sup,data_temp,'--')

# def add_peak(prefix, center, amplitude=1, sigma=1):
#     peak = LorentzianModel(prefix=prefix)
#     pars = peak.make_params()
#     pars[prefix + 'center'].set(center)
#     pars[prefix + 'amplitude'].set(amplitude)
#     pars[prefix + 'sigma'].set(sigma, min=0)
#     return peak, pars

# model = QuadraticModel(prefix='bkg_')
# params = model.make_params(a=0, b=0, c=0)

# for i, cen in enumerate(rough_peak_positions):
#     peak, pars = add_peak('lz%d_' % (i+1), cen)
#     model = model + peak
#     params.update(pars)
    
# init = model.eval(params, x=f_sup)
# result = model.fit(data_temp, params, x=f_sup)
# comps = result.eval_components()

# plt.plot(f_sup, data_temp, label='data')
# plt.plot(f_sup, result.best_fit, label='best fit')
# for name, comp in comps.items():
#     plt.plot(f_sup, comp, '--', label=name)

# #plt.plot(rough_peak_positions,data_temp[idx[1:5]],'o')
# plt.legend(loc='upper right')
# plt.show()

# # s = 'The value of x is ' + repr(i) + ', and y is ' + repr(cen) + '...'
# # print(s)

# #--------------------------------------------------
# for i, cen in enumerate(rough_peak_positions):
#     peak, pars = add_peak('lz%d_' % (i+1), cen)
#     model = model + peak
#     params.update(pars)

# init = model.eval(params, x=f_sup)
# result = model.fit(data_temp, params, x=f_sup)
# comps = result.eval_components()

# #print(result.fit_report(min_correl=0.5))

# plt.plot(f_sup, data_temp, label='data')
# plt.plot(f_sup, result.best_fit, label='best fit')
# for name, comp in comps.items():
#     plt.plot(f_sup, comp, '--', label=name)
# plt.legend(loc='upper right')
# plt.show()

# pos=np.where(peaks==idx)

# # order peaks by magnitude

# # plt.plot(f_sup,data_temp)
# # plt.plot(f_sup,data_temp_mean)
# # plt.plot(f_sup[peaks], data_temp_mean[peaks], "x")    

# # pars = mod.guess(y, x=x)
# # out = mod.fit(y, pars, x=x)

# # plt.plot(x, y, 'b')
# # plt.plot(x, out.init_fit, 'k--', label='initial fit')
# # plt.plot(x, out.best_fit, 'r-', label='best fit')
# # plt.legend(loc='best')
# # plt.show()


# #------------------------------------------------------------------------------
# xdat = f_sup
# ydat =data_temp

# peaks, properties = sci.find_peaks(data_temp_mean,prominence=10,width=10) 
# idx=peaks[np.flip(np.argsort(data_temp[peaks]))]
# rough_peak_positions = f_sup[idx]

# def add_peak(prefix, center, amplitude=5, sigma=5):
#     peak = LorentzianModel(prefix=prefix)
#     pars = peak.make_params()
#     pars[prefix + 'center'].set(center)
#     pars[prefix + 'amplitude'].set(amplitude)
#     pars[prefix + 'sigma'].set(sigma, min=0)
#     return peak, pars

# model = QuadraticModel(prefix='bkg_')
# params = model.make_params(a=0, b=0, c=0)

# #rough_peak_positions = (0.61, 0.76, 0.85, 0.99, 1.10, 1.40, 1.54, 1.7)
# for i, cen in enumerate(rough_peak_positions):
#     peak, pars = add_peak('lz%d_' % (i+1), cen)
#     model = model + peak
#     params.update(pars)

# init = model.eval(params, x=xdat)
# result = model.fit(ydat, params, x=xdat)
# comps = result.eval_components()

# print(result.fit_report(min_correl=0.5))

# plt.plot(xdat, ydat, label='data')
# plt.plot(xdat, result.best_fit, label='best fit')
# for name, comp in comps.items():
#     plt.plot(xdat, comp, '--', label=name)
# plt.legend(loc='upper right')
# plt.show()


# # result = minimize(objective_function, beta_init, args=(X,Y),
# #                   method='BFGS', options={'maxiter': 500})


# objective_function(beta_init,f_sup,data_min_s)

# #------------------------------------------------------------------------------

