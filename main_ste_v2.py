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
i_file= open('original_MG_solution.dat', "r")
temp = i_file.readlines()
temp=[i.strip('\n') for i in temp]
data_o=np.array([(row.split('\t')) for row in temp], dtype=np.float32)

f_vec=data_o[:,0]
data_o=data_o[:,1]

plt.plot(f_vec,data_o)

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

init_plot=1

if init_plot:
    plt.plot(f_vec,data_o,label='clean data')
    plt.plot(f_vec,np.mean(data1500,axis=(1,2))+0.0003,label='1500ppb')
    plt.plot(f_vec,np.mean(data15,axis=(1,2))+0.0006,label='15ppb')
    plt.plot(f_vec,np.mean(data1p5,axis=(1,2))+0.0009,label='1.500ppb')
    plt.legend()
    
#data1500[0,:]-data1p5[0,:] 
#data1p5[0,:] 


plot_on=1
# plot some mean data

data_mean_f=np.mean(data,axis=0)
data_mean_s=np.mean(data,axis=(1,2))


if plot_on:
    plt.plot(data_mean_s)
    plt.pcolormesh(data_mean_f)

# remove the lower bound by:
#    
#   -finding the min
#   -windowing
#   -interpolating
 
data_min_s=np.array([np.min(data[f,:,:])  for f in range(1600)]) 

#   plt.plot(f_vec,data_mean_s)
#   plt.plot(f_vec,data_min_s)

# smooth the min


# check NMF 

def  quick_NMF(data,k=3):

    model = NMF(n_components=k, init='random', random_state=0)
    W = model.fit_transform(data)
    H = model.components_
    
    plot_NMF=1
    
    if plot_NMF:
        plt.figure("Model")
        plt.plot(W)
        plt.show()
        plt.figure("Component")
        plt.plot(np.transpose(H))
        plt.show()

    return W,H


quick_NMF(data1500.reshape(res,dim_s),3)

quick_NMF(data15.reshape(res,dim_s),2)

quick_NMF(data1p5.reshape(res,dim_s),3)


# remove the bias 

INF=10**4

def one_side_MSE(y_pred,y_true):        
    return INF*(y_pred-y_true)**2 if y_pred-y_true>0 else (y_pred-y_true)**2


def obj_one_side_MSE(beta, X, Y):
    p=np.poly1d(beta)
    return sum(one_side_MSE(p(X[i]), Y[i]) for i in range(X.size)) / X.size

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

bias1500=quick_remove_bias(data1500)

plt.figure('Data 1500ppb')
plt.plot(f_vec,np.mean(data1500,axis=(1,2)),label='spatial mean')
plt.plot(f_vec,np.min(data1500,axis=(1,2)),label='spatial minimum')
plt.plot(f_vec,bias1500,label='bias spatial minimum')
plt.legend()


bias15=quick_remove_bias(data15)
plt.figure('Data 15ppb')
plt.plot(f_vec,np.mean(data15,axis=(1,2)),label='spatial mean')
plt.plot(f_vec,np.min(data15,axis=(1,2)),label='spatial minimum')
plt.plot(f_vec,bias15,label='bias spatial minimum')
plt.legend()


bias1p5=quick_remove_bias(data1p5)
plt.figure('Data 1.5ppb')
plt.plot(f_vec,np.mean(data1p5,axis=(1,2)),label='spatial mean')
plt.plot(f_vec,np.min(data1p5,axis=(1,2)),label='spatial minimum')
plt.plot(f_vec,bias1p5,label='bias spatial minimum')
plt.legend()

#------------------------------------------------------------------------------
#interpolate the original spectrum with lorenzian


beta_init=np.polyfit(f_vec,data_o,3)
poly_min=np.poly1d(np.polyfit(f_vec,min_spec,3))(f_vec)
beta_init[3]=beta_init[3]+min(min_spec-poly_min)

result = minimize(obj_one_side_MSE, beta_init, args=(f_vec,data_o), method='Nelder-Mead', tol=1e-9)   

beta_hat = result.x
beta_hat -beta_init
poly_min_hat=np.poly1d(beta_hat)(f_vec)

poly_min_hat=poly_min_hat+min(data_o-poly_min_hat)

data_o_nb=data_o-poly_min_hat

plt.plot(f_vec,data_o)
plt.plot(f_vec,poly_min_hat)
plt.plot(f_vec,data_o_nb)


mean = np.mean(data_o_nb)
data_o_nb-= np.mean(data_o_nb)
autocorr_f = np.correlate(data_o_nb, data_o_nb, mode='full')
mid=int(np.where(autocorr_f==max(autocorr_f ))[0])
temp = autocorr_f[mid:]/autocorr_f[mid]
win_t=np.where(temp>0.9)[0][-1]

from lmfit.models import LorentzianModel, QuadraticModel

mod = LorentzianModel()


data_temp=data_o-poly_min_hat
data_temp_mean=smooth_data(data_o_nb,win_t)

#plt.plot(data_temp)
#peaks, properties = sci.find_peaks(data_temp_mean,prominence=3,width=2)
peaks, properties = sci.find_peaks(data_temp_mean,width=win_t)

#idx=peaks[np.flip(np.argsort(data_temp[peaks]))]

idx=np.flip(np.argsort(data_temp[peaks],kind='quicksort'))
#idx=peaks[idx[0:7]]
idx = peaks[idx[:15]]

plt.plot(f_vec[peaks],data_temp[peaks],'*')
plt.plot(f_vec[idx], data_temp[idx],'.')
plt.plot(f_vec, data_temp)


rough_peak_positions = f_vec[idx]

def add_peak(prefix, center, amplitude=0.1, sigma=0.1):
    peak = LorentzianModel(prefix=prefix)
    pars = peak.make_params()
    pars[f'{prefix}center'].set(center)
    pars[f'{prefix}amplitude'].set(amplitude)
    pars[f'{prefix}sigma'].set(sigma, min=0)
    return peak, pars

model = QuadraticModel(prefix='bkg_')
params = model.make_params(a=0, b=0, c=0)

for i, cen in enumerate(rough_peak_positions):
    peak, pars = add_peak('lz%d_' % (i+1), cen)
    model = model + peak
    params.update(pars)

init = model.eval(params, x=f_vec)
result = model.fit(data_temp, params, x=f_vec)
comps = result.eval_components()

plt.plot(f_vec, data_temp, label='data')
plt.plot(f_vec, result.best_fit, label='best fit')
for name, comp in comps.items():
    plt.plot(f_vec, comp, '--', label=name)

#

plt.legend(loc='upper right')
plt.show()

win_t=np.where(temp>0.9)[0][-1]
min_mean=smooth_data(min_spec,win_t)

peaks, properties = sci.find_peaks(min_mean,width=2*win_t)

plt.plot(min_mean)
plt.plot(peaks,min_mean[peaks],'o')

win_t=np.where(temp>0.5)[0][-1]

pp=np.correlate(min_spec-np.mean(min_spec),min_spec-np.mean(min_spec), mode='same')/(np.var(min_spec))
#    pp=pp[int(min_spec.size/2)]


plt.plot(pp)


min_corr=np.correlate(min_spec)
avg_corr= diag(min_corr,k)
M=[np.mean(diag(min_corr,i)) for i in  range(min_spec.size)]

peaks, properties = sci.find_peaks(data_min_s,prominence=5,width=5)


#return bias_spect

data=data15

data=data15[:,].reshape(res,dim_s)


win=10

data_min_s=np.array([np.mean(data_min_s[int(np.max([j-win,0])):int(np.min([j+win,1600]))]) for j in  range(1600)])

peaks, properties = sci.find_peaks(data_min_s,prominence=5,width=5)
#                                   #,width=10,wlen=5)

data_peak = data_min_s[peaks]
data_peak_width=np.array([np.mean(data_min_s[int(properties["left_ips"][j]):int(properties["right_ips"][j])],axis=0) for j in  range(peaks.shape[0])] )
data_peak_env=np.array([np.min(data_min_s[int(properties["left_ips"][j]):int(properties["right_ips"][j])],axis=0) for j in  range(peaks.shape[0])] )

win=500    

min_wo_peaks=data_min_s.copy()

for p in range(peaks.shape[0]):
    # find the alpha & beta coefficient between the edges

    l_pos = int(np.floor(properties["left_ips"][p]))
    r_pos = int(np.ceil(properties["right_ips"][p]))

    plt.plot(l_pos,min_wo_peaks[l_pos],'*')
    plt.plot(r_pos,min_wo_peaks[r_pos],'o')

    win = range(l_pos,r_pos)
    win_len=r_pos-l_pos


    #l_max=l_pos-1000*win_len
    #r_max=r_pos+1000*win_len

    #min_win=min(min_wo_peaks[win]).copy()
    # extend the window to make sure that the spectrum is decreasing arond the window


    # while  (min_wo_peaks[l_pos-1]>=min_win) and (l_pos>l_max):
    #     l_pos=l_pos-1

    # while  (min_wo_peaks[r_pos+1]>=min_win) and (r_pos<r_max):
    #     r_pos=r_pos+1


    while (min_wo_peaks[l_pos-1]<=min_wo_peaks[l_pos]):
        l_pos -= 1

    while (min_wo_peaks[r_pos+1]<=min_wo_peaks[r_pos]):
        r_pos += 1

    min_wo_peaks[l_pos:r_pos]=np.min([min_wo_peaks[l_pos],min_wo_peaks[r_pos]])


poly_min=np.poly1d(np.polyfit(f_vec,min_wo_peaks,3))(f_vec)

poly_min_pos=poly_min+min(min_wo_peaks-poly_min)

peak_fig=1


#----------------------------------------------------------------------------
#
# NON NEGATIVE ERROR POLY FIT OF DEGREE 3
#
#----------------------------------------------------------------------------

INF=10**4

def loss_function(y_pred,y_true):        
    return INF*(y_pred-y_true)**2 if y_pred-y_true>0 else (y_pred-y_true)**2


def objective_function(beta, X, Y):
    p=np.poly1d(beta)
    return sum(loss_function(p(X[i]), Y[i]) for i in range(X.size)) / X.size

#
# p=np.poly1d(beta_init)
# sum([loss_function(p(f_vec[i]), min_wo_peaks[i]) for i in range(f_vec.size)])/f_vec.size
#

beta_init=np.polyfit(f_vec,min_wo_peaks,3)
beta_init[3]=beta_init[3]+min(min_wo_peaks-poly_min)


#result = minimize(objective_function, beta_init, args=(f_vec,data_min_s), method='BFGS', options={'gtol': 1e-8, 'disp': True})
result = minimize(objective_function, beta_init, args=(f_vec,data_min_s), method='Nelder-Mead', tol=1e-9)

result = minimize(objective_function, beta_init, args=(f_vec,data_min_s), method='Powell', tol=1e-9)

beta_hat = result.x                 
beta_hat -beta_init
poly_min_hat=np.poly1d(beta_hat)(f_vec)

# NOT WORKING : ask Anvar to do this

objective_function(beta_init,f_vec,data_min_s)
objective_function(beta_hat,f_vec,data_min_s)
objective_function([0, 0, 0, 0],f_vec,data_min_s)


if peak_fig:
    x=data_min_s
    plt.plot(f_vec,data_min_s)
    plt.plot(f_vec,min_wo_peaks)
    plt.plot(f_vec,poly_min_pos)
    plt.plot(f_vec,poly_min_hat)
    plt.plot(f_vec[peaks], x[peaks], "o")    
    #plt.plot(f_vec,poly_min)
    #plt.vlines(x=peaks, ymin=x[peaks] - properties["prominences"],ymax = x[peaks], color = "c1")
    #plt.hlines(y=properties["width_heights"], xmin=properties["left_ips"],xmax=properties["right_ips"], color = "c1")    
    plt.show()




#------------------------------------------------------------------------------
# CHECK NMF first 
#------------------------------------------------------------------------------

from sklearn.decomposition import NMF

model = NMF(n_components=3, init='random', random_state=0)
W = model.fit_transform(data.reshape([1600,100]))
H = model.components_

plot_NMF=1

if plot_NMF:
    plt.figure("Model 3 componenets")
    plt.plot(W)
    plt.show()
    plt.figure("Component 3 componenets")
    plt.plot(np.transpose(H))
    plt.show()
    


#------------------------------------------------------------------------------
# Cauchy fitting of the peaks
#------------------------------------------------------------------------------

import scipy.optimize.curve_fit as curve_fit

# pipeline
data_temp=data[:,1,1]
data_temp=data_temp-poly_min_hat
data_temp_mean=smooth_data(data_temp,5)


def lorentzian( f, c, A, gam ):
    return A / (gam*pi) * 1/ ( 1 + ( f - c)**2)


def gaussian( f, c,  A, sgs ):
    return A / (sqrt(2*pi*sgs))*math.exp(- 1/(2*sgs) (f-c)**2)


#order the peaks by magnitude and fit each of them in an interval around the peak

peaks, properties = sci.find_peaks(data_temp_mean,prominence=10,width=10)
idx_ord=peaks[np.flip(np.argsort(data_temp[peaks]))]
rough_peak_positions = f_vec[idx_ord]

num_fit_peaks=10


for i in range(num_fit_peaks):
    # fit with Gaussian or Lourenz: just pick the best fit    
            
    l_pos = int(np.floor(properties["left_ips"][idx_ord[i]]))
    r_pos = int(np.ceil(properties["right_ips"][idx_ord[i]]))
    
    plt.plot(l_pos,min_wo_peaks[l_pos],'*')
    plt.plot(r_pos,min_wo_peaks[r_pos],'o')
    
    c_temp=rough_peak_positions[idx_ord[i]]
    
    win = range(l_pos,r_pos)
    win_len=r_pos-l_pos
    
    popt, pcov = curve_fit(gaussian, xdata, ydata, bounds=(0, [3., 1., 0.5]))    
    

# pick a spectrum
# remove the poly minimum
# find peaks

from lmfit.models import LorentzianModel, QuadraticModel

mod = LorentzianModel()

data_temp=data[:,1,1]
data_temp=data_temp-poly_min_hat
data_temp_mean=smooth_data(data_temp,5)

#peaks, properties = sci.find_peaks(data_temp_mean,prominence=3,width=2)
peaks, properties = sci.find_peaks(data_temp_mean,prominence=10,width=10)
idx=peaks[np.flip(np.argsort(data_temp[peaks]))]

rough_peak_positions = f_vec[idx[1:10]]

plt.plot(f_vec,data_temp_mean)
plt.plot(f_vec,peaks,data_temp_mean[peaks],'o')
plt.plot(f_vec,data_temp,'--')

def add_peak(prefix, center, amplitude=1, sigma=1):
    peak = LorentzianModel(prefix=prefix)
    pars = peak.make_params()
    pars[f'{prefix}center'].set(center)
    pars[f'{prefix}amplitude'].set(amplitude)
    pars[f'{prefix}sigma'].set(sigma, min=0)
    return peak, pars

model = QuadraticModel(prefix='bkg_')
params = model.make_params(a=0, b=0, c=0)

for i, cen in enumerate(rough_peak_positions):
    peak, pars = add_peak('lz%d_' % (i+1), cen)
    model = model + peak
    params.update(pars)
    
init = model.eval(params, x=f_vec)
result = model.fit(data_temp, params, x=f_vec)
comps = result.eval_components()

plt.plot(f_vec, data_temp, label='data')
plt.plot(f_vec, result.best_fit, label='best fit')
for name, comp in comps.items():
    plt.plot(f_vec, comp, '--', label=name)

#plt.plot(rough_peak_positions,data_temp[idx[1:5]],'o')
plt.legend(loc='upper right')
plt.show()

# s = 'The value of x is ' + repr(i) + ', and y is ' + repr(cen) + '...'
# print(s)

#--------------------------------------------------
for i, cen in enumerate(rough_peak_positions):
    peak, pars = add_peak('lz%d_' % (i+1), cen)
    model = model + peak
    params.update(pars)

init = model.eval(params, x=f_vec)
result = model.fit(data_temp, params, x=f_vec)
comps = result.eval_components()

#print(result.fit_report(min_correl=0.5))

plt.plot(f_vec, data_temp, label='data')
plt.plot(f_vec, result.best_fit, label='best fit')
for name, comp in comps.items():
    plt.plot(f_vec, comp, '--', label=name)
plt.legend(loc='upper right')
plt.show()

pos=np.where(peaks==idx)

# order peaks by magnitude

# plt.plot(f_vec,data_temp)
# plt.plot(f_vec,data_temp_mean)
# plt.plot(f_vec[peaks], data_temp_mean[peaks], "x")    

# pars = mod.guess(y, x=x)
# out = mod.fit(y, pars, x=x)

# plt.plot(x, y, 'b')
# plt.plot(x, out.init_fit, 'k--', label='initial fit')
# plt.plot(x, out.best_fit, 'r-', label='best fit')
# plt.legend(loc='best')
# plt.show()


#------------------------------------------------------------------------------
xdat = f_vec
ydat =data_temp

peaks, properties = sci.find_peaks(data_temp_mean,prominence=10,width=10) 
idx=peaks[np.flip(np.argsort(data_temp[peaks]))]
rough_peak_positions = f_vec[idx]

def add_peak(prefix, center, amplitude=5, sigma=5):
    peak = LorentzianModel(prefix=prefix)
    pars = peak.make_params()
    pars[f'{prefix}center'].set(center)
    pars[f'{prefix}amplitude'].set(amplitude)
    pars[f'{prefix}sigma'].set(sigma, min=0)
    return peak, pars

model = QuadraticModel(prefix='bkg_')
params = model.make_params(a=0, b=0, c=0)

#rough_peak_positions = (0.61, 0.76, 0.85, 0.99, 1.10, 1.40, 1.54, 1.7)
for i, cen in enumerate(rough_peak_positions):
    peak, pars = add_peak('lz%d_' % (i+1), cen)
    model = model + peak
    params.update(pars)

init = model.eval(params, x=xdat)
result = model.fit(ydat, params, x=xdat)
comps = result.eval_components()

print(result.fit_report(min_correl=0.5))

plt.plot(xdat, ydat, label='data')
plt.plot(xdat, result.best_fit, label='best fit')
for name, comp in comps.items():
    plt.plot(xdat, comp, '--', label=name)
plt.legend(loc='upper right')
plt.show()


# result = minimize(objective_function, beta_init, args=(X,Y),
#                   method='BFGS', options={'maxiter': 500})


objective_function(beta_init,f_vec,data_min_s)

#------------------------------------------------------------------------------


# #reading files
# testfile_1_5_ppb = open('1.5ppb_35days_15ml-1_Exported.dat', "r")
# testfile_15_ppb = open('15ppb_4days_15ml-1_Exported.dat', "r")
# testfile_1500_ppb = open('1500ppb_47.5hr_1.5ml-1_Exported.dat', "r")
# testfile_original = open('Original_MG_solution-100um_HWP49_noND_9.2mW_20x(NA0.75)_Newton(S-S-H-C)_500ms_4avg_2bx_Exported.dat', "r")

# #obtaining frequency-amplitude pairs
# lines_1_5_ppb = testfile_1_5_ppb.readlines()
# lines_15_ppb = testfile_15_ppb.readlines()
# lines_1500_ppb = testfile_1500_ppb.readlines()
# lines_original = testfile_original.readlines()
# result_frequency_1_5 = []
# result_frequency_15 = []
# result_frequency_1500 = []
# result_frequency_original = []
# result_amplitude_1_5 = []
# result_amplitude_15 = []
# result_amplitude_1500 = []
# result_amplitude_original = []

# for (small, medium, big, original) in zip(lines_1_5_ppb, lines_15_ppb, lines_1500_ppb, lines_original):
#     small, medium, big, original = small.strip('\n'), medium.strip('\n'), big.strip('\n'), original.strip('\n')
#     small, medium, big, original = small.replace(",", "."), medium.replace(",", "."), big.replace(",", "."), original.replace(",", ".")
#     result_frequency_1_5.append(small.split("\t")[0])
#     result_frequency_15.append(medium.split("\t")[0])
#     result_frequency_1500.append(big.split("\t")[0])
#     result_frequency_original.append(original.split("\t")[0])
#     result_amplitude_1_5.append(small.split("\t")[1])
#     result_amplitude_15.append(medium.split("\t")[1])
#     result_amplitude_1500.append(big.split("\t")[1])
#     result_amplitude_original.append(original.split("\t")[1])


# new_list_small_x = []
# new_list_small_y = []
# new_list_medium_x = []
# new_list_medium_y = []
# new_list_big_x = []
# new_list_big_y = []
# new_list_original_x = []
# new_list_original_y = []
# for (item_s_x, item_m_x, item_b_x, item_or_x) in zip(result_frequency_1_5, result_frequency_15, result_frequency_1500, result_frequency_original):
#     new_list_small_x.append(float(item_s_x))
#     new_list_medium_x.append(float(item_m_x))
#     new_list_big_x.append(float(item_b_x))
#     new_list_original_x.append(float(item_or_x))

# for (item_s_y, item_m_y, item_b_y, item_or_y) in zip(result_amplitude_1_5, result_amplitude_15, result_amplitude_1500, result_amplitude_original):
#     new_list_small_y.append(float(item_s_y))
#     new_list_medium_y.append(float(item_m_y))
#     new_list_big_y.append(float(item_b_y))
#     new_list_original_y.append(float(item_or_y))

# #### getting envelopes
# # win = 10
# # j = 0
# # min_position = [0,]
# # max_position = [0,]
# # for (min_position[j + win], max_position[j + win]) in new_list_big_y:
# #     min_position.append(new_list_big_y)

# win = 150
# def window(seq, n=win):
#     "Returns a sliding window (of width n) over data from the iterable"
#     "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
#     it = iter(seq)
#     result = tuple(islice(it, n))
#     if len(result) == n:
#         yield result
#     for elem in it:
#         result = result[1:] + (elem,)
#         yield result

# list_windowed = []
# list_windowed = list(window(new_list_big_y))
# print(list_windowed)


# min_position = []
# max_position = []
# min_position_list = []
# for min_position in list_windowed:
#     min_position_list.append(min(min_position))
#     # max_position.append(max(list_windowed))

# print(min_position_list)
# ###########

# ###########
# array_big_list_y = np.array(new_list_big_y)
# peaks, properties = find_peaks(array_big_list_y, width = 10, prominence=1, distance=25)
# print(peaks)
# data_peak = array_big_list_y[peaks]
# data_peak_width=np.array([np.mean(array_big_list_y[int(properties['left_ips'][j]):int(properties["right_ips"][j])], axis=0) for j in  range(peaks.shape[0])])


# data_peak = array_big_list_y[peaks]
# data_peak_width=np.array([np.mean(array_big_list_y[int(properties["left_ips"][j]):int(properties["right_ips"][j])], axis=0) for j in  range(peaks.shape[0])] )


# # my_dark_peaks =  peaks[[4 ,5 ,10, 11]]

# dark_mean_fig=1

# if dark_mean_fig:
#     x=np.array(new_list_big_y)
#     plt.plot(x)
#     plt.plot(peaks, x[peaks], "x")
#     plt.vlines(x=peaks, ymin=x[peaks] - properties["prominences"],ymax = x[peaks], color = "c1")
#     plt.hlines(y=properties["width_heights"], xmin=properties["left_ips"],xmax=properties["right_ips"], color = "c1")
#     plt.show()
# ###########

# # print(max_position)
# # #####
# # # plot polynomial
# # # # plt.yscale('log')
# # xp = np.linspace(100, 2600, 1600 - win + 1)
# # plt.plot(new_list_big_x, new_list_big_y, 'r')
# # plt.plot(xp, min_position_list, 'g')
# # # plt.ylim(1,130)
# # #
# # # naming the x axis
# # plt.xlabel('Frequency')
# # # naming the y axis
# # plt.ylabel('Amplitude')
# #
# # # giving a title to my graph
# # plt.title('1500ppb with low envelope')
# #
# # # function to show the plot
# # plt.show()
# # # # plt.savefig('SRES_with_inter_1.png', bbox_inches='tight')
# # #
# # # # show roots
# # # print (p.roots)matplotlib.pyplot as plt
