from itertools import islice

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from scipy.signal import hilbert, find_peaks
import pylab as p
import scipy.signal as sci
import scipy.stats as stats
from scipy.optimize import minimize
from scipy.optimize import curve_fit
import math
from math import pi
import scipy.interpolate as si

# ---------------------------------------------------

from patsy import dmatrix
import statsmodels.api as sm
import statsmodels.formula.api as smf

# # Generating cubic spline with 3 knots at 25, 40 and 60
# transformed_x = dmatrix("bs(train, knots=(25,40,60), degree=3, include_intercept=False)", {"train": train_x},return_type='dataframe')

# # Fitting Generalised linear model on transformed dataset
# fit1 = sm.GLM(train_y, transformed_x).fit()

# # Generating cubic spline with 4 knots
# transformed_x2 = dmatrix("bs(train, knots=(25,40,50,65),degree =3, include_intercept=False)", {"train": train_x}, return_type='dataframe')

# # Fitting Generalised linear model on transformed dataset
# fit2 = sm.GLM(train_y, transformed_x2).fit()

# # Predictions on both splines
# pred1 = fit1.predict(dmatrix("bs(valid, knots=(25,40,60), include_intercept=False)", {"valid": valid_x}, return_type='dataframe'))
# pred2 = fit2.predict(dmatrix("bs(valid, knots=(25,40,50,65),degree =3, include_intercept=False)", {"valid": valid_x}, return_type='dataframe'))

# ----------------------------------------------------------

# import ste_model_spectrum.py

# from ste_model_spectrum_v4 import *
from ste_model_spectrum_v4 import *
from ste_model_spectrum_v5 import *

res = 964
dim_s = 100

# ------------------------------------------------------------------------------
import shelve

filename = 'shelve_save_data_analyte.out'

my_shelf = shelve.open(filename)
for key in my_shelf:
    globals()[key] = my_shelf[key]
my_shelf.close()

space_mean = np.mean(data[2], axis=1)
f_sup = f_sup[:-1]

fit_fun = gen_lor_amp
init_fit_fun = init_lor

slide_win = 100
num_th = 100

win_small = int(slide_win / 25)

# input
# fitting function florescence
# fitting function peak
# ranking metric

# procedure:

# start with poly interpolation
# choose the points where the linear interpolation crosses the  spectrum
# one parameter: number of threshold
# choose peaks below threshold
# fit and store

x_data = f_sup
y_data = space_mean

y_hat = np.ones(res) * np.min(y_data)

# eventually we want the proceduce to converge

num_th = 5

peak_pos = []
interpol_pos = []
interpol_mse = []
# interpol_pos.append([0,np.argmin(peak_est),res-1])
interpol_pos = interpol_pos + [0, np.argmin(y_data), res - 1]
interpol_mse = interpol_mse + [1000, 1000, 1000]
#

# random restarts?

nu = 0.01
min_peak_width = 4
peak_tol = 0.99

slide_win = int(res / num_th)

fitting_data = np.empty(num_th, dtype=object)

init_fit_fun = init_pseudo_voig
loss_peak = mse_loss
fit_fun = pseudo_voig

for j in range(num_th):

    # find the points that minimizes the variance of the data minus spline interpolation     
    # scan all points, who cares
    idx_left = list(set(range(res)) - set(peak_pos))

    while False:
        plt.plot(x_data, y_data)
        plt.plot(x_data[idx_left], y_data[idx_left], '*')

    interpol_pos = []
    interpol_pos = [0, np.argmin(y_data[idx_left]), res - 1]

    # for i in range(int(y_data.shape[0]/slide_win)):
    for i in range(2 * int(y_data.shape[0] / slide_win)):

        min_pos = 0

        y_hat = si.interp1d(x_data[interpol_pos], y_data[interpol_pos])(x_data)
        min_mse = (1 - nu) * np.dot((y_hat - y_data > 0), np.abs((y_hat - y_data))) + nu * np.var(y_hat - y_data)

        tmp_interpol_pos = list(
            set(range(int(i * (slide_win / 2)), min(int(i * (slide_win / 2) + (slide_win)), res))) - set(peak_pos))

        for k in range(len(tmp_interpol_pos)):

            tmp_pos = np.concatenate((interpol_pos, [tmp_interpol_pos[k]]))

            y_hat = si.interp1d(x_data[tmp_pos], y_data[tmp_pos])(x_data)

            # generalize to any loss        
            # tmp_mse=np.var(y_hat-y_data) 
            tmp_mse = (1 - nu) * np.dot((y_hat - y_data > 0), np.abs((y_hat - y_data))) + nu * np.var(y_hat - y_data)

            # update the minimum
            if tmp_mse < min_mse:
                min_pos = tmp_interpol_pos[k]
                min_mse = tmp_mse

        interpol_pos.append(min_pos)
        # interpol_pos.sort()
        interpol_mse.append(min_mse)

        unique_pos = np.array([int(interpol_pos.index(x)) for x in set(interpol_pos)])
        interpol_pos = list(np.array(interpol_pos)[unique_pos.astype(int)])
        interpol_mse = list(np.array(interpol_mse)[unique_pos.astype(int)])

        # sort points
        sort_pos = np.argsort(interpol_pos)
        interpol_pos = list(np.array(interpol_pos)[sort_pos.astype(int)])
        interpol_mse = list(np.array(interpol_mse)[sort_pos.astype(int)])

        # remove points that are too close         

    y_hat = si.interp1d(x_data[interpol_pos], y_data[interpol_pos])(x_data)
    y_bsp = np.poly1d(np.polyfit(x_data[interpol_pos], y_data[interpol_pos], 3))(x_data)

    # y_bsp=si.interp1d(x_data[interpol_pos],y_data[interpol_pos], kind='cubic')(x_data)
    while False:
        plt.plot(x_data, y_data)
        plt.plot(x_data, y_hat)
        plt.plot(x_data, y_bsp)

    mean_level = y_bsp

    # --------------------------------------------------------------------
    #   find when you acre over the mean level
    # --------------------------------------------------------------------

    # th=2*np.sqrt(np.var(y_data-mean_level))
    th = np.sqrt(np.var(y_data - mean_level))

    pos = np.array(np.where((y_data - mean_level) < th)[0])

    while False:
        plt.plot(x_data, y_data)
        plt.plot(x_data, y_bsp)
        plt.plot(x_data[pos], y_bsp[pos], '*')

    # --------------------------------------------------------------------
    #   merge the points
    # --------------------------------------------------------------------

    diff_pos = pos[1:] - pos[:-1] - 1
    jumps = np.where(diff_pos > 0)[0]

    # if the final element is res, the add a jomp an the end
    if pos[-1] == res - 1:
        jumps = np.append(jumps, pos.shape[0] - 1)

    final_lb = []
    final_rb = []

    if jumps.size == 0:
        final_lb.append(pos[0])
        final_rb.append(pos[-1])
    else:

        final_lb.append(pos[0])
        final_rb.append(pos[jumps[0]])

        k = 0
        while k < jumps.shape[0] - 1:
            #
            final_lb.append(pos[jumps[k] + 1])
            # go to the next gap
            k = k + 1
            final_rb.append(pos[jumps[k]])

    # add the first and the last intervals 

    idx_lr = np.zeros([2, len(final_rb)])
    idx_lr[0] = np.array(final_lb)
    idx_lr[1] = np.array(final_rb)
    idx_lr.astype(int)
    idx_lr = idx_lr.T

    # merge intervals 
    # remove the one intervals
    idx_lr = idx_lr[np.where(idx_lr[:, 1] - idx_lr[:, 0] > 2)[0], :]

    merged = recursive_merge(idx_lr.tolist())
    idx_lr = np.array(merged).astype(int)

    idx_lr_poly = idx_lr

    # -------------------------------------------------------------------
    # peak fitting 
    # -------------------------------------------------------------------

    # first obtain the complement of the intervals

    if idx_lr.size == 0:
        # idx_lr_comp=np.array(range(i,i+int(slide_win)-1))
        idx_lr_comp = np.array([0, slide_win - 1])
        idx_lr_comp = idx_lr_comp.reshape([1, 2])
    else:
        idx_lr_comp = []

        # include the first interval?
        if idx_lr[0, 0] > 0:
            idx_lr_comp.append([0, idx_lr[0, 0]])

        for k in range(1, idx_lr.shape[0]):
            idx_lr_comp.append([idx_lr[k - 1, 1], idx_lr[k, 0]])

        # include the last interval
        if idx_lr[-1, 1] < int(res) - 1:
            idx_lr_comp.append([idx_lr[-1, 1], res])

    idx_lr_comp = np.array(idx_lr_comp).reshape(int(np.size(idx_lr_comp) / 2), 2)

    # remove the intervals that are shorter than the number of parameters, 
    for k in range(idx_lr_comp.shape[0]):
        # expand the fitting window
        idx_lr_comp[k] = find_mul_peak_sym_supp(idx_lr_comp[k], y_data, mean_level, peak_tol)

    idx_lr_comp = np.array(recursive_merge(list(idx_lr_comp)))

    # remove below min peak width

    idx_lr_comp = np.array(idx_lr_comp[np.where(idx_lr_comp[:, 1] - idx_lr_comp[:, 0] > min_peak_width)[0], :])
    idx_lr_comp.astype(int)

    while False:
        peak_pos = []

        for i in range(len(idx_lr_comp)):
            peak_pos = peak_pos + list(range(idx_lr_comp[i][0], idx_lr_comp[i][1]))

        plt.plot(x_data, y_data)
        plt.plot(x_data[peak_pos], y_bsp[peak_pos], '*')

    # this variable is updated at each threshold window
    mse_comp = []
    range_comp = []
    beta_comp = []

    # fit the peaks recursively
    idx_lr_comp = list(idx_lr_comp)

    # idx_lr_comp_bck=idx_lr_comp.copy()

    while len(idx_lr_comp) > 0:

        idx_lr_tmp = idx_lr_comp.pop()

        print('Now processing the range', idx_lr_tmp[0], '--', idx_lr_tmp[1])

        # peak_tol must be set dynamically, based on the value of the max, but how?        
        # idx_comp_lr=find_mul_peak_sym_supp(idx_lr_tmp,y_data,mean_level,peak_tol)

        idx_comp_lr = idx_lr_tmp
        idx_comp = np.array(range(idx_comp_lr[0], idx_comp_lr[1]))

        x_rec = x_data[idx_comp]
        y_rec = y_data[idx_comp]
        mean_rec = mean_level[idx_comp]

        y_up_rec = y_rec - mean_rec

        peak_pos = idx_comp_lr[0] + np.argmax(y_up_rec)

        peak_lr_win = find_peak_supp(peak_pos, int(idx_comp_lr[0]), int(idx_comp_lr[1]), y_data, peak_tol)

        while False:
            # check peaks
            plt.plot(x_data, y_data)
            plt.plot(x_rec, y_rec, '-*')
            plt.plot(x_rec, mean_rec)
            plt.plot(x_data[range(peak_lr_win[0], peak_lr_win[1])], y_data[range(peak_lr_win[0], peak_lr_win[1])], '*')

        y_up_rec = y_rec - mean_rec

        # fitting the peak        
        y_data_temp = y_data[np.array(range(peak_lr_win[0], peak_lr_win[1]))]
        x_data_temp = x_data[np.array(range(peak_lr_win[0], peak_lr_win[1]))]
        mean_level_temp = mean_level[np.array(range(peak_lr_win[0], peak_lr_win[1]))]

        # from ste_model_spectrum_v5 import *        
        # init_fit_fun = init_pseudo_voig
        # loss_peak = mse_loss
        # fit_fun = pseudo_voig

        beta_init = init_fit_fun(x_data_temp, y_data_temp - mean_level_temp)

        result = minimize(loss_peak, beta_init, args=(x_data_temp, y_data_temp - mean_level_temp, fit_fun), tol=1e-12)
        beta_hat_fit = result.x
        mse_fit = result.fun

        if False:
            plt.plot(x_data_temp, y_data_temp)
            plt.plot(x_data_temp, mean_level_temp)
            # plt.plot(x_data_temp,fit_fun(x_data_temp,beta_init)+mean_level_temp)
            plt.plot(x_data_temp, fit_fun(x_data_temp, beta_hat_fit) + mean_level_temp, '*-')

        # now add head and tail to the list

        # include the first interval?
        if peak_lr_win[0] - idx_comp_lr[0] > min_peak_width and peak_lr_win[0] < idx_comp_lr[1]:
            idx_lr_comp.append(np.array([idx_comp_lr[0], peak_lr_win[0] - 1]))
        # includ the last interval?
        if idx_comp_lr[1] - peak_lr_win[1] > min_peak_width and peak_lr_win[1] > idx_comp_lr[0]:
            idx_lr_comp.append(np.array([peak_lr_win[1] + 1, idx_comp_lr[1]]))

        # # include the first interval?
        # if  peak_lr_win[0]-idx_lr_tmp[0]>min_peak_width and peak_lr_win[0]<idx_lr_tmp[1]:
        #     idx_lr_comp.append(np.array([idx_lr_tmp[0],peak_lr_win[0]-1]))
        # # includ the last interval?
        # if idx_lr_tmp[1]-peak_lr_win[1]>min_peak_width and peak_lr_win[1]>idx_lr_tmp[0]:
        #     idx_lr_comp.append(np.array([peak_lr_win[1]+1,idx_lr_tmp[1]]))

        # ------------------------------------------
        # update lists
        if len(range_comp) == 0:
            mse_comp.append(result.fun)
            # range_up.append(peak_l_r_win)
            range_comp.append(np.array(peak_lr_win))
            beta_comp.append(np.array(beta_hat_fit))
        elif not ((range_comp[-1] == peak_lr_win).all()):
            mse_comp.append(result.fun)
            # range_up.append(peak_l_r_win)
            range_comp.append(np.array(peak_lr_win))
            beta_comp.append(np.array(beta_hat_fit))

    # ------------------------------------------
    # storing bit

    if len(range_comp) == 0:
        print('Storing Th', j, ': No peak detected')
        fitting_data[j] = dict(idx_lr_poly=idx_lr_poly, mean_level=mean_level, range_peak=0)
    else:
        print('Storing Th', j, ': ', len(range_comp), ' peaks detected')
        fitting_data[j] = dict(idx_lr_poly=idx_lr_poly, mean_level=mean_level, mse_peak=mse_comp, range_peak=range_comp,
                               beta_peak=beta_comp)

    # -------------------------------------------
    # recap plot
    # title="iteration %i" % num_th
    plt.figure("iteration %i" % j)
    plt.plot(x_data, y_data)
    plt.plot(x_data, mean_level)
    for i in range(len(range_comp)):
        x_data_temp = x_data[range_comp[i][0]:range_comp[i][1]]
        mean_level_temp = mean_level[range_comp[i][0]:range_comp[i][1]]
        plt.plot(x_data_temp, fit_fun(x_data_temp, beta_comp[i]) + mean_level_temp, 'r--')

    # -------------------------------------------
    # update the poly fitting list

    peak_pos = []

    for i in range(len(range_comp)):
        peak_pos = peak_pos + list(range(range_comp[i][0], range_comp[i][1]))
