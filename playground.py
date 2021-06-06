import shelve
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import pandas as pd
import matplotlib.pyplot as plt
from tools.ramanflow.prep_data import PrepData
from tools.ramanflow.read_data import ReadData

f_sup, mg_15ppb_30_10s_hwp42 = ReadData.read_tiff_file(
    'data/20210427 MG ACE ACETAMIPRID colloidal SERS3/15ppb_MG/30_10s_15ppb_HWP80.tif')
f_sup, mg_150ppb_100_100ms_hwp42 = ReadData.read_tiff_file(
    'data/20210427 MG ACE ACETAMIPRID colloidal SERS3/150ppb_MG/100ms_100_batch2_MG_HWP42.tif')
f_sup, mg_1_5ppb_30_10s_hwp42 = ReadData.read_tiff_file(
    'data/20210427 MG ACE ACETAMIPRID colloidal SERS3/1_5ppb_MG/30_10s_1_5ppb_HWP42.tif')
f_sup, mg_150ppb_30_1s_hwp42 = ReadData.read_tiff_file(
    'data/20210427 MG ACE ACETAMIPRID colloidal SERS3/150ppb_MG/30_1s_batch2_HWP42.tif')
import math

filename = 'shelve_save_data.out'
my_shelf = shelve.open(filename)
# klist = list(my_shelf.keys())
for key in my_shelf:
    globals()[key] = my_shelf[key]
my_shelf.close()
f_sup_old = np.copy(f_sup)
f_sup, mg_150ppb_30_1s_hwp42 = ReadData.read_tiff_file(
    'data/20210427 MG ACE ACETAMIPRID colloidal SERS3/150ppb_MG/30_1s_batch2_HWP42.tif')
f_sup, mg_15ppb_30_10s_hwp42 = ReadData.read_tiff_file(
    'data/20210427 MG ACE ACETAMIPRID colloidal SERS3/15ppb_MG/30_10s_15ppb_HWP80.tif')

best_1_5 = np.mean(mg_1_5ppb_30_10s_hwp42, axis=0)
noise_diff_1_5 = np.zeros((30,1600))
mean_till_now_1_5 = np.zeros((30, 1600))
for i in range(0, mg_1_5ppb_30_10s_hwp42.shape[0]):
    mean_till_now_1_5[i] = np.mean(mg_1_5ppb_30_10s_hwp42[0:i], axis=0)
    noise_diff_1_5[i] = mean_till_now_1_5[i] - best_1_5

best_15 = np.mean(mg_15ppb_30_10s_hwp42)
noise_diff_15 = np.zeros((30, 1600))
mean_till_now_15 = np.zeros((30, 1600))
for i in range(0, mg_15ppb_30_10s_hwp42.shape[0]):
    mean_till_now_15[i] = np.mean(mg_15ppb_30_10s_hwp42[0:i], axis=0)
    noise_diff_15[i] = mean_till_now_15[i] - best_15

best_150_1 = np.mean(mg_150ppb_30_1s_hwp42)
noise_diff_150_1 = np.zeros((30, 1600))
mean_till_now_150_1 = np.zeros((30, 1600))
for i in range(0, mg_150ppb_30_1s_hwp42.shape[0]):
    mean_till_now_150_1[i] = np.mean(mg_150ppb_30_1s_hwp42[0:i], axis=0)
    noise_diff_150_1[i] = mean_till_now_150_1[i] - best_150_1

best_150_100 = np.mean(mg_150ppb_30_1s_hwp42)
noise_diff_150_100 = np.zeros((100, 1600))
mean_till_now_150_100 = np.zeros((100, 1600))
for i in range(0, mg_150ppb_100_100ms_hwp42.shape[0]):
    mean_till_now_150_100[i] = np.mean(mg_150ppb_100_100ms_hwp42[0:i], axis=0)
    noise_diff_150_100[i] = mean_till_now_150_100[i] - best_150_100

snr_150_100 = np.zeros((100,))
for i in range(0, noise_diff_150_100.shape[0]):
    snr_150_100[i] = np.sum(np.abs(mg_150ppb_100_100ms_hwp42[0:i]) ** 2) / np.sum(np.abs(noise_diff_150_100[i]) ** 2)
    snr_150_100[i] = 10 * math.log10(snr_150_100[i])
    print("SNR for 1.5ppb after {}th recording is {}".format(i, snr_150_100[i]))

snr_150_1 = np.zeros((30,))
for i in range(0, noise_diff_150_1.shape[0]):
    snr_150_1[i] = np.sum(np.abs(mg_150ppb_30_1s_hwp42[0:i]) ** 2) / np.sum(np.abs(noise_diff_150_1[i]) ** 2)
    snr_150_1[i] = 10 * math.log10(snr_150_1[i])
    print("SNR for 1.5ppb after {}th recording is {}".format(i, snr_150_1[i]))

snr_15 = np.zeros((30,))
for i in range(0, noise_diff_15.shape[0]):
    snr_15[i] = np.sum(np.abs(mg_15ppb_30_10s_hwp42[0:i]) ** 2) / np.sum(np.abs(noise_diff_15[i]) ** 2)
    snr_15[i] = 10 * math.log10(snr_15[i])
    print("SNR for 1.5ppb after {}th recording is {}".format(i, snr_15[i]))

snr_1_5 = np.zeros((30,))
for i in range(0, noise_diff_1_5.shape[0]):
    snr_1_5[i] = np.sum(np.abs(mg_1_5ppb_30_10s_hwp42[0:i]) ** 2) / np.sum(np.abs(noise_diff_1_5[i]) ** 2)
    snr_1_5[i] = 10 * math.log10(snr_1_5[i])
    print("SNR for 1.5ppb after {}th recording is {}".format(i, snr_1_5[i]))


check = np.diff(mean_till_now_1_5, axis=-1)
check_abs = np.abs(check)
check_slope = check_abs / np.abs(np.diff(f_sup))

plt.figure(1, figsize=(15, 10))
percentile = np.absolute(snr_150_1[1:] - np.nanpercentile(snr_150_1, 80)).argmin()
plt.vlines(percentile, ymin=np.nanmin(snr_150_1), ymax=np.nanmax(snr_150_1), alpha=.2, lw=3, label='80th percentile, after {}th recording'.format(percentile))
plt.plot(np.arange(0, 30), snr_150_1)
plt.xlabel("# of recordings")
plt.ylabel("SNR (db)")
plt.legend()
plt.title("150ppb 1s integration time MG SNR evolution")