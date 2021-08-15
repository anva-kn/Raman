print('PyDev console: using IPython 7.22.0\n')

import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['/Users/anvarkunanbaev/PycharmProjects/Raman_2.0/Raman'])
%load_ext autoreload
pwd
%autoreload 2
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
123/9:
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
f_sup, mg_15ppb_30_10s_hwp42 = ReadData.read_tiff_file(
    'data/20210427 MG ACE ACETAMIPRID colloidal SERS3/15ppb_MG/30_10s_15ppb_HWP80.tif')
f_sup, mg_150ppb_100_100ms_hwp42 = ReadData.read_tiff_file(
    'data/20210427 MG ACE ACETAMIPRID colloidal SERS3/150ppb_MG/100ms_100_batch2_MG_HWP42.tif')
f_sup, mg_1_5ppb_30_10s_hwp42 = ReadData.read_tiff_file(
    'data/20210427 MG ACE ACETAMIPRID colloidal SERS3/1_5ppb_MG/30_10s_1_5ppb_HWP42.tif')
f_sup, mg_150ppb_30_1s_hwp42 = ReadData.read_tiff_file(
    'data/20210427 MG ACE ACETAMIPRID colloidal SERS3/150ppb_MG/30_1s_batch2_HWP42.tif')
import math
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
clipped_f_sup = np.copy(f_sup[335:])
clipped_mg_1_5 = np.copy(mg_1_5ppb_30_10s_hwp42[:, 335:])
clipped_mg_15 = np.copy(mg_15ppb_30_10s_hwp42[:, 335:])
clipped_mg_150_1 = np.copy(mg_150ppb_30_1s_hwp42[:, 335:])
clipped_mg_150_100 = np.copy(mg_150ppb_100_100ms_hwp42[:, 335:])
clipped_wo_rays_1_5 = PrepData.remove_cosmic_rays(clipped_mg_1_5, 5)
clipped_wo_rays_15 = PrepData.remove_cosmic_rays(clipped_mg_15, 5)
clipped_wo_rays_150_1 = PrepData.remove_cosmic_rays(clipped_mg_150_1, 5)
clipped_wo_rays_150_100 = PrepData.remove_cosmic_rays(clipped_mg_150_100, 5)
from scipy.signal import find_peaks
mean_clipped_1_5 = np.mean(clipped_wo_rays_1_5, axis=0)
mean_clipped_15 = np.mean(clipped_wo_rays_15, axis=0)
mean_clipped_150_1 = np.mean(clipped_wo_rays_150_1, axis=0)
mean_clipped_150_100 = np.mean(clipped_wo_rays_150_100, axis=0)
plt.plot(clipped_f_sup, mean_till_now_1_5, label='1.5ppb')
plt.plot(clipped_f_sup, mean_till_now_15, label='15ppb')
plt.plot(clipped_f_sup, mean_till_now_150_1, label='150ppb 1sec')
plt.plot(clipped_f_sup, mean_till_now_150_100, label='150ppb 100ms')
plt.plot(clipped_f_sup, mean_clipped_1_5, label='1.5ppb')
plt.plot(clipped_f_sup, mean_clipped_15, label='15ppb')
plt.plot(clipped_f_sup, mean_clipped_150_1, label='150ppb 1sec')
plt.plot(clipped_f_sup, mean_clipped_150_100, label='150ppb 100ms')
peaks_raw_1_5, properties_raw = find_peaks(mean_clipped_1_5, width=10)
plt.plot(clipped_f_sup, mean_clipped_1_5)
plt.plot(clipped_f_sup(peaks_raw_1_5), mean_clipped_1_5(peaks_raw_1_5))
plt.plot(clipped_f_sup, mean_clipped_1_5)
plt.plot(clipped_f_sup[peaks_raw_1_5], mean_clipped_1_5[peaks_raw_1_5], '*')
peaks_raw_1_5, properties_raw = find_peaks(mean_clipped_1_5, prominence=50,width=10)
plt.plot(clipped_f_sup, mean_clipped_1_5)
plt.plot(clipped_f_sup[peaks_raw_1_5], mean_clipped_1_5[peaks_raw_1_5], '*')
plt.plot(clipped_f_sup, mean_clipped_1_5)
plt.plot(clipped_f_sup[peaks_raw_1_5], mean_clipped_1_5[peaks_raw_1_5], '*')
top_five_peak_amplitude_1_5 = np.zeros((30, 5))
top_five_peak_amplitude_15 = np.zeros((30, 5))
top_five_peak_amplitude_150_1 = np.zeros((30, 5))
top_five_peak_amplitude_150_100 = np.zeros((100, 5))
peaks_raw_15, _ = find_peaks(mean_clipped_15, prominence=50,width=10)
peaks_raw_15
peaks_raw_15, _ = find_peaks(mean_clipped_15, prominence=30,width=10)
plt.plot(clipped_f_sup, mean_clipped_15)
plt.plot(clipped_f_sup[peaks_raw_15], mean_clipped_15[peaks_raw_15], '*')
peaks_raw_150_1, _ = find_peaks(mean_clipped_150_1, prominence=30,width=10)
plt.plot(clipped_f_sup, mean_clipped_150_1)
plt.plot(clipped_f_sup[peaks_raw_150_1], mean_clipped_150_1[peaks_raw_150_1], '*')
clipped_wo_bg_1_5 = PrepData.spline_remove_est_fluorescence(clipped_f_sup, clipped_wo_rays_1_5, 20)
clipped_wo_bg_15 = PrepData.spline_remove_est_fluorescence(clipped_f_sup, clipped_wo_rays_15, 20)
clipped_wo_bg_150_1 = PrepData.spline_remove_est_fluorescence(clipped_f_sup, clipped_wo_rays_150_1, 20)
clipped_wo_bg_150_100 = PrepData.spline_remove_est_fluorescence(clipped_f_sup, clipped_wo_rays_150_100, 20)
peaks_raw_1_5, _ = find_peaks(mean_clipped_1_5, prominence=50,width=10)
peaks_raw_15, _ = find_peaks(mean_clipped_15, prominence=30,width=10)
peaks_raw_150_1, _ = find_peaks(mean_clipped_150_1, prominence=30,width=10)
peaks_raw_150_100, _ = find_peaks(np.mean(clipped_wo_bg_150_100,axis=0), prominence=30,width=10)
peaks_raw_1_5, _ = find_peaks(np.mean(clipped_wo_bg_1_5,axis=0), prominence=50,width=10)
peaks_raw_15, _ = find_peaks(np.mean(clipped_wo_bg_15,axis=0), prominence=30,width=10)
peaks_raw_150_1, _ = find_peaks(np.mean(clipped_wo_bg_150_1,axis=0), prominence=30,width=10)
peaks_150_100, _ = find_peaks(np.mean(clipped_wo_bg_150_100,axis=0), prominence=30,width=10)
peaks_1_5, _ = find_peaks(np.mean(clipped_wo_bg_1_5,axis=0), prominence=50,width=10)
peaks_15, _ = find_peaks(np.mean(clipped_wo_bg_15,axis=0), prominence=30,width=10)
peaks_150_1, _ = find_peaks(np.mean(clipped_wo_bg_150_1,axis=0), prominence=30,width=10)
final_peak_pos_sorted_1_5 = peaks_1_5[np.argsort(np.mean(clipped_wo_bg_1_5,axis=0)[peaks_1_5])]
final_peak_pos_sorted_15 = peaks_15[np.argsort(np.mean(clipped_wo_bg_15,axis=0)[peaks_15])]
final_peak_pos_sorted_150_1 = peaks_150_1[np.argsort(np.mean(clipped_wo_bg_150_1,axis=0)[peaks_150_1])]
final_peak_pos_sorted_150_100 = peaks_150_100[np.argsort(np.mean(clipped_wo_bg_150_100,axis=0)[peaks_150_100])]
peaks_150_100, _ = find_peaks(np.mean(clipped_wo_bg_150_100,axis=0), prominence=25,width=10)
peaks_150_100, _ = find_peaks(np.mean(clipped_wo_bg_150_100,axis=0), prominence=20,width=10)
plt.plot(clipped_f_sup, mean_clipped_150_100)
plt.plot(clipped_f_sup[peaks_150_100], mean_clipped_150_100[peaks_150_100])
plt.plot(clipped_f_sup, mean_clipped_150_100)
plt.plot(clipped_f_sup[peaks_150_100], mean_clipped_150_100[peaks_150_100], '*')
final_peak_pos_sorted_150_100 = peaks_150_100[np.argsort(np.mean(clipped_wo_bg_150_100,axis=0)[peaks_150_100])]
plt.plot(clipped_f_sup, np.mean(clipped_wo_bg_150_100,axis=0))
plt.plot(clipped_f_sup[peaks_150_100], np.mean(clipped_wo_bg_150_100,axis=0)[peaks_150_100], '*')
for i in range(0, top_five_peak_amplitude_1_5.shape[0]):
     top_five_peak_amplitude_1_5[i] = np.mean(clipped_wo_bg_1_5,axis=0)[i, final_peak_pos_sorted_1_5]
     top_five_peak_amplitude_15[i] = np.mean(clipped_wo_bg_15,axis=0)[i, final_peak_pos_sorted_15]
     top_five_peak_amplitude_150_1[i] = np.mean(clipped_wo_bg_150_1,axis=0)[i, final_peak_pos_sorted_150_1]
mean_till_now_1_5_no_bg = np.zeros((30, clipped_f_sup.size))
mean_till_now_15_no_bg = np.zeros((30, clipped_f_sup.size))
mean_till_now_150_1_no_bg = np.zeros((30, clipped_f_sup.size))
mean_till_now_150_100_no_bg = np.zeros((100, clipped_f_sup.size))
for i in range(0, top_five_peak_amplitude_1_5.shape[0]):
     top_five_peak_amplitude_1_5[i] = mean_till_now_1_5_no_bg[i, final_peak_pos_sorted_1_5]
     top_five_peak_amplitude_15[i] = mean_till_now_15_no_bg[i, final_peak_pos_sorted_15]
     top_five_peak_amplitude_150_1[i] = mean_till_now_150_1_no_bg[i, final_peak_pos_sorted_150_1]
for i in range(0, top_five_peak_amplitude_1_5.shape[0]):
     top_five_peak_amplitude_1_5[i] = mean_till_now_1_5_no_bg[i, final_peak_pos_sorted_1_5[-5:]]
     top_five_peak_amplitude_15[i] = mean_till_now_15_no_bg[i, final_peak_pos_sorted_15[-5:]]
     top_five_peak_amplitude_150_1[i] = mean_till_now_150_1_no_bg[i, final_peak_pos_sorted_150_1[-5:]]
for i in range(0, top_five_peak_amplitude_150_100.shape[0]):
     top_five_peak_amplitude_150_100[i] = mean_till_now_150_100_no_bg[i, final_peak_pos_sorted_150_100[-5:]]
plt.figure(1, figsize=(15, 10))
for i in range(0, top_five_peak_amplitude_1_5.shape[-1]):
    plt.plot(np.arange(1, 31), top_five_peak_amplitude_1_5[:, i], label="#{} largest peak".format(6 - (i+1)))
plt.xlabel("# of recordings")
plt.ylabel("Intensity")
plt.legend(loc='best')
plt.title("1.5ppb 1s integration time MG peak intensity evolution")
plt.show()
for i in range(0, clipped_wo_bg_1_5.shape[0]):
     mean_till_now_1_5_no_bg[i] = np.mean(clipped_wo_bg_1_5[0:i], axis=0)
     mean_till_now_15_no_bg[i] = np.mean(clipped_wo_bg_15[0:i], axis=0)
     mean_till_now_150_1_no_bg_no_bg[i] = np.mean(clipped_wo_bg_150_1[0:i], axis=0)
for i in range(0, clipped_wo_bg_1_5.shape[0]):
     mean_till_now_1_5_no_bg[i] = np.mean(clipped_wo_bg_1_5[0:i], axis=0)
     mean_till_now_15_no_bg[i] = np.mean(clipped_wo_bg_15[0:i], axis=0)
     mean_till_now_150_1_no_bg[i] = np.mean(clipped_wo_bg_150_1[0:i], axis=0)
for i in range(0, clipped_wo_bg_150_100.shape[0]):
     mean_till_now_150_100_no_bg[i] = np.mean(clipped_wo_bg_150_100[0:i], axis=0)
for i in range(0, top_five_peak_amplitude_1_5.shape[0]):
    top_five_peak_amplitude_1_5[i] = mean_till_now_1_5_no_bg[i, final_peak_pos_sorted_1_5[-5:]]
    top_five_peak_amplitude_15[i] = mean_till_now_15_no_bg[i, final_peak_pos_sorted_15[-5:]]
    top_five_peak_amplitude_150_1[i] = mean_till_now_150_1_no_bg[i, final_peak_pos_sorted_150_1[-5:]]

for i in range(0, top_five_peak_amplitude_150_100.shape[0]):
    top_five_peak_amplitude_150_100[i] = mean_till_now_150_100_no_bg[i, final_peak_pos_sorted_150_100[-5:]]
plt.figure(1, figsize=(15, 10))
for i in range(0, top_five_peak_amplitude_1_5.shape[-1]):
    plt.plot(np.arange(1, 31), top_five_peak_amplitude_1_5[:, i], label="#{} largest peak".format(6 - (i+1)))
plt.xlabel("# of recordings")
plt.ylabel("Intensity")
plt.legend(loc='best')
plt.title("1.5ppb 1s integration time MG peak intensity evolution")
plt.show()
plt.figure(1, figsize=(15, 10))
for i in range(0, top_five_peak_amplitude_1_5.shape[-1]):
    plt.plot(np.arange(1, 31), top_five_peak_amplitude_15[:, i], label="#{} largest peak".format(6 - (i+1)))
plt.xlabel("# of recordings")
plt.ylabel("Intensity")
plt.legend(loc='best')
plt.title("1.5ppb 1s integration time MG peak intensity evolution")
plt.show()
plt.figure(1, figsize=(15, 10))
for i in range(0, top_five_peak_amplitude_1_5.shape[-1]):
    plt.plot(np.arange(1, 31), top_five_peak_amplitude_1_5[:, i], label="#{} largest peak".format(6 - (i+1)))
plt.xlabel("# of recordings")
plt.ylabel("Intensity")
plt.legend(loc='best')
plt.title("1.5ppb 1s integration time MG peak intensity evolution")
plt.savefig('1_5ppb_peaks_evol.png', dpi=400)
plt.figure(1, figsize=(15, 10))
for i in range(0, top_five_peak_amplitude_15.shape[-1]):
    plt.plot(np.arange(1, 31), top_five_peak_amplitude_15[:, i], label="#{} largest peak".format(6 - (i+1)))
plt.xlabel("# of recordings")
plt.ylabel("Intensity")
plt.legend(loc='best')
plt.title("15ppb 1s integration time MG peak intensity evolution")
plt.savefig('15ppb_peaks_evol.png', dpi=400)
plt.figure(1, figsize=(15, 10))
for i in range(0, top_five_peak_amplitude_150_1.shape[-1]):
    plt.plot(np.arange(1, 31), top_five_peak_amplitude_150_1[:, i], label="#{} largest peak".format(6 - (i+1)))
plt.xlabel("# of recordings")
plt.ylabel("Intensity")
plt.legend(loc='best')
plt.title("150ppb 1s integration time MG peak intensity evolution")
plt.savefig('150ppb_1s_peaks_evol.png', dpi=400)
plt.figure(1, figsize=(15, 10))
for i in range(0, top_five_peak_amplitude_150_100.shape[-1]):
    plt.plot(np.arange(1, 101), top_five_peak_amplitude_150_100[:, i], label="#{} largest peak".format(6 - (i+1)))
plt.xlabel("# of recordings")
plt.ylabel("Intensity")
plt.legend(loc='best')
plt.title("150ppb 100ms integration time MG peak intensity evolution")
plt.savefig('150ppb_100ms_peaks_evol.png', dpi=400)
history -f 2021-06-24_logs.py
