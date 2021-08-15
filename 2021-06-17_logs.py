clear
pwd
%load_ext autoreload
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
dir()
_sup, mg_150ppb_30_1s_hwp42 = ReadData.read_tiff_file(
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
clear
dir()
mean_till_now_1_5.shape
plt.plot(f_sup, mg_1_5ppb_30_10s_hwp42[:, ])
plt.plot(f_sup, mg_1_5ppb_30_10s_hwp42[:, ])
plt.plot(f_sup, mg_1_5ppb_30_10s_hwp42[:, ])
for i in range(0, mg_1_5ppb_30_10s_hwp42.shapep[0]):
    plt.plot(f_sup, mg_1_5ppb_30_10s_hwp42[i])
for i in range(0, mg_1_5ppb_30_10s_hwp42.shape[0]):
    plt.plot(f_sup, mg_1_5ppb_30_10s_hwp42[i])
plt.show()
np.where(f_sup >= 500)
clipped_f_sup = np.copy(f_sup[335:])
clipped_f_sup.size
cl
clipped_mg_1_5 = np.copy(mg_1_5ppb_30_10s_hwp42[:, 335:])
clipped_mg_1_5.shape
for i in range(0, clipped_mg_1_5.shape[0]):
    plt.plot(clipped_f_sup, clipped_mg_1_5[i])
plt.show()
clipped_wo_rays = PrepData.remove_cosmic_rays(clipped_mg_1_5, 5)
for i in range(0, clipped_mg_1_5.shape[0]):
    plt.plot(clipped_f_sup, clipped_wo_rays[i])
plt.show()
check = np.copy(clipped_mg_1_5[19])
plt.plot(clipped_f_sup, check)
plt.show()
from scipy.signal import find_peaks
peaks_raw, properties_raw = find_peaks(check, prominence=1, width=10)
peaks_raw
peaks_raw, properties_raw = find_peaks(check, prominence=1, width=7)
peaks_raw
plt.plot(clipped_f_sup, check)
plt.plot(clipped_f_sup[peaks_raw], check[peaks_raw])
plt.show()
check_avg = np.mean(clipped_wo_rays, axis=0)
plt.plot(clipped_f_sup, check_avg)
plt.show()
peaks_avg, properties_avg = find_peaks(check_avg, width=7)
peaks_avg
peaks_avg, properties_avg = find_peaks(check_avg, width=10)
peaks_avg
peaks_avg, properties_avg = find_peaks(check_avg, width=15)
peaks_avg
plt.plot(clipped_f_sup, check_avg, label='1.5 average')
plt.plot(clipped_f_sup[peaks_avg], check_avg[peaks_avg], '*')
plt.show()
from scipy.signal import peak_prominences
prominence_abg, prom_prop = peak_prominences(check_avg, peaks_avg)
prominence_avg = peak_prominences(check_avg, peaks_avg)[0]
prominence_avg
peaks_avg
clipped_wo_bg = PrepData.spline_remove_est_fluorescence(clipped_f_sup, clipped_wo_rays, 20)
clipped_wo_bg = PrepData.spline_remove_est_fluorescence(clipped_f_sup, clipped_wo_rays, 20)
plt.plot(clipped_f_sup, np.mean(clipped_wo_bg, axis=0))
plt.show()
peaks_no_bg = np.where(clipped_wo_bg > 100)
peaks_no_bg
peaks_no_bg = np.where(np.mean(clipped_wo_bg, axis=0) > 100)
peaks_no_bg
peaks_no_bg = find_peaks(np.mean(clipped_wo_bg, axis=0), prominence=100)
peaks_no_bg
peaks_no_bg, _ = find_peaks(np.mean(clipped_wo_bg, axis=0), prominence=100)
peaks_no_bg
prom_no_bg = peak_prominences(clipped_wo_bg, [475, 640])[0]
prom_no_bg = peak_prominences(np.mean(clipped_wo_bg, axis=0), [475, 640])[0]
prom_no_bg
check_no_bg_avg = np.mean(clipped_wo_bg, axis=0)
peaks_no_bg, _ = find_peaks(check_no_bg_avg, width=7)
peaks_no_bg
plt.plot(clipped_f_sup, check_no_bg_avg)
plt.plot(clipped_f_sup[peaks_no_bg], check_no_bg_avg[peaks_no_bg])
plt.show()
plt.plot(clipped_f_sup, check_no_bg_avg)
plt.plot(clipped_f_sup[peaks_no_bg], check_no_bg_avg[peaks_no_bg], '*')
plt.show()
peaks_no_bg, _ = find_peaks(check_no_bg_avg, width=10)
plt.plot(clipped_f_sup, check_no_bg_avg)
plt.plot(clipped_f_sup[peaks_no_bg], check_no_bg_avg[peaks_no_bg], '*')
plt.show()
peaks_no_bg, _ = find_peaks(check_no_bg_avg, prominence=100, width=10)
plt.plot(clipped_f_sup, check_no_bg_avg)
plt.plot(clipped_f_sup[peaks_no_bg], check_no_bg_avg[peaks_no_bg], '*')
plt.show()
peaks_no_bg, _ = find_peaks(check_no_bg_avg, prominence=60, width=10)
plt.plot(clipped_f_sup, check_no_bg_avg)
plt.plot(clipped_f_sup[peaks_no_bg], check_no_bg_avg[peaks_no_bg], '*')
plt.show()
top_five_peak_amplitude = np.zeros(30, 5)
top_five_peak_amplitude = np.zeros((30, 5))
_ = np.argsort(check_no_bg_avg[peaks_no_bg])
_
peaks_no_bg
peaks_no_bg[np.argsort(check_no_bg_avg[peaks_no_bg])]
plt.plot(clipped_f_sup, check_no_bg_avg)
plt.plot(clipped_f_sup[peaks_no_bg[np.argsort(check_no_bg_avg[peaks_no_bg])[-1]]], check_no_bg_avg[peaks_no_bg[np.argsort(check_no_bg_avg[peaks_no_bg])[-1]]], '*')
plt.show()
plt.plot(clipped_f_sup, check_no_bg_avg)
plt.plot(clipped_f_sup[peaks_no_bg[np.argsort(check_no_bg_avg[peaks_no_bg])[-1]]], check_no_bg_avg[peaks_no_bg[np.argsort(check_no_bg_avg[peaks_no_bg])[-1]]], '*', label='1st')
plt.plot(clipped_f_sup, check_no_bg_avg)
plt.plot(clipped_f_sup[peaks_no_bg[np.argsort(check_no_bg_avg[peaks_no_bg])[-2]]], check_no_bg_avg[peaks_no_bg[np.argsort(check_no_bg_avg[peaks_no_bg])[-2]]], '*', label='2nd')
plt.plot(clipped_f_sup, check_no_bg_avg)
plt.plot(clipped_f_sup[peaks_no_bg[np.argsort(check_no_bg_avg[peaks_no_bg])[-3]]], check_no_bg_avg[peaks_no_bg[np.argsort(check_no_bg_avg[peaks_no_bg])[-3]]], '*', label='3rd')
plt.show()
plt.plot(clipped_f_sup, check_no_bg_avg)
plt.plot(clipped_f_sup[peaks_no_bg[np.argsort(check_no_bg_avg[peaks_no_bg])[-1]]], check_no_bg_avg[peaks_no_bg[np.argsort(check_no_bg_avg[peaks_no_bg])[-1]]], '*', label='1st')
plt.plot(clipped_f_sup[peaks_no_bg[np.argsort(check_no_bg_avg[peaks_no_bg])[-2]]], check_no_bg_avg[peaks_no_bg[np.argsort(check_no_bg_avg[peaks_no_bg])[-2]]], '*', label='2nd')
plt.plot(clipped_f_sup[peaks_no_bg[np.argsort(check_no_bg_avg[peaks_no_bg])[-3]]], check_no_bg_avg[peaks_no_bg[np.argsort(check_no_bg_avg[peaks_no_bg])[-3]]], '*', label='3rd')
plt.legend()
plt.show()
final_peak_pos_sorted = peaks_no_bg[np.argsort(check_no_bg_avg[peaks_no_bg])]
final_peak_pos_sorted
clear
dir()
top_five_peak_amplitude.shape
for i in range(0, mean_till_now_1_5.shape[0]):
    print(mean_till_now_1_5[i, final_peak_pos_sorted])
plt.plot(clipped_f_sup, check_no_bg_avg)
plt.plot(clipped_f_sup[final_peak_pos_sorted[-1]], check_no_bg_avg[final_peak_pos_sorted[-1]], '*')
plt.show()
mean_till_now_1_5_no_bg = np.zeros((30, clipped_f_sup.size))
for i in range(0, mg_1_5ppb_30_10s_hwp42.shape[0]):
    mean_till_now_1_5_no_bg[i] = np.mean(clipped_wo_bg[0:i], axis=0)
for i in range(0, mean_till_now_1_5.shape[0]):
    print(mean_till_now_1_5_no_bg[i, final_peak_pos_sorted])
for i in range(0, top_five_peak_amplitude.shape[0]):
    top_five_peak_amplitude[i] = mean_till_now_1_5_no_bg[i, final_peak_pos_sorted]
top_five_peak_amplitude
clear
for i in range(0, top_five_peak_amplitude.shape[0]):
    plt.plot(np.arange(1,31), top_five_peak_amplitude[i, -1])
plt.plot(np.arange(1,31), top_five_peak_amplitude[:, -1])
plt.show()
for i in range(0, top_five_peak_amplitude.shape[-1]):
    plt.plot(np.arange(1,31), top_five_peak_amplitude[:, i], label="#{} largest peak".format(i+1))
plt.show()
plt.legend()
for i in range(0, top_five_peak_amplitude.shape[-1]):
    plt.plot(np.arange(1,31), top_five_peak_amplitude[:, i], label="#{} largest peak".format(i+1))
plt.legend()
plt.show()
plt.figure(1, figsize=(15, 10))
for i in range(0, top_five_peak_amplitude.shape[-1]):
    plt.plot(np.arange(1, 31), top_five_peak_amplitude[:, i], label="#{} largest peak".format(i+1))
plt.xlabel("# of recordings")
plt.ylabel("Intensity")
plt.legend(loc='center right')
plt.title("1.5ppb 1s integration time MG peak intensity evolution")
plt.show()
plt.figure(1, figsize=(15, 10))
for i in range(0, top_five_peak_amplitude.shape[-1]):
    plt.plot(np.arange(1, 31), top_five_peak_amplitude[:, i], label="#{} largest peak".format(i+1))
plt.xlabel("# of recordings")
plt.ylabel("Intensity")
plt.legend(loc='best')
plt.title("1.5ppb 1s integration time MG peak intensity evolution")
plt.show()
plt.figure(1, figsize=(15, 10))
for i in range(0, top_five_peak_amplitude.shape[-1]):
    plt.plot(np.arange(1, 31), top_five_peak_amplitude[:, -i], label="#{} largest peak".format(i+1))
plt.xlabel("# of recordings")
plt.ylabel("Intensity")
plt.legend(loc='best')
plt.title("1.5ppb 1s integration time MG peak intensity evolution")
plt.show()
plt.figure(1, figsize=(15, 10))
for i in range(0, top_five_peak_amplitude.shape[-1]):
    plt.plot(np.arange(1, 31), top_five_peak_amplitude[:, i], label="#{} largest peak".format(-i+1))
plt.xlabel("# of recordings")
plt.ylabel("Intensity")
plt.legend(loc='best')
plt.title("1.5ppb 1s integration time MG peak intensity evolution")
plt.show()
plt.figure(1, figsize=(15, 10))
for i in range(0, top_five_peak_amplitude.shape[-1]):
    plt.plot(np.arange(1, 31), top_five_peak_amplitude[:, i], label="#{} largest peak".format(-i))
plt.xlabel("# of recordings")
plt.ylabel("Intensity")
plt.legend(loc='best')
plt.title("1.5ppb 1s integration time MG peak intensity evolution")
plt.show()
plt.figure(1, figsize=(15, 10))
for i in range(0, top_five_peak_amplitude.shape[-1]):
    plt.plot(np.arange(1, 31), top_five_peak_amplitude[:, i], label="#{} largest peak".format(6 - (i+1)))
plt.xlabel("# of recordings")
plt.ylabel("Intensity")
plt.legend(loc='best')
plt.title("1.5ppb 1s integration time MG peak intensity evolution")
plt.show()
plt.figure(1, figsize=(15, 10))
for i in range(0, top_five_peak_amplitude.shape[-1]):
    plt.plot(np.arange(1, 31), top_five_peak_amplitude[:, i], label="#{} largest peak".format(6 - (i+1)))
plt.xlabel("# of recordings")
plt.ylabel("Intensity")
plt.legend(loc='best')
plt.title("1.5ppb 1s integration time MG peak intensity evolution")
plt.savefig('1_5ppb_peaks_evol.png', dpi=400)
clipped_f_sup.shape
plt.figure(1, figsize=(15,10))
plt.plot(f_sup, np.mean(mg_1_5ppb_30_10s_hwp42), label='Original')
plt.plot(clipped_f_sup, check_avg, label='Clipped')
plt.figure(1, figsize=(15,10))
plt.plot(f_sup, np.mean(mg_1_5ppb_30_10s_hwp42, axis=0), label='Original')
plt.plot(clipped_f_sup, check_avg, label='Clipped')
plt.legend()
plt.savefig('orig_vs_clipped.png', dpi=400)
plt.figure(2, figsize=(15,10))
plt.plot(f_sup, np.mean(mg_1_5ppb_30_10s_hwp42, axis=0), label='Original')
plt.plot(clipped_f_sup, check_avg, label='Clipped')
plt.legend()
plt.savefig('orig_vs_clipped.png', dpi=400)
history -f '2021-06-17_logs.txt'
