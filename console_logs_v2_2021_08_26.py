>>> print('PyDev console: using IPython 7.26.0\n')
... 
... import sys; print('Python %s on %s' % (sys.version, sys.platform))
... sys.path.extend(['/home/anvarkunanbayev/PycharmProjects/Raman'])
...
>>> import matplotlib
... matplotlib.use('Qt5Agg')
... import matplotlib.pyplot as plt
... 
... import pandas as pd
... from tools.ramanflow.read_data import ReadData as RD
... from tools.ramanflow.prep_data import PrepData as PD
... 
... from scipy.signal import savgol_filter
... import numpy as np
...
>>> f_sup_0813, car_0813 = RD.read_dir_tiff_files('data/20210813 SERS timed immersion experiment/1')
... f_sup_0813, aceta_0813 = RD.read_dir_tiff_files('data/20210813 SERS timed immersion experiment/4')
... f_sup_0813, aceph_0813 = RD.read_dir_tiff_files('data/20210813 SERS timed immersion experiment/5')
... f_sup_0813, car_aceta_0813 = RD.read_dir_tiff_files('data/20210813 SERS timed immersion experiment/1+4')
... f_sup_0813, car_aceph_0813 = RD.read_dir_tiff_files('data/20210813 SERS timed immersion experiment/1+5')
... f_sup_0813, aceta_aceph_0813 = RD.read_dir_tiff_files('data/20210813 SERS timed immersion experiment/4+5')
... f_sup_0813, car_aceta_aceph_0813 = RD.read_dir_tiff_files('data/20210813 SERS timed immersion experiment/1+4+5')
... f_sup_0810, car_0810 = RD.read_dir_tiff_files('data/20210810 SERS timed immersion experiment/1')
... f_sup_0810, aceta_0810 = RD.read_dir_tiff_files('data/20210810 SERS timed immersion experiment/4')
... f_sup_0810, aceph_0810 = RD.read_dir_tiff_files('data/20210810 SERS timed immersion experiment/5')
... f_sup_0810, car_aceta_0810 = RD.read_dir_tiff_files('data/20210810 SERS timed immersion experiment/1+4')
... f_sup_0810, car_aceph_0810 = RD.read_dir_tiff_files('data/20210810 SERS timed immersion experiment/1+5')
... f_sup_0810, aceta_aceph_0810 = RD.read_dir_tiff_files('data/20210810 SERS timed immersion experiment/4+5')
... f_sup_0810, car_aceta_aceph_0810 = RD.read_dir_tiff_files('data/20210810 SERS timed immersion experiment/1+4+5')
... f_sup_0811, car_0811 = RD.read_dir_tiff_files('data/20210811 SERS timed immersion experiment/1')
... f_sup_0811, aceta_0811 = RD.read_dir_tiff_files('data/20210811 SERS timed immersion experiment/4')
... f_sup_0811, aceph_0811 = RD.read_dir_tiff_files('data/20210811 SERS timed immersion experiment/5')
... f_sup_0811, car_aceta_0811 = RD.read_dir_tiff_files('data/20210811 SERS timed immersion experiment/1+4')
... f_sup_0811, car_aceph_0811 = RD.read_dir_tiff_files('data/20210811 SERS timed immersion experiment/1+5')
... f_sup_0811, aceta_aceph_0811 = RD.read_dir_tiff_files('data/20210811 SERS timed immersion experiment/4+5')
... f_sup_0811, car_aceta_aceph_0811 = RD.read_dir_tiff_files('data/20210811 SERS timed immersion experiment/1+4+5')
...
>>> f_sup_0818, car_0818 = RD.read_dir_tiff_files('data/20210818 SERS timed immersion experiment/1')
... f_sup_0818, aceta_0818 = RD.read_dir_tiff_files('data/20210818 SERS timed immersion experiment/4')
... f_sup_0818, aceph_0818 = RD.read_dir_tiff_files('data/20210818 SERS timed immersion experiment/5')
... f_sup_0818, car_aceta_0818 = RD.read_dir_tiff_files('data/20210818 SERS timed immersion experiment/1+4')
... f_sup_0818, car_aceph_0818 = RD.read_dir_tiff_files('data/20210818 SERS timed immersion experiment/1+5')
... f_sup_0818, aceta_aceph_0818 = RD.read_dir_tiff_files('data/20210818 SERS timed immersion experiment/4+5')
... f_sup_0818, car_aceta_aceph_0818 = RD.read_dir_tiff_files('data/20210818 SERS timed immersion experiment/1+4+5')
...
>>> peak_amplitude_time_log = [1, 5, 30, 60, 180, 300, 1440, 4320, 11520]
>>> car_power_1_time_evol_mean = np.zeros((9, 1600))
... aceta_power_1_time_evol_mean = np.zeros((9, 1600))
... aceph_power_1_time_evol_mean = np.zeros((9, 1600))
... car_aceta_power_1_time_evol_mean = np.zeros((9, 1600))
... car_aceph_power_1_time_evol_mean = np.zeros((9, 1600))
... aceta_aceph_power_1_time_evol_mean = np.zeros((9, 1600))
... car_aceta_aceph_power_1_time_evol_mean = np.zeros((9, 1600))
...
>>> power_labels = ['5 min', '30 min', '1 hour', '3 hours', '5 hours', '24 hours', '3 days (4320 hours)', '8 days (11520 hours)']
... car_power_1_time_evol = ['1_5min_1', '1_30min_1', '1_1hour_1', '1_3hours_1', '1_5hours_1', '1_24hours_1', '1_1458hours_1_1', '1_1530_1']
... aceta_power_1_time_evol = ['4_5min_1', '4_30min_1', '4_1hour_1', '4_3hours_1', '4_5hours_1', '4_24hours_1', '1_1513hours_1_1_1', '4_1540_1']
... aceph_power_1_time_evol = ['5_5min_1', '5_30min_1', '5_1hour_1', '5_3hours_1', '5_5hours_1', '5_24hours_1', '5_1538hours_1_1_1', '5_1545(inside)_1']
... car_aceta_power_1_time_evol = ['1+4_5min_1', '1+4_30min_1', '1+4_1hour_1', '1+4_3hours_1', '1+4_5hours_1',
...                                '1+4_24hours_1', '1+4_1550hour_1', '1+4_1605_1']
... car_aceph_power_1_time_evol = ['1+5_5min_1', '1+5_30min_1', '1+5_1hour_1', '1+5_3hours_1', '1+5_5hours_1',
...                                '1+5_24hours_1', '1+5_1632hour_1', '1+5_1635_1']
... aceta_aceph_power_1_time_evol = ['4+5_5min_1', '4+5_30min_1', '4+5_1hour_1', '4+5_3hours_1', '4+5_5hours_1',
...                                  '4+5_24hours_1', '1+5_1642hour_1_1', '4+5_1645_1']
... car_aceta_aceph_power_1_time_evol = ['1+4+5_5min_1', '1+4+5_30min_1', '1+4+5_1hour_1', '1+4+5_3hours_1',
...                                      '1+4+5_5hours_1', '1+4+5_24hours_1', '1+4+5_1650hour_1', '1+4+5_1650_1']
...
>>> peak_amplitude_time_log = [5, 30, 60, 180, 300, 1440, 4320, 11520]
>>> car_power_1_time_evol_mean = np.zeros((8, 1600))
... aceta_power_1_time_evol_mean = np.zeros((8, 1600))
... aceph_power_1_time_evol_mean = np.zeros((8, 1600))
... car_aceta_power_1_time_evol_mean = np.zeros((8, 1600))
... car_aceph_power_1_time_evol_mean = np.zeros((8, 1600))
... aceta_aceph_power_1_time_evol_mean = np.zeros((8, 1600))
... car_aceta_aceph_power_1_time_evol_mean = np.zeros((8, 1600))
...
>>> j = 0
... for item in car_power_1_time_evol:
...     car_power_1_time_evol_mean[j] = np.mean(carbendanzim_data[item], axis=0)
...     j += 1
... j = 0
... for item in aceta_power_1_time_evol:
...     aceta_power_1_time_evol_mean[j] = np.mean(acetamiprid_data[item], axis=0)
...     j += 1
... j = 0
... for item in aceph_power_1_time_evol:
...     aceph_power_1_time_evol_mean[j] = np.mean(acephate_data[item], axis=0)
...     j += 1
... j = 0
... for item in car_aceta_power_1_time_evol:
...     car_aceta_power_1_time_evol_mean[j] = np.mean(car_aceta_data[item], axis=0)
...     j += 1
... j = 0
... for item in car_aceph_power_1_time_evol:
...     car_aceph_power_1_time_evol_mean[j] = np.mean(car_aceph_data[item], axis=0)
...     j += 1
... j = 0
... for item in aceta_aceph_power_1_time_evol:
...     aceta_aceph_power_1_time_evol_mean[j] = np.mean(aceta_aceph_data[item], axis=0)
...     j += 1
... j = 0
... for item in car_aceta_aceph_power_1_time_evol:
...     car_aceta_aceph_power_1_time_evol_mean[j] = np.mean(car_aceta_aceph_data[item], axis=0)
...     j += 1
...
>>> carbendanzim_data = {**car_0810, **car_0811, **car_0813, **car_0818}
... acetamiprid_data = {**aceta_0810, **aceta_0811, **aceta_0813, **aceta_0818}
... acephate_data = {**aceph_0810, **aceph_0811, **aceph_0813, **aceph_0818}
... car_aceta_data = {**car_aceta_0810, **car_aceta_0811, **car_aceta_0813, **car_aceta_0818}
... car_aceph_data = {**car_aceph_0810, **car_aceph_0811, **car_aceph_0813, **car_aceph_0818}
... aceta_aceph_data = {**aceta_aceph_0810, **aceta_aceph_0811, **aceta_aceph_0813, **aceta_aceph_0818}
... car_aceta_aceph_data = {**car_aceta_aceph_0810, **car_aceta_aceph_0811, **car_aceta_aceph_0813, **car_aceta_aceph_0818}
...
>>> j = 0
... for item in car_power_1_time_evol:
...     car_power_1_time_evol_mean[j] = np.mean(carbendanzim_data[item], axis=0)
...     j += 1
... j = 0
... for item in aceta_power_1_time_evol:
...     aceta_power_1_time_evol_mean[j] = np.mean(acetamiprid_data[item], axis=0)
...     j += 1
... j = 0
... for item in aceph_power_1_time_evol:
...     aceph_power_1_time_evol_mean[j] = np.mean(acephate_data[item], axis=0)
...     j += 1
... j = 0
... for item in car_aceta_power_1_time_evol:
...     car_aceta_power_1_time_evol_mean[j] = np.mean(car_aceta_data[item], axis=0)
...     j += 1
... j = 0
... for item in car_aceph_power_1_time_evol:
...     car_aceph_power_1_time_evol_mean[j] = np.mean(car_aceph_data[item], axis=0)
...     j += 1
... j = 0
... for item in aceta_aceph_power_1_time_evol:
...     aceta_aceph_power_1_time_evol_mean[j] = np.mean(aceta_aceph_data[item], axis=0)
...     j += 1
... j = 0
... for item in car_aceta_aceph_power_1_time_evol:
...     car_aceta_aceph_power_1_time_evol_mean[j] = np.mean(car_aceta_aceph_data[item], axis=0)
...     j += 1
...
>>> for i in range(len(car_power_1_time_evol_mean)):
...     plt.plot(f_sup, car_power_1_time_evol_mean[i])
... plt.legend(power_labels)
... plt.show()
...
>>> for i in range(len(car_power_1_time_evol_mean)):
...     plt.plot(f_sup_0810, car_power_1_time_evol_mean[i])
... plt.legend(power_labels)
... plt.show()
...
>>> power_labels = ['5 min', '30 min', '1 hour', '3 hours', '5 hours', '24 hours', '3 days (4320 mins)', '8 days (11520 mins)']
>>> for i in range(len(car_power_1_time_evol_mean)):
...     plt.plot(f_sup_0810, car_power_1_time_evol_mean[i])
... plt.legend(power_labels)
... plt.show()
...
>>> for i in range(len(car_power_1_time_evol_mean)):
...     plt.plot(f_sup_0810, car_power_1_time_evol_mean[i])
... plt.legend(power_labels, fontsize=17)
... plt.show()
...
>>> for i in range(len(car_power_1_time_evol_mean)):
...     plt.plot(f_sup_0810, aceta_power_1_time_evol_mean[i])
... plt.legend(power_labels, fontsize=17)
... plt.show()
...
>>> car_power_1_time_evol_mean_smooth = np.zeros_like(car_power_1_time_evol_mean)
... aceta_power_1_time_evol_mean_smooth = np.zeros_like(aceta_power_1_time_evol_mean)
... aceph_power_1_time_evol_mean_smooth = np.zeros_like(aceph_power_1_time_evol_mean)
... car_aceta_power_1_time_evol_mean_smooth = np.zeros_like(car_aceta_power_1_time_evol_mean)
... car_aceph_power_1_time_evol_smooth = np.zeros_like(car_aceph_power_1_time_evol_mean)
... aceta_aceph_power_1_time_evol_smooth = np.zeros_like(aceta_aceph_power_1_time_evol_mean)
... car_aceta_aceph_power_1_time_evol_smooth = np.zeros_like(car_aceta_aceph_power_1_time_evol_mean)
...
>>> for i in range(car_power_1_time_evol_mean.shape[0]):
...     car_power_1_time_evol_mean_smooth[i] = savgol_filter(car_power_1_time_evol_mean[i], 5, 3)
...     aceta_power_1_time_evol_mean_smooth[i] = savgol_filter(aceta_power_1_time_evol_mean[i], 5, 3)
...     aceph_power_1_time_evol_mean_smooth[i] = savgol_filter(aceph_power_1_time_evol_mean[i], 5, 3)
...     car_aceta_power_1_time_evol_mean_smooth[i] = savgol_filter(car_aceta_power_1_time_evol_mean[i], 5, 3)
...     car_aceph_power_1_time_evol_smooth[i] = savgol_filter(car_aceph_power_1_time_evol_mean[i], 5, 3)
...     aceta_aceph_power_1_time_evol_smooth[i] = savgol_filter(aceta_aceph_power_1_time_evol_mean[i], 5, 3)
...     car_aceta_aceph_power_1_time_evol_smooth[i] = savgol_filter(car_aceta_aceph_power_1_time_evol_mean[i], 5, 3)
...
>>> matplotlib.rc('xtick', labelsize=20)
... matplotlib.rc('ytick', labelsize=20)
...
>>> for i in range(len(car_power_1_time_evol_mean)):
...     plt.plot(f_sup, aceta_power_1_time_evol_mean_smooth[i])
... plt.legend(power_labels, fontsize=17)
... plt.show()
...
>>> for i in range(len(car_power_1_time_evol_mean)):
...     plt.plot(f_sup_0810, aceta_power_1_time_evol_mean_smooth[i])
... plt.legend(power_labels, fontsize=17)
... plt.show()
...
>>> for i in range(len(car_power_1_time_evol_mean)):
...     plt.plot(f_sup_0810, aceph_power_1_time_evol_mean_smooth[i])
... plt.legend(power_labels, fontsize=17)
... plt.show()
...
>>> aceph_power_1_time_evol_mean_smooth = PD.remove_cosmic_rays(aceph_power_1_time_evol_mean_smooth, 5)
>>> for i in range(len(car_power_1_time_evol_mean)):
...     plt.plot(f_sup_0810, aceph_power_1_time_evol_mean_smooth[i])
... plt.legend(power_labels, fontsize=17)
... plt.show()
...
>>> for i in range(len(car_power_1_time_evol_mean)):
...     plt.plot(f_sup_0810, car_aceta_power_1_time_evol_mean_smooth[i])
... plt.legend(power_labels, fontsize=17)
... plt.show()
...
>>> for i in range(len(car_power_1_time_evol_mean)):
...     plt.plot(f_sup_0810, car_aceph_power_1_time_evol_smooth[i])
... plt.legend(power_labels, fontsize=17)
... plt.show()
...
>>> car_aceph_power_1_time_evol_smooth = PD.remove_cosmic_rays(car_aceph_power_1_time_evol_smooth, 5)
>>> for i in range(len(car_power_1_time_evol_mean)):
...     plt.plot(f_sup_0810, car_aceph_power_1_time_evol_smooth[i])
... plt.legend(power_labels, fontsize=17)
... plt.show()
...
>>> for i in range(car_power_1_time_evol_mean.shape[0]):
...     car_aceph_power_1_time_evol_smooth[i] = savgol_filter(car_aceph_power_1_time_evol_mean[i], 5, 3)
...
>>> for i in range(len(car_power_1_time_evol_mean)):
...     plt.plot(f_sup_0810, car_aceph_power_1_time_evol_smooth[i])
... plt.legend(power_labels, fontsize=17)
... plt.show()
...
>>> for i in range(len(car_power_1_time_evol_mean)):
...     plt.plot(f_sup_0810, aceta_aceph_power_1_time_evol_smooth[i])
... plt.legend(power_labels, fontsize=17)
... plt.show()
...
>>> aceta_aceph_power_1_time_evol_smooth = PD.remove_cosmic_rays(aceta_aceph_power_1_time_evol_smooth, 5)
>>> for i in range(len(car_power_1_time_evol_mean)):
...     plt.plot(f_sup_0810, aceta_aceph_power_1_time_evol_smooth[i])
... plt.legend(power_labels, fontsize=17)
... plt.show()
...
>>> for i in range(len(car_power_1_time_evol_mean)):
...     plt.plot(f_sup_0810, car_aceta_aceph_power_1_time_evol_smooth[i])
... plt.legend(power_labels, fontsize=17)
... plt.show()
...
>>> car_power_1_time_evol_mean_smooth_snr = car_power_1_time_evol_mean - car_power_1_time_evol_mean_smooth
... aceta_power_1_time_evol_mean_smooth_snr = aceta_power_1_time_evol_mean - aceta_power_1_time_evol_mean_smooth
... aceph_power_1_time_evol_mean_smooth_snr = aceph_power_1_time_evol_mean - aceph_power_1_time_evol_mean_smooth
... car_aceta_power_1_time_evol_mean_smooth_snr = car_aceta_power_1_time_evol_mean - car_aceta_power_1_time_evol_mean_smooth
... car_aceph_power_1_time_evol_smooth_snr = car_aceph_power_1_time_evol_mean - car_aceph_power_1_time_evol_mean_smooth
... aceta_aceph_power_1_time_evol_smooth_snr = aceta_aceph_power_1_time_evol_mean - aceta_aceph_power_1_time_evol_mean_smooth
... car_aceta_aceph_power_1_time_evol_smooth_snr = car_aceta_aceph_power_1_time_evol_mean - car_aceta_aceph_power_1_time_evol_mean_smooth
...
>>> car_power_1_time_evol_mean_smooth_snr = car_power_1_time_evol_mean - car_power_1_time_evol_mean_smooth
... aceta_power_1_time_evol_mean_smooth_snr = aceta_power_1_time_evol_mean - aceta_power_1_time_evol_mean_smooth
... aceph_power_1_time_evol_mean_smooth_snr = aceph_power_1_time_evol_mean - aceph_power_1_time_evol_mean_smooth
... car_aceta_power_1_time_evol_mean_smooth_snr = car_aceta_power_1_time_evol_mean - car_aceta_power_1_time_evol_mean_smooth
... car_aceph_power_1_time_evol_smooth_snr = car_aceph_power_1_time_evol_mean - car_aceph_power_1_time_evol_smooth
... aceta_aceph_power_1_time_evol_smooth_snr = aceta_aceph_power_1_time_evol_mean - aceta_aceph_power_1_time_evol_mean_smooth
... car_aceta_aceph_power_1_time_evol_smooth_snr = car_aceta_aceph_power_1_time_evol_mean - car_aceta_aceph_power_1_time_evol_mean_smooth
...
>>> car_power_1_time_evol_mean_smooth_snr = car_power_1_time_evol_mean - car_power_1_time_evol_mean_smooth
... aceta_power_1_time_evol_mean_smooth_snr = aceta_power_1_time_evol_mean - aceta_power_1_time_evol_mean_smooth
... aceph_power_1_time_evol_mean_smooth_snr = aceph_power_1_time_evol_mean - aceph_power_1_time_evol_mean_smooth
... car_aceta_power_1_time_evol_mean_smooth_snr = car_aceta_power_1_time_evol_mean - car_aceta_power_1_time_evol_mean_smooth
... car_aceph_power_1_time_evol_smooth_snr = car_aceph_power_1_time_evol_mean - car_aceph_power_1_time_evol_smooth
... aceta_aceph_power_1_time_evol_smooth_snr = aceta_aceph_power_1_time_evol_mean - aceta_aceph_power_1_time_evol_smooth
... car_aceta_aceph_power_1_time_evol_smooth_snr = car_aceta_aceph_power_1_time_evol_mean - car_aceta_aceph_power_1_time_evol_smooth
...
>>> for i in range(len(car_power_1_time_evol_mean)):
...     plt.plot(f_sup, car_power_1_time_evol_mean_smooth_snr[i])
... plt.legend(power_labels, fontsize=17)
... plt.show()
...
>>> for i in range(len(car_power_1_time_evol_mean)):
...     plt.plot(f_sup_0810, car_power_1_time_evol_mean_smooth_snr[i])
... plt.legend(power_labels, fontsize=17)
... plt.show()
...
>>> car_power_1_time_evol_mean_smooth_snr_fft = np.fft.fftshift(np.fft.fft(car_power_1_time_evol_mean_smooth_snr))
... aceta_power_1_time_evol_mean_smooth_snr_fft = np.fft.fftshift(np.fft.fft(aceta_power_1_time_evol_mean_smooth_snr))
... aceph_power_1_time_evol_mean_smooth_snr_fft = np.fft.fftshift(np.fft.fft(aceph_power_1_time_evol_mean_smooth_snr))
... car_aceta_power_1_time_evol_mean_smooth_snr_fft = np.fft.fftshift(np.fft.fft(car_aceta_power_1_time_evol_mean_smooth_snr))
... car_aceph_power_1_time_evol_smooth_snr_fft = np.fft.fftshift(np.fft.fft(car_aceph_power_1_time_evol_smooth_snr))
... aceta_aceph_power_1_time_evol_smooth_snr_fft = np.fft.fftshift(np.fft.fft(aceta_aceph_power_1_time_evol_smooth_snr))
... car_aceta_aceph_power_1_time_evol_smooth_snr_fft = np.fft.fftshift(np.fft.fft(car_aceta_aceph_power_1_time_evol_smooth_snr))
...
>>> for i in range(len(car_power_1_time_evol_mean)):
...     plt.plot(f_sup_0810, car_power_1_time_evol_mean_smooth_snr_fft[i])
... plt.legend(power_labels, fontsize=17)
... plt.show()
...
>>> _ = np.fft.fft(car_power_1_time_evol_mean_smooth_snr)
>>> for i in range(len(car_power_1_time_evol_mean)):
...     plt.plot(f_sup_0810, _[i])
... plt.legend(power_labels, fontsize=17)
... plt.show()
...
>>> for i in range(car_power_1_time_evol_mean.shape[0]):
...     car_power_1_time_evol_mean_smooth[i] = savgol_filter(car_power_1_time_evol_mean[i], 15, 3)
...     aceta_power_1_time_evol_mean_smooth[i] = savgol_filter(aceta_power_1_time_evol_mean[i], 15, 3)
...     aceph_power_1_time_evol_mean_smooth[i] = savgol_filter(aceph_power_1_time_evol_mean[i], 15, 3)
...     car_aceta_power_1_time_evol_mean_smooth[i] = savgol_filter(car_aceta_power_1_time_evol_mean[i], 15, 3)
...     car_aceph_power_1_time_evol_smooth[i] = savgol_filter(car_aceph_power_1_time_evol_mean[i], 15, 3)
...     aceta_aceph_power_1_time_evol_smooth[i] = savgol_filter(aceta_aceph_power_1_time_evol_mean[i], 15, 3)
...     car_aceta_aceph_power_1_time_evol_smooth[i] = savgol_filter(car_aceta_aceph_power_1_time_evol_mean[i], 15, 3)
...
>>> aceph_power_1_time_evol_mean_smooth = PD.remove_cosmic_rays(aceph_power_1_time_evol_mean_smooth, 5)
... aceta_aceph_power_1_time_evol_smooth = PD.remove_cosmic_rays(aceta_aceph_power_1_time_evol_smooth, 5)
... car_power_1_time_evol_mean_smooth_snr = car_power_1_time_evol_mean - car_power_1_time_evol_mean_smooth
... aceta_power_1_time_evol_mean_smooth_snr = aceta_power_1_time_evol_mean - aceta_power_1_time_evol_mean_smooth
... aceph_power_1_time_evol_mean_smooth_snr = aceph_power_1_time_evol_mean - aceph_power_1_time_evol_mean_smooth
... car_aceta_power_1_time_evol_mean_smooth_snr = car_aceta_power_1_time_evol_mean - car_aceta_power_1_time_evol_mean_smooth
... car_aceph_power_1_time_evol_smooth_snr = car_aceph_power_1_time_evol_mean - car_aceph_power_1_time_evol_smooth
... aceta_aceph_power_1_time_evol_smooth_snr = aceta_aceph_power_1_time_evol_mean - aceta_aceph_power_1_time_evol_smooth
... car_aceta_aceph_power_1_time_evol_smooth_snr = car_aceta_aceph_power_1_time_evol_mean - car_aceta_aceph_power_1_time_evol_smooth
...
>>> car_power_1_time_evol_mean_smooth_snr_fft = np.fft.fftshift(np.fft.fft(car_power_1_time_evol_mean_smooth_snr))
... aceta_power_1_time_evol_mean_smooth_snr_fft = np.fft.fftshift(np.fft.fft(aceta_power_1_time_evol_mean_smooth_snr))
... aceph_power_1_time_evol_mean_smooth_snr_fft = np.fft.fftshift(np.fft.fft(aceph_power_1_time_evol_mean_smooth_snr))
... car_aceta_power_1_time_evol_mean_smooth_snr_fft = np.fft.fftshift(np.fft.fft(car_aceta_power_1_time_evol_mean_smooth_snr))
... car_aceph_power_1_time_evol_smooth_snr_fft = np.fft.fftshift(np.fft.fft(car_aceph_power_1_time_evol_smooth_snr))
... aceta_aceph_power_1_time_evol_smooth_snr_fft = np.fft.fftshift(np.fft.fft(aceta_aceph_power_1_time_evol_smooth_snr))
... car_aceta_aceph_power_1_time_evol_smooth_snr_fft = np.fft.fftshift(np.fft.fft(car_aceta_aceph_power_1_time_evol_smooth_snr))
...
>>> for i in range(len(car_power_1_time_evol_mean)):
...     plt.plot(f_sup_0810, car_power_1_time_evol_mean_smooth[i])
... plt.legend(power_labels, fontsize=17)
... plt.show()
...
>>> def peaks_cwt_1(data_mean):
...     peaks = sci.find_peaks_cwt(data_mean, np.arange(0.001,10)) 
...     [ prominences , l_ips, r_ips]=sci.peak_prominences(x, peaks ,wlen=50)
...     results_eighty = sci.peak_widths(x, peaks, rel_height=0.8)
...     peak_width=results_eighty[0]
...     l_ips=(floor(results_eighty[2]))
...     r_ips=(ceil(results_eighty[3]))
...     l_ips.astype(int)
...     r_ips.astype(int)
...     return [peaks, prominences , l_ips, r_ips, peak_width]
...
>>> peaks, prom, l_ips, r_ips, peak_width = peaks_cwt_1(car_power_1_time_evol_mean_smooth[0])
>>> def peaks_cwt_1(data_mean):
...     peaks = find_peaks_cwt(data_mean, np.arange(0.001,10)) 
...     [ prominences , l_ips, r_ips]=sci.peak_prominences(x, peaks ,wlen=50)
...     results_eighty = sci.peak_widths(x, peaks, rel_height=0.8)
...     peak_width=results_eighty[0]
...     l_ips=(floor(results_eighty[2]))
...     r_ips=(ceil(results_eighty[3]))
...     l_ips.astype(int)
...     r_ips.astype(int)
...     return [peaks, prominences , l_ips, r_ips, peak_width]
...
>>> from scipy.signal import find_peaks_cwt
>>> peaks, prom, l_ips, r_ips, peak_width = peaks_cwt_1(car_power_1_time_evol_mean_smooth[0])
>>> def peaks_cwt_1(data_mean):
...     peaks = find_peaks_cwt(data_mean, np.arange(0.001,10)) 
...     [ prominences , l_ips, r_ips]=sci.peak_prominences(x, peaks ,wlen=50)
...     results_eighty = sci.peak_widths(x, peaks, rel_height=0.8)
...     peak_width=results_eighty[0]
...     l_ips=(floor(results_eighty[2]))
...     r_ips=(ceil(results_eighty[3]))
...     l_ips.astype(int)
...     r_ips.astype(int)
...     return [peaks, prominences , l_ips, r_ips, peak_width]
...
>>> peaks, prom, l_ips, r_ips, peak_width = peaks_cwt_1(car_power_1_time_evol_mean_smooth[0])
>>> from scipy.signal import peak_prominences, peak_widths
>>> def peaks_cwt_1(data_mean):
...     peaks = find_peaks_cwt(data_mean, np.arange(0.001,10)) 
...     [ prominences , l_ips, r_ips] = peak_prominences(x, peaks ,wlen=50)
...     results_eighty = peak_widths(x, peaks, rel_height=0.8)
...     peak_width=results_eighty[0]
...     l_ips=(floor(results_eighty[2]))
...     r_ips=(ceil(results_eighty[3]))
...     l_ips.astype(int)
...     r_ips.astype(int)
...     return [peaks, prominences , l_ips, r_ips, peak_width]
...
>>> peaks, prom, l_ips, r_ips, peak_width = peaks_cwt_1(car_power_1_time_evol_mean_smooth[0])
>>> _ = find_peaks_cwt(car_power_1_time_evol_mean_smooth[0], np.arange(0.001, 10))
>>> def peakdet(v, delta, x=None):
...     """
...     Converted from MATLAB script at http://billauer.co.il/peakdet.html
... 
...     Returns two arrays
... 
...     function [maxtab, mintab]=peakdet(v, delta, x)
...     %PEAKDET Detect peaks in a vector
...     %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
...     %        maxima and minima ("peaks") in the vector V.
...     %        MAXTAB and MINTAB consists of two columns. Column 1
...     %        contains indices in V, and column 2 the found values.
...     %      
...     %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
...     %        in MAXTAB and MINTAB are replaced with the corresponding
...     %        X-values.
...     %
...     %        A point is considered a maximum peak if it has the maximal
...     %        value, and was preceded (to the left) by a value lower by
...     %        DELTA.
... 
...     % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
...     % This function is released to the public domain; Any use is allowed.
... 
...     """
...     maxtab = []
...     mintab = []
... 
...     if x is None:
...         x = arange(len(v))
... 
...     v = asarray(v)
... 
...     if len(v) != len(x):
...         sys.exit('Input vectors v and x must have same length')
... 
...     if not isscalar(delta):
...         sys.exit('Input argument delta must be a scalar')
... 
...     if delta <= 0:
...         sys.exit('Input argument delta must be positive')
... 
...     mn, mx = Inf, -Inf
...     mnpos, mxpos = NaN, NaN
... 
...     lookformax = True
... 
...     for i in arange(len(v)):
...         this = v[i]
...         if this > mx:
...             mx = this
...             mxpos = x[i]
...         if this < mn:
...             mn = this
...             mnpos = x[i]
... 
...         if lookformax:
...             if this < mx - delta:
...                 maxtab.append((mxpos, mx))
...                 mn = this
...                 mnpos = x[i]
...                 lookformax = False
...         else:
...             if this > mn + delta:
...                 mintab.append((mnpos, mn))
...                 mx = this
...                 mxpos = x[i]
...                 lookformax = True
... 
...     return array(maxtab), array(mintab)
...
>>> maxtab, mintab = peakdet(car_power_1_time_evol_mean_smooth, .3)
... plot(f_sup_0810, car_power_1_time_evol_mean_smooth)
... # scatter(array(maxtab)[:,0], array(maxtab)[:,1], color='blue')
... scatter(array(mintab)[:,0], array(mintab)[:,1], color='red')
...
>>> from numpy import NaN, Inf, arange, isscalar, asarray, array
>>> def peakdet(v, delta, x=None):
...     """
...     Converted from MATLAB script at http://billauer.co.il/peakdet.html
... 
...     Returns two arrays
... 
...     function [maxtab, mintab]=peakdet(v, delta, x)
...     %PEAKDET Detect peaks in a vector
...     %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
...     %        maxima and minima ("peaks") in the vector V.
...     %        MAXTAB and MINTAB consists of two columns. Column 1
...     %        contains indices in V, and column 2 the found values.
...     %      
...     %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
...     %        in MAXTAB and MINTAB are replaced with the corresponding
...     %        X-values.
...     %
...     %        A point is considered a maximum peak if it has the maximal
...     %        value, and was preceded (to the left) by a value lower by
...     %        DELTA.
... 
...     % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
...     % This function is released to the public domain; Any use is allowed.
... 
...     """
...     maxtab = []
...     mintab = []
... 
...     if x is None:
...         x = arange(len(v))
... 
...     v = asarray(v)
... 
...     if len(v) != len(x):
...         sys.exit('Input vectors v and x must have same length')
... 
...     if not isscalar(delta):
...         sys.exit('Input argument delta must be a scalar')
... 
...     if delta <= 0:
...         sys.exit('Input argument delta must be positive')
... 
...     mn, mx = Inf, -Inf
...     mnpos, mxpos = NaN, NaN
... 
...     lookformax = True
... 
...     for i in arange(len(v)):
...         this = v[i]
...         if this > mx:
...             mx = this
...             mxpos = x[i]
...         if this < mn:
...             mn = this
...             mnpos = x[i]
... 
...         if lookformax:
...             if this < mx - delta:
...                 maxtab.append((mxpos, mx))
...                 mn = this
...                 mnpos = x[i]
...                 lookformax = False
...         else:
...             if this > mn + delta:
...                 mintab.append((mnpos, mn))
...                 mx = this
...                 mxpos = x[i]
...                 lookformax = True
... 
...     return array(maxtab), array(mintab)
...
>>> maxtab, mintab = peakdet(car_power_1_time_evol_mean_smooth, .3)
... plot(f_sup_0810, car_power_1_time_evol_mean_smooth)
... # scatter(array(maxtab)[:,0], array(maxtab)[:,1], color='blue')
... scatter(array(mintab)[:,0], array(mintab)[:,1], color='red')
...
>>> maxtab, mintab = peakdet(car_power_1_time_evol_mean_smooth, .3)
... plot(f_sup_0810, car_power_1_time_evol_mean_smooth)
... # scatter(array(maxtab)[:,0], array(maxtab)[:,1], color='blue')
... scatter(array(mintab)[:,0], array(mintab)[:,1], color='red')
...
>>> maxtab, mintab = peakdet(car_power_1_time_evol_mean_smooth[0], .3)
... plot(f_sup_0810, car_power_1_time_evol_mean_smooth)
... # scatter(array(maxtab)[:,0], array(maxtab)[:,1], color='blue')
... scatter(array(mintab)[:,0], array(mintab)[:,1], color='red')
...
>>> from matplotlib.pyplot import plot, scatter, show
... maxtab, mintab = peakdet(car_power_1_time_evol_mean_smooth[0], .3)
... plot(f_sup_0810, car_power_1_time_evol_mean_smooth)
... # scatter(array(maxtab)[:,0], array(maxtab)[:,1], color='blue')
... scatter(array(mintab)[:,0], array(mintab)[:,1], color='red')
...
>>> from matplotlib.pyplot import plot, scatter, show
... maxtab, mintab = peakdet(car_power_1_time_evol_mean_smooth[0], .3)
... plot(f_sup_0810, car_power_1_time_evol_mean_smooth[0])
... # scatter(array(maxtab)[:,0], array(maxtab)[:,1], color='blue')
... scatter(array(mintab)[:,0], array(mintab)[:,1], color='red')
... plt.show()
...
>>> maxtab, mintab = peakdet(car_power_1_time_evol_mean_smooth[0], .3)
... plt.plot(car_power_1_time_evol_mean_smooth[0])
... # scatter(array(maxtab)[:,0], array(maxtab)[:,1], color='blue')
... scatter(array(mintab)[:,0], array(mintab)[:,1], color='red')
... plt.show()
...
>>> maxtab, mintab = peakdet(car_power_1_time_evol_mean_smooth[0], .3)
... plt.plot(car_power_1_time_evol_mean_smooth[0])
... scatter(array(maxtab)[:,0], array(maxtab)[:,1], color='blue')
... scatter(array(mintab)[:,0], array(mintab)[:,1], color='red')
... plt.show()
...
>>> maxtab, mintab = peakdet(car_power_1_time_evol_mean_smooth[0], .8)
... plt.plot(car_power_1_time_evol_mean_smooth[0])
... scatter(array(maxtab)[:,0], array(maxtab)[:,1], color='blue')
... # scatter(array(mintab)[:,0], array(mintab)[:,1], color='red')
... plt.show()
...
>>> maxtab, mintab = peakdet(car_power_1_time_evol_mean_smooth[0], .1)
... plt.plot(car_power_1_time_evol_mean_smooth[0])
... scatter(array(maxtab)[:,0], array(maxtab)[:,1], color='blue')
... # scatter(array(mintab)[:,0], array(mintab)[:,1], color='red')
... plt.show()
...
>>> maxtab, mintab = peakdet(car_power_1_time_evol_mean_smooth[0], 10)
... plt.plot(car_power_1_time_evol_mean_smooth[0])
... scatter(array(maxtab)[:,0], array(maxtab)[:,1], color='blue')
... # scatter(array(mintab)[:,0], array(mintab)[:,1], color='red')
... plt.show()
...
>>> maxtab, mintab = peakdet(car_power_1_time_evol_mean_smooth[0], 15)
... plt.plot(car_power_1_time_evol_mean_smooth[0])
... scatter(array(maxtab)[:,0], array(maxtab)[:,1], color='blue')
... # scatter(array(mintab)[:,0], array(mintab)[:,1], color='red')
... plt.show()
...
>>> maxtab, mintab = peakdet(car_power_1_time_evol_mean_smooth[0], 9)
... plt.plot(car_power_1_time_evol_mean_smooth[0])
... scatter(array(maxtab)[:,0], array(maxtab)[:,1], color='blue')
... # scatter(array(mintab)[:,0], array(mintab)[:,1], color='red')
... plt.show()
...
>>> maxtab, mintab = peakdet(car_power_1_time_evol_mean_smooth[0], 20)
... plt.plot(car_power_1_time_evol_mean_smooth[0])
... scatter(array(maxtab)[:,0], array(maxtab)[:,1], color='blue')
... # scatter(array(mintab)[:,0], array(mintab)[:,1], color='red')
... plt.show()
...
>>> maxtab, mintab = peakdet(car_power_1_time_evol_mean_smooth[0], 20, f_sup_0810)
... plt.plot(f_sup_0810, car_power_1_time_evol_mean_smooth[0])
... scatter(array(maxtab)[:,0], array(maxtab)[:,1], color='blue')
... # scatter(array(mintab)[:,0], array(mintab)[:,1], color='red')
... plt.show()
...
>>> car_peaks = []
... for i in range(le(car_power_1_time_evol_mean_smooth)):
... car_peaks[i], _ = peakdet(car_power_1_time_evol_mean_smooth[i], 20, f_sup_0810)
...
>>> car_peaks = []
... for i in range(le(car_power_1_time_evol_mean_smooth)):
...     car_peaks[i], _ = peakdet(car_power_1_time_evol_mean_smooth[i], 20, f_sup_0810)
...
>>> car_peaks = []
... for i in range(len(car_power_1_time_evol_mean_smooth)):
...     car_peaks[i], _ = peakdet(car_power_1_time_evol_mean_smooth[i], 20, f_sup_0810)
...
>>> car_peaks = {}
... for i in range(len(car_power_1_time_evol_mean_smooth)):
...     car_peaks[i], _ = peakdet(car_power_1_time_evol_mean_smooth[i], 20, f_sup_0810)
...
>>> aceta_peaks = {}
... for i in range(len(car_power_1_time_evol_mean_smooth)):
...     aceta_peaks[i], _ = peakdet(aceta_aceph_power_1_time_evol_smooth[i], 20, f_sup_0810)
...
>>> aceph_peaks = {}
... for i in range(len(car_power_1_time_evol_mean_smooth)):
...     aceph_peaks[i], _ = peakdet(aceph_power_1_time_evol_mean_smooth[i], 20, f_sup_0810)
...
>>> car_aceta_peaks = {}
... for i in range(len(car_power_1_time_evol_mean_smooth)):
...     car_aceta_peaks[i], _ = peakdet(car_aceta_power_1_time_evol_mean_smooth[i], 20, f_sup_0810)
...
>>> car_aceph_peaks = {}
... for i in range(len(car_power_1_time_evol_mean_smooth)):
...     car_aceph_peaks[i], _ = peakdet(car_aceph_power_1_time_evol_smooth[i], 20, f_sup_0810)
...
>>> aceta_aceph_peaks = {}
... for i in range(len(car_power_1_time_evol_mean_smooth)):
...     aceta_aceph_peaks[i], _ = peakdet(aceta_aceph_power_1_time_evol_smooth[i], 20, f_sup_0810)
...
>>> car_aceta_aceph_peaks = {}
... for i in range(len(car_power_1_time_evol_mean_smooth)):
...     car_aceta_aceph_peaks[i], _ = peakdet(car_aceta_aceph_power_1_time_evol_smooth[i], 20, f_sup_0810)
...
>>> plt.plot(f_sup_0810, car_aceta_power_1_time_evol_mean_smooth[0])
... scatter(array(car_aceta_peaks)[0, :,0], array(car_aceta_peaks)[0, :,1], color='blue')
... # scatter(array(mintab)[:,0], array(mintab)[:,1], color='red')
... plt.show()
...
>>> plt.plot(f_sup_0810, car_aceta_power_1_time_evol_mean_smooth[0])
... scatter(array(car_aceta_peaks[0])[:,0], array(car_aceta_peaks[0])[:,1], color='blue')
... # scatter(array(mintab)[:,0], array(mintab)[:,1], color='red')
... plt.show()
...
>>> car_peaks_clipped = car_peaks[5:]
>>> car_peaks_clipped_pos, car_peaks_inten = array(car_peaks[0])[:,0], array(car_peaks[0])[:,1]
>>> plt.plot(f_sup_0810, car_aceta_power_1_time_evol_mean_smooth[0])
... scatter(array(car_aceta_peaks[0])[:,0], array(car_aceta_peaks[0])[:,1], color='blue')
... scatter(car_peaks_clipped_pos, car_peaks_inten, color='green')
... # scatter(array(mintab)[:,0], array(mintab)[:,1], color='red')
... plt.show()
...
>>> for i in range(len(car_peaks)):
...     plt.plot(peak_amplitude_time_log, array(car_peaks[i])[:,1], color='blue')
... plt.show()
...
>>> car_peaks_pos, car_peaks_inten = [], []
... for i in range(len(car_peaks)):
...     car_peaks_pos.append(array(car_peaks[i])[:,0]), car_peaks_inten.append(array(car_peaks[i])[:, 1])
... plt.show()
...
>>> for i in range(len(car_power_1_time_evol_mean_smooth)):
...     plt.figure(i)
...     plt.plot(f_sup_0810, car_power_1_time_evol_mean_smooth[i])
...     plt.scatter(car_peaks_pos[i], car_peaks_inten[i])
...     plt.show()
...
>>> car_peaks_pos = [308, 669, 890, 1104, 1147, 1247, 1345, 1400, 1467]
>>> np.where(np.floor(f_sup_0810) == car_peaks_pos)
(array([], dtype=int64),)
>>> np.floor(f_sup_0810)
array([ -23.,  -21.,  -20., ..., 2466., 2467., 2469.])
>>> np.floor(f_sup_0810)[car_peaks_pos]
array([ 457., 1020., 1364., 1698., 1765., 1920., 2073., 2159., 2263.])
>>> car_peak_index = np.floor(f_sup_0810)[car_peaks_pos]
>>> plt.plot(f_sup_0810, car_power_1_time_evol_mean_smooth)
... plt.plot(f_sup_0810[car_peak_index], car_power_1_time_evol_mean_smooth[0, car_peak_index], 'r*')
... plt.show()
...
>>> plt.plot(f_sup_0810, car_power_1_time_evol_mean_smooth)
... plt.plot(f_sup_0810[car_peak_index], car_power_1_time_evol_mean_smooth[0][car_peak_index], 'r*')
... plt.show()
...
>>> plt.plot(f_sup_0810, car_power_1_time_evol_mean_smooth[0])
... plt.plot(f_sup_0810[car_peak_index], car_power_1_time_evol_mean_smooth[0][car_peak_index], 'r*')
... plt.show()
...
>>> plt.plot(f_sup_0810, car_power_1_time_evol_mean_smooth[0])
... plt.plot(f_sup_0810[car_peak_index], car_power_1_time_evol_mean_smooth[0, car_peak_index], 'r*')
... plt.show()
...
>>> plt.plot(f_sup_0810, car_power_1_time_evol_mean_smooth[0])
... plt.plot(f_sup_0810[car_peak_index], car_power_1_time_evol_mean_smooth[0][car_peak_index], 'r*')
... plt.show()
...
>>> car_peak_index = np.array(np.floor(f_sup_0810)[car_peaks_pos], dtype=np.int32)
>>> plt.plot(f_sup_0810, car_power_1_time_evol_mean_smooth[0])
... plt.plot(f_sup_0810[car_peak_index], car_power_1_time_evol_mean_smooth[0][car_peak_index], 'r*')
... plt.show()
...
>>> np.where(np.floow(f_sup_0810, dtype=np.int32) == car_peak_index)
>>> np.where(np.floor(f_sup_0810, dtype=np.int32) == car_peak_index)
>>> np.where(f_sup_0810 == car_peak_index)
(array([], dtype=int64),)
>>> np.floor(f_sup_0810)
array([ -23.,  -21.,  -20., ..., 2466., 2467., 2469.])
>>> np.floor(f_sup_0810, dtype=np.int32)
>>> np.floor(f_sup_0810, dtype='np.int32')
>>> np.floor(f_sup_0810, dtype=int)
>>> np.floor(f_sup_0810, dtype=np.int64)
>>> np.floor(f_sup_0810, dtype='int')
>>> car_peak_index = np.empty_like(car_peak_index, dtype=np.int64)
>>> car_peak_index = np.where(np.floor(f_sup_0810) == car_peak_pos)
>>> car_peak_index = np.where(np.floor(f_sup_0810) == car_peaks_pos)
>>> car_peak_index = np.floor(f_sup_0810)
>>> np.where(car_peak_index == car_peaks_pos)
(array([], dtype=int64),)
>>> np.where(car_peak_index == car_peaks_pos)
(array([], dtype=int64),)
>>> car_peak_index[car_peaks_pos]
array([ 457., 1020., 1364., 1698., 1765., 1920., 2073., 2159., 2263.])
>>> plt.plot(f_sup_0810, car_power_1_time_evol_mean_smooth[0])
... plt.plot(f_sup_0810[car_peak_pos], car_power_1_time_evol_mean_smooth[0][car_peaks_pos], 'r*')
... plt.show()
...
>>> plt.plot(f_sup_0810, car_power_1_time_evol_mean_smooth[0])
... plt.plot(f_sup_0810[car_peaks_pos], car_power_1_time_evol_mean_smooth[0][car_peaks_pos], 'r*')
... plt.show()
...
>>> np.where(car_peaks_pos == f_sup_0810)
(array([], dtype=int64),)
>>> np.where(car_peaks_pos == np.floor(f_sup_0810, dtype=np.int64))
>>> np.where(car_peaks_pos == math.floor(f_sup_0810))
>>> import math
>>> np.where(car_peaks_pos == math.floor(f_sup_0810))
>>> f_sup_int = f_sup_0810.astype(int)
>>> np.where(car_peaks_pos == f_sup_int)
(array([], dtype=int64),)
>>> f_sup_int
array([ -22,  -20,  -19, ..., 2466, 2467, 2469])
>>> f_sup_int = np.floor(f_sup_0810).astype(int)
>>> f_sup_int
array([ -23,  -21,  -20, ..., 2466, 2467, 2469])
>>> np.where(car_peaks_pos == f_sup_int)
(array([], dtype=int64),)
>>> f_sup_int[car_peaks_pos]
array([ 457, 1020, 1364, 1698, 1765, 1920, 2073, 2159, 2263])
>>> np.argwhere(f_sup_int == car_peaks_pos)
array([], shape=(0, 0), dtype=int64)
>>> np.argwhere(f_sup_int == array(car_peaks_pos, dtype=np.int64))
array([], shape=(0, 0), dtype=int64)
>>> f_sup_int[f_sup_int == car_peaks_pos]
array([], shape=(0, 1600), dtype=int64)
>>> car_peaks_pos = np.array(car_peaks_pos, dtype=np.int32)
>>> car_peaks_pos
array([ 308,  669,  890, 1104, 1147, 1247, 1345, 1400, 1467], dtype=int32)
>>> f_sup_int[f_sup_int == car_peaks_pos]
array([], shape=(0, 1600), dtype=int64)
>>> f_sup_int == car_peaks_pos
False
>>> car_peaks = np.equal(f_sup_int, car_peaks_pos)
>>> car_peaks = np.argwhere(f_sup_int == car_peaks_pos)
>>> car_peaks = []
... for i in range(len(car_peaks_pos)):
...     car_peaks.append(np.where(f_sup_int == car_peaks_pos[i]))
...
>>> car_peaks
[(array([212]),),
 (array([444]),),
 (array([586]),),
 (array([723]),),
 (array([751]),),
 (array([815]),),
 (array([878]),),
 (array([913]),),
 (array([956]),)]
>>> car_peaks = []
... for i in range(len(car_peaks_pos)):
...     car_peaks.append(np.where(f_sup_int == car_peaks_pos[i])[0])
...
>>> car_peaks
[array([212]),
 array([444]),
 array([586]),
 array([723]),
 array([751]),
 array([815]),
 array([878]),
 array([913]),
 array([956])]
>>> car_peaks = np.array(car_peaks)
>>> car_peaks
array([[212],
       [444],
       [586],
       [723],
       [751],
       [815],
       [878],
       [913],
       [956]])
>>> car_peaks.flatten()
array([212, 444, 586, 723, 751, 815, 878, 913, 956])
>>> car_peaks
array([[212],
       [444],
       [586],
       [723],
       [751],
       [815],
       [878],
       [913],
       [956]])
>>> car_peaks = car_peaks.flatten()
>>> car_peaks
array([212, 444, 586, 723, 751, 815, 878, 913, 956])
>>> plt.plot(f_sup_0810, car_power_1_time_evol_mean_smooth[0])
... plt.plot(f_sup_0810[car_peaks], car_power_1_time_evol_mean_smooth[0, car_peaks])
... plt.show()
...
>>> plt.plot(f_sup_0810, car_power_1_time_evol_mean_smooth[0])
... plt.plot(f_sup_0810[car_peaks], car_power_1_time_evol_mean_smooth[0, car_peaks], 'r*')
... plt.show()
...
>>> %history -f console_logs_2021_08_26.py
>>> for i in range(len(car_power_1_time_evol_mean_smooth)):
...     for peak in car_peaks:
...         plt.plot(peak_amplitude_time_log, car_power_1_time_evol_mean_smooth[i, peak], label='Peak #{}'.format(i+1))
... plt.plot()
...
>>> i=0
... for peak in car_peaks:
...     plt.plot(peak_amplitude_time_log, car_power_1_time_evol_mean_smooth[:, peak], label='Peak #{}'.format(i+1))
...     i+=1
... plt.plot()
...
[]
>>> i=0
... for peak in car_peaks:
...     plt.plot(peak_amplitude_time_log, car_power_1_time_evol_mean_smooth[:, peak], label='Peak #{}'.format(i+1))
...     i+=1
... plt.legend()
... plt.plot()
...
[]
>>> i=0
... for peak in car_peaks:
...     print(car_power_1_time_evol_mean_smooth[:, peak])
...     i+=1
...
>>> i=0
... for peak in car_peaks:
...     plt.plot(peak_amplitude_time_log, car_power_1_time_evol_mean_smooth[:, peak], label='Peak #{}'.format(i+1))
...     i+=1
... plt.legend(fontsize=15)
... plt.xlabel('Measuring time (mins)', fontsize=20)
... plt.ylabel('Intensity', fontsize=20)
... plt.title('Carbendazim peaks time evolution at HWP 42 (2.91 mW)', fontsize=17)
... plt.xticks(peak_amplitude_time_log)
... plt.yscale("log")
... plt.show()
...
>>> i=0
... for peak in car_peaks:
...     plt.plot(peak_amplitude_time_log, car_power_1_time_evol_mean_smooth[:, peak], label='Peak #{}'.format(i+1))
...     i+=1
... plt.legend(fontsize=15)
... plt.xlabel('Measuring time (mins)', fontsize=20)
... plt.ylabel('Intensity', fontsize=20)
... plt.title('Carbendazim peaks time evolution at HWP 42 (2.91 mW)', fontsize=17)
... plt.xticks(peak_amplitude_time_log)
... plt.xscale("log")
... plt.show()
...
>>> i=0
... for peak in car_peaks:
...     plt.plot(peak_amplitude_time_log, car_power_1_time_evol_mean_smooth[:, peak], label='Peak #{}'.format(i+1))
...     i+=1
... plt.legend(fontsize=15)
... plt.xlabel('Measuring time (mins)', fontsize=20)
... plt.ylabel('Intensity', fontsize=20)
... plt.title('Carbendazim peaks time evolution at HWP 42 (2.91 mW)', fontsize=17)
... plt.xticks(peak_amplitude_time_log)
... plt.xscale("log")
... plt.show()
...
>>> i=0
... for peak in car_peaks:
...     plt.plot(peak_amplitude_time_log, car_power_1_time_evol_mean_smooth[:, peak], label='Peak #{}'.format(i+1))
...     i+=1
... plt.legend(fontsize=15)
... plt.xlabel('Measuring time (mins)', fontsize=20)
... plt.ylabel('Intensity', fontsize=20)
... plt.title('Carbendazim peaks time evolution at HWP 42 (2.91 mW)', fontsize=17)
... plt.set_xticks(peak_amplitude_time_log)
... plt.xscale("log")
... plt.show()
...
>>> i=0
... fig1, ax1 = plt.subplots()
... for peak in car_peaks:
...     ax1.plot(peak_amplitude_time_log, car_power_1_time_evol_mean_smooth[:, peak], label='Peak #{}'.format(i+1))
...     i+=1
... ax1.legend(fontsize=15)
... ax1.xlabel('Measuring time (mins)', fontsize=20)
... ax1.ylabel('Intensity', fontsize=20)
... ax1.title('Carbendazim peaks time evolution at HWP 42 (2.91 mW)', fontsize=17)
... ax1.set_xticks(peak_amplitude_time_log)
... ax1.xscale("log")
... plt.show()
...
>>> i=0
... fig1, ax1 = plt.subplots()
... for peak in car_peaks:
...     ax1.plot(peak_amplitude_time_log, car_power_1_time_evol_mean_smooth[:, peak], label='Peak #{}'.format(i+1))
...     i+=1
... ax1.legend(fontsize=15)
... ax1.set_xlabel('Measuring time (mins)', fontsize=20)
... ax1.set_ylabel('Intensity', fontsize=20)
... ax1.title('Carbendazim peaks time evolution at HWP 42 (2.91 mW)', fontsize=17)
... ax1.set_xticks(peak_amplitude_time_log)
... ax1.set_xscale("log")
... plt.show()
...
>>> i=0
... fig1, ax1 = plt.subplots()
... for peak in car_peaks:
...     ax1.plot(peak_amplitude_time_log, car_power_1_time_evol_mean_smooth[:, peak], label='Peak #{}'.format(i+1))
...     i+=1
... plt.legend(fontsize=15)
... plt.xlabel('Measuring time (mins)', fontsize=20)
... ax1.set_ylabel('Intensity', fontsize=20)
... plt.title('Carbendazim peaks time evolution at HWP 42 (2.91 mW)', fontsize=17)
... ax1.set_xticks(peak_amplitude_time_log)
... plt.set_xscale("log")
... plt.show()
...
>>> i=0
... for peak in car_peaks:
...     plt.plot(peak_amplitude_time_log, car_power_1_time_evol_mean_smooth[:, peak], label='Peak #{}'.format(i+1))
...     i+=1
... plt.legend(fontsize=15)
... plt.xlabel('Measuring time (mins)', fontsize=20)
... plt.set_ylabel('Intensity', fontsize=20)
... plt.title('Carbendazim peaks time evolution at HWP 42 (2.91 mW)', fontsize=17)
... ax1.set_xticks(peak_amplitude_time_log)
... plt.set_xscale("log")
... plt.show()
...
>>> i=0
... for peak in car_peaks:
...     plt.plot(peak_amplitude_time_log, car_power_1_time_evol_mean_smooth[:, peak], label='Peak #{}'.format(i+1))
...     i+=1
... plt.legend(fontsize=15)
... plt.xlabel('Measuring time (mins)', fontsize=20)
... plt.ylabel('Intensity', fontsize=20)
... plt.title('Carbendazim peaks time evolution at HWP 42 (2.91 mW)', fontsize=17)
... ax1.set_xticks(peak_amplitude_time_log)
... plt.set_xscale("log")
... plt.show()
...
>>> i=0
... for peak in car_peaks:
...     plt.plot(peak_amplitude_time_log, car_power_1_time_evol_mean_smooth[:, peak], label='Peak #{}'.format(i+1))
...     i+=1
... plt.legend(fontsize=15)
... plt.xlabel('Measuring time (mins)', fontsize=20)
... plt.ylabel('Intensity', fontsize=20)
... plt.title('Carbendazim peaks time evolution at HWP 42 (2.91 mW)', fontsize=17)
... ax1.set_xticks(peak_amplitude_time_log)
... plt.xscale("log")
... plt.show()
...
>>> i=0
... for peak in car_peaks:
...     plt.plot(peak_amplitude_time_log, car_power_1_time_evol_mean_smooth[:, peak], label='Peak #{}'.format(i+1))
...     i+=1
... plt.legend(fontsize=15)
... plt.xlabel('Measuring time (mins)', fontsize=20)
... plt.ylabel('Intensity', fontsize=20)
... plt.title('Carbendazim peaks time evolution at HWP 42 (2.91 mW)', fontsize=17)
... ax1.xticks(peak_amplitude_time_log)
... plt.xscale("log")
... plt.show()
...
>>> i=0
... for peak in car_peaks:
...     plt.plot(peak_amplitude_time_log, car_power_1_time_evol_mean_smooth[:, peak], label='Peak #{}'.format(i+1))
...     i+=1
... plt.legend(fontsize=15)
... plt.xlabel('Measuring time (mins)', fontsize=20)
... plt.ylabel('Intensity', fontsize=20)
... plt.title('Carbendazim peaks time evolution at HWP 42 (2.91 mW)', fontsize=17)
... plt.xticks(peak_amplitude_time_log)
... plt.xscale("log")
... plt.show()
...
>>> i=0
... fig, ax = plt.subplots()
... for peak in car_peaks:
...     ax.plot(peak_amplitude_time_log, car_power_1_time_evol_mean_smooth[:, peak], label='Peak #{}'.format(i+1))
...     i+=1
... ax.legend(fontsize=15)
... ax.set_xlabel('Measuring time (mins)', fontsize=20)
... ax.set_ylabel('Intensity', fontsize=20)
... ax.title('Carbendazim peaks time evolution at HWP 42 (2.91 mW)', fontsize=17)
... ax.set_xticks(peak_amplitude_time_log)
... ax.xscale("log")
... ax.xaxis.set_minor_locator(ticker.FixedLocator(peak_amplitude_time_log))
... plt.show()
...
>>> i=0
... fig, ax = plt.subplots()
... for peak in car_peaks:
...     ax.plot(peak_amplitude_time_log, car_power_1_time_evol_mean_smooth[:, peak], label='Peak #{}'.format(i+1))
...     i+=1
... ax.legend(fontsize=15)
... ax.set_xlabel('Measuring time (mins)', fontsize=20)
... ax.set_ylabel('Intensity', fontsize=20)
... plt.title('Carbendazim peaks time evolution at HWP 42 (2.91 mW)', fontsize=17)
... ax.set_xticks(peak_amplitude_time_log)
... ax.xscale("log")
... ax.xaxis.set_minor_locator(ticker.FixedLocator(peak_amplitude_time_log))
... plt.show()
...
>>> i=0
... fig, ax = plt.subplots()
... for peak in car_peaks:
...     ax.plot(peak_amplitude_time_log, car_power_1_time_evol_mean_smooth[:, peak], label='Peak #{}'.format(i+1))
...     i+=1
... ax.legend(fontsize=15)
... ax.set_xlabel('Measuring time (mins)', fontsize=20)
... ax.set_ylabel('Intensity', fontsize=20)
... plt.title('Carbendazim peaks time evolution at HWP 42 (2.91 mW)', fontsize=17)
... ax.set_xticks(peak_amplitude_time_log)
... ax.set_xscale("log")
... ax.xaxis.set_minor_locator(ticker.FixedLocator(peak_amplitude_time_log))
... plt.show()
...
>>> from matplotlib import ticker
>>> i=0
... fig, ax = plt.subplots()
... for peak in car_peaks:
...     ax.plot(peak_amplitude_time_log, car_power_1_time_evol_mean_smooth[:, peak], label='Peak #{}'.format(i+1))
...     i+=1
... ax.legend(fontsize=15)
... ax.set_xlabel('Measuring time (mins)', fontsize=20)
... ax.set_ylabel('Intensity', fontsize=20)
... plt.title('Carbendazim peaks time evolution at HWP 42 (2.91 mW)', fontsize=17)
... ax.set_xticks(peak_amplitude_time_log)
... ax.set_xscale("log")
... ax.xaxis.set_minor_locator(ticker.FixedLocator(peak_amplitude_time_log))
... plt.show()
...
>>> i=0
... fig, ax = plt.subplots()
... for peak in car_peaks:
...     ax.plot(peak_amplitude_time_log, car_power_1_time_evol_mean_smooth[:, peak], label='Peak #{}'.format(i+1))
...     i+=1
... ax.legend(fontsize=15)
... ax.set_xlabel('Measuring time (mins)', fontsize=20)
... ax.set_ylabel('Intensity', fontsize=20)
... plt.title('Carbendazim peaks time evolution at HWP 42 (2.91 mW)', fontsize=17)
... ax.set_xticks(peak_amplitude_time_log)
... ax.set_xscale("log")
... # ax.xaxis.set_minor_locator(ticker.FixedLocator(peak_amplitude_time_log))
... plt.show()
...
>>> from matplotlib.ticker import ScalarFormatter
>>> i=0
... fig, ax = plt.subplots()
... for peak in car_peaks:
...     ax.plot(peak_amplitude_time_log, car_power_1_time_evol_mean_smooth[:, peak], label='Peak #{}'.format(i+1))
...     i+=1
... ax.legend(fontsize=15)
... ax.set_xlabel('Measuring time (mins)', fontsize=20)
... ax.set_ylabel('Intensity', fontsize=20)
... plt.title('Carbendazim peaks time evolution at HWP 42 (2.91 mW)', fontsize=17)
... ax.set_xticks(peak_amplitude_time_log)
... ax.set_xscale("log")
... ax.set_major_formatter(ScalarFormatter())
... # ax.xaxis.set_minor_locator(ticker.FixedLocator(peak_amplitude_time_log))
... plt.show()
...
>>> i=0
... fig, ax = plt.subplots()
... for peak in car_peaks:
...     ax.plot(peak_amplitude_time_log, car_power_1_time_evol_mean_smooth[:, peak], label='Peak #{}'.format(i+1))
...     i+=1
... ax.legend(fontsize=15)
... ax.set_xlabel('Measuring time (mins)', fontsize=20)
... ax.set_ylabel('Intensity', fontsize=20)
... plt.title('Carbendazim peaks time evolution at HWP 42 (2.91 mW)', fontsize=17)
... ax.set_xticks(peak_amplitude_time_log)
... ax.set_xscale("log")
... ax.set_major_locator(IndexLocator(peak_amplitude_time_log))
... # ax.xaxis.set_minor_locator(ticker.FixedLocator(peak_amplitude_time_log))
... plt.show()
...
>>> i=0
... fig, ax = plt.subplots()
... for peak in car_peaks:
...     ax.plot(peak_amplitude_time_log, car_power_1_time_evol_mean_smooth[:, peak], label='Peak #{}'.format(i+1))
...     i+=1
... ax.legend(fontsize=15)
... ax.set_xlabel('Measuring time (mins)', fontsize=20)
... ax.set_ylabel('Intensity', fontsize=20)
... plt.title('Carbendazim peaks time evolution at HWP 42 (2.91 mW)', fontsize=17)
... ax.set_xticks(peak_amplitude_time_log)
... ax.set_xscale("log")
... ax.xaxis.set_minor_locator(ticker.FixedLocator(peak_amplitude_time_log))
... ax.xaxis.set_major_locator(ticker.NullLocator())
... ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
... plt.show()
...
>>> plt.savefig('Carbendazim_peak_evol.png', dpi=500)
>>> aceta_peaks_pos, aceta_peaks_inten = [], []
... for i in range(len(aceta_peaks)):
...     aceta_peaks_pos.append(array(aceta_peaks[i])[:,0]), aceta_peaks_inten.append(array(aceta_peaks[i])[:, 1])
...
>>> for i in range(len(car_power_1_time_evol_mean_smooth)):
...     plt.figure(i)
...     plt.plot(f_sup_0810, aceta_power_1_time_evol_mean_smooth[i])
...     plt.scatter(aceta_peaks_pos[i], aceta_peaks_inten[i])
...     plt.show()
...
>>> acet_peak_wavenum = [538, 1376, 2113]
>>> aceph_peaks_pos, aceph_peaks_inten = [], []
... for i in range(len(aceph_peaks)):
...     aceph_peaks_pos.append(array(aceph_peaks[i])[:,0]), aceph_peaks_inten.append(array(aceph_peaks[i])[:, 1])
...
>>> plt.plot(f_sup_0810, aceph_power_1_time_evol_mean_smooth[-1])
[<matplotlib.lines.Line2D at 0x7f057c1db6d0>]
>>> for i in range(car_power_1_time_evol_mean.shape[0]):
...     aceph_power_1_time_evol_mean_smooth[i] = savgol_filter(aceph_power_1_time_evol_mean[i], 15, 3)
...
>>> car_power_1_time_evol_mean_smooth_snr = car_power_1_time_evol_mean - car_power_1_time_evol_mean_smooth
... aceta_power_1_time_evol_mean_smooth_snr = aceta_power_1_time_evol_mean - aceta_power_1_time_evol_mean_smooth
... aceph_power_1_time_evol_mean_smooth_snr = aceph_power_1_time_evol_mean - aceph_power_1_time_evol_mean_smooth
... car_aceta_power_1_time_evol_mean_smooth_snr = car_aceta_power_1_time_evol_mean - car_aceta_power_1_time_evol_mean_smooth
... car_aceph_power_1_time_evol_smooth_snr = car_aceph_power_1_time_evol_mean - car_aceph_power_1_time_evol_smooth
... aceta_aceph_power_1_time_evol_smooth_snr = aceta_aceph_power_1_time_evol_mean - aceta_aceph_power_1_time_evol_smooth
... car_aceta_aceph_power_1_time_evol_smooth_snr = car_aceta_aceph_power_1_time_evol_mean - car_aceta_aceph_power_1_time_evol_smooth
...
>>> car_power_1_time_evol_mean_smooth_snr_fft = np.fft.fftshift(np.fft.fft(car_power_1_time_evol_mean_smooth_snr))
... aceta_power_1_time_evol_mean_smooth_snr_fft = np.fft.fftshift(np.fft.fft(aceta_power_1_time_evol_mean_smooth_snr))
... aceph_power_1_time_evol_mean_smooth_snr_fft = np.fft.fftshift(np.fft.fft(aceph_power_1_time_evol_mean_smooth_snr))
... car_aceta_power_1_time_evol_mean_smooth_snr_fft = np.fft.fftshift(np.fft.fft(car_aceta_power_1_time_evol_mean_smooth_snr))
... car_aceph_power_1_time_evol_smooth_snr_fft = np.fft.fftshift(np.fft.fft(car_aceph_power_1_time_evol_smooth_snr))
... aceta_aceph_power_1_time_evol_smooth_snr_fft = np.fft.fftshift(np.fft.fft(aceta_aceph_power_1_time_evol_smooth_snr))
... car_aceta_aceph_power_1_time_evol_smooth_snr_fft = np.fft.fftshift(np.fft.fft(car_aceta_aceph_power_1_time_evol_smooth_snr))
...
>>> plt.plot(f_sup_0810, aceph_power_1_time_evol_mean_smooth[-1])
[<matplotlib.lines.Line2D at 0x7f057c82c310>]
>>> plt.plot(f_sup_0810, aceta_power_1_time_evol_mean_smooth[-1])
[<matplotlib.lines.Line2D at 0x7f057c3c3d00>]
>>> plt.plot(f_sup_0810, car_aceta_power_1_time_evol_mean_smooth[-1])
[<matplotlib.lines.Line2D at 0x7f057cd66310>]
>>> plt.plot(f_sup_0810, car_aceph_power_1_time_evol_smooth[-1])
[<matplotlib.lines.Line2D at 0x7f057c1847c0>]
>>> plt.plot(f_sup_0810, aceta_aceph_power_1_time_evol_smooth[-1])
[<matplotlib.lines.Line2D at 0x7f057c392f10>]
>>> for i in range(car_power_1_time_evol_mean.shape[0]):
...     aceta_aceph_power_1_time_evol_smooth[i] = savgol_filter(aceta_aceph_power_1_time_evol_mean[i], 15, 3)
...
>>> car_power_1_time_evol_mean_smooth_snr = car_power_1_time_evol_mean - car_power_1_time_evol_mean_smooth
... aceta_power_1_time_evol_mean_smooth_snr = aceta_power_1_time_evol_mean - aceta_power_1_time_evol_mean_smooth
... aceph_power_1_time_evol_mean_smooth_snr = aceph_power_1_time_evol_mean - aceph_power_1_time_evol_mean_smooth
... car_aceta_power_1_time_evol_mean_smooth_snr = car_aceta_power_1_time_evol_mean - car_aceta_power_1_time_evol_mean_smooth
... car_aceph_power_1_time_evol_smooth_snr = car_aceph_power_1_time_evol_mean - car_aceph_power_1_time_evol_smooth
... aceta_aceph_power_1_time_evol_smooth_snr = aceta_aceph_power_1_time_evol_mean - aceta_aceph_power_1_time_evol_smooth
... car_aceta_aceph_power_1_time_evol_smooth_snr = car_aceta_aceph_power_1_time_evol_mean - car_aceta_aceph_power_1_time_evol_smooth
...     
... car_power_1_time_evol_mean_smooth_snr_fft = np.fft.fftshift(np.fft.fft(car_power_1_time_evol_mean_smooth_snr))
... aceta_power_1_time_evol_mean_smooth_snr_fft = np.fft.fftshift(np.fft.fft(aceta_power_1_time_evol_mean_smooth_snr))
... aceph_power_1_time_evol_mean_smooth_snr_fft = np.fft.fftshift(np.fft.fft(aceph_power_1_time_evol_mean_smooth_snr))
... car_aceta_power_1_time_evol_mean_smooth_snr_fft = np.fft.fftshift(np.fft.fft(car_aceta_power_1_time_evol_mean_smooth_snr))
... car_aceph_power_1_time_evol_smooth_snr_fft = np.fft.fftshift(np.fft.fft(car_aceph_power_1_time_evol_smooth_snr))
... aceta_aceph_power_1_time_evol_smooth_snr_fft = np.fft.fftshift(np.fft.fft(aceta_aceph_power_1_time_evol_smooth_snr))
... car_aceta_aceph_power_1_time_evol_smooth_snr_fft = np.fft.fftshift(np.fft.fft(car_aceta_aceph_power_1_time_evol_smooth_snr))
...
>>> plt.plot(f_sup_0810, aceta_aceph_power_1_time_evol_smooth[-1])
[<matplotlib.lines.Line2D at 0x7f057c8a45b0>]
>>> plt.plot(f_sup_0810, car_aceta_aceph_power_1_time_evol_smooth[-1])
[<matplotlib.lines.Line2D at 0x7f057c4a0d00>]
>>> %history -f -p -o console_logs_v2_2021_08_26.py
>>> %history -p -o -f console_logs_v2_2021_08_26.py
