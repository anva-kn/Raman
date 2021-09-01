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
car_power_1_time_evol = ['1_5min_1', '1_30min_1', '1_1hour_1', '1_3hours_1', '1_5hours_1', '1_24hours_1', '1_1458hours_1_1', '1_1530_1']
aceta_power_1_time_evol = ['4_5min_1', '4_30min_1', '4_1hour_1', '4_3hours_1', '4_5hours_1', '4_24hours_1', '1_1513hours_1_1_1', '4_1540_1']
aceph_power_1_time_evol = ['5_5min_1', '5_30min_1', '5_1hour_1', '5_3hours_1', '5_5hours_1', '5_24hours_1', '5_1538hours_1_1_1', '5_1545(inside)_1']
car_aceta_power_1_time_evol = ['1+4_5min_1', '1+4_30min_1', '1+4_1hour_1', '1+4_3hours_1', '1+4_5hours_1',
                               '1+4_24hours_1', '1+4_1550hour_1', '1+4_1605_1']
car_aceph_power_1_time_evol = ['1+5_5min_1', '1+5_30min_1', '1+5_1hour_1', '1+5_3hours_1', '1+5_5hours_1',
                               '1+5_24hours_1', '1+5_1632hour_1', '1+5_1635_1']
aceta_aceph_power_1_time_evol = ['4+5_5min_1', '4+5_30min_1', '4+5_1hour_1', '4+5_3hours_1', '4+5_5hours_1',
                                 '4+5_24hours_1', '1+5_1642hour_1_1', '4+5_1645_1']
car_aceta_aceph_power_1_time_evol = ['1+4+5_5min_1', '1+4+5_30min_1', '1+4+5_1hour_1', '1+4+5_3hours_1',
                                     '1+4+5_5hours_1', '1+4+5_24hours_1', '1+4+5_1650hour_1', '1+4+5_1650_1']

>>> peak_amplitude_time_log = [5, 30, 60, 180, 300, 1440, 4320, 11520]
>>> car_power_1_time_evol_mean = np.zeros((8, 1600))
... aceta_power_1_time_evol_mean = np.zeros((8, 1600))
... aceph_power_1_time_evol_mean = np.zeros((8, 1600))
... car_aceta_power_1_time_evol_mean = np.zeros((8, 1600))
... car_aceph_power_1_time_evol_mean = np.zeros((8, 1600))
... aceta_aceph_power_1_time_evol_mean = np.zeros((8, 1600))
... car_aceta_aceph_power_1_time_evol_mean = np.zeros((8, 1600))

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

>>> car_power_1_time_evol_mean_smooth_snr = car_power_1_time_evol_mean - car_power_1_time_evol_mean_smooth
... aceta_power_1_time_evol_mean_smooth_snr = aceta_power_1_time_evol_mean - aceta_power_1_time_evol_mean_smooth
... aceph_power_1_time_evol_mean_smooth_snr = aceph_power_1_time_evol_mean - aceph_power_1_time_evol_mean_smooth
... car_aceta_power_1_time_evol_mean_smooth_snr = car_aceta_power_1_time_evol_mean - car_aceta_power_1_time_evol_mean_smooth
... car_aceph_power_1_time_evol_smooth_snr = car_aceph_power_1_time_evol_mean - car_aceph_power_1_time_evol_smooth
... aceta_aceph_power_1_time_evol_smooth_snr = aceta_aceph_power_1_time_evol_mean - aceta_aceph_power_1_time_evol_smooth
... car_aceta_aceph_power_1_time_evol_smooth_snr = car_aceta_aceph_power_1_time_evol_mean - car_aceta_aceph_power_1_time_evol_smooth
...
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
>>> %history -o -p -f console_logs_v3_2021_08_26.py
>>> aceta_peaks = [360, 898, 1371]
>>> aceph_peaks = [273, 377, 466, 1102]
>>> car_aceta_peaks = [364, 444, 586, 723, 815, 905, 965, 1371]
>>> car_aceph_peaks = [364, 444, 586, 723, 815, 913, 965]
>>> aceta_aceph_peaks =[358, 1371]
>>> car_aceta_aceph_peaks = [360, 444, 733, 806, 875, 910, 962, 1366]
>>> i=0
... fig, ax = plt.subplots()
... for peak in aceta_peaks:
...     ax.plot(peak_amplitude_time_log, aceta_power_1_time_evol_mean_smooth[:, peak], label='Peak #{}'.format(i+1))
...     i+=1
... ax.legend(fontsize=15)
... ax.set_xlabel('Measuring time (mins)', fontsize=20)
... ax.set_ylabel('Intensity', fontsize=20)
... plt.title('Acetamiprid peaks time evolution at HWP 42 (2.91 mW)', fontsize=17)
... ax.set_xticks(peak_amplitude_time_log)
... ax.set_xscale("log")
... ax.xaxis.set_minor_locator(ticker.FixedLocator(peak_amplitude_time_log))
... ax.xaxis.set_major_locator(ticker.NullLocator())
... ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
... plt.show()
...
>>> plt.savefig('Acetamiprid_peak_evol.png', dpi=500)
>>> i=0
... fig, ax = plt.subplots()
... for peak in aceph_peaks:
...     ax.plot(peak_amplitude_time_log, aceph_power_1_time_evol_mean_smooth[:, peak], label='Peak #{}'.format(i+1))
...     i+=1
... ax.legend(fontsize=15)
... ax.set_xlabel('Measuring time (mins)', fontsize=20)
... ax.set_ylabel('Intensity', fontsize=20)
... plt.title('Acephate peaks time evolution at HWP 42 (2.91 mW)', fontsize=17)
... ax.set_xticks(peak_amplitude_time_log)
... ax.set_xscale("log")
... ax.xaxis.set_minor_locator(ticker.FixedLocator(peak_amplitude_time_log))
... ax.xaxis.set_major_locator(ticker.NullLocator())
... ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
... plt.show()
...
>>> plt.savefig('Acephate_peak_evol.png', dpi=500)
>>> i=0
... fig, ax = plt.subplots()
... for peak in car_aceta_peaks:
...     ax.plot(peak_amplitude_time_log, car_aceta_power_1_time_evol_mean_smooth[:, peak], label='Peak #{}'.format(i+1))
...     i+=1
... ax.legend(fontsize=15)
... ax.set_xlabel('Measuring time (mins)', fontsize=20)
... ax.set_ylabel('Intensity', fontsize=20)
... plt.title('Carbendazim + Acetamiprid peaks time evolution at HWP 42 (2.91 mW)', fontsize=17)
... ax.set_xticks(peak_amplitude_time_log)
... ax.set_xscale("log")
... ax.xaxis.set_minor_locator(ticker.FixedLocator(peak_amplitude_time_log))
... ax.xaxis.set_major_locator(ticker.NullLocator())
... ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
... plt.show()
...
>>> plt.savefig('Car_aceta_peak_evol.png', dpi=500)
>>> i=0
... fig, ax = plt.subplots()
... for peak in car_aceph_peaks:
...     ax.plot(peak_amplitude_time_log, car_aceph_power_1_time_evol_smooth[:, peak], label='Peak #{}'.format(i+1))
...     i+=1
... ax.legend(fontsize=15)
... ax.set_xlabel('Measuring time (mins)', fontsize=20)
... ax.set_ylabel('Intensity', fontsize=20)
... plt.title('Carbendazim + Acephate peaks time evolution at HWP 42 (2.91 mW)', fontsize=17)
... ax.set_xticks(peak_amplitude_time_log)
... ax.set_xscale("log")
... ax.xaxis.set_minor_locator(ticker.FixedLocator(peak_amplitude_time_log))
... ax.xaxis.set_major_locator(ticker.NullLocator())
... ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
... plt.show()
...
>>> plt.savefig('Car_aceph_peak_evol.png', dpi=500)
>>> i=0
... fig, ax = plt.subplots()
... for peak in aceta_aceph_peaks:
...     ax.plot(peak_amplitude_time_log, aceta_aceph_power_1_time_evol_smooth[:, peak], label='Peak #{}'.format(i+1))
...     i+=1
... ax.legend(fontsize=15)
... ax.set_xlabel('Measuring time (mins)', fontsize=20)
... ax.set_ylabel('Intensity', fontsize=20)
... plt.title('Acetamiprid + Acephate peaks time evolution at HWP 42 (2.91 mW)', fontsize=17)
... ax.set_xticks(peak_amplitude_time_log)
... ax.set_xscale("log")
... ax.xaxis.set_minor_locator(ticker.FixedLocator(peak_amplitude_time_log))
... ax.xaxis.set_major_locator(ticker.NullLocator())
... ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
... plt.show()
...
>>> plt.savefig('Aceta_aceph_peak_evol.png', dpi=500)
>>> i=0
... fig, ax = plt.subplots()
... for peak in car_aceta_aceph_peaks:
...     ax.plot(peak_amplitude_time_log, car_aceta_aceph_power_1_time_evol_smooth[:, peak], label='Peak #{}'.format(i+1))
...     i+=1
... ax.legend(fontsize=15)
... ax.set_xlabel('Measuring time (mins)', fontsize=20)
... ax.set_ylabel('Intensity', fontsize=20)
... plt.title('Carbendazim + Acetamiprid + Acephate peaks time evolution at HWP 42 (2.91 mW)', fontsize=14)
... ax.set_xticks(peak_amplitude_time_log)
... ax.set_xscale("log")
... ax.xaxis.set_minor_locator(ticker.FixedLocator(peak_amplitude_time_log))
... ax.xaxis.set_major_locator(ticker.NullLocator())
... ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
... plt.show()
...
>>> i=0
... fig, ax = plt.subplots()
... for peak in car_aceta_aceph_peaks:
...     ax.plot(peak_amplitude_time_log, car_aceta_aceph_power_1_time_evol_smooth[:, peak], label='Peak #{}'.format(i+1))
...     i+=1
... ax.legend(fontsize=15)
... ax.set_xlabel('Measuring time (mins)', fontsize=20)
... ax.set_ylabel('Intensity', fontsize=20)
... plt.title('Carbendazim + Acetamiprid + Acephate peaks time evolution at HWP 42 (2.91 mW)', fontsize=17)
... ax.set_xticks(peak_amplitude_time_log)
... ax.set_xscale("log")
... ax.xaxis.set_minor_locator(ticker.FixedLocator(peak_amplitude_time_log))
... ax.xaxis.set_major_locator(ticker.NullLocator())
... ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
... plt.show()
...
>>> plt.savefig('Car_aceta_aceph_peak_evol.png', dpi=500)
>>> i=0
... fig, ax = plt.subplots()
... for peak in car_aceta_aceph_peaks:
...     for j in range(len(car_aceta_aceph_power_1_time_evol_smooth)):
...         ax.plot(peak_amplitude_time_log, car_aceta_aceph_power_1_time_evol_smooth[j, peak], label='Peak #{}'.format(i+1))
...     i+=1
... ax.legend(fontsize=15)
... ax.set_xlabel('Measuring time (mins)', fontsize=20)
... ax.set_ylabel('Intensity', fontsize=20)
... plt.title('Carbendazim + Acetamiprid + Acephate peaks time evolution at HWP 42 (2.91 mW)', fontsize=17)
... ax.set_xticks(peak_amplitude_time_log)
... ax.set_xscale("log")
... ax.xaxis.set_minor_locator(ticker.FixedLocator(peak_amplitude_time_log))
... ax.xaxis.set_major_locator(ticker.NullLocator())
... ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
... plt.show()
...
>>> i=0
... fig, ax = plt.subplots()
... for peak in car_aceta_aceph_peaks:
...     for j in range(len(car_aceta_aceph_power_1_time_evol_smooth)):
...         ax.plot(peak_amplitude_time_log[j], car_aceta_aceph_power_1_time_evol_smooth[j, peak], label='Peak #{}'.format(i+1))
...     i+=1
... ax.legend(fontsize=15)
... ax.set_xlabel('Measuring time (mins)', fontsize=20)
... ax.set_ylabel('Intensity', fontsize=20)
... plt.title('Carbendazim + Acetamiprid + Acephate peaks time evolution at HWP 42 (2.91 mW)', fontsize=17)
... ax.set_xticks(peak_amplitude_time_log)
... ax.set_xscale("log")
... ax.xaxis.set_minor_locator(ticker.FixedLocator(peak_amplitude_time_log))
... ax.xaxis.set_major_locator(ticker.NullLocator())
... ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
... plt.show()
...
>>> power_labels.append('Peaks')
>>>
>>> plt.savefig('Aceta_acephate_time_evol+peaks.png', dpi=500)
>>> for i in range(len(car_power_1_time_evol_mean_smooth)):
...     plt.plot(f_sup_0810, car_aceta_aceph_power_1_time_evol_smooth[i])
... plt.vlines((f_sup_0810[car_aceta_aceph_peaks]), ymin=np.min(car_aceta_aceph_power_1_time_evol_smooth[-1], axis=-1), ymax=np.max(car_aceta_aceph_power_1_time_evol_smooth), colors='r', linestyles='dashed')
... plt.xlabel('Raman shift cm^-1', fontsize=20)
... plt.ylabel('Intensity', fontsize=20)
... plt.title('Carbendazim + Acetamiprid + Acephate time evolution at HWP 42 (2.91 mW)', fontsize=17)
... plt.legend(power_labels)
... plt.xticks(np.arange(f_sup_0810[0], f_sup_0810[-1], 150))
... plt.show()
...
>>> plt.savefig('Car_aceta_acephate_time_evol+peaks.png', dpi=500)
