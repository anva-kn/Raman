print('PyDev console: using IPython 7.26.0\n')

import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['/home/anvarkunanbayev/PycharmProjects/Raman'])
import sys

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)

import pandas as pd
from tools.ramanflow.read_data import ReadData as RD
from tools.ramanflow.prep_data import PrepData as PD
from scipy.signal import savgol_filter
import numpy as np
import math

f_sup_0813, car_0813 = RD.read_dir_tiff_files('data/20210813 SERS timed immersion experiment/1')
f_sup_0813, aceta_0813 = RD.read_dir_tiff_files('data/20210813 SERS timed immersion experiment/4')
f_sup_0813, aceph_0813 = RD.read_dir_tiff_files('data/20210813 SERS timed immersion experiment/5')
f_sup_0813, car_aceta_0813 = RD.read_dir_tiff_files('data/20210813 SERS timed immersion experiment/1+4')
f_sup_0813, car_aceph_0813 = RD.read_dir_tiff_files('data/20210813 SERS timed immersion experiment/1+5')
f_sup_0813, aceta_aceph_0813 = RD.read_dir_tiff_files('data/20210813 SERS timed immersion experiment/4+5')
f_sup_0813, car_aceta_aceph_0813 = RD.read_dir_tiff_files('data/20210813 SERS timed immersion experiment/1+4+5')
f_sup_0810, car_0810 = RD.read_dir_tiff_files('data/20210810 SERS timed immersion experiment/1')
f_sup_0810, aceta_0810 = RD.read_dir_tiff_files('data/20210810 SERS timed immersion experiment/4')
f_sup_0810, aceph_0810 = RD.read_dir_tiff_files('data/20210810 SERS timed immersion experiment/5')
f_sup_0810, car_aceta_0810 = RD.read_dir_tiff_files('data/20210810 SERS timed immersion experiment/1+4')
f_sup_0810, car_aceph_0810 = RD.read_dir_tiff_files('data/20210810 SERS timed immersion experiment/1+5')
f_sup_0810, aceta_aceph_0810 = RD.read_dir_tiff_files('data/20210810 SERS timed immersion experiment/4+5')
f_sup_0810, car_aceta_aceph_0810 = RD.read_dir_tiff_files('data/20210810 SERS timed immersion experiment/1+4+5')
f_sup_0811, car_0811 = RD.read_dir_tiff_files('data/20210811 SERS timed immersion experiment/1')
f_sup_0811, aceta_0811 = RD.read_dir_tiff_files('data/20210811 SERS timed immersion experiment/4')
f_sup_0811, aceph_0811 = RD.read_dir_tiff_files('data/20210811 SERS timed immersion experiment/5')
f_sup_0811, car_aceta_0811 = RD.read_dir_tiff_files('data/20210811 SERS timed immersion experiment/1+4')
f_sup_0811, car_aceph_0811 = RD.read_dir_tiff_files('data/20210811 SERS timed immersion experiment/1+5')
f_sup_0811, aceta_aceph_0811 = RD.read_dir_tiff_files('data/20210811 SERS timed immersion experiment/4+5')
f_sup_0811, car_aceta_aceph_0811 = RD.read_dir_tiff_files('data/20210811 SERS timed immersion experiment/1+4+5')

f_sup_0818, car_0818 = RD.read_dir_tiff_files('data/20210818 SERS timed immersion experiment/1')
f_sup_0818, aceta_0818 = RD.read_dir_tiff_files('data/20210818 SERS timed immersion experiment/4')
f_sup_0818, aceph_0818 = RD.read_dir_tiff_files('data/20210818 SERS timed immersion experiment/5')
f_sup_0818, car_aceta_0818 = RD.read_dir_tiff_files('data/20210818 SERS timed immersion experiment/1+4')
f_sup_0818, car_aceph_0818 = RD.read_dir_tiff_files('data/20210818 SERS timed immersion experiment/1+5')
f_sup_0818, aceta_aceph_0818 = RD.read_dir_tiff_files('data/20210818 SERS timed immersion experiment/4+5')
f_sup_0818, car_aceta_aceph_0818 = RD.read_dir_tiff_files('data/20210818 SERS timed immersion experiment/1+4+5')



peak_amplitude_time_log = [5, 30, 60, 180, 300, 1440, 4320, 11520]

car_power_time_evol_mean = np.zeros((5, 8, 1600))
aceta_power_time_evol_mean = np.zeros((5, 8, 1600))
aceph_power_time_evol_mean = np.zeros((5, 8, 1600))
car_aceta_power_time_evol_mean = np.zeros((5, 8, 1600))
car_aceph_power_time_evol_mean = np.zeros((5, 8, 1600))
aceta_aceph_power_time_evol_mean = np.zeros((5, 8, 1600))
car_aceta_aceph_power_time_evol_mean = np.zeros((5, 8, 1600))

time_labels = ['5 min', '30 min', '1 hour', '3 hours', '5 hours', '24 hours', '3 days (4320 mins)', '8 days (11520 mins)']

car_power_1_time_evol = ['1_5min_1', '1_30min_1', '1_1hour_1', '1_3hours_1', '1_5hours_1', '1_24hours_1', '1_1458hours_1_1', '1_1530_1']
car_power_2_time_evol = ['1_5min_2', '1_30min_2', '1_1hour_2', '1_3hours_2', '1_5hours_2', '1_24hours_2', '1_1458hours_1_2', '1_1530_2']
car_power_3_time_evol = ['1_5min_3', '1_30min_3', '1_1hour_3', '1_3hours_3', '1_5hours_3', '1_24hours_3', '1_1458hours_1_3', '1_1530_3']
car_power_4_time_evol = ['1_5min_4', '1_30min_4', '1_1hour_4', '1_3hours_4', '1_5hours_4', '1_24hours_4', '1_1458hours_1_4', '1_1530_4']
car_power_5_time_evol = ['1_5min_5', '1_30min_5', '1_1hour_5', '1_3hours_5', '1_5hours_5', '1_24hours_5', '1_1458hours_1_5', '1_1530_5']

aceta_power_1_time_evol = ['4_5min_1', '4_30min_1', '4_1hour_1', '4_3hours_1', '4_5hours_1', '4_24hours_1', '1_1513hours_1_1_1', '4_1540_1']
aceta_power_2_time_evol = ['4_5min_2', '4_30min_2', '4_1hour_2', '4_3hours_2', '4_5hours_2', '4_24hours_2', '1_1513hours_1_1_2', '4_1540_2']
aceta_power_3_time_evol = ['4_5min_3', '4_30min_3', '4_1hour_3', '4_3hours_3', '4_5hours_3', '4_24hours_3', '1_1513hours_1_1_3', '4_1540_3']
aceta_power_4_time_evol = ['4_5min_4', '4_30min_4', '4_1hour_4', '4_3hours_4', '4_5hours_4', '4_24hours_4', '1_1513hours_1_1_4', '4_1540_4']
aceta_power_5_time_evol = ['4_5min_5', '4_30min_5', '4_1hour_5', '4_3hours_5', '4_5hours_5', '4_24hours_5', '1_1513hours_1_1_5', '4_1540_5']


aceph_power_1_time_evol = ['5_5min_1', '5_30min_1', '5_1hour_1', '5_3hours_1', '5_5hours_1', '5_24hours_1', '5_1538hours_1_1_1', '5_1545(inside)_1']
aceph_power_2_time_evol = ['5_5min_2', '5_30min_2', '5_1hour_2', '5_3hours_2', '5_5hours_2', '5_24hours_2', '5_1538hours_1_1_2', '5_1545(inside)_2']
aceph_power_3_time_evol = ['5_5min_3', '5_30min_3', '5_1hour_3', '5_3hours_3', '5_5hours_3', '5_24hours_3', '5_1538hours_1_1_3', '5_1545(inside)_3']
aceph_power_4_time_evol = ['5_5min_4', '5_30min_4', '5_1hour_4', '5_3hours_4', '5_5hours_4', '5_24hours_4', '5_1538hours_1_1_4', '5_1545(inside)_4']
aceph_power_5_time_evol = ['5_5min_5', '5_30min_5', '5_1hour_5', '5_3hours_5', '5_5hours_5', '5_24hours_5', '5_1538hours_1_1_5', '5_1545(inside)_5']


car_aceta_power_1_time_evol = ['1+4_5min_1', '1+4_30min_1', '1+4_1hour_1', '1+4_3hours_1', '1+4_5hours_1', '1+4_24hours_1', '1+4_1550hour_1', '1+4_1605_1']
car_aceta_power_2_time_evol = ['1+4_5min_2', '1+4_30min_2', '1+4_1hour_2', '1+4_3hours_2', '1+4_5hours_2', '1+4_24hours_2', '1+4_1550hour_2', '1+4_1605_2']
car_aceta_power_3_time_evol = ['1+4_5min_3', '1+4_30min_3', '1+4_1hour_3', '1+4_3hours_3', '1+4_5hours_3', '1+4_24hours_3', '1+4_1550hour_3', '1+4_1605_3']
car_aceta_power_4_time_evol = ['1+4_5min_4', '1+4_30min_4', '1+4_1hour_4', '1+4_3hours_4', '1+4_5hours_4', '1+4_24hours_4', '1+4_1550hour_4', '1+4_1605_4']
car_aceta_power_5_time_evol = ['1+4_5min_5', '1+4_30min_5', '1+4_1hour_5', '1+4_3hours_5', '1+4_5hours_5', '1+4_24hours_5', '1+4_1550hour_5', '1+4_1605_5']


car_aceph_power_1_time_evol = ['1+5_5min_1', '1+5_30min_1', '1+5_1hour_1', '1+5_3hours_1', '1+5_5hours_1', '1+5_24hours_1', '1+5_1632hour_1', '1+5_1635_1']
car_aceph_power_2_time_evol = ['1+5_5min_2', '1+5_30min_2', '1+5_1hour_2', '1+5_3hours_2', '1+5_5hours_2', '1+5_24hours_2', '1+5_1632hour_2', '1+5_1635_2']
car_aceph_power_3_time_evol = ['1+5_5min_3', '1+5_30min_3', '1+5_1hour_3', '1+5_3hours_3', '1+5_5hours_3', '1+5_24hours_3', '1+5_1632hour_3', '1+5_1635_3']
car_aceph_power_4_time_evol = ['1+5_5min_4', '1+5_30min_4', '1+5_1hour_4', '1+5_3hours_4', '1+5_5hours_4', '1+5_24hours_4', '1+5_1632hour_4', '1+5_1635_4']
car_aceph_power_5_time_evol = ['1+5_5min_5', '1+5_30min_5', '1+5_1hour_5', '1+5_3hours_5', '1+5_5hours_5', '1+5_24hours_5', '1+5_1632hour_5', '1+5_1635_5']


aceta_aceph_power_1_time_evol = ['4+5_5min_1', '4+5_30min_1', '4+5_1hour_1', '4+5_3hours_1', '4+5_5hours_1', '4+5_24hours_1', '1+5_1642hour_1_1', '4+5_1645_1']
aceta_aceph_power_2_time_evol = ['4+5_5min_2', '4+5_30min_2', '4+5_1hour_2', '4+5_3hours_2', '4+5_5hours_2', '4+5_24hours_2', '1+5_1642hour_1_2', '4+5_1645_2']
aceta_aceph_power_3_time_evol = ['4+5_5min_3', '4+5_30min_3', '4+5_1hour_3', '4+5_3hours_3', '4+5_5hours_3', '4+5_24hours_3', '1+5_1642hour_1_3', '4+5_1645_3']
aceta_aceph_power_4_time_evol = ['4+5_5min_4', '4+5_30min_4', '4+5_1hour_4', '4+5_3hours_4', '4+5_5hours_4', '4+5_24hours_4', '1+5_1642hour_1_4', '4+5_1645_4']
aceta_aceph_power_5_time_evol = ['4+5_5min_5', '4+5_30min_5', '4+5_1hour_5', '4+5_3hours_5', '4+5_5hours_5', '4+5_24hours_5', '1+5_1642hour_1_5', '4+5_1645_5']


car_aceta_aceph_power_1_time_evol = ['1+4+5_5min_1', '1+4+5_30min_1', '1+4+5_1hour_1', '1+4+5_3hours_1', '1+4+5_5hours_1', '1+4+5_24hours_1', '1+4+5_1650hour_1', '1+4+5_1650_1']
car_aceta_aceph_power_2_time_evol = ['1+4+5_5min_2', '1+4+5_30min_2', '1+4+5_1hour_2', '1+4+5_3hours_2', '1+4+5_5hours_2', '1+4+5_24hours_2', '1+4+5_1650hour_2', '1+4+5_1650_2']
car_aceta_aceph_power_3_time_evol = ['1+4+5_5min_3', '1+4+5_30min_3', '1+4+5_1hour_3', '1+4+5_3hours_3', '1+4+5_5hours_3', '1+4+5_24hours_3', '1+4+5_1650hour_3', '1+4+5_1650_3']
car_aceta_aceph_power_4_time_evol = ['1+4+5_5min_4', '1+4+5_30min_4', '1+4+5_1hour_4', '1+4+5_3hours_4', '1+4+5_5hours_4', '1+4+5_24hours_4', '1+4+5_1650hour_4', '1+4+5_1650_4']
car_aceta_aceph_power_5_time_evol = ['1+4+5_5min_5', '1+4+5_30min_5', '1+4+5_1hour_5', '1+4+5_3hours_5', '1+4+5_5hours_5', '1+4+5_24hours_5', '1+4+5_1650hour_5', '1+4+5_1650_5']

power_table = pd.read_csv("data/power_table.csv")
power_hwp = power_table["HWP Angle"].tolist()
power_watts = power_table["no ND"].tolist()

power_hwp_labels = power_table["HWP Angle"].apply(str).tolist()
power_watts_labels = power_table["no ND"].apply(str).tolist()

carbendanzim_data = {**car_0810, **car_0811, **car_0813, **car_0818}
acetamiprid_data = {**aceta_0810, **aceta_0811, **aceta_0813, **aceta_0818}
acephate_data = {**aceph_0810, **aceph_0811, **aceph_0813, **aceph_0818}
car_aceta_data = {**car_aceta_0810, **car_aceta_0811, **car_aceta_0813, **car_aceta_0818}
car_aceph_data = {**car_aceph_0810, **car_aceph_0811, **car_aceph_0813, **car_aceph_0818}
aceta_aceph_data = {**aceta_aceph_0810, **aceta_aceph_0811, **aceta_aceph_0813, **aceta_aceph_0818}
car_aceta_aceph_data = {**car_aceta_aceph_0810, **car_aceta_aceph_0811, **car_aceta_aceph_0813, **car_aceta_aceph_0818}


j = 0
for idx_1, idx_2, idx_3, idx_4, idx_5 in zip(car_power_1_time_evol, car_power_2_time_evol, car_power_3_time_evol, car_power_4_time_evol, car_power_5_time_evol):
    car_power_time_evol_mean[0, j] = np.mean(carbendanzim_data[idx_1], axis=0)
    car_power_time_evol_mean[1, j] = np.mean(carbendanzim_data[idx_2], axis=0)
    car_power_time_evol_mean[2, j] = np.mean(carbendanzim_data[idx_3], axis=0)
    car_power_time_evol_mean[3, j] = np.mean(carbendanzim_data[idx_4], axis=0)
    car_power_time_evol_mean[4, j] = np.mean(carbendanzim_data[idx_5], axis=0)
    j += 1

j = 0
for idx_1, idx_2, idx_3, idx_4, idx_5 in zip(aceta_power_1_time_evol, aceta_power_2_time_evol, aceta_power_3_time_evol, aceta_power_4_time_evol, aceta_power_5_time_evol):
    aceta_power_time_evol_mean[0, j] = np.mean(acetamiprid_data[idx_1], axis=0)
    aceta_power_time_evol_mean[1, j] = np.mean(acetamiprid_data[idx_2], axis=0)
    aceta_power_time_evol_mean[2, j] = np.mean(acetamiprid_data[idx_3], axis=0)
    aceta_power_time_evol_mean[3, j] = np.mean(acetamiprid_data[idx_4], axis=0)
    aceta_power_time_evol_mean[4, j] = np.mean(acetamiprid_data[idx_5], axis=0)
    j += 1

j = 0
for idx_1, idx_2, idx_3, idx_4, idx_5 in zip(aceph_power_1_time_evol, aceph_power_2_time_evol, aceph_power_3_time_evol, aceph_power_4_time_evol, aceph_power_5_time_evol):
    aceph_power_time_evol_mean[0, j] = np.mean(acephate_data[idx_1], axis=0)
    aceph_power_time_evol_mean[1, j] = np.mean(acephate_data[idx_2], axis=0)
    aceph_power_time_evol_mean[2, j] = np.mean(acephate_data[idx_3], axis=0)
    aceph_power_time_evol_mean[3, j] = np.mean(acephate_data[idx_4], axis=0)
    aceph_power_time_evol_mean[4, j] = np.mean(acephate_data[idx_5], axis=0)
    j += 1

j = 0
for idx_1, idx_2, idx_3, idx_4, idx_5 in zip(car_aceta_power_1_time_evol, car_aceta_power_2_time_evol, car_aceta_power_3_time_evol, car_aceta_power_4_time_evol, car_aceta_power_5_time_evol):
    car_aceta_power_time_evol_mean[0, j] = np.mean(car_aceta_data[idx_1], axis=0)
    car_aceta_power_time_evol_mean[1, j] = np.mean(car_aceta_data[idx_2], axis=0)
    car_aceta_power_time_evol_mean[2, j] = np.mean(car_aceta_data[idx_3], axis=0)
    car_aceta_power_time_evol_mean[3, j] = np.mean(car_aceta_data[idx_4], axis=0)
    car_aceta_power_time_evol_mean[4, j] = np.mean(car_aceta_data[idx_5], axis=0)
    j += 1

j = 0
for idx_1, idx_2, idx_3, idx_4, idx_5 in zip(car_aceph_power_1_time_evol, car_aceph_power_2_time_evol, car_aceph_power_3_time_evol, car_aceph_power_4_time_evol, car_aceph_power_5_time_evol):
    car_aceph_power_time_evol_mean[0, j] = np.mean(car_aceph_data[idx_1], axis=0)
    car_aceph_power_time_evol_mean[1, j] = np.mean(car_aceph_data[idx_2], axis=0)
    car_aceph_power_time_evol_mean[2, j] = np.mean(car_aceph_data[idx_3], axis=0)
    car_aceph_power_time_evol_mean[3, j] = np.mean(car_aceph_data[idx_4], axis=0)
    car_aceph_power_time_evol_mean[4, j] = np.mean(car_aceph_data[idx_5], axis=0)
    j += 1

j = 0
for idx_1, idx_2, idx_3, idx_4, idx_5 in zip(aceta_aceph_power_1_time_evol, aceta_aceph_power_2_time_evol, aceta_aceph_power_3_time_evol, aceta_aceph_power_4_time_evol, aceta_aceph_power_5_time_evol):
    aceta_aceph_power_time_evol_mean[0, j] = np.mean(aceta_aceph_data[idx_1], axis=0)
    aceta_aceph_power_time_evol_mean[1, j] = np.mean(aceta_aceph_data[idx_2], axis=0)
    aceta_aceph_power_time_evol_mean[2, j] = np.mean(aceta_aceph_data[idx_3], axis=0)
    aceta_aceph_power_time_evol_mean[3, j] = np.mean(aceta_aceph_data[idx_4], axis=0)
    aceta_aceph_power_time_evol_mean[4, j] = np.mean(aceta_aceph_data[idx_5], axis=0)
    j += 1

j = 0
for idx_1, idx_2, idx_3, idx_4, idx_5 in zip(car_aceta_aceph_power_1_time_evol, car_aceta_aceph_power_2_time_evol, car_aceta_aceph_power_3_time_evol, car_aceta_aceph_power_4_time_evol, car_aceta_aceph_power_5_time_evol):
    car_aceta_aceph_power_time_evol_mean[0, j] = np.mean(car_aceta_aceph_data[idx_1], axis=0)
    car_aceta_aceph_power_time_evol_mean[1, j] = np.mean(car_aceta_aceph_data[idx_2], axis=0)
    car_aceta_aceph_power_time_evol_mean[2, j] = np.mean(car_aceta_aceph_data[idx_3], axis=0)
    car_aceta_aceph_power_time_evol_mean[3, j] = np.mean(car_aceta_aceph_data[idx_4], axis=0)
    car_aceta_aceph_power_time_evol_mean[4, j] = np.mean(car_aceta_aceph_data[idx_5], axis=0)
    j += 1


car_power_time_evol_mean_smooth = np.zeros_like(car_power_time_evol_mean)
aceta_power_time_evol_mean_smooth = np.zeros_like(aceta_power_time_evol_mean)
aceph_power_time_evol_mean_smooth = np.zeros_like(aceph_power_time_evol_mean)
car_aceta_power_time_evol_mean_smooth = np.zeros_like(car_aceta_power_time_evol_mean)
car_aceph_power_time_evol_smooth = np.zeros_like(car_aceph_power_time_evol_mean)
aceta_aceph_power_time_evol_smooth = np.zeros_like(aceta_aceph_power_time_evol_mean)
car_aceta_aceph_power_time_evol_smooth = np.zeros_like(car_aceta_aceph_power_time_evol_mean)

for i in range(car_power_time_evol_mean.shape[0]):
    car_power_time_evol_mean_smooth[i] = savgol_filter(car_power_time_evol_mean[i], window_length=5, polyorder=3, axis=1)
    aceta_power_time_evol_mean_smooth[i] = savgol_filter(aceta_power_time_evol_mean[i], window_length=5, polyorder=3, axis=1)
    aceph_power_time_evol_mean_smooth[i] = savgol_filter(aceph_power_time_evol_mean[i], window_length=5, polyorder=3, axis=1)
    car_aceta_power_time_evol_mean_smooth[i] = savgol_filter(car_aceta_power_time_evol_mean[i], window_length=5, polyorder=3, axis=1)
    car_aceph_power_time_evol_smooth[i] = savgol_filter(car_aceph_power_time_evol_mean[i], window_length=5, polyorder=3, axis=1)
    aceta_aceph_power_time_evol_smooth[i] = savgol_filter(aceta_aceph_power_time_evol_mean[i], window_length=5, polyorder=3, axis=1)
    car_aceta_aceph_power_time_evol_smooth[i] = savgol_filter(car_aceta_aceph_power_time_evol_mean[i], window_length=5, polyorder=3, axis=1)


car_power_time_evol_noise = car_power_time_evol_mean_smooth - car_power_time_evol_mean
aceta_power_time_evol_noise = aceta_power_time_evol_mean_smooth - aceta_power_time_evol_mean
aceph_power_time_evol_noise = aceph_power_time_evol_mean_smooth - aceph_power_time_evol_mean
car_aceta_power_time_evol_noise = car_aceta_power_time_evol_mean_smooth - car_aceta_power_time_evol_mean
car_aceph_power_time_evol_noise = car_aceph_power_time_evol_smooth - car_aceph_power_time_evol_mean
aceta_aceph_power_time_evol_noise = aceta_aceph_power_time_evol_smooth - aceta_power_time_evol_mean
car_aceta_aceph_power_time_evol_noise = car_aceta_aceph_power_time_evol_smooth - car_aceta_aceph_power_time_evol_mean

car_power_snr = np.sum(np.abs(car_power_time_evol_mean) ** 2, axis=1) / np.sum(np.abs(car_power_time_evol_noise) ** 2, axis=1)
aceta_power_snr = np.sum(np.abs(aceta_power_time_evol_mean) ** 2, axis=1) / np.sum(np.abs(aceta_power_time_evol_noise) ** 2, axis=1)
aceph_power_snr = np.sum(np.abs(aceph_power_time_evol_mean) ** 2, axis=1) / np.sum(np.abs(aceph_power_time_evol_noise) ** 2, axis=1)
car_aceta_power_snr = np.sum(np.abs(car_aceta_power_time_evol_mean) ** 2, axis=1) / np.sum(np.abs(car_aceta_power_time_evol_noise) ** 2, axis=1)
car_aceph_power_snr = np.sum(np.abs(car_aceph_power_time_evol_mean) ** 2, axis=1) / np.sum(np.abs(car_aceph_power_time_evol_noise) ** 2, axis=1)
aceta_aceph_power_snr = np.sum(np.abs(aceta_power_time_evol_mean) ** 2, axis=1) / np.sum(np.abs(aceta_aceph_power_time_evol_noise) ** 2, axis=1)
car_aceta_aceph_power_snr = np.sum(np.abs(car_aceta_aceph_power_time_evol_mean) ** 2, axis=1) / np.sum(np.abs(car_aceta_aceph_power_time_evol_noise) ** 2, axis=1)

car_power_snr_log = 10 * math.log10(car_power_snr)
aceta_power_snr_log = 10 * math.log10(aceta_power_snr)
aceph_power_snr_log = 10 * math.log10(aceph_power_snr)
car_aceta_power_snr_log = 10 * math.log10(car_aceta_power_snr)
car_aceph_power_snr_log = 10 * math.log10(car_aceph_power_snr)
aceta_aceph_power_snr_log = 10 * math.log10(aceta_aceph_power_snr)
car_aceta_aceph_power_snr_log = 10 * math.log10(car_aceta_aceph_power_snr)
import sys

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)

import pandas as pd
from tools.ramanflow.read_data import ReadData as RD
from tools.ramanflow.prep_data import PrepData as PD
from scipy.signal import savgol_filter
import numpy as np
import math

f_sup_0813, car_0813 = RD.read_dir_tiff_files('data/20210813 SERS timed immersion experiment/1')
f_sup_0813, aceta_0813 = RD.read_dir_tiff_files('data/20210813 SERS timed immersion experiment/4')
f_sup_0813, aceph_0813 = RD.read_dir_tiff_files('data/20210813 SERS timed immersion experiment/5')
f_sup_0813, car_aceta_0813 = RD.read_dir_tiff_files('data/20210813 SERS timed immersion experiment/1+4')
f_sup_0813, car_aceph_0813 = RD.read_dir_tiff_files('data/20210813 SERS timed immersion experiment/1+5')
f_sup_0813, aceta_aceph_0813 = RD.read_dir_tiff_files('data/20210813 SERS timed immersion experiment/4+5')
f_sup_0813, car_aceta_aceph_0813 = RD.read_dir_tiff_files('data/20210813 SERS timed immersion experiment/1+4+5')
f_sup_0810, car_0810 = RD.read_dir_tiff_files('data/20210810 SERS timed immersion experiment/1')
f_sup_0810, aceta_0810 = RD.read_dir_tiff_files('data/20210810 SERS timed immersion experiment/4')
f_sup_0810, aceph_0810 = RD.read_dir_tiff_files('data/20210810 SERS timed immersion experiment/5')
f_sup_0810, car_aceta_0810 = RD.read_dir_tiff_files('data/20210810 SERS timed immersion experiment/1+4')
f_sup_0810, car_aceph_0810 = RD.read_dir_tiff_files('data/20210810 SERS timed immersion experiment/1+5')
f_sup_0810, aceta_aceph_0810 = RD.read_dir_tiff_files('data/20210810 SERS timed immersion experiment/4+5')
f_sup_0810, car_aceta_aceph_0810 = RD.read_dir_tiff_files('data/20210810 SERS timed immersion experiment/1+4+5')
f_sup_0811, car_0811 = RD.read_dir_tiff_files('data/20210811 SERS timed immersion experiment/1')
f_sup_0811, aceta_0811 = RD.read_dir_tiff_files('data/20210811 SERS timed immersion experiment/4')
f_sup_0811, aceph_0811 = RD.read_dir_tiff_files('data/20210811 SERS timed immersion experiment/5')
f_sup_0811, car_aceta_0811 = RD.read_dir_tiff_files('data/20210811 SERS timed immersion experiment/1+4')
f_sup_0811, car_aceph_0811 = RD.read_dir_tiff_files('data/20210811 SERS timed immersion experiment/1+5')
f_sup_0811, aceta_aceph_0811 = RD.read_dir_tiff_files('data/20210811 SERS timed immersion experiment/4+5')
f_sup_0811, car_aceta_aceph_0811 = RD.read_dir_tiff_files('data/20210811 SERS timed immersion experiment/1+4+5')

f_sup_0818, car_0818 = RD.read_dir_tiff_files('data/20210818 SERS timed immersion experiment/1')
f_sup_0818, aceta_0818 = RD.read_dir_tiff_files('data/20210818 SERS timed immersion experiment/4')
f_sup_0818, aceph_0818 = RD.read_dir_tiff_files('data/20210818 SERS timed immersion experiment/5')
f_sup_0818, car_aceta_0818 = RD.read_dir_tiff_files('data/20210818 SERS timed immersion experiment/1+4')
f_sup_0818, car_aceph_0818 = RD.read_dir_tiff_files('data/20210818 SERS timed immersion experiment/1+5')
f_sup_0818, aceta_aceph_0818 = RD.read_dir_tiff_files('data/20210818 SERS timed immersion experiment/4+5')
f_sup_0818, car_aceta_aceph_0818 = RD.read_dir_tiff_files('data/20210818 SERS timed immersion experiment/1+4+5')



peak_amplitude_time_log = [5, 30, 60, 180, 300, 1440, 4320, 11520]

car_power_time_evol_mean = np.zeros((5, 8, 1600))
aceta_power_time_evol_mean = np.zeros((5, 8, 1600))
aceph_power_time_evol_mean = np.zeros((5, 8, 1600))
car_aceta_power_time_evol_mean = np.zeros((5, 8, 1600))
car_aceph_power_time_evol_mean = np.zeros((5, 8, 1600))
aceta_aceph_power_time_evol_mean = np.zeros((5, 8, 1600))
car_aceta_aceph_power_time_evol_mean = np.zeros((5, 8, 1600))

time_labels = ['5 min', '30 min', '1 hour', '3 hours', '5 hours', '24 hours', '3 days (4320 mins)', '8 days (11520 mins)']

car_power_1_time_evol = ['1_5min_1', '1_30min_1', '1_1hour_1', '1_3hours_1', '1_5hours_1', '1_24hours_1', '1_1458hours_1_1', '1_1530_1']
car_power_2_time_evol = ['1_5min_2', '1_30min_2', '1_1hour_2', '1_3hours_2', '1_5hours_2', '1_24hours_2', '1_1458hours_1_2', '1_1530_2']
car_power_3_time_evol = ['1_5min_3', '1_30min_3', '1_1hour_3', '1_3hours_3', '1_5hours_3', '1_24hours_3', '1_1458hours_1_3', '1_1530_3']
car_power_4_time_evol = ['1_5min_4', '1_30min_4', '1_1hour_4', '1_3hours_4', '1_5hours_4', '1_24hours_4', '1_1458hours_1_4', '1_1530_4']
car_power_5_time_evol = ['1_5min_5', '1_30min_5', '1_1hour_5', '1_3hours_5', '1_5hours_5', '1_24hours_5', '1_1458hours_1_5', '1_1530_5']

aceta_power_1_time_evol = ['4_5min_1', '4_30min_1', '4_1hour_1', '4_3hours_1', '4_5hours_1', '4_24hours_1', '1_1513hours_1_1_1', '4_1540_1']
aceta_power_2_time_evol = ['4_5min_2', '4_30min_2', '4_1hour_2', '4_3hours_2', '4_5hours_2', '4_24hours_2', '1_1513hours_1_1_2', '4_1540_2']
aceta_power_3_time_evol = ['4_5min_3', '4_30min_3', '4_1hour_3', '4_3hours_3', '4_5hours_3', '4_24hours_3', '1_1513hours_1_1_3', '4_1540_3']
aceta_power_4_time_evol = ['4_5min_4', '4_30min_4', '4_1hour_4', '4_3hours_4', '4_5hours_4', '4_24hours_4', '1_1513hours_1_1_4', '4_1540_4']
aceta_power_5_time_evol = ['4_5min_5', '4_30min_5', '4_1hour_5', '4_3hours_5', '4_5hours_5', '4_24hours_5', '1_1513hours_1_1_5', '4_1540_5']


aceph_power_1_time_evol = ['5_5min_1', '5_30min_1', '5_1hour_1', '5_3hours_1', '5_5hours_1', '5_24hours_1', '5_1538hours_1_1_1', '5_1545(inside)_1']
aceph_power_2_time_evol = ['5_5min_2', '5_30min_2', '5_1hour_2', '5_3hours_2', '5_5hours_2', '5_24hours_2', '5_1538hours_1_1_2', '5_1545(inside)_2']
aceph_power_3_time_evol = ['5_5min_3', '5_30min_3', '5_1hour_3', '5_3hours_3', '5_5hours_3', '5_24hours_3', '5_1538hours_1_1_3', '5_1545(inside)_3']
aceph_power_4_time_evol = ['5_5min_4', '5_30min_4', '5_1hour_4', '5_3hours_4', '5_5hours_4', '5_24hours_4', '5_1538hours_1_1_4', '5_1545(inside)_4']
aceph_power_5_time_evol = ['5_5min_5', '5_30min_5', '5_1hour_5', '5_3hours_5', '5_5hours_5', '5_24hours_5', '5_1538hours_1_1_5', '5_1545(inside)_5']


car_aceta_power_1_time_evol = ['1+4_5min_1', '1+4_30min_1', '1+4_1hour_1', '1+4_3hours_1', '1+4_5hours_1', '1+4_24hours_1', '1+4_1550hour_1', '1+4_1605_1']
car_aceta_power_2_time_evol = ['1+4_5min_2', '1+4_30min_2', '1+4_1hour_2', '1+4_3hours_2', '1+4_5hours_2', '1+4_24hours_2', '1+4_1550hour_2', '1+4_1605_2']
car_aceta_power_3_time_evol = ['1+4_5min_3', '1+4_30min_3', '1+4_1hour_3', '1+4_3hours_3', '1+4_5hours_3', '1+4_24hours_3', '1+4_1550hour_3', '1+4_1605_3']
car_aceta_power_4_time_evol = ['1+4_5min_4', '1+4_30min_4', '1+4_1hour_4', '1+4_3hours_4', '1+4_5hours_4', '1+4_24hours_4', '1+4_1550hour_4', '1+4_1605_4']
car_aceta_power_5_time_evol = ['1+4_5min_5', '1+4_30min_5', '1+4_1hour_5', '1+4_3hours_5', '1+4_5hours_5', '1+4_24hours_5', '1+4_1550hour_5', '1+4_1605_5']


car_aceph_power_1_time_evol = ['1+5_5min_1', '1+5_30min_1', '1+5_1hour_1', '1+5_3hours_1', '1+5_5hours_1', '1+5_24hours_1', '1+5_1632hour_1', '1+5_1635_1']
car_aceph_power_2_time_evol = ['1+5_5min_2', '1+5_30min_2', '1+5_1hour_2', '1+5_3hours_2', '1+5_5hours_2', '1+5_24hours_2', '1+5_1632hour_2', '1+5_1635_2']
car_aceph_power_3_time_evol = ['1+5_5min_3', '1+5_30min_3', '1+5_1hour_3', '1+5_3hours_3', '1+5_5hours_3', '1+5_24hours_3', '1+5_1632hour_3', '1+5_1635_3']
car_aceph_power_4_time_evol = ['1+5_5min_4', '1+5_30min_4', '1+5_1hour_4', '1+5_3hours_4', '1+5_5hours_4', '1+5_24hours_4', '1+5_1632hour_4', '1+5_1635_4']
car_aceph_power_5_time_evol = ['1+5_5min_5', '1+5_30min_5', '1+5_1hour_5', '1+5_3hours_5', '1+5_5hours_5', '1+5_24hours_5', '1+5_1632hour_5', '1+5_1635_5']


aceta_aceph_power_1_time_evol = ['4+5_5min_1', '4+5_30min_1', '4+5_1hour_1', '4+5_3hours_1', '4+5_5hours_1', '4+5_24hours_1', '1+5_1642hour_1_1', '4+5_1645_1']
aceta_aceph_power_2_time_evol = ['4+5_5min_2', '4+5_30min_2', '4+5_1hour_2', '4+5_3hours_2', '4+5_5hours_2', '4+5_24hours_2', '1+5_1642hour_1_2', '4+5_1645_2']
aceta_aceph_power_3_time_evol = ['4+5_5min_3', '4+5_30min_3', '4+5_1hour_3', '4+5_3hours_3', '4+5_5hours_3', '4+5_24hours_3', '1+5_1642hour_1_3', '4+5_1645_3']
aceta_aceph_power_4_time_evol = ['4+5_5min_4', '4+5_30min_4', '4+5_1hour_4', '4+5_3hours_4', '4+5_5hours_4', '4+5_24hours_4', '1+5_1642hour_1_4', '4+5_1645_4']
aceta_aceph_power_5_time_evol = ['4+5_5min_5', '4+5_30min_5', '4+5_1hour_5', '4+5_3hours_5', '4+5_5hours_5', '4+5_24hours_5', '1+5_1642hour_1_5', '4+5_1645_5']


car_aceta_aceph_power_1_time_evol = ['1+4+5_5min_1', '1+4+5_30min_1', '1+4+5_1hour_1', '1+4+5_3hours_1', '1+4+5_5hours_1', '1+4+5_24hours_1', '1+4+5_1650hour_1', '1+4+5_1650_1']
car_aceta_aceph_power_2_time_evol = ['1+4+5_5min_2', '1+4+5_30min_2', '1+4+5_1hour_2', '1+4+5_3hours_2', '1+4+5_5hours_2', '1+4+5_24hours_2', '1+4+5_1650hour_2', '1+4+5_1650_2']
car_aceta_aceph_power_3_time_evol = ['1+4+5_5min_3', '1+4+5_30min_3', '1+4+5_1hour_3', '1+4+5_3hours_3', '1+4+5_5hours_3', '1+4+5_24hours_3', '1+4+5_1650hour_3', '1+4+5_1650_3']
car_aceta_aceph_power_4_time_evol = ['1+4+5_5min_4', '1+4+5_30min_4', '1+4+5_1hour_4', '1+4+5_3hours_4', '1+4+5_5hours_4', '1+4+5_24hours_4', '1+4+5_1650hour_4', '1+4+5_1650_4']
car_aceta_aceph_power_5_time_evol = ['1+4+5_5min_5', '1+4+5_30min_5', '1+4+5_1hour_5', '1+4+5_3hours_5', '1+4+5_5hours_5', '1+4+5_24hours_5', '1+4+5_1650hour_5', '1+4+5_1650_5']

power_table = pd.read_csv("data/power_table.csv")
power_hwp = power_table["HWP Angle"].tolist()
power_watts = power_table["no ND"].tolist()

power_hwp_labels = power_table["HWP Angle"].apply(str).tolist()
power_watts_labels = power_table["no ND"].apply(str).tolist()

carbendanzim_data = {**car_0810, **car_0811, **car_0813, **car_0818}
acetamiprid_data = {**aceta_0810, **aceta_0811, **aceta_0813, **aceta_0818}
acephate_data = {**aceph_0810, **aceph_0811, **aceph_0813, **aceph_0818}
car_aceta_data = {**car_aceta_0810, **car_aceta_0811, **car_aceta_0813, **car_aceta_0818}
car_aceph_data = {**car_aceph_0810, **car_aceph_0811, **car_aceph_0813, **car_aceph_0818}
aceta_aceph_data = {**aceta_aceph_0810, **aceta_aceph_0811, **aceta_aceph_0813, **aceta_aceph_0818}
car_aceta_aceph_data = {**car_aceta_aceph_0810, **car_aceta_aceph_0811, **car_aceta_aceph_0813, **car_aceta_aceph_0818}


j = 0
for idx_1, idx_2, idx_3, idx_4, idx_5 in zip(car_power_1_time_evol, car_power_2_time_evol, car_power_3_time_evol, car_power_4_time_evol, car_power_5_time_evol):
    car_power_time_evol_mean[0, j] = np.mean(carbendanzim_data[idx_1], axis=0)
    car_power_time_evol_mean[1, j] = np.mean(carbendanzim_data[idx_2], axis=0)
    car_power_time_evol_mean[2, j] = np.mean(carbendanzim_data[idx_3], axis=0)
    car_power_time_evol_mean[3, j] = np.mean(carbendanzim_data[idx_4], axis=0)
    car_power_time_evol_mean[4, j] = np.mean(carbendanzim_data[idx_5], axis=0)
    j += 1

j = 0
for idx_1, idx_2, idx_3, idx_4, idx_5 in zip(aceta_power_1_time_evol, aceta_power_2_time_evol, aceta_power_3_time_evol, aceta_power_4_time_evol, aceta_power_5_time_evol):
    aceta_power_time_evol_mean[0, j] = np.mean(acetamiprid_data[idx_1], axis=0)
    aceta_power_time_evol_mean[1, j] = np.mean(acetamiprid_data[idx_2], axis=0)
    aceta_power_time_evol_mean[2, j] = np.mean(acetamiprid_data[idx_3], axis=0)
    aceta_power_time_evol_mean[3, j] = np.mean(acetamiprid_data[idx_4], axis=0)
    aceta_power_time_evol_mean[4, j] = np.mean(acetamiprid_data[idx_5], axis=0)
    j += 1

j = 0
for idx_1, idx_2, idx_3, idx_4, idx_5 in zip(aceph_power_1_time_evol, aceph_power_2_time_evol, aceph_power_3_time_evol, aceph_power_4_time_evol, aceph_power_5_time_evol):
    aceph_power_time_evol_mean[0, j] = np.mean(acephate_data[idx_1], axis=0)
    aceph_power_time_evol_mean[1, j] = np.mean(acephate_data[idx_2], axis=0)
    aceph_power_time_evol_mean[2, j] = np.mean(acephate_data[idx_3], axis=0)
    aceph_power_time_evol_mean[3, j] = np.mean(acephate_data[idx_4], axis=0)
    aceph_power_time_evol_mean[4, j] = np.mean(acephate_data[idx_5], axis=0)
    j += 1

j = 0
for idx_1, idx_2, idx_3, idx_4, idx_5 in zip(car_aceta_power_1_time_evol, car_aceta_power_2_time_evol, car_aceta_power_3_time_evol, car_aceta_power_4_time_evol, car_aceta_power_5_time_evol):
    car_aceta_power_time_evol_mean[0, j] = np.mean(car_aceta_data[idx_1], axis=0)
    car_aceta_power_time_evol_mean[1, j] = np.mean(car_aceta_data[idx_2], axis=0)
    car_aceta_power_time_evol_mean[2, j] = np.mean(car_aceta_data[idx_3], axis=0)
    car_aceta_power_time_evol_mean[3, j] = np.mean(car_aceta_data[idx_4], axis=0)
    car_aceta_power_time_evol_mean[4, j] = np.mean(car_aceta_data[idx_5], axis=0)
    j += 1

j = 0
for idx_1, idx_2, idx_3, idx_4, idx_5 in zip(car_aceph_power_1_time_evol, car_aceph_power_2_time_evol, car_aceph_power_3_time_evol, car_aceph_power_4_time_evol, car_aceph_power_5_time_evol):
    car_aceph_power_time_evol_mean[0, j] = np.mean(car_aceph_data[idx_1], axis=0)
    car_aceph_power_time_evol_mean[1, j] = np.mean(car_aceph_data[idx_2], axis=0)
    car_aceph_power_time_evol_mean[2, j] = np.mean(car_aceph_data[idx_3], axis=0)
    car_aceph_power_time_evol_mean[3, j] = np.mean(car_aceph_data[idx_4], axis=0)
    car_aceph_power_time_evol_mean[4, j] = np.mean(car_aceph_data[idx_5], axis=0)
    j += 1

j = 0
for idx_1, idx_2, idx_3, idx_4, idx_5 in zip(aceta_aceph_power_1_time_evol, aceta_aceph_power_2_time_evol, aceta_aceph_power_3_time_evol, aceta_aceph_power_4_time_evol, aceta_aceph_power_5_time_evol):
    aceta_aceph_power_time_evol_mean[0, j] = np.mean(aceta_aceph_data[idx_1], axis=0)
    aceta_aceph_power_time_evol_mean[1, j] = np.mean(aceta_aceph_data[idx_2], axis=0)
    aceta_aceph_power_time_evol_mean[2, j] = np.mean(aceta_aceph_data[idx_3], axis=0)
    aceta_aceph_power_time_evol_mean[3, j] = np.mean(aceta_aceph_data[idx_4], axis=0)
    aceta_aceph_power_time_evol_mean[4, j] = np.mean(aceta_aceph_data[idx_5], axis=0)
    j += 1

j = 0
for idx_1, idx_2, idx_3, idx_4, idx_5 in zip(car_aceta_aceph_power_1_time_evol, car_aceta_aceph_power_2_time_evol, car_aceta_aceph_power_3_time_evol, car_aceta_aceph_power_4_time_evol, car_aceta_aceph_power_5_time_evol):
    car_aceta_aceph_power_time_evol_mean[0, j] = np.mean(car_aceta_aceph_data[idx_1], axis=0)
    car_aceta_aceph_power_time_evol_mean[1, j] = np.mean(car_aceta_aceph_data[idx_2], axis=0)
    car_aceta_aceph_power_time_evol_mean[2, j] = np.mean(car_aceta_aceph_data[idx_3], axis=0)
    car_aceta_aceph_power_time_evol_mean[3, j] = np.mean(car_aceta_aceph_data[idx_4], axis=0)
    car_aceta_aceph_power_time_evol_mean[4, j] = np.mean(car_aceta_aceph_data[idx_5], axis=0)
    j += 1


car_power_time_evol_mean_smooth = np.zeros_like(car_power_time_evol_mean)
aceta_power_time_evol_mean_smooth = np.zeros_like(aceta_power_time_evol_mean)
aceph_power_time_evol_mean_smooth = np.zeros_like(aceph_power_time_evol_mean)
car_aceta_power_time_evol_mean_smooth = np.zeros_like(car_aceta_power_time_evol_mean)
car_aceph_power_time_evol_smooth = np.zeros_like(car_aceph_power_time_evol_mean)
aceta_aceph_power_time_evol_smooth = np.zeros_like(aceta_aceph_power_time_evol_mean)
car_aceta_aceph_power_time_evol_smooth = np.zeros_like(car_aceta_aceph_power_time_evol_mean)

for i in range(car_power_time_evol_mean.shape[0]):
    car_power_time_evol_mean_smooth[i] = savgol_filter(car_power_time_evol_mean[i], window_length=5, polyorder=3, axis=1)
    aceta_power_time_evol_mean_smooth[i] = savgol_filter(aceta_power_time_evol_mean[i], window_length=5, polyorder=3, axis=1)
    aceph_power_time_evol_mean_smooth[i] = savgol_filter(aceph_power_time_evol_mean[i], window_length=5, polyorder=3, axis=1)
    car_aceta_power_time_evol_mean_smooth[i] = savgol_filter(car_aceta_power_time_evol_mean[i], window_length=5, polyorder=3, axis=1)
    car_aceph_power_time_evol_smooth[i] = savgol_filter(car_aceph_power_time_evol_mean[i], window_length=5, polyorder=3, axis=1)
    aceta_aceph_power_time_evol_smooth[i] = savgol_filter(aceta_aceph_power_time_evol_mean[i], window_length=5, polyorder=3, axis=1)
    car_aceta_aceph_power_time_evol_smooth[i] = savgol_filter(car_aceta_aceph_power_time_evol_mean[i], window_length=5, polyorder=3, axis=1)


car_power_time_evol_noise = car_power_time_evol_mean_smooth - car_power_time_evol_mean
aceta_power_time_evol_noise = aceta_power_time_evol_mean_smooth - aceta_power_time_evol_mean
aceph_power_time_evol_noise = aceph_power_time_evol_mean_smooth - aceph_power_time_evol_mean
car_aceta_power_time_evol_noise = car_aceta_power_time_evol_mean_smooth - car_aceta_power_time_evol_mean
car_aceph_power_time_evol_noise = car_aceph_power_time_evol_smooth - car_aceph_power_time_evol_mean
aceta_aceph_power_time_evol_noise = aceta_aceph_power_time_evol_smooth - aceta_power_time_evol_mean
car_aceta_aceph_power_time_evol_noise = car_aceta_aceph_power_time_evol_smooth - car_aceta_aceph_power_time_evol_mean

car_power_snr = np.sum(np.abs(car_power_time_evol_mean) ** 2, axis=1) / np.sum(np.abs(car_power_time_evol_noise) ** 2, axis=1)
aceta_power_snr = np.sum(np.abs(aceta_power_time_evol_mean) ** 2, axis=1) / np.sum(np.abs(aceta_power_time_evol_noise) ** 2, axis=1)
aceph_power_snr = np.sum(np.abs(aceph_power_time_evol_mean) ** 2, axis=1) / np.sum(np.abs(aceph_power_time_evol_noise) ** 2, axis=1)
car_aceta_power_snr = np.sum(np.abs(car_aceta_power_time_evol_mean) ** 2, axis=1) / np.sum(np.abs(car_aceta_power_time_evol_noise) ** 2, axis=1)
car_aceph_power_snr = np.sum(np.abs(car_aceph_power_time_evol_mean) ** 2, axis=1) / np.sum(np.abs(car_aceph_power_time_evol_noise) ** 2, axis=1)
aceta_aceph_power_snr = np.sum(np.abs(aceta_power_time_evol_mean) ** 2, axis=1) / np.sum(np.abs(aceta_aceph_power_time_evol_noise) ** 2, axis=1)
car_aceta_aceph_power_snr = np.sum(np.abs(car_aceta_aceph_power_time_evol_mean) ** 2, axis=1) / np.sum(np.abs(car_aceta_aceph_power_time_evol_noise) ** 2, axis=1)

car_power_snr_log = 10 * np.log10(car_power_snr)
aceta_power_snr_log = 10 * np.log10(aceta_power_snr)
aceph_power_snr_log = 10 * np.log10(aceph_power_snr)
car_aceta_power_snr_log = 10 * np.log10(car_aceta_power_snr)
car_aceph_power_snr_log = 10 * np.log10(car_aceph_power_snr)
aceta_aceph_power_snr_log = 10 * np.log10(aceta_aceph_power_snr)
car_aceta_aceph_power_snr_log = 10 * np.log10(car_aceta_aceph_power_snr)
np.sum(np.abs(car_power_time_evol_mean) ** 2, axis=1)
np.sum(np.abs(car_power_time_evol_mean) ** 2, axis=-1)
(np.sum(np.abs(car_power_time_evol_mean) ** 2, axis=-1)).shape
import sys

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)

import pandas as pd
from tools.ramanflow.read_data import ReadData as RD
from tools.ramanflow.prep_data import PrepData as PD
from scipy.signal import savgol_filter
import numpy as np
import math

f_sup_0813, car_0813 = RD.read_dir_tiff_files('data/20210813 SERS timed immersion experiment/1')
f_sup_0813, aceta_0813 = RD.read_dir_tiff_files('data/20210813 SERS timed immersion experiment/4')
f_sup_0813, aceph_0813 = RD.read_dir_tiff_files('data/20210813 SERS timed immersion experiment/5')
f_sup_0813, car_aceta_0813 = RD.read_dir_tiff_files('data/20210813 SERS timed immersion experiment/1+4')
f_sup_0813, car_aceph_0813 = RD.read_dir_tiff_files('data/20210813 SERS timed immersion experiment/1+5')
f_sup_0813, aceta_aceph_0813 = RD.read_dir_tiff_files('data/20210813 SERS timed immersion experiment/4+5')
f_sup_0813, car_aceta_aceph_0813 = RD.read_dir_tiff_files('data/20210813 SERS timed immersion experiment/1+4+5')
f_sup_0810, car_0810 = RD.read_dir_tiff_files('data/20210810 SERS timed immersion experiment/1')
f_sup_0810, aceta_0810 = RD.read_dir_tiff_files('data/20210810 SERS timed immersion experiment/4')
f_sup_0810, aceph_0810 = RD.read_dir_tiff_files('data/20210810 SERS timed immersion experiment/5')
f_sup_0810, car_aceta_0810 = RD.read_dir_tiff_files('data/20210810 SERS timed immersion experiment/1+4')
f_sup_0810, car_aceph_0810 = RD.read_dir_tiff_files('data/20210810 SERS timed immersion experiment/1+5')
f_sup_0810, aceta_aceph_0810 = RD.read_dir_tiff_files('data/20210810 SERS timed immersion experiment/4+5')
f_sup_0810, car_aceta_aceph_0810 = RD.read_dir_tiff_files('data/20210810 SERS timed immersion experiment/1+4+5')
f_sup_0811, car_0811 = RD.read_dir_tiff_files('data/20210811 SERS timed immersion experiment/1')
f_sup_0811, aceta_0811 = RD.read_dir_tiff_files('data/20210811 SERS timed immersion experiment/4')
f_sup_0811, aceph_0811 = RD.read_dir_tiff_files('data/20210811 SERS timed immersion experiment/5')
f_sup_0811, car_aceta_0811 = RD.read_dir_tiff_files('data/20210811 SERS timed immersion experiment/1+4')
f_sup_0811, car_aceph_0811 = RD.read_dir_tiff_files('data/20210811 SERS timed immersion experiment/1+5')
f_sup_0811, aceta_aceph_0811 = RD.read_dir_tiff_files('data/20210811 SERS timed immersion experiment/4+5')
f_sup_0811, car_aceta_aceph_0811 = RD.read_dir_tiff_files('data/20210811 SERS timed immersion experiment/1+4+5')

f_sup_0818, car_0818 = RD.read_dir_tiff_files('data/20210818 SERS timed immersion experiment/1')
f_sup_0818, aceta_0818 = RD.read_dir_tiff_files('data/20210818 SERS timed immersion experiment/4')
f_sup_0818, aceph_0818 = RD.read_dir_tiff_files('data/20210818 SERS timed immersion experiment/5')
f_sup_0818, car_aceta_0818 = RD.read_dir_tiff_files('data/20210818 SERS timed immersion experiment/1+4')
f_sup_0818, car_aceph_0818 = RD.read_dir_tiff_files('data/20210818 SERS timed immersion experiment/1+5')
f_sup_0818, aceta_aceph_0818 = RD.read_dir_tiff_files('data/20210818 SERS timed immersion experiment/4+5')
f_sup_0818, car_aceta_aceph_0818 = RD.read_dir_tiff_files('data/20210818 SERS timed immersion experiment/1+4+5')



peak_amplitude_time_log = [5, 30, 60, 180, 300, 1440, 4320, 11520]

car_power_time_evol_mean = np.zeros((5, 8, 1600))
aceta_power_time_evol_mean = np.zeros((5, 8, 1600))
aceph_power_time_evol_mean = np.zeros((5, 8, 1600))
car_aceta_power_time_evol_mean = np.zeros((5, 8, 1600))
car_aceph_power_time_evol_mean = np.zeros((5, 8, 1600))
aceta_aceph_power_time_evol_mean = np.zeros((5, 8, 1600))
car_aceta_aceph_power_time_evol_mean = np.zeros((5, 8, 1600))

time_labels = ['5 min', '30 min', '1 hour', '3 hours', '5 hours', '24 hours', '3 days (4320 mins)', '8 days (11520 mins)']

car_power_1_time_evol = ['1_5min_1', '1_30min_1', '1_1hour_1', '1_3hours_1', '1_5hours_1', '1_24hours_1', '1_1458hours_1_1', '1_1530_1']
car_power_2_time_evol = ['1_5min_2', '1_30min_2', '1_1hour_2', '1_3hours_2', '1_5hours_2', '1_24hours_2', '1_1458hours_1_2', '1_1530_2']
car_power_3_time_evol = ['1_5min_3', '1_30min_3', '1_1hour_3', '1_3hours_3', '1_5hours_3', '1_24hours_3', '1_1458hours_1_3', '1_1530_3']
car_power_4_time_evol = ['1_5min_4', '1_30min_4', '1_1hour_4', '1_3hours_4', '1_5hours_4', '1_24hours_4', '1_1458hours_1_4', '1_1530_4']
car_power_5_time_evol = ['1_5min_5', '1_30min_5', '1_1hour_5', '1_3hours_5', '1_5hours_5', '1_24hours_5', '1_1458hours_1_5', '1_1530_5']

aceta_power_1_time_evol = ['4_5min_1', '4_30min_1', '4_1hour_1', '4_3hours_1', '4_5hours_1', '4_24hours_1', '1_1513hours_1_1_1', '4_1540_1']
aceta_power_2_time_evol = ['4_5min_2', '4_30min_2', '4_1hour_2', '4_3hours_2', '4_5hours_2', '4_24hours_2', '1_1513hours_1_1_2', '4_1540_2']
aceta_power_3_time_evol = ['4_5min_3', '4_30min_3', '4_1hour_3', '4_3hours_3', '4_5hours_3', '4_24hours_3', '1_1513hours_1_1_3', '4_1540_3']
aceta_power_4_time_evol = ['4_5min_4', '4_30min_4', '4_1hour_4', '4_3hours_4', '4_5hours_4', '4_24hours_4', '1_1513hours_1_1_4', '4_1540_4']
aceta_power_5_time_evol = ['4_5min_5', '4_30min_5', '4_1hour_5', '4_3hours_5', '4_5hours_5', '4_24hours_5', '1_1513hours_1_1_5', '4_1540_5']


aceph_power_1_time_evol = ['5_5min_1', '5_30min_1', '5_1hour_1', '5_3hours_1', '5_5hours_1', '5_24hours_1', '5_1538hours_1_1_1', '5_1545(inside)_1']
aceph_power_2_time_evol = ['5_5min_2', '5_30min_2', '5_1hour_2', '5_3hours_2', '5_5hours_2', '5_24hours_2', '5_1538hours_1_1_2', '5_1545(inside)_2']
aceph_power_3_time_evol = ['5_5min_3', '5_30min_3', '5_1hour_3', '5_3hours_3', '5_5hours_3', '5_24hours_3', '5_1538hours_1_1_3', '5_1545(inside)_3']
aceph_power_4_time_evol = ['5_5min_4', '5_30min_4', '5_1hour_4', '5_3hours_4', '5_5hours_4', '5_24hours_4', '5_1538hours_1_1_4', '5_1545(inside)_4']
aceph_power_5_time_evol = ['5_5min_5', '5_30min_5', '5_1hour_5', '5_3hours_5', '5_5hours_5', '5_24hours_5', '5_1538hours_1_1_5', '5_1545(inside)_5']


car_aceta_power_1_time_evol = ['1+4_5min_1', '1+4_30min_1', '1+4_1hour_1', '1+4_3hours_1', '1+4_5hours_1', '1+4_24hours_1', '1+4_1550hour_1', '1+4_1605_1']
car_aceta_power_2_time_evol = ['1+4_5min_2', '1+4_30min_2', '1+4_1hour_2', '1+4_3hours_2', '1+4_5hours_2', '1+4_24hours_2', '1+4_1550hour_2', '1+4_1605_2']
car_aceta_power_3_time_evol = ['1+4_5min_3', '1+4_30min_3', '1+4_1hour_3', '1+4_3hours_3', '1+4_5hours_3', '1+4_24hours_3', '1+4_1550hour_3', '1+4_1605_3']
car_aceta_power_4_time_evol = ['1+4_5min_4', '1+4_30min_4', '1+4_1hour_4', '1+4_3hours_4', '1+4_5hours_4', '1+4_24hours_4', '1+4_1550hour_4', '1+4_1605_4']
car_aceta_power_5_time_evol = ['1+4_5min_5', '1+4_30min_5', '1+4_1hour_5', '1+4_3hours_5', '1+4_5hours_5', '1+4_24hours_5', '1+4_1550hour_5', '1+4_1605_5']


car_aceph_power_1_time_evol = ['1+5_5min_1', '1+5_30min_1', '1+5_1hour_1', '1+5_3hours_1', '1+5_5hours_1', '1+5_24hours_1', '1+5_1632hour_1', '1+5_1635_1']
car_aceph_power_2_time_evol = ['1+5_5min_2', '1+5_30min_2', '1+5_1hour_2', '1+5_3hours_2', '1+5_5hours_2', '1+5_24hours_2', '1+5_1632hour_2', '1+5_1635_2']
car_aceph_power_3_time_evol = ['1+5_5min_3', '1+5_30min_3', '1+5_1hour_3', '1+5_3hours_3', '1+5_5hours_3', '1+5_24hours_3', '1+5_1632hour_3', '1+5_1635_3']
car_aceph_power_4_time_evol = ['1+5_5min_4', '1+5_30min_4', '1+5_1hour_4', '1+5_3hours_4', '1+5_5hours_4', '1+5_24hours_4', '1+5_1632hour_4', '1+5_1635_4']
car_aceph_power_5_time_evol = ['1+5_5min_5', '1+5_30min_5', '1+5_1hour_5', '1+5_3hours_5', '1+5_5hours_5', '1+5_24hours_5', '1+5_1632hour_5', '1+5_1635_5']


aceta_aceph_power_1_time_evol = ['4+5_5min_1', '4+5_30min_1', '4+5_1hour_1', '4+5_3hours_1', '4+5_5hours_1', '4+5_24hours_1', '1+5_1642hour_1_1', '4+5_1645_1']
aceta_aceph_power_2_time_evol = ['4+5_5min_2', '4+5_30min_2', '4+5_1hour_2', '4+5_3hours_2', '4+5_5hours_2', '4+5_24hours_2', '1+5_1642hour_1_2', '4+5_1645_2']
aceta_aceph_power_3_time_evol = ['4+5_5min_3', '4+5_30min_3', '4+5_1hour_3', '4+5_3hours_3', '4+5_5hours_3', '4+5_24hours_3', '1+5_1642hour_1_3', '4+5_1645_3']
aceta_aceph_power_4_time_evol = ['4+5_5min_4', '4+5_30min_4', '4+5_1hour_4', '4+5_3hours_4', '4+5_5hours_4', '4+5_24hours_4', '1+5_1642hour_1_4', '4+5_1645_4']
aceta_aceph_power_5_time_evol = ['4+5_5min_5', '4+5_30min_5', '4+5_1hour_5', '4+5_3hours_5', '4+5_5hours_5', '4+5_24hours_5', '1+5_1642hour_1_5', '4+5_1645_5']


car_aceta_aceph_power_1_time_evol = ['1+4+5_5min_1', '1+4+5_30min_1', '1+4+5_1hour_1', '1+4+5_3hours_1', '1+4+5_5hours_1', '1+4+5_24hours_1', '1+4+5_1650hour_1', '1+4+5_1650_1']
car_aceta_aceph_power_2_time_evol = ['1+4+5_5min_2', '1+4+5_30min_2', '1+4+5_1hour_2', '1+4+5_3hours_2', '1+4+5_5hours_2', '1+4+5_24hours_2', '1+4+5_1650hour_2', '1+4+5_1650_2']
car_aceta_aceph_power_3_time_evol = ['1+4+5_5min_3', '1+4+5_30min_3', '1+4+5_1hour_3', '1+4+5_3hours_3', '1+4+5_5hours_3', '1+4+5_24hours_3', '1+4+5_1650hour_3', '1+4+5_1650_3']
car_aceta_aceph_power_4_time_evol = ['1+4+5_5min_4', '1+4+5_30min_4', '1+4+5_1hour_4', '1+4+5_3hours_4', '1+4+5_5hours_4', '1+4+5_24hours_4', '1+4+5_1650hour_4', '1+4+5_1650_4']
car_aceta_aceph_power_5_time_evol = ['1+4+5_5min_5', '1+4+5_30min_5', '1+4+5_1hour_5', '1+4+5_3hours_5', '1+4+5_5hours_5', '1+4+5_24hours_5', '1+4+5_1650hour_5', '1+4+5_1650_5']

power_table = pd.read_csv("data/power_table.csv")
power_hwp = power_table["HWP Angle"].tolist()
power_watts = power_table["no ND"].tolist()

power_hwp_labels = power_table["HWP Angle"].apply(str).tolist()
power_watts_labels = power_table["no ND"].apply(str).tolist()

carbendanzim_data = {**car_0810, **car_0811, **car_0813, **car_0818}
acetamiprid_data = {**aceta_0810, **aceta_0811, **aceta_0813, **aceta_0818}
acephate_data = {**aceph_0810, **aceph_0811, **aceph_0813, **aceph_0818}
car_aceta_data = {**car_aceta_0810, **car_aceta_0811, **car_aceta_0813, **car_aceta_0818}
car_aceph_data = {**car_aceph_0810, **car_aceph_0811, **car_aceph_0813, **car_aceph_0818}
aceta_aceph_data = {**aceta_aceph_0810, **aceta_aceph_0811, **aceta_aceph_0813, **aceta_aceph_0818}
car_aceta_aceph_data = {**car_aceta_aceph_0810, **car_aceta_aceph_0811, **car_aceta_aceph_0813, **car_aceta_aceph_0818}


j = 0
for idx_1, idx_2, idx_3, idx_4, idx_5 in zip(car_power_1_time_evol, car_power_2_time_evol, car_power_3_time_evol, car_power_4_time_evol, car_power_5_time_evol):
    car_power_time_evol_mean[0, j] = np.mean(carbendanzim_data[idx_1], axis=0)
    car_power_time_evol_mean[1, j] = np.mean(carbendanzim_data[idx_2], axis=0)
    car_power_time_evol_mean[2, j] = np.mean(carbendanzim_data[idx_3], axis=0)
    car_power_time_evol_mean[3, j] = np.mean(carbendanzim_data[idx_4], axis=0)
    car_power_time_evol_mean[4, j] = np.mean(carbendanzim_data[idx_5], axis=0)
    j += 1

j = 0
for idx_1, idx_2, idx_3, idx_4, idx_5 in zip(aceta_power_1_time_evol, aceta_power_2_time_evol, aceta_power_3_time_evol, aceta_power_4_time_evol, aceta_power_5_time_evol):
    aceta_power_time_evol_mean[0, j] = np.mean(acetamiprid_data[idx_1], axis=0)
    aceta_power_time_evol_mean[1, j] = np.mean(acetamiprid_data[idx_2], axis=0)
    aceta_power_time_evol_mean[2, j] = np.mean(acetamiprid_data[idx_3], axis=0)
    aceta_power_time_evol_mean[3, j] = np.mean(acetamiprid_data[idx_4], axis=0)
    aceta_power_time_evol_mean[4, j] = np.mean(acetamiprid_data[idx_5], axis=0)
    j += 1

j = 0
for idx_1, idx_2, idx_3, idx_4, idx_5 in zip(aceph_power_1_time_evol, aceph_power_2_time_evol, aceph_power_3_time_evol, aceph_power_4_time_evol, aceph_power_5_time_evol):
    aceph_power_time_evol_mean[0, j] = np.mean(acephate_data[idx_1], axis=0)
    aceph_power_time_evol_mean[1, j] = np.mean(acephate_data[idx_2], axis=0)
    aceph_power_time_evol_mean[2, j] = np.mean(acephate_data[idx_3], axis=0)
    aceph_power_time_evol_mean[3, j] = np.mean(acephate_data[idx_4], axis=0)
    aceph_power_time_evol_mean[4, j] = np.mean(acephate_data[idx_5], axis=0)
    j += 1

j = 0
for idx_1, idx_2, idx_3, idx_4, idx_5 in zip(car_aceta_power_1_time_evol, car_aceta_power_2_time_evol, car_aceta_power_3_time_evol, car_aceta_power_4_time_evol, car_aceta_power_5_time_evol):
    car_aceta_power_time_evol_mean[0, j] = np.mean(car_aceta_data[idx_1], axis=0)
    car_aceta_power_time_evol_mean[1, j] = np.mean(car_aceta_data[idx_2], axis=0)
    car_aceta_power_time_evol_mean[2, j] = np.mean(car_aceta_data[idx_3], axis=0)
    car_aceta_power_time_evol_mean[3, j] = np.mean(car_aceta_data[idx_4], axis=0)
    car_aceta_power_time_evol_mean[4, j] = np.mean(car_aceta_data[idx_5], axis=0)
    j += 1

j = 0
for idx_1, idx_2, idx_3, idx_4, idx_5 in zip(car_aceph_power_1_time_evol, car_aceph_power_2_time_evol, car_aceph_power_3_time_evol, car_aceph_power_4_time_evol, car_aceph_power_5_time_evol):
    car_aceph_power_time_evol_mean[0, j] = np.mean(car_aceph_data[idx_1], axis=0)
    car_aceph_power_time_evol_mean[1, j] = np.mean(car_aceph_data[idx_2], axis=0)
    car_aceph_power_time_evol_mean[2, j] = np.mean(car_aceph_data[idx_3], axis=0)
    car_aceph_power_time_evol_mean[3, j] = np.mean(car_aceph_data[idx_4], axis=0)
    car_aceph_power_time_evol_mean[4, j] = np.mean(car_aceph_data[idx_5], axis=0)
    j += 1

j = 0
for idx_1, idx_2, idx_3, idx_4, idx_5 in zip(aceta_aceph_power_1_time_evol, aceta_aceph_power_2_time_evol, aceta_aceph_power_3_time_evol, aceta_aceph_power_4_time_evol, aceta_aceph_power_5_time_evol):
    aceta_aceph_power_time_evol_mean[0, j] = np.mean(aceta_aceph_data[idx_1], axis=0)
    aceta_aceph_power_time_evol_mean[1, j] = np.mean(aceta_aceph_data[idx_2], axis=0)
    aceta_aceph_power_time_evol_mean[2, j] = np.mean(aceta_aceph_data[idx_3], axis=0)
    aceta_aceph_power_time_evol_mean[3, j] = np.mean(aceta_aceph_data[idx_4], axis=0)
    aceta_aceph_power_time_evol_mean[4, j] = np.mean(aceta_aceph_data[idx_5], axis=0)
    j += 1

j = 0
for idx_1, idx_2, idx_3, idx_4, idx_5 in zip(car_aceta_aceph_power_1_time_evol, car_aceta_aceph_power_2_time_evol, car_aceta_aceph_power_3_time_evol, car_aceta_aceph_power_4_time_evol, car_aceta_aceph_power_5_time_evol):
    car_aceta_aceph_power_time_evol_mean[0, j] = np.mean(car_aceta_aceph_data[idx_1], axis=0)
    car_aceta_aceph_power_time_evol_mean[1, j] = np.mean(car_aceta_aceph_data[idx_2], axis=0)
    car_aceta_aceph_power_time_evol_mean[2, j] = np.mean(car_aceta_aceph_data[idx_3], axis=0)
    car_aceta_aceph_power_time_evol_mean[3, j] = np.mean(car_aceta_aceph_data[idx_4], axis=0)
    car_aceta_aceph_power_time_evol_mean[4, j] = np.mean(car_aceta_aceph_data[idx_5], axis=0)
    j += 1


car_power_time_evol_mean_smooth = np.zeros_like(car_power_time_evol_mean)
aceta_power_time_evol_mean_smooth = np.zeros_like(aceta_power_time_evol_mean)
aceph_power_time_evol_mean_smooth = np.zeros_like(aceph_power_time_evol_mean)
car_aceta_power_time_evol_mean_smooth = np.zeros_like(car_aceta_power_time_evol_mean)
car_aceph_power_time_evol_smooth = np.zeros_like(car_aceph_power_time_evol_mean)
aceta_aceph_power_time_evol_smooth = np.zeros_like(aceta_aceph_power_time_evol_mean)
car_aceta_aceph_power_time_evol_smooth = np.zeros_like(car_aceta_aceph_power_time_evol_mean)

for i in range(car_power_time_evol_mean.shape[0]):
    car_power_time_evol_mean_smooth[i] = savgol_filter(car_power_time_evol_mean[i], window_length=5, polyorder=3, axis=1)
    aceta_power_time_evol_mean_smooth[i] = savgol_filter(aceta_power_time_evol_mean[i], window_length=5, polyorder=3, axis=1)
    aceph_power_time_evol_mean_smooth[i] = savgol_filter(aceph_power_time_evol_mean[i], window_length=5, polyorder=3, axis=1)
    car_aceta_power_time_evol_mean_smooth[i] = savgol_filter(car_aceta_power_time_evol_mean[i], window_length=5, polyorder=3, axis=1)
    car_aceph_power_time_evol_smooth[i] = savgol_filter(car_aceph_power_time_evol_mean[i], window_length=5, polyorder=3, axis=1)
    aceta_aceph_power_time_evol_smooth[i] = savgol_filter(aceta_aceph_power_time_evol_mean[i], window_length=5, polyorder=3, axis=1)
    car_aceta_aceph_power_time_evol_smooth[i] = savgol_filter(car_aceta_aceph_power_time_evol_mean[i], window_length=5, polyorder=3, axis=1)


car_power_time_evol_noise = car_power_time_evol_mean_smooth - car_power_time_evol_mean
aceta_power_time_evol_noise = aceta_power_time_evol_mean_smooth - aceta_power_time_evol_mean
aceph_power_time_evol_noise = aceph_power_time_evol_mean_smooth - aceph_power_time_evol_mean
car_aceta_power_time_evol_noise = car_aceta_power_time_evol_mean_smooth - car_aceta_power_time_evol_mean
car_aceph_power_time_evol_noise = car_aceph_power_time_evol_smooth - car_aceph_power_time_evol_mean
aceta_aceph_power_time_evol_noise = aceta_aceph_power_time_evol_smooth - aceta_power_time_evol_mean
car_aceta_aceph_power_time_evol_noise = car_aceta_aceph_power_time_evol_smooth - car_aceta_aceph_power_time_evol_mean

car_power_snr = np.sum(np.abs(car_power_time_evol_mean) ** 2, axis=-1) / np.sum(np.abs(car_power_time_evol_noise) ** 2, axis=-1)
aceta_power_snr = np.sum(np.abs(aceta_power_time_evol_mean) ** 2, axis=-1) / np.sum(np.abs(aceta_power_time_evol_noise) ** 2, axis=-1)
aceph_power_snr = np.sum(np.abs(aceph_power_time_evol_mean) ** 2, axis=-1) / np.sum(np.abs(aceph_power_time_evol_noise) ** 2, axis=-1)
car_aceta_power_snr = np.sum(np.abs(car_aceta_power_time_evol_mean) ** 2, axis=-1) / np.sum(np.abs(car_aceta_power_time_evol_noise) ** 2, axis=-1)
car_aceph_power_snr = np.sum(np.abs(car_aceph_power_time_evol_mean) ** 2, axis=-1) / np.sum(np.abs(car_aceph_power_time_evol_noise) ** 2, axis=-1)
aceta_aceph_power_snr = np.sum(np.abs(aceta_power_time_evol_mean) ** 2, axis=-1) / np.sum(np.abs(aceta_aceph_power_time_evol_noise) ** 2, axis=-1)
car_aceta_aceph_power_snr = np.sum(np.abs(car_aceta_aceph_power_time_evol_mean) ** 2, axis=-1) / np.sum(np.abs(car_aceta_aceph_power_time_evol_noise) ** 2, axis=-1)

car_power_snr_log = 10 * np.log10(car_power_snr)
aceta_power_snr_log = 10 * np.log10(aceta_power_snr)
aceph_power_snr_log = 10 * np.log10(aceph_power_snr)
car_aceta_power_snr_log = 10 * np.log10(car_aceta_power_snr)
car_aceph_power_snr_log = 10 * np.log10(car_aceph_power_snr)
aceta_aceph_power_snr_log = 10 * np.log10(aceta_aceph_power_snr)
car_aceta_aceph_power_snr_log = 10 * np.log10(car_aceta_aceph_power_snr)
i=0
fig, ax = plt.subplots()
for item in range(len(car_power_snr_log)):
    ax.plot(peak_amplitude_time_log, car_power_snr_log[i, :], label='Power #{}'.format(i+1))
    i+=1
ax.legend(fontsize=15)
ax.set_xlabel('Measuring time (mins)', fontsize=20)
ax.set_ylabel('Intensity', fontsize=20)
plt.title('Carbendazim power dependence', fontsize=17)
ax.set_xticks(peak_amplitude_time_log)
ax.set_xscale("log")
ax.xaxis.set_minor_locator(ticker.FixedLocator(peak_amplitude_time_log))
ax.xaxis.set_major_locator(ticker.NullLocator())
ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
plt.show()
from matplotlib import ticker
i=0
fig, ax = plt.subplots()
for item in range(len(car_power_snr_log)):
    ax.plot(peak_amplitude_time_log, car_power_snr_log[i, :], label='Power #{}'.format(i+1))
    i+=1
ax.legend(fontsize=15)
ax.set_xlabel('Measuring time (mins)', fontsize=20)
ax.set_ylabel('Intensity', fontsize=20)
plt.title('Carbendazim power dependence', fontsize=17)
ax.set_xticks(peak_amplitude_time_log)
ax.set_xscale("log")
ax.xaxis.set_minor_locator(ticker.FixedLocator(peak_amplitude_time_log))
ax.xaxis.set_major_locator(ticker.NullLocator())
ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
plt.show()
i=0
fig, ax = plt.subplots()
for item in range(len(car_power_snr_log)):
    ax.plot(peak_amplitude_time_log, car_power_snr_log[i, :], label='Power #{}'.format(i+1))
    i+=1
ax.legend(fontsize=15)
ax.set_xlabel('Measuring time (mins)', fontsize=20)
ax.set_ylabel('Intensity', fontsize=20)
plt.title('Carbendazim power dependence', fontsize=17)
ax.set_xticks(peak_amplitude_time_log)
ax.set_xscale("log")
ax.xaxis.set_minor_locator(ticker.FixedLocator(peak_amplitude_time_log))
ax.xaxis.set_major_locator(ticker.NullLocator())
ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
plt.show()
i=0
fig, ax = plt.subplots()
for item in range(len(car_power_snr_log)):
    ax.plot(peak_amplitude_time_log, car_power_snr_log[i, :], label='Power #{}'.format(i+1))
    i+=1
ax.legend(fontsize=15)
ax.set_xlabel('Measuring time (mins)', fontsize=20)
ax.set_ylabel('SNR (dB)', fontsize=20)
plt.title('Carbendazim power dependence', fontsize=17)
ax.set_xticks(peak_amplitude_time_log)
ax.set_xscale("log")
ax.xaxis.set_minor_locator(ticker.FixedLocator(peak_amplitude_time_log))
ax.xaxis.set_major_locator(ticker.NullLocator())
ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
plt.show()
experiment_power_hwp_labels = ['42', '46', '50', '54', '58']
experiment_power_watts_labels = ['2.71', '6.44', '11.41', '17.24', '23.48']
i=0
fig, ax = plt.subplots()
for item in range(len(car_power_snr_log)):
    ax.plot(peak_amplitude_time_log, car_power_snr_log[i, :], label='Power hwp {}, {}mW'.format(experiment_power_hwp_labels[i], experiment_power_watts_labels[i]))
    i+=1
ax.legend(fontsize=15)
ax.set_xlabel('Measuring time (mins)', fontsize=20)
ax.set_ylabel('SNR (dB)', fontsize=20)
plt.title('Carbendazim power dependence', fontsize=17)
ax.set_xticks(peak_amplitude_time_log)
ax.set_xscale("log")
ax.xaxis.set_minor_locator(ticker.FixedLocator(peak_amplitude_time_log))
ax.xaxis.set_major_locator(ticker.NullLocator())
ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
plt.show()
plt.savefig('Carbendazim_power_snr_evol.png', dpi=500)
i=0
fig, ax = plt.subplots()
for item in range(len(car_power_snr_log)):
    ax.plot(peak_amplitude_time_log, aceta_power_snr_log[i, :], label='Power hwp {}, {}mW'.format(experiment_power_hwp_labels[i], experiment_power_watts_labels[i]))
    i+=1
ax.legend(fontsize=15)
ax.set_xlabel('Measuring time (mins)', fontsize=20)
ax.set_ylabel('SNR (dB)', fontsize=20)
plt.title('Acetamiprid power dependence', fontsize=17)
ax.set_xticks(peak_amplitude_time_log)
ax.set_xscale("log")
ax.xaxis.set_minor_locator(ticker.FixedLocator(peak_amplitude_time_log))
ax.xaxis.set_major_locator(ticker.NullLocator())
ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
plt.show()
plt.savefig('Acetamiprid_power_snr_evol.png', dpi=500)
i=0
fig, ax = plt.subplots()
for item in range(len(car_power_snr_log)):
    ax.plot(peak_amplitude_time_log, aceph_power_snr_log[i, :], label='Power hwp {}, {}mW'.format(experiment_power_hwp_labels[i], experiment_power_watts_labels[i]))
    i+=1
ax.legend(fontsize=15)
ax.set_xlabel('Measuring time (mins)', fontsize=20)
ax.set_ylabel('SNR (dB)', fontsize=20)
plt.title('Acephate power dependence', fontsize=17)
ax.set_xticks(peak_amplitude_time_log)
ax.set_xscale("log")
ax.xaxis.set_minor_locator(ticker.FixedLocator(peak_amplitude_time_log))
ax.xaxis.set_major_locator(ticker.NullLocator())
ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
plt.show()
plt.savefig('Acephate_power_snr_evol.png', dpi=500)
i=0
fig, ax = plt.subplots()
for item in range(len(car_power_snr_log)):
    ax.plot(peak_amplitude_time_log, car_aceta_power_snr_log[i, :], label='Power hwp {}, {}mW'.format(experiment_power_hwp_labels[i], experiment_power_watts_labels[i]))
    i+=1
ax.legend(fontsize=15)
ax.set_xlabel('Measuring time (mins)', fontsize=20)
ax.set_ylabel('SNR (dB)', fontsize=20)
plt.title('Carbendazim + Acetamiprid power dependence', fontsize=17)
ax.set_xticks(peak_amplitude_time_log)
ax.set_xscale("log")
ax.xaxis.set_minor_locator(ticker.FixedLocator(peak_amplitude_time_log))
ax.xaxis.set_major_locator(ticker.NullLocator())
ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
plt.show()
plt.savefig('Carbendazim_acetamiprid_power_snr_evol.png', dpi=500)
i=0
fig, ax = plt.subplots()
for item in range(len(car_power_snr_log)):
    ax.plot(peak_amplitude_time_log, car_aceph_power_snr_log[i, :], label='Power hwp {}, {}mW'.format(experiment_power_hwp_labels[i], experiment_power_watts_labels[i]))
    i+=1
ax.legend(fontsize=15)
ax.set_xlabel('Measuring time (mins)', fontsize=20)
ax.set_ylabel('SNR (dB)', fontsize=20)
plt.title('Carbendazim + Acephate power dependence', fontsize=17)
ax.set_xticks(peak_amplitude_time_log)
ax.set_xscale("log")
ax.xaxis.set_minor_locator(ticker.FixedLocator(peak_amplitude_time_log))
ax.xaxis.set_major_locator(ticker.NullLocator())
ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
plt.show()
plt.savefig('Carbendazim_acephate_power_snr_evol.png', dpi=500)
i=0
fig, ax = plt.subplots()
for item in range(len(car_power_snr_log)):
    ax.plot(peak_amplitude_time_log, aceta_aceph_power_snr_log[i, :], label='Power hwp {}, {}mW'.format(experiment_power_hwp_labels[i], experiment_power_watts_labels[i]))
    i+=1
ax.legend(fontsize=15)
ax.set_xlabel('Measuring time (mins)', fontsize=20)
ax.set_ylabel('SNR (dB)', fontsize=20)
plt.title('Acetamiprid + Acephate power dependence', fontsize=17)
ax.set_xticks(peak_amplitude_time_log)
ax.set_xscale("log")
ax.xaxis.set_minor_locator(ticker.FixedLocator(peak_amplitude_time_log))
ax.xaxis.set_major_locator(ticker.NullLocator())
ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
plt.show()
plt.savefig('Acetamiprid_acephate_power_snr_evol.png', dpi=500)
i=0
fig, ax = plt.subplots()
for item in range(len(car_power_snr_log)):
    ax.plot(peak_amplitude_time_log, car_aceta_aceph_power_snr_log[i, :], label='Power hwp {}, {}mW'.format(experiment_power_hwp_labels[i], experiment_power_watts_labels[i]))
    i+=1
ax.legend(fontsize=15)
ax.set_xlabel('Measuring time (mins)', fontsize=20)
ax.set_ylabel('SNR (dB)', fontsize=20)
plt.title('Carbendazim + Acetamiprid + Acephate power dependence', fontsize=17)
ax.set_xticks(peak_amplitude_time_log)
ax.set_xscale("log")
ax.xaxis.set_minor_locator(ticker.FixedLocator(peak_amplitude_time_log))
ax.xaxis.set_major_locator(ticker.NullLocator())
ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
plt.show()
plt.savefig('Carbendazim_acetamiprid_acephate_power_snr_evol.png', dpi=500)
np.var(car_power_time_evol_noise, axis=0)
np.var(car_power_time_evol_noise, axis=1)
np.mean(np.var(car_power_time_evol_noise, axis=1))
np.mean(np.var(car_power_time_evol_noise, axis=1), axis=0)
car_intensity = np.sum(car_power_time_evol_mean, axis=-1)
aceta_intensity = np.sum(aceta_power_time_evol_mean, axis=-1)
aceph_intensity = np.sum(aceph_power_time_evol_mean, axis=-1)
car_aceta_intensity = np.sum(car_aceta_power_time_evol_mean, axis=-1)
car_aceph_intensity = np.sum(car_aceph_power_time_evol_mean, axis=-1)
aceta_aceph_intensity = np.sum(aceta_power_time_evol_mean, axis=-1)
car_aceta_aceph_intensity = np.sum(car_aceta_aceph_power_time_evol_mean, axis=-1)
experiment_power_hwp_labels
int(experiment_power_hwp_labels)
experiment_power_hwp = map(int, experiment_power_hwp_labels)
experiment_power_watts = map(int, experiment_power_watts_labels)
i=0
fig, ax = plt.subplots()
for item in range(len(car_power_snr_log)):
    ax.plot(experiment_power_watts, car_intensity[i], label='Power hwp {}, {}mW'.format(experiment_power_hwp_labels[i], experiment_power_watts_labels[i]))
    i+=1
ax.legend(fontsize=15)
ax.set_xlabel('Measuring time (mins)', fontsize=20)
ax.set_ylabel('SNR (dB)', fontsize=20)
plt.title('Carbendazim + Acetamiprid + Acephate power dependence', fontsize=17)
ax.set_xticks(experiment_power_watts_labels)
ax.set_xscale("log")
ax.xaxis.set_minor_locator(ticker.FixedLocator(peak_amplitude_time_log))
ax.xaxis.set_major_locator(ticker.NullLocator())
ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
plt.show()
i=0
fig, ax = plt.subplots()
for item in range(len(car_power_snr_log)):
    ax.plot(experiment_power_watts, car_intensity[i, 0], label='Power hwp {}, {}mW'.format(experiment_power_hwp_labels[i], experiment_power_watts_labels[i]))
    i+=1
ax.legend(fontsize=15)
ax.set_xlabel('Measuring time (mins)', fontsize=20)
ax.set_ylabel('SNR (dB)', fontsize=20)
plt.title('Carbendazim + Acetamiprid + Acephate power dependence', fontsize=17)
ax.set_xticks(experiment_power_watts_labels)
ax.set_xscale("log")
ax.xaxis.set_minor_locator(ticker.FixedLocator(peak_amplitude_time_log))
ax.xaxis.set_major_locator(ticker.NullLocator())
ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
plt.show()
i=0
fig, ax = plt.subplots()
for item in range(len(car_power_snr_log)):
    ax.plot(experiment_power_watts, car_intensity[i, 5], label='Power hwp {}, {}mW'.format(experiment_power_hwp_labels[i], experiment_power_watts_labels[i]))
    i+=1
ax.legend(fontsize=15)
ax.set_xlabel('Measuring time (mins)', fontsize=20)
ax.set_ylabel('SNR (dB)', fontsize=20)
plt.title('Carbendazim + Acetamiprid + Acephate power dependence', fontsize=17)
ax.set_xticks(experiment_power_watts_labels)
ax.set_xscale("log")
ax.xaxis.set_minor_locator(ticker.FixedLocator(peak_amplitude_time_log))
ax.xaxis.set_major_locator(ticker.NullLocator())
ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
plt.show()
i=0
fig, ax = plt.subplots()
for item in range(len(car_power_snr_log)):
    ax.plot(list(experiment_power_watts), car_intensity[i, 5], label='Power hwp {}, {}mW'.format(experiment_power_hwp_labels[i], experiment_power_watts_labels[i]))
    i+=1
ax.legend(fontsize=15)
ax.set_xlabel('Measuring time (mins)', fontsize=20)
ax.set_ylabel('SNR (dB)', fontsize=20)
plt.title('Carbendazim + Acetamiprid + Acephate power dependence', fontsize=17)
ax.set_xticks(experiment_power_watts_labels)
ax.set_xscale("log")
ax.xaxis.set_minor_locator(ticker.FixedLocator(peak_amplitude_time_log))
ax.xaxis.set_major_locator(ticker.NullLocator())
ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
plt.show()
experiment_power_hwp = list(map(int, experiment_power_hwp_labels))
experiment_power_watts = list(map(int, experiment_power_watts_labels))
experiment_power_hwp = list(map(float, experiment_power_hwp_labels))
experiment_power_watts = list(map(float, experiment_power_watts_labels))
i=0
fig, ax = plt.subplots()
for item in range(len(car_power_snr_log)):
    ax.plot(experiment_power_watts, car_intensity[i, 5], label='Power hwp {}, {}mW'.format(experiment_power_hwp_labels[i], experiment_power_watts_labels[i]))
    i+=1
ax.legend(fontsize=15)
ax.set_xlabel('Measuring time (mins)', fontsize=20)
ax.set_ylabel('SNR (dB)', fontsize=20)
plt.title('Carbendazim + Acetamiprid + Acephate power dependence', fontsize=17)
ax.set_xticks(experiment_power_watts_labels)
ax.set_xscale("log")
ax.xaxis.set_minor_locator(ticker.FixedLocator(peak_amplitude_time_log))
ax.xaxis.set_major_locator(ticker.NullLocator())
ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
plt.show()
i=0
fig, ax = plt.subplots()
for item in range(len(car_power_snr_log)):
    ax.plot(experiment_power_watts, car_intensity[i, 5], label='Power hwp {}, {}mW'.format(experiment_power_hwp_labels[i], experiment_power_watts_labels[i]))
    i+=1
ax.legend(fontsize=15)
ax.set_xlabel('Measuring time (mins)', fontsize=20)
ax.set_ylabel('SNR (dB)', fontsize=20)
plt.title('Carbendazim + Acetamiprid + Acephate power dependence', fontsize=17)
ax.set_xticks(experiment_power_watts_labels)
ax.set_xscale("log")
ax.xaxis.set_minor_locator(ticker.FixedLocator(experiment_power_watts))
ax.xaxis.set_major_locator(ticker.NullLocator())
ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
plt.show()
car_intensity[0, 1]
i=0
fig, ax = plt.subplots()
# for item in range(len(car_power_snr_log)):
ax.plot(experiment_power_watts, car_intensity[:, 5], label='Power hwp {}, {}mW'.format(experiment_power_hwp_labels[i], experiment_power_watts_labels[i]))
ax.legend(fontsize=15)
ax.set_xlabel('Measuring time (mins)', fontsize=20)
ax.set_ylabel('SNR (dB)', fontsize=20)
plt.title('Carbendazim + Acetamiprid + Acephate power dependence', fontsize=17)
ax.set_xticks(experiment_power_watts_labels)
ax.set_xscale("log")
ax.xaxis.set_minor_locator(ticker.FixedLocator(experiment_power_watts))
ax.xaxis.set_major_locator(ticker.NullLocator())
ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
plt.show()
i=0
fig, ax = plt.subplots()
# for item in range(len(car_power_snr_log)):
ax.plot(experiment_power_watts, car_intensity[:, 5], label='Power hwp {}, {}mW'.format(experiment_power_hwp_labels[i], experiment_power_watts_labels[i]))
ax.legend(fontsize=15)
ax.set_xlabel('Measuring time (mins)', fontsize=20)
ax.set_ylabel('SNR (dB)', fontsize=20)
plt.title('Carbendazim + Acetamiprid + Acephate power dependence', fontsize=17)
ax.set_xticks(experiment_power_watts_labels)
ax.set_xscale("log")
ax.xaxis.set_minor_locator(ticker.FixedLocator(experiment_power_watts_labels))
ax.xaxis.set_major_locator(ticker.NullLocator())
ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
plt.show()
i=0
fig, ax = plt.subplots()
# for item in range(len(car_power_snr_log)):
ax.plot(experiment_power_watts, car_intensity[:, 5], label='Power hwp {}, {}mW'.format(experiment_power_hwp_labels[i], experiment_power_watts_labels[i]))
ax.legend(fontsize=15)
ax.set_xlabel('Measuring time (mins)', fontsize=20)
ax.set_ylabel('Intensity', fontsize=20)
plt.title('Carbendazim + Acetamiprid + Acephate power dependence', fontsize=17)
ax.set_xticks(experiment_power_watts_labels)
# ax.set_xscale("log")
# ax.xaxis.set_minor_locator(ticker.FixedLocator(experiment_power_watts_labels))
# ax.xaxis.set_major_locator(ticker.NullLocator())
# ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
plt.show()
i=0
fig, ax = plt.subplots()
# for item in range(len(car_power_snr_log)):
ax.plot(experiment_power_watts, car_intensity[:, 5], label='Power hwp {}, {}mW'.format(experiment_power_hwp_labels[i], experiment_power_watts_labels[i]))
ax.legend(fontsize=15)
ax.set_xlabel('Measuring time (mins)', fontsize=20)
ax.set_ylabel('Intensity', fontsize=20)
plt.title('Carbendazim + Acetamiprid + Acephate power dependence', fontsize=17)
ax.set_xticks(experiment_power_watts_labels)
# ax.set_xscale("log")
ax.xaxis.set_minor_locator(ticker.FixedLocator(experiment_power_watts_labels))
ax.xaxis.set_major_locator(ticker.NullLocator())
ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
plt.show()
i=0
fig, ax = plt.subplots()
# for item in range(len(car_power_snr_log)):
ax.plot(experiment_power_watts, car_intensity[:, 5], label='Power hwp {}, {}mW'.format(experiment_power_hwp_labels[i], experiment_power_watts_labels[i]))
ax.legend(fontsize=15)
ax.set_xlabel('mW', fontsize=20)
ax.set_ylabel('Intensity', fontsize=20)
plt.title('Carbendazim power dependence', fontsize=17)
ax.set_xticks(experiment_power_watts_labels)
# ax.set_xscale("log")
ax.xaxis.set_minor_locator(ticker.FixedLocator(experiment_power_watts_labels))
ax.xaxis.set_major_locator(ticker.NullLocator())
ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
plt.show()
plt.figure()
plt.figure()
plt.plot(experiment_power_watts, car_intensity[:, 5])
plt.ylabel("Intensity")
plt.xlabel("Power mW")
plt.title("Carbendazim power")
plt.show()
plt.figure()
plt.plot(experiment_power_watts, car_intensity[:, 5])
plt.ylabel("Intensity", fontsize=20)
plt.xlabel("Power mW", fontsize=20)
plt.title("Carbendazim power", fontsize=17)
plt.show()
plt.plot(f_sup_0810, car_power_time_evol_mean[1, 1])
%history -f console_logs_20210903_v1.py
