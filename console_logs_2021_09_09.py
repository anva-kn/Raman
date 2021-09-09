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

car_intensity = np.sum(car_power_time_evol_mean, axis=-1)
aceta_intensity = np.sum(aceta_power_time_evol_mean, axis=-1)
aceph_intensity = np.sum(aceph_power_time_evol_mean, axis=-1)
car_aceta_intensity = np.sum(car_aceta_power_time_evol_mean, axis=-1)
car_aceph_intensity = np.sum(car_aceph_power_time_evol_mean, axis=-1)
aceta_aceph_intensity = np.sum(aceta_power_time_evol_mean, axis=-1)
car_aceta_aceph_intensity = np.sum(car_aceta_aceph_power_time_evol_mean, axis=-1)

colloidal_power_mean = np.zeros((5, 1600))
_, colloidal_power = RD.read_dir_tiff_files("data/20210810 SERS timed immersion experiment/colloidal solution")

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
colloidal_power_evol = ['colloidal_sol_1', 'colloidal_sol_2', 'colloidal_sol_3', 'colloidal_sol_4', 'colloidal_sol_5']
for idx, item in enumerate(colloidal_power_evol):
    colloidal_power_mean[idx] = np.mean(colloidal_power[item], axis=0)
for i in range(len(colloidal_power_mean)):
    plt.plot(f_sup_0810, colloidal_power_mean[i])
plt.show()
colloidal_power_mean_no_rays = PD.remove_cosmic_rays(colloidal_power_mean, 5)
for i in range(len(colloidal_power_mean)):
    plt.plot(f_sup_0810, colloidal_power_mean_no_rays[i])
plt.show()
colloidal_power_mean_no_rays = PD.remove_cosmic_rays(colloidal_power_mean, 10)
for i in range(len(colloidal_power_mean)):
    plt.plot(f_sup_0810, colloidal_power_mean_no_rays[i])
plt.show()
plt.plot(f_sup_0810, car_power_1_time_evol[1])
plt.plot(f_sup_0810, colloidal_power_mean_no_rays[1])
plt.show()
plt.plot(f_sup_0810, car_power_1_time_evol[1, 3, :])
plt.plot(f_sup_0810, colloidal_power_mean_no_rays[1])
plt.show()
plt.plot(f_sup_0810, car_power_time_evol[1, 3, :])
plt.plot(f_sup_0810, colloidal_power_mean_no_rays[1])
plt.show()
plt.plot(f_sup_0810, car_power_time_evol_mean[1, 3, :])
plt.plot(f_sup_0810, colloidal_power_mean_no_rays[1])
plt.show()
np.dot(car_power_time_evol_mean[1, 3, :], colloidal_power_mean_no_rays[1])
np.dot(car_power_time_evol_mean[1, 3, :], 10*colloidal_power_mean_no_rays[1])
np.dot(car_power_time_evol_mean[1, 3, :])
np.norm(car_power_time_evol_mean[1, 3, :])
np.dot(car_power_time_evol_mean[1, 3, :], car_power_time_evol_mean[1, 3, :])
np.linalg.norm(car_power_time_evol_mean[1, 3, :])
np.inner(car_power_time_evol_mean[1, 3, :],car_power_time_evol_mean[1, 3, :])
similarity = np.dot(car_power_time_evol_mean[1, 3, :], colloidal_power_mean_no_rays[1]) / np.linalg.norm(car_power_time_evol_mean[1, 3, :]) * np.linalg.norm(colloidal_power_mean_no_rays[1])
similarity
best_similarity = 0
for i in range(1, 100, 0.1):
    tmp_similarity = np.dot(car_power_time_evol_mean[1, 3, :], i * colloidal_power_mean_no_rays[1]) / np.linalg.norm(car_power_time_evol_mean[1, 3, :]) * np.linalg.norm(i * colloidal_power_mean_no_rays[1])
    if tmp_similarity > best_similarity:
        best_coefficient = i
best_similarity = 0
for i in range(1, 100, 1):
    tmp_similarity = np.dot(car_power_time_evol_mean[1, 3, :], i * colloidal_power_mean_no_rays[1]) / np.linalg.norm(car_power_time_evol_mean[1, 3, :]) * np.linalg.norm(i * colloidal_power_mean_no_rays[1])
    if tmp_similarity > best_similarity:
        best_coefficient = i
best_coefficient
plt.plot(f_sup_0810, car_power_time_evol_mean[1, 3, :])
plt.plot(f_sup_0810, 99*colloidal_power_mean_no_rays[1])
plt.show()
plt.plot(f_sup_0810, car_power_time_evol_mean[1, 3, :])
plt.plot(f_sup_0810, 10*colloidal_power_mean_no_rays[1])
plt.show()
plt.plot(f_sup_0810, car_power_time_evol_mean[1, 3, :])
plt.plot(f_sup_0810, 12*colloidal_power_mean_no_rays[1])
plt.show()
import scipy
from sklearn.decomposition import NMF
from sklearn.decomposition import NMF
model = NMF(n_components=2, init='random', random_state=0)
W = model.fit_transform(car_power_time_evol_mean[1, :, :])
plt.plot(f_sup_0810[58:], car_power_time_evol_mean[1, 1, 58:])
plt.show()
car_power_time_evol_mean_clipped = car_power_time_evol_mean[:,:,58:]
W = model.fit_transform(car_power_time_evol_mean_clipped[1, :, :])
model = NMF(n_components=2, init='random', random_state=0, max_iter=1000)
W = model.fit_transform(car_power_time_evol_mean_clipped[1, :, :])
H = model.components_
H
W
plt.plot(car_power_time_evol_mean)
plt.plot(car_power_time_evol_mean[1, 3])
plt.show()
plt.plot(f_sup_0810, [1, 3])
plt.plot(f_sup_0810, car_power_time_evol_mean[1, 3])
plt.plot(f_sup_0810, car_power_time_evol_mean[1, 3, 576:1044], 'r--')
plt.plot(f_sup_0810[576:1044], car_power_time_evol_mean[1, 3, 576:1044], 'r--')
car_local_bg = np.polyfit(f_sup_0810[576:1044], car_power_time_evol_mean[:, :, 576:1044])
car_local_bg = np.polyfit(f_sup_0810[576:1044], car_power_time_evol_mean[:, :, 576:1044], deg=2)
car_local_bg = np.polyfit(f_sup_0810[576:1044], car_power_time_evol_mean[0, :, 576:1044], deg=2)
f_sup_0810[576:1044].shape
car_power_time_evol_mean[0,:, 576:1044]
car_power_time_evol_mean[0,:, 576:1044].shape
car_local_bg=[]
for i in range(8):
    car_local_bg[i] = np.polyfit(f_sup_0810[576:1044], car_power_time_evol_mean[0, i, 576:1044], deg=2)
car_local_bg = np.array(car_local_bg)
car_local_bg=[]
for i in range(0, 7):
    car_local_bg[i] = np.polyfit(f_sup_0810[576:1044], car_power_time_evol_mean[0, i, 576:1044], deg=2)
car_local_bg = np.array(car_local_bg)
car_local_bg=[]
for i in range(0, 8):
    car_local_bg.append(np.polyfit(f_sup_0810[576:1044], car_power_time_evol_mean[0, i, 576:1044], deg=2))
car_local_bg = np.array(car_local_bg)
car_local_bg.shape
car_poly_local_bg = np.poly1d(car_local_bg)
car_local_bg
car_power_time_evol_mean[:, :, 723]
car_power_time_evol_mean[:, :, 723].shape
from matplotlib import ticker
power_table = pd.read_csv("data/power_table.csv")
power_hwp = power_table["HWP Angle"].tolist()
power_watts = power_table["no ND"].tolist()
power_hwp_labels = power_table["HWP Angle"].apply(str).tolist()
power_watts_labels = power_table["no ND"].apply(str).tolist()
experiment_power_hwp_labels = ['42', '46', '50', '54', '58']
experiment_power_watts_labels = ['2.71', '6.44', '11.41', '17.24', '23.48']
experiment_power_hwp = list(map(int, experiment_power_hwp_labels))
experiment_power_watts = list(map(int, experiment_power_watts_labels))
experiment_power_hwp = list(map(float, experiment_power_hwp_labels))
experiment_power_watts = list(map(float, experiment_power_watts_labels))
power_table = pd.read_csv("data/power_table.csv")
power_hwp = power_table["HWP Angle"].tolist()
power_watts = power_table["no ND"].tolist()
power_hwp_labels = power_table["HWP Angle"].apply(str).tolist()
power_watts_labels = power_table["no ND"].apply(str).tolist()
experiment_power_hwp_labels = ['42', '46', '50', '54', '58']
experiment_power_watts_labels = ['2.71', '6.44', '11.41', '17.24', '23.48']
experiment_power_hwp = list(map(float, experiment_power_hwp_labels))
experiment_power_watts = list(map(float, experiment_power_watts_labels))
plt.plot(experiment_power_watts, car_power_time_evol_mean[:, 3, 1104])
plt.show()
from matplotlib import ticker
i=0
car_peaks = np.array(723, 751, 815, 913)
fig, ax = plt.subplots()
for peak in range(len(car_peaks)):
    ax.plot(experiment_power_watts, car_power_1_time_evol_mean_smooth[i, 3, car_peaks[i]], label='Peak #{}'.format(i+1))
    i+=1
ax.legend(fontsize=15)
ax.set_xlabel('Power mW', fontsize=20)
ax.set_ylabel('Intensity', fontsize=20)
plt.title('Carbendazim peaks time evolution at HWP 42 (2.91 mW)', fontsize=17)
ax.set_xticks(peak_amplitude_time_log)
ax.set_xscale("log")
ax.xaxis.set_minor_locator(ticker.FixedLocator(peak_amplitude_time_log))
ax.xaxis.set_major_locator(ticker.NullLocator())
ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
plt.show()
from matplotlib import ticker
i=0
car_peaks = np.array([723, 751, 815, 913])
fig, ax = plt.subplots()
for peak in range(len(car_peaks)):
    ax.plot(experiment_power_watts, car_power_1_time_evol_mean_smooth[i, 3, car_peaks[i]], label='Peak #{}'.format(i+1))
    i+=1
ax.legend(fontsize=15)
ax.set_xlabel('Power mW', fontsize=20)
ax.set_ylabel('Intensity', fontsize=20)
plt.title('Carbendazim peaks time evolution at HWP 42 (2.91 mW)', fontsize=17)
ax.set_xticks(peak_amplitude_time_log)
ax.set_xscale("log")
ax.xaxis.set_minor_locator(ticker.FixedLocator(peak_amplitude_time_log))
ax.xaxis.set_major_locator(ticker.NullLocator())
ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
plt.show()
from matplotlib import ticker
i=0
car_peaks = np.array([723, 751, 815, 913])
fig, ax = plt.subplots()
for peak in range(len(car_peaks)):
    ax.plot(experiment_power_watts, car_power_1_time_evol_mean[i, 3, car_peaks[i]], label='Peak #{}'.format(i+1))
    i+=1
ax.legend(fontsize=15)
ax.set_xlabel('Power mW', fontsize=20)
ax.set_ylabel('Intensity', fontsize=20)
plt.title('Carbendazim peaks time evolution at HWP 42 (2.91 mW)', fontsize=17)
ax.set_xticks(peak_amplitude_time_log)
ax.set_xscale("log")
ax.xaxis.set_minor_locator(ticker.FixedLocator(peak_amplitude_time_log))
ax.xaxis.set_major_locator(ticker.NullLocator())
ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
plt.show()
from matplotlib import ticker
i=0
car_peaks = np.array([723, 751, 815, 913])
fig, ax = plt.subplots()
for peak in range(len(car_peaks)):
    ax.plot(experiment_power_watts, car_power_time_evol_mean[i, 3, car_peaks[i]], label='Peak #{}'.format(i+1))
    i+=1
ax.legend(fontsize=15)
ax.set_xlabel('Power mW', fontsize=20)
ax.set_ylabel('Intensity', fontsize=20)
plt.title('Carbendazim peaks time evolution at HWP 42 (2.91 mW)', fontsize=17)
ax.set_xticks(peak_amplitude_time_log)
ax.set_xscale("log")
ax.xaxis.set_minor_locator(ticker.FixedLocator(peak_amplitude_time_log))
ax.xaxis.set_major_locator(ticker.NullLocator())
ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
plt.show()
from matplotlib import ticker
i=0
car_peaks = np.array([723, 751, 815, 913])
fig, ax = plt.subplots()
for peak in range(len(car_peaks)):
    ax.plot(experiment_power_watts, car_power_time_evol_mean[:, 3, car_peaks[i]], label='Peak #{}'.format(i+1))
    i+=1
ax.legend(fontsize=15)
ax.set_xlabel('Power mW', fontsize=20)
ax.set_ylabel('Intensity', fontsize=20)
plt.title('Carbendazim peaks time evolution at HWP 42 (2.91 mW)', fontsize=17)
ax.set_xticks(peak_amplitude_time_log)
ax.set_xscale("log")
ax.xaxis.set_minor_locator(ticker.FixedLocator(peak_amplitude_time_log))
ax.xaxis.set_major_locator(ticker.NullLocator())
ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
plt.show()
from matplotlib import ticker
i=0
car_peaks = np.array([723, 751, 815, 913])
fig, ax = plt.subplots()
for peak in range(len(car_peaks)):
    ax.plot(experiment_power_watts, car_power_time_evol_mean[:, 3, car_peaks[i]], label='Peak #{}'.format(i+1))
    i+=1
ax.legend(fontsize=15)
ax.set_xlabel('Power mW', fontsize=20)
ax.set_ylabel('Intensity', fontsize=20)
plt.title('Carbendazim peaks time evolution at HWP 42 (2.91 mW)', fontsize=17)
ax.set_xticks(power_watts_labels)
ax.xaxis.set_major_locator(ticker.NullLocator())
ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
plt.show()
from matplotlib import ticker
i=0
car_peaks = np.array([723, 751, 815, 913])
fig, ax = plt.subplots()
for peak in range(len(car_peaks)):
    ax.plot(experiment_power_watts, car_power_time_evol_mean[:, 3, car_peaks[i]], label='Peak #{}'.format(i+1))
    i+=1
ax.legend(fontsize=15)
ax.set_xlabel('Power mW', fontsize=20)
ax.set_ylabel('Intensity', fontsize=20)
plt.title('Carbendazim peaks time evolution at HWP 42 (2.91 mW)', fontsize=17)
ax.set_xticks(power_watts_labels)
plt.show()
from matplotlib import ticker
i=0
car_peaks = np.array([723, 751, 815, 913])
fig, ax = plt.subplots()
for peak in range(len(car_peaks)):
    ax.plot(experiment_power_watts, car_power_time_evol_mean[:, 3, car_peaks[i]], label='Peak #{}'.format(i+1))
    i+=1
ax.legend(fontsize=15)
ax.set_xlabel('Power mW', fontsize=20)
ax.set_ylabel('Intensity', fontsize=20)
plt.title('Carbendazim peaks time evolution at HWP 42 (2.91 mW)', fontsize=17)
ax.set_xticks(experiment_power_watts_labels)
plt.show()
from matplotlib import ticker
i=0
car_peaks = np.array([723, 751, 815, 913])
fig, ax = plt.subplots()
for peak in range(len(car_peaks)):
    ax.plot(experiment_power_watts, car_power_time_evol_mean[:, 3, car_peaks[i]], label='Peak #{}'.format(i+1))
    i+=1
ax.legend(fontsize=15)
ax.set_xlabel('Power mW', fontsize=20)
ax.set_ylabel('Intensity', fontsize=20)
plt.title('Carbendazim peaks power dependence', fontsize=17)
ax.set_xticks(experiment_power_watts_labels)
plt.show()
from matplotlib import ticker
i=0
car_peaks = np.array([723, 751, 815, 913])
fig, ax = plt.subplots()
for peak in range(len(car_peaks)):
    ax.plot(experiment_power_watts, car_power_time_evol_mean[:, 3, car_peaks[i]], label='Peak #{}'.format(i+1))
    ax.plot(experiment_power_watts, car_power_time_evol_mean[:, 3, car_peaks[i]], 'r*')
    i+=1
ax.legend(fontsize=15)
ax.set_xlabel('Power mW', fontsize=20)
ax.set_ylabel('Intensity', fontsize=20)
plt.title('Carbendazim peaks power dependence', fontsize=17)
ax.set_xticks(experiment_power_watts_labels)
plt.show()
from matplotlib import ticker
i=0
car_peaks = np.array([723, 751, 815, 913])
fig, ax = plt.subplots(2)
for peak in range(len(car_peaks)):
    ax[0].plot(experiment_power_watts, car_power_time_evol_mean[:, 3, car_peaks[i]], label='Peak #{}'.format(i+1))
    ax[0].plot(experiment_power_watts, car_power_time_evol_mean[:, 3, car_peaks[i]], 'r*')
    ax[1].plot(f_sup_0810, car_power_time_evol_mean[4, 5, :])
    i+=1
ax.legend(fontsize=15)
ax.set_xlabel('Power mW', fontsize=20)
ax.set_ylabel('Intensity', fontsize=20)
plt.title('Carbendazim peaks power dependence', fontsize=17)
ax.set_xticks(experiment_power_watts_labels)
plt.show()
from matplotlib import ticker
i=0
car_peaks = np.array([723, 751, 815, 913])
fig, ax = plt.subplots(2)
for peak in range(len(car_peaks)):
    ax[0].plot(experiment_power_watts, car_power_time_evol_mean[:, 3, car_peaks[i]], label='Peak #{}'.format(i+1))
    ax[0].plot(experiment_power_watts, car_power_time_evol_mean[:, 3, car_peaks[i]], 'r*')
    i+=1
ax[0].legend(fontsize=15)
ax.set_xlabel('Power mW', fontsize=20)
ax.set_ylabel('Intensity', fontsize=20)
plt.title('Carbendazim peaks power dependence', fontsize=17)
ax.set_xticks(experiment_power_watts_labels)
plt.show()
from matplotlib import ticker
i=0
car_peaks = np.array([723, 751, 815, 913])
fig, ax = plt.subplots()
for peak in range(len(car_peaks)):
    ax.plot(experiment_power_watts, car_power_time_evol_mean[:, 3, car_peaks[i]], label='Peak #{}'.format(i+1))
    ax.plot(experiment_power_watts, car_power_time_evol_mean[:, 3, car_peaks[i]], 'r*')
    i+=1
ax[0].legend(fontsize=15)
ax.set_xlabel('Power mW', fontsize=20)
ax.set_ylabel('Intensity', fontsize=20)
plt.title('Carbendazim peaks power dependence', fontsize=17)
ax.set_xticks(experiment_power_watts_labels)
plt.show()
from matplotlib import ticker
i=0
car_peaks = np.array([723, 751, 815, 913])
fig, ax = plt.subplots()
for peak in range(len(car_peaks)):
    ax.plot(experiment_power_watts, car_power_time_evol_mean[:, 3, car_peaks[i]], label='Peak #{}'.format(i+1))
    ax.plot(experiment_power_watts, car_power_time_evol_mean[:, 3, car_peaks[i]], 'r*')
    i+=1
ax.legend(fontsize=15)
ax.set_xlabel('Power mW', fontsize=20)
ax.set_ylabel('Intensity', fontsize=20)
plt.title('Carbendazim peaks power dependence', fontsize=17)
ax.set_xticks(experiment_power_watts_labels)
plt.show()
from matplotlib import ticker
i=0
car_peaks = np.array([723, 751, 815, 913])
fig, ax = plt.subplots()
for peak in range(len(car_peaks)):
    ax.plot(experiment_power_watts, car_power_time_evol_mean[:, 3, car_peaks[i]], label='Peak at {}'.format(str(car_peaks[i])))
    ax.plot(experiment_power_watts, car_power_time_evol_mean[:, 3, car_peaks[i]], 'r*')
    i+=1
ax.legend(fontsize=15)
ax.set_xlabel('Power mW', fontsize=20)
ax.set_ylabel('Intensity', fontsize=20)
plt.title('Carbendazim peaks power dependence', fontsize=17)
ax.set_xticks(experiment_power_watts_labels)
plt.show()
i=0
car_peaks = np.array([723, 751, 815, 913])
fig, ax = plt.subplots()
for peak in range(len(car_peaks)):
    ax.plot(experiment_power_watts, car_power_time_evol_mean[:, 3, car_peaks[i]], label='Peak #{}'.format(i+1))
    ax.plot(experiment_power_watts, car_power_time_evol_mean[:, 3, car_peaks[i]], 'r*')
    i+=1
ax.legend(fontsize=15)
ax.set_xlabel('Power mW', fontsize=20)
ax.set_ylabel('Intensity', fontsize=20)
plt.title('Carbendazim peaks power dependence', fontsize=17)
ax.set_xticks(experiment_power_watts_labels)
plt.show()
plt.savefig("Car_peaks.png", dpi=500)
aceta_peaks = [360, 898, 1371]
aceph_peaks = [273, 377, 1102]
car_aceta = [**car_peaks, **aceta_peaks]
car_aceta_peaks = car_peaks+aceta_peaks
car_peaks = [723, 751, 815, 913]
car_aceta_peaks = car_peaks+aceta_peaks
car_aceph_peaks = car_peaks+aceph_peaks
aceta_aceph_peaks = aceta_peaks + aceph_peaks
car_aceta_aceph_peaks = car_peaks+ aceta_peaks + aceph_peaks
i=0
car_peaks = np.array(aceta_peaks)
fig, ax = plt.subplots()
for peak in range(len(car_peaks)):
    ax.plot(experiment_power_watts, car_power_time_evol_mean[:, 3, aceta_peaks[i]], label='Peak #{}'.format(i+1))
    ax.plot(experiment_power_watts, car_power_time_evol_mean[:, 3, aceta_peaks[i]], 'r*')
    i+=1
ax.legend(fontsize=15)
ax.set_xlabel('Power mW', fontsize=20)
ax.set_ylabel('Intensity', fontsize=20)
plt.title('Acetamiprid peaks power dependence', fontsize=17)
ax.set_xticks(experiment_power_watts_labels)
plt.show()
plt.savefig("Aceta_peaks.png", dpi=500)
i=0
car_peaks = np.array(aceph_peaks)
fig, ax = plt.subplots()
for peak in range(len(car_peaks)):
    ax.plot(experiment_power_watts, aceph_power_time_evol_mean[:, 3, aceph_peaks[i]], label='Peak #{}'.format(i+1))
    ax.plot(experiment_power_watts, aceph_power_time_evol_mean[:, 3, aceph_peaks[i]], 'r*')
    i+=1
ax.legend(fontsize=15)
ax.set_xlabel('Power mW', fontsize=20)
ax.set_ylabel('Intensity', fontsize=20)
plt.title('Acephate peaks power dependence', fontsize=17)
ax.set_xticks(experiment_power_watts_labels)
plt.show()
plt.savefig("Aceph_peaks.png", dpi=500)
i=0
car_peaks = np.array(aceta_peaks)
fig, ax = plt.subplots()
for peak in range(len(car_peaks)):
    ax.plot(experiment_power_watts, aceta_power_time_evol_mean[:, 3, aceta_peaks[i]], label='Peak #{}'.format(i+1))
    ax.plot(experiment_power_watts, aceta_power_time_evol_mean[:, 3, aceta_peaks[i]], 'r*')
    i+=1
ax.legend(fontsize=15)
ax.set_xlabel('Power mW', fontsize=20)
ax.set_ylabel('Intensity', fontsize=20)
plt.title('Acetamiprid peaks power dependence', fontsize=17)
ax.set_xticks(experiment_power_watts_labels)
plt.show()
plt.savefig("Aceta_peaks.png", dpi=500)
i=0
fig, ax = plt.subplots()
for peak in range(len(car_peaks)):
    ax.plot(experiment_power_watts, car_aceta_power_time_evol_mean[:, 3, car_aceta_peaks[i]], label='Peak #{}'.format(i+1))
    ax.plot(experiment_power_watts, car_aceta_power_time_evol_mean[:, 3, car_aceta_peaks[i]], 'r*')
    i+=1
ax.legend(fontsize=15)
ax.set_xlabel('Power mW', fontsize=20)
ax.set_ylabel('Intensity', fontsize=20)
plt.title('Carbendanzim + Acetamiprid peaks power dependence', fontsize=17)
ax.set_xticks(experiment_power_watts_labels)
plt.show()
plt.savefig("car_aceta_peaks.png", dpi=500)
i=0
fig, ax = plt.subplots()
for peak in range(len(car_aceph_peaks)):
    ax.plot(experiment_power_watts, car_aceph_power_time_evol_mean[:, 3, car_aceph_peaks[i]], label='Peak #{}'.format(i+1))
    ax.plot(experiment_power_watts, car_aceph_power_time_evol_mean[:, 3, car_aceph_peaks[i]], 'r*')
    i+=1
ax.legend(fontsize=15)
ax.set_xlabel('Power mW', fontsize=20)
ax.set_ylabel('Intensity', fontsize=20)
plt.title('Carbendanzim + Acephate peaks power dependence', fontsize=17)
ax.set_xticks(experiment_power_watts_labels)
plt.show()
plt.savefig("car_aceph_peaks.png", dpi=500)
i=0
fig, ax = plt.subplots()
for peak in range(len(car_aceta_peaks)):
    ax.plot(experiment_power_watts, car_aceta_power_time_evol_mean[:, 3, car_aceta_peaks[i]], label='Peak #{}'.format(i+1))
    ax.plot(experiment_power_watts, car_aceta_power_time_evol_mean[:, 3, car_aceta_peaks[i]], 'r*')
    i+=1
ax.legend(fontsize=15)
ax.set_xlabel('Power mW', fontsize=20)
ax.set_ylabel('Intensity', fontsize=20)
plt.title('Carbendanzim + Acetamiprid peaks power dependence', fontsize=17)
ax.set_xticks(experiment_power_watts_labels)
plt.show()
plt.savefig("car_aceta_peaks.png", dpi=500)
i=0
fig, ax = plt.subplots()
for peak in range(len(car_aceta_aceph_peaks)):
    ax.plot(experiment_power_watts, car_aceta_aceph_power_time_evol_mean[:, 3, car_aceta_aceph_peaks[i]], label='Peak #{}'.format(i+1))
    ax.plot(experiment_power_watts, car_aceta_aceph_power_time_evol_mean[:, 3, car_aceta_aceph_peaks[i]], 'r*')
    i+=1
ax.legend(fontsize=15)
ax.set_xlabel('Power mW', fontsize=20)
ax.set_ylabel('Intensity', fontsize=20)
plt.title('Carbendanzim + Acetamiprid + Acephate peaks power dependence', fontsize=17)
ax.set_xticks(experiment_power_watts_labels)
plt.show()
plt.savefig("car_aceta_aceph_peaks.png", dpi=500)
%history -f console_logs_2021_09_09.py
