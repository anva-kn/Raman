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

noise = mean - best
snr = np.sum(np.abs(signal[0:i]) ** 2) / np.sum(np.abs(noise[i]) ** 2)
snr = 10 * math.log10(snr[i])

for k in range(car_power_time_evol_mean.shape[0]):
    for i in range(car_power_time_evol_mean.shape[1]):
        plt.plot(f_sup_0810, aceta_aceph_power_time_evol_smooth[k, i])
plt.legend(time_labels, fontsize=17)
plt.show()

for i in range(car_power_time_evol_mean.shape[1]):
    plt.plot(f_sup_0810, aceta_aceph_power_time_evol_mean[0, i])
plt.legend(time_labels, fontsize=17)
plt.show()

for i in range(car_power_1_time_evol_mean.shape[0]):
     car_power_1_time_evol_mean_smooth[i] = savgol_filter(car_power_1_time_evol_mean[i], 15, 3)
     aceta_power_1_time_evol_mean_smooth[i] = savgol_filter(aceta_power_1_time_evol_mean[i], 15, 3)
     aceph_power_1_time_evol_mean_smooth[i] = savgol_filter(aceph_power_1_time_evol_mean[i], 15, 3)
     car_aceta_power_1_time_evol_mean_smooth[i] = savgol_filter(car_aceta_power_1_time_evol_mean[i], 15, 3)
     car_aceph_power_1_time_evol_smooth[i] = savgol_filter(car_aceph_power_1_time_evol_mean[i], 15, 3)
     aceta_aceph_power_1_time_evol_smooth[i] = savgol_filter(aceta_aceph_power_1_time_evol_mean[i], 15, 3)
     car_aceta_aceph_power_1_time_evol_smooth[i] = savgol_filter(car_aceta_aceph_power_1_time_evol_mean[i], 15, 3)

car_power_1_time_evol_mean_smooth_snr = car_power_1_time_evol_mean - car_power_1_time_evol_mean_smooth
aceta_power_1_time_evol_mean_smooth_snr = aceta_power_1_time_evol_mean - aceta_power_1_time_evol_mean_smooth
aceph_power_1_time_evol_mean_smooth_snr = aceph_power_1_time_evol_mean - aceph_power_1_time_evol_mean_smooth
car_aceta_power_1_time_evol_mean_smooth_snr = car_aceta_power_1_time_evol_mean - car_aceta_power_1_time_evol_mean_smooth
car_aceph_power_1_time_evol_smooth_snr = car_aceph_power_1_time_evol_mean - car_aceph_power_1_time_evol_smooth
aceta_aceph_power_1_time_evol_smooth_snr = aceta_aceph_power_1_time_evol_mean - aceta_aceph_power_1_time_evol_smooth
car_aceta_aceph_power_1_time_evol_smooth_snr = car_aceta_aceph_power_1_time_evol_mean - car_aceta_aceph_power_1_time_evol_smooth

for i in range(len(car_power_1_time_evol_mean)):
    plt.plot(f_sup_0810, car_power_1_time_evol_mean_smooth_snr[i])
plt.legend(power_labels, fontsize=17)
plt.show()

car_power_1_time_evol_mean_smooth_snr_fft = np.fft.fftshift(np.fft.fft(car_power_1_time_evol_mean_smooth_snr))
aceta_power_1_time_evol_mean_smooth_snr_fft = np.fft.fftshift(np.fft.fft(aceta_power_1_time_evol_mean_smooth_snr))
aceph_power_1_time_evol_mean_smooth_snr_fft = np.fft.fftshift(np.fft.fft(aceph_power_1_time_evol_mean_smooth_snr))
car_aceta_power_1_time_evol_mean_smooth_snr_fft = np.fft.fftshift(np.fft.fft(car_aceta_power_1_time_evol_mean_smooth_snr))
car_aceph_power_1_time_evol_smooth_snr_fft = np.fft.fftshift(np.fft.fft(car_aceph_power_1_time_evol_smooth_snr))
aceta_aceph_power_1_time_evol_smooth_snr_fft = np.fft.fftshift(np.fft.fft(aceta_aceph_power_1_time_evol_smooth_snr))
car_aceta_aceph_power_1_time_evol_smooth_snr_fft = np.fft.fftshift(np.fft.fft(car_aceta_aceph_power_1_time_evol_smooth_snr))

for i in range(len(car_power_1_time_evol_mean)):
    plt.plot(f_sup_0810, car_power_1_time_evol_mean_smooth_snr_fft[i])
plt.legend(power_labels, fontsize=17)
plt.show()

timestep = f_sup_0810[1] - f_sup_0810[0]
n = car_power_1_time_evol_mean_smooth.shape[-1]
fft_freq = np.fft.fftfreq(n, d=timestep)

for i in range(len(car_power_1_time_evol_mean)):
    plt.plot(fft_freq, car_power_1_time_evol_mean_smooth_snr_fft[i])
plt.legend(power_labels, fontsize=17)
plt.show()

np.diff(f_sup_0810)
for i in range(len(car_power_1_time_evol_mean)):
    plt.plot(fft_freq, car_power_1_time_evol_mean_smooth_snr_fft[i])
plt.legend(power_labels, fontsize=17)
plt.show()


def peakdet(v, delta, x=None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html
    
    Returns two arrays
    
    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %      
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.
    
    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.
    
    """
    maxtab = []
    mintab = []
    
    if x is None:
         x = np.arange(len(v))
    
    v = np.asarray(v)
    
    if len(v) != len(x):
         sys.exit('Input vectors v and x must have same length')
    
    if not np.isscalar(delta):
         sys.exit('Input argument delta must be a scalar')
    
    if delta <= 0:
         sys.exit('Input argument delta must be positive')
    
    mn, mx = np.Inf, -(np.Inf)
    mnpos, mxpos = np.NaN, np.NaN
    
    lookformax = True
    
    for i in np.arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
    
    if lookformax:
        if this < mx - delta:
            maxtab.append((mxpos, mx))
            mn = this
            mnpos = x[i]
            lookformax = False
    else:
        if this > mn + delta:
            mintab.append((mnpos, mn))
            mx = this
            mxpos = x[i]
            lookformax = True
    
    return np.array(maxtab), np.array(mintab)


car_peaks_pos = [308, 669, 890, 1104, 1147, 1247, 1345, 1400, 1467]
car_peaks = np.array([212, 444, 586, 723, 751, 815, 878, 913, 956])

i=0
fig, ax = plt.subplots()
for peak in car_peaks:
    ax.plot(peak_amplitude_time_log, car_power_1_time_evol_mean_smooth[:, peak], label='Peak #{}'.format(i+1))
    i+=1
ax.legend(fontsize=15)
ax.set_xlabel('Measuring time (mins)', fontsize=20)
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
fig, ax = plt.subplots()
for peak in car_peaks:
    ax.plot(peak_amplitude_time_log, car_power_1_time_evol_mean_smooth[:, peak], label='Peak #{}'.format(i+1))
    i+=1
ax.legend(fontsize=15)
ax.set_xlabel('Measuring time (mins)', fontsize=20)
ax.set_ylabel('Intensity', fontsize=20)
plt.title('Carbendazim peaks time evolution at HWP 42 (2.91 mW)', fontsize=17)
ax.set_xticks(peak_amplitude_time_log)
ax.set_xscale("log")
ax.xaxis.set_minor_locator(ticker.FixedLocator(peak_amplitude_time_log))
ax.xaxis.set_major_locator(ticker.NullLocator())
ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
plt.show()
_ , new_4_old_coll = RD.read_dir_tiff_files('data/20210813 SERS timed immersion experiment/4+o')
_ , old_4_old_coll = RD.read_dir_tiff_files('data/20210813 SERS timed immersion experiment/4o+o')
_, new_1_old_coll = RD.read_dir_tiff_files('data/20210813 SERS timed immersion experiment/1+o')
_, old_1_old_coll = RD.read_dir_tiff_files('data/20210813 SERS timed immersion experiment/1o+o')
check = {'1+o_1735hour_1', '1o+o_1805hour_1', '4+o_1700hour_1', '4o+o_1725hour_1'}
j = 0
discrepancy_check = np.zeros((4, 1600))
dicrepancy_data = {**new_4_old_coll, **old_4_old_coll, **new_1_old_coll, **old_1_old_coll}
for item in check:
    discrepancy_check[j] = np.mean(dicrepancy_data[item], axis=0)
    j += 1
dicrepancy_labels = ['1+o', '1o+o', '4+o', '4o+o']
for i in range(len(discrepancy_check)):
    plt.plot(f_sup_0813, discrepancy_check[i])
plt.legend(dicrepancy_labels)
plt.show()
dicrepancy_labels = ['1+o', '1o+o', '4+o', '4o+o']
for i in range(len(discrepancy_check)):
    plt.plot(f_sup_0813, discrepancy_check[i])
plt.title('Possible swap of 1 and 4')
plt.legend(dicrepancy_labels)
plt.show()
plt.savefig('possible_swap_1_4.png', dpi = 500)
dicrepancy_labels = ['1+o', '1o+o', '4+o', '4o+o']
for i in range(len(discrepancy_check)):
    plt.plot(f_sup_0813, discrepancy_check[i])
plt.plot(f_sup_0813, car_power_1_time_evol_mean[-2], label='Legit 1')
plt.plot(f_sup_0813, aceta_power_1_time_evol_mean[-2], label='Legit 4')
plt.title('Possible swap of 1 and 4 (8/13 data)')
plt.legend(dicrepancy_labels)
plt.show()
dicrepancy_labels.append('Legit 1')
dicrepancy_labels.append('Legit 4')
dicrepancy_labels = ['1+o', '1o+o', '4+o', '4o+o']
for i in range(len(discrepancy_check)):
    plt.plot(f_sup_0813, discrepancy_check[i])
plt.plot(f_sup_0813, car_power_1_time_evol_mean[-2])
plt.plot(f_sup_0813, aceta_power_1_time_evol_mean[-2])
plt.title('Possible swap of 1 and 4 (8/13 data)')
plt.legend(dicrepancy_labels)
plt.show()
dicrepancy_labels.append('Legit 1')
dicrepancy_labels.append('Legit 4')
for i in range(len(discrepancy_check)):
    plt.plot(f_sup_0813, discrepancy_check[i])
plt.plot(f_sup_0813, car_power_1_time_evol_mean[-2])
plt.plot(f_sup_0813, aceta_power_1_time_evol_mean[-2])
plt.title('Possible swap of 1 and 4 (8/13 data)')
plt.legend(dicrepancy_labels)
plt.show()
plt.savefig('possible_swap_legit.png')
for i in range(len(car_power_1_time_evol_mean_smooth_snr_fft))
    plt.plot(fft_freq, car_power_1_time_evol_mean_smooth_snr_fft[i])
plt.show()
for i in range(len(car_power_1_time_evol_mean_smooth_snr_fft)):
    plt.plot(fft_freq, car_power_1_time_evol_mean_smooth_snr_fft[i])
plt.show()
