import os
import numpy as np
import re
import pandas as pd
from tools.ramanflow.read_data import ReadData as RD
from tools.ramanflow.prep_data import PrepData as PD
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

carbendanzim_directory = 'data/20210810 SERS timed immersion experiment/1'
acetamiprid_directory = 'data/20210810 SERS timed immersion experiment/4'
acephate_directory = 'data/20210810 SERS timed immersion experiment/5'
carbendanzim_acetamiprid_directory = 'data/20210810 SERS timed immersion experiment/1+4'
carbendanzim_acephate_directory = 'data/20210810 SERS timed immersion experiment/1+5'
acetamiprid_acephate_directory = 'data/20210810 SERS timed immersion experiment/4+5'
carbendanzim_acetamiprid_acephate_directory = 'data/20210810 SERS timed immersion experiment/1+4+5'
colloidal_sollution_directory = 'data/20210810 SERS timed immersion experiment/colloidal solution'

list_of_carbendanzim_files = os.listdir(carbendanzim_directory)
list_of_acetamiprid_files = os.listdir(acetamiprid_directory)
list_of_acephate_files = os.listdir(acephate_directory)
list_of_car_aceta_files = os.listdir(carbendanzim_acetamiprid_directory)
list_of_car_aceph_files = os.listdir(carbendanzim_acephate_directory)
list_of_aceta_aceph_files = os.listdir(acetamiprid_acephate_directory)
list_of_car_aceta_aceph_files = os.listdir(carbendanzim_acetamiprid_acephate_directory)

list_of_carbendanzim_variables = [re.findall('.+?(?=\.)', item)[0] for item in list_of_carbendanzim_files]
list_of_acetamiprid_variables = [re.findall('.+?(?=\.)', item)[0] for item in list_of_acetamiprid_files]
list_of_acephate_variables = [re.findall('.+?(?=\.)', item)[0] for item in list_of_acephate_files]
list_of_car_aceta_variables = [re.findall('.+?(?=\.)', item)[0] for item in list_of_car_aceta_files]
list_of_car_aceph_variables = [re.findall('.+?(?=\.)', item)[0] for item in list_of_car_aceph_files]
list_of_aceta_aceph_variables = [re.findall('.+?(?=\.)', item)[0] for item in list_of_aceta_aceph_files]
list_of_car_aceta_aceph_variables = [re.findall('.+?(?=\.)', item)[0] for item in list_of_car_aceta_aceph_files]

carbendanzim_data = {}
for variable_name, item_filename in zip(list_of_carbendanzim_variables, list_of_carbendanzim_files):
    f_sup, carbendanzim_data[variable_name] = RD.read_tiff_file(carbendanzim_directory + '/' + item_filename)
acetamiprid_data = {}
for variable_name, item_filename in zip(list_of_acetamiprid_variables, list_of_acetamiprid_files):
    f_sup, acetamiprid_data[variable_name] = RD.read_tiff_file(acetamiprid_directory + '/' + item_filename)
acephate_data = {}
for variable_name, item_filename in zip(list_of_acephate_variables, list_of_acephate_files):
    f_sup, acephate_data[variable_name] = RD.read_tiff_file(acephate_directory + '/' + item_filename)
car_aceta_data = {}
for variable_name, item_filename in zip(list_of_car_aceta_variables, list_of_car_aceta_files):
    f_sup, car_aceta_data[variable_name] = RD.read_tiff_file(carbendanzim_acetamiprid_directory + '/' + item_filename)
car_aceph_data = {}
for variable_name, item_filename in zip(list_of_car_aceph_variables, list_of_car_aceph_files):
    f_sup, car_aceph_data[variable_name] = RD.read_tiff_file(carbendanzim_acephate_directory + '/' + item_filename)
aceta_aceph_data = {}
for variable_name, item_filename in zip(list_of_aceta_aceph_variables, list_of_aceta_aceph_files):
    f_sup, aceta_aceph_data[variable_name] = RD.read_tiff_file(acetamiprid_acephate_directory + '/' + item_filename)
car_aceta_aceph_data = {}
for variable_name, item_filename in zip(list_of_car_aceta_aceph_variables, list_of_car_aceta_aceph_files):
    f_sup, car_aceta_aceph_data[variable_name] = RD.read_tiff_file(
        carbendanzim_acetamiprid_acephate_directory + '/' + item_filename)


car_power_1_time_evol = ['1_5min_1', '1_30min_1', '1_1hour_1', '1_3hours_1', '1_5hours_1']
aceta_power_1_time_evol = ['4_5min_1', '4_30min_1', '4_1hour_1', '4_3hours_1', '4_5hours_1']
aceph_power_1_time_evol = ['5_5min_1', '5_30min_1', '5_1hour_1', '5_3hours_1', '5_5hours_1']
car_aceta_power_1_time_evol = ['1+4_5min_1', '1+4_30min_1', '1+4_1hour_1', '1+4_3hours_1', '1+4_5hours_1']
car_aceph_power_1_time_evol = ['1+5_5min_1', '1+5_30min_1', '1+5_1hour_1', '1+5_3hours_1', '1+5_5hours_1']
aceta_aceph_power_1_time_evol = ['4+5_5min_1', '4+5_30min_1', '4+5_1hour_1', '4+5_3hours_1', '4+5_5hours_1']
car_aceta_aceph_power_1_time_evol = ['1+4+5_5min_1', '1+4+5_30min_1', '1+4+5_1hour_1', '1+4+5_3hours_1',
                                     '1+4+5_5hours_1']

car_power_1_time_evol_mean = np.zeros((5, 1600))
aceta_power_1_time_evol_mean = np.zeros((5, 1600))
aceph_power_1_time_evol_mean = np.zeros((5, 1600))
car_aceta_power_1_time_evol_mean = np.zeros((5, 1600))
car_aceph_power_1_time_evol_mean = np.zeros((5, 1600))
aceta_aceph_power_1_time_evol_mean = np.zeros((5, 1600))
car_aceta_aceph_power_1_time_evol_mean = np.zeros((5, 1600))

j = 0
for item in car_power_1_time_evol:
    car_power_1_time_evol_mean[j] = np.mean(carbendanzim_data[item], axis=0)
    j += 1
j = 0
for item in aceta_power_1_time_evol:
    aceta_power_1_time_evol_mean[j] = np.mean(acetamiprid_data[item], axis=0)
    j += 1
j = 0
for item in aceph_power_1_time_evol:
    aceph_power_1_time_evol_mean[j] = np.mean(acephate_data[item], axis=0)
    j += 1
j = 0
for item in car_aceta_power_1_time_evol:
    car_aceta_power_1_time_evol_mean[j] = np.mean(car_aceta_data[item], axis=0)
    j += 1

j = 0
for item in car_aceph_power_1_time_evol:
    car_aceph_power_1_time_evol_mean[j] = np.mean(car_aceph_data[item], axis=0)
    j += 1

j = 0
for item in aceta_aceph_power_1_time_evol:
    aceta_aceph_power_1_time_evol_mean[j] = np.mean(aceta_aceph_data[item], axis=0)
    j += 1

j = 0
for item in aceta_aceph_power_1_time_evol:
    aceta_aceph_power_1_time_evol_mean[j] = np.mean(aceta_aceph_data[item], axis=0)
    j += 1
j = 0
for item in car_aceta_aceph_power_1_time_evol:
    car_aceta_aceph_power_1_time_evol_mean[j] = np.mean(car_aceta_aceph_data[item], axis=0)
    j += 1
car_aceta_aceph_power_1_time_evol = ['1+4+5_5min_1', '1+4+5_30min_1', '1+4+5_1hour_1', '1+4+5_3hours_1',
                                     '1+4+5_5hours_1']
j = 0
for item in car_aceta_aceph_power_1_time_evol:
    car_aceta_aceph_power_1_time_evol_mean[j] = np.mean(car_aceta_aceph_data[item], axis=0)
    j += 1

car_aceta_aceph_power_1_time_evol = ['1+4+5_5min_1', '1+4+5_30min_1', '1+4+5_1hour_1', '1+4+5_3hours_1',
                                     '1+4+5_5hours_1']
j = 0
for item in car_aceta_aceph_power_1_time_evol:
    car_aceta_aceph_power_1_time_evol_mean[j] = np.mean(car_aceta_aceph_data[item], axis=0)
    j += 1


car_power_1_time_evol_mean_smooth = np.zeros_like(car_power_1_time_evol_mean)
for i in range(car_power_1_time_evol_mean.shape[0]):
    car_power_1_time_evol_mean_smooth[i] = savgol_filter(car_power_1_time_evol_mean[i], 15, 3)

for i in range(car_power_1_time_evol_mean.shape[0]):
    car_power_1_time_evol_mean_smooth[i] = savgol_filter(car_power_1_time_evol_mean[i], 15, 3)

carbendanzim_directory_24hours = 'data/20210811 SERS timed immersion experiment/1'
acetamiprid_directory_24hours = 'data/20210811 SERS timed immersion experiment/4'
acephate_directory_24hours = 'data/20210811 SERS timed immersion experiment/5'
carbendanzim_acetamiprid_directory_24hours = 'data/20210811 SERS timed immersion experiment/1+4'
carbendanzim_acephate_directory_24hours = 'data/20210811 SERS timed immersion experiment/1+5'
acetamiprid_acephate_directory_24hours = 'data/20210811 SERS timed immersion experiment/4+5'
carbendanzim_acetamiprid_acephate_directory_24hours = 'data/20210811 SERS timed immersion experiment/1+4+5'
list_of_carbendanzim_files_24hours = os.listdir(carbendanzim_directory_24hours)
list_of_acetamiprid_files_24hours = os.listdir(acetamiprid_directory_24hours)
list_of_acephate_files_24hours = os.listdir(acephate_directory_24hours)
list_of_car_aceta_files_24hours = os.listdir(carbendanzim_acetamiprid_directory_24hours)
list_of_car_aceph_files_24hours = os.listdir(carbendanzim_acephate_directory_24hours)
list_of_aceta_aceph_files_24hours = os.listdir(acetamiprid_acephate_directory_24hours)
list_of_car_aceta_aceph_files_24hours = os.listdir(carbendanzim_acetamiprid_acephate_directory_24hours)
list_of_carbendanzim_variables = [re.findall('.+?(?=\.)', item)[0] for item in list_of_carbendanzim_files]
list_of_acetamiprid_variables = [re.findall('.+?(?=\.)', item)[0] for item in list_of_acetamiprid_files]
list_of_acephate_variables = [re.findall('.+?(?=\.)', item)[0] for item in list_of_acephate_files]
list_of_car_aceta_variables = [re.findall('.+?(?=\.)', item)[0] for item in list_of_car_aceta_files]
list_of_car_aceph_variables = [re.findall('.+?(?=\.)', item)[0] for item in list_of_car_aceph_files]
list_of_aceta_aceph_variables = [re.findall('.+?(?=\.)', item)[0] for item in list_of_aceta_aceph_files]
list_of_car_aceta_aceph_variables = [re.findall('.+?(?=\.)', item)[0] for item in list_of_car_aceta_aceph_files]

list_of_carbendanzim_variables_24_hours = [re.findall('.+?(?=\.)', item)[0] for item in
                                           list_of_carbendanzim_files_24hours]
list_of_acetamiprid_variables_24_hours = [re.findall('.+?(?=\.)', item)[0] for item in
                                          list_of_acetamiprid_files_24hours]
list_of_acephate_variables_24_hours = [re.findall('.+?(?=\.)', item)[0] for item in list_of_acephate_files_24hours]
list_of_car_aceta_variables_24_hours = [re.findall('.+?(?=\.)', item)[0] for item in list_of_car_aceta_files_24hours]
list_of_car_aceph_variables_24_hours = [re.findall('.+?(?=\.)', item)[0] for item in list_of_car_aceph_files_24hours]
list_of_aceta_aceph_variables_24_hours = [re.findall('.+?(?=\.)', item)[0] for item in
                                          list_of_aceta_aceph_files_24hours]
list_of_car_aceta_aceph_variables_24_hours = [re.findall('.+?(?=\.)', item)[0] for item in
                                              list_of_car_aceta_aceph_files_24hours]

for variable_name, item_filename in zip(list_of_carbendanzim_variables_24_hours, list_of_carbendanzim_files_24hours):
    f_sup, carbendanzim_data[variable_name] = RD.read_tiff_file(carbendanzim_directory_24hours + '/' + item_filename)

for variable_name, item_filename in zip(list_of_acetamiprid_variables_24_hours, list_of_acetamiprid_files_24hours):
    f_sup, acetamiprid_data[variable_name] = RD.read_tiff_file(acetamiprid_directory_24hours + '/' + item_filename)

for variable_name, item_filename in zip(list_of_acephate_variables_24_hours, list_of_acephate_files_24hours):
    f_sup, acephate_data[variable_name] = RD.read_tiff_file(acephate_directory_24hours + '/' + item_filename)

for variable_name, item_filename in zip(list_of_car_aceta_variables_24_hours, list_of_car_aceta_files_24hours):
    f_sup, car_aceta_data[variable_name] = RD.read_tiff_file(
        carbendanzim_acetamiprid_directory_24hours + '/' + item_filename)

for variable_name, item_filename in zip(list_of_car_aceph_variables_24_hours, list_of_car_aceph_files_24hours):
    f_sup, car_aceph_data[variable_name] = RD.read_tiff_file(
        carbendanzim_acephate_directory_24hours + '/' + item_filename)

for variable_name, item_filename in zip(list_of_aceta_aceph_variables_24_hours, list_of_aceta_aceph_files_24hours):
    f_sup, aceta_aceph_data[variable_name] = RD.read_tiff_file(
        acetamiprid_acephate_directory_24hours + '/' + item_filename)

for variable_name, item_filename in zip(list_of_car_aceta_aceph_variables_24_hours,
                                        list_of_car_aceta_aceph_files_24hours):
    f_sup, car_aceta_aceph_data[variable_name] = RD.read_tiff_file(
        carbendanzim_acetamiprid_acephate_directory_24hours + '/' + item_filename)

power_labels = ['5 min', '30 min', '1 hour', '3 hours', '5 hours', '24 hours']
car_power_1_time_evol = ['1_5min_1', '1_30min_1', '1_1hour_1', '1_3hours_1', '1_5hours_1', '1_24hours_1']
aceta_power_1_time_evol = ['4_5min_1', '4_30min_1', '4_1hour_1', '4_3hours_1', '4_5hours_1', '4_24hours_1']
aceph_power_1_time_evol = ['5_5min_1', '5_30min_1', '5_1hour_1', '5_3hours_1', '5_5hours_1', '5_24hours_1']
car_aceta_power_1_time_evol = ['1+4_5min_1', '1+4_30min_1', '1+4_1hour_1', '1+4_3hours_1', '1+4_5hours_1',
                               '1+4_24hours_1']
car_aceph_power_1_time_evol = ['1+5_5min_1', '1+5_30min_1', '1+5_1hour_1', '1+5_3hours_1', '1+5_5hours_1',
                               '1+5_24hours_1']
aceta_aceph_power_1_time_evol = ['4+5_5min_1', '4+5_30min_1', '4+5_1hour_1', '4+5_3hours_1', '4+5_5hours_1',
                                 '4+5_24hours_1']
car_aceta_aceph_power_1_time_evol = ['1+4+5_5min_1', '1+4+5_30min_1', '1+4+5_1hour_1', '1+4+5_3hours_1',
                                     '1+4+5_5hours_1', '1+4+5_24hours_1']
car_power_1_time_evol_mean = np.zeros((6, 1600))
aceta_power_1_time_evol_mean = np.zeros((6, 1600))
aceph_power_1_time_evol_mean = np.zeros((6, 1600))
car_aceta_power_1_time_evol_mean = np.zeros((6, 1600))
car_aceph_power_1_time_evol_mean = np.zeros((6, 1600))
aceta_aceph_power_1_time_evol_mean = np.zeros((6, 1600))
car_aceta_aceph_power_1_time_evol_mean = np.zeros((6, 1600))

j = 0
for item in car_power_1_time_evol:
    car_power_1_time_evol_mean[j] = np.mean(carbendanzim_data[item], axis=0)
    j += 1
j = 0
for item in aceta_power_1_time_evol:
    aceta_power_1_time_evol_mean[j] = np.mean(acetamiprid_data[item], axis=0)
    j += 1
j = 0
for item in aceph_power_1_time_evol:
    aceph_power_1_time_evol_mean[j] = np.mean(acephate_data[item], axis=0)
    j += 1
j = 0
for item in car_aceta_power_1_time_evol:
    car_aceta_power_1_time_evol_mean[j] = np.mean(car_aceta_data[item], axis=0)
    j += 1
j = 0
for item in car_aceph_power_1_time_evol:
    car_aceph_power_1_time_evol_mean[j] = np.mean(car_aceph_data[item], axis=0)
    j += 1
j = 0
for item in aceta_aceph_power_1_time_evol:
    aceta_aceph_power_1_time_evol_mean[j] = np.mean(aceta_aceph_data[item], axis=0)
    j += 1
j = 0
for item in car_aceta_aceph_power_1_time_evol:
    car_aceta_aceph_power_1_time_evol_mean[j] = np.mean(car_aceta_aceph_data[item], axis=0)
    j += 1

for i in range(len(car_power_1_time_evol_mean)):
    plt.plot(f_sup, car_power_1_time_evol_mean[i])
plt.legend(power_labels)
plt.show()
car_power_1_time_evol_mean_smooth = np.zeros_like(car_power_1_time_evol_mean)
aceta_power_1_time_evol_mean_smooth = np.zeros_like(aceta_power_1_time_evol_mean)
aceph_power_1_time_evol_mean_smooth = np.zeros_like(aceph_power_1_time_evol_mean)
car_aceta_power_1_time_evol_mean_smooth = np.zeros_like(car_aceta_power_1_time_evol_mean)
car_aceph_power_1_time_evol_smooth = np.zeros_like(car_aceph_power_1_time_evol_mean)
aceta_aceph_power_1_time_evol_smooth = np.zeros_like(aceta_aceph_power_1_time_evol_mean)
car_aceta_aceph_power_1_time_evol_smooth = np.zeros_like(car_aceta_aceph_power_1_time_evol_mean)
for i in range(car_power_1_time_evol_mean.shape[0]):
    car_power_1_time_evol_mean_smooth[i] = savgol_filter(car_power_1_time_evol_mean[i], 15, 3)
    aceta_power_1_time_evol_mean_smooth[i] = savgol_filter(aceta_power_1_time_evol_mean[i], 15, 3)
    aceph_power_1_time_evol_mean_smooth[i] = savgol_filter(aceph_power_1_time_evol_mean[i], 15, 3)
    car_aceta_power_1_time_evol_mean_smooth[i] = savgol_filter(car_aceta_power_1_time_evol_mean[i], 15, 3)
    car_aceph_power_1_time_evol_smooth[i] = savgol_filter(car_aceph_power_1_time_evol_mean[i], 15, 3)
    aceta_aceph_power_1_time_evol_smooth[i] = savgol_filter(aceta_aceph_power_1_time_evol_mean[i], 15, 3)
    car_aceta_aceph_power_1_time_evol_smooth[i] = savgol_filter(car_aceta_aceph_power_1_time_evol_mean[i], 15, 3)

for i in range(car_power_1_time_evol_mean.shape[0]):
    car_power_1_time_evol_mean_smooth[i] = savgol_filter(car_power_1_time_evol_mean[i], 15, 3)
    aceta_power_1_time_evol_mean_smooth[i] = savgol_filter(aceta_power_1_time_evol_mean[i], 15, 3)
    aceph_power_1_time_evol_mean_smooth[i] = savgol_filter(aceph_power_1_time_evol_mean[i], 15, 3)
    car_aceta_power_1_time_evol_mean_smooth[i] = savgol_filter(car_aceta_power_1_time_evol_mean[i], 15, 3)
    car_aceph_power_1_time_evol_smooth[i] = savgol_filter(car_aceph_power_1_time_evol_mean[i], 15, 3)
    aceta_aceph_power_1_time_evol_smooth[i] = savgol_filter(aceta_aceph_power_1_time_evol_mean[i], 15, 3)
    car_aceta_aceph_power_1_time_evol_smooth[i] = savgol_filter(car_aceta_aceph_power_1_time_evol_mean[i], 15, 3)

for i in range(len(car_power_1_time_evol_mean)):
    plt.plot(f_sup, car_power_1_time_evol_mean_smooth[i])
plt.legend(power_labels)
plt.show()
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
for i in range(len(car_power_1_time_evol_mean_smooth)):
    plt.plot(f_sup, car_power_1_time_evol_mean_smooth[i])
plt.legend(power_labels)
plt.xlabel('Raman shift cm^-1')
plt.ylabel('Intensity')
plt.xticks(np.arange(f_sup[0], f_sup[-1], 150))
plt.show()
matplotlib.rc('xtick', labelsize=15)
matplotlib.rc('ytick', labelsize=15)



for i in range(len(car_power_1_time_evol_mean_smooth)):
    plt.plot(f_sup, car_power_1_time_evol_mean_smooth[i])
plt.legend(power_labels)
plt.xlabel('Raman shift cm^-1', fontsize=20)
plt.ylabel('Intensity', fontsize=20)
plt.xticks(np.arange(f_sup[0], f_sup[-1], 150))
plt.show()
for i in range(len(car_power_1_time_evol_mean_smooth)):
    plt.plot(f_sup, car_power_1_time_evol_mean_smooth[i])
plt.legend(power_labels)
plt.xlabel('Raman shift cm^-1', fontsize=20)
plt.ylabel('Intensity', fontsize=20)
plt.title('Carbendanzim time evolution at HWP 42 (2.91 mW)', fontsize=17)
plt.xticks(np.arange(f_sup[0], f_sup[-1], 150))
plt.show()
for i in range(len(car_power_1_time_evol_mean_smooth)):
    plt.plot(f_sup, car_power_1_time_evol_mean_smooth[i])
plt.legend(power_labels, fontsize=15)
plt.xlabel('Raman shift cm^-1', fontsize=20)
plt.ylabel('Intensity', fontsize=20)
plt.title('Carbendanzim time evolution at HWP 42 (2.91 mW)', fontsize=17)
plt.xticks(np.arange(f_sup[0], f_sup[-1], 150))
plt.show()
color_label = ['b', 'g', 'r', 'c', 'y', 'k']
color_label = ['b', 'g', 'r', 'c', 'm', 'k']
for i in range(len(car_power_1_time_evol_mean_smooth)):
    plt.plot(f_sup, car_power_1_time_evol_mean_smooth[i], color=color_label[i])
plt.legend(power_labels, fontsize=15)
plt.xlabel('Raman shift cm^-1', fontsize=20)
plt.ylabel('Intensity', fontsize=20)
plt.title('Carbendanzim time evolution at HWP 42 (2.91 mW)', fontsize=17)
plt.xticks(np.arange(f_sup[0], f_sup[-1], 150))
plt.show()
plt.savefig('Carbendanzim_time_evol.png', dpi=500)
# for i in range(car_power_1_time_evol_mean.shape[0]):
#     aceta_power_1_time_evol_mean_smooth[i] = savgol_filter(aceta_power_1_time_evol_mean[i], 15, 3)
#
# plt.plot(f_sup, colloidal_sol_power_1_mean, label='colloidal sol')
# plt.legend()
# plt.show()
for i in range(len(car_power_1_time_evol_mean_smooth)):
    plt.plot(f_sup, aceta_power_1_time_evol_mean_smooth[i], color=color_label[i])
plt.legend(power_labels, fontsize=15)
plt.xlabel('Raman shift cm^-1', fontsize=20)
plt.ylabel('Intensity', fontsize=20)
plt.title('Acetamiprid time evolution at HWP 42 (2.91 mW)', fontsize=17)
plt.xticks(np.arange(f_sup[0], f_sup[-1], 150))
plt.show()
plt.savefig('Acetamiprid_time_evol.png', dpi=500)
for i in range(car_power_1_time_evol_mean.shape[0]):
    aceph_power_1_time_evol_mean_smooth[i] = savgol_filter(aceph_power_1_time_evol_mean[i], 15, 3)

for i in range(len(car_power_1_time_evol_mean_smooth)):
    plt.plot(f_sup, aceph_power_1_time_evol_mean_smooth[i], color=color_label[i])
plt.legend(power_labels, fontsize=15)
plt.xlabel('Raman shift cm^-1', fontsize=20)
plt.ylabel('Intensity', fontsize=20)
plt.title('Acephate time evolution at HWP 42 (2.91 mW)', fontsize=17)
plt.xticks(np.arange(f_sup[0], f_sup[-1], 150))
plt.show()
plt.savefig('Acephate_time_evol.png', dpi=500)
for i in range(len(car_power_1_time_evol_mean_smooth)):
    plt.plot(f_sup, car_aceta_power_1_time_evol_mean_smooth[i], color=color_label[i])
plt.legend(power_labels, fontsize=15)
plt.xlabel('Raman shift cm^-1', fontsize=20)
plt.ylabel('Intensity', fontsize=20)
plt.title('Carbendanzim + Acetamiprid time evolution at HWP 42 (2.91 mW)', fontsize=17)
plt.xticks(np.arange(f_sup[0], f_sup[-1], 150))
plt.show()
plt.savefig('Carbendanzim_acetamiprid_time_evol.png', dpi=500)
for i in range(len(car_power_1_time_evol_mean_smooth)):
    plt.plot(f_sup, car_aceph_power_1_time_evol_smooth[i], color=color_label[i])
plt.legend(power_labels, fontsize=15)
plt.xlabel('Raman shift cm^-1', fontsize=20)
plt.ylabel('Intensity', fontsize=20)
plt.title('Carbendanzim + Acephate time evolution at HWP 42 (2.91 mW)', fontsize=17)
plt.xticks(np.arange(f_sup[0], f_sup[-1], 150))
plt.show()
plt.savefig('Carbendanzim_acephate_time_evol.png', dpi=500)
for i in range(car_power_1_time_evol_mean.shape[0]):
    aceta_aceph_power_1_time_evol_smooth[i] = savgol_filter(aceta_aceph_power_1_time_evol_mean[i], 15, 3)

for i in range(len(car_power_1_time_evol_mean_smooth)):
    plt.plot(f_sup, aceta_aceph_power_1_time_evol_smooth[i], color=color_label[i])
plt.legend(power_labels, fontsize=15)
plt.xlabel('Raman shift cm^-1', fontsize=20)
plt.ylabel('Intensity', fontsize=20)
plt.title('Acetamiprid + Acephate time evolution at HWP 42 (2.91 mW)', fontsize=17)
plt.xticks(np.arange(f_sup[0], f_sup[-1], 150))
plt.show()
plt.savefig('Acetamiprid_acephate_time_evol.png', dpi=500)
for i in range(len(car_power_1_time_evol_mean_smooth)):
    plt.plot(f_sup, car_aceta_aceph_power_1_time_evol_smooth[i], color=color_label[i])
plt.legend(power_labels, fontsize=15)
plt.xlabel('Raman shift cm^-1', fontsize=20)
plt.ylabel('Intensity', fontsize=20)
plt.title('Carbendanzim + Acetamiprid + Acephate time evolution at HWP 42 (2.91 mW)', fontsize=17)
plt.xticks(np.arange(f_sup[0], f_sup[-1], 150))
plt.show()
plt.savefig('Carbendanzim_acetamiprid_acephate_time_evol.png', dpi=500)
aceta_time_evol_max = np.argmax(np.sum(aceta_power_1_time_evol_mean_smooth, axis=-1))
car_time_evol_max = np.argmax(np.sum(car_power_1_time_evol_mean_smooth, axis=-1))
aceph_time_evol_max = np.argmax(np.sum(aceph_power_1_time_evol_mean_smooth, axis=-1))
car_aceta_time_evol_max = np.argmax(np.sum(car_aceta_power_1_time_evol_mean_smooth, axis=-1))
car_aceph_time_evol_max = np.argmax(np.sum(car_aceph_power_1_time_evol_smooth, axis=-1))
aceta_aceph_time_evol_max = np.argmax(np.sum(aceta_aceph_power_1_time_evol_smooth, axis=-1))
car_aceta_aceph_time_evol_max = np.argmax(np.sum(car_aceta_aceph_power_1_time_evol_smooth, axis=-1))