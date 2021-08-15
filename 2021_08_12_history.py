print('PyDev console: using IPython 7.22.0\n')

import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['/Users/anvarkunanbaev/PycharmProjects/Raman_2.0/Raman'])
pwd
import os
import numpy as np
import pandas as pd
from tools.ramanflow.read_data import ReadData as RD
from tools.ramanflow.prep_data import PrepData as PD
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
carbendanzim_directory = 'data/20210810 SERS timed immersion experiment/1'
acetamiprid_directory = 'data/20210810 SERS timed immersion experiment/4'
acephate_directory = 'data/20210810 SERS timed immersion experiment/5'
carbendanzim_acetamiprid_directory = 'data/20210810 SERS timed immersion experiment/1+4'
carbendanzim_acephate_directory = 'data/20210810 SERS timed immersion experiment/1+5'
acetamiprid_acephate_directory = 'data/20210810 SERS timed immersion experiment/4+5'
carbendanzim_acetamiprid_acephate_directory = 'data/20210810 SERS timed immersion experiment/1+4+5'
colloidal_sollution_directory = 'data/20210810 SERS timed immersion experiment/colloidal solution'
os.listdir(carbendanzim_directory)
list_of_carbendanzim_files = os.listdir(carbendanzim_directory)
list_of_acetamiprid_files = os.listdir(acetamiprid_directory)
list_of_acephate_files = os.listdir(acephate_directory)
list_of_car_aceta_files = os.listdir(carbendanzim_acetamiprid_directory)
list_of_car_aceph_files = os.listdir(carbendanzim_acephate_directory)
list_of_aceta_aceph_files = os.listdir(acetamiprid_acephate_directory)
list_of_car_aceta_aceph_files= os.listdir(carbendanzim_acetamiprid_acephate_directory)
import re
list_of_carbendanzim_variables = [re.findall('.+?(?=\.)', item)[0] for item in list_of_carbendanzim_files]
list_of_acetamiprid_variables = [re.findall('.+?(?=\.)', item)[0] for item in list_of_acetamiprid_files]
list_of_acephate_variables = [re.findall('.+?(?=\.)', item)[0] for item in list_of_acephate_files]
list_of_car_aceta_variables = [re.findall('.+?(?=\.)', item)[0] for item in list_of_car_aceta_files]
list_of_car_aceph_variables = [re.findall('.+?(?=\.)', item)[0] for item in list_of_car_aceph_files]
list_of_aceta_aceph_variables = [re.findall('.+?(?=\.)', item)[0] for item in list_of_aceta_aceph_files]
list_of_car_aceta_aceph_variables = [re.findall('.+?(?=\.)', item)[0] for item in list_of_car_aceta_aceph_files]
list_of_carbendanzim_variables = [re.findall('.+?(?=\.)', item)[0] for item in list_of_carbendanzim_files]
list_of_acetamiprid_variables = [re.findall('.+?(?=\.)', item)[0] for item in list_of_acetamiprid_files]
list_of_acephate_variables = [re.findall('.+?(?=\.)', item)[0] for item in list_of_acephate_files]
list_of_car_aceta_variables = [re.findall('.+?(?=\.)', item)[0] for item in list_of_car_aceta_files]
list_of_car_aceph_variables = [re.findall('.+?(?=\.)', item)[0] for item in list_of_car_aceph_files]
list_of_aceta_aceph_variables = [re.findall('.+?(?=\.)', item)[0] for item in list_of_aceta_aceph_files]
list_of_car_aceta_aceph_variables = [re.findall('.+?(?=\.)', item)[0] for item in list_of_car_aceta_aceph_files]
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
list_of_car_aceta_aceph_files= os.listdir(carbendanzim_acetamiprid_acephate_directory)
list_of_carbendanzim_variables = [re.findall('.+?(?=\.)', item)[0] for item in list_of_carbendanzim_files]
list_of_acetamiprid_variables = [re.findall('.+?(?=\.)', item)[0] for item in list_of_acetamiprid_files]
list_of_acephate_variables = [re.findall('.+?(?=\.)', item)[0] for item in list_of_acephate_files]
list_of_car_aceta_variables = [re.findall('.+?(?=\.)', item)[0] for item in list_of_car_aceta_files]
list_of_car_aceph_variables = [re.findall('.+?(?=\.)', item)[0] for item in list_of_car_aceph_files]
list_of_aceta_aceph_variables = [re.findall('.+?(?=\.)', item)[0] for item in list_of_aceta_aceph_files]
list_of_car_aceta_aceph_variables = [re.findall('.+?(?=\.)', item)[0] for item in list_of_car_aceta_aceph_files]
list_of_carbendanzim_variables = [re.findall('.+?(?=\.)', item)[0] for item in list_of_carbendanzim_files]
list_of_acetamiprid_variables = [re.findall('.+?(?=\.)', item)[0] for item in list_of_acetamiprid_files]
list_of_acetamiprid_variables = [re.findall('.+?(?=\.)', item_1)[0] for item_1 in list_of_acetamiprid_files]
[re.findall('.+?(?=\.)', item_1)[0] for item_1 in list_of_acetamiprid_files]
list_of_acetamiprid_files
acetamiprid_directory = 'data/20210810 SERS timed immersion experiment/4'
acephate_directory = 'data/20210810 SERS timed immersion experiment/5'
list_of_carbendanzim_files = os.listdir(carbendanzim_directory)
list_of_acetamiprid_files = os.listdir(acetamiprid_directory)
list_of_acephate_files = os.listdir(acephate_directory)
list_of_car_aceta_files = os.listdir(carbendanzim_acetamiprid_directory)
list_of_car_aceph_files = os.listdir(carbendanzim_acephate_directory)
list_of_aceta_aceph_files = os.listdir(acetamiprid_acephate_directory)
list_of_car_aceta_aceph_files= os.listdir(carbendanzim_acetamiprid_acephate_directory)
[re.findall('.+?(?=\.)', item_1)[0] for item_1 in list_of_acetamiprid_files]
list_of_carbendanzim_variables = [re.findall('.+?(?=\.)', item)[0] for item in list_of_carbendanzim_files]
list_of_acetamiprid_variables = [re.findall('.+?(?=\.)', item)[0] for item in list_of_acetamiprid_files]
list_of_acephate_variables = [re.findall('.+?(?=\.)', item)[0] for item in list_of_acephate_files]
list_of_car_aceta_variables = [re.findall('.+?(?=\.)', item)[0] for item in list_of_car_aceta_files]
list_of_car_aceph_variables = [re.findall('.+?(?=\.)', item)[0] for item in list_of_car_aceph_files]
list_of_aceta_aceph_variables = [re.findall('.+?(?=\.)', item)[0] for item in list_of_aceta_aceph_files]
list_of_car_aceta_aceph_variables = [re.findall('.+?(?=\.)', item)[0] for item in list_of_car_aceta_aceph_files]
carbendanzim_data = {}
for variable_name, item_filename in zip(list_of_carbendanzim_variables, list_of_carbendanzim_files):
    carbendanzim_data[variable_name] = RD.read_tiff_files(carbendanzim_directory + '/' +item_filename)
carbendanzim_data = {}
for variable_name, item_filename in zip(list_of_carbendanzim_variables, list_of_carbendanzim_files):
    carbendanzim_data[variable_name] = RD.read_tiff_file(carbendanzim_directory + '/' +item_filename)
acetamiprid_data = {}
for variable_name, item_filename in zip(list_of_acetamiprid_variables, list_of_acetamiprid_files):
    acetamiprid_data[variable_name] = RD.read_tiff_file(acetamiprid_directory + '/' +item_filename)
carbendanzim_data = {}
for variable_name, item_filename in zip(list_of_carbendanzim_variables, list_of_carbendanzim_files):
    f_sup, carbendanzim_data[variable_name] = RD.read_tiff_file(carbendanzim_directory + '/' +item_filename)
    
acetamiprid_data = {}
for variable_name, item_filename in zip(list_of_acetamiprid_variables, list_of_acetamiprid_files):
    f_sup, acetamiprid_data[variable_name] = RD.read_tiff_file(acetamiprid_directory + '/' +item_filename)
acephate_data = {}
for variable_name, item_filename in zip(list_of_acephate_variables, list_of_acephate_files):
    f_sup, acephate_data[variable_name] = RD.read_tiff_file(acephate_directory + '/' +item_filename)
car_aceta_data = {}
for variable_name, item_filename in zip(list_of_car_aceta_variables, list_of_car_aceta_files):
    f_sup, car_aceta_data[variable_name] = RD.read_tiff_file(carbendanzim_acetamiprid_directory + '/' +item_filename)
car_aceph_data = {}
for variable_name, item_filename in zip(list_of_car_aceph_variables, list_of_car_aceph_files):
    f_sup, car_aceph_data[variable_name] = RD.read_tiff_file(carbendanzim_acephate_directory + '/' +item_filename)
aceta_aceph_data = {}
for variable_name, item_filename in zip(list_of_aceta_aceph_variables, list_of_aceta_aceph_files):
    f_sup, aceta_aceph_data[variable_name] = RD.read_tiff_file(acetamiprid_acephate_directory + '/' +item_filename)
car_aceta_aceph_data = {}
for variable_name, item_filename in zip(list_of_car_aceta_aceph_variables, list_of_car_aceta_aceph_files):
    f_sup, car_aceta_aceph_data[variable_name] = RD.read_tiff_file(carbendanzim_acetamiprid_acephate_directory + '/' +item_filename)
carbendanzim_data.keys()
car_power_1_time_evol = ['1_5min_1', '1_30min_1', '1_1hour_1', '1_3hours_1', '1_5hours_1']
car_power_1_time_evol = ['1_5min_1', '1_30min_1', '1_1hour_1', '1_3hours_1', '1_5hours_1']
car_power_1_time_evol = ['1_5min_1', '1_30min_1', '1_1hour_1', '1_3hours_1', '1_5hours_1']
aceta_power_1_time_evol = ['4_5min_1', '4_30min_1', '4_1hour_1', '4_3hours_1', '4_5hours_1']
aceph_power_1_time_evol = ['5_5min_1', '5_30min_1', '5_1hour_1', '5_3hours_1', '5_5hours_1']
car_aceta_power_1_time_evol = ['1+4_5min_1', '1+4_30min_1', '1+4_1hour_1', '1+4_3hours_1', '1+4_5hours_1']
car_aceph_power_1_time_evol = ['1+5_5min_1', '1+5_30min_1', '1+5_1hour_1', '1+5_3hours_1', '1+5_5hours_1']
car_aceta_aceph_power_1_time_evol = ['1+4+5_5min_1', '1+4+5_30min_1', '1+4+5_1hour_1', '1+4+5_3hours_1', '1+4+5_5hours_1']
car_power_1_time_evol_mean = np.zeros((5, 1600))
aceta_power_1_time_evol_mean = np.zeros((5, 1600))
aceph_power_1_time_evol_mean = np.zeros((5, 1600))
car_aceta_power_1_time_evol_mean = np.zeros((5, 1600))
car_aceph_power_1_time_evol_mean = np.zeros((5, 1600))
car_aceta_aceph_power_1_time_evol_mean = np.zeros((5, 1600))
carbendanzim_data.items()
j = 0
for item in car_power_1_time_evol:
    if item == carbendanzim_data.keys():
        car_power_1_time_evol_mean[j] = np.mean(carbendanzim_data[item], axis=0)
        j += 1
for i in range(car_power_1_time_evol_mean.shape[0]):
    plt.plot(f_sup, car_power_1_time_evol_mean[i])
plt.show()
carbendanzim_data[car_power_1_time_evol]
carbendanzim_data[car_power_1_time_evol[0]]
np.mean(carbendanzim_data[car_power_1_time_evol[0]], axis=0)
j = 0
for item in car_power_1_time_evol:
    car_power_1_time_evol_mean[j] = np.mean(carbendanzim_data[item], axis=0)
    j += 1
for i in range(car_power_1_time_evol_mean.shape[0]):
    plt.plot(f_sup, car_power_1_time_evol_mean[i])
plt.show()
car_power_labels = ['5 min', '30 min', '1 hour', '3 hours', '5 hours']
for i in range(car_power_1_time_evol_mean.shape[0]):
    plt.plot(f_sup, car_power_1_time_evol_mean[i])
plt.legend(car_power_labels)
plt.show()
plt.savefig('carbendanzim_power_evol_hwp42.png', dpi=500)
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
car_power_labels = ['5 min', '30 min', '1 hour', '3 hours', '5 hours']
for i in range(car_power_1_time_evol_mean.shape[0]):
    plt.plot(f_sup, aceta_power_1_time_evol_mean[i])
plt.legend(car_power_labels)
plt.show()
car_power_labels = ['5 min', '30 min', '1 hour', '3 hours', '5 hours']
for i in range(car_power_1_time_evol_mean.shape[0]):
    plt.plot(f_sup, aceta_power_1_time_evol_mean[i])
plt.legend(car_power_labels)
plt.show()
plt.savefig('acetamiprid_power_evol_hwp42.png', dpi=500)
car_power_labels = ['5 min', '30 min', '1 hour', '3 hours', '5 hours']
for i in range(car_power_1_time_evol_mean.shape[0]):
    plt.plot(f_sup, aceph_power_1_time_evol_mean[i])
plt.legend(car_power_labels)
plt.show()
car_power_labels = ['5 min', '30 min', '1 hour', '3 hours', '5 hours']
for i in range(car_power_1_time_evol_mean.shape[0]):
    plt.plot(f_sup, aceph_power_1_time_evol_mean[i])
plt.legend(car_power_labels)
plt.show()
plt.savefig('acephate_power_evol_hwp42.png', dpi=500)
j = 0
for item in car_aceph_power_1_time_evol:
    car_aceph_power_1_time_evol_mean[j] = np.mean(car_aceph_data[item], axis=0)
    j += 1
car_power_labels = ['5 min', '30 min', '1 hour', '3 hours', '5 hours']
for i in range(car_power_1_time_evol_mean.shape[0]):
    plt.plot(f_sup, car_aceta_power_1_time_evol_mean[i])
plt.legend(car_power_labels)
plt.show()
car_power_labels = ['5 min', '30 min', '1 hour', '3 hours', '5 hours']
for i in range(car_power_1_time_evol_mean.shape[0]):
    plt.plot(f_sup, car_aceta_power_1_time_evol_mean[i])
plt.legend(car_power_labels)
plt.show()
plt.savefig('car_aceta_power_evol_hwp42.png', dpi=500)
car_power_labels = ['5 min', '30 min', '1 hour', '3 hours', '5 hours']
for i in range(car_power_1_time_evol_mean.shape[0]):
    plt.plot(f_sup, car_aceph_power_1_time_evol_mean[i])
plt.legend(car_power_labels)
plt.show()
plt.savefig('car_aceph_power_evol_hwp42.png', dpi=500)
j = 0
for item in aceta_aceph_power_1_time_evol:
    aceta_aceph_power_1_time_evol_mean[j] = np.mean(aceta_aceph_data[item], axis=0)
    j += 1
aceta_aceph_power_1_time_evol = ['4+5_5min_1', '4+5_30min_1', '4+5_1hour_1', '4+5_3hours_1', '4+5_5hours_1']
aceta_aceph_power_1_time_evol_mean = np.zeros((5, 1600))
j = 0
for item in aceta_aceph_power_1_time_evol:
    aceta_aceph_power_1_time_evol_mean[j] = np.mean(aceta_aceph_data[item], axis=0)
    j += 1
j = 0
for item in car_aceta_aceph_power_1_time_evol:
    car_aceta_aceph_power_1_time_evol_mean[j] = np.mean(car_aceta_aceph_data[item], axis=0)
    j += 1
car_aceta_aceph_power_1_time_evol = ['1+4+5_5min_1', '1+4+5_30_min_1', '1+4+5_1hour_1', '1+4+5_3hours_1', '1+4+5_5hours_1']
j = 0
for item in car_aceta_aceph_power_1_time_evol:
    car_aceta_aceph_power_1_time_evol_mean[j] = np.mean(car_aceta_aceph_data[item], axis=0)
    j += 1
car_power_labels = ['5 min', '30 min', '1 hour', '3 hours', '5 hours']
for i in range(car_power_1_time_evol_mean.shape[0]):
    plt.plot(f_sup, aceta_aceph_power_1_time_evol_mean[i])
plt.legend(car_power_labels)
plt.show()
car_power_labels = ['5 min', '30 min', '1 hour', '3 hours', '5 hours']
for i in range(car_power_1_time_evol_mean.shape[0]):
    plt.plot(f_sup, aceta_aceph_power_1_time_evol_mean[i])
plt.legend(car_power_labels)
plt.show()
plt.savefig('aceta_aceph_power_evol_hwp42.png', dpi=500)
car_power_labels = ['5 min', '30 min', '1 hour', '3 hours', '5 hours']
for i in range(car_power_1_time_evol_mean.shape[0]):
    plt.plot(f_sup, car_aceta_aceph_power_1_time_evol_mean[i])
plt.legend(car_power_labels)
plt.show()
car_power_labels = ['5 min', '30 min', '1 hour', '3 hours', '5 hours']
for i in range(car_power_1_time_evol_mean.shape[0]):
    plt.plot(f_sup, car_aceta_aceph_power_1_time_evol_mean[i])
plt.legend(car_power_labels)
plt.show()
plt.savefig('car_aceta_aceph_power_evol_hwp42.png', dpi=500)
%history -f 2021_08_12_history.py
