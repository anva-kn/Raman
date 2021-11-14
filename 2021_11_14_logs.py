%load_ext autoreload
%load_ext autoreload
%autoreload 2
from tools.ramanflow.read_data import ReadData as rd
import matplotlib.pyplot as plt
path = 'data/20211112 multiple colloidal SERS test/'
check_f_sup, check = rd.read_dir_tiff_files(path + 'Analyte+C3min(1ml)')
check = rd.read_dir_tiff_files(path + 'Analyte+C3min(1ml)')
rd.read_dir_tiff_files(path + 'Analyte+C3min(1ml)')
f_sup, check = rd.read_data(path + 'Analyte+C3min(1ml)/1+C3min(1ml) 16.35_1.tif')
whos
import matplotlib.pyplot as plt
plt.plot(f_sup, np.mean(check, axis=0))
import numpy as np
plt.plot(f_sup, np.mean(check, axis=0))
f_sup, check_4 = rd.read_data(path + 'Analyte+C30s(1ml)+C3min(1ml)/4+C30s(1ml)+C3min(1ml) 15.45_1.tif')
f_sup, check_5 = rd.read_data(path + 'Analyte+C30s(1ml)+C3min(1ml)/5+C30s(1ml)+C3min(1ml) 15.50_1.tif')
plt.plot(f_sup, np.mean(check_4, axis=0))
plt.plot(f_sup, np.mean(check_5, axis=0))
f_sup, check_mg = rd.read_data(path + 'MG(0.5ml)+C30s(0.4ml)+C3min(0.1ml)/MG+C30s(0.4ml)+C3min(0.1ml) 17.50_2.tif')
plt.plot(f_sup, np.mean(check_mg, axis=0))
plt.plot(f_sup, np.mean(check_mg, axis=0))
f_sup, check_4_0_5 = rd.read_data(path + 'Analyte+C30s(0.5ml)+C3min(0.5ml)/4+C30s(0.5ml)+C3min(0.5ml) 15.25_2.tif')
plt.plot(f_sup, np.mean(check_4_0_5, axis=0))
f_sup, check_5_0_5 = rd.read_data(path + 'Analyte+C30s(0.5ml)+C3min(0.5ml)/5+C30s(0.5ml)+C3min(0.5ml) 15.35_2.tif')
plt.plot(f_sup, np.mean(check_5_0_5, axis=0))
f_sup, check_1_0_5 = rd.read_data(path + 'Analyte+C30s(0.5ml)+C3min(0.5ml)/1+C30s(0.5ml)+C3min(0.5ml) 16.10_2.tif')
plt.plot(f_sup, np.mean(check_1_0_5, axis=0))
%history -f 2021_11_14_logs.py
