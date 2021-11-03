%load_ext autoreload
%autoreload 2
from tools.ramanflow.read_data import ReadData as rd
rd.read_data('data/20211029 multiple colloidal SERS test/1_new/1_0s_aftermixing_2_min_coll_6X6_spectral_mapping_1s_msrmnt_4.tif')
rd.read_data('data/20211029 multiple colloidal SERS test/1_new/1_0s_aftermixing_2_min_coll_6X6_spectral_mapping_1s_msrmnt_4.tif')
rd.read_data('data/20211029 multiple colloidal SERS test/1_new/1_0s_aftermixing_2_min_coll_6X6_spectral_mapping_1s_msrmnt_4.tif')
rd.read_data('data/20211029 multiple colloidal SERS test/1_new/1_0s_aftermixing_2_min_coll_6X6_spectral_mapping_1s_msrmnt_4.tif')
rd.read_data('data/20211029 multiple colloidal SERS test/1_new/1_0s_aftermixing_2_min_coll_6X6_spectral_mapping_1s_msrmnt_4.tif')
f_sup, data = rd.read_data('data/20211029 multiple colloidal SERS test/1_new/1_0s_aftermixing_2_min_coll_6X6_spectral_mapping_1s_msrmnt_4.tif')
f_sup
f_sup.shape
data.shape
plt.plot(f_sup, data[1])
import matplotlib.pyplot as plt
plt.plot(f_sup, data[1])
plt.figure(figsize=(10,10))
plt.plot(f_sup, data[1])
plt.show()
plt.figure(figsize=(10,10))
plt.plot(f_sup, data[i])
plt.show()
plt.figure(figsize=(10,10))
plt.plot(f_sup, data[1])
plt.show()
x_data, y_data = rd.read_data('data/20211029 multiple colloidal SERS test/absorbance/3_min_1_to_4_colloids.txt')
x_data, y_data = rd.read_data('data/20211029 multiple colloidal SERS test/absorbance/3_min_1_to_4_colloids.txt')
x_data
plt.plot(x_data, y_data)
x_data, y_data_20sec = rd.read_data('data/20211029 multiple colloidal SERS test/absorbance/20_sec_1_to_4_colloids.txt')
x_data, y_data_30sec = rd.read_data('data/20211029 multiple colloidal SERS test/absorbance/30_sec_1_to_4_colloids.txt')
x_data, y_data_1min = rd.read_data('data/20211029 multiple colloidal SERS test/absorbance/1_min_1_to_4_colloids.txt')
x_data, y_data_2min = rd.read_data('data/20211029 multiple colloidal SERS test/absorbance/2_min_1_to_4_colloids.txt')
x_data, y_data_3min = rd.read_data('data/20211029 multiple colloidal SERS test/absorbance/3_min_1_to_4_colloids.txt')
x_data, y_data_10min = rd.read_data('data/20211029 multiple colloidal SERS test/absorbance/10_min_1_to_4_colloids.txt')
plt.figure(figsize=(15,10))
plt.plot(x_data, y_data_20sec, label='20 sec')
plt.plot(x_data, y_data_30sec, label='30 sec')
plt.plot(x_data, y_data_1min, label='1 min')
plt.plot(x_data, y_data_2min, label='2 min')
plt.plot(x_data, y_data_3min, label='3 min')
plt.plot(x_data, y_data_10min, label='10 min')
plt.legend()
plt.show()
plt.figure(figsize=(15,10))
plt.plot(x_data, y_data_20sec, label='20 sec')
plt.plot(x_data, y_data_30sec, label='30 sec')
plt.plot(x_data, y_data_1min, label='1 min')
plt.plot(x_data, y_data_2min, label='2 min')
plt.plot(x_data, y_data_3min, label='3 min')
plt.plot(x_data, y_data_10min, label='10 min')
plt.legend()
plt.savefig('uv_comparison_init.png', dpi=500)
plt.figure(figsize=(15,10))
plt.plot(x_data, y_data_20sec, label='20 sec')
plt.plot(x_data, y_data_30sec, 'go', label='30 sec')
plt.plot(x_data, y_data_1min, 'g*', label='1 min')
plt.plot(x_data, y_data_2min, 'ro', label='2 min')
plt.plot(x_data, y_data_3min, 'r*', label='3 min')
plt.plot(x_data, y_data_10min, label='10 min')
plt.legend()
plt.show()
import numpy as np
np.where(np.max(y_data_20sec))
np.where(np.max(y_data_20sec) == y_data_20sec)
np.where(np.max(y_data_30sec) == y_data_30sec)
np.where(np.max(y_data_1min) == y_data_1min)
np.where(np.max(y_data_2min) == y_data_2min)
np.where(np.max(y_data_3min) == y_data_3min)
np.where(np.max(y_data_10min) == y_data_10min)
x_data[55]
x_data[52]
plt.figure(figsize=(15,10))
plt.plot(x_data, y_data_20sec, label='20 sec')
plt.plot(x_data, y_data_30sec, 'go', label='30 sec')
plt.plot(x_data, y_data_1min, 'g*', label='1 min')
plt.plot(x_data, y_data_2min, 'ro', label='2 min')
plt.plot(x_data, y_data_3min, 'r*', label='3 min')
plt.plot(x_data, y_data_10min, label='10 min')
plt.vlines(x_data[52], ymin=np.min(y_data_2min), ymax=np.max(y_data_1min), colors='k', linestyles='dashed')
plt.legend()
plt.show()
plt.figure(figsize=(15,10))
plt.plot(x_data, y_data_20sec, label='20 sec')
plt.plot(x_data, y_data_30sec, 'go', label='30 sec')
plt.plot(x_data, y_data_1min, 'g*', label='1 min')
plt.plot(x_data, y_data_2min, 'ro', label='2 min')
plt.plot(x_data, y_data_3min, 'r*', label='3 min')
plt.plot(x_data, y_data_10min, label='10 min')
plt.vlines(x_data[52], ymin=np.min(y_data_2min), ymax=np.max(y_data_1min), colors='k', linestyles='dashed')
plt.vlines(x_data[55], ymin=np.min(y_data_2min), ymax=np.max(y_data_1min), colors='m', linestyles='solid')
plt.legend()
plt.show()
plt.figure(figsize=(15,10))
plt.plot(x_data, y_data_20sec, label='20 sec')
plt.plot(x_data, y_data_30sec, 'go', label='30 sec')
plt.plot(x_data, y_data_1min, 'g*', label='1 min')
plt.plot(x_data, y_data_2min, 'ro', label='2 min')
plt.plot(x_data, y_data_3min, 'r*', label='3 min')
plt.plot(x_data, y_data_10min, label='10 min')
plt.vlines(x_data[52], ymin=np.min(y_data_2min), ymax=np.max(y_data_1min), colors='k', linestyles='dashed')
plt.vlines(x_data[55], ymin=np.min(y_data_2min), ymax=np.max(y_data_1min), colors='m', linestyles='solid')
plt.legend()
plt.title('UV absorbance comparison', fontsize=16)
plt.xlabel('wavenumber', fontsize = 16)
plt.ylabel('Aba %', fontsize = 16)
plt.show()
plt.figure(figsize=(15,10))
plt.plot(x_data, y_data_20sec, label='20 sec')
plt.plot(x_data, y_data_30sec, 'go', label='30 sec')
plt.plot(x_data, y_data_1min, 'g*', label='1 min')
plt.plot(x_data, y_data_2min, 'ro', label='2 min')
plt.plot(x_data, y_data_3min, 'r*', label='3 min')
plt.plot(x_data, y_data_10min, label='10 min')
plt.vlines(x_data[52], ymin=np.min(y_data_2min), ymax=np.max(y_data_1min), colors='k', linestyles='dashed')
plt.vlines(x_data[55], ymin=np.min(y_data_2min), ymax=np.max(y_data_1min), colors='m', linestyles='solid')
plt.legend()
plt.title('UV absorbance comparison', fontsize=16)
plt.xlabel('wavenumber', fontsize = 16)
plt.ylabel('Abs %', fontsize = 16)
plt.show()
plt.figure(figsize=(15,10))
plt.plot(x_data, y_data_20sec, label='20 sec')
plt.plot(x_data, y_data_30sec, 'go', label='30 sec')
plt.plot(x_data, y_data_1min, 'g*', label='1 min')
plt.plot(x_data, y_data_2min, 'ro', label='2 min')
plt.plot(x_data, y_data_3min, 'r*', label='3 min')
plt.plot(x_data, y_data_10min, label='10 min')
plt.vlines(x_data[52], ymin=np.min(y_data_2min), ymax=np.max(y_data_1min), colors='k', linestyles='dashed')
plt.vlines(x_data[55], ymin=np.min(y_data_2min), ymax=np.max(y_data_1min), colors='m', linestyles='solid')
plt.legend()
plt.title('UV absorbance comparison', fontsize=16)
plt.xlabel('wavenumber', fontsize = 16)
plt.ylabel('Abs %', fontsize = 16)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
plt.figure(figsize=(15,10))
plt.plot(x_data, y_data_20sec, label='20 sec')
plt.plot(x_data, y_data_30sec, 'go', label='30 sec')
plt.plot(x_data, y_data_1min, 'g*', label='1 min')
plt.plot(x_data, y_data_2min, 'ro', label='2 min')
plt.plot(x_data, y_data_3min, 'r*', label='3 min')
plt.plot(x_data, y_data_10min, label='10 min')
plt.vlines(x_data[52], ymin=np.min(y_data_2min), ymax=np.max(y_data_1min), colors='k', linestyles='dashed')
plt.vlines(x_data[55], ymin=np.min(y_data_2min), ymax=np.max(y_data_1min), colors='m', linestyles='solid')
plt.legend()
plt.title('UV absorbance comparison', fontsize=16)
plt.xlabel('wavenumber', fontsize = 16)
plt.ylabel('Abs %', fontsize = 16)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('uv_comparison_v2.png', dpi=600)
plt.figure(figsize=(15,10))
plt.plot(x_data, y_data_20sec, label='20 sec')
plt.plot(x_data, y_data_30sec, 'go', label='30 sec')
plt.plot(x_data, y_data_1min, 'g*', label='1 min')
plt.plot(x_data, y_data_2min, 'ro', label='2 min')
plt.plot(x_data, y_data_3min, 'r*', label='3 min')
plt.plot(x_data, y_data_10min, label='10 min')
plt.vlines(x_data[52], ymin=np.min(y_data_2min), ymax=np.max(y_data_1min), colors='k', linestyles='dashed')
plt.vlines(x_data[55], ymin=np.min(y_data_20sec), ymax=np.max(y_data_1min), colors='m', linestyles='solid')
plt.legend()
plt.title('UV absorbance comparison', fontsize=16)
plt.xlabel('wavenumber', fontsize = 16)
plt.ylabel('Abs %', fontsize = 16)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('uv_comparison_v2.png', dpi=600)
plt.figure(figsize=(15,10))
plt.plot(x_data, y_data_20sec, label='20 sec')
plt.plot(x_data, y_data_30sec, 'go', label='30 sec')
plt.plot(x_data, y_data_1min, 'g*', label='1 min')
plt.plot(x_data, y_data_2min, 'ro', label='2 min')
plt.plot(x_data, y_data_3min, 'r*', label='3 min')
plt.plot(x_data, y_data_10min, label='10 min')
plt.vlines(x_data[52], ymin=np.min(y_data_2min), ymax=np.max(y_data_1min), colors='k', linestyles='dashed')
plt.vlines(x_data[55], ymin=np.min(y_data_2min), ymax=np.max(y_data_20sec), colors='m', linestyles='solid')
plt.legend()
plt.title('UV absorbance comparison', fontsize=16)
plt.xlabel('wavenumber', fontsize = 16)
plt.ylabel('Abs %', fontsize = 16)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('uv_comparison_v2.png', dpi=600)
path = 'data/20211029 multiple colloidal SERS test/'
f_sup, car_20sec = rd.read_data(path + '1_new/1_0s_aftermixing_20_sec_coll_6X6_spectral_mapping_1s_msrmnt_4.tif')
car_20sec_mean = np.mean(car_20sec, axis=0)
plt.plot(f_sup, car_20sec_mean)
f_sup, car_30sec = rd.read_data(path + '1_new/1_0s_aftermixing_30_sec_coll_6X6_spectral_mapping_1s_msrmnt_4.tif')
car_30sec_mean = np.mean(car_30sec, axis=0)
f_sup, car_1min = rd.read_data(path + '1_new/1_0s_aftermixing_1_min_coll_6X6_spectral_mapping_1s_msrmnt_4.tif')
car_1min_mean = np.mean(car_1min, axis=0)
f_sup, car_2min = rd.read_data(path + '1_new/1_0s_aftermixing_2_min_coll_6X6_spectral_mapping_1s_msrmnt_4.tif')
car_2min_mean = np.mean(car_2min, axis=0)
f_sup, car_3min = rd.read_data(path + '1_new/1_0s_aftermixing_3_min_coll_6X6_spectral_mapping_1s_msrmnt_4.tif')
f_sup, car_10min = rd.read_data(path + '1_new/1_0s_aftermixing_10_min_coll_6X6_spectral_mapping_1s_msrmnt_4.tif')
car_3min_mean = np.mean(car_3min, axis=0)
car_10min_mean = np.mean(car_10min, axis=0)
plt.figure(figsize=(15,10))
plt.plot(f_sup, car_20sec_mean / np.max(car_20sec_mean), label='20 sec')
plt.plot(f_sup, car_30sec_mean / np.max(car_30sec_mean), label='30 sec')
plt.plot(f_sup, car_1min_mean / np.max(car_1min_mean), label='1 min')
plt.plot(f_sup, car_2min_mean / np.max(car_2min_mean), label='2 min')
plt.plot(f_sup, car_3min_mean / np.max(car_3min_mean), label='3 min')
plt.plot(f_sup, car_10min_mean / np.max(car_10min_mean), label='10 min')
plt.legend()
plt.title('Carbendazim with different colloidal sol.', fontsize=16)
plt.xlabel('Raman Shift', fontsize = 16)
plt.ylabel('Normalized intenisty', fontsize = 16)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
plt.figure(figsize=(15,10))
plt.plot(f_sup, car_30sec_mean / np.max(car_30sec_mean), label='30 sec')
plt.plot(f_sup, car_1min_mean / np.max(car_1min_mean), label='1 min')
plt.legend()
plt.title('Carbendazim with different colloidal sol.', fontsize=16)
plt.xlabel('Raman Shift', fontsize = 16)
plt.ylabel('Normalized intenisty', fontsize = 16)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()
plt.figure(figsize=(15,10))
plt.plot(f_sup, car_20sec_mean / np.max(car_20sec_mean), label='20 sec')
plt.plot(f_sup, car_30sec_mean / np.max(car_30sec_mean), label='30 sec')
plt.plot(f_sup, car_1min_mean / np.max(car_1min_mean), label='1 min')
plt.plot(f_sup, car_2min_mean / np.max(car_2min_mean), label='2 min')
plt.plot(f_sup, car_3min_mean / np.max(car_3min_mean), label='3 min')
plt.plot(f_sup, car_10min_mean / np.max(car_10min_mean), label='10 min')
plt.legend()
plt.title('Carbendazim with different colloidal sol.', fontsize=16)
plt.xlabel('Raman Shift', fontsize = 16)
plt.ylabel('Normalized intenisty', fontsize = 16)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('car_all_comparison.png', dpi=600)
plt.figure(figsize=(15,10))
plt.plot(f_sup, car_30sec_mean / np.max(car_30sec_mean), label='30 sec')
plt.plot(f_sup, car_1min_mean / np.max(car_1min_mean), label='1 min')
plt.legend()
plt.title('Carbendazim with different colloidal sol.', fontsize=16)
plt.xlabel('Raman Shift', fontsize = 16)
plt.ylabel('Normalized intenisty', fontsize = 16)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('car_comparison_30sec_1min.png', dpi=600)
plt.figure(figsize=(15,10))
plt.plot(f_sup, car_20sec_mean / np.max(car_20sec_mean), label='20 sec')
plt.plot(f_sup, car_30sec_mean / np.max(car_30sec_mean), label='30 sec')
plt.plot(f_sup, car_1min_mean / np.max(car_1min_mean), label='1 min')
plt.plot(f_sup, car_2min_mean / np.max(car_2min_mean), label='2 min')
plt.plot(f_sup, car_3min_mean / np.max(car_3min_mean), label='3 min')
plt.plot(f_sup, car_10min_mean / np.max(car_10min_mean), label='10 min')
plt.legend()
plt.title('Carbendazim with different colloidal sol.', fontsize=16)
plt.xlabel('Raman Shift', fontsize = 16)
plt.ylabel('Normalized intenisty', fontsize = 16)
plt.xticks(np.arange(0, len(f_sup)+1, 150), fontsize=15)
plt.yticks(fontsize=15)
plt.figure(figsize=(15,10))
plt.plot(f_sup, car_20sec_mean / np.max(car_20sec_mean), label='20 sec')
plt.plot(f_sup, car_30sec_mean / np.max(car_30sec_mean), label='30 sec')
plt.plot(f_sup, car_1min_mean / np.max(car_1min_mean), label='1 min')
plt.plot(f_sup, car_2min_mean / np.max(car_2min_mean), label='2 min')
plt.plot(f_sup, car_3min_mean / np.max(car_3min_mean), label='3 min')
plt.plot(f_sup, car_10min_mean / np.max(car_10min_mean), label='10 min')
plt.legend()
plt.title('Carbendazim with different colloidal sol.', fontsize=16)
plt.xlabel('Raman Shift', fontsize = 16)
plt.ylabel('Normalized intenisty', fontsize = 16)
plt.xticks(np.arange(0, f_sup[-1], 150), fontsize=15)
plt.yticks(fontsize=15)
plt.show()
plt.figure(figsize=(20,15))
plt.plot(f_sup, car_20sec_mean / np.max(car_20sec_mean), label='20 sec')
plt.plot(f_sup, car_30sec_mean / np.max(car_30sec_mean), label='30 sec')
plt.plot(f_sup, car_1min_mean / np.max(car_1min_mean), label='1 min')
plt.plot(f_sup, car_2min_mean / np.max(car_2min_mean), label='2 min')
plt.plot(f_sup, car_3min_mean / np.max(car_3min_mean), label='3 min')
plt.plot(f_sup, car_10min_mean / np.max(car_10min_mean), label='10 min')
plt.legend()
plt.title('Carbendazim with different colloidal sol.', fontsize=16)
plt.xlabel('Raman Shift', fontsize = 16)
plt.ylabel('Normalized intenisty', fontsize = 16)
plt.xticks(np.arange(0, f_sup[-1], 150), fontsize=15)
plt.yticks(fontsize=15)
plt.show()
plt.figure(figsize=(20,15))
plt.plot(f_sup, car_20sec_mean / np.max(car_20sec_mean), label='20 sec')
plt.plot(f_sup, car_30sec_mean / np.max(car_30sec_mean), label='30 sec')
plt.plot(f_sup, car_1min_mean / np.max(car_1min_mean), label='1 min')
plt.plot(f_sup, car_2min_mean / np.max(car_2min_mean), label='2 min')
plt.plot(f_sup, car_3min_mean / np.max(car_3min_mean), label='3 min')
plt.plot(f_sup, car_10min_mean / np.max(car_10min_mean), label='10 min')
plt.legend()
plt.title('Carbendazim with different colloidal sol.', fontsize=16)
plt.xlabel('Raman Shift', fontsize = 16)
plt.ylabel('Normalized intenisty', fontsize = 16)
plt.xticks(np.arange(0, f_sup[-1], 150), fontsize=15)
plt.yticks(np.arange(0, 1, 0.05), fontsize=15)
plt.show()
%history -f 2021_11_03_logs.py
