print('PyDev console: using IPython 7.22.0\n')

import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['/Users/anvarkunanbaev/PycharmProjects/Raman_2.0/Raman'])
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import pandas as pd
import matplotlib.pyplot as plt
from tools.ramanflow.prep_data import PrepData as PD
from tools.ramanflow.read_data import ReadData as RD
carbendanzim_power_test_0707 = np.zeros((275, 1600))
thiacloprid_power_test_0707 = np.zeros((275, 1600))
imidacloprid_power_test_0707 = np.zeros((396, 1600))
acetamiprid_power_test_0707 = np.zeros((396, 1600))
acephate_power_test_0707 = np.zeros((396, 1600))
j = 0
for i in range(1, 12):
    f_sup, carbendanzim_power_test_0707[j * 25:(j + 1) * 25] = RD.read_tiff_file(
        'data/20210707 colloidal SERS multiple analytes/1_Carbendanzim/HWP42 to 82 step4 1s 5x5map 10xzoom/power_' + str(
            i) + '.tif')
    j += 1

j = 0
for i in range(1, 12):
    f_sup, thiacloprid_power_test_0707[j * 25:(j + 1) * 25] = RD.read_tiff_file(
        'data/20210707 colloidal SERS multiple analytes/2_Thiacloprid/HWP42 to 82 step4 1s 5x5map 10xzoom/power_' + str(
            i) + '.tif')
    j += 1

j = 0
for i in range(1, 12):
    f_sup, imidacloprid_power_test_0707[j * 36:(j + 1) * 36] = RD.read_tiff_file(
        'data/20210707 colloidal SERS multiple analytes/3_Imidacloprid/HWP42 to 62 step2 1s 6x6map 10xzoom/power_' + str(
            i) + '.tif')
    j += 1

j = 0
for i in range(1, 12):
    f_sup, acetamiprid_power_test_0707[j * 36:(j + 1) * 36] = RD.read_tiff_file(
        'data/20210707 colloidal SERS multiple analytes/4_Acetamiprid/HWP42 to 62 step2 1s 6x6map 10xzoom/power_' + str(
            i) + '.tif')
    j += 1

j = 0
for i in range(1, 12):
    f_sup, acephate_power_test_0707[j * 36:(j + 1) * 36] = RD.read_tiff_file(
        'data/20210707 colloidal SERS multiple analytes/5_Acephate/HWP42 to 62 step2 1s 6x6map 10xzoom/power_' + str(
            i) + '.tif')
    j += 1
carbendanzim_power_test_0722 = np.zeros((396, 1600))
thiacloprid_power_test_0722 = np.zeros((396, 1600))
imidacloprid_power_test_0722 = np.zeros((396, 1600))
acetamiprid_power_test_0722 = np.zeros((396, 1600))
acephate_power_test_0722 = np.zeros((396, 1600))
j = 0
for i in range(1, 12):
    f_sup, carbendanzim_power_test_0707[j * 25:(j + 1) * 25] = RD.read_tiff_file(
        'data/20210722 colloidal SERS multiple analytes/1_Carbendanzim/HWP42 to 62 step2 1s 6x6map 10xzoom/power_' + str(
            i) + '.tif')
    j += 1

j = 0
for i in range(1, 12):
    f_sup, thiacloprid_power_test_0707[j * 25:(j + 1) * 25] = RD.read_tiff_file(
        'data/20210722 colloidal SERS multiple analytes/2_Thiacloprid/HWP42 to 62 step2 1s 6x6map 10xzoom/power_' + str(
            i) + '.tif')
    j += 1

j = 0
for i in range(1, 12):
    f_sup, imidacloprid_power_test_0707[j * 36:(j + 1) * 36] = RD.read_tiff_file(
        'data/20210722 colloidal SERS multiple analytes/3_Imidacloprid/HWP42 to 62 step2 1s 6x6map 10xzoom/power_' + str(
            i) + '.tif')
    j += 1

j = 0
for i in range(1, 12):
    f_sup, acetamiprid_power_test_0707[j * 36:(j + 1) * 36] = RD.read_tiff_file(
        'data/20210722 colloidal SERS multiple analytes/4_Acetamiprid/HWP42 to 62 step2 1s 6x6map 10xzoom/power_' + str(
            i) + '.tif')
    j += 1

j = 0
for i in range(1, 12):
    f_sup, acephate_power_test_0707[j * 36:(j + 1) * 36] = RD.read_tiff_file(
        'data/20210722 colloidal SERS multiple analytes/5_Acephate/HWP42 to 62 step2 1s 6x6map 10xzoom/power_' + str(
            i) + '.tif')
    j += 1
j = 0
for i in range(1, 12):
    f_sup, carbendanzim_power_test_0707[j * 36:(j + 1) * 36] = RD.read_tiff_file(
        'data/20210722 colloidal SERS multiple analytes/1_Carbendanzim/HWP42 to 62 step2 1s 6x6map 10xzoom/power_' + str(
            i) + '.tif')
    j += 1

j = 0
for i in range(1, 12):
    f_sup, thiacloprid_power_test_0707[j * 36:(j + 1) * 36] = RD.read_tiff_file(
        'data/20210722 colloidal SERS multiple analytes/2_Thiacloprid/HWP42 to 62 step2 1s 6x6map 10xzoom/power_' + str(
            i) + '.tif')
    j += 1

j = 0
for i in range(1, 12):
    f_sup, imidacloprid_power_test_0707[j * 36:(j + 1) * 36] = RD.read_tiff_file(
        'data/20210722 colloidal SERS multiple analytes/3_Imidacloprid/HWP42 to 62 step2 1s 6x6map 10xzoom/power_' + str(
            i) + '.tif')
    j += 1

j = 0
for i in range(1, 12):
    f_sup, acetamiprid_power_test_0707[j * 36:(j + 1) * 36] = RD.read_tiff_file(
        'data/20210722 colloidal SERS multiple analytes/4_Acetamiprid/HWP42 to 62 step2 1s 6x6map 10xzoom/power_' + str(
            i) + '.tif')
    j += 1

j = 0
for i in range(1, 12):
    f_sup, acephate_power_test_0707[j * 36:(j + 1) * 36] = RD.read_tiff_file(
        'data/20210722 colloidal SERS multiple analytes/5_Acephate/HWP42 to 62 step2 1s 6x6map 10xzoom/power_' + str(
            i) + '.tif')
    j += 1
j = 0
for i in range(1, 12):
    f_sup, carbendanzim_power_test_0722[j * 36:(j + 1) * 36] = RD.read_tiff_file(
        'data/20210722 colloidal SERS multiple analytes/1_Carbendanzim/HWP42 to 62 step2 1s 6x6map 10xzoom/power_' + str(
            i) + '.tif')
    j += 1

j = 0
for i in range(1, 12):
    f_sup, thiacloprid_power_test_0722[j * 36:(j + 1) * 36] = RD.read_tiff_file(
        'data/20210722 colloidal SERS multiple analytes/2_Thiacloprid/HWP42 to 62 step2 1s 6x6map 10xzoom/power_' + str(
            i) + '.tif')
    j += 1

j = 0
for i in range(1, 12):
    f_sup, imidacloprid_power_test_0722[j * 36:(j + 1) * 36] = RD.read_tiff_file(
        'data/20210722 colloidal SERS multiple analytes/3_Imidacloprid/HWP42 to 62 step2 1s 6x6map 10xzoom/power_' + str(
            i) + '.tif')
    j += 1

j = 0
for i in range(1, 12):
    f_sup, acetamiprid_power_test_0722[j * 36:(j + 1) * 36] = RD.read_tiff_file(
        'data/20210722 colloidal SERS multiple analytes/4_Acetamiprid/HWP42 to 62 step2 1s 6x6map 10xzoom/power_' + str(
            i) + '.tif')
    j += 1

j = 0
for i in range(1, 12):
    f_sup, acephate_power_test_0722[j * 36:(j + 1) * 36] = RD.read_tiff_file(
        'data/20210722 colloidal SERS multiple analytes/5_Acephate/HWP42 to 62 step2 1s 6x6map 10xzoom/power_' + str(
            i) + '.tif')
    j += 1
j = 0
for i in range(1, 12):
    f_sup, carbendanzim_power_test_0707[j * 25:(j + 1) * 25] = RD.read_tiff_file(
        'data/20210707 colloidal SERS multiple analytes/1_Carbendanzim/HWP42 to 82 step4 1s 5x5map 10xzoom/power_' + str(
            i) + '.tif')
    j += 1

j = 0
for i in range(1, 12):
    f_sup, thiacloprid_power_test_0707[j * 25:(j + 1) * 25] = RD.read_tiff_file(
        'data/20210707 colloidal SERS multiple analytes/2_Thiacloprid/HWP42 to 82 step4 1s 5x5map 10xzoom/power_' + str(
            i) + '.tif')
    j += 1

j = 0
for i in range(1, 12):
    f_sup, imidacloprid_power_test_0707[j * 36:(j + 1) * 36] = RD.read_tiff_file(
        'data/20210707 colloidal SERS multiple analytes/3_Imidacloprid/HWP42 to 62 step2 1s 6x6map 10xzoom/power_' + str(
            i) + '.tif')
    j += 1

j = 0
for i in range(1, 12):
    f_sup, acetamiprid_power_test_0707[j * 36:(j + 1) * 36] = RD.read_tiff_file(
        'data/20210707 colloidal SERS multiple analytes/4_Acetamiprid/HWP42 to 62 step2 1s 6x6map 10xzoom/power_' + str(
            i) + '.tif')
    j += 1

j = 0
for i in range(1, 12):
    f_sup, acephate_power_test_0707[j * 36:(j + 1) * 36] = RD.read_tiff_file(
        'data/20210707 colloidal SERS multiple analytes/5_Acephate/HWP42 to 62 step2 1s 6x6map 10xzoom/power_' + str(
            i) + '.tif')
    j += 1
carbendanzim_power_test_0707_mean = np.zeros((11, 1600))
thiacloprid_power_test_0707_mean = np.zeros((11, 1600))
imidacloprid_power_test_0707_mean = np.zeros((11, 1600))
acetamiprid_power_test_0707_mean = np.zeros((11, 1600))
acephate_power_test_0707_mean = np.zeros((11, 1600))
carbendanzim_power_test_0722_mean = np.zeros((11, 1600))
thiacloprid_power_test_0722_mean = np.zeros((11, 1600))
imidacloprid_power_test_0722_mean = np.zeros((11, 1600))
acetamiprid_power_test_0722_mean = np.zeros((11, 1600))
acephate_power_test_0722_mean = np.zeros((11, 1600))
j = 0
for i in range(1,12):
    carbendanzim_power_test_0707_mean[j] = np.mean(carbendanzim_power_test_0707[j*25:(j+1)*25], axis=0)
    j += 1
for i in range(0, carbendanzim_power_test_0707_mean.shape[0] - 1):
    plt.plot(f_sup, carbendanzim_power_test_0707[i])
plt.show()
j = 0
for i in range(1,12):
    carbendanzim_power_test_0707_mean[j] = np.mean(carbendanzim_power_test_0707[j*25:(j+1)*25], axis=0)
    print(carbendanzim_power_test_0707_mean[j])
    j += 1
np.mean(carbendanzim_power_test_0707[0:25], axis=0)
plt.plot(carbendanzim_power_test_0707[1], 'power test')
plt.plot(carbendanzim_power_test_0707_mean[1], 'Mean power test')
plt.show()
plt.plot(carbendanzim_power_test_0707[1], label='power test')
plt.plot(carbendanzim_power_test_0707_mean[1], label='Mean power test')
plt.legend()
plt.show()
plt.plot(np.mean(carbendanzim_power_test_0707[0:12], axis=0), label='mean till 12th recording')
plt.legend()
np.var(carbendanzim_power_test_0707[1])
np.var(carbendanzim_power_test_0707_mean[0])
for i in range(0,11):
    thiacloprid_power_test_0707_mean[i] = np.mean(thiacloprid_power_test_0707[i*25:(i+1)*25], axis=0)
for i in range(0,11):
    imidacloprid_power_test_0707_mean[i] = np.mean(imidacloprid_power_test_0707[i*36:(i+1)*36], axis=0)
for i in range(0,11):
    acetamiprid_power_test_0707_mean[i] = np.mean(acetamiprid_power_test_0707[i*36:(i+1)*36], axis=0)
for i in range(0,11):
    acephate_power_test_0707_mean[i] = np.mean(acepahte_power_test_0707[i*36:(i+1)*36], axis=0)
for i in range(0,11):
    acephate_power_test_0707_mean[i] = np.mean(acephate_power_test_0707[i*36:(i+1)*36], axis=0)
for i in range(0,11):
    carbendanzim_power_test_0722_mean[j] = np.mean(carbendanzim_power_test_07022[j*36:(j+1)*36], axis=0)
    
for i in range(0,11):
    thiacloprid_power_test_0722_mean[i] = np.mean(thiacloprid_power_test_0722[i*36:(i+1)*36], axis=0)
    
for i in range(0,11):
    imidacloprid_power_test_0722_mean[i] = np.mean(imidacloprid_power_test_0722[i*36:(i+1)*36], axis=0)
    
for i in range(0,11):
    acetamiprid_power_test_0722_mean[i] = np.mean(acetamiprid_power_test_0722[i*36:(i+1)*36], axis=0)
    
for i in range(0,11):
    acephate_power_test_0722_mean[i] = np.mean(acephate_power_test_0722[i*36:(i+1)*36], axis=0)
for i in range(0,11):
    carbendanzim_power_test_0722_mean[j] = np.mean(carbendanzim_power_test_0722[j*36:(j+1)*36], axis=0)
    
for i in range(0,11):
    thiacloprid_power_test_0722_mean[i] = np.mean(thiacloprid_power_test_0722[i*36:(i+1)*36], axis=0)
    
for i in range(0,11):
    imidacloprid_power_test_0722_mean[i] = np.mean(imidacloprid_power_test_0722[i*36:(i+1)*36], axis=0)
    
for i in range(0,11):
    acetamiprid_power_test_0722_mean[i] = np.mean(acetamiprid_power_test_0722[i*36:(i+1)*36], axis=0)
    
for i in range(0,11):
    acephate_power_test_0722_mean[i] = np.mean(acephate_power_test_0722[i*36:(i+1)*36], axis=0)
for i in range(0,11):
    carbendanzim_power_test_0722_mean[i] = np.mean(carbendanzim_power_test_0722[i*36:(i+1)*36], axis=0)
    
for i in range(0,11):
    thiacloprid_power_test_0722_mean[i] = np.mean(thiacloprid_power_test_0722[i*36:(i+1)*36], axis=0)
    
for i in range(0,11):
    imidacloprid_power_test_0722_mean[i] = np.mean(imidacloprid_power_test_0722[i*36:(i+1)*36], axis=0)
    
for i in range(0,11):
    acetamiprid_power_test_0722_mean[i] = np.mean(acetamiprid_power_test_0722[i*36:(i+1)*36], axis=0)
    
for i in range(0,11):
    acephate_power_test_0722_mean[i] = np.mean(acephate_power_test_0722[i*36:(i+1)*36], axis=0)
plt.plot(f_sup, carbendanzim_power_test_0707[1], label='raw power test')
plt.plot(f_sup, carbendanzim_power_test_0707_mean[0], label='mean power test')
plt.legend()
plt.show()
plt.plot(f_sup, carbendanzim_power_test_0707[0], label='raw power test')
plt.legend()
plt.show()
plt.figure('Carbendanzim 07-07')
for i in range(0, carbendanzim_power_test_0707_mean.shape[0] - 1):
    plt.plot(f_sup, carbendanzim_power_test_0707_mean[i], label='hwp {}'.format(42 + i*4))
plt.legend()
plt.show()
plt.figure('Carbendanzim 07-07')
for i in range(0, carbendanzim_power_test_0707_mean.shape[0]):
    plt.plot(f_sup, carbendanzim_power_test_0707_mean[i], label='hwp {}'.format(42 + i*4))
plt.legend()
plt.show()
plt.figure('Thiacloprid 07-07')
for i in range(0, 11):
    plt.plot(f_sup, thiacloprid_power_test_0707_mean[i], label='hwp {}'.format(42 + i*4))
plt.legend()
plt.show()
import matplotlib.ticker as ticker
tick_spacing = 100
plt.figure('Carbendanzim 07-07')
for i in range(0, carbendanzim_power_test_0707_mean.shape[0]):
    plt.plot(f_sup, carbendanzim_power_test_0707_mean[i], label='hwp {}'.format(42 + i*4))
plt.ylabel('Intensity')
plt.xlabel('Raman shift')
plt.xticks(np.arange(f_sup[0], f_sup[-1], 100))
plt.legend()
plt.savefig('Carbendanzim-07-07.png', dpi=400)
import matplotlib.ticker as ticker
tick_spacing = 100
plt.figure('Carbendanzim 07-07')
for i in range(0, carbendanzim_power_test_0707_mean.shape[0]):
    plt.plot(f_sup, carbendanzim_power_test_0707_mean[i], label='hwp {}'.format(42 + i*4))
plt.ylabel('Intensity')
plt.xlabel('Raman shift')
plt.xticks(np.arange(f_sup[0], f_sup[-1], 100))
plt.legend()
plt.show()
import matplotlib.ticker as ticker
tick_spacing = 150
plt.figure()
for i in range(0, carbendanzim_power_test_0707_mean.shape[0]):
    plt.plot(f_sup, carbendanzim_power_test_0707_mean[i], label='hwp {}'.format(42 + i*4))
plt.ylabel('Intensity')
plt.xlabel('Raman shift')
plt.title('Carbendanzim 07-07')
plt.xticks(np.arange(f_sup[0], f_sup[-1], tick_spacing))
plt.legend()
plt.show()
plt.savefig('Carbendanzim-07-07.png', dpi=400)
power_table = pd.read_csv('data/power_table.csv')
power_table.iloc('no ND')
power_table.loc[['no ND']]
power_table.loc['no ND']
power_table.columns
tick_spacing = 150
plt.figure()
for i in range(0, carbendanzim_power_test_0707_mean.shape[0]):
    plt.plot(f_sup, thiacloprid_power_test_0707_mean[i], label='hwp {}'.format(42 + i*4))
plt.ylabel('Intensity')
plt.xlabel('Raman shift')
plt.title('Thiacloprid 07-07')
plt.xticks(np.arange(f_sup[0], f_sup[-1], tick_spacing))
plt.legend()
plt.show()
plt.savefig('Thiacloprid-07-07.png', dpi=400)
tick_spacing = 150
plt.figure()
for i in range(0, carbendanzim_power_test_0707_mean.shape[0]):
    plt.plot(f_sup, imidacloprid_power_test_0707_mean[i], label='hwp {}'.format(42 + i*4))
plt.ylabel('Intensity')
plt.xlabel('Raman shift')
plt.title('Imidacloprid 07-07')
plt.xticks(np.arange(f_sup[0], f_sup[-1], tick_spacing))
plt.legend()
plt.show()
plt.savefig('Imidacloprid-07-07.png', dpi=400)
tick_spacing = 150
plt.figure()
for i in range(0, 11):
    plt.plot(f_sup, acetamiprid_power_test_0707_mean[i], label='hwp {}'.format(42 + i*4))
plt.ylabel('Intensity')
plt.xlabel('Raman shift')
plt.title('Acetamiprid 07-07')
plt.xticks(np.arange(f_sup[0], f_sup[-1], tick_spacing))
plt.legend()
plt.show()
plt.savefig('Acetamiprid-07-07.png', dpi=400)
tick_spacing = 150
plt.figure()
for i in range(0, 11):
    plt.plot(f_sup, acephate_power_test_0707_mean[i], label='hwp {}'.format(42 + i*4))
plt.ylabel('Intensity')
plt.xlabel('Raman shift')
plt.title('Acephate 07-07')
plt.xticks(np.arange(f_sup[0], f_sup[-1], tick_spacing))
plt.legend()
plt.show()
plt.savefig('Acephate-07-07.png', dpi=400)
tick_spacing = 150
plt.figure()
for i in range(0, 11):
    plt.plot(f_sup, carbendanzim_power_test_0722_mean[i], label='hwp {}'.format(42 + i*4))
plt.ylabel('Intensity')
plt.xlabel('Raman shift')
plt.title('Carbendanzim 07-22')
plt.xticks(np.arange(f_sup[0], f_sup[-1], tick_spacing))
plt.legend()
plt.show()
plt.savefig('Carbendanzim-07-22.png', dpi=400)
tick_spacing = 150
plt.figure()
for i in range(0, 11):
    plt.plot(f_sup, thiacloprid_power_test_0722_mean[i], label='hwp {}'.format(42 + i*4))
plt.ylabel('Intensity')
plt.xlabel('Raman shift')
plt.title('Thiacloprid 07-22')
plt.xticks(np.arange(f_sup[0], f_sup[-1], tick_spacing))
plt.legend()
plt.show()
plt.savefig('Thiacloprid-07-22.png', dpi=400)
tick_spacing = 150
plt.figure()
for i in range(0, 11):
    plt.plot(f_sup, imidacloprid_power_test_0722_mean[i], label='hwp {}'.format(42 + i*4))
plt.ylabel('Intensity')
plt.xlabel('Raman shift')
plt.title('Imidacloprid 07-22')
plt.xticks(np.arange(f_sup[0], f_sup[-1], tick_spacing))
plt.legend()
plt.show()
plt.savefig('Imidacloprid-07-22.png', dpi=400)
tick_spacing = 150
plt.figure()
for i in range(0, 11):
    plt.plot(f_sup, acetamiprid_power_test_0722_mean[i], label='hwp {}'.format(42 + i*4))
plt.ylabel('Intensity')
plt.xlabel('Raman shift')
plt.title('Imidacloprid 07-22')
plt.xticks(np.arange(f_sup[0], f_sup[-1], tick_spacing))
plt.legend()
plt.show()
tick_spacing = 150
plt.figure()
for i in range(0, 11):
    plt.plot(f_sup, acetamiprid_power_test_0722_mean[i], label='hwp {}'.format(42 + i*4))
plt.ylabel('Intensity')
plt.xlabel('Raman shift')
plt.title('Acetamiprid 07-22')
plt.xticks(np.arange(f_sup[0], f_sup[-1], tick_spacing))
plt.legend()
plt.show()
plt.savefig('Acetamiprid-07-22.png', dpi=400)
tick_spacing = 150
plt.figure()
for i in range(0, 11):
    plt.plot(f_sup, acephate_power_test_0722_mean[i], label='hwp {}'.format(42 + i*2))
plt.ylabel('Intensity')
plt.xlabel('Raman shift')
plt.title('Acephate 07-22')
plt.xticks(np.arange(f_sup[0], f_sup[-1], tick_spacing))
plt.legend()
plt.show()
plt.savefig('Acephate-07-22.png', dpi=400)
tick_spacing = 150
plt.figure()
for i in range(0, 11):
    plt.plot(f_sup, carbendanzim_power_test_0722_mean[i], label='hwp {}'.format(42 + i*2))
plt.ylabel('Intensity')
plt.xlabel('Raman shift')
plt.title('Carbendanzim 07-22')
plt.xticks(np.arange(f_sup[0], f_sup[-1], tick_spacing))
plt.legend()
plt.show()
plt.savefig('Carbendanzim-07-22.png', dpi=400)
tick_spacing = 150
plt.figure()
for i in range(0, 11):
    plt.plot(f_sup, thiacloprid_power_test_0722_mean[i], label='hwp {}'.format(42 + i*2))
plt.ylabel('Intensity')
plt.xlabel('Raman shift')
plt.title('Thiacloprid 07-22')
plt.xticks(np.arange(f_sup[0], f_sup[-1], tick_spacing))
plt.legend()
plt.show()
plt.savefig('Thiacloprid-07-22.png', dpi=400)
tick_spacing = 150
plt.figure()
for i in range(0, 11):
    plt.plot(f_sup, imidacloprid_power_test_0722_mean[i], label='hwp {}'.format(42 + i*2))
plt.ylabel('Intensity')
plt.xlabel('Raman shift')
plt.title('Imidacloprid 07-22')
plt.xticks(np.arange(f_sup[0], f_sup[-1], tick_spacing))
plt.legend()
plt.show()
plt.savefig('Imidacloprid-07-22.png', dpi=400)
tick_spacing = 150
plt.figure()
for i in range(0, 11):
    plt.plot(f_sup, acetamiprid_power_test_0722_mean[i], label='hwp {}'.format(42 + i*2))
plt.ylabel('Intensity')
plt.xlabel('Raman shift')
plt.title('Acetamiprid 07-22')
plt.xticks(np.arange(f_sup[0], f_sup[-1], tick_spacing))
plt.legend()
plt.show()
plt.savefig('Acetamiprid-07-22.png', dpi=400)
j = 0
for i in range(1, 12):
    f_sup_old, carbendanzim_power_test_0707[j * 25:(j + 1) * 25] = RD.read_tiff_file(
        'data/20210707 colloidal SERS multiple analytes/1_Carbendanzim/HWP42 to 82 step4 1s 5x5map 10xzoom/power_' + str(
            i) + '.tif')
    j += 1

j = 0
for i in range(1, 12):
    f_sup_old, thiacloprid_power_test_0707[j * 25:(j + 1) * 25] = RD.read_tiff_file(
        'data/20210707 colloidal SERS multiple analytes/2_Thiacloprid/HWP42 to 82 step4 1s 5x5map 10xzoom/power_' + str(
            i) + '.tif')
    j += 1

j = 0
for i in range(1, 12):
    f_sup_old, imidacloprid_power_test_0707[j * 36:(j + 1) * 36] = RD.read_tiff_file(
        'data/20210707 colloidal SERS multiple analytes/3_Imidacloprid/HWP42 to 62 step2 1s 6x6map 10xzoom/power_' + str(
            i) + '.tif')
    j += 1

j = 0
for i in range(1, 12):
    f_sup_old, acetamiprid_power_test_0707[j * 36:(j + 1) * 36] = RD.read_tiff_file(
        'data/20210707 colloidal SERS multiple analytes/4_Acetamiprid/HWP42 to 62 step2 1s 6x6map 10xzoom/power_' + str(
            i) + '.tif')
    j += 1

j = 0
for i in range(1, 12):
    f_sup_old, acephate_power_test_0707[j * 36:(j + 1) * 36] = RD.read_tiff_file(
        'data/20210707 colloidal SERS multiple analytes/5_Acephate/HWP42 to 62 step2 1s 6x6map 10xzoom/power_' + str(
            i) + '.tif')
    j += 1
f_sup_old == f_sup
plt.plot(f_sup, acetamiprid_power_test_0707_mean[-1])
plt.plot(f_sup[f_sup == 635.48637], acetamiprid_power_test_0707_mean[-1, f_sup == 635.48637], 'r*')
plt.show()
plt.plot(f_sup, acetamiprid_power_test_0707_mean[-1])
plt.plot(f_sup[f_sup == 635.48637], acetamiprid_power_test_0707_mean[-1, f_sup == 635.48637], 'r*')
plt.show()
np.where(f_sup == 635.48637)
plt.plot(f_sup, acetamiprid_power_test_0707_mean[-1])
plt.plot(f_sup[421], acetamiprid_power_test_0707_mean[-1, 421], 'r*')
plt.show()
plt.plot(f_sup, acetamiprid_power_test_0707_mean[-1])
plt.plot(f_sup[421], acetamiprid_power_test_0707_mean[-1, 421], 'r*')
plt.show()
plt.plot(f_sup[466], acetamiprid_power_test_0707_mean[-1, 466], 'r*')
plt.show()
plt.plot(f_sup[702], acetamiprid_power_test_0707_mean[-1, 702], 'r*')
plt.plot(f_sup[544], acetamiprid_power_test_0707_mean[-1, 544], 'r*')
plt.plot(f_sup[727], acetamiprid_power_test_0707_mean[-1, 727], 'r*')
plt.show()
plt.figure()
plt.plot(f_sup, carbendanzim_power_test_0722_mean[-1])
plt.plot(f_sup[669], carbendanzim_power_test_0722_mean[-1, 669], 'r*')
plt.plot(f_sup[837], carbendanzim_power_test_0722_mean[-1, 837], 'r*')
plt.plot(f_sup[907], carbendanzim_power_test_0722_mean[-1, 907], 'r*')
plt.show()
