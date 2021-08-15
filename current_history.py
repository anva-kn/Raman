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
for i in range(1,12):
    print(i)
for i in range(1,12):
    f_sup, carbendanzim_power_test_2207 = RD.read_tiff_file('data/20210722 colloidal SERS multiple analytes/1_Carbendanzim/HWP42 to 62 step2 1s 6x6map 10xzoom/power_'+str(i)+'.tif')
plt.plot(f_sup, np.mean(carbendanzim_power_test_2207, axis=0))
plt.show()
carbendanzim_power_test_2207.shape
len(carbendanzim_power_test_2207)
for i in range(len(carbendanzim_power_test_2207)):
    plt.plot(f_sup, carbendanzim_power_test_2207)
plt.title('Carbendanzim 22-07-2021 power test')
plt.xlabel('Raman shift cm^-1')
plt.ylabel('Intensity')
plt.show()
for i in range(len(carbendanzim_power_test_2207)):
    plt.plot(f_sup, carbendanzim_power_test_2207[i])
plt.title('Carbendanzim 22-07-2021 power test')
plt.xlabel('Raman shift cm^-1')
plt.ylabel('Intensity')
plt.show()
for i in range(1,12):
    f_sup, thiacloprid_power_test_2207 = RD.read_tiff_file('data/20210722 colloidal SERS multiple analytes/2_Thiacloprid/HWP42 to 62 step2 1s 6x6map 10xzoom/power_'+str(i)+'.tif')
for i in range(1,12):
    f_sup, thiacloprid_power_test_2207 = RD.read_tiff_file('data/20210722 colloidal SERS multiple analytes/3_Imidacloprid/HWP42 to 62 step2 1s 6x6map 10xzoom/power_'+str(i)+'.tif')
for i in range(1,12):
    f_sup, acetamiprid_power_test_2207 = RD.read_tiff_file('data/20210722 colloidal SERS multiple analytes/4_Acetamiprid/HWP42 to 62 step2 1s 6x6map 10xzoom/power_'+str(i)+'.tif')
for i in range(1,12):
    f_sup, thiacloprid_power_test_2207 = RD.read_tiff_file('data/20210722 colloidal SERS multiple analytes/2_Thiacloprid/HWP42 to 62 step2 1s 6x6map 10xzoom/power_'+str(i)+'.tif')
for i in range(1,12):
    f_sup, imidacloprid_power_test_2207 = RD.read_tiff_file('data/20210722 colloidal SERS multiple analytes/3_Imidacloprid/HWP42 to 62 step2 1s 6x6map 10xzoom/power_'+str(i)+'.tif')
for i in range(1,12):
    f_sup, acephate_power_test_2207 = RD.read_tiff_file('data/20210722 colloidal SERS multiple analytes/5_Acephate/HWP42 to 62 step2 1s 6x6map 10xzoom/power_'+str(i)+'.tif')
for i in range(1,12):
    f_sup, carbendanzim_power_test_0707 = RD.read_tiff_file('data/20210707 colloidal SERS multiple analytes/1_Carbendanzim/HWP42 to 82 step4 1s 5x5map 10xzoom/power_'+str(i)+'.tif')

for i in range(1,12):
    f_sup, thiacloprid_power_test_0707 = RD.read_tiff_file('data/20210707 colloidal SERS multiple analytes/2_Thiacloprid/HWP42 to 82 step4 1s 5x5map 10xzoom/power_'+str(i)+'.tif')

for i in range(1,12):
    f_sup, imidacloprid_power_test_0707 = RD.read_tiff_file('data/20210707 colloidal SERS multiple analytes/3_Imidacloprid/HWP42 to 62 step2 1s 6x6map 10xzoom/power_'+str(i)+'.tif')

for i in range(1,12):
    f_sup, acetamiprid_power_test_0707 = RD.read_tiff_file('data/20210707 colloidal SERS multiple analytes/4_Acetamiprid/HWP42 to 62 step2 1s 6x6map 10xzoom/power_'+str(i)+'.tif')

for i in range(1,12):
    f_sup, acephate_power_test_0707 = RD.read_tiff_file('data/20210707 colloidal SERS multiple analytes/5_Acephate/HWP42 to 62 step2 1s 6x6map 10xzoom/power_'+str(i)+'.tif')
for i in range(1,12):
    f_sup_prev, carbendanzim_power_test_0707 = RD.read_tiff_file('data/20210707 colloidal SERS multiple analytes/1_Carbendanzim/HWP42 to 82 step4 1s 5x5map 10xzoom/power_'+str(i)+'.tif')

for i in range(1,12):
    f_sup_prev, thiacloprid_power_test_0707 = RD.read_tiff_file('data/20210707 colloidal SERS multiple analytes/2_Thiacloprid/HWP42 to 82 step4 1s 5x5map 10xzoom/power_'+str(i)+'.tif')

for i in range(1,12):
    f_sup_prev, imidacloprid_power_test_0707 = RD.read_tiff_file('data/20210707 colloidal SERS multiple analytes/3_Imidacloprid/HWP42 to 62 step2 1s 6x6map 10xzoom/power_'+str(i)+'.tif')

for i in range(1,12):
    f_sup_prev, acetamiprid_power_test_0707 = RD.read_tiff_file('data/20210707 colloidal SERS multiple analytes/4_Acetamiprid/HWP42 to 62 step2 1s 6x6map 10xzoom/power_'+str(i)+'.tif')

for i in range(1,12):
    f_sup_prev, acephate_power_test_0707 = RD.read_tiff_file('data/20210707 colloidal SERS multiple analytes/5_Acephate/HWP42 to 62 step2 1s 6x6map 10xzoom/power_'+str(i)+'.tif')
for i in range(1,12):
    f_sup, imidacloprid_power_test_2207 = RD.read_tiff_file('data/20210722 colloidal SERS multiple analytes/3_Imidacloprid/HWP42 to 62 step2 1s 6x6map 10xzoom/power_'+str(i)+'.tif')
carbendanzim_power_test_0707 = np.zeros((11, 1600))
thiacloprid_power_test_0707 = np.zeros((11, 1600))
imidacloprid_power_test_0707 = np.zeros((11, 1600))
acetamiprid_power_test_0707 = np.zeros((11, 1600))
acephate_power_test_0707 = np.zeros((11, 1600))
j = 0
for i in range(1,12):
    f_sup, carbendanzim_power_test_0707[j] = RD.read_tiff_file('data/20210707 colloidal SERS multiple analytes/1_Carbendanzim/HWP42 to 82 step4 1s 5x5map 10xzoom/power_'+str(i)+'.tif')
    j += 1

j = 0
for i in range(1,12):
    f_sup, thiacloprid_power_test_0707[j] = RD.read_tiff_file('data/20210707 colloidal SERS multiple analytes/2_Thiacloprid/HWP42 to 82 step4 1s 5x5map 10xzoom/power_'+str(i)+'.tif')
    j += 1

j = 0
for i in range(1,12):
    f_sup, imidacloprid_power_test_0707[j] = RD.read_tiff_file('data/20210707 colloidal SERS multiple analytes/3_Imidacloprid/HWP42 to 62 step2 1s 6x6map 10xzoom/power_'+str(i)+'.tif')
    j += 1
    
j = 0
for i in range(1,12):
    f_sup, acetamiprid_power_test_0707[j] = RD.read_tiff_file('data/20210707 colloidal SERS multiple analytes/4_Acetamiprid/HWP42 to 62 step2 1s 6x6map 10xzoom/power_'+str(i)+'.tif')
    j += 1
    
j = 0
for i in range(1,12):
    f_sup, acephate_power_test_0707[j] = RD.read_tiff_file('data/20210707 colloidal SERS multiple analytes/5_Acephate/HWP42 to 62 step2 1s 6x6map 10xzoom/power_'+str(i)+'.tif')
    j += 1
carbendanzim_power_test_0707 = np.zeros((11, 1600))
thiacloprid_power_test_0707 = np.zeros((11, 1600))
imidacloprid_power_test_0707 = np.zeros((11, 1600))
acetamiprid_power_test_0707 = np.zeros((11, 1600))
acephate_power_test_0707 = np.zeros((11, 1600))
j = 0
for i in range(1,12):
    f_sup, carbendanzim_power_test_0707[j] = np.mean(RD.read_tiff_file('data/20210707 colloidal SERS multiple analytes/1_Carbendanzim/HWP42 to 82 step4 1s 5x5map 10xzoom/power_'+str(i)+'.tif'), axis=0)
    j += 1

j = 0
for i in range(1,12):
    f_sup, thiacloprid_power_test_0707[j] = np.mean(RD.read_tiff_file('data/20210707 colloidal SERS multiple analytes/2_Thiacloprid/HWP42 to 82 step4 1s 5x5map 10xzoom/power_'+str(i)+'.tif',axis = 0))
    j += 1

j = 0
for i in range(1,12):
    f_sup, imidacloprid_power_test_0707[j] = np.mean(RD.read_tiff_file('data/20210707 colloidal SERS multiple analytes/3_Imidacloprid/HWP42 to 62 step2 1s 6x6map 10xzoom/power_'+str(i)+'.tif'), axis=0)
    j += 1
    
j = 0
for i in range(1,12):
    f_sup, acetamiprid_power_test_0707[j] = np.mean(RD.read_tiff_file('data/20210707 colloidal SERS multiple analytes/4_Acetamiprid/HWP42 to 62 step2 1s 6x6map 10xzoom/power_'+str(i)+'.tif'), axis=0)
    j += 1
    
j = 0
for i in range(1,12):
    f_sup, acephate_power_test_0707[j] = np.mean(RD.read_tiff_file('data/20210707 colloidal SERS multiple analytes/5_Acephate/HWP42 to 62 step2 1s 6x6map 10xzoom/power_'+str(i)+'.tif'), axis=0)
    j += 1
carbendanzim_power_test_0707 = np.zeros((275, 1600))
thiacloprid_power_test_0707 = np.zeros((275, 1600))
imidacloprid_power_test_0707 = np.zeros((396, 1600))
acetamiprid_power_test_0707 = np.zeros((396, 1600))
acephate_power_test_0707 = np.zeros((396, 1600))
j = 0
for i in range(1,12):
    f_sup, carbendanzim_power_test_0707[j*25:(j+1)*25] = RD.read_tiff_file('data/20210707 colloidal SERS multiple analytes/1_Carbendanzim/HWP42 to 82 step4 1s 5x5map 10xzoom/power_'+str(i)+'.tif')
    j += 1

j = 0
for i in range(1,12):
    f_sup, thiacloprid_power_test_0707[j*25:(j+1)*25] = RD.read_tiff_file('data/20210707 colloidal SERS multiple analytes/2_Thiacloprid/HWP42 to 82 step4 1s 5x5map 10xzoom/power_'+str(i)+'.tif')
    j += 1

j = 0
for i in range(1,12):
    f_sup, imidacloprid_power_test_0707[j*36:(j+1)*36] = RD.read_tiff_file('data/20210707 colloidal SERS multiple analytes/3_Imidacloprid/HWP42 to 62 step2 1s 6x6map 10xzoom/power_'+str(i)+'.tif')
    j += 1
    
j = 0
for i in range(1,12):
    f_sup, acetamiprid_power_test_0707[j*36:(j+1)*36] = RD.read_tiff_file('data/20210707 colloidal SERS multiple analytes/4_Acetamiprid/HWP42 to 62 step2 1s 6x6map 10xzoom/power_'+str(i)+'.tif')
    j += 1
    
j = 0
for i in range(1,12):
    f_sup, acephate_power_test_0707[j*36:(j+1)*36] = RD.read_tiff_file('data/20210707 colloidal SERS multiple analytes/5_Acephate/HWP42 to 62 step2 1s 6x6map 10xzoom/power_'+str(i)+'.tif')
    j += 1
%history -f current_history.py
