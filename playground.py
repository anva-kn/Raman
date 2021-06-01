import pathlib
pathlib.Path(__file__).parent.absolute()
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import matplotlib.pyplot as plt
from tools.ramanflow.read_data import ReadData
from tools.ramanflow.prep_data import PrepData


f_sup, mg_15ppb_30_10s_hwp42 = ReadData.read_tiff_file('data/20210427 MG ACE ACETAMIPRID colloidal SERS3/15ppb_MG/30_10s_15ppb_HWP80.tif')
f_sup, mg_150ppb_100_100ms_hwp42 = ReadData.read_tiff_file('data/20210427 MG ACE ACETAMIPRID colloidal SERS3/150ppb_MG/100ms_100_batch2_MG_HWP42.tif')
f_sup, mg_1_5ppb_30_10s_hwp42 = ReadData.read_tiff_file('data/20210427 MG ACE ACETAMIPRID colloidal SERS3/1_5ppb_MG/30_10s_1_5ppb_HWP42.tif')


best = np.mean(mg_1_5ppb_30_10s_hwp42, axis=0)
# plt.plot(f_sup, best)
noise_diff = np.zeros((30,))
for i in range(0, mg_1_5ppb_30_10s_hwp42.shape[0]):
    # fig = plt.figure(figsize=(15,7))
    mean_till_now = np.mean(mg_1_5ppb_30_10s_hwp42[0:i], axis=0)
    noise_diff[i] = mean_till_now - best
    print("Variance after {}th iteration is: {}".format(i, noise_diff[i]))
    # plt.plot(f_sup, mean_till_now, label="{:d} spectra average".format(i+1))
    # plt.xlabel('Raman Shift ^-1cm')
    # plt.ylabel('Intenisty')
    # plt.legend()
    # fig.tight_layout()
    # fig.savefig('a_{:d}.png'.format(i), dpi=300)
plt.figure(1, figsize=(15,10))
plt.plot(np.arange(1,31), noise_diff)
plt.xlabel("# of recordings")
plt.ylabel("Variance")
plt.title("1.5ppb 10s integration time")
plt.show()