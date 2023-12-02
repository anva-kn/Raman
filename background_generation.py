#Data libraries
import shelve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, ifftshift

#%%
pristine_area1 = pd.read_csv('area1_Exported.dat', sep="\s+", header=None)

length = pristine_area1.shape[1] - 1
f_sup_pristine = np.array(pristine_area1.iloc[:, 0], dtype=np.float64)
pristine_area1 = np.array(pristine_area1.iloc[:, 1], dtype=np.float64)

#%%
filename='shelve_save_data.out'

my_shelf = shelve.open(filename)
klist = list(my_shelf.keys())
for key in my_shelf:
    globals()[key] = my_shelf[key]
my_shelf.close()
#%%
#Save the wavelength variable
f_sup_pure = np.copy(f_sup)
f_sup_pure.shape

#%%
filename='shelve_save_data_analyte.out'

my_shelf = shelve.open(filename)
for key in my_shelf:
    globals()[key]=my_shelf[key]
my_shelf.close()

#Save the wavelength variable for mixed
f_sup_mix = np.copy(f_sup)
f_sup_mix.shape

#%%
whole_data_pure = np.zeros((30000, 1600))
# print(whole_data.shape)
whole_data_pure[:2500] = np.copy(data_15000[0])
whole_data_pure[2500:5000] = np.copy(data_15000[1])
whole_data_pure[5000:7500] = np.copy(data_15000[2])
whole_data_pure[7500:10000] = np.copy(data_15000[3])
whole_data_pure[10000:12500] = np.copy(data_1500[0])
whole_data_pure[12500:15000] = np.copy(data_1500[1])
whole_data_pure[15000:17500] = np.copy(data_1500[2])
whole_data_pure[17500:20000] = np.copy(data_1500[3])
whole_data_pure[20000:22500] = np.copy(data_150[0])
whole_data_pure[22500:25000] = np.copy(data_150[1])
whole_data_pure[25000:27500] = np.copy(data_150[2])
whole_data_pure[27500:30000] = np.copy(data_150[3])

#%%
training_data = np.copy(whole_data_pure) / np.max(whole_data_pure, axis=-1, keepdims=True)
mask = np.all(np.isfinite(training_data), axis=-1)
# mask = np.all(training_data, axis=-1)
labels = np.zeros((30000, 3))
labels[:10000, 0] = 1
labels[10000:20000, 1] = 1
labels[20000:, 2] = 1
labels_not_one_hot = np.where(labels==1)[1]

X = training_data[mask]
y = labels_not_one_hot[mask]
print(X.shape)
print(y.shape)

#%%
signal_orig = np.copy(X[1, :])

print(type(signal_orig))
print(type(np.flip(signal_orig)))
signal_orig = np.hstack((signal_orig, np.flip(signal_orig)))

signal_spec = np.fft.fftshift(np.fft.fft(signal_orig))

signal_spec[signal_spec < 5] = 0

# signal_spec[signal_spec < 40] = 0
signal_ifftd = np.fft.ifft(np.fft.ifftshift(signal_spec))

plt.figure(figsize=(10, 10))
plt.plot(range(0, 6400), np.abs(signal_ifftd))
# plt.plot(range(0,6400), signal_spec)
