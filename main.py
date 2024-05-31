from itertools import islice

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert, find_peaks

#reading files
testfile_1_5_ppb = open('1.5ppb_35days_15ml-1_Exported.dat', "r")
testfile_15_ppb = open('15ppb_4days_15ml-1_Exported.dat', "r")
testfile_1500_ppb = open('1500ppb_47.5hr_1.5ml-1_Exported.dat', "r")
testfile_original = open('Original_MG_solution-100um_HWP49_noND_9.2mW_20x(NA0.75)_Newton(S-S-H-C)_500ms_4avg_2bx_Exported.dat', "r")

#obtaining frequency-amplitude pairs
lines_1_5_ppb = testfile_1_5_ppb.readlines()
lines_15_ppb = testfile_15_ppb.readlines()
lines_1500_ppb = testfile_1500_ppb.readlines()
lines_original = testfile_original.readlines()
result_frequency_1_5 = []
result_frequency_15 = []
result_frequency_1500 = []
result_frequency_original = []
result_amplitude_1_5 = []
result_amplitude_15 = []
result_amplitude_1500 = []
result_amplitude_original = []

for (small, medium, big, original) in zip(lines_1_5_ppb, lines_15_ppb, lines_1500_ppb, lines_original):
    small, medium, big, original = small.strip('\n'), medium.strip('\n'), big.strip('\n'), original.strip('\n')
    small, medium, big, original = small.replace(",", "."), medium.replace(",", "."), big.replace(",", "."), original.replace(",", ".")
    result_frequency_1_5.append(small.split("\t")[0])
    result_frequency_15.append(medium.split("\t")[0])
    result_frequency_1500.append(big.split("\t")[0])
    result_frequency_original.append(original.split("\t")[0])
    result_amplitude_1_5.append(small.split("\t")[1])
    result_amplitude_15.append(medium.split("\t")[1])
    result_amplitude_1500.append(big.split("\t")[1])
    result_amplitude_original.append(original.split("\t")[1])


new_list_small_x = []
new_list_small_y = []
new_list_medium_x = []
new_list_medium_y = []
new_list_big_x = []
new_list_big_y = []
new_list_original_x = []
new_list_original_y = []
for (item_s_x, item_m_x, item_b_x, item_or_x) in zip(result_frequency_1_5, result_frequency_15, result_frequency_1500, result_frequency_original):
    new_list_small_x.append(float(item_s_x))
    new_list_medium_x.append(float(item_m_x))
    new_list_big_x.append(float(item_b_x))
    new_list_original_x.append(float(item_or_x))

for (item_s_y, item_m_y, item_b_y, item_or_y) in zip(result_amplitude_1_5, result_amplitude_15, result_amplitude_1500, result_amplitude_original):
    new_list_small_y.append(float(item_s_y))
    new_list_medium_y.append(float(item_m_y))
    new_list_big_y.append(float(item_b_y))
    new_list_original_y.append(float(item_or_y))

#### getting envelopes
# win = 10
# j = 0
# min_position = [0,]
# max_position = [0,]
# for (min_position[j + win], max_position[j + win]) in new_list_big_y:
#     min_position.append(new_list_big_y)

win = 150
def window(seq, n=win):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

list_windowed = []
list_windowed = list(window(new_list_big_y))
print(list_windowed)


min_position = []
max_position = []
min_position_list = [min(min_position) for min_position in list_windowed]
print(min_position_list)
###########

###########
array_big_list_y = np.array(new_list_big_y)
peaks, properties = find_peaks(array_big_list_y, width = 10, prominence=1, distance=25)
print(peaks)
data_peak = array_big_list_y[peaks]
data_peak_width=np.array([np.mean(array_big_list_y[int(properties['left_ips'][j]):int(properties["right_ips"][j])], axis=0) for j in  range(peaks.shape[0])])


data_peak = array_big_list_y[peaks]
data_peak_width=np.array([np.mean(array_big_list_y[int(properties["left_ips"][j]):int(properties["right_ips"][j])], axis=0) for j in  range(peaks.shape[0])] )


# my_dark_peaks =  peaks[[4 ,5 ,10, 11]]

dark_mean_fig=1

if dark_mean_fig:
    x=np.array(new_list_big_y)
    plt.plot(x)
    plt.plot(peaks, x[peaks], "x")
    plt.vlines(x=peaks, ymin=x[peaks] - properties["prominences"],ymax = x[peaks], color = "c1")
    plt.hlines(y=properties["width_heights"], xmin=properties["left_ips"],xmax=properties["right_ips"], color = "c1")
    plt.show()
###########

# print(max_position)
# #####
# # plot polynomial
# # # plt.yscale('log')
# xp = np.linspace(100, 2600, 1600 - win + 1)
# plt.plot(new_list_big_x, new_list_big_y, 'r')
# plt.plot(xp, min_position_list, 'g')
# # plt.ylim(1,130)
# #
# # naming the x axis
# plt.xlabel('Frequency')
# # naming the y axis
# plt.ylabel('Amplitude')
#
# # giving a title to my graph
# plt.title('1500ppb with low envelope')
#
# # function to show the plot
# plt.show()
# # # plt.savefig('SRES_with_inter_1.png', bbox_inches='tight')
# #
# # # show roots
# # print (p.roots)matplotlib.pyplot as plt
