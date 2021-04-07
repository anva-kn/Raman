#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 19:38:59 2021

@author: anvarkunanbaev
"""
import numpy as np
import pandas as pd
import scipy.interpolate as si
from skimage import io
from PIL import Image
from PIL.TiffTags import TAGS
import re

class ProcessData():
    
    def read_dat_file(filename):
        try:    
            data = pd.read_csv(filename, sep="\s+", header=None)
            f_sup = np.array(data.iloc[:, 0], dtype=np.float64)
            data_array = data.loc[:, 1:].to_numpy(dtype=np.float64)
            data_array = data_array.T
            if data_array.shape[0] == 1:
                data_array = np.squeeze(data_array)
            np.ascontiguousarray(data_array, dtype=np.float64)
            return f_sup, data_array
        except FileNotFoundError:
            print("Wrong filename or path.")
            
    def read_tiff_file(filename):
            try:    
                data = io.imread(filename)
                with Image.open(filename) as img:
                    meta_dict = {TAGS[key] : img.tag[key] for key in img.tag}
                info_string = meta_dict['Software'][0]
                spec_max = float(re.findall('(?<=\SpecMax=)[0-9.]+', info_string)[0])
                spec_min = float(re.findall('(?<=\SpecMin=)[0-9.]+', info_string)[0])
                size = int(re.findall('(?<=CCDPixel=)[0-9.]+', info_string)[0])
                f_sup = np.linspace(spec_min, spec_max, size)
                if data.ndim == 3:
                    shape = data.shape[0]
                    data_reshaped = data.transpose(2,1,0).reshape(-1, shape)
                    return f_sup, np.ascontiguousarray(data_reshaped, dtype=np.float64)
                return f_sup, data.astype('float64')
            except FileNotFoundError:
                print("Wrong filename or path.")
            

    def store_data(data_to_store, filename):
        np.save(filename + '.npy', data_to_store)
        
    def store_large_data(data_to_store, filename):
        np.savez_compressed(filename + '.npz', data_to_store)
        
    def read_npy_file(filename):
        data_npy = np.load(filename)
        return data_npy
    
    def read_npz_file(filename):
        data_npz_dict = np.load(filename)
        data_npz = data_npz_dict['arr_0']
        return data_npz
    
    def remove_zeros_or_nans(data, labels):
        if (np.logical_and(data[:] > 0, data[:] <= 1).all()):
            mask = np.all(np.isfinite(data), axis=-1)
        mask = np.all(data, axis=-1)
        X = data[mask]
        y = labels[mask]
        return X, y
    
    def normalize_data(data):
        normalized_data = data / np.max(data, axis=-1, keepdims=True)
        return normalized_data
    
    def fft_remove_noise(data):
        if data.ndim > 1:
            signal_hstacked = np.hstack((data, np.fliplr(data)))        
        else:
            signal_hstacked = np.hstack((data, np.flip(data)))
        
        signal_spec = np.fft.fftshift(np.fft.fft(signal_hstacked), axes=-1)
        N = data.shape[-1]
        first_segment_hp = np.array(range(0, int(0.005 * N)))
        second_segment_hp = np.array(range(int(0.005 * N), int(0.01 * N)))
        third_segment_hp = np.array(range(int(0.01 * N), N))
        first_segment_lp = np.array(range(0, int(0.25 * N)))
        second_segment_lp = np.array(range(int(0.25 * N), int(0.5 * N)))
        third_segment_lp = np.array(range(int(0.5 * N), N))
        
        

        
        
        # Highpass filtering
        # signal_spec[N + first_segment_hp] = signal_spec[N + first_segment_hp] * ((first_segment_hp * 2) / N)
        # signal_spec[N - first_segment_hp] = signal_spec[N - first_segment_hp] * ((first_segment_hp * 2) / N)
        # signal_spec[N + second_segment_hp] = signal_spec[N + second_segment_hp] * (
        #             ((second_segment_hp - 0.005 * N) / N) * (0.89 / 0.005))
        # signal_spec[N - second_segment_hp] = signal_spec[N - second_segment_hp] * (
        #             ((second_segment_hp - 0.005 * N) / N) * (0.89 / 0.005))
        # signal_spec[N + third_segment_hp] = signal_spec[N + third_segment_hp] * (
        #             ((third_segment_hp - 0.01 * N) / N) * (0.1 / 0.99))
        # signal_spec[N - third_segment_hp] = signal_spec[N - third_segment_hp] * (
        #             ((third_segment_hp - 0.01 * N) / N) * (0.1 / 0.99))
        
        
        
        
        # Lowpass filtering
        signal_spec[N + first_segment_lp] = signal_spec[N + first_segment_lp] * (1 - (first_segment_lp * 0.1 / (0.25 * N)))
        signal_spec[N - first_segment_lp] = signal_spec[N - first_segment_lp] * (1 - (first_segment_lp * 0.1 / (0.25 * N)))
        signal_spec[N + second_segment_lp] = signal_spec[N + second_segment_lp] * (
                    1.7 - ((second_segment_lp * 0.8) / (0.25 * N)))
        signal_spec[N - second_segment_lp] = signal_spec[N - second_segment_lp] * (
                    1.7 - ((second_segment_lp * 0.8) / (0.25 * N)))
        signal_spec[N + third_segment_lp] = signal_spec[N + third_segment_lp] * (0.2 - ((third_segment_lp * 0.1) / (0.5 * N)))
        signal_spec[N - third_segment_lp] = signal_spec[N - third_segment_lp] * (0.2 - ((third_segment_lp * 0.1) / (0.5 * N)))
        
        
        
        # Lowpass filtering
        # signal_spec[:, N + first_segment_lp] = signal_spec[:, N + first_segment_lp] * (
        #             1 - (first_segment_lp * 0.1 / (0.25 * N)))
        # signal_spec[:, N - first_segment_lp] = signal_spec[:, N - first_segment_lp] * (
        #             1 - (first_segment_lp * 0.1 / (0.25 * N)))
        # signal_spec[:, N + second_segment_lp] = signal_spec[:, N + second_segment_lp] * (
        #             1.7 - ((second_segment_lp * 0.8) / (0.25 * N)))
        # signal_spec[:, N - second_segment_lp] = signal_spec[:, N - second_segment_lp] * (
        #             1.7 - ((second_segment_lp * 0.8) / (0.25 * N)))
        # signal_spec[:, N + third_segment_lp] = signal_spec[:, N + third_segment_lp] * (
        #             0.2 - ((third_segment_lp * 0.1) / (0.5 * N)))
        # signal_spec[:, N - third_segment_lp] = signal_spec[:, N - third_segment_lp] * (
        #             0.2 - ((third_segment_lp * 0.1) / (0.5 * N)))
        
    
        signal_ifftd = np.fft.ifft(np.fft.ifftshift(signal_spec, axes=-1)) 
        sig_ifft = np.copy(signal_ifftd[:, 0:1600] / np.max(signal_ifftd, axis=-1, keepdims=True))
        return sig_ifft
        
    def slide_win_smooth_data(data, window):
        smooth = np.array([np.mean(data[int(np.max([j - win, 0])):int(np.min([j + win, data.size]))]) for j in
                       range(y_data.shape[0])])
        return smooth
        
    def recursive_merge(inter, start_index=0):
       for i in range(start_index, len(inter) - 1):
           if inter[i][1] > inter[i + 1][0]:
               new_start = inter[i][0]
               new_end = inter[i + 1][1]
               inter[i] = [new_start, new_end]
               del inter[i + 1]
               return recursive_merge(inter.copy(), start_index=i)
       return inter