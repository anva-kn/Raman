#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  24 20:42:59 2021

@author: anvarkunanbaev
"""

import numpy as np
import pandas as pd
from PIL import Image
from PIL.TiffTags import TAGS
import re
from skimage import io


class ReadData:

    @staticmethod
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

    @staticmethod
    def read_tiff_file(filename):
        try:
            data = io.imread(filename)
            with Image.open(filename) as img:
                meta_dict = {TAGS[key]: img.tag[key] for key in img.tag}
            info_string = meta_dict['Software'][0]
            spec_max = float(re.findall('(?<=\SpecMax=)[0-9.]+', info_string)[0])
            spec_min = float(re.findall('(?<=\SpecMin=)[-0-9.]+', info_string)[0])
            size = int(re.findall('(?<=CCDPixel=)[0-9.]+', info_string)[0])
            f_sup = np.linspace(spec_min, spec_max, size)
            if data.ndim == 3:
                shape = data.shape[0]
                data_reshaped = data.transpose(2, 1, 0).reshape(-1, shape)
                return f_sup, np.ascontiguousarray(data_reshaped, dtype=np.float64)
            if data.shape[0] == 1:
                data = np.squeeze(data)
            return f_sup, data.astype('float64')
        except FileNotFoundError:
            print("Wrong filename or path.")

    @staticmethod
    def read_uv_data(filename):
        uv_df = pd.read_csv(filename, delimiter='\t', header=None)
        wavelength = np.flip(uv_df[0].to_numpy(dtype=np.float64))
        absorbance = np.flip(uv_df[1].to_numpy(dtype=np.float64))
        return wavelength, absorbance

    @staticmethod
    def read_npy_file(filename):
        return np.load(filename, allow_pickle=True)

    @staticmethod
    def read_npz_file(filename):
        data_npz_dict = np.load(filename, allow_pickle=True)
        data_npz = data_npz_dict['arr_0']
        return data_npz