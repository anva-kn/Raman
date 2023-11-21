#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  24 20:42:59 2021

@author: anvarkunanbaev
"""
import os
import numpy as np
import pandas as pd
from PIL import Image
from PIL.TiffTags import TAGS
import re
from skimage import io
from dataclasses import dataclass
import linecache


@dataclass
class ReadData:
    """
    Class for reading data from files.
    """

    # dictionary with file extensions and corresponding methods
    _file_formats = {
        'dat': 'read_dat_file',
        'tif': 'read_tiff_file',
        'txt': 'read_uv_data',
        'npy': 'read_npy_file',
        'npz': 'read_npz_file'
    }

    @classmethod
    def read_data(cls, filename): 
        """
        Returns data from a file based on the file extension.

        Args:
            cls: The class object.
            filename (str): The path to the file.

        Returns:
            The data from the file.

        Example:
        ```python
        filename = 'data.csv'
        [data](VALID_DIRECTORY) = ReadData.read_data(filename)
        print(data)
        """
        _format = re.findall('[^.]+$', filename)[0] # get file extension
        return getattr(cls, cls._file_formats[_format])(filename) # return data from file

    @classmethod
    def read_dat_file(filename: str) -> tuple:
        """
        Reads data from a .dat file and returns the frequency support and data.

        Args:
            filename (str): The path to the .dat file.

        Returns:
            tuple: A tuple containing the frequency support (numpy array) and the data (numpy array).

        Raises:
            FileNotFoundError: If the specified file is not found.

        Example:
        ```python
        filename = 'data.dat'
        f_sup, data = ReadData.read_dat_file(filename)
        plt.plot(f_sup, np.mean(data, axis=0))
        ```
        """
        try:
            data = pd.read_csv(filename, sep="\s+", header=None)  # read data from file
            f_sup = np.array(data.iloc[:, 0], dtype=np.float64)  # get first column as numpy array
            data_array = data.loc[:, 1:].to_numpy(dtype=np.float64)  # get all columns except first as numpy array
            data_array = data_array.T  # transpose array
            if data_array.shape[0] == 1:  # if 2D array
                data_array = np.squeeze(data_array)  # remove single-dimensional entries from the shape of an array
            # np.ascontiguousarray(data_array, dtype=np.float64)  # return a contiguous array (C-style) in memory (ndarray subclass)
            return f_sup, np.ascontiguousarray(data_array, dtype=np.float64)  # return frequency support and data
        except FileNotFoundError:
            print("Wrong filename or path.")

    @classmethod
    def read_tiff_file(cls, filename):
        """
        Reads a TIFF file and returns the frequency support and data as numpy arrays.

        Parameters:
        -----------
        filename : str
            The path to the TIFF file.

        Returns:
        --------
        f_sup : numpy.ndarray
            The frequency support as a 1D numpy array.
        data : numpy.ndarray
            The data as a 2D numpy array, where each row represents a spectrum.
        """
        try:
            data = io.imread(filename)
            with open(filename, 'rb') as f:
                line = f.readlines()[-1]
                # Filter line with regex. Start reading numbers after '<XAxis>' until <End>
                f_sup = re.findall(r'(?<=<XAxis>)(.*?)(?=<End>)', line.decode('unicode_escape'))[0]
                # Split string with numbers into list of strings using coma as a separator
                f_sup = f_sup.split(',')
                # Convert list of strings to list of floats
                f_sup = [float(i) for i in f_sup]
                # Convert list of floats to numpy array
                f_sup = np.array(f_sup, dtype=np.float64)
        
            if data.ndim == 3: # if 3D array
                shape = data.shape[0] # number of spectra
                data_reshaped = data.transpose(2, 1, 0).reshape(-1, shape) # reshape to 2D array
                return f_sup, np.ascontiguousarray(data_reshaped, dtype=np.float64) # return reshaped array
            if data.shape[0] == 1: # if 2D array
                data = np.squeeze(data) 
            return f_sup, data.astype('float64')
        except FileNotFoundError:
            print("Wrong filename or path.")

    @classmethod
    def read_dir_tiff_files(cls, path):
        list_of_files = os.listdir(path) # list of files in the directory
        data = {} # dictionary to store data
        for item in list_of_files: # loop over files
            if re.findall('(?<=\.).*', item) and re.findall('(?<=\.).*', item)[0] == 'tif': # if file has .tif extension
                tmp_item = re.findall('.+?(?=\.)', item)[0] # get file name without extension
                f_sup, data[tmp_item] = cls.read_tiff_file(path + '/' + item) # read file and store data in dictionary
        return f_sup, data # return dictionary with data

    @classmethod
    def read_dir_dat_files(cls, path):
        list_of_files = os.listdir(path) # list of files in the directory
        data = {} # dictionary to store data
        for item in list_of_files:
            if re.findall('(?<=\.).*', item) and re.findall('(?<=\.).*', item)[0] == 'dat': # if file has .dat extension
                tmp_item = re.findall('.+?(?=\.)', item)[0] # get file name without extension
                f_sup, data[tmp_item] = cls.read_dat_file(path + '/' + item) # read file and store data in dictionary
        return f_sup, data # return dictionary with data

    @staticmethod
    def read_uv_data(filename):
        uv_df = pd.read_csv(filename, delimiter='\t', comment='#', header=None) # read data from file
        wavelength = np.flip(uv_df[0].to_numpy(dtype=np.float64)) # get first column as numpy array
        absorbance = np.flip(uv_df[1].to_numpy(dtype=np.float64)) # get second column as numpy array
        return wavelength, absorbance # return wavelength and absorbance

    @staticmethod
    def read_npy_file(filename):
        return np.load(filename, allow_pickle=True)

    @staticmethod
    def read_npz_file(filename):
        data_npz_dict = np.load(filename, allow_pickle=True)
        data_npz = data_npz_dict['arr_0']
        return data_npz
