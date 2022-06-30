# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 14:21:35 2021

@author: lab716a
"""

import scipy as si
import scipy.signal as ssi
import scipy.fft as sift
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from loss_functions import Loss
from fit_functions import Fit
from importance_functions import Importance

from spectrum_component_v2 import SpectrumComponent

from spectrum import Spectrum


def extrapolate(x_data: np.ndarray, y_data: np.ndarray, pad_left_per: float, pad_right_per: float, pad_left_deg: str, pad_right_deg: str):
    """
    Extrapolates values for a given spectrum.

    Parameters
    ----------
    x_data : np.ndarray
        X axis data.
    y_data : np.ndarray
        Y axis data. A spectrum.
    pad_left_per : float
        Percentage for left extrapolation, given as a decimal. This decides the length of the left spline.
    pad_right_per : float
        Percentage for right extrapolation, given as a decimal. This decides the length of the right spline.
    pad_left_deg : str
        Degree to be used for left extrapolation. One of: 'nearest', 'quadratic', 'cubic', 'linear'.
    pad_right_deg : str
        Degree to be used for right extrapolation. One of: 'nearest', 'quadratic', 'cubic', 'linear'.

    Returns
    -------
    X : np.ndarray
        X axis data after extrapolation.
    Y : np.ndarray
        New Y data with extrapolated values.

    """
    #check that given parameters are correct?
    
    #obtain window sizes
    left_win = pad_left_per * y_data.shape[0]
    right_win = (1.0 - pad_right_per) * y_data.shape[0]
    
    #left extrapolation
    #get slice
    left_y_slice = y_data[:int(left_win)]
    left_x_slice = x_data[:int(left_win)]
    
    #left deltaX
    lDeltaX = left_x_slice[1] - left_x_slice[0]
    
    #obtain new first x value
    lX1 = (2*left_x_slice[0]) - left_x_slice[-1]
    
    #generate extrapolation x values
    lX = np.arange(lX1, left_x_slice[0], lDeltaX)
    
    #extrapolate
    lE = si.interpolate.interp1d(left_x_slice, left_y_slice, kind=pad_left_deg, fill_value='extrapolate')(lX)
    
    #check extrapolation
    plt.figure("Left Extrapolation", figsize=(12, 10), dpi=80)
    plt.title("Left Extrapolation")
    plt.plot(lX, lE)
    plt.plot(left_x_slice, left_y_slice)
    plt.show()
    
    #right extrapolation
    #get slice
    right_y_slice = y_data[int(right_win):]
    right_x_slice = x_data[int(right_win):]
    
    #right deltaX
    rDeltaX = right_x_slice[1] - right_x_slice[0]
    
    #obtain new last x value
    rX1 = (2*right_x_slice[-1]) - right_x_slice[0]
    
    #generate extrapolation x values
    rX = np.arange(right_x_slice[-1], rX1, rDeltaX)
    
    #extrapolate
    rE = si.interpolate.interp1d(right_x_slice, right_y_slice, kind=pad_right_deg, fill_value='extrapolate')(rX)
    
    #check extrapolation
    plt.figure("Right Extrapolation", figsize=(12, 10), dpi=80)
    plt.title("Right Extrapolation")
    plt.plot(rX, rE)
    plt.plot(right_x_slice, right_y_slice)
    plt.show()
    
    #bring it all together
    X = np.append(lX, np.append(x_data, rX))
    Y = np.append(lE, np.append(y_data, rE))
    
    plt.figure("Full Extrapolation", figsize=(12, 10), dpi=80)
    plt.title("Full Extrapolation")
    plt.plot(X, Y)
    plt.show()
    
    return X, Y
    

def noise_aug(y_data: np.ndarray, amp: float):
    """
    Adds gaussian noise to the spectrum in direction of its variance.

    Parameters
    ----------
    y_data : np.ndarray
        Y data. A spectrum.
    amp : float
        A percentage given as a decimal. Strength of noise.

    Returns
    -------
    np.ndarray
        Spectrum plus noise.

    """    
    #adding noise to the spectrum
    n = np.random.normal(0, y_data.std(), y_data.shape[-1]) * amp
    return y_data + n


def merge_aug(y1: np.ndarray, x1: np.ndarray, y2: np.ndarray, x2: np.ndarray, degree: np.ndarray):
    """
    Sums two spectra together. The spectra must be of the same resolution in order to be merged,
    but in case they are not, an interpolation will be used to match their resolutions.

    Parameters
    ----------
    y1 : np.ndarray
        Y data. A spectrum.
    x1 : np.ndarray
        X data of spectrum y1.
    y2 : np.ndarray
        Y data. A spectrum.
    x2 : np.ndarray
        X data of spectrum y2.
    degree : np.ndarray
        Degree of interpolation for the extension if the spectra are of different resolution.

    Returns
    -------
    np.ndarray
        Sum of the two spectra passed as arguments.

    """
    #First spectrum is of greater res
    if y1.shape[-1] > y2.shape[-1]:
        f = si.interpolate.interp1d(x2, y2, kind=degree)
        new_x = np.linspace(x1[0], x1[-1], x1.shape[-1], endpoint=True)
        new_y2 = f(new_x)
        return new_y2 + y1
    
    #Second spectrum is of greater res
    elif y1.shape[-1] < y2.shape[-1]:
        f = si.interpolate.interp1d(x1, y1, kind=degree)
        new_x = np.linspace(x2[0], x2[-1], x2.shape[-1], endpoint=True)
        new_y1 = f(new_x)
        return new_y1 + y2
    
    #Spectra are of equal res
    else:
        return y1 + y2


def reverse_aug(y_data: np.ndarray):
    """
    Reverses the spectrum.

    Parameters
    ----------
    y_data : np.ndarray
        DESCRIPTION.

    Returns
    -------
    np.ndarray
        Reversed spectrum.

    """
    #pancake
    return np.flip(y_data)


def ft_aug(x_data, y_data):
    yf = si.fft(y_data)
    
    #b is the numerator of the filter and a is the denominator of the filter
    b,a = ssi.butter(N=10, Wn=0.5, btype='low', output='ba')
    #w is freq in z domain, h is magnitude in z domain
    w,h = ssi.freqz(b, a)
    plt.semilogx(w, 20 * np.log10(abs(h)))
    plt.title('Butterworth filter frequency response')
    plt.xlabel('Frequency [radians / second]')
    plt.ylabel('Amplitude [dB]')
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    plt.axvline(100, color='green') # cutoff frequency
    plt.show()
    
    imp = ssi.unit_impulse(20)
    c,d = ssi.butter(10, 0.5)
    response = ssi.lfilter(c, d, imp)
      
    plt.stem(np.arange(0, 20), imp, use_line_collection=True)
    plt.stem(np.arange(0, 20), response, use_line_collection=True)
    plt.margins(0, 0.01)
      
    plt.xlabel('Time [samples]')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()
    
    snc = np.sinc(np.linspace(-4,4,41))
    
    plt.plot(snc)
    plt.show()
    
    sncf = sift.fft(snc)
    
    plt.plot(sncf)
    plt.show()
    
    
    
xls = pd.ExcelFile("clean_spectrum.xlsx")

f_sup = np.array(pd.read_excel(xls, 'f_sup')).squeeze()
dict_data = ["dark", "laser", "quartz", "water", "gly", "leu", "phe", "trp"]
data = {}
for i in dict_data:
    data[i] = np.array(pd.read_excel(xls, i))
data_m = {}

for i in dict_data:
    data_m[i] = np.mean(np.array(pd.read_excel(xls, i)), axis=0)
g = data_m["gly"] - data_m["water"]
l = data_m["leu"] - data_m["water"]

y_data = g

y_data = (y_data-np.min(y_data) / np.max(y_data)-np.min(y_data))
x_data = f_sup

reverse_ydata = reverse_aug(y_data)

plt.figure("Reversed Spectrum", figsize=(12, 10), dpi=80)
plt.title("Reverse")
plt.plot(reverse_ydata)
plt.show()


y2_data = l
y2_data = (y2_data-np.min(y_data) / np.max(y2_data)-np.min(y2_data))
x2_data = np.linspace(x_data[0], 2064.73, 1500, endpoint=True)


yn = noise_aug(y_data, 0.05)

plt.figure("Noise Addition", figsize=(12, 10), dpi=80)
plt.title("Original and Noise Added")
plt.plot(x_data, yn)
plt.show()

extrapolate(x_data, y_data, 0.1, 0.1, 'linear', 'nearest')
