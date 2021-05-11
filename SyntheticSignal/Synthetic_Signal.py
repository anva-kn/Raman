# -*- coding: utf-8 -*-
"""
Created on Mon May 10 21:22:15 2021

Synthetic Signal

@author: lab716a
"""

#imports

import numpy as np
import random
import math
from astropy.modeling.models import Voigt1D


class Synthetic_Signal:
    '''
    Create a synthetic signal that's intended to look like a Raman spectrum.
    
    When first initializing the class, it is not necesarry to pass any arguments. The methods will 
    initialize the attributes below when they are called.
    
    Attributes
    -----------------
    N : int
        Number of bins in the signal. How long each signal is.
    
    batch_size : int
        Number of signals to be created.
        
    mu : float
        Mu is avg and is used as argument for gaussian distribution used in random walk.
    
    sigma : float
        S.D. Used as argument for gaussian distribution in random walk.
    
    data : float array [batch_size, N]
        The signals are stored in this array.
        
    peak : voigt-peak object list of length batch_size
        From the astropy library. The object stores HWHF og lorentzian and Gaussian of the voigt peak, as well as location.
    
    snr : float
        Signal-to-noise ratio. Calculated from the power of each after they are generated.
        
    
    Methods
    --------------
    generate_random_walk(N, batch_size, mu=None, sigma=None, reverse=False)
        Generates the baseline, the random walk.
        
    smoothdata(window_sizew=None)
        Smooths the generated random walk by using averaging values within a sliding window.
        
    r_add_peak(fwhmG=None, fwhmL=None, amplitudeL=None)
        Randomly adds a voigt shaped peak defined by the FWHM of its Lorentzian and Gaussian
        components in the smoothed-out random walk. The edges of the curve are off
        limits to ensure the entire peak appears in the curve.
        
    add_noise(mu, sigma, a)
        Generates Gaussian noise and adds it to the random walk.
    '''
    def __init__(self):
        '''
        Initialize Synthetic_Signal class.
        
        No arguments need to be passed for initialization.

        '''
        
        self.N = None
        self.batch_size = None
        self.mu = None
        self.sigma = None
        self.data = None
        self.peak = [None]
        self.snr = None
        
    def __randomWalk__(self, N, mu, sigma):
        walk = np.zeros(N)
        delta = random.gauss(mu, sigma)
        walk[0] = delta
        for i in range(1, N):
            delta = random.gauss(mu, sigma)
            walk[i] = walk[i-1] + delta
        walk += np.min(walk) * -1
        return walk
    
    def __smoothData__(self, y_data, win=None):
        if win == None:
            win = self.N/2.56
        smooth = np.array([np.mean(y_data[int(np.max([j-win,0])):int(np.min([j+win,y_data.size]))]) for j in  range(y_data.shape[0])])
        return smooth
    
    def __voigtPeak__(self, spectrum, x0, amplitudeL, fwhmG, fwhmL):
        voigt_peak = Voigt1D(x_0=x0, amplitude_L=amplitudeL, fwhm_L=fwhmL, fwhm_G=fwhmG)
        return voigt_peak
    
    def __normalizeSpectrum__(self):
        for spectrum in range(0, self.batch_size):
            self.data[spectrum] -= np.min(self.data[spectrum])
            self.data[spectrum] /= np.max(self.data[spectrum]) - np.min(self.data[spectrum])

    def __calculatePower__(self, noise):
        S = np.zeros(self.batch_size)
        n = np.sum(noise)
        n = ((1/noise.shape[0]) * (n**2))
        for spectrum in range(0, self.batch_size):
            S[spectrum] = np.sum(self.data[spectrum])
            S[spectrum] = ((1/self.N) * (S[spectrum]**2) )
        return S, n


    def __calculateSNR__(self, power, noise_power):
        self.snr = np.zeros((self.batch_size))
        for spectrum in range(0, self.batch_size):
            self.snr[spectrum] = ((10* math.log10(power[spectrum])) - (10* math.log10(noise_power)))

        
    def generate_random_walk(self, N, batch_size, mu=None, sigma=None, reverse=False):
        '''
        This method creates your baseline, the random walk. All other methods work on this baseline.
        You must pass N and batch_size parameters.
        All other parameters are optional. If they are not passed, preset values will be used.

        Parameters
        ----------
        N : int
            Bins. The length of each generated random walk.
        batch_size : int
            The number of random walks to generate.
        mu : float, optional
            Average value. Used as parameter for the Gaussian used in generating the walk. The default is 0. 
        sigma : float, optional
            S.D. Used as parametet for Gaussian used in generating the walk. The default is 1.
        reverse : boolean, optional
            It reverses the mu with a probability of 0.5. This allows for decreasing and increasing walks. The default is False.

        Returns
        -------
        None.
        The walk is saved in data.

        '''
        if mu == None:
            mu = 0
        if sigma == None:
            sigma = 1
        self.N = N
        self.batch_size = batch_size
        self.mu = mu
        self.sigma = sigma
        self.data = np.zeros((batch_size, N))
        
        for spectrum in range(0, batch_size):
            if reverse:
                means = [mu, -1*mu]
                mu = np.random.choice(means, p=[0.5,0.5])
            self.data[spectrum] = self.__randomWalk__(N, mu, sigma)
        self.__normalizeSpectrum__()
        
    def smooth_data(self, window_size=None):
        '''
        Average the points in the random walk within the window as it slides down the spectrum.

        Parameters
        ----------
        window_size : float, optional
            Size of the window used for the averaging. The default is N/2.56.

        Returns
        -------
        None.
        data overwritten.

        '''
        for spectrum in range(0, self.batch_size):
            self.data[spectrum] = self.__smoothData__(self.data[spectrum], window_size)
        
    def r_add_peak(self, fwhmG=None, fwhmL=None, amplitudeL=1):
        '''
        Randomly adds a voigt peak in the spectrum. 10% of the signal on either side is not included
        so that the peak is not generated outside the signal.

        Parameters
        ----------
        fwhmG : float, optional
            Full Width Half Max of lerentzian component. The default is N/20.45.
        fwhmL : float, optional
            Full Width Half Max of Gaussian component. The default is N/20.45.
        amplitudeL : float, optional
            Amplitude of L. component. The default is 1.
            
        For more information browse Astropy documentation.

        Returns
        -------
        None.
        data overwritten.
        Information stored in peak.

        '''
        if fwhmG == None:
            fwhmG = self.N/20.45
        if fwhmL == None:
            fwhmL = self.N/20.45
        x = np.arange(0, self.N, 1)
        self.peak = [None] * self.batch_size
        for spectrum in range(0, self.batch_size):
            x0 = random.randint(0 + self.N//10, self.N - self.N//10)
            self.peak[spectrum] = self.__voigtPeak__(self, x0, amplitudeL, fwhmG, fwhmL)
            self.data[spectrum] += np.array(self.peak[spectrum](x))
        self.__normalizeSpectrum__()
        
    def add_noise(self, mu, sigma, a):
        '''
        Generate Gaussian noise and add it to random walk.

        Parameters
        ----------
        mu : float
            For Gaussian.
        sigma : float
            For Gaussian.
        a : float
            modify amplitude of the noise. Larger number for larger amplitudes.

        Returns
        -------
        None.
        data overwritten.

        '''
        noise = np.random.default_rng().normal(mu, sigma, self.N)
        noise -= np.min(noise)
        noise /= np.max(noise) - np.min(noise)
        noise *= a

        power, noise_power = self.__calculatePower__(noise)
        self.__calculateSNR__(power, noise_power) 

        for spectrum in range(0, self.batch_size):
            self.data[spectrum] += noise
        self.__normalizeSpectrum__()