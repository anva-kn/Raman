#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  25 20:45:12 2021

@author: Anvar Kunanbayev
@author: Sergio Diaz
"""
import numpy as np
from scipy.optimize import minimize
import scipy.interpolate as si
import matplotlib.pyplot as plt
from tools.ramanflow.loss_functions import pos_mse_loss, positive_mse, poly4, reg_pos_mse_loss, reg_positive_mse, \
    reg_pos_loss, mse_loss
from scipy import interpolate as si
from itertools import takewhile
from joblib import Parallel, delayed


class PrepData:

    @staticmethod
    def store_data(data_to_store, filename):
        """
        Store array-like data into .npy file.
        
        Parameters
        ----------
        data_to_store : array-like.
            Data to store into file.
        filename : string.
            Name of file.
        """
        np.save(filename + '.npy', data_to_store)

    @staticmethod
    def store_large_data(data_to_store, filename):
        '''

        Parameters
        ----------
        data_to_store
        filename

        Returns
        -------

        '''
        np.savez_compressed(filename + '.npz', data_to_store)

    @staticmethod
    def remove_zeros_or_nans(data, labels=None):
        '''
        Remove rows from the input data where all values are zero or NaN.

        Parameters
        ----------
        data : numpy.ndarray
            The input array of spectral data.
        labels : numpy.ndarray, optional
            The corresponding array of labels for the spectral data.

        Returns
        -------
        numpy.ndarray
            The input array with zero or NaN rows removed.
        numpy.ndarray, optional
            The corresponding array of labels with zero or NaN rows removed.

        Notes
        -----
        This function removes rows from the input data where all values are zero or NaN. This is useful for removing
        low-quality or missing data from the dataset.

        Examples
        --------
        >>> data = np.array([[1, 2, 3], [0, 0, 0], [4, np.nan, 6]])
        >>> labels = np.array([1, 2, 3])
        >>> PrepData.remove_zeros_or_nans(data, labels)
        (array([[1., 2., 3.],
                [4., np.nan, 6.]]),
         array([1, 3]))
        '''
        if (np.logical_and(data[:] > 0, data[:] <= 1).all()):
            mask = np.all(np.isfinite(data), axis=-1)
        mask = np.all(data, axis=-1)
        X = data[mask]
        y = labels[mask]
        return X, y

    @staticmethod
    def normalize_data(data):
        '''
        Normalize the input data by dividing each row by the maximum value in that row.

        Parameters
        ----------
        data : numpy.ndarray
            The input array of spectral data.

        Returns
        -------
        numpy.ndarray
            The normalized array of spectral data.

        Notes
        -----
        This function normalizes the input data by dividing each row by the maximum value in that row. This ensures that
        the maximum value in each row is equal to 1, which can be useful for comparing spectra with different intensity
        scales.

        Examples
        --------
        >>> data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> PrepData.normalize_data(data)
        array([[0.33333333, 0.66666667, 1.        ],
               [0.66666667, 0.83333333, 1.        ],
               [0.77777778, 0.88888889, 1.        ]])
        '''
        normalized_data = data / np.max(data, axis=-1, keepdims=True)
        return normalized_data

    @staticmethod
    def remove_noise_fft(data) -> np.ndarray:
        """
        Applies a Fast Fourier Transform (FFT) to the input data, processes the FFT result to remove noise from the signal,
        and applies an Inverse Fast Fourier Transform (IFFT) to obtain the final output.

        This method uses the `numexpr` library to evaluate mathematical expressions efficiently.

        Parameters
        ----------
        data : numpy.ndarray
            The input array of spectral data.

        Returns
        -------
        numpy.ndarray
            The processed version of the input data.
        """
        
        N = data.shape[-1]
        first_segment_lp = np.arange(0, int(0.25 * N))
        second_segment_lp = np.arange(int(0.25 * N), int(0.5 * N))
        third_segment_lp = np.arange(int(0.5 * N), N)

        # Prepare the data for FFT by stacking it with its flipped version
        data_2d = np.atleast_2d(data)
        signal_hstacked = np.hstack((data_2d, np.fliplr(data_2d)))
        signal_spec = np.fft.fftshift(np.fft.fft(signal_hstacked), axes=-1)

        # Apply the noise removal expressions to the signal spectrum
        signal_spec[:, N + first_segment_lp] = signal_spec[:, N + first_segment_lp] * (
                    1 - (first_segment_lp * 0.1 / (0.25 * N)))
        signal_spec[:, N - first_segment_lp] = signal_spec[:, N - first_segment_lp] * (
                    1 - (first_segment_lp * 0.1 / (0.25 * N)))
        signal_spec[:, N + second_segment_lp] = signal_spec[:, N + second_segment_lp] * (
                    1.7 - ((second_segment_lp * 0.8) / (0.25 * N)))
        signal_spec[:, N - second_segment_lp] = signal_spec[:, N - second_segment_lp] * (
                    1.7 - ((second_segment_lp * 0.8) / (0.25 * N)))
        signal_spec[:, N + third_segment_lp] = signal_spec[:, N + third_segment_lp] * (
                    0.2 - ((third_segment_lp * 0.1) / (0.5 * N)))
        signal_spec[:, N - third_segment_lp] = signal_spec[:, N - third_segment_lp] * (
                    0.2 - ((third_segment_lp * 0.1) / (0.5 * N)))
        
        # Apply IFFT to obtain the processed signal
        signal_ifftd = np.fft.ifft(np.fft.ifftshift(signal_spec, axes=-1))

        # Flatten and trim the output if necessary
        if signal_ifftd.ndim == 2 and signal_ifftd.shape[0] == 1:
            signal_ifftd = np.ravel(signal_ifftd)
            return signal_ifftd[:data.shape[-1]]
    
        sig_ifft = signal_ifftd[:, :data.shape[-1]]

        return sig_ifft

    @classmethod
    def remove_cosmic_rays(cls, data, window):
        '''
        Remove cosmic rays from a 1D or 2D array of spectral data.

        Parameters
        ----------
        data : numpy.ndarray
            The input array of spectral data. If `data` is a 2D array, the function will be applied to each row or column
            of the array recursively.
        window : int
            The number of points to include on either side of a cosmic ray when calculating the replacement value.

        Returns
        -------
        numpy.ndarray
            The input array with cosmic rays removed.

        Notes
        -----
        Cosmic rays are identified as points where the absolute difference between consecutive points is greater than
        three times the standard deviation of the differences. The replacement value for each cosmic ray is calculated as
        the mean of the points within `window` points on either side of the cosmic ray.

        Examples
        --------
        >>> data = np.array([1, 2, 3, 100, 5, 6, 7])
        >>> PrepData.remove_cosmic_rays(data, window=1)
        array([1, 2, 3, 4, 5, 6, 7])

        >>> data = np.array([[1, 2, 3], [100, 5, 6], [7, 8, 9]])
        >>> PrepData.remove_cosmic_rays(data, window=1)
        array([[1, 2, 3],
               [53, 5, 6],
               [7, 8, 9]])
        '''

        # Find the difference between consecutive points
        delta_data = np.abs(np.diff(data, axis=-1))

        # If the input is a 2D array, apply the function parallelly on each row or column
        if data.ndim > 1:
            return np.array(Parallel(n_jobs=-1)(
                delayed(cls.remove_cosmic_rays)(row, window) for row in data
            ))

        # Find the indices of cosmic rays
        cosmic_ray_indices = np.where(delta_data > 3 * np.std(delta_data))[0] + 1

        for i in cosmic_ray_indices:
            # Select points around spikes within bounds
            w = np.arange(max(0, i - window), min(i + 1 + window, len(data)))

            # Select points apart from spikes
            w2 = np.setdiff1d(w, cosmic_ray_indices)

            # Clip points that are out of bounds
            arr = np.take(data, w2, mode='clip')

            # Replace cosmic ray with the average of selected points
            data[i] = np.mean(arr)

            # # Update delta_data after the modifications
            # delta_data = np.abs(np.diff(data, axis=-1))

            # # Find cosmic ray indices again
            # cosmic_ray_indices = np.where(delta_data > 3 * np.std(delta_data))[0] + 1

        return data
    
    @classmethod
    def remove_cosmic_rays_old(cls, data, window):

        data_out = np.copy(data)  # Copy the data to not modify original
        delta_data = np.abs(np.diff(data_out))  # Find the difference between consecutive points

        if data.ndim > 1:
            # If you feed the array of spectra, then the function will run recursively on each individual spectrum
            return np.apply_along_axis(func1d=cls.remove_cosmic_rays, axis=-1, arr=data_out, window=window)
        else:
            # Find the outliers (outlier > 3 standard deviations)
            # + 1 for the correct index
            cosmic_ray_indices = np.where(delta_data > 3 * np.std(delta_data))[0] + 1
            for i in cosmic_ray_indices:
                w = np.arange(i - window, i + 1 + window)  # select 2*window + 1 points around spikes
                w2 = w[np.in1d(w, cosmic_ray_indices) == False]  # Select points apart from spikes
                arr = np.take(data, w2, mode='clip')  # Check if selected points raise out of bound, if yes-clip them
                data_out[i] = np.mean(arr)  # Substitute spike with the average of the selected points
            return data_out

    @staticmethod
    def poly_remove_est_florescence(f_sup, data_sub, loss):
        '''

        Parameters
        ----------
        f_sup: Your frequency support vector. 1D array
        data_sub: Your spectra. Can be 1D or 2D
        loss: String. Possible options:
            1) pos-mse
            2) mse
            3) reg-pos-mse
        Returns
        -------
        array like data_sub
        '''
        global result
        if data_sub.ndim > 1:
            min_data_amm = np.mean(data_sub, axis=0)
        else:
            min_data_amm = data_sub

        poly_min = np.poly1d(np.polyfit(f_sup, min_data_amm, 3))(f_sup)
        beta_init = np.polyfit(f_sup, min_data_amm, 4)
        beta_init[4] = beta_init[4] + min(min_data_amm - poly_min)

        if loss == 'pos-mse':
            result = minimize(pos_mse_loss, beta_init, args=(f_sup, min_data_amm, poly4), method='Nelder-Mead',
                              tol=1e-12)
        elif loss == 'mse':
            result = minimize(mse_loss, beta_init, args=(f_sup, min_data_amm, poly4), method='Nelder-Mead', tol=1e-12)
        elif loss == 'reg-pos-mse':
            result = minimize(reg_pos_mse_loss, beta_init, args=(f_sup, min_data_amm, poly4), method='Nelder-Mead',
                              tol=1e-12)
        beta_hat = result.x
        # subtract mean
        if data_sub.ndim > 1:
            data_sub2 = np.subtract(data_sub, poly4(f_sup, beta_hat).reshape((data_sub.shape[-1])))
        else:
            data_sub2 = data_sub - poly4(f_sup, beta_hat)

        return data_sub2

    @classmethod
    def recursive_merge(cls, inter, start_index=0):
        for i in range(start_index, len(inter) - 1):
            if inter[i][1] > inter[i + 1][0]:
                new_start = inter[i][0]
                new_end = inter[i + 1][1]
                inter[i] = [new_start, new_end]
                del inter[i + 1]
                return recursive_merge(inter.copy(), start_index=i)
        return inter

    @classmethod
    def spline_remove_est_fluorescence(cls, x_data, y_data, slide_win) -> np.ndarray:
        '''
        Removes estimated fluorescence from spectra using spline interpolation.

        Parameters:
        -----------
        x_data : ndarray
            Your frequency support vector. Should be a 1D array.
        y_data : ndarray
            Spectra data. Can be 1D or 2D array.
        slide_win : int
            Sliding window size for interpolation.

        Returns:
        --------
        ndarray
            Spectra with estimated fluorescence removed.

        Notes:
        ------
        This function performs spline interpolation to estimate and remove fluorescence from spectra data. 
        The algorithm scans all points and finds the points that minimize the variance of the data minus the spline interpolation.

        The function returns the spectra with the estimated fluorescence removed.

        Example usage:
        --------------
        x = np.linspace(0, 10, 100)
        y = np.random.rand(100)
        corrected_y = spline_remove_est_fluorescence(x, y, 10)
        '''
        # maybe last x value? If so, obtain it instead of passing it.
        res = x_data.shape[-1]
        interpol_mse = [1000, 1000, 1000]
        slide_win = slide_win
        nu = 0.01
        if y_data.ndim > 1:
            data = np.mean(y_data, axis=0)
        else:
            data = np.copy(y_data)
        # find the points that minimizes the variance of the data minus spline interpolation
        # scan all points, who cares

        # Did away with set() and subtraction of empty set
        idx_left = list(range(res))
        interpol_pos = [0, np.argmin(data[idx_left]), res - 1]

        for i in range(2 * int(data.shape[-1] / slide_win)):
            min_pos = 0
            # interp1d returns a function, immediately used on x_data.
            y_hat = si.interp1d(x_data[interpol_pos], data[interpol_pos])(x_data)
            min_mse = (1 - nu) * np.dot((y_hat - data > 0), np.abs((y_hat - data))) + nu * np.var(y_hat - data)
            tmp_interpol_pos = list(
                set(range(int(i * (slide_win / 2)), min(int(i * (slide_win / 2) + (slide_win)), res))))

            for k in range(len(tmp_interpol_pos)):
                tmp_pos = np.concatenate((interpol_pos, [tmp_interpol_pos[k]]))
                # extrapolate lets it run but sometimes causes wrong results
                y_hat = si.interp1d(x_data[tmp_pos], data[tmp_pos])(x_data)
                # generalize to any loss
                tmp_mse = (1 - nu) * np.dot((y_hat - data > 0), np.abs((y_hat - data))) + nu * np.var(
                    y_hat - data)
                # update the minimum
                if tmp_mse < min_mse:
                    min_pos = tmp_interpol_pos[k]
                    min_mse = tmp_mse

            interpol_pos.append(min_pos)
            interpol_mse.append(min_mse)
            # set doesn't allow dups
            unique_pos = np.array([int(interpol_pos.index(x)) for x in set(interpol_pos)])
            interpol_pos = list(np.array(interpol_pos)[unique_pos.astype(int)])
            interpol_mse = list(np.array(interpol_mse)[unique_pos.astype(int)])
            # sort points
            sort_pos = np.argsort(interpol_pos)
            interpol_pos = list(np.array(interpol_pos)[sort_pos.astype(int)])
            interpol_mse = list(np.array(interpol_mse)[sort_pos.astype(int)])
            # remove points that are too close

        y_hat = si.interp1d(x_data[interpol_pos], data[interpol_pos])(x_data)
        y_bsp = np.poly1d(np.polyfit(x_data[interpol_pos], data[interpol_pos], 3))(x_data)

        th = np.std(data - y_bsp)
        pos = np.array(np.where((data - y_bsp) < th)[0])

        #   merge the points
        # pos[n+1] - pos[n] - 1; first and last element excluded from second mapping to prevent out of bounds
        diff_pos = np.diff(pos) - 1
        jumps = np.where(diff_pos > 0)[0]

        # if the final element is res, the add a jomp an the end
        if pos[-1] == res - 1:
            jumps = np.append(jumps, pos.shape[0] - 1)

        final_lb = []
        final_rb = []

        if jumps.size == 0:
            final_lb.append(pos[0])
            final_rb.append(pos[-1])

        else:
            final_lb.append(pos[0])
            final_rb.append(pos[jumps[0]])
            k = 0

            while k < jumps.shape[0] - 1:
                final_lb.append(pos[jumps[k] + 1])
                # go to the next gap
                k = k + 1
                final_rb.append(pos[jumps[k]])

        # add the first and the last intervals
        idx_lr = np.zeros([2, len(final_rb)])
        idx_lr[0] = np.array(final_lb)
        idx_lr[1] = np.array(final_rb)
        idx_lr.astype(int)
        idx_lr = idx_lr.T

        # merge intervals
        # remove the one intervals
        idx_lr = idx_lr[np.where(idx_lr[:, 1] - idx_lr[:, 0] > 2)[0], :]
        # Looks like it does nothing here, as they are ordered already. unless something else was meant
        merged = cls.recursive_merge(idx_lr.tolist())
        idx_lr = np.array(merged).astype(int)
        # undid copy with missing parts
        # at this point we can maybe return values
        return y_data - y_hat

    @classmethod
    def spline_find_fluorescence(cls, x_data, y_data, slide_win):
        '''
        Finds the estimated fluorescence in spectra using spline interpolation.

        Parameters:
        -----------
        x_data : ndarray
            Your frequency support vector. Should be a 1D array.
        y_data : ndarray
            Spectra data. Can be 1D or 2D array.
        slide_win : int
            Sliding window size for interpolation.

        Returns:
        --------
        ndarray
            Estimated fluorescence values for each point in the spectra.

        Notes:
        ------
        This function performs spline interpolation to estimate the fluorescence in the spectra data. It scans all points and finds the points that minimize the variance of the data minus the spline interpolation.

        Example usage:
        --------------
        x = np.linspace(0, 10, 100)
        y = np.random.rand(100)
        fluorescence = spline_find_fluorescence(x, y, 10)
        '''

        res = x_data.shape[-1]
        interpol_mse = [1000, 1000, 1000]
        slide_win = slide_win
        nu = 0.01

        if y_data.ndim > 1:
            data = np.mean(y_data, axis=0)
        else:
            data = np.copy(y_data)

        idx_left = list(range(res))
        interpol_pos = [0, np.argmin(data[idx_left]), res - 1]

        for i in range(2 * int(data.shape[-1] / slide_win)):
            min_pos = 0
            y_hat = si.interp1d(x_data[interpol_pos], data[interpol_pos])(x_data)
            min_mse = (1 - nu) * np.dot((y_hat - data > 0), np.abs((y_hat - data))) + nu * np.var(y_hat - data)
            tmp_interpol_pos = list(set(range(int(i * (slide_win / 2)), min(int(i * (slide_win / 2) + (slide_win)), res))))

            for k in range(len(tmp_interpol_pos)):
                tmp_pos = np.concatenate((interpol_pos, [tmp_interpol_pos[k]]))
                y_hat = si.interp1d(x_data[tmp_pos], data[tmp_pos])(x_data)
                tmp_mse = (1 - nu) * np.dot((y_hat - data > 0), np.abs((y_hat - data))) + nu * np.var(y_hat - data)

                if tmp_mse < min_mse:
                    min_pos = tmp_interpol_pos[k]
                    min_mse = tmp_mse

            interpol_pos.append(min_pos)
            interpol_mse.append(min_mse)
            unique_pos = np.array([int(interpol_pos.index(x)) for x in set(interpol_pos)])
            interpol_pos = list(np.array(interpol_pos)[unique_pos.astype(int)])
            interpol_mse = list(np.array(interpol_mse)[unique_pos.astype(int)])
            sort_pos = np.argsort(interpol_pos)
            interpol_pos = list(np.array(interpol_pos)[sort_pos.astype(int)])
            interpol_mse = list(np.array(interpol_mse)[sort_pos.astype(int)])

        y_hat = si.interp1d(x_data[interpol_pos], data[interpol_pos])(x_data)
        return y_hat

    @classmethod
    def spline_remove_fluorescence(cls, x_data, y_data, slide_win):
        '''
        Removes estimated fluorescence from spectra using spline interpolation.

        Parameters:
        -----------
        x_data : ndarray
            Your frequency support vector. Should be a 1D array.
        y_data : ndarray
            Spectra data. Can be 1D or 2D array.
        slide_win : int
            Sliding window size for interpolation.

        Returns:
        --------
        ndarray
            Spectra with estimated fluorescence removed.

        Notes:
        ------
        This function performs spline interpolation to estimate and remove fluorescence from spectra data.
        It calls `spline_find_fluorescence` to obtain the estimated fluorescence values and subtracts them from the original spectra.

        Example usage:
        --------------
        x = np.linspace(0, 10, 100)
        y = np.random.rand(100)
        corrected_y = spline_remove_fluorescence(x, y, 10)
        '''

        fluorescence = cls.spline_find_fluorescence(x_data, y_data, slide_win)
        return y_data - fluorescence