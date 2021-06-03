#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  25 20:45:12 2021

@author: anvarkunanbaev
"""
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from .loss_functions import pos_mse_loss, positive_mse, poly4, reg_pos_mse_loss, reg_positive_mse, reg_pos_loss

class PrepData:

    @staticmethod
    def store_data(data_to_store, filename):
        np.save(filename + '.npy', data_to_store)

    @staticmethod
    def store_large_data(data_to_store, filename):
        np.savez_compressed(filename + '.npz', data_to_store)

    @staticmethod
    def remove_zeros_or_nans(data, labels):
        if (np.logical_and(data[:] > 0, data[:] <= 1).all()):
            mask = np.all(np.isfinite(data), axis=-1)
        mask = np.all(data, axis=-1)
        X = data[mask]
        y = labels[mask]
        return X, y

    @staticmethod
    def normalize_data(data):
        normalized_data = data / np.max(data, axis=-1, keepdims=True)
        return normalized_data

    @staticmethod
    def fft_remove_noise(data):
        if data.ndim > 1:
            signal_hstacked = np.hstack((data, np.fliplr(data)))
        else:
            signal_hstacked = np.hstack((data, np.flip(data)))

        signal_spec = np.fft.fftshift(np.fft.fft(signal_hstacked), axes=-1)
        N = data.shape[-1]
        first_segment_lp = np.array(range(0, int(0.25 * N)))
        second_segment_lp = np.array(range(int(0.25 * N), int(0.5 * N)))
        third_segment_lp = np.array(range(int(0.5 * N), N))

        if data.ndim > 1:
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
            signal_ifftd = np.fft.ifft(np.fft.ifftshift(signal_spec, axes=-1))
            sig_ifft = np.copy(signal_ifftd[:, 0:1600])
            return sig_ifft
        else:
            signal_spec[N + first_segment_lp] = signal_spec[N + first_segment_lp] * (
                        1 - (first_segment_lp * 0.1 / (0.25 * N)))
            signal_spec[N - first_segment_lp] = signal_spec[N - first_segment_lp] * (
                        1 - (first_segment_lp * 0.1 / (0.25 * N)))
            signal_spec[N + second_segment_lp] = signal_spec[N + second_segment_lp] * (
                    1.7 - ((second_segment_lp * 0.8) / (0.25 * N)))
            signal_spec[N - second_segment_lp] = signal_spec[N - second_segment_lp] * (
                    1.7 - ((second_segment_lp * 0.8) / (0.25 * N)))
            signal_spec[N + third_segment_lp] = signal_spec[N + third_segment_lp] * (
                        0.2 - ((third_segment_lp * 0.1) / (0.5 * N)))
            signal_spec[N - third_segment_lp] = signal_spec[N - third_segment_lp] * (
                        0.2 - ((third_segment_lp * 0.1) / (0.5 * N)))
            signal_ifftd = np.fft.ifft(np.fft.ifftshift(signal_spec, axes=-1))
            sig_ifft = np.copy(signal_ifftd[0:1600])
            return sig_ifft


    @staticmethod
    def remove_cosmic_rays(f_sup, data, window):
        # if data.ndim > 1:
        data_out = np.copy(data)
        delta_data = np.abs(np.diff(data_out))
        delta_f_sup = np.abs(np.diff(f_sup))
        slope = delta_data / delta_f_sup
        cosmic_ray_indices = np.where(slope > 3 * np.std(slope))[0]
        for i in cosmic_ray_indices:
            w = np.arange(i - window, i + 1 + window)  # select 2*window + 1 points around spikes
            w2 = w[np.in1d(w, cosmic_ray_indices) == False]
            data_out[i] = np.mean(data[w2])
        return data_out

    @staticmethod
    def poly_remove_est_florescence(f_sup, data_sub, loss):
        # min_data_amm=np.min(data_sub,axis=1)
        global result
        if data_sub.ndim > 1:
            min_data_amm = np.mean(data_sub, axis=0)
        else:
            min_data_amm = data_sub

        poly_min = np.poly1d(np.polyfit(f_sup, min_data_amm, 3))(f_sup)
        poly_min_pos = poly_min + min(min_data_amm - poly_min)

        beta_init = np.polyfit(f_sup, min_data_amm, 4)
        beta_init[4] = beta_init[4] + min(min_data_amm - poly_min)

        if loss == 'pos-mse':
            result = minimize(pos_mse_loss, beta_init, args=(f_sup, min_data_amm, poly4), method='Nelder-Mead',
                          tol=1e-12)
        elif loss == 'mse':
            result = minimize(mse_loss, beta_init, args=(f_sup, min_data_amm, poly4), method='Nelder-Mead', tol=1e-12)
        elif loss == 'reg-pos-mse':
            result = minimize(reg_pos_mse_loss, beta_init, args=(f_sup, min_data_amm, poly4), method='Nelder-Mead', tol=1e-12)

        beta_hat = result.x

        plt.figure('Pos MSE fit')
        # plt.plot(f_sup,np.mean(data_sub,axis=1),'--',label='original data')
        plt.plot(f_sup, min_data_amm, label='original data')
        # plt.plot(f_sup,np.mean(data_sub,axis=1),'-.',label='data minus bias')
        plt.plot(f_sup, poly4(f_sup, beta_init), '-.', label='poly 4 init')
        plt.plot(f_sup, poly4(f_sup, beta_hat), '-.', label='poly 4 hat')
        plt.legend()
        # subtract mean
        if data_sub.ndim > 1:
            data_sub2 = np.subtract(data_sub, poly4(f_sup, beta_hat).reshape([data_sub.shape[0], 1]))
        else:
            data_sub2 = data_sub - poly4(f_sup, beta_hat)

        plt.figure('Final recap')
        plt.plot(f_sup, poly4(f_sup, beta_hat), '--', label='poly 4 hat')
        plt.plot(f_sup, min_data_amm, label='original data-back')
        if data_sub.ndim > 1:
            plt.plot(f_sup, np.mean(data_sub2, axis=1), label='original data-back-pos_MSE')
        else:
            plt.plot(f_sup, data_sub2, label='original data-back-pos_MSE')

        plt.legend()

        return data_sub2

    @staticmethod
    def spline_remove_est_fluorescence(x_data, y_data, res, num_th, slide_win):
        res = res
        num_th = num_th
        peak_pos = []
        interpol_mse = []
        slide_win = slide_win
        nu = 0.01
        interpol_mse = interpol_mse + [1000, 1000, 1000]

        for j in range(num_th):
            # find the points that minimizes the variance of the data minus spline interpolation
            # scan all points, who cares
            idx_left = list(set(range(res)) - set(peak_pos))
            interpol_pos = []
            interpol_pos = [0, np.argmin(y_data[idx_left]), res - 1]
            # for i in range(int(y_data.shape[0]/slide_win)):
            for i in range(2 * int(y_data.shape[0] / slide_win)):
                min_pos = 0
                y_hat = si.interp1d(x_data[interpol_pos], y_data[interpol_pos])(x_data)
                min_mse = (1 - nu) * np.dot((y_hat - y_data > 0), np.abs((y_hat - y_data))) + nu * np.var(
                    y_hat - y_data)
                tmp_interpol_pos = list(
                    set(range(int(i * (slide_win / 2)), min(int(i * (slide_win / 2) + (slide_win)), res))) - set(
                        peak_pos))
                for k in range(len(tmp_interpol_pos)):
                    tmp_pos = np.concatenate((interpol_pos, [tmp_interpol_pos[k]]))
                    y_hat = si.interp1d(x_data[tmp_pos], y_data[tmp_pos])(x_data)
                    # generalize to any loss
                    # tmp_mse=np.var(y_hat-y_data)
                    tmp_mse = (1 - nu) * np.dot((y_hat - y_data > 0), np.abs((y_hat - y_data))) + nu * np.var(
                        y_hat - y_data)
                    # update the minimum
                    if tmp_mse < min_mse:
                        min_pos = tmp_interpol_pos[k]
                        min_mse = tmp_mse
                interpol_pos.append(min_pos)
                # interpol_pos.sort()
                interpol_mse.append(min_mse)
                unique_pos = np.array([int(interpol_pos.index(x)) for x in set(interpol_pos)])
                interpol_pos = list(np.array(interpol_pos)[unique_pos.astype(int)])
                interpol_mse = list(np.array(interpol_mse)[unique_pos.astype(int)])
                # sort points
                sort_pos = np.argsort(interpol_pos)
                interpol_pos = list(np.array(interpol_pos)[sort_pos.astype(int)])
                interpol_mse = list(np.array(interpol_mse)[sort_pos.astype(int)])
                # remove points that are too close
            y_hat = si.interp1d(x_data[interpol_pos], y_data[interpol_pos])(x_data)
            y_bsp = np.poly1d(np.polyfit(x_data[interpol_pos], y_data[interpol_pos], 3))(x_data)
            # y_bsp=si.interp1d(x_data[interpol_pos],y_data[interpol_pos], kind='cubic')(x_data)
            while True:
                plt.plot(x_data, y_data)
                plt.plot(x_data, y_hat)
                plt.plot(x_data, y_bsp)
            mean_level = y_bsp
            # --------------------------------------------------------------------
            #   find when you acre over the mean level
            # --------------------------------------------------------------------
            # th=2*np.sqrt(np.var(y_data-mean_level))
            th = np.sqrt(np.var(y_data - mean_level))
            pos = np.array(np.where((y_data - mean_level) < th)[0])
            while True:
                plt.plot(x_data, y_data)
                plt.plot(x_data, y_bsp)
                plt.plot(x_data[pos], y_bsp[pos], '*')
            # --------------------------------------------------------------------
            #   merge the points
            # --------------------------------------------------------------------
            diff_pos = pos[1:] - pos[:-1] - 1
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
                    #
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
            merged = recursive_merge(idx_lr.tolist())
            idx_lr = np.array(merged).astype(int)
            idx_lr_poly = idx_lr
        for j in range(num_th):
            # find the points that minimizes the variance of the data minus spline interpolation
            # scan all points, who cares
            idx_left = list(set(range(res)) - set(peak_pos))
            interpol_pos = []
            interpol_pos = [0, np.argmin(y_data[idx_left]), res - 1]
            # for i in range(int(y_data.shape[0]/slide_win)):
            for i in range(2 * int(y_data.shape[0] / slide_win)):
                min_pos = 0
                y_hat = si.interp1d(x_data[interpol_pos], y_data[interpol_pos])(x_data)
                min_mse = (1 - nu) * np.dot((y_hat - y_data > 0), np.abs((y_hat - y_data))) + nu * np.var(
                    y_hat - y_data)
                tmp_interpol_pos = list(
                    set(range(int(i * (slide_win / 2)), min(int(i * (slide_win / 2) + (slide_win)), res))) - set(
                        peak_pos))
                for k in range(len(tmp_interpol_pos)):
                    tmp_pos = np.concatenate((interpol_pos, [tmp_interpol_pos[k]]))
                    y_hat = si.interp1d(x_data[tmp_pos], y_data[tmp_pos])(x_data)
                    # generalize to any loss
                    # tmp_mse=np.var(y_hat-y_data)
                    tmp_mse = (1 - nu) * np.dot((y_hat - y_data > 0), np.abs((y_hat - y_data))) + nu * np.var(
                        y_hat - y_data)
                    # update the minimum
                    if tmp_mse < min_mse:
                        min_pos = tmp_interpol_pos[k]
                        min_mse = tmp_mse
                interpol_pos.append(min_pos)
                # interpol_pos.sort()
                interpol_mse.append(min_mse)
                unique_pos = np.array([int(interpol_pos.index(x)) for x in set(interpol_pos)])
                interpol_pos = list(np.array(interpol_pos)[unique_pos.astype(int)])
                interpol_mse = list(np.array(interpol_mse)[unique_pos.astype(int)])
                # sort points
                sort_pos = np.argsort(interpol_pos)
                interpol_pos = list(np.array(interpol_pos)[sort_pos.astype(int)])
                interpol_mse = list(np.array(interpol_mse)[sort_pos.astype(int)])
                # remove points that are too close
            y_hat = si.interp1d(x_data[interpol_pos], y_data[interpol_pos])(x_data)
            y_bsp = np.poly1d(np.polyfit(x_data[interpol_pos], y_data[interpol_pos], 3))(x_data)
            # y_bsp=si.interp1d(x_data[interpol_pos],y_data[interpol_pos], kind='cubic')(x_data)
            # --------------------------------------------------------------------
            #   find when you acre over the mean level
            # --------------------------------------------------------------------
            # th=2*np.sqrt(np.var(y_data-mean_level))
            th = np.sqrt(np.var(y_data - mean_level))
            pos = np.array(np.where((y_data - mean_level) < th)[0])
