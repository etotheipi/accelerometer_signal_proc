import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import scipy.signal

class AccelUtils:
    
    @staticmethod
    def read_all_user_data(data_dir, num_subj):
        per_user_all_data = {}
        for i in range(num_subj):
            fpath = os.path.join(data_dir, f'{i+1}.csv')
            per_user_all_data[i] = pd.read_csv(fpath, names=['t', 'x_acc', 'y_acc', 'z_acc'])

            # For sure, we want magnitude of the acceleration, and it's good for initial plots
            acc_mag = lambda row: np.sqrt(row['x_acc']**2 + row['y_acc']**2 + row['z_acc']**2)
            per_user_all_data[i]['mag_acc'] = per_user_all_data[i].apply(acc_mag, axis=1)   
            
        return per_user_all_data
    
    @staticmethod
    def is_valid_time_window(
        t_list,
        mag_list,
        max_dt,
        avg_power_thresh=2.0,
        dc_ratio_thresh=0.3,
        power_ratio_thresh=0.5):

        # Looks like there's some gaps that leave use with windows containing no data
        if len(t_list.values) < 10:
            return False

        # Quick return if any abnormal time intervals
        if any(t_list[1:] - t_list[:-1] > max_dt):
            return False

        # Compute average magnitude/DC of the signal
        dc = np.mean(mag_list)

        # Quick return of total average power is too low
        avg_power = np.mean((mag_list - dc)**2)
        if avg_power < avg_power_thresh:
            return False

        sub_sz = len(t_list) // 3
        sub_windows = [
            mag_list[0*sub_sz:1*sub_sz],
            mag_list[1*sub_sz:2*sub_sz],
            mag_list[2*sub_sz:]
        ]

        sub_dcs = [np.mean(th) for th in sub_windows]
        sub_pows  = [np.mean((th - dc)**2) for th in sub_windows]

        for i in range(2):
            # Check that it is approximately stationary
            if abs(sub_dcs[i] - sub_dcs[i+1]) / sub_dcs[i+1] > dc_ratio_thresh:
                return False

            # Check that each sub-window has similar power
            if abs(sub_pows[i] - sub_pows[i+1]) / sub_pows[i+1] > power_ratio_thresh:
                return False

        return True
    
    @staticmethod
    def extract_valid_time_windows_for_subj(df, window_size, max_dt, **kwargs):
        curr_tstart = 0
        valid_windows = []
        while curr_tstart < df['t'].max() - window_size:
            window2s = df[(df['t'] >= curr_tstart) & (df['t'] < curr_tstart + window_size)]
            if AccelUtils.is_valid_time_window(window2s['t'], window2s['mag_acc'], max_dt, **kwargs):
                valid_windows.append(window2s.copy())
            curr_tstart += window_size

        return valid_windows
    
    @staticmethod
    def resample_and_truncate(window_df, dt_resample, max_pts):
        # Need to truncate to a 
        ts = window_df['t']
        mags = window_df['mag_acc']

        sample_pts = np.arange(ts.min(), ts.max(), step=dt_resample)
        new_sample = np.interp(sample_pts, ts, mags)

        return sample_pts[:max_pts], new_sample[:max_pts]

    @staticmethod
    def compute_periodogram_with_peaks(window_df, dt_resample, max_pts, max_peaks):
        """
        Finds the three most dominant frequences in the sample, returns them sorted by freq
        """
        ts, mags = AccelUtils.resample_and_truncate(window_df, dt_resample, max_pts)
        fs, pxx = scipy.signal.periodogram(mags, 1/dt_resample)
        pxx = np.sqrt(pxx)
        i_peaks, _ = scipy.signal.find_peaks(pxx)
        sorted_peaks = sorted([(pxx[i], fs[i], i) for i in i_peaks], reverse=True)
        sorted_freqs = sorted([(trip[1], trip[2]) for trip in sorted_peaks[:max_peaks]])
        top_n_indices = sorted([pair[1] for pair in sorted_freqs])
        top_freqs = [fs[i] for i in top_n_indices]
        top_pows = [pxx[i] for i in top_n_indices]
        top_pows = np.array(top_pows) / sum(top_pows)
        return fs, pxx, top_freqs, top_pows

    
    