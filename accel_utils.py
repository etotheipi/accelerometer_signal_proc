import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import scipy.signal

class AccelUtils:
    @staticmethod
    def read_params_file(fn):
        with open(fn, 'r') as f:
            params = json.load(f)

        # Some values that will remain constant throughout experiments
        csv_list = [f for f in os.listdir(params['data_dir']) if os.path.splitext(f)[1] == '.csv']
        num_subj = len(csv_list)
        params['num_subj'] = num_subj

        AccelUtils.recompute_dervied_params(params)

        print('Params:')
        print(json.dumps(params, indent=2))

        print(f'Number of files/subjects: {num_subj}')
        print(f'Each sample will be {params["max_resample_pts"]} time steps')
        return params

    @staticmethod
    def recompute_dervied_params(params):
        params['max_resample_pts'] = int(params['window_size_sec'] // params['resample_dt']) - 4
        params['min_timesteps_per_sample'] = int(params['max_resample_pts'] *
                                                 params['filt_min_timesteps_per_sample_ratio'])

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
        min_timesteps,
        avg_power_thresh=2.0,
        dc_ratio_thresh=0.3,
        power_ratio_thresh=0.5):

        # Looks like there's some gaps that leave use with windows containing no data
        if len(t_list.values) < 10:
            return False

        # Quick return if any abnormal time intervals
        if len(t_list.values) < min_timesteps:
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
    def extract_valid_time_windows_for_subj(subj_df, window_size, min_timesteps, **kwargs):
        curr_tstart = 0
        valid_windows = []
        subj_df.sort_values(['t'], inplace=True)
        while curr_tstart < subj_df['t'].max() - window_size:
            window2s = subj_df[(subj_df['t'] >= curr_tstart) & (subj_df['t'] < curr_tstart + window_size)]
            if AccelUtils.is_valid_time_window(window2s['t'], window2s['mag_acc'], min_timesteps, **kwargs):
                valid_windows.append(window2s.copy())
            curr_tstart += window_size

        return valid_windows
    
    @staticmethod
    def resample_and_truncate(window_df, dt_resample, max_pts, col='mag_acc'):
        # Need to truncate to a 
        ts = window_df['t']
        vals = window_df[col]

        sample_pts = np.arange(ts.min(), ts.max(), step=dt_resample)
        new_sample = np.interp(sample_pts, ts, vals)

        return sample_pts[:max_pts], new_sample[:max_pts]

    @staticmethod
    def compute_periodogram_with_peaks(window_df, dt_resample, max_pts, max_peaks):
        """
        Finds the three most dominant frequences in the sample, returns them sorted by freq
        """
        _, mags = AccelUtils.resample_and_truncate(window_df, dt_resample, max_pts)
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

    
    @staticmethod
    def compute_periodogram(window_df, dt_resample, max_pts, pow_exponent=0.5):
        _, mags = AccelUtils.resample_and_truncate(window_df, dt_resample, max_pts)
        fs, fpow = scipy.signal.periodogram(mags, 1/dt_resample)
        fpow = fpow ** pow_exponent
        fpow_norm = fpow / np.sum(fpow)
        return fs, fpow_norm
    
    @staticmethod
    def display_time_slices_for_subject(per_user_dataframes, i_subj, window_sz, max_dt):
        df_subj = per_user_dataframes[i_subj]
        valid_windows = AccelUtils.extract_valid_time_windows_for_subj(df_subj, window_sz, max_dt)
    
        fig,axs = plt.subplots(2, 1, figsize=(12, 5))
        axs[0].plot(df_subj['t'], df_subj['mag_acc'])
        axs[0].set_xlim(0, 200)
        axs[0].set_ylim(-1, 21)
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Accel Magnitude')
        axs[0].set_title(f'All Data for Subject {i_subj+1}')

        for win in valid_windows:
            axs[1].plot(win['t'], win['mag_acc'])
            axs[1].set_xlim(0, 200)
            axs[1].set_ylim(-1, 21)
            axs[1].set_xlabel('Time (s)')
            axs[1].set_ylabel('Accel Magnitude')
        axs[1].set_title('Valid Sample Windows')
        fig.tight_layout(pad=0.5)
        print('Show which time-windows are considered valid by the specified criteria')
        fig.savefig(f'example_valid_windows_subj_{i_subj}.png')
        

class AccelUtils3D:
    ## 3D Utils
    @staticmethod
    def rotation_matrix_a_onto_b(a, b):
        # Formula from https://math.stackexchange.com/a/476310
        v = np.cross(a, b)
        s = np.linalg.norm(v)
        c = np.dot(a, b)

        vx = np.array([
            [    0, -v[2],  v[1]],
            [ v[2],     0, -v[0]],
            [-v[1],  v[0],     0]
        ])

        rotation_matrix = np.eye(3) + vx + np.matmul(vx, vx) * (1 / (1 + c))
        return rotation_matrix

    @staticmethod
    def estimate_gravity_and_rotation_matrix(df_subj):
        grav_est = np.array([df_subj[col].mean() for col in ['x_acc', 'y_acc', 'z_acc']])

        grav_est = grav_est / np.linalg.norm(grav_est)
        grav_targ = np.array([0, 0, -1])

        rot_matrix = AccelUtils3D.rotation_matrix_a_onto_b(grav_est, grav_targ)
        return rot_matrix, grav_est

    @staticmethod
    def align_window_acc_to_gravity(subj_window_df, with_norm=True):
        rot_mtrx, _ = AccelUtils3D.estimate_gravity_and_rotation_matrix(subj_window_df)
        xyz_vals = subj_window_df[['x_acc', 'y_acc', 'z_acc']].values
        rotated_vals = rot_mtrx.dot(xyz_vals.T).T
        if with_norm:
            norm_vals = np.linalg.norm(rotated_vals, axis=1)
            rotated_vals = rotated_vals / np.expand_dims(norm_vals, axis=1)

        subj_window_df['x_rot'] = rotated_vals[:, 0]
        subj_window_df['y_rot'] = rotated_vals[:, 1]
        subj_window_df['z_rot'] = rotated_vals[:, 2]

    @staticmethod
    def compute_first_order_longitudes(subj_window_df):
        # Project into xy plane, calc angle between (x1, y1) and (x2, y2)
        xy_pairs = subj_window_df[['x_rot', 'y_rot']].values
        xy_norms = np.linalg.norm(xy_pairs, axis=1)
        xy_units = xy_pairs / np.expand_dims(xy_norms, axis=1) * 0.999
        out = []
        for r in range(xy_pairs.shape[0] - 1):
            x1, y1 = xy_units[r+0, :]
            x2, y2 = xy_units[r+1, :]
            out.append(np.arcsin(x1 * y2 - x2 * y1))

        return np.array(out)

    @staticmethod
    def compute_first_order_latitudes(subj_window_df):
        # Assume unit vector inputs
        lat_angles = np.arcsin(subj_window_df['z_rot'].values * 0.999)
        return lat_angles[1:] - lat_angles[:-1]
    
    @staticmethod
    def angle_periodogram(angle_diffs, dt_resample, pow_exponent=0.5):
        fs, fpow = scipy.signal.periodogram(angle_diffs, 1/dt_resample)
        fpow = fpow ** pow_exponent
        fpow_norm = fpow / np.sum(fpow)
        return fs, fpow_norm
    
    