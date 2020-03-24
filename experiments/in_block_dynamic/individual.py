import pandas as pd
import numpy as np
import scipy.signal as sg
import pylab as plt
from scipy.linalg import eigh
from mne.decoding import SPoC
from mne.cov import _regularized_covariance
from  mne.viz import plot_topomap
from scipy import linalg

from release.settings import CHANNELS, FS, FB_ALL, MONTAGE,  Montage
from pynfb.signal_processing.decompositions import ICADecomposition


eeg_path = 'release/data/FBLow_eeg_allsubjs_eyefree_1_45hz_down250hz.pkl'
info_path = 'release/data/info_allsubjs.pkl'

eeg_df = pd.read_pickle(eeg_path)
info_df = pd.read_pickle(info_path)

n_times = FS * 10
n_step = FS * 5
n_epochs = (FS * 120 - n_times) // n_step
n_blocks = len(FB_ALL)
stats_df = pd.DataFrame(columns=['subj_id', 'fb_type', 'time', 'var'])
for subj_id in range(50):
    fb_type = info_df[info_df['subj_id']==subj_id]['fb_type'].values[0]

    eeg_df_subj = eeg_df[eeg_df['subj_id'] == subj_id]
    band = info_df.loc[info_df['subj_id'] == subj_id, 'band'].values[0]
    filter_n_taps = 1000
    filter_all = sg.firwin2(filter_n_taps, [0, band[0]-2, band[0]-2, band[1]+2, band[1]+2, FS // 2], [0, 0, 1, 1, 0, 0], fs=FS)
    filter_band = sg.firwin2(filter_n_taps, [0, band[0], band[0], band[1], band[1], FS // 2], [0, 0, 1, 1, 0, 0], fs=FS)

    x = eeg_df_subj.loc[eeg_df_subj['block_number'].isin(FB_ALL), 'P4'].values*1e6
    x_band = sg.filtfilt(filter_band, [1, 0], x, 0)
    block_numbers = eeg_df_subj.loc[eeg_df_subj['block_number'].isin(FB_ALL), 'block_number'].values


    X_band = np.zeros((n_blocks, n_epochs))

    for j_block_number, block_number in enumerate(FB_ALL):
        x_block = x_band[block_numbers==block_number]
        for k in range(0, n_epochs):
            X_band[j_block_number, k] = x_block[k * n_step: k * n_step + n_times].var()


    vars = X_band
    vars /= vars[:, :n_epochs//2].mean(1)[:, None]
    stats_df = stats_df.append(pd.DataFrame({'subj_id': subj_id, 'fb_type': fb_type, 'time': np.arange(n_epochs)*n_step/FS,
                                             'var': vars.mean(0)}), ignore_index=True)
    print(stats_df.tail())

import seaborn as sns
sns.relplot('time', 'var', 'fb_type', data=stats_df, kind='line')