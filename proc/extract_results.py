import pandas as pd
import numpy as np
import seaborn as sns
import pylab as plt
from scipy import stats
from tqdm import tqdm
from time import time


from proc.settings import FS, CHANNELS, FB_ALL

def band_hilbert(x, fs, band, N=None, axis=-1):
    x = np.asarray(x)
    Xf = np.fft.fft(x, N, axis=axis)
    w = np.fft.fftfreq(x.shape[0], d=1. / fs)
    Xf[(w < band[0]) | (w > band[1])] = 0
    x = np.fft.ifft(Xf, axis=axis)
    return 2*x


threshold_factors = np.arange(1, 3.1, 0.125)

probes_df = pd.read_pickle('data/eeg_allsubjs_eyefree_1_45hz_down250hz.pkl')
datasets_df = pd.read_pickle('data/info_allsubjs.pkl')

medians = []
stats_df = pd.DataFrame(columns=['subj_id', 'channel', 'fb_type', 'metric', 'metric_type', 'block_number', 'threshold_factor'])
for subj_id in datasets_df['subj_id'].values[:]:
    print(subj_id)
    data = probes_df.query('subj_id=="{}" '.format(subj_id))
    fb_type = datasets_df.query('subj_id=={}'.format(subj_id))['fb_type'].values[0]
    band = datasets_df.query('subj_id=={}'.format(subj_id))['band'].values[0]
    block_numbers = data['block_number'].values
    unique_block_numbers = np.unique(block_numbers)
    for ch in tqdm(CHANNELS, str(subj_id)):
        env = np.abs(band_hilbert(data[ch], FS, band))
        median = np.median(env[np.isin(block_numbers, FB_ALL[:1])])
        for block_number in unique_block_numbers:
            signal = env[block_numbers == block_number]
            magnitude_j = signal.mean() * 1e6
            for threshold_factor in threshold_factors:
                threshold = threshold_factor * median
                spindles_mask = signal > threshold
                n_spindles_j = np.sum(np.diff(spindles_mask.astype(int)) == 1)
                duration_j = np.sum(spindles_mask) / n_spindles_j / FS
                amplitude_j = signal[spindles_mask].mean() * 1e6
                stats_df = stats_df.append(pd.DataFrame(
                    {'subj_id': subj_id, 'channel': ch, 'fb_type': fb_type,
                     'metric': [magnitude_j, n_spindles_j, duration_j, amplitude_j],
                     'metric_type': ['magnitude', 'n_spindles', 'duration', 'amplitude'],
                     'block_number': block_number, 'threshold_factor': threshold_factor}))

