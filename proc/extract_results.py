import pandas as pd
import numpy as np
from tqdm import tqdm
from proc.settings import FS, CHANNELS, FB_ALL, ICA_CHANNELS
from proc.settings import load_ica


def band_hilbert(x, fs, band, N=None, axis=-1):
    x = np.asarray(x)
    Xf = np.fft.fft(x, N, axis=axis)
    w = np.fft.fftfreq(x.shape[0], d=1. / fs)
    Xf[(w < band[0]) | (w > band[1])] = 0
    x = np.fft.ifft(Xf, axis=axis)
    return 2*x

SPLIT_FB_BLOCKS = True
GLOBAL_THRESHOLD = False
threshold_factors = [2] #np.arange(1, 3.1, 0.125)
res_df_name = 'multichannel_metrics_{split}_{threshold_type}'.format(threshold_type='global' if GLOBAL_THRESHOLD else 'local', split='split' if SPLIT_FB_BLOCKS else '')

# load pre filtered data
probes_df = pd.read_pickle('data/eeg_allsubjs_eyefree_1_45hz_down250hz.pkl')

# load datasets info
datasets_df = pd.read_pickle('data/info_allsubjs.pkl')

# prepare data frame to save metrics
columns=['subj_id', 'channel', 'fb_type', 'metric', 'metric_type', 'block_number','threshold_factor']

for subj_id in datasets_df['subj_id'].values[:]:
    stats_df = pd.DataFrame(columns=columns)

    # subj eeg
    data = probes_df.query('subj_id=="{}" '.format(subj_id))

    # subj fb type
    fb_type = datasets_df.query('subj_id=={}'.format(subj_id))['fb_type'].values[0]

    # subj band
    band = datasets_df.query('subj_id=={}'.format(subj_id))['band'].values[0]

    # subj ica
    ica = load_ica('data/ica/s{}ica.pkl'.format(subj_id))

    # block numbers utils
    block_numbers = data['block_number'].values
    unique_block_numbers = np.unique(block_numbers)
    if SPLIT_FB_BLOCKS:
        split_block_numbers = []
        for block_number in unique_block_numbers:
            if block_number in FB_ALL:
                split_block_numbers += [block_number*1000 + 1, block_number*1000 + 2]
            else:
                split_block_numbers += [block_number]
        unique_block_numbers = split_block_numbers

    if GLOBAL_THRESHOLD:
        envs = []
        for ch in tqdm(CHANNELS+ICA_CHANNELS, str(subj_id)):
            if 'ICA' in ch: continue
            # channel data if channels is ICA get projection
            ch_data = data[ch].values

            # compute envelope
            env = np.abs(band_hilbert(ch_data, FS, band))

            # first FB median
            envs.append(env)
        median = np.median(np.concatenate(envs))

    for ch in tqdm(CHANNELS+ICA_CHANNELS, str(subj_id)):
        # channel data if channels is ICA get projection
        ch_data = data[ch].values if 'ICA' not in ch else data[CHANNELS].values.dot(ica.filters[:, int(ch[3:])-1])

        # compute envelope
        env = np.abs(band_hilbert(ch_data, FS, band))

        if not GLOBAL_THRESHOLD:
            # first FB median
            median = np.median(env[block_numbers == FB_ALL])

        for block_number in unique_block_numbers:
            # get block envelope as signal
            if block_number < 1000:
                signal = env[block_numbers == block_number]
            else: # only if any block split
                signal = env[block_numbers == block_number // 1000]
                if block_number % 1000 == 1:
                    signal = signal[:len(signal) // 2]
                elif block_number % 1000 == 2:
                    signal = signal[len(signal) // 2:]

            # mean magnitude in uV
            magnitude_j = np.median(signal) * 1e6

            # iterate thresholds factors
            for threshold_factor in threshold_factors:
                threshold = threshold_factor * median

                # get spindles mask
                spindles_mask = signal > threshold

                # number of spindles
                n_spindles_j = np.sum(np.diff(spindles_mask.astype(int)) == 1)

                # mean spindle duration
                duration_j = np.sum(spindles_mask) / n_spindles_j / FS

                # mean spindle amplitue
                amplitude_j = np.median(signal[spindles_mask]) * 1e6

                # save metrics above for channel
                stats_df = stats_df.append(pd.DataFrame(
                    {'subj_id': subj_id, 'channel': ch, 'fb_type': fb_type,
                     'metric': [magnitude_j, n_spindles_j, duration_j, amplitude_j],
                     'metric_type': ['magnitude', 'n_spindles', 'duration', 'amplitude'],
                     'block_number': block_number, 'threshold_factor': threshold_factor}), ignore_index=True)

    print(stats_df.memory_usage().sum()/1024/1024)
    stats_df.to_pickle('data/{}_{}.pkl'.format(res_df_name, subj_id))

stats_df = pd.DataFrame(columns=columns)
for subj_id in datasets_df['subj_id'].values[:]:
    stats_df = stats_df.append(pd.read_pickle('data/{}_{}.pkl'.format(res_df_name, subj_id)))
stats_df.to_pickle('data/{}.pkl'.format(res_df_name))