import pandas as pd
import numpy as np
from tqdm import tqdm
from release.settings import FS, CHANNELS, FB_ALL, ICA_CHANNELS
from release.utils import band_hilbert


SPLIT_FB_BLOCKS = True
P4_ONLY = True
USE_PERCENTILES = False
channels = (['P4'] if P4_ONLY else CHANNELS)
if USE_PERCENTILES:
    threshold_factors = np.arange(50, 100, 2.5)
else:
    threshold_factors = np.arange(1, 3.5, 0.125)
#
bands = dict(zip(['alpha'], [1]))
res_df_name = '5groups_channels{}_bands{}_splited{}_{}_threshs{}'.format(len(channels), len(bands), SPLIT_FB_BLOCKS,
                                                                    'perc' if USE_PERCENTILES else 'median',
                                                                    len(threshold_factors))
print(res_df_name)

# load pre filtered data
probes_df = pd.read_pickle('release/data/eeg_5groups_eyefree_1_45hz_down250hz.pkl')

# load datasets info
datasets_df = pd.read_pickle('release/data/info_5groups.pkl')

# prepare data frame to save metrics
columns=['subj_id', 'channel', 'fb_type', 'metric', 'metric_type', 'block_number','threshold_factor', 'band']

for subj_id in datasets_df['subj_id'].values[:]:
    stats_df = pd.DataFrame(columns=columns)

    # subj eeg
    data = probes_df.query('subj_id=="{}" '.format(subj_id))

    # subj fb type
    fb_type = datasets_df.query('subj_id=={}'.format(subj_id))['fb_type'].values[0]

    # subj band
    for band_name, band_factor in bands.items():
        band = np.array(datasets_df.query('subj_id=={}'.format(subj_id))['band'].values[0]) * band_factor

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
        # print(unique_block_numbers)

        for ch in tqdm(channels, str(subj_id) + band_name):
            # channel data if channels is ICA get projection
            ch_data = data[ch].values

            # compute envelope
            env = np.abs(band_hilbert(ch_data, FS, band))
            median = np.median(env[np.isin(block_numbers, FB_ALL)])

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
                # print(block_number, len(signal), sep='\t')
                # mean magnitude in uV
                magnitude_j = np.mean(signal) * 1e6

                # iterate thresholds factors
                for threshold_factor in threshold_factors:
                    if USE_PERCENTILES:
                        threshold = np.percentile(env[np.isin(block_numbers, FB_ALL)], threshold_factor)
                    else:
                        threshold = threshold_factor * median
                    #

                    # get spindles mask
                    spindles_mask = signal > threshold
                    if np.sum(np.diff(spindles_mask.astype(int)) == 1) > 0:

                        # number of spindles
                        n_spindles_j = np.sum(np.diff(spindles_mask.astype(int)) == 1)

                        # mean spindle duration
                        duration_j = np.sum(spindles_mask) / n_spindles_j / FS

                        # mean spindle amplitue
                        amplitude_j = np.mean(signal[spindles_mask]) * 1e6
                    else:
                        n_spindles_j = 1
                        duration_j = 0.005
                        amplitude_j = threshold * 1e6

                    # save metrics above for channel
                    stats_df = stats_df.append(pd.DataFrame(
                        {'subj_id': subj_id, 'channel': ch, 'fb_type': fb_type,
                         'metric': [magnitude_j, n_spindles_j, duration_j, amplitude_j],
                         'metric_type': ['magnitude', 'n_spindles', 'duration', 'amplitude'],
                         'block_number': block_number, 'threshold_factor': threshold_factor, 'band': band_name}),
                        ignore_index=True)

    stats_df.to_pickle('release/data/temp/{}_{}.pkl'.format(res_df_name, subj_id))

stats_df = pd.DataFrame(columns=columns)
for subj_id in datasets_df['subj_id'].values[:]:
    stats_df = stats_df.append(pd.read_pickle('release/data/temp/{}_{}.pkl'.format(res_df_name, subj_id)))
stats_df.to_pickle('release/data/{}.pkl'.format(res_df_name))