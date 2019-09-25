import pandas as pd
import numpy as np
from tqdm import tqdm
from proc.settings import FS, CHANNELS, FB_ALL, ICA_CHANNELS
from proc.settings import load_ica


N_SAMPLES_PRE = 3*FS
N_SAMPLES_POST = 2*FS
def band_hilbert(x, fs, band, N=None, axis=-1):
    x = np.asarray(x)
    Xf = np.fft.fft(x, N, axis=axis)
    w = np.fft.fftfreq(x.shape[0], d=1. / fs)
    Xf[(w < band[0]) | (w > band[1])] = 0
    x = np.fft.ifft(Xf, axis=axis)
    return 2*x

SPLIT_FB_BLOCKS = True
GLOBAL_THRESHOLD = False
USE_ICA = False
P4_ONLY = True
channels = (['P4'] if P4_ONLY else CHANNELS) + (ICA_CHANNELS if USE_ICA else [])
threshold_factors = [1/2]
bands =dict(zip(['alpha'], [1]))
res_df_name = 'channels{}_bands{}_splited{}_thresholds{}_globthr{}'.format(len(channels), len(bands), SPLIT_FB_BLOCKS, len(threshold_factors), GLOBAL_THRESHOLD)
print(res_df_name)

# load pre filtered data
probes_df = pd.read_pickle('data/eeg_allsubjs_eyefree_1_45hz_down250hz.pkl')

# load datasets info
datasets_df = pd.read_pickle('data/info_allsubjs.pkl')

# prepare data frame to save metrics
columns=['subj_id', 'fb_type', 'spindle', 'threshold_factor', 'time']
stats_df = pd.DataFrame(columns=columns)
for subj_id in datasets_df['subj_id'].values[:]:


    # subj eeg
    data = probes_df.query('subj_id=="{}" '.format(subj_id))

    # subj fb type
    fb_type = datasets_df.query('subj_id=={}'.format(subj_id))['fb_type'].values[0]

    # subj band
    band = np.array(datasets_df.query('subj_id=={}'.format(subj_id))['band'].values[0]) * 1

    # subj ica
    ica = load_ica('data/ica/s{}ica.pkl'.format(subj_id))

    # block numbers utils
    block_numbers = data['block_number'].values


    ch = 'P4'
    # channel data if channels is ICA get projection
    ch_data = data[ch].values

    # compute envelope
    envelope = np.abs(band_hilbert(ch_data, FS, band))

    norm = np.median(envelope[np.isin(block_numbers, FB_ALL[1:8])])
    envelope = envelope[np.isin(block_numbers, FB_ALL[1:])]
    signal = data['PHOTO'].values[np.isin(block_numbers, FB_ALL[1:])]
    median = np.median(signal)
    delay = {'FB0': 0, 'FB250': 0*250 * FS // 1000, 'FB500': 0*500 * FS // 1000, 'FBMock': 0*60 * FS}
    # signal = np.roll(signal, delay[fb_type])
    #envelope = signal
    for threshold_factor in threshold_factors:
        threshold = threshold_factor * median
        spindles_mask = signal < threshold
        spindle_starts = np.where(np.diff(spindles_mask.astype(int)) == 1)[0]
        spindles = []
        for spindle_start in spindle_starts:
            if spindle_start < N_SAMPLES_PRE or spindle_start >= len(signal) - N_SAMPLES_POST: continue
            spindle = envelope[spindle_start - N_SAMPLES_PRE: spindle_start + N_SAMPLES_POST]
            spindles.append(spindle)

        spindle = np.mean(spindles, 0)
        spindle = spindle-np.median(spindle[:FS])

        spindle = np.roll(spindle, delay[fb_type])

        # spindle /= spindle.max()
        stats_df = stats_df.append(pd.DataFrame(
            {'subj_id': subj_id, 'fb_type': fb_type, 'spindle': spindle,
             'threshold_factor': threshold_factor, 'time': np.arange(-N_SAMPLES_PRE, N_SAMPLES_POST) / FS}))


import seaborn as sns
g = sns.relplot('time', 'spindle', data=stats_df, hue='fb_type', kind='line', ci=95, estimator=np.mean, hue_order=['FB0', 'FB250', 'FB500', 'FBMock'], col= 'threshold_factor')

[ax.axvline(0, color='k') for ax in g.axes.flatten()]




import pylab as plt
delay_mesure_example_df = probes_df.query('subj_id==2 & block_number==22')[['PHOTO', 'P4']]
envelope = np.abs(band_hilbert(delay_mesure_example_df['P4'].values, FS, [8, 12]))
nor = lambda x: (x-x.mean())/x.std()
plt.plot(nor(envelope))
plt.plot(nor(delay_mesure_example_df['PHOTO'].values))