import pandas as pd
import numpy as np
import scipy.signal as sg
import sys
import h5py

# import nfb lab data loader
sys.path.insert(0, '/home/kolai/Projects/nfblab/nfb')
from utils.load_results import load_data
from release.settings import CHANNELS as channels
from release.utils import band_hilbert
from pynfb.signal_processing.filters import CFIRBandEnvelopeDetector, IdentityFilter
channels += ['PHOTO']

def opt_delay_corr(env, online_env):
    corrs = [np.corrcoef(np.roll(env, k), online_env)[1, 0] for k in range(100)]
    opt = np.argmax(corrs)
    return opt/FS*1000, corrs[opt]



FLANKER_WIDTH = 2
GFP_THRESHOLD = 100e-6
FS = 500

# band hilbert helper


# collect info
data_path = '/home/kolai/Data/alpha_delay2'
info = pd.read_csv('release/data/alpha_subject_2_full.csv')
datasets = [d for d in info['dataset'].unique() if (d is not np.nan)
            and (info.query('dataset=="{}"'.format(d))['type'].values[0] in ['FB0', 'FBMock', 'FB250', 'FB500'])][:]



# store data
columns= ['subj_id', 'block_number'] + channels + ['online_envelope']

probes_df = pd.DataFrame(columns=columns, dtype='float32')
datasets_df = pd.DataFrame(columns=['dataset', 'subj_id', 'band', 'fb_type', 'snr'])



delays = []
corrds = []

for subj_id, dataset in enumerate(datasets[:]):
    dataset_path = '{}/{}/experiment_data.h5'.format(data_path, dataset)

    # load fb signal params
    with h5py.File(dataset_path) as f:
        eye_rejection_matrix = f['protocol10/signals_stats/Alpha0/rejections/rejection1'].value
        band = f['protocol10/signals_stats/Alpha0/bandpass'].value

    # load data
    df, fs, channels, p_names = load_data(dataset_path)
    df['online_envelope'] = df['signal_Alpha0']

    # get FB type
    fb_type = df.query('block_number==6')['block_name'].values[0]

    # rename FB blocks to "FB"
    df['block_name'] = df['block_name'].apply(lambda x: 'FB' if x == fb_type in x else x)

    # select FB blocks
    df = df.loc[df['block_name'].isin(['Baseline'])]

    # remove eyes artifacts ICA
    df[channels] = df[channels].values.dot(eye_rejection_matrix)


    raw_p4 = df['P4'].values
    ba = sg.butter(1, np.array([1, 50])/fs*2, 'band')

    env = np.abs(band_hilbert(sg.filtfilt(*ba, raw_p4), FS, band))
    online_env = CFIRBandEnvelopeDetector(band, FS, IdentityFilter(), n_taps=FS, delay_ms=100).apply(raw_p4)

    delay, corr = opt_delay_corr(env, online_env)
    delays.append(delay)
    corrds.append(corr)
    print(subj_id)



import pylab as plt
plt.scatter(corrds, delays, alpha=0.7, s=50)
# plt.scatter(corrds2, delays2, alpha=0.7, s=50)
plt.plot()





import os
data_path = '/home/kolai/Data/delay_tests/prefilters_neorec_and_delay_param/delay_test_kolai'
import seaborn as sns


labels = []
for exp in [r'/home/kolai/Data/alpha_delay2/alpha2-delay-subj-42_02-21_14-06-35']:
    exp_label = ' '.join(exp.split('/')[-1].split('_')[1:3])+'Hz'
    labels.append(exp_label)
    df, fs, channels, p_names = load_data(exp+'/experiment_data.h5')
    fs = int(fs)
    df = df.query('block_name == "FB0"')

    with h5py.File(exp+'/experiment_data.h5') as f:
        eye_rejection_matrix = f['protocol10/signals_stats/Alpha0/rejections/rejection1'].value
        band = f['protocol10/signals_stats/Alpha0/bandpass'].value

    raw_p4 = df['P4'].values
    ba = sg.butter(1, np.array([1, 45])/fs*2, 'band')
    raw_p4 = sg.filtfilt(*ba, raw_p4)

    env = np.abs(band_hilbert(raw_p4, FS, band))
    online_env = df['PHOTO'].values

    step = fs*10

    delays = []
    corrds = []
    for k in range(0, len(env)*2, step//2):
        delay, corr = opt_delay_corr(env[k//2:k//2+step], online_env[k//2:k//2+step])
        delays.append(delay)
        corrds.append(corr)
    # plt.scatter(corrds, delays, label=exp_label)
    # sns.kdeplot(corrds, delays, label=exp_label)

    print(exp_label, corrds)
    # plt.errorbar([np.nanmedian(corrds)], [np.nanmedian(delays)], [np.nanstd(delays)], [np.nanstd(corrds)], label=exp_label)
    sns.kdeplot(corrds, delays, shade=True, cbar=True)

plt.legend()
plt.xlabel('corr(online vs offline)')
plt.ylabel('effective delay')


plt.plot(np.arange(len(env))/fs/60, pd.Series(env).rolling(fs*60, center=True).median())

[plt.axvline(k*2, color='k') for k in range(15)]

plt.plot(np.arange(len(env))/fs/60, df['P4'])