import pandas as pd
import numpy as np
import scipy.signal as sg
import sys
import h5py

# import nfb lab data loader
sys.path.insert(0, '/home/kolai/Projects/nfblab/nfb')
from utils.load_results import load_data
from release.settings import CHANNELS as channels
channels += ['PHOTO']


FLANKER_WIDTH = 2
GFP_THRESHOLD = 100e-6

# band hilbert helper


# collect info
data_path = '/home/kolai/Data/alpha_delay2'
info = pd.read_csv('release/data/alpha_subject_2_full.csv')
datasets = [d for d in info['dataset'].unique() if (d is not np.nan)
            and (info.query('dataset=="{}"'.format(d))['type'].values[0] in ['FB0', 'FBMock', 'FB250', 'FB500', 'FBLow'])][:]



# store data
columns= ['subj_id', 'block_number'] + channels + ['online_envelope']

probes_df = pd.DataFrame(columns=columns, dtype='float32')
datasets_df = pd.DataFrame(columns=['dataset', 'subj_id', 'band', 'fb_type', 'snr'])

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
    # fb_type = df.query('block_number==6')['block_name'].values[0]
    fb_type = info.loc[info['dataset']==dataset, 'type'].values[0]

    # rename FB blocks to "FB"
    df['block_name'] = df['block_name'].apply(lambda x: 'FB' if 'FB' in x else x)

    # drop pauses
    df = df.loc[df['block_name'].isin(['Baseline0', 'Close', 'Baseline', 'FB'])]

    # remove eyes artifacts ICA
    df[channels] = df[channels].values.dot(eye_rejection_matrix)

    # GFP threshold arthifact segments
    th = np.abs(df[channels[:-1]]).rolling(int(fs), center=True).max().mean(1)
    df = df.loc[th < GFP_THRESHOLD]

    # down sample to 250
    ba = sg.butter(4, np.array([1, 45])/fs*2, 'band')
    df[channels[:-1]] = sg.filtfilt(*ba, df[channels[:-1]].values, axis=0) # except PHOTO
    # handle PHOTO
    y = sg.filtfilt(np.ones(20) / 20, [1], df['PHOTO'].values)
    y_down = pd.Series(y).rolling(1000, center=True).min().rolling(10000, center=True).mean().fillna(
        method='bfill').fillna(method='ffill').values
    df['PHOTO'] = y - y_down
    # handle envelope
    df['online_envelope'] = sg.filtfilt([0.5, 0.5], [1., 0.], df['online_envelope'].values, axis=0)
    df = df.iloc[::2]
    FS = 250

    # estimate snr
    freq, pxx = sg.welch(df.query('block_name=="Baseline0"')['P4'], FS, nperseg=FS * 2)
    sig = pxx[(freq >= band[0]) & (freq <= band[1])].mean()
    noise = pxx[((freq >= band[0] - FLANKER_WIDTH) & (freq <= band[0])) | (
            (freq >= band[1]) & (freq <= band[1] + FLANKER_WIDTH))].mean()
    snr = sig / noise

    # print info
    print('{:3.1f} {:40s} {:10s} {:.2f}'.format(subj_id/len(datasets)*100, dataset, fb_type, snr))

    # save data
    df['subj_id'] = subj_id
    probes_df = probes_df.append(df[columns].astype('float32'), ignore_index=True)
    datasets_df = datasets_df.append({'dataset': dataset, 'subj_id': subj_id, 'band':band, 'fb_type': fb_type, 'snr': snr},
                                     ignore_index=True)

probes_df[['subj_id', 'block_number']] = probes_df[['subj_id', 'block_number']].astype('int8')
probes_df = probes_df.loc[:,~probes_df.columns.duplicated()]
probes_df.to_pickle('release/data/FBLow_eeg_allsubjs_eyefree_1_45hz_down250hz.pkl')
datasets_df.to_pickle('release/data/info_allsubjs.pkl')