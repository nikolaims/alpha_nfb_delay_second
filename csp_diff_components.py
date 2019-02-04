import pandas as pd
import numpy as np
import scipy.signal as sg
import sys
import pylab as plt
import h5py
from mne.viz import plot_topomap

# import nfb lab data loader
sys.path.insert(0, '/home/kolai/Projects/nfblab/nfb')
from utils.load_results import load_data
from pynfb.inlets.montage import Montage
from pynfb.signal_processing.decompositions import CSPDecomposition


PLOT_ARTIFACTS_RES = False
FLANKER_WIDTH = 2
FS = 500
GFP_THRESHOLD = 100e-6

# band hilbert helper
def band_hilbert(x, fs, band, N=None, axis=-1):
    x = np.asarray(x)
    Xf = np.fft.fft(x, N, axis=axis)
    w = np.fft.fftfreq(x.shape[0], d=1. / fs)
    Xf[(w < band[0]) | (w > band[1])] = 0
    x = np.fft.ifft(Xf, axis=axis)
    return 2*x


# collect info
data_path = '/home/kolai/Data/alpha_delay2'
info = pd.read_csv('alpha_subject_2.csv')
datasets = [d for d in info['dataset'].unique() if (d is not np.nan)
            and (info.query('dataset=="{}"'.format(d))['type'].values[0] in ['FB0', 'FBMock', 'FB250', 'FB500'])][:]


# store data
subj_bands = {}
fig, axes = plt.subplots(len(datasets), 32)
for j_dataset, dataset in enumerate(datasets):
    dataset_path = '{}/{}/experiment_data.h5'.format(data_path, dataset)

    # load fb signal params
    with h5py.File(dataset_path) as f:
        eye_rejection_matrix = f['protocol10/signals_stats/Alpha0/rejections/rejection1'].value
        subj_bands[dataset] = f['protocol10/signals_stats/Alpha0/bandpass'].value

    # load data
    df, fs, channels, p_names = load_data(dataset_path)

    # drop pauses
    df = df.loc[df['block_name'].isin(['Baseline0', 'Close', 'Baseline', 'FB0', 'FB250', 'FB500', 'FBMock'])]

    # get FB type
    fb_type = df.query('block_number==6')['block_name'].values[0]

    # rename FB blocks to "FB"
    df['block_name'] = df['block_name'].apply(lambda x: 'FB' if 'FB' in x else x)

    # remove eyes artifacts ICA
    df[channels] = df[channels].values.dot(eye_rejection_matrix)

    # GFP threshold arthifact segments
    th = np.abs(df[channels[:-1]]).rolling(int(fs), center=True).max().mean(1)
    df = df.loc[th<GFP_THRESHOLD]

    # estimate snr
    freq, pxx = sg.welch(df.query('block_name=="Baseline0"')['P4'], FS, nperseg=FS * 2)
    band = subj_bands[dataset]
    sig = pxx[(freq >= band[0]) & (freq <= band[1])].mean()
    noise = pxx[((freq >= band[0] - FLANKER_WIDTH) & (freq <= band[0])) | (
                (freq >= band[1]) & (freq <= band[1] + FLANKER_WIDTH))].mean()
    snr = sig / noise

    # exclude subjects with snr < 1
    if snr<1: continue
    montage = Montage(channels[:-1])
    csp = CSPDecomposition(montage.get_names(), FS, subj_bands[dataset].tolist())
    b_numbers = df.query('block_name=="FB"')['block_number'].unique()
    x = np.concatenate([df.query('block_number=={}'.format(b))[channels[:-1]].values for b in b_numbers[1:]])
    y = np.concatenate([np.zeros(sum(df['block_number']==b)) + int(b<b_numbers[8]) for b in b_numbers[1:]])

    csp.fit(x, y)

    #plt.figure()
    plt.plot(x.dot(csp.filters[:, 1]))
    plt.plot(df['P4'])
    [plot_topomap(csp.topographies[:, j], montage.get_pos(), axes=axes[j_dataset, j], show=False, contours=0, mask=np.array(channels[:-1])=='P4') for j in range(32)]
    [axes[j_dataset, j].set_title('{:.1f}'.format(csp.scores[j])) for j in range(32)]
    axes[j_dataset, 0].set_ylabel('{}\n{:.1f}'.format(fb_type, snr))


