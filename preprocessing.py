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

# plot art. figures
if PLOT_ARTIFACTS_RES:
    _topo_fig, topo_axes = plt.subplots(1, len(datasets))
    ts_fig, ts_axes = plt.subplots(len(datasets) // 2 + 1, 2, sharex=True, sharey=True)
    spec_fig, spec_axes = plt.subplots(len(datasets) // 2 + 1, 2, sharex=True, sharey=False)

# store data
subj_bands = {}
columns=['online_envelope', 'dataset', 'fb_type', 'block_name', 'block_number', 'P4', 'envelope', 'snr']
probes_df = pd.DataFrame(columns=columns)
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
    if PLOT_ARTIFACTS_RES: var_before = df[channels[:-1]].values.copy()
    df[channels] = df[channels].values.dot(eye_rejection_matrix)
    if PLOT_ARTIFACTS_RES: var_after = df[channels[:-1]].values.copy()

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

    # extract offline signal
    env = np.abs(band_hilbert(df['P4'].values, fs, subj_bands[dataset]))

    # print info
    print('{:40s} {:10s} {:.2f}'.format(dataset, fb_type, snr))

    # save data
    probes_df = probes_df.append(pd.DataFrame({'online_envelope': df['signal_Alpha0'].values, 'dataset': dataset,
                                               'fb_type': fb_type, 'block_name': df['block_name'].values,
                                               'block_number': df['block_number'].values, 'P4': df['P4'].values,
                                               'envelope': env, 'snr': snr}), ignore_index=True)

    if PLOT_ARTIFACTS_RES:
        ax = ts_axes[j_dataset//2, j_dataset%2]
        ax.plot(df['P4'].loc[:fs*100])
        ax.plot(th.loc[th<GFP_THRESHOLD].loc[:fs*100])
        ax.axhline(GFP_THRESHOLD, color='k', alpha=0.4)
        ax.axhline(-GFP_THRESHOLD, color='k', alpha=0.4)
        ax.legend(['{:40s} {:10s}'.format(dataset, fb_type)])
        ax.set_ylabel('{:.2f}'.format(snr))
        topo = (var_before-var_after)[th.values<GFP_THRESHOLD].std(0)
        spec_axes[j_dataset // 2, j_dataset % 2].plot(freq, pxx)
        spec_axes[j_dataset // 2, j_dataset % 2].legend(['{:40s} {:10s}'.format(dataset, fb_type)])
        spec_axes[j_dataset // 2, j_dataset % 2].set_ylabel('{:.2f}'.format(snr))
        plot_topomap(topo, Montage(channels[:-1]).get_pos(), axes=topo_axes[j_dataset], show=False)

if PLOT_ARTIFACTS_RES:
    ts_axes[-1,-1].set_xlim(0, FS*30)
    spec_axes[-1, -1].set_xlim(0, 30)
    plt.savefig('raw.png', dpi=200)
    plt.show()


# save all subj envelopes
probes_df.to_pickle('envelopes.pkl')