import pandas as pd
import numpy as np
import scipy.signal as sg
import pylab as plt

from mne.decoding import SPoC
from  mne.viz import plot_topomap
from release.settings import CHANNELS, FS, FB_ALL, MONTAGE
from pynfb.signal_processing.decompositions import ICADecomposition

eeg_path = 'release/data/eeg_allsubjs_eyefree_1_45hz_down250hz.pkl'
info_path = 'release/data/info_allsubjs.pkl'

eeg_df = pd.read_pickle(eeg_path)
info_df = pd.read_pickle(info_path)


topos = []
corrss = []
for subj_id in range(40):
    print(subj_id)
    eeg_df_subj = eeg_df[eeg_df['subj_id']==subj_id]
    band = info_df.loc[info_df['subj_id']==subj_id, 'band'].values[0]
    fb_type = info_df.loc[info_df['subj_id']==subj_id, 'fb_type'].values[0]
    # filter data

    filter_n_taps = 250
    filter_b = sg.firwin2(filter_n_taps, [0, band[0], band[0], band[1], band[1], FS//2], [0, 0, 1, 1, 0, 0], fs=FS)



    x = eeg_df_subj.loc[eeg_df_subj['block_number'] <= 6, CHANNELS].values
    # x = sg.lfilter(filter_b, [1, 0], x, 0)[filter_n_taps // 2:]
    ica = ICADecomposition(CHANNELS, FS)
    ica.fit(x)

    fb_data = eeg_df_subj.loc[eeg_df_subj['block_number'].isin(FB_ALL), CHANNELS].values
    fb_data = sg.lfilter(filter_b, [1, 0], fb_data, 0)[filter_n_taps // 2:]
    y = fb_data[:, CHANNELS.index('P4')]

    get_pow = lambda x: sg.lfilter(np.arange(FS*10)/FS/10, [1, 0], np.abs(x), axis=0)[::FS*5]

    y_pow = get_pow(y)
    comp_pow = get_pow(fb_data.dot(ica.filters))
    # corrs = [np.corrcoef(comp_pow[:, k], y_pow)[1, 0] for k in range(32)]
    corrs = np.array([np.corrcoef(fb_data.dot(ica.filters[:, k]), y)[1, 0] for k in range(32)])
    corrs_argsort = np.argsort(np.abs(corrs))[::-1]
    fig, axes = plt.subplots(1, 2)
    for k in range(2):
        ax = axes[k]
        plot_topomap(ica.topographies[:, corrs_argsort][:, k], MONTAGE.get_pos(), axes=ax, contours=0)
        ax.set_xlabel('{:.2f}'.format(corrs[corrs_argsort[k]]))
    topos.append(ica.topographies[:, corrs_argsort])
    corrss.append(corrs[corrs_argsort])


pos = MONTAGE.get_pos()
fig, axes = plt.subplots(4, 10)
for i, fb_type in enumerate(['FB0', 'FB250', 'FB500', 'FBMock']):
    for j, subj_id in enumerate(info_df.loc[info_df['fb_type'] == fb_type, 'subj_id'].unique()[:10]):
        spoc = topos[subj_id]
        ax = axes[i, j]
        plot_topomap(spoc[:, 0], pos, axes=ax)
        ax.set_xlabel('{:.2f}'.format(corrss[subj_id][0]))
        fb_type = info_df.loc[info_df['subj_id'] == subj_id, 'fb_type'].values[0]
    axes[i, 0].set_ylabel('{}'.format(fb_type))