import pandas as pd
import numpy as np
import scipy.signal as sg
import pylab as plt

from mne.decoding import SPoC
from  mne.viz import plot_topomap
from release.settings import CHANNELS, FS, FB_ALL, MONTAGE

eeg_path = 'release/data/eeg_allsubjs_eyefree_1_45hz_down250hz.pkl'
info_path = 'release/data/info_allsubjs.pkl'

eeg_df = pd.read_pickle(eeg_path)
info_df = pd.read_pickle(info_path)


eeg_df = eeg_df[eeg_df['block_number'].isin(FB_ALL)]
eeg_df.loc[:, 'block_number'] = eeg_df['block_number'].map(dict(zip(FB_ALL, np.arange(len(FB_ALL)))))



n_components = 4
fig, axes = plt.subplots(4, 40)
spocs = []
for subj_id in range(40):
    print(subj_id)
    eeg_df_subj = eeg_df[eeg_df['subj_id']==subj_id]
    band = info_df.loc[info_df['subj_id']==subj_id, 'band'].values[0]
    fb_type = info_df.loc[info_df['subj_id']==subj_id, 'fb_type'].values[0]
    # filter data

    filter_n_taps = 250
    filter_b = sg.firwin2(filter_n_taps, [0, band[0], band[0], band[1], band[1], FS//2], [0, 0, 1, 1, 0, 0], fs=FS)



    x = eeg_df_subj[[ch for ch in CHANNELS if ch != 'P4']].values
    x = sg.lfilter(filter_b, [1, 0], x, 0)[filter_n_taps//2:]
    y = eeg_df_subj['PHOTO'].values[filter_n_taps//2:]

    # prepare data for SPOC
    n_times = FS//2
    n_epochs = len(x)//n_times
    x_train = np.zeros((n_epochs, len(CHANNELS)-1, n_times))
    y_train = np.zeros(n_epochs)
    for k in range(0, n_epochs):
        x_train[k] = x[k*n_times:(k+1)*n_times].T
        y_train[k] = x[k*n_times:(k+1)*n_times, CHANNELS.index('P4')].var()
        # y_train[k] = y[k*n_times:(k+1)*n_times].mean()

    spoc = SPoC(n_components=n_components, reg=0.00001)

    spoc.fit(x_train, y_train)
    x_transformed = spoc.transform(x_train)
    # for k in range(n_components):
        # ax = axes[k, subj_id]
        # plot_topomap(spoc.patterns_[k], np.array([MONTAGE.get_pos()[j] for j in range(32) if CHANNELS[j]!='P4']), axes=ax)
        # ax.set_xlabel('{:.2f}'.format(np.corrcoef(x_transformed[:, k], y_train)[1,0]))
    # axes[0, subj_id].set_title(fb_type)
    spocs.append(spoc)


pos = np.array([MONTAGE.get_pos()[j] for j in range(32) if CHANNELS[j]!='P4'])
fig, axes = plt.subplots(4, 10)
for i, fb_type in enumerate(['FB0', 'FB250', 'FB500', 'FBMock']):
    for j, subj_id in enumerate(info_df.loc[info_df['fb_type'] == fb_type, 'subj_id'].unique()[:10]):
        spoc = spocs[subj_id]
        for k in [0]:
            ax = axes[i, j]
            plot_topomap(spoc.patterns_[k]/spoc.patterns_[k, np.argmax(np.abs(spoc.patterns_[k]))], pos, axes=ax)
        fb_type = info_df.loc[info_df['subj_id'] == subj_id, 'fb_type'].values[0]
    axes[i, 0].set_ylabel('{}'.format(fb_type))