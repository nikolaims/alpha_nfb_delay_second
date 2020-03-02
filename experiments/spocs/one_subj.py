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

subj_id = 13
eeg_df = eeg_df[eeg_df['subj_id']==subj_id]
band = info_df.loc[info_df['subj_id']==subj_id, 'band'].values[0]

# filter data

filter_n_taps = 250
filter_b = sg.firwin2(filter_n_taps, [0, band[0], band[0], band[1], band[1], FS//2], [0, 0, 1, 1, 0, 0], fs=FS)


n_components = 4
fig, axes = plt.subplots(n_components, len(FB_ALL))

for b_number in range(len(FB_ALL)):

    x = eeg_df.loc[eeg_df['block_number']==b_number, CHANNELS].values
    x = sg.lfilter(filter_b, [1, 0], x, 0)[filter_n_taps//2:]
    y = eeg_df['PHOTO'].values[filter_n_taps//2:]

    # prepare data for SPOC
    n_times = FS//2
    n_epochs = len(x)//n_times
    x_train = np.zeros((n_epochs, len(CHANNELS), n_times))
    y_train = np.zeros(n_epochs)
    for k in range(0, n_epochs):
        x_train[k] = x[k*n_times:(k+1)*n_times].T
        y_train[k] = x[k*n_times:(k+1)*n_times, CHANNELS.index('P4')].var()
        # y_train[k] = y[k*n_times:(k+1)*n_times].mean()

    spoc = SPoC(n_components=n_components, reg=0.00001)

    spoc.fit(x_train, y_train)
    x_transformed = spoc.transform(x_train)
    for k in range(n_components):
        ax = axes[k, b_number]
        plot_topomap(spoc.patterns_[k], MONTAGE.get_pos(), axes=ax)
        ax.set_xlabel('{:.2f}'.format(np.corrcoef(x_transformed[:, k], y_train)[1,0]))