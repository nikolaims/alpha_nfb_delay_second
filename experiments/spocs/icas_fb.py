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


topographies = []
filters = []
for subj_id in range(40):
    print(subj_id)
    eeg_df_subj = eeg_df[eeg_df['subj_id']==subj_id]
    x = eeg_df_subj.loc[eeg_df_subj['block_number'].isin(FB_ALL), CHANNELS].values
    # x = sg.lfilter(filter_b, [1, 0], x, 0)[filter_n_taps // 2:]
    ica = ICADecomposition(CHANNELS, FS)
    ica.fit(x)
    topographies.append(ica.topographies)
    filters.append(ica.filters)


topo_array = np.array(topographies)
np.save('experiments/spocs/data/ica_fb_topo.npy', np.array(topographies))
np.save('experiments/spocs/data/ica_fb_filt.npy', np.array(filters))
#
# fb_data = eeg_df_subj.loc[:, CHANNELS].values
# fb_data = sg.lfilter(filter_b, [1, 0], fb_data, 0)[filter_n_taps // 2:]
# y = fb_data[:, CHANNELS.index('P4')]
#
#
# p4_coefs = ica.topographies[CHANNELS.index('P4')]
# argsort = np.argsort(np.abs(p4_coefs))[::-1]
# filters = ica.filters[:, argsort]
# comps = fb_data.dot(filters)*p4_coefs[argsort][None, :]
#
#
#
# get_pow = lambda x: sg.lfilter(np.arange(FS*20)/FS/20, [1, 0], np.abs(x), axis=0)[FS*20::FS*10]
# plt.plot(get_pow(y), '--')
# plt.plot(get_pow(comps[:, 0:5]))
