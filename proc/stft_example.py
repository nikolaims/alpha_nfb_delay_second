import pandas as pd
import numpy as np
import scipy.signal as sg
from proc.settings import FS, CHANNELS, FB_ALL, ICA_CHANNELS
import pylab as plt

SPLIT_FB_BLOCKS = True
GLOBAL_THRESHOLD = False
USE_ICA = False
P4_ONLY = True
channels = (['P4'] if P4_ONLY else CHANNELS) + (ICA_CHANNELS if USE_ICA else [])
threshold_factors = np.arange(1, 3.1, 0.125)
bands =dict(zip(['alpha'], [1]))
res_df_name = 'channels{}_bands{}_splited{}_thresholds{}_globthr{}'.format(len(channels), len(bands), SPLIT_FB_BLOCKS, len(threshold_factors), GLOBAL_THRESHOLD)
print(res_df_name)

# load pre filtered data
probes_df = pd.read_pickle('data/eeg_allsubjs_eyefree_1_45hz_down250hz.pkl')

# load datasets info
datasets_df = pd.read_pickle('data/info_allsubjs.pkl')


d = (600 + 26)*FS
slc = slice(FS*60 + d, FS*120 + d)
x = probes_df.query('subj_id==2')['P4'].values[slc]

f, t, Sxx = sg.spectrogram(x, FS, nperseg=250, noverlap=240, nfft=250)
t -= 2

fig, ax = plt.subplots(2, sharex='all')
ax[0].plot(np.arange(FS*60)/FS-2, probes_df.query('subj_id==2')[['FP1', 'CZ', 'PZ'][::-1]].values[slc]+np.arange(1, 4)*1e-4*0.5, color='k', alpha=0.3)
ax[0].plot(np.arange(FS*60)/FS-2, probes_df.query('subj_id==2')['P4'].values[slc], color='#05adf5')
ax[1].pcolormesh(t, f, Sxx**0.5, cmap='Blues')
ax[1].set_ylim(0, 20)
ax[1].set_xlabel('Time, s')
ax[1].set_ylabel('Frequency, Hz')
plt.xlim(0, 5)