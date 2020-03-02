import pandas as pd
import numpy as np
import scipy.signal as sg
import pylab as plt

probes_df = pd.read_pickle('release/data/eeg_5groups_eyefree_1_45hz_down250hz.pkl')


probes_df = probes_df.query('block_number > 4')

for k in range(42):
    x = probes_df.loc[probes_df.subj_id==k, 'PHOTO']
    x = (x - np.mean(x))/np.std(x)
    plt.semilogy(*sg.welch(x, 250, nperseg=250*10), color='k' if k<40 else 'r', alpha=0.2 if k<40 else 0.9)

plt.ylabel('PHOTO PSD')
plt.xlabel('Freq, HZ')