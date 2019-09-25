import pandas as pd
import scipy.signal as sg
from scipy.stats import linregress
import pylab as plt
import numpy as np

probes_df = pd.read_pickle('release/data/eeg_allsubjs_eyefree_1_45hz_down250hz.pkl')

datasets_df = pd.read_pickle('release/data/info_allsubjs.pkl')

probes_df.columns.duplicated()
probes_df.to_pickle('release/data/eeg_allsubjs_eyefree_1_45hz_down250hz.pkl')
for k in range(40):
    plt.plot(*sg.welch(probes_df.query('subj_id=={}'.format(k))['P4'].values, 250, nperseg=1000))

plt.plot(probes_df.query('subj_id==2')['P4'].values)

