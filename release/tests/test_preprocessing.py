import pandas as pd
import scipy.signal as sg
from scipy.stats import linregress
import pylab as plt
import numpy as np

probes_df = pd.read_pickle('release/data/eeg_allsubjs_eyefree_1_45hz_down250hz.pkl')

datasets_df = pd.read_pickle('release/data/info_allsubjs.pkl')

probes_df.columns.duplicated()
probes_df.to_pickle('release/data/eeg_allsubjs_eyefree_1_45hz_down250hz.pkl')

k = 0

plt.figure(figsize=(30, 4))
plt.plot(probes_df.query('subj_id=={}'.format(k))['P4'].values)
k += 1

