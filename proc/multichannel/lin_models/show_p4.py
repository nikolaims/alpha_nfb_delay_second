from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
from scipy.stats import linregress, ttest_1samp
import numpy as np
import pylab as plt
#from tqdm import tqdm
from proc.settings import FB_ALL
from proc.settings import CHANNELS, MONTAGE
from mne.viz import plot_topomap
import scipy.signal as sg
from scipy.stats import linregress
from tqdm import tqdm

import pandas as pd


all_stats_df = pd.read_pickle('data/multichannel_multiband_metrics_split_local.pkl')
all_stats_df = all_stats_df.loc[all_stats_df['band']=='alpha']
all_stats_df = all_stats_df.loc[all_stats_df['block_number']>1000]
all_stats_df = all_stats_df.loc[all_stats_df['channel'].isin(CHANNELS)]
all_stats_df = all_stats_df.loc[all_stats_df['threshold_factor'].isin([2])]



y_df = pd.DataFrame(columns=['metric_type', 'FB type', 'subj_id', 'channel', 'block number', 'Envelope, %'])
for metric_type, metric_type_df in all_stats_df.groupby('metric_type'):
    for fb_type, fb_type_df in metric_type_df.groupby('fb_type'):
        for s, (subj_id, subj_df) in enumerate(fb_type_df.groupby('subj_id')):
            for c, (ch, ch_df) in enumerate(subj_df.groupby('channel')):
                curve = ch_df['metric'].values
                curve[np.isinf(curve)] = np.nan
                curve[np.isnan(curve)] = 0.0001

                curve = pd.Series(curve).rolling(2, center=True).mean().fillna(method='ffill').fillna(method='bfill').values[1::2]
                #curve = sg.filtfilt(np.ones(3)/3, [1, 0], curve)
                x0 = linregress(np.arange(0, 15), curve).intercept
                #x0 = curve[:15].mean()
                curve = (curve/x0 - 1)*100
                y_df = y_df.append(pd.DataFrame({'metric_type':metric_type, 'FB type': fb_type, 'subj_id': 's'+str(subj_id), 'channel': ch, 'block number': np.arange(1, 15+1), 'Envelope, %': curve+0.0001}), ignore_index=True)



import seaborn as sns

sns.pointplot('block number', 'Envelope, %', 'FB type', data=y_df.query('channel=="P4"'), dodge=0.4)

