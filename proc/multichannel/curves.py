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


all_stats_df = pd.read_pickle('data/multichannel_metrics_split_local.pkl')
all_stats_df = all_stats_df.loc[all_stats_df['block_number']>1000]
all_stats_df = all_stats_df.loc[all_stats_df['channel'].isin(CHANNELS)]
all_stats_df = all_stats_df.loc[all_stats_df['threshold_factor'].isin([2])]



y_df = pd.DataFrame(columns=['metric_type', 'fb_type', 'subj_id', 'channel', 'k', 'env'])
for metric_type, metric_type_df in all_stats_df.groupby('metric_type'):
    for fb_type, fb_type_df in metric_type_df.groupby('fb_type'):
        for s, (subj_id, subj_df) in enumerate(fb_type_df.groupby('subj_id')):
            for c, (ch, ch_df) in enumerate(subj_df.groupby('channel')):
                curve = ch_df['metric'].values
                curve[np.isinf(curve)] = np.nan
                curve[np.isnan(curve)] = 0.0001

                curve = sg.filtfilt(np.ones(3)/3, [1, 0], curve)
                x0 = linregress(np.linspace(0, 1, 30), curve).intercept
                #x0 = curve[:15].mean()
                curve = curve/x0-1
                y_df = y_df.append(pd.DataFrame({'metric_type':metric_type, 'fb_type': fb_type, 'subj_id': 's'+str(subj_id), 'channel': ch, 'k': np.linspace(0, 1, 30), 'env': curve+0.0001}), ignore_index=True)


fig, axes = plt.subplots(4, 6, sharey='row', sharex='all')
for m, (metric_type, metric_type_df) in enumerate(y_df.query('channel=="C3"').groupby('metric_type')):
    for f, (fb_type, fb_type_df) in enumerate(metric_type_df.groupby('fb_type')):
        curves = []
        for s, (subj_id, subj_df) in enumerate(fb_type_df.groupby('subj_id')):
            curve = subj_df['env'].values
            axes[m, f].plot(curve)
            axes[m, f].set_title('{}\n{}'.format(metric_type, fb_type))
            curves.append(curve)
        axes[m, f].plot(np.mean(curves, 0), 'k')
        axes[m, f].plot(np.median(curves, 0), 'k--')
        axes[m, 4].plot(np.mean(curves, 0))
        axes[m, 4].fill_between(np.arange(30), np.mean(curves, 0) - np.std(curves, 0)/10**0.5, np.mean(curves, 0) + np.std(curves, 0)/10**0.5, alpha=0.2)
        axes[m, 5].plot(np.median(curves, 0))
        axes[m, 5].fill_between(np.arange(30), np.percentile(curves, 5, 0), np.percentile(curves, 95, 0), alpha=0.2)

