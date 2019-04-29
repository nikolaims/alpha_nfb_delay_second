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

                curve = pd.Series(curve).rolling(5, center=True).median().fillna(method='ffill').fillna(method='bfill')
                #curve = sg.filtfilt(np.ones(3)/3, [1, 0], curve)
                x0 = linregress(np.linspace(0, 1, 30), curve).intercept
                #x0 = curve[:15].mean()
                curve = curve - x0
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






import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.viz import iter_topography
from mne import io
from mne.time_frequency import psd_welch
from mne.datasets import sample

from mne import create_info
from mne.channels import read_montage

info = create_info(CHANNELS, 250, 'eeg', read_montage('standard_1005'))




metric_type = 'magnitude'


curves_array = np.zeros((4, 32, 30))
for ax, idx in iter_topography(info, fig_facecolor='white', axis_facecolor='white', axis_spinecolor='white'):
    curves = {}
    for f, fb_type in enumerate(['FB0', 'FB250', 'FB500', 'FBMock']):
        curves[fb_type] = y_df.query('metric_type=="{}" & fb_type=="{}" & channel=="{}"'.format(metric_type, fb_type, CHANNELS[idx])).groupby('k').median()['env'].values
        curves_array[f, idx, :] = curves[fb_type]
    ax.plot(curves['FB0']-curves['FBMock']*0)
    ax.plot(curves['FB250'] - curves['FBMock']*0)
    ax.plot(curves['FB500'] - curves['FBMock']*0)
    ax.plot(curves['FBMock'])
    ax.set_xlabel(CHANNELS[idx])
    ax.axhline(0, color='k', alpha=0.3)
    if idx>0:
        ax.get_shared_x_axes().join(prev_ax, ax)
        ax.get_shared_y_axes().join(prev_ax, ax)
    prev_ax = ax
#ax.set_ylim(0, y_df.query('metric_type=="{}"'.format(metric_type))['env'].max())
plt.gcf().suptitle(metric_type)
plt.show()





fig, axes = plt.subplots(4)
for f, fb_type in enumerate(['FB0', 'FB250', 'FB500', 'FBMock']):
    plot_topomap(curves_array[f].mean(1), MONTAGE.get_pos(), 0, curves_array.mean(2).max(), axes=axes[f], show=False, cmap='inferno')