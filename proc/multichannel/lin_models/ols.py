from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
from scipy.stats import linregress, ttest_1samp
import numpy as np
import pylab as plt
#from tqdm import tqdm
from proc.settings import FB_ALL
from proc.settings import CHANNELS, MONTAGE
from mne.viz import plot_topomap
import scipy.signal as sg


import pandas as pd



#fb_type = 'FBMock'
metric_type = 'n_spindles'

all_stats_df = pd.read_pickle('data/split_metrics_chs_ica_all.pkl')
all_stats_df = all_stats_df.loc[all_stats_df['block_number']>1000]
all_stats_df = all_stats_df.loc[all_stats_df['channel'].isin(CHANNELS)]
all_stats_df = all_stats_df.loc[all_stats_df['threshold_factor'].isin([2])]
all_stats_df = all_stats_df.loc[all_stats_df['metric_type'].isin([metric_type])]


y_df = pd.DataFrame(columns=['fb_type', 'subj_id', 'channel', 'k', 'env'])
for fb_type, fb_type_df in all_stats_df.groupby('fb_type'):
    for s, (subj_id, subj_df) in enumerate(fb_type_df.groupby('subj_id')):
        for c, (ch, ch_df) in enumerate(subj_df.groupby('channel')):
            curve = ch_df['metric'].values
            #
            curve[np.isinf(curve)] = np.nan
            #curve[np.isinf(-curve)] = np.nan
            #curve[np.isnan(curve)] = np.nanmedian(curve)
            curve = pd.Series(curve).fillna(method='bfill')#.rolling(3).median().fillna(method='bfill').rolling(
            #     3).mean().fillna(method='bfill').values
            y_df = y_df.append(pd.DataFrame({'fb_type': fb_type, 'subj_id': 's'+str(subj_id), 'channel': ch, 'k': np.linspace(0, 1, 30), 'env': curve-np.min(curve)*0.0 + 0.001}))


import numpy as np

import statsmodels.api as sm

import statsmodels.formula.api as smf


# Fit regression model (using the natural log of one of the regressors)

#print(results.summary())

f, ax  = plt.subplots(1, 4)
plt.subplots_adjust(bottom=0.5)
p4f, p4ax = plt.subplots(1, 4, sharex=True, sharey=True)
for k, fb_type in enumerate(['FB0', 'FB250', 'FB500', 'FBMock']):
    results = smf.ols('np.log(env) ~ k:fb_type + C(subj_id) + C(channel) - 1', data=y_df.query('fb_type == "{}" & subj_id!="s11"'.format(fb_type))).fit()
    ax[k].set_xlabel(results.summary(), horizontalalignment='left', x=0, fontdict={'family': 'monospace'})
    ax[k].set_title(fb_type)
    print(results.summary())

    indx = [CHANNELS.index('C3')] + [CHANNELS.index(chn.split('.')[1].split(']')[0]) for c, chn in enumerate(results.params.index) if 'channel' in chn]

    plot_topomap(np.exp(np.concatenate([[0], results.params[10:10+31]])), MONTAGE.get_pos()[indx], vmin=0, vmax=1.5, axes=ax[k], show=False, cmap='Reds', contours=[0.75, 1])


    for s, (subj_id, subj_df) in enumerate(y_df.query('fb_type == "{}" & channel=="P4"& subj_id!="s11"'.format(fb_type)).groupby('subj_id')):
        p4ax[k].plot(np.log(subj_df['env'].values)-results.params[s] - results.params[[('P4' in s) for s in results.params.index]][0], alpha=0.9, label=subj_id)

    ax[k].set_title('{}\n{}'.format(fb_type, metric_type))
    p4ax[k].plot(np.linspace(0, 1, 30)*results.params[-1], 'k', linewidth=2, label='model')
    p4ax[k].legend()

