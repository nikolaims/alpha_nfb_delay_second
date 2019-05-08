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
                #x0 = linregress(np.linspace(0, 1, 30), curve).intercept
                #x0 = curve[:15].mean()
                #curve = curve/x0 - 1
                y_df = y_df.append(pd.DataFrame({'metric_type':metric_type, 'fb_type': fb_type, 'subj_id': 's'+str(subj_id), 'channel': ch, 'k': np.linspace(0, 1, 30), 'env': curve+0.0001}), ignore_index=True)

y_df.to_csv('multichannel_metrics.csv')

metric_type = 'magnitude'
fb_type = 'FB0'
ch = 'P4'


import statsmodels.api as sm
from mne.stats import fdr_correction
#
# h0 = sm.MixedLM.from_formula('env ~ 1 + k:fb_type',  groups='subj_id', data=data, re_formula='1 + k').fit()
# h0.summary()
#
#
# h1 = sm.MixedLM.from_formula('env ~ 1 + k',  groups='subj_id', data=data, re_formula='1 + k').fit()
# h1.summary()


pval = np.zeros((4, 3, 32))


for m, metric_type in enumerate(['magnitude', 'n_spindles', 'duration', 'amplitude']):
    for f, fb_type in enumerate(['FB0', 'FB250', 'FBMock']):
        for c, ch in enumerate(CHANNELS):
            data = y_df.query('metric_type=="{}" & (fb_type=="{}" | fb_type=="FB500") & channel=="{}"'.format(metric_type, fb_type, ch))
            h0 = sm.OLS.from_formula('env ~ 1 + k:fb_type', data=data).fit()
            h1 = sm.OLS.from_formula('env ~ 1 + k', data=data).fit()
            pval[m, f, c] = h0.compare_lm_test(h1)[1]


fig, axes = plt.subplots(4, 3)
mask, pval_corr = fdr_correction(pval)
for m, metric_type in enumerate(['magnitude', 'n_spindles', 'duration', 'amplitude']):
    for f, fb_type in enumerate(['FB0', 'FB250', 'FB500']):
        plot_topomap(np.log10(pval_corr[m, f]), MONTAGE.get_pos(), vmin=-5, vmax=0, cmap='inferno_r', axes=axes[m, f], show=False, contours=[-np.log10(0.05)])