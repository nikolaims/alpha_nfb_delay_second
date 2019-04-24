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
metric_type = 'magnitude'

all_stats_df = pd.read_pickle('data/2split_metrics_chs_ica_all.pkl')
all_stats_df = all_stats_df.loc[all_stats_df['block_number']>1000]
all_stats_df = all_stats_df.loc[all_stats_df['channel'].isin(CHANNELS)]
all_stats_df = all_stats_df.loc[all_stats_df['threshold_factor'].isin([2])]



y_df = pd.DataFrame(columns=['metric_type', 'fb_type', 'subj_id', 'channel', 'k', 'env'])
for metric_type, metric_type_df in all_stats_df.groupby('metric_type'):
    for fb_type, fb_type_df in metric_type_df.groupby('fb_type'):
        for s, (subj_id, subj_df) in enumerate(fb_type_df.groupby('subj_id')):
            for c, (ch, ch_df) in enumerate(subj_df.groupby('channel')):
                curve = ch_df['metric'].values
                #
                curve[np.isinf(curve)] = np.nan
                #curve[np.isinf(-curve)] = np.nan
                #curve[np.isnan(curve)] = np.nanmedian(curve)
                curve = pd.Series(curve).fillna(method='bfill')#.rolling(3).median().fillna(method='bfill').rolling(
                #     3).mean().fillna(method='bfill').values
                y_df = y_df.append(pd.DataFrame({'metric_type':metric_type, 'fb_type': fb_type, 'subj_id': 's'+str(subj_id), 'channel': ch, 'k': np.linspace(0, 1, 30), 'env': curve/(1 if metric_type!='n_spindles' else 100)-np.min(curve)*0.0 + 0.001}))


import numpy as np

import statsmodels.api as sm

import statsmodels.formula.api as smf


# Fit regression model (using the natural log of one of the regressors)

#print(results.summary())

f, ax  = plt.subplots(4, 4)
#p4f, p4ax = plt.subplots(4, 4, sharex=True, sharey=True)
for m, metric_type in enumerate(['magnitude', 'n_spindles', 'amplitude', 'duration']):
    for k, fb_type in enumerate(['FB0', 'FB250', 'FB500', 'FBMock']):
        results = smf.ols('np.log(env) ~ k:fb_type +  C(channel) + C(subj_id) - 1', data=y_df.query('fb_type == "{}" & subj_id!="s11" & metric_type=="{}"'.format(fb_type, metric_type))).fit()
        ax[m, k].set_title('{}\n{}\n[{:.3f}, {:.3f}]'.format(fb_type, metric_type, *results.conf_int().values[-1]))
        print(results.summary())

        indx = [CHANNELS.index(chn.split('[')[1].split(']')[0]) for c, chn in enumerate(results.params.index) if 'channel' in chn]

        im = plot_topomap(np.exp(results.params[:32]-max(results.params[:32]))*(np.exp(results.params[-1])), MONTAGE.get_pos()[indx], vmin=0, vmax=1.2, axes=ax[m, k], show=False, cmap='inferno', contours=[1, 1.25])
        #
        #
        # for s, (subj_id, subj_df) in enumerate(y_df.query('fb_type == "{}" & channel=="P4"& subj_id!="s11"'.format(fb_type)).groupby('subj_id')):
        #     p4ax[k].plot(np.log(subj_df['env'].values)-results.params[s] - results.params[[('P4' in s) for s in results.params.index]][0], alpha=0.9, label=subj_id)
        #
        #
        # p4ax[k].plot(np.linspace(0, 1, 30)*results.params[-1], 'k', linewidth=2, label='model')
        # p4ax[k].legend()

plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cax = plt.axes([0.85, 0.1, 0.075, 0.8])
plt.colorbar(im[0], cax=cax)