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
                curve[np.isinf(curve)] = np.nan
                curve = pd.Series(curve).fillna(method='bfill').fillna(method='ffill')
                curve[np.isnan(curve)] = 0.001
                #if np.any(np.isnan(curve)): print(curve)
                y_df = y_df.append(pd.DataFrame({'metric_type':metric_type, 'fb_type': fb_type, 'subj_id': 's'+str(subj_id), 'channel': ch, 'k': np.linspace(0, 1, 30), 'env': curve+0.0001}), ignore_index=True)

y_df.to_csv('fb_curves.csv', index=False)

import statsmodels.api as sm

fig, axes = plt.subplots(4, 4)
for m, metric_type in enumerate(['magnitude', 'n_spindles', 'duration', 'amplitude']):
    for j, fb_type in enumerate(['FB0', 'FB250', 'FB500', 'FBMock']):

        data = y_df.query('metric_type == "{}" & fb_type=="{}"'.format(metric_type, fb_type))
        md = sm.MixedLM.from_formula("env ~ k:channel + channel", data, groups=data["subj_id"], re_formula='k')


        results = md.fit()

        print(results.summary())


        pvals = np.array([results.pvalues[results.pvalues.index=='k:channel[{}]'.format(ch)][0] for ch in CHANNELS])
        im = plot_topomap(np.log10(pvals), MONTAGE.get_pos(), vmin=-4, vmax=0, show=False, cmap='inferno_r', mask=np.array(CHANNELS)=='P4',contours=[np.log10(0.001), np.log10(0.01), np.log10(0.05)], axes=axes[m, j])
        axes[m, j].set_title('{}\n{}'.format(metric_type, fb_type))


plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cax = plt.axes([0.85, 0.1, 0.075, 0.8])

b = plt.colorbar(im[0], cax=cax)
b.set_ticks([np.log10(0.001), np.log10(0.01), np.log10(0.05)])
b.set_ticklabels(['***', '**', '*'])











fig, axes = plt.subplots(4, 4)
for m, metric_type in enumerate(['magnitude', 'n_spindles', 'duration', 'amplitude'][:]):
    data = y_df.query('metric_type == "{}"'.format(metric_type))
    md = sm.MixedLM.from_formula("np.log(env) ~ k:fb_type:channel + fb_type", data, groups="subj_id", re_formula='1')
    results = md.fit()
    cis = results.conf_int()
    ci_mock = np.array([cis[cis.index == 'k:fb_type[FBMock]:channel[{}]'.format(ch)][1][0] for ch in CHANNELS])
    for j, fb_type in enumerate(['FB0', 'FB250', 'FB500']):
        slopes = [results.params['k:fb_type[{}]:channel[{}]'.format(fb_type, ch)] for ch in CHANNELS]
        tvals = [results.tvalues['k:fb_type[{}]:channel[{}]'.format(fb_type, ch)] for ch in CHANNELS]
        ci = np.array([cis[cis.index == 'k:fb_type[{}]:channel[{}]'.format(fb_type, ch)][0][0] for ch in CHANNELS])
        plot_topomap(tvals, MONTAGE.get_pos(), vmin=-10, vmax=10, show=False, cmap='RdBu_r',
                     mask=ci>ci_mock,
                     axes=axes[m, j])







slopes = []
p_vals = []
for ch in CHANNELS:
    data = y_df.query('np.log(metric_type) == "amplitude" & channel=="{}"'.format(ch))

    md = sm.MixedLM.from_formula("np.log(env) ~ k:fb_type", data, groups=data["subj_id"], re_formula="k")

    results = md.fit()
    slopes.append(results.params[1:5])
    p_vals.append(results.pvalues[1:5])

    print(ch)



fig, axes = plt.subplots(1, 4)
for k in range(4):
    im = plot_topomap(np.log10(np.array(p_vals)[:, k]), MONTAGE.get_pos(), vmin=-5, vmax=0, mask=np.array(p_vals)[:, k]<0.0001, show=False, cmap='inferno_r', contours=[1, 1.25], axes=axes[k])
