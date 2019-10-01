from mne.stats import fdr_correction
from pymer4.models import Lm, Lmer
import pandas as pd
import numpy as np
from scipy import stats

def ranksums(x, y):
    p_value = stats.ranksums(x, y).pvalue
    x, y = map(np.asarray, (x, y))
    n1 = len(x)
    n2 = len(y)
    alldata = np.concatenate((x, y))
    ranked = stats.rankdata(alldata)
    x = ranked[:n1]
    s = np.sum(x, axis=0)
    return s, p_value


stats_df = pd.read_pickle('release/data/{}.pkl'.format('channels1_bands1_splitedTrue_thresholds17'))
stats_df = stats_df.loc[stats_df['block_number']> 1000]
unique_blocks = list(stats_df['block_number'].unique())
stats_df['k'] = stats_df['block_number'].apply(lambda x: unique_blocks.index(x))
stats_df['subj_id_str'] = 's' + stats_df['subj_id'].astype('str')

contrasts = ['FB0 - FBMock', 'FB250 - FBMock', 'FB500 - FBMock', 'FB0 - FB500', 'FB250 - FB500', 'FB0 - FB250']

t_stat_df = pd.DataFrame(columns=['metric_type', 'threshold_factor', 'Contrast', 'Stat', 'P-value'])


for threshold_factor in stats_df.threshold_factor.unique():
    for metric_type in ['magnitude', 'n_spindles', 'amplitude', 'duration']:
        data = stats_df.query('metric_type=="{}" & threshold_factor=={}'.format(metric_type, threshold_factor)).copy()
        data = data.replace([np.inf, -np.inf], np.nan)
        data.loc[:, 'metric'] = data['metric'].fillna(data['metric'].min()).values

        for contrast in contrasts:
            fb1_type, fb2_type = contrast.split(' - ')
            fb1_df = data.query('fb_type=="{}"'.format(fb1_type))
            fb2_df = data.query('fb_type=="{}"'.format(fb2_type))
            fb1_scores = fb1_df.query('k<15').groupby('subj_id').mean()['metric'] / fb1_df.query('k>=15').groupby(
                'subj_id').mean()['metric']
            fb2_scores = fb2_df.query('k<15').groupby('subj_id').mean()['metric'] / fb2_df.query('k>=15').groupby(
                'subj_id').mean()['metric']
            stat, p_value = ranksums(fb2_scores.values, fb1_scores.values)
            t_stat_df = t_stat_df.append({'metric_type': metric_type, 'threshold_factor': threshold_factor,
                                          'Contrast': contrast, 'Stat': stat, 'P-value': p_value}, ignore_index=True)




import pylab as plt

cm = dict(zip(['FB0', 'FB250', 'FB500', 'FBMock'], ['#3CB4E8', '#438BA8', '#002A3B', '#FE4A49']))

fig, axes = plt.subplots(3, 3, sharey='all', sharex='all', figsize=(6, 7))
plt.subplots_adjust(wspace=0, hspace=0)
for j_metric_type, metric_type in enumerate(['n_spindles', 'amplitude', 'duration']):
    axes[0, j_metric_type].set_title(metric_type)
    for contrast in contrasts:
        fb1_type, fb2_type = contrast.split(' - ')
        data = t_stat_df.query('Contrast=="{}" & metric_type=="{}"'.format(contrast, metric_type)).copy()
        ax = axes[(fb2_type=='FB500')*1 + (fb2_type=='FB250')*2, j_metric_type]
        ax.plot(data['Stat'].values, data['threshold_factor'].values, color=cm[fb1_type])
        reject_fdr, pval_fdr = fdr_correction(data['P-value'].values, alpha=0.1, method='indep')
        if len(data['Stat'].values[reject_fdr]) > 0:
            ax.plot(data['Stat'].values[reject_fdr], data['threshold_factor'].values[reject_fdr], '*',
                    color=cm[fb1_type], markersize=7, alpha=0.9)
        ax.axvline(128, color=cm[fb2_type], linestyle='--', zorder=-100)
        ax.axvline(105*2-128, color=cm[fb2_type], linestyle='--', zorder=-100)


#
# [ax.axvline(128, color=cm['FBMock'], linestyle='--', zorder=-100) for ax in axes.flatten()]

