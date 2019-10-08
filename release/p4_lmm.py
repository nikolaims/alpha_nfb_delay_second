from mne.stats import fdr_correction
from pymer4.models import Lm, Lmer
import pandas as pd
import numpy as np
from scipy import stats

LOG = False
stats_file = 'channels1_bands1_splitedTrue_median_thresholds_pers20.pkl'
stats_df = pd.read_pickle('release/data/{}'.format(stats_file))
stats_df = stats_df.loc[stats_df['block_number']> 1000]
unique_blocks = list(stats_df['block_number'].unique())
stats_df['k'] = stats_df['block_number'].apply(lambda x: unique_blocks.index(x))
stats_df['subj_id_str'] = 's' + stats_df['subj_id'].astype('str')



t_stat_df = pd.DataFrame(columns=['metric_type', 'threshold_factor', 'Contrast', 'T-stat'])


for threshold_factor in stats_df.threshold_factor.unique():
    for metric_type in ['magnitude', 'n_spindles', 'amplitude', 'duration']:
        data = stats_df.query('metric_type=="{}" & threshold_factor=={}'.format(metric_type, threshold_factor)).copy()
        if LOG and metric_type in ['magnitude', 'amplitude']:
            model = Lmer('log(metric) ~ k:fb_type + (1 |subj_id_str)', data=data)
        else:
            model = Lmer('metric ~ k:fb_type + (1 |subj_id_str)', data=data)
        model.fit(factors={'fb_type': ['FB0', 'FB250', 'FB500', 'FBMock']})
        print('***', metric_type)
        res = model.post_hoc('k', 'fb_type')[1][['Contrast', 'T-stat']]
        res['Stat'] = res['T-stat']
        res['P-value'] = stats.t.sf(res['T-stat'], 9)
        res['metric_type'] = metric_type
        res['threshold_factor'] = threshold_factor
        t_stat_df = t_stat_df.append(res, ignore_index=True)


import pylab as plt

cm = dict(zip(['FB0', 'FB250', 'FB500', 'FBMock'], ['#3CB4E8', '#438BA8', '#002A3B', '#FE4A49']))

fig, axes = plt.subplots(3, 4, sharey='all', sharex='all', figsize=(6, 7))
plt.subplots_adjust(wspace=0, hspace=0)
for j_metric_type, metric_type in enumerate(['magnitude', 'n_spindles', 'amplitude', 'duration']):
    axes[0, j_metric_type].set_title(metric_type)
    for contrast in t_stat_df['Contrast'].unique():
        fb1_type, fb2_type = contrast.split(' - ')
        data = t_stat_df.query('Contrast=="{}" & metric_type=="{}"'.format(contrast, metric_type)).copy()
        ax = axes[(fb2_type=='FB500')*1 + (fb2_type=='FB250')*2, j_metric_type]
        ax.plot(data['Stat'].values, data['threshold_factor'].values, color=cm[fb1_type])
        reject_fdr, pval_fdr = fdr_correction(data['P-value'].values, alpha=0.10, method='indep')
        if len(data['Stat'].values[reject_fdr]) > 0:
            ax.plot(data['Stat'].values[reject_fdr], data['threshold_factor'].values[reject_fdr], '*',
                    color=cm[fb1_type], markersize=7, alpha=0.9)
        ax.axvline(2.228, color=cm[fb2_type], linestyle='--', zorder=-100)
        ax.axvline(-2.228, color=cm[fb2_type], linestyle='--', zorder=-100)

axes[0,0].set_ylabel('FBMock\nThreshold')
axes[1,0].set_ylabel('FB500\nThreshold')
axes[2,0].set_ylabel('FB250\nThreshold')

[ax.set_xlabel('T-stat') for ax in axes[2, :]]
# plt.savefig('lmm-tests.png', dpi=200)