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

LOG = False
STAT = ['wilcoxon', 'lmm'][0]
stats_file = 'channels1_bands1_splitedTrue_median_threshs20.pkl'
stats_df = pd.read_pickle('release/data/{}'.format(stats_file))
stats_df = stats_df.loc[stats_df['block_number']> 1000]
unique_blocks = list(stats_df['block_number'].unique())
stats_df['k'] = stats_df['block_number'].apply(lambda x: unique_blocks.index(x))
stats_df['subj_id_str'] = 's' + stats_df['subj_id'].astype('str')



t_stat_df = pd.DataFrame(columns=['metric_type', 'threshold_factor', 'Contrast', 'T-stat'])

if STAT == "lmm":
    for threshold_factor in stats_df.threshold_factor.unique():
        for metric_type in ['magnitude', 'n_spindles', 'amplitude', 'duration']:
            data = stats_df.query('metric_type=="{}" & threshold_factor=={}'.format(metric_type, threshold_factor)).copy()
            if LOG and metric_type in ['magnitude', 'amplitude']:
                model = Lmer('log(metric) ~ k:fb_type + (1 |subj_id_str)', data=data)
            else:
                model = Lmer('metric ~ k:fb_type + (1 |subj_id_str)', data=data)
            model.fit(factors={'fb_type': ['FB0', 'FB250', 'FB500', 'FBMock']}, ordered=True, summarize=False)
            print('***', metric_type)
            res = model.post_hoc('fb_type')[1][['Contrast', 'T-stat']]
            res['Stat'] = res['T-stat']
            res['P-value'] = stats.t.sf(res['T-stat'], 9)
            res['metric_type'] = metric_type
            res['threshold_factor'] = threshold_factor
            t_stat_df = t_stat_df.append(res, ignore_index=True)
else:

    contrasts = ['FB0 - FBMock', 'FB250 - FBMock', 'FB500 - FBMock', 'FB0 - FB500', 'FB250 - FB500', 'FB0 - FB250']
    for threshold_factor in stats_df.threshold_factor.unique():
        for metric_type in ['magnitude', 'n_spindles', 'amplitude', 'duration']:
            data = stats_df.query(
                'metric_type=="{}" & threshold_factor=={}'.format(metric_type, threshold_factor)).copy()
            if not (np.all(np.isfinite(data['metric'])) and data['metric'].notna().all()):
                print(threshold_factor, metric_type)

            for contrast in contrasts:
                fb1_type, fb2_type = contrast.split(' - ')
                fb1_df = data.query('fb_type=="{}"'.format(fb1_type))
                fb2_df = data.query('fb_type=="{}"'.format(fb2_type))
                if LOG:
                    fun = (lambda x: np.log(x)) if metric_type in ['magnitude', 'amplitude'] else (lambda x: x)
                else:
                    fun = lambda x: x
                fb1_scores = fun(fb1_df.query('k>=15').groupby('subj_id').mean()['metric']) / fun(
                    fb1_df.query('k<15').groupby(
                        'subj_id').mean()['metric'])
                fb2_scores = fun(fb2_df.query('k>=15').groupby('subj_id').mean()['metric']) / fun(
                    fb2_df.query('k<15').groupby(
                        'subj_id').mean()['metric'])
                stat, p_value = ranksums(fb1_scores.values, fb2_scores.values)
                t_stat_df = t_stat_df.append({'metric_type': metric_type, 'threshold_factor': threshold_factor,
                                              'Contrast': contrast, 'Stat': stat, 'P-value': p_value},
                                             ignore_index=True)

import pylab as plt

cm = dict(zip(['FB0', 'FB250', 'FB500', 'FBMock'], ['#3CB4E8', '#438BA8', '#002A3B', '#FE4A49']))

for comp_with in ['FB0', 'FB250', 'FB500', 'FBMock']:

    contrasts = ['{} - {}'.format(fb_type, comp_with) for fb_type in ['FB0', 'FB250', 'FB500', 'FBMock'] if fb_type!=comp_with]
    fig, axes = plt.subplots(1, 4, sharey='all', sharex='all', figsize=(8, 3))

    plt.suptitle('Comparison with {} condition'.format(comp_with))
    plt.subplots_adjust(wspace=0, right=0.84, bottom=0.2, top=0.8)
    for j_metric_type, metric_type in enumerate(['magnitude', 'n_spindles', 'amplitude', 'duration']):
        axes[j_metric_type].set_title(r'{}, log($\beta_1$)'.format(metric_type) if (LOG and 'ude' in metric_type) else r'{}, $\beta_1$'.format(metric_type))
        for contrast in contrasts:
            fb1_type, fb2_type = contrast.split(' - ')
            data = t_stat_df.query('Contrast=="{}" & metric_type=="{}"'.format(contrast, metric_type)).copy()
            if len(data) == 0:
                data = t_stat_df.query('Contrast=="{}" & metric_type=="{}"'.format(' - '.join(contrast.split(' - ')[::-1]), metric_type)).copy()
                if STAT == 'lmm':
                    data['Stat'] *= -1
                else:
                    data['Stat'] = 105*2-data['Stat']
            ax = axes[j_metric_type]
            ax.plot(data['Stat'].values, data['threshold_factor'].values, color=cm[fb1_type])
            reject_fdr, pval_fdr = fdr_correction(data['P-value'].values, alpha=0.1, method='indep')
            if len(data['Stat'].values[reject_fdr]) > 0:
                ax.plot(data['Stat'].values[reject_fdr], data['threshold_factor'].values[reject_fdr], '*',
                        color=cm[fb1_type], markersize=5, alpha=0.9)
            if STAT == 'lmm':
                ax.axvline(1.812, color=cm[fb2_type], linestyle='--', zorder=-100)
                ax.axvline(-1.812, color=cm[fb2_type], linestyle='--', zorder=-100)
            else:
                ax.axvline(128, color=cm[fb2_type], linestyle='--', zorder=-100)
                ax.axvline(105 * 2 - 128, color=cm[fb2_type], linestyle='--', zorder=-100)

    [axes[-1].plot(np.nan, color=cm[fb_type], label=fb_type)  for fb_type in ['FB0', 'FB250', 'FB500', 'FBMock']]
    axes[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    [ax.set_xlabel('T-Stat' if STAT=='lmm' else 'Ranksum') for ax in axes[:]]

    if 'median' in stats_file:
        axes[0].set_ylabel('Threshold factor')
        axes[0].set_ylim(0.85, 3.55)
    elif 'perc' in stats_file:
        axes[0].set_ylabel('Percentile')
        axes[0].set_ylim(45, 105)


    plt.savefig('res_{}_{}{}_compwith{}.png'.format(STAT, 'median' if 'median' in stats_file else 'perc', '_log' if LOG else '', comp_with), dpi=200)
# [ax.axvline(128, color=cm['FBMock'], linestyle='--', zorder=-100) for ax in axes.flatten()]