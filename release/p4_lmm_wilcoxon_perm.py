from mne.stats import fdr_correction
from pymer4.models import Lm, Lmer
import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations

from scipy.stats import rankdata, linregress

from release.settings import FB_ALL

def get_stat_mul(x):
    slopes = np.zeros(x.shape[0])
    for j, xs in enumerate(x):
        lr = linregress(np.arange(15), xs)
        slopes[j] = xs.mean()/lr.intercept*100-100
    return np.mean(slopes)


def get_perm_pvalue(fb1_df, fb2_df, n_perm=1000):
    fb1_x = np.array([subj_df['metric'].values for _, subj_df in fb1_df.groupby('subj_id')])[:8]
    fb2_x = np.array([subj_df['metric'].values for _, subj_df in fb2_df.groupby('subj_id')])[:8]
    all_x = np.vstack([fb1_x, fb2_x])
    obs = get_stat_mul(fb1_x) - get_stat_mul(fb2_x)
    stats2 = np.zeros(n_perm)
    for j_perm in range(n_perm):
        indx = np.random.permutation(all_x.shape[0])
        fb1_x_perm = all_x[indx[:fb1_x.shape[0]]]
        fb2_x_perm = all_x[indx[fb1_x.shape[0]:]]
        stats2[j_perm] = get_stat_mul(fb1_x_perm) - get_stat_mul(fb2_x_perm)
    return obs, sum(stats2 > obs) / n_perm


def ranksums(x, y):
    res = stats.ranksums(x, y)
    p_value = res.pvalue
    # x, y = map(np.asarray, (x, y))
    # n1 = len(x)
    # n2 = len(y)
    # alldata = np.concatenate((x, y))
    # ranked = stats.rankdata(alldata)
    # x = ranked[:n1]
    # s = np.sum(x, axis=0)
    s = res.statistic
    return s, p_value

LOG = False
STAT = ['wilcoxon', 'lmm', 'perm'][0]
stats_file = 'block_stats_1channels_1bands_median_20ths.pkl'
stats_df = pd.read_pickle('release/data/{}'.format(stats_file))
# stats_df = stats_df.loc[stats_df.subj_id!=28]
if 'splitedTrue' in stats_file:
    stats_df = stats_df.loc[stats_df['block_number'] > 1000]
else:
    stats_df = stats_df.loc[stats_df['block_number'].isin(FB_ALL)]
unique_blocks = list(stats_df['block_number'].unique())
stats_df['k'] = stats_df['block_number'].apply(lambda x: unique_blocks.index(x))
halfs_query_str = ['k < {}'.format(len(unique_blocks)//2), 'k >= {}'.format(len(unique_blocks)//2)]


stats_df['subj_id_str'] = 's' + stats_df['subj_id'].astype('str')

USE_FBLOW = False
fb_types = ['FBLow', 'FB0', 'FB250', 'FB500', 'FBMock'][1 - int(USE_FBLOW):]
fb_type_colors = ['#5be7bd', '#3CB4E8', '#438BA8', '#002A3B', '#FE4A49'][1 - int(USE_FBLOW):]

t_stat_df = pd.DataFrame(columns=['metric_type', 'threshold_factor', 'Contrast', 'T-stat'])

if STAT == "lmm":
    for threshold_factor in stats_df.threshold_factor.unique():
        for metric_type in ['magnitude', 'n_spindles', 'amplitude', 'duration']:
            data = stats_df.query('metric_type=="{}" & threshold_factor=={}'.format(metric_type, threshold_factor)).copy()
            if LOG and metric_type in ['magnitude', 'amplitude']:
                model = Lmer('log(metric) ~ k:fb_type + (1 |subj_id_str)', data=data)
            else:
                model = Lmer('metric ~ k:fb_type + (1 |subj_id_str)', data=data)
            model.fit(factors={'fb_type': fb_types}, ordered=True, summarize=False)
            print('***', metric_type)
            res = model.post_hoc('fb_type')[1][['Contrast', 'T-stat']]
            res['Stat'] = res['T-stat']
            res['P-value'] = stats.t.sf(res['T-stat'], 9)
            res['metric_type'] = metric_type
            res['threshold_factor'] = threshold_factor
            t_stat_df = t_stat_df.append(res, ignore_index=True)
elif STAT == 'wilcoxon':

    # contrasts = ['FB0 - FBMock', 'FB250 - FBMock', 'FB500 - FBMock', 'FB0 - FB500', 'FB250 - FB500', 'FB0 - FB250']
    contrasts = [' - '.join(pair) for pair in combinations(fb_types, 2)]
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
                fb1_scores = fun(fb1_df.query(halfs_query_str[1]).groupby('subj_id').mean()['metric']) / fun(
                    fb1_df.query(halfs_query_str[0]).groupby(
                        'subj_id').mean()['metric'])
                fb2_scores = fun(fb2_df.query(halfs_query_str[1]).groupby('subj_id').mean()['metric']) / fun(
                    fb2_df.query(halfs_query_str[0]).groupby(
                        'subj_id').mean()['metric'])
                stat, p_value = ranksums(fb1_scores.values, fb2_scores.values)
                t_stat_df = t_stat_df.append({'metric_type': metric_type, 'threshold_factor': threshold_factor,
                                              'Contrast': contrast, 'Stat': stat, 'P-value': p_value},
                                             ignore_index=True)

else:
    contrasts = [' - '.join(pair) for pair in combinations(fb_types, 2)]
    for threshold_factor in stats_df.threshold_factor.unique()[::2]:
        for metric_type in ['magnitude', 'n_spindles', 'amplitude', 'duration']:
            data = stats_df.query(
                'metric_type=="{}" & threshold_factor=={}'.format(metric_type, threshold_factor)).copy()
            if not (np.all(np.isfinite(data['metric'])) and data['metric'].notna().all()):
                print(threshold_factor, metric_type)

            for contrast in contrasts:
                fb1_type, fb2_type = contrast.split(' - ')
                fb1_df = data.query('fb_type=="{}"'.format(fb1_type))
                fb2_df = data.query('fb_type=="{}"'.format(fb2_type))
                stat, p_value = get_perm_pvalue(fb1_df, fb2_df, 1000)
                t_stat_df = t_stat_df.append({'metric_type': metric_type, 'threshold_factor': threshold_factor,
                                              'Contrast': contrast, 'Stat': stat, 'P-value': p_value},
                                             ignore_index=True)


import pylab as plt

cm = dict(zip(fb_types, fb_type_colors))

for comp_with in fb_types:

    contrasts = ['{} - {}'.format(fb_type, comp_with) for fb_type in fb_types if fb_type!=comp_with]
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
                    pass#data['Stat'] = 105*2-data['Stat']
            ax = axes[j_metric_type]
            ax.plot(data['Stat'].values, data['threshold_factor'].values, color=cm[fb1_type])
            reject_fdr, pval_fdr = fdr_correction(data['P-value'].values, alpha=0.05, method='indep')
            if len(data['Stat'].values[reject_fdr]) > 0:
                ax.plot(data['Stat'].values[reject_fdr], data['threshold_factor'].values[reject_fdr], '*',
                        color=cm[fb1_type], markersize=5, alpha=0.9)
            if STAT == 'lmm':
                ax.axvline(1.812, color=cm[fb2_type], linestyle='--', zorder=-100)
                ax.axvline(-1.812, color=cm[fb2_type], linestyle='--', zorder=-100)
            else:
                ax.axvline(1.64, color=cm[fb2_type], linestyle='--', zorder=-100)
                ax.axvline(-1.64, color=cm[fb2_type], linestyle='--', zorder=-100)

    [axes[-1].plot(np.nan, color=cm[fb_type], label=fb_type)  for fb_type in fb_types]
    axes[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    [ax.set_xlabel('T-Stat' if STAT=='lmm' else 'Z-score') for ax in axes[:]]

    if 'median' in stats_file:
        axes[0].set_ylabel('Threshold factor')
        axes[0].set_ylim(0.85, 3.55)
    elif 'perc' in stats_file:
        axes[0].set_ylabel('Percentile')
        axes[0].set_ylim(-1.98, 1.98)


    # plt.savefig('res_{}_{}{}_compwith{}.png'.format(STAT, 'median' if 'median' in stats_file else 'perc', '_log' if LOG else '', comp_with), dpi=200)
# [ax.axvline(128, color=cm['FBMock'], linestyle='--', zorder=-100) for ax in axes.flatten()]