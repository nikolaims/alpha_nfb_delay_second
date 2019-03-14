import pandas as pd
from scipy import stats
import seaborn as sns
import numpy as np

stats_df = pd.read_csv('spindles_stats_norm.csv')
#stats_df.fillna(0)


factor_names = ['dataset', 'fb_type', 'metric_type', 'threshold_factor']
scores_df = pd.DataFrame(columns=factor_names + ['score'])

for factors_values, group in stats_df.groupby(factor_names):
    metric = group['metric'].values
    score = (metric[8:].mean()-metric[:8].mean())/metric[0]
    scores_df = scores_df.append(dict(zip(factor_names + ['score'], list(factors_values)+[score])), ignore_index=True)

#scores_df['fb_gain'] = (scores_df['second'] - scores_df['first'])/scores_df['first']

#scores_df = scores_df.fillna(0)

factor_names = ['fb_type', 'metric_type', 'threshold_factor']
p_values_df = pd.DataFrame(columns=factor_names + ['pvalue', 'pvalue_type'])
for factors_values, group in scores_df.groupby(factor_names):
    mock = scores_df.query('fb_type=="FBMock" & metric_type=="{}" & threshold_factor=={}'.format(*factors_values[1:]))
    p_value_mock = stats.ttest_ind(group['score'].values, mock['score'].values).pvalue

    p_value_0 = stats.ttest_1samp(group['score'].values, 0).pvalue

    pvalue_dict = dict(zip(factor_names + ['pvalue', 'pvalue_type'], list(factors_values) + [[p_value_mock, p_value_0], ['mock',0]]))
    print(pvalue_dict)
    p_values_df = p_values_df.append(pd.DataFrame(pvalue_dict), ignore_index=True)


g = sns.relplot('threshold_factor', 'pvalue', 'fb_type', col='metric_type', data=p_values_df, kind='line',
                row='pvalue_type', col_order=['magnitude', 'n_spindles', 'duration', 'amplitude'])
g.axes[0][0].semilogy()
[[ax.axhline(p, color='k', linestyle='--') for ax in g.axes.flatten()] for p in [0.05, 0.01, 0.001]]


g = sns.catplot('fb_type', 'score', 'fb_type', col='metric_type', data=scores_df, kind='box', sharey=False)
[[ax.axhline(p, color='k', linestyle='--') for ax in g.axes[0]] for p in [0]]

stats_df['logsnr'] = np.log10(stats_df['snr'])
sns.relplot('block_number', 'metric', data=stats_df.query('threshold_factor==1.5'), hue='logsnr', col='fb_type', kind='line', ci=None,
            estimator=np.median, col_order=['FB0', 'FB250', 'FB500', 'FBMock'], row='metric_type', facet_kws={'sharey': 'none'}, palette='viridis_r')