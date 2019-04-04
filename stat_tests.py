import pandas as pd
from scipy import stats
import seaborn as sns
import numpy as np
from scipy.stats import rankdata
import pylab as plt


def ranksums(x, y):
    x, y = map(np.asarray, (x, y))
    n1 = len(x)
    n2 = len(y)
    alldata = np.concatenate((x, y))
    ranked = rankdata(alldata)
    x = ranked[:n1]
    s = np.sum(x, axis=0)
    return s

stats_df = pd.read_csv('spindles_stats_norm.csv')
#stats_df.fillna(0)


factor_names = ['dataset', 'fb_type', 'metric_type', 'threshold_factor']
scores_df = pd.DataFrame(columns=factor_names + ['score'])
#courses_df = pd.DataFrame(columns=factor_names + ['time_course', 'block_number'])

for factors_values, group in stats_df.groupby(factor_names):
    metric = group['metric'].values
    score = (metric[8:].mean() - metric[1:8].mean()) / metric[1:8].mean()
    scores_df = scores_df.append(dict(zip(factor_names + ['score'], list(factors_values)+[score])), ignore_index=True)

    #courses_df = courses_df.append(pd.DataFrame(dict(zip(factor_names + ['time_course',  'block_number'], list(factors_values) + [metric[:]/metric[:8].mean(), np.arange(metric.shape[0])+1]))), ignore_index=True)



#scores_df['fb_gain'] = (scores_df['second'] - scores_df['first'])/scores_df['first']

#scores_df = scores_df.fillna(0)

factor_names = ['fb_type', 'metric_type', 'threshold_factor']
p_values_df = pd.DataFrame(columns=factor_names + ['runksum', 'comparison'])
pvalue_types = ['FBMock', 'FB500', 'FB250', 'FB0'][::-1]
for factors_values, group in scores_df.groupby(factor_names):
    for pvalue_type in ['FBMock', 'FB500', 'FB250', 'FB0']:
        if factors_values[0] == pvalue_type: continue
        mock = scores_df.query('fb_type=="{}" & metric_type=="{}" & threshold_factor=={}'.format(pvalue_type, *factors_values[1:]))
        p_value_mock = stats.ranksums(group['score'].values, mock['score'].values).statistic
        print('*' if stats.shapiro(group['score'].values)[1]<0.05 else '-', factors_values)

        #p_value_0 = stats.ranksums(group['score'].values, 0)[0]

        pvalue_dict = dict(zip(factor_names + ['runksum', 'comparison'], list(factors_values) + [[p_value_mock], [pvalue_type]]))
        #print(pvalue_dict)
        p_values_df = p_values_df.append(pd.DataFrame(pvalue_dict), ignore_index=True)

#sns.set(rc={'figure.figsize':(2,2)})
#sns.set(font_scale=2)
g = sns.relplot('threshold_factor', 'runksum', 'fb_type', col='metric_type', data=p_values_df, kind='line',
                row='comparison', col_order=['magnitude', 'n_spindles', 'duration', 'amplitude'], row_order=pvalue_types, height=2.5)
#g.axes[0][0].semilogy()
#g.fig.set_size_inches(10,10)
[[ax.axhline(p, color='k', linestyle='--') for ax in g.axes.flatten()] for p in [-2, 2]]
for color, axes in zip(sns.color_palette(), g.axes):
   [ax.axhspan(-2, 2, color=color, alpha=0.2) for ax in axes.flatten()]
#plt.tight_layout()


[ax.set_title(title) for title, ax in zip(['magnitude', 'n_spindles', 'duration', 'amplitude'], g.axes[0])]
[ax.set_title('') for ax in  g.axes[1:].flatten()]
[ax.set_ylabel(title) for title, ax in zip(list(map(lambda x: x+'\nranksum', pvalue_types)), g.axes[:, 0])]
# g = sns.catplot('fb_type', 'score', 'fb_type', col='metric_type', data=scores_df, kind='box', sharey=False)
# [[ax.axhline(p, color='k', linestyle='--') for ax in g.axes[0]] for p in [0]]
#
# stats_df['logsnr'] = np.log10(stats_df['snr'])
# sns.relplot('block_number', 'time_course', 'fb_type',data=courses_df.query('threshold_factor==1.5'), col='fb_type', kind='line', ci=95,
#             estimator=np.median, col_order=['FB0', 'FB250', 'FB500', 'FBMock'], hue_order=['FB0', 'FB250', 'FB500', 'FBMock'], row='metric_type', facet_kws={'sharey': 'row'}, row_order=['magnitude', 'n_spindles', 'duration', 'amplitude'])

plt.savefig('nfb_metric_tests.png', dpi=200)
