import pandas as pd
from scipy import stats
import seaborn as sns
import numpy as np
from scipy.stats import rankdata
import pylab as plt
from proc.settings import FB_ALL

def ranksums(x, y):
    x, y = map(np.asarray, (x, y))
    n1 = len(x)
    n2 = len(y)
    alldata = np.concatenate((x, y))
    ranked = rankdata(alldata)
    x = ranked[:n1]
    s = np.sum(x, axis=0)
    return s

stats_df = pd.read_pickle('data/metrics_chs_ica_all.pkl').query('channel=="P4"')
stats_df = stats_df.loc[stats_df['block_number'].isin(FB_ALL)]
#stats_df = stats_df.loc[stats_df['band'].isin(['alpha'])]
stats_df = stats_df.loc[stats_df['threshold_factor'].isin([1.75])]
#stats_df.fillna(0)


factor_names = ['subj_id', 'fb_type', 'metric_type']
scores_df = pd.DataFrame(columns=factor_names + ['metric', 'block_number'])
#courses_df = pd.DataFrame(columns=factor_names + ['time_course', 'block_number'])

for factors_values, group in stats_df.groupby(factor_names):
    curve = group['metric'].values
    curve[np.isinf(curve)] = np.nan
    curve[np.isnan(curve)] = 0.0001

    curve = pd.Series(curve).rolling(2, center=True).median().fillna(method='ffill').fillna(method='bfill')
    score = curve / curve[1:8].mean()
    #score = metric[-1]/metric[2]
    scores_df = scores_df.append(pd.DataFrame(dict(zip(factor_names + ['metric', 'block_number'], list(factors_values)+[score, np.arange(1, 16)]))), ignore_index=True)

    #courses_df = courses_df.append(pd.DataFrame(dict(zip(factor_names + ['time_course',  'block_number'], list(factors_values) + [metric[:]/metric[:8].mean(), np.arange(metric.shape[0])+1]))), ignore_index=True)

g = sns.relplot('block_number', 'metric', 'fb_type', data=scores_df, row='metric_type', col='fb_type', kind='line',
            facet_kws={'sharey': 'row'}, row_order=['magnitude', 'n_spindles', 'duration', 'amplitude'],
            col_order=['FB0', 'FB250', 'FB500', 'FBMock'], palette=['#002A3B', '#438BA8', '#6AC3E8', '#FE4A49'], hue_order=['FB0', 'FB250', 'FB500', 'FBMock'])

g.axes[-1,-1].set_xticks(np.arange(1, 15, 3))

# ['#084C61','#0099D8',   '#84BCDA', '#FE4A49']
g = sns.relplot('block_number', 'metric', 'fb_type', col='fb_type', kind='line', data=scores_df.query('metric_type=="magnitude"'), col_order=['FB0', 'FB250', 'FB500', 'FBMock'], hue_order=['FB0', 'FB250', 'FB500', 'FBMock'], palette=['#3CB4E8', '#438BA8', '#002A3B', '#FE4A49'], aspect=0.8, height=3)
g.axes[0, 0].set_ylabel('Magnitude, $\mu V$')
[ax.set_title(s) for ax, s in zip(g.axes.flatten(), ['FB0', 'FB250', 'FB500', 'FBMock'])]
[ax.set_xlabel('Block number') for ax in g.axes.flatten()]