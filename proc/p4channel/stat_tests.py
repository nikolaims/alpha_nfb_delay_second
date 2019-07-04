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

stats_df = pd.read_pickle('data/channels1_bands1_splitedTrue_thresholds17_globthrFalse.pkl').query('channel=="P4" & band=="alpha"')
stats_df = stats_df.loc[stats_df['block_number']> 1000]
#stats_df.fillna(0)


factor_names = ['subj_id', 'fb_type', 'metric_type', 'threshold_factor']
scores_df = pd.DataFrame(columns=factor_names + ['score'])
#courses_df = pd.DataFrame(columns=factor_names + ['time_course', 'block_number'])

for factors_values, group in stats_df.groupby(factor_names):
    metric = group['metric'].values
    curve = metric
    #curve = pd.Series(metric).rolling(3, center=True).mean().fillna(method='ffill').fillna(method='bfill')
    score = np.mean(curve[15:])/np.mean(curve[:15])
    #score = metric[-1]/metric[2]
    scores_df = scores_df.append(dict(zip(factor_names + ['score'], list(factors_values)+[score])), ignore_index=True)

    #courses_df = courses_df.append(pd.DataFrame(dict(zip(factor_names + ['time_course',  'block_number'], list(factors_values) + [metric[:]/metric[:8].mean(), np.arange(metric.shape[0])+1]))), ignore_index=True)



#scores_df['fb_gain'] = (scores_df['second'] - scores_df['first'])/scores_df['first']

#scores_df = scores_df.fillna(0)

factor_names = ['fb_type', 'metric_type', 'threshold_factor']
p_values_df = pd.DataFrame(columns=factor_names + ['runksum', 'comparison', 'pvalue'])
pvalue_types = ['FBMock', 'FB500', 'FB250', 'FB0'][::-1]
for factors_values, group in scores_df.groupby(factor_names):
    for pvalue_type in ['FBMock', 'FB500', 'FB250', 'FB0']:
        if factors_values[0] == pvalue_type: continue
        mock = scores_df.query('fb_type=="{}" & metric_type=="{}" & threshold_factor=={}'.format(pvalue_type, *factors_values[1:]))
        p_value_mock = ranksums(group['score'].values, mock['score'].values)
        pvalue = stats.ranksums(group['score'].values, mock['score'].values).pvalue
        print('*' if stats.shapiro(group['score'].values)[1]<0.05 else '-', factors_values)

        #p_value_0 = stats.ranksums(group['score'].values, 0)[0]

        pvalue_dict = dict(zip(factor_names + ['runksum', 'comparison', 'pvalue'], list(factors_values) + [[p_value_mock], [pvalue_type], [pvalue]]))
        #print(pvalue_dict)
        p_values_df = p_values_df.append(pd.DataFrame(pvalue_dict), ignore_index=True)

#sns.set(rc={'figure.figsize':(2,2)})
#sns.set(font_scale=2)
g = sns.relplot('threshold_factor', 'runksum', 'fb_type', col='metric_type', data=p_values_df, kind='line',
                row='comparison', col_order=['magnitude', 'n_spindles', 'duration', 'amplitude'], row_order=pvalue_types, height=2.5, palette=['#3CB4E8', '#438BA8', '#002A3B', '#FE4A49'])
#g.axes[0][0].semilogy()
#g.fig.set_size_inches(10,10)
[[ax.axhline(p, color='k', linestyle='--') for ax in g.axes.flatten()] for p in [-1.65, 1.65]]
for color, axes in zip(['#3CB4E8', '#438BA8', '#002A3B', '#FE4A49'], g.axes):
   [ax.axhspan(-1.65, 1.65, color=color, alpha=0.2) for ax in axes.flatten()]
#plt.tight_layout()

[ax.set_title('') for ax in  g.axes.flatten()]
[ax.set_title(title) for title, ax in zip(['A. average_power', 'B. n_spindles', 'C. duration', 'D. amplitude'], g.axes[0])]

[ax.set_ylabel(title) for title, ax in zip(list(map(lambda x: x+'\nstatistic', pvalue_types)), g.axes[:, 0])]
# g = sns.catplot('fb_type', 'score', 'fb_type', col='metric_type', data=scores_df, kind='box', sharey=False)
# [[ax.axhline(p, color='k', linestyle='--') for ax in g.axes[0]] for p in [0]]
#
# stats_df['logsnr'] = np.log10(stats_df['snr'])
# sns.relplot('block_number', 'time_course', 'fb_type',data=courses_df.query('threshold_factor==1.5'), col='fb_type', kind='line', ci=95,
#             estimator=np.median, col_order=['FB0', 'FB250', 'FB500', 'FBMock'], hue_order=['FB0', 'FB250', 'FB500', 'FBMock'], row='metric_type', facet_kws={'sharey': 'row'}, row_order=['magnitude', 'n_spindles', 'duration', 'amplitude'])

#plt.savefig('nfb_metric_tests_fdr_ranksum.png', dpi=200)

from mne.stats import fdr_correction
# pval = p_values_df.query('comparison=="FBMock" & metric_type=="n_spindles"')['pvalue']
# T = p_values_df.query('comparison=="FBMock" & metric_type=="n_spindles"')['runksum']
# reject_fdr, pval_fdr = fdr_correction(pval,method='indep')
# threshold_fdr = np.min(np.abs(T)[reject_fdr])
# threshold_fdr

th_factors = stats_df['threshold_factor'].unique()
for comparison, axes in zip(pvalue_types, g.axes):
    for metric_type, ax in zip(['magnitude', 'n_spindles', 'duration', 'amplitude'], axes):
        for fb_type, color in zip(pvalue_types, ['#3CB4E8', '#438BA8', '#002A3B', '#FE4A49']):
            pval = p_values_df.query('comparison=="{}" & fb_type=="{}" & metric_type=="{}"'.format(comparison, fb_type, metric_type))['pvalue'].values
            T = p_values_df.query(
                'comparison=="{}" & fb_type=="{}" & metric_type=="{}"'.format(comparison, fb_type, metric_type))[
                'runksum'].values
            reject_fdr, pval_fdr = fdr_correction(pval, alpha=0.10, method='indep')
            if len(T[reject_fdr]) > 0:
                ax.plot(th_factors[reject_fdr], T[reject_fdr], '*', color=color)

cm = dict(zip(['FB0', 'FB250', 'FB500', 'FBMock'], ['#3CB4E8', '#438BA8', '#002A3B', '#FE4A49']))
fig, axes = plt.subplots(1, 3, sharex='all', sharey='all', figsize=(7,4))
plt.subplots_adjust(wspace=0.1)
for j_metric_name, metric_name in enumerate(['n_spindles',  'amplitude', 'duration']):
    for fb_type in ['FB0', 'FB250', 'FB500'][::-1]:
        axes[j_metric_name].plot(p_values_df.query('fb_type=="{}" & metric_type=="{}" & comparison=="FBMock"'.format(fb_type, metric_name))['runksum'], th_factors, color=cm[fb_type], alpha=0.9)
        pval = p_values_df.query(
            'comparison=="FBMock" & fb_type=="{}" & metric_type=="{}"'.format(fb_type, metric_name))[
            'pvalue'].values
        T = p_values_df.query(
            'comparison=="FBMock" & fb_type=="{}" & metric_type=="{}"'.format(fb_type, metric_name))[
            'runksum'].values
        reject_fdr, pval_fdr = fdr_correction(pval, alpha=0.10, method='indep')
        if len(T[reject_fdr]) > 0:
            axes[j_metric_name].plot(T[reject_fdr], th_factors[reject_fdr], '*', color=cm[fb_type], markersize=7, alpha=0.9)

[ax.axvline(105, color=cm['FBMock'], zorder=-100) for ax in axes.flatten()]
[ax.axvline(128, color=cm['FBMock'], linestyle='--', zorder=-100) for ax in axes.flatten()]
[ax.set_xlabel('Rank-Sum') for ax in axes.flatten()]
[ax.set_title(n) for ax, n in zip(axes.flatten(), ['n_spindles',  'amplitude', 'duration'])]
axes[0].set_ylabel('Threshold factor $\delta$')
axes[0].set_yticks([1, 1.5, 2, 2.5, 3])
[ax.spines['right'].set_visible(False)for ax in axes.flatten()]
[ax.spines['top'].set_visible(False)for ax in axes.flatten()]

plt.savefig('mock_vs_rest_ths_NAT.png', dpi=300)