from release.settings import FB_ALL
import pandas as pd
import numpy as np
from release.stats.fan98test import fan98test, simulate_h0_distribution, get_p_val
import pylab as plt
import seaborn as sns
from mne.stats import fdr_correction

stats_file = 'block_stats_1channels_1bands_median_20ths.pkl'
stats_df_all = pd.read_pickle('release/data/{}'.format(stats_file))
# stats_df = stats_df.loc[stats_df.subj_id!=28]
stats_df_all = stats_df_all.loc[stats_df_all['block_number'].isin(FB_ALL)]
unique_blocks = list(stats_df_all['block_number'].unique())
stats_df_all['k'] = stats_df_all['block_number'].apply(lambda x: unique_blocks.index(x))

metric_type = 'amplitude'
unique_thresholds = stats_df_all['threshold_factor'].unique()
fb_types = ['FB0', 'FB250', 'FB500', 'FBMock'][::-1]

comps = []
for ind1 in range(len(fb_types)):
    for ind2 in range(len(fb_types) - 1, ind1, -1):
        comps.append(fb_types[ind1] + ' - ' + fb_types[ind2])

p_vals_all_metrics = {}
metric_types = ['magnitude', 'n_spindles', 'duration', 'amplitude']
for metric_type in metric_types:

    p_vals_all_th = []
    for th in unique_thresholds:
        stats_df = stats_df_all.query('threshold_factor=={} & metric_type=="{}"'.format(th, metric_type))
        fb_data_points = []
        for ind, fb_type in enumerate(fb_types):
            fb_stats_df = stats_df.query('fb_type=="{}"'.format(fb_type))
            unique_subj = fb_stats_df.subj_id.unique()
            data_points = np.zeros((len(unique_subj), len(FB_ALL)))
            for j, subj_id in enumerate(unique_subj):
                data_points[j, :] = fb_stats_df.query('subj_id=={}'.format(subj_id))['metric']
                # data_points[j, :] /= data_points[j, 0]
                data_points[j, :] /= data_points[j, :].mean()
            fb_data_points.append(data_points[:, :])

            # plt.errorbar(np.arange(15)+ind*0.05, data_points.mean(0), 2*data_points.std(0)/np.sqrt(len(unique_subj)))
            # plt.plot(np.arange(15), data_points.T, color='C{}'.format(ind), alpha=0.5)

        n_subjects_list = sorted([fb_data_points[k].shape[0] for k in range(len(fb_data_points))])
        d_list = np.arange(sum(n_subjects_list[:2]) - 2, sum(n_subjects_list[-2:]) - 2 + 1)
        h0_distributions_dict = {}
        for d in d_list:
            h0_distributions_dict[d] = simulate_h0_distribution(fb_data_points[0].shape[1], d, verbose=False)

        p_vals = []
        for comp in comps:
            ind1, ind2 = [fb_types.index(fb_type) for fb_type in comp.split(' - ')]
            stat = fan98test(fb_data_points[ind2], fb_data_points[ind1])
            d = fb_data_points[ind1].shape[0] + fb_data_points[ind2].shape[0] - 2
            p = get_p_val(stat, h0_distribution=h0_distributions_dict[d])
            p_vals.append(p)
        p_vals_all_th.append(p_vals)


    p = np.array(p_vals_all_th).T
    _, p_corrected = fdr_correction(p)
    p_vals_all_metrics[metric_type] = p_corrected


sns.set_style("dark")
fig = plt.figure(constrained_layout=True, figsize=(6, 2.7))
axes = []
gs = fig.add_gridspec(1, 7)
axes.append(fig.add_subplot(gs[0]))
for k in range(3):
    axes.append(fig.add_subplot(gs[k*2+1:k*2+3]))

# p_corrected = p
xticklabels0 = [th if th % 1. == 0 else '' for th in unique_thresholds]
# fig, axes = plt.subplots(1, len(metric_types))
for ax, metric_type in zip(axes, metric_types):
    p_corrected = p_vals_all_metrics[metric_type]

    if metric_type == 'magnitude':
        p_corrected = p_corrected[:, :1]
        xticklabels = []
        yticklabels = comps
    else:
        xticklabels = xticklabels0
        yticklabels = ['']*len(comps)

    cbar_ax = fig.add_axes([0.85, 0.2, 0.05, 0.6]) if metric_type == metric_types[-1] else None
    ax = sns.heatmap(np.log10(p_corrected), mask=p_corrected>0.05, yticklabels=yticklabels, xticklabels=xticklabels,
                     cmap='Blues_r', ax=ax, cbar=metric_type == metric_types[-1], vmax=0, vmin=-4, linewidths=0.5,
                     cbar_ax=cbar_ax, cbar_kws={'ticks': np.log10([0.05, 0.01, 0.001, 0.0001])})
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90 )
    ax.set_xlabel('th. factor' if metric_type != 'magnitude' else '')
    ax.set_ylim(len(comps), 0)
    ax.set_title(metric_type if metric_type != 'magnitude' else 'mag.')

    # plt.tight_layout()
cbar_ax.set_yticklabels([0.05, 0.01, 0.001, 0.0001])
cbar_ax.set_title('p-value\n(FDR)')
plt.subplots_adjust(left = 0.2, right=0.8, bottom=0.2, top=0.8)