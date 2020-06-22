from release.settings import FB_ALL
import pandas as pd
import numpy as np
from release.stats.fan98test import eval_z_score, simulate_h0_distribution, get_p_val_one_tailed, adaptive_neyman_test, legendre_transform, identity_transform, corrcoef_test, get_p_val_two_tailed
import pylab as plt
import seaborn as sns
from mne.stats import fdr_correction

# STAT_FUN = corrcoef_test
# TRANSFORM_FUN = identity_transform
# P_VAL_FUN = get_p_val_two_tailed

STAT_FUN = adaptive_neyman_test
TRANSFORM_FUN = legendre_transform
P_VAL_FUN = get_p_val_one_tailed

stats_file = 'block_stats_1channels_1bands_median_20ths.pkl'
stats_df_all = pd.read_pickle('release/data/{}'.format(stats_file))
# stats_df = stats_df.loc[stats_df.subj_id!=28]
stats_df_all = stats_df_all.loc[stats_df_all['block_number'].isin(FB_ALL)]
unique_blocks = list(stats_df_all['block_number'].unique())
stats_df_all['k'] = stats_df_all['block_number'].apply(lambda x: unique_blocks.index(x))

metric_type = 'amplitude'
unique_thresholds = stats_df_all['threshold_factor'].unique()
fb_types = ['FB0', 'FB250', 'FB500', 'FBMock'][::-1]
stats_df_all = stats_df_all.loc[stats_df_all['fb_type'].isin(fb_types)]

n_subjects_list = sorted([group['subj_id'].nunique() for name, group in stats_df_all.groupby('fb_type')])
d_list = np.arange(sum(n_subjects_list[:2]) - 2, sum(n_subjects_list[-2:]) - 2 + 1)
h0_distributions_dict = {}
for d in d_list:
    h0_distributions_dict[d] = simulate_h0_distribution(n=len(unique_blocks), d=d,
                                                        transform=TRANSFORM_FUN, stat_fun=STAT_FUN, verbose=True)


comps = []
for ind1 in range(len(fb_types)):
    for ind2 in range(len(fb_types) - 1, ind1, -1):
        comps.append(fb_types[ind2] + ' - ' + fb_types[ind1])

stats_all_metrics = {}
p_vals_all_metrics = {}
z_scores_all_metrics = {}
metric_types = ['magnitude', 'n_spindles', 'duration', 'amplitude']
for metric_type in metric_types:

    stats_all_th = []
    z_scores_all_th = []
    for th in (unique_thresholds if metric_type != 'magnitude' else unique_thresholds[:1]):
        stats_df = stats_df_all.query('threshold_factor=={} & metric_type=="{}"'.format(th, metric_type))
        fb_data_points = []
        for ind, fb_type in enumerate(fb_types):
            fb_stats_df = stats_df.query('fb_type=="{}"'.format(fb_type))
            unique_subj = fb_stats_df.subj_id.unique()
            data_points = np.zeros((len(unique_subj), len(FB_ALL)))
            for j, subj_id in enumerate(unique_subj):
                data_points[j, :] = fb_stats_df.query('subj_id=={}'.format(subj_id))['metric']
                data_points[j, :] /= data_points[j, :].mean()
            fb_data_points.append(data_points[:, :])
        stats = []
        z_scores = []
        ds = []
        for comp in comps:
            ind1, ind2 = [fb_types.index(fb_type) for fb_type in comp.split(' - ')]
            z_score, d = eval_z_score(fb_data_points[ind1], fb_data_points[ind2])
            stat = STAT_FUN(TRANSFORM_FUN(z_score), d)
            # p = P_VAL_FUN(stat, h0_distribution=h0_distributions_dict[d])
            stats.append(stat)
            z_scores.append(z_score)
            ds.append(d)
        stats_all_th.append(stats)
        z_scores_all_th.append(z_scores)


    stats_all_th = np.array(stats_all_th).T
    get_p_val_vec = np.vectorize(lambda x: P_VAL_FUN(x, h0_distributions_dict[d]))
    p = get_p_val_vec(stats_all_th)
    _, p_corrected = fdr_correction(p)
    stats_all_metrics[metric_type] = stats_all_th
    p_vals_all_metrics[metric_type] = p_corrected
    z_scores_all_metrics[metric_type] = np.array(z_scores_all_th)




sns.set_style("dark")
fig = plt.figure(constrained_layout=True, figsize=(6, 2.5))
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


from scipy.stats import t as tdist
fig, axes = plt.subplots(len(comps), len(metric_types), figsize=(6, 4))

for j_metric_type, metric_type in enumerate(metric_types):
    p_corrected = p_vals_all_metrics[metric_type]
    z_score = z_scores_all_metrics[metric_type]
    min_indexes = np.argmin(p_corrected, 1)
    axes[0, j_metric_type].set_title(metric_type if metric_type != 'magnitude' else 'mag.')

    for j_comp, comp in enumerate(comps):
        ax = axes[j_comp, j_metric_type]
        if p_corrected[j_comp,min_indexes[j_comp]] < 0.05:
            ax.plot(np.arange(len(unique_blocks))+1, z_score[min_indexes[j_comp], j_comp], '.-', markersize=2, linewidth=0.5)
        else:
            ax.plot(np.nan)
        ax.set_ylim(-5, 5)
        ax.set_xlim(0.5, 15.5)
        if j_comp < len(comps)-1:
            ax.set_xticks([])
        else:
            ax.set_xlabel('block')


        ax.axhline(0, color='k', linewidth=0.5, zorder=-100, alpha=0.5)
        ax.axhline(tdist.ppf(0.975, ds[j_comp]), color='k', linewidth=0.5, zorder=-100, alpha=0.5, linestyle='--')
        ax.axhline(tdist.ppf(0.025, ds[j_comp]), color='k', linewidth=0.5, zorder=-100, alpha=0.5, linestyle='--')

        if j_metric_type != 0:
            ax.set_yticks([])
        else:
            # ax.yaxis.tick_right()
            ax.set_yticks([0])
            ax.set_yticklabels([comp])
        # if j_metric_type == 0 :
        #     ax.set_ylabel(comp, rotation=0, labelpad=30, size=8)
plt.subplots_adjust(left = 0.2, right=0.8, bottom=0.2, top=0.8, hspace=0.01, wspace=0.07)