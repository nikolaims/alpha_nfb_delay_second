import pandas as pd
from scipy import stats
import seaborn as sns
import numpy as np
from scipy.stats import rankdata
import pylab as plt
from proc.settings import FB_ALL
from proc.settings import CHANNELS, MONTAGE
from mne.viz import plot_topomap
from tqdm import tqdm
import scipy.signal as sg
from itertools import combinations

poses = MONTAGE.get_pos()
def ranksums(x, y):
    x, y = map(np.asarray, (x, y))
    n1 = len(x)
    n2 = len(y)
    alldata = np.concatenate((x, y))
    ranked = rankdata(alldata)
    x = ranked[:n1]
    s = np.sum(x, axis=0)
    return s

all_stats_df = pd.read_pickle('data/metrics_chs_ica_all.pkl')
all_stats_df = all_stats_df.loc[all_stats_df['block_number'].isin(FB_ALL)]
all_stats_df = all_stats_df.loc[all_stats_df['channel'].isin(CHANNELS)]
all_stats_df = all_stats_df.loc[all_stats_df['threshold_factor'].isin([2])]

pairs = [('FB0', 'FBMock'),('FB250', 'FBMock'),('FB500', 'FBMock'),('FB0', 'FB250'),('FB250', 'FB500'),('FB0', 'FB500')]

metric_types = ['magnitude', 'n_spindles', 'duration', 'amplitude']

all_curves = np.zeros((len(metric_types), len(pairs), 2, 10, 32, 15))
for j_metric_type, metric_type in enumerate(metric_types):
    stats_df = all_stats_df.query('metric_type == "{}"'.format(metric_type))
    #stats_df.fillna(0)
    for j_pair, pair in tqdm(enumerate(pairs), metric_type):
        pair_scores_df = stats_df.loc[stats_df['fb_type'].isin(pair)]
        curves = np.zeros((2, 10, 32, 15))
        for g, fb_type in enumerate(pair):
            group_scores_df = pair_scores_df.query('fb_type == "{}"'.format(fb_type))
            for c, (ch, channel_scores_df) in enumerate(group_scores_df.groupby('channel')):
                for s, (subj_id, group) in enumerate(channel_scores_df.groupby('subj_id')):
                    curve = group['metric'].values
                    curve = curve / np.nanmedian(curve[:8])
                    curves[g, s, c, :] = curve
        all_curves[j_metric_type, j_pair] = curves


combinations = list(combinations(np.arange(20), 10))[:11000]
statistic = np.zeros((len(combinations), len(metric_types), len(pairs), 32))
statistic0 = np.zeros((len(metric_types), len(pairs), 32))
for n, combination in tqdm(enumerate(combinations), 'permutation'):
    curves1 = all_curves.copy()
    perm = np.concatenate([combination, np.arange(20)[~np.isin(np.arange(20), combination)]])
    all_curves_perm = np.concatenate([all_curves[:,:,0], all_curves[:,:,1]], axis=2)[:, :, perm]
    curves1[:, :, 0], curves1[:, :, 1] = all_curves_perm[:, :, :10], all_curves_perm[:, :, 10:]
    st = np.nanmean(np.nanmedian(curves1[:, :, 0], axis=2) - np.nanmedian(curves1[:, :, 1], axis=2), axis=-1)
    #st = np.median(curves1[0], axis=0)[:, 8:].mean(1)/np.median(curves1[0], axis=0)[:, 1:8].mean(1) - np.median(curves1[1], axis=0)[:, 8:].mean(1)/np.median(curves1[0], axis=0)[:, 1:8].mean(1)
    statistic[n] = st

#np.save('data/permutations_fb.npy', statistic)

fig, axes = plt.subplots(len(metric_types), len(pairs), figsize=(12, 4))
for j_metric_type, metric_type in enumerate(metric_types):
    for j_pair, pair in enumerate(pairs):
        st = statistic[:, j_metric_type, j_pair]
        p_vals_h = np.sum(st > st[0], axis=0)/len(combinations)
        mask_h = p_vals_h < 0.05
        p_vals_l = np.sum(st < st[0], axis=0) / len(combinations)
        mask_l = p_vals_l < 0.05
        topo = p_vals_h*mask_h.astype(int) + mask_l.astype(int)*(1 - p_vals_l) + 0.5 *(~(mask_h | mask_l))

        plot_topomap(topo, MONTAGE.get_pos(), vmin=-0.5, vmax=1.5, cmap='RdBu', mask=mask_h | mask_l, contours=0, show=False,
                     mask_params=dict(marker='*', markerfacecolor="None", markeredgecolor='k', linewidth=0, markersize=10),
                     axes=axes[j_metric_type, j_pair], image_interp='lanczos')
        axes[0, j_pair].set_title('{} > {}'.format(*pair))

    axes[j_metric_type, 0].set_ylabel('{}'.format(metric_type))

