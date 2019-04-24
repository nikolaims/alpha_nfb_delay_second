from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
from scipy.stats import linregress, ttest_1samp
import numpy as np
import pylab as plt
from tqdm import tqdm
from proc.settings import FB_ALL
from proc.settings import CHANNELS, MONTAGE
from mne.viz import plot_topomap
import scipy.signal as sg


def get_slope(y):
    k = np.arange(30)
    def fun(x):
        eta, beta, a = x
        return np.sum((eta + beta*(a*k+1) - y)**2)

    def jac(x):
        eta, beta, a = x
        d_eta = np.sum(2*(eta + beta*(a*k+1) - y))
        d_beta = np.sum(2 * (eta + beta * (a * k + 1) - y) * (a*k+1))
        d_a = np.sum(2 * (eta + beta * (a * k + 1) - y) * beta * k)
        return np.array([d_eta, d_beta, d_a])

    def hess(x):
        eta, beta, a = x
        d_eta2 = 2*(max(k)+1)
        d_beta2 = np.sum(2*(a*k+1)**2)
        d_a2 = np.sum(2 * (beta * k) ** 2)
        d_eta_beta = np.sum(2*(a*k+1))
        d_eta_a = np.sum(2 * beta * k)
        d_beta_a = np.sum(2*k*(eta+2*beta*(a*k+1) - y))
        return np.array([[d_eta2, d_eta_beta, d_eta_a], [d_eta_beta, d_beta2, d_beta_a], [d_eta_a, d_beta_a, d_a2]])

    def const(x):
        eta, beta, a = x
        return eta + beta*(a*max(k)+1)

    def jac_const(x):
        eta, beta, a = x
        d_eta = 1
        d_beta = a * max(k) + 1
        d_a = beta * max(k)
        return np.array([d_eta, d_beta, d_a])

    def hess_const(x, v):
        eta, beta, a = x
        d_eta2 = 0
        d_beta2 = 0
        d_a2 = 0
        d_eta_beta = 0
        d_eta_a = 0
        d_beta_a = max(k)
        return v[0]*np.array([[d_eta2, d_eta_beta, d_eta_a], [d_eta_beta, d_beta2, d_beta_a], [d_eta_a, d_beta_a, d_a2]])


    lin_const = LinearConstraint(np.vstack([np.eye(3), [1, 1, 0]]), [0, 0, -1/max(k), min(y)], [max(y), max(y), np.inf, max(y)])
    non_lin_const = NonlinearConstraint(const, min(y), max(y), jac_const, hess_const)
    res = minimize(fun, np.array([np.min(y)/2, np.min(y)/2, 0]), jac=jac, hess=hess, method='trust-constr', constraints=[lin_const, non_lin_const])
    return res.x


import pandas as pd


all_stats_df = pd.read_pickle('data/split_metrics_chs_ica_all.pkl')
all_stats_df = all_stats_df.loc[all_stats_df['block_number']>1000]
all_stats_df = all_stats_df.loc[all_stats_df['channel'].isin(CHANNELS)]
all_stats_df = all_stats_df.loc[all_stats_df['threshold_factor'].isin([2])]
all_stats_df = all_stats_df.loc[all_stats_df['metric_type'].isin(['n_spindles'])]


slopes = np.zeros((4, 10, 32))
curves = np.zeros((4, 10, 32, 30))
fig, axes = plt.subplots(4, 11, sharex=True, sharey=True)
for f, (fb_type, fb_type_df) in enumerate(all_stats_df.groupby('fb_type')):
    for s, (subj_id, subj_df) in tqdm(enumerate(fb_type_df.groupby('subj_id')), fb_type):
        for c, (ch, ch_df) in enumerate(subj_df.groupby('channel')):
            curve = ch_df['metric'].values
            curve[np.isinf(curve)] = np.nan
            curve = pd.Series(curve).fillna(method='bfill').rolling(3).median().fillna(method='bfill').rolling(3).mean().fillna(method='bfill').values
            #curve = sg.filtfilt(np.arange(3)/3, [1], curve)
            eta, beta, slope = get_slope(curve)
            slopes[f, s, c] = slope
            curves[f, s, c] = (curve - eta - beta) / beta
            # if ch == 'P4':
            #     axes[f, s].plot(curve)
            #     axes[f, s].plot((np.arange(30)*slope+1)*beta+eta)
            #     axes[f, s].set_title('{:.2f} {:.2f} {:.2f}'.format(slope*29, eta, beta))
            #     lin_r = linregress(np.arange(30), curve)
            #     axes[f, s].plot(np.arange(30)*lin_r.slope + lin_r.intercept)
            if ch == 'P4':
                axes[f, s].plot((curve - eta - beta) / beta)
                axes[f, s].plot(np.arange(30)*slope)
                axes[f, s].set_title('{:.2f} {:.2f} {:.2f}'.format(slope*29, eta, beta))
                #lin_r = linregress(np.arange(30), curve)
                #axes[f, s].plot(np.arange(30)*lin_r.slope + lin_r.intercept)
                if s == 9:
                    axes[0, s+1].plot(np.median(curves[f, :, c], 0))
                    if f > 0:
                        axes[f, s + 1].plot(np.median(curves[f, :, c], 0))





vmax = 10
fig, axes = plt.subplots(4, 11)
pers = 0.5/29*10000
for f in range(4):
    for s in range(10):
        plot_topomap(slopes[f, s], MONTAGE.get_pos(), mask=slopes[f, s]>pers, axes=axes[f, s], cmap='viridis', vmin=0, vmax=np.percentile(slopes, 90), contours=0)
    plot_topomap(np.median(slopes[f], 0), MONTAGE.get_pos(), mask=np.median(slopes[f], 0)>pers, axes=axes[f, -1], cmap='viridis', vmin=0, vmax=np.percentile(slopes, 90), contours=0)




vmax = 10
fig, axes = plt.subplots(4, 11)
pers = 0.5/29*10000
for f in range(4):
    for s in range(10):
        plot_topomap(curves[f, s].mean(1), MONTAGE.get_pos(), axes=axes[f, s], cmap='viridis', vmin=0, vmax=0.5, contours=0)

    pvals = np.array([ttest_1samp(np.mean(curves[f, :, c], 1), 0) for c in range(32)])
    plot_topomap(np.mean(curves[f], 0).mean(1), MONTAGE.get_pos(),  axes=axes[f, -1], mask=pvals<0.01, cmap='viridis', vmin=0, vmax=0.5, contours=0)




import itertools
pairs = [('FB0', 'FBMock'),('FB250', 'FBMock'),('FB500', 'FBMock'),('FB0', 'FB250'),('FB250', 'FB500'),('FB0', 'FB500')]
pairs_ind = [[0, 3], [1, 3], [2, 3], [0, 1], [1, 2], [0, 2]]
combinations = list(itertools.combinations(np.arange(20), 10))
statistic = np.zeros((len(pairs_ind), len(combinations), 32))
statistic0 = np.zeros((len(pairs_ind), 32))
for n, combination in tqdm(enumerate(combinations), 'permutation'):
    for j_pair, pair in enumerate(pairs):
        slopes1 = slopes[pairs_ind[j_pair]].copy()
        perm = np.concatenate([combination, np.arange(20)[~np.isin(np.arange(20), combination)]])
        slopes_perm = np.concatenate([slopes1[0], slopes1[1]], axis=0)[perm]
        st = np.median(slopes_perm[:10], 0) - np.median(slopes_perm[10:], 0)
        statistic[j_pair, n] = st




fig, axes = plt.subplots(1, len(pairs), figsize=(12, 4))
for j_pair, pair in enumerate(pairs):
    st = statistic[j_pair]
    p_vals_h = np.sum(st > st[0], axis=0)/len(combinations)
    mask_h = p_vals_h < 0.05
    p_vals_l = np.sum(st < st[0], axis=0) / len(combinations)
    mask_l = p_vals_l < 0.05
    topo = p_vals_h*mask_h.astype(int) + mask_l.astype(int)*(1 - p_vals_l) + 0.5 *(~(mask_h | mask_l))

    plot_topomap(topo, MONTAGE.get_pos(), vmin=-0.5, vmax=1.5, cmap='RdBu', mask=mask_h | mask_l, contours=0, show=False,
                 mask_params=dict(marker='*', markerfacecolor="None", markeredgecolor='k', linewidth=0, markersize=10), axes=axes[j_pair])
    axes[j_pair].set_title('{} > {}'.format(*pair))












import itertools
pairs = [('FB0', 'FBMock'),('FB250', 'FBMock'),('FB500', 'FBMock'),('FB0', 'FB250'),('FB250', 'FB500'),('FB0', 'FB500')]
pairs_ind = [[0, 3], [1, 3], [2, 3], [0, 1], [1, 2], [0, 2]]
combinations = list(itertools.combinations(np.arange(20), 10))
statistic = np.zeros((len(pairs_ind), len(combinations), 32))
statistic0 = np.zeros((len(pairs_ind), 32))
for n, combination in tqdm(enumerate(combinations), 'permutation'):
    for j_pair, pair in enumerate(pairs):
        slopes1 = curves[pairs_ind[j_pair]].copy()
        perm = np.concatenate([combination, np.arange(20)[~np.isin(np.arange(20), combination)]])
        slopes_perm = np.concatenate([slopes1[0], slopes1[1]], axis=0)[perm]
        st = np.mean(np.mean(slopes_perm[:10], 0) - np.mean(slopes_perm[10:], 0), axis=1)
        statistic[j_pair, n] = st




fig, axes = plt.subplots(1, len(pairs), figsize=(12, 4))
for j_pair, pair in enumerate(pairs):
    st = statistic[j_pair]
    p_vals_h = np.sum(st > st[0], axis=0)/len(combinations)
    mask_h = p_vals_h < 0.001
    p_vals_l = np.sum(st < st[0], axis=0) / len(combinations)
    mask_l = p_vals_l < 0.001
    topo = p_vals_h*mask_h.astype(int) + mask_l.astype(int)*(1 - p_vals_l) + 0.5 *(~(mask_h | mask_l))

    plot_topomap(topo, MONTAGE.get_pos(), vmin=-0.5, vmax=1.5, cmap='RdBu', mask=mask_h | mask_l, contours=0, show=False,
                 mask_params=dict(marker='*', markerfacecolor="None", markeredgecolor='k', linewidth=0, markersize=10), axes=axes[j_pair])
    axes[j_pair].set_title('{} > {}'.format(*pair))