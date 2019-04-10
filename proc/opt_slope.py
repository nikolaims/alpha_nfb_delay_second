from scipy.optimize import minimize, LinearConstraint
from scipy.stats import linregress
import numpy as np
import pylab as plt
from tqdm import tqdm
from proc.settings import FB_ALL
from proc.settings import CHANNELS, MONTAGE
from mne.viz import plot_topomap


def get_slope(y, max_x, ini_x):
    k = np.arange(15)
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

    bounds = LinearConstraint(np.eye(3), [0,0,-1/len(k)], max_x)
    res = minimize(fun, ini_x, jac=jac, hess=hess, method='trust-constr', constraints=bounds)
    return res.x[2]


import pandas as pd


all_stats_df = pd.read_pickle('data/metrics_chs_ica_all.pkl')
all_stats_df = all_stats_df.loc[all_stats_df['block_number'].isin(FB_ALL)]
all_stats_df = all_stats_df.loc[all_stats_df['channel'].isin(CHANNELS)]
all_stats_df = all_stats_df.loc[all_stats_df['threshold_factor'].isin([2])]
all_stats_df = all_stats_df.loc[all_stats_df['metric_type'].isin(['magnitude'])]


slopes = np.zeros((4, 10, 32))
for f, (fb_type, fb_type_df) in enumerate(all_stats_df.groupby('fb_type')):
    for s, (subj_id, subj_df) in tqdm(enumerate(fb_type_df.groupby('subj_id')), fb_type):
        for c, (ch, ch_df) in enumerate(subj_df.groupby('channel')):
            curve = ch_df['metric'].values
            curve[np.isinf(curve)] = np.nan
            curve = pd.Series(curve).fillna(method='bfill').values
            slopes[f, s, c] = get_slope(curve, [np.max(curve), np.max(curve),  1/14], [np.mean(curve)/2, np.mean(curve)/2, 0])
            #slopes[f, s, c] = linregress(np.arange(15), curve).slope



vmax = 10
fig, axes = plt.subplots(4, 11)
pers = 0.5/14
for f in range(4):
    for s in range(10):
        plot_topomap(slopes[f, s], MONTAGE.get_pos(), mask=slopes[f, s]>pers, axes=axes[f, s], cmap='viridis', vmin=0, vmax=np.percentile(slopes, 90), contours=0)
    plot_topomap(np.mean(slopes[f], 0), MONTAGE.get_pos(), mask=np.mean(slopes[f], 0)>pers, axes=axes[f, -1], cmap='viridis', vmin=0, vmax=np.percentile(slopes, 90), contours=0)