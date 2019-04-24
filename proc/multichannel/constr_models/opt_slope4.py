from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
from scipy.stats import linregress, ttest_1samp, ttest_ind
import numpy as np
import pylab as plt
from tqdm import tqdm
from proc.settings import FB_ALL
from proc.settings import CHANNELS, MONTAGE
from mne.viz import plot_topomap
import scipy.signal as sg
import numpy as np
from scipy.optimize import BFGS


def get_slope(y):
    k = np.arange(30)
    K = len(k)
    def fun(x):
        eta, beta, a, b = x
        return np.sum((eta + beta*(1 + a*k + b*k**2) - y)**2)

    def jac(x):
        eta, beta, a, b = x
        err = (eta + beta*(1 + a*k + b*k**2) - y)
        d_eta = np.sum(2*err)
        d_beta = np.sum(2 * err * (1 + a*k + b*k**2))
        d_a = np.sum(2 * err * beta * k)
        d_b = np.sum(2 * err * beta * k**2)
        return np.array([d_eta, d_beta, d_a, d_b])

    def hess(x):
        eta, beta, a, b = x
        pol = (1 + a*k + b*k**2)
        err = (eta + beta * pol - y)
        d_eta2 = 2*K
        d_beta2 = np.sum(2*pol**2)
        d_a2 = np.sum(2 * (beta * k) ** 2)
        d_b2 = np.sum(2 * (beta * k**2) ** 2)
        d_eta_beta = np.sum(2*pol)
        d_eta_a = np.sum(2 * beta * k)
        d_eta_b = np.sum(2 * beta * k**2)
        d_beta_a = 2*np.sum(k*beta*pol + err*k)
        d_beta_b = 2 * np.sum(k**2 * beta * pol + err * k**2)
        d_a_b = 2 * np.sum(beta**2*k**3)
        return np.array([[d_eta2, d_eta_beta, d_eta_a, d_eta_b], [d_eta_beta, d_beta2, d_beta_a, d_beta_b],
                         [d_eta_a, d_beta_a, d_a2, d_a_b], [d_eta_b, d_beta_b, d_a_b, d_b2]])

    def const(x):
        eta, beta, a, b = x
        return [[eta + beta * (1 + a * kk + b * kk**2 )] for kk in k]

    def jac_const(x):
        eta, beta, a, b = x
        jjac = np.zeros((len(k), 4))
        for kk in k:
            d_eta = 1
            d_beta = 1 + a * kk + b * kk ** 2
            d_a = beta * kk
            d_b = beta * kk ** 2
            jjac[kk] = np.array([d_eta, d_beta, d_a, d_b])
        return jjac

    def hess_const(x, v):
        eta, beta, a, b = x
        hess = np.zeros((len(k), 4, 4))
        for kk in k:
            d_eta2 = 0
            d_beta2 = 0
            d_a2 = 0
            d_b2 = 0
            d_eta_beta = 0
            d_eta_a = 0
            d_eta_b = 0
            d_beta_a = kk
            d_beta_b = kk**2
            d_a_b = 0
            hess[kk] =  np.array([[d_eta2, d_eta_beta, d_eta_a, d_eta_b], [d_eta_beta, d_beta2, d_beta_a, d_beta_b],
                  [d_eta_a, d_beta_a, d_a2, d_a_b], [d_eta_b, d_beta_b, d_a_b, d_b2]])
        return np.sum([v[l]*hess[l] for l in k], 0)

    constr_matrix = np.zeros((3 + len(k)-1, 4))
    constr_matrix[0, 0] = 1
    constr_matrix[1, 1] = 1
    constr_matrix[2, :2] = 1
    for j_k, k_val in enumerate(k[1:]):
        constr_matrix[j_k+3, 2] = k_val
        constr_matrix[j_k + 3, 3] = k_val**2

    constr_left = np.zeros(3 + len(k)-1)
    constr_left[1] = np.min(y)/2
    constr_left[2] = np.min(y)
    constr_left[3:] = -1
    constr_right = np.zeros(3 + len(k) - 1) + np.max(y)
    constr_right[3:] = np.max(y)/np.min(y) - 1
    constr_right
    lin_const = LinearConstraint(constr_matrix, constr_left, constr_right)

    non_lin_const = NonlinearConstraint(const, np.min(y), np.max(y), jac_const, hess_const)

    res = minimize(fun, np.array([np.min(y)/2, np.min(y)/2, 0, 0]), jac=jac, hess=hess, method='trust-constr', constraints=[lin_const, non_lin_const])
    return res.x


import pandas as pd


all_stats_df = pd.read_pickle('data/split_metrics_chs_ica_all.pkl')
all_stats_df = all_stats_df.loc[all_stats_df['block_number']>1000]
all_stats_df = all_stats_df.loc[all_stats_df['channel'].isin(CHANNELS)]
all_stats_df = all_stats_df.loc[all_stats_df['threshold_factor'].isin([2])]
all_stats_df = all_stats_df.loc[all_stats_df['metric_type'].isin(['magnitude'])]


slopes = np.zeros((4, 10, 32))
curves = np.zeros((4, 10, 32, 30))
fig, axes = plt.subplots(4, 11, sharex=True, sharey=True)
for f, (fb_type, fb_type_df) in enumerate(all_stats_df.groupby('fb_type')):
    for s, (subj_id, subj_df) in tqdm(enumerate(fb_type_df.groupby('subj_id')), fb_type):
        for c, (ch, ch_df) in enumerate(subj_df.groupby('channel')):
            curve = ch_df['metric'].values
            curve[np.isinf(curve)] = np.nan
            curve = pd.Series(curve).fillna(method='bfill')#.rolling(3).median().fillna(method='bfill').values
            #curve = sg.filtfilt(np.arange(3)/3, [1], curve)
            eta, beta, a, b = get_slope(curve)
            slopes[f, s, c] = np.mean(a*np.arange(30)+b*np.arange(30)**2)
            curves[f, s, c] = np.arange(30)*a + np.arange(30)**2*b
            # if ch == 'P4':
            #     axes[f, s].plot(curve)
            #     axes[f, s].plot((np.arange(30)*slope+1)*beta+eta)
            #     axes[f, s].set_title('{:.2f} {:.2f} {:.2f}'.format(slope*29, eta, beta))
            #     lin_r = linregress(np.arange(30), curve)
            #     axes[f, s].plot(np.arange(30)*lin_r.slope + lin_r.intercept)
            if ch == 'P4':
                axes[f, s].plot((curve - eta - beta) / beta)
                axes[f, s].plot(np.arange(30)*a + np.arange(30)**2*b)
                axes[f, s].set_title('{:+}%\nnoise={:.2f}\nsignal={:.2f}'.format(int(np.mean(a*np.arange(30)+b*np.arange(30)**2)*100), eta, beta))
                #lin_r = linregress(np.arange(30), curve)
                #axes[f, s].plot(np.arange(30)*lin_r.slope + lin_r.intercept)
                if s == 9:
                    axes[0, s+1].set_title('MEAN')
                    axes[0, s+1].plot(np.mean(curves[f, :, c], 0))
                    axes[f, 0].set_ylabel(fb_type)
                    if f > 0:
                        axes[f, s + 1].plot(np.mean(curves[f, :, c], 0))



