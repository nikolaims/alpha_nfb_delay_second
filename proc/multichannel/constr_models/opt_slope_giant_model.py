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
    C = 32
    S = 10
    K = 30
    k_axis = 0
    s_axis = 1
    c_axis = 2
    def fun(x):
        h_s, c_c, s_s, a = x[:S], x[S:S+C], x[S+C:2*S+C], x[-1]
        y_hat = np.zeros((K, S, C))
        for c in range(C):
            for s in range(S):
                y_hat[:, s, c] = h_s[s] + c_c[c]*s_s[s]*(1 + a*k)
        return np.sum((y_hat - y)**2)

    def jac(x):
        h_s, c_c, s_s, a = x[:S], x[S:S+C], x[S+C:2*S+C], x[-1]
        y_hat = np.zeros((K, S, C))
        for c in range(C):
            for s in range(S):
                y_hat[:, s, c] = h_s[s] + c_c[c] * s_s[s] * (1 + a * k)
        err = y_hat - y
        d_h_s = 2*np.sum(err, axis=(c_axis, k_axis))
        d_c_c = 2*np.array([np.sum([err[:, s, c] * s_s[s] * (1 + a * k) for s in range(S)]) for c in range(C)])
        d_s_s = 2*np.array([np.sum([err[:, s, c] * c_c[c] * (1 + a * k) for c in range(C)]) for s in range(S)])
        d_a = 2*np.sum([[err[:, s, s]*c_c[c]*s_s[s]*k for s in range(S)] for c in range(C)])
        return np.concatenate([d_h_s, d_c_c, d_s_s, [d_a]])

    def hess(x):
        h_s, c_c, s_s, a = x[:S], x[S:S+C], x[S+C:2*S+C], x[-1]
        y_hat = np.zeros((K, S, C))
        for c in range(C):
            for s in range(S):
                y_hat[:, s, c] = h_s[s] + c_c[c] * s_s[s] * (1 + a * k)
        err = y_hat - y

        H = np.zeros([len(x), len(x)])
        for s in range(S):
            H[s, s] = K*C
        for c in range(C):
            H[S+c, S+c] = np.sum([s_s[s]**2*(1+a*k)**2 for s in range(S)])
        for s in range(S):
            H[S+C+s, S+C+s] = np.sum([c_c[c] ** 2 * (1 + a * k) ** 2 for c in range(C)])
        H[-1, -1] = np.sum([[c_c[c]**2 * s_s[s]**2 * k**2 for s in range(S)] for c in range(C)])

        for s in range(S):
            for c in range(C):
                H[s, S+c] = 2 * np.sum(s_s[s]*(1+a*k))

            for s1 in range(S):
                if s == s1:
                    H[s, S + C + s1] = 2 * np.sum(s_s[s] * (1 + a * k))

            H[s, -1] = 2 * np.sum([c_c[c]*s_s[s]*(1+a*k) for c in range(C)])

        for c in range(C):
            for s in range(S):
                H[S+c, S+C+s] = 2*np.sum(err[:, s, c]*(1+a*k) + c_c[c]*(1+a*k)**2)
            H[S+c, -1] = 2*np.sum([err[:, s, c]*s_s[s]*k + c_c[c]*s_s[s]**2*(1+a*k)*k for s in range(S)])

        for s in range(S):
            H[S+C+s, -1] = 2*np.sum([err[:, s, c]*c_c[c]*k + c_c[c]**2*s_s[s]*(1+a*k)*k for c in range(C)])

        H = H.T + H
        return H

    const_matrix = np.eye(S+C+S+1)
    lb = np.zeros(len(const_matrix))
    lb[-1] = -1/(K-1)
    rb = np.ones(len(const_matrix)) * np.inf
    rb[S:S+C] = 1
    rb[:S] = np.max(np.max(y, 2), 0)
    rb[S+C:S+C+S] = np.max(np.max(y, 2), 0)
    rb[-1] = 3/(K-1)
    lin_const = LinearConstraint(const_matrix, lb, rb)

    x_0 = np.concatenate([np.random.normal(0,1)* np.min(np.min(y, 0), 1), np.random.uniform(0, 1, C), np.random.normal(0,1)* np.min(np.min(y, 0), 1), [0]])
    res = minimize(fun, x_0, jac=jac, hess=hess, method='trust-constr', constraints=lin_const)
    return res


import pandas as pd



fb_type = 'FB0'

all_stats_df = pd.read_pickle('data/split_metrics_chs_ica_all.pkl')
all_stats_df = all_stats_df.loc[all_stats_df['block_number']>1000]
all_stats_df = all_stats_df.loc[all_stats_df['channel'].isin(CHANNELS)]
all_stats_df = all_stats_df.loc[all_stats_df['threshold_factor'].isin([2])]
all_stats_df = all_stats_df.loc[all_stats_df['metric_type'].isin(['magnitude'])]
all_stats_df = all_stats_df.loc[all_stats_df['fb_type'].isin([fb_type])]


y = np.zeros((30, 10, 32))
for s, (subj_id, subj_df) in tqdm(enumerate(all_stats_df.groupby('subj_id'))):
    for c, (ch, ch_df) in enumerate(subj_df.groupby('channel')):
        curve = ch_df['metric'].values
        curve[np.isinf(curve)] = np.nan
        curve = pd.Series(curve).fillna(method='bfill').rolling(3).median().fillna(method='bfill').rolling(3).mean().fillna(method='bfill').values
        #curve = sg.filtfilt(np.arange(3)/3, [1], curve)
        y[:, s, c] = curve



topos = []
funs = []
slopes = []

for r in tqdm(range(1)):
    res = get_slope(y[:, np.random.randint(0, 10, 10)])
    print(res.fun, res.x[-1]*29*100)
    topos.append(res.x[10:10+32])
    funs.append(res.fun)
    slopes.append(res.x[-1])


plt.figure()
wean = lambda x, a: np.sum([topo*1/fun for topo, fun in zip(x, funs)], 0)/np.sum(1/np.array(funs), a)

indx = [CHANNELS.index(chn) for c, chn in enumerate(all_stats_df['channel'].unique())]


plot_topomap(wean(topos, 0), MONTAGE.get_pos()[indx], vmin=0, vmax=1)
plt.title('{}\n{} +- {}'.format(fb_type, int(wean(slopes, 0)*29*100), int(wean((np.array(slopes) - wean(slopes, 0))**2, 0)**0.5*29*100)))
plt.tight_layout()



fig, axes = plt.subplots(11, sharey=True, sharex=True)
for c in np.arange(32):
    for s in range(10):
        axes[s].plot(np.arange(30)/30 + c, (y[:, s, c] - res.x[s])/res.x[10+c]/res.x[10+32+s]-1, 'b')
        axes[s].plot(np.arange(30)/30 + c, np.arange(30)*res.x[-1], 'r')

axes[0].set_ylim(-1, 1)
axes[-1].set_xticks(np.arange(32)+0.5)
axes[-1].set_xticklabels(CHANNELS)
plt.show()
pass
