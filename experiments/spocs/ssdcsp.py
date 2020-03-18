import pandas as pd
import numpy as np
import scipy.signal as sg
import pylab as plt
from scipy.linalg import eigh
from mne.decoding import SPoC
from mne.cov import _regularized_covariance
from  mne.viz import plot_topomap
from scipy import linalg

from release.settings import CHANNELS, FS, FB_ALL, MONTAGE,  Montage
from pynfb.signal_processing.decompositions import ICADecomposition

REG=0.001
def decompose(C1, C2, reg):
    evals, evecs = linalg.eigh(C1, C2+np.eye(len(C2))*reg/np.trace(C2))
    evals = evals.real
    evecs = evecs.real
    # sort vectors
    ix = np.argsort(evals)[::-1]

    # sort eigenvectors
    evecs = evecs[:, ix]
    evals = evals[ix]

    # spatial patterns
    patterns = linalg.pinv(evecs).T  # n_channels x n_channels
    filters = evecs.T  # n_channels x n_channels
    return patterns, filters, evals

def csp(X_first, X_second, reg=REG):
    C_first = np.mean([xx.dot(xx.T) for xx in X_first], 0)
    C_second = np.mean([xx.dot(xx.T) for xx in X_second], 0)
    return decompose(C_second, C_first, reg)

def ssd(X_band, X_flankers, reg=REG):
    C_band = np.mean([xx.dot(xx.T) for xx in X_band], 0)
    C_flankers = np.mean([xx.dot(xx.T) for xx in X_flankers], 0)
    return decompose(C_band, C_flankers, reg)

eeg_path = 'release/data/eeg_allsubjs_eyefree_1_45hz_down250hz.pkl'
info_path = 'release/data/info_allsubjs.pkl'

eeg_df = pd.read_pickle(eeg_path)
info_df = pd.read_pickle(info_path)

LINEAR_TREND = 1
DEL_P4 = False
pos = Montage(CHANNELS).get_pos() if not DEL_P4 else Montage([ch for ch in CHANNELS if ch!='P4']).get_pos()

for subj_id in range(40):
    fb_type = info_df[info_df['subj_id']==subj_id]['fb_type'].values[0]

    eeg_df_subj = eeg_df[eeg_df['subj_id'] == subj_id]
    band = info_df.loc[info_df['subj_id'] == subj_id, 'band'].values[0]
    filter_n_taps = 1000
    filter_all = sg.firwin2(filter_n_taps, [0, band[0]-2, band[0]-2, band[1]+2, band[1]+2, FS // 2], [0, 0, 1, 1, 0, 0], fs=FS)
    filter_band = sg.firwin2(filter_n_taps, [0, band[0], band[0], band[1], band[1], FS // 2], [0, 0, 1, 1, 0, 0], fs=FS)

    x = eeg_df_subj.loc[eeg_df_subj['block_number'].isin(FB_ALL), CHANNELS].values*1e6
    x = sg.filtfilt(filter_all, [1, 0], x, 0)[filter_n_taps:]
    y = x[:, CHANNELS.index('P4')]

    x_band = sg.filtfilt(filter_band, [1, 0], x, 0)
    y_band = x_band[:, CHANNELS.index('P4')]

    x_flankers = x - x_band
    y_flankers = y - y_band

    # plt.plot(y_band)
    # plt.plot(y_flankers)

    n_times = FS * 20
    n_step = FS * 5
    n_epochs = (len(x) - n_times) // n_step
    n_channels = len(CHANNELS)
    X_band = np.zeros((n_epochs, n_channels, n_times))
    X_flankers = np.zeros((n_epochs, n_channels, n_times))
    for k in range(0, n_epochs):
        X_band[k] = x_band[k * n_step: k * n_step + n_times].T
        X_flankers[k] = x_flankers[k * n_step: k * n_step + n_times].T

    segment_number = np.arange(n_epochs)
    iqr_factor = 1.5
    for k in range(1):
        spoc_X_var = (X_band+X_flankers).var(2)
        mask = (spoc_X_var<(iqr_factor+1)*np.percentile(spoc_X_var, 75, 0)- iqr_factor*np.percentile(spoc_X_var, 25, 0)).all(1)
        X_band = X_band[mask]
        X_flankers = X_flankers[mask]
        segment_number = segment_number[mask]
        print(sum(~mask), len(mask), sum(~mask)/len(mask))

    ssd_patterns, ssd_filters, ssd_evals = ssd(X_band, X_flankers)
    # fig, axes = plt.subplots(4, 8)
    # for k in range(32):
    #     ax = axes[k//8, k%8]
    #     plot_topomap(ssd_patterns[:, k], pos, axes=ax)
    #     ax.set_xlabel('{:.2f}'.format(ssd_evals[k]))


    X_alpha = np.array([ssd_filters[ssd_evals>1].dot(x) for x in X_band])
    # plt.plot(X_alpha.var(2))
    # segment_number = np.arange(n_epochs)
    iqr_factor = 1.5
    for k in range(1):
        spoc_X_var = X_alpha.var(2)
        mask = (spoc_X_var < (iqr_factor + 1) * np.percentile(spoc_X_var, 75, 0) - iqr_factor * np.percentile(
            spoc_X_var, 25, 0)).all(1)
        X_band = X_band[mask]
        X_flankers = X_flankers[mask]
        X_alpha = X_alpha[mask]
        segment_number = segment_number[mask]
        print(sum(~mask), len(mask), sum(~mask) / len(mask))
    print(len(segment_number)/n_epochs)


    csp_patterns, csp_filters, csp_evals = csp(X_alpha[segment_number<n_epochs//2], X_alpha[segment_number>=n_epochs//2])

    X_alpha_csp = np.array([csp_filters.dot(x) for x in X_alpha])
    # plt.plot(X_alpha_csp.var(2))

    patterns = ssd_patterns[:, ssd_evals>1].dot(csp_patterns)



    # fig, axes = plt.subplots(1, 10)
    # for k in range(10):
    #     ax = axes[k]
    #     plot_topomap(patterns[:, k], pos, axes=ax)
    #     ax.set_xlabel('{:.2f}'.format(csp_evals[k]))

    comps = X_alpha_csp.var(2)
    topo = patterns
    n_comp = min(10, sum(ssd_evals>1))
    fig3 = plt.figure(constrained_layout=True, figsize=(12, 7))
    gs = fig3.add_gridspec(3, n_comp)
    main_ax = fig3.add_subplot(gs[:2, :])
    for k in range(n_comp):
        main_ax.plot(comps[::1, k], color='C{}'.format(k), label='C{}'.format(k+1), alpha=0.9, zorder=-k*10-100)

        ax = fig3.add_subplot(gs[2, k])
        ax.plot([-0.5, 0.5], [1,1], color='C{}'.format(k), linewidth=3, alpha=0.9)
        plot_topomap(topo[:, k], pos, axes=ax)
        str_form = '{:.1f}'
        ax.set_xlabel(str_form.format(csp_evals[k]))
        ax.set_title('C{}'.format(k+1))
    if n_comp < 10: main_ax.legend()
    plt.subplots_adjust(hspace=0.5)
    plt.suptitle('{} {}'.format(fb_type, subj_id))
    plt.savefig('experiments/spocs/res/ssdcsp/{}ssdcsp{}_s{}.png'.format(n_comp, fb_type, subj_id), dpi=200)
    plt.close()