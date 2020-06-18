import pandas as pd
import numpy as np
import scipy.signal as sg
import pylab as plt
from scipy.linalg import eigh
from mne.decoding import SPoC
from mne.cov import _regularized_covariance
from  mne.viz import plot_topomap
from scipy import linalg

from release.settings import CHANNELS_SEL, FS, FB_ALL, MONTAGE,  Montage
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

eeg_path = 'release/data/FBLow_eeg_allsubjs_eyefree_1_45hz_down250hz.pkl'
info_path = 'release/data/info_allsubjs.pkl'

eeg_df = pd.read_pickle(eeg_path)
eeg_df = eeg_df[eeg_df['block_number'].isin(FB_ALL)]
info_df = pd.read_pickle(info_path)

pos = Montage(CHANNELS_SEL).get_pos()


for subj_id in eeg_df.subj_id.unique():
    fb_type = info_df[info_df['subj_id']==subj_id]['fb_type'].values[0]
    eeg_df_subj = eeg_df[eeg_df['subj_id'] == subj_id]
    band = info_df.loc[info_df['subj_id'] == subj_id, 'band'].values[0]
    filter_n_taps = 1000
    filter_all = sg.firwin2(filter_n_taps, [0, band[0]-2, band[0]-2, band[1]+2, band[1]+2, FS // 2], [0, 0, 1, 1, 0, 0], fs=FS)
    filter_band = sg.firwin2(filter_n_taps, [0, band[0], band[0], band[1], band[1], FS // 2], [0, 0, 1, 1, 0, 0], fs=FS)

    x = eeg_df_subj[CHANNELS_SEL].values*1e6
    block_numbers = eeg_df_subj['block_number'].values
    x = sg.filtfilt(filter_all, [1, 0], x, 0)
    y = x[:, CHANNELS_SEL.index('P4')]

    x_band = sg.filtfilt(filter_band, [1, 0], x, 0)
    y_band = x_band[:, CHANNELS_SEL.index('P4')]

    x_flankers = x - x_band
    y_flankers = y - y_band

    n_blocks = len(FB_ALL)
    n_channels = len(CHANNELS_SEL)
    X_band = []
    X_flankers = []
    for block_number in FB_ALL:
        X_band.append(x_band[block_numbers==block_number].T)
        X_flankers.append(x_flankers[block_numbers == block_number].T)

    ssd_patterns, ssd_filters, ssd_evals = ssd(X_band, X_flankers)
    # fig, axes = plt.subplots(4, 8)
    # for k in range(32):
    #     ax = axes[k//8, k%8]
    #     plot_topomap(ssd_patterns[:, k], pos, axes=ax)
    #     ax.set_xlabel('{:.2f}'.format(ssd_evals[k]))
    transform = ssd_patterns[:, ssd_evals > 1].dot(ssd_filters[ssd_evals > 1])
    np.savez('release/data/ssd_filters/s{}.npz'.format(subj_id),
             filters=ssd_filters, patterns=ssd_patterns, evals=ssd_evals)

    print(subj_id)
    eeg_df.loc[eeg_df['subj_id'] == subj_id, CHANNELS_SEL] = x_band.dot(transform.T)
    # plt.plot(eeg_df.loc[eeg_df['subj_id'] == subj_id, 'P4'])
    # plt.show(block=True)

    p4 = np.array([x.var(1) for x in X_band])[:, CHANNELS_SEL.index('P4')]
    p4_ssdcsp = np.array([transform.dot(x).var(1) for x in X_band])[:, CHANNELS_SEL.index('P4')]
    # p4_ssd = np.array([x.var(1) for x in X_alpha_ssd])[:, CHANNELS.index('P4')]
    # p4 = np.array([x.var(1) for x in X_band])[:, CHANNELS.index('P4')]
    # p4 = np.array([x.var(1) for x in X_alpha_csp])[:, CHANNELS.index('P4')
    # p4 = np.array([x.var(1) for x in X_alpha_csp])[:, CHANNELS.index('P4')
    # comps = np.array([x.var(1) for x in X_alpha_csp])
    topo = ssd_patterns[:, ssd_evals>1]
    n_comp = min(10, sum(ssd_evals>1))
    fig3 = plt.figure(constrained_layout=True, figsize=(12, 7))
    gs = fig3.add_gridspec(3, n_comp)
    main_ax = fig3.add_subplot(gs[:2, :])

    main_ax.plot(p4)#, label='P4 {:.2f}'.format(p4[segment_number >= n_blocks // 2].mean() / p4[segment_number < n_blocks // 2].mean()))

    main_ax.plot(p4_ssdcsp)#, label='P4 ssd csp {:.2f}'.format(p4_ssdcsp[segment_number >= n_blocks // 2].mean() / p4_ssdcsp[segment_number < n_blocks // 2].mean()))
    for k in range(n_comp):
        # main_ax.plot(comps[::1, k], color='C{}'.format(k), label='C{}'.format(k+1), alpha=0.9, zorder=-k*10-100)

        ax = fig3.add_subplot(gs[2, k])
        ax.plot([-0.5, 0.5], [1,1], color='C{}'.format(k), linewidth=3, alpha=0.9)
        plot_topomap(topo[:, k], pos, axes=ax)
        str_form = '{:.1f}'
        ax.set_xlabel(str_form.format(ssd_evals[k]))
        ax.set_title('C{}'.format(k+1))
    main_ax.legend()
    plt.subplots_adjust(hspace=0.5)
    plt.suptitle('{} {}'.format(fb_type, subj_id))
    plt.savefig('experiments/spocs/res/ssd/{}ssd{}_s{}.png'.format(n_comp, fb_type, subj_id), dpi=200)
    # plt.show(block=True)
    plt.close()

eeg_df.to_pickle('release/data/SSD_FBLow_eeg_allsubjs_eyefree_1_45hz_down250hz.pkl')