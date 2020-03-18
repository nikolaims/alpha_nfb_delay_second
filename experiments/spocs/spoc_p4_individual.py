import pandas as pd
import numpy as np
import scipy.signal as sg
import pylab as plt

from mne.decoding import SPoC
from mne.cov import _regularized_covariance
from  mne.viz import plot_topomap
from scipy import linalg

from release.settings import CHANNELS, FS, FB_ALL, MONTAGE,  Montage
from pynfb.signal_processing.decompositions import ICADecomposition


class FixSPoC(SPoC):
    def fit(self, X, y):
        target = y.astype(np.float64)
        target -= target.mean()
        target /= target.std()

        n_epochs, n_channels = X.shape[:2]

        # Estimate single trial covariance
        covs = np.empty((n_epochs, n_channels, n_channels))
        for ii, epoch in enumerate(X):
            covs[ii] = _regularized_covariance(
                epoch, reg=self.reg, method_params=self.cov_method_params,
                rank=self.rank)

        C = covs.mean(0)
        Cz = np.mean(covs * target[:, np.newaxis, np.newaxis], axis=0)

        # solve eigenvalue decomposition
        evals, evecs = linalg.eigh(Cz, C)
        evals = evals.real
        evecs = evecs.real
        # sort vectors
        ix = np.argsort(evals)[::-1]

        # sort eigenvectors
        evecs = evecs[:, ix].T

        # spatial patterns
        self.patterns_ = linalg.pinv(evecs).T  # n_channels x n_channels
        self.filters_ = evecs  # n_channels x n_channels

        pick_filters = self.filters_[:self.n_components]
        X = np.asarray([np.dot(pick_filters, epoch) for epoch in X])

        # compute features (mean band power)
        X = X.var(axis=2)

        # To standardize features
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)

        self.evals = evals[ix]

        return self

    def transform(self, X):
        X = np.asarray([np.dot(self.filters_, epoch) for epoch in X])
        X = X.var(axis=2)
        X -= self.mean_
        X /= self.std_
        return X

eeg_path = 'release/data/eeg_allsubjs_eyefree_1_45hz_down250hz.pkl'
info_path = 'release/data/info_allsubjs.pkl'

eeg_df = pd.read_pickle(eeg_path)
info_df = pd.read_pickle(info_path)

LINEAR_TREND = False
DEL_P4 = True
pos = Montage(CHANNELS).get_pos() if not DEL_P4 else Montage([ch for ch in CHANNELS if ch!='P4']).get_pos()


for subj_id in range(40):
    fb_type = info_df[info_df['subj_id']==subj_id]['fb_type'].values[0]

    eeg_df_subj = eeg_df[eeg_df['subj_id'] == subj_id]
    band = info_df.loc[info_df['subj_id'] == subj_id, 'band'].values[0]
    filter_n_taps = 250
    filter_b = sg.firwin2(filter_n_taps, [0, band[0], band[0], band[1], band[1], FS // 2], [0, 0, 1, 1, 0, 0], fs=FS)

    x = eeg_df_subj.loc[eeg_df_subj['block_number'].isin(FB_ALL), CHANNELS].values
    x = sg.lfilter(filter_b, [1, 0], x, 0)[filter_n_taps:]
    y = x[:, CHANNELS.index('P4')]
    if DEL_P4: x = x[:, np.array(CHANNELS)!='P4']

    n_times = FS * 30
    n_step = FS * 1
    n_epochs = (len(x) - n_times) // n_step
    n_channels = len(CHANNELS) - int(DEL_P4)
    spoc_X = np.zeros((n_epochs, n_channels, n_times))
    spoc_y = np.zeros(n_epochs)
    for k in range(0, n_epochs):
        spoc_X[k] = x[k * n_step: k * n_step + n_times].T
        spoc_y[k] = y[k * n_step: k * n_step + n_times].var()
    if LINEAR_TREND: spoc_y = np.arange(len(spoc_y))

    for k in range(2):
        spoc_X_var = spoc_X.var(2)
        mask = (spoc_X_var<2.5*np.percentile(spoc_X_var, 75, 0)- 1.5*np.percentile(spoc_X_var, 25, 0)).all(1)
        spoc_X = spoc_X[mask]
        spoc_y = spoc_y[mask]


    spoc_y = (spoc_y - spoc_y.mean()) / spoc_y.std()
    spoc = FixSPoC(len(CHANNELS)- int(DEL_P4), reg=0.00001)
    spoc.fit(spoc_X, spoc_y)

    topo = spoc.patterns_.T
    comps = spoc.transform(spoc_X)
    target = spoc_y


    for n_comp in [5, 32-int(DEL_P4)]:
        fig3 = plt.figure(constrained_layout=True, figsize=(12, 7))
        gs = fig3.add_gridspec(3, n_comp)
        main_ax = fig3.add_subplot(gs[:2, :])
        for k in range(n_comp):
            main_ax.plot(comps[::4, k], color='C{}'.format(k), label='C{}'.format(k+1), alpha=0.9, zorder=-k*10-100)

            ax = fig3.add_subplot(gs[2, k])
            ax.plot([-0.5, 0.5], [1,1], color='C{}'.format(k), linewidth=3, alpha=0.9)
            plot_topomap(topo[:, k], pos, axes=ax)
            str_form = '{:.1f}\n{:.1f}\n{:.1f}\n{:.1f}' if k>0 else 'spoc={:.1f}\n cov={:.1f}\ncorr={:.1f}\n  r2={:.1f}'
            ax.set_xlabel(str_form.format(spoc.evals[k],
                                          np.cov(target, comps[:, k])[1,0],
                                          np.corrcoef(target, comps[:, k])[1,0],
                                          1-np.sum((target-comps[:, k])**2)/np.sum(target**2)))
            ax.set_title('C{}'.format(k+1))

        main_ax.plot(target[::4],  '--', color='k', label=r'P4 $\alpha$-power', alpha=0.9)
        if n_comp < 10: main_ax.legend()
        plt.subplots_adjust(hspace=0.5)
        plt.suptitle('{} {}'.format(fb_type, subj_id))
        plt.savefig('experiments/spocs/res/{}spoc{}_s{}.png'.format(n_comp, fb_type, subj_id), dpi=200)
        plt.close()