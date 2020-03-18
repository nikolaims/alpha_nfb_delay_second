import pandas as pd
import numpy as np
import scipy.signal as sg
from scipy import linalg
import pylab as plt
from mne.cov import _regularized_covariance
from mne.decoding import SPoC
from  mne.viz import plot_topomap

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
        X = (X ** 2).mean(axis=-1)

        # To standardize features
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)

        self.evals = evals[ix]

        return self


from release.settings import CHANNELS, FS, FB_ALL, MONTAGE, Montage
from pynfb.signal_processing.decompositions import ICADecomposition
pos = Montage(CHANNELS).get_pos()
NO_P4_CH = [ch for ch in CHANNELS if ch!='P4']
pos_no_P4 = Montage(NO_P4_CH).get_pos()

eeg_path = 'release/data/eeg_allsubjs_eyefree_1_45hz_down250hz.pkl'
info_path = 'release/data/info_allsubjs.pkl'

eeg_df = pd.read_pickle(eeg_path)
info_df = pd.read_pickle(info_path)


topographies = []
components = []
targets = []
for subj_id in range(50):
    print(subj_id)
    eeg_df_subj = eeg_df[eeg_df['subj_id']==subj_id]
    band = info_df.loc[info_df['subj_id']==subj_id, 'band'].values[0]
    filter_n_taps = 250
    filter_b = sg.firwin2(filter_n_taps, [0, band[0], band[0], band[1], band[1], FS//2], [0, 0, 1, 1, 0, 0], fs=FS)

    x = eeg_df_subj.loc[eeg_df_subj['block_number'].isin(FB_ALL), CHANNELS].values
    x = sg.lfilter(filter_b, [1, 0], x, 0)[filter_n_taps:]
    x = x[:, np.array(CHANNELS)!='P4']
    y = x[:, CHANNELS.index('P4')]

    n_times = FS*10
    n_step = FS*1
    n_epochs = (len(x)-n_times)//n_step
    n_channels = len(CHANNELS) - 1
    spoc_X = np.zeros((n_epochs, n_channels, n_times))
    spoc_y = np.zeros(n_epochs)
    for k in range(0, n_epochs):
        spoc_X[k] = x[k * n_step: k * n_step + n_times].T
        spoc_y[k] = y[k * n_step: k * n_step + n_times].var()


    spoc_y = (spoc_y - spoc_y.mean())/spoc_y.std()
    spoc = FixSPoC(len(CHANNELS)-1, reg=0.01)
    spoc.fit(spoc_X, spoc_y)

    topographies.append(spoc.patterns_.T)
    components.append(spoc.transform(spoc_X))
    targets.append(spoc_y)
    # filters.append(ica.filters)
#

min_dim = min([c.shape[0] for c in components])

np.save('experiments/spocs/data/spoc_fb_topos.npy', np.array(topographies))
np.save('experiments/spocs/data/spoc_fb_components.npy', np.array([c[:min_dim] for c in components]))
np.save('experiments/spocs/data/spoc_fb_targets.npy', np.array([c[:min_dim] for c in targets]))

