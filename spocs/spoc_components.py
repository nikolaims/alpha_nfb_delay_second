import pandas as pd
import numpy as np
import scipy.signal as sg
import sys
import pylab as plt
import h5py
from mne.cov import _regularized_covariance
from mne.viz import plot_topomap
from mne import create_info

# import nfb lab data loader
from scipy import linalg

sys.path.insert(0, '/home/kolai/Projects/nfblab/nfb')
from utils.load_results import load_data
from pynfb.inlets.montage import Montage
from pynfb.signal_processing.decompositions import CSPDecomposition
from mne.decoding import SPoC

PLOT_ARTIFACTS_RES = False
FLANKER_WIDTH = 2
FS = 500
GFP_THRESHOLD = 100e-6

# band hilbert helper
def band_hilbert(x, fs, band, N=None, axis=-1):
    x = np.asarray(x)
    Xf = np.fft.fft(x, N, axis=axis)
    w = np.fft.fftfreq(x.shape[0], d=1. / fs)
    Xf[(w < band[0]) | (w > band[1])] = 0
    x = np.fft.ifft(Xf, axis=axis)
    return 2*x


class SPoCFix(SPoC):
    def fit(self, X, y):
        """Estimate the SPoC decomposition on epochs.
        Parameters
        ----------
        X : ndarray, shape (n_epochs, n_channels, n_times)
            The data on which to estimate the SPoC.
        y : array, shape (n_epochs,)
            The class for each epoch.
        Returns
        -------
        self : instance of SPoC
            Returns the modified instance.
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be of type ndarray (got %s)."
                             % type(X))
        self._check_Xy(X, y)

        if len(np.unique(y)) < 2:
            raise ValueError("y must have at least two distinct values.")

        # The following code is direclty copied from pyRiemann

        # Normalize target variable
        target = y.astype(np.float64)
        target -= target.mean()
        target /= target.std()

        n_epochs, n_channels = X.shape[:2]

        # Estimate single trial covariance
        covs = np.empty((n_epochs, n_channels, n_channels))
        for ii, epoch in enumerate(X):
            covs[ii] = _regularized_covariance(
                epoch, reg=self.reg, method_params=self.cov_method_params)

        C = covs.mean(0)
        Cz = np.mean(covs * target[:, np.newaxis, np.newaxis], axis=0)

        # solve eigenvalue decomposition
        evals, evecs = linalg.eigh(Cz, C)
        evals = evals.real
        evecs = evecs.real
        # sort vectors
        ix = np.argsort(evals)[::-1]
        print(evals[ix])

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

        return self

# collect info
data_path = '/home/kolai/Data/alpha_delay2'
info = pd.read_csv('alpha_subject_2_full.csv')
datasets = [d for d in info['dataset'].unique() if (d is not np.nan)
            and (info.query('dataset=="{}"'.format(d))['type'].values[0] in ['FB0', 'FBMock', 'FB250', 'FB500'])][:]
stats_df = pd.read_csv('spindles_stats_norm.csv')
for FB in ['FB0', 'FB250', 'FB500', 'FBMock']:
    # store data
    subj_bands = {}

    x = []
    y = []
    for j_dataset, dataset in enumerate(datasets[:]):
        dataset_path = '{}/{}/experiment_data.h5'.format(data_path, dataset)

        # load fb signal params
        with h5py.File(dataset_path) as f:
            eye_rejection_matrix = f['protocol10/signals_stats/Alpha0/rejections/rejection1'].value
            subj_bands[dataset] = f['protocol10/signals_stats/Alpha0/bandpass'].value

        # load data
        df, fs, channels, p_names = load_data(dataset_path)

        # drop pauses
        df = df.loc[df['block_name'].isin(['Baseline0', 'Close', 'Baseline', 'FB0', 'FB250', 'FB500', 'FBMock'])]

        # get FB type
        fb_type = df.query('block_number==6')['block_name'].values[0]
        if fb_type != FB: continue
        print(dataset)

        # rename FB blocks to "FB"
        df['block_name'] = df['block_name'].apply(lambda x: 'FB' if 'FB' in x else x)

        # remove eyes artifacts ICA
        df[channels] = df[channels].values.dot(eye_rejection_matrix)

        # GFP threshold arthifact segments
        th = np.abs(df[channels[:-1]]).rolling(int(fs), center=True).max().mean(1)
        df = df.loc[th<GFP_THRESHOLD]

        # filter data


        # estimate snr

        montage = Montage(channels[:-1])
        b_numbers = df.query('block_name=="FB"')['block_number'].unique()
        baseline_b = [2, 4]
        band = subj_bands[dataset]
        ba = sg.butter(4, [band[0]/fs*2, band[1]/fs*2], 'band')
        xs = [sg.filtfilt(*ba, df.query('block_number=={}'.format(b))[channels[:-1]].values[:int(fs)*100], axis=0).T for b in b_numbers]
        x += xs


        metric = stats_df.query('dataset=="{}" & metric_type=="magnitude" & threshold_factor==2'.format(dataset))['metric'].values
        score = metric[8:].mean() / metric[1:8].mean()
        y += metric.tolist()
        # xs = [sg.filtfilt(*ba, df.query('block_number=={}'.format(b))[channels[:-1]].values[:int(fs)*100], axis=0).T for b in b_numbers]
        # x += [xx for xx in (np.array(xs) - np.mean(xs))/np.std(xs)]
        # # y += [sg.filtfilt(*ba, df.query('block_number=={}'.format(b))['P4'].values[:int(fs)*100]).std() for b in b_numbers]
        # # ys = [df.query('block_number=={}'.format(b))['signal_Alpha0'].values[:int(fs) * 100].mean() for b in
        # #       b_numbers]
        # y += [n for n in range(1, 16)]
        #y += sg.filtfilt(np.ones(4)/4, [1.], ys)

    # SPoC
    print('spoc')
    spoc = SPoCFix(n_components=32)
    spoc.fit(np.array(x), np.array(y))
    a = spoc.transform(np.array(x))
    corrs = [np.corrcoef(a[:, k], np.array(y))[0,1] for k in range(32)]
    ix = np.argsort(corrs)[::-1][[0, 1, 2, 3, -4, -3, -2, -1]]



    from mne.channels import read_montage
    fig = spoc.plot_patterns(create_info(channels[:-1], sfreq=fs, ch_types='eeg', montage=read_montage('standard_1005')), components=ix)#, vmin=-2, vmax=2)


    [ax.set_title('{}_{}'.format(FB, k)) for k, ax in enumerate(fig.axes[:-1])]
    [ax.set_xlabel('{:.3f}'.format(corrs[k])) for k, ax in zip(ix, fig.axes)]
    plt.savefig('spocs/{}_topo_fb.png'.format(FB))
    plt.close('all')



    #y = (np.array(y)
    plt.plot(y, 'k', alpha=0.8)
    [plt.plot(a[:, ix[k]]*np.std(y)+np.mean(y), alpha=0.8, zorder=-k+100, linestyle='--' if k>=len(ix)//2 else '-') for k in range(len(ix))]
    plt.legend(['signal_Alpha0'] + ['{}_{}'.format(FB, k) for k in range(len(ix))] )

    plt.xticks(np.arange(0, 150, 15))
    plt.gca().set_xticklabels(['s{}'.format(k+1) for k in range(10)])
    plt.savefig('spocs/{}_target_fb.png'.format(FB))
    plt.close('all')
        #plt.subplots_adjust(left=2)

        #plt.figure()
        # plt.plot(x.dot(spoc.filters[:, 1]))
        # plt.plot(df['P4'])
        # [plot_topomap(csp.topographies[:, j], montage.get_pos(), axes=axes[j_dataset, j], show=False, contours=0, mask=np.array(channels[:-1])=='P4') for j in range(32)]
        # [axes[j_dataset, j].set_title('{:.1f}'.format(csp.scores[j])) for j in range(32)]
        # axes[j_dataset, 0].set_ylabel('{}\n{:.1f}'.format(fb_type, snr))


