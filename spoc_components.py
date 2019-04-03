import pandas as pd
import numpy as np
import scipy.signal as sg
import sys
import pylab as plt
import h5py
from mne.viz import plot_topomap
from mne import create_info

# import nfb lab data loader
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


# collect info
data_path = '/home/kolai/Data/alpha_delay2'
info = pd.read_csv('alpha_subject_2_full.csv')
datasets = [d for d in info['dataset'].unique() if (d is not np.nan)
            and (info.query('dataset=="{}"'.format(d))['type'].values[0] in ['FB0', 'FBMock', 'FB250', 'FB500'])][:]

FB = 'FBMock'
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
    band = subj_bands[dataset]
    ba = sg.butter(4, [band[0]/fs*2, band[1]/fs*2], 'band')
    # x = np.array([sg.filtfilt(*ba, df.query('block_number=={}'.format(b))[channels[:-1]].values[:int(fs)*100], axis=0).T for b in b_numbers])
    # y = np.array([sg.filtfilt(*ba, df.query('block_number=={}'.format(b))['P4'].values[:int(fs)*100]).var() for b in enumerate(b_numbers)])
    x += [sg.filtfilt(*ba, df.query('block_number=={}'.format(b))[channels[:-1]].values[:int(fs)*100], axis=0).T for b in b_numbers]
    # y += [sg.filtfilt(*ba, df.query('block_number=={}'.format(b))['P4'].values[:int(fs)*100]).std() for b in b_numbers]
    y += [df.query('block_number=={}'.format(b))['signal_Alpha0'.format(FB[2:])].values[:int(fs) * 100].mean() for b in
          b_numbers]
    #y += sg.filtfilt(np.ones(4)/4, [1.], ys)

# SPoC
spoc = SPoC(n_components=4)
spoc.fit(np.array(x), np.array(y))



from mne.channels import read_montage
fig = spoc.plot_patterns(create_info(channels[:-1], sfreq=fs, ch_types='eeg', montage=read_montage('standard_1005')), vmin=-2, vmax=2)
fig.axes[0].set_ylabel(FB)
plt.savefig('spocs/{}_stim.png'.format(FB))
plt.close('all')

a = spoc.transform(np.array(x))
plt.plot(a)
plt.plot((np.array(y) - np.mean(y))/np.std(y), 'k')
plt.legend(['spoc1', 'spoc2', 'spoc3', 'spoc4', 'signal_Alpha0'])

plt.xticks(np.arange(0, 150, 15))
plt.gca().set_xticklabels(['s{}'.format(k+1) for k in range(10)])
plt.savefig('spocs/{}_stim_dyn.png'.format(FB))
plt.close('all')
    #plt.subplots_adjust(left=2)

    #plt.figure()
    # plt.plot(x.dot(spoc.filters[:, 1]))
    # plt.plot(df['P4'])
    # [plot_topomap(csp.topographies[:, j], montage.get_pos(), axes=axes[j_dataset, j], show=False, contours=0, mask=np.array(channels[:-1])=='P4') for j in range(32)]
    # [axes[j_dataset, j].set_title('{:.1f}'.format(csp.scores[j])) for j in range(32)]
    # axes[j_dataset, 0].set_ylabel('{}\n{:.1f}'.format(fb_type, snr))


