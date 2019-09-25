import pandas as pd
import numpy as np
import scipy.signal as sg
import sys
import pylab as plt
import h5py
from mne.viz import plot_topomap
from tqdm import tqdm_gui

# import nfb lab data loader

def band_hilbert(x, fs, band, N=None, axis=-1):
    x = np.asarray(x)
    Xf = np.fft.fft(x, N, axis=axis)
    w = np.fft.fftfreq(x.shape[0], d=1. / fs)
    Xf[(w < band[0]) | (w > band[1])] = 0
    x = np.fft.ifft(Xf, axis=axis)
    return 2*x

sys.path.insert(0, '/home/kolai/Projects/nfblab/nfb')
from utils.load_results import load_data
data_path = '/home/kolai/Data/alpha_delay2'
dataset = 'alpha2-delay-subj-1_11-06_17-15-29'
dataset_path = '{}/{}/experiment_data.h5'.format(data_path, dataset)
df, fs, channels, p_names = load_data(dataset_path)
nor = lambda x: (x - x.mean())/x.std()
data = df.query('block_number==20')[['P4', 'PHOTO']].values
env = nor(np.abs(band_hilbert(data[:,0], 500, [8, 12])))
t = np.arange(len(env))/500
n = 20
photo = nor(sg.filtfilt([1/n]*n, [1],data[:,1]))

corrs = [np.corrcoef(np.roll(env, k), photo)[0,1] for k in range(500)]
opt_t = np.arange(500)[np.argmax(corrs)]/500*1000

fig, axes = plt.subplots(2, figsize=(5,4))
axes[1].plot(np.arange(500)/500*1000, corrs, color='k')
axes[1].plot(opt_t, corrs[np.argmax(corrs)], 'o', color='#FE4A49')
axes[1].axvline(opt_t, linestyle='--', color='k', alpha=0.5, zorder=-11)
axes[1].set_xlabel('Time, ms')
axes[1].set_ylabel('Corr.')
axes[1].set_xticks([0, opt_t, 500, 1000])


axes[0].plot(t-98, photo, color=(0, 176/255, 240/255), label='photosensor')
axes[0].plot(t-98, env, color='#FE4A49', label='offline env.')
axes[0].set_xlim([0, 10])
axes[0].set_xlabel('Time, s')
axes[0].set_ylabel('Envelope')
axes[0].legend(loc='upper right')

[(ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False)) for ax in axes]
plt.subplots_adjust(hspace=0.4)
plt.savefig('delay_mes.png', dpi=250)