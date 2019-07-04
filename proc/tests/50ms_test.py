from utils.load_results import load_data
import pylab as plt
import numpy as np
from seaborn import color_palette
cm = color_palette()

def band_hilbert(x, fs, band, N=None, axis=-1):
    x = np.asarray(x)
    Xf = np.fft.fft(x, N, axis=axis)
    w = np.fft.fftfreq(x.shape[0], d=1. / fs)
    Xf[(w < band[0]) | (w > band[1])] = 0
    x = np.fft.ifft(Xf, axis=axis)
    return 2*x

df50, fs, channels, p_names = load_data('/home/kolai/Data/delay_tests/50ms_test/50ms_delay_test_50ms_06-19_15-32-50/experiment_data.h5')
df100, fs, channels, p_names = load_data('/home/kolai/Data/delay_tests/50ms_test/50ms_delay_test_100ms_06-19_15-43-02/experiment_data.h5')


signal = ['PHOTO', 'signal_Alpha0'][1]

a50 = df50.query('block_name=="FB0"')[signal].values
a100 = df100.query('block_name=="FB0"')[signal].values
raw50 = np.abs(band_hilbert(df50.query('block_name=="FB0"')['P4'].values, fs, [8,12]))
raw100 = np.abs(band_hilbert(df100.query('block_name=="FB0"')['P4'].values, fs, [8,12]))


delays = np.arange(0, 200, 5)
corr50 = [np.corrcoef(np.roll(raw50, k), a50)[1,0] for k in delays]
corr100 = [np.corrcoef(np.roll(raw100, k), a100)[1,0] for k in delays]
plt.plot(delays*2, corr50)
plt.plot(delays*2, corr100)

plt.axvline(delays[np.argmax(corr50)]*2, color=cm[0])
plt.axvline(delays[np.argmax(corr100)]*2, color=cm[1])

plt.legend(['50ms', '100ms'])
plt.xlabel('Delay, ms')
plt.ylabel('Corr.')


nor = lambda x: (x - x.mean())/x.std()
fig, axes = plt.subplots(2, sharex='all', sharey='all')
axes[0].plot(nor(np.roll(a50, -delays[np.argmax(corr50)])), color=cm[0], alpha=0.5)
axes[0].plot(nor(raw50), color=cm[0], alpha=1)
axes[1].plot(nor(np.roll(a100, -delays[np.argmax(corr100)])), color=cm[1], alpha=0.5)
axes[1].plot(nor(raw100), color=cm[1], alpha=1)