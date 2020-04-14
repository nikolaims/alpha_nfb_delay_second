import mne
import numpy as np
import pylab as plt
import scipy.signal as sg


def band_hilbert(x, fs, band, N=None, axis=-1):
    x = np.asarray(x)
    Xf = np.fft.fft(x, N, axis=axis)
    w = np.fft.fftfreq(x.shape[0], d=1. / fs)
    Xf[(w < band[0]) | (w > band[1])] = 0
    x = np.fft.ifft(Xf, axis=axis)
    return 2*x

class CFIRBandEnvelopeDetector:
    def __init__(self, band, fs, delay, n_taps=500, n_fft=2000, weights=None, **kwargs):
        w = np.arange(n_fft)
        H = 2 * np.exp(-2j * np.pi * w / n_fft * delay)
        H[(w / n_fft * fs < band[0]) | (w / n_fft * fs > band[1])] = 0
        F = np.array([np.exp(-2j * np.pi / n_fft * k * np.arange(n_taps)) for k in np.arange(n_fft)])
        if weights is None:
            self.b = F.T.conj().dot(H)/n_fft
        else:
            W = np.diag(weights)
            self.b = np.linalg.solve(F.T.dot(W.dot(F.conj())), (F.T.conj()).dot(W.dot(H)))
        self.a = np.array([1.])
        self.zi = np.zeros(len(self.b)-1)

    def apply(self, chunk: np.ndarray):
        y, self.zi = sg.lfilter(self.b, self.a, chunk, zi=self.zi)
        return y


def annotate_bad(data, fs, channels, threshold):
    th = (np.abs(data).rolling(int(fs)//2, center=True).max().max(1).fillna(0).values > threshold).astype(int)
    onsets = np.where(np.diff(th)>0)[0]/fs
    durations = np.where(np.diff(th)<0)[0]/fs - onsets
    gfp_ann = mne.Annotations(onsets, durations, 'BAD')

    mne_info = mne.create_info(ch_names=channels, sfreq=fs, ch_types='eeg')
    raw = mne.io.RawArray(data.values.T, mne_info)
    raw.set_annotations(gfp_ann)
    fig = raw.plot(n_channels=32, duration=30, scalings={'eeg': 60e-6})
    fig.canvas.key_press_event('a')
    plt.show(block=True)
    plt.close()

    good_mask = np.ones(len(data))
    t = np.arange(len(data))/fs
    for onset, duration in zip(raw.annotations.onset, raw.annotations.duration):
        good_mask[(t >= onset) & (t <= onset+duration)] = 0
    return good_mask > 0, raw.annotations