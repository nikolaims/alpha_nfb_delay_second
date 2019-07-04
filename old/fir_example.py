import scipy.signal as sg
import pylab as plt
import numpy as np
from matplotlib.animation import FuncAnimation

cm = {'b':'#0099d8', 'lb': '#84BCDA', 'r':'#FE4A49', 'g':'#A2A79E', 'dg': '#444444'}

np.random.seed(42)
fs = 500
t = np.arange(fs*2)/fs
signal = ((np.sin(2 * np.pi * 10 * t) + 0.2) - 0.2 * (np.cos(2 * np.pi * 20 * t) + 0.25)) * np.sin(2 * np.pi * t * 3 / 4)
signal[(t < 1 / 3 * 2) | (t > 2 / 3 * 2)] = 0
noise_spec = np.fft.rfft(np.random.normal(size=2*fs))*1/(1+np.arange(fs+1)**1)
noise_spec[:10]=0
noise = np.fft.irfft(noise_spec)*12
x = signal + noise

n_taps = fs//2
band = [8, 12]
freq = [0, band[0], band[0], band[1], band[1], fs/2]
gain = [0, 0, 1, 1, 0, 0]
b1 =sg.minimum_phase(sg.firwin2(n_taps*2, freq, gain, fs=fs))
b = sg.firwin2(n_taps, freq, gain, fs=fs)

y = sg.lfilter(b, 1, x)
y1 = sg.lfilter(b1, 1, x)
# plt.semilogy(*sg.welch(y))

def setup_gca(ax, bottom=False):
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if bottom: ax.spines['bottom'].set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_ylim(-1.4, 1.4)
    #ax.get_xaxis().set_visible(False)



fig, axes = plt.subplots(2, sharex='all', sharey='all')

plt.subplots_adjust(hspace=0)
def plot_frame(n):
    axes[0].cla()
    axes[1].cla()
    axes[0].plot(t, x, color=cm['b'])
    axes[0].plot(t[n-n_taps:n], b[::-1]/max(b), color=cm['r'])
    #axes[0].axvspan(t[n-n_taps], t[n], alpha=0.15, color=cm['r'], linewidth=0)
    axes[0].plot([t[n], t[n-n_taps], t[n-n_taps], t[n]], [1.35, 1.35, -1.35, -1.35], '--', color=cm['r'])
    axes[1].plot(t[:n], y[:n], color=cm['b'])
    axes[1].plot(t[n], y[n], 'o', color=cm['r'], markersize=10)
    axes[1].set_xlabel('Time, s')
    [ax.axvline(t[n], color=cm['r']) for ax in axes]
    setup_gca(axes[0], True)
    setup_gca(axes[1], False)


plot_frame(n_taps)

anim = FuncAnimation(fig, plot_frame, frames=np.arange(n_taps, fs*2, 5), interval=20*3)
anim.save('line.gif', dpi=150, writer='imagemagick',  bitrate=2)