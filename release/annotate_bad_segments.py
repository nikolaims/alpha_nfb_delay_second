import numpy as np
import mne
import pylab as plt

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

# mask, annotations = annotate_bad(df[channels[:-1]], fs, channels[:-1])
# annotations.save('release/data/bad_annotations/s{}.csv'.format(subj_id))
# np.save('release/data/bad_annotations/good_mask_s{}.npy', mask)
# df = df.loc[mask]