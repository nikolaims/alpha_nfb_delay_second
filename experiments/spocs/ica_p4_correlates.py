import pandas as pd
import numpy as np
import scipy.signal as sg
import pylab as plt

from mne.decoding import SPoC
from  mne.viz import plot_topomap
from release.settings import CHANNELS, FS, FB_ALL, MONTAGE
from pynfb.signal_processing.decompositions import ICADecomposition

eeg_path = 'release/data/eeg_allsubjs_eyefree_1_45hz_down250hz.pkl'
info_path = 'release/data/info_allsubjs.pkl'

eeg_df = pd.read_pickle(eeg_path)
info_df = pd.read_pickle(info_path)

n_ma_sec = 20
def get_pow(x):
    return sg.lfilter(np.arange(FS * n_ma_sec) / FS / n_ma_sec, [1, 0], np.abs(x), axis=0)[FS * n_ma_sec::FS * n_ma_sec//2]

topos = np.load('experiments/spocs/data/ica_fb_topo.npy')
filts = np.load('experiments/spocs/data/ica_fb_filt.npy')


r2_comps = []
gains = []
for subj_id in range(40):
    print(subj_id)
    eeg_df_subj = eeg_df[eeg_df['subj_id']==subj_id]
    band = info_df.loc[info_df['subj_id']==subj_id, 'band'].values[0]

    fb_type = info_df.loc[info_df['subj_id']==subj_id, 'fb_type'].values[0]

    filter_n_taps = 250
    filter_b = sg.firwin2(filter_n_taps, [0, band[0], band[0], band[1], band[1], FS//2], [0, 0, 1, 1, 0, 0], fs=FS)


    x = eeg_df_subj.loc[eeg_df_subj['block_number'].isin(FB_ALL), CHANNELS].values
    x = sg.lfilter(filter_b, [1, 0], x, 0)[filter_n_taps // 2:]
    y = x[:, CHANNELS.index('P4')]

    p4_coefs = topos[subj_id][CHANNELS.index('P4')]
    # argsort = np.argsort(np.abs(p4_coefs))[::-1]
    filters = filts[subj_id]
    comps = x.dot(filters)*p4_coefs[None, :]

    comps_pow = get_pow(comps)
    y_pow = get_pow(y)
    r2_comps.append([1-np.sum((y_pow-comp)**2)/np.sum(y_pow**2) for comp in comps_pow.T])
    gains.append([comp[len(comp)//2:].mean()/comp[:len(comp)//2].mean() for comp in comps_pow.T])
    print(fb_type, r2_comps[-1])
    print(fb_type, gains[-1])

np.save('experiments/spocs/data/ica_fb_r2_with_p4.npy', np.array(r2_comps))
np.save('experiments/spocs/data/ica_fb_gain_with_p4.npy', np.array(gains))
r2_comps_ = np.array(r2_comps)
# r2_comps = np.array(gains)*(r2_comps_)
r2_comps = gains*(r2_comps_>0.07).astype(int)

n_rows = 1
fig, axes = plt.subplots(7, 40)
best_topos = []
for j_fb, fb_type in enumerate(['FB0', 'FB250', 'FB500', 'FBMock']):
    axes[0, j_fb * 10].set_title(fb_type)
    for j, subj_id in enumerate(info_df.loc[info_df.fb_type == fb_type, 'subj_id'].unique()[:10]):
        argsort = np.argsort(r2_comps[subj_id])[::-1]
        for k in range(len(axes)):
            ax = axes[k, j_fb*10 + j]
            topo = topos[subj_id][:, argsort[k]]
            topo = topo/topo[np.argmax(np.abs(topo))]
            topo = topo*(1 if sum(topo)>1 else np.sign(topo[CHANNELS.index('P4')]))
            # topo /= (topo**2).sum()**0.5
            if k < n_rows:
                best_topos.append(topo)
            plot_topomap(topo, MONTAGE.get_pos(), axes=ax, contours=0)
            ax.set_xlabel('{:.1f}'.format(r2_comps[subj_id][argsort[k]]))


from sklearn.cluster import KMeans
n_clust = 4
kmeans = KMeans(n_clust)
x = np.array(best_topos)
kmeans.fit(x)
pred = kmeans.predict(x)
perc = [sum(pred == k) for k in range(n_clust)]

fb_types = info_df.fb_type.values[:40]
fig, axes = plt.subplots(1, n_clust)
for j, k in enumerate(np.argsort(perc)[::-1]):
    plot_topomap(kmeans.cluster_centers_[k], MONTAGE.get_pos(), axes=axes[j], contours=0)

    groups, counts = np.unique(fb_types[[j//n_rows for j, p in enumerate(pred==k) if p]], return_counts=True)
    c = np.array([list(counts[groups == fb_type]) or [0] for fb_type in ['FB0', 'FB250', 'FB500', 'FBMock']])[:, 0]

    axes[j].set_xlabel('{}\n{}-FB0\n{}-FB250\n{}-FB500\n{}-FBMock'.format(perc[k], *c), ha='left')


