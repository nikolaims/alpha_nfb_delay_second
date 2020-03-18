import pandas as pd
import numpy as np
import scipy.signal as sg
import pylab as plt

from mne.decoding import SPoC
from  mne.viz import plot_topomap
from release.settings import CHANNELS, FS, FB_ALL, MONTAGE,  Montage
from pynfb.signal_processing.decompositions import ICADecomposition

eeg_path = 'release/data/eeg_allsubjs_eyefree_1_45hz_down250hz.pkl'
info_path = 'release/data/info_allsubjs.pkl'

eeg_df = pd.read_pickle(eeg_path)
info_df = pd.read_pickle(info_path)

topos = np.load('experiments/spocs/data/spoc_fb_topos.npy')
components = np.load('experiments/spocs/data/spoc_fb_components.npy')
targets = np.load('experiments/spocs/data/spoc_fb_targets.npy')


NO_P4_CH = [ch for ch in CHANNELS if ch!='P4']
pos = Montage(NO_P4_CH).get_pos()

corrs = []
for subj_id in range(40):
    print(subj_id)
    corrs.append([np.cov(targets[subj_id], comp)[1, 0] for comp in components[subj_id].T])


corrs = np.array(corrs)

n_rows = 1
fig, axes = plt.subplots(7, 40)
best_topos = []
for j_fb, fb_type in enumerate(['FB0', 'FB250', 'FB500', 'FBMock']):
    axes[0, j_fb * 10].set_title(fb_type)
    for j, subj_id in enumerate(info_df.loc[info_df.fb_type == fb_type, 'subj_id'].unique()[:10]):
        argsort = np.argsort(corrs[subj_id])[::-1]
        for k in range(len(axes)):
            ax = axes[k, j_fb*10 + j]
            topo = topos[subj_id][:, argsort[k]]
            topo = topo/topo[np.argmax(np.abs(topo))]
            topo = topo*(1 if sum(topo)>1 else np.sign(topo[CHANNELS.index('P4')]))
            # topo /= (topo**2).sum()**0.5
            if k < n_rows:
                best_topos.append(topo)
            plot_topomap(topo, pos, axes=ax, contours=0)
            ax.set_xlabel('{:.1f}'.format(corrs[subj_id][argsort[k]]))


from sklearn.cluster import KMeans
n_clust = 5
kmeans = KMeans(n_clust)
x = np.array(best_topos)
kmeans.fit(x)
pred = kmeans.predict(x)
perc = [sum(pred == k) for k in range(n_clust)]

fb_types = info_df.fb_type.values[:40]
fig, axes = plt.subplots(1, n_clust)
for j, k in enumerate(np.argsort(perc)[::-1]):
    plot_topomap(kmeans.cluster_centers_[k], pos, axes=axes[j], contours=0)

    groups, counts = np.unique(fb_types[[j//n_rows for j, p in enumerate(pred==k) if p]], return_counts=True)
    c = np.array([list(counts[groups == fb_type]) or [0] for fb_type in ['FB0', 'FB250', 'FB500', 'FBMock']])[:, 0]

    axes[j].set_xlabel('{}\n{}-FB0\n{}-FB250\n{}-FB500\n{}-FBMock'.format(perc[k], *c), ha='left')


