from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
from scipy.stats import linregress, ttest_1samp
import numpy as np
import pylab as plt
#from tqdm import tqdm
from proc.settings import FB_ALL
from proc.settings import CHANNELS, MONTAGE
from mne.viz import plot_topomap
import scipy.signal as sg
from scipy.stats import linregress
from tqdm import tqdm

import pandas as pd


all_stats_df = pd.read_pickle('data/multichannel_multiband_metrics_split_local.pkl')
all_stats_df = all_stats_df.loc[all_stats_df['band']=='alpha']
all_stats_df = all_stats_df.loc[all_stats_df['block_number']>1000]
all_stats_df = all_stats_df.loc[all_stats_df['channel'].isin(CHANNELS)]
all_stats_df = all_stats_df.loc[all_stats_df['threshold_factor'].isin([2])]



y_df = pd.DataFrame(columns=['metric_type', 'fb_type', 'subj_id', 'channel', 'k', 'env'])
for metric_type, metric_type_df in all_stats_df.groupby('metric_type'):
    for fb_type, fb_type_df in metric_type_df.groupby('fb_type'):
        for s, (subj_id, subj_df) in enumerate(fb_type_df.groupby('subj_id')):
            for c, (ch, ch_df) in enumerate(subj_df.groupby('channel')):
                curve = ch_df['metric'].values
                curve[np.isinf(curve)] = np.nan
                curve[np.isnan(curve)] = 0.0001

                curve = pd.Series(curve).rolling(5, center=True).median().fillna(method='ffill').fillna(method='bfill')
                #curve = sg.filtfilt(np.ones(3)/3, [1, 0], curve)
                x0 = linregress(np.linspace(0, 1, 30), curve).intercept
                #x0 = curve[:15].mean()
                curve = curve/x0 - 1
                y_df = y_df.append(pd.DataFrame({'metric_type':metric_type, 'fb_type': fb_type, 'subj_id': 's'+str(subj_id), 'channel': ch, 'k': np.linspace(0, 1, 30), 'env': curve+0.0001}), ignore_index=True)



from pymer4.models import Lm, Lmer
from pymer4.utils import get_resource_path

for c, ch in enumerate(CHANNELS):
    for m, metric_type in enumerate(['magnitude', 'n_spindles', 'duration', 'amplitude']):
        data = y_df.query('metric_type=="{}" & channel=="{}"'.format(metric_type, ch))
        model = Lmer('env ~ k:fb_type + (1 |subj_id)', data=data, )
        model.fit(factors={'fb_type': ['FB0', 'FB250', 'FB500', 'FBMock']})
        a = model.post_hoc('k', 'fb_type')[1]
        a['channel'] = ch
        a['metric_type'] = metric_type
        if c==0 and m==0:
            all_stats_df = a.copy()
        else:
            all_stats_df = all_stats_df.append(a, ignore_index=True)
        print(ch, metric_type)

from scipy import stats
from mne.stats import fdr_correction


data = np.zeros((6, 4, 32))
for c, comp in enumerate(all_stats_df['Contrast'].unique()):
    for m, metric_type in enumerate(['magnitude', 'n_spindles', 'duration', 'amplitude']):
        comp_df = all_stats_df.query('metric_type=="{}" & Contrast == "{}"'.format(metric_type, comp))
        data[c, m] = np.array([stats.t.sf(comp_df.query('channel == "{}"'.format(ch))['T-stat'].values[0], 9) for ch in CHANNELS])

data = fdr_correction(data)[1]

fig, axes = plt.subplots(4, 6)
col_order = {'FB0 - FBMock': 0, 'FB250 - FBMock': 1, 'FB500 - FBMock': 2, 'FB0 - FB250': 3, 'FB0 - FB500': 4, 'FB250 - FB500': 5}
for c, comp in enumerate(all_stats_df['Contrast'].unique()):
    for m, metric_type in enumerate(['magnitude', 'n_spindles', 'duration', 'amplitude']):
        im=plot_topomap(np.log10(data[c, m]), MONTAGE.get_pos(), axes=axes[m, col_order[comp]], show=False, vmin=-4, vmax=0, cmap='inferno_r', contours=[np.log10(0.001), np.log10(0.01), np.log10(0.05)])
        axes[m, col_order[comp]].set_title('{}\n{}'.format(metric_type, comp))


plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cax = plt.axes([0.85, 0.1, 0.075, 0.8])

b = plt.colorbar(im[0], cax=cax)
b.set_ticks([np.log10(0.0001), np.log10(0.001), np.log10(0.01), np.log10(0.05), 0])
b.set_ticklabels(['0.0001', '0.001', '0.01', '0.05', '1'])
cax.set_title('p-value')
