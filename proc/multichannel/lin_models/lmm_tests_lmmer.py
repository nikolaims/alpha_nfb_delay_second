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
from scipy import stats


all_stats_df = pd.read_pickle('data/multichannel_multiband_metrics_split_local.pkl')
all_stats_df = all_stats_df.loc[all_stats_df['band'].isin(['alpha', 'beta', 'theta'])]
all_stats_df = all_stats_df.loc[all_stats_df['block_number']>1000]
all_stats_df = all_stats_df.loc[all_stats_df['channel'].isin(CHANNELS)]
all_stats_df = all_stats_df.loc[all_stats_df['threshold_factor'].isin([2])]



y_df = pd.DataFrame(columns=['metric_type', 'fb_type', 'subj_id', 'channel', 'k', 'env', 'band'])
for band, band_df in all_stats_df.groupby('band'):
    for metric_type, metric_type_df in band_df.groupby('metric_type'):
        for fb_type, fb_type_df in metric_type_df.groupby('fb_type'):
            for s, (subj_id, subj_df) in tqdm(enumerate(fb_type_df.groupby('subj_id')), band+metric_type+fb_type):
                for c, (ch, ch_df) in enumerate(subj_df.groupby('channel')):
                    curve = ch_df['metric'].values
                    curve[np.isinf(curve)] = np.nan
                    curve[np.isnan(curve)] = 0.0001

                    curve = pd.Series(curve).rolling(5, center=True).median().fillna(method='ffill').fillna(method='bfill')
                    #curve = sg.filtfilt(np.ones(3)/3, [1, 0], curve)
                    x0 = linregress(np.linspace(0, 1, 30), curve).intercept
                    #x0 = curve[:15].mean()
                    curve = curve/x0 - 1
                    y_df = y_df.append(pd.DataFrame({'metric_type':metric_type, 'fb_type': fb_type, 'subj_id': 's'+str(subj_id), 'channel': ch, 'k': np.linspace(0, 1, 30), 'env': curve+0.0001, 'band': band}), ignore_index=True)



from pymer4.models import Lm, Lmer
from pymer4.utils import get_resource_path

for b, band in enumerate(['alpha', 'beta', 'theta']):
    for c, ch in enumerate(CHANNELS):
        for m, metric_type in enumerate(['magnitude', 'n_spindles', 'duration', 'amplitude']):
            data = y_df.query('metric_type=="{}" & channel=="{}" & band=="{}"'.format(metric_type, ch, band))
            model = Lmer('env ~ k:fb_type + (1 |subj_id)', data=data, )
            model.fit(factors={'fb_type': ['FB0', 'FB250', 'FB500', 'FBMock']})
            a = model.post_hoc('k', 'fb_type')[1]
            a['channel'] = ch
            a['metric_type'] = metric_type
            a['band'] = band
            a['P-val-full'] = stats.t.sf(a['T-stat'], 9)
            if c==0 and m==0 and b==0:
                all_stats_df = a.copy()
            else:
                all_stats_df = all_stats_df.append(a, ignore_index=True)
            print(ch, metric_type)

from mne.stats import fdr_correction


data = np.zeros((3, 6, 4, 32))
data_t = np.zeros((3, 6, 4, 32))
for b, band in enumerate(['alpha', 'beta', 'theta']):
    for c, comp in enumerate(all_stats_df['Contrast'].unique()):
        for m, metric_type in enumerate(['magnitude', 'n_spindles', 'duration', 'amplitude']):
            comp_df = all_stats_df.query('metric_type=="{}" & Contrast == "{}"'.format(metric_type, comp))
            data_t[b, c, m] = np.array([comp_df.query('channel == "{}" & band=="{}"'.format(ch, band))['T-stat'].values[0] for ch in CHANNELS])
            data[b, c, m] = np.array([stats.t.sf(comp_df.query('channel == "{}" & band=="{}"'.format(ch, band))['T-stat'].values[0], 9) for ch in CHANNELS])

data = fdr_correction(data, 0.05)[1]

fig, axes = plt.subplots(3*4, 6)
col_order = {'FB0 - FBMock': 0, 'FB250 - FBMock': 1, 'FB500 - FBMock': 2, 'FB0 - FB250': 3, 'FB0 - FB500': 4, 'FB250 - FB500': 5}
for b, band in enumerate(['alpha', 'beta', 'theta']):
    for c, comp in enumerate(all_stats_df['Contrast'].unique()):
        for m, metric_type in enumerate(['magnitude', 'n_spindles', 'duration', 'amplitude']):
            d = np.log10(data[b, c, m])
            d[(data_t[b, c, m] > 0)] *= -1
            im=plot_topomap(d, MONTAGE.get_pos(), axes=axes[m+4*b, col_order[comp]], show=False, vmin=-4, vmax=4, cmap='RdBu_r', contours=[np.log10(0.05), -np.log10(0.05)])
            axes[m+4*b, col_order[comp]].set_title('{}\n{}\n{}'.format(band, metric_type, comp))


plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cax = plt.axes([0.85, 0.1, 0.075, 0.8])

b = plt.colorbar(im[0], cax=cax)
b.set_ticks([np.log10(0.05), -np.log10(0.05), 0, np.log10(0.01), -np.log10(0.01), np.log10(0.001), -np.log10(0.001)])
b.set_ticklabels([ '* (<)', '* (>)', 'ns', '** (<)', '** (>)', '*** (<)', '*** (>)'])
cax.set_title('p-value')



col_order = {'FB0 - FBMock': 0, 'FB250 - FBMock': 1, 'FB500 - FBMock': 2, 'FB0 - FB250': 3, 'FB0 - FB500': 4, 'FB250 - FB500': 5}
for b, band in enumerate(['alpha', 'beta', 'theta']):

    fig, axes = plt.subplots(4, 6, figsize=(6, 5))
    plt.subplots_adjust(right=4)

    for c, comp in enumerate(all_stats_df['Contrast'].unique()):
        for m, metric_type in enumerate(['magnitude', 'n_spindles', 'duration', 'amplitude']):
            d = np.log10(data[b, c, m])
            d[(data_t[b, c, m] > 0)] *= -1
            im=plot_topomap(d, MONTAGE.get_pos(), axes=axes[m, col_order[comp]], show=False, vmin=-4, vmax=4, cmap='RdBu_r', contours=[np.log10(0.05), -np.log10(0.05)])
            axes[0, col_order[comp]].set_title('{}\n{}'.format(comp.split()[0], comp.split()[2]))
            axes[m, 0].set_ylabel(metric_type)

    fig.suptitle(band)
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    cax = plt.axes([0.85, 0.1, 0.03, 0.8])

    b = plt.colorbar(im[0], cax=cax)
    b.set_ticks([np.log10(0.05), -np.log10(0.05), 0, np.log10(0.01), -np.log10(0.01), np.log10(0.001), -np.log10(0.001)])
    b.set_ticklabels([ '* (<)', '* (>)', 'ns', '** (<)', '** (>)', '*** (<)', '*** (>)'])
    cax.set_title('p-value')
    plt.savefig(band+'_comp.png', dpi=200)