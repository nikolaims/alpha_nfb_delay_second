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


all_stats_df = pd.read_pickle('data/multichannel_metrics_split_local.pkl')
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
                #curve = sg.filtfilt([1/6, 1/3, 1/3, 1/6], [1, 0], curve)
                x0 = linregress(np.linspace(0, 1, 30), curve).intercept
                curve = curve/x0-1
                #if np.any(np.isnan(curve)): print(curve)
                y_df = y_df.append(pd.DataFrame({'metric_type':metric_type, 'fb_type': fb_type, 'subj_id': 's'+str(subj_id), 'channel': ch, 'k': np.linspace(0, 1, 30), 'env': curve+0.0001}), ignore_index=True)




metric_type = 'magnitude'
fb1_type = 'FB500'
fb2_type = 'FBMock'
ch = 'P4'
x = np.linspace(0, 1, 30).repeat(10)

data = y_df.query('metric_type=="{}" & channel=="{}"'.format(metric_type, ch))

fb1_subjs = data.query('fb_type=="{}"'.format(fb1_type))['subj_id'].unique()
fb2_subjs = data.query('fb_type=="{}"'.format(fb2_type))['subj_id'].unique()
all_subjs = np.concatenate([fb1_subjs, fb2_subjs])


def get_stat(subj1, subj2, fun=np.log):
    y1 = np.array([data.query('subj_id=="{}"'.format(s))['env'] for s in subj1])
    y2 = np.array([data.query('subj_id=="{}"'.format(s))['env'] for s in subj2])
    #stat = linregress(x, fun(y1.flatten())).slope - linregress(x, fun(y2).flatten()).slope
    stat = linregress(np.arange(30), (np.mean(y1, 0)-np.mean(y2, 0))).slope
    return stat

n_perm = 1000
stats = np.zeros(n_perm)
for k in range(n_perm):
    stats[k] = get_stat(*np.split(all_subjs[np.random.permutation(np.arange(20))], 2))

print(np.sum(stats>get_stat(fb1_subjs, fb2_subjs))/n_perm)






fb2_type = 'FBMock'
x = np.linspace(0, 1, 30).repeat(10)
ch_p_vals = np.zeros((4, 4, len(CHANNELS)))
ch_stats = np.zeros((4, 4, len(CHANNELS)))



for m, metric_type in enumerate(['magnitude', 'n_spindles', 'duration', 'amplitude']):
    for f, fb1_type in tqdm(enumerate(['FB0', 'FB250', 'FB500', 'FBMock']), metric_type):

        fb1_subjs = y_df.query('fb_type=="{}"'.format(fb1_type))['subj_id'].unique()
        fb2_subjs = y_df.query('fb_type=="{}"'.format(fb2_type))['subj_id'].unique()
        all_subjs = np.concatenate([fb1_subjs, fb2_subjs])
        y = np.array([[y_df.query('subj_id=="{}" & channel=="{}" & metric_type=="{}"'.format(s, ch, metric_type))['env'] for s in all_subjs] for ch in CHANNELS])


        for c, ch in enumerate(CHANNELS):
            #
            # def get_stat(y1, y2, fun=np.array):
            #     y1 = y1.T
            #     y2 = y2.T
            #     stat = linregress(x, fun(y1.flatten())).slope# - linregress(x, fun(y2.flatten())).slope
            #     #stat = np.mean(np.mean(y1[15:], 0)/np.mean(y1[:15], 0)) - np.mean(np.mean(y2[15:], 0)/np.mean(y2[:15], 0))
            #     return stat

            def get_stat(y1, y2, fun=np.array):
                stat = linregress(x, fun(y1.T.flatten())).slope - linregress(x, fun(y2.T.flatten())).slope
                #stat = linregress(np.arange(30), (np.mean(y1, 0)-np.mean(y2, 0))).slope
                #stat = np.sum((np.mean(y1, 0)[15:] - np.mean(y2, 0)[15:]))
                #stat = np.mean(np.mean(y1[:, 15:], 1)/np.mean(y1[:, :15], 1)) - np.mean(np.mean(y2[:, 15:], 1)/np.mean(y2[:, :15], 1))
                return stat



            n_perm = 1000
            stats = np.zeros(n_perm)
            for k in range(n_perm):
                # y1_ind, y2_ind = np.split(np.random.permutation(np.arange(10).repeat(2)), 2)
                y1_ind, y2_ind = np.split(np.random.permutation(np.arange(20)), 2)
                stats[k] = get_stat(y[c, y1_ind], y[c, y2_ind])


            ch_stats[m, f, c] = get_stat(y[c, :10], y[c, 10:])
            #ch_p_vals[m, f, c] = (n_perm-np.sum(stats>0))/n_perm
            ch_p_vals[m, f, c] =  np.sum(stats > ch_stats[m, f, c]) / n_perm


from mne.stats import fdr_correction

fig, axes = plt.subplots(4, 4)
for m, metric_type in enumerate(['magnitude', 'n_spindles', 'duration', 'amplitude']):
    for f, fb1_type in tqdm(enumerate(['FB0', 'FB250', 'FB500', 'FBMock']), metric_type):

        fdr_mask = fdr_correction(ch_p_vals, 0.05)[0][m, f]
        im = plot_topomap(np.log10(ch_p_vals[m, f]+0.0001), MONTAGE.get_pos(), mask=fdr_mask, axes=axes[m, f], show=False, vmin=-2, vmax=0, cmap='inferno_r', contours=[np.log10(0.05)])
        axes[0, f].set_title(fb1_type)

    axes[m, 0].set_ylabel(metric_type)

plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cax = plt.axes([0.85, 0.1, 0.075, 0.8])

b = plt.colorbar(im[0], cax=cax)
b.set_ticks([np.log10(0.01), np.log10(0.05), 0])
b.set_ticklabels(['0.01', '0.05', '1'])
cax.set_title('p-value')



from scipy.stats import rankdata
def rankstat(x, y):
    s = np.array([np.sum(rankdata(np.concatenate((x[:, k], y[:, k])))[:len(x)]) for k in range(x.shape[1])])
    return s


n_observations = 10
n_times = 30
n_vertices = 32


for metric_type in ['magnitude', 'n_spindles', 'duration', 'amplitude']:
    for fb_type in ['FB0', 'FB250', 'FB500']:

        g1 = np.swapaxes(np.array([[y_df.query('metric_type=="{}" & subj_id=="{}" & channel=="{}"'.format(metric_type, s, ch))['env'].values
             for ch in CHANNELS] for s in y_df.query('fb_type=="{}"'.format(fb_type))['subj_id'].unique()]), 2,1)

        g2 = np.swapaxes(np.array([[y_df.query('metric_type=="{}" & subj_id=="{}" & channel=="{}"'.format(metric_type, s, ch))['env'].values
             for ch in CHANNELS] for s in y_df.query('fb_type=="FBMock"')['subj_id'].unique()]), 2,1)


        from mne.stats import  spatio_temporal_cluster_test

        from mne import create_info
        from mne.channels import read_montage, find_ch_connectivity

        cnk = find_ch_connectivity(create_info(CHANNELS, 250, 'eeg', read_montage('standard_1005')), 'eeg')[0]


        t_obs, clusters, cluster_pv, h0 = spatio_temporal_cluster_test([g1, g2], 138, stat_fun=rankstat, tail=1, connectivity=cnk)
        cluster_pv


        good_cluster_inds = np.where(cluster_pv < 0.05)[0]



        for i_clu, clu_idx in enumerate(good_cluster_inds[:10]):
            # unpack cluster information, get unique indices
            time_inds, space_inds = np.squeeze(clusters[clu_idx])
            ch_inds = np.unique(space_inds)
            time_inds = np.unique(time_inds)

            # get topography for F stat
            f_map = t_obs[time_inds, ...].mean(axis=0)

            # get signals at the sensors contributing to the cluster
            sig_times = np.arange(30)[time_inds]

            # create spatial mask
            mask = np.zeros((f_map.shape[0], 1), dtype=bool)
            mask[ch_inds, :] = True

            plt.figure()
            plt.title('{}\n{}_{}'.format(metric_type, fb_type, i_clu))
            plt.xlabel('{}-{}'.format(sig_times[0], sig_times[-1]))
            plot_topomap(f_map, MONTAGE.get_pos(), mask=mask, cmap='inferno',
                         vmin=105, vmax=139, show=True)
            plt.savefig('data/img/clust/{}_{}_{}'.format(metric_type, fb_type, i_clu))
            plt.close()


