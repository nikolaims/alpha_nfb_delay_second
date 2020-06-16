from release.settings import FB_ALL
import pandas as pd
import numpy as np
from release.stats.fan98test import fan98test
import pylab as plt

stats_file = 'block_stats_1channels_1bands_median_20ths.pkl'
stats_df = pd.read_pickle('release/data/{}'.format(stats_file))
# stats_df = stats_df.loc[stats_df.subj_id!=28]
stats_df = stats_df.loc[stats_df['block_number'].isin(FB_ALL)]
unique_blocks = list(stats_df['block_number'].unique())
stats_df['k'] = stats_df['block_number'].apply(lambda x: unique_blocks.index(x))
stats_df = stats_df.query('threshold_factor==2.5 & metric_type=="magnitude"')

fb_types = ['FB0', 'FB250', 'FB500', 'FBMock']
fb_data_points = []

for ind, fb_type in enumerate(fb_types):
    fb_stats_df = stats_df.query('fb_type=="{}"'.format(fb_type))
    unique_subj = fb_stats_df.subj_id.unique()
    data_points = np.zeros((len(unique_subj), len(FB_ALL)))
    for j, subj_id in enumerate(unique_subj):
        data_points[j, :] = fb_stats_df.query('subj_id=={}'.format(subj_id))['metric']
        data_points[j, :] /= data_points[j, 0]
        # data_points[j, :] /= data_points[j, :].mean()
    fb_data_points.append(data_points[:, 1:])

    plt.errorbar(np.arange(15)+ind*0.05, data_points.mean(0), 2*data_points.std(0)/np.sqrt(len(unique_subj)))
    # plt.plot(np.arange(15), data_points.T, color='C{}'.format(ind), alpha=0.5)

for ind1 in range(len(fb_types)):
    for ind2 in range(len(fb_types)-1, ind1, -1):
        print(fb_types[ind1], fb_types[ind2], fan98test(fb_data_points[ind1], fb_data_points[ind2])[0])
