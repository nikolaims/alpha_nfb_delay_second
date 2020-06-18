from release.settings import FB_ALL
import pandas as pd
import numpy as np
from release.stats.fan98test import fan98test, simulate_h0_distribution, get_p_val
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
        # data_points[j, :] /= data_points[j, 0]
        data_points[j, :] /= data_points[j, :].mean()
    fb_data_points.append(data_points[:, :])

    plt.errorbar(np.arange(15)+ind*0.05, data_points.mean(0), 2*data_points.std(0)/np.sqrt(len(unique_subj)))
    # plt.plot(np.arange(15), data_points.T, color='C{}'.format(ind), alpha=0.5)

n_subjects_list = sorted([fb_data_points[k].shape[0] for k in range(len(fb_data_points))])
d_list = np.arange(sum(n_subjects_list[:2]) - 2, sum(n_subjects_list[-2:]) - 2 + 1)
h0_distributions_dict = {}
for d in d_list:
    h0_distributions_dict[d] = simulate_h0_distribution(fb_data_points[0].shape[1], d)


for ind1 in range(len(fb_types)):
    for ind2 in range(len(fb_types)-1, ind1, -1):
        stat = fan98test(fb_data_points[ind2], fb_data_points[ind1])
        d = fb_data_points[ind1].shape[0] + fb_data_points[ind2].shape[0] - 2
        p = get_p_val(stat, h0_distribution=h0_distributions_dict[d])
        print(fb_types[ind1], fb_types[ind2], stat, p)
