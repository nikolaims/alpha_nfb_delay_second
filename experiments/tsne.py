import pandas as pd
import numpy as np
import pylab as plt
from release.settings import FB_ALL
stats_file = 'FBLow_channels1_bands1_splitedFalse_median_threshs20.pkl'
stats_df = pd.read_pickle('release/data/{}'.format(stats_file))
stats_df = stats_df.loc[stats_df['block_number'].isin(FB_ALL)]
unique_blocks = list(stats_df['block_number'].unique())
stats_df['k'] = stats_df['block_number'].apply(lambda x: unique_blocks.index(x))

data = stats_df.query('threshold_factor==2.5 & metric_type=="magnitude"')

subj_list = data.subj_id.unique()
block_list = data.k.unique()

all_curves = np.zeros((len(subj_list), len(block_list)))
all_fb_types = []

for j, subj_id in enumerate(subj_list):
    subj_data = data.query('subj_id=={}'.format(subj_id))
    curve = subj_data['metric'].values
    curve/=curve[:1].mean()
    fb_type = subj_data['fb_type'].values[0]
    print(fb_type)
    all_curves[j] = curve
    all_fb_types.append(fb_type)

from sklearn.manifold import TSNE

tsne = TSNE(perplexity=5, metric='cosine')

Y = tsne.fit_transform(all_curves)

fb_types = data.fb_type.unique()
fb_dict = dict(zip(fb_types, ['C{}'.format(k) for k in range(len(fb_types))]))

color = list(map(lambda x: fb_dict[x], all_fb_types))

plt.scatter(Y[:, 0], Y[:, 1], c=color,)

for j, subj_id in enumerate(subj_list):
    plt.text(Y[j, 0], Y[j, 1], all_fb_types[j], color=color[j])


