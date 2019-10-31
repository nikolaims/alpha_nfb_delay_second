import pylab as plt
import pandas as pd
import numpy as np
import seaborn as sns

stats_df = pd.read_pickle('release/data/{}.pkl'.format('channels1_bands1_splitedTrue_median_threshs20'))
stats_df = stats_df.loc[stats_df['block_number']> 1000]
stats_df = stats_df.query('threshold_factor==2.75')
unique_blocks = list(stats_df['block_number'].unique())
stats_df['k'] = stats_df['block_number'].apply(lambda x: unique_blocks.index(x))
stats_df['subj_id'] = 's'+stats_df['subj_id'].astype('str')

# stats_df.loc[stats_df['metric_type']=="magnitude", 'metric'] = np.log(stats_df.query('metric_type=="magnitude"')['metric'])
# stats_df.loc[stats_df['metric_type']=="amplitude", 'metric'] = np.log(stats_df.query('metric_type=="amplitude"')['metric'])

for group, data in stats_df.groupby(['subj_id', 'metric_type']):
    stats_df.loc[(stats_df['subj_id']==group[0]) & (stats_df['metric_type']==group[1]), 'metric'] = data['metric']/ data.query('k<15')['metric'].mean()


sns.set_style("whitegrid")
g= sns.relplot('k', 'metric', 'fb_type', row='metric_type', col='fb_type', data=stats_df.query('metric_type=="magnitude"'),
            hue_order=['FB0', 'FB250', 'FB500', 'FBMock'], palette=['#3CB4E8', '#438BA8', '#002A3B', '#FE4A49'],
            col_order=['FB0', 'FB250', 'FB500', 'FBMock'], kind='line', facet_kws={'sharey': 'row'},
            marker='.', markeredgecolor='none', markersize=10, aspect=0.8, height=3,)
g.axes[0, 0].set_ylabel('Magnitude, $\mu V$')
[ax.set_title(s) for ax, s in zip(g.axes.flatten(), ['FB0', 'FB250', 'FB500', 'FBMock'])]
[ax.set_xlabel('Block number') for ax in g.axes.flatten()]
g.axes[0,0].set_xlim(0, 30)
plt.subplots_adjust(wspace=0.1)

plt.savefig('curves_p4_grid_30.png', dpi=200)