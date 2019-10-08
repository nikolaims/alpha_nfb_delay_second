import pylab as plt
import pandas as pd
import numpy as np
import seaborn as sns

stats_df = pd.read_pickle('release/data/{}.pkl'.format('channels1_bands1_splitedTrue_median_thresholds_pers20'))
stats_df = stats_df.loc[stats_df['block_number']> 1000]
stats_df = stats_df.query('threshold_factor==90')
unique_blocks = list(stats_df['block_number'].unique())
stats_df['k'] = stats_df['block_number'].apply(lambda x: unique_blocks.index(x))
stats_df['subj_id'] = 's'+stats_df['subj_id'].astype('str')

# stats_df.loc[stats_df['metric_type']=="magnitude", 'metric'] = np.log(stats_df.query('metric_type=="magnitude"')['metric'])
# stats_df.loc[stats_df['metric_type']=="amplitude", 'metric'] = np.log(stats_df.query('metric_type=="amplitude"')['metric'])

for group, data in stats_df.groupby(['subj_id', 'metric_type']):
    stats_df.loc[(stats_df['subj_id']==group[0]) & (stats_df['metric_type']==group[1]), 'metric'] = data['metric']/ data.query('k<15')['metric'].mean()

sns.relplot('k', 'metric', 'subj_id', row='metric_type', col='fb_type', data=stats_df,
            #hue_order=['FB0', 'FB250', 'FB500', 'FBMock'], palette=['#3CB4E8', '#438BA8', '#002A3B', '#FE4A49'],
            col_order=['FB0', 'FB250', 'FB500', 'FBMock'], kind='line', facet_kws={'sharey': 'row'},
            units="subj_id", estimator=None)