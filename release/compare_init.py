from mne.stats import fdr_correction
from pymer4.models import Lm, Lmer
import pandas as pd
import numpy as np
from scipy import stats

stats_df = pd.read_pickle('release/data/{}.pkl'.format('channels1_bands1_splitedTrue_thresholds17'))
stats_df = stats_df.loc[stats_df['block_number']> 1000]
unique_blocks = list(stats_df['block_number'].unique())
stats_df['k'] = stats_df['block_number'].apply(lambda x: unique_blocks.index(x))
stats_df['subj_id_str'] = 's' + stats_df['subj_id'].astype('str')
stats_df = stats_df.query('k < 15')

import seaborn as sns

sns.catplot('fb_type', 'metric', kind='box', col='metric_type', sharey='col',
            data=stats_df.query('threshold_factor==2.75').groupby(['subj_id', 'fb_type', 'metric_type']).mean().reset_index())




stats_df = pd.read_pickle('release/data/{}.pkl'.format('channels1_bands1_splitedTrue_thresholds17'))
stats_df = stats_df.query('block_number==4 | block_number==36')
stats_df['block_name'] = stats_df['block_number'].apply(lambda x: 0 if x ==4 else 1)
stats_df['subj_id_str'] = 's' + stats_df['subj_id'].astype('str')

sns.catplot('block_name', 'metric', 'fb_type', kind='point', col='metric_type', sharey='col', data=stats_df.query('threshold_factor==2.75'), dodge=True)


from pymer4.models import Lm, Lmer
metric_type = 'duration'
threshold_factor = 2.75
data = stats_df.query('metric_type=="{}" & threshold_factor=={}'.format(metric_type, threshold_factor)).copy()
print(len(data))
data = data.replace([np.inf, -np.inf], np.nan)
data.loc[:, 'metric'] = data['metric'].fillna(data['metric'].min()).values
model = Lmer('metric ~ block_name:fb_type + (1 |subj_id_str)', data=data)
model.fit(factors={'fb_type': ['FB0', 'FB250', 'FB500', 'FBMock']})
model.post_hoc('block_name', 'fb_type')