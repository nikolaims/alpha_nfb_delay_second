from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
from scipy.stats import linregress, ttest_1samp
import numpy as np
import pylab as plt
#from tqdm import tqdm
from proc.settings import FB_ALL
from proc.settings import CHANNELS, MONTAGE
from mne.viz import plot_topomap
import scipy.signal as sg


import pandas as pd


all_stats_df = pd.read_pickle('data/2split_metrics_chs_ica_all.pkl')
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
                curve = pd.Series(curve).fillna(method='bfill').fillna(method='ffill')
                curve[np.isnan(curve)] = 0.001
                #if np.any(np.isnan(curve)): print(curve)
                y_df = y_df.append(pd.DataFrame({'metric_type':metric_type, 'fb_type': fb_type, 'subj_id': 's'+str(subj_id), 'channel': ch, 'k': np.linspace(0, 1, 30), 'env': curve+0.0001}), ignore_index=True)

y_df.to_csv('fb_curves.csv', index=False)

import statsmodels.api as sm

for m, metric_type in enumerate(['magnitude', 'n_spindles', 'duration', 'amplitude'][1:2]):
    data = y_df.query('metric_type == "{}"'.format(metric_type))
    md = sm.MixedLM.from_formula("np.log(env) ~ k:fb_type:channel", data, groups=data["subj_id"], re_formula='1+k:channel:fb_type')

    results = md.fit()

    print(results.summary())



#topo0 = np.exp([results.params['channel[T.{}]'.format(ch)] if ch !='C3' else 0 for ch in CHANNELS])
topo_fb0 = np.exp([results.params['k:fb_type[FBMock]:channel[{}]'.format(ch)] for ch in CHANNELS])
plot_topomap(topo_fb0, MONTAGE.get_pos(), vmin=0, vmax=1)




md = sm.MixedLM.from_formula("np.log(env) ~ channel + k:fb_type:channel", data, groups=data["subj_id"], re_formula='1')
results = md.fit()
print(results.summary())



m = sm.OLS.from_formula('env ~ fb_type*channel', data=y_df.query('metric_type == "n_spindles" & k==0')).fit()
#m1 = sm.OLS.from_formula('env ~ fb_type:k', data=y_df.query('metric_type == "n_spindles" & channel=="P4"')).fit()
print(m.summary())
from statsmodels.stats.anova import anova_lm
anova_lm(m)