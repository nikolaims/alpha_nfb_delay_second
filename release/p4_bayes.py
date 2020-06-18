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

fb_types = ['FB0', 'FB250', 'FB500', 'FBMock', 'FBLow']
df = stats_df.loc[stats_df['fb_type'].isin(fb_types)]

# for subj in df['subj_id'].unique():
#     curve = df.loc[df['subj_id']==subj, 'metric'].values
#     curve /= curve.mean()
#     df.loc[df['subj_id']==subj, 'metric'] = curve

y = df['metric'].values
x = df['k'].values
subjects = df['subj_id'].map(dict(zip(df['subj_id'].unique(), np.arange(len(df['subj_id'].unique()))))).values
groups = df['fb_type'].map(dict(zip(df['fb_type'].unique(), np.arange(len(df['fb_type'].unique()))))).values
subjects_groups = [groups[subjects==s][0] for s in np.unique(subjects)]

import pymc3 as pm


with pm.Model() as model: # model specifications in PyMC3 are wrapped in a with-statement
    # Define priors
    # responder = pm.Bernoulli('respnder', p=0.8, shape=subjects.max() + 1)


    sigma_intercept_subject = pm.HalfCauchy('sigma_intercept_subject', beta=y[x==0].std())
    intercept = pm.Normal('intercept', y[x==0].mean(), sigma=y[x==0].std())
    intercept_subject = pm.Normal('intercept_subj', intercept, sigma=sigma_intercept_subject, shape=subjects.max()+1)

    slope_group = pm.Normal('slope_group', 0, sigma=(max(y)-min(y))/15, shape=groups.max() + 1)
    sigma_slope_subject = pm.HalfCauchy('sigma_slope_subject', beta=(max(y)-min(y))/15, shape=groups.max() + 1)
    slope_subject = pm.Normal('slope_subject', slope_group[subjects_groups], sigma=sigma_slope_subject[subjects_groups], shape=subjects.max() + 1)

    sigma = pm.HalfCauchy('sigma', beta=np.array([y[subjects==s].std() for s in np.unique(subjects)]), shape=subjects.max() + 1)

    # Define likelihood
    likelihood = pm.Normal('y', mu=intercept_subject[subjects] + slope_subject[subjects]*x,
                        sigma=sigma[subjects], observed=y)

    # Inference!
    trace = pm.sample(4000, cores=4) # draw 3000 posterior samples using NUTS sampling


# pm.plot_posterior(trace, var_names=['slope_subject'])
# pm.traceplot(trace)

from seaborn import kdeplot
[kdeplot(trace['slope_group'][:, k], label=fb_types[k]) for k in range(len(fb_types))]
plt.axvline(0, color='k', linestyle='--')
plt.xlabel('Slope')
plt.ylabel('PDF')

pm.model_to_graphviz(model).render()
