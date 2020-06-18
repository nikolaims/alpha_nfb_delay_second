import numpy as np
import pylab as plt
import pandas as pd
from scipy.stats import rankdata, linregress

# one subj
def get_stat(x):
    y = rankdata(x)
    return y[8:].sum() - y[:7].sum()

n_points = 15
x = np.random.normal(size=n_points) + np.arange(n_points)*0.1
y = rankdata(x)

fig, axes = plt.subplots(2)
axes[0].plot(x)
axes[1].plot(y)
axes[1].legend(['{} {}'.format(get_stat(x), '*' if np.abs(get_stat(x))>=33 else '')])


n_boot = 10000
stats = [get_stat(np.random.normal(size=n_points) + np.arange(n_points)*0) for k in range(n_boot)]
np.percentile(stats, 97.5)
np.percentile(stats, 2.5)

# mul subjs
n_subjs=10
x1 = np.random.normal(size=(n_subjs, n_points)) + np.arange(n_points)*0.1
y = np.array([rankdata(x_) for x_ in x1]).mean(0)

def get_stat_mul(x):
    y = np.array([rankdata(x_) for x_ in x]).mean(0)
    return y[8:].sum() - y[:7].sum()
print(get_stat_mul(x1))

stats1 = [get_stat_mul(np.random.normal(size=(n_subjs-2, n_points)) + np.arange(n_points)*0) -
          get_stat_mul(np.random.normal(size=(n_subjs, n_points))) for k in range(n_boot)]
np.percentile(stats1, 97.5)
a = plt.hist(stats1, np.arange(-50, 50))

print(get_stat_mul(np.random.normal(size=(n_subjs-2, n_points)) + np.arange(n_points)*0.3) - get_stat_mul(np.random.normal(size=(n_subjs, n_points))))

# real data
from release.settings import FB_ALL
stats_file = 'FBLow_channels1_bands1_splitedFalse_median_threshs20.pkl'
stats_df = pd.read_pickle('release/data/{}'.format(stats_file))
stats_df = stats_df.loc[stats_df['block_number'].isin(FB_ALL)]
unique_blocks = list(stats_df['block_number'].unique())
stats_df['k'] = stats_df['block_number'].apply(lambda x: unique_blocks.index(x))

data = stats_df.query('threshold_factor==2.5 & metric_type=="magnitude"')

fb1_type = 'FB0'
fb2_type = 'FBMock'
fb1_df = data.query('fb_type=="{}"'.format(fb1_type))
fb2_df = data.query('fb_type=="{}"'.format(fb2_type))


def get_stat_mul(x):
    slopes = np.zeros(x.shape[0])
    for j, xs in enumerate(x):
        lr = linregress(np.arange(15), xs)
        slopes[j] = xs.mean()/lr.intercept*100-100
    return np.mean(slopes)


def get_perm_pvalue(fb1_df, fb2_df, n_perm=1000):
    fb1_x = np.array([subj_df['metric'].values for _, subj_df in fb1_df.groupby('subj_id')])
    fb2_x = np.array([subj_df['metric'].values for _, subj_df in fb2_df.groupby('subj_id')])
    all_x = np.vstack([fb1_x, fb2_x])
    print(get_stat_mul(fb1_x), get_stat_mul(fb2_x))
    obs = get_stat_mul(fb1_x) - get_stat_mul(fb2_x)
    stats2 = np.zeros(n_perm)
    for j_perm in range(n_perm):
        indx = np.random.permutation(all_x.shape[0])
        fb1_x_perm = all_x[indx[:fb1_x.shape[0]]]
        fb2_x_perm = all_x[indx[fb1_x.shape[0]:]]
        stats2[j_perm] = get_stat_mul(fb1_x_perm) - get_stat_mul(fb2_x_perm)
    return obs, sum(stats2 > obs) / n_perm

print(get_perm_pvalue(fb1_df, fb2_df))
