import pandas as pd
import numpy as np
from tqdm import tqdm
from release.settings import FS, CHANNELS, FB_ALL, ICA_CHANNELS
from release.utils import band_hilbert
import seaborn as sns

def opt_delay_corr(env, online_env):
    corrs = [np.corrcoef(np.roll(env, k), online_env)[1, 0] for k in range(200)]
    opt = np.argmax(corrs)
    return opt/FS*1000, corrs[opt]

# load pre filtered data
probes_df = pd.read_pickle('release/data/P4FBLow_eeg_allsubjs_eyefree_1_45hz_down250hz.pkl')

# load datasets info
datasets_df = pd.read_pickle('release/data/info_allsubjs.pkl')
datasets_df['opt_delay'] = 0
datasets_df['opt_corr'] = 0
datasets_df['open'] = 0
datasets_df['close'] = 0
datasets_df['fb1'] = 0

for subj_id in datasets_df['subj_id'].values[:]:
    print(subj_id)
    # subj eeg
    data = probes_df.query('subj_id=="{}" '.format(subj_id))
    # data = data.loc[data.block_number.isin(FB_ALL)]

    # subj fb type
    fb_type = datasets_df.query('subj_id=={}'.format(subj_id))['fb_type'].values[0]
    band = np.array(datasets_df.query('subj_id=={}'.format(subj_id))['band'].values[0])

    # compute envelope
    env = np.abs(band_hilbert(data['P4'].loc[data.block_number.isin(FB_ALL)].values, FS, band))
    online_env = data['PHOTO'].loc[data.block_number.isin(FB_ALL)].values
    close_env = np.abs(band_hilbert(data['P4'].loc[data.block_number.isin([1])].values, FS, band))
    open_env = np.abs(band_hilbert(data['P4'].loc[data.block_number.isin([2, 4])].values, FS, band))
    fb1_env = np.abs(band_hilbert(data['P4'].loc[data.block_number.isin([6])].values, FS, band))
    fb1_env = fb1_env[:len(fb1_env)//2]


    delay, corr = opt_delay_corr(env, online_env)
    datasets_df.loc[datasets_df['subj_id'] == subj_id, 'opt_delay'] = delay
    datasets_df.loc[datasets_df['subj_id'] == subj_id, 'opt_corr'] = corr
    datasets_df.loc[datasets_df['subj_id'] == subj_id, 'close'] = close_env.mean()*1e6
    datasets_df.loc[datasets_df['subj_id'] == subj_id, 'open'] = open_env.mean()*1e6
    datasets_df.loc[datasets_df['subj_id'] == subj_id, 'fb1'] = fb1_env.mean()*1e6

sns.swarmplot('fb_type', 'fb1', data=datasets_df, order=['FBLow', 'FB0', 'FB250', 'FB500', 'FBMock'])
import pylab as plt
plt.ylim(0, 14)
plt.figure()

sns.pairplot(datasets_df[['snr', 'open', 'close', 'fb1', 'fb_type']], hue='fb_type', hue_order=['FBLow', 'FB0', 'FB250', 'FB500', 'FBMock'])