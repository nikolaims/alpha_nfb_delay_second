import pandas as pd
import numpy as np
from tqdm import tqdm
from release.settings import FS, CHANNELS, FB_ALL, ICA_CHANNELS
from release.utils import band_hilbert
import pylab as plt
from pynfb.signal_processing.filters import CFIRBandEnvelopeDetector, IdentityFilter

def opt_delay_corr(env, online_env):
    corrs = [np.corrcoef(np.roll(env, k), online_env)[1, 0] for k in range(50)]
    opt = np.argmax(corrs)
    return opt/FS*1000, corrs[opt]

# FS = 250
SPLIT_FB_BLOCKS = True
P4_ONLY = True
USE_PERCENTILES = False

# load pre filtered data
probes_df = pd.read_pickle('release/data/eeg_allsubjs_eyefree_1_45hz_down250hz.pkl')

# load datasets info
datasets_df = pd.read_pickle('release/data/info_allsubjs.pkl')



delays = []
corrds = []

delays2 = []
corrds2 = []
for subj_id in datasets_df['subj_id'].values[:]:

    # subj eeg
    data = probes_df.query('subj_id=="{}" '.format(subj_id))
    data = data.loc[data.block_number.isin(FB_ALL)]

    # subj fb type
    fb_type = datasets_df.query('subj_id=={}'.format(subj_id))['fb_type'].values[0]

    # subj band

    band = np.array(datasets_df.query('subj_id=={}'.format(subj_id))['band'].values[0])

    # compute envelope
    env = np.abs(band_hilbert(data['P4'].values, FS, band))
    online_env = data['online_envelope']

    delay, corr = opt_delay_corr(env, online_env)
    delays.append(delay)
    corrds.append(corr)

    online_env = CFIRBandEnvelopeDetector(band, FS, IdentityFilter(), n_taps=FS, delay_ms=60).apply(data['P4'].values)
    delay2, corr2 = opt_delay_corr(env, online_env)

    delays2.append(delay2)
    corrds2.append(corr2)
    print(subj_id)


import seaborn as sns
plt.scatter(corrds, delays, alpha=0.7, s=50)
plt.scatter(corrds2, delays2, alpha=0.7, s=50)


plt.plot()
