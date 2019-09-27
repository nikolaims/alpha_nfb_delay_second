import pandas as pd
import numpy as np
from tqdm import tqdm
from release.settings import FS, CHANNELS, FB_ALL, ICA_CHANNELS
from release.utils import band_hilbert, CFIRBandEnvelopeDetector




SPLIT_FB_BLOCKS = True
P4_ONLY = True
channel = 'P4'
threshold_factor = 2.5
N_SAMPLES_PRE = 3*FS
N_SAMPLES_POST = 2*FS
delay = {'FB0': 0, 'FB250': 250 * FS // 1000, 'FB500': 500 * FS // 1000, 'FBMock': 0*60 * FS}
time = np.arange(-N_SAMPLES_PRE, N_SAMPLES_POST) / FS


def find_spindles(env, spindles_mask):
    spindle_starts = np.where(np.diff(spindles_mask.astype(int)) == 1)[0]
    spindle_starts = spindle_starts[(spindle_starts >= N_SAMPLES_PRE) & (spindle_starts < len(env) - N_SAMPLES_POST)]
    spindles = np.zeros((len(spindle_starts), N_SAMPLES_POST + N_SAMPLES_PRE))
    for s, spindle_start in enumerate(spindle_starts):
        spindles[s] = env[spindle_start - N_SAMPLES_PRE:spindle_start + N_SAMPLES_POST]
    return spindles, spindle_starts


# load pre filtered data
probes_df = pd.read_pickle('release/data/eeg_allsubjs_eyefree_1_45hz_down250hz.pkl')

# load datasets info
datasets_df = pd.read_pickle('release/data/info_allsubjs.pkl')

# prepare data frame to save metrics
columns=['subj_id', 'fb_type', 'spindle', 'threshold_factor', 'time', 'half', 'detection_by']
stats_df = pd.DataFrame(columns=columns)
for subj_id in datasets_df['subj_id'].values[:]:
    # subj eeg
    data = probes_df.query('subj_id=="{}" '.format(subj_id))

    # subj fb type
    fb_type = datasets_df.query('subj_id=={}'.format(subj_id))['fb_type'].values[0]

    # subj band
    band = np.array(datasets_df.query('subj_id=={}'.format(subj_id))['band'].values[0])

    # block numbers utils
    block_numbers = data['block_number'].values

    ch_data = data.loc[data.block_number.isin(FB_ALL), channel].values
    env = np.abs(band_hilbert(ch_data, FS, band))
    # env = np.roll(np.abs(CFIRBandEnvelopeDetector(band, FS, 250).apply(ch_data)), -250)
    photo_data = data.loc[data.block_number.isin(FB_ALL), 'PHOTO'].values
    # photo_data = env

    # compute envelope
    spindles, spindle_starts = find_spindles(env, photo_data > threshold_factor * np.median(photo_data))
    spindle = np.mean(spindles[spindle_starts<len(env)//2], 0)
    norm = np.median(spindle[:FS])
    spindle1 = spindle/norm
    stats_df = stats_df.append(pd.DataFrame(
        {'subj_id': subj_id, 'fb_type': fb_type, 'spindle': spindle1,
         'threshold_factor': threshold_factor, 'time':time, 'half': 'first', 'detection_by': 'photo'}))

    spindle = np.mean(spindles[spindle_starts >= len(env) // 2], 0)
    spindle = spindle / norm
    stats_df = stats_df.append(pd.DataFrame(
        {'subj_id': subj_id, 'fb_type': fb_type, 'spindle': spindle,
         'threshold_factor': threshold_factor, 'time': time, 'half': 'second', 'detection_by': 'photo'}))

    spindle = np.mean(spindles, 0)
    spindle = spindle / norm
    stats_df = stats_df.append(pd.DataFrame(
        {'subj_id': subj_id, 'fb_type': fb_type, 'spindle': spindle,
         'threshold_factor': threshold_factor, 'time': time, 'half': 'both', 'detection_by': 'photo'}))

    # compute envelope
    spindles, spindle_starts = find_spindles(env, env > threshold_factor * np.median(env))
    spindle = np.mean(spindles[spindle_starts<len(env)//2], 0)
    norm = np.median(spindle[:FS])
    spindle1 = spindle/norm
    stats_df = stats_df.append(pd.DataFrame(
        {'subj_id': subj_id, 'fb_type': fb_type, 'spindle': spindle1,
         'threshold_factor': threshold_factor, 'time':time, 'half': 'first', 'detection_by': 'offline_env'}))

    spindle = np.mean(spindles[spindle_starts >= len(env) // 2], 0)
    spindle = spindle / norm
    stats_df = stats_df.append(pd.DataFrame(
        {'subj_id': subj_id, 'fb_type': fb_type, 'spindle': spindle,
         'threshold_factor': threshold_factor, 'time': time, 'half': 'second', 'detection_by': 'offline_env'}))

    spindle = np.mean(spindles, 0)
    spindle = spindle / norm
    stats_df = stats_df.append(pd.DataFrame(
        {'subj_id': subj_id, 'fb_type': fb_type, 'spindle': spindle,
         'threshold_factor': threshold_factor, 'time': time, 'half': 'both', 'detection_by': 'offline_env'}))

import seaborn as sns
sns.set_style("whitegrid")
g = sns.relplot('time', 'spindle', data=stats_df.query('detection_by=="offline_env" & half!="both"'), hue='fb_type', kind='line', ci=None, estimator=np.mean,
                hue_order=['FB0', 'FB250', 'FB500', 'FBMock'], style= 'half', row='detection_by',
                palette=['#6AC3E8', '#438BA8', '#005375', '#FE4A49'])

[ax.axvline(0, color='k') for ax in g.axes.flatten()]

g.axes.flatten()[0].set_xlim(-.5, 1)
g.axes.flatten()[0].set_ylabel('Mean envelope')
g.axes.flatten()[0].set_xlabel('Time, s')
g.axes.flatten()[0].set_title('')

import pylab as plt
plt.savefig('mean_spindle_by_offline_env.png', dpi=200)


g = sns.relplot('time', 'spindle', data=stats_df.query('detection_by=="photo" & half=="both"'), hue='fb_type', kind='line', ci=95, estimator=np.mean,
                hue_order=['FB0', 'FB250', 'FB500', 'FBMock'], col= 'half', row='detection_by',
                palette=['#6AC3E8', '#438BA8', '#005375', '#FE4A49'])

[ax.axvline(0, color='k') for ax in g.axes.flatten()]

g.axes.flatten()[0].set_xlim(-1.5, 0.5)
g.axes.flatten()[0].set_ylabel('Mean envelope')
g.axes.flatten()[0].set_xlabel('Time, s')
g.axes.flatten()[0].set_title('')

import pylab as plt
plt.savefig('mean_spindle_by_photo_conf_int.png', dpi=200)