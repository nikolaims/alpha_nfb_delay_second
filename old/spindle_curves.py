import pandas as pd
import numpy as np
import seaborn as sns
import pylab as plt
from scipy import stats



FLANKER_WIDTH = 2
FS = 500
N_SAMPLES_PRE = 2*FS
N_SAMPLES_POST = 2*FS
N_SAMPLES_NORM = FS


probes_df = pd.read_pickle('envelopes.pkl')
datasets = probes_df['dataset'].unique()[:]


stats_df = pd.DataFrame(columns=['dataset', 'fb_type', 'spindle', 'snr', 'threshold_factor', 'time'])
for j_dataset, dataset in enumerate(datasets[:]):
    print(dataset)

    data = probes_df.query('dataset=="{}" '.format(dataset))
    snr = data['snr'].values[0]
    fb_type = data['fb_type'].values[0]


    median = data.query('block_name=="FB"')['online_envelope'].median()

    threshold_factors = [1.75]# np.arange(1, 3.1, 0.125)

    delay = {'FB0': 0, 'FB250': 250*FS//1000, 'FB500': 500*FS//1000, 'FBMock': 60*FS}
    signal = np.roll(data.query('block_name=="FB"')['online_envelope'].values, delay[fb_type])
    envelope = data.query('block_name=="FB"')['envelope'].values
    for threshold_factor in threshold_factors:
        threshold = threshold_factor * median
        spindles_mask = signal > threshold
        spindle_starts = np.where(np.diff(spindles_mask.astype(int)) == 1)[0]
        spindles = []
        for spindle_start in spindle_starts:
            if spindle_start < N_SAMPLES_PRE or spindle_start > len(signal)-N_SAMPLES_POST: continue
            spindle = envelope[spindle_start-N_SAMPLES_PRE: spindle_start+N_SAMPLES_POST]
            spindles.append(spindle)

        spindle = np.median(spindles, 0)
        spindle = (spindle )#/median

        #spindle /= spindle.max()
        stats_df = stats_df.append(pd.DataFrame(
            {'dataset': dataset, 'fb_type': fb_type, 'spindle': spindle,
             'snr': snr, 'threshold_factor': threshold_factor, 'time': np.arange(-N_SAMPLES_PRE, N_SAMPLES_POST)/FS}))





g = sns.relplot('time', 'spindle', data=stats_df, hue='fb_type', kind='line', ci=None, estimator=np.mean, hue_order=['FB0', 'FB250', 'FB500', 'FBMock'], col= 'threshold_factor')

[ax.axvline(0, color='k') for ax in g.axes.flatten()]