import pandas as pd
import numpy as np
import seaborn as sns
import pylab as plt
from scipy import stats



FLANKER_WIDTH = 2
FS = 500


probes_df = pd.read_pickle('envelopes.pkl')
datasets = probes_df['dataset'].unique()


stats_df = pd.DataFrame(columns=['dataset', 'fb_type', 'metric', 'metric_type', 'block_number', 'snr', 'threshold_factor'])
for j_dataset, dataset in enumerate(datasets[:]):
    print(dataset)

    data = probes_df.query('dataset=="{}" '.format(dataset))
    data['signal'] = data['envelope']
    snr = data['snr'].values[0]


    median = data.query('block_name=="FB"')['signal'].median()

    threshold_factors = np.arange(1.25, 3.1, 0.25)

    for block_number_j, block_number in enumerate(data.query('block_name=="FB"')['block_number'].unique()):
        for threshold_factor in threshold_factors:
            threshold = threshold_factor * median
            signal = data.query('block_number=={}'.format(block_number))['signal'].values


            magnitude_j = signal.mean() * 1e6
            spindles_mask = signal > threshold
            n_spindles_j = sum(np.diff(spindles_mask.astype(int)) == 1)
            duration_j = sum(spindles_mask) / n_spindles_j / FS
            amplitude_j = signal[spindles_mask].mean() * 1e6

            stats_df = stats_df.append(pd.DataFrame(
                {'dataset': dataset, 'fb_type': data['fb_type'].values[0],
                 'metric': [magnitude_j, n_spindles_j, duration_j, amplitude_j],
                 'metric_type': ['magnitude', 'n_spindles', 'duration', 'amplitude'],
                 'block_number': block_number_j+1, 'snr': snr, 'threshold_factor': threshold_factor}))



stats_df.to_csv('spindles_stats_norm.csv', index=False)

