import pandas as pd
import numpy as np
import scipy.signal as sg
import sys
import pylab as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression

# import nfblab data loader
sys.path.insert(0, '/home/kolai/Projects/nfblab/nfb')
from utils.load_results import load_data


# collect info
data_path = '/home/kolai/Data/alpha_delay2'
info = pd.read_csv('alpha_subject_2.csv')
datasets = [d for d in info['dataset'].unique() if (d is not np.nan)
            and (info.query('dataset=="{}"'.format(d))['type'].values[0] in ['FB0', 'FBMock', 'FB250', 'FB500'])]


# collect recordings
probes_df = pd.DataFrame(columns=['signal',  'dataset', 'fb_type', 'block_name', 'block_number', 'p4'])
for j_dataset, dataset in enumerate(datasets[:]):
    dataset_path = '{}/{}/experiment_data.h5'.format(data_path, dataset)
    df, fs, channels, p_names = load_data(dataset_path)
    fb_type = df.query('block_number==6')['block_name'].values[0]
    print(dataset)
    df = df.loc[df['block_name'].isin(['Baseline0', 'Close', 'Baseline', 'FB0', 'FB250', 'FB500', 'FBMock'])]
    df['block_name'] = df['block_name'].apply(lambda x: 'FB' if 'FB' in x else x)
    th = df['signal_Alpha0'].quantile(0.99)*3
    print('{} outliers'.format(sum(df['signal_Alpha0'] > th)/fs))
    df = df.query('signal_Alpha0 < {}'.format(th))

    probes_df = probes_df.append(pd.DataFrame({'signal': df['signal_Alpha0'].values, 'dataset': dataset,
                                               'fb_type': fb_type, 'block_name': df['block_name'].values,
                                               'block_number': df['block_number'].values, 'p4': df['P4'].values}), ignore_index=True)

ALPHA_BAND = (9, 11)
ALPHA_BAND_EXT = (6, 16)
FS = 500
stats_df = pd.DataFrame(columns=['dataset', 'fb_type', 'metric', 'metric_type', 'block_number', 'snr'])
for j_dataset, dataset in enumerate(datasets[:]):
    print(dataset)

    data = probes_df.query('dataset=="{}" '.format(dataset))
    th = data.query('block_name=="FB"')['signal'].median()*2
    y = []
    n_spindles = []
    duration = []
    amplitude = []

    for block_number in data.query('block_name=="FB"')['block_number'].unique():
        signal = data.query('block_number=={}'.format(block_number))['signal']
        y_j = signal.mean()*1e6
        y.append(y_j)


        y_j = sum(np.diff((signal.values > th).astype(int)) == 1)
        n_spindles.append(y_j)

        y_j = sum((signal.values > th))/y_j/FS
        duration.append(y_j)

        amplitude.append(signal.values[signal.values > th].mean()*1e6)



    freq, pxx = sg.welch(data.query('block_name=="Baseline0"')['p4'], 500, nperseg=500 * 2)

    alpha_pxx = pxx[(freq >= ALPHA_BAND[0]) & (freq <= ALPHA_BAND[1])]
    snr = pxx[(freq >= ALPHA_BAND[0]) & (freq <= ALPHA_BAND[1])].max() / pxx[(freq >= ALPHA_BAND_EXT[0]) & (freq <= ALPHA_BAND_EXT[1])].min()



    stats_df = stats_df.append(pd.DataFrame(
        {'dataset': dataset, 'fb_type':  data['fb_type'].values[0],
         'metric': y, 'metric_type': 'magnitude', 'block_number': np.arange(len(y))+1, 'snr': snr}))


    stats_df = stats_df.append(pd.DataFrame(
        {'dataset': dataset, 'fb_type':  data['fb_type'].values[0],
         'metric': n_spindles, 'metric_type': 'n_spindles', 'block_number': np.arange(len(y))+1, 'snr': snr}))


    stats_df = stats_df.append(pd.DataFrame(
        {'dataset': dataset, 'fb_type':  data['fb_type'].values[0],
         'metric': duration, 'metric_type': 'spindle_duration', 'block_number': np.arange(len(y))+1, 'snr': snr}))

    stats_df = stats_df.append(pd.DataFrame(
        {'dataset': dataset, 'fb_type':  data['fb_type'].values[0],
         'metric': amplitude, 'metric_type': 'spindle_amplitude', 'block_number': np.arange(len(y))+1, 'snr': snr}))



#stats_df['magnitude'] *= 1e6


stats_df['logsnr'] = np.log10(stats_df['snr'])
sns.relplot('block_number', 'metric', hue='logsnr', units='dataset', col='fb_type', row='metric_type', data=stats_df,
            kind='line', estimator=None, col_order=['FB0', 'FB250', 'FB500', 'FBMock'], palette='viridis_r', facet_kws={'sharey':'row'})

sns.relplot('block_number', 'metric', col='fb_type', row='metric_type', data=stats_df,
            kind='line',  col_order=['FB0', 'FB250', 'FB500', 'FBMock'], palette='viridis_r', facet_kws={'sharey':'row'})
#plt.savefig('rt_signal_stats.png', dpi=100)
#sns.relplot('block_number', 'magnitude', col='fb_type', data=stats_df, kind='line', col_order=['FB0', 'FB250', 'FB500', 'FBMock'], color='k', fig=plt.gcf()))

