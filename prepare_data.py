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
stats_df = pd.DataFrame(columns=['dataset', 'fb_type', 'slope', 'snr'])
for j_dataset, dataset in enumerate(datasets[:]):
    print(dataset)

    data = probes_df.query('dataset=="{}" '.format(dataset))
    y = []
    for block_number in data.query('block_name=="FB"')['block_number'].unique():
        y_j = data.query('block_number=={}'.format(block_number))['signal'].mean()
        y.append(y_j)

    y = np.array(y)
    lin_reg = stats.linregress(np.arange(len(y)), y)
    slope = lin_reg.slope*1e6
    #slope = np.max(y[len(y)//2:])*1e6 - np.min(y[:len(y)//2])*1e6

    #lin_reg = LinearRegression(fit_intercept=False)
    #lin_reg.fit(np.arange(len(y))[:, None], y)
    #slope = lin_reg.coef_[0]


    #plt.plot(y)
    #plt.plot(np.arange(len(y))*lin_reg.slope + lin_reg.intercept)
    #print(dataset, )

    freq, pxx = sg.welch(data.query('block_name=="Baseline0"')['p4'], 500, nperseg=500 * 2)

    alpha_pxx = pxx[(freq >= ALPHA_BAND[0]) & (freq <= ALPHA_BAND[1])]
    snr = pxx[(freq >= ALPHA_BAND[0]) & (freq <= ALPHA_BAND[1])].max() / pxx[(freq >= ALPHA_BAND_EXT[0]) & (freq <= ALPHA_BAND_EXT[1])].min()





    stats_df = stats_df.append({'dataset': dataset, 'fb_type': data['fb_type'].values[0], 'slope': slope, 'snr': snr}, ignore_index=True)



stats_df['logsnr'] = np.log10(stats_df['snr'])
plt.figure(dpi=200)
sns.lmplot('logsnr', 'slope', stats_df, 'fb_type')
#sns.scatterplot('snr', 'slope', 'fb_type', data=stats_df)


