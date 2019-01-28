import pandas as pd
import numpy as np
import scipy.signal as sg
import sys
import pylab as plt
import seaborn as sns

# import nfblab data loader
sys.path.insert(0, '/home/kolai/Projects/nfblab/nfb')
from utils.load_results import load_data


# collect info
data_path = '/home/kolai/Data/alpha_delay2'
info = pd.read_csv('alpha_subject_2.csv')
datasets = [d for d in info['dataset'].unique() if d is not np.nan]


# collect probes
probes_df = pd.DataFrame(columns=['P4', 'Dataset', 'FBType', 'block_name'])
for j_dataset, dataset in enumerate(datasets[:]):
    dataset_path = '{}/{}/experiment_data.h5'.format(data_path, dataset)
    print(dataset_path)
    df, fs, channels, p_names = load_data(dataset_path)
    fb_type = df.query('block_number==6')['block_name'].values[0]
    for block_name in ['Close', 'Baseline0']:
        x = df.query('block_name=="{}"'.format(block_name))['P4'].values
        probes_df = probes_df.append(
            pd.DataFrame({'P4': x, 'Dataset': dataset, 'FBType': fb_type, 'block_name': block_name}), ignore_index=True)


# subj snr viz
ALPHA_BAND = (7, 15)
snr_df = pd.DataFrame(columns=['Dataset', 'FBType', 'SNR'])
colors = sns.color_palette('nipy_spectral', len(datasets))
plt.figure()
for j_dataset, dataset in enumerate(datasets[:]):
    x = probes_df.query('Dataset=="{}" & block_name=="Baseline0"'.format(dataset))['P4'].values
    fb_type = probes_df.query('Dataset=="{}"'.format(dataset))['FBType'].values[0]
    print(dataset, x)
    freq, pxx = sg.welch(x, fs, nperseg=fs*2)
    alpha_pxx = pxx[(freq>=ALPHA_BAND[0]) & (freq<=ALPHA_BAND[1])]
    snr = alpha_pxx.max()/alpha_pxx.min()
    snr_df = snr_df.append({'Dataset': dataset, 'FBType': fb_type, 'SNR': snr}, ignore_index=True)
    if snr > 20:
        plt.plot(freq, pxx, label='SNR={:.2f} {} {}'.format(snr, fb_type, dataset), color=colors[j_dataset])
plt.legend()


# viz SNRs vs FBType
plt.figure()
sns.swarmplot('FBType', 'SNR', data=snr_df, size=10)

