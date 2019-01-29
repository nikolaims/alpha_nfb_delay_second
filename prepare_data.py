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
    for block_name in ['Close', 'Baseline0', 'Baseline']:
        names = (df.query('block_name=="{}"'.format(block_name))['block_number'] > 4).astype(int) + 1
        if block_name == 'Baseline':
            names = block_name + names.astype(str)
        else:
            names = block_name
        x = df.query('block_name=="{}"'.format(block_name))['P4'].values
        probes_df = probes_df.append(
            pd.DataFrame({'P4': x, 'Dataset': dataset, 'FBType': fb_type, 'block_name': names}), ignore_index=True)


# subj snr viz
ALPHA_BAND = (7, 15)
snr_df = pd.DataFrame(columns=['Dataset', 'FBType', 'SNR', 'block_name'])
colors = sns.color_palette('nipy_spectral', len(datasets))
plt.figure()
for j_dataset, dataset in enumerate(datasets[:]):
    for block_name in probes_df['block_name'].unique():
        x = probes_df.query('Dataset=="{}" & block_name=="{}"'.format(dataset, block_name))['P4'].values
        fb_type = probes_df.query('Dataset=="{}"'.format(dataset))['FBType'].values[0]
        print(dataset, x)
        if len(x)==0: continue
        freq, pxx = sg.welch(x, fs, nperseg=fs*2)
        alpha_pxx = pxx[(freq>=ALPHA_BAND[0]) & (freq<=ALPHA_BAND[1])]
        snr = alpha_pxx.max()/alpha_pxx.min()
        snr_df = snr_df.append({'Dataset': dataset, 'FBType': fb_type, 'SNR': snr, 'block_name': block_name}, ignore_index=True)
        if snr < 4:
            plt.plot(freq, pxx, label='SNR={:.2f} {} {} {}'.format(snr, fb_type, dataset, block_name), color=colors[j_dataset])
plt.legend()


# viz SNRs vs FBType

sns.set_style("whitegrid")
#sns.pointplot('block_name', 'SNR', 'FBType', data=snr_df, dodge=True)
sns.factorplot('block_name', 'SNR', 'Dataset', col='FBType', data=snr_df.query('block_name!="Close" & block_name!="Baseline0"'), kind='point')

