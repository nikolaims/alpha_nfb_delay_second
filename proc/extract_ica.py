from pynfb.signal_processing.decompositions import ICADecomposition

from proc.settings import FS, CHANNELS, MONTAGE, ICA_BLOCKS
import pandas as pd
import numpy as np
import seaborn as sns
import pylab as plt
from scipy import stats
from tqdm import tqdm
from time import time
import pickle



probes_df = pd.read_pickle('data/eeg_allsubjs_eyefree_1_45hz_down250hz.pkl')
datasets_df = pd.read_pickle('data/info_allsubjs.pkl')

for subj_id in tqdm(datasets_df['subj_id'].values[:]):
    data = probes_df.query('subj_id=="{}" '.format(subj_id))
    ica_data = data.loc[data['block_number'].isin(ICA_BLOCKS)][CHANNELS].values
    ica = ICADecomposition(CHANNELS, FS)
    ica.fit(ica_data)
    save_ica(ica, 'data/ica/s{}ica.pkl'.format(subj_id))

