import h5py
import scipy.signal as sg
import numpy as np
import glob
import os

FLANKER_WIDTH = 2
ALPHA_BAND_WIDTH = 2

data_folder = '/home/kolai/Data/alpha_delay2'
list_of_files = glob.glob(os.path.join(data_folder, '*'))
exp_name = os.path.split(max(list_of_files, key=os.path.getctime))[-1]

########## change exp name if needed (default - latest exp)

exp_name = exp_name

###########################################################

dataset = os.path.join(data_folder, exp_name, 'experiment_data.h5')
with h5py.File(dataset) as f:
    FS = int(f['fs'].value)
    p4_index = list(map(bytes.decode, f['channels'][:])).index('P4')
    x = f['protocol1/raw_data'][:][:, p4_index]
    # estimate snr
    freq, pxx = sg.welch(x, FS, nperseg=FS * 2)
    alpha_mask = (freq >= 8) & (freq <= 12)
    main_freq = freq[alpha_mask][np.argmax(pxx[alpha_mask])]
    band = (main_freq - ALPHA_BAND_WIDTH, main_freq + ALPHA_BAND_WIDTH)
    sig = pxx[(freq >= band[0]) & (freq <= band[1])].mean()
    noise = pxx[((freq >= band[0] - FLANKER_WIDTH) & (freq <= band[0])) | (
            (freq >= band[1]) & (freq <= band[1] + FLANKER_WIDTH))].mean()
    snr = sig / noise

    print('{}\noptBAND = [{} {}] Hz\nSNR = {:.3f}'.format(exp_name, *band, snr))