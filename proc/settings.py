from pynfb.inlets.montage import Montage
import pickle

FS_RAW = 500
FS = 250
CHANNELS = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FT9', 'FC5', 'FC1', 'FC2', 'FC6', 'FT10', 'C3', 'CZ', 'C4',
            'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'P4', 'P8', 'O1', 'OZ', 'O2', 'T7', 'PZ']
BLOCK_NAMES = [None, 'Close', 'Baseline', 'PauseBL', 'Baseline', 'PauseFB', 'FB', 'PauseFB', 'FB', 'PauseFB', 'FB',
               'PauseFB', 'FB', 'PauseFB', 'FB', 'PauseFB', 'FB', 'PauseFB', 'FB', 'PauseFB', 'FB', 'PauseFB', 'FB',
               'PauseFB', 'FB', 'PauseFB', 'FB', 'PauseFB', 'FB', 'PauseFB', 'FB', 'PauseFB', 'FB', 'PauseFB', 'FB',
               'PauseBL', 'Baseline']
FB_ALL = [k for k, name in enumerate(BLOCK_NAMES) if name=='FB']
CLOSE = 1
OPEN = 2
BASELINE_BEFORE = 4
BASELINE_AFTER = len(BLOCK_NAMES) - 1
ICA_BLOCKS = [CLOSE, OPEN, BASELINE_BEFORE, FB_ALL[0]]
MONTAGE = Montage(CHANNELS)
ICA_CHANNELS = ['ICA{}'.format(k + 1) for k in range(len(CHANNELS))]

def save_ica(ica, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(ica, output, pickle.HIGHEST_PROTOCOL)

def load_ica(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)