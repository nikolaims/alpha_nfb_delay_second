import pandas as pd
import numpy as np
import scipy.signal as sg
import sys
import pylab as plt
import h5py
from mne.viz import plot_topomap

# import nfb lab data loader
sys.path.insert(0, '/home/kolai/Projects/nfblab/nfb')
from utils.load_results import load_data
from pynfb.inlets.montage import Montage

h5_file = '/media/kolai/SP UFD U3/alpha2-delay-subj-35_02-08_15-22-36/experiment_data.h5'

df, fs, channels, p_names = load_data(h5_file)