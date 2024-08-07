 # run_filter_8_coherent.py
# Runs filter 8 on all coherent data
# Noah Stiegler
# 7/29/24

### Import useful packages
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
import multiprocessing

### Setup for logging
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

### Read in the data
# Check which server we're on (in case the data is in different places on different servers)
import socket
hostname = socket.gethostname()

# Get paths to data
if hostname == "blpc1" or hostname == "blpc2":
    data_path = "/datax/scratch/nstieg/"
elif hostname == "cosmic-gpu-1":
    data_path = "/mnt/cosmic-gpu-1/data0/nstiegle/"
else:
    raise Exception("Data path not known")

full_dataset_path = data_path + "25GHz_higher.pkl"
coherent_dataset_path = data_path + "25GHz_higher_coherent.pkl"
coherent_after_7_path = data_path + "25GHz_higher_coherent_post_filter7.pkl"

# Read in data
coherent = pd.read_pickle(coherent_after_7_path)
# incoherent = pd.read_pickle(incoherent_dataset_path)
# df = pd.read_pickle(full_dataset_path)

### Run filter 8
not_crazy_high_snr = coherent[coherent.signal_snr < 100]
np.save(script_dir + "/run_filter_8_coherent_results", not_crazy_high_snr.id.values)