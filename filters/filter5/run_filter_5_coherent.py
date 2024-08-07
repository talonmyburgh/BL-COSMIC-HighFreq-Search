# run_filter_5_coherent.py
# Runs filter 5 on all coherent data
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

# Read in data
coherent = pd.read_pickle(coherent_dataset_path)
# incoherent = pd.read_pickle(incoherent_dataset_path)
# df = pd.read_pickle(full_dataset_path)

### Run filter 5
groups = coherent.groupby("signal_frequency")
filtered_groups = groups.filter(lambda x: len(x) == 1) # Save all data in a group by itself
np.save(script_dir + "/run_filter_5_coherent_results", filtered_groups.id.values)