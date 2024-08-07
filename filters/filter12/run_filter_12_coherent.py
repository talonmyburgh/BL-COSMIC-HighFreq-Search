 # run_filter_12_coherent.py
# Runs filter 12 on all coherent data
# Noah Stiegler
# 7/29/24

### Import useful packages
import numpy as np
import pandas as pd
import os
import socket
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

from astropy.time import Time

# Get the coherent dataset but with all the columns
full_coherent = pd.read_pickle(data_path + "25GHz_higher_coherent_all_columns.pkl")

# Reject hits which have fewer than 16 timesteps and an snr less than 15
enough_timesteps = full_coherent.num_timesteps >= 16
enough_snr = full_coherent.signal_snr >= 15
full_coherent = full_coherent[enough_timesteps & enough_snr]

# Save results
results = full_coherent.id
np.save("/home/nstieg/BL-COSMIC-2024-proj/filters/filter12/run_filter_12_coherent_results.npy", results.values)