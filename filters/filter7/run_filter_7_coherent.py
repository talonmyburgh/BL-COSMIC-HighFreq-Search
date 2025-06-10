 # run_filter_7_coherent.py
# Runs filter 7 on all coherent data
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
log_filepath = script_dir + "/run_filter_7_coherent_log.txt"
# Setup for logging messages
def log_message(message):
    with open(log_filepath, 'a') as f:
        f.write(f"{datetime.now()}: {message}" + '\n')
# Print something and log it a the same time
def print_and_log(message):
    print(message)
    log_message(message)

### Read in the data
# Check which server we're on (in case the data is in different places on different servers)
# import socket
# hostname = socket.gethostname()

# # Get paths to data
# if hostname == "blpc1" or hostname == "blpc2":
#     data_path = "/datax/scratch/nstieg/"
# elif hostname == "cosmic-gpu-1":
#     data_path = "/mnt/cosmic-gpu-1/data0/nstiegle/"
# else:
#     raise Exception("Data path not known")

full_dataset_path = os.path.join(script_dir,"../../../highfrequency_hit_feb12024_apr302025_coherent_full.pkl")
coherent_dataset_path = full_dataset_path
print_and_log("Reading in coherent data from: " + coherent_dataset_path)
coherent_orig = pd.read_pickle(coherent_dataset_path)
good_indices_path = os.path.join(script_dir,"../filter6/run_filter_6_coherent_results.npy")
print_and_log("Reading in good indices from: " + good_indices_path)
good_indices = np.load(good_indices_path)
coherent = coherent_orig[coherent_orig.id.isin(good_indices)]

# Read in data
# coherent = pd.read_pickle(coherent_after_6_path)
# incoherent = pd.read_pickle(incoherent_dataset_path)
# df = pd.read_pickle(full_dataset_path)

### Run filter 7
high_snr = coherent[coherent.signal_snr > 10]
np.save(script_dir + "/run_filter_7_coherent_results", high_snr.id.values)