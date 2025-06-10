# run_filter_1_coherent.py
# Runs filter 1 on all coherent data
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
log_filepath = script_dir + "/run_filter_1_coherent_log.txt"
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
# incoherent_dataset_path = data_path + "25GHz_higher_incoherent.pkl"

# Read in data
coherent = pd.read_pickle(coherent_dataset_path)
# incoherent = pd.read_pickle(incoherent_dataset_path)
# df = pd.read_pickle(full_dataset_path)
print_and_log("Coherent data read in correctly")

### Run filter 1
# The filter (for filter 1) to run on a single group of frequencies to see if they
# count as RFI or not
# Takes:
# - A df containing hits from a single source all at the same frequency
# Returns:
# - True if those hits don't count as RFI or false if they do
def pass_filter1_single_group(group):
    if len(group) == 1:
        # No other hit at this frequency, could be a genuine signal
        return True
    else:
        unique_drs = group.signal_drift_rate.unique()
        # Not all hits had 0 drift rate
        return (len(unique_drs) != 1) or (0 not in unique_drs)

# Apply filter one to a single source
# Extend COSMIC's ability to detect zero drift rate RFI
# Removes hits from a single target which are at exactly the same frequency as other hits 
# from that target and where all the hits detected at that frequency have a zero drift rate
# Parameters:
# - coherent: The df of coherent data
# Returns the ids of the hits which pass the filter
def filter1_single_source(source, name=None):
    source_good_ids = np.array([], dtype=int) # Put all the good ids from this source

    # Put things into groups of frequencies which are *exactly* the same
    groups = source.groupby("signal_frequency")
    for f, group in groups:
        if pass_filter1_single_group(group):
            source_good_ids = np.concatenate((source_good_ids, group.id.values))

    # Log and return
    if name != None:
        print_and_log("Done with source: " + name)
    return source_good_ids

if __name__ == "__main__":
    sources = coherent.groupby("source_name")
    inputs = [(source, source_name) for source_name, source in sources] 
    p = multiprocessing.Pool() 

    # Run algorithm with multiprocessing
    print_and_log("Running algorithm")
    results = p.starmap(filter1_single_source, inputs)
    print_and_log("Algorithm done. Saving")

    # Save results
    good_indices = np.array([], dtype=int)
    for result in results:
        good_indices = np.concatenate([good_indices, result])
    np.save(script_dir + "/run_filter_1_coherent_results", good_indices)
    print_and_log("Saved. Done!")