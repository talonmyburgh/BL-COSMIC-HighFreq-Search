# run_filter_5_coherent.py
# Runs filter 5 on all coherent data
# Noah Stiegler
# 7/29/24

### Import useful packages
import numpy as np
import pandas as pd
import os
from datetime import datetime
import argparse

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
log_filepath = script_dir + "/run_filter_5_coherent_log.txt"

def log_message(message):
    with open(log_filepath, 'a') as f:
        f.write(f"{datetime.now()}: {message}" + '\n')
# Print something and log it a the same time
def print_and_log(message):
    print(message)
    log_message(message)

### Setup for logging
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run filter 5 on coherent data.")
    parser.add_argument(
        "--coherent_dataset_path",
        type=str,
        default=None,
        help="Path to the coherent dataset pickle file."
    )
    args = parser.parse_args()

    if args.coherent_dataset_path:
        coherent_dataset_path = args.coherent_dataset_path
    else:
        # Default path if not provided
        coherent_dataset_path = os.path.join(
            script_dir, "../../../highfrequency_hit_feb12024_apr302025_coherent_full.pkl"
        )

    print_and_log("Reading in coherent data from: " + coherent_dataset_path)
    coherent_orig = pd.read_pickle(coherent_dataset_path)
    good_indices_path = os.path.join(script_dir, "../filter4/run_filter_4_coherent_results.npy")
    print_and_log("Reading in good indices from: " + good_indices_path)
    good_indices = np.load(good_indices_path)
    coherent = coherent_orig[coherent_orig.id.isin(good_indices)]

    # Run filter 5: keep only signal_frequency groups with a single entry
    groups = coherent.groupby("signal_frequency")
    filtered_groups = groups.filter(lambda x: len(x) == 1)
    np.save(script_dir + "/run_filter_5_coherent_results", filtered_groups.id.values)