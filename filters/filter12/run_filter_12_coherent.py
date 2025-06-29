# run_filter_12_coherent.py
# Runs filter 12 on all coherent data
# Noah Stiegler
# 7/29/24

### Import useful packages
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

### Setup for logging
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
log_filepath = os.path.join(script_dir, "run_filter_12_coherent_log.txt")

def log_message(message):
    with open(log_filepath, 'a') as f:
        f.write(f"{datetime.now()}: {message}" + '\n')
# Print something and log it at the same time
def print_and_log(message):
    print(message)
    log_message(message)

### Read in the data
full_dataset_path = os.path.abspath(os.path.join(script_dir, "../../../highfrequency_hit_feb12024_apr302025_coherent_full.pkl"))
print_and_log(f"Reading in coherent data from: {full_dataset_path}")
full_coherent = pd.read_pickle(full_dataset_path)

good_indices_path = os.path.join(script_dir, "../filter11/run_filter_11_coherent_results.npy")
print_and_log(f"Reading in good indices from: {good_indices_path}")
good_indices = np.load(good_indices_path)
full_coherent = full_coherent[full_coherent.id.isin(good_indices)]

# Apply filter: if num_timesteps < 16, require signal_snr > 15; otherwise, keep all
mask = ((full_coherent.num_timesteps < 16) & (full_coherent.signal_snr > 15)) | (full_coherent.num_timesteps >= 16)
filtered = full_coherent[mask]

# Save results
output_path = os.path.join(script_dir, "run_filter_12_coherent_results.npy")
np.save(output_path, filtered.id.values)
print_and_log(f"Saved results to: {output_path}")