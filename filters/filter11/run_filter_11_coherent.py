 # run_filter_11_coherent.py
# Runs filter 11 on all coherent data
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

# Pass in row of dataframe for a single hit, get the error on that drift rate
def sigma_drift_rate(hit):
    # Error propagation on the error of the drift rate as dr =  df/dt (change in frequency / change in time)
    signal_dt = hit.tsamp * hit.signal_num_timesteps # Total number of seconds observed for
    signal_dr = hit.signal_drift_rate # Drift rate observed
    sigma_df = 2 # Error in measured frequency - 2Hz bins
    sigma_dt = hit.tsamp # Error in measured time - tsamp integration time per timestep
    return abs((signal_dr / signal_dt) * np.sqrt((sigma_df/ signal_dr)**2 + (sigma_dt)**2)) # Error propagation formula for division substituting df = dr * dt

# TOOD / FUTURE WORK
# - Figure out what the distribution of dts are for sources
# - The results of this search approach are naturally described by a directed graph (or collection of trees)
#   where nodes are hits and hits point to hits they may have drifted to. By following 'chains' in this directed
#   graph you might find hits which continue to drift for some time or change drift rate (maybe sinusoidally?)

# Parameters of search
max_drift_time_to_search = 10 * 60 # in seconds

# Setup dataframe to flag hits which are validated by search
full_coherent["valid"] = False

# Do search within each source
for source_name, source_group in full_coherent.groupby('source_name'):
    # Group by time and figure out what all the times observed are
    time_groups = source_group.groupby('tstart_h')
    time_names = list(time_groups.groups.keys())

    # Look at all times for this source which have a following time (all but the last)
    for t_idx in range(0, len(time_names) - 1):
        this_time = time_names[t_idx]
        next_time = time_names[t_idx + 1]
        dt = (next_time - this_time).total_seconds()

        # If the source was observed again within 10 minutes, look for
        # signals which drifted in the next observation time
        if dt <= max_drift_time_to_search:
            time_group = time_groups.get_group(this_time)
            next_time_group = time_groups.get_group(next_time)
            for i, hit in time_group.iterrows():
                # Ignore zero drift rate signals
                if hit.signal_drift_rate != 0:
                    # Compute some useful quantities
                    drift = (dt * hit.signal_drift_rate) * 1e-6 # Total drift in MHz
                    sigma_drift = max((dt * sigma_drift_rate(hit)) * 1e-6, 2 * 1e-6) # Error in drift in Mhz
                    expected_new_frequency = hit.signal_frequency + drift # Where we expect it to drift to in Mhz

                    # Get candidate hits from the next time
                    candidates = next_time_group[(next_time_group.signal_frequency > expected_new_frequency - sigma_drift) &
                                                (next_time_group.signal_frequency < expected_new_frequency + sigma_drift)]
                    
                    # If there was a match between this hit and a hit in the target range, validate them both in the full dataset
                    full_coherent.loc[full_coherent.id == hit.id, 'valid'] = True
                    full_coherent.loc[candidates.index, 'valid'] = True

results = full_coherent["id"][full_coherent["valid"]]
np.save("/home/nstieg/BL-COSMIC-2024-proj/filters/filter11/run_filter_11_coherent_results.npy", results.values)