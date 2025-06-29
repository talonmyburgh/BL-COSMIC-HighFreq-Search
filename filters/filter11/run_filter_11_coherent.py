# run_filter_11_coherent.py
# Runs filter 11 on all coherent data
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
log_filepath = os.path.join(script_dir, "run_filter_11_coherent_log.txt")

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

good_indices_path = os.path.join(script_dir, "../filter10/run_filter_10_coherent_results.npy")
print_and_log(f"Reading in good indices from: {good_indices_path}")
good_indices = np.load(good_indices_path)
full_coherent = full_coherent[full_coherent.id.isin(good_indices)]

# Pass in row of dataframe for a single hit, get the error on that drift rate
def sigma_drift_rate(hit):
    # Error propagation on the error of the drift rate as dr =  df/dt (change in frequency / change in time)
    signal_dt = hit.tsamp * hit.signal_num_timesteps # Total number of seconds observed for
    signal_dr = hit.signal_drift_rate # Drift rate observed
    sigma_df = 2 # Error in measured frequency - 2Hz bins
    sigma_dt = hit.tsamp # Error in measured time - tsamp integration time per timestep
    if signal_dr == 0 or signal_dt == 0:
        return 0.0
    return abs((signal_dr / signal_dt) * np.sqrt((sigma_df/ signal_dr)**2 + (sigma_dt)**2)) # Error propagation formula for division substituting df = dr * dt

# Parameters of search
max_drift_time_to_search = 10 * 60 # in seconds

# Setup dataframe to flag hits which are validated by search
full_coherent["valid"] = False

# Do search within each source
for source_name, source_group in full_coherent.groupby('source_name'):
    # Group by time and figure out what all the times observed are
    if 'tstart_h' in source_group.columns:
        time_groups = source_group.groupby('tstart_h')
        time_names = list(time_groups.groups.keys())
    else:
        # fallback: use tstart if tstart_h is not present
        time_groups = source_group.groupby('tstart')
        time_names = list(time_groups.groups.keys())

    # Look at all times for this source which have a following time (all but the last)
    for t_idx in range(0, len(time_names) - 1):
        this_time = time_names[t_idx]
        next_time = time_names[t_idx + 1]
        # If using astropy Time objects, convert to datetime
        if hasattr(this_time, 'to_datetime'):
            this_time_dt = this_time.to_datetime()
            next_time_dt = next_time.to_datetime()
        elif isinstance(this_time, (float, int)):
            # Assume MJD if float, or Unix timestamp if int
            # Try MJD first (most likely for astronomy)
            try:
                from astropy.time import Time
                this_time_dt = Time(this_time, format='mjd').to_datetime()
                next_time_dt = Time(next_time, format='mjd').to_datetime()
            except Exception:
                # Fallback: treat as Unix timestamp
                this_time_dt = datetime.datetime.utcfromtimestamp(this_time)
                next_time_dt = datetime.datetime.utcfromtimestamp(next_time)
        else:
            this_time_dt = this_time
            next_time_dt = next_time
        dt = (next_time_dt - this_time_dt).total_seconds()

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
output_path = os.path.join(script_dir, "run_filter_11_coherent_results.npy")
np.save(output_path, results.values)
print_and_log(f"Saved results to: {output_path}")