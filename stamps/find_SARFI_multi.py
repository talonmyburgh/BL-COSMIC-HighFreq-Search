# Import useful packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime, timedelta
from seticore import viewer, hit_capnp, stamp_capnp
import traceback
import multiprocessing

# Load in the indexed targets
targets_indexed = pd.read_csv("/mnt/cosmic-gpu-1/data0/nstiegle/representative_samples/1in25_targets_indexed.csv") # Read in indexed his from above

# Take out the files which didn't have .hits files
good_targets = targets_indexed.dropna()

def find_stamp_recipe(stamp_filepath, directory_path=None):
    """
    Get the Recipe for the BFR5 that matches the given stamp
    (just the filepath that is most similar and ends with .bfr5).
    """
    if directory_path is None:
        directory_path = os.path.dirname(stamp_filepath)

    closest_bfr5 = None
    closest_commonlen = 0
    for root, dirs, files in os.walk(directory_path, topdown=True):
        for f in filter(lambda x: x.endswith("bfr5"), files):
            filepath = os.path.join(root, f)
            commonpath = os.path.commonpath([filepath, stamp_filepath])
            commonlen = len(commonpath)
            if commonlen > closest_commonlen:
                closest_bfr5 = filepath
                closest_commonlen = commonlen
        break

    if closest_bfr5 is None:
        return None
    try:
        return viewer.Recipe(closest_bfr5)
    except BaseException as err:
        print(f"Error encountered instantiating Recipe from '{closest_bfr5}': {err}")
        print(traceback.format_exc())
        return False

def get_stamp(stamp_uri, frameidx):
    # Load up the stamp
    with open(stamp_uri, 'r') as f:
        f.seek(8 * (frameidx - 1)) # -1 for Julia being 1 indexed, *8 for some bit/byte reason
        s = stamp_capnp.Stamp.read(f)
        recipe = find_stamp_recipe(stamp_uri)
        return viewer.Stamp(s, recipe)

def antenna_signal_snr_power(stamp):
    # Get the powers in the frequency bins of each antenna by summing 
    # over polarization and complex magnitude
    # Also rearrange so indices are (antenna, time bin, frequency bin)
    antenna_powers = np.square(stamp.real_array()).sum(axis=(2, 4)).transpose(2, 0, 1)
    snr_and_signals = np.array([stamp.snr_and_signal(antenna_power) for antenna_power in antenna_powers])
    return (snr_and_signals[:, 0], snr_and_signals[:, 1])

def find_sarfi(stamp_uri, frameidx):
    stamp = get_stamp(stamp_uri, frameidx)
    snrs, signals = antenna_signal_snr_power(stamp)
    antenna_titles = [stamp.recipe.antenna_names[i] for i in range(stamp.stamp.numAntennas)]
    return antenna_titles, snrs, signals

# Setup for multiprocessing
inputs = []
for i, row in good_targets.iterrows():
    inputs.append((row.stamp_uri, row.frameidx))
p = multiprocessing.Pool() 

# Run algorithm with multiprocessing
results = p.starmap(find_sarfi, inputs)

# Save results
antenna_titles = np.array([results[0] for result in results], dtype='str')
antenna_snrs = np.array([results[1] for result in results], dtype='object')
antenna_signals = np.array([results[2] for result in results], dtype='object')
good_targets["antenna_titles"] = antenna_titles
good_targets["antenna_snrs"] = antenna_snrs
good_targets["antenna_signals"] = antenna_signals
good_targets.to_csv("/mnt/cosmic-gpu-1/data0/nstiegle/representative_samples/1in25_good_targets_results")