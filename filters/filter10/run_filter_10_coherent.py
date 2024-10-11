 # run_filter_10_coherent.py
# Runs filter 10 on all coherent data
# Noah Stiegler
# 7/29/24

### Import useful packages
import numpy as np
import pandas as pd
import os
import socket
from datetime import datetime, timedelta
import multiprocessing
from scipy.sparse import csr_array
from scipy.sparse import lil_array
from scipy.sparse import save_npz
from scipy.sparse import load_npz

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
coherent_after_9_path = data_path + "25GHz_higher_coherent_post_filter9.pkl"

# Read in data
coherent = pd.read_pickle(coherent_dataset_path)
coherent_after_9 = pd.read_pickle(coherent_after_9_path)
# incoherent = pd.read_pickle(incoherent_dataset_path)
# df = pd.read_pickle(full_dataset_path)

# Read in adjacency information
adjacency_path = os.path.expanduser("~") + "/BL-COSMIC-2024-proj/frequency_adjacency/adjacent_in_coherent/"
distances_path = adjacency_path + "coherent_within_1000hz.distances.npz"
mask_path = adjacency_path + "coherent_within_1000hz.mask.npz"
distances = load_npz(distances_path)
mask = load_npz(mask_path)
assert(distances.shape[0] == coherent.shape[0])
assert(mask.shape[0] == coherent.shape[0])

### Run filter 8
# Let's look at how many collisions there were within a threshold
# Want to do (distances <= threshold) & mask to get all the values which are under the threshold
# including zeroes which are let in by the mask
# However, if we turn all the 0s in the sparse array to Trues, it's not sparse anymore (so it's super slow)
# So how do we maintain the sparsity?
# Well, we can look for the opposite first, the values which are outside the threshold
# Then we can remove those from the mask so there's a new mask just contains the values we care about below the threshold 
# Then the number of collisions within that threshold is the number of Trues in the new mask
# And if we want to get those distances (or the hits which are close to each other), then we can do distances[new_mask] (or new_mask.nonzero())
def find_collisions_at_threshold(threshold):
    # Get those outside threshold
    outside_threshold = distances > threshold

    # Remove those outside threshold from the mask
    # We want to do mask & (~outside_threshold), but note that ~outside_threshold produces an array which is mostly true
    # So instead we'll have to do mask - outside_threshold 
    # (which does xor, so it will have values which are in mask but not outside_threshold, and those in outside_threshold but not mask)
    # Finally, to get rid of values in outside_threshold but not mask, we'll and by mask
    new_mask = mask.multiply(mask - outside_threshold)

    return new_mask

# Find all hits which are within 10hz of another hit
within_10_hz = find_collisions_at_threshold(10e-6)

# Get the indices of those hits
indices_of_hits_within_10hz = np.unique(np.concatenate(within_10_hz.nonzero()))

# Get a list of booleans which are true for hits which have another hit within 10hz
hit_with_close_neighbor = np.zeros(len(coherent), dtype=bool)
hit_with_close_neighbor[indices_of_hits_within_10hz] = True

# Get list of booleans which are true for hits with no other hits within 10hz of them
hits_with_no_close_neighbor = ~hit_with_close_neighbor

# Then index the coherent dataframe at those locations
lonely_hits = coherent.iloc[hits_with_no_close_neighbor]
np.save(script_dir + "/results_on_all_coherent", lonely_hits.id.values)

# Make sure to take out the hits which were taken out by previous filters
filtered_lonely_hits = pd.merge(lonely_hits, coherent_after_9, on='id', how='inner')

# Save result
np.save(script_dir + "/run_filter_10_coherent_results", filtered_lonely_hits.id.values)