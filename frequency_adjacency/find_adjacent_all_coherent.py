#!/home/nstieg/.conda/envs/cosmic/bin/python

## Import useful packages
import numpy as np
import pandas as pd
from scipy.sparse import csr_array
from scipy.sparse import lil_array
from scipy.sparse import save_npz
from scipy.sparse import load_npz
import os

## Read in the data
# Check which server we're on (in case the data is in different places on different servers)
import socket
hostname = socket.gethostname()

# Get paths to data
if hostname == "blpc1" or hostname == "blpc2":
    full_dataset_path = "/datax/scratch/nstieg/25GHz_higher.pkl"
    coherent_dataset_path = "/datax/scratch/nstieg/25GHz_higher_coherent.pkl"
    incoherent_dataset_path = "/datax/scratch/nstieg/25GHz_higher_incoherent.pkl"
else:
    raise Exception("Data path not known")

# Read in data
coherent = pd.read_pickle(coherent_dataset_path)
# incoherent = pd.read_pickle(incoherent_dataset_path)
# df = pd.read_pickle(full_dataset_path)

## Setup for logging messages
def log_message(log_path, message):
    with open(log_path, 'a') as f:
        f.write(message + '\n')

## Define algorithm to find distances
# Find adjacent points (like algorithm described above)
# data: pandas series with reset index (index goes 0...n-1 consecutively)
# window_width: width of window to find adjacency in same units as data (ie MHz and MHz)
# Returns:
# (distances, mask) tuple
# distances is a scipy sparse lil_array of float distances between adjacent points
# mask is a scipy sparse lil_array of booleans indicating whether 
def find_adjacent_distances(data, window_width, log_path):
    # Sort the data by frequency
    sdata = data.sort_values() # sfs is sorted frequencies
    # Make sure to keep track of the original indices
    original_indices = sdata.index # Maps index in sfs to index in fs

    # Setup empty arrays, indices are original indices of data
    num_hits = len(data)
    mask = lil_array((num_hits, num_hits), dtype=bool) 
    distances = lil_array((num_hits, num_hits), dtype=np.float32) 

    # Find which hits are adjacent in order. Stop if we find one that isn't
    for i, datum in enumerate(sdata):
        j = i + 1 # Index of point to compare to
        while ((j < num_hits) and # Don't index off the end of the array
               (abs(sdata.iloc[j] - datum) <= window_width)): 
            # Find coordinates in the non-sorted list
            u = original_indices[i]
            v = original_indices[j]

            # Make sure it's upper triangular (might have to flip over diagonal)
            if u > v: u, v = v, u # swap u and v
            
            # Set elements in matrix
            mask[u, v] = True
            distances[u, v] = abs(sdata.iloc[j] - datum)

            # Check next point
            j += 1
        
        # Log progress
        if (i % 100_000 == 0):
            log_message(log_path, f"On hit {i}")

    # Return data
    return distances, mask



## Run algorithm
# Set the threshold distance in hz to call two hits 'adjacent' and record their relative distances
threshold_hz = 1000
threshold = threshold_hz * 1e-6 # in MHz
path = "/home/nstieg/BL-COSMIC-2024-proj/frequency_adjacency/adjacent_in_coherent/" # Place to save arrays
distances_file_path = path + f'coherent_within_{round(threshold_hz)}hz.distances.npz'
mask_file_path = path + f'coherent_within_{round(threshold_hz)}hz.mask.npz'
log_path = path + f"log.txt"
if not (os.path.exists(distances_file_path) and os.path.exists(mask_file_path)):
    # Log progress
    log_message(log_path, "Starting calculation")
    
    # Compute results
    frequencies = coherent["signal_frequency"].copy().reset_index(drop=True)
    distances, mask = find_adjacent_distances(frequencies, threshold, log_path)

    # Log
    log_message(log_path, "Done with calculation, saving")
    
    # Save results to file
    save_npz(distances_file_path, csr_array(distances))
    save_npz(mask_file_path, csr_array(mask))

    # Log again
    log_message(log_path, "Script finishing")
