# find_SARFI_multi.py
# Looking for Single Antenna RFI (SARFI) in COSMIC stamps corresponding to certain COSMIC hits
# now with multiprocessing
# Noah Stiegler
# 7/24/24
#
# run with $python3 find_SARFI_multi.py [filepath_to_target_hits]
#
# [path_to_target_hits] is the path leading to a file (like a .pkl or .csv file) containing
# the targets to look for SARFI in with one target on each row. Each row should have a column
# named file_uri which points to the COSMIC .stamps or .hits file corresponding to that target
# as well as a signal_frequency column which contains the frequency (in MHz). 
# 
# The script saves a new file to [filepath_to_target_hits]_with_antenna_info.pkl
# containing all the info in the original [filepath_to_target_hits] file as well as information 
# about how much of the signal is detected in each individual antenna (snrs and signals) for
# later analysis to pick out the SARFI
#
# Because the script can take a long time to run, it saves its work periodically

### Import Packages
import numpy as np
import pandas as pd
import pickle
import os
import socket
import argparse
from datetime import datetime
from seticore import viewer # If you don't have it, use pip install "git+https://github.com/MydonSolutions/seticore#egg=seticore&subdirectory=python"

# Ensure we're on the right server
hostname = socket.gethostname()
if hostname != "cosmic-gpu-1":
    raise Exception(f"On server {hostname}. Should be on cosmic-gpu-1")

### Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("filepath_to_target_hits")
parser.add_argument('-log', '--log_file_path', 
                    type=str, 
                    default="", 
                    help='Select where to save logged info about this script running. Default empty string saves a log to the same location as the target hits file')
parser.add_argument('-cf', '--checkpoint_frequency', 
                    type=int, 
                    default=1000, 
                    help='How often to save a checkpoint of the snrs and signals. Default is every 1000 stamps investigated')
parser.add_argument('-sfc', '--start_from_checkpoint',
                    type=str,
                    default="",
                    help='If starting from a checkpoint, the filepath of the saved checkpoint file. Default empty string means start from the beginning of the targets file')
args = parser.parse_args()

# Get arguments as variables
target_filepath = args.filepath_to_target_hits
log_filepath = args.log_file_path
checkpoint_frequency = args.checkpoint_frequency
checkpoint_file = args.start_from_checkpoint


### Setup log file
if log_filepath == "":
    log_filepath = os.path.splitext(target_filepath)[0] + "_log.txt"
if not os.path.exists(os.path.dirname(log_filepath)):
    raise Exception(f"Cannot open file {log_filepath}")
print(f"Logging to {log_filepath}")

# Setup for logging messages
def log_message(message):
    with open(log_filepath, 'a') as f:
        f.write(message + '\n')

# Print something and log it a the same time
def print_and_log(message):
    print(message)
    log_message(message)

### Load in targets
if target_filepath.endswith('.csv'):
    df = pd.read_csv(target_filepath)
elif target_filepath.endswith('.pkl'):
    df = pd.read_pickle(target_filepath)
else:
    raise Exception(f"Don't know how to read in the filepath_to_target_hits {target_filepath}")
print_and_log(f"Targets read in from {target_filepath}")

# Check the targets df has the right columns
if ('file_uri' not in df.columns) or ('signal_frequency' not in df.columns):
    raise Exception(f"Targets file must have columns 'file_uri' and 'signal_frequency'")

## Define helper functions for main script loop

# Given a URI and hit frequency finds the corresponding stamp object in the stamp file
# and will check if the stamp is in sequential stamp files as well (if #stamps > 500,
# seticore will save them as 0000.stamps, 0001.stamps, etc so you might miss the stamp
# if you don't check all the files)
# Parameter:
# - hit_uri - The uri from the hit (ending in .hits or .stamps)
# - frequency - The frequency from the hit (in MHz)
# - threshold - How close in frequency in hz the hit_frequency and the frequency of the stamp have
#               to be to say the hit is in the stamp file. Most stamp files are 1000Hz wide so
#               +/-500Hz from the center signal is the default
# Returns:
# - The stamp object corresponding to that hit in that stamp file (if it exists) otherwise None
def find_stamp_of_hit(hit_uri, hit_frequency, threshold=500):
    # Convert a filepath from pointing to a .hits
    # to a .stamps file for the same uri
    # If given a .stamps ending, doesn't change it
    def stamp_filepath_of(hits_filepath):
        return hits_filepath.replace('.hits', '.0000.stamps')
    
    # Look for the stamp in a single stamp file
    def find_stamp_in_single_file(hit_uri, hit_frequency, threshold=500):
        stamp_uri = stamp_filepath_of(hit_uri)
        stamps_gen = viewer.read_stamps(stamp_uri, find_recipe=True)
        for stamp in stamps_gen:
            assert(stamp != None)
            assert(stamp.recipe != None)
            if abs(stamp.stamp.signal.frequency - hit_frequency) < threshold*1e-6:
                # Found the stamp!
                return stamp
        
        # Didn't find the stamp in this whole file of stamps
        return None
    
    # Look for stamp in the 0000 stamp file
    stamp = find_stamp_in_single_file(hit_uri, hit_frequency, threshold)
    
    # Given a stamp_uri, increments the index of the stamp file by one (so if it's /.../...seticore.0000.stamps it goes to /.../...seticore.0001.stamps)
    def increment_stamp_uri(stamp_uri):
        split_uri = stamp_uri.split('.')
        assert(split_uri[-1] == 'stamps')
        num = int(split_uri[-2])
        num += 1
        split_uri[-2] = str(num).zfill(4)
        return ".".join(split_uri)
    
    # If not found in 0000 stamp file, look for it in the 0001 etc stamp files
    # until we either find it or run out of files
    stamp_uri = increment_stamp_uri(stamp_filepath_of(hit_uri))
    while (stamp == None) and (os.path.exists(stamp_uri)):
        # print("Looking in", stamp_uri)
        stamp = find_stamp_in_single_file(stamp_uri, hit_frequency, threshold)
        stamp_uri = increment_stamp_uri(stamp_uri)

    # If we've found the stamp or run out of files, return it
    return stamp

# Get the power and snr of the signal in each antenna of a stamp file
# Parameter:
# - stamp (the stamp in the stamp file)
# Returns:
# - (list of antenna snrs, list of signals signals)
def antenna_signal_snr_power(stamp):
    # Get the powers in the frequency bins of each antenna by summing 
    # over polarization and complex magnitude
    # Also rearrange so indices are (antenna, time bin, frequency bin)
    antenna_powers = np.square(stamp.real_array()).sum(axis=(2, 4)).transpose(2, 0, 1)
    snr_and_signals = np.array([stamp.snr_and_signal(antenna_power) for antenna_power in antenna_powers])
    return (snr_and_signals[:, 0], snr_and_signals[:, 1])

## Setup checkpoint stuff
# Load in from a checkpoint if we have one
item_to_start_on = 0
if checkpoint_file != "":
    print_and_log(f"Starting from checkpoint {checkpoint_file}")
    checkpoint = np.load(checkpoint_file, allow_pickle=True)
    if checkpoint.shape != (len(df), 2):
        raise Exception("Checkpoint file isn't the right shape")
    all_snrs = checkpoint[:, 0]
    all_signals = checkpoint[:, 1]
    print_and_log("Checkpoint loaded in!")

    # Figure out what number to start at
    def find_first_none(array):
        for i, item in enumerate(array):
            if type(item) == type(None):
                return i
    item_to_start_on = find_first_none(all_snrs)
else:
    print_and_log("Not starting from a checkpoint")
    # Setup containers for calculation
    all_snrs = np.full_like(df.file_uri, None, dtype='object')
    all_signals = np.full_like(df.file_uri, None, dtype='object')

# Save current work as a checkpoint
# Pass in the snrs and signals calculated so far
# Pass in which number file we're on
def save_checkpoint(all_snrs, all_signals, num_target):
    # Package data to save
    checkpoint = np.column_stack((all_snrs, all_signals))

    # Figure out where to save (in its own directory next to the data)
    dir_of_data = os.path.dirname(target_filepath) # /home/nstiegle/BL-COSMIC-2024-proj/stamps/representative_samples
    filename_of_data = os.path.basename(target_filepath) # name.csv
    filename_of_data = os.path.splitext(filename_of_data)[0] # name
    path_to_checkpoint_folder = dir_of_data + "/" + filename_of_data + "_ckpts/" # /home/nstiegle/BL-COSMIC-2024-proj/stamps/representative_samples/name_ckpts/
    
    # Make a directory for the checkpoints if it doesn't already exist
    if not os.path.exists(path_to_checkpoint_folder):
        os.makedirs(path_to_checkpoint_folder)
    path_to_save_checkpoint = path_to_checkpoint_folder + filename_of_data + f"_ckpt{num_target}.npy" # /home/.../name_ckpts/name_ckpt1000.npy
    
    # Save the data
    np.save(path_to_save_checkpoint, checkpoint)

### Main loop of script
# Calculate the antenna snrs and signals for each hit
print_and_log(f"Starting on target {item_to_start_on} at {datetime.now()}")
bad_stamp_uris = []
for i in range(item_to_start_on, len(df)):
    # Get the stamp file to check
    row = df.iloc[i]
    
    # Load in stamp from stamp file
    try: 
        stamp = find_stamp_of_hit(row.file_uri, row.signal_frequency)
    except:
        # Stamp file doesn't exist :/
        stamp = None
        bad_stamp_uris.append(row.file_uri)
        print_and_log(f"Row {i}'s stamp file couldn't be loaded: {row.file_uri}")

    # Check the stamp file
    if stamp != None:
        snrs, signals = antenna_signal_snr_power(stamp)
    else:
        print_and_log(f"Row {i} couldn't find the stamp in a stamp file :/")
        snrs = []
        signals = []

    # Save results
    all_snrs[i] = snrs
    all_signals[i] = signals

    # Save a checkpoint if we're saving one (and log about it)
    if (i != 0) and (i % checkpoint_frequency == 0):
        print_and_log(f"On target {i} at {datetime.now()}. Saving progress")
        save_checkpoint(all_snrs, all_signals, i)
        print_and_log(f"Done intermediate save for target {i} at {datetime.now()}")

# Log that we're done
print_and_log(f"Done with all targets at {datetime.now()}. Adding to target data to save back out.")

# Make new columns in the df for the antenna snrs and signals on each hit
df["antenna_snrs"] = all_snrs
df["antenna_signals"] = all_signals

# Save final results
df.to_pickle(os.path.splitext(target_filepath)[0] + "_with_antenna_info.pkl")
print_and_log(f"Saved and done!")

# Final logs
if len(bad_stamp_uris) > 0:
    print_and_log("Note that there were stamp URIs which caused errors when loading:")
    print_and_log("-"*80) # Horizontal line
    for stamp_uri in bad_stamp_uris:
        print_and_log(stamp_uri)
