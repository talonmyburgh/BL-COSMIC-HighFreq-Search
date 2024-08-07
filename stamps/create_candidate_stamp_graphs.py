# Import packages needed
import pandas as pd
import os
os.environ["H5PY_DEFAULT_READONLY"] = "1" # Surpress h5py deprecation warnings
from seticore import viewer # If you don't have it, use pip install "git+https://github.com/MydonSolutions/seticore#egg=seticore&subdirectory=python"
import multiprocessing


# Read in candidate hits with stamp uri and indexes on them
candidates = pd.read_csv("/home/nstiegle/candidates_indexed.csv")

# Get all the info to plot each stamp
inputs = [(row.stamp_uri, row.stamp_index, row.source_name, row.id) for i, row in candidates.iterrows()]

# Create function to plot a single stamp
def plot_stamp(stamp_uri, stamp_index, source_name, id):
    # Get the part of the stamp uri after the first / and up to 
    def interesting_part_of_stamp_uri(stamp_uri):
        return ".".join("".join(stamp_uri.split('/')[-1]).split(".")[:-1])
    
    # Get the name and path of where to save these pngs
    data_dir = "/mnt/cosmic-gpu-1/data0/nstiegle/candidate_plots/"
    filename = f"stamp_{interesting_part_of_stamp_uri(stamp_uri)}_{stamp_index}"
    path_to_save_to = data_dir + filename
    
    # Get the stamps
    stamps = [stamp for stamp in viewer.read_stamps(stamp_uri, find_recipe=True)]
    stamp = stamps[stamp_index]

    # Make the plots
    title =  f"file {interesting_part_of_stamp_uri(stamp_uri)} index {stamp_index}\n"
    title += f"Source: {source_name} ID: {id}\n"
    title += f"Freq: {stamp.stamp.signal.frequency} DR: {stamp.stamp.signal.driftRate} SNR: {stamp.stamp.signal.snr}"
    stamp.show_antennas(save_to=path_to_save_to + f"_antennas.png", title=title + ' antennas')
    try:
        stamp.show_best_beam(save_to=path_to_save_to + f"_coherent.png", title=title + ' coherent')
    except:
        stamp.show_beam(0, save_to=path_to_save_to + f"_coherent.png", title=title + ' coherent')
    stamp.show_classic_incoherent(save_to=path_to_save_to + f"_incoherent.png", title=title + ' incoherent')

# Setup for multiprocessing
p = multiprocessing.Pool() 

# Make all the stamps in parallel
results = p.starmap(plot_stamp, inputs)