# trim_dataset.py
# Noah Stiegler - 6/14/24
#
# Trims off the unnecessary columns of a dataset from COSMIC, leaving
# only the columns that seem useful for anomaly detection and the
# id so any hits can be cross referenced back to the main dataset.
# Also turns MJD dates into human readable dates
#

# Import useful files
import pandas as pd
from astropy.time import Time

# Read in file
data_folder = "/datax/scratch/nstieg/"
filename = data_folder + "25GHz_higher.pkl"
df = pd.read_pickle(filename)
print("File read in correctly")

# Convert to human readable dates
df["tstart_h"] = Time(df["tstart"], format="mjd").datetime
print("Converted to human readable time")

# Keep just these columns
cols_to_care_about = ["id",
    "signal_frequency",
    "signal_drift_rate",
	"signal_snr",
	"signal_beam",
	"signal_power",
	"signal_incoherent_power",
    "signal_num_timesteps",
    "tstart",
	"tstart_h",
	"ra_hours",
	"dec_degrees",
    "source_name"]
trimmed_df = df[cols_to_care_about]
print("Trimmed dataset")

# Split up all the incoherent data
incoherent = df["source_name"].isin(["Incoherent"]) # Find indices of df with incoherent data
print("Indexed incoherent")
incoherent_df = trimmed_df[incoherent][cols_to_care_about] # Make new df
print("Created incoherent df")
assert(len(incoherent_df["source_name"].unique()) == 1) # Check it only has incoherent data in it
assert(incoherent_df["source_name"].unique()[0] == "Incoherent")
incoherent_df.to_pickle(data_folder + "25GHz_higher_incoherent.pkl") # Save new df
print("Saved incoherent df")

# Same with phase_center data (coherent beam formed at center of incoherent beam
# when no targets are available)
phase_center = df["source_name"].isin(["PHASE_CENTER"])
print("Indexed phase center")
phase_center_df = trimmed_df[phase_center][cols_to_care_about]
print("Created phase center df")
assert(len(phase_center_df["source_name"].unique()) == 1)
assert(phase_center_df["source_name"].unique()[0] == "PHASE_CENTER")
phase_center_df.to_pickle(data_folder + "25GHz_higher_phase_center.pkl")
print("Saved phase center df")

# Same with the coherent beams aimed at real COSMIC targets
coherent = ~df["source_name"].isin(["Incoherent", "PHASE_CENTER"])
print("Indexed coherent")
coherent_df = trimmed_df[coherent][cols_to_care_about]
print("Created coherent df")
assert("Incoherent" not in coherent_df["source_name"].unique())
assert("PHASE_CENTER" not in coherent_df["source_name"].unique())
coherent_df.to_pickle(data_folder + "25GHz_higher_coherent.pkl")
print("Saved coherent df")

