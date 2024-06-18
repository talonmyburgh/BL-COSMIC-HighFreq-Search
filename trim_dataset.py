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

# Throw away columns I don't care about
cols_to_care_about = ["id", 
    "signal_frequency",
    "signal_drift_rate",
	"signal_snr",
	"signal_beam",
	"signal_power",
	"signal_incoherent_power",
	"tstart_h",
	"ra_hours",
	"dec_degrees"]
trimmed_df = df[cols_to_care_about]
print("Trimmed dataset")

# Divide into coherent vs. incoherent and save
incoherent = df["source_name"] == "incoherent"
print("Indexed incoherent")

incoherent_df = trimmed_df[incoherent][cols_to_care_about]
print("Created incoherent df")

incoherent_df.to_pickle(data_folder + "25GHz_higher_incoherent.pkl")
print("Saved incoherent df")

coherent_df = trimmed_df[~incoherent][cols_to_care_about]
print("Created coherent df")

coherent_df.to_pickle(data_folder + "25GHz_higher_coherent.pkl")
print("Saved coherent df")

