import os

script_path = os.path.abspath(__file__)

# Get the directory containing the script
script_dir = os.path.dirname(script_path)
log_filepath = script_dir + "/run_filter_2_coherent_log.txt"
print(log_filepath)