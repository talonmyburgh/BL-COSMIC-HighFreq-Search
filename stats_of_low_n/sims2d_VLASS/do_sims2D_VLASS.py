# stamp_viewer env
import numpy as np
import matplotlib.pyplot as plt
import setigen as stg
from astropy import units as u
import multiprocessing
from datetime import datetime


shared_data = None

def init_shared_data(data):
    """
    init_shared_data
    Something to initialize the shared data resource as a global variable. Not
    sure how/why it works, but StackOverflow recommended it and it works 

    data: The (n_ints, n_fine) numpy array of data

    Returns: Nothing

    AFFECTS GLOBAL VARIABLE shared_data
    """
    global shared_data
    shared_data = data

def gen_obs_data(n_fine, n_ints, n_pols, n_complex, PFB_filepath=""):
    """
    gen_obs_data
    Generate one observational coarse channel

    n_fine: Number of fine channels per coarse channel - usually a power of 2
    n_ints: Number of integrations performed, amount of data per frequency bin
    n_pols: Number of polarizations observed - usually 2
    n_complex: Number of complex components to an observation - usually 2 for IQ
    PFB_filepath: Filepath to the PFB response for the observation, or "" if no
                  PFB response should be added

    Returns a numpy array which is (n_ints, n_fine), where each element represents
    the data observed at that frequency and integration, a Chi-Square(4)
    distribution (sum of squares of 4 normal samples)
    """
    # generate fake data, one for each polarization / complex num, then square and sum
    mean = 0
    std_dev = 1
    n_nums_per_pixel = n_pols * n_complex
    data = np.random.normal(loc=mean, scale=std_dev, size=(n_ints, n_fine, n_nums_per_pixel))
    data = np.square(data)
    data =  np.sum(data, axis=2)

    # Add PFB correction
    if PFB_filepath != "": # '/home/nstiegle/raw_data_work/COSMIC_response.f32' for high freq data
        # Read in PFB shape data
        with open(PFB_filepath, 'rb') as f:
            # Use numpy's fromfile method to read in the 32-bit floats
            pfb = np.fromfile(f, dtype=np.float32)

        # Check shapes and apply PFB correction to data
        assert(len(pfb) == data.shape[1])
        before_shape = data.shape
        data *= pfb
        assert(data.shape == before_shape)

    return data

def above_snr_at_drift(data_shape, drift_rate, snr_thresh, f_off, t_samp, freq):
    """
    above_snr_at_drift
    Figure out how many drift lines yield an integrated intensity above the
    given SNR threshold at the given drift rate

    data_shape: Shape of the shared_data object stored in global memory, usually
                (n_ints, n_fine)
    drift_rate: Drift rate to dedrift and integrate at, in Hz/s
    snr_thresh: Threshold in SNR to define a detection
    f_off:      Frequency offset/spacing between fine channels in MHz
    t_samp:     Length in seconds of each integration
    freq:       Frequency of the first fine channel in MHz

    The integer number of drift lines with a positive detection
    """
    global shared_data
    
    # Convert shared data back to a 2D NumPy array
    data = np.frombuffer(shared_data.get_obj(), dtype=np.float64).reshape(data_shape)
    
    # Get snr of all signals
    def give_snr(integrated):
        median = np.percentile(integrated, 50)
        top_5 = np.percentile(integrated, 95)
        bottom_5 = np.percentile(integrated, 5)
        center_data = integrated[(integrated < top_5) * (integrated > bottom_5)]
        std_dev = center_data.std()

        return (integrated - median) / std_dev

    # Dedrift
    frame = stg.Frame.from_data(f_off * u.MHz, t_samp * u.s, freq * u.MHz, True, data)
    dedrifted = stg.dedrift(frame, drift_rate)
    integrated = stg.integrate(dedrifted, mode='sum')

    # See how many signals above threshold
    snrs = give_snr(integrated)
    return sum(snrs > snr_thresh)


def sim_one(n_fine, n_ints, n_pols, n_complex, drifts, snr_thresh=8, PFB_filepath="",
            f_off=1.9073486328125e-06, t_samp=0.524288, freq=43255.90455616229):
    """
    sim_one
    Simulate a search on one observational coarse channel at a range of drift rates

    n_fine: Number of fine channels per coarse channel - usually a power of 2
    n_ints: Number of integrations performed, amount of data per frequency bin
    n_pols: Number of polarizations observed - usually 2
    n_complex: Number of complex components to an observation - usually 2 for IQ
    drifts: Range of drift rates simulated - usually -50 to 50 in increments of 0.25 Hz/s
    snr_thresh: Signal to Noise Ratio threshold for a detection
    PFB_filepath: Filepath to the PFB response for the observation, or "" if no
                  PFB response should be added
    f_off:      Frequency offset/spacing between fine channels in MHz (1.9073486328125e-06 at high freq)
    t_samp:     Length in seconds of each integration (0.524288 at high freq)
    freq:       Frequency of the first fine channel in MHz (43255.90455616229 at high freq)

    Returns:
    The number of total drift lines, over all drift rates, which have a positive
    detection, an integer
    """

    global shared_data
    data = gen_obs_data(n_fine, n_ints, n_pols, n_complex, PFB_filepath=PFB_filepath)
    shared_data = multiprocessing.Array('d', data.flatten()) # d for double, 1d only

    # Run sim with multiprocessing
    with multiprocessing.Pool(initializer=init_shared_data, initargs=(shared_data,)) as p:
        args = [(data.shape, drift_rate, snr_thresh, f_off, t_samp, freq) for drift_rate in drifts]
        results = np.array(p.starmap(above_snr_at_drift, args))

    return results.sum()



################################################################################
# Run Simulation
################################################################################
if __name__ == "__main__":
    # Do at different SNRs
    snrs = np.arange(5, 15.5, 0.5)
    n_ints_to_simulate = [64]
    total_results = []
    for n_ints in n_ints_to_simulate:
        result_at_n_ints = [] 
        for snr in snrs:
            with open("log2d.txt", "a") as f:
                f.write(f"on snr: {snr} n_ints: {n_ints} at {datetime.now()}\n")
            individual_result = []
            # repeat trial 5x
            for i in range(5):
                n_fine = 2**17 # fine channels - 2^17 for VLASS, 2^19 for high freq
                n_pols = 2 # polarizations
                n_complex = 2 # parts of complex result
                drifts = np.arange(-50, 50.25, .25)
                # Channel properties from https://ui.adsabs.harvard.edu/abs/2025AJ....169..122T/abstract
                """
                Coherently and incoherently
                beamformed data are searched using these parameters, with a
                frequency resolution of 7.63 Hz channel–1 for the total 8 s
                recording segment

                The data recorded during an OTF scan are partitioned into
                files spanning 8.338608 s each. 
                
                Before the actual beamforming computation, each coarse channel is upchannelized to the requested
                resolution without reducing the number of fine spectra below
                50 (typically 64 time steps for VLASS observations with an
                FFT size of 131,072), 

                The VLA was in
                B-configuration and recording data at 2–4 GHz in the 8 bit
                recording mode, and 24 of the 27 antennas were online and
                recording for the duration of the observation. E
                """
                count = sim_one(n_fine, n_ints, n_pols, n_complex, drifts, snr_thresh=snr, 
                                PFB_filepath='',
                                f_off=7.63e-06, t_samp=8.338608 / 64, freq=3000)
                individual_result.append(count)
            result_at_n_ints.append(individual_result)
        total_results.append(result_at_n_ints)

    to_save = np.array(total_results)
    np.save("results2d.npy", to_save)
    np.save("snrs2d.npy", snrs)
    np.save("n_ints2d.npy", n_ints_to_simulate)