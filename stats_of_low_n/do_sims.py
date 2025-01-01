# stamp_viewer env
import numpy as np
import matplotlib.pyplot as plt
import setigen as stg
from astropy import units as u
import multiprocessing
from datetime import datetime


shared_data = None

def init_shared_data(data):
    global shared_data
    shared_data = data

def gen_obs_data(n_fine, n_ints, n_pols, n_complex):
    # generate fake data, one for each polarization / complex num, then square and sum
    mean = 0
    std_dev = 1
    n_nums_per_pixel = n_pols * n_complex
    data = np.random.normal(loc=mean, scale=std_dev, size=(n_ints, n_fine, n_nums_per_pixel))
    data = np.square(data)
    data =  np.sum(data, axis=2)
    return data

def max_snr_at_drift(data_shape, drift_rate):
    global shared_data
    
    # Convert shared data back to a 2D NumPy array
    data = np.frombuffer(shared_data.get_obj(), dtype=np.float64).reshape(data_shape)
    
    # See what strongest signal was - detectable?
    def give_snr_of_max(integrated):
        median = np.percentile(integrated, 50)
        top_5 = np.percentile(integrated, 95)
        bottom_5 = np.percentile(integrated, 5)
        center_data = integrated[(integrated < top_5) * (integrated > bottom_5)]
        std_dev = center_data.std()

        return (max(integrated) - median) / std_dev

    # Dedrift
    f_off = 1.9073486328125e-06 # fine channel spacing
    t_samp = 0.524288 # len of integration
    freq = 43255.90455616229 # freq of 1st channel
    frame = stg.Frame.from_data(f_off * u.MHz, t_samp * u.s, freq * u.MHz, True, data)
    dedrifted = stg.dedrift(frame, drift_rate)
    integrated = stg.integrate(dedrifted, mode='sum')

    max_snr = give_snr_of_max(integrated)
    return max_snr

def sim_one(n_fine, n_ints, n_pols, n_complex, drifts, snr_thresh=8):
        global shared_data
        data = gen_obs_data(n_fine, n_ints, n_pols, n_complex)
        shared_data = multiprocessing.Array('d', data.flatten()) # d for double, 1d only

        # Run sim with multiprocessing
        with multiprocessing.Pool(initializer=init_shared_data, initargs=(shared_data,)) as p:
            args = [(data.shape, drift_rate) for drift_rate in drifts]
            results = np.array(p.starmap(max_snr_at_drift, args))

        detections = results > snr_thresh
        return detections.sum()

# Do at different SNRs
snrs = np.arange(5, 15, 0.5)
results = []
for snr in snrs:
    with open("log.txt", "a") as f:
        f.write(f"on {snr} at {datetime.now()}\n")
    result = []
    # repeat trial 5x
    for i in range(5):
        n_fine = 2**19 # fine channels
        n_ints = 16 # integrations
        n_pols = 2 # polarizations
        n_complex = 2 # parts of complex result
        drifts = np.arange(-50, 50, .25)
        count = sim_one(n_fine, n_ints, n_pols, n_complex, drifts, snr_thresh=snr)
        result.append(count)
    results.append(result)

to_save = np.array(results)
np.save("results.npy", to_save)
np.save("snrs.npy", snrs)
