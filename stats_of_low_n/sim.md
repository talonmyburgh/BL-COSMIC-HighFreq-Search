```python
# stamp_viewer env
import numpy as np
import matplotlib.pyplot as plt
import setigen as stg
from astropy import units as u
import multiprocessing
```


```python
# Generate one coarse channel of fake observational data - just Gaussian noise
def gen_obs_data(n_fine, n_ints, n_pols, n_complex):
    # generate fake data, one for each polarization / complex num, then square and sum
    mean = 0
    std_dev = 1
    n_nums_per_pixel = n_pols * n_complex
    data = np.random.normal(loc=mean, scale=std_dev, size=(n_ints, n_fine, n_nums_per_pixel))
    data = np.square(data)
    data =  np.sum(data, axis=2)
    return data

# Determine number of detectable signals in data at this drift rate
def above_snr_at_drift(data, drift_rate, snr_thresh):
    
    # Get snr of all signals
    def give_snr(integrated):
        median = np.percentile(integrated, 50)
        top_5 = np.percentile(integrated, 95)
        bottom_5 = np.percentile(integrated, 5)
        center_data = integrated[(integrated < top_5) * (integrated > bottom_5)]
        std_dev = center_data.std()

        return (max(integrated) - median) / std_dev

    # Dedrift and integrate a given dr
    f_off = 1.9073486328125e-06 # fine channel spacing
    t_samp = 0.524288 # len of integration
    freq = 43255.90455616229 # freq of 1st channel
    frame = stg.Frame.from_data(f_off * u.MHz, t_samp * u.s, freq * u.MHz, True, data)
    dedrifted = stg.dedrift(frame, drift_rate)
    integrated = stg.integrate(dedrifted, mode='sum')

    # See how many signals above threshold
    snrs = give_snr(integrated)
    return snrs > snr_thresh

# Figure out how many signals would be found in data at a given snr
def sim_one(n_fine, n_ints, n_pols, n_complex, drifts, snr_thresh=8):
        data = gen_obs_data(n_fine, n_ints, n_pols, n_complex)
        # Run sim with multiprocessing
        with multiprocessing.Pool() as p:
            args = [(data, drift_rate, snr_thresh) for drift_rate in drifts]
            results = np.array(p.starmap(above_snr_at_drift, args))

        return results.sum()
```


```python
# Do a simulation run
n_fine = 2**19 # fine channels
n_ints = 16 # integrations
n_pols = 2 # polarizations
n_complex = 2 # parts of complex result
drifts = np.arange(-50, 50, .25)
count = sim_one(n_fine, n_ints, n_pols, n_complex, drifts, snr_thresh=8)
print(count)
# num_seached_signals = # tbd
# print(f"{count} / {num_seached_signals} = {round(count / (num_seached_signals) * 100, 3)}%")
# print(f"{count} / {len(drifts)} = {round(count / len(drifts) * 100, 3)}%") # <-- used to see how many drift rates had 1+ signals > thresh
```
