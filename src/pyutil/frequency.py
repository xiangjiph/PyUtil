import numpy as np
import scipy as sp
from matplotlib import pyplot as plt


def spectral_power_lomb_scargle(t_s, signal, vis_Q=False):
    result = {}
    result['num_sample'] = t_s.size
    result['total_length'] = t_s[-1] - t_s[0]
    assert t_s.size == signal.size, 'Input t and f dimension mismatch'
    is_finite_Q = np.isfinite(signal)
    t_s = t_s[is_finite_Q]
    signal = signal[is_finite_Q]
    if not np.all((t_s[1:] - t_s[:-1]) > 0): 
        s_idx = np.argsort(t_s)
        t_s = t_s[s_idx]
        signal = signal[s_idx]
    result['min_sample_interval'] = np.min(t_s[1:] - t_s[:-1])
    result['Nyquest_f'] = 1 / result['min_sample_interval'] / 2
    result['f'] = np.linspace(1/result['total_length'], result['Nyquest_f'], result['num_sample'])
    result['pwr'] = sp.signal.lombscargle(t_s, signal, result['f'] * 2 * np.pi, normalize=True, precenter=True)

    if vis_Q: 
        f = plt.figure(figsize=(8, 6))
        a = f.add_subplot()
        a.plot(result['f'], result['pwr'])
        # a.plot(f_cpt, pg_pwr)
        a.set_xscale('log')
        # a.set_yscale('log')
        a.grid()
        a.set_xlabel(f"f (Hz)")
        a.set_ylabel(f"Power")
        # a.legend()

    return result



    

    
    
    
