import scipy.ndimage as ndi
import numpy as np
import pandas as pd
import scipy as sp

def medfilt3(block, size=3, chunks=None):
    import dask.array as da 
    if isinstance(size, int):
        overlap = size // 2
    elif isinstance(size, tuple):
        overlap = [i // 2 for i in tuple]
    if chunks is None:
        chunks = (64, 64, 64)
    block = da.from_array(block, chunks=chunks)
    output = block.map_overlap(lambda block: ndi.median_filter(block, size=size), \
                               depth=overlap)
    return output.compute()

def dog(data, sig1, sig2):
    # kernel size = 2r + 1 = 8 sigma + 1
    data_1 = data
    if sig1 is not None: 
        sig1 = np.array(sig1) if not isinstance(sig1, np.ndarray) else sig1
        if np.all(sig1 > 0):
            data_1 = ndi.gaussian_filter(data, sig1)

    data_2 = data
    if sig2 is not None: 
        sig2 = np.array(sig2) if not isinstance(sig2, np.ndarray) else sig2
        if np.all(sig2 > 0):
            data_2 = ndi.gaussian_filter(data, sig2)

    return (data_1 - data_2).astype(data.dtype)

def imgaussfilt2(data, wd_sz):
    # import cv2
    data_dim = data.ndim
    if data_dim == 2:
        data = data[None, :, :]
    for i in range(data.shape[0]):
        data[i, :, :] = cv2.GaussianBlur(data[i, :, :], wd_sz, 0)
    if data_dim == 2:
        data = np.squeeze(data)
    return data

def central_difference_1d(x): 
    x = np.asarray(x)
    assert x.ndims == 1, "x should have dimension 1"
    if x.size <= 1: 
        return np.full(x.size, np.nan)
    elif x.size == 2: 
        return np.full(x.size, x[1] - x[0])
    else: 
        return np.concatenate(([x[1] - x[0]], (x[2:] - x[:-2]) / 2, [x[-1] - x[-2]]))
    
def central_difference_2d(X, axis=0):
    if axis >= len(X.shape):
        raise ValueError("Axis out of bounds for array dimension")
    
    dX = np.full(X.shape, np.nan)
    if X.shape[axis] >= 2: 
        if axis == 0:
            dX[1:-1, :] = (X[2:, :] - X[:-2, :]) / 2
            dX[0, :] = (X[1, :] - X[0, :])
            dX[-1, :] = (X[-1, :] - X[-2, :])
        elif axis == 1:
            dX[:, 1:-1] = (X[:, 2:] - X[:, :-2]) / 2
            dX[:, 0] = (X[:, 1] - X[:, 0])
            dX[:, -1] = (X[:, -1] - X[:, -2])
    
    return dX

def moving_average_1d(x, window_size, min_periods=None):
    if min_periods is None:
        min_periods = window_size
    x = pd.Series(x)
    moving_avg = x.rolling(window=window_size, min_periods=min_periods,
                            center=True).mean().to_numpy()
    return moving_avg

def find_peaks_in_irregular_sampled_data(x, y, min_dist, fill_gap_Q=False): 
    """
    Find peaks in irregularly sampled data.

    Parameters
    ----------
    x : array-like
        The x-coordinates of the data.
    y : array-like
        The y-coordinates of the data.
    min_dist : float or array-like
        The minimum distance between peaks.
    fill_gap_Q : bool, default False
        Insert low-valued samples into large x-gaps before calling
        ``scipy.signal.find_peaks`` so isolated edge clusters can contribute
        local maxima without changing the default behavior.

    Returns
    -------
    p_x : array-like
        The x-coordinates of the peaks.
    p_val : array-like
        The y-coordinates of the peaks.
    p_idx : array-like
        The indices of the peaks in the original data.
    """
    assert len(x) == len(y), "x and y must have the same length"
    x = np.asarray(x)
    y = np.asarray(y)
    orig_idx = np.arange(len(x))
    min_dist = np.asarray(min_dist)
    assert np.all(min_dist > 0), "min_dist must be positive"
    if min_dist.size == 1:
            min_dist = np.full(len(x), min_dist)
    else:
        assert min_dist.size == len(x), "min_dist must be scalar or match x"

    if not np.all(x[1:] > x[:-1]):
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        y = y[sort_idx]
        orig_idx = orig_idx[sort_idx]
        min_dist = min_dist[sort_idx]

    peak_x = x
    peak_y = y
    peak_orig_idx = orig_idx
    peak_min_dist = min_dist
    if fill_gap_Q and len(x) >= 2:
        gap_dist = x[1:] - x[:-1]
        gap_th = np.maximum(min_dist[:-1], min_dist[1:])
        gap_idx = np.where(gap_dist > gap_th)[0]
        if gap_idx.size > 0:
            gap_x = (x[gap_idx] + x[gap_idx + 1]) / 2.0
            gap_y = np.full(gap_idx.size, np.nextafter(np.nanmin(np.asarray(y, dtype=float)), -np.inf))
            gap_orig_idx = np.full(gap_idx.size, -1, dtype=int)
            gap_min_dist = gap_th[gap_idx]
            peak_x = np.concatenate((x, gap_x))
            peak_y = np.concatenate((np.asarray(y, dtype=float), gap_y))
            peak_orig_idx = np.concatenate((orig_idx, gap_orig_idx))
            peak_min_dist = np.concatenate((min_dist, gap_min_dist))
            sort_idx = np.argsort(peak_x)
            peak_x = peak_x[sort_idx]
            peak_y = peak_y[sort_idx]
            peak_orig_idx = peak_orig_idx[sort_idx]
            peak_min_dist = peak_min_dist[sort_idx]

    peak_idx = sp.signal.find_peaks(peak_y)[0]
    valid_peak_Q = peak_orig_idx[peak_idx] >= 0
    peak_idx = peak_idx[valid_peak_Q]
    p_idx = peak_orig_idx[peak_idx]
    p_val = peak_y[peak_idx]
    p_x = peak_x[peak_idx]
    p_min_dist = peak_min_dist[peak_idx]
    n = peak_idx.size
    priority = np.argsort(p_val)[::-1]
    kept_Q = np.ones_like(p_val, dtype=bool)
    accepted_Q = np.zeros_like(p_val, dtype=bool)
    for i in priority: 
        if not kept_Q[i]: 
            continue
        xi = p_x[i]
        tmp_min_dist = p_min_dist[i]

        # A lower-priority peak should not evict a higher-priority peak that is
        # already kept, but its own exclusion radius still applies to acceptance.
        reject_Q = False
        j = i - 1
        while j >= 0 and (xi - p_x[j]) < tmp_min_dist:
            if accepted_Q[j]:
                reject_Q = True
                break
            j -= 1
        if not reject_Q:
            j = i + 1
            while j < n and (p_x[j] - xi) < tmp_min_dist:
                if accepted_Q[j]:
                    reject_Q = True
                    break
                j += 1
        if reject_Q:
            kept_Q[i] = False
            continue

        accepted_Q[i] = True
        # left
        j = i - 1
        while j >= 0 and (xi - p_x[j]) < tmp_min_dist: 
            if not accepted_Q[j]:
                kept_Q[j] = False
            j -= 1
        # right
        j = i + 1
        while j < n and (p_x[j] - xi) < tmp_min_dist: 
            if not accepted_Q[j]:
                kept_Q[j] = False
            j += 1 

    p_x = p_x[accepted_Q]
    p_val = p_val[accepted_Q]
    p_idx = p_idx[accepted_Q]
    return p_x, p_val, p_idx