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


def gaussian_distributions_resolvable(spacing, std_1, std_2):
    """Test whether an equal-weight sum of two Gaussian PDFs has two peaks.

    Parameters
    ----------
    spacing : float or array-like
        Absolute spacing between the two Gaussian centers.
    std_1, std_2 : float or array-like
        Positive standard deviations. Inputs are broadcast together.

    Returns
    -------
    bool or ndarray
        True where the sum has two distinct local maxima. The critical spacing
        itself is not considered resolvable because it produces a stationary
        inflection point rather than two peaks.
    """
    spacing, std_1, std_2 = np.broadcast_arrays(
        np.asarray(spacing, dtype=float), np.asarray(std_1, dtype=float),
        np.asarray(std_2, dtype=float))
    if np.any(~np.isfinite(spacing)) or np.any(spacing < 0):
        raise ValueError("spacing must be finite and nonnegative")
    if np.any(~np.isfinite(std_1)) or np.any(~np.isfinite(std_2)) \
            or np.any(std_1 <= 0) or np.any(std_2 <= 0):
        raise ValueError("standard deviations must be finite and positive")
    equal_std_Q = std_1 == std_2
    if np.all(equal_std_Q):
        resolvable_Q = spacing > 2 * std_1
        return resolvable_Q.item() if resolvable_Q.ndim == 0 else resolvable_Q

    std_hi = np.maximum(std_1, std_2)
    std_lo = np.minimum(std_1, std_2)
    log_r = np.log(std_lo) - np.log(std_hi)

    # At the critical spacing, f = (phi_1 + phi_2) / 2 has f' = f'' = 0
    # at one point between its centers. Set r = std_lo / std_hi and let
    # q be the ratio of that point's standardized distances from the centers.
    # Eliminating the two PDF values gives the one-dimensional equation
    # C sinh(log(q)) - log(q) - 2 log(r) = 0, C = (r + q)/(1 + rq).
    # Its relevant root has log(q) <= 0, so vectorized bisection is stable and
    # avoids sampling the Gaussian mixture separately for every comparison.
    log_q_lo = 2 * log_r - 50.0
    log_q_hi = np.zeros_like(log_r)
    for _ in range(48):
        log_q = (log_q_lo + log_q_hi) / 2.0
        log_c = np.logaddexp(log_r, log_q) - np.logaddexp(0.0, log_r + log_q)
        log_abs_sinh = np.log(-np.expm1(2 * log_q)) - log_q - np.log(2.0)
        with np.errstate(over="ignore"):
            transition_value = -np.exp(log_c + log_abs_sinh) - log_q - 2 * log_r
        lower_half_Q = transition_value < 0
        log_q_lo = np.where(lower_half_Q, log_q, log_q_lo)
        log_q_hi = np.where(lower_half_Q, log_q_hi, log_q)

    log_q = (log_q_lo + log_q_hi) / 2.0
    # The second-derivative equation also gives
    # d_critical/std_hi = (q + r)^(3/2) / sqrt(q * (1 + rq)).
    log_critical_spacing = (
        np.log(std_hi) + 1.5 * np.logaddexp(log_q, log_r)
        - 0.5 * (log_q + np.logaddexp(0.0, log_q + log_r)))
    with np.errstate(divide="ignore"):
        resolvable_Q = np.log(spacing) > log_critical_spacing
    resolvable_Q = np.where(equal_std_Q, spacing > 2 * std_1, resolvable_Q)
    return resolvable_Q.item() if resolvable_Q.ndim == 0 else resolvable_Q


def find_resolvable_peaks_in_irregular_sampled_data(x, y, std, fill_gap_Q=False):
    """Find locally maximal, Gaussian-resolvable peaks in irregular data.

    Parameters
    ----------
    x, y : array-like
        One-dimensional sample coordinates and values.
    std : float or array-like
        Positive Gaussian standard deviation for every sample, or one scalar
        standard deviation shared by all samples.
    fill_gap_Q : bool, default False
        Insert low-valued samples between adjacent resolvable Gaussians before
        local-peak detection, allowing separated edge clusters to contribute
        peaks.

    Returns
    -------
    p_x, p_val, p_idx : ndarray
        Peak coordinates, values, and indices into the original inputs. Peaks
        are returned in increasing x order. Candidate values determine greedy
        priority; resolution itself is a pairwise, equal-weight criterion.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError("x and y must be one-dimensional arrays of equal length")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ValueError("x and y must contain only finite values")

    std = np.asarray(std, dtype=float)
    if std.ndim == 0 or std.size == 1:
        std = np.full(x.size, std.item(), dtype=float)
    elif std.ndim != 1 or std.size != x.size:
        raise ValueError("std must be scalar or a one-dimensional array matching x")
    if np.any(~np.isfinite(std)) or np.any(std <= 0):
        raise ValueError("std must contain only finite, positive values")

    orig_idx = np.arange(x.size)
    if not np.all(x[1:] > x[:-1]):
        sort_idx = np.argsort(x, kind="stable")
        x = x[sort_idx]
        y = y[sort_idx]
        std = std[sort_idx]
        orig_idx = orig_idx[sort_idx]
    if np.any(x[1:] <= x[:-1]):
        raise ValueError("x coordinates must be unique")

    peak_x = x
    peak_y = y
    peak_std = std
    peak_orig_idx = orig_idx
    if fill_gap_Q and x.size >= 2:
        gap_Q = gaussian_distributions_resolvable(np.diff(x), std[:-1], std[1:])
        gap_idx = np.flatnonzero(gap_Q)
        if gap_idx.size > 0:
            gap_x = x[gap_idx] + (x[gap_idx + 1] - x[gap_idx]) / 2.0
            gap_y = np.full(gap_idx.size, np.nextafter(np.min(y), -np.inf))
            gap_std = np.maximum(std[gap_idx], std[gap_idx + 1])
            gap_orig_idx = np.full(gap_idx.size, -1, dtype=int)
            peak_x = np.concatenate((x, gap_x))
            peak_y = np.concatenate((y, gap_y))
            peak_std = np.concatenate((std, gap_std))
            peak_orig_idx = np.concatenate((orig_idx, gap_orig_idx))
            sort_idx = np.argsort(peak_x, kind="stable")
            peak_x = peak_x[sort_idx]
            peak_y = peak_y[sort_idx]
            peak_std = peak_std[sort_idx]
            peak_orig_idx = peak_orig_idx[sort_idx]

    peak_idx = sp.signal.find_peaks(peak_y)[0]
    peak_idx = peak_idx[peak_orig_idx[peak_idx] >= 0]
    p_x = peak_x[peak_idx]
    p_val = peak_y[peak_idx]
    p_std = peak_std[peak_idx]
    p_idx = peak_orig_idx[peak_idx]

    # Greedily accept the highest-valued candidate and suppress every remaining
    # candidate whose equal-weight Gaussian pair cannot form two distinct modes.
    priority = np.lexsort((p_x, -p_val))
    pending_Q = np.ones(p_val.size, dtype=bool)
    accepted_Q = np.zeros(p_val.size, dtype=bool)
    for i in priority:
        if not pending_Q[i]:
            continue
        pending_Q[i] = False
        accepted_Q[i] = True
        neighbor_idx = np.flatnonzero(pending_Q)
        if neighbor_idx.size == 0:
            continue
        resolvable_Q = gaussian_distributions_resolvable(
            np.abs(p_x[neighbor_idx] - p_x[i]), p_std[neighbor_idx], p_std[i])
        pending_Q[neighbor_idx[~resolvable_Q]] = False

    return p_x[accepted_Q], p_val[accepted_Q], p_idx[accepted_Q]
