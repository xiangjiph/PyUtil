from collections import defaultdict

import numpy as np
import scipy as sp
import pandas as pd
import scipy.interpolate as spi
import sklearn.mixture as sklm
import matplotlib.pyplot as plt

def compute_basic_statistics(data, bins=None, opt_stat=['pdf', 'percentile'], \
                             reject_outlier_ipr=None):
    # bins:
    #   If scalar, should be an integer that specifies the number of bins used for the histogram calculation.
    #   If a list or numpy array, the PDF, CDF calculation will be based on the bin edge. Data outside the edge will be ignored.
    # Takes 5.7 seconds for data.shape = (150, 1280, 1024)
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    try: 
        data = data[np.isfinite(data)] # data might contain nan
    except: 
        print(data)
        raise
    data = data.flatten()
    if reject_outlier_ipr is not None and data.size > 0: 
        outlier_th = compute_percentile_outlier_threshold(data, ipr=reject_outlier_ipr)
        in_lier_Q = (data >= outlier_th[0]) & (data <= outlier_th[1])
        data = data[in_lier_Q]

    result = {'num_data': data.size, 'mean': np.nan, 'std': np.nan, 'cv': np.nan, 'max': np.nan, 'min': np.nan, 'median': np.nan, 
        'num_bins': np.array([]), 'hist_count': np.array([]), 'bin_val': np.array([]), 
        'bin_edge': np.array([]), 'bin_width': np.array([]), 'probability': np.array([]), 'pdf': np.array([]),
        'cdf': np.array([]),'prctile_th': np.array([]), 'prctile_th': np.array([]), 
        'eff_ptrl_std': np.nan, 'eff_ptrl_cv': np.nan}
    if bins is None:
        result['num_bins'] = 10
    elif np.isscalar(bins):
        result['num_bins'] = bins
    else:
        # input is bin edge
        if not isinstance(bins, np.ndarray):
            bins = np.array(bins)
        
    if data.size > 0: 
        result['mean'] = np.mean(data)
        mean_x2 = np.mean(data ** 2)
        result['std'] = np.sqrt(mean_x2 - result['mean'] ** 2)
        result['cv'] = result['std'] / np.abs(result['mean']) if result['mean'] != 0 else np.nan
        if 'pdf' in opt_stat: 
            if isinstance(bins, np.ndarray):
                result['num_bins'] = bins.size - 1 # bin edge
                result['hist_count'], result['bin_edge'] = np.histogram(data, bins=bins)
            else:
                result['hist_count'], result['bin_edge'] = np.histogram(
                    data, bins=result['num_bins'])

            result['bin_val'] = (result['bin_edge'][:-1] + result['bin_edge'][1:]) / 2
            result['bin_width'] = result['bin_edge'][1:] - result['bin_edge'][:-1]
            result['probability'] = result['hist_count'] / np.sum(result['hist_count'])
            result['pdf'] = result['probability'] / result['bin_width']
            result['cdf'] = np.cumsum(result['probability'])

        if 'percentile' in opt_stat: 
            result['prctile_th'] = np.array(
                [0, 0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9, 100])
            result['prctile_val'] = np.percentile(data, q=result['prctile_th'])
            result['median'] = result['prctile_val'][6]
            result['max'] = result['prctile_val'][-1]
            result['min'] = result['prctile_val'][0]
            result['eff_ptrl_std'] = (result['prctile_val'][7] - result['prctile_val'][5]) / 1.349
            result['eff_ptrl_cv'] = np.abs(result['eff_ptrl_std'] / result['median'])
        
    return result

def compute_stat_by_idx_bin(data_1d, bin_idx:list, stat_list=['mean', 'std'], 
                                    feature_name=None):
    if isinstance(data_1d, pd.Series):
        data_1d = data_1d.values
    else: 
        data_1d = np.asarray(data_1d)
    # assert isinstance(data_1d, np.ndarray) and data_1d.ndim == 1

    num_bins = len(bin_idx)
    feature_stat_list = [f"{feature_name}_{stat_name}" if feature_name is not None else f"{stat_name}"\
                            for stat_name in stat_list]
    bin_stat = {}
    for s_k in feature_stat_list: 
        bin_stat[s_k] = np.full(num_bins, np.nan, dtype=np.float32)

    for i, tmp_idx in enumerate(bin_idx):                
        tmp_gd = data_1d[tmp_idx]
        if tmp_gd.size: 
            for s_k in feature_stat_list: 
                if s_k.endswith('mean'): 
                    tmp_stat = np.mean(tmp_gd)
                elif s_k.endswith('std'): 
                    tmp_stat = np.std(tmp_gd)
                else: 
                    print(f"Unknown statistical value {s_k}. Skip")
                    continue
                bin_stat[s_k][i] = tmp_stat
    
    return bin_stat

def compute_percentile_outlier_threshold(x, ipr = 1.5):
    # ipr = 1.5 -> 2 sigma
    #       2.2 -> 3 sigma
    p25, p75 = np.nanpercentile(x, [25, 75])
    width = ipr * (p75 - p25)
    return [p25 - width, p75 + width]

def is_outerlier_by_percentile(x, ipr = 1.5):
    th = compute_percentile_outlier_threshold(x, ipr=ipr)
    return (x < th[0]) | (x > th[1])

def is_inlier_by_percentile(x, ipr = 1.5):
    th = compute_percentile_outlier_threshold(x, ipr=ipr)
    return (x >= th[0]) & (x <= th[1])

def remove_outlier_by_percentile(x, ipr = 1.5):
    return x[is_inlier_by_percentile(x, ipr=ipr)]

def fill_internal_nan(data, method='linear'):
    mask = np.isnan(data)
    if np.any(mask):
        itp_hdl = spi.interp1d(np.flatnonzero(~mask), data[~mask], kind=method)
        data[mask] = itp_hdl(np.flatnonzero(mask))
    return data

def fill_edge_nan_1d(data, method='nearest'):
    if method == 'nearest':
        ind = np.flatnonzero(~np.isnan(data))
        data[:ind[0]] = data[ind[0]]
        data[ind[-1]:] = data[ind[-1]]
    return data 

def estimate_background_by_fraction(data, bg_frac=1, sample_step=None, bg_mask=None):
    if bg_mask is not None: 
        data = data[bg_mask]
    data = data.flatten()
    if sample_step is not None: 
        data = data[::sample_step]
    data = np.sort(data)
    if bg_frac < 1:
        data = data[:int(bg_frac * data.size)]
    result = {}
    result['mean'] = np.mean(data)
    result['std'] = np.std(data)
    result['p25'] = data[int(0.25 * data.size)]
    result['p50'] = data[int(0.50 * data.size)]
    result['p75'] = data[int(0.75 * data.size)]
    return result

def estimate_percentile_from_hist_count(count, ptrl, bin_val=None):

    count = np.asarray(count)
    ptrl = np.asarray(ptrl)
    if bin_val is None: 
        bin_val = np.arange(count.size) + 0.5
    cdf = np.cumsum(count) / np.sum(count)
    cdf_itp = spi.interp1d(cdf, bin_val, kind='linear', fill_value='extrapolate')
    return cdf_itp(ptrl / 100)
    
def analyze_hist_count_peaks_with_gaussian_mixture(bin_val, bin_count, num_comp_list, vis_Q=False):
    data = np.repeat(bin_val, bin_count.astype(np.int64))[:, None]
    result = analyze_1d_data_with_gaussian_mixture(data, num_comp_list=num_comp_list, vis_Q=vis_Q)
    return result

def analyze_1d_data_with_gaussian_mixture(data, num_comp_list, vis_Q=False):
    data = np.asarray(data).flatten()[:, None]
    num_sample = data.size
    if num_sample < 2:
        result = {'num_sample': num_sample, 'num_component': 0, 'weight': [], 
                  'mean': [], 'variance': [], 'std': [], 'cv': [], 'peak': [], 'peak_idx': [], 
                'peak_mean': [], 'peak_std': [], 'bic': [], 'num_comp_list': num_comp_list}
        return result

    lowest_bic = np.inf
    best_gm = None
    bics = []
    for n_comp in num_comp_list:
        if data.size >= n_comp:
            
            gm = sklm.GaussianMixture(n_components=n_comp, covariance_type='full').fit(data)

            bic = gm.bic(data)
            bics.append(bic)
            if bic < lowest_bic:
                best_gm = gm
                lowest_bic = bic
    
    means = best_gm.means_.flatten()
    covariances = best_gm.covariances_.flatten()  # For 1D, covariances are just variances
    stds = np.sqrt(covariances)
    weights = best_gm.weights_
    with np.errstate(divide='ignore'):
        cvs = stds / means
        peak_prob = weights / (np.sqrt(2 * np.pi * covariances))

    result = {'num_sample': num_sample, 'num_component': best_gm.n_components, 'weight': weights,
               'mean': means, 'variance': covariances, 'std': stds, 'cv': cvs, 'peak': peak_prob, 
              'bic': bics, 'num_comp_list': num_comp_list}
    result['peak_idx'] = np.argmax(result['peak']) 
    result['peak_mean'] = result['mean'][result['peak_idx']]
    result['peak_std'] = result['std'][result['peak_idx']]
    

    if vis_Q:
        print(f"Optimal number of Gaussians: {best_gm.n_components}")
        print("Component means:", means)
        print("Component stds:", stds)
        print("Component weights:", best_gm.weights_)
        print("Component peaks:", peak_prob)

        fig = plt.figure(figsize=(8, 4))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.hist(data, bins=50, alpha=0.5, density=True, label='Data')

        x_values = np.linspace(np.min(data), np.max(data), 500)
        pdf = np.zeros_like(x_values)
        for peak, mean, var in zip(peak_prob, means, covariances):
            component_pdf = peak * np.exp(-(x_values-mean)**2/(2*var))
            pdf += component_pdf
            ax1.plot(x_values, component_pdf, '--', label='Component')

        ax1.plot(x_values, pdf, 'k-', lw=2, label='GMM fit')

        ax1.set_xlabel('X')
        ax1.set_ylabel('Density')
        ax1.set_title('GMM Fit and Components')
        ax1.legend()

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(num_comp_list, bics, marker='o')
        ax2.set_xlabel('Number of components')
        ax2.set_ylabel('BIC Score')
        ax2.set_title('BIC vs. Number of Components')
        ax2.grid(True)

    return result

def relative_difference(x, y):
    x = np.asarray(x).astype(np.float32)
    y = np.asarray(y).astype(np.float32)
    if np.all(x >= 0) and np.all(y >= 0):
        return np.abs(x - y) / (x + y)
    else: 
        raise NotImplementedError
    
def moving_outlier_detection_by_z_score_1d(trace, wd_sz):
    trace = np.asarray(trace)
    pad_width = int(wd_sz) // 2
    trace = np.pad(trace, pad_width=pad_width, mode='reflect')
    ker = np.ones(wd_sz) / wd_sz
    mov_m = sp.signal.convolve(trace, ker, mode='valid')
    mov_m2 = sp.signal.convolve(trace ** 2, ker, mode='valid')
    std = np.sqrt(mov_m2 - mov_m ** 2) # prevent overflow? 
    z_score = mov_m / std
    return z_score

# def max_bidir_rel_diff(t_trace):
#     t_trace = np.asarray(t_trace).astype(np.float32)
#     t_diff = np.diff(t_trace)
#     forward_dn = np.concatenate(([0], np.abs(t_diff / t_trace[:-1]))) 
#     backward_dn = np.concatenate((np.abs(t_diff / t_trace[1:]), [0]))
#     return np.maximum(forward_dn, backward_dn)


def rolling_outlier_detection(data, window_size, quantile_range=(0.25, 0.75), k=3, min_lower=None, max_high=None):
    """
    Detect outliers in a time series using a rolling window approach with quantile-based thresholds.

    Parameters
    ----------
    data : array-like
        1D time series data.
    window_size : int
        Size of the rolling window.
    quantile_range : tuple of float
        The lower and upper quantiles to compute (e.g., (0.25, 0.75) for Q1 and Q3).
    k : float
        Multiplier for the IQR to define the outlier range.
    
    Returns
    -------
    outliers : pd.Series (dtype=bool)
        Boolean Series indicating True for outliers and False for inliers.
    """
    series = pd.Series(data)
    
    # Compute rolling quantiles
    q_low = series.rolling(window=window_size, center=True).quantile(quantile_range[0])
    q_high = series.rolling(window=window_size, center=True).quantile(quantile_range[1])
    
    # Compute rolling IQR and thresholds
    iqr = q_high - q_low
    lower_bound = q_low - k * iqr
    upper_bound = q_high + k * iqr
    if min_lower is not None: 
        lower_bound = np.maximum(min_lower, lower_bound)
    if max_high is not None:
        upper_bound = np.minimum(max_high, upper_bound)
    # Identify outliers
    outliers = (series < lower_bound) | (series > upper_bound)
    
    return outliers.values

def compute_rolling_inliner_range(data, wd_sz, k=3):

    data = pd.Series(data)
    data = pd.concat((data[:wd_sz-1][::-1], data, data[-(wd_sz-1):][::-1]))
    q_low = data.rolling(window=wd_sz, center=True).quantile(0.25)[wd_sz-1 : -wd_sz + 1]
    q_high = data.rolling(window=wd_sz, center=True).quantile(0.75)[wd_sz-1 : -wd_sz + 1]
    iqr = q_high - q_low
    upper_bound = q_high + k * iqr
    lower_bound = q_low - k * iqr

    return lower_bound.values, upper_bound.values

def analyze_matrix_diagonals(mat):
    """
    
    """
    assert mat.shape[0] == mat.shape[1], 'mat is not a square matrix'
    l = mat.shape[0]
    diag_offset = np.arange(-l + 1, l)
    stat = defaultdict(lambda : np.full(diag_offset.shape, np.nan))
    stat['offset'] = diag_offset
    for i, j in enumerate(diag_offset): 
        tmp_terms = np.diagonal(mat, offset=j)
        stat['num'][i] = tmp_terms.size
        stat['sum'][i] = np.sum(tmp_terms)
        stat['std'][i] = np.std(tmp_terms)
    
    stat['mean'] = stat['sum'] / stat['num']

    return stat

def analyze_twp_traces_corr(x, y):
    valid_Q = np.logical_and(np.isfinite(x), np.isfinite(y))
    x = x[valid_Q]
    y = y[valid_Q]
    x_avg = np.mean(x)
    y_avg = np.mean(y)
    x -= x_avg
    y -= y_avg
    xy = np.sum(x * y)
    result = {}
    result['corr'], result['p'] = sp.stats.pearsonr(x, y)
    result['k_y2x'] = xy / np.sum(x ** 2) # y = k_x * x
    result['k_x2y'] = xy / np.sum(y ** 2) # x = k_y * y
    result['k_y2x_n'] = result['k_y2x'] * x_avg / y_avg 
    result['k_x2y_n'] = result['k_x2y'] * y_avg / x_avg
    return result

def analyze_traces_corr_mat(X, skip_vec=None): 
    """
        X: (num_traces, T)
    
    """
    num_traces, num_T = X.shape
    if skip_vec is None: 
        skip_vec = np.ones((num_traces, ), bool)

    tree_corr_mat = np.full((num_traces, num_traces), np.nan)
    tree_corr_p_mat = np.full((num_traces, num_traces), 1.0)
    tree_slope_mat = np.full((num_traces, num_traces), np.nan)
    tree_slope_n_mat = np.full((num_traces, num_traces), np.nan)
    for i in range(num_traces):
        tmp_trace_1 = X[i]
        if skip_vec[i]:
            continue
        for j in range(i+1, num_traces): 
            tmp_trace_2 = X[j]
            if skip_vec[j]: 
                continue
            tmp_valid_Q = np.logical_and(np.isfinite(tmp_trace_1), np.isfinite(tmp_trace_2))
            if (np.count_nonzero(tmp_valid_Q) > 10):  
                tmp_corr = analyze_twp_traces_corr(tmp_trace_1, tmp_trace_2)
                tree_corr_mat[i, j] = tmp_corr['corr']
                tree_corr_mat[j, i] = tmp_corr['corr']
                tree_corr_p_mat[i, j] = tmp_corr['p']
                tree_corr_p_mat[j, i] = tmp_corr['p']
                # f_j = k_y2x * f_i, i.e. the linear response of edge j in response to the change in edge i
                tree_slope_mat[i, j] = tmp_corr['k_y2x'] 
                tree_slope_mat[j, i] = tmp_corr['k_x2y']
                tree_slope_n_mat[i, j] = tmp_corr['k_y2x_n']
                tree_slope_n_mat[j, i] = tmp_corr['k_x2y_n']
    
    return {'corr': tree_corr_mat, 'p': tree_corr_p_mat, 'k': tree_slope_mat, 'k_n': tree_slope_n_mat}

def compute_ratio_uncertainty(x, y, x_std, y_std): 
    s = x + y
    f_x = x / s
    f_y = y / s
    f_x_sigma = np.sqrt( (y * x_std/ s ** 2) ** 2 + (x * y_std / s ** 2) ** 2 )
    return f_x, f_y, f_x_sigma, f_x_sigma

def compute_point_cloud_basic_statistics(point_cloud, weights=None, compute_cov_Q=True, 
                                         compute_eig_Q=True):
    """
    Compute basic statistics for a d-dimensional point cloud.

    Args:
        point_cloud (np.ndarray): A (N, d) array of points.
        weights (np.ndarray, optional): A (N,) array of weights for each point.

    Returns:
        dict: A dictionary containing the mean and covariance of the point cloud.
            - mean: The mean of the point cloud (d-dimensional).
            - cov: The covariance matrix of the point cloud (d x d).
            - eig_s: The square root of eigenvalues of the covariance matrix (d-dimensional), sorted in descending order.
            - eig_v: The eigenvectors of the covariance matrix (d x d). Each column is the right eigenvector
    """
    compute_cov_Q = compute_cov_Q or compute_eig_Q

    point_cloud = np.atleast_2d(np.asarray(point_cloud))
    is_valid_Q = np.isfinite(point_cloud).all(axis=1)
    point_cloud = point_cloud[is_valid_Q]
    if weights is not None:
        weights = np.atleast_1d(np.asarray(weights))
        weights = weights[is_valid_Q]

    num_pts, num_d = point_cloud.shape
    stats = {'n': num_pts,
             'mean': np.full((num_d,), np.nan),
             'cov': np.full((num_d, num_d), np.nan), 
             'eig_s': np.full((num_d,), np.nan), 
             'eig_v': np.full((num_d, num_d), np.nan), 
             'tot_weight': np.nansum(weights) if weights is not None else num_pts}
    if num_pts > 0: 
        stats['mean'] = np.average(point_cloud, axis=0, weights=weights)
        if compute_cov_Q: 
            if num_pts > 1:
                stats['cov'] = np.cov(point_cloud - stats['mean'], rowvar=False, fweights=weights, ddof=0)
            else: 
                stats['cov'] = np.zeros((num_d, num_d))
            if compute_eig_Q: 
                if num_pts > 1: 
                    eig_val, eig_vec = np.linalg.eig(stats['cov'])
                    # sort eigenvalue in descending order
                    sorted_indices = np.argsort(eig_val)[::-1]
                    stats['eig_s'] = np.sqrt(eig_val[sorted_indices])
                    stats['eig_v'] = eig_vec[:, sorted_indices]
                else: 
                    stats['eig_s'] = np.zeros((num_d,))
                    stats['eig_v'] = np.eye(num_d)
            else: 
                stats.pop('eig_s', None)
                stats.pop('eig_v', None)
        else: 
            stats.pop('cov', None)

    return stats

def point_cloud_linear_outlier_rejection(point_pos, ipr=1.5): 
    """
    Reject outliers in a point cloud based on the inter-percentile range (IPR) method.

    Args:
        point_pos (np.ndarray): A (N, d) array of points.
        ipr (float): The multiplier for the inter-percentile range to define outliers.

    Returns:
        np.ndarray: A boolean array indicating which points are inliers (True) and which are outliers (False).
    """
    if point_pos.shape[0] < 3: 
        return point_pos
    pt_stat = compute_point_cloud_basic_statistics(point_pos, compute_cov_Q=True, compute_eig_Q=True)

    pt_mean = np.mean(point_pos, axis=0)
    pt_xyz_dm = point_pos - pt_mean
    pt_vec = pt_stat['eig_v']
    pt_std = pt_stat['eig_s']
    pt_std_ratio = pt_std[1] / pt_std[0]
    if pt_std_ratio > 0.5: 
        print(f"Warning: the second PC explains {pt_std_ratio:.3f} of the first PC, indicating a planar distribution of synapses.")
    # compute the residual 
    pt_res_dist = np.sqrt(np.sum((pt_xyz_dm @ pt_vec[:, 1:]) ** 2, axis=1))
    pt_res_lim = compute_percentile_outlier_threshold(pt_res_dist, ipr=ipr)
    pt_is_inlier_Q = (pt_res_dist <= pt_res_lim[1])
    point_pos = point_pos[pt_is_inlier_Q, :]
    
    return point_pos


def eigs_to_cov(eig_sigma, eig_v): 
    cov = eig_v @ np.diag(eig_sigma ** 2) @ eig_v.T
    return cov

def compute_point_cloud_dist_from_stat(stat1, stat2, project_to_eig1=True):
    """
    Compute the distance between two point cloud statistics from compute_point_cloud_basic_statistics
    Inputs: 
        - stat1: The first point cloud statistics
        - stat2: The second point cloud statistics.
    """
    dist_info = {}
    # Chi-squared test for 3D points
    dist_info['d_mu'] = stat2['mean'] - stat1['mean']
    dist_info['l_mu'] = np.linalg.norm(dist_info['d_mu'])
    dist_info['n_1'] = stat1['n']
    dist_info['n_2'] = stat2['n']
    if stat1['n'] < 2 and stat2['n'] < 2:
        dist_info['V'] = np.full((3, 3), np.nan)
    elif stat1['n'] < 2:
        sig = stat2['cov']
    elif stat2['n'] < 2: 
        sig = stat1['cov']
    else:
        sig = stat1['cov'] / stat1['n'] + stat2['cov'] / stat2['n']
    dist_info['T2'] = dist_info['d_mu'] @ np.linalg.pinv(sig) @ dist_info['d_mu']
    dist_info['D2'] = dist_info['d_mu'] @ np.linalg.pinv(stat1['cov']) @ dist_info['d_mu']
    # Project the mean from the second point cloud into the eigenvector 
    # of the first point cloud
    if project_to_eig1: 
        if 'eig_v' in stat1: 
            eig_v = stat1['eig_v']
            eig_s = stat1['eig_s']
        else: 
            eig_val, eig_v = np.linalg.eig(stat1['cov'])
            sorted_indices = np.argsort(eig_val)[::-1]
            eig_s = np.sqrt(eig_val[sorted_indices])
            eig_v = eig_v[:, sorted_indices]
        # Unify eigenvetor direction
        eig_v_d = eig_v.copy()
        for i in range(eig_v.shape[1]):
            tmp_idx = np.argmax(np.abs(eig_v_d[:, i]))
            if eig_v_d[tmp_idx, i] < 0:
                eig_v_d[:, i] *= -1
        dist_info['eig_v_1'] = eig_v_d
        dist_info['eig_s_1'] = eig_s
        dist_info['d_mu2eig_v1'] = eig_v_d.T @ dist_info['d_mu']

    return dist_info

def rebin_histogram(
    bin_val: np.ndarray,
    count: np.ndarray,
    new_bin_val: np.ndarray,
    *,
    input_is_pdf: bool = False,
    output_pdf: bool = False,
    even_tol: float = 1e-5
) -> np.ndarray:
    """
    Re-bin a 1D histogram from original bin centers to new bin centers via
    linear interpolation of the cumulative distribution (CDF). This preserves
    area/total count.

    Parameters
    ----------
    bin_val : (N,) np.ndarray
        Original bin centers (assumed evenly spaced and strictly increasing).
    count : (N,) np.ndarray
        Values at the original bins. If `input_is_pdf=False` these are counts
        (mass per bin). If `input_is_pdf=True` these are PDF values
        (mass per unit x) sampled at bin centers and assumed constant within
        each bin.
    new_bin_val : (M,) np.ndarray
        Target bin centers (assumed evenly spaced and strictly increasing).
    input_is_pdf : bool, default False
        When True, `count` is treated as PDF values; otherwise as bin counts.
    output_pdf : bool, default False
        When True, returns a PDF at `new_bin_val`; otherwise returns counts
        per new bin.
    even_tol : float, default 1e-9
        Tolerance for checking even spacing.

    Returns
    -------
    np.ndarray
        Re-binned values aligned with `new_bin_val`. Type matches `output_pdf`.

    Notes
    -----
    * Original and new bin spacings are inferred from the centers.
    * Outside the support of the original histogram, mass is taken as 0.
    * Complexity is O(N + M), using vectorized NumPy operations.
    """

    bin_val = np.asarray(bin_val, dtype=float)
    count   = np.asarray(count,   dtype=float)
    new_bin_val = np.asarray(new_bin_val, dtype=float)

    if bin_val.ndim != 1 or count.ndim != 1 or new_bin_val.ndim != 1:
        raise ValueError("All inputs must be 1D arrays.")
    if len(bin_val) != len(count):
        raise ValueError("bin_val and count must have the same length.")
    if len(bin_val) < 2 or len(new_bin_val) < 1:
        raise ValueError("Need at least 2 original bins and 1 target bin.")

    # Ensure strictly increasing
    if not (np.all(np.diff(bin_val) > 0) and np.all(np.diff(new_bin_val) > 0)):
        raise ValueError("bin_val and new_bin_val must be strictly increasing.")

    # Original spacing (must be even)
    dxs = np.diff(bin_val)
    dx  = float(np.mean(dxs))
    if np.max(np.abs(dxs - dx)) > even_tol * max(1.0, abs(dx)):
        raise ValueError("bin_val must be evenly spaced within tolerance.")

    # New spacing (must be even)
    if len(new_bin_val) > 1:
        ndxs = np.diff(new_bin_val)
        ndx  = float(np.mean(ndxs))
        if np.max(np.abs(ndxs - ndx)) > even_tol * max(1.0, abs(ndx)):
            raise ValueError("new_bin_val must be evenly spaced within tolerance.")
    else:
        # If only one new bin, choose the same width as original for PDF conversion
        ndx = dx

    # Build original edges from centers
    edges = np.concatenate(([bin_val[0] - 0.5 * dx],
                            0.5 * (bin_val[1:] + bin_val[:-1]),
                            [bin_val[-1] + 0.5 * dx]))

    # Convert inputs to per-bin mass and per-bin density
    if input_is_pdf:
        # mass in each original bin = pdf * dx (assume piecewise-constant within bin)
        bin_mass = count * dx
        density  = count                  # mass per unit x within each bin
    else:
        bin_mass = count
        density  = bin_mass / dx          # mass per unit x within each bin

    # CDF at left edges: cdf_edges[i] = mass up to edges[i]
    cdf_edges = np.concatenate(([0.0], np.cumsum(bin_mass)))
    total_mass = cdf_edges[-1]

    # Helper: evaluate CDF at arbitrary x via linear-in-bin interpolation.
    # Vectorized with searchsorted; outside support is clamped to [0, total_mass].
    def cdf_at(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        # Clip to support to keep indices valid; below becomes edges[0], above edges[-1]
        x_clipped = np.clip(x, edges[0], edges[-1])
        # Find bin index j such that edges[j] <= x < edges[j+1]; j in [0, N-1]
        j = np.searchsorted(edges, x_clipped, side="right") - 1
        j = np.clip(j, 0, len(density) - 1)
        dx_in = x_clipped - edges[j]
        # CDF is linear within bin with slope = density[j]
        return cdf_edges[j] + density[j] * dx_in

    # New bin edges from new centers
    new_edges = np.concatenate(([new_bin_val[0] - 0.5 * ndx],
                                0.5 * (new_bin_val[1:] + new_bin_val[:-1]),
                                [new_bin_val[-1] + 0.5 * ndx]))

    # Re-binned mass per new bin via CDF difference
    cdf_right = cdf_at(new_edges[1:])
    cdf_left  = cdf_at(new_edges[:-1])
    new_mass  = cdf_right - cdf_left

    if output_pdf:
        # Convert mass back to density (PDF) on the new uniform grid
        new_pdf = new_mass / ndx
        return new_pdf
    else:
        return new_mass
    

