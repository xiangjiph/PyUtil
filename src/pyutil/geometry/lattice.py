import numpy as np
import scipy.spatial as sps
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt
import pyutil.stat as py_stat
from pyutil.geometry.point_cloud import PointCloud3DSurfaceFit



def analyze_two_lattice_xcorr(pt_set_1, pt_set_2, num_nb=1, num_auto_nb=1): 
    kdt_1 = sps.cKDTree(pt_set_1)
    kdt_2 = sps.cKDTree(pt_set_2)
    # set 1 autocorrelation
    self_dists, self_idx = kdt_1.query(pt_set_1, k=num_auto_nb+1)
    self_idx = self_idx[:, 1:]  # remove self
    acorr_nb_vec_1 = np.zeros((pt_set_1.shape[0], num_auto_nb, 3))
    for i in range(pt_set_1.shape[0]):
        acorr_nb_vec_1[i] = pt_set_1[self_idx[i]] - pt_set_1[i]

    # set 2 in set 1 cross-correlation
    dists, idxs = kdt_1.query(pt_set_2, k=num_nb)
    xcorr_vec = np.zeros((pt_set_2.shape[0], num_nb, 3))
    for i in range(pt_set_2.shape[0]):
        xcorr_vec[i] = pt_set_1[idxs[i]] - pt_set_2[i]

    # set 2 autocorrelation
    self_dists, self_idx = kdt_2.query(pt_set_2, k=num_auto_nb+1)
    self_idx = self_idx[:, 1:]  # remove self
    acorr_nb_vec_2 = np.zeros((pt_set_2.shape[0], num_auto_nb, 3))
    for i in range(pt_set_2.shape[0]):
        acorr_nb_vec_2[i] = pt_set_2[self_idx[i]] - pt_set_2[i]

    tmp_x = acorr_nb_vec_1[:, :, 0].flatten()
    tmp_y = acorr_nb_vec_1[:, :, 1].flatten()
    # x_range = np.percentile(tmp_x, [0.5, 99.5])
    x_range = np.percentile(acorr_nb_vec_2[:, :, 0].flatten(), [0.5, 99.5])
    x_range = np.asarray([-1, 1]) * max(abs(x_range))
    # y_range = np.percentile(tmp_y, [0.5, 99.5])
    y_range = np.percentile(acorr_nb_vec_2[:, :, 1].flatten(), [0.5, 99.5])
    y_range = np.asarray([-1, 1]) * max(abs(y_range))
    num_bins = [20, 20]
    # hist_count is in [x, y] order, which is different from the usual [row, col] order of images.
    acorr_hist_1, x_edge, y_edge = np.histogram2d(tmp_x, tmp_y, bins=num_bins,
                                                        range=[x_range, y_range])
    tmp_u = xcorr_vec[:, :, 0].flatten()
    tmp_v = xcorr_vec[:, :, 1].flatten()
    xcorr_hist, _, _ = np.histogram2d(tmp_u, tmp_v, bins=num_bins,
                                            range=[x_range, y_range])

    tmp_u_self = acorr_nb_vec_2[:, :, 0].flatten()
    tmp_v_self = acorr_nb_vec_2[:, :, 1].flatten()
    acorr_hist_2, _, _ = np.histogram2d(tmp_u_self, tmp_v_self, bins=num_bins,
                                            range=[x_range, y_range])
    result = {
        'acorr_nb_vec_1': acorr_nb_vec_1,
        'xcorr_vec': xcorr_vec,
        'acorr_nb_vec_2': acorr_nb_vec_2,
        'acorr_hist_1': acorr_hist_1,   
        'xcorr_hist': xcorr_hist,
        'acorr_hist_2': acorr_hist_2,
        'x_range': x_range,
        'y_range': y_range,
    }
    return result

def compute_single_point_orientation_order(vecs, m_list): 
    # assume vecs are in shape (N, 2) and sorted 
    # by the distnacen to the reference point. 
    m_list = np.atleast_1d(np.asarray(m_list))
    vecs_n = vecs / np.linalg.norm(vecs, axis=-1, keepdims=True)
    theta = np.arctan2(vecs_n[:, 1], vecs_n[:, 0])
    psi_m = np.full(m_list.shape, fill_value=np.nan, dtype=np.complex64)
    for i, m in enumerate(m_list):
        if m <= vecs.shape[0]:
            psi_m[i] = np.sum(np.exp(1j * m * theta[:m])) / m

    if len(m_list) == 1:
        psi_m = psi_m[0]

    return psi_m

def compute_orientation_order(data_pts, max_dist, max_knn, m_list=None, 
                              surf_obj=None): 
    pt_kdt = sps.cKDTree(data_pts)
    nb_dist, nb_idx = pt_kdt.query(data_pts, k=max_knn+1, distance_upper_bound=max_dist)
    nb_dist = nb_dist[:, 1:]
    nb_idx = nb_idx[:, 1:]

    if m_list is None:
        m_list = np.arange(3, max_knn+1)
    pt_oo = np.full((data_pts.shape[0], m_list.shape[0]), np.nan, dtype=np.complex64)
    pt_oo_n = np.full((data_pts.shape[0],), np.nan, dtype=np.float32)
    pt_max_oo_nb = np.zeros(data_pts.shape[0], dtype=np.int32)
    pt_max_oo = np.full((data_pts.shape[0],), np.nan, dtype=np.complex64)

    for i in range(data_pts.shape[0]):
        tmp_dist = nb_dist[i]
        tmp_idx = nb_idx[i]
        tmp_valid_Q = tmp_dist <= max_dist
        tmp_idx = tmp_idx[tmp_valid_Q]
        if surf_obj is None: 
            tmp_vecs = data_pts[tmp_idx] - data_pts[i]
        else: 
            tmp_vecs = PointCloud3DSurfaceFit.uvw_to_tangent_plane(data_pts[tmp_idx], data_pts[i], 
                                                    surf_obj.coeffs)
        # consider the first 2 components at the moment 
        # Not sure how to deal with 3D 
        tmp_oo = compute_single_point_orientation_order(tmp_vecs[:, :2], m_list=m_list)
        if np.all(np.isnan(tmp_oo)):
            continue
        tmp_oo_abs = np.abs(tmp_oo)
        tmp_max_idx = np.nanargmax(tmp_oo_abs)
        pt_max_oo_nb[i] = m_list[tmp_max_idx]
        pt_oo[i] = tmp_oo
        pt_max_oo[i] = tmp_oo[tmp_max_idx]
        pt_oo_n[i] = tmp_oo_abs[tmp_max_idx]
    
    result = {
        'm_list': m_list,
        'oo': pt_oo,
        'm_oo': pt_max_oo, 
        'm_oo_nn': pt_max_oo_nb,
        'm_oo_n': pt_oo_n,
    }
    return result

def compute_radial_distribution_function(pts, max_dist, max_knn, num_bins, remove_self_Q=True): 
    pts = np.asarray(pts).astype(np.float32)

    hist_bins = np.arange(0, max_dist+1e-9, max_dist/num_bins)
    hist_bin_val = (hist_bins[:-1] + hist_bins[1:]) / 2
    hist_bin_area = np.pi * (hist_bins[1:]**2 - hist_bins[:-1]**2)
    pts_r_counts = np.full((pts.shape[0], num_bins), fill_value=np.nan)

    pt_kdt = sps.cKDTree(pts)

    for i, tmp_pt in enumerate(pts):
        tmp_dist, tmp_idx = pt_kdt.query(tmp_pt, k=max_knn, distance_upper_bound=max_dist)
        tmp_valid_Q = tmp_dist <= max_dist
        if remove_self_Q:
            assert tmp_dist[0] == 0, 'The closest point should be itself with distance 0.'
            tmp_valid_Q[0] = False
        if tmp_valid_Q[-1] == True: 
            print(f'Warning: point {i} has more than {max_knn} neighbors within {max_dist} distance. Consider increasing max_knn or max_dist.')
        tmp_dist = tmp_dist[tmp_valid_Q]
        tmp_idx = tmp_idx[tmp_valid_Q]
        tmp_counts = np.histogram(tmp_dist, bins=hist_bins)[0].astype(np.float32)
        # set the trailing 0 to nan
        for j in range(len(tmp_counts)-1, -1, -1):
            if tmp_counts[j] > 0:
                break
            else:
                tmp_counts[j] = np.nan
        pts_r_counts[i] = tmp_counts
    
    avg_r_counts = np.nanmean(pts_r_counts, axis=0)
    avg_r_den = avg_r_counts / hist_bin_area
    bond_length = hist_bin_val[np.argmax(avg_r_den)]

    result = {
        'hist_bins': hist_bins,
        'hist_bin_val': hist_bin_val,
        'hist_bin_area': hist_bin_area,
        'pts_r_counts': pts_r_counts,
        'avg_r_counts': avg_r_counts, 
        'avg_r_density': avg_r_den, 
        'peak_dist': bond_length
    }

    return result

def compute_orientation_order_correlation(data_pts, pt_ori_order, 
                                          max_knn, max_dist, num_bins, 
                                          selection_Q=None):
    """ Compute the orientation order correlation as a function of distance. 
    Inputs: 
        data_pts: (N, d) array of point coordinates. 
        pt_ori_order: (N,) array of complex orientation order for each point. 
        max_knn: maximum number of neighbors to consider for each point. 
        max_dist: maximum distance to consider for neighbors. 
        num_bins: number of bins to use for distance binning.
    Outputs: 
        bin_oo_diff: (N, num_bins) array of mean orientation order correlation in each distance bin for each point. 
        hist_bin_val: (num_bins,) array of the center value of each distance bin. 
    """
    pt_kdt = sps.cKDTree(data_pts)
    nb_dist, nb_idx = pt_kdt.query(data_pts, k=max_knn+1, distance_upper_bound=max_dist)
    # do not remove self
    hist_bins = np.arange(-1e-9, max_dist+1e-9, max_dist/num_bins)
    hist_bin_val = (hist_bins[:-1] + hist_bins[1:]) / 2

    bin_oo_diff = np.full((data_pts.shape[0], hist_bin_val.size), 
                          fill_value=np.nan, dtype=np.complex64)
    
    for tmp_idx in range(data_pts.shape[0]):
        tmp_dist = nb_dist[tmp_idx]
        assert (tmp_dist[0] == 0), 'The closest point should be itself with distance 0.'
        tmp_nb_idx = nb_idx[tmp_idx]
        tmp_valid_Q = tmp_dist <= max_dist            
        if np.any(tmp_valid_Q): 
            tmp_dist = tmp_dist[tmp_valid_Q]
            tmp_nb_idx = tmp_nb_idx[tmp_valid_Q]
            if selection_Q is not None:
                tmp_valid_Q = selection_Q[tmp_nb_idx]
                tmp_dist = tmp_dist[tmp_valid_Q]
                tmp_nb_idx = tmp_nb_idx[tmp_valid_Q]

            # compute the orientation order correlation
            tmp_self_oo = np.conjugate(pt_ori_order[tmp_idx])
            tmp_nb_oo = pt_ori_order[tmp_nb_idx]
            tmp_oo_diff = tmp_self_oo * tmp_nb_oo
            try:
                tmp = binned_statistic(tmp_dist, tmp_oo_diff, statistic='mean', bins=hist_bins)
                bin_oo_diff[tmp_idx] = tmp.statistic
            except Exception as e:
                # print(f"Error in binned_statistic for point {tmp_idx}: {e}")
                continue
    
    return bin_oo_diff, hist_bin_val

def analyze_orientation_order_correlation(data_pts, oo_knn, oo_max_dist,  
                                          r_knn, r_max_dist, r_num_bins,
                                          match_oo_m_Q=True, surf_obj=None):
    
    m_list = np.arange(3, oo_knn+1)
    pt_oo_info = compute_orientation_order(data_pts, oo_max_dist, oo_knn, m_list, 
                                           surf_obj=surf_obj)
    m_oo_corr = {}
    for i, m in enumerate(m_list):        
        selection_Q = (pt_oo_info['m_oo_nn'] == m) if match_oo_m_Q else None

        if selection_Q is None or np.any(selection_Q):
            bin_oo_diff, hist_bin_val = compute_orientation_order_correlation(data_pts, 
                                pt_oo_info['oo'][:, i], max_knn=r_knn, 
                                max_dist=r_max_dist, num_bins=r_num_bins, 
                                selection_Q=selection_Q)
            
            tmp_selected_Q = (pt_oo_info['m_oo_nn'] == m)
            bin_oo_diff_n = np.nanmean(np.abs(bin_oo_diff[tmp_selected_Q]), axis=0)
            bin_oo_diff_mean_abs = np.abs(np.nanmean(bin_oo_diff[tmp_selected_Q], axis=0))
            bin_oo_diff_ptrl = np.abs(py_stat.percentile(bin_oo_diff[tmp_selected_Q], [25, 50, 75], axis=0))

            m_oo_corr[m] = {
                'bin_oo_diff': bin_oo_diff,
                'hist_bin_val': hist_bin_val,
                'bin_oo_diff_n': bin_oo_diff_n,
                'bin_oo_diff_mean_abs': bin_oo_diff_mean_abs,
                'bin_oo_diff_ptrl': bin_oo_diff_ptrl
            }
            print(f"Finished analyzing orientation order correlation for m={m}.")
        else: 
            print(f"No points with m_oo_nn={m}, skipping analysis.")

    return pt_oo_info, m_oo_corr    
    
def vis_orientation_order_correlation(m_oo_corr, pt_oo_info, x_label='r (nm)', 
                                      y_label='Orientation order correlation'):
    
    f, a = plt.subplots(2, 1, figsize=(6, 6))
    for i, syn_fold in enumerate(m_oo_corr.keys()):
        vis_data = m_oo_corr[syn_fold]
        vis_num_pts = np.sum(pt_oo_info['m_oo_nn'] == syn_fold)
        vis_x = vis_data['hist_bin_val']
        bin_oo_diff_n = vis_data['bin_oo_diff_n']
        bin_oo_diff_mean_abs = vis_data['bin_oo_diff_mean_abs']
        bin_oo_diff_n[np.isnan(bin_oo_diff_n)] = 0
        bin_oo_diff_mean_abs[np.isnan(bin_oo_diff_mean_abs)] = 0
        a[0].plot(vis_x, bin_oo_diff_n, '-', label=f'{syn_fold}-fold ({vis_num_pts} pts)', alpha=0.7)
        a[1].plot(vis_x, bin_oo_diff_mean_abs, '-', label=f'{syn_fold}-fold ({vis_num_pts} pts)', alpha=0.7)

    a[0].set_xlabel(x_label)
    a[1].set_xlabel(x_label)
    a[0].set_ylabel(y_label)
    a[1].set_ylabel(y_label)
    a[0].legend()
    a[1].legend()
    a[0].grid()
    a[1].grid()
    a[0].set_ylim(-0.05, 1.05)
    a[1].set_ylim(-0.05, 1.05)
    return f, a

def vis_syn_ctr_pos(ref_pts, ct_pts, ref_ct=None, ct=None): 
    f, a = plt.subplots(1, 2, figsize=(12, 6))
    a[0].scatter(ct_pts[:, 0], ct_pts[:, 1], s=10, label=ct, alpha=0.5)
    a[0].scatter(ref_pts[:, 0], ref_pts[:, 1],
             s=10, label=ref_ct, alpha=0.5)
    a[0].set_aspect('equal')

    a[0].legend()
    a[0].set_xlabel('u (nm)')
    a[0].set_ylabel('v (nm)')
    a[0].grid()

    a[1].scatter(ct_pts[:, 0], ct_pts[:, 1], s=10, label=ct, alpha=0.5)
    a[1].set_aspect('equal')
    a[1].legend()
    a[1].set_xlabel('u (nm)')
    a[1].set_ylabel('v (nm)')
    a[1].grid()
    f.tight_layout()
    return f, a

def vis_two_lattice_xcorr(result):
    vis_gamma = 1
    auto_im = pyutil.vis.imfuse_2d(result['acorr_hist_1'].T, result['acorr_hist_2'].T, gamma=vis_gamma)
    cross_im = pyutil.vis.imfuse_2d(result['acorr_hist_1'].T, result['xcorr_hist'].T, gamma=vis_gamma)

    f, a = plt.subplots(1, 2, figsize=(10, 5))
    a[0].imshow(auto_im, extent=[result['x_range'][0], result['x_range'][-1], 
                                result['y_range'][0], result['y_range'][-1]], origin='lower')
    a[0].set_xlabel('u (nm)')
    a[0].set_ylabel('v (nm)')
    a[0].grid()
    a[0].set_aspect('equal')
    a[1].imshow(cross_im, extent=[result['x_range'][0], result['x_range'][-1], 
                                result['y_range'][0], result['y_range'][-1]], origin='lower')
    a[1].set_xlabel('u (nm)')
    a[1].set_ylabel('v (nm)')
    a[1].grid()
    a[1].set_aspect('equal')
    f.tight_layout()
    return f, a

def vis_m_fold_orientation_order_map(pt_oo_info, m_fold_syn, tmp_cp_proj_uvw, 
                                     arrow_len=5000): 
    oo_idx = int(np.nonzero(pt_oo_info['m_list'] == m_fold_syn)[0])
    opt_oo = pt_oo_info['oo'][:, oo_idx]
    # opt_oo_arg = np.angle(opt_oo)
    # opt_oo_ep =  np.column_stack([
    #         tmp_cp_proj_uvw[:, 0] + arrow_len * np.cos(opt_oo_arg), 
    #         tmp_cp_proj_uvw[:, 1] + arrow_len * np.sin(opt_oo_arg)
    # ])
    opt_oo_ep =  np.column_stack([
            tmp_cp_proj_uvw[:, 0] + arrow_len * np.real(opt_oo), 
            tmp_cp_proj_uvw[:, 1] + arrow_len * np.imag(opt_oo)
    ])

    # Visualize orientation of each point 
    f, a = plt.subplots(1, 1, figsize=(10, 6))
    a.scatter(tmp_cp_proj_uvw[:, 0], tmp_cp_proj_uvw[:, 1],
            c=pt_oo_info['m_oo_nn'], cmap='jet', 
            s=20, vmin=3)
    for tmp_uv, tmp_uv1 in zip(tmp_cp_proj_uvw[:, [0, 1]], opt_oo_ep): 
        plt.arrow(tmp_uv[0], tmp_uv[1], 
                  tmp_uv1[0]-tmp_uv[0], tmp_uv1[1]-tmp_uv[1], 
                color='red', alpha=0.5, head_width=20, head_length=30)
    f.colorbar(a.collections[0], ax=a, label='max oo m')
    a.set_aspect('equal')
    a.grid()
    a.set_xlabel('u (nm)')
    a.set_ylabel('v (nm)')
    f.tight_layout()
    return f, a