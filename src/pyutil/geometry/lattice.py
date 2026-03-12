import numpy as np
import scipy.spatial as sps
from sklearn.cluster import KMeans
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt
import pyutil.stat as py_stat
from pyutil.geometry.point_cloud import PointCloud3DSurfaceFit
import pyutil


def analyze_two_lattice_xcorr(pt_set_1, pt_set_2, num_nb=1, num_auto_nb=1): 
    pt_set_1 = pt_set_1[np.all(np.isfinite(pt_set_1), axis=1)]
    pt_set_2 = pt_set_2[np.all(np.isfinite(pt_set_2), axis=1)]
    kdt_1 = sps.cKDTree(pt_set_1)
    kdt_2 = sps.cKDTree(pt_set_2)
    num_dim = pt_set_1.shape[1]
    # set 1 autocorrelation
    self_dists, self_idx = kdt_1.query(pt_set_1, k=num_auto_nb+1)
    self_idx = self_idx[:, 1:]  # remove self
    acorr_nb_vec_1 = np.zeros((pt_set_1.shape[0], num_auto_nb, num_dim))
    for i in range(pt_set_1.shape[0]):
        acorr_nb_vec_1[i] = pt_set_1[self_idx[i]] - pt_set_1[i]

    # set 2 in set 1 cross-correlation
    dists, idxs = kdt_1.query(pt_set_2, k=num_nb)
    xcorr_vec_2i1 = np.zeros((pt_set_2.shape[0], num_nb, num_dim))
    for i in range(pt_set_2.shape[0]):
        xcorr_vec_2i1[i] = pt_set_1[idxs[i]] - pt_set_2[i]
    # set 1 in set 2 cross-correlation
    dists, idxs = kdt_2.query(pt_set_1, k=num_nb)
    xcorr_vec_1i2 = np.zeros((pt_set_1.shape[0], num_nb, num_dim))
    for i in range(pt_set_1.shape[0]):
        xcorr_vec_1i2[i] = pt_set_2[idxs[i]] - pt_set_1[i]

    # set 2 autocorrelation
    self_dists, self_idx = kdt_2.query(pt_set_2, k=num_auto_nb+1)
    self_idx = self_idx[:, 1:]  # remove self
    acorr_nb_vec_2 = np.zeros((pt_set_2.shape[0], num_auto_nb, num_dim))
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
    num_bins = [21, 21]
    # hist_count is in [x, y] order, which is different from the usual [row, col] order of images.
    acorr_hist_1, x_edge, y_edge = np.histogram2d(tmp_x, tmp_y, bins=num_bins,
                                                        range=[x_range, y_range])
    tmp_u = xcorr_vec_2i1[:, :, 0].flatten()
    tmp_v = xcorr_vec_2i1[:, :, 1].flatten()
    xcorr_hist_2i1, _, _ = np.histogram2d(tmp_u, tmp_v, bins=num_bins,
                                            range=[x_range, y_range])
    tmp_u = xcorr_vec_1i2[:, :, 0].flatten()
    tmp_v = xcorr_vec_1i2[:, :, 1].flatten()
    xcorr_hist_1i2, _, _ = np.histogram2d(tmp_u, tmp_v, bins=num_bins,
                                            range=[x_range, y_range])

    tmp_u_self = acorr_nb_vec_2[:, :, 0].flatten()
    tmp_v_self = acorr_nb_vec_2[:, :, 1].flatten()
    acorr_hist_2, _, _ = np.histogram2d(tmp_u_self, tmp_v_self, bins=num_bins,
                                            range=[x_range, y_range])
    result = {
        'acorr_nb_vec_1': acorr_nb_vec_1,
        'xcorr_vec_2i1': xcorr_vec_2i1,
        'xcorr_vec_1i2': xcorr_vec_2i1,
        'acorr_nb_vec_2': acorr_nb_vec_2,
        'acorr_hist_1': acorr_hist_1,   
        'xcorr_hist_2i1': xcorr_hist_2i1,
        'xcorr_hist_1i2': xcorr_hist_1i2,
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

def vis_syn_ctr_pos(ref_pts, ct_pts, ref_ct=None, ct=None, 
                    x_label='u (nm)', y_label='v (nm)'): 
    f, a = plt.subplots(1, 2, figsize=(12, 6))
    a[0].scatter(ct_pts[:, 0], ct_pts[:, 1], s=10, label=ct, alpha=0.5)
    a[0].scatter(ref_pts[:, 0], ref_pts[:, 1],
             s=10, label=ref_ct, alpha=0.5)
    a[0].set_aspect('equal')

    a[0].legend()
    a[0].set_xlabel(x_label)
    a[0].set_ylabel(y_label)
    a[0].grid()

    a[1].scatter(ct_pts[:, 0], ct_pts[:, 1], s=10, label=ct, alpha=0.5)
    a[1].set_aspect('equal')
    a[1].legend()
    a[1].set_xlabel(x_label)
    a[1].set_ylabel(y_label)
    a[1].grid()
    f.tight_layout()
    return f, a

def vis_two_lattice_xcorr(result, x_label='u (nm)', y_label='v (nm)', 
                          xcorr_label='xcorr_hist_2i1'):
    vis_gamma = 1
    auto_im = pyutil.vis.imfuse_2d(result['acorr_hist_1'].T, result['acorr_hist_2'].T, gamma=vis_gamma)
    
    cross_im = pyutil.vis.imfuse_2d(result['acorr_hist_1'].T, result[xcorr_label].T, gamma=vis_gamma)

    f, a = plt.subplots(1, 2, figsize=(10, 5))
    a[0].imshow(auto_im, extent=[result['x_range'][0], result['x_range'][-1], 
                                result['y_range'][0], result['y_range'][-1]], origin='lower')
    a[0].set_xlabel(x_label)
    a[0].set_ylabel(y_label)
    a[0].grid()
    a[0].set_aspect('equal')
    a[1].imshow(cross_im, extent=[result['x_range'][0], result['x_range'][-1], 
                                result['y_range'][0], result['y_range'][-1]], origin='lower')
    a[1].set_xlabel(x_label)
    a[1].set_ylabel(y_label)
    a[1].grid()
    a[1].set_aspect('equal')
    f.tight_layout()
    return f, a

def vis_m_fold_orientation_order_map(pt_oo_info, m_fold_syn, tmp_cp_proj_uvw, 
                                     arrow_len=5000, x_label='u (nm)', y_label='v (nm)'): 
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
                color='red', alpha=0.5)
    f.colorbar(a.collections[0], ax=a, label='max oo m')
    a.set_aspect('equal')
    a.grid()
    a.set_xlabel(x_label)
    a.set_ylabel(y_label)
    f.tight_layout()
    return f, a


#region Translation 
def compute_reciprocal_vector_2d(data_pts, num_nb, max_dist, return_kdt_Q=False):
    """ Compute the reciprocal lattice vectors given the point cloud data of a unit cell.
    Input: 
        data_pts: (N, d) array of point coordinates.
        num_nb: number of nearest neighbors to consider for computing the lattice vectors.
        max_dist: maximum distance to consider for neighbors when computing the lattice vectors.
    
    """
    # Collect neighbor vectors from all points
    pt_kdt = sps.cKDTree(data_pts)
    nb_dist, nb_idx = pt_kdt.query(data_pts, k=num_nb+1)
    # remove self
    nb_dist = nb_dist[:, 1:]
    nb_idx = nb_idx[:, 1:]
    nb_vec = np.zeros((data_pts.shape[0], num_nb, data_pts.shape[1]))
    for i in range(data_pts.shape[0]):
        nb_vec[i] = data_pts[nb_idx[i]] - data_pts[i]
    # only consider the first two components 
    nb_x = nb_vec[:, :, 0].flatten()
    nb_y = nb_vec[:, :, 1].flatten()
    nb_valid_Q = (nb_dist < max_dist).flatten()
    nb_x = nb_x[nb_valid_Q]
    nb_y = nb_y[nb_valid_Q]

    # Use KMeans to find the main directions of the neighbor vectors
    kmeans = KMeans(n_clusters=num_nb)
    kmeans.fit(np.column_stack([nb_x, nb_y]))
    ctr_xy = kmeans.cluster_centers_
    ctr_len = np.linalg.norm(ctr_xy, axis=1)
    # vec_1_idx = np.nonzero(np.all(ctr_xy > 0, axis=1))[0][0]
    vec_1_idx = np.argmin(ctr_len)
    vec_1 = ctr_xy[vec_1_idx]
    vec_2_idx = np.argmin(np.abs(ctr_xy @ vec_1[:, None]))
    vec_2 = ctr_xy[vec_2_idx]
    G_mat = np.linalg.inv(np.column_stack((vec_1, vec_2))) * 2 * np.pi
    if return_kdt_Q: 
        return G_mat, pt_kdt
    else: 
        return G_mat

def compute_translation_correlation(data_pts, G_mat, num_nb, max_dist, bin_width, 
                                        pt_kdt=None): 

    bin_edges = np.arange(-bin_width/2, max_dist + 3 * bin_width/2, bin_width)
    num_bin = bin_edges.shape[0] - 1
    bin_val = bin_edges[:-1] + bin_width / 2
    # Get neighbors
    if pt_kdt is None:
        pt_kdt = sps.cKDTree(data_pts)
    nb_dist, nb_idx = pt_kdt.query(data_pts, k=num_nb+1, distance_upper_bound=max_dist)
    nb_dist = nb_dist[:, 1:]
    nb_idx = nb_idx[:, 1:]

    trans_order = np.full((nb_dist.shape[0], num_bin), np.nan, dtype=np.complex128)
    for r_max_idx in range(data_pts.shape[0]):
        tmp_nb_dist = nb_dist[r_max_idx]
        tmp_nb_idx = nb_idx[r_max_idx]
        tmp_nb_valid_Q = (tmp_nb_dist < max_dist)
        tmp_nb_dist = tmp_nb_dist[tmp_nb_valid_Q]
        tmp_nb_idx = tmp_nb_idx[tmp_nb_valid_Q]
        # site-centered
        tmp_nb_vec = data_pts[tmp_nb_idx] - data_pts[r_max_idx]
        tmp_nb_vec_12 = tmp_nb_vec[:, 0:2]
        tmp_nb_vec_phi = G_mat @ tmp_nb_vec_12.T
        tmp_nb_rho = np.exp(1j * tmp_nb_vec_phi)

        tmp_nb_bin_idx = np.round(tmp_nb_dist / bin_width).astype(np.int32)
        tmp_bin_idx = pyutil.util.bin_data_to_idx_list(tmp_nb_bin_idx, return_type='dict')
        tmp_to = np.full(bin_edges.shape[0] - 1, np.nan, dtype=np.complex128)
        for k, idx_list in tmp_bin_idx.items():
            tmp_to[k] = np.mean(tmp_nb_rho[0][idx_list])
        tmp_to[np.isnan(tmp_to)] = 0
        trans_order[r_max_idx] = tmp_to

    result = {
        'bin_edges': bin_edges,
        'bin_val': bin_val,
        'pts_translation_order': trans_order, 
        'avg_translation_order': np.nanmean(trans_order, axis=0)
    }
    result['avg_translation_order_abs'] = np.abs(result['avg_translation_order'])
    result['avg_translation_order_r'] = np.real(result['avg_translation_order'])
    r_max_idx = np.argmax(np.real(result['avg_translation_order_r']))
    abs_max_idx = np.argmax(np.real(result['avg_translation_order_abs']))
    result['peak_bin_val_r'] = bin_val[r_max_idx]
    result['peak_bin_val_abs'] = bin_val[abs_max_idx]
    result['bin_val_n_r'] = bin_val / result['peak_bin_val_r']
    result['bin_val_n_abs'] = bin_val / result['peak_bin_val_abs']
    return result

def vis_translation_correlation(result, title=None,
                                x_label='r/a', y_label='Translational correlation', 
                                x_key='bin_val_n_abs'): 
    x = result[x_key]
    max_bin = np.ceil(np.max(x))
    lattice_r = np.arange(max_bin) ** 2
    lattice_r = np.unique(np.sqrt(lattice_r[:, None] + lattice_r[None, :]))
    lattice_r = lattice_r[(lattice_r > 0) & (lattice_r < max_bin)]

    f, a = plt.subplots(1, 1, figsize=(5, 4))
    a.plot(x, result['avg_translation_order_r'], label='Re')
    a.plot(x, result['avg_translation_order_abs'], label='Abs')
    for i in lattice_r: 
        a.axvline(i, color='gray', linestyle='--', alpha=0.5)
    a.set_xlabel(x_label)
    a.set_ylabel(y_label)
    vis_min = np.round(np.nanmin(result['avg_translation_order_r']), 2) 
    vis_min = np.minimum(-0.05, vis_min)
    a.set_ylim([vis_min, 1.05])
    a.legend()
    a.grid()
    a.set_title(title)
    return f, a
#endregion

#region Radial 
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

def vis_radial_distribution_function(result, title=None, 
                                x_label='r (nm)', y_label='g(r)'):
    f, a = plt.subplots(1, 1, figsize=(5, 4))
    a.plot(result['hist_bin_val'], result['avg_r_density'], label='g(r)')
    a.axvline(result['peak_dist'], color='red', linestyle='--', label=f'Peak at {result["peak_dist"]:.2f} nm')
    a.set_xlabel(x_label)
    a.set_ylabel(y_label)
    a.legend()
    a.grid()
    a.set_title(title)
    f.tight_layout()
    return f, a

#endregion