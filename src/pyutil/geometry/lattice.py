import numpy as np
import scipy.spatial as sps
from sklearn.cluster import KMeans
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt
from .. import stat as py_stat
from .. import vis as py_vis
from .. import util as py_util
from . import point_cloud as pypc
from scipy.spatial import Voronoi

# to be deleted. 
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
    # for each set 1 point, find the nearest set 2 points. 
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
        'acorr_nb_vec_1': acorr_nb_vec_1, # (num_pts_1, num_auto_nb, num_dim)
        'xcorr_vec_2i1': xcorr_vec_2i1, # (num_pts_2, num_nb, num_dim)
        'xcorr_vec_1i2': xcorr_vec_1i2, # (num_pts_1, num_nb, num_dim)
        'acorr_nb_vec_2': acorr_nb_vec_2, # (num_pts_2, num_auto_nb, num_dim)
        'acorr_hist_1': acorr_hist_1,   # (num_bins, num_bins)
        'xcorr_hist_2i1': xcorr_hist_2i1, # (num_bins, num_bins)
        'xcorr_hist_1i2': xcorr_hist_1i2, # (num_bins, num_bins)
        'acorr_hist_2': acorr_hist_2,
        'x_range': x_range,
        'y_range': y_range,
    }
    return result

def compute_single_point_orientation_order(vecs, m_list, num_nb=4): 
    # assume vecs are in shape (N, 2) and sorted 
    # by the distance to the reference point. 
    m_list = np.atleast_1d(np.asarray(m_list))
    vecs_n = vecs / np.linalg.norm(vecs, axis=-1, keepdims=True)
    theta = np.arctan2(vecs_n[:, 1], vecs_n[:, 0])
    psi_m = np.full(m_list.shape, fill_value=np.nan, dtype=np.complex64)
    for i, m in enumerate(m_list):
        if num_nb is None: 
            # This is not quite correct. 
            # orientation order should be computed with fix number of
            # neighbors. 
            if m <= vecs.shape[0]:
                psi_m[i] = np.sum(np.exp(1j * m * theta[:m])) / m
        else: 
            psi_m[i] = np.sum(np.exp(1j * m * theta[:num_nb])) / num_nb

    if len(m_list) == 1:
        psi_m = psi_m[0]
    assert np.all((np.abs(psi_m) <= (1 + 1e-6)) | np.isnan(psi_m)), 'The magnitude of orientation order should be less than or equal to 1.'
    return psi_m

def compute_single_point_orientation_order_with_voronoi_neighbors(\
        nb_vecs, m_list, weight_exponent=1, return_info_Q=False): 
    nb_vecs = np.atleast_2d(nb_vecs)
    all_vecs = np.vstack([np.asarray([[0, 0]]), nb_vecs]) # add self 
    # Get neighbors using Voronoi tessellation
    if all_vecs.shape[0] < 4:
        # Not enough points to compute Voronoi tessellation
        psi_m = np.full((len(m_list),), fill_value=np.nan, dtype=np.complex64)
        if return_info_Q:
            return psi_m, {'nb_theta': np.full((0,), fill_value=np.nan), 
                           'nb_len': np.full((0,), fill_value=np.nan), 
                           'weight': np.full((0,), fill_value=np.nan)}
        return psi_m
    vq = VoronoiQuery(all_vecs)
    nb_idx, nb_len = vq.neighbors(0)
    # compute neithbor theta 
    vq_nb_vec = all_vecs[nb_idx]
    vq_nb_vec = vq_nb_vec / np.linalg.norm(vq_nb_vec, axis=-1, keepdims=True)
    vq_nb_theta = np.atleast_1d(np.arctan2(vq_nb_vec[:, 1], vq_nb_vec[:, 0]))
    # weight by shared edge length 
    weight = np.power(nb_len, weight_exponent)
    weight = weight / np.sum(weight)

    m_list = np.atleast_1d(np.asarray(m_list)).reshape(-1, 1)
    psi_m = np.sum(weight * np.exp(1j * m_list * vq_nb_theta), axis=1)
    if len(m_list) == 1:
        psi_m = psi_m[0]

    assert np.all((np.abs(psi_m) <= (1 + 1e-6)) | np.isnan(psi_m)), 'The magnitude of orientation order should be less than or equal to 1.'
    if return_info_Q:
        return psi_m, {'nb_theta': vq_nb_theta, 'nb_len': nb_len, 'weight': weight}
    return psi_m

def compute_orientation_order(data_pts, max_dist, max_knn, m_list=None, 
                              surf_obj=None, align_axis=0,
                              orthogonalized_Q=True): 
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
            if isinstance(surf_obj, pypc.PCSurface3D):
                tmp_vecs = surf_obj.uvw_to_tangent_plane(data_pts[tmp_idx], data_pts[i], 
                                                         align_axis=align_axis,
                                                         orthogonalized_Q=orthogonalized_Q)
            else: 
                tmp_vecs = pypc.PointCloud3DSurfaceFit.uvw_to_tangent_plane(data_pts[tmp_idx], data_pts[i], 
                                                        surf_obj.coeffs)
        # consider the first 2 components at the moment 
        # Not sure how to deal with 3D 
        tmp_oo = compute_single_point_orientation_order(tmp_vecs[:, :2], m_list=m_list, 
                                                        num_nb=max_knn)
        if np.all(np.isnan(tmp_oo)):
            continue
        tmp_oo_abs = np.abs(tmp_oo)
        tmp_max_idx = np.nanargmax(tmp_oo_abs)
        pt_max_oo_nb[i] = m_list[tmp_max_idx]
        pt_oo[i] = tmp_oo
        pt_max_oo[i] = tmp_oo[tmp_max_idx]
        pt_oo_n[i] = tmp_oo_abs[tmp_max_idx]
    
    knn_stat = {}
    for tmp_m in m_list: 
        tmp_nb_dist = nb_dist[:, :tmp_m] if tmp_m <= nb_dist.shape[1] else nb_dist
        tmp_nb_dist = tmp_nb_dist[tmp_nb_dist <= max_dist]
        tmp_stat = {'mean': np.mean(tmp_nb_dist), 'median': np.median(tmp_nb_dist),
                    'p25': np.percentile(tmp_nb_dist, 25), 'p75': np.percentile(tmp_nb_dist, 75)}
        knn_stat[int(tmp_m)] = tmp_stat

    result = {
        'm_list': m_list,
        'num_nb': max_knn,
        'oo': pt_oo, # orientation order
        'm_oo': pt_max_oo, # orientation order with maximum magnitude
        'm_oo_nn': pt_max_oo_nb, # m that maximize the orientation order
        'm_oo_n': pt_oo_n, # maximum orientation order magnitude
        'knn_stat': knn_stat
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
                                          match_oo_m_Q=False, surf_obj=None, 
                                          align_axis=0,
                                          orthogonalized_Q=True, 
                                          m_list=None):
    
    m_list = np.arange(2, oo_knn+1) if m_list is None else m_list
    pt_oo_info = compute_orientation_order(data_pts, oo_max_dist, oo_knn, m_list, 
                                           surf_obj=surf_obj, align_axis=align_axis,
                                           orthogonalized_Q=orthogonalized_Q)
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

            m_oo_corr[int(m)] = {
                'bin_oo_diff': bin_oo_diff,
                'hist_bin_val': hist_bin_val,
                'bin_oo_diff_n': bin_oo_diff_n,
                'bin_oo_diff_mean_abs': bin_oo_diff_mean_abs,
                'bin_oo_diff_ptrl': bin_oo_diff_ptrl
            }
            # print(f"Finished analyzing orientation order correlation for m={m}.")
        else: 
            print(f"No points with m_oo_nn={m}, skipping analysis.")

    return pt_oo_info, m_oo_corr    

def vis_orientation_order_correlation(m_oo_corr, pt_oo_info, y_key='bin_oo_diff_mean_abs',
                                      x_label='r (nm)', 
                                      y_label='Orientation order correlation', 
                                      normalized_dist_Q=False, title_prefix=''):
    f, a = plt.subplots(1, 1, figsize=(6, 4))
    if normalized_dist_Q: 
        x_label = 'r / <a>'
    for i, syn_fold in enumerate(m_oo_corr.keys()):
        vis_data = m_oo_corr[syn_fold]
        vis_num_pts = np.sum(pt_oo_info['m_oo_nn'] == syn_fold)
        vis_x = vis_data['hist_bin_val']
        if normalized_dist_Q: 
            vis_x = vis_x / pt_oo_info['knn_stat'][syn_fold]['mean']

        y_val = vis_data[y_key]
        y_val[np.isnan(y_val)] = 0
        tmp_label_str = f"{syn_fold}-fold ({vis_num_pts} pts)"
        if 'ooc_avg_norm_lra' in vis_data:
            tmp_label_str += f", LRA: {vis_data['ooc_avg_norm_lra']:.2e}"
        a.plot(vis_x, y_val, '-', label=tmp_label_str, alpha=0.7)
    a.grid()
    a.set_xlabel(x_label)
    a.set_ylabel(y_label)
    a.set_ylim(-0.05, 1.05)
    a.legend()
    a.set_title(f"{title_prefix} OOC ({pt_oo_info['num_nb']}nn)")
    f.tight_layout()

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

def vis_two_lattice_auto_corr(data, x_label='u (nm)', y_label='v (nm)', 
                              cmap='jet'):
    
    f, a = plt.subplots(1, 2, figsize=(12, 5))
    mi1_self_hist = data['acorr_hist_1'].T
    ct_self_hist = data['acorr_hist_2'].T

    im_1 = a[0].imshow(mi1_self_hist, extent=[data['x_range'][0], data['x_range'][-1], 
                                        data['y_range'][0], data['y_range'][-1]], origin='lower', 
                                        cmap=cmap)
    cbar_1 = f.colorbar(im_1, ax=a[0], label='Count')
    im_2 = a[1].imshow(ct_self_hist, extent=[data['x_range'][0], data['x_range'][-1], 
                                        data['y_range'][0], data['y_range'][-1]], origin='lower', 
                                        cmap=cmap)
    cbar_2 = f.colorbar(im_2, ax=a[1], label='Count')

    f.tight_layout()
    a[0].set_xlabel(x_label)
    a[0].set_ylabel(y_label)
    a[0].grid(True)
    a[1].set_xlabel(x_label)
    a[1].set_ylabel(y_label)
    a[1].grid(True)
    return f, a

def vis_two_lattice_xcorr(result, x_label='u (nm)', y_label='v (nm)', 
                          xcorr_label='xcorr_hist_1i2'):
    vis_gamma = 1
    auto_im = py_vis.imfuse_2d(result['acorr_hist_1'].T, result['acorr_hist_2'].T, gamma=vis_gamma)
    
    cross_im = py_vis.imfuse_2d(result['acorr_hist_1'].T, result[xcorr_label].T, gamma=vis_gamma)

    f, a = plt.subplots(1, 2, figsize=(10, 5))
    a[0].imshow(auto_im, extent=[result['x_range'][0], result['x_range'][-1], 
                                result['y_range'][0], result['y_range'][-1]], origin='lower')
    # a[1].scatter(result['acorr_nb_vec_1'][:, :, 0].flatten(), 
    #              result['acorr_nb_vec_1'][:, :, 1].flatten(),
    #              s=5, alpha=0.1)
    a[0].set_xlabel(x_label)
    a[0].set_ylabel(y_label)
    a[0].grid()
    a[0].set_aspect('equal')
    a[1].imshow(cross_im, extent=[result['x_range'][0], result['x_range'][-1], 
                                result['y_range'][0], result['y_range'][-1]], origin='lower')
    # a[1].scatter(result['xcorr_vec_1i2'][:, :, 0].flatten(), 
    #              result['xcorr_vec_1i2'][:, :, 1].flatten(),
    #              s=5, alpha=0.1)
    a[1].set_xlabel(x_label)
    a[1].set_ylabel(y_label)
    a[1].grid()
    a[1].set_aspect('equal')
    f.tight_layout()
    return f, a

def vis_m_fold_orientation_order_map(pt_oo_info, m_fold_syn, proj_uvw, 
                                     arrow_len=5000, x_label='u (nm)', y_label='v (nm)'): 
    oo_idx = np.flatnonzero(pt_oo_info['m_list'] == m_fold_syn).item()
    opt_oo = pt_oo_info['oo'][:, oo_idx]
    c_val = pt_oo_info['m_oo_nn']
    valid_Q = np.isfinite(c_val) & np.isfinite(opt_oo)
    c_val = c_val[valid_Q]
    proj_uvw = proj_uvw[valid_Q]
    opt_oo = opt_oo[valid_Q]
    # opt_oo_arg = np.angle(opt_oo)
    # opt_oo_ep =  np.column_stack([
    #         tmp_cp_proj_uvw[:, 0] + arrow_len * np.cos(opt_oo_arg), 
    #         tmp_cp_proj_uvw[:, 1] + arrow_len * np.sin(opt_oo_arg)
    # ])
    opt_oo_ep =  np.column_stack([
            proj_uvw[:, 0] + arrow_len * np.real(opt_oo), 
            proj_uvw[:, 1] + arrow_len * np.imag(opt_oo)
    ])

    # Visualize orientation of each point 
    f, a = plt.subplots(1, 1, figsize=(10, 6))
    a.scatter(proj_uvw[:, 0], proj_uvw[:, 1], c=c_val, 
              cmap='jet', s=20)
    for tmp_uv, tmp_uv1 in zip(proj_uvw[:, [0, 1]], opt_oo_ep): 
        plt.arrow(tmp_uv[0], tmp_uv[1], 
                  tmp_uv1[0]-tmp_uv[0], tmp_uv1[1]-tmp_uv[1], 
                color='red', alpha=0.5)
    f.colorbar(a.collections[0], ax=a, label='max oo m')
    a.set_aspect('equal')
    a.grid()
    a.set_xlabel(x_label)
    a.set_ylabel(y_label)
    a.set_title(f"Orientation order map (m={m_fold_syn})")
    f.tight_layout()
    return f, a


#region Translation 
def _compute_knn_disp_vec(data_pts, num_nb): 
    pt_kdt = sps.cKDTree(data_pts)
    nb_dist, nb_idx = pt_kdt.query(data_pts, k=num_nb+1)
    # remove self
    nb_dist = nb_dist[:, 1:]
    nb_idx = nb_idx[:, 1:]
    nb_vec = np.zeros((data_pts.shape[0], num_nb, data_pts.shape[1]))
    for i in range(data_pts.shape[0]):
        nb_vec[i] = data_pts[nb_idx[i]] - data_pts[i]
    
    return nb_vec, nb_dist, pt_kdt   

def compute_lattice_unit_vector_2d(data_pts, num_nb, max_dist, return_kdt_Q=False): 
    # only consider the first two components 
    nb_vec, nb_dist, pt_kdt = _compute_knn_disp_vec(data_pts, num_nb)
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
    vecs = np.column_stack((vec_1, vec_2))
    if return_kdt_Q: 
        return vecs, pt_kdt
    else: 
        return vecs

def compute_reciprocal_vector_2d(data_pts, num_nb, max_dist, return_kdt_Q=False):
    """ Compute the reciprocal lattice vectors given the point cloud data of a unit cell.
    Input: 
        data_pts: (N, d) array of point coordinates.
        num_nb: number of nearest neighbors to consider for computing the lattice vectors.
        max_dist: maximum distance to consider for neighbors when computing the lattice vectors.
    
    """
    vecs, pt_kdt = compute_lattice_unit_vector_2d(data_pts, num_nb, max_dist, return_kdt_Q=True)
    G_mat = np.linalg.inv(vecs) * 2 * np.pi
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
        tmp_bin_idx = py_util.bin_data_to_idx_list(tmp_nb_bin_idx, return_type='dict')
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

#region Voronoi
class VoronoiQuery:
    def __init__(self, xy):
        self.xy = np.asarray(xy, dtype=float)
        self.vor = Voronoi(self.xy)
        self.rp = self.vor.ridge_points
        self.rv = np.asarray(self.vor.ridge_vertices)

    def neighbors(self, i, remove_infinite_Q=True):
        mask = np.any(self.rp == i, axis=1)
        rp_i, rv_i = self.rp[mask], self.rv[mask]
        neighbors = rp_i.sum(axis=1) - i
        lengths = np.full(len(rv_i), np.inf)
        finite = np.all(rv_i >= 0, axis=1)
        v = self.vor.vertices
        lengths[finite] = np.linalg.norm(v[rv_i[finite, 0]] - v[rv_i[finite, 1]], axis=1)
        if remove_infinite_Q:
            neighbors = neighbors[finite]
            lengths = lengths[finite]
        return neighbors, lengths

#region Transformation
def hexagonal_pq_to_xy(p, q, e_p=None, e_q=None, x_0=0, y_0=0):
    """
    Convert lattice (p, q) coordinates to Cartesian (x, y).

    By default:
    e_p = (np.sqrt(3)/2, 1/2)
    e_q = (-np.sqrt(3)/2, 1/2)
    which are unit vectors 120 deg apart.
    """
    p = np.asarray(p).ravel()
    q = np.asarray(q).ravel()
    if p.ndim != 1 or q.ndim != 1 or p.size != q.size:
        raise ValueError("p and q must be 1D arrays of the same length.")

    if e_p is None:
        # e_p = np.array([1.0, 0.0], dtype=float)
        e_p = np.asarray([np.sqrt(3)/2, 1/2])
    else:
        e_p = np.asarray(e_p, dtype=float).ravel()
        assert e_p.size == 2, "e_p must be length-2."

    if e_q is None:
        e_q = np.asarray([-np.sqrt(3)/2, 1/2])
        # e_q = np.array([-0.5, np.sqrt(3.0) / 2.0], dtype=float)
    else:
        e_q = np.asarray(e_q, dtype=float).ravel()
        assert e_q.size == 2, "e_q must be length-2."

    x = p * e_p[0] + q * e_q[0] + x_0
    y = p * e_p[1] + q * e_q[1] + y_0
    return x, y