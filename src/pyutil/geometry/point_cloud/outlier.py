import numpy as np

from ... import stat


def select_points_near_pc1(pts, ipr=1.5, max_dist_th=None):
    pts = np.asarray(pts)
    assert pts.shape[1] == 3, 'Expect 3D point cloud'
    pts_stat = stat.compute_point_cloud_basic_statistics(pts)
    pts_uvw = (pts - pts_stat['mean']) @ pts_stat['eig_v']
    off_axis_n = np.linalg.norm(pts_uvw[:, 1:], axis=1)
    off_axis_th = stat.compute_percentile_outlier_threshold(off_axis_n,
                                                            ipr=ipr)
    off_axis_th = np.minimum(off_axis_th[1], max_dist_th) if max_dist_th is not None else off_axis_th[1]
    off_axis_outlier_Q = off_axis_n > off_axis_th

    pts_stat['pts_uvw'] = pts_uvw
    pts_stat['off_axis_len'] = off_axis_n
    pts_stat['off_axis_th'] = off_axis_th
    pts_stat['off_axis_outlier_Q'] = off_axis_outlier_Q
    pts_stat['num_outliers'] = off_axis_outlier_Q.sum()
    pts_stat['inlier_pts'] = pts[~off_axis_outlier_Q]

    return pts_stat

def select_points_near_pc1_iterative(pts, ipr=1.5, max_iter=10, max_dist_th=None):
    num_raw_pts = pts.shape[0]
    inlier_idx = np.arange(num_raw_pts)
    num_outlier = 1
    iter = 0
    while num_outlier > 0 and iter < max_iter:
        pts_stat = select_points_near_pc1(pts[inlier_idx], ipr=ipr, max_dist_th=max_dist_th)
        num_outlier = pts_stat['num_outliers']
        inlier_idx = inlier_idx[~pts_stat['off_axis_outlier_Q']]
        iter += 1
    pts_stat['num_iterations'] = iter
    pts_stat['num_raw_pts'] = num_raw_pts
    pts_stat['inlier_idx'] = inlier_idx
    pts_stat['frac_inliers'] = pts_stat['inlier_pts'].shape[0] / num_raw_pts
    pts_stat['num_outliers'] = num_raw_pts - pts_stat['inlier_pts'].shape[0]
    pts_stat.pop('off_axis_outlier_Q', None)
    return pts_stat
