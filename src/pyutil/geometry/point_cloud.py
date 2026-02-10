from typing import Dict, Optional
from functools import cached_property
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

class PointCloud3DSurfaceFit():
    def __init__(self, points_xyz: np.ndarray, 
                 R=None, t=None):
        """
        Fit a 2nd-order surface to a 3D point cloud, assuming the global vertical axis is z.

        Args:
            points_xyz: (N, 3) array of [x,y,z] points. 
        """
        self.points_xyz = np.asarray(points_xyz, dtype=float)
        if self.points_xyz.ndim != 2 or self.points_xyz.shape[1] != 3:
            raise ValueError("points_xyz must be shape (N, 3)")
        if R is None and t is None: 
            self.centroid = self.points_xyz.mean(axis=0)
            X = self.points_xyz - self.centroid
            self.U, self.S, self.V = np.linalg.svd(X, full_matrices=False)
            # PCA axes: columns of V = [u,v,w] in world coords
            self.R = self.V.T  # columns: x-axis, y-axis, z-axis
        elif R is not None and t is not None: 
            assert t.shape == (3, ) and R.shape == (3, 3), 'Inconnect input size'
            self.centroid = t
            self.R = R
            X = self.points_xyz - self.centroid
        else: 
            raise f"Both R and t should be provided"
        
        # Transform to local coords: [u,v,w] = (P - centroid) @ R
        self.points_uvw = X @ self.R
        u, v, w = self.points_uvw[:, 0], self.points_uvw[:, 1], self.points_uvw[:, 2]

        # Fit quadratic: w = a + b u + c v + d u^2 + e u v + f v^2
        A = PointCloud3DSurfaceFit._uv_to_polynomial_array(u, v)
        self.coeffs, *_ = np.linalg.lstsq(A, w, rcond=None) # the returned residual is the sum of squared residuals

        self.w_fit = self.f(u, v, self.coeffs)
        self.residuals_w = w - self.w_fit

    @property
    def convexity(self):
        coeff = self.coeffs[3:]
        if np.all(coeff > 0): 
            return 1
        elif np.all(coeff < 0):
            return -1
        else: 
            return 0

    def xyz_to_uvw(self, xyz): 
        xyz = np.asarray(xyz, dtype=float)
        if xyz.ndim == 1: 
            xyz = xyz.reshape((1, -1))
        return (xyz - self.centroid) @ self.R
    
    def uvw_to_xyz(self, uvw):
        uvw = np.asarray(uvw, dtype=float)
        if uvw.ndim == 1: 
            uvw = uvw.reshape((1, -1))
        return uvw @ self.R.T + self.centroid

    def compute_xyz_residual(self, xyz): 
        uvw = self.xyz_to_uvw(xyz)
        res = self._uvw_to_residual(uvw)
        return res

    def _uvw_to_residual(self, uvw):
        w_hat = self.f(uvw[:, 0], uvw[:, 1], self.coeffs)
        res = uvw[:, 2] - w_hat
        return res

    @staticmethod
    def _uv_to_polynomial_array(u, v):
        assert np.asarray(u).shape == np.asarray(v).shape, "u and v must have the same shape"
        u = u.flatten()
        v = v.flatten()
        # w = a + b u + c v + d u^2 + e u v + f v^2
        return np.column_stack([np.ones_like(u), u, v, u**2, u*v, v**2])

    @staticmethod
    def f(u, v, coeffs):
        u_shape = np.asarray(u).shape
        assert np.asarray(v).shape == u_shape, "u and v must have the same shape"
        u = u.flatten()
        v = v.flatten()
        poly_array = PointCloud3DSurfaceFit._uv_to_polynomial_array(u, v)
        w = poly_array @ coeffs
        w = w.reshape(u_shape)
        return w
    
    @staticmethod
    def df_du(u, v, coeffs):
        u = np.asarray(u).flatten()
        v = np.asarray(v).flatten()
        # dw/du = b + 2 d u + e v
        b = coeffs[1]
        d = coeffs[3]
        e = coeffs[4]
        dw_du = b + 2 * d * u + e * v
        dw_du = dw_du.reshape(np.asarray(u).shape)
        return dw_du
    
    @staticmethod
    def df_dv(u, v, coeffs):
        u = np.asarray(u).flatten()
        v = np.asarray(v).flatten()
        # dw/dv = c + e u + 2 f v
        c = coeffs[2]
        e = coeffs[4]
        f = coeffs[5]
        dw_dv = c + e * u + 2 * f * v
        dw_dv = dw_dv.reshape(np.asarray(u).shape)
        return dw_dv

    @staticmethod
    def uv_to_polynomial_array(u, v):
        """Convert (u,v) to polynomial feature array [1, u, v, u^2, uv, v^2]
        Input:
            u: (N,) or (N,1) array of u-coordinates
            v: (N,) or (N,1) array of v-coordinates
        Returns:
            (N, 6) array of polynomial features
        """
        u = np.asarray(u)
        v = np.asarray(v)
        return np.column_stack([np.ones_like(u), u, v, u**2, u*v, v**2])
    
    def project_xyz_points_to_surface(self, xyz_points, 
                                  max_iter=100, tol=1e-6, 
                                  return_dist_Q=False):
        """
        Project points in world coordinates onto the fitted surface.

        Args:
            xyz_points: (M, 3) array of points in world coordinates.
            max_iter: Maximum number of iterations for projection.
            tol: Tolerance for convergence.

        Returns:
            uvw_proj: (M, 3) array of projected points in world coordinates.
        """
        uvw = self.xyz_to_uvw(xyz_points)
        result = self.project_uvw_points_to_surface(
            uvw, max_iter=max_iter, tol=tol, return_dist_Q=return_dist_Q)
        
        if return_dist_Q:
            uvw_proj, dists = result
        else:
            uvw_proj = result
            dists = None

        xyz_proj = self.uvw_to_xyz(uvw_proj)
        if return_dist_Q:
            return xyz_proj, dists
        else:
            return xyz_proj
    
    def project_uvw_points_to_surface(self, uvw_points, 
                                  max_iter=100, tol=1e-6, 
                                  return_dist_Q=False):
        """
        Project points in local uvw coordinates onto the fitted surface.

        Args:
            uvw_points: (M, 3) array of points in local uvw coordinates.
            max_iter: Maximum number of iterations for projection.
            tol: Tolerance for convergence. 
        Returns:
            uvw_proj: (M, 3) array of projected points in local uvw coordinates.
        """ 
        u1, v1, w1 = PointCloud3DSurfaceFit._project_uvw_points_to_surface(
            uvw_points[:, 0], uvw_points[:, 1], uvw_points[:, 2], self.coeffs,
            max_iter=max_iter, tol=tol)
        uvw_proj = np.column_stack([u1, v1, w1])
        dists = np.linalg.norm(uvw_points - uvw_proj, axis=1) if return_dist_Q else None
        if return_dist_Q:
            return uvw_proj, dists
        else:
            return uvw_proj

    @staticmethod
    def _project_uvw_points_to_surface(u0, v0, w0, coeffs, 
                                  max_iter=100, tol=1e-6):
        """
        Project points (u0, v0, w0) onto the fitted surface defined by coeffs
        using iterative Newton-Raphson method.

        Args:
            u0, v0, w0: Initial coordinates of points to project.
            coeffs: Coefficients of the fitted surface.
            max_iter: Maximum number of iterations.
            tol: Tolerance for convergence.

        Returns:
            u_proj, v_proj, w_proj: Projected coordinates on the surface.
        """
        u0 = np.asarray(u0, dtype=float).flatten()
        v0 = np.asarray(v0, dtype=float).flatten()
        w0 = np.asarray(w0, dtype=float).flatten()

        u = u0.copy()
        v = v0.copy()
        
        fuu = 2 * coeffs[3]
        fuv = coeffs[4]
        fvv = 2 * coeffs[5]
        eps = 1e-12
        for _ in range(max_iter):
            w_fit = PointCloud3DSurfaceFit.f(u, v, coeffs)
            fu = PointCloud3DSurfaceFit.df_du(u, v, coeffs)
            fv = PointCloud3DSurfaceFit.df_dv(u, v, coeffs)
            # Residuals
            r = w_fit - w0
            
            F1 = (u - u0) + r * fu
            F2 = (v - v0) + r * fv

            # Jacobian
            A = 1.0 + fu ** 2 + 2 * r * fuu
            B = fu * fv + r * fuv
            D = 1.0 + fv ** 2 + 2 * r * fvv
            det = A * D - B * B
            det = np.where(np.abs(det) < eps, np.sign(det) * eps, det)

            # Update step
            du = (D * F1 - B * F2) / det
            dv = (A * F2 - B * F1) / det

            u -= du
            v -= dv

            if np.max(np.abs(du) + np.abs(dv)) < tol:
                break

        w_proj = PointCloud3DSurfaceFit.f(u, v, coeffs)
        return u, v, w_proj

    @property
    def point_uvw(self):
        return self.xyz_to_uvw(self.points_xyz)
    
    @property
    def point_nearest_uvw(self): 
        uvw_proj = self.project_uvw_points_to_surface(self.point_uvw)
        return uvw_proj
    
    @cached_property
    def point_xyz_bbox(self):
        P = np.asarray(self.points_xyz, dtype=float)
        min_xyz = P.min(axis=0)
        max_xyz = P.max(axis=0)
        return min_xyz, max_xyz
    
    @cached_property
    def point_uvw_bbox(self):
        L = self.point_uvw
        min_uvw = L.min(axis=0)
        max_uvw = L.max(axis=0)
        return min_uvw, max_uvw

    def vis_points_w_fitted_surface(self, grid_res=80, pad=0.05, 
                                    vis_frame='world', fig=None, ax=None, 
                                    vis_pts_Q=True, label=None):
        
        P = np.asarray(self.points_xyz, dtype=float)
        # Local coords
        L = self.point_uvw
        u, v, w = L[:, 0], L[:, 1], L[:, 2]

        # Grid in local (u,v)
        umin, umax = u.min(), u.max()
        vmin, vmax = v.min(), v.max()
        du = (umax - umin) * pad
        dv = (vmax - vmin) * pad

        ug = np.linspace(umin - du, umax + du, grid_res)
        vg = np.linspace(vmin - dv, vmax + dv, grid_res)
        U, V = np.meshgrid(ug, vg)
        W = PointCloud3DSurfaceFit.f(U, V, self.coeffs)

        if fig is None or ax is None:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection="3d")
        
        if vis_frame == 'local':
            if vis_pts_Q:
                ax.scatter(L[:, 0], L[:, 1], L[:, 2], s=1, alpha=0.9)
            ax.plot_surface(U, V, W, alpha=0.25, linewidth=0, label=label)
        else:
            if vis_pts_Q:
                ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=1, alpha=0.9)
            world_grid = self.uvw_to_xyz(np.stack([U, V, W], axis=-1))                
            ax.plot_surface(world_grid[..., 0], world_grid[..., 1], 
                            world_grid[..., 2], alpha=0.25, linewidth=0, label=label)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Quadratic surface fit in PCA-aligned {vis_frame} frame")

        try:
            if vis_frame == 'local':
                ax.set_box_aspect([np.ptp(L[:, 0]), np.ptp(L[:, 1]), np.ptp(L[:, 2])])
            else:
                ax.set_box_aspect([np.ptp(P[:, 0]), np.ptp(P[:, 1]), np.ptp(P[:, 2])])
        except Exception:
            pass

        plt.tight_layout()
        return fig, ax
    
    @staticmethod
    def test_points_between_two_surfaces(surf1, surf2, 
                                        points_xyz, 
                                        select_horizontal_Q=False):
        """Select points that lie between two surfaces.

        Args:
            surf1: First surface fit object.
            surf2: Second surface fit object.
            points_xyz: (N, 3) array of points in XYZ coordinates.

        Returns:
            A boolean array indicating which points lie between the two surfaces.
        """
        # Convert points to local UVW coordinates for both surfaces
        uvw1 = surf1.xyz_to_uvw(points_xyz)
        if np.all(surf1.R == surf2.R) and np.all(surf1.centroid == surf2.centroid):
            uvw2 = uvw1
        else:
            print(f"Warning: The two surfaces have different orientations or centroids. Converting points to local UVW coordinates separately for each surface.")
            uvw2 = surf2.xyz_to_uvw(points_xyz)

        # Compute residuals for both surfaces
        res1 = surf1._uvw_to_residual(uvw1)
        res2 = surf2._uvw_to_residual(uvw2)

        # Points are between surfaces if residuals have opposite signs
        between_Q = (res1 * res2) < 0
        if select_horizontal_Q: 
            # Also check if points are within the bounding box of the surfaces
            uvw1_min, uvw1_max = surf1.point_uvw_bbox
            uvw2_min, uvw2_max = surf2.point_uvw_bbox
            uvw_min = np.minimum(uvw1_min, uvw2_min)
            uvw_max = np.maximum(uvw1_max, uvw2_max)
            within_bbox_Q = np.all((uvw1 >= uvw_min) & (uvw1 <= uvw_max), axis=1)
            between_Q = between_Q & within_bbox_Q      

        return between_Q

# Assume dict key, val are in the same order (Python 3.7+)
class PointCloudGroup: 
    def __init__(self, pc_dict:dict, group_info: Optional[Dict] = None):
        """ Initialize the point cloud group with a dictionary of point clouds.

        Args:
            pc_dict: A dictionary where keys are point cloud IDs and 
            values are (N, d) arrays of points.
            group_info: Optional additional information about the groups
        """
        self.id = np.asarray(list(pc_dict.keys()))
        self.num_pts = np.zeros(self.id.size, dtype=np.uint32)
        node_pos = []
        for i, k in enumerate(self.id):
            v = pc_dict[k]
            self.num_pts[i] = v.shape[0]
            node_pos.append(v)
        
        self.c_idx_1 = np.cumsum(self.num_pts).astype(np.uint32)
        self.c_idx_0 = np.concatenate(([int(0)], self.c_idx_1[:-1]))
        node_pos = np.concatenate(node_pos, axis=0)
        self.dim = node_pos.shape[1]
        self.kdt = KDTree(node_pos)

        if group_info is not None: 
            # all pc_dict key are in group_info
            assert isinstance(group_info, (dict, type(None))), f'Unexpected group_info type: {type(group_info)}'
            # Ensure group_info keys cover all pc_dict keys
            missing = set(pc_dict.keys()) - set(group_info.keys())
            if missing:
                raise ValueError(f"group_info missing keys: {missing}")
            # Align group_info to self.id order
            self.group_info = {k: group_info[k] for k in self.id}
        else:
            self.group_info = None

    @property
    def xyz(self):
        # skeleton xyz in nm
        return self.kdt.data.reshape((-1, self.dim))
    
    def get_knn_nodes(self, xyz_nm, k, distance_upper_bound=np.inf):
        """ Get k-nearest neighbors for a set of points in 3D space.

        Args:
            xyz_nm: (N, self.dim) array of points in nanometers
            k: int, number of neighbors to retrieve
            distance_upper_bound: float, maximum distance to consider (in nanometers)

        Returns:
            k_dist: (N, k) array of distances to the k-nearest neighbors
            k_idx: (N, k) array of indices of the k-nearest neighbors
        """
        xyz_nm = np.atleast_2d(np.asarray(xyz_nm))
        assert xyz_nm.shape[1] == self.dim, f'Unexpected input coordinate dimension: {xyz_nm.shape}'

        k_dist, k_idx = self.kdt.query(xyz_nm, k=k, 
                                       distance_upper_bound=distance_upper_bound)
        return k_dist, k_idx

    def get_all_nodes_within_distance(self, xyz_nm, distance_upper_bound, return_dist_Q=False):
        """ Get all nodes within a certain distance for a set of points in 3D space.

        Args:
            xyz_nm: (N, self.dim) array of points in nanometers
            distance_upper_bound: float, maximum distance to consider (in nanometers)

        Returns:
            all_dist: (N, M) array of distances to all neighbors within the distance_upper_bound
            all_idx: (N, M) array of indices of all neighbors within the distance_upper_bound
        """
        xyz_nm = np.atleast_2d(np.asarray(xyz_nm))
        assert xyz_nm.shape[1] == self.dim, f'Unexpected input coordinate dimension: {xyz_nm.shape}'

        all_idx = self.kdt.query_ball_point(xyz_nm, r=distance_upper_bound, 
                                                      workers=-1, return_sorted=False, 
                                                      return_length=False)
        all_idx = [np.asarray(idx) for idx in all_idx]
        if return_dist_Q:
            all_dist = [np.linalg.norm(xyz_nm[i] - self.xyz[all_idx[i]], axis=1) for i in range(xyz_nm.shape[0])]
            return all_dist, all_idx
        else: 
            return all_idx
    
    def _idx_to_id_idx(self, idx):
        """ Convert point index in the array to group index.
        Inputs: 
            idx: int or array-like of point indices
        Outputs: 
            key_idx: int or array of group indices corresponding to the input point indices
        """
        key_idx = np.searchsorted(self.c_idx_1, idx, 'right')
        return key_idx
    
    def _idx_to_id(self, idx):
        """ Convert point index in the array to group id.
        Inputs: 
            idx: int or array-like of point indices
        Outputs: 
            id: int or array of group ids corresponding to the input point indices
        """
        key_idx = self._idx_to_id_idx(idx)
        return self.id[key_idx]

    def get_neighbor_group_info(self, nb_dist, nb_idx):
        """ Get information about neighboring groups based on distance and index
        Inputs: 
            nb_dist: (N, k) array of distances to neighboring k points 
            nb_idx: (N, k) array of indices of neighboring k points
        Outputs: 
            A dictionary containing information about the neighboring groups, including:
                - 'id': (N, ) id of the nearby groups
                - 'num_pts': (N, ) number of query points falling in each nearby group
                - 'dist': (N, ) distances to the nearest processes of each nearby groups
                - 'xyz_nm': (N, k, self.dim) array of xyz coordinates of the nearest points in each group to the query points
        """
        # Minimal distance between pre_type neurons to the target neuron 
        nb_dist = nb_dist.flatten()
        nb_idx = nb_idx.flatten()
        # Merge the data from all query points - input spatial information lost
        is_valid_dist_Q = np.isfinite(nb_dist.flatten())
        if not np.all(is_valid_dist_Q):
            nb_dist = nb_dist[is_valid_dist_Q]
            nb_idx = nb_idx[is_valid_dist_Q]
        # Use the binning information to find the corresponding neuron in the skeleton point cloud
        nn_pt_rid_idx = self._idx_to_id_idx(nb_idx)
        bin_idx, nn_pt_idx_u = pyutil.util.bin_data_to_idx_list(nn_pt_rid_idx.flatten())
        # Get the nearest points of each group 
        nn_pt_dist = np.zeros(nn_pt_idx_u.size, dtype=np.float32)
        nn_pt_idx = np.zeros(nn_pt_idx_u.size, dtype=np.uint)
        nn_num_pts = np.zeros(nn_pt_idx_u.size, dtype=np.uint)
        for i, tmp_idx in enumerate(bin_idx): 
            # Each bin is the distances to all the nearby points of one group
            tmp_dist = nb_dist.flat[tmp_idx]
            nn_num_pts[i] = tmp_idx.size

            tmp_min_idx = np.argmin(tmp_dist)
            nn_pt_dist[i] = tmp_dist[tmp_min_idx]
            nn_pt_idx[i] = nb_idx.flat[tmp_idx[tmp_min_idx]]

        nn_pt_xyz_nm = self.xyz[nn_pt_idx]
        nn_pt_id = self.id.flat[nn_pt_idx_u]
        result = {'id': nn_pt_id, 
                  'num_pts': nn_num_pts, # number of query points falling in the group
                  'dist': nn_pt_dist, 
                  'xyz_nm': nn_pt_xyz_nm # nearest points in each group to the query points
                  } 
        return result
    

def grid_centroids(points: np.ndarray,
                   grid_size: tuple[int, int, int],
                   origin: np.ndarray | None = None,
                   extent: np.ndarray | None = None,
                   return_counts: bool = False):
    """
    Compute centroids of 3D points inside each grid cell.

    Parameters
    ----------
    points : (N, 3) array_like
        3D point coordinates (float32/float64).
    grid_size : (nx, ny, nz)
        Number of cells along x, y, z.
    origin : (3,) array_like, optional
        Grid origin (min corner). Default: points.min(axis=0).
    extent : (3,) array_like, optional
        Physical size of the grid along x,y,z. Default: points.max - origin.
        Cell size = extent / grid_size.
    return_counts : bool
        If True, also return counts per occupied cell.

    Returns
    -------
    centroids : (M, 3) ndarray
        Centroids for occupied cells.
    ijk : (M, 3) ndarray (int64)
        Integer cell indices (ix, iy, iz) corresponding to each centroid.
    counts : (M,) ndarray (int64), optional
        Number of points in each occupied cell (if return_counts=True).
    """
    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")

    nx, ny, nz = map(int, grid_size)
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError("grid_size must be positive integers (nx, ny, nz)")

    if origin is None:
        origin = pts.min(axis=0)
    else:
        origin = np.asarray(origin, dtype=pts.dtype)

    if extent is None:
        # Use bounding box of points; add tiny epsilon so max lies inside last bin
        max_corner = pts.max(axis=0)
        extent = max_corner - origin
        extent = extent + np.finfo(pts.dtype).eps
    else:
        extent = np.asarray(extent, dtype=pts.dtype)

    cell = extent / np.array([nx, ny, nz], dtype=pts.dtype)

    # Convert to cell indices
    ijk = np.floor((pts - origin) / cell).astype(np.int64)

    # Keep only points inside grid
    mask = (
        (ijk[:, 0] >= 0) & (ijk[:, 0] < nx) &
        (ijk[:, 1] >= 0) & (ijk[:, 1] < ny) &
        (ijk[:, 2] >= 0) & (ijk[:, 2] < nz)
    )
    ijk = ijk[mask]
    pts_in = pts[mask]

    # Flatten 3D indices to 1D bin ids
    # id = ix*(ny*nz) + iy*nz + iz
    stride_yz = ny * nz
    ids = ijk[:, 0] * stride_yz + ijk[:, 1] * nz + ijk[:, 2]
    n_bins = nx * ny * nz

    # Accumulate counts and coordinate sums
    counts = np.bincount(ids, minlength=n_bins).astype(np.int64)
    sums_x = np.bincount(ids, weights=pts_in[:, 0], minlength=n_bins)
    sums_y = np.bincount(ids, weights=pts_in[:, 1], minlength=n_bins)
    sums_z = np.bincount(ids, weights=pts_in[:, 2], minlength=n_bins)

    occupied = counts > 0
    centroids = np.stack([
        sums_x[occupied] / counts[occupied],
        sums_y[occupied] / counts[occupied],
        sums_z[occupied] / counts[occupied],
    ], axis=1)

    # Recover (ix, iy, iz) for occupied bins
    occ_ids = np.nonzero(occupied)[0]
    ix = occ_ids // stride_yz
    rem = occ_ids - ix * stride_yz
    iy = rem // nz
    iz = rem - iy * nz
    occ_ijk = np.stack([ix, iy, iz], axis=1)

    if return_counts:
        return centroids, occ_ijk, counts[occupied]
    return centroids, occ_ijk
