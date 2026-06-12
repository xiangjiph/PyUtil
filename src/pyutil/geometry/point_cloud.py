from typing import Any, Dict, Optional
from functools import cached_property
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

from .. import stat
from .. import util

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
    def R2(self): 
        ss_res = np.sum(self.residuals_w ** 2)
        ss_tot = np.sum((self.points_uvw[:, 2] - self.points_uvw[:, 2].mean()) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    
    def residual_ipr_range(self, ipr=1.5):
        range = stat.compute_percentile_outlier_threshold(self.residuals_w, ipr=ipr)
        return range

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
    
    def remove_xyz_by_residual_threshold(self, xyz, inliner_range, return_res_Q=False):
        res = self.compute_xyz_residual(xyz)
        inlier_Q = (res >= inliner_range[0]) & (res <= inliner_range[1])
        if return_res_Q:
            return xyz[inlier_Q], res[inlier_Q]
        else:
            return xyz[inlier_Q]

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
    def normal_vector(u, v, coeffs): 
        dw_du = PointCloud3DSurfaceFit.df_du(u, v, coeffs)
        dw_dv = PointCloud3DSurfaceFit.df_dv(u, v, coeffs)
        n = np.stack([-dw_du, -dw_dv, np.ones_like(dw_du)], axis=-1)
        n_norm = np.linalg.norm(n, axis=-1, keepdims=True)
        n_unit = n / n_norm
        return n_unit
    
    @staticmethod
    def tangent_vectors(u, v, coeffs):
        dw_du = PointCloud3DSurfaceFit.df_du(u, v, coeffs)
        dw_dv = PointCloud3DSurfaceFit.df_dv(u, v, coeffs)
        t_u = np.stack([np.ones_like(dw_du), np.zeros_like(dw_du), dw_du], axis=-1)
        t_v = np.stack([np.zeros_like(dw_dv), np.ones_like(dw_dv), dw_dv], axis=-1)
        t_u_norm = np.linalg.norm(t_u, axis=-1, keepdims=True)
        e1 = t_u / t_u_norm
        e2 = t_v - np.sum(t_v * e1, axis=-1, keepdims=True) * e1
        e2 = e2 / np.linalg.norm(e2, axis=-1, keepdims=True)
        return e1, e2
    
    @staticmethod
    def uvw_to_tangent_plane(uvw, uvw0, coeffs):
        """Project points in local uvw coordinates to the tangent plane at uvw0."""
        t_u, t_v = PointCloud3DSurfaceFit.tangent_vectors(uvw0[0], uvw0[1], coeffs)
        delta_uvw = uvw - uvw0
        proj_u = np.sum(delta_uvw * t_u, axis=-1)
        proj_v = np.sum(delta_uvw * t_v, axis=-1)
        proj_uv = np.column_stack([proj_u, proj_v])
        return proj_uv

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
    def points_uvw(self):
        return self.xyz_to_uvw(self.points_xyz)
    
    @property
    def point_nearest_uvw(self): 
        uvw_proj = self.project_uvw_points_to_surface(self.points_uvw)
        return uvw_proj
    
    @cached_property
    def point_xyz_bbox(self):
        P = np.asarray(self.points_xyz, dtype=float)
        min_xyz = P.min(axis=0)
        max_xyz = P.max(axis=0)
        return min_xyz, max_xyz
    
    @cached_property
    def points_uvw_bbox(self):
        L = self.points_uvw
        min_uvw = L.min(axis=0)
        max_uvw = L.max(axis=0)
        return min_uvw, max_uvw

    def vis_points_w_fitted_surface(self, grid_res=80, pad=0.05, 
                                    vis_frame='world', fig=None, ax=None, 
                                    vis_pts_Q=True, label=None):
        
        P = np.asarray(self.points_xyz, dtype=float)
        # Local coords
        L = self.points_uvw
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
                # c = self.residuals_w
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
        ax.set_title(f"Quadratic surface fit in {vis_frame} frame")

        # try:
        #     if vis_frame == 'local':
        #         ax.set_box_aspect([np.ptp(L[:, 0]), np.ptp(L[:, 1]), np.ptp(L[:, 2])])
        #     else:
        #         ax.set_box_aspect([np.ptp(P[:, 0]), np.ptp(P[:, 1]), np.ptp(P[:, 2])])
        # except Exception:
        #     pass

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
            uvw1_min, uvw1_max = surf1.points_uvw_bbox
            uvw2_min, uvw2_max = surf2.points_uvw_bbox
            uvw_min = np.minimum(uvw1_min, uvw2_min)
            uvw_max = np.maximum(uvw1_max, uvw2_max)
            within_bbox_Q = np.all((uvw1 >= uvw_min) & (uvw1 <= uvw_max), axis=1)
            between_Q = between_Q & within_bbox_Q      

        return between_Q

    def vis_residual(self, fig=None, ax=None):
        if fig is None or ax is None:
            f, a = plt.subplots(1, 1, figsize=(6, 4))
        abs_max_res = np.max(np.abs(self.residuals_w))
        a.scatter(self.points_uvw[:, 0], self.points_uvw[:, 1], c=self.residuals_w, s=5, alpha=0.9, 
                  cmap='coolwarm', vmin=-abs_max_res, vmax=abs_max_res)
        a.set_xlabel("u")
        a.set_ylabel("v")
        f.colorbar(a.collections[0], ax=a, label="Residual w")
        f.tight_layout()
        return f, a

class PolySurface3D:
    """Parameterized polynomial surface in local uvw coordinates.

    Surface model:
        w = sum_{i+j<=k} c_{ij} * u^i * v^j

    This class is initialized directly from polynomial parameters and an
    optional world-to-local frame (R, t), so no point-cloud refit is needed.
    """

    def __init__(self, coeffs, k: Optional[int] = None,
                 exponents: Optional[tuple] = None, 
                 R: Optional[np.ndarray] = None,
                 t: Optional[np.ndarray] = None, 
                 bbox_xyz: Optional[tuple] = None, 
                 bbox_uvw: Optional[tuple] = None):
        coeffs = np.asarray(coeffs, dtype=float).flatten()
        if coeffs.ndim != 1 or coeffs.size == 0:
            raise ValueError("coeffs must be a non-empty 1D array")

        if exponents is None:
            if k is None:
                raise ValueError("Either exponents or k must be provided")
            exponents = PolySurface3D._generate_exponents(k)

        exponents = tuple(tuple(int(v) for v in ij) for ij in exponents)
        if len(exponents) != coeffs.size:
            raise ValueError("coeffs and exponents must have the same length")
        if any((len(ij) != 2 or ij[0] < 0 or ij[1] < 0) for ij in exponents):
            raise ValueError("exponents must be iterable of (i, j) with i,j >= 0")

        if R is None:
            R = np.eye(3, dtype=float)
        if t is None:
            t = np.zeros(3, dtype=float)

        R = np.asarray(R, dtype=float)
        t = np.asarray(t, dtype=float)
        if R.shape != (3, 3) or t.shape != (3,):
            raise ValueError("Expect R.shape == (3, 3) and t.shape == (3,)")

        self.coeffs = coeffs
        self.exponents = exponents
        self.R = R
        self.centroid = t
        self.k = max(i + j for i, j in self.exponents)
        self.bbox_xyz = bbox_xyz
        self.bbox_uvw = bbox_uvw

    @staticmethod
    def _generate_exponents(k: int):
        if int(k) < 1:
            raise ValueError("k must be >= 1")
        exponents = []
        for total_deg in range(int(k) + 1):
            for i in range(total_deg, -1, -1):
                j = total_deg - i
                exponents.append((i, j))
        return tuple(exponents)

    @staticmethod
    def _uv_to_polynomial_array(u, v, exponents):
        u_arr = np.asarray(u)
        v_arr = np.asarray(v)
        if u_arr.shape != v_arr.shape:
            raise ValueError("u and v must have the same shape")

        u_flat = u_arr.flatten()
        v_flat = v_arr.flatten()
        ij = np.asarray(exponents, dtype=int)
        i_idx = ij[:, 0]
        j_idx = ij[:, 1]

        max_i = int(i_idx.max(initial=0))
        max_j = int(j_idx.max(initial=0))
        u_pow = np.power(u_flat[:, None], np.arange(max_i + 1, dtype=int))
        v_pow = np.power(v_flat[:, None], np.arange(max_j + 1, dtype=int))
        return u_pow[:, i_idx] * v_pow[:, j_idx]

    @staticmethod
    def _f(u, v, coeffs, exponents):
        u_shape = np.asarray(u).shape
        if np.asarray(v).shape != u_shape:
            raise ValueError("u and v must have the same shape")
        poly = PolySurface3D._uv_to_polynomial_array(u, v, exponents)
        w = poly @ np.asarray(coeffs, dtype=float).reshape(-1, 1)
        return w.reshape(u_shape)

    def f(self, u, v, coeffs=None, exponents=None):
        if coeffs is None:
            coeffs = self.coeffs
        if exponents is None:
            exponents = self.exponents
        return PolySurface3D._f(u, v, coeffs, exponents)

    @property
    def convexity(self):
        if self.k != 2:
            raise ValueError("Global convexity is only defined for quadratic surfaces; use local_convexity(u, v) for k > 2")
        return int(np.asarray(self.local_convexity(0.0, 0.0)).item())

    def local_convexity(self, u, v):
        _, _, fuu, fuv, fvv = PolySurface3D._poly_derivatives(
            u, v, self.coeffs, self.exponents)
        det = fuu * fvv - fuv * fuv
        result = np.zeros(np.asarray(fuu).shape, dtype=int)
        result[(fuu > 0) & (det > 0)] = 1
        result[(fuu < 0) & (det > 0)] = -1
        if result.shape == ():
            return int(result)
        return result

    @staticmethod
    def _poly_derivatives(u, v, coeffs, exponents):
        u_arr = np.asarray(u)
        v_arr = np.asarray(v)
        if u_arr.shape != v_arr.shape:
            raise ValueError("u and v must have the same shape")

        shape = u_arr.shape
        u_flat = u_arr.flatten()
        v_flat = v_arr.flatten()
        coeffs = np.asarray(coeffs, dtype=float)
        ij = np.asarray(exponents, dtype=int)
        i_idx = ij[:, 0]
        j_idx = ij[:, 1]

        max_i = int(i_idx.max(initial=0))
        max_j = int(j_idx.max(initial=0))
        u_pow = np.power(u_flat[:, None], np.arange(max_i + 1, dtype=int))
        v_pow = np.power(v_flat[:, None], np.arange(max_j + 1, dtype=int))

        n = u_flat.size
        fu = np.zeros(n, dtype=float)
        fv = np.zeros(n, dtype=float)
        fuu = np.zeros(n, dtype=float)
        fuv = np.zeros(n, dtype=float)
        fvv = np.zeros(n, dtype=float)

        mask = i_idx >= 1
        if np.any(mask):
            basis = u_pow[:, i_idx[mask] - 1] * v_pow[:, j_idx[mask]]
            fu = basis @ (coeffs[mask] * i_idx[mask])

        mask = j_idx >= 1
        if np.any(mask):
            basis = u_pow[:, i_idx[mask]] * v_pow[:, j_idx[mask] - 1]
            fv = basis @ (coeffs[mask] * j_idx[mask])

        mask = i_idx >= 2
        if np.any(mask):
            basis = u_pow[:, i_idx[mask] - 2] * v_pow[:, j_idx[mask]]
            fuu = basis @ (coeffs[mask] * i_idx[mask] * (i_idx[mask] - 1))

        mask = (i_idx >= 1) & (j_idx >= 1)
        if np.any(mask):
            basis = u_pow[:, i_idx[mask] - 1] * v_pow[:, j_idx[mask] - 1]
            fuv = basis @ (coeffs[mask] * i_idx[mask] * j_idx[mask])

        mask = j_idx >= 2
        if np.any(mask):
            basis = u_pow[:, i_idx[mask]] * v_pow[:, j_idx[mask] - 2]
            fvv = basis @ (coeffs[mask] * j_idx[mask] * (j_idx[mask] - 1))

        return (
            fu.reshape(shape),
            fv.reshape(shape),
            fuu.reshape(shape),
            fuv.reshape(shape),
            fvv.reshape(shape),
        )

    def df_du(self, u, v, coeffs=None, exponents=None):
        if coeffs is None:
            coeffs = self.coeffs
        if exponents is None:
            exponents = self.exponents
        fu, _, _, _, _ = PolySurface3D._poly_derivatives(u, v, coeffs, exponents)
        return fu

    def df_dv(self, u, v, coeffs=None, exponents=None):
        if coeffs is None:
            coeffs = self.coeffs
        if exponents is None:
            exponents = self.exponents
        _, fv, _, _, _ = PolySurface3D._poly_derivatives(u, v, coeffs, exponents)
        return fv

    def normal_vector(self, u, v, coeffs=None, exponents=None):
        if coeffs is None:
            coeffs = self.coeffs
        if exponents is None:
            exponents = self.exponents
        fu, fv, _, _, _ = PolySurface3D._poly_derivatives(u, v, coeffs, exponents)
        n = np.stack([-fu, -fv, np.ones_like(fu)], axis=-1)
        n_norm = np.linalg.norm(n, axis=-1, keepdims=True)
        return n / n_norm

    def tangent_vectors(self, u, v, coeffs=None, exponents=None, 
                        align_axis=0, orthogonalized_Q=True):
        if coeffs is None:
            coeffs = self.coeffs
        if exponents is None:
            exponents = self.exponents
        fu, fv, _, _, _ = PolySurface3D._poly_derivatives(u, v, coeffs, exponents)
        t_u = np.stack([np.ones_like(fu), np.zeros_like(fu), fu], axis=-1)
        t_v = np.stack([np.zeros_like(fv), np.ones_like(fv), fv], axis=-1)

        t_u = t_u / np.linalg.norm(t_u, axis=-1, keepdims=True)
        t_v = t_v / np.linalg.norm(t_v, axis=-1, keepdims=True)
        if orthogonalized_Q: 
            if align_axis == 0: 
                e1 = t_u 
                e2 = t_v - np.sum(t_v * e1, axis=-1, keepdims=True) * e1
                e2 = e2 / np.linalg.norm(e2, axis=-1, keepdims=True)
            elif align_axis == 1: 
                e2 = t_v 
                e1 = t_u - np.sum(t_u * e2, axis=-1, keepdims=True) * e2
                e1 = e1 / np.linalg.norm(e1, axis=-1, keepdims=True)        
            else: 
                raise ValueError("align_axis must be 0 or 1")
        else: 
            e1, e2 = t_u, t_v
        return e1, e2

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

    def _uvw_to_residual(self, uvw):
        uvw = np.asarray(uvw, dtype=float)
        w_hat = self.f(uvw[:, 0], uvw[:, 1])
        return uvw[:, 2] - w_hat

    def compute_xyz_residual(self, xyz):
        uvw = self.xyz_to_uvw(xyz)
        return self._uvw_to_residual(uvw)

    def remove_xyz_by_residual_threshold(self, xyz, inliner_range, return_res_Q=False, 
                                         dctr_Q=False):
        assert len(inliner_range) == 2 and inliner_range[0] <= inliner_range[1], "inliner_range must be a tuple (min, max) with min <= max"
        res = self.compute_xyz_residual(xyz)
        if dctr_Q:
            res = res - np.nanmedian(res)
        inlier_Q = (res >= inliner_range[0]) & (res <= inliner_range[1])
        if return_res_Q:
            return xyz[inlier_Q], res[inlier_Q]
        return xyz[inlier_Q]

    def project_xyz_points_to_surface(self, xyz_points,
                                      max_iter=100, tol=1e-6,
                                      return_dist_Q=False, 
                                      return_xyz_Q=True):
        uvw = self.xyz_to_uvw(xyz_points)
        result = self.project_uvw_points_to_surface(
            uvw, max_iter=max_iter, tol=tol, return_dist_Q=return_dist_Q)

        if return_dist_Q:
            uvw_proj, dists = result
            return self.uvw_to_xyz(uvw_proj) if return_xyz_Q else uvw_proj, dists
        return self.uvw_to_xyz(result) if return_xyz_Q else result

    def project_uvw_points_to_surface(self, uvw_points,
                                      max_iter=100, tol=1e-6,
                                      return_dist_Q=False):
        uvw_points = np.asarray(uvw_points, dtype=float)
        if uvw_points.ndim != 2 or uvw_points.shape[1] != 3:
            raise ValueError("uvw_points must be shape (N, 3)")

        u1, v1, w1 = PolySurface3D._project_uvw_points_to_surface(
            uvw_points[:, 0], uvw_points[:, 1], uvw_points[:, 2],
            self.coeffs, self.exponents, max_iter=max_iter, tol=tol)
        uvw_proj = np.column_stack([u1, v1, w1])

        if return_dist_Q:
            dists = np.linalg.norm(uvw_points - uvw_proj, axis=1)
            return uvw_proj, dists
        return uvw_proj

    @staticmethod
    def _project_uvw_points_to_surface(u0, v0, w0, coeffs, exponents,
                                       max_iter=100, tol=1e-6):
        """Project points onto polynomial surface by Newton updates in (u,v)."""
        u0 = np.asarray(u0, dtype=float).flatten()
        v0 = np.asarray(v0, dtype=float).flatten()
        w0 = np.asarray(w0, dtype=float).flatten()

        if not (u0.shape == v0.shape == w0.shape):
            raise ValueError("u0, v0, w0 must have the same shape")

        u = u0.copy()
        v = v0.copy()
        eps = 1e-12

        for _ in range(max_iter):
            w_fit = PolySurface3D._f(u, v, coeffs, exponents)
            fu, fv, fuu, fuv, fvv = PolySurface3D._poly_derivatives(
                u, v, coeffs, exponents)
            r = w_fit - w0

            F1 = (u - u0) + r * fu
            F2 = (v - v0) + r * fv

            # Jacobian of [F1, F2] w.r.t [u, v]
            A = 1.0 + fu * fu + r * fuu
            B = fu * fv + r * fuv
            D = 1.0 + fv * fv + r * fvv
            det = A * D - B * B
            det = np.where(np.abs(det) < eps, np.where(det >= 0, eps, -eps), det)

            du = (D * F1 - B * F2) / det
            dv = (A * F2 - B * F1) / det

            u -= du
            v -= dv

            if np.max(np.abs(du) + np.abs(dv)) < tol:
                break

        w_proj = PolySurface3D._f(u, v, coeffs, exponents)
        return u, v, w_proj

    def uvw_to_tangent_plane(self, uvw, uvw0, coeffs=None, exponents=None):
        """Project points in local uvw coordinates to tangent plane at uvw0."""
        uvw = np.asarray(uvw, dtype=float)
        uvw0 = np.asarray(uvw0, dtype=float).reshape(3,)
        t_u, t_v = self.tangent_vectors(uvw0[0], uvw0[1], coeffs=coeffs, exponents=exponents)
        delta_uvw = uvw - uvw0
        proj_u = np.sum(delta_uvw * t_u, axis=-1)
        proj_v = np.sum(delta_uvw * t_v, axis=-1)
        return np.column_stack([proj_u, proj_v])

    def xyz_in_bbox_Q(self, xyz, extra_pad=0.0):
        if self.bbox_xyz is None:
            raise ValueError("bbox_xyz is not defined for this surface")
        xyz = np.asarray(xyz, dtype=float)
        min_xyz = self.bbox_xyz[:3] - extra_pad
        max_xyz = self.bbox_xyz[3:] + extra_pad
        return np.all((xyz >= min_xyz) & (xyz <= max_xyz), axis=-1)
    
    def uvw_in_bbox_Q(self, uvw, extra_pad=0.0):
        if self.bbox_uvw is None:
            raise ValueError("bbox_uvw is not defined for this surface")
        uvw = np.asarray(uvw, dtype=float)
        min_uvw = self.bbox_uvw[:3] - extra_pad
        max_uvw = self.bbox_uvw[3:] + extra_pad
        return np.all((uvw >= min_uvw) & (uvw <= max_uvw), axis=-1)
    
    def sample_uvw_on_surf(self, step, extra_pad=0.0):
        if self.bbox_uvw is None:
            raise ValueError("bbox_uvw is not defined for this surface")
        min_uvw = self.bbox_uvw[:3] - extra_pad
        max_uvw = self.bbox_uvw[3:] + extra_pad
        u_samples = np.arange(min_uvw[0], max_uvw[0] + step, step)
        v_samples = np.arange(min_uvw[1], max_uvw[1] + step, step)
        U, V = np.meshgrid(u_samples, v_samples)
        W = self.f(U, V)
        return np.column_stack([U.flatten(), V.flatten(), W.flatten()])

    def sample_xyz_on_surf(self, step, extra_pad=0.0):
        uvw_samples = self.sample_uvw_on_surf(step, extra_pad=extra_pad)
        return self.uvw_to_xyz(uvw_samples)

class PCSurface3D(PolySurface3D):
    """Fit a k-order polynomial surface in local uvw coordinates.

    Surface model:
        w = sum_{i+j<=k} c_{ij} * u^i * v^j

    Notes:
        - Works for any integer order k >= 1.
    """

    def __init__(self, points_xyz: np.ndarray, k: int = 2,
                 R: Optional[np.ndarray] = None, t: Optional[np.ndarray] = None):
        self.points_xyz = np.asarray(points_xyz, dtype=float)
        assert self.points_xyz.ndim == 2 and self.points_xyz.shape[1] == 3, "points_xyz must be shape (N, 3)"
        assert k >=1, "k must be >= 1"
        exponents = PolySurface3D._generate_exponents(k)

        if R is None and t is None:
            t_fit = self.points_xyz.mean(axis=0)
            X = self.points_xyz - t_fit
            _, _, V = np.linalg.svd(X, full_matrices=False)
            R_fit = V.T
        elif R is not None and t is not None:
            R_fit = np.asarray(R, dtype=float)
            t_fit = np.asarray(t, dtype=float)
            if t_fit.shape != (3,) or R_fit.shape != (3, 3):
                raise ValueError("Expect R.shape == (3, 3) and t.shape == (3,)")
            X = self.points_xyz - t_fit
        else:
            raise ValueError("Both R and t should be provided")

        self.points_uvw = X @ R_fit
        u, v, w = self.points_uvw[:, 0], self.points_uvw[:, 1], self.points_uvw[:, 2]

        A = PolySurface3D._uv_to_polynomial_array(u, v, exponents)
        coeffs = PCSurface3D._solve_scaled_lstsq(A, w)
        
        super().__init__(coeffs=coeffs, exponents=exponents, R=R_fit, t=t_fit, bbox_xyz=None, bbox_uvw=None)
        self.points_uvw = X @ self.R
        self.bbox_xyz = np.concatenate([self.points_xyz.min(axis=0), self.points_xyz.max(axis=0)])
        self.bbox_uvw = np.concatenate([self.points_uvw.min(axis=0), self.points_uvw.max(axis=0)])
        u, v, w = self.points_uvw[:, 0], self.points_uvw[:, 1], self.points_uvw[:, 2]
        self.w_fit = self.f(u, v)
        self.residuals_w = w - self.w_fit

    @staticmethod
    def _solve_scaled_lstsq(A, b):
        # Solve on column-normalized A to improve conditioning for higher-order terms.
        col_scale = np.linalg.norm(A, axis=0)
        col_scale = np.where(col_scale > 0.0, col_scale, 1.0)
        A_scaled = A / col_scale
        coeffs_scaled, *_ = np.linalg.lstsq(A_scaled, b, rcond=None)
        return coeffs_scaled / col_scale

    @cached_property
    def R2(self):
        ss_res = np.sum(self.residuals_w ** 2)
        ss_tot = np.sum((self.points_uvw[:, 2] - self.points_uvw[:, 2].mean()) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    
    @property
    def IQR_std(self):
        return float(np.diff(np.percentile(self.residuals_w, [25, 75]))[0]) / 1.349

    def compute_residual_stats(self, xyz): 
        uvw = self.xyz_to_uvw(xyz)
        res = self._uvw_to_residual(uvw)
        ss_res = np.sum(res ** 2)
        ss_tot = np.sum((uvw[:, 2] - uvw[:, 2].mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
        rmse = np.sqrt(ss_res / uvw.shape[0]) if uvw.shape[0] > 0 else 0.0
        return r2, rmse

    def residual_ipr_range(self, ipr=1.5):
        return stat.compute_percentile_outlier_threshold(self.residuals_w, ipr=ipr)

    @cached_property
    def point_nearest_uvw(self):
        return self.project_uvw_points_to_surface(self.points_uvw)

    def vis_points_w_fitted_surface(self, grid_res=80, pad=0.05,
                                    vis_frame='world', fig=None, ax=None,
                                    vis_pts_Q=True, label=None):
        P = self.points_xyz
        L = self.points_uvw
        u, v = L[:, 0], L[:, 1]

        umin, umax = u.min(), u.max()
        vmin, vmax = v.min(), v.max()
        du = (umax - umin) * pad
        dv = (vmax - vmin) * pad

        ug = np.linspace(umin - du, umax + du, grid_res)
        vg = np.linspace(vmin - dv, vmax + dv, grid_res)
        U, V = np.meshgrid(ug, vg)
        W = self.f(U, V)

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
        ax.set_title(f"Order-{self.k} polynomial surface fit in {vis_frame} frame")
        plt.tight_layout()
        return fig, ax

    @staticmethod
    def test_points_between_two_surfaces(surf1, surf2,
                                        points_xyz,
                                        select_horizontal_Q=False):
        uvw1 = surf1.xyz_to_uvw(points_xyz)
        if np.all(surf1.R == surf2.R) and np.all(surf1.centroid == surf2.centroid):
            uvw2 = uvw1
        else:
            print("Warning: The two surfaces have different orientations or centroids. Converting points to local UVW coordinates separately for each surface.")
            uvw2 = surf2.xyz_to_uvw(points_xyz)

        res1 = surf1._uvw_to_residual(uvw1)
        res2 = surf2._uvw_to_residual(uvw2)

        between_Q = (res1 * res2) < 0
        if select_horizontal_Q:
            uvw1_min, uvw1_max = surf1.bbox_uvw[:3], surf1.bbox_uvw[3:]
            uvw2_min, uvw2_max = surf2.bbox_uvw[:3], surf2.bbox_uvw[3:]
            uvw_min = np.minimum(uvw1_min, uvw2_min)
            uvw_max = np.maximum(uvw1_max, uvw2_max)
            within_bbox_Q = np.all((uvw1 >= uvw_min) & (uvw1 <= uvw_max), axis=1)
            between_Q = between_Q & within_bbox_Q

        return between_Q

    def vis_residual(self, fig=None, ax=None):
        if fig is None or ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        abs_max_res = np.max(np.abs(self.residuals_w))
        ax.scatter(self.points_uvw[:, 0], self.points_uvw[:, 1], c=self.residuals_w, s=5, alpha=0.9,
                   cmap='coolwarm', vmin=-abs_max_res, vmax=abs_max_res)
        ax.set_xlabel("u")
        ax.set_ylabel("v")
        ax.set_aspect('equal')
        fig.colorbar(ax.collections[0], ax=ax, label="Residual")
        fig.tight_layout()
        return fig, ax

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
        bin_idx, nn_pt_idx_u = util.bin_data_to_idx_list(nn_pt_rid_idx.flatten())
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
    
class GridDownsamplerND:
    """n-dimensional grid downsampling with centroid aggregation.

    Provide ``points`` and either ``cell_size`` or ``grid_shape`` at
    initialization. Core binning state is computed immediately.
    Optional outputs are computed lazily when queried.
    """

    def __init__(self, points, cell_size=None, grid_shape=None,
                 origin=None, extent=None, features=None):
        if cell_size is None and grid_shape is None:
            raise ValueError("Either cell_size or grid_shape must be provided")

        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2:
            raise ValueError("points must have shape (N, D)")
        if pts.shape[0] == 0:
            raise ValueError("points must contain at least one point")

        self.points = pts
        self.dim = int(pts.shape[1])

        if origin is None:
            self.origin = pts.min(axis=0)
        else:
            self.origin = np.asarray(origin, dtype=float).reshape(-1)
            if self.origin.size != self.dim:
                raise ValueError("origin must be shape (D,)")

        if extent is None:
            self.extent = pts.max(axis=0) - self.origin
            self.extent = self.extent + np.finfo(float).eps
        else:
            self.extent = np.asarray(extent, dtype=float).reshape(-1)
            if self.extent.size != self.dim:
                raise ValueError("extent must be shape (D,)")
            if np.any(self.extent <= 0):
                raise ValueError("extent must be strictly positive")

        if grid_shape is not None:
            grid_shape_arr = np.asarray(grid_shape, dtype=np.int64).reshape(-1)
            if grid_shape_arr.size != self.dim:
                raise ValueError("grid_shape must be shape (D,)")
            if np.any(grid_shape_arr <= 0):
                raise ValueError("grid_shape must be strictly positive integers")
        else:
            grid_shape_arr = None

        if cell_size is not None:
            cell_size_arr = np.asarray(cell_size, dtype=float).reshape(-1)
            if cell_size_arr.size == 1:
                cell_size_arr = np.repeat(cell_size_arr[0], self.dim)
            if cell_size_arr.size != self.dim:
                raise ValueError("cell_size vector length must match point dimensionality")
            if np.any(cell_size_arr <= 0):
                raise ValueError("cell_size must be strictly positive")
        else:
            cell_size_arr = None

        if grid_shape_arr is None:
            grid_shape_arr = np.ceil(self.extent / cell_size_arr).astype(np.int64)
            grid_shape_arr = np.maximum(grid_shape_arr, 1)
        if cell_size_arr is None:
            cell_size_arr = self.extent / grid_shape_arr

        self.grid_shape = grid_shape_arr
        self.cell_size_vec = cell_size_arr

        self.subscripts = np.floor((self.points - self.origin) / self.cell_size_vec).astype(np.int64)
        self.input_mask = np.all((self.subscripts >= 0) & (self.subscripts < self.grid_shape), axis=1)
        self.input_indices = np.flatnonzero(self.input_mask)
        self.points_in = self.points[self.input_mask]
        self.subscripts_in = self.subscripts[self.input_mask]

        if self.points_in.shape[0] == 0:
            self.flat_ids_in = np.empty(0, dtype=np.int64)
        else:
            self.flat_ids_in = np.ravel_multi_index(self.subscripts_in.T, dims=tuple(self.grid_shape))

        if features is not None:
            feat = np.asarray(features, dtype=float)
            if feat.ndim == 1:
                feat = feat.reshape(-1, 1)
            if feat.ndim != 2:
                raise ValueError("features must be a 1D vector or 2D (N, d) array")
            if feat.shape[0] != self.points.shape[0]:
                raise ValueError(
                    f"features.shape[0] ({feat.shape[0]}) must match "
                    f"number of points ({self.points.shape[0]})"
                )
            self.features = feat
            self.n_features = int(feat.shape[1])
            self.features_in = feat[self.input_mask]
        else:
            self.features = None
            self.n_features = 0
            self.features_in = None

        self._grouped_cache = None

    def _grouped(self):
        if self._grouped_cache is not None:
            return self._grouped_cache

        if self.flat_ids_in.size == 0:
            empty = {
                "cell_ids": np.empty(0, dtype=np.int64),
                "cell_subscripts": np.empty((0, self.dim), dtype=np.int64),
                "counts": np.empty(0, dtype=np.int64),
                "centroids": np.empty((0, self.dim), dtype=float),
                "inverse": np.empty(0, dtype=np.int64),
                "sorted_idx": np.empty(0, dtype=np.int64),
                "start": np.empty(0, dtype=np.int64),
                "end": np.empty(0, dtype=np.int64),
            }
            self._grouped_cache = empty
            return empty

        cell_ids, inverse, counts = np.unique(
            self.flat_ids_in, return_inverse=True, return_counts=True
        )
        n_occ = int(cell_ids.size)
        sums = np.empty((n_occ, self.dim), dtype=float)
        for axis in range(self.dim):
            sums[:, axis] = np.bincount(
                inverse,
                weights=self.points_in[:, axis],
                minlength=n_occ,
            )
        centroids = sums / counts[:, None]
        cell_subscripts = np.array(np.unravel_index(cell_ids, tuple(self.grid_shape))).T

        sorted_idx = np.argsort(inverse, kind="mergesort")
        end = np.cumsum(counts, dtype=np.int64)
        start = np.concatenate(([0], end[:-1]))

        self._grouped_cache = {
            "cell_ids": cell_ids,
            "cell_subscripts": cell_subscripts,
            "counts": counts.astype(np.int64),
            "centroids": centroids,
            "inverse": inverse,
            "sorted_idx": sorted_idx,
            "start": start,
            "end": end,
        }
        return self._grouped_cache

    @property
    def centroids(self):
        return self._grouped()["centroids"]

    @property
    def counts(self):
        return self._grouped()["counts"]

    @property
    def cell_ids(self):
        return self._grouped()["cell_ids"]

    @property
    def cell_subscripts(self):
        return self._grouped()["cell_subscripts"]

    def get_cell_bboxes(self):
        """Return occupied-cell bounding boxes as [min..., max...]."""
        mins = self.origin + self.cell_subscripts * self.cell_size_vec
        maxs = mins + self.cell_size_vec
        return np.concatenate([mins, maxs], axis=1)

    def get_cell_feature_stats(self, func):
        """Compute per-cell feature statistics.

        Parameters
        ----------
        func : callable
            Aggregation function applied to a ``(k, d)`` sub-array of
            features along ``axis=0``.  Standard NumPy reductions
            (``np.mean``, ``np.std``, ``np.min``, ``np.max``,
            ``np.median``, ...) are directly supported.  Custom callables
            must accept an ``axis`` keyword argument.

        Returns
        -------
        stats : ndarray, shape ``(M, d)``
            Per-cell statistics aligned with ``self.centroids``.
            Returns an empty ``(0, d)`` array when no in-bounds points
            exist.

        Raises
        ------
        ValueError
            If ``features`` was not provided at initialization.
        """
        if self.features is None:
            raise ValueError(
                "No features provided at initialization. "
                "Pass 'features' when constructing GridDownsamplerND."
            )
        grouped = self._grouped()
        m = int(grouped["cell_ids"].size)
        if m == 0:
            return np.empty((0, self.n_features), dtype=float)

        stats_rows = []
        for cell_idx in range(m):
            s = int(grouped["start"][cell_idx])
            e = int(grouped["end"][cell_idx])
            block_local = grouped["sorted_idx"][s:e]
            cell_feats = self.features_in[block_local]  # (k, d)
            stats_rows.append(func(cell_feats, axis=0))
        return np.array(stats_rows, dtype=float)

    def nearest_points_to_centroids(self, return_dist_Q=False, return_mask_Q=True):
        """Find nearest in-cell input point for each occupied cell centroid."""
        grouped = self._grouped()
        m = int(grouped["cell_ids"].size)

        nearest_indices = np.empty(m, dtype=np.int64)
        nearest_points = np.empty((m, self.dim), dtype=float)
        nearest_dists = np.empty(m, dtype=float) if return_dist_Q else None

        for cell_idx in range(m):
            s = int(grouped["start"][cell_idx])
            e = int(grouped["end"][cell_idx])
            block_local = grouped["sorted_idx"][s:e]
            block_points = self.points_in[block_local]
            delta = block_points - grouped["centroids"][cell_idx]
            dist2 = np.einsum("ij,ij->i", delta, delta)
            best_local = int(np.argmin(dist2))
            inlier_idx = int(block_local[best_local])
            global_idx = int(self.input_indices[inlier_idx])

            nearest_indices[cell_idx] = global_idx
            nearest_points[cell_idx] = self.points[global_idx]
            if return_dist_Q:
                nearest_dists[cell_idx] = float(np.sqrt(dist2[best_local]))

        matched_mask = None
        if return_mask_Q:
            matched_mask = np.zeros(self.points.shape[0], dtype=bool)
            matched_mask[nearest_indices] = True

        if return_dist_Q and return_mask_Q:
            return nearest_points, nearest_indices, nearest_dists, matched_mask
        if return_dist_Q:
            return nearest_points, nearest_indices, nearest_dists
        if return_mask_Q:
            return nearest_points, nearest_indices, matched_mask
        return nearest_points, nearest_indices

    def downsample(self, return_counts=False, return_bboxes=False,
                   return_cell_ids=False, return_subscripts=False):
        """Return centroids or requested extra per-cell outputs."""
        if not (return_counts or return_bboxes or return_cell_ids or return_subscripts):
            return self.centroids

        result: dict[str, Any] = {"centroids": self.centroids}
        if return_counts:
            result["counts"] = self.counts
        if return_bboxes:
            result["bboxes"] = self.get_cell_bboxes()
        if return_cell_ids:
            result["cell_ids"] = self.cell_ids
        if return_subscripts:
            result["subscripts"] = self.cell_subscripts
        return result

def grid_centroids(points: np.ndarray,
                   grid_shape: tuple[int, int, int],
                   origin: np.ndarray | None = None,
                   extent: np.ndarray | None = None,
                   return_counts: bool = False):
    """Backward-compatible 3D centroid computation on a regular grid."""
    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")

    shape = np.asarray(grid_shape, dtype=np.int64).reshape(-1)
    if shape.size != 3 or np.any(shape <= 0):
        raise ValueError("grid_shape must be positive integers (nx, ny, nz)")

    model = GridDownsamplerND(
        points=pts,
        grid_shape=shape,
        origin=origin,
        extent=extent,
    )

    if return_counts:
        return model.centroids, model.cell_subscripts, model.counts
    return model.centroids, model.cell_subscripts

def downsample_points_by_averaging(pts, grid_size, features=None):
    if pts is None or len(pts) == 0:
        return pts
    else: 
        pts = np.asarray(pts)
        if pts.ndim != 2:
            raise ValueError("pts must have shape (N, D)")
        model = GridDownsamplerND(points=pts, cell_size=grid_size, 
                                  features=features)
        if features is None: 
            return model.downsample()
        else: 
            pts_ds = model.downsample()
            features_ds = model.get_cell_feature_stats(func=np.nanmean)
            return pts_ds, features_ds

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
    pts_stat = select_points_near_pc1(pts, ipr=ipr, max_dist_th=max_dist_th)
    for i in range(max_iter):
        if pts_stat['num_outliers'] == 0:
            break
        pts_stat = select_points_near_pc1(pts_stat['inlier_pts'], ipr=ipr, 
                                          max_dist_th=max_dist_th)
    return pts_stat
