from functools import cached_property

import numpy as np
import matplotlib.pyplot as plt

from ... import stat

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
        points_uvw = X @ self.R
        u, v, w = points_uvw[:, 0], points_uvw[:, 1], points_uvw[:, 2]

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
