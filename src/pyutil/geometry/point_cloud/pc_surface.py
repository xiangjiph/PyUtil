from functools import cached_property
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ... import stat
from .surface import PolySurface3D


class PCSurface3D(PolySurface3D):
    """Fit a k-order polynomial surface in local uvw coordinates.

    Surface model:
        w = sum_{i+j<=k} c_{ij} * u^i * v^j
    """

    def __init__(self, points_xyz: np.ndarray, k: int = 2,
                 R: Optional[np.ndarray] = None, t: Optional[np.ndarray] = None,
                 residual_range: Optional[tuple] = None):
        self.points_xyz = np.asarray(points_xyz, dtype=float)
        assert self.points_xyz.ndim == 2 and self.points_xyz.shape[1] == 3, "points_xyz must be shape (N, 3)"
        assert k >= 1, "k must be >= 1"
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

        super().__init__(coeffs=coeffs, exponents=exponents, R=R_fit, t=t_fit,
                         bbox_xyz=None, bbox_uvw=None, residual_range=residual_range)
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
    def points_kdtree(self):
        """Return the cached KD-tree for the fitted point cloud."""
        return cKDTree(self.points_xyz.astype(np.int32))

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

    def surface_range(self, ipr=None):
        """If ipr is provided, compute the inter-percentile range of residuals;
        otherwise, return the range provided in initialization."""
        if ipr is not None:
            return self.residual_ipr_range(ipr=ipr)
        assert len(self.residual_range) == 2 and self.residual_range[0] <= self.residual_range[1], \
            "residual_range must be a tuple (min, max) with min <= max"
        return self.residual_range

    @cached_property
    def point_nearest_uvw(self):
        return self.project_uvw_points_to_surface(self.points_uvw)

    def sample_surface_xyz(self, grid_spacing, ret_coord="xyz"):
        """Sample the fitted surface on a regular uv grid.

        Parameters
        ----------
        grid_spacing : float
            Sampling step in the local u and v coordinates.
        ret_coord : {"xyz", "uvw"}, default "xyz"
            Coordinate frame of the returned points.

        Returns
        -------
        ndarray, shape (N, 3)
            Sampled surface points.
        """
        u_min, v_min = np.floor(self.points_uvw[:, :2].min(axis=0) / grid_spacing) * grid_spacing
        u_max, v_max = np.ceil(self.points_uvw[:, :2].max(axis=0) / grid_spacing) * grid_spacing
        u, v = np.meshgrid(np.arange(u_min, u_max + grid_spacing, grid_spacing),
                           np.arange(v_min, v_max + grid_spacing, grid_spacing), indexing="xy")
        uvw = np.column_stack([u.ravel(), v.ravel(), self.f(u.ravel(), v.ravel())])
        assert ret_coord in {"xyz", "uvw"}, "ret_coord must be 'xyz' or 'uvw'"
        return uvw if ret_coord == "uvw" else self.uvw_to_xyz(uvw)

    def query_points_near_normal(self, xyz, h, l, dr=None):
        """Find fitted points near the surface normal through one point.

        Parameters
        ----------
        xyz : array-like, shape (3,)
            Point on the fitted surface in world coordinates.
        h : float
            Search distance along both directions of the surface normal.
        l : float
            KD-tree query radius around each sampled normal-line point.
        dr : float, optional
            Normal-line sampling step. Defaults to ``l / 10``.

        Returns
        -------
        dict
            Normal-line data plus the unique point indices and coordinates
            found by the KD-tree queries.
        """
        dr = float(l) / 10 if dr is None else float(dr)
        result = self.sample_normal_line_xyz(xyz, h, dr)
        point_idx = np.unique(np.concatenate(self.points_kdtree.query_ball_point(result["X"], float(l)))).astype(int)
        return result | {"point_idx": point_idx, "points_xyz": self.points_xyz[point_idx],
                         "h": float(h), "l": float(l), "dr": dr}

    def compute_normal_spread(self, xyz, h, l, dr=None, coor='xyz'):
        """Compute the signed normal spread of nearby fitted points.

        Parameters
        ----------
        xyz : array-like, shape (3,)
            Point on the fitted surface in world coordinates.
        h, l, dr : float
            Axial half-range, lateral query radius, and optional axial step
            passed to :meth:`query_points_near_normal`.

        Returns
        -------
        dict
            Query data, signed normal offsets, and their mean, standard
            deviation, median, 25th percentile, and 75th percentile.
        """
        if coor == 'uvw':
            xyz = self.uvw_to_xyz(xyz)
        result = self.query_points_near_normal(xyz, h, l, dr=dr)
        # Signed spread is the point displacement projected onto the unit surface normal.
        normal_offset = (result["points_xyz"] - result["r0"]) @ result["n"]
        values = ([np.mean(normal_offset), np.std(normal_offset), np.median(normal_offset),
                   *np.percentile(normal_offset, [25, 75])]
                  if normal_offset.size else [np.nan] * 5)
        keys = ["normal_mean", "normal_std", "normal_median", "normal_p25", "normal_p75"]
        return result | {"normal_offset": normal_offset, **dict(zip(keys, values))}

    def compute_surface_normal_spread(self, grid_spacing, h, l, dr=None):
        """Compute local normal-spread statistics across a sampled surface.

        Parameters
        ----------
        grid_spacing : float
            Surface sampling step in local u and v coordinates.
        h, l, dr : float
            Axial half-range, lateral query radius, and optional axial step
            passed to :meth:`compute_normal_spread`.

        Returns
        -------
        pandas.DataFrame
            One row per sampled surface point with xyz, normal vector, nearby
            point count, and signed normal-spread statistics.
        """
        rows = []
        for xyz in self.sample_surface_xyz(grid_spacing):
            data = self.compute_normal_spread(xyz, h, l, dr=dr)
            rows.append({"x": xyz[0], "y": xyz[1], "z": xyz[2],
                         "nx": data["n"][0], "ny": data["n"][1], "nz": data["n"][2],
                         "num_points": data["points_xyz"].shape[0],
                         **{key: data[key] for key in ["normal_mean", "normal_std", "normal_median",
                                                       "normal_p25", "normal_p75"]}})
        return pd.DataFrame(rows)

    def vis_points_w_fitted_surface(self, grid_res=80, pad=0.05,
                                    vis_frame='world', fig=None, ax=None,
                                    vis_pts_Q=True, label=None, vis_unit='nm'):
        if vis_unit in ['nm']:
            scale = 1.0
        elif vis_unit in ['um', 'µm']:
            scale = 1e-3
        else:
            raise NotImplementedError
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
                ax.scatter(L[:, 0] * scale, L[:, 1] * scale, L[:, 2] * scale, s=1, alpha=0.9)
            ax.plot_surface(U * scale, V * scale, W * scale, alpha=0.25, linewidth=0, label=label)
        else:
            if vis_pts_Q:
                ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=1, alpha=0.9)
            world_grid = self.uvw_to_xyz(np.stack([U, V, W], axis=-1)) * scale
            ax.plot_surface(world_grid[..., 0], world_grid[..., 1], world_grid[..., 2], alpha=0.25,
                            linewidth=0, label=label)

        ax.set_xlabel(f"X ({vis_unit})")
        ax.set_ylabel(f"Y ({vis_unit})")
        ax.set_zlabel(f"Z ({vis_unit})")
        ax.set_title(f"Order-{self.k} polynomial surface fit in {vis_frame} frame")
        plt.tight_layout()
        return fig, ax

    @staticmethod
    def test_points_between_two_surfaces(surf1, surf2, points_xyz, select_horizontal_Q=False):
        uvw1 = surf1.xyz_to_uvw(points_xyz)
        if np.all(surf1.R == surf2.R) and np.all(surf1.centroid == surf2.centroid):
            uvw2 = uvw1
        else:
            print("Warning: The two surfaces have different orientations or centroids. "
                  "Converting points to local UVW coordinates separately for each surface.")
            uvw2 = surf2.xyz_to_uvw(points_xyz)
        between_Q = (surf1._uvw_to_residual(uvw1) * surf2._uvw_to_residual(uvw2)) < 0
        if select_horizontal_Q:
            uvw_min = np.minimum(surf1.bbox_uvw[:3], surf2.bbox_uvw[:3])
            uvw_max = np.maximum(surf1.bbox_uvw[3:], surf2.bbox_uvw[3:])
            between_Q &= np.all((uvw1 >= uvw_min) & (uvw1 <= uvw_max), axis=1)
        return between_Q

    def vis_residual(self, fig=None, ax=None):
        if fig is None or ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        abs_max_res = np.max(np.abs(self.residuals_w))
        ax.scatter(self.points_uvw[:, 0], self.points_uvw[:, 1], c=self.residuals_w, s=5, alpha=0.9,
                   cmap='coolwarm', vmin=-abs_max_res, vmax=abs_max_res)
        ax.set(xlabel="u", ylabel="v", aspect='equal')
        fig.colorbar(ax.collections[0], ax=ax, label="Residual")
        fig.tight_layout()
        return fig, ax

    def compute_point_distance_to_surface_stat(self, num_sample=None, return_pts_Q=False):
        """ Compute the distance from points to the fitted surface and return basic statistics.

        """
        if num_sample is None or num_sample >= self.points_xyz.shape[0]:
            pts_uvw = self.points_uvw
        else:
            idx = np.random.choice(self.points_xyz.shape[0], size=num_sample, replace=False)
            pts_uvw = self.points_uvw[idx]
        nearest_uvw, nearest_dist = self.project_uvw_points_to_surface(pts_uvw, return_dist_Q=True)
        
        # determine if the points are above or below the surface by checking the sign of the residuals
        sample_pts_res = self._uvw_to_residual(pts_uvw)
        sample_pts_sign = np.sign(sample_pts_res)
        nearest_dist *= sample_pts_sign

        dist_stat = {
            'std': float(np.std(nearest_dist)),
            'iqr': float(np.diff(np.percentile(nearest_dist, [25, 75]))[0]),
        }
        if return_pts_Q:
            result = {'pts_uvw': pts_uvw, 'pts_uvwp': nearest_uvw, 'pts_dist': nearest_dist} | dist_stat
            return result
        return dist_stat
