from typing import Any

import numpy as np


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
