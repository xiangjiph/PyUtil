from typing import Optional

import numpy as np
from numpy.polynomial import polynomial as nppoly
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.spatial import cKDTree

class PolySurface3D:
    """Parameterized polynomial surface in local uvw coordinates.

    Surface model:
        w = sum_{i+j<=k} c_{ij} * u^i * v^j

    This class is initialized directly from polynomial parameters and an
    optional world-to-local frame (R, t), so no point-cloud refit is needed.
    ``R`` is expected to be orthonormal, as it represents a pure rotation.
    """

    def __init__(self, coeffs, k: Optional[int] = None,
                 exponents: Optional[tuple] = None,
                 R: Optional[np.ndarray] = None,
                 t: Optional[np.ndarray] = None,
                 bbox_xyz: Optional[tuple] = None,
                 bbox_uvw: Optional[tuple] = None, 
                 residual_range: Optional[tuple] = None):
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
        
        if residual_range is None: 
            residual_range = (None, None)
        else: 
            if len(residual_range) != 2:
                raise ValueError("residual_range must be a tuple (min, max)")
            if residual_range[0] is not None and residual_range[1] is not None:
                if residual_range[0] > residual_range[1]:
                    raise ValueError("residual_range min must be <= max")

        self.coeffs = coeffs
        self.exponents = exponents
        self.R = R
        self.centroid = t
        self.k = max(i + j for i, j in self.exponents)
        self.bbox_xyz = bbox_xyz
        self.bbox_uvw = bbox_uvw
        self.residual_range = residual_range

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

    def sample_normal_line_xyz(self, xyz, h, dh):
        """Sample a line through one surface point along its normal.

        Parameters
        ----------
        xyz : array-like, shape (3,)
            Point on the surface in world coordinates.
        h : float
            Distance sampled in both normal directions.
        dh : float
            Distance between consecutive samples.

        Returns
        -------
        dict
            ``r0`` is the input point, ``n`` is the unit normal in world
            coordinates, and ``X`` contains the sampled world coordinates.
        """
        r0 = np.asarray(xyz, dtype=float).reshape(3)
        uvw = self.xyz_to_uvw(r0)[0]
        n = self.normal_vector(uvw[0], uvw[1]) @ self.R.T
        distance = np.arange(-float(h), float(h) + float(dh) / 2, float(dh))
        return {"r0": r0, "n": n, "X": r0 + distance[:, None] * n}

    def tangent_vectors(self, u, v, align_axis=0, orthogonalized_Q=True):
        coeffs = self.coeffs
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

    def geodesic_distance(self, uvw1, uvw2, num_segments=24,
                          quadrature_order=8, tol=1e-8, max_iter=500,
                          surface_tol=1e-6, return_path_Q=False):
        """Approximate geodesic distance between two surface points.

        The path is represented by ``num_segments`` straight segments in the
        ``(u, v)`` parameter domain. Interior knots are optimized by minimizing
        the discrete path energy while the endpoint coordinates remain fixed.
        Segment lengths are then integrated on the lifted polynomial surface
        with Gauss-Legendre quadrature. Increasing ``num_segments`` improves
        the spatial approximation; increasing ``quadrature_order`` improves
        integration accuracy along each segment.

        Parameters
        ----------
        uvw1, uvw2 : array-like, shape (3,)
            Endpoints in local coordinates. Their ``w`` values must lie on the
            polynomial surface within ``surface_tol``.
        num_segments : int, default 24
            Number of piecewise-linear segments in the parameter domain.
        quadrature_order : int, default 8
            Gauss-Legendre nodes used to integrate each segment.
        tol : float, default 1e-8
            Gradient and objective tolerance for the L-BFGS-B optimization.
        max_iter : int, default 500
            Maximum optimizer iterations.
        surface_tol : float, default 1e-6
            Absolute and relative tolerance for endpoint surface validation.
        return_path_Q : bool, default False
            Return the optimized surface knots with the distance.

        Returns
        -------
        distance : float
            Approximate geodesic distance in world-coordinate length units.
        path_uvw : ndarray, shape (num_segments + 1, 3), optional
            Optimized surface knots, returned when ``return_path_Q`` is true.

        Notes
        -----
        This finite-dimensional solve converges to a geodesic as the path is
        refined, but a non-convex surface can have multiple local geodesics.
        The optimizer starts from the straight line in ``(u, v)`` and returns
        the locally shortest path reached from that initialization.
        """
        uvw1 = np.asarray(uvw1, dtype=float)
        uvw2 = np.asarray(uvw2, dtype=float)
        if uvw1.shape != (3,) or uvw2.shape != (3,):
            raise ValueError("uvw1 and uvw2 must each have shape (3,)")
        if not (np.all(np.isfinite(uvw1)) and np.all(np.isfinite(uvw2))):
            raise ValueError("uvw1 and uvw2 must contain only finite values")

        num_segments = int(num_segments)
        quadrature_order = int(quadrature_order)
        if num_segments < 1:
            raise ValueError("num_segments must be >= 1")
        if quadrature_order < 2:
            raise ValueError("quadrature_order must be >= 2")
        if tol <= 0 or max_iter < 1 or surface_tol < 0:
            raise ValueError("tol and max_iter must be positive; surface_tol must be non-negative")

        for name, point in (("uvw1", uvw1), ("uvw2", uvw2)):
            w_surface = float(np.asarray(self.f(point[0], point[1])).item())
            if not np.isclose(point[2], w_surface, atol=surface_tol, rtol=surface_tol):
                raise ValueError(f"{name} does not lie on the polynomial surface")

        if np.array_equal(uvw1[:2], uvw2[:2]):
            path_uvw = np.vstack([uvw1, uvw2])
            return (0.0, path_uvw) if return_path_Q else 0.0

        alpha = np.linspace(0.0, 1.0, num_segments + 1)[:, None]
        initial_uv = uvw1[:2] + alpha * (uvw2[:2] - uvw1[:2])
        quadrature_x, quadrature_w = np.polynomial.legendre.leggauss(quadrature_order)
        quadrature_t = 0.5 * (quadrature_x + 1.0)
        quadrature_w = 0.5 * quadrature_w

        def path_integral(interior_uv, energy_Q):
            if num_segments == 1:
                path_uv = initial_uv
            else:
                path_uv = np.vstack([
                    uvw1[:2],
                    np.asarray(interior_uv).reshape(num_segments - 1, 2),
                    uvw2[:2],
                ])

            delta_uv = np.diff(path_uv, axis=0)
            uv_quad = (
                path_uv[:-1, None, :]
                + quadrature_t[None, :, None] * delta_uv[:, None, :]
            )
            fu = self.df_du(uv_quad[..., 0], uv_quad[..., 1])
            fv = self.df_dv(uv_quad[..., 0], uv_quad[..., 1])
            du = np.broadcast_to(delta_uv[:, 0, None], fu.shape)
            dv = np.broadcast_to(delta_uv[:, 1, None], fv.shape)

            # Lift each parameter-space velocity through (u, v, f(u, v)).
            # R is orthonormal, so rotating to xyz would not change its norm.
            velocity_uvw = np.stack([du, dv, fu * du + fv * dv], axis=-1)
            speed = np.linalg.norm(velocity_uvw, axis=-1)
            if energy_Q:
                # Equal parameter-time per segment makes this a smooth energy
                # objective and discourages collapsed or unevenly spaced knots.
                return float(num_segments * np.sum(speed * speed * quadrature_w))
            return float(np.sum(speed * quadrature_w))

        if num_segments == 1:
            optimized_uv = initial_uv
        else:
            result = minimize(
                lambda interior_uv: path_integral(interior_uv, energy_Q=True),
                initial_uv[1:-1].ravel(),
                method="L-BFGS-B",
                options={"maxiter": int(max_iter), "ftol": tol, "gtol": tol},
            )
            if not result.success:
                raise RuntimeError(f"Geodesic optimization failed: {result.message}")
            optimized_uv = np.vstack([
                uvw1[:2], result.x.reshape(num_segments - 1, 2), uvw2[:2]
            ])

        # Re-integrate the optimized path with the arc-length functional.
        distance = path_integral(optimized_uv[1:-1].ravel(), energy_Q=False)
        path_uvw = np.column_stack([
            optimized_uv,
            self.f(optimized_uv[:, 0], optimized_uv[:, 1]),
        ])
        if return_path_Q:
            return distance, path_uvw
        return distance

    def pairwise_geodesic_distance(self, uvw, num_neighbors=12,
                                   quadrature_order=4, surface_tol=1e-6):
        """Approximate all pairwise geodesic distances on the surface.

        A connected k-nearest-neighbor graph is built in the ``(u, v)``
        parameter domain. Each edge weight is the polynomial-surface arc
        length of the straight parameter-space segment, evaluated with
        Gauss-Legendre quadrature. Sparse all-pairs Dijkstra then estimates the
        geodesic distance between every input pair.

        This graph method is intended for point sets of roughly 1,000 samples,
        where independently optimizing every pair would be prohibitively slow.
        Increase ``num_neighbors`` for a denser, usually more accurate graph at
        greater time and memory cost. The neighbor count is increased
        automatically only when necessary to connect the graph.

        Parameters
        ----------
        uvw : ndarray, shape (N, 3)
            Points in local coordinates. Every point must lie on the surface.
        num_neighbors : int, default 12
            Initial number of UV-space neighbors per point.
        quadrature_order : int, default 4
            Gauss-Legendre nodes used to integrate each graph edge.
        surface_tol : float, default 1e-6
            Absolute and relative tolerance for surface-point validation.

        Returns
        -------
        distances : ndarray, shape (N, N)
            Symmetric approximate geodesic-distance matrix. The diagonal is
            zero and values use the same length units as the surface frame.
        """
        uvw = np.asarray(uvw, dtype=float)
        if uvw.ndim != 2 or uvw.shape[1] != 3:
            raise ValueError("uvw must have shape (N, 3)")
        if uvw.shape[0] == 0:
            raise ValueError("uvw must contain at least one point")
        if not np.all(np.isfinite(uvw)):
            raise ValueError("uvw must contain only finite values")

        num_neighbors = int(num_neighbors)
        quadrature_order = int(quadrature_order)
        if num_neighbors < 1:
            raise ValueError("num_neighbors must be >= 1")
        if quadrature_order < 2:
            raise ValueError("quadrature_order must be >= 2")
        if surface_tol < 0:
            raise ValueError("surface_tol must be non-negative")

        w_surface = self.f(uvw[:, 0], uvw[:, 1])
        if not np.all(np.isclose(
                uvw[:, 2], w_surface, atol=surface_tol, rtol=surface_tol)):
            raise ValueError("all uvw points must lie on the polynomial surface")

        num_points = uvw.shape[0]
        if num_points == 1:
            return np.zeros((1, 1), dtype=float)

        uv = uvw[:, :2]
        tree = cKDTree(uv)
        neighbor_count = min(num_neighbors, num_points - 1)

        # Expand k only when disconnected samples would otherwise produce
        # infinite distances. Edge lengths are computed after connectivity is
        # established so retries remain inexpensive.
        while True:
            _, neighbor_idx = tree.query(uv, k=neighbor_count + 1)
            neighbor_idx = np.asarray(neighbor_idx).reshape(num_points, -1)
            self_idx = np.arange(num_points)[:, None]
            neighbor_idx = neighbor_idx[neighbor_idx != self_idx].reshape(
                num_points, neighbor_count)

            row = np.repeat(np.arange(num_points), neighbor_count)
            col = neighbor_idx.ravel()
            edges = np.column_stack([np.minimum(row, col), np.maximum(row, col)])
            edges = np.unique(edges[edges[:, 0] != edges[:, 1]], axis=0)

            connectivity = csr_matrix(
                (np.ones(edges.shape[0] * 2),
                 (np.concatenate([edges[:, 0], edges[:, 1]]),
                  np.concatenate([edges[:, 1], edges[:, 0]]))),
                shape=(num_points, num_points),
            )
            if connected_components(connectivity, directed=False)[0] == 1:
                break
            if neighbor_count == num_points - 1:
                raise RuntimeError("Unable to construct a connected surface graph")
            neighbor_count = min(
                num_points - 1, max(neighbor_count + 1, 2 * neighbor_count)
            )

        quadrature_x, quadrature_w = np.polynomial.legendre.leggauss(
            quadrature_order)
        quadrature_t = 0.5 * (quadrature_x + 1.0)
        quadrature_w = 0.5 * quadrature_w
        edge_lengths = np.empty(edges.shape[0], dtype=float)

        # Batching caps temporary memory if connectivity expansion approaches
        # a dense graph. The normal case evaluates all sparse edges at once.
        for start in range(0, edges.shape[0], 100_000):
            stop = min(start + 100_000, edges.shape[0])
            edge_uv0 = uv[edges[start:stop, 0]]
            delta_uv = uv[edges[start:stop, 1]] - edge_uv0
            uv_quad = (
                edge_uv0[:, None, :]
                + quadrature_t[None, :, None] * delta_uv[:, None, :]
            )
            fu = self.df_du(uv_quad[..., 0], uv_quad[..., 1])
            fv = self.df_dv(uv_quad[..., 0], uv_quad[..., 1])
            du = np.broadcast_to(delta_uv[:, 0, None], fu.shape)
            dv = np.broadcast_to(delta_uv[:, 1, None], fv.shape)
            velocity_uvw = np.stack([du, dv, fu * du + fv * dv], axis=-1)
            speed = np.linalg.norm(velocity_uvw, axis=-1)
            edge_lengths[start:stop] = speed @ quadrature_w

        graph = csr_matrix(
            (np.concatenate([edge_lengths, edge_lengths]),
             (np.concatenate([edges[:, 0], edges[:, 1]]),
              np.concatenate([edges[:, 1], edges[:, 0]]))),
            shape=(num_points, num_points),
        )
        return shortest_path(graph, directed=False, method="D")

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

    def intersect_line(
        self,
        line_point,
        line_direction,
        local_Q: bool = True,
        return_xyz_Q: bool = False,
        real_tol: float = 1e-7,
        residual_tol: float = 1e-6,
    ):
        """Return all real intersections between a line and this surface.

        ``line_point`` and ``line_direction`` are interpreted in local ``uvw``
        coordinates when ``local_Q`` is true and in world ``xyz`` coordinates
        otherwise. The solve reduces the surface-line residual to a 1D
        polynomial in the line parameter.
        """
        line_point = np.asarray(line_point, dtype=float).reshape(3)
        line_direction = np.asarray(line_direction, dtype=float).reshape(3)
        if not np.all(np.isfinite(line_point)) or not np.all(np.isfinite(line_direction)):
            raise ValueError("line_point and line_direction must be finite")
        if np.linalg.norm(line_direction) == 0:
            raise ValueError("line_direction must be nonzero")
        if not local_Q:
            line_point = self.xyz_to_uvw(line_point)[0]
            line_direction = line_direction @ self.R

        residual_poly = np.zeros(max(self.k, 1) + 1, dtype=float)
        residual_poly[:2] = [line_point[2], line_direction[2]]
        u_poly = np.asarray([line_point[0], line_direction[0]], dtype=float)
        v_poly = np.asarray([line_point[1], line_direction[1]], dtype=float)
        for coeff, (i, j) in zip(self.coeffs, self.exponents):
            term = nppoly.polymul(nppoly.polypow(u_poly, i), nppoly.polypow(v_poly, j))
            residual_poly[:term.size] -= coeff * term

        # Do not drop small fitted high-order terms by magnitude.  In nanometer
        # coordinates, those coefficients can still dominate at line parameters
        # of tens of microns.
        nonzero_idx = np.flatnonzero(residual_poly != 0)
        if nonzero_idx.size == 0:
            return np.zeros((0, 3), dtype=float)

        roots = nppoly.polyroots(residual_poly[:nonzero_idx[-1] + 1])
        real_Q = np.abs(roots.imag) <= real_tol * np.maximum(1.0, np.abs(roots.real))
        s_values = np.sort(roots.real[real_Q])
        if s_values.size == 0:
            return np.zeros((0, 3), dtype=float)

        unique_s = [s_values[0]]
        for s in s_values[1:]:
            if abs(s - unique_s[-1]) > real_tol * max(1.0, abs(s), abs(unique_s[-1])):
                unique_s.append(s)
        uvw = line_point + np.asarray(unique_s)[:, None] * line_direction
        residual = self._uvw_to_residual(uvw)
        keep_Q = np.isclose(residual, 0.0, atol=residual_tol, rtol=residual_tol)
        uvw = uvw[keep_Q]
        return self.uvw_to_xyz(uvw) if return_xyz_Q else uvw

    def remove_xyz_by_residual_threshold(self, xyz, inliner_range, return_res_Q=False,
                                         dctr_Q=False):
        inlier_Q, res = self.xyz_is_inlier_Q(
            xyz, inliner_range, return_res_Q=True, dctr_Q=dctr_Q)
        if return_res_Q:
            return xyz[inlier_Q], res[inlier_Q]
        return xyz[inlier_Q]
    
    def xyz_is_inlier_Q(self, xyz, inliner_range, return_res_Q=False, dctr_Q=False):
        assert len(inliner_range) == 2 and inliner_range[0] <= inliner_range[1], "inliner_range must be a tuple (min, max) with min <= max"
        res = self.compute_xyz_residual(xyz)
        if dctr_Q:
            res = res - np.nanmedian(res)
        # Snap roundoff-scale boundary values so returned residuals obey the
        # same inclusive interval used for selection.
        res = np.where(np.isclose(res, inliner_range[0]), inliner_range[0], res)
        res = np.where(np.isclose(res, inliner_range[1]), inliner_range[1], res)
        inlier_Q = (res >= inliner_range[0]) & (res <= inliner_range[1])
        if return_res_Q:
            return inlier_Q, res
        return inlier_Q

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

    def uvw_to_tangent_plane(self, uvw, uvw0, 
                             align_axis=0, orthogonalized_Q=True):
        """Project points in local uvw coordinates to tangent plane at uvw0."""
        uvw = np.asarray(uvw, dtype=float)
        uvw0 = np.asarray(uvw0, dtype=float).reshape(3,)
        t_u, t_v = self.tangent_vectors(uvw0[0], uvw0[1], 
                                        align_axis=align_axis,
                                        orthogonalized_Q=orthogonalized_Q)
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


def __getattr__(name):
    if name == "PCSurface3D":
        from .pc_surface import PCSurface3D
        return PCSurface3D
    raise AttributeError(name)
