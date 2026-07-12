# Point-cloud refactor notes

## Implemented split

- `geometry/point_cloud/__init__.py` is the compatibility facade.
- `surface.py` owns `PolySurface3D` and `PCSurface3D`.
- `legacy_surface.py` owns `PointCloud3DSurfaceFit`.
- `group.py` owns grouped KD-tree lookup.
- `downsample.py` owns regular-grid aggregation.
- `outlier.py` owns principal-axis outlier selection.
- All tests live under the single root-level `tests/` folder.

## Geodesic methods

- `PolySurface3D.geodesic_distance` optimizes one path in the polynomial
  parameter domain and integrates its surface arc length.
- `PolySurface3D.pairwise_geodesic_distance` targets collections of roughly
  1,000 points. It integrates a sparse UV-neighbor graph in vectorized batches
  and computes all-pairs shortest paths rather than running a nonlinear solve
  independently for every pair.
- The local frame rotation is not applied while measuring tangent speed:
  `R` is orthonormal, so rotating a vector cannot change its Euclidean norm.

## Duplication and risks

- `PointCloud3DSurfaceFit` duplicates polynomial evaluation, first
  derivatives, normals, tangent vectors, coordinate transforms, residual
  filtering, nearest-surface projection, plotting, and between-surface tests
  already generalized by `PolySurface3D` and `PCSurface3D`.
- `PointCloud3DSurfaceFit._uv_to_polynomial_array` and
  `PointCloud3DSurfaceFit.uv_to_polynomial_array` are duplicate quadratic
  basis builders in the same class.
- The legacy fitter assigned to a read-only `points_uvw` property during
  construction. The split removes that assignment and uses the property as
  intended.
- The existing polynomial-surface tests import
  `PointCloud3DPolynomialSurface` and `PointCloud3DPolynomialSurfaceFit`, but
  the old module did not define them. The facade restores these aliases.
- `grid_centroids` and `downsample_points_by_averaging` overlap with
  `GridDownsamplerND`, but they are small compatibility entry points rather
  than independent implementations.

## Proposed follow-up

1. Add parity tests between the quadratic legacy fitter and `PCSurface3D` for
   fitting, derivatives, projection, residual thresholds, and plotting data.
2. Move the static tangent-plane use in `geometry/lattice.py` to the
   `PolySurface3D` instance API or a shared polynomial-surface helper.
3. Deprecate `PointCloud3DSurfaceFit` after downstream callers migrate, then
   remove its duplicate numerical and visualization implementation.
4. Keep the two downsampling wrapper functions until downstream imports are
   inventoried; deprecate only if the class API can replace them without
   changing return shapes.
5. Split plotting from `surface.py` only if that module grows
   further; the numerical ownership is currently coherent.
