import numpy as np
import pytest

import pyutil.geometry.point_cloud as point_cloud
from pyutil.geometry.point_cloud import PolySurface3D


def _surface_point(surface, u, v):
    return np.array([u, v, float(surface.f(u, v))], dtype=float)


def test_polynomial_surface_aliases_remain_importable():
    assert point_cloud.PointCloud3DPolynomialSurface is point_cloud.PolySurface3D
    assert point_cloud.PointCloud3DPolynomialSurfaceFit is point_cloud.PCSurface3D


def test_plane_geodesic_matches_endpoint_euclidean_distance():
    surface = PolySurface3D(coeffs=[1.0, 0.5, -0.25], k=1)
    uvw1 = _surface_point(surface, -0.8, 0.3)
    uvw2 = _surface_point(surface, 1.1, -0.6)

    distance, path = surface.geodesic_distance(
        uvw1, uvw2, num_segments=8, return_path_Q=True
    )
    expected = np.linalg.norm(
        surface.uvw_to_xyz(uvw2)[0] - surface.uvw_to_xyz(uvw1)[0]
    )

    assert distance == pytest.approx(expected, rel=1e-8, abs=1e-10)
    assert np.allclose(path[[0, -1]], [uvw1, uvw2])
    assert np.allclose(path[:, 2], surface.f(path[:, 0], path[:, 1]))


def test_radial_paraboloid_geodesic_matches_analytic_meridian_length():
    curvature = 0.4
    radius = 1.25
    surface = PolySurface3D(
        coeffs=[0.0, 0.0, 0.0, curvature, 0.0, curvature],
        k=2,
    )
    uvw1 = _surface_point(surface, 0.0, 0.0)
    uvw2 = _surface_point(surface, radius, 0.0)

    distance, path = surface.geodesic_distance(
        uvw1, uvw2, num_segments=12, quadrature_order=10,
        return_path_Q=True,
    )
    slope = 2.0 * curvature * radius
    expected = (
        0.5 * radius * np.sqrt(1.0 + slope * slope)
        + np.arcsinh(slope) / (4.0 * curvature)
    )

    assert distance == pytest.approx(expected, rel=2e-7)
    assert np.max(np.abs(path[:, 1])) < 5e-7


def test_geodesic_rejects_endpoint_off_surface():
    surface = PolySurface3D(coeffs=[0.0, 0.0, 0.0], k=1)
    uvw1 = np.array([0.0, 0.0, 0.1])
    uvw2 = np.array([1.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="uvw1 does not lie"):
        surface.geodesic_distance(uvw1, uvw2)


def test_pairwise_geodesic_distance_is_exact_on_plane_with_dense_graph():
    surface = PolySurface3D(coeffs=[0.2, 0.4, -0.3], k=1)
    uv = np.array([
        [-1.0, -0.5],
        [-0.2, 0.8],
        [0.4, -0.7],
        [1.1, 0.2],
        [0.7, 1.0],
    ])
    uvw = np.column_stack([uv, surface.f(uv[:, 0], uv[:, 1])])

    distances = surface.pairwise_geodesic_distance(
        uvw, num_neighbors=uvw.shape[0] - 1, quadrature_order=4)
    xyz = surface.uvw_to_xyz(uvw)
    expected = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)

    assert np.allclose(distances, expected)


def test_pairwise_geodesic_distance_sparse_graph_is_finite_and_symmetric():
    surface = PolySurface3D(
        coeffs=[0.0, 0.0, 0.0, 0.25, 0.0, 0.15], k=2)
    u, v = np.meshgrid(np.linspace(-1.0, 1.0, 8), np.linspace(-1.0, 1.0, 8))
    uvw = np.column_stack([
        u.ravel(), v.ravel(), surface.f(u.ravel(), v.ravel())
    ])

    distances = surface.pairwise_geodesic_distance(uvw, num_neighbors=6)

    assert distances.shape == (uvw.shape[0], uvw.shape[0])
    assert np.all(np.isfinite(distances))
    assert np.allclose(distances, distances.T)
    assert np.allclose(np.diag(distances), 0.0)


def test_pairwise_geodesic_distance_rejects_off_surface_points():
    surface = PolySurface3D(coeffs=[0.0, 0.0, 0.0], k=1)
    uvw = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.1]])

    with pytest.raises(ValueError, match="all uvw points"):
        surface.pairwise_geodesic_distance(uvw)
