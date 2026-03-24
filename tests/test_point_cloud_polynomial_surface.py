import numpy as np
import pytest

from pyutil.geometry.point_cloud import (
    PointCloud3DPolynomialSurface,
    PointCloud3DPolynomialSurfaceFit,
)


def _rotation_z(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _make_coeffs_for_order(k, rng):
    exponents = PointCloud3DPolynomialSurface._generate_exponents(k)
    coeffs = np.zeros(len(exponents), dtype=float)
    for idx, (i, j) in enumerate(exponents):
        deg = i + j
        scale = 0.4 / (deg + 1)
        coeffs[idx] = rng.normal(scale=scale)
    coeffs[0] = rng.normal(scale=0.5)
    return coeffs, exponents


def test_parameterized_surface_defaults_and_eval():
    exponents = PointCloud3DPolynomialSurface._generate_exponents(2)
    coeffs = np.array([1.0, 2.0, -0.5, 0.3, 0.2, 0.1], dtype=float)
    surf = PointCloud3DPolynomialSurface(coeffs=coeffs, k=2)

    uvw = np.array([[1.0, 2.0, 0.0], [-1.0, 0.5, 0.0]], dtype=float)
    xyz = surf.uvw_to_xyz(uvw)
    uvw_back = surf.xyz_to_uvw(xyz)

    assert np.allclose(uvw_back, uvw)
    expected_w = (
        1.0
        + 2.0 * uvw[:, 0]
        - 0.5 * uvw[:, 1]
        + 0.3 * uvw[:, 0] ** 2
        + 0.2 * uvw[:, 0] * uvw[:, 1]
        + 0.1 * uvw[:, 1] ** 2
    )
    assert np.allclose(surf.f(uvw[:, 0], uvw[:, 1]), expected_w)
    assert surf.exponents == exponents


def test_parameterized_surface_derivatives_match_finite_difference():
    exponents = PointCloud3DPolynomialSurface._generate_exponents(3)
    coeffs = np.array([0.2, -0.1, 0.3, 0.4, -0.2, 0.05, -0.1, 0.02, 0.04, 0.01], dtype=float)
    surf = PointCloud3DPolynomialSurface(coeffs=coeffs, exponents=exponents)

    u = np.array([0.3, -0.7], dtype=float)
    v = np.array([-0.4, 0.25], dtype=float)
    eps = 1e-6

    df_du_num = (surf.f(u + eps, v) - surf.f(u - eps, v)) / (2.0 * eps)
    df_dv_num = (surf.f(u, v + eps) - surf.f(u, v - eps)) / (2.0 * eps)

    assert np.allclose(surf.df_du(u, v), df_du_num, atol=1e-5)
    assert np.allclose(surf.df_dv(u, v), df_dv_num, atol=1e-5)


def test_projection_returns_points_on_surface():
    exponents = PointCloud3DPolynomialSurface._generate_exponents(2)
    coeffs = np.array([0.0, 0.5, -0.4, 0.1, 0.05, 0.07], dtype=float)
    R = _rotation_z(np.deg2rad(20.0))
    t = np.array([2.0, -1.5, 0.7], dtype=float)
    surf = PointCloud3DPolynomialSurface(coeffs=coeffs, exponents=exponents, R=R, t=t)

    uvw_query = np.array(
        [
            [0.2, -0.3, 1.0],
            [-0.5, 0.4, -0.7],
            [0.8, 0.1, 0.2],
        ],
        dtype=float,
    )
    xyz_query = surf.uvw_to_xyz(uvw_query)

    xyz_proj, dists = surf.project_xyz_points_to_surface(xyz_query, return_dist_Q=True)
    uvw_proj = surf.xyz_to_uvw(xyz_proj)

    residuals = surf._uvw_to_residual(uvw_proj)
    assert np.max(np.abs(residuals)) < 1e-6
    assert np.all(dists >= 0.0)


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6])
def test_fit_recovers_coeffs_for_orders_1_to_6(k):
    rng = np.random.default_rng(100 + k)
    true_coeffs, exponents = _make_coeffs_for_order(k, rng)
    R = _rotation_z(np.deg2rad(7.5 * k))
    t = np.array([1.5 * k, -0.75 * k, 0.4 * k], dtype=float)

    n = 4000
    u = rng.uniform(-1.0, 1.0, n)
    v = rng.uniform(-1.0, 1.0, n)
    w_true = PointCloud3DPolynomialSurface._f(u, v, true_coeffs, exponents)
    w_noisy = w_true + rng.normal(scale=0.005, size=n)
    xyz = np.column_stack([u, v, w_noisy]) @ R.T + t

    fit = PointCloud3DPolynomialSurfaceFit(points_xyz=xyz, k=k, R=R, t=t)
    param = PointCloud3DPolynomialSurface(coeffs=fit.coeffs, k=k, R=fit.R, t=fit.centroid)

    assert fit.exponents == exponents
    assert np.allclose(fit.coeffs, true_coeffs, atol=0.05, rtol=0.2)

    uv_test = np.column_stack([
        rng.uniform(-0.9, 0.9, 200),
        rng.uniform(-0.9, 0.9, 200),
    ])
    w_expected = PointCloud3DPolynomialSurface._f(uv_test[:, 0], uv_test[:, 1], true_coeffs, exponents)
    w_fit = fit.f(uv_test[:, 0], uv_test[:, 1])
    w_param = param.f(uv_test[:, 0], uv_test[:, 1])

    assert np.allclose(w_fit, w_param)
    assert np.allclose(w_fit, w_expected, atol=0.03, rtol=0.1)
    assert fit.R2 > 0.95
    assert fit.points_uvw.shape == xyz.shape
    assert fit.point_nearest_uvw.shape == xyz.shape


def test_quadratic_convexity_and_threshold_helpers():
    exponents = PointCloud3DPolynomialSurface._generate_exponents(2)
    coeffs = np.array([0.0, 0.0, 0.0, 0.5, 0.0, 0.25], dtype=float)

    u = np.array([-1.0, 0.0, 1.0, 0.5], dtype=float)
    v = np.array([-0.5, 0.0, 0.75, 1.5], dtype=float)
    w = PointCloud3DPolynomialSurface._f(u, v, coeffs, exponents)
    xyz = np.column_stack([u, v, w])

    fit = PointCloud3DPolynomialSurfaceFit(points_xyz=xyz, k=2, R=np.eye(3), t=np.zeros(3))

    assert fit.convexity == 1
    assert np.all(fit.local_convexity(u, v) == 1)

    noisy_xyz = xyz.copy()
    noisy_xyz[:, 2] += np.array([0.0, 0.02, -0.03, 0.01])
    kept_xyz, kept_res = fit.remove_xyz_by_residual_threshold(noisy_xyz, (-0.02, 0.02), return_res_Q=True)
    assert kept_xyz.shape[0] == 3
    assert np.all((kept_res >= -0.02) & (kept_res <= 0.02))

    res_range = fit.residual_ipr_range()
    assert len(res_range) == 2


def test_higher_order_convexity_requires_local_query():
    exponents = PointCloud3DPolynomialSurface._generate_exponents(3)
    coeffs = np.array([0.0, 0.0, 0.0, 0.4, 0.0, 0.4, 0.2, 0.0, 0.0, 0.2], dtype=float)
    surf = PointCloud3DPolynomialSurface(coeffs=coeffs, exponents=exponents)

    with pytest.raises(ValueError, match="Global convexity"):
        _ = surf.convexity

    local = surf.local_convexity(np.array([0.0, -2.0]), np.array([0.0, -2.0]))
    assert np.array_equal(local, np.array([1, -1]))
