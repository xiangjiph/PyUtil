import numpy as np

from pyutil.geometry.point_cloud import PCSurface3D, PolySurface3D
from pyutil.geometry.point_cloud.surface import PCSurface3D as SurfaceModulePCSurface3D


def test_pc_surface_direct_module_import_is_preserved():
    assert SurfaceModulePCSurface3D is PCSurface3D


def test_sample_normal_line_xyz_uses_world_surface_normal():
    theta = np.deg2rad(30)
    R = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0],
                  [-np.sin(theta), 0, np.cos(theta)]])
    surface = PolySurface3D([0, 0, 0], k=1, R=R, t=np.array([2, 3, 4]))
    r0 = surface.uvw_to_xyz([0, 0, 0])[0]

    result = surface.sample_normal_line_xyz(r0, h=2, dh=1)

    expected_n = np.array([0, 0, 1]) @ R.T
    np.testing.assert_allclose(result["n"], expected_n)
    np.testing.assert_allclose(result["X"], r0 + np.arange(-2, 3)[:, None] * expected_n)


def test_pc_surface_queries_and_summarizes_normal_spread():
    points = np.asarray([[x, y, z] for x in [-1, 0, 1]
                         for y in [-1, 0, 1] for z in [-0.5, 0, 0.5]])
    surface = PCSurface3D(points, k=1, R=np.eye(3), t=np.zeros(3))

    query = surface.query_points_near_normal([0, 0, 0], h=0.6, l=0.2)
    spread = surface.compute_normal_spread([0, 0, 0], h=0.6, l=0.2)
    table = surface.compute_surface_normal_spread(grid_spacing=1, h=0.6, l=0.2)

    assert surface.points_kdtree is surface.points_kdtree
    np.testing.assert_allclose(query["points_xyz"], [[0, 0, -0.5], [0, 0, 0], [0, 0, 0.5]])
    np.testing.assert_allclose(np.sort(spread["normal_offset"]), [-0.5, 0, 0.5])
    assert spread["normal_mean"] == 0
    assert spread["normal_median"] == 0
    np.testing.assert_allclose([spread["normal_p25"], spread["normal_p75"]], [-0.25, 0.25])
    assert table.shape[0] == 9
    assert np.all(table["num_points"] == 3)


def test_pc_surface_sampling_returns_matching_coordinate_frames():
    points = np.asarray([[x, y, 0] for x in [-1, 0, 1] for y in [-1, 0, 1]])
    surface = PCSurface3D(points, k=1, R=np.eye(3), t=np.zeros(3))

    uvw = surface.sample_surface_xyz(1, ret_coord="uvw")
    xyz = surface.sample_surface_xyz(1, ret_coord="xyz")

    np.testing.assert_allclose(xyz, surface.uvw_to_xyz(uvw))
