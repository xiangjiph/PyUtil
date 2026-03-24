import numpy as np
import pytest

from pyutil.geometry.point_cloud import (
    GridDownsamplerND,
    downsample_points_by_averaging,
    grid_centroids,
)


def test_nd_downsample_scalar_cell_size_centroids_and_counts():
    pts = np.array(
        [
            [0.1, 0.1, 0.2],
            [0.2, 0.1, 0.3],
            [1.1, 1.2, 1.1],
            [1.4, 1.3, 1.15],
            [1.6, 1.3, 1.05],
        ],
        dtype=float,
    )

    ds = GridDownsamplerND(points=pts, cell_size=1.0)

    assert ds.centroids.shape == (2, 3)
    assert np.array_equal(ds.counts, np.array([2, 3]))

    expected = np.array(
        [
            pts[:2].mean(axis=0),
            pts[2:].mean(axis=0),
        ]
    )
    assert np.allclose(ds.centroids, expected)


def test_nd_downsample_vector_cell_size_in_4d_with_optional_outputs():
    pts = np.array(
        [
            [0.10, 0.10, 0.10, 0.10],
            [0.40, 0.20, 0.10, 0.10],
            [0.90, 0.40, 0.20, 0.10],
            [1.10, 0.60, 0.60, 0.30],
        ],
        dtype=float,
    )

    ds = GridDownsamplerND(points=pts, cell_size=np.array([0.5, 0.5, 0.5, 0.5]))
    out = ds.downsample(
        return_counts=True,
        return_bboxes=True,
        return_cell_ids=True,
        return_subscripts=True,
    )

    assert set(out.keys()) == {"centroids", "counts", "bboxes", "cell_ids", "subscripts"}
    assert out["centroids"].shape[1] == 4
    assert out["bboxes"].shape[1] == 8
    assert out["cell_ids"].ndim == 1
    assert out["subscripts"].shape == (out["centroids"].shape[0], 4)
    assert np.sum(out["counts"]) == pts.shape[0]


def test_cell_bboxes_match_subscripts_and_cell_size():
    rng = np.random.default_rng(0)
    pts = rng.uniform(0.0, 3.0, size=(100, 3))
    ds = GridDownsamplerND(points=pts, cell_size=[0.5, 1.0, 1.5])

    bboxes = ds.get_cell_bboxes()
    mins = bboxes[:, :3]
    maxs = bboxes[:, 3:]

    expected_mins = ds.origin + ds.cell_subscripts * ds.cell_size_vec
    expected_maxs = expected_mins + ds.cell_size_vec

    assert np.allclose(mins, expected_mins)
    assert np.allclose(maxs, expected_maxs)


def test_nearest_points_to_centroids_and_matched_mask():
    pts = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.0],
            [1.0, 1.0],
            [1.3, 1.1],
            [2.2, 2.2],
        ],
        dtype=float,
    )

    ds = GridDownsamplerND(points=pts, cell_size=1.0)
    nearest_pts, nearest_idx, nearest_dist, matched = ds.nearest_points_to_centroids(
        return_dist_Q=True, return_mask_Q=True
    )

    assert nearest_pts.shape == ds.centroids.shape
    assert nearest_idx.shape == (ds.centroids.shape[0],)
    assert nearest_dist.shape == (ds.centroids.shape[0],)
    assert matched.shape == (pts.shape[0],)
    assert np.array_equal(np.sort(nearest_idx), np.flatnonzero(matched))

    for i, idx in enumerate(nearest_idx):
        cell_sub = ds.cell_subscripts[i]
        same_cell = np.all(ds.subscripts_in == cell_sub, axis=1)
        candidate_idx = ds.input_indices[same_cell]
        cand = pts[candidate_idx]
        d = np.linalg.norm(cand - ds.centroids[i], axis=1)
        assert np.isclose(np.min(d), nearest_dist[i])
        assert np.allclose(pts[idx], nearest_pts[i])


def test_points_outside_explicit_grid_are_excluded():
    pts = np.array(
        [
            [-0.1, 0.0],
            [0.1, 0.2],
            [0.7, 0.7],
            [1.2, 1.2],
        ],
        dtype=float,
    )

    ds = GridDownsamplerND(
        points=pts,
        cell_size=0.5,
        origin=np.array([0.0, 0.0]),
        extent=np.array([1.0, 1.0]),
    )

    assert np.sum(ds.input_mask) == 2
    assert ds.points_in.shape[0] == 2
    assert np.sum(ds.counts) == 2


def test_grid_centroids_backward_compatibility_for_3d_case():
    pts = np.array(
        [
            [0.1, 0.1, 0.1],
            [0.3, 0.2, 0.1],
            [1.2, 1.1, 1.1],
            [1.5, 1.3, 1.4],
        ],
        dtype=float,
    )
    centroids, ijk, counts = grid_centroids(pts, (2, 2, 2), return_counts=True)

    assert centroids.shape == (2, 3)
    assert ijk.shape == (2, 3)
    assert np.array_equal(counts, np.array([2, 2]))


def test_downsample_points_by_averaging_matches_class_default_api():
    rng = np.random.default_rng(42)
    pts = rng.normal(size=(200, 5))
    cell_size = np.array([0.5, 0.7, 1.0, 0.6, 0.8])

    centroids_fn = downsample_points_by_averaging(pts, cell_size)
    centroids_cls = GridDownsamplerND(points=pts, cell_size=cell_size).downsample()

    assert np.allclose(centroids_fn, centroids_cls)


def test_invalid_inputs_raise_useful_errors():
    with pytest.raises(ValueError, match="strictly positive"):
        GridDownsamplerND(points=np.array([[0.0], [1.0]]), cell_size=0.0)

    with pytest.raises(ValueError, match=r"shape \(N, D\)"):
        GridDownsamplerND(points=np.array([1.0, 2.0, 3.0]), cell_size=1.0)

    pts = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float)
    with pytest.raises(ValueError, match="origin must be shape"):
        GridDownsamplerND(points=pts, cell_size=1.0, origin=np.array([0.0, 0.0, 0.0]))

    with pytest.raises(ValueError, match="grid_shape must be shape"):
        GridDownsamplerND(points=pts, cell_size=1.0, grid_shape=np.array([2, 2, 2]))


# ---------------------------------------------------------------------------
# Feature tests
# ---------------------------------------------------------------------------

def test_feature_stats_mean_matches_manual():
    """get_cell_feature_stats(np.mean) should equal manually grouped means."""
    pts = np.array(
        [
            [0.1, 0.1],  # cell (0,0)
            [0.4, 0.3],  # cell (0,0)
            [1.1, 1.2],  # cell (1,1)
            [1.6, 1.8],  # cell (1,1)
            [2.2, 0.2],  # cell (2,0)  — avoids exact-integer extent boundary
        ],
        dtype=float,
    )
    feats = np.array(
        [
            [1.0, 10.0],
            [3.0, 20.0],
            [5.0, 30.0],
            [7.0, 40.0],
            [9.0, 50.0],
        ],
        dtype=float,
    )

    ds = GridDownsamplerND(points=pts, cell_size=1.0, features=feats)
    stats = ds.get_cell_feature_stats(np.mean)

    assert stats.shape == (3, 2)

    # Verify each cell's mean is computed correctly by matching cell subscripts
    for i, sub in enumerate(ds.cell_subscripts):
        cell_mask = np.all(ds.subscripts_in == sub, axis=1)
        expected_mean = feats[ds.input_mask][cell_mask].mean(axis=0)
        assert np.allclose(stats[i], expected_mean), (
            f"Mismatch at cell {sub}: got {stats[i]}, expected {expected_mean}"
        )


def test_feature_1d_vector_treated_as_single_feature():
    """A 1D feature vector should be stored as (N,1) and stats return (M,1)."""
    pts = np.array([[0.1, 0.1], [0.4, 0.3], [1.1, 1.2]], dtype=float)
    feat_1d = np.array([2.0, 4.0, 6.0])

    ds = GridDownsamplerND(points=pts, cell_size=1.0, features=feat_1d)

    assert ds.features.shape == (3, 1)
    assert ds.n_features == 1

    stats = ds.get_cell_feature_stats(np.mean)
    assert stats.shape == (2, 1)
    assert np.isclose(stats[0, 0], np.mean([2.0, 4.0]))  # cell (0,0)
    assert np.isclose(stats[1, 0], 6.0)                  # cell (1,1)


def test_feature_stats_multiple_aggregation_functions():
    """np.std, np.min, np.max should all produce (M, d) outputs."""
    rng = np.random.default_rng(7)
    pts = rng.uniform(0.0, 5.0, size=(200, 3))
    feats = rng.standard_normal(size=(200, 4))

    ds = GridDownsamplerND(points=pts, cell_size=1.0, features=feats)

    for fn in (np.std, np.min, np.max, np.median):
        stats = ds.get_cell_feature_stats(fn)
        assert stats.shape == (ds.centroids.shape[0], 4), (
            f"Shape mismatch for {fn.__name__}: {stats.shape}"
        )

    # Verify np.min ≤ np.mean ≤ np.max element-wise
    mean_s = ds.get_cell_feature_stats(np.mean)
    min_s  = ds.get_cell_feature_stats(np.min)
    max_s  = ds.get_cell_feature_stats(np.max)
    assert np.all(min_s <= mean_s + 1e-12)
    assert np.all(mean_s <= max_s + 1e-12)


def test_feature_stats_out_of_bounds_points_excluded():
    """Features for out-of-bounds points must not influence cell statistics."""
    pts = np.array(
        [
            [0.1, 0.1],   # in bounds
            [0.6, 0.2],   # in bounds
            [2.0, 2.0],   # OUT of bounds (origin=[0,0], extent=[1,1])
        ],
        dtype=float,
    )
    feats = np.array([[1.0], [3.0], [999.0]], dtype=float)

    ds = GridDownsamplerND(
        points=pts,
        cell_size=1.0,
        origin=np.array([0.0, 0.0]),
        extent=np.array([1.0, 1.0]),
        features=feats,
    )

    assert ds.points_in.shape[0] == 2
    assert ds.features_in.shape == (2, 1)

    stats = ds.get_cell_feature_stats(np.mean)
    # Only one cell is occupied; its mean should be (1+3)/2=2, not influenced by 999
    assert stats.shape == (1, 1)
    assert np.isclose(stats[0, 0], 2.0)


def test_feature_stats_empty_when_all_points_out_of_bounds():
    """When all points are clipped out, feature stats returns (0, d)."""
    pts = np.array([[5.0, 5.0], [6.0, 6.0]], dtype=float)
    feats = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)

    ds = GridDownsamplerND(
        points=pts,
        cell_size=1.0,
        origin=np.array([0.0, 0.0]),
        extent=np.array([1.0, 1.0]),
        features=feats,
    )

    assert ds.points_in.shape[0] == 0
    stats = ds.get_cell_feature_stats(np.mean)
    assert stats.shape == (0, 2)


def test_feature_stats_raises_when_no_features():
    """Calling get_cell_feature_stats without features should raise ValueError."""
    pts = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float)
    ds = GridDownsamplerND(points=pts, cell_size=0.5)

    with pytest.raises(ValueError, match="No features provided"):
        ds.get_cell_feature_stats(np.mean)


def test_feature_shape_mismatch_raises_error():
    """features with wrong N should raise a descriptive ValueError."""
    pts = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=float)
    bad_feats = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)  # N=2, not 3

    with pytest.raises(ValueError, match="features.shape\\[0\\]"):
        GridDownsamplerND(points=pts, cell_size=1.0, features=bad_feats)


def test_feature_stats_order_matches_centroids():
    """Row i of get_cell_feature_stats output aligns with centroids[i]."""
    rng = np.random.default_rng(99)
    pts = rng.uniform(0.0, 4.0, size=(500, 2))
    # Feature = x-coordinate of each point (so cell mean ≈ cell centroid x)
    feats = pts[:, [0]]

    ds = GridDownsamplerND(points=pts, cell_size=1.0, features=feats)
    feat_means = ds.get_cell_feature_stats(np.mean)  # (M, 1)

    # The feature mean (x-coord) should be very close to centroid x
    assert np.allclose(feat_means[:, 0], ds.centroids[:, 0])
