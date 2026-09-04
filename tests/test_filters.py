import numpy as np
import pytest

from pyutil.filters import (
    find_resolvable_peaks_in_irregular_sampled_data,
    gaussian_distributions_resolvable,
)


def test_equal_width_gaussians_require_more_than_two_standard_deviations():
    assert not gaussian_distributions_resolvable(2.0, 1.0, 1.0)
    assert gaussian_distributions_resolvable(np.nextafter(2.0, np.inf), 1.0, 1.0)


def test_unequal_width_gaussian_resolution_is_symmetric_and_scale_invariant():
    spacing = np.array([1.625, 1.626])

    result = gaussian_distributions_resolvable(spacing, 1.0, 0.5)
    swapped = gaussian_distributions_resolvable(spacing, 0.5, 1.0)
    scaled = gaussian_distributions_resolvable(spacing * 10, 10.0, 5.0)

    assert np.array_equal(result, [False, True])
    assert np.array_equal(swapped, result)
    assert np.array_equal(scaled, result)


def test_peak_finder_suppresses_lower_unresolvable_peak_and_preserves_indices():
    x = np.array([7.0, 2.5, 0.0, 6.0, 1.5, 1.0, 3.0])
    y = np.array([0.0, 4.0, 0.0, 3.0, 0.0, 5.0, 0.0])

    peak_x, peak_y, peak_idx = find_resolvable_peaks_in_irregular_sampled_data(
        x, y, std=0.8)

    assert np.array_equal(peak_x, [1.0, 6.0])
    assert np.array_equal(peak_y, [5.0, 3.0])
    assert np.array_equal(peak_idx, [5, 3])


def test_peak_finder_fills_only_resolvable_gaps():
    x = np.array([0.0, 1.0, 5.0, 6.0])
    y = np.array([0.0, 2.0, 2.0, 0.0])

    no_gap = find_resolvable_peaks_in_irregular_sampled_data(
        x, y, std=0.5, fill_gap_Q=False)
    with_gap = find_resolvable_peaks_in_irregular_sampled_data(
        x, y, std=0.5, fill_gap_Q=True)

    assert np.array_equal(no_gap[2], [1])
    assert np.array_equal(with_gap[2], [1, 2])


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"x": [0, 1], "y": [1], "std": 1}, "equal length"),
        ({"x": [0, 0], "y": [1, 0], "std": 1}, "unique"),
        ({"x": [0, 1], "y": [1, 0], "std": [1, 0]}, "positive"),
    ],
)
def test_peak_finder_rejects_invalid_inputs(kwargs, message):
    with pytest.raises(ValueError, match=message):
        find_resolvable_peaks_in_irregular_sampled_data(**kwargs)
