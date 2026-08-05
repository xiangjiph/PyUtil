import numpy as np
import pytest

import pyutil.stat as pystat


def test_compute_radial_distribution_density_normalizes_by_annulus_area():
    d = np.asarray([0.5, 1.2, 1.4, 1.8])

    stat = pystat.compute_radial_distribution_density(d, bin_edge=np.asarray([0.0, 1.0, 2.0]))

    np.testing.assert_array_equal(stat["hist_count"], [1, 3])
    np.testing.assert_allclose(stat["probability"], [0.25, 0.75])
    np.testing.assert_allclose(stat["bin_area"], [np.pi, 3 * np.pi])
    np.testing.assert_allclose(stat["pdf"], [0.25 / np.pi, 0.25 / np.pi])
    np.testing.assert_allclose(stat["radial_pdf"], [0.5, 0.5])
    assert stat["radial_mean"] == pytest.approx(1.0)
    assert stat["radial_median"] == pytest.approx(1.0)
    assert stat["radial_std"] == pytest.approx(0.5)
    assert stat["num_data"] == 4
    assert stat["mean"] == pytest.approx(np.mean(d))


def test_compute_radial_distribution_density_rejects_negative_distances():
    with pytest.raises(ValueError, match="non-negative"):
        pystat.compute_radial_distribution_density([0, -1, 2])
