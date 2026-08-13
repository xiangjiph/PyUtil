import numpy as np

import pyutil.stat as pystat


def test_rebin_histogram_accepts_nonuniform_centers_and_preserves_mass():
    bin_val = np.asarray([0.0, 1.0, 3.0])
    count = np.asarray([1.0, 3.0, 4.0])
    new_bin_val = np.asarray([0.0, 1.8, 4.0])

    rebinned = pystat.rebin_histogram(bin_val, count, new_bin_val, even_tol=0.0)

    np.testing.assert_allclose(rebinned, [1.8, 4.0, 2.2])
    np.testing.assert_allclose(rebinned.sum(), np.sum(count))


def test_rebin_histogram_converts_nonuniform_pdf_to_nonuniform_pdf():
    bin_val = np.asarray([0.0, 1.0, 3.0])
    pdf = np.asarray([1.0, 2.0, 2.0])
    new_bin_val = np.asarray([0.0, 1.8, 4.0])

    rebinned_pdf = pystat.rebin_histogram(
        bin_val, pdf, new_bin_val, input_is_pdf=True, output_pdf=True, even_tol=0.0
    )

    np.testing.assert_allclose(rebinned_pdf, [1.0, 2.0, 1.0])


def test_rebin_histogram_rebins_known_nonuniform_edges_exactly():
    source_edges = np.asarray([0.5, 1.5, 3.5])
    count = np.asarray([2.0, 2.0])
    target_edges = np.asarray([0.5, 1.5, 2.5, 3.5])

    rebinned = pystat.rebin_histogram(
        source_edges, count, target_edges, bin_val_is_edge_Q=True, new_bin_val_is_edge_Q=True
    )

    np.testing.assert_allclose(rebinned, [2.0, 1.0, 1.0])
    np.testing.assert_allclose(rebinned.sum(), count.sum())
