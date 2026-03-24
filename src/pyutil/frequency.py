import numpy as np
import scipy as sp
from matplotlib import pyplot as plt


def spectral_power_lomb_scargle(t_s, signal, vis_Q=False):
    result = {}
    result['num_sample'] = t_s.size
    result['total_length'] = t_s[-1] - t_s[0]
    assert t_s.size == signal.size, 'Input t and f dimension mismatch'
    is_finite_Q = np.isfinite(signal)
    t_s = t_s[is_finite_Q]
    signal = signal[is_finite_Q]
    if not np.all((t_s[1:] - t_s[:-1]) > 0): 
        s_idx = np.argsort(t_s)
        t_s = t_s[s_idx]
        signal = signal[s_idx]
    result['min_sample_interval'] = np.min(t_s[1:] - t_s[:-1])
    result['Nyquest_f'] = 1 / result['min_sample_interval'] / 2
    result['f'] = np.linspace(1/result['total_length'], result['Nyquest_f'], result['num_sample'])
    result['pwr'] = sp.signal.lombscargle(t_s, signal, result['f'] * 2 * np.pi, normalize=True, precenter=True)

    if vis_Q: 
        f = plt.figure(figsize=(8, 6))
        a = f.add_subplot()
        a.plot(result['f'], result['pwr'])
        # a.plot(f_cpt, pg_pwr)
        a.set_xscale('log')
        # a.set_yscale('log')
        a.grid()
        a.set_xlabel(f"f (Hz)")
        a.set_ylabel(f"Power")
        # a.legend()

    return result

def compute_points_diffraction_2d(points, grid_size=1024, pad_factor=2,
    box=None, return_amplitude=False):
    """
    Compute 2D diffraction intensity from N x 2 point coordinates.

    Parameters
    ----------
    points : array-like, shape (N, 2)
        Point coordinates (x, y).
    grid_size : int
        Base raster size before zero-padding.
    pad_factor : int
        Zero-padding factor for finer reciprocal-space sampling.
    box : tuple or None
        (xmin, xmax, ymin, ymax). If None, inferred from points with tiny margin.
    return_amplitude : bool
        If True, also return complex Fourier amplitude.

    Returns
    -------
    qx, qy : 1D arrays
        Reciprocal axes (cycles per unit length).
    I : 2D array
        Diffraction intensity |F(qx, qy)|^2, fftshifted.
    Fshift : 2D complex array, optional
        Complex Fourier amplitude, fftshifted.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    if pts.shape[0] == 0:
        raise ValueError("points must be non-empty")

    # Bounding box
    if box is None:
        xmin, ymin = pts.min(axis=0)
        xmax, ymax = pts.max(axis=0)
        epsx = max((xmax - xmin) * 1e-12, 1e-12)
        epsy = max((ymax - ymin) * 1e-12, 1e-12)
        xmin -= epsx
        xmax += epsx
        ymin -= epsy
        ymax += epsy
    else:
        xmin, xmax, ymin, ymax = map(float, box)

    Lx = xmax - xmin
    Ly = ymax - ymin
    if Lx <= 0 or Ly <= 0:
        raise ValueError("invalid box extents")

    # Raster size (power of 2 for FFT efficiency)
    n = int(grid_size)
    if n <= 0:
        raise ValueError("grid_size must be positive")
    nfft = int(2 ** np.ceil(np.log2(n * int(pad_factor))))

    # Map points into [0, n) indices
    x = (pts[:, 0] - xmin) / Lx * n
    y = (pts[:, 1] - ymin) / Ly * n

    # Keep points inside bounds
    mask = (x >= 0) & (x < n) & (y >= 0) & (y < n)
    x = x[mask]
    y = y[mask]

    # Bilinear deposition (reduces aliasing vs nearest-bin)
    i0 = np.floor(x).astype(np.int64)
    j0 = np.floor(y).astype(np.int64)
    fx = x - i0
    fy = y - j0

    i1 = np.clip(i0 + 1, 0, n - 1)
    j1 = np.clip(j0 + 1, 0, n - 1)

    w00 = (1.0 - fx) * (1.0 - fy)
    w10 = fx * (1.0 - fy)
    w01 = (1.0 - fx) * fy
    w11 = fx * fy

    rho = np.zeros((n, n), dtype=np.float64)
    np.add.at(rho, (j0, i0), w00)
    np.add.at(rho, (j0, i1), w10)
    np.add.at(rho, (j1, i0), w01)
    np.add.at(rho, (j1, i1), w11)

    # Zero-pad to nfft x nfft
    rho_pad = np.zeros((nfft, nfft), dtype=np.float64)
    rho_pad[:n, :n] = rho

    # FFT and intensity
    F = np.fft.fft2(rho_pad)
    I = (F.real * F.real + F.imag * F.imag)

    # Shift zero frequency to center
    Fshift = np.fft.fftshift(F)
    Ishift = np.fft.fftshift(I)

    # Reciprocal axes in cycles per coordinate unit
    dx = Lx / n
    dy = Ly / n
    qx = np.fft.fftshift(np.fft.fftfreq(nfft, d=dx))
    qy = np.fft.fftshift(np.fft.fftfreq(nfft, d=dy))

    if return_amplitude:
        return qx, qy, Ishift, Fshift, rho_pad
    return qx, qy, Ishift

def radial_average_intensity(
    I,
    qx,
    qy,
    bin_width=None,
    n_bins=200,
    qmax=None,
    exclude_q0=False,
    return_counts=False,
):
    """
    Radial average of 2D intensity I(qx, qy).

    Parameters
    ----------
    I : ndarray, shape (Ny, Nx)
        Intensity map, e.g. output of diffraction_pattern_direct_nusum.
    qx, qy : ndarray
        Either 1D axes (Nx,) and (Ny,), or 2D mesh arrays matching I.
    bin_width : float or None
        Radial bin width. If None, inferred from qmax / n_bins.
    n_bins : int
        Used only when bin_width is None.
    qmax : float or None
        Max radius to include. If None, uses max radius in grid.
    exclude_q0 : bool
        If True, excludes the exact q=0 pixel from averaging.
    return_counts : bool
        If True, also return number of samples per radial bin.

    Returns
    -------
    r_centers : ndarray, shape (Nbins,)
        Radial bin centers.
    I_radial : ndarray, shape (Nbins,)
        Mean intensity in each radial bin (NaN for empty bins).
    counts : ndarray, optional
        Sample count per bin.
    """
    I = np.asarray(I, dtype=np.float64)
    if I.ndim != 2:
        raise ValueError("I must be 2D (Ny, Nx).")

    qx = np.asarray(qx, dtype=np.float64)
    qy = np.asarray(qy, dtype=np.float64)

    # Build q-grid
    if qx.ndim == 1 and qy.ndim == 1:
        if I.shape != (qy.size, qx.size):
            raise ValueError("I.shape must be (len(qy), len(qx)) for 1D axes.")
        QX, QY = np.meshgrid(qx, qy, indexing="xy")
    elif qx.ndim == 2 and qy.ndim == 2:
        if qx.shape != I.shape or qy.shape != I.shape:
            raise ValueError("2D qx/qy must match I.shape.")
        QX, QY = qx, qy
    else:
        raise ValueError("qx/qy must be both 1D axes or both 2D meshes.")

    r = np.hypot(QX, QY).ravel()
    v = I.ravel()

    finite = np.isfinite(r) & np.isfinite(v)
    r = r[finite]
    v = v[finite]

    if exclude_q0:
        nz = r > 0
        r = r[nz]
        v = v[nz]

    if r.size == 0:
        raise ValueError("No valid samples after filtering.")

    if qmax is None:
        qmax = r.max()
    qmax = float(qmax)
    if qmax <= 0:
        raise ValueError("qmax must be positive.")

    if bin_width is None:
        if n_bins <= 0:
            raise ValueError("n_bins must be positive.")
        bin_width = qmax / float(n_bins)
    if bin_width <= 0:
        raise ValueError("bin_width must be positive.")

    nbins = int(np.ceil(qmax / bin_width))

    # Uniform-bin indexing (faster than digitize for this case)
    b = np.floor(r / bin_width).astype(np.int64)
    keep = (b >= 0) & (b < nbins)
    b = b[keep]
    v = v[keep]

    sums = np.bincount(b, weights=v, minlength=nbins)
    counts = np.bincount(b, minlength=nbins)

    I_radial = np.full(nbins, np.nan, dtype=np.float64)
    good = counts > 0
    I_radial[good] = sums[good] / counts[good]

    r_centers = (np.arange(nbins, dtype=np.float64) + 0.5) * bin_width

    if return_counts:
        return r_centers, I_radial, counts
    return r_centers, I_radial
    
    
    
