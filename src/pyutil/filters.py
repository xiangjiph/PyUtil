import scipy.ndimage as ndi
import dask.array as da 
import numpy as np
import pandas as pd
# import cv2

def medfilt3(block, size=3, chunks=None):
    if isinstance(size, int):
        overlap = size // 2
    elif isinstance(size, tuple):
        overlap = [i // 2 for i in tuple]
    if chunks is None:
        chunks = (64, 64, 64)
    block = da.from_array(block, chunks=chunks)
    output = block.map_overlap(lambda block: ndi.median_filter(block, size=size), \
                               depth=overlap)
    return output.compute()

def dog(data, sig1, sig2):
    # kernel size = 2r + 1 = 8 sigma + 1
    data_1 = data
    if sig1 is not None: 
        sig1 = np.array(sig1) if not isinstance(sig1, np.ndarray) else sig1
        if np.all(sig1 > 0):
            data_1 = ndi.gaussian_filter(data, sig1)

    data_2 = data
    if sig2 is not None: 
        sig2 = np.array(sig2) if not isinstance(sig2, np.ndarray) else sig2
        if np.all(sig2 > 0):
            data_2 = ndi.gaussian_filter(data, sig2)

    return (data_1 - data_2).astype(data.dtype)

# def imgaussfilt2(data, wd_sz):
#     data_dim = data.ndim
#     if data_dim == 2:
#         data = data[None, :, :]
#     for i in range(data.shape[0]):
#         data[i, :, :] = cv2.GaussianBlur(data[i, :, :], wd_sz, 0)
#     if data_dim == 2:
#         data = np.squeeze(data)
#     return data

def central_difference_1d(x): 
    x = np.asarray(x)
    assert x.ndims == 1, "x should have dimension 1"
    if x.size <= 1: 
        return np.full(x.size, np.nan)
    elif x.size == 2: 
        return np.full(x.size, x[1] - x[0])
    else: 
        return np.concatenate(([x[1] - x[0]], (x[2:] - x[:-2]) / 2, [x[-1] - x[-2]]))
    
def central_difference_2d(X, axis=0):
    if axis >= len(X.shape):
        raise ValueError("Axis out of bounds for array dimension")
    
    dX = np.full(X.shape, np.nan)
    if X.shape[axis] >= 2: 
        if axis == 0:
            dX[1:-1, :] = (X[2:, :] - X[:-2, :]) / 2
            dX[0, :] = (X[1, :] - X[0, :])
            dX[-1, :] = (X[-1, :] - X[-2, :])
        elif axis == 1:
            dX[:, 1:-1] = (X[:, 2:] - X[:, :-2]) / 2
            dX[:, 0] = (X[:, 1] - X[:, 0])
            dX[:, -1] = (X[:, -1] - X[:, -2])
    
    return dX

def moving_average_1d(x, window_size, min_periods=None):
    if min_periods is None:
        min_periods = window_size
    x = pd.Series(x)
    moving_avg = x.rolling(window=window_size, min_periods=min_periods,
                            center=True).mean().to_numpy()
    return moving_avg