import numpy as np
from scipy import ndimage

from . import stat


def correct_global_lineshift(im, shift):
    if shift != 0:
        if im.ndim == 2:
            im[1::2, :] = np.roll(im[1::2, :], shift, axis=1)
        elif im.ndim == 3:
            for i in range(im.shape[0]):
                im[i, 1::2, :] = np.roll(im[i, 1::2, :], shift, axis=1)
    return im

def compute_global_column_shift(im, search_radius=30):
    assert type(
        im).__module__ == np.__name__, 'The first input should be a numpy array'
    assert im.ndim == 2, 'The first input should be of dimension 2'
    # Rescale image
    if np.issubdtype(im.dtype, np.integer):
        im = (im / np.iinfo(im.dtype).max).astype(np.float16)
    else:
        im = ((im - im.mean()) / (im.max() - im.min())).astype(np.float16)
    im_1 = im[::2]
    im_2 = im[1::2]
    search_list = np.arange(-search_radius, search_radius + 1)
    corr_list = np.zeros(search_list.shape)
    for i, shift in enumerate(search_list):
        corr_list[i] = np.mean(im_1 * np.roll(im_2, shift, axis=1))
        # If tier, use the last one
    max_corr_idx = search_list[np.argmax(corr_list)]
    # Erhan's implementation uses interpolation
    # itp_x = np.linspace(search_list[0], search_list[-1], num=1000, endpoint=True)
    # itp_f = interp1d(search_list, corr_list, kind='cubic')
    # max_corr_idx_f = itp_x[np.argmax(itp_f(itp_x))]
    # max_corr_idx = int(np.round(max_corr_idx_f))
    # return max_corr_idx, [search_list, corr_list]
    return max_corr_idx

#region 2P scanner desynchronization
def compute_single_image_y_gradient_peak(im):
    im_grad = np.diff(im, axis=0)
    abs_avg_grad = np.abs(im_grad.mean(axis=1))
    grad_prctile = np.percentile(abs_avg_grad, [25, 50, 75])
    dev2int = (abs_avg_grad - grad_prctile[1]) / \
        (grad_prctile[2] - grad_prctile[0])
    return dev2int.max, dev2int.argmax

def row_shift_detection(tmp_im, min_th=0.05, sm_wd_sz=3):
    
    row_mean = np.mean(tmp_im, axis=1, keepdims=True)
    # This need to be updated later. We might need to change the acquisition format 
    row_mean_n = row_mean / np.iinfo(np.int16).max 
    row2_mean = np.mean(tmp_im.astype(np.double) ** 2, axis=1, keepdims=True)
    row_std = np.sqrt(row2_mean - row_mean**2)
    row_demean = tmp_im.astype(np.double) - row_mean
    # Compute the row-adjacent correlation value 
    row_adj_corr = np.mean(row_demean[0:-1, :] * row_demean[1:, :],
                        axis=1, keepdims=True) / (row_std[0:-1, :] * row_std[1:, :])
    row_adj_corr_ad = np.abs(np.diff(row_adj_corr, axis=0)) 
    # Mean-weighted adjacent correlation 
    adj_corr_ad_mw = row_adj_corr_ad * row_mean_n[1:-1]
    # Outlier detection 
    p25, p50, p75 = np.percentile(adj_corr_ad_mw, [25, 50, 75])
    pth = max(p50 + 3 * (p75 - p25), min_th)
    is_outlier = adj_corr_ad_mw > pth
    is_outlier = ndimage.binary_closing(is_outlier.flatten(), np.ones((sm_wd_sz, )))
    # Todo: find the most prominant one 
    # Analyze the distance to the button of the frame 
    if np.any(is_outlier):
        outlier_ep_Q = np.concatenate(([False], is_outlier, [False]), axis=None)
        outlier_ep_Q = np.diff(outlier_ep_Q, )
        outlier_ep_idx = np.nonzero(outlier_ep_Q)
    else:
        outlier_ep_idx = None
    
    return outlier_ep_idx

def compute_row_stat(im, visQ=False):
    im = np.double(im)
    result = {}
    result['row_mean'] = np.mean(im, axis=1)
    row2_mean = np.mean(im ** 2, axis=1)
    std = np.sqrt(row2_mean - result['row_mean'] ** 2)
    std[std == 0] = np.nan
    std = stat.fill_edge_nan_1d(std, method='nearest')
    std = stat.fill_internal_nan(std, method='linear')
    result['std'] = std
    im_dm = im - result['row_mean'].reshape((-1, 1))
    
    # Adjacent row correlation 
    adj_corr = np.mean(im_dm[0:-1, :] * im_dm[1:, :], axis=1) / \
        (result['std'][0:-1] * result['std'][1:])
    adj_corr = np.concatenate(([adj_corr[0]], np.maximum(adj_corr[1:], adj_corr[:-1]), [adj_corr[-1]]))
    # adj_corr = np.concatenate(([adj_corr[0]], (adj_corr[1:] + adj_corr[:-1]) / 2, [adj_corr[-1]]))
    result['adj_corr'] = adj_corr
    result['adj_corr_d'] = np.gradient(adj_corr)
    result['adj_corr_da'] = np.abs(result['adj_corr_d'])
    
    # Average adjacent row intensity gradient absolute difference 
    im_grad = np.abs(im[1:, :] - im[0:-1, :])
    im_grad = np.concatenate((im_grad[0].reshape((1, -1)), (im_grad[0:-1, :] + im_grad[1:, ])/2, im_grad[-1].reshape((1, -1))), axis=0)
    im_grad2 = np.abs(im_grad[1:, :] - im_grad[0:-1, :])
    im_grad2 = np.concatenate((im_grad2[0].reshape((1, -1)), (im_grad2[0:-1, :] + im_grad2[1:, ])/2, im_grad2[-1].reshape((1, -1))), axis=0)

    # im_grad = np.concatenate((im_grad[0].reshape((1, -1)), np.maximum(im_grad[0:-1, :], im_grad[1:, ]), im_grad[-1].reshape((1, -1))), axis=0)
    result['row_abs_diff'] = np.mean(im_grad, axis=1)
    result['row_abs_diff2'] = np.mean(im_grad2, axis=1)
    # result['row_abs_diff'] = np.mean(np.abs(np.gradient(im, axis=0)), axis=1)

    return result
#endregion