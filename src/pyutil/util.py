import numpy as np
import pandas as pd


def mask_data_in_dict(data, mask, mask_priority=None):
    masked_data = {}
    assert isinstance(data, dict), "data should be a dictionary"
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            match_dim = np.nonzero(np.array(v.shape) == mask.size)[0]
            indexer = [slice(None)] * v.ndim
            if match_dim.size == 1:
                match_dim = int(match_dim)
            else:
                if mask_priority is None: 
                    raise "More than one dimension match"
                elif mask_priority == 'first':
                    match_dim = int(match_dim[0])
                else:
                    match_dim = int(match_dim[-1]) 
            indexer[match_dim] = mask
            v = v[tuple(indexer)]
        masked_data[k] = v
    return masked_data

def get_value_in_list_of_dict(data, key):
    result = []
    for d in data: 
        result.append(d[key])
    return result

def bin_data_to_idx_list(data, return_type='list'):
    data = np.asarray(data)
    sort_idx = np.argsort(data, kind='stable')
    unique_val, first_ind, counts = np.unique(data[sort_idx], return_index=True, return_counts=True)
    # must initialize the array first, and then fill 
    # If converting the list to numpy object array later and if 
    # each element in the list is a scalar, numpy will automatically 
    # convert each element into object type and form a 2d numpy object array 
    # there seems to be no way to avoid this at this point. 
    bin_idx_list = np.empty(first_ind.size, dtype=object) 
    for n, i0 in enumerate(first_ind):
        c = counts[n]
        tmp_idx = sort_idx[i0:(i0 + c)]
        bin_idx_list[n] = np.asarray(tmp_idx)
    if return_type == 'list': 
        return bin_idx_list, unique_val
    elif return_type == 'dict':
        return {k : v for k, v in zip(unique_val, bin_idx_list)}
    else: 
        raise ValueError(f"Unrecognized return_type. Options: 'list', 'dict'")

def find_ind_in_sub_array(sub_array, ind, mask_size, select='all'):
    sub = np.hstack(np.unravel_index(ind, mask_size))[:, None]
    num_sub = sub_array.shape[1]
    assert sub.shape == (3, 1), "ind should be a integer scalar"
    assert sub_array.shape[0] == 3, "sub_array should be a (3, n) numpy integer array"
    match_Q = np.all(sub_array == sub, axis=0)
    idx = np.nonzero(match_Q)[0]
    if select == 'all':
        return idx
    elif select == 'first':
        return idx[0]
    elif select == 'last':
        return idx[-1]
    elif select == 'shrink':
        if np.all(idx == np.array([0, 1])):
            return 1
        elif np.all(idx == np.array([num_sub-2, num_sub-1])):
            return num_sub-2
        else:
            return idx

def get_intervals_in_1d_binary_array(vec, prepend=0, append=0, including_end_Q=False):
    """
    
    Output: 
        intervals: (N, 2) np.ndarray
    """
    vec = np.asarray(vec)
    vec_diff = np.diff(vec, prepend=prepend, append=append)
    non_zeros = np.nonzero(vec_diff)[0]
    is_start_Q = (vec_diff[non_zeros] == 1)
    start_idx = non_zeros[is_start_Q]
    end_idx = non_zeros[~is_start_Q]
    assert np.all(start_idx < end_idx), f"{start_idx}, {end_idx}"
    if including_end_Q:
        end_idx -= 1
    intervals = np.stack((start_idx, end_idx), axis=1).astype(np.int64)
    return intervals

def get_continuous_interval_endpoint_idx_in_1d_integer_vector(vec, including_end_Q=True):

    vec = np.sort(np.asarray(vec))
    assert np.all(vec[1:] > vec[:-1]), 'vector elements are not strictly monotonically increasing'
    ep_mask = np.zeros(vec.shape, bool)
    diff = vec[1:] - vec[:-1]
    ep_mask[1:] = np.logical_or(ep_mask[1:], diff != 1)
    ep_mask[:-1] = np.logical_or(ep_mask[:-1], diff != 1)
    if including_end_Q: 
        ep_mask[0] = True
        ep_mask[-1] = True
    idx = np.nonzero(ep_mask)[0]
    return idx

def split_table_by_key(table:pd.DataFrame, split_key, output_type='list'):
    table = table.sort_values(by=split_key)
    f_val, f_idx, f_c = np.unique(table[split_key].values, return_index=True, return_counts=True)
    assert np.all(np.diff(f_val) >= 1), f"{split_key} is not sorted in ascending order"
    if output_type == 'list':
        t_num_frame = np.max(f_val) + 1
        result = [[] for _ in range(t_num_frame)]
        for f, i, c in zip(f_val, f_idx, f_c):
            result[f] = table.iloc[i : i + c].sort_index()
    elif output_type == 'dict':
        result = {}
        for f, i, c in zip(f_val, f_idx, f_c):
            result[f] = table.iloc[i : i + c].sort_index()
    else: 
        raise NotImplementedError

    return result

def get_table_value_to_idx_dict(table:pd.DataFrame, key, filter=None):
    idx, key_val = bin_data_to_idx_list(table[key].values)
    if filter is not None: 
        selected_Q = filter(key_val)
        idx = idx[selected_Q]
        key_val = key_val[selected_Q]
    v_to_idx_dict = {}
    for i, e in zip(idx, key_val):
        v_to_idx_dict[e] = i

    return v_to_idx_dict

def get_largest_interval_in_1d_array(vec, prepend=0, append=0):
    ints = get_intervals_in_1d_binary_array(vec, prepend, append)
    if ints.shape[0] > 0: 
        if ints.shape[0] > 1: 
            int_length = ints[:, 1] - ints[:, 0]
            max_int_idx = np.argmax(int_length)
            ints = ints[max_int_idx]
        else: 
            ints = ints[0]
        return int(ints[0]), int(ints[1])
    else: 
        return None, None

def ind_coordinate_transform_euclidean(c1_ind, c1_shape, vec_c1_to_c2, c2_shape):
    c1_ind = np.asarray(c1_ind)
    c1_sub = np.vstack(np.unravel_index(c1_ind, c1_shape))
    for i in range(c1_sub.shape[0]):
        c1_sub[i] += vec_c1_to_c2[i]
    c2_ind = np.ravel_multi_index(c1_sub, c2_shape)
    return c2_ind

def list_of_dict_to_dict_of_array(lod, to_numpy_Q=True): 
    result = {}

    for tmp_d in lod: 
        for k, v in tmp_d.items():
            if k not in result: 
                result[k] = []
            result[k].append(v)
    
    if to_numpy_Q: 
        for k, v in result.items():
            result[k] = np.asarray(v)
    
    return result

def dict_of_dict_to_dict_of_array(dod, to_numpy_Q=True):
    result = {}
    result['key'] = list(dod.keys())
    lod = [dod[k] for k in result['key']]
    result |= list_of_dict_to_dict_of_array(lod, to_numpy_Q=False)

    if to_numpy_Q: 
        for k, v in result.items():
            result[k] = np.asarray(v)
    return result

def select_val_by_num_repeat(id_list, min_syn_per_rid=1):
    """Select root ids by the number of repeats.
    Input:
        rid_list: list or array of root ids
        min_syn_per_rid: int, minimum number of synapses per root id
    Output:
        selected_Q: logical array, 1 for selected
    """
    if min_syn_per_rid <= 1: 
        return np.ones_like(id_list, dtype=bool)
    else: 
        id2idx = bin_data_to_idx_list(id_list, return_type='dict')
        selected_Q = np.zeros_like(id_list, dtype=bool)
        for k, v in id2idx.items():
            if v.size >= min_syn_per_rid:
                selected_Q[v] = True
        return selected_Q

def rows_in(A, B):
    """Check if each rows in A is in B
    Input: 
        A: (N, d) numpy array
        B: (M, d) numpy array
    Output: 
        mask: (N,) boolean array, 1 if row in A is in B, 0 otherwise
    """
    A = np.ascontiguousarray(A)
    B = np.ascontiguousarray(B)
    dt = np.dtype([('', A.dtype)] * A.shape[1])   # 1 field per column
    Av = A.view(dt).ravel()
    Bv = B.view(dt).ravel()
    return np.isin(Av, Bv)   # boolean mask of length N


class ScalarDict: 
    def __init__(self, key, value=None): 
        idx = np.argsort(key, kind='stable')
        self.key = key[idx]
        if value is None: 
            self.value = np.arange(key.size)
        else: 
            assert key.shape == value.shape, "ind and label should have the same shape"
            self.value = value[idx]
        
    def get_value(self, keys, not_found_val=-1):
        keys = np.asarray(keys)
        idx = np.searchsorted(self.key, keys, side='left')
        found_Q = (idx < self.key.size) & (self.key[idx] == keys)
        result = np.full(keys.shape, not_found_val, dtype=self.value.dtype)
        result[found_Q] = self.value[idx[found_Q]]
        return result

    def get_label(self, query_idx):
        query_idx = np.asarray(query_idx)
        assert np.all((query_idx >= 0) & (query_idx < self.value.size)), "query_idx out of range"
        return self.key[query_idx]

        

        
