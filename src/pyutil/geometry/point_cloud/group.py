from typing import Dict, Optional

import numpy as np
from scipy.spatial import KDTree

from ... import util


# Assume dict key, val are in the same order (Python 3.7+)
class PointCloudGroup:
    def __init__(self, pc_dict:dict, group_info: Optional[Dict] = None):
        """ Initialize the point cloud group with a dictionary of point clouds.

        Args:
            pc_dict: A dictionary where keys are point cloud IDs and
            values are (N, d) arrays of points.
            group_info: Optional additional information about the groups
        """
        self.id = np.asarray(list(pc_dict.keys()))
        self.num_pts = np.zeros(self.id.size, dtype=np.uint32)
        node_pos = []
        for i, k in enumerate(self.id):
            v = pc_dict[k]
            self.num_pts[i] = v.shape[0]
            node_pos.append(v)

        self.c_idx_1 = np.cumsum(self.num_pts).astype(np.uint32)
        self.c_idx_0 = np.concatenate(([int(0)], self.c_idx_1[:-1]))
        node_pos = np.concatenate(node_pos, axis=0)
        self.dim = node_pos.shape[1]
        self.kdt = KDTree(node_pos)

        if group_info is not None:
            # all pc_dict key are in group_info
            assert isinstance(group_info, (dict, type(None))), f'Unexpected group_info type: {type(group_info)}'
            # Ensure group_info keys cover all pc_dict keys
            missing = set(pc_dict.keys()) - set(group_info.keys())
            if missing:
                raise ValueError(f"group_info missing keys: {missing}")
            # Align group_info to self.id order
            self.group_info = {k: group_info[k] for k in self.id}
        else:
            self.group_info = None

    @property
    def xyz(self):
        # skeleton xyz in nm
        return self.kdt.data.reshape((-1, self.dim))

    def get_knn_nodes(self, xyz_nm, k, distance_upper_bound=np.inf):
        """ Get k-nearest neighbors for a set of points in 3D space.

        Args:
            xyz_nm: (N, self.dim) array of points in nanometers
            k: int, number of neighbors to retrieve
            distance_upper_bound: float, maximum distance to consider (in nanometers)

        Returns:
            k_dist: (N, k) array of distances to the k-nearest neighbors
            k_idx: (N, k) array of indices of the k-nearest neighbors
        """
        xyz_nm = np.atleast_2d(np.asarray(xyz_nm))
        assert xyz_nm.shape[1] == self.dim, f'Unexpected input coordinate dimension: {xyz_nm.shape}'

        k_dist, k_idx = self.kdt.query(xyz_nm, k=k,
                                       distance_upper_bound=distance_upper_bound)
        return k_dist, k_idx

    def get_all_nodes_within_distance(self, xyz_nm, distance_upper_bound, return_dist_Q=False):
        """ Get all nodes within a certain distance for a set of points in 3D space.

        Args:
            xyz_nm: (N, self.dim) array of points in nanometers
            distance_upper_bound: float, maximum distance to consider (in nanometers)

        Returns:
            all_dist: (N, M) array of distances to all neighbors within the distance_upper_bound
            all_idx: (N, M) array of indices of all neighbors within the distance_upper_bound
        """
        xyz_nm = np.atleast_2d(np.asarray(xyz_nm))
        assert xyz_nm.shape[1] == self.dim, f'Unexpected input coordinate dimension: {xyz_nm.shape}'

        all_idx = self.kdt.query_ball_point(xyz_nm, r=distance_upper_bound,
                                                      workers=-1, return_sorted=False,
                                                      return_length=False)
        all_idx = [np.asarray(idx) for idx in all_idx]
        if return_dist_Q:
            all_dist = [np.linalg.norm(xyz_nm[i] - self.xyz[all_idx[i]], axis=1) for i in range(xyz_nm.shape[0])]
            return all_dist, all_idx
        else:
            return all_idx

    def _idx_to_id_idx(self, idx):
        """ Convert point index in the array to group index.
        Inputs:
            idx: int or array-like of point indices
        Outputs:
            key_idx: int or array of group indices corresponding to the input point indices
        """
        key_idx = np.searchsorted(self.c_idx_1, idx, 'right')
        return key_idx

    def _idx_to_id(self, idx):
        """ Convert point index in the array to group id.
        Inputs:
            idx: int or array-like of point indices
        Outputs:
            id: int or array of group ids corresponding to the input point indices
        """
        key_idx = self._idx_to_id_idx(idx)
        return self.id[key_idx]

    def get_neighbor_group_info(self, nb_dist, nb_idx):
        """ Get information about neighboring groups based on distance and index
        Inputs:
            nb_dist: (N, k) array of distances to neighboring k points
            nb_idx: (N, k) array of indices of neighboring k points
        Outputs:
            A dictionary containing information about the neighboring groups, including:
                - 'id': (N, ) id of the nearby groups
                - 'num_pts': (N, ) number of query points falling in each nearby group
                - 'dist': (N, ) distances to the nearest processes of each nearby groups
                - 'xyz_nm': (N, k, self.dim) array of xyz coordinates of the nearest points in each group to the query points
        """
        # Minimal distance between pre_type neurons to the target neuron
        nb_dist = nb_dist.flatten()
        nb_idx = nb_idx.flatten()
        # Merge the data from all query points - input spatial information lost
        is_valid_dist_Q = np.isfinite(nb_dist.flatten())
        if not np.all(is_valid_dist_Q):
            nb_dist = nb_dist[is_valid_dist_Q]
            nb_idx = nb_idx[is_valid_dist_Q]
        # Use the binning information to find the corresponding neuron in the skeleton point cloud
        nn_pt_rid_idx = self._idx_to_id_idx(nb_idx)
        bin_idx, nn_pt_idx_u = util.bin_data_to_idx_list(nn_pt_rid_idx.flatten())
        # Get the nearest points of each group
        nn_pt_dist = np.zeros(nn_pt_idx_u.size, dtype=np.float32)
        nn_pt_idx = np.zeros(nn_pt_idx_u.size, dtype=np.uint)
        nn_num_pts = np.zeros(nn_pt_idx_u.size, dtype=np.uint)
        for i, tmp_idx in enumerate(bin_idx):
            # Each bin is the distances to all the nearby points of one group
            tmp_dist = nb_dist.flat[tmp_idx]
            nn_num_pts[i] = tmp_idx.size

            tmp_min_idx = np.argmin(tmp_dist)
            nn_pt_dist[i] = tmp_dist[tmp_min_idx]
            nn_pt_idx[i] = nb_idx.flat[tmp_idx[tmp_min_idx]]

        nn_pt_xyz_nm = self.xyz[nn_pt_idx]
        nn_pt_id = self.id.flat[nn_pt_idx_u]
        result = {'id': nn_pt_id,
                  'num_pts': nn_num_pts, # number of query points falling in the group
                  'dist': nn_pt_dist,
                  'xyz_nm': nn_pt_xyz_nm # nearest points in each group to the query points
                  }
        return result

