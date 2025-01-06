from __future__ import absolute_import
from collections.abc import Iterable
import torch
import numpy as np
import scipy.sparse as sp



class Skeleton(object):
    def __init__(
        self,
        parents,
        joints_left,
        joints_right,
        joints_group=None,
        joints_names=None,
    ):
        assert len(joints_left) == len(joints_right)

        self._parents = np.array(parents)
        self._joints_left = joints_left
        self._joints_right = joints_right
        self._joints_group = joints_group
        self._joints_names = joints_names
        if self._joints_names is None:
            self._joints_names = [""] * len(self._parents)
        assert isinstance(self._joints_names, Iterable) and len(
            self._joints_names
        ) == len(
            self._parents
        ), "joint_names should be an iterable with as many elements as joints."
        self._compute_metadata()

    def num_joints(self):
        return len(self._parents)

    def num_bones(self):
        return len(list(filter(lambda x: x >= 0, self._parents)))

    def parents(self):
        return self._parents

    def has_children(self):
        return self._has_children

    def children(self):
        return self._children

    def remove_joints(self, joints_to_remove):
        # Compute list complementary to joints to be removed
        valid_joints = [
            i for i in range(len(self._parents)) if i not in joints_to_remove
        ]

        # Recursive update of parents
        for i in range(len(self._parents)):
            while self._parents[i] in joints_to_remove:
                self._parents[i] = self._parents[self._parents[i]]

        index_offsets = np.zeros(len(self._parents), dtype=int)
        new_parents = []
        for i, parent in enumerate(self._parents):
            if i not in joints_to_remove:
                new_parents.append(parent - index_offsets[parent])
            else:
                index_offsets[i:] += 1
        self._parents = np.array(new_parents)

        # Zip other metadata indexed by joint
        jointwise_metadata = [
            (
                self._joints_names[i],
                i in self._joints_left,  # l/r joints index replaced by mask
                i in self._joints_right,
            )
            for i in range(len(self._joints_names))
        ]

        # Drop zipped entried from higher indices to lower
        joints_to_remove.sort(reverse=True)
        for i_to_pop in joints_to_remove:
            jointwise_metadata.pop(i_to_pop)

        # Unzip
        self._joints_names, ljoints_mask, rjoints_mask = zip(
            *jointwise_metadata
        )

        # Convert l/r masks back to indices
        self._joints_left = [
            i for i, is_left in enumerate(ljoints_mask) if is_left
        ]
        self._joints_right = [
            i for i, is_right in enumerate(rjoints_mask) if is_right
        ]

        # Update other metadata
        self._compute_metadata()

        return valid_joints

    def joints_left(self):
        return self._joints_left

    def joints_right(self):
        return self._joints_right

    def joints_group(self):
        return self._joints_group

    def joints_names(self):
        return self._joints_names

    def bones(self):
        return self._bones

    def bones_left(self):
        return self._bones_left

    def bones_right(self):
        return self._bones_right

    def bones_names(self):
        return self._bones_names

    def _compute_metadata(self):
        self._has_children = np.zeros(len(self._parents)).astype(bool)
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._has_children[parent] = True

        self._children = []
        for i, parent in enumerate(self._parents):
            self._children.append([])
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._children[parent].append(i)

        # Creates bones as a tuple of (joint, joint_parent) tuples
        self._bones = tuple(
            (j, p) for j, p in enumerate(self._parents) if p >= 0
        )

        self._bones_names = tuple(
            f"{self._joints_names[j]}->{self._joints_names[i]}"
            for i, j in self._bones
        )

        # Creates left and right bones indices list
        bone_parent = dict(self._bones)
        bone_index = {b: i for i, b in enumerate(self._bones)}
        skeleton_left = tuple(
            (j, bone_parent[j]) for j in self._joints_left if j >= 0
        )
        skeleton_right = tuple(
            (j, bone_parent[j]) for j in self._joints_right if j >= 0
        )
        self._bones_left = tuple(bone_index[b] for b in skeleton_left)
        self._bones_right = tuple(bone_index[b] for b in skeleton_right)


#bone graph conv
#原来是建模关节点之间的关系，这里尝试建模使用graph建模bone向量之间的关系
def get_sketch_setting():
    # return [[0, 1], [1, 2], [2, 3], [3, 4],
    #             [0, 5], [5, 6], [6, 7], [7, 8],
    #             [0, 9], [9, 10], [10, 11], [11, 12],
    #             [0, 13], [13, 14], [14,  15], [15, 16],
    #             [0, 17], [17, 18], [18, 19], [19, 20]]

    return [[0, 1], [1, 2], [2, 3],
            [4,5], [5, 6], [6, 7],
            [8,9],[9,10],[10,11],
            [12,13],[13,14],[14,15],
            [16,17],[17,18],[18,19]]

                
def adj_mx_from_edges(num_pts, edges, sparse=False, eye=True):
    edges = np.array(edges, dtype=np.int32)
    data, i, j = np.ones(edges.shape[0]), edges[:, 0], edges[:, 1]
    adj_mx = sp.coo_matrix((data, (i, j)), shape=(num_pts, num_pts), dtype=np.float32)

    # build symmetric adjacency matrix
    adj_mx = adj_mx + adj_mx.T.multiply(adj_mx.T > adj_mx) - adj_mx.multiply(adj_mx.T > adj_mx)
    if eye:
        adj_mx = normalize(adj_mx + sp.eye(adj_mx.shape[0]))
    else:
        adj_mx = normalize(adj_mx)

    if sparse:
        adj_mx = sparse_mx_to_torch_sparse_tensor(adj_mx)
    else:
        adj_mx = torch.tensor(adj_mx.todense(), dtype=torch.float)
    return adj_mx

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
