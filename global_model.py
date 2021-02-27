import numpy as np
from structure_mask import StructureMask
from voxel_model import VoxelModel


# Using nadaraya watson method to calculate region projection matrix based on global data
class GlobalModel(object):
    def __init__(self, region_id_list, exp_list):
        self.id_list = region_id_list
        self.exp_list = exp_list
        self.gamma = 0.013

    # Return - mat[i][j]: i -> j projection
    def get_regional_projection_matrix(self):
        num_regions = len(self.id_list)
        mat = np.zeros([num_regions, num_regions])

        projection_list = np.array([self._get_region_projection(x) for x in self.id_list])

        structure_mask = StructureMask()
        for i in range(num_regions):
            mask_idx = structure_mask.get_mask([self.id_list[i]]).nonzero()
            region_projection_list = np.array([x[mask_idx] for x in projection_list])
            assert len(region_projection_list.shape) == 2
            mat[i, :] = np.mean(region_projection_list, axis=1)

        # mat = self._normalize_projection_matrix(mat)
        return mat

    @staticmethod
    def _normalize_projection_matrix(mat):
        _sum = np.sum(mat, axis=1)
        return mat / _sum

    def _get_region_projection(self, source_structure_id):
        print("Calculating projection of %d" % source_structure_id)
        voxel_model = VoxelModel(self.gamma)
        voxel_projection_matrix = voxel_model.get_voxel_mean_projection_matrix(
            [source_structure_id],
            self.id_list,
            self.exp_list)
        return voxel_projection_matrix










