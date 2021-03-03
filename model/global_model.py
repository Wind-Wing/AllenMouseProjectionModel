import numpy as np
from utils.structure_mask import StructureMask
from model.voxel_model import VoxelModel


# TODO: compare with regional model and homogeneous model
# Using Nadaraya Watson method to calculate region projection matrix based on global data
class GlobalModel(object):
    def __init__(self, gamma=0.013):
        self.voxel_model = VoxelModel(gamma)
        self.structure_mask = StructureMask()

    # Return - mat[i][j]: i -> j projection
    def get_regional_projection_matrix(self, source_structure_id_list, target_structure_id_list, exp_list):
        assert all(exp_list.injection_centroid[2] >= 57)

        _get_region_projection_vector = np.vectorize(self.get_region_projection)
        projection_list = _get_region_projection_vector(source_structure_id_list, target_structure_id_list, exp_list)

        contralateral_mat = self._get_hemisphere_regional_projection_matrix(
            projection_list,
            source_structure_id_list,
            target_structure_id_list,
            0)

        lpsilateral_mat = self._get_hemisphere_regional_projection_matrix(
            projection_list,
            source_structure_id_list,
            target_structure_id_list,
            1)
        return lpsilateral_mat, contralateral_mat

    def _get_hemisphere_regional_projection_matrix(
            self,
            projection_list,
            source_structure_id_list,
            target_structure_id_list,
            target_hemisphere):
        mat = np.zeros([len(source_structure_id_list), len(target_structure_id_list)])
        for i in range(len(source_structure_id_list)):
            mask_idx = self.structure_mask.get_mask([source_structure_id_list[i]], target_hemisphere).nonzero()
            region_projection_list = np.array([x[mask_idx] for x in projection_list])
            assert len(region_projection_list.shape) == 2
            mat[i, :] = np.mean(region_projection_list, axis=1)
        return mat

    @staticmethod
    def _normalize_projection_matrix(mat):
        _sum = np.sum(mat, axis=1)
        return mat / _sum

    # Return - volume[x][y][z]: voxel level projection density from source regions
    def get_region_projection(self, source_structure_id, target_structure_id_list, exp_list):
        assert all(exp_list.injection_centroid[2] >= 57)
        print("Calculating projection of %d" % source_structure_id)
        voxel_projection_matrix = self.voxel_model.get_voxel_mean_projection_matrix(
            [source_structure_id], target_structure_id_list, exp_list)
        return voxel_projection_matrix
