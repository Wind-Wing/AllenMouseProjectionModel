import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from utils.structure_mask import StructureMask


class VoxelModel(object):
    def __init__(self, gamma):
        self.gamma = gamma
        self.kernel_type = "rbf"

    # x - [num_voxel, 3]
    # y - [num_experience, 3]
    # Return: kernel - [num_voxel, num_experience]
    def _get_kernel(self, x, y):
        params = {"gamma": self.gamma}
        kernel = pairwise_kernels(x, y, metric=self.kernel_type, filter_params=True, **params)
        return kernel

    # kernel - [num_voxel, num_experience]
    # normalize on num_experience axis
    @staticmethod
    def _normalize_kernel(kernel):
        _sum = np.sum(kernel, axis=1, keepdims=True)
        _kernel = kernel / _sum
        return _kernel

    # Calculate projection strength using Nadaraya Watson Method
    # x - [num_regional_voxel, 3] Coordinate of source voxels that need to predict its projection density
    # y - [num_experience, 3] Centroid injection coordinate of selected experiences
    # projection_density - [num_experiences, num_target_voxel] Normalized projection density from selected experiences
    # Return: projection weight matrix - [num_regional_voxel, num_target_voxel] a[i][j]: i -> j 's projection strength
    def _calc_projection_matrix(self, x, y, projection_density):
        kernel = self._get_kernel(x, y)
        normalized_kernel = self._normalize_kernel(kernel)
        projection_matrix = np.matmul(normalized_kernel, projection_density)
        return projection_matrix

    # source_id_list - [num_structure_id] Voxels in which structures that needed to predict it's projection
    #                   taking this voxel as injection point.
    # target_id_list - [num_structure_id] Voxels in which structures that source voxel project to.
    # exp_list - [num_exp] experiences as training data that used to predict the unknown projection from above voxels.
    # Return: projection_matrix - [x, y, z] mean density of projection from all voxels in source region
    def get_voxel_mean_projection_matrix(self, source_id_list, target_id_list, exp_list):
        assert all(exp_list.injection_centroid[2] >= 57)
        structure_mask = StructureMask()
        source_mask = structure_mask.get_mask(source_id_list, hemisphere=1)
        source_voxel_coordinate = np.array(source_mask.nonzero()).transpose()
        source_voxel_num = int(np.sum(source_mask))

        target_mask = structure_mask.get_mask(target_id_list, hemisphere=2)
        target_voxel_idx = target_mask.nonzero()
        target_voxel_num = int(np.sum(target_mask))
        print("Source voxels' number %d, target voxels' number %d" % (source_voxel_num,  target_voxel_num))

        injection_centroid = np.array([x.injection_centroid for x in exp_list])
        normalized_projection = np.array([x.normalized_projection_density[target_voxel_idx] for x in exp_list])

        _projection_matrix = self._calc_projection_matrix(
            source_voxel_coordinate,
            injection_centroid,
            normalized_projection)
        mean_projection_matrix = np.max(_projection_matrix, axis=0)

        projection_matrix = np.zeros(source_mask.shape)
        projection_matrix[target_voxel_idx] = mean_projection_matrix
        projection_matrix = np.reshape(projection_matrix, source_mask.shape)
        return projection_matrix


if __name__ == "__main__":
    voxel_model = VoxelModel()












