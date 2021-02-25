import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from structure_mask import StructureMask


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
        return _sum

    # Calculate projection strength using Nadaraya Watson Method
    # x - [num_voxel, 3] Coordinate of source voxels that need to predict its projection density
    # y - [num_experience, 3] Centroid injection coordinate of selected experiences
    # projection_density - [num_experiences, num_voxel] Normalized projection density from selected experiences
    # Return: projection weight matrix - [num_voxel, num_voxel] a[i][j] means i -> j 's projection strength
    def _calc_projection_matrix(self, x, y, projection_density):
        kernel = self._get_kernel(x, y)
        normalized_kernel = self._normalize_kernel(kernel)
        projection_matrix = np.matmul(normalized_kernel, projection_density)
        return projection_matrix

    # structure_id_list - [num_structure_id] Voxels in which structures that needed to predict it's projection
    #                   taking this voxel as injection point.
    # exp_list - [num_exp] experiences as training data that used to predict the unknown projection from above voxels.
    def get_projection_matrix(self, structure_id_list, exp_list):
        structure_mask = StructureMask()
        mask = structure_mask.get_mask(structure_id_list)
        voxel_coordinate = mask.nonzeros().transpose()

        injection_centroid = [x.injection_centroid for x in exp_list]
        normalized_projection = [x.normalized_projection_density for x in exp_list]

        projection_matrix = self._calc_projection_matrix(voxel_coordinate, injection_centroid, normalized_projection)
        return projection_matrix


if __name__ == "__main__":
    voxel_model = VoxelModel()












