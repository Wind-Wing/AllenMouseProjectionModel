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
    # x - [num_regional_voxel, 3] Coordinate of source voxels that need to predict its projection density
    # y - [num_experience, 3] Centroid injection coordinate of selected experiences
    # projection_density - [num_experiences, num_all_voxel] Normalized projection density from selected experiences
    # Return: projection weight matrix - [num_regional_voxel, num_all_voxel] a[i][j] means i -> j 's projection strength
    def _calc_projection_matrix(self, x, y, projection_density):
        kernel = self._get_kernel(x, y)
        normalized_kernel = self._normalize_kernel(kernel)
        projection_matrix = np.matmul(normalized_kernel, projection_density)
        return projection_matrix

    # structure_id_list - [num_structure_id] Voxels in which structures that needed to predict it's projection
    #                   taking this voxel as injection point.
    # exp_list - [num_exp] experiences as training data that used to predict the unknown projection from above voxels.
    # Return: projection_matrix - [num_regional_voxel, x, y, z] density of projection from this region voxel
    def get_projection_matrix(self, structure_id_list, exp_list):
        structure_mask = StructureMask()
        mask = structure_mask.get_mask(structure_id_list)
        voxel_coordinate = mask.nonzeros().transpose()

        injection_centroid = np.array([x.injection_centroid for x in exp_list])
        normalized_projection = np.array([x.normalized_projection_density for x in exp_list])
        _shape = tuple(normalized_projection.shape)
        normalized_projection = np.reshape(normalized_projection, (int(_shape[0]), -1))

        projection_matrix = self._calc_projection_matrix(voxel_coordinate, injection_centroid, normalized_projection)
        projection_matrix = np.reshape(projection_matrix, (-1, int(_shape[1]), int(_shape[2]), int(_shape[3])))
        return projection_matrix


if __name__ == "__main__":
    voxel_model = VoxelModel()












