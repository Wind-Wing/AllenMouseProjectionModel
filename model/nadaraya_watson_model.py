import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels


class NadarayaWatsonModel(object):
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
    # exp_list - [num_exp] experiences as training data that used to predict the unknown projection from above voxels.
    # Return: projection_matrix - [num_target_voxel] aggregated density of projection from all voxels in source region
    def get_region_voxel_projection_matrix(self, source_mask_idx, exp_list, aggregate_func=np.mean):
        # assert all([x.injection_centroid[2] >= 57 for x in exp_list])
        source_voxel_coordinate = np.array(source_mask_idx).transpose()
        source_voxel_num = int(len(source_mask_idx[0]))
        print("Source voxels' number %d" % source_voxel_num)

        injection_centroid = np.array([x.injection_centroid for x in exp_list])
        normalized_projection = np.array([x.normalized_projection_density for x in exp_list])

        projection_matrix_list = self._calc_projection_matrix(
            source_voxel_coordinate,
            injection_centroid,
            normalized_projection)
        projection_matrix = aggregate_func(projection_matrix_list, axis=0)
        return projection_matrix


if __name__ == "__main__":
    pass