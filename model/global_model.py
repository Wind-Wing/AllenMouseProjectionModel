import numpy as np
from model.nadaraya_watson_model import NadarayaWatsonModel
from utils.constants import VOXEL_SHAPE

# TODO: compare with regional model and homogeneous model
# Using Nadaraya Watson method to calculate region projection matrix based on global data
class GlobalModel(object):
    def __init__(self, gamma=0.013):
        self.nadaraya_watson_model = NadarayaWatsonModel(gamma)

    # Return - projection [x, y, z]
    def get_one_region_projection(self, source_mask_idx, cortex_mask_idx, exp_list, aggregate_func=np.max):
        # assert all([x.injection_centroid[2] >= 57 for x in exp_list])
        projection_vector = self.nadaraya_watson_model.get_region_voxel_projection_matrix(
            source_mask_idx, exp_list, aggregate_func)

        projection = np.zeros(shape=VOXEL_SHAPE)
        projection[cortex_mask_idx] = projection_vector
        return projection

    @staticmethod
    # Return - mat[i][j]: i -> j projection
    def get_regional_projection_matrix(projection_list, target_masks_idx, aggregate_func=np.mean):
        regional_projection_vectors = []
        for voxel_projection in projection_list:
            region_projection_vector = [aggregate_func(voxel_projection[idx]) for idx in target_masks_idx]
            regional_projection_vectors.append(region_projection_vector)

        mat = np.array(regional_projection_vectors)
        return mat
