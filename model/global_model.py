import numpy as np
from model.nadaraya_watson_model import NadarayaWatsonModel


# TODO: compare with regional model and homogeneous model
# Using Nadaraya Watson method to calculate region projection matrix based on global data
class GlobalModel(object):
    def __init__(self, gamma=0.013):
        self.nadaraya_watson_model = NadarayaWatsonModel(gamma)

    # Return - mat[i][j]: i -> j projection
    def get_regional_projection_matrix(self, source_masks_idx, exp_list):
        # assert all([x.injection_centroid[2] >= 57 for x in exp_list])
        _func = lambda x: self.nadaraya_watson_model.get_region_voxel_projection_matrix(
            x, exp_list, aggregate_func=np.max)
        projection_list = [_func(x) for x in source_masks_idx]

        mat = self._get_regional_projection_matrix(projection_list)
        return mat

    @staticmethod
    def _get_regional_projection_matrix(projection_list, aggregate_func=np.mean):
        regional_projection_vectors = [aggregate_func(x, axis=1) for x in projection_list]
        mat = np.array(regional_projection_vectors)
        return mat
