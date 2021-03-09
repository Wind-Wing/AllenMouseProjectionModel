import pytest
import math
import numpy as np
from model.nadaraya_watson_model import NadarayaWatsonModel


class FakeExp(object):
    def __init__(self, injection_centroid, projection_density):
        self.injection_centroid = injection_centroid
        self.normalized_projection_density = projection_density


def test_get_region_voxel_projection_matrix():
    source_mask = np.zeros([4, 2, 8])
    source_mask[1][1][7] = 1
    source_mask[3][0][4] = 1
    source_mask_idx = source_mask.nonzero()

    projection1 = np.array([0.2, 0, 0.3])
    projection2 = np.array([0.8, 0.4, 0])
    exp_list = [FakeExp([2, 1, 5], projection1),
                FakeExp([3, 0, 2], projection2)]

    model = NadarayaWatsonModel(1)
    mat = model.get_region_voxel_projection_matrix(source_mask_idx, exp_list)
    mat = np.round(mat, 3)

    k1 = math.exp(-5)
    k2 = math.exp(-30)
    k3 = math.exp(-3)
    k4 = math.exp(-4)
    p1 = (k1 * projection1 + k2 * projection2) / (k1 + k2)
    p2 = (k3 * projection1 + k4 * projection2) / (k3 + k4)
    excepted_mat = (p1 + p2) / 2
    excepted_mat = np.round(excepted_mat, 3)

    assert all(mat == excepted_mat)
