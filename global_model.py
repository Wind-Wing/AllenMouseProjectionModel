import numpy as np
from structure_mask import StructureMask
from voxel_model import VoxelModel


class FullCortexModel(object):
    def __init__(self, region_id_list, exp_list):
        self.id_list = region_id_list
        self.exp_list = exp_list
        self.gamma = 0.013

    def get_regional_projection_matrix(self):
        num_regions = len(self.id_list)
        mat = np.zeros([num_regions, num_regions])

        projection_list = np.array([self._get_region_projection(x) for x in self.id_list])

        structure_mask = StructureMask()
        for i in self.id_list:
            mask_idx = structure_mask.get_mask(i).nonzeros()
            region_projection_list = projection_list[:, mask_idx]
            assert len(region_projection_list.shape) == 2
            mat[:, i] = np.mean(region_projection_list, axis=1)
        return mat

    def _get_region_projection(self, structure_id):
        voxel_model = VoxelModel(self.gamma)
        voxel_projection_matrix = voxel_model.get_projection_matrix([structure_id], self.exp_list)
        voxel_projection = np.mean(voxel_projection_matrix, axis=0)
        return voxel_projection










