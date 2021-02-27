import numpy as np
from utils.structure_mask import StructureMask


class RegionalModel(object):
    def __init__(self, voxel_projection_matrix, region_id_list):
        self.voxel_matrix = voxel_projection_matrix
        self.id_list = region_id_list

    def get_regional_projection_matrix(self):
        num_regions = len(self.id_list)
        mat = np.zeros([num_regions, num_regions])


