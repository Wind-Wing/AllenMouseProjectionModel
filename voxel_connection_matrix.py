import matplotlib.pyplot as plt
import numpy as np
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from experiment import Experiment
from structure_mask import StructureMask


class VoxelConnectionMatrix(object):
    def __init__(self, structure_id_list=None):
        exp_list = self._read_exp(structure_id_list)
        self.mat = self._calc_connection_density_matrix(exp_list, structure_id_list)

    def _read_exp(self, structure_id_list):
        self.mcc = MouseConnectivityCache(resolution=100)
        exp_list = self.mcc.get_experiments(injection_structure_ids=structure_id_list)
        exp_formater = lambda exp: Experiment(exp, resolution=100)
        return list(map(exp_formater, exp_list))

    @staticmethod
    def _calc_connection_density_matrix(exp_list, structure_id_list):
        injection_centroid_list = [np.around(x.injection_centroid).astype(int).tolist() for x in exp_list]
        normalized_projection_density_list = [x.normalized_projection_density for x in exp_list]

        # region_shape = normalized_projection_density_list[0].shape
        # assert len(region_shape) == 3
        # voxel_num = region_shape[0] * region_shape[1] * region_shape[2]

        structure_mask = StructureMask()
        if structure_id_list is None:
            mask = np.ones_like(normalized_projection_density_list[0])
        else:
            mask = structure_mask.get_mask(structure_id_list)
        mask_coordinate = [x.tolist() for x in np.argwhere(mask == 1)]
        mask_idx = mask.nonzero()
        voxel_num = len(mask_coordinate)

        # TODO: consider how to normalize matrix between different voxels
        # TODO: check one voxel will not injection 2 different experiences
        mat = np.zeros([voxel_num, voxel_num])
        for i in range(len(injection_centroid_list)):
            centroid = injection_centroid_list[i]
            try:
                idx = mask_coordinate.index(centroid)
            except ValueError:
                print(centroid)
            else:
                projection = normalized_projection_density_list[i]
                norm = np.linalg.norm(projection)
                mat[:, idx] = projection[mask_idx] / norm
        return mat

    def show(self):
        plt.imshow(self.mat)
        plt.show()


if __name__ == "__main__":
    mat = VoxelConnectionMatrix([961])
    mat.show()
