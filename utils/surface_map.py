import os
import numpy as np
import matplotlib.pyplot as plt
from structure_mask import StructureMask
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache


# TODO: get a voxel level matrix
# Default view is coronal plane
class SurfaceMap(object):
    def __init__(self, data_path):
        self.projection = np.loadtxt(data_path)
        mcc = MouseConnectivityCache(resolution=100)
        self.structure_tree = mcc.get_structure_tree()
        self.structure_mask = StructureMask()

        mask = self._get_cortex_volume_mask()
        idx = self._find_cross_section_idx(mask)
        self._get_cortex_mask(mask, idx)

    def _get_cortex_volume_mask(self):
        summary_structures = self.structure_tree.get_structures_by_set_id([688152357])
        id_list = [x['id'] for x in summary_structures]
        mask = self.structure_mask.get_mask(id_list)
        return mask

    @staticmethod
    def _find_cross_section_idx(mask):
        voxel_count = np.sum(mask, axis=(0, 2))
        return np.argmax(voxel_count)

    @staticmethod
    def _get_cortex_mask(mask, idx):
        plt.imshow(mask[:, idx, :])
        plt.show()


if __name__ == "__main__":
    os.chdir("../")
    surface_map = SurfaceMap("results/max_mean-withoutNorm-1614412388.541507.txt")


