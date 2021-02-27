import numpy as np
import matplotlib.pyplot as plt
from structure_mask import StructureMask
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache


# TODO: get a voxel level matrix
class SurfaceMap(object):
    def __init__(self, data_path):
        self.projection = np.loadtxt(data_path)
        mcc = MouseConnectivityCache(resolution=100)
        self.structure_tree = mcc.get_structure_tree()
        self.structure_mask = StructureMask()

        # TODO: Test code
        self._get_full_brain_mask()

    def _get_full_brain_mask(self):
        summary_structures = self.structure_tree.get_structures_by_set_id([167587189])
        mask = self.structure_mask.get_mask(summary_structures)
        plt.imshow(mask)
        plt.show()


if __name__ == "__main__":
    surface_map = SurfaceMap("../results/max_mean-withoutNorm-1614412388.541507.txt")


