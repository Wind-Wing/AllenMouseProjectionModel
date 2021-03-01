import os
import numpy as np
import matplotlib.pyplot as plt
from structure_mask import StructureMask
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from utils.cortical_map import CorticalMap


class SurfaceMap(object):
    def __init__(self, resolution=100):
        mcc = MouseConnectivityCache(resolution=resolution)
        self.structure_tree = mcc.get_structure_tree()
        self.structure_mask = StructureMask()
        self.cortical_map = CorticalMap(projection='top_view')
        module_path = os.path.dirname(__file__)
        self.base_path = os.path.join(module_path, '../')

    def draw_cortical_surface_map(self, relative_data_path, source_structure_id):
        source_mask = self.structure_mask.get_mask([source_structure_id])
        projection_matrix = self._load_projection_matrix(relative_data_path)
        source_map = self.cortical_map.transform(source_mask)
        projection_map = self.cortical_map.transform(projection_matrix)

        save_name = os.path.basename(relative_data_path)
        save_name = os.path.splitext(save_name)[0]
        self._draw_surface_map(source_map, projection_map, save_name)

    def _draw_surface_map(self, source_map, projection_map, save_name):
        save_path = self.base_path + "/results"
        save_path = os.path.join(save_path, save_name)
        print(save_path)

        plt.contour(source_map)
        plt.imshow(projection_map)
        plt.savefig(save_path + ".png")
        plt.show()

    def _load_projection_matrix(self, relative_data_path):
        data_path = os.path.join(self.base_path, relative_data_path)
        projection = np.load(data_path)
        return projection


if __name__ == "__main__":
    os.chdir("../")
    surface_map = SurfaceMap()
    surface_map.draw_cortical_surface_map("results/max_mean-withoutNorm-projection_volume-1614604515.124253.npy", 184)



