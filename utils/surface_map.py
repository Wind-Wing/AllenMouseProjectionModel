import os
import numpy as np
import matplotlib.pyplot as plt
from structure_mask import StructureMask
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from model.cortex_model import CortexModel
from utils.cortical_map import CorticalMap
from utils.constants import *


# TODO: when using cortical_map, distinguish value out of mask and zero value. (If the structure is smaller than cortex)
class SurfaceMap(object):
    def __init__(self):
        mcc = MouseConnectivityCache(resolution=RESOLUTION)
        self.structure_tree = mcc.get_structure_tree()
        self.structure_mask = StructureMask()
        self.cortical_map = CorticalMap(projection='top_view')
        module_path = os.path.dirname(__file__)
        self.base_path = os.path.join(module_path, '../')
        self.top_down_overlay = plt.imread(self.base_path + "cortical_coordinates/" + "cortical_map_top_down.png")

    def draw_cortical_surface_map(self, relative_data_path, source_structure_id, source_structure_name):
        print(relative_data_path, source_structure_id)
        source_mask = self.structure_mask.get_mask([source_structure_id], hemisphere=R_HEMISPHERE)
        projection_matrix = self._load_projection_matrix(relative_data_path)
        source_map = self.cortical_map.transform(source_mask)
        projection_map = self.cortical_map.transform(projection_matrix)

        save_name = os.path.basename(relative_data_path)
        save_name = os.path.splitext(save_name)[0]
        self._draw_surface_map(source_map, projection_map, source_structure_name + "-" + save_name)

    def _draw_surface_map(self, source_map, projection_map, save_name):
        plt.subplots(figsize=(6, 6))
        im = plt.imshow(projection_map, zorder=1, cmap=plt.cm.inferno)
        source_map[source_map > 0] = 1
        plt.contour(source_map, zorder=2, cmap=plt.cm.cool)

        extent = plt.gca().get_xlim() + plt.gca().get_ylim()
        plt.imshow(self.top_down_overlay, interpolation="nearest", extent=extent, zorder=3)

        cbar = plt.colorbar(im, shrink=0.3, use_gridspec=True)
        cbar.ax.tick_params(labelsize=6)
        plt.tight_layout()

        save_path = self.base_path + "/results"
        save_path = os.path.join(save_path, save_name)
        plt.savefig(save_path + ".png")
        plt.close()

    def _load_projection_matrix(self, relative_data_path):
        data_path = os.path.join(self.base_path, relative_data_path)
        projection = np.load(data_path)
        return projection


if __name__ == "__main__":

    os.chdir("../")
    files = os.listdir("results/")
    _filter = lambda x: x.endswith(".npy") and not x.startswith("mean") and not x.startswith("amax")
    data_files = [x for x in filter(_filter, files)]
    data_paths = ["results/" + x for x in filter(_filter, files)]
    structure_ids = [int(x.split("-")[0]) for x in data_files]
    print("Len %d - " % len(structure_ids), structure_ids)

    ids, names = CortexModel.get_structures_info()
    id_to_name = dict(zip(ids, names))

    surface_map = SurfaceMap()
    func = np.vectorize(surface_map.draw_cortical_surface_map)
    func(data_paths, structure_ids, [id_to_name[x] for x in structure_ids])

    # from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
    # mcc = MouseConnectivityCache(resolution=100)
    # structure_tree = mcc.get_structure_tree()
    # summary_structures = structure_tree.get_structures_by_set_id([688152357])
    # acronym = [x['acronym'] for x in summary_structures]
    # names = [x['name'] for x in summary_structures]
    # for i in zip(acronym, names): print(i)
