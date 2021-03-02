import os
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
import numpy as np
import matplotlib.pyplot as plt
from retry import retry


class Experiment(object):
    def __init__(self, exp_dict, resolution):
        self.id = exp_dict["id"]
        self.structure_id = exp_dict["structure_id"]
        self.structure_name = exp_dict["structure_name"]
        self.injection_structures = exp_dict["injection_structures"]
        self.primary_injection_structure = exp_dict["primary_injection_structure"]
        # self.injection_volume = exp_dict["injection_volume"]
        # self.injection_coordinate = (exp_dict["injection_x"], exp_dict["injection_y"], exp_dict["injection_z"])

        # Density - fraction of fluorescing pixels per voxel
        self.cache = MouseConnectivityCache(resolution=resolution)
        self._fetch_data_from_server()
        self.hemisphere = self.injection_centroid >= 57 # 0 left, 1 right

    @retry()
    def _fetch_data_from_server(self):
        # Projection density is superset of injection density.
        self.valid_mask = self.cache.get_data_mask(self.id)[0]
        # self.injection_fraction = self.cache.get_injection_fraction(self.id)[0]
        self.projection_density = self.cache.get_projection_density(self.id)[0] * self.valid_mask
        self.injection_density = self.cache.get_injection_density(self.id)[0] * self.valid_mask

    @property
    def injection_centroid(self):
        # centroid = coords * weight / sum(weight)
        nnz = self.injection_density.nonzero()
        coords = np.vstack(nnz)
        return np.dot(coords, self.injection_density[nnz]) / self.injection_density.sum()

    @property
    def normalized_projection_density(self):
        return self.projection_density / np.sum(self.injection_density)

    def visualize(self):
        plt.pcolor(self.projection_density[100, :, :])
        plt.show()
        plt.clf()

    # All experience will be flip to R domain hemisphere
    def flip(self):
        self.valid_mask = np.flip(self.valid_mask, axis=2)
        self.projection_density = np.flip(self.projection_density, axis=2)
        self.injection_density = np.flip(self.injection_density, axis=2)


def main():
    os.chdir("../")
    mcc = MouseConnectivityCache(resolution=100)
    structure_tree = mcc.get_structure_tree()
    cortex_structures = structure_tree.get_structures_by_set_id([688152357])
    cortex_region_ids = [x['id'] for x in cortex_structures]
    all_experiments = mcc.get_experiments(dataframe=False, injection_structure_ids=cortex_region_ids)

    exp_formater = lambda exp: Experiment(exp, 100)
    experience_list = list(map(exp_formater, all_experiments))
    centroid = [x.injection_centroid for x in experience_list]
    x = np.array([x[2] for x in centroid])
    print(np.sum(x >= 57))
    print(np.sum(x < 57))


if __name__ == "__main__":
    main()
