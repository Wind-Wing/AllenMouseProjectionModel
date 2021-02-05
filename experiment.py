from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Experiment(object):
    def __init__(self, exp_dict, cache):
        self.id = exp_dict["id"]
        self.structure_id = exp_dict["structure_id"]
        self.structure_name = exp_dict["structure_name"]
        self.primary_injection_structure = exp_dict["primary_injection_structure"]
        self.injection_structures = exp_dict["injection_structures"]
        self.injection_volume = exp_dict["injection_volume"]
        self.injection_coordinate = (exp_dict["injection_x"], exp_dict["injection_y"], exp_dict["injection_z"])

        # Density - fraction of fluorescing pixels per voxel
        self.cache = cache
        self.valid_mask = self.cache.get_data_mask(self.id)[0]
        self.injection_fraction = self.cache.get_injection_fraction(self.id)[0]
        self.projection_density = self.cache.get_projection_density(self.id)[0]
        self.injection_density = self.cache.get_injection_density(self.id)[0]

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


def main():
    mcc = MouseConnectivityCache(resolution=100)
    all_experiments = mcc.get_experiments(dataframe=False)

    exp_formater = lambda exp: Experiment(exp, mcc)
    experience_list = list(map(exp_formater, all_experiments))


if __name__ == "__main__":
    main()
