import os
import numpy as np
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from retry import retry
from utils.constants import *


class Experiment(object):
    def __init__(self, exp_dict, projection_structure_mask_idx=None, flip_to=None, injection=None, projection=None):
        # self.id = exp_dict["id"]
        # self.structure_id = exp_dict["structure_id"]
        # self.structure_name = exp_dict["structure_name"]
        # self.injection_structures = exp_dict["injection_structures"]
        # self.primary_injection_structure = exp_dict["primary_injection_structure"]
        # self.injection_volume = exp_dict["injection_volume"]
        # self.injection_coordinate = (exp_dict["injection_x"], exp_dict["injection_y"], exp_dict["injection_z"])

        if injection is None and projection is None:
            injection_density, projection_density = self._fetch_data_from_server(exp_dict['id'])
        else:
            injection_density = injection
            projection_density = projection
        self.injection_centroid = self._calc_injection_centroid(injection_density)
        self.hemisphere = self._calc_hemisphere()

        # mask and injection do not need to be flipped
        projection_density, self.injection_centroid = self._flip(projection_density, self.injection_centroid, flip_to)
        self.normalized_projection_density = self._calc_normalized_projection_density(
            injection_density, projection_density, projection_structure_mask_idx)

    @staticmethod
    @retry()
    def _fetch_data_from_server(exp_id):
        # Projection density is superset of injection density.
        # Density - fraction of fluorescing pixels per voxel
        cache = MouseConnectivityCache(resolution=RESOLUTION)
        valid_mask = cache.get_data_mask(exp_id)[0]
        injection_density = cache.get_injection_density(exp_id)[0] * valid_mask
        projection_density = cache.get_projection_density(exp_id)[0] * valid_mask
        return injection_density, projection_density

    @staticmethod
    def _calc_injection_centroid(injection_density):
        # centroid = coords * weight / sum(weight)
        nnz = injection_density.nonzero()
        coords = np.vstack(nnz)
        return np.dot(coords, injection_density[nnz]) / injection_density.sum()

    def _calc_hemisphere(self):
        if self.injection_centroid[VERTICAL_PLANE_IDX] >= VOXEL_SHAPE[VERTICAL_PLANE_IDX] // 2:
            hemisphere = R_HEMISPHERE
        else:
            hemisphere = L_HEMISPHERE
        return hemisphere

    @staticmethod
    def _calc_normalized_projection_density(injection_density, projection_density, structure_mask_idx):
        normalized_projection_density = projection_density / np.sum(injection_density)
        if structure_mask_idx is not None:
            normalized_projection_density = normalized_projection_density[structure_mask_idx]
        return normalized_projection_density

    def _flip(self, projection_density, injection_centroid, flip_to):
        if (flip_to is not None) and (self.hemisphere != flip_to):
            projection_density = np.flip(projection_density, axis=2)

            vertical_plane_coordinate = injection_centroid[VERTICAL_PLANE_IDX]
            vertical_plane_max_coordinate = VOXEL_SHAPE[VERTICAL_PLANE_IDX] - 1
            injection_centroid[VERTICAL_PLANE_IDX] = vertical_plane_max_coordinate - vertical_plane_coordinate
        return projection_density, injection_centroid


def main():
    os.chdir("../")
    mcc = MouseConnectivityCache(resolution=100)
    structure_tree = mcc.get_structure_tree()
    cortex_structures = structure_tree.get_structures_by_set_id([688152357])
    cortex_region_ids = [x['id'] for x in cortex_structures]
    all_experiments = mcc.get_experiments(dataframe=False, injection_structure_ids=cortex_region_ids)

    exp_func = lambda exp: Experiment(exp)
    experience_list = list(map(exp_func, all_experiments))
    print(np.mean([x.hemisphere for x in experience_list]))


if __name__ == "__main__":
    main()
