import os
import time
import numpy as np
import matplotlib.pyplot as plt
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from utils.structure_mask import StructureMask
from utils.experiment import Experiment
from utils.constants import *


class CortexModel(object):
    def __init__(self, interpolation_model):
        self.interpolation_model = interpolation_model

        cortex_structures = self._get_structures_info()
        self.cortex_region_ids = [x['id'] for x in cortex_structures]
        self.cortex_region_names = [x['acronym'] for x in cortex_structures]
        print("cortex region ids " + str(self.cortex_region_ids))

        self.structure_mask = StructureMask()
        self.cortex_mask_idx = self.structure_mask.get_mask_idx(self.cortex_region_ids)
        self.experiences = self._load_experiences(self.cortex_region_ids, self.cortex_mask_idx)

        module_path = os.path.dirname(__file__)
        base_path = os.path.join(module_path, '../')
        self.save_dir = base_path + "results/"

    @staticmethod
    def _get_structures_info():
        # TODO: fill out experience that is not mainly inject on cortex area, some injection will spread into sub-cortex area.
        mcc = MouseConnectivityCache(resolution=RESOLUTION)
        structure_tree = mcc.get_structure_tree()
        cortex_structures = structure_tree.get_structures_by_set_id([688152357])
        return cortex_structures

    def _load_experiences(self, injection_structure_ids, projection_structure_mask_idx):
        mcc = MouseConnectivityCache(resolution=RESOLUTION)

        print("Getting experiences' ids")
        all_experiments = mcc.get_experiments(dataframe=False, injection_structure_ids=injection_structure_ids)
        print("Total %d experiences" % len(all_experiments))
        print("Loading and formating experience data")
        exp_formater = lambda exp: Experiment(exp, projection_structure_mask_idx)
        exp_list = list(map(exp_formater, all_experiments))

        exp_list = self._flip_to_right_hemisphere(exp_list)
        return exp_list

    @staticmethod
    def _flip_to_right_hemisphere(exp_list):
        for exp in exp_list:
            if exp.hemisphere == 0:
                exp.flip()
        return exp_list

    def calc_regional_projection_matrix(self):
        r_region_masks_idx = [self.structure_mask.get_mask_idx([x], R_HEMISPHERE) for x in self.cortex_region_ids]
        l_region_masks_idx = [self.structure_mask.get_mask_idx([x], L_HEMISPHERE) for x in self.cortex_region_ids]

        ipsilateral_mat, = self.interpolation_model.get_regional_projection_matrix(
            r_region_masks_idx, r_region_masks_idx, self.experiences)
        contralateral_mat = self.interpolation_model.get_regional_projection_matrix(
            r_region_masks_idx, l_region_masks_idx, self.experiences)
        mat = np.concatenate([ipsilateral_mat, contralateral_mat], axis=1)

        _time = time.time()
        _name = "max_mean-%f" % _time
        np.save(self.save_dir + _name + ".npy", mat)
        labels = self.cortex_region_names

        plt.subplot(121)
        plt.imshow(ipsilateral_mat, cmap=plt.cm.afmhot)
        plt.xticks(range(len(labels)), labels, rotation=60)
        plt.yticks(range(len(labels)), labels)

        plt.subplot(122)
        plt.imshow(contralateral_mat, cmap=plt.cm.afmhot)
        plt.xticks(range(len(labels)), labels, rotation=60)
        plt.yticks(range(len(labels)), labels)

        fig = plt.gcf()
        fig.set_size_inches((20, 10), forward=False)
        fig.savefig(self.save_dir + _name + ".png")

    def calc_region_projection_volume(self):
        _time = time.time()
        r_region_masks_idx = [self.structure_mask.get_mask_idx([x], R_HEMISPHERE) for x in self.cortex_region_ids]
        for _id, idx in zip(self.cortex_region_ids, r_region_masks_idx):
            volume = self.interpolation_model.get_regional_projection_matrix(idx, self.experiences)
            _name = "max_mean-projection_volume%d-%f" % (_id, _time)
            np.save(self.save_dir + _name + ".npy", volume)
