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

        all_experiments = mcc.get_experiments(dataframe=False, injection_structure_ids=injection_structure_ids)
        print("Total %d experiences" % len(all_experiments))

        t0 = time.time()
        exp_formater = lambda exp: Experiment(exp, projection_structure_mask_idx, flip_to=R_HEMISPHERE)
        exp_list = list(map(exp_formater, all_experiments))
        print("Preprocessing data %f" % (time.time() - t0))
        return exp_list

    def calc_regional_projection_matrix(self):
        r_region_masks_idx = [self.structure_mask.get_mask_idx([x], R_HEMISPHERE) for x in self.cortex_region_ids]
        l_region_masks_idx = [self.structure_mask.get_mask_idx([x], L_HEMISPHERE) for x in self.cortex_region_ids]

        _get_vector = lambda x: self.interpolation_model.get_one_region_projection(
            x, self.cortex_mask_idx, self.experiences)
        regional_projection_vectors = [_get_vector(x) for x in r_region_masks_idx]

        ipsilateral_mat = self.interpolation_model.get_regional_projection_matrix(
            regional_projection_vectors, target_masks_idx=r_region_masks_idx)
        contralateral_mat = self.interpolation_model.get_regional_projection_matrix(
            regional_projection_vectors, target_masks_idx=l_region_masks_idx)

        self._save_results(ipsilateral_mat, contralateral_mat)

    def _save_results(self, ipsilateral_mat, contralateral_mat):
        _time = time.time()

        _name = "max_mean-%f" % _time
        mat = np.concatenate([ipsilateral_mat, contralateral_mat], axis=1)
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
        image_path = self.save_dir + _name + ".png"
        fig.savefig(image_path)
        print("Save image to %s" % image_path)

    def calc_region_projection_volume(self):
        _time = time.time()
        r_region_masks_idx = [self.structure_mask.get_mask_idx([x], R_HEMISPHERE) for x in self.cortex_region_ids]
        for _id, idx in zip(self.cortex_region_ids, r_region_masks_idx):
            flatten_volume = self.interpolation_model.get_one_region_projection(
                idx, self.cortex_region_ids, self.experiences)
            volume = np.zeros(shape=VOXEL_SHAPE)
            volume[self.cortex_mask_idx] = flatten_volume
            _name = "max_mean-projection_volume%d-%f" % (_id, _time)
            np.save(self.save_dir + _name + ".npy", volume)
