import os
import time
import seaborn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from utils.structure_mask import StructureMask
from utils.experiment import Experiment
from utils.constants import *


class CortexModel(object):
    def __init__(self, interpolation_model):
        self.interpolation_model = interpolation_model

        self.cortex_region_ids, self.cortex_region_names = self.get_structures_info()
        print("cortex region ids " + str(self.cortex_region_ids))

        self.structure_mask = StructureMask()
        self.cortex_mask_idx = self.structure_mask.get_mask_idx(self.cortex_region_ids)
        self.experiences = self._load_experiences(self.cortex_region_ids, self.cortex_mask_idx)

        module_path = os.path.dirname(__file__)
        base_path = os.path.join(module_path, '../')
        self.save_dir = base_path + "results/"

        self.source_aggregate_func = np.mean
        self.target_aggregate_func = np.mean

    @staticmethod
    def get_structures_info():
        # TODO: fill out experience that is not mainly inject on cortex area, some injection will spread into sub-cortex area.
        # TODO: distinguish experiences with different mice gene type.
        # TODO: drop experiences that injection in multi-major area.
        mcc = MouseConnectivityCache(resolution=RESOLUTION)
        structure_tree = mcc.get_structure_tree()
        cortex_structures = structure_tree.get_structures_by_set_id([688152357])
        cortex_region_ids = [x['id'] for x in cortex_structures]
        cortex_region_names = [x['acronym'] for x in cortex_structures]
        return cortex_region_ids, cortex_region_names

    @staticmethod
    def _load_experiences(injection_structure_ids, projection_structure_mask_idx):
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
            x, self.cortex_mask_idx, self.experiences, self.source_aggregate_func)
        regional_projection_vectors = [_get_vector(x) for x in r_region_masks_idx]

        ipsilateral_mat = self.interpolation_model.get_regional_projection_matrix(
            regional_projection_vectors, target_masks_idx=r_region_masks_idx, aggregate_func=self.target_aggregate_func)
        contralateral_mat = self.interpolation_model.get_regional_projection_matrix(
            regional_projection_vectors, target_masks_idx=l_region_masks_idx, aggregate_func=self.target_aggregate_func)

        self._save_results(ipsilateral_mat, contralateral_mat)

    def _save_results(self, ipsilateral_mat, contralateral_mat):
        _time = time.strftime("%Y%m%d%H%M")

        _name = "%s_%s-%s" % (self.source_aggregate_func.__name__, self.target_aggregate_func.__name__, _time)
        mat = np.concatenate([ipsilateral_mat, contralateral_mat], axis=1)
        np.save(self.save_dir + _name + ".npy", mat)

        # for log(x + e)
        epsilon = 1e-10
        mat += epsilon

        fig = plt.gcf()
        fig.set_size_inches((20, 10), forward=False)

        gs = gridspec.GridSpec(nrows=1, ncols=3, width_ratios=(0.49, 0.49, 0.02), wspace=0.01)
        heatmap_ax1 = fig.add_subplot(gs[0])
        heatmap_ax2 = fig.add_subplot(gs[1])
        cbar_ax = fig.add_subplot(gs[2])

        vmin = 1e-5
        vmax = 10 ** -2.5
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)
        seaborn.heatmap(ipsilateral_mat, ax=heatmap_ax1, cbar=False, cmap=plt.cm.CMRmap, norm=norm, vmin=vmin, vmax=vmax, alpha=0.8)
        seaborn.heatmap(contralateral_mat, ax=heatmap_ax2, cbar_ax=cbar_ax, yticklabels=False, cmap=plt.cm.CMRmap, norm=norm, vmin=vmin, vmax=vmax, alpha=0.8)
        heatmap_ax1.set_xticklabels(labels=self.cortex_region_names, rotation=60)
        heatmap_ax1.set_yticklabels(labels=self.cortex_region_names, rotation=0)
        heatmap_ax2.set_xticklabels(labels=self.cortex_region_names, rotation=60)

        image_path = self.save_dir + _name + ".png"
        fig.savefig(image_path)
        print("Save image to %s" % image_path)

    def calc_region_projection_volume(self):
        _time = time.strftime("%Y%m%d%H%M")
        _name = "%s_%s-%s" % (self.source_aggregate_func.__name__, self.target_aggregate_func.__name__, _time)
        r_region_masks_idx = [self.structure_mask.get_mask_idx([x], R_HEMISPHERE) for x in self.cortex_region_ids]
        for _id, idx in zip(self.cortex_region_ids, r_region_masks_idx):
            volume = self.interpolation_model.get_one_region_projection(
                idx, self.cortex_mask_idx, self.experiences, self.source_aggregate_func)
            np.save(self.save_dir + "%d-" % _id + _name + ".npy", volume)
