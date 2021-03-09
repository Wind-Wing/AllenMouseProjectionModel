import pytest
import os
import numpy as np
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from utils.structure_mask import StructureMask
from utils.constants import *


def test_hemisphere():
    os.chdir("..")
    structure_mask = StructureMask()

    r_mask = structure_mask.get_mask([961], R_HEMISPHERE)
    assert np.sum(r_mask[:, :, :VOXEL_SHAPE[VERTICAL_PLANE_IDX] // 2]) == 0

    l_mask = structure_mask.get_mask([961], L_HEMISPHERE)
    assert np.sum(l_mask[:, :, VOXEL_SHAPE[VERTICAL_PLANE_IDX] // 2:]) == 0

    mask = structure_mask.get_mask([961])
    assert np.sum(mask) == (np.sum(r_mask) + np.sum(l_mask))


def test_multi_structures():
    os.chdir("..")
    mcc = MouseConnectivityCache(resolution=RESOLUTION)
    structure_tree = mcc.get_structure_tree()
    cortex_structures = structure_tree.get_structures_by_set_id([688152357])
    cortex_region_ids = [x['id'] for x in cortex_structures]

    structure_mask = StructureMask()
    masks = [structure_mask.get_mask([_id]) for _id in cortex_region_ids]
    mask = structure_mask.get_mask(cortex_region_ids)
    assert np.sum(mask) == np.sum(masks)


def test_high_level_structure():
    os.chdir("..")
    structure_mask = StructureMask()
    mask = structure_mask.get_mask([500])
    sub_masks = structure_mask.get_mask([985, 993])
    assert np.sum(mask) == np.sum(sub_masks)
