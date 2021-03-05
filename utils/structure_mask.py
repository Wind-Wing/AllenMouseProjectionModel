from queue import Queue
import numpy as np
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from allensdk.api.queries.ontologies_api import OntologiesApi
from utils.constants import *


class StructureMask(object):
    def __init__(self):
        self.mcc = MouseConnectivityCache(resolution=RESOLUTION)
        self.structure_tree = self.mcc.get_structure_tree()
        self.annotations = self.mcc.get_annotation_volume()[0]

    def get_mask_idx(self, structure_id_list, hemisphere=BOTH_HEMISPHERE):
        mask = self.get_mask(structure_id_list, hemisphere)
        return mask.nonzero()

    def get_mask(self, structure_id_list, hemisphere=BOTH_HEMISPHERE):
        masks = [self._get_mask(x) for x in structure_id_list]
        mask = np.logical_or.reduce(masks)
        mask = self._get_hemisphere(mask, hemisphere)
        return mask

    def _get_mask(self, structure_id):
        fine_grained_id_list = self._get_fine_grained_annotation_list(structure_id)

        masks = [self.annotations == x for x in fine_grained_id_list]
        mask = np.logical_or.reduce(masks)

        if np.sum(mask) == 0:
            raise Exception("Structure id error: %d is not a valid id." % structure_id)
        return mask

    @staticmethod
    def _get_hemisphere(mask, hemisphere):
        middle_line = VOXEL_SHAPE[VERTICAL_PLANE_IDX] // 2
        if hemisphere == L_HEMISPHERE:
            mask[:, :, middle_line:] = 0
        elif hemisphere == R_HEMISPHERE:
            mask[:, :, :middle_line] = 0
        return mask

    def _get_fine_grained_annotation_list(self, structure_id):
        query_queue = Queue()
        query_queue.put(structure_id)

        annotation_id_list = set()
        while not query_queue.empty():
            _id = query_queue.get()
            annotation_id_list.add(_id)

            _child_ids = self.structure_tree.child_ids([_id])[0]
            for i in _child_ids:
                query_queue.put(i)
        return list(annotation_id_list)


if __name__ == "__main__":
    structure_mask = StructureMask()
    structure_tree = structure_mask.structure_tree

    cortex_structures = structure_tree.get_structures_by_set_id([688152357])
    print(cortex_structures)
    child = structure_tree.child_ids([526322264])[0]
    print(child)

    isocortex = structure_tree.get_structures_by_name(['Isocortex'])[0]
    print(isocortex)
    child = structure_tree.child_ids([isocortex['id']])[0]
    print(child)
    all_structures = structure_tree.get_structures_by_id(child)
    print(all_structures)

    # summary_structures = structure_tree.get_structures_by_set_id([167587189])
    # print(summary_structures)
    #
    # oapi = OntologiesApi()
    # structure_set_ids = structure_tree.get_structure_sets()
    # all_structure = oapi.get_structure_sets(structure_set_ids)
    # print(all_structure)
