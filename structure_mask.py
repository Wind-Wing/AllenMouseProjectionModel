from queue import Queue
import numpy as np
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache


class StructureMask(object):
    def __init__(self):
        self.mcc = MouseConnectivityCache(resolution=100)
        self.structure_tree = self.mcc.get_structure_tree()
        self.annotations = self.mcc.get_annotation_volume()[0]

    def get_mask(self, structure_id_list):
        mask_list = []
        for id in structure_id_list:
            fine_grained_id_list = self._get_fine_grained_annotation_list(id)
            mask = None
            for i in fine_grained_id_list:
                _mask = (self.annotations == i)
                if mask is None:
                    mask = _mask
                else:
                    mask = np.logical_or(mask, _mask)

            if np.sum(mask) == 0:
                raise Exception("Structure id error: %d is not a valid id." % id)
            else:
                mask_list.append(mask)

        mask = mask_list[0]
        for m in mask_list[1:]:
            mask = np.logical_or(mask, m)
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
    summary_structures = structure_mask.structure_tree.get_structures_by_set_id([167587189])
    id_list = set([int(i['id']) for i in summary_structures])
    annotation_list = set(structure_mask.annotations.flatten())
    print(id_list.difference(annotation_list))
    print(annotation_list.difference(id_list))

