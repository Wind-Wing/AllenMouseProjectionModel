from queue import Queue
from numpy import np
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache


class StructureMask(object):
    def __init__(self):
        self.mcc = MouseConnectivityCache(resolution=100)
        self.structure_tree = self.mcc.get_structure_tree()
        self.annotations = self.mcc.get_annotation_volume()[0]

    def get_mask(self, structure_id_list):
        fine_grained_structure_id_list = self._get_fine_grained_annotation_list(structure_id_list)
        mask = None
        for i in fine_grained_structure_id_list:
            _mask = (self.annotations == i)
            if mask is None:
                mask = _mask
            else:
                mask = np.logical_or(mask, _mask)
        return mask

    def _get_fine_grained_annotation_list(self, structure_id_list):
        query_queue = Queue()
        for i in structure_id_list:
            query_queue.put(i)

        annotation_id_list = set()
        while not query_queue.empty():
            _id = query_queue.get()
            _child_ids = self.structure_tree.child_ids([_id])[0]
            if len(_child_ids) == 0:
                annotation_id_list.add(_id)
            else:
                for i in _child_ids:
                    query_queue.put(i)
        return list(annotation_id_list)


if __name__ == "__main__":
    structure_mask = StructureMask()
