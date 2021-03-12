import os
import numpy as np


class Controllability(object):
    def __init__(self):
        self.netctrl_path = "/home/wind/Desktop/DrivenNode/build/src/ui/netctrl"
        module_path = os.path.dirname(__file__)
        self.base_path = os.path.join(module_path, '../')
        self.tmp_path = self.base_path + "tmp.txt"

    def set_data(self, npy_relative_path, threshold=0):
        file_path = self.base_path + npy_relative_path
        projection_mat = np.load(file_path)
        assert projection_mat.shape[1] == (projection_mat.shape[0] * 2)

        # TODO: only consider ipsilateral connection now. Extend to 2N * 2N mat when consider whole brain [[i, c], [c, i]]
        num_node = len(projection_mat)
        ipsilateral_mat = projection_mat[:, :num_node]
        edge_list = self._to_edge(ipsilateral_mat, threshold)
        self._save_to_text(edge_list)

    def _to_edge(self, mat, threshold=0):
        assert len(mat.shape) == 2
        num_col = int(mat.shape[0])
        num_row = int(mat.shape[1])
        assert num_row == num_col

        edge_list = []
        for i in range(num_col):
            for j in range(num_col):
                if mat[i][j] >= threshold:
                    edge_list.append([i, j])
        return edge_list

    def _save_to_text(self, edge_list):
        lines = [f"{x[0]} {x[1]}\n" for x in edge_list]
        with open(self.tmp_path, "w") as f:
            f.writelines(lines)

    def statistics(self):
        results = os.system(f"{self.netctrl_path} --mode statistics {self.tmp_path}")
        print(results)


if __name__ == "__main__":
    controllability = Controllability()
    controllability.set_data("results/mean_mean-202103091517.npy")
    controllability.statistics()
