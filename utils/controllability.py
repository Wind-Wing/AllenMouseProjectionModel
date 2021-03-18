import os
import numpy as np
import matplotlib.pyplot as plt


class Controllability(object):
    def __init__(self):
        self.netctrl_path = "/home/wind/Desktop/DrivenNode/build/src/ui/netctrl"
        module_path = os.path.dirname(__file__)
        self.base_path = os.path.join(module_path, '../')
        self.tmp_path = self.base_path + "tmp.txt"

    def set_data(self, npy_relative_path, threshold=0., show_hist=False):
        file_path = self.base_path + npy_relative_path
        projection_mat = np.load(file_path)
        assert projection_mat.shape[1] == (projection_mat.shape[0] * 2)

        # TODO: only consider ipsilateral connection now. Extend to 2N * 2N mat when consider whole brain [[i, c], [c, i]]
        num_node = len(projection_mat)
        ipsilateral_mat = projection_mat[:, :num_node]

        if show_hist:
            self._show_histogram(ipsilateral_mat)

        edge_list = self._to_edge(ipsilateral_mat, threshold)
        self._save_to_text(edge_list)

    @staticmethod
    def _show_histogram(mat, num_bins=20):
        fig, ax = plt.subplots()
        n, bins, patches = ax.hist(mat.flatten(), num_bins)
        print(["{:.2%}".format(x) for x in (n / mat.size)])
        print(["{:.5f}".format(x) for x in bins])
        plt.show()

    def _to_edge(self, mat, threshold=0.):
        assert len(mat.shape) == 2
        num_col = int(mat.shape[0])
        num_row = int(mat.shape[1])
        assert num_row == num_col

        edge_list = []
        for i in range(num_col):
            for j in range(num_col):
                if mat[i][j] >= threshold:
                    edge_list.append([i, j])

        ratio = len(edge_list) / (num_col * num_row)
        print("Threshold {}, {:.2%} valid connection".format(threshold, ratio))
        return edge_list

    def _save_to_text(self, edge_list):
        lines = [f"{x[0]} {x[1]}\n" for x in edge_list]
        with open(self.tmp_path, "w") as f:
            f.writelines(lines)

    @property
    def driver_nodes(self):
        return self._run("driver_nodes")

    @property
    def control_paths(self):
        return self._run("control_paths")

    @property
    def statistics(self):
        return self._run("statistics")

    @property
    def significance(self):
        return self._run("significance")

    def _run(self, mode):
        results = os.system(f"{self.netctrl_path} --mode {mode} {self.tmp_path}")
        return results


if __name__ == "__main__":
    controllability = Controllability()
    threshold_list = [0.00021, 0.00042, 0.00064, 0.00085, 0.00106, 0.00127, 0.00148, 0.00169, 0.00191][:1]
    for thres in threshold_list:
        controllability.set_data("results/mean_mean-202103091517.npy", show_hist=True, threshold=thres)
        print(controllability.driver_nodes)
        # print(controllability.significance)
