import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


# TODO: abstract connection mat as a class
class GraphStaticAnalysis(object):
    def __init__(self, npy_relative_path, threshold=None, show_data=False):
        module_path = os.path.dirname(__file__)
        self.base_path = os.path.join(module_path, '../')
        self.graph = self._load_data(npy_relative_path, threshold, show_data)

    def _load_data(self, npy_relative_path, threshold, show_data):
        file_path = self.base_path + npy_relative_path
        mat = np.load(file_path)
        num_node = len(mat)
        mat = mat[:, :num_node]

        if show_data:
            self._plot_edge_weights(mat)

        if threshold is not None:
            mat = mat * (mat >= threshold)

        graph = nx.DiGraph(mat)

        if show_data:
            nx.draw(graph)
            plt.show()
            plt.clf()

        return graph

    @staticmethod
    def _plot_edge_weights(mat):
        weights = np.sort(mat.flatten())
        plt.plot(weights)
        plt.show()
        plt.clf()

    @property
    def degree(self):
        return self.graph.degree

    @property
    def betweenness_centrality(self):
        return nx.betweenness_centrality(self.graph)

    @property
    def modularity(self):
        return nx.directed_modularity_matrix(self.graph)


if __name__ == "__main__":
    analysis = GraphStaticAnalysis("results/mean_mean-202103091517.npy", threshold=0.0002, show_data=False)
    print(analysis.degree)
    print(analysis.betweenness_centrality)
    print(analysis.modularity)
