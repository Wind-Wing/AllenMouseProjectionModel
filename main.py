import time
import matplotlib.pyplot as plt
import numpy as np
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from model.global_model import GlobalModel
from utils.experiment import Experiment


def calc_cortex_regional_projection_matrix():
    mcc = MouseConnectivityCache(resolution=100)

    structure_tree = mcc.get_structure_tree()
    cortex_structures = structure_tree.get_structures_by_set_id([688152357])
    cortex_region_ids = [x['id'] for x in cortex_structures]
    print(cortex_region_ids)

    print("Getting experiences' ids")
    all_experiments = mcc.get_experiments(dataframe=False, injection_structure_ids=cortex_region_ids)
    print("Total %d experiences" % len(all_experiments))
    print("Loading and formating experience data")
    exp_formater = lambda exp: Experiment(exp, 100)
    exp_list = list(map(exp_formater, all_experiments))

    # TODO: fill out experience that is not mainly inject on cortex area, some injection will spread into sub-cortex area.
    global_model = GlobalModel()
    mat = global_model.get_regional_projection_matrix(cortex_region_ids, cortex_region_ids, exp_list)

    _time = time.time()
    _name = "max_mean-withoutNorm-%f" % _time
    np.savetxt("results/"+_name + ".txt", mat)
    labels = [x['acronym'] for x in cortex_structures]

    plt.imshow(mat, cmap=plt.cm.afmhot)
    plt.xticks(range(len(labels)), labels, rotation=60)
    plt.yticks(range(len(labels)), labels)
    fig = plt.gcf()
    fig.set_size_inches((10, 10), forward=False)
    fig.savefig("results/"+_name + ".png")
    plt.show()


def calc_cortex_region_projection_volume(structure_id):
    mcc = MouseConnectivityCache(resolution=100)

    structure_tree = mcc.get_structure_tree()
    cortex_structures = structure_tree.get_structures_by_set_id([688152357])
    cortex_region_ids = [x['id'] for x in cortex_structures]
    print(structure_id, cortex_region_ids)

    print("Getting experiences' ids")
    all_experiments = mcc.get_experiments(dataframe=False, injection_structure_ids=cortex_region_ids)
    print("Total %d experiences" % len(all_experiments))
    print("Loading and formating experience data")
    exp_formater = lambda exp: Experiment(exp, 100)
    exp_list = list(map(exp_formater, all_experiments))

    global_model = GlobalModel()
    volume = global_model.get_region_projection(structure_id, cortex_region_ids, exp_list)

    _time = time.time()
    _name = "max_mean-withoutNorm-projection_volume-%f" % _time
    np.save("results/" + _name + ".npy", volume)


if __name__ == "__main__":
    # calc_cortex_regional_projection_matrix()

    calc_cortex_region_projection_volume(184)
