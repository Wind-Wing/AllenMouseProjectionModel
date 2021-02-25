import matplotlib.pyplot as plt
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from global_model import GlobalModel
from experiment import Experiment


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
    regional_model = GlobalModel(cortex_region_ids, exp_list)
    mat = regional_model.get_regional_projection_matrix()
    return mat


if __name__ == "__main__":
    # mcc = MouseConnectivityCache(resolution=100)
    # structure_tree = mcc.get_structure_tree()
    # cortex_structures = structure_tree.get_structures_by_set_id([688152357])
    # cortex_region_ids = [cortex_structures[0]['id']]
    # all_experiments = mcc.get_experiments(dataframe=False, injection_structure_ids=cortex_region_ids)
    # a = mcc.get_structure_unionizes(experiment_ids=[all_experiments[0]['id']], structure_ids=cortex_region_ids)
    # print(a)

    mat = calc_cortex_regional_projection_matrix()
    plt.imshow(mat)
    plt.show()
