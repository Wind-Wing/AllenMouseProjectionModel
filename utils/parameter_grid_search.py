import numpy as np
from sklearn.model_selection import LeaveOneOut


# Use leave one out cross validation to grid search the best parameter
class GridSearch(object):
    def __init__(self, parameter_grids, set_func, training_func, predict_func):
        self.parameter_grids = parameter_grids
        self.set_func = set_func
        self.training_func = training_func
        self.predict_func = predict_func

        self.best_parameters = None
        self.min_loss = None

    def search_parameter(self, inputs, labels):
        assert len(inputs) == len(labels)

        for params in self.parameter_grids:
            t0 = time.time()
            self.set_func(params)
            _loss = self._cross_valid(inputs, labels)
            self._update_best_parameters(params, _loss)
            print(params, _loss, time.time() - t0)

    def _cross_valid(self, inputs, labels):
        inputs = np.array(inputs)
        labels = np.array(labels)

        loo = LeaveOneOut()
        loo_datasets = loo.split(inputs, labels)

        losses = []
        for train_idx, test_idx in loo_datasets:
            train_x = inputs[train_idx]
            train_y = labels[train_idx]
            test_x = inputs[test_idx]
            test_y = labels[test_idx]

            self.training_func(train_x, train_y)
            _pred = self.predict_func(test_x)
            _loss = self._mse_loss(_pred, test_y)
            losses.append(_loss)
        mean_loss = np.mean(losses)
        return mean_loss

    # Pred - [NUM_Samples, NUM_Features]
    # Label - [NUM_Samples, NUM_Features]
    def _mse_loss(self, pred, label):
        assert pred.shape == label.shape
        _losses = (pred - label) ** 2
        _loss = np.mean([np.sum(x) for x in _losses])
        return _loss

    def _update_best_parameters(self, params, loss):
        if (self.min_loss is None) or (loss < self.min_loss):
            self.min_loss = loss
            self.best_parameters = params


if __name__ == "__main__":
    import os
    import time
    from utils.experiment import Experiment
    from utils.structure_mask import StructureMask
    from model.nadaraya_watson_model import NadarayaWatsonModel
    from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
    from constants import *

    os.chdir("..")
    # Params
    params = [1. / (x*x*2) for x in range(4, 52, 2)]
    _model = NadarayaWatsonModel(0.013)

    # Read exp
    mcc = MouseConnectivityCache(resolution=RESOLUTION)
    structure_tree = mcc.get_structure_tree()
    cortex_structures = structure_tree.get_structures_by_set_id([688152357])
    cortex_region_ids = [x['id'] for x in cortex_structures]
    all_experiments = mcc.get_experiments(dataframe=False, injection_structure_ids=cortex_region_ids)
    print("Total %d experiences" % len(all_experiments))

    # Format exp
    t0 = time.time()
    structure_mask = StructureMask()
    cortex_mask_idx = structure_mask.get_mask_idx(cortex_region_ids)
    exp_formater = lambda exp: Experiment(exp, cortex_mask_idx, flip_to=R_HEMISPHERE)
    exp_list = list(map(exp_formater, all_experiments))
    print("Preprocessing data %f" % (time.time() - t0))

    coordinates = [x.injection_centroid for x in exp_list]
    projections = [x.normalized_projection_density for x in exp_list]

    grid_search = GridSearch(params, _model.set_gamma, _model.training, _model.predict)
    grid_search.search_parameter(coordinates, projections)
    print(grid_search.best_parameters)
