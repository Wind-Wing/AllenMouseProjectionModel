import numpy as np
from sklearn.model_selection import LeaveOneOut


# Use leave one out cross validation to grid search the best parameter
class GridSearch(object):
    def __init__(self, parameter_grids, target_function):
        self.parameter_grids = parameter_grids
        self.target_function = target_function
        self.best_parameters = None
        self.min_loss = None

    def search_parameter(self, inputs, labels):
        assert len(inputs) == len(labels)
        full_dataset = zip(inputs, labels)
        loo = LeaveOneOut()
        loo_datasets = loo.get_n_splits(full_dataset)

        for params in self.parameter_grids:
            losses = []
            for dataset in loo_datasets:
                x, y = dataset
                _pred = self.target_function(x)
                _loss = self._mse_loss(_pred, y)
                losses.append(_loss)
            mean_loss = np.mean(losses)
            self._update_best_parameters(params, mean_loss)

    # Pred - [NUM_Samples, NUM_Features]
    # Label - [NUM_Samples, NUM_Features]
    def _mse_loss(self, pred, label):
        assert pred.shape == label.shape
        _loss = np.mean((pred - label) ** 2, axis=0)
        return _loss

    def _update_best_parameters(self, params, loss):
        if loss < self.min_loss:
            self.min_loss = loss
            self.best_parameters = params


if __name__ == "__main__":
    pass
