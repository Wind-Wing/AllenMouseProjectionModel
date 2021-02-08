# Refactor based on the Joseoh Knox's code
# Authors: Zishuo Zheng <windwing.me@gmail.com>
# Previous Authors: Joseph Knox <josephk@alleninstitute.org>
# License: Allen Institute Software License
from __future__ import division

import numpy as np
from scipy.sparse import issparse
from sklearn.utils.extmath import squared_norm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics.scorer import check_scoring
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.model_selection._search import _check_param_grid
from sklearn.utils import check_X_y


class NadarayaWatson(BaseEstimator, RegressorMixin):
    """NadarayaWatson Estimator.

    Parameters
    ----------
    gamma : float, default=None
        Gamma parameter for the RBF, laplacian, polynomial, exponential chi2
        and sigmoid kernels. Ignored by other kernels.
    """

    def __init__(self, x, y, gamma=None):
        self.kernel = "rbf"
        self.gamma = gamma
        self.x, self.y = self._valid_input_data(x, y)

    def _get_kernel(self, x, y=None):
        params = {"gamma": self.gamma}
        _kernel = pairwise_kernels(x, y, metric=self.kernel, filter_params=True, **params)

        # Normalizes kernel to have row sum == 1 if sum != 0
        factor = _kernel.sum(axis=1)
        # if kernel has finite support, do not divide by zero
        factor[factor == 0] = 1
        return _kernel / factor[:, np.newaxis]

    @staticmethod
    def _valid_input_data(x, y):
        # Convert datatod
        x, y = check_X_y(x, y, accept_sparse=("csr", "csc"), multi_output=True, y_numeric=True)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        return x, y

    def get_weights(self, x):
        """Return model weights."""
        k = self._get_kernel(x, self.x)
        return k

    def predict(self, x):
        """Predict using the Nadaraya Watson model.

        Parameters
        ----------
        x : array, shape (n_samples, n_features)
            Training data.

        Returns
        -------
        C : array, shape (n_samples,) or (n_samples, n_targets)
            Returns predicted values.
        """
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)

        w = self.get_weights(x)

        if issparse(self.y):
            # has to be of form sparse.dot(dense)
            # more efficient than w.dot( y_.toarray() )
            return self.y.T.dot(w.T).T
        return w.dot(self.y)


class NadarayaWatsonSearchParameters(NadarayaWatson):
    """NadarayaWatson Estimator with built in Leave-one-out cross validation.

    By default, it performs Leave-one-out cross validation efficiently, but
    can accept cv argument to perform arbitrary cross validation splits.

    Parameters
    ----------
    param_grid : dict or list of dictionaries
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values or a list of such dictionaries,
        in which case the grids spanned by each dictionary in the list are
        explored. This enables searching over any sequence of parameter settings.

    scoring : string, callable or None, optional, default: None
        A string (see sklearn.model_evaluation documentation) or a scorer
        callable object / function with signature
        ``scorer(estimator, x, y)``

    cv : int, cross-validation generator or an iterable, optional, default: None
        Determines the cross-validation splitting strategy. If None, perform
        efficient leave-one-out cross validation, else use
        sklearn.model_selection.GridSearchCV.

    Attributes
    ----------
    cv_scores_ : array, shape = (n_samples, ~len(param_grid))
        Cross-validation scores for each candidate parameter (if
        `store_cv_scores=True` and `cv=None`)

    best_score_ : float
        Mean cross-validated score of the best performing estimator.

    n_splits_ : int
        Number of cross-validation splits (folds/iterations)
"""

    def __init__(self, param_grid, scoring=None, cv=None, kernel="linear", gamma=None):
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv
        self.kernel = kernel
        self.gamma = gamma

    def _update_params(self, param_dict):
        for k, v in param_dict.items():
            setattr(self, k, v)

    def fit(self, x, y):
        """Fit Nadaraya Watson estimator.

        Parameters
        ----------
        x : array, shape (n_samples, n_features)
            Training data.

        y : array, shape (n_samples, n_features)
            Target values.

        Returns
        -------
        self : returns an instance of self
        """
        if self.cv is None:
            estimator = NadarayaWatsonLeaveOneOutCrossValidation(param_grid=self.param_grid, scoring=self.scoring)
            estimator.fit(x, y)
            self.best_score_ = estimator.best_score_
            self.n_splits_ = estimator.n_splits_
            best_params_ = estimator.best_params_
            if self.store_cv_scores:
                self.best_index_ = estimator.best_index_
                self.cv_scores_ = estimator.cv_scores_
        else:
            if self.store_cv_scores:
                raise ValueError("cv!=None and store_cv_score=True "
                                 "are incompatible")
            gs = GridSearchCV(NadarayaWatson(), self.param_grid,
                              cv=self.cv, scoring=self.scoring, refit=True)
            gs.fit(x, y)
            estimator = gs.best_estimator_
            self.n_splits_ = gs.n_splits_
            self.best_score_ = gs.best_score_
            best_params_ = gs.best_params_

        # set params for predict
        self._update_params(best_params_)

        # store data for predict
        self.x_ = x
        self.y_ = y

        return self


class NadarayaWatsonLeaveOneOutCrossValidation(NadarayaWatson):
    """Nadaraya watson with built-in Cross-Validation

    It allows efficient Leave-One-Out cross validation
    """
    def __init__(self, param_grid, scoring=None, store_cv_scores=False):
        self.param_grid = param_grid
        self.scoring = scoring
        self.store_cv_scores = store_cv_scores
        _check_param_grid(param_grid)

    @property
    def _param_iterator(self):
        return ParameterGrid(self.param_grid)

    def _errors_and_values_helper(self, K):
        """Helper function to avoid duplication between self._errors and
        self._values.

        fill diagonal with 0, renormalize
        """
        np.fill_diagonal(K, 0)
        S = self._normalize_kernel(K, overwrite=True)

        return S

    def _errors(self, K, y):
        """ mean((y - Sy)**2) = mean( ((I-S)y)**2 )"""
        S = self._errors_and_values_helper(K)

        # I - S (S has 0 on diagonal)
        S *= -1
        np.fill_diagonal(S, 1.0)

        mse = lambda x: squared_norm(x) / x.size
        return mse(S.dot(y))

    def _values(self, K, y):
        """ prediction """
        S = self._errors_and_values_helper(K)

        return S.dot(y)

    def fit(self, x, y):
        """Fit the model using efficient leave-one-out cross validation"""
        x, y = self._valid_input_data(x, y)

        candidate_params = list(self._param_iterator)

        scorer = check_scoring(self, scoring=self.scoring, allow_none=True)
        # error = scorer is None
        error = self.scoring is None

        if not error:
            # scorer wants an object to make predictions
            # but are already computed efficiently by _NadarayaWatsonCV.
            # This identity_estimator will just return them
            def identity_estimator():
                pass
            identity_estimator.predict = lambda y_pred: y_pred

        cv_scores = []
        for candidate in candidate_params:
            # NOTE: a bit hacky, find better way
            K = NadarayaWatson(**candidate)._get_kernel(x)
            if error:
                # NOTE: score not error!
                score = -self._errors(K, y)
            else:
                y_pred = self._values(K, y)
                score = scorer(identity_estimator, y, y_pred)
            cv_scores.append(score)

        self.n_splits_ = x.shape[0]
        self.best_index_ = np.argmax(cv_scores)
        self.best_score_ = cv_scores[self.best_index_]
        self.best_params_ = candidate_params[self.best_index_]
        if self.store_cv_scores:
            self.cv_scores_ = cv_scores

        return self


def test():
    n_samples, n_features = 10, 5
    np.random.seed(0)
    y = np.random.randn(n_samples)
    x = np.random.randn(n_samples, n_features)

    param_grid = [dict(gamma=np.logspace(-1, 1, 3))]
    reg = NadarayaWatsonSearchParameters(param_grid)
    reg.fit(x, y)


if __name__ == "__main__":
    # test()
    print(np.ones([6, 6]))
    print(pairwise_kernels(np.ones([6, 6])))
