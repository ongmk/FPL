from functools import partial

import lightgbm
import sklearn
import xgboost
from sklearn import ensemble, kernel_ridge, neighbors, neural_network, tree

from fpl.pipelines.modelling.all_models.model_utils import (
    get_model_default_parameters,
    get_model_instance,
)

# References pycaret models()
regression_models = {
    "lr": sklearn.linear_model._base.LinearRegression,
    "lasso": sklearn.linear_model._coordinate_descent.Lasso,
    "ridge": sklearn.linear_model._ridge.Ridge,
    "en": sklearn.linear_model._coordinate_descent.ElasticNet,
    "lar": sklearn.linear_model._least_angle.Lars,
    "llar": sklearn.linear_model._least_angle.LassoLars,
    "omp": sklearn.linear_model._omp.OrthogonalMatchingPursuit,
    "br": sklearn.linear_model._bayes.BayesianRidge,
    "ard": sklearn.linear_model._bayes.ARDRegression,
    "par": sklearn.linear_model._passive_aggressive.PassiveAggressiveRegressor,
    "ransac": sklearn.linear_model._ransac.RANSACRegressor,
    "tr": sklearn.linear_model._theil_sen.TheilSenRegressor,
    "huber": sklearn.linear_model._huber.HuberRegressor,
    "kr": kernel_ridge.KernelRidge,
    "svm": sklearn.svm._classes.SVR,
    "knn": neighbors._regression.KNeighborsRegressor,
    "dt": tree._classes.DecisionTreeRegressor,
    "rf": ensemble._forest.RandomForestRegressor,
    "et": ensemble._forest.ExtraTreesRegressor,
    "ada": ensemble._weight_boosting.AdaBoostRegressor,
    "gbr": ensemble._gb.GradientBoostingRegressor,
    "mlp": neural_network._multilayer_perceptron.MLPRegressor,
    "xgboost": xgboost.sklearn.XGBRegressor,
    "lightgbm": lightgbm.sklearn.LGBMRegressor,
    "dummy": sklearn.dummy.DummyRegressor,
}

get_regression_model_instance = partial(
    get_model_instance, available_models=regression_models
)

get_regression_default_parameters = partial(
    get_model_default_parameters, available_models=regression_models
)


if __name__ == "__main__":
    from pprint import pprint

    # from pycaret.containers.models.regression import LightGBMContainer

    default_params = get_regression_default_parameters("xgboost")
    pprint(default_params)
