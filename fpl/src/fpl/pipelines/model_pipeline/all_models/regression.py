from typing import Any
import sklearn
from sklearn import kernel_ridge, neighbors, tree, ensemble, neural_network
import xgboost
import lightgbm
from .model_utils import get_model_instance

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


def get_regression_model_instance(
    model_id: str, model_params: dict[str, Any] = {}
) -> Any:
    return get_model_instance(
        model_id=model_id,
        model_params=model_params,
        available_models=regression_models,
    )
