from typing import Any
import sklearn
import sklearn.kernel_ridge
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
    "kr": sklearn.kernel_ridge.KernelRidge,
    "svm": sklearn.svm._classes.SVR,
    "knn": sklearn.neighbors._regression.KNeighborsRegressor,
    "dt": sklearn.tree._classes.DecisionTreeRegressor,
    "rf": sklearn.ensemble._forest.RandomForestRegressor,
    "et": sklearn.ensemble._forest.ExtraTreesRegressor,
    "ada": sklearn.ensemble._weight_boosting.AdaBoostRegressor,
    "gbr": sklearn.ensemble._gb.GradientBoostingRegressor,
    "mlp": sklearn.neural_network._multilayer_perceptron.MLPRegressor,
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
