from typing import Any
import sklearn
import xgboost
import lightgbm
from pycaret.classification import models, setup
from .model_utils import get_model_instance
import pandas as pd

classification_models = {
    "lr": sklearn.linear_model._logistic.LogisticRegression,
    "knn": sklearn.neighbors._classification.KNeighborsClassifier,
    "nb": sklearn.naive_bayes.GaussianNB,
    "dt": sklearn.tree._classes.DecisionTreeClassifier,
    "svm": sklearn.linear_model._stochastic_gradient.SGDClassifier,
    "rbfsvm": sklearn.svm._classes.SVC,
    "gpc": sklearn.gaussian_process._gpc.GaussianProcessClassifier,
    "mlp": sklearn.neural_network._multilayer_perceptron.MLPClassifier,
    "ridge": sklearn.linear_model._ridge.RidgeClassifier,
    "rf": sklearn.ensemble._forest.RandomForestClassifier,
    "qda": sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis,
    "ada": sklearn.ensemble._weight_boosting.AdaBoostClassifier,
    "gbc": sklearn.ensemble._gb.GradientBoostingClassifier,
    "lda": sklearn.discriminant_analysis.LinearDiscriminantAnalysis,
    "et": sklearn.ensemble._forest.ExtraTreesClassifier,
    "xgboost": xgboost.sklearn.XGBClassifier,
    "lightgbm": lightgbm.sklearn.LGBMClassifier,
    "dummy": sklearn.dummy.DummyClassifier,
}


def get_classification_model_instance(
    model_id: str, model_params: dict[str, Any] = {}
) -> Any:
    return get_model_instance(
        model_id=model_id,
        model_params=model_params,
        available_models=classification_models,
    )
