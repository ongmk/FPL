from functools import partial

import lightgbm
import sklearn
import xgboost
from sklearn import ensemble, kernel_ridge, neighbors, neural_network, tree
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from fpl.pipelines.modelling.all_models.model_utils import get_model_instance

classification_models = {
    "lr": LogisticRegression,
    "knn": sklearn.neighbors._classification.KNeighborsClassifier,
    "nb": GaussianNB,
    "dt": sklearn.tree._classes.DecisionTreeClassifier,
    "svm": sklearn.linear_model._stochastic_gradient.SGDClassifier,
    "rbfsvm": sklearn.svm._classes.SVC,
    "gpc": GaussianProcessClassifier,
    "mlp": sklearn.neural_network._multilayer_perceptron.MLPClassifier,
    "ridge": sklearn.linear_model._ridge.RidgeClassifier,
    "rf": sklearn.ensemble._forest.RandomForestClassifier,
    "qda": QuadraticDiscriminantAnalysis,
    "ada": sklearn.ensemble._weight_boosting.AdaBoostClassifier,
    "gbc": sklearn.ensemble._gb.GradientBoostingClassifier,
    "lda": sklearn.discriminant_analysis.LinearDiscriminantAnalysis,
    "et": sklearn.ensemble._forest.ExtraTreesClassifier,
    "xgboost": xgboost.sklearn.XGBClassifier,
    "lightgbm": lightgbm.sklearn.LGBMClassifier,
    "dummy": sklearn.dummy.DummyClassifier,
}


get_classification_model_instance = partial(
    get_model_instance, available_models=classification_models
)
if __name__ == "__main__":
    from pycaret.containers.models.classification import (
        QuadraticDiscriminantAnalysisContainer,
    )
