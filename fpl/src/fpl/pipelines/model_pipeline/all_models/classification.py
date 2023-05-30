import sklearn
import xgboost
import lightgbm
from .model_utils import get_model_instance
from functools import partial

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


get_classification_model_instanc = partial(
    get_model_instance, available_models=classification_models
)
