from enum import Enum, unique


def get_all_ml_classifiers_names():
    ml_algorithms_list = [
        Classifier.ADA_BOOST_CLASSIFIER.name,
        Classifier.BERNOULLI_NB.name,
        Classifier.COMPLEMENT_NB.name,
        Classifier.DECISION_TREE_CLASSIFIER.name,
        Classifier.EXTRA_TREE_CLASSIFIER.name,
        Classifier.EXTRA_TREES_CLASSIFIER.name,
        Classifier.GRADIENT_BOOSTING_CLASSIFIER.name,
        Classifier.K_NEIGHBORS_CLASSIFIER.name,
        Classifier.LINEAR_SVC.name,
        Classifier.LOGISTIC_REGRESSION.name,
        Classifier.LOGISTIC_REGRESSION_CV.name,
        Classifier.MLP_CLASSIFIER.name,
        Classifier.MULTINOMIAL_NB.name,
        Classifier.NEAREST_CENTROID.name,
        Classifier.NU_SVC.name,
        Classifier.PASSIVE_AGGRESSIVE_CLASSIFIER.name,
        Classifier.PERCEPTRON.name,
        Classifier.RANDOM_FOREST_CLASSIFIER.name,
        Classifier.RIDGE_CLASSIFIER.name,
        Classifier.RIDGE_CLASSIFIERCV.name,
        Classifier.SGD_CLASSIFIER.name
    ]

    return ml_algorithms_list


@unique
class Classifier(Enum):
    ADA_BOOST_CLASSIFIER = 1
    BERNOULLI_NB = 2
    COMPLEMENT_NB = 3
    DECISION_TREE_CLASSIFIER = 4
    EXTRA_TREE_CLASSIFIER = 5
    EXTRA_TREES_CLASSIFIER = 6
    GRADIENT_BOOSTING_CLASSIFIER = 7
    K_NEIGHBORS_CLASSIFIER = 8
    LINEAR_SVC = 9
    LOGISTIC_REGRESSION = 10
    LOGISTIC_REGRESSION_CV = 11
    MLP_CLASSIFIER = 12
    MULTINOMIAL_NB = 13
    NEAREST_CENTROID = 14
    NU_SVC = 15
    PASSIVE_AGGRESSIVE_CLASSIFIER = 16
    PERCEPTRON = 17
    RANDOM_FOREST_CLASSIFIER = 18
    RIDGE_CLASSIFIER = 19
    RIDGE_CLASSIFIERCV = 20
    SGD_CLASSIFIER = 21
