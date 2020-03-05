import logging
from enum import Enum, unique


def validate_ml_list(ml_algorithm_list):
    ml_options = {classifier.name for classifier in Classifier}

    for ml in ml_algorithm_list:
        if ml not in ml_options:
            logging.error("Invalid ML algorithm name: {}. "
                          "You should provide one of the following ML algorithms names: {}".format(ml, ml_options))
            exit(0)


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
