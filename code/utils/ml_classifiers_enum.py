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
    GRADIENT_BOOSTING_CLASSIFIER = 5
    K_NEIGHBORS_CLASSIFIER = 6
    LINEAR_SVC = 7
    LOGISTIC_REGRESSION = 8
    MULTINOMIAL_NB = 9
    NEAREST_CENTROID = 10
    PASSIVE_AGGRESSIVE_CLASSIFIER = 11
    PERCEPTRON = 12
    RANDOM_FOREST_CLASSIFIER = 13
    RIDGE_CLASSIFIER = 14
