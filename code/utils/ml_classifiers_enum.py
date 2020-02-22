from enum import Enum, unique


@unique
class Classifier(Enum):
    LOGISTIC_REGRESSION = 1
    DECISION_TREES = 2
    SUPPORT_VECTOR_MACHINES = 3
    ADA_BOOST = 4
    RANDOM_FOREST = 5
