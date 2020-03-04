from enum import Enum, unique


@unique
class Vectorizer(Enum):
    COUNT_VECTORIZER = 1
    HASHING_VECTORIZER = 2
    TF_IDF_VECTORIZER = 3
