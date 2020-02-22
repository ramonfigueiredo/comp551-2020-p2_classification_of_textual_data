from enum import Enum, unique


@unique
class Datasets(Enum):
    TWENTY_NEWS_GROUP_DATASET = 1
    IMDB_REVIEWS = 2
