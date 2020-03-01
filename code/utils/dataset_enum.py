from enum import Enum, unique


@unique
class Dataset(Enum):
    TWENTY_NEWS_GROUP = 1
    IMDB_REVIEWS = 2
