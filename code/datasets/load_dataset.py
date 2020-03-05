import logging
import os
import random
from os import listdir
from os.path import isfile, join

from sklearn.datasets import fetch_20newsgroups

from utils.dataset_enum import Dataset


def load_dataset(dataset, options):
    if dataset == Dataset.TWENTY_NEWS_GROUPS.name:

        X_test, X_train, data_train, y_test, y_train = load_twenty_news_group_dataset(options)

    elif dataset == Dataset.IMDB_REVIEWS.name:

        X_test, X_train, y_test, y_train = load_imdb_reviews_dataset(options)
    else:
        logging.error("Loading dataset: Wrong dataset name = '{}'. Expecting: {} OR {}".format(dataset,
                                                                                               Dataset.TWENTY_NEWS_GROUPS.name,
                                                                                               Dataset.IMDB_REVIEWS.name))
        exit(0)

    print('data loaded')

    if dataset == Dataset.TWENTY_NEWS_GROUPS.name:
        # order of labels in `target_names` can be different from `categories`
        target_names = data_train.target_names
    else:
        # IMDB_REVIEWS dataset
        # If binary classification: 0 = neg and 1 = pos.
        # If multi-class classification use the review scores: 1, 2, 3, 4, 7, 8, 9, 10
        if options.use_imdb_binary_labels:
            target_names = ['0', '1']
        else:
            target_names = ['1', '2', '3', '4', '7', '8', '9', '10']

    def size_mb(docs):
        return sum(len(s.encode('utf-8')) for s in docs) / 1e6

    data_train_size_mb = size_mb(X_train)
    data_test_size_mb = size_mb(X_test)

    print("%d documents - %0.3fMB (training set)" % (
        len(X_train), data_train_size_mb))

    print("%d documents - %0.3fMB (test set)" % (
        len(X_test), data_test_size_mb))

    if dataset == Dataset.TWENTY_NEWS_GROUPS.name:
        print("%d categories" % len(target_names))
    print()

    return X_train, y_train, X_test, y_test, target_names, data_train_size_mb, data_test_size_mb


def load_imdb_reviews_dataset(options):
    print("Loading {} dataset:".format(Dataset.IMDB_REVIEWS.name))
    X_train, y_train = load_imdb_reviews(subset='train', binary_labels=options.use_imdb_binary_labels,
                                         verbose=options.show_imdb_reviews, shuffle=(not options.not_shuffle_dataset),
                                         random_state=options.random_state)
    X_test, y_test = load_imdb_reviews(subset='test', binary_labels=options.use_imdb_binary_labels,
                                       verbose=options.show_imdb_reviews, shuffle=(not options.not_shuffle_dataset),
                                       random_state=options.random_state)
    return X_test, X_train, y_test, y_train


def load_twenty_news_group_dataset(options):
    if options.twenty_news_using_four_categories:
        categories = [
            'alt.atheism',
            'talk.religion.misc',
            'comp.graphics',
            'sci.space',
        ]
    else:
        categories = None

    if options.twenty_news_with_no_filter:
        remove = ()
    else:
        remove = ('headers', 'footers', 'quotes')

    print("Loading {} dataset for categories:".format(Dataset.TWENTY_NEWS_GROUPS.name))

    data_train = load_twenty_news_groups(subset='train', categories=categories, shuffle=(not options.not_shuffle_dataset), random_state=0,
                                         remove=remove)

    data_test = load_twenty_news_groups(subset='test', categories=categories, shuffle=(not options.not_shuffle_dataset),
                                        random_state=options.random_state,
                                        remove=remove)
    X_train, y_train = data_train.data, data_train.target
    X_test, y_test = data_test.data, data_test.target

    return X_test, X_train, data_train, y_test, y_train


def load_twenty_news_groups(subset, categories=None, shuffle=True, random_state=None, remove=('headers', 'footers', 'quotes')):
    if subset not in ['train', 'test']:
        logging.error("load_twenty_news_groups: Wrong subset = '{}'. Expecting 'train' or 'test'".format(subset))
        exit(0)

    return fetch_20newsgroups(subset=subset, categories=categories, shuffle=shuffle, random_state=random_state, remove=remove)


def load_imdb_reviews(subset, binary_labels=False, verbose=False, shuffle=True, random_state=0, db_parent_path=None):
    X = []
    y = []
    dataset = {}

    subset = subset.lower().strip()
    if subset not in ['train', 'test']:
        logging.error("load_imdb_reviews: Wrong subset = '{}'. Expecting 'train' or 'test'".format(subset))
        exit(0)

    '''
    If binary classification: 0 = neg and 1 = pos
    Otherwise, it is multi class classification (classes: 1, 2, 3, 4, 7, 8, 9, 10)
    
    See IMDB README file
    * a negative review has a score <= 4 out of 10,
    * and a positive review has a score >= 7 out of 10
    * reviews with more neutral ratings are not included in the train/test sets
    '''
    # label = 0, if neg
    # label = 1, if pos
    subfolders = ['neg', 'pos']

    for folder in subfolders:

        if db_parent_path == None:
            path = os.path.join(os.getcwd(), 'datasets/imdb_reviews/aclImdb', subset, folder)
        else:
            path = os.path.join(db_parent_path, 'datasets/imdb_reviews/aclImdb', subset, folder)

        files_list = [f for f in listdir(path) if isfile(join(path, f))]

        print('\n===> Reading files from {}'.format(path))
        file_id = 1
        for f in files_list:

            if binary_labels:
                if folder == 'neg':
                    label = 0
                else:
                    label = 1
            else:
                label = f[f.index('_')+1:f.index('.')]

            file_path = os.path.join(path, f)
            if verbose:
                print("{}) File path = {}".format(file_id, file_path))
            with open(file_path, "r") as file:

                key = str(file_id) + "#" + str(label)
                file_text = file.read()

                dataset[key] = file_text

                if not shuffle:
                    X.append(file_text)
                    y.append(label)

                if verbose:
                    print("Review:\n", file_text)
                    print("Label:\n", label)
                file_id = file_id + 1

    if shuffle:
        random.seed(random_state)

        keys = list(dataset.keys())
        random.shuffle(keys)

        shuffled_dataset = dict()
        for key in keys:
            shuffled_dataset.update({key: dataset[key]})

        X = []
        for file_text in shuffled_dataset.values():
            X.append(file_text)

        y = []
        for key in shuffled_dataset.keys():
            label = key[key.index('#')+1:]
            y.append(int(label))

        return X, y
    else:
        return X, y
