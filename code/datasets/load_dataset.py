import logging
import os
from os import listdir
from os.path import isfile, join
import random

from sklearn.datasets import fetch_20newsgroups


def load_twenty_news_groups(subset, categories=None, shuffle=True, random_state=None, remove=('headers', 'footers', 'quotes')):
    if subset not in ['train', 'test']:
        logging.error("load_twenty_news_groups: Wrong subset = '{}'. Expecting 'train' or 'test'".format(subset))
        exit(0)

    return fetch_20newsgroups(subset=subset, categories=categories, shuffle=shuffle, random_state=random_state, remove=remove)


def load_imdb_reviews(subset, binary_labels=False, verbose=False, shuffle=True, random_state=0):
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

        path = os.path.join(os.getcwd(), 'datasets/imdb_reviews/aclImdb', subset, folder)
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
