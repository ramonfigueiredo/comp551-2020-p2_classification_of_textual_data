'''
####################################
# Classification of text documents
####################################

This code uses many machine learning approaches to classify documents by topics using a bag-of-words approach.

The datasets used in this are the 20 news groups dataset (https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html) and the IMDB Reviews dataset (http://ai.stanford.edu/~amaas/data/sentiment/).
'''

import logging
import os
from time import time

import matplotlib.pyplot as plt
import numpy as np

from argument_parser.argument_parser import get_options
from datasets.load_dataset import load_dataset
from feature_extraction.vectorizer import extract_text_features
from feature_selection.select_k_best import select_k_best_using_chi2
from machine_learning.ml_algorithms import run_all_classifiers
from machine_learning.ml_algorithms import run_just_miniproject_classifiers
from machine_learning.ml_algorithms import run_ml_algorithm_list
from metrics.ml_metrics import print_final_classification_report
from utils.dataset_enum import Dataset
from utils.ml_classifiers_enum import validate_ml_list


def plot_results(dataset, options):
    plt.figure(figsize=(12, 8))
    title = ""
    if dataset == Dataset.TWENTY_NEWS_GROUPS.name:
        if options.twenty_news_with_no_filter:
            title = "{} dataset".format(Dataset.TWENTY_NEWS_GROUPS.name)
            plt.title()
        else:
            title = "{} dataset (removing headers signatures and quoting)".format(
                Dataset.TWENTY_NEWS_GROUPS.name)
            plt.title(title)


    elif dataset == Dataset.IMDB_REVIEWS.name:
        if options.use_imdb_binary_labels:
            imdb_classification_type = "Binary classification"
        else:
            imdb_classification_type = "Multi-class classification"

        title = "{} dataset ({})".format(Dataset.IMDB_REVIEWS.name, imdb_classification_type)
        plt.title(title)
    plt.barh(indices, score, .2, label="score", color='navy')
    if options.plot_accurary_and_time_together:
        plt.barh(indices + .3, training_time, .2, label="training time", color='c')
        plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
    plt.yticks(())
    plt.legend(loc='best')
    plt.subplots_adjust(left=.25)
    plt.subplots_adjust(top=.95)
    plt.subplots_adjust(bottom=.05)

    for i, c, s, tr, te in zip(indices, clf_names, score, training_time, test_time):
        plt.text(-.3, i, c)
        plt.text(tr / 2, i + .3, round(tr, 2), ha='center', va='center', color='white')
        plt.text(te / 2, i + .6, round(te, 2), ha='center', va='center', color='white')
        plt.text(s / 2, i, round(s, 2), ha='center', va='center', color='white')

    plt.tight_layout()
    plt.show()

    return title


if __name__ == '__main__':

    options = get_options()

    if options.save_logs_in_file:
        if not os.path.exists('logs'):
            os.mkdir('logs')
        logging.basicConfig(filename='logs/all.log', format='%(asctime)s - %(levelname)s - %(message)s',
                            level=logging.INFO, datefmt='%m/%d/%Y %I:%M:%S %p')
    else:
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO,
                            datefmt='%m/%d/%Y %I:%M:%S %p')

    logging.info("Program started...")

    start = time()

    dataset_list = []

    if options.dataset == Dataset.TWENTY_NEWS_GROUPS.name:
        dataset_list.append(Dataset.TWENTY_NEWS_GROUPS.name)
    elif options.dataset == Dataset.IMDB_REVIEWS.name:
        dataset_list.append(Dataset.IMDB_REVIEWS.name)
    else:
        dataset_list.append(Dataset.TWENTY_NEWS_GROUPS.name)
        dataset_list.append(Dataset.IMDB_REVIEWS.name)

    for dataset in dataset_list:

        X_train, y_train, X_test, y_test, target_names, data_train_size_mb, data_test_size_mb = load_dataset(dataset, options)

        vectorizer, X_train, X_test = extract_text_features(X_train, X_test, options, data_train_size_mb, data_test_size_mb)

        # mapping from integer feature name to original token string
        if options.use_hashing:
            feature_names = None
        else:
            feature_names = vectorizer.get_feature_names()

        select_k_best_using_chi2(X_train, y_train, X_test, feature_names, options)

        results = []
        if options.use_just_miniproject_classifiers:
            results = run_just_miniproject_classifiers(options, X_train, y_train, X_test, y_test,
                                                       options.use_classifiers_with_default_parameters, dataset,
                                                       feature_names, target_names, results)
        elif options.ml_algorithm_list:
            validate_ml_list(options.ml_algorithm_list)
            results = run_ml_algorithm_list(options, X_train, y_train, X_test, y_test, options.ml_algorithm_list,
                                            options.use_classifiers_with_default_parameters, dataset,
                                            feature_names, target_names, results)
        else:
            results = run_all_classifiers(options, X_train, y_train, X_test, y_test,
                                          options.use_classifiers_with_default_parameters, dataset,
                                          feature_names, target_names, results)

        indices = np.arange(len(results))

        if options.run_cross_validation:
            results = [[x[i] for x in results] for i in range(6)]
            clf_names, score, training_time, test_time, cross_val_scores, cross_val_accuracy_score_mean_std = results
        else:
            results = [[x[i] for x in results] for i in range(4)]
            clf_names, score, training_time, test_time = results

        training_time = np.array(training_time) / np.max(training_time)
        test_time = np.array(test_time) / np.max(test_time)

        title = plot_results(dataset, options)

        print_final_classification_report(options, results, title)

    print('\n\nDONE!')

    print("Program finished. It took {} seconds".format(time() - start))
