'''
####################################
# Classification of text documents
####################################

This code uses many machine learning approaches to classify documents by topics using a bag-of-words approach.

The datasets used in this are the 20 news groups dataset (https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html) and the IMDB Reviews dataset (http://ai.stanford.edu/~amaas/data/sentiment/).
'''

import argparse
import logging
import multiprocessing
import operator
import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import cross_validate
from sklearn.utils.extmath import density

from datasets.load_dataset import load_twenty_news_groups, load_imdb_reviews
from model_selection import get_ml_algorithm_pair_list
from utils.dataset_enum import Dataset
from utils.ml_classifiers_enum import Classifier


def get_options():
    parser = argparse.ArgumentParser(prog='main.py',
                                     description='MiniProject 2: Classification of textual data. Authors: Ramon Figueiredo Pessoa, Rafael Gomes Braga, Ege Odaci',
                                     epilog='COMP 551 (001/002), Applied Machine Learning, Winter 2020, McGill University.')
    parser.add_argument("-d", "--dataset",
                        action="store", dest="dataset",
                        help="Dataset used (Options: TWENTY_NEWS_GROUP OR IMDB_REVIEWS). Default: ALL",
                        default='ALL')
    parser.add_argument("-ml", "--ml_algorithm_list",
                        action="append", dest="ml_algorithm_list",
                        help="List of machine learning algorithm to be executed. "
                             "This stores a list of ML algorithms, and appends each algorithm value to the list. "
                             "For example: -ml LINEAR_SVC -ml RANDOM_FOREST_CLASSIFIER, means "
                             "ml_algorithm_list = ['LINEAR_SVC', 'RANDOM_FOREST_CLASSIFIER']. "
                             "(Options of ML algorithms: "
                             "1) ADA_BOOST_CLASSIFIER, 2) BERNOULLI_NB, 3) COMPLEMENT_NB, 4) DECISION_TREE_CLASSIFIER, "
                             "5) EXTRA_TREE_CLASSIFIER, 6) EXTRA_TREES_CLASSIFIER, 7) GRADIENT_BOOSTING_CLASSIFIER, "
                             "8) K_NEIGHBORS_CLASSIFIER, 9) LINEAR_SVC, 10) LOGISTIC_REGRESSION, "
                             "11) LOGISTIC_REGRESSION_CV, 12) MLP_CLASSIFIER, 13) MULTINOMIAL_NB, 14) NEAREST_CENTROID, "
                             "15) NU_SVC, 16) PASSIVE_AGGRESSIVE_CLASSIFIER, 17) PERCEPTRON, "
                             "18) RANDOM_FOREST_CLASSIFIER, 19) RIDGE_CLASSIFIER, 20) RIDGE_CLASSIFIERCV, "
                             "21) SGD_CLASSIFIER,). "
                             "Default: None. If ml_algorithm_list = None, all ML algorithms will be executed.",
                        default=None)
    parser.add_argument("-use_default_parameters", "--use_classifiers_with_default_parameters",
                        action="store_true", default=False, dest="use_classifiers_with_default_parameters",
                        help="Use classifiers with default parameters. "
                             "Default: False = Use classifiers with best parameters found using grid search.")
    parser.add_argument("-not_shuffle", "--not_shuffle_dataset",
                        action="store_true", default=False, dest="not_shuffle_dataset",
                        help="Read dataset without shuffle data. Default: False")
    parser.add_argument("-n_jobs",
                        action="store", type=int, dest="n_jobs", default=-1,
                        help="The number of CPUs to use to do the computation. "
                             "If the provided number is negative or greater than the number of available CPUs, "
                             "the system will use all the available CPUs. Default: -1 (-1 == all CPUs)")
    parser.add_argument("-cv", "--run_cross_validation",
                        action="store_true", dest="run_cross_validation",
                        help="Run cross validation. Default: False")
    parser.add_argument("-n_splits",
                        action="store", type=int, dest="n_splits", default=5,
                        help="Number of cross validation folds. Default: 5. Must be at least 2. Default: 5")
    parser.add_argument("-use_5_classifiers", "--use_just_miniproject_classifiers",
                        action="store_true", dest="use_just_miniproject_classifiers",
                        help="Use just the miniproject classifiers (1. LogisticRegression, 2. DecisionTreeClassifier, "
                             "3. LinearSVC (L1), 4. LinearSVC (L2), 5. AdaBoostClassifier, 6. RandomForestClassifier). Default: False")
    parser.add_argument("-news_with_4_classes", "--twenty_news_using_four_categories",
                        action="store_true", default=False, dest="twenty_news_using_four_categories",
                        help="TWENTY_NEWS_GROUP dataset using some categories "
                             "('alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space'). "
                             "Default: False (use all categories). Default: False")
    parser.add_argument("-news_no_filter", "--twenty_news_with_no_filter",
                        action="store_true", default=False, dest="twenty_news_with_no_filter",
                        help="Do not remove newsgroup information that is easily overfit: "
                             "('headers', 'footers', 'quotes'). Default: False")
    parser.add_argument("-imdb_binary", "--use_imdb_binary_labels",
                        action="store_true", default=False, dest="use_imdb_binary_labels",
                        help="Use binary classification: 0 = neg and 1 = pos. If --use_imdb_binary_labels is False, "
                             "the system use IMDB multi-class labels (review score: 1, 2, 3, 4, 7, 8, 9, 10). Default: False")
    parser.add_argument("-show_reviews", "--show_imdb_reviews",
                        action="store_true", default=False, dest="show_imdb_reviews",
                        help="Show the IMDB_REVIEWS and respective labels while read the dataset. Default: False")
    parser.add_argument("-r", "--report",
                        action="store_true", dest="report",
                        help="Print a detailed classification report.")
    parser.add_argument("-m", "--all_metrics",
                        action="store_true", dest="all_metrics",
                        help="Print all classification metrics.")
    parser.add_argument("--chi2_select",
                        action="store", type=int, dest="chi2_select",
                        help="Select some number of features using a chi-squared test")
    parser.add_argument("-cm", "--confusion_matrix",
                        action="store_true", dest="print_cm",
                        help="Print the confusion matrix.")
    parser.add_argument("-top10", "--print_top10_terms",
                        action="store_true", default=False, dest="print_top10_terms",
                        help="Print ten most discriminative terms per class"
                             " for every classifier. Default: False")
    parser.add_argument("-use_hashing", "--use_hashing_vectorizer", dest="use_hashing",
                        action="store_true", default=False,
                        help="Use a hashing vectorizer. Default: False")
    parser.add_argument("-use_count", "--use_count_vectorizer", dest="use_count_vectorizer",
                        action="store_true", default=False,
                        help="Use a count vectorizer. Default: False")
    parser.add_argument("-n_features", "--n_features_using_hashing", dest="n_features",
                        action="store", type=int, default=2 ** 16,
                        help="n_features when using the hashing vectorizer. Default: 65536")
    parser.add_argument("-plot_time", "--plot_accurary_and_time_together",
                        action="store_true", default=False, dest="plot_accurary_and_time_together",
                        help="Plot training time and test time together with accuracy score. Default: False (Plot just accuracy)")
    parser.add_argument('-save_logs', '--save_logs_in_file', action='store_true', default=False,
                        dest='save_logs_in_file',
                        help='Save logs in a file. Default: False (show logs in the prompt)')
    parser.add_argument('-verbose', '--verbosity', action='store_true', default=False,
                        dest='verbose',
                        help='Increase output verbosity. Default: False')
    parser.add_argument("-random_state",
                        action="store", type=int, dest="random_state", default=0,
                        help="Seed used by the random number generator. Default: 0")
    parser.add_argument('-v', '--version', action='version', dest='version', version='%(prog)s 1.0')

    return parser.parse_args(), parser


def show_option(options, parser):
    print('=' * 130)
    print(parser.description)

    print('\nRunning with options: ')
    print('\tDataset =', options.dataset)
    print('\tML algorithm list (If ml_algorithm_list = None, all ML algorithms will be executed) =',
          options.ml_algorithm_list)
    print('\tUse classifiers with default parameters. '
          'Default: False = Use classifiers with best parameters found using grid search.',
          options.use_classifiers_with_default_parameters)
    print('\tRead dataset without shuffle data =', options.not_shuffle_dataset)
    print('\tThe number of CPUs to use to do the computation. '
          'If the provided number is negative or greater than the number of available CPUs, '
          'the system will use all the available CPUs. Default: -1 (-1 == all CPUs) =', options.n_jobs)
    print('\tRun cross validation. Default: False =', options.run_cross_validation)
    print('\tNumber of cross validation folds. Default: 5 =', options.n_splits)
    print('\tUse just the miniproject classifiers (1. LogisticRegression, 2. DecisionTreeClassifier, '
          '3. LinearSVC, 4. AdaBoostClassifier, 5. RandomForestClassifier) = ',
          options.use_just_miniproject_classifiers)
    print(
        '\tTWENTY_NEWS_GROUPS dataset using some categories (alt.atheism, talk.religion.misc, comp.graphics, sci.space) =',
        options.twenty_news_using_four_categories)
    print('\tDo not remove newsgroup information that is easily overfit (headers, footers, quotes) =',
          options.twenty_news_with_no_filter)
    print('\tUse IMDB Binary Labels (Negative / Positive) =', options.use_imdb_binary_labels)
    print('\tShow the IMDB_REVIEWS and respective labels while read the dataset =', options.show_imdb_reviews)
    print('\tPrint Classification Report =', options.report)
    print('\tPrint all classification metrics = ', options.all_metrics)
    print('\tSelect some number of features using a chi-squared test =', options.chi2_select)
    print('\tPrint the confusion matrix =', options.print_cm)
    print('\tPrint ten most discriminative terms per class for every classifier =', options.print_top10_terms)
    print('\tUse a hashing vectorizer =', options.use_hashing)
    print('\tUse a count vectorizer =', options.use_count_vectorizer)
    print('\tUse a tf-idf vectorizer =', (not options.use_hashing and not options.use_count_vectorizer))
    print('\tN features when using the hashing vectorizer =', options.n_features)
    print('\tPlot training time and test time together with accuracy score =', options.plot_accurary_and_time_together)
    print('\tSave logs in a file =', options.save_logs_in_file)
    print('\tSeed used by the random number generator (random_state) =', options.random_state)
    print('\tVerbose =', options.verbose)
    print('=' * 130)
    print()


def pre_process_options():
    if options.n_jobs > multiprocessing.cpu_count() or (options.n_jobs != -1 and options.n_jobs < 1):
        options.n_jobs = -1  # use all available cpus
    if options.save_logs_in_file:
        if not os.path.exists('logs'):
            os.mkdir('logs')
        logging.basicConfig(filename='logs/all.log', format='%(asctime)s - %(levelname)s - %(message)s',
                            level=logging.INFO, datefmt='%m/%d/%Y %I:%M:%S %p')
    else:
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO,
                            datefmt='%m/%d/%Y %I:%M:%S %p')
    dataset = options.dataset
    dataset = dataset.upper().strip()
    shuffle = (not options.not_shuffle_dataset)

    return options, dataset, shuffle


def load_dataset(dataset):
    if dataset == Dataset.TWENTY_NEWS_GROUPS.name:

        X_test, X_train, data_train, y_test, y_train = load_twenty_news_group_dataset()

    elif dataset == Dataset.IMDB_REVIEWS.name:

        X_test, X_train, y_test, y_train = load_imdb_reviews_dataset()
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


def load_imdb_reviews_dataset():
    print("Loading {} dataset:".format(Dataset.IMDB_REVIEWS.name))
    X_train, y_train = load_imdb_reviews(subset='train', binary_labels=options.use_imdb_binary_labels,
                                         verbose=options.show_imdb_reviews, shuffle=shuffle,
                                         random_state=options.random_state)
    X_test, y_test = load_imdb_reviews(subset='test', binary_labels=options.use_imdb_binary_labels,
                                       verbose=options.show_imdb_reviews, shuffle=shuffle,
                                       random_state=options.random_state)
    return X_test, X_train, y_test, y_train


def load_twenty_news_group_dataset():
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

    data_train = load_twenty_news_groups(subset='train', categories=categories, shuffle=shuffle, random_state=0,
                                         remove=remove)

    data_test = load_twenty_news_groups(subset='test', categories=categories, shuffle=shuffle,
                                        random_state=options.random_state,
                                        remove=remove)
    X_train, y_train = data_train.data, data_train.target
    X_test, y_test = data_test.data, data_test.target

    return X_test, X_train, data_train, y_test, y_train


def extracting_features(X_train, X_test):
    print("Extracting features from the training data using a vectorizer")
    t0 = time()
    if options.use_hashing:
        vectorizer = HashingVectorizer(stop_words='english', strip_accents='unicode', analyzer='word', binary=True)
        X_train = vectorizer.transform(X_train)
    elif options.use_count_vectorizer:
        vectorizer = CountVectorizer(stop_words='english', strip_accents='unicode', analyzer='word', binary=True)
        X_train = vectorizer.fit_transform(X_train)
    else:
        vectorizer = TfidfVectorizer(stop_words='english', strip_accents='unicode', analyzer='word', binary=True)
        X_train = vectorizer.fit_transform(X_train)
    duration = time() - t0
    print("done in %fs at %0.3fMB/s" % (duration, data_train_size_mb / duration))
    print("n_samples: %d, n_features: %d" % X_train.shape)
    print()

    print("Extracting features from the test data using the same vectorizer")
    t0 = time()
    X_test = vectorizer.transform(X_test)
    duration = time() - t0
    print("done in %fs at %0.3fMB/s" % (duration, data_test_size_mb / duration))
    print("n_samples: %d, n_features: %d" % X_test.shape)
    print()

    return vectorizer, X_train, X_test


def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."


def options_select_chi2(X_train, X_test, feature_names):
    if options.chi2_select:
        print("Extracting %d best features using the chi-squared test" %
              options.chi2_select)
        t0 = time()
        ch2 = SelectKBest(chi2, k=options.chi2_select)
        X_train = ch2.fit_transform(X_train, y_train)
        X_test = ch2.transform(X_test)
        if feature_names:
            # keep selected feature names
            feature_names = [feature_names[i] for i
                             in ch2.get_support(indices=True)]
        print("done in %fs" % (time() - t0))
        print()

    if feature_names:
        feature_names = np.asarray(feature_names)

    return X_train, X_test, feature_names


def benchmark(clf, classifier_enum, X_train, y_train, X_test, y_test):
    train_time = train_model(X_train, clf, y_train)

    test_time, y_pred = predict_using_model(X_test, clf)

    score = show_accuracy_score(y_pred, y_test)

    if options.run_cross_validation:
        print("\n\ncross validation:")
        scoring = ['accuracy', 'precision_macro', 'precision_micro', 'precision_weighted', 'recall_macro',
                   'recall_micro', 'recall_weighted', 'f1_macro', 'f1_micro', 'f1_weighted', 'jaccard_macro']
        cross_val_scores = cross_validate(clf, X_train, y_train, scoring=scoring, cv=options.n_splits,
                                          n_jobs=options.n_jobs, verbose=options.verbose)

        cv_test_accuracy = cross_val_scores['test_accuracy']
        print("\taccuracy: {}-fold cross validation: {}".format(options.n_splits, cv_test_accuracy))
        cv_accuracy_score_mean_std = "%0.2f (+/- %0.2f)" % (cv_test_accuracy.mean(), cv_test_accuracy.std() * 2)
        print("\ttest accuracy: {}-fold cross validation accuracy: {}".format(options.n_splits,
                                                                              cv_accuracy_score_mean_std))

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

        if options.print_top10_terms and feature_names is not None and not options.use_imdb_binary_labels:
            print("top 10 keywords per class:")
            for i, label in enumerate(target_names):
                top10 = np.argsort(clf.coef_[i])[-10:]
                print(trim("%s: %s" % (label, " ".join(feature_names[top10]))))
        print()

    options_classification_report(options, y_pred, y_test, target_names)

    options_show_all_metrics(options, y_pred, y_test)

    options_print_cm(options, y_pred, y_test)

    print()

    if options.run_cross_validation:
        return classifier_enum.name, score, train_time, test_time, cv_test_accuracy, cv_accuracy_score_mean_std
    else:
        return classifier_enum.name, score, train_time, test_time


def options_print_cm(options, y_pred, y_test):
    if options.print_cm:
        print("confusion matrix:")
        print(metrics.confusion_matrix(y_test, y_pred))


def options_show_all_metrics(options, y_pred, y_test):
    if options.all_metrics:
        print("\n\n===> Classification Metrics:\n")
        print('accuracy classification score')
        print('\taccuracy score: ', metrics.accuracy_score(y_test, y_pred))
        print('\taccuracy score (normalize=False): ', metrics.accuracy_score(y_test, y_pred, normalize=False))
        print()
        print('compute the precision')
        print('\tprecision score (average=macro): ', metrics.precision_score(y_test, y_pred, average='macro'))
        print('\tprecision score (average=micro): ', metrics.precision_score(y_test, y_pred, average='micro'))
        print('\tprecision score (average=weighted): ', metrics.precision_score(y_test, y_pred, average='weighted'))
        print('\tprecision score (average=None): ', metrics.precision_score(y_test, y_pred, average=None))
        print('\tprecision score (average=None, zero_division=1): ',
              metrics.precision_score(y_test, y_pred, average=None, zero_division=1))
        print()
        print('compute the precision')
        print('\trecall score (average=macro): ', metrics.recall_score(y_test, y_pred, average='macro'))
        print('\trecall score (average=micro): ', metrics.recall_score(y_test, y_pred, average='micro'))
        print('\trecall score (average=weighted): ', metrics.recall_score(y_test, y_pred, average='weighted'))
        print('\trecall score (average=None): ', metrics.recall_score(y_test, y_pred, average=None))
        print('\trecall score (average=None, zero_division=1): ',
              metrics.recall_score(y_test, y_pred, average=None, zero_division=1))
        print()
        print('compute the F1 score, also known as balanced F-score or F-measure')
        print('\tf1 score (average=macro): ', metrics.f1_score(y_test, y_pred, average='macro'))
        print('\tf1 score (average=micro): ', metrics.f1_score(y_test, y_pred, average='micro'))
        print('\tf1 score (average=weighted): ', metrics.f1_score(y_test, y_pred, average='weighted'))
        print('\tf1 score (average=None): ', metrics.f1_score(y_test, y_pred, average=None))
        print()
        print('compute the F-beta score')
        print('\tf beta score (average=macro): ', metrics.fbeta_score(y_test, y_pred, average='macro', beta=0.5))
        print('\tf beta score (average=micro): ', metrics.fbeta_score(y_test, y_pred, average='micro', beta=0.5))
        print('\tf beta score (average=weighted): ', metrics.fbeta_score(y_test, y_pred, average='weighted', beta=0.5))
        print('\tf beta score (average=None): ', metrics.fbeta_score(y_test, y_pred, average=None, beta=0.5))
        print()
        print('compute the average Hamming loss')
        print('\thamming loss: ', metrics.hamming_loss(y_test, y_pred))
        print()
        print('jaccard similarity coefficient score')
        print('\tjaccard score (average=macro): ', metrics.jaccard_score(y_test, y_pred, average='macro'))
        print('\tjaccard score (average=None): ', metrics.jaccard_score(y_test, y_pred, average=None))
        print()


def options_classification_report(options, y_pred, y_test, target_names):
    if options.report:
        print("\n\n===> Classification Report:\n")
        print(metrics.classification_report(y_test, y_pred, target_names=target_names))


def show_accuracy_score(y_pred, y_test):
    score = metrics.accuracy_score(y_test, y_pred)
    print("accuracy:   %0.3f" % score)

    return score


def predict_using_model(X_test, clf):
    t0 = time()

    y_pred = clf.predict(X_test)

    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    return test_time, y_pred


def train_model(X_train, clf, y_train):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()

    clf.fit(X_train, y_train)

    train_time = time() - t0

    print("train time: %0.3fs" % train_time)

    return train_time


def run_just_miniproject_classifiers(options, X_train, y_train, X_test, y_test,
                                     use_classifiers_with_default_parameters, dataset):
    ml_algorithm_list = [
        Classifier.ADA_BOOST_CLASSIFIER.name,
        Classifier.DECISION_TREE_CLASSIFIER.name,
        Classifier.LINEAR_SVC.name,
        Classifier.LOGISTIC_REGRESSION.name
    ]

    try:
        for clf, classifier_name in (
                get_ml_algorithm_pair_list(options, ml_algorithm_list, use_classifiers_with_default_parameters, dataset)
        ):
            print('=' * 80)
            print(classifier_name)
            results.append(benchmark(clf, classifier_name, X_train, y_train, X_test, y_test))
    except MemoryError as error:
        # Output expected MemoryErrors.
        logging.error(error)

    except Exception as exception:
        # Output unexpected Exceptions.
        logging.error(exception)

    return results


def validate_ml_list(ml_algorithm_list):
    ml_options = {classifier.name for classifier in Classifier}

    for ml in ml_algorithm_list:
        if ml not in ml_options:
            logging.error("Invalid ML algorithm name: {}. "
                          "You should provide one of the following ML algorithms names: {}".format(ml, ml_options))
            exit(0)


def run_all_classifiers(options, X_train, y_train, X_test, y_test, use_classifiers_with_default_parameters, dataset):
    ml_algorithm_list = {classifier.name for classifier in Classifier}

    try:
        for clf, classifier_name in (
                get_ml_algorithm_pair_list(options, ml_algorithm_list, use_classifiers_with_default_parameters, dataset)
        ):
            print('=' * 80)
            print(classifier_name)
            results.append(benchmark(clf, classifier_name, X_train, y_train, X_test, y_test))
    except MemoryError as error:
        # Output expected MemoryErrors.
        logging.error(error)

    except Exception as exception:
        # Output unexpected Exceptions.
        logging.error(exception)

    return results


def run_ml_algorithm_list(options, X_train, y_train, X_test, y_test, ml_algorithm_list,
                          use_classifiers_with_default_parameters, dataset):
    try:
        for clf, classifier_name in (
                get_ml_algorithm_pair_list(options, ml_algorithm_list, use_classifiers_with_default_parameters, dataset)
        ):
            print('=' * 80)
            print(classifier_name)
            results.append(benchmark(clf, classifier_name, X_train, y_train, X_test, y_test))

    except MemoryError as error:
        # Output expected MemoryErrors.
        logging.error(error)

    except Exception as exception:
        # Output unexpected Exceptions.
        logging.error(exception)

    return results


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


def show_final_classification_report(results, title):
    print("FINAL CLASSIFICATION TABLE: {}".format(title))

    classifier_name_list = results[0]
    accuracy_score_list = results[1]
    train_time_list = results[2]
    test_time_list = results[3]

    if options.run_cross_validation:
        cross_val_scores = results[4]
        cross_val_accuracy_score_mean_std = results[5]

    if options.run_cross_validation:
        print('| ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | '
              'Training time (seconds) | Test time (seconds) |')
        print(
            '| --- | ------------- | ------------------ | ------------------------------------ | ----------------- | '
            ' ------------------ | ------------------ |')
    else:
        print('| ID | ML Algorithm | Accuracy Score (%) | Training time (seconds) | Test time (seconds) |')
        print('| --- | ------------- | ------------------ | ----------------------- | ------------------- |')

    index = 1

    for classifier_name, accuracy_score, train_time, test_time in zip(classifier_name_list, accuracy_score_list,
                                                                      train_time_list, test_time_list):
        if classifier_name in ["Logistic Regression", "Decision Tree Classifier", "Linear SVC (penalty = L2)",
                               "Linear SVC (penalty = L1)", "Ada Boost Classifier", "Random forest"]:
            classifier_name = classifier_name + " [MANDATORY FOR COMP 551, ASSIGNMENT 2]"
        if options.run_cross_validation:
            print("|  {}  |  {}  |  {}  |  {}  |  {}  |  {}  |  {}  |".format(index, classifier_name,
                                                                              format(accuracy_score, ".2%"),
                                                                              cross_val_scores[index - 1],
                                                                              cross_val_accuracy_score_mean_std[
                                                                                  index - 1], format(train_time, ".4"),
                                                                              format(test_time, ".4")))
        else:
            print("|  {}  |  {}  |  {}  |  {}  |  {}  |".format(index, classifier_name,
                                                                format(accuracy_score, ".2%"),
                                                                format(train_time, ".4"),
                                                                format(test_time, ".4")))
        index = index + 1

    print("\n\nBest algorithm:")
    index_max_accuracy_score, accuracy_score = max(enumerate(accuracy_score_list), key=operator.itemgetter(1))
    print("===> {}) {}\n\t\tAccuracy score = {}\t\tTraining time = {}\t\tTest time = {}\n".format(
        index_max_accuracy_score + 1,
        classifier_name_list[index_max_accuracy_score],
        format(accuracy_score_list[index_max_accuracy_score], ".2%"),
        format(train_time_list[index_max_accuracy_score], ".4"),
        format(test_time_list[index_max_accuracy_score], ".4")))


if __name__ == '__main__':

    options, parser = get_options()

    options, dataset_option, shuffle = pre_process_options()

    show_option(options, parser)

    logging.info("Program started...")

    start = time()

    dataset_list = []

    if dataset_option == Dataset.TWENTY_NEWS_GROUPS.name:
        dataset_list.append(Dataset.TWENTY_NEWS_GROUPS.name)
    elif dataset_option == Dataset.IMDB_REVIEWS.name:
        dataset_list.append(Dataset.IMDB_REVIEWS.name)
    else:
        dataset_list.append(Dataset.TWENTY_NEWS_GROUPS.name)
        dataset_list.append(Dataset.IMDB_REVIEWS.name)

    for dataset in dataset_list:

        X_train, y_train, X_test, y_test, target_names, data_train_size_mb, data_test_size_mb = load_dataset(dataset)

        vectorizer, X_train, X_test = extracting_features(X_train, X_test)

        # mapping from integer feature name to original token string
        if options.use_hashing:
            feature_names = None
        else:
            feature_names = vectorizer.get_feature_names()

        options_select_chi2(X_train, X_test, feature_names)

        results = []
        if options.use_just_miniproject_classifiers:
            results = run_just_miniproject_classifiers(options, X_train, y_train, X_test, y_test,
                                                       options.use_classifiers_with_default_parameters, dataset)
        elif options.ml_algorithm_list:
            validate_ml_list(options.ml_algorithm_list)
            results = run_ml_algorithm_list(options, X_train, y_train, X_test, y_test, options.ml_algorithm_list,
                                            options.use_classifiers_with_default_parameters, dataset)
        else:
            results = run_all_classifiers(options, X_train, y_train, X_test, y_test,
                                          options.use_classifiers_with_default_parameters, dataset)

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

        show_final_classification_report(results, title)

    print('\n\nDONE!')

    print("Program finished. It took {} seconds".format(time() - start))
