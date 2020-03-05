'''
####################################
# Classification of text documents
####################################

This code uses many machine learning approaches to classify documents by topics using a bag-of-words approach.

The datasets used in this are the 20 news groups dataset (https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html) and the IMDB Reviews dataset (http://ai.stanford.edu/~amaas/data/sentiment/).
'''

import logging
import operator
import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.model_selection import cross_validate
from sklearn.utils.extmath import density

from argument_parser.argument_parser import get_options
from datasets.load_dataset import load_dataset
from feature_extraction.vectorizer import extract_text_features
from feature_selection.select_k_best import select_k_best_using_chi2
from model_selection.ml_algorithm_pair_list import get_ml_algorithm_pair_list
from utils.dataset_enum import Dataset
from utils.ml_classifiers_enum import Classifier
from utils.string_utils import trim


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
