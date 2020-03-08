import logging
from time import time

from sklearn.model_selection import cross_validate
from sklearn.utils.extmath import density

from metrics.ml_metrics import accuracy_score
from metrics.ml_metrics import print_classification_report
from metrics.ml_metrics import print_confusion_matrix
from metrics.ml_metrics import print_ml_metrics
from model_selection.ml_algorithm_pair_list import get_ml_algorithm_pair_list
from utils.ml_classifiers_enum import Classifier


def benchmark(clf, classifier_enum, X_train, y_train, X_test, y_test, options, feature_names, target_names):
    train_time = train_model(X_train, clf, y_train)

    test_time, y_pred = predict_using_model(X_test, clf)

    score = accuracy_score(y_pred, y_test)

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

        print()

    print_classification_report(options, y_pred, y_test, target_names)

    print_ml_metrics(options, y_pred, y_test)

    print_confusion_matrix(options, y_pred, y_test)

    print()

    if options.run_cross_validation:
        return classifier_enum.name, score, train_time, test_time, cv_test_accuracy, cv_accuracy_score_mean_std
    else:
        return classifier_enum.name, score, train_time, test_time


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
                                     use_classifiers_with_default_parameters,
                                     use_imdb_multi_class_labels, dataset,
                                     feature_names, target_names, results):
    ml_algorithm_list = [
        Classifier.ADA_BOOST_CLASSIFIER.name,
        Classifier.DECISION_TREE_CLASSIFIER.name,
        Classifier.LINEAR_SVC.name,
        Classifier.LOGISTIC_REGRESSION.name,
        Classifier.RANDOM_FOREST_CLASSIFIER
    ]

    try:
        for clf, classifier_name in (
                get_ml_algorithm_pair_list(options, ml_algorithm_list, use_classifiers_with_default_parameters,
                                           use_imdb_multi_class_labels, dataset)
        ):
            print('=' * 80)
            print(classifier_name)
            results.append(benchmark(clf, classifier_name, X_train, y_train, X_test, y_test,
                                     options, feature_names, target_names))
    except MemoryError as error:
        logging.error(error)

    except Exception as exception:
        logging.error(exception)

    return results


def run_all_classifiers(options, X_train, y_train, X_test, y_test, use_classifiers_with_default_parameters,
                        use_imdb_multi_class_labels, dataset,
                        feature_names, target_names, results):
    ml_algorithm_list = [classifier.name for classifier in Classifier]
    ml_algorithm_list.sort()

    try:
        for clf, classifier_name in (
                get_ml_algorithm_pair_list(options, ml_algorithm_list, use_classifiers_with_default_parameters,
                                           use_imdb_multi_class_labels, dataset)
        ):
            print('=' * 80)
            print(classifier_name)
            results.append(benchmark(clf, classifier_name, X_train, y_train, X_test, y_test,
                                     options, feature_names, target_names))
    except MemoryError as error:
        logging.error(error)

    except Exception as exception:
        logging.error(exception)

    return results


def run_ml_algorithm_list(options, X_train, y_train, X_test, y_test, ml_algorithm_list,
                          use_classifiers_with_default_parameters,
                          use_imdb_multi_class_labels, dataset,
                          feature_names, target_names, results):
    try:
        for clf, classifier_name in (
                get_ml_algorithm_pair_list(options, ml_algorithm_list, use_classifiers_with_default_parameters,
                                           use_imdb_multi_class_labels, dataset)
        ):
            print('=' * 80)
            print(classifier_name)
            results.append(benchmark(clf, classifier_name, X_train, y_train, X_test, y_test,
                                     options, feature_names, target_names))

    except MemoryError as error:
        logging.error(error)

    except Exception as exception:
        logging.error(exception)

    return results
