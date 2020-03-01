'''
####################################
# Classification of text documents
####################################

This code uses many machine learning approaches to classify documents by topics using a bag-of-words approach.

The datasets used in this are the 20 newsgroups dataset (https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html) and the IMDB Reviews dataset (http://ai.stanford.edu/~amaas/data/sentiment/).
'''

import argparse
import logging
import operator
import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.extmath import density
import multiprocessing

from datasets.load_dataset import load_twenty_news_groups, load_imdb_reviews

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='main.py',
        description='MiniProject 2: Classification of textual data. Authors: Ramon Figueiredo Pessoa, Rafael Gomes Braga, Ege Odaci',
        epilog='COMP 551 (001/002), Applied Machine Learning, Winter 2020, McGill University.')

    parser.add_argument("-d", "--dataset",
                        action="store", dest="dataset",
                        help="Dataset used (Options: 20news OR imdb). Default: imdb", default='imdb')

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
                        help="20 news groups dataset using some categories "
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
                        help="Show the IMDB reviews and respective labels while read the dataset. Default: False")

    parser.add_argument("-r", "--report",
                        action="store_true", dest="report",
                        help="Print a detailed classification report.")

    parser.add_argument("-m", "--all_metrics",
                        action="store_true", dest="all_metrics",
                        help="Print all classification metrics.")

    parser.add_argument("--chi2_select",
                        action="store", type=int, dest="select_chi2",
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

    parser.add_argument('-v', '--version', action='version', dest='version', version='%(prog)s 1.0')

    options = parser.parse_args()

    if options.n_jobs > multiprocessing.cpu_count() or (options.n_jobs != -1 and options.n_jobs < 1):
        options.n_jobs = -1 # use all available cpus

    print('=' * 130)
    print(parser.description)

    print('\nRunning with options: ')
    # print('\tClassifier =', options.classifier.upper())
    print('\tDataset =', options.dataset)
    print('\tRead dataset without shuffle data =', options.not_shuffle_dataset)
    print('\tThe number of CPUs to use to do the computation. '
          'If the provided number is negative or greater than the number of available CPUs, '
          'the system will use all the available CPUs. Default: -1 (-1 == all CPUs) =', options.n_jobs)
    print('\tRun cross validation. Default: False =', options.run_cross_validation)
    print('\tNumber of cross validation folds. Default: 5 =', options.n_splits)
    print('\tUse just the miniproject classifiers (1. LogisticRegression, 2. DecisionTreeClassifier, '
          '3. LinearSVC, 4. AdaBoostClassifier, 5. RandomForestClassifier) = ', options.use_just_miniproject_classifiers)
    print('\t20 news groups dataset using some categories (alt.atheism, talk.religion.misc, comp.graphics, sci.space) =',
          options.twenty_news_using_four_categories)
    print('\tDo not remove newsgroup information that is easily overfit (headers, footers, quotes) =',
          options.twenty_news_with_no_filter)
    print('\tUse IMDB Binary Labels (Negative / Positive) =', options.use_imdb_binary_labels)
    print('\tShow the IMDB reviews and respective labels while read the dataset =', options.show_imdb_reviews)
    print('\tPrint Classification Report =', options.report)
    print('\tPrint all classification metrics = ', options.all_metrics)
    print('\tSelect some number of features using a chi-squared test =', options.select_chi2)
    print('\tPrint the confusion matrix =', options.print_cm)
    print('\tPrint ten most discriminative terms per class for every classifier =', options.print_top10_terms)
    print('\tUse a hashing vectorizer =', options.use_hashing)
    print('\tN features when using the hashing vectorizer =', options.n_features)
    print('\tPlot training time and test time together with accuracy score =', options.plot_accurary_and_time_together)
    print('\tSave logs in a file =', options.save_logs_in_file)
    print('\tVerbose =', options.verbose)
    print('=' * 130)
    print()

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

    #######################################
    # Load data from the training set
    #######################################

    dataset = options.dataset
    dataset = dataset.lower().strip()

    shuffle = (not options.not_shuffle_dataset)

    if dataset == '20news':

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

        print("Loading 20 newsgroups dataset for categories:")

        data_train = load_twenty_news_groups(subset='train', categories=categories, shuffle=shuffle, random_state=0,
                                             remove=remove)
        data_test = load_twenty_news_groups(subset='test', categories=categories, shuffle=shuffle, random_state=0,
                                            remove=remove)

        X_train, y_train = data_train.data, data_train.target
        X_test, y_test = data_test.data, data_test.target

    elif dataset == 'imdb':

        print("Loading IMDB Reviews dataset:")

        X_train, y_train = load_imdb_reviews(subset='train', binary_labels=options.use_imdb_binary_labels,
                                             verbose=options.show_imdb_reviews, shuffle=shuffle, random_state=0)
        X_test, y_test = load_imdb_reviews(subset='test', binary_labels=options.use_imdb_binary_labels,
                                           verbose=options.show_imdb_reviews, shuffle=shuffle, random_state=0)
    else:
        logging.error("Loading dataset: Wrong dataset name = '{}'. Expecting: 20news OR imdb".format(dataset))
        exit(0)

    print('data loaded')

    if dataset == '20news':
        # order of labels in `target_names` can be different from `categories`
        target_names = data_train.target_names
    else:
        # IMDB reviews dataset
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
    if dataset == '20news':
        print("%d categories" % len(target_names))
    print()

    print("Extracting features from the training data using a sparse vectorizer")
    t0 = time()
    if options.use_hashing:
        vectorizer = HashingVectorizer(stop_words='english', alternate_sign=False,
                                       n_features=options.n_features)
        X_train = vectorizer.transform(X_train)
    else:
        vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                     stop_words='english')
        X_train = vectorizer.fit_transform(X_train)
    duration = time() - t0
    print("done in %fs at %0.3fMB/s" % (duration, data_train_size_mb / duration))
    print("n_samples: %d, n_features: %d" % X_train.shape)
    print()

    #######################################
    # Extracting features
    #######################################

    print("Extracting features from the test data using the same vectorizer")
    t0 = time()
    X_test = vectorizer.transform(X_test)
    duration = time() - t0
    print("done in %fs at %0.3fMB/s" % (duration, data_test_size_mb / duration))
    print("n_samples: %d, n_features: %d" % X_test.shape)
    print()

    # mapping from integer feature name to original token string
    if options.use_hashing:
        feature_names = None
    else:
        feature_names = vectorizer.get_feature_names()

    if options.select_chi2:
        print("Extracting %d best featureop.print_help()s by a chi-squared test" %
              options.select_chi2)
        t0 = time()
        ch2 = SelectKBest(chi2, k=options.select_chi2)
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


    def trim(s):
        """Trim string to fit on terminal (assuming 80-column display)"""
        return s if len(s) <= 80 else s[:77] + "..."

    ##############################################
    # Benchmark classifiers
    ##############################################

    def benchmark(clf, classifier_name, X_train, y_train, X_test, y_test):
        print('_' * 80)
        print("Training: ")
        print(clf)
        t0 = time()
        clf.fit(X_train, y_train)
        train_time = time() - t0
        print("train time: %0.3fs" % train_time)

        t0 = time()
        y_pred = clf.predict(X_test)
        test_time = time() - t0
        print("test time:  %0.3fs" % test_time)

        score = metrics.accuracy_score(y_test, y_pred)
        print("accuracy:   %0.3f" % score)

        if options.run_cross_validation:
            print("\n\ncross validation:")
            cross_val_scores = cross_val_score(clf, X_train, y_train, cv=options.n_splits, n_jobs=options.n_jobs, verbose=options.verbose)
            print("{}-fold cross validation: {}".format(options.n_splits, cross_val_scores))
            cross_val_accuracy_score_mean_std = "%0.2f (+/- %0.2f)" % (cross_val_scores.mean(), cross_val_scores.std() * 2)
            print("{}-fold cross validation accuracy: {}".format(options.n_splits, cross_val_accuracy_score_mean_std))

        if hasattr(clf, 'coef_'):
            print("dimensionality: %d" % clf.coef_.shape[1])
            print("density: %f" % density(clf.coef_))

            if options.print_top10_terms and feature_names is not None and not options.use_imdb_binary_labels:
                print("top 10 keywords per class:")
                for i, label in enumerate(target_names):
                    top10 = np.argsort(clf.coef_[i])[-10:]
                    print(trim("%s: %s" % (label, " ".join(feature_names[top10]))))
            print()

        if options.report:
            print("\n\n===> Classification Report:\n")
            print(metrics.classification_report(y_test, y_pred, target_names=target_names))

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
            print('\tprecision score (average=None, zero_division=1): ', metrics.precision_score(y_test, y_pred, average=None, zero_division=1))
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

        if options.print_cm:
            print("confusion matrix:")
            print(metrics.confusion_matrix(y_test, y_pred))

        print()
        # clf_descr = str(clf).split('(')[0]
        return classifier_name, score, train_time, test_time, cross_val_accuracy_score_mean_std


    results = []
    if options.use_just_miniproject_classifiers:
        for clf, classifier_name in (
                (LogisticRegression(), "Logistic Regression"),
                (DecisionTreeClassifier(), "Decision Tree Classifier"),
                (LinearSVC(penalty="l2", dual=False, tol=1e-3), "Linear SVC (penalty = L2)"),
                (AdaBoostClassifier(), "Ada Boost Classifier"),
                (RandomForestClassifier(), "Random forest")):
            print('=' * 80)
            print(classifier_name)
            results.append(benchmark(clf, classifier_name, X_train, y_train, X_test, y_test))
    else:
        for clf, classifier_name in (
                (RidgeClassifier(tol=1e-2, solver="sag"), "Ridge Classifier"),
                (Perceptron(max_iter=50), "Perceptron"),
                (PassiveAggressiveClassifier(max_iter=50), "Passive-Aggressive"),
                (KNeighborsClassifier(n_neighbors=10), "kNN"),
                (LogisticRegression(), "Logistic Regression"),
                (DecisionTreeClassifier(), "Decision Tree Classifier"),
                (LinearSVC(penalty="l2", dual=False, tol=1e-3), "Linear SVC (penalty = L2)"),
                (LinearSVC(penalty="l1", dual=False, tol=1e-3), "Linear SVC (penalty = L1)"),
                (SGDClassifier(alpha=.0001, max_iter=50, penalty="l2"), "SGD Classifier (penalty = L2)"),
                (SGDClassifier(alpha=.0001, max_iter=50, penalty="l2"), "SGD Classifier (penalty = L1)"),
                (AdaBoostClassifier(), "Ada Boost Classifier"),
                (RandomForestClassifier(), "Random forest")):
            print('=' * 80)
            print(classifier_name)
            results.append(benchmark(clf, classifier_name, X_train, y_train, X_test, y_test))

        # Train SGD with Elastic Net penalty
        print('=' * 80)
        print("SGDClassifier Elastic-Net penalty")
        results.append(benchmark(SGDClassifier(alpha=.0001, max_iter=50, penalty="elasticnet"),
                                 "SGDClassifier using Elastic-Net penalty", X_train, y_train, X_test, y_test))

        # Train NearestCentroid without threshold
        print('=' * 80)
        print("NearestCentroid (aka Rocchio classifier)")
        results.append(benchmark(NearestCentroid(), "NearestCentroid (aka Rocchio classifier)", X_train, y_train, X_test, y_test))

        # Train sparse Naive Bayes classifiers
        print('=' * 80)
        print("Naive Bayes")
        results.append(benchmark(MultinomialNB(alpha=.01), "MultinomialNB(alpha=.01)", X_train, y_train, X_test, y_test))
        results.append(benchmark(BernoulliNB(alpha=.01), "BernoulliNB(alpha=.01)", X_train, y_train, X_test, y_test))
        results.append(benchmark(ComplementNB(alpha=.1), "ComplementNB(alpha=.1)", X_train, y_train, X_test, y_test))

        print('=' * 80)
        print("LinearSVC with L1-based feature selection")
        # The smaller C, the stronger the regularization.
        # The more regularization, the more sparsity.
        results.append(benchmark(Pipeline([
            ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False, tol=1e-3))),
            ('classification', LinearSVC(penalty="l2"))]), "LinearSVC with L1-based feature selection", X_train, y_train, X_test, y_test))

    ###########################################################################################################################
    # Add plots: The bar plot indicates the accuracy, training time (normalized) and test time (normalized) of each classifier
    ###########################################################################################################################

    indices = np.arange(len(results))

    results = [[x[i] for x in results] for i in range(4)]

    clf_names, score, training_time, test_time = results
    training_time = np.array(training_time) / np.max(training_time)
    test_time = np.array(test_time) / np.max(test_time)

    plt.figure(figsize=(12, 8))
    title = ""
    if dataset == '20news':
        if options.twenty_news_with_no_filter:
            title = "20 News Groups: Accuracy score for the 20 news group dataset"
            plt.title()
        else:
            title = "20 News Groups: Accuracy score for the 20 news group dataset (removing headers signatures and quoting)"
            plt.title(title)


    elif dataset == 'imdb':
        if options.use_imdb_binary_labels:
            imdb_classification_type = "Binary classification"
        else:
            imdb_classification_type = "Multi-class classification"

        title = "IMDB Reviews: Accuracy score for the 20 news group dataset ({})".format(imdb_classification_type)
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

    print("{}: Final classification report: ".format(title))
    classifier_name_list = results[0]
    accuracy_score_list = results[1]
    train_time_list = results[2]
    test_time_list = results[3]
    if options.run_cross_validation:
        cross_val_accuracy_score_mean_std = results[4]
    index = 1
    for classifier_name, accuracy_score, train_time, test_time in zip(classifier_name_list, accuracy_score_list,
                                                                      train_time_list, test_time_list):
        if classifier_name in ["Logistic Regression", "Decision Tree Classifier", "Linear SVC (penalty = L2)",
                               "Linear SVC (penalty = L1)", "Ada Boost Classifier", "Random forest"]:
            classifier_name = classifier_name + " [MANDATORY FOR COMP 551, ASSIGNMENT 2]"
        if options.run_cross_validation:
            print("{}) {}\n\t\tAccuracy score = {}\t\tTraining time = {}\t\tTest time = {}\t\tAverage Accuracy Score (Cross Validation)\n".format(index,
                                                                                                     classifier_name,
                                                                                                     accuracy_score,
                                                                                                     train_time,
                                                                                                     test_time, cross_val_accuracy_score_mean_std))
        else:
            print("{}) {}\n\t\tAccuracy score = {}\t\tTraining time = {}\t\tTest time = {}\n".format(index, classifier_name,
                                                                                                 accuracy_score,
                                                                                                 train_time, test_time))
        index = index + 1

    print("\n\nBest algorithm:")
    index_max_accuracy_score, accuracy_score = max(enumerate(accuracy_score_list), key=operator.itemgetter(1))

    print("===> {}) {}\n\t\tAccuracy score = {}\t\tTraining time = {}\t\tTest time = {}\n".format(
        index_max_accuracy_score + 1,
        classifier_name_list[index_max_accuracy_score],
        accuracy_score_list[index_max_accuracy_score],
        train_time_list[index_max_accuracy_score],
        test_time_list[index_max_accuracy_score]))

    print('\n\nDONE!')

    print("Program finished. It took {} seconds".format(time() - start))
