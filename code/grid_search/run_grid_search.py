# https://towardsdatascience.com/logistic-regression-model-tuning-with-scikit-learn-part-1-425142e01af5
import logging
import os
from time import time

import numpy as np
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from datasets.load_dataset import load_twenty_news_groups, load_imdb_reviews
from utils.dataset_enum import Dataset
from utils.ml_classifiers_enum import Classifier


def run_classifier_grid_search(classifer, param_grid, dataset):
    if param_grid is None:
        return

    if dataset == Dataset.TWENTY_NEWS_GROUP:
        remove = ('headers', 'footers', 'quotes')

        data_train = \
            load_twenty_news_groups(subset='train', categories=None, shuffle=True, random_state=0, remove=remove)

        data_test = \
            load_twenty_news_groups(subset='test', categories=None, shuffle=True, random_state=0, remove=remove)

        X_train, y_train = data_train.data, data_train.target
        X_test, y_test = data_test.data, data_test.target

        target_names = data_train.target_names

    elif dataset == Dataset.IMDB_REVIEWS:
        db_parent_path = os.getcwd()
        db_parent_path = db_parent_path.replace('grid_search', '')

        X_train, y_train = \
            load_imdb_reviews(subset='train', binary_labels=False, verbose=False, shuffle=True, random_state=0,
                              db_parent_path=db_parent_path)
        X_test, y_test = \
            load_imdb_reviews(subset='test', binary_labels=False, verbose=False, shuffle=True, random_state=0,
                              db_parent_path=db_parent_path)

        # IMDB reviews dataset
        # If binary classification: 0 = neg and 1 = pos.
        # If multi-class classification use the review scores: 1, 2, 3, 4, 7, 8, 9, 10
        target_names = ['1', '2', '3', '4', '7', '8', '9', '10']

    # Extracting features
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    # Create pipeline
    pipeline = Pipeline([('classifier', classifer)])

    # Create param grid.


    # Create grid search object
    grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5, verbose=True, n_jobs=-1)

    logging.info("\n\nPerforming grid search...\n")
    logging.info("\tPipeline:", [name for name, _ in pipeline.steps])
    logging.info("\tParameters:")
    logging.info(param_grid)
    t0 = time()
    grid_search.fit(X_train, y_train)
    logging.info("\tDone in %0.3fs" % (time() - t0))
    logging.info()

    # Fit on data
    logging.info("\tBest score: %0.3f" % grid_search.best_score_)
    logging.info("\tBest parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        logging.info("\t\t%s: %r" % (param_name, best_parameters[param_name]))

    logging.info("Running RandomForestClassifier with default values")
    logging.info('_' * 80)
    clf = classifer
    logging.info('_' * 80)
    logging.info("Training: ")
    logging.info(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    logging.info("Train time: %0.3fs" % train_time)

    t0 = time()
    y_pred = clf.predict(X_test)
    test_time = time() - t0
    logging.info("Test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, y_pred)
    logging.info("Accuracy score:   %0.3f" % score)

    logging.info("\n\n===> Classification Report:\n")
    logging.info(metrics.classification_report(y_test, y_pred, target_names=target_names))


def run_grid_search(save_logs_in_file):

    if save_logs_in_file:
        if not os.path.exists('logs_grid_search'):
            os.mkdir('logs_grid_search')
        logging.basicConfig(filename='logs_grid_search/all.log', format='%(asctime)s - %(levelname)s - %(message)s',
                            level=logging.INFO, datefmt='%m/%d/%Y %I:%M:%S %p')
    else:
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO,
                            datefmt='%m/%d/%Y %I:%M:%S %p')

    classifier_list = [
        Classifier.ADA_BOOST_CLASSIFIER,
        Classifier.BERNOULLI_NB,
        Classifier.COMPLEMENT_NB,
        Classifier.DECISION_TREE_CLASSIFIER,
        Classifier.K_NEIGHBORS_CLASSIFIER,
        Classifier.LINEAR_SVC,
        Classifier.LOGISTIC_REGRESSION,
        Classifier.MULTINOMIAL_NB,
        Classifier.NEAREST_CENTROID,
        Classifier.PASSIVE_AGGRESSIVE_CLASSIFIER,
        Classifier.PERCEPTRON,
        Classifier.RANDOM_FOREST_CLASSIFIER,
        Classifier.RIDGE_CLASSIFIER,
        Classifier.SGD_CLASSIFIER
    ]

    dataset_list = [
        Dataset.TWENTY_NEWS_GROUP,
        Dataset.IMDB_REVIEWS
    ]

    logging.info("\n>>> GRID SEARCH")
    c_count = 1
    for classifier in classifier_list:
        logging.info("\n")
        logging.info("#" * 80)
        logging.info("{})".format(c_count))

        if classifier == Classifier.ADA_BOOST_CLASSIFIER:
            clf = AdaBoostClassifier()
            parameters = None

        elif classifier == Classifier.BERNOULLI_NB:
            clf = BernoulliNB()
            parameters = None

        elif classifier == Classifier.COMPLEMENT_NB:
            clf = ComplementNB()
            parameters = None

        elif classifier == Classifier.DECISION_TREE_CLASSIFIER:
            clf = DecisionTreeClassifier()
            parameters = None

        elif classifier == Classifier.K_NEIGHBORS_CLASSIFIER:
            clf = KNeighborsClassifier()
            parameters = None

        elif classifier == Classifier.LINEAR_SVC:
            clf = LinearSVC()
            parameters = None

        elif classifier == Classifier.LOGISTIC_REGRESSION:
            clf = LogisticRegression()
            parameters = {
                'classifier__penalty': ['l2'],
                'classifier__C': np.logspace(0, 4, 10)
            }

        elif classifier == Classifier.MULTINOMIAL_NB:
            clf = MultinomialNB()
            parameters = None

        elif classifier == Classifier.NEAREST_CENTROID:
            clf = NearestCentroid()
            parameters = None

        elif classifier == Classifier.PASSIVE_AGGRESSIVE_CLASSIFIER:
            clf = PassiveAggressiveClassifier()
            parameters = None

        elif classifier == Classifier.PERCEPTRON:
            clf = Perceptron()
            parameters = None

        elif classifier == Classifier.RANDOM_FOREST_CLASSIFIER:
            clf = RandomForestClassifier()
            parameters = {
                'classifier__n_estimators': list(range(10, 101, 10)),
                'classifier__max_features': list(range(6, 32, 5))
            }

        elif classifier == Classifier.RIDGE_CLASSIFIER:
            clf = RidgeClassifier()
            parameters = None

        elif classifier == Classifier.SGD_CLASSIFIER:
            clf = SGDClassifier()
            parameters = {
                'classifier__max_iter': (20,),
                'classifier__alpha': (0.00001, 0.000001),
                'classifier__penalty': ('l2', 'elasticnet')
            }

        for dataset in dataset_list:
            logging.info("*" * 80)
            logging.info("*" * 80)
            print("Classifier: {}, Dataset: {}".format(classifier.name, dataset.name))
            logging.info("Classifier: {}, Dataset: {}".format(classifier.name, dataset.name))
            start = time()
            run_classifier_grid_search(clf, parameters, dataset)
            logging.info("It took {} seconds".format(time() - start))
            logging.info("*" * 80)
        logging.info("#" * 80)
        c_count = c_count + 1


if __name__ == '__main__':
    run_grid_search(True)
