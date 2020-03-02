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
        if save_logs_in_file:
            print("#" * 80)
        logging.info("{})".format(c_count))

        if classifier == Classifier.ADA_BOOST_CLASSIFIER:
            '''
            AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
                   n_estimators=50, random_state=None)
            '''
            clf = AdaBoostClassifier()
            parameters = None

        elif classifier == Classifier.BERNOULLI_NB:
            '''
            BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
            '''
            clf = BernoulliNB()
            parameters = None

        elif classifier == Classifier.COMPLEMENT_NB:
            '''
            ComplementNB(alpha=1.0, class_prior=None, fit_prior=True, norm=False)
            '''
            clf = ComplementNB()
            parameters = None

        elif classifier == Classifier.DECISION_TREE_CLASSIFIER:
            '''
            DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=None, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=None, splitter='best')
            '''
            clf = DecisionTreeClassifier()
            parameters = None

        elif classifier == Classifier.K_NEIGHBORS_CLASSIFIER:
            '''
            KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                     weights='uniform')
            '''
            clf = KNeighborsClassifier()
            parameters = None

        elif classifier == Classifier.LINEAR_SVC:
            '''
            LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
                      intercept_scaling=1, loss='squared_hinge', max_iter=1000,
                      multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
                      verbose=0)
            '''
            clf = LinearSVC()
            parameters = None

        elif classifier == Classifier.LOGISTIC_REGRESSION:
            '''
            LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                               intercept_scaling=1, l1_ratio=None, max_iter=100,
                               multi_class='auto', n_jobs=None, penalty='l2',
                               random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                               warm_start=False)
            '''
            clf = LogisticRegression()
            parameters = {
                'classifier__penalty': ['l2'],
                'classifier__C': np.logspace(0, 4, 10)
            }

        elif classifier == Classifier.MULTINOMIAL_NB:
            '''
            MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
            '''
            clf = MultinomialNB()
            parameters = None

        elif classifier == Classifier.NEAREST_CENTROID:
            '''
            NearestCentroid(metric='euclidean', shrink_threshold=None)
            '''
            clf = NearestCentroid()
            parameters = None

        elif classifier == Classifier.PASSIVE_AGGRESSIVE_CLASSIFIER:
            '''
            PassiveAggressiveClassifier(C=1.0, average=False, class_weight=None,
                            early_stopping=False, fit_intercept=True,
                            loss='hinge', max_iter=1000, n_iter_no_change=5,
                            n_jobs=None, random_state=None, shuffle=True,
                            tol=0.001, validation_fraction=0.1, verbose=0,
                            warm_start=False)
            '''
            clf = PassiveAggressiveClassifier()
            parameters = None

        elif classifier == Classifier.PERCEPTRON:
            '''
            Perceptron(alpha=0.0001, class_weight=None, early_stopping=False, eta0=1.0,
                       fit_intercept=True, max_iter=1000, n_iter_no_change=5, n_jobs=None,
                       penalty=None, random_state=0, shuffle=True, tol=0.001,
                       validation_fraction=0.1, verbose=0, warm_start=False)
            '''
            clf = Perceptron()
            parameters = None

        elif classifier == Classifier.RANDOM_FOREST_CLASSIFIER:
            '''
            RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                                   criterion='gini', max_depth=None, max_features='auto',
                                   max_leaf_nodes=None, max_samples=None,
                                   min_impurity_decrease=0.0, min_impurity_split=None,
                                   min_samples_leaf=1, min_samples_split=2,
                                   min_weight_fraction_leaf=0.0, n_estimators=100,
                                   n_jobs=None, oob_score=False, random_state=None,
                                   verbose=0, warm_start=False)
            '''
            clf = RandomForestClassifier()
            parameters = {
                'classifier__n_estimators': list(range(10, 101, 10)),
                'classifier__max_features': list(range(6, 32, 5))
            }

        elif classifier == Classifier.RIDGE_CLASSIFIER:
            '''
            RidgeClassifier(alpha=1.0, class_weight=None, copy_X=True, fit_intercept=True,
                            max_iter=None, normalize=False, random_state=None,
                            solver='auto', tol=0.001)
            '''
            clf = RidgeClassifier()
            parameters = None

        elif classifier == Classifier.SGD_CLASSIFIER:
            '''
            SGDClassifier(alpha=0.0001, average=False, class_weight=None,
                          early_stopping=False, epsilon=0.1, eta0=0.0, fit_intercept=True,
                          l1_ratio=0.15, learning_rate='optimal', loss='hinge',
                          max_iter=1000, n_iter_no_change=5, n_jobs=None, penalty='l2',
                          power_t=0.5, random_state=None, shuffle=True, tol=0.001,
                          validation_fraction=0.1, verbose=0, warm_start=False)
            '''
            clf = SGDClassifier()
            parameters = {
                'classifier__max_iter': (20,),
                'classifier__alpha': (0.00001, 0.000001),
                'classifier__penalty': ('l2', 'elasticnet')
            }

        for dataset in dataset_list:
            logging.info("*" * 80)
            logging.info("Classifier: {}, Dataset: {}".format(classifier.name, dataset.name))
            start = time()
            run_classifier_grid_search(clf, parameters, dataset)
            end = time() - start
            logging.info("It took {} seconds".format(end))
            logging.info("*" * 80)

            if save_logs_in_file:
                print("*" * 80)
                print("Classifier: {}, Dataset: {}".format(classifier.name, dataset.name))
                print(clf)
                print("It took {} seconds".format(end))
                print("*" * 80)

        logging.info("#" * 80)
        if save_logs_in_file:
            print("#" * 80)
        c_count = c_count + 1


if __name__ == '__main__':
    run_grid_search(True)
