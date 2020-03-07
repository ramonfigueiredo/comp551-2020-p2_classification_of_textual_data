import json
import logging
import os
from time import time

from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from datasets.load_dataset import load_twenty_news_groups, load_imdb_reviews
from model_selection.ml_algorithm_pair_list import JSON_FOLDER
from utils.dataset_enum import Dataset
from utils.ml_classifiers_enum import Classifier


def get_classifier_with_best_parameters(classifier_enum, best_parameters):

    if classifier_enum == Classifier.ADA_BOOST_CLASSIFIER:
        return AdaBoostClassifier(**best_parameters)

    elif classifier_enum == Classifier.BERNOULLI_NB:
        return BernoulliNB(**best_parameters)

    elif classifier_enum == Classifier.COMPLEMENT_NB:
        return ComplementNB(**best_parameters)

    elif classifier_enum == Classifier.DECISION_TREE_CLASSIFIER:
        return DecisionTreeClassifier(**best_parameters)

    elif classifier_enum == Classifier.GRADIENT_BOOSTING_CLASSIFIER:
        return GradientBoostingClassifier(**best_parameters)

    elif classifier_enum == Classifier.K_NEIGHBORS_CLASSIFIER:
        return KNeighborsClassifier(**best_parameters)

    elif classifier_enum == Classifier.LINEAR_SVC:
        return LinearSVC(**best_parameters)

    elif classifier_enum == Classifier.LOGISTIC_REGRESSION:
        return LogisticRegression(**best_parameters)

    elif classifier_enum == Classifier.MULTINOMIAL_NB:
        return MultinomialNB(**best_parameters)

    elif classifier_enum == Classifier.NEAREST_CENTROID:
        return NearestCentroid(**best_parameters)

    elif classifier_enum == Classifier.PASSIVE_AGGRESSIVE_CLASSIFIER:
        return PassiveAggressiveClassifier(**best_parameters)

    elif classifier_enum == Classifier.PERCEPTRON:
        return Perceptron(**best_parameters)

    elif classifier_enum == Classifier.RANDOM_FOREST_CLASSIFIER:
        return RandomForestClassifier(**best_parameters)

    elif classifier_enum == Classifier.RIDGE_CLASSIFIER:
        return RidgeClassifier(**best_parameters)


def run_classifier_grid_search(classifer, classifier_enum, param_grid, dataset, final_classification_table_default_parameters, final_classification_table_best_parameters, imdb_multi_class, save_json_with_best_parameters):

    if param_grid is None:
        return

    if dataset == Dataset.TWENTY_NEWS_GROUPS:
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

        if imdb_multi_class:
            X_train, y_train = \
                load_imdb_reviews(subset='train', multi_class_labels=True, verbose=False, shuffle=True, random_state=0,
                                  db_parent_path=db_parent_path)
            X_test, y_test = \
                load_imdb_reviews(subset='test', multi_class_labels=True, verbose=False, shuffle=True, random_state=0,
                                  db_parent_path=db_parent_path)
        else:
            X_train, y_train = \
                load_imdb_reviews(subset='train', multi_class_labels=False, verbose=False, shuffle=True, random_state=0,
                                  db_parent_path=db_parent_path)
            X_test, y_test = \
                load_imdb_reviews(subset='test', multi_class_labels=False, verbose=False, shuffle=True, random_state=0,
                                  db_parent_path=db_parent_path)

        if imdb_multi_class:
            # IMDB_REVIEWS dataset
            # If binary classification: 0 = neg and 1 = pos.
            # If multi-class classification use the review scores: 1, 2, 3, 4, 7, 8, 9, 10
            target_names = ['1', '2', '3', '4', '7', '8', '9', '10']
        else:
            # IMDB_REVIEWS dataset
            # If binary classification: 0 = neg and 1 = pos.
            # If multi-class classification use the review scores: 1, 2, 3, 4, 7, 8, 9, 10
            target_names = ['0', '1']

    try:
        # Extracting features
        vectorizer = TfidfVectorizer(stop_words='english', strip_accents='unicode', analyzer='word', binary=True)
        X_train = vectorizer.fit_transform(X_train)
        X_test = vectorizer.transform(X_test)

        # Create pipeline
        pipeline = Pipeline([('classifier', classifer)])

        # Create grid search object
        grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5, verbose=True, n_jobs=-1)

        logging.info("\n\nPerforming grid search...\n")
        logging.info("Parameters:")
        logging.info(param_grid)
        t0 = time()
        grid_search.fit(X_train, y_train)
        logging.info("\tDone in %0.3fs" % (time() - t0))

        # Get best parameters
        logging.info("\tBest score: %0.3f" % grid_search.best_score_)
        logging.info("\tBest parameters set:")
        best_parameters = grid_search.best_estimator_.get_params()

        new_parameters = {}
        for param_name in sorted(param_grid.keys()):
            logging.info("\t\t%s: %r" % (param_name, best_parameters[param_name]))
            key = param_name.replace('classifier__', '')
            value = best_parameters[param_name]
            new_parameters[key] = value

        if save_json_with_best_parameters:
            if dataset == Dataset.TWENTY_NEWS_GROUPS:
                json_path = os.path.join(os.getcwd(), JSON_FOLDER, dataset.name, classifier_enum.name + ".json")
                with open(json_path, 'w') as outfile:
                    json.dump(new_parameters, outfile)
            else:
                if imdb_multi_class:
                    json_path = os.path.join(os.getcwd(), JSON_FOLDER, dataset.name, 'multi_class_classification', classifier_enum.name + ".json")
                    with open(json_path, 'w') as outfile:
                        json.dump(new_parameters, outfile)
                else:
                    json_path = os.path.join(os.getcwd(), JSON_FOLDER, dataset.name, 'binary_classification', classifier_enum.name + ".json")
                    with open(json_path, 'w') as outfile:
                        json.dump(new_parameters, outfile)

        logging.info('\n\nUSING {} WITH DEFAULT PARAMETERS'.format(classifier_enum.name))
        clf = classifer
        final_classification_report(clf, X_train, y_train,  X_test, y_test, target_names, classifier_enum, final_classification_table_default_parameters)

        logging.info('\n\nUSING {} WITH BEST PARAMETERS: {}'.format(classifier_enum.name, new_parameters))
        clf = get_classifier_with_best_parameters(classifier_enum, new_parameters)
        final_classification_report(clf, X_train, y_train, X_test, y_test, target_names, classifier_enum, final_classification_table_best_parameters)

    except MemoryError as error:
        # Output expected MemoryErrors.
        logging.error(error)

    except Exception as exception:
        # Output unexpected Exceptions.
        logging.error(exception)


def final_classification_report(clf, X_train, y_train,  X_test, y_test, target_names, classifier_enum, final_classification_table):
    # Fit on data
    logging.info('_' * 80)
    logging.info("Training: ")
    logging.info(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    logging.info("Train time: %0.3fs" % train_time)
    # Predict
    t0 = time()
    y_pred = clf.predict(X_test)
    test_time = time() - t0
    logging.info("Test time:  %0.3fs" % test_time)
    accuracy_score = metrics.accuracy_score(y_test, y_pred)
    logging.info("Accuracy score:   %0.3f" % accuracy_score)
    logging.info("\n\n===> Classification Report:\n")
    # logging.info(metrics.classification_report(y_test, y_pred, target_names=target_names))
    logging.info(metrics.classification_report(y_test, y_pred))

    n_splits = 5
    logging.info("\n\nCross validation:")
    scoring = ['accuracy', 'precision_macro', 'precision_micro', 'precision_weighted', 'recall_macro', 'recall_micro',
               'recall_weighted', 'f1_macro', 'f1_micro', 'f1_weighted', 'jaccard_macro']
    cross_val_scores = cross_validate(clf, X_train, y_train, scoring=scoring, cv=n_splits,
                                      n_jobs=-1, verbose=True)
    cv_test_accuracy = cross_val_scores['test_accuracy']
    logging.info("\taccuracy: {}-fold cross validation: {}".format(5, cv_test_accuracy))
    cv_accuracy_score_mean_std = "%0.2f (+/- %0.2f)" % (cv_test_accuracy.mean() * 100, cv_test_accuracy.std() * 2 * 100)
    logging.info("\ttest accuracy: {}-fold cross validation accuracy: {}".format(n_splits, cv_accuracy_score_mean_std))

    final_classification_table[classifier_enum.value] = classifier_enum.name, format(accuracy_score, ".2%"), str(cv_test_accuracy), cv_accuracy_score_mean_std, format(train_time, ".4"), format(test_time, ".4")


def get_classifier_with_default_parameters(classifier_enum):
    if classifier_enum == Classifier.ADA_BOOST_CLASSIFIER:
        '''
        AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
               n_estimators=50, random_state=None)
        '''
        clf = AdaBoostClassifier()
        parameters = {
            'classifier__learning_rate': [0.1, 1],
            'classifier__n_estimators': [200, 500]
        }

    elif classifier_enum == Classifier.BERNOULLI_NB:
        '''
        BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
        '''
        clf = BernoulliNB()
        parameters = {
            'classifier__alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0],
            'classifier__binarize': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0],
            'classifier__fit_prior': [False, True]
        }

    elif classifier_enum == Classifier.COMPLEMENT_NB:
        '''
        ComplementNB(alpha=1.0, class_prior=None, fit_prior=True, norm=False)
        '''
        clf = ComplementNB()
        parameters = {
            'classifier__alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0],
            'classifier__fit_prior': [False, True],
            'classifier__norm': [False, True]
        }

    elif classifier_enum == Classifier.DECISION_TREE_CLASSIFIER:
        '''
        DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                   max_depth=None, max_features=None, max_leaf_nodes=None,
                   min_impurity_decrease=0.0, min_impurity_split=None,
                   min_samples_leaf=1, min_samples_split=2,
                   min_weight_fraction_leaf=0.0, presort='deprecated',
                   random_state=None, splitter='best')
        '''
        clf = DecisionTreeClassifier()
        parameters = {
            'classifier__criterion': ["entropy", "gini"],
            'classifier__splitter': ["best", "random"],
            'classifier__min_samples_split': [2, 100, 250]
        }

    elif classifier_enum == Classifier.K_NEIGHBORS_CLASSIFIER:
        '''
        KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                 metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                 weights='uniform')
        '''
        clf = KNeighborsClassifier()
        parameters = {
            'classifier__leaf_size': [5, 30],
            'classifier__metric': ['euclidean', 'minkowski'],
            'classifier__n_neighbors': [3, 50],
            'classifier__weights': ['uniform', 'distance']
        }

    elif classifier_enum == Classifier.LINEAR_SVC:
        '''
        LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
                  intercept_scaling=1, loss='squared_hinge', max_iter=1000,
                  multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
                  verbose=0)
        '''
        clf = LinearSVC()
        parameters = {
            'classifier__C': [0.01, 1.0],
            'classifier__multi_class': ['ovr', 'crammer_singer'],
            'classifier__tol': [0.0001, 0.001]
        }

    elif classifier_enum == Classifier.LOGISTIC_REGRESSION:
        '''
        LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                           intercept_scaling=1, l1_ratio=None, max_iter=100,
                           multi_class='auto', n_jobs=None, penalty='l2',
                           random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                           warm_start=False)
        '''
        clf = LogisticRegression()
        parameters = {
            'classifier__C': [1, 10],
            'classifier__tol': [0.001, 0.01]
        }

    elif classifier_enum == Classifier.MULTINOMIAL_NB:
        '''
        MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
        '''
        clf = MultinomialNB()
        parameters = {
            'classifier__alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0],
            'classifier__fit_prior': [False, True]
        }

    elif classifier_enum == Classifier.NEAREST_CENTROID:
        '''
        NearestCentroid(metric='euclidean', shrink_threshold=None)
        '''
        clf = NearestCentroid()
        parameters = {
            'classifier__metric': ['euclidean', 'cosine']
        }

    elif classifier_enum == Classifier.PASSIVE_AGGRESSIVE_CLASSIFIER:
        '''
        PassiveAggressiveClassifier(C=1.0, average=False, class_weight=None,
                        early_stopping=False, fit_intercept=True,
                        loss='hinge', max_iter=1000, n_iter_no_change=5,
                        n_jobs=None, random_state=None, shuffle=True,
                        tol=0.001, validation_fraction=0.1, verbose=0,
                        warm_start=False)
        '''
        clf = PassiveAggressiveClassifier()
        parameters = {
            'classifier__C': [0.01, 1.0],
            'classifier__early_stopping': [False, True],
            'classifier__tol': [0.0001, 0.001, 0.01],
            'classifier__validation_fraction': [0.0001, 0.01]
        }

    elif classifier_enum == Classifier.PERCEPTRON:
        '''
        Perceptron(alpha=0.0001, class_weight=None, early_stopping=False, eta0=1.0,
                   fit_intercept=True, max_iter=1000, n_iter_no_change=5, n_jobs=None,
                   penalty=None, random_state=0, shuffle=True, tol=0.001,
                   validation_fraction=0.1, verbose=0, warm_start=False)
        '''
        clf = Perceptron()
        parameters = {
            'classifier__early_stopping': [True],
            'classifier__max_iter': [100],
            'classifier__n_iter_no_change': [3, 15],
            'classifier__penalty': ['l2'],
            'classifier__tol': [0.0001, 0.1],
            'classifier__validation_fraction': [0.0001, 0.01]
        }

    elif classifier_enum == Classifier.RANDOM_FOREST_CLASSIFIER:
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
            'classifier__min_samples_leaf': [1, 2],
            'classifier__min_samples_split': [2, 5],
            'classifier__n_estimators': [100, 200]
        }

    elif classifier_enum == Classifier.RIDGE_CLASSIFIER:
        '''
        RidgeClassifier(alpha=1.0, class_weight=None, copy_X=True, fit_intercept=True,
                        max_iter=None, normalize=False, random_state=None,
                        solver='auto', tol=0.001)
        '''
        clf = RidgeClassifier()
        parameters = {
            'classifier__alpha': [0.5, 1.0],
            'classifier__tol': [0.0001, 0.001]
        }

    elif classifier_enum == Classifier.GRADIENT_BOOSTING_CLASSIFIER:
        '''
        GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                       learning_rate=0.1, loss='deviance', max_depth=3,
                       max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_iter_no_change=None, presort='deprecated',
                       random_state=None, subsample=1.0, tol=0.0001,
                       validation_fraction=0.1, verbose=0,
                       warm_start=False)
        '''
        clf = GradientBoostingClassifier()
        parameters = {
            'classifier__learning_rate': [0.01, 0.1],
            'classifier__n_estimators': [100, 200]
        }

    return clf, parameters


def run_grid_search(save_logs_in_file, just_imdb_dataset, imdb_multi_class, save_json_with_best_parameters=False):
    if imdb_multi_class:
        if save_logs_in_file:
            if not os.path.exists('grid_search/just_imdb_using_multi_class_classification'):
                os.mkdir('grid_search/just_imdb_using_multi_class_classification')
            logging.basicConfig(filename='grid_search/just_imdb_using_multi_class_classification/all.log', format='%(asctime)s - %(levelname)s - %(message)s',
                                level=logging.INFO, datefmt='%m/%d/%Y %I:%M:%S %p')
        else:
            logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO,
                                datefmt='%m/%d/%Y %I:%M:%S %p')
    else:
        if save_logs_in_file:
            if not os.path.exists('grid_search/20newsgroups_and_imdb_using_binary_classification'):
                os.mkdir('grid_search/20newsgroups_and_imdb_using_binary_classification')
            logging.basicConfig(filename='grid_search/20newsgroups_and_imdb_using_binary_classification/all.log', format='%(asctime)s - %(levelname)s - %(message)s',
                                level=logging.INFO, datefmt='%m/%d/%Y %I:%M:%S %p')
        else:
            logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO,
                                datefmt='%m/%d/%Y %I:%M:%S %p')

    classifier_list = [
        Classifier.ADA_BOOST_CLASSIFIER,
        Classifier.DECISION_TREE_CLASSIFIER,
        Classifier.LINEAR_SVC,
        Classifier.LOGISTIC_REGRESSION,
        Classifier.RANDOM_FOREST_CLASSIFIER,

        Classifier.BERNOULLI_NB,
        Classifier.COMPLEMENT_NB,
        Classifier.MULTINOMIAL_NB,
        Classifier.NEAREST_CENTROID,
        Classifier.PASSIVE_AGGRESSIVE_CLASSIFIER,
        Classifier.K_NEIGHBORS_CLASSIFIER,
        Classifier.PERCEPTRON,
        Classifier.RIDGE_CLASSIFIER,
        Classifier.GRADIENT_BOOSTING_CLASSIFIER
    ]

    if just_imdb_dataset:
        dataset_list = [
            Dataset.IMDB_REVIEWS
        ]
    else:
        dataset_list = [
            Dataset.IMDB_REVIEWS,
            Dataset.TWENTY_NEWS_GROUPS
        ]

    logging.info("\n>>> GRID SEARCH")
    for dataset in dataset_list:
        c_count = 1
        final_classification_table_default_parameters = {}
        final_classification_table_best_parameters = {}
        for classifier_enum in classifier_list:
            logging.info("\n")
            logging.info("#" * 80)
            if save_logs_in_file:
                print("#" * 80)
            logging.info("{})".format(c_count))

            clf, parameters = get_classifier_with_default_parameters(classifier_enum)

            logging.info("*" * 80)
            logging.info("Classifier: {}, Dataset: {}".format(classifier_enum.name, dataset.name))
            logging.info("*" * 80)
            start = time()
            run_classifier_grid_search(clf, classifier_enum, parameters, dataset, final_classification_table_default_parameters, final_classification_table_best_parameters, imdb_multi_class, save_json_with_best_parameters)
            end = time() - start
            logging.info("It took {} seconds".format(end))
            logging.info("*" * 80)

            if save_logs_in_file:
                print("*" * 80)
                print("Classifier: {}, Dataset: {}".format(classifier_enum.name, dataset.name))
                print(clf)
                print("It took {} seconds".format(end))
                print("*" * 80)

            logging.info("#" * 80)
            if save_logs_in_file:
                print("#" * 80)
            c_count = c_count + 1

            logging.info(
                '\n\nCURRENT CLASSIFICATION TABLE: {} DATASET, CLASSIFIER WITH DEFAULT PARAMETERS'.format(dataset.name))
            print_final_classification_table(final_classification_table_default_parameters)

            logging.info(
                '\n\nCURRENT CLASSIFICATION TABLE: {} DATASET, CLASSIFIER WITH BEST PARAMETERS'.format(dataset.name))
            print_final_classification_table(final_classification_table_best_parameters)

        logging.info('\n\nFINAL CLASSIFICATION TABLE: {} DATASET, CLASSIFIER WITH DEFAULT PARAMETERS'.format(dataset.name))
        print_final_classification_table(final_classification_table_default_parameters)

        logging.info('\n\nFINAL CLASSIFICATION TABLE: {} DATASET, CLASSIFIER WITH BEST PARAMETERS'.format(dataset.name))
        print_final_classification_table(final_classification_table_best_parameters)


def print_final_classification_table(final_classification_table_default_parameters):
    logging.info(
        '| ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | '
        'Training time (seconds) | Test time (seconds) |')
    logging.info(
        '| -- | ------------ | ------------------ | ------------------------------------ | ----------------- | '
        ' ------------------ | ------------------ |')
    for key in sorted(final_classification_table_default_parameters.keys()):
        values = final_classification_table_default_parameters[key]
        logging.info(
            "|  {}  |  {}  |  {}  |  {}  |  {}  |  {}  |  {}  |".format(key, values[0], values[1], values[2],
                                                                        values[3], values[4], values[5]))
