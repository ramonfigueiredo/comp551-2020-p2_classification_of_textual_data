import json
import logging
import os

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from utils.dataset_enum import Dataset
from utils.ml_classifiers_enum import Classifier

JSON_FOLDER = 'model_selection' + os.sep + 'json_with_best_parameters'


def get_json_with_best_parameters(filename):
    try:
        with open(filename) as json_file:
            return (json.load(json_file))
    except json.decoder.JSONDecodeError as jde:
        logging.error("Error opening or reading the json file = {}. Error = {}".format(filename, jde))
        exit()
    except Exception as exception:
        logging.error("Error = {}. You need to create the JSON file ({}) with the best parameters of the classifier in order to use the best parameters.".format(exception, filename))
        exit()


def get_ml_algorithm_pair_list(options, ml_algorithm_list, use_classifiers_with_default_parameters,
                               use_imdb_multi_class_labels, dataset):
    ml_final_list = []

    if Classifier.ADA_BOOST_CLASSIFIER.name in ml_algorithm_list:
        if use_classifiers_with_default_parameters:
            ml_final_list.append((AdaBoostClassifier(random_state=options.random_state), Classifier.ADA_BOOST_CLASSIFIER))
        else:
            if dataset == Dataset.TWENTY_NEWS_GROUPS.name:
                # read json with best parameters
                json_path = os.path.join(os.getcwd(), JSON_FOLDER, dataset, Classifier.ADA_BOOST_CLASSIFIER.name + ".json")
                json_with_best_parameters = get_json_with_best_parameters(json_path)
                print("\t==> Using JSON with best parameters (selected using grid search) to the {} classifier and {} dataset ===> JSON in dictionary format: {}".format(Classifier.ADA_BOOST_CLASSIFIER.name, dataset, json_with_best_parameters))

                # adding options.random_state in the map
                json_with_best_parameters['random_state'] = options.random_state

                # create classifier with best parameters
                classifier_with_best_parameters = AdaBoostClassifier(**json_with_best_parameters)
                print('\t', classifier_with_best_parameters)

                ml_final_list.append((classifier_with_best_parameters, Classifier.ADA_BOOST_CLASSIFIER))
            elif dataset == Dataset.IMDB_REVIEWS.name:
                if use_imdb_multi_class_labels:
                    ml_final_list.append((AdaBoostClassifier(random_state=options.random_state, algorithm='SAMME.R',
                                                             learning_rate=0.1, n_estimators=500),
                                          Classifier.ADA_BOOST_CLASSIFIER))
                else:
                    # IMDb with binary classification
                    ml_final_list.append(
                        (AdaBoostClassifier(random_state=options.random_state), Classifier.ADA_BOOST_CLASSIFIER))

    if Classifier.BERNOULLI_NB.name in ml_algorithm_list:
        if use_classifiers_with_default_parameters:
            ml_final_list.append((BernoulliNB(), Classifier.BERNOULLI_NB))
        else:
            if dataset == Dataset.TWENTY_NEWS_GROUPS.name:
                '''
                
                '''
                ml_final_list.append((BernoulliNB(alpha=0.1, binarize=0.1, fit_prior=False), Classifier.BERNOULLI_NB))
            elif dataset == Dataset.IMDB_REVIEWS.name:
                if use_imdb_multi_class_labels:
                    '''
                    
                    '''
                    ml_final_list.append(
                        (BernoulliNB(alpha=0.5, binarize=0.0001, fit_prior=True), Classifier.BERNOULLI_NB))
                else:
                    # IMDb with binary classification
                    # TODO: Include best machine learning parameters
                    ml_final_list.append((BernoulliNB(), Classifier.BERNOULLI_NB))

    if Classifier.COMPLEMENT_NB.name in ml_algorithm_list:
        if use_classifiers_with_default_parameters:
            ml_final_list.append((ComplementNB(), Classifier.COMPLEMENT_NB))
        else:
            if dataset == Dataset.TWENTY_NEWS_GROUPS.name:
                '''
                
                '''
                ml_final_list.append((ComplementNB(alpha=0.5, fit_prior=False, norm=False), Classifier.COMPLEMENT_NB))
            elif dataset == Dataset.IMDB_REVIEWS.name:
                if use_imdb_multi_class_labels:
                    '''
                    
                    '''
                    ml_final_list.append(
                        (ComplementNB(alpha=0.5, fit_prior=False, norm=False), Classifier.COMPLEMENT_NB))
                else:
                    # IMDb with binary classification
                    # TODO: Include best machine learning parameters
                    ml_final_list.append((ComplementNB(), Classifier.COMPLEMENT_NB))

    if Classifier.DECISION_TREE_CLASSIFIER.name in ml_algorithm_list:
        if use_classifiers_with_default_parameters:
            ml_final_list.append \
                ((DecisionTreeClassifier(random_state=options.random_state), Classifier.DECISION_TREE_CLASSIFIER))
        else:
            if dataset == Dataset.TWENTY_NEWS_GROUPS.name:
                '''
                
                '''
                ml_final_list.append((DecisionTreeClassifier(random_state=options.random_state, criterion='entropy',
                                                             max_depth=19, min_samples_split=110, splitter='best'),
                                      Classifier.DECISION_TREE_CLASSIFIER))
            elif dataset == Dataset.IMDB_REVIEWS.name:
                if use_imdb_multi_class_labels:
                    '''
                    
                    '''
                    ml_final_list.append((DecisionTreeClassifier(random_state=options.random_state, criterion='entropy',
                                                                 max_depth=19, min_samples_split=250,
                                                                 splitter='random'),
                                          Classifier.DECISION_TREE_CLASSIFIER))
                else:
                    # IMDb with binary classification
                    '''
                    
                    '''
                    ml_final_list.append \
                        ((DecisionTreeClassifier(random_state=options.random_state),
                          Classifier.DECISION_TREE_CLASSIFIER))

    if Classifier.GRADIENT_BOOSTING_CLASSIFIER.name in ml_algorithm_list:
        if use_classifiers_with_default_parameters:
            ml_final_list.append(
                (GradientBoostingClassifier(verbose=options.verbose, random_state=options.random_state),
                 Classifier.GRADIENT_BOOSTING_CLASSIFIER))
        else:
            if dataset == Dataset.TWENTY_NEWS_GROUPS.name:
                '''
                
                '''
                ml_final_list.append((GradientBoostingClassifier(verbose=options.verbose,
                                                                 random_state=options.random_state),
                                      Classifier.GRADIENT_BOOSTING_CLASSIFIER))
            elif dataset == Dataset.IMDB_REVIEWS.name:
                if use_imdb_multi_class_labels:
                    '''
                    
                    '''
                    ml_final_list.append((GradientBoostingClassifier(verbose=options.verbose,
                                                                     random_state=options.random_state),
                                          Classifier.GRADIENT_BOOSTING_CLASSIFIER))
                else:
                    # IMDb with binary classification
                    '''
                    
                    '''
                    ml_final_list.append(
                        (GradientBoostingClassifier(verbose=options.verbose, random_state=options.random_state),
                         Classifier.GRADIENT_BOOSTING_CLASSIFIER))

    if Classifier.K_NEIGHBORS_CLASSIFIER.name in ml_algorithm_list:
        if use_classifiers_with_default_parameters:
            '''
            
            '''
            ml_final_list.append((KNeighborsClassifier(n_jobs=options.n_jobs), Classifier.K_NEIGHBORS_CLASSIFIER))
        else:
            if dataset == Dataset.TWENTY_NEWS_GROUPS.name:
                '''
                
                '''
                ml_final_list.append((KNeighborsClassifier(n_jobs=options.n_jobs, leaf_size=5, metric='euclidean',
                                                           n_neighbors=3, weights='distance'),
                                      Classifier.K_NEIGHBORS_CLASSIFIER))
            elif dataset == Dataset.IMDB_REVIEWS.name:
                if use_imdb_multi_class_labels:
                    '''
                    
                    '''
                    ml_final_list.append((KNeighborsClassifier(n_jobs=options.n_jobs, leaf_size=5, metric='euclidean',
                                                               n_neighbors=50, weights='distance'),
                                          Classifier.K_NEIGHBORS_CLASSIFIER))
                else:
                    # IMDb with binary classification
                    '''
                    
                    '''
                    ml_final_list.append(
                        (KNeighborsClassifier(n_jobs=options.n_jobs), Classifier.K_NEIGHBORS_CLASSIFIER))

    if Classifier.LINEAR_SVC.name in ml_algorithm_list:
        if use_classifiers_with_default_parameters:
            ml_final_list.append \
                ((LinearSVC(verbose=options.verbose, random_state=options.random_state), Classifier.LINEAR_SVC))
        else:
            if dataset == Dataset.TWENTY_NEWS_GROUPS.name:
                '''
                
                '''
                ml_final_list.append((LinearSVC(verbose=options.verbose, random_state=options.random_state, C=1.0,
                                                dual=True, max_iter=100, multi_class='ovr', tol=0.1),
                                      Classifier.LINEAR_SVC))
            elif dataset == Dataset.IMDB_REVIEWS.name:
                if use_imdb_multi_class_labels:
                    '''
                    
                    '''
                    ml_final_list.append((LinearSVC(verbose=options.verbose, random_state=options.random_state, C=0.01,
                                                    dual=True, max_iter=5000, multi_class='crammer_singer', tol=0.001),
                                          Classifier.LINEAR_SVC))
                else:
                    # IMDb with binary classification
                    '''
                    
                    '''
                    ml_final_list.append \
                        ((LinearSVC(verbose=options.verbose, random_state=options.random_state), Classifier.LINEAR_SVC))

    if Classifier.LOGISTIC_REGRESSION.name in ml_algorithm_list:
        if use_classifiers_with_default_parameters:
            ml_final_list.append((LogisticRegression(n_jobs=options.n_jobs, verbose=options.verbose,
                                                     random_state=options.random_state),
                                  Classifier.LOGISTIC_REGRESSION))
        else:
            if dataset == Dataset.TWENTY_NEWS_GROUPS.name:
                '''
                
                '''
                ml_final_list.append((LogisticRegression(n_jobs=options.n_jobs, verbose=options.verbose,
                                                         random_state=options.random_state, C=10.01, dual=False,
                                                         max_iter=100, multi_class='ovr', tol=0.01),
                                      Classifier.LOGISTIC_REGRESSION))
            elif dataset == Dataset.IMDB_REVIEWS.name:
                if use_imdb_multi_class_labels:
                    '''
                    
                    '''
                    ml_final_list.append((LogisticRegression(n_jobs=options.n_jobs, verbose=options.verbose,
                                                             random_state=options.random_state, C=1.0, dual=False,
                                                             max_iter=100, multi_class='ovr', tol=0.01),
                                          Classifier.LOGISTIC_REGRESSION))
                else:
                    # IMDb with binary classification
                    ml_final_list.append((LogisticRegression(n_jobs=options.n_jobs, verbose=options.verbose,
                                                             random_state=options.random_state),
                                          Classifier.LOGISTIC_REGRESSION))

    if Classifier.MULTINOMIAL_NB.name in ml_algorithm_list:
        if use_classifiers_with_default_parameters:
            ml_final_list.append((MultinomialNB(), Classifier.MULTINOMIAL_NB))
        else:
            if dataset == Dataset.TWENTY_NEWS_GROUPS.name:
                '''
                
                '''
                ml_final_list.append((MultinomialNB(alpha=0.01, fit_prior=True), Classifier.MULTINOMIAL_NB))
            elif dataset == Dataset.IMDB_REVIEWS.name:
                if use_imdb_multi_class_labels:
                    '''
                    
                    '''
                    ml_final_list.append((MultinomialNB(alpha=0.1, fit_prior=True), Classifier.MULTINOMIAL_NB))
                else:
                    # IMDb with binary classification
                    '''
                    
                    '''
                    ml_final_list.append((MultinomialNB(), Classifier.MULTINOMIAL_NB))

    if Classifier.NEAREST_CENTROID.name in ml_algorithm_list:
        if use_classifiers_with_default_parameters:
            ml_final_list.append((NearestCentroid(), Classifier.NEAREST_CENTROID))
        else:
            if dataset == Dataset.TWENTY_NEWS_GROUPS.name:
                '''
                
                '''
                ml_final_list.append((NearestCentroid(metric='cosine'), Classifier.NEAREST_CENTROID))
            elif dataset == Dataset.IMDB_REVIEWS.name:
                if use_imdb_multi_class_labels:
                    '''
                    
                    '''
                    ml_final_list.append((NearestCentroid(metric='cosine'), Classifier.NEAREST_CENTROID))
                else:
                    # IMDb with binary classification
                    '''
                    
                    '''
                    ml_final_list.append((NearestCentroid(), Classifier.NEAREST_CENTROID))

    if Classifier.PASSIVE_AGGRESSIVE_CLASSIFIER.name in ml_algorithm_list:
        if use_classifiers_with_default_parameters:
            ml_final_list.append((PassiveAggressiveClassifier(n_jobs=options.n_jobs, verbose=options.verbose,
                                                              random_state=options.random_state),
                                  Classifier.PASSIVE_AGGRESSIVE_CLASSIFIER))
        else:
            if dataset == Dataset.TWENTY_NEWS_GROUPS.name:
                '''
                
                '''
                ml_final_list.append((PassiveAggressiveClassifier(n_jobs=options.n_jobs, verbose=options.verbose,
                                                                  random_state=options.random_state, C=0.01,
                                                                  average=False, class_weight='balanced',
                                                                  early_stopping=False, max_iter=100,
                                                                  n_iter_no_change=5, tol=0.0001,
                                                                  validation_fraction=0.1),
                                      Classifier.PASSIVE_AGGRESSIVE_CLASSIFIER))
            elif dataset == Dataset.IMDB_REVIEWS.name:
                if use_imdb_multi_class_labels:
                    '''
                    
                    '''
                    ml_final_list.append((PassiveAggressiveClassifier(n_jobs=options.n_jobs, verbose=options.verbose,
                                                                      random_state=options.random_state, C=0.01,
                                                                      average=False, class_weight=None,
                                                                      early_stopping=True, max_iter=100,
                                                                      n_iter_no_change=5, tol=0.01,
                                                                      validation_fraction=0.01),
                                          Classifier.PASSIVE_AGGRESSIVE_CLASSIFIER))
                else:
                    # IMDb with binary classification
                    '''
                    
                    '''
                    ml_final_list.append((PassiveAggressiveClassifier(n_jobs=options.n_jobs, verbose=options.verbose,
                                                                      random_state=options.random_state),
                                          Classifier.PASSIVE_AGGRESSIVE_CLASSIFIER))

    if Classifier.PERCEPTRON.name in ml_algorithm_list:
        if use_classifiers_with_default_parameters:
            ml_final_list.append((Perceptron(n_jobs=options.n_jobs, verbose=options.verbose,
                                             random_state=options.random_state), Classifier.PERCEPTRON))
        else:
            if dataset == Dataset.TWENTY_NEWS_GROUPS.name:
                '''
                
                '''
                ml_final_list.append((Perceptron(n_jobs=options.n_jobs, verbose=options.verbose,
                                                 random_state=options.random_state, alpha=0.0001,
                                                 class_weight='balanced', early_stopping=True, max_iter=100,
                                                 n_iter_no_change=15, penalty='l2', tol=0.1, validation_fraction=0.01),
                                      Classifier.PERCEPTRON))
            elif dataset == Dataset.IMDB_REVIEWS.name:
                if use_imdb_multi_class_labels:
                    '''
                    
                    '''
                    ml_final_list.append((Perceptron(n_jobs=options.n_jobs, verbose=options.verbose,
                                                     random_state=options.random_state, alpha=0.0001, class_weight=None,
                                                     early_stopping=True, max_iter=100, n_iter_no_change=3,
                                                     penalty='l2', tol=0.0001, validation_fraction=0.001),
                                          Classifier.PERCEPTRON))
                else:
                    # IMDb with binary classification
                    '''
                    
                    '''
                    ml_final_list.append((Perceptron(n_jobs=options.n_jobs, verbose=options.verbose,
                                                     random_state=options.random_state), Classifier.PERCEPTRON))

    if Classifier.RANDOM_FOREST_CLASSIFIER.name in ml_algorithm_list:
        if use_classifiers_with_default_parameters:
            ml_final_list.append((RandomForestClassifier(n_jobs=options.n_jobs, verbose=options.verbose,
                                                         random_state=options.random_state),
                                  Classifier.RANDOM_FOREST_CLASSIFIER))
        else:
            '''
            
            '''
            if dataset == Dataset.TWENTY_NEWS_GROUPS.name:
                ml_final_list.append((RandomForestClassifier(n_jobs=options.n_jobs, verbose=options.verbose,
                                                             random_state=options.random_state),
                                      Classifier.RANDOM_FOREST_CLASSIFIER))
            elif dataset == Dataset.IMDB_REVIEWS.name:
                if use_imdb_multi_class_labels:
                    ml_final_list.append((RandomForestClassifier(n_jobs=options.n_jobs, verbose=options.verbose,
                                                                 random_state=options.random_state),
                                          Classifier.RANDOM_FOREST_CLASSIFIER))
                else:
                    # IMDb with binary classification
                    '''
                    
                    '''
                    ml_final_list.append((RandomForestClassifier(n_jobs=options.n_jobs, verbose=options.verbose,
                                                                 random_state=options.random_state),
                                          Classifier.RANDOM_FOREST_CLASSIFIER))

    if Classifier.RIDGE_CLASSIFIER.name in ml_algorithm_list:
        if use_classifiers_with_default_parameters:
            ml_final_list.append((RidgeClassifier(random_state=options.random_state), Classifier.RIDGE_CLASSIFIER))
        else:
            '''
            
            '''
            if dataset == Dataset.TWENTY_NEWS_GROUPS.name:
                ml_final_list.append((RidgeClassifier(random_state=options.random_state), Classifier.RIDGE_CLASSIFIER))
            elif dataset == Dataset.IMDB_REVIEWS.name:
                if use_imdb_multi_class_labels:
                    ml_final_list.append(
                        (RidgeClassifier(random_state=options.random_state), Classifier.RIDGE_CLASSIFIER))
                else:
                    # IMDb with binary classification
                    '''
                    
                    '''
                    ml_final_list.append(
                        (RidgeClassifier(random_state=options.random_state), Classifier.RIDGE_CLASSIFIER))

    return ml_final_list
