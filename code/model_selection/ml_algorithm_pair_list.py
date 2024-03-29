import json
import logging
import os

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import VotingClassifier
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


def open_json_with_best_parameters(filename):
    try:
        with open(filename) as json_file:
            return (json.load(json_file))
    except json.decoder.JSONDecodeError as jde:
        logging.error("Error opening or reading the json file = {}. Error = {}".format(filename, jde))
        exit()
    except Exception as exception:
        logging.error(
            "Error = {}. You need to create the JSON file ({}) with the best parameters of the classifier in order to use the best parameters.".format(
                exception, filename))
        exit()


def get_json_with_best_parameters(dataset, classifier_enum, imdb_multi_class):
    # read json with best parameters
    if dataset == Dataset.TWENTY_NEWS_GROUPS.name:
        json_path = os.path.join(os.getcwd(), JSON_FOLDER, dataset, classifier_enum.name + ".json")
        classification_type = "multi-class classification"
    else:
        if imdb_multi_class:
            classification_type = "multi-class classification"
            json_path = os.path.join(os.getcwd(), JSON_FOLDER, dataset, 'multi_class_classification',
                                     classifier_enum.name + ".json")
        else:
            classification_type = "binary classification"
            json_path = os.path.join(os.getcwd(), JSON_FOLDER, dataset, 'binary_classification',
                                     classifier_enum.name + ".json")

    json_with_best_parameters = open_json_with_best_parameters(json_path)

    print(
        "\t==> Using JSON with best parameters (selected using grid search) to the {} classifier ({}) and {} dataset ===> JSON in dictionary format: {}".format(
            classifier_enum.name, classification_type, dataset, json_with_best_parameters))

    return json_with_best_parameters


def get_ml_algorithm_pair_list(options, ml_algorithm_list, use_classifiers_with_default_parameters,
                               use_imdb_multi_class_labels, dataset):
    ml_final_list = []

    if Classifier.ADA_BOOST_CLASSIFIER.name in ml_algorithm_list:
        if use_classifiers_with_default_parameters:
            ml_final_list.append(
                (AdaBoostClassifier(random_state=options.random_state), Classifier.ADA_BOOST_CLASSIFIER))
        else:
            json_with_best_parameters = get_json_with_best_parameters(dataset, Classifier.ADA_BOOST_CLASSIFIER,
                                                                      use_imdb_multi_class_labels)
            # adding options.random_state in the map
            json_with_best_parameters['random_state'] = options.random_state

            # create classifier with best parameters
            classifier_with_best_parameters = AdaBoostClassifier(**json_with_best_parameters)
            print('\t', classifier_with_best_parameters)

            ml_final_list.append((classifier_with_best_parameters, Classifier.ADA_BOOST_CLASSIFIER))

    if Classifier.BERNOULLI_NB.name in ml_algorithm_list:
        if use_classifiers_with_default_parameters:
            ml_final_list.append((BernoulliNB(), Classifier.BERNOULLI_NB))
        else:
            json_with_best_parameters = get_json_with_best_parameters(dataset, Classifier.BERNOULLI_NB,
                                                                      use_imdb_multi_class_labels)

            # create classifier with best parameters
            classifier_with_best_parameters = BernoulliNB(**json_with_best_parameters)
            print('\t', classifier_with_best_parameters)

            ml_final_list.append((classifier_with_best_parameters, Classifier.BERNOULLI_NB))

    if Classifier.COMPLEMENT_NB.name in ml_algorithm_list:
        if use_classifiers_with_default_parameters:
            ml_final_list.append((ComplementNB(), Classifier.COMPLEMENT_NB))
        else:
            json_with_best_parameters = get_json_with_best_parameters(dataset, Classifier.COMPLEMENT_NB,
                                                                      use_imdb_multi_class_labels)

            # create classifier with best parameters
            classifier_with_best_parameters = ComplementNB(**json_with_best_parameters)
            print('\t', classifier_with_best_parameters)

            ml_final_list.append((classifier_with_best_parameters, Classifier.COMPLEMENT_NB))

    if Classifier.DECISION_TREE_CLASSIFIER.name in ml_algorithm_list:
        if use_classifiers_with_default_parameters:
            ml_final_list.append(
                (DecisionTreeClassifier(random_state=options.random_state), Classifier.DECISION_TREE_CLASSIFIER))
        else:
            json_with_best_parameters = get_json_with_best_parameters(dataset, Classifier.DECISION_TREE_CLASSIFIER,
                                                                      use_imdb_multi_class_labels)
            # adding options.random_state in the map
            json_with_best_parameters['random_state'] = options.random_state

            # create classifier with best parameters
            classifier_with_best_parameters = DecisionTreeClassifier(**json_with_best_parameters)
            print('\t', classifier_with_best_parameters)

            ml_final_list.append((classifier_with_best_parameters, Classifier.DECISION_TREE_CLASSIFIER))

    if Classifier.GRADIENT_BOOSTING_CLASSIFIER.name in ml_algorithm_list:
        if use_classifiers_with_default_parameters:
            ml_final_list.append(
                (GradientBoostingClassifier(verbose=options.verbose, random_state=options.random_state),
                 Classifier.GRADIENT_BOOSTING_CLASSIFIER))
        else:
            json_with_best_parameters = get_json_with_best_parameters(dataset, Classifier.GRADIENT_BOOSTING_CLASSIFIER,
                                                                      use_imdb_multi_class_labels)
            # adding options.verbose in the map
            json_with_best_parameters['verbose'] = options.verbose
            # adding options.random_state in the map
            json_with_best_parameters['random_state'] = options.random_state

            # create classifier with best parameters
            classifier_with_best_parameters = GradientBoostingClassifier(**json_with_best_parameters)
            print('\t', classifier_with_best_parameters)

            ml_final_list.append((classifier_with_best_parameters, Classifier.GRADIENT_BOOSTING_CLASSIFIER))

    if Classifier.K_NEIGHBORS_CLASSIFIER.name in ml_algorithm_list:
        if use_classifiers_with_default_parameters:
            ml_final_list.append((KNeighborsClassifier(n_jobs=options.n_jobs), Classifier.K_NEIGHBORS_CLASSIFIER))
        else:
            json_with_best_parameters = get_json_with_best_parameters(dataset, Classifier.K_NEIGHBORS_CLASSIFIER,
                                                                      use_imdb_multi_class_labels)
            # adding options.random_state in the map
            json_with_best_parameters['n_jobs'] = options.n_jobs

            # create classifier with best parameters
            classifier_with_best_parameters = KNeighborsClassifier(**json_with_best_parameters)
            print('\t', classifier_with_best_parameters)

            ml_final_list.append((classifier_with_best_parameters, Classifier.K_NEIGHBORS_CLASSIFIER))

    if Classifier.LINEAR_SVC.name in ml_algorithm_list:
        if use_classifiers_with_default_parameters:
            ml_final_list.append(
                (LinearSVC(verbose=options.verbose, random_state=options.random_state), Classifier.LINEAR_SVC))
        else:
            json_with_best_parameters = get_json_with_best_parameters(dataset, Classifier.LINEAR_SVC,
                                                                      use_imdb_multi_class_labels)
            # adding options.verbose in the map
            json_with_best_parameters['verbose'] = options.verbose
            # adding options.random_state in the map
            json_with_best_parameters['random_state'] = options.random_state

            # create classifier with best parameters
            classifier_with_best_parameters = LinearSVC(**json_with_best_parameters)
            print('\t', classifier_with_best_parameters)

            ml_final_list.append((classifier_with_best_parameters, Classifier.LINEAR_SVC))

    if Classifier.LOGISTIC_REGRESSION.name in ml_algorithm_list:
        if use_classifiers_with_default_parameters:
            ml_final_list.append((LogisticRegression(n_jobs=options.n_jobs, verbose=options.verbose,
                                                     random_state=options.random_state),
                                  Classifier.LOGISTIC_REGRESSION))
        else:
            json_with_best_parameters = get_json_with_best_parameters(dataset, Classifier.LOGISTIC_REGRESSION,
                                                                      use_imdb_multi_class_labels)
            # adding options.n_jobs in the map
            json_with_best_parameters['n_jobs'] = options.n_jobs
            # adding options.verbose in the map
            json_with_best_parameters['verbose'] = options.verbose
            # adding options.random_state in the map
            json_with_best_parameters['random_state'] = options.random_state

            # create classifier with best parameters
            classifier_with_best_parameters = LogisticRegression(**json_with_best_parameters)
            print('\t', classifier_with_best_parameters)

            ml_final_list.append((classifier_with_best_parameters, Classifier.LOGISTIC_REGRESSION))

    if Classifier.MULTINOMIAL_NB.name in ml_algorithm_list:
        if use_classifiers_with_default_parameters:
            ml_final_list.append((MultinomialNB(), Classifier.MULTINOMIAL_NB))
        else:
            json_with_best_parameters = get_json_with_best_parameters(dataset, Classifier.MULTINOMIAL_NB,
                                                                      use_imdb_multi_class_labels)

            # create classifier with best parameters
            classifier_with_best_parameters = MultinomialNB(**json_with_best_parameters)
            print('\t', classifier_with_best_parameters)

            ml_final_list.append((classifier_with_best_parameters, Classifier.MULTINOMIAL_NB))

    if Classifier.NEAREST_CENTROID.name in ml_algorithm_list:
        if use_classifiers_with_default_parameters:
            ml_final_list.append((NearestCentroid(), Classifier.NEAREST_CENTROID))
        else:
            json_with_best_parameters = get_json_with_best_parameters(dataset, Classifier.NEAREST_CENTROID,
                                                                      use_imdb_multi_class_labels)

            # create classifier with best parameters
            classifier_with_best_parameters = NearestCentroid(**json_with_best_parameters)
            print('\t', classifier_with_best_parameters)

            ml_final_list.append((classifier_with_best_parameters, Classifier.NEAREST_CENTROID))

    if Classifier.PASSIVE_AGGRESSIVE_CLASSIFIER.name in ml_algorithm_list:
        if use_classifiers_with_default_parameters:
            ml_final_list.append((PassiveAggressiveClassifier(n_jobs=options.n_jobs, verbose=options.verbose,
                                                              random_state=options.random_state),
                                  Classifier.PASSIVE_AGGRESSIVE_CLASSIFIER))
        else:
            json_with_best_parameters = get_json_with_best_parameters(dataset, Classifier.PASSIVE_AGGRESSIVE_CLASSIFIER,
                                                                      use_imdb_multi_class_labels)
            # adding options.n_jobs in the map
            json_with_best_parameters['n_jobs'] = options.n_jobs
            # adding options.verbose in the map
            json_with_best_parameters['verbose'] = options.verbose
            # adding options.random_state in the map
            json_with_best_parameters['random_state'] = options.random_state

            # create classifier with best parameters
            classifier_with_best_parameters = PassiveAggressiveClassifier(**json_with_best_parameters)
            print('\t', classifier_with_best_parameters)

            ml_final_list.append((classifier_with_best_parameters, Classifier.PASSIVE_AGGRESSIVE_CLASSIFIER))

    if Classifier.PERCEPTRON.name in ml_algorithm_list:
        if use_classifiers_with_default_parameters:
            ml_final_list.append((Perceptron(n_jobs=options.n_jobs, verbose=options.verbose,
                                             random_state=options.random_state), Classifier.PERCEPTRON))
        else:
            json_with_best_parameters = get_json_with_best_parameters(dataset, Classifier.PERCEPTRON,
                                                                      use_imdb_multi_class_labels)
            # adding options.n_jobs in the map
            json_with_best_parameters['n_jobs'] = options.n_jobs
            # adding options.verbose in the map
            json_with_best_parameters['verbose'] = options.verbose
            # adding options.random_state in the map
            json_with_best_parameters['random_state'] = options.random_state

            # create classifier with best parameters
            classifier_with_best_parameters = Perceptron(**json_with_best_parameters)
            print('\t', classifier_with_best_parameters)

            ml_final_list.append((classifier_with_best_parameters, Classifier.PERCEPTRON))

    if Classifier.RANDOM_FOREST_CLASSIFIER.name in ml_algorithm_list:
        if use_classifiers_with_default_parameters:
            ml_final_list.append((RandomForestClassifier(n_jobs=options.n_jobs, verbose=options.verbose,
                                                         random_state=options.random_state),
                                  Classifier.RANDOM_FOREST_CLASSIFIER))
        else:
            json_with_best_parameters = get_json_with_best_parameters(dataset, Classifier.RANDOM_FOREST_CLASSIFIER,
                                                                      use_imdb_multi_class_labels)
            # adding options.n_jobs in the map
            json_with_best_parameters['n_jobs'] = options.n_jobs
            # adding options.verbose in the map
            json_with_best_parameters['verbose'] = options.verbose
            # adding options.random_state in the map
            json_with_best_parameters['random_state'] = options.random_state

            # create classifier with best parameters
            classifier_with_best_parameters = RandomForestClassifier(**json_with_best_parameters)
            print('\t', classifier_with_best_parameters)

            ml_final_list.append((classifier_with_best_parameters, Classifier.RANDOM_FOREST_CLASSIFIER))

    if Classifier.RIDGE_CLASSIFIER.name in ml_algorithm_list:
        if use_classifiers_with_default_parameters:
            ml_final_list.append((RidgeClassifier(random_state=options.random_state), Classifier.RIDGE_CLASSIFIER))
        else:
            json_with_best_parameters = get_json_with_best_parameters(dataset, Classifier.RIDGE_CLASSIFIER,
                                                                      use_imdb_multi_class_labels)
            # adding options.random_state in the map
            json_with_best_parameters['random_state'] = options.random_state

            # create classifier with best parameters
            classifier_with_best_parameters = RidgeClassifier(**json_with_best_parameters)
            print('\t', classifier_with_best_parameters)

            ml_final_list.append((classifier_with_best_parameters, Classifier.RIDGE_CLASSIFIER))

    if Classifier.MAJORITY_VOTING_CLASSIFIER.name in ml_algorithm_list:
        estimators_list = get_estimators_list(dataset, options, use_imdb_multi_class_labels, is_soft_voting=False, is_stacking_classifier=False)

        classifier_with_best_parameters = VotingClassifier(
            estimators=estimators_list,
            voting='hard',  # voting='hard' means majority voting
            n_jobs=options.n_jobs
        )
        print('\t', classifier_with_best_parameters)

        ml_final_list.append((classifier_with_best_parameters, Classifier.MAJORITY_VOTING_CLASSIFIER))

    if Classifier.SOFT_VOTING_CLASSIFIER.name in ml_algorithm_list:
        estimators_list = get_estimators_list(dataset, options, use_imdb_multi_class_labels, is_soft_voting=True, is_stacking_classifier=False)

        classifier_with_best_parameters = VotingClassifier(
            estimators=estimators_list,
            voting='soft',
            # voting='soft' predicts the class label based on the argmax of the sums of the predicted probabilities
            n_jobs=options.n_jobs
        )
        print('\t', classifier_with_best_parameters)

        ml_final_list.append((classifier_with_best_parameters, Classifier.SOFT_VOTING_CLASSIFIER))

    if Classifier.STACKING_CLASSIFIER.name in ml_algorithm_list:
        estimators_list, final_estimator = get_estimators_list(dataset, options, use_imdb_multi_class_labels,
                                                               is_stacking_classifier=True,
                                                               final_estimator=Classifier.LINEAR_SVC.name)

        classifier_with_best_parameters = StackingClassifier(
            estimators=estimators_list,
            final_estimator=final_estimator,
            verbose=options.verbose,
            n_jobs=options.n_jobs
        )
        print('\t', classifier_with_best_parameters)

        ml_final_list.append((classifier_with_best_parameters, Classifier.STACKING_CLASSIFIER))


    return ml_final_list


def get_estimators_list(dataset, options, use_imdb_multi_class_labels, is_soft_voting=False, is_stacking_classifier=False, final_estimator=None):

    if is_stacking_classifier:
        ml_algorithm_list = [
            Classifier.COMPLEMENT_NB.name,
            Classifier.RIDGE_CLASSIFIER.name,
            Classifier.LINEAR_SVC.name,
            Classifier.LOGISTIC_REGRESSION.name,
            Classifier.PASSIVE_AGGRESSIVE_CLASSIFIER.name,
            Classifier.RANDOM_FOREST_CLASSIFIER.name
        ]
    else: # is VotingClassifier
        if is_soft_voting:
            ml_algorithm_list = [
                Classifier.COMPLEMENT_NB.name,
                Classifier.LOGISTIC_REGRESSION.name,
                Classifier.MULTINOMIAL_NB.name,
                Classifier.RANDOM_FOREST_CLASSIFIER.name
            ]
        else:
            ml_algorithm_list = [
                Classifier.COMPLEMENT_NB.name,
                Classifier.RIDGE_CLASSIFIER.name,
                Classifier.LINEAR_SVC.name,
                Classifier.LOGISTIC_REGRESSION.name,
                Classifier.PASSIVE_AGGRESSIVE_CLASSIFIER.name,
                Classifier.RANDOM_FOREST_CLASSIFIER.name
            ]

    estimators_list = []

    if Classifier.ADA_BOOST_CLASSIFIER.name in ml_algorithm_list:
        json_with_best_parameters = get_json_with_best_parameters(dataset, Classifier.ADA_BOOST_CLASSIFIER,
                                                                  use_imdb_multi_class_labels)
        # adding options.random_state in the map
        json_with_best_parameters['random_state'] = options.random_state
        # create classifier with best parameters
        ada_boost_classifier = AdaBoostClassifier(**json_with_best_parameters)
        print('\t', ada_boost_classifier)
        estimators_list.append(('ada_boost_classifier', ada_boost_classifier))

    if Classifier.BERNOULLI_NB.name in ml_algorithm_list:
        json_with_best_parameters = get_json_with_best_parameters(dataset, Classifier.BERNOULLI_NB,
                                                                  use_imdb_multi_class_labels)
        # create classifier with best parameters
        bernoulli_nb = BernoulliNB(**json_with_best_parameters)
        print('\t', bernoulli_nb)
        estimators_list.append(('bernoulli_nb', bernoulli_nb))

    if Classifier.COMPLEMENT_NB.name in ml_algorithm_list:
        json_with_best_parameters = get_json_with_best_parameters(dataset, Classifier.COMPLEMENT_NB,
                                                                  use_imdb_multi_class_labels)
        # create classifier with best parameters
        complement_nb = ComplementNB(**json_with_best_parameters)
        print('\t', complement_nb)
        estimators_list.append(('complement_nb', complement_nb))

    if Classifier.DECISION_TREE_CLASSIFIER.name in ml_algorithm_list:
        json_with_best_parameters = get_json_with_best_parameters(dataset, Classifier.DECISION_TREE_CLASSIFIER,
                                                                  use_imdb_multi_class_labels)
        # adding options.random_state in the map
        json_with_best_parameters['random_state'] = options.random_state
        # create classifier with best parameters
        decision_tree_classifier = DecisionTreeClassifier(**json_with_best_parameters)
        print('\t', decision_tree_classifier)
        estimators_list.append(('decision_tree_classifier', decision_tree_classifier))

    if Classifier.GRADIENT_BOOSTING_CLASSIFIER.name in ml_algorithm_list:
        json_with_best_parameters = get_json_with_best_parameters(dataset, Classifier.GRADIENT_BOOSTING_CLASSIFIER,
                                                                  use_imdb_multi_class_labels)
        # adding options.verbose in the map
        json_with_best_parameters['verbose'] = options.verbose
        # adding options.random_state in the map
        json_with_best_parameters['random_state'] = options.random_state
        # create classifier with best parameters
        gradient_boosting_classifier = GradientBoostingClassifier(**json_with_best_parameters)
        print('\t', gradient_boosting_classifier)
        estimators_list.append(('gradient_boosting_classifier', gradient_boosting_classifier))

    if Classifier.K_NEIGHBORS_CLASSIFIER.name in ml_algorithm_list:
        json_with_best_parameters = get_json_with_best_parameters(dataset, Classifier.K_NEIGHBORS_CLASSIFIER,
                                                                  use_imdb_multi_class_labels)
        # adding options.random_state in the map
        json_with_best_parameters['n_jobs'] = options.n_jobs
        # create classifier with best parameters
        k_neighbors_classifier = KNeighborsClassifier(**json_with_best_parameters)
        print('\t', k_neighbors_classifier)
        estimators_list.append(('k_neighbors_classifier', k_neighbors_classifier))

    if Classifier.LINEAR_SVC.name in ml_algorithm_list:
        json_with_best_parameters = get_json_with_best_parameters(dataset, Classifier.LINEAR_SVC,
                                                                  use_imdb_multi_class_labels)
        # adding options.verbose in the map
        json_with_best_parameters['verbose'] = options.verbose
        # adding options.random_state in the map
        json_with_best_parameters['random_state'] = options.random_state
        # create classifier with best parameters
        linear_svc = LinearSVC(**json_with_best_parameters)
        print('\t', linear_svc)
        estimators_list.append(('linear_svc', linear_svc))

    if Classifier.LOGISTIC_REGRESSION.name in ml_algorithm_list:
        json_with_best_parameters = get_json_with_best_parameters(dataset, Classifier.LOGISTIC_REGRESSION,
                                                                  use_imdb_multi_class_labels)
        # adding options.n_jobs in the map
        json_with_best_parameters['n_jobs'] = options.n_jobs
        # adding options.verbose in the map
        json_with_best_parameters['verbose'] = options.verbose
        # adding options.random_state in the map
        json_with_best_parameters['random_state'] = options.random_state
        # create classifier with best parameters
        logistic_regression = LogisticRegression(**json_with_best_parameters)
        print('\t', logistic_regression)
        estimators_list.append(('logistic_regression', logistic_regression))

    if Classifier.MULTINOMIAL_NB.name in ml_algorithm_list:
        json_with_best_parameters = get_json_with_best_parameters(dataset, Classifier.MULTINOMIAL_NB,
                                                                  use_imdb_multi_class_labels)
        # create classifier with best parameters
        multinomial_nb = MultinomialNB(**json_with_best_parameters)
        print('\t', multinomial_nb)
        estimators_list.append(('multinomial_nb', multinomial_nb))

    if Classifier.NEAREST_CENTROID.name in ml_algorithm_list:
        json_with_best_parameters = get_json_with_best_parameters(dataset, Classifier.NEAREST_CENTROID,
                                                                  use_imdb_multi_class_labels)
        # create classifier with best parameters
        nearest_centroid = NearestCentroid(**json_with_best_parameters)
        print('\t', nearest_centroid)
        estimators_list.append(('nearest_centroid', nearest_centroid))

    if Classifier.PASSIVE_AGGRESSIVE_CLASSIFIER.name in ml_algorithm_list:
        json_with_best_parameters = get_json_with_best_parameters(dataset, Classifier.PASSIVE_AGGRESSIVE_CLASSIFIER,
                                                                  use_imdb_multi_class_labels)
        # adding options.n_jobs in the map
        json_with_best_parameters['n_jobs'] = options.n_jobs
        # adding options.verbose in the map
        json_with_best_parameters['verbose'] = options.verbose
        # adding options.random_state in the map
        json_with_best_parameters['random_state'] = options.random_state
        # create classifier with best parameters
        passive_aggressive_classifier = PassiveAggressiveClassifier(**json_with_best_parameters)
        print('\t', passive_aggressive_classifier)
        estimators_list.append(('passive_aggressive_classifier', passive_aggressive_classifier))

    if Classifier.PERCEPTRON.name in ml_algorithm_list:
        json_with_best_parameters = get_json_with_best_parameters(dataset, Classifier.PERCEPTRON,
                                                                  use_imdb_multi_class_labels)
        # adding options.n_jobs in the map
        json_with_best_parameters['n_jobs'] = options.n_jobs
        # adding options.verbose in the map
        json_with_best_parameters['verbose'] = options.verbose
        # adding options.random_state in the map
        json_with_best_parameters['random_state'] = options.random_state
        # create classifier with best parameters
        perceptron = Perceptron(**json_with_best_parameters)
        print('\t', perceptron)
        estimators_list.append(('perceptron', perceptron))

    if Classifier.RANDOM_FOREST_CLASSIFIER.name in ml_algorithm_list:
        json_with_best_parameters = get_json_with_best_parameters(dataset, Classifier.RANDOM_FOREST_CLASSIFIER,
                                                                  use_imdb_multi_class_labels)
        # adding options.n_jobs in the map
        json_with_best_parameters['n_jobs'] = options.n_jobs
        # adding options.verbose in the map
        json_with_best_parameters['verbose'] = options.verbose
        # adding options.random_state in the map
        json_with_best_parameters['random_state'] = options.random_state
        # create classifier with best parameters
        random_forest_classifier = RandomForestClassifier(**json_with_best_parameters)
        print('\t', random_forest_classifier)
        estimators_list.append(('random_forest_classifier', random_forest_classifier))

    if Classifier.RIDGE_CLASSIFIER.name in ml_algorithm_list:
        json_with_best_parameters = get_json_with_best_parameters(dataset, Classifier.RIDGE_CLASSIFIER,
                                                                  use_imdb_multi_class_labels)
        # adding options.random_state in the map
        json_with_best_parameters['random_state'] = options.random_state
        # create classifier with best parameters
        ridge_classifier = RidgeClassifier(**json_with_best_parameters)
        print('\t', ridge_classifier)
        estimators_list.append(('ridge_classifier', ridge_classifier))


    if is_stacking_classifier:
        if final_estimator == Classifier.LINEAR_SVC.name:
            return estimators_list, linear_svc
        elif final_estimator == Classifier.LOGISTIC_REGRESSION.name:
            return estimators_list, logistic_regression
        elif final_estimator == Classifier.PASSIVE_AGGRESSIVE_CLASSIFIER.name:
            return estimators_list, passive_aggressive_classifier
        elif final_estimator == Classifier.RIDGE_CLASSIFIER.name:
            return estimators_list, ridge_classifier
        else:
            # Default
            return estimators_list, LinearSVC()

    return estimators_list
