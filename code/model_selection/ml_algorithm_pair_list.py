from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier

from utils.dataset_enum import Dataset
from utils.ml_classifiers_enum import Classifier


def get_ml_algorithm_pair_list(options, ml_algorithm_list, use_classifiers_with_default_parameters, dataset):
    ml_final_list = []

    if Classifier.ADA_BOOST_CLASSIFIER.name in ml_algorithm_list:
        if use_classifiers_with_default_parameters:
            ml_final_list.append \
                ((AdaBoostClassifier(random_state=options.random_state), Classifier.ADA_BOOST_CLASSIFIER))
        else:
            if dataset == Dataset.TWENTY_NEWS_GROUPS.name:
                '''
                03/01/2020 11:03:56 PM - INFO - Classifier: ADA_BOOST_CLASSIFIER, Dataset: TWENTY_NEWS_GROUP
                03/01/2020 11:03:59 PM - INFO - 

                Performing grid search...

                03/01/2020 11:03:59 PM - INFO - 	Parameters:
                03/01/2020 11:03:59 PM - INFO - {'classifier__algorithm': ['SAMME', 'SAMME.R'], 'classifier__learning_rate': [0.01, 0.05, 0.1, 1], 'classifier__n_estimators': [10, 30, 50, 100, 200, 500]}
                03/02/2020 12:14:09 AM - INFO - 	Done in 4210.260s
                03/02/2020 12:14:09 AM - INFO - 	Best score: 0.473
                03/02/2020 12:14:09 AM - INFO - 	Best parameters set:
                03/02/2020 12:14:09 AM - INFO - 		classifier__algorithm: 'SAMME.R'
                03/02/2020 12:14:09 AM - INFO - 		classifier__learning_rate: 1
                03/02/2020 12:14:09 AM - INFO - 		classifier__n_estimators: 200
                '''
                ml_final_list.append((AdaBoostClassifier(random_state=options.random_state, algorithm='SAMME.R', learning_rate=1, n_estimators=200), Classifier.ADA_BOOST_CLASSIFIER))
            elif dataset == Dataset.IMDB_REVIEWS.name:
                '''
                03/02/2020 12:14:15 AM - INFO - Classifier: ADA_BOOST_CLASSIFIER, Dataset: IMDB_REVIEWS
                03/02/2020 12:14:21 AM - INFO -

                Performing grid search...

                03/02/2020 12:14:21 AM - INFO - 	Parameters:
                03/02/2020 12:14:21 AM - INFO - {'classifier__algorithm': ['SAMME', 'SAMME.R'], 'classifier__learning_rate': [0.01, 0.05, 0.1, 1], 'classifier__n_estimators': [10, 30, 50, 100, 200, 500]}
                03/02/2020 01:48:07 AM - INFO - 	Done in 5626.508s
                03/02/2020 01:48:07 AM - INFO - 	Best score: 0.380
                03/02/2020 01:48:07 AM - INFO - 	Best parameters set:
                03/02/2020 01:48:07 AM - INFO - 		classifier__algorithm: 'SAMME.R'
                03/02/2020 01:48:07 AM - INFO - 		classifier__learning_rate: 0.1
                03/02/2020 01:48:07 AM - INFO - 		classifier__n_estimators: 500
                '''
                ml_final_list.append((AdaBoostClassifier(random_state=options.random_state, algorithm='SAMME.R', learning_rate=0.1, n_estimators=500), Classifier.ADA_BOOST_CLASSIFIER))

    if Classifier.BERNOULLI_NB.name in ml_algorithm_list:
        if use_classifiers_with_default_parameters:
            ml_final_list.append((BernoulliNB(), Classifier.BERNOULLI_NB))
        else:
            if dataset == Dataset.TWENTY_NEWS_GROUPS.name:
                '''
                03/02/2020 01:48:20 AM - INFO - Classifier: BERNOULLI_NB, Dataset: TWENTY_NEWS_GROUP
                03/02/2020 01:48:23 AM - INFO -

                Performing grid search...

                03/02/2020 01:48:23 AM - INFO - 	Parameters:
                03/02/2020 01:48:23 AM - INFO - {'classifier__alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0], 'classifier__binarize': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0], 'classifier__fit_prior': [False, True]}
                03/02/2020 01:48:59 AM - INFO - 	Done in 36.482s
                03/02/2020 01:48:59 AM - INFO - 	Best score: 0.721
                03/02/2020 01:48:59 AM - INFO - 	Best parameters set:
                03/02/2020 01:48:59 AM - INFO - 		classifier__alpha: 0.1
                03/02/2020 01:48:59 AM - INFO - 		classifier__binarize: 0.1
                03/02/2020 01:48:59 AM - INFO - 		classifier__fit_prior: False
                '''
                ml_final_list.append((BernoulliNB(alpha=0.1, binarize=0.1, fit_prior=False), Classifier.BERNOULLI_NB))
            elif dataset == Dataset.IMDB_REVIEWS.name:
                '''
                03/02/2020 01:49:00 AM - INFO - Classifier: BERNOULLI_NB, Dataset: IMDB_REVIEWS
                03/02/2020 01:49:06 AM - INFO -

                Performing grid search...

                03/02/2020 01:49:06 AM - INFO - 	Parameters:
                03/02/2020 01:49:06 AM - INFO - {'classifier__alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0], 'classifier__binarize': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0], 'classifier__fit_prior': [False, True]}
                03/02/2020 01:49:33 AM - INFO - 	Done in 27.066s
                03/02/2020 01:49:33 AM - INFO - 	Best score: 0.380
                03/02/2020 01:49:33 AM - INFO - 	Best parameters set:
                03/02/2020 01:49:33 AM - INFO - 		classifier__alpha: 0.5
                03/02/2020 01:49:33 AM - INFO - 		classifier__binarize: 0.0001
                03/02/2020 01:49:33 AM - INFO - 		classifier__fit_prior: True
                '''
                ml_final_list.append((BernoulliNB(alpha=0.5, binarize=0.0001, fit_prior=True), Classifier.BERNOULLI_NB))

    if Classifier.COMPLEMENT_NB.name in ml_algorithm_list:
        if use_classifiers_with_default_parameters:
            ml_final_list.append((ComplementNB(), Classifier.COMPLEMENT_NB))
        else:
            if dataset == Dataset.TWENTY_NEWS_GROUPS.name:
                '''
                03/02/2020 01:49:33 AM - INFO - Classifier: COMPLEMENT_NB, Dataset: TWENTY_NEWS_GROUP
                03/02/2020 01:49:36 AM - INFO -

                Performing grid search...

                03/02/2020 01:49:36 AM - INFO - 	Parameters:
                03/02/2020 01:49:36 AM - INFO - {'classifier__alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0], 'classifier__fit_prior': [False, True], 'classifier__norm': [False, True]}
                03/02/2020 01:49:43 AM - INFO - 	Done in 7.018s
                03/02/2020 01:49:43 AM - INFO - 	Best score: 0.775
                03/02/2020 01:49:43 AM - INFO - 	Best parameters set:
                03/02/2020 01:49:43 AM - INFO - 		classifier__alpha: 0.5
                03/02/2020 01:49:43 AM - INFO - 		classifier__fit_prior: False
                03/02/2020 01:49:43 AM - INFO - 		classifier__norm: False
                '''
                ml_final_list.append((ComplementNB(alpha=0.5, fit_prior=False, norm=False), Classifier.COMPLEMENT_NB))
            elif dataset == Dataset.IMDB_REVIEWS.name:
                '''
                03/02/2020 01:49:43 AM - INFO - Classifier: COMPLEMENT_NB, Dataset: IMDB_REVIEWS
                03/02/2020 01:49:49 AM - INFO -

                Performing grid search...

                03/02/2020 01:49:49 AM - INFO - 	Parameters:
                03/02/2020 01:49:49 AM - INFO - {'classifier__alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0], 'classifier__fit_prior': [False, True], 'classifier__norm': [False, True]}
                03/02/2020 01:49:54 AM - INFO - 	Done in 5.037s
                03/02/2020 01:49:54 AM - INFO - 	Best score: 0.391
                03/02/2020 01:49:54 AM - INFO - 	Best parameters set:
                03/02/2020 01:49:54 AM - INFO - 		classifier__alpha: 0.5
                03/02/2020 01:49:54 AM - INFO - 		classifier__fit_prior: False
                03/02/2020 01:49:54 AM - INFO - 		classifier__norm: False
                '''
                ml_final_list.append((ComplementNB(alpha=0.5, fit_prior=False, norm=False), Classifier.COMPLEMENT_NB))

    if Classifier.DECISION_TREE_CLASSIFIER.name in ml_algorithm_list:
        if use_classifiers_with_default_parameters:
            ml_final_list.append \
                ((DecisionTreeClassifier(random_state=options.random_state), Classifier.DECISION_TREE_CLASSIFIER))
        else:
            if dataset == Dataset.TWENTY_NEWS_GROUPS.name:
                '''
                03/02/2020 01:49:54 AM - INFO - Classifier: DECISION_TREE_CLASSIFIER, Dataset: TWENTY_NEWS_GROUP
                03/02/2020 01:49:57 AM - INFO -

                Performing grid search...

                03/02/2020 01:49:57 AM - INFO - 	Parameters:
                03/02/2020 01:49:57 AM - INFO - {'classifier__criterion': ['gini', 'entropy'], 'classifier__splitter': ['best', 'random'], 'classifier__min_samples_split': range(10, 500, 20), 'classifier__max_depth': range(1, 20, 2)}
                03/02/2020 02:18:00 AM - INFO - 	Done in 1683.200s
                03/02/2020 02:18:00 AM - INFO - 	Best score: 0.301
                03/02/2020 02:18:00 AM - INFO - 	Best parameters set:
                03/02/2020 02:18:00 AM - INFO - 		classifier__criterion: 'entropy'
                03/02/2020 02:18:00 AM - INFO - 		classifier__max_depth: 19
                03/02/2020 02:18:00 AM - INFO - 		classifier__min_samples_split: 110
                03/02/2020 02:18:00 AM - INFO - 		classifier__splitter: 'best'
                '''
                ml_final_list.append((DecisionTreeClassifier(random_state=options.random_state, criterion='entropy', max_depth=19, min_samples_split=110, splitter='best'), Classifier.DECISION_TREE_CLASSIFIER))
            elif dataset == Dataset.IMDB_REVIEWS.name:
                '''
                03/02/2020 02:18:12 AM - INFO - Classifier: DECISION_TREE_CLASSIFIER, Dataset: IMDB_REVIEWS
                03/02/2020 02:18:17 AM - INFO -

                Performing grid search...

                03/02/2020 02:18:17 AM - INFO - 	Parameters:
                03/02/2020 02:18:17 AM - INFO - {'classifier__criterion': ['gini', 'entropy'], 'classifier__splitter': ['best', 'random'], 'classifier__min_samples_split': range(10, 500, 20), 'classifier__max_depth': range(1, 20, 2)}
                03/02/2020 03:11:03 AM - INFO - 	Done in 3165.119s
                03/02/2020 03:11:03 AM - INFO - 	Best score: 0.317
                03/02/2020 03:11:03 AM - INFO - 	Best parameters set:
                03/02/2020 03:11:03 AM - INFO - 		classifier__criterion: 'entropy'
                03/02/2020 03:11:03 AM - INFO - 		classifier__max_depth: 19
                03/02/2020 03:11:03 AM - INFO - 		classifier__min_samples_split: 250
                03/02/2020 03:11:03 AM - INFO - 		classifier__splitter: 'random'
                '''
                ml_final_list.append((DecisionTreeClassifier(random_state=options.random_state, criterion='entropy', max_depth=19, min_samples_split=250, splitter='random'), Classifier.DECISION_TREE_CLASSIFIER))

    if Classifier.EXTRA_TREE_CLASSIFIER.name in ml_algorithm_list:
        if use_classifiers_with_default_parameters:
            ml_final_list.append \
                ((ExtraTreeClassifier(random_state=options.random_state), Classifier.EXTRA_TREE_CLASSIFIER))
        else:
            # TODO: Include best machine learning parameters
            if dataset == Dataset.TWENTY_NEWS_GROUPS.name:
                ml_final_list.append \
                    ((ExtraTreeClassifier(random_state=options.random_state), Classifier.EXTRA_TREE_CLASSIFIER))
            elif dataset == Dataset.IMDB_REVIEWS.name:
                ml_final_list.append \
                    ((ExtraTreeClassifier(random_state=options.random_state), Classifier.EXTRA_TREE_CLASSIFIER))

    if Classifier.EXTRA_TREES_CLASSIFIER.name in ml_algorithm_list:
        if use_classifiers_with_default_parameters:
            ml_final_list.append((ExtraTreesClassifier(n_jobs=options.n_jobs, verbose=options.verbose, random_state=options.random_state), Classifier.EXTRA_TREES_CLASSIFIER))
        else:
            # TODO: Include best machine learning parameters
            if dataset == Dataset.TWENTY_NEWS_GROUPS.name:
                ml_final_list.append((ExtraTreesClassifier(n_jobs=options.n_jobs, verbose=options.verbose, random_state=options.random_state), Classifier.EXTRA_TREES_CLASSIFIER))
            elif dataset == Dataset.IMDB_REVIEWS.name:
                ml_final_list.append((ExtraTreesClassifier(n_jobs=options.n_jobs, verbose=options.verbose, random_state=options.random_state), Classifier.EXTRA_TREES_CLASSIFIER))

    if Classifier.GRADIENT_BOOSTING_CLASSIFIER.name in ml_algorithm_list:
        if use_classifiers_with_default_parameters:
            ml_final_list.append(
                                 (GradientBoostingClassifier(verbose=options.verbose, random_state=options.random_state), Classifier.GRADIENT_BOOSTING_CLASSIFIER))
        else:
            # TODO: Include best machine learning parameters
            if dataset == Dataset.TWENTY_NEWS_GROUPS.name:
                ml_final_list.append((GradientBoostingClassifier(verbose=options.verbose, random_state=options.random_state), Classifier.GRADIENT_BOOSTING_CLASSIFIER))
            elif dataset == Dataset.IMDB_REVIEWS.name:
                ml_final_list.append((GradientBoostingClassifier(verbose=options.verbose, random_state=options.random_state), Classifier.GRADIENT_BOOSTING_CLASSIFIER))

    if Classifier.K_NEIGHBORS_CLASSIFIER.name in ml_algorithm_list:
        if use_classifiers_with_default_parameters:
            ml_final_list.append((KNeighborsClassifier(n_jobs=options.n_jobs), Classifier.K_NEIGHBORS_CLASSIFIER))
        else:
            if dataset == Dataset.TWENTY_NEWS_GROUPS.name:
                '''
                03/02/2020 03:11:41 AM - INFO - Classifier: K_NEIGHBORS_CLASSIFIER, Dataset: TWENTY_NEWS_GROUP
                03/02/2020 03:11:44 AM - INFO -

                Performing grid search...

                03/02/2020 03:11:44 AM - INFO - 	Parameters:
                03/02/2020 03:11:44 AM - INFO - {'classifier__leaf_size': [5, 10, 20, 30, 40, 50, 100], 'classifier__metric': ['euclidean', 'manhattan', 'minkowski'], 'classifier__n_neighbors': [3, 5, 8, 12, 15, 20, 50], 'classifier__weights': ['uniform', 'distance']}
                03/02/2020 03:30:35 AM - INFO - 	Done in 1131.079s
                03/02/2020 03:30:35 AM - INFO - 	Best score: 0.133
                03/02/2020 03:30:35 AM - INFO - 	Best parameters set:
                03/02/2020 03:30:35 AM - INFO - 		classifier__leaf_size: 5
                03/02/2020 03:30:35 AM - INFO - 		classifier__metric: 'euclidean'
                03/02/2020 03:30:35 AM - INFO - 		classifier__n_neighbors: 3
                03/02/2020 03:30:35 AM - INFO - 		classifier__weights: 'distance'
                '''
                ml_final_list.append((KNeighborsClassifier(n_jobs=options.n_jobs, leaf_size=5, metric='euclidean', n_neighbors=3, weights='distance'), Classifier.K_NEIGHBORS_CLASSIFIER))
            elif dataset == Dataset.IMDB_REVIEWS.name:
                '''
                03/02/2020 03:30:37 AM - INFO - Classifier: K_NEIGHBORS_CLASSIFIER, Dataset: IMDB_REVIEWS
                03/02/2020 03:30:43 AM - INFO -

                Performing grid search...

                03/02/2020 03:30:43 AM - INFO - 	Parameters:
                03/02/2020 03:30:43 AM - INFO - {'classifier__leaf_size': [5, 10, 20, 30, 40, 50, 100], 'classifier__metric': ['euclidean', 'manhattan', 'minkowski'], 'classifier__n_neighbors': [3, 5, 8, 12, 15, 20, 50], 'classifier__weights': ['uniform', 'distance']}
                03/02/2020 05:58:23 AM - INFO - 	Done in 8859.881s
                03/02/2020 05:58:23 AM - INFO - 	Best score: 0.376
                03/02/2020 05:58:23 AM - INFO - 	Best parameters set:
                03/02/2020 05:58:23 AM - INFO - 		classifier__leaf_size: 5
                03/02/2020 05:58:23 AM - INFO - 		classifier__metric: 'euclidean'
                03/02/2020 05:58:23 AM - INFO - 		classifier__n_neighbors: 50
                03/02/2020 05:58:23 AM - INFO - 		classifier__weights: 'distance'
                '''
                ml_final_list.append((KNeighborsClassifier(n_jobs=options.n_jobs, leaf_size=5, metric='euclidean', n_neighbors=50, weights='distance'), Classifier.K_NEIGHBORS_CLASSIFIER))

    if Classifier.LINEAR_SVC.name in ml_algorithm_list:
        if use_classifiers_with_default_parameters:
            ml_final_list.append \
                ((LinearSVC(verbose=options.verbose, random_state=options.random_state), Classifier.LINEAR_SVC))
        else:
            if dataset == Dataset.TWENTY_NEWS_GROUPS.name:
                '''
                03/02/2020 05:58:49 AM - INFO - Classifier: LINEAR_SVC, Dataset: TWENTY_NEWS_GROUP
                03/02/2020 05:58:52 AM - INFO -

                Performing grid search...

                03/02/2020 05:58:52 AM - INFO - 	Parameters:
                03/02/2020 05:58:52 AM - INFO - {'classifier__C': [0.01, 10.01, 20.01, 30.01, 40.01, 50.01, 60.01, 70.01, 80.01, 90.01, 1.0], 'classifier__dual': [False, True], 'classifier__max_iter': [100, 1000, 5000], 'classifier__multi_class': ['ovr', 'crammer_singer'], 'classifier__tol': [0.0001, 0.001, 0.01, 0.1]}
                03/02/2020 11:28:11 AM - INFO - 	Done in 19758.637s
                03/02/2020 11:28:11 AM - INFO - 	Best score: 0.761
                03/02/2020 11:28:11 AM - INFO - 	Best parameters set:
                03/02/2020 11:28:11 AM - INFO - 		classifier__C: 1.0
                03/02/2020 11:28:11 AM - INFO - 		classifier__dual: True
                03/02/2020 11:28:11 AM - INFO - 		classifier__max_iter: 100
                03/02/2020 11:28:11 AM - INFO - 		classifier__multi_class: 'ovr'
                03/02/2020 11:28:11 AM - INFO - 		classifier__tol: 0.1
                '''
                ml_final_list.append((LinearSVC(verbose=options.verbose, random_state=options.random_state, C=1.0, dual=True, max_iter=100, multi_class='ovr', tol=0.1), Classifier.LINEAR_SVC))
            elif dataset == Dataset.IMDB_REVIEWS.name:
                '''
                03/02/2020 11:28:12 AM - INFO - Classifier: LINEAR_SVC, Dataset: IMDB_REVIEWS
                03/02/2020 11:32:08 AM - INFO -

                Performing grid search...

                03/02/2020 11:32:08 AM - INFO - 	Parameters:
                03/02/2020 11:32:08 AM - INFO - {'classifier__C': [0.01, 10.01, 20.01, 30.01, 40.01, 50.01, 60.01, 70.01, 80.01, 90.01, 1.0], 'classifier__dual': [False, True], 'classifier__max_iter': [100, 1000, 5000], 'classifier__multi_class': ['ovr', 'crammer_singer'], 'classifier__tol': [0.0001, 0.001, 0.01, 0.1]}
                03/02/2020 02:28:32 PM - INFO - 	Done in 10584.167s
                03/02/2020 02:28:32 PM - INFO - 	Best score: 0.410
                03/02/2020 02:28:32 PM - INFO - 	Best parameters set:
                03/02/2020 02:28:32 PM - INFO - 		classifier__C: 0.01
                03/02/2020 02:28:32 PM - INFO - 		classifier__dual: True
                03/02/2020 02:28:32 PM - INFO - 		classifier__max_iter: 5000
                03/02/2020 02:28:32 PM - INFO - 		classifier__multi_class: 'crammer_singer'
                03/02/2020 02:28:32 PM - INFO - 		classifier__tol: 0.001
                '''
                ml_final_list.append((LinearSVC(verbose=options.verbose, random_state=options.random_state, C=0.01, dual=True, max_iter=5000, multi_class='crammer_singer', tol=0.001), Classifier.LINEAR_SVC))

    if Classifier.LOGISTIC_REGRESSION.name in ml_algorithm_list:
        if use_classifiers_with_default_parameters:
            ml_final_list.append((LogisticRegression(n_jobs=options.n_jobs, verbose=options.verbose, random_state=options.random_state), Classifier.LOGISTIC_REGRESSION))
        else:
            if dataset == Dataset.TWENTY_NEWS_GROUPS.name:
                '''
                03/02/2020 02:28:34 PM - INFO - Classifier: LOGISTIC_REGRESSION, Dataset: TWENTY_NEWS_GROUP
                03/02/2020 02:28:37 PM - INFO -

                Performing grid search...

                03/02/2020 02:28:37 PM - INFO - 	Parameters:
                03/02/2020 02:28:37 PM - INFO - {'classifier__C': [0.01, 10.01, 20.01, 30.01, 40.01, 50.01, 60.01, 70.01, 80.01, 90.01, 1.0], 'classifier__dual': [False, True], 'classifier__max_iter': [100, 1000, 5000], 'classifier__multi_class': ['ovr', 'multinomial'], 'classifier__tol': [0.0001, 0.001, 0.01, 0.1]}
                03/02/2020 07:45:04 PM - INFO - 	Done in 18986.225s
                03/02/2020 07:45:04 PM - INFO - 	Best score: 0.759
                03/02/2020 07:45:04 PM - INFO - 	Best parameters set:
                03/02/2020 07:45:04 PM - INFO - 		classifier__C: 10.01
                03/02/2020 07:45:04 PM - INFO - 		classifier__dual: False
                03/02/2020 07:45:04 PM - INFO - 		classifier__max_iter: 100
                03/02/2020 07:45:04 PM - INFO - 		classifier__multi_class: 'ovr'
                03/02/2020 07:45:04 PM - INFO - 		classifier__tol: 0.01
                '''
                ml_final_list.append((LogisticRegression(n_jobs=options.n_jobs, verbose=options.verbose, random_state=options.random_state, C=10.01, dual=False, max_iter=100, multi_class='ovr', tol=0.01), Classifier.LOGISTIC_REGRESSION))
            elif dataset == Dataset.IMDB_REVIEWS.name:
                '''
                03/02/2020 07:45:28 PM - INFO - Classifier: LOGISTIC_REGRESSION, Dataset: IMDB_REVIEWS
                03/02/2020 07:45:35 PM - INFO -

                Performing grid search...

                03/02/2020 07:45:35 PM - INFO - 	Parameters:
                03/02/2020 07:45:35 PM - INFO - {'classifier__C': [0.01, 10.01, 20.01, 30.01, 40.01, 50.01, 60.01, 70.01, 80.01, 90.01, 1.0], 'classifier__dual': [False, True], 'classifier__max_iter': [100, 1000, 5000], 'classifier__multi_class': ['ovr', 'multinomial'], 'classifier__tol': [0.0001, 0.001, 0.01, 0.1]}
                03/02/2020 11:51:46 PM - INFO - 	Done in 14771.776s
                03/02/2020 11:51:46 PM - INFO - 	Best score: 0.424
                03/02/2020 11:51:46 PM - INFO - 	Best parameters set:
                03/02/2020 11:51:46 PM - INFO - 		classifier__C: 1.0
                03/02/2020 11:51:46 PM - INFO - 		classifier__dual: False
                03/02/2020 11:51:46 PM - INFO - 		classifier__max_iter: 100
                03/02/2020 11:51:46 PM - INFO - 		classifier__multi_class: 'ovr'
                03/02/2020 11:51:46 PM - INFO - 		classifier__tol: 0.01
                '''
                ml_final_list.append((LogisticRegression(n_jobs=options.n_jobs, verbose=options.verbose, random_state=options.random_state, C=1.0, dual=False, max_iter=100, multi_class='ovr', tol=0.01), Classifier.LOGISTIC_REGRESSION))

    if Classifier.LOGISTIC_REGRESSION_CV.name in ml_algorithm_list:
        if use_classifiers_with_default_parameters:
            ml_final_list.append((LogisticRegressionCV(n_jobs=options.n_jobs, verbose=options.verbose, random_state=options.random_state), Classifier.LOGISTIC_REGRESSION_CV))
        else:
            # TODO: Include best machine learning parameters
            if dataset == Dataset.TWENTY_NEWS_GROUPS.name:
                ml_final_list.append((LogisticRegressionCV(n_jobs=options.n_jobs, verbose=options.verbose, random_state=options.random_state), Classifier.LOGISTIC_REGRESSION_CV))
            elif dataset == Dataset.IMDB_REVIEWS.name:
                ml_final_list.append((LogisticRegressionCV(n_jobs=options.n_jobs, verbose=options.verbose, random_state=options.random_state), Classifier.LOGISTIC_REGRESSION_CV))

    if Classifier.MLP_CLASSIFIER.name in ml_algorithm_list:
        if use_classifiers_with_default_parameters:
            ml_final_list.append \
                ((MLPClassifier(verbose=options.verbose, random_state=options.random_state), Classifier.MLP_CLASSIFIER))
        else:
            # TODO: Include best machine learning parameters
            if dataset == Dataset.TWENTY_NEWS_GROUPS.name:
                ml_final_list.append((MLPClassifier(verbose=options.verbose, random_state=options.random_state), Classifier.MLP_CLASSIFIER))
            elif dataset == Dataset.IMDB_REVIEWS.name:
                ml_final_list.append((MLPClassifier(verbose=options.verbose, random_state=options.random_state), Classifier.MLP_CLASSIFIER))

    if Classifier.MULTINOMIAL_NB.name in ml_algorithm_list:
        if use_classifiers_with_default_parameters:
            ml_final_list.append((MultinomialNB(), Classifier.MULTINOMIAL_NB))
        else:
            if dataset == Dataset.TWENTY_NEWS_GROUPS.name:
                '''
                03/02/2020 11:52:04 PM - INFO - Classifier: MULTINOMIAL_NB, Dataset: TWENTY_NEWS_GROUP
                03/02/2020 11:52:07 PM - INFO -

                Performing grid search...

                03/02/2020 11:52:07 PM - INFO - 	Parameters:
                03/02/2020 11:52:07 PM - INFO - {'classifier__alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0], 'classifier__fit_prior': [False, True]}
                03/02/2020 11:52:10 PM - INFO - 	Done in 3.098s
                03/02/2020 11:52:10 PM - INFO - 	Best score: 0.763
                03/02/2020 11:52:10 PM - INFO - 	Best parameters set:
                03/02/2020 11:52:10 PM - INFO - 		classifier__alpha: 0.01
                03/02/2020 11:52:10 PM - INFO - 		classifier__fit_prior: True
                '''
                ml_final_list.append((MultinomialNB(alpha=0.01, fit_prior=True), Classifier.MULTINOMIAL_NB))
            elif dataset == Dataset.IMDB_REVIEWS.name:
                '''
                03/02/2020 11:52:10 PM - INFO - Classifier: MULTINOMIAL_NB, Dataset: IMDB_REVIEWS
                03/02/2020 11:52:16 PM - INFO -

                Performing grid search...

                03/02/2020 11:52:16 PM - INFO - 	Parameters:
                03/02/2020 11:52:16 PM - INFO - {'classifier__alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0], 'classifier__fit_prior': [False, True]}
                03/02/2020 11:52:19 PM - INFO - 	Done in 2.561s
                03/02/2020 11:52:19 PM - INFO - 	Best score: 0.393
                03/02/2020 11:52:19 PM - INFO - 	Best parameters set:
                03/02/2020 11:52:19 PM - INFO - 		classifier__alpha: 0.1
                03/02/2020 11:52:19 PM - INFO - 		classifier__fit_prior: True
                '''
                ml_final_list.append((MultinomialNB(alpha=0.1, fit_prior=True), Classifier.MULTINOMIAL_NB))

    if Classifier.NEAREST_CENTROID.name in ml_algorithm_list:
        if use_classifiers_with_default_parameters:
            ml_final_list.append((NearestCentroid(), Classifier.NEAREST_CENTROID))
        else:
            if dataset == Dataset.TWENTY_NEWS_GROUPS.name:
                '''
                03/02/2020 11:52:19 PM - INFO - Classifier: NEAREST_CENTROID, Dataset: TWENTY_NEWS_GROUP
                03/02/2020 11:52:22 PM - INFO -

                Performing grid search...

                03/02/2020 11:52:22 PM - INFO - 	Parameters:
                03/02/2020 11:52:22 PM - INFO - {'classifier__metric': ['euclidean', 'cosine']}
                03/02/2020 11:52:22 PM - INFO - 	Done in 0.203s
                03/02/2020 11:52:22 PM - INFO - 	Best score: 0.715
                03/02/2020 11:52:22 PM - INFO - 	Best parameters set:
                03/02/2020 11:52:22 PM - INFO - 		classifier__metric: 'cosine'
                '''
                ml_final_list.append((NearestCentroid(metric='cosine'), Classifier.NEAREST_CENTROID))
            elif dataset == Dataset.IMDB_REVIEWS.name:
                '''
                03/02/2020 11:52:22 PM - INFO - Classifier: NEAREST_CENTROID, Dataset: IMDB_REVIEWS
                03/02/2020 11:52:28 PM - INFO -

                Performing grid search...

                03/02/2020 11:52:28 PM - INFO - 	Parameters:
                03/02/2020 11:52:28 PM - INFO - {'classifier__metric': ['euclidean', 'cosine']}
                03/02/2020 11:52:29 PM - INFO - 	Done in 0.378s
                03/02/2020 11:52:29 PM - INFO - 	Best score: 0.379
                03/02/2020 11:52:29 PM - INFO - 	Best parameters set:
                03/02/2020 11:52:29 PM - INFO - 		classifier__metric: 'cosine'
                '''
                ml_final_list.append((NearestCentroid(metric='cosine'), Classifier.NEAREST_CENTROID))

    if Classifier.NU_SVC.name in ml_algorithm_list:
        if use_classifiers_with_default_parameters:
            ml_final_list.append((NuSVC(verbose=options.verbose, random_state=options.random_state), Classifier.NU_SVC))
        else:
            # TODO: Include best machine learning parameters
            if dataset == Dataset.TWENTY_NEWS_GROUPS.name:
                ml_final_list.append \
                    ((NuSVC(verbose=options.verbose, random_state=options.random_state), Classifier.NU_SVC))
            elif dataset == Dataset.IMDB_REVIEWS.name:
                ml_final_list.append \
                    ((NuSVC(verbose=options.verbose, random_state=options.random_state), Classifier.NU_SVC))

    if Classifier.PASSIVE_AGGRESSIVE_CLASSIFIER.name in ml_algorithm_list:
        if use_classifiers_with_default_parameters:
            ml_final_list.append((PassiveAggressiveClassifier(n_jobs=options.n_jobs, verbose=options.verbose, random_state=options.random_state), Classifier.PASSIVE_AGGRESSIVE_CLASSIFIER))
        else:
            if dataset == Dataset.TWENTY_NEWS_GROUPS.name:
                '''
                03/02/2020 11:52:29 PM - INFO - Classifier: PASSIVE_AGGRESSIVE_CLASSIFIER, Dataset: TWENTY_NEWS_GROUP
                03/02/2020 11:52:32 PM - INFO -

                Performing grid search...

                03/02/2020 11:52:32 PM - INFO - 	Parameters:
                03/02/2020 11:52:32 PM - INFO - {'classifier__C': [0.01, 10.01, 20.01, 30.01, 40.01, 50.01, 60.01, 70.01, 80.01, 90.01, 1.0], 'classifier__average': [False, True], 'classifier__class_weight': ['balanced', None], 'classifier__early_stopping': [False, True], 'classifier__max_iter': [100, 1000, 5000], 'classifier__n_iter_no_change': [3, 5, 10, 15], 'classifier__tol': [0.0001, 0.001, 0.01, 0.1], 'classifier__validation_fraction': [0.0001, 0.001, 0.01, 0.1]}
                03/03/2020 06:29:41 AM - INFO - 	Done in 23828.898s
                03/03/2020 06:29:41 AM - INFO - 	Best score: 0.764
                03/03/2020 06:29:41 AM - INFO - 	Best parameters set:
                03/03/2020 06:29:41 AM - INFO - 		classifier__C: 0.01
                03/03/2020 06:29:41 AM - INFO - 		classifier__average: False
                03/03/2020 06:29:41 AM - INFO - 		classifier__class_weight: 'balanced'
                03/03/2020 06:29:41 AM - INFO - 		classifier__early_stopping: False
                03/03/2020 06:29:41 AM - INFO - 		classifier__max_iter: 100
                03/03/2020 06:29:41 AM - INFO - 		classifier__n_iter_no_change: 5
                03/03/2020 06:29:41 AM - INFO - 		classifier__tol: 0.0001
                03/03/2020 06:29:41 AM - INFO - 		classifier__validation_fraction: 0.1
                '''
                ml_final_list.append((PassiveAggressiveClassifier(n_jobs=options.n_jobs, verbose=options.verbose, random_state=options.random_state, C=0.01, average=False, class_weight='balanced', early_stopping=False, max_iter=100, n_iter_no_change=5, tol=0.0001, validation_fraction=0.1), Classifier.PASSIVE_AGGRESSIVE_CLASSIFIER))
            elif dataset == Dataset.IMDB_REVIEWS.name:
                '''
                03/03/2020 06:29:42 AM - INFO - Classifier: PASSIVE_AGGRESSIVE_CLASSIFIER, Dataset: IMDB_REVIEWS
                03/03/2020 06:29:48 AM - INFO -

                Performing grid search...

                03/03/2020 06:29:48 AM - INFO - 	Parameters:
                03/03/2020 06:29:48 AM - INFO - {'classifier__C': [0.01, 10.01, 20.01, 30.01, 40.01, 50.01, 60.01, 70.01, 80.01, 90.01, 1.0], 'classifier__average': [False, True], 'classifier__class_weight': ['balanced', None], 'classifier__early_stopping': [False, True], 'classifier__max_iter': [100, 1000, 5000], 'classifier__n_iter_no_change': [3, 5, 10, 15], 'classifier__tol': [0.0001, 0.001, 0.01, 0.1], 'classifier__validation_fraction': [0.0001, 0.001, 0.01, 0.1]}
                03/03/2020 05:03:40 PM - INFO - 	Done in 38032.286s
                03/03/2020 05:03:40 PM - INFO - 	Best score: 0.418
                03/03/2020 05:03:40 PM - INFO - 	Best parameters set:
                03/03/2020 05:03:40 PM - INFO - 		classifier__C: 0.01
                03/03/2020 05:03:40 PM - INFO - 		classifier__average: False
                03/03/2020 05:03:40 PM - INFO - 		classifier__class_weight: None
                03/03/2020 05:03:40 PM - INFO - 		classifier__early_stopping: True
                03/03/2020 05:03:40 PM - INFO - 		classifier__max_iter: 100
                03/03/2020 05:03:40 PM - INFO - 		classifier__n_iter_no_change: 5
                03/03/2020 05:03:40 PM - INFO - 		classifier__tol: 0.01
                03/03/2020 05:03:40 PM - INFO - 		classifier__validation_fraction: 0.01
                '''
                ml_final_list.append((PassiveAggressiveClassifier(n_jobs=options.n_jobs, verbose=options.verbose, random_state=options.random_state, C=0.01, average=False, class_weight=None, early_stopping=True, max_iter=100, n_iter_no_change=5, tol=0.01, validation_fraction=0.01), Classifier.PASSIVE_AGGRESSIVE_CLASSIFIER))

    if Classifier.PERCEPTRON.name in ml_algorithm_list:
        if use_classifiers_with_default_parameters:
            ml_final_list.append((Perceptron(n_jobs=options.n_jobs, verbose=options.verbose, random_state=options.random_state), Classifier.PERCEPTRON))
        else:
            if dataset == Dataset.TWENTY_NEWS_GROUPS.name:
                '''
                03/03/2020 05:03:42 PM - INFO - Classifier: PERCEPTRON, Dataset: TWENTY_NEWS_GROUP
                03/03/2020 05:03:45 PM - INFO -

                Performing grid search...

                03/03/2020 05:03:45 PM - INFO - 	Parameters:
                03/03/2020 05:03:45 PM - INFO - {'classifier__alpha': [0.0001, 0.001, 0.01, 0.1], 'classifier__class_weight': ['balanced', None], 'classifier__early_stopping': [False, True], 'classifier__max_iter': [100, 1000, 5000], 'classifier__n_iter_no_change': [3, 5, 10, 15], 'classifier__penalty': ['l2', 'l1', 'elasticnet'], 'classifier__tol': [0.0001, 0.001, 0.01, 0.1], 'classifier__validation_fraction': [0.0001, 0.001, 0.01, 0.1]}
                03/03/2020 08:24:28 PM - INFO - 	Done in 12043.305s
                03/03/2020 08:24:28 PM - INFO - 	Best score: 0.626
                03/03/2020 08:24:28 PM - INFO - 	Best parameters set:
                03/03/2020 08:24:28 PM - INFO - 		classifier__alpha: 0.0001
                03/03/2020 08:24:28 PM - INFO - 		classifier__class_weight: 'balanced'
                03/03/2020 08:24:28 PM - INFO - 		classifier__early_stopping: True
                03/03/2020 08:24:28 PM - INFO - 		classifier__max_iter: 100
                03/03/2020 08:24:28 PM - INFO - 		classifier__n_iter_no_change: 15
                03/03/2020 08:24:28 PM - INFO - 		classifier__penalty: 'l2'
                03/03/2020 08:24:28 PM - INFO - 		classifier__tol: 0.1
                03/03/2020 08:24:28 PM - INFO - 		classifier__validation_fraction: 0.01
                '''
                ml_final_list.append((Perceptron(n_jobs=options.n_jobs, verbose=options.verbose, random_state=options.random_state, alpha=0.0001, class_weight='balanced', early_stopping=True, max_iter=100, n_iter_no_change=15, penalty='l2', tol=0.1, validation_fraction=0.01), Classifier.PERCEPTRON))
            elif dataset == Dataset.IMDB_REVIEWS.name:
                '''
                03/03/2020 08:24:29 PM - INFO - Classifier: PERCEPTRON, Dataset: IMDB_REVIEWS
                03/03/2020 08:24:35 PM - INFO -

                Performing grid search...

                03/03/2020 08:24:35 PM - INFO - 	Parameters:
                03/03/2020 08:24:35 PM - INFO - {'classifier__alpha': [0.0001, 0.001, 0.01, 0.1], 'classifier__class_weight': ['balanced', None], 'classifier__early_stopping': [False, True], 'classifier__max_iter': [100, 1000, 5000], 'classifier__n_iter_no_change': [3, 5, 10, 15], 'classifier__penalty': ['l2', 'l1', 'elasticnet'], 'classifier__tol': [0.0001, 0.001, 0.01, 0.1], 'classifier__validation_fraction': [0.0001, 0.001, 0.01, 0.1]}
                03/04/2020 12:51:33 AM - INFO - 	Done in 16017.731s
                03/04/2020 12:51:33 AM - INFO - 	Best score: 0.326
                03/04/2020 12:51:33 AM - INFO - 	Best parameters set:
                03/04/2020 12:51:33 AM - INFO - 		classifier__alpha: 0.0001
                03/04/2020 12:51:33 AM - INFO - 		classifier__class_weight: None
                03/04/2020 12:51:33 AM - INFO - 		classifier__early_stopping: True
                03/04/2020 12:51:33 AM - INFO - 		classifier__max_iter: 100
                03/04/2020 12:51:33 AM - INFO - 		classifier__n_iter_no_change: 3
                03/04/2020 12:51:33 AM - INFO - 		classifier__penalty: 'l2'
                03/04/2020 12:51:33 AM - INFO - 		classifier__tol: 0.0001
                03/04/2020 12:51:33 AM - INFO - 		classifier__validation_fraction: 0.001
                '''
                ml_final_list.append((Perceptron(n_jobs=options.n_jobs, verbose=options.verbose, random_state=options.random_state, alpha=0.0001, class_weight=None, early_stopping=True, max_iter=100, n_iter_no_change=3, penalty='l2', tol=0.0001, validation_fraction=0.001), Classifier.PERCEPTRON))

    if Classifier.RANDOM_FOREST_CLASSIFIER.name in ml_algorithm_list:
        if use_classifiers_with_default_parameters:
            ml_final_list.append((RandomForestClassifier(n_jobs=options.n_jobs, verbose=options.verbose, random_state=options.random_state), Classifier.RANDOM_FOREST_CLASSIFIER))
        else:
            # TODO: Include best machine learning parameters
            if dataset == Dataset.TWENTY_NEWS_GROUPS.name:
                ml_final_list.append((RandomForestClassifier(n_jobs=options.n_jobs, verbose=options.verbose, random_state=options.random_state), Classifier.RANDOM_FOREST_CLASSIFIER))
            elif dataset == Dataset.IMDB_REVIEWS.name:
                ml_final_list.append((RandomForestClassifier(n_jobs=options.n_jobs, verbose=options.verbose, random_state=options.random_state), Classifier.RANDOM_FOREST_CLASSIFIER))

    if Classifier.RIDGE_CLASSIFIER.name in ml_algorithm_list:
        if use_classifiers_with_default_parameters:
            ml_final_list.append((RidgeClassifier(random_state=options.random_state), Classifier.RIDGE_CLASSIFIER))
        else:
            # TODO: Include best machine learning parameters
            if dataset == Dataset.TWENTY_NEWS_GROUPS.name:
                ml_final_list.append((RidgeClassifier(random_state=options.random_state), Classifier.RIDGE_CLASSIFIER))
            elif dataset == Dataset.IMDB_REVIEWS.name:
                ml_final_list.append((RidgeClassifier(random_state=options.random_state), Classifier.RIDGE_CLASSIFIER))

    if Classifier.RIDGE_CLASSIFIERCV.name in ml_algorithm_list:
        if use_classifiers_with_default_parameters:
            ml_final_list.append((RidgeClassifierCV(), Classifier.RIDGE_CLASSIFIERCV))
        else:
            # TODO: Include best machine learning parameters
            if dataset == Dataset.TWENTY_NEWS_GROUPS.name:
                ml_final_list.append((RidgeClassifierCV(), Classifier.RIDGE_CLASSIFIERCV))
            elif dataset == Dataset.IMDB_REVIEWS.name:
                ml_final_list.append((RidgeClassifierCV(), Classifier.RIDGE_CLASSIFIERCV))

    if Classifier.SGD_CLASSIFIER.name in ml_algorithm_list:
        if use_classifiers_with_default_parameters:
            ml_final_list.append((SGDClassifier(n_jobs=options.n_jobs, verbose=options.verbose, random_state=options.random_state), Classifier.SGD_CLASSIFIER))
        else:
            # TODO: Include best machine learning parameters
            if dataset == Dataset.TWENTY_NEWS_GROUPS.name:
                ml_final_list.append((SGDClassifier(n_jobs=options.n_jobs, verbose=options.verbose, random_state=options.random_state), Classifier.SGD_CLASSIFIER))
            elif dataset == Dataset.IMDB_REVIEWS.name:
                ml_final_list.append((SGDClassifier(n_jobs=options.n_jobs, verbose=options.verbose, random_state=options.random_state), Classifier.SGD_CLASSIFIER))

    return ml_final_list
