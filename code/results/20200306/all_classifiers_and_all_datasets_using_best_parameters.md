## Results using all classifiers and all datasets (classifier with best parameters)

### TWENTY_NEWS_GROUPS dataset (removing headers signatures and quoting)  

#### IMDB_REVIEWS dataset (binary classification)

```
/home/ramon/virtual_envs/comp551_p2/bin/python /home/ramon/github/comp551-2020-p2_classification_of_textual_data/code/main.py
03/05/2020 11:50:56 PM - INFO - Program started...
usage: main.py [-h] [-d DATASET] [-ml ML_ALGORITHM_LIST]
               [-use_default_parameters] [-not_shuffle] [-n_jobs N_JOBS] [-cv]
               [-n_splits N_SPLITS] [-required_classifiers]
               [-news_with_4_classes] [-news_no_filter] [-imdb_multi_class]
               [-show_reviews] [-r] [-m] [--chi2_select CHI2_SELECT] [-cm]
               [-use_hashing] [-use_count] [-n_features N_FEATURES]
               [-plot_time] [-save_logs] [-verbose]
               [-random_state RANDOM_STATE] [-v]

MiniProject 2: Classification of textual data. Authors: Ramon Figueiredo
Pessoa, Rafael Gomes Braga, Ege Odaci

optional arguments:
  -h, --help            show this help message and exit
  -d DATASET, --dataset DATASET
                        Dataset used (Options: TWENTY_NEWS_GROUP OR
                        IMDB_REVIEWS). Default: ALL
  -ml ML_ALGORITHM_LIST, --ml_algorithm_list ML_ALGORITHM_LIST
                        List of machine learning algorithm to be executed.
                        This stores a list of ML algorithms, and appends each
                        algorithm value to the list. For example: -ml
                        LINEAR_SVC -ml RANDOM_FOREST_CLASSIFIER, means
                        ml_algorithm_list = ['LINEAR_SVC',
                        'RANDOM_FOREST_CLASSIFIER']. (Options of ML
                        algorithms: 1) ADA_BOOST_CLASSIFIER, 2) BERNOULLI_NB,
                        3) COMPLEMENT_NB, 4) DECISION_TREE_CLASSIFIER, 5)
                        EXTRA_TREE_CLASSIFIER, 6) EXTRA_TREES_CLASSIFIER, 7)
                        GRADIENT_BOOSTING_CLASSIFIER, 8)
                        K_NEIGHBORS_CLASSIFIER, 9) LINEAR_SVC, 10)
                        LOGISTIC_REGRESSION, 11) LOGISTIC_REGRESSION_CV, 12)
                        MLP_CLASSIFIER, 13) MULTINOMIAL_NB, 14)
                        NEAREST_CENTROID, 15) NU_SVC, 16)
                        PASSIVE_AGGRESSIVE_CLASSIFIER, 17) PERCEPTRON, 18)
                        RANDOM_FOREST_CLASSIFIER, 19) RIDGE_CLASSIFIER, 20)
                        RIDGE_CLASSIFIERCV, 21) SGD_CLASSIFIER,). Default:
                        None. If ml_algorithm_list is not provided, all ML
                        algorithms will be executed.
  -use_default_parameters, --use_classifiers_with_default_parameters
                        Use classifiers with default parameters. Default:
                        False = Use classifiers with best parameters found
                        using grid search.
  -not_shuffle, --not_shuffle_dataset
                        Read dataset without shuffle data. Default: False
  -n_jobs N_JOBS        The number of CPUs to use to do the computation. If
                        the provided number is negative or greater than the
                        number of available CPUs, the system will use all the
                        available CPUs. Default: -1 (-1 == all CPUs)
  -cv, --run_cross_validation
                        Run cross validation. Default: False
  -n_splits N_SPLITS    Number of cross validation folds. Default: 5. Must be
                        at least 2. Default: 5
  -required_classifiers, --use_just_miniproject_classifiers
                        Use just the miniproject classifiers (1.
                        LogisticRegression, 2. DecisionTreeClassifier, 3.
                        LinearSVC (L1), 4. LinearSVC (L2), 5.
                        AdaBoostClassifier, 6. RandomForestClassifier).
                        Default: False
  -news_with_4_classes, --twenty_news_using_four_categories
                        TWENTY_NEWS_GROUP dataset using some categories
                        ('alt.atheism', 'talk.religion.misc', 'comp.graphics',
                        'sci.space'). Default: False (use all categories).
                        Default: False
  -news_no_filter, --twenty_news_with_no_filter
                        Do not remove newsgroup information that is easily
                        overfit: ('headers', 'footers', 'quotes'). Default:
                        False
  -imdb_multi_class, --use_imdb_multi_class_labels
                        Use IMDB multi-class labels (review score: 1, 2, 3, 4,
                        7, 8, 9, 10). If --use_imdb_multi_class_labels is
                        False, the system uses binary classification. (0 = neg
                        and 1 = pos) Default: False
  -show_reviews, --show_imdb_reviews
                        Show the IMDB_REVIEWS and respective labels while read
                        the dataset. Default: False
  -r, --report          Print a detailed classification report.
  -m, --all_metrics     Print all classification metrics.
  --chi2_select CHI2_SELECT
                        Select some number of features using a chi-squared
                        test
  -cm, --confusion_matrix
                        Print the confusion matrix.
  -use_hashing, --use_hashing_vectorizer
                        Use a hashing vectorizer. Default: False
  -use_count, --use_count_vectorizer
                        Use a count vectorizer. Default: False
  -n_features N_FEATURES, --n_features_using_hashing N_FEATURES
                        n_features when using the hashing vectorizer. Default:
                        65536
  -plot_time, --plot_accurary_and_time_together
                        Plot training time and test time together with
                        accuracy score. Default: False (Plot just accuracy)
  -save_logs, --save_logs_in_file
                        Save logs in a file. Default: False (show logs in the
                        prompt)
  -verbose, --verbosity
                        Increase output verbosity. Default: False
  -random_state RANDOM_STATE
                        Seed used by the random number generator. Default: 0
  -v, --version         show program's version number and exit

COMP 551 (001/002), Applied Machine Learning, Winter 2020, McGill University.
None
==================================================================================================================================

Running with options: 
	Dataset = ALL
	ML algorithm list (If ml_algorithm_list is not provided, all ML algorithms will be executed) = None
	Use classifiers with default parameters. Default: False = Use classifiers with best parameters found using grid search. False
	Read dataset without shuffle data = False
	The number of CPUs to use to do the computation. If the provided number is negative or greater than the number of available CPUs, the system will use all the available CPUs. Default: -1 (-1 == all CPUs) = -1
	Run cross validation. Default: False = False
	Number of cross validation folds. Default: 5 = 5
	Use just the miniproject classifiers (1. LogisticRegression, 2. DecisionTreeClassifier, 3. LinearSVC, 4. AdaBoostClassifier, 5. RandomForestClassifier) =  False
	TWENTY_NEWS_GROUPS dataset using some categories (alt.atheism, talk.religion.misc, comp.graphics, sci.space) = False
	Do not remove newsgroup information that is easily overfit (headers, footers, quotes) = False
	Use IMDB multi-class labels (review score: 1, 2, 3, 4, 7, 8, 9, 10). If --use_imdb_multi_class_labels is False, the system uses binary classification (0 = neg and 1 = pos). Default: False = False
	Show the IMDB_REVIEWS and respective labels while read the dataset = False
	Print classification report = False
	Print all classification metrics (accuracy score, precision score, recall score, f1 score, f-beta score, jaccard score) =  False
	Select some number of features using a chi-squared test (For example: --chi2_select 10 = select 10 features using a chi-squared test) =  No number provided
	Print the confusion matrix = False
	Use a hashing vectorizer = False
	Use a count vectorizer = False
	Use a tf-idf vectorizer = True
	N features when using the hashing vectorizer = 65536
	Plot training time and test time together with accuracy score = False
	Save logs in a file = False
	Seed used by the random number generator (random_state) = 0
	Verbose = False
==================================================================================================================================

Loading TWENTY_NEWS_GROUPS dataset for categories:
data loaded
11314 documents - 13.782MB (training set)
7532 documents - 8.262MB (test set)
20 categories

Extracting features from the training data using a vectorizer
done in 1.980865s at 6.958MB/s
n_samples: 11314, n_features: 101321

Extracting features from the test data using the same vectorizer
done in 0.984851s at 8.389MB/s
n_samples: 7532, n_features: 101321

================================================================================
Classifier.ADA_BOOST_CLASSIFIER
________________________________________________________________________________
Training: 
AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1,
                   n_estimators=200, random_state=0)
train time: 30.780s
test time:  1.898s
accuracy:   0.440

================================================================================
Classifier.BERNOULLI_NB
________________________________________________________________________________
Training: 
BernoulliNB(alpha=0.1, binarize=0.1, class_prior=None, fit_prior=False)
train time: 0.129s
test time:  0.123s
accuracy:   0.626
dimensionality: 101321
density: 1.000000


================================================================================
Classifier.COMPLEMENT_NB
________________________________________________________________________________
Training: 
ComplementNB(alpha=0.5, class_prior=None, fit_prior=False, norm=False)
train time: 0.133s
test time:  0.020s
accuracy:   0.712
dimensionality: 101321
density: 1.000000


================================================================================
Classifier.DECISION_TREE_CLASSIFIER
________________________________________________________________________________
Training: 
DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='entropy',
                       max_depth=19, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=110,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=0, splitter='best')
train time: 12.032s
test time:  0.004s
accuracy:   0.277

================================================================================
Classifier.EXTRA_TREE_CLASSIFIER
________________________________________________________________________________
Training: 
ExtraTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                    max_depth=None, max_features='auto', max_leaf_nodes=None,
                    min_impurity_decrease=0.0, min_impurity_split=None,
                    min_samples_leaf=1, min_samples_split=2,
                    min_weight_fraction_leaf=0.0, random_state=0,
                    splitter='random')
train time: 0.948s
test time:  0.013s
accuracy:   0.320

================================================================================
Classifier.EXTRA_TREES_CLASSIFIER
________________________________________________________________________________
Training: 
ExtraTreesClassifier(bootstrap=False, ccp_alpha=0.0, class_weight=None,
                     criterion='gini', max_depth=None, max_features='auto',
                     max_leaf_nodes=None, max_samples=None,
                     min_impurity_decrease=0.0, min_impurity_split=None,
                     min_samples_leaf=1, min_samples_split=2,
                     min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=-1,
                     oob_score=False, random_state=0, verbose=False,
                     warm_start=False)
train time: 48.788s
test time:  0.612s
accuracy:   0.647

================================================================================
Classifier.GRADIENT_BOOSTING_CLASSIFIER
________________________________________________________________________________
Training: 
GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.1, loss='deviance', max_depth=3,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_iter_no_change=None, presort='deprecated',
                           random_state=0, subsample=1.0, tol=0.0001,
                           validation_fraction=0.1, verbose=False,
                           warm_start=False)
train time: 588.413s
test time:  0.310s
accuracy:   0.590

================================================================================
Classifier.K_NEIGHBORS_CLASSIFIER
________________________________________________________________________________
Training: 
KNeighborsClassifier(algorithm='auto', leaf_size=5, metric='euclidean',
                     metric_params=None, n_jobs=-1, n_neighbors=3, p=2,
                     weights='distance')
train time: 0.004s
test time:  2.828s
accuracy:   0.085

================================================================================
Classifier.LINEAR_SVC
________________________________________________________________________________
Training: 
LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
          intercept_scaling=1, loss='squared_hinge', max_iter=100,
          multi_class='ovr', penalty='l2', random_state=0, tol=0.1,
          verbose=False)
train time: 1.070s
test time:  0.017s
accuracy:   0.698
dimensionality: 101321
density: 0.639402


================================================================================
Classifier.LOGISTIC_REGRESSION
________________________________________________________________________________
Training: 
LogisticRegression(C=10.01, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='ovr', n_jobs=-1, penalty='l2', random_state=0,
                   solver='lbfgs', tol=0.01, verbose=False, warm_start=False)
train time: 16.747s
test time:  0.023s
accuracy:   0.695
dimensionality: 101321
density: 1.000000


================================================================================
Classifier.LOGISTIC_REGRESSION_CV
________________________________________________________________________________
Training: 
LogisticRegressionCV(Cs=10, class_weight=None, cv=None, dual=False,
                     fit_intercept=True, intercept_scaling=1.0, l1_ratios=None,
                     max_iter=100, multi_class='auto', n_jobs=-1, penalty='l2',
                     random_state=0, refit=True, scoring=None, solver='lbfgs',
                     tol=0.0001, verbose=False)
/home/ramon/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
/home/ramon/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
/home/ramon/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
/home/ramon/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
/home/ramon/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
/home/ramon/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
train time: 858.608s
test time:  0.020s
accuracy:   0.689
dimensionality: 101321
density: 1.000000


================================================================================
Classifier.MLP_CLASSIFIER
________________________________________________________________________________
Training: 
MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
              beta_2=0.999, early_stopping=False, epsilon=1e-08,
              hidden_layer_sizes=(100,), learning_rate='constant',
              learning_rate_init=0.001, max_fun=15000, max_iter=200,
              momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
              power_t=0.5, random_state=0, shuffle=True, solver='adam',
              tol=0.0001, validation_fraction=0.1, verbose=False,
              warm_start=False)
train time: 2059.732s
test time:  0.076s
accuracy:   0.699

================================================================================
Classifier.MULTINOMIAL_NB
________________________________________________________________________________
Training: 
MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True)
train time: 0.147s
test time:  0.020s
accuracy:   0.688
dimensionality: 101321
density: 1.000000


================================================================================
Classifier.NEAREST_CENTROID
________________________________________________________________________________
Training: 
NearestCentroid(metric='cosine', shrink_threshold=None)
/home/ramon/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ramon/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ramon/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ramon/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ramon/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ramon/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ramon/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ramon/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ramon/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ramon/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ramon/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ramon/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ramon/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ramon/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ramon/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ramon/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ramon/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ramon/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ramon/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ramon/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
train time: 0.050s
test time:  0.037s
accuracy:   0.667

================================================================================
Classifier.NU_SVC
________________________________________________________________________________
Training: 
NuSVC(break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
      max_iter=-1, nu=0.5, probability=False, random_state=0, shrinking=True,
      tol=0.001, verbose=False)
train time: 101.402s
test time:  34.208s
accuracy:   0.689

================================================================================
Classifier.PASSIVE_AGGRESSIVE_CLASSIFIER
________________________________________________________________________________
Training: 
PassiveAggressiveClassifier(C=0.01, average=False, class_weight='balanced',
                            early_stopping=False, fit_intercept=True,
                            loss='hinge', max_iter=100, n_iter_no_change=5,
                            n_jobs=-1, random_state=0, shuffle=True, tol=0.0001,
                            validation_fraction=0.1, verbose=False,
                            warm_start=False)
train time: 4.931s
/home/ramon/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/linear_model/_stochastic_gradient.py:557: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
  ConvergenceWarning)
test time:  0.027s
accuracy:   0.704
dimensionality: 101321
density: 0.843517


================================================================================
Classifier.PERCEPTRON
________________________________________________________________________________
Training: 
Perceptron(alpha=0.0001, class_weight='balanced', early_stopping=True, eta0=1.0,
           fit_intercept=True, max_iter=100, n_iter_no_change=15, n_jobs=-1,
           penalty='l2', random_state=0, shuffle=True, tol=0.1,
           validation_fraction=0.01, verbose=False, warm_start=False)
train time: 1.918s
test time:  0.027s
accuracy:   0.532
dimensionality: 101321
density: 0.206251


================================================================================
Classifier.RANDOM_FOREST_CLASSIFIER
________________________________________________________________________________
Training: 
RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=-1, oob_score=False, random_state=0,
                       verbose=False, warm_start=False)
train time: 29.417s
test time:  0.514s
accuracy:   0.622

================================================================================
Classifier.RIDGE_CLASSIFIER
________________________________________________________________________________
Training: 
RidgeClassifier(alpha=1.0, class_weight=None, copy_X=True, fit_intercept=True,
                max_iter=None, normalize=False, random_state=0, solver='auto',
                tol=0.001)
train time: 3.325s
test time:  0.034s
accuracy:   0.707
dimensionality: 101321
density: 1.000000


================================================================================
Classifier.RIDGE_CLASSIFIERCV
________________________________________________________________________________
Training: 
RidgeClassifierCV(alphas=array([ 0.1,  1. , 10. ]), class_weight=None, cv=None,
                  fit_intercept=True, normalize=False, scoring=None,
                  store_cv_values=False)
train time: 337.532s
test time:  0.029s
accuracy:   0.707
dimensionality: 101321
density: 1.000000


================================================================================
Classifier.SGD_CLASSIFIER
________________________________________________________________________________
Training: 
SGDClassifier(alpha=0.0001, average=False, class_weight=None,
              early_stopping=False, epsilon=0.1, eta0=0.0, fit_intercept=True,
              l1_ratio=0.15, learning_rate='optimal', loss='hinge',
              max_iter=1000, n_iter_no_change=5, n_jobs=-1, penalty='l2',
              power_t=0.5, random_state=0, shuffle=True, tol=0.001,
              validation_fraction=0.1, verbose=False, warm_start=False)
train time: 0.817s
test time:  0.026s
accuracy:   0.702
dimensionality: 101321
density: 0.328711


FINAL CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS dataset (removing headers signatures and quoting)
| ID | ML Algorithm | Accuracy Score (%) | Training time (seconds) | Test time (seconds) |
| --- | ------------- | ------------------ | ----------------------- | ------------------- |
|  1  |  ADA_BOOST_CLASSIFIER  |  44.04%  |  30.78  |  1.898  |
|  2  |  BERNOULLI_NB  |  62.56%  |  0.1286  |  0.1225  |
|  3  |  COMPLEMENT_NB  |  71.22%  |  0.1325  |  0.01989  |
|  4  |  DECISION_TREE_CLASSIFIER  |  27.72%  |  12.03  |  0.004409  |
|  5  |  EXTRA_TREE_CLASSIFIER  |  32.01%  |  0.9475  |  0.01314  |
|  6  |  EXTRA_TREES_CLASSIFIER  |  64.66%  |  48.79  |  0.6123  |
|  7  |  GRADIENT_BOOSTING_CLASSIFIER  |  59.04%  |  588.4  |  0.31  |
|  8  |  K_NEIGHBORS_CLASSIFIER  |  8.48%  |  0.003802  |  2.828  |
|  9  |  LINEAR_SVC  |  69.77%  |  1.07  |  0.01691  |
|  10  |  LOGISTIC_REGRESSION  |  69.54%  |  16.75  |  0.0232  |
|  11  |  LOGISTIC_REGRESSION_CV  |  68.95%  |  858.6  |  0.0197  |
|  12  |  MLP_CLASSIFIER  |  69.92%  |  2.06e+03  |  0.07586  |
|  13  |  MULTINOMIAL_NB  |  68.79%  |  0.1469  |  0.02044  |
|  14  |  NEAREST_CENTROID  |  66.70%  |  0.05023  |  0.03666  |
|  15  |  NU_SVC  |  68.95%  |  101.4  |  34.21  |
|  16  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  70.42%  |  4.931  |  0.02671  |
|  17  |  PERCEPTRON  |  53.23%  |  1.918  |  0.02685  |
|  18  |  RANDOM_FOREST_CLASSIFIER  |  62.21%  |  29.42  |  0.5142  |
|  19  |  RIDGE_CLASSIFIER  |  70.67%  |  3.325  |  0.03407  |
|  20  |  RIDGE_CLASSIFIERCV  |  70.67%  |  337.5  |  0.02896  |
|  21  |  SGD_CLASSIFIER  |  70.22%  |  0.8174  |  0.02582  |


Best algorithm:
===> 3) COMPLEMENT_NB
		Accuracy score = 71.22%		Training time = 0.1325		Test time = 0.01989

Loading IMDB_REVIEWS dataset:

===> Reading files from /home/ramon/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/train/neg

===> Reading files from /home/ramon/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/train/pos

===> Reading files from /home/ramon/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/test/neg

===> Reading files from /home/ramon/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/test/pos
data loaded
25000 documents - 33.133MB (training set)
25000 documents - 32.351MB (test set)

Extracting features from the training data using a vectorizer
done in 4.623612s at 7.166MB/s
n_samples: 25000, n_features: 74170

Extracting features from the test data using the same vectorizer
done in 4.584074s at 7.057MB/s
n_samples: 25000, n_features: 74170

================================================================================
Classifier.ADA_BOOST_CLASSIFIER
________________________________________________________________________________
Training: 
AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
                   n_estimators=50, random_state=0)
train time: 16.345s
test time:  0.931s
accuracy:   0.802

================================================================================
Classifier.BERNOULLI_NB
________________________________________________________________________________
Training: 
BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
train time: 0.039s
test time:  0.035s
accuracy:   0.815
dimensionality: 74170
density: 1.000000


================================================================================
Classifier.COMPLEMENT_NB
________________________________________________________________________________
Training: 
ComplementNB(alpha=1.0, class_prior=None, fit_prior=True, norm=False)
train time: 0.027s
test time:  0.015s
accuracy:   0.839
dimensionality: 74170
density: 1.000000


================================================================================
Classifier.EXTRA_TREE_CLASSIFIER
________________________________________________________________________________
Training: 
ExtraTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                    max_depth=None, max_features='auto', max_leaf_nodes=None,
                    min_impurity_decrease=0.0, min_impurity_split=None,
                    min_samples_leaf=1, min_samples_split=2,
                    min_weight_fraction_leaf=0.0, random_state=0,
                    splitter='random')
train time: 1.002s
test time:  0.028s
accuracy:   0.636

================================================================================
Classifier.EXTRA_TREES_CLASSIFIER
________________________________________________________________________________
Training: 
ExtraTreesClassifier(bootstrap=False, ccp_alpha=0.0, class_weight=None,
                     criterion='gini', max_depth=None, max_features='auto',
                     max_leaf_nodes=None, max_samples=None,
                     min_impurity_decrease=0.0, min_impurity_split=None,
                     min_samples_leaf=1, min_samples_split=2,
                     min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=-1,
                     oob_score=False, random_state=0, verbose=False,
                     warm_start=False)
train time: 40.666s
test time:  1.320s
accuracy:   0.856

================================================================================
Classifier.GRADIENT_BOOSTING_CLASSIFIER
________________________________________________________________________________
Training: 
GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.1, loss='deviance', max_depth=3,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_iter_no_change=None, presort='deprecated',
                           random_state=0, subsample=1.0, tol=0.0001,
                           validation_fraction=0.1, verbose=False,
                           warm_start=False)
train time: 74.790s
test time:  0.068s
accuracy:   0.807

================================================================================
Classifier.K_NEIGHBORS_CLASSIFIER
________________________________________________________________________________
Training: 
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=-1, n_neighbors=5, p=2,
                     weights='uniform')
train time: 0.009s
test time:  29.239s
accuracy:   0.733

================================================================================
Classifier.LINEAR_SVC
________________________________________________________________________________
Training: 
LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
          intercept_scaling=1, loss='squared_hinge', max_iter=1000,
          multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
          verbose=False)
train time: 0.438s
test time:  0.006s
accuracy:   0.871
dimensionality: 74170
density: 0.872388


================================================================================
Classifier.LOGISTIC_REGRESSION
________________________________________________________________________________
Training: 
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=-1, penalty='l2', random_state=0,
                   solver='lbfgs', tol=0.0001, verbose=False, warm_start=False)
train time: 1.811s
test time:  0.012s
accuracy:   0.884
dimensionality: 74170
density: 1.000000


================================================================================
Classifier.LOGISTIC_REGRESSION_CV
________________________________________________________________________________
Training: 
LogisticRegressionCV(Cs=10, class_weight=None, cv=None, dual=False,
                     fit_intercept=True, intercept_scaling=1.0, l1_ratios=None,
                     max_iter=100, multi_class='auto', n_jobs=-1, penalty='l2',
                     random_state=0, refit=True, scoring=None, solver='lbfgs',
                     tol=0.0001, verbose=False)
/home/ramon/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
/home/ramon/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
/home/ramon/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
/home/ramon/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
/home/ramon/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
/home/ramon/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
/home/ramon/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
/home/ramon/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
/home/ramon/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
/home/ramon/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
/home/ramon/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
/home/ramon/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
/home/ramon/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
/home/ramon/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
train time: 39.293s
test time:  0.012s
accuracy:   0.884
dimensionality: 74170
density: 1.000000


================================================================================
Classifier.MLP_CLASSIFIER
________________________________________________________________________________
Training: 
MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
              beta_2=0.999, early_stopping=False, epsilon=1e-08,
              hidden_layer_sizes=(100,), learning_rate='constant',
              learning_rate_init=0.001, max_fun=15000, max_iter=200,
              momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
              power_t=0.5, random_state=0, shuffle=True, solver='adam',
              tol=0.0001, validation_fraction=0.1, verbose=False,
              warm_start=False)
train time: 1362.259s
test time:  0.315s
accuracy:   0.857

================================================================================
Classifier.MULTINOMIAL_NB
________________________________________________________________________________
Training: 
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
train time: 0.050s
test time:  0.029s
accuracy:   0.839
dimensionality: 74170
density: 1.000000


================================================================================
Classifier.NEAREST_CENTROID
________________________________________________________________________________
Training: 
NearestCentroid(metric='euclidean', shrink_threshold=None)
train time: 0.031s
test time:  0.019s
accuracy:   0.837

================================================================================
Classifier.NU_SVC
________________________________________________________________________________
Training: 
NuSVC(break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
      max_iter=-1, nu=0.5, probability=False, random_state=0, shrinking=True,
      tol=0.001, verbose=False)
train time: 1052.422s
test time:  281.504s
accuracy:   0.884

================================================================================
Classifier.PASSIVE_AGGRESSIVE_CLASSIFIER
________________________________________________________________________________
Training: 
PassiveAggressiveClassifier(C=1.0, average=False, class_weight=None,
                            early_stopping=False, fit_intercept=True,
                            loss='hinge', max_iter=1000, n_iter_no_change=5,
                            n_jobs=-1, random_state=0, shuffle=True, tol=0.001,
                            validation_fraction=0.1, verbose=False,
                            warm_start=False)
train time: 0.275s
test time:  0.007s
accuracy:   0.852
dimensionality: 74170
density: 0.835324


================================================================================
Classifier.PERCEPTRON
________________________________________________________________________________
Training: 
Perceptron(alpha=0.0001, class_weight=None, early_stopping=False, eta0=1.0,
           fit_intercept=True, max_iter=1000, n_iter_no_change=5, n_jobs=-1,
           penalty=None, random_state=0, shuffle=True, tol=0.001,
           validation_fraction=0.1, verbose=False, warm_start=False)
train time: 0.142s
test time:  0.007s
accuracy:   0.846
dimensionality: 74170
density: 0.627949


================================================================================
Classifier.RANDOM_FOREST_CLASSIFIER
________________________________________________________________________________
Training: 
RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=-1, oob_score=False, random_state=0,
                       verbose=False, warm_start=False)
train time: 24.844s
test time:  1.114s
accuracy:   0.848

================================================================================
Classifier.RIDGE_CLASSIFIER
________________________________________________________________________________
Training: 
RidgeClassifier(alpha=1.0, class_weight=None, copy_X=True, fit_intercept=True,
                max_iter=None, normalize=False, random_state=0, solver='auto',
                tol=0.001)
train time: 0.597s
test time:  0.012s
accuracy:   0.869
dimensionality: 74170
density: 1.000000


================================================================================
Classifier.RIDGE_CLASSIFIERCV
________________________________________________________________________________
Training: 
RidgeClassifierCV(alphas=array([ 0.1,  1. , 10. ]), class_weight=None, cv=None,
                  fit_intercept=True, normalize=False, scoring=None,
                  store_cv_values=False)
03/06/2020 01:49:12 AM - ERROR - Unable to allocate 4.49 GiB for an array with shape (602485574,) and data type float64
FINAL CLASSIFICATION TABLE: IMDB_REVIEWS dataset (Binary classification)
| ID | ML Algorithm | Accuracy Score (%) | Training time (seconds) | Test time (seconds) |
| --- | ------------- | ------------------ | ----------------------- | ------------------- |
|  1  |  ADA_BOOST_CLASSIFIER  |  80.24%  |  16.35  |  0.9315  |
|  2  |  BERNOULLI_NB  |  81.52%  |  0.03934  |  0.03539  |
|  3  |  COMPLEMENT_NB  |  83.93%  |  0.02731  |  0.0146  |
|  4  |  EXTRA_TREE_CLASSIFIER  |  63.64%  |  1.002  |  0.02799  |
|  5  |  EXTRA_TREES_CLASSIFIER  |  85.62%  |  40.67  |  1.32  |
|  6  |  GRADIENT_BOOSTING_CLASSIFIER  |  80.72%  |  74.79  |  0.06831  |
|  7  |  K_NEIGHBORS_CLASSIFIER  |  73.29%  |  0.009489  |  29.24  |
|  8  |  LINEAR_SVC  |  87.13%  |  0.4377  |  0.006407  |
|  9  |  LOGISTIC_REGRESSION  |  88.39%  |  1.811  |  0.01204  |
|  10  |  LOGISTIC_REGRESSION_CV  |  88.39%  |  39.29  |  0.01243  |
|  11  |  MLP_CLASSIFIER  |  85.75%  |  1.362e+03  |  0.3155  |
|  12  |  MULTINOMIAL_NB  |  83.93%  |  0.04968  |  0.0289  |
|  13  |  NEAREST_CENTROID  |  83.72%  |  0.03134  |  0.01913  |
|  14  |  NU_SVC  |  88.39%  |  1.052e+03  |  281.5  |
|  15  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  85.17%  |  0.2745  |  0.006755  |
|  16  |  PERCEPTRON  |  84.59%  |  0.1425  |  0.006603  |
|  17  |  RANDOM_FOREST_CLASSIFIER  |  84.80%  |  24.84  |  1.114  |
|  18  |  RIDGE_CLASSIFIER  |  86.90%  |  0.5975  |  0.01191  |


Best algorithm:
===> 10) LOGISTIC_REGRESSION_CV
		Accuracy score = 88.39%		Training time = 39.29		Test time = 0.01243



DONE!
Program finished. It took 7097.163280963898 seconds

Process finished with exit code 0
```