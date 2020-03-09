#### FINAL CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS dataset (removing headers signatures and quoting)

| ID | ML Algorithm | Accuracy Score (%) | Training time (seconds) | Test time (seconds) |
| --- | ------------- | ------------------ | ----------------------- | ------------------- |
|  1  |  STACKING_CLASSIFIER  |  71.26%  |  195.3  |  0.3803  |


#### FINAL CLASSIFICATION TABLE: IMDB_REVIEWS dataset (Binary classification)

| ID | ML Algorithm | Accuracy Score (%) | Training time (seconds) | Test time (seconds) |
| --- | ------------- | ------------------ | ----------------------- | ------------------- |
|  1  |  STACKING_CLASSIFIER  |  88.29%  |  93.63  |  0.6581  |


#### All logs: TWENTY_NEWS_GROUPS dataset and IMDB_REVIEWS dataset (Binary classification)

```
python /comp551-2020-p2_classification_of_textual_data/code/main.py -ml STACKING_CLASSIFIER
Using TensorFlow backend.
2020-03-09 00:48:41.052989: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libnvinfer.so.6'; dlerror: libnvinfer.so.6: cannot open shared object file: No such file or directory
2020-03-09 00:48:41.053046: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libnvinfer_plugin.so.6'; dlerror: libnvinfer_plugin.so.6: cannot open shared object file: No such file or directory
2020-03-09 00:48:41.053051: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:30] Cannot dlopen some TensorRT libraries. If you would like to use Nvidia GPU with TensorRT, please make sure the missing libraries mentioned above are installed properly.
03/09/2020 12:48:41 AM - INFO - Program started...
usage: main.py [-h] [-d DATASET] [-ml ML_ALGORITHM_LIST]
               [-use_default_parameters] [-not_shuffle] [-n_jobs N_JOBS] [-cv]
               [-n_splits N_SPLITS] [-required_classifiers]
               [-news_with_4_classes] [-news_no_filter] [-imdb_multi_class]
               [-show_reviews] [-r] [-m] [--chi2_select CHI2_SELECT] [-cm]
               [-use_hashing] [-use_count] [-n_features N_FEATURES]
               [-plot_time] [-save_logs] [-verbose]
               [-random_state RANDOM_STATE] [-dl] [-v]

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
                        GRADIENT_BOOSTING_CLASSIFIER, 6)
                        K_NEIGHBORS_CLASSIFIER, 7) LINEAR_SVC, 8)
                        LOGISTIC_REGRESSION, 9) MULTINOMIAL_NB, 10)
                        NEAREST_CENTROID, 11) PASSIVE_AGGRESSIVE_CLASSIFIER,
                        12) PERCEPTRON, 13) RANDOM_FOREST_CLASSIFIER, 14)
                        RIDGE_CLASSIFIER). Default: None. If ml_algorithm_list
                        is not provided, all ML algorithms will be executed.
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
  -dl, --run_deep_learning_using_keras
                        Run deep learning using keras. Default: False (Run
                        scikit-learn algorithms)
  -v, --version         show program's version number and exit

COMP 551 (001/002), Applied Machine Learning, Winter 2020, McGill University.
None
==================================================================================================================================

Running with options: 
	Dataset = ALL
	ML algorithm list (If ml_algorithm_list is not provided, all ML algorithms will be executed) = ['STACKING_CLASSIFIER']
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
	Run deep learning using keras. Default: False (Run scikit-learn algorithms) = False
==================================================================================================================================

Loading TWENTY_NEWS_GROUPS dataset for categories:
data loaded
11314 documents - 13.782MB (training set)
7532 documents - 8.262MB (test set)
20 categories

Extracting features from the training data using a vectorizer
done in 1.053744s at 13.079MB/s
n_samples: 11314, n_features: 101321

Extracting features from the test data using the same vectorizer
done in 0.569355s at 14.510MB/s
n_samples: 7532, n_features: 101321

	==> Using JSON with best parameters (selected using grid search) to the COMPLEMENT_NB classifier (multi-class classification) and TWENTY_NEWS_GROUPS dataset ===> JSON in dictionary format: {'alpha': 0.5, 'fit_prior': False, 'norm': False}
	 ComplementNB(alpha=0.5, class_prior=None, fit_prior=False, norm=False)
	==> Using JSON with best parameters (selected using grid search) to the LINEAR_SVC classifier (multi-class classification) and TWENTY_NEWS_GROUPS dataset ===> JSON in dictionary format: {'C': 1.0, 'multi_class': 'ovr', 'tol': 0.0001}
	 LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
          intercept_scaling=1, loss='squared_hinge', max_iter=1000,
          multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
          verbose=False)
	==> Using JSON with best parameters (selected using grid search) to the LOGISTIC_REGRESSION classifier (multi-class classification) and TWENTY_NEWS_GROUPS dataset ===> JSON in dictionary format: {'C': 10, 'tol': 0.001}
	 LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=-1, penalty='l2', random_state=0,
                   solver='lbfgs', tol=0.001, verbose=False, warm_start=False)
	==> Using JSON with best parameters (selected using grid search) to the PASSIVE_AGGRESSIVE_CLASSIFIER classifier (multi-class classification) and TWENTY_NEWS_GROUPS dataset ===> JSON in dictionary format: {'C': 0.01, 'early_stopping': False, 'tol': 0.0001, 'validation_fraction': 0.0001}
	 PassiveAggressiveClassifier(C=0.01, average=False, class_weight=None,
                            early_stopping=False, fit_intercept=True,
                            loss='hinge', max_iter=1000, n_iter_no_change=5,
                            n_jobs=-1, random_state=0, shuffle=True, tol=0.0001,
                            validation_fraction=0.0001, verbose=False,
                            warm_start=False)
	==> Using JSON with best parameters (selected using grid search) to the RANDOM_FOREST_CLASSIFIER classifier (multi-class classification) and TWENTY_NEWS_GROUPS dataset ===> JSON in dictionary format: {'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 200}
	 RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=5,
                       min_weight_fraction_leaf=0.0, n_estimators=200,
                       n_jobs=-1, oob_score=False, random_state=0,
                       verbose=False, warm_start=False)
	==> Using JSON with best parameters (selected using grid search) to the RIDGE_CLASSIFIER classifier (multi-class classification) and TWENTY_NEWS_GROUPS dataset ===> JSON in dictionary format: {'alpha': 0.5, 'tol': 0.001}
	 RidgeClassifier(alpha=0.5, class_weight=None, copy_X=True, fit_intercept=True,
                max_iter=None, normalize=False, random_state=0, solver='auto',
                tol=0.001)
	 StackingClassifier(cv=None,
                   estimators=[('complement_nb',
                                ComplementNB(alpha=0.5, class_prior=None,
                                             fit_prior=False, norm=False)),
                               ('linear_svc',
                                LinearSVC(C=1.0, class_weight=None, dual=True,
                                          fit_intercept=True,
                                          intercept_scaling=1,
                                          loss='squared_hinge', max_iter=1000,
                                          multi_class='ovr', penalty='l2',
                                          random_state=0, tol=0.0001,
                                          verbose=False)),
                               ('logistic_regressio...
                                                copy_X=True, fit_intercept=True,
                                                max_iter=None, normalize=False,
                                                random_state=0, solver='auto',
                                                tol=0.001))],
                   final_estimator=LinearSVC(C=1.0, class_weight=None,
                                             dual=True, fit_intercept=True,
                                             intercept_scaling=1,
                                             loss='squared_hinge',
                                             max_iter=1000, multi_class='ovr',
                                             penalty='l2', random_state=0,
                                             tol=0.0001, verbose=False),
                   n_jobs=-1, passthrough=False, stack_method='auto',
                   verbose=False)
================================================================================
Classifier.STACKING_CLASSIFIER
________________________________________________________________________________
Training: 
StackingClassifier(cv=None,
                   estimators=[('complement_nb',
                                ComplementNB(alpha=0.5, class_prior=None,
                                             fit_prior=False, norm=False)),
                               ('linear_svc',
                                LinearSVC(C=1.0, class_weight=None, dual=True,
                                          fit_intercept=True,
                                          intercept_scaling=1,
                                          loss='squared_hinge', max_iter=1000,
                                          multi_class='ovr', penalty='l2',
                                          random_state=0, tol=0.0001,
                                          verbose=False)),
                               ('logistic_regressio...
                                                copy_X=True, fit_intercept=True,
                                                max_iter=None, normalize=False,
                                                random_state=0, solver='auto',
                                                tol=0.001))],
                   final_estimator=LinearSVC(C=1.0, class_weight=None,
                                             dual=True, fit_intercept=True,
                                             intercept_scaling=1,
                                             loss='squared_hinge',
                                             max_iter=1000, multi_class='ovr',
                                             penalty='l2', random_state=0,
                                             tol=0.0001, verbose=False),
                   n_jobs=-1, passthrough=False, stack_method='auto',
                   verbose=False)
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
train time: 195.318s
test time:  0.380s
accuracy:   0.713

FINAL CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS dataset (removing headers signatures and quoting)
| ID | ML Algorithm | Accuracy Score (%) | Training time (seconds) | Test time (seconds) |
| --- | ------------- | ------------------ | ----------------------- | ------------------- |
|  1  |  STACKING_CLASSIFIER  |  71.26%  |  195.3  |  0.3803  |


Best algorithm:
===> 1) STACKING_CLASSIFIER
		Accuracy score = 71.26%		Training time = 195.3		Test time = 0.3803

Loading IMDB_REVIEWS dataset:

===> Reading files from /home/rpessoa/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/train/neg

===> Reading files from /home/rpessoa/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/train/pos

===> Reading files from /home/rpessoa/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/test/neg

===> Reading files from /home/rpessoa/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/test/pos
data loaded
25000 documents - 33.133MB (training set)
25000 documents - 32.351MB (test set)

Extracting features from the training data using a vectorizer
done in 2.796573s at 11.848MB/s
n_samples: 25000, n_features: 74170

Extracting features from the test data using the same vectorizer
done in 2.727815s at 11.860MB/s
n_samples: 25000, n_features: 74170

	==> Using JSON with best parameters (selected using grid search) to the COMPLEMENT_NB classifier (binary classification) and IMDB_REVIEWS dataset ===> JSON in dictionary format: {'alpha': 1.0, 'fit_prior': False, 'norm': False}
	 ComplementNB(alpha=1.0, class_prior=None, fit_prior=False, norm=False)
	==> Using JSON with best parameters (selected using grid search) to the LINEAR_SVC classifier (binary classification) and IMDB_REVIEWS dataset ===> JSON in dictionary format: {'C': 1.0, 'multi_class': 'ovr', 'tol': 0.0001}
	 LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
          intercept_scaling=1, loss='squared_hinge', max_iter=1000,
          multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
          verbose=False)
	==> Using JSON with best parameters (selected using grid search) to the LOGISTIC_REGRESSION classifier (binary classification) and IMDB_REVIEWS dataset ===> JSON in dictionary format: {'C': 10, 'tol': 0.01}
	 LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=-1, penalty='l2', random_state=0,
                   solver='lbfgs', tol=0.01, verbose=False, warm_start=False)
	==> Using JSON with best parameters (selected using grid search) to the PASSIVE_AGGRESSIVE_CLASSIFIER classifier (binary classification) and IMDB_REVIEWS dataset ===> JSON in dictionary format: {'C': 0.01, 'early_stopping': False, 'tol': 0.001, 'validation_fraction': 0.01}
	 PassiveAggressiveClassifier(C=0.01, average=False, class_weight=None,
                            early_stopping=False, fit_intercept=True,
                            loss='hinge', max_iter=1000, n_iter_no_change=5,
                            n_jobs=-1, random_state=0, shuffle=True, tol=0.001,
                            validation_fraction=0.01, verbose=False,
                            warm_start=False)
	==> Using JSON with best parameters (selected using grid search) to the RANDOM_FOREST_CLASSIFIER classifier (binary classification) and IMDB_REVIEWS dataset ===> JSON in dictionary format: {'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 200}
	 RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=5,
                       min_weight_fraction_leaf=0.0, n_estimators=200,
                       n_jobs=-1, oob_score=False, random_state=0,
                       verbose=False, warm_start=False)
	==> Using JSON with best parameters (selected using grid search) to the RIDGE_CLASSIFIER classifier (binary classification) and IMDB_REVIEWS dataset ===> JSON in dictionary format: {'alpha': 1.0, 'tol': 0.0001}
	 RidgeClassifier(alpha=1.0, class_weight=None, copy_X=True, fit_intercept=True,
                max_iter=None, normalize=False, random_state=0, solver='auto',
                tol=0.0001)
	 StackingClassifier(cv=None,
                   estimators=[('complement_nb',
                                ComplementNB(alpha=1.0, class_prior=None,
                                             fit_prior=False, norm=False)),
                               ('linear_svc',
                                LinearSVC(C=1.0, class_weight=None, dual=True,
                                          fit_intercept=True,
                                          intercept_scaling=1,
                                          loss='squared_hinge', max_iter=1000,
                                          multi_class='ovr', penalty='l2',
                                          random_state=0, tol=0.0001,
                                          verbose=False)),
                               ('logistic_regressio...
                                                copy_X=True, fit_intercept=True,
                                                max_iter=None, normalize=False,
                                                random_state=0, solver='auto',
                                                tol=0.0001))],
                   final_estimator=LinearSVC(C=1.0, class_weight=None,
                                             dual=True, fit_intercept=True,
                                             intercept_scaling=1,
                                             loss='squared_hinge',
                                             max_iter=1000, multi_class='ovr',
                                             penalty='l2', random_state=0,
                                             tol=0.0001, verbose=False),
                   n_jobs=-1, passthrough=False, stack_method='auto',
                   verbose=False)
================================================================================
Classifier.STACKING_CLASSIFIER
________________________________________________________________________________
Training: 
StackingClassifier(cv=None,
                   estimators=[('complement_nb',
                                ComplementNB(alpha=1.0, class_prior=None,
                                             fit_prior=False, norm=False)),
                               ('linear_svc',
                                LinearSVC(C=1.0, class_weight=None, dual=True,
                                          fit_intercept=True,
                                          intercept_scaling=1,
                                          loss='squared_hinge', max_iter=1000,
                                          multi_class='ovr', penalty='l2',
                                          random_state=0, tol=0.0001,
                                          verbose=False)),
                               ('logistic_regressio...
                                                copy_X=True, fit_intercept=True,
                                                max_iter=None, normalize=False,
                                                random_state=0, solver='auto',
                                                tol=0.0001))],
                   final_estimator=LinearSVC(C=1.0, class_weight=None,
                                             dual=True, fit_intercept=True,
                                             intercept_scaling=1,
                                             loss='squared_hinge',
                                             max_iter=1000, multi_class='ovr',
                                             penalty='l2', random_state=0,
                                             tol=0.0001, verbose=False),
                   n_jobs=-1, passthrough=False, stack_method='auto',
                   verbose=False)
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
train time: 93.631s
test time:  0.658s
accuracy:   0.883

FINAL CLASSIFICATION TABLE: IMDB_REVIEWS dataset (Binary classification)
| ID | ML Algorithm | Accuracy Score (%) | Training time (seconds) | Test time (seconds) |
| --- | ------------- | ------------------ | ----------------------- | ------------------- |
|  1  |  STACKING_CLASSIFIER  |  88.29%  |  93.63  |  0.6581  |


Best algorithm:
===> 1) STACKING_CLASSIFIER
		Accuracy score = 88.29%		Training time = 93.63		Test time = 0.6581



DONE!
Program finished. It took 300.26136088371277 seconds

Process finished with exit code 0
```