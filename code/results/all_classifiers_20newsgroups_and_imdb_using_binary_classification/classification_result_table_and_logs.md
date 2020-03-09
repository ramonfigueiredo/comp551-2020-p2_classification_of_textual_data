# Running all classifiers (TWENTY_NEWS_GROUPS dataset and IMDB_REVIEWS dataset using binary classification)

### 20 News Groups dataset (removing headers signatures and quoting)

#### FINAL CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS dataset (removing headers signatures and quoting)

| ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
| --- | ------------- | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
|  1  |  ADA_BOOST_CLASSIFIER  |  44.04%  |  [0.46973045 0.45779938 0.46575342 0.47017234 0.46993811]  |  0.47 (+/- 0.01)  |  19.26  |  1.015  |
|  2  |  BERNOULLI_NB  |  62.56%  |  [0.69200177 0.67741935 0.6752099  0.67388422 0.67992927]  |  0.68 (+/- 0.01)  |  0.07593  |  0.0528  |
|  3  |  COMPLEMENT_NB  |  71.22%  |  [0.776403   0.75828546 0.78126381 0.77507733 0.76967286]  |  0.77 (+/- 0.02)  |  0.0646  |  0.01039  |
|  4  |  DECISION_TREE_CLASSIFIER  |  44.72%  |  [0.49094123 0.48696421 0.47547503 0.49270879 0.49646331]  |  0.49 (+/- 0.01)  |  9.037  |  0.006431  |
|  5  |  GRADIENT_BOOSTING_CLASSIFIER  |  59.43%  |  [0.65930181 0.62704375 0.64781264 0.6548829  0.64721485]  |  0.65 (+/- 0.02)  |  658.5  |  0.3929  |
|  6  |  K_NEIGHBORS_CLASSIFIER  |  8.48%  |  [0.12417145 0.11842687 0.12063632 0.12240389 0.12113174]  |  0.12 (+/- 0.00)  |  0.003191  |  1.298  |
|  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  0.76 (+/- 0.02)  |  0.8115  |  0.008989  |
|  8  |  LOGISTIC_REGRESSION  |  69.28%  |  [0.75961114 0.74016792 0.75519222 0.7565179  0.73607427]  |  0.75 (+/- 0.02)  |  22.88  |  0.01089  |
|  9  |  MULTINOMIAL_NB  |  68.79%  |  [0.76049492 0.74149359 0.75519222 0.75695979 0.74270557]  |  0.75 (+/- 0.02)  |  0.07197  |  0.01174  |
|  10  |  NEAREST_CENTROID  |  66.70%  |  [0.71674768 0.71807335 0.71851525 0.71939903 0.70601238]  |  0.72 (+/- 0.01)  |  0.01669  |  0.01906  |
|  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  69.62%  |  [0.76535572 0.74856385 0.76182059 0.77065842 0.74889478]  |  0.76 (+/- 0.02)  |  2.319  |  0.01587  |
|  12  |  PERCEPTRON  |  53.86%  |  [0.63146266 0.61025188 0.58948299 0.60450729 0.60477454]  |  0.61 (+/- 0.03)  |  0.4178  |  0.0171  |
|  13  |  RANDOM_FOREST_CLASSIFIER  |  63.71%  |  [0.69465312 0.66902342 0.68272205 0.69730446 0.67462423]  |  0.68 (+/- 0.02)  |  7.79  |  0.3067  |
|  14  |  RIDGE_CLASSIFIER  |  70.02%  |  [0.76756518 0.74502872 0.76800707 0.776403   0.76083112]  |  0.76 (+/- 0.02)  |  3.12  |  0.02272  |
|  15  |  MAJORITY_VOTING_CLASSIFIER  |  70.37%  |  [0.76712329 0.75077331 0.76800707 0.77551922 0.7515473 ]  |  0.76 (+/- 0.02)  |  31.06  |  0.4181  |
|  16  |  SOFT_VOTING_CLASSIFIER  |  71.73%  |  [0.79231109 0.75828546 0.79098542 0.7870084  0.76702034]  |  0.78 (+/- 0.03)  |  28.07  |  0.3526  |
|  17  |  STACKING_CLASSIFIER  |  71.28%  |  [0.77507733 0.75961114 0.77198409 0.77110031 0.75066313]  |  0.77 (+/- 0.02)  |  184.0  |  0.368  |

* Accuracy score

![FINAL CLASSIFICATION TABLE (Accuracy score): TWENTY_NEWS_GROUPS dataset (removing headers signatures and quoting)](https://github.com/ramonfigueiredopessoa/comp551-2020-p2_classification_of_textual_data/blob/master/code/results/all_classifiers_20newsgroups_and_imdb_using_binary_classification/TWENTY_NEWS_GROUPS-just_accuracy_score.png)

* Accuracy score, normalized training time and normalized test time

![FINAL CLASSIFICATION TABLE (Accuracy score and normalized training and test time): TWENTY_NEWS_GROUPS dataset (removing headers signatures and quoting)](https://github.com/ramonfigueiredopessoa/comp551-2020-p2_classification_of_textual_data/blob/master/code/results/all_classifiers_20newsgroups_and_imdb_using_binary_classification/TWENTY_NEWS_GROUPS.png)

### IMDB using Binary Classification

#### FINAL CLASSIFICATION TABLE: IMDB_REVIEWS dataset (Binary classification)

| ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
| --- | ------------- | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
|  1  |  ADA_BOOST_CLASSIFIER  |  84.60%  |  [0.8398 0.8516 0.8416 0.8366 0.8416]  |  0.84 (+/- 0.01)  |  103.3  |  5.553  |
|  2  |  BERNOULLI_NB  |  81.28%  |  [0.8398 0.8424 0.8514 0.8396 0.8516]  |  0.84 (+/- 0.01)  |  0.02759  |  0.02151  |
|  3  |  COMPLEMENT_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  0.87 (+/- 0.01)  |  0.01739  |  0.00831  |
|  4  |  DECISION_TREE_CLASSIFIER  |  74.14%  |  [0.735  0.7292 0.746  0.739  0.7342]  |  0.74 (+/- 0.01)  |  7.808  |  0.01236  |
|  5  |  GRADIENT_BOOSTING_CLASSIFIER  |  82.86%  |  [0.8278 0.8294 0.8238 0.823  0.8284]  |  0.83 (+/- 0.01)  |  100.8  |  0.06589  |
|  6  |  K_NEIGHBORS_CLASSIFIER  |  82.66%  |  [0.8632 0.8744 0.8694 0.864  0.8618]  |  0.87 (+/- 0.01)  |  0.006417  |  13.02  |
|  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  0.88 (+/- 0.01)  |  0.2095  |  0.004025  |
|  8  |  LOGISTIC_REGRESSION  |  87.75%  |  [0.8882 0.897  0.8878 0.8876 0.8818]  |  0.89 (+/- 0.01)  |  1.075  |  0.005046  |
|  9  |  MULTINOMIAL_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  0.87 (+/- 0.01)  |  0.01648  |  0.0088  |
|  10  |  NEAREST_CENTROID  |  84.65%  |  [0.8426 0.8546 0.8426 0.8522 0.8502]  |  0.85 (+/- 0.01)  |  0.01818  |  0.01677  |
|  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  88.07%  |  [0.8874 0.8966 0.8886 0.888  0.8846]  |  0.89 (+/- 0.01)  |  0.8581  |  0.003922  |
|  12  |  PERCEPTRON  |  80.64%  |  [0.8166 0.8264 0.8144 0.81   0.8102]  |  0.82 (+/- 0.01)  |  0.09105  |  0.007187  |
|  13  |  RANDOM_FOREST_CLASSIFIER  |  85.45%  |  [0.8504 0.8584 0.8488 0.8518 0.8568]  |  0.85 (+/- 0.01)  |  8.792  |  0.7235  |
|  14  |  RIDGE_CLASSIFIER  |  86.90%  |  [0.8838 0.8952 0.8892 0.882  0.8788]  |  0.89 (+/- 0.01)  |  0.5019  |  0.00815  |
|  15  |  MAJORITY_VOTING_CLASSIFIER  |  87.88%  |  [0.8882 0.8976 0.8896 0.8884 0.8846]  |  0.89 (+/- 0.01)  |  10.81  |  0.7945  |
|  16  |  SOFT_VOTING_CLASSIFIER  |  87.73%  |  [0.8852 0.8972 0.8906 0.8908 0.8856]  |  0.89 (+/- 0.01)  |  10.29  |  0.6372  |
|  17  |  STACKING_CLASSIFIER  |  88.29%  |  [0.8924 0.902  0.8934 0.8898 0.889 ]  |  0.89 (+/- 0.01)  |  93.17  |  0.6377  |

* Accuracy score

![FINAL CLASSIFICATION TABLE (Accuracy score): IMDB_REVIEWS dataset (Binary classification)](https://github.com/ramonfigueiredopessoa/comp551-2020-p2_classification_of_textual_data/blob/master/code/results/all_classifiers_20newsgroups_and_imdb_using_binary_classification/IMDB_REVIEWS_binary_classification-just_accuracy_score.png)

* Accuracy score, normalized training time and normalized test time

![FINAL CLASSIFICATION TABLE (Accuracy score and normalized training and test time): IMDB_REVIEWS dataset (Binary classification)](https://github.com/ramonfigueiredopessoa/comp551-2020-p2_classification_of_textual_data/blob/master/code/results/all_classifiers_20newsgroups_and_imdb_using_binary_classification/IMDB_REVIEWS_binary_classification.png)

#### Computer settings used to run

* Operating system: Ubuntu 18.04.4 LTS (64-bit)
* Processor: Intel® Core™ i7-8700 CPU @ 3.20GHz × 12
* Memory: 64 GB

#### All logs

```
python /comp551-2020-p2_classification_of_textual_data/code/main.py --run_cross_validation --report --all_metrics --confusion_matrix --plot_accurary_and_time_together
Using TensorFlow backend.
2020-03-09 01:14:08.328185: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libnvinfer.so.6'; dlerror: libnvinfer.so.6: cannot open shared object file: No such file or directory
2020-03-09 01:14:08.328238: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libnvinfer_plugin.so.6'; dlerror: libnvinfer_plugin.so.6: cannot open shared object file: No such file or directory
2020-03-09 01:14:08.328243: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:30] Cannot dlopen some TensorRT libraries. If you would like to use Nvidia GPU with TensorRT, please make sure the missing libraries mentioned above are installed properly.
03/09/2020 01:14:08 AM - INFO - Program started...
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
                        RIDGE_CLASSIFIER, 15) MAJORITY_VOTING_CLASSIFIER
                        (using COMPLEMENT_NB, RIDGE_CLASSIFIER, LINEAR_SVC,
                        LOGISTIC_REGRESSION, PASSIVE_AGGRESSIVE_CLASSIFIER,
                        RANDOM_FOREST_CLASSIFIER), 16) SOFT_VOTING_CLASSIFIER
                        (using COMPLEMENT_NB, LOGISTIC_REGRESSION,
                        MULTINOMIAL_NB, RANDOM_FOREST_CLASSIFIER), 17)
                        STACKING_CLASSIFIER (using COMPLEMENT_NB,
                        RIDGE_CLASSIFIER, LINEAR_SVC, LOGISTIC_REGRESSION,
                        PASSIVE_AGGRESSIVE_CLASSIFIER,
                        RANDOM_FOREST_CLASSIFIER,
                        final_estimator=LINEAR_SVC)). Default: None. If
                        ml_algorithm_list is not provided, all ML algorithms
                        will be executed.
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
	ML algorithm list (If ml_algorithm_list is not provided, all ML algorithms will be executed) = None
	Use classifiers with default parameters. Default: False = Use classifiers with best parameters found using grid search. False
	Read dataset without shuffle data = False
	The number of CPUs to use to do the computation. If the provided number is negative or greater than the number of available CPUs, the system will use all the available CPUs. Default: -1 (-1 == all CPUs) = -1
	Run cross validation. Default: False = True
	Number of cross validation folds. Default: 5 = 5
	Use just the miniproject classifiers (1. LogisticRegression, 2. DecisionTreeClassifier, 3. LinearSVC, 4. AdaBoostClassifier, 5. RandomForestClassifier) =  False
	TWENTY_NEWS_GROUPS dataset using some categories (alt.atheism, talk.religion.misc, comp.graphics, sci.space) = False
	Do not remove newsgroup information that is easily overfit (headers, footers, quotes) = False
	Use IMDB multi-class labels (review score: 1, 2, 3, 4, 7, 8, 9, 10). If --use_imdb_multi_class_labels is False, the system uses binary classification (0 = neg and 1 = pos). Default: False = False
	Show the IMDB_REVIEWS and respective labels while read the dataset = False
	Print classification report = True
	Print all classification metrics (accuracy score, precision score, recall score, f1 score, f-beta score, jaccard score) =  True
	Select some number of features using a chi-squared test (For example: --chi2_select 10 = select 10 features using a chi-squared test) =  No number provided
	Print the confusion matrix = True
	Use a hashing vectorizer = False
	Use a count vectorizer = False
	Use a tf-idf vectorizer = True
	N features when using the hashing vectorizer = 65536
	Plot training time and test time together with accuracy score = True
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
done in 1.133759s at 12.156MB/s
n_samples: 11314, n_features: 101321

Extracting features from the test data using the same vectorizer
done in 0.633444s at 13.042MB/s
n_samples: 7532, n_features: 101321

	==> Using JSON with best parameters (selected using grid search) to the ADA_BOOST_CLASSIFIER classifier (multi-class classification) and TWENTY_NEWS_GROUPS dataset ===> JSON in dictionary format: {'learning_rate': 1, 'n_estimators': 200}
	 AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1,
                   n_estimators=200, random_state=0)
	==> Using JSON with best parameters (selected using grid search) to the BERNOULLI_NB classifier (multi-class classification) and TWENTY_NEWS_GROUPS dataset ===> JSON in dictionary format: {'alpha': 0.1, 'binarize': 0.1, 'fit_prior': False}
	 BernoulliNB(alpha=0.1, binarize=0.1, class_prior=None, fit_prior=False)
	==> Using JSON with best parameters (selected using grid search) to the COMPLEMENT_NB classifier (multi-class classification) and TWENTY_NEWS_GROUPS dataset ===> JSON in dictionary format: {'alpha': 0.5, 'fit_prior': False, 'norm': False}
	 ComplementNB(alpha=0.5, class_prior=None, fit_prior=False, norm=False)
	==> Using JSON with best parameters (selected using grid search) to the DECISION_TREE_CLASSIFIER classifier (multi-class classification) and TWENTY_NEWS_GROUPS dataset ===> JSON in dictionary format: {'criterion': 'gini', 'min_samples_split': 2, 'splitter': 'random'}
	 DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=None, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=0, splitter='random')
	==> Using JSON with best parameters (selected using grid search) to the GRADIENT_BOOSTING_CLASSIFIER classifier (multi-class classification) and TWENTY_NEWS_GROUPS dataset ===> JSON in dictionary format: {'learning_rate': 0.1, 'n_estimators': 200}
	 GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.1, loss='deviance', max_depth=3,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=200,
                           n_iter_no_change=None, presort='deprecated',
                           random_state=0, subsample=1.0, tol=0.0001,
                           validation_fraction=0.1, verbose=False,
                           warm_start=False)
	==> Using JSON with best parameters (selected using grid search) to the K_NEIGHBORS_CLASSIFIER classifier (multi-class classification) and TWENTY_NEWS_GROUPS dataset ===> JSON in dictionary format: {'leaf_size': 5, 'metric': 'euclidean', 'n_neighbors': 3, 'weights': 'distance'}
	 KNeighborsClassifier(algorithm='auto', leaf_size=5, metric='euclidean',
                     metric_params=None, n_jobs=-1, n_neighbors=3, p=2,
                     weights='distance')
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
	==> Using JSON with best parameters (selected using grid search) to the MULTINOMIAL_NB classifier (multi-class classification) and TWENTY_NEWS_GROUPS dataset ===> JSON in dictionary format: {'alpha': 0.01, 'fit_prior': True}
	 MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True)
	==> Using JSON with best parameters (selected using grid search) to the NEAREST_CENTROID classifier (multi-class classification) and TWENTY_NEWS_GROUPS dataset ===> JSON in dictionary format: {'metric': 'cosine'}
	 NearestCentroid(metric='cosine', shrink_threshold=None)
	==> Using JSON with best parameters (selected using grid search) to the PASSIVE_AGGRESSIVE_CLASSIFIER classifier (multi-class classification) and TWENTY_NEWS_GROUPS dataset ===> JSON in dictionary format: {'C': 0.01, 'early_stopping': False, 'tol': 0.0001, 'validation_fraction': 0.0001}
	 PassiveAggressiveClassifier(C=0.01, average=False, class_weight=None,
                            early_stopping=False, fit_intercept=True,
                            loss='hinge', max_iter=1000, n_iter_no_change=5,
                            n_jobs=-1, random_state=0, shuffle=True, tol=0.0001,
                            validation_fraction=0.0001, verbose=False,
                            warm_start=False)
	==> Using JSON with best parameters (selected using grid search) to the PERCEPTRON classifier (multi-class classification) and TWENTY_NEWS_GROUPS dataset ===> JSON in dictionary format: {'early_stopping': True, 'max_iter': 100, 'n_iter_no_change': 3, 'penalty': 'l2', 'tol': 0.0001, 'validation_fraction': 0.01}
	 Perceptron(alpha=0.0001, class_weight=None, early_stopping=True, eta0=1.0,
           fit_intercept=True, max_iter=100, n_iter_no_change=3, n_jobs=-1,
           penalty='l2', random_state=0, shuffle=True, tol=0.0001,
           validation_fraction=0.01, verbose=False, warm_start=False)
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
	 VotingClassifier(estimators=[('complement_nb',
                              ComplementNB(alpha=0.5, class_prior=None,
                                           fit_prior=False, norm=False)),
                             ('linear_svc',
                              LinearSVC(C=1.0, class_weight=None, dual=True,
                                        fit_intercept=True, intercept_scaling=1,
                                        loss='squared_hinge', max_iter=1000,
                                        multi_class='ovr', penalty='l2',
                                        random_state=0, tol=0.0001,
                                        verbose=False)),
                             ('logistic_regression',
                              Logisti...
                                                     min_samples_split=5,
                                                     min_weight_fraction_leaf=0.0,
                                                     n_estimators=200,
                                                     n_jobs=-1, oob_score=False,
                                                     random_state=0,
                                                     verbose=False,
                                                     warm_start=False)),
                             ('ridge_classifier',
                              RidgeClassifier(alpha=0.5, class_weight=None,
                                              copy_X=True, fit_intercept=True,
                                              max_iter=None, normalize=False,
                                              random_state=0, solver='auto',
                                              tol=0.001))],
                 flatten_transform=True, n_jobs=-1, voting='hard',
                 weights=None)
	==> Using JSON with best parameters (selected using grid search) to the COMPLEMENT_NB classifier (multi-class classification) and TWENTY_NEWS_GROUPS dataset ===> JSON in dictionary format: {'alpha': 0.5, 'fit_prior': False, 'norm': False}
	 ComplementNB(alpha=0.5, class_prior=None, fit_prior=False, norm=False)
	==> Using JSON with best parameters (selected using grid search) to the LOGISTIC_REGRESSION classifier (multi-class classification) and TWENTY_NEWS_GROUPS dataset ===> JSON in dictionary format: {'C': 10, 'tol': 0.001}
	 LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=-1, penalty='l2', random_state=0,
                   solver='lbfgs', tol=0.001, verbose=False, warm_start=False)
	==> Using JSON with best parameters (selected using grid search) to the MULTINOMIAL_NB classifier (multi-class classification) and TWENTY_NEWS_GROUPS dataset ===> JSON in dictionary format: {'alpha': 0.01, 'fit_prior': True}
	 MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True)
	==> Using JSON with best parameters (selected using grid search) to the RANDOM_FOREST_CLASSIFIER classifier (multi-class classification) and TWENTY_NEWS_GROUPS dataset ===> JSON in dictionary format: {'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 200}
	 RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=5,
                       min_weight_fraction_leaf=0.0, n_estimators=200,
                       n_jobs=-1, oob_score=False, random_state=0,
                       verbose=False, warm_start=False)
	 VotingClassifier(estimators=[('complement_nb',
                              ComplementNB(alpha=0.5, class_prior=None,
                                           fit_prior=False, norm=False)),
                             ('logistic_regression',
                              LogisticRegression(C=10, class_weight=None,
                                                 dual=False, fit_intercept=True,
                                                 intercept_scaling=1,
                                                 l1_ratio=None, max_iter=100,
                                                 multi_class='auto', n_jobs=-1,
                                                 penalty='l2', random_state=0,
                                                 solver='lbfgs', tol=0.001,
                                                 verbose=Fal...
                                                     criterion='gini',
                                                     max_depth=None,
                                                     max_features='auto',
                                                     max_leaf_nodes=None,
                                                     max_samples=None,
                                                     min_impurity_decrease=0.0,
                                                     min_impurity_split=None,
                                                     min_samples_leaf=1,
                                                     min_samples_split=5,
                                                     min_weight_fraction_leaf=0.0,
                                                     n_estimators=200,
                                                     n_jobs=-1, oob_score=False,
                                                     random_state=0,
                                                     verbose=False,
                                                     warm_start=False))],
                 flatten_transform=True, n_jobs=-1, voting='soft',
                 weights=None)
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
Classifier.ADA_BOOST_CLASSIFIER
________________________________________________________________________________
Training: 
AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1,
                   n_estimators=200, random_state=0)
train time: 19.260s
test time:  1.015s
accuracy:   0.440


cross validation:
	accuracy: 5-fold cross validation: [0.46973045 0.45779938 0.46575342 0.47017234 0.46993811]
	test accuracy: 5-fold cross validation accuracy: 0.47 (+/- 0.01)


===> Classification Report:

                          precision    recall  f1-score   support

             alt.atheism       0.39      0.25      0.30       319
           comp.graphics       0.38      0.46      0.41       389
 comp.os.ms-windows.misc       0.53      0.41      0.46       394
comp.sys.ibm.pc.hardware       0.50      0.48      0.49       392
   comp.sys.mac.hardware       0.62      0.48      0.54       385
          comp.windows.x       0.62      0.49      0.54       395
            misc.forsale       0.53      0.54      0.53       390
               rec.autos       0.73      0.39      0.51       396
         rec.motorcycles       0.89      0.44      0.59       398
      rec.sport.baseball       0.56      0.49      0.52       397
        rec.sport.hockey       0.77      0.43      0.55       399
               sci.crypt       0.80      0.49      0.61       396
         sci.electronics       0.12      0.64      0.20       393
                 sci.med       0.81      0.31      0.45       396
               sci.space       0.70      0.41      0.51       394
  soc.religion.christian       0.53      0.55      0.54       398
      talk.politics.guns       0.54      0.40      0.46       364
   talk.politics.mideast       0.88      0.54      0.67       376
      talk.politics.misc       0.25      0.30      0.27       310
      talk.religion.misc       0.23      0.15      0.18       251

                accuracy                           0.44      7532
               macro avg       0.57      0.43      0.47      7532
            weighted avg       0.58      0.44      0.48      7532



===> Classification Metrics:

accuracy classification score
	accuracy score:  0.4403876792352629
	accuracy score (normalize=False):  3317

compute the precision
	precision score (average=macro):  0.5691978779786757
	precision score (average=micro):  0.4403876792352629
	precision score (average=weighted):  0.5808641023152044
	precision score (average=None):  [0.39108911 0.375      0.53311258 0.50132626 0.62080537 0.61538462
 0.53316327 0.7255814  0.89393939 0.56268222 0.77130045 0.79752066
 0.11750936 0.80519481 0.69868996 0.53414634 0.54135338 0.87931034
 0.25414365 0.2327044 ]
	precision score (average=None, zero_division=1):  [0.39108911 0.375      0.53311258 0.50132626 0.62080537 0.61538462
 0.53316327 0.7255814  0.89393939 0.56268222 0.77130045 0.79752066
 0.11750936 0.80519481 0.69868996 0.53414634 0.54135338 0.87931034
 0.25414365 0.2327044 ]

compute the precision
	recall score (average=macro):  0.43186962332124706
	recall score (average=micro):  0.4403876792352629
	recall score (average=weighted):  0.4403876792352629
	recall score (average=None):  [0.2476489  0.46272494 0.40862944 0.48214286 0.48051948 0.48607595
 0.53589744 0.39393939 0.44472362 0.4861461  0.43107769 0.48737374
 0.63867684 0.31313131 0.40609137 0.55025126 0.3956044  0.54255319
 0.29677419 0.14741036]
	recall score (average=None, zero_division=1):  [0.2476489  0.46272494 0.40862944 0.48214286 0.48051948 0.48607595
 0.53589744 0.39393939 0.44472362 0.4861461  0.43107769 0.48737374
 0.63867684 0.31313131 0.40609137 0.55025126 0.3956044  0.54255319
 0.29677419 0.14741036]

compute the F1 score, also known as balanced F-score or F-measure
	f1 score (average=macro):  0.46815150604947026
	f1 score (average=micro):  0.4403876792352629
	f1 score (average=weighted):  0.4771601329912773
	f1 score (average=None):  [0.30326296 0.41426928 0.46264368 0.49154746 0.54172767 0.54314003
 0.53452685 0.5106383  0.59395973 0.52162162 0.55305466 0.60501567
 0.19849743 0.45090909 0.51364366 0.54207921 0.45714286 0.67105263
 0.27380952 0.1804878 ]

compute the F-beta score
	f beta score (average=macro):  0.5193119240389356
	f beta score (average=micro):  0.4403876792352629
	f beta score (average=weighted):  0.5295991356121272
	f beta score (average=None):  [0.35048802 0.38977913 0.50249688 0.49736842 0.58655675 0.58429702
 0.53370787 0.62101911 0.74369748 0.54550594 0.66615027 0.70747801
 0.14042744 0.61264822 0.61068702 0.53729146 0.50420168 0.78220859
 0.26166098 0.20856821]

compute the average Hamming loss
	hamming loss:  0.5596123207647371

jaccard similarity coefficient score
	jaccard score (average=macro):  0.31437989469133243
	jaccard score (average=None):  [0.17873303 0.26124819 0.30093458 0.32586207 0.37148594 0.37281553
 0.36474695 0.34285714 0.42243437 0.35283364 0.38222222 0.43370787
 0.11018437 0.29107981 0.34557235 0.37181664 0.2962963  0.5049505
 0.15862069 0.09919571]

confusion matrix:
[[ 79   6   2   0   0   0   5   1   1   1   3   1  82   1  10  65   2   8
   23  29]
 [  0 180  28  22   7  44   9   3   1   1   0   1  77   1   7   1   0   2
    5   0]
 [  0  40 161  45  13  40   1   0   1   0   0   1  71   2   6   0   5   0
    7   1]
 [  1  29  25 189  24   4  18   4   0   0   2   2  88   1   0   0   0   0
    4   1]
 [  1  23   6  32 185   6  19   0   0   1   0   8  94   2   4   0   0   1
    2   1]
 [  0  58  33  12   4 192   4   9   1   6   0   1  63   0   4   0   2   0
    5   1]
 [  1  21   7  41  14   2 209   5   1   4   2   3  66   1   6   3   0   0
    4   0]
 [  2  12   2   2  10   1  22 156   7   2   2   2 146   2   2   1   6   1
   17   1]
 [  2   6   4   3   9   0  16  12 177   1   0   2 148   0   2   5   3   0
    8   0]
 [  0   9   1   1   3   0   8   0   0 193  36   0 120   2   0   3   2   0
   11   8]
 [  0   0   0   1   1   1  11   0   4 119 172   1  76   0   1   1   2   1
    6   2]
 [  1  11   6   2   4   4   7   1   1   0   1 193 119   0   4   2  15   1
   23   1]
 [  0  26  15  22  12   3  17  12   2   3   1  11 251   0   6   2   1   0
    9   0]
 [ 11  28   1   1   0   2  11   1   0   1   0   0 181 124   3   8   1   0
   21   2]
 [  3  10   5   2  10   9  12   2   1   4   1   2 134   5 160   5   6   0
   21   2]
 [ 35   5   1   1   0   0   7   0   0   0   1   0  69   3   1 219   2   5
   13  36]
 [  8   6   1   1   1   0   5   3   0   0   1   8 100   4   5   8 144   2
   48  19]
 [ 21   4   0   0   0   2   2   2   0   4   0   3  75   1   1  16   6 204
   29   6]
 [  8   3   2   0   1   2   3   2   1   3   1   1 112   5   5   4  50   3
   92  12]
 [ 29   3   2   0   0   0   6   2   0   0   0   2  64   0   2  67  19   4
   14  37]]

================================================================================
Classifier.BERNOULLI_NB
________________________________________________________________________________
Training: 
BernoulliNB(alpha=0.1, binarize=0.1, class_prior=None, fit_prior=False)
train time: 0.076s
test time:  0.053s
accuracy:   0.626


cross validation:
	accuracy: 5-fold cross validation: [0.69200177 0.67741935 0.6752099  0.67388422 0.67992927]
	test accuracy: 5-fold cross validation accuracy: 0.68 (+/- 0.01)
dimensionality: 101321
density: 1.000000



===> Classification Report:

                          precision    recall  f1-score   support

             alt.atheism       0.45      0.39      0.42       319
           comp.graphics       0.51      0.63      0.57       389
 comp.os.ms-windows.misc       0.50      0.58      0.54       394
comp.sys.ibm.pc.hardware       0.61      0.64      0.62       392
   comp.sys.mac.hardware       0.64      0.65      0.65       385
          comp.windows.x       0.80      0.63      0.70       395
            misc.forsale       0.86      0.65      0.74       390
               rec.autos       0.67      0.72      0.69       396
         rec.motorcycles       0.67      0.72      0.70       398
      rec.sport.baseball       0.35      0.87      0.49       397
        rec.sport.hockey       0.90      0.83      0.86       399
               sci.crypt       0.79      0.69      0.73       396
         sci.electronics       0.70      0.51      0.59       393
                 sci.med       0.86      0.62      0.72       396
               sci.space       0.70      0.71      0.70       394
  soc.religion.christian       0.61      0.71      0.66       398
      talk.politics.guns       0.57      0.64      0.60       364
   talk.politics.mideast       0.76      0.62      0.68       376
      talk.politics.misc       0.64      0.30      0.41       310
      talk.religion.misc       0.40      0.09      0.15       251

                accuracy                           0.63      7532
               macro avg       0.65      0.61      0.61      7532
            weighted avg       0.66      0.63      0.62      7532



===> Classification Metrics:

accuracy classification score
	accuracy score:  0.6255974508762613
	accuracy score (normalize=False):  4712

compute the precision
	precision score (average=macro):  0.6496495031030469
	precision score (average=micro):  0.6255974508762613
	precision score (average=weighted):  0.6566832164966478
	precision score (average=None):  [0.45387454 0.51351351 0.50328228 0.6097561  0.64194373 0.80194805
 0.85762712 0.66901408 0.67213115 0.34534535 0.8972973  0.7884058
 0.69550173 0.86363636 0.69924812 0.61304348 0.56934307 0.75816993
 0.64335664 0.39655172]
	precision score (average=None, zero_division=1):  [0.45387454 0.51351351 0.50328228 0.6097561  0.64194373 0.80194805
 0.85762712 0.66901408 0.67213115 0.34534535 0.8972973  0.7884058
 0.69550173 0.86363636 0.69924812 0.61304348 0.56934307 0.75816993
 0.64335664 0.39655172]

compute the precision
	recall score (average=macro):  0.6098471336476433
	recall score (average=micro):  0.6255974508762613
	recall score (average=weighted):  0.6255974508762613
	recall score (average=None):  [0.38557994 0.63496144 0.58375635 0.6377551  0.65194805 0.62531646
 0.64871795 0.71969697 0.72110553 0.86901763 0.8320802  0.68686869
 0.51145038 0.62373737 0.70812183 0.70854271 0.64285714 0.61702128
 0.29677419 0.09163347]
	recall score (average=None, zero_division=1):  [0.38557994 0.63496144 0.58375635 0.6377551  0.65194805 0.62531646
 0.64871795 0.71969697 0.72110553 0.86901763 0.8320802  0.68686869
 0.51145038 0.62373737 0.70812183 0.70854271 0.64285714 0.61702128
 0.29677419 0.09163347]

compute the F1 score, also known as balanced F-score or F-measure
	f1 score (average=macro):  0.6116078371996332
	f1 score (average=micro):  0.6255974508762613
	f1 score (average=weighted):  0.6246940514115146
	f1 score (average=None):  [0.41694915 0.56781609 0.54054054 0.6234414  0.64690722 0.7027027
 0.73868613 0.69343066 0.69575758 0.49426934 0.86345904 0.73414305
 0.58944282 0.72434018 0.703657   0.65734266 0.60387097 0.68035191
 0.40618102 0.14886731]

compute the F-beta score
	f beta score (average=macro):  0.6282323519959406
	f beta score (average=micro):  0.6255974508762613
	f beta score (average=weighted):  0.6388387216064568
	f beta score (average=None):  [0.4383464  0.53393861 0.51755176 0.61515748 0.64391996 0.75906577
 0.80573248 0.67857143 0.68138651 0.39267016 0.88344864 0.76576577
 0.64880568 0.80194805 0.70100503 0.63002681 0.58266932 0.725
 0.52154195 0.23809524]

compute the average Hamming loss
	hamming loss:  0.3744025491237387

jaccard similarity coefficient score
	jaccard score (average=macro):  0.4561045781857641
	jaccard score (average=None):  [0.2633833  0.3964687  0.37037037 0.45289855 0.47809524 0.54166667
 0.58564815 0.53072626 0.53345725 0.3282588  0.7597254  0.57995736
 0.41787942 0.56781609 0.54280156 0.48958333 0.43253235 0.51555556
 0.25484765 0.08041958]

confusion matrix:
[[123   7   3   2   0   2   0  11   7  42   3   6   4   1  11  56  14  13
    1  13]
 [  5 247  33   5  16  27   4   1   4  30   0   4   4   0   7   0   0   2
    0   0]
 [  4  31 230  46  13  17   2   1   3  29   0   3   2   1   8   2   2   0
    0   0]
 [  0  11  55 250  37   1   7   3   0  11   0   2  14   0   1   0   0   0
    0   0]
 [  0  15  21  32 251   1   8   8   3  20   0   3  12   1   7   0   1   1
    1   0]
 [  0  56  37   6   8 247   1   3   3  24   1   2   3   0   2   0   0   2
    0   0]
 [  0   7  10  31  20   0 253  14   9  33   2   0   5   0   3   0   1   0
    1   1]
 [  2   3   4   0   5   0   5 285  30  38   1   0   9   1   7   1   3   1
    1   0]
 [  2   7   1   2   4   0   2  36 287  31   0   1  10   1   3   1   5   2
    2   1]
 [  4   8   4   1   0   0   1   3   3 345  14   2   0   1   0   2   3   2
    4   0]
 [  3   3   2   0   1   0   0   0   3  46 332   1   0   3   1   1   1   1
    0   1]
 [  4  17  14   4   3   1   1   4   5  35   2 272   2   1   7   2  13   5
    4   0]
 [  1  25  22  25  25   4   6  17  12  22   2  14 201   7  10   0   0   0
    0   0]
 [  5  16   6   3   1   0   2  15  14  40   2   0   9 247  15   9   7   3
    2   0]
 [  6   8   5   0   1   5   0   6   7  35   3   5  11   3 279   2   0   9
    8   1]
 [ 23   7   3   0   2   1   1   2   5  45   0   2   0   1   6 282   2   3
    4   9]
 [  5   3   5   2   0   0   0   6  14  31   0  13   1   4  13   7 234  11
   11   4]
 [ 19   4   0   0   0   0   0   1   9  71   0   3   0   0   3  14  11 232
    8   1]
 [ 19   1   0   1   2   2   1   4   7  42   6   8   1   7   9   6  87  11
   92   4]
 [ 46   5   2   0   2   0   1   6   2  29   2   4   1   7   7  75  27   8
    4  23]]

================================================================================
Classifier.COMPLEMENT_NB
________________________________________________________________________________
Training: 
ComplementNB(alpha=0.5, class_prior=None, fit_prior=False, norm=False)
train time: 0.065s
test time:  0.010s
accuracy:   0.712


cross validation:
	accuracy: 5-fold cross validation: [0.776403   0.75828546 0.78126381 0.77507733 0.76967286]
	test accuracy: 5-fold cross validation accuracy: 0.77 (+/- 0.02)
dimensionality: 101321
density: 1.000000



===> Classification Report:

                          precision    recall  f1-score   support

             alt.atheism       0.31      0.42      0.35       319
           comp.graphics       0.72      0.69      0.70       389
 comp.os.ms-windows.misc       0.74      0.54      0.62       394
comp.sys.ibm.pc.hardware       0.64      0.72      0.68       392
   comp.sys.mac.hardware       0.77      0.73      0.75       385
          comp.windows.x       0.77      0.80      0.79       395
            misc.forsale       0.76      0.74      0.75       390
               rec.autos       0.83      0.75      0.79       396
         rec.motorcycles       0.82      0.76      0.79       398
      rec.sport.baseball       0.90      0.85      0.88       397
        rec.sport.hockey       0.87      0.94      0.90       399
               sci.crypt       0.74      0.80      0.77       396
         sci.electronics       0.71      0.56      0.63       393
                 sci.med       0.78      0.81      0.79       396
               sci.space       0.81      0.80      0.80       394
  soc.religion.christian       0.55      0.91      0.68       398
      talk.politics.guns       0.59      0.72      0.65       364
   talk.politics.mideast       0.76      0.85      0.80       376
      talk.politics.misc       0.68      0.42      0.52       310
      talk.religion.misc       0.46      0.11      0.17       251

                accuracy                           0.71      7532
               macro avg       0.71      0.70      0.69      7532
            weighted avg       0.72      0.71      0.71      7532



===> Classification Metrics:

accuracy classification score
	accuracy score:  0.712161444503452
	accuracy score (normalize=False):  5364

compute the precision
	precision score (average=macro):  0.7096240654324812
	precision score (average=micro):  0.712161444503452
	precision score (average=weighted):  0.7192685035239748
	precision score (average=None):  [0.30787037 0.71774194 0.73702422 0.64449541 0.76775956 0.76941748
 0.75789474 0.83146067 0.8172043  0.90133333 0.87006961 0.74125874
 0.70926518 0.7799511  0.8056266  0.54946728 0.58916479 0.76076555
 0.67708333 0.45762712]
	precision score (average=None, zero_division=1):  [0.30787037 0.71774194 0.73702422 0.64449541 0.76775956 0.76941748
 0.75789474 0.83146067 0.8172043  0.90133333 0.87006961 0.74125874
 0.70926518 0.7799511  0.8056266  0.54946728 0.58916479 0.76076555
 0.67708333 0.45762712]

compute the precision
	recall score (average=macro):  0.6951921194129647
	recall score (average=micro):  0.712161444503452
	recall score (average=weighted):  0.712161444503452
	recall score (average=None):  [0.4169279  0.68637532 0.54060914 0.71683673 0.72987013 0.80253165
 0.73846154 0.74747475 0.7638191  0.85138539 0.93984962 0.8030303
 0.5648855  0.80555556 0.79949239 0.90703518 0.71703297 0.84574468
 0.41935484 0.10756972]
	recall score (average=None, zero_division=1):  [0.4169279  0.68637532 0.54060914 0.71683673 0.72987013 0.80253165
 0.73846154 0.74747475 0.7638191  0.85138539 0.93984962 0.8030303
 0.5648855  0.80555556 0.79949239 0.90703518 0.71703297 0.84574468
 0.41935484 0.10756972]

compute the F1 score, also known as balanced F-score or F-measure
	f1 score (average=macro):  0.690785686397327
	f1 score (average=micro):  0.712161444503452
	f1 score (average=weighted):  0.7060400139153535
	f1 score (average=None):  [0.35419441 0.70170828 0.62371889 0.67874396 0.74833555 0.78562577
 0.74805195 0.78723404 0.78961039 0.87564767 0.90361446 0.77090909
 0.62889518 0.79254658 0.80254777 0.68436019 0.64684015 0.80100756
 0.51792829 0.17419355]

compute the F-beta score
	f beta score (average=macro):  0.69723615755366
	f beta score (average=micro):  0.712161444503452
	f beta score (average=weighted):  0.7103077176297662
	f beta score (average=None):  [0.32486566 0.71124134 0.68709677 0.65777154 0.7598702  0.77581987
 0.7539267  0.81318681 0.80593849 0.89088034 0.88318417 0.75284091
 0.67477204 0.78494094 0.80439224 0.59649703 0.61095506 0.77636719
 0.60296846 0.27720739]

compute the average Hamming loss
	hamming loss:  0.2878385554965481

jaccard similarity coefficient score
	jaccard score (average=macro):  0.5496500067595335
	jaccard score (average=None):  [0.21521036 0.54048583 0.45319149 0.51371115 0.59787234 0.64693878
 0.59751037 0.64912281 0.65236052 0.77880184 0.82417582 0.62721893
 0.45867769 0.6563786  0.67021277 0.52017291 0.47802198 0.66806723
 0.34946237 0.09540636]

confusion matrix:
[[133   0   2   2   1   3   1   2   0   3   7   5   2   7  10  97   7  22
    3  12]
 [ 10 267  14  17   7  38   7   0   2   2   0  13   1   0   5   1   2   2
    1   0]
 [ 22  20 213  46  17  25   8   3   2   0   3   4   4   5   9   6   1   3
    3   0]
 [  7  11  22 281  16   6  13   1   0   1   2   9  20   0   1   0   0   0
    1   1]
 [ 14   4   8  25 281   8  12   5   1   0   0   6  10   3   3   2   3   0
    0   0]
 [  7  37  11   5   4 317   1   0   0   1   0   2   2   1   4   0   2   1
    0   0]
 [  8   0   2  31  17   1 288   8   6   5   3   1   9   2   2   2   2   1
    2   0]
 [ 25   1   1   0   2   1   9 296  25   0   2   2  15   2   3   3   1   4
    3   1]
 [ 17   3   0   2   1   0   5  14 304   4   6   4   9   5   3   4   8   2
    5   2]
 [ 20   2   0   0   0   1   4   0   4 338  14   1   1   4   0   4   1   0
    3   0]
 [ 10   0   0   0   0   1   0   0   0   3 375   1   0   2   0   5   2   0
    0   0]
 [ 18   5   5   1   2   3   1   2   1   2   0 318   5   1   4   2  19   4
    3   0]
 [ 12   6   6  22  16   2  16   9   8   2   4  35 222  19   4   3   3   2
    2   0]
 [ 17   4   0   1   0   0   5   5   3   2   4   2   5 319   4   9   7   4
    4   1]
 [ 21   7   1   0   1   2   4   6   3   1   1   1   2   6 315   5   2   9
    4   3]
 [ 21   2   1   0   0   1   1   0   1   2   0   1   0   2   1 361   0   1
    2   1]
 [ 13   1   2   1   0   0   2   1   5   3   2  12   1   7  10  15 261  13
   10   5]
 [ 11   0   0   0   1   0   1   1   4   3   2   3   1   2   0  11   6 318
   10   2]
 [ 16   0   0   1   0   2   1   2   1   1   5   8   2  11   7   9  92  18
  130   4]
 [ 30   2   1   1   0   1   1   1   2   2   1   1   2  11   6 118  24  14
    6  27]]

================================================================================
Classifier.DECISION_TREE_CLASSIFIER
________________________________________________________________________________
Training: 
DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=None, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=0, splitter='random')
train time: 9.037s
test time:  0.006s
accuracy:   0.447


cross validation:
	accuracy: 5-fold cross validation: [0.49094123 0.48696421 0.47547503 0.49270879 0.49646331]
	test accuracy: 5-fold cross validation accuracy: 0.49 (+/- 0.01)


===> Classification Report:

                          precision    recall  f1-score   support

             alt.atheism       0.30      0.28      0.29       319
           comp.graphics       0.44      0.44      0.44       389
 comp.os.ms-windows.misc       0.41      0.41      0.41       394
comp.sys.ibm.pc.hardware       0.39      0.41      0.40       392
   comp.sys.mac.hardware       0.44      0.45      0.45       385
          comp.windows.x       0.53      0.46      0.49       395
            misc.forsale       0.56      0.58      0.57       390
               rec.autos       0.26      0.54      0.35       396
         rec.motorcycles       0.57      0.55      0.56       398
      rec.sport.baseball       0.50      0.42      0.46       397
        rec.sport.hockey       0.59      0.64      0.61       399
               sci.crypt       0.69      0.49      0.58       396
         sci.electronics       0.34      0.33      0.33       393
                 sci.med       0.47      0.43      0.45       396
               sci.space       0.53      0.51      0.52       394
  soc.religion.christian       0.48      0.54      0.51       398
      talk.politics.guns       0.44      0.38      0.41       364
   talk.politics.mideast       0.62      0.52      0.56       376
      talk.politics.misc       0.27      0.21      0.24       310
      talk.religion.misc       0.17      0.14      0.15       251

                accuracy                           0.45      7532
               macro avg       0.45      0.44      0.44      7532
            weighted avg       0.46      0.45      0.45      7532



===> Classification Metrics:

accuracy classification score
	accuracy score:  0.4471587891662241
	accuracy score (normalize=False):  3368

compute the precision
	precision score (average=macro):  0.4504638503629657
	precision score (average=micro):  0.4471587891662241
	precision score (average=weighted):  0.45921788675797415
	precision score (average=None):  [0.30067568 0.44102564 0.41388175 0.39130435 0.43969849 0.52737752
 0.5591133  0.26237624 0.56847545 0.50299401 0.58932715 0.69014085
 0.33854167 0.47486034 0.52755906 0.48098434 0.44267516 0.62379421
 0.26859504 0.16587678]
	precision score (average=None, zero_division=1):  [0.30067568 0.44102564 0.41388175 0.39130435 0.43969849 0.52737752
 0.5591133  0.26237624 0.56847545 0.50299401 0.58932715 0.69014085
 0.33854167 0.47486034 0.52755906 0.48098434 0.44267516 0.62379421
 0.26859504 0.16587678]

compute the precision
	recall score (average=macro):  0.4371575627601687
	recall score (average=micro):  0.4471587891662241
	recall score (average=weighted):  0.4471587891662241
	recall score (average=None):  [0.27899687 0.44215938 0.40862944 0.41326531 0.45454545 0.46329114
 0.58205128 0.53535354 0.55276382 0.4231738  0.63659148 0.49494949
 0.3307888  0.42929293 0.51015228 0.54020101 0.38186813 0.51595745
 0.20967742 0.13944223]
	recall score (average=None, zero_division=1):  [0.27899687 0.44215938 0.40862944 0.41326531 0.45454545 0.46329114
 0.58205128 0.53535354 0.55276382 0.4231738  0.63659148 0.49494949
 0.3307888  0.42929293 0.51015228 0.54020101 0.38186813 0.51595745
 0.20967742 0.13944223]

compute the F1 score, also known as balanced F-score or F-measure
	f1 score (average=macro):  0.43953255451929796
	f1 score (average=micro):  0.4471587891662241
	f1 score (average=weighted):  0.44878004006584765
	f1 score (average=None):  [0.28943089 0.44159178 0.41123883 0.40198511 0.44699872 0.49326146
 0.57035176 0.35215947 0.56050955 0.45964432 0.61204819 0.57647059
 0.33462033 0.45092838 0.51870968 0.50887574 0.4100295  0.56477438
 0.23550725 0.15151515]

compute the F-beta score
	f beta score (average=macro):  0.44523090867793036
	f beta score (average=micro):  0.4471587891662241
	f beta score (average=weighted):  0.45416341436837726
	f beta score (average=None):  [0.29607452 0.44125192 0.41282051 0.39550781 0.44258978 0.51318003
 0.56355511 0.292172   0.56526208 0.4847086  0.59821008 0.63968668
 0.33696216 0.46498906 0.52398332 0.49176578 0.42901235 0.59876543
 0.2543036  0.15981735]

compute the average Hamming loss
	hamming loss:  0.5528412108337759

jaccard similarity coefficient score
	jaccard score (average=macro):  0.2887411846666291
	jaccard score (average=None):  [0.16920152 0.28336079 0.25884244 0.2515528  0.28782895 0.3273703
 0.39894552 0.21370968 0.38938053 0.29840142 0.44097222 0.40495868
 0.20092736 0.29109589 0.35017422 0.34126984 0.25788497 0.39350913
 0.13347023 0.08196721]

confusion matrix:
[[ 89   4   1   1   6   3   4  30  11  13   7   5   4   8  14  60   6  13
   11  29]
 [  5 172  38  26  19  27  13  16   4   9   2   6  19   7  14   1   2   5
    1   3]
 [ 10  37 161  46  20  27  11  25   5  10   2   1   9   9   4   1   3   5
    2   6]
 [  3  29  35 162  27  16  14  22  11   4   4   3  34   4  11   2   4   2
    3   2]
 [  3  14  11  34 175  10  20  30   7   4   5   6  30   7   7   2   2   5
    9   4]
 [  6  34  47  20  19 183  10  17   3   6   1   4  13  11  13   1   1   2
    1   3]
 [  2  16   3  24  19   3 227  34   7   1   3   4  18   2   8   4   2   2
    9   2]
 [  4   5   9  18  14   8  19 212  27   6   4   0  21  10  10   2   8   5
    9   5]
 [ 11   4   4   6  12   6  10  45 220  11   5   2  12   9   9   8   5   8
   10   1]
 [ 12   1   3   6   3   4  10  40   9 168  94   4   6   9   6   2   1   5
   11   3]
 [  3   1   1   2   6   2   7  27   7  54 254   3   2   3   4   4   2   4
    5   8]
 [  5  13   7   4  14   7  11  42   8   2   3 196  23   9   8   6  19   2
   10   7]
 [  4  21  13  34  29   9  16  38  15   8   6  15 130  16  16   5   1   5
    8   4]
 [ 17  15   9   8   8  16  15  43  16   5   4   2  18 170  10  11   4  10
    8   7]
 [ 12  11   5   7   7   9   7  38   6   7  11   5  16  21 201   4   8   4
   10   5]
 [ 34   2   9   3   4   2   1  33   4   5   3   2   5   8   7 215   4  10
    8  39]
 [  7   4  15   3   4   4   6  45  11   3   8  10   5  13  12  15 139   9
   31  20]
 [ 25   1   4   2   4   3   2  20   5  10   4   7   3   9  11  17  18 194
   22  15]
 [ 14   3  10   5   4   6   0  29   8   5   6   8  12  22  10  13  62  15
   65  13]
 [ 30   3   4   3   4   2   3  22   3   3   5   1   4  11   6  74  23   6
    9  35]]

================================================================================
Classifier.GRADIENT_BOOSTING_CLASSIFIER
________________________________________________________________________________
Training: 
GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.1, loss='deviance', max_depth=3,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=200,
                           n_iter_no_change=None, presort='deprecated',
                           random_state=0, subsample=1.0, tol=0.0001,
                           validation_fraction=0.1, verbose=False,
                           warm_start=False)
train time: 658.514s
test time:  0.393s
accuracy:   0.594


cross validation:
	accuracy: 5-fold cross validation: [0.65930181 0.62704375 0.64781264 0.6548829  0.64721485]
	test accuracy: 5-fold cross validation accuracy: 0.65 (+/- 0.02)


===> Classification Report:

                          precision    recall  f1-score   support

             alt.atheism       0.43      0.35      0.38       319
           comp.graphics       0.57      0.64      0.60       389
 comp.os.ms-windows.misc       0.60      0.57      0.58       394
comp.sys.ibm.pc.hardware       0.56      0.58      0.57       392
   comp.sys.mac.hardware       0.67      0.63      0.65       385
          comp.windows.x       0.73      0.58      0.65       395
            misc.forsale       0.69      0.67      0.68       390
               rec.autos       0.69      0.59      0.64       396
         rec.motorcycles       0.76      0.64      0.70       398
      rec.sport.baseball       0.79      0.71      0.75       397
        rec.sport.hockey       0.80      0.76      0.78       399
               sci.crypt       0.78      0.64      0.70       396
         sci.electronics       0.21      0.57      0.30       393
                 sci.med       0.73      0.62      0.67       396
               sci.space       0.71      0.59      0.64       394
  soc.religion.christian       0.62      0.69      0.65       398
      talk.politics.guns       0.56      0.59      0.57       364
   talk.politics.mideast       0.81      0.62      0.70       376
      talk.politics.misc       0.53      0.38      0.44       310
      talk.religion.misc       0.31      0.22      0.25       251

                accuracy                           0.59      7532
               macro avg       0.63      0.58      0.60      7532
            weighted avg       0.64      0.59      0.61      7532



===> Classification Metrics:

accuracy classification score
	accuracy score:  0.5942644715878916
	accuracy score (normalize=False):  4476

compute the precision
	precision score (average=macro):  0.6279924415990819
	precision score (average=micro):  0.5942644715878916
	precision score (average=weighted):  0.6373849455098752
	precision score (average=None):  [0.42692308 0.56719818 0.60053619 0.56265356 0.67217631 0.73015873
 0.68865435 0.68823529 0.75964392 0.79271709 0.80474934 0.7826087
 0.2056933  0.73432836 0.70820669 0.62217195 0.56498674 0.8125
 0.52888889 0.30681818]
	precision score (average=None, zero_division=1):  [0.42692308 0.56719818 0.60053619 0.56265356 0.67217631 0.73015873
 0.68865435 0.68823529 0.75964392 0.79271709 0.80474934 0.7826087
 0.2056933  0.73432836 0.70820669 0.62217195 0.56498674 0.8125
 0.52888889 0.30681818]

compute the precision
	recall score (average=macro):  0.5826913074077184
	recall score (average=micro):  0.5942644715878916
	recall score (average=weighted):  0.5942644715878916
	recall score (average=None):  [0.34796238 0.64010283 0.56852792 0.58418367 0.63376623 0.58227848
 0.66923077 0.59090909 0.64321608 0.71284635 0.76441103 0.63636364
 0.56997455 0.62121212 0.59137056 0.69095477 0.58516484 0.62234043
 0.38387097 0.21513944]
	recall score (average=None, zero_division=1):  [0.34796238 0.64010283 0.56852792 0.58418367 0.63376623 0.58227848
 0.66923077 0.59090909 0.64321608 0.71284635 0.76441103 0.63636364
 0.56997455 0.62121212 0.59137056 0.69095477 0.58516484 0.62234043
 0.38387097 0.21513944]

compute the F1 score, also known as balanced F-score or F-measure
	f1 score (average=macro):  0.5971284138924248
	f1 score (average=micro):  0.5942644715878916
	f1 score (average=weighted):  0.6075875455649488
	f1 score (average=None):  [0.38341969 0.60144928 0.58409387 0.57321652 0.65240642 0.64788732
 0.67880364 0.63586957 0.69659864 0.75066313 0.7840617  0.70194986
 0.3022942  0.67305062 0.64453665 0.6547619  0.57489879 0.70481928
 0.44485981 0.2529274 ]

compute the F-beta score
	f beta score (average=macro):  0.614099661932886
	f beta score (average=micro):  0.5942644715878916
	f beta score (average=weighted):  0.6239543204533676
	f beta score (average=None):  [0.40838852 0.58041958 0.59384942 0.56683168 0.66412629 0.69486405
 0.68467996 0.66628702 0.73310424 0.77534247 0.79634465 0.74821853
 0.23583912 0.70852535 0.68128655 0.63481071 0.56891026 0.76570681
 0.49173554 0.28272251]

compute the average Hamming loss
	hamming loss:  0.40573552841210836

jaccard similarity coefficient score
	jaccard score (average=macro):  0.4385771958123489
	jaccard score (average=None):  [0.23717949 0.43005181 0.41252302 0.40175439 0.48412698 0.47916667
 0.51377953 0.46613546 0.53444676 0.60084926 0.6448203  0.54077253
 0.17806041 0.50721649 0.4755102  0.48672566 0.40340909 0.54418605
 0.28605769 0.14477212]

confusion matrix:
[[111   5   2   4   3   2  10   3   4   5   2   4  39   6  12  54   7   7
    6  33]
 [  3 249  21  11  10  24   3   1   2   0   3   3  41   2  10   0   1   4
    1   0]
 [  3  31 224  37  13  14   4   5   2   2   2   1  36   2   6   3   2   1
    1   5]
 [  1  13  37 229  17   9  19   1   1   2   1   2  50   2   3   2   0   2
    1   0]
 [  0   9   8  36 244   3  10   0   1   1   4   6  51   5   2   3   0   2
    0   0]
 [  1  54  38   7   6 230   3   1   0   1   2   2  35   3   6   2   1   1
    1   1]
 [  1  10   3  21  18   3 261  12   3   3   4   1  37   1   5   2   3   1
    1   0]
 [  2   3   4   5   5   4   9 234  17   0   2   3  79   9   2   2  11   1
    2   2]
 [  2   4   3   6   5   1  10  21 256   3   2   1  57   4   3   3   4   2
    5   6]
 [  4   3   1   2   1   1   4   2   5 283  33   2  39   3   1   3   5   1
    2   2]
 [  0   0   5   0   4   2   4   0   4  25 305   1  32   3   3   1   1   3
    4   2]
 [  4   6   0   6   8   4   4   5   2   1   2 252  55   4   7   1  19   3
   10   3]
 [  0  16  11  27  11   5  13  20   6  10   5  17 224   6  11   0   4   0
    4   3]
 [  4  10   3   1   3   1  10   9  11   4   2   0  65 246   3   4   4   2
   12   2]
 [  6  12   3   8   8   3   4   4   8   3   3   6  72   4 233   4   2   3
    6   2]
 [ 27   4   1   0   0   2   2   0   2   1   0   2  39   3   2 275   0   4
    8  26]
 [ 10   4   2   1   1   3   4   6   3   5   2  11  41   7   3   5 213   2
   20  21]
 [ 34   0   2   2   3   3   2   4   3   5   1   3  39   3   2  10   6 234
   17   3]
 [ 12   2   3   2   2   0   3   6   2   2   3   4  29  16   6   3  76   9
  119  11]
 [ 35   4   2   2   1   1   0   6   5   1   1   1  29   6   9  65  18   6
    5  54]]

================================================================================
Classifier.K_NEIGHBORS_CLASSIFIER
________________________________________________________________________________
Training: 
KNeighborsClassifier(algorithm='auto', leaf_size=5, metric='euclidean',
                     metric_params=None, n_jobs=-1, n_neighbors=3, p=2,
                     weights='distance')
train time: 0.003s
test time:  1.298s
accuracy:   0.085


cross validation:
	accuracy: 5-fold cross validation: [0.12417145 0.11842687 0.12063632 0.12240389 0.12113174]
	test accuracy: 5-fold cross validation accuracy: 0.12 (+/- 0.00)


===> Classification Report:

                          precision    recall  f1-score   support

             alt.atheism       0.95      0.06      0.11       319
           comp.graphics       0.79      0.03      0.05       389
 comp.os.ms-windows.misc       0.67      0.02      0.03       394
comp.sys.ibm.pc.hardware       0.88      0.06      0.11       392
   comp.sys.mac.hardware       0.79      0.03      0.06       385
          comp.windows.x       0.95      0.05      0.09       395
            misc.forsale       1.00      0.12      0.21       390
               rec.autos       0.62      0.01      0.02       396
         rec.motorcycles       0.88      0.06      0.11       398
      rec.sport.baseball       0.92      0.03      0.06       397
        rec.sport.hockey       0.96      0.06      0.11       399
               sci.crypt       0.61      0.04      0.07       396
         sci.electronics       0.92      0.03      0.05       393
                 sci.med       1.00      0.02      0.03       396
               sci.space       0.79      0.06      0.11       394
  soc.religion.christian       0.75      0.02      0.03       398
      talk.politics.guns       0.50      0.01      0.02       364
   talk.politics.mideast       0.05      1.00      0.10       376
      talk.politics.misc       1.00      0.01      0.01       310
      talk.religion.misc       0.50      0.00      0.01       251

                accuracy                           0.08      7532
               macro avg       0.78      0.08      0.07      7532
            weighted avg       0.78      0.08      0.07      7532



===> Classification Metrics:

accuracy classification score
	accuracy score:  0.0848380244291025
	accuracy score (normalize=False):  639

compute the precision
	precision score (average=macro):  0.7765581805953655
	precision score (average=micro):  0.0848380244291025
	precision score (average=weighted):  0.7804623598851294
	precision score (average=None):  [0.95       0.78571429 0.66666667 0.88       0.78571429 0.95
 1.         0.625      0.88461538 0.92307692 0.96       0.60869565
 0.91666667 1.         0.79310345 0.75       0.5        0.0519103
 1.         0.5       ]
	precision score (average=None, zero_division=1):  [0.95       0.78571429 0.66666667 0.88       0.78571429 0.95
 1.         0.625      0.88461538 0.92307692 0.96       0.60869565
 0.91666667 1.         0.79310345 0.75       0.5        0.0519103
 1.         0.5       ]

compute the precision
	recall score (average=macro):  0.08412641144607133
	recall score (average=micro):  0.0848380244291025
	recall score (average=weighted):  0.0848380244291025
	recall score (average=None):  [0.05956113 0.02827763 0.01522843 0.05612245 0.02857143 0.04810127
 0.11538462 0.01262626 0.05778894 0.0302267  0.06015038 0.03535354
 0.02798982 0.01767677 0.05837563 0.01507538 0.00824176 0.99734043
 0.00645161 0.00398406]
	recall score (average=None, zero_division=1):  [0.05956113 0.02827763 0.01522843 0.05612245 0.02857143 0.04810127
 0.11538462 0.01262626 0.05778894 0.0302267  0.06015038 0.03535354
 0.02798982 0.01767677 0.05837563 0.01507538 0.00824176 0.99734043
 0.00645161 0.00398406]

compute the F1 score, also known as balanced F-score or F-measure
	f1 score (average=macro):  0.06951905270780404
	f1 score (average=micro):  0.0848380244291025
	f1 score (average=weighted):  0.07099160936848287
	f1 score (average=None):  [0.1120944  0.05459057 0.02977667 0.10551559 0.05513784 0.09156627
 0.20689655 0.02475248 0.10849057 0.05853659 0.11320755 0.06682578
 0.05432099 0.03473945 0.10874704 0.02955665 0.01621622 0.09868421
 0.01282051 0.00790514]

compute the F-beta score
	f beta score (average=macro):  0.141776233646035
	f beta score (average=micro):  0.0848380244291025
	f beta score (average=weighted):  0.1448886000438275
	f beta score (average=None):  [0.23809524 0.12359551 0.06976744 0.22357724 0.12471655 0.2
 0.39473684 0.05841121 0.22908367 0.13363029 0.24048096 0.14344262
 0.12471655 0.08254717 0.2254902  0.06976744 0.03865979 0.06405439
 0.03144654 0.01930502]

compute the average Hamming loss
	hamming loss:  0.9151619755708975

jaccard similarity coefficient score
	jaccard score (average=macro):  0.0366580107580351
	jaccard score (average=None):  [0.059375   0.02806122 0.01511335 0.0556962  0.02835052 0.0479798
 0.11538462 0.01253133 0.05735661 0.03015075 0.06       0.0345679
 0.02791878 0.01767677 0.0575     0.015      0.00817439 0.05190311
 0.00645161 0.00396825]

confusion matrix:
[[ 19   0   0   0   1   0   0   1   0   0   0   3   0   0   0   1   0 294
    0   0]
 [  0  11   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0 377
    0   0]
 [  0   0   6   1   0   0   0   0   0   0   0   0   0   0   3   0   0 384
    0   0]
 [  0   1   2  22   0   0   0   0   0   0   0   0   0   0   0   0   0 367
    0   0]
 [  0   0   0   0  11   0   0   0   0   0   0   0   0   0   0   0   0 374
    0   0]
 [  0   0   0   1   0  19   0   0   0   0   0   0   0   0   0   0   0 375
    0   0]
 [  0   0   0   1   1   0  45   0   0   0   0   0   1   0   0   0   0 342
    0   0]
 [  0   0   0   0   0   1   0   5   1   0   0   0   0   0   0   0   0 389
    0   0]
 [  0   0   1   0   0   0   0   0  23   1   0   0   0   0   0   0   0 373
    0   0]
 [  0   0   0   0   0   0   0   0   0  12   0   0   0   0   0   0   0 385
    0   0]
 [  0   0   0   0   0   0   0   0   1   0  24   0   0   0   0   0   0 374
    0   0]
 [  0   0   0   0   0   0   0   0   0   0   1  14   0   0   0   0   1 379
    0   1]
 [  0   0   0   0   1   0   0   0   0   0   0   0  11   0   0   0   0 381
    0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   7   0   0   0 389
    0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0  23   0   0 371
    0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   6   0 392
    0   0]
 [  0   2   0   0   0   0   0   0   0   0   0   1   0   0   2   0   3 356
    0   0]
 [  0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0 375
    0   0]
 [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   2 306
    2   0]
 [  1   0   0   0   0   0   0   1   0   0   0   5   0   0   1   1   0 241
    0   1]]

================================================================================
Classifier.LINEAR_SVC
________________________________________________________________________________
Training: 
LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
          intercept_scaling=1, loss='squared_hinge', max_iter=1000,
          multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
          verbose=False)
train time: 0.811s
test time:  0.009s
accuracy:   0.698


cross validation:
	accuracy: 5-fold cross validation: [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]
	test accuracy: 5-fold cross validation accuracy: 0.76 (+/- 0.02)
dimensionality: 101321
density: 0.643472



===> Classification Report:

                          precision    recall  f1-score   support

             alt.atheism       0.54      0.47      0.50       319
           comp.graphics       0.65      0.73      0.69       389
 comp.os.ms-windows.misc       0.62      0.60      0.61       394
comp.sys.ibm.pc.hardware       0.66      0.68      0.67       392
   comp.sys.mac.hardware       0.73      0.69      0.71       385
          comp.windows.x       0.82      0.70      0.76       395
            misc.forsale       0.77      0.79      0.78       390
               rec.autos       0.75      0.72      0.74       396
         rec.motorcycles       0.79      0.76      0.77       398
      rec.sport.baseball       0.55      0.87      0.68       397
        rec.sport.hockey       0.89      0.86      0.88       399
               sci.crypt       0.83      0.72      0.77       396
         sci.electronics       0.65      0.58      0.61       393
                 sci.med       0.78      0.77      0.78       396
               sci.space       0.76      0.74      0.75       394
  soc.religion.christian       0.65      0.80      0.72       398
      talk.politics.guns       0.58      0.68      0.63       364
   talk.politics.mideast       0.83      0.77      0.80       376
      talk.politics.misc       0.58      0.47      0.52       310
      talk.religion.misc       0.45      0.29      0.36       251

                accuracy                           0.70      7532
               macro avg       0.69      0.69      0.69      7532
            weighted avg       0.70      0.70      0.70      7532



===> Classification Metrics:

accuracy classification score
	accuracy score:  0.6982209240573553
	accuracy score (normalize=False):  5259

compute the precision
	precision score (average=macro):  0.6949764430480109
	precision score (average=micro):  0.6982209240573553
	precision score (average=weighted):  0.7027147732766844
	precision score (average=None):  [0.53928571 0.6462585  0.62303665 0.6641604  0.73150685 0.81764706
 0.77192982 0.75461741 0.79419525 0.55183413 0.88917526 0.83333333
 0.64689266 0.7826087  0.75844156 0.65439673 0.57808858 0.82857143
 0.58232932 0.45121951]
	precision score (average=None, zero_division=1):  [0.53928571 0.6462585  0.62303665 0.6641604  0.73150685 0.81764706
 0.77192982 0.75461741 0.79419525 0.55183413 0.88917526 0.83333333
 0.64689266 0.7826087  0.75844156 0.65439673 0.57808858 0.82857143
 0.58232932 0.45121951]

compute the precision
	recall score (average=macro):  0.6861624476286413
	recall score (average=micro):  0.6982209240573553
	recall score (average=weighted):  0.6982209240573553
	recall score (average=None):  [0.47335423 0.73264781 0.60406091 0.67602041 0.69350649 0.70379747
 0.78974359 0.72222222 0.75628141 0.87153652 0.86466165 0.71969697
 0.5826972  0.77272727 0.74111675 0.8040201  0.68131868 0.7712766
 0.46774194 0.29482072]
	recall score (average=None, zero_division=1):  [0.47335423 0.73264781 0.60406091 0.67602041 0.69350649 0.70379747
 0.78974359 0.72222222 0.75628141 0.87153652 0.86466165 0.71969697
 0.5826972  0.77272727 0.74111675 0.8040201  0.68131868 0.7712766
 0.46774194 0.29482072]

compute the F1 score, also known as balanced F-score or F-measure
	f1 score (average=macro):  0.6861516388373394
	f1 score (average=micro):  0.6982209240573553
	f1 score (average=weighted):  0.6962497549115577
	f1 score (average=None):  [0.50417362 0.68674699 0.61340206 0.67003793 0.712      0.75646259
 0.78073511 0.73806452 0.77477477 0.67578125 0.87674714 0.77235772
 0.61311914 0.77763659 0.74967908 0.72153326 0.62547289 0.79889807
 0.51878354 0.35662651]

compute the F-beta score
	f beta score (average=macro):  0.6904384390358744
	f beta score (average=micro):  0.6982209240573553
	f beta score (average=weighted):  0.6992040519095292
	f beta score (average=None):  [0.52466991 0.66186716 0.61914672 0.66649899 0.72357724 0.79202279
 0.775428   0.74790795 0.78631139 0.59552496 0.88416197 0.80782313
 0.63294638 0.78061224 0.7549121  0.67969414 0.59615385 0.81644144
 0.55513017 0.40793826]

compute the average Hamming loss
	hamming loss:  0.3017790759426447

jaccard similarity coefficient score
	jaccard score (average=macro):  0.5337248386325865
	jaccard score (average=None):  [0.33705357 0.52293578 0.44237918 0.50380228 0.55279503 0.6083151
 0.64033264 0.58486708 0.63235294 0.51032448 0.78054299 0.62913907
 0.44208494 0.63617464 0.59958932 0.5643739  0.45504587 0.66513761
 0.35024155 0.2170088 ]

confusion matrix:
[[151   1   2   1   2   1   3   6   2  12   3   1   5   5   9  58  10  12
    8  27]
 [  3 285  21   6   7  22   6   2   1   8   0   8   7   0   9   2   0   0
    1   1]
 [  5  27 238  37  15  13   1   4   2  16   2   3   3   7   7   2   1   2
    6   3]
 [  0  14  35 265  22   6   9   3   0   8   2   5  20   0   2   0   0   0
    1   0]
 [  2  10   8  25 267   4  15   4   1  15   2   4  19   3   1   1   3   1
    0   0]
 [  1  42  36   6   3 278   1   1   1  10   0   2   3   0   4   0   2   2
    0   3]
 [  0   3   3  12  17   2 308   7   5  10   0   1   8   2   1   2   4   2
    2   1]
 [  4   2   1   2   3   2  13 286  20  27   1   1  12   2   6   2   5   2
    4   1]
 [  4   3   2   1   2   0   7  19 301  18   2   1   9   4   7   3   4   1
    8   2]
 [  3   2   0   1   0   1   5   2   5 346  14   0   2   3   2   4   1   2
    3   1]
 [  1   2   2   0   1   0   0   3   2  27 345   0   0   3   1   0   6   1
    2   3]
 [  3   8   5   7   3   0   5   2   4  19   2 285   9   1   6   5  18   4
    7   3]
 [  4  14  14  23  17   5  14  10   7  16   2  11 229  13   6   1   1   2
    3   1]
 [  4   6   3   1   1   0   2   7   5  15   6   0   8 306   7   4   6   5
    6   4]
 [  6   9   5   2   2   3   3   9   6  19   1   0  13   8 292   3   4   1
    6   2]
 [ 17   3   2   1   0   0   2   0   2  15   0   2   0   5   3 320   1   3
    7  15]
 [  5   4   3   2   1   0   2   6   7  13   0  11   0   8   7   8 248   8
   20  11]
 [ 24   0   1   3   0   1   2   2   5  11   1   1   1   3   2   7   9 290
   12   1]
 [ 10   1   0   0   0   1   0   3   3  13   3   3   3  10   9   1  87   7
  145  11]
 [ 33   5   1   4   2   1   1   3   0   9   2   3   3   8   4  66  19   5
    8  74]]

================================================================================
Classifier.LOGISTIC_REGRESSION
________________________________________________________________________________
Training: 
LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=-1, penalty='l2', random_state=0,
                   solver='lbfgs', tol=0.001, verbose=False, warm_start=False)
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
train time: 22.877s
test time:  0.011s
accuracy:   0.693


cross validation:
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
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
	accuracy: 5-fold cross validation: [0.75961114 0.74016792 0.75519222 0.7565179  0.73607427]
	test accuracy: 5-fold cross validation accuracy: 0.75 (+/- 0.02)
dimensionality: 101321
density: 1.000000



===> Classification Report:

                          precision    recall  f1-score   support

             alt.atheism       0.50      0.46      0.48       319
           comp.graphics       0.65      0.74      0.69       389
 comp.os.ms-windows.misc       0.62      0.60      0.61       394
comp.sys.ibm.pc.hardware       0.67      0.66      0.66       392
   comp.sys.mac.hardware       0.73      0.69      0.71       385
          comp.windows.x       0.84      0.71      0.77       395
            misc.forsale       0.78      0.77      0.78       390
               rec.autos       0.49      0.78      0.60       396
         rec.motorcycles       0.77      0.77      0.77       398
      rec.sport.baseball       0.83      0.81      0.82       397
        rec.sport.hockey       0.90      0.85      0.88       399
               sci.crypt       0.86      0.69      0.76       396
         sci.electronics       0.60      0.61      0.61       393
                 sci.med       0.79      0.78      0.78       396
               sci.space       0.74      0.74      0.74       394
  soc.religion.christian       0.66      0.79      0.72       398
      talk.politics.guns       0.58      0.69      0.63       364
   talk.politics.mideast       0.82      0.76      0.79       376
      talk.politics.misc       0.57      0.44      0.50       310
      talk.religion.misc       0.45      0.29      0.36       251

                accuracy                           0.69      7532
               macro avg       0.69      0.68      0.68      7532
            weighted avg       0.70      0.69      0.69      7532



===> Classification Metrics:

accuracy classification score
	accuracy score:  0.692777482740308
	accuracy score (normalize=False):  5218

compute the precision
	precision score (average=macro):  0.6922007701148025
	precision score (average=micro):  0.692777482740308
	precision score (average=weighted):  0.7004248801241143
	precision score (average=None):  [0.50171821 0.64559819 0.61842105 0.67101828 0.72876712 0.84036145
 0.78181818 0.48652932 0.76691729 0.82945736 0.90185676 0.85579937
 0.60353535 0.78680203 0.74234694 0.66384778 0.5800464  0.82183908
 0.5661157  0.45121951]
	precision score (average=None, zero_division=1):  [0.50171821 0.64559819 0.61842105 0.67101828 0.72876712 0.84036145
 0.78181818 0.48652932 0.76691729 0.82945736 0.90185676 0.85579937
 0.60353535 0.78680203 0.74234694 0.66384778 0.5800464  0.82183908
 0.5661157  0.45121951]

compute the precision
	recall score (average=macro):  0.6805438945547672
	recall score (average=micro):  0.692777482740308
	recall score (average=weighted):  0.692777482740308
	recall score (average=None):  [0.45768025 0.73521851 0.5964467  0.65561224 0.69090909 0.70632911
 0.77179487 0.77525253 0.76884422 0.80856423 0.85213033 0.68939394
 0.60814249 0.78282828 0.73857868 0.78894472 0.68681319 0.7606383
 0.44193548 0.29482072]
	recall score (average=None, zero_division=1):  [0.45768025 0.73521851 0.5964467  0.65561224 0.69090909 0.70632911
 0.77179487 0.77525253 0.76884422 0.80856423 0.85213033 0.68939394
 0.60814249 0.78282828 0.73857868 0.78894472 0.68681319 0.7606383
 0.44193548 0.29482072]

compute the F1 score, also known as balanced F-score or F-measure
	f1 score (average=macro):  0.6819466405630513
	f1 score (average=micro):  0.692777482740308
	f1 score (average=weighted):  0.6923856386302216
	f1 score (average=None):  [0.47868852 0.6875     0.60723514 0.66322581 0.70933333 0.76753783
 0.77677419 0.59785784 0.76787955 0.81887755 0.87628866 0.76363636
 0.60583016 0.78481013 0.74045802 0.72101033 0.62893082 0.79005525
 0.49637681 0.35662651]

compute the F-beta score
	f beta score (average=macro):  0.6870558354821853
	f beta score (average=micro):  0.692777482740308
	f beta score (average=weighted):  0.6962504900747963
	f beta score (average=None):  [0.49224545 0.66173068 0.6138976  0.66787942 0.72086721 0.80963436
 0.77979275 0.52568493 0.76730191 0.8251928  0.89145254 0.81638756
 0.60445119 0.78600406 0.74159021 0.68558952 0.598659   0.80882353
 0.53599374 0.40793826]

compute the average Hamming loss
	hamming loss:  0.30722251725969196

jaccard similarity coefficient score
	jaccard score (average=macro):  0.5304575850510889
	jaccard score (average=None):  [0.31465517 0.52380952 0.43599258 0.496139   0.54958678 0.62276786
 0.6350211  0.42638889 0.62321792 0.69330454 0.77981651 0.61764706
 0.43454545 0.64583333 0.58787879 0.56373429 0.4587156  0.65296804
 0.33012048 0.2170088 ]

confusion matrix:
[[146   0   4   1   1   1   2  14   5   5   2   2   3   7  11  54   9  13
   10  29]
 [  5 286  21   4   6  19   7   9   2   3   0   4  10   0  10   2   0   0
    0   1]
 [  3  25 235  40  18  15   1  18   2   1   1   4   3   8   8   1   2   2
    5   2]
 [  0  15  36 257  27   4  11  10   0   2   2   2  24   1   1   0   0   0
    0   0]
 [  2   9  10  27 266   1  14  22   0   2   2   3  21   4   2   0   0   0
    0   0]
 [  0  43  33   6   4 279   4   8   2   3   0   2   6   1   3   0   1   0
    0   0]
 [  1   2   4  15  15   1 301  19   6   1   1   1  10   2   2   3   3   2
    1   0]
 [  5   1   1   2   2   2  12 307  23   4   1   1  15   1   6   2   4   2
    3   2]
 [  3   4   0   0   1   0   6  36 306   4   1   0   9   4   6   3   5   1
    8   1]
 [  4   4   0   0   0   2   5  20   5 321  16   0   3   3   1   4   2   2
    4   1]
 [  3   2   0   0   0   0   0  12   6  20 340   0   1   3   0   0   6   2
    2   2]
 [  2  12   6   4   4   0   3  18   4   2   2 273  15   2   8   4  18   5
   12   2]
 [  3  13  13  23  14   4  10  24   7   2   1  10 239  12  10   2   0   3
    1   2]
 [  5   5   2   2   1   0   2  27   5   1   3   0   8 310   5   5   3   4
    5   3]
 [  7  10   6   0   3   1   2  25   6   2   1   1  16   7 291   1   6   1
    6   2]
 [ 19   4   2   0   0   0   1  14   3   2   0   3   2   3   4 314   0   3
    4  20]
 [  7   2   4   0   0   0   2  17   8   2   0   9   1   7   9   7 250   8
   20  11]
 [ 27   1   2   1   0   1   1   8   6   5   0   1   1   3   2   5   9 286
   15   2]
 [ 13   1   0   0   2   1   0  11   1   4   2   2   5   7   9   3  93   9
  137  10]
 [ 36   4   1   1   1   1   1  12   2   1   2   1   4   9   4  63  20   5
    9  74]]

================================================================================
Classifier.MULTINOMIAL_NB
________________________________________________________________________________
Training: 
MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True)
train time: 0.072s
test time:  0.012s
accuracy:   0.688


cross validation:
	accuracy: 5-fold cross validation: [0.76049492 0.74149359 0.75519222 0.75695979 0.74270557]
	test accuracy: 5-fold cross validation accuracy: 0.75 (+/- 0.02)
dimensionality: 101321
density: 1.000000



===> Classification Report:

                          precision    recall  f1-score   support

             alt.atheism       0.57      0.43      0.49       319
           comp.graphics       0.64      0.71      0.67       389
 comp.os.ms-windows.misc       0.74      0.43      0.54       394
comp.sys.ibm.pc.hardware       0.57      0.73      0.64       392
   comp.sys.mac.hardware       0.71      0.67      0.69       385
          comp.windows.x       0.77      0.75      0.76       395
            misc.forsale       0.81      0.71      0.76       390
               rec.autos       0.76      0.72      0.74       396
         rec.motorcycles       0.76      0.71      0.73       398
      rec.sport.baseball       0.91      0.80      0.85       397
        rec.sport.hockey       0.59      0.93      0.72       399
               sci.crypt       0.70      0.76      0.73       396
         sci.electronics       0.74      0.57      0.64       393
                 sci.med       0.83      0.76      0.80       396
               sci.space       0.73      0.79      0.76       394
  soc.religion.christian       0.57      0.86      0.69       398
      talk.politics.guns       0.56      0.72      0.63       364
   talk.politics.mideast       0.81      0.78      0.80       376
      talk.politics.misc       0.56      0.44      0.49       310
      talk.religion.misc       0.45      0.20      0.28       251

                accuracy                           0.69      7532
               macro avg       0.69      0.67      0.67      7532
            weighted avg       0.70      0.69      0.68      7532



===> Classification Metrics:

accuracy classification score
	accuracy score:  0.6878651088688263
	accuracy score (normalize=False):  5181

compute the precision
	precision score (average=macro):  0.6892132891624817
	precision score (average=micro):  0.6878651088688263
	precision score (average=weighted):  0.6966939677098184
	precision score (average=None):  [0.56846473 0.63678161 0.74449339 0.56886228 0.7107438  0.7734375
 0.81176471 0.75935829 0.76216216 0.91354467 0.58675079 0.69515012
 0.73684211 0.83379501 0.73130841 0.57237937 0.55863539 0.81043956
 0.56198347 0.44736842]
	precision score (average=None, zero_division=1):  [0.56846473 0.63678161 0.74449339 0.56886228 0.7107438  0.7734375
 0.81176471 0.75935829 0.76216216 0.91354467 0.58675079 0.69515012
 0.73684211 0.83379501 0.73130841 0.57237937 0.55863539 0.81043956
 0.56198347 0.44736842]

compute the precision
	recall score (average=macro):  0.6739472525504127
	recall score (average=micro):  0.6878651088688263
	recall score (average=weighted):  0.6878651088688263
	recall score (average=None):  [0.42946708 0.71208226 0.42893401 0.72704082 0.67012987 0.75189873
 0.70769231 0.71717172 0.70854271 0.79848866 0.93233083 0.76010101
 0.56997455 0.76010101 0.79441624 0.86432161 0.71978022 0.78457447
 0.43870968 0.20318725]
	recall score (average=None, zero_division=1):  [0.42946708 0.71208226 0.42893401 0.72704082 0.67012987 0.75189873
 0.70769231 0.71717172 0.70854271 0.79848866 0.93233083 0.76010101
 0.56997455 0.76010101 0.79441624 0.86432161 0.71978022 0.78457447
 0.43870968 0.20318725]

compute the F1 score, also known as balanced F-score or F-measure
	f1 score (average=macro):  0.6705056467222805
	f1 score (average=micro):  0.6878651088688263
	f1 score (average=weighted):  0.6817483476667928
	f1 score (average=None):  [0.48928571 0.6723301  0.54428341 0.63829787 0.68983957 0.76251605
 0.75616438 0.73766234 0.734375   0.85215054 0.72023233 0.72617612
 0.64275466 0.79524439 0.76155718 0.68868869 0.62905162 0.7972973
 0.49275362 0.27945205]

compute the F-beta score
	f beta score (average=macro):  0.6785983543022136
	f beta score (average=micro):  0.6878651088688263
	f beta score (average=weighted):  0.687865392413504
	f beta score (average=None):  [0.53390491 0.65054016 0.64900154 0.59474124 0.7022319  0.76903159
 0.78857143 0.75052854 0.75079872 0.88795518 0.63373083 0.70723684
 0.69608452 0.81793478 0.74311491 0.61384725 0.58482143 0.805131
 0.53208138 0.36067893]

compute the average Hamming loss
	hamming loss:  0.31213489113117365

jaccard similarity coefficient score
	jaccard score (average=macro):  0.5174155509251992
	jaccard score (average=None):  [0.32387707 0.50639854 0.37389381 0.46875    0.52653061 0.61618257
 0.60792952 0.58436214 0.58024691 0.74238876 0.56278366 0.57007576
 0.47357294 0.66008772 0.61493124 0.52519084 0.45884413 0.66292135
 0.32692308 0.16242038]

confusion matrix:
[[137   1   2   2   2   2   0   3   3   2  12   4   0   3   9  77  12  14
    9  25]
 [  1 277   6  18  15  27   5   0   5   3   6  15   0   0   8   1   0   1
    1   0]
 [  4  31 169  76  16  30   5   2   4   1  16  13   3   5  11   1   0   0
    4   3]
 [  0  14  21 285  28   3   9   4   0   1   8   5  14   0   0   0   0   0
    0   0]
 [  0   8   6  36 258   8   8   7   2   1  14   7  15   2   9   2   1   0
    1   0]
 [  0  49   8  11   4 297   1   0   0   1   7   5   3   4   4   0   1   0
    0   0]
 [  1   4   1  34  17   0 276  14   8   3  11   1   7   1   5   4   1   0
    2   0]
 [  1   1   1   1   2   0   9 284  26   1  25   4  12   1   8   3   6   3
    8   0]
 [  6   3   1   1   2   3   7  30 282   4  16   1   6   2   6   4  12   3
    8   1]
 [  6   2   0   1   0   1   5   0   3 317  34   4   1   3   2   5   6   1
    6   0]
 [  3   0   0   0   0   1   0   1   3   2 372   5   0   2   1   5   3   0
    1   0]
 [  3   9   6   4   4   2   0   0   3   2  18 301   3   1   6   4  18   3
    8   1]
 [  1  12   4  27  13   2  10  11   8   1  12  38 224  12   9   3   1   2
    3   0]
 [  4   5   0   1   0   0   2   6   4   0  15   1   6 301  10  18  11   5
    5   2]
 [  4   8   2   1   0   2   1   5   2   1  18   3   6   3 313   3   3   8
    9   2]
 [  7   3   0   1   1   3   0   0   1   1  14   0   0   2   1 344   5   0
    4  11]
 [  5   0   0   0   0   0   1   1   4   2  12  12   1   5   9  13 262   7
   16  14]
 [ 12   3   0   0   0   1   0   2   4   2   8   4   0   0   3  17  10 295
   14   1]
 [ 13   2   0   1   0   2   1   2   5   1   8   5   1   8   8   8  93  13
  136   3]
 [ 33   3   0   1   1   0   0   2   3   1   8   5   2   6   6  89  24   9
    7  51]]

================================================================================
Classifier.NEAREST_CENTROID
________________________________________________________________________________
Training: 
NearestCentroid(metric='cosine', shrink_threshold=None)
train time: 0.017s
test time:  0.019s
accuracy:   0.667


cross validation:
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
	accuracy: 5-fold cross validation: [0.71674768 0.71807335 0.71851525 0.71939903 0.70601238]
	test accuracy: 5-fold cross validation accuracy: 0.72 (+/- 0.01)


===> Classification Report:

                          precision    recall  f1-score   support

             alt.atheism       0.26      0.48      0.34       319
           comp.graphics       0.53      0.68      0.59       389
 comp.os.ms-windows.misc       0.64      0.60      0.62       394
comp.sys.ibm.pc.hardware       0.64      0.64      0.64       392
   comp.sys.mac.hardware       0.76      0.68      0.72       385
          comp.windows.x       0.85      0.70      0.76       395
            misc.forsale       0.78      0.78      0.78       390
               rec.autos       0.78      0.70      0.74       396
         rec.motorcycles       0.85      0.71      0.77       398
      rec.sport.baseball       0.90      0.79      0.84       397
        rec.sport.hockey       0.95      0.86      0.90       399
               sci.crypt       0.84      0.66      0.74       396
         sci.electronics       0.57      0.59      0.58       393
                 sci.med       0.90      0.56      0.69       396
               sci.space       0.72      0.73      0.73       394
  soc.religion.christian       0.62      0.79      0.69       398
      talk.politics.guns       0.55      0.69      0.62       364
   talk.politics.mideast       0.87      0.73      0.79       376
      talk.politics.misc       0.41      0.48      0.44       310
      talk.religion.misc       0.35      0.28      0.31       251

                accuracy                           0.67      7532
               macro avg       0.69      0.66      0.66      7532
            weighted avg       0.70      0.67      0.68      7532



===> Classification Metrics:

accuracy classification score
	accuracy score:  0.667020711630377
	accuracy score (normalize=False):  5024

compute the precision
	precision score (average=macro):  0.6873820977354106
	precision score (average=micro):  0.667020711630377
	precision score (average=weighted):  0.7017088668941149
	precision score (average=None):  [0.25838926 0.52683897 0.63709677 0.6377551  0.75872093 0.84615385
 0.77692308 0.77591036 0.84684685 0.89714286 0.94505495 0.84466019
 0.56617647 0.90283401 0.72474747 0.61886051 0.55384615 0.87220447
 0.40599455 0.35148515]
	precision score (average=None, zero_division=1):  [0.25838926 0.52683897 0.63709677 0.6377551  0.75872093 0.84615385
 0.77692308 0.77591036 0.84684685 0.89714286 0.94505495 0.84466019
 0.56617647 0.90283401 0.72474747 0.61886051 0.55384615 0.87220447
 0.40599455 0.35148515]

compute the precision
	recall score (average=macro):  0.656361029994913
	recall score (average=micro):  0.667020711630377
	recall score (average=weighted):  0.667020711630377
	recall score (average=None):  [0.48275862 0.68123393 0.60152284 0.6377551  0.67792208 0.69620253
 0.77692308 0.69949495 0.70854271 0.79093199 0.86215539 0.65909091
 0.58778626 0.56313131 0.7284264  0.79145729 0.69230769 0.72606383
 0.48064516 0.28286853]
	recall score (average=None, zero_division=1):  [0.48275862 0.68123393 0.60152284 0.6377551  0.67792208 0.69620253
 0.77692308 0.69949495 0.70854271 0.79093199 0.86215539 0.65909091
 0.58778626 0.56313131 0.7284264  0.79145729 0.69230769 0.72606383
 0.48064516 0.28286853]

compute the F1 score, also known as balanced F-score or F-measure
	f1 score (average=macro):  0.664367794074035
	f1 score (average=micro):  0.667020711630377
	f1 score (average=weighted):  0.6769874492455416
	f1 score (average=None):  [0.33661202 0.5941704  0.61879896 0.6377551  0.71604938 0.76388889
 0.77692308 0.73572377 0.77154583 0.84069612 0.9017038  0.74042553
 0.57677903 0.69362364 0.72658228 0.69459757 0.61538462 0.79245283
 0.44017725 0.31346578]

compute the F-beta score
	f beta score (average=macro):  0.6763187693213877
	f beta score (average=micro):  0.667020711630377
	f beta score (average=weighted):  0.6899571137735194
	f beta score (average=None):  [0.28486866 0.55185339 0.62964931 0.6377551  0.74105622 0.81120944
 0.77692308 0.75932018 0.8150289  0.87367835 0.92722372 0.79963235
 0.57037037 0.80563584 0.72548028 0.64708299 0.57692308 0.83845209
 0.41901012 0.33522191]

compute the average Hamming loss
	hamming loss:  0.33297928836962293

jaccard similarity coefficient score
	jaccard score (average=macro):  0.5151866623196801
	jaccard score (average=None):  [0.20236531 0.42264753 0.44801512 0.46816479 0.55769231 0.61797753
 0.63522013 0.58193277 0.62806236 0.72517321 0.82100239 0.58783784
 0.40526316 0.53095238 0.57057654 0.53209459 0.44444444 0.65625
 0.28219697 0.18586387]

confusion matrix:
[[154   4   2   1   0   3   0   2   2   1   2   2   3   3  12  60   7  10
    8  43]
 [ 16 265  26  10  12  16   5   2   2   2   0   3  13   0   9   0   1   0
    6   1]
 [ 24  32 237  35  14  15   1   2   1   0   0   3   3   3   8   2   3   0
    7   4]
 [  8  15  41 250  22   2   8   3   0   0   1   4  33   0   2   0   0   0
    3   0]
 [ 17  10   8  31 261   4  11   3   1   0   1   3  22   2   5   2   2   0
    1   1]
 [  8  53  29   9   2 275   1   1   1   1   0   2   2   0   3   1   2   0
    5   0]
 [  8   4   5  20  12   0 303   9   1   0   0   1  11   0   4   3   5   0
    2   2]
 [ 28   4   1   2   2   1   8 277  14   0   0   2  20   0   9   2   8   1
    9   8]
 [ 21   2   1   3   0   0   9  24 282   4   0   1  15   2   5   3   7   1
   17   1]
 [ 29   8   0   1   0   3   7   0   4 314  11   0   3   1   1   2   2   2
    8   1]
 [ 17   1   0   0   1   0   0   1   1  14 344   0   1   1   0   1   5   0
    6   6]
 [ 22  17   4   3   4   1   2   0   4   0   0 261  13   1   4   2  29   4
   17   8]
 [ 16  31   8  26  12   1  13  12   4   3   0  13 231   3   8   3   2   1
    4   2]
 [ 30  30   1   0   0   1  10  11   6   2   1   3  17 223  14  11   7   3
   22   4]
 [ 25  13   3   1   1   1   3   3   2   2   2   0  13   2 287   4   5   2
   23   2]
 [ 38   7   0   0   0   0   3   0   0   0   0   0   1   1   4 315   3   3
    9  14]
 [ 24   1   2   0   1   1   2   1   3   1   1   8   2   0   6   8 252   4
   32  15]
 [ 39   3   1   0   0   0   0   2   3   3   0   1   2   1   2   7   7 273
   26   6]
 [ 27   1   0   0   0   1   2   1   1   2   0   2   2   3   8   4  89   5
  149  13]
 [ 45   2   3   0   0   0   2   3   1   1   1   0   1   1   5  79  19   4
   13  71]]

================================================================================
Classifier.PASSIVE_AGGRESSIVE_CLASSIFIER
________________________________________________________________________________
Training: 
PassiveAggressiveClassifier(C=0.01, average=False, class_weight=None,
                            early_stopping=False, fit_intercept=True,
                            loss='hinge', max_iter=1000, n_iter_no_change=5,
                            n_jobs=-1, random_state=0, shuffle=True, tol=0.0001,
                            validation_fraction=0.0001, verbose=False,
                            warm_start=False)
train time: 2.319s
test time:  0.016s
accuracy:   0.696


cross validation:
	accuracy: 5-fold cross validation: [0.76535572 0.74856385 0.76182059 0.77065842 0.74889478]
	test accuracy: 5-fold cross validation accuracy: 0.76 (+/- 0.02)
dimensionality: 101321
density: 0.845597



===> Classification Report:

                          precision    recall  f1-score   support

             alt.atheism       0.52      0.48      0.50       319
           comp.graphics       0.66      0.74      0.70       389
 comp.os.ms-windows.misc       0.62      0.60      0.61       394
comp.sys.ibm.pc.hardware       0.64      0.65      0.65       392
   comp.sys.mac.hardware       0.73      0.69      0.71       385
          comp.windows.x       0.81      0.71      0.76       395
            misc.forsale       0.79      0.77      0.78       390
               rec.autos       0.79      0.71      0.75       396
         rec.motorcycles       0.52      0.81      0.63       398
      rec.sport.baseball       0.88      0.82      0.85       397
        rec.sport.hockey       0.90      0.88      0.89       399
               sci.crypt       0.82      0.72      0.77       396
         sci.electronics       0.63      0.58      0.60       393
                 sci.med       0.78      0.78      0.78       396
               sci.space       0.74      0.75      0.75       394
  soc.religion.christian       0.67      0.77      0.72       398
      talk.politics.guns       0.59      0.68      0.63       364
   talk.politics.mideast       0.82      0.77      0.79       376
      talk.politics.misc       0.56      0.46      0.51       310
      talk.religion.misc       0.41      0.32      0.36       251

                accuracy                           0.70      7532
               macro avg       0.69      0.68      0.69      7532
            weighted avg       0.70      0.70      0.70      7532



===> Classification Metrics:

accuracy classification score
	accuracy score:  0.6962294211364843
	accuracy score (normalize=False):  5244

compute the precision
	precision score (average=macro):  0.6933981414281503
	precision score (average=micro):  0.6962294211364843
	precision score (average=weighted):  0.7023355973378457
	precision score (average=None):  [0.51515152 0.66435185 0.61780105 0.64160401 0.72727273 0.81104651
 0.78684211 0.7877095  0.5184     0.8766756  0.9        0.82
 0.63231198 0.78371501 0.73880597 0.6681128  0.58711217 0.82102273
 0.56299213 0.40703518]
	precision score (average=None, zero_division=1):  [0.51515152 0.66435185 0.61780105 0.64160401 0.72727273 0.81104651
 0.78684211 0.7877095  0.5184     0.8766756  0.9        0.82
 0.63231198 0.78371501 0.73880597 0.6681128  0.58711217 0.82102273
 0.56299213 0.40703518]

compute the precision
	recall score (average=macro):  0.6846994007574254
	recall score (average=micro):  0.6962294211364843
	recall score (average=weighted):  0.6962294211364843
	recall score (average=None):  [0.47962382 0.7377892  0.59898477 0.65306122 0.68571429 0.70632911
 0.76666667 0.71212121 0.81407035 0.82367758 0.87969925 0.72474747
 0.57760814 0.77777778 0.75380711 0.77386935 0.67582418 0.76861702
 0.46129032 0.32270916]
	recall score (average=None, zero_division=1):  [0.47962382 0.7377892  0.59898477 0.65306122 0.68571429 0.70632911
 0.76666667 0.71212121 0.81407035 0.82367758 0.87969925 0.72474747
 0.57760814 0.77777778 0.75380711 0.77386935 0.67582418 0.76861702
 0.46129032 0.32270916]

compute the F1 score, also known as balanced F-score or F-measure
	f1 score (average=macro):  0.6858088315790827
	f1 score (average=micro):  0.6962294211364843
	f1 score (average=weighted):  0.696043076121644
	f1 score (average=None):  [0.49675325 0.69914738 0.60824742 0.64728192 0.70588235 0.75507442
 0.77662338 0.74801061 0.63343109 0.84935065 0.88973384 0.769437
 0.6037234  0.78073511 0.74623116 0.71711292 0.62835249 0.79395604
 0.5070922  0.36      ]

compute the F-beta score
	f beta score (average=macro):  0.6896839113084612
	f beta score (average=micro):  0.6962294211364843
	f beta score (average=weighted):  0.6991525746888037
	f beta score (average=None):  [0.50763106 0.67784601 0.61394381 0.64386318 0.71856287 0.78769057
 0.78272251 0.77133479 0.55900621 0.86553732 0.89586524 0.79899777
 0.62055768 0.78252033 0.74175824 0.68688671 0.60294118 0.80997758
 0.53921569 0.38681948]

compute the average Hamming loss
	hamming loss:  0.30377057886351566

jaccard similarity coefficient score
	jaccard score (average=macro):  0.5349247298000687
	jaccard score (average=None):  [0.33045356 0.53745318 0.43703704 0.47850467 0.54545455 0.60652174
 0.63481953 0.59745763 0.46351931 0.73814898 0.80136986 0.62527233
 0.43238095 0.64033264 0.59519038 0.55898367 0.45810056 0.65831435
 0.33966746 0.2195122 ]

confusion matrix:
[[153   1   4   1   1   2   2   4  12   2   3   3   6   6   9  47   9  13
    8  33]
 [  4 287  19   7   6  22   4   2   7   2   0  10   6   0   9   2   0   0
    1   1]
 [  7  22 236  42  12  13   2   3  16   0   3   2   5   7   9   2   1   2
    7   3]
 [  0  14  37 256  28   4  11   1   7   0   2   5  22   1   3   0   0   0
    1   0]
 [  3  11   8  26 264   4  14   4  15   1   0   5  17   2   2   1   6   1
    0   1]
 [  2  40  38   6   3 279   1   0   8   1   0   2   3   1   5   0   2   1
    0   3]
 [  1   3   3  13  15   1 299   7  15   2   0   1  11   3   4   2   4   2
    2   2]
 [  5   2   2   2   3   1  12 282  43   5   1   1  14   2   6   2   4   3
    5   1]
 [  4   3   1   1   2   0   4  15 324   2   1   0   9   4   7   3   4   1
   10   3]
 [  3   3   0   1   0   1   4   2  22 327  16   1   2   3   1   5   1   1
    3   1]
 [  0   1   1   1   2   0   1   2  12  12 351   0   0   2   1   1   6   1
    2   3]
 [  3   7   6   8   2   1   3   3  20   1   1 287   8   2   6   6  17   3
    8   4]
 [  4  11  13  21  18   8  14   9  19   3   1  13 227  13   9   1   1   2
    3   3]
 [  6   3   3   2   0   0   1   7  21   0   3   0   9 308   6   5   6   5
    6   5]
 [  5  11   4   1   2   3   2   5  21   2   2   3  11   7 297   2   4   1
    7   4]
 [ 20   4   2   1   0   0   2   0  17   0   0   1   1   4   3 308   1   3
    9  22]
 [  8   3   4   3   2   0   1   4  16   2   1  10   2   6   7   8 246   9
   19  13]
 [ 25   0   0   2   0   1   2   2  12   5   1   1   1   3   2   7   9 289
   13   1]
 [ 11   1   0   1   1   2   0   2  10   4   2   5   3  10  11   0  81   8
  143  15]
 [ 33   5   1   4   2   2   1   4   8   2   2   0   2   9   5  59  17   7
    7  81]]

================================================================================
Classifier.PERCEPTRON
________________________________________________________________________________
Training: 
Perceptron(alpha=0.0001, class_weight=None, early_stopping=True, eta0=1.0,
           fit_intercept=True, max_iter=100, n_iter_no_change=3, n_jobs=-1,
           penalty='l2', random_state=0, shuffle=True, tol=0.0001,
           validation_fraction=0.01, verbose=False, warm_start=False)
train time: 0.418s
test time:  0.017s
accuracy:   0.539


cross validation:
	accuracy: 5-fold cross validation: [0.63146266 0.61025188 0.58948299 0.60450729 0.60477454]
	test accuracy: 5-fold cross validation accuracy: 0.61 (+/- 0.03)
dimensionality: 101321
density: 0.146054



===> Classification Report:

                          precision    recall  f1-score   support

             alt.atheism       0.46      0.15      0.23       319
           comp.graphics       0.54      0.55      0.55       389
 comp.os.ms-windows.misc       0.61      0.44      0.51       394
comp.sys.ibm.pc.hardware       0.52      0.47      0.49       392
   comp.sys.mac.hardware       0.52      0.49      0.51       385
          comp.windows.x       0.74      0.54      0.62       395
            misc.forsale       0.61      0.73      0.67       390
               rec.autos       0.37      0.64      0.47       396
         rec.motorcycles       0.51      0.66      0.58       398
      rec.sport.baseball       0.65      0.66      0.65       397
        rec.sport.hockey       0.84      0.72      0.77       399
               sci.crypt       0.47      0.66      0.55       396
         sci.electronics       0.51      0.37      0.43       393
                 sci.med       0.53      0.57      0.55       396
               sci.space       0.78      0.51      0.61       394
  soc.religion.christian       0.41      0.76      0.53       398
      talk.politics.guns       0.49      0.47      0.48       364
   talk.politics.mideast       0.68      0.68      0.68       376
      talk.politics.misc       0.45      0.25      0.32       310
      talk.religion.misc       0.26      0.18      0.21       251

                accuracy                           0.54      7532
               macro avg       0.55      0.53      0.52      7532
            weighted avg       0.56      0.54      0.53      7532



===> Classification Metrics:

accuracy classification score
	accuracy score:  0.5386351566648965
	accuracy score (normalize=False):  4057

compute the precision
	precision score (average=macro):  0.547790606369934
	precision score (average=micro):  0.5386351566648965
	precision score (average=weighted):  0.5553553291131907
	precision score (average=None):  [0.46153846 0.5443038  0.60701754 0.52136752 0.51912568 0.73958333
 0.61422414 0.37205882 0.51277014 0.6525     0.83870968 0.46797153
 0.51048951 0.53427896 0.77734375 0.41025641 0.48857143 0.68181818
 0.44632768 0.25555556]
	precision score (average=None, zero_division=1):  [0.46153846 0.5443038  0.60701754 0.52136752 0.51912568 0.73958333
 0.61422414 0.37205882 0.51277014 0.6525     0.83870968 0.46797153
 0.51048951 0.53427896 0.77734375 0.41025641 0.48857143 0.68181818
 0.44632768 0.25555556]

compute the precision
	recall score (average=macro):  0.5251410773772462
	recall score (average=micro):  0.5386351566648965
	recall score (average=weighted):  0.5386351566648965
	recall score (average=None):  [0.15047022 0.55269923 0.43908629 0.46683673 0.49350649 0.53924051
 0.73076923 0.63888889 0.65577889 0.65743073 0.71679198 0.66414141
 0.37150127 0.57070707 0.50507614 0.7638191  0.46978022 0.67819149
 0.25483871 0.18326693]
	recall score (average=None, zero_division=1):  [0.15047022 0.55269923 0.43908629 0.46683673 0.49350649 0.53924051
 0.73076923 0.63888889 0.65577889 0.65743073 0.71679198 0.66414141
 0.37150127 0.57070707 0.50507614 0.7638191  0.46978022 0.67819149
 0.25483871 0.18326693]

compute the F1 score, also known as balanced F-score or F-measure
	f1 score (average=macro):  0.5211225976540017
	f1 score (average=micro):  0.5386351566648965
	f1 score (average=weighted):  0.5320884217363683
	f1 score (average=None):  [0.22695035 0.54846939 0.5095729  0.49259758 0.50599201 0.62371889
 0.66744731 0.47026022 0.5755237  0.65495609 0.77297297 0.54906054
 0.43004418 0.55189255 0.61230769 0.53380158 0.4789916  0.68
 0.32443532 0.21345708]

compute the F-beta score
	f beta score (average=macro):  0.5323605573706321
	f beta score (average=micro):  0.5386351566648965
	f beta score (average=weighted):  0.5416099612621674
	f beta score (average=None):  [0.32653061 0.54596242 0.56388527 0.50946548 0.51379124 0.68842922
 0.63446126 0.40596919 0.53615448 0.65348022 0.81111741 0.4973525
 0.4749512  0.54118774 0.70169252 0.45211184 0.48469388 0.68108974
 0.38801572 0.23686921]

compute the average Hamming loss
	hamming loss:  0.4613648433351036

jaccard similarity coefficient score
	jaccard score (average=macro):  0.36387822564431943
	jaccard score (average=None):  [0.128      0.37785589 0.34189723 0.32678571 0.33868093 0.45319149
 0.50087873 0.30741191 0.40402477 0.4869403  0.62995595 0.37841727
 0.2739212  0.38111298 0.44124169 0.36407186 0.31491713 0.51515152
 0.19362745 0.11948052]

confusion matrix:
[[ 48   9   2   3   5   4   3  15  19  16   3  10   3  14   6  99   8  15
   11  26]
 [  0 215  15  18  18  27  11  14   7  10   1  19   7  10   6   4   1   3
    0   3]
 [  3  20 173  40  15  13  12  27   7   7   1  20   4  15   7  10   4   7
    6   3]
 [  0  18  29 183  23   5  22  24   6   4   2  14  17  13   0  15   4   5
    4   4]
 [  1  12   8  37 190   6  25  26   8   7   0  21  20   8   0   3   5   5
    3   0]
 [  0  31  29   9   9 213  13  19   6   5   3  16  10   7   6   9   4   5
    0   1]
 [  1   4   4   9  16   1 285  19  10   1   1   3  13   3   2  10   5   2
    0   1]
 [  1   4   1   2   9   1  18 253  34   7   2  17   9  12   1   7   8   4
    4   2]
 [  4  10   2   4   5   0   8  37 261   5   5   9   7   6   2   7   8   5
   10   3]
 [  2   2   0   2   8   1   9  32  19 261  16  10   8   6   4  11   2   1
    2   1]
 [  3   6   0   0   4   2   1  14  17  22 286   9   1   4   3  12   5   4
    5   1]
 [  0   9   5   7   7   1   4  26   8   8   1 263  11  11   2   8  11   4
    7   3]
 [  3  21   9  14  16   7  22  24  23   6   2  47 146  14   1  18  10   3
    3   4]
 [  2   4   1   7  11   1   6  33  19   6   1  13   7 226   5  29   8   5
    9   3]
 [  2  12   1   5  13   4   8  28  15   5   1  17  15  29 199  15   7   3
    2  13]
 [ 13   4   0   2   1   0   6  21   4   1   0   5   1   4   0 304   0   6
    6  20]
 [  5   6   2   4   6   1   3  23  20   9   5  23   1   7   5  23 171  16
   15  19]
 [  4   2   3   0   3   1   3  16   9   5   1  12   0   9   1  29   6 255
    8   9]
 [  2   3   1   3   1   0   0  17   9  10   5  25   2  13   3  35  64  20
   79  18]
 [ 10   3   0   2   6   0   5  12   8   5   5   9   4  12   3  93  19   6
    3  46]]

================================================================================
Classifier.RANDOM_FOREST_CLASSIFIER
________________________________________________________________________________
Training: 
RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=5,
                       min_weight_fraction_leaf=0.0, n_estimators=200,
                       n_jobs=-1, oob_score=False, random_state=0,
                       verbose=False, warm_start=False)
train time: 7.790s
test time:  0.307s
accuracy:   0.637


cross validation:
	accuracy: 5-fold cross validation: [0.69465312 0.66902342 0.68272205 0.69730446 0.67462423]
	test accuracy: 5-fold cross validation accuracy: 0.68 (+/- 0.02)


===> Classification Report:

                          precision    recall  f1-score   support

             alt.atheism       0.46      0.38      0.42       319
           comp.graphics       0.61      0.63      0.62       389
 comp.os.ms-windows.misc       0.58      0.66      0.62       394
comp.sys.ibm.pc.hardware       0.60      0.59      0.59       392
   comp.sys.mac.hardware       0.67      0.65      0.66       385
          comp.windows.x       0.71      0.69      0.70       395
            misc.forsale       0.68      0.74      0.71       390
               rec.autos       0.43      0.71      0.53       396
         rec.motorcycles       0.71      0.71      0.71       398
      rec.sport.baseball       0.71      0.78      0.74       397
        rec.sport.hockey       0.83      0.85      0.84       399
               sci.crypt       0.81      0.67      0.73       396
         sci.electronics       0.55      0.45      0.49       393
                 sci.med       0.73      0.65      0.69       396
               sci.space       0.68      0.68      0.68       394
  soc.religion.christian       0.58      0.81      0.68       398
      talk.politics.guns       0.54      0.62      0.58       364
   talk.politics.mideast       0.86      0.69      0.77       376
      talk.politics.misc       0.53      0.36      0.43       310
      talk.religion.misc       0.45      0.12      0.19       251

                accuracy                           0.64      7532
               macro avg       0.64      0.62      0.62      7532
            weighted avg       0.64      0.64      0.63      7532



===> Classification Metrics:

accuracy classification score
	accuracy score:  0.6371481678173128
	accuracy score (normalize=False):  4799

compute the precision
	precision score (average=macro):  0.6359307397441603
	precision score (average=micro):  0.6371481678173128
	precision score (average=weighted):  0.6424446791797502
	precision score (average=None):  [0.46387833 0.60891089 0.57615894 0.59895833 0.6657754  0.71465969
 0.67681499 0.42727273 0.71139241 0.71198157 0.83251232 0.80851064
 0.546875   0.73163842 0.68286445 0.58227848 0.53809524 0.86423841
 0.52803738 0.44776119]
	precision score (average=None, zero_division=1):  [0.46387833 0.60891089 0.57615894 0.59895833 0.6657754  0.71465969
 0.67681499 0.42727273 0.71139241 0.71198157 0.83251232 0.80851064
 0.546875   0.73163842 0.68286445 0.58227848 0.53809524 0.86423841
 0.52803738 0.44776119]

compute the precision
	recall score (average=macro):  0.6221679220802712
	recall score (average=micro):  0.6371481678173128
	recall score (average=weighted):  0.6371481678173128
	recall score (average=None):  [0.38244514 0.63239075 0.66243655 0.58673469 0.64675325 0.69113924
 0.74102564 0.71212121 0.70603015 0.77833753 0.84711779 0.67171717
 0.44529262 0.6540404  0.67766497 0.80904523 0.62087912 0.69414894
 0.36451613 0.11952191]
	recall score (average=None, zero_division=1):  [0.38244514 0.63239075 0.66243655 0.58673469 0.64675325 0.69113924
 0.74102564 0.71212121 0.70603015 0.77833753 0.84711779 0.67171717
 0.44529262 0.6540404  0.67766497 0.80904523 0.62087912 0.69414894
 0.36451613 0.11952191]

compute the F1 score, also known as balanced F-score or F-measure
	f1 score (average=macro):  0.6190234798433347
	f1 score (average=micro):  0.6371481678173128
	f1 score (average=weighted):  0.6313298990881601
	f1 score (average=None):  [0.41924399 0.62042875 0.6162928  0.59278351 0.65612648 0.7027027
 0.70746634 0.53409091 0.70870113 0.74368231 0.83975155 0.7337931
 0.49088359 0.69066667 0.68025478 0.67718191 0.57653061 0.7699115
 0.43129771 0.18867925]

compute the F-beta score
	f beta score (average=macro):  0.6252442258023045
	f beta score (average=micro):  0.6371481678173129
	f beta score (average=weighted):  0.6350096229019236
	f beta score (average=None):  [0.44493071 0.61346633 0.59156845 0.59647303 0.66188198 0.70982839
 0.68875119 0.46442688 0.71031345 0.72433193 0.83539298 0.77686916
 0.52301255 0.71467991 0.68181818 0.61685824 0.55283757 0.82386364
 0.48456261 0.28901734]

compute the average Hamming loss
	hamming loss:  0.3628518321826872

jaccard similarity coefficient score
	jaccard score (average=macro):  0.46287043286978813
	jaccard score (average=None):  [0.26521739 0.44972578 0.44539249 0.42124542 0.48823529 0.54166667
 0.54734848 0.36434109 0.54882812 0.59195402 0.72376874 0.5795207
 0.32527881 0.52749491 0.51544402 0.51192369 0.40501792 0.62589928
 0.27493917 0.10416667]

confusion matrix:
[[122   2   2   3   2   2   9  18   6   5   7   3   2  10  17  79   9   8
    6   7]
 [  3 246  31  13  11  34   6  12   4   1   1   2  10   2  12   1   0   0
    0   0]
 [  3  21 261  29  17  18   0  16   4   3   2   2   1   2   8   0   3   1
    3   0]
 [  0  13  49 230  26   6  16  11   1   4   2   1  30   1   2   0   0   0
    0   0]
 [  1   8  12  32 249   6  13  22   4   5   3   3  18   3   5   0   1   0
    0   0]
 [  2  30  38   7   7 273   6  11   1   3   2   3   2   1   7   0   1   0
    1   0]
 [  0   6   4  24  18   1 289  18   1   3   2   1   7   2   6   1   5   0
    2   0]
 [  4   6   6   4   3   3  17 282  23   5   1   1  16   4   6   2   8   1
    3   1]
 [  3   1   4   3   1   1  11  44 281  14   1   2   7   5   3   5   5   0
    7   0]
 [  3   5   2   2   0   2   2  25   9 309  25   2   0   1   1   2   0   1
    5   1]
 [  0   2   2   0   0   1   1  14   5  21 338   1   1   2   3   2   1   1
    3   1]
 [  4   8   8   4   7   4   6  19   5   2   1 266  12   2   7   1  24   4
   10   2]
 [  2  24  11  27  23  13  13  32   9  10   6  19 175  12   9   1   2   2
    2   1]
 [  7  13   7   1   3   3  17  31   6   9   3   0  11 259   7   4   4   2
    7   2]
 [  5   9   3   2   3   4   6  29   9   8   4   2  12  10 267   5   6   0
   10   0]
 [ 14   2   4   0   0   2   4  17   7   1   0   0   1   1   6 322   1   4
    5   7]
 [  9   5   4   0   2   5   1  19   6   8   1  14   6   8   6  12 226   4
   18  10]
 [ 26   0   0   0   1   4   4  10   7  12   1   3   2   4   6  12  10 261
   12   1]
 [ 16   1   0   2   0   0   5  14   5   9   4   3   6  17   8   8  89   6
  113   4]
 [ 39   2   5   1   1   0   1  16   2   2   2   1   1   8   5  96  25   7
    7  30]]

================================================================================
Classifier.RIDGE_CLASSIFIER
________________________________________________________________________________
Training: 
RidgeClassifier(alpha=0.5, class_weight=None, copy_X=True, fit_intercept=True,
                max_iter=None, normalize=False, random_state=0, solver='auto',
                tol=0.001)
train time: 3.120s
test time:  0.023s
accuracy:   0.700


cross validation:
	accuracy: 5-fold cross validation: [0.76756518 0.74502872 0.76800707 0.776403   0.76083112]
	test accuracy: 5-fold cross validation accuracy: 0.76 (+/- 0.02)
dimensionality: 101321
density: 1.000000



===> Classification Report:

                          precision    recall  f1-score   support

             alt.atheism       0.53      0.49      0.51       319
           comp.graphics       0.65      0.74      0.69       389
 comp.os.ms-windows.misc       0.64      0.63      0.64       394
comp.sys.ibm.pc.hardware       0.66      0.68      0.67       392
   comp.sys.mac.hardware       0.74      0.70      0.72       385
          comp.windows.x       0.82      0.72      0.76       395
            misc.forsale       0.77      0.78      0.77       390
               rec.autos       0.75      0.71      0.73       396
         rec.motorcycles       0.80      0.75      0.77       398
      rec.sport.baseball       0.56      0.87      0.68       397
        rec.sport.hockey       0.89      0.87      0.88       399
               sci.crypt       0.83      0.71      0.77       396
         sci.electronics       0.64      0.58      0.61       393
                 sci.med       0.80      0.78      0.79       396
               sci.space       0.75      0.75      0.75       394
  soc.religion.christian       0.64      0.80      0.71       398
      talk.politics.guns       0.59      0.70      0.64       364
   talk.politics.mideast       0.86      0.77      0.81       376
      talk.politics.misc       0.56      0.46      0.51       310
      talk.religion.misc       0.45      0.26      0.33       251

                accuracy                           0.70      7532
               macro avg       0.70      0.69      0.69      7532
            weighted avg       0.70      0.70      0.70      7532



===> Classification Metrics:

accuracy classification score
	accuracy score:  0.7002124269782263
	accuracy score (normalize=False):  5274

compute the precision
	precision score (average=macro):  0.6965231139871264
	precision score (average=micro):  0.7002124269782263
	precision score (average=weighted):  0.7045273971902621
	precision score (average=None):  [0.52901024 0.65227273 0.64248705 0.65925926 0.74033149 0.81609195
 0.76767677 0.75466667 0.79946524 0.55537721 0.88946015 0.82748538
 0.64225352 0.79792746 0.75126904 0.64128257 0.59440559 0.85756677
 0.56078431 0.45138889]
	precision score (average=None, zero_division=1):  [0.52901024 0.65227273 0.64248705 0.65925926 0.74033149 0.81609195
 0.76767677 0.75466667 0.79946524 0.55537721 0.88946015 0.82748538
 0.64225352 0.79792746 0.75126904 0.64128257 0.59440559 0.85756677
 0.56078431 0.45138889]

compute the precision
	recall score (average=macro):  0.6875359644275982
	recall score (average=micro):  0.7002124269782263
	recall score (average=weighted):  0.7002124269782263
	recall score (average=None):  [0.48589342 0.7377892  0.62944162 0.68112245 0.6961039  0.71898734
 0.77948718 0.71464646 0.75125628 0.87153652 0.86716792 0.71464646
 0.58015267 0.77777778 0.75126904 0.8040201  0.70054945 0.76861702
 0.46129032 0.25896414]
	recall score (average=None, zero_division=1):  [0.48589342 0.7377892  0.62944162 0.68112245 0.6961039  0.71898734
 0.77948718 0.71464646 0.75125628 0.87153652 0.86716792 0.71464646
 0.58015267 0.77777778 0.75126904 0.8040201  0.70054945 0.76861702
 0.46129032 0.25896414]

compute the F1 score, also known as balanced F-score or F-measure
	f1 score (average=macro):  0.687192806549338
	f1 score (average=micro):  0.7002124269782263
	f1 score (average=weighted):  0.6978726408253572
	f1 score (average=None):  [0.50653595 0.69240048 0.63589744 0.67001255 0.71753681 0.76446837
 0.7735369  0.73411154 0.7746114  0.67843137 0.87817259 0.76693767
 0.60962567 0.78772379 0.75126904 0.71348941 0.64312736 0.81065919
 0.50619469 0.32911392]

compute the F-beta score
	f beta score (average=macro):  0.6916021850659753
	f beta score (average=micro):  0.7002124269782263
	f beta score (average=weighted):  0.7008213984997628
	f beta score (average=None):  [0.51978538 0.66775244 0.63983488 0.66351889 0.73104201 0.79462787
 0.77001013 0.74630802 0.78933474 0.59882312 0.88491049 0.8021542
 0.62879206 0.79381443 0.75126904 0.66833751 0.61298077 0.83816705
 0.53759398 0.3929867 ]

compute the average Hamming loss
	hamming loss:  0.2997875730217738

jaccard similarity coefficient score
	jaccard score (average=macro):  0.5356820727344597
	jaccard score (average=None):  [0.33916849 0.5295203  0.46616541 0.50377358 0.55949896 0.61873638
 0.63070539 0.57991803 0.63213531 0.51335312 0.78280543 0.62197802
 0.43846154 0.64978903 0.60162602 0.55459272 0.4739777  0.68160377
 0.33886256 0.1969697 ]

confusion matrix:
[[155   1   3   1   2   1   5   6   2  13   3   2   6   5  10  54   7   9
    8  26]
 [  4 287  16   7   7  24   4   1   2   6   0   9   6   3   7   2   1   1
    1   1]
 [  3  23 248  36  13  15   3   3   2  17   3   3   4   2   6   2   1   2
    6   2]
 [  0  15  33 267  22   6  10   3   0   8   1   4  19   0   2   1   0   0
    1   0]
 [  2   9   8  27 268   5  13   5   1  15   1   5  16   2   3   1   3   1
    0   0]
 [  1  43  33   5   3 284   2   0   1   8   0   2   4   1   5   0   1   1
    0   1]
 [  0   2   3  14  16   0 304   7   5  10   0   1  12   2   2   3   4   3
    2   0]
 [  4   1   3   2   3   1  13 283  17  28   1   2  17   2   4   1   5   3
    5   1]
 [  5   3   2   3   2   0   5  20 299  17   4   0   7   6   7   3   4   1
    8   2]
 [  3   2   0   2   0   1   4   2   6 346  14   1   2   3   0   4   1   1
    5   0]
 [  1   2   2   0   2   0   0   2   3  26 346   0   0   4   1   1   5   0
    2   2]
 [  4   8   6   7   4   0   5   2   3  18   2 283   8   1   7   5  17   2
   11   3]
 [  7  16  17  23  13   4  13  11   6  15   2  11 228  12   9   1   1   1
    2   1]
 [  3   7   2   2   1   0   2   8   7  15   7   0   6 308   7   4   6   5
    4   2]
 [  6  10   3   1   2   2   4   7   6  19   1   2  11   7 296   1   4   1
    8   3]
 [ 17   3   2   1   0   1   2   0   2  15   0   1   0   5   3 320   1   4
    9  12]
 [  7   3   3   2   1   0   2   6   4  13   0   7   2   7  10   8 255   4
   19  11]
 [ 24   0   1   3   0   0   2   2   5  13   1   2   1   2   2   9   6 289
   12   2]
 [ 13   0   0   0   1   2   1   3   3  13   3   5   3   8   9   3  86   4
  143  10]
 [ 34   5   1   2   2   2   2   4   0   8   0   2   3   6   4  76  21   5
    9  65]]

================================================================================
Classifier.MAJORITY_VOTING_CLASSIFIER
________________________________________________________________________________
Training: 
VotingClassifier(estimators=[('complement_nb',
                              ComplementNB(alpha=0.5, class_prior=None,
                                           fit_prior=False, norm=False)),
                             ('linear_svc',
                              LinearSVC(C=1.0, class_weight=None, dual=True,
                                        fit_intercept=True, intercept_scaling=1,
                                        loss='squared_hinge', max_iter=1000,
                                        multi_class='ovr', penalty='l2',
                                        random_state=0, tol=0.0001,
                                        verbose=False)),
                             ('logistic_regression',
                              Logisti...
                                                     min_samples_split=5,
                                                     min_weight_fraction_leaf=0.0,
                                                     n_estimators=200,
                                                     n_jobs=-1, oob_score=False,
                                                     random_state=0,
                                                     verbose=False,
                                                     warm_start=False)),
                             ('ridge_classifier',
                              RidgeClassifier(alpha=0.5, class_weight=None,
                                              copy_X=True, fit_intercept=True,
                                              max_iter=None, normalize=False,
                                              random_state=0, solver='auto',
                                              tol=0.001))],
                 flatten_transform=True, n_jobs=-1, voting='hard',
                 weights=None)
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
train time: 31.059s
test time:  0.418s
accuracy:   0.704


cross validation:
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
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
	accuracy: 5-fold cross validation: [0.76712329 0.75077331 0.76800707 0.77551922 0.7515473 ]
	test accuracy: 5-fold cross validation accuracy: 0.76 (+/- 0.02)


===> Classification Report:

                          precision    recall  f1-score   support

             alt.atheism       0.52      0.47      0.49       319
           comp.graphics       0.65      0.75      0.70       389
 comp.os.ms-windows.misc       0.63      0.62      0.63       394
comp.sys.ibm.pc.hardware       0.66      0.68      0.67       392
   comp.sys.mac.hardware       0.73      0.70      0.71       385
          comp.windows.x       0.83      0.71      0.76       395
            misc.forsale       0.78      0.79      0.78       390
               rec.autos       0.50      0.78      0.61       396
         rec.motorcycles       0.79      0.77      0.78       398
      rec.sport.baseball       0.84      0.84      0.84       397
        rec.sport.hockey       0.90      0.87      0.89       399
               sci.crypt       0.84      0.73      0.78       396
         sci.electronics       0.66      0.58      0.62       393
                 sci.med       0.79      0.79      0.79       396
               sci.space       0.76      0.75      0.76       394
  soc.religion.christian       0.65      0.81      0.72       398
      talk.politics.guns       0.59      0.70      0.64       364
   talk.politics.mideast       0.84      0.77      0.80       376
      talk.politics.misc       0.61      0.46      0.52       310
      talk.religion.misc       0.47      0.25      0.33       251

                accuracy                           0.70      7532
               macro avg       0.70      0.69      0.69      7532
            weighted avg       0.71      0.70      0.70      7532



===> Classification Metrics:

accuracy classification score
	accuracy score:  0.7036643653744026
	accuracy score (normalize=False):  5300

compute the precision
	precision score (average=macro):  0.70142419223004
	precision score (average=micro):  0.7036643653744026
	precision score (average=weighted):  0.7091020761445109
	precision score (average=None):  [0.51535836 0.64955357 0.63471503 0.66253102 0.73224044 0.8259587
 0.77525253 0.50162866 0.79220779 0.84263959 0.89717224 0.83526012
 0.65994236 0.78734177 0.7642487  0.6498994  0.58796296 0.84057971
 0.60683761 0.46715328]
	precision score (average=None, zero_division=1):  [0.51535836 0.64955357 0.63471503 0.66253102 0.73224044 0.8259587
 0.77525253 0.50162866 0.79220779 0.84263959 0.89717224 0.83526012
 0.65994236 0.78734177 0.7642487  0.6498994  0.58796296 0.84057971
 0.60683761 0.46715328]

compute the precision
	recall score (average=macro):  0.6905924633217626
	recall score (average=micro):  0.7036643653744026
	recall score (average=weighted):  0.7036643653744026
	recall score (average=None):  [0.47335423 0.74807198 0.62182741 0.68112245 0.6961039  0.70886076
 0.78717949 0.77777778 0.76633166 0.83627204 0.87468672 0.72979798
 0.5826972  0.78535354 0.74873096 0.81155779 0.6978022  0.7712766
 0.45806452 0.25498008]
	recall score (average=None, zero_division=1):  [0.47335423 0.74807198 0.62182741 0.68112245 0.6961039  0.70886076
 0.78717949 0.77777778 0.76633166 0.83627204 0.87468672 0.72979798
 0.5826972  0.78535354 0.74873096 0.81155779 0.6978022  0.7712766
 0.45806452 0.25498008]

compute the F1 score, also known as balanced F-score or F-measure
	f1 score (average=macro):  0.6908873272390695
	f1 score (average=micro):  0.7036643653744026
	f1 score (average=weighted):  0.7017532405600586
	f1 score (average=None):  [0.49346405 0.6953405  0.62820513 0.67169811 0.71371505 0.76294278
 0.78117048 0.60990099 0.77905492 0.83944374 0.8857868  0.77897574
 0.61891892 0.7863464  0.75641026 0.72178771 0.63819095 0.80443828
 0.52205882 0.32989691]

compute the F-beta score
	f beta score (average=macro):  0.6958585389874006
	f beta score (average=micro):  0.7036643653744026
	f beta score (average=weighted):  0.7050119961857817
	f beta score (average=None):  [0.50637156 0.66712517 0.63209494 0.66616766 0.72471606 0.79954312
 0.77760892 0.53997195 0.7868937  0.84135834 0.89258312 0.81179775
 0.64289725 0.78694332 0.76109391 0.67686505 0.60707457 0.82574032
 0.56982343 0.40050063]

compute the average Hamming loss
	hamming loss:  0.2963356346255975

jaccard similarity coefficient score
	jaccard score (average=macro):  0.5415500097478312
	jaccard score (average=None):  [0.32754881 0.53296703 0.45794393 0.50568182 0.55486542 0.61674009
 0.64091858 0.43874644 0.63807531 0.72331155 0.79498861 0.63796909
 0.4481409  0.64791667 0.60824742 0.56468531 0.46863469 0.67285383
 0.35323383 0.19753086]

confusion matrix:
[[151   0   2   1   2   2   3  14   2   4   2   1   4   6  11  60   9  12
    8  25]
 [  5 291  18   5   6  22   6   8   1   2   0   7   6   0   9   2   0   0
    0   1]
 [  5  25 245  37  15  12   2  18   2   1   2   3   4   6   6   1   2   2
    4   2]
 [  0  16  35 267  22   6   8  10   0   1   1   5  19   0   2   0   0   0
    0   0]
 [  2  11   8  27 268   3  15  20   1   1   1   5  16   2   1   0   3   1
    0   0]
 [  1  44  35   6   4 280   2   7   1   2   0   2   2   1   5   0   1   1
    0   1]
 [  0   2   4  14  16   1 307  16   5   2   0   1   9   2   1   2   4   2
    2   0]
 [  4   2   2   2   3   1  14 308  21   3   1   1  15   1   6   1   4   3
    4   0]
 [  4   3   1   1   2   0   5  33 305   4   2   0   8   4   7   3   4   2
    8   2]
 [  3   3   0   1   0   1   4  19   5 332  15   0   1   3   0   4   1   1
    3   1]
 [  1   1   1   0   1   0   0  12   2  18 349   0   0   3   1   0   5   1
    2   2]
 [  4   9   6   8   4   0   3  18   4   2   2 289   7   1   6   5  17   2
    8   1]
 [  4  13  16  23  16   5  14  22   8   4   1  13 229  14   5   1   1   2
    1   1]
 [  4   4   2   1   1   0   3  25   6   0   5   0   7 311   5   4   6   4
    6   2]
 [  6  11   4   1   2   2   3  24   6   2   2   1  13   7 295   1   4   1
    6   3]
 [ 20   4   2   1   0   0   2  14   2   1   0   1   0   5   3 323   1   2
    4  13]
 [  6   3   3   2   1   0   2  16   6   2   0  11   0   9   8   8 254   7
   17   9]
 [ 25   0   1   3   0   1   2   8   6   6   0   1   1   3   2   7   8 290
   11   1]
 [ 12   1   0   0   1   1   0  11   2   5   4   4   3   9   9   1  90   6
  142   9]
 [ 36   5   1   3   2   2   1  11   0   2   2   1   3   8   4  74  18   6
    8  64]]

================================================================================
Classifier.SOFT_VOTING_CLASSIFIER
________________________________________________________________________________
Training: 
VotingClassifier(estimators=[('complement_nb',
                              ComplementNB(alpha=0.5, class_prior=None,
                                           fit_prior=False, norm=False)),
                             ('logistic_regression',
                              LogisticRegression(C=10, class_weight=None,
                                                 dual=False, fit_intercept=True,
                                                 intercept_scaling=1,
                                                 l1_ratio=None, max_iter=100,
                                                 multi_class='auto', n_jobs=-1,
                                                 penalty='l2', random_state=0,
                                                 solver='lbfgs', tol=0.001,
                                                 verbose=Fal...
                                                     criterion='gini',
                                                     max_depth=None,
                                                     max_features='auto',
                                                     max_leaf_nodes=None,
                                                     max_samples=None,
                                                     min_impurity_decrease=0.0,
                                                     min_impurity_split=None,
                                                     min_samples_leaf=1,
                                                     min_samples_split=5,
                                                     min_weight_fraction_leaf=0.0,
                                                     n_estimators=200,
                                                     n_jobs=-1, oob_score=False,
                                                     random_state=0,
                                                     verbose=False,
                                                     warm_start=False))],
                 flatten_transform=True, n_jobs=-1, voting='soft',
                 weights=None)
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
train time: 28.071s
test time:  0.353s
accuracy:   0.717


cross validation:
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
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
	accuracy: 5-fold cross validation: [0.79231109 0.75828546 0.79098542 0.7870084  0.76702034]
	test accuracy: 5-fold cross validation accuracy: 0.78 (+/- 0.03)


===> Classification Report:

                          precision    recall  f1-score   support

             alt.atheism       0.56      0.45      0.50       319
           comp.graphics       0.65      0.73      0.69       389
 comp.os.ms-windows.misc       0.73      0.55      0.63       394
comp.sys.ibm.pc.hardware       0.65      0.71      0.68       392
   comp.sys.mac.hardware       0.74      0.73      0.73       385
          comp.windows.x       0.82      0.77      0.79       395
            misc.forsale       0.81      0.78      0.80       390
               rec.autos       0.53      0.82      0.64       396
         rec.motorcycles       0.81      0.78      0.79       398
      rec.sport.baseball       0.89      0.83      0.86       397
        rec.sport.hockey       0.91      0.91      0.91       399
               sci.crypt       0.78      0.76      0.77       396
         sci.electronics       0.70      0.60      0.64       393
                 sci.med       0.84      0.79      0.81       396
               sci.space       0.78      0.80      0.79       394
  soc.religion.christian       0.61      0.88      0.72       398
      talk.politics.guns       0.58      0.73      0.64       364
   talk.politics.mideast       0.83      0.79      0.81       376
      talk.politics.misc       0.59      0.43      0.50       310
      talk.religion.misc       0.49      0.22      0.30       251

                accuracy                           0.72      7532
               macro avg       0.72      0.70      0.70      7532
            weighted avg       0.72      0.72      0.71      7532



===> Classification Metrics:

accuracy classification score
	accuracy score:  0.7173393520977164
	accuracy score (normalize=False):  5403

compute the precision
	precision score (average=macro):  0.715148718205421
	precision score (average=micro):  0.7173393520977164
	precision score (average=weighted):  0.7228926841676532
	precision score (average=None):  [0.55642023 0.64759725 0.7337884  0.65105386 0.74270557 0.82336957
 0.81167109 0.53017945 0.81315789 0.89459459 0.9120603  0.77979275
 0.69616519 0.83733333 0.77641278 0.61120841 0.57608696 0.83240223
 0.58590308 0.49107143]
	precision score (average=None, zero_division=1):  [0.55642023 0.64759725 0.7337884  0.65105386 0.74270557 0.82336957
 0.81167109 0.53017945 0.81315789 0.89459459 0.9120603  0.77979275
 0.69616519 0.83733333 0.77641278 0.61120841 0.57608696 0.83240223
 0.58590308 0.49107143]

compute the precision
	recall score (average=macro):  0.702571477365105
	recall score (average=micro):  0.7173393520977164
	recall score (average=weighted):  0.7173393520977164
	recall score (average=None):  [0.44827586 0.72750643 0.54568528 0.70918367 0.72727273 0.76708861
 0.78461538 0.82070707 0.77638191 0.83375315 0.90977444 0.76010101
 0.60050891 0.79292929 0.80203046 0.87688442 0.72802198 0.79255319
 0.42903226 0.21912351]
	recall score (average=None, zero_division=1):  [0.44827586 0.72750643 0.54568528 0.70918367 0.72727273 0.76708861
 0.78461538 0.82070707 0.77638191 0.83375315 0.90977444 0.76010101
 0.60050891 0.79292929 0.80203046 0.87688442 0.72802198 0.79255319
 0.42903226 0.21912351]

compute the F1 score, also known as balanced F-score or F-measure
	f1 score (average=macro):  0.7009116632385404
	f1 score (average=micro):  0.7173393520977164
	f1 score (average=weighted):  0.7129991383187694
	f1 score (average=None):  [0.49652778 0.68523002 0.62590975 0.67887668 0.73490814 0.79423329
 0.79791395 0.64420218 0.79434447 0.863103   0.91091593 0.76982097
 0.64480874 0.81452659 0.78901373 0.72033024 0.64320388 0.8119891
 0.49534451 0.3030303 ]

compute the F-beta score
	f beta score (average=macro):  0.7071561929195752
	f beta score (average=micro):  0.7173393520977163
	f beta score (average=weighted):  0.7170191188675598
	f beta score (average=None):  [0.53080921 0.66214319 0.68646232 0.66190476 0.73956683 0.81146224
 0.8061117  0.57057584 0.80552659 0.88172616 0.91160221 0.7757732
 0.67467124 0.82805907 0.78140455 0.65063386 0.60117967 0.82411504
 0.54597701 0.39341917]

compute the average Hamming loss
	hamming loss:  0.28266064790228357

jaccard similarity coefficient score
	jaccard score (average=macro):  0.5561106882196765
	jaccard score (average=None):  [0.33025404 0.52117864 0.45550847 0.51386322 0.58091286 0.65869565
 0.6637744  0.4751462  0.65884861 0.75917431 0.83640553 0.62577963
 0.47580645 0.68708972 0.65154639 0.56290323 0.47406082 0.68348624
 0.32920792 0.17857143]

confusion matrix:
[[143   0   2   1   1   2   0  13   4   3   4   1   1   4  10  75   9  11
   10  25]
 [  3 283  11  12  11  26   5   8   2   3   0   9   4   0   9   1   0   1
    1   0]
 [  4  30 215  48  18  21   6  17   4   1   1   6   2   3  10   1   2   0
    3   2]
 [  0  14  23 278  25   4  10  10   0   0   1   3  23   0   1   0   0   0
    0   0]
 [  0   9   6  27 280   4   8  19   1   0   1   5  17   3   4   1   0   0
    0   0]
 [  0  42  18   4   6 303   2   7   1   1   0   2   3   2   3   0   1   0
    0   0]
 [  0   3   1  23  15   0 306  17   4   2   1   1   8   2   1   0   3   1
    2   0]
 [  2   1   1   1   2   0  11 325  20   0   2   2  10   1   3   2   5   2
    6   0]
 [  5   3   2   0   1   0   4  30 309   3   2   0  10   5   6   2   7   2
    5   2]
 [  4   4   0   0   0   1   6  17   4 331  15   1   1   3   0   3   4   0
    3   0]
 [  2   1   0   0   0   0   0  12   1   9 363   2   0   2   0   2   4   0
    1   0]
 [  2   7   4   4   5   3   1  17   2   4   1 301   6   1   3   1  19   5
    8   2]
 [  1  14   6  24  10   2  10  23   8   2   0  27 236  13   8   2   0   3
    3   1]
 [  2   6   0   2   0   0   3  22   3   0   0   0   7 314   5  14   9   3
    4   2]
 [  4   8   1   1   1   1   3  21   3   3   2   2   5   3 316   1   3   7
    8   1]
 [ 11   3   1   0   0   1   0  14   1   1   1   1   0   2   1 349   1   0
    4   7]
 [  3   1   2   0   1   0   2  15   3   1   0  13   0   5   9  13 265   6
   16   9]
 [ 19   3   0   0   0   0   0   8   4   4   2   3   0   1   2   9   9 298
   14   0]
 [ 17   2   0   1   0   0   0   9   4   2   0   4   3   7  10   5  95  12
  133   6]
 [ 35   3   0   1   1   0   0   9   2   0   2   3   3   4   6  90  24   7
    6  55]]

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
train time: 183.956s
test time:  0.368s
accuracy:   0.713


cross validation:
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
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/svm/_base.py:947: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  "the number of iterations.", ConvergenceWarning)
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/svm/_base.py:947: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  "the number of iterations.", ConvergenceWarning)
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/svm/_base.py:947: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  "the number of iterations.", ConvergenceWarning)
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/svm/_base.py:947: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  "the number of iterations.", ConvergenceWarning)
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/svm/_base.py:947: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  "the number of iterations.", ConvergenceWarning)
	accuracy: 5-fold cross validation: [0.77507733 0.75961114 0.77198409 0.77110031 0.75066313]
	test accuracy: 5-fold cross validation accuracy: 0.77 (+/- 0.02)


===> Classification Report:

                          precision    recall  f1-score   support

             alt.atheism       0.54      0.50      0.52       319
           comp.graphics       0.66      0.75      0.70       389
 comp.os.ms-windows.misc       0.70      0.61      0.65       394
comp.sys.ibm.pc.hardware       0.70      0.67      0.69       392
   comp.sys.mac.hardware       0.76      0.72      0.74       385
          comp.windows.x       0.84      0.73      0.78       395
            misc.forsale       0.79      0.78      0.79       390
               rec.autos       0.52      0.77      0.62       396
         rec.motorcycles       0.77      0.79      0.78       398
      rec.sport.baseball       0.85      0.85      0.85       397
        rec.sport.hockey       0.90      0.90      0.90       399
               sci.crypt       0.84      0.75      0.79       396
         sci.electronics       0.64      0.61      0.62       393
                 sci.med       0.77      0.81      0.79       396
               sci.space       0.75      0.77      0.76       394
  soc.religion.christian       0.67      0.77      0.72       398
      talk.politics.guns       0.60      0.68      0.63       364
   talk.politics.mideast       0.87      0.77      0.82       376
      talk.politics.misc       0.56      0.49      0.52       310
      talk.religion.misc       0.42      0.31      0.36       251

                accuracy                           0.71      7532
               macro avg       0.71      0.70      0.70      7532
            weighted avg       0.72      0.71      0.71      7532



===> Classification Metrics:

accuracy classification score
	accuracy score:  0.7128252788104089
	accuracy score (normalize=False):  5369

compute the precision
	precision score (average=macro):  0.7085113323950883
	precision score (average=micro):  0.7128252788104089
	precision score (average=weighted):  0.7174746299131989
	precision score (average=None):  [0.54081633 0.65695067 0.69970845 0.70320856 0.76243094 0.8425656
 0.79274611 0.5170068  0.77395577 0.85496183 0.89526185 0.84375
 0.64498645 0.76682692 0.75308642 0.67252747 0.59661836 0.87048193
 0.55925926 0.42307692]
	precision score (average=None, zero_division=1):  [0.54081633 0.65695067 0.69970845 0.70320856 0.76243094 0.8425656
 0.79274611 0.5170068  0.77395577 0.85496183 0.89526185 0.84375
 0.64498645 0.76682692 0.75308642 0.67252747 0.59661836 0.87048193
 0.55925926 0.42307692]

compute the precision
	recall score (average=macro):  0.7007622023480741
	recall score (average=micro):  0.7128252788104089
	recall score (average=weighted):  0.7128252788104089
	recall score (average=None):  [0.4984326  0.75321337 0.60913706 0.67091837 0.71688312 0.73164557
 0.78461538 0.76767677 0.79145729 0.84634761 0.89974937 0.75
 0.60559796 0.80555556 0.77411168 0.76884422 0.67857143 0.76861702
 0.48709677 0.30677291]
	recall score (average=None, zero_division=1):  [0.4984326  0.75321337 0.60913706 0.67091837 0.71688312 0.73164557
 0.78461538 0.76767677 0.79145729 0.84634761 0.89974937 0.75
 0.60559796 0.80555556 0.77411168 0.76884422 0.67857143 0.76861702
 0.48709677 0.30677291]

compute the F1 score, also known as balanced F-score or F-measure
	f1 score (average=macro):  0.7015545161116471
	f1 score (average=micro):  0.7128252788104089
	f1 score (average=weighted):  0.7121492080897529
	f1 score (average=None):  [0.5187602  0.70179641 0.65128901 0.68668407 0.73895582 0.78319783
 0.78865979 0.61788618 0.7826087  0.85063291 0.8975     0.79411765
 0.62467192 0.78571429 0.76345432 0.71746776 0.63496144 0.81638418
 0.52068966 0.3556582 ]

compute the F-beta score
	f beta score (average=macro):  0.7050289210801781
	f beta score (average=micro):  0.7128252788104089
	f beta score (average=weighted):  0.714679339679733
	f beta score (average=None):  [0.53177258 0.67418316 0.6795017  0.69650424 0.75286416 0.81777023
 0.79110651 0.55312955 0.77739388 0.85322499 0.89615577 0.82317073
 0.63670412 0.77427184 0.7571996  0.68981064 0.61138614 0.84800469
 0.54316547 0.39325843]

compute the average Hamming loss
	hamming loss:  0.2871747211895911

jaccard similarity coefficient score
	jaccard score (average=macro):  0.5540563185514078
	jaccard score (average=None):  [0.35022026 0.54059041 0.48289738 0.52286282 0.58598726 0.64365256
 0.65106383 0.44705882 0.64285714 0.74008811 0.81405896 0.65853659
 0.45419847 0.64705882 0.61740891 0.55941499 0.46516008 0.68973747
 0.35198135 0.21629213]

confusion matrix:
[[159   1   1   1   1   2   2  15   3   3   4   3   4   6  11  45   8  10
   10  30]
 [  2 293  15   6   7  21   6   7   5   1   0   5  10   1   6   1   0   0
    0   3]
 [  2  27 240  36  15  14   0  16   3   0   1   3   3   9  10   4   2   2
    5   2]
 [  0  15  29 263  25   3  13  10   0   3   1   3  24   1   2   0   0   0
    0   0]
 [  2   8   7  21 276   5  10  18   2   1   0   4  18   3   5   1   2   2
    0   0]
 [  2  41  28   5   2 289   2   7   1   2   0   2   5   1   5   0   1   1
    0   1]
 [  2   2   2  13  14   0 306  15   4   1   2   1  13   2   3   1   4   2
    3   0]
 [  4   2   1   3   2   1  10 304  25   5   1   1  16   2   4   2   5   2
    6   0]
 [  3   3   0   0   2   0   5  27 315   3   3   1   8   5   4   3   4   0
   10   2]
 [  2   3   0   0   0   0   3  19   6 336  13   1   2   2   0   3   1   1
    4   1]
 [  1   2   0   0   0   0   0  11   2  14 359   0   0   3   0   1   4   1
    1   0]
 [  3   7   3   5   2   1   2  19   4   3   2 297   7   4   6   4  14   1
    9   3]
 [  3  13   8  20  13   3  14  24   5   3   2  13 238  16  11   2   0   1
    2   2]
 [  6   5   2   0   0   0   3  18   7   1   5   0   5 319   4   6   4   3
    6   2]
 [  5  11   3   0   1   1   2  23   3   2   2   2   8  10 305   2   3   1
    8   2]
 [ 20   3   1   0   0   0   1  14   3   2   1   1   1   5   4 306   0   3
    7  26]
 [  7   3   1   0   2   0   3  14   6   3   0  12   0   8  10   4 247   4
   22  18]
 [ 27   0   1   0   0   1   1   7   5   6   2   0   1   4   1   4  10 289
   14   3]
 [ 10   1   0   0   0   1   1  11   4   3   2   3   3   9   9   3  85   4
  151  10]
 [ 34   6   1   1   0   1   2   9   4   1   1   0   3   6   5  63  20   5
   12  77]]

FINAL CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS dataset (removing headers signatures and quoting)
| ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
| --- | ------------- | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
|  1  |  ADA_BOOST_CLASSIFIER  |  44.04%  |  [0.46973045 0.45779938 0.46575342 0.47017234 0.46993811]  |  0.47 (+/- 0.01)  |  19.26  |  1.015  |
|  2  |  BERNOULLI_NB  |  62.56%  |  [0.69200177 0.67741935 0.6752099  0.67388422 0.67992927]  |  0.68 (+/- 0.01)  |  0.07593  |  0.0528  |
|  3  |  COMPLEMENT_NB  |  71.22%  |  [0.776403   0.75828546 0.78126381 0.77507733 0.76967286]  |  0.77 (+/- 0.02)  |  0.0646  |  0.01039  |
|  4  |  DECISION_TREE_CLASSIFIER  |  44.72%  |  [0.49094123 0.48696421 0.47547503 0.49270879 0.49646331]  |  0.49 (+/- 0.01)  |  9.037  |  0.006431  |
|  5  |  GRADIENT_BOOSTING_CLASSIFIER  |  59.43%  |  [0.65930181 0.62704375 0.64781264 0.6548829  0.64721485]  |  0.65 (+/- 0.02)  |  658.5  |  0.3929  |
|  6  |  K_NEIGHBORS_CLASSIFIER  |  8.48%  |  [0.12417145 0.11842687 0.12063632 0.12240389 0.12113174]  |  0.12 (+/- 0.00)  |  0.003191  |  1.298  |
|  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  0.76 (+/- 0.02)  |  0.8115  |  0.008989  |
|  8  |  LOGISTIC_REGRESSION  |  69.28%  |  [0.75961114 0.74016792 0.75519222 0.7565179  0.73607427]  |  0.75 (+/- 0.02)  |  22.88  |  0.01089  |
|  9  |  MULTINOMIAL_NB  |  68.79%  |  [0.76049492 0.74149359 0.75519222 0.75695979 0.74270557]  |  0.75 (+/- 0.02)  |  0.07197  |  0.01174  |
|  10  |  NEAREST_CENTROID  |  66.70%  |  [0.71674768 0.71807335 0.71851525 0.71939903 0.70601238]  |  0.72 (+/- 0.01)  |  0.01669  |  0.01906  |
|  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  69.62%  |  [0.76535572 0.74856385 0.76182059 0.77065842 0.74889478]  |  0.76 (+/- 0.02)  |  2.319  |  0.01587  |
|  12  |  PERCEPTRON  |  53.86%  |  [0.63146266 0.61025188 0.58948299 0.60450729 0.60477454]  |  0.61 (+/- 0.03)  |  0.4178  |  0.0171  |
|  13  |  RANDOM_FOREST_CLASSIFIER  |  63.71%  |  [0.69465312 0.66902342 0.68272205 0.69730446 0.67462423]  |  0.68 (+/- 0.02)  |  7.79  |  0.3067  |
|  14  |  RIDGE_CLASSIFIER  |  70.02%  |  [0.76756518 0.74502872 0.76800707 0.776403   0.76083112]  |  0.76 (+/- 0.02)  |  3.12  |  0.02272  |
|  15  |  MAJORITY_VOTING_CLASSIFIER  |  70.37%  |  [0.76712329 0.75077331 0.76800707 0.77551922 0.7515473 ]  |  0.76 (+/- 0.02)  |  31.06  |  0.4181  |
|  16  |  SOFT_VOTING_CLASSIFIER  |  71.73%  |  [0.79231109 0.75828546 0.79098542 0.7870084  0.76702034]  |  0.78 (+/- 0.03)  |  28.07  |  0.3526  |
|  17  |  STACKING_CLASSIFIER  |  71.28%  |  [0.77507733 0.75961114 0.77198409 0.77110031 0.75066313]  |  0.77 (+/- 0.02)  |  184.0  |  0.368  |


Best algorithm:
===> 16) SOFT_VOTING_CLASSIFIER
		Accuracy score = 71.73%		Training time = 28.07		Test time = 0.3526

Loading IMDB_REVIEWS dataset:

===> Reading files from /home/rpessoa/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/train/neg

===> Reading files from /home/rpessoa/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/train/pos

===> Reading files from /home/rpessoa/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/test/neg

===> Reading files from /home/rpessoa/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/test/pos
data loaded
25000 documents - 33.133MB (training set)
25000 documents - 32.351MB (test set)

Extracting features from the training data using a vectorizer
done in 2.762614s at 11.993MB/s
n_samples: 25000, n_features: 74170

Extracting features from the test data using the same vectorizer
done in 2.696052s at 11.999MB/s
n_samples: 25000, n_features: 74170

	==> Using JSON with best parameters (selected using grid search) to the ADA_BOOST_CLASSIFIER classifier (binary classification) and IMDB_REVIEWS dataset ===> JSON in dictionary format: {'learning_rate': 1, 'n_estimators': 500}
	 AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1,
                   n_estimators=500, random_state=0)
	==> Using JSON with best parameters (selected using grid search) to the BERNOULLI_NB classifier (binary classification) and IMDB_REVIEWS dataset ===> JSON in dictionary format: {'alpha': 0.5, 'binarize': 0.0001, 'fit_prior': False}
	 BernoulliNB(alpha=0.5, binarize=0.0001, class_prior=None, fit_prior=False)
	==> Using JSON with best parameters (selected using grid search) to the COMPLEMENT_NB classifier (binary classification) and IMDB_REVIEWS dataset ===> JSON in dictionary format: {'alpha': 1.0, 'fit_prior': False, 'norm': False}
	 ComplementNB(alpha=1.0, class_prior=None, fit_prior=False, norm=False)
	==> Using JSON with best parameters (selected using grid search) to the DECISION_TREE_CLASSIFIER classifier (binary classification) and IMDB_REVIEWS dataset ===> JSON in dictionary format: {'criterion': 'entropy', 'min_samples_split': 250, 'splitter': 'random'}
	 DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='entropy',
                       max_depth=None, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=250,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=0, splitter='random')
	==> Using JSON with best parameters (selected using grid search) to the GRADIENT_BOOSTING_CLASSIFIER classifier (binary classification) and IMDB_REVIEWS dataset ===> JSON in dictionary format: {'learning_rate': 0.1, 'n_estimators': 200}
	 GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.1, loss='deviance', max_depth=3,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=200,
                           n_iter_no_change=None, presort='deprecated',
                           random_state=0, subsample=1.0, tol=0.0001,
                           validation_fraction=0.1, verbose=False,
                           warm_start=False)
	==> Using JSON with best parameters (selected using grid search) to the K_NEIGHBORS_CLASSIFIER classifier (binary classification) and IMDB_REVIEWS dataset ===> JSON in dictionary format: {'leaf_size': 5, 'metric': 'euclidean', 'n_neighbors': 50, 'weights': 'distance'}
	 KNeighborsClassifier(algorithm='auto', leaf_size=5, metric='euclidean',
                     metric_params=None, n_jobs=-1, n_neighbors=50, p=2,
                     weights='distance')
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
	==> Using JSON with best parameters (selected using grid search) to the MULTINOMIAL_NB classifier (binary classification) and IMDB_REVIEWS dataset ===> JSON in dictionary format: {'alpha': 1.0, 'fit_prior': False}
	 MultinomialNB(alpha=1.0, class_prior=None, fit_prior=False)
	==> Using JSON with best parameters (selected using grid search) to the NEAREST_CENTROID classifier (binary classification) and IMDB_REVIEWS dataset ===> JSON in dictionary format: {'metric': 'cosine'}
	 NearestCentroid(metric='cosine', shrink_threshold=None)
	==> Using JSON with best parameters (selected using grid search) to the PASSIVE_AGGRESSIVE_CLASSIFIER classifier (binary classification) and IMDB_REVIEWS dataset ===> JSON in dictionary format: {'C': 0.01, 'early_stopping': False, 'tol': 0.001, 'validation_fraction': 0.01}
	 PassiveAggressiveClassifier(C=0.01, average=False, class_weight=None,
                            early_stopping=False, fit_intercept=True,
                            loss='hinge', max_iter=1000, n_iter_no_change=5,
                            n_jobs=-1, random_state=0, shuffle=True, tol=0.001,
                            validation_fraction=0.01, verbose=False,
                            warm_start=False)
	==> Using JSON with best parameters (selected using grid search) to the PERCEPTRON classifier (binary classification) and IMDB_REVIEWS dataset ===> JSON in dictionary format: {'early_stopping': True, 'max_iter': 100, 'n_iter_no_change': 3, 'penalty': 'l2', 'tol': 0.0001, 'validation_fraction': 0.01}
	 Perceptron(alpha=0.0001, class_weight=None, early_stopping=True, eta0=1.0,
           fit_intercept=True, max_iter=100, n_iter_no_change=3, n_jobs=-1,
           penalty='l2', random_state=0, shuffle=True, tol=0.0001,
           validation_fraction=0.01, verbose=False, warm_start=False)
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
	 VotingClassifier(estimators=[('complement_nb',
                              ComplementNB(alpha=1.0, class_prior=None,
                                           fit_prior=False, norm=False)),
                             ('linear_svc',
                              LinearSVC(C=1.0, class_weight=None, dual=True,
                                        fit_intercept=True, intercept_scaling=1,
                                        loss='squared_hinge', max_iter=1000,
                                        multi_class='ovr', penalty='l2',
                                        random_state=0, tol=0.0001,
                                        verbose=False)),
                             ('logistic_regression',
                              Logisti...
                                                     min_samples_split=5,
                                                     min_weight_fraction_leaf=0.0,
                                                     n_estimators=200,
                                                     n_jobs=-1, oob_score=False,
                                                     random_state=0,
                                                     verbose=False,
                                                     warm_start=False)),
                             ('ridge_classifier',
                              RidgeClassifier(alpha=1.0, class_weight=None,
                                              copy_X=True, fit_intercept=True,
                                              max_iter=None, normalize=False,
                                              random_state=0, solver='auto',
                                              tol=0.0001))],
                 flatten_transform=True, n_jobs=-1, voting='hard',
                 weights=None)
	==> Using JSON with best parameters (selected using grid search) to the COMPLEMENT_NB classifier (binary classification) and IMDB_REVIEWS dataset ===> JSON in dictionary format: {'alpha': 1.0, 'fit_prior': False, 'norm': False}
	 ComplementNB(alpha=1.0, class_prior=None, fit_prior=False, norm=False)
	==> Using JSON with best parameters (selected using grid search) to the LOGISTIC_REGRESSION classifier (binary classification) and IMDB_REVIEWS dataset ===> JSON in dictionary format: {'C': 10, 'tol': 0.01}
	 LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=-1, penalty='l2', random_state=0,
                   solver='lbfgs', tol=0.01, verbose=False, warm_start=False)
	==> Using JSON with best parameters (selected using grid search) to the MULTINOMIAL_NB classifier (binary classification) and IMDB_REVIEWS dataset ===> JSON in dictionary format: {'alpha': 1.0, 'fit_prior': False}
	 MultinomialNB(alpha=1.0, class_prior=None, fit_prior=False)
	==> Using JSON with best parameters (selected using grid search) to the RANDOM_FOREST_CLASSIFIER classifier (binary classification) and IMDB_REVIEWS dataset ===> JSON in dictionary format: {'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 200}
	 RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=5,
                       min_weight_fraction_leaf=0.0, n_estimators=200,
                       n_jobs=-1, oob_score=False, random_state=0,
                       verbose=False, warm_start=False)
	 VotingClassifier(estimators=[('complement_nb',
                              ComplementNB(alpha=1.0, class_prior=None,
                                           fit_prior=False, norm=False)),
                             ('logistic_regression',
                              LogisticRegression(C=10, class_weight=None,
                                                 dual=False, fit_intercept=True,
                                                 intercept_scaling=1,
                                                 l1_ratio=None, max_iter=100,
                                                 multi_class='auto', n_jobs=-1,
                                                 penalty='l2', random_state=0,
                                                 solver='lbfgs', tol=0.01,
                                                 verbose=Fals...
                                                     criterion='gini',
                                                     max_depth=None,
                                                     max_features='auto',
                                                     max_leaf_nodes=None,
                                                     max_samples=None,
                                                     min_impurity_decrease=0.0,
                                                     min_impurity_split=None,
                                                     min_samples_leaf=1,
                                                     min_samples_split=5,
                                                     min_weight_fraction_leaf=0.0,
                                                     n_estimators=200,
                                                     n_jobs=-1, oob_score=False,
                                                     random_state=0,
                                                     verbose=False,
                                                     warm_start=False))],
                 flatten_transform=True, n_jobs=-1, voting='soft',
                 weights=None)
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
Classifier.ADA_BOOST_CLASSIFIER
________________________________________________________________________________
Training: 
AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1,
                   n_estimators=500, random_state=0)
train time: 103.253s
test time:  5.553s
accuracy:   0.846


cross validation:
	accuracy: 5-fold cross validation: [0.8398 0.8516 0.8416 0.8366 0.8416]
	test accuracy: 5-fold cross validation accuracy: 0.84 (+/- 0.01)


===> Classification Report:

              precision    recall  f1-score   support

           0       0.85      0.83      0.84     12500
           1       0.84      0.86      0.85     12500

    accuracy                           0.85     25000
   macro avg       0.85      0.85      0.85     25000
weighted avg       0.85      0.85      0.85     25000



===> Classification Metrics:

accuracy classification score
	accuracy score:  0.846
	accuracy score (normalize=False):  21150

compute the precision
	precision score (average=macro):  0.8461687664124815
	precision score (average=micro):  0.846
	precision score (average=weighted):  0.8461687664124815
	precision score (average=None):  [0.85381217 0.83852536]
	precision score (average=None, zero_division=1):  [0.85381217 0.83852536]

compute the precision
	recall score (average=macro):  0.8460000000000001
	recall score (average=micro):  0.846
	recall score (average=weighted):  0.846
	recall score (average=None):  [0.83496 0.85704]
	recall score (average=None, zero_division=1):  [0.83496 0.85704]

compute the F1 score, also known as balanced F-score or F-measure
	f1 score (average=macro):  0.8459812279456319
	f1 score (average=micro):  0.8459999999999999
	f1 score (average=weighted):  0.8459812279456318
	f1 score (average=None):  [0.84428086 0.8476816 ]

compute the F-beta score
	f beta score (average=macro):  0.8460689772560853
	f beta score (average=micro):  0.846
	f beta score (average=weighted):  0.8460689772560853
	f beta score (average=None):  [0.84997394 0.84216401]

compute the average Hamming loss
	hamming loss:  0.154

jaccard similarity coefficient score
	jaccard score (average=macro):  0.7330778237237369
	jaccard score (average=None):  [0.73052425 0.73563139]

confusion matrix:
[[10437  2063]
 [ 1787 10713]]

================================================================================
Classifier.BERNOULLI_NB
________________________________________________________________________________
Training: 
BernoulliNB(alpha=0.5, binarize=0.0001, class_prior=None, fit_prior=False)
train time: 0.028s
test time:  0.022s
accuracy:   0.813


cross validation:
	accuracy: 5-fold cross validation: [0.8398 0.8424 0.8514 0.8396 0.8516]
	test accuracy: 5-fold cross validation accuracy: 0.84 (+/- 0.01)
dimensionality: 74170
density: 1.000000



===> Classification Report:

              precision    recall  f1-score   support

           0       0.77      0.89      0.83     12500
           1       0.87      0.74      0.80     12500

    accuracy                           0.81     25000
   macro avg       0.82      0.81      0.81     25000
weighted avg       0.82      0.81      0.81     25000



===> Classification Metrics:

accuracy classification score
	accuracy score:  0.8128
	accuracy score (normalize=False):  20320

compute the precision
	precision score (average=macro):  0.8199607733001818
	precision score (average=micro):  0.8128
	precision score (average=weighted):  0.8199607733001819
	precision score (average=None):  [0.77209464 0.8678269 ]
	precision score (average=None, zero_division=1):  [0.77209464 0.8678269 ]

compute the precision
	recall score (average=macro):  0.8128
	recall score (average=micro):  0.8128
	recall score (average=weighted):  0.8128
	recall score (average=None):  [0.8876 0.738 ]
	recall score (average=None, zero_division=1):  [0.8876 0.738 ]

compute the F1 score, also known as balanced F-score or F-measure
	f1 score (average=macro):  0.8117467153422083
	f1 score (average=micro):  0.8128
	f1 score (average=weighted):  0.8117467153422083
	f1 score (average=None):  [0.82582806 0.79766537]

compute the F-beta score
	f beta score (average=macro):  0.8155290045481136
	f beta score (average=micro):  0.8128
	f beta score (average=weighted):  0.8155290045481137
	f beta score (average=None):  [0.79272649 0.83833152]

compute the average Hamming loss
	hamming loss:  0.1872

jaccard similarity coefficient score
	jaccard score (average=macro):  0.6833792357125639
	jaccard score (average=None):  [0.70332805 0.66343042]

confusion matrix:
[[11095  1405]
 [ 3275  9225]]

================================================================================
Classifier.COMPLEMENT_NB
________________________________________________________________________________
Training: 
ComplementNB(alpha=1.0, class_prior=None, fit_prior=False, norm=False)
train time: 0.017s
test time:  0.008s
accuracy:   0.839


cross validation:
	accuracy: 5-fold cross validation: [0.8564 0.8678 0.8682 0.8678 0.8672]
	test accuracy: 5-fold cross validation accuracy: 0.87 (+/- 0.01)
dimensionality: 74170
density: 1.000000



===> Classification Report:

              precision    recall  f1-score   support

           0       0.82      0.88      0.85     12500
           1       0.87      0.80      0.83     12500

    accuracy                           0.84     25000
   macro avg       0.84      0.84      0.84     25000
weighted avg       0.84      0.84      0.84     25000



===> Classification Metrics:

accuracy classification score
	accuracy score:  0.83928
	accuracy score (normalize=False):  20982

compute the precision
	precision score (average=macro):  0.8412761894108505
	precision score (average=micro):  0.83928
	precision score (average=weighted):  0.8412761894108506
	precision score (average=None):  [0.81517539 0.86737699]
	precision score (average=None, zero_division=1):  [0.81517539 0.86737699]

compute the precision
	recall score (average=macro):  0.83928
	recall score (average=micro):  0.83928
	recall score (average=weighted):  0.83928
	recall score (average=None):  [0.87752 0.80104]
	recall score (average=None, zero_division=1):  [0.87752 0.80104]

compute the F1 score, also known as balanced F-score or F-measure
	f1 score (average=macro):  0.8390446353565748
	f1 score (average=micro):  0.83928
	f1 score (average=weighted):  0.8390446353565747
	f1 score (average=None):  [0.84519957 0.8328897 ]

compute the F-beta score
	f beta score (average=macro):  0.8400851674217381
	f beta score (average=micro):  0.83928
	f beta score (average=weighted):  0.8400851674217381
	f beta score (average=None):  [0.8269254  0.85324494]

compute the average Hamming loss
	hamming loss:  0.16072

jaccard similarity coefficient score
	jaccard score (average=macro):  0.7227675383902549
	jaccard score (average=None):  [0.73190098 0.7136341 ]

confusion matrix:
[[10969  1531]
 [ 2487 10013]]

================================================================================
Classifier.DECISION_TREE_CLASSIFIER
________________________________________________________________________________
Training: 
DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='entropy',
                       max_depth=None, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=250,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=0, splitter='random')
train time: 7.808s
test time:  0.012s
accuracy:   0.741


cross validation:
	accuracy: 5-fold cross validation: [0.735  0.7292 0.746  0.739  0.7342]
	test accuracy: 5-fold cross validation accuracy: 0.74 (+/- 0.01)


===> Classification Report:

              precision    recall  f1-score   support

           0       0.74      0.75      0.74     12500
           1       0.75      0.73      0.74     12500

    accuracy                           0.74     25000
   macro avg       0.74      0.74      0.74     25000
weighted avg       0.74      0.74      0.74     25000



===> Classification Metrics:

accuracy classification score
	accuracy score:  0.7414
	accuracy score (normalize=False):  18535

compute the precision
	precision score (average=macro):  0.7415101889270939
	precision score (average=micro):  0.7414
	precision score (average=weighted):  0.741510188927094
	precision score (average=None):  [0.73635153 0.74666885]
	precision score (average=None, zero_division=1):  [0.73635153 0.74666885]

compute the precision
	recall score (average=macro):  0.7414000000000001
	recall score (average=micro):  0.7414
	recall score (average=weighted):  0.7414
	recall score (average=None):  [0.75208 0.73072]
	recall score (average=None, zero_division=1):  [0.75208 0.73072]

compute the F1 score, also known as balanced F-score or F-measure
	f1 score (average=macro):  0.7413705000985304
	f1 score (average=micro):  0.7414
	f1 score (average=weighted):  0.7413705000985304
	f1 score (average=None):  [0.74413266 0.73860834]

compute the F-beta score
	f beta score (average=macro):  0.7414339986978127
	f beta score (average=micro):  0.7414
	f beta score (average=weighted):  0.7414339986978127
	f beta score (average=None):  [0.73944437 0.74342362]

compute the average Hamming loss
	hamming loss:  0.2586

jaccard similarity coefficient score
	jaccard score (average=macro):  0.5890376258980359
	jaccard score (average=None):  [0.5925249  0.58555036]

confusion matrix:
[[9401 3099]
 [3366 9134]]

================================================================================
Classifier.GRADIENT_BOOSTING_CLASSIFIER
________________________________________________________________________________
Training: 
GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.1, loss='deviance', max_depth=3,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=200,
                           n_iter_no_change=None, presort='deprecated',
                           random_state=0, subsample=1.0, tol=0.0001,
                           validation_fraction=0.1, verbose=False,
                           warm_start=False)
train time: 100.751s
test time:  0.066s
accuracy:   0.829


cross validation:
	accuracy: 5-fold cross validation: [0.8278 0.8294 0.8238 0.823  0.8284]
	test accuracy: 5-fold cross validation accuracy: 0.83 (+/- 0.01)


===> Classification Report:

              precision    recall  f1-score   support

           0       0.86      0.79      0.82     12500
           1       0.80      0.87      0.84     12500

    accuracy                           0.83     25000
   macro avg       0.83      0.83      0.83     25000
weighted avg       0.83      0.83      0.83     25000



===> Classification Metrics:

accuracy classification score
	accuracy score:  0.82864
	accuracy score (normalize=False):  20716

compute the precision
	precision score (average=macro):  0.8309664110922771
	precision score (average=micro):  0.82864
	precision score (average=weighted):  0.830966411092277
	precision score (average=None):  [0.85871463 0.80321819]
	precision score (average=None, zero_division=1):  [0.85871463 0.80321819]

compute the precision
	recall score (average=macro):  0.82864
	recall score (average=micro):  0.82864
	recall score (average=weighted):  0.82864
	recall score (average=None):  [0.78672 0.87056]
	recall score (average=None, zero_division=1):  [0.78672 0.87056]

compute the F1 score, also known as balanced F-score or F-measure
	f1 score (average=macro):  0.8283383413017682
	f1 score (average=micro):  0.82864
	f1 score (average=weighted):  0.8283383413017681
	f1 score (average=None):  [0.82114228 0.8355344 ]

compute the F-beta score
	f beta score (average=macro):  0.8295602456001892
	f beta score (average=micro):  0.82864
	f beta score (average=weighted):  0.829560245600189
	f beta score (average=None):  [0.84328051 0.81583998]

compute the average Hamming loss
	hamming loss:  0.17136

jaccard similarity coefficient score
	jaccard score (average=macro):  0.707041815580616
	jaccard score (average=None):  [0.69655759 0.71752605]

confusion matrix:
[[ 9834  2666]
 [ 1618 10882]]

================================================================================
Classifier.K_NEIGHBORS_CLASSIFIER
________________________________________________________________________________
Training: 
KNeighborsClassifier(algorithm='auto', leaf_size=5, metric='euclidean',
                     metric_params=None, n_jobs=-1, n_neighbors=50, p=2,
                     weights='distance')
train time: 0.006s
test time:  13.022s
accuracy:   0.827


cross validation:
	accuracy: 5-fold cross validation: [0.8632 0.8744 0.8694 0.864  0.8618]
	test accuracy: 5-fold cross validation accuracy: 0.87 (+/- 0.01)


===> Classification Report:

              precision    recall  f1-score   support

           0       0.80      0.86      0.83     12500
           1       0.85      0.79      0.82     12500

    accuracy                           0.83     25000
   macro avg       0.83      0.83      0.83     25000
weighted avg       0.83      0.83      0.83     25000



===> Classification Metrics:

accuracy classification score
	accuracy score:  0.8266
	accuracy score (normalize=False):  20665

compute the precision
	precision score (average=macro):  0.8283209797789661
	precision score (average=micro):  0.8266
	precision score (average=weighted):  0.8283209797789662
	precision score (average=None):  [0.80455054 0.85209142]
	precision score (average=None, zero_division=1):  [0.80455054 0.85209142]

compute the precision
	recall score (average=macro):  0.8266
	recall score (average=micro):  0.8266
	recall score (average=weighted):  0.8266
	recall score (average=None):  [0.8628 0.7904]
	recall score (average=None, zero_division=1):  [0.8628 0.7904]

compute the F1 score, also known as balanced F-score or F-measure
	f1 score (average=macro):  0.826372471541607
	f1 score (average=micro):  0.8266
	f1 score (average=weighted):  0.826372471541607
	f1 score (average=None):  [0.83265779 0.82008716]

compute the F-beta score
	f beta score (average=macro):  0.8272785893237591
	f beta score (average=micro):  0.8266
	f beta score (average=weighted):  0.8272785893237591
	f beta score (average=None):  [0.81556261 0.83899457]

compute the average Hamming loss
	hamming loss:  0.1734

jaccard similarity coefficient score
	jaccard score (average=macro):  0.7041670505111413
	jaccard score (average=None):  [0.71329365 0.69504045]

confusion matrix:
[[10785  1715]
 [ 2620  9880]]

================================================================================
Classifier.LINEAR_SVC
________________________________________________________________________________
Training: 
LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
          intercept_scaling=1, loss='squared_hinge', max_iter=1000,
          multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
          verbose=False)
train time: 0.209s
test time:  0.004s
accuracy:   0.871


cross validation:
	accuracy: 5-fold cross validation: [0.8838 0.8932 0.883  0.8836 0.8782]
	test accuracy: 5-fold cross validation accuracy: 0.88 (+/- 0.01)
dimensionality: 74170
density: 0.870622



===> Classification Report:

              precision    recall  f1-score   support

           0       0.87      0.88      0.87     12500
           1       0.88      0.87      0.87     12500

    accuracy                           0.87     25000
   macro avg       0.87      0.87      0.87     25000
weighted avg       0.87      0.87      0.87     25000



===> Classification Metrics:

accuracy classification score
	accuracy score:  0.87128
	accuracy score (normalize=False):  21782

compute the precision
	precision score (average=macro):  0.8713265792060956
	precision score (average=micro):  0.87128
	precision score (average=weighted):  0.8713265792060956
	precision score (average=None):  [0.86716772 0.87548544]
	precision score (average=None, zero_division=1):  [0.86716772 0.87548544]

compute the precision
	recall score (average=macro):  0.87128
	recall score (average=micro):  0.87128
	recall score (average=weighted):  0.87128
	recall score (average=None):  [0.87688 0.86568]
	recall score (average=None, zero_division=1):  [0.87688 0.86568]

compute the F1 score, also known as balanced F-score or F-measure
	f1 score (average=macro):  0.8712759632142064
	f1 score (average=micro):  0.87128
	f1 score (average=weighted):  0.8712759632142065
	f1 score (average=None):  [0.87199682 0.87055511]

compute the F-beta score
	f beta score (average=macro):  0.8712997733398833
	f beta score (average=micro):  0.87128
	f beta score (average=weighted):  0.8712997733398834
	f beta score (average=None):  [0.86909293 0.87350662]

compute the average Hamming loss
	hamming loss:  0.12872

jaccard similarity coefficient score
	jaccard score (average=macro):  0.771913019086539
	jaccard score (average=None):  [0.77304464 0.77078139]

confusion matrix:
[[10961  1539]
 [ 1679 10821]]

================================================================================
Classifier.LOGISTIC_REGRESSION
________________________________________________________________________________
Training: 
LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=-1, penalty='l2', random_state=0,
                   solver='lbfgs', tol=0.01, verbose=False, warm_start=False)
train time: 1.075s
test time:  0.005s
accuracy:   0.877


cross validation:
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
	accuracy: 5-fold cross validation: [0.8882 0.897  0.8878 0.8876 0.8818]
	test accuracy: 5-fold cross validation accuracy: 0.89 (+/- 0.01)
dimensionality: 74170
density: 1.000000



===> Classification Report:

              precision    recall  f1-score   support

           0       0.88      0.88      0.88     12500
           1       0.88      0.87      0.88     12500

    accuracy                           0.88     25000
   macro avg       0.88      0.88      0.88     25000
weighted avg       0.88      0.88      0.88     25000



===> Classification Metrics:

accuracy classification score
	accuracy score:  0.87748
	accuracy score (normalize=False):  21937

compute the precision
	precision score (average=macro):  0.8774921788036694
	precision score (average=micro):  0.87748
	precision score (average=weighted):  0.8774921788036694
	precision score (average=None):  [0.87534802 0.87963633]
	precision score (average=None, zero_division=1):  [0.87534802 0.87963633]

compute the precision
	recall score (average=macro):  0.87748
	recall score (average=micro):  0.87748
	recall score (average=weighted):  0.87748
	recall score (average=None):  [0.88032 0.87464]
	recall score (average=None, zero_division=1):  [0.88032 0.87464]

compute the F1 score, also known as balanced F-score or F-measure
	f1 score (average=macro):  0.8774790117947175
	f1 score (average=micro):  0.87748
	f1 score (average=weighted):  0.8774790117947175
	f1 score (average=None):  [0.87782697 0.87713105]

compute the F-beta score
	f beta score (average=macro):  0.8774852132985251
	f beta score (average=micro):  0.87748
	f beta score (average=weighted):  0.8774852132985252
	f beta score (average=None):  [0.87633792 0.87863251]

compute the average Hamming loss
	hamming loss:  0.12252

jaccard similarity coefficient score
	jaccard score (average=macro):  0.7817040511407123
	jaccard score (average=None):  [0.78225634 0.78115176]

confusion matrix:
[[11004  1496]
 [ 1567 10933]]

================================================================================
Classifier.MULTINOMIAL_NB
________________________________________________________________________________
Training: 
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=False)
train time: 0.016s
test time:  0.009s
accuracy:   0.839


cross validation:
	accuracy: 5-fold cross validation: [0.8564 0.8678 0.8682 0.8678 0.8672]
	test accuracy: 5-fold cross validation accuracy: 0.87 (+/- 0.01)
dimensionality: 74170
density: 1.000000



===> Classification Report:

              precision    recall  f1-score   support

           0       0.82      0.88      0.85     12500
           1       0.87      0.80      0.83     12500

    accuracy                           0.84     25000
   macro avg       0.84      0.84      0.84     25000
weighted avg       0.84      0.84      0.84     25000



===> Classification Metrics:

accuracy classification score
	accuracy score:  0.83928
	accuracy score (normalize=False):  20982

compute the precision
	precision score (average=macro):  0.8412761894108505
	precision score (average=micro):  0.83928
	precision score (average=weighted):  0.8412761894108506
	precision score (average=None):  [0.81517539 0.86737699]
	precision score (average=None, zero_division=1):  [0.81517539 0.86737699]

compute the precision
	recall score (average=macro):  0.83928
	recall score (average=micro):  0.83928
	recall score (average=weighted):  0.83928
	recall score (average=None):  [0.87752 0.80104]
	recall score (average=None, zero_division=1):  [0.87752 0.80104]

compute the F1 score, also known as balanced F-score or F-measure
	f1 score (average=macro):  0.8390446353565748
	f1 score (average=micro):  0.83928
	f1 score (average=weighted):  0.8390446353565747
	f1 score (average=None):  [0.84519957 0.8328897 ]

compute the F-beta score
	f beta score (average=macro):  0.8400851674217381
	f beta score (average=micro):  0.83928
	f beta score (average=weighted):  0.8400851674217381
	f beta score (average=None):  [0.8269254  0.85324494]

compute the average Hamming loss
	hamming loss:  0.16072

jaccard similarity coefficient score
	jaccard score (average=macro):  0.7227675383902549
	jaccard score (average=None):  [0.73190098 0.7136341 ]

confusion matrix:
[[10969  1531]
 [ 2487 10013]]

================================================================================
Classifier.NEAREST_CENTROID
________________________________________________________________________________
Training: 
NearestCentroid(metric='cosine', shrink_threshold=None)
train time: 0.018s
test time:  0.017s
accuracy:   0.847


cross validation:
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
	accuracy: 5-fold cross validation: [0.8426 0.8546 0.8426 0.8522 0.8502]
	test accuracy: 5-fold cross validation accuracy: 0.85 (+/- 0.01)


===> Classification Report:

              precision    recall  f1-score   support

           0       0.86      0.83      0.84     12500
           1       0.83      0.87      0.85     12500

    accuracy                           0.85     25000
   macro avg       0.85      0.85      0.85     25000
weighted avg       0.85      0.85      0.85     25000



===> Classification Metrics:

accuracy classification score
	accuracy score:  0.84652
	accuracy score (normalize=False):  21163

compute the precision
	precision score (average=macro):  0.8470910041313341
	precision score (average=micro):  0.84652
	precision score (average=weighted):  0.8470910041313342
	precision score (average=None):  [0.86116902 0.83301299]
	precision score (average=None, zero_division=1):  [0.86116902 0.83301299]

compute the precision
	recall score (average=macro):  0.8465199999999999
	recall score (average=micro):  0.84652
	recall score (average=weighted):  0.84652
	recall score (average=None):  [0.82624 0.8668 ]
	recall score (average=None, zero_division=1):  [0.82624 0.8668 ]

compute the F1 score, also known as balanced F-score or F-measure
	f1 score (average=macro):  0.8464568510193563
	f1 score (average=micro):  0.84652
	f1 score (average=weighted):  0.8464568510193564
	f1 score (average=None):  [0.843343   0.84957071]

compute the F-beta score
	f beta score (average=macro):  0.8467534781841972
	f beta score (average=micro):  0.84652
	f beta score (average=weighted):  0.8467534781841971
	f beta score (average=None):  [0.85394894 0.83955802]

compute the average Hamming loss
	hamming loss:  0.15348

jaccard similarity coefficient score
	jaccard score (average=macro):  0.7338012671771122
	jaccard score (average=None):  [0.72912107 0.73848146]

confusion matrix:
[[10328  2172]
 [ 1665 10835]]

================================================================================
Classifier.PASSIVE_AGGRESSIVE_CLASSIFIER
________________________________________________________________________________
Training: 
PassiveAggressiveClassifier(C=0.01, average=False, class_weight=None,
                            early_stopping=False, fit_intercept=True,
                            loss='hinge', max_iter=1000, n_iter_no_change=5,
                            n_jobs=-1, random_state=0, shuffle=True, tol=0.001,
                            validation_fraction=0.01, verbose=False,
                            warm_start=False)
train time: 0.858s
test time:  0.004s
accuracy:   0.881


cross validation:
	accuracy: 5-fold cross validation: [0.8874 0.8966 0.8886 0.888  0.8846]
	test accuracy: 5-fold cross validation accuracy: 0.89 (+/- 0.01)
dimensionality: 74170
density: 0.999892



===> Classification Report:

              precision    recall  f1-score   support

           0       0.88      0.88      0.88     12500
           1       0.88      0.88      0.88     12500

    accuracy                           0.88     25000
   macro avg       0.88      0.88      0.88     25000
weighted avg       0.88      0.88      0.88     25000



===> Classification Metrics:

accuracy classification score
	accuracy score:  0.88068
	accuracy score (normalize=False):  22017

compute the precision
	precision score (average=macro):  0.8806902938655461
	precision score (average=micro):  0.88068
	precision score (average=weighted):  0.8806902938655461
	precision score (average=None):  [0.8787107  0.88266988]
	precision score (average=None, zero_division=1):  [0.8787107  0.88266988]

compute the precision
	recall score (average=macro):  0.8806799999999999
	recall score (average=micro):  0.88068
	recall score (average=weighted):  0.88068
	recall score (average=None):  [0.88328 0.87808]
	recall score (average=None, zero_division=1):  [0.88328 0.87808]

compute the F1 score, also known as balanced F-score or F-measure
	f1 score (average=macro):  0.8806791933913473
	f1 score (average=micro):  0.88068
	f1 score (average=weighted):  0.8806791933913474
	f1 score (average=None):  [0.88098943 0.88036896]

compute the F-beta score
	f beta score (average=macro):  0.8806844247723813
	f beta score (average=micro):  0.88068
	f beta score (average=weighted):  0.8806844247723813
	f beta score (average=None):  [0.87962078 0.88174807]

compute the average Hamming loss
	hamming loss:  0.11932

jaccard similarity coefficient score
	jaccard score (average=macro):  0.7867979776933728
	jaccard score (average=None):  [0.78729321 0.78630274]

confusion matrix:
[[11041  1459]
 [ 1524 10976]]

================================================================================
Classifier.PERCEPTRON
________________________________________________________________________________
Training: 
Perceptron(alpha=0.0001, class_weight=None, early_stopping=True, eta0=1.0,
           fit_intercept=True, max_iter=100, n_iter_no_change=3, n_jobs=-1,
           penalty='l2', random_state=0, shuffle=True, tol=0.0001,
           validation_fraction=0.01, verbose=False, warm_start=False)
train time: 0.091s
test time:  0.007s
accuracy:   0.806


cross validation:
	accuracy: 5-fold cross validation: [0.8166 0.8264 0.8144 0.81   0.8102]
	test accuracy: 5-fold cross validation accuracy: 0.82 (+/- 0.01)
dimensionality: 74170
density: 0.712363



===> Classification Report:

              precision    recall  f1-score   support

           0       0.81      0.80      0.81     12500
           1       0.80      0.81      0.81     12500

    accuracy                           0.81     25000
   macro avg       0.81      0.81      0.81     25000
weighted avg       0.81      0.81      0.81     25000



===> Classification Metrics:

accuracy classification score
	accuracy score:  0.80636
	accuracy score (normalize=False):  20159

compute the precision
	precision score (average=macro):  0.806385038275544
	precision score (average=micro):  0.80636
	precision score (average=weighted):  0.806385038275544
	precision score (average=None):  [0.80915476 0.80361532]
	precision score (average=None, zero_division=1):  [0.80915476 0.80361532]

compute the precision
	recall score (average=macro):  0.80636
	recall score (average=micro):  0.80636
	recall score (average=weighted):  0.80636
	recall score (average=None):  [0.80184 0.81088]
	recall score (average=None, zero_division=1):  [0.80184 0.81088]

compute the F1 score, also known as balanced F-score or F-measure
	f1 score (average=macro):  0.8063560437765167
	f1 score (average=micro):  0.80636
	f1 score (average=weighted):  0.8063560437765167
	f1 score (average=None):  [0.80548077 0.80723131]

compute the F-beta score
	f beta score (average=macro):  0.8063694859549333
	f beta score (average=micro):  0.80636
	f beta score (average=weighted):  0.8063694859549333
	f beta score (average=None):  [0.80768115 0.80505782]

compute the average Hamming loss
	hamming loss:  0.19364

jaccard similarity coefficient score
	jaccard score (average=macro):  0.6755424135989501
	jaccard score (average=None):  [0.67431378 0.67677105]

confusion matrix:
[[10023  2477]
 [ 2364 10136]]

================================================================================
Classifier.RANDOM_FOREST_CLASSIFIER
________________________________________________________________________________
Training: 
RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=5,
                       min_weight_fraction_leaf=0.0, n_estimators=200,
                       n_jobs=-1, oob_score=False, random_state=0,
                       verbose=False, warm_start=False)
train time: 8.792s
test time:  0.724s
accuracy:   0.855


cross validation:
	accuracy: 5-fold cross validation: [0.8504 0.8584 0.8488 0.8518 0.8568]
	test accuracy: 5-fold cross validation accuracy: 0.85 (+/- 0.01)


===> Classification Report:

              precision    recall  f1-score   support

           0       0.85      0.86      0.86     12500
           1       0.86      0.85      0.85     12500

    accuracy                           0.85     25000
   macro avg       0.85      0.85      0.85     25000
weighted avg       0.85      0.85      0.85     25000



===> Classification Metrics:

accuracy classification score
	accuracy score:  0.85452
	accuracy score (normalize=False):  21363

compute the precision
	precision score (average=macro):  0.8546328674423362
	precision score (average=micro):  0.85452
	precision score (average=weighted):  0.8546328674423362
	precision score (average=None):  [0.84830622 0.86095952]
	precision score (average=None, zero_division=1):  [0.84830622 0.86095952]

compute the precision
	recall score (average=macro):  0.85452
	recall score (average=micro):  0.85452
	recall score (average=weighted):  0.85452
	recall score (average=None):  [0.86344 0.8456 ]
	recall score (average=None, zero_division=1):  [0.86344 0.8456 ]

compute the F1 score, also known as balanced F-score or F-measure
	f1 score (average=macro):  0.8545084237590483
	f1 score (average=micro):  0.85452
	f1 score (average=weighted):  0.8545084237590481
	f1 score (average=None):  [0.85580621 0.85321064]

compute the F-beta score
	f beta score (average=macro):  0.8545667604497629
	f beta score (average=micro):  0.85452
	f beta score (average=weighted):  0.8545667604497628
	f beta score (average=None):  [0.85129038 0.85784314]

compute the average Hamming loss
	hamming loss:  0.14548

jaccard similarity coefficient score
	jaccard score (average=macro):  0.7459775424264761
	jaccard score (average=None):  [0.74795565 0.74399944]

confusion matrix:
[[10793  1707]
 [ 1930 10570]]

================================================================================
Classifier.RIDGE_CLASSIFIER
________________________________________________________________________________
Training: 
RidgeClassifier(alpha=1.0, class_weight=None, copy_X=True, fit_intercept=True,
                max_iter=None, normalize=False, random_state=0, solver='auto',
                tol=0.0001)
train time: 0.502s
test time:  0.008s
accuracy:   0.869


cross validation:
	accuracy: 5-fold cross validation: [0.8838 0.8952 0.8892 0.882  0.8788]
	test accuracy: 5-fold cross validation accuracy: 0.89 (+/- 0.01)
dimensionality: 74170
density: 1.000000



===> Classification Report:

              precision    recall  f1-score   support

           0       0.87      0.87      0.87     12500
           1       0.87      0.87      0.87     12500

    accuracy                           0.87     25000
   macro avg       0.87      0.87      0.87     25000
weighted avg       0.87      0.87      0.87     25000



===> Classification Metrics:

accuracy classification score
	accuracy score:  0.86904
	accuracy score (normalize=False):  21726

compute the precision
	precision score (average=macro):  0.8690599918321336
	precision score (average=micro):  0.86904
	precision score (average=weighted):  0.8690599918321336
	precision score (average=None):  [0.86634371 0.87177627]
	precision score (average=None, zero_division=1):  [0.86634371 0.87177627]

compute the precision
	recall score (average=macro):  0.86904
	recall score (average=micro):  0.86904
	recall score (average=weighted):  0.86904
	recall score (average=None):  [0.87272 0.86536]
	recall score (average=None, zero_division=1):  [0.87272 0.86536]

compute the F1 score, also known as balanced F-score or F-measure
	f1 score (average=macro):  0.8690382264632781
	f1 score (average=micro):  0.86904
	f1 score (average=weighted):  0.8690382264632781
	f1 score (average=None):  [0.86952017 0.86855629]

compute the F-beta score
	f beta score (average=macro):  0.8690484608048015
	f beta score (average=micro):  0.86904
	f beta score (average=weighted):  0.8690484608048015
	f beta score (average=None):  [0.8676115  0.87048542]

compute the average Hamming loss
	hamming loss:  0.13096

jaccard similarity coefficient score
	jaccard score (average=macro):  0.768406687100621
	jaccard score (average=None):  [0.76916026 0.76765311]

confusion matrix:
[[10909  1591]
 [ 1683 10817]]

================================================================================
Classifier.MAJORITY_VOTING_CLASSIFIER
________________________________________________________________________________
Training: 
VotingClassifier(estimators=[('complement_nb',
                              ComplementNB(alpha=1.0, class_prior=None,
                                           fit_prior=False, norm=False)),
                             ('linear_svc',
                              LinearSVC(C=1.0, class_weight=None, dual=True,
                                        fit_intercept=True, intercept_scaling=1,
                                        loss='squared_hinge', max_iter=1000,
                                        multi_class='ovr', penalty='l2',
                                        random_state=0, tol=0.0001,
                                        verbose=False)),
                             ('logistic_regression',
                              Logisti...
                                                     min_samples_split=5,
                                                     min_weight_fraction_leaf=0.0,
                                                     n_estimators=200,
                                                     n_jobs=-1, oob_score=False,
                                                     random_state=0,
                                                     verbose=False,
                                                     warm_start=False)),
                             ('ridge_classifier',
                              RidgeClassifier(alpha=1.0, class_weight=None,
                                              copy_X=True, fit_intercept=True,
                                              max_iter=None, normalize=False,
                                              random_state=0, solver='auto',
                                              tol=0.0001))],
                 flatten_transform=True, n_jobs=-1, voting='hard',
                 weights=None)
train time: 10.806s
test time:  0.795s
accuracy:   0.879


cross validation:
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
	accuracy: 5-fold cross validation: [0.8882 0.8976 0.8896 0.8884 0.8846]
	test accuracy: 5-fold cross validation accuracy: 0.89 (+/- 0.01)


===> Classification Report:

              precision    recall  f1-score   support

           0       0.87      0.89      0.88     12500
           1       0.89      0.87      0.88     12500

    accuracy                           0.88     25000
   macro avg       0.88      0.88      0.88     25000
weighted avg       0.88      0.88      0.88     25000



===> Classification Metrics:

accuracy classification score
	accuracy score:  0.87884
	accuracy score (normalize=False):  21971

compute the precision
	precision score (average=macro):  0.879093122600882
	precision score (average=micro):  0.87884
	precision score (average=weighted):  0.8790931226008822
	precision score (average=None):  [0.86929736 0.88888889]
	precision score (average=None, zero_division=1):  [0.86929736 0.88888889]

compute the precision
	recall score (average=macro):  0.8788400000000001
	recall score (average=micro):  0.87884
	recall score (average=weighted):  0.87884
	recall score (average=None):  [0.89176 0.86592]
	recall score (average=None, zero_division=1):  [0.89176 0.86592]

compute the F1 score, also known as balanced F-score or F-measure
	f1 score (average=macro):  0.8788197718207589
	f1 score (average=micro):  0.8788399999999998
	f1 score (average=weighted):  0.8788197718207588
	f1 score (average=None):  [0.88038542 0.87725412]

compute the F-beta score
	f beta score (average=macro):  0.8789485202234015
	f beta score (average=micro):  0.87884
	f beta score (average=weighted):  0.8789485202234014
	f beta score (average=None):  [0.8736989  0.88419814]

compute the average Hamming loss
	hamming loss:  0.12116

jaccard similarity coefficient score
	jaccard score (average=macro):  0.7838380037108439
	jaccard score (average=None):  [0.78632901 0.781347  ]

confusion matrix:
[[11147  1353]
 [ 1676 10824]]

================================================================================
Classifier.SOFT_VOTING_CLASSIFIER
________________________________________________________________________________
Training: 
VotingClassifier(estimators=[('complement_nb',
                              ComplementNB(alpha=1.0, class_prior=None,
                                           fit_prior=False, norm=False)),
                             ('logistic_regression',
                              LogisticRegression(C=10, class_weight=None,
                                                 dual=False, fit_intercept=True,
                                                 intercept_scaling=1,
                                                 l1_ratio=None, max_iter=100,
                                                 multi_class='auto', n_jobs=-1,
                                                 penalty='l2', random_state=0,
                                                 solver='lbfgs', tol=0.01,
                                                 verbose=Fals...
                                                     criterion='gini',
                                                     max_depth=None,
                                                     max_features='auto',
                                                     max_leaf_nodes=None,
                                                     max_samples=None,
                                                     min_impurity_decrease=0.0,
                                                     min_impurity_split=None,
                                                     min_samples_leaf=1,
                                                     min_samples_split=5,
                                                     min_weight_fraction_leaf=0.0,
                                                     n_estimators=200,
                                                     n_jobs=-1, oob_score=False,
                                                     random_state=0,
                                                     verbose=False,
                                                     warm_start=False))],
                 flatten_transform=True, n_jobs=-1, voting='soft',
                 weights=None)
train time: 10.289s
test time:  0.637s
accuracy:   0.877


cross validation:
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
	accuracy: 5-fold cross validation: [0.8852 0.8972 0.8906 0.8908 0.8856]
	test accuracy: 5-fold cross validation accuracy: 0.89 (+/- 0.01)


===> Classification Report:

              precision    recall  f1-score   support

           0       0.87      0.89      0.88     12500
           1       0.89      0.86      0.88     12500

    accuracy                           0.88     25000
   macro avg       0.88      0.88      0.88     25000
weighted avg       0.88      0.88      0.88     25000



===> Classification Metrics:

accuracy classification score
	accuracy score:  0.87732
	accuracy score (normalize=False):  21933

compute the precision
	precision score (average=macro):  0.8775567286729686
	precision score (average=micro):  0.87732
	precision score (average=weighted):  0.8775567286729687
	precision score (average=None):  [0.86810271 0.88701075]
	precision score (average=None, zero_division=1):  [0.86810271 0.88701075]

compute the precision
	recall score (average=macro):  0.87732
	recall score (average=micro):  0.87732
	recall score (average=weighted):  0.87732
	recall score (average=None):  [0.88984 0.8648 ]
	recall score (average=None, zero_division=1):  [0.88984 0.8648 ]

compute the F1 score, also known as balanced F-score or F-measure
	f1 score (average=macro):  0.8773007668461235
	f1 score (average=micro):  0.87732
	f1 score (average=weighted):  0.8773007668461235
	f1 score (average=None):  [0.87883696 0.87576457]

compute the F-beta score
	f beta score (average=macro):  0.8774212918744827
	f beta score (average=micro):  0.87732
	f beta score (average=weighted):  0.8774212918744827
	f beta score (average=None):  [0.87236479 0.8824778 ]

compute the average Hamming loss
	hamming loss:  0.12268

jaccard similarity coefficient score
	jaccard score (average=macro):  0.7814243436356152
	jaccard score (average=None):  [0.78386187 0.77898681]

confusion matrix:
[[11123  1377]
 [ 1690 10810]]

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
Please also refer to the documentsation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
train time: 93.174s
test time:  0.638s
accuracy:   0.883


cross validation:
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
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/svm/_base.py:947: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  "the number of iterations.", ConvergenceWarning)
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/svm/_base.py:947: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  "the number of iterations.", ConvergenceWarning)
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/svm/_base.py:947: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  "the number of iterations.", ConvergenceWarning)
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/svm/_base.py:947: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  "the number of iterations.", ConvergenceWarning)
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/svm/_base.py:947: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  "the number of iterations.", ConvergenceWarning)
	accuracy: 5-fold cross validation: [0.8924 0.902  0.8934 0.8898 0.889 ]
	test accuracy: 5-fold cross validation accuracy: 0.89 (+/- 0.01)


===> Classification Report:

              precision    recall  f1-score   support

           0       0.88      0.89      0.88     12500
           1       0.89      0.88      0.88     12500

    accuracy                           0.88     25000
   macro avg       0.88      0.88      0.88     25000
weighted avg       0.88      0.88      0.88     25000



===> Classification Metrics:

accuracy classification score
	accuracy score:  0.88288
	accuracy score (normalize=False):  22072

compute the precision
	precision score (average=macro):  0.8829189070116334
	precision score (average=micro):  0.88288
	precision score (average=weighted):  0.8829189070116334
	precision score (average=None):  [0.87905908 0.88677873]
	precision score (average=None, zero_division=1):  [0.87905908 0.88677873]

compute the precision
	recall score (average=macro):  0.88288
	recall score (average=micro):  0.88288
	recall score (average=weighted):  0.88288
	recall score (average=None):  [0.88792 0.87784]
	recall score (average=None, zero_division=1):  [0.88792 0.87784]

compute the F1 score, also known as balanced F-score or F-measure
	f1 score (average=macro):  0.8828770248890354
	f1 score (average=micro):  0.88288
	f1 score (average=weighted):  0.8828770248890354
	f1 score (average=None):  [0.88346732 0.88228673]

compute the F-beta score
	f beta score (average=macro):  0.8828967705359524
	f beta score (average=micro):  0.88288
	f beta score (average=weighted):  0.8828967705359524
	f beta score (average=None):  [0.88081709 0.88497645]

compute the average Hamming loss
	hamming loss:  0.11712

jaccard similarity coefficient score
	jaccard score (average=macro):  0.7903136924001999
	jaccard score (average=None):  [0.79125971 0.78936767]

confusion matrix:
[[11099  1401]
 [ 1527 10973]]

FINAL CLASSIFICATION TABLE: IMDB_REVIEWS dataset (Binary classification)
| ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
| --- | ------------- | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
|  1  |  ADA_BOOST_CLASSIFIER  |  84.60%  |  [0.8398 0.8516 0.8416 0.8366 0.8416]  |  0.84 (+/- 0.01)  |  103.3  |  5.553  |
|  2  |  BERNOULLI_NB  |  81.28%  |  [0.8398 0.8424 0.8514 0.8396 0.8516]  |  0.84 (+/- 0.01)  |  0.02759  |  0.02151  |
|  3  |  COMPLEMENT_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  0.87 (+/- 0.01)  |  0.01739  |  0.00831  |
|  4  |  DECISION_TREE_CLASSIFIER  |  74.14%  |  [0.735  0.7292 0.746  0.739  0.7342]  |  0.74 (+/- 0.01)  |  7.808  |  0.01236  |
|  5  |  GRADIENT_BOOSTING_CLASSIFIER  |  82.86%  |  [0.8278 0.8294 0.8238 0.823  0.8284]  |  0.83 (+/- 0.01)  |  100.8  |  0.06589  |
|  6  |  K_NEIGHBORS_CLASSIFIER  |  82.66%  |  [0.8632 0.8744 0.8694 0.864  0.8618]  |  0.87 (+/- 0.01)  |  0.006417  |  13.02  |
|  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  0.88 (+/- 0.01)  |  0.2095  |  0.004025  |
|  8  |  LOGISTIC_REGRESSION  |  87.75%  |  [0.8882 0.897  0.8878 0.8876 0.8818]  |  0.89 (+/- 0.01)  |  1.075  |  0.005046  |
|  9  |  MULTINOMIAL_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  0.87 (+/- 0.01)  |  0.01648  |  0.0088  |
|  10  |  NEAREST_CENTROID  |  84.65%  |  [0.8426 0.8546 0.8426 0.8522 0.8502]  |  0.85 (+/- 0.01)  |  0.01818  |  0.01677  |
|  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  88.07%  |  [0.8874 0.8966 0.8886 0.888  0.8846]  |  0.89 (+/- 0.01)  |  0.8581  |  0.003922  |
|  12  |  PERCEPTRON  |  80.64%  |  [0.8166 0.8264 0.8144 0.81   0.8102]  |  0.82 (+/- 0.01)  |  0.09105  |  0.007187  |
|  13  |  RANDOM_FOREST_CLASSIFIER  |  85.45%  |  [0.8504 0.8584 0.8488 0.8518 0.8568]  |  0.85 (+/- 0.01)  |  8.792  |  0.7235  |
|  14  |  RIDGE_CLASSIFIER  |  86.90%  |  [0.8838 0.8952 0.8892 0.882  0.8788]  |  0.89 (+/- 0.01)  |  0.5019  |  0.00815  |
|  15  |  MAJORITY_VOTING_CLASSIFIER  |  87.88%  |  [0.8882 0.8976 0.8896 0.8884 0.8846]  |  0.89 (+/- 0.01)  |  10.81  |  0.7945  |
|  16  |  SOFT_VOTING_CLASSIFIER  |  87.73%  |  [0.8852 0.8972 0.8906 0.8908 0.8856]  |  0.89 (+/- 0.01)  |  10.29  |  0.6372  |
|  17  |  STACKING_CLASSIFIER  |  88.29%  |  [0.8924 0.902  0.8934 0.8898 0.889 ]  |  0.89 (+/- 0.01)  |  93.17  |  0.6377  |


Best algorithm:
===> 17) STACKING_CLASSIFIER
		Accuracy score = 88.29%		Training time = 93.17		Test time = 0.6377



DONE!
Program finished. It took 6212.019875049591 seconds

Process finished with exit code 0
```