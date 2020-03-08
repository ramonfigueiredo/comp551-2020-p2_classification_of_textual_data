# Running all classifiers (just IMDB_REVIEWS dataset using multi-class classification)


### IMDB using Multi-Class Classification

#### FINAL CLASSIFICATION TABLE: IMDB_REVIEWS dataset (Multi-class classification)

| ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
| --- | ------------- | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
|  1  |  ADA_BOOST_CLASSIFIER  |  38.02%  |  [0.3792 0.379  0.374  0.3704 0.3746]  |  0.38 (+/- 0.01)  |  125.8  |  7.788  |
|  2  |  BERNOULLI_NB  |  37.03%  |  [0.377  0.389  0.3782 0.38   0.373 ]  |  0.38 (+/- 0.01)  |  0.04582  |  0.04338  |
|  3  |  COMPLEMENT_NB  |  37.34%  |  [0.3878 0.3942 0.3976 0.3938 0.3832]  |  0.39 (+/- 0.01)  |  0.03707  |  0.01888  |
|  4  |  DECISION_TREE_CLASSIFIER  |  30.82%  |  [0.3072 0.3132 0.2996 0.3066 0.3064]  |  0.31 (+/- 0.01)  |  6.834  |  0.01294  |
|  5  |  GRADIENT_BOOSTING_CLASSIFIER  |  37.88%  |  [0.3848 0.379  0.3696 0.3692 0.3728]  |  0.38 (+/- 0.01)  |  872.7  |  0.5021  |
|  6  |  K_NEIGHBORS_CLASSIFIER  |  37.26%  |  [0.3822 0.3916 0.3842 0.386  0.388 ]  |  0.39 (+/- 0.01)  |  0.006473  |  14.71  |
|  7  |  LINEAR_SVC  |  40.80%  |  [0.41   0.4206 0.4064 0.3992 0.4088]  |  0.41 (+/- 0.01)  |  0.5438  |  0.01863  |
|  8  |  LOGISTIC_REGRESSION  |  42.04%  |  [0.4282 0.4334 0.4152 0.4194 0.4218]  |  0.42 (+/- 0.01)  |  9.81  |  0.01971  |
|  9  |  MULTINOMIAL_NB  |  37.82%  |  [0.389  0.3928 0.3918 0.3942 0.386 ]  |  0.39 (+/- 0.01)  |  0.03475  |  0.01914  |
|  10  |  NEAREST_CENTROID  |  37.33%  |  [0.3872 0.3786 0.3894 0.3672 0.3782]  |  0.38 (+/- 0.02)  |  0.02605  |  0.03179  |
|  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  41.81%  |  [0.4172 0.4284 0.4096 0.409  0.4164]  |  0.42 (+/- 0.01)  |  0.5241  |  0.02055  |
|  12  |  PERCEPTRON  |  31.60%  |  [0.3364 0.3094 0.3268 0.298  0.2964]  |  0.31 (+/- 0.03)  |  0.4149  |  0.0195  |
|  13  |  RANDOM_FOREST_CLASSIFIER  |  37.72%  |  [0.369  0.3796 0.3768 0.3728 0.3718]  |  0.37 (+/- 0.01)  |  9.596  |  0.708  |
|  14  |  RIDGE_CLASSIFIER  |  38.55%  |  [0.4036 0.4074 0.402  0.3954 0.4   ]  |  0.40 (+/- 0.01)  |  2.934  |  0.04121  |

![FINAL CLASSIFICATION TABLE: IMDB_REVIEWS dataset (Multi-class classification)](https://github.com/ramonfigueiredopessoa/comp551-2020-p2_classification_of_textual_data/blob/master/code/results/all_classifiers_just_imdb_using_multi_class_classification/IMDB_REVIEWS_multi_class_classification.png)

#### Computer settings used to run

* Operating system: Ubuntu 18.04.4 LTS (64-bit)
* Processor: Intel® Core™ i7-7700 CPU @ 3.60GHz × 8 
* Memory: 32 GB

#### All logs

```
python /comp551-2020-p2_classification_of_textual_data/code/main.py --dataset IMDB_REVIEWS --use_imdb_multi_class_labels --run_cross_validation --report --all_metrics --confusion_matrix --plot_accurary_and_time_together
03/08/2020 10:51:16 AM - INFO - Program started...
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
  -v, --version         show program's version number and exit

COMP 551 (001/002), Applied Machine Learning, Winter 2020, McGill University.
None
==================================================================================================================================

Running with options: 
	Dataset = IMDB_REVIEWS
	ML algorithm list (If ml_algorithm_list is not provided, all ML algorithms will be executed) = None
	Use classifiers with default parameters. Default: False = Use classifiers with best parameters found using grid search. False
	Read dataset without shuffle data = False
	The number of CPUs to use to do the computation. If the provided number is negative or greater than the number of available CPUs, the system will use all the available CPUs. Default: -1 (-1 == all CPUs) = -1
	Run cross validation. Default: False = True
	Number of cross validation folds. Default: 5 = 5
	Use just the miniproject classifiers (1. LogisticRegression, 2. DecisionTreeClassifier, 3. LinearSVC, 4. AdaBoostClassifier, 5. RandomForestClassifier) =  False
	TWENTY_NEWS_GROUPS dataset using some categories (alt.atheism, talk.religion.misc, comp.graphics, sci.space) = False
	Do not remove newsgroup information that is easily overfit (headers, footers, quotes) = False
	Use IMDB multi-class labels (review score: 1, 2, 3, 4, 7, 8, 9, 10). If --use_imdb_multi_class_labels is False, the system uses binary classification (0 = neg and 1 = pos). Default: False = True
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
==================================================================================================================================

Loading IMDB_REVIEWS dataset:

===> Reading files from /home/ets-crchum/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/train/neg

===> Reading files from /home/ets-crchum/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/train/pos

===> Reading files from /home/ets-crchum/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/test/neg

===> Reading files from /home/ets-crchum/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/test/pos
data loaded
25000 documents - 33.133MB (training set)
25000 documents - 32.351MB (test set)

Extracting features from the training data using a vectorizer
done in 2.996497s at 11.057MB/s
n_samples: 25000, n_features: 74170

Extracting features from the test data using the same vectorizer
done in 2.920075s at 11.079MB/s
n_samples: 25000, n_features: 74170

	==> Using JSON with best parameters (selected using grid search) to the ADA_BOOST_CLASSIFIER classifier (multi-class classification) and IMDB_REVIEWS dataset ===> JSON in dictionary format: {'learning_rate': 0.1, 'n_estimators': 500}
	 AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=0.1,
                   n_estimators=500, random_state=0)
	==> Using JSON with best parameters (selected using grid search) to the BERNOULLI_NB classifier (multi-class classification) and IMDB_REVIEWS dataset ===> JSON in dictionary format: {'alpha': 0.5, 'binarize': 0.0001, 'fit_prior': True}
	 BernoulliNB(alpha=0.5, binarize=0.0001, class_prior=None, fit_prior=True)
	==> Using JSON with best parameters (selected using grid search) to the COMPLEMENT_NB classifier (multi-class classification) and IMDB_REVIEWS dataset ===> JSON in dictionary format: {'alpha': 0.5, 'fit_prior': False, 'norm': False}
	 ComplementNB(alpha=0.5, class_prior=None, fit_prior=False, norm=False)
	==> Using JSON with best parameters (selected using grid search) to the DECISION_TREE_CLASSIFIER classifier (multi-class classification) and IMDB_REVIEWS dataset ===> JSON in dictionary format: {'criterion': 'entropy', 'min_samples_split': 250, 'splitter': 'random'}
	 DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='entropy',
                       max_depth=None, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=250,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=0, splitter='random')
	==> Using JSON with best parameters (selected using grid search) to the GRADIENT_BOOSTING_CLASSIFIER classifier (multi-class classification) and IMDB_REVIEWS dataset ===> JSON in dictionary format: {'learning_rate': 0.1, 'n_estimators': 200}
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
	==> Using JSON with best parameters (selected using grid search) to the K_NEIGHBORS_CLASSIFIER classifier (multi-class classification) and IMDB_REVIEWS dataset ===> JSON in dictionary format: {'leaf_size': 5, 'metric': 'euclidean', 'n_neighbors': 50, 'weights': 'distance'}
	 KNeighborsClassifier(algorithm='auto', leaf_size=5, metric='euclidean',
                     metric_params=None, n_jobs=-1, n_neighbors=50, p=2,
                     weights='distance')
	==> Using JSON with best parameters (selected using grid search) to the LINEAR_SVC classifier (multi-class classification) and IMDB_REVIEWS dataset ===> JSON in dictionary format: {'C': 0.01, 'multi_class': 'crammer_singer', 'tol': 0.001}
	 LinearSVC(C=0.01, class_weight=None, dual=True, fit_intercept=True,
          intercept_scaling=1, loss='squared_hinge', max_iter=1000,
          multi_class='crammer_singer', penalty='l2', random_state=0, tol=0.001,
          verbose=False)
	==> Using JSON with best parameters (selected using grid search) to the LOGISTIC_REGRESSION classifier (multi-class classification) and IMDB_REVIEWS dataset ===> JSON in dictionary format: {'C': 1, 'tol': 0.001}
	 LogisticRegression(C=1, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=-1, penalty='l2', random_state=0,
                   solver='lbfgs', tol=0.001, verbose=False, warm_start=False)
	==> Using JSON with best parameters (selected using grid search) to the MULTINOMIAL_NB classifier (multi-class classification) and IMDB_REVIEWS dataset ===> JSON in dictionary format: {'alpha': 0.1, 'fit_prior': True}
	 MultinomialNB(alpha=0.1, class_prior=None, fit_prior=True)
	==> Using JSON with best parameters (selected using grid search) to the NEAREST_CENTROID classifier (multi-class classification) and IMDB_REVIEWS dataset ===> JSON in dictionary format: {'metric': 'cosine'}
	 NearestCentroid(metric='cosine', shrink_threshold=None)
	==> Using JSON with best parameters (selected using grid search) to the PASSIVE_AGGRESSIVE_CLASSIFIER classifier (multi-class classification) and IMDB_REVIEWS dataset ===> JSON in dictionary format: {'C': 0.01, 'early_stopping': True, 'tol': 0.0001, 'validation_fraction': 0.01}
	 PassiveAggressiveClassifier(C=0.01, average=False, class_weight=None,
                            early_stopping=True, fit_intercept=True,
                            loss='hinge', max_iter=1000, n_iter_no_change=5,
                            n_jobs=-1, random_state=0, shuffle=True, tol=0.0001,
                            validation_fraction=0.01, verbose=False,
                            warm_start=False)
	==> Using JSON with best parameters (selected using grid search) to the PERCEPTRON classifier (multi-class classification) and IMDB_REVIEWS dataset ===> JSON in dictionary format: {'early_stopping': True, 'max_iter': 100, 'n_iter_no_change': 3, 'penalty': 'l2', 'tol': 0.0001, 'validation_fraction': 0.01}
	 Perceptron(alpha=0.0001, class_weight=None, early_stopping=True, eta0=1.0,
           fit_intercept=True, max_iter=100, n_iter_no_change=3, n_jobs=-1,
           penalty='l2', random_state=0, shuffle=True, tol=0.0001,
           validation_fraction=0.01, verbose=False, warm_start=False)
	==> Using JSON with best parameters (selected using grid search) to the RANDOM_FOREST_CLASSIFIER classifier (multi-class classification) and IMDB_REVIEWS dataset ===> JSON in dictionary format: {'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 200}
	 RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=2, min_samples_split=5,
                       min_weight_fraction_leaf=0.0, n_estimators=200,
                       n_jobs=-1, oob_score=False, random_state=0,
                       verbose=False, warm_start=False)
	==> Using JSON with best parameters (selected using grid search) to the RIDGE_CLASSIFIER classifier (multi-class classification) and IMDB_REVIEWS dataset ===> JSON in dictionary format: {'alpha': 1.0, 'tol': 0.001}
	 RidgeClassifier(alpha=1.0, class_weight=None, copy_X=True, fit_intercept=True,
                max_iter=None, normalize=False, random_state=0, solver='auto',
                tol=0.001)
================================================================================
Classifier.ADA_BOOST_CLASSIFIER
________________________________________________________________________________
Training: 
AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=0.1,
                   n_estimators=500, random_state=0)
train time: 125.801s
test time:  7.788s
accuracy:   0.380


cross validation:
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/metrics/_classification.py:1272: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/metrics/_classification.py:1272: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/metrics/_classification.py:1272: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
	accuracy: 5-fold cross validation: [0.3792 0.379  0.374  0.3704 0.3746]
	test accuracy: 5-fold cross validation accuracy: 0.38 (+/- 0.01)


===> Classification Report:

              precision    recall  f1-score   support

           1       0.46      0.79      0.58      5022
           2       0.23      0.00      0.00      2302
           3       0.32      0.02      0.03      2541
           4       0.27      0.24      0.25      2635
           7       0.27      0.11      0.15      2307
           8       0.25      0.10      0.14      2850
           9       0.00      0.00      0.00      2344
          10       0.37      0.87      0.51      4999

    accuracy                           0.38     25000
   macro avg       0.27      0.27      0.21     25000
weighted avg       0.30      0.38      0.28     25000



===> Classification Metrics:

accuracy classification score
	accuracy score:  0.3802
	accuracy score (normalize=False):  9505

compute the precision
	precision score (average=macro):  0.2704729826951742
	precision score (average=micro):  0.3802
	precision score (average=weighted):  0.3009704255557748
	precision score (average=None):  [0.46052785 0.23076923 0.31690141 0.26638655 0.27203482 0.25114155
 0.         0.36602245]
	precision score (average=None, zero_division=1):  [0.46052785 0.23076923 0.31690141 0.26638655 0.27203482 0.25114155
 0.         0.36602245]

compute the precision
	recall score (average=macro):  0.2650975205063165
	recall score (average=micro):  0.3802
	recall score (average=weighted):  0.3802
	recall score (average=None):  [0.78872959 0.00130321 0.01770956 0.24060721 0.10836584 0.09649123
 0.         0.86757351]
	recall score (average=None, zero_division=1):  [0.78872959 0.00130321 0.01770956 0.24060721 0.10836584 0.09649123
 0.         0.86757351]

compute the F1 score, also known as balanced F-score or F-measure
	f1 score (average=macro):  0.20996757517492176
	f1 score (average=micro):  0.3802
	f1 score (average=weighted):  0.2802558516799662
	f1 score (average=None):  [0.58151655 0.00259179 0.03354454 0.25284148 0.1549907  0.13941698
 0.         0.51483856]

compute the F-beta score
	f beta score (average=macro):  0.2068568306472644
	f beta score (average=micro):  0.3802
	f beta score (average=weighted):  0.26005799229290627
	f beta score (average=None):  [0.50233349 0.00637213 0.07237054 0.26079803 0.20892529 0.19017981
 0.         0.41387537]

compute the average Hamming loss
	hamming loss:  0.6198

jaccard similarity coefficient score
	jaccard score (average=macro):  0.1348275630701743
	jaccard score (average=None):  [0.40995653 0.00129758 0.01705838 0.14471582 0.08400538 0.07493188
 0.         0.34665494]

confusion matrix:
[[3961    2   28  299   27   20    0  685]
 [1482    3   25  303   38   20    0  431]
 [1231    5   45  558   72   57    0  573]
 [ 967    3   36  634  128  101    0  766]
 [ 268    0    4  259  250  278    0 1248]
 [ 231    0    2  161  216  275    0 1965]
 [ 153    0    2   75   98  172    0 1844]
 [ 308    0    0   91   90  172    1 4337]]

================================================================================
Classifier.BERNOULLI_NB
________________________________________________________________________________
Training: 
BernoulliNB(alpha=0.5, binarize=0.0001, class_prior=None, fit_prior=True)
train time: 0.046s
test time:  0.043s
accuracy:   0.370


cross validation:
	accuracy: 5-fold cross validation: [0.377  0.389  0.3782 0.38   0.373 ]
	test accuracy: 5-fold cross validation accuracy: 0.38 (+/- 0.01)
dimensionality: 74170
density: 1.000000



===> Classification Report:

              precision    recall  f1-score   support

           1       0.41      0.82      0.55      5022
           2       0.18      0.02      0.04      2302
           3       0.24      0.11      0.15      2541
           4       0.27      0.18      0.22      2635
           7       0.24      0.14      0.18      2307
           8       0.25      0.15      0.19      2850
           9       0.21      0.05      0.08      2344
          10       0.42      0.69      0.53      4999

    accuracy                           0.37     25000
   macro avg       0.28      0.27      0.24     25000
weighted avg       0.31      0.37      0.30     25000



===> Classification Metrics:

accuracy classification score
	accuracy score:  0.37032
	accuracy score (normalize=False):  9258

compute the precision
	precision score (average=macro):  0.2779483104445877
	precision score (average=micro):  0.37032
	precision score (average=weighted):  0.30714043090090976
	precision score (average=None):  [0.41326633 0.18148148 0.24039321 0.26506024 0.24235808 0.24913694
 0.2078853  0.4240049 ]
	precision score (average=None, zero_division=1):  [0.41326633 0.18148148 0.24039321 0.26506024 0.24235808 0.24913694
 0.2078853  0.4240049 ]

compute the precision
	recall score (average=macro):  0.2709909834670155
	recall score (average=micro):  0.37032
	recall score (average=weighted):  0.37032
	recall score (average=None):  [0.81879729 0.02128584 0.10586383 0.18368121 0.1443433  0.15192982
 0.04948805 0.69253851]
	recall score (average=None, zero_division=1):  [0.81879729 0.02128584 0.10586383 0.18368121 0.1443433  0.15192982
 0.04948805 0.69253851]

compute the F1 score, also known as balanced F-score or F-measure
	f1 score (average=macro):  0.24087350908688981
	f1 score (average=micro):  0.37032
	f1 score (average=weighted):  0.30254631125288545
	f1 score (average=None):  [0.54929201 0.03810264 0.14699454 0.21699171 0.1809291  0.18875327
 0.07994487 0.52597995]

compute the F-beta score
	f beta score (average=macro):  0.24837004549984215
	f beta score (average=micro):  0.37032
	f beta score (average=weighted):  0.29262620415103124
	f beta score (average=None):  [0.45870332 0.07244234 0.19167735 0.24348526 0.21337947 0.22087329
 0.12674825 0.45965108]

compute the average Hamming loss
	hamming loss:  0.62968

jaccard similarity coefficient score
	jaccard score (average=macro):  0.1501538108234517
	jaccard score (average=None):  [0.3786372  0.01942132 0.07932763 0.12169977 0.09946237 0.10421179
 0.04163676 0.35683364]

confusion matrix:
[[4112   91  259  251   36   38   12  223]
 [1527   49  208  263   40   24    6  185]
 [1379   55  269  397  101   59   19  262]
 [1169   37  204  484  186  136   27  392]
 [ 448   10   70  175  333  333   68  870]
 [ 417   11   61  121  289  433  120 1398]
 [ 269    5   26   58  180  317  116 1373]
 [ 629   12   22   77  209  398  190 3462]]

================================================================================
Classifier.COMPLEMENT_NB
________________________________________________________________________________
Training: 
ComplementNB(alpha=0.5, class_prior=None, fit_prior=False, norm=False)
train time: 0.037s
test time:  0.019s
accuracy:   0.373


cross validation:
	accuracy: 5-fold cross validation: [0.3878 0.3942 0.3976 0.3938 0.3832]
	test accuracy: 5-fold cross validation accuracy: 0.39 (+/- 0.01)
dimensionality: 74170
density: 1.000000



===> Classification Report:

              precision    recall  f1-score   support

           1       0.39      0.90      0.55      5022
           2       0.13      0.02      0.03      2302
           3       0.19      0.05      0.07      2541
           4       0.28      0.14      0.18      2635
           7       0.26      0.12      0.17      2307
           8       0.24      0.16      0.19      2850
           9       0.17      0.04      0.06      2344
          10       0.45      0.69      0.55      4999

    accuracy                           0.37     25000
   macro avg       0.26      0.26      0.23     25000
weighted avg       0.30      0.37      0.29     25000



===> Classification Metrics:

accuracy classification score
	accuracy score:  0.3734
	accuracy score (normalize=False):  9335

compute the precision
	precision score (average=macro):  0.2633570674940663
	precision score (average=micro):  0.3734
	precision score (average=weighted):  0.2965870398222459
	precision score (average=None):  [0.39263964 0.12650602 0.19243421 0.27773527 0.25604297 0.23582539
 0.17315175 0.45252128]
	precision score (average=None, zero_division=1):  [0.39263964 0.12650602 0.19243421 0.27773527 0.25604297 0.23582539
 0.17315175 0.45252128]

compute the precision
	recall score (average=macro):  0.2648358816608535
	recall score (average=micro):  0.3734
	recall score (average=weighted):  0.3734
	recall score (average=None):  [0.89864596 0.018245   0.04604486 0.13776091 0.12397052 0.16491228
 0.03796928 0.69113823]
	recall score (average=None, zero_division=1):  [0.89864596 0.018245   0.04604486 0.13776091 0.12397052 0.16491228
 0.03796928 0.69113823]

compute the F1 score, also known as balanced F-score or F-measure
	f1 score (average=macro):  0.22590494965234917
	f1 score (average=micro):  0.3734
	f1 score (average=weighted):  0.29242955204878873
	f1 score (average=None):  [0.54650036 0.03189066 0.0743093  0.18417047 0.16705607 0.19409457
 0.06228132 0.54693684]

compute the F-beta score
	f beta score (average=macro):  0.23302814534539784
	f beta score (average=micro):  0.3734
	f beta score (average=weighted):  0.28140861115643284
	f beta score (average=None):  [0.44246833 0.05785124 0.11763523 0.23082793 0.21107011 0.21715025
 0.10113636 0.48608571]

compute the average Hamming loss
	hamming loss:  0.6266

jaccard similarity coefficient score
	jaccard score (average=macro):  0.14242114912296494
	jaccard score (average=None):  [0.37598934 0.0162037  0.03858839 0.10142498 0.09114085 0.1074777
 0.03214157 0.37640266]

confusion matrix:
[[4513   54   53  106   42   54   19  181]
 [1818   42   73  120   50   46   13  140]
 [1711   56  117  229   95  108   25  200]
 [1355   51  150  363  170  203   62  281]
 [ 505   34   75  180  286  405   91  731]
 [ 498   38   54  134  233  470  110 1313]
 [ 352   18   43   67  108  333   89 1334]
 [ 742   39   43  108  133  374  105 3455]]

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
train time: 6.834s
test time:  0.013s
accuracy:   0.308


cross validation:
	accuracy: 5-fold cross validation: [0.3072 0.3132 0.2996 0.3066 0.3064]
	test accuracy: 5-fold cross validation accuracy: 0.31 (+/- 0.01)


===> Classification Report:

              precision    recall  f1-score   support

           1       0.41      0.66      0.51      5022
           2       0.11      0.02      0.03      2302
           3       0.16      0.08      0.11      2541
           4       0.17      0.16      0.17      2635
           7       0.14      0.13      0.13      2307
           8       0.19      0.14      0.16      2850
           9       0.15      0.04      0.07      2344
          10       0.37      0.58      0.45      4999

    accuracy                           0.31     25000
   macro avg       0.21      0.23      0.20     25000
weighted avg       0.25      0.31      0.26     25000



===> Classification Metrics:

accuracy classification score
	accuracy score:  0.30824
	accuracy score (normalize=False):  7706

compute the precision
	precision score (average=macro):  0.21271217352747812
	precision score (average=micro):  0.30824
	precision score (average=weighted):  0.24970205983596508
	precision score (average=None):  [0.41108374 0.1092233  0.15671642 0.17178159 0.14203178 0.18653101
 0.15239478 0.37193478]
	precision score (average=None, zero_division=1):  [0.41108374 0.1092233  0.15671642 0.17178159 0.14203178 0.18653101
 0.15239478 0.37193478]

compute the precision
	recall score (average=macro):  0.22721327209350345
	recall score (average=micro):  0.30824
	recall score (average=weighted):  0.30824
	recall score (average=None):  [0.66467543 0.01954822 0.08264463 0.16356736 0.12787169 0.13508772
 0.04479522 0.5795159 ]
	recall score (average=None, zero_division=1):  [0.66467543 0.01954822 0.08264463 0.16356736 0.12787169 0.13508772
 0.04479522 0.5795159 ]

compute the F1 score, also known as balanced F-score or F-measure
	f1 score (average=macro):  0.203817410048659
	f1 score (average=micro):  0.30824
	f1 score (average=weighted):  0.26113236781842686
	f1 score (average=None):  [0.50798965 0.03316139 0.10821953 0.16757387 0.13458029 0.15669516
 0.06923838 0.45308101]

compute the F-beta score
	f beta score (average=macro):  0.20260428632590213
	f beta score (average=micro):  0.30824
	f beta score (average=weighted):  0.24842368971188364
	f beta score (average=None):  [0.44504293 0.05696203 0.13289457 0.1700734  0.13895431 0.17332973
 0.10294118 0.40063615]

compute the average Hamming loss
	hamming loss:  0.69176

jaccard similarity coefficient score
	jaccard score (average=macro):  0.12398668984714059
	jaccard score (average=None):  [0.34047328 0.01686025 0.05720512 0.09144918 0.07214478 0.08500773
 0.03586066 0.29289253]

confusion matrix:
[[3338  116  259  399  189  131   43  547]
 [1226   45  167  270  148  100   28  318]
 [1070   48  210  400  199  154   40  420]
 [ 909   50  240  431  266  190   46  503]
 [ 374   44  131  275  295  315   83  790]
 [ 406   37  102  268  348  385  128 1176]
 [ 268   20   85  176  236  316  105 1138]
 [ 529   52  146  290  396  473  216 2897]]

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
train time: 872.650s
test time:  0.502s
accuracy:   0.379


cross validation:
	accuracy: 5-fold cross validation: [0.3848 0.379  0.3696 0.3692 0.3728]
	test accuracy: 5-fold cross validation accuracy: 0.38 (+/- 0.01)


===> Classification Report:

              precision    recall  f1-score   support

           1       0.49      0.76      0.59      5022
           2       0.20      0.06      0.09      2302
           3       0.25      0.10      0.14      2541
           4       0.28      0.21      0.24      2635
           7       0.25      0.16      0.19      2307
           8       0.22      0.18      0.19      2850
           9       0.16      0.04      0.06      2344
          10       0.41      0.75      0.53      4999

    accuracy                           0.38     25000
   macro avg       0.28      0.28      0.26     25000
weighted avg       0.32      0.38      0.32     25000



===> Classification Metrics:

accuracy classification score
	accuracy score:  0.37876
	accuracy score (normalize=False):  9469

compute the precision
	precision score (average=macro):  0.28118486779217305
	precision score (average=micro):  0.37876
	precision score (average=weighted):  0.3153639690043761
	precision score (average=None):  [0.48563837 0.20364742 0.24598394 0.27834525 0.25329632 0.2159383
 0.15608919 0.41054016]
	precision score (average=None, zero_division=1):  [0.48563837 0.20364742 0.24598394 0.27834525 0.25329632 0.2159383
 0.15608919 0.41054016]

compute the precision
	recall score (average=macro):  0.2811407664581688
	recall score (average=micro):  0.37876
	recall score (average=weighted):  0.37876
	recall score (average=None):  [0.76423736 0.05821025 0.09641873 0.20683112 0.15821413 0.17684211
 0.03882253 0.74954991]
	recall score (average=None, zero_division=1):  [0.76423736 0.05821025 0.09641873 0.20683112 0.15821413 0.17684211
 0.03882253 0.74954991]

compute the F1 score, also known as balanced F-score or F-measure
	f1 score (average=macro):  0.2552734129226687
	f1 score (average=micro):  0.37876
	f1 score (average=weighted):  0.3187822487398697
	f1 score (average=None):  [0.59388781 0.09054054 0.13853548 0.23731766 0.19477054 0.19444444
 0.06217971 0.53051111]

compute the F-beta score
	f beta score (average=macro):  0.2611615242583102
	f beta score (average=micro):  0.37876
	f beta score (average=weighted):  0.3080727031034825
	f beta score (average=None):  [0.52383032 0.13579246 0.18773946 0.26034203 0.2261182  0.20679468
 0.09730539 0.45136965]

compute the average Hamming loss
	hamming loss:  0.62124

jaccard similarity coefficient score
	jaccard score (average=macro):  0.16094066084387398
	jaccard score (average=None):  [0.42236162 0.04741684 0.07442284 0.13463439 0.1078924  0.10769231
 0.03208745 0.36101744]

confusion matrix:
[[3838  156  159  202   67  100   21  479]
 [1306  134  171  208   62   91   25  305]
 [1057  124  245  414  121  151   35  394]
 [ 815  110  219  545  198  236   41  471]
 [ 226   42   77  226  365  454   75  842]
 [ 223   39   52  166  321  504  129 1416]
 [ 146   17   32   89  147  349   91 1473]
 [ 292   36   41  108  160  449  166 3747]]

================================================================================
Classifier.K_NEIGHBORS_CLASSIFIER
________________________________________________________________________________
Training: 
KNeighborsClassifier(algorithm='auto', leaf_size=5, metric='euclidean',
                     metric_params=None, n_jobs=-1, n_neighbors=50, p=2,
                     weights='distance')
train time: 0.006s
test time:  14.708s
accuracy:   0.373


cross validation:
	accuracy: 5-fold cross validation: [0.3822 0.3916 0.3842 0.386  0.388 ]
	test accuracy: 5-fold cross validation accuracy: 0.39 (+/- 0.01)


===> Classification Report:

              precision    recall  f1-score   support

           1       0.38      0.91      0.53      5022
           2       0.23      0.02      0.04      2302
           3       0.25      0.04      0.07      2541
           4       0.35      0.09      0.14      2635
           7       0.28      0.06      0.10      2307
           8       0.29      0.10      0.14      2850
           9       0.15      0.02      0.03      2344
          10       0.40      0.78      0.53      4999

    accuracy                           0.37     25000
   macro avg       0.29      0.25      0.20     25000
weighted avg       0.31      0.37      0.27     25000



===> Classification Metrics:

accuracy classification score
	accuracy score:  0.3726
	accuracy score (normalize=False):  9315

compute the precision
	precision score (average=macro):  0.28964641955039383
	precision score (average=micro):  0.3726
	precision score (average=weighted):  0.3107151509008421
	precision score (average=None):  [0.37598555 0.2254902  0.24937656 0.34718101 0.2848723  0.28797468
 0.14942529 0.39686578]
	precision score (average=None, zero_division=1):  [0.37598555 0.2254902  0.24937656 0.34718101 0.2848723  0.28797468
 0.14942529 0.39686578]

compute the precision
	recall score (average=macro):  0.25189583627423373
	recall score (average=micro):  0.3726
	recall score (average=weighted):  0.3726
	recall score (average=None):  [0.91158901 0.01998262 0.03935458 0.08880455 0.06285219 0.09578947
 0.01663823 0.78015603]
	recall score (average=None, zero_division=1):  [0.91158901 0.01998262 0.03935458 0.08880455 0.06285219 0.09578947
 0.01663823 0.78015603]

compute the F1 score, also known as balanced F-score or F-measure
	f1 score (average=macro):  0.19766260495298899
	f1 score (average=micro):  0.3726
	f1 score (average=weighted):  0.2660418037708692
	f1 score (average=None):  [0.53238749 0.03671189 0.06798097 0.14143246 0.10298295 0.14375987
 0.02994242 0.52610279]

compute the F-beta score
	f beta score (average=macro):  0.21375337830852037
	f beta score (average=micro):  0.3726
	f beta score (average=weighted):  0.2600042038841227
	f beta score (average=None):  [0.4260507  0.07376523 0.12062726 0.21947102 0.1669353  0.20551039
 0.05755608 0.44011104]

compute the average Hamming loss
	hamming loss:  0.6274

jaccard similarity coefficient score
	jaccard score (average=macro):  0.12457748025796672
	jaccard score (average=None):  [0.36275753 0.01869919 0.03518649 0.07609756 0.05428678 0.07744681
 0.01519875 0.35694673]

confusion matrix:
[[4578   36   25   34   10   18    2  319]
 [1853   46   37   63   15   19   11  258]
 [1827   31  100  107   39   46   13  378]
 [1521   42  116  234   67   97   21  537]
 [ 637   20   60  107  145  200   51 1087]
 [ 580   14   31   71  123  273   61 1697]
 [ 411    7   15   27   55  139   39 1651]
 [ 769    8   17   31   55  156   63 3900]]

================================================================================
Classifier.LINEAR_SVC
________________________________________________________________________________
Training: 
LinearSVC(C=0.01, class_weight=None, dual=True, fit_intercept=True,
          intercept_scaling=1, loss='squared_hinge', max_iter=1000,
          multi_class='crammer_singer', penalty='l2', random_state=0, tol=0.001,
          verbose=False)
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/svm/_base.py:947: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  "the number of iterations.", ConvergenceWarning)
train time: 0.544s
test time:  0.019s
accuracy:   0.408


cross validation:
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/svm/_base.py:947: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  "the number of iterations.", ConvergenceWarning)
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/svm/_base.py:947: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  "the number of iterations.", ConvergenceWarning)
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/svm/_base.py:947: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  "the number of iterations.", ConvergenceWarning)
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/svm/_base.py:947: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  "the number of iterations.", ConvergenceWarning)
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/svm/_base.py:947: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  "the number of iterations.", ConvergenceWarning)
	accuracy: 5-fold cross validation: [0.41   0.4206 0.4064 0.3992 0.4088]
	test accuracy: 5-fold cross validation accuracy: 0.41 (+/- 0.01)
dimensionality: 74170
density: 0.761615



===> Classification Report:

              precision    recall  f1-score   support

           1       0.47      0.88      0.62      5022
           2       0.14      0.04      0.06      2302
           3       0.22      0.09      0.13      2541
           4       0.31      0.23      0.27      2635
           7       0.27      0.20      0.23      2307
           8       0.27      0.15      0.19      2850
           9       0.20      0.06      0.09      2344
          10       0.48      0.76      0.59      4999

    accuracy                           0.41     25000
   macro avg       0.29      0.30      0.27     25000
weighted avg       0.33      0.41      0.34     25000



===> Classification Metrics:

accuracy classification score
	accuracy score:  0.40796
	accuracy score (normalize=False):  10199

compute the precision
	precision score (average=macro):  0.29434164418715825
	precision score (average=micro):  0.40796
	precision score (average=weighted):  0.3323495119992305
	precision score (average=None):  [0.47259323 0.13554217 0.21877934 0.31329598 0.27390029 0.26607257
 0.19599428 0.4785553 ]
	precision score (average=None, zero_division=1):  [0.47259323 0.13554217 0.21877934 0.31329598 0.27390029 0.26607257
 0.19599428 0.4785553 ]

compute the precision
	recall score (average=macro):  0.3019759808524948
	recall score (average=micro):  0.40796
	recall score (average=weighted):  0.40796
	recall score (average=None):  [0.88072481 0.03909644 0.09169618 0.23339658 0.20242739 0.14666667
 0.0584471  0.76335267]
	recall score (average=None, zero_division=1):  [0.88072481 0.03909644 0.09169618 0.23339658 0.20242739 0.14666667
 0.0584471  0.76335267]

compute the F1 score, also known as balanced F-score or F-measure
	f1 score (average=macro):  0.2715977772527003
	f1 score (average=micro):  0.40796
	f1 score (average=weighted):  0.3396016866324532
	f1 score (average=None):  [0.61511717 0.0606878  0.12922906 0.26750761 0.2328016  0.18909749
 0.09004272 0.58829877]

compute the F-beta score
	f beta score (average=macro):  0.27640126006058496
	f beta score (average=micro):  0.40796
	f beta score (average=weighted):  0.3269020377955
	f beta score (average=None):  [0.52086768 0.0907624  0.17129834 0.29322018 0.25583434 0.22881541
 0.13326848 0.51714324]

compute the average Hamming loss
	hamming loss:  0.59204

jaccard similarity coefficient score
	jaccard score (average=macro):  0.17487173611156706
	jaccard score (average=None):  [0.4441655  0.03129346 0.06907797 0.15440623 0.13173484 0.10442168
 0.04714384 0.41673037]

confusion matrix:
[[4423   92  115  143   40   31   17  161]
 [1612   90  138  193   77   31   17  144]
 [1308  150  233  410  133   77   35  195]
 [ 936  145  287  615  238  143   45  226]
 [ 241   56  119  256  467  348  146  674]
 [ 247   41   81  186  380  418  167 1330]
 [ 188   36   40   69  184  262  137 1428]
 [ 404   54   52   91  186  261  135 3816]]

================================================================================
Classifier.LOGISTIC_REGRESSION
________________________________________________________________________________
Training: 
LogisticRegression(C=1, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=-1, penalty='l2', random_state=0,
                   solver='lbfgs', tol=0.001, verbose=False, warm_start=False)
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
train time: 9.810s
test time:  0.020s
accuracy:   0.420


cross validation:
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
	accuracy: 5-fold cross validation: [0.4282 0.4334 0.4152 0.4194 0.4218]
	test accuracy: 5-fold cross validation accuracy: 0.42 (+/- 0.01)
dimensionality: 74170
density: 1.000000



===> Classification Report:

              precision    recall  f1-score   support

           1       0.51      0.84      0.64      5022
           2       0.20      0.04      0.07      2302
           3       0.26      0.12      0.16      2541
           4       0.32      0.29      0.31      2635
           7       0.31      0.22      0.26      2307
           8       0.26      0.23      0.25      2850
           9       0.21      0.04      0.07      2344
          10       0.48      0.77      0.59      4999

    accuracy                           0.42     25000
   macro avg       0.32      0.32      0.29     25000
weighted avg       0.36      0.42      0.36     25000



===> Classification Metrics:

accuracy classification score
	accuracy score:  0.42036
	accuracy score (normalize=False):  10509

compute the precision
	precision score (average=macro):  0.3189167349145974
	precision score (average=micro):  0.42036
	precision score (average=weighted):  0.3550807840819632
	precision score (average=None):  [0.51318359 0.20235756 0.26182137 0.32086451 0.30882353 0.2614897
 0.20631579 0.47647783]
	precision score (average=None, zero_division=1):  [0.51318359 0.20235756 0.26182137 0.32086451 0.30882353 0.2614897
 0.20631579 0.47647783]

compute the precision
	recall score (average=macro):  0.31978973439139713
	recall score (average=micro):  0.42036
	recall score (average=weighted):  0.42036
	recall score (average=None):  [0.83711669 0.0447437  0.11767021 0.29297913 0.21846554 0.23157895
 0.04180887 0.77395479]
	recall score (average=None, zero_division=1):  [0.83711669 0.0447437  0.11767021 0.29297913 0.21846554 0.23157895
 0.04180887 0.77395479]

compute the F1 score, also known as balanced F-score or F-measure
	f1 score (average=macro):  0.29239047349802016
	f1 score (average=micro):  0.42036
	f1 score (average=weighted):  0.3594305539270417
	f1 score (average=None):  [0.63629484 0.07328353 0.16236764 0.30628843 0.25590251 0.24562709
 0.0695282  0.58983154]

compute the F-beta score
	f beta score (average=macro):  0.29648308082656616
	f beta score (average=micro):  0.42036
	f beta score (average=weighted):  0.34664524218142806
	f beta score (average=None):  [0.55623181 0.1187183  0.21029681 0.31487071 0.2852292  0.25490499
 0.11545712 0.51615571]

compute the average Hamming loss
	hamming loss:  0.57964

jaccard similarity coefficient score
	jaccard score (average=macro):  0.18935544002434804
	jaccard score (average=None):  [0.46659267 0.03803545 0.08835697 0.1808386  0.14672489 0.14000849
 0.03601617 0.41827027]

confusion matrix:
[[4204  120  161  229   31   48    5  224]
 [1435  103  199  293   51   48   11  162]
 [1090  132  299  591  106   91   14  218]
 [ 771   87  306  772  203  209   23  264]
 [ 164   22   86  239  504  563   75  654]
 [ 164   18   40  144  405  660  121 1298]
 [ 117   11   21   65  155  446   98 1431]
 [ 247   16   30   73  177  459  128 3869]]

================================================================================
Classifier.MULTINOMIAL_NB
________________________________________________________________________________
Training: 
MultinomialNB(alpha=0.1, class_prior=None, fit_prior=True)
train time: 0.035s
test time:  0.019s
accuracy:   0.378


cross validation:
	accuracy: 5-fold cross validation: [0.389  0.3928 0.3918 0.3942 0.386 ]
	test accuracy: 5-fold cross validation accuracy: 0.39 (+/- 0.01)
dimensionality: 74170
density: 1.000000



===> Classification Report:

              precision    recall  f1-score   support

           1       0.38      0.90      0.54      5022
           2       0.29      0.01      0.02      2302
           3       0.26      0.03      0.06      2541
           4       0.32      0.15      0.21      2635
           7       0.30      0.09      0.14      2307
           8       0.24      0.16      0.20      2850
           9       0.23      0.01      0.03      2344
          10       0.42      0.74      0.54      4999

    accuracy                           0.38     25000
   macro avg       0.31      0.26      0.22     25000
weighted avg       0.33      0.38      0.28     25000



===> Classification Metrics:

accuracy classification score
	accuracy score:  0.37824
	accuracy score (normalize=False):  9456

compute the precision
	precision score (average=macro):  0.30708901562176777
	precision score (average=micro):  0.37824
	precision score (average=weighted):  0.3260138469455144
	precision score (average=None):  [0.38291755 0.28888889 0.26461538 0.32188498 0.30396476 0.24458531
 0.22794118 0.42191407]
	precision score (average=None, zero_division=1):  [0.38291755 0.28888889 0.26461538 0.32188498 0.30396476 0.24458531
 0.22794118 0.42191407]

compute the precision
	recall score (average=macro):  0.26345878570217296
	recall score (average=micro):  0.37824
	recall score (average=weighted):  0.37824
	recall score (average=None):  [0.90163282 0.01129453 0.03384494 0.15294118 0.08972692 0.16245614
 0.01322526 0.74254851]
	recall score (average=None, zero_division=1):  [0.90163282 0.01129453 0.03384494 0.15294118 0.08972692 0.16245614
 0.01322526 0.74254851]

compute the F1 score, also known as balanced F-score or F-measure
	f1 score (average=macro):  0.21544150166644488
	f1 score (average=micro):  0.37824
	f1 score (average=weighted):  0.2829215270141919
	f1 score (average=None):  [0.53754378 0.02173913 0.06001396 0.20735786 0.13855422 0.19523508
 0.025      0.53808799]

compute the F-beta score
	f beta score (average=macro):  0.22505583094794418
	f beta score (average=micro):  0.37824
	f beta score (average=weighted):  0.272264131959853
	f beta score (average=None):  [0.43270517 0.04883546 0.11195001 0.26363993 0.20572451 0.22212627
 0.05367036 0.46179493]

compute the average Hamming loss
	hamming loss:  0.62176

jaccard similarity coefficient score
	jaccard score (average=macro):  0.1360623818463968
	jaccard score (average=None):  [0.3675623  0.01098901 0.03093525 0.11567164 0.07443366 0.10817757
 0.01265823 0.36807139]

confusion matrix:
[[4528   18   38  121   20   47    1  249]
 [1839   26   53  143   30   29    1  181]
 [1753   17   86  244   56  106    3  276]
 [1440   14   76  403  101  193    7  401]
 [ 580    3   30  141  207  398   18  930]
 [ 537    5   21   96  146  463   37 1545]
 [ 389    1    5   40   62  312   31 1504]
 [ 759    6   16   64   59  345   38 3712]]

================================================================================
Classifier.NEAREST_CENTROID
________________________________________________________________________________
Training: 
NearestCentroid(metric='cosine', shrink_threshold=None)
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
train time: 0.026s
test time:  0.032s
accuracy:   0.373


cross validation:
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/neighbors/_nearest_centroid.py:145: UserWarning: Averaging for metrics other than euclidean and manhattan not supported. The average is set to be the mean.
  warnings.warn("Averaging for metrics other than "
	accuracy: 5-fold cross validation: [0.3872 0.3786 0.3894 0.3672 0.3782]
	test accuracy: 5-fold cross validation accuracy: 0.38 (+/- 0.02)


===> Classification Report:

              precision    recall  f1-score   support

           1       0.62      0.59      0.61      5022
           2       0.20      0.20      0.20      2302
           3       0.23      0.22      0.22      2541
           4       0.28      0.31      0.29      2635
           7       0.26      0.31      0.28      2307
           8       0.25      0.24      0.24      2850
           9       0.21      0.22      0.22      2344
          10       0.56      0.52      0.54      4999

    accuracy                           0.37     25000
   macro avg       0.33      0.33      0.33     25000
weighted avg       0.38      0.37      0.38     25000



===> Classification Metrics:

accuracy classification score
	accuracy score:  0.37332
	accuracy score (normalize=False):  9333

compute the precision
	precision score (average=macro):  0.3265426711646353
	precision score (average=micro):  0.37332
	precision score (average=weighted):  0.38058616067675366
	precision score (average=None):  [0.62353188 0.20397112 0.23048327 0.27760892 0.25949821 0.24533138
 0.21095666 0.56095993]
	precision score (average=None, zero_division=1):  [0.62353188 0.20397112 0.23048327 0.27760892 0.25949821 0.24533138
 0.21095666 0.56095993]

compute the precision
	recall score (average=macro):  0.3265819653990282
	recall score (average=micro):  0.37332
	recall score (average=weighted):  0.37332
	recall score (average=None):  [0.59199522 0.196351   0.21959858 0.31195446 0.31382748 0.23508772
 0.22013652 0.52370474]
	recall score (average=None, zero_division=1):  [0.59199522 0.196351   0.21959858 0.31195446 0.31382748 0.23508772
 0.22013652 0.52370474]

compute the F1 score, also known as balanced F-score or F-measure
	f1 score (average=macro):  0.325932995438585
	f1 score (average=micro):  0.37332
	f1 score (average=weighted):  0.3763583062121627
	f1 score (average=None):  [0.60735444 0.20008853 0.22490931 0.29378127 0.28408868 0.24010034
 0.21544885 0.54169253]

compute the F-beta score
	f beta score (average=macro):  0.3261597139868786
	f beta score (average=micro):  0.37332000000000004
	f beta score (average=weighted):  0.378759971777586
	f beta score (average=None):  [0.61695858 0.20240014 0.22822086 0.28385938 0.26880523 0.24321185
 0.21273087 0.5530908 ]

compute the average Hamming loss
	hamming loss:  0.62668

jaccard similarity coefficient score
	jaccard score (average=macro):  0.20504246585198954
	jaccard score (average=None):  [0.43611559 0.11116576 0.126703   0.17218266 0.1655614  0.13642843
 0.12073    0.37145289]

confusion matrix:
[[2973  765  526  372   82   81   71  152]
 [ 721  452  461  386  103   52   48   79]
 [ 436  415  558  669  193  111   51  108]
 [ 287  302  448  822  390  206   84   96]
 [  65   75  141  283  724  475  267  277]
 [  73   76  109  183  664  670  497  578]
 [  57   49   57  106  309  491  516  759]
 [ 156   82  121  140  325  645  912 2618]]

================================================================================
Classifier.PASSIVE_AGGRESSIVE_CLASSIFIER
________________________________________________________________________________
Training: 
PassiveAggressiveClassifier(C=0.01, average=False, class_weight=None,
                            early_stopping=True, fit_intercept=True,
                            loss='hinge', max_iter=1000, n_iter_no_change=5,
                            n_jobs=-1, random_state=0, shuffle=True, tol=0.0001,
                            validation_fraction=0.01, verbose=False,
                            warm_start=False)
train time: 0.524s
test time:  0.021s
accuracy:   0.418


cross validation:
	accuracy: 5-fold cross validation: [0.4172 0.4284 0.4096 0.409  0.4164]
	test accuracy: 5-fold cross validation accuracy: 0.42 (+/- 0.01)
dimensionality: 74170
density: 0.749636



===> Classification Report:

              precision    recall  f1-score   support

           1       0.48      0.89      0.62      5022
           2       0.12      0.01      0.02      2302
           3       0.26      0.09      0.13      2541
           4       0.32      0.26      0.29      2635
           7       0.29      0.21      0.24      2307
           8       0.26      0.20      0.23      2850
           9       0.21      0.03      0.05      2344
          10       0.47      0.79      0.59      4999

    accuracy                           0.42     25000
   macro avg       0.30      0.31      0.27     25000
weighted avg       0.34      0.42      0.34     25000



===> Classification Metrics:

accuracy classification score
	accuracy score:  0.41812
	accuracy score (normalize=False):  10453

compute the precision
	precision score (average=macro):  0.3022455043862155
	precision score (average=micro):  0.41812
	precision score (average=weighted):  0.3387808855334155
	precision score (average=None):  [0.47979308 0.12062257 0.26061321 0.32059646 0.29263804 0.26254647
 0.20833333 0.47282088]
	precision score (average=None, zero_division=1):  [0.47979308 0.12062257 0.26061321 0.32059646 0.29263804 0.26254647
 0.20833333 0.47282088]

compute the precision
	recall score (average=macro):  0.3091086588265754
	recall score (average=micro):  0.41812
	recall score (average=weighted):  0.41812
	recall score (average=None):  [0.8864994  0.01346655 0.08697363 0.26110057 0.20676203 0.19824561
 0.02986348 0.78995799]
	recall score (average=None, zero_division=1):  [0.8864994  0.01346655 0.08697363 0.26110057 0.20676203 0.19824561
 0.02986348 0.78995799]

compute the F1 score, also known as balanced F-score or F-measure
	f1 score (average=macro):  0.2721376213337405
	f1 score (average=micro):  0.41812
	f1 score (average=weighted):  0.3421945702657313
	f1 score (average=None):  [0.6226138  0.02422821 0.13042195 0.2878059  0.24231648 0.22590964
 0.05223881 0.59156617]

compute the F-beta score
	f beta score (average=macro):  0.27417845779146544
	f beta score (average=micro):  0.41812
	f beta score (average=weighted):  0.326389609888617
	f beta score (average=None):  [0.52826427 0.04654655 0.18624642 0.30662269 0.27019372 0.24655263
 0.09490239 0.51409899]

compute the average Hamming loss
	hamming loss:  0.58188

jaccard similarity coefficient score
	jaccard score (average=macro):  0.17677208926155916
	jaccard score (average=None):  [0.45202559 0.01226266 0.0697601  0.16809186 0.13786127 0.12733829
 0.02681992 0.42001702]

confusion matrix:
[[4452   46   97  157   43   36    8  183]
 [1617   31  127  248   65   47   10  157]
 [1314   62  221  505  113  100   12  214]
 [ 949   68  239  688  210  210   20  251]
 [ 228   14   67  244  477  505   58  714]
 [ 220   12   45  158  378  565   83 1389]
 [ 161    8   26   69  163  352   70 1495]
 [ 338   16   26   77  181  337   75 3949]]

================================================================================
Classifier.PERCEPTRON
________________________________________________________________________________
Training: 
Perceptron(alpha=0.0001, class_weight=None, early_stopping=True, eta0=1.0,
           fit_intercept=True, max_iter=100, n_iter_no_change=3, n_jobs=-1,
           penalty='l2', random_state=0, shuffle=True, tol=0.0001,
           validation_fraction=0.01, verbose=False, warm_start=False)
train time: 0.415s
test time:  0.019s
accuracy:   0.316


cross validation:
	accuracy: 5-fold cross validation: [0.3364 0.3094 0.3268 0.298  0.2964]
	test accuracy: 5-fold cross validation accuracy: 0.31 (+/- 0.03)
dimensionality: 74170
density: 0.652402



===> Classification Report:

              precision    recall  f1-score   support

           1       0.52      0.49      0.50      5022
           2       0.15      0.07      0.10      2302
           3       0.17      0.21      0.19      2541
           4       0.19      0.32      0.24      2635
           7       0.21      0.10      0.13      2307
           8       0.22      0.17      0.20      2850
           9       0.17      0.10      0.13      2344
          10       0.42      0.59      0.49      4999

    accuracy                           0.32     25000
   macro avg       0.26      0.26      0.25     25000
weighted avg       0.30      0.32      0.30     25000



===> Classification Metrics:

accuracy classification score
	accuracy score:  0.31596
	accuracy score (normalize=False):  7899

compute the precision
	precision score (average=macro):  0.25718855710298494
	precision score (average=micro):  0.31596
	precision score (average=weighted):  0.30052572103534
	precision score (average=None):  [0.52194391 0.15458015 0.1723356  0.19321568 0.20730594 0.22491984
 0.16785714 0.4153502 ]
	precision score (average=None, zero_division=1):  [0.52194391 0.15458015 0.1723356  0.19321568 0.20730594 0.22491984
 0.16785714 0.4153502 ]

compute the precision
	recall score (average=macro):  0.2562974700460544
	recall score (average=micro):  0.31596
	recall score (average=weighted):  0.31596
	recall score (average=None):  [0.48546396 0.07037359 0.20936639 0.3199241  0.09839619 0.1722807
 0.10025597 0.59431886]
	recall score (average=None, zero_division=1):  [0.48546396 0.07037359 0.20936639 0.3199241  0.09839619 0.1722807
 0.10025597 0.59431886]

compute the F1 score, also known as balanced F-score or F-measure
	f1 score (average=macro):  0.24660136543080036
	f1 score (average=micro):  0.31596
	f1 score (average=weighted):  0.29866890910850635
	f1 score (average=None):  [0.50304343 0.09671642 0.18905473 0.24092598 0.13345091 0.19511226
 0.12553419 0.48897301]

compute the F-beta score
	f beta score (average=macro):  0.24987710743576333
	f beta score (average=micro):  0.31596
	f beta score (average=weighted):  0.2971274301337376
	f beta score (average=None):  [0.51421581 0.12473052 0.17865538 0.20983721 0.16973232 0.21196685
 0.14791037 0.4419684 ]

compute the average Hamming loss
	hamming loss:  0.68404

jaccard similarity coefficient score
	jaccard score (average=macro):  0.14979863134938787
	jaccard score (average=None):  [0.33604411 0.05081556 0.1043956  0.13696182 0.07149606 0.10810216
 0.06697065 0.32360309]

confusion matrix:
[[2438  353  664  710   70  122  109  556]
 [ 700  162  434  547   57   77   56  269]
 [ 530  158  532  680   91  121   81  348]
 [ 391  137  498  843  133  186   99  348]
 [ 120   51  273  500  227  345  182  609]
 [ 145   55  252  416  214  491  275 1002]
 [ 121   42  181  259  133  323  235 1050]
 [ 226   90  253  408  170  518  363 2971]]

================================================================================
Classifier.RANDOM_FOREST_CLASSIFIER
________________________________________________________________________________
Training: 
RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=2, min_samples_split=5,
                       min_weight_fraction_leaf=0.0, n_estimators=200,
                       n_jobs=-1, oob_score=False, random_state=0,
                       verbose=False, warm_start=False)
train time: 9.596s
test time:  0.708s
accuracy:   0.377


cross validation:
	accuracy: 5-fold cross validation: [0.369  0.3796 0.3768 0.3728 0.3718]
	test accuracy: 5-fold cross validation accuracy: 0.37 (+/- 0.01)


===> Classification Report:

              precision    recall  f1-score   support

           1       0.38      0.92      0.54      5022
           2       1.00      0.01      0.01      2302
           3       0.66      0.01      0.02      2541
           4       0.37      0.06      0.10      2635
           7       0.33      0.04      0.07      2307
           8       0.24      0.09      0.13      2850
           9       0.50      0.00      0.00      2344
          10       0.38      0.86      0.53      4999

    accuracy                           0.38     25000
   macro avg       0.48      0.25      0.18     25000
weighted avg       0.46      0.38      0.25     25000



===> Classification Metrics:

accuracy classification score
	accuracy score:  0.37716
	accuracy score (normalize=False):  9429

compute the precision
	precision score (average=macro):  0.4837319845864658
	precision score (average=micro):  0.37716
	precision score (average=weighted):  0.4568253194712929
	precision score (average=None):  [0.38154737 1.         0.65957447 0.37349398 0.32818533 0.24262607
 0.5        0.38442866]
	precision score (average=None, zero_division=1):  [0.38154737 1.         0.65957447 0.37349398 0.32818533 0.24262607
 0.5        0.38442866]

compute the precision
	recall score (average=macro):  0.2472240042977994
	recall score (average=micro):  0.37716
	recall score (average=weighted):  0.37716
	recall score (average=None):  [0.9181601  0.00521286 0.01219992 0.05882353 0.03684439 0.08947368
 0.00170648 0.85537107]
	recall score (average=None, zero_division=1):  [0.9181601  0.00521286 0.01219992 0.05882353 0.03684439 0.08947368
 0.00170648 0.85537107]

compute the F1 score, also known as balanced F-score or F-measure
	f1 score (average=macro):  0.17573607639053812
	f1 score (average=micro):  0.37716
	f1 score (average=weighted):  0.24979893199471714
	f1 score (average=None):  [0.53907757 0.01037165 0.02395672 0.10163934 0.06625097 0.13073571
 0.00340136 0.53045528]

compute the F-beta score
	f beta score (average=macro):  0.18038946888823176
	f beta score (average=micro):  0.37716
	f beta score (average=weighted):  0.23344099825873055
	f beta score (average=None):  [0.43204902 0.02553191 0.05679736 0.18044237 0.12713132 0.18074851
 0.00841751 0.43199774]

compute the average Hamming loss
	hamming loss:  0.62284

jaccard similarity coefficient score
	jaccard score (average=macro):  0.11334305630411744
	jaccard score (average=None):  [0.36899808 0.00521286 0.01212358 0.05354059 0.03426038 0.06993966
 0.00170358 0.36096573]

confusion matrix:
[[4611    0    1   20    2   27    0  361]
 [1917   12    4   38    9   22    0  300]
 [1910    0   31   93   14   63    0  430]
 [1692    0    8  155   48  125    0  607]
 [ 587    0    3   56   85  265    1 1310]
 [ 515    0    0   28   63  255    2 1987]
 [ 314    0    0   15   17  142    4 1852]
 [ 539    0    0   10   21  152    1 4276]]

================================================================================
Classifier.RIDGE_CLASSIFIER
________________________________________________________________________________
Training: 
RidgeClassifier(alpha=1.0, class_weight=None, copy_X=True, fit_intercept=True,
                max_iter=None, normalize=False, random_state=0, solver='auto',
                tol=0.001)
train time: 2.934s
test time:  0.041s
accuracy:   0.386


cross validation:
	accuracy: 5-fold cross validation: [0.4036 0.4074 0.402  0.3954 0.4   ]
	test accuracy: 5-fold cross validation accuracy: 0.40 (+/- 0.01)
dimensionality: 74170
density: 1.000000



===> Classification Report:

              precision    recall  f1-score   support

           1       0.50      0.77      0.61      5022
           2       0.17      0.08      0.11      2302
           3       0.23      0.13      0.17      2541
           4       0.27      0.24      0.25      2635
           7       0.24      0.17      0.20      2307
           8       0.24      0.20      0.22      2850
           9       0.19      0.09      0.12      2344
          10       0.47      0.69      0.56      4999

    accuracy                           0.39     25000
   macro avg       0.29      0.30      0.28     25000
weighted avg       0.33      0.39      0.34     25000



===> Classification Metrics:

accuracy classification score
	accuracy score:  0.38552
	accuracy score (normalize=False):  9638

compute the precision
	precision score (average=macro):  0.2896418933588193
	precision score (average=micro):  0.38552
	precision score (average=weighted):  0.3296106857136312
	precision score (average=None):  [0.50142229 0.17075472 0.23024055 0.27080638 0.24321062 0.24203273
 0.19326923 0.46539862]
	precision score (average=None, zero_division=1):  [0.50142229 0.17075472 0.23024055 0.27080638 0.24321062 0.24203273
 0.19326923 0.46539862]

compute the precision
	recall score (average=macro):  0.2960956528387336
	recall score (average=micro):  0.38552
	recall score (average=weighted):  0.38552
	recall score (average=None):  [0.77220231 0.07862728 0.13183786 0.23833017 0.17468574 0.19719298
 0.08575085 0.69013803]
	recall score (average=None, zero_division=1):  [0.77220231 0.07862728 0.13183786 0.23833017 0.17468574 0.19719298
 0.08575085 0.69013803]

compute the F1 score, also known as balanced F-score or F-measure
	f1 score (average=macro):  0.2790329682017034
	f1 score (average=micro):  0.38552000000000003
	f1 score (average=weighted):  0.34165615011768363
	f1 score (average=None):  [0.60802759 0.107674   0.16766767 0.2535325  0.20332997 0.21732405
 0.11879433 0.55591363]

compute the F-beta score
	f beta score (average=macro):  0.2813621618399667
	f beta score (average=micro):  0.38552
	f beta score (average=weighted):  0.3304428209948625
	f beta score (average=None):  [0.53924022 0.1383369  0.20033489 0.26362186 0.22551763 0.23150437
 0.1545203  0.49782113]

compute the average Hamming loss
	hamming loss:  0.61448

jaccard similarity coefficient score
	jaccard score (average=macro):  0.17669628477116775
	jaccard score (average=None):  [0.43681009 0.05690035 0.09150505 0.14516875 0.11317046 0.12190889
 0.06314797 0.38495871]

confusion matrix:
[[3878  240  216  256   57   66   26  283]
 [1278  181  214  279   92   66   15  177]
 [1008  226  335  486  121  101   38  226]
 [ 742  182  329  628  227  200   69  258]
 [ 184   71  146  264  403  458  165  616]
 [ 194   55  100  198  358  562  231 1152]
 [ 137   48   53   93  188  373  201 1251]
 [ 313   57   62  115  211  496  295 3450]]

FINAL CLASSIFICATION TABLE: IMDB_REVIEWS dataset (Multi-class classification)
| ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
| --- | ------------- | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
|  1  |  ADA_BOOST_CLASSIFIER  |  38.02%  |  [0.3792 0.379  0.374  0.3704 0.3746]  |  0.38 (+/- 0.01)  |  125.8  |  7.788  |
|  2  |  BERNOULLI_NB  |  37.03%  |  [0.377  0.389  0.3782 0.38   0.373 ]  |  0.38 (+/- 0.01)  |  0.04582  |  0.04338  |
|  3  |  COMPLEMENT_NB  |  37.34%  |  [0.3878 0.3942 0.3976 0.3938 0.3832]  |  0.39 (+/- 0.01)  |  0.03707  |  0.01888  |
|  4  |  DECISION_TREE_CLASSIFIER  |  30.82%  |  [0.3072 0.3132 0.2996 0.3066 0.3064]  |  0.31 (+/- 0.01)  |  6.834  |  0.01294  |
|  5  |  GRADIENT_BOOSTING_CLASSIFIER  |  37.88%  |  [0.3848 0.379  0.3696 0.3692 0.3728]  |  0.38 (+/- 0.01)  |  872.7  |  0.5021  |
|  6  |  K_NEIGHBORS_CLASSIFIER  |  37.26%  |  [0.3822 0.3916 0.3842 0.386  0.388 ]  |  0.39 (+/- 0.01)  |  0.006473  |  14.71  |
|  7  |  LINEAR_SVC  |  40.80%  |  [0.41   0.4206 0.4064 0.3992 0.4088]  |  0.41 (+/- 0.01)  |  0.5438  |  0.01863  |
|  8  |  LOGISTIC_REGRESSION  |  42.04%  |  [0.4282 0.4334 0.4152 0.4194 0.4218]  |  0.42 (+/- 0.01)  |  9.81  |  0.01971  |
|  9  |  MULTINOMIAL_NB  |  37.82%  |  [0.389  0.3928 0.3918 0.3942 0.386 ]  |  0.39 (+/- 0.01)  |  0.03475  |  0.01914  |
|  10  |  NEAREST_CENTROID  |  37.33%  |  [0.3872 0.3786 0.3894 0.3672 0.3782]  |  0.38 (+/- 0.02)  |  0.02605  |  0.03179  |
|  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  41.81%  |  [0.4172 0.4284 0.4096 0.409  0.4164]  |  0.42 (+/- 0.01)  |  0.5241  |  0.02055  |
|  12  |  PERCEPTRON  |  31.60%  |  [0.3364 0.3094 0.3268 0.298  0.2964]  |  0.31 (+/- 0.03)  |  0.4149  |  0.0195  |
|  13  |  RANDOM_FOREST_CLASSIFIER  |  37.72%  |  [0.369  0.3796 0.3768 0.3728 0.3718]  |  0.37 (+/- 0.01)  |  9.596  |  0.708  |
|  14  |  RIDGE_CLASSIFIER  |  38.55%  |  [0.4036 0.4074 0.402  0.3954 0.4   ]  |  0.40 (+/- 0.01)  |  2.934  |  0.04121  |


Best algorithm:
===> 8) LOGISTIC_REGRESSION
		Accuracy score = 42.04%		Training time = 9.81		Test time = 0.01971



DONE!
Program finished. It took 3434.2735683918 seconds

Process finished with exit code 0
```