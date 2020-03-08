# Running just miniproject classifiers (TWENTY_NEWS_GROUPS dataset and IMDB_REVIEWS dataset using binary classification)

### IMDB using Binary Classification

#### FINAL CLASSIFICATION TABLE: IMDB_REVIEWS dataset (Multi-class classification)

| ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
| --- | ------------- | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
|  1  |  ADA_BOOST_CLASSIFIER  |  38.02%  |  [0.371  0.3702 0.382  0.38   0.3756]  |  0.38 (+/- 0.01)  |  197.4  |  12.03  |
|  2  |  DECISION_TREE_CLASSIFIER  |  30.82%  |  [0.3028 0.303  0.3094 0.3028 0.2986]  |  0.30 (+/- 0.01)  |  13.02  |  0.01931  |
|  3  |  LINEAR_SVC  |  40.76%  |  [0.4064 0.4048 0.4118 0.418  0.409 ]  |  0.41 (+/- 0.01)  |  1.445  |  0.04247  |
|  4  |  LOGISTIC_REGRESSION  |  42.04%  |  [0.422  0.4256 0.4264 0.424  0.4288]  |  0.43 (+/- 0.00)  |  20.88  |  0.04449  |
|  5  |  RANDOM_FOREST_CLASSIFIER  |  37.84%  |  [0.3736 0.3752 0.3736 0.3766 0.3738]  |  0.37 (+/- 0.00)  |  31.31  |  2.228  |

![FINAL CLASSIFICATION TABLE: IMDB_REVIEWS dataset (Multi-class classification)](https://github.com/ramonfigueiredopessoa/comp551-2020-p2_classification_of_textual_data/blob/master/code/results/just_miniproject_classifiers_imdb_using_multi_class_classification/IMDB_REVIEWS_multi_class_classification.png)

#### Computer settings used to run

* Operating system: Ubuntu 18.04.3 LTS (64-bit)
* Processor: Intel® Core™ i7-2620M CPU @ 2.70GHz × 4
* Memory: 16 GB

#### All logs

```
python /comp551-2020-p2_classification_of_textual_data/code/main.py --dataset IMDB_REVIEWS --use_imdb_multi_class_labels --run_cross_validation --report --all_metrics --confusion_matrix --plot_accurary_and_time_together --use_just_miniproject_classifiers
03/08/2020 11:54:33 AM - INFO - Program started...
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
	Use just the miniproject classifiers (1. LogisticRegression, 2. DecisionTreeClassifier, 3. LinearSVC, 4. AdaBoostClassifier, 5. RandomForestClassifier) =  True
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

===> Reading files from /home/ramon/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/train/neg

===> Reading files from /home/ramon/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/train/pos

===> Reading files from /home/ramon/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/test/neg

===> Reading files from /home/ramon/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/test/pos
data loaded
25000 documents - 33.133MB (training set)
25000 documents - 32.351MB (test set)

Extracting features from the training data using a vectorizer
done in 4.864082s at 6.812MB/s
n_samples: 25000, n_features: 74170

Extracting features from the test data using the same vectorizer
done in 4.860752s at 6.655MB/s
n_samples: 25000, n_features: 74170

	==> Using JSON with best parameters (selected using grid search) to the ADA_BOOST_CLASSIFIER classifier (multi-class classification) and IMDB_REVIEWS dataset ===> JSON in dictionary format: {'learning_rate': 0.1, 'n_estimators': 500}
	 AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=0.1,
                   n_estimators=500, random_state=0)
	==> Using JSON with best parameters (selected using grid search) to the DECISION_TREE_CLASSIFIER classifier (multi-class classification) and IMDB_REVIEWS dataset ===> JSON in dictionary format: {'criterion': 'entropy', 'min_samples_split': 250, 'splitter': 'random'}
	 DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='entropy',
                       max_depth=None, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=250,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=0, splitter='random')
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
	==> Using JSON with best parameters (selected using grid search) to the RANDOM_FOREST_CLASSIFIER classifier (multi-class classification) and IMDB_REVIEWS dataset ===> JSON in dictionary format: {'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 200}
	 RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=2, min_samples_split=5,
                       min_weight_fraction_leaf=0.0, n_estimators=200,
                       n_jobs=-1, oob_score=False, random_state=0,
                       verbose=False, warm_start=False)
================================================================================
Classifier.ADA_BOOST_CLASSIFIER
________________________________________________________________________________
Training: 
AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=0.1,
                   n_estimators=500, random_state=0)
train time: 197.413s
test time:  12.031s
accuracy:   0.380


cross validation:
/home/ramon/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/metrics/_classification.py:1272: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/home/ramon/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/metrics/_classification.py:1272: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/home/ramon/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/metrics/_classification.py:1272: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
	accuracy: 5-fold cross validation: [0.371  0.3702 0.382  0.38   0.3756]
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
Classifier.DECISION_TREE_CLASSIFIER
________________________________________________________________________________
Training: 
DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='entropy',
                       max_depth=None, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=250,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=0, splitter='random')
train time: 13.016s
test time:  0.019s
accuracy:   0.308


cross validation:
	accuracy: 5-fold cross validation: [0.3028 0.303  0.3094 0.3028 0.2986]
	test accuracy: 5-fold cross validation accuracy: 0.30 (+/- 0.01)


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
Classifier.LINEAR_SVC
________________________________________________________________________________
Training: 
LinearSVC(C=0.01, class_weight=None, dual=True, fit_intercept=True,
          intercept_scaling=1, loss='squared_hinge', max_iter=1000,
          multi_class='crammer_singer', penalty='l2', random_state=0, tol=0.001,
          verbose=False)
train time: 1.445s
/home/ramon/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/svm/_base.py:947: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  "the number of iterations.", ConvergenceWarning)
test time:  0.042s
accuracy:   0.408


cross validation:
/home/ramon/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/svm/_base.py:947: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  "the number of iterations.", ConvergenceWarning)
/home/ramon/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/svm/_base.py:947: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  "the number of iterations.", ConvergenceWarning)
/home/ramon/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/svm/_base.py:947: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  "the number of iterations.", ConvergenceWarning)
/home/ramon/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/svm/_base.py:947: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  "the number of iterations.", ConvergenceWarning)
	accuracy: 5-fold cross validation: [0.4064 0.4048 0.4118 0.418  0.409 ]
	test accuracy: 5-fold cross validation accuracy: 0.41 (+/- 0.01)
dimensionality: 74170
density: 0.761876



===> Classification Report:

              precision    recall  f1-score   support

           1       0.47      0.88      0.62      5022
           2       0.14      0.04      0.06      2302
           3       0.22      0.09      0.13      2541
           4       0.31      0.23      0.27      2635
           7       0.27      0.20      0.23      2307
           8       0.27      0.15      0.19      2850
           9       0.19      0.06      0.09      2344
          10       0.48      0.76      0.59      4999

    accuracy                           0.41     25000
   macro avg       0.29      0.30      0.27     25000
weighted avg       0.33      0.41      0.34     25000



===> Classification Metrics:

accuracy classification score
	accuracy score:  0.40764
	accuracy score (normalize=False):  10191

compute the precision
	precision score (average=macro):  0.29375667373209935
	precision score (average=micro):  0.40764
	precision score (average=weighted):  0.33184032319683393
	precision score (average=None):  [0.47258737 0.13671275 0.21857411 0.31135903 0.27354529 0.26700572
 0.19230769 0.47796143]
	precision score (average=None, zero_division=1):  [0.47258737 0.13671275 0.21857411 0.31135903 0.27354529 0.26700572
 0.19230769 0.47796143]

compute the precision
	recall score (average=macro):  0.3015260465297024
	recall score (average=micro):  0.40764
	recall score (average=weighted):  0.40764
	recall score (average=None):  [0.88052569 0.03866203 0.09169618 0.23301708 0.1976593  0.14736842
 0.05972696 0.76355271]
	recall score (average=None, zero_division=1):  [0.88052569 0.03866203 0.09169618 0.23301708 0.1976593  0.14736842
 0.05972696 0.76355271]

compute the F1 score, also known as balanced F-score or F-measure
	f1 score (average=macro):  0.27119355700005543
	f1 score (average=micro):  0.40764
	f1 score (average=weighted):  0.3392621167427323
	f1 score (average=None):  [0.61506363 0.06027768 0.12919324 0.2665509  0.2294917  0.18991635
 0.09114583 0.58790913]

compute the F-beta score
	f beta score (average=macro):  0.2760035857473251
	f beta score (average=micro):  0.40764
	f beta score (average=weighted):  0.32654750265988075
	f beta score (average=None):  [0.52084806 0.09070526 0.17119765 0.2917419  0.254039   0.22970904
 0.13318113 0.51660666]

compute the average Hamming loss
	hamming loss:  0.59236

jaccard similarity coefficient score
	jaccard score (average=macro):  0.17458006426777434
	jaccard score (average=None):  [0.44410967 0.03107542 0.0690575  0.1537691  0.1296191  0.10492131
 0.04774898 0.41633944]

confusion matrix:
[[4422   90  117  144   39   31   17  162]
 [1611   89  137  192   76   35   18  144]
 [1310  149  233  412  129   77   35  196]
 [ 936  140  290  614  237  145   45  228]
 [ 241   56  119  258  456  348  153  676]
 [ 247   40   80  186  369  420  175 1333]
 [ 187   36   40   72  181  258  140 1430]
 [ 403   51   50   94  180  259  145 3817]]

================================================================================
Classifier.LOGISTIC_REGRESSION
________________________________________________________________________________
Training: 
LogisticRegression(C=1, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=-1, penalty='l2', random_state=0,
                   solver='lbfgs', tol=0.001, verbose=False, warm_start=False)
/home/ramon/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
train time: 20.877s
test time:  0.044s
accuracy:   0.420


cross validation:
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
	accuracy: 5-fold cross validation: [0.422  0.4256 0.4264 0.424  0.4288]
	test accuracy: 5-fold cross validation accuracy: 0.43 (+/- 0.00)
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
train time: 31.312s
test time:  2.228s
accuracy:   0.378


cross validation:
	accuracy: 5-fold cross validation: [0.3736 0.3752 0.3736 0.3766 0.3738]
	test accuracy: 5-fold cross validation accuracy: 0.37 (+/- 0.00)


===> Classification Report:

              precision    recall  f1-score   support

           1       0.38      0.92      0.54      5022
           2       1.00      0.00      0.01      2302
           3       0.56      0.01      0.02      2541
           4       0.35      0.05      0.09      2635
           7       0.32      0.04      0.07      2307
           8       0.25      0.09      0.14      2850
           9       0.67      0.00      0.00      2344
          10       0.39      0.86      0.53      4999

    accuracy                           0.38     25000
   macro avg       0.49      0.25      0.18     25000
weighted avg       0.46      0.38      0.25     25000



===> Classification Metrics:

accuracy classification score
	accuracy score:  0.3784
	accuracy score (normalize=False):  9460

compute the precision
	precision score (average=macro):  0.4897372222017187
	precision score (average=micro):  0.3784
	precision score (average=weighted):  0.46082871276587206
	precision score (average=None):  [0.38205787 1.         0.55555556 0.35384615 0.3202847  0.25239006
 0.66666667 0.38709677]
	precision score (average=None, zero_division=1):  [0.38205787 1.         0.55555556 0.35384615 0.3202847  0.25239006
 0.66666667 0.38709677]

compute the precision
	recall score (average=macro):  0.2477592927388514
	recall score (average=micro):  0.3784
	recall score (average=weighted):  0.3784
	recall score (average=None):  [0.91756272 0.00477845 0.00983865 0.05237192 0.0390117  0.09263158
 0.00170648 0.86417283]
	recall score (average=None, zero_division=1):  [0.91756272 0.00477845 0.00983865 0.05237192 0.0390117  0.09263158
 0.00170648 0.86417283]

compute the F1 score, also known as balanced F-score or F-measure
	f1 score (average=macro):  0.17534198785906818
	f1 score (average=micro):  0.3784
	f1 score (average=weighted):  0.24993219444106443
	f1 score (average=None):  [0.5394837  0.00951146 0.01933488 0.09123967 0.06955178 0.13552361
 0.00340426 0.53468655]

compute the F-beta score
	f beta score (average=macro):  0.17860193400555363
	f beta score (average=micro):  0.3784
	f beta score (average=weighted):  0.2323534294320341
	f beta score (average=None):  [0.43254609 0.02344416 0.04593899 0.16448153 0.1311571  0.18765994
 0.00844595 0.43514172]

compute the average Hamming loss
	hamming loss:  0.6216

jaccard similarity coefficient score
	jaccard score (average=macro):  0.11337953363519757
	jaccard score (average=None):  [0.36937876 0.00477845 0.00976181 0.04780048 0.03602882 0.07268722
 0.00170503 0.36489568]

confusion matrix:
[[4608    0    4   22    7   20    0  361]
 [1931   11    0   35    8   15    0  302]
 [1902    0   25   95   14   74    0  431]
 [1687    0    8  138   52  130    1  619]
 [ 611    0    6   49   90  265    0 1286]
 [ 495    0    1   28   69  264    1 1992]
 [ 307    0    0   13   22  149    4 1849]
 [ 520    0    1   10   19  129    0 4320]]

FINAL CLASSIFICATION TABLE: IMDB_REVIEWS dataset (Multi-class classification)
| ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
| --- | ------------- | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
|  1  |  ADA_BOOST_CLASSIFIER  |  38.02%  |  [0.371  0.3702 0.382  0.38   0.3756]  |  0.38 (+/- 0.01)  |  197.4  |  12.03  |
|  2  |  DECISION_TREE_CLASSIFIER  |  30.82%  |  [0.3028 0.303  0.3094 0.3028 0.2986]  |  0.30 (+/- 0.01)  |  13.02  |  0.01931  |
|  3  |  LINEAR_SVC  |  40.76%  |  [0.4064 0.4048 0.4118 0.418  0.409 ]  |  0.41 (+/- 0.01)  |  1.445  |  0.04247  |
|  4  |  LOGISTIC_REGRESSION  |  42.04%  |  [0.422  0.4256 0.4264 0.424  0.4288]  |  0.43 (+/- 0.00)  |  20.88  |  0.04449  |
|  5  |  RANDOM_FOREST_CLASSIFIER  |  37.84%  |  [0.3736 0.3752 0.3736 0.3766 0.3738]  |  0.37 (+/- 0.00)  |  31.31  |  2.228  |


Best algorithm:
===> 4) LOGISTIC_REGRESSION
		Accuracy score = 42.04%		Training time = 20.88		Test time = 0.04449



DONE!
Program finished. It took 1917.1267108917236 seconds

Process finished with exit code 0
```