# Running just miniproject classifiers (TWENTY_NEWS_GROUPS dataset and IMDB_REVIEWS dataset using binary classification)


### 20 News Groups dataset (removing headers signatures and quoting)

#### FINAL CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS dataset (removing headers signatures and quoting)

| ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
| --- | ------------- | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
|  1  |  ADA_BOOST_CLASSIFIER  |  44.04%  |  [0.46973045 0.45779938 0.46575342 0.47017234 0.46993811]  |  0.47 (+/- 0.01)  |  37.35  |  2.693  |
|  2  |  DECISION_TREE_CLASSIFIER  |  44.72%  |  [0.49094123 0.48696421 0.47547503 0.49270879 0.49646331]  |  0.49 (+/- 0.01)  |  17.83  |  0.01238  |
|  3  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  0.76 (+/- 0.02)  |  2.208  |  0.01807  |
|  4  |  LOGISTIC_REGRESSION  |  69.28%  |  [0.75961114 0.74016792 0.75519222 0.7565179  0.73607427]  |  0.75 (+/- 0.02)  |  50.95  |  0.03105  |
|  5  |  RANDOM_FOREST_CLASSIFIER  |  63.71%  |  [0.69465312 0.66902342 0.68272205 0.69730446 0.67462423]  |  0.68 (+/- 0.02)  |  42.94  |  1.114  |



### IMDB using Binary Classification

#### FINAL CLASSIFICATION TABLE: IMDB_REVIEWS dataset (Binary classification)

| ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
| --- | ------------- | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
|  1  |  ADA_BOOST_CLASSIFIER  |  84.60%  |  [0.8426 0.8384 0.8454 0.8406 0.8374]  |  0.84 (+/- 0.01)  |  174.1  |  9.467  |
|  2  |  DECISION_TREE_CLASSIFIER  |  74.14%  |  [0.744  0.7276 0.7354 0.7408 0.7332]  |  0.74 (+/- 0.01)  |  12.23  |  0.01961  |
|  3  |  LINEAR_SVC  |  87.13%  |  [0.8862 0.8798 0.8838 0.8856 0.8876]  |  0.88 (+/- 0.01)  |  0.5033  |  0.006712  |
|  4  |  LOGISTIC_REGRESSION  |  87.75%  |  [0.8892 0.8856 0.889  0.8876 0.891 ]  |  0.89 (+/- 0.00)  |  1.837  |  0.01146  |
|  5  |  RANDOM_FOREST_CLASSIFIER  |  85.17%  |  [0.8576 0.8544 0.8528 0.8522 0.8546]  |  0.85 (+/- 0.00)  |  43.13  |  2.226  |

#### Computer settings used to run

* Operating system: Ubuntu 18.04.3 LTS (64-bit)
* Processor: Intel® Core™ i7-2620M CPU @ 2.70GHz × 4
* Memory: 16 GB

#### All logs

```
python /comp551-2020-p2_classification_of_textual_data/code/main.py --run_cross_validation --report --all_metrics --confusion_matrix --plot_accurary_and_time_together --use_just_miniproject_classifiers
03/08/2020 10:37:16 AM - INFO - Program started...
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
  Dataset = ALL
  ML algorithm list (If ml_algorithm_list is not provided, all ML algorithms will be executed) = None
  Use classifiers with default parameters. Default: False = Use classifiers with best parameters found using grid search. False
  Read dataset without shuffle data = False
  The number of CPUs to use to do the computation. If the provided number is negative or greater than the number of available CPUs, the system will use all the available CPUs. Default: -1 (-1 == all CPUs) = -1
  Run cross validation. Default: False = True
  Number of cross validation folds. Default: 5 = 5
  Use just the miniproject classifiers (1. LogisticRegression, 2. DecisionTreeClassifier, 3. LinearSVC, 4. AdaBoostClassifier, 5. RandomForestClassifier) =  True
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
==================================================================================================================================

Loading TWENTY_NEWS_GROUPS dataset for categories:
data loaded
11314 documents - 13.782MB (training set)
7532 documents - 8.262MB (test set)
20 categories

Extracting features from the training data using a vectorizer
done in 2.127138s at 6.479MB/s
n_samples: 11314, n_features: 101321

Extracting features from the test data using the same vectorizer
done in 1.165567s at 7.088MB/s
n_samples: 7532, n_features: 101321

  ==> Using JSON with best parameters (selected using grid search) to the ADA_BOOST_CLASSIFIER classifier (multi-class classification) and TWENTY_NEWS_GROUPS dataset ===> JSON in dictionary format: {'learning_rate': 1, 'n_estimators': 200}
   AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1,
                   n_estimators=200, random_state=0)
  ==> Using JSON with best parameters (selected using grid search) to the DECISION_TREE_CLASSIFIER classifier (multi-class classification) and TWENTY_NEWS_GROUPS dataset ===> JSON in dictionary format: {'criterion': 'gini', 'min_samples_split': 2, 'splitter': 'random'}
   DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=None, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=0, splitter='random')
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
  ==> Using JSON with best parameters (selected using grid search) to the RANDOM_FOREST_CLASSIFIER classifier (multi-class classification) and TWENTY_NEWS_GROUPS dataset ===> JSON in dictionary format: {'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 200}
   RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=5,
                       min_weight_fraction_leaf=0.0, n_estimators=200,
                       n_jobs=-1, oob_score=False, random_state=0,
                       verbose=False, warm_start=False)
================================================================================
Classifier.ADA_BOOST_CLASSIFIER
________________________________________________________________________________
Training: 
AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1,
                   n_estimators=200, random_state=0)
train time: 37.351s
test time:  2.693s
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
Classifier.DECISION_TREE_CLASSIFIER
________________________________________________________________________________
Training: 
DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=None, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=0, splitter='random')
train time: 17.829s
test time:  0.012s
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
Classifier.LINEAR_SVC
________________________________________________________________________________
Training: 
LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
          intercept_scaling=1, loss='squared_hinge', max_iter=1000,
          multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
          verbose=False)
train time: 2.208s
test time:  0.018s
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
/home/ramon/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
train time: 50.946s
test time:  0.031s
accuracy:   0.693


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
train time: 42.938s
test time:  1.114s
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

FINAL CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS dataset (removing headers signatures and quoting)
| ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
| --- | ------------- | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
|  1  |  ADA_BOOST_CLASSIFIER  |  44.04%  |  [0.46973045 0.45779938 0.46575342 0.47017234 0.46993811]  |  0.47 (+/- 0.01)  |  37.35  |  2.693  |
|  2  |  DECISION_TREE_CLASSIFIER  |  44.72%  |  [0.49094123 0.48696421 0.47547503 0.49270879 0.49646331]  |  0.49 (+/- 0.01)  |  17.83  |  0.01238  |
|  3  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  0.76 (+/- 0.02)  |  2.208  |  0.01807  |
|  4  |  LOGISTIC_REGRESSION  |  69.28%  |  [0.75961114 0.74016792 0.75519222 0.7565179  0.73607427]  |  0.75 (+/- 0.02)  |  50.95  |  0.03105  |
|  5  |  RANDOM_FOREST_CLASSIFIER  |  63.71%  |  [0.69465312 0.66902342 0.68272205 0.69730446 0.67462423]  |  0.68 (+/- 0.02)  |  42.94  |  1.114  |


Best algorithm:
===> 3) LINEAR_SVC
    Accuracy score = 69.82%   Training time = 2.208   Test time = 0.01807

Loading IMDB_REVIEWS dataset:

===> Reading files from /home/ramon/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/train/neg

===> Reading files from /home/ramon/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/train/pos

===> Reading files from /home/ramon/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/test/neg

===> Reading files from /home/ramon/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/test/pos
data loaded
25000 documents - 33.133MB (training set)
25000 documents - 32.351MB (test set)

Extracting features from the training data using a vectorizer
done in 4.980431s at 6.653MB/s
n_samples: 25000, n_features: 74170

Extracting features from the test data using the same vectorizer
done in 4.739115s at 6.826MB/s
n_samples: 25000, n_features: 74170

  ==> Using JSON with best parameters (selected using grid search) to the ADA_BOOST_CLASSIFIER classifier (binary classification) and IMDB_REVIEWS dataset ===> JSON in dictionary format: {'learning_rate': 1, 'n_estimators': 500}
   AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1,
                   n_estimators=500, random_state=0)
  ==> Using JSON with best parameters (selected using grid search) to the DECISION_TREE_CLASSIFIER classifier (binary classification) and IMDB_REVIEWS dataset ===> JSON in dictionary format: {'criterion': 'entropy', 'min_samples_split': 250, 'splitter': 'random'}
   DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='entropy',
                       max_depth=None, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=250,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=0, splitter='random')
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
  ==> Using JSON with best parameters (selected using grid search) to the RANDOM_FOREST_CLASSIFIER classifier (binary classification) and IMDB_REVIEWS dataset ===> JSON in dictionary format: {'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 200}
   RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=5,
                       min_weight_fraction_leaf=0.0, n_estimators=200,
                       n_jobs=-1, oob_score=False, random_state=0,
                       verbose=False, warm_start=False)
================================================================================
Classifier.ADA_BOOST_CLASSIFIER
________________________________________________________________________________
Training: 
AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1,
                   n_estimators=500, random_state=0)
train time: 174.115s
test time:  9.467s
accuracy:   0.846


cross validation:
  accuracy: 5-fold cross validation: [0.8426 0.8384 0.8454 0.8406 0.8374]
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
Classifier.DECISION_TREE_CLASSIFIER
________________________________________________________________________________
Training: 
DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='entropy',
                       max_depth=None, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=250,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=0, splitter='random')
train time: 12.232s
test time:  0.020s
accuracy:   0.741


cross validation:
  accuracy: 5-fold cross validation: [0.744  0.7276 0.7354 0.7408 0.7332]
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
Classifier.LINEAR_SVC
________________________________________________________________________________
Training: 
LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
          intercept_scaling=1, loss='squared_hinge', max_iter=1000,
          multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
          verbose=False)
train time: 0.503s
test time:  0.007s
accuracy:   0.871


cross validation:
  accuracy: 5-fold cross validation: [0.8862 0.8798 0.8838 0.8856 0.8876]
  test accuracy: 5-fold cross validation accuracy: 0.88 (+/- 0.01)
dimensionality: 74170
density: 0.872388



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
train time: 1.837s
test time:  0.011s
accuracy:   0.877


cross validation:
  accuracy: 5-fold cross validation: [0.8892 0.8856 0.889  0.8876 0.891 ]
  test accuracy: 5-fold cross validation accuracy: 0.89 (+/- 0.00)
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
train time: 43.126s
test time:  2.226s
accuracy:   0.852


cross validation:
  accuracy: 5-fold cross validation: [0.8576 0.8544 0.8528 0.8522 0.8546]
  test accuracy: 5-fold cross validation accuracy: 0.85 (+/- 0.00)


===> Classification Report:

              precision    recall  f1-score   support

           0       0.85      0.86      0.85     12500
           1       0.86      0.84      0.85     12500

    accuracy                           0.85     25000
   macro avg       0.85      0.85      0.85     25000
weighted avg       0.85      0.85      0.85     25000



===> Classification Metrics:

accuracy classification score
  accuracy score:  0.85172
  accuracy score (normalize=False):  21293

compute the precision
  precision score (average=macro):  0.8518183537760722
  precision score (average=micro):  0.85172
  precision score (average=weighted):  0.8518183537760723
  precision score (average=None):  [0.84593595 0.85770076]
  precision score (average=None, zero_division=1):  [0.84593595 0.85770076]

compute the precision
  recall score (average=macro):  0.85172
  recall score (average=micro):  0.85172
  recall score (average=weighted):  0.85172
  recall score (average=None):  [0.86008 0.84336]
  recall score (average=None, zero_division=1):  [0.86008 0.84336]

compute the F1 score, also known as balanced F-score or F-measure
  f1 score (average=macro):  0.8517096360457794
  f1 score (average=micro):  0.85172
  f1 score (average=weighted):  0.8517096360457793
  f1 score (average=None):  [0.85294934 0.85046993]

compute the F-beta score
  f beta score (average=macro):  0.8517605714064163
  f beta score (average=micro):  0.85172
  f beta score (average=weighted):  0.8517605714064164
  f beta score (average=None):  [0.84872742 0.85479372]

compute the average Hamming loss
  hamming loss:  0.14828

jaccard similarity coefficient score
  jaccard score (average=macro):  0.7417217751766181
  jaccard score (average=None):  [0.74360216 0.73984139]

confusion matrix:
[[10751  1749]
 [ 1958 10542]]

FINAL CLASSIFICATION TABLE: IMDB_REVIEWS dataset (Binary classification)
| ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
| --- | ------------- | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
|  1  |  ADA_BOOST_CLASSIFIER  |  84.60%  |  [0.8426 0.8384 0.8454 0.8406 0.8374]  |  0.84 (+/- 0.01)  |  174.1  |  9.467  |
|  2  |  DECISION_TREE_CLASSIFIER  |  74.14%  |  [0.744  0.7276 0.7354 0.7408 0.7332]  |  0.74 (+/- 0.01)  |  12.23  |  0.01961  |
|  3  |  LINEAR_SVC  |  87.13%  |  [0.8862 0.8798 0.8838 0.8856 0.8876]  |  0.88 (+/- 0.01)  |  0.5033  |  0.006712  |
|  4  |  LOGISTIC_REGRESSION  |  87.75%  |  [0.8892 0.8856 0.889  0.8876 0.891 ]  |  0.89 (+/- 0.00)  |  1.837  |  0.01146  |
|  5  |  RANDOM_FOREST_CLASSIFIER  |  85.17%  |  [0.8576 0.8544 0.8528 0.8522 0.8546]  |  0.85 (+/- 0.00)  |  43.13  |  2.226  |


Best algorithm:
===> 4) LOGISTIC_REGRESSION
    Accuracy score = 87.75%   Training time = 1.837   Test time = 0.01146



DONE!
Program finished. It took 2816.1343352794647 seconds

Process finished with exit code 0
```