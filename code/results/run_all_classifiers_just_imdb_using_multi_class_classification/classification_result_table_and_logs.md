# Running all classifiers


### IMDB using Multi-Class Classification



#### FINAL CLASSIFICATION TABLE: IMDB_REVIEWS dataset (Multi-class classification)

| ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
| --- | ------------- | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
|  1  |  ADA_BOOST_CLASSIFIER  |  38.02%  |  [0.3792 0.379  0.374  0.3704 0.3746]  |  0.38 (+/- 0.01)  |  120.7  |  7.37  |
|  2  |  BERNOULLI_NB  |  37.03%  |  [0.377  0.389  0.3782 0.38   0.373 ]  |  0.38 (+/- 0.01)  |  0.04659  |  0.04255  |
|  3  |  COMPLEMENT_NB  |  37.34%  |  [0.3878 0.3942 0.3976 0.3938 0.3832]  |  0.39 (+/- 0.01)  |  0.0406  |  0.02028  |
|  4  |  DECISION_TREE_CLASSIFIER  |  31.34%  |  [0.3058 0.3122 0.3044 0.3036 0.311 ]  |  0.31 (+/- 0.01)  |  3.227  |  0.01253  |
|  5  |  GRADIENT_BOOSTING_CLASSIFIER  |  37.36%  |  [0.38   0.3688 0.3596 0.3634 0.3682]  |  0.37 (+/- 0.01)  |  433.1  |  0.2829  |
|  6  |  K_NEIGHBORS_CLASSIFIER  |  37.26%  |  [0.3822 0.3916 0.3842 0.386  0.388 ]  |  0.39 (+/- 0.01)  |  0.007069  |  14.19  |
|  7  |  LINEAR_SVC  |  40.80%  |  [0.41   0.4206 0.4064 0.3992 0.4088]  |  0.41 (+/- 0.01)  |  0.5336  |  0.0213  |
|  8  |  LOGISTIC_REGRESSION  |  42.31%  |  [0.4286 0.4296 0.4152 0.4178 0.4224]  |  0.42 (+/- 0.01)  |  2.719  |  0.02371  |
|  9  |  MULTINOMIAL_NB  |  37.82%  |  [0.389  0.3928 0.3918 0.3942 0.386 ]  |  0.39 (+/- 0.01)  |  0.03834  |  0.02014  |
|  10  |  NEAREST_CENTROID  |  37.33%  |  [0.3872 0.3786 0.3894 0.3672 0.3782]  |  0.38 (+/- 0.02)  |  0.03892  |  0.03368  |
|  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  41.81%  |  [0.4172 0.4284 0.4096 0.409  0.4164]  |  0.42 (+/- 0.01)  |  0.52  |  0.01889  |
|  12  |  PERCEPTRON  |  30.99%  |  [0.3294 0.318  0.3202 0.3254 0.3198]  |  0.32 (+/- 0.01)  |  0.5218  |  0.06201  |
|  13  |  RANDOM_FOREST_CLASSIFIER  |  37.03%  |  [0.3588 0.3712 0.3626 0.3668 0.36  ]  |  0.36 (+/- 0.01)  |  16.38  |  0.617  |
|  14  |  RIDGE_CLASSIFIER  |  38.55%  |  [0.4036 0.4074 0.402  0.3954 0.4   ]  |  0.40 (+/- 0.01)  |  2.942  |  0.04052  |


#### All logs

```
/home/ets-crchum/virtual_envs/comp551_p2/bin/python /home/ets-crchum/github/comp551-2020-p2_classification_of_textual_data/code/main.py --dataset IMDB_REVIEWS --use_imdb_multi_class_labels --run_cross_validation --report --all_metrics --confusion_matrix --plot_accurary_and_time_together -verbose
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
	Verbose = True
==================================================================================================================================

Loading IMDB_REVIEWS dataset:
03/07/2020 11:14:31 PM - INFO - Program started...

===> Reading files from /home/ets-crchum/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/train/neg

===> Reading files from /home/ets-crchum/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/train/pos

===> Reading files from /home/ets-crchum/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/test/neg

===> Reading files from /home/ets-crchum/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/test/pos
data loaded
25000 documents - 33.133MB (training set)
25000 documents - 32.351MB (test set)

Extracting features from the training data using a vectorizer
done in 2.989161s at 11.084MB/s
n_samples: 25000, n_features: 74170

Extracting features from the test data using the same vectorizer
done in 2.978358s at 10.862MB/s
n_samples: 25000, n_features: 74170

================================================================================
Classifier.ADA_BOOST_CLASSIFIER
________________________________________________________________________________
Training: 
AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=0.1,
                   n_estimators=500, random_state=0)
train time: 120.700s
test time:  7.370s
accuracy:   0.380


cross validation:
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/metrics/_classification.py:1272: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
[Parallel(n_jobs=-1)]: Done   2 out of   5 | elapsed:  5.3min remaining:  8.0min
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/metrics/_classification.py:1272: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/metrics/_classification.py:1272: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
	accuracy: 5-fold cross validation: [0.3792 0.379  0.374  0.3704 0.3746]
	test accuracy: 5-fold cross validation accuracy: 0.38 (+/- 0.01)


===> Classification Report:

[Parallel(n_jobs=-1)]: Done   5 out of   5 | elapsed:  7.8min finished
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
train time: 0.047s
test time:  0.043s
accuracy:   0.370


cross validation:
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
[Parallel(n_jobs=-1)]: Done   2 out of   5 | elapsed:    0.3s remaining:    0.4s
	accuracy: 5-fold cross validation: [0.377  0.389  0.3782 0.38   0.373 ]
	test accuracy: 5-fold cross validation accuracy: 0.38 (+/- 0.01)
dimensionality: 74170
density: 1.000000



===> Classification Report:

[Parallel(n_jobs=-1)]: Done   5 out of   5 | elapsed:    0.3s finished
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
train time: 0.041s
test time:  0.020s
accuracy:   0.373


cross validation:
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
[Parallel(n_jobs=-1)]: Done   2 out of   5 | elapsed:    0.3s remaining:    0.4s
[Parallel(n_jobs=-1)]: Done   5 out of   5 | elapsed:    0.4s finished
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
                       max_depth=19, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=250,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=0, splitter='random')
train time: 3.227s
test time:  0.013s
accuracy:   0.313


cross validation:
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
[Parallel(n_jobs=-1)]: Done   2 out of   5 | elapsed:    3.5s remaining:    5.3s
[Parallel(n_jobs=-1)]: Done   5 out of   5 | elapsed:    4.7s finished
	accuracy: 5-fold cross validation: [0.3058 0.3122 0.3044 0.3036 0.311 ]
	test accuracy: 5-fold cross validation accuracy: 0.31 (+/- 0.01)


===> Classification Report:

              precision    recall  f1-score   support

           1       0.40      0.67      0.50      5022
           2       0.13      0.02      0.03      2302
           3       0.14      0.03      0.05      2541
           4       0.18      0.10      0.13      2635
           7       0.14      0.06      0.08      2307
           8       0.20      0.09      0.12      2850
           9       0.14      0.01      0.01      2344
          10       0.31      0.74      0.43      4999

    accuracy                           0.31     25000
   macro avg       0.21      0.21      0.17     25000
weighted avg       0.24      0.31      0.23     25000



===> Classification Metrics:

accuracy classification score
	accuracy score:  0.31336
	accuracy score (normalize=False):  7834

compute the precision
	precision score (average=macro):  0.20546675267809772
	precision score (average=micro):  0.31336
	precision score (average=weighted):  0.23643846652626585
	precision score (average=None):  [0.39870053 0.13422819 0.14371257 0.1814346  0.13641026 0.1992278
 0.14150943 0.30851064]
	precision score (average=None, zero_division=1):  [0.39870053 0.13422819 0.14371257 0.1814346  0.13641026 0.1992278
 0.14150943 0.30851064]

compute the precision
	recall score (average=macro):  0.21337385388606603
	recall score (average=micro):  0.31336
	recall score (average=weighted):  0.31336
	recall score (average=None):  [0.67204301 0.01737619 0.0283353  0.09791271 0.05765063 0.09052632
 0.00639932 0.73674735]
	recall score (average=None, zero_division=1):  [0.67204301 0.01737619 0.0283353  0.09791271 0.05765063 0.09052632
 0.00639932 0.73674735]

compute the F1 score, also known as balanced F-score or F-measure
	f1 score (average=macro):  0.1698077790540665
	f1 score (average=micro):  0.31336
	f1 score (average=weighted):  0.23136950503539244
	f1 score (average=None):  [0.50048195 0.03076923 0.04733728 0.12718758 0.08104814 0.12448733
 0.0122449  0.43490583]

compute the F-beta score
	f beta score (average=macro):  0.17117743835478294
	f beta score (average=micro):  0.31336
	f beta score (average=weighted):  0.21738615046512277
	f beta score (average=None):  [0.43400545 0.05724098 0.07920792 0.15499219 0.1071371  0.16064757
 0.02709538 0.34909291]

compute the average Hamming loss
	hamming loss:  0.68664

jaccard similarity coefficient score
	jaccard score (average=macro):  0.10427389591959725
	jaccard score (average=None):  [0.33376187 0.015625   0.02424242 0.06791261 0.04223563 0.0663751
 0.00616016 0.27787838]

confusion matrix:
[[3375   78   93  249   96   79    8 1044]
 [1259   40   62  156   67   50    5  663]
 [1138   42   72  270   79   81    5  854]
 [ 995   51   91  258  131  100    3 1006]
 [ 428   22   46  151  133  212   10 1305]
 [ 418   16   43  122  166  258   13 1814]
 [ 295   17   35   79  118  216   15 1569]
 [ 557   32   59  137  185  299   47 3683]]

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
                           validation_fraction=0.1, verbose=True,
                           warm_start=False)
      Iter       Train Loss   Remaining Time 
         1       49963.4791            7.29m
         2       49393.7042            7.22m
         3       48915.5018            7.15m
         4       48503.8446            7.07m
         5       48111.8869            6.99m
         6       47770.5472            6.92m
         7       47448.4328            6.84m
         8       47148.8630            6.76m
         9       46872.8996            6.68m
        10       46615.3748            6.61m
        20       44594.4251            5.84m
        30       43122.9632            5.10m
        40       41983.7103            4.36m
        50       41013.0040            3.63m
        60       40200.9968            2.90m
        70       39486.0752            2.17m
        80       38842.4214            1.45m
        90       38205.7529           43.34s
       100       37674.2618            0.00s
train time: 433.118s
test time:  0.283s
accuracy:   0.374


cross validation:
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
      Iter       Train Loss   Remaining Time 
      Iter       Train Loss   Remaining Time 
      Iter       Train Loss   Remaining Time 
      Iter       Train Loss   Remaining Time 
      Iter       Train Loss   Remaining Time 
         1       39949.8375           12.16m
         1       39942.3140           12.22m
         1       39950.0175           12.23m
         1       39966.0639           21.01m
         1       39953.4182           21.15m
         2       39481.0536           12.00m
         2       39467.2520           12.00m
         2       39473.8487           12.01m
         3       39088.5332           11.84m
         3       39057.8840           11.82m
         3       39073.7688           11.85m
         2       39499.4057           20.78m
         2       39482.4417           20.93m
         4       38702.6269           11.67m
         4       38732.7388           11.68m
         4       38727.9375           11.70m
         5       38414.9400           11.53m
         5       38392.5900           11.53m
         5       38416.8160           11.54m
         3       39103.4497           20.57m
         3       39085.1823           20.72m
         6       38116.0044           11.41m
         6       38096.5552           11.43m
         6       38133.5970           11.44m
         4       38755.2011           20.34m
         7       37850.5050           11.28m
         7       37831.7705           11.29m
         7       37859.8187           11.31m
         4       38723.8122           20.48m
         8       37601.6102           11.14m
         8       37584.9329           11.14m
         8       37610.8438           11.17m
         5       38441.4535           20.12m
         5       38406.7903           20.25m
         9       37370.8355           11.00m
         9       37359.9297           11.01m
         9       37374.7936           11.04m
        10       37147.1936           10.87m
        10       37141.6516           10.87m
        10       37160.2993           10.90m
         6       38157.1591           19.89m
         6       38122.4898           20.01m
         7       37882.6231           19.67m
         7       37856.5777           19.79m
         8       37639.3393           19.45m
         8       37608.9425           19.56m
         9       37409.6890           19.22m
         9       37362.8568           19.34m
        10       37196.1793           19.00m
        10       37131.5293           19.11m
        20       35449.5624            9.58m
        20       35451.2618            9.61m
        20       35455.1937            9.61m
        30       34224.7479            8.35m
        30       34228.5818            8.36m
        30       34228.1227            8.38m
        20       35487.9814           16.84m
        20       35383.8124           16.91m
        40       33243.7930            7.14m
        40       33258.5000            7.14m
        40       33253.2658            7.16m
        30       34135.5343           13.71m
        50       32482.1647            5.94m
        50       32477.5774            5.97m
        30       34282.9684           14.72m
        50       32472.4035            6.44m
        40       33173.3125           10.58m
        60       31760.6770            4.75m
        60       31781.9939            4.76m
        50       32387.9968            8.22m
        70       31112.6801            3.55m
        70       31170.9227            3.56m
        40       33311.5499           12.62m
        60       31784.8170            5.68m
        60       31696.2032            6.26m
        80       30547.0898            2.37m
        80       30611.3367            2.37m
        50       32534.8363           10.51m
        70       31080.6290            4.53m
        70       31153.4164            4.54m
        90       30043.7083            1.18m
        90       30084.1827            1.18m
        80       30519.6301            2.93m
       100       29540.3508            0.00s
       100       29606.1502            0.00s
[Parallel(n_jobs=-1)]: Done   2 out of   5 | elapsed: 11.8min remaining: 17.8min
        60       31858.5363            8.16m
        80       30604.1141            3.07m
        90       29989.5078            1.43m
        70       31240.5840            5.74m
        90       30065.3598            1.49m
       100       29484.6763            0.00s
        80       30659.9333            3.63m
       100       29588.3585            0.00s
        90       30153.3098            1.74m
       100       29658.3655            0.00s
	accuracy: 5-fold cross validation: [0.38   0.3688 0.3596 0.3634 0.3682]
	test accuracy: 5-fold cross validation accuracy: 0.37 (+/- 0.01)


===> Classification Report:

[Parallel(n_jobs=-1)]: Done   5 out of   5 | elapsed: 16.7min finished
              precision    recall  f1-score   support

           1       0.47      0.76      0.58      5022
           2       0.21      0.04      0.07      2302
           3       0.27      0.08      0.12      2541
           4       0.27      0.17      0.21      2635
           7       0.26      0.15      0.19      2307
           8       0.23      0.17      0.20      2850
           9       0.15      0.03      0.04      2344
          10       0.38      0.78      0.51      4999

    accuracy                           0.37     25000
   macro avg       0.28      0.27      0.24     25000
weighted avg       0.31      0.37      0.30     25000



===> Classification Metrics:

accuracy classification score
	accuracy score:  0.3736
	accuracy score (normalize=False):  9340

compute the precision
	precision score (average=macro):  0.2801432762411965
	precision score (average=micro):  0.3736
	precision score (average=weighted):  0.3101675752553352
	precision score (average=None):  [0.46835132 0.205074   0.26745718 0.27222563 0.26426896 0.22741862
 0.15288221 0.3834683 ]
	precision score (average=None, zero_division=1):  [0.46835132 0.205074   0.26745718 0.27222563 0.26426896 0.22741862
 0.15288221 0.3834683 ]

compute the precision
	recall score (average=macro):  0.2717634819785187
	recall score (average=micro):  0.3736
	recall score (average=weighted):  0.3736
	recall score (average=None):  [0.76025488 0.04213727 0.07988981 0.16850095 0.14651062 0.17403509
 0.02602389 0.77675535]
	recall score (average=None, zero_division=1):  [0.76025488 0.04213727 0.07988981 0.16850095 0.14651062 0.17403509
 0.02602389 0.77675535]

compute the F1 score, also known as balanced F-score or F-measure
	f1 score (average=macro):  0.2405430056776401
	f1 score (average=micro):  0.3736
	f1 score (average=weighted):  0.3040318253018397
	f1 score (average=None):  [0.57962654 0.06990991 0.1230303  0.20815752 0.18851088 0.1971775
 0.04447685 0.51345455]

compute the F-beta score
	f beta score (average=macro):  0.24917023463638902
	f beta score (average=micro):  0.3736
	f beta score (average=weighted):  0.294614503932418
	f beta score (average=None):  [0.507308   0.11564139 0.18199749 0.24238454 0.22767075 0.21427337
 0.07741117 0.42667516]

compute the average Hamming loss
	hamming loss:  0.6264

jaccard similarity coefficient score
	jaccard score (average=macro):  0.1509499092736188
	jaccard score (average=None):  [0.40808038 0.03622106 0.0655473  0.11616954 0.10406404 0.10937155
 0.02274422 0.34540117]

confusion matrix:
[[3818  121  126  166   52   99   15  625]
 [1318   97  129  191   64   87   15  401]
 [1108   88  203  355  104  136   24  523]
 [ 887   91  174  444  177  227   27  608]
 [ 273   19   54  184  338  405   48  986]
 [ 256   27   29  134  279  496   86 1543]
 [ 165    7   17   73  134  330   61 1557]
 [ 327   23   27   84  131  401  123 3883]]

================================================================================
Classifier.K_NEIGHBORS_CLASSIFIER
________________________________________________________________________________
Training: 
KNeighborsClassifier(algorithm='auto', leaf_size=5, metric='euclidean',
                     metric_params=None, n_jobs=-1, n_neighbors=50, p=2,
                     weights='distance')
train time: 0.007s
test time:  14.195s
accuracy:   0.373


cross validation:
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
[Parallel(n_jobs=-1)]: Done   2 out of   5 | elapsed:    6.8s remaining:   10.1s
[Parallel(n_jobs=-1)]: Done   5 out of   5 | elapsed:    7.4s finished
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
          intercept_scaling=1, loss='squared_hinge', max_iter=5000,
          multi_class='crammer_singer', penalty='l2', random_state=0, tol=0.001,
          verbose=True)
********.***
optimization finished, #iter = 15
Objective value = -247.906008
nSV = 83211
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/svm/_base.py:947: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  "the number of iterations.", ConvergenceWarning)
[LibLinear]train time: 0.534s
test time:  0.021s
accuracy:   0.408


cross validation:
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
**********************.**************.**
optimization finished, #iter = 17
Objective value = -198.418979
nSV = 68611
*/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/svm/_base.py:947: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  "the number of iterations.", ConvergenceWarning)
**.*****.*
optimization finished, #iter = 14
Objective value = -198.442299
nSV = 69295
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/svm/_base.py:947: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  "the number of iterations.", ConvergenceWarning)
*[Parallel(n_jobs=-1)]: Done   2 out of   5 | elapsed:    1.1s remaining:    1.6s
**.*
optimization finished, #iter = 15
Objective value = -198.432294
nSV = 68927
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/svm/_base.py:947: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  "the number of iterations.", ConvergenceWarning)
***
optimization finished, #iter = 13
Objective value = -198.440115
nSV = 69281

optimization finished, #iter = 18
Objective value = -198.408928
nSV = 68947
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/svm/_base.py:947: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  "the number of iterations.", ConvergenceWarning)
/home/ets-crchum/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/svm/_base.py:947: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  "the number of iterations.", ConvergenceWarning)
	accuracy: 5-fold cross validation: [0.41   0.4206 0.4064 0.3992 0.4088]
	test accuracy: 5-fold cross validation accuracy: 0.41 (+/- 0.01)
dimensionality: 74170
density: 0.761615



===> Classification Report:

[Parallel(n_jobs=-1)]: Done   5 out of   5 | elapsed:    1.3s finished
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
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='ovr', n_jobs=-1, penalty='l2', random_state=0,
                   solver='lbfgs', tol=0.01, verbose=True, warm_start=False)
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
[Parallel(n_jobs=-1)]: Done   2 out of   8 | elapsed:    2.5s remaining:    7.4s
train time: 2.719s
[Parallel(n_jobs=-1)]: Done   8 out of   8 | elapsed:    2.7s finished
test time:  0.024s
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
accuracy:   0.423


cross validation:
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 8 concurrent workers.
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 8 concurrent workers.
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 8 concurrent workers.
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 8 concurrent workers.
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 8 concurrent workers.
[Parallel(n_jobs=-1)]: Done   2 out of   8 | elapsed:    9.6s remaining:   28.8s
[Parallel(n_jobs=-1)]: Done   2 out of   8 | elapsed:    9.9s remaining:   29.6s
[Parallel(n_jobs=-1)]: Done   2 out of   8 | elapsed:    9.9s remaining:   29.7s
[Parallel(n_jobs=-1)]: Done   2 out of   8 | elapsed:   10.0s remaining:   30.1s
[Parallel(n_jobs=-1)]: Done   2 out of   8 | elapsed:   10.3s remaining:   30.8s
[Parallel(n_jobs=-1)]: Done   8 out of   8 | elapsed:   11.0s finished
[Parallel(n_jobs=-1)]: Done   8 out of   8 | elapsed:   11.4s finished
[Parallel(n_jobs=-1)]: Done   2 out of   5 | elapsed:   11.6s remaining:   17.4s
[Parallel(n_jobs=-1)]: Done   8 out of   8 | elapsed:   11.6s finished
[Parallel(n_jobs=-1)]: Done   8 out of   8 | elapsed:   11.5s finished
[Parallel(n_jobs=-1)]: Done   8 out of   8 | elapsed:   11.6s finished
[Parallel(n_jobs=-1)]: Done   5 out of   5 | elapsed:   11.8s finished
	accuracy: 5-fold cross validation: [0.4286 0.4296 0.4152 0.4178 0.4224]
	test accuracy: 5-fold cross validation accuracy: 0.42 (+/- 0.01)
dimensionality: 74170
density: 1.000000



===> Classification Report:

              precision    recall  f1-score   support

           1       0.50      0.87      0.63      5022
           2       0.20      0.03      0.05      2302
           3       0.26      0.09      0.14      2541
           4       0.33      0.29      0.31      2635
           7       0.32      0.21      0.25      2307
           8       0.27      0.21      0.24      2850
           9       0.22      0.02      0.04      2344
          10       0.46      0.81      0.59      4999

    accuracy                           0.42     25000
   macro avg       0.32      0.32      0.28     25000
weighted avg       0.35      0.42      0.35     25000



===> Classification Metrics:

accuracy classification score
	accuracy score:  0.42312
	accuracy score (normalize=False):  10578

compute the precision
	precision score (average=macro):  0.31903531102019134
	precision score (average=micro):  0.42312
	precision score (average=weighted):  0.35211866134865705
	precision score (average=None):  [0.49584235 0.19871795 0.26393629 0.32902113 0.31857814 0.26696231
 0.216      0.46322433]
	precision score (average=None, zero_division=1):  [0.49584235 0.19871795 0.26393629 0.32902113 0.31857814 0.26696231
 0.216      0.46322433]

compute the precision
	recall score (average=macro):  0.3152884592192433
	recall score (average=micro):  0.42312
	recall score (average=weighted):  0.42312
	recall score (average=None):  [0.86678614 0.0269331  0.09130264 0.28956357 0.2058951  0.21122807
 0.02303754 0.80756151]
	recall score (average=None, zero_division=1):  [0.86678614 0.0269331  0.09130264 0.28956357 0.2058951  0.21122807
 0.02303754 0.80756151]

compute the F1 score, also known as balanced F-score or F-measure
	f1 score (average=macro):  0.2797902485648293
	f1 score (average=micro):  0.42312
	f1 score (average=weighted):  0.3489415337623386
	f1 score (average=None):  [0.63082385 0.04743688 0.13567251 0.30803391 0.25013165 0.23584721
 0.04163454 0.58874143]

compute the F-beta score
	f beta score (average=macro):  0.2836580409557108
	f beta score (average=micro):  0.42312
	f beta score (average=weighted):  0.3344313639674907
	f beta score (average=None):  [0.54225422 0.08732394 0.19151395 0.32029217 0.28714787 0.25358045
 0.08074163 0.5064101 ]

compute the average Hamming loss
	hamming loss:  0.57688

jaccard similarity coefficient score
	jaccard score (average=macro):  0.18186539377087713
	jaccard score (average=None):  [0.46073243 0.02429467 0.0727729  0.18205679 0.14294312 0.13368865
 0.02125984 0.41717474]

confusion matrix:
[[4353   61  109  195   26   35    3  240]
 [1563   62  141  263   50   44    6  173]
 [1222   82  232  568   99   85    7  246]
 [ 864   64  253  763  181  204   12  294]
 [ 189   14   68  247  475  517   41  756]
 [ 189   11   33  147  367  602   65 1436]
 [ 136    7   17   65  142  390   54 1533]
 [ 263   11   26   71  151  378   62 4037]]

================================================================================
Classifier.MULTINOMIAL_NB
________________________________________________________________________________
Training: 
MultinomialNB(alpha=0.1, class_prior=None, fit_prior=True)
train time: 0.038s
test time:  0.020s
accuracy:   0.378


cross validation:
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
[Parallel(n_jobs=-1)]: Done   2 out of   5 | elapsed:    0.2s remaining:    0.3s
	accuracy: 5-fold cross validation: [0.389  0.3928 0.3918 0.3942 0.386 ]
	test accuracy: 5-fold cross validation accuracy: 0.39 (+/- 0.01)
dimensionality: 74170
density: 1.000000



===> Classification Report:

[Parallel(n_jobs=-1)]: Done   5 out of   5 | elapsed:    0.3s finished
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
train time: 0.039s
test time:  0.034s
accuracy:   0.373


cross validation:
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
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
[Parallel(n_jobs=-1)]: Done   2 out of   5 | elapsed:    0.2s remaining:    0.3s
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

[Parallel(n_jobs=-1)]: Done   5 out of   5 | elapsed:    0.2s finished
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
                            loss='hinge', max_iter=100, n_iter_no_change=5,
                            n_jobs=-1, random_state=0, shuffle=True, tol=0.01,
                            validation_fraction=0.01, verbose=True,
                            warm_start=False)
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 8 concurrent workers.
-- Epoch 1
-- Epoch 1
-- Epoch 1
-- Epoch 1
-- Epoch 1
-- Epoch 1
-- Epoch 1
Norm: 5.14, NNZs: 50675, Bias: -0.504695, T: 24750, Avg. loss: 0.291683
Total training time: 0.03 seconds.
-- Epoch 1
Norm: 5.29, NNZs: 50233, Bias: -0.500445, T: 24750, Avg. loss: 0.275204
Total training time: 0.02 seconds.
Norm: 5.09, NNZs: 52085, Bias: -0.497469, T: 24750, Avg. loss: 0.313303
Total training time: 0.02 seconds.
Norm: 5.41, NNZs: 58761, Bias: -0.471007, T: 24750, Avg. loss: 0.492475
Total training time: 0.04 seconds.
Norm: 5.16, NNZs: 57803, Bias: -0.451356, T: 24750, Avg. loss: 0.464166
Total training time: 0.05 seconds.
-- Epoch 2
-- Epoch 2
Norm: 5.05, NNZs: 53357, Bias: -0.492494, T: 24750, Avg. loss: 0.338070
Total training time: 0.05 seconds.
Norm: 5.19, NNZs: 51557, Bias: -0.503904, T: 24750, Avg. loss: 0.296568
Total training time: 0.07 seconds.
Norm: 4.78, NNZs: 51056, Bias: -0.574962, T: 49500, Avg. loss: 0.213019
Total training time: 0.08 seconds.
Norm: 5.21, NNZs: 49888, Bias: -0.504167, T: 24750, Avg. loss: 0.279446
Total training time: 0.08 seconds.
Norm: 4.88, NNZs: 50380, Bias: -0.559639, T: 49500, Avg. loss: 0.199770
Total training time: 0.08 seconds.
-- Epoch 2
-- Epoch 2
-- Epoch 2
-- Epoch 3
-- Epoch 2
Norm: 4.76, NNZs: 52516, Bias: -0.573090, T: 49500, Avg. loss: 0.235233
Total training time: 0.10 seconds.
Norm: 5.52, NNZs: 58301, Bias: -0.529210, T: 49500, Avg. loss: 0.385002
Total training time: 0.12 seconds.
Norm: 6.34, NNZs: 59703, Bias: -0.552904, T: 49500, Avg. loss: 0.400136
Total training time: 0.12 seconds.
-- Epoch 2
-- Epoch 2
Norm: 4.63, NNZs: 51556, Bias: -0.638858, T: 74250, Avg. loss: 0.205538
Total training time: 0.13 seconds.
-- Epoch 3
Norm: 4.71, NNZs: 53732, Bias: -0.571417, T: 49500, Avg. loss: 0.261041
Total training time: 0.11 seconds.
Norm: 4.75, NNZs: 51893, Bias: -0.570502, T: 49500, Avg. loss: 0.219936
Total training time: 0.14 seconds.
Norm: 4.86, NNZs: 50212, Bias: -0.568145, T: 49500, Avg. loss: 0.200722
Total training time: 0.14 seconds.
Norm: 4.68, NNZs: 50728, Bias: -0.619185, T: 74250, Avg. loss: 0.193823
Total training time: 0.13 seconds.
-- Epoch 3
-- Epoch 3
-- Epoch 3
-- Epoch 4
-- Epoch 3
Norm: 7.46, NNZs: 60412, Bias: -0.603771, T: 74250, Avg. loss: 0.383905
Total training time: 0.17 seconds.
-- Epoch 3
-- Epoch 3
Norm: 6.10, NNZs: 58860, Bias: -0.587430, T: 74250, Avg. loss: 0.373782
Total training time: 0.18 seconds.
-- Epoch 4
Norm: 4.69, NNZs: 53186, Bias: -0.641575, T: 74250, Avg. loss: 0.226947
Total training time: 0.17 seconds.
Norm: 4.63, NNZs: 52126, Bias: -0.695224, T: 99000, Avg. loss: 0.200370
Total training time: 0.19 seconds.
Norm: 4.61, NNZs: 54202, Bias: -0.641985, T: 74250, Avg. loss: 0.252744
Total training time: 0.17 seconds.
Norm: 4.59, NNZs: 52345, Bias: -0.636320, T: 74250, Avg. loss: 0.212353
Total training time: 0.19 seconds.
Norm: 4.75, NNZs: 50772, Bias: -0.630059, T: 74250, Avg. loss: 0.194115
Total training time: 0.19 seconds.
Norm: 4.62, NNZs: 51108, Bias: -0.672989, T: 99000, Avg. loss: 0.188951
Total training time: 0.19 seconds.
-- Epoch 4
-- Epoch 4
-- Epoch 4
Norm: 8.66, NNZs: 60887, Bias: -0.638859, T: 99000, Avg. loss: 0.372471
Total training time: 0.23 seconds.
-- Epoch 5
-- Epoch 4
-- Epoch 4
-- Epoch 4
Norm: 6.77, NNZs: 59301, Bias: -0.631705, T: 99000, Avg. loss: 0.366236
Total training time: 0.24 seconds.
-- Epoch 5
Norm: 4.68, NNZs: 54814, Bias: -0.703729, T: 99000, Avg. loss: 0.246445
Total training time: 0.23 seconds.
Norm: 4.63, NNZs: 53039, Bias: -0.701197, T: 99000, Avg. loss: 0.206554
Total training time: 0.25 seconds.
Norm: 4.72, NNZs: 51249, Bias: -0.681518, T: 99000, Avg. loss: 0.189539
Total training time: 0.25 seconds.
Norm: 4.75, NNZs: 53874, Bias: -0.699789, T: 99000, Avg. loss: 0.221244
Total training time: 0.23 seconds.
Norm: 4.70, NNZs: 52649, Bias: -0.742161, T: 123750, Avg. loss: 0.196748
Total training time: 0.26 seconds.
Norm: 4.66, NNZs: 51559, Bias: -0.720032, T: 123750, Avg. loss: 0.185158
Total training time: 0.25 seconds.
-- Epoch 5
-- Epoch 5
-- Epoch 5
-- Epoch 5
Norm: 9.90, NNZs: 61207, Bias: -0.658873, T: 123750, Avg. loss: 0.362867
Total training time: 0.29 seconds.
-- Epoch 5
-- Epoch 5
-- Epoch 6
-- Epoch 6
Norm: 7.51, NNZs: 59682, Bias: -0.666030, T: 123750, Avg. loss: 0.360566
Total training time: 0.30 seconds.
Norm: 4.85, NNZs: 55590, Bias: -0.756193, T: 123750, Avg. loss: 0.242092
Total training time: 0.29 seconds.
Norm: 4.72, NNZs: 53636, Bias: -0.747843, T: 123750, Avg. loss: 0.202341
Total training time: 0.31 seconds.
Norm: 4.76, NNZs: 51719, Bias: -0.725057, T: 123750, Avg. loss: 0.186232
Total training time: 0.31 seconds.
Norm: 4.87, NNZs: 54530, Bias: -0.744723, T: 123750, Avg. loss: 0.217340
Total training time: 0.29 seconds.
Norm: 4.79, NNZs: 53226, Bias: -0.774771, T: 148500, Avg. loss: 0.194131
Total training time: 0.31 seconds.
Norm: 4.77, NNZs: 52072, Bias: -0.758466, T: 148500, Avg. loss: 0.182278
Total training time: 0.30 seconds.
-- Epoch 6
-- Epoch 6
-- Epoch 6
-- Epoch 6
Norm: 11.18, NNZs: 61515, Bias: -0.678481, T: 148500, Avg. loss: 0.354174
Total training time: 0.35 seconds.
-- Epoch 6
-- Epoch 6
Convergence after 6 epochs took 0.34 seconds
Convergence after 6 epochs took 0.36 seconds
[Parallel(n_jobs=-1)]: Done   2 out of   8 | elapsed:    0.4s remaining:    1.3s
Norm: 8.26, NNZs: 60033, Bias: -0.687031, T: 148500, Avg. loss: 0.355899
Total training time: 0.36 seconds.
Norm: 4.88, NNZs: 54335, Bias: -0.787859, T: 148500, Avg. loss: 0.199347
Total training time: 0.37 seconds.
Norm: 5.04, NNZs: 56288, Bias: -0.792170, T: 148500, Avg. loss: 0.239025
Total training time: 0.35 seconds.
Norm: 4.84, NNZs: 52135, Bias: -0.758036, T: 148500, Avg. loss: 0.183735
Total training time: 0.37 seconds.
Norm: 5.04, NNZs: 55200, Bias: -0.778700, T: 148500, Avg. loss: 0.214581
Total training time: 0.35 seconds.
Convergence after 6 epochs took 0.37 seconds
Convergence after 6 epochs took 0.39 seconds
Convergence after 6 epochs took 0.40 seconds
Convergence after 6 epochs took 0.38 seconds
Convergence after 6 epochs took 0.40 seconds
Convergence after 6 epochs took 0.38 seconds
[Parallel(n_jobs=-1)]: Done   8 out of   8 | elapsed:    0.5s finished
train time: 0.520s
test time:  0.019s
accuracy:   0.418


cross validation:
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 8 concurrent workers.
-- Epoch 1
-- Epoch 1
-- Epoch 1
-- Epoch 1
-- Epoch 1
-- Epoch 1
-- Epoch 1
Norm: 5.21, NNZs: 50133, Bias: -0.477816, T: 19800, Avg. loss: 0.354920
Total training time: 0.01 seconds.
Norm: 5.20, NNZs: 54364, Bias: -0.436523, T: 19800, Avg. loss: 0.480607
Total training time: 0.02 seconds.
Norm: 5.25, NNZs: 54933, Bias: -0.451372, T: 19800, Avg. loss: 0.512837
Total training time: 0.02 seconds.
Norm: 5.29, NNZs: 48499, Bias: -0.491122, T: 19800, Avg. loss: 0.309146
Total training time: 0.02 seconds.
-- Epoch 2
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 8 concurrent workers.
-- Epoch 1
-- Epoch 2
Norm: 5.37, NNZs: 47800, Bias: -0.484199, T: 19800, Avg. loss: 0.292726
Total training time: 0.03 seconds.
-- Epoch 2
Norm: 5.23, NNZs: 49302, Bias: -0.481795, T: 19800, Avg. loss: 0.330291
Total training time: 0.02 seconds.
Norm: 4.83, NNZs: 50282, Bias: -0.535038, T: 39600, Avg. loss: 0.263178
Total training time: 0.04 seconds.
-- Epoch 2
Norm: 5.32, NNZs: 47051, Bias: -0.488949, T: 19800, Avg. loss: 0.297232
Total training time: 0.03 seconds.
-- Epoch 2
Norm: 5.35, NNZs: 54589, Bias: -0.494067, T: 39600, Avg. loss: 0.388240
Total training time: 0.04 seconds.
Norm: 5.27, NNZs: 48869, Bias: -0.483789, T: 19800, Avg. loss: 0.313897
Total training time: 0.01 seconds.
Norm: 4.90, NNZs: 48637, Bias: -0.542824, T: 39600, Avg. loss: 0.215200
Total training time: 0.04 seconds.
-- Epoch 2
Norm: 5.01, NNZs: 47869, Bias: -0.531393, T: 39600, Avg. loss: 0.201302
Total training time: 0.05 seconds.
-- Epoch 2
-- Epoch 3
Norm: 5.95, NNZs: 55460, Bias: -0.527220, T: 39600, Avg. loss: 0.406092
Total training time: 0.06 seconds.
-- Epoch 2
-- Epoch 3
Norm: 4.89, NNZs: 49561, Bias: -0.540798, T: 39600, Avg. loss: 0.237451
Total training time: 0.05 seconds.
-- Epoch 3
-- Epoch 3
Norm: 4.77, NNZs: 49065, Bias: -0.599295, T: 59400, Avg. loss: 0.208073
Total training time: 0.07 seconds.
-- Epoch 3
-- Epoch 3
Norm: 4.73, NNZs: 50596, Bias: -0.596566, T: 59400, Avg. loss: 0.255596
Total training time: 0.08 seconds.
Norm: 4.91, NNZs: 48945, Bias: -0.541473, T: 39600, Avg. loss: 0.221761
Total training time: 0.05 seconds.
Norm: 4.83, NNZs: 48094, Bias: -0.581224, T: 59400, Avg. loss: 0.195832
Total training time: 0.08 seconds.
Norm: 6.83, NNZs: 56065, Bias: -0.579594, T: 59400, Avg. loss: 0.389747
Total training time: 0.09 seconds.
Norm: 4.98, NNZs: 47203, Bias: -0.539845, T: 39600, Avg. loss: 0.203021
Total training time: 0.07 seconds.
Norm: 4.77, NNZs: 49937, Bias: -0.596678, T: 59400, Avg. loss: 0.229624
Total training time: 0.07 seconds.
Norm: 5.84, NNZs: 54945, Bias: -0.552105, T: 59400, Avg. loss: 0.376908
Total training time: 0.09 seconds.
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 8 concurrent workers.
-- Epoch 4
-- Epoch 4
-- Epoch 3
-- Epoch 4
-- Epoch 4
-- Epoch 4
-- Epoch 4
Norm: 4.76, NNZs: 48369, Bias: -0.627233, T: 79200, Avg. loss: 0.191143
Total training time: 0.14 seconds.
Norm: 6.40, NNZs: 55302, Bias: -0.595583, T: 79200, Avg. loss: 0.368877
Total training time: 0.15 seconds.
-- Epoch 1
Norm: 7.77, NNZs: 56420, Bias: -0.613615, T: 79200, Avg. loss: 0.378469
Total training time: 0.16 seconds.
-- Epoch 1
-- Epoch 1
Norm: 4.74, NNZs: 50969, Bias: -0.648320, T: 79200, Avg. loss: 0.249320
Total training time: 0.16 seconds.
Norm: 4.75, NNZs: 49392, Bias: -0.648831, T: 79200, Avg. loss: 0.202690
Total training time: 0.16 seconds.
-- Epoch 5
Norm: 4.79, NNZs: 50362, Bias: -0.646823, T: 79200, Avg. loss: 0.223582
Total training time: 0.16 seconds.
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 8 concurrent workers.
-- Epoch 1
Norm: 4.88, NNZs: 47608, Bias: -0.594421, T: 59400, Avg. loss: 0.196430
Total training time: 0.16 seconds.
-- Epoch 1
-- Epoch 1
-- Epoch 5
-- Epoch 5
Norm: 6.98, NNZs: 55645, Bias: -0.623374, T: 99000, Avg. loss: 0.362747
Total training time: 0.19 seconds.
Norm: 5.29, NNZs: 47264, Bias: -0.487612, T: 19800, Avg. loss: 0.297396
Total training time: 0.03 seconds.
-- Epoch 1
-- Epoch 5
-- Epoch 1
Norm: 5.25, NNZs: 54903, Bias: -0.451655, T: 19800, Avg. loss: 0.512983
Norm: 5.29, NNZs: 48162, Bias: -0.491223, T: 19800, Avg. loss: 0.309229
Total training time: 0.05 seconds.
Total training time: 0.03 seconds.
-- Epoch 5
-- Epoch 3
-- Epoch 4
-- Epoch 5
Norm: 5.23, NNZs: 54298, Bias: -0.438241, T: 19800, Avg. loss: 0.480561
Total training time: 0.06 seconds.
Norm: 5.35, NNZs: 47505, Bias: -0.483226, T: 19800, Avg. loss: 0.293072
Total training time: 0.07 seconds.
Norm: 4.77, NNZs: 48683, Bias: -0.667144, T: 99000, Avg. loss: 0.187237
Total training time: 0.23 seconds.
-- Epoch 6
-- Epoch 2
Norm: 4.87, NNZs: 51482, Bias: -0.696129, T: 99000, Avg. loss: 0.244463
Total training time: 0.25 seconds.
Norm: 5.21, NNZs: 50152, Bias: -0.477127, T: 19800, Avg. loss: 0.354438
Total training time: 0.06 seconds.
[LibLinear]-- Epoch 1
Norm: 5.23, NNZs: 49358, Bias: -0.483220, T: 19800, Avg. loss: 0.330591
Total training time: 0.07 seconds.
-- Epoch 1
-- Epoch 1
Norm: 4.84, NNZs: 47987, Bias: -0.639853, T: 79200, Avg. loss: 0.191609
Total training time: 0.26 seconds.
-- Epoch 1
-- Epoch 1
-- Epoch 2
Norm: 5.28, NNZs: 48735, Bias: -0.486643, T: 19800, Avg. loss: 0.314323
Total training time: 0.10 seconds.
-- Epoch 1
-- Epoch 2
-- Epoch 1
Norm: 5.95, NNZs: 55500, Bias: -0.527137, T: 39600, Avg. loss: 0.405934
Total training time: 0.12 seconds.
Norm: 4.78, NNZs: 49822, Bias: -0.689134, T: 99000, Avg. loss: 0.198683
Total training time: 0.30 seconds.
-- Epoch 1
-- Epoch 2
-- Epoch 2
-- Epoch 2
Norm: 4.71, NNZs: 49222, Bias: -0.593302, T: 59400, Avg. loss: 0.214919
Total training time: 0.28 seconds.
Norm: 7.63, NNZs: 56042, Bias: -0.650915, T: 118800, Avg. loss: 0.357779
Total training time: 0.31 seconds.
-- Epoch 6
Norm: 4.92, NNZs: 48322, Bias: -0.544678, T: 39600, Avg. loss: 0.215221
Total training time: 0.16 seconds.
Norm: 4.97, NNZs: 47429, Bias: -0.539917, T: 39600, Avg. loss: 0.202862
Total training time: 0.16 seconds.
Norm: 5.31, NNZs: 47429, Bias: -0.490204, T: 19800, Avg. loss: 0.297710
Total training time: 0.05 seconds.
Norm: 8.75, NNZs: 56808, Bias: -0.634452, T: 99000, Avg. loss: 0.369330
-- Epoch 6
Total training time: 0.33 seconds.
-- Epoch 6
Norm: 5.02, NNZs: 47581, Bias: -0.534272, T: 39600, Avg. loss: 0.201522
Total training time: 0.16 seconds.
Norm: 4.92, NNZs: 50871, Bias: -0.692429, T: 99000, Avg. loss: 0.219181
Total training time: 0.32 seconds.
Norm: 5.30, NNZs: 48485, Bias: -0.493238, T: 19800, Avg. loss: 0.309577
Total training time: 0.06 seconds.
Norm: 5.27, NNZs: 49186, Bias: -0.487181, T: 19800, Avg. loss: 0.314774
Total training time: 0.08 seconds.
Norm: 5.22, NNZs: 54451, Bias: -0.438179, T: 19800, Avg. loss: 0.480750
Total training time: 0.07 seconds.
Norm: 4.91, NNZs: 49538, Bias: -0.542325, T: 39600, Avg. loss: 0.237341
Total training time: 0.15 seconds.
Norm: 4.85, NNZs: 50324, Bias: -0.534970, T: 39600, Avg. loss: 0.263132
Total training time: 0.15 seconds.
Norm: 5.37, NNZs: 47634, Bias: -0.484475, T: 19800, Avg. loss: 0.292931
Total training time: 0.07 seconds.
-- Epoch 3
-- Epoch 5
-- Epoch 4
-- Epoch 3
Convergence after 6 epochs took 0.36 seconds
Norm: 5.21, NNZs: 49323, Bias: -0.484560, T: 19800, Avg. loss: 0.331659
Total training time: 0.07 seconds.
Norm: 5.28, NNZs: 54962, Bias: -0.453100, T: 19800, Avg. loss: 0.512753
Total training time: 0.10 seconds.
Norm: 4.85, NNZs: 49040, Bias: -0.704304, T: 118800, Avg. loss: 0.184102
Total training time: 0.37 seconds.
Norm: 5.22, NNZs: 50213, Bias: -0.478597, T: 19800, Avg. loss: 0.354565
Total training time: 0.08 seconds.
Norm: 4.86, NNZs: 48349, Bias: -0.678063, T: 99000, Avg. loss: 0.188020
Total training time: 0.36 seconds.
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 8 concurrent workers.
Norm: 5.05, NNZs: 51965, Bias: -0.735837, T: 118800, Avg. loss: 0.240842
Total training time: 0.39 seconds.
-- Epoch 6
-- Epoch 3
-- Epoch 3
-- Epoch 6
Norm: 4.70, NNZs: 49672, Bias: -0.648130, T: 79200, Avg. loss: 0.209303
Total training time: 0.38 seconds.
-- Epoch 6
-- Epoch 2
Norm: 6.81, NNZs: 56054, Bias: -0.577312, T: 59400, Avg. loss: 0.389726
Convergence after 6 epochs took 0.42 seconds
-- Epoch 2
Norm: 4.88, NNZs: 50191, Bias: -0.724151, T: 118800, Avg. loss: 0.195671
Total training time: 0.42 seconds.
-- Epoch 3
[Parallel(n_jobs=-1)]: Done   2 out of   8 | elapsed:    0.4s remaining:    1.3s
-- Epoch 2
Norm: 4.73, NNZs: 50620, Bias: -0.594443, T: 59400, Avg. loss: 0.255535
Total training time: 0.24 seconds.
Convergence after 6 epochs took 0.44 seconds
-- Epoch 2
Total training time: 0.26 seconds.
-- Epoch 5
Norm: 5.07, NNZs: 51314, Bias: -0.727721, T: 118800, Avg. loss: 0.215921
Total training time: 0.43 seconds.
-- Epoch 2
Norm: 4.78, NNZs: 48721, Bias: -0.600229, T: 59400, Avg. loss: 0.208078
Total training time: 0.30 seconds.
Convergence after 6 epochs took 0.46 seconds
Norm: 4.87, NNZs: 47763, Bias: -0.595049, T: 59400, Avg. loss: 0.196333
Total training time: 0.30 seconds.
Norm: 9.78, NNZs: 56999, Bias: -0.651899, T: 118800, Avg. loss: 0.361257
Total training time: 0.46 seconds.
Norm: 5.38, NNZs: 54571, Bias: -0.496053, T: 39600, Avg. loss: 0.388171
Convergence after 6 epochs took 0.45 seconds
Norm: 4.94, NNZs: 48609, Bias: -0.711216, T: 118800, Avg. loss: 0.185218
Total training time: 0.46 seconds.
Total training time: 0.30 seconds.
-- Epoch 3
-- Epoch 2
Norm: 4.93, NNZs: 49294, Bias: -0.546714, T: 39600, Avg. loss: 0.221790
Total training time: 0.23 seconds.
-- Epoch 1
-- Epoch 1
-- Epoch 1
Norm: 4.92, NNZs: 48673, Bias: -0.547164, T: 39600, Avg. loss: 0.215347
-- Epoch 2
Total training time: 0.23 seconds.
-- Epoch 2
-- Epoch 1
Norm: 4.76, NNZs: 50131, Bias: -0.692908, T: 99000, Avg. loss: 0.204794
Total training time: 0.47 seconds.
Norm: 4.90, NNZs: 48825, Bias: -0.541025, T: 39600, Avg. loss: 0.221730
Total training time: 0.33 seconds.
-- Epoch 1
Convergence after 6 epochs took 0.53 seconds
Norm: 4.82, NNZs: 47764, Bias: -0.583013, T: 59400, Avg. loss: 0.196027
Total training time: 0.37 seconds.
Convergence after 6 epochs took 0.52 seconds
-- Epoch 6
-- Epoch 4
-- Epoch 1
-- Epoch 2
-- Epoch 1
-- Epoch 1
-- Epoch 2
Norm: 5.03, NNZs: 47680, Bias: -0.534131, T: 39600, Avg. loss: 0.201185
Total training time: 0.27 seconds.
Norm: 5.31, NNZs: 48213, Bias: -0.491530, T: 19800, Avg. loss: 0.308918
Total training time: 0.07 seconds.
Norm: 4.88, NNZs: 49475, Bias: -0.545545, T: 39600, Avg. loss: 0.237744
Total training time: 0.27 seconds.
Norm: 5.21, NNZs: 54137, Bias: -0.436390, T: 19800, Avg. loss: 0.480412
Total training time: 0.07 seconds.
Norm: 5.36, NNZs: 54706, Bias: -0.495503, T: 39600, Avg. loss: 0.388377
Total training time: 0.30 seconds.
-- Epoch 4
Norm: 4.98, NNZs: 47561, Bias: -0.541936, T: 39600, Avg. loss: 0.202796
Total training time: 0.32 seconds.
Norm: 4.78, NNZs: 49906, Bias: -0.596726, T: 59400, Avg. loss: 0.229523
Total training time: 0.39 seconds.
-- Epoch 3
[LibLinear]-- Epoch 1
-- Epoch 1
-- Epoch 1
Norm: 5.36, NNZs: 47675, Bias: -0.484201, T: 19800, Avg. loss: 0.292999
Total training time: 0.12 seconds.
Norm: 4.83, NNZs: 48127, Bias: -0.639795, T: 79200, Avg. loss: 0.191545
Total training time: 0.45 seconds.
Norm: 5.22, NNZs: 50227, Bias: -0.473922, T: 19800, Avg. loss: 0.353335
Total training time: 0.15 seconds.
-- Epoch 1
-- Epoch 1
Norm: 4.88, NNZs: 50583, Bias: -0.731442, T: 118800, Avg. loss: 0.201377
Total training time: 0.59 seconds.
-- Epoch 3
-- Epoch 1
Norm: 5.29, NNZs: 54692, Bias: -0.452982, T: 19800, Avg. loss: 0.512446
Total training time: 0.09 seconds.
-- Epoch 1
Norm: 4.82, NNZs: 50418, Bias: -0.535725, T: 39600, Avg. loss: 0.263765
Total training time: 0.35 seconds.
-- Epoch 4
Norm: 5.20, NNZs: 49180, Bias: -0.482316, T: 19800, Avg. loss: 0.331312
Total training time: 0.09 seconds.
-- Epoch 1
-- Epoch 4
Norm: 4.75, NNZs: 49106, Bias: -0.648818, T: 79200, Avg. loss: 0.202678
Total training time: 0.49 seconds.
Convergence after 6 epochs took 0.62 seconds
-- Epoch 3
Norm: 5.24, NNZs: 48894, Bias: -0.484541, T: 19800, Avg. loss: 0.314912
Total training time: 0.10 seconds.
-- Epoch 3
Norm: 5.33, NNZs: 47673, Bias: -0.489583, T: 19800, Avg. loss: 0.296941
Total training time: 0.15 seconds.
[Parallel(n_jobs=-1)]: Done   8 out of   8 | elapsed:    0.7s finished
Norm: 5.86, NNZs: 54886, Bias: -0.553551, T: 59400, Avg. loss: 0.376746
Total training time: 0.50 seconds.
Norm: 5.23, NNZs: 50443, Bias: -0.476201, T: 19800, Avg. loss: 0.353670
Total training time: 0.08 seconds.
-- Epoch 3
Norm: 5.32, NNZs: 47651, Bias: -0.488986, T: 19800, Avg. loss: 0.297025
Total training time: 0.07 seconds.
-- Epoch 3
-- Epoch 3
Norm: 5.38, NNZs: 47945, Bias: -0.487338, T: 19800, Avg. loss: 0.293178
Total training time: 0.08 seconds.
-- Epoch 3
Norm: 5.95, NNZs: 55500, Bias: -0.526022, T: 39600, Avg. loss: 0.405741
Total training time: 0.43 seconds.
Norm: 4.75, NNZs: 50966, Bias: -0.647983, T: 79200, Avg. loss: 0.249242
Total training time: 0.51 seconds.
-- Epoch 4
Norm: 5.28, NNZs: 48318, Bias: -0.491239, T: 19800, Avg. loss: 0.309557
Total training time: 0.09 seconds.
Norm: 4.76, NNZs: 49015, Bias: -0.601208, T: 59400, Avg. loss: 0.208158
Total training time: 0.44 seconds.
Norm: 4.70, NNZs: 49053, Bias: -0.592838, T: 59400, Avg. loss: 0.214825
Total training time: 0.54 seconds.
-- Epoch 2
Norm: 5.28, NNZs: 54956, Bias: -0.453591, T: 19800, Avg. loss: 0.512883
Total training time: 0.08 seconds.
-- Epoch 2
Norm: 4.69, NNZs: 49516, Bias: -0.594065, T: 59400, Avg. loss: 0.214790
Total training time: 0.47 seconds.
Norm: 5.26, NNZs: 49000, Bias: -0.485728, T: 19800, Avg. loss: 0.314613
Total training time: 0.10 seconds.
-- Epoch 2
-- Epoch 2
Norm: 4.83, NNZs: 47879, Bias: -0.582602, T: 59400, Avg. loss: 0.195741
Total training time: 0.45 seconds.
-- Epoch 3
-- Epoch 4
Norm: 7.77, NNZs: 56485, Bias: -0.615083, T: 79200, Avg. loss: 0.378564
Total training time: 0.56 seconds.
Norm: 5.18, NNZs: 49411, Bias: -0.483379, T: 19800, Avg. loss: 0.332090
Total training time: 0.12 seconds.
-- Epoch 5
-- Epoch 2
Norm: 4.73, NNZs: 49800, Bias: -0.600348, T: 59400, Avg. loss: 0.229924
Total training time: 0.45 seconds.
-- Epoch 2
Norm: 5.23, NNZs: 54538, Bias: -0.438451, T: 19800, Avg. loss: 0.480668
Total training time: 0.13 seconds.
Norm: 4.86, NNZs: 47936, Bias: -0.594693, T: 59400, Avg. loss: 0.196293
Total training time: 0.49 seconds.
Norm: 5.01, NNZs: 47755, Bias: -0.532711, T: 39600, Avg. loss: 0.201315
Total training time: 0.29 seconds.
-- Epoch 4
Norm: 4.81, NNZs: 50312, Bias: -0.646484, T: 79200, Avg. loss: 0.223585
Total training time: 0.59 seconds.
Norm: 5.84, NNZs: 55043, Bias: -0.552406, T: 59400, Avg. loss: 0.377104
Total training time: 0.50 seconds.
Norm: 4.86, NNZs: 50442, Bias: -0.532285, T: 39600, Avg. loss: 0.263350
Total training time: 0.32 seconds.
-- Epoch 5
Norm: 4.75, NNZs: 48002, Bias: -0.629946, T: 79200, Avg. loss: 0.191320
Total training time: 0.63 seconds.
Norm: 4.89, NNZs: 48315, Bias: -0.540980, T: 39600, Avg. loss: 0.215249
Total training time: 0.30 seconds.
-- Epoch 2
-- Epoch 2
-- Epoch 2
-- Epoch 2
Norm: 5.39, NNZs: 54402, Bias: -0.495509, T: 39600, Avg. loss: 0.387961
Total training time: 0.31 seconds.
Norm: 4.85, NNZs: 48494, Bias: -0.677966, T: 99000, Avg. loss: 0.187955
Total training time: 0.65 seconds.
Norm: 5.96, NNZs: 55204, Bias: -0.524782, T: 39600, Avg. loss: 0.405602
Total training time: 0.28 seconds.
Norm: 4.73, NNZs: 50735, Bias: -0.600487, T: 59400, Avg. loss: 0.256017
Total training time: 0.53 seconds.
-- Epoch 2
Norm: 4.89, NNZs: 49336, Bias: -0.543361, T: 39600, Avg. loss: 0.237374
Total training time: 0.28 seconds.
-- Epoch 3
-- Epoch 4
-- Epoch 4
-- Epoch 4
Norm: 6.41, NNZs: 55179, Bias: -0.594797, T: 79200, Avg. loss: 0.368691
Total training time: 0.68 seconds.
Norm: 4.78, NNZs: 49477, Bias: -0.689250, T: 99000, Avg. loss: 0.198701
Total training time: 0.70 seconds.
-- Epoch 4
-- Epoch 2
Norm: 5.00, NNZs: 47827, Bias: -0.540177, T: 39600, Avg. loss: 0.202793
Total training time: 0.35 seconds.
Norm: 4.87, NNZs: 50693, Bias: -0.534410, T: 39600, Avg. loss: 0.263339
Total training time: 0.28 seconds.
-- Epoch 5
-- Epoch 5
Norm: 4.70, NNZs: 49439, Bias: -0.647262, T: 79200, Avg. loss: 0.209192
Total training time: 0.70 seconds.
-- Epoch 2
Norm: 5.02, NNZs: 48018, Bias: -0.534393, T: 39600, Avg. loss: 0.201274
Total training time: 0.29 seconds.
-- Epoch 5
-- Epoch 2
Norm: 4.91, NNZs: 49022, Bias: -0.544624, T: 39600, Avg. loss: 0.221806
Total training time: 0.34 seconds.
-- Epoch 2
-- Epoch 4
-- Epoch 3
Norm: 4.71, NNZs: 49916, Bias: -0.650614, T: 79200, Avg. loss: 0.209140
Total training time: 0.65 seconds.
Norm: 6.83, NNZs: 56091, Bias: -0.576818, T: 59400, Avg. loss: 0.389275
-- Epoch 4
Norm: 5.00, NNZs: 47798, Bias: -0.540914, T: 39600, Avg. loss: 0.202873
Total training time: 0.30 seconds.
Norm: 4.73, NNZs: 49431, Bias: -0.650021, T: 79200, Avg. loss: 0.202688
Total training time: 0.64 seconds.
-- Epoch 4
-- Epoch 2
-- Epoch 6
Total training time: 0.65 seconds.
Norm: 4.77, NNZs: 48130, Bias: -0.629823, T: 79200, Avg. loss: 0.191109
Total training time: 0.63 seconds.
Norm: 8.75, NNZs: 56798, Bias: -0.635205, T: 99000, Avg. loss: 0.369497
Total training time: 0.74 seconds.
-- Epoch 5
-- Epoch 3
-- Epoch 3
Norm: 4.84, NNZs: 48252, Bias: -0.641347, T: 79200, Avg. loss: 0.191514
Total training time: 0.66 seconds.
Norm: 6.40, NNZs: 55384, Bias: -0.597590, T: 79200, Avg. loss: 0.369102
Total training time: 0.66 seconds.
Norm: 4.89, NNZs: 51464, Bias: -0.696438, T: 99000, Avg. loss: 0.244296
Total training time: 0.74 seconds.
Norm: 4.84, NNZs: 49528, Bias: -0.543121, T: 39600, Avg. loss: 0.237782
Total training time: 0.33 seconds.
Norm: 4.92, NNZs: 50827, Bias: -0.691172, T: 99000, Avg. loss: 0.219246
Total training time: 0.75 seconds.
-- Epoch 4
-- Epoch 3
Norm: 4.76, NNZs: 50302, Bias: -0.652277, T: 79200, Avg. loss: 0.223889
Total training time: 0.66 seconds.
-- Epoch 3
Norm: 4.77, NNZs: 48730, Bias: -0.598600, T: 59400, Avg. loss: 0.208194
Total training time: 0.48 seconds.
-- Epoch 3
Norm: 4.93, NNZs: 49117, Bias: -0.545899, T: 39600, Avg. loss: 0.221967
Total training time: 0.35 seconds.
Norm: 4.90, NNZs: 48437, Bias: -0.544724, T: 39600, Avg. loss: 0.215279
Total training time: 0.36 seconds.
Norm: 5.96, NNZs: 55467, Bias: -0.526855, T: 39600, Avg. loss: 0.405591
Total training time: 0.35 seconds.
-- Epoch 5
Norm: 4.81, NNZs: 47928, Bias: -0.581434, T: 59400, Avg. loss: 0.195813
Total training time: 0.50 seconds.
Norm: 4.75, NNZs: 50786, Bias: -0.593978, T: 59400, Avg. loss: 0.255748
Total training time: 0.54 seconds.
-- Epoch 3
-- Epoch 5
-- Epoch 3
Norm: 4.73, NNZs: 50986, Bias: -0.593430, T: 59400, Avg. loss: 0.255783
Total training time: 0.41 seconds.
-- Epoch 5
-- Epoch 3
-- Epoch 6
Norm: 6.83, NNZs: 55732, Bias: -0.574117, T: 59400, Avg. loss: 0.389155
Total training time: 0.47 seconds.
-- Epoch 5
-- Epoch 5
Norm: 5.38, NNZs: 54768, Bias: -0.496480, T: 39600, Avg. loss: 0.388358
Total training time: 0.38 seconds.
Norm: 4.74, NNZs: 48279, Bias: -0.669276, T: 99000, Avg. loss: 0.187375
Total training time: 0.85 seconds.
-- Epoch 5
Norm: 4.93, NNZs: 48865, Bias: -0.711324, T: 118800, Avg. loss: 0.185185
Total training time: 0.87 seconds.
-- Epoch 3
Norm: 4.72, NNZs: 51134, Bias: -0.653445, T: 79200, Avg. loss: 0.249578
Total training time: 0.75 seconds.
-- Epoch 3
Norm: 4.80, NNZs: 48217, Bias: -0.580225, T: 59400, Avg. loss: 0.195809
Total training time: 0.45 seconds.
-- Epoch 4
Norm: 5.85, NNZs: 54729, Bias: -0.549058, T: 59400, Avg. loss: 0.376746
Total training time: 0.55 seconds.
Norm: 4.78, NNZs: 50339, Bias: -0.696201, T: 99000, Avg. loss: 0.204585
Total training time: 0.80 seconds.
-- Epoch 5
Norm: 4.77, NNZs: 49671, Bias: -0.598979, T: 59400, Avg. loss: 0.229498
Total training time: 0.51 seconds.
-- Epoch 5
Norm: 4.77, NNZs: 48428, Bias: -0.669395, T: 99000, Avg. loss: 0.187204
Total training time: 0.77 seconds.
Norm: 6.97, NNZs: 55690, Bias: -0.623242, T: 99000, Avg. loss: 0.362942
Total training time: 0.79 seconds.
Norm: 4.77, NNZs: 49903, Bias: -0.692185, T: 99000, Avg. loss: 0.204699
Total training time: 0.89 seconds.
-- Epoch 3
Norm: 4.79, NNZs: 49853, Bias: -0.693334, T: 99000, Avg. loss: 0.198619
Total training time: 0.81 seconds.
-- Epoch 6
Norm: 7.00, NNZs: 55509, Bias: -0.624532, T: 99000, Avg. loss: 0.362556
Total training time: 0.92 seconds.
-- Epoch 4
-- Epoch 6
-- Epoch 4
Norm: 7.82, NNZs: 56562, Bias: -0.615631, T: 79200, Avg. loss: 0.377907
Total training time: 0.81 seconds.
-- Epoch 3
Norm: 4.87, NNZs: 49940, Bias: -0.724755, T: 118800, Avg. loss: 0.195704
Total training time: 0.94 seconds.
-- Epoch 6
-- Epoch 3
Norm: 4.87, NNZs: 50765, Bias: -0.697861, T: 99000, Avg. loss: 0.219464
Total training time: 0.80 seconds.
Norm: 4.69, NNZs: 49227, Bias: -0.594175, T: 59400, Avg. loss: 0.214812
Total training time: 0.56 seconds.
Norm: 4.71, NNZs: 49389, Bias: -0.596707, T: 59400, Avg. loss: 0.214979
Total training time: 0.49 seconds.
Norm: 4.85, NNZs: 48152, Bias: -0.590888, T: 59400, Avg. loss: 0.196320
Total training time: 0.52 seconds.
Norm: 4.86, NNZs: 48212, Bias: -0.590933, T: 59400, Avg. loss: 0.196268
Total training time: 0.61 seconds.
Norm: 4.85, NNZs: 48609, Bias: -0.678392, T: 99000, Avg. loss: 0.187937
Total training time: 0.86 seconds.
-- Epoch 3
Norm: 4.75, NNZs: 51388, Bias: -0.647867, T: 79200, Avg. loss: 0.249480
Total training time: 0.55 seconds.
Convergence after 6 epochs took 0.99 seconds
Norm: 5.08, NNZs: 51306, Bias: -0.726325, T: 118800, Avg. loss: 0.215988
Total training time: 0.96 seconds.
-- Epoch 6
-- Epoch 5
Norm: 6.86, NNZs: 56059, Bias: -0.579168, T: 59400, Avg. loss: 0.388933
Total training time: 0.52 seconds.
Norm: 9.77, NNZs: 56955, Bias: -0.652412, T: 118800, Avg. loss: 0.361375
Total training time: 0.99 seconds.
-- Epoch 4
Norm: 4.74, NNZs: 49931, Bias: -0.603112, T: 59400, Avg. loss: 0.229817
Total training time: 0.55 seconds.
-- Epoch 6
-- Epoch 4
-- Epoch 3
-- Epoch 4
-- Epoch 4
-- Epoch 4
-- Epoch 6
Norm: 4.77, NNZs: 48837, Bias: -0.601157, T: 59400, Avg. loss: 0.208022
Total training time: 0.57 seconds.
-- Epoch 5
-- Epoch 6
Norm: 4.73, NNZs: 49094, Bias: -0.646816, T: 79200, Avg. loss: 0.202874
Total training time: 0.69 seconds.
-- Epoch 6
Norm: 5.08, NNZs: 52045, Bias: -0.737754, T: 118800, Avg. loss: 0.240588
Total training time: 1.00 seconds.
-- Epoch 6
Norm: 4.85, NNZs: 51645, Bias: -0.702487, T: 99000, Avg. loss: 0.244621
Total training time: 0.92 seconds.
-- Epoch 4
-- Epoch 6
-- Epoch 6
-- Epoch 4
Norm: 4.92, NNZs: 50793, Bias: -0.735811, T: 118800, Avg. loss: 0.201065
Total training time: 0.97 seconds.
-- Epoch 5
Norm: 4.80, NNZs: 50097, Bias: -0.649294, T: 79200, Avg. loss: 0.223531
Total training time: 0.68 seconds.
Norm: 4.73, NNZs: 48187, Bias: -0.626414, T: 79200, Avg. loss: 0.191102
Total training time: 0.74 seconds.
Norm: 4.84, NNZs: 48835, Bias: -0.704597, T: 118800, Avg. loss: 0.184061
Total training time: 0.95 seconds.
Norm: 4.82, NNZs: 48691, Bias: -0.706670, T: 118800, Avg. loss: 0.184203
Total training time: 1.07 seconds.
Norm: 4.73, NNZs: 48472, Bias: -0.627663, T: 79200, Avg. loss: 0.191123
Total training time: 0.64 seconds.
Norm: 4.76, NNZs: 51188, Bias: -0.646284, T: 79200, Avg. loss: 0.249446
Total training time: 0.78 seconds.
Convergence after 6 epochs took 1.09 seconds
Norm: 4.72, NNZs: 49795, Bias: -0.652869, T: 79200, Avg. loss: 0.209217
Total training time: 0.62 seconds.
Norm: 7.62, NNZs: 56054, Bias: -0.650619, T: 118800, Avg. loss: 0.358005
Total training time: 0.97 seconds.
-- Epoch 4
Norm: 7.82, NNZs: 56245, Bias: -0.611635, T: 79200, Avg. loss: 0.377755
Total training time: 0.72 seconds.
-- Epoch 4
Norm: 8.81, NNZs: 56844, Bias: -0.635639, T: 99000, Avg. loss: 0.368679
Total training time: 0.98 seconds.
-- Epoch 4
Norm: 5.83, NNZs: 55090, Bias: -0.550124, T: 59400, Avg. loss: 0.377026
Total training time: 0.63 seconds.
Norm: 4.88, NNZs: 50175, Bias: -0.727433, T: 118800, Avg. loss: 0.195540
Total training time: 0.99 seconds.
-- Epoch 6
Norm: 4.90, NNZs: 50330, Bias: -0.731735, T: 118800, Avg. loss: 0.201255
Total training time: 1.09 seconds.
Norm: 6.42, NNZs: 55027, Bias: -0.592785, T: 79200, Avg. loss: 0.368715
Total training time: 0.77 seconds.
[Parallel(n_jobs=-1)]: Done   2 out of   8 | elapsed:    1.2s remaining:    3.7s
Convergence after 6 epochs took 1.10 seconds
Convergence after 6 epochs took 1.09 seconds
Norm: 5.00, NNZs: 51254, Bias: -0.731639, T: 118800, Avg. loss: 0.216174
Total training time: 0.98 seconds.
-- Epoch 5
Norm: 4.71, NNZs: 49659, Bias: -0.650900, T: 79200, Avg. loss: 0.209116
Total training time: 0.74 seconds.
-- Epoch 4
Norm: 7.65, NNZs: 55787, Bias: -0.650078, T: 118800, Avg. loss: 0.357552
Total training time: 1.14 seconds.
Norm: 4.86, NNZs: 48551, Bias: -0.640275, T: 79200, Avg. loss: 0.191514
Total training time: 0.79 seconds.
Norm: 4.87, NNZs: 51851, Bias: -0.695079, T: 99000, Avg. loss: 0.244554
Total training time: 0.72 seconds.
Convergence after 6 epochs took 1.12 seconds
-- Epoch 4
Norm: 4.85, NNZs: 48537, Bias: -0.640009, T: 79200, Avg. loss: 0.191549
Total training time: 0.72 seconds.
Norm: 4.76, NNZs: 49435, Bias: -0.687959, T: 99000, Avg. loss: 0.198872
Total training time: 0.83 seconds.
-- Epoch 5
Convergence after 6 epochs took 1.06 seconds
Norm: 4.92, NNZs: 48949, Bias: -0.711524, T: 118800, Avg. loss: 0.185184
Total training time: 1.07 seconds.
-- Epoch 4
-- Epoch 6
Convergence after 6 epochs took 1.17 seconds
Convergence after 6 epochs took 1.19 seconds
-- Epoch 5
Norm: 7.85, NNZs: 56506, Bias: -0.616545, T: 79200, Avg. loss: 0.377335
Total training time: 0.72 seconds.
-- Epoch 5
Convergence after 6 epochs took 1.11 seconds
Norm: 4.76, NNZs: 50417, Bias: -0.654756, T: 79200, Avg. loss: 0.223812
Total training time: 0.75 seconds.
Convergence after 6 epochs took 1.10 seconds
[Parallel(n_jobs=-1)]: Done   2 out of   8 | elapsed:    1.3s remaining:    3.8s
Norm: 5.01, NNZs: 52095, Bias: -0.741122, T: 118800, Avg. loss: 0.240906
Total training time: 1.08 seconds.
-- Epoch 6
Norm: 4.75, NNZs: 49222, Bias: -0.650666, T: 79200, Avg. loss: 0.202579
Total training time: 0.76 seconds.
Convergence after 6 epochs took 1.09 seconds
-- Epoch 4
Convergence after 6 epochs took 1.08 seconds
-- Epoch 6
[Parallel(n_jobs=-1)]: Done   8 out of   8 | elapsed:    1.4s finished
Convergence after 6 epochs took 1.22 seconds
Norm: 4.77, NNZs: 48807, Bias: -0.671716, T: 99000, Avg. loss: 0.187185
Total training time: 0.79 seconds.
-- Epoch 5
Norm: 4.76, NNZs: 48508, Bias: -0.669800, T: 99000, Avg. loss: 0.187169
-- Epoch 5
-- Epoch 5
Convergence after 6 epochs took 1.13 seconds
Norm: 4.88, NNZs: 51627, Bias: -0.693847, T: 99000, Avg. loss: 0.244538
Total training time: 0.94 seconds.
-- Epoch 5
Total training time: 0.91 seconds.
Convergence after 6 epochs took 1.12 seconds
-- Epoch 5
Norm: 8.82, NNZs: 56546, Bias: -0.634250, T: 99000, Avg. loss: 0.368488
Total training time: 0.87 seconds.
Norm: 7.02, NNZs: 55333, Bias: -0.621761, T: 99000, Avg. loss: 0.362470
Total training time: 0.92 seconds.
Norm: 6.38, NNZs: 55390, Bias: -0.592949, T: 79200, Avg. loss: 0.368988
Norm: 9.85, NNZs: 57062, Bias: -0.653959, T: 118800, Avg. loss: 0.360448
Total training time: 1.15 seconds.
-- Epoch 6
Norm: 4.87, NNZs: 48845, Bias: -0.676619, T: 99000, Avg. loss: 0.187888
Total training time: 0.91 seconds.
Total training time: 0.79 seconds.
Norm: 4.92, NNZs: 50597, Bias: -0.694548, T: 99000, Avg. loss: 0.219171
Total training time: 0.88 seconds.
Norm: 5.04, NNZs: 52270, Bias: -0.733103, T: 118800, Avg. loss: 0.240885
Total training time: 0.85 seconds.
-- Epoch 6
Norm: 4.85, NNZs: 49804, Bias: -0.722400, T: 118800, Avg. loss: 0.195822
Total training time: 0.95 seconds.
-- Epoch 5
-- Epoch 5
Norm: 4.77, NNZs: 50064, Bias: -0.695020, T: 99000, Avg. loss: 0.204569
Total training time: 0.89 seconds.
-- Epoch 5
Norm: 4.84, NNZs: 49118, Bias: -0.707138, T: 118800, Avg. loss: 0.184029
Total training time: 0.86 seconds.
-- Epoch 6
-- Epoch 6
-- Epoch 5
Norm: 8.85, NNZs: 56826, Bias: -0.636899, T: 99000, Avg. loss: 0.367929
Total training time: 0.84 seconds.
-- Epoch 5
-- Epoch 6
Norm: 5.06, NNZs: 52064, Bias: -0.732826, T: 118800, Avg. loss: 0.240848
Total training time: 1.03 seconds.
Norm: 4.86, NNZs: 48871, Bias: -0.676449, T: 99000, Avg. loss: 0.187970
Total training time: 0.89 seconds.
Norm: 7.67, NNZs: 55621, Bias: -0.646128, T: 118800, Avg. loss: 0.357389
Total training time: 0.99 seconds.
-- Epoch 6
Convergence after 6 epochs took 1.22 seconds
[Parallel(n_jobs=-1)]: Done   8 out of   8 | elapsed:    1.4s finished
-- Epoch 6
Convergence after 6 epochs took 0.90 seconds
Norm: 9.86, NNZs: 56825, Bias: -0.652770, T: 118800, Avg. loss: 0.360243
Total training time: 0.96 seconds.
Convergence after 6 epochs took 1.01 seconds
-- Epoch 6
Convergence after 6 epochs took 0.92 seconds
[Parallel(n_jobs=-1)]: Done   2 out of   8 | elapsed:    1.1s remaining:    3.4s
Norm: 4.76, NNZs: 50158, Bias: -0.695500, T: 99000, Avg. loss: 0.204627
-- Epoch 5
-- Epoch 6
Norm: 4.87, NNZs: 50848, Bias: -0.698884, T: 99000, Avg. loss: 0.219395
Total training time: 0.90 seconds.
-- Epoch 6
Total training time: 0.90 seconds.
Norm: 4.95, NNZs: 49215, Bias: -0.709390, T: 118800, Avg. loss: 0.185088
Total training time: 1.01 seconds.
Convergence after 6 epochs took 1.06 seconds
Norm: 5.08, NNZs: 50980, Bias: -0.730059, T: 118800, Avg. loss: 0.215924
Total training time: 0.97 seconds.
[Parallel(n_jobs=-1)]: Done   2 out of   8 | elapsed:    1.3s remaining:    4.0s
Norm: 4.80, NNZs: 49661, Bias: -0.693038, T: 99000, Avg. loss: 0.198621
Total training time: 0.91 seconds.
Convergence after 6 epochs took 0.99 seconds
[Parallel(n_jobs=-1)]: Done   2 out of   5 | elapsed:    1.6s remaining:    2.4s
Convergence after 6 epochs took 1.04 seconds
Norm: 6.97, NNZs: 55716, Bias: -0.622406, T: 99000, Avg. loss: 0.362834
Total training time: 0.91 seconds.
Norm: 4.91, NNZs: 50532, Bias: -0.734650, T: 118800, Avg. loss: 0.201087
Total training time: 0.99 seconds.
Norm: 4.84, NNZs: 48878, Bias: -0.705675, T: 118800, Avg. loss: 0.183984
Total training time: 1.05 seconds.
-- Epoch 6
Convergence after 6 epochs took 1.04 seconds
Norm: 9.90, NNZs: 57057, Bias: -0.655535, T: 118800, Avg. loss: 0.359578
Total training time: 0.91 seconds.
-- Epoch 6
Convergence after 6 epochs took 1.02 seconds
Norm: 5.02, NNZs: 51290, Bias: -0.736330, T: 118800, Avg. loss: 0.216123
Total training time: 0.95 seconds.
-- Epoch 6
-- Epoch 6
Norm: 4.87, NNZs: 50042, Bias: -0.724786, T: 118800, Avg. loss: 0.195651
Total training time: 0.95 seconds.
-- Epoch 6
Convergence after 6 epochs took 1.09 seconds
Convergence after 6 epochs took 1.02 seconds
[Parallel(n_jobs=-1)]: Done   8 out of   8 | elapsed:    1.4s finished
Norm: 7.63, NNZs: 55983, Bias: -0.650320, T: 118800, Avg. loss: 0.357885
Total training time: 0.96 seconds.
Norm: 4.90, NNZs: 50633, Bias: -0.735550, T: 118800, Avg. loss: 0.201150
Total training time: 0.96 seconds.
Norm: 4.93, NNZs: 49253, Bias: -0.709421, T: 118800, Avg. loss: 0.185213
Total training time: 0.99 seconds.
Convergence after 6 epochs took 0.95 seconds
Convergence after 6 epochs took 0.98 seconds
Convergence after 6 epochs took 0.98 seconds
Convergence after 6 epochs took 0.98 seconds
Convergence after 6 epochs took 0.98 seconds
Convergence after 6 epochs took 1.00 seconds
[Parallel(n_jobs=-1)]: Done   8 out of   8 | elapsed:    1.2s finished
	accuracy: 5-fold cross validation: [0.4172 0.4284 0.4096 0.409  0.4164]
	test accuracy: 5-fold cross validation accuracy: 0.42 (+/- 0.01)
dimensionality: 74170
density: 0.749636



===> Classification Report:

[Parallel(n_jobs=-1)]: Done   5 out of   5 | elapsed:    1.8s finished
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
           validation_fraction=0.001, verbose=True, warm_start=False)
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 8 concurrent workers.
-- Epoch 1
-- Epoch 1
-- Epoch 1
-- Epoch 1
-- Epoch 1
-- Epoch 1
-- Epoch 1
Norm: 22.96, NNZs: 34405, Bias: -0.300000, T: 24975, Avg. loss: 0.028100
Total training time: 0.03 seconds.
Norm: 22.52, NNZs: 36092, Bias: -0.290000, T: 24975, Avg. loss: 0.027337
Total training time: 0.04 seconds.
Norm: 24.34, NNZs: 37880, Bias: -0.230000, T: 24975, Avg. loss: 0.033129
Norm: 25.21, NNZs: 36226, Bias: -0.150000, T: 24975, Avg. loss: 0.033243
Total training time: 0.05 seconds.
Total training time: 0.04 seconds.
-- Epoch 1
Norm: 23.24, NNZs: 36326, Bias: -0.180000, T: 24975, Avg. loss: 0.028693
Total training time: 0.05 seconds.
Norm: 21.91, NNZs: 33980, Bias: -0.230000, T: 24975, Avg. loss: 0.025914
Total training time: 0.05 seconds.
Norm: 26.12, NNZs: 37758, Bias: -0.100000, T: 24975, Avg. loss: 0.037173
Total training time: 0.06 seconds.
-- Epoch 2
-- Epoch 2
Norm: 22.17, NNZs: 32117, Bias: -0.240000, T: 24975, Avg. loss: 0.026834
Total training time: 0.05 seconds.
-- Epoch 2
-- Epoch 2
-- Epoch 2
Norm: 22.54, NNZs: 42049, Bias: -0.260000, T: 49950, Avg. loss: 0.023353
Total training time: 0.12 seconds.
Norm: 22.49, NNZs: 40446, Bias: -0.250000, T: 49950, Avg. loss: 0.023738
Total training time: 0.12 seconds.
-- Epoch 2
Norm: 24.34, NNZs: 44616, Bias: -0.220000, T: 49950, Avg. loss: 0.028659
Total training time: 0.11 seconds.
-- Epoch 2
Norm: 24.59, NNZs: 43080, Bias: -0.160000, T: 49950, Avg. loss: 0.028629
Total training time: 0.13 seconds.
Norm: 22.02, NNZs: 39383, Bias: -0.230000, T: 49950, Avg. loss: 0.022294
Total training time: 0.11 seconds.
Norm: 25.75, NNZs: 44741, Bias: -0.140000, T: 49950, Avg. loss: 0.031813
Total training time: 0.12 seconds.
Norm: 23.78, NNZs: 42917, Bias: -0.220000, T: 49950, Avg. loss: 0.025439
Total training time: 0.12 seconds.
-- Epoch 2
-- Epoch 3
-- Epoch 3
Norm: 22.28, NNZs: 37383, Bias: -0.210000, T: 49950, Avg. loss: 0.022975
Total training time: 0.12 seconds.
-- Epoch 3
-- Epoch 3
Norm: 22.97, NNZs: 45656, Bias: -0.230000, T: 74925, Avg. loss: 0.023146
Total training time: 0.17 seconds.
-- Epoch 3
-- Epoch 3
Norm: 23.10, NNZs: 43994, Bias: -0.170000, T: 74925, Avg. loss: 0.023312
Total training time: 0.18 seconds.
Norm: 24.51, NNZs: 48315, Bias: -0.220000, T: 74925, Avg. loss: 0.029130
Total training time: 0.17 seconds.
-- Epoch 3
Norm: 25.30, NNZs: 46554, Bias: -0.140000, T: 74925, Avg. loss: 0.028568
Total training time: 0.19 seconds.
Norm: 21.73, NNZs: 42771, Bias: -0.210000, T: 74925, Avg. loss: 0.022392
Total training time: 0.17 seconds.
Norm: 25.62, NNZs: 48310, Bias: -0.130000, T: 74925, Avg. loss: 0.032973
Total training time: 0.18 seconds.
Norm: 23.60, NNZs: 46533, Bias: -0.190000, T: 74925, Avg. loss: 0.024344
Total training time: 0.18 seconds.
-- Epoch 3
-- Epoch 4
-- Epoch 4
-- Epoch 4
Norm: 22.18, NNZs: 48109, Bias: -0.290000, T: 99900, Avg. loss: 0.023366
Total training time: 0.23 seconds.
-- Epoch 4
Norm: 22.49, NNZs: 41050, Bias: -0.250000, T: 74925, Avg. loss: 0.023497
Total training time: 0.18 seconds.
-- Epoch 4
-- Epoch 4
Norm: 22.85, NNZs: 46663, Bias: -0.190000, T: 99900, Avg. loss: 0.024000
Total training time: 0.24 seconds.
Norm: 24.62, NNZs: 50750, Bias: -0.230000, T: 99900, Avg. loss: 0.029398
Total training time: 0.23 seconds.
-- Epoch 4
Norm: 21.88, NNZs: 45320, Bias: -0.250000, T: 99900, Avg. loss: 0.022178
Total training time: 0.22 seconds.
Norm: 24.62, NNZs: 49045, Bias: -0.180000, T: 99900, Avg. loss: 0.028574
Total training time: 0.25 seconds.
Norm: 26.37, NNZs: 50834, Bias: -0.110000, T: 99900, Avg. loss: 0.031826
Total training time: 0.23 seconds.
Norm: 23.05, NNZs: 48990, Bias: -0.190000, T: 99900, Avg. loss: 0.025304
Total training time: 0.24 seconds.
Convergence after 4 epochs took 0.27 seconds
-- Epoch 4
Convergence after 4 epochs took 0.27 seconds
[Parallel(n_jobs=-1)]: Done   2 out of   8 | elapsed:    0.3s remaining:    1.0s
-- Epoch 5
Convergence after 4 epochs took 0.26 seconds
Convergence after 4 epochs took 0.28 seconds
Norm: 22.15, NNZs: 43624, Bias: -0.280000, T: 99900, Avg. loss: 0.022702
Total training time: 0.24 seconds.
Convergence after 4 epochs took 0.29 seconds
Convergence after 4 epochs took 0.27 seconds
Norm: 22.08, NNZs: 47454, Bias: -0.200000, T: 124875, Avg. loss: 0.022461
Total training time: 0.27 seconds.
-- Epoch 5
Convergence after 5 epochs took 0.29 seconds
Norm: 22.01, NNZs: 45736, Bias: -0.280000, T: 124875, Avg. loss: 0.023352
Total training time: 0.26 seconds.
-- Epoch 6
Norm: 21.71, NNZs: 47488, Bias: -0.190000, T: 149850, Avg. loss: 0.023174
Total training time: 0.28 seconds.
Convergence after 6 epochs took 0.30 seconds
[Parallel(n_jobs=-1)]: Done   8 out of   8 | elapsed:    0.4s finished
train time: 0.522s
test time:  0.062s
accuracy:   0.310


cross validation:
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 8 concurrent workers.
[LibLinear]-- Epoch 1
-- Epoch 1
-- Epoch 1
-- Epoch 1
-- Epoch 1
-- Epoch 1
-- Epoch 1
Norm: 23.51, NNZs: 33488, Bias: -0.140000, T: 19980, Avg. loss: 0.029201
Total training time: 0.01 seconds.
Norm: 26.24, NNZs: 34558, Bias: -0.120000, T: 19980, Avg. loss: 0.036272
Total training time: 0.01 seconds.
Norm: 21.67, NNZs: 31126, Bias: -0.260000, T: 19980, Avg. loss: 0.027330
Total training time: 0.01 seconds.
Norm: 24.41, NNZs: 34509, Bias: -0.230000, T: 19980, Avg. loss: 0.033172
Total training time: 0.01 seconds.
Norm: 21.98, NNZs: 30641, Bias: -0.210000, T: 19980, Avg. loss: 0.025281
Total training time: 0.01 seconds.
Norm: 22.73, NNZs: 32692, Bias: -0.270000, T: 19980, Avg. loss: 0.027278
Total training time: 0.01 seconds.
-- Epoch 1
Norm: 22.13, NNZs: 28808, Bias: -0.230000, T: 19980, Avg. loss: 0.026771
Total training time: 0.02 seconds.
-- Epoch 2
Norm: 25.23, NNZs: 32693, Bias: -0.120000, T: 19980, Avg. loss: 0.033756
Total training time: 0.01 seconds.
-- Epoch 2
-- Epoch 2
-- Epoch 2
-- Epoch 2
Norm: 26.13, NNZs: 40652, Bias: -0.120000, T: 39960, Avg. loss: 0.028923
Total training time: 0.04 seconds.
-- Epoch 2
-- Epoch 2
Norm: 22.17, NNZs: 36437, Bias: -0.200000, T: 39960, Avg. loss: 0.022012
Norm: 22.08, NNZs: 35794, Bias: -0.270000, T: 39960, Avg. loss: 0.019287
Total training time: 0.04 seconds.
Total training time: 0.04 seconds.
Norm: 24.32, NNZs: 40346, Bias: -0.200000, T: 39960, Avg. loss: 0.026266
Total training time: 0.04 seconds.
-- Epoch 3
-- Epoch 2
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 8 concurrent workers.
Norm: 23.22, NNZs: 38913, Bias: -0.190000, T: 39960, Avg. loss: 0.022912
Total training time: 0.05 seconds.
Norm: 25.03, NNZs: 38489, Bias: -0.140000, T: 39960, Avg. loss: 0.026128
Total training time: 0.04 seconds.
Norm: 25.33, NNZs: 43781, Bias: -0.130000, T: 59940, Avg. loss: 0.028222
Norm: 22.59, NNZs: 33716, Bias: -0.240000, T: 39960, Avg. loss: 0.021428
Total training time: 0.06 seconds.
Total training time: 0.06 seconds.
-- Epoch 3
Norm: 22.67, NNZs: 38320, Bias: -0.290000, T: 39960, Avg. loss: 0.021546
Total training time: 0.06 seconds.
-- Epoch 3
-- Epoch 3
-- Epoch 3
-- Epoch 3
-- Epoch 4
-- Epoch 3
Norm: 21.71, NNZs: 38853, Bias: -0.220000, T: 59940, Avg. loss: 0.019443
Total training time: 0.07 seconds.
Norm: 22.68, NNZs: 39510, Bias: -0.260000, T: 59940, Avg. loss: 0.021707
Total training time: 0.07 seconds.
Norm: 23.34, NNZs: 42279, Bias: -0.230000, T: 59940, Avg. loss: 0.022726
Total training time: 0.08 seconds.
Norm: 23.92, NNZs: 43704, Bias: -0.220000, T: 59940, Avg. loss: 0.026194
Total training time: 0.08 seconds.
Norm: 21.93, NNZs: 36882, Bias: -0.220000, T: 59940, Avg. loss: 0.020960
Total training time: 0.09 seconds.
-- Epoch 4
-- Epoch 4
-- Epoch 4
-- Epoch 3
Norm: 25.71, NNZs: 46134, Bias: -0.160000, T: 79920, Avg. loss: 0.028917
Total training time: 0.10 seconds.
Norm: 22.54, NNZs: 41799, Bias: -0.240000, T: 79920, Avg. loss: 0.021316
Total training time: 0.11 seconds.
Norm: 23.69, NNZs: 44609, Bias: -0.180000, T: 79920, Avg. loss: 0.022330
Total training time: 0.12 seconds.
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 8 concurrent workers.
Convergence after 4 epochs took 0.13 seconds
-- Epoch 4
[LibLinear]-- Epoch 1
Norm: 22.35, NNZs: 41302, Bias: -0.220000, T: 59940, Avg. loss: 0.020780
Total training time: 0.13 seconds.
-- Epoch 4
-- Epoch 5
Norm: 22.01, NNZs: 41045, Bias: -0.260000, T: 79920, Avg. loss: 0.019645
Total training time: 0.14 seconds.
-- Epoch 1
-- Epoch 5
-- Epoch 1
Norm: 24.79, NNZs: 45999, Bias: -0.200000, T: 79920, Avg. loss: 0.026274
Total training time: 0.15 seconds.
-- Epoch 4
Norm: 23.33, NNZs: 46334, Bias: -0.170000, T: 99900, Avg. loss: 0.022384
Total training time: 0.16 seconds.
-- Epoch 1
Norm: 24.72, NNZs: 41674, Bias: -0.080000, T: 59940, Avg. loss: 0.026244
Total training time: 0.15 seconds.
Convergence after 4 epochs took 0.17 seconds
Norm: 22.34, NNZs: 43581, Bias: -0.240000, T: 79920, Avg. loss: 0.020363
Total training time: 0.17 seconds.
[Parallel(n_jobs=-1)]: Done   2 out of   8 | elapsed:    0.2s remaining:    0.6s
-- Epoch 1
-- Epoch 6
-- Epoch 1
Norm: 26.03, NNZs: 47777, Bias: -0.140000, T: 99900, Avg. loss: 0.029007
Total training time: 0.18 seconds.
-- Epoch 1
-- Epoch 1
Norm: 22.06, NNZs: 31550, Bias: -0.250000, T: 19980, Avg. loss: 0.027502
Total training time: 0.06 seconds.
Convergence after 4 epochs took 0.19 seconds
Convergence after 4 epochs took 0.19 seconds
-- Epoch 4
Norm: 25.33, NNZs: 32416, Bias: -0.150000, T: 19980, Avg. loss: 0.033242
Total training time: 0.05 seconds.
Norm: 23.25, NNZs: 47698, Bias: -0.230000, T: 119880, Avg. loss: 0.022492
Total training time: 0.20 seconds.
Norm: 22.09, NNZs: 28993, Bias: -0.210000, T: 19980, Avg. loss: 0.027009
Total training time: 0.03 seconds.
Norm: 21.98, NNZs: 30514, Bias: -0.260000, T: 19980, Avg. loss: 0.025840
Total training time: 0.05 seconds.
Norm: 25.75, NNZs: 34631, Bias: -0.110000, T: 19980, Avg. loss: 0.037593
Total training time: 0.02 seconds.
Norm: 22.29, NNZs: 39379, Bias: -0.190000, T: 79920, Avg. loss: 0.020729
Total training time: 0.20 seconds.
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 8 concurrent workers.
Norm: 24.75, NNZs: 34978, Bias: -0.230000, T: 19980, Avg. loss: 0.032935
Total training time: 0.04 seconds.
Norm: 23.08, NNZs: 32855, Bias: -0.160000, T: 19980, Avg. loss: 0.028713
Total training time: 0.03 seconds.
Convergence after 5 epochs took 0.22 seconds
Convergence after 6 epochs took 0.22 seconds
Norm: 24.92, NNZs: 43771, Bias: -0.100000, T: 79920, Avg. loss: 0.025970
Total training time: 0.21 seconds.
-- Epoch 2
Norm: 22.61, NNZs: 32078, Bias: -0.260000, T: 19980, Avg. loss: 0.027462
Total training time: 0.04 seconds.
-- Epoch 2
Convergence after 4 epochs took 0.23 seconds
Norm: 22.56, NNZs: 36815, Bias: -0.190000, T: 39960, Avg. loss: 0.021764
Total training time: 0.11 seconds.
Convergence after 4 epochs took 0.23 seconds
-- Epoch 2
-- Epoch 2
Norm: 22.40, NNZs: 33874, Bias: -0.230000, T: 39960, Avg. loss: 0.021418
Total training time: 0.08 seconds.
-- Epoch 2
-- Epoch 2
[Parallel(n_jobs=-1)]: Done   8 out of   8 | elapsed:    0.3s finished
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 8 concurrent workers.
-- Epoch 2
Norm: 22.03, NNZs: 35499, Bias: -0.210000, T: 39960, Avg. loss: 0.020101
Total training time: 0.13 seconds.
-- Epoch 3
Norm: 23.31, NNZs: 38672, Bias: -0.260000, T: 39960, Avg. loss: 0.022771
Total training time: 0.11 seconds.
Norm: 24.41, NNZs: 40531, Bias: -0.230000, T: 39960, Avg. loss: 0.025478
Total training time: 0.14 seconds.
[LibLinear]-- Epoch 1
-- Epoch 1
-- Epoch 3
Norm: 24.98, NNZs: 38753, Bias: -0.160000, T: 39960, Avg. loss: 0.026607
Total training time: 0.16 seconds.
Norm: 22.62, NNZs: 40097, Bias: -0.300000, T: 59940, Avg. loss: 0.021468
Total training time: 0.18 seconds.
Norm: 26.01, NNZs: 40792, Bias: -0.150000, T: 39960, Avg. loss: 0.029750
Total training time: 0.14 seconds.
-- Epoch 3
-- Epoch 2
-- Epoch 1
-- Epoch 3
Norm: 22.17, NNZs: 37254, Bias: -0.230000, T: 59940, Avg. loss: 0.020230
Total training time: 0.18 seconds.
Norm: 22.62, NNZs: 32183, Bias: -0.260000, T: 19980, Avg. loss: 0.026260
Total training time: 0.06 seconds.
-- Epoch 1
-- Epoch 1
-- Epoch 3
-- Epoch 1
Norm: 21.58, NNZs: 38784, Bias: -0.220000, T: 59940, Avg. loss: 0.020229
Total training time: 0.23 seconds.
-- Epoch 1
Norm: 24.82, NNZs: 41884, Bias: -0.140000, T: 59940, Avg. loss: 0.026960
Total training time: 0.24 seconds.
-- Epoch 1
Norm: 24.44, NNZs: 34623, Bias: -0.250000, T: 19980, Avg. loss: 0.033467
Total training time: 0.09 seconds.
-- Epoch 3
Norm: 22.49, NNZs: 37618, Bias: -0.300000, T: 39960, Avg. loss: 0.020732
Total training time: 0.21 seconds.
-- Epoch 3
-- Epoch 1
-- Epoch 1
Norm: 24.56, NNZs: 32226, Bias: -0.130000, T: 19980, Avg. loss: 0.033166
Total training time: 0.04 seconds.
Norm: 22.28, NNZs: 30643, Bias: -0.230000, T: 19980, Avg. loss: 0.025686
Total training time: 0.04 seconds.
Norm: 23.39, NNZs: 33327, Bias: -0.180000, T: 19980, Avg. loss: 0.029494
Total training time: 0.08 seconds.
Norm: 23.46, NNZs: 42031, Bias: -0.230000, T: 59940, Avg. loss: 0.022457
Total training time: 0.26 seconds.
Norm: 25.45, NNZs: 43904, Bias: -0.130000, T: 59940, Avg. loss: 0.028614
Total training time: 0.25 seconds.
Norm: 23.77, NNZs: 43833, Bias: -0.180000, T: 59940, Avg. loss: 0.026070
Total training time: 0.28 seconds.
-- Epoch 4
-- Epoch 1
Norm: 22.02, NNZs: 28938, Bias: -0.240000, T: 19980, Avg. loss: 0.026636
Total training time: 0.08 seconds.
-- Epoch 4
-- Epoch 1
-- Epoch 1
-- Epoch 1
-- Epoch 2
-- Epoch 4
-- Epoch 1
Norm: 21.94, NNZs: 31290, Bias: -0.280000, T: 19980, Avg. loss: 0.027321
Total training time: 0.09 seconds.
-- Epoch 1
Norm: 22.16, NNZs: 39658, Bias: -0.240000, T: 79920, Avg. loss: 0.021382
Total training time: 0.30 seconds.
Norm: 22.82, NNZs: 32732, Bias: -0.280000, T: 19980, Avg. loss: 0.026654
Total training time: 0.03 seconds.
Norm: 25.95, NNZs: 34906, Bias: -0.140000, T: 19980, Avg. loss: 0.036833
Total training time: 0.09 seconds.
Norm: 21.41, NNZs: 30747, Bias: -0.210000, T: 19980, Avg. loss: 0.025979
Total training time: 0.07 seconds.
Norm: 21.74, NNZs: 29155, Bias: -0.230000, T: 19980, Avg. loss: 0.026750
Total training time: 0.04 seconds.
-- Epoch 2
Norm: 21.76, NNZs: 30732, Bias: -0.270000, T: 19980, Avg. loss: 0.027860
Total training time: 0.04 seconds.
Norm: 24.38, NNZs: 34563, Bias: -0.230000, T: 19980, Avg. loss: 0.032847
Total training time: 0.08 seconds.
-- Epoch 2
Norm: 24.88, NNZs: 32746, Bias: -0.170000, T: 19980, Avg. loss: 0.033515
Total training time: 0.06 seconds.
Norm: 21.75, NNZs: 35725, Bias: -0.180000, T: 39960, Avg. loss: 0.020154
Total training time: 0.12 seconds.
Norm: 22.08, NNZs: 37856, Bias: -0.280000, T: 39960, Avg. loss: 0.020560
Total training time: 0.21 seconds.
Norm: 23.48, NNZs: 32742, Bias: -0.200000, T: 19980, Avg. loss: 0.028609
Total training time: 0.07 seconds.
Norm: 24.56, NNZs: 40470, Bias: -0.250000, T: 39960, Avg. loss: 0.026591
Total training time: 0.22 seconds.
Norm: 25.73, NNZs: 34469, Bias: -0.150000, T: 19980, Avg. loss: 0.036685
Total training time: 0.05 seconds.
Norm: 22.49, NNZs: 42457, Bias: -0.270000, T: 79920, Avg. loss: 0.021411
Total training time: 0.40 seconds.
-- Epoch 4
Norm: 24.88, NNZs: 44156, Bias: -0.150000, T: 79920, Avg. loss: 0.025944
Total training time: 0.39 seconds.
-- Epoch 3
-- Epoch 4
-- Epoch 4
-- Epoch 2
-- Epoch 1
-- Epoch 2
-- Epoch 1
-- Epoch 2
-- Epoch 2
Norm: 22.89, NNZs: 40932, Bias: -0.230000, T: 59940, Avg. loss: 0.020845
Total training time: 0.42 seconds.
-- Epoch 2
-- Epoch 2
Norm: 21.76, NNZs: 41125, Bias: -0.260000, T: 79920, Avg. loss: 0.019403
Total training time: 0.47 seconds.
-- Epoch 2
-- Epoch 1
-- Epoch 1
-- Epoch 1
-- Epoch 2
-- Epoch 3
-- Epoch 1
-- Epoch 2
Norm: 25.93, NNZs: 41036, Bias: -0.110000, T: 39960, Avg. loss: 0.028925
Total training time: 0.25 seconds.
-- Epoch 1
Norm: 25.29, NNZs: 38485, Bias: -0.140000, T: 39960, Avg. loss: 0.026740
Total training time: 0.27 seconds.
-- Epoch 1
Norm: 23.00, NNZs: 38825, Bias: -0.240000, T: 39960, Avg. loss: 0.022038
Total training time: 0.29 seconds.
Norm: 23.38, NNZs: 44310, Bias: -0.210000, T: 79920, Avg. loss: 0.022056
Total training time: 0.48 seconds.
-- Epoch 4
Norm: 22.22, NNZs: 37916, Bias: -0.260000, T: 39960, Avg. loss: 0.020828
Total training time: 0.21 seconds.
-- Epoch 5
-- Epoch 2
Convergence after 4 epochs took 0.50 seconds
-- Epoch 2
Norm: 22.35, NNZs: 33752, Bias: -0.230000, T: 39960, Avg. loss: 0.020467
Total training time: 0.31 seconds.
-- Epoch 2
-- Epoch 3
Norm: 23.27, NNZs: 38416, Bias: -0.170000, T: 39960, Avg. loss: 0.021863
Total training time: 0.24 seconds.
Norm: 22.29, NNZs: 34022, Bias: -0.200000, T: 39960, Avg. loss: 0.020066
Total training time: 0.23 seconds.
Norm: 22.15, NNZs: 31336, Bias: -0.230000, T: 19980, Avg. loss: 0.027596
Total training time: 0.10 seconds.
-- Epoch 2
Norm: 21.63, NNZs: 30289, Bias: -0.230000, T: 19980, Avg. loss: 0.025276
Norm: 24.94, NNZs: 38488, Bias: -0.150000, T: 39960, Avg. loss: 0.026537
Total training time: 0.24 seconds.
Norm: 22.66, NNZs: 36572, Bias: -0.230000, T: 39960, Avg. loss: 0.020638
Total training time: 0.31 seconds.
Norm: 22.01, NNZs: 29035, Bias: -0.250000, T: 19980, Avg. loss: 0.026750
Total training time: 0.10 seconds.
-- Epoch 3
Norm: 23.66, NNZs: 33359, Bias: -0.170000, T: 19980, Avg. loss: 0.028888
Total training time: 0.07 seconds.
Norm: 21.29, NNZs: 38846, Bias: -0.200000, T: 59940, Avg. loss: 0.020059
Total training time: 0.31 seconds.
Norm: 22.99, NNZs: 32166, Bias: -0.260000, T: 19980, Avg. loss: 0.026083
Total training time: 0.07 seconds.
Norm: 23.26, NNZs: 44423, Bias: -0.270000, T: 99900, Avg. loss: 0.021933
Total training time: 0.57 seconds.
Norm: 25.41, NNZs: 45932, Bias: -0.150000, T: 79920, Avg. loss: 0.029175
Total training time: 0.53 seconds.
Norm: 24.31, NNZs: 34721, Bias: -0.260000, T: 19980, Avg. loss: 0.032728
Total training time: 0.08 seconds.
Convergence after 4 epochs took 0.57 seconds
Norm: 25.90, NNZs: 34714, Bias: -0.130000, T: 19980, Avg. loss: 0.035908
Total training time: 0.08 seconds.
Total training time: 0.05 seconds.
Norm: 25.16, NNZs: 32852, Bias: -0.160000, T: 19980, Avg. loss: 0.033840
Total training time: 0.10 seconds.
Norm: 25.60, NNZs: 40393, Bias: -0.150000, T: 39960, Avg. loss: 0.029055
Total training time: 0.26 seconds.
Norm: 24.16, NNZs: 40584, Bias: -0.230000, T: 39960, Avg. loss: 0.026739
Total training time: 0.32 seconds.
-- Epoch 3
[Parallel(n_jobs=-1)]: Done   2 out of   8 | elapsed:    0.7s remaining:    2.1s
Norm: 24.49, NNZs: 46254, Bias: -0.180000, T: 79920, Avg. loss: 0.025907
Total training time: 0.58 seconds.
-- Epoch 4
Norm: 23.50, NNZs: 43764, Bias: -0.260000, T: 59940, Avg. loss: 0.025744
Total training time: 0.45 seconds.
Norm: 22.07, NNZs: 35731, Bias: -0.230000, T: 39960, Avg. loss: 0.020225
-- Epoch 3
Norm: 22.66, NNZs: 41059, Bias: -0.240000, T: 59940, Avg. loss: 0.020616
Total training time: 0.46 seconds.
Total training time: 0.35 seconds.
-- Epoch 3
Norm: 22.84, NNZs: 36151, Bias: -0.220000, T: 39960, Avg. loss: 0.021569
Total training time: 0.31 seconds.
-- Epoch 2
-- Epoch 3
-- Epoch 2
-- Epoch 3
-- Epoch 3
Convergence after 4 epochs took 0.64 seconds
-- Epoch 5
Norm: 22.83, NNZs: 43229, Bias: -0.210000, T: 79920, Avg. loss: 0.020471
Total training time: 0.60 seconds.
Norm: 22.81, NNZs: 41209, Bias: -0.260000, T: 59940, Avg. loss: 0.021127
Total training time: 0.34 seconds.
Convergence after 4 epochs took 0.61 seconds
Norm: 23.67, NNZs: 42252, Bias: -0.230000, T: 59940, Avg. loss: 0.022894
Total training time: 0.45 seconds.
Norm: 21.98, NNZs: 37042, Bias: -0.240000, T: 59940, Avg. loss: 0.020298
Total training time: 0.36 seconds.
Norm: 24.62, NNZs: 41843, Bias: -0.150000, T: 59940, Avg. loss: 0.025979
Total training time: 0.44 seconds.
Norm: 25.78, NNZs: 44201, Bias: -0.170000, T: 59940, Avg. loss: 0.028269
Total training time: 0.42 seconds.
-- Epoch 4
-- Epoch 2
Convergence after 5 epochs took 0.69 seconds
Norm: 22.94, NNZs: 36742, Bias: -0.190000, T: 39960, Avg. loss: 0.021363
Total training time: 0.24 seconds.
-- Epoch 2
-- Epoch 2
-- Epoch 3
Norm: 23.33, NNZs: 46137, Bias: -0.190000, T: 99900, Avg. loss: 0.021865
-- Epoch 5
Norm: 23.67, NNZs: 38964, Bias: -0.190000, T: 39960, Avg. loss: 0.022478
Total training time: 0.21 seconds.
Total training time: 0.66 seconds.
-- Epoch 2
-- Epoch 3
-- Epoch 3
Norm: 22.21, NNZs: 36894, Bias: -0.200000, T: 59940, Avg. loss: 0.020453
Total training time: 0.48 seconds.
-- Epoch 3
Norm: 21.56, NNZs: 41128, Bias: -0.220000, T: 79920, Avg. loss: 0.020013
Total training time: 0.47 seconds.
-- Epoch 2
-- Epoch 4
-- Epoch 3
-- Epoch 2
-- Epoch 3
-- Epoch 3
Norm: 23.55, NNZs: 41802, Bias: -0.230000, T: 59940, Avg. loss: 0.022607
Total training time: 0.43 seconds.
Norm: 24.22, NNZs: 40634, Bias: -0.260000, T: 39960, Avg. loss: 0.026929
Total training time: 0.25 seconds.
Norm: 24.39, NNZs: 48124, Bias: -0.220000, T: 99900, Avg. loss: 0.026224
Total training time: 0.72 seconds.
-- Epoch 4
Norm: 25.01, NNZs: 41803, Bias: -0.140000, T: 59940, Avg. loss: 0.026570
Total training time: 0.44 seconds.
Norm: 22.35, NNZs: 39793, Bias: -0.270000, T: 59940, Avg. loss: 0.021416
Total training time: 0.51 seconds.
Convergence after 4 epochs took 0.71 seconds
Norm: 22.36, NNZs: 34034, Bias: -0.210000, T: 39960, Avg. loss: 0.020975
-- Epoch 4
Norm: 22.32, NNZs: 39574, Bias: -0.220000, T: 59940, Avg. loss: 0.021338
Total training time: 0.44 seconds.
Total training time: 0.30 seconds.
-- Epoch 4
-- Epoch 4
Norm: 24.91, NNZs: 44041, Bias: -0.130000, T: 79920, Avg. loss: 0.026389
Total training time: 0.53 seconds.
Norm: 24.21, NNZs: 46054, Bias: -0.250000, T: 79920, Avg. loss: 0.027080
Total training time: 0.61 seconds.
Norm: 22.15, NNZs: 35777, Bias: -0.190000, T: 39960, Avg. loss: 0.019962
Total training time: 0.27 seconds.
Norm: 25.59, NNZs: 41180, Bias: -0.090000, T: 39960, Avg. loss: 0.029121
Total training time: 0.28 seconds.
Norm: 24.69, NNZs: 38316, Bias: -0.130000, T: 39960, Avg. loss: 0.025863
Total training time: 0.29 seconds.
-- Epoch 3
-- Epoch 4
Norm: 25.36, NNZs: 43467, Bias: -0.130000, T: 59940, Avg. loss: 0.028815
Total training time: 0.45 seconds.
Norm: 21.86, NNZs: 37912, Bias: -0.310000, T: 39960, Avg. loss: 0.020675
Total training time: 0.29 seconds.
Norm: 24.06, NNZs: 43993, Bias: -0.240000, T: 59940, Avg. loss: 0.025799
Total training time: 0.51 seconds.
-- Epoch 4
-- Epoch 4
-- Epoch 3
Norm: 21.72, NNZs: 38954, Bias: -0.230000, T: 59940, Avg. loss: 0.020103
Total training time: 0.54 seconds.
Norm: 25.65, NNZs: 46538, Bias: -0.150000, T: 79920, Avg. loss: 0.028820
Total training time: 0.57 seconds.
Norm: 21.91, NNZs: 43591, Bias: -0.280000, T: 79920, Avg. loss: 0.020666
Total training time: 0.51 seconds.
Norm: 22.58, NNZs: 43369, Bias: -0.290000, T: 79920, Avg. loss: 0.020741
Total training time: 0.66 seconds.
Norm: 22.39, NNZs: 39819, Bias: -0.240000, T: 59940, Avg. loss: 0.021783
Total training time: 0.38 seconds.
Norm: 23.66, NNZs: 44683, Bias: -0.230000, T: 79920, Avg. loss: 0.022698
Total training time: 0.61 seconds.
Convergence after 5 epochs took 0.78 seconds
-- Epoch 5
Norm: 22.22, NNZs: 39279, Bias: -0.230000, T: 79920, Avg. loss: 0.020810
Total training time: 0.61 seconds.
Convergence after 4 epochs took 0.68 seconds
-- Epoch 4
[Parallel(n_jobs=-1)]: Done   8 out of   8 | elapsed:    0.9s finished
Norm: 21.54, NNZs: 39305, Bias: -0.280000, T: 79920, Avg. loss: 0.020541
Total training time: 0.54 seconds.
Convergence after 5 epochs took 0.83 seconds
-- Epoch 3
[Parallel(n_jobs=-1)]: Done   2 out of   8 | elapsed:    0.9s remaining:    2.6s
Norm: 23.43, NNZs: 42457, Bias: -0.190000, T: 59940, Avg. loss: 0.023388
Total training time: 0.37 seconds.
Convergence after 4 epochs took 0.63 seconds
Norm: 22.39, NNZs: 42002, Bias: -0.260000, T: 79920, Avg. loss: 0.021921
Total training time: 0.55 seconds.
Norm: 21.58, NNZs: 42751, Bias: -0.230000, T: 99900, Avg. loss: 0.020050
Total training time: 0.62 seconds.
-- Epoch 5
-- Epoch 4
-- Epoch 3
-- Epoch 4
Convergence after 4 epochs took 0.63 seconds
-- Epoch 3
-- Epoch 4
Norm: 23.18, NNZs: 46464, Bias: -0.180000, T: 99900, Avg. loss: 0.022202
Total training time: 0.67 seconds.
Norm: 21.77, NNZs: 39022, Bias: -0.220000, T: 59940, Avg. loss: 0.019060
Total training time: 0.39 seconds.
Norm: 24.11, NNZs: 46086, Bias: -0.220000, T: 79920, Avg. loss: 0.025745
Total training time: 0.62 seconds.
-- Epoch 4
Convergence after 4 epochs took 0.67 seconds
-- Epoch 5
-- Epoch 3
-- Epoch 3
Convergence after 5 epochs took 0.70 seconds
Convergence after 4 epochs took 0.60 seconds
Norm: 24.94, NNZs: 44077, Bias: -0.130000, T: 79920, Avg. loss: 0.026534
Total training time: 0.61 seconds.
-- Epoch 3
-- Epoch 4
Norm: 24.09, NNZs: 43656, Bias: -0.230000, T: 59940, Avg. loss: 0.025417
Total training time: 0.44 seconds.
Convergence after 4 epochs took 0.62 seconds
Norm: 22.99, NNZs: 42362, Bias: -0.240000, T: 79920, Avg. loss: 0.021474
Total training time: 0.68 seconds.
[Parallel(n_jobs=-1)]: Done   2 out of   8 | elapsed:    0.9s remaining:    2.6s
Norm: 23.02, NNZs: 43985, Bias: -0.200000, T: 79920, Avg. loss: 0.022058
Norm: 22.33, NNZs: 45093, Bias: -0.250000, T: 99900, Avg. loss: 0.019924
Total training time: 0.77 seconds.
Total training time: 0.63 seconds.
Norm: 25.57, NNZs: 44156, Bias: -0.120000, T: 59940, Avg. loss: 0.028292
Total training time: 0.44 seconds.
Norm: 22.17, NNZs: 37161, Bias: -0.220000, T: 59940, Avg. loss: 0.020391
Total training time: 0.48 seconds.
Norm: 24.49, NNZs: 41365, Bias: -0.170000, T: 59940, Avg. loss: 0.026122
Total training time: 0.45 seconds.
-- Epoch 4
-- Epoch 4
-- Epoch 5
-- Epoch 4
-- Epoch 5
Norm: 25.45, NNZs: 45793, Bias: -0.170000, T: 79920, Avg. loss: 0.028879
-- Epoch 4
Total training time: 0.63 seconds.
Convergence after 5 epochs took 0.71 seconds
Norm: 21.95, NNZs: 41376, Bias: -0.260000, T: 79920, Avg. loss: 0.019888
Total training time: 0.46 seconds.
-- Epoch 4
Norm: 22.96, NNZs: 44034, Bias: -0.250000, T: 99900, Avg. loss: 0.021425
Total training time: 0.66 seconds.
Norm: 22.21, NNZs: 41364, Bias: -0.220000, T: 79920, Avg. loss: 0.020436
Total training time: 0.71 seconds.
Norm: 22.88, NNZs: 41207, Bias: -0.260000, T: 59940, Avg. loss: 0.020090
Total training time: 0.48 seconds.
Norm: 25.55, NNZs: 46262, Bias: -0.170000, T: 79920, Avg. loss: 0.028599
Total training time: 0.49 seconds.
Norm: 22.41, NNZs: 42160, Bias: -0.230000, T: 79920, Avg. loss: 0.021314
Total training time: 0.55 seconds.
Norm: 24.25, NNZs: 47824, Bias: -0.180000, T: 99900, Avg. loss: 0.025401
Total training time: 0.72 seconds.
-- Epoch 4
-- Epoch 5
Norm: 23.46, NNZs: 44895, Bias: -0.220000, T: 79920, Avg. loss: 0.022064
Total training time: 0.52 seconds.
-- Epoch 4
Convergence after 4 epochs took 0.70 seconds
-- Epoch 5
Norm: 23.23, NNZs: 45918, Bias: -0.190000, T: 99900, Avg. loss: 0.022850
Total training time: 0.71 seconds.
Convergence after 4 epochs took 0.79 seconds
Convergence after 5 epochs took 0.72 seconds
Norm: 25.38, NNZs: 47952, Bias: -0.120000, T: 99900, Avg. loss: 0.028131
Norm: 24.52, NNZs: 46032, Bias: -0.240000, T: 79920, Avg. loss: 0.026827
Convergence after 4 epochs took 0.53 seconds
-- Epoch 5
Convergence after 4 epochs took 0.77 seconds
Total training time: 0.55 seconds.
Convergence after 5 epochs took 0.88 seconds
Total training time: 0.54 seconds.
Norm: 24.87, NNZs: 43514, Bias: -0.110000, T: 79920, Avg. loss: 0.025691
Total training time: 0.55 seconds.
-- Epoch 4
[Parallel(n_jobs=-1)]: Done   8 out of   8 | elapsed:    1.1s finished
-- Epoch 6
-- Epoch 5
Convergence after 5 epochs took 0.74 seconds
Convergence after 4 epochs took 0.60 seconds
[Parallel(n_jobs=-1)]: Done   2 out of   8 | elapsed:    0.9s remaining:    2.8s
Norm: 22.42, NNZs: 43443, Bias: -0.290000, T: 79920, Avg. loss: 0.020357
Total training time: 0.55 seconds.
-- Epoch 4
Norm: 24.54, NNZs: 49247, Bias: -0.210000, T: 119880, Avg. loss: 0.025804
Total training time: 0.78 seconds.
Norm: 23.53, NNZs: 46774, Bias: -0.190000, T: 99900, Avg. loss: 0.022219
Total training time: 0.57 seconds.
[Parallel(n_jobs=-1)]: Done   2 out of   5 | elapsed:    1.3s remaining:    1.9s
Norm: 25.48, NNZs: 47304, Bias: -0.110000, T: 99900, Avg. loss: 0.029055
Total training time: 0.73 seconds.
Norm: 21.58, NNZs: 39424, Bias: -0.250000, T: 79920, Avg. loss: 0.020796
Total training time: 0.62 seconds.
Convergence after 4 epochs took 0.58 seconds
-- Epoch 6
-- Epoch 5
-- Epoch 7
Norm: 25.51, NNZs: 49204, Bias: -0.130000, T: 119880, Avg. loss: 0.028758
Total training time: 0.59 seconds.
Norm: 23.94, NNZs: 47974, Bias: -0.190000, T: 99900, Avg. loss: 0.026792
Total training time: 0.60 seconds.
-- Epoch 6
Convergence after 4 epochs took 0.60 seconds
Convergence after 5 epochs took 0.76 seconds
Norm: 24.05, NNZs: 50424, Bias: -0.210000, T: 139860, Avg. loss: 0.025310
Total training time: 0.82 seconds.
Norm: 23.59, NNZs: 48447, Bias: -0.250000, T: 119880, Avg. loss: 0.022490
Total training time: 0.61 seconds.
Convergence after 4 epochs took 0.65 seconds
Convergence after 5 epochs took 0.62 seconds
Convergence after 6 epochs took 0.62 seconds
Convergence after 7 epochs took 0.84 seconds
[Parallel(n_jobs=-1)]: Done   8 out of   8 | elapsed:    1.1s finished
[Parallel(n_jobs=-1)]: Done   8 out of   8 | elapsed:    1.0s finished
Convergence after 6 epochs took 0.62 seconds
[Parallel(n_jobs=-1)]: Done   5 out of   5 | elapsed:    1.4s finished
	accuracy: 5-fold cross validation: [0.3294 0.318  0.3202 0.3254 0.3198]
	test accuracy: 5-fold cross validation accuracy: 0.32 (+/- 0.01)
dimensionality: 74170
density: 0.656150



===> Classification Report:

              precision    recall  f1-score   support

           1       0.53      0.49      0.51      5022
           2       0.14      0.13      0.13      2302
           3       0.18      0.18      0.18      2541
           4       0.20      0.23      0.21      2635
           7       0.19      0.09      0.12      2307
           8       0.20      0.18      0.19      2850
           9       0.16      0.11      0.13      2344
          10       0.41      0.58      0.48      4999

    accuracy                           0.31     25000
   macro avg       0.25      0.25      0.25     25000
weighted avg       0.29      0.31      0.30     25000



===> Classification Metrics:

accuracy classification score
	accuracy score:  0.30992
	accuracy score (normalize=False):  7748

compute the precision
	precision score (average=macro):  0.2501996770023255
	precision score (average=micro):  0.30992
	precision score (average=weighted):  0.2946157694571242
	precision score (average=None):  [0.52651757 0.14203455 0.17653716 0.19921105 0.19418386 0.1952221
 0.15913201 0.40875912]
	precision score (average=None, zero_division=1):  [0.52651757 0.14203455 0.17653716 0.19921105 0.19418386 0.1952221
 0.15913201 0.40875912]

compute the precision
	recall score (average=macro):  0.2504198338776532
	recall score (average=micro):  0.30992
	recall score (average=weighted):  0.30992
	recall score (average=None):  [0.49223417 0.12858384 0.18417946 0.22998102 0.08972692 0.18350877
 0.11262799 0.5825165 ]
	recall score (average=None, zero_division=1):  [0.49223417 0.12858384 0.18417946 0.22998102 0.08972692 0.18350877
 0.11262799 0.5825165 ]

compute the F1 score, also known as balanced F-score or F-measure
	f1 score (average=macro):  0.24522228000517482
	f1 score (average=micro):  0.30992
	f1 score (average=weighted):  0.2967846511851355
	f1 score (average=None):  [0.50879901 0.13497492 0.18027735 0.21349304 0.1227394  0.1891843
 0.13190107 0.48040914]

compute the F-beta score
	f beta score (average=macro):  0.24663341665772368
	f beta score (average=micro):  0.30992
	f beta score (average=weighted):  0.2940047392829202
	f beta score (average=None):  [0.51928409 0.1391239  0.17801445 0.20468824 0.15751027 0.19276132
 0.14699332 0.43469175]

compute the average Hamming loss
	hamming loss:  0.69008

jaccard similarity coefficient score
	jaccard score (average=macro):  0.14859397254838347
	jaccard score (average=None):  [0.34120083 0.07237164 0.09906859 0.11950306 0.06538219 0.10447463
 0.07060711 0.31614374]

confusion matrix:
[[2472  520  620  487   80  177  114  552]
 [ 725  296  357  397   55  112   84  276]
 [ 525  349  468  513  103  171   86  326]
 [ 382  324  439  606  141  247  125  371]
 [ 123  160  226  313  207  455  222  601]
 [ 139  146  208  275  197  523  310 1052]
 [ 118  106  127  170  115  410  264 1034]
 [ 211  183  206  281  168  584  454 2912]]

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
                       n_jobs=-1, oob_score=False, random_state=0, verbose=True,
                       warm_start=False)
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 8 concurrent workers.
[Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:    6.4s
[Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:   16.2s finished
train time: 16.383s
[Parallel(n_jobs=8)]: Using backend ThreadingBackend with 8 concurrent workers.
[Parallel(n_jobs=8)]: Done  34 tasks      | elapsed:    0.3s
[Parallel(n_jobs=8)]: Done 100 out of 100 | elapsed:    0.5s finished
test time:  0.617s
accuracy:   0.370


cross validation:
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 8 concurrent workers.
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 8 concurrent workers.
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 8 concurrent workers.
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 8 concurrent workers.
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 8 concurrent workers.
[Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:   31.5s
[Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:   32.3s
[Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:   32.4s
[Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:   32.6s
[Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:   32.6s
[Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:  1.3min finished
[Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:  1.4min finished
[Parallel(n_jobs=8)]: Using backend ThreadingBackend with 8 concurrent workers.
[Parallel(n_jobs=8)]: Using backend ThreadingBackend with 8 concurrent workers.
[Parallel(n_jobs=8)]: Done  34 tasks      | elapsed:    0.2s
[Parallel(n_jobs=8)]: Done  34 tasks      | elapsed:    0.2s
[Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:  1.4min finished
[Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:  1.4min finished
[Parallel(n_jobs=8)]: Using backend ThreadingBackend with 8 concurrent workers.
[Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:  1.4min finished
[Parallel(n_jobs=8)]: Using backend ThreadingBackend with 8 concurrent workers.
[Parallel(n_jobs=8)]: Done 100 out of 100 | elapsed:    0.3s finished
[Parallel(n_jobs=8)]: Using backend ThreadingBackend with 8 concurrent workers.
[Parallel(n_jobs=8)]: Done 100 out of 100 | elapsed:    0.3s finished
[Parallel(n_jobs=8)]: Done  34 tasks      | elapsed:    0.1s
[Parallel(n_jobs=8)]: Done  34 tasks      | elapsed:    0.2s
[Parallel(n_jobs=-1)]: Done   2 out of   5 | elapsed:  1.4min remaining:  2.0min
[Parallel(n_jobs=8)]: Done  34 tasks      | elapsed:    0.1s
[Parallel(n_jobs=8)]: Done 100 out of 100 | elapsed:    0.3s finished
[Parallel(n_jobs=8)]: Done 100 out of 100 | elapsed:    0.3s finished
[Parallel(n_jobs=8)]: Done 100 out of 100 | elapsed:    0.3s finished
[Parallel(n_jobs=-1)]: Done   5 out of   5 | elapsed:  1.4min finished
	accuracy: 5-fold cross validation: [0.3588 0.3712 0.3626 0.3668 0.36  ]
	test accuracy: 5-fold cross validation accuracy: 0.36 (+/- 0.01)


===> Classification Report:

              precision    recall  f1-score   support

           1       0.38      0.89      0.53      5022
           2       0.49      0.01      0.02      2302
           3       0.34      0.02      0.03      2541
           4       0.30      0.06      0.10      2635
           7       0.28      0.05      0.08      2307
           8       0.24      0.10      0.14      2850
           9       0.30      0.01      0.01      2344
          10       0.38      0.83      0.52      4999

    accuracy                           0.37     25000
   macro avg       0.34      0.24      0.18     25000
weighted avg       0.34      0.37      0.25     25000



===> Classification Metrics:

accuracy classification score
	accuracy score:  0.37028
	accuracy score (normalize=False):  9257

compute the precision
	precision score (average=macro):  0.3383695751907134
	precision score (average=micro):  0.37028
	precision score (average=weighted):  0.3446353830739655
	precision score (average=None):  [0.37911531 0.48717949 0.3358209  0.29924242 0.28346457 0.24392439
 0.29787234 0.38033718]
	precision score (average=None, zero_division=1):  [0.37911531 0.48717949 0.3358209  0.29924242 0.28346457 0.24392439
 0.29787234 0.38033718]

compute the precision
	recall score (average=macro):  0.2448038838610155
	recall score (average=micro):  0.37028
	recall score (average=weighted):  0.37028
	recall score (average=None):  [0.89426523 0.00825369 0.01770956 0.05996205 0.04681404 0.09508772
 0.0059727  0.83036607]
	recall score (average=None, zero_division=1):  [0.89426523 0.00825369 0.01770956 0.05996205 0.04681404 0.09508772
 0.0059727  0.83036607]

compute the F1 score, also known as balanced F-score or F-measure
	f1 score (average=macro):  0.17911045097740516
	f1 score (average=micro):  0.37028
	f1 score (average=weighted):  0.25084436883116956
	f1 score (average=None):  [0.53248755 0.01623238 0.03364486 0.09990515 0.08035714 0.13683413
 0.01171058 0.52171181]

compute the F-beta score
	f beta score (average=macro):  0.18595254151498258
	f beta score (average=micro):  0.37028
	f beta score (average=weighted):  0.23668024027531337
	f beta score (average=None):  [0.42848147 0.03864931 0.07312317 0.1664209  0.14095536 0.18576913
 0.02764613 0.42657486]

compute the average Hamming loss
	hamming loss:  0.62972

jaccard similarity coefficient score
	jaccard score (average=macro):  0.11435381225030658
	jaccard score (average=None):  [0.36285045 0.0081826  0.01711027 0.05257903 0.04186047 0.07344173
 0.00588978 0.35291617]

confusion matrix:
[[4491    2    8   27   14   20    1  459]
 [1862   19   17   46   16   33    2  307]
 [1820    8   45  110   28   62    1  467]
 [1607    5   36  158   53  134    2  640]
 [ 605    2   13   73  108  248    6 1252]
 [ 542    1    6   47   76  271   10 1897]
 [ 346    2    4   34   44  159   14 1741]
 [ 573    0    5   33   42  184   11 4151]]

================================================================================
Classifier.RIDGE_CLASSIFIER
________________________________________________________________________________
Training: 
RidgeClassifier(alpha=1.0, class_weight=None, copy_X=True, fit_intercept=True,
                max_iter=None, normalize=False, random_state=0, solver='auto',
                tol=0.001)
train time: 2.942s
test time:  0.041s
accuracy:   0.386


cross validation:
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
[Parallel(n_jobs=-1)]: Done   2 out of   5 | elapsed:    1.6s remaining:    2.4s
	accuracy: 5-fold cross validation: [0.4036 0.4074 0.402  0.3954 0.4   ]
	test accuracy: 5-fold cross validation accuracy: 0.40 (+/- 0.01)
dimensionality: 74170
density: 1.000000



===> Classification Report:

[Parallel(n_jobs=-1)]: Done   5 out of   5 | elapsed:    2.2s finished
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
|  1  |  ADA_BOOST_CLASSIFIER  |  38.02%  |  [0.3792 0.379  0.374  0.3704 0.3746]  |  0.38 (+/- 0.01)  |  120.7  |  7.37  |
|  2  |  BERNOULLI_NB  |  37.03%  |  [0.377  0.389  0.3782 0.38   0.373 ]  |  0.38 (+/- 0.01)  |  0.04659  |  0.04255  |
|  3  |  COMPLEMENT_NB  |  37.34%  |  [0.3878 0.3942 0.3976 0.3938 0.3832]  |  0.39 (+/- 0.01)  |  0.0406  |  0.02028  |
|  4  |  DECISION_TREE_CLASSIFIER  |  31.34%  |  [0.3058 0.3122 0.3044 0.3036 0.311 ]  |  0.31 (+/- 0.01)  |  3.227  |  0.01253  |
|  5  |  GRADIENT_BOOSTING_CLASSIFIER  |  37.36%  |  [0.38   0.3688 0.3596 0.3634 0.3682]  |  0.37 (+/- 0.01)  |  433.1  |  0.2829  |
|  6  |  K_NEIGHBORS_CLASSIFIER  |  37.26%  |  [0.3822 0.3916 0.3842 0.386  0.388 ]  |  0.39 (+/- 0.01)  |  0.007069  |  14.19  |
|  7  |  LINEAR_SVC  |  40.80%  |  [0.41   0.4206 0.4064 0.3992 0.4088]  |  0.41 (+/- 0.01)  |  0.5336  |  0.0213  |
|  8  |  LOGISTIC_REGRESSION  |  42.31%  |  [0.4286 0.4296 0.4152 0.4178 0.4224]  |  0.42 (+/- 0.01)  |  2.719  |  0.02371  |
|  9  |  MULTINOMIAL_NB  |  37.82%  |  [0.389  0.3928 0.3918 0.3942 0.386 ]  |  0.39 (+/- 0.01)  |  0.03834  |  0.02014  |
|  10  |  NEAREST_CENTROID  |  37.33%  |  [0.3872 0.3786 0.3894 0.3672 0.3782]  |  0.38 (+/- 0.02)  |  0.03892  |  0.03368  |
|  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  41.81%  |  [0.4172 0.4284 0.4096 0.409  0.4164]  |  0.42 (+/- 0.01)  |  0.52  |  0.01889  |
|  12  |  PERCEPTRON  |  30.99%  |  [0.3294 0.318  0.3202 0.3254 0.3198]  |  0.32 (+/- 0.01)  |  0.5218  |  0.06201  |
|  13  |  RANDOM_FOREST_CLASSIFIER  |  37.03%  |  [0.3588 0.3712 0.3626 0.3668 0.36  ]  |  0.36 (+/- 0.01)  |  16.38  |  0.617  |
|  14  |  RIDGE_CLASSIFIER  |  38.55%  |  [0.4036 0.4074 0.402  0.3954 0.4   ]  |  0.40 (+/- 0.01)  |  2.942  |  0.04052  |


Best algorithm:
===> 8) LOGISTIC_REGRESSION
		Accuracy score = 42.31%		Training time = 2.719		Test time = 0.02371



DONE!
Program finished. It took 2199.9160685539246 seconds
 This problem is unconstrained.
 This problem is unconstrained.
 This problem is unconstrained.
 This problem is unconstrained.
 This problem is unconstrained.
 This problem is unconstrained.
 This problem is unconstrained.
 This problem is unconstrained.
 This problem is unconstrained.
 This problem is unconstrained.
 This problem is unconstrained.
 This problem is unconstrained.
 This problem is unconstrained.
 This problem is unconstrained.
 This problem is unconstrained.
 This problem is unconstrained.
 This problem is unconstrained.
 This problem is unconstrained.
 This problem is unconstrained.
 This problem is unconstrained.
 This problem is unconstrained.
 This problem is unconstrained.
 This problem is unconstrained.
 This problem is unconstrained.
 This problem is unconstrained.
 This problem is unconstrained.
 This problem is unconstrained.
 This problem is unconstrained.
 This problem is unconstrained.
 This problem is unconstrained.
 This problem is unconstrained.
 This problem is unconstrained.
 This problem is unconstrained.
 This problem is unconstrained.
 This problem is unconstrained.
 This problem is unconstrained.
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =        74171     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  1.73287D+04    |proj g|=  9.49100D+03

At iterate   50    f=  7.34242D+03    |proj g|=  8.17966D-01

           * * *

Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
74171     60     68      1     0     0   2.945D-02   7.342D+03
  F =   7342.4177995262398     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =        74171     M =           10

At X0         0 variables are exactly at the bounds
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =        74171     M =           10

At X0         0 variables are exactly at the bounds
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =        74171     M =           10

At X0         0 variables are exactly at the bounds
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =        74171     M =           10

At X0         0 variables are exactly at the bounds
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =        74171     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  1.38629D+04    |proj g|=  8.17300D+03

At iterate    0    f=  1.38629D+04    |proj g|=  7.59300D+03
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =        74171     M =           10

At X0         0 variables are exactly at the bounds
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =        74171     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  1.38629D+04    |proj g|=  8.00300D+03
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =        74171     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  1.38629D+04    |proj g|=  8.19000D+03

At iterate    0    f=  1.38629D+04    |proj g|=  6.21400D+03

At iterate    0    f=  1.38629D+04    |proj g|=  5.92000D+03

At iterate    0    f=  1.38629D+04    |proj g|=  8.06400D+03

At iterate    0    f=  1.38629D+04    |proj g|=  7.84300D+03

           * * *

Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
74171     43     48      1     0     0   2.575D-02   5.223D+03
  F =   5222.5921686115908     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             

           * * *

Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
74171     49     53      1     0     0   7.895D-03   5.577D+03
  F =   5577.4844023307205     

CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL            

At iterate   50    f=  5.90114D+03    |proj g|=  1.15205D-02

At iterate   50    f=  5.02285D+03    |proj g|=  7.53486D-02

At iterate   50    f=  4.94380D+03    |proj g|=  1.12893D-01

           * * *

Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
74171     52     57      1     0     0   7.525D-03   5.023D+03
  F =   5022.8450011963396     

CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL            

At iterate   50    f=  6.67726D+03    |proj g|=  1.80453D-01

At iterate   50    f=  6.41957D+03    |proj g|=  4.70320D-01

At iterate   50    f=  5.29118D+03    |proj g|=  1.43080D-01

           * * *

Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
74171     56     62      1     0     0   4.693D-03   5.901D+03
  F =   5901.1418808157250     

CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL            

           * * *

Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
74171     54     60      1     0     0   1.408D-03   5.291D+03
  F =   5291.1778404089282     

CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL            

           * * *

Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
74171     57     63      1     0     0   7.752D-02   4.944D+03
  F =   4943.8000411339435     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             

           * * *

Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
74171     63     71      1     0     0   1.078D-02   6.420D+03
  F =   6419.5650765134324     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             

           * * *

Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
74171     66     77      1     0     0   1.100D-01   6.677D+03
  F =   6677.2528867375886     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =        74171     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  1.73287D+04    |proj g|=  1.00040D+04

At iterate   50    f=  6.49948D+03    |proj g|=  1.51158D+00

           * * *

Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
74171     62     66      1     0     0   8.506D-03   6.499D+03
  F =   6499.4731067077882     

CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL            
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =        74171     M =           10

At X0         0 variables are exactly at the bounds
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =        74171     M =           10

At X0         0 variables are exactly at the bounds
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =        74171     M =           10

At X0         0 variables are exactly at the bounds
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =        74171     M =           10

At X0         0 variables are exactly at the bounds
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =        74171     M =           10

At X0         0 variables are exactly at the bounds
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =        74171     M =           10

At X0         0 variables are exactly at the bounds
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =        74171     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  1.38629D+04    |proj g|=  8.17300D+03
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =        74171     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  1.38629D+04    |proj g|=  5.92000D+03

At iterate    0    f=  1.38629D+04    |proj g|=  8.06400D+03

At iterate    0    f=  1.38629D+04    |proj g|=  8.00300D+03

At iterate    0    f=  1.38629D+04    |proj g|=  7.84300D+03

At iterate    0    f=  1.38629D+04    |proj g|=  7.59300D+03

At iterate    0    f=  1.38629D+04    |proj g|=  6.21400D+03

At iterate    0    f=  1.38629D+04    |proj g|=  8.19000D+03

           * * *

Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
74171     40     43      1     0     0   4.757D-03   6.402D+03
  F =   6402.0262870793722     

CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL            

           * * *

Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
74171     44     49      1     0     0   4.912D-02   5.035D+03
  F =   5034.9399276408858     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             

           * * *

Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
74171     46     53      1     0     0   6.937D-03   6.699D+03
  F =   6698.7465059677470     

CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL            

At iterate   50    f=  5.21930D+03    |proj g|=  1.13423D-01

At iterate   50    f=  5.91786D+03    |proj g|=  1.65807D-02

           * * *

Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
74171     47     54      1     0     0   2.205D-02   4.953D+03
  F =   4953.4654794628750     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             

At iterate   50    f=  5.26953D+03    |proj g|=  1.09835D-01

           * * *

Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
74171     52     61      1     0     0   7.773D-02   5.270D+03
  F =   5269.5296278893029     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             

           * * *

Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
74171     48     58      1     0     0   3.491D-02   5.611D+03
  F =   5610.7881774715743     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             

           * * *

Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
74171     58     65      1     0     0   6.446D-03   5.219D+03
  F =   5219.2957596053984     

CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL            

           * * *

Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
74171     60     66      1     0     0   8.588D-03   5.918D+03
  F =   5917.8556789360136     

CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL            
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =        74171     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  1.73287D+04    |proj g|=  7.40000D+03

At iterate   50    f=  7.86301D+03    |proj g|=  1.01424D+00

           * * *

Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
74171     61     69      1     0     0   1.295D-02   7.863D+03
  F =   7862.9632481031549     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =        74171     M =           10

At X0         0 variables are exactly at the bounds
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =        74171     M =           10

At X0         0 variables are exactly at the bounds
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =        74171     M =           10

At X0         0 variables are exactly at the bounds
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =        74171     M =           10

At X0         0 variables are exactly at the bounds
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =        74171     M =           10

At X0         0 variables are exactly at the bounds
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =        74171     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  1.38629D+04    |proj g|=  8.06400D+03

At iterate    0    f=  1.38629D+04    |proj g|=  8.17300D+03
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =        74171     M =           10

At X0         0 variables are exactly at the bounds
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =        74171     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  1.38629D+04    |proj g|=  8.19000D+03

At iterate    0    f=  1.38629D+04    |proj g|=  8.00300D+03

At iterate    0    f=  1.38629D+04    |proj g|=  7.59300D+03

At iterate    0    f=  1.38629D+04    |proj g|=  7.84300D+03

At iterate    0    f=  1.38629D+04    |proj g|=  6.21400D+03

At iterate    0    f=  1.38629D+04    |proj g|=  5.92000D+03

           * * *

Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
74171     38     43      1     0     0   9.592D-03   6.389D+03
  F =   6388.9237648497201     

CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL            

At iterate   50    f=  5.22478D+03    |proj g|=  1.55685D-02

At iterate   50    f=  4.93519D+03    |proj g|=  2.19892D-01

           * * *

Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
74171     51     56      1     0     0   5.077D-03   5.225D+03
  F =   5224.7830840371753     

CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL            

           * * *

Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
74171     48     55      1     0     0   4.472D-03   5.616D+03
  F =   5615.6636667878174     

CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL            

At iterate   50    f=  5.91905D+03    |proj g|=  7.15787D-02

At iterate   50    f=  5.04531D+03    |proj g|=  5.81774D-03

           * * *

Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
74171     50     54      1     0     0   5.818D-03   5.045D+03
  F =   5045.3108640439086     

CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL            

At iterate   50    f=  6.68329D+03    |proj g|=  4.04305D-01

At iterate   50    f=  5.27208D+03    |proj g|=  7.23765D-02

           * * *

Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
74171     55     66      1     0     0   3.936D-03   5.272D+03
  F =   5272.0755940931649     

CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL            

           * * *

Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
74171     64     70      1     0     0   8.254D-02   4.935D+03
  F =   4935.1895008446263     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             

           * * *

Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
74171     61     70      1     0     0   1.529D-01   5.919D+03
  F =   5919.0489881738777     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             

           * * *

Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
74171     63     75      1     0     0   4.343D-01   6.683D+03
  F =   6683.2855437539092     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =        74171     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  1.73287D+04    |proj g|=  1.02370D+04

           * * *

Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
74171     42     47      1     0     0   2.269D-01   6.144D+03
  F =   6144.2934862286002     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =        74171     M =           10

At X0         0 variables are exactly at the bounds
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =        74171     M =           10

At X0         0 variables are exactly at the bounds
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =        74171     M =           10

At X0         0 variables are exactly at the bounds
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =        74171     M =           10

At X0         0 variables are exactly at the bounds
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =        74171     M =           10

At X0         0 variables are exactly at the bounds
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =        74171     M =           10

At X0         0 variables are exactly at the bounds
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =        74171     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  1.38629D+04    |proj g|=  8.06400D+03
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =        74171     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  1.38629D+04    |proj g|=  5.92000D+03

At iterate    0    f=  1.38629D+04    |proj g|=  7.59200D+03

At iterate    0    f=  1.38629D+04    |proj g|=  8.00400D+03

At iterate    0    f=  1.38629D+04    |proj g|=  8.17200D+03

At iterate    0    f=  1.38629D+04    |proj g|=  6.21500D+03

At iterate    0    f=  1.38629D+04    |proj g|=  7.84400D+03

At iterate    0    f=  1.38629D+04    |proj g|=  8.18900D+03

           * * *

Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
74171     43     48      1     0     0   8.953D-03   4.931D+03
  F =   4930.5869182416209     

CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL            

At iterate   50    f=  5.26115D+03    |proj g|=  8.91480D-03

           * * *

Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
74171     50     52      1     0     0   8.915D-03   5.261D+03
  F =   5261.1497373597831     

CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL            

At iterate   50    f=  5.60962D+03    |proj g|=  5.10605D-01

At iterate   50    f=  5.27976D+03    |proj g|=  6.84587D-02

At iterate   50    f=  5.03028D+03    |proj g|=  9.41612D-01

At iterate   50    f=  5.91049D+03    |proj g|=  6.92997D-02

At iterate   50    f=  6.69443D+03    |proj g|=  1.64224D-01

           * * *

Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
74171     52     58      1     0     0   4.889D-03   5.280D+03
  F =   5279.7599575837212     

CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL            

At iterate   50    f=  6.41510D+03    |proj g|=  8.38146D-02

           * * *

Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
74171     59     69      1     0     0   3.748D-03   5.610D+03
  F =   5609.6221981726949     

CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL            

           * * *

Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
74171     59     67      1     0     0   2.436D-03   5.910D+03
  F =   5910.4908896110555     

CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL            

           * * *

Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
74171     59     68      1     0     0   4.206D-03   6.415D+03
  F =   6415.1025798844221     

CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL            

           * * *

Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
74171     63     71      1     0     0   7.264D-03   6.694D+03
  F =   6694.3797360185454     

CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL            

           * * *

Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
74171     64     70      1     0     0   2.529D-01   5.030D+03
  F =   5030.2748162644011     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
 This problem is unconstrained.
 This problem is unconstrained.
 This problem is unconstrained.
 This problem is unconstrained.
 This problem is unconstrained.
 This problem is unconstrained.
 This problem is unconstrained.
 This problem is unconstrained.
 This problem is unconstrained.
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =        74171     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  1.73287D+04    |proj g|=  1.02160D+04

At iterate   50    f=  6.26205D+03    |proj g|=  5.89953D-01

           * * *

Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
74171     57     64      1     0     0   2.770D-02   6.262D+03
  F =   6262.0492985093560     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =        74171     M =           10

At X0         0 variables are exactly at the bounds
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =        74171     M =           10

At X0         0 variables are exactly at the bounds
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =        74171     M =           10

At X0         0 variables are exactly at the bounds
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =        74171     M =           10

At X0         0 variables are exactly at the bounds
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =        74171     M =           10

At X0         0 variables are exactly at the bounds
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =        74171     M =           10

At X0         0 variables are exactly at the bounds
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =        74171     M =           10

At X0         0 variables are exactly at the bounds
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =        74171     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  1.38629D+04    |proj g|=  8.17300D+03

At iterate    0    f=  1.38629D+04    |proj g|=  8.00300D+03

At iterate    0    f=  1.38629D+04    |proj g|=  8.18900D+03

At iterate    0    f=  1.38629D+04    |proj g|=  5.92000D+03

At iterate    0    f=  1.38629D+04    |proj g|=  6.21500D+03

At iterate    0    f=  1.38629D+04    |proj g|=  7.84300D+03

At iterate    0    f=  1.38629D+04    |proj g|=  7.59300D+03

At iterate    0    f=  1.38629D+04    |proj g|=  8.06400D+03

           * * *

Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
74171     43     49      1     0     0   6.124D-02   4.942D+03
  F =   4942.2224854816241     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             

           * * *

Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
74171     46     51      1     0     0   5.628D-03   5.037D+03
  F =   5036.6502709279985     

CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL            

           * * *

Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
74171     47     54      1     0     0   4.820D-03   6.432D+03
  F =   6432.1868311455110     

CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL            

At iterate   50    f=  6.69043D+03    |proj g|=  8.46592D-01

           * * *

Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
74171     48     53      1     0     0   7.405D-03   5.258D+03
  F =   5257.7876598643015     

CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL            

At iterate   50    f=  5.58533D+03    |proj g|=  3.01481D-02

           * * *

Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
74171     51     58      1     0     0   1.176D-02   5.585D+03
  F =   5585.3331370311062     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             

At iterate   50    f=  5.88214D+03    |proj g|=  4.82765D-01

At iterate   50    f=  5.28142D+03    |proj g|=  2.86499D-01

           * * *

Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
74171     53     59      1     0     0   8.183D-03   5.882D+03
  F =   5882.1393776162022     

CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL            

           * * *

Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
74171     59     65      1     0     0   3.086D-01   5.281D+03
  F =   5281.4128702251246     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             

           * * *

Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
74171     70     78      1     0     0   6.233D-01   6.690D+03
  F =   6690.4056959635782     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
 This problem is unconstrained.
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =        74171     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  1.73287D+04    |proj g|=  7.76800D+03

At iterate   50    f=  8.23219D+03    |proj g|=  7.02270D+00

           * * *

Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
74171     71     81      1     0     0   4.647D-02   8.232D+03
  F =   8231.9667400064682     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
 This problem is unconstrained.
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =        74171     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  1.73287D+04    |proj g|=  1.00800D+04

At iterate   50    f=  6.56877D+03    |proj g|=  2.23879D-01

           * * *

Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
74171     63     69      1     0     0   3.353D-03   6.569D+03
  F =   6568.7630821725461     

CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL            
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =        74171     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  1.73287D+04    |proj g|=  9.80400D+03

At iterate   50    f=  6.95534D+03    |proj g|=  4.93032D-01

           * * *

Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *
 This problem is unconstrained.

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
74171     63     72      1     0     0   3.970D-01   6.955D+03
  F =   6955.3226686313528     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             

Process finished with exit code 0

```