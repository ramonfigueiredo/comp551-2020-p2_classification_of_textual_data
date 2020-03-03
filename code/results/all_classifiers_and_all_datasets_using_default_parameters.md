## Results using all classifiers and all datasets (classifier with default parameters)

### TWENTY_NEWS_GROUPS dataset (removing headers signatures and quoting)  

#### TWENTY_NEWS_GROUPS: Accuracy score table

| ID | ML Algorithm                     | Accuracy Score   | K-fold Cross Validation (CV) (k = 5)  |  CV (Mean +/- Std)       | Training time (seconds)  | Test time (seconds)  | 
| -- | -------------------------------  | ---------------- | ------------------------------------- |------------------------  | ------------------------ | -------------------  |
| 1  | ADA_BOOST_CLASSIFIER             | 36.54%           |                                       |                          | 4.842                    | 0.256                |
| 2  | BERNOULLI_NB                     | 45.84%           |                                       |                          | 0.062                    | 0.053                |
| 3  | COMPLEMENT_NB                    | **71.34%**       |                                       |                          | 0.063                    | 0.010                |
| 4  | DECISION_TREE_CLASSIFIER         | 43.92%           |                                       |                          | 10.921                   | 0.006                |
| 5  | EXTRA_TREE_CLASSIFIER            | 29.42%           |                                       |                          | 0.578                    | 0.009                |
| 6  | EXTRA_TREES_CLASSIFIER           | 65.31%           |                                       |                          | 10.459                   | 0.204                |
| 7  | GRADIENT_BOOSTING_CLASSIFIER     | 59.68%           |                                       |                          | 337.842                  | 0.181                |
| 8  | K_NEIGHBORS_CLASSIFIER           | 07.01%           |                                       |                          | 0.002                    | 1.693                |
| 9  | LINEAR_SVC                       | 69.68%           |                                       |                          | 0.763                    | 0.009                |
| 10 | LOGISTIC_REGRESSION              | 69.46%           |                                       |                          | 17.369                   | 0.011                |
| 11 | LOGISTIC_REGRESSION_CV           | 69.35%           |                                       |                          | 409.508                  | 0.011                |
| 12 | MLP_CLASSIFIER                   | 69.98%           |                                       |                          | 1357.282                 | 0.040                |
| 13 | MULTINOMIAL_NB                   | 67.13%           |                                       |                          | 0.083                    | 0.010                |
| 14 | NEAREST_CENTROID                 | 64.27%           |                                       |                          | 0.016                    | 0.013                |
| 15 | NU_SVC                           | 69.20%           |                                       |                          | 82.380                   | 27.448               |
| 16 | PASSIVE_AGGRESSIVE_CLASSIFIER    | 68.48%           |                                       |                          | 0.410                    | 0.013                |
| 17 | PERCEPTRON                       | 63.36%           |                                       |                          | 0.411                    | 0.013                |
| 18 | RANDOM_FOREST_CLASSIFIER         | 62.68%           |                                       |                          | 6.569                    | 0.305                |
| 19 | RIDGE_CLASSIFIER                 | 70.35%           |                                       |                          | 2.367                    | 0.021                |
| 20 | RIDGE_CLASSIFIERCV               | 70.37%           |                                       |                          | 173.823                  | 0.018                |
| 21 | SGD_CLASSIFIER                   | 70.11%           |                                       |                          | 0.422                    | 0.011                |

#### TWENTY_NEWS_GROUPS: Plotting

* Accuracy score for the TWENTY_NEWS_GROUPS dataset (Removing headers signatures and quoting)
 
    ![TWENTY_NEWS_GROUPS](https://github.com/ramonfigueiredopessoa/comp551-2020-p2_classification_of_textual_data/blob/master/code/results/TWENTY_NEWS_GROUPS-ml_with_default_parameters.png)


### IMDB_REVIEWS dataset (Multi-class classification)  

#### IMDB_REVIEWS: Accuracy score table

| ID | ML Algorithm 				    | Accuracy Score   |
| -- | -------------------------------  | ---------------- |
| 1  | ADA_BOOST_CLASSIFIER             | 35.86		       |
| 2  | BERNOULLI_NB                     | 37.132           |
| 3  | COMPLEMENT_NB                    | 37.312		   |
| 4  | DECISION_TREE_CLASSIFIER         | 25.764		   |
| 5  | EXTRA_TREE_CLASSIFIER            | 22.12		       |
| 6  | EXTRA_TREES_CLASSIFIER           | 37.404		   |
| 7  | GRADIENT_BOOSTING_CLASSIFIER     | 37.624		   |
| 8  | K_NEIGHBORS_CLASSIFIER           | 26.352	       |
| 9  | LINEAR_SVC                       | 37.328		   |
| 10 | LOGISTIC_REGRESSION              | 42.084		   |
| 11 | LOGISTIC_REGRESSION_CV           | 40.532		   |
| 12 | MLP_CLASSIFIER                   | 34.468	       |
| 13 | MULTINOMIAL_NB                   | 34.924		   |
| 14 | NEAREST_CENTROID                 | 36.844    	   |
| 15 | NU_SVC                           | **42.32**	       |
| 16 | PASSIVE_AGGRESSIVE_CLASSIFIER    | 33.112		   |
| 17 | PERCEPTRON                       | 32.66		       |
| 18 | RANDOM_FOREST_CLASSIFIER         | 37.54	           |
| 19 | RIDGE_CLASSIFIER                 | 38.716		   |
| 20 | RIDGE_CLASSIFIERCV               | 41.54		       |
| 21 | SGD_CLASSIFIER                   | 40.676	       |

#### IMDB_REVIEWS: Plotting

* Accuracy score of IMDB_REVIEWS dataset (Multi-class classification)
 
    ![IMDB_REVIEWS](https://github.com/ramonfigueiredopessoa/comp551-2020-p2_classification_of_textual_data/blob/master/code/results/IMDB_REVIEWS-ml_with_default_parameters.png)

### Logs: TWENTY_NEWS_GROUPS and IMDB_REVIEWS (Multi-class classification)

Logs after run all classifiers using the TWENTY_NEWS_GROUPS dataset (removing headers signatures and quoting) and IMDB_REVIEWS dataset (Multi-class classification).

```
python /comp551-2020-p2_classification_of_textual_data/code/main.py -verbose
03/03/2020 09:09:43 AM - INFO - Program started...
==================================================================================================================================
MiniProject 2: Classification of textual data. Authors: Ramon Figueiredo Pessoa, Rafael Gomes Braga, Ege Odaci

Running with options: 
	Dataset = ALL
	ML algorithm list (If ml_algorithm_list = None, all ML algorithms will be executed) = None
	Read dataset without shuffle data = False
	The number of CPUs to use to do the computation. If the provided number is negative or greater than the number of available CPUs, the system will use all the available CPUs. Default: -1 (-1 == all CPUs) = -1
	Run cross validation. Default: False = False
	Number of cross validation folds. Default: 5 = 5
	Use just the miniproject classifiers (1. LogisticRegression, 2. DecisionTreeClassifier, 3. LinearSVC, 4. AdaBoostClassifier, 5. RandomForestClassifier) =  False
	TWENTY_NEWS_GROUPS dataset using some categories (alt.atheism, talk.religion.misc, comp.graphics, sci.space) = False
	Do not remove newsgroup information that is easily overfit (headers, footers, quotes) = False
	Use IMDB Binary Labels (Negative / Positive) = False
	Show the IMDB_REVIEWS and respective labels while read the dataset = False
	Print Classification Report = False
	Print all classification metrics =  False
	Select some number of features using a chi-squared test = None
	Print the confusion matrix = False
	Print ten most discriminative terms per class for every classifier = False
	Use a hashing vectorizer = False
	N features when using the hashing vectorizer = 65536
	Plot training time and test time together with accuracy score = False
	Save logs in a file = False
	Seed used by the random number generator (random_state) = 0
	Verbose = True
==================================================================================================================================

Loading TWENTY_NEWS_GROUPS dataset for categories:
data loaded
11314 documents - 13.782MB (training set)
7532 documents - 8.262MB (test set)
20 categories

Extracting features from the training data using a vectorizer
done in 1.086463s at 12.685MB/s
n_samples: 11314, n_features: 101322

Extracting features from the test data using the same vectorizer
done in 0.575287s at 14.361MB/s
n_samples: 7532, n_features: 101322

================================================================================
Classifier.ADA_BOOST_CLASSIFIER
________________________________________________________________________________
Training: 
AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
                   n_estimators=50, random_state=0)
train time: 4.842s
test time:  0.256s
accuracy:   0.365

================================================================================
Classifier.BERNOULLI_NB
________________________________________________________________________________
Training: 
BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
train time: 0.062s
test time:  0.053s
accuracy:   0.458
dimensionality: 101322
density: 1.000000


================================================================================
Classifier.COMPLEMENT_NB
________________________________________________________________________________
Training: 
ComplementNB(alpha=1.0, class_prior=None, fit_prior=True, norm=False)
train time: 0.063s
test time:  0.010s
accuracy:   0.713
dimensionality: 101322
density: 1.000000


================================================================================
Classifier.DECISION_TREE_CLASSIFIER
________________________________________________________________________________
Training: 
DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=None, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=0, splitter='best')
train time: 10.921s
test time:  0.006s
accuracy:   0.439

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
train time: 0.578s
test time:  0.009s
accuracy:   0.294

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
                     oob_score=False, random_state=0, verbose=True,
                     warm_start=False)
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 12 concurrent workers.
[Parallel(n_jobs=-1)]: Done  26 tasks      | elapsed:    3.9s
[Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:   10.4s finished
train time: 10.459s
[Parallel(n_jobs=12)]: Using backend ThreadingBackend with 12 concurrent workers.
[Parallel(n_jobs=12)]: Done  26 tasks      | elapsed:    0.1s
[Parallel(n_jobs=12)]: Done 100 out of 100 | elapsed:    0.2s finished
test time:  0.204s
accuracy:   0.653

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
         1       28056.6825            5.81m
         2       26199.6361            5.74m
         3       24912.0118            5.68m
         4       23912.0968            5.64m
         5       23131.4108            5.58m
         6       22425.3725            5.53m
         7       21795.5900            5.47m
         8       21252.9841            5.43m
         9       20737.4870            5.44m
        10       20288.5082            5.43m
        20       17198.0626            4.69m
        30       15104.5420            4.06m
        40       13626.6847            3.45m
        50       12376.4629            2.86m
        60       11409.5329            2.28m
        70       10612.1645            1.70m
        80        9862.0856            1.13m
        90        9150.1695           33.89s
       100        8541.8865            0.00s
train time: 337.842s
test time:  0.181s
accuracy:   0.597

================================================================================
Classifier.K_NEIGHBORS_CLASSIFIER
________________________________________________________________________________
Training: 
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=-1, n_neighbors=5, p=2,
                     weights='uniform')
train time: 0.002s
test time:  1.693s
accuracy:   0.070

================================================================================
Classifier.LINEAR_SVC
________________________________________________________________________________
Training: 
LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
          intercept_scaling=1, loss='squared_hinge', max_iter=1000,
          multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
          verbose=True)
...*
optimization finished, #iter = 35
Objective value = -473.691648
nSV = 3834
...*.
optimization finished, #iter = 40
Objective value = -556.684631
nSV = 4090
....*
optimization finished, #iter = 42
Objective value = -549.858093
nSV = 3923
...*.
optimization finished, #iter = 40
Objective value = -575.056831
nSV = 3946
....*
optimization finished, #iter = 41
Objective value = -502.006590
nSV = 4009
...*
optimization finished, #iter = 37
Objective value = -389.997187
nSV = 3537
....*
optimization finished, #iter = 41
Objective value = -435.913185
nSV = 3457
....*
optimization finished, #iter = 41
Objective value = -564.376789
nSV = 4327
....*
optimization finished, #iter = 41
Objective value = -479.144006
nSV = 4183
...*
optimization finished, #iter = 34
Objective value = -444.264296
nSV = 3787
...*
optimization finished, #iter = 38
Objective value = -355.379836
nSV = 3500
...*
optimization finished, #iter = 36
Objective value = -388.086088
nSV = 3962
...*
optimization finished, #iter = 39
Objective value = -577.790149
nSV = 4371
...*.
optimization finished, #iter = 40
Objective value = -429.155657
nSV = 4173
...*
optimization finished, #iter = 39
Objective value = -469.985904
nSV = 4253
...*
optimization finished, #iter = 37
Objective value = -480.257843
nSV = 3794
...*
optimization finished, #iter = 34
Objective value = -452.277848
nSV = 3922
...*
optimization finished, #iter = 33
Objective value = -377.777581
nSV = 3805
...*
optimization finished, #iter = 32
Objective value = -461.770782
nSV = 3912
...*
optimization finished, #iter = 35
Objective value = -481.544880
nSV = 3749
[LibLinear]train time: 0.763s
test time:  0.009s
accuracy:   0.697
dimensionality: 101322
density: 0.597634


================================================================================
Classifier.LOGISTIC_REGRESSION
________________________________________________________________________________
Training: 
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=-1, penalty='l2', random_state=0,
                   solver='lbfgs', tol=0.0001, verbose=True, warm_start=False)
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 12 concurrent workers.
[Parallel(n_jobs=-1)]: Done   1 out of   1 | elapsed:   17.4s finished
train time: 17.369s
test time:  0.011s
accuracy:   0.695
dimensionality: 101322
density: 1.000000


================================================================================
Classifier.LOGISTIC_REGRESSION_CV
________________________________________________________________________________
Training: 
LogisticRegressionCV(Cs=10, class_weight=None, cv=None, dual=False,
                     fit_intercept=True, intercept_scaling=1.0, l1_ratios=None,
                     max_iter=100, multi_class='auto', n_jobs=-1, penalty='l2',
                     random_state=0, refit=True, scoring=None, solver='lbfgs',
                     tol=0.0001, verbose=True)
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 12 concurrent workers.
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
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =      2026460     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  3.38937D+04    |proj g|=  1.88700D+02

At iterate   50    f=  1.83877D+04    |proj g|=  5.12826D+00

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
*****     75     83      1     0     0   2.354D-02   1.839D+04
  F =   18387.069000792828     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
 This problem is unconstrained.
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =      2026460     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  2.71144D+04    |proj g|=  1.51550D+02

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
*****     11     16      1     0     0   6.312D-03   2.706D+04
  F =   27061.979765735461     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =      2026460     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  2.70597D+04    |proj g|=  2.50087D+01

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
*****      4      6      1     0     0   1.766D-02   2.704D+04
  F =   27044.578094202214     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =      2026460     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  2.70272D+04    |proj g|=  2.49931D+01

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
*****      5      8      1     0     0   1.175D-03   2.691D+04
  F =   26910.438723225867     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =      2026460     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  2.67768D+04    |proj g|=  2.48714D+01

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
*****     13     14      1     0     0   1.110D-02   2.591D+04
  F =   25909.744985645746     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =      2026460     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  2.49444D+04    |proj g|=  2.37978D+01

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
*****     37     44      1     0     0   2.697D-02   2.048D+04
  F =   20479.827196281800     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =      2026460     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  1.66465D+04    |proj g|=  1.45984D+01

At iterate   50    f=  1.03148D+04    |proj g|=  8.00106D-01

At iterate  100    f=  1.03143D+04    |proj g|=  1.69827D-02

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
*****    100    115      1     0     0   1.698D-02   1.031D+04
  F =   10314.277894777159     

STOP: TOTAL NO. of ITERATIONS REACHED LIMIT                 
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =      2026460     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  6.24746D+03    |proj g|=  3.42242D+00

At iterate   50    f=  3.60891D+03    |proj g|=  1.23409D+00

At iterate  100    f=  3.60738D+03    |proj g|=  2.53719D-01

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
*****    100    109      1     0     0   2.537D-01   3.607D+03
  F =   3607.3795128403785     

STOP: TOTAL NO. of ITERATIONS REACHED LIMIT                 
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =      2026460     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  1.95647D+03    |proj g|=  6.28335D-01

At iterate   50    f=  1.43749D+03    |proj g|=  2.02626D-01

At iterate  100    f=  1.43647D+03    |proj g|=  3.84198D-02

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
*****    100    110      1     0     0   3.842D-02   1.436D+03
  F =   1436.4677947683838     

STOP: TOTAL NO. of ITERATIONS REACHED LIMIT                 
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =      2026460     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  1.00919D+03    |proj g|=  1.08578D-01

At iterate   50    f=  9.15308D+02    |proj g|=  1.20018D-01

At iterate  100    f=  9.14935D+02    |proj g|=  2.24214D-01

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
*****    100    110      1     0     0   2.242D-01   9.149D+02
  F =   914.93506334047333     

STOP: TOTAL NO. of ITERATIONS REACHED LIMIT                 
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =      2026460     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  8.22439D+02    |proj g|=  2.24214D-01

At iterate   50    f=  8.05319D+02    |proj g|=  2.02813D-01

At iterate  100    f=  8.05200D+02    |proj g|=  4.33144D-02

           * * *

Tit   = total number of iterations
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =      2026460     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  2.71144D+04    |proj g|=  1.50550D+02

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
*****     11     16      1     0     0   6.450D-03   2.706D+04
  F =   27062.364153944029     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =      2026460     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  2.70601D+04    |proj g|=  2.56968D+01

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
*****      4      6      1     0     0   1.889D-02   2.704D+04
  F =   27044.889404510053     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =      2026460     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  2.70274D+04    |proj g|=  2.56800D+01

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
*****      5      8      1     0     0   7.868D-04   2.691D+04
  F =   26910.197675883624     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =      2026460     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  2.67760D+04    |proj g|=  2.55481D+01

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
*****     12     14      1     0     0   7.981D-03   2.591D+04
  F =   25906.237852706541     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =      2026460     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  2.49387D+04    |proj g|=  2.43771D+01

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
*****     37     45      1     0     0   9.114D-02   2.049D+04
  F =   20491.550276037480     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =      2026460     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  1.66798D+04    |proj g|=  1.44699D+01

At iterate   50    f=  1.03475D+04    |proj g|=  9.56277D-01

At iterate  100    f=  1.03458D+04    |proj g|=  1.69913D-01

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
*****    100    112      1     0     0   1.699D-01   1.035D+04
  F =   10345.777750541165     

STOP: TOTAL NO. of ITERATIONS REACHED LIMIT                 
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =      2026460     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  6.26527D+03    |proj g|=  3.41282D+00

At iterate   50    f=  3.60014D+03    |proj g|=  1.07099D+00

At iterate  100    f=  3.59719D+03    |proj g|=  2.44382D-01

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
*****    100    109      1     0     0   2.444D-01   3.597D+03
  F =   3597.1915150185046     

STOP: TOTAL NO. of ITERATIONS REACHED LIMIT                 
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =      2026460     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  1.93579D+03    |proj g|=  5.99676D-01

At iterate   50    f=  1.40958D+03    |proj g|=  6.42522D-01

At iterate  100    f=  1.40880D+03    |proj g|=  6.31361D-02

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
*****    100    109      1     0     0   6.314D-02   1.409D+03
  F =   1408.7979199797105     

STOP: TOTAL NO. of ITERATIONS REACHED LIMIT                 
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =      2026460     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  9.77336D+02    |proj g|=  9.95729D-02

At iterate   50    f=  8.83780D+02    |proj g|=  1.62778D-01

At iterate  100    f=  8.83469D+02    |proj g|=  1.12273D-01

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
*****    100    108      1     0     0   1.123D-01   8.835D+02
  F =   883.46861461841820     

STOP: TOTAL NO. of ITERATIONS REACHED LIMIT                 
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =      2026460     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  7.88864D+02    |proj g|=  1.12273D-01

At iterate   50    f=  7.73286D+02    |proj g|=  7.43389D-02

At iterate  100    f=  7.73200D+02    |proj g|=  1.19022D-01

           * * *

Tit   = total number of iterations
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =      2026460     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  2.71174D+04    |proj g|=  1.50600D+02

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
*****     11     16      1     0     0   6.060D-03   2.707D+04
  F =   27065.346310513585     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =      2026460     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  2.70631D+04    |proj g|=  2.64452D+01

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
*****      4      6      1     0     0   1.848D-02   2.705D+04
  F =   27047.996729707149     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =      2026460     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  2.70307D+04    |proj g|=  2.64283D+01

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
*****      5      8      1     0     0   8.580D-04   2.691D+04
  F =   26914.259280754828     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =      2026460     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  2.67811D+04    |proj g|=  2.62947D+01

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
*****     13     15      1     0     0   6.091D-03   2.592D+04
  F =   25916.496834756555     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =      2026460     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  2.49538D+04    |proj g|=  2.51117D+01

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
*****     39     46      1     0     0   1.321D-01   2.049D+04
  F =   20493.570140982949     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =      2026460     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  1.66589D+04    |proj g|=  1.51263D+01

At iterate   50    f=  1.03101D+04    |proj g|=  2.20484D+00

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
*****     96    107      1     0     0   1.287D-02   1.031D+04
  F =   10309.642082408467     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =      2026460     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  6.23375D+03    |proj g|=  3.52371D+00

At iterate   50    f=  3.59522D+03    |proj g|=  3.61064D+00

At iterate  100    f=  3.59373D+03    |proj g|=  5.13986D-01

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
*****    100    108      1     0     0   5.140D-01   3.594D+03
  F =   3593.7303747146134     

STOP: TOTAL NO. of ITERATIONS REACHED LIMIT                 
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =      2026460     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  1.94326D+03    |proj g|=  6.24939D-01

At iterate   50    f=  1.42386D+03    |proj g|=  7.54046D-01

At iterate  100    f=  1.42279D+03    |proj g|=  9.48463D-02

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
*****    100    110      1     0     0   9.485D-02   1.423D+03
  F =   1422.7873388457463     

STOP: TOTAL NO. of ITERATIONS REACHED LIMIT                 
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =      2026460     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  9.95874D+02    |proj g|=  1.10617D-01

At iterate   50    f=  9.02229D+02    |proj g|=  6.18956D-01

At iterate  100    f=  9.01749D+02    |proj g|=  8.69974D-02

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
*****    100    106      1     0     0   8.700D-02   9.017D+02
  F =   901.74947732800888     

STOP: TOTAL NO. of ITERATIONS REACHED LIMIT                 
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =      2026460     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  8.08040D+02    |proj g|=  8.69974D-02

At iterate   50    f=  7.92352D+02    |proj g|=  1.44109D-01

At iterate  100    f=  7.92212D+02    |proj g|=  1.24080D-01

           * * *

Tit   = total number of iterations
Tnf   = total number of function evaluations
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =      2026460     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  2.71144D+04    |proj g|=  1.51550D+02

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
*****     11     16      1     0     0   6.327D-03   2.706D+04
  F =   27061.959428612874     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =      2026460     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  2.70597D+04    |proj g|=  2.61266D+01

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
*****      4      6      1     0     0   1.729D-02   2.704D+04
  F =   27044.654836614725     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =      2026460     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  2.70274D+04    |proj g|=  2.61107D+01

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
*****      5      8      1     0     0   5.740D-04   2.691D+04
  F =   26911.261327978547     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =      2026460     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  2.67784D+04    |proj g|=  2.59861D+01

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
*****     13     15      1     0     0   4.505D-03   2.592D+04
  F =   25915.971651482316     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =      2026460     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  2.49557D+04    |proj g|=  2.48783D+01

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
*****     48     56      1     0     0   8.375D-02   2.051D+04
  F =   20509.610690910231     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =      2026460     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  1.66889D+04    |proj g|=  1.51744D+01

At iterate   50    f=  1.03439D+04    |proj g|=  1.95008D+00

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
*****     93    104      1     0     0   2.488D-02   1.034D+04
  F =   10343.094202846625     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =      2026460     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  6.26058D+03    |proj g|=  3.49887D+00

At iterate   50    f=  3.60005D+03    |proj g|=  6.25206D-01

At iterate  100    f=  3.59842D+03    |proj g|=  9.83137D-02

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
*****    100    109      1     0     0   9.831D-02   3.598D+03
  F =   3598.4225407341869     

STOP: TOTAL NO. of ITERATIONS REACHED LIMIT                 
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =      2026460     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  1.93600D+03    |proj g|=  6.05422D-01

At iterate   50    f=  1.41137D+03    |proj g|=  4.93223D-01

At iterate  100    f=  1.41025D+03    |proj g|=  6.30431D-02

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
*****    100    109      1     0     0   6.304D-02   1.410D+03
  F =   1410.2468119459797     

STOP: TOTAL NO. of ITERATIONS REACHED LIMIT                 
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =      2026460     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  9.78462D+02    |proj g|=  1.25267D-01

At iterate   50    f=  8.84010D+02    |proj g|=  2.70666D-01

At iterate  100    f=  8.83642D+02    |proj g|=  5.87035D-02

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
*****    100    107      1     0     0   5.870D-02   8.836D+02
  F =   883.64160353298428     

STOP: TOTAL NO. of ITERATIONS REACHED LIMIT                 
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =      2026460     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  7.89651D+02    |proj g|=  5.87035D-02

At iterate   50    f=  7.72862D+02    |proj g|=  5.61870D-02

At iterate  100    f=  7.72732D+02    |proj g|=  2.64859D-02

           * * *

Tit   = total number of iterations
Tnf   = total number of function evaluations
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =      2026460     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  2.71144D+04    |proj g|=  1.50550D+02

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
*****     11     16      1     0     0   6.118D-03   2.706D+04
  F =   27062.270194324530     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =      2026460     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  2.70600D+04    |proj g|=  2.65870D+01

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
*****      4      6      1     0     0   1.815D-02   2.704D+04
  F =   27044.781218782427     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =      2026460     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  2.70273D+04    |proj g|=  2.65703D+01

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
*****      5      8      1     0     0   2.018D-03   2.691D+04
  F =   26909.971762406345     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =      2026460     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  2.67757D+04    |proj g|=  2.64386D+01

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
*****     13     15      1     0     0   4.328D-03   2.590D+04
  F =   25904.489789040828     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =      2026460     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  2.49347D+04    |proj g|=  2.52682D+01

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
*****     41     48      1     0     0   2.082D-02   2.046D+04
  F =   20458.640484125572     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =      2026460     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  1.66180D+04    |proj g|=  1.52293D+01

At iterate   50    f=  1.02811D+04    |proj g|=  5.01204D+00

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
*****     99    112      1     0     0   2.212D-02   1.028D+04
  F =   10280.306545051284     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =      2026460     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  6.20546D+03    |proj g|=  3.47573D+00

At iterate   50    f=  3.55604D+03    |proj g|=  2.50710D-01

At iterate  100    f=  3.55469D+03    |proj g|=  2.35738D-01

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
*****    100    110      1     0     0   2.357D-01   3.555D+03
  F =   3554.6893464441055     

STOP: TOTAL NO. of ITERATIONS REACHED LIMIT                 
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =      2026460     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  1.89694D+03    |proj g|=  6.00203D-01

At iterate   50    f=  1.37602D+03    |proj g|=  5.66640D-01

At iterate  100    f=  1.37493D+03    |proj g|=  3.03996D-01

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
*****    100    111      1     0     0   3.040D-01   1.375D+03
  F =   1374.9284216741785     

STOP: TOTAL NO. of ITERATIONS REACHED LIMIT                 
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =      2026460     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  9.45566D+02    |proj g|=  3.03996D-01

At iterate   50    f=  8.51976D+02    |proj g|=  4.43678D-01

At iterate  100    f=  8.51499D+02    |proj g|=  1.03362D-01

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
*****    100    106      1     0     0   1.034D-01   8.515D+02
  F =   851.49897335316393     

STOP: TOTAL NO. of ITERATIONS REACHED LIMIT                 
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =      2026460     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  7.57390D+02    |proj g|=  1.03362D-01

At iterate   50    f=  7.41678D+02    |proj g|=  1.35024D-02

At iterate  100    f=  7.41544D+02    |proj g|=  2.05265D-02

           * * *

Tit   = total number of iterations
Tnf   = total number of function evaluations
[Parallel(n_jobs=-1)]: Done   5 out of   5 | elapsed:  6.3min finished
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
train time: 409.508s
test time:  0.011s
accuracy:   0.694
dimensionality: 101322
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
              tol=0.0001, validation_fraction=0.1, verbose=True,
              warm_start=False)
Iteration 1, loss = 2.84244666
Iteration 2, loss = 2.12518662
Iteration 3, loss = 1.32335457
Iteration 4, loss = 0.80687302
Iteration 5, loss = 0.53090363
Iteration 6, loss = 0.37710010
Iteration 7, loss = 0.28623473
Iteration 8, loss = 0.22935593
Iteration 9, loss = 0.19214325
Iteration 10, loss = 0.16724415
Iteration 11, loss = 0.15002446
Iteration 12, loss = 0.13759579
Iteration 13, loss = 0.12864837
Iteration 14, loss = 0.12176185
Iteration 15, loss = 0.11650552
Iteration 16, loss = 0.11238933
Iteration 17, loss = 0.10915325
Iteration 18, loss = 0.10653320
Iteration 19, loss = 0.10449348
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
*****    100    109      1     0     0   4.331D-02   8.052D+02
  F =   805.19967799517826     

STOP: TOTAL NO. of ITERATIONS REACHED LIMIT                 
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
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
*****    100    106      1     0     0   1.190D-01   7.732D+02
  F =   773.19981726877722     

STOP: TOTAL NO. of ITERATIONS REACHED LIMIT                 
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
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
*****    100    108      1     0     0   1.241D-01   7.922D+02
  F =   792.21208046803110     

STOP: TOTAL NO. of ITERATIONS REACHED LIMIT                 
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
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
*****    100    110      1     0     0   2.649D-02   7.727D+02
  F =   772.73181761734236     

STOP: TOTAL NO. of ITERATIONS REACHED LIMIT                 
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
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
*****    100    110      1     0     0   2.053D-02   7.415D+02
  F =   741.54418989951205     

STOP: TOTAL NO. of ITERATIONS REACHED LIMIT                 
Iteration 20, loss = 0.10285421
Iteration 21, loss = 0.10127518
Iteration 22, loss = 0.10010704
Iteration 23, loss = 0.09889485
Iteration 24, loss = 0.09794122
Iteration 25, loss = 0.09714883
Iteration 26, loss = 0.09663372
Iteration 27, loss = 0.09592383
Iteration 28, loss = 0.09551061
Iteration 29, loss = 0.09511400
Iteration 30, loss = 0.09462799
Iteration 31, loss = 0.09425381
Iteration 32, loss = 0.09387927
Iteration 33, loss = 0.09368113
Iteration 34, loss = 0.09334295
Iteration 35, loss = 0.09321180
Iteration 36, loss = 0.09292905
Iteration 37, loss = 0.09251190
Iteration 38, loss = 0.09259804
Iteration 39, loss = 0.09234462
Iteration 40, loss = 0.09215921
Iteration 41, loss = 0.09212406
Iteration 42, loss = 0.09183027
Iteration 43, loss = 0.09175370
Iteration 44, loss = 0.09164405
Iteration 45, loss = 0.09152643
Iteration 46, loss = 0.09125197
Iteration 47, loss = 0.09129978
Iteration 48, loss = 0.09137112
Iteration 49, loss = 0.09132581
Iteration 50, loss = 0.09099389
Iteration 51, loss = 0.09115296
Iteration 52, loss = 0.09096964
Iteration 53, loss = 0.09074447
Iteration 54, loss = 0.09098808
Iteration 55, loss = 0.09084899
Iteration 56, loss = 0.09069683
Iteration 57, loss = 0.09043249
Iteration 58, loss = 0.09072294
Iteration 59, loss = 0.09082055
Iteration 60, loss = 0.09062080
Iteration 61, loss = 0.09049093
Iteration 62, loss = 0.09060166
Iteration 63, loss = 0.09038990
Iteration 64, loss = 0.09048436
Iteration 65, loss = 0.09046448
Iteration 66, loss = 0.09025488
Iteration 67, loss = 0.09048134
Iteration 68, loss = 0.09038925
Iteration 69, loss = 0.09016626
Iteration 70, loss = 0.09033531
Iteration 71, loss = 0.09009271
Iteration 72, loss = 0.09010525
Iteration 73, loss = 0.09014305
Iteration 74, loss = 0.09007413
Iteration 75, loss = 0.08996051
Iteration 76, loss = 0.09015024
Iteration 77, loss = 0.08989490
Iteration 78, loss = 0.09000421
Iteration 79, loss = 0.08977875
Iteration 80, loss = 0.08971717
Iteration 81, loss = 0.08963376
Iteration 82, loss = 0.09002753
Iteration 83, loss = 0.08997788
Iteration 84, loss = 0.09001769
Iteration 85, loss = 0.08955409
Iteration 86, loss = 0.08990907
Iteration 87, loss = 0.08992850
Iteration 88, loss = 0.08966552
Iteration 89, loss = 0.08944874
Iteration 90, loss = 0.08973240
Iteration 91, loss = 0.08968808
Iteration 92, loss = 0.08970180
Iteration 93, loss = 0.08958844
Iteration 94, loss = 0.08941281
Iteration 95, loss = 0.08935778
Iteration 96, loss = 0.08998019
Iteration 97, loss = 0.08973046
Iteration 98, loss = 0.08969582
Iteration 99, loss = 0.08960015
Iteration 100, loss = 0.08951443
Training loss did not improve more than tol=0.000100 for 10 consecutive epochs. Stopping.
train time: 1357.282s
test time:  0.040s
accuracy:   0.700

================================================================================
Classifier.MULTINOMIAL_NB
________________________________________________________________________________
Training: 
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
train time: 0.083s
test time:  0.010s
accuracy:   0.671
dimensionality: 101322
density: 1.000000


================================================================================
Classifier.NEAREST_CENTROID
________________________________________________________________________________
Training: 
NearestCentroid(metric='euclidean', shrink_threshold=None)
train time: 0.016s
test time:  0.013s
accuracy:   0.643

================================================================================
Classifier.NU_SVC
________________________________________________________________________________
Training: 
NuSVC(break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
      max_iter=-1, nu=0.5, probability=False, random_state=0, shrinking=True,
      tol=0.001, verbose=True)
.*
optimization finished, #iter = 1372
C = 1.088316
obj = 274.070411, rho = -0.472773
nSV = 963, nBSV = 81
.*
optimization finished, #iter = 1235
C = 0.946119
obj = 236.532738, rho = -0.636959
nSV = 913, nBSV = 129
.*
optimization finished, #iter = 1356
C = 0.970558
obj = 244.329233, rho = -0.047993
nSV = 918, nBSV = 105
.*
optimization finished, #iter = 1307
C = 1.013951
obj = 252.968181, rho = -0.579032
nSV = 947, nBSV = 98
.*
optimization finished, #iter = 1331
C = 0.943223
obj = 245.362171, rho = 0.504048
nSV = 940, nBSV = 104
.*
optimization finished, #iter = 1287
C = 0.948776
obj = 239.745448, rho = 0.551232
nSV = 930, nBSV = 121
.*
optimization finished, #iter = 1471
C = 1.195590
obj = 303.249617, rho = -0.385983
nSV = 995, nBSV = 68
.*
optimization finished, #iter = 1448
C = 1.232366
obj = 314.444217, rho = -0.273601
nSV = 1014, nBSV = 70
.*
optimization finished, #iter = 1410
C = 1.112782
obj = 282.909328, rho = -0.429376
nSV = 970, nBSV = 83
.*
optimization finished, #iter = 1297
C = 0.996435
obj = 250.341386, rho = -0.592516
nSV = 926, nBSV = 139
.*
optimization finished, #iter = 1320
C = 1.111472
obj = 281.260920, rho = -0.303849
nSV = 929, nBSV = 90
.*
optimization finished, #iter = 1449
C = 1.162805
obj = 294.916047, rho = -0.329556
nSV = 1020, nBSV = 68
.*
optimization finished, #iter = 1458
C = 1.244251
obj = 318.077124, rho = -0.395367
nSV = 999, nBSV = 48
.*
optimization finished, #iter = 1462
C = 1.234674
obj = 314.910438, rho = -0.336883
nSV = 988, nBSV = 51
.*
optimization finished, #iter = 1556
C = 1.475829
obj = 382.131106, rho = 0.429190
nSV = 1029, nBSV = 42
.*
optimization finished, #iter = 1445
C = 1.323148
obj = 323.008200, rho = -0.521188
nSV = 976, nBSV = 36
.*
optimization finished, #iter = 1398
C = 1.252845
obj = 308.869823, rho = -0.572861
nSV = 963, nBSV = 66
.*
optimization finished, #iter = 1321
C = 1.493741
obj = 334.509917, rho = -0.337210
nSV = 916, nBSV = 26
.*
optimization finished, #iter = 1277
C = 1.848981
obj = 373.727181, rho = -0.686254
nSV = 847, nBSV = 29
.*
optimization finished, #iter = 1592
C = 1.386478
obj = 378.140291, rho = -0.747571
nSV = 1107, nBSV = 69
.*
optimization finished, #iter = 1544
C = 1.334827
obj = 369.352989, rho = 0.374804
nSV = 1073, nBSV = 81
.*
optimization finished, #iter = 1456
C = 1.253250
obj = 339.306242, rho = -0.655073
nSV = 1074, nBSV = 83
.*
optimization finished, #iter = 1520
C = 1.332711
obj = 382.605293, rho = 0.408722
nSV = 1088, nBSV = 65
.*
optimization finished, #iter = 1433
C = 1.097687
obj = 303.755335, rho = 0.463805
nSV = 1052, nBSV = 100
.*
optimization finished, #iter = 1398
C = 1.124674
obj = 309.358387, rho = -0.468624
nSV = 1057, nBSV = 85
.*
optimization finished, #iter = 1512
C = 1.137304
obj = 315.089708, rho = 0.381656
nSV = 1083, nBSV = 75
.*
optimization finished, #iter = 1353
C = 1.030340
obj = 283.404151, rho = -0.500825
nSV = 1041, nBSV = 116
.*
optimization finished, #iter = 1218
C = 0.896238
obj = 240.464314, rho = -0.085485
nSV = 1010, nBSV = 184
.*
optimization finished, #iter = 1285
C = 1.020379
obj = 276.510726, rho = 0.333608
nSV = 1013, nBSV = 146
.*
optimization finished, #iter = 1643
C = 1.376644
obj = 381.245285, rho = -0.216048
nSV = 1124, nBSV = 46
.*
optimization finished, #iter = 1508
C = 1.142040
obj = 314.088775, rho = 0.038137
nSV = 1071, nBSV = 77
.*
optimization finished, #iter = 1523
C = 1.210480
obj = 333.712676, rho = 0.077002
nSV = 1080, nBSV = 72
.*
optimization finished, #iter = 1191
C = 0.901795
obj = 248.831123, rho = 0.313537
nSV = 993, nBSV = 162
.*
optimization finished, #iter = 1368
C = 1.038432
obj = 275.343485, rho = 0.434867
nSV = 1005, nBSV = 101
.*
optimization finished, #iter = 1280
C = 0.953047
obj = 250.658908, rho = -0.286791
nSV = 987, nBSV = 158
.*
optimization finished, #iter = 1454
C = 1.117162
obj = 276.265148, rho = 0.462940
nSV = 958, nBSV = 68
.*
optimization finished, #iter = 1305
C = 1.135439
obj = 251.197987, rho = -0.121435
nSV = 883, nBSV = 85
.*
optimization finished, #iter = 1537
C = 1.362011
obj = 376.387868, rho = 0.595487
nSV = 1075, nBSV = 96
.*
optimization finished, #iter = 1404
C = 1.183404
obj = 314.009155, rho = 0.632313
nSV = 1040, nBSV = 117
.*
optimization finished, #iter = 1508
C = 1.186228
obj = 337.621394, rho = 0.613526
nSV = 1064, nBSV = 97
.*
optimization finished, #iter = 1264
C = 0.977769
obj = 267.710315, rho = 0.646732
nSV = 1013, nBSV = 164
.*
optimization finished, #iter = 1378
C = 1.034023
obj = 272.115955, rho = -0.282675
nSV = 1039, nBSV = 157
.*
optimization finished, #iter = 1397
C = 0.990677
obj = 271.415657, rho = 0.921422
nSV = 1041, nBSV = 140
.*
optimization finished, #iter = 1307
C = 0.925632
obj = 244.433810, rho = 0.356177
nSV = 1006, nBSV = 170
.*
optimization finished, #iter = 1182
C = 0.786289
obj = 208.631785, rho = 0.591921
nSV = 966, nBSV = 246
.*
optimization finished, #iter = 1202
C = 0.894470
obj = 240.860013, rho = 0.528522
nSV = 956, nBSV = 202
.*
optimization finished, #iter = 1473
C = 1.168538
obj = 319.048644, rho = 0.897946
nSV = 1080, nBSV = 118
.*
optimization finished, #iter = 1377
C = 0.974256
obj = 265.169536, rho = 0.795669
nSV = 1021, nBSV = 151
.*
optimization finished, #iter = 1386
C = 1.013296
obj = 275.607448, rho = 0.863103
nSV = 1031, nBSV = 137
.*
optimization finished, #iter = 1130
C = 0.773944
obj = 210.050420, rho = 0.508488
nSV = 938, nBSV = 239
.*
optimization finished, #iter = 1254
C = 0.910296
obj = 239.797685, rho = 0.666425
nSV = 958, nBSV = 148
.*
optimization finished, #iter = 1185
C = 0.836253
obj = 217.204572, rho = 0.603863
nSV = 941, nBSV = 220
.*
optimization finished, #iter = 1262
C = 0.986188
obj = 243.401855, rho = 0.629368
nSV = 906, nBSV = 112
.*
optimization finished, #iter = 1209
C = 1.002409
obj = 222.166223, rho = 0.503615
nSV = 828, nBSV = 115
.*
optimization finished, #iter = 1605
C = 1.440215
obj = 401.216742, rho = -0.510219
nSV = 1089, nBSV = 46
.*
optimization finished, #iter = 1443
C = 1.045935
obj = 298.755743, rho = 0.576202
nSV = 1037, nBSV = 119
.*
optimization finished, #iter = 1424
C = 1.165355
obj = 320.610384, rho = 0.592004
nSV = 1073, nBSV = 112
.*
optimization finished, #iter = 1436
C = 1.049948
obj = 292.086985, rho = -0.302818
nSV = 1034, nBSV = 111
.*
optimization finished, #iter = 1508
C = 1.031344
obj = 288.395342, rho = -0.193408
nSV = 1048, nBSV = 106
.*
optimization finished, #iter = 1331
C = 0.913555
obj = 253.912209, rho = -0.343942
nSV = 999, nBSV = 139
.*
optimization finished, #iter = 1260
C = 0.810480
obj = 219.875090, rho = -0.524091
nSV = 968, nBSV = 212
.*
optimization finished, #iter = 1257
C = 0.919340
obj = 250.187266, rho = -0.225700
nSV = 983, nBSV = 183
.*
optimization finished, #iter = 1555
C = 1.330258
obj = 370.490230, rho = -0.222265
nSV = 1100, nBSV = 73
.*
optimization finished, #iter = 1402
C = 0.999024
obj = 277.565919, rho = -0.314188
nSV = 1018, nBSV = 118
.*
optimization finished, #iter = 1386
C = 1.029320
obj = 286.309402, rho = -0.254654
nSV = 1034, nBSV = 106
.*
optimization finished, #iter = 1208
C = 0.806959
obj = 221.895449, rho = 0.472573
nSV = 959, nBSV = 199
.*
optimization finished, #iter = 1287
C = 0.935800
obj = 249.376339, rho = -0.357075
nSV = 979, nBSV = 132
.*
optimization finished, #iter = 1261
C = 0.842865
obj = 223.995666, rho = -0.508499
nSV = 950, nBSV = 195
.*
optimization finished, #iter = 1366
C = 1.012862
obj = 251.545570, rho = -0.187504
nSV = 931, nBSV = 88
.*
optimization finished, #iter = 1255
C = 1.012932
obj = 228.801889, rho = -0.593695
nSV = 846, nBSV = 100
.*
optimization finished, #iter = 1349
C = 1.014026
obj = 286.889882, rho = 0.551390
nSV = 1023, nBSV = 135
.*
optimization finished, #iter = 1403
C = 1.138170
obj = 313.624557, rho = 0.574994
nSV = 1043, nBSV = 97
.*
optimization finished, #iter = 1436
C = 1.136241
obj = 303.547899, rho = -0.342025
nSV = 1055, nBSV = 103
.*
optimization finished, #iter = 1494
C = 1.096983
obj = 300.269700, rho = 0.855546
nSV = 1061, nBSV = 104
.*
optimization finished, #iter = 1352
C = 0.984649
obj = 262.537387, rho = -0.382267
nSV = 1026, nBSV = 142
.*
optimization finished, #iter = 1212
C = 0.846591
obj = 224.730245, rho = 0.522458
nSV = 986, nBSV = 202
.*
optimization finished, #iter = 1281
C = 0.950040
obj = 256.196187, rho = 0.480786
nSV = 986, nBSV = 174
.*
optimization finished, #iter = 1503
C = 1.350430
obj = 368.365209, rho = 0.836714
nSV = 1113, nBSV = 76
.*
optimization finished, #iter = 1398
C = 1.057293
obj = 288.044833, rho = 0.730771
nSV = 1039, nBSV = 113
.*
optimization finished, #iter = 1362
C = 1.092775
obj = 297.726085, rho = 0.796896
nSV = 1046, nBSV = 109
.*
optimization finished, #iter = 1175
C = 0.830700
obj = 227.211788, rho = 0.437169
nSV = 972, nBSV = 185
.*
optimization finished, #iter = 1248
C = 0.977004
obj = 256.667755, rho = 0.604241
nSV = 990, nBSV = 123
.*
optimization finished, #iter = 1208
C = 0.895843
obj = 233.388073, rho = 0.542681
nSV = 964, nBSV = 179
.*
optimization finished, #iter = 1365
C = 1.063893
obj = 261.236562, rho = 0.574156
nSV = 942, nBSV = 77
.*
optimization finished, #iter = 1247
C = 1.071115
obj = 235.498303, rho = 0.457900
nSV = 853, nBSV = 101
.*
optimization finished, #iter = 1314
C = 0.908529
obj = 255.036491, rho = -0.502215
nSV = 1020, nBSV = 149
.*
optimization finished, #iter = 1341
C = 0.954238
obj = 272.460636, rho = -0.348060
nSV = 1027, nBSV = 131
.*
optimization finished, #iter = 1481
C = 0.964328
obj = 277.567195, rho = -0.240414
nSV = 1050, nBSV = 116
.*
optimization finished, #iter = 1318
C = 0.865132
obj = 245.959429, rho = -0.386125
nSV = 1003, nBSV = 155
.*
optimization finished, #iter = 1191
C = 0.755148
obj = 209.147625, rho = -0.568504
nSV = 969, nBSV = 223
.*
optimization finished, #iter = 1273
C = 0.885582
obj = 247.789367, rho = -0.625266
nSV = 985, nBSV = 182
.*
optimization finished, #iter = 1624
C = 1.122141
obj = 324.160661, rho = -0.266661
nSV = 1083, nBSV = 74
.*
optimization finished, #iter = 1384
C = 0.960606
obj = 274.835761, rho = -0.351625
nSV = 1033, nBSV = 118
.*
optimization finished, #iter = 1507
C = 0.995459
obj = 285.249731, rho = -0.295112
nSV = 1039, nBSV = 113
.*
optimization finished, #iter = 1200
C = 0.787840
obj = 220.114589, rho = -0.646613
nSV = 964, nBSV = 206
.*
optimization finished, #iter = 1307
C = 0.890607
obj = 244.967589, rho = -0.475172
nSV = 981, nBSV = 132
.*
optimization finished, #iter = 1220
C = 0.809422
obj = 220.954889, rho = -0.546182
nSV = 952, nBSV = 198
.*
optimization finished, #iter = 1350
C = 0.959128
obj = 246.956072, rho = -0.510751
nSV = 934, nBSV = 96
.*
optimization finished, #iter = 1237
C = 0.964573
obj = 225.135772, rho = -0.641175
nSV = 845, nBSV = 104
.*
optimization finished, #iter = 1383
C = 1.102063
obj = 307.075136, rho = -0.403157
nSV = 1041, nBSV = 115
.*
optimization finished, #iter = 1446
C = 1.091461
obj = 306.444404, rho = -0.294587
nSV = 1052, nBSV = 97
.*
optimization finished, #iter = 1310
C = 0.958993
obj = 268.894848, rho = -0.430174
nSV = 1013, nBSV = 131
.*
optimization finished, #iter = 1249
C = 0.859819
obj = 236.652752, rho = -0.598681
nSV = 992, nBSV = 187
.*
optimization finished, #iter = 1153
C = 0.877255
obj = 238.855176, rho = -0.651896
nSV = 965, nBSV = 211
.*
optimization finished, #iter = 1540
C = 1.256876
obj = 351.197976, rho = -0.311911
nSV = 1090, nBSV = 77
.*
optimization finished, #iter = 1360
C = 0.983590
obj = 274.564092, rho = -0.408888
nSV = 1020, nBSV = 127
.*
optimization finished, #iter = 1418
C = 1.050185
obj = 293.881944, rho = -0.350336
nSV = 1044, nBSV = 111
.*
optimization finished, #iter = 1167
C = 0.805340
obj = 219.528851, rho = -0.279073
nSV = 950, nBSV = 216
.*
optimization finished, #iter = 1317
C = 0.952405
obj = 255.214335, rho = -0.527739
nSV = 975, nBSV = 125
.*
optimization finished, #iter = 1138
C = 0.852797
obj = 227.963097, rho = -0.594658
nSV = 955, nBSV = 186
.*
optimization finished, #iter = 1389
C = 1.030974
obj = 257.769603, rho = -0.571814
nSV = 939, nBSV = 95
.*
optimization finished, #iter = 1213
C = 0.996634
obj = 225.721332, rho = -0.681913
nSV = 846, nBSV = 109
.*
optimization finished, #iter = 1693
C = 1.466984
obj = 411.905729, rho = 0.665699
nSV = 1133, nBSV = 45
.*
optimization finished, #iter = 1411
C = 1.159555
obj = 314.546498, rho = 0.515854
nSV = 1081, nBSV = 95
.*
optimization finished, #iter = 1293
C = 0.982939
obj = 266.902976, rho = 0.320415
nSV = 1019, nBSV = 157
.*
optimization finished, #iter = 1241
C = 1.027346
obj = 283.556282, rho = 0.281623
nSV = 1007, nBSV = 151
.*
optimization finished, #iter = 1592
C = 1.365912
obj = 378.601969, rho = 0.619847
nSV = 1130, nBSV = 62
.*
optimization finished, #iter = 1545
C = 1.203112
obj = 336.767857, rho = 0.537906
nSV = 1076, nBSV = 62
.*
optimization finished, #iter = 1580
C = 1.258470
obj = 351.625935, rho = 0.595174
nSV = 1088, nBSV = 57
.*
optimization finished, #iter = 1257
C = 0.963224
obj = 269.523162, rho = 0.232766
nSV = 1004, nBSV = 153
.*
optimization finished, #iter = 1407
C = 1.202468
obj = 322.868204, rho = 0.407592
nSV = 1039, nBSV = 73
.*
optimization finished, #iter = 1298
C = 1.057064
obj = 281.643042, rho = 0.339523
nSV = 1009, nBSV = 133
.*
optimization finished, #iter = 1490
C = 1.268879
obj = 317.891714, rho = 0.381173
nSV = 985, nBSV = 51
.*
optimization finished, #iter = 1345
C = 1.243382
obj = 278.665692, rho = 0.269013
nSV = 899, nBSV = 89
.*
optimization finished, #iter = 1520
C = 1.169541
obj = 329.580596, rho = -0.696557
nSV = 1082, nBSV = 77
.*
optimization finished, #iter = 1435
C = 1.012601
obj = 277.250540, rho = -0.446961
nSV = 1054, nBSV = 140
.*
optimization finished, #iter = 1435
C = 1.041351
obj = 287.777963, rho = 0.172629
nSV = 1030, nBSV = 143
.*
optimization finished, #iter = 1764
C = 1.331284
obj = 375.376495, rho = -0.595662
nSV = 1143, nBSV = 41
.*
optimization finished, #iter = 1637
C = 1.236099
obj = 348.602404, rho = -0.363818
nSV = 1094, nBSV = 52
.*
optimization finished, #iter = 1576
C = 1.285073
obj = 362.349036, rho = -0.334536
nSV = 1100, nBSV = 46
.*
optimization finished, #iter = 1351
C = 0.975115
obj = 273.588031, rho = 0.124236
nSV = 1030, nBSV = 147
.*
optimization finished, #iter = 1493
C = 1.229253
obj = 332.229948, rho = 0.114847
nSV = 1061, nBSV = 68
.*
optimization finished, #iter = 1425
C = 1.073689
obj = 288.507059, rho = -0.691156
nSV = 1031, nBSV = 121
.*
optimization finished, #iter = 1456
C = 1.293053
obj = 326.177690, rho = 0.269051
nSV = 1003, nBSV = 44
.*
optimization finished, #iter = 1389
C = 1.255916
obj = 282.520815, rho = -0.534779
nSV = 918, nBSV = 90
.*
optimization finished, #iter = 1577
C = 1.259192
obj = 352.625732, rho = 0.388589
nSV = 1116, nBSV = 75
.*
optimization finished, #iter = 1223
C = 0.917140
obj = 251.401278, rho = 0.323002
nSV = 979, nBSV = 177
.*
optimization finished, #iter = 1576
C = 1.134482
obj = 315.532751, rho = 0.636817
nSV = 1100, nBSV = 84
.*
optimization finished, #iter = 1523
C = 1.095957
obj = 307.053861, rho = 0.568158
nSV = 1060, nBSV = 87
.*
optimization finished, #iter = 1512
C = 1.140871
obj = 319.326715, rho = 0.627212
nSV = 1064, nBSV = 79
.*
optimization finished, #iter = 1316
C = 0.900885
obj = 250.698122, rho = 0.264271
nSV = 993, nBSV = 172
.*
optimization finished, #iter = 1373
C = 1.089829
obj = 292.935165, rho = 0.440719
nSV = 1011, nBSV = 92
.*
optimization finished, #iter = 1381
C = 0.997321
obj = 265.000262, rho = 0.376527
nSV = 989, nBSV = 154
.*
optimization finished, #iter = 1447
C = 1.160999
obj = 291.427378, rho = 0.411200
nSV = 970, nBSV = 58
.*
optimization finished, #iter = 1304
C = 1.158304
obj = 259.983447, rho = 0.306993
nSV = 892, nBSV = 87
.*
optimization finished, #iter = 1142
C = 0.819753
obj = 220.249027, rho = 0.501850
nSV = 945, nBSV = 244
.*
optimization finished, #iter = 1427
C = 0.981050
obj = 266.330854, rho = -0.046221
nSV = 1059, nBSV = 143
.*
optimization finished, #iter = 1406
C = 0.944216
obj = 257.458539, rho = 0.135659
nSV = 1021, nBSV = 153
.*
optimization finished, #iter = 1332
C = 0.990293
obj = 270.014026, rho = 0.173644
nSV = 1033, nBSV = 150
.*
optimization finished, #iter = 1132
C = 0.794860
obj = 217.318201, rho = 0.460420
nSV = 953, nBSV = 239
.*
optimization finished, #iter = 1301
C = 0.969846
obj = 257.149169, rho = 0.487190
nSV = 981, nBSV = 149
.*
optimization finished, #iter = 1186
C = 0.883297
obj = 230.995795, rho = -0.193646
nSV = 952, nBSV = 213
.*
optimization finished, #iter = 1381
C = 1.061497
obj = 264.837856, rho = 0.577127
nSV = 930, nBSV = 105
.*
optimization finished, #iter = 1216
C = 1.052068
obj = 235.606109, rho = -0.062803
nSV = 839, nBSV = 116
.*
optimization finished, #iter = 1433
C = 1.131741
obj = 312.176579, rho = -0.196098
nSV = 1066, nBSV = 128
.*
optimization finished, #iter = 1364
C = 1.040293
obj = 287.334779, rho = -0.286772
nSV = 1007, nBSV = 137
.*
optimization finished, #iter = 1401
C = 1.093338
obj = 302.491058, rho = -0.230454
nSV = 1025, nBSV = 121
.*
optimization finished, #iter = 1174
C = 0.882466
obj = 244.417522, rho = 0.499572
nSV = 954, nBSV = 198
.*
optimization finished, #iter = 1387
C = 1.174602
obj = 314.752723, rho = -0.180192
nSV = 990, nBSV = 102
.*
optimization finished, #iter = 1213
C = 0.984393
obj = 263.734413, rho = -0.485430
nSV = 955, nBSV = 176
.*
optimization finished, #iter = 1364
C = 1.272037
obj = 318.330815, rho = 0.056370
nSV = 931, nBSV = 55
.*
optimization finished, #iter = 1258
C = 1.154948
obj = 263.062973, rho = -0.601125
nSV = 833, nBSV = 75
.*
optimization finished, #iter = 1679
C = 1.275560
obj = 355.746198, rho = 0.235431
nSV = 1116, nBSV = 50
.*
optimization finished, #iter = 1700
C = 1.352497
obj = 377.731473, rho = 0.293262
nSV = 1125, nBSV = 44
.*
optimization finished, #iter = 1426
C = 0.956195
obj = 266.195261, rho = 0.166826
nSV = 1038, nBSV = 132
.*
optimization finished, #iter = 1550
C = 1.174655
obj = 315.298593, rho = 0.344269
nSV = 1068, nBSV = 71
.*
optimization finished, #iter = 1387
C = 1.035810
obj = 274.559159, rho = -0.165983
nSV = 1036, nBSV = 135
.*
optimization finished, #iter = 1561
C = 1.237768
obj = 309.708066, rho = 0.315082
nSV = 1018, nBSV = 51
.*
optimization finished, #iter = 1378
C = 1.213041
obj = 270.126174, rho = 0.062677
nSV = 935, nBSV = 88
.*
optimization finished, #iter = 1665
C = 1.266898
obj = 355.399102, rho = 0.042808
nSV = 1087, nBSV = 46
.*
optimization finished, #iter = 1465
C = 1.035967
obj = 292.144529, rho = 0.228227
nSV = 1022, nBSV = 119
.*
optimization finished, #iter = 1530
C = 1.178097
obj = 317.549550, rho = 0.412524
nSV = 1039, nBSV = 65
.*
optimization finished, #iter = 1469
C = 1.075907
obj = 288.268988, rho = -0.370911
nSV = 1015, nBSV = 113
.*
optimization finished, #iter = 1520
C = 1.306323
obj = 327.954005, rho = 0.383141
nSV = 988, nBSV = 37
.*
optimization finished, #iter = 1397
C = 1.280745
obj = 288.107267, rho = -0.178828
nSV = 910, nBSV = 65
.*
optimization finished, #iter = 1382
C = 1.006282
obj = 282.154661, rho = 0.180622
nSV = 1023, nBSV = 139
.*
optimization finished, #iter = 1489
C = 1.215773
obj = 327.950059, rho = 0.361052
nSV = 1046, nBSV = 55
.*
optimization finished, #iter = 1386
C = 1.093064
obj = 292.741154, rho = -0.407298
nSV = 1022, nBSV = 113
.*
optimization finished, #iter = 1539
C = 1.332109
obj = 334.561979, rho = 0.325624
nSV = 995, nBSV = 40
.*
optimization finished, #iter = 1420
C = 1.267725
obj = 283.773169, rho = -0.208305
nSV = 905, nBSV = 82
.*
optimization finished, #iter = 1329
C = 1.042645
obj = 283.016652, rho = -0.377791
nSV = 1008, nBSV = 115
.*
optimization finished, #iter = 1250
C = 1.018444
obj = 278.102335, rho = -0.455293
nSV = 996, nBSV = 155
.*
optimization finished, #iter = 1371
C = 1.186580
obj = 302.721036, rho = -0.418785
nSV = 956, nBSV = 60
.*
optimization finished, #iter = 1561
C = 1.769879
obj = 412.871730, rho = -0.591718
nSV = 961, nBSV = 38
.*
optimization finished, #iter = 1443
C = 1.191157
obj = 311.998821, rho = -0.611154
nSV = 1016, nBSV = 61
.*
optimization finished, #iter = 1441
C = 1.527767
obj = 365.640663, rho = 0.329822
nSV = 983, nBSV = 31
.*
optimization finished, #iter = 1362
C = 1.439616
obj = 309.483974, rho = -0.717488
nSV = 884, nBSV = 53
.*
optimization finished, #iter = 1449
C = 1.329337
obj = 323.353597, rho = 0.565783
nSV = 955, nBSV = 46
.*
optimization finished, #iter = 1258
C = 1.320009
obj = 287.373781, rho = 0.236756
nSV = 867, nBSV = 54
.*
optimization finished, #iter = 1269
C = 1.539570
obj = 304.009156, rho = -0.677169
nSV = 824, nBSV = 29
Total nSV = 10983
[LibSVM]train time: 82.380s
test time:  27.448s
accuracy:   0.692

================================================================================
Classifier.PASSIVE_AGGRESSIVE_CLASSIFIER
________________________________________________________________________________
Training: 
PassiveAggressiveClassifier(C=1.0, average=False, class_weight=None,
                            early_stopping=False, fit_intercept=True,
                            loss='hinge', max_iter=1000, n_iter_no_change=5,
                            n_jobs=-1, random_state=0, shuffle=True, tol=0.001,
                            validation_fraction=0.1, verbose=True,
                            warm_start=False)
-- Epoch 1
-- Epoch 1
-- Epoch 1
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 12 concurrent workers.
-- Epoch 1
-- Epoch 1
-- Epoch 1
-- Epoch 1
-- Epoch 1
Norm: 27.33, NNZs: 41670, Bias: -0.996441, T: 10996, Avg. loss: 0.118214
Total training time: 0.04 seconds.
-- Epoch 1
-- Epoch 1
Norm: 27.36, NNZs: 30510, Bias: -0.971647, T: 10996, Avg. loss: 0.100438
Total training time: 0.03 seconds.
-- Epoch 2
-- Epoch 2
-- Epoch 1
Norm: 34.45, NNZs: 32082, Bias: -0.996701, T: 21992, Avg. loss: 0.037589
Total training time: 0.06 seconds.
-- Epoch 3
Norm: 28.93, NNZs: 42532, Bias: -0.944554, T: 10996, Avg. loss: 0.116244
Norm: 36.65, NNZs: 52830, Bias: -1.052024, T: 21992, Avg. loss: 0.051521
Total training time: 0.04 seconds.
Total training time: 0.07 seconds.
-- Epoch 3
Norm: 37.67, NNZs: 32458, Bias: -0.981784, T: 32988, Avg. loss: 0.018198
Total training time: 0.07 seconds.
-- Epoch 4
-- Epoch 1
Norm: 27.98, NNZs: 47267, Bias: -0.945293, T: 10996, Avg. loss: 0.110202
Total training time: 0.04 seconds.
-- Epoch 2
-- Epoch 2
Norm: 26.75, NNZs: 35573, Bias: -0.864553, T: 10996, Avg. loss: 0.078703
Total training time: 0.04 seconds.
-- Epoch 2
Norm: 35.29, NNZs: 51765, Bias: -0.972246, T: 21992, Avg. loss: 0.041207
Total training time: 0.05 seconds.
-- Epoch 3
Norm: 27.10, NNZs: 39027, Bias: -0.911128, T: 10996, Avg. loss: 0.096573
Total training time: 0.08 seconds.
Norm: 28.67, NNZs: 37327, Bias: -0.860799, T: 10996, Avg. loss: 0.105703
Total training time: 0.06 seconds.
-- Epoch 2
-- Epoch 2
Norm: 41.39, NNZs: 53367, Bias: -1.105282, T: 32988, Avg. loss: 0.025847
Total training time: 0.09 seconds.
Norm: 36.75, NNZs: 45362, Bias: -0.962913, T: 21992, Avg. loss: 0.043456
Total training time: 0.05 seconds.
-- Epoch 4
-- Epoch 3
Norm: 27.15, NNZs: 54071, Bias: -0.968311, T: 10996, Avg. loss: 0.107565
Total training time: 0.08 seconds.
-- Epoch 2
Norm: 35.46, NNZs: 38706, Bias: -0.829593, T: 21992, Avg. loss: 0.036197
Total training time: 0.06 seconds.
-- Epoch 3
Norm: 38.54, NNZs: 52126, Bias: -0.966423, T: 32988, Avg. loss: 0.020547
Total training time: 0.06 seconds.
-- Epoch 4
Norm: 34.92, NNZs: 56241, Bias: -1.059370, T: 21992, Avg. loss: 0.042508
Total training time: 0.08 seconds.
-- Epoch 3
Norm: 37.91, NNZs: 38839, Bias: -0.863413, T: 32988, Avg. loss: 0.016839
Total training time: 0.06 seconds.
-- Epoch 4
Norm: 26.60, NNZs: 35242, Bias: -0.979819, T: 10996, Avg. loss: 0.104018
Total training time: 0.11 seconds.
-- Epoch 2
Norm: 27.71, NNZs: 40967, Bias: -0.945913, T: 10996, Avg. loss: 0.094605
Total training time: 0.08 seconds.
-- Epoch 2
Norm: 33.17, NNZs: 41175, Bias: -0.982790, T: 21992, Avg. loss: 0.032841
Total training time: 0.10 seconds.
Norm: 43.88, NNZs: 53493, Bias: -1.120161, T: 43984, Avg. loss: 0.014355
Total training time: 0.10 seconds.
Norm: 33.31, NNZs: 42607, Bias: -0.922222, T: 21992, Avg. loss: 0.027806
Total training time: 0.08 seconds.
-- Epoch 5
-- Epoch 3
Norm: 34.76, NNZs: 36969, Bias: -0.931064, T: 21992, Avg. loss: 0.038250
Total training time: 0.11 seconds.
-- Epoch 3
Norm: 35.82, NNZs: 42959, Bias: -0.999113, T: 32988, Avg. loss: 0.012464
Total training time: 0.09 seconds.
-- Epoch 4
Norm: 26.26, NNZs: 37796, Bias: -0.987034, T: 10996, Avg. loss: 0.083580
Total training time: 0.06 seconds.
-- Epoch 2
Norm: 36.01, NNZs: 41547, Bias: -1.029927, T: 32988, Avg. loss: 0.015509
Total training time: 0.11 seconds.
-- Epoch 4
Norm: 38.82, NNZs: 38911, Bias: -0.866989, T: 43984, Avg. loss: 0.010081
Total training time: 0.08 seconds.
-- Epoch 5
Norm: 45.31, NNZs: 53534, Bias: -1.119522, T: 54980, Avg. loss: 0.009042
Total training time: 0.12 seconds.
-- Epoch 6
Norm: 32.00, NNZs: 43802, Bias: -1.035085, T: 21992, Avg. loss: 0.025991
Total training time: 0.06 seconds.
-- Epoch 3
Norm: 39.20, NNZs: 38944, Bias: -0.874059, T: 54980, Avg. loss: 0.007689
Total training time: 0.08 seconds.
-- Epoch 6
Norm: 40.00, NNZs: 52256, Bias: -1.022210, T: 43984, Avg. loss: 0.012806
Total training time: 0.08 seconds.
-- Epoch 5
Norm: 30.73, NNZs: 36527, Bias: -0.846601, T: 21992, Avg. loss: 0.021717
Total training time: 0.08 seconds.
-- Epoch 3
Norm: 39.38, NNZs: 38949, Bias: -0.881696, T: 65976, Avg. loss: 0.007090
Total training time: 0.09 seconds.
Norm: 40.28, NNZs: 45680, Bias: -0.974167, T: 32988, Avg. loss: 0.021032
Total training time: 0.09 seconds.
-- Epoch 4
Norm: 32.37, NNZs: 36716, Bias: -0.891051, T: 32988, Avg. loss: 0.012079
Total training time: 0.08 seconds.
-- Epoch 4
Norm: 28.61, NNZs: 39455, Bias: -0.918639, T: 10996, Avg. loss: 0.095312
Total training time: 0.06 seconds.
Norm: 37.46, NNZs: 41700, Bias: -1.000761, T: 43984, Avg. loss: 0.008263
Total training time: 0.12 seconds.
-- Epoch 2
-- Epoch 5
Norm: 37.06, NNZs: 43021, Bias: -0.986109, T: 43984, Avg. loss: 0.005770
Total training time: 0.11 seconds.
-- Epoch 5
Norm: 33.52, NNZs: 40317, Bias: -0.855815, T: 21992, Avg. loss: 0.026779
Total training time: 0.06 seconds.
Norm: 39.22, NNZs: 32544, Bias: -1.025596, T: 43984, Avg. loss: 0.011191
Total training time: 0.12 seconds.
-- Epoch 3
Norm: 45.99, NNZs: 53562, Bias: -1.133079, T: 65976, Avg. loss: 0.005927
Total training time: 0.13 seconds.
-- Epoch 7
-- Epoch 7
Norm: 38.74, NNZs: 56731, Bias: -1.074171, T: 32988, Avg. loss: 0.022156
Total training time: 0.12 seconds.
Norm: 38.00, NNZs: 37359, Bias: -0.982682, T: 32988, Avg. loss: 0.016356
Total training time: 0.14 seconds.
-- Epoch 4
-- Epoch 5
Norm: 40.78, NNZs: 52285, Bias: -1.037063, T: 54980, Avg. loss: 0.009661
Total training time: 0.10 seconds.
Norm: 39.50, NNZs: 38962, Bias: -0.886838, T: 76972, Avg. loss: 0.006766
Total training time: 0.10 seconds.
-- Epoch 8
-- Epoch 6
Norm: 40.06, NNZs: 32560, Bias: -1.034341, T: 54980, Avg. loss: 0.007649
Total training time: 0.13 seconds.
-- Epoch 6
-- Epoch 3
Norm: 37.75, NNZs: 43051, Bias: -0.996714, T: 54980, Avg. loss: 0.003331
Total training time: 0.12 seconds.
-- Epoch 6
Norm: 42.02, NNZs: 45734, Bias: -0.999995, T: 43984, Avg. loss: 0.012321
Total training time: 0.11 seconds.
-- Epoch 5
Norm: 33.20, NNZs: 36779, Bias: -0.915502, T: 43984, Avg. loss: 0.008200
Total training time: 0.10 seconds.
-- Epoch 5
-- Epoch 4
Norm: 41.31, NNZs: 52308, Bias: -1.045878, T: 65976, Avg. loss: 0.008693
Total training time: 0.11 seconds.
-- Epoch 7
Norm: 43.00, NNZs: 45780, Bias: -1.014681, T: 54980, Avg. loss: 0.008695
Total training time: 0.11 seconds.
-- Epoch 6
Norm: 46.37, NNZs: 53568, Bias: -1.135664, T: 76972, Avg. loss: 0.004442
Total training time: 0.15 seconds.
Norm: 38.16, NNZs: 41757, Bias: -1.036440, T: 54980, Avg. loss: 0.005898
Total training time: 0.15 seconds.
-- Epoch 8
-- Epoch 6
Norm: 34.12, NNZs: 44102, Bias: -1.043412, T: 32988, Avg. loss: 0.010677
Total training time: 0.10 seconds.
Norm: 40.56, NNZs: 32607, Bias: -1.042492, T: 65976, Avg. loss: 0.006562
Total training time: 0.14 seconds.
-- Epoch 4
-- Epoch 7
Norm: 35.16, NNZs: 40457, Bias: -0.901474, T: 32988, Avg. loss: 0.013694
Total training time: 0.08 seconds.
-- Epoch 4
Norm: 39.58, NNZs: 38962, Bias: -0.883633, T: 87968, Avg. loss: 0.006427
Total training time: 0.12 seconds.
Norm: 38.14, NNZs: 43560, Bias: -1.004014, T: 65976, Avg. loss: 0.002121
Total training time: 0.13 seconds.
-- Epoch 9
-- Epoch 7
Norm: 41.62, NNZs: 52308, Bias: -1.046209, T: 76972, Avg. loss: 0.007631
Total training time: 0.12 seconds.
Norm: 39.33, NNZs: 37401, Bias: -1.012542, T: 43984, Avg. loss: 0.007545
Total training time: 0.16 seconds.
-- Epoch 8
-- Epoch 5
Norm: 40.49, NNZs: 56842, Bias: -1.093641, T: 43984, Avg. loss: 0.013139
Total training time: 0.14 seconds.
-- Epoch 5
Norm: 33.57, NNZs: 36793, Bias: -0.937984, T: 54980, Avg. loss: 0.006465
Total training time: 0.11 seconds.
-- Epoch 6
Norm: 43.55, NNZs: 45786, Bias: -1.022045, T: 65976, Avg. loss: 0.006684
Total training time: 0.12 seconds.
-- Epoch 7
Norm: 38.63, NNZs: 41772, Bias: -1.061381, T: 65976, Avg. loss: 0.004916
Total training time: 0.15 seconds.
Norm: 40.86, NNZs: 32607, Bias: -1.046354, T: 76972, Avg. loss: 0.005477
Total training time: 0.15 seconds.
-- Epoch 7
Norm: 35.09, NNZs: 44190, Bias: -1.054817, T: 43984, Avg. loss: 0.006230
Total training time: 0.10 seconds.
-- Epoch 8
Norm: 35.89, NNZs: 40470, Bias: -0.914350, T: 43984, Avg. loss: 0.009441
Total training time: 0.09 seconds.
-- Epoch 5
-- Epoch 5
Norm: 38.35, NNZs: 43560, Bias: -1.005212, T: 76972, Avg. loss: 0.001367
Total training time: 0.14 seconds.
-- Epoch 8
Norm: 46.62, NNZs: 53586, Bias: -1.138906, T: 87968, Avg. loss: 0.003941
Total training time: 0.16 seconds.
Norm: 39.63, NNZs: 38963, Bias: -0.878959, T: 98964, Avg. loss: 0.006305
Total training time: 0.12 seconds.
-- Epoch 10
-- Epoch 9
Norm: 41.88, NNZs: 52312, Bias: -1.044910, T: 87968, Avg. loss: 0.007460
Total training time: 0.12 seconds.
Norm: 39.92, NNZs: 37421, Bias: -1.010441, T: 54980, Avg. loss: 0.004349
Total training time: 0.17 seconds.
-- Epoch 9
-- Epoch 6
Norm: 41.40, NNZs: 56877, Bias: -1.103380, T: 54980, Avg. loss: 0.009364
Total training time: 0.15 seconds.
-- Epoch 6
Norm: 33.93, NNZs: 36815, Bias: -0.939778, T: 65976, Avg. loss: 0.005731
Total training time: 0.12 seconds.
-- Epoch 7
Norm: 43.92, NNZs: 45816, Bias: -1.043742, T: 76972, Avg. loss: 0.006051
Total training time: 0.13 seconds.
Norm: 41.07, NNZs: 32621, Bias: -1.065743, T: 87968, Avg. loss: 0.005315
Total training time: 0.15 seconds.
-- Epoch 8
-- Epoch 9
Norm: 38.94, NNZs: 41808, Bias: -1.055977, T: 76972, Avg. loss: 0.004011
Total training time: 0.16 seconds.
-- Epoch 8
Norm: 36.30, NNZs: 40528, Bias: -0.915388, T: 54980, Avg. loss: 0.007779
Total training time: 0.09 seconds.
-- Epoch 6
Norm: 35.72, NNZs: 44233, Bias: -1.093285, T: 54980, Avg. loss: 0.005107
Total training time: 0.11 seconds.
-- Epoch 6
Norm: 38.52, NNZs: 43564, Bias: -1.009174, T: 87968, Avg. loss: 0.001244
Total training time: 0.15 seconds.
-- Epoch 9
Norm: 39.67, NNZs: 38963, Bias: -0.890426, T: 109960, Avg. loss: 0.006545
Total training time: 0.13 seconds.
Convergence after 10 epochs took 0.13 seconds
Norm: 46.81, NNZs: 53589, Bias: -1.143558, T: 98964, Avg. loss: 0.003626
Total training time: 0.17 seconds.
Norm: 40.19, NNZs: 37435, Bias: -1.015551, T: 65976, Avg. loss: 0.003171
-- Epoch 10
Norm: 42.06, NNZs: 52317, Bias: -1.050813, T: 98964, Avg. loss: 0.007031
Total training time: 0.14 seconds.
-- Epoch 10
Total training time: 0.18 seconds.
-- Epoch 1
-- Epoch 7
Norm: 34.15, NNZs: 36824, Bias: -0.945715, T: 76972, Avg. loss: 0.005202
Total training time: 0.13 seconds.
-- Epoch 8
Norm: 38.64, NNZs: 43578, Bias: -1.009710, T: 98964, Avg. loss: 0.001022
Total training time: 0.16 seconds.
-- Epoch 10
Norm: 36.15, NNZs: 44262, Bias: -1.114435, T: 65976, Avg. loss: 0.004213
Total training time: 0.12 seconds.
-- Epoch 7
Norm: 42.20, NNZs: 52334, Bias: -1.061770, T: 109960, Avg. loss: 0.007064
Total training time: 0.14 seconds.
-- Epoch 11
Norm: 39.15, NNZs: 41819, Bias: -1.056372, T: 87968, Avg. loss: 0.003607
Total training time: 0.18 seconds.
Norm: 36.55, NNZs: 40528, Bias: -0.924656, T: 65976, Avg. loss: 0.007098
Total training time: 0.11 seconds.
-- Epoch 7
-- Epoch 9
Norm: 38.70, NNZs: 43579, Bias: -1.010933, T: 109960, Avg. loss: 0.000861
Total training time: 0.16 seconds.
-- Epoch 11
Norm: 42.30, NNZs: 52334, Bias: -1.064803, T: 120956, Avg. loss: 0.006748
Total training time: 0.15 seconds.
-- Epoch 12
Norm: 36.66, NNZs: 40535, Bias: -0.930511, T: 76972, Avg. loss: 0.006537
Total training time: 0.11 seconds.
-- Epoch 8
Norm: 46.93, NNZs: 53591, Bias: -1.146738, T: 109960, Avg. loss: 0.003354
Total training time: 0.19 seconds.
-- Epoch 11
Norm: 34.36, NNZs: 36825, Bias: -0.951973, T: 87968, Avg. loss: 0.004892
Total training time: 0.14 seconds.
-- Epoch 9
Norm: 44.17, NNZs: 45820, Bias: -1.052912, T: 87968, Avg. loss: 0.005500
Total training time: 0.15 seconds.
-- Epoch 9
Norm: 38.78, NNZs: 43579, Bias: -1.003835, T: 120956, Avg. loss: 0.000751
Total training time: 0.17 seconds.
Convergence after 11 epochs took 0.17 seconds
Norm: 42.42, NNZs: 52334, Bias: -1.048243, T: 131952, Avg. loss: 0.006624
Total training time: 0.15 seconds.
-- Epoch 1
Norm: 39.35, NNZs: 41844, Bias: -1.077577, T: 98964, Avg. loss: 0.003794
Total training time: 0.19 seconds.
Norm: 42.00, NNZs: 56883, Bias: -1.114242, T: 65976, Avg. loss: 0.008034
Total training time: 0.18 seconds.
-- Epoch 7
-- Epoch 10
Convergence after 12 epochs took 0.16 seconds
Norm: 34.51, NNZs: 36831, Bias: -0.959750, T: 98964, Avg. loss: 0.004624
Total training time: 0.15 seconds.
-- Epoch 10
Norm: 47.05, NNZs: 53593, Bias: -1.140085, T: 120956, Avg. loss: 0.003236
Total training time: 0.20 seconds.
-- Epoch 12
Norm: 44.37, NNZs: 45836, Bias: -1.043107, T: 98964, Avg. loss: 0.005034
Total training time: 0.17 seconds.
Norm: 42.29, NNZs: 56897, Bias: -1.126083, T: 76972, Avg. loss: 0.006903
Total training time: 0.19 seconds.
-- Epoch 8
Norm: 29.43, NNZs: 45137, Bias: -0.946504, T: 10996, Avg. loss: 0.122827
Total training time: 0.03 seconds.
Norm: 40.34, NNZs: 37437, Bias: -1.022036, T: 76972, Avg. loss: 0.002753
Total training time: 0.21 seconds.
Norm: 36.74, NNZs: 40536, Bias: -0.930452, T: 87968, Avg. loss: 0.006308
Total training time: 0.13 seconds.
-- Epoch 1
-- Epoch 2
-- Epoch 10
Norm: 36.46, NNZs: 44268, Bias: -1.110203, T: 76972, Avg. loss: 0.003499
Total training time: 0.15 seconds.
Norm: 34.68, NNZs: 36842, Bias: -0.961370, T: 109960, Avg. loss: 0.004403
Norm: 41.29, NNZs: 32634, Bias: -1.065789, T: 98964, Avg. loss: 0.005235
Total training time: 0.19 seconds.
-- Epoch 8
Norm: 28.72, NNZs: 38221, Bias: -0.915622, T: 10996, Avg. loss: 0.098110
Total training time: 0.02 seconds.
Norm: 42.50, NNZs: 56906, Bias: -1.128036, T: 87968, Avg. loss: 0.006460
Total training time: 0.19 seconds.
-- Epoch 2
Norm: 39.60, NNZs: 41859, Bias: -1.060330, T: 109960, Avg. loss: 0.003533
-- Epoch 9
Norm: 47.14, NNZs: 53593, Bias: -1.145202, T: 131952, Avg. loss: 0.003234
Total training time: 0.21 seconds.
-- Epoch 8
Total training time: 0.21 seconds.
Total training time: 0.17 seconds.
-- Epoch 10
Convergence after 10 epochs took 0.21 seconds
Convergence after 12 epochs took 0.22 seconds
-- Epoch 1
Norm: 28.72, NNZs: 46202, Bias: -0.881742, T: 10996, Avg. loss: 0.102494
Total training time: 0.02 seconds.
Norm: 41.39, NNZs: 32634, Bias: -1.064913, T: 109960, Avg. loss: 0.004842
Total training time: 0.21 seconds.
-- Epoch 11
Norm: 36.60, NNZs: 44274, Bias: -1.119858, T: 87968, Avg. loss: 0.003003
Convergence after 10 epochs took 0.19 seconds
Norm: 44.56, NNZs: 45839, Bias: -1.037010, T: 109960, Avg. loss: 0.004940
-- Epoch 1
Total training time: 0.17 seconds.
-- Epoch 2
-- Epoch 9
-- Epoch 1
Norm: 27.56, NNZs: 37978, Bias: -0.973259, T: 10996, Avg. loss: 0.098420
Total training time: 0.01 seconds.
-- Epoch 2
-- Epoch 9
Norm: 34.13, NNZs: 39028, Bias: -0.936495, T: 21992, Avg. loss: 0.026581
Total training time: 0.05 seconds.
-- Epoch 3
Total training time: 0.20 seconds.
-- Epoch 11
Norm: 36.77, NNZs: 44274, Bias: -1.120161, T: 98964, Avg. loss: 0.003010
Total training time: 0.18 seconds.
-- Epoch 10
Norm: 37.65, NNZs: 49600, Bias: -0.971086, T: 21992, Avg. loss: 0.045019
Total training time: 0.06 seconds.
-- Epoch 3
Norm: 40.43, NNZs: 37437, Bias: -1.021431, T: 87968, Avg. loss: 0.002460
Total training time: 0.25 seconds.
Norm: 41.53, NNZs: 32639, Bias: -1.062785, T: 120956, Avg. loss: 0.004890
Total training time: 0.23 seconds.
Norm: 42.63, NNZs: 56989, Bias: -1.125076, T: 98964, Avg. loss: 0.006171
Total training time: 0.23 seconds.
-- Epoch 9
-- Epoch 12
Norm: 27.46, NNZs: 40392, Bias: -1.012909, T: 10996, Avg. loss: 0.106254
Total training time: 0.02 seconds.
-- Epoch 2
Norm: 36.84, NNZs: 44285, Bias: -1.124316, T: 109960, Avg. loss: 0.002739
Total training time: 0.19 seconds.
Convergence after 10 epochs took 0.19 seconds
Norm: 36.79, NNZs: 40538, Bias: -0.935699, T: 98964, Avg. loss: 0.006229
Total training time: 0.17 seconds.
-- Epoch 10
Norm: 34.92, NNZs: 47519, Bias: -0.902524, T: 21992, Avg. loss: 0.032652
Total training time: 0.04 seconds.
-- Epoch 3
-- Epoch 10
-- Epoch 1
Norm: 44.68, NNZs: 45845, Bias: -1.049750, T: 120956, Avg. loss: 0.004950
Norm: 35.53, NNZs: 39401, Bias: -0.963450, T: 32988, Avg. loss: 0.009716
Total training time: 0.22 seconds.
Convergence after 11 epochs took 0.22 seconds
Total training time: 0.06 seconds.
-- Epoch 4
-- Epoch 1
Norm: 41.61, NNZs: 32652, Bias: -1.065702, T: 131952, Avg. loss: 0.004716
Total training time: 0.25 seconds.
Convergence after 12 epochs took 0.25 seconds
Norm: 36.06, NNZs: 39440, Bias: -0.973411, T: 43984, Avg. loss: 0.005818
Total training time: 0.07 seconds.
-- Epoch 5
Norm: 25.54, NNZs: 38208, Bias: -0.921021, T: 10996, Avg. loss: 0.083113
Total training time: 0.03 seconds.
-- Epoch 2
Norm: 42.74, NNZs: 56989, Bias: -1.127447, T: 109960, Avg. loss: 0.006150
Total training time: 0.24 seconds.
Norm: 34.39, NNZs: 56102, Bias: -0.912973, T: 21992, Avg. loss: 0.032713
Total training time: 0.03 seconds.
-- Epoch 3
Norm: 36.83, NNZs: 40538, Bias: -0.932338, T: 109960, Avg. loss: 0.006098
Total training time: 0.19 seconds.
Convergence after 10 epochs took 0.19 seconds
Norm: 37.20, NNZs: 47686, Bias: -0.957685, T: 32988, Avg. loss: 0.014636
Total training time: 0.06 seconds.
-- Epoch 4
-- Epoch 11
Norm: 26.98, NNZs: 39864, Bias: -1.009594, T: 10996, Avg. loss: 0.102557
Total training time: 0.01 seconds.
-- Epoch 2
Norm: 36.36, NNZs: 39453, Bias: -0.981086, T: 54980, Avg. loss: 0.004736
Total training time: 0.07 seconds.
-- Epoch 6
Norm: 42.84, NNZs: 56989, Bias: -1.136768, T: 120956, Avg. loss: 0.006215
Total training time: 0.25 seconds.
-- Epoch 12
Norm: 40.48, NNZs: 37442, Bias: -1.022867, T: 98964, Avg. loss: 0.002327
Total training time: 0.28 seconds.
-- Epoch 10
Norm: 36.53, NNZs: 39454, Bias: -0.994749, T: 65976, Avg. loss: 0.004130
Total training time: 0.08 seconds.
-- Epoch 7
Norm: 41.21, NNZs: 50014, Bias: -0.999486, T: 32988, Avg. loss: 0.019831
Total training time: 0.10 seconds.
Norm: 22.84, NNZs: 31695, Bias: -1.008150, T: 10996, Avg. loss: 0.098000
Total training time: 0.01 seconds.
-- Epoch 2
-- Epoch 4
Norm: 36.70, NNZs: 39455, Bias: -0.996003, T: 76972, Avg. loss: 0.003775
Total training time: 0.08 seconds.
-- Epoch 8
Norm: 37.07, NNZs: 56441, Bias: -0.994015, T: 32988, Avg. loss: 0.014084
Total training time: 0.05 seconds.
Norm: 35.55, NNZs: 41785, Bias: -1.005315, T: 21992, Avg. loss: 0.038326
Total training time: 0.06 seconds.
-- Epoch 4
-- Epoch 3
Norm: 38.10, NNZs: 47712, Bias: -0.957558, T: 43984, Avg. loss: 0.007838
Total training time: 0.07 seconds.
-- Epoch 5
Norm: 34.81, NNZs: 41712, Bias: -1.013167, T: 21992, Avg. loss: 0.034666
Total training time: 0.03 seconds.
-- Epoch 3
Norm: 36.78, NNZs: 39460, Bias: -0.993621, T: 87968, Avg. loss: 0.003443
Total training time: 0.09 seconds.
-- Epoch 9
Norm: 38.07, NNZs: 56460, Bias: -1.001582, T: 43984, Avg. loss: 0.006144
Total training time: 0.05 seconds.
-- Epoch 5
Norm: 31.08, NNZs: 39892, Bias: -0.930870, T: 21992, Avg. loss: 0.026026
Total training time: 0.05 seconds.
-- Epoch 3
Norm: 36.90, NNZs: 39460, Bias: -1.006542, T: 98964, Avg. loss: 0.003615
Total training time: 0.09 seconds.
-- Epoch 10
Norm: 38.41, NNZs: 56480, Bias: -1.011259, T: 54980, Avg. loss: 0.003748
Total training time: 0.06 seconds.
-- Epoch 6
Norm: 37.40, NNZs: 41927, Bias: -1.068447, T: 32988, Avg. loss: 0.012759
Total training time: 0.04 seconds.
-- Epoch 4
Norm: 36.98, NNZs: 39460, Bias: -1.008093, T: 109960, Avg. loss: 0.003371
Total training time: 0.10 seconds.
Convergence after 10 epochs took 0.10 seconds
Norm: 42.66, NNZs: 50095, Bias: -1.005911, T: 43984, Avg. loss: 0.009847
Total training time: 0.12 seconds.
-- Epoch 5
Norm: 33.23, NNZs: 34443, Bias: -0.941688, T: 21992, Avg. loss: 0.045533
Total training time: 0.04 seconds.
-- Epoch 3
Norm: 38.58, NNZs: 47726, Bias: -0.967438, T: 54980, Avg. loss: 0.006115
Total training time: 0.09 seconds.
-- Epoch 6
Norm: 42.91, NNZs: 56989, Bias: -1.133451, T: 131952, Avg. loss: 0.005932
Total training time: 0.28 seconds.
Convergence after 12 epochs took 0.28 seconds
Norm: 39.02, NNZs: 42157, Bias: -1.045235, T: 32988, Avg. loss: 0.016241
Total training time: 0.08 seconds.
-- Epoch 4
Norm: 37.22, NNZs: 34755, Bias: -1.035706, T: 32988, Avg. loss: 0.019892
Total training time: 0.04 seconds.
-- Epoch 4
Norm: 39.02, NNZs: 34785, Bias: -1.042791, T: 43984, Avg. loss: 0.009886
Total training time: 0.05 seconds.
-- Epoch 5
Norm: 40.51, NNZs: 37442, Bias: -1.022131, T: 109960, Avg. loss: 0.002271
Total training time: 0.31 seconds.
-- Epoch 11
Norm: 38.53, NNZs: 56480, Bias: -1.009681, T: 65976, Avg. loss: 0.002890
Total training time: 0.08 seconds.
-- Epoch 7
Norm: 39.77, NNZs: 34808, Bias: -1.055185, T: 54980, Avg. loss: 0.005834
Total training time: 0.05 seconds.
-- Epoch 6
Norm: 38.75, NNZs: 47727, Bias: -0.968563, T: 65976, Avg. loss: 0.004680
Total training time: 0.10 seconds.
Norm: 38.57, NNZs: 56480, Bias: -1.010861, T: 76972, Avg. loss: 0.002633
Total training time: 0.08 seconds.
Norm: 38.39, NNZs: 41987, Bias: -1.063037, T: 43984, Avg. loss: 0.005884
Total training time: 0.06 seconds.
-- Epoch 8
-- Epoch 5
-- Epoch 7
Norm: 40.15, NNZs: 34818, Bias: -1.055212, T: 65976, Avg. loss: 0.004354
Total training time: 0.06 seconds.
-- Epoch 7
Norm: 38.59, NNZs: 56480, Bias: -1.010823, T: 87968, Avg. loss: 0.002546
Total training time: 0.08 seconds.
-- Epoch 9
Norm: 38.75, NNZs: 42002, Bias: -1.079209, T: 54980, Avg. loss: 0.003805
Total training time: 0.06 seconds.
-- Epoch 6
Norm: 43.32, NNZs: 50136, Bias: -1.028639, T: 54980, Avg. loss: 0.006636
Total training time: 0.14 seconds.
-- Epoch 6
Norm: 39.03, NNZs: 47753, Bias: -0.967529, T: 76972, Avg. loss: 0.004936
Total training time: 0.11 seconds.
-- Epoch 8
Norm: 40.36, NNZs: 34822, Bias: -1.054305, T: 76972, Avg. loss: 0.003673
Total training time: 0.06 seconds.
-- Epoch 8
Norm: 38.59, NNZs: 56480, Bias: -1.010849, T: 98964, Avg. loss: 0.002511
Total training time: 0.09 seconds.
-- Epoch 10
Norm: 38.89, NNZs: 42018, Bias: -1.082491, T: 65976, Avg. loss: 0.002917
Total training time: 0.07 seconds.
Norm: 32.95, NNZs: 40103, Bias: -0.950144, T: 32988, Avg. loss: 0.011158
Total training time: 0.09 seconds.
-- Epoch 7
-- Epoch 4
Norm: 40.46, NNZs: 34822, Bias: -1.060721, T: 87968, Avg. loss: 0.003355
Total training time: 0.06 seconds.
-- Epoch 9
Norm: 40.53, NNZs: 37442, Bias: -1.023271, T: 120956, Avg. loss: 0.002198
Total training time: 0.33 seconds.
Convergence after 11 epochs took 0.33 seconds
Norm: 38.60, NNZs: 56480, Bias: -1.011030, T: 109960, Avg. loss: 0.002497
Total training time: 0.09 seconds.
Convergence after 10 epochs took 0.09 seconds
Norm: 39.22, NNZs: 47762, Bias: -0.966761, T: 87968, Avg. loss: 0.004408
Total training time: 0.12 seconds.
Norm: 38.96, NNZs: 42023, Bias: -1.084573, T: 76972, Avg. loss: 0.002648
Total training time: 0.07 seconds.
-- Epoch 9
-- Epoch 8
Norm: 40.53, NNZs: 34822, Bias: -1.063238, T: 98964, Avg. loss: 0.003207
Total training time: 0.07 seconds.
-- Epoch 10
Norm: 33.71, NNZs: 40160, Bias: -0.961092, T: 43984, Avg. loss: 0.006633
Total training time: 0.10 seconds.
-- Epoch 5
Norm: 40.54, NNZs: 42309, Bias: -1.052708, T: 43984, Avg. loss: 0.007680
Total training time: 0.11 seconds.
-- Epoch 5
Norm: 39.40, NNZs: 47766, Bias: -0.967375, T: 98964, Avg. loss: 0.004243
Total training time: 0.12 seconds.
Norm: 38.99, NNZs: 42029, Bias: -1.085608, T: 87968, Avg. loss: 0.002526
Total training time: 0.08 seconds.
-- Epoch 10
-- Epoch 9
Norm: 40.60, NNZs: 34822, Bias: -1.061915, T: 109960, Avg. loss: 0.003141
Total training time: 0.07 seconds.
-- Epoch 11
Norm: 43.65, NNZs: 50149, Bias: -1.018594, T: 65976, Avg. loss: 0.005030
Total training time: 0.16 seconds.
-- Epoch 7
Norm: 33.98, NNZs: 40168, Bias: -0.962043, T: 54980, Avg. loss: 0.004846
Total training time: 0.10 seconds.
-- Epoch 6
Norm: 40.63, NNZs: 34823, Bias: -1.061689, T: 120956, Avg. loss: 0.003028
Total training time: 0.08 seconds.
Convergence after 11 epochs took 0.08 seconds
Norm: 41.12, NNZs: 42333, Bias: -1.063488, T: 54980, Avg. loss: 0.004041
Total training time: 0.11 seconds.
Norm: 39.01, NNZs: 42030, Bias: -1.085472, T: 98964, Avg. loss: 0.002447
Total training time: 0.08 seconds.
-- Epoch 6
-- Epoch 10
Norm: 43.82, NNZs: 50155, Bias: -1.024616, T: 76972, Avg. loss: 0.004546
Total training time: 0.16 seconds.
Norm: 34.12, NNZs: 40177, Bias: -0.966186, T: 65976, Avg. loss: 0.004340
Total training time: 0.11 seconds.
-- Epoch 8
Norm: 41.41, NNZs: 42347, Bias: -1.075976, T: 65976, Avg. loss: 0.002904
Total training time: 0.12 seconds.
-- Epoch 7
-- Epoch 7
Norm: 39.02, NNZs: 42030, Bias: -1.085763, T: 109960, Avg. loss: 0.002437
Total training time: 0.09 seconds.
Convergence after 10 epochs took 0.09 seconds
Norm: 39.59, NNZs: 47818, Bias: -0.977464, T: 109960, Avg. loss: 0.004254
Total training time: 0.14 seconds.
-- Epoch 11
Norm: 34.17, NNZs: 40188, Bias: -0.965856, T: 76972, Avg. loss: 0.004007
Total training time: 0.11 seconds.
-- Epoch 8
Norm: 43.93, NNZs: 50156, Bias: -1.024806, T: 87968, Avg. loss: 0.004213
Total training time: 0.17 seconds.
-- Epoch 9
Norm: 41.59, NNZs: 42347, Bias: -1.071882, T: 76972, Avg. loss: 0.002317
Total training time: 0.12 seconds.
-- Epoch 8
Norm: 39.67, NNZs: 47825, Bias: -0.973882, T: 120956, Avg. loss: 0.003670
Total training time: 0.14 seconds.
Convergence after 11 epochs took 0.14 seconds
Norm: 34.19, NNZs: 40188, Bias: -0.967874, T: 87968, Avg. loss: 0.003908
Total training time: 0.12 seconds.
-- Epoch 9
Norm: 44.04, NNZs: 50159, Bias: -1.036672, T: 98964, Avg. loss: 0.004354
Total training time: 0.17 seconds.
-- Epoch 10
Norm: 41.72, NNZs: 42350, Bias: -1.084393, T: 87968, Avg. loss: 0.002285
Total training time: 0.13 seconds.
-- Epoch 9
Norm: 34.21, NNZs: 40188, Bias: -0.968206, T: 98964, Avg. loss: 0.003850
Total training time: 0.12 seconds.
-- Epoch 10
Norm: 44.14, NNZs: 50159, Bias: -1.035881, T: 109960, Avg. loss: 0.004194
Total training time: 0.18 seconds.
Norm: 34.21, NNZs: 40188, Bias: -0.968026, T: 109960, Avg. loss: 0.003830
Total training time: 0.12 seconds.
-- Epoch 11
Convergence after 10 epochs took 0.12 seconds
Norm: 41.80, NNZs: 42350, Bias: -1.088147, T: 98964, Avg. loss: 0.002007
Total training time: 0.13 seconds.
-- Epoch 10
Norm: 44.20, NNZs: 50159, Bias: -1.030244, T: 120956, Avg. loss: 0.004053
Total training time: 0.18 seconds.
Convergence after 11 epochs took 0.18 seconds
Norm: 41.86, NNZs: 42350, Bias: -1.090826, T: 109960, Avg. loss: 0.001925
Total training time: 0.14 seconds.
-- Epoch 11
Norm: 41.89, NNZs: 42350, Bias: -1.092577, T: 120956, Avg. loss: 0.001795
Total training time: 0.14 seconds.
Convergence after 11 epochs took 0.14 seconds
[Parallel(n_jobs=-1)]: Done  18 out of  20 | elapsed:    0.4s remaining:    0.0s
[Parallel(n_jobs=-1)]: Done  20 out of  20 | elapsed:    0.4s finished
train time: 0.410s
test time:  0.013s
accuracy:   0.685
dimensionality: 101322
density: 0.433389


================================================================================
Classifier.PERCEPTRON
________________________________________________________________________________
Training: 
Perceptron(alpha=0.0001, class_weight=None, early_stopping=False, eta0=1.0,
           fit_intercept=True, max_iter=1000, n_iter_no_change=5, n_jobs=-1,
           penalty=None, random_state=0, shuffle=True, tol=0.001,
           validation_fraction=0.1, verbose=True, warm_start=False)
-- Epoch 1
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 12 concurrent workers.
Norm: 20.62, NNZs: 7918, Bias: -0.200000, T: 11314, Avg. loss: 0.008704
Total training time: 0.01 seconds.
-- Epoch 2
-- Epoch 1
-- Epoch 1
-- Epoch 1
-- Epoch 1
-- Epoch 1
Norm: 23.71, NNZs: 9048, Bias: -0.340000, T: 22628, Avg. loss: 0.003264
Total training time: 0.04 seconds.
-- Epoch 1
-- Epoch 1
-- Epoch 3
Norm: 19.88, NNZs: 6980, Bias: -0.190000, T: 11314, Avg. loss: 0.005904
Total training time: 0.02 seconds.
-- Epoch 2
Norm: 24.99, NNZs: 9465, Bias: -0.340000, T: 33942, Avg. loss: 0.001355
Norm: 22.19, NNZs: 7835, Bias: -0.180000, T: 22628, Avg. loss: 0.001949
Total training time: 0.03 seconds.
-- Epoch 3
-- Epoch 1
-- Epoch 1
Norm: 19.60, NNZs: 6427, Bias: -0.170000, T: 11314, Avg. loss: 0.004413
Norm: 19.30, NNZs: 8905, Bias: -0.240000, T: 11314, Avg. loss: 0.005700
Norm: 19.19, NNZs: 6532, Bias: -0.210000, T: 11314, Avg. loss: 0.005987
Norm: 19.29, NNZs: 18626, Bias: -0.150000, T: 11314, Avg. loss: 0.006543
Total training time: 0.06 seconds.
-- Epoch 2
Norm: 17.97, NNZs: 5380, Bias: -0.170000, T: 11314, Avg. loss: 0.003644
Total training time: 0.05 seconds.
Total training time: 0.04 seconds.
-- Epoch 2
Total training time: 0.07 seconds.
-- Epoch 4
Norm: 20.63, NNZs: 8513, Bias: -0.240000, T: 11314, Avg. loss: 0.006834
Total training time: 0.06 seconds.
-- Epoch 2
-- Epoch 1
Norm: 16.12, NNZs: 5554, Bias: -0.170000, T: 11314, Avg. loss: 0.002815
Total training time: 0.01 seconds.
-- Epoch 2
Norm: 23.23, NNZs: 8415, Bias: -0.130000, T: 33942, Avg. loss: 0.001079
Total training time: 0.05 seconds.
-- Epoch 4
Norm: 23.18, NNZs: 9685, Bias: -0.280000, T: 22628, Avg. loss: 0.002558
Total training time: 0.06 seconds.
Total training time: 0.06 seconds.
-- Epoch 3
-- Epoch 2
Norm: 25.58, NNZs: 9624, Bias: -0.280000, T: 45256, Avg. loss: 0.000741
Total training time: 0.09 seconds.
Norm: 17.91, NNZs: 6153, Bias: -0.160000, T: 22628, Avg. loss: 0.000944
-- Epoch 1
Total training time: 0.05 seconds.
-- Epoch 2
Norm: 17.42, NNZs: 9812, Bias: -0.180000, T: 11314, Avg. loss: 0.003640
Total training time: 0.01 seconds.
Norm: 21.33, NNZs: 7481, Bias: -0.220000, T: 22628, Avg. loss: 0.001912
Total training time: 0.06 seconds.
-- Epoch 3
Norm: 17.99, NNZs: 8447, Bias: -0.200000, T: 11314, Avg. loss: 0.004400
Total training time: 0.05 seconds.
-- Epoch 2
Norm: 21.78, NNZs: 23952, Bias: -0.260000, T: 22628, Avg. loss: 0.002653
Total training time: 0.09 seconds.
-- Epoch 3
Norm: 22.38, NNZs: 7803, Bias: -0.190000, T: 33942, Avg. loss: 0.000855
Total training time: 0.06 seconds.
-- Epoch 4
Norm: 19.90, NNZs: 8955, Bias: -0.290000, T: 22628, Avg. loss: 0.001069
Total training time: 0.05 seconds.
-- Epoch 3
Norm: 22.86, NNZs: 24950, Bias: -0.170000, T: 33942, Avg. loss: 0.001145
Total training time: 0.09 seconds.
-- Epoch 2
Norm: 22.95, NNZs: 7988, Bias: -0.150000, T: 45256, Avg. loss: 0.000622
Total training time: 0.06 seconds.
-- Epoch 5
Norm: 24.08, NNZs: 10232, Bias: -0.260000, T: 33942, Avg. loss: 0.001313
Total training time: 0.08 seconds.
-- Epoch 4
Norm: 21.52, NNZs: 9850, Bias: -0.260000, T: 22628, Avg. loss: 0.001481
Total training time: 0.09 seconds.
-- Epoch 3
Norm: 24.13, NNZs: 8611, Bias: -0.120000, T: 45256, Avg. loss: 0.000777
Total training time: 0.08 seconds.
Norm: 19.81, NNZs: 6026, Bias: -0.150000, T: 22628, Avg. loss: 0.001120
Total training time: 0.08 seconds.
-- Epoch 5
Norm: 20.57, NNZs: 9099, Bias: -0.270000, T: 33942, Avg. loss: 0.000234
Total training time: 0.06 seconds.
-- Epoch 3
-- Epoch 4
-- Epoch 5
Norm: 21.07, NNZs: 9218, Bias: -0.290000, T: 45256, Avg. loss: 0.000237
Total training time: 0.06 seconds.
-- Epoch 5
Norm: 24.89, NNZs: 8876, Bias: -0.060000, T: 56570, Avg. loss: 0.000635
Total training time: 0.08 seconds.
-- Epoch 6
Norm: 26.02, NNZs: 9811, Bias: -0.300000, T: 56570, Avg. loss: 0.000536
Total training time: 0.11 seconds.
-- Epoch 2
-- Epoch 6
Norm: 18.96, NNZs: 10453, Bias: -0.280000, T: 22628, Avg. loss: 0.001115
Total training time: 0.03 seconds.
Norm: 21.76, NNZs: 7233, Bias: -0.200000, T: 22628, Avg. loss: 0.001191
Norm: 24.63, NNZs: 10374, Bias: -0.210000, T: 45256, Avg. loss: 0.000931
Total training time: 0.08 seconds.
-- Epoch 3
Norm: 20.77, NNZs: 6259, Bias: -0.150000, T: 33942, Avg. loss: 0.000558
Total training time: 0.09 seconds.
Norm: 23.64, NNZs: 8226, Bias: -0.140000, T: 56570, Avg. loss: 0.000444
Total training time: 0.08 seconds.
Total training time: 0.06 seconds.
-- Epoch 4
-- Epoch 3
Norm: 22.45, NNZs: 7439, Bias: -0.200000, T: 33942, Avg. loss: 0.000578
Total training time: 0.09 seconds.
-- Epoch 4
-- Epoch 6
Norm: 18.80, NNZs: 6387, Bias: -0.180000, T: 33942, Avg. loss: 0.000581
Total training time: 0.06 seconds.
Norm: 26.23, NNZs: 9841, Bias: -0.260000, T: 67884, Avg. loss: 0.000521
Total training time: 0.12 seconds.
-- Epoch 4
-- Epoch 7
Norm: 24.39, NNZs: 8423, Bias: -0.100000, T: 67884, Avg. loss: 0.000406
Total training time: 0.09 seconds.
-- Epoch 4
-- Epoch 7
Norm: 19.32, NNZs: 6518, Bias: -0.150000, T: 45256, Avg. loss: 0.000468
Total training time: 0.06 seconds.
-- Epoch 5
-- Epoch 3
Total training time: 0.11 seconds.
Norm: 26.43, NNZs: 9908, Bias: -0.230000, T: 79198, Avg. loss: 0.000358
Total training time: 0.13 seconds.
-- Epoch 5
-- Epoch 8
Norm: 21.52, NNZs: 6432, Bias: -0.060000, T: 45256, Avg. loss: 0.000389
Total training time: 0.11 seconds.
Norm: 22.43, NNZs: 10417, Bias: -0.290000, T: 33942, Avg. loss: 0.000622
Total training time: 0.12 seconds.
-- Epoch 5
-- Epoch 4
Norm: 25.74, NNZs: 9114, Bias: -0.030000, T: 67884, Avg. loss: 0.000402
Total training time: 0.11 seconds.
-- Epoch 7
Norm: 24.99, NNZs: 10465, Bias: -0.180000, T: 56570, Avg. loss: 0.000819
Total training time: 0.12 seconds.
-- Epoch 6
Norm: 22.22, NNZs: 6683, Bias: -0.030000, T: 56570, Avg. loss: 0.000238
Total training time: 0.11 seconds.
-- Epoch 6
Norm: 26.34, NNZs: 9509, Bias: -0.020000, T: 79198, Avg. loss: 0.000250
Total training time: 0.11 seconds.
Convergence after 7 epochs took 0.11 seconds
Norm: 18.85, NNZs: 7676, Bias: -0.220000, T: 11314, Avg. loss: 0.005340
Total training time: 0.07 seconds.
-- Epoch 2
Norm: 23.71, NNZs: 25149, Bias: -0.140000, T: 45256, Avg. loss: 0.000924
Total training time: 0.13 seconds.
-- Epoch 5
Norm: 26.65, NNZs: 9964, Bias: -0.200000, T: 90512, Avg. loss: 0.000394
Total training time: 0.14 seconds.
Convergence after 8 epochs took 0.14 seconds
Norm: 19.77, NNZs: 10701, Bias: -0.230000, T: 33942, Avg. loss: 0.000517
Total training time: 0.05 seconds.
Norm: 21.35, NNZs: 9254, Bias: -0.290000, T: 56570, Avg. loss: 0.000157
Total training time: 0.10 seconds.
-- Epoch 4
-- Epoch 6
Norm: 23.01, NNZs: 10561, Bias: -0.220000, T: 45256, Avg. loss: 0.000476
Total training time: 0.12 seconds.
-- Epoch 5
Norm: 20.97, NNZs: 8776, Bias: -0.320000, T: 22628, Avg. loss: 0.001648
Total training time: 0.08 seconds.
Norm: 24.57, NNZs: 25311, Bias: -0.100000, T: 56570, Avg. loss: 0.000718
Total training time: 0.14 seconds.
-- Epoch 6
Norm: 24.84, NNZs: 8510, Bias: -0.050000, T: 79198, Avg. loss: 0.000466
Total training time: 0.11 seconds.
-- Epoch 8
Norm: 19.84, NNZs: 6602, Bias: -0.100000, T: 56570, Avg. loss: 0.000289
Norm: 23.32, NNZs: 10723, Bias: -0.250000, T: 56570, Avg. loss: 0.000332
Total training time: 0.09 seconds.
-- Epoch 3
Total training time: 0.13 seconds.
-- Epoch 6
Norm: 22.74, NNZs: 7517, Bias: -0.150000, T: 45256, Avg. loss: 0.000479
Total training time: 0.12 seconds.
-- Epoch 6
Norm: 20.22, NNZs: 10848, Bias: -0.220000, T: 45256, Avg. loss: 0.000481
Total training time: 0.06 seconds.
-- Epoch 5
Norm: 25.34, NNZs: 25446, Bias: -0.090000, T: 67884, Avg. loss: 0.000607
Norm: 21.91, NNZs: 8989, Bias: -0.270000, T: 33942, Avg. loss: 0.000908
Total training time: 0.15 seconds.
-- Epoch 5
-- Epoch 1
Total training time: 0.09 seconds.
Norm: 25.57, NNZs: 11488, Bias: -0.170000, T: 67884, Avg. loss: 0.000664
Total training time: 0.14 seconds.
-- Epoch 4
-- Epoch 7
-- Epoch 7
Norm: 23.14, NNZs: 7634, Bias: -0.130000, T: 56570, Avg. loss: 0.000351
Total training time: 0.13 seconds.
-- Epoch 6
Norm: 21.54, NNZs: 9340, Bias: -0.280000, T: 67884, Avg. loss: 0.000101
Total training time: 0.12 seconds.
-- Epoch 7
Norm: 22.30, NNZs: 9147, Bias: -0.260000, T: 45256, Avg. loss: 0.000814
Total training time: 0.09 seconds.
Norm: 25.92, NNZs: 11552, Bias: -0.150000, T: 79198, Avg. loss: 0.000560
Norm: 25.86, NNZs: 25544, Bias: -0.050000, T: 79198, Avg. loss: 0.000552
Norm: 22.97, NNZs: 6939, Bias: -0.030000, T: 67884, Avg. loss: 0.000163
Total training time: 0.14 seconds.
Total training time: 0.16 seconds.
-- Epoch 8
-- Epoch 7
Norm: 20.24, NNZs: 6682, Bias: -0.060000, T: 67884, Avg. loss: 0.000216
-- Epoch 5
-- Epoch 1
Norm: 18.19, NNZs: 10573, Bias: -0.200000, T: 11314, Avg. loss: 0.003505
Total training time: 0.02 seconds.
-- Epoch 2
Total training time: 0.16 seconds.
Norm: 20.64, NNZs: 11107, Bias: -0.200000, T: 56570, Avg. loss: 0.000317
-- Epoch 8
Norm: 23.33, NNZs: 7034, Bias: 0.000000, T: 79198, Avg. loss: 0.000084
Total training time: 0.16 seconds.
Convergence after 7 epochs took 0.16 seconds
Norm: 22.61, NNZs: 9258, Bias: -0.250000, T: 56570, Avg. loss: 0.000706
Total training time: 0.12 seconds.
Total training time: 0.10 seconds.
Total training time: 0.12 seconds.
-- Epoch 6
Norm: 21.66, NNZs: 9645, Bias: -0.280000, T: 79198, Avg. loss: 0.000094
Total training time: 0.14 seconds.
Convergence after 7 epochs took 0.15 seconds
Norm: 26.52, NNZs: 25680, Bias: -0.030000, T: 90512, Avg. loss: 0.000324
Total training time: 0.18 seconds.
-- Epoch 7
Norm: 21.33, NNZs: 8590, Bias: -0.210000, T: 11314, Avg. loss: 0.006747
Norm: 23.60, NNZs: 10830, Bias: -0.200000, T: 67884, Avg. loss: 0.000282
Total training time: 0.18 seconds.
-- Epoch 7
-- Epoch 6
Norm: 23.46, NNZs: 7767, Bias: -0.120000, T: 67884, Avg. loss: 0.000426
Total training time: 0.02 seconds.
Norm: 25.32, NNZs: 8626, Bias: -0.050000, T: 90512, Avg. loss: 0.000196
Total training time: 0.16 seconds.
Norm: 20.20, NNZs: 11540, Bias: -0.220000, T: 22628, Avg. loss: 0.001118
Total training time: 0.17 seconds.
Convergence after 8 epochs took 0.16 seconds
Norm: 26.25, NNZs: 11641, Bias: -0.110000, T: 90512, Avg. loss: 0.000650
Total training time: 0.18 seconds.
Convergence after 8 epochs took 0.18 seconds
Total training time: 0.04 seconds.
Convergence after 8 epochs took 0.19 seconds
-- Epoch 7
Norm: 20.96, NNZs: 11191, Bias: -0.170000, T: 67884, Avg. loss: 0.000263
-- Epoch 1
Norm: 22.84, NNZs: 9337, Bias: -0.230000, T: 67884, Avg. loss: 0.000391
Total training time: 0.12 seconds.
Total training time: 0.13 seconds.
-- Epoch 7
-- Epoch 1
-- Epoch 1
Norm: 23.85, NNZs: 7864, Bias: -0.070000, T: 79198, Avg. loss: 0.000388
Total training time: 0.19 seconds.
Convergence after 7 epochs took 0.19 seconds
-- Epoch 1
Norm: 20.79, NNZs: 6830, Bias: -0.080000, T: 79198, Avg. loss: 0.000159
Total training time: 0.17 seconds.
Convergence after 7 epochs took 0.17 seconds
-- Epoch 7
Norm: 16.86, NNZs: 7416, Bias: -0.160000, T: 11314, Avg. loss: 0.003266
Total training time: 0.01 seconds.
-- Epoch 2
Norm: 23.10, NNZs: 9361, Bias: -0.220000, T: 79198, Avg. loss: 0.000527
Total training time: 0.16 seconds.
Convergence after 7 epochs took 0.16 seconds
-- Epoch 1
-- Epoch 1
Norm: 18.76, NNZs: 8919, Bias: -0.220000, T: 22628, Avg. loss: 0.000984
Total training time: 0.02 seconds.
-- Epoch 3
-- Epoch 3
Norm: 19.41, NNZs: 12186, Bias: -0.220000, T: 11314, Avg. loss: 0.004918
Total training time: 0.04 seconds.
Norm: 20.26, NNZs: 11776, Bias: -0.240000, T: 11314, Avg. loss: 0.006116
Total training time: 0.03 seconds.
-- Epoch 2
Norm: 19.32, NNZs: 9480, Bias: -0.190000, T: 33942, Avg. loss: 0.000447
Total training time: 0.04 seconds.
-- Epoch 4
-- Epoch 2
Norm: 19.21, NNZs: 11546, Bias: -0.220000, T: 11314, Avg. loss: 0.004775
Total training time: 0.02 seconds.
-- Epoch 2
Norm: 21.18, NNZs: 11263, Bias: -0.150000, T: 79198, Avg. loss: 0.000215
Total training time: 0.17 seconds.
Convergence after 7 epochs took 0.17 seconds
Norm: 23.83, NNZs: 10910, Bias: -0.200000, T: 79198, Avg. loss: 0.000246
Total training time: 0.24 seconds.
Convergence after 7 epochs took 0.24 seconds
Norm: 21.70, NNZs: 12911, Bias: -0.330000, T: 22628, Avg. loss: 0.001566
Total training time: 0.02 seconds.
-- Epoch 2
-- Epoch 3
Norm: 19.10, NNZs: 7931, Bias: -0.180000, T: 11314, Avg. loss: 0.004092
Total training time: 0.03 seconds.
-- Epoch 2
Norm: 21.18, NNZs: 8897, Bias: -0.250000, T: 22628, Avg. loss: 0.001225
Total training time: 0.03 seconds.
-- Epoch 3
Norm: 19.89, NNZs: 9636, Bias: -0.140000, T: 45256, Avg. loss: 0.000344
Total training time: 0.06 seconds.
-- Epoch 5
Norm: 19.48, NNZs: 10373, Bias: -0.270000, T: 11314, Avg. loss: 0.006088
Total training time: 0.05 seconds.
-- Epoch 2
Norm: 20.85, NNZs: 11878, Bias: -0.190000, T: 33942, Avg. loss: 0.000524
Total training time: 0.11 seconds.
-- Epoch 4
Norm: 20.39, NNZs: 9841, Bias: -0.130000, T: 56570, Avg. loss: 0.000298
Total training time: 0.06 seconds.
-- Epoch 6
Norm: 23.66, NNZs: 9647, Bias: -0.250000, T: 22628, Avg. loss: 0.002051
Total training time: 0.10 seconds.
Norm: 22.41, NNZs: 13819, Bias: -0.310000, T: 33942, Avg. loss: 0.000524
Total training time: 0.04 seconds.
-- Epoch 4
Norm: 21.40, NNZs: 12330, Bias: -0.170000, T: 45256, Avg. loss: 0.000448
Total training time: 0.13 seconds.
-- Epoch 5
-- Epoch 3
Norm: 21.28, NNZs: 17413, Bias: -0.260000, T: 22628, Avg. loss: 0.001119
Total training time: 0.08 seconds.
Norm: 22.16, NNZs: 9175, Bias: -0.210000, T: 33942, Avg. loss: 0.000699
Total training time: 0.05 seconds.
-- Epoch 3
-- Epoch 4
Norm: 22.42, NNZs: 13061, Bias: -0.330000, T: 22628, Avg. loss: 0.001592
Total training time: 0.07 seconds.
-- Epoch 3
Norm: 22.68, NNZs: 9697, Bias: -0.200000, T: 45256, Avg. loss: 0.000466
Total training time: 0.06 seconds.
Norm: 24.41, NNZs: 9960, Bias: -0.200000, T: 33942, Avg. loss: 0.000818
Total training time: 0.12 seconds.
Norm: 20.79, NNZs: 9938, Bias: -0.090000, T: 67884, Avg. loss: 0.000246
Total training time: 0.09 seconds.
-- Epoch 4
-- Epoch 7
Norm: 21.77, NNZs: 11762, Bias: -0.260000, T: 22628, Avg. loss: 0.002045
Total training time: 0.08 seconds.
-- Epoch 3
Norm: 22.74, NNZs: 13875, Bias: -0.270000, T: 45256, Avg. loss: 0.000386
Total training time: 0.07 seconds.
-- Epoch 5
-- Epoch 5
Norm: 23.25, NNZs: 13295, Bias: -0.330000, T: 33942, Avg. loss: 0.000690
Total training time: 0.10 seconds.
-- Epoch 4
Norm: 21.79, NNZs: 17543, Bias: -0.220000, T: 33942, Avg. loss: 0.000464
Total training time: 0.11 seconds.
-- Epoch 4
Norm: 22.53, NNZs: 12051, Bias: -0.280000, T: 33942, Avg. loss: 0.000879
Total training time: 0.09 seconds.
-- Epoch 4
Norm: 23.11, NNZs: 13987, Bias: -0.200000, T: 56570, Avg. loss: 0.000377
Total training time: 0.08 seconds.
-- Epoch 6
Norm: 21.77, NNZs: 12484, Bias: -0.140000, T: 56570, Avg. loss: 0.000368
Total training time: 0.16 seconds.
-- Epoch 6
Norm: 24.85, NNZs: 10108, Bias: -0.210000, T: 45256, Avg. loss: 0.000635
Total training time: 0.14 seconds.
-- Epoch 5
Norm: 21.19, NNZs: 10017, Bias: -0.060000, T: 79198, Avg. loss: 0.000188
Total training time: 0.11 seconds.
Convergence after 7 epochs took 0.11 seconds
Norm: 23.67, NNZs: 13423, Bias: -0.300000, T: 45256, Avg. loss: 0.000379
Total training time: 0.11 seconds.
-- Epoch 5
Norm: 22.95, NNZs: 9786, Bias: -0.150000, T: 56570, Avg. loss: 0.000311
Total training time: 0.09 seconds.
Norm: 22.20, NNZs: 17680, Bias: -0.210000, T: 45256, Avg. loss: 0.000419
Total training time: 0.12 seconds.
-- Epoch 6
-- Epoch 5
Norm: 22.27, NNZs: 12633, Bias: -0.140000, T: 67884, Avg. loss: 0.000370
Total training time: 0.16 seconds.
Norm: 23.54, NNZs: 14064, Bias: -0.220000, T: 67884, Avg. loss: 0.000296
Total training time: 0.09 seconds.
-- Epoch 7
-- Epoch 7
Norm: 23.08, NNZs: 12206, Bias: -0.220000, T: 45256, Avg. loss: 0.000550
Total training time: 0.10 seconds.
-- Epoch 5
Norm: 24.08, NNZs: 13566, Bias: -0.310000, T: 56570, Avg. loss: 0.000372
Total training time: 0.11 seconds.
-- Epoch 6
Norm: 25.28, NNZs: 10239, Bias: -0.130000, T: 56570, Avg. loss: 0.000566
Total training time: 0.15 seconds.
-- Epoch 6
Norm: 22.64, NNZs: 18225, Bias: -0.190000, T: 56570, Avg. loss: 0.000341
Total training time: 0.12 seconds.
-- Epoch 6
Norm: 23.39, NNZs: 9872, Bias: -0.120000, T: 67884, Avg. loss: 0.000256
Total training time: 0.09 seconds.
-- Epoch 7
Norm: 22.45, NNZs: 12671, Bias: -0.090000, T: 79198, Avg. loss: 0.000281
Total training time: 0.17 seconds.
Norm: 23.56, NNZs: 12332, Bias: -0.210000, T: 56570, Avg. loss: 0.000421
Total training time: 0.10 seconds.
Norm: 23.78, NNZs: 14146, Bias: -0.160000, T: 79198, Avg. loss: 0.000266
Total training time: 0.09 seconds.
Convergence after 7 epochs took 0.17 seconds
-- Epoch 6
-- Epoch 8
Norm: 24.35, NNZs: 13638, Bias: -0.270000, T: 67884, Avg. loss: 0.000290
Total training time: 0.11 seconds.
-- Epoch 7
Norm: 22.90, NNZs: 18310, Bias: -0.150000, T: 67884, Avg. loss: 0.000261
Total training time: 0.12 seconds.
Norm: 25.78, NNZs: 10466, Bias: -0.140000, T: 67884, Avg. loss: 0.000477
Total training time: 0.15 seconds.
-- Epoch 7
-- Epoch 7
Norm: 23.86, NNZs: 12444, Bias: -0.190000, T: 67884, Avg. loss: 0.000321
Total training time: 0.11 seconds.
-- Epoch 7
Norm: 24.51, NNZs: 13662, Bias: -0.240000, T: 79198, Avg. loss: 0.000245
Total training time: 0.12 seconds.
Norm: 23.81, NNZs: 9980, Bias: -0.100000, T: 79198, Avg. loss: 0.000213
Total training time: 0.10 seconds.
Convergence after 7 epochs took 0.12 seconds
Convergence after 7 epochs took 0.10 seconds
Norm: 23.22, NNZs: 18421, Bias: -0.140000, T: 79198, Avg. loss: 0.000215
Total training time: 0.13 seconds.
Norm: 24.12, NNZs: 14246, Bias: -0.170000, T: 90512, Avg. loss: 0.000255
Total training time: 0.09 seconds.
Convergence after 7 epochs took 0.13 seconds
Norm: 26.09, NNZs: 10551, Bias: -0.100000, T: 79198, Avg. loss: 0.000459
Total training time: 0.15 seconds.
Convergence after 8 epochs took 0.09 seconds
-- Epoch 8
Norm: 24.23, NNZs: 12503, Bias: -0.170000, T: 79198, Avg. loss: 0.000314
Total training time: 0.11 seconds.
-- Epoch 8
Norm: 26.54, NNZs: 11573, Bias: -0.080000, T: 90512, Avg. loss: 0.000349
Total training time: 0.16 seconds.
Convergence after 8 epochs took 0.16 seconds
Norm: 24.71, NNZs: 12605, Bias: -0.150000, T: 90512, Avg. loss: 0.000334
Total training time: 0.11 seconds.
Convergence after 8 epochs took 0.11 seconds
[Parallel(n_jobs=-1)]: Done  18 out of  20 | elapsed:    0.3s remaining:    0.0s
[Parallel(n_jobs=-1)]: Done  20 out of  20 | elapsed:    0.3s finished
train time: 0.411s
test time:  0.013s
accuracy:   0.634
dimensionality: 101322
density: 0.114241


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
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 12 concurrent workers.
[Parallel(n_jobs=-1)]: Done  26 tasks      | elapsed:    2.3s
[Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:    6.4s finished
train time: 6.569s
[Parallel(n_jobs=12)]: Using backend ThreadingBackend with 12 concurrent workers.
[Parallel(n_jobs=12)]: Done  26 tasks      | elapsed:    0.1s
[Parallel(n_jobs=12)]: Done 100 out of 100 | elapsed:    0.2s finished
test time:  0.305s
accuracy:   0.627

================================================================================
Classifier.RIDGE_CLASSIFIER
________________________________________________________________________________
Training: 
RidgeClassifier(alpha=1.0, class_weight=None, copy_X=True, fit_intercept=True,
                max_iter=None, normalize=False, random_state=0, solver='auto',
                tol=0.001)
train time: 2.367s
test time:  0.021s
accuracy:   0.704
dimensionality: 101322
density: 1.000000


================================================================================
Classifier.RIDGE_CLASSIFIERCV
________________________________________________________________________________
Training: 
RidgeClassifierCV(alphas=array([ 0.1,  1. , 10. ]), class_weight=None, cv=None,
                  fit_intercept=True, normalize=False, scoring=None,
                  store_cv_values=False)
train time: 173.823s
test time:  0.018s
accuracy:   0.704
dimensionality: 101322
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
              validation_fraction=0.1, verbose=True, warm_start=False)
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 12 concurrent workers.
-- Epoch 1
-- Epoch 1
-- Epoch 1
Norm: 26.28, NNZs: 12182, Bias: -1.384905, T: 11314, Avg. loss: 0.094156
Total training time: 0.01 seconds.
-- Epoch 2
Norm: 28.33, NNZs: 14055, Bias: -1.276419, T: 11314, Avg. loss: 0.090048
Total training time: 0.00 seconds.
-- Epoch 2
-- Epoch 1
-- Epoch 1
-- Epoch 1
-- Epoch 1
-- Epoch 1
Norm: 26.84, NNZs: 40423, Bias: -1.277664, T: 11314, Avg. loss: 0.085164
Total training time: 0.03 seconds.
-- Epoch 2
Norm: 29.03, NNZs: 10565, Bias: -1.207232, T: 11314, Avg. loss: 0.063908
Total training time: 0.01 seconds.
-- Epoch 2
Norm: 27.82, NNZs: 14837, Bias: -1.258507, T: 11314, Avg. loss: 0.066281
Total training time: 0.02 seconds.
-- Epoch 2
Norm: 26.70, NNZs: 10627, Bias: -1.272591, T: 11314, Avg. loss: 0.075749
Total training time: 0.03 seconds.
-- Epoch 2
-- Epoch 1
-- Epoch 1
-- Epoch 1
Norm: 24.08, NNZs: 12910, Bias: -1.213154, T: 22628, Avg. loss: 0.045494
Total training time: 0.05 seconds.
Norm: 25.30, NNZs: 18472, Bias: -1.202169, T: 22628, Avg. loss: 0.053994
Total training time: 0.06 seconds.
-- Epoch 3
-- Epoch 3
-- Epoch 1
Norm: 27.88, NNZs: 11211, Bias: -1.186235, T: 11314, Avg. loss: 0.071502
Total training time: 0.01 seconds.
-- Epoch 2
Norm: 27.71, NNZs: 14292, Bias: -1.279018, T: 11314, Avg. loss: 0.070411
Total training time: 0.04 seconds.
-- Epoch 2
Norm: 23.59, NNZs: 14430, Bias: -1.277384, T: 22628, Avg. loss: 0.059869
Total training time: 0.07 seconds.
-- Epoch 3
Norm: 25.81, NNZs: 15364, Bias: -1.334667, T: 11314, Avg. loss: 0.056573
Total training time: 0.03 seconds.
Norm: 25.44, NNZs: 13387, Bias: -1.287355, T: 11314, Avg. loss: 0.078009
Total training time: 0.02 seconds.
-- Epoch 2
-- Epoch 2
Norm: 26.77, NNZs: 12873, Bias: -1.178053, T: 11314, Avg. loss: 0.048313
Total training time: 0.02 seconds.
-- Epoch 2
Norm: 24.31, NNZs: 17691, Bias: -1.184354, T: 22628, Avg. loss: 0.035969
Total training time: 0.06 seconds.
-- Epoch 3
Norm: 23.90, NNZs: 19409, Bias: -1.208973, T: 22628, Avg. loss: 0.040264
Total training time: 0.05 seconds.
Norm: 22.64, NNZs: 18848, Bias: -1.252101, T: 22628, Avg. loss: 0.032760
Total training time: 0.03 seconds.
-- Epoch 3
-- Epoch 3
Norm: 23.15, NNZs: 15670, Bias: -1.074247, T: 22628, Avg. loss: 0.026795
Total training time: 0.02 seconds.
-- Epoch 3
Norm: 23.42, NNZs: 20599, Bias: -1.103202, T: 33942, Avg. loss: 0.030228
Total training time: 0.07 seconds.
-- Epoch 4
Norm: 21.59, NNZs: 21772, Bias: -1.172095, T: 33942, Avg. loss: 0.028616
Total training time: 0.04 seconds.
-- Epoch 4
Norm: 27.86, NNZs: 11707, Bias: -1.186255, T: 11314, Avg. loss: 0.082575
Total training time: 0.06 seconds.
-- Epoch 2
Norm: 23.17, NNZs: 26480, Bias: -1.081974, T: 45256, Avg. loss: 0.027827
Total training time: 0.07 seconds.
-- Epoch 5
Norm: 23.47, NNZs: 16606, Bias: -1.186201, T: 22628, Avg. loss: 0.047258
Total training time: 0.03 seconds.
Norm: 22.96, NNZs: 20661, Bias: -1.243450, T: 33942, Avg. loss: 0.052433
Total training time: 0.09 seconds.
-- Epoch 4
Norm: 24.31, NNZs: 22279, Bias: -1.122861, T: 33942, Avg. loss: 0.046081
Total training time: 0.09 seconds.
-- Epoch 4
Norm: 24.61, NNZs: 16772, Bias: -1.160026, T: 22628, Avg. loss: 0.049493
Total training time: 0.06 seconds.
-- Epoch 3
Norm: 25.17, NNZs: 16987, Bias: -1.081764, T: 22628, Avg. loss: 0.035400
Total training time: 0.06 seconds.
-- Epoch 3
-- Epoch 3
Norm: 23.90, NNZs: 21142, Bias: -1.048302, T: 33942, Avg. loss: 0.030297
Total training time: 0.07 seconds.
-- Epoch 4
Norm: 24.77, NNZs: 14476, Bias: -1.096449, T: 22628, Avg. loss: 0.042898
Total training time: 0.04 seconds.
Norm: 22.97, NNZs: 21529, Bias: -1.161275, T: 33942, Avg. loss: 0.034973
Total training time: 0.07 seconds.
-- Epoch 3
-- Epoch 4
Norm: 22.72, NNZs: 22060, Bias: -1.214591, T: 45256, Avg. loss: 0.048642
Total training time: 0.09 seconds.
-- Epoch 5
Norm: 23.98, NNZs: 23445, Bias: -1.111383, T: 45256, Avg. loss: 0.042898
Total training time: 0.10 seconds.
-- Epoch 5
Norm: 23.85, NNZs: 16315, Bias: -1.028794, T: 33942, Avg. loss: 0.037313
Total training time: 0.05 seconds.
-- Epoch 4
Norm: 22.51, NNZs: 18213, Bias: -1.115144, T: 33942, Avg. loss: 0.039901
Total training time: 0.05 seconds.
-- Epoch 4
Norm: 23.61, NNZs: 22329, Bias: -1.000810, T: 45256, Avg. loss: 0.027927
Total training time: 0.08 seconds.
-- Epoch 5
Norm: 23.01, NNZs: 27652, Bias: -1.066084, T: 56570, Avg. loss: 0.026159
Total training time: 0.09 seconds.
Norm: 23.08, NNZs: 14478, Bias: -1.160264, T: 33942, Avg. loss: 0.039257
Total training time: 0.09 seconds.
-- Epoch 6
-- Epoch 4
Norm: 22.57, NNZs: 22707, Bias: -1.105501, T: 45256, Avg. loss: 0.032151
Total training time: 0.08 seconds.
-- Epoch 5
Norm: 22.88, NNZs: 28253, Bias: -1.050624, T: 67884, Avg. loss: 0.025451
Total training time: 0.09 seconds.
Norm: 22.63, NNZs: 15749, Bias: -1.134775, T: 45256, Avg. loss: 0.036477
Total training time: 0.10 seconds.
Norm: 21.07, NNZs: 23492, Bias: -1.137277, T: 45256, Avg. loss: 0.026665
Total training time: 0.06 seconds.
-- Epoch 5
Norm: 24.06, NNZs: 45055, Bias: -1.253349, T: 22628, Avg. loss: 0.052142
Total training time: 0.11 seconds.
-- Epoch 3
Norm: 22.55, NNZs: 22767, Bias: -1.169325, T: 56570, Avg. loss: 0.046463
Total training time: 0.11 seconds.
-- Epoch 6
Norm: 23.37, NNZs: 23056, Bias: -1.002038, T: 56570, Avg. loss: 0.026483
Total training time: 0.09 seconds.
-- Epoch 6
Norm: 23.53, NNZs: 47199, Bias: -1.198766, T: 33942, Avg. loss: 0.045206
Total training time: 0.11 seconds.
Norm: 22.23, NNZs: 19279, Bias: -1.110268, T: 45256, Avg. loss: 0.037730
Total training time: 0.06 seconds.
-- Epoch 4
-- Epoch 5
Norm: 23.25, NNZs: 28115, Bias: -1.003884, T: 67884, Avg. loss: 0.025418
Total training time: 0.09 seconds.
-- Epoch 7
Norm: 23.77, NNZs: 21317, Bias: -1.096530, T: 33942, Avg. loss: 0.043571
Total training time: 0.09 seconds.
-- Epoch 4
Norm: 23.79, NNZs: 25426, Bias: -1.077750, T: 56570, Avg. loss: 0.041147
Total training time: 0.11 seconds.
-- Epoch 6
Norm: 22.08, NNZs: 19909, Bias: -1.086478, T: 56570, Avg. loss: 0.035557
Total training time: 0.07 seconds.
-- Epoch 6
-- Epoch 7
Norm: 23.53, NNZs: 22897, Bias: -1.054314, T: 45256, Avg. loss: 0.040522
Total training time: 0.09 seconds.
-- Epoch 5
Norm: 22.37, NNZs: 23366, Bias: -1.107080, T: 56570, Avg. loss: 0.030725
-- Epoch 5
Total training time: 0.10 seconds.
-- Epoch 6
Norm: 22.76, NNZs: 31367, Bias: -1.048311, T: 79198, Avg. loss: 0.024541
Total training time: 0.11 seconds.
-- Epoch 8
Norm: 22.14, NNZs: 17380, Bias: -1.041265, T: 33942, Avg. loss: 0.023627
Total training time: 0.06 seconds.
Norm: 22.53, NNZs: 29270, Bias: -1.169886, T: 67884, Avg. loss: 0.044934
Total training time: 0.12 seconds.
-- Epoch 4
Norm: 20.98, NNZs: 24855, Bias: -1.147236, T: 56570, Avg. loss: 0.025618
Total training time: 0.08 seconds.
-- Epoch 6
-- Epoch 7
Norm: 22.75, NNZs: 34146, Bias: -1.033172, T: 90512, Avg. loss: 0.024083
Total training time: 0.11 seconds.
-- Epoch 9
Norm: 23.39, NNZs: 22748, Bias: -1.021593, T: 45256, Avg. loss: 0.033899
Total training time: 0.08 seconds.
-- Epoch 5
Norm: 23.69, NNZs: 25956, Bias: -1.051355, T: 67884, Avg. loss: 0.039925
Total training time: 0.13 seconds.
-- Epoch 7
Norm: 22.48, NNZs: 29709, Bias: -1.152985, T: 79198, Avg. loss: 0.044032
Total training time: 0.13 seconds.
-- Epoch 8
Norm: 23.59, NNZs: 27527, Bias: -1.054993, T: 79198, Avg. loss: 0.038749
Total training time: 0.13 seconds.
Norm: 22.21, NNZs: 23910, Bias: -1.096599, T: 67884, Avg. loss: 0.029606
Total training time: 0.11 seconds.
-- Epoch 8
-- Epoch 7
Norm: 22.00, NNZs: 20772, Bias: -1.064987, T: 67884, Avg. loss: 0.034451
Total training time: 0.08 seconds.
-- Epoch 7
Norm: 21.72, NNZs: 21076, Bias: -1.023475, T: 45256, Avg. loss: 0.021883
Total training time: 0.08 seconds.
-- Epoch 5
Norm: 21.45, NNZs: 21899, Bias: -1.016229, T: 56570, Avg. loss: 0.020493
Total training time: 0.08 seconds.
-- Epoch 6
Norm: 23.13, NNZs: 28802, Bias: -1.005145, T: 79198, Avg. loss: 0.024848
Total training time: 0.12 seconds.
-- Epoch 8
Norm: 23.25, NNZs: 48068, Bias: -1.149664, T: 45256, Avg. loss: 0.041747
Total training time: 0.14 seconds.
-- Epoch 5
Norm: 22.34, NNZs: 19024, Bias: -1.108675, T: 56570, Avg. loss: 0.034666
Total training time: 0.13 seconds.
-- Epoch 6
Norm: 20.72, NNZs: 26375, Bias: -1.124065, T: 67884, Avg. loss: 0.024537
Norm: 21.92, NNZs: 21193, Bias: -1.059068, T: 79198, Avg. loss: 0.033577
Total training time: 0.09 seconds.
Norm: 22.21, NNZs: 25007, Bias: -1.084528, T: 79198, Avg. loss: 0.028947
Total training time: 0.12 seconds.
-- Epoch 8
-- Epoch 8
Norm: 22.38, NNZs: 29962, Bias: -1.140181, T: 90512, Avg. loss: 0.043173
Total training time: 0.14 seconds.
-- Epoch 9
Norm: 22.64, NNZs: 38443, Bias: -1.028025, T: 101826, Avg. loss: 0.023507
Total training time: 0.13 seconds.
-- Epoch 10
Norm: 23.14, NNZs: 48717, Bias: -1.118282, T: 56570, Avg. loss: 0.039776
Total training time: 0.15 seconds.
-- Epoch 6
Total training time: 0.10 seconds.
Norm: 23.24, NNZs: 23726, Bias: -1.046860, T: 56570, Avg. loss: 0.038096
Total training time: 0.12 seconds.
-- Epoch 7
-- Epoch 6
Norm: 23.52, NNZs: 27913, Bias: -1.048146, T: 90512, Avg. loss: 0.038081
Total training time: 0.15 seconds.
Norm: 22.25, NNZs: 19637, Bias: -1.097694, T: 67884, Avg. loss: 0.033631
Total training time: 0.14 seconds.
-- Epoch 9
-- Epoch 7
Norm: 23.01, NNZs: 30968, Bias: -1.004788, T: 90512, Avg. loss: 0.024165
Total training time: 0.13 seconds.
Norm: 21.32, NNZs: 22466, Bias: -1.003006, T: 67884, Avg. loss: 0.019631
Total training time: 0.09 seconds.
-- Epoch 9
-- Epoch 7
Norm: 23.06, NNZs: 23752, Bias: -1.001494, T: 56570, Avg. loss: 0.032292
Total training time: 0.10 seconds.
Norm: 21.92, NNZs: 21675, Bias: -1.050793, T: 90512, Avg. loss: 0.032961
Total training time: 0.10 seconds.
-- Epoch 6
-- Epoch 9
Norm: 22.19, NNZs: 30172, Bias: -1.071987, T: 90512, Avg. loss: 0.028208
Total training time: 0.12 seconds.
-- Epoch 9
Norm: 22.39, NNZs: 30158, Bias: -1.137182, T: 101826, Avg. loss: 0.042841
Norm: 22.65, NNZs: 38605, Bias: -1.017262, T: 113140, Avg. loss: 0.023331
Total training time: 0.14 seconds.
Total training time: 0.15 seconds.
Convergence after 10 epochs took 0.14 seconds
-- Epoch 10
-- Epoch 1
Norm: 23.00, NNZs: 49417, Bias: -1.123236, T: 67884, Avg. loss: 0.038499
Total training time: 0.16 seconds.
Norm: 22.97, NNZs: 31185, Bias: -1.008436, T: 101826, Avg. loss: 0.023886
Total training time: 0.13 seconds.
-- Epoch 7
-- Epoch 10
Norm: 21.22, NNZs: 23459, Bias: -1.003020, T: 79198, Avg. loss: 0.019108
Total training time: 0.10 seconds.
Norm: 22.98, NNZs: 25107, Bias: -1.009215, T: 67884, Avg. loss: 0.031401
-- Epoch 8
Total training time: 0.11 seconds.
Norm: 21.86, NNZs: 21978, Bias: -1.046325, T: 101826, Avg. loss: 0.032246
Total training time: 0.11 seconds.
Norm: 22.14, NNZs: 20127, Bias: -1.082775, T: 79198, Avg. loss: 0.032969
Total training time: 0.16 seconds.
Norm: 20.71, NNZs: 26982, Bias: -1.104531, T: 79198, Avg. loss: 0.024024
Total training time: 0.12 seconds.
-- Epoch 7
-- Epoch 8
Norm: 22.38, NNZs: 32961, Bias: -1.129800, T: 113140, Avg. loss: 0.042423
Total training time: 0.17 seconds.
-- Epoch 8
-- Epoch 11
Norm: 23.54, NNZs: 28223, Bias: -1.039054, T: 101826, Avg. loss: 0.037498
Total training time: 0.17 seconds.
Norm: 22.91, NNZs: 49733, Bias: -1.110357, T: 79198, Avg. loss: 0.037547
Total training time: 0.17 seconds.
-- Epoch 10
Norm: 23.19, NNZs: 29134, Bias: -1.019011, T: 67884, Avg. loss: 0.037495
Total training time: 0.14 seconds.
-- Epoch 7
-- Epoch 8
Norm: 23.49, NNZs: 28472, Bias: -1.031342, T: 113140, Avg. loss: 0.036840
Total training time: 0.17 seconds.
Norm: 22.94, NNZs: 31980, Bias: -1.001652, T: 113140, Avg. loss: 0.023719
Total training time: 0.15 seconds.
-- Epoch 11
Norm: 22.06, NNZs: 20521, Bias: -1.071964, T: 90512, Avg. loss: 0.032439
Total training time: 0.17 seconds.
Norm: 23.13, NNZs: 31611, Bias: -1.011432, T: 79198, Avg. loss: 0.036395
Total training time: 0.15 seconds.
-- Epoch 9
-- Epoch 8
Norm: 22.93, NNZs: 33384, Bias: -0.998478, T: 124454, Avg. loss: 0.023609
Total training time: 0.16 seconds.
Norm: 28.52, NNZs: 13342, Bias: -1.273055, T: 11314, Avg. loss: 0.094472
Total training time: 0.03 seconds.
-- Epoch 2
Norm: 22.86, NNZs: 25829, Bias: -1.003314, T: 79198, Avg. loss: 0.030869
Total training time: 0.13 seconds.
-- Epoch 8
Norm: 22.03, NNZs: 21016, Bias: -1.076129, T: 101826, Avg. loss: 0.032103
Total training time: 0.17 seconds.
-- Epoch 10
Norm: 25.62, NNZs: 16949, Bias: -1.140498, T: 22628, Avg. loss: 0.056893
Total training time: 0.04 seconds.
-- Epoch 3
Norm: 22.20, NNZs: 30721, Bias: -1.072982, T: 101826, Avg. loss: 0.027685
Total training time: 0.17 seconds.
-- Epoch 11
Norm: 20.63, NNZs: 27572, Bias: -1.108488, T: 90512, Avg. loss: 0.023345
Total training time: 0.15 seconds.
-- Epoch 9
Norm: 22.84, NNZs: 50373, Bias: -1.090030, T: 90512, Avg. loss: 0.036913
Total training time: 0.19 seconds.
-- Epoch 9
-- Epoch 10
Norm: 20.57, NNZs: 30473, Bias: -1.093924, T: 101826, Avg. loss: 0.023081
Total training time: 0.16 seconds.
-- Epoch 10
Norm: 21.99, NNZs: 21242, Bias: -1.060543, T: 113140, Avg. loss: 0.031562
Total training time: 0.19 seconds.
-- Epoch 10
Norm: 23.11, NNZs: 32103, Bias: -1.012453, T: 90512, Avg. loss: 0.035845
Total training time: 0.17 seconds.
-- Epoch 11
Norm: 21.12, NNZs: 23796, Bias: -1.004060, T: 90512, Avg. loss: 0.018710
Total training time: 0.14 seconds.
-- Epoch 9
Norm: 22.36, NNZs: 37474, Bias: -1.131293, T: 124454, Avg. loss: 0.041914
Total training time: 0.20 seconds.
Norm: 22.81, NNZs: 50990, Bias: -1.091381, T: 101826, Avg. loss: 0.036307
Total training time: 0.20 seconds.
-- Epoch 10
Convergence after 11 epochs took 0.20 seconds
Norm: 21.86, NNZs: 22203, Bias: -1.037723, T: 113140, Avg. loss: 0.031986
Total training time: 0.15 seconds.
-- Epoch 11
Norm: 23.50, NNZs: 29369, Bias: -1.027461, T: 124454, Avg. loss: 0.036826
Total training time: 0.21 seconds.
Norm: 22.90, NNZs: 51212, Bias: -1.088616, T: 113140, Avg. loss: 0.035924
Total training time: 0.21 seconds.
-- Epoch 11
Norm: 21.09, NNZs: 24140, Bias: -1.011675, T: 101826, Avg. loss: 0.018335
Total training time: 0.15 seconds.
-- Epoch 10
Convergence after 11 epochs took 0.19 seconds
-- Epoch 1
-- Epoch 12
Norm: 22.82, NNZs: 51300, Bias: -1.081112, T: 124454, Avg. loss: 0.035434
Total training time: 0.21 seconds.
Norm: 22.03, NNZs: 24111, Bias: -1.056155, T: 124454, Avg. loss: 0.031325
Total training time: 0.20 seconds.
Convergence after 11 epochs took 0.21 seconds
Convergence after 11 epochs took 0.21 seconds
-- Epoch 9
Norm: 22.18, NNZs: 31387, Bias: -1.061484, T: 113140, Avg. loss: 0.027304
Total training time: 0.19 seconds.
Norm: 24.91, NNZs: 24306, Bias: -1.124754, T: 33942, Avg. loss: 0.048811
Total training time: 0.07 seconds.
-- Epoch 4
-- Epoch 11
Norm: 21.85, NNZs: 22387, Bias: -1.040755, T: 124454, Avg. loss: 0.031535
Total training time: 0.17 seconds.
Convergence after 11 epochs took 0.17 seconds
Norm: 22.79, NNZs: 26357, Bias: -1.003260, T: 90512, Avg. loss: 0.030089
Total training time: 0.17 seconds.
-- Epoch 9
Norm: 21.06, NNZs: 29339, Bias: -1.001927, T: 113140, Avg. loss: 0.018123
-- Epoch 1
Norm: 20.59, NNZs: 30663, Bias: -1.085331, T: 113140, Avg. loss: 0.022776
Total training time: 0.19 seconds.
-- Epoch 1
Norm: 23.07, NNZs: 32330, Bias: -1.004153, T: 101826, Avg. loss: 0.035238
Total training time: 0.17 seconds.
Convergence after 10 epochs took 0.17 seconds
Norm: 24.61, NNZs: 26264, Bias: -1.104438, T: 45256, Avg. loss: 0.044854
Total training time: 0.08 seconds.
-- Epoch 1
Norm: 22.68, NNZs: 31683, Bias: -1.007975, T: 101826, Avg. loss: 0.029663
Total training time: 0.19 seconds.
-- Epoch 5
Norm: 25.90, NNZs: 17821, Bias: -1.331532, T: 11314, Avg. loss: 0.077669
Total training time: 0.01 seconds.
Norm: 23.47, NNZs: 29547, Bias: -1.022957, T: 135768, Avg. loss: 0.036339
Norm: 28.71, NNZs: 16846, Bias: -1.163916, T: 11314, Avg. loss: 0.071072
Total training time: 0.04 seconds.
Norm: 22.18, NNZs: 31755, Bias: -1.060057, T: 124454, Avg. loss: 0.026962
Total training time: 0.23 seconds.
Convergence after 11 epochs took 0.23 seconds
Total training time: 0.23 seconds.
Total training time: 0.25 seconds.
-- Epoch 11
-- Epoch 2
-- Epoch 10
-- Epoch 1
Norm: 24.35, NNZs: 31884, Bias: -1.092531, T: 56570, Avg. loss: 0.042197
Total training time: 0.10 seconds.
-- Epoch 10
-- Epoch 6
Convergence after 12 epochs took 0.26 seconds
-- Epoch 1
-- Epoch 1
Norm: 27.82, NNZs: 18012, Bias: -1.323153, T: 11314, Avg. loss: 0.072002
Total training time: 0.04 seconds.
-- Epoch 2
-- Epoch 2
Norm: 26.50, NNZs: 17760, Bias: -1.388290, T: 11314, Avg. loss: 0.081462
Total training time: 0.01 seconds.
-- Epoch 2
Norm: 25.42, NNZs: 14742, Bias: -1.227796, T: 11314, Avg. loss: 0.054779
Total training time: 0.04 seconds.
-- Epoch 2
Norm: 28.26, NNZs: 15823, Bias: -1.235868, T: 11314, Avg. loss: 0.065592
Norm: 24.36, NNZs: 20560, Bias: -1.218327, T: 22628, Avg. loss: 0.048013
Total training time: 0.01 seconds.
-- Epoch 3
Norm: 21.01, NNZs: 13213, Bias: -1.343747, T: 11314, Avg. loss: 0.076112
Total training time: 0.01 seconds.
-- Epoch 2
Norm: 24.21, NNZs: 26405, Bias: -1.180986, T: 22628, Avg. loss: 0.041738
Total training time: 0.05 seconds.
-- Epoch 3
Norm: 24.26, NNZs: 34892, Bias: -1.061660, T: 67884, Avg. loss: 0.041745
Total training time: 0.12 seconds.
-- Epoch 7
Norm: 23.34, NNZs: 20985, Bias: -1.281570, T: 22628, Avg. loss: 0.045441
Total training time: 0.04 seconds.
-- Epoch 3
Norm: 23.59, NNZs: 22146, Bias: -1.195948, T: 33942, Avg. loss: 0.041451
Total training time: 0.02 seconds.
-- Epoch 4
Norm: 23.06, NNZs: 32763, Bias: -0.998945, T: 113140, Avg. loss: 0.034796
Total training time: 0.25 seconds.
-- Epoch 11
Norm: 22.66, NNZs: 31947, Bias: -1.001244, T: 113140, Avg. loss: 0.029387
Total training time: 0.23 seconds.
Convergence after 10 epochs took 0.23 seconds
Total training time: 0.04 seconds.
-- Epoch 2
Norm: 23.28, NNZs: 23053, Bias: -1.161686, T: 45256, Avg. loss: 0.038365
Total training time: 0.02 seconds.
-- Epoch 5
Norm: 20.62, NNZs: 31015, Bias: -1.080236, T: 124454, Avg. loss: 0.022597
Total training time: 0.24 seconds.
Norm: 18.72, NNZs: 16790, Bias: -1.178518, T: 22628, Avg. loss: 0.051335
Total training time: 0.02 seconds.
Convergence after 11 epochs took 0.24 seconds
-- Epoch 3
Norm: 22.34, NNZs: 18082, Bias: -1.148855, T: 22628, Avg. loss: 0.031477
Total training time: 0.05 seconds.
Norm: 23.03, NNZs: 33078, Bias: -1.007344, T: 124454, Avg. loss: 0.034344
Total training time: 0.26 seconds.
-- Epoch 3
-- Epoch 12
Norm: 24.20, NNZs: 35270, Bias: -1.061495, T: 79198, Avg. loss: 0.040139
Total training time: 0.13 seconds.
-- Epoch 8
Norm: 25.09, NNZs: 19792, Bias: -1.134833, T: 22628, Avg. loss: 0.035957
Total training time: 0.05 seconds.
-- Epoch 3
Norm: 22.67, NNZs: 23046, Bias: -1.206437, T: 33942, Avg. loss: 0.039099
Total training time: 0.05 seconds.
Norm: 24.19, NNZs: 35826, Bias: -1.047333, T: 90512, Avg. loss: 0.039445
Total training time: 0.14 seconds.
Norm: 22.99, NNZs: 33271, Bias: -1.005587, T: 135768, Avg. loss: 0.034130
Total training time: 0.26 seconds.
-- Epoch 4
-- Epoch 9
Convergence after 12 epochs took 0.26 seconds
Norm: 25.19, NNZs: 19734, Bias: -1.132730, T: 22628, Avg. loss: 0.040752
Norm: 21.42, NNZs: 20020, Bias: -1.070641, T: 33942, Avg. loss: 0.026980
Total training time: 0.06 seconds.
-- Epoch 4
Total training time: 0.08 seconds.
-- Epoch 3
Norm: 18.29, NNZs: 18775, Bias: -1.158044, T: 33942, Avg. loss: 0.045310
Total training time: 0.03 seconds.
Norm: 23.13, NNZs: 24105, Bias: -1.142217, T: 56570, Avg. loss: 0.036402
Total training time: 0.03 seconds.
-- Epoch 4
-- Epoch 6
Norm: 22.42, NNZs: 24503, Bias: -1.172633, T: 45256, Avg. loss: 0.036095
Total training time: 0.06 seconds.
Norm: 24.16, NNZs: 38819, Bias: -1.040026, T: 101826, Avg. loss: 0.038917
Total training time: 0.14 seconds.
-- Epoch 5
-- Epoch 10
Norm: 24.36, NNZs: 22091, Bias: -1.065058, T: 33942, Avg. loss: 0.035087
Total training time: 0.09 seconds.
Norm: 24.09, NNZs: 22511, Bias: -1.095418, T: 33942, Avg. loss: 0.031088
Total training time: 0.06 seconds.
-- Epoch 4
Norm: 21.14, NNZs: 21426, Bias: -1.049096, T: 45256, Avg. loss: 0.025343
Total training time: 0.07 seconds.
Norm: 23.48, NNZs: 33259, Bias: -1.129037, T: 33942, Avg. loss: 0.035986
Total training time: 0.08 seconds.
-- Epoch 4
Norm: 18.08, NNZs: 20144, Bias: -1.123876, T: 45256, Avg. loss: 0.042231
Total training time: 0.04 seconds.
-- Epoch 5
Norm: 22.30, NNZs: 25366, Bias: -1.152519, T: 56570, Avg. loss: 0.034187
Total training time: 0.07 seconds.
Norm: 23.02, NNZs: 34892, Bias: -1.118993, T: 45256, Avg. loss: 0.033266
Total training time: 0.08 seconds.
-- Epoch 6
-- Epoch 5
-- Epoch 4
Norm: 17.98, NNZs: 20998, Bias: -1.094565, T: 56570, Avg. loss: 0.040456
Total training time: 0.05 seconds.
Norm: 23.55, NNZs: 24021, Bias: -1.073831, T: 45256, Avg. loss: 0.028115
Total training time: 0.07 seconds.
-- Epoch 6
-- Epoch 5
-- Epoch 5
Norm: 17.95, NNZs: 21868, Bias: -1.086560, T: 67884, Avg. loss: 0.039278
Total training time: 0.05 seconds.
-- Epoch 7
Norm: 24.16, NNZs: 39040, Bias: -1.035378, T: 113140, Avg. loss: 0.038264
Total training time: 0.17 seconds.
-- Epoch 11
Norm: 22.19, NNZs: 25795, Bias: -1.139880, T: 67884, Avg. loss: 0.032926
Total training time: 0.09 seconds.
-- Epoch 7
Norm: 23.16, NNZs: 24509, Bias: -1.118543, T: 67884, Avg. loss: 0.035166
Total training time: 0.06 seconds.
-- Epoch 7
Norm: 23.85, NNZs: 25355, Bias: -1.058386, T: 45256, Avg. loss: 0.032096
Total training time: 0.12 seconds.
-- Epoch 5
Norm: 23.51, NNZs: 24959, Bias: -1.044996, T: 56570, Avg. loss: 0.026791
Total training time: 0.09 seconds.
-- Epoch 6
Norm: 24.15, NNZs: 39275, Bias: -1.029636, T: 124454, Avg. loss: 0.037918
Total training time: 0.18 seconds.
Norm: 22.85, NNZs: 39811, Bias: -1.098012, T: 56570, Avg. loss: 0.031316
Total training time: 0.11 seconds.
-- Epoch 12
-- Epoch 6
Norm: 23.03, NNZs: 24838, Bias: -1.108001, T: 79198, Avg. loss: 0.034107
Total training time: 0.07 seconds.
-- Epoch 8
Norm: 23.61, NNZs: 26687, Bias: -1.023504, T: 56570, Avg. loss: 0.030491
Total training time: 0.12 seconds.
-- Epoch 6
Norm: 22.15, NNZs: 26268, Bias: -1.119168, T: 79198, Avg. loss: 0.031964
Norm: 23.33, NNZs: 28949, Bias: -1.031227, T: 67884, Avg. loss: 0.025754
Total training time: 0.09 seconds.
Total training time: 0.10 seconds.
-- Epoch 7
-- Epoch 8
Norm: 17.93, NNZs: 22441, Bias: -1.083432, T: 79198, Avg. loss: 0.038476
Total training time: 0.07 seconds.
-- Epoch 8
Norm: 22.69, NNZs: 41903, Bias: -1.074075, T: 67884, Avg. loss: 0.030415
Total training time: 0.11 seconds.
-- Epoch 7
Norm: 23.00, NNZs: 25165, Bias: -1.097361, T: 90512, Avg. loss: 0.033625
Total training time: 0.08 seconds.
-- Epoch 9
Norm: 23.24, NNZs: 29321, Bias: -1.017932, T: 79198, Avg. loss: 0.024824
Total training time: 0.10 seconds.
Norm: 22.13, NNZs: 26603, Bias: -1.124429, T: 90512, Avg. loss: 0.031383
Total training time: 0.10 seconds.
-- Epoch 8
-- Epoch 9
Norm: 20.77, NNZs: 22770, Bias: -1.037532, T: 56570, Avg. loss: 0.024107
Total training time: 0.11 seconds.
Norm: 22.54, NNZs: 42499, Bias: -1.068645, T: 79198, Avg. loss: 0.029719
Total training time: 0.12 seconds.
-- Epoch 6
-- Epoch 8
Norm: 24.13, NNZs: 42075, Bias: -1.016739, T: 135768, Avg. loss: 0.037604
Total training time: 0.19 seconds.
Convergence after 12 epochs took 0.19 seconds
Norm: 22.11, NNZs: 26924, Bias: -1.112293, T: 101826, Avg. loss: 0.030911
Total training time: 0.11 seconds.
-- Epoch 10
Norm: 22.55, NNZs: 43450, Bias: -1.063771, T: 90512, Avg. loss: 0.029210
Total training time: 0.12 seconds.
Norm: 20.62, NNZs: 23611, Bias: -1.022526, T: 67884, Avg. loss: 0.023288
Total training time: 0.11 seconds.
Norm: 23.49, NNZs: 27580, Bias: -1.021080, T: 67884, Avg. loss: 0.029103
Total training time: 0.13 seconds.
-- Epoch 7
-- Epoch 9
-- Epoch 7
Norm: 17.91, NNZs: 22835, Bias: -1.085403, T: 90512, Avg. loss: 0.037843
Total training time: 0.08 seconds.
-- Epoch 9
Norm: 23.03, NNZs: 25414, Bias: -1.105382, T: 101826, Avg. loss: 0.032925
Total training time: 0.09 seconds.
-- Epoch 10
Norm: 20.60, NNZs: 24158, Bias: -1.014532, T: 79198, Avg. loss: 0.022786
Total training time: 0.12 seconds.
Norm: 22.08, NNZs: 27601, Bias: -1.106991, T: 113140, Avg. loss: 0.030379
Total training time: 0.11 seconds.
-- Epoch 8
-- Epoch 11
Norm: 23.41, NNZs: 28100, Bias: -1.013246, T: 79198, Avg. loss: 0.028213
Total training time: 0.14 seconds.
-- Epoch 8
Norm: 22.50, NNZs: 43671, Bias: -1.060922, T: 101826, Avg. loss: 0.028503
Total training time: 0.13 seconds.
Norm: 23.28, NNZs: 29763, Bias: -1.009089, T: 90512, Avg. loss: 0.024281
Total training time: 0.11 seconds.
-- Epoch 10
-- Epoch 9
Norm: 17.92, NNZs: 23091, Bias: -1.080652, T: 101826, Avg. loss: 0.037346
Total training time: 0.09 seconds.
-- Epoch 10
Norm: 22.94, NNZs: 25762, Bias: -1.089460, T: 113140, Avg. loss: 0.032426
Total training time: 0.09 seconds.
-- Epoch 11
Norm: 23.33, NNZs: 30270, Bias: -1.005720, T: 90512, Avg. loss: 0.027929
Total training time: 0.14 seconds.
-- Epoch 9
Norm: 22.56, NNZs: 43924, Bias: -1.046023, T: 113140, Avg. loss: 0.028098
Total training time: 0.13 seconds.
Convergence after 10 epochs took 0.13 seconds
Norm: 20.54, NNZs: 24844, Bias: -1.005494, T: 90512, Avg. loss: 0.022182
Total training time: 0.12 seconds.
Norm: 22.10, NNZs: 27787, Bias: -1.098450, T: 124454, Avg. loss: 0.030216
Total training time: 0.12 seconds.
Norm: 17.91, NNZs: 23615, Bias: -1.062481, T: 113140, Avg. loss: 0.037074
Total training time: 0.09 seconds.
-- Epoch 11
-- Epoch 9
Convergence after 11 epochs took 0.12 seconds
Norm: 23.12, NNZs: 30074, Bias: -0.998911, T: 101826, Avg. loss: 0.023771
Total training time: 0.11 seconds.
-- Epoch 10
Norm: 23.26, NNZs: 30618, Bias: -1.003762, T: 101826, Avg. loss: 0.027369
Total training time: 0.14 seconds.
-- Epoch 10
Norm: 22.97, NNZs: 25982, Bias: -1.091569, T: 124454, Avg. loss: 0.032122
Total training time: 0.09 seconds.
-- Epoch 12
Norm: 17.92, NNZs: 24121, Bias: -1.057422, T: 124454, Avg. loss: 0.036924
Total training time: 0.09 seconds.
Convergence after 11 epochs took 0.09 seconds
Norm: 23.11, NNZs: 30886, Bias: -1.004737, T: 113140, Avg. loss: 0.023683
Total training time: 0.12 seconds.
-- Epoch 11
Norm: 20.40, NNZs: 25245, Bias: -1.011320, T: 101826, Avg. loss: 0.021803
Total training time: 0.13 seconds.
-- Epoch 10
Norm: 23.23, NNZs: 30804, Bias: -1.008138, T: 113140, Avg. loss: 0.027114
Total training time: 0.15 seconds.
-- Epoch 11
Norm: 23.10, NNZs: 31074, Bias: -1.003084, T: 124454, Avg. loss: 0.023294
Total training time: 0.12 seconds.
Norm: 22.93, NNZs: 26136, Bias: -1.091904, T: 135768, Avg. loss: 0.031822
Total training time: 0.10 seconds.
Convergence after 11 epochs took 0.12 seconds
Convergence after 12 epochs took 0.10 seconds
Norm: 20.36, NNZs: 25583, Bias: -1.002859, T: 113140, Avg. loss: 0.021533
Total training time: 0.13 seconds.
Norm: 23.22, NNZs: 31075, Bias: -1.006857, T: 124454, Avg. loss: 0.026853
Total training time: 0.15 seconds.
Convergence after 10 epochs took 0.13 seconds
Convergence after 11 epochs took 0.15 seconds
[Parallel(n_jobs=-1)]: Done  18 out of  20 | elapsed:    0.4s remaining:    0.0s
[Parallel(n_jobs=-1)]: Done  20 out of  20 | elapsed:    0.4s finished
train time: 0.422s
test time:  0.011s
accuracy:   0.701
dimensionality: 101322
density: 0.318741


Accuracy score for the TWENTY_NEWS_GROUPS dataset (removing headers signatures and quoting): Final classification report: 
1) ADA_BOOST_CLASSIFIER
		Accuracy score = 0.36537440254912373		Training time = 4.842162132263184		Test time = 0.2556889057159424

2) BERNOULLI_NB
		Accuracy score = 0.4584439723844928		Training time = 0.06185650825500488		Test time = 0.05335426330566406

3) COMPLEMENT_NB
		Accuracy score = 0.7133563462559745		Training time = 0.06284403800964355		Test time = 0.010244131088256836

4) DECISION_TREE_CLASSIFIER
		Accuracy score = 0.4391927774827403		Training time = 10.920867204666138		Test time = 0.0055506229400634766

5) EXTRA_TREE_CLASSIFIER
		Accuracy score = 0.2942113648433351		Training time = 0.5782861709594727		Test time = 0.008541107177734375

6) EXTRA_TREES_CLASSIFIER
		Accuracy score = 0.6530801911842804		Training time = 10.459269046783447		Test time = 0.203660249710083

7) GRADIENT_BOOSTING_CLASSIFIER
		Accuracy score = 0.5967870419543282		Training time = 337.8422865867615		Test time = 0.1811671257019043

8) K_NEIGHBORS_CLASSIFIER
		Accuracy score = 0.07010090281465746		Training time = 0.002396821975708008		Test time = 1.6931359767913818

9) LINEAR_SVC
		Accuracy score = 0.69676048858205		Training time = 0.7634968757629395		Test time = 0.008660554885864258

10) LOGISTIC_REGRESSION
		Accuracy score = 0.6946362187997875		Training time = 17.368773460388184		Test time = 0.011348724365234375

11) LOGISTIC_REGRESSION_CV
		Accuracy score = 0.6935740839086564		Training time = 409.5075442790985		Test time = 0.010601282119750977

12) MLP_CLASSIFIER
		Accuracy score = 0.699814126394052		Training time = 1357.2824909687042		Test time = 0.04013490676879883

13) MULTINOMIAL_NB
		Accuracy score = 0.6712692511949018		Training time = 0.08335185050964355		Test time = 0.009906768798828125

14) NEAREST_CENTROID
		Accuracy score = 0.6427243759957515		Training time = 0.016117095947265625		Test time = 0.013181686401367188

15) NU_SVC
		Accuracy score = 0.6919808815719597		Training time = 82.38007354736328		Test time = 27.44763970375061

16) PASSIVE_AGGRESSIVE_CLASSIFIER
		Accuracy score = 0.6848114710568242		Training time = 0.40989017486572266		Test time = 0.012604236602783203

17) PERCEPTRON
		Accuracy score = 0.6336962294211365		Training time = 0.41126322746276855		Test time = 0.013092756271362305

18) RANDOM_FOREST_CLASSIFIER
		Accuracy score = 0.6267923526287839		Training time = 6.5693678855896		Test time = 0.30539488792419434

19) RIDGE_CLASSIFIER
		Accuracy score = 0.7035315985130112		Training time = 2.367365598678589		Test time = 0.02110910415649414

20) RIDGE_CLASSIFIERCV
		Accuracy score = 0.7036643653744026		Training time = 173.8229410648346		Test time = 0.017713546752929688

21) SGD_CLASSIFIER
		Accuracy score = 0.701141795007966		Training time = 0.421708345413208		Test time = 0.011092901229858398



Best algorithm:
===> 3) COMPLEMENT_NB
		Accuracy score = 0.7133563462559745		Training time = 0.06284403800964355		Test time = 0.010244131088256836

Loading IMDB_REVIEWS dataset:

===> Reading files from /home/rpessoa/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/train/neg

===> Reading files from /home/rpessoa/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/train/pos

===> Reading files from /home/rpessoa/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/test/neg

===> Reading files from /home/rpessoa/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/test/pos
data loaded
25000 documents - 33.133MB (training set)
25000 documents - 32.351MB (test set)

Extracting features from the training data using a vectorizer
done in 2.379225s at 13.926MB/s
n_samples: 25000, n_features: 74535

Extracting features from the test data using the same vectorizer
done in 2.308750s at 14.012MB/s
n_samples: 25000, n_features: 74535

================================================================================
Classifier.ADA_BOOST_CLASSIFIER
________________________________________________________________________________
Training: 
AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
                   n_estimators=50, random_state=0)
train time: 11.368s
test time:  0.717s
accuracy:   0.359

================================================================================
Classifier.BERNOULLI_NB
________________________________________________________________________________
Training: 
BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
train time: 0.039s
test time:  0.038s
accuracy:   0.371
dimensionality: 74535
density: 1.000000


================================================================================
Classifier.COMPLEMENT_NB
________________________________________________________________________________
Training: 
ComplementNB(alpha=1.0, class_prior=None, fit_prior=True, norm=False)
train time: 0.034s
test time:  0.019s
accuracy:   0.373
dimensionality: 74535
density: 1.000000


================================================================================
Classifier.DECISION_TREE_CLASSIFIER
________________________________________________________________________________
Training: 
DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=None, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=0, splitter='best')
train time: 35.066s
test time:  0.014s
accuracy:   0.258

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
train time: 1.008s
test time:  0.020s
accuracy:   0.221

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
                     oob_score=False, random_state=0, verbose=True,
                     warm_start=False)
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 12 concurrent workers.
[Parallel(n_jobs=-1)]: Done  26 tasks      | elapsed:    5.5s
[Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:   15.7s finished
train time: 15.878s
[Parallel(n_jobs=12)]: Using backend ThreadingBackend with 12 concurrent workers.
[Parallel(n_jobs=12)]: Done  26 tasks      | elapsed:    0.2s
[Parallel(n_jobs=12)]: Done 100 out of 100 | elapsed:    0.5s finished
test time:  0.510s
accuracy:   0.374

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
         1       49944.5758            6.72m
         2       49361.8139            6.64m
         3       48865.8490            6.57m
         4       48435.6162            6.48m
         5       48044.9568            6.41m
         6       47698.4252            6.34m
         7       47371.1355            6.27m
         8       47062.4232            6.20m
         9       46785.6489            6.13m
        10       46523.9006            6.06m
        20       44475.3602            5.36m
        30       43009.1486            4.68m
        40       41840.5535            4.00m
        50       40860.7617            3.33m
        60       40042.0764            2.66m
        70       39335.0087            1.99m
        80       38676.0345            1.33m
        90       38064.4440           39.80s
       100       37507.5335            0.00s
train time: 397.786s
test time:  0.258s
accuracy:   0.376

================================================================================
Classifier.K_NEIGHBORS_CLASSIFIER
________________________________________________________________________________
Training: 
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=-1, n_neighbors=5, p=2,
                     weights='uniform')
train time: 0.006s
test time:  12.872s
accuracy:   0.264

================================================================================
Classifier.LINEAR_SVC
________________________________________________________________________________
Training: 
LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
          intercept_scaling=1, loss='squared_hinge', max_iter=1000,
          multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
          verbose=True)
....*
optimization finished, #iter = 41
Objective value = -4835.989524
nSV = 13807
...*.
optimization finished, #iter = 40
Objective value = -3863.811590
nSV = 13560
...*.
optimization finished, #iter = 40
Objective value = -4034.135653
nSV = 14041
...*
optimization finished, #iter = 39
Objective value = -4215.146690
nSV = 14325
...*
optimization finished, #iter = 39
Objective value = -3959.934463
nSV = 13434
....*
optimization finished, #iter = 41
Objective value = -4767.545340
nSV = 14563
...*.
optimization finished, #iter = 40
Objective value = -3808.434762
nSV = 13173
...*.
optimization finished, #iter = 40
Objective value = -5277.013048
nSV = 14543
[LibLinear]train time: 1.769s
test time:  0.018s
accuracy:   0.373
dimensionality: 74535
density: 0.846716


================================================================================
Classifier.LOGISTIC_REGRESSION
________________________________________________________________________________
Training: 
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=-1, penalty='l2', random_state=0,
                   solver='lbfgs', tol=0.0001, verbose=True, warm_start=False)
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 12 concurrent workers.
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
train time: 9.818s
[Parallel(n_jobs=-1)]: Done   1 out of   1 | elapsed:    9.8s finished
test time:  0.021s
accuracy:   0.421
dimensionality: 74535
density: 1.000000


================================================================================
Classifier.LOGISTIC_REGRESSION_CV
________________________________________________________________________________
Training: 
LogisticRegressionCV(Cs=10, class_weight=None, cv=None, dual=False,
                     fit_intercept=True, intercept_scaling=1.0, l1_ratios=None,
                     max_iter=100, multi_class='auto', n_jobs=-1, penalty='l2',
                     random_state=0, refit=True, scoring=None, solver='lbfgs',
                     tol=0.0001, verbose=True)
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 12 concurrent workers.
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
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =       596288     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  4.15888D+04    |proj g|=  1.58000D+03

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
*****     10     13      1     0     0   6.381D-02   4.053D+04
  F =   40525.163214755936     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =       596288     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  4.05204D+04    |proj g|=  5.36626D+01

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
*****      5      8      1     0     0   5.028D-02   4.049D+04
  F =   40488.786795757544     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =       596288     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  4.04526D+04    |proj g|=  5.32781D+01

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
*****     10     14      1     0     0   3.485D-02   4.022D+04
  F =   40216.954191496952     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =       596288     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  3.99534D+04    |proj g|=  5.07273D+01

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
*****     20     23      1     0     0   3.957D-02   3.855D+04
  F =   38545.694533354821     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =       596288     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  3.71563D+04    |proj g|=  3.84242D+01

At iterate   50    f=  3.27595D+04    |proj g|=  4.89860D+00

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
*****     68     75      1     0     0   2.475D-02   3.276D+04
  F =   32759.440681164193     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =       596288     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  2.91064D+04    |proj g|=  1.43825D+01

At iterate   50    f=  2.12448D+04    |proj g|=  1.04701D+01

At iterate  100    f=  2.12152D+04    |proj g|=  1.68785D+01

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
*****    100    107      1     0     0   1.688D+01   2.122D+04
  F =   21215.185616145238     

STOP: TOTAL NO. of ITERATIONS REACHED LIMIT                 
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =       596288     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  1.52854D+04    |proj g|=  1.68785D+01

At iterate   50    f=  8.72365D+03    |proj g|=  7.49560D+01

At iterate  100    f=  8.54829D+03    |proj g|=  4.02206D+00

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
*****    100    108      1     0     0   4.022D+00   8.548D+03
  F =   8548.2949501604508     

STOP: TOTAL NO. of ITERATIONS REACHED LIMIT                 
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =       596288     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  4.43398D+03    |proj g|=  4.02206D+00

At iterate   50    f=  2.41854D+03    |proj g|=  1.89405D+00

At iterate  100    f=  2.38228D+03    |proj g|=  2.73061D+00

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
*****    100    114      1     0     0   2.731D+00   2.382D+03
  F =   2382.2803879496205     

STOP: TOTAL NO. of ITERATIONS REACHED LIMIT                 
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =       596288     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  9.77508D+02    |proj g|=  2.73061D+00

At iterate   50    f=  5.56706D+02    |proj g|=  4.09203D+00

At iterate  100    f=  5.50823D+02    |proj g|=  1.52505D-01

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
*****    100    106      1     0     0   1.525D-01   5.508D+02
  F =   550.82282956565098     

STOP: TOTAL NO. of ITERATIONS REACHED LIMIT                 
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =       596288     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  1.99984D+02    |proj g|=  1.52505D-01

At iterate   50    f=  1.14726D+02    |proj g|=  4.87173D-02

At iterate  100    f=  1.14366D+02    |proj g|=  1.23712D-01

           * * *

RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =       596288     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  4.15888D+04    |proj g|=  1.58000D+03

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
*****     10     13      1     0     0   6.446D-02   4.052D+04
  F =   40524.873344502572     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =       596288     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  4.05202D+04    |proj g|=  5.32077D+01

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
*****      5      8      1     0     0   4.531D-02   4.049D+04
  F =   40488.462971938090     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =       596288     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  4.04522D+04    |proj g|=  5.28297D+01

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
*****     10     14      1     0     0   4.122D-02   4.022D+04
  F =   40216.311786751598     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =       596288     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  3.99524D+04    |proj g|=  5.04810D+01

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
*****     20     24      1     0     0   4.793D-02   3.854D+04
  F =   38540.933347699807     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =       596288     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  3.71469D+04    |proj g|=  3.86599D+01

At iterate   50    f=  3.27342D+04    |proj g|=  1.42455D+00

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
*****     83     91      1     0     0   1.027D+00   3.273D+04
  F =   32734.060862339960     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =       596288     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  2.90677D+04    |proj g|=  1.50291D+01

At iterate   50    f=  2.12403D+04    |proj g|=  3.88914D+01

At iterate  100    f=  2.11746D+04    |proj g|=  2.40698D+01

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
*****    100    106      1     0     0   2.407D+01   2.117D+04
  F =   21174.616221561624     

STOP: TOTAL NO. of ITERATIONS REACHED LIMIT                 
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =       596288     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  1.53236D+04    |proj g|=  2.40698D+01

At iterate   50    f=  8.67594D+03    |proj g|=  4.86027D+01

At iterate  100    f=  8.50677D+03    |proj g|=  4.13472D+00

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
*****    100    108      1     0     0   4.135D+00   8.507D+03
  F =   8506.7713577178947     

STOP: TOTAL NO. of ITERATIONS REACHED LIMIT                 
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =       596288     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  4.41647D+03    |proj g|=  4.13472D+00

At iterate   50    f=  2.40105D+03    |proj g|=  6.07605D+00

At iterate  100    f=  2.37568D+03    |proj g|=  4.07409D+00

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
*****    100    110      1     0     0   4.074D+00   2.376D+03
  F =   2375.6825639502704     

STOP: TOTAL NO. of ITERATIONS REACHED LIMIT                 
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =       596288     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  9.91660D+02    |proj g|=  4.07409D+00

At iterate   50    f=  5.49600D+02    |proj g|=  8.45548D-01

At iterate  100    f=  5.47917D+02    |proj g|=  4.67537D-01

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
*****    100    108      1     0     0   4.675D-01   5.479D+02
  F =   547.91681158447534     

STOP: TOTAL NO. of ITERATIONS REACHED LIMIT                 
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =       596288     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  1.98594D+02    |proj g|=  4.67537D-01

At iterate   50    f=  1.14880D+02    |proj g|=  2.62736D-01

At iterate  100    f=  1.13827D+02    |proj g|=  3.83129D-02

           * * *

RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =       596288     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  4.15888D+04    |proj g|=  1.58000D+03

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
*****     10     13      1     0     0   6.414D-02   4.052D+04
  F =   40524.398622218971     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =       596288     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  4.05197D+04    |proj g|=  5.46948D+01

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
*****      5      8      1     0     0   5.086D-02   4.049D+04
  F =   40487.841139988428     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =       596288     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  4.04514D+04    |proj g|=  5.44239D+01

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
*****     10     14      1     0     0   3.415D-02   4.021D+04
  F =   40214.670383686978     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =       596288     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  3.99499D+04    |proj g|=  5.23731D+01

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
*****     24     27      1     0     0   5.758D-02   3.854D+04
  F =   38536.053843946102     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =       596288     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  3.71414D+04    |proj g|=  3.98422D+01

At iterate   50    f=  3.27377D+04    |proj g|=  7.99769D-01

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
*****     92     99      1     0     0   7.035D-02   3.274D+04
  F =   32737.609111901158     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =       596288     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  2.90798D+04    |proj g|=  1.50778D+01

At iterate   50    f=  2.12984D+04    |proj g|=  1.18616D+01

At iterate  100    f=  2.12026D+04    |proj g|=  6.18979D+00

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
*****    100    112      1     0     0   6.190D+00   2.120D+04
  F =   21202.579367588285     

STOP: TOTAL NO. of ITERATIONS REACHED LIMIT                 
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =       596288     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  1.53509D+04    |proj g|=  6.18979D+00

At iterate   50    f=  8.64547D+03    |proj g|=  4.88469D+01

At iterate  100    f=  8.52111D+03    |proj g|=  6.64629D+00

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
*****    100    104      1     0     0   6.646D+00   8.521D+03
  F =   8521.1128897203707     

STOP: TOTAL NO. of ITERATIONS REACHED LIMIT                 
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =       596288     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  4.38402D+03    |proj g|=  6.64629D+00

At iterate   50    f=  2.39556D+03    |proj g|=  1.14521D+00

At iterate  100    f=  2.37723D+03    |proj g|=  6.52253D+00

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
*****    100    106      1     0     0   6.523D+00   2.377D+03
  F =   2377.2255170766289     

STOP: TOTAL NO. of ITERATIONS REACHED LIMIT                 
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =       596288     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  9.78142D+02    |proj g|=  6.52253D+00

At iterate   50    f=  5.57836D+02    |proj g|=  2.36645D+00

At iterate  100    f=  5.49915D+02    |proj g|=  4.52650D-01

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
*****    100    107      1     0     0   4.526D-01   5.499D+02
  F =   549.91542192644852     

STOP: TOTAL NO. of ITERATIONS REACHED LIMIT                 
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =       596288     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  1.92968D+02    |proj g|=  4.52650D-01

At iterate   50    f=  1.15940D+02    |proj g|=  3.11895D-01

At iterate  100    f=  1.14298D+02    |proj g|=  5.74866D-02

           * * *

RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =       596288     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  4.15888D+04    |proj g|=  1.58000D+03

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
*****     10     13      1     0     0   6.428D-02   4.052D+04
  F =   40524.570090035697     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =       596288     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  4.05198D+04    |proj g|=  5.32039D+01

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
*****      5      8      1     0     0   4.883D-02   4.049D+04
  F =   40488.051335825468     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =       596288     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  4.04517D+04    |proj g|=  5.28242D+01

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
*****     10     14      1     0     0   3.553D-02   4.022D+04
  F =   40215.113431805948     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =       596288     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  3.99505D+04    |proj g|=  5.04966D+01

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
*****     25     29      1     0     0   1.718D-02   3.854D+04
  F =   38536.372502215410     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =       596288     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  3.71407D+04    |proj g|=  3.85088D+01

At iterate   50    f=  3.27369D+04    |proj g|=  3.91803D+00

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
*****     85     94      1     0     0   2.077D-02   3.274D+04
  F =   32736.684084094417     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =       596288     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  2.90819D+04    |proj g|=  1.47510D+01

At iterate   50    f=  2.12723D+04    |proj g|=  1.05286D+01

At iterate  100    f=  2.12076D+04    |proj g|=  9.56054D+00

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
*****    100    109      1     0     0   9.561D+00   2.121D+04
  F =   21207.606106306124     

STOP: TOTAL NO. of ITERATIONS REACHED LIMIT                 
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =       596288     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  1.52685D+04    |proj g|=  9.56054D+00

At iterate   50    f=  8.68734D+03    |proj g|=  5.65199D+01

At iterate  100    f=  8.54726D+03    |proj g|=  1.62988D+01

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
*****    100    113      1     0     0   1.630D+01   8.547D+03
  F =   8547.2628356034093     

STOP: TOTAL NO. of ITERATIONS REACHED LIMIT                 
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =       596288     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  4.56693D+03    |proj g|=  1.62988D+01

At iterate   50    f=  2.41192D+03    |proj g|=  2.48277D+00

At iterate  100    f=  2.38606D+03    |proj g|=  1.45769D+00

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
*****    100    106      1     0     0   1.458D+00   2.386D+03
  F =   2386.0640002575497     

STOP: TOTAL NO. of ITERATIONS REACHED LIMIT                 
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =       596288     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  9.53352D+02    |proj g|=  1.45769D+00

At iterate   50    f=  5.50272D+02    |proj g|=  1.49478D+00

At iterate  100    f=  5.47941D+02    |proj g|=  6.34361D-01

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
*****    100    110      1     0     0   6.344D-01   5.479D+02
  F =   547.94074699718567     

STOP: TOTAL NO. of ITERATIONS REACHED LIMIT                 
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =       596288     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  1.91972D+02    |proj g|=  6.34361D-01

At iterate   50    f=  1.13722D+02    |proj g|=  2.50699D-01

At iterate  100    f=  1.12788D+02    |proj g|=  1.26071D-01

           * * *

RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =       596288     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  4.15888D+04    |proj g|=  1.58000D+03

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
*****     10     13      1     0     0   6.414D-02   4.052D+04
  F =   40524.507557670862     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =       596288     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  4.05198D+04    |proj g|=  5.34908D+01

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
*****      5      8      1     0     0   5.053D-02   4.049D+04
  F =   40488.024997764289     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =       596288     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  4.04517D+04    |proj g|=  5.30987D+01

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
*****     10     14      1     0     0   2.537D-02   4.022D+04
  F =   40215.421641801462     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =       596288     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  3.99512D+04    |proj g|=  5.01825D+01

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
*****     19     23      1     0     0   8.708D-02   3.854D+04
  F =   38540.135343612463     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =       596288     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  3.71476D+04    |proj g|=  3.75282D+01

At iterate   50    f=  3.27356D+04    |proj g|=  1.80871D+00

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
*****     93    105      1     0     0   2.116D-01   3.274D+04
  F =   32735.220193766188     

CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH             
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =       596288     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  2.90698D+04    |proj g|=  1.39866D+01

At iterate   50    f=  2.12199D+04    |proj g|=  3.56201D+01

At iterate  100    f=  2.11603D+04    |proj g|=  1.65724D+01

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
*****    100    109      1     0     0   1.657D+01   2.116D+04
  F =   21160.297096453742     

STOP: TOTAL NO. of ITERATIONS REACHED LIMIT                 
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =       596288     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  1.52048D+04    |proj g|=  1.65724D+01

At iterate   50    f=  8.68705D+03    |proj g|=  9.16636D+01

At iterate  100    f=  8.51753D+03    |proj g|=  2.92682D+01

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
*****    100    108      1     0     0   2.927D+01   8.518D+03
  F =   8517.5335810628785     

STOP: TOTAL NO. of ITERATIONS REACHED LIMIT                 
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =       596288     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  4.46795D+03    |proj g|=  2.92682D+01

At iterate   50    f=  2.40100D+03    |proj g|=  7.41542D+00

At iterate  100    f=  2.37489D+03    |proj g|=  3.03382D+00

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
*****    100    109      1     0     0   3.034D+00   2.375D+03
  F =   2374.8887798274095     

STOP: TOTAL NO. of ITERATIONS REACHED LIMIT                 
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =       596288     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  9.78711D+02    |proj g|=  3.03382D+00

At iterate   50    f=  5.55127D+02    |proj g|=  2.82779D+00

At iterate  100    f=  5.47752D+02    |proj g|=  2.88639D-01

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
*****    100    107      1     0     0   2.886D-01   5.478D+02
  F =   547.75240274225644     

STOP: TOTAL NO. of ITERATIONS REACHED LIMIT                 
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =       596288     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  1.96563D+02    |proj g|=  2.88639D-01

At iterate   50    f=  1.14508D+02    |proj g|=  4.97745D-01

At iterate  100    f=  1.13718D+02    |proj g|=  5.16099D-02

           * * *

[Parallel(n_jobs=-1)]: Done   5 out of   5 | elapsed:  2.2min finished
/home/rpessoa/virtual_envs/comp551_p2/lib/python3.6/site-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
train time: 152.318s
test time:  0.038s
accuracy:   0.405
dimensionality: 74535
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
              tol=0.0001, validation_fraction=0.1, verbose=True,
              warm_start=False)
Iteration 1, loss = 1.88821954
Iteration 2, loss = 1.38074371
Iteration 3, loss = 0.99497041
Iteration 4, loss = 0.66169230
Iteration 5, loss = 0.40625200
Iteration 6, loss = 0.24563625
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =       596288     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  5.19860D+04    |proj g|=  1.97500D+03

At iterate   50    f=  3.42068D+04    |proj g|=  6.15924D+01

At iterate  100    f=  3.41640D+04    |proj g|=  2.57417D+00

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
*****    100    106      1     0     0   2.574D+00   3.416D+04
  F =   34164.019144800834     

STOP: TOTAL NO. of ITERATIONS REACHED LIMIT                 
 This problem is unconstrained.
Iteration 7, loss = 0.15386605
Iteration 8, loss = 0.10187055
Iteration 9, loss = 0.07194523
Iteration 10, loss = 0.05388829
Iteration 11, loss = 0.04236856
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
Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
*****    100    110      1     0     0   1.237D-01   1.144D+02
  F =   114.36637995206640     

STOP: TOTAL NO. of ITERATIONS REACHED LIMIT                 
Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
*****    100    106      1     0     0   3.831D-02   1.138D+02
  F =   113.82735303221637     

STOP: TOTAL NO. of ITERATIONS REACHED LIMIT                 
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
Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
*****    100    106      1     0     0   5.749D-02   1.143D+02
  F =   114.29822919670841     

STOP: TOTAL NO. of ITERATIONS REACHED LIMIT                 
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
Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
*****    100    106      1     0     0   1.261D-01   1.128D+02
  F =   112.78828072788849     

STOP: TOTAL NO. of ITERATIONS REACHED LIMIT                 
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
Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
*****    100    112      1     0     0   5.161D-02   1.137D+02
  F =   113.71836320660441     

STOP: TOTAL NO. of ITERATIONS REACHED LIMIT                 
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
Iteration 12, loss = 0.03470406
Iteration 13, loss = 0.02940594
Iteration 14, loss = 0.02567412
Iteration 15, loss = 0.02281451
Iteration 16, loss = 0.02067803
Iteration 17, loss = 0.01896610
Iteration 18, loss = 0.01764399
Iteration 19, loss = 0.01658385
Iteration 20, loss = 0.01565647
Iteration 21, loss = 0.01492700
Iteration 22, loss = 0.01435956
Iteration 23, loss = 0.01384673
Iteration 24, loss = 0.01337927
Iteration 25, loss = 0.01290557
Iteration 26, loss = 0.01258513
Iteration 27, loss = 0.01215420
Iteration 28, loss = 0.01190445
Iteration 29, loss = 0.01157396
Iteration 30, loss = 0.01127146
Iteration 31, loss = 0.01102765
Iteration 32, loss = 0.01080103
Iteration 33, loss = 0.01057912
Iteration 34, loss = 0.01041932
Iteration 35, loss = 0.01016635
Iteration 36, loss = 0.00998523
Iteration 37, loss = 0.00982923
Iteration 38, loss = 0.00966837
Iteration 39, loss = 0.00943278
Iteration 40, loss = 0.00935212
Iteration 41, loss = 0.00910540
Iteration 42, loss = 0.00892370
Iteration 43, loss = 0.00876417
Iteration 44, loss = 0.00870520
Iteration 45, loss = 0.00846627
Iteration 46, loss = 0.00847361
Iteration 47, loss = 0.00867171
Iteration 48, loss = 0.00808645
Iteration 49, loss = 0.00795000
Iteration 50, loss = 0.00771547
Iteration 51, loss = 0.00756760
Iteration 52, loss = 0.00743549
Iteration 53, loss = 0.00718323
Iteration 54, loss = 0.00716503
Iteration 55, loss = 0.00705607
Iteration 56, loss = 0.00741243
Iteration 57, loss = 0.00686271
Iteration 58, loss = 0.00685018
Iteration 59, loss = 0.00661466
Iteration 60, loss = 0.00644355
Iteration 61, loss = 0.00636596
Iteration 62, loss = 0.00632173
Iteration 63, loss = 0.00690166
Iteration 64, loss = 0.00628380
Iteration 65, loss = 0.00627582
Iteration 66, loss = 0.00635025
Iteration 67, loss = 0.00608425
Iteration 68, loss = 0.00579360
Iteration 69, loss = 0.00579813
Iteration 70, loss = 0.00572884
Iteration 71, loss = 0.00552667
Iteration 72, loss = 0.00610480
Iteration 73, loss = 0.00626026
Iteration 74, loss = 0.00566808
Iteration 75, loss = 0.00590936
Iteration 76, loss = 0.00536676
Iteration 77, loss = 0.00528492
Iteration 78, loss = 0.00620725
Iteration 79, loss = 0.00522557
Iteration 80, loss = 0.00528428
Iteration 81, loss = 0.00527695
Iteration 82, loss = 0.00505894
Iteration 83, loss = 0.00502477
Iteration 84, loss = 0.00484341
Iteration 85, loss = 0.00500680
Iteration 86, loss = 0.00548555
Iteration 87, loss = 0.00515078
Iteration 88, loss = 0.00511730
Iteration 89, loss = 0.00500726
Iteration 90, loss = 0.00544699
Iteration 91, loss = 0.00531749
Iteration 92, loss = 0.00550182
Iteration 93, loss = 0.00556722
Iteration 94, loss = 0.00574197
Iteration 95, loss = 0.00522414
Training loss did not improve more than tol=0.000100 for 10 consecutive epochs. Stopping.
train time: 2304.338s
test time:  0.161s
accuracy:   0.345

================================================================================
Classifier.MULTINOMIAL_NB
________________________________________________________________________________
Training: 
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
train time: 0.067s
test time:  0.019s
accuracy:   0.349
dimensionality: 74535
density: 1.000000


================================================================================
Classifier.NEAREST_CENTROID
________________________________________________________________________________
Training: 
NearestCentroid(metric='euclidean', shrink_threshold=None)
train time: 0.023s
test time:  0.023s
accuracy:   0.368

================================================================================
Classifier.NU_SVC
________________________________________________________________________________
Training: 
NuSVC(break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
      max_iter=-1, nu=0.5, probability=False, random_state=0, shrinking=True,
      tol=0.001, verbose=True)
..........*
optimization finished, #iter = 10639
C = 1.873700
obj = 3374.687991, rho = -0.294802
nSV = 7028, nBSV = 466
.........*
optimization finished, #iter = 9816
C = 1.626149
obj = 2937.502954, rho = -0.281922
nSV = 6752, nBSV = 626
.........*
optimization finished, #iter = 9273
C = 1.406271
obj = 2576.353944, rho = -0.176354
nSV = 6593, nBSV = 940
......*
optimization finished, #iter = 6569
C = 0.801415
obj = 1296.801867, rho = -0.279008
nSV = 5566, nBSV = 1914
.......*
optimization finished, #iter = 7178
C = 0.784839
obj = 1372.901620, rho = -0.129376
nSV = 5946, nBSV = 1998
.....*
optimization finished, #iter = 5804
C = 0.712190
obj = 1100.163805, rho = -0.281573
nSV = 5235, nBSV = 2060
.......*
optimization finished, #iter = 7746
C = 0.699238
obj = 1438.793268, rho = 0.127067
nSV = 7070, nBSV = 2708
.......*
optimization finished, #iter = 7200
C = 2.326902
obj = 2735.638171, rho = -0.021450
nSV = 4689, nBSV = 3
.......*
optimization finished, #iter = 7845
C = 2.123547
obj = 2641.054803, rho = 0.064896
nSV = 4939, nBSV = 17
......*
optimization finished, #iter = 6366
C = 1.439193
obj = 1686.639978, rho = -0.034043
nSV = 4463, nBSV = 270
......*
optimization finished, #iter = 6860
C = 1.262909
obj = 1611.918427, rho = 0.101039
nSV = 4823, nBSV = 505
.....*
optimization finished, #iter = 5710
C = 1.267520
obj = 1402.515370, rho = -0.003011
nSV = 4153, nBSV = 338
......*
optimization finished, #iter = 6712
C = 0.840687
obj = 1255.692341, rho = 0.384719
nSV = 5545, nBSV = 1638
.......*
optimization finished, #iter = 7906
C = 2.357481
obj = 3012.501338, rho = 0.094116
nSV = 5102, nBSV = 4
......*
optimization finished, #iter = 6915
C = 1.660735
obj = 2023.137506, rho = -0.054381
nSV = 4736, nBSV = 137
.......*
optimization finished, #iter = 7499
C = 1.446394
obj = 1924.123345, rho = 0.080891
nSV = 5095, nBSV = 328
......*
optimization finished, #iter = 6126
C = 1.401072
obj = 1609.874692, rho = -0.024211
nSV = 4335, nBSV = 266
.......*
optimization finished, #iter = 7371
C = 0.965383
obj = 1528.219746, rho = 0.339050
nSV = 5770, nBSV = 1431
.......*
optimization finished, #iter = 7506
C = 1.829153
obj = 2363.456723, rho = -0.160545
nSV = 5091, nBSV = 75
........*
optimization finished, #iter = 8050
C = 1.598838
obj = 2253.899881, rho = -0.007228
nSV = 5439, nBSV = 210
......*
optimization finished, #iter = 6791
C = 1.483856
obj = 1809.852492, rho = -0.117972
nSV = 4679, nBSV = 235
........*
optimization finished, #iter = 8153
C = 1.092947
obj = 1863.465955, rho = 0.246807
nSV = 6165, nBSV = 1212
........*
optimization finished, #iter = 8870
C = 2.329104
obj = 3204.665480, rho = 0.157410
nSV = 5496, nBSV = 3
.......*
optimization finished, #iter = 7166
C = 2.120370
obj = 2518.132400, rho = -0.003647
nSV = 4730, nBSV = 30
.........*
optimization finished, #iter = 9389
C = 1.541375
obj = 2651.112535, rho = 0.372311
nSV = 6538, nBSV = 740
........*
optimization finished, #iter = 8691
C = 2.335305
obj = 3076.598453, rho = -0.145412
nSV = 5267, nBSV = 10
...........*
optimization finished, #iter = 11806
C = 1.958312
obj = 3742.370511, rho = 0.264561
nSV = 7464, nBSV = 243
..........*
optimization finished, #iter = 10807
C = 2.010412
obj = 3472.032148, rho = 0.400780
nSV = 6855, nBSV = 261
Total nSV = 24683
[LibSVM]train time: 726.955s
test time:  359.606s
accuracy:   0.423

================================================================================
Classifier.PASSIVE_AGGRESSIVE_CLASSIFIER
________________________________________________________________________________
Training: 
PassiveAggressiveClassifier(C=1.0, average=False, class_weight=None,
                            early_stopping=False, fit_intercept=True,
                            loss='hinge', max_iter=1000, n_iter_no_change=5,
                            n_jobs=-1, random_state=0, shuffle=True, tol=0.001,
                            validation_fraction=0.1, verbose=True,
                            warm_start=False)
-- Epoch 1
-- Epoch 1
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 12 concurrent workers.
-- Epoch 1
-- Epoch 1
-- Epoch 1
-- Epoch 1
-- Epoch 1
Norm: 45.94, NNZs: 56463, Bias: -1.007615, T: 25000, Avg. loss: 0.281790
Total training time: 0.04 seconds.
-- Epoch 2
-- Epoch 1
Norm: 67.55, NNZs: 56155, Bias: -0.569719, T: 25000, Avg. loss: 0.403276
Total training time: 0.04 seconds.
-- Epoch 2
Norm: 48.51, NNZs: 56734, Bias: -1.144661, T: 25000, Avg. loss: 0.283122
Total training time: 0.05 seconds.
-- Epoch 2
Norm: 66.69, NNZs: 54100, Bias: -0.661343, T: 25000, Avg. loss: 0.375045
Total training time: 0.06 seconds.
-- Epoch 2
Norm: 52.82, NNZs: 58416, Bias: -1.047044, T: 25000, Avg. loss: 0.339288
Total training time: 0.07 seconds.
-- Epoch 2
Norm: 44.18, NNZs: 54761, Bias: -1.035598, T: 25000, Avg. loss: 0.263862
Total training time: 0.10 seconds.
-- Epoch 2
Norm: 44.37, NNZs: 53609, Bias: -0.984532, T: 25000, Avg. loss: 0.266433
Total training time: 0.06 seconds.
-- Epoch 2
Norm: 75.86, NNZs: 60536, Bias: -1.028257, T: 50000, Avg. loss: 0.211677
Total training time: 0.10 seconds.
-- Epoch 3
Norm: 50.97, NNZs: 57829, Bias: -0.905457, T: 25000, Avg. loss: 0.304023
Total training time: 0.06 seconds.
-- Epoch 2
Norm: 94.48, NNZs: 59098, Bias: -0.697991, T: 50000, Avg. loss: 0.250326
Total training time: 0.10 seconds.
-- Epoch 3
Norm: 76.81, NNZs: 60280, Bias: -1.073611, T: 50000, Avg. loss: 0.203767
Total training time: 0.12 seconds.
-- Epoch 3
Norm: 92.40, NNZs: 56598, Bias: -0.661227, T: 50000, Avg. loss: 0.225275
Total training time: 0.10 seconds.
-- Epoch 3
Norm: 96.67, NNZs: 61541, Bias: -1.190784, T: 75000, Avg. loss: 0.151132
Total training time: 0.13 seconds.
-- Epoch 4
Norm: 83.65, NNZs: 61895, Bias: -1.010183, T: 50000, Avg. loss: 0.246052
Total training time: 0.11 seconds.
-- Epoch 3
Norm: 73.53, NNZs: 57920, Bias: -0.938501, T: 50000, Avg. loss: 0.202666
Total training time: 0.10 seconds.
-- Epoch 3
Norm: 72.81, NNZs: 58568, Bias: -1.125855, T: 50000, Avg. loss: 0.199903
Total training time: 0.14 seconds.
-- Epoch 3
Norm: 81.18, NNZs: 61643, Bias: -0.960147, T: 50000, Avg. loss: 0.217893
Total training time: 0.09 seconds.
-- Epoch 3
Norm: 114.38, NNZs: 59904, Bias: -0.655942, T: 75000, Avg. loss: 0.178840
Total training time: 0.13 seconds.
-- Epoch 4
Norm: 96.66, NNZs: 61367, Bias: -1.198449, T: 75000, Avg. loss: 0.144352
Total training time: 0.14 seconds.
-- Epoch 4
Norm: 111.30, NNZs: 57594, Bias: -0.611261, T: 75000, Avg. loss: 0.159518
Total training time: 0.12 seconds.
-- Epoch 4
Norm: 111.88, NNZs: 61937, Bias: -1.172707, T: 100000, Avg. loss: 0.108835
Total training time: 0.14 seconds.
-- Epoch 5
Norm: 105.47, NNZs: 62876, Bias: -1.069766, T: 75000, Avg. loss: 0.176436
Total training time: 0.13 seconds.
-- Epoch 4
Norm: 129.37, NNZs: 60293, Bias: -0.645908, T: 100000, Avg. loss: 0.132230
Total training time: 0.14 seconds.
-- Epoch 5
Norm: 93.93, NNZs: 59112, Bias: -1.049628, T: 75000, Avg. loss: 0.146813
Total training time: 0.11 seconds.
-- Epoch 4
Norm: 93.45, NNZs: 59685, Bias: -1.158776, T: 75000, Avg. loss: 0.145836
Total training time: 0.15 seconds.
-- Epoch 4
Norm: 101.26, NNZs: 62604, Bias: -1.042872, T: 75000, Avg. loss: 0.149572
Total training time: 0.10 seconds.
-- Epoch 4
Norm: 125.24, NNZs: 57990, Bias: -0.688941, T: 100000, Avg. loss: 0.115627
Total training time: 0.13 seconds.
-- Epoch 5
Norm: 111.11, NNZs: 61716, Bias: -1.259510, T: 100000, Avg. loss: 0.103871
Total training time: 0.15 seconds.
-- Epoch 5
Norm: 141.01, NNZs: 60520, Bias: -0.680312, T: 125000, Avg. loss: 0.100588
Total training time: 0.15 seconds.
-- Epoch 6
Norm: 123.01, NNZs: 62161, Bias: -1.210542, T: 125000, Avg. loss: 0.078914
Total training time: 0.16 seconds.
-- Epoch 6
Norm: 121.52, NNZs: 63256, Bias: -1.086505, T: 100000, Avg. loss: 0.129288
Total training time: 0.14 seconds.
-- Epoch 5
Norm: 135.93, NNZs: 58223, Bias: -0.715602, T: 125000, Avg. loss: 0.085999
Total training time: 0.14 seconds.
-- Epoch 6
Norm: 109.01, NNZs: 59662, Bias: -1.048799, T: 100000, Avg. loss: 0.106225
Total training time: 0.12 seconds.
-- Epoch 5
Norm: 108.46, NNZs: 60182, Bias: -1.183592, T: 100000, Avg. loss: 0.105009
Total training time: 0.17 seconds.
Norm: 115.74, NNZs: 62990, Bias: -1.001357, T: 100000, Avg. loss: 0.106194
Total training time: 0.11 seconds.
-- Epoch 5
-- Epoch 5
Norm: 121.90, NNZs: 61876, Bias: -1.329958, T: 125000, Avg. loss: 0.076225
Total training time: 0.16 seconds.
-- Epoch 6
Norm: 150.04, NNZs: 60639, Bias: -0.741573, T: 150000, Avg. loss: 0.076718
Total training time: 0.16 seconds.
-- Epoch 7
Norm: 144.22, NNZs: 58339, Bias: -0.741086, T: 150000, Avg. loss: 0.065265
Total training time: 0.15 seconds.
-- Epoch 7
Norm: 131.40, NNZs: 62264, Bias: -1.266068, T: 150000, Avg. loss: 0.058550
Total training time: 0.17 seconds.
-- Epoch 7
Norm: 133.48, NNZs: 63507, Bias: -1.175799, T: 125000, Avg. loss: 0.096032
Total training time: 0.16 seconds.
Norm: 120.08, NNZs: 59885, Bias: -1.121855, T: 125000, Avg. loss: 0.077664
Total training time: 0.14 seconds.
-- Epoch 6
-- Epoch 6
Norm: 126.58, NNZs: 63187, Bias: -1.079443, T: 125000, Avg. loss: 0.077796
Total training time: 0.13 seconds.
Norm: 119.37, NNZs: 60382, Bias: -1.281028, T: 125000, Avg. loss: 0.075526
Total training time: 0.18 seconds.
-- Epoch 6
-- Epoch 6
Norm: 157.22, NNZs: 60727, Bias: -0.721577, T: 175000, Avg. loss: 0.059681
Total training time: 0.17 seconds.
-- Epoch 8
Norm: 129.89, NNZs: 61974, Bias: -1.375051, T: 150000, Avg. loss: 0.055249
Total training time: 0.18 seconds.
-- Epoch 7
Norm: 150.63, NNZs: 58391, Bias: -0.693349, T: 175000, Avg. loss: 0.049350
Total training time: 0.16 seconds.
-- Epoch 8
Norm: 137.59, NNZs: 62314, Bias: -1.344487, T: 175000, Avg. loss: 0.042798
Total training time: 0.19 seconds.
-- Epoch 8
Norm: 162.89, NNZs: 60777, Bias: -0.760201, T: 200000, Avg. loss: 0.046528
Total training time: 0.18 seconds.
-- Epoch 9
Norm: 128.46, NNZs: 60009, Bias: -1.208620, T: 150000, Avg. loss: 0.057727
Total training time: 0.15 seconds.
-- Epoch 7
Norm: 142.81, NNZs: 63603, Bias: -1.173085, T: 150000, Avg. loss: 0.072149
Total training time: 0.17 seconds.
-- Epoch 7
Norm: 134.51, NNZs: 63272, Bias: -1.148457, T: 150000, Avg. loss: 0.055759
Total training time: 0.14 seconds.
-- Epoch 7
Norm: 127.70, NNZs: 60475, Bias: -1.295125, T: 150000, Avg. loss: 0.055920
Total training time: 0.20 seconds.
-- Epoch 7
Norm: 155.47, NNZs: 58433, Bias: -0.751235, T: 200000, Avg. loss: 0.037036
Total training time: 0.17 seconds.
-- Epoch 9
Norm: 135.95, NNZs: 62030, Bias: -1.428890, T: 175000, Avg. loss: 0.041091
Total training time: 0.19 seconds.
-- Epoch 8
Norm: 167.35, NNZs: 60798, Bias: -0.775682, T: 225000, Avg. loss: 0.036210
Total training time: 0.19 seconds.
-- Epoch 10
Norm: 142.29, NNZs: 62346, Bias: -1.362235, T: 200000, Avg. loss: 0.031691
Total training time: 0.20 seconds.
-- Epoch 9
Norm: 134.74, NNZs: 60069, Bias: -1.234996, T: 175000, Avg. loss: 0.042397
Total training time: 0.17 seconds.
-- Epoch 8
Norm: 149.84, NNZs: 63650, Bias: -1.234977, T: 175000, Avg. loss: 0.054087
Total training time: 0.19 seconds.
-- Epoch 8
Norm: 140.45, NNZs: 63326, Bias: -1.148691, T: 175000, Avg. loss: 0.040761
Total training time: 0.16 seconds.
-- Epoch 8
Norm: 133.78, NNZs: 60559, Bias: -1.316626, T: 175000, Avg. loss: 0.040656
Total training time: 0.21 seconds.
-- Epoch 8
Norm: 159.34, NNZs: 58471, Bias: -0.767768, T: 225000, Avg. loss: 0.029169
Total training time: 0.19 seconds.
-- Epoch 10
Norm: 140.61, NNZs: 62048, Bias: -1.444587, T: 200000, Avg. loss: 0.030934
Total training time: 0.21 seconds.
-- Epoch 9
Norm: 170.84, NNZs: 60821, Bias: -0.788623, T: 250000, Avg. loss: 0.028126
Total training time: 0.20 seconds.
-- Epoch 11
Norm: 145.91, NNZs: 62375, Bias: -1.387993, T: 225000, Avg. loss: 0.024160
Total training time: 0.22 seconds.
-- Epoch 10
Norm: 144.85, NNZs: 63339, Bias: -1.178205, T: 200000, Avg. loss: 0.029855
Total training time: 0.17 seconds.
-- Epoch 9
Norm: 139.41, NNZs: 60122, Bias: -1.251294, T: 200000, Avg. loss: 0.031100
Total training time: 0.19 seconds.
-- Epoch 9
Norm: 155.38, NNZs: 63666, Bias: -1.246735, T: 200000, Avg. loss: 0.041632
Total training time: 0.21 seconds.
-- Epoch 9
Norm: 144.00, NNZs: 62070, Bias: -1.467357, T: 225000, Avg. loss: 0.022224
Total training time: 0.22 seconds.
-- Epoch 10
Norm: 138.38, NNZs: 60586, Bias: -1.324314, T: 200000, Avg. loss: 0.030027
Total training time: 0.23 seconds.
-- Epoch 9
Norm: 173.71, NNZs: 60827, Bias: -0.779406, T: 275000, Avg. loss: 0.022679
Total training time: 0.22 seconds.
-- Epoch 12
Norm: 162.29, NNZs: 58487, Bias: -0.758277, T: 250000, Avg. loss: 0.021995
Total training time: 0.20 seconds.
-- Epoch 11
Norm: 148.08, NNZs: 63372, Bias: -1.189445, T: 225000, Avg. loss: 0.021709
Total training time: 0.19 seconds.
-- Epoch 10
Norm: 148.54, NNZs: 62379, Bias: -1.365987, T: 250000, Avg. loss: 0.017219
Total training time: 0.24 seconds.
-- Epoch 11
Norm: 146.50, NNZs: 62082, Bias: -1.486625, T: 250000, Avg. loss: 0.016214
Total training time: 0.24 seconds.
-- Epoch 11
Norm: 176.00, NNZs: 60834, Bias: -0.801260, T: 300000, Avg. loss: 0.018039
Total training time: 0.23 seconds.
-- Epoch 13
Norm: 142.95, NNZs: 60136, Bias: -1.253382, T: 225000, Avg. loss: 0.023281
Total training time: 0.20 seconds.
-- Epoch 10
Norm: 159.54, NNZs: 63672, Bias: -1.246878, T: 225000, Avg. loss: 0.030871
Total training time: 0.22 seconds.
-- Epoch 10
Norm: 141.83, NNZs: 60606, Bias: -1.355584, T: 225000, Avg. loss: 0.022404
Total training time: 0.25 seconds.
-- Epoch 10
Norm: 164.57, NNZs: 58489, Bias: -0.771938, T: 275000, Avg. loss: 0.016963
Total training time: 0.22 seconds.
-- Epoch 12
Norm: 150.47, NNZs: 63385, Bias: -1.192607, T: 250000, Avg. loss: 0.015845
Total training time: 0.20 seconds.
-- Epoch 11
Norm: 148.38, NNZs: 62094, Bias: -1.488577, T: 275000, Avg. loss: 0.011993
Total training time: 0.25 seconds.
-- Epoch 12
Norm: 177.86, NNZs: 60840, Bias: -0.809747, T: 325000, Avg. loss: 0.014505
Total training time: 0.24 seconds.
-- Epoch 14
Norm: 150.53, NNZs: 62388, Bias: -1.399973, T: 275000, Avg. loss: 0.013217
Total training time: 0.25 seconds.
-- Epoch 12
Norm: 145.62, NNZs: 60158, Bias: -1.260868, T: 250000, Avg. loss: 0.017391
Total training time: 0.22 seconds.
-- Epoch 11
Norm: 152.23, NNZs: 63390, Bias: -1.206707, T: 275000, Avg. loss: 0.011627
Total training time: 0.21 seconds.
Norm: 162.72, NNZs: 63683, Bias: -1.280622, T: 250000, Avg. loss: 0.023558
Total training time: 0.24 seconds.
-- Epoch 12
-- Epoch 11
Norm: 144.47, NNZs: 60612, Bias: -1.377214, T: 250000, Avg. loss: 0.016966
Total training time: 0.27 seconds.
Norm: 166.44, NNZs: 58491, Bias: -0.761781, T: 300000, Avg. loss: 0.013690
Total training time: 0.24 seconds.
-- Epoch 11
-- Epoch 13
Norm: 149.81, NNZs: 62100, Bias: -1.495285, T: 300000, Avg. loss: 0.009079
Total training time: 0.26 seconds.
-- Epoch 13
Norm: 179.34, NNZs: 60847, Bias: -0.806133, T: 350000, Avg. loss: 0.011388
Total training time: 0.26 seconds.
-- Epoch 15
Norm: 152.00, NNZs: 62389, Bias: -1.410984, T: 300000, Avg. loss: 0.009708
Total training time: 0.27 seconds.
-- Epoch 13
Norm: 153.58, NNZs: 63393, Bias: -1.213941, T: 300000, Avg. loss: 0.008890
Total training time: 0.22 seconds.
-- Epoch 13
Norm: 147.65, NNZs: 60166, Bias: -1.275404, T: 275000, Avg. loss: 0.013202
Total training time: 0.24 seconds.
-- Epoch 12
Norm: 150.81, NNZs: 62100, Bias: -1.508151, T: 325000, Avg. loss: 0.006373
Total training time: 0.27 seconds.
-- Epoch 14
Norm: 165.17, NNZs: 63691, Bias: -1.269310, T: 275000, Avg. loss: 0.017856
Total training time: 0.26 seconds.
-- Epoch 12
Norm: 167.88, NNZs: 58497, Bias: -0.770383, T: 325000, Avg. loss: 0.010576
Total training time: 0.25 seconds.
-- Epoch 14
Norm: 180.51, NNZs: 60850, Bias: -0.817402, T: 375000, Avg. loss: 0.008972
Total training time: 0.27 seconds.
Norm: 146.40, NNZs: 60619, Bias: -1.373181, T: 275000, Avg. loss: 0.012232
Total training time: 0.28 seconds.
-- Epoch 16
-- Epoch 12
Norm: 154.54, NNZs: 63397, Bias: -1.218618, T: 325000, Avg. loss: 0.006359
Total training time: 0.24 seconds.
-- Epoch 14
Norm: 153.09, NNZs: 62389, Bias: -1.413208, T: 325000, Avg. loss: 0.007174
Total training time: 0.29 seconds.
-- Epoch 14
Norm: 151.58, NNZs: 62107, Bias: -1.511959, T: 350000, Avg. loss: 0.004799
Total training time: 0.29 seconds.
-- Epoch 15
Norm: 149.16, NNZs: 60172, Bias: -1.280567, T: 300000, Avg. loss: 0.009733
Total training time: 0.25 seconds.
-- Epoch 13
Norm: 181.44, NNZs: 60854, Bias: -0.811399, T: 400000, Avg. loss: 0.007060
Total training time: 0.28 seconds.
-- Epoch 17
Norm: 169.10, NNZs: 58500, Bias: -0.763403, T: 350000, Avg. loss: 0.008807
Total training time: 0.27 seconds.
Norm: 167.10, NNZs: 63697, Bias: -1.299269, T: 300000, Avg. loss: 0.014122
Total training time: 0.28 seconds.
-- Epoch 15
-- Epoch 13
Norm: 147.85, NNZs: 60622, Bias: -1.387326, T: 300000, Avg. loss: 0.009203
Total training time: 0.30 seconds.
-- Epoch 13
Norm: 155.28, NNZs: 63397, Bias: -1.218586, T: 350000, Avg. loss: 0.004820
Total training time: 0.25 seconds.
-- Epoch 15
Norm: 152.15, NNZs: 62114, Bias: -1.513443, T: 375000, Avg. loss: 0.003586
Total training time: 0.30 seconds.
-- Epoch 16
Norm: 153.97, NNZs: 62395, Bias: -1.426439, T: 350000, Avg. loss: 0.005821
Total training time: 0.30 seconds.
-- Epoch 15
Norm: 182.18, NNZs: 60855, Bias: -0.819656, T: 425000, Avg. loss: 0.005631
Total training time: 0.29 seconds.
-- Epoch 18
Norm: 150.33, NNZs: 60185, Bias: -1.299024, T: 325000, Avg. loss: 0.007606
Total training time: 0.27 seconds.
-- Epoch 14
Norm: 155.84, NNZs: 63397, Bias: -1.220694, T: 375000, Avg. loss: 0.003712
Total training time: 0.26 seconds.
-- Epoch 16
Norm: 170.03, NNZs: 58505, Bias: -0.774497, T: 375000, Avg. loss: 0.006787
Total training time: 0.28 seconds.
-- Epoch 16
Norm: 168.59, NNZs: 63702, Bias: -1.304289, T: 325000, Avg. loss: 0.010705
Total training time: 0.29 seconds.
-- Epoch 14
Norm: 148.93, NNZs: 60630, Bias: -1.399331, T: 325000, Avg. loss: 0.006859
Total training time: 0.32 seconds.
-- Epoch 14
Norm: 152.57, NNZs: 62114, Bias: -1.517006, T: 400000, Avg. loss: 0.002630
Total training time: 0.31 seconds.
-- Epoch 17
Norm: 182.75, NNZs: 60855, Bias: -0.822961, T: 450000, Avg. loss: 0.004338
Total training time: 0.31 seconds.
-- Epoch 19
Norm: 154.64, NNZs: 62398, Bias: -1.425536, T: 375000, Avg. loss: 0.004363
Total training time: 0.32 seconds.
-- Epoch 16
Norm: 156.27, NNZs: 63398, Bias: -1.225491, T: 400000, Avg. loss: 0.002874
Total training time: 0.27 seconds.
-- Epoch 17
Norm: 151.21, NNZs: 60191, Bias: -1.309047, T: 350000, Avg. loss: 0.005676
Total training time: 0.29 seconds.
-- Epoch 15
Norm: 170.77, NNZs: 58506, Bias: -0.768129, T: 400000, Avg. loss: 0.005286
Total training time: 0.30 seconds.
-- Epoch 17
Norm: 152.89, NNZs: 62114, Bias: -1.517878, T: 425000, Avg. loss: 0.002007
Total training time: 0.32 seconds.
-- Epoch 18
Norm: 183.23, NNZs: 60855, Bias: -0.822736, T: 475000, Avg. loss: 0.003647
Norm: 169.73, NNZs: 63703, Bias: -1.302317, T: 350000, Avg. loss: 0.008148
Total training time: 0.32 seconds.
Total training time: 0.31 seconds.
-- Epoch 20
-- Epoch 15
Norm: 149.74, NNZs: 60633, Bias: -1.405013, T: 350000, Avg. loss: 0.005068
Total training time: 0.33 seconds.
-- Epoch 15
Norm: 156.62, NNZs: 63398, Bias: -1.223090, T: 425000, Avg. loss: 0.002293
Total training time: 0.28 seconds.
-- Epoch 18
Norm: 155.12, NNZs: 62398, Bias: -1.429033, T: 400000, Avg. loss: 0.003204
Total training time: 0.33 seconds.
-- Epoch 17
Norm: 171.36, NNZs: 58515, Bias: -0.777129, T: 425000, Avg. loss: 0.004258
Total training time: 0.31 seconds.
-- Epoch 18
Norm: 153.12, NNZs: 62114, Bias: -1.519271, T: 450000, Avg. loss: 0.001441
Total training time: 0.33 seconds.
-- Epoch 19
Norm: 183.61, NNZs: 60855, Bias: -0.824509, T: 500000, Avg. loss: 0.002829
Total training time: 0.33 seconds.
-- Epoch 21
Norm: 151.93, NNZs: 60191, Bias: -1.302366, T: 375000, Avg. loss: 0.004547
Total training time: 0.30 seconds.
-- Epoch 16
Norm: 170.62, NNZs: 63709, Bias: -1.307324, T: 375000, Avg. loss: 0.006437
Total training time: 0.32 seconds.
Norm: 156.86, NNZs: 63398, Bias: -1.229054, T: 450000, Avg. loss: 0.001676
Total training time: 0.29 seconds.
-- Epoch 16
-- Epoch 19
Norm: 150.36, NNZs: 60633, Bias: -1.405020, T: 375000, Avg. loss: 0.003842
Total training time: 0.35 seconds.
-- Epoch 16
Norm: 171.89, NNZs: 58524, Bias: -0.777365, T: 450000, Avg. loss: 0.003732
Total training time: 0.32 seconds.
-- Epoch 19
Norm: 153.30, NNZs: 62114, Bias: -1.519685, T: 475000, Avg. loss: 0.001094
Total training time: 0.34 seconds.
Norm: 155.52, NNZs: 62398, Bias: -1.430571, T: 425000, Avg. loss: 0.002665
-- Epoch 20
Total training time: 0.35 seconds.
-- Epoch 18
Norm: 183.90, NNZs: 60855, Bias: -0.827256, T: 525000, Avg. loss: 0.002267
Total training time: 0.34 seconds.
-- Epoch 22
Norm: 157.05, NNZs: 63398, Bias: -1.226084, T: 475000, Avg. loss: 0.001287
Total training time: 0.30 seconds.
Norm: 152.45, NNZs: 60191, Bias: -1.306593, T: 400000, Avg. loss: 0.003403
Total training time: 0.32 seconds.
-- Epoch 20
-- Epoch 17
Norm: 172.25, NNZs: 58525, Bias: -0.779455, T: 475000, Avg. loss: 0.002556
Total training time: 0.33 seconds.
-- Epoch 20
Norm: 171.29, NNZs: 63709, Bias: -1.309807, T: 400000, Avg. loss: 0.004820
Total training time: 0.34 seconds.
-- Epoch 17
Norm: 153.43, NNZs: 62114, Bias: -1.522205, T: 500000, Avg. loss: 0.000800
Total training time: 0.36 seconds.
Convergence after 20 epochs took 0.36 seconds
Norm: 150.84, NNZs: 60634, Bias: -1.405323, T: 400000, Avg. loss: 0.002974
Total training time: 0.36 seconds.
-- Epoch 17
Norm: 184.13, NNZs: 60863, Bias: -0.831908, T: 550000, Avg. loss: 0.001708
Total training time: 0.35 seconds.
-- Epoch 23
Norm: 155.84, NNZs: 62402, Bias: -1.431580, T: 450000, Avg. loss: 0.002121
Total training time: 0.36 seconds.
-- Epoch 19
Norm: 157.19, NNZs: 63398, Bias: -1.226327, T: 500000, Avg. loss: 0.000999
Total training time: 0.31 seconds.
Convergence after 20 epochs took 0.32 seconds
Norm: 172.57, NNZs: 58525, Bias: -0.778819, T: 500000, Avg. loss: 0.002277
Total training time: 0.34 seconds.
-- Epoch 21
Norm: 152.92, NNZs: 60197, Bias: -1.309796, T: 425000, Avg. loss: 0.002964
Total training time: 0.33 seconds.
-- Epoch 18
Norm: 184.34, NNZs: 60863, Bias: -0.826838, T: 575000, Avg. loss: 0.001632
Total training time: 0.36 seconds.
Convergence after 23 epochs took 0.36 seconds
Norm: 171.82, NNZs: 63709, Bias: -1.306075, T: 425000, Avg. loss: 0.003719
Total training time: 0.35 seconds.
-- Epoch 18
Norm: 151.21, NNZs: 60634, Bias: -1.407957, T: 425000, Avg. loss: 0.002320
Total training time: 0.38 seconds.
-- Epoch 18
Norm: 156.08, NNZs: 62402, Bias: -1.432737, T: 475000, Avg. loss: 0.001639
Total training time: 0.37 seconds.
-- Epoch 20
Norm: 172.81, NNZs: 58525, Bias: -0.782249, T: 525000, Avg. loss: 0.001683
Total training time: 0.35 seconds.
-- Epoch 22
Norm: 153.28, NNZs: 60197, Bias: -1.309382, T: 450000, Avg. loss: 0.002301
Total training time: 0.34 seconds.
-- Epoch 19
Norm: 172.23, NNZs: 63711, Bias: -1.312792, T: 450000, Avg. loss: 0.003027
Total training time: 0.36 seconds.
-- Epoch 19
[Parallel(n_jobs=-1)]: Done   3 out of   8 | elapsed:    0.4s remaining:    0.6s
Norm: 173.01, NNZs: 58528, Bias: -0.782915, T: 550000, Avg. loss: 0.001403
Total training time: 0.36 seconds.
Norm: 151.48, NNZs: 60634, Bias: -1.410524, T: 450000, Avg. loss: 0.001707
Total training time: 0.39 seconds.
-- Epoch 23
-- Epoch 19
Norm: 156.25, NNZs: 62402, Bias: -1.434435, T: 500000, Avg. loss: 0.001173
Total training time: 0.38 seconds.
-- Epoch 21
Norm: 153.54, NNZs: 60198, Bias: -1.310341, T: 475000, Avg. loss: 0.001695
Total training time: 0.35 seconds.
-- Epoch 20
Norm: 173.16, NNZs: 58528, Bias: -0.782353, T: 575000, Avg. loss: 0.001058
Total training time: 0.37 seconds.
-- Epoch 24
Norm: 172.55, NNZs: 63711, Bias: -1.311724, T: 475000, Avg. loss: 0.002312
Total training time: 0.38 seconds.
-- Epoch 20
Norm: 151.69, NNZs: 60634, Bias: -1.412248, T: 475000, Avg. loss: 0.001300
Total training time: 0.40 seconds.
-- Epoch 20
Norm: 156.41, NNZs: 62402, Bias: -1.435662, T: 525000, Avg. loss: 0.001177
Total training time: 0.40 seconds.
Convergence after 21 epochs took 0.40 seconds
Norm: 173.29, NNZs: 58528, Bias: -0.782487, T: 600000, Avg. loss: 0.000886
Total training time: 0.38 seconds.
Convergence after 24 epochs took 0.38 seconds
Norm: 153.81, NNZs: 60198, Bias: -1.308723, T: 500000, Avg. loss: 0.001701
Total training time: 0.37 seconds.
-- Epoch 21
Norm: 172.81, NNZs: 63713, Bias: -1.314728, T: 500000, Avg. loss: 0.001849
Total training time: 0.38 seconds.
-- Epoch 21
Norm: 151.87, NNZs: 60634, Bias: -1.413340, T: 500000, Avg. loss: 0.001087
Total training time: 0.41 seconds.
Convergence after 20 epochs took 0.41 seconds
Norm: 173.03, NNZs: 63713, Bias: -1.313003, T: 525000, Avg. loss: 0.001626
Total training time: 0.39 seconds.
-- Epoch 22
Norm: 154.03, NNZs: 60199, Bias: -1.312583, T: 525000, Avg. loss: 0.001406
Total training time: 0.38 seconds.
Convergence after 21 epochs took 0.38 seconds
Norm: 173.22, NNZs: 63713, Bias: -1.314154, T: 550000, Avg. loss: 0.001352
Total training time: 0.40 seconds.
Convergence after 22 epochs took 0.40 seconds
[Parallel(n_jobs=-1)]: Done   8 out of   8 | elapsed:    0.4s finished
train time: 0.508s
test time:  0.028s
accuracy:   0.331
dimensionality: 74535
density: 0.824866


================================================================================
Classifier.PERCEPTRON
________________________________________________________________________________
Training: 
Perceptron(alpha=0.0001, class_weight=None, early_stopping=False, eta0=1.0,
           fit_intercept=True, max_iter=1000, n_iter_no_change=5, n_jobs=-1,
           penalty=None, random_state=0, shuffle=True, tol=0.001,
           validation_fraction=0.1, verbose=True, warm_start=False)
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 12 concurrent workers.
-- Epoch 1
-- Epoch 1
-- Epoch 1
-- Epoch 1
-- Epoch 1
-- Epoch 1
-- Epoch 1
Norm: 44.97, NNZs: 35540, Bias: -0.440000, T: 25000, Avg. loss: 0.038791
Total training time: 0.06 seconds.
-- Epoch 2
Norm: 43.54, NNZs: 33732, Bias: -0.410000, T: 25000, Avg. loss: 0.038814
Total training time: 0.04 seconds.
Norm: 48.01, NNZs: 37649, Bias: -0.450000, T: 25000, Avg. loss: 0.048343
Total training time: 0.04 seconds.
-- Epoch 2
Norm: 49.94, NNZs: 35277, Bias: -0.220000, T: 25000, Avg. loss: 0.048511
Total training time: 0.05 seconds.
-- Epoch 2
-- Epoch 2
-- Epoch 1
Norm: 45.45, NNZs: 34185, Bias: -0.430000, T: 25000, Avg. loss: 0.040543
Total training time: 0.06 seconds.
Norm: 50.73, NNZs: 37352, Bias: -0.230000, T: 25000, Avg. loss: 0.052154
Total training time: 0.04 seconds.
-- Epoch 2
-- Epoch 2
Norm: 61.37, NNZs: 40079, Bias: -0.290000, T: 50000, Avg. loss: 0.024697
Total training time: 0.07 seconds.
-- Epoch 3
Norm: 60.41, NNZs: 42788, Bias: -0.490000, T: 50000, Avg. loss: 0.025419
Total training time: 0.07 seconds.
-- Epoch 3
Norm: 55.42, NNZs: 37736, Bias: -0.510000, T: 50000, Avg. loss: 0.019377
Total training time: 0.07 seconds.
-- Epoch 3
Norm: 56.54, NNZs: 39984, Bias: -0.520000, T: 50000, Avg. loss: 0.020487
Total training time: 0.09 seconds.
-- Epoch 3
Norm: 44.05, NNZs: 31997, Bias: -0.420000, T: 25000, Avg. loss: 0.038278
Total training time: 0.06 seconds.
-- Epoch 2
Norm: 68.73, NNZs: 44762, Bias: -0.540000, T: 75000, Avg. loss: 0.013614
Total training time: 0.09 seconds.
-- Epoch 4
Norm: 68.94, NNZs: 41976, Bias: -0.290000, T: 75000, Avg. loss: 0.013021
Total training time: 0.11 seconds.
-- Epoch 4
Norm: 47.02, NNZs: 36445, Bias: -0.340000, T: 25000, Avg. loss: 0.042001
Total training time: 0.06 seconds.
Norm: 62.75, NNZs: 39523, Bias: -0.630000, T: 75000, Avg. loss: 0.009860
Total training time: 0.10 seconds.
-- Epoch 2
-- Epoch 4
Norm: 63.05, NNZs: 42238, Bias: -0.330000, T: 50000, Avg. loss: 0.029690
Total training time: 0.09 seconds.
-- Epoch 3
Norm: 57.09, NNZs: 38368, Bias: -0.570000, T: 50000, Avg. loss: 0.019603
Total training time: 0.12 seconds.
-- Epoch 3
Norm: 55.34, NNZs: 35755, Bias: -0.470000, T: 50000, Avg. loss: 0.019360
Total training time: 0.09 seconds.
-- Epoch 3
Norm: 63.45, NNZs: 41979, Bias: -0.620000, T: 75000, Avg. loss: 0.010577
Total training time: 0.13 seconds.
-- Epoch 4
Norm: 74.26, NNZs: 45888, Bias: -0.540000, T: 100000, Avg. loss: 0.007461
Total training time: 0.11 seconds.
-- Epoch 5
Norm: 73.93, NNZs: 42991, Bias: -0.290000, T: 100000, Avg. loss: 0.007164
Total training time: 0.13 seconds.
-- Epoch 5
Norm: 58.74, NNZs: 41093, Bias: -0.500000, T: 50000, Avg. loss: 0.020389
Total training time: 0.08 seconds.
-- Epoch 3
Norm: 64.49, NNZs: 40317, Bias: -0.620000, T: 75000, Avg. loss: 0.010719
Total training time: 0.13 seconds.
-- Epoch 4
Norm: 71.54, NNZs: 44332, Bias: -0.340000, T: 75000, Avg. loss: 0.015780
Total training time: 0.11 seconds.
-- Epoch 4
Norm: 62.81, NNZs: 37785, Bias: -0.540000, T: 75000, Avg. loss: 0.010493
Total training time: 0.10 seconds.
-- Epoch 4
Norm: 78.01, NNZs: 46636, Bias: -0.630000, T: 125000, Avg. loss: 0.004188
Total training time: 0.12 seconds.
-- Epoch 6
Norm: 67.20, NNZs: 40461, Bias: -0.610000, T: 100000, Avg. loss: 0.005380
Total training time: 0.12 seconds.
-- Epoch 5
Norm: 68.00, NNZs: 42900, Bias: -0.640000, T: 100000, Avg. loss: 0.005159
Total training time: 0.15 seconds.
-- Epoch 5
Norm: 77.11, NNZs: 43588, Bias: -0.370000, T: 125000, Avg. loss: 0.003832
Total training time: 0.14 seconds.
-- Epoch 6
Norm: 69.10, NNZs: 41322, Bias: -0.670000, T: 100000, Avg. loss: 0.005387
Total training time: 0.15 seconds.
Norm: 80.74, NNZs: 47022, Bias: -0.580000, T: 150000, Avg. loss: 0.002574
Total training time: 0.13 seconds.
Norm: 65.70, NNZs: 42855, Bias: -0.540000, T: 75000, Avg. loss: 0.009921
Total training time: 0.10 seconds.
-- Epoch 5
-- Epoch 7
-- Epoch 4
Norm: 77.04, NNZs: 45379, Bias: -0.290000, T: 100000, Avg. loss: 0.008503
Total training time: 0.12 seconds.
-- Epoch 5
Norm: 67.70, NNZs: 38741, Bias: -0.570000, T: 100000, Avg. loss: 0.005412
Total training time: 0.11 seconds.
-- Epoch 5
Norm: 79.49, NNZs: 43830, Bias: -0.390000, T: 150000, Avg. loss: 0.002403
Total training time: 0.15 seconds.
-- Epoch 7
Norm: 70.16, NNZs: 41117, Bias: -0.650000, T: 125000, Avg. loss: 0.002676
Total training time: 0.14 seconds.
-- Epoch 6
Norm: 70.92, NNZs: 43456, Bias: -0.710000, T: 125000, Avg. loss: 0.002823
Total training time: 0.16 seconds.
-- Epoch 6
Norm: 82.53, NNZs: 47281, Bias: -0.610000, T: 175000, Avg. loss: 0.001698
Total training time: 0.14 seconds.
-- Epoch 8
Norm: 72.22, NNZs: 41840, Bias: -0.610000, T: 125000, Avg. loss: 0.003166
Total training time: 0.16 seconds.
-- Epoch 6
Norm: 81.54, NNZs: 44093, Bias: -0.380000, T: 175000, Avg. loss: 0.001704
Total training time: 0.16 seconds.
-- Epoch 8
Norm: 70.47, NNZs: 43734, Bias: -0.500000, T: 100000, Avg. loss: 0.005069
Total training time: 0.11 seconds.
-- Epoch 5
Norm: 80.90, NNZs: 46006, Bias: -0.360000, T: 125000, Avg. loss: 0.005030
Total training time: 0.14 seconds.
-- Epoch 6
Norm: 71.12, NNZs: 39382, Bias: -0.630000, T: 125000, Avg. loss: 0.003405
Total training time: 0.13 seconds.
-- Epoch 6
Norm: 72.29, NNZs: 41480, Bias: -0.660000, T: 150000, Avg. loss: 0.001835
Total training time: 0.15 seconds.
-- Epoch 7
Norm: 83.97, NNZs: 47457, Bias: -0.650000, T: 200000, Avg. loss: 0.001088
Total training time: 0.15 seconds.
-- Epoch 9
Norm: 73.03, NNZs: 43866, Bias: -0.710000, T: 150000, Avg. loss: 0.001642
Total training time: 0.18 seconds.
-- Epoch 7
Norm: 83.07, NNZs: 44301, Bias: -0.380000, T: 200000, Avg. loss: 0.001160
Total training time: 0.17 seconds.
-- Epoch 9
Norm: 74.46, NNZs: 42170, Bias: -0.650000, T: 150000, Avg. loss: 0.001790
Total training time: 0.17 seconds.
-- Epoch 7
Norm: 73.46, NNZs: 44240, Bias: -0.590000, T: 125000, Avg. loss: 0.002559
Total training time: 0.12 seconds.
-- Epoch 6
Norm: 73.23, NNZs: 39686, Bias: -0.650000, T: 150000, Avg. loss: 0.001732
Norm: 85.17, NNZs: 47607, Bias: -0.630000, T: 225000, Avg. loss: 0.000798
Total training time: 0.16 seconds.
Total training time: 0.14 seconds.
-- Epoch 10
-- Epoch 7
Norm: 83.67, NNZs: 46408, Bias: -0.390000, T: 150000, Avg. loss: 0.002919
Total training time: 0.15 seconds.
-- Epoch 7
Norm: 73.70, NNZs: 41739, Bias: -0.680000, T: 175000, Avg. loss: 0.000874
Total training time: 0.16 seconds.
Norm: 84.18, NNZs: 44463, Bias: -0.410000, T: 225000, Avg. loss: 0.000735
Total training time: 0.18 seconds.
-- Epoch 10
-- Epoch 8
Norm: 74.44, NNZs: 44088, Bias: -0.740000, T: 175000, Avg. loss: 0.000911
Total training time: 0.19 seconds.
-- Epoch 8
Norm: 86.03, NNZs: 47676, Bias: -0.630000, T: 250000, Avg. loss: 0.000516
Total training time: 0.17 seconds.
-- Epoch 11
Norm: 76.08, NNZs: 42429, Bias: -0.690000, T: 175000, Avg. loss: 0.001230
Total training time: 0.19 seconds.
-- Epoch 8
Norm: 84.98, NNZs: 44546, Bias: -0.390000, T: 250000, Avg. loss: 0.000511
Total training time: 0.19 seconds.
-- Epoch 11
Norm: 75.82, NNZs: 44556, Bias: -0.610000, T: 150000, Avg. loss: 0.001885
Total training time: 0.14 seconds.
-- Epoch 7
Norm: 75.01, NNZs: 39969, Bias: -0.660000, T: 175000, Avg. loss: 0.001218
Total training time: 0.15 seconds.
-- Epoch 8
Norm: 85.86, NNZs: 46630, Bias: -0.380000, T: 175000, Avg. loss: 0.002064
Total training time: 0.16 seconds.
-- Epoch 8
Norm: 74.72, NNZs: 41928, Bias: -0.710000, T: 200000, Avg. loss: 0.000546
Total training time: 0.18 seconds.
-- Epoch 9
Norm: 75.59, NNZs: 44222, Bias: -0.740000, T: 200000, Avg. loss: 0.000620
Total training time: 0.20 seconds.
-- Epoch 9
Norm: 86.83, NNZs: 47773, Bias: -0.660000, T: 275000, Avg. loss: 0.000488
Total training time: 0.18 seconds.
Convergence after 11 epochs took 0.18 seconds
Norm: 85.56, NNZs: 44595, Bias: -0.400000, T: 275000, Avg. loss: 0.000297
Total training time: 0.20 seconds.
Convergence after 11 epochs took 0.20 seconds
Norm: 77.24, NNZs: 42548, Bias: -0.730000, T: 200000, Avg. loss: 0.000761
Total training time: 0.20 seconds.
-- Epoch 9
Norm: 77.38, NNZs: 44773, Bias: -0.610000, T: 175000, Avg. loss: 0.001113
Total training time: 0.15 seconds.
Norm: 76.21, NNZs: 40095, Bias: -0.640000, T: 200000, Avg. loss: 0.000627
Total training time: 0.16 seconds.
-- Epoch 8
-- Epoch 9
Norm: 87.44, NNZs: 46801, Bias: -0.390000, T: 200000, Avg. loss: 0.001291
Total training time: 0.17 seconds.
-- Epoch 9
Norm: 75.60, NNZs: 42020, Bias: -0.680000, T: 225000, Avg. loss: 0.000447
Total training time: 0.19 seconds.
-- Epoch 10
Norm: 76.42, NNZs: 44370, Bias: -0.780000, T: 225000, Avg. loss: 0.000432
Total training time: 0.21 seconds.
-- Epoch 10
Norm: 78.30, NNZs: 42677, Bias: -0.720000, T: 225000, Avg. loss: 0.000559
Total training time: 0.21 seconds.
-- Epoch 10
Norm: 78.48, NNZs: 44896, Bias: -0.620000, T: 200000, Avg. loss: 0.000607
Total training time: 0.16 seconds.
-- Epoch 9
Norm: 77.29, NNZs: 40256, Bias: -0.620000, T: 225000, Avg. loss: 0.000533
Total training time: 0.17 seconds.
-- Epoch 10
Norm: 76.29, NNZs: 42088, Bias: -0.730000, T: 250000, Avg. loss: 0.000387
Total training time: 0.20 seconds.
Convergence after 10 epochs took 0.20 seconds
Norm: 88.57, NNZs: 46920, Bias: -0.410000, T: 225000, Avg. loss: 0.000925
Total training time: 0.19 seconds.
-- Epoch 10
Norm: 77.11, NNZs: 44491, Bias: -0.770000, T: 250000, Avg. loss: 0.000348
Total training time: 0.22 seconds.
-- Epoch 11
Norm: 79.41, NNZs: 45032, Bias: -0.630000, T: 225000, Avg. loss: 0.000565
Total training time: 0.17 seconds.
-- Epoch 10
Norm: 78.96, NNZs: 42809, Bias: -0.730000, T: 250000, Avg. loss: 0.000308
Total training time: 0.22 seconds.
-- Epoch 11
Norm: 77.94, NNZs: 40385, Bias: -0.670000, T: 250000, Avg. loss: 0.000317
Total training time: 0.18 seconds.
-- Epoch 11
Norm: 89.73, NNZs: 47058, Bias: -0.420000, T: 250000, Avg. loss: 0.000909
Total training time: 0.20 seconds.
-- Epoch 11
Norm: 79.98, NNZs: 45076, Bias: -0.640000, T: 250000, Avg. loss: 0.000245
Total training time: 0.17 seconds.
Convergence after 10 epochs took 0.17 seconds
Norm: 77.68, NNZs: 44578, Bias: -0.740000, T: 275000, Avg. loss: 0.000245
Total training time: 0.23 seconds.
Convergence after 11 epochs took 0.23 seconds
[Parallel(n_jobs=-1)]: Done   3 out of   8 | elapsed:    0.2s remaining:    0.4s
[Parallel(n_jobs=-1)]: Done   8 out of   8 | elapsed:    0.2s finished
Norm: 79.49, NNZs: 42923, Bias: -0.740000, T: 275000, Avg. loss: 0.000254
Total training time: 0.23 seconds.
Convergence after 11 epochs took 0.23 seconds
Norm: 90.55, NNZs: 47146, Bias: -0.440000, T: 275000, Avg. loss: 0.000549
Total training time: 0.20 seconds.
Norm: 78.63, NNZs: 40488, Bias: -0.650000, T: 275000, Avg. loss: 0.000303
Total training time: 0.19 seconds.
Convergence after 11 epochs took 0.20 seconds
Convergence after 11 epochs took 0.19 seconds
train time: 0.314s
test time:  0.027s
accuracy:   0.327
dimensionality: 74535
density: 0.594799


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
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 12 concurrent workers.
[Parallel(n_jobs=-1)]: Done  26 tasks      | elapsed:    3.7s
[Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:   10.6s finished
train time: 10.693s
[Parallel(n_jobs=12)]: Using backend ThreadingBackend with 12 concurrent workers.
[Parallel(n_jobs=12)]: Done  26 tasks      | elapsed:    0.1s
[Parallel(n_jobs=12)]: Done 100 out of 100 | elapsed:    0.4s finished
test time:  0.417s
accuracy:   0.375

================================================================================
Classifier.RIDGE_CLASSIFIER
________________________________________________________________________________
Training: 
RidgeClassifier(alpha=1.0, class_weight=None, copy_X=True, fit_intercept=True,
                max_iter=None, normalize=False, random_state=0, solver='auto',
                tol=0.001)
train time: 3.194s
test time:  0.043s
accuracy:   0.387
dimensionality: 74535
density: 1.000000


================================================================================
Classifier.RIDGE_CLASSIFIERCV
________________________________________________________________________________
Training: 
RidgeClassifierCV(alphas=array([ 0.1,  1. , 10. ]), class_weight=None, cv=None,
                  fit_intercept=True, normalize=False, scoring=None,
                  store_cv_values=False)
train time: 1730.003s
test time:  0.037s
accuracy:   0.415
dimensionality: 74535
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
              validation_fraction=0.1, verbose=True, warm_start=False)
[Parallel(n_jobs=-1)]: Using backend ThreadingBackend with 12 concurrent workers.
-- Epoch 1
-- Epoch 1
-- Epoch 1
-- Epoch 1
-- Epoch 1
-- Epoch 1
-- Epoch 1
-- Epoch 1
Norm: 20.97, NNZs: 39560, Bias: -1.288235, T: 25000, Avg. loss: 0.242676
Total training time: 0.04 seconds.
-- Epoch 2
Norm: 37.80, NNZs: 45064, Bias: -1.031304, T: 25000, Avg. loss: 0.359016
Total training time: 0.07 seconds.
-- Epoch 2
Norm: 19.42, NNZs: 34620, Bias: -1.202955, T: 25000, Avg. loss: 0.224105
Total training time: 0.06 seconds.
-- Epoch 2
Norm: 19.25, NNZs: 36452, Bias: -1.236809, T: 25000, Avg. loss: 0.221792
Total training time: 0.05 seconds.
Norm: 22.53, NNZs: 42409, Bias: -1.244643, T: 25000, Avg. loss: 0.293944
Total training time: 0.05 seconds.
-- Epoch 2
-- Epoch 2
Norm: 19.90, NNZs: 37106, Bias: -1.210876, T: 25000, Avg. loss: 0.238195
Total training time: 0.06 seconds.
-- Epoch 2
Norm: 35.88, NNZs: 46633, Bias: -0.953820, T: 25000, Avg. loss: 0.382406
Total training time: 0.04 seconds.
-- Epoch 2
Norm: 21.82, NNZs: 40388, Bias: -1.189974, T: 25000, Avg. loss: 0.261148
Total training time: 0.07 seconds.
-- Epoch 2
Norm: 35.56, NNZs: 49389, Bias: -0.966569, T: 50000, Avg. loss: 0.278240
Total training time: 0.10 seconds.
-- Epoch 3
Norm: 16.89, NNZs: 43692, Bias: -1.129397, T: 50000, Avg. loss: 0.193947
Total training time: 0.09 seconds.
-- Epoch 3
Norm: 17.89, NNZs: 45875, Bias: -1.169491, T: 50000, Avg. loss: 0.197574
Total training time: 0.09 seconds.
-- Epoch 3
Norm: 34.03, NNZs: 51297, Bias: -0.966949, T: 50000, Avg. loss: 0.302968
Total training time: 0.08 seconds.
-- Epoch 3
Norm: 19.32, NNZs: 48740, Bias: -1.142484, T: 50000, Avg. loss: 0.239033
Total training time: 0.09 seconds.
-- Epoch 3
Norm: 16.18, NNZs: 42656, Bias: -1.168513, T: 50000, Avg. loss: 0.182038
Total training time: 0.09 seconds.
-- Epoch 3
Norm: 18.74, NNZs: 46757, Bias: -1.124699, T: 50000, Avg. loss: 0.212607
Total training time: 0.10 seconds.
Norm: 16.29, NNZs: 40672, Bias: -1.106958, T: 50000, Avg. loss: 0.183598
Total training time: 0.11 seconds.
-- Epoch 3
-- Epoch 3
Norm: 15.95, NNZs: 47287, Bias: -1.117345, T: 75000, Avg. loss: 0.185723
Total training time: 0.11 seconds.
-- Epoch 4
Norm: 16.97, NNZs: 49107, Bias: -1.136762, T: 75000, Avg. loss: 0.189164
Total training time: 0.10 seconds.
-- Epoch 4
Norm: 17.84, NNZs: 50254, Bias: -1.083557, T: 75000, Avg. loss: 0.203343
Total training time: 0.11 seconds.
-- Epoch 4
Norm: 35.01, NNZs: 50653, Bias: -0.916998, T: 75000, Avg. loss: 0.263130
Total training time: 0.13 seconds.
-- Epoch 4
Norm: 33.46, NNZs: 52608, Bias: -0.912504, T: 75000, Avg. loss: 0.286642
Total training time: 0.10 seconds.
-- Epoch 4
Norm: 18.38, NNZs: 51816, Bias: -1.101954, T: 75000, Avg. loss: 0.228629
Total training time: 0.11 seconds.
-- Epoch 4
Norm: 15.27, NNZs: 46281, Bias: -1.116344, T: 75000, Avg. loss: 0.174232
Total training time: 0.10 seconds.
-- Epoch 4
Norm: 15.31, NNZs: 44599, Bias: -1.088582, T: 75000, Avg. loss: 0.176059
Total training time: 0.12 seconds.
-- Epoch 4
Norm: 16.58, NNZs: 50951, Bias: -1.122044, T: 100000, Avg. loss: 0.185099
Total training time: 0.11 seconds.
-- Epoch 5
Norm: 15.54, NNZs: 49379, Bias: -1.088179, T: 100000, Avg. loss: 0.181935
Total training time: 0.12 seconds.
-- Epoch 5
Norm: 17.42, NNZs: 52047, Bias: -1.068258, T: 100000, Avg. loss: 0.199313
Total training time: 0.12 seconds.
-- Epoch 5
Norm: 16.34, NNZs: 52050, Bias: -1.112791, T: 125000, Avg. loss: 0.183150
Total training time: 0.12 seconds.
-- Epoch 6
Norm: 34.83, NNZs: 51283, Bias: -0.910122, T: 100000, Avg. loss: 0.256037
Total training time: 0.14 seconds.
Norm: 33.30, NNZs: 53217, Bias: -0.895458, T: 100000, Avg. loss: 0.279367
-- Epoch 5
Total training time: 0.11 seconds.
-- Epoch 5
Norm: 14.86, NNZs: 48303, Bias: -1.096527, T: 100000, Avg. loss: 0.170799
Total training time: 0.12 seconds.
Norm: 17.92, NNZs: 53522, Bias: -1.085708, T: 100000, Avg. loss: 0.224193
Total training time: 0.12 seconds.
-- Epoch 5
-- Epoch 5
Norm: 14.88, NNZs: 46677, Bias: -1.064913, T: 100000, Avg. loss: 0.172547
Total training time: 0.14 seconds.
-- Epoch 5
Norm: 17.18, NNZs: 53210, Bias: -1.058028, T: 125000, Avg. loss: 0.197084
Total training time: 0.13 seconds.
-- Epoch 6
Norm: 15.30, NNZs: 50650, Bias: -1.075253, T: 125000, Avg. loss: 0.179979
Total training time: 0.14 seconds.
-- Epoch 6
Norm: 16.20, NNZs: 52716, Bias: -1.105182, T: 150000, Avg. loss: 0.181708
Total training time: 0.13 seconds.
-- Epoch 7
Norm: 14.63, NNZs: 49689, Bias: -1.092281, T: 125000, Avg. loss: 0.168742
Total training time: 0.13 seconds.
-- Epoch 6
Norm: 17.68, NNZs: 54530, Bias: -1.075375, T: 125000, Avg. loss: 0.221511
Total training time: 0.13 seconds.
Norm: 33.23, NNZs: 53597, Bias: -0.876839, T: 125000, Avg. loss: 0.275328
Total training time: 0.13 seconds.
-- Epoch 6
-- Epoch 6
Norm: 34.80, NNZs: 51595, Bias: -0.912543, T: 125000, Avg. loss: 0.252077
Total training time: 0.16 seconds.
-- Epoch 6
Norm: 17.04, NNZs: 53938, Bias: -1.058344, T: 150000, Avg. loss: 0.195492
Total training time: 0.14 seconds.
-- Epoch 7
Norm: 14.65, NNZs: 47987, Bias: -1.051743, T: 125000, Avg. loss: 0.170533
Total training time: 0.15 seconds.
-- Epoch 6
Norm: 16.10, NNZs: 53186, Bias: -1.097545, T: 175000, Avg. loss: 0.180714
Total training time: 0.14 seconds.
-- Epoch 8
Norm: 15.15, NNZs: 51760, Bias: -1.069981, T: 150000, Avg. loss: 0.178677
Total training time: 0.15 seconds.
-- Epoch 7
Norm: 16.94, NNZs: 54486, Bias: -1.049751, T: 175000, Avg. loss: 0.194451
Total training time: 0.15 seconds.
-- Epoch 8
Norm: 14.50, NNZs: 50572, Bias: -1.078530, T: 150000, Avg. loss: 0.167473
Total training time: 0.14 seconds.
-- Epoch 7
Norm: 17.53, NNZs: 55205, Bias: -1.065246, T: 150000, Avg. loss: 0.219885
Total training time: 0.15 seconds.
Norm: 33.18, NNZs: 53795, Bias: -0.899601, T: 150000, Avg. loss: 0.272520
Total training time: 0.14 seconds.
-- Epoch 7
-- Epoch 7
Norm: 16.04, NNZs: 53517, Bias: -1.093736, T: 200000, Avg. loss: 0.179962
Total training time: 0.15 seconds.
-- Epoch 9
Norm: 34.77, NNZs: 51800, Bias: -0.895038, T: 150000, Avg. loss: 0.249485
Total training time: 0.17 seconds.
-- Epoch 7
Norm: 14.49, NNZs: 48987, Bias: -1.051981, T: 150000, Avg. loss: 0.169330
Total training time: 0.17 seconds.
-- Epoch 7
Norm: 15.05, NNZs: 52357, Bias: -1.061949, T: 175000, Avg. loss: 0.177709
Total training time: 0.17 seconds.
-- Epoch 8
Norm: 16.88, NNZs: 54882, Bias: -1.040805, T: 200000, Avg. loss: 0.193662
Total training time: 0.16 seconds.
-- Epoch 9
Norm: 15.99, NNZs: 53731, Bias: -1.084278, T: 225000, Avg. loss: 0.179466
Total training time: 0.16 seconds.
-- Epoch 10
Norm: 14.40, NNZs: 51153, Bias: -1.073768, T: 175000, Avg. loss: 0.166610
Total training time: 0.16 seconds.
-- Epoch 8
Norm: 17.43, NNZs: 55642, Bias: -1.056246, T: 175000, Avg. loss: 0.218636
Total training time: 0.16 seconds.
Norm: 33.19, NNZs: 53936, Bias: -0.869835, T: 175000, Avg. loss: 0.270374
Total training time: 0.15 seconds.
-- Epoch 8
-- Epoch 8
Norm: 14.38, NNZs: 49730, Bias: -1.044470, T: 175000, Avg. loss: 0.168447
Total training time: 0.18 seconds.
-- Epoch 8
Norm: 34.71, NNZs: 51956, Bias: -0.896061, T: 175000, Avg. loss: 0.247376
Total training time: 0.19 seconds.
-- Epoch 8
Norm: 16.82, NNZs: 55181, Bias: -1.040517, T: 225000, Avg. loss: 0.193106
Total training time: 0.17 seconds.
-- Epoch 10
Norm: 14.97, NNZs: 52722, Bias: -1.064636, T: 200000, Avg. loss: 0.177005
Total training time: 0.18 seconds.
-- Epoch 9
Norm: 15.95, NNZs: 53895, Bias: -1.084312, T: 250000, Avg. loss: 0.178965
Total training time: 0.17 seconds.
-- Epoch 11
Norm: 14.33, NNZs: 51599, Bias: -1.060815, T: 200000, Avg. loss: 0.165968
Total training time: 0.17 seconds.
-- Epoch 9
Norm: 16.78, NNZs: 55340, Bias: -1.036383, T: 250000, Avg. loss: 0.192577
Total training time: 0.19 seconds.
Norm: 17.35, NNZs: 55971, Bias: -1.058543, T: 200000, Avg. loss: 0.217899
Total training time: 0.18 seconds.
-- Epoch 11
-- Epoch 9
Norm: 33.18, NNZs: 54036, Bias: -0.882360, T: 200000, Avg. loss: 0.268854
Total training time: 0.17 seconds.
-- Epoch 9
Norm: 14.31, NNZs: 50125, Bias: -1.040344, T: 200000, Avg. loss: 0.167814
Total training time: 0.20 seconds.
-- Epoch 9
Norm: 34.72, NNZs: 52040, Bias: -0.882251, T: 200000, Avg. loss: 0.246145
Total training time: 0.20 seconds.
-- Epoch 9
Norm: 15.92, NNZs: 54071, Bias: -1.083258, T: 275000, Avg. loss: 0.178590
Total training time: 0.19 seconds.
Convergence after 11 epochs took 0.19 seconds
Norm: 14.92, NNZs: 53177, Bias: -1.054738, T: 225000, Avg. loss: 0.176461
Total training time: 0.19 seconds.
-- Epoch 10
Norm: 16.75, NNZs: 55521, Bias: -1.034858, T: 275000, Avg. loss: 0.192234
Total training time: 0.19 seconds.
-- Epoch 12
Norm: 14.28, NNZs: 51946, Bias: -1.064692, T: 225000, Avg. loss: 0.165450
Total training time: 0.18 seconds.
-- Epoch 10
Norm: 17.29, NNZs: 56199, Bias: -1.052896, T: 225000, Avg. loss: 0.217183
Total training time: 0.19 seconds.
-- Epoch 10
Norm: 33.16, NNZs: 54095, Bias: -0.872551, T: 225000, Avg. loss: 0.267695
Total training time: 0.18 seconds.
-- Epoch 10
Norm: 14.26, NNZs: 50450, Bias: -1.036489, T: 225000, Avg. loss: 0.167202
Total training time: 0.21 seconds.
-- Epoch 10
Norm: 34.72, NNZs: 52078, Bias: -0.892915, T: 225000, Avg. loss: 0.244895
Total training time: 0.22 seconds.
-- Epoch 10
Norm: 14.88, NNZs: 53436, Bias: -1.052637, T: 250000, Avg. loss: 0.176001
Total training time: 0.21 seconds.
-- Epoch 11
Norm: 16.73, NNZs: 55624, Bias: -1.032437, T: 300000, Avg. loss: 0.191928
Total training time: 0.20 seconds.
Convergence after 12 epochs took 0.20 seconds
Norm: 14.24, NNZs: 52172, Bias: -1.058403, T: 250000, Avg. loss: 0.165033
Total training time: 0.20 seconds.
-- Epoch 11
Norm: 17.25, NNZs: 56384, Bias: -1.054173, T: 250000, Avg. loss: 0.216611
Total training time: 0.20 seconds.
-- Epoch 11
Norm: 33.18, NNZs: 54153, Bias: -0.877506, T: 250000, Avg. loss: 0.266773
Total training time: 0.19 seconds.
Norm: 14.22, NNZs: 50695, Bias: -1.031178, T: 250000, Avg. loss: 0.166840
-- Epoch 11
Total training time: 0.22 seconds.
-- Epoch 11
Norm: 14.85, NNZs: 53639, Bias: -1.050860, T: 275000, Avg. loss: 0.175725
Total training time: 0.22 seconds.
Norm: 34.70, NNZs: 52102, Bias: -0.874113, T: 250000, Avg. loss: 0.244034
Total training time: 0.23 seconds.
Convergence after 11 epochs took 0.22 seconds
-- Epoch 11
Norm: 14.21, NNZs: 52352, Bias: -1.056002, T: 275000, Avg. loss: 0.164708
Total training time: 0.21 seconds.
Convergence after 11 epochs took 0.21 seconds
Norm: 33.20, NNZs: 54165, Bias: -0.870246, T: 275000, Avg. loss: 0.265979
Total training time: 0.21 seconds.
Norm: 17.22, NNZs: 56519, Bias: -1.042313, T: 275000, Avg. loss: 0.216166
Total training time: 0.21 seconds.
-- Epoch 12
-- Epoch 12
Norm: 14.19, NNZs: 50902, Bias: -1.030204, T: 275000, Avg. loss: 0.166506
Total training time: 0.23 seconds.
Convergence after 11 epochs took 0.23 seconds
Norm: 34.70, NNZs: 52127, Bias: -0.878077, T: 275000, Avg. loss: 0.243270
Total training time: 0.24 seconds.
-- Epoch 12
[Parallel(n_jobs=-1)]: Done   3 out of   8 | elapsed:    0.2s remaining:    0.4s
Norm: 33.20, NNZs: 54186, Bias: -0.873907, T: 300000, Avg. loss: 0.265341
Total training time: 0.21 seconds.
Norm: 17.19, NNZs: 56648, Bias: -1.044978, T: 300000, Avg. loss: 0.215868
-- Epoch 13
Total training time: 0.22 seconds.
Convergence after 12 epochs took 0.22 seconds
Norm: 34.71, NNZs: 52142, Bias: -0.879955, T: 300000, Avg. loss: 0.242794
Total training time: 0.25 seconds.
-- Epoch 13
Norm: 33.21, NNZs: 54195, Bias: -0.869171, T: 325000, Avg. loss: 0.264896
Total training time: 0.22 seconds.
-- Epoch 14
Norm: 34.70, NNZs: 52158, Bias: -0.874743, T: 325000, Avg. loss: 0.242146
Total training time: 0.25 seconds.
-- Epoch 14
Norm: 33.21, NNZs: 54229, Bias: -0.865124, T: 350000, Avg. loss: 0.264455
Total training time: 0.23 seconds.
Convergence after 14 epochs took 0.23 seconds
Norm: 34.73, NNZs: 52175, Bias: -0.874796, T: 350000, Avg. loss: 0.241781
Total training time: 0.26 seconds.
Convergence after 14 epochs took 0.26 seconds
[Parallel(n_jobs=-1)]: Done   8 out of   8 | elapsed:    0.3s finished
train time: 0.320s
test time:  0.021s
accuracy:   0.407
dimensionality: 74535
density: 0.720534


Accuracy score for the IMDB_REVIEWS dataset (Multi-class classification): Final classification report: 
1) ADA_BOOST_CLASSIFIER
		Accuracy score = 0.3586		Training time = 11.367851257324219		Test time = 0.7166264057159424

2) BERNOULLI_NB
		Accuracy score = 0.37132		Training time = 0.03876781463623047		Test time = 0.03800320625305176

3) COMPLEMENT_NB
		Accuracy score = 0.37312		Training time = 0.034267425537109375		Test time = 0.018863916397094727

4) DECISION_TREE_CLASSIFIER
		Accuracy score = 0.25764		Training time = 35.06614112854004		Test time = 0.014446496963500977

5) EXTRA_TREE_CLASSIFIER
		Accuracy score = 0.2212		Training time = 1.0077941417694092		Test time = 0.019968032836914062

6) EXTRA_TREES_CLASSIFIER
		Accuracy score = 0.37404		Training time = 15.878169536590576		Test time = 0.5103204250335693

7) GRADIENT_BOOSTING_CLASSIFIER
		Accuracy score = 0.37624		Training time = 397.786301612854		Test time = 0.25756216049194336

8) K_NEIGHBORS_CLASSIFIER
		Accuracy score = 0.26352		Training time = 0.006181240081787109		Test time = 12.872493505477905

9) LINEAR_SVC
		Accuracy score = 0.37328		Training time = 1.7689294815063477		Test time = 0.017616987228393555

10) LOGISTIC_REGRESSION
		Accuracy score = 0.42084		Training time = 9.81763243675232		Test time = 0.021485567092895508

11) LOGISTIC_REGRESSION_CV
		Accuracy score = 0.40532		Training time = 152.31824135780334		Test time = 0.03794264793395996

12) MLP_CLASSIFIER
		Accuracy score = 0.34468		Training time = 2304.3378138542175		Test time = 0.16078424453735352

13) MULTINOMIAL_NB
		Accuracy score = 0.34924		Training time = 0.06655263900756836		Test time = 0.019091367721557617

14) NEAREST_CENTROID
		Accuracy score = 0.36844		Training time = 0.023381471633911133		Test time = 0.022855281829833984

15) NU_SVC
		Accuracy score = 0.4232		Training time = 726.954843044281		Test time = 359.60646748542786

16) PASSIVE_AGGRESSIVE_CLASSIFIER
		Accuracy score = 0.33112		Training time = 0.5084974765777588		Test time = 0.028000593185424805

17) PERCEPTRON
		Accuracy score = 0.3266		Training time = 0.31379008293151855		Test time = 0.027352094650268555

18) RANDOM_FOREST_CLASSIFIER
		Accuracy score = 0.3754		Training time = 10.692729234695435		Test time = 0.4168663024902344

19) RIDGE_CLASSIFIER
		Accuracy score = 0.38716		Training time = 3.1944680213928223		Test time = 0.04255342483520508

20) RIDGE_CLASSIFIERCV
		Accuracy score = 0.4154		Training time = 1730.0031440258026		Test time = 0.03655815124511719

21) SGD_CLASSIFIER
		Accuracy score = 0.40676		Training time = 0.3198878765106201		Test time = 0.020763397216796875



Best algorithm:
===> 15) NU_SVC
		Accuracy score = 0.4232		Training time = 726.954843044281		Test time = 359.60646748542786



DONE!
Program finished. It took 8232.752387523651 seconds

Process finished with exit code 0
```