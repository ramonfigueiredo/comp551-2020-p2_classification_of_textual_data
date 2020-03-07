## Grid search logs: Multi-class Classification

### IMDB using Multi-class Classification


#### FINAL CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS

| ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
| -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
|  1  |  ADA_BOOST_CLASSIFIER  |  35.78%  |  [0.352  0.3546 0.3442 0.3492 0.3464]  |  34.93 (+/- 0.75)  |  12.15  |  0.7267  |
|  2  |  BERNOULLI_NB  |  37.25%  |  [0.3786 0.3812 0.3678 0.374  0.373 ]  |  37.49 (+/- 0.93)  |  0.04141  |  0.04044  |
|  3  |  COMPLEMENT_NB  |  37.30%  |  [0.3832 0.3858 0.3834 0.386  0.3776]  |  38.32 (+/- 0.61)  |  0.03622  |  0.01918  |
|  4  |  DECISION_TREE_CLASSIFIER  |  25.18%  |  [0.2476 0.2484 0.2502 0.248  0.2476]  |  24.84 (+/- 0.19)  |  38.92  |  0.03254  |
|  5  |  GRADIENT_BOOSTING_CLASSIFIER  |  37.31%  |  [0.377  0.368  0.3602 0.3648 0.368 ]  |  36.76 (+/- 1.10)  |  433.1  |  0.2777  |
|  6  |  K_NEIGHBORS_CLASSIFIER  |  28.00%  |  [0.3366 0.33   0.3216 0.3218 0.3168]  |  32.54 (+/- 1.41)  |  0.009076  |  27.5  |
|  7  |  LINEAR_SVC  |  37.36%  |  [0.3934 0.3974 0.3888 0.3802 0.3858]  |  38.91 (+/- 1.19)  |  1.907  |  0.01799  |
|  8  |  LOGISTIC_REGRESSION  |  42.04%  |  [0.4282 0.4334 0.4152 0.4194 0.4218]  |  42.36 (+/- 1.29)  |  17.75  |  0.03906  |
|  9  |  MULTINOMIAL_NB  |  35.05%  |  [0.353  0.3502 0.3528 0.3514 0.3488]  |  35.12 (+/- 0.32)  |  0.03259  |  0.01869  |
|  10  |  NEAREST_CENTROID  |  37.12%  |  [0.3884 0.373  0.3818 0.367  0.372 ]  |  37.64 (+/- 1.53)  |  0.02067  |  0.02279  |
|  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  33.42%  |  [0.357  0.3594 0.3554 0.3486 0.3494]  |  35.40 (+/- 0.85)  |  1.72  |  0.01868  |
|  12  |  PERCEPTRON  |  33.09%  |  [0.3508 0.3604 0.3438 0.3348 0.3414]  |  34.62 (+/- 1.75)  |  0.8017  |  0.01915  |
|  13  |  RANDOM_FOREST_CLASSIFIER  |  37.18%  |  [0.3698 0.3676 0.36   0.3652 0.3718]  |  36.69 (+/- 0.82)  |  72.1  |  1.514  |
|  14  |  RIDGE_CLASSIFIER  |  38.56%  |  [0.4036 0.4074 0.402  0.3954 0.4   ]  |  40.17 (+/- 0.79)  |  3.0  |  0.04056  |


#### FINAL CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS

| ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
| -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
|  1  |  ADA_BOOST_CLASSIFIER  |  38.02%  |  [0.3792 0.379  0.374  0.3704 0.3746]  |  37.54 (+/- 0.66)  |  120.6  |  7.146  |
|  2  |  BERNOULLI_NB  |  37.03%  |  [0.377  0.389  0.3782 0.38   0.373 ]  |  37.94 (+/- 1.06)  |  0.04355  |  0.04111  |
|  3  |  COMPLEMENT_NB  |  37.34%  |  [0.3878 0.3942 0.3976 0.3938 0.3832]  |  39.13 (+/- 1.03)  |  0.03701  |  0.01924  |
|  4  |  DECISION_TREE_CLASSIFIER  |  31.11%  |  [0.3162 0.305  0.31   0.299  0.3104]  |  30.81 (+/- 1.16)  |  6.501  |  0.0122  |
|  5  |  GRADIENT_BOOSTING_CLASSIFIER  |  38.08%  |  [0.379  0.3772 0.3658 0.3688 0.376 ]  |  37.34 (+/- 1.03)  |  859.9  |  0.4938  |
|  6  |  K_NEIGHBORS_CLASSIFIER  |  37.26%  |  [0.3822 0.3916 0.3842 0.386  0.388 ]  |  38.64 (+/- 0.65)  |  0.007525  |  27.27  |
|  7  |  LINEAR_SVC  |  40.76%  |  [0.4108 0.4212 0.406  0.3992 0.4082]  |  40.91 (+/- 1.44)  |  0.5699  |  0.01858  |
|  8  |  LOGISTIC_REGRESSION  |  42.04%  |  [0.4282 0.4334 0.4152 0.4194 0.4218]  |  42.36 (+/- 1.29)  |  17.79  |  0.03888  |
|  9  |  MULTINOMIAL_NB  |  37.82%  |  [0.389  0.3928 0.3918 0.3942 0.386 ]  |  39.08 (+/- 0.59)  |  0.03721  |  0.01845  |
|  10  |  NEAREST_CENTROID  |  37.33%  |  [0.3872 0.3786 0.3894 0.3672 0.3782]  |  38.01 (+/- 1.57)  |  0.02343  |  0.03291  |
|  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  41.47%  |  [0.4166 0.426  0.4118 0.408  0.4144]  |  41.54 (+/- 1.21)  |  0.796  |  0.01899  |
|  12  |  PERCEPTRON  |  31.60%  |  [0.3364 0.3094 0.3268 0.298  0.2964]  |  31.34 (+/- 3.16)  |  1.034  |  0.01938  |
|  13  |  RANDOM_FOREST_CLASSIFIER  |  37.90%  |  [0.3758 0.379  0.3742 0.3758 0.3716]  |  37.53 (+/- 0.48)  |  45.27  |  2.492  |
|  14  |  RIDGE_CLASSIFIER  |  38.56%  |  [0.4036 0.4074 0.402  0.3954 0.4   ]  |  40.17 (+/- 0.79)  |  2.98  |  0.04123  |


#### All logs:

```
03/07/2020 12:47:04 AM - INFO - 
>>> GRID SEARCH
03/07/2020 12:47:04 AM - INFO - 

03/07/2020 12:47:04 AM - INFO - ################################################################################
03/07/2020 12:47:04 AM - INFO - 1)
03/07/2020 12:47:04 AM - INFO - ********************************************************************************
03/07/2020 12:47:04 AM - INFO - Classifier: ADA_BOOST_CLASSIFIER, Dataset: IMDB_REVIEWS
03/07/2020 12:47:04 AM - INFO - ********************************************************************************
03/07/2020 12:47:49 AM - INFO - 

Performing grid search...

03/07/2020 12:47:49 AM - INFO - Parameters:
03/07/2020 12:47:49 AM - INFO - {'classifier__learning_rate': [0.1, 1], 'classifier__n_estimators': [200, 500]}
03/07/2020 01:08:24 AM - INFO - 	Done in 1234.289s
03/07/2020 01:08:24 AM - INFO - 	Best score: 0.375
03/07/2020 01:08:24 AM - INFO - 	Best parameters set:
03/07/2020 01:08:24 AM - INFO - 		classifier__learning_rate: 0.1
03/07/2020 01:08:24 AM - INFO - 		classifier__n_estimators: 500
03/07/2020 01:08:24 AM - INFO - 

USING ADA_BOOST_CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 01:08:24 AM - INFO - ________________________________________________________________________________
03/07/2020 01:08:24 AM - INFO - Training: 
03/07/2020 01:08:24 AM - INFO - AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
                   n_estimators=50, random_state=None)
03/07/2020 01:08:36 AM - INFO - Train time: 12.152s
03/07/2020 01:08:36 AM - INFO - Test time:  0.727s
03/07/2020 01:08:36 AM - INFO - Accuracy score:   0.358
03/07/2020 01:08:36 AM - INFO - 

===> Classification Report:

03/07/2020 01:08:36 AM - INFO -               precision    recall  f1-score   support

           1       0.44      0.75      0.56      5022
           2       0.18      0.03      0.06      2302
           3       0.25      0.03      0.06      2541
           4       0.22      0.17      0.19      2635
           7       0.21      0.14      0.17      2307
           8       0.21      0.07      0.10      2850
           9       0.13      0.01      0.01      2344
          10       0.36      0.81      0.50      4999

    accuracy                           0.36     25000
   macro avg       0.25      0.25      0.20     25000
weighted avg       0.28      0.36      0.27     25000

03/07/2020 01:08:36 AM - INFO - 

Cross validation:
03/07/2020 01:09:23 AM - INFO - 	accuracy: 5-fold cross validation: [0.352  0.3546 0.3442 0.3492 0.3464]
03/07/2020 01:09:23 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 34.93 (+/- 0.75)
03/07/2020 01:09:23 AM - INFO - 

USING ADA_BOOST_CLASSIFIER WITH BEST PARAMETERS: {'learning_rate': 0.1, 'n_estimators': 500}
03/07/2020 01:09:23 AM - INFO - ________________________________________________________________________________
03/07/2020 01:09:23 AM - INFO - Training: 
03/07/2020 01:09:23 AM - INFO - AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=0.1,
                   n_estimators=500, random_state=None)
03/07/2020 01:11:24 AM - INFO - Train time: 120.572s
03/07/2020 01:11:31 AM - INFO - Test time:  7.146s
03/07/2020 01:11:31 AM - INFO - Accuracy score:   0.380
03/07/2020 01:11:31 AM - INFO - 

===> Classification Report:

03/07/2020 01:11:31 AM - INFO -               precision    recall  f1-score   support

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

03/07/2020 01:11:31 AM - INFO - 

Cross validation:
03/07/2020 01:19:14 AM - INFO - 	accuracy: 5-fold cross validation: [0.3792 0.379  0.374  0.3704 0.3746]
03/07/2020 01:19:14 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 37.54 (+/- 0.66)
03/07/2020 01:19:14 AM - INFO - It took 1930.2064473628998 seconds
03/07/2020 01:19:14 AM - INFO - ********************************************************************************
03/07/2020 01:19:14 AM - INFO - ################################################################################
03/07/2020 01:19:14 AM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 01:19:14 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:19:14 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:19:14 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  35.78%  |  [0.352  0.3546 0.3442 0.3492 0.3464]  |  34.93 (+/- 0.75)  |  12.15  |  0.7267  |
03/07/2020 01:19:14 AM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 01:19:14 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:19:14 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:19:14 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  38.02%  |  [0.3792 0.379  0.374  0.3704 0.3746]  |  37.54 (+/- 0.66)  |  120.6  |  7.146  |
03/07/2020 01:19:14 AM - INFO - 

03/07/2020 01:19:14 AM - INFO - ################################################################################
03/07/2020 01:19:14 AM - INFO - 2)
03/07/2020 01:19:14 AM - INFO - ********************************************************************************
03/07/2020 01:19:14 AM - INFO - Classifier: DECISION_TREE_CLASSIFIER, Dataset: IMDB_REVIEWS
03/07/2020 01:19:14 AM - INFO - ********************************************************************************
03/07/2020 01:19:21 AM - INFO - 

Performing grid search...

03/07/2020 01:19:21 AM - INFO - Parameters:
03/07/2020 01:19:21 AM - INFO - {'classifier__criterion': ['entropy', 'gini'], 'classifier__splitter': ['best', 'random'], 'classifier__min_samples_split': [2, 100, 250]}
03/07/2020 01:22:56 AM - INFO - 	Done in 215.733s
03/07/2020 01:22:56 AM - INFO - 	Best score: 0.303
03/07/2020 01:22:56 AM - INFO - 	Best parameters set:
03/07/2020 01:22:56 AM - INFO - 		classifier__criterion: 'entropy'
03/07/2020 01:22:56 AM - INFO - 		classifier__min_samples_split: 250
03/07/2020 01:22:56 AM - INFO - 		classifier__splitter: 'random'
03/07/2020 01:22:56 AM - INFO - 

USING DECISION_TREE_CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 01:22:56 AM - INFO - ________________________________________________________________________________
03/07/2020 01:22:56 AM - INFO - Training: 
03/07/2020 01:22:56 AM - INFO - DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=None, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=None, splitter='best')
03/07/2020 01:23:35 AM - INFO - Train time: 38.915s
03/07/2020 01:23:35 AM - INFO - Test time:  0.033s
03/07/2020 01:23:35 AM - INFO - Accuracy score:   0.252
03/07/2020 01:23:35 AM - INFO - 

===> Classification Report:

03/07/2020 01:23:35 AM - INFO -               precision    recall  f1-score   support

           1       0.41      0.45      0.43      5022
           2       0.13      0.11      0.12      2302
           3       0.14      0.14      0.14      2541
           4       0.16      0.17      0.16      2635
           7       0.13      0.13      0.13      2307
           8       0.17      0.17      0.17      2850
           9       0.14      0.12      0.13      2344
          10       0.36      0.39      0.38      4999

    accuracy                           0.25     25000
   macro avg       0.21      0.21      0.21     25000
weighted avg       0.24      0.25      0.25     25000

03/07/2020 01:23:35 AM - INFO - 

Cross validation:
03/07/2020 01:24:16 AM - INFO - 	accuracy: 5-fold cross validation: [0.2476 0.2484 0.2502 0.248  0.2476]
03/07/2020 01:24:16 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 24.84 (+/- 0.19)
03/07/2020 01:24:16 AM - INFO - 

USING DECISION_TREE_CLASSIFIER WITH BEST PARAMETERS: {'criterion': 'entropy', 'min_samples_split': 250, 'splitter': 'random'}
03/07/2020 01:24:16 AM - INFO - ________________________________________________________________________________
03/07/2020 01:24:16 AM - INFO - Training: 
03/07/2020 01:24:16 AM - INFO - DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='entropy',
                       max_depth=None, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=250,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=None, splitter='random')
03/07/2020 01:24:23 AM - INFO - Train time: 6.501s
03/07/2020 01:24:23 AM - INFO - Test time:  0.012s
03/07/2020 01:24:23 AM - INFO - Accuracy score:   0.311
03/07/2020 01:24:23 AM - INFO - 

===> Classification Report:

03/07/2020 01:24:23 AM - INFO -               precision    recall  f1-score   support

           1       0.39      0.69      0.50      5022
           2       0.14      0.03      0.05      2302
           3       0.14      0.05      0.07      2541
           4       0.17      0.14      0.15      2635
           7       0.18      0.15      0.16      2307
           8       0.19      0.13      0.15      2850
           9       0.16      0.03      0.06      2344
          10       0.37      0.59      0.45      4999

    accuracy                           0.31     25000
   macro avg       0.22      0.23      0.20     25000
weighted avg       0.25      0.31      0.26     25000

03/07/2020 01:24:23 AM - INFO - 

Cross validation:
03/07/2020 01:24:31 AM - INFO - 	accuracy: 5-fold cross validation: [0.3162 0.305  0.31   0.299  0.3104]
03/07/2020 01:24:31 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 30.81 (+/- 1.16)
03/07/2020 01:24:31 AM - INFO - It took 317.07057332992554 seconds
03/07/2020 01:24:31 AM - INFO - ********************************************************************************
03/07/2020 01:24:31 AM - INFO - ################################################################################
03/07/2020 01:24:31 AM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 01:24:31 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:24:31 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:24:31 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  35.78%  |  [0.352  0.3546 0.3442 0.3492 0.3464]  |  34.93 (+/- 0.75)  |  12.15  |  0.7267  |
03/07/2020 01:24:31 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  25.18%  |  [0.2476 0.2484 0.2502 0.248  0.2476]  |  24.84 (+/- 0.19)  |  38.92  |  0.03254  |
03/07/2020 01:24:31 AM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 01:24:31 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:24:31 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:24:31 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  38.02%  |  [0.3792 0.379  0.374  0.3704 0.3746]  |  37.54 (+/- 0.66)  |  120.6  |  7.146  |
03/07/2020 01:24:31 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  31.11%  |  [0.3162 0.305  0.31   0.299  0.3104]  |  30.81 (+/- 1.16)  |  6.501  |  0.0122  |
03/07/2020 01:24:31 AM - INFO - 

03/07/2020 01:24:31 AM - INFO - ################################################################################
03/07/2020 01:24:31 AM - INFO - 3)
03/07/2020 01:24:31 AM - INFO - ********************************************************************************
03/07/2020 01:24:31 AM - INFO - Classifier: LINEAR_SVC, Dataset: IMDB_REVIEWS
03/07/2020 01:24:31 AM - INFO - ********************************************************************************
03/07/2020 01:24:38 AM - INFO - 

Performing grid search...

03/07/2020 01:24:38 AM - INFO - Parameters:
03/07/2020 01:24:38 AM - INFO - {'classifier__C': [0.01, 1.0], 'classifier__multi_class': ['ovr', 'crammer_singer'], 'classifier__tol': [0.0001, 0.001]}
03/07/2020 01:25:22 AM - INFO - 	Done in 43.904s
03/07/2020 01:25:22 AM - INFO - 	Best score: 0.409
03/07/2020 01:25:22 AM - INFO - 	Best parameters set:
03/07/2020 01:25:22 AM - INFO - 		classifier__C: 0.01
03/07/2020 01:25:22 AM - INFO - 		classifier__multi_class: 'crammer_singer'
03/07/2020 01:25:22 AM - INFO - 		classifier__tol: 0.001
03/07/2020 01:25:22 AM - INFO - 

USING LINEAR_SVC WITH DEFAULT PARAMETERS
03/07/2020 01:25:22 AM - INFO - ________________________________________________________________________________
03/07/2020 01:25:22 AM - INFO - Training: 
03/07/2020 01:25:22 AM - INFO - LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
          intercept_scaling=1, loss='squared_hinge', max_iter=1000,
          multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
          verbose=0)
03/07/2020 01:25:24 AM - INFO - Train time: 1.907s
03/07/2020 01:25:24 AM - INFO - Test time:  0.018s
03/07/2020 01:25:24 AM - INFO - Accuracy score:   0.374
03/07/2020 01:25:24 AM - INFO - 

===> Classification Report:

03/07/2020 01:25:24 AM - INFO -               precision    recall  f1-score   support

           1       0.54      0.71      0.61      5022
           2       0.17      0.12      0.14      2302
           3       0.22      0.16      0.19      2541
           4       0.26      0.25      0.26      2635
           7       0.23      0.20      0.21      2307
           8       0.23      0.21      0.22      2850
           9       0.20      0.13      0.16      2344
          10       0.49      0.62      0.55      4999

    accuracy                           0.37     25000
   macro avg       0.29      0.30      0.29     25000
weighted avg       0.34      0.37      0.35     25000

03/07/2020 01:25:24 AM - INFO - 

Cross validation:
03/07/2020 01:25:27 AM - INFO - 	accuracy: 5-fold cross validation: [0.3934 0.3974 0.3888 0.3802 0.3858]
03/07/2020 01:25:27 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 38.91 (+/- 1.19)
03/07/2020 01:25:27 AM - INFO - 

USING LINEAR_SVC WITH BEST PARAMETERS: {'C': 0.01, 'multi_class': 'crammer_singer', 'tol': 0.001}
03/07/2020 01:25:27 AM - INFO - ________________________________________________________________________________
03/07/2020 01:25:27 AM - INFO - Training: 
03/07/2020 01:25:27 AM - INFO - LinearSVC(C=0.01, class_weight=None, dual=True, fit_intercept=True,
          intercept_scaling=1, loss='squared_hinge', max_iter=1000,
          multi_class='crammer_singer', penalty='l2', random_state=None,
          tol=0.001, verbose=0)
03/07/2020 01:25:27 AM - INFO - Train time: 0.570s
03/07/2020 01:25:27 AM - INFO - Test time:  0.019s
03/07/2020 01:25:27 AM - INFO - Accuracy score:   0.408
03/07/2020 01:25:27 AM - INFO - 

===> Classification Report:

03/07/2020 01:25:27 AM - INFO -               precision    recall  f1-score   support

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

03/07/2020 01:25:27 AM - INFO - 

Cross validation:
03/07/2020 01:25:29 AM - INFO - 	accuracy: 5-fold cross validation: [0.4108 0.4212 0.406  0.3992 0.4082]
03/07/2020 01:25:29 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 40.91 (+/- 1.44)
03/07/2020 01:25:29 AM - INFO - It took 57.60839915275574 seconds
03/07/2020 01:25:29 AM - INFO - ********************************************************************************
03/07/2020 01:25:29 AM - INFO - ################################################################################
03/07/2020 01:25:29 AM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 01:25:29 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:25:29 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:25:29 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  35.78%  |  [0.352  0.3546 0.3442 0.3492 0.3464]  |  34.93 (+/- 0.75)  |  12.15  |  0.7267  |
03/07/2020 01:25:29 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  25.18%  |  [0.2476 0.2484 0.2502 0.248  0.2476]  |  24.84 (+/- 0.19)  |  38.92  |  0.03254  |
03/07/2020 01:25:29 AM - INFO - |  7  |  LINEAR_SVC  |  37.36%  |  [0.3934 0.3974 0.3888 0.3802 0.3858]  |  38.91 (+/- 1.19)  |  1.907  |  0.01799  |
03/07/2020 01:25:29 AM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 01:25:29 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:25:29 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:25:29 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  38.02%  |  [0.3792 0.379  0.374  0.3704 0.3746]  |  37.54 (+/- 0.66)  |  120.6  |  7.146  |
03/07/2020 01:25:29 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  31.11%  |  [0.3162 0.305  0.31   0.299  0.3104]  |  30.81 (+/- 1.16)  |  6.501  |  0.0122  |
03/07/2020 01:25:29 AM - INFO - |  7  |  LINEAR_SVC  |  40.76%  |  [0.4108 0.4212 0.406  0.3992 0.4082]  |  40.91 (+/- 1.44)  |  0.5699  |  0.01858  |
03/07/2020 01:25:29 AM - INFO - 

03/07/2020 01:25:29 AM - INFO - ################################################################################
03/07/2020 01:25:29 AM - INFO - 4)
03/07/2020 01:25:29 AM - INFO - ********************************************************************************
03/07/2020 01:25:29 AM - INFO - Classifier: LOGISTIC_REGRESSION, Dataset: IMDB_REVIEWS
03/07/2020 01:25:29 AM - INFO - ********************************************************************************
03/07/2020 01:25:35 AM - INFO - 

Performing grid search...

03/07/2020 01:25:35 AM - INFO - Parameters:
03/07/2020 01:25:35 AM - INFO - {'classifier__C': [1, 10], 'classifier__tol': [0.001, 0.01]}
03/07/2020 01:27:30 AM - INFO - 	Done in 114.595s
03/07/2020 01:27:30 AM - INFO - 	Best score: 0.424
03/07/2020 01:27:30 AM - INFO - 	Best parameters set:
03/07/2020 01:27:30 AM - INFO - 		classifier__C: 1
03/07/2020 01:27:30 AM - INFO - 		classifier__tol: 0.001
03/07/2020 01:27:30 AM - INFO - 

USING LOGISTIC_REGRESSION WITH DEFAULT PARAMETERS
03/07/2020 01:27:30 AM - INFO - ________________________________________________________________________________
03/07/2020 01:27:30 AM - INFO - Training: 
03/07/2020 01:27:30 AM - INFO - LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)
03/07/2020 01:27:48 AM - INFO - Train time: 17.753s
03/07/2020 01:27:48 AM - INFO - Test time:  0.039s
03/07/2020 01:27:48 AM - INFO - Accuracy score:   0.420
03/07/2020 01:27:48 AM - INFO - 

===> Classification Report:

03/07/2020 01:27:48 AM - INFO -               precision    recall  f1-score   support

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

03/07/2020 01:27:48 AM - INFO - 

Cross validation:
03/07/2020 01:28:12 AM - INFO - 	accuracy: 5-fold cross validation: [0.4282 0.4334 0.4152 0.4194 0.4218]
03/07/2020 01:28:12 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 42.36 (+/- 1.29)
03/07/2020 01:28:12 AM - INFO - 

USING LOGISTIC_REGRESSION WITH BEST PARAMETERS: {'C': 1, 'tol': 0.001}
03/07/2020 01:28:12 AM - INFO - ________________________________________________________________________________
03/07/2020 01:28:12 AM - INFO - Training: 
03/07/2020 01:28:12 AM - INFO - LogisticRegression(C=1, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.001, verbose=0,
                   warm_start=False)
03/07/2020 01:28:29 AM - INFO - Train time: 17.786s
03/07/2020 01:28:30 AM - INFO - Test time:  0.039s
03/07/2020 01:28:30 AM - INFO - Accuracy score:   0.420
03/07/2020 01:28:30 AM - INFO - 

===> Classification Report:

03/07/2020 01:28:30 AM - INFO -               precision    recall  f1-score   support

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

03/07/2020 01:28:30 AM - INFO - 

Cross validation:
03/07/2020 01:28:53 AM - INFO - 	accuracy: 5-fold cross validation: [0.4282 0.4334 0.4152 0.4194 0.4218]
03/07/2020 01:28:53 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 42.36 (+/- 1.29)
03/07/2020 01:28:53 AM - INFO - It took 204.64489793777466 seconds
03/07/2020 01:28:53 AM - INFO - ********************************************************************************
03/07/2020 01:28:53 AM - INFO - ################################################################################
03/07/2020 01:28:53 AM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 01:28:53 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:28:53 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:28:53 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  35.78%  |  [0.352  0.3546 0.3442 0.3492 0.3464]  |  34.93 (+/- 0.75)  |  12.15  |  0.7267  |
03/07/2020 01:28:53 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  25.18%  |  [0.2476 0.2484 0.2502 0.248  0.2476]  |  24.84 (+/- 0.19)  |  38.92  |  0.03254  |
03/07/2020 01:28:53 AM - INFO - |  7  |  LINEAR_SVC  |  37.36%  |  [0.3934 0.3974 0.3888 0.3802 0.3858]  |  38.91 (+/- 1.19)  |  1.907  |  0.01799  |
03/07/2020 01:28:53 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  42.04%  |  [0.4282 0.4334 0.4152 0.4194 0.4218]  |  42.36 (+/- 1.29)  |  17.75  |  0.03906  |
03/07/2020 01:28:53 AM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 01:28:53 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:28:53 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:28:53 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  38.02%  |  [0.3792 0.379  0.374  0.3704 0.3746]  |  37.54 (+/- 0.66)  |  120.6  |  7.146  |
03/07/2020 01:28:53 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  31.11%  |  [0.3162 0.305  0.31   0.299  0.3104]  |  30.81 (+/- 1.16)  |  6.501  |  0.0122  |
03/07/2020 01:28:53 AM - INFO - |  7  |  LINEAR_SVC  |  40.76%  |  [0.4108 0.4212 0.406  0.3992 0.4082]  |  40.91 (+/- 1.44)  |  0.5699  |  0.01858  |
03/07/2020 01:28:53 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  42.04%  |  [0.4282 0.4334 0.4152 0.4194 0.4218]  |  42.36 (+/- 1.29)  |  17.79  |  0.03888  |
03/07/2020 01:28:53 AM - INFO - 

03/07/2020 01:28:53 AM - INFO - ################################################################################
03/07/2020 01:28:53 AM - INFO - 5)
03/07/2020 01:28:53 AM - INFO - ********************************************************************************
03/07/2020 01:28:53 AM - INFO - Classifier: RANDOM_FOREST_CLASSIFIER, Dataset: IMDB_REVIEWS
03/07/2020 01:28:53 AM - INFO - ********************************************************************************
03/07/2020 01:29:00 AM - INFO - 

Performing grid search...

03/07/2020 01:29:00 AM - INFO - Parameters:
03/07/2020 01:29:00 AM - INFO - {'classifier__min_samples_leaf': [1, 2], 'classifier__min_samples_split': [2, 5], 'classifier__n_estimators': [100, 200]}
03/07/2020 01:40:17 AM - INFO - 	Done in 677.146s
03/07/2020 01:40:17 AM - INFO - 	Best score: 0.374
03/07/2020 01:40:17 AM - INFO - 	Best parameters set:
03/07/2020 01:40:17 AM - INFO - 		classifier__min_samples_leaf: 2
03/07/2020 01:40:17 AM - INFO - 		classifier__min_samples_split: 2
03/07/2020 01:40:17 AM - INFO - 		classifier__n_estimators: 200
03/07/2020 01:40:17 AM - INFO - 

USING RANDOM_FOREST_CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 01:40:17 AM - INFO - ________________________________________________________________________________
03/07/2020 01:40:17 AM - INFO - Training: 
03/07/2020 01:40:17 AM - INFO - RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
03/07/2020 01:41:29 AM - INFO - Train time: 72.096s
03/07/2020 01:41:31 AM - INFO - Test time:  1.514s
03/07/2020 01:41:31 AM - INFO - Accuracy score:   0.372
03/07/2020 01:41:31 AM - INFO - 

===> Classification Report:

03/07/2020 01:41:31 AM - INFO -               precision    recall  f1-score   support

           1       0.38      0.90      0.53      5022
           2       0.39      0.01      0.01      2302
           3       0.32      0.02      0.03      2541
           4       0.34      0.07      0.11      2635
           7       0.27      0.05      0.09      2307
           8       0.25      0.09      0.13      2850
           9       0.40      0.01      0.01      2344
          10       0.38      0.83      0.52      4999

    accuracy                           0.37     25000
   macro avg       0.34      0.25      0.18     25000
weighted avg       0.35      0.37      0.25     25000

03/07/2020 01:41:31 AM - INFO - 

Cross validation:
03/07/2020 01:43:17 AM - INFO - 	accuracy: 5-fold cross validation: [0.3698 0.3676 0.36   0.3652 0.3718]
03/07/2020 01:43:17 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 36.69 (+/- 0.82)
03/07/2020 01:43:17 AM - INFO - 

USING RANDOM_FOREST_CLASSIFIER WITH BEST PARAMETERS: {'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 200}
03/07/2020 01:43:17 AM - INFO - ________________________________________________________________________________
03/07/2020 01:43:17 AM - INFO - Training: 
03/07/2020 01:43:17 AM - INFO - RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=2, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=200,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
03/07/2020 01:44:02 AM - INFO - Train time: 45.273s
03/07/2020 01:44:04 AM - INFO - Test time:  2.492s
03/07/2020 01:44:04 AM - INFO - Accuracy score:   0.379
03/07/2020 01:44:04 AM - INFO - 

===> Classification Report:

03/07/2020 01:44:04 AM - INFO -               precision    recall  f1-score   support

           1       0.38      0.92      0.54      5022
           2       1.00      0.01      0.01      2302
           3       0.46      0.01      0.02      2541
           4       0.32      0.05      0.09      2635
           7       0.38      0.04      0.07      2307
           8       0.27      0.10      0.14      2850
           9       0.57      0.00      0.00      2344
          10       0.39      0.86      0.54      4999

    accuracy                           0.38     25000
   macro avg       0.47      0.25      0.18     25000
weighted avg       0.45      0.38      0.25     25000

03/07/2020 01:44:04 AM - INFO - 

Cross validation:
03/07/2020 01:45:26 AM - INFO - 	accuracy: 5-fold cross validation: [0.3758 0.379  0.3742 0.3758 0.3716]
03/07/2020 01:45:26 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 37.53 (+/- 0.48)
03/07/2020 01:45:26 AM - INFO - It took 992.3556616306305 seconds
03/07/2020 01:45:26 AM - INFO - ********************************************************************************
03/07/2020 01:45:26 AM - INFO - ################################################################################
03/07/2020 01:45:26 AM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 01:45:26 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:45:26 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:45:26 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  35.78%  |  [0.352  0.3546 0.3442 0.3492 0.3464]  |  34.93 (+/- 0.75)  |  12.15  |  0.7267  |
03/07/2020 01:45:26 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  25.18%  |  [0.2476 0.2484 0.2502 0.248  0.2476]  |  24.84 (+/- 0.19)  |  38.92  |  0.03254  |
03/07/2020 01:45:26 AM - INFO - |  7  |  LINEAR_SVC  |  37.36%  |  [0.3934 0.3974 0.3888 0.3802 0.3858]  |  38.91 (+/- 1.19)  |  1.907  |  0.01799  |
03/07/2020 01:45:26 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  42.04%  |  [0.4282 0.4334 0.4152 0.4194 0.4218]  |  42.36 (+/- 1.29)  |  17.75  |  0.03906  |
03/07/2020 01:45:26 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  37.18%  |  [0.3698 0.3676 0.36   0.3652 0.3718]  |  36.69 (+/- 0.82)  |  72.1  |  1.514  |
03/07/2020 01:45:26 AM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 01:45:26 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:45:26 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:45:26 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  38.02%  |  [0.3792 0.379  0.374  0.3704 0.3746]  |  37.54 (+/- 0.66)  |  120.6  |  7.146  |
03/07/2020 01:45:26 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  31.11%  |  [0.3162 0.305  0.31   0.299  0.3104]  |  30.81 (+/- 1.16)  |  6.501  |  0.0122  |
03/07/2020 01:45:26 AM - INFO - |  7  |  LINEAR_SVC  |  40.76%  |  [0.4108 0.4212 0.406  0.3992 0.4082]  |  40.91 (+/- 1.44)  |  0.5699  |  0.01858  |
03/07/2020 01:45:26 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  42.04%  |  [0.4282 0.4334 0.4152 0.4194 0.4218]  |  42.36 (+/- 1.29)  |  17.79  |  0.03888  |
03/07/2020 01:45:26 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  37.90%  |  [0.3758 0.379  0.3742 0.3758 0.3716]  |  37.53 (+/- 0.48)  |  45.27  |  2.492  |
03/07/2020 01:45:26 AM - INFO - 

03/07/2020 01:45:26 AM - INFO - ################################################################################
03/07/2020 01:45:26 AM - INFO - 6)
03/07/2020 01:45:26 AM - INFO - ********************************************************************************
03/07/2020 01:45:26 AM - INFO - Classifier: BERNOULLI_NB, Dataset: IMDB_REVIEWS
03/07/2020 01:45:26 AM - INFO - ********************************************************************************
03/07/2020 01:45:32 AM - INFO - 

Performing grid search...

03/07/2020 01:45:32 AM - INFO - Parameters:
03/07/2020 01:45:32 AM - INFO - {'classifier__alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0], 'classifier__binarize': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0], 'classifier__fit_prior': [False, True]}
03/07/2020 01:45:58 AM - INFO - 	Done in 25.462s
03/07/2020 01:45:58 AM - INFO - 	Best score: 0.379
03/07/2020 01:45:58 AM - INFO - 	Best parameters set:
03/07/2020 01:45:58 AM - INFO - 		classifier__alpha: 0.5
03/07/2020 01:45:58 AM - INFO - 		classifier__binarize: 0.0001
03/07/2020 01:45:58 AM - INFO - 		classifier__fit_prior: True
03/07/2020 01:45:58 AM - INFO - 

USING BERNOULLI_NB WITH DEFAULT PARAMETERS
03/07/2020 01:45:58 AM - INFO - ________________________________________________________________________________
03/07/2020 01:45:58 AM - INFO - Training: 
03/07/2020 01:45:58 AM - INFO - BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
03/07/2020 01:45:58 AM - INFO - Train time: 0.041s
03/07/2020 01:45:58 AM - INFO - Test time:  0.040s
03/07/2020 01:45:58 AM - INFO - Accuracy score:   0.372
03/07/2020 01:45:58 AM - INFO - 

===> Classification Report:

03/07/2020 01:45:58 AM - INFO -               precision    recall  f1-score   support

           1       0.39      0.87      0.54      5022
           2       0.26      0.01      0.01      2302
           3       0.24      0.06      0.09      2541
           4       0.28      0.16      0.20      2635
           7       0.25      0.11      0.15      2307
           8       0.26      0.15      0.19      2850
           9       0.25      0.03      0.05      2344
          10       0.41      0.72      0.52      4999

    accuracy                           0.37     25000
   macro avg       0.29      0.26      0.22     25000
weighted avg       0.32      0.37      0.29     25000

03/07/2020 01:45:58 AM - INFO - 

Cross validation:
03/07/2020 01:45:58 AM - INFO - 	accuracy: 5-fold cross validation: [0.3786 0.3812 0.3678 0.374  0.373 ]
03/07/2020 01:45:58 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 37.49 (+/- 0.93)
03/07/2020 01:45:58 AM - INFO - 

USING BERNOULLI_NB WITH BEST PARAMETERS: {'alpha': 0.5, 'binarize': 0.0001, 'fit_prior': True}
03/07/2020 01:45:58 AM - INFO - ________________________________________________________________________________
03/07/2020 01:45:58 AM - INFO - Training: 
03/07/2020 01:45:58 AM - INFO - BernoulliNB(alpha=0.5, binarize=0.0001, class_prior=None, fit_prior=True)
03/07/2020 01:45:58 AM - INFO - Train time: 0.044s
03/07/2020 01:45:58 AM - INFO - Test time:  0.041s
03/07/2020 01:45:58 AM - INFO - Accuracy score:   0.370
03/07/2020 01:45:58 AM - INFO - 

===> Classification Report:

03/07/2020 01:45:58 AM - INFO -               precision    recall  f1-score   support

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

03/07/2020 01:45:58 AM - INFO - 

Cross validation:
03/07/2020 01:45:59 AM - INFO - 	accuracy: 5-fold cross validation: [0.377  0.389  0.3782 0.38   0.373 ]
03/07/2020 01:45:59 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 37.94 (+/- 1.06)
03/07/2020 01:45:59 AM - INFO - It took 33.02889060974121 seconds
03/07/2020 01:45:59 AM - INFO - ********************************************************************************
03/07/2020 01:45:59 AM - INFO - ################################################################################
03/07/2020 01:45:59 AM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 01:45:59 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:45:59 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:45:59 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  35.78%  |  [0.352  0.3546 0.3442 0.3492 0.3464]  |  34.93 (+/- 0.75)  |  12.15  |  0.7267  |
03/07/2020 01:45:59 AM - INFO - |  2  |  BERNOULLI_NB  |  37.25%  |  [0.3786 0.3812 0.3678 0.374  0.373 ]  |  37.49 (+/- 0.93)  |  0.04141  |  0.04044  |
03/07/2020 01:45:59 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  25.18%  |  [0.2476 0.2484 0.2502 0.248  0.2476]  |  24.84 (+/- 0.19)  |  38.92  |  0.03254  |
03/07/2020 01:45:59 AM - INFO - |  7  |  LINEAR_SVC  |  37.36%  |  [0.3934 0.3974 0.3888 0.3802 0.3858]  |  38.91 (+/- 1.19)  |  1.907  |  0.01799  |
03/07/2020 01:45:59 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  42.04%  |  [0.4282 0.4334 0.4152 0.4194 0.4218]  |  42.36 (+/- 1.29)  |  17.75  |  0.03906  |
03/07/2020 01:45:59 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  37.18%  |  [0.3698 0.3676 0.36   0.3652 0.3718]  |  36.69 (+/- 0.82)  |  72.1  |  1.514  |
03/07/2020 01:45:59 AM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 01:45:59 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:45:59 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:45:59 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  38.02%  |  [0.3792 0.379  0.374  0.3704 0.3746]  |  37.54 (+/- 0.66)  |  120.6  |  7.146  |
03/07/2020 01:45:59 AM - INFO - |  2  |  BERNOULLI_NB  |  37.03%  |  [0.377  0.389  0.3782 0.38   0.373 ]  |  37.94 (+/- 1.06)  |  0.04355  |  0.04111  |
03/07/2020 01:45:59 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  31.11%  |  [0.3162 0.305  0.31   0.299  0.3104]  |  30.81 (+/- 1.16)  |  6.501  |  0.0122  |
03/07/2020 01:45:59 AM - INFO - |  7  |  LINEAR_SVC  |  40.76%  |  [0.4108 0.4212 0.406  0.3992 0.4082]  |  40.91 (+/- 1.44)  |  0.5699  |  0.01858  |
03/07/2020 01:45:59 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  42.04%  |  [0.4282 0.4334 0.4152 0.4194 0.4218]  |  42.36 (+/- 1.29)  |  17.79  |  0.03888  |
03/07/2020 01:45:59 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  37.90%  |  [0.3758 0.379  0.3742 0.3758 0.3716]  |  37.53 (+/- 0.48)  |  45.27  |  2.492  |
03/07/2020 01:45:59 AM - INFO - 

03/07/2020 01:45:59 AM - INFO - ################################################################################
03/07/2020 01:45:59 AM - INFO - 7)
03/07/2020 01:45:59 AM - INFO - ********************************************************************************
03/07/2020 01:45:59 AM - INFO - Classifier: COMPLEMENT_NB, Dataset: IMDB_REVIEWS
03/07/2020 01:45:59 AM - INFO - ********************************************************************************
03/07/2020 01:46:05 AM - INFO - 

Performing grid search...

03/07/2020 01:46:05 AM - INFO - Parameters:
03/07/2020 01:46:05 AM - INFO - {'classifier__alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0], 'classifier__fit_prior': [False, True], 'classifier__norm': [False, True]}
03/07/2020 01:46:10 AM - INFO - 	Done in 4.777s
03/07/2020 01:46:10 AM - INFO - 	Best score: 0.391
03/07/2020 01:46:10 AM - INFO - 	Best parameters set:
03/07/2020 01:46:10 AM - INFO - 		classifier__alpha: 0.5
03/07/2020 01:46:10 AM - INFO - 		classifier__fit_prior: False
03/07/2020 01:46:10 AM - INFO - 		classifier__norm: False
03/07/2020 01:46:10 AM - INFO - 

USING COMPLEMENT_NB WITH DEFAULT PARAMETERS
03/07/2020 01:46:10 AM - INFO - ________________________________________________________________________________
03/07/2020 01:46:10 AM - INFO - Training: 
03/07/2020 01:46:10 AM - INFO - ComplementNB(alpha=1.0, class_prior=None, fit_prior=True, norm=False)
03/07/2020 01:46:10 AM - INFO - Train time: 0.036s
03/07/2020 01:46:10 AM - INFO - Test time:  0.019s
03/07/2020 01:46:10 AM - INFO - Accuracy score:   0.373
03/07/2020 01:46:10 AM - INFO - 

===> Classification Report:

03/07/2020 01:46:10 AM - INFO -               precision    recall  f1-score   support

           1       0.36      0.94      0.53      5022
           2       0.19      0.01      0.02      2302
           3       0.23      0.02      0.04      2541
           4       0.34      0.09      0.15      2635
           7       0.28      0.08      0.12      2307
           8       0.24      0.12      0.16      2850
           9       0.14      0.01      0.02      2344
          10       0.42      0.75      0.54      4999

    accuracy                           0.37     25000
   macro avg       0.28      0.25      0.20     25000
weighted avg       0.30      0.37      0.27     25000

03/07/2020 01:46:10 AM - INFO - 

Cross validation:
03/07/2020 01:46:11 AM - INFO - 	accuracy: 5-fold cross validation: [0.3832 0.3858 0.3834 0.386  0.3776]
03/07/2020 01:46:11 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 38.32 (+/- 0.61)
03/07/2020 01:46:11 AM - INFO - 

USING COMPLEMENT_NB WITH BEST PARAMETERS: {'alpha': 0.5, 'fit_prior': False, 'norm': False}
03/07/2020 01:46:11 AM - INFO - ________________________________________________________________________________
03/07/2020 01:46:11 AM - INFO - Training: 
03/07/2020 01:46:11 AM - INFO - ComplementNB(alpha=0.5, class_prior=None, fit_prior=False, norm=False)
03/07/2020 01:46:11 AM - INFO - Train time: 0.037s
03/07/2020 01:46:11 AM - INFO - Test time:  0.019s
03/07/2020 01:46:11 AM - INFO - Accuracy score:   0.373
03/07/2020 01:46:11 AM - INFO - 

===> Classification Report:

03/07/2020 01:46:11 AM - INFO -               precision    recall  f1-score   support

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

03/07/2020 01:46:11 AM - INFO - 

Cross validation:
03/07/2020 01:46:11 AM - INFO - 	accuracy: 5-fold cross validation: [0.3878 0.3942 0.3976 0.3938 0.3832]
03/07/2020 01:46:11 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 39.13 (+/- 1.03)
03/07/2020 01:46:11 AM - INFO - It took 12.213337898254395 seconds
03/07/2020 01:46:11 AM - INFO - ********************************************************************************
03/07/2020 01:46:11 AM - INFO - ################################################################################
03/07/2020 01:46:11 AM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 01:46:11 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:46:11 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:46:11 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  35.78%  |  [0.352  0.3546 0.3442 0.3492 0.3464]  |  34.93 (+/- 0.75)  |  12.15  |  0.7267  |
03/07/2020 01:46:11 AM - INFO - |  2  |  BERNOULLI_NB  |  37.25%  |  [0.3786 0.3812 0.3678 0.374  0.373 ]  |  37.49 (+/- 0.93)  |  0.04141  |  0.04044  |
03/07/2020 01:46:11 AM - INFO - |  3  |  COMPLEMENT_NB  |  37.30%  |  [0.3832 0.3858 0.3834 0.386  0.3776]  |  38.32 (+/- 0.61)  |  0.03622  |  0.01918  |
03/07/2020 01:46:11 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  25.18%  |  [0.2476 0.2484 0.2502 0.248  0.2476]  |  24.84 (+/- 0.19)  |  38.92  |  0.03254  |
03/07/2020 01:46:11 AM - INFO - |  7  |  LINEAR_SVC  |  37.36%  |  [0.3934 0.3974 0.3888 0.3802 0.3858]  |  38.91 (+/- 1.19)  |  1.907  |  0.01799  |
03/07/2020 01:46:11 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  42.04%  |  [0.4282 0.4334 0.4152 0.4194 0.4218]  |  42.36 (+/- 1.29)  |  17.75  |  0.03906  |
03/07/2020 01:46:11 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  37.18%  |  [0.3698 0.3676 0.36   0.3652 0.3718]  |  36.69 (+/- 0.82)  |  72.1  |  1.514  |
03/07/2020 01:46:11 AM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 01:46:11 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:46:11 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:46:11 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  38.02%  |  [0.3792 0.379  0.374  0.3704 0.3746]  |  37.54 (+/- 0.66)  |  120.6  |  7.146  |
03/07/2020 01:46:11 AM - INFO - |  2  |  BERNOULLI_NB  |  37.03%  |  [0.377  0.389  0.3782 0.38   0.373 ]  |  37.94 (+/- 1.06)  |  0.04355  |  0.04111  |
03/07/2020 01:46:11 AM - INFO - |  3  |  COMPLEMENT_NB  |  37.34%  |  [0.3878 0.3942 0.3976 0.3938 0.3832]  |  39.13 (+/- 1.03)  |  0.03701  |  0.01924  |
03/07/2020 01:46:11 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  31.11%  |  [0.3162 0.305  0.31   0.299  0.3104]  |  30.81 (+/- 1.16)  |  6.501  |  0.0122  |
03/07/2020 01:46:11 AM - INFO - |  7  |  LINEAR_SVC  |  40.76%  |  [0.4108 0.4212 0.406  0.3992 0.4082]  |  40.91 (+/- 1.44)  |  0.5699  |  0.01858  |
03/07/2020 01:46:11 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  42.04%  |  [0.4282 0.4334 0.4152 0.4194 0.4218]  |  42.36 (+/- 1.29)  |  17.79  |  0.03888  |
03/07/2020 01:46:11 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  37.90%  |  [0.3758 0.379  0.3742 0.3758 0.3716]  |  37.53 (+/- 0.48)  |  45.27  |  2.492  |
03/07/2020 01:46:11 AM - INFO - 

03/07/2020 01:46:11 AM - INFO - ################################################################################
03/07/2020 01:46:11 AM - INFO - 8)
03/07/2020 01:46:11 AM - INFO - ********************************************************************************
03/07/2020 01:46:11 AM - INFO - Classifier: MULTINOMIAL_NB, Dataset: IMDB_REVIEWS
03/07/2020 01:46:11 AM - INFO - ********************************************************************************
03/07/2020 01:46:18 AM - INFO - 

Performing grid search...

03/07/2020 01:46:18 AM - INFO - Parameters:
03/07/2020 01:46:18 AM - INFO - {'classifier__alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0], 'classifier__fit_prior': [False, True]}
03/07/2020 01:46:20 AM - INFO - 	Done in 2.454s
03/07/2020 01:46:20 AM - INFO - 	Best score: 0.391
03/07/2020 01:46:20 AM - INFO - 	Best parameters set:
03/07/2020 01:46:20 AM - INFO - 		classifier__alpha: 0.1
03/07/2020 01:46:20 AM - INFO - 		classifier__fit_prior: True
03/07/2020 01:46:20 AM - INFO - 

USING MULTINOMIAL_NB WITH DEFAULT PARAMETERS
03/07/2020 01:46:20 AM - INFO - ________________________________________________________________________________
03/07/2020 01:46:20 AM - INFO - Training: 
03/07/2020 01:46:20 AM - INFO - MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
03/07/2020 01:46:20 AM - INFO - Train time: 0.033s
03/07/2020 01:46:20 AM - INFO - Test time:  0.019s
03/07/2020 01:46:20 AM - INFO - Accuracy score:   0.350
03/07/2020 01:46:20 AM - INFO - 

===> Classification Report:

03/07/2020 01:46:20 AM - INFO -               precision    recall  f1-score   support

           1       0.32      0.97      0.49      5022
           2       0.00      0.00      0.00      2302
           3       0.00      0.00      0.00      2541
           4       1.00      0.00      0.00      2635
           7       0.00      0.00      0.00      2307
           8       0.62      0.00      0.00      2850
           9       0.00      0.00      0.00      2344
          10       0.39      0.77      0.52      4999

    accuracy                           0.35     25000
   macro avg       0.29      0.22      0.13     25000
weighted avg       0.32      0.35      0.20     25000

03/07/2020 01:46:20 AM - INFO - 

Cross validation:
03/07/2020 01:46:21 AM - INFO - 	accuracy: 5-fold cross validation: [0.353  0.3502 0.3528 0.3514 0.3488]
03/07/2020 01:46:21 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 35.12 (+/- 0.32)
03/07/2020 01:46:21 AM - INFO - 

USING MULTINOMIAL_NB WITH BEST PARAMETERS: {'alpha': 0.1, 'fit_prior': True}
03/07/2020 01:46:21 AM - INFO - ________________________________________________________________________________
03/07/2020 01:46:21 AM - INFO - Training: 
03/07/2020 01:46:21 AM - INFO - MultinomialNB(alpha=0.1, class_prior=None, fit_prior=True)
03/07/2020 01:46:21 AM - INFO - Train time: 0.037s
03/07/2020 01:46:21 AM - INFO - Test time:  0.018s
03/07/2020 01:46:21 AM - INFO - Accuracy score:   0.378
03/07/2020 01:46:21 AM - INFO - 

===> Classification Report:

03/07/2020 01:46:21 AM - INFO -               precision    recall  f1-score   support

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

03/07/2020 01:46:21 AM - INFO - 

Cross validation:
03/07/2020 01:46:21 AM - INFO - 	accuracy: 5-fold cross validation: [0.389  0.3928 0.3918 0.3942 0.386 ]
03/07/2020 01:46:21 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 39.08 (+/- 0.59)
03/07/2020 01:46:21 AM - INFO - It took 9.935324430465698 seconds
03/07/2020 01:46:21 AM - INFO - ********************************************************************************
03/07/2020 01:46:21 AM - INFO - ################################################################################
03/07/2020 01:46:21 AM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 01:46:21 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:46:21 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:46:21 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  35.78%  |  [0.352  0.3546 0.3442 0.3492 0.3464]  |  34.93 (+/- 0.75)  |  12.15  |  0.7267  |
03/07/2020 01:46:21 AM - INFO - |  2  |  BERNOULLI_NB  |  37.25%  |  [0.3786 0.3812 0.3678 0.374  0.373 ]  |  37.49 (+/- 0.93)  |  0.04141  |  0.04044  |
03/07/2020 01:46:21 AM - INFO - |  3  |  COMPLEMENT_NB  |  37.30%  |  [0.3832 0.3858 0.3834 0.386  0.3776]  |  38.32 (+/- 0.61)  |  0.03622  |  0.01918  |
03/07/2020 01:46:21 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  25.18%  |  [0.2476 0.2484 0.2502 0.248  0.2476]  |  24.84 (+/- 0.19)  |  38.92  |  0.03254  |
03/07/2020 01:46:21 AM - INFO - |  7  |  LINEAR_SVC  |  37.36%  |  [0.3934 0.3974 0.3888 0.3802 0.3858]  |  38.91 (+/- 1.19)  |  1.907  |  0.01799  |
03/07/2020 01:46:21 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  42.04%  |  [0.4282 0.4334 0.4152 0.4194 0.4218]  |  42.36 (+/- 1.29)  |  17.75  |  0.03906  |
03/07/2020 01:46:21 AM - INFO - |  9  |  MULTINOMIAL_NB  |  35.05%  |  [0.353  0.3502 0.3528 0.3514 0.3488]  |  35.12 (+/- 0.32)  |  0.03259  |  0.01869  |
03/07/2020 01:46:21 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  37.18%  |  [0.3698 0.3676 0.36   0.3652 0.3718]  |  36.69 (+/- 0.82)  |  72.1  |  1.514  |
03/07/2020 01:46:21 AM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 01:46:21 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:46:21 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:46:21 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  38.02%  |  [0.3792 0.379  0.374  0.3704 0.3746]  |  37.54 (+/- 0.66)  |  120.6  |  7.146  |
03/07/2020 01:46:21 AM - INFO - |  2  |  BERNOULLI_NB  |  37.03%  |  [0.377  0.389  0.3782 0.38   0.373 ]  |  37.94 (+/- 1.06)  |  0.04355  |  0.04111  |
03/07/2020 01:46:21 AM - INFO - |  3  |  COMPLEMENT_NB  |  37.34%  |  [0.3878 0.3942 0.3976 0.3938 0.3832]  |  39.13 (+/- 1.03)  |  0.03701  |  0.01924  |
03/07/2020 01:46:21 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  31.11%  |  [0.3162 0.305  0.31   0.299  0.3104]  |  30.81 (+/- 1.16)  |  6.501  |  0.0122  |
03/07/2020 01:46:21 AM - INFO - |  7  |  LINEAR_SVC  |  40.76%  |  [0.4108 0.4212 0.406  0.3992 0.4082]  |  40.91 (+/- 1.44)  |  0.5699  |  0.01858  |
03/07/2020 01:46:21 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  42.04%  |  [0.4282 0.4334 0.4152 0.4194 0.4218]  |  42.36 (+/- 1.29)  |  17.79  |  0.03888  |
03/07/2020 01:46:21 AM - INFO - |  9  |  MULTINOMIAL_NB  |  37.82%  |  [0.389  0.3928 0.3918 0.3942 0.386 ]  |  39.08 (+/- 0.59)  |  0.03721  |  0.01845  |
03/07/2020 01:46:21 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  37.90%  |  [0.3758 0.379  0.3742 0.3758 0.3716]  |  37.53 (+/- 0.48)  |  45.27  |  2.492  |
03/07/2020 01:46:21 AM - INFO - 

03/07/2020 01:46:21 AM - INFO - ################################################################################
03/07/2020 01:46:21 AM - INFO - 9)
03/07/2020 01:46:21 AM - INFO - ********************************************************************************
03/07/2020 01:46:21 AM - INFO - Classifier: NEAREST_CENTROID, Dataset: IMDB_REVIEWS
03/07/2020 01:46:21 AM - INFO - ********************************************************************************
03/07/2020 01:46:28 AM - INFO - 

Performing grid search...

03/07/2020 01:46:28 AM - INFO - Parameters:
03/07/2020 01:46:28 AM - INFO - {'classifier__metric': ['euclidean', 'cosine']}
03/07/2020 01:46:28 AM - INFO - 	Done in 0.381s
03/07/2020 01:46:28 AM - INFO - 	Best score: 0.380
03/07/2020 01:46:28 AM - INFO - 	Best parameters set:
03/07/2020 01:46:28 AM - INFO - 		classifier__metric: 'cosine'
03/07/2020 01:46:28 AM - INFO - 

USING NEAREST_CENTROID WITH DEFAULT PARAMETERS
03/07/2020 01:46:28 AM - INFO - ________________________________________________________________________________
03/07/2020 01:46:28 AM - INFO - Training: 
03/07/2020 01:46:28 AM - INFO - NearestCentroid(metric='euclidean', shrink_threshold=None)
03/07/2020 01:46:28 AM - INFO - Train time: 0.021s
03/07/2020 01:46:28 AM - INFO - Test time:  0.023s
03/07/2020 01:46:28 AM - INFO - Accuracy score:   0.371
03/07/2020 01:46:28 AM - INFO - 

===> Classification Report:

03/07/2020 01:46:28 AM - INFO -               precision    recall  f1-score   support

           1       0.62      0.58      0.60      5022
           2       0.20      0.19      0.20      2302
           3       0.23      0.22      0.23      2541
           4       0.28      0.31      0.29      2635
           7       0.27      0.31      0.29      2307
           8       0.24      0.24      0.24      2850
           9       0.21      0.22      0.21      2344
          10       0.55      0.53      0.54      4999

    accuracy                           0.37     25000
   macro avg       0.32      0.32      0.32     25000
weighted avg       0.38      0.37      0.37     25000

03/07/2020 01:46:28 AM - INFO - 

Cross validation:
03/07/2020 01:46:28 AM - INFO - 	accuracy: 5-fold cross validation: [0.3884 0.373  0.3818 0.367  0.372 ]
03/07/2020 01:46:28 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 37.64 (+/- 1.53)
03/07/2020 01:46:28 AM - INFO - 

USING NEAREST_CENTROID WITH BEST PARAMETERS: {'metric': 'cosine'}
03/07/2020 01:46:28 AM - INFO - ________________________________________________________________________________
03/07/2020 01:46:28 AM - INFO - Training: 
03/07/2020 01:46:28 AM - INFO - NearestCentroid(metric='cosine', shrink_threshold=None)
03/07/2020 01:46:28 AM - INFO - Train time: 0.023s
03/07/2020 01:46:28 AM - INFO - Test time:  0.033s
03/07/2020 01:46:28 AM - INFO - Accuracy score:   0.373
03/07/2020 01:46:28 AM - INFO - 

===> Classification Report:

03/07/2020 01:46:28 AM - INFO -               precision    recall  f1-score   support

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

03/07/2020 01:46:28 AM - INFO - 

Cross validation:
03/07/2020 01:46:29 AM - INFO - 	accuracy: 5-fold cross validation: [0.3872 0.3786 0.3894 0.3672 0.3782]
03/07/2020 01:46:29 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 38.01 (+/- 1.57)
03/07/2020 01:46:29 AM - INFO - It took 7.852216958999634 seconds
03/07/2020 01:46:29 AM - INFO - ********************************************************************************
03/07/2020 01:46:29 AM - INFO - ################################################################################
03/07/2020 01:46:29 AM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 01:46:29 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:46:29 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:46:29 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  35.78%  |  [0.352  0.3546 0.3442 0.3492 0.3464]  |  34.93 (+/- 0.75)  |  12.15  |  0.7267  |
03/07/2020 01:46:29 AM - INFO - |  2  |  BERNOULLI_NB  |  37.25%  |  [0.3786 0.3812 0.3678 0.374  0.373 ]  |  37.49 (+/- 0.93)  |  0.04141  |  0.04044  |
03/07/2020 01:46:29 AM - INFO - |  3  |  COMPLEMENT_NB  |  37.30%  |  [0.3832 0.3858 0.3834 0.386  0.3776]  |  38.32 (+/- 0.61)  |  0.03622  |  0.01918  |
03/07/2020 01:46:29 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  25.18%  |  [0.2476 0.2484 0.2502 0.248  0.2476]  |  24.84 (+/- 0.19)  |  38.92  |  0.03254  |
03/07/2020 01:46:29 AM - INFO - |  7  |  LINEAR_SVC  |  37.36%  |  [0.3934 0.3974 0.3888 0.3802 0.3858]  |  38.91 (+/- 1.19)  |  1.907  |  0.01799  |
03/07/2020 01:46:29 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  42.04%  |  [0.4282 0.4334 0.4152 0.4194 0.4218]  |  42.36 (+/- 1.29)  |  17.75  |  0.03906  |
03/07/2020 01:46:29 AM - INFO - |  9  |  MULTINOMIAL_NB  |  35.05%  |  [0.353  0.3502 0.3528 0.3514 0.3488]  |  35.12 (+/- 0.32)  |  0.03259  |  0.01869  |
03/07/2020 01:46:29 AM - INFO - |  10  |  NEAREST_CENTROID  |  37.12%  |  [0.3884 0.373  0.3818 0.367  0.372 ]  |  37.64 (+/- 1.53)  |  0.02067  |  0.02279  |
03/07/2020 01:46:29 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  37.18%  |  [0.3698 0.3676 0.36   0.3652 0.3718]  |  36.69 (+/- 0.82)  |  72.1  |  1.514  |
03/07/2020 01:46:29 AM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 01:46:29 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:46:29 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:46:29 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  38.02%  |  [0.3792 0.379  0.374  0.3704 0.3746]  |  37.54 (+/- 0.66)  |  120.6  |  7.146  |
03/07/2020 01:46:29 AM - INFO - |  2  |  BERNOULLI_NB  |  37.03%  |  [0.377  0.389  0.3782 0.38   0.373 ]  |  37.94 (+/- 1.06)  |  0.04355  |  0.04111  |
03/07/2020 01:46:29 AM - INFO - |  3  |  COMPLEMENT_NB  |  37.34%  |  [0.3878 0.3942 0.3976 0.3938 0.3832]  |  39.13 (+/- 1.03)  |  0.03701  |  0.01924  |
03/07/2020 01:46:29 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  31.11%  |  [0.3162 0.305  0.31   0.299  0.3104]  |  30.81 (+/- 1.16)  |  6.501  |  0.0122  |
03/07/2020 01:46:29 AM - INFO - |  7  |  LINEAR_SVC  |  40.76%  |  [0.4108 0.4212 0.406  0.3992 0.4082]  |  40.91 (+/- 1.44)  |  0.5699  |  0.01858  |
03/07/2020 01:46:29 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  42.04%  |  [0.4282 0.4334 0.4152 0.4194 0.4218]  |  42.36 (+/- 1.29)  |  17.79  |  0.03888  |
03/07/2020 01:46:29 AM - INFO - |  9  |  MULTINOMIAL_NB  |  37.82%  |  [0.389  0.3928 0.3918 0.3942 0.386 ]  |  39.08 (+/- 0.59)  |  0.03721  |  0.01845  |
03/07/2020 01:46:29 AM - INFO - |  10  |  NEAREST_CENTROID  |  37.33%  |  [0.3872 0.3786 0.3894 0.3672 0.3782]  |  38.01 (+/- 1.57)  |  0.02343  |  0.03291  |
03/07/2020 01:46:29 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  37.90%  |  [0.3758 0.379  0.3742 0.3758 0.3716]  |  37.53 (+/- 0.48)  |  45.27  |  2.492  |
03/07/2020 01:46:29 AM - INFO - 

03/07/2020 01:46:29 AM - INFO - ################################################################################
03/07/2020 01:46:29 AM - INFO - 10)
03/07/2020 01:46:29 AM - INFO - ********************************************************************************
03/07/2020 01:46:29 AM - INFO - Classifier: PASSIVE_AGGRESSIVE_CLASSIFIER, Dataset: IMDB_REVIEWS
03/07/2020 01:46:29 AM - INFO - ********************************************************************************
03/07/2020 01:46:36 AM - INFO - 

Performing grid search...

03/07/2020 01:46:36 AM - INFO - Parameters:
03/07/2020 01:46:36 AM - INFO - {'classifier__C': [0.01, 1.0], 'classifier__early_stopping': [False, True], 'classifier__tol': [0.0001, 0.001, 0.01], 'classifier__validation_fraction': [0.0001, 0.01]}
03/07/2020 01:48:14 AM - INFO - 	Done in 98.422s
03/07/2020 01:48:14 AM - INFO - 	Best score: 0.417
03/07/2020 01:48:14 AM - INFO - 	Best parameters set:
03/07/2020 01:48:14 AM - INFO - 		classifier__C: 0.01
03/07/2020 01:48:14 AM - INFO - 		classifier__early_stopping: False
03/07/2020 01:48:14 AM - INFO - 		classifier__tol: 0.01
03/07/2020 01:48:14 AM - INFO - 		classifier__validation_fraction: 0.01
03/07/2020 01:48:14 AM - INFO - 

USING PASSIVE_AGGRESSIVE_CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 01:48:14 AM - INFO - ________________________________________________________________________________
03/07/2020 01:48:14 AM - INFO - Training: 
03/07/2020 01:48:14 AM - INFO - PassiveAggressiveClassifier(C=1.0, average=False, class_weight=None,
                            early_stopping=False, fit_intercept=True,
                            loss='hinge', max_iter=1000, n_iter_no_change=5,
                            n_jobs=None, random_state=None, shuffle=True,
                            tol=0.001, validation_fraction=0.1, verbose=0,
                            warm_start=False)
03/07/2020 01:48:16 AM - INFO - Train time: 1.720s
03/07/2020 01:48:16 AM - INFO - Test time:  0.019s
03/07/2020 01:48:16 AM - INFO - Accuracy score:   0.334
03/07/2020 01:48:16 AM - INFO - 

===> Classification Report:

03/07/2020 01:48:16 AM - INFO -               precision    recall  f1-score   support

           1       0.53      0.59      0.56      5022
           2       0.17      0.16      0.16      2302
           3       0.19      0.17      0.18      2541
           4       0.24      0.23      0.24      2635
           7       0.20      0.19      0.19      2307
           8       0.21      0.21      0.21      2850
           9       0.18      0.16      0.17      2344
          10       0.48      0.51      0.49      4999

    accuracy                           0.33     25000
   macro avg       0.28      0.28      0.28     25000
weighted avg       0.32      0.33      0.33     25000

03/07/2020 01:48:16 AM - INFO - 

Cross validation:
03/07/2020 01:48:18 AM - INFO - 	accuracy: 5-fold cross validation: [0.357  0.3594 0.3554 0.3486 0.3494]
03/07/2020 01:48:18 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 35.40 (+/- 0.85)
03/07/2020 01:48:18 AM - INFO - 

USING PASSIVE_AGGRESSIVE_CLASSIFIER WITH BEST PARAMETERS: {'C': 0.01, 'early_stopping': False, 'tol': 0.01, 'validation_fraction': 0.01}
03/07/2020 01:48:18 AM - INFO - ________________________________________________________________________________
03/07/2020 01:48:18 AM - INFO - Training: 
03/07/2020 01:48:18 AM - INFO - PassiveAggressiveClassifier(C=0.01, average=False, class_weight=None,
                            early_stopping=False, fit_intercept=True,
                            loss='hinge', max_iter=1000, n_iter_no_change=5,
                            n_jobs=None, random_state=None, shuffle=True,
                            tol=0.01, validation_fraction=0.01, verbose=0,
                            warm_start=False)
03/07/2020 01:48:18 AM - INFO - Train time: 0.796s
03/07/2020 01:48:18 AM - INFO - Test time:  0.019s
03/07/2020 01:48:18 AM - INFO - Accuracy score:   0.415
03/07/2020 01:48:18 AM - INFO - 

===> Classification Report:

03/07/2020 01:48:18 AM - INFO -               precision    recall  f1-score   support

           1       0.47      0.89      0.61      5022
           2       0.15      0.03      0.04      2302
           3       0.25      0.08      0.13      2541
           4       0.32      0.26      0.29      2635
           7       0.29      0.19      0.23      2307
           8       0.26      0.20      0.23      2850
           9       0.23      0.03      0.06      2344
          10       0.49      0.76      0.59      4999

    accuracy                           0.41     25000
   macro avg       0.31      0.31      0.27     25000
weighted avg       0.34      0.41      0.34     25000

03/07/2020 01:48:18 AM - INFO - 

Cross validation:
03/07/2020 01:48:19 AM - INFO - 	accuracy: 5-fold cross validation: [0.4166 0.426  0.4118 0.408  0.4144]
03/07/2020 01:48:19 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 41.54 (+/- 1.21)
03/07/2020 01:48:19 AM - INFO - It took 110.59389472007751 seconds
03/07/2020 01:48:19 AM - INFO - ********************************************************************************
03/07/2020 01:48:19 AM - INFO - ################################################################################
03/07/2020 01:48:19 AM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 01:48:19 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:48:19 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:48:19 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  35.78%  |  [0.352  0.3546 0.3442 0.3492 0.3464]  |  34.93 (+/- 0.75)  |  12.15  |  0.7267  |
03/07/2020 01:48:19 AM - INFO - |  2  |  BERNOULLI_NB  |  37.25%  |  [0.3786 0.3812 0.3678 0.374  0.373 ]  |  37.49 (+/- 0.93)  |  0.04141  |  0.04044  |
03/07/2020 01:48:19 AM - INFO - |  3  |  COMPLEMENT_NB  |  37.30%  |  [0.3832 0.3858 0.3834 0.386  0.3776]  |  38.32 (+/- 0.61)  |  0.03622  |  0.01918  |
03/07/2020 01:48:19 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  25.18%  |  [0.2476 0.2484 0.2502 0.248  0.2476]  |  24.84 (+/- 0.19)  |  38.92  |  0.03254  |
03/07/2020 01:48:19 AM - INFO - |  7  |  LINEAR_SVC  |  37.36%  |  [0.3934 0.3974 0.3888 0.3802 0.3858]  |  38.91 (+/- 1.19)  |  1.907  |  0.01799  |
03/07/2020 01:48:19 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  42.04%  |  [0.4282 0.4334 0.4152 0.4194 0.4218]  |  42.36 (+/- 1.29)  |  17.75  |  0.03906  |
03/07/2020 01:48:19 AM - INFO - |  9  |  MULTINOMIAL_NB  |  35.05%  |  [0.353  0.3502 0.3528 0.3514 0.3488]  |  35.12 (+/- 0.32)  |  0.03259  |  0.01869  |
03/07/2020 01:48:19 AM - INFO - |  10  |  NEAREST_CENTROID  |  37.12%  |  [0.3884 0.373  0.3818 0.367  0.372 ]  |  37.64 (+/- 1.53)  |  0.02067  |  0.02279  |
03/07/2020 01:48:19 AM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  33.42%  |  [0.357  0.3594 0.3554 0.3486 0.3494]  |  35.40 (+/- 0.85)  |  1.72  |  0.01868  |
03/07/2020 01:48:19 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  37.18%  |  [0.3698 0.3676 0.36   0.3652 0.3718]  |  36.69 (+/- 0.82)  |  72.1  |  1.514  |
03/07/2020 01:48:19 AM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 01:48:19 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:48:19 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:48:19 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  38.02%  |  [0.3792 0.379  0.374  0.3704 0.3746]  |  37.54 (+/- 0.66)  |  120.6  |  7.146  |
03/07/2020 01:48:19 AM - INFO - |  2  |  BERNOULLI_NB  |  37.03%  |  [0.377  0.389  0.3782 0.38   0.373 ]  |  37.94 (+/- 1.06)  |  0.04355  |  0.04111  |
03/07/2020 01:48:19 AM - INFO - |  3  |  COMPLEMENT_NB  |  37.34%  |  [0.3878 0.3942 0.3976 0.3938 0.3832]  |  39.13 (+/- 1.03)  |  0.03701  |  0.01924  |
03/07/2020 01:48:19 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  31.11%  |  [0.3162 0.305  0.31   0.299  0.3104]  |  30.81 (+/- 1.16)  |  6.501  |  0.0122  |
03/07/2020 01:48:19 AM - INFO - |  7  |  LINEAR_SVC  |  40.76%  |  [0.4108 0.4212 0.406  0.3992 0.4082]  |  40.91 (+/- 1.44)  |  0.5699  |  0.01858  |
03/07/2020 01:48:19 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  42.04%  |  [0.4282 0.4334 0.4152 0.4194 0.4218]  |  42.36 (+/- 1.29)  |  17.79  |  0.03888  |
03/07/2020 01:48:19 AM - INFO - |  9  |  MULTINOMIAL_NB  |  37.82%  |  [0.389  0.3928 0.3918 0.3942 0.386 ]  |  39.08 (+/- 0.59)  |  0.03721  |  0.01845  |
03/07/2020 01:48:19 AM - INFO - |  10  |  NEAREST_CENTROID  |  37.33%  |  [0.3872 0.3786 0.3894 0.3672 0.3782]  |  38.01 (+/- 1.57)  |  0.02343  |  0.03291  |
03/07/2020 01:48:19 AM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  41.47%  |  [0.4166 0.426  0.4118 0.408  0.4144]  |  41.54 (+/- 1.21)  |  0.796  |  0.01899  |
03/07/2020 01:48:19 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  37.90%  |  [0.3758 0.379  0.3742 0.3758 0.3716]  |  37.53 (+/- 0.48)  |  45.27  |  2.492  |
03/07/2020 01:48:19 AM - INFO - 

03/07/2020 01:48:19 AM - INFO - ################################################################################
03/07/2020 01:48:19 AM - INFO - 11)
03/07/2020 01:48:19 AM - INFO - ********************************************************************************
03/07/2020 01:48:19 AM - INFO - Classifier: K_NEIGHBORS_CLASSIFIER, Dataset: IMDB_REVIEWS
03/07/2020 01:48:19 AM - INFO - ********************************************************************************
03/07/2020 01:48:26 AM - INFO - 

Performing grid search...

03/07/2020 01:48:26 AM - INFO - Parameters:
03/07/2020 01:48:26 AM - INFO - {'classifier__leaf_size': [5, 30], 'classifier__metric': ['euclidean', 'minkowski'], 'classifier__n_neighbors': [3, 50], 'classifier__weights': ['uniform', 'distance']}
03/07/2020 01:49:49 AM - INFO - 	Done in 82.608s
03/07/2020 01:49:49 AM - INFO - 	Best score: 0.386
03/07/2020 01:49:49 AM - INFO - 	Best parameters set:
03/07/2020 01:49:49 AM - INFO - 		classifier__leaf_size: 5
03/07/2020 01:49:49 AM - INFO - 		classifier__metric: 'euclidean'
03/07/2020 01:49:49 AM - INFO - 		classifier__n_neighbors: 50
03/07/2020 01:49:49 AM - INFO - 		classifier__weights: 'distance'
03/07/2020 01:49:49 AM - INFO - 

USING K_NEIGHBORS_CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 01:49:49 AM - INFO - ________________________________________________________________________________
03/07/2020 01:49:49 AM - INFO - Training: 
03/07/2020 01:49:49 AM - INFO - KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                     weights='uniform')
03/07/2020 01:49:49 AM - INFO - Train time: 0.009s
03/07/2020 01:50:16 AM - INFO - Test time:  27.500s
03/07/2020 01:50:16 AM - INFO - Accuracy score:   0.280
03/07/2020 01:50:16 AM - INFO - 

===> Classification Report:

03/07/2020 01:50:16 AM - INFO -               precision    recall  f1-score   support

           1       0.34      0.70      0.46      5022
           2       0.12      0.10      0.11      2302
           3       0.15      0.09      0.11      2541
           4       0.19      0.11      0.14      2635
           7       0.15      0.08      0.11      2307
           8       0.18      0.11      0.14      2850
           9       0.16      0.09      0.12      2344
          10       0.37      0.41      0.39      4999

    accuracy                           0.28     25000
   macro avg       0.21      0.21      0.20     25000
weighted avg       0.24      0.28      0.24     25000

03/07/2020 01:50:16 AM - INFO - 

Cross validation:
03/07/2020 01:50:23 AM - INFO - 	accuracy: 5-fold cross validation: [0.3366 0.33   0.3216 0.3218 0.3168]
03/07/2020 01:50:23 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 32.54 (+/- 1.41)
03/07/2020 01:50:23 AM - INFO - 

USING K_NEIGHBORS_CLASSIFIER WITH BEST PARAMETERS: {'leaf_size': 5, 'metric': 'euclidean', 'n_neighbors': 50, 'weights': 'distance'}
03/07/2020 01:50:23 AM - INFO - ________________________________________________________________________________
03/07/2020 01:50:23 AM - INFO - Training: 
03/07/2020 01:50:23 AM - INFO - KNeighborsClassifier(algorithm='auto', leaf_size=5, metric='euclidean',
                     metric_params=None, n_jobs=None, n_neighbors=50, p=2,
                     weights='distance')
03/07/2020 01:50:23 AM - INFO - Train time: 0.008s
03/07/2020 01:50:50 AM - INFO - Test time:  27.272s
03/07/2020 01:50:50 AM - INFO - Accuracy score:   0.373
03/07/2020 01:50:50 AM - INFO - 

===> Classification Report:

03/07/2020 01:50:50 AM - INFO -               precision    recall  f1-score   support

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

03/07/2020 01:50:50 AM - INFO - 

Cross validation:
03/07/2020 01:50:57 AM - INFO - 	accuracy: 5-fold cross validation: [0.3822 0.3916 0.3842 0.386  0.388 ]
03/07/2020 01:50:57 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 38.64 (+/- 0.65)
03/07/2020 01:50:57 AM - INFO - It took 158.00079345703125 seconds
03/07/2020 01:50:57 AM - INFO - ********************************************************************************
03/07/2020 01:50:57 AM - INFO - ################################################################################
03/07/2020 01:50:57 AM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 01:50:57 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:50:57 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:50:57 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  35.78%  |  [0.352  0.3546 0.3442 0.3492 0.3464]  |  34.93 (+/- 0.75)  |  12.15  |  0.7267  |
03/07/2020 01:50:57 AM - INFO - |  2  |  BERNOULLI_NB  |  37.25%  |  [0.3786 0.3812 0.3678 0.374  0.373 ]  |  37.49 (+/- 0.93)  |  0.04141  |  0.04044  |
03/07/2020 01:50:57 AM - INFO - |  3  |  COMPLEMENT_NB  |  37.30%  |  [0.3832 0.3858 0.3834 0.386  0.3776]  |  38.32 (+/- 0.61)  |  0.03622  |  0.01918  |
03/07/2020 01:50:57 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  25.18%  |  [0.2476 0.2484 0.2502 0.248  0.2476]  |  24.84 (+/- 0.19)  |  38.92  |  0.03254  |
03/07/2020 01:50:57 AM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  28.00%  |  [0.3366 0.33   0.3216 0.3218 0.3168]  |  32.54 (+/- 1.41)  |  0.009076  |  27.5  |
03/07/2020 01:50:57 AM - INFO - |  7  |  LINEAR_SVC  |  37.36%  |  [0.3934 0.3974 0.3888 0.3802 0.3858]  |  38.91 (+/- 1.19)  |  1.907  |  0.01799  |
03/07/2020 01:50:57 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  42.04%  |  [0.4282 0.4334 0.4152 0.4194 0.4218]  |  42.36 (+/- 1.29)  |  17.75  |  0.03906  |
03/07/2020 01:50:57 AM - INFO - |  9  |  MULTINOMIAL_NB  |  35.05%  |  [0.353  0.3502 0.3528 0.3514 0.3488]  |  35.12 (+/- 0.32)  |  0.03259  |  0.01869  |
03/07/2020 01:50:57 AM - INFO - |  10  |  NEAREST_CENTROID  |  37.12%  |  [0.3884 0.373  0.3818 0.367  0.372 ]  |  37.64 (+/- 1.53)  |  0.02067  |  0.02279  |
03/07/2020 01:50:57 AM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  33.42%  |  [0.357  0.3594 0.3554 0.3486 0.3494]  |  35.40 (+/- 0.85)  |  1.72  |  0.01868  |
03/07/2020 01:50:57 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  37.18%  |  [0.3698 0.3676 0.36   0.3652 0.3718]  |  36.69 (+/- 0.82)  |  72.1  |  1.514  |
03/07/2020 01:50:57 AM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 01:50:57 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:50:57 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:50:57 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  38.02%  |  [0.3792 0.379  0.374  0.3704 0.3746]  |  37.54 (+/- 0.66)  |  120.6  |  7.146  |
03/07/2020 01:50:57 AM - INFO - |  2  |  BERNOULLI_NB  |  37.03%  |  [0.377  0.389  0.3782 0.38   0.373 ]  |  37.94 (+/- 1.06)  |  0.04355  |  0.04111  |
03/07/2020 01:50:57 AM - INFO - |  3  |  COMPLEMENT_NB  |  37.34%  |  [0.3878 0.3942 0.3976 0.3938 0.3832]  |  39.13 (+/- 1.03)  |  0.03701  |  0.01924  |
03/07/2020 01:50:57 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  31.11%  |  [0.3162 0.305  0.31   0.299  0.3104]  |  30.81 (+/- 1.16)  |  6.501  |  0.0122  |
03/07/2020 01:50:57 AM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  37.26%  |  [0.3822 0.3916 0.3842 0.386  0.388 ]  |  38.64 (+/- 0.65)  |  0.007525  |  27.27  |
03/07/2020 01:50:57 AM - INFO - |  7  |  LINEAR_SVC  |  40.76%  |  [0.4108 0.4212 0.406  0.3992 0.4082]  |  40.91 (+/- 1.44)  |  0.5699  |  0.01858  |
03/07/2020 01:50:57 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  42.04%  |  [0.4282 0.4334 0.4152 0.4194 0.4218]  |  42.36 (+/- 1.29)  |  17.79  |  0.03888  |
03/07/2020 01:50:57 AM - INFO - |  9  |  MULTINOMIAL_NB  |  37.82%  |  [0.389  0.3928 0.3918 0.3942 0.386 ]  |  39.08 (+/- 0.59)  |  0.03721  |  0.01845  |
03/07/2020 01:50:57 AM - INFO - |  10  |  NEAREST_CENTROID  |  37.33%  |  [0.3872 0.3786 0.3894 0.3672 0.3782]  |  38.01 (+/- 1.57)  |  0.02343  |  0.03291  |
03/07/2020 01:50:57 AM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  41.47%  |  [0.4166 0.426  0.4118 0.408  0.4144]  |  41.54 (+/- 1.21)  |  0.796  |  0.01899  |
03/07/2020 01:50:57 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  37.90%  |  [0.3758 0.379  0.3742 0.3758 0.3716]  |  37.53 (+/- 0.48)  |  45.27  |  2.492  |
03/07/2020 01:50:57 AM - INFO - 

03/07/2020 01:50:57 AM - INFO - ################################################################################
03/07/2020 01:50:57 AM - INFO - 12)
03/07/2020 01:50:57 AM - INFO - ********************************************************************************
03/07/2020 01:50:57 AM - INFO - Classifier: PERCEPTRON, Dataset: IMDB_REVIEWS
03/07/2020 01:50:57 AM - INFO - ********************************************************************************
03/07/2020 01:54:54 AM - INFO - 

Performing grid search...

03/07/2020 01:54:54 AM - INFO - Parameters:
03/07/2020 01:54:54 AM - INFO - {'classifier__early_stopping': [True], 'classifier__max_iter': [100], 'classifier__n_iter_no_change': [3, 15], 'classifier__penalty': ['l2'], 'classifier__tol': [0.0001, 0.1], 'classifier__validation_fraction': [0.0001, 0.01]}
03/07/2020 01:55:06 AM - INFO - 	Done in 12.247s
03/07/2020 01:55:06 AM - INFO - 	Best score: 0.313
03/07/2020 01:55:06 AM - INFO - 	Best parameters set:
03/07/2020 01:55:06 AM - INFO - 		classifier__early_stopping: True
03/07/2020 01:55:06 AM - INFO - 		classifier__max_iter: 100
03/07/2020 01:55:06 AM - INFO - 		classifier__n_iter_no_change: 3
03/07/2020 01:55:06 AM - INFO - 		classifier__penalty: 'l2'
03/07/2020 01:55:06 AM - INFO - 		classifier__tol: 0.0001
03/07/2020 01:55:06 AM - INFO - 		classifier__validation_fraction: 0.01
03/07/2020 01:55:06 AM - INFO - 

USING PERCEPTRON WITH DEFAULT PARAMETERS
03/07/2020 01:55:06 AM - INFO - ________________________________________________________________________________
03/07/2020 01:55:06 AM - INFO - Training: 
03/07/2020 01:55:06 AM - INFO - Perceptron(alpha=0.0001, class_weight=None, early_stopping=False, eta0=1.0,
           fit_intercept=True, max_iter=1000, n_iter_no_change=5, n_jobs=None,
           penalty=None, random_state=0, shuffle=True, tol=0.001,
           validation_fraction=0.1, verbose=0, warm_start=False)
03/07/2020 01:55:07 AM - INFO - Train time: 0.802s
03/07/2020 01:55:07 AM - INFO - Test time:  0.019s
03/07/2020 01:55:07 AM - INFO - Accuracy score:   0.331
03/07/2020 01:55:07 AM - INFO - 

===> Classification Report:

03/07/2020 01:55:07 AM - INFO -               precision    recall  f1-score   support

           1       0.51      0.60      0.55      5022
           2       0.16      0.17      0.17      2302
           3       0.20      0.17      0.18      2541
           4       0.22      0.22      0.22      2635
           7       0.20      0.19      0.20      2307
           8       0.21      0.19      0.20      2850
           9       0.18      0.15      0.16      2344
          10       0.48      0.51      0.49      4999

    accuracy                           0.33     25000
   macro avg       0.27      0.27      0.27     25000
weighted avg       0.32      0.33      0.32     25000

03/07/2020 01:55:07 AM - INFO - 

Cross validation:
03/07/2020 01:55:08 AM - INFO - 	accuracy: 5-fold cross validation: [0.3508 0.3604 0.3438 0.3348 0.3414]
03/07/2020 01:55:08 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 34.62 (+/- 1.75)
03/07/2020 01:55:08 AM - INFO - 

USING PERCEPTRON WITH BEST PARAMETERS: {'early_stopping': True, 'max_iter': 100, 'n_iter_no_change': 3, 'penalty': 'l2', 'tol': 0.0001, 'validation_fraction': 0.01}
03/07/2020 01:55:08 AM - INFO - ________________________________________________________________________________
03/07/2020 01:55:08 AM - INFO - Training: 
03/07/2020 01:55:08 AM - INFO - Perceptron(alpha=0.0001, class_weight=None, early_stopping=True, eta0=1.0,
           fit_intercept=True, max_iter=100, n_iter_no_change=3, n_jobs=None,
           penalty='l2', random_state=0, shuffle=True, tol=0.0001,
           validation_fraction=0.01, verbose=0, warm_start=False)
03/07/2020 01:55:09 AM - INFO - Train time: 1.034s
03/07/2020 01:55:09 AM - INFO - Test time:  0.019s
03/07/2020 01:55:09 AM - INFO - Accuracy score:   0.316
03/07/2020 01:55:09 AM - INFO - 

===> Classification Report:

03/07/2020 01:55:09 AM - INFO -               precision    recall  f1-score   support

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

03/07/2020 01:55:09 AM - INFO - 

Cross validation:
03/07/2020 01:55:10 AM - INFO - 	accuracy: 5-fold cross validation: [0.3364 0.3094 0.3268 0.298  0.2964]
03/07/2020 01:55:10 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 31.34 (+/- 3.16)
03/07/2020 01:55:10 AM - INFO - It took 252.83769297599792 seconds
03/07/2020 01:55:10 AM - INFO - ********************************************************************************
03/07/2020 01:55:10 AM - INFO - ################################################################################
03/07/2020 01:55:10 AM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 01:55:10 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:55:10 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:55:10 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  35.78%  |  [0.352  0.3546 0.3442 0.3492 0.3464]  |  34.93 (+/- 0.75)  |  12.15  |  0.7267  |
03/07/2020 01:55:10 AM - INFO - |  2  |  BERNOULLI_NB  |  37.25%  |  [0.3786 0.3812 0.3678 0.374  0.373 ]  |  37.49 (+/- 0.93)  |  0.04141  |  0.04044  |
03/07/2020 01:55:10 AM - INFO - |  3  |  COMPLEMENT_NB  |  37.30%  |  [0.3832 0.3858 0.3834 0.386  0.3776]  |  38.32 (+/- 0.61)  |  0.03622  |  0.01918  |
03/07/2020 01:55:10 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  25.18%  |  [0.2476 0.2484 0.2502 0.248  0.2476]  |  24.84 (+/- 0.19)  |  38.92  |  0.03254  |
03/07/2020 01:55:10 AM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  28.00%  |  [0.3366 0.33   0.3216 0.3218 0.3168]  |  32.54 (+/- 1.41)  |  0.009076  |  27.5  |
03/07/2020 01:55:10 AM - INFO - |  7  |  LINEAR_SVC  |  37.36%  |  [0.3934 0.3974 0.3888 0.3802 0.3858]  |  38.91 (+/- 1.19)  |  1.907  |  0.01799  |
03/07/2020 01:55:10 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  42.04%  |  [0.4282 0.4334 0.4152 0.4194 0.4218]  |  42.36 (+/- 1.29)  |  17.75  |  0.03906  |
03/07/2020 01:55:10 AM - INFO - |  9  |  MULTINOMIAL_NB  |  35.05%  |  [0.353  0.3502 0.3528 0.3514 0.3488]  |  35.12 (+/- 0.32)  |  0.03259  |  0.01869  |
03/07/2020 01:55:10 AM - INFO - |  10  |  NEAREST_CENTROID  |  37.12%  |  [0.3884 0.373  0.3818 0.367  0.372 ]  |  37.64 (+/- 1.53)  |  0.02067  |  0.02279  |
03/07/2020 01:55:10 AM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  33.42%  |  [0.357  0.3594 0.3554 0.3486 0.3494]  |  35.40 (+/- 0.85)  |  1.72  |  0.01868  |
03/07/2020 01:55:10 AM - INFO - |  12  |  PERCEPTRON  |  33.09%  |  [0.3508 0.3604 0.3438 0.3348 0.3414]  |  34.62 (+/- 1.75)  |  0.8017  |  0.01915  |
03/07/2020 01:55:10 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  37.18%  |  [0.3698 0.3676 0.36   0.3652 0.3718]  |  36.69 (+/- 0.82)  |  72.1  |  1.514  |
03/07/2020 01:55:10 AM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 01:55:10 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:55:10 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:55:10 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  38.02%  |  [0.3792 0.379  0.374  0.3704 0.3746]  |  37.54 (+/- 0.66)  |  120.6  |  7.146  |
03/07/2020 01:55:10 AM - INFO - |  2  |  BERNOULLI_NB  |  37.03%  |  [0.377  0.389  0.3782 0.38   0.373 ]  |  37.94 (+/- 1.06)  |  0.04355  |  0.04111  |
03/07/2020 01:55:10 AM - INFO - |  3  |  COMPLEMENT_NB  |  37.34%  |  [0.3878 0.3942 0.3976 0.3938 0.3832]  |  39.13 (+/- 1.03)  |  0.03701  |  0.01924  |
03/07/2020 01:55:10 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  31.11%  |  [0.3162 0.305  0.31   0.299  0.3104]  |  30.81 (+/- 1.16)  |  6.501  |  0.0122  |
03/07/2020 01:55:10 AM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  37.26%  |  [0.3822 0.3916 0.3842 0.386  0.388 ]  |  38.64 (+/- 0.65)  |  0.007525  |  27.27  |
03/07/2020 01:55:10 AM - INFO - |  7  |  LINEAR_SVC  |  40.76%  |  [0.4108 0.4212 0.406  0.3992 0.4082]  |  40.91 (+/- 1.44)  |  0.5699  |  0.01858  |
03/07/2020 01:55:10 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  42.04%  |  [0.4282 0.4334 0.4152 0.4194 0.4218]  |  42.36 (+/- 1.29)  |  17.79  |  0.03888  |
03/07/2020 01:55:10 AM - INFO - |  9  |  MULTINOMIAL_NB  |  37.82%  |  [0.389  0.3928 0.3918 0.3942 0.386 ]  |  39.08 (+/- 0.59)  |  0.03721  |  0.01845  |
03/07/2020 01:55:10 AM - INFO - |  10  |  NEAREST_CENTROID  |  37.33%  |  [0.3872 0.3786 0.3894 0.3672 0.3782]  |  38.01 (+/- 1.57)  |  0.02343  |  0.03291  |
03/07/2020 01:55:10 AM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  41.47%  |  [0.4166 0.426  0.4118 0.408  0.4144]  |  41.54 (+/- 1.21)  |  0.796  |  0.01899  |
03/07/2020 01:55:10 AM - INFO - |  12  |  PERCEPTRON  |  31.60%  |  [0.3364 0.3094 0.3268 0.298  0.2964]  |  31.34 (+/- 3.16)  |  1.034  |  0.01938  |
03/07/2020 01:55:10 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  37.90%  |  [0.3758 0.379  0.3742 0.3758 0.3716]  |  37.53 (+/- 0.48)  |  45.27  |  2.492  |
03/07/2020 01:55:10 AM - INFO - 

03/07/2020 01:55:10 AM - INFO - ################################################################################
03/07/2020 01:55:10 AM - INFO - 13)
03/07/2020 01:55:10 AM - INFO - ********************************************************************************
03/07/2020 01:55:10 AM - INFO - Classifier: RIDGE_CLASSIFIER, Dataset: IMDB_REVIEWS
03/07/2020 01:55:10 AM - INFO - ********************************************************************************
03/07/2020 01:55:17 AM - INFO - 

Performing grid search...

03/07/2020 01:55:17 AM - INFO - Parameters:
03/07/2020 01:55:17 AM - INFO - {'classifier__alpha': [0.5, 1.0], 'classifier__tol': [0.0001, 0.001]}
03/07/2020 01:55:30 AM - INFO - 	Done in 12.710s
03/07/2020 01:55:30 AM - INFO - 	Best score: 0.402
03/07/2020 01:55:30 AM - INFO - 	Best parameters set:
03/07/2020 01:55:30 AM - INFO - 		classifier__alpha: 1.0
03/07/2020 01:55:30 AM - INFO - 		classifier__tol: 0.001
03/07/2020 01:55:30 AM - INFO - 

USING RIDGE_CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 01:55:30 AM - INFO - ________________________________________________________________________________
03/07/2020 01:55:30 AM - INFO - Training: 
03/07/2020 01:55:30 AM - INFO - RidgeClassifier(alpha=1.0, class_weight=None, copy_X=True, fit_intercept=True,
                max_iter=None, normalize=False, random_state=None,
                solver='auto', tol=0.001)
03/07/2020 01:55:33 AM - INFO - Train time: 3.000s
03/07/2020 01:55:33 AM - INFO - Test time:  0.041s
03/07/2020 01:55:33 AM - INFO - Accuracy score:   0.386
03/07/2020 01:55:33 AM - INFO - 

===> Classification Report:

03/07/2020 01:55:33 AM - INFO -               precision    recall  f1-score   support

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

03/07/2020 01:55:33 AM - INFO - 

Cross validation:
03/07/2020 01:55:35 AM - INFO - 	accuracy: 5-fold cross validation: [0.4036 0.4074 0.402  0.3954 0.4   ]
03/07/2020 01:55:35 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 40.17 (+/- 0.79)
03/07/2020 01:55:35 AM - INFO - 

USING RIDGE_CLASSIFIER WITH BEST PARAMETERS: {'alpha': 1.0, 'tol': 0.001}
03/07/2020 01:55:35 AM - INFO - ________________________________________________________________________________
03/07/2020 01:55:35 AM - INFO - Training: 
03/07/2020 01:55:35 AM - INFO - RidgeClassifier(alpha=1.0, class_weight=None, copy_X=True, fit_intercept=True,
                max_iter=None, normalize=False, random_state=None,
                solver='auto', tol=0.001)
03/07/2020 01:55:38 AM - INFO - Train time: 2.980s
03/07/2020 01:55:38 AM - INFO - Test time:  0.041s
03/07/2020 01:55:38 AM - INFO - Accuracy score:   0.386
03/07/2020 01:55:38 AM - INFO - 

===> Classification Report:

03/07/2020 01:55:38 AM - INFO -               precision    recall  f1-score   support

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

03/07/2020 01:55:38 AM - INFO - 

Cross validation:
03/07/2020 01:55:40 AM - INFO - 	accuracy: 5-fold cross validation: [0.4036 0.4074 0.402  0.3954 0.4   ]
03/07/2020 01:55:40 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 40.17 (+/- 0.79)
03/07/2020 01:55:40 AM - INFO - It took 29.96195387840271 seconds
03/07/2020 01:55:40 AM - INFO - ********************************************************************************
03/07/2020 01:55:40 AM - INFO - ################################################################################
03/07/2020 01:55:40 AM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 01:55:40 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:55:40 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:55:40 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  35.78%  |  [0.352  0.3546 0.3442 0.3492 0.3464]  |  34.93 (+/- 0.75)  |  12.15  |  0.7267  |
03/07/2020 01:55:40 AM - INFO - |  2  |  BERNOULLI_NB  |  37.25%  |  [0.3786 0.3812 0.3678 0.374  0.373 ]  |  37.49 (+/- 0.93)  |  0.04141  |  0.04044  |
03/07/2020 01:55:40 AM - INFO - |  3  |  COMPLEMENT_NB  |  37.30%  |  [0.3832 0.3858 0.3834 0.386  0.3776]  |  38.32 (+/- 0.61)  |  0.03622  |  0.01918  |
03/07/2020 01:55:40 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  25.18%  |  [0.2476 0.2484 0.2502 0.248  0.2476]  |  24.84 (+/- 0.19)  |  38.92  |  0.03254  |
03/07/2020 01:55:40 AM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  28.00%  |  [0.3366 0.33   0.3216 0.3218 0.3168]  |  32.54 (+/- 1.41)  |  0.009076  |  27.5  |
03/07/2020 01:55:40 AM - INFO - |  7  |  LINEAR_SVC  |  37.36%  |  [0.3934 0.3974 0.3888 0.3802 0.3858]  |  38.91 (+/- 1.19)  |  1.907  |  0.01799  |
03/07/2020 01:55:40 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  42.04%  |  [0.4282 0.4334 0.4152 0.4194 0.4218]  |  42.36 (+/- 1.29)  |  17.75  |  0.03906  |
03/07/2020 01:55:40 AM - INFO - |  9  |  MULTINOMIAL_NB  |  35.05%  |  [0.353  0.3502 0.3528 0.3514 0.3488]  |  35.12 (+/- 0.32)  |  0.03259  |  0.01869  |
03/07/2020 01:55:40 AM - INFO - |  10  |  NEAREST_CENTROID  |  37.12%  |  [0.3884 0.373  0.3818 0.367  0.372 ]  |  37.64 (+/- 1.53)  |  0.02067  |  0.02279  |
03/07/2020 01:55:40 AM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  33.42%  |  [0.357  0.3594 0.3554 0.3486 0.3494]  |  35.40 (+/- 0.85)  |  1.72  |  0.01868  |
03/07/2020 01:55:40 AM - INFO - |  12  |  PERCEPTRON  |  33.09%  |  [0.3508 0.3604 0.3438 0.3348 0.3414]  |  34.62 (+/- 1.75)  |  0.8017  |  0.01915  |
03/07/2020 01:55:40 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  37.18%  |  [0.3698 0.3676 0.36   0.3652 0.3718]  |  36.69 (+/- 0.82)  |  72.1  |  1.514  |
03/07/2020 01:55:40 AM - INFO - |  14  |  RIDGE_CLASSIFIER  |  38.56%  |  [0.4036 0.4074 0.402  0.3954 0.4   ]  |  40.17 (+/- 0.79)  |  3.0  |  0.04056  |
03/07/2020 01:55:40 AM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 01:55:40 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:55:40 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:55:40 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  38.02%  |  [0.3792 0.379  0.374  0.3704 0.3746]  |  37.54 (+/- 0.66)  |  120.6  |  7.146  |
03/07/2020 01:55:40 AM - INFO - |  2  |  BERNOULLI_NB  |  37.03%  |  [0.377  0.389  0.3782 0.38   0.373 ]  |  37.94 (+/- 1.06)  |  0.04355  |  0.04111  |
03/07/2020 01:55:40 AM - INFO - |  3  |  COMPLEMENT_NB  |  37.34%  |  [0.3878 0.3942 0.3976 0.3938 0.3832]  |  39.13 (+/- 1.03)  |  0.03701  |  0.01924  |
03/07/2020 01:55:40 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  31.11%  |  [0.3162 0.305  0.31   0.299  0.3104]  |  30.81 (+/- 1.16)  |  6.501  |  0.0122  |
03/07/2020 01:55:40 AM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  37.26%  |  [0.3822 0.3916 0.3842 0.386  0.388 ]  |  38.64 (+/- 0.65)  |  0.007525  |  27.27  |
03/07/2020 01:55:40 AM - INFO - |  7  |  LINEAR_SVC  |  40.76%  |  [0.4108 0.4212 0.406  0.3992 0.4082]  |  40.91 (+/- 1.44)  |  0.5699  |  0.01858  |
03/07/2020 01:55:40 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  42.04%  |  [0.4282 0.4334 0.4152 0.4194 0.4218]  |  42.36 (+/- 1.29)  |  17.79  |  0.03888  |
03/07/2020 01:55:40 AM - INFO - |  9  |  MULTINOMIAL_NB  |  37.82%  |  [0.389  0.3928 0.3918 0.3942 0.386 ]  |  39.08 (+/- 0.59)  |  0.03721  |  0.01845  |
03/07/2020 01:55:40 AM - INFO - |  10  |  NEAREST_CENTROID  |  37.33%  |  [0.3872 0.3786 0.3894 0.3672 0.3782]  |  38.01 (+/- 1.57)  |  0.02343  |  0.03291  |
03/07/2020 01:55:40 AM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  41.47%  |  [0.4166 0.426  0.4118 0.408  0.4144]  |  41.54 (+/- 1.21)  |  0.796  |  0.01899  |
03/07/2020 01:55:40 AM - INFO - |  12  |  PERCEPTRON  |  31.60%  |  [0.3364 0.3094 0.3268 0.298  0.2964]  |  31.34 (+/- 3.16)  |  1.034  |  0.01938  |
03/07/2020 01:55:40 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  37.90%  |  [0.3758 0.379  0.3742 0.3758 0.3716]  |  37.53 (+/- 0.48)  |  45.27  |  2.492  |
03/07/2020 01:55:40 AM - INFO - |  14  |  RIDGE_CLASSIFIER  |  38.56%  |  [0.4036 0.4074 0.402  0.3954 0.4   ]  |  40.17 (+/- 0.79)  |  2.98  |  0.04123  |
03/07/2020 01:55:40 AM - INFO - 

03/07/2020 01:55:40 AM - INFO - ################################################################################
03/07/2020 01:55:40 AM - INFO - 14)
03/07/2020 01:55:40 AM - INFO - ********************************************************************************
03/07/2020 01:55:40 AM - INFO - Classifier: GRADIENT_BOOSTING_CLASSIFIER, Dataset: IMDB_REVIEWS
03/07/2020 01:55:40 AM - INFO - ********************************************************************************
03/07/2020 01:55:47 AM - INFO - 

Performing grid search...

03/07/2020 01:55:47 AM - INFO - Parameters:
03/07/2020 01:55:47 AM - INFO - {'classifier__learning_rate': [0.01, 0.1], 'classifier__n_estimators': [100, 200]}
03/07/2020 03:41:44 AM - INFO - 	Done in 6356.577s
03/07/2020 03:41:44 AM - INFO - 	Best score: 0.374
03/07/2020 03:41:44 AM - INFO - 	Best parameters set:
03/07/2020 03:41:44 AM - INFO - 		classifier__learning_rate: 0.1
03/07/2020 03:41:44 AM - INFO - 		classifier__n_estimators: 200
03/07/2020 03:41:44 AM - INFO - 

USING GRADIENT_BOOSTING_CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 03:41:44 AM - INFO - ________________________________________________________________________________
03/07/2020 03:41:44 AM - INFO - Training: 
03/07/2020 03:41:44 AM - INFO - GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.1, loss='deviance', max_depth=3,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_iter_no_change=None, presort='deprecated',
                           random_state=None, subsample=1.0, tol=0.0001,
                           validation_fraction=0.1, verbose=0,
                           warm_start=False)
03/07/2020 03:48:57 AM - INFO - Train time: 433.142s
03/07/2020 03:48:57 AM - INFO - Test time:  0.278s
03/07/2020 03:48:57 AM - INFO - Accuracy score:   0.373
03/07/2020 03:48:57 AM - INFO - 

===> Classification Report:

03/07/2020 03:48:57 AM - INFO -               precision    recall  f1-score   support

           1       0.47      0.76      0.58      5022
           2       0.18      0.03      0.06      2302
           3       0.26      0.08      0.12      2541
           4       0.28      0.17      0.21      2635
           7       0.27      0.15      0.19      2307
           8       0.22      0.17      0.20      2850
           9       0.15      0.03      0.04      2344
          10       0.38      0.78      0.51      4999

    accuracy                           0.37     25000
   macro avg       0.28      0.27      0.24     25000
weighted avg       0.31      0.37      0.30     25000

03/07/2020 03:48:57 AM - INFO - 

Cross validation:
03/07/2020 04:03:47 AM - INFO - 	accuracy: 5-fold cross validation: [0.377  0.368  0.3602 0.3648 0.368 ]
03/07/2020 04:03:47 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 36.76 (+/- 1.10)
03/07/2020 04:03:47 AM - INFO - 

USING GRADIENT_BOOSTING_CLASSIFIER WITH BEST PARAMETERS: {'learning_rate': 0.1, 'n_estimators': 200}
03/07/2020 04:03:47 AM - INFO - ________________________________________________________________________________
03/07/2020 04:03:47 AM - INFO - Training: 
03/07/2020 04:03:47 AM - INFO - GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.1, loss='deviance', max_depth=3,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=200,
                           n_iter_no_change=None, presort='deprecated',
                           random_state=None, subsample=1.0, tol=0.0001,
                           validation_fraction=0.1, verbose=0,
                           warm_start=False)
03/07/2020 04:18:07 AM - INFO - Train time: 859.912s
03/07/2020 04:18:08 AM - INFO - Test time:  0.494s
03/07/2020 04:18:08 AM - INFO - Accuracy score:   0.381
03/07/2020 04:18:08 AM - INFO - 

===> Classification Report:

03/07/2020 04:18:08 AM - INFO -               precision    recall  f1-score   support

           1       0.49      0.77      0.59      5022
           2       0.20      0.05      0.09      2302
           3       0.24      0.10      0.14      2541
           4       0.28      0.21      0.24      2635
           7       0.27      0.17      0.21      2307
           8       0.22      0.18      0.20      2850
           9       0.16      0.04      0.06      2344
          10       0.41      0.75      0.53      4999

    accuracy                           0.38     25000
   macro avg       0.28      0.28      0.26     25000
weighted avg       0.32      0.38      0.32     25000

03/07/2020 04:18:08 AM - INFO - 

Cross validation:
03/07/2020 04:51:27 AM - INFO - 	accuracy: 5-fold cross validation: [0.379  0.3772 0.3658 0.3688 0.376 ]
03/07/2020 04:51:27 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 37.34 (+/- 1.03)
03/07/2020 04:51:27 AM - INFO - It took 10546.406470298767 seconds
03/07/2020 04:51:27 AM - INFO - ********************************************************************************
03/07/2020 04:51:27 AM - INFO - ################################################################################
03/07/2020 04:51:27 AM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 04:51:27 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 04:51:27 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 04:51:27 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  35.78%  |  [0.352  0.3546 0.3442 0.3492 0.3464]  |  34.93 (+/- 0.75)  |  12.15  |  0.7267  |
03/07/2020 04:51:27 AM - INFO - |  2  |  BERNOULLI_NB  |  37.25%  |  [0.3786 0.3812 0.3678 0.374  0.373 ]  |  37.49 (+/- 0.93)  |  0.04141  |  0.04044  |
03/07/2020 04:51:27 AM - INFO - |  3  |  COMPLEMENT_NB  |  37.30%  |  [0.3832 0.3858 0.3834 0.386  0.3776]  |  38.32 (+/- 0.61)  |  0.03622  |  0.01918  |
03/07/2020 04:51:27 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  25.18%  |  [0.2476 0.2484 0.2502 0.248  0.2476]  |  24.84 (+/- 0.19)  |  38.92  |  0.03254  |
03/07/2020 04:51:27 AM - INFO - |  5  |  GRADIENT_BOOSTING_CLASSIFIER  |  37.31%  |  [0.377  0.368  0.3602 0.3648 0.368 ]  |  36.76 (+/- 1.10)  |  433.1  |  0.2777  |
03/07/2020 04:51:27 AM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  28.00%  |  [0.3366 0.33   0.3216 0.3218 0.3168]  |  32.54 (+/- 1.41)  |  0.009076  |  27.5  |
03/07/2020 04:51:27 AM - INFO - |  7  |  LINEAR_SVC  |  37.36%  |  [0.3934 0.3974 0.3888 0.3802 0.3858]  |  38.91 (+/- 1.19)  |  1.907  |  0.01799  |
03/07/2020 04:51:27 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  42.04%  |  [0.4282 0.4334 0.4152 0.4194 0.4218]  |  42.36 (+/- 1.29)  |  17.75  |  0.03906  |
03/07/2020 04:51:27 AM - INFO - |  9  |  MULTINOMIAL_NB  |  35.05%  |  [0.353  0.3502 0.3528 0.3514 0.3488]  |  35.12 (+/- 0.32)  |  0.03259  |  0.01869  |
03/07/2020 04:51:27 AM - INFO - |  10  |  NEAREST_CENTROID  |  37.12%  |  [0.3884 0.373  0.3818 0.367  0.372 ]  |  37.64 (+/- 1.53)  |  0.02067  |  0.02279  |
03/07/2020 04:51:27 AM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  33.42%  |  [0.357  0.3594 0.3554 0.3486 0.3494]  |  35.40 (+/- 0.85)  |  1.72  |  0.01868  |
03/07/2020 04:51:27 AM - INFO - |  12  |  PERCEPTRON  |  33.09%  |  [0.3508 0.3604 0.3438 0.3348 0.3414]  |  34.62 (+/- 1.75)  |  0.8017  |  0.01915  |
03/07/2020 04:51:27 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  37.18%  |  [0.3698 0.3676 0.36   0.3652 0.3718]  |  36.69 (+/- 0.82)  |  72.1  |  1.514  |
03/07/2020 04:51:27 AM - INFO - |  14  |  RIDGE_CLASSIFIER  |  38.56%  |  [0.4036 0.4074 0.402  0.3954 0.4   ]  |  40.17 (+/- 0.79)  |  3.0  |  0.04056  |
03/07/2020 04:51:27 AM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 04:51:27 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 04:51:27 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 04:51:27 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  38.02%  |  [0.3792 0.379  0.374  0.3704 0.3746]  |  37.54 (+/- 0.66)  |  120.6  |  7.146  |
03/07/2020 04:51:27 AM - INFO - |  2  |  BERNOULLI_NB  |  37.03%  |  [0.377  0.389  0.3782 0.38   0.373 ]  |  37.94 (+/- 1.06)  |  0.04355  |  0.04111  |
03/07/2020 04:51:27 AM - INFO - |  3  |  COMPLEMENT_NB  |  37.34%  |  [0.3878 0.3942 0.3976 0.3938 0.3832]  |  39.13 (+/- 1.03)  |  0.03701  |  0.01924  |
03/07/2020 04:51:27 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  31.11%  |  [0.3162 0.305  0.31   0.299  0.3104]  |  30.81 (+/- 1.16)  |  6.501  |  0.0122  |
03/07/2020 04:51:27 AM - INFO - |  5  |  GRADIENT_BOOSTING_CLASSIFIER  |  38.08%  |  [0.379  0.3772 0.3658 0.3688 0.376 ]  |  37.34 (+/- 1.03)  |  859.9  |  0.4938  |
03/07/2020 04:51:27 AM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  37.26%  |  [0.3822 0.3916 0.3842 0.386  0.388 ]  |  38.64 (+/- 0.65)  |  0.007525  |  27.27  |
03/07/2020 04:51:27 AM - INFO - |  7  |  LINEAR_SVC  |  40.76%  |  [0.4108 0.4212 0.406  0.3992 0.4082]  |  40.91 (+/- 1.44)  |  0.5699  |  0.01858  |
03/07/2020 04:51:27 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  42.04%  |  [0.4282 0.4334 0.4152 0.4194 0.4218]  |  42.36 (+/- 1.29)  |  17.79  |  0.03888  |
03/07/2020 04:51:27 AM - INFO - |  9  |  MULTINOMIAL_NB  |  37.82%  |  [0.389  0.3928 0.3918 0.3942 0.386 ]  |  39.08 (+/- 0.59)  |  0.03721  |  0.01845  |
03/07/2020 04:51:27 AM - INFO - |  10  |  NEAREST_CENTROID  |  37.33%  |  [0.3872 0.3786 0.3894 0.3672 0.3782]  |  38.01 (+/- 1.57)  |  0.02343  |  0.03291  |
03/07/2020 04:51:27 AM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  41.47%  |  [0.4166 0.426  0.4118 0.408  0.4144]  |  41.54 (+/- 1.21)  |  0.796  |  0.01899  |
03/07/2020 04:51:27 AM - INFO - |  12  |  PERCEPTRON  |  31.60%  |  [0.3364 0.3094 0.3268 0.298  0.2964]  |  31.34 (+/- 3.16)  |  1.034  |  0.01938  |
03/07/2020 04:51:27 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  37.90%  |  [0.3758 0.379  0.3742 0.3758 0.3716]  |  37.53 (+/- 0.48)  |  45.27  |  2.492  |
03/07/2020 04:51:27 AM - INFO - |  14  |  RIDGE_CLASSIFIER  |  38.56%  |  [0.4036 0.4074 0.402  0.3954 0.4   ]  |  40.17 (+/- 0.79)  |  2.98  |  0.04123  |
03/07/2020 04:51:27 AM - INFO - 

FINAL CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 04:51:27 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 04:51:27 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 04:51:27 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  35.78%  |  [0.352  0.3546 0.3442 0.3492 0.3464]  |  34.93 (+/- 0.75)  |  12.15  |  0.7267  |
03/07/2020 04:51:27 AM - INFO - |  2  |  BERNOULLI_NB  |  37.25%  |  [0.3786 0.3812 0.3678 0.374  0.373 ]  |  37.49 (+/- 0.93)  |  0.04141  |  0.04044  |
03/07/2020 04:51:27 AM - INFO - |  3  |  COMPLEMENT_NB  |  37.30%  |  [0.3832 0.3858 0.3834 0.386  0.3776]  |  38.32 (+/- 0.61)  |  0.03622  |  0.01918  |
03/07/2020 04:51:27 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  25.18%  |  [0.2476 0.2484 0.2502 0.248  0.2476]  |  24.84 (+/- 0.19)  |  38.92  |  0.03254  |
03/07/2020 04:51:27 AM - INFO - |  5  |  GRADIENT_BOOSTING_CLASSIFIER  |  37.31%  |  [0.377  0.368  0.3602 0.3648 0.368 ]  |  36.76 (+/- 1.10)  |  433.1  |  0.2777  |
03/07/2020 04:51:27 AM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  28.00%  |  [0.3366 0.33   0.3216 0.3218 0.3168]  |  32.54 (+/- 1.41)  |  0.009076  |  27.5  |
03/07/2020 04:51:27 AM - INFO - |  7  |  LINEAR_SVC  |  37.36%  |  [0.3934 0.3974 0.3888 0.3802 0.3858]  |  38.91 (+/- 1.19)  |  1.907  |  0.01799  |
03/07/2020 04:51:27 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  42.04%  |  [0.4282 0.4334 0.4152 0.4194 0.4218]  |  42.36 (+/- 1.29)  |  17.75  |  0.03906  |
03/07/2020 04:51:27 AM - INFO - |  9  |  MULTINOMIAL_NB  |  35.05%  |  [0.353  0.3502 0.3528 0.3514 0.3488]  |  35.12 (+/- 0.32)  |  0.03259  |  0.01869  |
03/07/2020 04:51:27 AM - INFO - |  10  |  NEAREST_CENTROID  |  37.12%  |  [0.3884 0.373  0.3818 0.367  0.372 ]  |  37.64 (+/- 1.53)  |  0.02067  |  0.02279  |
03/07/2020 04:51:27 AM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  33.42%  |  [0.357  0.3594 0.3554 0.3486 0.3494]  |  35.40 (+/- 0.85)  |  1.72  |  0.01868  |
03/07/2020 04:51:27 AM - INFO - |  12  |  PERCEPTRON  |  33.09%  |  [0.3508 0.3604 0.3438 0.3348 0.3414]  |  34.62 (+/- 1.75)  |  0.8017  |  0.01915  |
03/07/2020 04:51:27 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  37.18%  |  [0.3698 0.3676 0.36   0.3652 0.3718]  |  36.69 (+/- 0.82)  |  72.1  |  1.514  |
03/07/2020 04:51:27 AM - INFO - |  14  |  RIDGE_CLASSIFIER  |  38.56%  |  [0.4036 0.4074 0.402  0.3954 0.4   ]  |  40.17 (+/- 0.79)  |  3.0  |  0.04056  |
03/07/2020 04:51:27 AM - INFO - 

FINAL CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 04:51:27 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 04:51:27 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 04:51:27 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  38.02%  |  [0.3792 0.379  0.374  0.3704 0.3746]  |  37.54 (+/- 0.66)  |  120.6  |  7.146  |
03/07/2020 04:51:27 AM - INFO - |  2  |  BERNOULLI_NB  |  37.03%  |  [0.377  0.389  0.3782 0.38   0.373 ]  |  37.94 (+/- 1.06)  |  0.04355  |  0.04111  |
03/07/2020 04:51:27 AM - INFO - |  3  |  COMPLEMENT_NB  |  37.34%  |  [0.3878 0.3942 0.3976 0.3938 0.3832]  |  39.13 (+/- 1.03)  |  0.03701  |  0.01924  |
03/07/2020 04:51:27 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  31.11%  |  [0.3162 0.305  0.31   0.299  0.3104]  |  30.81 (+/- 1.16)  |  6.501  |  0.0122  |
03/07/2020 04:51:27 AM - INFO - |  5  |  GRADIENT_BOOSTING_CLASSIFIER  |  38.08%  |  [0.379  0.3772 0.3658 0.3688 0.376 ]  |  37.34 (+/- 1.03)  |  859.9  |  0.4938  |
03/07/2020 04:51:27 AM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  37.26%  |  [0.3822 0.3916 0.3842 0.386  0.388 ]  |  38.64 (+/- 0.65)  |  0.007525  |  27.27  |
03/07/2020 04:51:27 AM - INFO - |  7  |  LINEAR_SVC  |  40.76%  |  [0.4108 0.4212 0.406  0.3992 0.4082]  |  40.91 (+/- 1.44)  |  0.5699  |  0.01858  |
03/07/2020 04:51:27 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  42.04%  |  [0.4282 0.4334 0.4152 0.4194 0.4218]  |  42.36 (+/- 1.29)  |  17.79  |  0.03888  |
03/07/2020 04:51:27 AM - INFO - |  9  |  MULTINOMIAL_NB  |  37.82%  |  [0.389  0.3928 0.3918 0.3942 0.386 ]  |  39.08 (+/- 0.59)  |  0.03721  |  0.01845  |
03/07/2020 04:51:27 AM - INFO - |  10  |  NEAREST_CENTROID  |  37.33%  |  [0.3872 0.3786 0.3894 0.3672 0.3782]  |  38.01 (+/- 1.57)  |  0.02343  |  0.03291  |
03/07/2020 04:51:27 AM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  41.47%  |  [0.4166 0.426  0.4118 0.408  0.4144]  |  41.54 (+/- 1.21)  |  0.796  |  0.01899  |
03/07/2020 04:51:27 AM - INFO - |  12  |  PERCEPTRON  |  31.60%  |  [0.3364 0.3094 0.3268 0.298  0.2964]  |  31.34 (+/- 3.16)  |  1.034  |  0.01938  |
03/07/2020 04:51:27 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  37.90%  |  [0.3758 0.379  0.3742 0.3758 0.3716]  |  37.53 (+/- 0.48)  |  45.27  |  2.492  |
03/07/2020 04:51:27 AM - INFO - |  14  |  RIDGE_CLASSIFIER  |  38.56%  |  [0.4036 0.4074 0.402  0.3954 0.4   ]  |  40.17 (+/- 0.79)  |  2.98  |  0.04123  |
```