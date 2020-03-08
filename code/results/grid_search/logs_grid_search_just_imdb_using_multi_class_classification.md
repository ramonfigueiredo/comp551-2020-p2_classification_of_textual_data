## Grid search logs: Multi-class Classification

### IMDB using Multi-class Classification

#### FINAL CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS

| ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
| -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
|  1  |  ADA_BOOST_CLASSIFIER  |  35.78%  |  [0.352  0.3546 0.3442 0.3492 0.3464]  |  34.93 (+/- 0.75)  |  12.08  |  0.732  |
|  2  |  BERNOULLI_NB  |  37.25%  |  [0.3786 0.3812 0.3678 0.374  0.373 ]  |  37.49 (+/- 0.93)  |  0.03953  |  0.04032  |
|  3  |  COMPLEMENT_NB  |  37.30%  |  [0.3832 0.3858 0.3834 0.386  0.3776]  |  38.32 (+/- 0.61)  |  0.03533  |  0.0185  |
|  4  |  DECISION_TREE_CLASSIFIER  |  25.04%  |  [0.2516 0.2634 0.2528 0.249  0.2542]  |  25.42 (+/- 0.98)  |  38.77  |  0.03135  |
|  5  |  GRADIENT_BOOSTING_CLASSIFIER  |  37.46%  |  [0.3746 0.367  0.3602 0.3652 0.3682]  |  36.70 (+/- 0.93)  |  430.9  |  0.2771  |
|  6  |  K_NEIGHBORS_CLASSIFIER  |  28.00%  |  [0.3366 0.33   0.3216 0.3218 0.3168]  |  32.54 (+/- 1.41)  |  0.008613  |  27.45  |
|  7  |  LINEAR_SVC  |  37.36%  |  [0.3934 0.3974 0.3888 0.3802 0.3858]  |  38.91 (+/- 1.19)  |  1.893  |  0.01857  |
|  8  |  LOGISTIC_REGRESSION  |  42.04%  |  [0.4282 0.4334 0.4152 0.4194 0.4218]  |  42.36 (+/- 1.29)  |  17.71  |  0.03946  |
|  9  |  MULTINOMIAL_NB  |  35.05%  |  [0.353  0.3502 0.3528 0.3514 0.3488]  |  35.12 (+/- 0.32)  |  0.03723  |  0.02325  |
|  10  |  NEAREST_CENTROID  |  37.12%  |  [0.3884 0.373  0.3818 0.367  0.372 ]  |  37.64 (+/- 1.53)  |  0.02274  |  0.02617  |
|  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  33.42%  |  [0.3546 0.3592 0.354  0.3474 0.3492]  |  35.29 (+/- 0.84)  |  1.618  |  0.01907  |
|  12  |  PERCEPTRON  |  33.09%  |  [0.3508 0.3604 0.3438 0.3348 0.3414]  |  34.62 (+/- 1.75)  |  0.8114  |  0.02037  |
|  13  |  RANDOM_FOREST_CLASSIFIER  |  37.19%  |  [0.3676 0.3678 0.369  0.362  0.3684]  |  36.70 (+/- 0.51)  |  72.59  |  1.511  |
|  14  |  RIDGE_CLASSIFIER  |  38.56%  |  [0.4036 0.4074 0.402  0.3954 0.4   ]  |  40.17 (+/- 0.79)  |  2.93  |  0.04089  |


#### FINAL CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS

| ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
| -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
|  1  |  ADA_BOOST_CLASSIFIER  |  38.02%  |  [0.3792 0.379  0.374  0.3704 0.3746]  |  37.54 (+/- 0.66)  |  120.6  |  7.117  |
|  2  |  BERNOULLI_NB  |  37.03%  |  [0.377  0.389  0.3782 0.38   0.373 ]  |  37.94 (+/- 1.06)  |  0.0429  |  0.04091  |
|  3  |  COMPLEMENT_NB  |  37.34%  |  [0.3878 0.3942 0.3976 0.3938 0.3832]  |  39.13 (+/- 1.03)  |  0.03774  |  0.01899  |
|  4  |  DECISION_TREE_CLASSIFIER  |  30.69%  |  [0.3152 0.3036 0.3054 0.3154 0.3048]  |  30.89 (+/- 1.05)  |  6.42  |  0.01237  |
|  5  |  GRADIENT_BOOSTING_CLASSIFIER  |  37.86%  |  [0.3766 0.3724 0.3666 0.3708 0.3784]  |  37.30 (+/- 0.84)  |  863.3  |  0.4914  |
|  6  |  K_NEIGHBORS_CLASSIFIER  |  37.26%  |  [0.3822 0.3916 0.3842 0.386  0.388 ]  |  38.64 (+/- 0.65)  |  0.008674  |  27.41  |
|  7  |  LINEAR_SVC  |  40.75%  |  [0.4114 0.4206 0.4068 0.3992 0.4088]  |  40.94 (+/- 1.39)  |  0.6048  |  0.01785  |
|  8  |  LOGISTIC_REGRESSION  |  42.04%  |  [0.4282 0.4334 0.4152 0.4194 0.4218]  |  42.36 (+/- 1.29)  |  17.84  |  0.03951  |
|  9  |  MULTINOMIAL_NB  |  37.82%  |  [0.389  0.3928 0.3918 0.3942 0.386 ]  |  39.08 (+/- 0.59)  |  0.03643  |  0.02055  |
|  10  |  NEAREST_CENTROID  |  37.33%  |  [0.3872 0.3786 0.3894 0.3672 0.3782]  |  38.01 (+/- 1.57)  |  0.02111  |  0.03207  |
|  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  41.83%  |  [0.4144 0.427  0.4118 0.4142 0.4138]  |  41.62 (+/- 1.09)  |  1.373  |  0.01845  |
|  12  |  PERCEPTRON  |  31.60%  |  [0.3364 0.3094 0.3268 0.298  0.2964]  |  31.34 (+/- 3.16)  |  1.051  |  0.01868  |
|  13  |  RANDOM_FOREST_CLASSIFIER  |  37.82%  |  [0.3716 0.3772 0.3768 0.3742 0.368 ]  |  37.36 (+/- 0.69)  |  42.89  |  2.501  |
|  14  |  RIDGE_CLASSIFIER  |  38.56%  |  [0.4036 0.4074 0.402  0.3954 0.4   ]  |  40.17 (+/- 0.79)  |  2.896  |  0.04107  |


#### Computer settings used to run

* Operating system: Ubuntu 18.04.4 LTS (64-bit)
* Processor: Intel® Core™ i7-7700 CPU @ 3.60GHz × 8 
* Memory: 32 GB

#### All logs:

```
03/07/2020 05:16:58 PM - INFO - 
>>> GRID SEARCH
03/07/2020 05:16:58 PM - INFO - 

03/07/2020 05:16:58 PM - INFO - ################################################################################
03/07/2020 05:16:58 PM - INFO - 1)
03/07/2020 05:16:58 PM - INFO - ********************************************************************************
03/07/2020 05:16:58 PM - INFO - Classifier: ADA_BOOST_CLASSIFIER, Dataset: IMDB_REVIEWS
03/07/2020 05:16:58 PM - INFO - ********************************************************************************
03/07/2020 05:17:05 PM - INFO - 

Performing grid search...

03/07/2020 05:17:05 PM - INFO - Parameters:
03/07/2020 05:17:05 PM - INFO - {'classifier__learning_rate': [0.1, 1], 'classifier__n_estimators': [200, 500]}
03/07/2020 05:38:05 PM - INFO - 	Done in 1260.387s
03/07/2020 05:38:05 PM - INFO - 	Best score: 0.375
03/07/2020 05:38:05 PM - INFO - 	Best parameters set:
03/07/2020 05:38:05 PM - INFO - 		classifier__learning_rate: 0.1
03/07/2020 05:38:05 PM - INFO - 		classifier__n_estimators: 500
03/07/2020 05:38:05 PM - INFO - 

USING ADA_BOOST_CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 05:38:05 PM - INFO - ________________________________________________________________________________
03/07/2020 05:38:05 PM - INFO - Training: 
03/07/2020 05:38:05 PM - INFO - AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
                   n_estimators=50, random_state=None)
03/07/2020 05:38:17 PM - INFO - Train time: 12.081s
03/07/2020 05:38:18 PM - INFO - Test time:  0.732s
03/07/2020 05:38:18 PM - INFO - Accuracy score:   0.358
03/07/2020 05:38:18 PM - INFO - 

===> Classification Report:

03/07/2020 05:38:18 PM - INFO -               precision    recall  f1-score   support

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

03/07/2020 05:38:18 PM - INFO - 

Cross validation:
03/07/2020 05:39:05 PM - INFO - 	accuracy: 5-fold cross validation: [0.352  0.3546 0.3442 0.3492 0.3464]
03/07/2020 05:39:05 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 34.93 (+/- 0.75)
03/07/2020 05:39:05 PM - INFO - 

USING ADA_BOOST_CLASSIFIER WITH BEST PARAMETERS: {'learning_rate': 0.1, 'n_estimators': 500}
03/07/2020 05:39:05 PM - INFO - ________________________________________________________________________________
03/07/2020 05:39:05 PM - INFO - Training: 
03/07/2020 05:39:05 PM - INFO - AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=0.1,
                   n_estimators=500, random_state=None)
03/07/2020 05:41:05 PM - INFO - Train time: 120.643s
03/07/2020 05:41:12 PM - INFO - Test time:  7.117s
03/07/2020 05:41:12 PM - INFO - Accuracy score:   0.380
03/07/2020 05:41:12 PM - INFO - 

===> Classification Report:

03/07/2020 05:41:13 PM - INFO -               precision    recall  f1-score   support

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

03/07/2020 05:41:13 PM - INFO - 

Cross validation:
03/07/2020 05:48:55 PM - INFO - 	accuracy: 5-fold cross validation: [0.3792 0.379  0.374  0.3704 0.3746]
03/07/2020 05:48:55 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 37.54 (+/- 0.66)
03/07/2020 05:48:55 PM - INFO - It took 1917.1703777313232 seconds
03/07/2020 05:48:55 PM - INFO - ********************************************************************************
03/07/2020 05:48:55 PM - INFO - ################################################################################
03/07/2020 05:48:55 PM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 05:48:55 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 05:48:55 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 05:48:55 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  35.78%  |  [0.352  0.3546 0.3442 0.3492 0.3464]  |  34.93 (+/- 0.75)  |  12.08  |  0.732  |
03/07/2020 05:48:55 PM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 05:48:55 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 05:48:55 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 05:48:55 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  38.02%  |  [0.3792 0.379  0.374  0.3704 0.3746]  |  37.54 (+/- 0.66)  |  120.6  |  7.117  |
03/07/2020 05:48:55 PM - INFO - 

03/07/2020 05:48:55 PM - INFO - ################################################################################
03/07/2020 05:48:55 PM - INFO - 2)
03/07/2020 05:48:55 PM - INFO - ********************************************************************************
03/07/2020 05:48:55 PM - INFO - Classifier: DECISION_TREE_CLASSIFIER, Dataset: IMDB_REVIEWS
03/07/2020 05:48:55 PM - INFO - ********************************************************************************
03/07/2020 05:49:01 PM - INFO - 

Performing grid search...

03/07/2020 05:49:01 PM - INFO - Parameters:
03/07/2020 05:49:01 PM - INFO - {'classifier__criterion': ['entropy', 'gini'], 'classifier__splitter': ['best', 'random'], 'classifier__min_samples_split': [2, 100, 250]}
03/07/2020 05:52:37 PM - INFO - 	Done in 215.674s
03/07/2020 05:52:37 PM - INFO - 	Best score: 0.304
03/07/2020 05:52:37 PM - INFO - 	Best parameters set:
03/07/2020 05:52:37 PM - INFO - 		classifier__criterion: 'entropy'
03/07/2020 05:52:37 PM - INFO - 		classifier__min_samples_split: 250
03/07/2020 05:52:37 PM - INFO - 		classifier__splitter: 'random'
03/07/2020 05:52:37 PM - INFO - 

USING DECISION_TREE_CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 05:52:37 PM - INFO - ________________________________________________________________________________
03/07/2020 05:52:37 PM - INFO - Training: 
03/07/2020 05:52:37 PM - INFO - DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=None, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=None, splitter='best')
03/07/2020 05:53:16 PM - INFO - Train time: 38.770s
03/07/2020 05:53:16 PM - INFO - Test time:  0.031s
03/07/2020 05:53:16 PM - INFO - Accuracy score:   0.250
03/07/2020 05:53:16 PM - INFO - 

===> Classification Report:

03/07/2020 05:53:16 PM - INFO -               precision    recall  f1-score   support

           1       0.41      0.45      0.43      5022
           2       0.13      0.12      0.13      2302
           3       0.15      0.14      0.14      2541
           4       0.16      0.16      0.16      2635
           7       0.12      0.13      0.13      2307
           8       0.17      0.17      0.17      2850
           9       0.14      0.12      0.13      2344
          10       0.37      0.38      0.37      4999

    accuracy                           0.25     25000
   macro avg       0.21      0.21      0.21     25000
weighted avg       0.24      0.25      0.25     25000

03/07/2020 05:53:16 PM - INFO - 

Cross validation:
03/07/2020 05:53:57 PM - INFO - 	accuracy: 5-fold cross validation: [0.2516 0.2634 0.2528 0.249  0.2542]
03/07/2020 05:53:57 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 25.42 (+/- 0.98)
03/07/2020 05:53:57 PM - INFO - 

USING DECISION_TREE_CLASSIFIER WITH BEST PARAMETERS: {'criterion': 'entropy', 'min_samples_split': 250, 'splitter': 'random'}
03/07/2020 05:53:57 PM - INFO - ________________________________________________________________________________
03/07/2020 05:53:57 PM - INFO - Training: 
03/07/2020 05:53:57 PM - INFO - DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='entropy',
                       max_depth=None, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=250,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=None, splitter='random')
03/07/2020 05:54:03 PM - INFO - Train time: 6.420s
03/07/2020 05:54:03 PM - INFO - Test time:  0.012s
03/07/2020 05:54:03 PM - INFO - Accuracy score:   0.307
03/07/2020 05:54:03 PM - INFO - 

===> Classification Report:

03/07/2020 05:54:03 PM - INFO -               precision    recall  f1-score   support

           1       0.39      0.69      0.50      5022
           2       0.11      0.03      0.05      2302
           3       0.15      0.05      0.07      2541
           4       0.17      0.13      0.15      2635
           7       0.16      0.14      0.15      2307
           8       0.18      0.14      0.16      2850
           9       0.16      0.04      0.06      2344
          10       0.37      0.57      0.45      4999

    accuracy                           0.31     25000
   macro avg       0.21      0.22      0.20     25000
weighted avg       0.25      0.31      0.25     25000

03/07/2020 05:54:03 PM - INFO - 

Cross validation:
03/07/2020 05:54:11 PM - INFO - 	accuracy: 5-fold cross validation: [0.3152 0.3036 0.3054 0.3154 0.3048]
03/07/2020 05:54:11 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 30.89 (+/- 1.05)
03/07/2020 05:54:11 PM - INFO - It took 316.26338386535645 seconds
03/07/2020 05:54:11 PM - INFO - ********************************************************************************
03/07/2020 05:54:11 PM - INFO - ################################################################################
03/07/2020 05:54:11 PM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 05:54:11 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 05:54:11 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 05:54:11 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  35.78%  |  [0.352  0.3546 0.3442 0.3492 0.3464]  |  34.93 (+/- 0.75)  |  12.08  |  0.732  |
03/07/2020 05:54:11 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  25.04%  |  [0.2516 0.2634 0.2528 0.249  0.2542]  |  25.42 (+/- 0.98)  |  38.77  |  0.03135  |
03/07/2020 05:54:11 PM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 05:54:11 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 05:54:11 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 05:54:11 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  38.02%  |  [0.3792 0.379  0.374  0.3704 0.3746]  |  37.54 (+/- 0.66)  |  120.6  |  7.117  |
03/07/2020 05:54:11 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  30.69%  |  [0.3152 0.3036 0.3054 0.3154 0.3048]  |  30.89 (+/- 1.05)  |  6.42  |  0.01237  |
03/07/2020 05:54:11 PM - INFO - 

03/07/2020 05:54:11 PM - INFO - ################################################################################
03/07/2020 05:54:11 PM - INFO - 3)
03/07/2020 05:54:11 PM - INFO - ********************************************************************************
03/07/2020 05:54:11 PM - INFO - Classifier: LINEAR_SVC, Dataset: IMDB_REVIEWS
03/07/2020 05:54:11 PM - INFO - ********************************************************************************
03/07/2020 05:54:18 PM - INFO - 

Performing grid search...

03/07/2020 05:54:18 PM - INFO - Parameters:
03/07/2020 05:54:18 PM - INFO - {'classifier__C': [0.01, 1.0], 'classifier__multi_class': ['ovr', 'crammer_singer'], 'classifier__tol': [0.0001, 0.001]}
03/07/2020 05:54:50 PM - INFO - 	Done in 32.087s
03/07/2020 05:54:50 PM - INFO - 	Best score: 0.409
03/07/2020 05:54:50 PM - INFO - 	Best parameters set:
03/07/2020 05:54:50 PM - INFO - 		classifier__C: 0.01
03/07/2020 05:54:50 PM - INFO - 		classifier__multi_class: 'crammer_singer'
03/07/2020 05:54:50 PM - INFO - 		classifier__tol: 0.001
03/07/2020 05:54:50 PM - INFO - 

USING LINEAR_SVC WITH DEFAULT PARAMETERS
03/07/2020 05:54:50 PM - INFO - ________________________________________________________________________________
03/07/2020 05:54:50 PM - INFO - Training: 
03/07/2020 05:54:50 PM - INFO - LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
          intercept_scaling=1, loss='squared_hinge', max_iter=1000,
          multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
          verbose=0)
03/07/2020 05:54:52 PM - INFO - Train time: 1.893s
03/07/2020 05:54:52 PM - INFO - Test time:  0.019s
03/07/2020 05:54:52 PM - INFO - Accuracy score:   0.374
03/07/2020 05:54:52 PM - INFO - 

===> Classification Report:

03/07/2020 05:54:52 PM - INFO -               precision    recall  f1-score   support

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

03/07/2020 05:54:52 PM - INFO - 

Cross validation:
03/07/2020 05:54:55 PM - INFO - 	accuracy: 5-fold cross validation: [0.3934 0.3974 0.3888 0.3802 0.3858]
03/07/2020 05:54:55 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 38.91 (+/- 1.19)
03/07/2020 05:54:55 PM - INFO - 

USING LINEAR_SVC WITH BEST PARAMETERS: {'C': 0.01, 'multi_class': 'crammer_singer', 'tol': 0.001}
03/07/2020 05:54:55 PM - INFO - ________________________________________________________________________________
03/07/2020 05:54:55 PM - INFO - Training: 
03/07/2020 05:54:55 PM - INFO - LinearSVC(C=0.01, class_weight=None, dual=True, fit_intercept=True,
          intercept_scaling=1, loss='squared_hinge', max_iter=1000,
          multi_class='crammer_singer', penalty='l2', random_state=None,
          tol=0.001, verbose=0)
03/07/2020 05:54:56 PM - INFO - Train time: 0.605s
03/07/2020 05:54:56 PM - INFO - Test time:  0.018s
03/07/2020 05:54:56 PM - INFO - Accuracy score:   0.407
03/07/2020 05:54:56 PM - INFO - 

===> Classification Report:

03/07/2020 05:54:56 PM - INFO -               precision    recall  f1-score   support

           1       0.47      0.88      0.62      5022
           2       0.14      0.04      0.06      2302
           3       0.22      0.09      0.13      2541
           4       0.31      0.24      0.27      2635
           7       0.27      0.19      0.22      2307
           8       0.27      0.15      0.19      2850
           9       0.19      0.06      0.09      2344
          10       0.48      0.76      0.59      4999

    accuracy                           0.41     25000
   macro avg       0.29      0.30      0.27     25000
weighted avg       0.33      0.41      0.34     25000

03/07/2020 05:54:56 PM - INFO - 

Cross validation:
03/07/2020 05:54:57 PM - INFO - 	accuracy: 5-fold cross validation: [0.4114 0.4206 0.4068 0.3992 0.4088]
03/07/2020 05:54:57 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 40.94 (+/- 1.39)
03/07/2020 05:54:57 PM - INFO - It took 45.83590340614319 seconds
03/07/2020 05:54:57 PM - INFO - ********************************************************************************
03/07/2020 05:54:57 PM - INFO - ################################################################################
03/07/2020 05:54:57 PM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 05:54:57 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 05:54:57 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 05:54:57 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  35.78%  |  [0.352  0.3546 0.3442 0.3492 0.3464]  |  34.93 (+/- 0.75)  |  12.08  |  0.732  |
03/07/2020 05:54:57 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  25.04%  |  [0.2516 0.2634 0.2528 0.249  0.2542]  |  25.42 (+/- 0.98)  |  38.77  |  0.03135  |
03/07/2020 05:54:57 PM - INFO - |  7  |  LINEAR_SVC  |  37.36%  |  [0.3934 0.3974 0.3888 0.3802 0.3858]  |  38.91 (+/- 1.19)  |  1.893  |  0.01857  |
03/07/2020 05:54:57 PM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 05:54:57 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 05:54:57 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 05:54:57 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  38.02%  |  [0.3792 0.379  0.374  0.3704 0.3746]  |  37.54 (+/- 0.66)  |  120.6  |  7.117  |
03/07/2020 05:54:57 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  30.69%  |  [0.3152 0.3036 0.3054 0.3154 0.3048]  |  30.89 (+/- 1.05)  |  6.42  |  0.01237  |
03/07/2020 05:54:57 PM - INFO - |  7  |  LINEAR_SVC  |  40.75%  |  [0.4114 0.4206 0.4068 0.3992 0.4088]  |  40.94 (+/- 1.39)  |  0.6048  |  0.01785  |
03/07/2020 05:54:57 PM - INFO - 

03/07/2020 05:54:57 PM - INFO - ################################################################################
03/07/2020 05:54:57 PM - INFO - 4)
03/07/2020 05:54:57 PM - INFO - ********************************************************************************
03/07/2020 05:54:57 PM - INFO - Classifier: LOGISTIC_REGRESSION, Dataset: IMDB_REVIEWS
03/07/2020 05:54:57 PM - INFO - ********************************************************************************
03/07/2020 05:55:04 PM - INFO - 

Performing grid search...

03/07/2020 05:55:04 PM - INFO - Parameters:
03/07/2020 05:55:04 PM - INFO - {'classifier__C': [1, 10], 'classifier__tol': [0.001, 0.01]}
03/07/2020 05:56:59 PM - INFO - 	Done in 115.009s
03/07/2020 05:56:59 PM - INFO - 	Best score: 0.424
03/07/2020 05:56:59 PM - INFO - 	Best parameters set:
03/07/2020 05:56:59 PM - INFO - 		classifier__C: 1
03/07/2020 05:56:59 PM - INFO - 		classifier__tol: 0.001
03/07/2020 05:56:59 PM - INFO - 

USING LOGISTIC_REGRESSION WITH DEFAULT PARAMETERS
03/07/2020 05:56:59 PM - INFO - ________________________________________________________________________________
03/07/2020 05:56:59 PM - INFO - Training: 
03/07/2020 05:56:59 PM - INFO - LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)
03/07/2020 05:57:16 PM - INFO - Train time: 17.708s
03/07/2020 05:57:16 PM - INFO - Test time:  0.039s
03/07/2020 05:57:16 PM - INFO - Accuracy score:   0.420
03/07/2020 05:57:16 PM - INFO - 

===> Classification Report:

03/07/2020 05:57:16 PM - INFO -               precision    recall  f1-score   support

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

03/07/2020 05:57:16 PM - INFO - 

Cross validation:
03/07/2020 05:57:41 PM - INFO - 	accuracy: 5-fold cross validation: [0.4282 0.4334 0.4152 0.4194 0.4218]
03/07/2020 05:57:41 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 42.36 (+/- 1.29)
03/07/2020 05:57:41 PM - INFO - 

USING LOGISTIC_REGRESSION WITH BEST PARAMETERS: {'C': 1, 'tol': 0.001}
03/07/2020 05:57:41 PM - INFO - ________________________________________________________________________________
03/07/2020 05:57:41 PM - INFO - Training: 
03/07/2020 05:57:41 PM - INFO - LogisticRegression(C=1, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.001, verbose=0,
                   warm_start=False)
03/07/2020 05:57:58 PM - INFO - Train time: 17.836s
03/07/2020 05:57:58 PM - INFO - Test time:  0.040s
03/07/2020 05:57:58 PM - INFO - Accuracy score:   0.420
03/07/2020 05:57:58 PM - INFO - 

===> Classification Report:

03/07/2020 05:57:58 PM - INFO -               precision    recall  f1-score   support

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

03/07/2020 05:57:58 PM - INFO - 

Cross validation:
03/07/2020 05:58:22 PM - INFO - 	accuracy: 5-fold cross validation: [0.4282 0.4334 0.4152 0.4194 0.4218]
03/07/2020 05:58:22 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 42.36 (+/- 1.29)
03/07/2020 05:58:22 PM - INFO - It took 205.50172805786133 seconds
03/07/2020 05:58:22 PM - INFO - ********************************************************************************
03/07/2020 05:58:22 PM - INFO - ################################################################################
03/07/2020 05:58:22 PM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 05:58:22 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 05:58:22 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 05:58:22 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  35.78%  |  [0.352  0.3546 0.3442 0.3492 0.3464]  |  34.93 (+/- 0.75)  |  12.08  |  0.732  |
03/07/2020 05:58:22 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  25.04%  |  [0.2516 0.2634 0.2528 0.249  0.2542]  |  25.42 (+/- 0.98)  |  38.77  |  0.03135  |
03/07/2020 05:58:22 PM - INFO - |  7  |  LINEAR_SVC  |  37.36%  |  [0.3934 0.3974 0.3888 0.3802 0.3858]  |  38.91 (+/- 1.19)  |  1.893  |  0.01857  |
03/07/2020 05:58:22 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  42.04%  |  [0.4282 0.4334 0.4152 0.4194 0.4218]  |  42.36 (+/- 1.29)  |  17.71  |  0.03946  |
03/07/2020 05:58:22 PM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 05:58:22 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 05:58:22 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 05:58:22 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  38.02%  |  [0.3792 0.379  0.374  0.3704 0.3746]  |  37.54 (+/- 0.66)  |  120.6  |  7.117  |
03/07/2020 05:58:22 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  30.69%  |  [0.3152 0.3036 0.3054 0.3154 0.3048]  |  30.89 (+/- 1.05)  |  6.42  |  0.01237  |
03/07/2020 05:58:22 PM - INFO - |  7  |  LINEAR_SVC  |  40.75%  |  [0.4114 0.4206 0.4068 0.3992 0.4088]  |  40.94 (+/- 1.39)  |  0.6048  |  0.01785  |
03/07/2020 05:58:22 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  42.04%  |  [0.4282 0.4334 0.4152 0.4194 0.4218]  |  42.36 (+/- 1.29)  |  17.84  |  0.03951  |
03/07/2020 05:58:22 PM - INFO - 

03/07/2020 05:58:22 PM - INFO - ################################################################################
03/07/2020 05:58:22 PM - INFO - 5)
03/07/2020 05:58:22 PM - INFO - ********************************************************************************
03/07/2020 05:58:22 PM - INFO - Classifier: RANDOM_FOREST_CLASSIFIER, Dataset: IMDB_REVIEWS
03/07/2020 05:58:22 PM - INFO - ********************************************************************************
03/07/2020 05:58:29 PM - INFO - 

Performing grid search...

03/07/2020 05:58:29 PM - INFO - Parameters:
03/07/2020 05:58:29 PM - INFO - {'classifier__min_samples_leaf': [1, 2], 'classifier__min_samples_split': [2, 5], 'classifier__n_estimators': [100, 200]}
03/07/2020 06:09:44 PM - INFO - 	Done in 675.103s
03/07/2020 06:09:44 PM - INFO - 	Best score: 0.373
03/07/2020 06:09:44 PM - INFO - 	Best parameters set:
03/07/2020 06:09:44 PM - INFO - 		classifier__min_samples_leaf: 2
03/07/2020 06:09:44 PM - INFO - 		classifier__min_samples_split: 5
03/07/2020 06:09:44 PM - INFO - 		classifier__n_estimators: 200
03/07/2020 06:09:44 PM - INFO - 

USING RANDOM_FOREST_CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 06:09:44 PM - INFO - ________________________________________________________________________________
03/07/2020 06:09:44 PM - INFO - Training: 
03/07/2020 06:09:44 PM - INFO - RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
03/07/2020 06:10:57 PM - INFO - Train time: 72.588s
03/07/2020 06:10:58 PM - INFO - Test time:  1.511s
03/07/2020 06:10:58 PM - INFO - Accuracy score:   0.372
03/07/2020 06:10:58 PM - INFO - 

===> Classification Report:

03/07/2020 06:10:58 PM - INFO -               precision    recall  f1-score   support

           1       0.38      0.90      0.53      5022
           2       0.53      0.01      0.02      2302
           3       0.36      0.02      0.04      2541
           4       0.29      0.06      0.10      2635
           7       0.28      0.05      0.09      2307
           8       0.25      0.09      0.13      2850
           9       0.22      0.00      0.01      2344
          10       0.38      0.83      0.52      4999

    accuracy                           0.37     25000
   macro avg       0.34      0.25      0.18     25000
weighted avg       0.34      0.37      0.25     25000

03/07/2020 06:10:58 PM - INFO - 

Cross validation:
03/07/2020 06:12:43 PM - INFO - 	accuracy: 5-fold cross validation: [0.3676 0.3678 0.369  0.362  0.3684]
03/07/2020 06:12:43 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 36.70 (+/- 0.51)
03/07/2020 06:12:43 PM - INFO - 

USING RANDOM_FOREST_CLASSIFIER WITH BEST PARAMETERS: {'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 200}
03/07/2020 06:12:43 PM - INFO - ________________________________________________________________________________
03/07/2020 06:12:43 PM - INFO - Training: 
03/07/2020 06:12:43 PM - INFO - RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=2, min_samples_split=5,
                       min_weight_fraction_leaf=0.0, n_estimators=200,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
03/07/2020 06:13:26 PM - INFO - Train time: 42.889s
03/07/2020 06:13:29 PM - INFO - Test time:  2.501s
03/07/2020 06:13:29 PM - INFO - Accuracy score:   0.378
03/07/2020 06:13:29 PM - INFO - 

===> Classification Report:

03/07/2020 06:13:29 PM - INFO -               precision    recall  f1-score   support

           1       0.38      0.92      0.54      5022
           2       0.93      0.01      0.01      2302
           3       0.52      0.01      0.02      2541
           4       0.35      0.05      0.09      2635
           7       0.34      0.04      0.07      2307
           8       0.25      0.10      0.14      2850
           9       1.00      0.00      0.01      2344
          10       0.39      0.86      0.53      4999

    accuracy                           0.38     25000
   macro avg       0.52      0.25      0.17     25000
weighted avg       0.48      0.38      0.25     25000

03/07/2020 06:13:29 PM - INFO - 

Cross validation:
03/07/2020 06:14:47 PM - INFO - 	accuracy: 5-fold cross validation: [0.3716 0.3772 0.3768 0.3742 0.368 ]
03/07/2020 06:14:47 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 37.36 (+/- 0.69)
03/07/2020 06:14:47 PM - INFO - It took 984.8159382343292 seconds
03/07/2020 06:14:47 PM - INFO - ********************************************************************************
03/07/2020 06:14:47 PM - INFO - ################################################################################
03/07/2020 06:14:47 PM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 06:14:47 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 06:14:47 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 06:14:47 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  35.78%  |  [0.352  0.3546 0.3442 0.3492 0.3464]  |  34.93 (+/- 0.75)  |  12.08  |  0.732  |
03/07/2020 06:14:47 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  25.04%  |  [0.2516 0.2634 0.2528 0.249  0.2542]  |  25.42 (+/- 0.98)  |  38.77  |  0.03135  |
03/07/2020 06:14:47 PM - INFO - |  7  |  LINEAR_SVC  |  37.36%  |  [0.3934 0.3974 0.3888 0.3802 0.3858]  |  38.91 (+/- 1.19)  |  1.893  |  0.01857  |
03/07/2020 06:14:47 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  42.04%  |  [0.4282 0.4334 0.4152 0.4194 0.4218]  |  42.36 (+/- 1.29)  |  17.71  |  0.03946  |
03/07/2020 06:14:47 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  37.19%  |  [0.3676 0.3678 0.369  0.362  0.3684]  |  36.70 (+/- 0.51)  |  72.59  |  1.511  |
03/07/2020 06:14:47 PM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 06:14:47 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 06:14:47 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 06:14:47 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  38.02%  |  [0.3792 0.379  0.374  0.3704 0.3746]  |  37.54 (+/- 0.66)  |  120.6  |  7.117  |
03/07/2020 06:14:47 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  30.69%  |  [0.3152 0.3036 0.3054 0.3154 0.3048]  |  30.89 (+/- 1.05)  |  6.42  |  0.01237  |
03/07/2020 06:14:47 PM - INFO - |  7  |  LINEAR_SVC  |  40.75%  |  [0.4114 0.4206 0.4068 0.3992 0.4088]  |  40.94 (+/- 1.39)  |  0.6048  |  0.01785  |
03/07/2020 06:14:47 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  42.04%  |  [0.4282 0.4334 0.4152 0.4194 0.4218]  |  42.36 (+/- 1.29)  |  17.84  |  0.03951  |
03/07/2020 06:14:47 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  37.82%  |  [0.3716 0.3772 0.3768 0.3742 0.368 ]  |  37.36 (+/- 0.69)  |  42.89  |  2.501  |
03/07/2020 06:14:47 PM - INFO - 

03/07/2020 06:14:47 PM - INFO - ################################################################################
03/07/2020 06:14:47 PM - INFO - 6)
03/07/2020 06:14:47 PM - INFO - ********************************************************************************
03/07/2020 06:14:47 PM - INFO - Classifier: BERNOULLI_NB, Dataset: IMDB_REVIEWS
03/07/2020 06:14:47 PM - INFO - ********************************************************************************
03/07/2020 06:14:54 PM - INFO - 

Performing grid search...

03/07/2020 06:14:54 PM - INFO - Parameters:
03/07/2020 06:14:54 PM - INFO - {'classifier__alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0], 'classifier__binarize': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0], 'classifier__fit_prior': [False, True]}
03/07/2020 06:15:20 PM - INFO - 	Done in 25.779s
03/07/2020 06:15:20 PM - INFO - 	Best score: 0.379
03/07/2020 06:15:20 PM - INFO - 	Best parameters set:
03/07/2020 06:15:20 PM - INFO - 		classifier__alpha: 0.5
03/07/2020 06:15:20 PM - INFO - 		classifier__binarize: 0.0001
03/07/2020 06:15:20 PM - INFO - 		classifier__fit_prior: True
03/07/2020 06:15:20 PM - INFO - 

USING BERNOULLI_NB WITH DEFAULT PARAMETERS
03/07/2020 06:15:20 PM - INFO - ________________________________________________________________________________
03/07/2020 06:15:20 PM - INFO - Training: 
03/07/2020 06:15:20 PM - INFO - BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
03/07/2020 06:15:20 PM - INFO - Train time: 0.040s
03/07/2020 06:15:20 PM - INFO - Test time:  0.040s
03/07/2020 06:15:20 PM - INFO - Accuracy score:   0.372
03/07/2020 06:15:20 PM - INFO - 

===> Classification Report:

03/07/2020 06:15:20 PM - INFO -               precision    recall  f1-score   support

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

03/07/2020 06:15:20 PM - INFO - 

Cross validation:
03/07/2020 06:15:20 PM - INFO - 	accuracy: 5-fold cross validation: [0.3786 0.3812 0.3678 0.374  0.373 ]
03/07/2020 06:15:20 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 37.49 (+/- 0.93)
03/07/2020 06:15:20 PM - INFO - 

USING BERNOULLI_NB WITH BEST PARAMETERS: {'alpha': 0.5, 'binarize': 0.0001, 'fit_prior': True}
03/07/2020 06:15:20 PM - INFO - ________________________________________________________________________________
03/07/2020 06:15:20 PM - INFO - Training: 
03/07/2020 06:15:20 PM - INFO - BernoulliNB(alpha=0.5, binarize=0.0001, class_prior=None, fit_prior=True)
03/07/2020 06:15:20 PM - INFO - Train time: 0.043s
03/07/2020 06:15:20 PM - INFO - Test time:  0.041s
03/07/2020 06:15:20 PM - INFO - Accuracy score:   0.370
03/07/2020 06:15:20 PM - INFO - 

===> Classification Report:

03/07/2020 06:15:20 PM - INFO -               precision    recall  f1-score   support

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

03/07/2020 06:15:20 PM - INFO - 

Cross validation:
03/07/2020 06:15:21 PM - INFO - 	accuracy: 5-fold cross validation: [0.377  0.389  0.3782 0.38   0.373 ]
03/07/2020 06:15:21 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 37.94 (+/- 1.06)
03/07/2020 06:15:21 PM - INFO - It took 33.317975759506226 seconds
03/07/2020 06:15:21 PM - INFO - ********************************************************************************
03/07/2020 06:15:21 PM - INFO - ################################################################################
03/07/2020 06:15:21 PM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 06:15:21 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 06:15:21 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 06:15:21 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  35.78%  |  [0.352  0.3546 0.3442 0.3492 0.3464]  |  34.93 (+/- 0.75)  |  12.08  |  0.732  |
03/07/2020 06:15:21 PM - INFO - |  2  |  BERNOULLI_NB  |  37.25%  |  [0.3786 0.3812 0.3678 0.374  0.373 ]  |  37.49 (+/- 0.93)  |  0.03953  |  0.04032  |
03/07/2020 06:15:21 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  25.04%  |  [0.2516 0.2634 0.2528 0.249  0.2542]  |  25.42 (+/- 0.98)  |  38.77  |  0.03135  |
03/07/2020 06:15:21 PM - INFO - |  7  |  LINEAR_SVC  |  37.36%  |  [0.3934 0.3974 0.3888 0.3802 0.3858]  |  38.91 (+/- 1.19)  |  1.893  |  0.01857  |
03/07/2020 06:15:21 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  42.04%  |  [0.4282 0.4334 0.4152 0.4194 0.4218]  |  42.36 (+/- 1.29)  |  17.71  |  0.03946  |
03/07/2020 06:15:21 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  37.19%  |  [0.3676 0.3678 0.369  0.362  0.3684]  |  36.70 (+/- 0.51)  |  72.59  |  1.511  |
03/07/2020 06:15:21 PM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 06:15:21 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 06:15:21 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 06:15:21 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  38.02%  |  [0.3792 0.379  0.374  0.3704 0.3746]  |  37.54 (+/- 0.66)  |  120.6  |  7.117  |
03/07/2020 06:15:21 PM - INFO - |  2  |  BERNOULLI_NB  |  37.03%  |  [0.377  0.389  0.3782 0.38   0.373 ]  |  37.94 (+/- 1.06)  |  0.0429  |  0.04091  |
03/07/2020 06:15:21 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  30.69%  |  [0.3152 0.3036 0.3054 0.3154 0.3048]  |  30.89 (+/- 1.05)  |  6.42  |  0.01237  |
03/07/2020 06:15:21 PM - INFO - |  7  |  LINEAR_SVC  |  40.75%  |  [0.4114 0.4206 0.4068 0.3992 0.4088]  |  40.94 (+/- 1.39)  |  0.6048  |  0.01785  |
03/07/2020 06:15:21 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  42.04%  |  [0.4282 0.4334 0.4152 0.4194 0.4218]  |  42.36 (+/- 1.29)  |  17.84  |  0.03951  |
03/07/2020 06:15:21 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  37.82%  |  [0.3716 0.3772 0.3768 0.3742 0.368 ]  |  37.36 (+/- 0.69)  |  42.89  |  2.501  |
03/07/2020 06:15:21 PM - INFO - 

03/07/2020 06:15:21 PM - INFO - ################################################################################
03/07/2020 06:15:21 PM - INFO - 7)
03/07/2020 06:15:21 PM - INFO - ********************************************************************************
03/07/2020 06:15:21 PM - INFO - Classifier: COMPLEMENT_NB, Dataset: IMDB_REVIEWS
03/07/2020 06:15:21 PM - INFO - ********************************************************************************
03/07/2020 06:15:27 PM - INFO - 

Performing grid search...

03/07/2020 06:15:27 PM - INFO - Parameters:
03/07/2020 06:15:27 PM - INFO - {'classifier__alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0], 'classifier__fit_prior': [False, True], 'classifier__norm': [False, True]}
03/07/2020 06:15:32 PM - INFO - 	Done in 4.746s
03/07/2020 06:15:32 PM - INFO - 	Best score: 0.391
03/07/2020 06:15:32 PM - INFO - 	Best parameters set:
03/07/2020 06:15:32 PM - INFO - 		classifier__alpha: 0.5
03/07/2020 06:15:32 PM - INFO - 		classifier__fit_prior: False
03/07/2020 06:15:32 PM - INFO - 		classifier__norm: False
03/07/2020 06:15:32 PM - INFO - 

USING COMPLEMENT_NB WITH DEFAULT PARAMETERS
03/07/2020 06:15:32 PM - INFO - ________________________________________________________________________________
03/07/2020 06:15:32 PM - INFO - Training: 
03/07/2020 06:15:32 PM - INFO - ComplementNB(alpha=1.0, class_prior=None, fit_prior=True, norm=False)
03/07/2020 06:15:32 PM - INFO - Train time: 0.035s
03/07/2020 06:15:32 PM - INFO - Test time:  0.018s
03/07/2020 06:15:32 PM - INFO - Accuracy score:   0.373
03/07/2020 06:15:32 PM - INFO - 

===> Classification Report:

03/07/2020 06:15:32 PM - INFO -               precision    recall  f1-score   support

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

03/07/2020 06:15:32 PM - INFO - 

Cross validation:
03/07/2020 06:15:32 PM - INFO - 	accuracy: 5-fold cross validation: [0.3832 0.3858 0.3834 0.386  0.3776]
03/07/2020 06:15:32 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 38.32 (+/- 0.61)
03/07/2020 06:15:32 PM - INFO - 

USING COMPLEMENT_NB WITH BEST PARAMETERS: {'alpha': 0.5, 'fit_prior': False, 'norm': False}
03/07/2020 06:15:32 PM - INFO - ________________________________________________________________________________
03/07/2020 06:15:32 PM - INFO - Training: 
03/07/2020 06:15:32 PM - INFO - ComplementNB(alpha=0.5, class_prior=None, fit_prior=False, norm=False)
03/07/2020 06:15:32 PM - INFO - Train time: 0.038s
03/07/2020 06:15:32 PM - INFO - Test time:  0.019s
03/07/2020 06:15:32 PM - INFO - Accuracy score:   0.373
03/07/2020 06:15:32 PM - INFO - 

===> Classification Report:

03/07/2020 06:15:32 PM - INFO -               precision    recall  f1-score   support

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

03/07/2020 06:15:32 PM - INFO - 

Cross validation:
03/07/2020 06:15:33 PM - INFO - 	accuracy: 5-fold cross validation: [0.3878 0.3942 0.3976 0.3938 0.3832]
03/07/2020 06:15:33 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 39.13 (+/- 1.03)
03/07/2020 06:15:33 PM - INFO - It took 12.12444806098938 seconds
03/07/2020 06:15:33 PM - INFO - ********************************************************************************
03/07/2020 06:15:33 PM - INFO - ################################################################################
03/07/2020 06:15:33 PM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 06:15:33 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 06:15:33 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 06:15:33 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  35.78%  |  [0.352  0.3546 0.3442 0.3492 0.3464]  |  34.93 (+/- 0.75)  |  12.08  |  0.732  |
03/07/2020 06:15:33 PM - INFO - |  2  |  BERNOULLI_NB  |  37.25%  |  [0.3786 0.3812 0.3678 0.374  0.373 ]  |  37.49 (+/- 0.93)  |  0.03953  |  0.04032  |
03/07/2020 06:15:33 PM - INFO - |  3  |  COMPLEMENT_NB  |  37.30%  |  [0.3832 0.3858 0.3834 0.386  0.3776]  |  38.32 (+/- 0.61)  |  0.03533  |  0.0185  |
03/07/2020 06:15:33 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  25.04%  |  [0.2516 0.2634 0.2528 0.249  0.2542]  |  25.42 (+/- 0.98)  |  38.77  |  0.03135  |
03/07/2020 06:15:33 PM - INFO - |  7  |  LINEAR_SVC  |  37.36%  |  [0.3934 0.3974 0.3888 0.3802 0.3858]  |  38.91 (+/- 1.19)  |  1.893  |  0.01857  |
03/07/2020 06:15:33 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  42.04%  |  [0.4282 0.4334 0.4152 0.4194 0.4218]  |  42.36 (+/- 1.29)  |  17.71  |  0.03946  |
03/07/2020 06:15:33 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  37.19%  |  [0.3676 0.3678 0.369  0.362  0.3684]  |  36.70 (+/- 0.51)  |  72.59  |  1.511  |
03/07/2020 06:15:33 PM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 06:15:33 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 06:15:33 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 06:15:33 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  38.02%  |  [0.3792 0.379  0.374  0.3704 0.3746]  |  37.54 (+/- 0.66)  |  120.6  |  7.117  |
03/07/2020 06:15:33 PM - INFO - |  2  |  BERNOULLI_NB  |  37.03%  |  [0.377  0.389  0.3782 0.38   0.373 ]  |  37.94 (+/- 1.06)  |  0.0429  |  0.04091  |
03/07/2020 06:15:33 PM - INFO - |  3  |  COMPLEMENT_NB  |  37.34%  |  [0.3878 0.3942 0.3976 0.3938 0.3832]  |  39.13 (+/- 1.03)  |  0.03774  |  0.01899  |
03/07/2020 06:15:33 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  30.69%  |  [0.3152 0.3036 0.3054 0.3154 0.3048]  |  30.89 (+/- 1.05)  |  6.42  |  0.01237  |
03/07/2020 06:15:33 PM - INFO - |  7  |  LINEAR_SVC  |  40.75%  |  [0.4114 0.4206 0.4068 0.3992 0.4088]  |  40.94 (+/- 1.39)  |  0.6048  |  0.01785  |
03/07/2020 06:15:33 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  42.04%  |  [0.4282 0.4334 0.4152 0.4194 0.4218]  |  42.36 (+/- 1.29)  |  17.84  |  0.03951  |
03/07/2020 06:15:33 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  37.82%  |  [0.3716 0.3772 0.3768 0.3742 0.368 ]  |  37.36 (+/- 0.69)  |  42.89  |  2.501  |
03/07/2020 06:15:33 PM - INFO - 

03/07/2020 06:15:33 PM - INFO - ################################################################################
03/07/2020 06:15:33 PM - INFO - 8)
03/07/2020 06:15:33 PM - INFO - ********************************************************************************
03/07/2020 06:15:33 PM - INFO - Classifier: MULTINOMIAL_NB, Dataset: IMDB_REVIEWS
03/07/2020 06:15:33 PM - INFO - ********************************************************************************
03/07/2020 06:15:39 PM - INFO - 

Performing grid search...

03/07/2020 06:15:39 PM - INFO - Parameters:
03/07/2020 06:15:39 PM - INFO - {'classifier__alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0], 'classifier__fit_prior': [False, True]}
03/07/2020 06:15:42 PM - INFO - 	Done in 2.544s
03/07/2020 06:15:42 PM - INFO - 	Best score: 0.391
03/07/2020 06:15:42 PM - INFO - 	Best parameters set:
03/07/2020 06:15:42 PM - INFO - 		classifier__alpha: 0.1
03/07/2020 06:15:42 PM - INFO - 		classifier__fit_prior: True
03/07/2020 06:15:42 PM - INFO - 

USING MULTINOMIAL_NB WITH DEFAULT PARAMETERS
03/07/2020 06:15:42 PM - INFO - ________________________________________________________________________________
03/07/2020 06:15:42 PM - INFO - Training: 
03/07/2020 06:15:42 PM - INFO - MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
03/07/2020 06:15:42 PM - INFO - Train time: 0.037s
03/07/2020 06:15:42 PM - INFO - Test time:  0.023s
03/07/2020 06:15:42 PM - INFO - Accuracy score:   0.350
03/07/2020 06:15:42 PM - INFO - 

===> Classification Report:

03/07/2020 06:15:42 PM - INFO -               precision    recall  f1-score   support

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

03/07/2020 06:15:42 PM - INFO - 

Cross validation:
03/07/2020 06:15:42 PM - INFO - 	accuracy: 5-fold cross validation: [0.353  0.3502 0.3528 0.3514 0.3488]
03/07/2020 06:15:42 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 35.12 (+/- 0.32)
03/07/2020 06:15:42 PM - INFO - 

USING MULTINOMIAL_NB WITH BEST PARAMETERS: {'alpha': 0.1, 'fit_prior': True}
03/07/2020 06:15:42 PM - INFO - ________________________________________________________________________________
03/07/2020 06:15:42 PM - INFO - Training: 
03/07/2020 06:15:42 PM - INFO - MultinomialNB(alpha=0.1, class_prior=None, fit_prior=True)
03/07/2020 06:15:42 PM - INFO - Train time: 0.036s
03/07/2020 06:15:42 PM - INFO - Test time:  0.021s
03/07/2020 06:15:42 PM - INFO - Accuracy score:   0.378
03/07/2020 06:15:42 PM - INFO - 

===> Classification Report:

03/07/2020 06:15:42 PM - INFO -               precision    recall  f1-score   support

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

03/07/2020 06:15:42 PM - INFO - 

Cross validation:
03/07/2020 06:15:43 PM - INFO - 	accuracy: 5-fold cross validation: [0.389  0.3928 0.3918 0.3942 0.386 ]
03/07/2020 06:15:43 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 39.08 (+/- 0.59)
03/07/2020 06:15:43 PM - INFO - It took 10.053622961044312 seconds
03/07/2020 06:15:43 PM - INFO - ********************************************************************************
03/07/2020 06:15:43 PM - INFO - ################################################################################
03/07/2020 06:15:43 PM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 06:15:43 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 06:15:43 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 06:15:43 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  35.78%  |  [0.352  0.3546 0.3442 0.3492 0.3464]  |  34.93 (+/- 0.75)  |  12.08  |  0.732  |
03/07/2020 06:15:43 PM - INFO - |  2  |  BERNOULLI_NB  |  37.25%  |  [0.3786 0.3812 0.3678 0.374  0.373 ]  |  37.49 (+/- 0.93)  |  0.03953  |  0.04032  |
03/07/2020 06:15:43 PM - INFO - |  3  |  COMPLEMENT_NB  |  37.30%  |  [0.3832 0.3858 0.3834 0.386  0.3776]  |  38.32 (+/- 0.61)  |  0.03533  |  0.0185  |
03/07/2020 06:15:43 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  25.04%  |  [0.2516 0.2634 0.2528 0.249  0.2542]  |  25.42 (+/- 0.98)  |  38.77  |  0.03135  |
03/07/2020 06:15:43 PM - INFO - |  7  |  LINEAR_SVC  |  37.36%  |  [0.3934 0.3974 0.3888 0.3802 0.3858]  |  38.91 (+/- 1.19)  |  1.893  |  0.01857  |
03/07/2020 06:15:43 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  42.04%  |  [0.4282 0.4334 0.4152 0.4194 0.4218]  |  42.36 (+/- 1.29)  |  17.71  |  0.03946  |
03/07/2020 06:15:43 PM - INFO - |  9  |  MULTINOMIAL_NB  |  35.05%  |  [0.353  0.3502 0.3528 0.3514 0.3488]  |  35.12 (+/- 0.32)  |  0.03723  |  0.02325  |
03/07/2020 06:15:43 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  37.19%  |  [0.3676 0.3678 0.369  0.362  0.3684]  |  36.70 (+/- 0.51)  |  72.59  |  1.511  |
03/07/2020 06:15:43 PM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 06:15:43 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 06:15:43 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 06:15:43 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  38.02%  |  [0.3792 0.379  0.374  0.3704 0.3746]  |  37.54 (+/- 0.66)  |  120.6  |  7.117  |
03/07/2020 06:15:43 PM - INFO - |  2  |  BERNOULLI_NB  |  37.03%  |  [0.377  0.389  0.3782 0.38   0.373 ]  |  37.94 (+/- 1.06)  |  0.0429  |  0.04091  |
03/07/2020 06:15:43 PM - INFO - |  3  |  COMPLEMENT_NB  |  37.34%  |  [0.3878 0.3942 0.3976 0.3938 0.3832]  |  39.13 (+/- 1.03)  |  0.03774  |  0.01899  |
03/07/2020 06:15:43 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  30.69%  |  [0.3152 0.3036 0.3054 0.3154 0.3048]  |  30.89 (+/- 1.05)  |  6.42  |  0.01237  |
03/07/2020 06:15:43 PM - INFO - |  7  |  LINEAR_SVC  |  40.75%  |  [0.4114 0.4206 0.4068 0.3992 0.4088]  |  40.94 (+/- 1.39)  |  0.6048  |  0.01785  |
03/07/2020 06:15:43 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  42.04%  |  [0.4282 0.4334 0.4152 0.4194 0.4218]  |  42.36 (+/- 1.29)  |  17.84  |  0.03951  |
03/07/2020 06:15:43 PM - INFO - |  9  |  MULTINOMIAL_NB  |  37.82%  |  [0.389  0.3928 0.3918 0.3942 0.386 ]  |  39.08 (+/- 0.59)  |  0.03643  |  0.02055  |
03/07/2020 06:15:43 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  37.82%  |  [0.3716 0.3772 0.3768 0.3742 0.368 ]  |  37.36 (+/- 0.69)  |  42.89  |  2.501  |
03/07/2020 06:15:43 PM - INFO - 

03/07/2020 06:15:43 PM - INFO - ################################################################################
03/07/2020 06:15:43 PM - INFO - 9)
03/07/2020 06:15:43 PM - INFO - ********************************************************************************
03/07/2020 06:15:43 PM - INFO - Classifier: NEAREST_CENTROID, Dataset: IMDB_REVIEWS
03/07/2020 06:15:43 PM - INFO - ********************************************************************************
03/07/2020 06:15:49 PM - INFO - 

Performing grid search...

03/07/2020 06:15:49 PM - INFO - Parameters:
03/07/2020 06:15:49 PM - INFO - {'classifier__metric': ['euclidean', 'cosine']}
03/07/2020 06:15:50 PM - INFO - 	Done in 0.375s
03/07/2020 06:15:50 PM - INFO - 	Best score: 0.380
03/07/2020 06:15:50 PM - INFO - 	Best parameters set:
03/07/2020 06:15:50 PM - INFO - 		classifier__metric: 'cosine'
03/07/2020 06:15:50 PM - INFO - 

USING NEAREST_CENTROID WITH DEFAULT PARAMETERS
03/07/2020 06:15:50 PM - INFO - ________________________________________________________________________________
03/07/2020 06:15:50 PM - INFO - Training: 
03/07/2020 06:15:50 PM - INFO - NearestCentroid(metric='euclidean', shrink_threshold=None)
03/07/2020 06:15:50 PM - INFO - Train time: 0.023s
03/07/2020 06:15:50 PM - INFO - Test time:  0.026s
03/07/2020 06:15:50 PM - INFO - Accuracy score:   0.371
03/07/2020 06:15:50 PM - INFO - 

===> Classification Report:

03/07/2020 06:15:50 PM - INFO -               precision    recall  f1-score   support

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

03/07/2020 06:15:50 PM - INFO - 

Cross validation:
03/07/2020 06:15:50 PM - INFO - 	accuracy: 5-fold cross validation: [0.3884 0.373  0.3818 0.367  0.372 ]
03/07/2020 06:15:50 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 37.64 (+/- 1.53)
03/07/2020 06:15:50 PM - INFO - 

USING NEAREST_CENTROID WITH BEST PARAMETERS: {'metric': 'cosine'}
03/07/2020 06:15:50 PM - INFO - ________________________________________________________________________________
03/07/2020 06:15:50 PM - INFO - Training: 
03/07/2020 06:15:50 PM - INFO - NearestCentroid(metric='cosine', shrink_threshold=None)
03/07/2020 06:15:50 PM - INFO - Train time: 0.021s
03/07/2020 06:15:50 PM - INFO - Test time:  0.032s
03/07/2020 06:15:50 PM - INFO - Accuracy score:   0.373
03/07/2020 06:15:50 PM - INFO - 

===> Classification Report:

03/07/2020 06:15:50 PM - INFO -               precision    recall  f1-score   support

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

03/07/2020 06:15:50 PM - INFO - 

Cross validation:
03/07/2020 06:15:51 PM - INFO - 	accuracy: 5-fold cross validation: [0.3872 0.3786 0.3894 0.3672 0.3782]
03/07/2020 06:15:51 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 38.01 (+/- 1.57)
03/07/2020 06:15:51 PM - INFO - It took 7.827193021774292 seconds
03/07/2020 06:15:51 PM - INFO - ********************************************************************************
03/07/2020 06:15:51 PM - INFO - ################################################################################
03/07/2020 06:15:51 PM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 06:15:51 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 06:15:51 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 06:15:51 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  35.78%  |  [0.352  0.3546 0.3442 0.3492 0.3464]  |  34.93 (+/- 0.75)  |  12.08  |  0.732  |
03/07/2020 06:15:51 PM - INFO - |  2  |  BERNOULLI_NB  |  37.25%  |  [0.3786 0.3812 0.3678 0.374  0.373 ]  |  37.49 (+/- 0.93)  |  0.03953  |  0.04032  |
03/07/2020 06:15:51 PM - INFO - |  3  |  COMPLEMENT_NB  |  37.30%  |  [0.3832 0.3858 0.3834 0.386  0.3776]  |  38.32 (+/- 0.61)  |  0.03533  |  0.0185  |
03/07/2020 06:15:51 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  25.04%  |  [0.2516 0.2634 0.2528 0.249  0.2542]  |  25.42 (+/- 0.98)  |  38.77  |  0.03135  |
03/07/2020 06:15:51 PM - INFO - |  7  |  LINEAR_SVC  |  37.36%  |  [0.3934 0.3974 0.3888 0.3802 0.3858]  |  38.91 (+/- 1.19)  |  1.893  |  0.01857  |
03/07/2020 06:15:51 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  42.04%  |  [0.4282 0.4334 0.4152 0.4194 0.4218]  |  42.36 (+/- 1.29)  |  17.71  |  0.03946  |
03/07/2020 06:15:51 PM - INFO - |  9  |  MULTINOMIAL_NB  |  35.05%  |  [0.353  0.3502 0.3528 0.3514 0.3488]  |  35.12 (+/- 0.32)  |  0.03723  |  0.02325  |
03/07/2020 06:15:51 PM - INFO - |  10  |  NEAREST_CENTROID  |  37.12%  |  [0.3884 0.373  0.3818 0.367  0.372 ]  |  37.64 (+/- 1.53)  |  0.02274  |  0.02617  |
03/07/2020 06:15:51 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  37.19%  |  [0.3676 0.3678 0.369  0.362  0.3684]  |  36.70 (+/- 0.51)  |  72.59  |  1.511  |
03/07/2020 06:15:51 PM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 06:15:51 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 06:15:51 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 06:15:51 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  38.02%  |  [0.3792 0.379  0.374  0.3704 0.3746]  |  37.54 (+/- 0.66)  |  120.6  |  7.117  |
03/07/2020 06:15:51 PM - INFO - |  2  |  BERNOULLI_NB  |  37.03%  |  [0.377  0.389  0.3782 0.38   0.373 ]  |  37.94 (+/- 1.06)  |  0.0429  |  0.04091  |
03/07/2020 06:15:51 PM - INFO - |  3  |  COMPLEMENT_NB  |  37.34%  |  [0.3878 0.3942 0.3976 0.3938 0.3832]  |  39.13 (+/- 1.03)  |  0.03774  |  0.01899  |
03/07/2020 06:15:51 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  30.69%  |  [0.3152 0.3036 0.3054 0.3154 0.3048]  |  30.89 (+/- 1.05)  |  6.42  |  0.01237  |
03/07/2020 06:15:51 PM - INFO - |  7  |  LINEAR_SVC  |  40.75%  |  [0.4114 0.4206 0.4068 0.3992 0.4088]  |  40.94 (+/- 1.39)  |  0.6048  |  0.01785  |
03/07/2020 06:15:51 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  42.04%  |  [0.4282 0.4334 0.4152 0.4194 0.4218]  |  42.36 (+/- 1.29)  |  17.84  |  0.03951  |
03/07/2020 06:15:51 PM - INFO - |  9  |  MULTINOMIAL_NB  |  37.82%  |  [0.389  0.3928 0.3918 0.3942 0.386 ]  |  39.08 (+/- 0.59)  |  0.03643  |  0.02055  |
03/07/2020 06:15:51 PM - INFO - |  10  |  NEAREST_CENTROID  |  37.33%  |  [0.3872 0.3786 0.3894 0.3672 0.3782]  |  38.01 (+/- 1.57)  |  0.02111  |  0.03207  |
03/07/2020 06:15:51 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  37.82%  |  [0.3716 0.3772 0.3768 0.3742 0.368 ]  |  37.36 (+/- 0.69)  |  42.89  |  2.501  |
03/07/2020 06:15:51 PM - INFO - 

03/07/2020 06:15:51 PM - INFO - ################################################################################
03/07/2020 06:15:51 PM - INFO - 10)
03/07/2020 06:15:51 PM - INFO - ********************************************************************************
03/07/2020 06:15:51 PM - INFO - Classifier: PASSIVE_AGGRESSIVE_CLASSIFIER, Dataset: IMDB_REVIEWS
03/07/2020 06:15:51 PM - INFO - ********************************************************************************
03/07/2020 06:15:57 PM - INFO - 

Performing grid search...

03/07/2020 06:15:57 PM - INFO - Parameters:
03/07/2020 06:15:57 PM - INFO - {'classifier__C': [0.01, 1.0], 'classifier__early_stopping': [False, True], 'classifier__tol': [0.0001, 0.001, 0.01], 'classifier__validation_fraction': [0.0001, 0.01]}
03/07/2020 06:17:37 PM - INFO - 	Done in 99.425s
03/07/2020 06:17:37 PM - INFO - 	Best score: 0.417
03/07/2020 06:17:37 PM - INFO - 	Best parameters set:
03/07/2020 06:17:37 PM - INFO - 		classifier__C: 0.01
03/07/2020 06:17:37 PM - INFO - 		classifier__early_stopping: True
03/07/2020 06:17:37 PM - INFO - 		classifier__tol: 0.0001
03/07/2020 06:17:37 PM - INFO - 		classifier__validation_fraction: 0.01
03/07/2020 06:17:37 PM - INFO - 

USING PASSIVE_AGGRESSIVE_CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 06:17:37 PM - INFO - ________________________________________________________________________________
03/07/2020 06:17:37 PM - INFO - Training: 
03/07/2020 06:17:37 PM - INFO - PassiveAggressiveClassifier(C=1.0, average=False, class_weight=None,
                            early_stopping=False, fit_intercept=True,
                            loss='hinge', max_iter=1000, n_iter_no_change=5,
                            n_jobs=None, random_state=None, shuffle=True,
                            tol=0.001, validation_fraction=0.1, verbose=0,
                            warm_start=False)
03/07/2020 06:17:38 PM - INFO - Train time: 1.618s
03/07/2020 06:17:38 PM - INFO - Test time:  0.019s
03/07/2020 06:17:38 PM - INFO - Accuracy score:   0.334
03/07/2020 06:17:38 PM - INFO - 

===> Classification Report:

03/07/2020 06:17:38 PM - INFO -               precision    recall  f1-score   support

           1       0.53      0.59      0.56      5022
           2       0.17      0.16      0.16      2302
           3       0.19      0.17      0.18      2541
           4       0.24      0.24      0.24      2635
           7       0.20      0.19      0.19      2307
           8       0.21      0.21      0.21      2850
           9       0.19      0.16      0.17      2344
          10       0.48      0.51      0.50      4999

    accuracy                           0.33     25000
   macro avg       0.28      0.28      0.28     25000
weighted avg       0.32      0.33      0.33     25000

03/07/2020 06:17:38 PM - INFO - 

Cross validation:
03/07/2020 06:17:40 PM - INFO - 	accuracy: 5-fold cross validation: [0.3546 0.3592 0.354  0.3474 0.3492]
03/07/2020 06:17:40 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 35.29 (+/- 0.84)
03/07/2020 06:17:40 PM - INFO - 

USING PASSIVE_AGGRESSIVE_CLASSIFIER WITH BEST PARAMETERS: {'C': 0.01, 'early_stopping': True, 'tol': 0.0001, 'validation_fraction': 0.01}
03/07/2020 06:17:40 PM - INFO - ________________________________________________________________________________
03/07/2020 06:17:40 PM - INFO - Training: 
03/07/2020 06:17:40 PM - INFO - PassiveAggressiveClassifier(C=0.01, average=False, class_weight=None,
                            early_stopping=True, fit_intercept=True,
                            loss='hinge', max_iter=1000, n_iter_no_change=5,
                            n_jobs=None, random_state=None, shuffle=True,
                            tol=0.0001, validation_fraction=0.01, verbose=0,
                            warm_start=False)
03/07/2020 06:17:42 PM - INFO - Train time: 1.373s
03/07/2020 06:17:42 PM - INFO - Test time:  0.018s
03/07/2020 06:17:42 PM - INFO - Accuracy score:   0.418
03/07/2020 06:17:42 PM - INFO - 

===> Classification Report:

03/07/2020 06:17:42 PM - INFO -               precision    recall  f1-score   support

           1       0.48      0.88      0.62      5022
           2       0.12      0.02      0.03      2302
           3       0.26      0.06      0.10      2541
           4       0.31      0.30      0.30      2635
           7       0.28      0.24      0.26      2307
           8       0.27      0.18      0.21      2850
           9       0.22      0.03      0.06      2344
          10       0.48      0.78      0.60      4999

    accuracy                           0.42     25000
   macro avg       0.30      0.31      0.27     25000
weighted avg       0.34      0.42      0.34     25000

03/07/2020 06:17:42 PM - INFO - 

Cross validation:
03/07/2020 06:17:43 PM - INFO - 	accuracy: 5-fold cross validation: [0.4144 0.427  0.4118 0.4142 0.4138]
03/07/2020 06:17:43 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 41.62 (+/- 1.09)
03/07/2020 06:17:43 PM - INFO - It took 112.92583227157593 seconds
03/07/2020 06:17:43 PM - INFO - ********************************************************************************
03/07/2020 06:17:43 PM - INFO - ################################################################################
03/07/2020 06:17:43 PM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 06:17:43 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 06:17:43 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 06:17:43 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  35.78%  |  [0.352  0.3546 0.3442 0.3492 0.3464]  |  34.93 (+/- 0.75)  |  12.08  |  0.732  |
03/07/2020 06:17:43 PM - INFO - |  2  |  BERNOULLI_NB  |  37.25%  |  [0.3786 0.3812 0.3678 0.374  0.373 ]  |  37.49 (+/- 0.93)  |  0.03953  |  0.04032  |
03/07/2020 06:17:43 PM - INFO - |  3  |  COMPLEMENT_NB  |  37.30%  |  [0.3832 0.3858 0.3834 0.386  0.3776]  |  38.32 (+/- 0.61)  |  0.03533  |  0.0185  |
03/07/2020 06:17:43 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  25.04%  |  [0.2516 0.2634 0.2528 0.249  0.2542]  |  25.42 (+/- 0.98)  |  38.77  |  0.03135  |
03/07/2020 06:17:43 PM - INFO - |  7  |  LINEAR_SVC  |  37.36%  |  [0.3934 0.3974 0.3888 0.3802 0.3858]  |  38.91 (+/- 1.19)  |  1.893  |  0.01857  |
03/07/2020 06:17:43 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  42.04%  |  [0.4282 0.4334 0.4152 0.4194 0.4218]  |  42.36 (+/- 1.29)  |  17.71  |  0.03946  |
03/07/2020 06:17:43 PM - INFO - |  9  |  MULTINOMIAL_NB  |  35.05%  |  [0.353  0.3502 0.3528 0.3514 0.3488]  |  35.12 (+/- 0.32)  |  0.03723  |  0.02325  |
03/07/2020 06:17:43 PM - INFO - |  10  |  NEAREST_CENTROID  |  37.12%  |  [0.3884 0.373  0.3818 0.367  0.372 ]  |  37.64 (+/- 1.53)  |  0.02274  |  0.02617  |
03/07/2020 06:17:43 PM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  33.42%  |  [0.3546 0.3592 0.354  0.3474 0.3492]  |  35.29 (+/- 0.84)  |  1.618  |  0.01907  |
03/07/2020 06:17:43 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  37.19%  |  [0.3676 0.3678 0.369  0.362  0.3684]  |  36.70 (+/- 0.51)  |  72.59  |  1.511  |
03/07/2020 06:17:43 PM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 06:17:43 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 06:17:43 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 06:17:43 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  38.02%  |  [0.3792 0.379  0.374  0.3704 0.3746]  |  37.54 (+/- 0.66)  |  120.6  |  7.117  |
03/07/2020 06:17:43 PM - INFO - |  2  |  BERNOULLI_NB  |  37.03%  |  [0.377  0.389  0.3782 0.38   0.373 ]  |  37.94 (+/- 1.06)  |  0.0429  |  0.04091  |
03/07/2020 06:17:43 PM - INFO - |  3  |  COMPLEMENT_NB  |  37.34%  |  [0.3878 0.3942 0.3976 0.3938 0.3832]  |  39.13 (+/- 1.03)  |  0.03774  |  0.01899  |
03/07/2020 06:17:43 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  30.69%  |  [0.3152 0.3036 0.3054 0.3154 0.3048]  |  30.89 (+/- 1.05)  |  6.42  |  0.01237  |
03/07/2020 06:17:43 PM - INFO - |  7  |  LINEAR_SVC  |  40.75%  |  [0.4114 0.4206 0.4068 0.3992 0.4088]  |  40.94 (+/- 1.39)  |  0.6048  |  0.01785  |
03/07/2020 06:17:43 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  42.04%  |  [0.4282 0.4334 0.4152 0.4194 0.4218]  |  42.36 (+/- 1.29)  |  17.84  |  0.03951  |
03/07/2020 06:17:43 PM - INFO - |  9  |  MULTINOMIAL_NB  |  37.82%  |  [0.389  0.3928 0.3918 0.3942 0.386 ]  |  39.08 (+/- 0.59)  |  0.03643  |  0.02055  |
03/07/2020 06:17:43 PM - INFO - |  10  |  NEAREST_CENTROID  |  37.33%  |  [0.3872 0.3786 0.3894 0.3672 0.3782]  |  38.01 (+/- 1.57)  |  0.02111  |  0.03207  |
03/07/2020 06:17:43 PM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  41.83%  |  [0.4144 0.427  0.4118 0.4142 0.4138]  |  41.62 (+/- 1.09)  |  1.373  |  0.01845  |
03/07/2020 06:17:43 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  37.82%  |  [0.3716 0.3772 0.3768 0.3742 0.368 ]  |  37.36 (+/- 0.69)  |  42.89  |  2.501  |
03/07/2020 06:17:43 PM - INFO - 

03/07/2020 06:17:43 PM - INFO - ################################################################################
03/07/2020 06:17:43 PM - INFO - 11)
03/07/2020 06:17:43 PM - INFO - ********************************************************************************
03/07/2020 06:17:43 PM - INFO - Classifier: K_NEIGHBORS_CLASSIFIER, Dataset: IMDB_REVIEWS
03/07/2020 06:17:43 PM - INFO - ********************************************************************************
03/07/2020 06:17:50 PM - INFO - 

Performing grid search...

03/07/2020 06:17:50 PM - INFO - Parameters:
03/07/2020 06:17:50 PM - INFO - {'classifier__leaf_size': [5, 30], 'classifier__metric': ['euclidean', 'minkowski'], 'classifier__n_neighbors': [3, 50], 'classifier__weights': ['uniform', 'distance']}
03/07/2020 06:19:11 PM - INFO - 	Done in 80.733s
03/07/2020 06:19:11 PM - INFO - 	Best score: 0.386
03/07/2020 06:19:11 PM - INFO - 	Best parameters set:
03/07/2020 06:19:11 PM - INFO - 		classifier__leaf_size: 5
03/07/2020 06:19:11 PM - INFO - 		classifier__metric: 'euclidean'
03/07/2020 06:19:11 PM - INFO - 		classifier__n_neighbors: 50
03/07/2020 06:19:11 PM - INFO - 		classifier__weights: 'distance'
03/07/2020 06:19:11 PM - INFO - 

USING K_NEIGHBORS_CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 06:19:11 PM - INFO - ________________________________________________________________________________
03/07/2020 06:19:11 PM - INFO - Training: 
03/07/2020 06:19:11 PM - INFO - KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                     weights='uniform')
03/07/2020 06:19:11 PM - INFO - Train time: 0.009s
03/07/2020 06:19:38 PM - INFO - Test time:  27.447s
03/07/2020 06:19:38 PM - INFO - Accuracy score:   0.280
03/07/2020 06:19:38 PM - INFO - 

===> Classification Report:

03/07/2020 06:19:38 PM - INFO -               precision    recall  f1-score   support

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

03/07/2020 06:19:38 PM - INFO - 

Cross validation:
03/07/2020 06:19:45 PM - INFO - 	accuracy: 5-fold cross validation: [0.3366 0.33   0.3216 0.3218 0.3168]
03/07/2020 06:19:45 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 32.54 (+/- 1.41)
03/07/2020 06:19:45 PM - INFO - 

USING K_NEIGHBORS_CLASSIFIER WITH BEST PARAMETERS: {'leaf_size': 5, 'metric': 'euclidean', 'n_neighbors': 50, 'weights': 'distance'}
03/07/2020 06:19:45 PM - INFO - ________________________________________________________________________________
03/07/2020 06:19:45 PM - INFO - Training: 
03/07/2020 06:19:45 PM - INFO - KNeighborsClassifier(algorithm='auto', leaf_size=5, metric='euclidean',
                     metric_params=None, n_jobs=None, n_neighbors=50, p=2,
                     weights='distance')
03/07/2020 06:19:45 PM - INFO - Train time: 0.009s
03/07/2020 06:20:13 PM - INFO - Test time:  27.407s
03/07/2020 06:20:13 PM - INFO - Accuracy score:   0.373
03/07/2020 06:20:13 PM - INFO - 

===> Classification Report:

03/07/2020 06:20:13 PM - INFO -               precision    recall  f1-score   support

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

03/07/2020 06:20:13 PM - INFO - 

Cross validation:
03/07/2020 06:20:20 PM - INFO - 	accuracy: 5-fold cross validation: [0.3822 0.3916 0.3842 0.386  0.388 ]
03/07/2020 06:20:20 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 38.64 (+/- 0.65)
03/07/2020 06:20:20 PM - INFO - It took 156.27848052978516 seconds
03/07/2020 06:20:20 PM - INFO - ********************************************************************************
03/07/2020 06:20:20 PM - INFO - ################################################################################
03/07/2020 06:20:20 PM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 06:20:20 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 06:20:20 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 06:20:20 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  35.78%  |  [0.352  0.3546 0.3442 0.3492 0.3464]  |  34.93 (+/- 0.75)  |  12.08  |  0.732  |
03/07/2020 06:20:20 PM - INFO - |  2  |  BERNOULLI_NB  |  37.25%  |  [0.3786 0.3812 0.3678 0.374  0.373 ]  |  37.49 (+/- 0.93)  |  0.03953  |  0.04032  |
03/07/2020 06:20:20 PM - INFO - |  3  |  COMPLEMENT_NB  |  37.30%  |  [0.3832 0.3858 0.3834 0.386  0.3776]  |  38.32 (+/- 0.61)  |  0.03533  |  0.0185  |
03/07/2020 06:20:20 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  25.04%  |  [0.2516 0.2634 0.2528 0.249  0.2542]  |  25.42 (+/- 0.98)  |  38.77  |  0.03135  |
03/07/2020 06:20:20 PM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  28.00%  |  [0.3366 0.33   0.3216 0.3218 0.3168]  |  32.54 (+/- 1.41)  |  0.008613  |  27.45  |
03/07/2020 06:20:20 PM - INFO - |  7  |  LINEAR_SVC  |  37.36%  |  [0.3934 0.3974 0.3888 0.3802 0.3858]  |  38.91 (+/- 1.19)  |  1.893  |  0.01857  |
03/07/2020 06:20:20 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  42.04%  |  [0.4282 0.4334 0.4152 0.4194 0.4218]  |  42.36 (+/- 1.29)  |  17.71  |  0.03946  |
03/07/2020 06:20:20 PM - INFO - |  9  |  MULTINOMIAL_NB  |  35.05%  |  [0.353  0.3502 0.3528 0.3514 0.3488]  |  35.12 (+/- 0.32)  |  0.03723  |  0.02325  |
03/07/2020 06:20:20 PM - INFO - |  10  |  NEAREST_CENTROID  |  37.12%  |  [0.3884 0.373  0.3818 0.367  0.372 ]  |  37.64 (+/- 1.53)  |  0.02274  |  0.02617  |
03/07/2020 06:20:20 PM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  33.42%  |  [0.3546 0.3592 0.354  0.3474 0.3492]  |  35.29 (+/- 0.84)  |  1.618  |  0.01907  |
03/07/2020 06:20:20 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  37.19%  |  [0.3676 0.3678 0.369  0.362  0.3684]  |  36.70 (+/- 0.51)  |  72.59  |  1.511  |
03/07/2020 06:20:20 PM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 06:20:20 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 06:20:20 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 06:20:20 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  38.02%  |  [0.3792 0.379  0.374  0.3704 0.3746]  |  37.54 (+/- 0.66)  |  120.6  |  7.117  |
03/07/2020 06:20:20 PM - INFO - |  2  |  BERNOULLI_NB  |  37.03%  |  [0.377  0.389  0.3782 0.38   0.373 ]  |  37.94 (+/- 1.06)  |  0.0429  |  0.04091  |
03/07/2020 06:20:20 PM - INFO - |  3  |  COMPLEMENT_NB  |  37.34%  |  [0.3878 0.3942 0.3976 0.3938 0.3832]  |  39.13 (+/- 1.03)  |  0.03774  |  0.01899  |
03/07/2020 06:20:20 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  30.69%  |  [0.3152 0.3036 0.3054 0.3154 0.3048]  |  30.89 (+/- 1.05)  |  6.42  |  0.01237  |
03/07/2020 06:20:20 PM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  37.26%  |  [0.3822 0.3916 0.3842 0.386  0.388 ]  |  38.64 (+/- 0.65)  |  0.008674  |  27.41  |
03/07/2020 06:20:20 PM - INFO - |  7  |  LINEAR_SVC  |  40.75%  |  [0.4114 0.4206 0.4068 0.3992 0.4088]  |  40.94 (+/- 1.39)  |  0.6048  |  0.01785  |
03/07/2020 06:20:20 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  42.04%  |  [0.4282 0.4334 0.4152 0.4194 0.4218]  |  42.36 (+/- 1.29)  |  17.84  |  0.03951  |
03/07/2020 06:20:20 PM - INFO - |  9  |  MULTINOMIAL_NB  |  37.82%  |  [0.389  0.3928 0.3918 0.3942 0.386 ]  |  39.08 (+/- 0.59)  |  0.03643  |  0.02055  |
03/07/2020 06:20:20 PM - INFO - |  10  |  NEAREST_CENTROID  |  37.33%  |  [0.3872 0.3786 0.3894 0.3672 0.3782]  |  38.01 (+/- 1.57)  |  0.02111  |  0.03207  |
03/07/2020 06:20:20 PM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  41.83%  |  [0.4144 0.427  0.4118 0.4142 0.4138]  |  41.62 (+/- 1.09)  |  1.373  |  0.01845  |
03/07/2020 06:20:20 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  37.82%  |  [0.3716 0.3772 0.3768 0.3742 0.368 ]  |  37.36 (+/- 0.69)  |  42.89  |  2.501  |
03/07/2020 06:20:20 PM - INFO - 

03/07/2020 06:20:20 PM - INFO - ################################################################################
03/07/2020 06:20:20 PM - INFO - 12)
03/07/2020 06:20:20 PM - INFO - ********************************************************************************
03/07/2020 06:20:20 PM - INFO - Classifier: PERCEPTRON, Dataset: IMDB_REVIEWS
03/07/2020 06:20:20 PM - INFO - ********************************************************************************
03/07/2020 06:24:18 PM - INFO - 

Performing grid search...

03/07/2020 06:24:18 PM - INFO - Parameters:
03/07/2020 06:24:18 PM - INFO - {'classifier__early_stopping': [True], 'classifier__max_iter': [100], 'classifier__n_iter_no_change': [3, 15], 'classifier__penalty': ['l2'], 'classifier__tol': [0.0001, 0.1], 'classifier__validation_fraction': [0.0001, 0.01]}
03/07/2020 06:24:30 PM - INFO - 	Done in 12.230s
03/07/2020 06:24:30 PM - INFO - 	Best score: 0.313
03/07/2020 06:24:30 PM - INFO - 	Best parameters set:
03/07/2020 06:24:30 PM - INFO - 		classifier__early_stopping: True
03/07/2020 06:24:30 PM - INFO - 		classifier__max_iter: 100
03/07/2020 06:24:30 PM - INFO - 		classifier__n_iter_no_change: 3
03/07/2020 06:24:30 PM - INFO - 		classifier__penalty: 'l2'
03/07/2020 06:24:30 PM - INFO - 		classifier__tol: 0.0001
03/07/2020 06:24:30 PM - INFO - 		classifier__validation_fraction: 0.01
03/07/2020 06:24:30 PM - INFO - 

USING PERCEPTRON WITH DEFAULT PARAMETERS
03/07/2020 06:24:30 PM - INFO - ________________________________________________________________________________
03/07/2020 06:24:30 PM - INFO - Training: 
03/07/2020 06:24:30 PM - INFO - Perceptron(alpha=0.0001, class_weight=None, early_stopping=False, eta0=1.0,
           fit_intercept=True, max_iter=1000, n_iter_no_change=5, n_jobs=None,
           penalty=None, random_state=0, shuffle=True, tol=0.001,
           validation_fraction=0.1, verbose=0, warm_start=False)
03/07/2020 06:24:31 PM - INFO - Train time: 0.811s
03/07/2020 06:24:31 PM - INFO - Test time:  0.020s
03/07/2020 06:24:31 PM - INFO - Accuracy score:   0.331
03/07/2020 06:24:31 PM - INFO - 

===> Classification Report:

03/07/2020 06:24:31 PM - INFO -               precision    recall  f1-score   support

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

03/07/2020 06:24:31 PM - INFO - 

Cross validation:
03/07/2020 06:24:32 PM - INFO - 	accuracy: 5-fold cross validation: [0.3508 0.3604 0.3438 0.3348 0.3414]
03/07/2020 06:24:32 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 34.62 (+/- 1.75)
03/07/2020 06:24:32 PM - INFO - 

USING PERCEPTRON WITH BEST PARAMETERS: {'early_stopping': True, 'max_iter': 100, 'n_iter_no_change': 3, 'penalty': 'l2', 'tol': 0.0001, 'validation_fraction': 0.01}
03/07/2020 06:24:32 PM - INFO - ________________________________________________________________________________
03/07/2020 06:24:32 PM - INFO - Training: 
03/07/2020 06:24:32 PM - INFO - Perceptron(alpha=0.0001, class_weight=None, early_stopping=True, eta0=1.0,
           fit_intercept=True, max_iter=100, n_iter_no_change=3, n_jobs=None,
           penalty='l2', random_state=0, shuffle=True, tol=0.0001,
           validation_fraction=0.01, verbose=0, warm_start=False)
03/07/2020 06:24:33 PM - INFO - Train time: 1.051s
03/07/2020 06:24:33 PM - INFO - Test time:  0.019s
03/07/2020 06:24:33 PM - INFO - Accuracy score:   0.316
03/07/2020 06:24:33 PM - INFO - 

===> Classification Report:

03/07/2020 06:24:33 PM - INFO -               precision    recall  f1-score   support

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

03/07/2020 06:24:33 PM - INFO - 

Cross validation:
03/07/2020 06:24:35 PM - INFO - 	accuracy: 5-fold cross validation: [0.3364 0.3094 0.3268 0.298  0.2964]
03/07/2020 06:24:35 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 31.34 (+/- 3.16)
03/07/2020 06:24:35 PM - INFO - It took 254.94830799102783 seconds
03/07/2020 06:24:35 PM - INFO - ********************************************************************************
03/07/2020 06:24:35 PM - INFO - ################################################################################
03/07/2020 06:24:35 PM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 06:24:35 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 06:24:35 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 06:24:35 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  35.78%  |  [0.352  0.3546 0.3442 0.3492 0.3464]  |  34.93 (+/- 0.75)  |  12.08  |  0.732  |
03/07/2020 06:24:35 PM - INFO - |  2  |  BERNOULLI_NB  |  37.25%  |  [0.3786 0.3812 0.3678 0.374  0.373 ]  |  37.49 (+/- 0.93)  |  0.03953  |  0.04032  |
03/07/2020 06:24:35 PM - INFO - |  3  |  COMPLEMENT_NB  |  37.30%  |  [0.3832 0.3858 0.3834 0.386  0.3776]  |  38.32 (+/- 0.61)  |  0.03533  |  0.0185  |
03/07/2020 06:24:35 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  25.04%  |  [0.2516 0.2634 0.2528 0.249  0.2542]  |  25.42 (+/- 0.98)  |  38.77  |  0.03135  |
03/07/2020 06:24:35 PM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  28.00%  |  [0.3366 0.33   0.3216 0.3218 0.3168]  |  32.54 (+/- 1.41)  |  0.008613  |  27.45  |
03/07/2020 06:24:35 PM - INFO - |  7  |  LINEAR_SVC  |  37.36%  |  [0.3934 0.3974 0.3888 0.3802 0.3858]  |  38.91 (+/- 1.19)  |  1.893  |  0.01857  |
03/07/2020 06:24:35 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  42.04%  |  [0.4282 0.4334 0.4152 0.4194 0.4218]  |  42.36 (+/- 1.29)  |  17.71  |  0.03946  |
03/07/2020 06:24:35 PM - INFO - |  9  |  MULTINOMIAL_NB  |  35.05%  |  [0.353  0.3502 0.3528 0.3514 0.3488]  |  35.12 (+/- 0.32)  |  0.03723  |  0.02325  |
03/07/2020 06:24:35 PM - INFO - |  10  |  NEAREST_CENTROID  |  37.12%  |  [0.3884 0.373  0.3818 0.367  0.372 ]  |  37.64 (+/- 1.53)  |  0.02274  |  0.02617  |
03/07/2020 06:24:35 PM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  33.42%  |  [0.3546 0.3592 0.354  0.3474 0.3492]  |  35.29 (+/- 0.84)  |  1.618  |  0.01907  |
03/07/2020 06:24:35 PM - INFO - |  12  |  PERCEPTRON  |  33.09%  |  [0.3508 0.3604 0.3438 0.3348 0.3414]  |  34.62 (+/- 1.75)  |  0.8114  |  0.02037  |
03/07/2020 06:24:35 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  37.19%  |  [0.3676 0.3678 0.369  0.362  0.3684]  |  36.70 (+/- 0.51)  |  72.59  |  1.511  |
03/07/2020 06:24:35 PM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 06:24:35 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 06:24:35 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 06:24:35 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  38.02%  |  [0.3792 0.379  0.374  0.3704 0.3746]  |  37.54 (+/- 0.66)  |  120.6  |  7.117  |
03/07/2020 06:24:35 PM - INFO - |  2  |  BERNOULLI_NB  |  37.03%  |  [0.377  0.389  0.3782 0.38   0.373 ]  |  37.94 (+/- 1.06)  |  0.0429  |  0.04091  |
03/07/2020 06:24:35 PM - INFO - |  3  |  COMPLEMENT_NB  |  37.34%  |  [0.3878 0.3942 0.3976 0.3938 0.3832]  |  39.13 (+/- 1.03)  |  0.03774  |  0.01899  |
03/07/2020 06:24:35 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  30.69%  |  [0.3152 0.3036 0.3054 0.3154 0.3048]  |  30.89 (+/- 1.05)  |  6.42  |  0.01237  |
03/07/2020 06:24:35 PM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  37.26%  |  [0.3822 0.3916 0.3842 0.386  0.388 ]  |  38.64 (+/- 0.65)  |  0.008674  |  27.41  |
03/07/2020 06:24:35 PM - INFO - |  7  |  LINEAR_SVC  |  40.75%  |  [0.4114 0.4206 0.4068 0.3992 0.4088]  |  40.94 (+/- 1.39)  |  0.6048  |  0.01785  |
03/07/2020 06:24:35 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  42.04%  |  [0.4282 0.4334 0.4152 0.4194 0.4218]  |  42.36 (+/- 1.29)  |  17.84  |  0.03951  |
03/07/2020 06:24:35 PM - INFO - |  9  |  MULTINOMIAL_NB  |  37.82%  |  [0.389  0.3928 0.3918 0.3942 0.386 ]  |  39.08 (+/- 0.59)  |  0.03643  |  0.02055  |
03/07/2020 06:24:35 PM - INFO - |  10  |  NEAREST_CENTROID  |  37.33%  |  [0.3872 0.3786 0.3894 0.3672 0.3782]  |  38.01 (+/- 1.57)  |  0.02111  |  0.03207  |
03/07/2020 06:24:35 PM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  41.83%  |  [0.4144 0.427  0.4118 0.4142 0.4138]  |  41.62 (+/- 1.09)  |  1.373  |  0.01845  |
03/07/2020 06:24:35 PM - INFO - |  12  |  PERCEPTRON  |  31.60%  |  [0.3364 0.3094 0.3268 0.298  0.2964]  |  31.34 (+/- 3.16)  |  1.051  |  0.01868  |
03/07/2020 06:24:35 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  37.82%  |  [0.3716 0.3772 0.3768 0.3742 0.368 ]  |  37.36 (+/- 0.69)  |  42.89  |  2.501  |
03/07/2020 06:24:35 PM - INFO - 

03/07/2020 06:24:35 PM - INFO - ################################################################################
03/07/2020 06:24:35 PM - INFO - 13)
03/07/2020 06:24:35 PM - INFO - ********************************************************************************
03/07/2020 06:24:35 PM - INFO - Classifier: RIDGE_CLASSIFIER, Dataset: IMDB_REVIEWS
03/07/2020 06:24:35 PM - INFO - ********************************************************************************
03/07/2020 06:24:42 PM - INFO - 

Performing grid search...

03/07/2020 06:24:42 PM - INFO - Parameters:
03/07/2020 06:24:42 PM - INFO - {'classifier__alpha': [0.5, 1.0], 'classifier__tol': [0.0001, 0.001]}
03/07/2020 06:24:54 PM - INFO - 	Done in 12.715s
03/07/2020 06:24:54 PM - INFO - 	Best score: 0.402
03/07/2020 06:24:54 PM - INFO - 	Best parameters set:
03/07/2020 06:24:54 PM - INFO - 		classifier__alpha: 1.0
03/07/2020 06:24:54 PM - INFO - 		classifier__tol: 0.001
03/07/2020 06:24:54 PM - INFO - 

USING RIDGE_CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 06:24:54 PM - INFO - ________________________________________________________________________________
03/07/2020 06:24:54 PM - INFO - Training: 
03/07/2020 06:24:54 PM - INFO - RidgeClassifier(alpha=1.0, class_weight=None, copy_X=True, fit_intercept=True,
                max_iter=None, normalize=False, random_state=None,
                solver='auto', tol=0.001)
03/07/2020 06:24:57 PM - INFO - Train time: 2.930s
03/07/2020 06:24:57 PM - INFO - Test time:  0.041s
03/07/2020 06:24:57 PM - INFO - Accuracy score:   0.386
03/07/2020 06:24:57 PM - INFO - 

===> Classification Report:

03/07/2020 06:24:57 PM - INFO -               precision    recall  f1-score   support

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

03/07/2020 06:24:57 PM - INFO - 

Cross validation:
03/07/2020 06:24:59 PM - INFO - 	accuracy: 5-fold cross validation: [0.4036 0.4074 0.402  0.3954 0.4   ]
03/07/2020 06:24:59 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 40.17 (+/- 0.79)
03/07/2020 06:24:59 PM - INFO - 

USING RIDGE_CLASSIFIER WITH BEST PARAMETERS: {'alpha': 1.0, 'tol': 0.001}
03/07/2020 06:24:59 PM - INFO - ________________________________________________________________________________
03/07/2020 06:24:59 PM - INFO - Training: 
03/07/2020 06:24:59 PM - INFO - RidgeClassifier(alpha=1.0, class_weight=None, copy_X=True, fit_intercept=True,
                max_iter=None, normalize=False, random_state=None,
                solver='auto', tol=0.001)
03/07/2020 06:25:02 PM - INFO - Train time: 2.896s
03/07/2020 06:25:02 PM - INFO - Test time:  0.041s
03/07/2020 06:25:02 PM - INFO - Accuracy score:   0.386
03/07/2020 06:25:02 PM - INFO - 

===> Classification Report:

03/07/2020 06:25:02 PM - INFO -               precision    recall  f1-score   support

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

03/07/2020 06:25:02 PM - INFO - 

Cross validation:
03/07/2020 06:25:05 PM - INFO - 	accuracy: 5-fold cross validation: [0.4036 0.4074 0.402  0.3954 0.4   ]
03/07/2020 06:25:05 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 40.17 (+/- 0.79)
03/07/2020 06:25:05 PM - INFO - It took 29.88110899925232 seconds
03/07/2020 06:25:05 PM - INFO - ********************************************************************************
03/07/2020 06:25:05 PM - INFO - ################################################################################
03/07/2020 06:25:05 PM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 06:25:05 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 06:25:05 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 06:25:05 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  35.78%  |  [0.352  0.3546 0.3442 0.3492 0.3464]  |  34.93 (+/- 0.75)  |  12.08  |  0.732  |
03/07/2020 06:25:05 PM - INFO - |  2  |  BERNOULLI_NB  |  37.25%  |  [0.3786 0.3812 0.3678 0.374  0.373 ]  |  37.49 (+/- 0.93)  |  0.03953  |  0.04032  |
03/07/2020 06:25:05 PM - INFO - |  3  |  COMPLEMENT_NB  |  37.30%  |  [0.3832 0.3858 0.3834 0.386  0.3776]  |  38.32 (+/- 0.61)  |  0.03533  |  0.0185  |
03/07/2020 06:25:05 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  25.04%  |  [0.2516 0.2634 0.2528 0.249  0.2542]  |  25.42 (+/- 0.98)  |  38.77  |  0.03135  |
03/07/2020 06:25:05 PM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  28.00%  |  [0.3366 0.33   0.3216 0.3218 0.3168]  |  32.54 (+/- 1.41)  |  0.008613  |  27.45  |
03/07/2020 06:25:05 PM - INFO - |  7  |  LINEAR_SVC  |  37.36%  |  [0.3934 0.3974 0.3888 0.3802 0.3858]  |  38.91 (+/- 1.19)  |  1.893  |  0.01857  |
03/07/2020 06:25:05 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  42.04%  |  [0.4282 0.4334 0.4152 0.4194 0.4218]  |  42.36 (+/- 1.29)  |  17.71  |  0.03946  |
03/07/2020 06:25:05 PM - INFO - |  9  |  MULTINOMIAL_NB  |  35.05%  |  [0.353  0.3502 0.3528 0.3514 0.3488]  |  35.12 (+/- 0.32)  |  0.03723  |  0.02325  |
03/07/2020 06:25:05 PM - INFO - |  10  |  NEAREST_CENTROID  |  37.12%  |  [0.3884 0.373  0.3818 0.367  0.372 ]  |  37.64 (+/- 1.53)  |  0.02274  |  0.02617  |
03/07/2020 06:25:05 PM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  33.42%  |  [0.3546 0.3592 0.354  0.3474 0.3492]  |  35.29 (+/- 0.84)  |  1.618  |  0.01907  |
03/07/2020 06:25:05 PM - INFO - |  12  |  PERCEPTRON  |  33.09%  |  [0.3508 0.3604 0.3438 0.3348 0.3414]  |  34.62 (+/- 1.75)  |  0.8114  |  0.02037  |
03/07/2020 06:25:05 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  37.19%  |  [0.3676 0.3678 0.369  0.362  0.3684]  |  36.70 (+/- 0.51)  |  72.59  |  1.511  |
03/07/2020 06:25:05 PM - INFO - |  14  |  RIDGE_CLASSIFIER  |  38.56%  |  [0.4036 0.4074 0.402  0.3954 0.4   ]  |  40.17 (+/- 0.79)  |  2.93  |  0.04089  |
03/07/2020 06:25:05 PM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 06:25:05 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 06:25:05 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 06:25:05 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  38.02%  |  [0.3792 0.379  0.374  0.3704 0.3746]  |  37.54 (+/- 0.66)  |  120.6  |  7.117  |
03/07/2020 06:25:05 PM - INFO - |  2  |  BERNOULLI_NB  |  37.03%  |  [0.377  0.389  0.3782 0.38   0.373 ]  |  37.94 (+/- 1.06)  |  0.0429  |  0.04091  |
03/07/2020 06:25:05 PM - INFO - |  3  |  COMPLEMENT_NB  |  37.34%  |  [0.3878 0.3942 0.3976 0.3938 0.3832]  |  39.13 (+/- 1.03)  |  0.03774  |  0.01899  |
03/07/2020 06:25:05 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  30.69%  |  [0.3152 0.3036 0.3054 0.3154 0.3048]  |  30.89 (+/- 1.05)  |  6.42  |  0.01237  |
03/07/2020 06:25:05 PM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  37.26%  |  [0.3822 0.3916 0.3842 0.386  0.388 ]  |  38.64 (+/- 0.65)  |  0.008674  |  27.41  |
03/07/2020 06:25:05 PM - INFO - |  7  |  LINEAR_SVC  |  40.75%  |  [0.4114 0.4206 0.4068 0.3992 0.4088]  |  40.94 (+/- 1.39)  |  0.6048  |  0.01785  |
03/07/2020 06:25:05 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  42.04%  |  [0.4282 0.4334 0.4152 0.4194 0.4218]  |  42.36 (+/- 1.29)  |  17.84  |  0.03951  |
03/07/2020 06:25:05 PM - INFO - |  9  |  MULTINOMIAL_NB  |  37.82%  |  [0.389  0.3928 0.3918 0.3942 0.386 ]  |  39.08 (+/- 0.59)  |  0.03643  |  0.02055  |
03/07/2020 06:25:05 PM - INFO - |  10  |  NEAREST_CENTROID  |  37.33%  |  [0.3872 0.3786 0.3894 0.3672 0.3782]  |  38.01 (+/- 1.57)  |  0.02111  |  0.03207  |
03/07/2020 06:25:05 PM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  41.83%  |  [0.4144 0.427  0.4118 0.4142 0.4138]  |  41.62 (+/- 1.09)  |  1.373  |  0.01845  |
03/07/2020 06:25:05 PM - INFO - |  12  |  PERCEPTRON  |  31.60%  |  [0.3364 0.3094 0.3268 0.298  0.2964]  |  31.34 (+/- 3.16)  |  1.051  |  0.01868  |
03/07/2020 06:25:05 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  37.82%  |  [0.3716 0.3772 0.3768 0.3742 0.368 ]  |  37.36 (+/- 0.69)  |  42.89  |  2.501  |
03/07/2020 06:25:05 PM - INFO - |  14  |  RIDGE_CLASSIFIER  |  38.56%  |  [0.4036 0.4074 0.402  0.3954 0.4   ]  |  40.17 (+/- 0.79)  |  2.896  |  0.04107  |
03/07/2020 06:25:05 PM - INFO - 

03/07/2020 06:25:05 PM - INFO - ################################################################################
03/07/2020 06:25:05 PM - INFO - 14)
03/07/2020 06:25:05 PM - INFO - ********************************************************************************
03/07/2020 06:25:05 PM - INFO - Classifier: GRADIENT_BOOSTING_CLASSIFIER, Dataset: IMDB_REVIEWS
03/07/2020 06:25:05 PM - INFO - ********************************************************************************
03/07/2020 06:25:11 PM - INFO - 

Performing grid search...

03/07/2020 06:25:11 PM - INFO - Parameters:
03/07/2020 06:25:11 PM - INFO - {'classifier__learning_rate': [0.01, 0.1], 'classifier__n_estimators': [100, 200]}
03/07/2020 08:07:11 PM - INFO - 	Done in 6119.255s
03/07/2020 08:07:11 PM - INFO - 	Best score: 0.373
03/07/2020 08:07:11 PM - INFO - 	Best parameters set:
03/07/2020 08:07:11 PM - INFO - 		classifier__learning_rate: 0.1
03/07/2020 08:07:11 PM - INFO - 		classifier__n_estimators: 200
03/07/2020 08:07:11 PM - INFO - 

USING GRADIENT_BOOSTING_CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 08:07:11 PM - INFO - ________________________________________________________________________________
03/07/2020 08:07:11 PM - INFO - Training: 
03/07/2020 08:07:11 PM - INFO - GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.1, loss='deviance', max_depth=3,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_iter_no_change=None, presort='deprecated',
                           random_state=None, subsample=1.0, tol=0.0001,
                           validation_fraction=0.1, verbose=0,
                           warm_start=False)
03/07/2020 08:14:22 PM - INFO - Train time: 430.919s
03/07/2020 08:14:22 PM - INFO - Test time:  0.277s
03/07/2020 08:14:22 PM - INFO - Accuracy score:   0.375
03/07/2020 08:14:22 PM - INFO - 

===> Classification Report:

03/07/2020 08:14:22 PM - INFO -               precision    recall  f1-score   support

           1       0.47      0.76      0.58      5022
           2       0.17      0.03      0.06      2302
           3       0.26      0.08      0.12      2541
           4       0.28      0.18      0.22      2635
           7       0.27      0.15      0.19      2307
           8       0.23      0.17      0.19      2850
           9       0.16      0.03      0.05      2344
          10       0.39      0.78      0.52      4999

    accuracy                           0.37     25000
   macro avg       0.28      0.27      0.24     25000
weighted avg       0.31      0.37      0.30     25000

03/07/2020 08:14:22 PM - INFO - 

Cross validation:
03/07/2020 08:31:12 PM - INFO - 	accuracy: 5-fold cross validation: [0.3746 0.367  0.3602 0.3652 0.3682]
03/07/2020 08:31:12 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 36.70 (+/- 0.93)
03/07/2020 08:31:12 PM - INFO - 

USING GRADIENT_BOOSTING_CLASSIFIER WITH BEST PARAMETERS: {'learning_rate': 0.1, 'n_estimators': 200}
03/07/2020 08:31:12 PM - INFO - ________________________________________________________________________________
03/07/2020 08:31:12 PM - INFO - Training: 
03/07/2020 08:31:12 PM - INFO - GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.1, loss='deviance', max_depth=3,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=200,
                           n_iter_no_change=None, presort='deprecated',
                           random_state=None, subsample=1.0, tol=0.0001,
                           validation_fraction=0.1, verbose=0,
                           warm_start=False)
03/07/2020 08:45:35 PM - INFO - Train time: 863.329s
03/07/2020 08:45:36 PM - INFO - Test time:  0.491s
03/07/2020 08:45:36 PM - INFO - Accuracy score:   0.379
03/07/2020 08:45:36 PM - INFO - 

===> Classification Report:

03/07/2020 08:45:36 PM - INFO -               precision    recall  f1-score   support

           1       0.49      0.77      0.59      5022
           2       0.20      0.06      0.09      2302
           3       0.25      0.10      0.14      2541
           4       0.28      0.20      0.24      2635
           7       0.25      0.16      0.20      2307
           8       0.21      0.18      0.19      2850
           9       0.15      0.04      0.06      2344
          10       0.41      0.75      0.53      4999

    accuracy                           0.38     25000
   macro avg       0.28      0.28      0.25     25000
weighted avg       0.31      0.38      0.32     25000

03/07/2020 08:45:36 PM - INFO - 

Cross validation:
03/07/2020 09:18:54 PM - INFO - 	accuracy: 5-fold cross validation: [0.3766 0.3724 0.3666 0.3708 0.3784]
03/07/2020 09:18:54 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 37.30 (+/- 0.84)
03/07/2020 09:18:54 PM - INFO - It took 10428.95735836029 seconds
03/07/2020 09:18:54 PM - INFO - ********************************************************************************
03/07/2020 09:18:54 PM - INFO - ################################################################################
03/07/2020 09:18:54 PM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 09:18:54 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 09:18:54 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 09:18:54 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  35.78%  |  [0.352  0.3546 0.3442 0.3492 0.3464]  |  34.93 (+/- 0.75)  |  12.08  |  0.732  |
03/07/2020 09:18:54 PM - INFO - |  2  |  BERNOULLI_NB  |  37.25%  |  [0.3786 0.3812 0.3678 0.374  0.373 ]  |  37.49 (+/- 0.93)  |  0.03953  |  0.04032  |
03/07/2020 09:18:54 PM - INFO - |  3  |  COMPLEMENT_NB  |  37.30%  |  [0.3832 0.3858 0.3834 0.386  0.3776]  |  38.32 (+/- 0.61)  |  0.03533  |  0.0185  |
03/07/2020 09:18:54 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  25.04%  |  [0.2516 0.2634 0.2528 0.249  0.2542]  |  25.42 (+/- 0.98)  |  38.77  |  0.03135  |
03/07/2020 09:18:54 PM - INFO - |  5  |  GRADIENT_BOOSTING_CLASSIFIER  |  37.46%  |  [0.3746 0.367  0.3602 0.3652 0.3682]  |  36.70 (+/- 0.93)  |  430.9  |  0.2771  |
03/07/2020 09:18:54 PM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  28.00%  |  [0.3366 0.33   0.3216 0.3218 0.3168]  |  32.54 (+/- 1.41)  |  0.008613  |  27.45  |
03/07/2020 09:18:54 PM - INFO - |  7  |  LINEAR_SVC  |  37.36%  |  [0.3934 0.3974 0.3888 0.3802 0.3858]  |  38.91 (+/- 1.19)  |  1.893  |  0.01857  |
03/07/2020 09:18:54 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  42.04%  |  [0.4282 0.4334 0.4152 0.4194 0.4218]  |  42.36 (+/- 1.29)  |  17.71  |  0.03946  |
03/07/2020 09:18:54 PM - INFO - |  9  |  MULTINOMIAL_NB  |  35.05%  |  [0.353  0.3502 0.3528 0.3514 0.3488]  |  35.12 (+/- 0.32)  |  0.03723  |  0.02325  |
03/07/2020 09:18:54 PM - INFO - |  10  |  NEAREST_CENTROID  |  37.12%  |  [0.3884 0.373  0.3818 0.367  0.372 ]  |  37.64 (+/- 1.53)  |  0.02274  |  0.02617  |
03/07/2020 09:18:54 PM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  33.42%  |  [0.3546 0.3592 0.354  0.3474 0.3492]  |  35.29 (+/- 0.84)  |  1.618  |  0.01907  |
03/07/2020 09:18:54 PM - INFO - |  12  |  PERCEPTRON  |  33.09%  |  [0.3508 0.3604 0.3438 0.3348 0.3414]  |  34.62 (+/- 1.75)  |  0.8114  |  0.02037  |
03/07/2020 09:18:54 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  37.19%  |  [0.3676 0.3678 0.369  0.362  0.3684]  |  36.70 (+/- 0.51)  |  72.59  |  1.511  |
03/07/2020 09:18:54 PM - INFO - |  14  |  RIDGE_CLASSIFIER  |  38.56%  |  [0.4036 0.4074 0.402  0.3954 0.4   ]  |  40.17 (+/- 0.79)  |  2.93  |  0.04089  |
03/07/2020 09:18:54 PM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 09:18:54 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 09:18:54 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 09:18:54 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  38.02%  |  [0.3792 0.379  0.374  0.3704 0.3746]  |  37.54 (+/- 0.66)  |  120.6  |  7.117  |
03/07/2020 09:18:54 PM - INFO - |  2  |  BERNOULLI_NB  |  37.03%  |  [0.377  0.389  0.3782 0.38   0.373 ]  |  37.94 (+/- 1.06)  |  0.0429  |  0.04091  |
03/07/2020 09:18:54 PM - INFO - |  3  |  COMPLEMENT_NB  |  37.34%  |  [0.3878 0.3942 0.3976 0.3938 0.3832]  |  39.13 (+/- 1.03)  |  0.03774  |  0.01899  |
03/07/2020 09:18:54 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  30.69%  |  [0.3152 0.3036 0.3054 0.3154 0.3048]  |  30.89 (+/- 1.05)  |  6.42  |  0.01237  |
03/07/2020 09:18:54 PM - INFO - |  5  |  GRADIENT_BOOSTING_CLASSIFIER  |  37.86%  |  [0.3766 0.3724 0.3666 0.3708 0.3784]  |  37.30 (+/- 0.84)  |  863.3  |  0.4914  |
03/07/2020 09:18:54 PM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  37.26%  |  [0.3822 0.3916 0.3842 0.386  0.388 ]  |  38.64 (+/- 0.65)  |  0.008674  |  27.41  |
03/07/2020 09:18:54 PM - INFO - |  7  |  LINEAR_SVC  |  40.75%  |  [0.4114 0.4206 0.4068 0.3992 0.4088]  |  40.94 (+/- 1.39)  |  0.6048  |  0.01785  |
03/07/2020 09:18:54 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  42.04%  |  [0.4282 0.4334 0.4152 0.4194 0.4218]  |  42.36 (+/- 1.29)  |  17.84  |  0.03951  |
03/07/2020 09:18:54 PM - INFO - |  9  |  MULTINOMIAL_NB  |  37.82%  |  [0.389  0.3928 0.3918 0.3942 0.386 ]  |  39.08 (+/- 0.59)  |  0.03643  |  0.02055  |
03/07/2020 09:18:54 PM - INFO - |  10  |  NEAREST_CENTROID  |  37.33%  |  [0.3872 0.3786 0.3894 0.3672 0.3782]  |  38.01 (+/- 1.57)  |  0.02111  |  0.03207  |
03/07/2020 09:18:54 PM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  41.83%  |  [0.4144 0.427  0.4118 0.4142 0.4138]  |  41.62 (+/- 1.09)  |  1.373  |  0.01845  |
03/07/2020 09:18:54 PM - INFO - |  12  |  PERCEPTRON  |  31.60%  |  [0.3364 0.3094 0.3268 0.298  0.2964]  |  31.34 (+/- 3.16)  |  1.051  |  0.01868  |
03/07/2020 09:18:54 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  37.82%  |  [0.3716 0.3772 0.3768 0.3742 0.368 ]  |  37.36 (+/- 0.69)  |  42.89  |  2.501  |
03/07/2020 09:18:54 PM - INFO - |  14  |  RIDGE_CLASSIFIER  |  38.56%  |  [0.4036 0.4074 0.402  0.3954 0.4   ]  |  40.17 (+/- 0.79)  |  2.896  |  0.04107  |
03/07/2020 09:18:54 PM - INFO - 

FINAL CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 09:18:54 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 09:18:54 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 09:18:54 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  35.78%  |  [0.352  0.3546 0.3442 0.3492 0.3464]  |  34.93 (+/- 0.75)  |  12.08  |  0.732  |
03/07/2020 09:18:54 PM - INFO - |  2  |  BERNOULLI_NB  |  37.25%  |  [0.3786 0.3812 0.3678 0.374  0.373 ]  |  37.49 (+/- 0.93)  |  0.03953  |  0.04032  |
03/07/2020 09:18:54 PM - INFO - |  3  |  COMPLEMENT_NB  |  37.30%  |  [0.3832 0.3858 0.3834 0.386  0.3776]  |  38.32 (+/- 0.61)  |  0.03533  |  0.0185  |
03/07/2020 09:18:54 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  25.04%  |  [0.2516 0.2634 0.2528 0.249  0.2542]  |  25.42 (+/- 0.98)  |  38.77  |  0.03135  |
03/07/2020 09:18:54 PM - INFO - |  5  |  GRADIENT_BOOSTING_CLASSIFIER  |  37.46%  |  [0.3746 0.367  0.3602 0.3652 0.3682]  |  36.70 (+/- 0.93)  |  430.9  |  0.2771  |
03/07/2020 09:18:54 PM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  28.00%  |  [0.3366 0.33   0.3216 0.3218 0.3168]  |  32.54 (+/- 1.41)  |  0.008613  |  27.45  |
03/07/2020 09:18:54 PM - INFO - |  7  |  LINEAR_SVC  |  37.36%  |  [0.3934 0.3974 0.3888 0.3802 0.3858]  |  38.91 (+/- 1.19)  |  1.893  |  0.01857  |
03/07/2020 09:18:54 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  42.04%  |  [0.4282 0.4334 0.4152 0.4194 0.4218]  |  42.36 (+/- 1.29)  |  17.71  |  0.03946  |
03/07/2020 09:18:54 PM - INFO - |  9  |  MULTINOMIAL_NB  |  35.05%  |  [0.353  0.3502 0.3528 0.3514 0.3488]  |  35.12 (+/- 0.32)  |  0.03723  |  0.02325  |
03/07/2020 09:18:54 PM - INFO - |  10  |  NEAREST_CENTROID  |  37.12%  |  [0.3884 0.373  0.3818 0.367  0.372 ]  |  37.64 (+/- 1.53)  |  0.02274  |  0.02617  |
03/07/2020 09:18:54 PM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  33.42%  |  [0.3546 0.3592 0.354  0.3474 0.3492]  |  35.29 (+/- 0.84)  |  1.618  |  0.01907  |
03/07/2020 09:18:54 PM - INFO - |  12  |  PERCEPTRON  |  33.09%  |  [0.3508 0.3604 0.3438 0.3348 0.3414]  |  34.62 (+/- 1.75)  |  0.8114  |  0.02037  |
03/07/2020 09:18:54 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  37.19%  |  [0.3676 0.3678 0.369  0.362  0.3684]  |  36.70 (+/- 0.51)  |  72.59  |  1.511  |
03/07/2020 09:18:54 PM - INFO - |  14  |  RIDGE_CLASSIFIER  |  38.56%  |  [0.4036 0.4074 0.402  0.3954 0.4   ]  |  40.17 (+/- 0.79)  |  2.93  |  0.04089  |
03/07/2020 09:18:54 PM - INFO - 

FINAL CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 09:18:54 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 09:18:54 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 09:18:54 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  38.02%  |  [0.3792 0.379  0.374  0.3704 0.3746]  |  37.54 (+/- 0.66)  |  120.6  |  7.117  |
03/07/2020 09:18:54 PM - INFO - |  2  |  BERNOULLI_NB  |  37.03%  |  [0.377  0.389  0.3782 0.38   0.373 ]  |  37.94 (+/- 1.06)  |  0.0429  |  0.04091  |
03/07/2020 09:18:54 PM - INFO - |  3  |  COMPLEMENT_NB  |  37.34%  |  [0.3878 0.3942 0.3976 0.3938 0.3832]  |  39.13 (+/- 1.03)  |  0.03774  |  0.01899  |
03/07/2020 09:18:54 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  30.69%  |  [0.3152 0.3036 0.3054 0.3154 0.3048]  |  30.89 (+/- 1.05)  |  6.42  |  0.01237  |
03/07/2020 09:18:54 PM - INFO - |  5  |  GRADIENT_BOOSTING_CLASSIFIER  |  37.86%  |  [0.3766 0.3724 0.3666 0.3708 0.3784]  |  37.30 (+/- 0.84)  |  863.3  |  0.4914  |
03/07/2020 09:18:54 PM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  37.26%  |  [0.3822 0.3916 0.3842 0.386  0.388 ]  |  38.64 (+/- 0.65)  |  0.008674  |  27.41  |
03/07/2020 09:18:54 PM - INFO - |  7  |  LINEAR_SVC  |  40.75%  |  [0.4114 0.4206 0.4068 0.3992 0.4088]  |  40.94 (+/- 1.39)  |  0.6048  |  0.01785  |
03/07/2020 09:18:54 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  42.04%  |  [0.4282 0.4334 0.4152 0.4194 0.4218]  |  42.36 (+/- 1.29)  |  17.84  |  0.03951  |
03/07/2020 09:18:54 PM - INFO - |  9  |  MULTINOMIAL_NB  |  37.82%  |  [0.389  0.3928 0.3918 0.3942 0.386 ]  |  39.08 (+/- 0.59)  |  0.03643  |  0.02055  |
03/07/2020 09:18:54 PM - INFO - |  10  |  NEAREST_CENTROID  |  37.33%  |  [0.3872 0.3786 0.3894 0.3672 0.3782]  |  38.01 (+/- 1.57)  |  0.02111  |  0.03207  |
03/07/2020 09:18:54 PM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  41.83%  |  [0.4144 0.427  0.4118 0.4142 0.4138]  |  41.62 (+/- 1.09)  |  1.373  |  0.01845  |
03/07/2020 09:18:54 PM - INFO - |  12  |  PERCEPTRON  |  31.60%  |  [0.3364 0.3094 0.3268 0.298  0.2964]  |  31.34 (+/- 3.16)  |  1.051  |  0.01868  |
03/07/2020 09:18:54 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  37.82%  |  [0.3716 0.3772 0.3768 0.3742 0.368 ]  |  37.36 (+/- 0.69)  |  42.89  |  2.501  |
03/07/2020 09:18:54 PM - INFO - |  14  |  RIDGE_CLASSIFIER  |  38.56%  |  [0.4036 0.4074 0.402  0.3954 0.4   ]  |  40.17 (+/- 0.79)  |  2.896  |  0.04107  |
```