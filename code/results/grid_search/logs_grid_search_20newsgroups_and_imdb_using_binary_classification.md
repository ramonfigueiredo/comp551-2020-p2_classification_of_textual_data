## Grid search logs: IMDB using Binary Classification and 20 News Groups dataset (removing headers signatures and quoting)


### IMDB using Binary Classification

#### FINAL CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS

| ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
| -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
|  1  |  ADA_BOOST_CLASSIFIER  |  80.24%  |  [0.801  0.803  0.799  0.7968 0.7986]  |  79.97 (+/- 0.43)  |  10.33  |  0.5739  |
|  2  |  BERNOULLI_NB  |  81.52%  |  [0.8384 0.8398 0.8524 0.8404 0.8524]  |  84.47 (+/- 1.27)  |  0.02375  |  0.01999  |
|  3  |  COMPLEMENT_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01545  |  0.008116  |
|  4  |  DECISION_TREE_CLASSIFIER  |  71.35%  |  [0.7128 0.7138 0.7202 0.7098 0.714 ]  |  71.41 (+/- 0.68)  |  23.46  |  0.02994  |
|  5  |  GRADIENT_BOOSTING_CLASSIFIER  |  80.72%  |  [0.805  0.8092 0.8048 0.7986 0.8074]  |  80.50 (+/- 0.72)  |  50.88  |  0.05231  |
|  6  |  K_NEIGHBORS_CLASSIFIER  |  73.29%  |  [0.8144 0.8248 0.8262 0.814  0.815 ]  |  81.89 (+/- 1.09)  |  0.008312  |  26.67  |
|  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  88.44 (+/- 0.98)  |  0.2373  |  0.003927  |
|  8  |  LOGISTIC_REGRESSION  |  88.39%  |  [0.8822 0.8946 0.8848 0.887  0.8852]  |  88.68 (+/- 0.84)  |  1.206  |  0.008219  |
|  9  |  MULTINOMIAL_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01547  |  0.008373  |
|  10  |  NEAREST_CENTROID  |  83.72%  |  [0.8316 0.838  0.8342 0.8392 0.8358]  |  83.58 (+/- 0.54)  |  0.01616  |  0.0117  |
|  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  85.20%  |  [0.8668 0.8766 0.8722 0.8722 0.8652]  |  87.06 (+/- 0.82)  |  0.1836  |  0.004088  |
|  12  |  PERCEPTRON  |  84.43%  |  [0.861  0.868  0.8654 0.8614 0.8536]  |  86.19 (+/- 0.98)  |  0.1075  |  0.004236  |
|  13  |  RANDOM_FOREST_CLASSIFIER  |  84.91%  |  [0.845  0.8524 0.8434 0.8442 0.8432]  |  84.56 (+/- 0.69)  |  35.12  |  1.306  |
|  14  |  RIDGE_CLASSIFIER  |  86.90%  |  [0.8836 0.895  0.8888 0.882  0.8788]  |  88.56 (+/- 1.14)  |  0.4265  |  0.007972  |
 

#### FINAL CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS

| ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
| -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
|  1  |  ADA_BOOST_CLASSIFIER  |  84.60%  |  [0.8398 0.8516 0.8416 0.8366 0.8416]  |  84.22 (+/- 1.00)  |  103.1  |  5.54  |
|  2  |  BERNOULLI_NB  |  81.28%  |  [0.8398 0.8424 0.8514 0.8396 0.8516]  |  84.50 (+/- 1.09)  |  0.02419  |  0.01985  |
|  3  |  COMPLEMENT_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01547  |  0.008101  |
|  4  |  DECISION_TREE_CLASSIFIER  |  73.76%  |  [0.7384 0.731  0.7326 0.7306 0.7188]  |  73.03 (+/- 1.28)  |  9.083  |  0.01224  |
|  5  |  GRADIENT_BOOSTING_CLASSIFIER  |  82.87%  |  [0.8278 0.8286 0.8238 0.8228 0.8284]  |  82.63 (+/- 0.49)  |  101.9  |  0.0661  |
|  6  |  K_NEIGHBORS_CLASSIFIER  |  82.66%  |  [0.8632 0.8744 0.8694 0.864  0.8618]  |  86.66 (+/- 0.94)  |  0.01147  |  26.35  |
|  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  88.44 (+/- 0.98)  |  0.2349  |  0.003867  |
|  8  |  LOGISTIC_REGRESSION  |  87.75%  |  [0.8882 0.897  0.8878 0.8876 0.8818]  |  88.85 (+/- 0.97)  |  1.727  |  0.008252  |
|  9  |  MULTINOMIAL_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.0155  |  0.008379  |
|  10  |  NEAREST_CENTROID  |  84.65%  |  [0.8426 0.8546 0.8426 0.8522 0.8502]  |  84.84 (+/- 0.99)  |  0.01615  |  0.01705  |
|  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  88.06%  |  [0.8872 0.897  0.8884 0.8878 0.8852]  |  88.91 (+/- 0.82)  |  0.8665  |  0.003932  |
|  12  |  PERCEPTRON  |  80.64%  |  [0.8166 0.8264 0.8144 0.81   0.8102]  |  81.55 (+/- 1.20)  |  0.09231  |  0.006239  |
|  13  |  RANDOM_FOREST_CLASSIFIER  |  85.54%  |  [0.8574 0.8582 0.8564 0.8446 0.854 ]  |  85.41 (+/- 0.99)  |  58.66  |  2.501  |
|  14  |  RIDGE_CLASSIFIER  |  86.90%  |  [0.8838 0.8952 0.8892 0.882  0.8788]  |  88.58 (+/- 1.16)  |  0.4877  |  0.008011  |


### 20 News Groups dataset (removing headers signatures and quoting)

#### FINAL CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS

| ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
| -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
|  1  |  ADA_BOOST_CLASSIFIER  |  36.52%  |  [0.39460893 0.39019001 0.39991162 0.39019001 0.40539346]  |  39.61 (+/- 1.18)  |  4.611  |  0.2475  |
|  2  |  BERNOULLI_NB  |  45.84%  |  [0.48033584 0.46133451 0.48563853 0.47547503 0.48938992]  |  47.84 (+/- 1.95)  |  0.05548  |  0.05272  |
|  3  |  COMPLEMENT_NB  |  71.04%  |  [0.77110031 0.75740168 0.77993814 0.7715422  0.76038904]  |  76.81 (+/- 1.64)  |  0.06326  |  0.01035  |
|  4  |  DECISION_TREE_CLASSIFIER  |  42.82%  |  [0.49094123 0.47591692 0.46442775 0.47105612 0.5       ]  |  48.05 (+/- 2.62)  |  10.34  |  0.008546  |
|  5  |  GRADIENT_BOOSTING_CLASSIFIER  |  59.32%  |  [0.65046399 0.61555457 0.63411401 0.64869642 0.64456233]  |  63.87 (+/- 2.58)  |  331.6  |  0.1763  |
|  6  |  K_NEIGHBORS_CLASSIFIER  |  7.20%  |  [0.06937693 0.08174989 0.07909854 0.08263367 0.08311229]  |  7.92 (+/- 1.02)  |  0.002982  |  2.159  |
|  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  75.73 (+/- 2.09)  |  0.8659  |  0.008424  |
|  8  |  LOGISTIC_REGRESSION  |  69.17%  |  [0.74458683 0.72646929 0.74193548 0.74768007 0.72413793]  |  73.70 (+/- 1.94)  |  29.42  |  0.01162  |
|  9  |  MULTINOMIAL_NB  |  65.64%  |  [0.7105612  0.69951392 0.69863014 0.69995581 0.69142352]  |  70.00 (+/- 1.22)  |  0.04385  |  0.01045  |
|  10  |  NEAREST_CENTROID  |  64.91%  |  [0.69774635 0.69244366 0.69818825 0.70393283 0.69849691]  |  69.82 (+/- 0.73)  |  0.01918  |  0.01427  |
|  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  68.26%  |  [0.75607601 0.73000442 0.74370305 0.76226248 0.73607427]  |  74.56 (+/- 2.41)  |  1.037  |  0.02384  |
|  12  |  PERCEPTRON  |  63.34%  |  [0.69818825 0.67565179 0.6955369  0.70349094 0.68302387]  |  69.12 (+/- 2.05)  |  0.5521  |  0.02452  |
|  13  |  RANDOM_FOREST_CLASSIFIER  |  62.16%  |  [0.68360583 0.67034909 0.66725586 0.68272205 0.66534041]  |  67.39 (+/- 1.55)  |  35.03  |  0.6005  |
|  14  |  RIDGE_CLASSIFIER  |  70.67%  |  [0.76623951 0.74900574 0.76889085 0.77330977 0.75464191]  |  76.24 (+/- 1.82)  |  2.19  |  0.0226  |


#### FINAL CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH BEST PARAMETERS

03/07/2020 10:00:05 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 10:00:05 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 10:00:05 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  44.03%  |  [0.46973045 0.45779938 0.46531153 0.47017234 0.46993811]  |  46.66 (+/- 0.95)  |  18.29  |  0.9501  |
03/07/2020 10:00:05 PM - INFO - |  2  |  BERNOULLI_NB  |  62.56%  |  [0.69200177 0.67741935 0.6752099  0.67388422 0.67992927]  |  67.97 (+/- 1.30)  |  0.05404  |  0.05173  |
03/07/2020 10:00:05 PM - INFO - |  3  |  COMPLEMENT_NB  |  71.22%  |  [0.776403   0.75828546 0.78126381 0.77507733 0.76967286]  |  77.21 (+/- 1.57)  |  0.06653  |  0.01054  |
03/07/2020 10:00:05 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  44.57%  |  [0.4874061  0.48608042 0.49138312 0.49359258 0.48099027]  |  48.79 (+/- 0.88)  |  8.941  |  0.0056  |
03/07/2020 10:00:05 PM - INFO - |  5  |  GRADIENT_BOOSTING_CLASSIFIER  |  59.60%  |  [0.6597437  0.62836942 0.64692886 0.66327883 0.64633068]  |  64.89 (+/- 2.46)  |  649.9  |  0.3243  |
03/07/2020 10:00:05 PM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  8.48%  |  [0.12417145 0.11842687 0.12063632 0.12240389 0.12113174]  |  12.14 (+/- 0.38)  |  0.00427  |  1.762  |
03/07/2020 10:00:05 PM - INFO - |  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  75.73 (+/- 2.09)  |  0.8605  |  0.008546  |
03/07/2020 10:00:05 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  69.28%  |  [0.75961114 0.74016792 0.75519222 0.7565179  0.73607427]  |  74.95 (+/- 1.90)  |  32.91  |  0.01159  |
03/07/2020 10:00:05 PM - INFO - |  9  |  MULTINOMIAL_NB  |  68.79%  |  [0.76049492 0.74149359 0.75519222 0.75695979 0.74270557]  |  75.14 (+/- 1.55)  |  0.05682  |  0.01024  |
03/07/2020 10:00:05 PM - INFO - |  10  |  NEAREST_CENTROID  |  66.70%  |  [0.71674768 0.71807335 0.71851525 0.71939903 0.70601238]  |  71.57 (+/- 0.99)  |  0.01737  |  0.01961  |
03/07/2020 10:00:05 PM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  69.69%  |  [0.76623951 0.74856385 0.76314627 0.77021653 0.7484527 ]  |  75.93 (+/- 1.82)  |  9.017  |  0.01138  |
03/07/2020 10:00:05 PM - INFO - |  12  |  PERCEPTRON  |  53.86%  |  [0.63146266 0.61025188 0.58948299 0.60450729 0.60477454]  |  60.81 (+/- 2.71)  |  0.7855  |  0.0223  |
03/07/2020 10:00:05 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  63.64%  |  [0.69067609 0.67255855 0.69067609 0.69730446 0.67241379]  |  68.47 (+/- 2.06)  |  39.54  |  1.178  |
03/07/2020 10:00:05 PM - INFO - |  14  |  RIDGE_CLASSIFIER  |  70.02%  |  [0.76756518 0.74502872 0.76800707 0.776403   0.76083112]  |  76.36 (+/- 2.10)  |  2.787  |  0.02243  |



#### All logs

```
03/07/2020 05:09:06 PM - INFO - 
>>> GRID SEARCH
03/07/2020 05:09:06 PM - INFO - 

03/07/2020 05:09:06 PM - INFO - ################################################################################
03/07/2020 05:09:06 PM - INFO - 1)
03/07/2020 05:09:06 PM - INFO - ********************************************************************************
03/07/2020 05:09:06 PM - INFO - Classifier: ADA_BOOST_CLASSIFIER, Dataset: IMDB_REVIEWS
03/07/2020 05:09:06 PM - INFO - ********************************************************************************
03/07/2020 05:09:13 PM - INFO - 

Performing grid search...

03/07/2020 05:09:13 PM - INFO - Parameters:
03/07/2020 05:09:13 PM - INFO - {'classifier__learning_rate': [0.1, 1], 'classifier__n_estimators': [200, 500]}
03/07/2020 05:25:34 PM - INFO - 	Done in 980.195s
03/07/2020 05:25:34 PM - INFO - 	Best score: 0.842
03/07/2020 05:25:34 PM - INFO - 	Best parameters set:
03/07/2020 05:25:34 PM - INFO - 		classifier__learning_rate: 1
03/07/2020 05:25:34 PM - INFO - 		classifier__n_estimators: 500
03/07/2020 05:25:34 PM - INFO - 

USING ADA_BOOST_CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 05:25:34 PM - INFO - ________________________________________________________________________________
03/07/2020 05:25:34 PM - INFO - Training: 
03/07/2020 05:25:34 PM - INFO - AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
                   n_estimators=50, random_state=None)
03/07/2020 05:25:44 PM - INFO - Train time: 10.333s
03/07/2020 05:25:44 PM - INFO - Test time:  0.574s
03/07/2020 05:25:44 PM - INFO - Accuracy score:   0.802
03/07/2020 05:25:44 PM - INFO - 

===> Classification Report:

03/07/2020 05:25:44 PM - INFO -               precision    recall  f1-score   support

           0       0.82      0.77      0.80     12500
           1       0.78      0.84      0.81     12500

    accuracy                           0.80     25000
   macro avg       0.80      0.80      0.80     25000
weighted avg       0.80      0.80      0.80     25000

03/07/2020 05:25:44 PM - INFO - 

Cross validation:
03/07/2020 05:26:16 PM - INFO - 	accuracy: 5-fold cross validation: [0.801  0.803  0.799  0.7968 0.7986]
03/07/2020 05:26:16 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 79.97 (+/- 0.43)
03/07/2020 05:26:16 PM - INFO - 

USING ADA_BOOST_CLASSIFIER WITH BEST PARAMETERS: {'learning_rate': 1, 'n_estimators': 500}
03/07/2020 05:26:16 PM - INFO - ________________________________________________________________________________
03/07/2020 05:26:16 PM - INFO - Training: 
03/07/2020 05:26:16 PM - INFO - AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1,
                   n_estimators=500, random_state=None)
03/07/2020 05:27:59 PM - INFO - Train time: 103.115s
03/07/2020 05:28:04 PM - INFO - Test time:  5.540s
03/07/2020 05:28:04 PM - INFO - Accuracy score:   0.846
03/07/2020 05:28:04 PM - INFO - 

===> Classification Report:

03/07/2020 05:28:05 PM - INFO -               precision    recall  f1-score   support

           0       0.85      0.83      0.84     12500
           1       0.84      0.86      0.85     12500

    accuracy                           0.85     25000
   macro avg       0.85      0.85      0.85     25000
weighted avg       0.85      0.85      0.85     25000

03/07/2020 05:28:05 PM - INFO - 

Cross validation:
03/07/2020 05:33:25 PM - INFO - 	accuracy: 5-fold cross validation: [0.8398 0.8516 0.8416 0.8366 0.8416]
03/07/2020 05:33:25 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 84.22 (+/- 1.00)
03/07/2020 05:33:25 PM - INFO - It took 1458.616847038269 seconds
03/07/2020 05:33:25 PM - INFO - ********************************************************************************
03/07/2020 05:33:25 PM - INFO - ################################################################################
03/07/2020 05:33:25 PM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 05:33:25 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 05:33:25 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 05:33:25 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  80.24%  |  [0.801  0.803  0.799  0.7968 0.7986]  |  79.97 (+/- 0.43)  |  10.33  |  0.5739  |
03/07/2020 05:33:25 PM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 05:33:25 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 05:33:25 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 05:33:25 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  84.60%  |  [0.8398 0.8516 0.8416 0.8366 0.8416]  |  84.22 (+/- 1.00)  |  103.1  |  5.54  |
03/07/2020 05:33:25 PM - INFO - 

03/07/2020 05:33:25 PM - INFO - ################################################################################
03/07/2020 05:33:25 PM - INFO - 2)
03/07/2020 05:33:25 PM - INFO - ********************************************************************************
03/07/2020 05:33:25 PM - INFO - Classifier: DECISION_TREE_CLASSIFIER, Dataset: IMDB_REVIEWS
03/07/2020 05:33:25 PM - INFO - ********************************************************************************
03/07/2020 05:33:31 PM - INFO - 

Performing grid search...

03/07/2020 05:33:31 PM - INFO - Parameters:
03/07/2020 05:33:31 PM - INFO - {'classifier__criterion': ['entropy', 'gini'], 'classifier__splitter': ['best', 'random'], 'classifier__min_samples_split': [2, 100, 250]}
03/07/2020 05:35:39 PM - INFO - 	Done in 128.046s
03/07/2020 05:35:39 PM - INFO - 	Best score: 0.735
03/07/2020 05:35:39 PM - INFO - 	Best parameters set:
03/07/2020 05:35:39 PM - INFO - 		classifier__criterion: 'entropy'
03/07/2020 05:35:39 PM - INFO - 		classifier__min_samples_split: 250
03/07/2020 05:35:39 PM - INFO - 		classifier__splitter: 'random'
03/07/2020 05:35:39 PM - INFO - 

USING DECISION_TREE_CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 05:35:39 PM - INFO - ________________________________________________________________________________
03/07/2020 05:35:39 PM - INFO - Training: 
03/07/2020 05:35:39 PM - INFO - DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=None, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=None, splitter='best')
03/07/2020 05:36:03 PM - INFO - Train time: 23.459s
03/07/2020 05:36:03 PM - INFO - Test time:  0.030s
03/07/2020 05:36:03 PM - INFO - Accuracy score:   0.714
03/07/2020 05:36:03 PM - INFO - 

===> Classification Report:

03/07/2020 05:36:03 PM - INFO -               precision    recall  f1-score   support

           0       0.71      0.71      0.71     12500
           1       0.71      0.71      0.71     12500

    accuracy                           0.71     25000
   macro avg       0.71      0.71      0.71     25000
weighted avg       0.71      0.71      0.71     25000

03/07/2020 05:36:03 PM - INFO - 

Cross validation:
03/07/2020 05:36:23 PM - INFO - 	accuracy: 5-fold cross validation: [0.7128 0.7138 0.7202 0.7098 0.714 ]
03/07/2020 05:36:23 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 71.41 (+/- 0.68)
03/07/2020 05:36:23 PM - INFO - 

USING DECISION_TREE_CLASSIFIER WITH BEST PARAMETERS: {'criterion': 'entropy', 'min_samples_split': 250, 'splitter': 'random'}
03/07/2020 05:36:23 PM - INFO - ________________________________________________________________________________
03/07/2020 05:36:23 PM - INFO - Training: 
03/07/2020 05:36:23 PM - INFO - DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='entropy',
                       max_depth=None, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=250,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=None, splitter='random')
03/07/2020 05:36:32 PM - INFO - Train time: 9.083s
03/07/2020 05:36:32 PM - INFO - Test time:  0.012s
03/07/2020 05:36:32 PM - INFO - Accuracy score:   0.738
03/07/2020 05:36:32 PM - INFO - 

===> Classification Report:

03/07/2020 05:36:32 PM - INFO -               precision    recall  f1-score   support

           0       0.73      0.76      0.74     12500
           1       0.75      0.71      0.73     12500

    accuracy                           0.74     25000
   macro avg       0.74      0.74      0.74     25000
weighted avg       0.74      0.74      0.74     25000

03/07/2020 05:36:32 PM - INFO - 

Cross validation:
03/07/2020 05:36:41 PM - INFO - 	accuracy: 5-fold cross validation: [0.7384 0.731  0.7326 0.7306 0.7188]
03/07/2020 05:36:41 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 73.03 (+/- 1.28)
03/07/2020 05:36:41 PM - INFO - It took 195.89976716041565 seconds
03/07/2020 05:36:41 PM - INFO - ********************************************************************************
03/07/2020 05:36:41 PM - INFO - ################################################################################
03/07/2020 05:36:41 PM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 05:36:41 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 05:36:41 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 05:36:41 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  80.24%  |  [0.801  0.803  0.799  0.7968 0.7986]  |  79.97 (+/- 0.43)  |  10.33  |  0.5739  |
03/07/2020 05:36:41 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  71.35%  |  [0.7128 0.7138 0.7202 0.7098 0.714 ]  |  71.41 (+/- 0.68)  |  23.46  |  0.02994  |
03/07/2020 05:36:41 PM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 05:36:41 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 05:36:41 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 05:36:41 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  84.60%  |  [0.8398 0.8516 0.8416 0.8366 0.8416]  |  84.22 (+/- 1.00)  |  103.1  |  5.54  |
03/07/2020 05:36:41 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  73.76%  |  [0.7384 0.731  0.7326 0.7306 0.7188]  |  73.03 (+/- 1.28)  |  9.083  |  0.01224  |
03/07/2020 05:36:41 PM - INFO - 

03/07/2020 05:36:41 PM - INFO - ################################################################################
03/07/2020 05:36:41 PM - INFO - 3)
03/07/2020 05:36:41 PM - INFO - ********************************************************************************
03/07/2020 05:36:41 PM - INFO - Classifier: LINEAR_SVC, Dataset: IMDB_REVIEWS
03/07/2020 05:36:41 PM - INFO - ********************************************************************************
03/07/2020 05:36:47 PM - INFO - 

Performing grid search...

03/07/2020 05:36:47 PM - INFO - Parameters:
03/07/2020 05:36:47 PM - INFO - {'classifier__C': [0.01, 1.0], 'classifier__multi_class': ['ovr', 'crammer_singer'], 'classifier__tol': [0.0001, 0.001]}
03/07/2020 05:36:53 PM - INFO - 	Done in 5.718s
03/07/2020 05:36:53 PM - INFO - 	Best score: 0.884
03/07/2020 05:36:53 PM - INFO - 	Best parameters set:
03/07/2020 05:36:53 PM - INFO - 		classifier__C: 1.0
03/07/2020 05:36:53 PM - INFO - 		classifier__multi_class: 'ovr'
03/07/2020 05:36:53 PM - INFO - 		classifier__tol: 0.0001
03/07/2020 05:36:53 PM - INFO - 

USING LINEAR_SVC WITH DEFAULT PARAMETERS
03/07/2020 05:36:53 PM - INFO - ________________________________________________________________________________
03/07/2020 05:36:53 PM - INFO - Training: 
03/07/2020 05:36:53 PM - INFO - LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
          intercept_scaling=1, loss='squared_hinge', max_iter=1000,
          multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
          verbose=0)
03/07/2020 05:36:53 PM - INFO - Train time: 0.237s
03/07/2020 05:36:53 PM - INFO - Test time:  0.004s
03/07/2020 05:36:53 PM - INFO - Accuracy score:   0.871
03/07/2020 05:36:53 PM - INFO - 

===> Classification Report:

03/07/2020 05:36:53 PM - INFO -               precision    recall  f1-score   support

           0       0.87      0.88      0.87     12500
           1       0.88      0.87      0.87     12500

    accuracy                           0.87     25000
   macro avg       0.87      0.87      0.87     25000
weighted avg       0.87      0.87      0.87     25000

03/07/2020 05:36:53 PM - INFO - 

Cross validation:
03/07/2020 05:36:54 PM - INFO - 	accuracy: 5-fold cross validation: [0.8838 0.8932 0.883  0.8836 0.8782]
03/07/2020 05:36:54 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 88.44 (+/- 0.98)
03/07/2020 05:36:54 PM - INFO - 

USING LINEAR_SVC WITH BEST PARAMETERS: {'C': 1.0, 'multi_class': 'ovr', 'tol': 0.0001}
03/07/2020 05:36:54 PM - INFO - ________________________________________________________________________________
03/07/2020 05:36:54 PM - INFO - Training: 
03/07/2020 05:36:54 PM - INFO - LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
          intercept_scaling=1, loss='squared_hinge', max_iter=1000,
          multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
          verbose=0)
03/07/2020 05:36:54 PM - INFO - Train time: 0.235s
03/07/2020 05:36:54 PM - INFO - Test time:  0.004s
03/07/2020 05:36:54 PM - INFO - Accuracy score:   0.871
03/07/2020 05:36:54 PM - INFO - 

===> Classification Report:

03/07/2020 05:36:54 PM - INFO -               precision    recall  f1-score   support

           0       0.87      0.88      0.87     12500
           1       0.88      0.87      0.87     12500

    accuracy                           0.87     25000
   macro avg       0.87      0.87      0.87     25000
weighted avg       0.87      0.87      0.87     25000

03/07/2020 05:36:54 PM - INFO - 

Cross validation:
03/07/2020 05:36:55 PM - INFO - 	accuracy: 5-fold cross validation: [0.8838 0.8932 0.883  0.8836 0.8782]
03/07/2020 05:36:55 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 88.44 (+/- 0.98)
03/07/2020 05:36:55 PM - INFO - It took 13.591684818267822 seconds
03/07/2020 05:36:55 PM - INFO - ********************************************************************************
03/07/2020 05:36:55 PM - INFO - ################################################################################
03/07/2020 05:36:55 PM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 05:36:55 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 05:36:55 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 05:36:55 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  80.24%  |  [0.801  0.803  0.799  0.7968 0.7986]  |  79.97 (+/- 0.43)  |  10.33  |  0.5739  |
03/07/2020 05:36:55 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  71.35%  |  [0.7128 0.7138 0.7202 0.7098 0.714 ]  |  71.41 (+/- 0.68)  |  23.46  |  0.02994  |
03/07/2020 05:36:55 PM - INFO - |  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  88.44 (+/- 0.98)  |  0.2373  |  0.003927  |
03/07/2020 05:36:55 PM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 05:36:55 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 05:36:55 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 05:36:55 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  84.60%  |  [0.8398 0.8516 0.8416 0.8366 0.8416]  |  84.22 (+/- 1.00)  |  103.1  |  5.54  |
03/07/2020 05:36:55 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  73.76%  |  [0.7384 0.731  0.7326 0.7306 0.7188]  |  73.03 (+/- 1.28)  |  9.083  |  0.01224  |
03/07/2020 05:36:55 PM - INFO - |  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  88.44 (+/- 0.98)  |  0.2349  |  0.003867  |
03/07/2020 05:36:55 PM - INFO - 

03/07/2020 05:36:55 PM - INFO - ################################################################################
03/07/2020 05:36:55 PM - INFO - 4)
03/07/2020 05:36:55 PM - INFO - ********************************************************************************
03/07/2020 05:36:55 PM - INFO - Classifier: LOGISTIC_REGRESSION, Dataset: IMDB_REVIEWS
03/07/2020 05:36:55 PM - INFO - ********************************************************************************
03/07/2020 05:37:01 PM - INFO - 

Performing grid search...

03/07/2020 05:37:01 PM - INFO - Parameters:
03/07/2020 05:37:01 PM - INFO - {'classifier__C': [1, 10], 'classifier__tol': [0.001, 0.01]}
03/07/2020 05:37:10 PM - INFO - 	Done in 8.747s
03/07/2020 05:37:10 PM - INFO - 	Best score: 0.888
03/07/2020 05:37:10 PM - INFO - 	Best parameters set:
03/07/2020 05:37:10 PM - INFO - 		classifier__C: 10
03/07/2020 05:37:10 PM - INFO - 		classifier__tol: 0.01
03/07/2020 05:37:10 PM - INFO - 

USING LOGISTIC_REGRESSION WITH DEFAULT PARAMETERS
03/07/2020 05:37:10 PM - INFO - ________________________________________________________________________________
03/07/2020 05:37:10 PM - INFO - Training: 
03/07/2020 05:37:10 PM - INFO - LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)
03/07/2020 05:37:11 PM - INFO - Train time: 1.206s
03/07/2020 05:37:11 PM - INFO - Test time:  0.008s
03/07/2020 05:37:11 PM - INFO - Accuracy score:   0.884
03/07/2020 05:37:11 PM - INFO - 

===> Classification Report:

03/07/2020 05:37:11 PM - INFO -               precision    recall  f1-score   support

           0       0.89      0.88      0.88     12500
           1       0.88      0.89      0.88     12500

    accuracy                           0.88     25000
   macro avg       0.88      0.88      0.88     25000
weighted avg       0.88      0.88      0.88     25000

03/07/2020 05:37:11 PM - INFO - 

Cross validation:
03/07/2020 05:37:12 PM - INFO - 	accuracy: 5-fold cross validation: [0.8822 0.8946 0.8848 0.887  0.8852]
03/07/2020 05:37:12 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 88.68 (+/- 0.84)
03/07/2020 05:37:12 PM - INFO - 

USING LOGISTIC_REGRESSION WITH BEST PARAMETERS: {'C': 10, 'tol': 0.01}
03/07/2020 05:37:12 PM - INFO - ________________________________________________________________________________
03/07/2020 05:37:12 PM - INFO - Training: 
03/07/2020 05:37:12 PM - INFO - LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.01, verbose=0,
                   warm_start=False)
03/07/2020 05:37:14 PM - INFO - Train time: 1.727s
03/07/2020 05:37:14 PM - INFO - Test time:  0.008s
03/07/2020 05:37:14 PM - INFO - Accuracy score:   0.877
03/07/2020 05:37:14 PM - INFO - 

===> Classification Report:

03/07/2020 05:37:14 PM - INFO -               precision    recall  f1-score   support

           0       0.88      0.88      0.88     12500
           1       0.88      0.87      0.88     12500

    accuracy                           0.88     25000
   macro avg       0.88      0.88      0.88     25000
weighted avg       0.88      0.88      0.88     25000

03/07/2020 05:37:14 PM - INFO - 

Cross validation:
03/07/2020 05:37:16 PM - INFO - 	accuracy: 5-fold cross validation: [0.8882 0.897  0.8878 0.8876 0.8818]
03/07/2020 05:37:16 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 88.85 (+/- 0.97)
03/07/2020 05:37:16 PM - INFO - It took 21.47944450378418 seconds
03/07/2020 05:37:16 PM - INFO - ********************************************************************************
03/07/2020 05:37:16 PM - INFO - ################################################################################
03/07/2020 05:37:16 PM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 05:37:16 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 05:37:16 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 05:37:16 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  80.24%  |  [0.801  0.803  0.799  0.7968 0.7986]  |  79.97 (+/- 0.43)  |  10.33  |  0.5739  |
03/07/2020 05:37:16 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  71.35%  |  [0.7128 0.7138 0.7202 0.7098 0.714 ]  |  71.41 (+/- 0.68)  |  23.46  |  0.02994  |
03/07/2020 05:37:16 PM - INFO - |  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  88.44 (+/- 0.98)  |  0.2373  |  0.003927  |
03/07/2020 05:37:16 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  88.39%  |  [0.8822 0.8946 0.8848 0.887  0.8852]  |  88.68 (+/- 0.84)  |  1.206  |  0.008219  |
03/07/2020 05:37:16 PM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 05:37:16 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 05:37:16 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 05:37:16 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  84.60%  |  [0.8398 0.8516 0.8416 0.8366 0.8416]  |  84.22 (+/- 1.00)  |  103.1  |  5.54  |
03/07/2020 05:37:16 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  73.76%  |  [0.7384 0.731  0.7326 0.7306 0.7188]  |  73.03 (+/- 1.28)  |  9.083  |  0.01224  |
03/07/2020 05:37:16 PM - INFO - |  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  88.44 (+/- 0.98)  |  0.2349  |  0.003867  |
03/07/2020 05:37:16 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  87.75%  |  [0.8882 0.897  0.8878 0.8876 0.8818]  |  88.85 (+/- 0.97)  |  1.727  |  0.008252  |
03/07/2020 05:37:16 PM - INFO - 

03/07/2020 05:37:16 PM - INFO - ################################################################################
03/07/2020 05:37:16 PM - INFO - 5)
03/07/2020 05:37:16 PM - INFO - ********************************************************************************
03/07/2020 05:37:16 PM - INFO - Classifier: RANDOM_FOREST_CLASSIFIER, Dataset: IMDB_REVIEWS
03/07/2020 05:37:16 PM - INFO - ********************************************************************************
03/07/2020 05:37:22 PM - INFO - 

Performing grid search...

03/07/2020 05:37:22 PM - INFO - Parameters:
03/07/2020 05:37:22 PM - INFO - {'classifier__min_samples_leaf': [1, 2], 'classifier__min_samples_split': [2, 5], 'classifier__n_estimators': [100, 200]}
03/07/2020 05:43:35 PM - INFO - 	Done in 372.664s
03/07/2020 05:43:35 PM - INFO - 	Best score: 0.854
03/07/2020 05:43:35 PM - INFO - 	Best parameters set:
03/07/2020 05:43:35 PM - INFO - 		classifier__min_samples_leaf: 1
03/07/2020 05:43:35 PM - INFO - 		classifier__min_samples_split: 5
03/07/2020 05:43:35 PM - INFO - 		classifier__n_estimators: 200
03/07/2020 05:43:35 PM - INFO - 

USING RANDOM_FOREST_CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 05:43:35 PM - INFO - ________________________________________________________________________________
03/07/2020 05:43:35 PM - INFO - Training: 
03/07/2020 05:43:35 PM - INFO - RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
03/07/2020 05:44:10 PM - INFO - Train time: 35.122s
03/07/2020 05:44:12 PM - INFO - Test time:  1.306s
03/07/2020 05:44:12 PM - INFO - Accuracy score:   0.849
03/07/2020 05:44:12 PM - INFO - 

===> Classification Report:

03/07/2020 05:44:12 PM - INFO -               precision    recall  f1-score   support

           0       0.84      0.86      0.85     12500
           1       0.86      0.84      0.85     12500

    accuracy                           0.85     25000
   macro avg       0.85      0.85      0.85     25000
weighted avg       0.85      0.85      0.85     25000

03/07/2020 05:44:12 PM - INFO - 

Cross validation:
03/07/2020 05:44:56 PM - INFO - 	accuracy: 5-fold cross validation: [0.845  0.8524 0.8434 0.8442 0.8432]
03/07/2020 05:44:56 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 84.56 (+/- 0.69)
03/07/2020 05:44:56 PM - INFO - 

USING RANDOM_FOREST_CLASSIFIER WITH BEST PARAMETERS: {'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 200}
03/07/2020 05:44:56 PM - INFO - ________________________________________________________________________________
03/07/2020 05:44:56 PM - INFO - Training: 
03/07/2020 05:44:56 PM - INFO - RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=5,
                       min_weight_fraction_leaf=0.0, n_estimators=200,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
03/07/2020 05:45:55 PM - INFO - Train time: 58.659s
03/07/2020 05:45:57 PM - INFO - Test time:  2.501s
03/07/2020 05:45:57 PM - INFO - Accuracy score:   0.855
03/07/2020 05:45:57 PM - INFO - 

===> Classification Report:

03/07/2020 05:45:57 PM - INFO -               precision    recall  f1-score   support

           0       0.85      0.86      0.86     12500
           1       0.86      0.85      0.85     12500

    accuracy                           0.86     25000
   macro avg       0.86      0.86      0.86     25000
weighted avg       0.86      0.86      0.86     25000

03/07/2020 05:45:57 PM - INFO - 

Cross validation:
03/07/2020 05:47:13 PM - INFO - 	accuracy: 5-fold cross validation: [0.8574 0.8582 0.8564 0.8446 0.854 ]
03/07/2020 05:47:13 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 85.41 (+/- 0.99)
03/07/2020 05:47:13 PM - INFO - It took 596.7505943775177 seconds
03/07/2020 05:47:13 PM - INFO - ********************************************************************************
03/07/2020 05:47:13 PM - INFO - ################################################################################
03/07/2020 05:47:13 PM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 05:47:13 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 05:47:13 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 05:47:13 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  80.24%  |  [0.801  0.803  0.799  0.7968 0.7986]  |  79.97 (+/- 0.43)  |  10.33  |  0.5739  |
03/07/2020 05:47:13 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  71.35%  |  [0.7128 0.7138 0.7202 0.7098 0.714 ]  |  71.41 (+/- 0.68)  |  23.46  |  0.02994  |
03/07/2020 05:47:13 PM - INFO - |  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  88.44 (+/- 0.98)  |  0.2373  |  0.003927  |
03/07/2020 05:47:13 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  88.39%  |  [0.8822 0.8946 0.8848 0.887  0.8852]  |  88.68 (+/- 0.84)  |  1.206  |  0.008219  |
03/07/2020 05:47:13 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  84.91%  |  [0.845  0.8524 0.8434 0.8442 0.8432]  |  84.56 (+/- 0.69)  |  35.12  |  1.306  |
03/07/2020 05:47:13 PM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 05:47:13 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 05:47:13 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 05:47:13 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  84.60%  |  [0.8398 0.8516 0.8416 0.8366 0.8416]  |  84.22 (+/- 1.00)  |  103.1  |  5.54  |
03/07/2020 05:47:13 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  73.76%  |  [0.7384 0.731  0.7326 0.7306 0.7188]  |  73.03 (+/- 1.28)  |  9.083  |  0.01224  |
03/07/2020 05:47:13 PM - INFO - |  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  88.44 (+/- 0.98)  |  0.2349  |  0.003867  |
03/07/2020 05:47:13 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  87.75%  |  [0.8882 0.897  0.8878 0.8876 0.8818]  |  88.85 (+/- 0.97)  |  1.727  |  0.008252  |
03/07/2020 05:47:13 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  85.54%  |  [0.8574 0.8582 0.8564 0.8446 0.854 ]  |  85.41 (+/- 0.99)  |  58.66  |  2.501  |
03/07/2020 05:47:13 PM - INFO - 

03/07/2020 05:47:13 PM - INFO - ################################################################################
03/07/2020 05:47:13 PM - INFO - 6)
03/07/2020 05:47:13 PM - INFO - ********************************************************************************
03/07/2020 05:47:13 PM - INFO - Classifier: BERNOULLI_NB, Dataset: IMDB_REVIEWS
03/07/2020 05:47:13 PM - INFO - ********************************************************************************
03/07/2020 05:47:19 PM - INFO - 

Performing grid search...

03/07/2020 05:47:19 PM - INFO - Parameters:
03/07/2020 05:47:19 PM - INFO - {'classifier__alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0], 'classifier__binarize': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0], 'classifier__fit_prior': [False, True]}
03/07/2020 05:47:36 PM - INFO - 	Done in 16.919s
03/07/2020 05:47:36 PM - INFO - 	Best score: 0.845
03/07/2020 05:47:36 PM - INFO - 	Best parameters set:
03/07/2020 05:47:36 PM - INFO - 		classifier__alpha: 0.5
03/07/2020 05:47:36 PM - INFO - 		classifier__binarize: 0.0001
03/07/2020 05:47:36 PM - INFO - 		classifier__fit_prior: False
03/07/2020 05:47:36 PM - INFO - 

USING BERNOULLI_NB WITH DEFAULT PARAMETERS
03/07/2020 05:47:36 PM - INFO - ________________________________________________________________________________
03/07/2020 05:47:36 PM - INFO - Training: 
03/07/2020 05:47:36 PM - INFO - BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
03/07/2020 05:47:36 PM - INFO - Train time: 0.024s
03/07/2020 05:47:36 PM - INFO - Test time:  0.020s
03/07/2020 05:47:36 PM - INFO - Accuracy score:   0.815
03/07/2020 05:47:36 PM - INFO - 

===> Classification Report:

03/07/2020 05:47:36 PM - INFO -               precision    recall  f1-score   support

           0       0.77      0.89      0.83     12500
           1       0.87      0.74      0.80     12500

    accuracy                           0.82     25000
   macro avg       0.82      0.82      0.81     25000
weighted avg       0.82      0.82      0.81     25000

03/07/2020 05:47:36 PM - INFO - 

Cross validation:
03/07/2020 05:47:36 PM - INFO - 	accuracy: 5-fold cross validation: [0.8384 0.8398 0.8524 0.8404 0.8524]
03/07/2020 05:47:36 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 84.47 (+/- 1.27)
03/07/2020 05:47:36 PM - INFO - 

USING BERNOULLI_NB WITH BEST PARAMETERS: {'alpha': 0.5, 'binarize': 0.0001, 'fit_prior': False}
03/07/2020 05:47:36 PM - INFO - ________________________________________________________________________________
03/07/2020 05:47:36 PM - INFO - Training: 
03/07/2020 05:47:36 PM - INFO - BernoulliNB(alpha=0.5, binarize=0.0001, class_prior=None, fit_prior=False)
03/07/2020 05:47:36 PM - INFO - Train time: 0.024s
03/07/2020 05:47:36 PM - INFO - Test time:  0.020s
03/07/2020 05:47:36 PM - INFO - Accuracy score:   0.813
03/07/2020 05:47:36 PM - INFO - 

===> Classification Report:

03/07/2020 05:47:36 PM - INFO -               precision    recall  f1-score   support

           0       0.77      0.89      0.83     12500
           1       0.87      0.74      0.80     12500

    accuracy                           0.81     25000
   macro avg       0.82      0.81      0.81     25000
weighted avg       0.82      0.81      0.81     25000

03/07/2020 05:47:36 PM - INFO - 

Cross validation:
03/07/2020 05:47:37 PM - INFO - 	accuracy: 5-fold cross validation: [0.8398 0.8424 0.8514 0.8396 0.8516]
03/07/2020 05:47:37 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 84.50 (+/- 1.09)
03/07/2020 05:47:37 PM - INFO - It took 23.90962028503418 seconds
03/07/2020 05:47:37 PM - INFO - ********************************************************************************
03/07/2020 05:47:37 PM - INFO - ################################################################################
03/07/2020 05:47:37 PM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 05:47:37 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 05:47:37 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 05:47:37 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  80.24%  |  [0.801  0.803  0.799  0.7968 0.7986]  |  79.97 (+/- 0.43)  |  10.33  |  0.5739  |
03/07/2020 05:47:37 PM - INFO - |  2  |  BERNOULLI_NB  |  81.52%  |  [0.8384 0.8398 0.8524 0.8404 0.8524]  |  84.47 (+/- 1.27)  |  0.02375  |  0.01999  |
03/07/2020 05:47:37 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  71.35%  |  [0.7128 0.7138 0.7202 0.7098 0.714 ]  |  71.41 (+/- 0.68)  |  23.46  |  0.02994  |
03/07/2020 05:47:37 PM - INFO - |  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  88.44 (+/- 0.98)  |  0.2373  |  0.003927  |
03/07/2020 05:47:37 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  88.39%  |  [0.8822 0.8946 0.8848 0.887  0.8852]  |  88.68 (+/- 0.84)  |  1.206  |  0.008219  |
03/07/2020 05:47:37 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  84.91%  |  [0.845  0.8524 0.8434 0.8442 0.8432]  |  84.56 (+/- 0.69)  |  35.12  |  1.306  |
03/07/2020 05:47:37 PM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 05:47:37 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 05:47:37 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 05:47:37 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  84.60%  |  [0.8398 0.8516 0.8416 0.8366 0.8416]  |  84.22 (+/- 1.00)  |  103.1  |  5.54  |
03/07/2020 05:47:37 PM - INFO - |  2  |  BERNOULLI_NB  |  81.28%  |  [0.8398 0.8424 0.8514 0.8396 0.8516]  |  84.50 (+/- 1.09)  |  0.02419  |  0.01985  |
03/07/2020 05:47:37 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  73.76%  |  [0.7384 0.731  0.7326 0.7306 0.7188]  |  73.03 (+/- 1.28)  |  9.083  |  0.01224  |
03/07/2020 05:47:37 PM - INFO - |  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  88.44 (+/- 0.98)  |  0.2349  |  0.003867  |
03/07/2020 05:47:37 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  87.75%  |  [0.8882 0.897  0.8878 0.8876 0.8818]  |  88.85 (+/- 0.97)  |  1.727  |  0.008252  |
03/07/2020 05:47:37 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  85.54%  |  [0.8574 0.8582 0.8564 0.8446 0.854 ]  |  85.41 (+/- 0.99)  |  58.66  |  2.501  |
03/07/2020 05:47:37 PM - INFO - 

03/07/2020 05:47:37 PM - INFO - ################################################################################
03/07/2020 05:47:37 PM - INFO - 7)
03/07/2020 05:47:37 PM - INFO - ********************************************************************************
03/07/2020 05:47:37 PM - INFO - Classifier: COMPLEMENT_NB, Dataset: IMDB_REVIEWS
03/07/2020 05:47:37 PM - INFO - ********************************************************************************
03/07/2020 05:47:43 PM - INFO - 

Performing grid search...

03/07/2020 05:47:43 PM - INFO - Parameters:
03/07/2020 05:47:43 PM - INFO - {'classifier__alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0], 'classifier__fit_prior': [False, True], 'classifier__norm': [False, True]}
03/07/2020 05:47:46 PM - INFO - 	Done in 3.234s
03/07/2020 05:47:46 PM - INFO - 	Best score: 0.865
03/07/2020 05:47:46 PM - INFO - 	Best parameters set:
03/07/2020 05:47:46 PM - INFO - 		classifier__alpha: 1.0
03/07/2020 05:47:46 PM - INFO - 		classifier__fit_prior: False
03/07/2020 05:47:46 PM - INFO - 		classifier__norm: False
03/07/2020 05:47:46 PM - INFO - 

USING COMPLEMENT_NB WITH DEFAULT PARAMETERS
03/07/2020 05:47:46 PM - INFO - ________________________________________________________________________________
03/07/2020 05:47:46 PM - INFO - Training: 
03/07/2020 05:47:46 PM - INFO - ComplementNB(alpha=1.0, class_prior=None, fit_prior=True, norm=False)
03/07/2020 05:47:46 PM - INFO - Train time: 0.015s
03/07/2020 05:47:46 PM - INFO - Test time:  0.008s
03/07/2020 05:47:46 PM - INFO - Accuracy score:   0.839
03/07/2020 05:47:46 PM - INFO - 

===> Classification Report:

03/07/2020 05:47:46 PM - INFO -               precision    recall  f1-score   support

           0       0.82      0.88      0.85     12500
           1       0.87      0.80      0.83     12500

    accuracy                           0.84     25000
   macro avg       0.84      0.84      0.84     25000
weighted avg       0.84      0.84      0.84     25000

03/07/2020 05:47:46 PM - INFO - 

Cross validation:
03/07/2020 05:47:47 PM - INFO - 	accuracy: 5-fold cross validation: [0.8564 0.8678 0.8682 0.8678 0.8672]
03/07/2020 05:47:47 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 86.55 (+/- 0.91)
03/07/2020 05:47:47 PM - INFO - 

USING COMPLEMENT_NB WITH BEST PARAMETERS: {'alpha': 1.0, 'fit_prior': False, 'norm': False}
03/07/2020 05:47:47 PM - INFO - ________________________________________________________________________________
03/07/2020 05:47:47 PM - INFO - Training: 
03/07/2020 05:47:47 PM - INFO - ComplementNB(alpha=1.0, class_prior=None, fit_prior=False, norm=False)
03/07/2020 05:47:47 PM - INFO - Train time: 0.015s
03/07/2020 05:47:47 PM - INFO - Test time:  0.008s
03/07/2020 05:47:47 PM - INFO - Accuracy score:   0.839
03/07/2020 05:47:47 PM - INFO - 

===> Classification Report:

03/07/2020 05:47:47 PM - INFO -               precision    recall  f1-score   support

           0       0.82      0.88      0.85     12500
           1       0.87      0.80      0.83     12500

    accuracy                           0.84     25000
   macro avg       0.84      0.84      0.84     25000
weighted avg       0.84      0.84      0.84     25000

03/07/2020 05:47:47 PM - INFO - 

Cross validation:
03/07/2020 05:47:47 PM - INFO - 	accuracy: 5-fold cross validation: [0.8564 0.8678 0.8682 0.8678 0.8672]
03/07/2020 05:47:47 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 86.55 (+/- 0.91)
03/07/2020 05:47:47 PM - INFO - It took 10.144769191741943 seconds
03/07/2020 05:47:47 PM - INFO - ********************************************************************************
03/07/2020 05:47:47 PM - INFO - ################################################################################
03/07/2020 05:47:47 PM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 05:47:47 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 05:47:47 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 05:47:47 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  80.24%  |  [0.801  0.803  0.799  0.7968 0.7986]  |  79.97 (+/- 0.43)  |  10.33  |  0.5739  |
03/07/2020 05:47:47 PM - INFO - |  2  |  BERNOULLI_NB  |  81.52%  |  [0.8384 0.8398 0.8524 0.8404 0.8524]  |  84.47 (+/- 1.27)  |  0.02375  |  0.01999  |
03/07/2020 05:47:47 PM - INFO - |  3  |  COMPLEMENT_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01545  |  0.008116  |
03/07/2020 05:47:47 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  71.35%  |  [0.7128 0.7138 0.7202 0.7098 0.714 ]  |  71.41 (+/- 0.68)  |  23.46  |  0.02994  |
03/07/2020 05:47:47 PM - INFO - |  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  88.44 (+/- 0.98)  |  0.2373  |  0.003927  |
03/07/2020 05:47:47 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  88.39%  |  [0.8822 0.8946 0.8848 0.887  0.8852]  |  88.68 (+/- 0.84)  |  1.206  |  0.008219  |
03/07/2020 05:47:47 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  84.91%  |  [0.845  0.8524 0.8434 0.8442 0.8432]  |  84.56 (+/- 0.69)  |  35.12  |  1.306  |
03/07/2020 05:47:47 PM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 05:47:47 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 05:47:47 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 05:47:47 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  84.60%  |  [0.8398 0.8516 0.8416 0.8366 0.8416]  |  84.22 (+/- 1.00)  |  103.1  |  5.54  |
03/07/2020 05:47:47 PM - INFO - |  2  |  BERNOULLI_NB  |  81.28%  |  [0.8398 0.8424 0.8514 0.8396 0.8516]  |  84.50 (+/- 1.09)  |  0.02419  |  0.01985  |
03/07/2020 05:47:47 PM - INFO - |  3  |  COMPLEMENT_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01547  |  0.008101  |
03/07/2020 05:47:47 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  73.76%  |  [0.7384 0.731  0.7326 0.7306 0.7188]  |  73.03 (+/- 1.28)  |  9.083  |  0.01224  |
03/07/2020 05:47:47 PM - INFO - |  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  88.44 (+/- 0.98)  |  0.2349  |  0.003867  |
03/07/2020 05:47:47 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  87.75%  |  [0.8882 0.897  0.8878 0.8876 0.8818]  |  88.85 (+/- 0.97)  |  1.727  |  0.008252  |
03/07/2020 05:47:47 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  85.54%  |  [0.8574 0.8582 0.8564 0.8446 0.854 ]  |  85.41 (+/- 0.99)  |  58.66  |  2.501  |
03/07/2020 05:47:47 PM - INFO - 

03/07/2020 05:47:47 PM - INFO - ################################################################################
03/07/2020 05:47:47 PM - INFO - 8)
03/07/2020 05:47:47 PM - INFO - ********************************************************************************
03/07/2020 05:47:47 PM - INFO - Classifier: MULTINOMIAL_NB, Dataset: IMDB_REVIEWS
03/07/2020 05:47:47 PM - INFO - ********************************************************************************
03/07/2020 05:47:53 PM - INFO - 

Performing grid search...

03/07/2020 05:47:53 PM - INFO - Parameters:
03/07/2020 05:47:53 PM - INFO - {'classifier__alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0], 'classifier__fit_prior': [False, True]}
03/07/2020 05:47:55 PM - INFO - 	Done in 1.873s
03/07/2020 05:47:55 PM - INFO - 	Best score: 0.865
03/07/2020 05:47:55 PM - INFO - 	Best parameters set:
03/07/2020 05:47:55 PM - INFO - 		classifier__alpha: 1.0
03/07/2020 05:47:55 PM - INFO - 		classifier__fit_prior: False
03/07/2020 05:47:55 PM - INFO - 

USING MULTINOMIAL_NB WITH DEFAULT PARAMETERS
03/07/2020 05:47:55 PM - INFO - ________________________________________________________________________________
03/07/2020 05:47:55 PM - INFO - Training: 
03/07/2020 05:47:55 PM - INFO - MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
03/07/2020 05:47:55 PM - INFO - Train time: 0.015s
03/07/2020 05:47:55 PM - INFO - Test time:  0.008s
03/07/2020 05:47:55 PM - INFO - Accuracy score:   0.839
03/07/2020 05:47:55 PM - INFO - 

===> Classification Report:

03/07/2020 05:47:55 PM - INFO -               precision    recall  f1-score   support

           0       0.82      0.88      0.85     12500
           1       0.87      0.80      0.83     12500

    accuracy                           0.84     25000
   macro avg       0.84      0.84      0.84     25000
weighted avg       0.84      0.84      0.84     25000

03/07/2020 05:47:55 PM - INFO - 

Cross validation:
03/07/2020 05:47:55 PM - INFO - 	accuracy: 5-fold cross validation: [0.8564 0.8678 0.8682 0.8678 0.8672]
03/07/2020 05:47:55 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 86.55 (+/- 0.91)
03/07/2020 05:47:55 PM - INFO - 

USING MULTINOMIAL_NB WITH BEST PARAMETERS: {'alpha': 1.0, 'fit_prior': False}
03/07/2020 05:47:55 PM - INFO - ________________________________________________________________________________
03/07/2020 05:47:55 PM - INFO - Training: 
03/07/2020 05:47:55 PM - INFO - MultinomialNB(alpha=1.0, class_prior=None, fit_prior=False)
03/07/2020 05:47:55 PM - INFO - Train time: 0.016s
03/07/2020 05:47:55 PM - INFO - Test time:  0.008s
03/07/2020 05:47:55 PM - INFO - Accuracy score:   0.839
03/07/2020 05:47:55 PM - INFO - 

===> Classification Report:

03/07/2020 05:47:55 PM - INFO -               precision    recall  f1-score   support

           0       0.82      0.88      0.85     12500
           1       0.87      0.80      0.83     12500

    accuracy                           0.84     25000
   macro avg       0.84      0.84      0.84     25000
weighted avg       0.84      0.84      0.84     25000

03/07/2020 05:47:55 PM - INFO - 

Cross validation:
03/07/2020 05:47:56 PM - INFO - 	accuracy: 5-fold cross validation: [0.8564 0.8678 0.8682 0.8678 0.8672]
03/07/2020 05:47:56 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 86.55 (+/- 0.91)
03/07/2020 05:47:56 PM - INFO - It took 8.7919602394104 seconds
03/07/2020 05:47:56 PM - INFO - ********************************************************************************
03/07/2020 05:47:56 PM - INFO - ################################################################################
03/07/2020 05:47:56 PM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 05:47:56 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 05:47:56 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 05:47:56 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  80.24%  |  [0.801  0.803  0.799  0.7968 0.7986]  |  79.97 (+/- 0.43)  |  10.33  |  0.5739  |
03/07/2020 05:47:56 PM - INFO - |  2  |  BERNOULLI_NB  |  81.52%  |  [0.8384 0.8398 0.8524 0.8404 0.8524]  |  84.47 (+/- 1.27)  |  0.02375  |  0.01999  |
03/07/2020 05:47:56 PM - INFO - |  3  |  COMPLEMENT_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01545  |  0.008116  |
03/07/2020 05:47:56 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  71.35%  |  [0.7128 0.7138 0.7202 0.7098 0.714 ]  |  71.41 (+/- 0.68)  |  23.46  |  0.02994  |
03/07/2020 05:47:56 PM - INFO - |  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  88.44 (+/- 0.98)  |  0.2373  |  0.003927  |
03/07/2020 05:47:56 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  88.39%  |  [0.8822 0.8946 0.8848 0.887  0.8852]  |  88.68 (+/- 0.84)  |  1.206  |  0.008219  |
03/07/2020 05:47:56 PM - INFO - |  9  |  MULTINOMIAL_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01547  |  0.008373  |
03/07/2020 05:47:56 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  84.91%  |  [0.845  0.8524 0.8434 0.8442 0.8432]  |  84.56 (+/- 0.69)  |  35.12  |  1.306  |
03/07/2020 05:47:56 PM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 05:47:56 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 05:47:56 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 05:47:56 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  84.60%  |  [0.8398 0.8516 0.8416 0.8366 0.8416]  |  84.22 (+/- 1.00)  |  103.1  |  5.54  |
03/07/2020 05:47:56 PM - INFO - |  2  |  BERNOULLI_NB  |  81.28%  |  [0.8398 0.8424 0.8514 0.8396 0.8516]  |  84.50 (+/- 1.09)  |  0.02419  |  0.01985  |
03/07/2020 05:47:56 PM - INFO - |  3  |  COMPLEMENT_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01547  |  0.008101  |
03/07/2020 05:47:56 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  73.76%  |  [0.7384 0.731  0.7326 0.7306 0.7188]  |  73.03 (+/- 1.28)  |  9.083  |  0.01224  |
03/07/2020 05:47:56 PM - INFO - |  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  88.44 (+/- 0.98)  |  0.2349  |  0.003867  |
03/07/2020 05:47:56 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  87.75%  |  [0.8882 0.897  0.8878 0.8876 0.8818]  |  88.85 (+/- 0.97)  |  1.727  |  0.008252  |
03/07/2020 05:47:56 PM - INFO - |  9  |  MULTINOMIAL_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.0155  |  0.008379  |
03/07/2020 05:47:56 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  85.54%  |  [0.8574 0.8582 0.8564 0.8446 0.854 ]  |  85.41 (+/- 0.99)  |  58.66  |  2.501  |
03/07/2020 05:47:56 PM - INFO - 

03/07/2020 05:47:56 PM - INFO - ################################################################################
03/07/2020 05:47:56 PM - INFO - 9)
03/07/2020 05:47:56 PM - INFO - ********************************************************************************
03/07/2020 05:47:56 PM - INFO - Classifier: NEAREST_CENTROID, Dataset: IMDB_REVIEWS
03/07/2020 05:47:56 PM - INFO - ********************************************************************************
03/07/2020 05:48:02 PM - INFO - 

Performing grid search...

03/07/2020 05:48:02 PM - INFO - Parameters:
03/07/2020 05:48:02 PM - INFO - {'classifier__metric': ['euclidean', 'cosine']}
03/07/2020 05:48:02 PM - INFO - 	Done in 0.325s
03/07/2020 05:48:02 PM - INFO - 	Best score: 0.848
03/07/2020 05:48:02 PM - INFO - 	Best parameters set:
03/07/2020 05:48:02 PM - INFO - 		classifier__metric: 'cosine'
03/07/2020 05:48:02 PM - INFO - 

USING NEAREST_CENTROID WITH DEFAULT PARAMETERS
03/07/2020 05:48:02 PM - INFO - ________________________________________________________________________________
03/07/2020 05:48:02 PM - INFO - Training: 
03/07/2020 05:48:02 PM - INFO - NearestCentroid(metric='euclidean', shrink_threshold=None)
03/07/2020 05:48:02 PM - INFO - Train time: 0.016s
03/07/2020 05:48:02 PM - INFO - Test time:  0.012s
03/07/2020 05:48:03 PM - INFO - Accuracy score:   0.837
03/07/2020 05:48:03 PM - INFO - 

===> Classification Report:

03/07/2020 05:48:03 PM - INFO -               precision    recall  f1-score   support

           0       0.86      0.81      0.83     12500
           1       0.82      0.87      0.84     12500

    accuracy                           0.84     25000
   macro avg       0.84      0.84      0.84     25000
weighted avg       0.84      0.84      0.84     25000

03/07/2020 05:48:03 PM - INFO - 

Cross validation:
03/07/2020 05:48:03 PM - INFO - 	accuracy: 5-fold cross validation: [0.8316 0.838  0.8342 0.8392 0.8358]
03/07/2020 05:48:03 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 83.58 (+/- 0.54)
03/07/2020 05:48:03 PM - INFO - 

USING NEAREST_CENTROID WITH BEST PARAMETERS: {'metric': 'cosine'}
03/07/2020 05:48:03 PM - INFO - ________________________________________________________________________________
03/07/2020 05:48:03 PM - INFO - Training: 
03/07/2020 05:48:03 PM - INFO - NearestCentroid(metric='cosine', shrink_threshold=None)
03/07/2020 05:48:03 PM - INFO - Train time: 0.016s
03/07/2020 05:48:03 PM - INFO - Test time:  0.017s
03/07/2020 05:48:03 PM - INFO - Accuracy score:   0.847
03/07/2020 05:48:03 PM - INFO - 

===> Classification Report:

03/07/2020 05:48:03 PM - INFO -               precision    recall  f1-score   support

           0       0.86      0.83      0.84     12500
           1       0.83      0.87      0.85     12500

    accuracy                           0.85     25000
   macro avg       0.85      0.85      0.85     25000
weighted avg       0.85      0.85      0.85     25000

03/07/2020 05:48:03 PM - INFO - 

Cross validation:
03/07/2020 05:48:03 PM - INFO - 	accuracy: 5-fold cross validation: [0.8426 0.8546 0.8426 0.8522 0.8502]
03/07/2020 05:48:03 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 84.84 (+/- 0.99)
03/07/2020 05:48:03 PM - INFO - It took 7.381277561187744 seconds
03/07/2020 05:48:03 PM - INFO - ********************************************************************************
03/07/2020 05:48:03 PM - INFO - ################################################################################
03/07/2020 05:48:03 PM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 05:48:03 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 05:48:03 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 05:48:03 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  80.24%  |  [0.801  0.803  0.799  0.7968 0.7986]  |  79.97 (+/- 0.43)  |  10.33  |  0.5739  |
03/07/2020 05:48:03 PM - INFO - |  2  |  BERNOULLI_NB  |  81.52%  |  [0.8384 0.8398 0.8524 0.8404 0.8524]  |  84.47 (+/- 1.27)  |  0.02375  |  0.01999  |
03/07/2020 05:48:03 PM - INFO - |  3  |  COMPLEMENT_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01545  |  0.008116  |
03/07/2020 05:48:03 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  71.35%  |  [0.7128 0.7138 0.7202 0.7098 0.714 ]  |  71.41 (+/- 0.68)  |  23.46  |  0.02994  |
03/07/2020 05:48:03 PM - INFO - |  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  88.44 (+/- 0.98)  |  0.2373  |  0.003927  |
03/07/2020 05:48:03 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  88.39%  |  [0.8822 0.8946 0.8848 0.887  0.8852]  |  88.68 (+/- 0.84)  |  1.206  |  0.008219  |
03/07/2020 05:48:03 PM - INFO - |  9  |  MULTINOMIAL_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01547  |  0.008373  |
03/07/2020 05:48:03 PM - INFO - |  10  |  NEAREST_CENTROID  |  83.72%  |  [0.8316 0.838  0.8342 0.8392 0.8358]  |  83.58 (+/- 0.54)  |  0.01616  |  0.0117  |
03/07/2020 05:48:03 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  84.91%  |  [0.845  0.8524 0.8434 0.8442 0.8432]  |  84.56 (+/- 0.69)  |  35.12  |  1.306  |
03/07/2020 05:48:03 PM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 05:48:03 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 05:48:03 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 05:48:03 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  84.60%  |  [0.8398 0.8516 0.8416 0.8366 0.8416]  |  84.22 (+/- 1.00)  |  103.1  |  5.54  |
03/07/2020 05:48:03 PM - INFO - |  2  |  BERNOULLI_NB  |  81.28%  |  [0.8398 0.8424 0.8514 0.8396 0.8516]  |  84.50 (+/- 1.09)  |  0.02419  |  0.01985  |
03/07/2020 05:48:03 PM - INFO - |  3  |  COMPLEMENT_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01547  |  0.008101  |
03/07/2020 05:48:03 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  73.76%  |  [0.7384 0.731  0.7326 0.7306 0.7188]  |  73.03 (+/- 1.28)  |  9.083  |  0.01224  |
03/07/2020 05:48:03 PM - INFO - |  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  88.44 (+/- 0.98)  |  0.2349  |  0.003867  |
03/07/2020 05:48:03 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  87.75%  |  [0.8882 0.897  0.8878 0.8876 0.8818]  |  88.85 (+/- 0.97)  |  1.727  |  0.008252  |
03/07/2020 05:48:03 PM - INFO - |  9  |  MULTINOMIAL_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.0155  |  0.008379  |
03/07/2020 05:48:03 PM - INFO - |  10  |  NEAREST_CENTROID  |  84.65%  |  [0.8426 0.8546 0.8426 0.8522 0.8502]  |  84.84 (+/- 0.99)  |  0.01615  |  0.01705  |
03/07/2020 05:48:03 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  85.54%  |  [0.8574 0.8582 0.8564 0.8446 0.854 ]  |  85.41 (+/- 0.99)  |  58.66  |  2.501  |
03/07/2020 05:48:03 PM - INFO - 

03/07/2020 05:48:03 PM - INFO - ################################################################################
03/07/2020 05:48:03 PM - INFO - 10)
03/07/2020 05:48:03 PM - INFO - ********************************************************************************
03/07/2020 05:48:03 PM - INFO - Classifier: PASSIVE_AGGRESSIVE_CLASSIFIER, Dataset: IMDB_REVIEWS
03/07/2020 05:48:03 PM - INFO - ********************************************************************************
03/07/2020 05:48:09 PM - INFO - 

Performing grid search...

03/07/2020 05:48:09 PM - INFO - Parameters:
03/07/2020 05:48:09 PM - INFO - {'classifier__C': [0.01, 1.0], 'classifier__early_stopping': [False, True], 'classifier__tol': [0.0001, 0.001, 0.01], 'classifier__validation_fraction': [0.0001, 0.01]}
03/07/2020 05:48:21 PM - INFO - 	Done in 11.575s
03/07/2020 05:48:21 PM - INFO - 	Best score: 0.889
03/07/2020 05:48:21 PM - INFO - 	Best parameters set:
03/07/2020 05:48:21 PM - INFO - 		classifier__C: 0.01
03/07/2020 05:48:21 PM - INFO - 		classifier__early_stopping: False
03/07/2020 05:48:21 PM - INFO - 		classifier__tol: 0.001
03/07/2020 05:48:21 PM - INFO - 		classifier__validation_fraction: 0.01
03/07/2020 05:48:21 PM - INFO - 

USING PASSIVE_AGGRESSIVE_CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 05:48:21 PM - INFO - ________________________________________________________________________________
03/07/2020 05:48:21 PM - INFO - Training: 
03/07/2020 05:48:21 PM - INFO - PassiveAggressiveClassifier(C=1.0, average=False, class_weight=None,
                            early_stopping=False, fit_intercept=True,
                            loss='hinge', max_iter=1000, n_iter_no_change=5,
                            n_jobs=None, random_state=None, shuffle=True,
                            tol=0.001, validation_fraction=0.1, verbose=0,
                            warm_start=False)
03/07/2020 05:48:21 PM - INFO - Train time: 0.184s
03/07/2020 05:48:21 PM - INFO - Test time:  0.004s
03/07/2020 05:48:21 PM - INFO - Accuracy score:   0.852
03/07/2020 05:48:21 PM - INFO - 

===> Classification Report:

03/07/2020 05:48:21 PM - INFO -               precision    recall  f1-score   support

           0       0.84      0.86      0.85     12500
           1       0.86      0.84      0.85     12500

    accuracy                           0.85     25000
   macro avg       0.85      0.85      0.85     25000
weighted avg       0.85      0.85      0.85     25000

03/07/2020 05:48:21 PM - INFO - 

Cross validation:
03/07/2020 05:48:22 PM - INFO - 	accuracy: 5-fold cross validation: [0.8668 0.8766 0.8722 0.8722 0.8652]
03/07/2020 05:48:22 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 87.06 (+/- 0.82)
03/07/2020 05:48:22 PM - INFO - 

USING PASSIVE_AGGRESSIVE_CLASSIFIER WITH BEST PARAMETERS: {'C': 0.01, 'early_stopping': False, 'tol': 0.001, 'validation_fraction': 0.01}
03/07/2020 05:48:22 PM - INFO - ________________________________________________________________________________
03/07/2020 05:48:22 PM - INFO - Training: 
03/07/2020 05:48:22 PM - INFO - PassiveAggressiveClassifier(C=0.01, average=False, class_weight=None,
                            early_stopping=False, fit_intercept=True,
                            loss='hinge', max_iter=1000, n_iter_no_change=5,
                            n_jobs=None, random_state=None, shuffle=True,
                            tol=0.001, validation_fraction=0.01, verbose=0,
                            warm_start=False)
03/07/2020 05:48:22 PM - INFO - Train time: 0.866s
03/07/2020 05:48:22 PM - INFO - Test time:  0.004s
03/07/2020 05:48:22 PM - INFO - Accuracy score:   0.881
03/07/2020 05:48:22 PM - INFO - 

===> Classification Report:

03/07/2020 05:48:22 PM - INFO -               precision    recall  f1-score   support

           0       0.88      0.88      0.88     12500
           1       0.88      0.88      0.88     12500

    accuracy                           0.88     25000
   macro avg       0.88      0.88      0.88     25000
weighted avg       0.88      0.88      0.88     25000

03/07/2020 05:48:22 PM - INFO - 

Cross validation:
03/07/2020 05:48:23 PM - INFO - 	accuracy: 5-fold cross validation: [0.8872 0.897  0.8884 0.8878 0.8852]
03/07/2020 05:48:23 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 88.91 (+/- 0.82)
03/07/2020 05:48:23 PM - INFO - It took 20.460453033447266 seconds
03/07/2020 05:48:23 PM - INFO - ********************************************************************************
03/07/2020 05:48:23 PM - INFO - ################################################################################
03/07/2020 05:48:23 PM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 05:48:23 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 05:48:23 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 05:48:23 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  80.24%  |  [0.801  0.803  0.799  0.7968 0.7986]  |  79.97 (+/- 0.43)  |  10.33  |  0.5739  |
03/07/2020 05:48:23 PM - INFO - |  2  |  BERNOULLI_NB  |  81.52%  |  [0.8384 0.8398 0.8524 0.8404 0.8524]  |  84.47 (+/- 1.27)  |  0.02375  |  0.01999  |
03/07/2020 05:48:23 PM - INFO - |  3  |  COMPLEMENT_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01545  |  0.008116  |
03/07/2020 05:48:23 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  71.35%  |  [0.7128 0.7138 0.7202 0.7098 0.714 ]  |  71.41 (+/- 0.68)  |  23.46  |  0.02994  |
03/07/2020 05:48:23 PM - INFO - |  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  88.44 (+/- 0.98)  |  0.2373  |  0.003927  |
03/07/2020 05:48:23 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  88.39%  |  [0.8822 0.8946 0.8848 0.887  0.8852]  |  88.68 (+/- 0.84)  |  1.206  |  0.008219  |
03/07/2020 05:48:23 PM - INFO - |  9  |  MULTINOMIAL_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01547  |  0.008373  |
03/07/2020 05:48:23 PM - INFO - |  10  |  NEAREST_CENTROID  |  83.72%  |  [0.8316 0.838  0.8342 0.8392 0.8358]  |  83.58 (+/- 0.54)  |  0.01616  |  0.0117  |
03/07/2020 05:48:23 PM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  85.20%  |  [0.8668 0.8766 0.8722 0.8722 0.8652]  |  87.06 (+/- 0.82)  |  0.1836  |  0.004088  |
03/07/2020 05:48:23 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  84.91%  |  [0.845  0.8524 0.8434 0.8442 0.8432]  |  84.56 (+/- 0.69)  |  35.12  |  1.306  |
03/07/2020 05:48:23 PM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 05:48:23 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 05:48:23 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 05:48:23 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  84.60%  |  [0.8398 0.8516 0.8416 0.8366 0.8416]  |  84.22 (+/- 1.00)  |  103.1  |  5.54  |
03/07/2020 05:48:23 PM - INFO - |  2  |  BERNOULLI_NB  |  81.28%  |  [0.8398 0.8424 0.8514 0.8396 0.8516]  |  84.50 (+/- 1.09)  |  0.02419  |  0.01985  |
03/07/2020 05:48:23 PM - INFO - |  3  |  COMPLEMENT_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01547  |  0.008101  |
03/07/2020 05:48:23 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  73.76%  |  [0.7384 0.731  0.7326 0.7306 0.7188]  |  73.03 (+/- 1.28)  |  9.083  |  0.01224  |
03/07/2020 05:48:23 PM - INFO - |  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  88.44 (+/- 0.98)  |  0.2349  |  0.003867  |
03/07/2020 05:48:23 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  87.75%  |  [0.8882 0.897  0.8878 0.8876 0.8818]  |  88.85 (+/- 0.97)  |  1.727  |  0.008252  |
03/07/2020 05:48:23 PM - INFO - |  9  |  MULTINOMIAL_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.0155  |  0.008379  |
03/07/2020 05:48:23 PM - INFO - |  10  |  NEAREST_CENTROID  |  84.65%  |  [0.8426 0.8546 0.8426 0.8522 0.8502]  |  84.84 (+/- 0.99)  |  0.01615  |  0.01705  |
03/07/2020 05:48:23 PM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  88.06%  |  [0.8872 0.897  0.8884 0.8878 0.8852]  |  88.91 (+/- 0.82)  |  0.8665  |  0.003932  |
03/07/2020 05:48:23 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  85.54%  |  [0.8574 0.8582 0.8564 0.8446 0.854 ]  |  85.41 (+/- 0.99)  |  58.66  |  2.501  |
03/07/2020 05:48:23 PM - INFO - 

03/07/2020 05:48:23 PM - INFO - ################################################################################
03/07/2020 05:48:23 PM - INFO - 11)
03/07/2020 05:48:23 PM - INFO - ********************************************************************************
03/07/2020 05:48:23 PM - INFO - Classifier: K_NEIGHBORS_CLASSIFIER, Dataset: IMDB_REVIEWS
03/07/2020 05:48:23 PM - INFO - ********************************************************************************
03/07/2020 05:48:30 PM - INFO - 

Performing grid search...

03/07/2020 05:48:30 PM - INFO - Parameters:
03/07/2020 05:48:30 PM - INFO - {'classifier__leaf_size': [5, 30], 'classifier__metric': ['euclidean', 'minkowski'], 'classifier__n_neighbors': [3, 50], 'classifier__weights': ['uniform', 'distance']}
03/07/2020 05:49:44 PM - INFO - 	Done in 73.530s
03/07/2020 05:49:44 PM - INFO - 	Best score: 0.867
03/07/2020 05:49:44 PM - INFO - 	Best parameters set:
03/07/2020 05:49:44 PM - INFO - 		classifier__leaf_size: 5
03/07/2020 05:49:44 PM - INFO - 		classifier__metric: 'euclidean'
03/07/2020 05:49:44 PM - INFO - 		classifier__n_neighbors: 50
03/07/2020 05:49:44 PM - INFO - 		classifier__weights: 'distance'
03/07/2020 05:49:44 PM - INFO - 

USING K_NEIGHBORS_CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 05:49:44 PM - INFO - ________________________________________________________________________________
03/07/2020 05:49:44 PM - INFO - Training: 
03/07/2020 05:49:44 PM - INFO - KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                     weights='uniform')
03/07/2020 05:49:44 PM - INFO - Train time: 0.008s
03/07/2020 05:50:10 PM - INFO - Test time:  26.672s
03/07/2020 05:50:10 PM - INFO - Accuracy score:   0.733
03/07/2020 05:50:10 PM - INFO - 

===> Classification Report:

03/07/2020 05:50:10 PM - INFO -               precision    recall  f1-score   support

           0       0.73      0.75      0.74     12500
           1       0.74      0.72      0.73     12500

    accuracy                           0.73     25000
   macro avg       0.73      0.73      0.73     25000
weighted avg       0.73      0.73      0.73     25000

03/07/2020 05:50:10 PM - INFO - 

Cross validation:
03/07/2020 05:50:16 PM - INFO - 	accuracy: 5-fold cross validation: [0.8144 0.8248 0.8262 0.814  0.815 ]
03/07/2020 05:50:16 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 81.89 (+/- 1.09)
03/07/2020 05:50:16 PM - INFO - 

USING K_NEIGHBORS_CLASSIFIER WITH BEST PARAMETERS: {'leaf_size': 5, 'metric': 'euclidean', 'n_neighbors': 50, 'weights': 'distance'}
03/07/2020 05:50:16 PM - INFO - ________________________________________________________________________________
03/07/2020 05:50:16 PM - INFO - Training: 
03/07/2020 05:50:16 PM - INFO - KNeighborsClassifier(algorithm='auto', leaf_size=5, metric='euclidean',
                     metric_params=None, n_jobs=None, n_neighbors=50, p=2,
                     weights='distance')
03/07/2020 05:50:16 PM - INFO - Train time: 0.011s
03/07/2020 05:50:42 PM - INFO - Test time:  26.352s
03/07/2020 05:50:42 PM - INFO - Accuracy score:   0.827
03/07/2020 05:50:42 PM - INFO - 

===> Classification Report:

03/07/2020 05:50:42 PM - INFO -               precision    recall  f1-score   support

           0       0.80      0.86      0.83     12500
           1       0.85      0.79      0.82     12500

    accuracy                           0.83     25000
   macro avg       0.83      0.83      0.83     25000
weighted avg       0.83      0.83      0.83     25000

03/07/2020 05:50:42 PM - INFO - 

Cross validation:
03/07/2020 05:50:48 PM - INFO - 	accuracy: 5-fold cross validation: [0.8632 0.8744 0.8694 0.864  0.8618]
03/07/2020 05:50:48 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 86.66 (+/- 0.94)
03/07/2020 05:50:48 PM - INFO - It took 144.57419657707214 seconds
03/07/2020 05:50:48 PM - INFO - ********************************************************************************
03/07/2020 05:50:48 PM - INFO - ################################################################################
03/07/2020 05:50:48 PM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 05:50:48 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 05:50:48 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 05:50:48 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  80.24%  |  [0.801  0.803  0.799  0.7968 0.7986]  |  79.97 (+/- 0.43)  |  10.33  |  0.5739  |
03/07/2020 05:50:48 PM - INFO - |  2  |  BERNOULLI_NB  |  81.52%  |  [0.8384 0.8398 0.8524 0.8404 0.8524]  |  84.47 (+/- 1.27)  |  0.02375  |  0.01999  |
03/07/2020 05:50:48 PM - INFO - |  3  |  COMPLEMENT_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01545  |  0.008116  |
03/07/2020 05:50:48 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  71.35%  |  [0.7128 0.7138 0.7202 0.7098 0.714 ]  |  71.41 (+/- 0.68)  |  23.46  |  0.02994  |
03/07/2020 05:50:48 PM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  73.29%  |  [0.8144 0.8248 0.8262 0.814  0.815 ]  |  81.89 (+/- 1.09)  |  0.008312  |  26.67  |
03/07/2020 05:50:48 PM - INFO - |  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  88.44 (+/- 0.98)  |  0.2373  |  0.003927  |
03/07/2020 05:50:48 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  88.39%  |  [0.8822 0.8946 0.8848 0.887  0.8852]  |  88.68 (+/- 0.84)  |  1.206  |  0.008219  |
03/07/2020 05:50:48 PM - INFO - |  9  |  MULTINOMIAL_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01547  |  0.008373  |
03/07/2020 05:50:48 PM - INFO - |  10  |  NEAREST_CENTROID  |  83.72%  |  [0.8316 0.838  0.8342 0.8392 0.8358]  |  83.58 (+/- 0.54)  |  0.01616  |  0.0117  |
03/07/2020 05:50:48 PM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  85.20%  |  [0.8668 0.8766 0.8722 0.8722 0.8652]  |  87.06 (+/- 0.82)  |  0.1836  |  0.004088  |
03/07/2020 05:50:48 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  84.91%  |  [0.845  0.8524 0.8434 0.8442 0.8432]  |  84.56 (+/- 0.69)  |  35.12  |  1.306  |
03/07/2020 05:50:48 PM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 05:50:48 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 05:50:48 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 05:50:48 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  84.60%  |  [0.8398 0.8516 0.8416 0.8366 0.8416]  |  84.22 (+/- 1.00)  |  103.1  |  5.54  |
03/07/2020 05:50:48 PM - INFO - |  2  |  BERNOULLI_NB  |  81.28%  |  [0.8398 0.8424 0.8514 0.8396 0.8516]  |  84.50 (+/- 1.09)  |  0.02419  |  0.01985  |
03/07/2020 05:50:48 PM - INFO - |  3  |  COMPLEMENT_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01547  |  0.008101  |
03/07/2020 05:50:48 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  73.76%  |  [0.7384 0.731  0.7326 0.7306 0.7188]  |  73.03 (+/- 1.28)  |  9.083  |  0.01224  |
03/07/2020 05:50:48 PM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  82.66%  |  [0.8632 0.8744 0.8694 0.864  0.8618]  |  86.66 (+/- 0.94)  |  0.01147  |  26.35  |
03/07/2020 05:50:48 PM - INFO - |  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  88.44 (+/- 0.98)  |  0.2349  |  0.003867  |
03/07/2020 05:50:48 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  87.75%  |  [0.8882 0.897  0.8878 0.8876 0.8818]  |  88.85 (+/- 0.97)  |  1.727  |  0.008252  |
03/07/2020 05:50:48 PM - INFO - |  9  |  MULTINOMIAL_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.0155  |  0.008379  |
03/07/2020 05:50:48 PM - INFO - |  10  |  NEAREST_CENTROID  |  84.65%  |  [0.8426 0.8546 0.8426 0.8522 0.8502]  |  84.84 (+/- 0.99)  |  0.01615  |  0.01705  |
03/07/2020 05:50:48 PM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  88.06%  |  [0.8872 0.897  0.8884 0.8878 0.8852]  |  88.91 (+/- 0.82)  |  0.8665  |  0.003932  |
03/07/2020 05:50:48 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  85.54%  |  [0.8574 0.8582 0.8564 0.8446 0.854 ]  |  85.41 (+/- 0.99)  |  58.66  |  2.501  |
03/07/2020 05:50:48 PM - INFO - 

03/07/2020 05:50:48 PM - INFO - ################################################################################
03/07/2020 05:50:48 PM - INFO - 12)
03/07/2020 05:50:48 PM - INFO - ********************************************************************************
03/07/2020 05:50:48 PM - INFO - Classifier: PERCEPTRON, Dataset: IMDB_REVIEWS
03/07/2020 05:50:48 PM - INFO - ********************************************************************************
03/07/2020 05:50:55 PM - INFO - 

Performing grid search...

03/07/2020 05:50:55 PM - INFO - Parameters:
03/07/2020 05:50:55 PM - INFO - {'classifier__early_stopping': [True], 'classifier__max_iter': [100], 'classifier__n_iter_no_change': [3, 15], 'classifier__penalty': ['l2'], 'classifier__tol': [0.0001, 0.1], 'classifier__validation_fraction': [0.0001, 0.01]}
03/07/2020 05:50:56 PM - INFO - 	Done in 1.694s
03/07/2020 05:50:56 PM - INFO - 	Best score: 0.816
03/07/2020 05:50:56 PM - INFO - 	Best parameters set:
03/07/2020 05:50:56 PM - INFO - 		classifier__early_stopping: True
03/07/2020 05:50:56 PM - INFO - 		classifier__max_iter: 100
03/07/2020 05:50:56 PM - INFO - 		classifier__n_iter_no_change: 3
03/07/2020 05:50:56 PM - INFO - 		classifier__penalty: 'l2'
03/07/2020 05:50:56 PM - INFO - 		classifier__tol: 0.0001
03/07/2020 05:50:56 PM - INFO - 		classifier__validation_fraction: 0.01
03/07/2020 05:50:56 PM - INFO - 

USING PERCEPTRON WITH DEFAULT PARAMETERS
03/07/2020 05:50:56 PM - INFO - ________________________________________________________________________________
03/07/2020 05:50:56 PM - INFO - Training: 
03/07/2020 05:50:56 PM - INFO - Perceptron(alpha=0.0001, class_weight=None, early_stopping=False, eta0=1.0,
           fit_intercept=True, max_iter=1000, n_iter_no_change=5, n_jobs=None,
           penalty=None, random_state=0, shuffle=True, tol=0.001,
           validation_fraction=0.1, verbose=0, warm_start=False)
03/07/2020 05:50:56 PM - INFO - Train time: 0.107s
03/07/2020 05:50:56 PM - INFO - Test time:  0.004s
03/07/2020 05:50:56 PM - INFO - Accuracy score:   0.844
03/07/2020 05:50:56 PM - INFO - 

===> Classification Report:

03/07/2020 05:50:56 PM - INFO -               precision    recall  f1-score   support

           0       0.83      0.86      0.85     12500
           1       0.86      0.83      0.84     12500

    accuracy                           0.84     25000
   macro avg       0.84      0.84      0.84     25000
weighted avg       0.84      0.84      0.84     25000

03/07/2020 05:50:56 PM - INFO - 

Cross validation:
03/07/2020 05:50:57 PM - INFO - 	accuracy: 5-fold cross validation: [0.861  0.868  0.8654 0.8614 0.8536]
03/07/2020 05:50:57 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 86.19 (+/- 0.98)
03/07/2020 05:50:57 PM - INFO - 

USING PERCEPTRON WITH BEST PARAMETERS: {'early_stopping': True, 'max_iter': 100, 'n_iter_no_change': 3, 'penalty': 'l2', 'tol': 0.0001, 'validation_fraction': 0.01}
03/07/2020 05:50:57 PM - INFO - ________________________________________________________________________________
03/07/2020 05:50:57 PM - INFO - Training: 
03/07/2020 05:50:57 PM - INFO - Perceptron(alpha=0.0001, class_weight=None, early_stopping=True, eta0=1.0,
           fit_intercept=True, max_iter=100, n_iter_no_change=3, n_jobs=None,
           penalty='l2', random_state=0, shuffle=True, tol=0.0001,
           validation_fraction=0.01, verbose=0, warm_start=False)
03/07/2020 05:50:57 PM - INFO - Train time: 0.092s
03/07/2020 05:50:57 PM - INFO - Test time:  0.006s
03/07/2020 05:50:57 PM - INFO - Accuracy score:   0.806
03/07/2020 05:50:57 PM - INFO - 

===> Classification Report:

03/07/2020 05:50:57 PM - INFO -               precision    recall  f1-score   support

           0       0.81      0.80      0.81     12500
           1       0.80      0.81      0.81     12500

    accuracy                           0.81     25000
   macro avg       0.81      0.81      0.81     25000
weighted avg       0.81      0.81      0.81     25000

03/07/2020 05:50:57 PM - INFO - 

Cross validation:
03/07/2020 05:50:57 PM - INFO - 	accuracy: 5-fold cross validation: [0.8166 0.8264 0.8144 0.81   0.8102]
03/07/2020 05:50:57 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 81.55 (+/- 1.20)
03/07/2020 05:50:57 PM - INFO - It took 8.906213760375977 seconds
03/07/2020 05:50:57 PM - INFO - ********************************************************************************
03/07/2020 05:50:57 PM - INFO - ################################################################################
03/07/2020 05:50:57 PM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 05:50:57 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 05:50:57 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 05:50:57 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  80.24%  |  [0.801  0.803  0.799  0.7968 0.7986]  |  79.97 (+/- 0.43)  |  10.33  |  0.5739  |
03/07/2020 05:50:57 PM - INFO - |  2  |  BERNOULLI_NB  |  81.52%  |  [0.8384 0.8398 0.8524 0.8404 0.8524]  |  84.47 (+/- 1.27)  |  0.02375  |  0.01999  |
03/07/2020 05:50:57 PM - INFO - |  3  |  COMPLEMENT_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01545  |  0.008116  |
03/07/2020 05:50:57 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  71.35%  |  [0.7128 0.7138 0.7202 0.7098 0.714 ]  |  71.41 (+/- 0.68)  |  23.46  |  0.02994  |
03/07/2020 05:50:57 PM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  73.29%  |  [0.8144 0.8248 0.8262 0.814  0.815 ]  |  81.89 (+/- 1.09)  |  0.008312  |  26.67  |
03/07/2020 05:50:57 PM - INFO - |  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  88.44 (+/- 0.98)  |  0.2373  |  0.003927  |
03/07/2020 05:50:57 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  88.39%  |  [0.8822 0.8946 0.8848 0.887  0.8852]  |  88.68 (+/- 0.84)  |  1.206  |  0.008219  |
03/07/2020 05:50:57 PM - INFO - |  9  |  MULTINOMIAL_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01547  |  0.008373  |
03/07/2020 05:50:57 PM - INFO - |  10  |  NEAREST_CENTROID  |  83.72%  |  [0.8316 0.838  0.8342 0.8392 0.8358]  |  83.58 (+/- 0.54)  |  0.01616  |  0.0117  |
03/07/2020 05:50:57 PM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  85.20%  |  [0.8668 0.8766 0.8722 0.8722 0.8652]  |  87.06 (+/- 0.82)  |  0.1836  |  0.004088  |
03/07/2020 05:50:57 PM - INFO - |  12  |  PERCEPTRON  |  84.43%  |  [0.861  0.868  0.8654 0.8614 0.8536]  |  86.19 (+/- 0.98)  |  0.1075  |  0.004236  |
03/07/2020 05:50:57 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  84.91%  |  [0.845  0.8524 0.8434 0.8442 0.8432]  |  84.56 (+/- 0.69)  |  35.12  |  1.306  |
03/07/2020 05:50:57 PM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 05:50:57 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 05:50:57 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 05:50:57 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  84.60%  |  [0.8398 0.8516 0.8416 0.8366 0.8416]  |  84.22 (+/- 1.00)  |  103.1  |  5.54  |
03/07/2020 05:50:57 PM - INFO - |  2  |  BERNOULLI_NB  |  81.28%  |  [0.8398 0.8424 0.8514 0.8396 0.8516]  |  84.50 (+/- 1.09)  |  0.02419  |  0.01985  |
03/07/2020 05:50:57 PM - INFO - |  3  |  COMPLEMENT_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01547  |  0.008101  |
03/07/2020 05:50:57 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  73.76%  |  [0.7384 0.731  0.7326 0.7306 0.7188]  |  73.03 (+/- 1.28)  |  9.083  |  0.01224  |
03/07/2020 05:50:57 PM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  82.66%  |  [0.8632 0.8744 0.8694 0.864  0.8618]  |  86.66 (+/- 0.94)  |  0.01147  |  26.35  |
03/07/2020 05:50:57 PM - INFO - |  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  88.44 (+/- 0.98)  |  0.2349  |  0.003867  |
03/07/2020 05:50:57 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  87.75%  |  [0.8882 0.897  0.8878 0.8876 0.8818]  |  88.85 (+/- 0.97)  |  1.727  |  0.008252  |
03/07/2020 05:50:57 PM - INFO - |  9  |  MULTINOMIAL_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.0155  |  0.008379  |
03/07/2020 05:50:57 PM - INFO - |  10  |  NEAREST_CENTROID  |  84.65%  |  [0.8426 0.8546 0.8426 0.8522 0.8502]  |  84.84 (+/- 0.99)  |  0.01615  |  0.01705  |
03/07/2020 05:50:57 PM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  88.06%  |  [0.8872 0.897  0.8884 0.8878 0.8852]  |  88.91 (+/- 0.82)  |  0.8665  |  0.003932  |
03/07/2020 05:50:57 PM - INFO - |  12  |  PERCEPTRON  |  80.64%  |  [0.8166 0.8264 0.8144 0.81   0.8102]  |  81.55 (+/- 1.20)  |  0.09231  |  0.006239  |
03/07/2020 05:50:57 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  85.54%  |  [0.8574 0.8582 0.8564 0.8446 0.854 ]  |  85.41 (+/- 0.99)  |  58.66  |  2.501  |
03/07/2020 05:50:57 PM - INFO - 

03/07/2020 05:50:57 PM - INFO - ################################################################################
03/07/2020 05:50:57 PM - INFO - 13)
03/07/2020 05:50:57 PM - INFO - ********************************************************************************
03/07/2020 05:50:57 PM - INFO - Classifier: RIDGE_CLASSIFIER, Dataset: IMDB_REVIEWS
03/07/2020 05:50:57 PM - INFO - ********************************************************************************
03/07/2020 05:51:04 PM - INFO - 

Performing grid search...

03/07/2020 05:51:04 PM - INFO - Parameters:
03/07/2020 05:51:04 PM - INFO - {'classifier__alpha': [0.5, 1.0], 'classifier__tol': [0.0001, 0.001]}
03/07/2020 05:51:06 PM - INFO - 	Done in 2.013s
03/07/2020 05:51:06 PM - INFO - 	Best score: 0.886
03/07/2020 05:51:06 PM - INFO - 	Best parameters set:
03/07/2020 05:51:06 PM - INFO - 		classifier__alpha: 1.0
03/07/2020 05:51:06 PM - INFO - 		classifier__tol: 0.0001
03/07/2020 05:51:06 PM - INFO - 

USING RIDGE_CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 05:51:06 PM - INFO - ________________________________________________________________________________
03/07/2020 05:51:06 PM - INFO - Training: 
03/07/2020 05:51:06 PM - INFO - RidgeClassifier(alpha=1.0, class_weight=None, copy_X=True, fit_intercept=True,
                max_iter=None, normalize=False, random_state=None,
                solver='auto', tol=0.001)
03/07/2020 05:51:06 PM - INFO - Train time: 0.426s
03/07/2020 05:51:06 PM - INFO - Test time:  0.008s
03/07/2020 05:51:06 PM - INFO - Accuracy score:   0.869
03/07/2020 05:51:06 PM - INFO - 

===> Classification Report:

03/07/2020 05:51:06 PM - INFO -               precision    recall  f1-score   support

           0       0.87      0.87      0.87     12500
           1       0.87      0.87      0.87     12500

    accuracy                           0.87     25000
   macro avg       0.87      0.87      0.87     25000
weighted avg       0.87      0.87      0.87     25000

03/07/2020 05:51:06 PM - INFO - 

Cross validation:
03/07/2020 05:51:06 PM - INFO - 	accuracy: 5-fold cross validation: [0.8836 0.895  0.8888 0.882  0.8788]
03/07/2020 05:51:06 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 88.56 (+/- 1.14)
03/07/2020 05:51:06 PM - INFO - 

USING RIDGE_CLASSIFIER WITH BEST PARAMETERS: {'alpha': 1.0, 'tol': 0.0001}
03/07/2020 05:51:06 PM - INFO - ________________________________________________________________________________
03/07/2020 05:51:06 PM - INFO - Training: 
03/07/2020 05:51:06 PM - INFO - RidgeClassifier(alpha=1.0, class_weight=None, copy_X=True, fit_intercept=True,
                max_iter=None, normalize=False, random_state=None,
                solver='auto', tol=0.0001)
03/07/2020 05:51:07 PM - INFO - Train time: 0.488s
03/07/2020 05:51:07 PM - INFO - Test time:  0.008s
03/07/2020 05:51:07 PM - INFO - Accuracy score:   0.869
03/07/2020 05:51:07 PM - INFO - 

===> Classification Report:

03/07/2020 05:51:07 PM - INFO -               precision    recall  f1-score   support

           0       0.87      0.87      0.87     12500
           1       0.87      0.87      0.87     12500

    accuracy                           0.87     25000
   macro avg       0.87      0.87      0.87     25000
weighted avg       0.87      0.87      0.87     25000

03/07/2020 05:51:07 PM - INFO - 

Cross validation:
03/07/2020 05:51:07 PM - INFO - 	accuracy: 5-fold cross validation: [0.8838 0.8952 0.8892 0.882  0.8788]
03/07/2020 05:51:07 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 88.58 (+/- 1.16)
03/07/2020 05:51:07 PM - INFO - It took 10.504548072814941 seconds
03/07/2020 05:51:07 PM - INFO - ********************************************************************************
03/07/2020 05:51:07 PM - INFO - ################################################################################
03/07/2020 05:51:07 PM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 05:51:07 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 05:51:07 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 05:51:07 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  80.24%  |  [0.801  0.803  0.799  0.7968 0.7986]  |  79.97 (+/- 0.43)  |  10.33  |  0.5739  |
03/07/2020 05:51:07 PM - INFO - |  2  |  BERNOULLI_NB  |  81.52%  |  [0.8384 0.8398 0.8524 0.8404 0.8524]  |  84.47 (+/- 1.27)  |  0.02375  |  0.01999  |
03/07/2020 05:51:07 PM - INFO - |  3  |  COMPLEMENT_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01545  |  0.008116  |
03/07/2020 05:51:07 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  71.35%  |  [0.7128 0.7138 0.7202 0.7098 0.714 ]  |  71.41 (+/- 0.68)  |  23.46  |  0.02994  |
03/07/2020 05:51:07 PM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  73.29%  |  [0.8144 0.8248 0.8262 0.814  0.815 ]  |  81.89 (+/- 1.09)  |  0.008312  |  26.67  |
03/07/2020 05:51:07 PM - INFO - |  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  88.44 (+/- 0.98)  |  0.2373  |  0.003927  |
03/07/2020 05:51:07 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  88.39%  |  [0.8822 0.8946 0.8848 0.887  0.8852]  |  88.68 (+/- 0.84)  |  1.206  |  0.008219  |
03/07/2020 05:51:07 PM - INFO - |  9  |  MULTINOMIAL_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01547  |  0.008373  |
03/07/2020 05:51:07 PM - INFO - |  10  |  NEAREST_CENTROID  |  83.72%  |  [0.8316 0.838  0.8342 0.8392 0.8358]  |  83.58 (+/- 0.54)  |  0.01616  |  0.0117  |
03/07/2020 05:51:07 PM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  85.20%  |  [0.8668 0.8766 0.8722 0.8722 0.8652]  |  87.06 (+/- 0.82)  |  0.1836  |  0.004088  |
03/07/2020 05:51:07 PM - INFO - |  12  |  PERCEPTRON  |  84.43%  |  [0.861  0.868  0.8654 0.8614 0.8536]  |  86.19 (+/- 0.98)  |  0.1075  |  0.004236  |
03/07/2020 05:51:07 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  84.91%  |  [0.845  0.8524 0.8434 0.8442 0.8432]  |  84.56 (+/- 0.69)  |  35.12  |  1.306  |
03/07/2020 05:51:07 PM - INFO - |  14  |  RIDGE_CLASSIFIER  |  86.90%  |  [0.8836 0.895  0.8888 0.882  0.8788]  |  88.56 (+/- 1.14)  |  0.4265  |  0.007972  |
03/07/2020 05:51:07 PM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 05:51:07 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 05:51:07 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 05:51:07 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  84.60%  |  [0.8398 0.8516 0.8416 0.8366 0.8416]  |  84.22 (+/- 1.00)  |  103.1  |  5.54  |
03/07/2020 05:51:07 PM - INFO - |  2  |  BERNOULLI_NB  |  81.28%  |  [0.8398 0.8424 0.8514 0.8396 0.8516]  |  84.50 (+/- 1.09)  |  0.02419  |  0.01985  |
03/07/2020 05:51:07 PM - INFO - |  3  |  COMPLEMENT_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01547  |  0.008101  |
03/07/2020 05:51:07 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  73.76%  |  [0.7384 0.731  0.7326 0.7306 0.7188]  |  73.03 (+/- 1.28)  |  9.083  |  0.01224  |
03/07/2020 05:51:07 PM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  82.66%  |  [0.8632 0.8744 0.8694 0.864  0.8618]  |  86.66 (+/- 0.94)  |  0.01147  |  26.35  |
03/07/2020 05:51:07 PM - INFO - |  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  88.44 (+/- 0.98)  |  0.2349  |  0.003867  |
03/07/2020 05:51:07 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  87.75%  |  [0.8882 0.897  0.8878 0.8876 0.8818]  |  88.85 (+/- 0.97)  |  1.727  |  0.008252  |
03/07/2020 05:51:07 PM - INFO - |  9  |  MULTINOMIAL_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.0155  |  0.008379  |
03/07/2020 05:51:07 PM - INFO - |  10  |  NEAREST_CENTROID  |  84.65%  |  [0.8426 0.8546 0.8426 0.8522 0.8502]  |  84.84 (+/- 0.99)  |  0.01615  |  0.01705  |
03/07/2020 05:51:07 PM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  88.06%  |  [0.8872 0.897  0.8884 0.8878 0.8852]  |  88.91 (+/- 0.82)  |  0.8665  |  0.003932  |
03/07/2020 05:51:07 PM - INFO - |  12  |  PERCEPTRON  |  80.64%  |  [0.8166 0.8264 0.8144 0.81   0.8102]  |  81.55 (+/- 1.20)  |  0.09231  |  0.006239  |
03/07/2020 05:51:07 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  85.54%  |  [0.8574 0.8582 0.8564 0.8446 0.854 ]  |  85.41 (+/- 0.99)  |  58.66  |  2.501  |
03/07/2020 05:51:07 PM - INFO - |  14  |  RIDGE_CLASSIFIER  |  86.90%  |  [0.8838 0.8952 0.8892 0.882  0.8788]  |  88.58 (+/- 1.16)  |  0.4877  |  0.008011  |
03/07/2020 05:51:07 PM - INFO - 

03/07/2020 05:51:07 PM - INFO - ################################################################################
03/07/2020 05:51:07 PM - INFO - 14)
03/07/2020 05:51:07 PM - INFO - ********************************************************************************
03/07/2020 05:51:07 PM - INFO - Classifier: GRADIENT_BOOSTING_CLASSIFIER, Dataset: IMDB_REVIEWS
03/07/2020 05:51:07 PM - INFO - ********************************************************************************
03/07/2020 05:51:14 PM - INFO - 

Performing grid search...

03/07/2020 05:51:14 PM - INFO - Parameters:
03/07/2020 05:51:14 PM - INFO - {'classifier__learning_rate': [0.01, 0.1], 'classifier__n_estimators': [100, 200]}
03/07/2020 06:01:29 PM - INFO - 	Done in 615.141s
03/07/2020 06:01:29 PM - INFO - 	Best score: 0.826
03/07/2020 06:01:29 PM - INFO - 	Best parameters set:
03/07/2020 06:01:29 PM - INFO - 		classifier__learning_rate: 0.1
03/07/2020 06:01:29 PM - INFO - 		classifier__n_estimators: 200
03/07/2020 06:01:29 PM - INFO - 

USING GRADIENT_BOOSTING_CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 06:01:29 PM - INFO - ________________________________________________________________________________
03/07/2020 06:01:29 PM - INFO - Training: 
03/07/2020 06:01:29 PM - INFO - GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.1, loss='deviance', max_depth=3,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_iter_no_change=None, presort='deprecated',
                           random_state=None, subsample=1.0, tol=0.0001,
                           validation_fraction=0.1, verbose=0,
                           warm_start=False)
03/07/2020 06:02:20 PM - INFO - Train time: 50.883s
03/07/2020 06:02:20 PM - INFO - Test time:  0.052s
03/07/2020 06:02:20 PM - INFO - Accuracy score:   0.807
03/07/2020 06:02:20 PM - INFO - 

===> Classification Report:

03/07/2020 06:02:20 PM - INFO -               precision    recall  f1-score   support

           0       0.85      0.75      0.79     12500
           1       0.77      0.87      0.82     12500

    accuracy                           0.81     25000
   macro avg       0.81      0.81      0.81     25000
weighted avg       0.81      0.81      0.81     25000

03/07/2020 06:02:20 PM - INFO - 

Cross validation:
03/07/2020 06:03:48 PM - INFO - 	accuracy: 5-fold cross validation: [0.805  0.8092 0.8048 0.7986 0.8074]
03/07/2020 06:03:48 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 80.50 (+/- 0.72)
03/07/2020 06:03:48 PM - INFO - 

USING GRADIENT_BOOSTING_CLASSIFIER WITH BEST PARAMETERS: {'learning_rate': 0.1, 'n_estimators': 200}
03/07/2020 06:03:48 PM - INFO - ________________________________________________________________________________
03/07/2020 06:03:48 PM - INFO - Training: 
03/07/2020 06:03:48 PM - INFO - GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.1, loss='deviance', max_depth=3,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=200,
                           n_iter_no_change=None, presort='deprecated',
                           random_state=None, subsample=1.0, tol=0.0001,
                           validation_fraction=0.1, verbose=0,
                           warm_start=False)
03/07/2020 06:05:30 PM - INFO - Train time: 101.876s
03/07/2020 06:05:30 PM - INFO - Test time:  0.066s
03/07/2020 06:05:30 PM - INFO - Accuracy score:   0.829
03/07/2020 06:05:30 PM - INFO - 

===> Classification Report:

03/07/2020 06:05:30 PM - INFO -               precision    recall  f1-score   support

           0       0.86      0.79      0.82     12500
           1       0.80      0.87      0.84     12500

    accuracy                           0.83     25000
   macro avg       0.83      0.83      0.83     25000
weighted avg       0.83      0.83      0.83     25000

03/07/2020 06:05:30 PM - INFO - 

Cross validation:
03/07/2020 06:08:29 PM - INFO - 	accuracy: 5-fold cross validation: [0.8278 0.8286 0.8238 0.8228 0.8284]
03/07/2020 06:08:29 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 82.63 (+/- 0.49)
03/07/2020 06:08:29 PM - INFO - It took 1041.0706496238708 seconds
03/07/2020 06:08:29 PM - INFO - ********************************************************************************
03/07/2020 06:08:29 PM - INFO - ################################################################################
03/07/2020 06:08:29 PM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 06:08:29 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 06:08:29 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 06:08:29 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  80.24%  |  [0.801  0.803  0.799  0.7968 0.7986]  |  79.97 (+/- 0.43)  |  10.33  |  0.5739  |
03/07/2020 06:08:29 PM - INFO - |  2  |  BERNOULLI_NB  |  81.52%  |  [0.8384 0.8398 0.8524 0.8404 0.8524]  |  84.47 (+/- 1.27)  |  0.02375  |  0.01999  |
03/07/2020 06:08:29 PM - INFO - |  3  |  COMPLEMENT_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01545  |  0.008116  |
03/07/2020 06:08:29 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  71.35%  |  [0.7128 0.7138 0.7202 0.7098 0.714 ]  |  71.41 (+/- 0.68)  |  23.46  |  0.02994  |
03/07/2020 06:08:29 PM - INFO - |  5  |  GRADIENT_BOOSTING_CLASSIFIER  |  80.72%  |  [0.805  0.8092 0.8048 0.7986 0.8074]  |  80.50 (+/- 0.72)  |  50.88  |  0.05231  |
03/07/2020 06:08:29 PM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  73.29%  |  [0.8144 0.8248 0.8262 0.814  0.815 ]  |  81.89 (+/- 1.09)  |  0.008312  |  26.67  |
03/07/2020 06:08:29 PM - INFO - |  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  88.44 (+/- 0.98)  |  0.2373  |  0.003927  |
03/07/2020 06:08:29 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  88.39%  |  [0.8822 0.8946 0.8848 0.887  0.8852]  |  88.68 (+/- 0.84)  |  1.206  |  0.008219  |
03/07/2020 06:08:29 PM - INFO - |  9  |  MULTINOMIAL_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01547  |  0.008373  |
03/07/2020 06:08:29 PM - INFO - |  10  |  NEAREST_CENTROID  |  83.72%  |  [0.8316 0.838  0.8342 0.8392 0.8358]  |  83.58 (+/- 0.54)  |  0.01616  |  0.0117  |
03/07/2020 06:08:29 PM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  85.20%  |  [0.8668 0.8766 0.8722 0.8722 0.8652]  |  87.06 (+/- 0.82)  |  0.1836  |  0.004088  |
03/07/2020 06:08:29 PM - INFO - |  12  |  PERCEPTRON  |  84.43%  |  [0.861  0.868  0.8654 0.8614 0.8536]  |  86.19 (+/- 0.98)  |  0.1075  |  0.004236  |
03/07/2020 06:08:29 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  84.91%  |  [0.845  0.8524 0.8434 0.8442 0.8432]  |  84.56 (+/- 0.69)  |  35.12  |  1.306  |
03/07/2020 06:08:29 PM - INFO - |  14  |  RIDGE_CLASSIFIER  |  86.90%  |  [0.8836 0.895  0.8888 0.882  0.8788]  |  88.56 (+/- 1.14)  |  0.4265  |  0.007972  |
03/07/2020 06:08:29 PM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 06:08:29 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 06:08:29 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 06:08:29 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  84.60%  |  [0.8398 0.8516 0.8416 0.8366 0.8416]  |  84.22 (+/- 1.00)  |  103.1  |  5.54  |
03/07/2020 06:08:29 PM - INFO - |  2  |  BERNOULLI_NB  |  81.28%  |  [0.8398 0.8424 0.8514 0.8396 0.8516]  |  84.50 (+/- 1.09)  |  0.02419  |  0.01985  |
03/07/2020 06:08:29 PM - INFO - |  3  |  COMPLEMENT_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01547  |  0.008101  |
03/07/2020 06:08:29 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  73.76%  |  [0.7384 0.731  0.7326 0.7306 0.7188]  |  73.03 (+/- 1.28)  |  9.083  |  0.01224  |
03/07/2020 06:08:29 PM - INFO - |  5  |  GRADIENT_BOOSTING_CLASSIFIER  |  82.87%  |  [0.8278 0.8286 0.8238 0.8228 0.8284]  |  82.63 (+/- 0.49)  |  101.9  |  0.0661  |
03/07/2020 06:08:29 PM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  82.66%  |  [0.8632 0.8744 0.8694 0.864  0.8618]  |  86.66 (+/- 0.94)  |  0.01147  |  26.35  |
03/07/2020 06:08:29 PM - INFO - |  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  88.44 (+/- 0.98)  |  0.2349  |  0.003867  |
03/07/2020 06:08:29 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  87.75%  |  [0.8882 0.897  0.8878 0.8876 0.8818]  |  88.85 (+/- 0.97)  |  1.727  |  0.008252  |
03/07/2020 06:08:29 PM - INFO - |  9  |  MULTINOMIAL_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.0155  |  0.008379  |
03/07/2020 06:08:29 PM - INFO - |  10  |  NEAREST_CENTROID  |  84.65%  |  [0.8426 0.8546 0.8426 0.8522 0.8502]  |  84.84 (+/- 0.99)  |  0.01615  |  0.01705  |
03/07/2020 06:08:29 PM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  88.06%  |  [0.8872 0.897  0.8884 0.8878 0.8852]  |  88.91 (+/- 0.82)  |  0.8665  |  0.003932  |
03/07/2020 06:08:29 PM - INFO - |  12  |  PERCEPTRON  |  80.64%  |  [0.8166 0.8264 0.8144 0.81   0.8102]  |  81.55 (+/- 1.20)  |  0.09231  |  0.006239  |
03/07/2020 06:08:29 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  85.54%  |  [0.8574 0.8582 0.8564 0.8446 0.854 ]  |  85.41 (+/- 0.99)  |  58.66  |  2.501  |
03/07/2020 06:08:29 PM - INFO - |  14  |  RIDGE_CLASSIFIER  |  86.90%  |  [0.8838 0.8952 0.8892 0.882  0.8788]  |  88.58 (+/- 1.16)  |  0.4877  |  0.008011  |
03/07/2020 06:08:29 PM - INFO - 

FINAL CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 06:08:29 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 06:08:29 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 06:08:29 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  80.24%  |  [0.801  0.803  0.799  0.7968 0.7986]  |  79.97 (+/- 0.43)  |  10.33  |  0.5739  |
03/07/2020 06:08:29 PM - INFO - |  2  |  BERNOULLI_NB  |  81.52%  |  [0.8384 0.8398 0.8524 0.8404 0.8524]  |  84.47 (+/- 1.27)  |  0.02375  |  0.01999  |
03/07/2020 06:08:29 PM - INFO - |  3  |  COMPLEMENT_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01545  |  0.008116  |
03/07/2020 06:08:29 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  71.35%  |  [0.7128 0.7138 0.7202 0.7098 0.714 ]  |  71.41 (+/- 0.68)  |  23.46  |  0.02994  |
03/07/2020 06:08:29 PM - INFO - |  5  |  GRADIENT_BOOSTING_CLASSIFIER  |  80.72%  |  [0.805  0.8092 0.8048 0.7986 0.8074]  |  80.50 (+/- 0.72)  |  50.88  |  0.05231  |
03/07/2020 06:08:29 PM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  73.29%  |  [0.8144 0.8248 0.8262 0.814  0.815 ]  |  81.89 (+/- 1.09)  |  0.008312  |  26.67  |
03/07/2020 06:08:29 PM - INFO - |  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  88.44 (+/- 0.98)  |  0.2373  |  0.003927  |
03/07/2020 06:08:29 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  88.39%  |  [0.8822 0.8946 0.8848 0.887  0.8852]  |  88.68 (+/- 0.84)  |  1.206  |  0.008219  |
03/07/2020 06:08:29 PM - INFO - |  9  |  MULTINOMIAL_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01547  |  0.008373  |
03/07/2020 06:08:29 PM - INFO - |  10  |  NEAREST_CENTROID  |  83.72%  |  [0.8316 0.838  0.8342 0.8392 0.8358]  |  83.58 (+/- 0.54)  |  0.01616  |  0.0117  |
03/07/2020 06:08:29 PM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  85.20%  |  [0.8668 0.8766 0.8722 0.8722 0.8652]  |  87.06 (+/- 0.82)  |  0.1836  |  0.004088  |
03/07/2020 06:08:29 PM - INFO - |  12  |  PERCEPTRON  |  84.43%  |  [0.861  0.868  0.8654 0.8614 0.8536]  |  86.19 (+/- 0.98)  |  0.1075  |  0.004236  |
03/07/2020 06:08:29 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  84.91%  |  [0.845  0.8524 0.8434 0.8442 0.8432]  |  84.56 (+/- 0.69)  |  35.12  |  1.306  |
03/07/2020 06:08:29 PM - INFO - |  14  |  RIDGE_CLASSIFIER  |  86.90%  |  [0.8836 0.895  0.8888 0.882  0.8788]  |  88.56 (+/- 1.14)  |  0.4265  |  0.007972  |
03/07/2020 06:08:29 PM - INFO - 

FINAL CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 06:08:29 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 06:08:29 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 06:08:29 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  84.60%  |  [0.8398 0.8516 0.8416 0.8366 0.8416]  |  84.22 (+/- 1.00)  |  103.1  |  5.54  |
03/07/2020 06:08:29 PM - INFO - |  2  |  BERNOULLI_NB  |  81.28%  |  [0.8398 0.8424 0.8514 0.8396 0.8516]  |  84.50 (+/- 1.09)  |  0.02419  |  0.01985  |
03/07/2020 06:08:29 PM - INFO - |  3  |  COMPLEMENT_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01547  |  0.008101  |
03/07/2020 06:08:29 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  73.76%  |  [0.7384 0.731  0.7326 0.7306 0.7188]  |  73.03 (+/- 1.28)  |  9.083  |  0.01224  |
03/07/2020 06:08:29 PM - INFO - |  5  |  GRADIENT_BOOSTING_CLASSIFIER  |  82.87%  |  [0.8278 0.8286 0.8238 0.8228 0.8284]  |  82.63 (+/- 0.49)  |  101.9  |  0.0661  |
03/07/2020 06:08:29 PM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  82.66%  |  [0.8632 0.8744 0.8694 0.864  0.8618]  |  86.66 (+/- 0.94)  |  0.01147  |  26.35  |
03/07/2020 06:08:29 PM - INFO - |  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  88.44 (+/- 0.98)  |  0.2349  |  0.003867  |
03/07/2020 06:08:29 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  87.75%  |  [0.8882 0.897  0.8878 0.8876 0.8818]  |  88.85 (+/- 0.97)  |  1.727  |  0.008252  |
03/07/2020 06:08:29 PM - INFO - |  9  |  MULTINOMIAL_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.0155  |  0.008379  |
03/07/2020 06:08:29 PM - INFO - |  10  |  NEAREST_CENTROID  |  84.65%  |  [0.8426 0.8546 0.8426 0.8522 0.8502]  |  84.84 (+/- 0.99)  |  0.01615  |  0.01705  |
03/07/2020 06:08:29 PM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  88.06%  |  [0.8872 0.897  0.8884 0.8878 0.8852]  |  88.91 (+/- 0.82)  |  0.8665  |  0.003932  |
03/07/2020 06:08:29 PM - INFO - |  12  |  PERCEPTRON  |  80.64%  |  [0.8166 0.8264 0.8144 0.81   0.8102]  |  81.55 (+/- 1.20)  |  0.09231  |  0.006239  |
03/07/2020 06:08:29 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  85.54%  |  [0.8574 0.8582 0.8564 0.8446 0.854 ]  |  85.41 (+/- 0.99)  |  58.66  |  2.501  |
03/07/2020 06:08:29 PM - INFO - |  14  |  RIDGE_CLASSIFIER  |  86.90%  |  [0.8838 0.8952 0.8892 0.882  0.8788]  |  88.58 (+/- 1.16)  |  0.4877  |  0.008011  |
03/07/2020 06:08:29 PM - INFO - 

03/07/2020 06:08:29 PM - INFO - ################################################################################
03/07/2020 06:08:29 PM - INFO - 1)
03/07/2020 06:08:29 PM - INFO - ********************************************************************************
03/07/2020 06:08:29 PM - INFO - Classifier: ADA_BOOST_CLASSIFIER, Dataset: TWENTY_NEWS_GROUPS
03/07/2020 06:08:29 PM - INFO - ********************************************************************************
03/07/2020 06:08:32 PM - INFO - 

Performing grid search...

03/07/2020 06:08:32 PM - INFO - Parameters:
03/07/2020 06:08:32 PM - INFO - {'classifier__learning_rate': [0.1, 1], 'classifier__n_estimators': [200, 500]}
03/07/2020 06:19:49 PM - INFO - 	Done in 677.766s
03/07/2020 06:19:49 PM - INFO - 	Best score: 0.467
03/07/2020 06:19:49 PM - INFO - 	Best parameters set:
03/07/2020 06:19:49 PM - INFO - 		classifier__learning_rate: 1
03/07/2020 06:19:49 PM - INFO - 		classifier__n_estimators: 200
03/07/2020 06:19:49 PM - INFO - 

USING ADA_BOOST_CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 06:19:49 PM - INFO - ________________________________________________________________________________
03/07/2020 06:19:49 PM - INFO - Training: 
03/07/2020 06:19:49 PM - INFO - AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
                   n_estimators=50, random_state=None)
03/07/2020 06:19:54 PM - INFO - Train time: 4.611s
03/07/2020 06:19:54 PM - INFO - Test time:  0.248s
03/07/2020 06:19:54 PM - INFO - Accuracy score:   0.365
03/07/2020 06:19:54 PM - INFO - 

===> Classification Report:

03/07/2020 06:19:54 PM - INFO -               precision    recall  f1-score   support

           0       0.11      0.00      0.01       319
           1       0.59      0.23      0.33       389
           2       0.62      0.39      0.48       394
           3       0.47      0.31      0.38       392
           4       0.69      0.39      0.50       385
           5       0.74      0.42      0.53       395
           6       0.74      0.52      0.61       390
           7       0.78      0.39      0.52       396
           8       0.91      0.33      0.48       398
           9       0.77      0.19      0.30       397
          10       0.64      0.57      0.60       399
          11       0.81      0.44      0.57       396
          12       0.08      0.80      0.15       393
          13       0.92      0.19      0.32       396
          14       0.71      0.34      0.46       394
          15       0.52      0.63      0.57       398
          16       0.49      0.23      0.32       364
          17       0.92      0.50      0.64       376
          18       0.28      0.17      0.22       310
          19       0.29      0.02      0.04       251

    accuracy                           0.37      7532
   macro avg       0.60      0.35      0.40      7532
weighted avg       0.62      0.37      0.41      7532

03/07/2020 06:19:54 PM - INFO - 

Cross validation:
03/07/2020 06:20:19 PM - INFO - 	accuracy: 5-fold cross validation: [0.39460893 0.39019001 0.39991162 0.39019001 0.40539346]
03/07/2020 06:20:19 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 39.61 (+/- 1.18)
03/07/2020 06:20:19 PM - INFO - 

USING ADA_BOOST_CLASSIFIER WITH BEST PARAMETERS: {'learning_rate': 1, 'n_estimators': 200}
03/07/2020 06:20:19 PM - INFO - ________________________________________________________________________________
03/07/2020 06:20:19 PM - INFO - Training: 
03/07/2020 06:20:19 PM - INFO - AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1,
                   n_estimators=200, random_state=None)
03/07/2020 06:20:37 PM - INFO - Train time: 18.294s
03/07/2020 06:20:38 PM - INFO - Test time:  0.950s
03/07/2020 06:20:38 PM - INFO - Accuracy score:   0.440
03/07/2020 06:20:38 PM - INFO - 

===> Classification Report:

03/07/2020 06:20:38 PM - INFO -               precision    recall  f1-score   support

           0       0.39      0.25      0.30       319
           1       0.38      0.46      0.41       389
           2       0.53      0.41      0.46       394
           3       0.50      0.48      0.49       392
           4       0.62      0.48      0.54       385
           5       0.62      0.49      0.54       395
           6       0.53      0.54      0.53       390
           7       0.72      0.39      0.51       396
           8       0.89      0.44      0.59       398
           9       0.56      0.49      0.52       397
          10       0.77      0.43      0.56       399
          11       0.80      0.49      0.61       396
          12       0.12      0.64      0.20       393
          13       0.81      0.31      0.45       396
          14       0.70      0.41      0.51       394
          15       0.53      0.55      0.54       398
          16       0.54      0.40      0.46       364
          17       0.88      0.54      0.67       376
          18       0.25      0.30      0.27       310
          19       0.23      0.15      0.18       251

    accuracy                           0.44      7532
   macro avg       0.57      0.43      0.47      7532
weighted avg       0.58      0.44      0.48      7532

03/07/2020 06:20:38 PM - INFO - 

Cross validation:
03/07/2020 06:22:17 PM - INFO - 	accuracy: 5-fold cross validation: [0.46973045 0.45779938 0.46531153 0.47017234 0.46993811]
03/07/2020 06:22:17 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 46.66 (+/- 0.95)
03/07/2020 06:22:17 PM - INFO - It took 828.5440471172333 seconds
03/07/2020 06:22:17 PM - INFO - ********************************************************************************
03/07/2020 06:22:17 PM - INFO - ################################################################################
03/07/2020 06:22:17 PM - INFO - 

CURRENT CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 06:22:17 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 06:22:17 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 06:22:17 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  36.52%  |  [0.39460893 0.39019001 0.39991162 0.39019001 0.40539346]  |  39.61 (+/- 1.18)  |  4.611  |  0.2475  |
03/07/2020 06:22:17 PM - INFO - 

CURRENT CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 06:22:17 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 06:22:17 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 06:22:17 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  44.03%  |  [0.46973045 0.45779938 0.46531153 0.47017234 0.46993811]  |  46.66 (+/- 0.95)  |  18.29  |  0.9501  |
03/07/2020 06:22:17 PM - INFO - 

03/07/2020 06:22:17 PM - INFO - ################################################################################
03/07/2020 06:22:17 PM - INFO - 2)
03/07/2020 06:22:17 PM - INFO - ********************************************************************************
03/07/2020 06:22:17 PM - INFO - Classifier: DECISION_TREE_CLASSIFIER, Dataset: TWENTY_NEWS_GROUPS
03/07/2020 06:22:17 PM - INFO - ********************************************************************************
03/07/2020 06:22:20 PM - INFO - 

Performing grid search...

03/07/2020 06:22:20 PM - INFO - Parameters:
03/07/2020 06:22:20 PM - INFO - {'classifier__criterion': ['entropy', 'gini'], 'classifier__splitter': ['best', 'random'], 'classifier__min_samples_split': [2, 100, 250]}
03/07/2020 06:23:38 PM - INFO - 	Done in 78.090s
03/07/2020 06:23:38 PM - INFO - 	Best score: 0.492
03/07/2020 06:23:38 PM - INFO - 	Best parameters set:
03/07/2020 06:23:38 PM - INFO - 		classifier__criterion: 'gini'
03/07/2020 06:23:38 PM - INFO - 		classifier__min_samples_split: 2
03/07/2020 06:23:38 PM - INFO - 		classifier__splitter: 'random'
03/07/2020 06:23:38 PM - INFO - 

USING DECISION_TREE_CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 06:23:38 PM - INFO - ________________________________________________________________________________
03/07/2020 06:23:38 PM - INFO - Training: 
03/07/2020 06:23:38 PM - INFO - DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=None, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=None, splitter='best')
03/07/2020 06:23:48 PM - INFO - Train time: 10.336s
03/07/2020 06:23:48 PM - INFO - Test time:  0.009s
03/07/2020 06:23:48 PM - INFO - Accuracy score:   0.428
03/07/2020 06:23:48 PM - INFO - 

===> Classification Report:

03/07/2020 06:23:49 PM - INFO -               precision    recall  f1-score   support

           0       0.27      0.22      0.24       319
           1       0.42      0.44      0.43       389
           2       0.39      0.37      0.38       394
           3       0.42      0.38      0.39       392
           4       0.45      0.46      0.46       385
           5       0.50      0.45      0.47       395
           6       0.56      0.56      0.56       390
           7       0.26      0.53      0.35       396
           8       0.52      0.54      0.53       398
           9       0.52      0.47      0.50       397
          10       0.64      0.59      0.61       399
          11       0.63      0.46      0.53       396
          12       0.29      0.27      0.28       393
          13       0.47      0.40      0.43       396
          14       0.46      0.50      0.48       394
          15       0.48      0.47      0.48       398
          16       0.35      0.37      0.36       364
          17       0.62      0.49      0.55       376
          18       0.25      0.23      0.24       310
          19       0.18      0.18      0.18       251

    accuracy                           0.43      7532
   macro avg       0.43      0.42      0.42      7532
weighted avg       0.44      0.43      0.43      7532

03/07/2020 06:23:49 PM - INFO - 

Cross validation:
03/07/2020 06:23:58 PM - INFO - 	accuracy: 5-fold cross validation: [0.49094123 0.47591692 0.46442775 0.47105612 0.5       ]
03/07/2020 06:23:58 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 48.05 (+/- 2.62)
03/07/2020 06:23:58 PM - INFO - 

USING DECISION_TREE_CLASSIFIER WITH BEST PARAMETERS: {'criterion': 'gini', 'min_samples_split': 2, 'splitter': 'random'}
03/07/2020 06:23:58 PM - INFO - ________________________________________________________________________________
03/07/2020 06:23:58 PM - INFO - Training: 
03/07/2020 06:23:58 PM - INFO - DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=None, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=None, splitter='random')
03/07/2020 06:24:07 PM - INFO - Train time: 8.941s
03/07/2020 06:24:07 PM - INFO - Test time:  0.006s
03/07/2020 06:24:07 PM - INFO - Accuracy score:   0.446
03/07/2020 06:24:07 PM - INFO - 

===> Classification Report:

03/07/2020 06:24:07 PM - INFO -               precision    recall  f1-score   support

           0       0.32      0.28      0.29       319
           1       0.48      0.43      0.45       389
           2       0.41      0.37      0.39       394
           3       0.41      0.43      0.42       392
           4       0.48      0.46      0.47       385
           5       0.48      0.47      0.48       395
           6       0.52      0.57      0.54       390
           7       0.26      0.54      0.35       396
           8       0.58      0.53      0.56       398
           9       0.49      0.48      0.49       397
          10       0.65      0.59      0.62       399
          11       0.61      0.47      0.53       396
          12       0.30      0.33      0.32       393
          13       0.51      0.44      0.47       396
          14       0.55      0.51      0.53       394
          15       0.47      0.53      0.50       398
          16       0.41      0.37      0.38       364
          17       0.59      0.51      0.55       376
          18       0.30      0.24      0.27       310
          19       0.20      0.16      0.18       251

    accuracy                           0.45      7532
   macro avg       0.45      0.44      0.44      7532
weighted avg       0.46      0.45      0.45      7532

03/07/2020 06:24:07 PM - INFO - 

Cross validation:
03/07/2020 06:24:16 PM - INFO - 	accuracy: 5-fold cross validation: [0.4874061  0.48608042 0.49138312 0.49359258 0.48099027]
03/07/2020 06:24:16 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 48.79 (+/- 0.88)
03/07/2020 06:24:16 PM - INFO - It took 119.00125980377197 seconds
03/07/2020 06:24:16 PM - INFO - ********************************************************************************
03/07/2020 06:24:16 PM - INFO - ################################################################################
03/07/2020 06:24:16 PM - INFO - 

CURRENT CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 06:24:16 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 06:24:16 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 06:24:16 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  36.52%  |  [0.39460893 0.39019001 0.39991162 0.39019001 0.40539346]  |  39.61 (+/- 1.18)  |  4.611  |  0.2475  |
03/07/2020 06:24:16 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  42.82%  |  [0.49094123 0.47591692 0.46442775 0.47105612 0.5       ]  |  48.05 (+/- 2.62)  |  10.34  |  0.008546  |
03/07/2020 06:24:16 PM - INFO - 

CURRENT CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 06:24:16 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 06:24:16 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 06:24:16 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  44.03%  |  [0.46973045 0.45779938 0.46531153 0.47017234 0.46993811]  |  46.66 (+/- 0.95)  |  18.29  |  0.9501  |
03/07/2020 06:24:16 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  44.57%  |  [0.4874061  0.48608042 0.49138312 0.49359258 0.48099027]  |  48.79 (+/- 0.88)  |  8.941  |  0.0056  |
03/07/2020 06:24:16 PM - INFO - 

03/07/2020 06:24:16 PM - INFO - ################################################################################
03/07/2020 06:24:16 PM - INFO - 3)
03/07/2020 06:24:16 PM - INFO - ********************************************************************************
03/07/2020 06:24:16 PM - INFO - Classifier: LINEAR_SVC, Dataset: TWENTY_NEWS_GROUPS
03/07/2020 06:24:16 PM - INFO - ********************************************************************************
03/07/2020 06:24:19 PM - INFO - 

Performing grid search...

03/07/2020 06:24:19 PM - INFO - Parameters:
03/07/2020 06:24:19 PM - INFO - {'classifier__C': [0.01, 1.0], 'classifier__multi_class': ['ovr', 'crammer_singer'], 'classifier__tol': [0.0001, 0.001]}
03/07/2020 06:25:04 PM - INFO - 	Done in 45.362s
03/07/2020 06:25:04 PM - INFO - 	Best score: 0.757
03/07/2020 06:25:04 PM - INFO - 	Best parameters set:
03/07/2020 06:25:04 PM - INFO - 		classifier__C: 1.0
03/07/2020 06:25:04 PM - INFO - 		classifier__multi_class: 'ovr'
03/07/2020 06:25:04 PM - INFO - 		classifier__tol: 0.0001
03/07/2020 06:25:04 PM - INFO - 

USING LINEAR_SVC WITH DEFAULT PARAMETERS
03/07/2020 06:25:04 PM - INFO - ________________________________________________________________________________
03/07/2020 06:25:04 PM - INFO - Training: 
03/07/2020 06:25:04 PM - INFO - LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
          intercept_scaling=1, loss='squared_hinge', max_iter=1000,
          multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
          verbose=0)
03/07/2020 06:25:05 PM - INFO - Train time: 0.866s
03/07/2020 06:25:05 PM - INFO - Test time:  0.008s
03/07/2020 06:25:05 PM - INFO - Accuracy score:   0.698
03/07/2020 06:25:05 PM - INFO - 

===> Classification Report:

03/07/2020 06:25:05 PM - INFO -               precision    recall  f1-score   support

           0       0.54      0.47      0.50       319
           1       0.65      0.73      0.69       389
           2       0.62      0.60      0.61       394
           3       0.66      0.68      0.67       392
           4       0.73      0.69      0.71       385
           5       0.82      0.70      0.76       395
           6       0.77      0.79      0.78       390
           7       0.75      0.72      0.74       396
           8       0.79      0.76      0.77       398
           9       0.55      0.87      0.68       397
          10       0.89      0.86      0.88       399
          11       0.83      0.72      0.77       396
          12       0.65      0.58      0.61       393
          13       0.78      0.77      0.78       396
          14       0.76      0.74      0.75       394
          15       0.65      0.80      0.72       398
          16       0.58      0.68      0.63       364
          17       0.83      0.77      0.80       376
          18       0.58      0.47      0.52       310
          19       0.45      0.29      0.36       251

    accuracy                           0.70      7532
   macro avg       0.69      0.69      0.69      7532
weighted avg       0.70      0.70      0.70      7532

03/07/2020 06:25:05 PM - INFO - 

Cross validation:
03/07/2020 06:25:07 PM - INFO - 	accuracy: 5-fold cross validation: [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]
03/07/2020 06:25:07 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 75.73 (+/- 2.09)
03/07/2020 06:25:07 PM - INFO - 

USING LINEAR_SVC WITH BEST PARAMETERS: {'C': 1.0, 'multi_class': 'ovr', 'tol': 0.0001}
03/07/2020 06:25:07 PM - INFO - ________________________________________________________________________________
03/07/2020 06:25:07 PM - INFO - Training: 
03/07/2020 06:25:07 PM - INFO - LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
          intercept_scaling=1, loss='squared_hinge', max_iter=1000,
          multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
          verbose=0)
03/07/2020 06:25:08 PM - INFO - Train time: 0.860s
03/07/2020 06:25:08 PM - INFO - Test time:  0.009s
03/07/2020 06:25:08 PM - INFO - Accuracy score:   0.698
03/07/2020 06:25:08 PM - INFO - 

===> Classification Report:

03/07/2020 06:25:08 PM - INFO -               precision    recall  f1-score   support

           0       0.54      0.47      0.50       319
           1       0.65      0.73      0.69       389
           2       0.62      0.60      0.61       394
           3       0.66      0.68      0.67       392
           4       0.73      0.69      0.71       385
           5       0.82      0.70      0.76       395
           6       0.77      0.79      0.78       390
           7       0.75      0.72      0.74       396
           8       0.79      0.76      0.77       398
           9       0.55      0.87      0.68       397
          10       0.89      0.86      0.88       399
          11       0.83      0.72      0.77       396
          12       0.65      0.58      0.61       393
          13       0.78      0.77      0.78       396
          14       0.76      0.74      0.75       394
          15       0.65      0.80      0.72       398
          16       0.58      0.68      0.63       364
          17       0.83      0.77      0.80       376
          18       0.58      0.47      0.52       310
          19       0.45      0.29      0.36       251

    accuracy                           0.70      7532
   macro avg       0.69      0.69      0.69      7532
weighted avg       0.70      0.70      0.70      7532

03/07/2020 06:25:08 PM - INFO - 

Cross validation:
03/07/2020 06:25:09 PM - INFO - 	accuracy: 5-fold cross validation: [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]
03/07/2020 06:25:09 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 75.73 (+/- 2.09)
03/07/2020 06:25:09 PM - INFO - It took 52.96172070503235 seconds
03/07/2020 06:25:09 PM - INFO - ********************************************************************************
03/07/2020 06:25:09 PM - INFO - ################################################################################
03/07/2020 06:25:09 PM - INFO - 

CURRENT CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 06:25:09 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 06:25:09 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 06:25:09 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  36.52%  |  [0.39460893 0.39019001 0.39991162 0.39019001 0.40539346]  |  39.61 (+/- 1.18)  |  4.611  |  0.2475  |
03/07/2020 06:25:09 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  42.82%  |  [0.49094123 0.47591692 0.46442775 0.47105612 0.5       ]  |  48.05 (+/- 2.62)  |  10.34  |  0.008546  |
03/07/2020 06:25:09 PM - INFO - |  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  75.73 (+/- 2.09)  |  0.8659  |  0.008424  |
03/07/2020 06:25:09 PM - INFO - 

CURRENT CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 06:25:09 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 06:25:09 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 06:25:09 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  44.03%  |  [0.46973045 0.45779938 0.46531153 0.47017234 0.46993811]  |  46.66 (+/- 0.95)  |  18.29  |  0.9501  |
03/07/2020 06:25:09 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  44.57%  |  [0.4874061  0.48608042 0.49138312 0.49359258 0.48099027]  |  48.79 (+/- 0.88)  |  8.941  |  0.0056  |
03/07/2020 06:25:09 PM - INFO - |  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  75.73 (+/- 2.09)  |  0.8605  |  0.008546  |
03/07/2020 06:25:09 PM - INFO - 

03/07/2020 06:25:09 PM - INFO - ################################################################################
03/07/2020 06:25:09 PM - INFO - 4)
03/07/2020 06:25:09 PM - INFO - ********************************************************************************
03/07/2020 06:25:09 PM - INFO - Classifier: LOGISTIC_REGRESSION, Dataset: TWENTY_NEWS_GROUPS
03/07/2020 06:25:09 PM - INFO - ********************************************************************************
03/07/2020 06:25:12 PM - INFO - 

Performing grid search...

03/07/2020 06:25:12 PM - INFO - Parameters:
03/07/2020 06:25:12 PM - INFO - {'classifier__C': [1, 10], 'classifier__tol': [0.001, 0.01]}
03/07/2020 06:29:48 PM - INFO - 	Done in 276.248s
03/07/2020 06:29:48 PM - INFO - 	Best score: 0.750
03/07/2020 06:29:48 PM - INFO - 	Best parameters set:
03/07/2020 06:29:48 PM - INFO - 		classifier__C: 10
03/07/2020 06:29:48 PM - INFO - 		classifier__tol: 0.001
03/07/2020 06:29:48 PM - INFO - 

USING LOGISTIC_REGRESSION WITH DEFAULT PARAMETERS
03/07/2020 06:29:48 PM - INFO - ________________________________________________________________________________
03/07/2020 06:29:48 PM - INFO - Training: 
03/07/2020 06:29:48 PM - INFO - LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)
03/07/2020 06:30:18 PM - INFO - Train time: 29.425s
03/07/2020 06:30:18 PM - INFO - Test time:  0.012s
03/07/2020 06:30:18 PM - INFO - Accuracy score:   0.692
03/07/2020 06:30:18 PM - INFO - 

===> Classification Report:

03/07/2020 06:30:18 PM - INFO -               precision    recall  f1-score   support

           0       0.48      0.44      0.46       319
           1       0.62      0.70      0.66       389
           2       0.65      0.64      0.65       394
           3       0.69      0.65      0.67       392
           4       0.75      0.69      0.72       385
           5       0.83      0.72      0.77       395
           6       0.79      0.79      0.79       390
           7       0.77      0.72      0.74       396
           8       0.48      0.80      0.60       398
           9       0.80      0.83      0.82       397
          10       0.92      0.87      0.90       399
          11       0.88      0.67      0.76       396
          12       0.57      0.60      0.58       393
          13       0.78      0.79      0.78       396
          14       0.71      0.76      0.73       394
          15       0.60      0.83      0.70       398
          16       0.58      0.71      0.64       364
          17       0.86      0.77      0.81       376
          18       0.63      0.42      0.50       310
          19       0.55      0.14      0.22       251

    accuracy                           0.69      7532
   macro avg       0.70      0.68      0.67      7532
weighted avg       0.70      0.69      0.69      7532

03/07/2020 06:30:18 PM - INFO - 

Cross validation:
03/07/2020 06:31:09 PM - INFO - 	accuracy: 5-fold cross validation: [0.74458683 0.72646929 0.74193548 0.74768007 0.72413793]
03/07/2020 06:31:09 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 73.70 (+/- 1.94)
03/07/2020 06:31:09 PM - INFO - 

USING LOGISTIC_REGRESSION WITH BEST PARAMETERS: {'C': 10, 'tol': 0.001}
03/07/2020 06:31:09 PM - INFO - ________________________________________________________________________________
03/07/2020 06:31:09 PM - INFO - Training: 
03/07/2020 06:31:09 PM - INFO - LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.001, verbose=0,
                   warm_start=False)
03/07/2020 06:31:42 PM - INFO - Train time: 32.913s
03/07/2020 06:31:42 PM - INFO - Test time:  0.012s
03/07/2020 06:31:42 PM - INFO - Accuracy score:   0.693
03/07/2020 06:31:42 PM - INFO - 

===> Classification Report:

03/07/2020 06:31:42 PM - INFO -               precision    recall  f1-score   support

           0       0.50      0.46      0.48       319
           1       0.65      0.74      0.69       389
           2       0.62      0.60      0.61       394
           3       0.67      0.66      0.66       392
           4       0.73      0.69      0.71       385
           5       0.84      0.71      0.77       395
           6       0.78      0.77      0.78       390
           7       0.49      0.78      0.60       396
           8       0.77      0.77      0.77       398
           9       0.83      0.81      0.82       397
          10       0.90      0.85      0.88       399
          11       0.86      0.69      0.76       396
          12       0.60      0.61      0.61       393
          13       0.79      0.78      0.78       396
          14       0.74      0.74      0.74       394
          15       0.66      0.79      0.72       398
          16       0.58      0.69      0.63       364
          17       0.82      0.76      0.79       376
          18       0.57      0.44      0.50       310
          19       0.45      0.29      0.36       251

    accuracy                           0.69      7532
   macro avg       0.69      0.68      0.68      7532
weighted avg       0.70      0.69      0.69      7532

03/07/2020 06:31:42 PM - INFO - 

Cross validation:
03/07/2020 06:32:50 PM - INFO - 	accuracy: 5-fold cross validation: [0.75961114 0.74016792 0.75519222 0.7565179  0.73607427]
03/07/2020 06:32:50 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 74.95 (+/- 1.90)
03/07/2020 06:32:50 PM - INFO - It took 460.6458270549774 seconds
03/07/2020 06:32:50 PM - INFO - ********************************************************************************
03/07/2020 06:32:50 PM - INFO - ################################################################################
03/07/2020 06:32:50 PM - INFO - 

CURRENT CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 06:32:50 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 06:32:50 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 06:32:50 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  36.52%  |  [0.39460893 0.39019001 0.39991162 0.39019001 0.40539346]  |  39.61 (+/- 1.18)  |  4.611  |  0.2475  |
03/07/2020 06:32:50 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  42.82%  |  [0.49094123 0.47591692 0.46442775 0.47105612 0.5       ]  |  48.05 (+/- 2.62)  |  10.34  |  0.008546  |
03/07/2020 06:32:50 PM - INFO - |  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  75.73 (+/- 2.09)  |  0.8659  |  0.008424  |
03/07/2020 06:32:50 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  69.17%  |  [0.74458683 0.72646929 0.74193548 0.74768007 0.72413793]  |  73.70 (+/- 1.94)  |  29.42  |  0.01162  |
03/07/2020 06:32:50 PM - INFO - 

CURRENT CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 06:32:50 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 06:32:50 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 06:32:50 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  44.03%  |  [0.46973045 0.45779938 0.46531153 0.47017234 0.46993811]  |  46.66 (+/- 0.95)  |  18.29  |  0.9501  |
03/07/2020 06:32:50 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  44.57%  |  [0.4874061  0.48608042 0.49138312 0.49359258 0.48099027]  |  48.79 (+/- 0.88)  |  8.941  |  0.0056  |
03/07/2020 06:32:50 PM - INFO - |  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  75.73 (+/- 2.09)  |  0.8605  |  0.008546  |
03/07/2020 06:32:50 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  69.28%  |  [0.75961114 0.74016792 0.75519222 0.7565179  0.73607427]  |  74.95 (+/- 1.90)  |  32.91  |  0.01159  |
03/07/2020 06:32:50 PM - INFO - 

03/07/2020 06:32:50 PM - INFO - ################################################################################
03/07/2020 06:32:50 PM - INFO - 5)
03/07/2020 06:32:50 PM - INFO - ********************************************************************************
03/07/2020 06:32:50 PM - INFO - Classifier: RANDOM_FOREST_CLASSIFIER, Dataset: TWENTY_NEWS_GROUPS
03/07/2020 06:32:50 PM - INFO - ********************************************************************************
03/07/2020 06:32:53 PM - INFO - 

Performing grid search...

03/07/2020 06:32:53 PM - INFO - Parameters:
03/07/2020 06:32:53 PM - INFO - {'classifier__min_samples_leaf': [1, 2], 'classifier__min_samples_split': [2, 5], 'classifier__n_estimators': [100, 200]}
03/07/2020 06:38:32 PM - INFO - 	Done in 338.952s
03/07/2020 06:38:32 PM - INFO - 	Best score: 0.684
03/07/2020 06:38:32 PM - INFO - 	Best parameters set:
03/07/2020 06:38:32 PM - INFO - 		classifier__min_samples_leaf: 1
03/07/2020 06:38:32 PM - INFO - 		classifier__min_samples_split: 5
03/07/2020 06:38:32 PM - INFO - 		classifier__n_estimators: 200
03/07/2020 06:38:32 PM - INFO - 

USING RANDOM_FOREST_CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 06:38:32 PM - INFO - ________________________________________________________________________________
03/07/2020 06:38:32 PM - INFO - Training: 
03/07/2020 06:38:32 PM - INFO - RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
03/07/2020 06:39:07 PM - INFO - Train time: 35.026s
03/07/2020 06:39:07 PM - INFO - Test time:  0.600s
03/07/2020 06:39:07 PM - INFO - Accuracy score:   0.622
03/07/2020 06:39:07 PM - INFO - 

===> Classification Report:

03/07/2020 06:39:07 PM - INFO -               precision    recall  f1-score   support

           0       0.44      0.38      0.41       319
           1       0.59      0.59      0.59       389
           2       0.54      0.64      0.59       394
           3       0.61      0.59      0.60       392
           4       0.66      0.68      0.67       385
           5       0.66      0.69      0.67       395
           6       0.70      0.75      0.72       390
           7       0.40      0.70      0.51       396
           8       0.69      0.70      0.69       398
           9       0.71      0.77      0.74       397
          10       0.84      0.83      0.84       399
          11       0.78      0.66      0.71       396
          12       0.51      0.39      0.44       393
          13       0.74      0.63      0.68       396
          14       0.65      0.66      0.65       394
          15       0.57      0.78      0.66       398
          16       0.54      0.59      0.56       364
          17       0.83      0.69      0.76       376
          18       0.55      0.33      0.41       310
          19       0.28      0.07      0.11       251

    accuracy                           0.62      7532
   macro avg       0.61      0.61      0.60      7532
weighted avg       0.62      0.62      0.61      7532

03/07/2020 06:39:07 PM - INFO - 

Cross validation:
03/07/2020 06:39:57 PM - INFO - 	accuracy: 5-fold cross validation: [0.68360583 0.67034909 0.66725586 0.68272205 0.66534041]
03/07/2020 06:39:57 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 67.39 (+/- 1.55)
03/07/2020 06:39:57 PM - INFO - 

USING RANDOM_FOREST_CLASSIFIER WITH BEST PARAMETERS: {'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 200}
03/07/2020 06:39:57 PM - INFO - ________________________________________________________________________________
03/07/2020 06:39:57 PM - INFO - Training: 
03/07/2020 06:39:57 PM - INFO - RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=5,
                       min_weight_fraction_leaf=0.0, n_estimators=200,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
03/07/2020 06:40:37 PM - INFO - Train time: 39.544s
03/07/2020 06:40:38 PM - INFO - Test time:  1.178s
03/07/2020 06:40:38 PM - INFO - Accuracy score:   0.636
03/07/2020 06:40:38 PM - INFO - 

===> Classification Report:

03/07/2020 06:40:38 PM - INFO -               precision    recall  f1-score   support

           0       0.47      0.38      0.42       319
           1       0.61      0.61      0.61       389
           2       0.58      0.65      0.61       394
           3       0.61      0.60      0.60       392
           4       0.66      0.64      0.65       385
           5       0.68      0.70      0.69       395
           6       0.70      0.74      0.72       390
           7       0.43      0.71      0.53       396
           8       0.70      0.69      0.70       398
           9       0.72      0.78      0.75       397
          10       0.83      0.87      0.85       399
          11       0.80      0.66      0.73       396
          12       0.52      0.45      0.48       393
          13       0.74      0.67      0.70       396
          14       0.71      0.68      0.70       394
          15       0.58      0.80      0.68       398
          16       0.53      0.63      0.58       364
          17       0.86      0.71      0.78       376
          18       0.51      0.35      0.42       310
          19       0.34      0.10      0.16       251

    accuracy                           0.64      7532
   macro avg       0.63      0.62      0.62      7532
weighted avg       0.64      0.64      0.63      7532

03/07/2020 06:40:38 PM - INFO - 

Cross validation:
03/07/2020 06:41:45 PM - INFO - 	accuracy: 5-fold cross validation: [0.69067609 0.67255855 0.69067609 0.69730446 0.67241379]
03/07/2020 06:41:45 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 68.47 (+/- 2.06)
03/07/2020 06:41:45 PM - INFO - It took 535.5474169254303 seconds
03/07/2020 06:41:45 PM - INFO - ********************************************************************************
03/07/2020 06:41:45 PM - INFO - ################################################################################
03/07/2020 06:41:45 PM - INFO - 

CURRENT CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 06:41:45 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 06:41:45 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 06:41:45 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  36.52%  |  [0.39460893 0.39019001 0.39991162 0.39019001 0.40539346]  |  39.61 (+/- 1.18)  |  4.611  |  0.2475  |
03/07/2020 06:41:45 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  42.82%  |  [0.49094123 0.47591692 0.46442775 0.47105612 0.5       ]  |  48.05 (+/- 2.62)  |  10.34  |  0.008546  |
03/07/2020 06:41:45 PM - INFO - |  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  75.73 (+/- 2.09)  |  0.8659  |  0.008424  |
03/07/2020 06:41:45 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  69.17%  |  [0.74458683 0.72646929 0.74193548 0.74768007 0.72413793]  |  73.70 (+/- 1.94)  |  29.42  |  0.01162  |
03/07/2020 06:41:45 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  62.16%  |  [0.68360583 0.67034909 0.66725586 0.68272205 0.66534041]  |  67.39 (+/- 1.55)  |  35.03  |  0.6005  |
03/07/2020 06:41:45 PM - INFO - 

CURRENT CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 06:41:45 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 06:41:45 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 06:41:45 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  44.03%  |  [0.46973045 0.45779938 0.46531153 0.47017234 0.46993811]  |  46.66 (+/- 0.95)  |  18.29  |  0.9501  |
03/07/2020 06:41:45 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  44.57%  |  [0.4874061  0.48608042 0.49138312 0.49359258 0.48099027]  |  48.79 (+/- 0.88)  |  8.941  |  0.0056  |
03/07/2020 06:41:45 PM - INFO - |  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  75.73 (+/- 2.09)  |  0.8605  |  0.008546  |
03/07/2020 06:41:45 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  69.28%  |  [0.75961114 0.74016792 0.75519222 0.7565179  0.73607427]  |  74.95 (+/- 1.90)  |  32.91  |  0.01159  |
03/07/2020 06:41:45 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  63.64%  |  [0.69067609 0.67255855 0.69067609 0.69730446 0.67241379]  |  68.47 (+/- 2.06)  |  39.54  |  1.178  |
03/07/2020 06:41:45 PM - INFO - 

03/07/2020 06:41:45 PM - INFO - ################################################################################
03/07/2020 06:41:45 PM - INFO - 6)
03/07/2020 06:41:45 PM - INFO - ********************************************************************************
03/07/2020 06:41:45 PM - INFO - Classifier: BERNOULLI_NB, Dataset: TWENTY_NEWS_GROUPS
03/07/2020 06:41:45 PM - INFO - ********************************************************************************
03/07/2020 06:41:48 PM - INFO - 

Performing grid search...

03/07/2020 06:41:48 PM - INFO - Parameters:
03/07/2020 06:41:48 PM - INFO - {'classifier__alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0], 'classifier__binarize': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0], 'classifier__fit_prior': [False, True]}
03/07/2020 06:42:23 PM - INFO - 	Done in 34.795s
03/07/2020 06:42:23 PM - INFO - 	Best score: 0.680
03/07/2020 06:42:23 PM - INFO - 	Best parameters set:
03/07/2020 06:42:23 PM - INFO - 		classifier__alpha: 0.1
03/07/2020 06:42:23 PM - INFO - 		classifier__binarize: 0.1
03/07/2020 06:42:23 PM - INFO - 		classifier__fit_prior: False
03/07/2020 06:42:23 PM - INFO - 

USING BERNOULLI_NB WITH DEFAULT PARAMETERS
03/07/2020 06:42:23 PM - INFO - ________________________________________________________________________________
03/07/2020 06:42:23 PM - INFO - Training: 
03/07/2020 06:42:23 PM - INFO - BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
03/07/2020 06:42:23 PM - INFO - Train time: 0.055s
03/07/2020 06:42:23 PM - INFO - Test time:  0.053s
03/07/2020 06:42:23 PM - INFO - Accuracy score:   0.458
03/07/2020 06:42:23 PM - INFO - 

===> Classification Report:

03/07/2020 06:42:23 PM - INFO -               precision    recall  f1-score   support

           0       1.00      0.02      0.03       319
           1       0.62      0.46      0.53       389
           2       0.33      0.00      0.01       394
           3       0.42      0.74      0.54       392
           4       0.50      0.72      0.59       385
           5       0.82      0.40      0.54       395
           6       0.90      0.64      0.74       390
           7       0.39      0.71      0.50       396
           8       0.16      0.95      0.27       398
           9       0.66      0.80      0.72       397
          10       0.99      0.52      0.68       399
          11       0.75      0.36      0.49       396
          12       0.52      0.51      0.52       393
          13       0.90      0.35      0.50       396
          14       0.80      0.34      0.48       394
          15       0.45      0.62      0.52       398
          16       0.65      0.21      0.32       364
          17       0.93      0.38      0.54       376
          18       0.93      0.08      0.15       310
          19       0.00      0.00      0.00       251

    accuracy                           0.46      7532
   macro avg       0.64      0.44      0.43      7532
weighted avg       0.64      0.46      0.45      7532

03/07/2020 06:42:23 PM - INFO - 

Cross validation:
03/07/2020 06:42:23 PM - INFO - 	accuracy: 5-fold cross validation: [0.48033584 0.46133451 0.48563853 0.47547503 0.48938992]
03/07/2020 06:42:23 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 47.84 (+/- 1.95)
03/07/2020 06:42:23 PM - INFO - 

USING BERNOULLI_NB WITH BEST PARAMETERS: {'alpha': 0.1, 'binarize': 0.1, 'fit_prior': False}
03/07/2020 06:42:23 PM - INFO - ________________________________________________________________________________
03/07/2020 06:42:23 PM - INFO - Training: 
03/07/2020 06:42:23 PM - INFO - BernoulliNB(alpha=0.1, binarize=0.1, class_prior=None, fit_prior=False)
03/07/2020 06:42:23 PM - INFO - Train time: 0.054s
03/07/2020 06:42:23 PM - INFO - Test time:  0.052s
03/07/2020 06:42:23 PM - INFO - Accuracy score:   0.626
03/07/2020 06:42:23 PM - INFO - 

===> Classification Report:

03/07/2020 06:42:23 PM - INFO -               precision    recall  f1-score   support

           0       0.45      0.39      0.42       319
           1       0.51      0.63      0.57       389
           2       0.50      0.58      0.54       394
           3       0.61      0.64      0.62       392
           4       0.64      0.65      0.65       385
           5       0.80      0.63      0.70       395
           6       0.86      0.65      0.74       390
           7       0.67      0.72      0.69       396
           8       0.67      0.72      0.70       398
           9       0.35      0.87      0.49       397
          10       0.90      0.83      0.86       399
          11       0.79      0.69      0.73       396
          12       0.70      0.51      0.59       393
          13       0.86      0.62      0.72       396
          14       0.70      0.71      0.70       394
          15       0.61      0.71      0.66       398
          16       0.57      0.64      0.60       364
          17       0.76      0.62      0.68       376
          18       0.64      0.30      0.41       310
          19       0.40      0.09      0.15       251

    accuracy                           0.63      7532
   macro avg       0.65      0.61      0.61      7532
weighted avg       0.66      0.63      0.62      7532

03/07/2020 06:42:23 PM - INFO - 

Cross validation:
03/07/2020 06:42:24 PM - INFO - 	accuracy: 5-fold cross validation: [0.69200177 0.67741935 0.6752099  0.67388422 0.67992927]
03/07/2020 06:42:24 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 67.97 (+/- 1.30)
03/07/2020 06:42:24 PM - INFO - It took 38.404728412628174 seconds
03/07/2020 06:42:24 PM - INFO - ********************************************************************************
03/07/2020 06:42:24 PM - INFO - ################################################################################
03/07/2020 06:42:24 PM - INFO - 

CURRENT CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 06:42:24 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 06:42:24 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 06:42:24 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  36.52%  |  [0.39460893 0.39019001 0.39991162 0.39019001 0.40539346]  |  39.61 (+/- 1.18)  |  4.611  |  0.2475  |
03/07/2020 06:42:24 PM - INFO - |  2  |  BERNOULLI_NB  |  45.84%  |  [0.48033584 0.46133451 0.48563853 0.47547503 0.48938992]  |  47.84 (+/- 1.95)  |  0.05548  |  0.05272  |
03/07/2020 06:42:24 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  42.82%  |  [0.49094123 0.47591692 0.46442775 0.47105612 0.5       ]  |  48.05 (+/- 2.62)  |  10.34  |  0.008546  |
03/07/2020 06:42:24 PM - INFO - |  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  75.73 (+/- 2.09)  |  0.8659  |  0.008424  |
03/07/2020 06:42:24 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  69.17%  |  [0.74458683 0.72646929 0.74193548 0.74768007 0.72413793]  |  73.70 (+/- 1.94)  |  29.42  |  0.01162  |
03/07/2020 06:42:24 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  62.16%  |  [0.68360583 0.67034909 0.66725586 0.68272205 0.66534041]  |  67.39 (+/- 1.55)  |  35.03  |  0.6005  |
03/07/2020 06:42:24 PM - INFO - 

CURRENT CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 06:42:24 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 06:42:24 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 06:42:24 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  44.03%  |  [0.46973045 0.45779938 0.46531153 0.47017234 0.46993811]  |  46.66 (+/- 0.95)  |  18.29  |  0.9501  |
03/07/2020 06:42:24 PM - INFO - |  2  |  BERNOULLI_NB  |  62.56%  |  [0.69200177 0.67741935 0.6752099  0.67388422 0.67992927]  |  67.97 (+/- 1.30)  |  0.05404  |  0.05173  |
03/07/2020 06:42:24 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  44.57%  |  [0.4874061  0.48608042 0.49138312 0.49359258 0.48099027]  |  48.79 (+/- 0.88)  |  8.941  |  0.0056  |
03/07/2020 06:42:24 PM - INFO - |  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  75.73 (+/- 2.09)  |  0.8605  |  0.008546  |
03/07/2020 06:42:24 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  69.28%  |  [0.75961114 0.74016792 0.75519222 0.7565179  0.73607427]  |  74.95 (+/- 1.90)  |  32.91  |  0.01159  |
03/07/2020 06:42:24 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  63.64%  |  [0.69067609 0.67255855 0.69067609 0.69730446 0.67241379]  |  68.47 (+/- 2.06)  |  39.54  |  1.178  |
03/07/2020 06:42:24 PM - INFO - 

03/07/2020 06:42:24 PM - INFO - ################################################################################
03/07/2020 06:42:24 PM - INFO - 7)
03/07/2020 06:42:24 PM - INFO - ********************************************************************************
03/07/2020 06:42:24 PM - INFO - Classifier: COMPLEMENT_NB, Dataset: TWENTY_NEWS_GROUPS
03/07/2020 06:42:24 PM - INFO - ********************************************************************************
03/07/2020 06:42:27 PM - INFO - 

Performing grid search...

03/07/2020 06:42:27 PM - INFO - Parameters:
03/07/2020 06:42:27 PM - INFO - {'classifier__alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0], 'classifier__fit_prior': [False, True], 'classifier__norm': [False, True]}
03/07/2020 06:42:33 PM - INFO - 	Done in 6.314s
03/07/2020 06:42:33 PM - INFO - 	Best score: 0.772
03/07/2020 06:42:33 PM - INFO - 	Best parameters set:
03/07/2020 06:42:33 PM - INFO - 		classifier__alpha: 0.5
03/07/2020 06:42:33 PM - INFO - 		classifier__fit_prior: False
03/07/2020 06:42:33 PM - INFO - 		classifier__norm: False
03/07/2020 06:42:33 PM - INFO - 

USING COMPLEMENT_NB WITH DEFAULT PARAMETERS
03/07/2020 06:42:33 PM - INFO - ________________________________________________________________________________
03/07/2020 06:42:33 PM - INFO - Training: 
03/07/2020 06:42:33 PM - INFO - ComplementNB(alpha=1.0, class_prior=None, fit_prior=True, norm=False)
03/07/2020 06:42:33 PM - INFO - Train time: 0.063s
03/07/2020 06:42:33 PM - INFO - Test time:  0.010s
03/07/2020 06:42:33 PM - INFO - Accuracy score:   0.710
03/07/2020 06:42:33 PM - INFO - 

===> Classification Report:

03/07/2020 06:42:33 PM - INFO -               precision    recall  f1-score   support

           0       0.30      0.39      0.34       319
           1       0.72      0.68      0.70       389
           2       0.72      0.55      0.63       394
           3       0.63      0.72      0.67       392
           4       0.77      0.73      0.75       385
           5       0.78      0.81      0.79       395
           6       0.77      0.74      0.76       390
           7       0.85      0.73      0.79       396
           8       0.84      0.76      0.80       398
           9       0.92      0.84      0.88       397
          10       0.86      0.94      0.90       399
          11       0.73      0.80      0.76       396
          12       0.70      0.56      0.62       393
          13       0.80      0.81      0.81       396
          14       0.80      0.80      0.80       394
          15       0.52      0.91      0.66       398
          16       0.58      0.72      0.65       364
          17       0.75      0.86      0.80       376
          18       0.70      0.41      0.51       310
          19       0.48      0.10      0.16       251

    accuracy                           0.71      7532
   macro avg       0.71      0.69      0.69      7532
weighted avg       0.72      0.71      0.70      7532

03/07/2020 06:42:33 PM - INFO - 

Cross validation:
03/07/2020 06:42:33 PM - INFO - 	accuracy: 5-fold cross validation: [0.77110031 0.75740168 0.77993814 0.7715422  0.76038904]
03/07/2020 06:42:33 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 76.81 (+/- 1.64)
03/07/2020 06:42:33 PM - INFO - 

USING COMPLEMENT_NB WITH BEST PARAMETERS: {'alpha': 0.5, 'fit_prior': False, 'norm': False}
03/07/2020 06:42:33 PM - INFO - ________________________________________________________________________________
03/07/2020 06:42:33 PM - INFO - Training: 
03/07/2020 06:42:33 PM - INFO - ComplementNB(alpha=0.5, class_prior=None, fit_prior=False, norm=False)
03/07/2020 06:42:33 PM - INFO - Train time: 0.067s
03/07/2020 06:42:33 PM - INFO - Test time:  0.011s
03/07/2020 06:42:33 PM - INFO - Accuracy score:   0.712
03/07/2020 06:42:33 PM - INFO - 

===> Classification Report:

03/07/2020 06:42:33 PM - INFO -               precision    recall  f1-score   support

           0       0.31      0.42      0.35       319
           1       0.72      0.69      0.70       389
           2       0.74      0.54      0.62       394
           3       0.64      0.72      0.68       392
           4       0.77      0.73      0.75       385
           5       0.77      0.80      0.79       395
           6       0.76      0.74      0.75       390
           7       0.83      0.75      0.79       396
           8       0.82      0.76      0.79       398
           9       0.90      0.85      0.88       397
          10       0.87      0.94      0.90       399
          11       0.74      0.80      0.77       396
          12       0.71      0.56      0.63       393
          13       0.78      0.81      0.79       396
          14       0.81      0.80      0.80       394
          15       0.55      0.91      0.68       398
          16       0.59      0.72      0.65       364
          17       0.76      0.85      0.80       376
          18       0.68      0.42      0.52       310
          19       0.46      0.11      0.17       251

    accuracy                           0.71      7532
   macro avg       0.71      0.70      0.69      7532
weighted avg       0.72      0.71      0.71      7532

03/07/2020 06:42:33 PM - INFO - 

Cross validation:
03/07/2020 06:42:33 PM - INFO - 	accuracy: 5-fold cross validation: [0.776403   0.75828546 0.78126381 0.77507733 0.76967286]
03/07/2020 06:42:33 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 77.21 (+/- 1.57)
03/07/2020 06:42:33 PM - INFO - It took 9.795810461044312 seconds
03/07/2020 06:42:33 PM - INFO - ********************************************************************************
03/07/2020 06:42:33 PM - INFO - ################################################################################
03/07/2020 06:42:33 PM - INFO - 

CURRENT CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 06:42:33 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 06:42:33 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 06:42:33 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  36.52%  |  [0.39460893 0.39019001 0.39991162 0.39019001 0.40539346]  |  39.61 (+/- 1.18)  |  4.611  |  0.2475  |
03/07/2020 06:42:33 PM - INFO - |  2  |  BERNOULLI_NB  |  45.84%  |  [0.48033584 0.46133451 0.48563853 0.47547503 0.48938992]  |  47.84 (+/- 1.95)  |  0.05548  |  0.05272  |
03/07/2020 06:42:33 PM - INFO - |  3  |  COMPLEMENT_NB  |  71.04%  |  [0.77110031 0.75740168 0.77993814 0.7715422  0.76038904]  |  76.81 (+/- 1.64)  |  0.06326  |  0.01035  |
03/07/2020 06:42:33 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  42.82%  |  [0.49094123 0.47591692 0.46442775 0.47105612 0.5       ]  |  48.05 (+/- 2.62)  |  10.34  |  0.008546  |
03/07/2020 06:42:33 PM - INFO - |  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  75.73 (+/- 2.09)  |  0.8659  |  0.008424  |
03/07/2020 06:42:33 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  69.17%  |  [0.74458683 0.72646929 0.74193548 0.74768007 0.72413793]  |  73.70 (+/- 1.94)  |  29.42  |  0.01162  |
03/07/2020 06:42:33 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  62.16%  |  [0.68360583 0.67034909 0.66725586 0.68272205 0.66534041]  |  67.39 (+/- 1.55)  |  35.03  |  0.6005  |
03/07/2020 06:42:33 PM - INFO - 

CURRENT CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 06:42:33 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 06:42:33 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 06:42:33 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  44.03%  |  [0.46973045 0.45779938 0.46531153 0.47017234 0.46993811]  |  46.66 (+/- 0.95)  |  18.29  |  0.9501  |
03/07/2020 06:42:33 PM - INFO - |  2  |  BERNOULLI_NB  |  62.56%  |  [0.69200177 0.67741935 0.6752099  0.67388422 0.67992927]  |  67.97 (+/- 1.30)  |  0.05404  |  0.05173  |
03/07/2020 06:42:33 PM - INFO - |  3  |  COMPLEMENT_NB  |  71.22%  |  [0.776403   0.75828546 0.78126381 0.77507733 0.76967286]  |  77.21 (+/- 1.57)  |  0.06653  |  0.01054  |
03/07/2020 06:42:33 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  44.57%  |  [0.4874061  0.48608042 0.49138312 0.49359258 0.48099027]  |  48.79 (+/- 0.88)  |  8.941  |  0.0056  |
03/07/2020 06:42:33 PM - INFO - |  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  75.73 (+/- 2.09)  |  0.8605  |  0.008546  |
03/07/2020 06:42:33 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  69.28%  |  [0.75961114 0.74016792 0.75519222 0.7565179  0.73607427]  |  74.95 (+/- 1.90)  |  32.91  |  0.01159  |
03/07/2020 06:42:33 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  63.64%  |  [0.69067609 0.67255855 0.69067609 0.69730446 0.67241379]  |  68.47 (+/- 2.06)  |  39.54  |  1.178  |
03/07/2020 06:42:33 PM - INFO - 

03/07/2020 06:42:33 PM - INFO - ################################################################################
03/07/2020 06:42:33 PM - INFO - 8)
03/07/2020 06:42:34 PM - INFO - ********************************************************************************
03/07/2020 06:42:34 PM - INFO - Classifier: MULTINOMIAL_NB, Dataset: TWENTY_NEWS_GROUPS
03/07/2020 06:42:34 PM - INFO - ********************************************************************************
03/07/2020 06:42:36 PM - INFO - 

Performing grid search...

03/07/2020 06:42:36 PM - INFO - Parameters:
03/07/2020 06:42:36 PM - INFO - {'classifier__alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0], 'classifier__fit_prior': [False, True]}
03/07/2020 06:42:39 PM - INFO - 	Done in 2.922s
03/07/2020 06:42:39 PM - INFO - 	Best score: 0.751
03/07/2020 06:42:39 PM - INFO - 	Best parameters set:
03/07/2020 06:42:39 PM - INFO - 		classifier__alpha: 0.01
03/07/2020 06:42:39 PM - INFO - 		classifier__fit_prior: True
03/07/2020 06:42:39 PM - INFO - 

USING MULTINOMIAL_NB WITH DEFAULT PARAMETERS
03/07/2020 06:42:39 PM - INFO - ________________________________________________________________________________
03/07/2020 06:42:39 PM - INFO - Training: 
03/07/2020 06:42:39 PM - INFO - MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
03/07/2020 06:42:39 PM - INFO - Train time: 0.044s
03/07/2020 06:42:39 PM - INFO - Test time:  0.010s
03/07/2020 06:42:39 PM - INFO - Accuracy score:   0.656
03/07/2020 06:42:39 PM - INFO - 

===> Classification Report:

03/07/2020 06:42:39 PM - INFO -               precision    recall  f1-score   support

           0       0.82      0.13      0.22       319
           1       0.67      0.63      0.65       389
           2       0.68      0.50      0.58       394
           3       0.57      0.78      0.66       392
           4       0.80      0.63      0.70       385
           5       0.75      0.80      0.77       395
           6       0.80      0.76      0.78       390
           7       0.83      0.71      0.77       396
           8       0.87      0.69      0.77       398
           9       0.92      0.77      0.84       397
          10       0.57      0.94      0.71       399
          11       0.52      0.80      0.63       396
          12       0.74      0.50      0.60       393
          13       0.86      0.73      0.79       396
          14       0.77      0.72      0.75       394
          15       0.34      0.93      0.50       398
          16       0.59      0.68      0.63       364
          17       0.76      0.79      0.78       376
          18       0.90      0.19      0.32       310
          19       1.00      0.01      0.02       251

    accuracy                           0.66      7532
   macro avg       0.74      0.63      0.62      7532
weighted avg       0.73      0.66      0.64      7532

03/07/2020 06:42:39 PM - INFO - 

Cross validation:
03/07/2020 06:42:40 PM - INFO - 	accuracy: 5-fold cross validation: [0.7105612  0.69951392 0.69863014 0.69995581 0.69142352]
03/07/2020 06:42:40 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 70.00 (+/- 1.22)
03/07/2020 06:42:40 PM - INFO - 

USING MULTINOMIAL_NB WITH BEST PARAMETERS: {'alpha': 0.01, 'fit_prior': True}
03/07/2020 06:42:40 PM - INFO - ________________________________________________________________________________
03/07/2020 06:42:40 PM - INFO - Training: 
03/07/2020 06:42:40 PM - INFO - MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True)
03/07/2020 06:42:40 PM - INFO - Train time: 0.057s
03/07/2020 06:42:40 PM - INFO - Test time:  0.010s
03/07/2020 06:42:40 PM - INFO - Accuracy score:   0.688
03/07/2020 06:42:40 PM - INFO - 

===> Classification Report:

03/07/2020 06:42:40 PM - INFO -               precision    recall  f1-score   support

           0       0.57      0.43      0.49       319
           1       0.64      0.71      0.67       389
           2       0.74      0.43      0.54       394
           3       0.57      0.73      0.64       392
           4       0.71      0.67      0.69       385
           5       0.77      0.75      0.76       395
           6       0.81      0.71      0.76       390
           7       0.76      0.72      0.74       396
           8       0.76      0.71      0.73       398
           9       0.91      0.80      0.85       397
          10       0.59      0.93      0.72       399
          11       0.70      0.76      0.73       396
          12       0.74      0.57      0.64       393
          13       0.83      0.76      0.80       396
          14       0.73      0.79      0.76       394
          15       0.57      0.86      0.69       398
          16       0.56      0.72      0.63       364
          17       0.81      0.78      0.80       376
          18       0.56      0.44      0.49       310
          19       0.45      0.20      0.28       251

    accuracy                           0.69      7532
   macro avg       0.69      0.67      0.67      7532
weighted avg       0.70      0.69      0.68      7532

03/07/2020 06:42:40 PM - INFO - 

Cross validation:
03/07/2020 06:42:40 PM - INFO - 	accuracy: 5-fold cross validation: [0.76049492 0.74149359 0.75519222 0.75695979 0.74270557]
03/07/2020 06:42:40 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 75.14 (+/- 1.55)
03/07/2020 06:42:40 PM - INFO - It took 6.348746299743652 seconds
03/07/2020 06:42:40 PM - INFO - ********************************************************************************
03/07/2020 06:42:40 PM - INFO - ################################################################################
03/07/2020 06:42:40 PM - INFO - 

CURRENT CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 06:42:40 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 06:42:40 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 06:42:40 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  36.52%  |  [0.39460893 0.39019001 0.39991162 0.39019001 0.40539346]  |  39.61 (+/- 1.18)  |  4.611  |  0.2475  |
03/07/2020 06:42:40 PM - INFO - |  2  |  BERNOULLI_NB  |  45.84%  |  [0.48033584 0.46133451 0.48563853 0.47547503 0.48938992]  |  47.84 (+/- 1.95)  |  0.05548  |  0.05272  |
03/07/2020 06:42:40 PM - INFO - |  3  |  COMPLEMENT_NB  |  71.04%  |  [0.77110031 0.75740168 0.77993814 0.7715422  0.76038904]  |  76.81 (+/- 1.64)  |  0.06326  |  0.01035  |
03/07/2020 06:42:40 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  42.82%  |  [0.49094123 0.47591692 0.46442775 0.47105612 0.5       ]  |  48.05 (+/- 2.62)  |  10.34  |  0.008546  |
03/07/2020 06:42:40 PM - INFO - |  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  75.73 (+/- 2.09)  |  0.8659  |  0.008424  |
03/07/2020 06:42:40 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  69.17%  |  [0.74458683 0.72646929 0.74193548 0.74768007 0.72413793]  |  73.70 (+/- 1.94)  |  29.42  |  0.01162  |
03/07/2020 06:42:40 PM - INFO - |  9  |  MULTINOMIAL_NB  |  65.64%  |  [0.7105612  0.69951392 0.69863014 0.69995581 0.69142352]  |  70.00 (+/- 1.22)  |  0.04385  |  0.01045  |
03/07/2020 06:42:40 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  62.16%  |  [0.68360583 0.67034909 0.66725586 0.68272205 0.66534041]  |  67.39 (+/- 1.55)  |  35.03  |  0.6005  |
03/07/2020 06:42:40 PM - INFO - 

CURRENT CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 06:42:40 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 06:42:40 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 06:42:40 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  44.03%  |  [0.46973045 0.45779938 0.46531153 0.47017234 0.46993811]  |  46.66 (+/- 0.95)  |  18.29  |  0.9501  |
03/07/2020 06:42:40 PM - INFO - |  2  |  BERNOULLI_NB  |  62.56%  |  [0.69200177 0.67741935 0.6752099  0.67388422 0.67992927]  |  67.97 (+/- 1.30)  |  0.05404  |  0.05173  |
03/07/2020 06:42:40 PM - INFO - |  3  |  COMPLEMENT_NB  |  71.22%  |  [0.776403   0.75828546 0.78126381 0.77507733 0.76967286]  |  77.21 (+/- 1.57)  |  0.06653  |  0.01054  |
03/07/2020 06:42:40 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  44.57%  |  [0.4874061  0.48608042 0.49138312 0.49359258 0.48099027]  |  48.79 (+/- 0.88)  |  8.941  |  0.0056  |
03/07/2020 06:42:40 PM - INFO - |  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  75.73 (+/- 2.09)  |  0.8605  |  0.008546  |
03/07/2020 06:42:40 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  69.28%  |  [0.75961114 0.74016792 0.75519222 0.7565179  0.73607427]  |  74.95 (+/- 1.90)  |  32.91  |  0.01159  |
03/07/2020 06:42:40 PM - INFO - |  9  |  MULTINOMIAL_NB  |  68.79%  |  [0.76049492 0.74149359 0.75519222 0.75695979 0.74270557]  |  75.14 (+/- 1.55)  |  0.05682  |  0.01024  |
03/07/2020 06:42:40 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  63.64%  |  [0.69067609 0.67255855 0.69067609 0.69730446 0.67241379]  |  68.47 (+/- 2.06)  |  39.54  |  1.178  |
03/07/2020 06:42:40 PM - INFO - 

03/07/2020 06:42:40 PM - INFO - ################################################################################
03/07/2020 06:42:40 PM - INFO - 9)
03/07/2020 06:42:40 PM - INFO - ********************************************************************************
03/07/2020 06:42:40 PM - INFO - Classifier: NEAREST_CENTROID, Dataset: TWENTY_NEWS_GROUPS
03/07/2020 06:42:40 PM - INFO - ********************************************************************************
03/07/2020 06:42:43 PM - INFO - 

Performing grid search...

03/07/2020 06:42:43 PM - INFO - Parameters:
03/07/2020 06:42:43 PM - INFO - {'classifier__metric': ['euclidean', 'cosine']}
03/07/2020 06:42:43 PM - INFO - 	Done in 0.197s
03/07/2020 06:42:43 PM - INFO - 	Best score: 0.716
03/07/2020 06:42:43 PM - INFO - 	Best parameters set:
03/07/2020 06:42:43 PM - INFO - 		classifier__metric: 'cosine'
03/07/2020 06:42:43 PM - INFO - 

USING NEAREST_CENTROID WITH DEFAULT PARAMETERS
03/07/2020 06:42:43 PM - INFO - ________________________________________________________________________________
03/07/2020 06:42:43 PM - INFO - Training: 
03/07/2020 06:42:43 PM - INFO - NearestCentroid(metric='euclidean', shrink_threshold=None)
03/07/2020 06:42:43 PM - INFO - Train time: 0.019s
03/07/2020 06:42:43 PM - INFO - Test time:  0.014s
03/07/2020 06:42:43 PM - INFO - Accuracy score:   0.649
03/07/2020 06:42:43 PM - INFO - 

===> Classification Report:

03/07/2020 06:42:43 PM - INFO -               precision    recall  f1-score   support

           0       0.39      0.43      0.41       319
           1       0.53      0.68      0.59       389
           2       0.67      0.60      0.63       394
           3       0.67      0.61      0.64       392
           4       0.80      0.65      0.72       385
           5       0.88      0.65      0.75       395
           6       0.83      0.75      0.78       390
           7       0.71      0.71      0.71       396
           8       0.43      0.77      0.55       398
           9       0.88      0.78      0.83       397
          10       0.97      0.80      0.88       399
          11       0.92      0.59      0.72       396
          12       0.42      0.65      0.51       393
          13       0.94      0.49      0.65       396
          14       0.68      0.74      0.71       394
          15       0.68      0.70      0.69       398
          16       0.57      0.68      0.62       364
          17       0.93      0.70      0.80       376
          18       0.42      0.49      0.45       310
          19       0.33      0.32      0.33       251

    accuracy                           0.65      7532
   macro avg       0.68      0.64      0.65      7532
weighted avg       0.69      0.65      0.66      7532

03/07/2020 06:42:43 PM - INFO - 

Cross validation:
03/07/2020 06:42:43 PM - INFO - 	accuracy: 5-fold cross validation: [0.69774635 0.69244366 0.69818825 0.70393283 0.69849691]
03/07/2020 06:42:43 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 69.82 (+/- 0.73)
03/07/2020 06:42:43 PM - INFO - 

USING NEAREST_CENTROID WITH BEST PARAMETERS: {'metric': 'cosine'}
03/07/2020 06:42:43 PM - INFO - ________________________________________________________________________________
03/07/2020 06:42:43 PM - INFO - Training: 
03/07/2020 06:42:43 PM - INFO - NearestCentroid(metric='cosine', shrink_threshold=None)
03/07/2020 06:42:43 PM - INFO - Train time: 0.017s
03/07/2020 06:42:43 PM - INFO - Test time:  0.020s
03/07/2020 06:42:43 PM - INFO - Accuracy score:   0.667
03/07/2020 06:42:43 PM - INFO - 

===> Classification Report:

03/07/2020 06:42:43 PM - INFO -               precision    recall  f1-score   support

           0       0.26      0.48      0.34       319
           1       0.53      0.68      0.59       389
           2       0.64      0.60      0.62       394
           3       0.64      0.64      0.64       392
           4       0.76      0.68      0.72       385
           5       0.85      0.70      0.76       395
           6       0.78      0.78      0.78       390
           7       0.78      0.70      0.74       396
           8       0.85      0.71      0.77       398
           9       0.90      0.79      0.84       397
          10       0.95      0.86      0.90       399
          11       0.84      0.66      0.74       396
          12       0.57      0.59      0.58       393
          13       0.90      0.56      0.69       396
          14       0.72      0.73      0.73       394
          15       0.62      0.79      0.69       398
          16       0.55      0.69      0.62       364
          17       0.87      0.73      0.79       376
          18       0.41      0.48      0.44       310
          19       0.35      0.28      0.31       251

    accuracy                           0.67      7532
   macro avg       0.69      0.66      0.66      7532
weighted avg       0.70      0.67      0.68      7532

03/07/2020 06:42:43 PM - INFO - 

Cross validation:
03/07/2020 06:42:43 PM - INFO - 	accuracy: 5-fold cross validation: [0.71674768 0.71807335 0.71851525 0.71939903 0.70601238]
03/07/2020 06:42:43 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 71.57 (+/- 0.99)
03/07/2020 06:42:43 PM - INFO - It took 3.4531943798065186 seconds
03/07/2020 06:42:43 PM - INFO - ********************************************************************************
03/07/2020 06:42:43 PM - INFO - ################################################################################
03/07/2020 06:42:43 PM - INFO - 

CURRENT CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 06:42:43 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 06:42:43 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 06:42:43 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  36.52%  |  [0.39460893 0.39019001 0.39991162 0.39019001 0.40539346]  |  39.61 (+/- 1.18)  |  4.611  |  0.2475  |
03/07/2020 06:42:43 PM - INFO - |  2  |  BERNOULLI_NB  |  45.84%  |  [0.48033584 0.46133451 0.48563853 0.47547503 0.48938992]  |  47.84 (+/- 1.95)  |  0.05548  |  0.05272  |
03/07/2020 06:42:43 PM - INFO - |  3  |  COMPLEMENT_NB  |  71.04%  |  [0.77110031 0.75740168 0.77993814 0.7715422  0.76038904]  |  76.81 (+/- 1.64)  |  0.06326  |  0.01035  |
03/07/2020 06:42:43 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  42.82%  |  [0.49094123 0.47591692 0.46442775 0.47105612 0.5       ]  |  48.05 (+/- 2.62)  |  10.34  |  0.008546  |
03/07/2020 06:42:43 PM - INFO - |  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  75.73 (+/- 2.09)  |  0.8659  |  0.008424  |
03/07/2020 06:42:43 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  69.17%  |  [0.74458683 0.72646929 0.74193548 0.74768007 0.72413793]  |  73.70 (+/- 1.94)  |  29.42  |  0.01162  |
03/07/2020 06:42:43 PM - INFO - |  9  |  MULTINOMIAL_NB  |  65.64%  |  [0.7105612  0.69951392 0.69863014 0.69995581 0.69142352]  |  70.00 (+/- 1.22)  |  0.04385  |  0.01045  |
03/07/2020 06:42:43 PM - INFO - |  10  |  NEAREST_CENTROID  |  64.91%  |  [0.69774635 0.69244366 0.69818825 0.70393283 0.69849691]  |  69.82 (+/- 0.73)  |  0.01918  |  0.01427  |
03/07/2020 06:42:43 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  62.16%  |  [0.68360583 0.67034909 0.66725586 0.68272205 0.66534041]  |  67.39 (+/- 1.55)  |  35.03  |  0.6005  |
03/07/2020 06:42:43 PM - INFO - 

CURRENT CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 06:42:43 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 06:42:43 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 06:42:43 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  44.03%  |  [0.46973045 0.45779938 0.46531153 0.47017234 0.46993811]  |  46.66 (+/- 0.95)  |  18.29  |  0.9501  |
03/07/2020 06:42:43 PM - INFO - |  2  |  BERNOULLI_NB  |  62.56%  |  [0.69200177 0.67741935 0.6752099  0.67388422 0.67992927]  |  67.97 (+/- 1.30)  |  0.05404  |  0.05173  |
03/07/2020 06:42:43 PM - INFO - |  3  |  COMPLEMENT_NB  |  71.22%  |  [0.776403   0.75828546 0.78126381 0.77507733 0.76967286]  |  77.21 (+/- 1.57)  |  0.06653  |  0.01054  |
03/07/2020 06:42:43 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  44.57%  |  [0.4874061  0.48608042 0.49138312 0.49359258 0.48099027]  |  48.79 (+/- 0.88)  |  8.941  |  0.0056  |
03/07/2020 06:42:43 PM - INFO - |  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  75.73 (+/- 2.09)  |  0.8605  |  0.008546  |
03/07/2020 06:42:43 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  69.28%  |  [0.75961114 0.74016792 0.75519222 0.7565179  0.73607427]  |  74.95 (+/- 1.90)  |  32.91  |  0.01159  |
03/07/2020 06:42:43 PM - INFO - |  9  |  MULTINOMIAL_NB  |  68.79%  |  [0.76049492 0.74149359 0.75519222 0.75695979 0.74270557]  |  75.14 (+/- 1.55)  |  0.05682  |  0.01024  |
03/07/2020 06:42:43 PM - INFO - |  10  |  NEAREST_CENTROID  |  66.70%  |  [0.71674768 0.71807335 0.71851525 0.71939903 0.70601238]  |  71.57 (+/- 0.99)  |  0.01737  |  0.01961  |
03/07/2020 06:42:43 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  63.64%  |  [0.69067609 0.67255855 0.69067609 0.69730446 0.67241379]  |  68.47 (+/- 2.06)  |  39.54  |  1.178  |
03/07/2020 06:42:43 PM - INFO - 

03/07/2020 06:42:43 PM - INFO - ################################################################################
03/07/2020 06:42:43 PM - INFO - 10)
03/07/2020 06:42:43 PM - INFO - ********************************************************************************
03/07/2020 06:42:43 PM - INFO - Classifier: PASSIVE_AGGRESSIVE_CLASSIFIER, Dataset: TWENTY_NEWS_GROUPS
03/07/2020 06:42:43 PM - INFO - ********************************************************************************
03/07/2020 06:42:46 PM - INFO - 

Performing grid search...

03/07/2020 06:42:46 PM - INFO - Parameters:
03/07/2020 06:42:46 PM - INFO - {'classifier__C': [0.01, 1.0], 'classifier__early_stopping': [False, True], 'classifier__tol': [0.0001, 0.001, 0.01], 'classifier__validation_fraction': [0.0001, 0.01]}
03/07/2020 06:43:36 PM - INFO - 	Done in 49.547s
03/07/2020 06:43:36 PM - INFO - 	Best score: 0.759
03/07/2020 06:43:36 PM - INFO - 	Best parameters set:
03/07/2020 06:43:36 PM - INFO - 		classifier__C: 0.01
03/07/2020 06:43:36 PM - INFO - 		classifier__early_stopping: False
03/07/2020 06:43:36 PM - INFO - 		classifier__tol: 0.0001
03/07/2020 06:43:36 PM - INFO - 		classifier__validation_fraction: 0.0001
03/07/2020 06:43:36 PM - INFO - 

USING PASSIVE_AGGRESSIVE_CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 06:43:36 PM - INFO - ________________________________________________________________________________
03/07/2020 06:43:36 PM - INFO - Training: 
03/07/2020 06:43:36 PM - INFO - PassiveAggressiveClassifier(C=1.0, average=False, class_weight=None,
                            early_stopping=False, fit_intercept=True,
                            loss='hinge', max_iter=1000, n_iter_no_change=5,
                            n_jobs=None, random_state=None, shuffle=True,
                            tol=0.001, validation_fraction=0.1, verbose=0,
                            warm_start=False)
03/07/2020 06:43:37 PM - INFO - Train time: 1.037s
03/07/2020 06:43:37 PM - INFO - Test time:  0.024s
03/07/2020 06:43:37 PM - INFO - Accuracy score:   0.683
03/07/2020 06:43:37 PM - INFO - 

===> Classification Report:

03/07/2020 06:43:37 PM - INFO -               precision    recall  f1-score   support

           0       0.51      0.46      0.48       319
           1       0.64      0.72      0.68       389
           2       0.61      0.57      0.59       394
           3       0.63      0.63      0.63       392
           4       0.69      0.69      0.69       385
           5       0.81      0.71      0.76       395
           6       0.78      0.76      0.77       390
           7       0.75      0.71      0.73       396
           8       0.49      0.80      0.61       398
           9       0.84      0.80      0.82       397
          10       0.88      0.86      0.87       399
          11       0.81      0.71      0.76       396
          12       0.66      0.57      0.61       393
          13       0.78      0.76      0.77       396
          14       0.75      0.73      0.74       394
          15       0.66      0.75      0.70       398
          16       0.57      0.66      0.61       364
          17       0.82      0.78      0.80       376
          18       0.55      0.45      0.50       310
          19       0.39      0.32      0.35       251

    accuracy                           0.68      7532
   macro avg       0.68      0.67      0.67      7532
weighted avg       0.69      0.68      0.68      7532

03/07/2020 06:43:37 PM - INFO - 

Cross validation:
03/07/2020 06:43:38 PM - INFO - 	accuracy: 5-fold cross validation: [0.75607601 0.73000442 0.74370305 0.76226248 0.73607427]
03/07/2020 06:43:38 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 74.56 (+/- 2.41)
03/07/2020 06:43:38 PM - INFO - 

USING PASSIVE_AGGRESSIVE_CLASSIFIER WITH BEST PARAMETERS: {'C': 0.01, 'early_stopping': False, 'tol': 0.0001, 'validation_fraction': 0.0001}
03/07/2020 06:43:38 PM - INFO - ________________________________________________________________________________
03/07/2020 06:43:38 PM - INFO - Training: 
03/07/2020 06:43:38 PM - INFO - PassiveAggressiveClassifier(C=0.01, average=False, class_weight=None,
                            early_stopping=False, fit_intercept=True,
                            loss='hinge', max_iter=1000, n_iter_no_change=5,
                            n_jobs=None, random_state=None, shuffle=True,
                            tol=0.0001, validation_fraction=0.0001, verbose=0,
                            warm_start=False)
03/07/2020 06:43:47 PM - INFO - Train time: 9.017s
03/07/2020 06:43:47 PM - INFO - Test time:  0.011s
03/07/2020 06:43:47 PM - INFO - Accuracy score:   0.697
03/07/2020 06:43:47 PM - INFO - 

===> Classification Report:

03/07/2020 06:43:47 PM - INFO -               precision    recall  f1-score   support

           0       0.51      0.48      0.49       319
           1       0.67      0.74      0.70       389
           2       0.62      0.60      0.61       394
           3       0.64      0.66      0.65       392
           4       0.74      0.69      0.71       385
           5       0.81      0.71      0.75       395
           6       0.79      0.76      0.78       390
           7       0.78      0.71      0.75       396
           8       0.52      0.81      0.63       398
           9       0.87      0.82      0.85       397
          10       0.90      0.88      0.89       399
          11       0.82      0.72      0.77       396
          12       0.64      0.58      0.61       393
          13       0.78      0.78      0.78       396
          14       0.74      0.75      0.75       394
          15       0.67      0.77      0.72       398
          16       0.59      0.68      0.63       364
          17       0.82      0.77      0.80       376
          18       0.57      0.46      0.51       310
          19       0.40      0.32      0.36       251

    accuracy                           0.70      7532
   macro avg       0.69      0.69      0.69      7532
weighted avg       0.70      0.70      0.70      7532

03/07/2020 06:43:47 PM - INFO - 

Cross validation:
03/07/2020 06:43:58 PM - INFO - 	accuracy: 5-fold cross validation: [0.76623951 0.74856385 0.76314627 0.77021653 0.7484527 ]
03/07/2020 06:43:58 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 75.93 (+/- 1.82)
03/07/2020 06:43:58 PM - INFO - It took 74.68414282798767 seconds
03/07/2020 06:43:58 PM - INFO - ********************************************************************************
03/07/2020 06:43:58 PM - INFO - ################################################################################
03/07/2020 06:43:58 PM - INFO - 

CURRENT CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 06:43:58 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 06:43:58 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 06:43:58 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  36.52%  |  [0.39460893 0.39019001 0.39991162 0.39019001 0.40539346]  |  39.61 (+/- 1.18)  |  4.611  |  0.2475  |
03/07/2020 06:43:58 PM - INFO - |  2  |  BERNOULLI_NB  |  45.84%  |  [0.48033584 0.46133451 0.48563853 0.47547503 0.48938992]  |  47.84 (+/- 1.95)  |  0.05548  |  0.05272  |
03/07/2020 06:43:58 PM - INFO - |  3  |  COMPLEMENT_NB  |  71.04%  |  [0.77110031 0.75740168 0.77993814 0.7715422  0.76038904]  |  76.81 (+/- 1.64)  |  0.06326  |  0.01035  |
03/07/2020 06:43:58 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  42.82%  |  [0.49094123 0.47591692 0.46442775 0.47105612 0.5       ]  |  48.05 (+/- 2.62)  |  10.34  |  0.008546  |
03/07/2020 06:43:58 PM - INFO - |  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  75.73 (+/- 2.09)  |  0.8659  |  0.008424  |
03/07/2020 06:43:58 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  69.17%  |  [0.74458683 0.72646929 0.74193548 0.74768007 0.72413793]  |  73.70 (+/- 1.94)  |  29.42  |  0.01162  |
03/07/2020 06:43:58 PM - INFO - |  9  |  MULTINOMIAL_NB  |  65.64%  |  [0.7105612  0.69951392 0.69863014 0.69995581 0.69142352]  |  70.00 (+/- 1.22)  |  0.04385  |  0.01045  |
03/07/2020 06:43:58 PM - INFO - |  10  |  NEAREST_CENTROID  |  64.91%  |  [0.69774635 0.69244366 0.69818825 0.70393283 0.69849691]  |  69.82 (+/- 0.73)  |  0.01918  |  0.01427  |
03/07/2020 06:43:58 PM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  68.26%  |  [0.75607601 0.73000442 0.74370305 0.76226248 0.73607427]  |  74.56 (+/- 2.41)  |  1.037  |  0.02384  |
03/07/2020 06:43:58 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  62.16%  |  [0.68360583 0.67034909 0.66725586 0.68272205 0.66534041]  |  67.39 (+/- 1.55)  |  35.03  |  0.6005  |
03/07/2020 06:43:58 PM - INFO - 

CURRENT CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 06:43:58 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 06:43:58 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 06:43:58 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  44.03%  |  [0.46973045 0.45779938 0.46531153 0.47017234 0.46993811]  |  46.66 (+/- 0.95)  |  18.29  |  0.9501  |
03/07/2020 06:43:58 PM - INFO - |  2  |  BERNOULLI_NB  |  62.56%  |  [0.69200177 0.67741935 0.6752099  0.67388422 0.67992927]  |  67.97 (+/- 1.30)  |  0.05404  |  0.05173  |
03/07/2020 06:43:58 PM - INFO - |  3  |  COMPLEMENT_NB  |  71.22%  |  [0.776403   0.75828546 0.78126381 0.77507733 0.76967286]  |  77.21 (+/- 1.57)  |  0.06653  |  0.01054  |
03/07/2020 06:43:58 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  44.57%  |  [0.4874061  0.48608042 0.49138312 0.49359258 0.48099027]  |  48.79 (+/- 0.88)  |  8.941  |  0.0056  |
03/07/2020 06:43:58 PM - INFO - |  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  75.73 (+/- 2.09)  |  0.8605  |  0.008546  |
03/07/2020 06:43:58 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  69.28%  |  [0.75961114 0.74016792 0.75519222 0.7565179  0.73607427]  |  74.95 (+/- 1.90)  |  32.91  |  0.01159  |
03/07/2020 06:43:58 PM - INFO - |  9  |  MULTINOMIAL_NB  |  68.79%  |  [0.76049492 0.74149359 0.75519222 0.75695979 0.74270557]  |  75.14 (+/- 1.55)  |  0.05682  |  0.01024  |
03/07/2020 06:43:58 PM - INFO - |  10  |  NEAREST_CENTROID  |  66.70%  |  [0.71674768 0.71807335 0.71851525 0.71939903 0.70601238]  |  71.57 (+/- 0.99)  |  0.01737  |  0.01961  |
03/07/2020 06:43:58 PM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  69.69%  |  [0.76623951 0.74856385 0.76314627 0.77021653 0.7484527 ]  |  75.93 (+/- 1.82)  |  9.017  |  0.01138  |
03/07/2020 06:43:58 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  63.64%  |  [0.69067609 0.67255855 0.69067609 0.69730446 0.67241379]  |  68.47 (+/- 2.06)  |  39.54  |  1.178  |
03/07/2020 06:43:58 PM - INFO - 

03/07/2020 06:43:58 PM - INFO - ################################################################################
03/07/2020 06:43:58 PM - INFO - 11)
03/07/2020 06:43:58 PM - INFO - ********************************************************************************
03/07/2020 06:43:58 PM - INFO - Classifier: K_NEIGHBORS_CLASSIFIER, Dataset: TWENTY_NEWS_GROUPS
03/07/2020 06:43:58 PM - INFO - ********************************************************************************
03/07/2020 06:44:01 PM - INFO - 

Performing grid search...

03/07/2020 06:44:01 PM - INFO - Parameters:
03/07/2020 06:44:01 PM - INFO - {'classifier__leaf_size': [5, 30], 'classifier__metric': ['euclidean', 'minkowski'], 'classifier__n_neighbors': [3, 50], 'classifier__weights': ['uniform', 'distance']}
03/07/2020 06:44:11 PM - INFO - 	Done in 9.540s
03/07/2020 06:44:11 PM - INFO - 	Best score: 0.121
03/07/2020 06:44:11 PM - INFO - 	Best parameters set:
03/07/2020 06:44:11 PM - INFO - 		classifier__leaf_size: 5
03/07/2020 06:44:11 PM - INFO - 		classifier__metric: 'euclidean'
03/07/2020 06:44:11 PM - INFO - 		classifier__n_neighbors: 3
03/07/2020 06:44:11 PM - INFO - 		classifier__weights: 'distance'
03/07/2020 06:44:11 PM - INFO - 

USING K_NEIGHBORS_CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 06:44:11 PM - INFO - ________________________________________________________________________________
03/07/2020 06:44:11 PM - INFO - Training: 
03/07/2020 06:44:11 PM - INFO - KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                     weights='uniform')
03/07/2020 06:44:11 PM - INFO - Train time: 0.003s
03/07/2020 06:44:13 PM - INFO - Test time:  2.159s
03/07/2020 06:44:13 PM - INFO - Accuracy score:   0.072
03/07/2020 06:44:13 PM - INFO - 

===> Classification Report:

03/07/2020 06:44:13 PM - INFO -               precision    recall  f1-score   support

           0       0.05      0.14      0.08       319
           1       0.06      0.16      0.09       389
           2       0.05      0.19      0.08       394
           3       0.09      0.10      0.09       392
           4       0.06      0.13      0.09       385
           5       0.12      0.02      0.04       395
           6       0.14      0.06      0.08       390
           7       0.07      0.15      0.10       396
           8       0.11      0.06      0.08       398
           9       0.07      0.07      0.07       397
          10       0.11      0.04      0.06       399
          11       0.05      0.01      0.02       396
          12       0.07      0.06      0.07       393
          13       0.08      0.04      0.05       396
          14       0.15      0.05      0.08       394
          15       0.05      0.01      0.01       398
          16       0.06      0.02      0.03       364
          17       0.18      0.05      0.07       376
          18       0.11      0.03      0.05       310
          19       0.07      0.03      0.04       251

    accuracy                           0.07      7532
   macro avg       0.09      0.07      0.06      7532
weighted avg       0.09      0.07      0.06      7532

03/07/2020 06:44:13 PM - INFO - 

Cross validation:
03/07/2020 06:44:13 PM - INFO - 	accuracy: 5-fold cross validation: [0.06937693 0.08174989 0.07909854 0.08263367 0.08311229]
03/07/2020 06:44:13 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 7.92 (+/- 1.02)
03/07/2020 06:44:13 PM - INFO - 

USING K_NEIGHBORS_CLASSIFIER WITH BEST PARAMETERS: {'leaf_size': 5, 'metric': 'euclidean', 'n_neighbors': 3, 'weights': 'distance'}
03/07/2020 06:44:13 PM - INFO - ________________________________________________________________________________
03/07/2020 06:44:13 PM - INFO - Training: 
03/07/2020 06:44:13 PM - INFO - KNeighborsClassifier(algorithm='auto', leaf_size=5, metric='euclidean',
                     metric_params=None, n_jobs=None, n_neighbors=3, p=2,
                     weights='distance')
03/07/2020 06:44:13 PM - INFO - Train time: 0.004s
03/07/2020 06:44:15 PM - INFO - Test time:  1.762s
03/07/2020 06:44:15 PM - INFO - Accuracy score:   0.085
03/07/2020 06:44:15 PM - INFO - 

===> Classification Report:

03/07/2020 06:44:15 PM - INFO -               precision    recall  f1-score   support

           0       0.95      0.06      0.11       319
           1       0.79      0.03      0.05       389
           2       0.67      0.02      0.03       394
           3       0.88      0.06      0.11       392
           4       0.79      0.03      0.06       385
           5       0.95      0.05      0.09       395
           6       1.00      0.12      0.21       390
           7       0.62      0.01      0.02       396
           8       0.88      0.06      0.11       398
           9       0.92      0.03      0.06       397
          10       0.96      0.06      0.11       399
          11       0.61      0.04      0.07       396
          12       0.92      0.03      0.05       393
          13       1.00      0.02      0.03       396
          14       0.79      0.06      0.11       394
          15       0.75      0.02      0.03       398
          16       0.50      0.01      0.02       364
          17       0.05      1.00      0.10       376
          18       1.00      0.01      0.01       310
          19       0.50      0.00      0.01       251

    accuracy                           0.08      7532
   macro avg       0.78      0.08      0.07      7532
weighted avg       0.78      0.08      0.07      7532

03/07/2020 06:44:15 PM - INFO - 

Cross validation:
03/07/2020 06:44:16 PM - INFO - 	accuracy: 5-fold cross validation: [0.12417145 0.11842687 0.12063632 0.12240389 0.12113174]
03/07/2020 06:44:16 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 12.14 (+/- 0.38)
03/07/2020 06:44:16 PM - INFO - It took 17.961363792419434 seconds
03/07/2020 06:44:16 PM - INFO - ********************************************************************************
03/07/2020 06:44:16 PM - INFO - ################################################################################
03/07/2020 06:44:16 PM - INFO - 

CURRENT CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 06:44:16 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 06:44:16 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 06:44:16 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  36.52%  |  [0.39460893 0.39019001 0.39991162 0.39019001 0.40539346]  |  39.61 (+/- 1.18)  |  4.611  |  0.2475  |
03/07/2020 06:44:16 PM - INFO - |  2  |  BERNOULLI_NB  |  45.84%  |  [0.48033584 0.46133451 0.48563853 0.47547503 0.48938992]  |  47.84 (+/- 1.95)  |  0.05548  |  0.05272  |
03/07/2020 06:44:16 PM - INFO - |  3  |  COMPLEMENT_NB  |  71.04%  |  [0.77110031 0.75740168 0.77993814 0.7715422  0.76038904]  |  76.81 (+/- 1.64)  |  0.06326  |  0.01035  |
03/07/2020 06:44:16 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  42.82%  |  [0.49094123 0.47591692 0.46442775 0.47105612 0.5       ]  |  48.05 (+/- 2.62)  |  10.34  |  0.008546  |
03/07/2020 06:44:16 PM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  7.20%  |  [0.06937693 0.08174989 0.07909854 0.08263367 0.08311229]  |  7.92 (+/- 1.02)  |  0.002982  |  2.159  |
03/07/2020 06:44:16 PM - INFO - |  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  75.73 (+/- 2.09)  |  0.8659  |  0.008424  |
03/07/2020 06:44:16 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  69.17%  |  [0.74458683 0.72646929 0.74193548 0.74768007 0.72413793]  |  73.70 (+/- 1.94)  |  29.42  |  0.01162  |
03/07/2020 06:44:16 PM - INFO - |  9  |  MULTINOMIAL_NB  |  65.64%  |  [0.7105612  0.69951392 0.69863014 0.69995581 0.69142352]  |  70.00 (+/- 1.22)  |  0.04385  |  0.01045  |
03/07/2020 06:44:16 PM - INFO - |  10  |  NEAREST_CENTROID  |  64.91%  |  [0.69774635 0.69244366 0.69818825 0.70393283 0.69849691]  |  69.82 (+/- 0.73)  |  0.01918  |  0.01427  |
03/07/2020 06:44:16 PM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  68.26%  |  [0.75607601 0.73000442 0.74370305 0.76226248 0.73607427]  |  74.56 (+/- 2.41)  |  1.037  |  0.02384  |
03/07/2020 06:44:16 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  62.16%  |  [0.68360583 0.67034909 0.66725586 0.68272205 0.66534041]  |  67.39 (+/- 1.55)  |  35.03  |  0.6005  |
03/07/2020 06:44:16 PM - INFO - 

CURRENT CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 06:44:16 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 06:44:16 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 06:44:16 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  44.03%  |  [0.46973045 0.45779938 0.46531153 0.47017234 0.46993811]  |  46.66 (+/- 0.95)  |  18.29  |  0.9501  |
03/07/2020 06:44:16 PM - INFO - |  2  |  BERNOULLI_NB  |  62.56%  |  [0.69200177 0.67741935 0.6752099  0.67388422 0.67992927]  |  67.97 (+/- 1.30)  |  0.05404  |  0.05173  |
03/07/2020 06:44:16 PM - INFO - |  3  |  COMPLEMENT_NB  |  71.22%  |  [0.776403   0.75828546 0.78126381 0.77507733 0.76967286]  |  77.21 (+/- 1.57)  |  0.06653  |  0.01054  |
03/07/2020 06:44:16 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  44.57%  |  [0.4874061  0.48608042 0.49138312 0.49359258 0.48099027]  |  48.79 (+/- 0.88)  |  8.941  |  0.0056  |
03/07/2020 06:44:16 PM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  8.48%  |  [0.12417145 0.11842687 0.12063632 0.12240389 0.12113174]  |  12.14 (+/- 0.38)  |  0.00427  |  1.762  |
03/07/2020 06:44:16 PM - INFO - |  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  75.73 (+/- 2.09)  |  0.8605  |  0.008546  |
03/07/2020 06:44:16 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  69.28%  |  [0.75961114 0.74016792 0.75519222 0.7565179  0.73607427]  |  74.95 (+/- 1.90)  |  32.91  |  0.01159  |
03/07/2020 06:44:16 PM - INFO - |  9  |  MULTINOMIAL_NB  |  68.79%  |  [0.76049492 0.74149359 0.75519222 0.75695979 0.74270557]  |  75.14 (+/- 1.55)  |  0.05682  |  0.01024  |
03/07/2020 06:44:16 PM - INFO - |  10  |  NEAREST_CENTROID  |  66.70%  |  [0.71674768 0.71807335 0.71851525 0.71939903 0.70601238]  |  71.57 (+/- 0.99)  |  0.01737  |  0.01961  |
03/07/2020 06:44:16 PM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  69.69%  |  [0.76623951 0.74856385 0.76314627 0.77021653 0.7484527 ]  |  75.93 (+/- 1.82)  |  9.017  |  0.01138  |
03/07/2020 06:44:16 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  63.64%  |  [0.69067609 0.67255855 0.69067609 0.69730446 0.67241379]  |  68.47 (+/- 2.06)  |  39.54  |  1.178  |
03/07/2020 06:44:16 PM - INFO - 

03/07/2020 06:44:16 PM - INFO - ################################################################################
03/07/2020 06:44:16 PM - INFO - 12)
03/07/2020 06:44:16 PM - INFO - ********************************************************************************
03/07/2020 06:44:16 PM - INFO - Classifier: PERCEPTRON, Dataset: TWENTY_NEWS_GROUPS
03/07/2020 06:44:16 PM - INFO - ********************************************************************************
03/07/2020 06:44:19 PM - INFO - 

Performing grid search...

03/07/2020 06:44:19 PM - INFO - Parameters:
03/07/2020 06:44:19 PM - INFO - {'classifier__early_stopping': [True], 'classifier__max_iter': [100], 'classifier__n_iter_no_change': [3, 15], 'classifier__penalty': ['l2'], 'classifier__tol': [0.0001, 0.1], 'classifier__validation_fraction': [0.0001, 0.01]}
03/07/2020 06:44:25 PM - INFO - 	Done in 6.198s
03/07/2020 06:44:25 PM - INFO - 	Best score: 0.608
03/07/2020 06:44:25 PM - INFO - 	Best parameters set:
03/07/2020 06:44:25 PM - INFO - 		classifier__early_stopping: True
03/07/2020 06:44:25 PM - INFO - 		classifier__max_iter: 100
03/07/2020 06:44:25 PM - INFO - 		classifier__n_iter_no_change: 3
03/07/2020 06:44:25 PM - INFO - 		classifier__penalty: 'l2'
03/07/2020 06:44:25 PM - INFO - 		classifier__tol: 0.0001
03/07/2020 06:44:25 PM - INFO - 		classifier__validation_fraction: 0.01
03/07/2020 06:44:25 PM - INFO - 

USING PERCEPTRON WITH DEFAULT PARAMETERS
03/07/2020 06:44:25 PM - INFO - ________________________________________________________________________________
03/07/2020 06:44:25 PM - INFO - Training: 
03/07/2020 06:44:25 PM - INFO - Perceptron(alpha=0.0001, class_weight=None, early_stopping=False, eta0=1.0,
           fit_intercept=True, max_iter=1000, n_iter_no_change=5, n_jobs=None,
           penalty=None, random_state=0, shuffle=True, tol=0.001,
           validation_fraction=0.1, verbose=0, warm_start=False)
03/07/2020 06:44:26 PM - INFO - Train time: 0.552s
03/07/2020 06:44:26 PM - INFO - Test time:  0.025s
03/07/2020 06:44:26 PM - INFO - Accuracy score:   0.633
03/07/2020 06:44:26 PM - INFO - 

===> Classification Report:

03/07/2020 06:44:26 PM - INFO -               precision    recall  f1-score   support

           0       0.45      0.49      0.47       319
           1       0.57      0.68      0.62       389
           2       0.60      0.53      0.57       394
           3       0.62      0.55      0.58       392
           4       0.63      0.66      0.64       385
           5       0.76      0.65      0.70       395
           6       0.70      0.75      0.72       390
           7       0.74      0.62      0.67       396
           8       0.73      0.70      0.71       398
           9       0.49      0.79      0.61       397
          10       0.81      0.82      0.82       399
          11       0.69      0.70      0.69       396
          12       0.57      0.50      0.53       393
          13       0.69      0.70      0.69       396
          14       0.73      0.67      0.70       394
          15       0.64      0.69      0.67       398
          16       0.54      0.57      0.56       364
          17       0.76      0.73      0.74       376
          18       0.47      0.39      0.43       310
          19       0.40      0.27      0.33       251

    accuracy                           0.63      7532
   macro avg       0.63      0.62      0.62      7532
weighted avg       0.64      0.63      0.63      7532

03/07/2020 06:44:26 PM - INFO - 

Cross validation:
03/07/2020 06:44:26 PM - INFO - 	accuracy: 5-fold cross validation: [0.69818825 0.67565179 0.6955369  0.70349094 0.68302387]
03/07/2020 06:44:26 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 69.12 (+/- 2.05)
03/07/2020 06:44:26 PM - INFO - 

USING PERCEPTRON WITH BEST PARAMETERS: {'early_stopping': True, 'max_iter': 100, 'n_iter_no_change': 3, 'penalty': 'l2', 'tol': 0.0001, 'validation_fraction': 0.01}
03/07/2020 06:44:26 PM - INFO - ________________________________________________________________________________
03/07/2020 06:44:26 PM - INFO - Training: 
03/07/2020 06:44:26 PM - INFO - Perceptron(alpha=0.0001, class_weight=None, early_stopping=True, eta0=1.0,
           fit_intercept=True, max_iter=100, n_iter_no_change=3, n_jobs=None,
           penalty='l2', random_state=0, shuffle=True, tol=0.0001,
           validation_fraction=0.01, verbose=0, warm_start=False)
03/07/2020 06:44:27 PM - INFO - Train time: 0.785s
03/07/2020 06:44:27 PM - INFO - Test time:  0.022s
03/07/2020 06:44:27 PM - INFO - Accuracy score:   0.539
03/07/2020 06:44:27 PM - INFO - 

===> Classification Report:

03/07/2020 06:44:27 PM - INFO -               precision    recall  f1-score   support

           0       0.46      0.15      0.23       319
           1       0.54      0.55      0.55       389
           2       0.61      0.44      0.51       394
           3       0.52      0.47      0.49       392
           4       0.52      0.49      0.51       385
           5       0.74      0.54      0.62       395
           6       0.61      0.73      0.67       390
           7       0.37      0.64      0.47       396
           8       0.51      0.66      0.58       398
           9       0.65      0.66      0.65       397
          10       0.84      0.72      0.77       399
          11       0.47      0.66      0.55       396
          12       0.51      0.37      0.43       393
          13       0.53      0.57      0.55       396
          14       0.78      0.51      0.61       394
          15       0.41      0.76      0.53       398
          16       0.49      0.47      0.48       364
          17       0.68      0.68      0.68       376
          18       0.45      0.25      0.32       310
          19       0.26      0.18      0.21       251

    accuracy                           0.54      7532
   macro avg       0.55      0.53      0.52      7532
weighted avg       0.56      0.54      0.53      7532

03/07/2020 06:44:27 PM - INFO - 

Cross validation:
03/07/2020 06:44:28 PM - INFO - 	accuracy: 5-fold cross validation: [0.63146266 0.61025188 0.58948299 0.60450729 0.60477454]
03/07/2020 06:44:28 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 60.81 (+/- 2.71)
03/07/2020 06:44:28 PM - INFO - It took 11.718533039093018 seconds
03/07/2020 06:44:28 PM - INFO - ********************************************************************************
03/07/2020 06:44:28 PM - INFO - ################################################################################
03/07/2020 06:44:28 PM - INFO - 

CURRENT CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 06:44:28 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 06:44:28 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 06:44:28 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  36.52%  |  [0.39460893 0.39019001 0.39991162 0.39019001 0.40539346]  |  39.61 (+/- 1.18)  |  4.611  |  0.2475  |
03/07/2020 06:44:28 PM - INFO - |  2  |  BERNOULLI_NB  |  45.84%  |  [0.48033584 0.46133451 0.48563853 0.47547503 0.48938992]  |  47.84 (+/- 1.95)  |  0.05548  |  0.05272  |
03/07/2020 06:44:28 PM - INFO - |  3  |  COMPLEMENT_NB  |  71.04%  |  [0.77110031 0.75740168 0.77993814 0.7715422  0.76038904]  |  76.81 (+/- 1.64)  |  0.06326  |  0.01035  |
03/07/2020 06:44:28 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  42.82%  |  [0.49094123 0.47591692 0.46442775 0.47105612 0.5       ]  |  48.05 (+/- 2.62)  |  10.34  |  0.008546  |
03/07/2020 06:44:28 PM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  7.20%  |  [0.06937693 0.08174989 0.07909854 0.08263367 0.08311229]  |  7.92 (+/- 1.02)  |  0.002982  |  2.159  |
03/07/2020 06:44:28 PM - INFO - |  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  75.73 (+/- 2.09)  |  0.8659  |  0.008424  |
03/07/2020 06:44:28 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  69.17%  |  [0.74458683 0.72646929 0.74193548 0.74768007 0.72413793]  |  73.70 (+/- 1.94)  |  29.42  |  0.01162  |
03/07/2020 06:44:28 PM - INFO - |  9  |  MULTINOMIAL_NB  |  65.64%  |  [0.7105612  0.69951392 0.69863014 0.69995581 0.69142352]  |  70.00 (+/- 1.22)  |  0.04385  |  0.01045  |
03/07/2020 06:44:28 PM - INFO - |  10  |  NEAREST_CENTROID  |  64.91%  |  [0.69774635 0.69244366 0.69818825 0.70393283 0.69849691]  |  69.82 (+/- 0.73)  |  0.01918  |  0.01427  |
03/07/2020 06:44:28 PM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  68.26%  |  [0.75607601 0.73000442 0.74370305 0.76226248 0.73607427]  |  74.56 (+/- 2.41)  |  1.037  |  0.02384  |
03/07/2020 06:44:28 PM - INFO - |  12  |  PERCEPTRON  |  63.34%  |  [0.69818825 0.67565179 0.6955369  0.70349094 0.68302387]  |  69.12 (+/- 2.05)  |  0.5521  |  0.02452  |
03/07/2020 06:44:28 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  62.16%  |  [0.68360583 0.67034909 0.66725586 0.68272205 0.66534041]  |  67.39 (+/- 1.55)  |  35.03  |  0.6005  |
03/07/2020 06:44:28 PM - INFO - 

CURRENT CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 06:44:28 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 06:44:28 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 06:44:28 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  44.03%  |  [0.46973045 0.45779938 0.46531153 0.47017234 0.46993811]  |  46.66 (+/- 0.95)  |  18.29  |  0.9501  |
03/07/2020 06:44:28 PM - INFO - |  2  |  BERNOULLI_NB  |  62.56%  |  [0.69200177 0.67741935 0.6752099  0.67388422 0.67992927]  |  67.97 (+/- 1.30)  |  0.05404  |  0.05173  |
03/07/2020 06:44:28 PM - INFO - |  3  |  COMPLEMENT_NB  |  71.22%  |  [0.776403   0.75828546 0.78126381 0.77507733 0.76967286]  |  77.21 (+/- 1.57)  |  0.06653  |  0.01054  |
03/07/2020 06:44:28 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  44.57%  |  [0.4874061  0.48608042 0.49138312 0.49359258 0.48099027]  |  48.79 (+/- 0.88)  |  8.941  |  0.0056  |
03/07/2020 06:44:28 PM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  8.48%  |  [0.12417145 0.11842687 0.12063632 0.12240389 0.12113174]  |  12.14 (+/- 0.38)  |  0.00427  |  1.762  |
03/07/2020 06:44:28 PM - INFO - |  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  75.73 (+/- 2.09)  |  0.8605  |  0.008546  |
03/07/2020 06:44:28 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  69.28%  |  [0.75961114 0.74016792 0.75519222 0.7565179  0.73607427]  |  74.95 (+/- 1.90)  |  32.91  |  0.01159  |
03/07/2020 06:44:28 PM - INFO - |  9  |  MULTINOMIAL_NB  |  68.79%  |  [0.76049492 0.74149359 0.75519222 0.75695979 0.74270557]  |  75.14 (+/- 1.55)  |  0.05682  |  0.01024  |
03/07/2020 06:44:28 PM - INFO - |  10  |  NEAREST_CENTROID  |  66.70%  |  [0.71674768 0.71807335 0.71851525 0.71939903 0.70601238]  |  71.57 (+/- 0.99)  |  0.01737  |  0.01961  |
03/07/2020 06:44:28 PM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  69.69%  |  [0.76623951 0.74856385 0.76314627 0.77021653 0.7484527 ]  |  75.93 (+/- 1.82)  |  9.017  |  0.01138  |
03/07/2020 06:44:28 PM - INFO - |  12  |  PERCEPTRON  |  53.86%  |  [0.63146266 0.61025188 0.58948299 0.60450729 0.60477454]  |  60.81 (+/- 2.71)  |  0.7855  |  0.0223  |
03/07/2020 06:44:28 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  63.64%  |  [0.69067609 0.67255855 0.69067609 0.69730446 0.67241379]  |  68.47 (+/- 2.06)  |  39.54  |  1.178  |
03/07/2020 06:44:28 PM - INFO - 

03/07/2020 06:44:28 PM - INFO - ################################################################################
03/07/2020 06:44:28 PM - INFO - 13)
03/07/2020 06:44:28 PM - INFO - ********************************************************************************
03/07/2020 06:44:28 PM - INFO - Classifier: RIDGE_CLASSIFIER, Dataset: TWENTY_NEWS_GROUPS
03/07/2020 06:44:28 PM - INFO - ********************************************************************************
03/07/2020 06:44:31 PM - INFO - 

Performing grid search...

03/07/2020 06:44:31 PM - INFO - Parameters:
03/07/2020 06:44:31 PM - INFO - {'classifier__alpha': [0.5, 1.0], 'classifier__tol': [0.0001, 0.001]}
03/07/2020 06:44:43 PM - INFO - 	Done in 12.406s
03/07/2020 06:44:43 PM - INFO - 	Best score: 0.764
03/07/2020 06:44:43 PM - INFO - 	Best parameters set:
03/07/2020 06:44:43 PM - INFO - 		classifier__alpha: 0.5
03/07/2020 06:44:43 PM - INFO - 		classifier__tol: 0.001
03/07/2020 06:44:43 PM - INFO - 

USING RIDGE_CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 06:44:43 PM - INFO - ________________________________________________________________________________
03/07/2020 06:44:43 PM - INFO - Training: 
03/07/2020 06:44:43 PM - INFO - RidgeClassifier(alpha=1.0, class_weight=None, copy_X=True, fit_intercept=True,
                max_iter=None, normalize=False, random_state=None,
                solver='auto', tol=0.001)
03/07/2020 06:44:45 PM - INFO - Train time: 2.190s
03/07/2020 06:44:45 PM - INFO - Test time:  0.023s
03/07/2020 06:44:45 PM - INFO - Accuracy score:   0.707
03/07/2020 06:44:45 PM - INFO - 

===> Classification Report:

03/07/2020 06:44:45 PM - INFO -               precision    recall  f1-score   support

           0       0.53      0.47      0.50       319
           1       0.67      0.74      0.70       389
           2       0.64      0.65      0.65       394
           3       0.69      0.68      0.68       392
           4       0.75      0.70      0.72       385
           5       0.83      0.72      0.77       395
           6       0.75      0.79      0.77       390
           7       0.78      0.72      0.75       396
           8       0.81      0.76      0.79       398
           9       0.56      0.89      0.68       397
          10       0.89      0.87      0.88       399
          11       0.84      0.72      0.78       396
          12       0.67      0.60      0.63       393
          13       0.79      0.79      0.79       396
          14       0.75      0.75      0.75       394
          15       0.63      0.82      0.72       398
          16       0.58      0.71      0.64       364
          17       0.86      0.78      0.82       376
          18       0.60      0.47      0.53       310
          19       0.45      0.24      0.31       251

    accuracy                           0.71      7532
   macro avg       0.70      0.69      0.69      7532
weighted avg       0.71      0.71      0.70      7532

03/07/2020 06:44:45 PM - INFO - 

Cross validation:
03/07/2020 06:44:47 PM - INFO - 	accuracy: 5-fold cross validation: [0.76623951 0.74900574 0.76889085 0.77330977 0.75464191]
03/07/2020 06:44:47 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 76.24 (+/- 1.82)
03/07/2020 06:44:47 PM - INFO - 

USING RIDGE_CLASSIFIER WITH BEST PARAMETERS: {'alpha': 0.5, 'tol': 0.001}
03/07/2020 06:44:47 PM - INFO - ________________________________________________________________________________
03/07/2020 06:44:47 PM - INFO - Training: 
03/07/2020 06:44:47 PM - INFO - RidgeClassifier(alpha=0.5, class_weight=None, copy_X=True, fit_intercept=True,
                max_iter=None, normalize=False, random_state=None,
                solver='auto', tol=0.001)
03/07/2020 06:44:50 PM - INFO - Train time: 2.787s
03/07/2020 06:44:50 PM - INFO - Test time:  0.022s
03/07/2020 06:44:50 PM - INFO - Accuracy score:   0.700
03/07/2020 06:44:50 PM - INFO - 

===> Classification Report:

03/07/2020 06:44:50 PM - INFO -               precision    recall  f1-score   support

           0       0.53      0.49      0.51       319
           1       0.65      0.74      0.69       389
           2       0.64      0.63      0.64       394
           3       0.66      0.68      0.67       392
           4       0.74      0.70      0.72       385
           5       0.82      0.72      0.76       395
           6       0.77      0.78      0.77       390
           7       0.75      0.71      0.73       396
           8       0.80      0.75      0.77       398
           9       0.56      0.87      0.68       397
          10       0.89      0.87      0.88       399
          11       0.83      0.71      0.77       396
          12       0.64      0.58      0.61       393
          13       0.80      0.78      0.79       396
          14       0.75      0.75      0.75       394
          15       0.64      0.80      0.71       398
          16       0.59      0.70      0.64       364
          17       0.86      0.77      0.81       376
          18       0.56      0.46      0.51       310
          19       0.45      0.26      0.33       251

    accuracy                           0.70      7532
   macro avg       0.70      0.69      0.69      7532
weighted avg       0.70      0.70      0.70      7532

03/07/2020 06:44:50 PM - INFO - 

Cross validation:
03/07/2020 06:44:52 PM - INFO - 	accuracy: 5-fold cross validation: [0.76756518 0.74502872 0.76800707 0.776403   0.76083112]
03/07/2020 06:44:52 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 76.36 (+/- 2.10)
03/07/2020 06:44:52 PM - INFO - It took 23.857691764831543 seconds
03/07/2020 06:44:52 PM - INFO - ********************************************************************************
03/07/2020 06:44:52 PM - INFO - ################################################################################
03/07/2020 06:44:52 PM - INFO - 

CURRENT CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 06:44:52 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 06:44:52 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 06:44:52 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  36.52%  |  [0.39460893 0.39019001 0.39991162 0.39019001 0.40539346]  |  39.61 (+/- 1.18)  |  4.611  |  0.2475  |
03/07/2020 06:44:52 PM - INFO - |  2  |  BERNOULLI_NB  |  45.84%  |  [0.48033584 0.46133451 0.48563853 0.47547503 0.48938992]  |  47.84 (+/- 1.95)  |  0.05548  |  0.05272  |
03/07/2020 06:44:52 PM - INFO - |  3  |  COMPLEMENT_NB  |  71.04%  |  [0.77110031 0.75740168 0.77993814 0.7715422  0.76038904]  |  76.81 (+/- 1.64)  |  0.06326  |  0.01035  |
03/07/2020 06:44:52 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  42.82%  |  [0.49094123 0.47591692 0.46442775 0.47105612 0.5       ]  |  48.05 (+/- 2.62)  |  10.34  |  0.008546  |
03/07/2020 06:44:52 PM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  7.20%  |  [0.06937693 0.08174989 0.07909854 0.08263367 0.08311229]  |  7.92 (+/- 1.02)  |  0.002982  |  2.159  |
03/07/2020 06:44:52 PM - INFO - |  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  75.73 (+/- 2.09)  |  0.8659  |  0.008424  |
03/07/2020 06:44:52 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  69.17%  |  [0.74458683 0.72646929 0.74193548 0.74768007 0.72413793]  |  73.70 (+/- 1.94)  |  29.42  |  0.01162  |
03/07/2020 06:44:52 PM - INFO - |  9  |  MULTINOMIAL_NB  |  65.64%  |  [0.7105612  0.69951392 0.69863014 0.69995581 0.69142352]  |  70.00 (+/- 1.22)  |  0.04385  |  0.01045  |
03/07/2020 06:44:52 PM - INFO - |  10  |  NEAREST_CENTROID  |  64.91%  |  [0.69774635 0.69244366 0.69818825 0.70393283 0.69849691]  |  69.82 (+/- 0.73)  |  0.01918  |  0.01427  |
03/07/2020 06:44:52 PM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  68.26%  |  [0.75607601 0.73000442 0.74370305 0.76226248 0.73607427]  |  74.56 (+/- 2.41)  |  1.037  |  0.02384  |
03/07/2020 06:44:52 PM - INFO - |  12  |  PERCEPTRON  |  63.34%  |  [0.69818825 0.67565179 0.6955369  0.70349094 0.68302387]  |  69.12 (+/- 2.05)  |  0.5521  |  0.02452  |
03/07/2020 06:44:52 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  62.16%  |  [0.68360583 0.67034909 0.66725586 0.68272205 0.66534041]  |  67.39 (+/- 1.55)  |  35.03  |  0.6005  |
03/07/2020 06:44:52 PM - INFO - |  14  |  RIDGE_CLASSIFIER  |  70.67%  |  [0.76623951 0.74900574 0.76889085 0.77330977 0.75464191]  |  76.24 (+/- 1.82)  |  2.19  |  0.0226  |
03/07/2020 06:44:52 PM - INFO - 

CURRENT CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 06:44:52 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 06:44:52 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 06:44:52 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  44.03%  |  [0.46973045 0.45779938 0.46531153 0.47017234 0.46993811]  |  46.66 (+/- 0.95)  |  18.29  |  0.9501  |
03/07/2020 06:44:52 PM - INFO - |  2  |  BERNOULLI_NB  |  62.56%  |  [0.69200177 0.67741935 0.6752099  0.67388422 0.67992927]  |  67.97 (+/- 1.30)  |  0.05404  |  0.05173  |
03/07/2020 06:44:52 PM - INFO - |  3  |  COMPLEMENT_NB  |  71.22%  |  [0.776403   0.75828546 0.78126381 0.77507733 0.76967286]  |  77.21 (+/- 1.57)  |  0.06653  |  0.01054  |
03/07/2020 06:44:52 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  44.57%  |  [0.4874061  0.48608042 0.49138312 0.49359258 0.48099027]  |  48.79 (+/- 0.88)  |  8.941  |  0.0056  |
03/07/2020 06:44:52 PM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  8.48%  |  [0.12417145 0.11842687 0.12063632 0.12240389 0.12113174]  |  12.14 (+/- 0.38)  |  0.00427  |  1.762  |
03/07/2020 06:44:52 PM - INFO - |  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  75.73 (+/- 2.09)  |  0.8605  |  0.008546  |
03/07/2020 06:44:52 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  69.28%  |  [0.75961114 0.74016792 0.75519222 0.7565179  0.73607427]  |  74.95 (+/- 1.90)  |  32.91  |  0.01159  |
03/07/2020 06:44:52 PM - INFO - |  9  |  MULTINOMIAL_NB  |  68.79%  |  [0.76049492 0.74149359 0.75519222 0.75695979 0.74270557]  |  75.14 (+/- 1.55)  |  0.05682  |  0.01024  |
03/07/2020 06:44:52 PM - INFO - |  10  |  NEAREST_CENTROID  |  66.70%  |  [0.71674768 0.71807335 0.71851525 0.71939903 0.70601238]  |  71.57 (+/- 0.99)  |  0.01737  |  0.01961  |
03/07/2020 06:44:52 PM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  69.69%  |  [0.76623951 0.74856385 0.76314627 0.77021653 0.7484527 ]  |  75.93 (+/- 1.82)  |  9.017  |  0.01138  |
03/07/2020 06:44:52 PM - INFO - |  12  |  PERCEPTRON  |  53.86%  |  [0.63146266 0.61025188 0.58948299 0.60450729 0.60477454]  |  60.81 (+/- 2.71)  |  0.7855  |  0.0223  |
03/07/2020 06:44:52 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  63.64%  |  [0.69067609 0.67255855 0.69067609 0.69730446 0.67241379]  |  68.47 (+/- 2.06)  |  39.54  |  1.178  |
03/07/2020 06:44:52 PM - INFO - |  14  |  RIDGE_CLASSIFIER  |  70.02%  |  [0.76756518 0.74502872 0.76800707 0.776403   0.76083112]  |  76.36 (+/- 2.10)  |  2.787  |  0.02243  |
03/07/2020 06:44:52 PM - INFO - 

03/07/2020 06:44:52 PM - INFO - ################################################################################
03/07/2020 06:44:52 PM - INFO - 14)
03/07/2020 06:44:52 PM - INFO - ********************************************************************************
03/07/2020 06:44:52 PM - INFO - Classifier: GRADIENT_BOOSTING_CLASSIFIER, Dataset: TWENTY_NEWS_GROUPS
03/07/2020 06:44:52 PM - INFO - ********************************************************************************
03/07/2020 06:44:55 PM - INFO - 

Performing grid search...

03/07/2020 06:44:55 PM - INFO - Parameters:
03/07/2020 06:44:55 PM - INFO - {'classifier__learning_rate': [0.01, 0.1], 'classifier__n_estimators': [100, 200]}
03/07/2020 08:43:09 PM - INFO - 	Done in 7094.316s
03/07/2020 08:43:09 PM - INFO - 	Best score: 0.651
03/07/2020 08:43:09 PM - INFO - 	Best parameters set:
03/07/2020 08:43:09 PM - INFO - 		classifier__learning_rate: 0.1
03/07/2020 08:43:09 PM - INFO - 		classifier__n_estimators: 200
03/07/2020 08:43:09 PM - INFO - 

USING GRADIENT_BOOSTING_CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 08:43:09 PM - INFO - ________________________________________________________________________________
03/07/2020 08:43:09 PM - INFO - Training: 
03/07/2020 08:43:09 PM - INFO - GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.1, loss='deviance', max_depth=3,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_iter_no_change=None, presort='deprecated',
                           random_state=None, subsample=1.0, tol=0.0001,
                           validation_fraction=0.1, verbose=0,
                           warm_start=False)
03/07/2020 08:48:40 PM - INFO - Train time: 331.641s
03/07/2020 08:48:41 PM - INFO - Test time:  0.176s
03/07/2020 08:48:41 PM - INFO - Accuracy score:   0.593
03/07/2020 08:48:41 PM - INFO - 

===> Classification Report:

03/07/2020 08:48:41 PM - INFO -               precision    recall  f1-score   support

           0       0.45      0.34      0.39       319
           1       0.60      0.64      0.62       389
           2       0.59      0.57      0.58       394
           3       0.55      0.57      0.56       392
           4       0.68      0.63      0.65       385
           5       0.76      0.59      0.67       395
           6       0.70      0.67      0.68       390
           7       0.70      0.58      0.64       396
           8       0.78      0.63      0.70       398
           9       0.82      0.70      0.75       397
          10       0.81      0.76      0.79       399
          11       0.77      0.63      0.70       396
          12       0.19      0.61      0.29       393
          13       0.77      0.61      0.68       396
          14       0.75      0.62      0.68       394
          15       0.61      0.67      0.64       398
          16       0.57      0.58      0.58       364
          17       0.84      0.63      0.72       376
          18       0.55      0.37      0.44       310
          19       0.31      0.21      0.25       251

    accuracy                           0.59      7532
   macro avg       0.64      0.58      0.60      7532
weighted avg       0.65      0.59      0.61      7532

03/07/2020 08:48:41 PM - INFO - 

Cross validation:
03/07/2020 09:08:23 PM - INFO - 	accuracy: 5-fold cross validation: [0.65046399 0.61555457 0.63411401 0.64869642 0.64456233]
03/07/2020 09:08:23 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 63.87 (+/- 2.58)
03/07/2020 09:08:23 PM - INFO - 

USING GRADIENT_BOOSTING_CLASSIFIER WITH BEST PARAMETERS: {'learning_rate': 0.1, 'n_estimators': 200}
03/07/2020 09:08:23 PM - INFO - ________________________________________________________________________________
03/07/2020 09:08:23 PM - INFO - Training: 
03/07/2020 09:08:23 PM - INFO - GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.1, loss='deviance', max_depth=3,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=200,
                           n_iter_no_change=None, presort='deprecated',
                           random_state=None, subsample=1.0, tol=0.0001,
                           validation_fraction=0.1, verbose=0,
                           warm_start=False)
03/07/2020 09:19:12 PM - INFO - Train time: 649.869s
03/07/2020 09:19:13 PM - INFO - Test time:  0.324s
03/07/2020 09:19:13 PM - INFO - Accuracy score:   0.596
03/07/2020 09:19:13 PM - INFO - 

===> Classification Report:

03/07/2020 09:19:13 PM - INFO -               precision    recall  f1-score   support

           0       0.44      0.36      0.40       319
           1       0.58      0.65      0.61       389
           2       0.60      0.57      0.58       394
           3       0.56      0.59      0.57       392
           4       0.68      0.63      0.65       385
           5       0.75      0.60      0.67       395
           6       0.71      0.67      0.69       390
           7       0.71      0.58      0.64       396
           8       0.76      0.65      0.70       398
           9       0.78      0.72      0.75       397
          10       0.81      0.76      0.78       399
          11       0.77      0.63      0.69       396
          12       0.20      0.59      0.30       393
          13       0.77      0.61      0.68       396
          14       0.70      0.60      0.65       394
          15       0.63      0.67      0.65       398
          16       0.57      0.60      0.58       364
          17       0.82      0.64      0.72       376
          18       0.51      0.37      0.43       310
          19       0.27      0.19      0.22       251

    accuracy                           0.60      7532
   macro avg       0.63      0.58      0.60      7532
weighted avg       0.64      0.60      0.61      7532

03/07/2020 09:19:13 PM - INFO - 

Cross validation:
03/07/2020 10:00:05 PM - INFO - 	accuracy: 5-fold cross validation: [0.6597437  0.62836942 0.64692886 0.66327883 0.64633068]
03/07/2020 10:00:05 PM - INFO - 	test accuracy: 5-fold cross validation accuracy: 64.89 (+/- 2.46)
03/07/2020 10:00:05 PM - INFO - It took 11713.619091272354 seconds
03/07/2020 10:00:05 PM - INFO - ********************************************************************************
03/07/2020 10:00:05 PM - INFO - ################################################################################
03/07/2020 10:00:05 PM - INFO - 

CURRENT CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 10:00:05 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 10:00:05 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 10:00:05 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  36.52%  |  [0.39460893 0.39019001 0.39991162 0.39019001 0.40539346]  |  39.61 (+/- 1.18)  |  4.611  |  0.2475  |
03/07/2020 10:00:05 PM - INFO - |  2  |  BERNOULLI_NB  |  45.84%  |  [0.48033584 0.46133451 0.48563853 0.47547503 0.48938992]  |  47.84 (+/- 1.95)  |  0.05548  |  0.05272  |
03/07/2020 10:00:05 PM - INFO - |  3  |  COMPLEMENT_NB  |  71.04%  |  [0.77110031 0.75740168 0.77993814 0.7715422  0.76038904]  |  76.81 (+/- 1.64)  |  0.06326  |  0.01035  |
03/07/2020 10:00:05 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  42.82%  |  [0.49094123 0.47591692 0.46442775 0.47105612 0.5       ]  |  48.05 (+/- 2.62)  |  10.34  |  0.008546  |
03/07/2020 10:00:05 PM - INFO - |  5  |  GRADIENT_BOOSTING_CLASSIFIER  |  59.32%  |  [0.65046399 0.61555457 0.63411401 0.64869642 0.64456233]  |  63.87 (+/- 2.58)  |  331.6  |  0.1763  |
03/07/2020 10:00:05 PM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  7.20%  |  [0.06937693 0.08174989 0.07909854 0.08263367 0.08311229]  |  7.92 (+/- 1.02)  |  0.002982  |  2.159  |
03/07/2020 10:00:05 PM - INFO - |  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  75.73 (+/- 2.09)  |  0.8659  |  0.008424  |
03/07/2020 10:00:05 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  69.17%  |  [0.74458683 0.72646929 0.74193548 0.74768007 0.72413793]  |  73.70 (+/- 1.94)  |  29.42  |  0.01162  |
03/07/2020 10:00:05 PM - INFO - |  9  |  MULTINOMIAL_NB  |  65.64%  |  [0.7105612  0.69951392 0.69863014 0.69995581 0.69142352]  |  70.00 (+/- 1.22)  |  0.04385  |  0.01045  |
03/07/2020 10:00:05 PM - INFO - |  10  |  NEAREST_CENTROID  |  64.91%  |  [0.69774635 0.69244366 0.69818825 0.70393283 0.69849691]  |  69.82 (+/- 0.73)  |  0.01918  |  0.01427  |
03/07/2020 10:00:05 PM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  68.26%  |  [0.75607601 0.73000442 0.74370305 0.76226248 0.73607427]  |  74.56 (+/- 2.41)  |  1.037  |  0.02384  |
03/07/2020 10:00:05 PM - INFO - |  12  |  PERCEPTRON  |  63.34%  |  [0.69818825 0.67565179 0.6955369  0.70349094 0.68302387]  |  69.12 (+/- 2.05)  |  0.5521  |  0.02452  |
03/07/2020 10:00:05 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  62.16%  |  [0.68360583 0.67034909 0.66725586 0.68272205 0.66534041]  |  67.39 (+/- 1.55)  |  35.03  |  0.6005  |
03/07/2020 10:00:05 PM - INFO - |  14  |  RIDGE_CLASSIFIER  |  70.67%  |  [0.76623951 0.74900574 0.76889085 0.77330977 0.75464191]  |  76.24 (+/- 1.82)  |  2.19  |  0.0226  |
03/07/2020 10:00:05 PM - INFO - 

CURRENT CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 10:00:05 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 10:00:05 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 10:00:05 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  44.03%  |  [0.46973045 0.45779938 0.46531153 0.47017234 0.46993811]  |  46.66 (+/- 0.95)  |  18.29  |  0.9501  |
03/07/2020 10:00:05 PM - INFO - |  2  |  BERNOULLI_NB  |  62.56%  |  [0.69200177 0.67741935 0.6752099  0.67388422 0.67992927]  |  67.97 (+/- 1.30)  |  0.05404  |  0.05173  |
03/07/2020 10:00:05 PM - INFO - |  3  |  COMPLEMENT_NB  |  71.22%  |  [0.776403   0.75828546 0.78126381 0.77507733 0.76967286]  |  77.21 (+/- 1.57)  |  0.06653  |  0.01054  |
03/07/2020 10:00:05 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  44.57%  |  [0.4874061  0.48608042 0.49138312 0.49359258 0.48099027]  |  48.79 (+/- 0.88)  |  8.941  |  0.0056  |
03/07/2020 10:00:05 PM - INFO - |  5  |  GRADIENT_BOOSTING_CLASSIFIER  |  59.60%  |  [0.6597437  0.62836942 0.64692886 0.66327883 0.64633068]  |  64.89 (+/- 2.46)  |  649.9  |  0.3243  |
03/07/2020 10:00:05 PM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  8.48%  |  [0.12417145 0.11842687 0.12063632 0.12240389 0.12113174]  |  12.14 (+/- 0.38)  |  0.00427  |  1.762  |
03/07/2020 10:00:05 PM - INFO - |  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  75.73 (+/- 2.09)  |  0.8605  |  0.008546  |
03/07/2020 10:00:05 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  69.28%  |  [0.75961114 0.74016792 0.75519222 0.7565179  0.73607427]  |  74.95 (+/- 1.90)  |  32.91  |  0.01159  |
03/07/2020 10:00:05 PM - INFO - |  9  |  MULTINOMIAL_NB  |  68.79%  |  [0.76049492 0.74149359 0.75519222 0.75695979 0.74270557]  |  75.14 (+/- 1.55)  |  0.05682  |  0.01024  |
03/07/2020 10:00:05 PM - INFO - |  10  |  NEAREST_CENTROID  |  66.70%  |  [0.71674768 0.71807335 0.71851525 0.71939903 0.70601238]  |  71.57 (+/- 0.99)  |  0.01737  |  0.01961  |
03/07/2020 10:00:05 PM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  69.69%  |  [0.76623951 0.74856385 0.76314627 0.77021653 0.7484527 ]  |  75.93 (+/- 1.82)  |  9.017  |  0.01138  |
03/07/2020 10:00:05 PM - INFO - |  12  |  PERCEPTRON  |  53.86%  |  [0.63146266 0.61025188 0.58948299 0.60450729 0.60477454]  |  60.81 (+/- 2.71)  |  0.7855  |  0.0223  |
03/07/2020 10:00:05 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  63.64%  |  [0.69067609 0.67255855 0.69067609 0.69730446 0.67241379]  |  68.47 (+/- 2.06)  |  39.54  |  1.178  |
03/07/2020 10:00:05 PM - INFO - |  14  |  RIDGE_CLASSIFIER  |  70.02%  |  [0.76756518 0.74502872 0.76800707 0.776403   0.76083112]  |  76.36 (+/- 2.10)  |  2.787  |  0.02243  |
03/07/2020 10:00:05 PM - INFO - 

FINAL CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 10:00:05 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 10:00:05 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 10:00:05 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  36.52%  |  [0.39460893 0.39019001 0.39991162 0.39019001 0.40539346]  |  39.61 (+/- 1.18)  |  4.611  |  0.2475  |
03/07/2020 10:00:05 PM - INFO - |  2  |  BERNOULLI_NB  |  45.84%  |  [0.48033584 0.46133451 0.48563853 0.47547503 0.48938992]  |  47.84 (+/- 1.95)  |  0.05548  |  0.05272  |
03/07/2020 10:00:05 PM - INFO - |  3  |  COMPLEMENT_NB  |  71.04%  |  [0.77110031 0.75740168 0.77993814 0.7715422  0.76038904]  |  76.81 (+/- 1.64)  |  0.06326  |  0.01035  |
03/07/2020 10:00:05 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  42.82%  |  [0.49094123 0.47591692 0.46442775 0.47105612 0.5       ]  |  48.05 (+/- 2.62)  |  10.34  |  0.008546  |
03/07/2020 10:00:05 PM - INFO - |  5  |  GRADIENT_BOOSTING_CLASSIFIER  |  59.32%  |  [0.65046399 0.61555457 0.63411401 0.64869642 0.64456233]  |  63.87 (+/- 2.58)  |  331.6  |  0.1763  |
03/07/2020 10:00:05 PM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  7.20%  |  [0.06937693 0.08174989 0.07909854 0.08263367 0.08311229]  |  7.92 (+/- 1.02)  |  0.002982  |  2.159  |
03/07/2020 10:00:05 PM - INFO - |  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  75.73 (+/- 2.09)  |  0.8659  |  0.008424  |
03/07/2020 10:00:05 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  69.17%  |  [0.74458683 0.72646929 0.74193548 0.74768007 0.72413793]  |  73.70 (+/- 1.94)  |  29.42  |  0.01162  |
03/07/2020 10:00:05 PM - INFO - |  9  |  MULTINOMIAL_NB  |  65.64%  |  [0.7105612  0.69951392 0.69863014 0.69995581 0.69142352]  |  70.00 (+/- 1.22)  |  0.04385  |  0.01045  |
03/07/2020 10:00:05 PM - INFO - |  10  |  NEAREST_CENTROID  |  64.91%  |  [0.69774635 0.69244366 0.69818825 0.70393283 0.69849691]  |  69.82 (+/- 0.73)  |  0.01918  |  0.01427  |
03/07/2020 10:00:05 PM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  68.26%  |  [0.75607601 0.73000442 0.74370305 0.76226248 0.73607427]  |  74.56 (+/- 2.41)  |  1.037  |  0.02384  |
03/07/2020 10:00:05 PM - INFO - |  12  |  PERCEPTRON  |  63.34%  |  [0.69818825 0.67565179 0.6955369  0.70349094 0.68302387]  |  69.12 (+/- 2.05)  |  0.5521  |  0.02452  |
03/07/2020 10:00:05 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  62.16%  |  [0.68360583 0.67034909 0.66725586 0.68272205 0.66534041]  |  67.39 (+/- 1.55)  |  35.03  |  0.6005  |
03/07/2020 10:00:05 PM - INFO - |  14  |  RIDGE_CLASSIFIER  |  70.67%  |  [0.76623951 0.74900574 0.76889085 0.77330977 0.75464191]  |  76.24 (+/- 1.82)  |  2.19  |  0.0226  |
03/07/2020 10:00:05 PM - INFO - 

FINAL CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 10:00:05 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 10:00:05 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 10:00:05 PM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  44.03%  |  [0.46973045 0.45779938 0.46531153 0.47017234 0.46993811]  |  46.66 (+/- 0.95)  |  18.29  |  0.9501  |
03/07/2020 10:00:05 PM - INFO - |  2  |  BERNOULLI_NB  |  62.56%  |  [0.69200177 0.67741935 0.6752099  0.67388422 0.67992927]  |  67.97 (+/- 1.30)  |  0.05404  |  0.05173  |
03/07/2020 10:00:05 PM - INFO - |  3  |  COMPLEMENT_NB  |  71.22%  |  [0.776403   0.75828546 0.78126381 0.77507733 0.76967286]  |  77.21 (+/- 1.57)  |  0.06653  |  0.01054  |
03/07/2020 10:00:05 PM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  44.57%  |  [0.4874061  0.48608042 0.49138312 0.49359258 0.48099027]  |  48.79 (+/- 0.88)  |  8.941  |  0.0056  |
03/07/2020 10:00:05 PM - INFO - |  5  |  GRADIENT_BOOSTING_CLASSIFIER  |  59.60%  |  [0.6597437  0.62836942 0.64692886 0.66327883 0.64633068]  |  64.89 (+/- 2.46)  |  649.9  |  0.3243  |
03/07/2020 10:00:05 PM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  8.48%  |  [0.12417145 0.11842687 0.12063632 0.12240389 0.12113174]  |  12.14 (+/- 0.38)  |  0.00427  |  1.762  |
03/07/2020 10:00:05 PM - INFO - |  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  75.73 (+/- 2.09)  |  0.8605  |  0.008546  |
03/07/2020 10:00:05 PM - INFO - |  8  |  LOGISTIC_REGRESSION  |  69.28%  |  [0.75961114 0.74016792 0.75519222 0.7565179  0.73607427]  |  74.95 (+/- 1.90)  |  32.91  |  0.01159  |
03/07/2020 10:00:05 PM - INFO - |  9  |  MULTINOMIAL_NB  |  68.79%  |  [0.76049492 0.74149359 0.75519222 0.75695979 0.74270557]  |  75.14 (+/- 1.55)  |  0.05682  |  0.01024  |
03/07/2020 10:00:05 PM - INFO - |  10  |  NEAREST_CENTROID  |  66.70%  |  [0.71674768 0.71807335 0.71851525 0.71939903 0.70601238]  |  71.57 (+/- 0.99)  |  0.01737  |  0.01961  |
03/07/2020 10:00:05 PM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  69.69%  |  [0.76623951 0.74856385 0.76314627 0.77021653 0.7484527 ]  |  75.93 (+/- 1.82)  |  9.017  |  0.01138  |
03/07/2020 10:00:05 PM - INFO - |  12  |  PERCEPTRON  |  53.86%  |  [0.63146266 0.61025188 0.58948299 0.60450729 0.60477454]  |  60.81 (+/- 2.71)  |  0.7855  |  0.0223  |
03/07/2020 10:00:05 PM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  63.64%  |  [0.69067609 0.67255855 0.69067609 0.69730446 0.67241379]  |  68.47 (+/- 2.06)  |  39.54  |  1.178  |
03/07/2020 10:00:05 PM - INFO - |  14  |  RIDGE_CLASSIFIER  |  70.02%  |  [0.76756518 0.74502872 0.76800707 0.776403   0.76083112]  |  76.36 (+/- 2.10)  |  2.787  |  0.02243  |
```