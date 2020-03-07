## Grid search logs: IMDB using Binary Classification and 20 News Groups dataset (removing headers signatures and quoting)

### IMDB using Binary Classification

```
FINAL CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 01:41:55 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:41:55 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:41:55 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  80.24%  |  [0.801  0.803  0.799  0.7968 0.7986]  |  79.97 (+/- 0.43)  |  10.44  |  0.5797  |
03/07/2020 01:41:55 AM - INFO - |  2  |  BERNOULLI_NB  |  81.52%  |  [0.8384 0.8398 0.8524 0.8404 0.8524]  |  84.47 (+/- 1.27)  |  0.02364  |  0.02019  |
03/07/2020 01:41:55 AM - INFO - |  3  |  COMPLEMENT_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01559  |  0.008243  |
03/07/2020 01:41:55 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  71.44%  |  [0.7116 0.7088 0.7172 0.7138 0.7112]  |  71.25 (+/- 0.57)  |  23.48  |  0.03173  |
03/07/2020 01:41:55 AM - INFO - |  5  |  GRADIENT_BOOSTING_CLASSIFIER  |  80.72%  |  [0.805  0.8092 0.8048 0.7984 0.807 ]  |  80.49 (+/- 0.72)  |  50.35  |  0.05117  |
03/07/2020 01:41:55 AM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  73.29%  |  [0.8144 0.8248 0.8262 0.814  0.815 ]  |  81.89 (+/- 1.09)  |  0.009318  |  26.89  |
03/07/2020 01:41:55 AM - INFO - |  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  88.44 (+/- 0.98)  |  0.2275  |  0.003976  |
03/07/2020 01:41:55 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  88.39%  |  [0.8822 0.8946 0.8848 0.887  0.8852]  |  88.68 (+/- 0.84)  |  1.157  |  0.008317  |
03/07/2020 01:41:55 AM - INFO - |  9  |  MULTINOMIAL_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01562  |  0.008306  |
03/07/2020 01:41:55 AM - INFO - |  10  |  NEAREST_CENTROID  |  83.72%  |  [0.8316 0.838  0.8342 0.8392 0.8358]  |  83.58 (+/- 0.54)  |  0.01647  |  0.01182  |
03/07/2020 01:41:55 AM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  85.13%  |  [0.8666 0.8772 0.8748 0.8698 0.8644]  |  87.06 (+/- 0.96)  |  0.1929  |  0.003881  |
03/07/2020 01:41:55 AM - INFO - |  12  |  PERCEPTRON  |  84.43%  |  [0.861  0.868  0.8654 0.8614 0.8536]  |  86.19 (+/- 0.98)  |  0.1085  |  0.00389  |
03/07/2020 01:41:55 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  84.94%  |  [0.8498 0.8534 0.8468 0.8452 0.8464]  |  84.83 (+/- 0.59)  |  35.85  |  1.299  |
03/07/2020 01:41:55 AM - INFO - |  14  |  RIDGE_CLASSIFIER  |  86.90%  |  [0.8836 0.895  0.8888 0.882  0.8788]  |  88.56 (+/- 1.14)  |  0.4334  |  0.008216  |
03/07/2020 01:41:55 AM - INFO - 

FINAL CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 01:41:55 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:41:55 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:41:55 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  84.60%  |  [0.8398 0.8516 0.8416 0.8366 0.8416]  |  84.22 (+/- 1.00)  |  103.8  |  5.642  |
03/07/2020 01:41:55 AM - INFO - |  2  |  BERNOULLI_NB  |  81.28%  |  [0.8398 0.8424 0.8514 0.8396 0.8516]  |  84.50 (+/- 1.09)  |  0.02541  |  0.02126  |
03/07/2020 01:41:55 AM - INFO - |  3  |  COMPLEMENT_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01709  |  0.009721  |
03/07/2020 01:41:55 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  73.55%  |  [0.7342 0.728  0.7374 0.7336 0.7266]  |  73.20 (+/- 0.81)  |  13.62  |  0.01241  |
03/07/2020 01:41:55 AM - INFO - |  5  |  GRADIENT_BOOSTING_CLASSIFIER  |  82.87%  |  [0.828  0.8306 0.8238 0.8226 0.8284]  |  82.67 (+/- 0.60)  |  100.6  |  0.0646  |
03/07/2020 01:41:55 AM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  82.66%  |  [0.8632 0.8744 0.8694 0.864  0.8618]  |  86.66 (+/- 0.94)  |  0.007226  |  26.41  |
03/07/2020 01:41:55 AM - INFO - |  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  88.44 (+/- 0.98)  |  0.2293  |  0.003983  |
03/07/2020 01:41:55 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  87.75%  |  [0.8882 0.897  0.8878 0.8876 0.8818]  |  88.85 (+/- 0.97)  |  1.763  |  0.008308  |
03/07/2020 01:41:55 AM - INFO - |  9  |  MULTINOMIAL_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01575  |  0.008321  |
03/07/2020 01:41:55 AM - INFO - |  10  |  NEAREST_CENTROID  |  84.65%  |  [0.8426 0.8546 0.8426 0.8522 0.8502]  |  84.84 (+/- 0.99)  |  0.01649  |  0.01698  |
03/07/2020 01:41:55 AM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  88.08%  |  [0.8874 0.897  0.8884 0.8874 0.8852]  |  88.91 (+/- 0.82)  |  0.8401  |  0.003811  |
03/07/2020 01:41:55 AM - INFO - |  12  |  PERCEPTRON  |  80.64%  |  [0.8166 0.8264 0.8144 0.81   0.8102]  |  81.55 (+/- 1.20)  |  0.09111  |  0.00689  |
03/07/2020 01:41:55 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  85.39%  |  [0.8488 0.8534 0.8516 0.8448 0.8564]  |  85.10 (+/- 0.79)  |  33.64  |  2.238  |
03/07/2020 01:41:55 AM - INFO - |  14  |  RIDGE_CLASSIFIER  |  86.90%  |  [0.8838 0.8952 0.8892 0.882  0.8788]  |  88.58 (+/- 1.16)  |  0.4678  |  0.008244  |
03/07/2020 01:41:55 AM - INFO - 
```

### 20 News Groups dataset (removing headers signatures and quoting)

```
FINAL CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 05:32:42 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 05:32:42 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 05:32:42 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  36.52%  |  [0.39460893 0.39019001 0.39991162 0.39019001 0.40539346]  |  39.61 (+/- 1.18)  |  4.665  |  0.2575  |
03/07/2020 05:32:42 AM - INFO - |  2  |  BERNOULLI_NB  |  45.84%  |  [0.48033584 0.46133451 0.48563853 0.47547503 0.48938992]  |  47.84 (+/- 1.95)  |  0.05716  |  0.05391  |
03/07/2020 05:32:42 AM - INFO - |  3  |  COMPLEMENT_NB  |  71.04%  |  [0.77110031 0.75740168 0.77993814 0.7715422  0.76038904]  |  76.81 (+/- 1.64)  |  0.06305  |  0.01045  |
03/07/2020 05:32:42 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  42.95%  |  [0.4922669  0.47370747 0.46486964 0.47282369 0.48983201]  |  47.87 (+/- 2.11)  |  10.42  |  0.008561  |
03/07/2020 05:32:42 AM - INFO - |  5  |  GRADIENT_BOOSTING_CLASSIFIER  |  59.25%  |  [0.65178966 0.6195316  0.63013699 0.65090588 0.6357206 ]  |  63.76 (+/- 2.47)  |  334.0  |  0.1755  |
03/07/2020 05:32:42 AM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  7.20%  |  [0.06937693 0.08174989 0.07909854 0.08263367 0.08311229]  |  7.92 (+/- 1.02)  |  0.003774  |  2.168  |
03/07/2020 05:32:42 AM - INFO - |  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  75.73 (+/- 2.09)  |  0.8393  |  0.008751  |
03/07/2020 05:32:42 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  69.17%  |  [0.74458683 0.72646929 0.74193548 0.74768007 0.72413793]  |  73.70 (+/- 1.94)  |  29.69  |  0.01173  |
03/07/2020 05:32:42 AM - INFO - |  9  |  MULTINOMIAL_NB  |  65.64%  |  [0.7105612  0.69951392 0.69863014 0.69995581 0.69142352]  |  70.00 (+/- 1.22)  |  0.04449  |  0.01024  |
03/07/2020 05:32:42 AM - INFO - |  10  |  NEAREST_CENTROID  |  64.91%  |  [0.69774635 0.69244366 0.69818825 0.70393283 0.69849691]  |  69.82 (+/- 0.73)  |  0.02025  |  0.01422  |
03/07/2020 05:32:42 AM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  68.35%  |  [0.76005303 0.7308882  0.7410517  0.75784357 0.73386384]  |  74.47 (+/- 2.42)  |  1.014  |  0.02255  |
03/07/2020 05:32:42 AM - INFO - |  12  |  PERCEPTRON  |  63.34%  |  [0.69818825 0.67565179 0.6955369  0.70349094 0.68302387]  |  69.12 (+/- 2.05)  |  0.535  |  0.02502  |
03/07/2020 05:32:42 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  62.43%  |  [0.68228016 0.6597437  0.67034909 0.67609368 0.66534041]  |  67.08 (+/- 1.58)  |  36.12  |  0.619  |
03/07/2020 05:32:42 AM - INFO - |  14  |  RIDGE_CLASSIFIER  |  70.67%  |  [0.76623951 0.74900574 0.76889085 0.77330977 0.75464191]  |  76.24 (+/- 1.82)  |  2.184  |  0.02251  |
03/07/2020 05:32:42 AM - INFO - 

FINAL CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 05:32:42 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 05:32:42 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 05:32:42 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  44.04%  |  [0.4719399  0.45735749 0.46531153 0.46973045 0.47126437]  |  46.71 (+/- 1.08)  |  18.42  |  0.9331  |
03/07/2020 05:32:42 AM - INFO - |  2  |  BERNOULLI_NB  |  62.56%  |  [0.69200177 0.67741935 0.6752099  0.67388422 0.67992927]  |  67.97 (+/- 1.30)  |  0.0534  |  0.05116  |
03/07/2020 05:32:42 AM - INFO - |  3  |  COMPLEMENT_NB  |  71.22%  |  [0.776403   0.75828546 0.78126381 0.77507733 0.76967286]  |  77.21 (+/- 1.57)  |  0.06775  |  0.0101  |
03/07/2020 05:32:42 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  45.63%  |  [0.49536014 0.49005745 0.48077773 0.47989395 0.49955791]  |  48.91 (+/- 1.56)  |  4.511  |  0.005316  |
03/07/2020 05:32:42 AM - INFO - |  5  |  GRADIENT_BOOSTING_CLASSIFIER  |  59.65%  |  [0.65930181 0.62615996 0.64295183 0.65709236 0.65119363]  |  64.73 (+/- 2.40)  |  658.7  |  0.3424  |
03/07/2020 05:32:42 AM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  8.48%  |  [0.12417145 0.11842687 0.12063632 0.12240389 0.12113174]  |  12.14 (+/- 0.38)  |  0.004078  |  1.755  |
03/07/2020 05:32:42 AM - INFO - |  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  75.73 (+/- 2.09)  |  0.8455  |  0.008558  |
03/07/2020 05:32:42 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  69.28%  |  [0.75961114 0.74016792 0.75519222 0.7565179  0.73607427]  |  74.95 (+/- 1.90)  |  33.13  |  0.01171  |
03/07/2020 05:32:42 AM - INFO - |  9  |  MULTINOMIAL_NB  |  68.79%  |  [0.76049492 0.74149359 0.75519222 0.75695979 0.74270557]  |  75.14 (+/- 1.55)  |  0.05633  |  0.01006  |
03/07/2020 05:32:42 AM - INFO - |  10  |  NEAREST_CENTROID  |  66.70%  |  [0.71674768 0.71807335 0.71851525 0.71939903 0.70601238]  |  71.57 (+/- 0.99)  |  0.01741  |  0.02014  |
03/07/2020 05:32:42 AM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  69.64%  |  [0.76491383 0.74900574 0.76270437 0.76977464 0.74889478]  |  75.91 (+/- 1.71)  |  8.78  |  0.01103  |
03/07/2020 05:32:42 AM - INFO - |  12  |  PERCEPTRON  |  53.86%  |  [0.63146266 0.61025188 0.58948299 0.60450729 0.60477454]  |  60.81 (+/- 2.71)  |  0.7884  |  0.02289  |
03/07/2020 05:32:42 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  63.81%  |  [0.69774635 0.6752099  0.68448962 0.69421122 0.67992927]  |  68.63 (+/- 1.70)  |  40.29  |  1.201  |
03/07/2020 05:32:42 AM - INFO - |  14  |  RIDGE_CLASSIFIER  |  70.02%  |  [0.76756518 0.74502872 0.76800707 0.776403   0.76083112]  |  76.36 (+/- 2.10)  |  2.765  |  0.02252  |
```

All logs:

```
03/07/2020 12:43:50 AM - INFO - 
>>> GRID SEARCH
03/07/2020 12:43:50 AM - INFO - 

03/07/2020 12:43:50 AM - INFO - ################################################################################
03/07/2020 12:43:50 AM - INFO - 1)
03/07/2020 12:43:50 AM - INFO - ********************************************************************************
03/07/2020 12:43:50 AM - INFO - Classifier: ADA_BOOST_CLASSIFIER, Dataset: IMDB_REVIEWS
03/07/2020 12:43:50 AM - INFO - ********************************************************************************
03/07/2020 12:43:57 AM - INFO - 

Performing grid search...

03/07/2020 12:43:57 AM - INFO - Parameters:
03/07/2020 12:43:57 AM - INFO - {'classifier__learning_rate': [0.1, 1], 'classifier__n_estimators': [200, 500]}
03/07/2020 01:00:01 AM - INFO - 	Done in 964.406s
03/07/2020 01:00:01 AM - INFO - 	Best score: 0.842
03/07/2020 01:00:01 AM - INFO - 	Best parameters set:
03/07/2020 01:00:01 AM - INFO - 		classifier__learning_rate: 1
03/07/2020 01:00:01 AM - INFO - 		classifier__n_estimators: 500
03/07/2020 01:00:01 AM - INFO - 

USING ADA_BOOST_CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 01:00:01 AM - INFO - ________________________________________________________________________________
03/07/2020 01:00:01 AM - INFO - Training: 
03/07/2020 01:00:01 AM - INFO - AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
                   n_estimators=50, random_state=None)
03/07/2020 01:00:11 AM - INFO - Train time: 10.436s
03/07/2020 01:00:12 AM - INFO - Test time:  0.580s
03/07/2020 01:00:12 AM - INFO - Accuracy score:   0.802
03/07/2020 01:00:12 AM - INFO - 

===> Classification Report:

03/07/2020 01:00:12 AM - INFO -               precision    recall  f1-score   support

           0       0.82      0.77      0.80     12500
           1       0.78      0.84      0.81     12500

    accuracy                           0.80     25000
   macro avg       0.80      0.80      0.80     25000
weighted avg       0.80      0.80      0.80     25000

03/07/2020 01:00:12 AM - INFO - 

Cross validation:
03/07/2020 01:00:43 AM - INFO - 	accuracy: 5-fold cross validation: [0.801  0.803  0.799  0.7968 0.7986]
03/07/2020 01:00:43 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 79.97 (+/- 0.43)
03/07/2020 01:00:43 AM - INFO - 

USING ADA_BOOST_CLASSIFIER WITH BEST PARAMETERS: {'learning_rate': 1, 'n_estimators': 500}
03/07/2020 01:00:43 AM - INFO - ________________________________________________________________________________
03/07/2020 01:00:43 AM - INFO - Training: 
03/07/2020 01:00:43 AM - INFO - AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1,
                   n_estimators=500, random_state=None)
03/07/2020 01:02:27 AM - INFO - Train time: 103.792s
03/07/2020 01:02:33 AM - INFO - Test time:  5.642s
03/07/2020 01:02:33 AM - INFO - Accuracy score:   0.846
03/07/2020 01:02:33 AM - INFO - 

===> Classification Report:

03/07/2020 01:02:33 AM - INFO -               precision    recall  f1-score   support

           0       0.85      0.83      0.84     12500
           1       0.84      0.86      0.85     12500

    accuracy                           0.85     25000
   macro avg       0.85      0.85      0.85     25000
weighted avg       0.85      0.85      0.85     25000

03/07/2020 01:02:33 AM - INFO - 

Cross validation:
03/07/2020 01:07:50 AM - INFO - 	accuracy: 5-fold cross validation: [0.8398 0.8516 0.8416 0.8366 0.8416]
03/07/2020 01:07:50 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 84.22 (+/- 1.00)
03/07/2020 01:07:50 AM - INFO - It took 1440.2348561286926 seconds
03/07/2020 01:07:50 AM - INFO - ********************************************************************************
03/07/2020 01:07:50 AM - INFO - ################################################################################
03/07/2020 01:07:50 AM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 01:07:50 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:07:50 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:07:50 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  80.24%  |  [0.801  0.803  0.799  0.7968 0.7986]  |  79.97 (+/- 0.43)  |  10.44  |  0.5797  |
03/07/2020 01:07:50 AM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 01:07:50 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:07:50 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:07:50 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  84.60%  |  [0.8398 0.8516 0.8416 0.8366 0.8416]  |  84.22 (+/- 1.00)  |  103.8  |  5.642  |
03/07/2020 01:07:50 AM - INFO - 

03/07/2020 01:07:50 AM - INFO - ################################################################################
03/07/2020 01:07:50 AM - INFO - 2)
03/07/2020 01:07:50 AM - INFO - ********************************************************************************
03/07/2020 01:07:50 AM - INFO - Classifier: DECISION_TREE_CLASSIFIER, Dataset: IMDB_REVIEWS
03/07/2020 01:07:50 AM - INFO - ********************************************************************************
03/07/2020 01:07:57 AM - INFO - 

Performing grid search...

03/07/2020 01:07:57 AM - INFO - Parameters:
03/07/2020 01:07:57 AM - INFO - {'classifier__criterion': ['entropy', 'gini'], 'classifier__splitter': ['best', 'random'], 'classifier__min_samples_split': [2, 100, 250]}
03/07/2020 01:10:11 AM - INFO - 	Done in 134.170s
03/07/2020 01:10:11 AM - INFO - 	Best score: 0.731
03/07/2020 01:10:11 AM - INFO - 	Best parameters set:
03/07/2020 01:10:11 AM - INFO - 		classifier__criterion: 'entropy'
03/07/2020 01:10:11 AM - INFO - 		classifier__min_samples_split: 250
03/07/2020 01:10:11 AM - INFO - 		classifier__splitter: 'best'
03/07/2020 01:10:11 AM - INFO - 

USING DECISION_TREE_CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 01:10:11 AM - INFO - ________________________________________________________________________________
03/07/2020 01:10:11 AM - INFO - Training: 
03/07/2020 01:10:11 AM - INFO - DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=None, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=None, splitter='best')
03/07/2020 01:10:34 AM - INFO - Train time: 23.479s
03/07/2020 01:10:34 AM - INFO - Test time:  0.032s
03/07/2020 01:10:34 AM - INFO - Accuracy score:   0.714
03/07/2020 01:10:34 AM - INFO - 

===> Classification Report:

03/07/2020 01:10:34 AM - INFO -               precision    recall  f1-score   support

           0       0.71      0.72      0.72     12500
           1       0.72      0.71      0.71     12500

    accuracy                           0.71     25000
   macro avg       0.71      0.71      0.71     25000
weighted avg       0.71      0.71      0.71     25000

03/07/2020 01:10:34 AM - INFO - 

Cross validation:
03/07/2020 01:10:54 AM - INFO - 	accuracy: 5-fold cross validation: [0.7116 0.7088 0.7172 0.7138 0.7112]
03/07/2020 01:10:54 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 71.25 (+/- 0.57)
03/07/2020 01:10:54 AM - INFO - 

USING DECISION_TREE_CLASSIFIER WITH BEST PARAMETERS: {'criterion': 'entropy', 'min_samples_split': 250, 'splitter': 'best'}
03/07/2020 01:10:54 AM - INFO - ________________________________________________________________________________
03/07/2020 01:10:54 AM - INFO - Training: 
03/07/2020 01:10:54 AM - INFO - DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='entropy',
                       max_depth=None, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=250,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=None, splitter='best')
03/07/2020 01:11:08 AM - INFO - Train time: 13.616s
03/07/2020 01:11:08 AM - INFO - Test time:  0.012s
03/07/2020 01:11:08 AM - INFO - Accuracy score:   0.736
03/07/2020 01:11:08 AM - INFO - 

===> Classification Report:

03/07/2020 01:11:08 AM - INFO -               precision    recall  f1-score   support

           0       0.73      0.75      0.74     12500
           1       0.74      0.73      0.73     12500

    accuracy                           0.74     25000
   macro avg       0.74      0.74      0.74     25000
weighted avg       0.74      0.74      0.74     25000

03/07/2020 01:11:08 AM - INFO - 

Cross validation:
03/07/2020 01:11:20 AM - INFO - 	accuracy: 5-fold cross validation: [0.7342 0.728  0.7374 0.7336 0.7266]
03/07/2020 01:11:20 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 73.20 (+/- 0.81)
03/07/2020 01:11:20 AM - INFO - It took 209.8690619468689 seconds
03/07/2020 01:11:20 AM - INFO - ********************************************************************************
03/07/2020 01:11:20 AM - INFO - ################################################################################
03/07/2020 01:11:20 AM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 01:11:20 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:11:20 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:11:20 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  80.24%  |  [0.801  0.803  0.799  0.7968 0.7986]  |  79.97 (+/- 0.43)  |  10.44  |  0.5797  |
03/07/2020 01:11:20 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  71.44%  |  [0.7116 0.7088 0.7172 0.7138 0.7112]  |  71.25 (+/- 0.57)  |  23.48  |  0.03173  |
03/07/2020 01:11:20 AM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 01:11:20 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:11:20 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:11:20 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  84.60%  |  [0.8398 0.8516 0.8416 0.8366 0.8416]  |  84.22 (+/- 1.00)  |  103.8  |  5.642  |
03/07/2020 01:11:20 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  73.55%  |  [0.7342 0.728  0.7374 0.7336 0.7266]  |  73.20 (+/- 0.81)  |  13.62  |  0.01241  |
03/07/2020 01:11:20 AM - INFO - 

03/07/2020 01:11:20 AM - INFO - ################################################################################
03/07/2020 01:11:20 AM - INFO - 3)
03/07/2020 01:11:20 AM - INFO - ********************************************************************************
03/07/2020 01:11:20 AM - INFO - Classifier: LINEAR_SVC, Dataset: IMDB_REVIEWS
03/07/2020 01:11:20 AM - INFO - ********************************************************************************
03/07/2020 01:11:26 AM - INFO - 

Performing grid search...

03/07/2020 01:11:26 AM - INFO - Parameters:
03/07/2020 01:11:26 AM - INFO - {'classifier__C': [0.01, 1.0], 'classifier__multi_class': ['ovr', 'crammer_singer'], 'classifier__tol': [0.0001, 0.001]}
03/07/2020 01:11:33 AM - INFO - 	Done in 6.148s
03/07/2020 01:11:33 AM - INFO - 	Best score: 0.884
03/07/2020 01:11:33 AM - INFO - 	Best parameters set:
03/07/2020 01:11:33 AM - INFO - 		classifier__C: 1.0
03/07/2020 01:11:33 AM - INFO - 		classifier__multi_class: 'ovr'
03/07/2020 01:11:33 AM - INFO - 		classifier__tol: 0.0001
03/07/2020 01:11:33 AM - INFO - 

USING LINEAR_SVC WITH DEFAULT PARAMETERS
03/07/2020 01:11:33 AM - INFO - ________________________________________________________________________________
03/07/2020 01:11:33 AM - INFO - Training: 
03/07/2020 01:11:33 AM - INFO - LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
          intercept_scaling=1, loss='squared_hinge', max_iter=1000,
          multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
          verbose=0)
03/07/2020 01:11:33 AM - INFO - Train time: 0.228s
03/07/2020 01:11:33 AM - INFO - Test time:  0.004s
03/07/2020 01:11:33 AM - INFO - Accuracy score:   0.871
03/07/2020 01:11:33 AM - INFO - 

===> Classification Report:

03/07/2020 01:11:33 AM - INFO -               precision    recall  f1-score   support

           0       0.87      0.88      0.87     12500
           1       0.88      0.87      0.87     12500

    accuracy                           0.87     25000
   macro avg       0.87      0.87      0.87     25000
weighted avg       0.87      0.87      0.87     25000

03/07/2020 01:11:33 AM - INFO - 

Cross validation:
03/07/2020 01:11:33 AM - INFO - 	accuracy: 5-fold cross validation: [0.8838 0.8932 0.883  0.8836 0.8782]
03/07/2020 01:11:33 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 88.44 (+/- 0.98)
03/07/2020 01:11:33 AM - INFO - 

USING LINEAR_SVC WITH BEST PARAMETERS: {'C': 1.0, 'multi_class': 'ovr', 'tol': 0.0001}
03/07/2020 01:11:33 AM - INFO - ________________________________________________________________________________
03/07/2020 01:11:33 AM - INFO - Training: 
03/07/2020 01:11:33 AM - INFO - LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
          intercept_scaling=1, loss='squared_hinge', max_iter=1000,
          multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
          verbose=0)
03/07/2020 01:11:34 AM - INFO - Train time: 0.229s
03/07/2020 01:11:34 AM - INFO - Test time:  0.004s
03/07/2020 01:11:34 AM - INFO - Accuracy score:   0.871
03/07/2020 01:11:34 AM - INFO - 

===> Classification Report:

03/07/2020 01:11:34 AM - INFO -               precision    recall  f1-score   support

           0       0.87      0.88      0.87     12500
           1       0.88      0.87      0.87     12500

    accuracy                           0.87     25000
   macro avg       0.87      0.87      0.87     25000
weighted avg       0.87      0.87      0.87     25000

03/07/2020 01:11:34 AM - INFO - 

Cross validation:
03/07/2020 01:11:34 AM - INFO - 	accuracy: 5-fold cross validation: [0.8838 0.8932 0.883  0.8836 0.8782]
03/07/2020 01:11:34 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 88.44 (+/- 0.98)
03/07/2020 01:11:34 AM - INFO - It took 14.011280298233032 seconds
03/07/2020 01:11:34 AM - INFO - ********************************************************************************
03/07/2020 01:11:34 AM - INFO - ################################################################################
03/07/2020 01:11:34 AM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 01:11:34 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:11:34 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:11:34 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  80.24%  |  [0.801  0.803  0.799  0.7968 0.7986]  |  79.97 (+/- 0.43)  |  10.44  |  0.5797  |
03/07/2020 01:11:34 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  71.44%  |  [0.7116 0.7088 0.7172 0.7138 0.7112]  |  71.25 (+/- 0.57)  |  23.48  |  0.03173  |
03/07/2020 01:11:34 AM - INFO - |  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  88.44 (+/- 0.98)  |  0.2275  |  0.003976  |
03/07/2020 01:11:34 AM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 01:11:34 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:11:34 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:11:34 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  84.60%  |  [0.8398 0.8516 0.8416 0.8366 0.8416]  |  84.22 (+/- 1.00)  |  103.8  |  5.642  |
03/07/2020 01:11:34 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  73.55%  |  [0.7342 0.728  0.7374 0.7336 0.7266]  |  73.20 (+/- 0.81)  |  13.62  |  0.01241  |
03/07/2020 01:11:34 AM - INFO - |  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  88.44 (+/- 0.98)  |  0.2293  |  0.003983  |
03/07/2020 01:11:34 AM - INFO - 

03/07/2020 01:11:34 AM - INFO - ################################################################################
03/07/2020 01:11:34 AM - INFO - 4)
03/07/2020 01:11:34 AM - INFO - ********************************************************************************
03/07/2020 01:11:34 AM - INFO - Classifier: LOGISTIC_REGRESSION, Dataset: IMDB_REVIEWS
03/07/2020 01:11:34 AM - INFO - ********************************************************************************
03/07/2020 01:11:40 AM - INFO - 

Performing grid search...

03/07/2020 01:11:40 AM - INFO - Parameters:
03/07/2020 01:11:40 AM - INFO - {'classifier__C': [1, 10], 'classifier__tol': [0.001, 0.01]}
03/07/2020 01:11:49 AM - INFO - 	Done in 8.606s
03/07/2020 01:11:49 AM - INFO - 	Best score: 0.888
03/07/2020 01:11:49 AM - INFO - 	Best parameters set:
03/07/2020 01:11:49 AM - INFO - 		classifier__C: 10
03/07/2020 01:11:49 AM - INFO - 		classifier__tol: 0.01
03/07/2020 01:11:49 AM - INFO - 

USING LOGISTIC_REGRESSION WITH DEFAULT PARAMETERS
03/07/2020 01:11:49 AM - INFO - ________________________________________________________________________________
03/07/2020 01:11:49 AM - INFO - Training: 
03/07/2020 01:11:49 AM - INFO - LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)
03/07/2020 01:11:50 AM - INFO - Train time: 1.157s
03/07/2020 01:11:50 AM - INFO - Test time:  0.008s
03/07/2020 01:11:50 AM - INFO - Accuracy score:   0.884
03/07/2020 01:11:50 AM - INFO - 

===> Classification Report:

03/07/2020 01:11:50 AM - INFO -               precision    recall  f1-score   support

           0       0.89      0.88      0.88     12500
           1       0.88      0.89      0.88     12500

    accuracy                           0.88     25000
   macro avg       0.88      0.88      0.88     25000
weighted avg       0.88      0.88      0.88     25000

03/07/2020 01:11:50 AM - INFO - 

Cross validation:
03/07/2020 01:11:51 AM - INFO - 	accuracy: 5-fold cross validation: [0.8822 0.8946 0.8848 0.887  0.8852]
03/07/2020 01:11:51 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 88.68 (+/- 0.84)
03/07/2020 01:11:51 AM - INFO - 

USING LOGISTIC_REGRESSION WITH BEST PARAMETERS: {'C': 10, 'tol': 0.01}
03/07/2020 01:11:51 AM - INFO - ________________________________________________________________________________
03/07/2020 01:11:51 AM - INFO - Training: 
03/07/2020 01:11:51 AM - INFO - LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.01, verbose=0,
                   warm_start=False)
03/07/2020 01:11:53 AM - INFO - Train time: 1.763s
03/07/2020 01:11:53 AM - INFO - Test time:  0.008s
03/07/2020 01:11:53 AM - INFO - Accuracy score:   0.877
03/07/2020 01:11:53 AM - INFO - 

===> Classification Report:

03/07/2020 01:11:53 AM - INFO -               precision    recall  f1-score   support

           0       0.88      0.88      0.88     12500
           1       0.88      0.87      0.88     12500

    accuracy                           0.88     25000
   macro avg       0.88      0.88      0.88     25000
weighted avg       0.88      0.88      0.88     25000

03/07/2020 01:11:53 AM - INFO - 

Cross validation:
03/07/2020 01:11:55 AM - INFO - 	accuracy: 5-fold cross validation: [0.8882 0.897  0.8878 0.8876 0.8818]
03/07/2020 01:11:55 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 88.85 (+/- 0.97)
03/07/2020 01:11:55 AM - INFO - It took 21.25937008857727 seconds
03/07/2020 01:11:55 AM - INFO - ********************************************************************************
03/07/2020 01:11:55 AM - INFO - ################################################################################
03/07/2020 01:11:55 AM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 01:11:55 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:11:55 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:11:55 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  80.24%  |  [0.801  0.803  0.799  0.7968 0.7986]  |  79.97 (+/- 0.43)  |  10.44  |  0.5797  |
03/07/2020 01:11:55 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  71.44%  |  [0.7116 0.7088 0.7172 0.7138 0.7112]  |  71.25 (+/- 0.57)  |  23.48  |  0.03173  |
03/07/2020 01:11:55 AM - INFO - |  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  88.44 (+/- 0.98)  |  0.2275  |  0.003976  |
03/07/2020 01:11:55 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  88.39%  |  [0.8822 0.8946 0.8848 0.887  0.8852]  |  88.68 (+/- 0.84)  |  1.157  |  0.008317  |
03/07/2020 01:11:55 AM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 01:11:55 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:11:55 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:11:55 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  84.60%  |  [0.8398 0.8516 0.8416 0.8366 0.8416]  |  84.22 (+/- 1.00)  |  103.8  |  5.642  |
03/07/2020 01:11:55 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  73.55%  |  [0.7342 0.728  0.7374 0.7336 0.7266]  |  73.20 (+/- 0.81)  |  13.62  |  0.01241  |
03/07/2020 01:11:55 AM - INFO - |  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  88.44 (+/- 0.98)  |  0.2293  |  0.003983  |
03/07/2020 01:11:55 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  87.75%  |  [0.8882 0.897  0.8878 0.8876 0.8818]  |  88.85 (+/- 0.97)  |  1.763  |  0.008308  |
03/07/2020 01:11:55 AM - INFO - 

03/07/2020 01:11:55 AM - INFO - ################################################################################
03/07/2020 01:11:55 AM - INFO - 5)
03/07/2020 01:11:55 AM - INFO - ********************************************************************************
03/07/2020 01:11:55 AM - INFO - Classifier: RANDOM_FOREST_CLASSIFIER, Dataset: IMDB_REVIEWS
03/07/2020 01:11:55 AM - INFO - ********************************************************************************
03/07/2020 01:12:02 AM - INFO - 

Performing grid search...

03/07/2020 01:12:02 AM - INFO - Parameters:
03/07/2020 01:12:02 AM - INFO - {'classifier__min_samples_leaf': [1, 2], 'classifier__min_samples_split': [2, 5], 'classifier__n_estimators': [100, 200]}
03/07/2020 01:17:50 AM - INFO - 	Done in 348.349s
03/07/2020 01:17:50 AM - INFO - 	Best score: 0.853
03/07/2020 01:17:50 AM - INFO - 	Best parameters set:
03/07/2020 01:17:50 AM - INFO - 		classifier__min_samples_leaf: 2
03/07/2020 01:17:50 AM - INFO - 		classifier__min_samples_split: 5
03/07/2020 01:17:50 AM - INFO - 		classifier__n_estimators: 200
03/07/2020 01:17:50 AM - INFO - 

USING RANDOM_FOREST_CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 01:17:50 AM - INFO - ________________________________________________________________________________
03/07/2020 01:17:50 AM - INFO - Training: 
03/07/2020 01:17:50 AM - INFO - RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
03/07/2020 01:18:26 AM - INFO - Train time: 35.853s
03/07/2020 01:18:27 AM - INFO - Test time:  1.299s
03/07/2020 01:18:27 AM - INFO - Accuracy score:   0.849
03/07/2020 01:18:27 AM - INFO - 

===> Classification Report:

03/07/2020 01:18:27 AM - INFO -               precision    recall  f1-score   support

           0       0.84      0.86      0.85     12500
           1       0.86      0.84      0.85     12500

    accuracy                           0.85     25000
   macro avg       0.85      0.85      0.85     25000
weighted avg       0.85      0.85      0.85     25000

03/07/2020 01:18:27 AM - INFO - 

Cross validation:
03/07/2020 01:19:12 AM - INFO - 	accuracy: 5-fold cross validation: [0.8498 0.8534 0.8468 0.8452 0.8464]
03/07/2020 01:19:12 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 84.83 (+/- 0.59)
03/07/2020 01:19:12 AM - INFO - 

USING RANDOM_FOREST_CLASSIFIER WITH BEST PARAMETERS: {'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 200}
03/07/2020 01:19:12 AM - INFO - ________________________________________________________________________________
03/07/2020 01:19:12 AM - INFO - Training: 
03/07/2020 01:19:12 AM - INFO - RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=2, min_samples_split=5,
                       min_weight_fraction_leaf=0.0, n_estimators=200,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
03/07/2020 01:19:45 AM - INFO - Train time: 33.643s
03/07/2020 01:19:48 AM - INFO - Test time:  2.238s
03/07/2020 01:19:48 AM - INFO - Accuracy score:   0.854
03/07/2020 01:19:48 AM - INFO - 

===> Classification Report:

03/07/2020 01:19:48 AM - INFO -               precision    recall  f1-score   support

           0       0.85      0.85      0.85     12500
           1       0.85      0.86      0.85     12500

    accuracy                           0.85     25000
   macro avg       0.85      0.85      0.85     25000
weighted avg       0.85      0.85      0.85     25000

03/07/2020 01:19:48 AM - INFO - 

Cross validation:
03/07/2020 01:20:38 AM - INFO - 	accuracy: 5-fold cross validation: [0.8488 0.8534 0.8516 0.8448 0.8564]
03/07/2020 01:20:38 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 85.10 (+/- 0.79)
03/07/2020 01:20:38 AM - INFO - It took 522.24636054039 seconds
03/07/2020 01:20:38 AM - INFO - ********************************************************************************
03/07/2020 01:20:38 AM - INFO - ################################################################################
03/07/2020 01:20:38 AM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 01:20:38 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:20:38 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:20:38 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  80.24%  |  [0.801  0.803  0.799  0.7968 0.7986]  |  79.97 (+/- 0.43)  |  10.44  |  0.5797  |
03/07/2020 01:20:38 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  71.44%  |  [0.7116 0.7088 0.7172 0.7138 0.7112]  |  71.25 (+/- 0.57)  |  23.48  |  0.03173  |
03/07/2020 01:20:38 AM - INFO - |  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  88.44 (+/- 0.98)  |  0.2275  |  0.003976  |
03/07/2020 01:20:38 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  88.39%  |  [0.8822 0.8946 0.8848 0.887  0.8852]  |  88.68 (+/- 0.84)  |  1.157  |  0.008317  |
03/07/2020 01:20:38 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  84.94%  |  [0.8498 0.8534 0.8468 0.8452 0.8464]  |  84.83 (+/- 0.59)  |  35.85  |  1.299  |
03/07/2020 01:20:38 AM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 01:20:38 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:20:38 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:20:38 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  84.60%  |  [0.8398 0.8516 0.8416 0.8366 0.8416]  |  84.22 (+/- 1.00)  |  103.8  |  5.642  |
03/07/2020 01:20:38 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  73.55%  |  [0.7342 0.728  0.7374 0.7336 0.7266]  |  73.20 (+/- 0.81)  |  13.62  |  0.01241  |
03/07/2020 01:20:38 AM - INFO - |  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  88.44 (+/- 0.98)  |  0.2293  |  0.003983  |
03/07/2020 01:20:38 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  87.75%  |  [0.8882 0.897  0.8878 0.8876 0.8818]  |  88.85 (+/- 0.97)  |  1.763  |  0.008308  |
03/07/2020 01:20:38 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  85.39%  |  [0.8488 0.8534 0.8516 0.8448 0.8564]  |  85.10 (+/- 0.79)  |  33.64  |  2.238  |
03/07/2020 01:20:38 AM - INFO - 

03/07/2020 01:20:38 AM - INFO - ################################################################################
03/07/2020 01:20:38 AM - INFO - 6)
03/07/2020 01:20:38 AM - INFO - ********************************************************************************
03/07/2020 01:20:38 AM - INFO - Classifier: BERNOULLI_NB, Dataset: IMDB_REVIEWS
03/07/2020 01:20:38 AM - INFO - ********************************************************************************
03/07/2020 01:20:44 AM - INFO - 

Performing grid search...

03/07/2020 01:20:44 AM - INFO - Parameters:
03/07/2020 01:20:44 AM - INFO - {'classifier__alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0], 'classifier__binarize': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0], 'classifier__fit_prior': [False, True]}
03/07/2020 01:21:01 AM - INFO - 	Done in 16.959s
03/07/2020 01:21:01 AM - INFO - 	Best score: 0.845
03/07/2020 01:21:01 AM - INFO - 	Best parameters set:
03/07/2020 01:21:01 AM - INFO - 		classifier__alpha: 0.5
03/07/2020 01:21:01 AM - INFO - 		classifier__binarize: 0.0001
03/07/2020 01:21:01 AM - INFO - 		classifier__fit_prior: False
03/07/2020 01:21:01 AM - INFO - 

USING BERNOULLI_NB WITH DEFAULT PARAMETERS
03/07/2020 01:21:01 AM - INFO - ________________________________________________________________________________
03/07/2020 01:21:01 AM - INFO - Training: 
03/07/2020 01:21:01 AM - INFO - BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
03/07/2020 01:21:01 AM - INFO - Train time: 0.024s
03/07/2020 01:21:01 AM - INFO - Test time:  0.020s
03/07/2020 01:21:01 AM - INFO - Accuracy score:   0.815
03/07/2020 01:21:01 AM - INFO - 

===> Classification Report:

03/07/2020 01:21:01 AM - INFO -               precision    recall  f1-score   support

           0       0.77      0.89      0.83     12500
           1       0.87      0.74      0.80     12500

    accuracy                           0.82     25000
   macro avg       0.82      0.82      0.81     25000
weighted avg       0.82      0.82      0.81     25000

03/07/2020 01:21:01 AM - INFO - 

Cross validation:
03/07/2020 01:21:01 AM - INFO - 	accuracy: 5-fold cross validation: [0.8384 0.8398 0.8524 0.8404 0.8524]
03/07/2020 01:21:01 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 84.47 (+/- 1.27)
03/07/2020 01:21:01 AM - INFO - 

USING BERNOULLI_NB WITH BEST PARAMETERS: {'alpha': 0.5, 'binarize': 0.0001, 'fit_prior': False}
03/07/2020 01:21:01 AM - INFO - ________________________________________________________________________________
03/07/2020 01:21:01 AM - INFO - Training: 
03/07/2020 01:21:01 AM - INFO - BernoulliNB(alpha=0.5, binarize=0.0001, class_prior=None, fit_prior=False)
03/07/2020 01:21:01 AM - INFO - Train time: 0.025s
03/07/2020 01:21:01 AM - INFO - Test time:  0.021s
03/07/2020 01:21:01 AM - INFO - Accuracy score:   0.813
03/07/2020 01:21:01 AM - INFO - 

===> Classification Report:

03/07/2020 01:21:01 AM - INFO -               precision    recall  f1-score   support

           0       0.77      0.89      0.83     12500
           1       0.87      0.74      0.80     12500

    accuracy                           0.81     25000
   macro avg       0.82      0.81      0.81     25000
weighted avg       0.82      0.81      0.81     25000

03/07/2020 01:21:01 AM - INFO - 

Cross validation:
03/07/2020 01:21:01 AM - INFO - 	accuracy: 5-fold cross validation: [0.8398 0.8424 0.8514 0.8396 0.8516]
03/07/2020 01:21:01 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 84.50 (+/- 1.09)
03/07/2020 01:21:01 AM - INFO - It took 23.81202507019043 seconds
03/07/2020 01:21:01 AM - INFO - ********************************************************************************
03/07/2020 01:21:01 AM - INFO - ################################################################################
03/07/2020 01:21:01 AM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 01:21:01 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:21:01 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:21:01 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  80.24%  |  [0.801  0.803  0.799  0.7968 0.7986]  |  79.97 (+/- 0.43)  |  10.44  |  0.5797  |
03/07/2020 01:21:01 AM - INFO - |  2  |  BERNOULLI_NB  |  81.52%  |  [0.8384 0.8398 0.8524 0.8404 0.8524]  |  84.47 (+/- 1.27)  |  0.02364  |  0.02019  |
03/07/2020 01:21:01 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  71.44%  |  [0.7116 0.7088 0.7172 0.7138 0.7112]  |  71.25 (+/- 0.57)  |  23.48  |  0.03173  |
03/07/2020 01:21:01 AM - INFO - |  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  88.44 (+/- 0.98)  |  0.2275  |  0.003976  |
03/07/2020 01:21:01 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  88.39%  |  [0.8822 0.8946 0.8848 0.887  0.8852]  |  88.68 (+/- 0.84)  |  1.157  |  0.008317  |
03/07/2020 01:21:01 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  84.94%  |  [0.8498 0.8534 0.8468 0.8452 0.8464]  |  84.83 (+/- 0.59)  |  35.85  |  1.299  |
03/07/2020 01:21:01 AM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 01:21:01 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:21:01 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:21:01 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  84.60%  |  [0.8398 0.8516 0.8416 0.8366 0.8416]  |  84.22 (+/- 1.00)  |  103.8  |  5.642  |
03/07/2020 01:21:01 AM - INFO - |  2  |  BERNOULLI_NB  |  81.28%  |  [0.8398 0.8424 0.8514 0.8396 0.8516]  |  84.50 (+/- 1.09)  |  0.02541  |  0.02126  |
03/07/2020 01:21:01 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  73.55%  |  [0.7342 0.728  0.7374 0.7336 0.7266]  |  73.20 (+/- 0.81)  |  13.62  |  0.01241  |
03/07/2020 01:21:01 AM - INFO - |  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  88.44 (+/- 0.98)  |  0.2293  |  0.003983  |
03/07/2020 01:21:01 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  87.75%  |  [0.8882 0.897  0.8878 0.8876 0.8818]  |  88.85 (+/- 0.97)  |  1.763  |  0.008308  |
03/07/2020 01:21:01 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  85.39%  |  [0.8488 0.8534 0.8516 0.8448 0.8564]  |  85.10 (+/- 0.79)  |  33.64  |  2.238  |
03/07/2020 01:21:01 AM - INFO - 

03/07/2020 01:21:01 AM - INFO - ################################################################################
03/07/2020 01:21:01 AM - INFO - 7)
03/07/2020 01:21:01 AM - INFO - ********************************************************************************
03/07/2020 01:21:01 AM - INFO - Classifier: COMPLEMENT_NB, Dataset: IMDB_REVIEWS
03/07/2020 01:21:01 AM - INFO - ********************************************************************************
03/07/2020 01:21:08 AM - INFO - 

Performing grid search...

03/07/2020 01:21:08 AM - INFO - Parameters:
03/07/2020 01:21:08 AM - INFO - {'classifier__alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0], 'classifier__fit_prior': [False, True], 'classifier__norm': [False, True]}
03/07/2020 01:21:11 AM - INFO - 	Done in 3.220s
03/07/2020 01:21:11 AM - INFO - 	Best score: 0.865
03/07/2020 01:21:11 AM - INFO - 	Best parameters set:
03/07/2020 01:21:11 AM - INFO - 		classifier__alpha: 1.0
03/07/2020 01:21:11 AM - INFO - 		classifier__fit_prior: False
03/07/2020 01:21:11 AM - INFO - 		classifier__norm: False
03/07/2020 01:21:11 AM - INFO - 

USING COMPLEMENT_NB WITH DEFAULT PARAMETERS
03/07/2020 01:21:11 AM - INFO - ________________________________________________________________________________
03/07/2020 01:21:11 AM - INFO - Training: 
03/07/2020 01:21:11 AM - INFO - ComplementNB(alpha=1.0, class_prior=None, fit_prior=True, norm=False)
03/07/2020 01:21:11 AM - INFO - Train time: 0.016s
03/07/2020 01:21:11 AM - INFO - Test time:  0.008s
03/07/2020 01:21:11 AM - INFO - Accuracy score:   0.839
03/07/2020 01:21:11 AM - INFO - 

===> Classification Report:

03/07/2020 01:21:11 AM - INFO -               precision    recall  f1-score   support

           0       0.82      0.88      0.85     12500
           1       0.87      0.80      0.83     12500

    accuracy                           0.84     25000
   macro avg       0.84      0.84      0.84     25000
weighted avg       0.84      0.84      0.84     25000

03/07/2020 01:21:11 AM - INFO - 

Cross validation:
03/07/2020 01:21:11 AM - INFO - 	accuracy: 5-fold cross validation: [0.8564 0.8678 0.8682 0.8678 0.8672]
03/07/2020 01:21:11 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 86.55 (+/- 0.91)
03/07/2020 01:21:11 AM - INFO - 

USING COMPLEMENT_NB WITH BEST PARAMETERS: {'alpha': 1.0, 'fit_prior': False, 'norm': False}
03/07/2020 01:21:11 AM - INFO - ________________________________________________________________________________
03/07/2020 01:21:11 AM - INFO - Training: 
03/07/2020 01:21:11 AM - INFO - ComplementNB(alpha=1.0, class_prior=None, fit_prior=False, norm=False)
03/07/2020 01:21:11 AM - INFO - Train time: 0.017s
03/07/2020 01:21:11 AM - INFO - Test time:  0.010s
03/07/2020 01:21:11 AM - INFO - Accuracy score:   0.839
03/07/2020 01:21:11 AM - INFO - 

===> Classification Report:

03/07/2020 01:21:11 AM - INFO -               precision    recall  f1-score   support

           0       0.82      0.88      0.85     12500
           1       0.87      0.80      0.83     12500

    accuracy                           0.84     25000
   macro avg       0.84      0.84      0.84     25000
weighted avg       0.84      0.84      0.84     25000

03/07/2020 01:21:11 AM - INFO - 

Cross validation:
03/07/2020 01:21:12 AM - INFO - 	accuracy: 5-fold cross validation: [0.8564 0.8678 0.8682 0.8678 0.8672]
03/07/2020 01:21:12 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 86.55 (+/- 0.91)
03/07/2020 01:21:12 AM - INFO - It took 10.124709129333496 seconds
03/07/2020 01:21:12 AM - INFO - ********************************************************************************
03/07/2020 01:21:12 AM - INFO - ################################################################################
03/07/2020 01:21:12 AM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 01:21:12 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:21:12 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:21:12 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  80.24%  |  [0.801  0.803  0.799  0.7968 0.7986]  |  79.97 (+/- 0.43)  |  10.44  |  0.5797  |
03/07/2020 01:21:12 AM - INFO - |  2  |  BERNOULLI_NB  |  81.52%  |  [0.8384 0.8398 0.8524 0.8404 0.8524]  |  84.47 (+/- 1.27)  |  0.02364  |  0.02019  |
03/07/2020 01:21:12 AM - INFO - |  3  |  COMPLEMENT_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01559  |  0.008243  |
03/07/2020 01:21:12 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  71.44%  |  [0.7116 0.7088 0.7172 0.7138 0.7112]  |  71.25 (+/- 0.57)  |  23.48  |  0.03173  |
03/07/2020 01:21:12 AM - INFO - |  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  88.44 (+/- 0.98)  |  0.2275  |  0.003976  |
03/07/2020 01:21:12 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  88.39%  |  [0.8822 0.8946 0.8848 0.887  0.8852]  |  88.68 (+/- 0.84)  |  1.157  |  0.008317  |
03/07/2020 01:21:12 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  84.94%  |  [0.8498 0.8534 0.8468 0.8452 0.8464]  |  84.83 (+/- 0.59)  |  35.85  |  1.299  |
03/07/2020 01:21:12 AM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 01:21:12 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:21:12 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:21:12 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  84.60%  |  [0.8398 0.8516 0.8416 0.8366 0.8416]  |  84.22 (+/- 1.00)  |  103.8  |  5.642  |
03/07/2020 01:21:12 AM - INFO - |  2  |  BERNOULLI_NB  |  81.28%  |  [0.8398 0.8424 0.8514 0.8396 0.8516]  |  84.50 (+/- 1.09)  |  0.02541  |  0.02126  |
03/07/2020 01:21:12 AM - INFO - |  3  |  COMPLEMENT_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01709  |  0.009721  |
03/07/2020 01:21:12 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  73.55%  |  [0.7342 0.728  0.7374 0.7336 0.7266]  |  73.20 (+/- 0.81)  |  13.62  |  0.01241  |
03/07/2020 01:21:12 AM - INFO - |  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  88.44 (+/- 0.98)  |  0.2293  |  0.003983  |
03/07/2020 01:21:12 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  87.75%  |  [0.8882 0.897  0.8878 0.8876 0.8818]  |  88.85 (+/- 0.97)  |  1.763  |  0.008308  |
03/07/2020 01:21:12 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  85.39%  |  [0.8488 0.8534 0.8516 0.8448 0.8564]  |  85.10 (+/- 0.79)  |  33.64  |  2.238  |
03/07/2020 01:21:12 AM - INFO - 

03/07/2020 01:21:12 AM - INFO - ################################################################################
03/07/2020 01:21:12 AM - INFO - 8)
03/07/2020 01:21:12 AM - INFO - ********************************************************************************
03/07/2020 01:21:12 AM - INFO - Classifier: MULTINOMIAL_NB, Dataset: IMDB_REVIEWS
03/07/2020 01:21:12 AM - INFO - ********************************************************************************
03/07/2020 01:21:18 AM - INFO - 

Performing grid search...

03/07/2020 01:21:18 AM - INFO - Parameters:
03/07/2020 01:21:18 AM - INFO - {'classifier__alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0], 'classifier__fit_prior': [False, True]}
03/07/2020 01:21:20 AM - INFO - 	Done in 1.935s
03/07/2020 01:21:20 AM - INFO - 	Best score: 0.865
03/07/2020 01:21:20 AM - INFO - 	Best parameters set:
03/07/2020 01:21:20 AM - INFO - 		classifier__alpha: 1.0
03/07/2020 01:21:20 AM - INFO - 		classifier__fit_prior: False
03/07/2020 01:21:20 AM - INFO - 

USING MULTINOMIAL_NB WITH DEFAULT PARAMETERS
03/07/2020 01:21:20 AM - INFO - ________________________________________________________________________________
03/07/2020 01:21:20 AM - INFO - Training: 
03/07/2020 01:21:20 AM - INFO - MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
03/07/2020 01:21:20 AM - INFO - Train time: 0.016s
03/07/2020 01:21:20 AM - INFO - Test time:  0.008s
03/07/2020 01:21:20 AM - INFO - Accuracy score:   0.839
03/07/2020 01:21:20 AM - INFO - 

===> Classification Report:

03/07/2020 01:21:20 AM - INFO -               precision    recall  f1-score   support

           0       0.82      0.88      0.85     12500
           1       0.87      0.80      0.83     12500

    accuracy                           0.84     25000
   macro avg       0.84      0.84      0.84     25000
weighted avg       0.84      0.84      0.84     25000

03/07/2020 01:21:20 AM - INFO - 

Cross validation:
03/07/2020 01:21:20 AM - INFO - 	accuracy: 5-fold cross validation: [0.8564 0.8678 0.8682 0.8678 0.8672]
03/07/2020 01:21:20 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 86.55 (+/- 0.91)
03/07/2020 01:21:20 AM - INFO - 

USING MULTINOMIAL_NB WITH BEST PARAMETERS: {'alpha': 1.0, 'fit_prior': False}
03/07/2020 01:21:20 AM - INFO - ________________________________________________________________________________
03/07/2020 01:21:20 AM - INFO - Training: 
03/07/2020 01:21:20 AM - INFO - MultinomialNB(alpha=1.0, class_prior=None, fit_prior=False)
03/07/2020 01:21:20 AM - INFO - Train time: 0.016s
03/07/2020 01:21:20 AM - INFO - Test time:  0.008s
03/07/2020 01:21:20 AM - INFO - Accuracy score:   0.839
03/07/2020 01:21:20 AM - INFO - 

===> Classification Report:

03/07/2020 01:21:20 AM - INFO -               precision    recall  f1-score   support

           0       0.82      0.88      0.85     12500
           1       0.87      0.80      0.83     12500

    accuracy                           0.84     25000
   macro avg       0.84      0.84      0.84     25000
weighted avg       0.84      0.84      0.84     25000

03/07/2020 01:21:20 AM - INFO - 

Cross validation:
03/07/2020 01:21:20 AM - INFO - 	accuracy: 5-fold cross validation: [0.8564 0.8678 0.8682 0.8678 0.8672]
03/07/2020 01:21:20 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 86.55 (+/- 0.91)
03/07/2020 01:21:20 AM - INFO - It took 8.813841581344604 seconds
03/07/2020 01:21:20 AM - INFO - ********************************************************************************
03/07/2020 01:21:20 AM - INFO - ################################################################################
03/07/2020 01:21:20 AM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 01:21:20 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:21:20 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:21:20 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  80.24%  |  [0.801  0.803  0.799  0.7968 0.7986]  |  79.97 (+/- 0.43)  |  10.44  |  0.5797  |
03/07/2020 01:21:20 AM - INFO - |  2  |  BERNOULLI_NB  |  81.52%  |  [0.8384 0.8398 0.8524 0.8404 0.8524]  |  84.47 (+/- 1.27)  |  0.02364  |  0.02019  |
03/07/2020 01:21:20 AM - INFO - |  3  |  COMPLEMENT_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01559  |  0.008243  |
03/07/2020 01:21:20 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  71.44%  |  [0.7116 0.7088 0.7172 0.7138 0.7112]  |  71.25 (+/- 0.57)  |  23.48  |  0.03173  |
03/07/2020 01:21:20 AM - INFO - |  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  88.44 (+/- 0.98)  |  0.2275  |  0.003976  |
03/07/2020 01:21:20 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  88.39%  |  [0.8822 0.8946 0.8848 0.887  0.8852]  |  88.68 (+/- 0.84)  |  1.157  |  0.008317  |
03/07/2020 01:21:20 AM - INFO - |  9  |  MULTINOMIAL_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01562  |  0.008306  |
03/07/2020 01:21:20 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  84.94%  |  [0.8498 0.8534 0.8468 0.8452 0.8464]  |  84.83 (+/- 0.59)  |  35.85  |  1.299  |
03/07/2020 01:21:20 AM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 01:21:20 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:21:20 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:21:20 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  84.60%  |  [0.8398 0.8516 0.8416 0.8366 0.8416]  |  84.22 (+/- 1.00)  |  103.8  |  5.642  |
03/07/2020 01:21:20 AM - INFO - |  2  |  BERNOULLI_NB  |  81.28%  |  [0.8398 0.8424 0.8514 0.8396 0.8516]  |  84.50 (+/- 1.09)  |  0.02541  |  0.02126  |
03/07/2020 01:21:20 AM - INFO - |  3  |  COMPLEMENT_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01709  |  0.009721  |
03/07/2020 01:21:20 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  73.55%  |  [0.7342 0.728  0.7374 0.7336 0.7266]  |  73.20 (+/- 0.81)  |  13.62  |  0.01241  |
03/07/2020 01:21:20 AM - INFO - |  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  88.44 (+/- 0.98)  |  0.2293  |  0.003983  |
03/07/2020 01:21:20 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  87.75%  |  [0.8882 0.897  0.8878 0.8876 0.8818]  |  88.85 (+/- 0.97)  |  1.763  |  0.008308  |
03/07/2020 01:21:20 AM - INFO - |  9  |  MULTINOMIAL_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01575  |  0.008321  |
03/07/2020 01:21:20 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  85.39%  |  [0.8488 0.8534 0.8516 0.8448 0.8564]  |  85.10 (+/- 0.79)  |  33.64  |  2.238  |
03/07/2020 01:21:20 AM - INFO - 

03/07/2020 01:21:20 AM - INFO - ################################################################################
03/07/2020 01:21:20 AM - INFO - 9)
03/07/2020 01:21:20 AM - INFO - ********************************************************************************
03/07/2020 01:21:20 AM - INFO - Classifier: NEAREST_CENTROID, Dataset: IMDB_REVIEWS
03/07/2020 01:21:20 AM - INFO - ********************************************************************************
03/07/2020 01:21:27 AM - INFO - 

Performing grid search...

03/07/2020 01:21:27 AM - INFO - Parameters:
03/07/2020 01:21:27 AM - INFO - {'classifier__metric': ['euclidean', 'cosine']}
03/07/2020 01:21:27 AM - INFO - 	Done in 0.311s
03/07/2020 01:21:27 AM - INFO - 	Best score: 0.848
03/07/2020 01:21:27 AM - INFO - 	Best parameters set:
03/07/2020 01:21:27 AM - INFO - 		classifier__metric: 'cosine'
03/07/2020 01:21:27 AM - INFO - 

USING NEAREST_CENTROID WITH DEFAULT PARAMETERS
03/07/2020 01:21:27 AM - INFO - ________________________________________________________________________________
03/07/2020 01:21:27 AM - INFO - Training: 
03/07/2020 01:21:27 AM - INFO - NearestCentroid(metric='euclidean', shrink_threshold=None)
03/07/2020 01:21:27 AM - INFO - Train time: 0.016s
03/07/2020 01:21:27 AM - INFO - Test time:  0.012s
03/07/2020 01:21:27 AM - INFO - Accuracy score:   0.837
03/07/2020 01:21:27 AM - INFO - 

===> Classification Report:

03/07/2020 01:21:27 AM - INFO -               precision    recall  f1-score   support

           0       0.86      0.81      0.83     12500
           1       0.82      0.87      0.84     12500

    accuracy                           0.84     25000
   macro avg       0.84      0.84      0.84     25000
weighted avg       0.84      0.84      0.84     25000

03/07/2020 01:21:27 AM - INFO - 

Cross validation:
03/07/2020 01:21:27 AM - INFO - 	accuracy: 5-fold cross validation: [0.8316 0.838  0.8342 0.8392 0.8358]
03/07/2020 01:21:27 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 83.58 (+/- 0.54)
03/07/2020 01:21:27 AM - INFO - 

USING NEAREST_CENTROID WITH BEST PARAMETERS: {'metric': 'cosine'}
03/07/2020 01:21:27 AM - INFO - ________________________________________________________________________________
03/07/2020 01:21:27 AM - INFO - Training: 
03/07/2020 01:21:27 AM - INFO - NearestCentroid(metric='cosine', shrink_threshold=None)
03/07/2020 01:21:27 AM - INFO - Train time: 0.016s
03/07/2020 01:21:27 AM - INFO - Test time:  0.017s
03/07/2020 01:21:27 AM - INFO - Accuracy score:   0.847
03/07/2020 01:21:27 AM - INFO - 

===> Classification Report:

03/07/2020 01:21:27 AM - INFO -               precision    recall  f1-score   support

           0       0.86      0.83      0.84     12500
           1       0.83      0.87      0.85     12500

    accuracy                           0.85     25000
   macro avg       0.85      0.85      0.85     25000
weighted avg       0.85      0.85      0.85     25000

03/07/2020 01:21:27 AM - INFO - 

Cross validation:
03/07/2020 01:21:28 AM - INFO - 	accuracy: 5-fold cross validation: [0.8426 0.8546 0.8426 0.8522 0.8502]
03/07/2020 01:21:28 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 84.84 (+/- 0.99)
03/07/2020 01:21:28 AM - INFO - It took 7.222600698471069 seconds
03/07/2020 01:21:28 AM - INFO - ********************************************************************************
03/07/2020 01:21:28 AM - INFO - ################################################################################
03/07/2020 01:21:28 AM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 01:21:28 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:21:28 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:21:28 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  80.24%  |  [0.801  0.803  0.799  0.7968 0.7986]  |  79.97 (+/- 0.43)  |  10.44  |  0.5797  |
03/07/2020 01:21:28 AM - INFO - |  2  |  BERNOULLI_NB  |  81.52%  |  [0.8384 0.8398 0.8524 0.8404 0.8524]  |  84.47 (+/- 1.27)  |  0.02364  |  0.02019  |
03/07/2020 01:21:28 AM - INFO - |  3  |  COMPLEMENT_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01559  |  0.008243  |
03/07/2020 01:21:28 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  71.44%  |  [0.7116 0.7088 0.7172 0.7138 0.7112]  |  71.25 (+/- 0.57)  |  23.48  |  0.03173  |
03/07/2020 01:21:28 AM - INFO - |  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  88.44 (+/- 0.98)  |  0.2275  |  0.003976  |
03/07/2020 01:21:28 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  88.39%  |  [0.8822 0.8946 0.8848 0.887  0.8852]  |  88.68 (+/- 0.84)  |  1.157  |  0.008317  |
03/07/2020 01:21:28 AM - INFO - |  9  |  MULTINOMIAL_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01562  |  0.008306  |
03/07/2020 01:21:28 AM - INFO - |  10  |  NEAREST_CENTROID  |  83.72%  |  [0.8316 0.838  0.8342 0.8392 0.8358]  |  83.58 (+/- 0.54)  |  0.01647  |  0.01182  |
03/07/2020 01:21:28 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  84.94%  |  [0.8498 0.8534 0.8468 0.8452 0.8464]  |  84.83 (+/- 0.59)  |  35.85  |  1.299  |
03/07/2020 01:21:28 AM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 01:21:28 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:21:28 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:21:28 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  84.60%  |  [0.8398 0.8516 0.8416 0.8366 0.8416]  |  84.22 (+/- 1.00)  |  103.8  |  5.642  |
03/07/2020 01:21:28 AM - INFO - |  2  |  BERNOULLI_NB  |  81.28%  |  [0.8398 0.8424 0.8514 0.8396 0.8516]  |  84.50 (+/- 1.09)  |  0.02541  |  0.02126  |
03/07/2020 01:21:28 AM - INFO - |  3  |  COMPLEMENT_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01709  |  0.009721  |
03/07/2020 01:21:28 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  73.55%  |  [0.7342 0.728  0.7374 0.7336 0.7266]  |  73.20 (+/- 0.81)  |  13.62  |  0.01241  |
03/07/2020 01:21:28 AM - INFO - |  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  88.44 (+/- 0.98)  |  0.2293  |  0.003983  |
03/07/2020 01:21:28 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  87.75%  |  [0.8882 0.897  0.8878 0.8876 0.8818]  |  88.85 (+/- 0.97)  |  1.763  |  0.008308  |
03/07/2020 01:21:28 AM - INFO - |  9  |  MULTINOMIAL_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01575  |  0.008321  |
03/07/2020 01:21:28 AM - INFO - |  10  |  NEAREST_CENTROID  |  84.65%  |  [0.8426 0.8546 0.8426 0.8522 0.8502]  |  84.84 (+/- 0.99)  |  0.01649  |  0.01698  |
03/07/2020 01:21:28 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  85.39%  |  [0.8488 0.8534 0.8516 0.8448 0.8564]  |  85.10 (+/- 0.79)  |  33.64  |  2.238  |
03/07/2020 01:21:28 AM - INFO - 

03/07/2020 01:21:28 AM - INFO - ################################################################################
03/07/2020 01:21:28 AM - INFO - 10)
03/07/2020 01:21:28 AM - INFO - ********************************************************************************
03/07/2020 01:21:28 AM - INFO - Classifier: PASSIVE_AGGRESSIVE_CLASSIFIER, Dataset: IMDB_REVIEWS
03/07/2020 01:21:28 AM - INFO - ********************************************************************************
03/07/2020 01:21:34 AM - INFO - 

Performing grid search...

03/07/2020 01:21:34 AM - INFO - Parameters:
03/07/2020 01:21:34 AM - INFO - {'classifier__C': [0.01, 1.0], 'classifier__early_stopping': [False, True], 'classifier__tol': [0.0001, 0.001, 0.01], 'classifier__validation_fraction': [0.0001, 0.01]}
03/07/2020 01:21:46 AM - INFO - 	Done in 11.632s
03/07/2020 01:21:46 AM - INFO - 	Best score: 0.889
03/07/2020 01:21:46 AM - INFO - 	Best parameters set:
03/07/2020 01:21:46 AM - INFO - 		classifier__C: 0.01
03/07/2020 01:21:46 AM - INFO - 		classifier__early_stopping: False
03/07/2020 01:21:46 AM - INFO - 		classifier__tol: 0.001
03/07/2020 01:21:46 AM - INFO - 		classifier__validation_fraction: 0.01
03/07/2020 01:21:46 AM - INFO - 

USING PASSIVE_AGGRESSIVE_CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 01:21:46 AM - INFO - ________________________________________________________________________________
03/07/2020 01:21:46 AM - INFO - Training: 
03/07/2020 01:21:46 AM - INFO - PassiveAggressiveClassifier(C=1.0, average=False, class_weight=None,
                            early_stopping=False, fit_intercept=True,
                            loss='hinge', max_iter=1000, n_iter_no_change=5,
                            n_jobs=None, random_state=None, shuffle=True,
                            tol=0.001, validation_fraction=0.1, verbose=0,
                            warm_start=False)
03/07/2020 01:21:46 AM - INFO - Train time: 0.193s
03/07/2020 01:21:46 AM - INFO - Test time:  0.004s
03/07/2020 01:21:46 AM - INFO - Accuracy score:   0.851
03/07/2020 01:21:46 AM - INFO - 

===> Classification Report:

03/07/2020 01:21:46 AM - INFO -               precision    recall  f1-score   support

           0       0.84      0.86      0.85     12500
           1       0.86      0.84      0.85     12500

    accuracy                           0.85     25000
   macro avg       0.85      0.85      0.85     25000
weighted avg       0.85      0.85      0.85     25000

03/07/2020 01:21:46 AM - INFO - 

Cross validation:
03/07/2020 01:21:46 AM - INFO - 	accuracy: 5-fold cross validation: [0.8666 0.8772 0.8748 0.8698 0.8644]
03/07/2020 01:21:46 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 87.06 (+/- 0.96)
03/07/2020 01:21:46 AM - INFO - 

USING PASSIVE_AGGRESSIVE_CLASSIFIER WITH BEST PARAMETERS: {'C': 0.01, 'early_stopping': False, 'tol': 0.001, 'validation_fraction': 0.01}
03/07/2020 01:21:46 AM - INFO - ________________________________________________________________________________
03/07/2020 01:21:46 AM - INFO - Training: 
03/07/2020 01:21:46 AM - INFO - PassiveAggressiveClassifier(C=0.01, average=False, class_weight=None,
                            early_stopping=False, fit_intercept=True,
                            loss='hinge', max_iter=1000, n_iter_no_change=5,
                            n_jobs=None, random_state=None, shuffle=True,
                            tol=0.001, validation_fraction=0.01, verbose=0,
                            warm_start=False)
03/07/2020 01:21:47 AM - INFO - Train time: 0.840s
03/07/2020 01:21:47 AM - INFO - Test time:  0.004s
03/07/2020 01:21:47 AM - INFO - Accuracy score:   0.881
03/07/2020 01:21:47 AM - INFO - 

===> Classification Report:

03/07/2020 01:21:47 AM - INFO -               precision    recall  f1-score   support

           0       0.88      0.88      0.88     12500
           1       0.88      0.88      0.88     12500

    accuracy                           0.88     25000
   macro avg       0.88      0.88      0.88     25000
weighted avg       0.88      0.88      0.88     25000

03/07/2020 01:21:47 AM - INFO - 

Cross validation:
03/07/2020 01:21:48 AM - INFO - 	accuracy: 5-fold cross validation: [0.8874 0.897  0.8884 0.8874 0.8852]
03/07/2020 01:21:48 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 88.91 (+/- 0.82)
03/07/2020 01:21:48 AM - INFO - It took 20.571793794631958 seconds
03/07/2020 01:21:48 AM - INFO - ********************************************************************************
03/07/2020 01:21:48 AM - INFO - ################################################################################
03/07/2020 01:21:48 AM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 01:21:48 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:21:48 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:21:48 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  80.24%  |  [0.801  0.803  0.799  0.7968 0.7986]  |  79.97 (+/- 0.43)  |  10.44  |  0.5797  |
03/07/2020 01:21:48 AM - INFO - |  2  |  BERNOULLI_NB  |  81.52%  |  [0.8384 0.8398 0.8524 0.8404 0.8524]  |  84.47 (+/- 1.27)  |  0.02364  |  0.02019  |
03/07/2020 01:21:48 AM - INFO - |  3  |  COMPLEMENT_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01559  |  0.008243  |
03/07/2020 01:21:48 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  71.44%  |  [0.7116 0.7088 0.7172 0.7138 0.7112]  |  71.25 (+/- 0.57)  |  23.48  |  0.03173  |
03/07/2020 01:21:48 AM - INFO - |  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  88.44 (+/- 0.98)  |  0.2275  |  0.003976  |
03/07/2020 01:21:48 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  88.39%  |  [0.8822 0.8946 0.8848 0.887  0.8852]  |  88.68 (+/- 0.84)  |  1.157  |  0.008317  |
03/07/2020 01:21:48 AM - INFO - |  9  |  MULTINOMIAL_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01562  |  0.008306  |
03/07/2020 01:21:48 AM - INFO - |  10  |  NEAREST_CENTROID  |  83.72%  |  [0.8316 0.838  0.8342 0.8392 0.8358]  |  83.58 (+/- 0.54)  |  0.01647  |  0.01182  |
03/07/2020 01:21:48 AM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  85.13%  |  [0.8666 0.8772 0.8748 0.8698 0.8644]  |  87.06 (+/- 0.96)  |  0.1929  |  0.003881  |
03/07/2020 01:21:48 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  84.94%  |  [0.8498 0.8534 0.8468 0.8452 0.8464]  |  84.83 (+/- 0.59)  |  35.85  |  1.299  |
03/07/2020 01:21:48 AM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 01:21:48 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:21:48 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:21:48 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  84.60%  |  [0.8398 0.8516 0.8416 0.8366 0.8416]  |  84.22 (+/- 1.00)  |  103.8  |  5.642  |
03/07/2020 01:21:48 AM - INFO - |  2  |  BERNOULLI_NB  |  81.28%  |  [0.8398 0.8424 0.8514 0.8396 0.8516]  |  84.50 (+/- 1.09)  |  0.02541  |  0.02126  |
03/07/2020 01:21:48 AM - INFO - |  3  |  COMPLEMENT_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01709  |  0.009721  |
03/07/2020 01:21:48 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  73.55%  |  [0.7342 0.728  0.7374 0.7336 0.7266]  |  73.20 (+/- 0.81)  |  13.62  |  0.01241  |
03/07/2020 01:21:48 AM - INFO - |  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  88.44 (+/- 0.98)  |  0.2293  |  0.003983  |
03/07/2020 01:21:48 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  87.75%  |  [0.8882 0.897  0.8878 0.8876 0.8818]  |  88.85 (+/- 0.97)  |  1.763  |  0.008308  |
03/07/2020 01:21:48 AM - INFO - |  9  |  MULTINOMIAL_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01575  |  0.008321  |
03/07/2020 01:21:48 AM - INFO - |  10  |  NEAREST_CENTROID  |  84.65%  |  [0.8426 0.8546 0.8426 0.8522 0.8502]  |  84.84 (+/- 0.99)  |  0.01649  |  0.01698  |
03/07/2020 01:21:48 AM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  88.08%  |  [0.8874 0.897  0.8884 0.8874 0.8852]  |  88.91 (+/- 0.82)  |  0.8401  |  0.003811  |
03/07/2020 01:21:48 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  85.39%  |  [0.8488 0.8534 0.8516 0.8448 0.8564]  |  85.10 (+/- 0.79)  |  33.64  |  2.238  |
03/07/2020 01:21:48 AM - INFO - 

03/07/2020 01:21:48 AM - INFO - ################################################################################
03/07/2020 01:21:48 AM - INFO - 11)
03/07/2020 01:21:48 AM - INFO - ********************************************************************************
03/07/2020 01:21:48 AM - INFO - Classifier: K_NEIGHBORS_CLASSIFIER, Dataset: IMDB_REVIEWS
03/07/2020 01:21:48 AM - INFO - ********************************************************************************
03/07/2020 01:21:54 AM - INFO - 

Performing grid search...

03/07/2020 01:21:54 AM - INFO - Parameters:
03/07/2020 01:21:54 AM - INFO - {'classifier__leaf_size': [5, 30], 'classifier__metric': ['euclidean', 'minkowski'], 'classifier__n_neighbors': [3, 50], 'classifier__weights': ['uniform', 'distance']}
03/07/2020 01:23:08 AM - INFO - 	Done in 73.729s
03/07/2020 01:23:08 AM - INFO - 	Best score: 0.867
03/07/2020 01:23:08 AM - INFO - 	Best parameters set:
03/07/2020 01:23:08 AM - INFO - 		classifier__leaf_size: 5
03/07/2020 01:23:08 AM - INFO - 		classifier__metric: 'euclidean'
03/07/2020 01:23:08 AM - INFO - 		classifier__n_neighbors: 50
03/07/2020 01:23:08 AM - INFO - 		classifier__weights: 'distance'
03/07/2020 01:23:08 AM - INFO - 

USING K_NEIGHBORS_CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 01:23:08 AM - INFO - ________________________________________________________________________________
03/07/2020 01:23:08 AM - INFO - Training: 
03/07/2020 01:23:08 AM - INFO - KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                     weights='uniform')
03/07/2020 01:23:08 AM - INFO - Train time: 0.009s
03/07/2020 01:23:35 AM - INFO - Test time:  26.891s
03/07/2020 01:23:35 AM - INFO - Accuracy score:   0.733
03/07/2020 01:23:35 AM - INFO - 

===> Classification Report:

03/07/2020 01:23:35 AM - INFO -               precision    recall  f1-score   support

           0       0.73      0.75      0.74     12500
           1       0.74      0.72      0.73     12500

    accuracy                           0.73     25000
   macro avg       0.73      0.73      0.73     25000
weighted avg       0.73      0.73      0.73     25000

03/07/2020 01:23:35 AM - INFO - 

Cross validation:
03/07/2020 01:23:41 AM - INFO - 	accuracy: 5-fold cross validation: [0.8144 0.8248 0.8262 0.814  0.815 ]
03/07/2020 01:23:41 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 81.89 (+/- 1.09)
03/07/2020 01:23:41 AM - INFO - 

USING K_NEIGHBORS_CLASSIFIER WITH BEST PARAMETERS: {'leaf_size': 5, 'metric': 'euclidean', 'n_neighbors': 50, 'weights': 'distance'}
03/07/2020 01:23:41 AM - INFO - ________________________________________________________________________________
03/07/2020 01:23:41 AM - INFO - Training: 
03/07/2020 01:23:41 AM - INFO - KNeighborsClassifier(algorithm='auto', leaf_size=5, metric='euclidean',
                     metric_params=None, n_jobs=None, n_neighbors=50, p=2,
                     weights='distance')
03/07/2020 01:23:41 AM - INFO - Train time: 0.007s
03/07/2020 01:24:07 AM - INFO - Test time:  26.407s
03/07/2020 01:24:07 AM - INFO - Accuracy score:   0.827
03/07/2020 01:24:07 AM - INFO - 

===> Classification Report:

03/07/2020 01:24:07 AM - INFO -               precision    recall  f1-score   support

           0       0.80      0.86      0.83     12500
           1       0.85      0.79      0.82     12500

    accuracy                           0.83     25000
   macro avg       0.83      0.83      0.83     25000
weighted avg       0.83      0.83      0.83     25000

03/07/2020 01:24:07 AM - INFO - 

Cross validation:
03/07/2020 01:24:13 AM - INFO - 	accuracy: 5-fold cross validation: [0.8632 0.8744 0.8694 0.864  0.8618]
03/07/2020 01:24:13 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 86.66 (+/- 0.94)
03/07/2020 01:24:13 AM - INFO - It took 144.80358839035034 seconds
03/07/2020 01:24:13 AM - INFO - ********************************************************************************
03/07/2020 01:24:13 AM - INFO - ################################################################################
03/07/2020 01:24:13 AM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 01:24:13 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:24:13 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:24:13 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  80.24%  |  [0.801  0.803  0.799  0.7968 0.7986]  |  79.97 (+/- 0.43)  |  10.44  |  0.5797  |
03/07/2020 01:24:13 AM - INFO - |  2  |  BERNOULLI_NB  |  81.52%  |  [0.8384 0.8398 0.8524 0.8404 0.8524]  |  84.47 (+/- 1.27)  |  0.02364  |  0.02019  |
03/07/2020 01:24:13 AM - INFO - |  3  |  COMPLEMENT_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01559  |  0.008243  |
03/07/2020 01:24:13 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  71.44%  |  [0.7116 0.7088 0.7172 0.7138 0.7112]  |  71.25 (+/- 0.57)  |  23.48  |  0.03173  |
03/07/2020 01:24:13 AM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  73.29%  |  [0.8144 0.8248 0.8262 0.814  0.815 ]  |  81.89 (+/- 1.09)  |  0.009318  |  26.89  |
03/07/2020 01:24:13 AM - INFO - |  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  88.44 (+/- 0.98)  |  0.2275  |  0.003976  |
03/07/2020 01:24:13 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  88.39%  |  [0.8822 0.8946 0.8848 0.887  0.8852]  |  88.68 (+/- 0.84)  |  1.157  |  0.008317  |
03/07/2020 01:24:13 AM - INFO - |  9  |  MULTINOMIAL_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01562  |  0.008306  |
03/07/2020 01:24:13 AM - INFO - |  10  |  NEAREST_CENTROID  |  83.72%  |  [0.8316 0.838  0.8342 0.8392 0.8358]  |  83.58 (+/- 0.54)  |  0.01647  |  0.01182  |
03/07/2020 01:24:13 AM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  85.13%  |  [0.8666 0.8772 0.8748 0.8698 0.8644]  |  87.06 (+/- 0.96)  |  0.1929  |  0.003881  |
03/07/2020 01:24:13 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  84.94%  |  [0.8498 0.8534 0.8468 0.8452 0.8464]  |  84.83 (+/- 0.59)  |  35.85  |  1.299  |
03/07/2020 01:24:13 AM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 01:24:13 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:24:13 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:24:13 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  84.60%  |  [0.8398 0.8516 0.8416 0.8366 0.8416]  |  84.22 (+/- 1.00)  |  103.8  |  5.642  |
03/07/2020 01:24:13 AM - INFO - |  2  |  BERNOULLI_NB  |  81.28%  |  [0.8398 0.8424 0.8514 0.8396 0.8516]  |  84.50 (+/- 1.09)  |  0.02541  |  0.02126  |
03/07/2020 01:24:13 AM - INFO - |  3  |  COMPLEMENT_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01709  |  0.009721  |
03/07/2020 01:24:13 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  73.55%  |  [0.7342 0.728  0.7374 0.7336 0.7266]  |  73.20 (+/- 0.81)  |  13.62  |  0.01241  |
03/07/2020 01:24:13 AM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  82.66%  |  [0.8632 0.8744 0.8694 0.864  0.8618]  |  86.66 (+/- 0.94)  |  0.007226  |  26.41  |
03/07/2020 01:24:13 AM - INFO - |  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  88.44 (+/- 0.98)  |  0.2293  |  0.003983  |
03/07/2020 01:24:13 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  87.75%  |  [0.8882 0.897  0.8878 0.8876 0.8818]  |  88.85 (+/- 0.97)  |  1.763  |  0.008308  |
03/07/2020 01:24:13 AM - INFO - |  9  |  MULTINOMIAL_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01575  |  0.008321  |
03/07/2020 01:24:13 AM - INFO - |  10  |  NEAREST_CENTROID  |  84.65%  |  [0.8426 0.8546 0.8426 0.8522 0.8502]  |  84.84 (+/- 0.99)  |  0.01649  |  0.01698  |
03/07/2020 01:24:13 AM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  88.08%  |  [0.8874 0.897  0.8884 0.8874 0.8852]  |  88.91 (+/- 0.82)  |  0.8401  |  0.003811  |
03/07/2020 01:24:13 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  85.39%  |  [0.8488 0.8534 0.8516 0.8448 0.8564]  |  85.10 (+/- 0.79)  |  33.64  |  2.238  |
03/07/2020 01:24:13 AM - INFO - 

03/07/2020 01:24:13 AM - INFO - ################################################################################
03/07/2020 01:24:13 AM - INFO - 12)
03/07/2020 01:24:13 AM - INFO - ********************************************************************************
03/07/2020 01:24:13 AM - INFO - Classifier: PERCEPTRON, Dataset: IMDB_REVIEWS
03/07/2020 01:24:13 AM - INFO - ********************************************************************************
03/07/2020 01:24:19 AM - INFO - 

Performing grid search...

03/07/2020 01:24:19 AM - INFO - Parameters:
03/07/2020 01:24:19 AM - INFO - {'classifier__early_stopping': [True], 'classifier__max_iter': [100], 'classifier__n_iter_no_change': [3, 15], 'classifier__penalty': ['l2'], 'classifier__tol': [0.0001, 0.1], 'classifier__validation_fraction': [0.0001, 0.01]}
03/07/2020 01:24:21 AM - INFO - 	Done in 1.738s
03/07/2020 01:24:21 AM - INFO - 	Best score: 0.816
03/07/2020 01:24:21 AM - INFO - 	Best parameters set:
03/07/2020 01:24:21 AM - INFO - 		classifier__early_stopping: True
03/07/2020 01:24:21 AM - INFO - 		classifier__max_iter: 100
03/07/2020 01:24:21 AM - INFO - 		classifier__n_iter_no_change: 3
03/07/2020 01:24:21 AM - INFO - 		classifier__penalty: 'l2'
03/07/2020 01:24:21 AM - INFO - 		classifier__tol: 0.0001
03/07/2020 01:24:21 AM - INFO - 		classifier__validation_fraction: 0.01
03/07/2020 01:24:21 AM - INFO - 

USING PERCEPTRON WITH DEFAULT PARAMETERS
03/07/2020 01:24:21 AM - INFO - ________________________________________________________________________________
03/07/2020 01:24:21 AM - INFO - Training: 
03/07/2020 01:24:21 AM - INFO - Perceptron(alpha=0.0001, class_weight=None, early_stopping=False, eta0=1.0,
           fit_intercept=True, max_iter=1000, n_iter_no_change=5, n_jobs=None,
           penalty=None, random_state=0, shuffle=True, tol=0.001,
           validation_fraction=0.1, verbose=0, warm_start=False)
03/07/2020 01:24:21 AM - INFO - Train time: 0.109s
03/07/2020 01:24:21 AM - INFO - Test time:  0.004s
03/07/2020 01:24:21 AM - INFO - Accuracy score:   0.844
03/07/2020 01:24:21 AM - INFO - 

===> Classification Report:

03/07/2020 01:24:21 AM - INFO -               precision    recall  f1-score   support

           0       0.83      0.86      0.85     12500
           1       0.86      0.83      0.84     12500

    accuracy                           0.84     25000
   macro avg       0.84      0.84      0.84     25000
weighted avg       0.84      0.84      0.84     25000

03/07/2020 01:24:21 AM - INFO - 

Cross validation:
03/07/2020 01:24:22 AM - INFO - 	accuracy: 5-fold cross validation: [0.861  0.868  0.8654 0.8614 0.8536]
03/07/2020 01:24:22 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 86.19 (+/- 0.98)
03/07/2020 01:24:22 AM - INFO - 

USING PERCEPTRON WITH BEST PARAMETERS: {'early_stopping': True, 'max_iter': 100, 'n_iter_no_change': 3, 'penalty': 'l2', 'tol': 0.0001, 'validation_fraction': 0.01}
03/07/2020 01:24:22 AM - INFO - ________________________________________________________________________________
03/07/2020 01:24:22 AM - INFO - Training: 
03/07/2020 01:24:22 AM - INFO - Perceptron(alpha=0.0001, class_weight=None, early_stopping=True, eta0=1.0,
           fit_intercept=True, max_iter=100, n_iter_no_change=3, n_jobs=None,
           penalty='l2', random_state=0, shuffle=True, tol=0.0001,
           validation_fraction=0.01, verbose=0, warm_start=False)
03/07/2020 01:24:22 AM - INFO - Train time: 0.091s
03/07/2020 01:24:22 AM - INFO - Test time:  0.007s
03/07/2020 01:24:22 AM - INFO - Accuracy score:   0.806
03/07/2020 01:24:22 AM - INFO - 

===> Classification Report:

03/07/2020 01:24:22 AM - INFO -               precision    recall  f1-score   support

           0       0.81      0.80      0.81     12500
           1       0.80      0.81      0.81     12500

    accuracy                           0.81     25000
   macro avg       0.81      0.81      0.81     25000
weighted avg       0.81      0.81      0.81     25000

03/07/2020 01:24:22 AM - INFO - 

Cross validation:
03/07/2020 01:24:22 AM - INFO - 	accuracy: 5-fold cross validation: [0.8166 0.8264 0.8144 0.81   0.8102]
03/07/2020 01:24:22 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 81.55 (+/- 1.20)
03/07/2020 01:24:22 AM - INFO - It took 8.940368890762329 seconds
03/07/2020 01:24:22 AM - INFO - ********************************************************************************
03/07/2020 01:24:22 AM - INFO - ################################################################################
03/07/2020 01:24:22 AM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 01:24:22 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:24:22 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:24:22 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  80.24%  |  [0.801  0.803  0.799  0.7968 0.7986]  |  79.97 (+/- 0.43)  |  10.44  |  0.5797  |
03/07/2020 01:24:22 AM - INFO - |  2  |  BERNOULLI_NB  |  81.52%  |  [0.8384 0.8398 0.8524 0.8404 0.8524]  |  84.47 (+/- 1.27)  |  0.02364  |  0.02019  |
03/07/2020 01:24:22 AM - INFO - |  3  |  COMPLEMENT_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01559  |  0.008243  |
03/07/2020 01:24:22 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  71.44%  |  [0.7116 0.7088 0.7172 0.7138 0.7112]  |  71.25 (+/- 0.57)  |  23.48  |  0.03173  |
03/07/2020 01:24:22 AM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  73.29%  |  [0.8144 0.8248 0.8262 0.814  0.815 ]  |  81.89 (+/- 1.09)  |  0.009318  |  26.89  |
03/07/2020 01:24:22 AM - INFO - |  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  88.44 (+/- 0.98)  |  0.2275  |  0.003976  |
03/07/2020 01:24:22 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  88.39%  |  [0.8822 0.8946 0.8848 0.887  0.8852]  |  88.68 (+/- 0.84)  |  1.157  |  0.008317  |
03/07/2020 01:24:22 AM - INFO - |  9  |  MULTINOMIAL_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01562  |  0.008306  |
03/07/2020 01:24:22 AM - INFO - |  10  |  NEAREST_CENTROID  |  83.72%  |  [0.8316 0.838  0.8342 0.8392 0.8358]  |  83.58 (+/- 0.54)  |  0.01647  |  0.01182  |
03/07/2020 01:24:22 AM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  85.13%  |  [0.8666 0.8772 0.8748 0.8698 0.8644]  |  87.06 (+/- 0.96)  |  0.1929  |  0.003881  |
03/07/2020 01:24:22 AM - INFO - |  12  |  PERCEPTRON  |  84.43%  |  [0.861  0.868  0.8654 0.8614 0.8536]  |  86.19 (+/- 0.98)  |  0.1085  |  0.00389  |
03/07/2020 01:24:22 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  84.94%  |  [0.8498 0.8534 0.8468 0.8452 0.8464]  |  84.83 (+/- 0.59)  |  35.85  |  1.299  |
03/07/2020 01:24:22 AM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 01:24:22 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:24:22 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:24:22 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  84.60%  |  [0.8398 0.8516 0.8416 0.8366 0.8416]  |  84.22 (+/- 1.00)  |  103.8  |  5.642  |
03/07/2020 01:24:22 AM - INFO - |  2  |  BERNOULLI_NB  |  81.28%  |  [0.8398 0.8424 0.8514 0.8396 0.8516]  |  84.50 (+/- 1.09)  |  0.02541  |  0.02126  |
03/07/2020 01:24:22 AM - INFO - |  3  |  COMPLEMENT_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01709  |  0.009721  |
03/07/2020 01:24:22 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  73.55%  |  [0.7342 0.728  0.7374 0.7336 0.7266]  |  73.20 (+/- 0.81)  |  13.62  |  0.01241  |
03/07/2020 01:24:22 AM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  82.66%  |  [0.8632 0.8744 0.8694 0.864  0.8618]  |  86.66 (+/- 0.94)  |  0.007226  |  26.41  |
03/07/2020 01:24:22 AM - INFO - |  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  88.44 (+/- 0.98)  |  0.2293  |  0.003983  |
03/07/2020 01:24:22 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  87.75%  |  [0.8882 0.897  0.8878 0.8876 0.8818]  |  88.85 (+/- 0.97)  |  1.763  |  0.008308  |
03/07/2020 01:24:22 AM - INFO - |  9  |  MULTINOMIAL_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01575  |  0.008321  |
03/07/2020 01:24:22 AM - INFO - |  10  |  NEAREST_CENTROID  |  84.65%  |  [0.8426 0.8546 0.8426 0.8522 0.8502]  |  84.84 (+/- 0.99)  |  0.01649  |  0.01698  |
03/07/2020 01:24:22 AM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  88.08%  |  [0.8874 0.897  0.8884 0.8874 0.8852]  |  88.91 (+/- 0.82)  |  0.8401  |  0.003811  |
03/07/2020 01:24:22 AM - INFO - |  12  |  PERCEPTRON  |  80.64%  |  [0.8166 0.8264 0.8144 0.81   0.8102]  |  81.55 (+/- 1.20)  |  0.09111  |  0.00689  |
03/07/2020 01:24:22 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  85.39%  |  [0.8488 0.8534 0.8516 0.8448 0.8564]  |  85.10 (+/- 0.79)  |  33.64  |  2.238  |
03/07/2020 01:24:22 AM - INFO - 

03/07/2020 01:24:22 AM - INFO - ################################################################################
03/07/2020 01:24:22 AM - INFO - 13)
03/07/2020 01:24:22 AM - INFO - ********************************************************************************
03/07/2020 01:24:22 AM - INFO - Classifier: RIDGE_CLASSIFIER, Dataset: IMDB_REVIEWS
03/07/2020 01:24:22 AM - INFO - ********************************************************************************
03/07/2020 01:24:28 AM - INFO - 

Performing grid search...

03/07/2020 01:24:28 AM - INFO - Parameters:
03/07/2020 01:24:28 AM - INFO - {'classifier__alpha': [0.5, 1.0], 'classifier__tol': [0.0001, 0.001]}
03/07/2020 01:24:30 AM - INFO - 	Done in 2.037s
03/07/2020 01:24:30 AM - INFO - 	Best score: 0.886
03/07/2020 01:24:30 AM - INFO - 	Best parameters set:
03/07/2020 01:24:30 AM - INFO - 		classifier__alpha: 1.0
03/07/2020 01:24:30 AM - INFO - 		classifier__tol: 0.0001
03/07/2020 01:24:30 AM - INFO - 

USING RIDGE_CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 01:24:30 AM - INFO - ________________________________________________________________________________
03/07/2020 01:24:30 AM - INFO - Training: 
03/07/2020 01:24:30 AM - INFO - RidgeClassifier(alpha=1.0, class_weight=None, copy_X=True, fit_intercept=True,
                max_iter=None, normalize=False, random_state=None,
                solver='auto', tol=0.001)
03/07/2020 01:24:31 AM - INFO - Train time: 0.433s
03/07/2020 01:24:31 AM - INFO - Test time:  0.008s
03/07/2020 01:24:31 AM - INFO - Accuracy score:   0.869
03/07/2020 01:24:31 AM - INFO - 

===> Classification Report:

03/07/2020 01:24:31 AM - INFO -               precision    recall  f1-score   support

           0       0.87      0.87      0.87     12500
           1       0.87      0.87      0.87     12500

    accuracy                           0.87     25000
   macro avg       0.87      0.87      0.87     25000
weighted avg       0.87      0.87      0.87     25000

03/07/2020 01:24:31 AM - INFO - 

Cross validation:
03/07/2020 01:24:31 AM - INFO - 	accuracy: 5-fold cross validation: [0.8836 0.895  0.8888 0.882  0.8788]
03/07/2020 01:24:31 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 88.56 (+/- 1.14)
03/07/2020 01:24:31 AM - INFO - 

USING RIDGE_CLASSIFIER WITH BEST PARAMETERS: {'alpha': 1.0, 'tol': 0.0001}
03/07/2020 01:24:31 AM - INFO - ________________________________________________________________________________
03/07/2020 01:24:31 AM - INFO - Training: 
03/07/2020 01:24:31 AM - INFO - RidgeClassifier(alpha=1.0, class_weight=None, copy_X=True, fit_intercept=True,
                max_iter=None, normalize=False, random_state=None,
                solver='auto', tol=0.0001)
03/07/2020 01:24:32 AM - INFO - Train time: 0.468s
03/07/2020 01:24:32 AM - INFO - Test time:  0.008s
03/07/2020 01:24:32 AM - INFO - Accuracy score:   0.869
03/07/2020 01:24:32 AM - INFO - 

===> Classification Report:

03/07/2020 01:24:32 AM - INFO -               precision    recall  f1-score   support

           0       0.87      0.87      0.87     12500
           1       0.87      0.87      0.87     12500

    accuracy                           0.87     25000
   macro avg       0.87      0.87      0.87     25000
weighted avg       0.87      0.87      0.87     25000

03/07/2020 01:24:32 AM - INFO - 

Cross validation:
03/07/2020 01:24:32 AM - INFO - 	accuracy: 5-fold cross validation: [0.8838 0.8952 0.8892 0.882  0.8788]
03/07/2020 01:24:32 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 88.58 (+/- 1.16)
03/07/2020 01:24:32 AM - INFO - It took 10.482115507125854 seconds
03/07/2020 01:24:32 AM - INFO - ********************************************************************************
03/07/2020 01:24:32 AM - INFO - ################################################################################
03/07/2020 01:24:32 AM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 01:24:32 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:24:32 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:24:32 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  80.24%  |  [0.801  0.803  0.799  0.7968 0.7986]  |  79.97 (+/- 0.43)  |  10.44  |  0.5797  |
03/07/2020 01:24:32 AM - INFO - |  2  |  BERNOULLI_NB  |  81.52%  |  [0.8384 0.8398 0.8524 0.8404 0.8524]  |  84.47 (+/- 1.27)  |  0.02364  |  0.02019  |
03/07/2020 01:24:32 AM - INFO - |  3  |  COMPLEMENT_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01559  |  0.008243  |
03/07/2020 01:24:32 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  71.44%  |  [0.7116 0.7088 0.7172 0.7138 0.7112]  |  71.25 (+/- 0.57)  |  23.48  |  0.03173  |
03/07/2020 01:24:32 AM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  73.29%  |  [0.8144 0.8248 0.8262 0.814  0.815 ]  |  81.89 (+/- 1.09)  |  0.009318  |  26.89  |
03/07/2020 01:24:32 AM - INFO - |  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  88.44 (+/- 0.98)  |  0.2275  |  0.003976  |
03/07/2020 01:24:32 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  88.39%  |  [0.8822 0.8946 0.8848 0.887  0.8852]  |  88.68 (+/- 0.84)  |  1.157  |  0.008317  |
03/07/2020 01:24:32 AM - INFO - |  9  |  MULTINOMIAL_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01562  |  0.008306  |
03/07/2020 01:24:32 AM - INFO - |  10  |  NEAREST_CENTROID  |  83.72%  |  [0.8316 0.838  0.8342 0.8392 0.8358]  |  83.58 (+/- 0.54)  |  0.01647  |  0.01182  |
03/07/2020 01:24:32 AM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  85.13%  |  [0.8666 0.8772 0.8748 0.8698 0.8644]  |  87.06 (+/- 0.96)  |  0.1929  |  0.003881  |
03/07/2020 01:24:32 AM - INFO - |  12  |  PERCEPTRON  |  84.43%  |  [0.861  0.868  0.8654 0.8614 0.8536]  |  86.19 (+/- 0.98)  |  0.1085  |  0.00389  |
03/07/2020 01:24:32 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  84.94%  |  [0.8498 0.8534 0.8468 0.8452 0.8464]  |  84.83 (+/- 0.59)  |  35.85  |  1.299  |
03/07/2020 01:24:32 AM - INFO - |  14  |  RIDGE_CLASSIFIER  |  86.90%  |  [0.8836 0.895  0.8888 0.882  0.8788]  |  88.56 (+/- 1.14)  |  0.4334  |  0.008216  |
03/07/2020 01:24:32 AM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 01:24:32 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:24:32 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:24:32 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  84.60%  |  [0.8398 0.8516 0.8416 0.8366 0.8416]  |  84.22 (+/- 1.00)  |  103.8  |  5.642  |
03/07/2020 01:24:32 AM - INFO - |  2  |  BERNOULLI_NB  |  81.28%  |  [0.8398 0.8424 0.8514 0.8396 0.8516]  |  84.50 (+/- 1.09)  |  0.02541  |  0.02126  |
03/07/2020 01:24:32 AM - INFO - |  3  |  COMPLEMENT_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01709  |  0.009721  |
03/07/2020 01:24:32 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  73.55%  |  [0.7342 0.728  0.7374 0.7336 0.7266]  |  73.20 (+/- 0.81)  |  13.62  |  0.01241  |
03/07/2020 01:24:32 AM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  82.66%  |  [0.8632 0.8744 0.8694 0.864  0.8618]  |  86.66 (+/- 0.94)  |  0.007226  |  26.41  |
03/07/2020 01:24:32 AM - INFO - |  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  88.44 (+/- 0.98)  |  0.2293  |  0.003983  |
03/07/2020 01:24:32 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  87.75%  |  [0.8882 0.897  0.8878 0.8876 0.8818]  |  88.85 (+/- 0.97)  |  1.763  |  0.008308  |
03/07/2020 01:24:32 AM - INFO - |  9  |  MULTINOMIAL_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01575  |  0.008321  |
03/07/2020 01:24:32 AM - INFO - |  10  |  NEAREST_CENTROID  |  84.65%  |  [0.8426 0.8546 0.8426 0.8522 0.8502]  |  84.84 (+/- 0.99)  |  0.01649  |  0.01698  |
03/07/2020 01:24:32 AM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  88.08%  |  [0.8874 0.897  0.8884 0.8874 0.8852]  |  88.91 (+/- 0.82)  |  0.8401  |  0.003811  |
03/07/2020 01:24:32 AM - INFO - |  12  |  PERCEPTRON  |  80.64%  |  [0.8166 0.8264 0.8144 0.81   0.8102]  |  81.55 (+/- 1.20)  |  0.09111  |  0.00689  |
03/07/2020 01:24:32 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  85.39%  |  [0.8488 0.8534 0.8516 0.8448 0.8564]  |  85.10 (+/- 0.79)  |  33.64  |  2.238  |
03/07/2020 01:24:32 AM - INFO - |  14  |  RIDGE_CLASSIFIER  |  86.90%  |  [0.8838 0.8952 0.8892 0.882  0.8788]  |  88.58 (+/- 1.16)  |  0.4678  |  0.008244  |
03/07/2020 01:24:32 AM - INFO - 

03/07/2020 01:24:32 AM - INFO - ################################################################################
03/07/2020 01:24:32 AM - INFO - 14)
03/07/2020 01:24:32 AM - INFO - ********************************************************************************
03/07/2020 01:24:32 AM - INFO - Classifier: GRADIENT_BOOSTING_CLASSIFIER, Dataset: IMDB_REVIEWS
03/07/2020 01:24:32 AM - INFO - ********************************************************************************
03/07/2020 01:24:39 AM - INFO - 

Performing grid search...

03/07/2020 01:24:39 AM - INFO - Parameters:
03/07/2020 01:24:39 AM - INFO - {'classifier__learning_rate': [0.01, 0.1], 'classifier__n_estimators': [100, 200]}
03/07/2020 01:34:57 AM - INFO - 	Done in 618.682s
03/07/2020 01:34:57 AM - INFO - 	Best score: 0.826
03/07/2020 01:34:57 AM - INFO - 	Best parameters set:
03/07/2020 01:34:57 AM - INFO - 		classifier__learning_rate: 0.1
03/07/2020 01:34:57 AM - INFO - 		classifier__n_estimators: 200
03/07/2020 01:34:57 AM - INFO - 

USING GRADIENT_BOOSTING_CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 01:34:57 AM - INFO - ________________________________________________________________________________
03/07/2020 01:34:57 AM - INFO - Training: 
03/07/2020 01:34:57 AM - INFO - GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.1, loss='deviance', max_depth=3,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_iter_no_change=None, presort='deprecated',
                           random_state=None, subsample=1.0, tol=0.0001,
                           validation_fraction=0.1, verbose=0,
                           warm_start=False)
03/07/2020 01:35:48 AM - INFO - Train time: 50.354s
03/07/2020 01:35:48 AM - INFO - Test time:  0.051s
03/07/2020 01:35:48 AM - INFO - Accuracy score:   0.807
03/07/2020 01:35:48 AM - INFO - 

===> Classification Report:

03/07/2020 01:35:48 AM - INFO -               precision    recall  f1-score   support

           0       0.85      0.75      0.80     12500
           1       0.77      0.87      0.82     12500

    accuracy                           0.81     25000
   macro avg       0.81      0.81      0.81     25000
weighted avg       0.81      0.81      0.81     25000

03/07/2020 01:35:48 AM - INFO - 

Cross validation:
03/07/2020 01:37:16 AM - INFO - 	accuracy: 5-fold cross validation: [0.805  0.8092 0.8048 0.7984 0.807 ]
03/07/2020 01:37:16 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 80.49 (+/- 0.72)
03/07/2020 01:37:16 AM - INFO - 

USING GRADIENT_BOOSTING_CLASSIFIER WITH BEST PARAMETERS: {'learning_rate': 0.1, 'n_estimators': 200}
03/07/2020 01:37:16 AM - INFO - ________________________________________________________________________________
03/07/2020 01:37:16 AM - INFO - Training: 
03/07/2020 01:37:16 AM - INFO - GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.1, loss='deviance', max_depth=3,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=200,
                           n_iter_no_change=None, presort='deprecated',
                           random_state=None, subsample=1.0, tol=0.0001,
                           validation_fraction=0.1, verbose=0,
                           warm_start=False)
03/07/2020 01:38:57 AM - INFO - Train time: 100.627s
03/07/2020 01:38:57 AM - INFO - Test time:  0.065s
03/07/2020 01:38:57 AM - INFO - Accuracy score:   0.829
03/07/2020 01:38:57 AM - INFO - 

===> Classification Report:

03/07/2020 01:38:57 AM - INFO -               precision    recall  f1-score   support

           0       0.86      0.79      0.82     12500
           1       0.80      0.87      0.84     12500

    accuracy                           0.83     25000
   macro avg       0.83      0.83      0.83     25000
weighted avg       0.83      0.83      0.83     25000

03/07/2020 01:38:57 AM - INFO - 

Cross validation:
03/07/2020 01:41:55 AM - INFO - 	accuracy: 5-fold cross validation: [0.828  0.8306 0.8238 0.8226 0.8284]
03/07/2020 01:41:55 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 82.67 (+/- 0.60)
03/07/2020 01:41:55 AM - INFO - It took 1042.7061727046967 seconds
03/07/2020 01:41:55 AM - INFO - ********************************************************************************
03/07/2020 01:41:55 AM - INFO - ################################################################################
03/07/2020 01:41:55 AM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 01:41:55 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:41:55 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:41:55 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  80.24%  |  [0.801  0.803  0.799  0.7968 0.7986]  |  79.97 (+/- 0.43)  |  10.44  |  0.5797  |
03/07/2020 01:41:55 AM - INFO - |  2  |  BERNOULLI_NB  |  81.52%  |  [0.8384 0.8398 0.8524 0.8404 0.8524]  |  84.47 (+/- 1.27)  |  0.02364  |  0.02019  |
03/07/2020 01:41:55 AM - INFO - |  3  |  COMPLEMENT_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01559  |  0.008243  |
03/07/2020 01:41:55 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  71.44%  |  [0.7116 0.7088 0.7172 0.7138 0.7112]  |  71.25 (+/- 0.57)  |  23.48  |  0.03173  |
03/07/2020 01:41:55 AM - INFO - |  5  |  GRADIENT_BOOSTING_CLASSIFIER  |  80.72%  |  [0.805  0.8092 0.8048 0.7984 0.807 ]  |  80.49 (+/- 0.72)  |  50.35  |  0.05117  |
03/07/2020 01:41:55 AM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  73.29%  |  [0.8144 0.8248 0.8262 0.814  0.815 ]  |  81.89 (+/- 1.09)  |  0.009318  |  26.89  |
03/07/2020 01:41:55 AM - INFO - |  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  88.44 (+/- 0.98)  |  0.2275  |  0.003976  |
03/07/2020 01:41:55 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  88.39%  |  [0.8822 0.8946 0.8848 0.887  0.8852]  |  88.68 (+/- 0.84)  |  1.157  |  0.008317  |
03/07/2020 01:41:55 AM - INFO - |  9  |  MULTINOMIAL_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01562  |  0.008306  |
03/07/2020 01:41:55 AM - INFO - |  10  |  NEAREST_CENTROID  |  83.72%  |  [0.8316 0.838  0.8342 0.8392 0.8358]  |  83.58 (+/- 0.54)  |  0.01647  |  0.01182  |
03/07/2020 01:41:55 AM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  85.13%  |  [0.8666 0.8772 0.8748 0.8698 0.8644]  |  87.06 (+/- 0.96)  |  0.1929  |  0.003881  |
03/07/2020 01:41:55 AM - INFO - |  12  |  PERCEPTRON  |  84.43%  |  [0.861  0.868  0.8654 0.8614 0.8536]  |  86.19 (+/- 0.98)  |  0.1085  |  0.00389  |
03/07/2020 01:41:55 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  84.94%  |  [0.8498 0.8534 0.8468 0.8452 0.8464]  |  84.83 (+/- 0.59)  |  35.85  |  1.299  |
03/07/2020 01:41:55 AM - INFO - |  14  |  RIDGE_CLASSIFIER  |  86.90%  |  [0.8836 0.895  0.8888 0.882  0.8788]  |  88.56 (+/- 1.14)  |  0.4334  |  0.008216  |
03/07/2020 01:41:55 AM - INFO - 

CURRENT CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 01:41:55 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:41:55 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:41:55 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  84.60%  |  [0.8398 0.8516 0.8416 0.8366 0.8416]  |  84.22 (+/- 1.00)  |  103.8  |  5.642  |
03/07/2020 01:41:55 AM - INFO - |  2  |  BERNOULLI_NB  |  81.28%  |  [0.8398 0.8424 0.8514 0.8396 0.8516]  |  84.50 (+/- 1.09)  |  0.02541  |  0.02126  |
03/07/2020 01:41:55 AM - INFO - |  3  |  COMPLEMENT_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01709  |  0.009721  |
03/07/2020 01:41:55 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  73.55%  |  [0.7342 0.728  0.7374 0.7336 0.7266]  |  73.20 (+/- 0.81)  |  13.62  |  0.01241  |
03/07/2020 01:41:55 AM - INFO - |  5  |  GRADIENT_BOOSTING_CLASSIFIER  |  82.87%  |  [0.828  0.8306 0.8238 0.8226 0.8284]  |  82.67 (+/- 0.60)  |  100.6  |  0.0646  |
03/07/2020 01:41:55 AM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  82.66%  |  [0.8632 0.8744 0.8694 0.864  0.8618]  |  86.66 (+/- 0.94)  |  0.007226  |  26.41  |
03/07/2020 01:41:55 AM - INFO - |  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  88.44 (+/- 0.98)  |  0.2293  |  0.003983  |
03/07/2020 01:41:55 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  87.75%  |  [0.8882 0.897  0.8878 0.8876 0.8818]  |  88.85 (+/- 0.97)  |  1.763  |  0.008308  |
03/07/2020 01:41:55 AM - INFO - |  9  |  MULTINOMIAL_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01575  |  0.008321  |
03/07/2020 01:41:55 AM - INFO - |  10  |  NEAREST_CENTROID  |  84.65%  |  [0.8426 0.8546 0.8426 0.8522 0.8502]  |  84.84 (+/- 0.99)  |  0.01649  |  0.01698  |
03/07/2020 01:41:55 AM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  88.08%  |  [0.8874 0.897  0.8884 0.8874 0.8852]  |  88.91 (+/- 0.82)  |  0.8401  |  0.003811  |
03/07/2020 01:41:55 AM - INFO - |  12  |  PERCEPTRON  |  80.64%  |  [0.8166 0.8264 0.8144 0.81   0.8102]  |  81.55 (+/- 1.20)  |  0.09111  |  0.00689  |
03/07/2020 01:41:55 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  85.39%  |  [0.8488 0.8534 0.8516 0.8448 0.8564]  |  85.10 (+/- 0.79)  |  33.64  |  2.238  |
03/07/2020 01:41:55 AM - INFO - |  14  |  RIDGE_CLASSIFIER  |  86.90%  |  [0.8838 0.8952 0.8892 0.882  0.8788]  |  88.58 (+/- 1.16)  |  0.4678  |  0.008244  |
03/07/2020 01:41:55 AM - INFO - 

FINAL CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 01:41:55 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:41:55 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:41:55 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  80.24%  |  [0.801  0.803  0.799  0.7968 0.7986]  |  79.97 (+/- 0.43)  |  10.44  |  0.5797  |
03/07/2020 01:41:55 AM - INFO - |  2  |  BERNOULLI_NB  |  81.52%  |  [0.8384 0.8398 0.8524 0.8404 0.8524]  |  84.47 (+/- 1.27)  |  0.02364  |  0.02019  |
03/07/2020 01:41:55 AM - INFO - |  3  |  COMPLEMENT_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01559  |  0.008243  |
03/07/2020 01:41:55 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  71.44%  |  [0.7116 0.7088 0.7172 0.7138 0.7112]  |  71.25 (+/- 0.57)  |  23.48  |  0.03173  |
03/07/2020 01:41:55 AM - INFO - |  5  |  GRADIENT_BOOSTING_CLASSIFIER  |  80.72%  |  [0.805  0.8092 0.8048 0.7984 0.807 ]  |  80.49 (+/- 0.72)  |  50.35  |  0.05117  |
03/07/2020 01:41:55 AM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  73.29%  |  [0.8144 0.8248 0.8262 0.814  0.815 ]  |  81.89 (+/- 1.09)  |  0.009318  |  26.89  |
03/07/2020 01:41:55 AM - INFO - |  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  88.44 (+/- 0.98)  |  0.2275  |  0.003976  |
03/07/2020 01:41:55 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  88.39%  |  [0.8822 0.8946 0.8848 0.887  0.8852]  |  88.68 (+/- 0.84)  |  1.157  |  0.008317  |
03/07/2020 01:41:55 AM - INFO - |  9  |  MULTINOMIAL_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01562  |  0.008306  |
03/07/2020 01:41:55 AM - INFO - |  10  |  NEAREST_CENTROID  |  83.72%  |  [0.8316 0.838  0.8342 0.8392 0.8358]  |  83.58 (+/- 0.54)  |  0.01647  |  0.01182  |
03/07/2020 01:41:55 AM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  85.13%  |  [0.8666 0.8772 0.8748 0.8698 0.8644]  |  87.06 (+/- 0.96)  |  0.1929  |  0.003881  |
03/07/2020 01:41:55 AM - INFO - |  12  |  PERCEPTRON  |  84.43%  |  [0.861  0.868  0.8654 0.8614 0.8536]  |  86.19 (+/- 0.98)  |  0.1085  |  0.00389  |
03/07/2020 01:41:55 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  84.94%  |  [0.8498 0.8534 0.8468 0.8452 0.8464]  |  84.83 (+/- 0.59)  |  35.85  |  1.299  |
03/07/2020 01:41:55 AM - INFO - |  14  |  RIDGE_CLASSIFIER  |  86.90%  |  [0.8836 0.895  0.8888 0.882  0.8788]  |  88.56 (+/- 1.14)  |  0.4334  |  0.008216  |
03/07/2020 01:41:55 AM - INFO - 

FINAL CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 01:41:55 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:41:55 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:41:55 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  84.60%  |  [0.8398 0.8516 0.8416 0.8366 0.8416]  |  84.22 (+/- 1.00)  |  103.8  |  5.642  |
03/07/2020 01:41:55 AM - INFO - |  2  |  BERNOULLI_NB  |  81.28%  |  [0.8398 0.8424 0.8514 0.8396 0.8516]  |  84.50 (+/- 1.09)  |  0.02541  |  0.02126  |
03/07/2020 01:41:55 AM - INFO - |  3  |  COMPLEMENT_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01709  |  0.009721  |
03/07/2020 01:41:55 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  73.55%  |  [0.7342 0.728  0.7374 0.7336 0.7266]  |  73.20 (+/- 0.81)  |  13.62  |  0.01241  |
03/07/2020 01:41:55 AM - INFO - |  5  |  GRADIENT_BOOSTING_CLASSIFIER  |  82.87%  |  [0.828  0.8306 0.8238 0.8226 0.8284]  |  82.67 (+/- 0.60)  |  100.6  |  0.0646  |
03/07/2020 01:41:55 AM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  82.66%  |  [0.8632 0.8744 0.8694 0.864  0.8618]  |  86.66 (+/- 0.94)  |  0.007226  |  26.41  |
03/07/2020 01:41:55 AM - INFO - |  7  |  LINEAR_SVC  |  87.13%  |  [0.8838 0.8932 0.883  0.8836 0.8782]  |  88.44 (+/- 0.98)  |  0.2293  |  0.003983  |
03/07/2020 01:41:55 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  87.75%  |  [0.8882 0.897  0.8878 0.8876 0.8818]  |  88.85 (+/- 0.97)  |  1.763  |  0.008308  |
03/07/2020 01:41:55 AM - INFO - |  9  |  MULTINOMIAL_NB  |  83.93%  |  [0.8564 0.8678 0.8682 0.8678 0.8672]  |  86.55 (+/- 0.91)  |  0.01575  |  0.008321  |
03/07/2020 01:41:55 AM - INFO - |  10  |  NEAREST_CENTROID  |  84.65%  |  [0.8426 0.8546 0.8426 0.8522 0.8502]  |  84.84 (+/- 0.99)  |  0.01649  |  0.01698  |
03/07/2020 01:41:55 AM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  88.08%  |  [0.8874 0.897  0.8884 0.8874 0.8852]  |  88.91 (+/- 0.82)  |  0.8401  |  0.003811  |
03/07/2020 01:41:55 AM - INFO - |  12  |  PERCEPTRON  |  80.64%  |  [0.8166 0.8264 0.8144 0.81   0.8102]  |  81.55 (+/- 1.20)  |  0.09111  |  0.00689  |
03/07/2020 01:41:55 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  85.39%  |  [0.8488 0.8534 0.8516 0.8448 0.8564]  |  85.10 (+/- 0.79)  |  33.64  |  2.238  |
03/07/2020 01:41:55 AM - INFO - |  14  |  RIDGE_CLASSIFIER  |  86.90%  |  [0.8838 0.8952 0.8892 0.882  0.8788]  |  88.58 (+/- 1.16)  |  0.4678  |  0.008244  |
03/07/2020 01:41:55 AM - INFO - 

03/07/2020 01:41:55 AM - INFO - ################################################################################
03/07/2020 01:41:55 AM - INFO - 1)
03/07/2020 01:41:55 AM - INFO - ********************************************************************************
03/07/2020 01:41:55 AM - INFO - Classifier: ADA_BOOST_CLASSIFIER, Dataset: TWENTY_NEWS_GROUPS
03/07/2020 01:41:55 AM - INFO - ********************************************************************************
03/07/2020 01:41:58 AM - INFO - 

Performing grid search...

03/07/2020 01:41:58 AM - INFO - Parameters:
03/07/2020 01:41:58 AM - INFO - {'classifier__learning_rate': [0.1, 1], 'classifier__n_estimators': [200, 500]}
03/07/2020 01:53:11 AM - INFO - 	Done in 672.493s
03/07/2020 01:53:11 AM - INFO - 	Best score: 0.466
03/07/2020 01:53:11 AM - INFO - 	Best parameters set:
03/07/2020 01:53:11 AM - INFO - 		classifier__learning_rate: 1
03/07/2020 01:53:11 AM - INFO - 		classifier__n_estimators: 200
03/07/2020 01:53:11 AM - INFO - 

USING ADA_BOOST_CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 01:53:11 AM - INFO - ________________________________________________________________________________
03/07/2020 01:53:11 AM - INFO - Training: 
03/07/2020 01:53:11 AM - INFO - AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
                   n_estimators=50, random_state=None)
03/07/2020 01:53:15 AM - INFO - Train time: 4.665s
03/07/2020 01:53:15 AM - INFO - Test time:  0.258s
03/07/2020 01:53:15 AM - INFO - Accuracy score:   0.365
03/07/2020 01:53:15 AM - INFO - 

===> Classification Report:

03/07/2020 01:53:15 AM - INFO -               precision    recall  f1-score   support

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

03/07/2020 01:53:15 AM - INFO - 

Cross validation:
03/07/2020 01:53:40 AM - INFO - 	accuracy: 5-fold cross validation: [0.39460893 0.39019001 0.39991162 0.39019001 0.40539346]
03/07/2020 01:53:40 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 39.61 (+/- 1.18)
03/07/2020 01:53:40 AM - INFO - 

USING ADA_BOOST_CLASSIFIER WITH BEST PARAMETERS: {'learning_rate': 1, 'n_estimators': 200}
03/07/2020 01:53:40 AM - INFO - ________________________________________________________________________________
03/07/2020 01:53:40 AM - INFO - Training: 
03/07/2020 01:53:40 AM - INFO - AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1,
                   n_estimators=200, random_state=None)
03/07/2020 01:53:59 AM - INFO - Train time: 18.420s
03/07/2020 01:54:00 AM - INFO - Test time:  0.933s
03/07/2020 01:54:00 AM - INFO - Accuracy score:   0.440
03/07/2020 01:54:00 AM - INFO - 

===> Classification Report:

03/07/2020 01:54:00 AM - INFO -               precision    recall  f1-score   support

           0       0.39      0.25      0.30       319
           1       0.38      0.46      0.41       389
           2       0.53      0.41      0.46       394
           3       0.50      0.48      0.49       392
           4       0.62      0.48      0.54       385
           5       0.62      0.49      0.54       395
           6       0.53      0.54      0.53       390
           7       0.73      0.39      0.51       396
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

03/07/2020 01:54:00 AM - INFO - 

Cross validation:
03/07/2020 01:55:39 AM - INFO - 	accuracy: 5-fold cross validation: [0.4719399  0.45735749 0.46531153 0.46973045 0.47126437]
03/07/2020 01:55:39 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 46.71 (+/- 1.08)
03/07/2020 01:55:39 AM - INFO - It took 823.6370670795441 seconds
03/07/2020 01:55:39 AM - INFO - ********************************************************************************
03/07/2020 01:55:39 AM - INFO - ################################################################################
03/07/2020 01:55:39 AM - INFO - 

CURRENT CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 01:55:39 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:55:39 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:55:39 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  36.52%  |  [0.39460893 0.39019001 0.39991162 0.39019001 0.40539346]  |  39.61 (+/- 1.18)  |  4.665  |  0.2575  |
03/07/2020 01:55:39 AM - INFO - 

CURRENT CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 01:55:39 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:55:39 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:55:39 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  44.04%  |  [0.4719399  0.45735749 0.46531153 0.46973045 0.47126437]  |  46.71 (+/- 1.08)  |  18.42  |  0.9331  |
03/07/2020 01:55:39 AM - INFO - 

03/07/2020 01:55:39 AM - INFO - ################################################################################
03/07/2020 01:55:39 AM - INFO - 2)
03/07/2020 01:55:39 AM - INFO - ********************************************************************************
03/07/2020 01:55:39 AM - INFO - Classifier: DECISION_TREE_CLASSIFIER, Dataset: TWENTY_NEWS_GROUPS
03/07/2020 01:55:39 AM - INFO - ********************************************************************************
03/07/2020 01:55:42 AM - INFO - 

Performing grid search...

03/07/2020 01:55:42 AM - INFO - Parameters:
03/07/2020 01:55:42 AM - INFO - {'classifier__criterion': ['entropy', 'gini'], 'classifier__splitter': ['best', 'random'], 'classifier__min_samples_split': [2, 100, 250]}
03/07/2020 01:56:55 AM - INFO - 	Done in 73.606s
03/07/2020 01:56:55 AM - INFO - 	Best score: 0.496
03/07/2020 01:56:55 AM - INFO - 	Best parameters set:
03/07/2020 01:56:55 AM - INFO - 		classifier__criterion: 'gini'
03/07/2020 01:56:55 AM - INFO - 		classifier__min_samples_split: 100
03/07/2020 01:56:55 AM - INFO - 		classifier__splitter: 'random'
03/07/2020 01:56:55 AM - INFO - 

USING DECISION_TREE_CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 01:56:55 AM - INFO - ________________________________________________________________________________
03/07/2020 01:56:55 AM - INFO - Training: 
03/07/2020 01:56:55 AM - INFO - DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=None, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=None, splitter='best')
03/07/2020 01:57:06 AM - INFO - Train time: 10.423s
03/07/2020 01:57:06 AM - INFO - Test time:  0.009s
03/07/2020 01:57:06 AM - INFO - Accuracy score:   0.430
03/07/2020 01:57:06 AM - INFO - 

===> Classification Report:

03/07/2020 01:57:06 AM - INFO -               precision    recall  f1-score   support

           0       0.28      0.26      0.27       319
           1       0.40      0.43      0.41       389
           2       0.45      0.38      0.42       394
           3       0.40      0.37      0.38       392
           4       0.47      0.45      0.46       385
           5       0.53      0.48      0.50       395
           6       0.55      0.54      0.55       390
           7       0.26      0.55      0.35       396
           8       0.58      0.51      0.54       398
           9       0.54      0.46      0.50       397
          10       0.59      0.60      0.60       399
          11       0.56      0.47      0.51       396
          12       0.29      0.29      0.29       393
          13       0.45      0.40      0.43       396
          14       0.48      0.50      0.49       394
          15       0.46      0.45      0.46       398
          16       0.37      0.39      0.38       364
          17       0.61      0.51      0.56       376
          18       0.22      0.18      0.20       310
          19       0.18      0.18      0.18       251

    accuracy                           0.43      7532
   macro avg       0.43      0.42      0.42      7532
weighted avg       0.44      0.43      0.43      7532

03/07/2020 01:57:06 AM - INFO - 

Cross validation:
03/07/2020 01:57:16 AM - INFO - 	accuracy: 5-fold cross validation: [0.4922669  0.47370747 0.46486964 0.47282369 0.48983201]
03/07/2020 01:57:16 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 47.87 (+/- 2.11)
03/07/2020 01:57:16 AM - INFO - 

USING DECISION_TREE_CLASSIFIER WITH BEST PARAMETERS: {'criterion': 'gini', 'min_samples_split': 100, 'splitter': 'random'}
03/07/2020 01:57:16 AM - INFO - ________________________________________________________________________________
03/07/2020 01:57:16 AM - INFO - Training: 
03/07/2020 01:57:16 AM - INFO - DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=None, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=100,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=None, splitter='random')
03/07/2020 01:57:20 AM - INFO - Train time: 4.511s
03/07/2020 01:57:20 AM - INFO - Test time:  0.005s
03/07/2020 01:57:20 AM - INFO - Accuracy score:   0.456
03/07/2020 01:57:20 AM - INFO - 

===> Classification Report:

03/07/2020 01:57:20 AM - INFO -               precision    recall  f1-score   support

           0       0.36      0.28      0.31       319
           1       0.45      0.40      0.42       389
           2       0.44      0.45      0.44       394
           3       0.39      0.43      0.41       392
           4       0.43      0.50      0.46       385
           5       0.45      0.54      0.49       395
           6       0.63      0.58      0.61       390
           7       0.26      0.54      0.35       396
           8       0.56      0.54      0.55       398
           9       0.53      0.48      0.50       397
          10       0.55      0.60      0.58       399
          11       0.71      0.46      0.56       396
          12       0.34      0.32      0.33       393
          13       0.57      0.43      0.49       396
          14       0.59      0.46      0.52       394
          15       0.49      0.62      0.55       398
          16       0.41      0.46      0.43       364
          17       0.67      0.51      0.58       376
          18       0.25      0.23      0.24       310
          19       0.15      0.06      0.09       251

    accuracy                           0.46      7532
   macro avg       0.46      0.44      0.45      7532
weighted avg       0.47      0.46      0.46      7532

03/07/2020 01:57:20 AM - INFO - 

Cross validation:
03/07/2020 01:57:25 AM - INFO - 	accuracy: 5-fold cross validation: [0.49536014 0.49005745 0.48077773 0.47989395 0.49955791]
03/07/2020 01:57:25 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 48.91 (+/- 1.56)
03/07/2020 01:57:25 AM - INFO - It took 106.55138754844666 seconds
03/07/2020 01:57:25 AM - INFO - ********************************************************************************
03/07/2020 01:57:25 AM - INFO - ################################################################################
03/07/2020 01:57:25 AM - INFO - 

CURRENT CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 01:57:25 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:57:25 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:57:25 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  36.52%  |  [0.39460893 0.39019001 0.39991162 0.39019001 0.40539346]  |  39.61 (+/- 1.18)  |  4.665  |  0.2575  |
03/07/2020 01:57:25 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  42.95%  |  [0.4922669  0.47370747 0.46486964 0.47282369 0.48983201]  |  47.87 (+/- 2.11)  |  10.42  |  0.008561  |
03/07/2020 01:57:25 AM - INFO - 

CURRENT CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 01:57:25 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:57:25 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:57:25 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  44.04%  |  [0.4719399  0.45735749 0.46531153 0.46973045 0.47126437]  |  46.71 (+/- 1.08)  |  18.42  |  0.9331  |
03/07/2020 01:57:25 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  45.63%  |  [0.49536014 0.49005745 0.48077773 0.47989395 0.49955791]  |  48.91 (+/- 1.56)  |  4.511  |  0.005316  |
03/07/2020 01:57:25 AM - INFO - 

03/07/2020 01:57:25 AM - INFO - ################################################################################
03/07/2020 01:57:25 AM - INFO - 3)
03/07/2020 01:57:25 AM - INFO - ********************************************************************************
03/07/2020 01:57:25 AM - INFO - Classifier: LINEAR_SVC, Dataset: TWENTY_NEWS_GROUPS
03/07/2020 01:57:25 AM - INFO - ********************************************************************************
03/07/2020 01:57:28 AM - INFO - 

Performing grid search...

03/07/2020 01:57:28 AM - INFO - Parameters:
03/07/2020 01:57:28 AM - INFO - {'classifier__C': [0.01, 1.0], 'classifier__multi_class': ['ovr', 'crammer_singer'], 'classifier__tol': [0.0001, 0.001]}
03/07/2020 01:58:20 AM - INFO - 	Done in 51.833s
03/07/2020 01:58:20 AM - INFO - 	Best score: 0.757
03/07/2020 01:58:20 AM - INFO - 	Best parameters set:
03/07/2020 01:58:20 AM - INFO - 		classifier__C: 1.0
03/07/2020 01:58:20 AM - INFO - 		classifier__multi_class: 'ovr'
03/07/2020 01:58:20 AM - INFO - 		classifier__tol: 0.0001
03/07/2020 01:58:20 AM - INFO - 

USING LINEAR_SVC WITH DEFAULT PARAMETERS
03/07/2020 01:58:20 AM - INFO - ________________________________________________________________________________
03/07/2020 01:58:20 AM - INFO - Training: 
03/07/2020 01:58:20 AM - INFO - LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
          intercept_scaling=1, loss='squared_hinge', max_iter=1000,
          multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
          verbose=0)
03/07/2020 01:58:21 AM - INFO - Train time: 0.839s
03/07/2020 01:58:21 AM - INFO - Test time:  0.009s
03/07/2020 01:58:21 AM - INFO - Accuracy score:   0.698
03/07/2020 01:58:21 AM - INFO - 

===> Classification Report:

03/07/2020 01:58:21 AM - INFO -               precision    recall  f1-score   support

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

03/07/2020 01:58:21 AM - INFO - 

Cross validation:
03/07/2020 01:58:22 AM - INFO - 	accuracy: 5-fold cross validation: [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]
03/07/2020 01:58:22 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 75.73 (+/- 2.09)
03/07/2020 01:58:22 AM - INFO - 

USING LINEAR_SVC WITH BEST PARAMETERS: {'C': 1.0, 'multi_class': 'ovr', 'tol': 0.0001}
03/07/2020 01:58:22 AM - INFO - ________________________________________________________________________________
03/07/2020 01:58:22 AM - INFO - Training: 
03/07/2020 01:58:22 AM - INFO - LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
          intercept_scaling=1, loss='squared_hinge', max_iter=1000,
          multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
          verbose=0)
03/07/2020 01:58:23 AM - INFO - Train time: 0.846s
03/07/2020 01:58:23 AM - INFO - Test time:  0.009s
03/07/2020 01:58:23 AM - INFO - Accuracy score:   0.698
03/07/2020 01:58:23 AM - INFO - 

===> Classification Report:

03/07/2020 01:58:23 AM - INFO -               precision    recall  f1-score   support

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

03/07/2020 01:58:23 AM - INFO - 

Cross validation:
03/07/2020 01:58:25 AM - INFO - 	accuracy: 5-fold cross validation: [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]
03/07/2020 01:58:25 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 75.73 (+/- 2.09)
03/07/2020 01:58:25 AM - INFO - It took 59.31565260887146 seconds
03/07/2020 01:58:25 AM - INFO - ********************************************************************************
03/07/2020 01:58:25 AM - INFO - ################################################################################
03/07/2020 01:58:25 AM - INFO - 

CURRENT CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 01:58:25 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:58:25 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:58:25 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  36.52%  |  [0.39460893 0.39019001 0.39991162 0.39019001 0.40539346]  |  39.61 (+/- 1.18)  |  4.665  |  0.2575  |
03/07/2020 01:58:25 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  42.95%  |  [0.4922669  0.47370747 0.46486964 0.47282369 0.48983201]  |  47.87 (+/- 2.11)  |  10.42  |  0.008561  |
03/07/2020 01:58:25 AM - INFO - |  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  75.73 (+/- 2.09)  |  0.8393  |  0.008751  |
03/07/2020 01:58:25 AM - INFO - 

CURRENT CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 01:58:25 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 01:58:25 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 01:58:25 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  44.04%  |  [0.4719399  0.45735749 0.46531153 0.46973045 0.47126437]  |  46.71 (+/- 1.08)  |  18.42  |  0.9331  |
03/07/2020 01:58:25 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  45.63%  |  [0.49536014 0.49005745 0.48077773 0.47989395 0.49955791]  |  48.91 (+/- 1.56)  |  4.511  |  0.005316  |
03/07/2020 01:58:25 AM - INFO - |  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  75.73 (+/- 2.09)  |  0.8455  |  0.008558  |
03/07/2020 01:58:25 AM - INFO - 

03/07/2020 01:58:25 AM - INFO - ################################################################################
03/07/2020 01:58:25 AM - INFO - 4)
03/07/2020 01:58:25 AM - INFO - ********************************************************************************
03/07/2020 01:58:25 AM - INFO - Classifier: LOGISTIC_REGRESSION, Dataset: TWENTY_NEWS_GROUPS
03/07/2020 01:58:25 AM - INFO - ********************************************************************************
03/07/2020 01:58:28 AM - INFO - 

Performing grid search...

03/07/2020 01:58:28 AM - INFO - Parameters:
03/07/2020 01:58:28 AM - INFO - {'classifier__C': [1, 10], 'classifier__tol': [0.001, 0.01]}
03/07/2020 02:03:04 AM - INFO - 	Done in 276.252s
03/07/2020 02:03:04 AM - INFO - 	Best score: 0.750
03/07/2020 02:03:04 AM - INFO - 	Best parameters set:
03/07/2020 02:03:04 AM - INFO - 		classifier__C: 10
03/07/2020 02:03:04 AM - INFO - 		classifier__tol: 0.001
03/07/2020 02:03:04 AM - INFO - 

USING LOGISTIC_REGRESSION WITH DEFAULT PARAMETERS
03/07/2020 02:03:04 AM - INFO - ________________________________________________________________________________
03/07/2020 02:03:04 AM - INFO - Training: 
03/07/2020 02:03:04 AM - INFO - LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)
03/07/2020 02:03:33 AM - INFO - Train time: 29.692s
03/07/2020 02:03:34 AM - INFO - Test time:  0.012s
03/07/2020 02:03:34 AM - INFO - Accuracy score:   0.692
03/07/2020 02:03:34 AM - INFO - 

===> Classification Report:

03/07/2020 02:03:34 AM - INFO -               precision    recall  f1-score   support

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

03/07/2020 02:03:34 AM - INFO - 

Cross validation:
03/07/2020 02:04:25 AM - INFO - 	accuracy: 5-fold cross validation: [0.74458683 0.72646929 0.74193548 0.74768007 0.72413793]
03/07/2020 02:04:25 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 73.70 (+/- 1.94)
03/07/2020 02:04:25 AM - INFO - 

USING LOGISTIC_REGRESSION WITH BEST PARAMETERS: {'C': 10, 'tol': 0.001}
03/07/2020 02:04:25 AM - INFO - ________________________________________________________________________________
03/07/2020 02:04:25 AM - INFO - Training: 
03/07/2020 02:04:25 AM - INFO - LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.001, verbose=0,
                   warm_start=False)
03/07/2020 02:04:58 AM - INFO - Train time: 33.127s
03/07/2020 02:04:58 AM - INFO - Test time:  0.012s
03/07/2020 02:04:58 AM - INFO - Accuracy score:   0.693
03/07/2020 02:04:58 AM - INFO - 

===> Classification Report:

03/07/2020 02:04:58 AM - INFO -               precision    recall  f1-score   support

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

03/07/2020 02:04:58 AM - INFO - 

Cross validation:
03/07/2020 02:06:06 AM - INFO - 	accuracy: 5-fold cross validation: [0.75961114 0.74016792 0.75519222 0.7565179  0.73607427]
03/07/2020 02:06:06 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 74.95 (+/- 1.90)
03/07/2020 02:06:06 AM - INFO - It took 461.20302391052246 seconds
03/07/2020 02:06:06 AM - INFO - ********************************************************************************
03/07/2020 02:06:06 AM - INFO - ################################################################################
03/07/2020 02:06:06 AM - INFO - 

CURRENT CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 02:06:06 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 02:06:06 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 02:06:06 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  36.52%  |  [0.39460893 0.39019001 0.39991162 0.39019001 0.40539346]  |  39.61 (+/- 1.18)  |  4.665  |  0.2575  |
03/07/2020 02:06:06 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  42.95%  |  [0.4922669  0.47370747 0.46486964 0.47282369 0.48983201]  |  47.87 (+/- 2.11)  |  10.42  |  0.008561  |
03/07/2020 02:06:06 AM - INFO - |  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  75.73 (+/- 2.09)  |  0.8393  |  0.008751  |
03/07/2020 02:06:06 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  69.17%  |  [0.74458683 0.72646929 0.74193548 0.74768007 0.72413793]  |  73.70 (+/- 1.94)  |  29.69  |  0.01173  |
03/07/2020 02:06:06 AM - INFO - 

CURRENT CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 02:06:06 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 02:06:06 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 02:06:06 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  44.04%  |  [0.4719399  0.45735749 0.46531153 0.46973045 0.47126437]  |  46.71 (+/- 1.08)  |  18.42  |  0.9331  |
03/07/2020 02:06:06 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  45.63%  |  [0.49536014 0.49005745 0.48077773 0.47989395 0.49955791]  |  48.91 (+/- 1.56)  |  4.511  |  0.005316  |
03/07/2020 02:06:06 AM - INFO - |  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  75.73 (+/- 2.09)  |  0.8455  |  0.008558  |
03/07/2020 02:06:06 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  69.28%  |  [0.75961114 0.74016792 0.75519222 0.7565179  0.73607427]  |  74.95 (+/- 1.90)  |  33.13  |  0.01171  |
03/07/2020 02:06:06 AM - INFO - 

03/07/2020 02:06:06 AM - INFO - ################################################################################
03/07/2020 02:06:06 AM - INFO - 5)
03/07/2020 02:06:06 AM - INFO - ********************************************************************************
03/07/2020 02:06:06 AM - INFO - Classifier: RANDOM_FOREST_CLASSIFIER, Dataset: TWENTY_NEWS_GROUPS
03/07/2020 02:06:06 AM - INFO - ********************************************************************************
03/07/2020 02:06:09 AM - INFO - 

Performing grid search...

03/07/2020 02:06:09 AM - INFO - Parameters:
03/07/2020 02:06:09 AM - INFO - {'classifier__min_samples_leaf': [1, 2], 'classifier__min_samples_split': [2, 5], 'classifier__n_estimators': [100, 200]}
03/07/2020 02:11:49 AM - INFO - 	Done in 339.848s
03/07/2020 02:11:49 AM - INFO - 	Best score: 0.685
03/07/2020 02:11:49 AM - INFO - 	Best parameters set:
03/07/2020 02:11:49 AM - INFO - 		classifier__min_samples_leaf: 1
03/07/2020 02:11:49 AM - INFO - 		classifier__min_samples_split: 5
03/07/2020 02:11:49 AM - INFO - 		classifier__n_estimators: 200
03/07/2020 02:11:49 AM - INFO - 

USING RANDOM_FOREST_CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 02:11:49 AM - INFO - ________________________________________________________________________________
03/07/2020 02:11:49 AM - INFO - Training: 
03/07/2020 02:11:49 AM - INFO - RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
03/07/2020 02:12:25 AM - INFO - Train time: 36.115s
03/07/2020 02:12:25 AM - INFO - Test time:  0.619s
03/07/2020 02:12:25 AM - INFO - Accuracy score:   0.624
03/07/2020 02:12:25 AM - INFO - 

===> Classification Report:

03/07/2020 02:12:25 AM - INFO -               precision    recall  f1-score   support

           0       0.47      0.41      0.44       319
           1       0.58      0.61      0.59       389
           2       0.56      0.65      0.60       394
           3       0.59      0.56      0.58       392
           4       0.66      0.64      0.65       385
           5       0.67      0.66      0.66       395
           6       0.67      0.74      0.70       390
           7       0.42      0.72      0.53       396
           8       0.72      0.67      0.70       398
           9       0.67      0.78      0.72       397
          10       0.83      0.82      0.83       399
          11       0.78      0.65      0.71       396
          12       0.51      0.44      0.47       393
          13       0.73      0.64      0.68       396
          14       0.69      0.66      0.68       394
          15       0.60      0.80      0.69       398
          16       0.53      0.60      0.56       364
          17       0.82      0.70      0.75       376
          18       0.51      0.31      0.39       310
          19       0.38      0.12      0.18       251

    accuracy                           0.62      7532
   macro avg       0.62      0.61      0.61      7532
weighted avg       0.63      0.62      0.62      7532

03/07/2020 02:12:25 AM - INFO - 

Cross validation:
03/07/2020 02:13:14 AM - INFO - 	accuracy: 5-fold cross validation: [0.68228016 0.6597437  0.67034909 0.67609368 0.66534041]
03/07/2020 02:13:14 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 67.08 (+/- 1.58)
03/07/2020 02:13:14 AM - INFO - 

USING RANDOM_FOREST_CLASSIFIER WITH BEST PARAMETERS: {'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 200}
03/07/2020 02:13:14 AM - INFO - ________________________________________________________________________________
03/07/2020 02:13:14 AM - INFO - Training: 
03/07/2020 02:13:14 AM - INFO - RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=5,
                       min_weight_fraction_leaf=0.0, n_estimators=200,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
03/07/2020 02:13:55 AM - INFO - Train time: 40.294s
03/07/2020 02:13:56 AM - INFO - Test time:  1.201s
03/07/2020 02:13:56 AM - INFO - Accuracy score:   0.638
03/07/2020 02:13:56 AM - INFO - 

===> Classification Report:

03/07/2020 02:13:56 AM - INFO -               precision    recall  f1-score   support

           0       0.46      0.38      0.42       319
           1       0.58      0.61      0.59       389
           2       0.58      0.64      0.61       394
           3       0.63      0.60      0.62       392
           4       0.68      0.67      0.67       385
           5       0.67      0.71      0.69       395
           6       0.72      0.74      0.73       390
           7       0.43      0.72      0.54       396
           8       0.71      0.71      0.71       398
           9       0.72      0.78      0.75       397
          10       0.83      0.85      0.84       399
          11       0.81      0.67      0.73       396
          12       0.55      0.45      0.49       393
          13       0.77      0.67      0.72       396
          14       0.68      0.69      0.68       394
          15       0.59      0.81      0.68       398
          16       0.53      0.62      0.58       364
          17       0.85      0.69      0.77       376
          18       0.48      0.34      0.40       310
          19       0.39      0.09      0.15       251

    accuracy                           0.64      7532
   macro avg       0.63      0.62      0.62      7532
weighted avg       0.64      0.64      0.63      7532

03/07/2020 02:13:56 AM - INFO - 

Cross validation:
03/07/2020 02:15:04 AM - INFO - 	accuracy: 5-fold cross validation: [0.69774635 0.6752099  0.68448962 0.69421122 0.67992927]
03/07/2020 02:15:04 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 68.63 (+/- 1.70)
03/07/2020 02:15:04 AM - INFO - It took 537.946937084198 seconds
03/07/2020 02:15:04 AM - INFO - ********************************************************************************
03/07/2020 02:15:04 AM - INFO - ################################################################################
03/07/2020 02:15:04 AM - INFO - 

CURRENT CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 02:15:04 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 02:15:04 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 02:15:04 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  36.52%  |  [0.39460893 0.39019001 0.39991162 0.39019001 0.40539346]  |  39.61 (+/- 1.18)  |  4.665  |  0.2575  |
03/07/2020 02:15:04 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  42.95%  |  [0.4922669  0.47370747 0.46486964 0.47282369 0.48983201]  |  47.87 (+/- 2.11)  |  10.42  |  0.008561  |
03/07/2020 02:15:04 AM - INFO - |  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  75.73 (+/- 2.09)  |  0.8393  |  0.008751  |
03/07/2020 02:15:04 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  69.17%  |  [0.74458683 0.72646929 0.74193548 0.74768007 0.72413793]  |  73.70 (+/- 1.94)  |  29.69  |  0.01173  |
03/07/2020 02:15:04 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  62.43%  |  [0.68228016 0.6597437  0.67034909 0.67609368 0.66534041]  |  67.08 (+/- 1.58)  |  36.12  |  0.619  |
03/07/2020 02:15:04 AM - INFO - 

CURRENT CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 02:15:04 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 02:15:04 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 02:15:04 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  44.04%  |  [0.4719399  0.45735749 0.46531153 0.46973045 0.47126437]  |  46.71 (+/- 1.08)  |  18.42  |  0.9331  |
03/07/2020 02:15:04 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  45.63%  |  [0.49536014 0.49005745 0.48077773 0.47989395 0.49955791]  |  48.91 (+/- 1.56)  |  4.511  |  0.005316  |
03/07/2020 02:15:04 AM - INFO - |  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  75.73 (+/- 2.09)  |  0.8455  |  0.008558  |
03/07/2020 02:15:04 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  69.28%  |  [0.75961114 0.74016792 0.75519222 0.7565179  0.73607427]  |  74.95 (+/- 1.90)  |  33.13  |  0.01171  |
03/07/2020 02:15:04 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  63.81%  |  [0.69774635 0.6752099  0.68448962 0.69421122 0.67992927]  |  68.63 (+/- 1.70)  |  40.29  |  1.201  |
03/07/2020 02:15:04 AM - INFO - 

03/07/2020 02:15:04 AM - INFO - ################################################################################
03/07/2020 02:15:04 AM - INFO - 6)
03/07/2020 02:15:04 AM - INFO - ********************************************************************************
03/07/2020 02:15:04 AM - INFO - Classifier: BERNOULLI_NB, Dataset: TWENTY_NEWS_GROUPS
03/07/2020 02:15:04 AM - INFO - ********************************************************************************
03/07/2020 02:15:07 AM - INFO - 

Performing grid search...

03/07/2020 02:15:07 AM - INFO - Parameters:
03/07/2020 02:15:07 AM - INFO - {'classifier__alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0], 'classifier__binarize': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0], 'classifier__fit_prior': [False, True]}
03/07/2020 02:15:41 AM - INFO - 	Done in 34.733s
03/07/2020 02:15:41 AM - INFO - 	Best score: 0.680
03/07/2020 02:15:41 AM - INFO - 	Best parameters set:
03/07/2020 02:15:41 AM - INFO - 		classifier__alpha: 0.1
03/07/2020 02:15:41 AM - INFO - 		classifier__binarize: 0.1
03/07/2020 02:15:41 AM - INFO - 		classifier__fit_prior: False
03/07/2020 02:15:41 AM - INFO - 

USING BERNOULLI_NB WITH DEFAULT PARAMETERS
03/07/2020 02:15:41 AM - INFO - ________________________________________________________________________________
03/07/2020 02:15:41 AM - INFO - Training: 
03/07/2020 02:15:41 AM - INFO - BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
03/07/2020 02:15:42 AM - INFO - Train time: 0.057s
03/07/2020 02:15:42 AM - INFO - Test time:  0.054s
03/07/2020 02:15:42 AM - INFO - Accuracy score:   0.458
03/07/2020 02:15:42 AM - INFO - 

===> Classification Report:

03/07/2020 02:15:42 AM - INFO -               precision    recall  f1-score   support

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

03/07/2020 02:15:42 AM - INFO - 

Cross validation:
03/07/2020 02:15:42 AM - INFO - 	accuracy: 5-fold cross validation: [0.48033584 0.46133451 0.48563853 0.47547503 0.48938992]
03/07/2020 02:15:42 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 47.84 (+/- 1.95)
03/07/2020 02:15:42 AM - INFO - 

USING BERNOULLI_NB WITH BEST PARAMETERS: {'alpha': 0.1, 'binarize': 0.1, 'fit_prior': False}
03/07/2020 02:15:42 AM - INFO - ________________________________________________________________________________
03/07/2020 02:15:42 AM - INFO - Training: 
03/07/2020 02:15:42 AM - INFO - BernoulliNB(alpha=0.1, binarize=0.1, class_prior=None, fit_prior=False)
03/07/2020 02:15:42 AM - INFO - Train time: 0.053s
03/07/2020 02:15:42 AM - INFO - Test time:  0.051s
03/07/2020 02:15:42 AM - INFO - Accuracy score:   0.626
03/07/2020 02:15:42 AM - INFO - 

===> Classification Report:

03/07/2020 02:15:42 AM - INFO -               precision    recall  f1-score   support

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

03/07/2020 02:15:42 AM - INFO - 

Cross validation:
03/07/2020 02:15:42 AM - INFO - 	accuracy: 5-fold cross validation: [0.69200177 0.67741935 0.6752099  0.67388422 0.67992927]
03/07/2020 02:15:42 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 67.97 (+/- 1.30)
03/07/2020 02:15:42 AM - INFO - It took 38.37780261039734 seconds
03/07/2020 02:15:42 AM - INFO - ********************************************************************************
03/07/2020 02:15:42 AM - INFO - ################################################################################
03/07/2020 02:15:42 AM - INFO - 

CURRENT CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 02:15:42 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 02:15:42 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 02:15:42 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  36.52%  |  [0.39460893 0.39019001 0.39991162 0.39019001 0.40539346]  |  39.61 (+/- 1.18)  |  4.665  |  0.2575  |
03/07/2020 02:15:42 AM - INFO - |  2  |  BERNOULLI_NB  |  45.84%  |  [0.48033584 0.46133451 0.48563853 0.47547503 0.48938992]  |  47.84 (+/- 1.95)  |  0.05716  |  0.05391  |
03/07/2020 02:15:42 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  42.95%  |  [0.4922669  0.47370747 0.46486964 0.47282369 0.48983201]  |  47.87 (+/- 2.11)  |  10.42  |  0.008561  |
03/07/2020 02:15:42 AM - INFO - |  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  75.73 (+/- 2.09)  |  0.8393  |  0.008751  |
03/07/2020 02:15:42 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  69.17%  |  [0.74458683 0.72646929 0.74193548 0.74768007 0.72413793]  |  73.70 (+/- 1.94)  |  29.69  |  0.01173  |
03/07/2020 02:15:42 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  62.43%  |  [0.68228016 0.6597437  0.67034909 0.67609368 0.66534041]  |  67.08 (+/- 1.58)  |  36.12  |  0.619  |
03/07/2020 02:15:42 AM - INFO - 

CURRENT CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 02:15:42 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 02:15:42 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 02:15:42 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  44.04%  |  [0.4719399  0.45735749 0.46531153 0.46973045 0.47126437]  |  46.71 (+/- 1.08)  |  18.42  |  0.9331  |
03/07/2020 02:15:42 AM - INFO - |  2  |  BERNOULLI_NB  |  62.56%  |  [0.69200177 0.67741935 0.6752099  0.67388422 0.67992927]  |  67.97 (+/- 1.30)  |  0.0534  |  0.05116  |
03/07/2020 02:15:42 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  45.63%  |  [0.49536014 0.49005745 0.48077773 0.47989395 0.49955791]  |  48.91 (+/- 1.56)  |  4.511  |  0.005316  |
03/07/2020 02:15:42 AM - INFO - |  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  75.73 (+/- 2.09)  |  0.8455  |  0.008558  |
03/07/2020 02:15:42 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  69.28%  |  [0.75961114 0.74016792 0.75519222 0.7565179  0.73607427]  |  74.95 (+/- 1.90)  |  33.13  |  0.01171  |
03/07/2020 02:15:42 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  63.81%  |  [0.69774635 0.6752099  0.68448962 0.69421122 0.67992927]  |  68.63 (+/- 1.70)  |  40.29  |  1.201  |
03/07/2020 02:15:42 AM - INFO - 

03/07/2020 02:15:42 AM - INFO - ################################################################################
03/07/2020 02:15:42 AM - INFO - 7)
03/07/2020 02:15:42 AM - INFO - ********************************************************************************
03/07/2020 02:15:42 AM - INFO - Classifier: COMPLEMENT_NB, Dataset: TWENTY_NEWS_GROUPS
03/07/2020 02:15:42 AM - INFO - ********************************************************************************
03/07/2020 02:15:45 AM - INFO - 

Performing grid search...

03/07/2020 02:15:45 AM - INFO - Parameters:
03/07/2020 02:15:45 AM - INFO - {'classifier__alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0], 'classifier__fit_prior': [False, True], 'classifier__norm': [False, True]}
03/07/2020 02:15:51 AM - INFO - 	Done in 6.341s
03/07/2020 02:15:51 AM - INFO - 	Best score: 0.772
03/07/2020 02:15:51 AM - INFO - 	Best parameters set:
03/07/2020 02:15:51 AM - INFO - 		classifier__alpha: 0.5
03/07/2020 02:15:51 AM - INFO - 		classifier__fit_prior: False
03/07/2020 02:15:51 AM - INFO - 		classifier__norm: False
03/07/2020 02:15:51 AM - INFO - 

USING COMPLEMENT_NB WITH DEFAULT PARAMETERS
03/07/2020 02:15:51 AM - INFO - ________________________________________________________________________________
03/07/2020 02:15:51 AM - INFO - Training: 
03/07/2020 02:15:51 AM - INFO - ComplementNB(alpha=1.0, class_prior=None, fit_prior=True, norm=False)
03/07/2020 02:15:52 AM - INFO - Train time: 0.063s
03/07/2020 02:15:52 AM - INFO - Test time:  0.010s
03/07/2020 02:15:52 AM - INFO - Accuracy score:   0.710
03/07/2020 02:15:52 AM - INFO - 

===> Classification Report:

03/07/2020 02:15:52 AM - INFO -               precision    recall  f1-score   support

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

03/07/2020 02:15:52 AM - INFO - 

Cross validation:
03/07/2020 02:15:52 AM - INFO - 	accuracy: 5-fold cross validation: [0.77110031 0.75740168 0.77993814 0.7715422  0.76038904]
03/07/2020 02:15:52 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 76.81 (+/- 1.64)
03/07/2020 02:15:52 AM - INFO - 

USING COMPLEMENT_NB WITH BEST PARAMETERS: {'alpha': 0.5, 'fit_prior': False, 'norm': False}
03/07/2020 02:15:52 AM - INFO - ________________________________________________________________________________
03/07/2020 02:15:52 AM - INFO - Training: 
03/07/2020 02:15:52 AM - INFO - ComplementNB(alpha=0.5, class_prior=None, fit_prior=False, norm=False)
03/07/2020 02:15:52 AM - INFO - Train time: 0.068s
03/07/2020 02:15:52 AM - INFO - Test time:  0.010s
03/07/2020 02:15:52 AM - INFO - Accuracy score:   0.712
03/07/2020 02:15:52 AM - INFO - 

===> Classification Report:

03/07/2020 02:15:52 AM - INFO -               precision    recall  f1-score   support

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

03/07/2020 02:15:52 AM - INFO - 

Cross validation:
03/07/2020 02:15:52 AM - INFO - 	accuracy: 5-fold cross validation: [0.776403   0.75828546 0.78126381 0.77507733 0.76967286]
03/07/2020 02:15:52 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 77.21 (+/- 1.57)
03/07/2020 02:15:52 AM - INFO - It took 9.839982032775879 seconds
03/07/2020 02:15:52 AM - INFO - ********************************************************************************
03/07/2020 02:15:52 AM - INFO - ################################################################################
03/07/2020 02:15:52 AM - INFO - 

CURRENT CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 02:15:52 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 02:15:52 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 02:15:52 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  36.52%  |  [0.39460893 0.39019001 0.39991162 0.39019001 0.40539346]  |  39.61 (+/- 1.18)  |  4.665  |  0.2575  |
03/07/2020 02:15:52 AM - INFO - |  2  |  BERNOULLI_NB  |  45.84%  |  [0.48033584 0.46133451 0.48563853 0.47547503 0.48938992]  |  47.84 (+/- 1.95)  |  0.05716  |  0.05391  |
03/07/2020 02:15:52 AM - INFO - |  3  |  COMPLEMENT_NB  |  71.04%  |  [0.77110031 0.75740168 0.77993814 0.7715422  0.76038904]  |  76.81 (+/- 1.64)  |  0.06305  |  0.01045  |
03/07/2020 02:15:52 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  42.95%  |  [0.4922669  0.47370747 0.46486964 0.47282369 0.48983201]  |  47.87 (+/- 2.11)  |  10.42  |  0.008561  |
03/07/2020 02:15:52 AM - INFO - |  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  75.73 (+/- 2.09)  |  0.8393  |  0.008751  |
03/07/2020 02:15:52 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  69.17%  |  [0.74458683 0.72646929 0.74193548 0.74768007 0.72413793]  |  73.70 (+/- 1.94)  |  29.69  |  0.01173  |
03/07/2020 02:15:52 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  62.43%  |  [0.68228016 0.6597437  0.67034909 0.67609368 0.66534041]  |  67.08 (+/- 1.58)  |  36.12  |  0.619  |
03/07/2020 02:15:52 AM - INFO - 

CURRENT CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 02:15:52 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 02:15:52 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 02:15:52 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  44.04%  |  [0.4719399  0.45735749 0.46531153 0.46973045 0.47126437]  |  46.71 (+/- 1.08)  |  18.42  |  0.9331  |
03/07/2020 02:15:52 AM - INFO - |  2  |  BERNOULLI_NB  |  62.56%  |  [0.69200177 0.67741935 0.6752099  0.67388422 0.67992927]  |  67.97 (+/- 1.30)  |  0.0534  |  0.05116  |
03/07/2020 02:15:52 AM - INFO - |  3  |  COMPLEMENT_NB  |  71.22%  |  [0.776403   0.75828546 0.78126381 0.77507733 0.76967286]  |  77.21 (+/- 1.57)  |  0.06775  |  0.0101  |
03/07/2020 02:15:52 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  45.63%  |  [0.49536014 0.49005745 0.48077773 0.47989395 0.49955791]  |  48.91 (+/- 1.56)  |  4.511  |  0.005316  |
03/07/2020 02:15:52 AM - INFO - |  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  75.73 (+/- 2.09)  |  0.8455  |  0.008558  |
03/07/2020 02:15:52 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  69.28%  |  [0.75961114 0.74016792 0.75519222 0.7565179  0.73607427]  |  74.95 (+/- 1.90)  |  33.13  |  0.01171  |
03/07/2020 02:15:52 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  63.81%  |  [0.69774635 0.6752099  0.68448962 0.69421122 0.67992927]  |  68.63 (+/- 1.70)  |  40.29  |  1.201  |
03/07/2020 02:15:52 AM - INFO - 

03/07/2020 02:15:52 AM - INFO - ################################################################################
03/07/2020 02:15:52 AM - INFO - 8)
03/07/2020 02:15:52 AM - INFO - ********************************************************************************
03/07/2020 02:15:52 AM - INFO - Classifier: MULTINOMIAL_NB, Dataset: TWENTY_NEWS_GROUPS
03/07/2020 02:15:52 AM - INFO - ********************************************************************************
03/07/2020 02:15:55 AM - INFO - 

Performing grid search...

03/07/2020 02:15:55 AM - INFO - Parameters:
03/07/2020 02:15:55 AM - INFO - {'classifier__alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0], 'classifier__fit_prior': [False, True]}
03/07/2020 02:15:58 AM - INFO - 	Done in 2.917s
03/07/2020 02:15:58 AM - INFO - 	Best score: 0.751
03/07/2020 02:15:58 AM - INFO - 	Best parameters set:
03/07/2020 02:15:58 AM - INFO - 		classifier__alpha: 0.01
03/07/2020 02:15:58 AM - INFO - 		classifier__fit_prior: True
03/07/2020 02:15:58 AM - INFO - 

USING MULTINOMIAL_NB WITH DEFAULT PARAMETERS
03/07/2020 02:15:58 AM - INFO - ________________________________________________________________________________
03/07/2020 02:15:58 AM - INFO - Training: 
03/07/2020 02:15:58 AM - INFO - MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
03/07/2020 02:15:58 AM - INFO - Train time: 0.044s
03/07/2020 02:15:58 AM - INFO - Test time:  0.010s
03/07/2020 02:15:58 AM - INFO - Accuracy score:   0.656
03/07/2020 02:15:58 AM - INFO - 

===> Classification Report:

03/07/2020 02:15:58 AM - INFO -               precision    recall  f1-score   support

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

03/07/2020 02:15:58 AM - INFO - 

Cross validation:
03/07/2020 02:15:58 AM - INFO - 	accuracy: 5-fold cross validation: [0.7105612  0.69951392 0.69863014 0.69995581 0.69142352]
03/07/2020 02:15:58 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 70.00 (+/- 1.22)
03/07/2020 02:15:58 AM - INFO - 

USING MULTINOMIAL_NB WITH BEST PARAMETERS: {'alpha': 0.01, 'fit_prior': True}
03/07/2020 02:15:58 AM - INFO - ________________________________________________________________________________
03/07/2020 02:15:58 AM - INFO - Training: 
03/07/2020 02:15:58 AM - INFO - MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True)
03/07/2020 02:15:58 AM - INFO - Train time: 0.056s
03/07/2020 02:15:58 AM - INFO - Test time:  0.010s
03/07/2020 02:15:58 AM - INFO - Accuracy score:   0.688
03/07/2020 02:15:58 AM - INFO - 

===> Classification Report:

03/07/2020 02:15:58 AM - INFO -               precision    recall  f1-score   support

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

03/07/2020 02:15:58 AM - INFO - 

Cross validation:
03/07/2020 02:15:58 AM - INFO - 	accuracy: 5-fold cross validation: [0.76049492 0.74149359 0.75519222 0.75695979 0.74270557]
03/07/2020 02:15:58 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 75.14 (+/- 1.55)
03/07/2020 02:15:58 AM - INFO - It took 6.337788820266724 seconds
03/07/2020 02:15:58 AM - INFO - ********************************************************************************
03/07/2020 02:15:58 AM - INFO - ################################################################################
03/07/2020 02:15:58 AM - INFO - 

CURRENT CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 02:15:58 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 02:15:58 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 02:15:58 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  36.52%  |  [0.39460893 0.39019001 0.39991162 0.39019001 0.40539346]  |  39.61 (+/- 1.18)  |  4.665  |  0.2575  |
03/07/2020 02:15:58 AM - INFO - |  2  |  BERNOULLI_NB  |  45.84%  |  [0.48033584 0.46133451 0.48563853 0.47547503 0.48938992]  |  47.84 (+/- 1.95)  |  0.05716  |  0.05391  |
03/07/2020 02:15:58 AM - INFO - |  3  |  COMPLEMENT_NB  |  71.04%  |  [0.77110031 0.75740168 0.77993814 0.7715422  0.76038904]  |  76.81 (+/- 1.64)  |  0.06305  |  0.01045  |
03/07/2020 02:15:58 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  42.95%  |  [0.4922669  0.47370747 0.46486964 0.47282369 0.48983201]  |  47.87 (+/- 2.11)  |  10.42  |  0.008561  |
03/07/2020 02:15:58 AM - INFO - |  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  75.73 (+/- 2.09)  |  0.8393  |  0.008751  |
03/07/2020 02:15:58 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  69.17%  |  [0.74458683 0.72646929 0.74193548 0.74768007 0.72413793]  |  73.70 (+/- 1.94)  |  29.69  |  0.01173  |
03/07/2020 02:15:58 AM - INFO - |  9  |  MULTINOMIAL_NB  |  65.64%  |  [0.7105612  0.69951392 0.69863014 0.69995581 0.69142352]  |  70.00 (+/- 1.22)  |  0.04449  |  0.01024  |
03/07/2020 02:15:58 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  62.43%  |  [0.68228016 0.6597437  0.67034909 0.67609368 0.66534041]  |  67.08 (+/- 1.58)  |  36.12  |  0.619  |
03/07/2020 02:15:58 AM - INFO - 

CURRENT CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 02:15:58 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 02:15:58 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 02:15:58 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  44.04%  |  [0.4719399  0.45735749 0.46531153 0.46973045 0.47126437]  |  46.71 (+/- 1.08)  |  18.42  |  0.9331  |
03/07/2020 02:15:58 AM - INFO - |  2  |  BERNOULLI_NB  |  62.56%  |  [0.69200177 0.67741935 0.6752099  0.67388422 0.67992927]  |  67.97 (+/- 1.30)  |  0.0534  |  0.05116  |
03/07/2020 02:15:58 AM - INFO - |  3  |  COMPLEMENT_NB  |  71.22%  |  [0.776403   0.75828546 0.78126381 0.77507733 0.76967286]  |  77.21 (+/- 1.57)  |  0.06775  |  0.0101  |
03/07/2020 02:15:58 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  45.63%  |  [0.49536014 0.49005745 0.48077773 0.47989395 0.49955791]  |  48.91 (+/- 1.56)  |  4.511  |  0.005316  |
03/07/2020 02:15:58 AM - INFO - |  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  75.73 (+/- 2.09)  |  0.8455  |  0.008558  |
03/07/2020 02:15:58 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  69.28%  |  [0.75961114 0.74016792 0.75519222 0.7565179  0.73607427]  |  74.95 (+/- 1.90)  |  33.13  |  0.01171  |
03/07/2020 02:15:58 AM - INFO - |  9  |  MULTINOMIAL_NB  |  68.79%  |  [0.76049492 0.74149359 0.75519222 0.75695979 0.74270557]  |  75.14 (+/- 1.55)  |  0.05633  |  0.01006  |
03/07/2020 02:15:58 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  63.81%  |  [0.69774635 0.6752099  0.68448962 0.69421122 0.67992927]  |  68.63 (+/- 1.70)  |  40.29  |  1.201  |
03/07/2020 02:15:58 AM - INFO - 

03/07/2020 02:15:58 AM - INFO - ################################################################################
03/07/2020 02:15:58 AM - INFO - 9)
03/07/2020 02:15:58 AM - INFO - ********************************************************************************
03/07/2020 02:15:58 AM - INFO - Classifier: NEAREST_CENTROID, Dataset: TWENTY_NEWS_GROUPS
03/07/2020 02:15:58 AM - INFO - ********************************************************************************
03/07/2020 02:16:01 AM - INFO - 

Performing grid search...

03/07/2020 02:16:01 AM - INFO - Parameters:
03/07/2020 02:16:01 AM - INFO - {'classifier__metric': ['euclidean', 'cosine']}
03/07/2020 02:16:01 AM - INFO - 	Done in 0.197s
03/07/2020 02:16:01 AM - INFO - 	Best score: 0.716
03/07/2020 02:16:01 AM - INFO - 	Best parameters set:
03/07/2020 02:16:01 AM - INFO - 		classifier__metric: 'cosine'
03/07/2020 02:16:01 AM - INFO - 

USING NEAREST_CENTROID WITH DEFAULT PARAMETERS
03/07/2020 02:16:01 AM - INFO - ________________________________________________________________________________
03/07/2020 02:16:01 AM - INFO - Training: 
03/07/2020 02:16:01 AM - INFO - NearestCentroid(metric='euclidean', shrink_threshold=None)
03/07/2020 02:16:02 AM - INFO - Train time: 0.020s
03/07/2020 02:16:02 AM - INFO - Test time:  0.014s
03/07/2020 02:16:02 AM - INFO - Accuracy score:   0.649
03/07/2020 02:16:02 AM - INFO - 

===> Classification Report:

03/07/2020 02:16:02 AM - INFO -               precision    recall  f1-score   support

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

03/07/2020 02:16:02 AM - INFO - 

Cross validation:
03/07/2020 02:16:02 AM - INFO - 	accuracy: 5-fold cross validation: [0.69774635 0.69244366 0.69818825 0.70393283 0.69849691]
03/07/2020 02:16:02 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 69.82 (+/- 0.73)
03/07/2020 02:16:02 AM - INFO - 

USING NEAREST_CENTROID WITH BEST PARAMETERS: {'metric': 'cosine'}
03/07/2020 02:16:02 AM - INFO - ________________________________________________________________________________
03/07/2020 02:16:02 AM - INFO - Training: 
03/07/2020 02:16:02 AM - INFO - NearestCentroid(metric='cosine', shrink_threshold=None)
03/07/2020 02:16:02 AM - INFO - Train time: 0.017s
03/07/2020 02:16:02 AM - INFO - Test time:  0.020s
03/07/2020 02:16:02 AM - INFO - Accuracy score:   0.667
03/07/2020 02:16:02 AM - INFO - 

===> Classification Report:

03/07/2020 02:16:02 AM - INFO -               precision    recall  f1-score   support

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

03/07/2020 02:16:02 AM - INFO - 

Cross validation:
03/07/2020 02:16:02 AM - INFO - 	accuracy: 5-fold cross validation: [0.71674768 0.71807335 0.71851525 0.71939903 0.70601238]
03/07/2020 02:16:02 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 71.57 (+/- 0.99)
03/07/2020 02:16:02 AM - INFO - It took 3.4536526203155518 seconds
03/07/2020 02:16:02 AM - INFO - ********************************************************************************
03/07/2020 02:16:02 AM - INFO - ################################################################################
03/07/2020 02:16:02 AM - INFO - 

CURRENT CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 02:16:02 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 02:16:02 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 02:16:02 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  36.52%  |  [0.39460893 0.39019001 0.39991162 0.39019001 0.40539346]  |  39.61 (+/- 1.18)  |  4.665  |  0.2575  |
03/07/2020 02:16:02 AM - INFO - |  2  |  BERNOULLI_NB  |  45.84%  |  [0.48033584 0.46133451 0.48563853 0.47547503 0.48938992]  |  47.84 (+/- 1.95)  |  0.05716  |  0.05391  |
03/07/2020 02:16:02 AM - INFO - |  3  |  COMPLEMENT_NB  |  71.04%  |  [0.77110031 0.75740168 0.77993814 0.7715422  0.76038904]  |  76.81 (+/- 1.64)  |  0.06305  |  0.01045  |
03/07/2020 02:16:02 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  42.95%  |  [0.4922669  0.47370747 0.46486964 0.47282369 0.48983201]  |  47.87 (+/- 2.11)  |  10.42  |  0.008561  |
03/07/2020 02:16:02 AM - INFO - |  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  75.73 (+/- 2.09)  |  0.8393  |  0.008751  |
03/07/2020 02:16:02 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  69.17%  |  [0.74458683 0.72646929 0.74193548 0.74768007 0.72413793]  |  73.70 (+/- 1.94)  |  29.69  |  0.01173  |
03/07/2020 02:16:02 AM - INFO - |  9  |  MULTINOMIAL_NB  |  65.64%  |  [0.7105612  0.69951392 0.69863014 0.69995581 0.69142352]  |  70.00 (+/- 1.22)  |  0.04449  |  0.01024  |
03/07/2020 02:16:02 AM - INFO - |  10  |  NEAREST_CENTROID  |  64.91%  |  [0.69774635 0.69244366 0.69818825 0.70393283 0.69849691]  |  69.82 (+/- 0.73)  |  0.02025  |  0.01422  |
03/07/2020 02:16:02 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  62.43%  |  [0.68228016 0.6597437  0.67034909 0.67609368 0.66534041]  |  67.08 (+/- 1.58)  |  36.12  |  0.619  |
03/07/2020 02:16:02 AM - INFO - 

CURRENT CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 02:16:02 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 02:16:02 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 02:16:02 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  44.04%  |  [0.4719399  0.45735749 0.46531153 0.46973045 0.47126437]  |  46.71 (+/- 1.08)  |  18.42  |  0.9331  |
03/07/2020 02:16:02 AM - INFO - |  2  |  BERNOULLI_NB  |  62.56%  |  [0.69200177 0.67741935 0.6752099  0.67388422 0.67992927]  |  67.97 (+/- 1.30)  |  0.0534  |  0.05116  |
03/07/2020 02:16:02 AM - INFO - |  3  |  COMPLEMENT_NB  |  71.22%  |  [0.776403   0.75828546 0.78126381 0.77507733 0.76967286]  |  77.21 (+/- 1.57)  |  0.06775  |  0.0101  |
03/07/2020 02:16:02 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  45.63%  |  [0.49536014 0.49005745 0.48077773 0.47989395 0.49955791]  |  48.91 (+/- 1.56)  |  4.511  |  0.005316  |
03/07/2020 02:16:02 AM - INFO - |  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  75.73 (+/- 2.09)  |  0.8455  |  0.008558  |
03/07/2020 02:16:02 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  69.28%  |  [0.75961114 0.74016792 0.75519222 0.7565179  0.73607427]  |  74.95 (+/- 1.90)  |  33.13  |  0.01171  |
03/07/2020 02:16:02 AM - INFO - |  9  |  MULTINOMIAL_NB  |  68.79%  |  [0.76049492 0.74149359 0.75519222 0.75695979 0.74270557]  |  75.14 (+/- 1.55)  |  0.05633  |  0.01006  |
03/07/2020 02:16:02 AM - INFO - |  10  |  NEAREST_CENTROID  |  66.70%  |  [0.71674768 0.71807335 0.71851525 0.71939903 0.70601238]  |  71.57 (+/- 0.99)  |  0.01741  |  0.02014  |
03/07/2020 02:16:02 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  63.81%  |  [0.69774635 0.6752099  0.68448962 0.69421122 0.67992927]  |  68.63 (+/- 1.70)  |  40.29  |  1.201  |
03/07/2020 02:16:02 AM - INFO - 

03/07/2020 02:16:02 AM - INFO - ################################################################################
03/07/2020 02:16:02 AM - INFO - 10)
03/07/2020 02:16:02 AM - INFO - ********************************************************************************
03/07/2020 02:16:02 AM - INFO - Classifier: PASSIVE_AGGRESSIVE_CLASSIFIER, Dataset: TWENTY_NEWS_GROUPS
03/07/2020 02:16:02 AM - INFO - ********************************************************************************
03/07/2020 02:16:05 AM - INFO - 

Performing grid search...

03/07/2020 02:16:05 AM - INFO - Parameters:
03/07/2020 02:16:05 AM - INFO - {'classifier__C': [0.01, 1.0], 'classifier__early_stopping': [False, True], 'classifier__tol': [0.0001, 0.001, 0.01], 'classifier__validation_fraction': [0.0001, 0.01]}
03/07/2020 02:16:55 AM - INFO - 	Done in 49.805s
03/07/2020 02:16:55 AM - INFO - 	Best score: 0.759
03/07/2020 02:16:55 AM - INFO - 	Best parameters set:
03/07/2020 02:16:55 AM - INFO - 		classifier__C: 0.01
03/07/2020 02:16:55 AM - INFO - 		classifier__early_stopping: False
03/07/2020 02:16:55 AM - INFO - 		classifier__tol: 0.0001
03/07/2020 02:16:55 AM - INFO - 		classifier__validation_fraction: 0.0001
03/07/2020 02:16:55 AM - INFO - 

USING PASSIVE_AGGRESSIVE_CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 02:16:55 AM - INFO - ________________________________________________________________________________
03/07/2020 02:16:55 AM - INFO - Training: 
03/07/2020 02:16:55 AM - INFO - PassiveAggressiveClassifier(C=1.0, average=False, class_weight=None,
                            early_stopping=False, fit_intercept=True,
                            loss='hinge', max_iter=1000, n_iter_no_change=5,
                            n_jobs=None, random_state=None, shuffle=True,
                            tol=0.001, validation_fraction=0.1, verbose=0,
                            warm_start=False)
03/07/2020 02:16:56 AM - INFO - Train time: 1.014s
03/07/2020 02:16:56 AM - INFO - Test time:  0.023s
03/07/2020 02:16:56 AM - INFO - Accuracy score:   0.683
03/07/2020 02:16:56 AM - INFO - 

===> Classification Report:

03/07/2020 02:16:56 AM - INFO -               precision    recall  f1-score   support

           0       0.51      0.45      0.48       319
           1       0.64      0.72      0.68       389
           2       0.63      0.57      0.60       394
           3       0.62      0.65      0.64       392
           4       0.70      0.67      0.68       385
           5       0.79      0.71      0.75       395
           6       0.76      0.75      0.76       390
           7       0.76      0.71      0.73       396
           8       0.49      0.80      0.60       398
           9       0.86      0.80      0.83       397
          10       0.88      0.86      0.87       399
          11       0.82      0.71      0.76       396
          12       0.66      0.58      0.61       393
          13       0.77      0.76      0.77       396
          14       0.74      0.73      0.74       394
          15       0.66      0.76      0.71       398
          16       0.57      0.66      0.61       364
          17       0.80      0.78      0.79       376
          18       0.58      0.44      0.50       310
          19       0.40      0.32      0.36       251

    accuracy                           0.68      7532
   macro avg       0.68      0.67      0.67      7532
weighted avg       0.69      0.68      0.68      7532

03/07/2020 02:16:56 AM - INFO - 

Cross validation:
03/07/2020 02:16:56 AM - INFO - 	accuracy: 5-fold cross validation: [0.76005303 0.7308882  0.7410517  0.75784357 0.73386384]
03/07/2020 02:16:56 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 74.47 (+/- 2.42)
03/07/2020 02:16:56 AM - INFO - 

USING PASSIVE_AGGRESSIVE_CLASSIFIER WITH BEST PARAMETERS: {'C': 0.01, 'early_stopping': False, 'tol': 0.0001, 'validation_fraction': 0.0001}
03/07/2020 02:16:56 AM - INFO - ________________________________________________________________________________
03/07/2020 02:16:56 AM - INFO - Training: 
03/07/2020 02:16:56 AM - INFO - PassiveAggressiveClassifier(C=0.01, average=False, class_weight=None,
                            early_stopping=False, fit_intercept=True,
                            loss='hinge', max_iter=1000, n_iter_no_change=5,
                            n_jobs=None, random_state=None, shuffle=True,
                            tol=0.0001, validation_fraction=0.0001, verbose=0,
                            warm_start=False)
03/07/2020 02:17:05 AM - INFO - Train time: 8.780s
03/07/2020 02:17:05 AM - INFO - Test time:  0.011s
03/07/2020 02:17:05 AM - INFO - Accuracy score:   0.696
03/07/2020 02:17:05 AM - INFO - 

===> Classification Report:

03/07/2020 02:17:05 AM - INFO -               precision    recall  f1-score   support

           0       0.52      0.48      0.50       319
           1       0.67      0.74      0.70       389
           2       0.62      0.60      0.61       394
           3       0.64      0.65      0.65       392
           4       0.73      0.69      0.71       385
           5       0.81      0.71      0.76       395
           6       0.78      0.77      0.77       390
           7       0.78      0.71      0.74       396
           8       0.52      0.81      0.63       398
           9       0.87      0.82      0.85       397
          10       0.89      0.88      0.89       399
          11       0.82      0.72      0.77       396
          12       0.63      0.58      0.60       393
          13       0.78      0.78      0.78       396
          14       0.74      0.76      0.75       394
          15       0.67      0.77      0.72       398
          16       0.59      0.68      0.63       364
          17       0.83      0.77      0.80       376
          18       0.56      0.46      0.51       310
          19       0.42      0.33      0.37       251

    accuracy                           0.70      7532
   macro avg       0.69      0.68      0.69      7532
weighted avg       0.70      0.70      0.70      7532

03/07/2020 02:17:05 AM - INFO - 

Cross validation:
03/07/2020 02:17:17 AM - INFO - 	accuracy: 5-fold cross validation: [0.76491383 0.74900574 0.76270437 0.76977464 0.74889478]
03/07/2020 02:17:17 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 75.91 (+/- 1.71)
03/07/2020 02:17:17 AM - INFO - It took 74.77503967285156 seconds
03/07/2020 02:17:17 AM - INFO - ********************************************************************************
03/07/2020 02:17:17 AM - INFO - ################################################################################
03/07/2020 02:17:17 AM - INFO - 

CURRENT CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 02:17:17 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 02:17:17 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 02:17:17 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  36.52%  |  [0.39460893 0.39019001 0.39991162 0.39019001 0.40539346]  |  39.61 (+/- 1.18)  |  4.665  |  0.2575  |
03/07/2020 02:17:17 AM - INFO - |  2  |  BERNOULLI_NB  |  45.84%  |  [0.48033584 0.46133451 0.48563853 0.47547503 0.48938992]  |  47.84 (+/- 1.95)  |  0.05716  |  0.05391  |
03/07/2020 02:17:17 AM - INFO - |  3  |  COMPLEMENT_NB  |  71.04%  |  [0.77110031 0.75740168 0.77993814 0.7715422  0.76038904]  |  76.81 (+/- 1.64)  |  0.06305  |  0.01045  |
03/07/2020 02:17:17 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  42.95%  |  [0.4922669  0.47370747 0.46486964 0.47282369 0.48983201]  |  47.87 (+/- 2.11)  |  10.42  |  0.008561  |
03/07/2020 02:17:17 AM - INFO - |  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  75.73 (+/- 2.09)  |  0.8393  |  0.008751  |
03/07/2020 02:17:17 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  69.17%  |  [0.74458683 0.72646929 0.74193548 0.74768007 0.72413793]  |  73.70 (+/- 1.94)  |  29.69  |  0.01173  |
03/07/2020 02:17:17 AM - INFO - |  9  |  MULTINOMIAL_NB  |  65.64%  |  [0.7105612  0.69951392 0.69863014 0.69995581 0.69142352]  |  70.00 (+/- 1.22)  |  0.04449  |  0.01024  |
03/07/2020 02:17:17 AM - INFO - |  10  |  NEAREST_CENTROID  |  64.91%  |  [0.69774635 0.69244366 0.69818825 0.70393283 0.69849691]  |  69.82 (+/- 0.73)  |  0.02025  |  0.01422  |
03/07/2020 02:17:17 AM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  68.35%  |  [0.76005303 0.7308882  0.7410517  0.75784357 0.73386384]  |  74.47 (+/- 2.42)  |  1.014  |  0.02255  |
03/07/2020 02:17:17 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  62.43%  |  [0.68228016 0.6597437  0.67034909 0.67609368 0.66534041]  |  67.08 (+/- 1.58)  |  36.12  |  0.619  |
03/07/2020 02:17:17 AM - INFO - 

CURRENT CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 02:17:17 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 02:17:17 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 02:17:17 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  44.04%  |  [0.4719399  0.45735749 0.46531153 0.46973045 0.47126437]  |  46.71 (+/- 1.08)  |  18.42  |  0.9331  |
03/07/2020 02:17:17 AM - INFO - |  2  |  BERNOULLI_NB  |  62.56%  |  [0.69200177 0.67741935 0.6752099  0.67388422 0.67992927]  |  67.97 (+/- 1.30)  |  0.0534  |  0.05116  |
03/07/2020 02:17:17 AM - INFO - |  3  |  COMPLEMENT_NB  |  71.22%  |  [0.776403   0.75828546 0.78126381 0.77507733 0.76967286]  |  77.21 (+/- 1.57)  |  0.06775  |  0.0101  |
03/07/2020 02:17:17 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  45.63%  |  [0.49536014 0.49005745 0.48077773 0.47989395 0.49955791]  |  48.91 (+/- 1.56)  |  4.511  |  0.005316  |
03/07/2020 02:17:17 AM - INFO - |  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  75.73 (+/- 2.09)  |  0.8455  |  0.008558  |
03/07/2020 02:17:17 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  69.28%  |  [0.75961114 0.74016792 0.75519222 0.7565179  0.73607427]  |  74.95 (+/- 1.90)  |  33.13  |  0.01171  |
03/07/2020 02:17:17 AM - INFO - |  9  |  MULTINOMIAL_NB  |  68.79%  |  [0.76049492 0.74149359 0.75519222 0.75695979 0.74270557]  |  75.14 (+/- 1.55)  |  0.05633  |  0.01006  |
03/07/2020 02:17:17 AM - INFO - |  10  |  NEAREST_CENTROID  |  66.70%  |  [0.71674768 0.71807335 0.71851525 0.71939903 0.70601238]  |  71.57 (+/- 0.99)  |  0.01741  |  0.02014  |
03/07/2020 02:17:17 AM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  69.64%  |  [0.76491383 0.74900574 0.76270437 0.76977464 0.74889478]  |  75.91 (+/- 1.71)  |  8.78  |  0.01103  |
03/07/2020 02:17:17 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  63.81%  |  [0.69774635 0.6752099  0.68448962 0.69421122 0.67992927]  |  68.63 (+/- 1.70)  |  40.29  |  1.201  |
03/07/2020 02:17:17 AM - INFO - 

03/07/2020 02:17:17 AM - INFO - ################################################################################
03/07/2020 02:17:17 AM - INFO - 11)
03/07/2020 02:17:17 AM - INFO - ********************************************************************************
03/07/2020 02:17:17 AM - INFO - Classifier: K_NEIGHBORS_CLASSIFIER, Dataset: TWENTY_NEWS_GROUPS
03/07/2020 02:17:17 AM - INFO - ********************************************************************************
03/07/2020 02:17:20 AM - INFO - 

Performing grid search...

03/07/2020 02:17:20 AM - INFO - Parameters:
03/07/2020 02:17:20 AM - INFO - {'classifier__leaf_size': [5, 30], 'classifier__metric': ['euclidean', 'minkowski'], 'classifier__n_neighbors': [3, 50], 'classifier__weights': ['uniform', 'distance']}
03/07/2020 02:17:29 AM - INFO - 	Done in 9.449s
03/07/2020 02:17:29 AM - INFO - 	Best score: 0.121
03/07/2020 02:17:29 AM - INFO - 	Best parameters set:
03/07/2020 02:17:29 AM - INFO - 		classifier__leaf_size: 5
03/07/2020 02:17:29 AM - INFO - 		classifier__metric: 'euclidean'
03/07/2020 02:17:29 AM - INFO - 		classifier__n_neighbors: 3
03/07/2020 02:17:29 AM - INFO - 		classifier__weights: 'distance'
03/07/2020 02:17:29 AM - INFO - 

USING K_NEIGHBORS_CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 02:17:29 AM - INFO - ________________________________________________________________________________
03/07/2020 02:17:29 AM - INFO - Training: 
03/07/2020 02:17:29 AM - INFO - KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                     weights='uniform')
03/07/2020 02:17:29 AM - INFO - Train time: 0.004s
03/07/2020 02:17:31 AM - INFO - Test time:  2.168s
03/07/2020 02:17:31 AM - INFO - Accuracy score:   0.072
03/07/2020 02:17:31 AM - INFO - 

===> Classification Report:

03/07/2020 02:17:31 AM - INFO -               precision    recall  f1-score   support

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

03/07/2020 02:17:31 AM - INFO - 

Cross validation:
03/07/2020 02:17:32 AM - INFO - 	accuracy: 5-fold cross validation: [0.06937693 0.08174989 0.07909854 0.08263367 0.08311229]
03/07/2020 02:17:32 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 7.92 (+/- 1.02)
03/07/2020 02:17:32 AM - INFO - 

USING K_NEIGHBORS_CLASSIFIER WITH BEST PARAMETERS: {'leaf_size': 5, 'metric': 'euclidean', 'n_neighbors': 3, 'weights': 'distance'}
03/07/2020 02:17:32 AM - INFO - ________________________________________________________________________________
03/07/2020 02:17:32 AM - INFO - Training: 
03/07/2020 02:17:32 AM - INFO - KNeighborsClassifier(algorithm='auto', leaf_size=5, metric='euclidean',
                     metric_params=None, n_jobs=None, n_neighbors=3, p=2,
                     weights='distance')
03/07/2020 02:17:32 AM - INFO - Train time: 0.004s
03/07/2020 02:17:34 AM - INFO - Test time:  1.755s
03/07/2020 02:17:34 AM - INFO - Accuracy score:   0.085
03/07/2020 02:17:34 AM - INFO - 

===> Classification Report:

03/07/2020 02:17:34 AM - INFO -               precision    recall  f1-score   support

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

03/07/2020 02:17:34 AM - INFO - 

Cross validation:
03/07/2020 02:17:34 AM - INFO - 	accuracy: 5-fold cross validation: [0.12417145 0.11842687 0.12063632 0.12240389 0.12113174]
03/07/2020 02:17:34 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 12.14 (+/- 0.38)
03/07/2020 02:17:34 AM - INFO - It took 17.858898639678955 seconds
03/07/2020 02:17:34 AM - INFO - ********************************************************************************
03/07/2020 02:17:34 AM - INFO - ################################################################################
03/07/2020 02:17:34 AM - INFO - 

CURRENT CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 02:17:34 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 02:17:34 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 02:17:34 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  36.52%  |  [0.39460893 0.39019001 0.39991162 0.39019001 0.40539346]  |  39.61 (+/- 1.18)  |  4.665  |  0.2575  |
03/07/2020 02:17:34 AM - INFO - |  2  |  BERNOULLI_NB  |  45.84%  |  [0.48033584 0.46133451 0.48563853 0.47547503 0.48938992]  |  47.84 (+/- 1.95)  |  0.05716  |  0.05391  |
03/07/2020 02:17:34 AM - INFO - |  3  |  COMPLEMENT_NB  |  71.04%  |  [0.77110031 0.75740168 0.77993814 0.7715422  0.76038904]  |  76.81 (+/- 1.64)  |  0.06305  |  0.01045  |
03/07/2020 02:17:34 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  42.95%  |  [0.4922669  0.47370747 0.46486964 0.47282369 0.48983201]  |  47.87 (+/- 2.11)  |  10.42  |  0.008561  |
03/07/2020 02:17:34 AM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  7.20%  |  [0.06937693 0.08174989 0.07909854 0.08263367 0.08311229]  |  7.92 (+/- 1.02)  |  0.003774  |  2.168  |
03/07/2020 02:17:34 AM - INFO - |  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  75.73 (+/- 2.09)  |  0.8393  |  0.008751  |
03/07/2020 02:17:34 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  69.17%  |  [0.74458683 0.72646929 0.74193548 0.74768007 0.72413793]  |  73.70 (+/- 1.94)  |  29.69  |  0.01173  |
03/07/2020 02:17:34 AM - INFO - |  9  |  MULTINOMIAL_NB  |  65.64%  |  [0.7105612  0.69951392 0.69863014 0.69995581 0.69142352]  |  70.00 (+/- 1.22)  |  0.04449  |  0.01024  |
03/07/2020 02:17:34 AM - INFO - |  10  |  NEAREST_CENTROID  |  64.91%  |  [0.69774635 0.69244366 0.69818825 0.70393283 0.69849691]  |  69.82 (+/- 0.73)  |  0.02025  |  0.01422  |
03/07/2020 02:17:34 AM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  68.35%  |  [0.76005303 0.7308882  0.7410517  0.75784357 0.73386384]  |  74.47 (+/- 2.42)  |  1.014  |  0.02255  |
03/07/2020 02:17:34 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  62.43%  |  [0.68228016 0.6597437  0.67034909 0.67609368 0.66534041]  |  67.08 (+/- 1.58)  |  36.12  |  0.619  |
03/07/2020 02:17:34 AM - INFO - 

CURRENT CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 02:17:34 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 02:17:34 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 02:17:34 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  44.04%  |  [0.4719399  0.45735749 0.46531153 0.46973045 0.47126437]  |  46.71 (+/- 1.08)  |  18.42  |  0.9331  |
03/07/2020 02:17:34 AM - INFO - |  2  |  BERNOULLI_NB  |  62.56%  |  [0.69200177 0.67741935 0.6752099  0.67388422 0.67992927]  |  67.97 (+/- 1.30)  |  0.0534  |  0.05116  |
03/07/2020 02:17:34 AM - INFO - |  3  |  COMPLEMENT_NB  |  71.22%  |  [0.776403   0.75828546 0.78126381 0.77507733 0.76967286]  |  77.21 (+/- 1.57)  |  0.06775  |  0.0101  |
03/07/2020 02:17:34 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  45.63%  |  [0.49536014 0.49005745 0.48077773 0.47989395 0.49955791]  |  48.91 (+/- 1.56)  |  4.511  |  0.005316  |
03/07/2020 02:17:34 AM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  8.48%  |  [0.12417145 0.11842687 0.12063632 0.12240389 0.12113174]  |  12.14 (+/- 0.38)  |  0.004078  |  1.755  |
03/07/2020 02:17:34 AM - INFO - |  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  75.73 (+/- 2.09)  |  0.8455  |  0.008558  |
03/07/2020 02:17:34 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  69.28%  |  [0.75961114 0.74016792 0.75519222 0.7565179  0.73607427]  |  74.95 (+/- 1.90)  |  33.13  |  0.01171  |
03/07/2020 02:17:34 AM - INFO - |  9  |  MULTINOMIAL_NB  |  68.79%  |  [0.76049492 0.74149359 0.75519222 0.75695979 0.74270557]  |  75.14 (+/- 1.55)  |  0.05633  |  0.01006  |
03/07/2020 02:17:34 AM - INFO - |  10  |  NEAREST_CENTROID  |  66.70%  |  [0.71674768 0.71807335 0.71851525 0.71939903 0.70601238]  |  71.57 (+/- 0.99)  |  0.01741  |  0.02014  |
03/07/2020 02:17:34 AM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  69.64%  |  [0.76491383 0.74900574 0.76270437 0.76977464 0.74889478]  |  75.91 (+/- 1.71)  |  8.78  |  0.01103  |
03/07/2020 02:17:34 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  63.81%  |  [0.69774635 0.6752099  0.68448962 0.69421122 0.67992927]  |  68.63 (+/- 1.70)  |  40.29  |  1.201  |
03/07/2020 02:17:34 AM - INFO - 

03/07/2020 02:17:34 AM - INFO - ################################################################################
03/07/2020 02:17:34 AM - INFO - 12)
03/07/2020 02:17:34 AM - INFO - ********************************************************************************
03/07/2020 02:17:34 AM - INFO - Classifier: PERCEPTRON, Dataset: TWENTY_NEWS_GROUPS
03/07/2020 02:17:34 AM - INFO - ********************************************************************************
03/07/2020 02:17:37 AM - INFO - 

Performing grid search...

03/07/2020 02:17:37 AM - INFO - Parameters:
03/07/2020 02:17:37 AM - INFO - {'classifier__early_stopping': [True], 'classifier__max_iter': [100], 'classifier__n_iter_no_change': [3, 15], 'classifier__penalty': ['l2'], 'classifier__tol': [0.0001, 0.1], 'classifier__validation_fraction': [0.0001, 0.01]}
03/07/2020 02:17:44 AM - INFO - 	Done in 6.198s
03/07/2020 02:17:44 AM - INFO - 	Best score: 0.608
03/07/2020 02:17:44 AM - INFO - 	Best parameters set:
03/07/2020 02:17:44 AM - INFO - 		classifier__early_stopping: True
03/07/2020 02:17:44 AM - INFO - 		classifier__max_iter: 100
03/07/2020 02:17:44 AM - INFO - 		classifier__n_iter_no_change: 3
03/07/2020 02:17:44 AM - INFO - 		classifier__penalty: 'l2'
03/07/2020 02:17:44 AM - INFO - 		classifier__tol: 0.0001
03/07/2020 02:17:44 AM - INFO - 		classifier__validation_fraction: 0.01
03/07/2020 02:17:44 AM - INFO - 

USING PERCEPTRON WITH DEFAULT PARAMETERS
03/07/2020 02:17:44 AM - INFO - ________________________________________________________________________________
03/07/2020 02:17:44 AM - INFO - Training: 
03/07/2020 02:17:44 AM - INFO - Perceptron(alpha=0.0001, class_weight=None, early_stopping=False, eta0=1.0,
           fit_intercept=True, max_iter=1000, n_iter_no_change=5, n_jobs=None,
           penalty=None, random_state=0, shuffle=True, tol=0.001,
           validation_fraction=0.1, verbose=0, warm_start=False)
03/07/2020 02:17:44 AM - INFO - Train time: 0.535s
03/07/2020 02:17:44 AM - INFO - Test time:  0.025s
03/07/2020 02:17:44 AM - INFO - Accuracy score:   0.633
03/07/2020 02:17:44 AM - INFO - 

===> Classification Report:

03/07/2020 02:17:44 AM - INFO -               precision    recall  f1-score   support

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

03/07/2020 02:17:44 AM - INFO - 

Cross validation:
03/07/2020 02:17:45 AM - INFO - 	accuracy: 5-fold cross validation: [0.69818825 0.67565179 0.6955369  0.70349094 0.68302387]
03/07/2020 02:17:45 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 69.12 (+/- 2.05)
03/07/2020 02:17:45 AM - INFO - 

USING PERCEPTRON WITH BEST PARAMETERS: {'early_stopping': True, 'max_iter': 100, 'n_iter_no_change': 3, 'penalty': 'l2', 'tol': 0.0001, 'validation_fraction': 0.01}
03/07/2020 02:17:45 AM - INFO - ________________________________________________________________________________
03/07/2020 02:17:45 AM - INFO - Training: 
03/07/2020 02:17:45 AM - INFO - Perceptron(alpha=0.0001, class_weight=None, early_stopping=True, eta0=1.0,
           fit_intercept=True, max_iter=100, n_iter_no_change=3, n_jobs=None,
           penalty='l2', random_state=0, shuffle=True, tol=0.0001,
           validation_fraction=0.01, verbose=0, warm_start=False)
03/07/2020 02:17:45 AM - INFO - Train time: 0.788s
03/07/2020 02:17:45 AM - INFO - Test time:  0.023s
03/07/2020 02:17:45 AM - INFO - Accuracy score:   0.539
03/07/2020 02:17:45 AM - INFO - 

===> Classification Report:

03/07/2020 02:17:45 AM - INFO -               precision    recall  f1-score   support

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

03/07/2020 02:17:45 AM - INFO - 

Cross validation:
03/07/2020 02:17:46 AM - INFO - 	accuracy: 5-fold cross validation: [0.63146266 0.61025188 0.58948299 0.60450729 0.60477454]
03/07/2020 02:17:46 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 60.81 (+/- 2.71)
03/07/2020 02:17:46 AM - INFO - It took 11.719040393829346 seconds
03/07/2020 02:17:46 AM - INFO - ********************************************************************************
03/07/2020 02:17:46 AM - INFO - ################################################################################
03/07/2020 02:17:46 AM - INFO - 

CURRENT CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 02:17:46 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 02:17:46 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 02:17:46 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  36.52%  |  [0.39460893 0.39019001 0.39991162 0.39019001 0.40539346]  |  39.61 (+/- 1.18)  |  4.665  |  0.2575  |
03/07/2020 02:17:46 AM - INFO - |  2  |  BERNOULLI_NB  |  45.84%  |  [0.48033584 0.46133451 0.48563853 0.47547503 0.48938992]  |  47.84 (+/- 1.95)  |  0.05716  |  0.05391  |
03/07/2020 02:17:46 AM - INFO - |  3  |  COMPLEMENT_NB  |  71.04%  |  [0.77110031 0.75740168 0.77993814 0.7715422  0.76038904]  |  76.81 (+/- 1.64)  |  0.06305  |  0.01045  |
03/07/2020 02:17:46 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  42.95%  |  [0.4922669  0.47370747 0.46486964 0.47282369 0.48983201]  |  47.87 (+/- 2.11)  |  10.42  |  0.008561  |
03/07/2020 02:17:46 AM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  7.20%  |  [0.06937693 0.08174989 0.07909854 0.08263367 0.08311229]  |  7.92 (+/- 1.02)  |  0.003774  |  2.168  |
03/07/2020 02:17:46 AM - INFO - |  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  75.73 (+/- 2.09)  |  0.8393  |  0.008751  |
03/07/2020 02:17:46 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  69.17%  |  [0.74458683 0.72646929 0.74193548 0.74768007 0.72413793]  |  73.70 (+/- 1.94)  |  29.69  |  0.01173  |
03/07/2020 02:17:46 AM - INFO - |  9  |  MULTINOMIAL_NB  |  65.64%  |  [0.7105612  0.69951392 0.69863014 0.69995581 0.69142352]  |  70.00 (+/- 1.22)  |  0.04449  |  0.01024  |
03/07/2020 02:17:46 AM - INFO - |  10  |  NEAREST_CENTROID  |  64.91%  |  [0.69774635 0.69244366 0.69818825 0.70393283 0.69849691]  |  69.82 (+/- 0.73)  |  0.02025  |  0.01422  |
03/07/2020 02:17:46 AM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  68.35%  |  [0.76005303 0.7308882  0.7410517  0.75784357 0.73386384]  |  74.47 (+/- 2.42)  |  1.014  |  0.02255  |
03/07/2020 02:17:46 AM - INFO - |  12  |  PERCEPTRON  |  63.34%  |  [0.69818825 0.67565179 0.6955369  0.70349094 0.68302387]  |  69.12 (+/- 2.05)  |  0.535  |  0.02502  |
03/07/2020 02:17:46 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  62.43%  |  [0.68228016 0.6597437  0.67034909 0.67609368 0.66534041]  |  67.08 (+/- 1.58)  |  36.12  |  0.619  |
03/07/2020 02:17:46 AM - INFO - 

CURRENT CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 02:17:46 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 02:17:46 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 02:17:46 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  44.04%  |  [0.4719399  0.45735749 0.46531153 0.46973045 0.47126437]  |  46.71 (+/- 1.08)  |  18.42  |  0.9331  |
03/07/2020 02:17:46 AM - INFO - |  2  |  BERNOULLI_NB  |  62.56%  |  [0.69200177 0.67741935 0.6752099  0.67388422 0.67992927]  |  67.97 (+/- 1.30)  |  0.0534  |  0.05116  |
03/07/2020 02:17:46 AM - INFO - |  3  |  COMPLEMENT_NB  |  71.22%  |  [0.776403   0.75828546 0.78126381 0.77507733 0.76967286]  |  77.21 (+/- 1.57)  |  0.06775  |  0.0101  |
03/07/2020 02:17:46 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  45.63%  |  [0.49536014 0.49005745 0.48077773 0.47989395 0.49955791]  |  48.91 (+/- 1.56)  |  4.511  |  0.005316  |
03/07/2020 02:17:46 AM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  8.48%  |  [0.12417145 0.11842687 0.12063632 0.12240389 0.12113174]  |  12.14 (+/- 0.38)  |  0.004078  |  1.755  |
03/07/2020 02:17:46 AM - INFO - |  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  75.73 (+/- 2.09)  |  0.8455  |  0.008558  |
03/07/2020 02:17:46 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  69.28%  |  [0.75961114 0.74016792 0.75519222 0.7565179  0.73607427]  |  74.95 (+/- 1.90)  |  33.13  |  0.01171  |
03/07/2020 02:17:46 AM - INFO - |  9  |  MULTINOMIAL_NB  |  68.79%  |  [0.76049492 0.74149359 0.75519222 0.75695979 0.74270557]  |  75.14 (+/- 1.55)  |  0.05633  |  0.01006  |
03/07/2020 02:17:46 AM - INFO - |  10  |  NEAREST_CENTROID  |  66.70%  |  [0.71674768 0.71807335 0.71851525 0.71939903 0.70601238]  |  71.57 (+/- 0.99)  |  0.01741  |  0.02014  |
03/07/2020 02:17:46 AM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  69.64%  |  [0.76491383 0.74900574 0.76270437 0.76977464 0.74889478]  |  75.91 (+/- 1.71)  |  8.78  |  0.01103  |
03/07/2020 02:17:46 AM - INFO - |  12  |  PERCEPTRON  |  53.86%  |  [0.63146266 0.61025188 0.58948299 0.60450729 0.60477454]  |  60.81 (+/- 2.71)  |  0.7884  |  0.02289  |
03/07/2020 02:17:46 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  63.81%  |  [0.69774635 0.6752099  0.68448962 0.69421122 0.67992927]  |  68.63 (+/- 1.70)  |  40.29  |  1.201  |
03/07/2020 02:17:46 AM - INFO - 

03/07/2020 02:17:46 AM - INFO - ################################################################################
03/07/2020 02:17:46 AM - INFO - 13)
03/07/2020 02:17:46 AM - INFO - ********************************************************************************
03/07/2020 02:17:46 AM - INFO - Classifier: RIDGE_CLASSIFIER, Dataset: TWENTY_NEWS_GROUPS
03/07/2020 02:17:46 AM - INFO - ********************************************************************************
03/07/2020 02:17:49 AM - INFO - 

Performing grid search...

03/07/2020 02:17:49 AM - INFO - Parameters:
03/07/2020 02:17:49 AM - INFO - {'classifier__alpha': [0.5, 1.0], 'classifier__tol': [0.0001, 0.001]}
03/07/2020 02:18:02 AM - INFO - 	Done in 12.406s
03/07/2020 02:18:02 AM - INFO - 	Best score: 0.764
03/07/2020 02:18:02 AM - INFO - 	Best parameters set:
03/07/2020 02:18:02 AM - INFO - 		classifier__alpha: 0.5
03/07/2020 02:18:02 AM - INFO - 		classifier__tol: 0.001
03/07/2020 02:18:02 AM - INFO - 

USING RIDGE_CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 02:18:02 AM - INFO - ________________________________________________________________________________
03/07/2020 02:18:02 AM - INFO - Training: 
03/07/2020 02:18:02 AM - INFO - RidgeClassifier(alpha=1.0, class_weight=None, copy_X=True, fit_intercept=True,
                max_iter=None, normalize=False, random_state=None,
                solver='auto', tol=0.001)
03/07/2020 02:18:04 AM - INFO - Train time: 2.184s
03/07/2020 02:18:04 AM - INFO - Test time:  0.023s
03/07/2020 02:18:04 AM - INFO - Accuracy score:   0.707
03/07/2020 02:18:04 AM - INFO - 

===> Classification Report:

03/07/2020 02:18:04 AM - INFO -               precision    recall  f1-score   support

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

03/07/2020 02:18:04 AM - INFO - 

Cross validation:
03/07/2020 02:18:05 AM - INFO - 	accuracy: 5-fold cross validation: [0.76623951 0.74900574 0.76889085 0.77330977 0.75464191]
03/07/2020 02:18:05 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 76.24 (+/- 1.82)
03/07/2020 02:18:05 AM - INFO - 

USING RIDGE_CLASSIFIER WITH BEST PARAMETERS: {'alpha': 0.5, 'tol': 0.001}
03/07/2020 02:18:05 AM - INFO - ________________________________________________________________________________
03/07/2020 02:18:05 AM - INFO - Training: 
03/07/2020 02:18:05 AM - INFO - RidgeClassifier(alpha=0.5, class_weight=None, copy_X=True, fit_intercept=True,
                max_iter=None, normalize=False, random_state=None,
                solver='auto', tol=0.001)
03/07/2020 02:18:08 AM - INFO - Train time: 2.765s
03/07/2020 02:18:08 AM - INFO - Test time:  0.023s
03/07/2020 02:18:08 AM - INFO - Accuracy score:   0.700
03/07/2020 02:18:08 AM - INFO - 

===> Classification Report:

03/07/2020 02:18:08 AM - INFO -               precision    recall  f1-score   support

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

03/07/2020 02:18:08 AM - INFO - 

Cross validation:
03/07/2020 02:18:10 AM - INFO - 	accuracy: 5-fold cross validation: [0.76756518 0.74502872 0.76800707 0.776403   0.76083112]
03/07/2020 02:18:10 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 76.36 (+/- 2.10)
03/07/2020 02:18:10 AM - INFO - It took 23.852034091949463 seconds
03/07/2020 02:18:10 AM - INFO - ********************************************************************************
03/07/2020 02:18:10 AM - INFO - ################################################################################
03/07/2020 02:18:10 AM - INFO - 

CURRENT CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 02:18:10 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 02:18:10 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 02:18:10 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  36.52%  |  [0.39460893 0.39019001 0.39991162 0.39019001 0.40539346]  |  39.61 (+/- 1.18)  |  4.665  |  0.2575  |
03/07/2020 02:18:10 AM - INFO - |  2  |  BERNOULLI_NB  |  45.84%  |  [0.48033584 0.46133451 0.48563853 0.47547503 0.48938992]  |  47.84 (+/- 1.95)  |  0.05716  |  0.05391  |
03/07/2020 02:18:10 AM - INFO - |  3  |  COMPLEMENT_NB  |  71.04%  |  [0.77110031 0.75740168 0.77993814 0.7715422  0.76038904]  |  76.81 (+/- 1.64)  |  0.06305  |  0.01045  |
03/07/2020 02:18:10 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  42.95%  |  [0.4922669  0.47370747 0.46486964 0.47282369 0.48983201]  |  47.87 (+/- 2.11)  |  10.42  |  0.008561  |
03/07/2020 02:18:10 AM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  7.20%  |  [0.06937693 0.08174989 0.07909854 0.08263367 0.08311229]  |  7.92 (+/- 1.02)  |  0.003774  |  2.168  |
03/07/2020 02:18:10 AM - INFO - |  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  75.73 (+/- 2.09)  |  0.8393  |  0.008751  |
03/07/2020 02:18:10 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  69.17%  |  [0.74458683 0.72646929 0.74193548 0.74768007 0.72413793]  |  73.70 (+/- 1.94)  |  29.69  |  0.01173  |
03/07/2020 02:18:10 AM - INFO - |  9  |  MULTINOMIAL_NB  |  65.64%  |  [0.7105612  0.69951392 0.69863014 0.69995581 0.69142352]  |  70.00 (+/- 1.22)  |  0.04449  |  0.01024  |
03/07/2020 02:18:10 AM - INFO - |  10  |  NEAREST_CENTROID  |  64.91%  |  [0.69774635 0.69244366 0.69818825 0.70393283 0.69849691]  |  69.82 (+/- 0.73)  |  0.02025  |  0.01422  |
03/07/2020 02:18:10 AM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  68.35%  |  [0.76005303 0.7308882  0.7410517  0.75784357 0.73386384]  |  74.47 (+/- 2.42)  |  1.014  |  0.02255  |
03/07/2020 02:18:10 AM - INFO - |  12  |  PERCEPTRON  |  63.34%  |  [0.69818825 0.67565179 0.6955369  0.70349094 0.68302387]  |  69.12 (+/- 2.05)  |  0.535  |  0.02502  |
03/07/2020 02:18:10 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  62.43%  |  [0.68228016 0.6597437  0.67034909 0.67609368 0.66534041]  |  67.08 (+/- 1.58)  |  36.12  |  0.619  |
03/07/2020 02:18:10 AM - INFO - |  14  |  RIDGE_CLASSIFIER  |  70.67%  |  [0.76623951 0.74900574 0.76889085 0.77330977 0.75464191]  |  76.24 (+/- 1.82)  |  2.184  |  0.02251  |
03/07/2020 02:18:10 AM - INFO - 

CURRENT CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 02:18:10 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 02:18:10 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 02:18:10 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  44.04%  |  [0.4719399  0.45735749 0.46531153 0.46973045 0.47126437]  |  46.71 (+/- 1.08)  |  18.42  |  0.9331  |
03/07/2020 02:18:10 AM - INFO - |  2  |  BERNOULLI_NB  |  62.56%  |  [0.69200177 0.67741935 0.6752099  0.67388422 0.67992927]  |  67.97 (+/- 1.30)  |  0.0534  |  0.05116  |
03/07/2020 02:18:10 AM - INFO - |  3  |  COMPLEMENT_NB  |  71.22%  |  [0.776403   0.75828546 0.78126381 0.77507733 0.76967286]  |  77.21 (+/- 1.57)  |  0.06775  |  0.0101  |
03/07/2020 02:18:10 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  45.63%  |  [0.49536014 0.49005745 0.48077773 0.47989395 0.49955791]  |  48.91 (+/- 1.56)  |  4.511  |  0.005316  |
03/07/2020 02:18:10 AM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  8.48%  |  [0.12417145 0.11842687 0.12063632 0.12240389 0.12113174]  |  12.14 (+/- 0.38)  |  0.004078  |  1.755  |
03/07/2020 02:18:10 AM - INFO - |  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  75.73 (+/- 2.09)  |  0.8455  |  0.008558  |
03/07/2020 02:18:10 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  69.28%  |  [0.75961114 0.74016792 0.75519222 0.7565179  0.73607427]  |  74.95 (+/- 1.90)  |  33.13  |  0.01171  |
03/07/2020 02:18:10 AM - INFO - |  9  |  MULTINOMIAL_NB  |  68.79%  |  [0.76049492 0.74149359 0.75519222 0.75695979 0.74270557]  |  75.14 (+/- 1.55)  |  0.05633  |  0.01006  |
03/07/2020 02:18:10 AM - INFO - |  10  |  NEAREST_CENTROID  |  66.70%  |  [0.71674768 0.71807335 0.71851525 0.71939903 0.70601238]  |  71.57 (+/- 0.99)  |  0.01741  |  0.02014  |
03/07/2020 02:18:10 AM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  69.64%  |  [0.76491383 0.74900574 0.76270437 0.76977464 0.74889478]  |  75.91 (+/- 1.71)  |  8.78  |  0.01103  |
03/07/2020 02:18:10 AM - INFO - |  12  |  PERCEPTRON  |  53.86%  |  [0.63146266 0.61025188 0.58948299 0.60450729 0.60477454]  |  60.81 (+/- 2.71)  |  0.7884  |  0.02289  |
03/07/2020 02:18:10 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  63.81%  |  [0.69774635 0.6752099  0.68448962 0.69421122 0.67992927]  |  68.63 (+/- 1.70)  |  40.29  |  1.201  |
03/07/2020 02:18:10 AM - INFO - |  14  |  RIDGE_CLASSIFIER  |  70.02%  |  [0.76756518 0.74502872 0.76800707 0.776403   0.76083112]  |  76.36 (+/- 2.10)  |  2.765  |  0.02252  |
03/07/2020 02:18:10 AM - INFO - 

03/07/2020 02:18:10 AM - INFO - ################################################################################
03/07/2020 02:18:10 AM - INFO - 14)
03/07/2020 02:18:10 AM - INFO - ********************************************************************************
03/07/2020 02:18:10 AM - INFO - Classifier: GRADIENT_BOOSTING_CLASSIFIER, Dataset: TWENTY_NEWS_GROUPS
03/07/2020 02:18:10 AM - INFO - ********************************************************************************
03/07/2020 02:18:13 AM - INFO - 

Performing grid search...

03/07/2020 02:18:13 AM - INFO - Parameters:
03/07/2020 02:18:13 AM - INFO - {'classifier__learning_rate': [0.01, 0.1], 'classifier__n_estimators': [100, 200]}
03/07/2020 04:16:58 AM - INFO - 	Done in 7124.909s
03/07/2020 04:16:58 AM - INFO - 	Best score: 0.647
03/07/2020 04:16:58 AM - INFO - 	Best parameters set:
03/07/2020 04:16:58 AM - INFO - 		classifier__learning_rate: 0.1
03/07/2020 04:16:58 AM - INFO - 		classifier__n_estimators: 200
03/07/2020 04:16:58 AM - INFO - 

USING GRADIENT_BOOSTING_CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 04:16:58 AM - INFO - ________________________________________________________________________________
03/07/2020 04:16:58 AM - INFO - Training: 
03/07/2020 04:16:58 AM - INFO - GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.1, loss='deviance', max_depth=3,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_iter_no_change=None, presort='deprecated',
                           random_state=None, subsample=1.0, tol=0.0001,
                           validation_fraction=0.1, verbose=0,
                           warm_start=False)
03/07/2020 04:22:32 AM - INFO - Train time: 334.025s
03/07/2020 04:22:32 AM - INFO - Test time:  0.175s
03/07/2020 04:22:32 AM - INFO - Accuracy score:   0.593
03/07/2020 04:22:32 AM - INFO - 

===> Classification Report:

03/07/2020 04:22:32 AM - INFO -               precision    recall  f1-score   support

           0       0.46      0.35      0.40       319
           1       0.59      0.65      0.62       389
           2       0.60      0.57      0.59       394
           3       0.57      0.58      0.57       392
           4       0.70      0.62      0.66       385
           5       0.76      0.59      0.66       395
           6       0.70      0.68      0.69       390
           7       0.71      0.58      0.64       396
           8       0.79      0.63      0.70       398
           9       0.80      0.70      0.75       397
          10       0.81      0.77      0.79       399
          11       0.77      0.62      0.69       396
          12       0.18      0.61      0.28       393
          13       0.77      0.60      0.68       396
          14       0.72      0.60      0.65       394
          15       0.63      0.70      0.66       398
          16       0.59      0.57      0.58       364
          17       0.83      0.62      0.71       376
          18       0.57      0.38      0.46       310
          19       0.32      0.21      0.25       251

    accuracy                           0.59      7532
   macro avg       0.64      0.58      0.60      7532
weighted avg       0.65      0.59      0.61      7532

03/07/2020 04:22:32 AM - INFO - 

Cross validation:
03/07/2020 04:42:18 AM - INFO - 	accuracy: 5-fold cross validation: [0.65178966 0.6195316  0.63013699 0.65090588 0.6357206 ]
03/07/2020 04:42:18 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 63.76 (+/- 2.47)
03/07/2020 04:42:18 AM - INFO - 

USING GRADIENT_BOOSTING_CLASSIFIER WITH BEST PARAMETERS: {'learning_rate': 0.1, 'n_estimators': 200}
03/07/2020 04:42:18 AM - INFO - ________________________________________________________________________________
03/07/2020 04:42:18 AM - INFO - Training: 
03/07/2020 04:42:18 AM - INFO - GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.1, loss='deviance', max_depth=3,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=200,
                           n_iter_no_change=None, presort='deprecated',
                           random_state=None, subsample=1.0, tol=0.0001,
                           validation_fraction=0.1, verbose=0,
                           warm_start=False)
03/07/2020 04:53:17 AM - INFO - Train time: 658.662s
03/07/2020 04:53:17 AM - INFO - Test time:  0.342s
03/07/2020 04:53:17 AM - INFO - Accuracy score:   0.597
03/07/2020 04:53:17 AM - INFO - 

===> Classification Report:

03/07/2020 04:53:17 AM - INFO -               precision    recall  f1-score   support

           0       0.43      0.36      0.39       319
           1       0.58      0.64      0.61       389
           2       0.59      0.56      0.58       394
           3       0.55      0.57      0.56       392
           4       0.68      0.63      0.65       385
           5       0.74      0.59      0.66       395
           6       0.70      0.68      0.69       390
           7       0.71      0.60      0.65       396
           8       0.76      0.65      0.70       398
           9       0.79      0.72      0.75       397
          10       0.80      0.77      0.78       399
          11       0.78      0.63      0.70       396
          12       0.20      0.58      0.30       393
          13       0.75      0.62      0.68       396
          14       0.71      0.59      0.64       394
          15       0.63      0.68      0.65       398
          16       0.57      0.60      0.59       364
          17       0.84      0.63      0.72       376
          18       0.55      0.38      0.45       310
          19       0.31      0.22      0.25       251

    accuracy                           0.60      7532
   macro avg       0.63      0.59      0.60      7532
weighted avg       0.64      0.60      0.61      7532

03/07/2020 04:53:17 AM - INFO - 

Cross validation:
03/07/2020 05:32:42 AM - INFO - 	accuracy: 5-fold cross validation: [0.65930181 0.62615996 0.64295183 0.65709236 0.65119363]
03/07/2020 05:32:42 AM - INFO - 	test accuracy: 5-fold cross validation accuracy: 64.73 (+/- 2.40)
03/07/2020 05:32:42 AM - INFO - It took 11671.681336641312 seconds
03/07/2020 05:32:42 AM - INFO - ********************************************************************************
03/07/2020 05:32:42 AM - INFO - ################################################################################
03/07/2020 05:32:42 AM - INFO - 

CURRENT CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 05:32:42 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 05:32:42 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 05:32:42 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  36.52%  |  [0.39460893 0.39019001 0.39991162 0.39019001 0.40539346]  |  39.61 (+/- 1.18)  |  4.665  |  0.2575  |
03/07/2020 05:32:42 AM - INFO - |  2  |  BERNOULLI_NB  |  45.84%  |  [0.48033584 0.46133451 0.48563853 0.47547503 0.48938992]  |  47.84 (+/- 1.95)  |  0.05716  |  0.05391  |
03/07/2020 05:32:42 AM - INFO - |  3  |  COMPLEMENT_NB  |  71.04%  |  [0.77110031 0.75740168 0.77993814 0.7715422  0.76038904]  |  76.81 (+/- 1.64)  |  0.06305  |  0.01045  |
03/07/2020 05:32:42 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  42.95%  |  [0.4922669  0.47370747 0.46486964 0.47282369 0.48983201]  |  47.87 (+/- 2.11)  |  10.42  |  0.008561  |
03/07/2020 05:32:42 AM - INFO - |  5  |  GRADIENT_BOOSTING_CLASSIFIER  |  59.25%  |  [0.65178966 0.6195316  0.63013699 0.65090588 0.6357206 ]  |  63.76 (+/- 2.47)  |  334.0  |  0.1755  |
03/07/2020 05:32:42 AM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  7.20%  |  [0.06937693 0.08174989 0.07909854 0.08263367 0.08311229]  |  7.92 (+/- 1.02)  |  0.003774  |  2.168  |
03/07/2020 05:32:42 AM - INFO - |  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  75.73 (+/- 2.09)  |  0.8393  |  0.008751  |
03/07/2020 05:32:42 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  69.17%  |  [0.74458683 0.72646929 0.74193548 0.74768007 0.72413793]  |  73.70 (+/- 1.94)  |  29.69  |  0.01173  |
03/07/2020 05:32:42 AM - INFO - |  9  |  MULTINOMIAL_NB  |  65.64%  |  [0.7105612  0.69951392 0.69863014 0.69995581 0.69142352]  |  70.00 (+/- 1.22)  |  0.04449  |  0.01024  |
03/07/2020 05:32:42 AM - INFO - |  10  |  NEAREST_CENTROID  |  64.91%  |  [0.69774635 0.69244366 0.69818825 0.70393283 0.69849691]  |  69.82 (+/- 0.73)  |  0.02025  |  0.01422  |
03/07/2020 05:32:42 AM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  68.35%  |  [0.76005303 0.7308882  0.7410517  0.75784357 0.73386384]  |  74.47 (+/- 2.42)  |  1.014  |  0.02255  |
03/07/2020 05:32:42 AM - INFO - |  12  |  PERCEPTRON  |  63.34%  |  [0.69818825 0.67565179 0.6955369  0.70349094 0.68302387]  |  69.12 (+/- 2.05)  |  0.535  |  0.02502  |
03/07/2020 05:32:42 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  62.43%  |  [0.68228016 0.6597437  0.67034909 0.67609368 0.66534041]  |  67.08 (+/- 1.58)  |  36.12  |  0.619  |
03/07/2020 05:32:42 AM - INFO - |  14  |  RIDGE_CLASSIFIER  |  70.67%  |  [0.76623951 0.74900574 0.76889085 0.77330977 0.75464191]  |  76.24 (+/- 1.82)  |  2.184  |  0.02251  |
03/07/2020 05:32:42 AM - INFO - 

CURRENT CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 05:32:42 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 05:32:42 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 05:32:42 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  44.04%  |  [0.4719399  0.45735749 0.46531153 0.46973045 0.47126437]  |  46.71 (+/- 1.08)  |  18.42  |  0.9331  |
03/07/2020 05:32:42 AM - INFO - |  2  |  BERNOULLI_NB  |  62.56%  |  [0.69200177 0.67741935 0.6752099  0.67388422 0.67992927]  |  67.97 (+/- 1.30)  |  0.0534  |  0.05116  |
03/07/2020 05:32:42 AM - INFO - |  3  |  COMPLEMENT_NB  |  71.22%  |  [0.776403   0.75828546 0.78126381 0.77507733 0.76967286]  |  77.21 (+/- 1.57)  |  0.06775  |  0.0101  |
03/07/2020 05:32:42 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  45.63%  |  [0.49536014 0.49005745 0.48077773 0.47989395 0.49955791]  |  48.91 (+/- 1.56)  |  4.511  |  0.005316  |
03/07/2020 05:32:42 AM - INFO - |  5  |  GRADIENT_BOOSTING_CLASSIFIER  |  59.65%  |  [0.65930181 0.62615996 0.64295183 0.65709236 0.65119363]  |  64.73 (+/- 2.40)  |  658.7  |  0.3424  |
03/07/2020 05:32:42 AM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  8.48%  |  [0.12417145 0.11842687 0.12063632 0.12240389 0.12113174]  |  12.14 (+/- 0.38)  |  0.004078  |  1.755  |
03/07/2020 05:32:42 AM - INFO - |  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  75.73 (+/- 2.09)  |  0.8455  |  0.008558  |
03/07/2020 05:32:42 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  69.28%  |  [0.75961114 0.74016792 0.75519222 0.7565179  0.73607427]  |  74.95 (+/- 1.90)  |  33.13  |  0.01171  |
03/07/2020 05:32:42 AM - INFO - |  9  |  MULTINOMIAL_NB  |  68.79%  |  [0.76049492 0.74149359 0.75519222 0.75695979 0.74270557]  |  75.14 (+/- 1.55)  |  0.05633  |  0.01006  |
03/07/2020 05:32:42 AM - INFO - |  10  |  NEAREST_CENTROID  |  66.70%  |  [0.71674768 0.71807335 0.71851525 0.71939903 0.70601238]  |  71.57 (+/- 0.99)  |  0.01741  |  0.02014  |
03/07/2020 05:32:42 AM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  69.64%  |  [0.76491383 0.74900574 0.76270437 0.76977464 0.74889478]  |  75.91 (+/- 1.71)  |  8.78  |  0.01103  |
03/07/2020 05:32:42 AM - INFO - |  12  |  PERCEPTRON  |  53.86%  |  [0.63146266 0.61025188 0.58948299 0.60450729 0.60477454]  |  60.81 (+/- 2.71)  |  0.7884  |  0.02289  |
03/07/2020 05:32:42 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  63.81%  |  [0.69774635 0.6752099  0.68448962 0.69421122 0.67992927]  |  68.63 (+/- 1.70)  |  40.29  |  1.201  |
03/07/2020 05:32:42 AM - INFO - |  14  |  RIDGE_CLASSIFIER  |  70.02%  |  [0.76756518 0.74502872 0.76800707 0.776403   0.76083112]  |  76.36 (+/- 2.10)  |  2.765  |  0.02252  |
03/07/2020 05:32:42 AM - INFO - 

FINAL CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH DEFAULT PARAMETERS
03/07/2020 05:32:42 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 05:32:42 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 05:32:42 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  36.52%  |  [0.39460893 0.39019001 0.39991162 0.39019001 0.40539346]  |  39.61 (+/- 1.18)  |  4.665  |  0.2575  |
03/07/2020 05:32:42 AM - INFO - |  2  |  BERNOULLI_NB  |  45.84%  |  [0.48033584 0.46133451 0.48563853 0.47547503 0.48938992]  |  47.84 (+/- 1.95)  |  0.05716  |  0.05391  |
03/07/2020 05:32:42 AM - INFO - |  3  |  COMPLEMENT_NB  |  71.04%  |  [0.77110031 0.75740168 0.77993814 0.7715422  0.76038904]  |  76.81 (+/- 1.64)  |  0.06305  |  0.01045  |
03/07/2020 05:32:42 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  42.95%  |  [0.4922669  0.47370747 0.46486964 0.47282369 0.48983201]  |  47.87 (+/- 2.11)  |  10.42  |  0.008561  |
03/07/2020 05:32:42 AM - INFO - |  5  |  GRADIENT_BOOSTING_CLASSIFIER  |  59.25%  |  [0.65178966 0.6195316  0.63013699 0.65090588 0.6357206 ]  |  63.76 (+/- 2.47)  |  334.0  |  0.1755  |
03/07/2020 05:32:42 AM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  7.20%  |  [0.06937693 0.08174989 0.07909854 0.08263367 0.08311229]  |  7.92 (+/- 1.02)  |  0.003774  |  2.168  |
03/07/2020 05:32:42 AM - INFO - |  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  75.73 (+/- 2.09)  |  0.8393  |  0.008751  |
03/07/2020 05:32:42 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  69.17%  |  [0.74458683 0.72646929 0.74193548 0.74768007 0.72413793]  |  73.70 (+/- 1.94)  |  29.69  |  0.01173  |
03/07/2020 05:32:42 AM - INFO - |  9  |  MULTINOMIAL_NB  |  65.64%  |  [0.7105612  0.69951392 0.69863014 0.69995581 0.69142352]  |  70.00 (+/- 1.22)  |  0.04449  |  0.01024  |
03/07/2020 05:32:42 AM - INFO - |  10  |  NEAREST_CENTROID  |  64.91%  |  [0.69774635 0.69244366 0.69818825 0.70393283 0.69849691]  |  69.82 (+/- 0.73)  |  0.02025  |  0.01422  |
03/07/2020 05:32:42 AM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  68.35%  |  [0.76005303 0.7308882  0.7410517  0.75784357 0.73386384]  |  74.47 (+/- 2.42)  |  1.014  |  0.02255  |
03/07/2020 05:32:42 AM - INFO - |  12  |  PERCEPTRON  |  63.34%  |  [0.69818825 0.67565179 0.6955369  0.70349094 0.68302387]  |  69.12 (+/- 2.05)  |  0.535  |  0.02502  |
03/07/2020 05:32:42 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  62.43%  |  [0.68228016 0.6597437  0.67034909 0.67609368 0.66534041]  |  67.08 (+/- 1.58)  |  36.12  |  0.619  |
03/07/2020 05:32:42 AM - INFO - |  14  |  RIDGE_CLASSIFIER  |  70.67%  |  [0.76623951 0.74900574 0.76889085 0.77330977 0.75464191]  |  76.24 (+/- 1.82)  |  2.184  |  0.02251  |
03/07/2020 05:32:42 AM - INFO - 

FINAL CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/07/2020 05:32:42 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/07/2020 05:32:42 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/07/2020 05:32:42 AM - INFO - |  1  |  ADA_BOOST_CLASSIFIER  |  44.04%  |  [0.4719399  0.45735749 0.46531153 0.46973045 0.47126437]  |  46.71 (+/- 1.08)  |  18.42  |  0.9331  |
03/07/2020 05:32:42 AM - INFO - |  2  |  BERNOULLI_NB  |  62.56%  |  [0.69200177 0.67741935 0.6752099  0.67388422 0.67992927]  |  67.97 (+/- 1.30)  |  0.0534  |  0.05116  |
03/07/2020 05:32:42 AM - INFO - |  3  |  COMPLEMENT_NB  |  71.22%  |  [0.776403   0.75828546 0.78126381 0.77507733 0.76967286]  |  77.21 (+/- 1.57)  |  0.06775  |  0.0101  |
03/07/2020 05:32:42 AM - INFO - |  4  |  DECISION_TREE_CLASSIFIER  |  45.63%  |  [0.49536014 0.49005745 0.48077773 0.47989395 0.49955791]  |  48.91 (+/- 1.56)  |  4.511  |  0.005316  |
03/07/2020 05:32:42 AM - INFO - |  5  |  GRADIENT_BOOSTING_CLASSIFIER  |  59.65%  |  [0.65930181 0.62615996 0.64295183 0.65709236 0.65119363]  |  64.73 (+/- 2.40)  |  658.7  |  0.3424  |
03/07/2020 05:32:42 AM - INFO - |  6  |  K_NEIGHBORS_CLASSIFIER  |  8.48%  |  [0.12417145 0.11842687 0.12063632 0.12240389 0.12113174]  |  12.14 (+/- 0.38)  |  0.004078  |  1.755  |
03/07/2020 05:32:42 AM - INFO - |  7  |  LINEAR_SVC  |  69.82%  |  [0.76270437 0.74458683 0.76182059 0.7715422  0.74580018]  |  75.73 (+/- 2.09)  |  0.8455  |  0.008558  |
03/07/2020 05:32:42 AM - INFO - |  8  |  LOGISTIC_REGRESSION  |  69.28%  |  [0.75961114 0.74016792 0.75519222 0.7565179  0.73607427]  |  74.95 (+/- 1.90)  |  33.13  |  0.01171  |
03/07/2020 05:32:42 AM - INFO - |  9  |  MULTINOMIAL_NB  |  68.79%  |  [0.76049492 0.74149359 0.75519222 0.75695979 0.74270557]  |  75.14 (+/- 1.55)  |  0.05633  |  0.01006  |
03/07/2020 05:32:42 AM - INFO - |  10  |  NEAREST_CENTROID  |  66.70%  |  [0.71674768 0.71807335 0.71851525 0.71939903 0.70601238]  |  71.57 (+/- 0.99)  |  0.01741  |  0.02014  |
03/07/2020 05:32:42 AM - INFO - |  11  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  69.64%  |  [0.76491383 0.74900574 0.76270437 0.76977464 0.74889478]  |  75.91 (+/- 1.71)  |  8.78  |  0.01103  |
03/07/2020 05:32:42 AM - INFO - |  12  |  PERCEPTRON  |  53.86%  |  [0.63146266 0.61025188 0.58948299 0.60450729 0.60477454]  |  60.81 (+/- 2.71)  |  0.7884  |  0.02289  |
03/07/2020 05:32:42 AM - INFO - |  13  |  RANDOM_FOREST_CLASSIFIER  |  63.81%  |  [0.69774635 0.6752099  0.68448962 0.69421122 0.67992927]  |  68.63 (+/- 1.70)  |  40.29  |  1.201  |
03/07/2020 05:32:42 AM - INFO - |  14  |  RIDGE_CLASSIFIER  |  70.02%  |  [0.76756518 0.74502872 0.76800707 0.776403   0.76083112]  |  76.36 (+/- 2.10)  |  2.765  |  0.02252  |
```