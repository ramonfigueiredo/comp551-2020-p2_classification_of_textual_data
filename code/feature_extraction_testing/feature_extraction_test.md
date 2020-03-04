## Feature extraction test

#### Summary

Use the following best CountVectorizer, HashingVectorizer, TfidfVectorizer options:

```
if vectorizer_enum == Vectorizer.COUNT_VECTORIZER:
    if dataset == Dataset.TWENTY_NEWS_GROUPS:
        vectorizer = CountVectorizer(stop_words='english', strip_accents='unicode', analyzer='word', binary=True)
    elif dataset == Dataset.IMDB_REVIEWS:
        vectorizer = CountVectorizer(stop_words='english', strip_accents='unicode', ngram_range=(1, 2), analyzer='word', binary=True)

if vectorizer_enum == Vectorizer.HASHING_VECTORIZER:
    if dataset == Dataset.TWENTY_NEWS_GROUPS:
        vectorizer = HashingVectorizer(stop_words='english', strip_accents='unicode', analyzer='word', binary=True)
    elif dataset == Dataset.IMDB_REVIEWS:
        vectorizer = HashingVectorizer(stop_words='english', strip_accents='unicode', ngram_range=(1, 2), analyzer='word', binary=True)

if vectorizer_enum == Vectorizer.TF_IDF_VECTORIZER:
    if dataset == Dataset.TWENTY_NEWS_GROUPS:
        vectorizer = TfidfVectorizer(stop_words='english', strip_accents='unicode', analyzer='word', binary=True)
    elif dataset == Dataset.IMDB_REVIEWS:
        vectorizer = TfidfVectorizer(stop_words='english', strip_accents='unicode', ngram_range=(1, 2), analyzer='word', binary=True)

if vectorizer_enum == Vectorizer.HASHING_VECTORIZER:
    X_train = vectorizer.transform(X_train)
    X_test = vectorizer.transform(X_test)
else:
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
```

#### Test 1: BernoulliNB using CountVectorizer(), HashingVectorizer(), TfidfVectorizer()

```
FINAL CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/03/2020 11:32:51 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/03/2020 11:32:51 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/03/2020 11:32:51 PM - INFO - |  2-1  |  BERNOULLI_NB using COUNT_VECTORIZER  |  53.64%  |  [0.58992488 0.55634114 0.58550597 0.59301812 0.59018568]  |  58.30 (+/- 2.71)  |  0.1485  |  0.1398  |
03/03/2020 11:32:51 PM - INFO - |  2-2  |  BERNOULLI_NB using HASHING_VECTORIZER  |  46.20%  |  [0.49977905 0.48298719 0.48696421 0.49447636 0.49602122]  |  49.20 (+/- 1.23)  |  1.049  |  1.167  |
03/03/2020 11:32:51 PM - INFO - |  2-3  |  BERNOULLI_NB using TF_IDF_VECTORIZER  |  53.64%  |  [0.58992488 0.55634114 0.58550597 0.59301812 0.59018568]  |  58.30 (+/- 2.71)  |  0.1432  |  0.1301  |
03/03/2020 11:32:51 PM - INFO - 


FINAL CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/03/2020 11:33:35 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/03/2020 11:33:35 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/03/2020 11:33:35 PM - INFO - |  2-1  |  BERNOULLI_NB using COUNT_VECTORIZER  |  35.98%  |  [0.3688 0.369  0.3678 0.3736 0.3792]  |  37.17 (+/- 0.85)  |  0.1221  |  0.1209  |
03/03/2020 11:33:35 PM - INFO - |  2-2  |  BERNOULLI_NB using HASHING_VECTORIZER  |  33.72%  |  [0.3436 0.3514 0.3474 0.345  0.347 ]  |  34.69 (+/- 0.53)  |  0.5036  |  0.5367  |
03/03/2020 11:33:35 PM - INFO - |  2-3  |  BERNOULLI_NB using TF_IDF_VECTORIZER  |  35.98%  |  [0.3688 0.369  0.3678 0.3736 0.3792]  |  37.17 (+/- 0.85)  |  0.1051  |  0.1091  |
```

#### Test 2: BernoulliNB using CountVectorizer(stop_words='english'), HashingVectorizer(stop_words='english'), TfidfVectorizer(stop_words='english')

```	
FINAL CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/03/2020 11:36:21 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/03/2020 11:36:21 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/03/2020 11:36:21 PM - INFO - |  2-1  |  BERNOULLI_NB using COUNT_VECTORIZER  |  54.39%  |  [0.59611136 0.5687141  0.6040654  0.59743703 0.59902741]  |  59.31 (+/- 2.49)  |  0.1656  |  0.1452  |
03/03/2020 11:36:21 PM - INFO - |  2-2  |  BERNOULLI_NB using HASHING_VECTORIZER  |  46.55%  |  [0.49315068 0.48387097 0.49447636 0.49933716 0.49823165]  |  49.38 (+/- 1.09)  |  1.178  |  1.39  |
03/03/2020 11:36:21 PM - INFO - |  2-3  |  BERNOULLI_NB using TF_IDF_VECTORIZER  |  54.39%  |  [0.59611136 0.5687141  0.6040654  0.59743703 0.59902741]  |  59.31 (+/- 2.49)  |  0.1373  |  0.1255  |
03/03/2020 11:36:21 PM - INFO - 


FINAL CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/03/2020 11:37:02 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/03/2020 11:37:02 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/03/2020 11:37:02 PM - INFO - |  2-1  |  BERNOULLI_NB using COUNT_VECTORIZER  |  35.89%  |  [0.37   0.3638 0.3702 0.3734 0.381 ]  |  37.17 (+/- 1.12)  |  0.08984  |  0.0995  |
03/03/2020 11:37:02 PM - INFO - |  2-2  |  BERNOULLI_NB using HASHING_VECTORIZER  |  33.32%  |  [0.3318 0.3428 0.3412 0.3338 0.342 ]  |  33.83 (+/- 0.92)  |  0.4654  |  0.5212  |
03/03/2020 11:37:02 PM - INFO - |  2-3  |  BERNOULLI_NB using TF_IDF_VECTORIZER  |  35.89%  |  [0.37   0.3638 0.3702 0.3734 0.381 ]  |  37.17 (+/- 1.12)  |  0.1012  |  0.08075  |
```

#### Test 3: BernoulliNB using CountVectorizer(stop_words='english'), HashingVectorizer(stop_words='english'), TfidfVectorizer(stop_words='english')

strip_accents='ascii'
```
FINAL CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/03/2020 11:52:50 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/03/2020 11:52:50 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/03/2020 11:52:50 PM - INFO - |  2-1  |  BERNOULLI_NB using COUNT_VECTORIZER  |  54.39%  |  [0.59611136 0.5687141  0.6040654  0.59743703 0.59902741]  |  59.31 (+/- 2.49)  |  0.1396  |  0.1229  |
03/03/2020 11:52:50 PM - INFO - |  2-2  |  BERNOULLI_NB using HASHING_VECTORIZER  |  46.55%  |  [0.49315068 0.48387097 0.49447636 0.49933716 0.49823165]  |  49.38 (+/- 1.09)  |  1.019  |  1.123  |
03/03/2020 11:52:50 PM - INFO - |  2-3  |  BERNOULLI_NB using TF_IDF_VECTORIZER  |  54.39%  |  [0.59611136 0.5687141  0.6040654  0.59743703 0.59902741]  |  59.31 (+/- 2.49)  |  0.1294  |  0.1277  |
03/03/2020 11:52:50 PM - INFO - 


FINAL CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/03/2020 11:53:32 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/03/2020 11:53:32 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/03/2020 11:53:32 PM - INFO - |  2-1  |  BERNOULLI_NB using COUNT_VECTORIZER  |  35.89%  |  [0.3712 0.364  0.3698 0.3724 0.3806]  |  37.16 (+/- 1.07)  |  0.09603  |  0.09535  |
03/03/2020 11:53:32 PM - INFO - |  2-2  |  BERNOULLI_NB using HASHING_VECTORIZER  |  33.36%  |  [0.3326 0.3434 0.3414 0.334  0.342 ]  |  33.87 (+/- 0.89)  |  0.4606  |  0.5065  |
03/03/2020 11:53:32 PM - INFO - |  2-3  |  BERNOULLI_NB using TF_IDF_VECTORIZER  |  35.89%  |  [0.3712 0.364  0.3698 0.3724 0.3806]  |  37.16 (+/- 1.07)  |  0.09539  |  0.0941  |

```

strip_accents='unicode' => BEST
```
FINAL CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/03/2020 11:55:32 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/03/2020 11:55:32 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/03/2020 11:55:32 PM - INFO - |  2-1  |  BERNOULLI_NB using COUNT_VECTORIZER  |  54.39%  |  [0.59611136 0.5687141  0.6040654  0.59743703 0.59902741]  |  59.31 (+/- 2.49)  |  0.155  |  0.1284  |
03/03/2020 11:55:32 PM - INFO - |  2-2  |  BERNOULLI_NB using HASHING_VECTORIZER  |  46.55%  |  [0.49315068 0.48387097 0.49447636 0.49933716 0.49823165]  |  49.38 (+/- 1.09)  |  1.026  |  1.149  |
03/03/2020 11:55:32 PM - INFO - |  2-3  |  BERNOULLI_NB using TF_IDF_VECTORIZER  |  54.39%  |  [0.59611136 0.5687141  0.6040654  0.59743703 0.59902741]  |  59.31 (+/- 2.49)  |  0.1526  |  0.1273  |
03/03/2020 11:55:32 PM - INFO - 


FINAL CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/03/2020 11:56:17 PM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/03/2020 11:56:17 PM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/03/2020 11:56:17 PM - INFO - |  2-1  |  BERNOULLI_NB using COUNT_VECTORIZER  |  35.90%  |  [0.3706 0.3634 0.37   0.3724 0.3802]  |  37.13 (+/- 1.08)  |  0.08362  |  0.09447  |
03/03/2020 11:56:17 PM - INFO - |  2-2  |  BERNOULLI_NB using HASHING_VECTORIZER  |  33.34%  |  [0.3324 0.343  0.3414 0.3338 0.342 ]  |  33.85 (+/- 0.90)  |  0.4685  |  0.5146  |
03/03/2020 11:56:17 PM - INFO - |  2-3  |  BERNOULLI_NB using TF_IDF_VECTORIZER  |  35.90%  |  [0.3706 0.3634 0.37   0.3724 0.3802]  |  37.13 (+/- 1.08)  |  0.08556  |  0.08952  |

```

#### Test 4: BernoulliNB using CountVectorizer(stop_words='english', strip_accents='unicode'), HashingVectorizer(stop_words='english', strip_accents='unicode'), TfidfVectorizer(stop_words='english', strip_accents='unicode')

ngram_range=(1, 2) => WORST AT TWENTY_NEWS_GROUPS, DON'T USE
ngram_range=(1, 2) => BEST AT IMDB_REVIEWS, USE
```
FINAL CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/04/2020 12:02:07 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/04/2020 12:02:07 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/04/2020 12:02:07 AM - INFO - |  2-1  |  BERNOULLI_NB using COUNT_VECTORIZER  |  41.60%  |  [0.47503314 0.43968184 0.48431286 0.48033584 0.47656941]  |  47.12 (+/- 3.22)  |  1.156  |  1.014  |
03/04/2020 12:02:07 AM - INFO - |  2-2  |  BERNOULLI_NB using HASHING_VECTORIZER  |  38.99%  |  [0.43747238 0.40653999 0.44498453 0.44763588 0.44076039]  |  43.55 (+/- 2.98)  |  1.21  |  1.716  |
03/04/2020 12:02:07 AM - INFO - |  2-3  |  BERNOULLI_NB using TF_IDF_VECTORIZER  |  41.60%  |  [0.47503314 0.43968184 0.48431286 0.48033584 0.47656941]  |  47.12 (+/- 3.22)  |  1.152  |  1.025  |
03/04/2020 12:02:07 AM - INFO - 


FINAL CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/04/2020 12:03:55 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/04/2020 12:03:55 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/04/2020 12:03:55 AM - INFO - |  2-1  |  BERNOULLI_NB using COUNT_VECTORIZER  |  37.48%  |  [0.376  0.3746 0.3738 0.3744 0.3752]  |  37.48 (+/- 0.15)  |  0.9097  |  0.8735  |
03/04/2020 12:03:55 AM - INFO - |  2-2  |  BERNOULLI_NB using HASHING_VECTORIZER  |  33.89%  |  [0.3604 0.3568 0.3542 0.3596 0.3534]  |  35.69 (+/- 0.56)  |  0.6297  |  0.6932  |
03/04/2020 12:03:55 AM - INFO - |  2-3  |  BERNOULLI_NB using TF_IDF_VECTORIZER  |  37.48%  |  [0.376  0.3746 0.3738 0.3744 0.3752]  |  37.48 (+/- 0.15)  |  0.9621  |  0.89  |
```

#### Test 5: BernoulliNB using CountVectorizer(stop_words='english', strip_accents='unicode'), HashingVectorizer(stop_words='english', strip_accents='unicode'), TfidfVectorizer(stop_words='english', strip_accents='unicode') using ngram_range=(1, 1) at TWENTY_NEWS_GROUPS and ngram_range=(1, 2) at IMDB_REVIEWS

analyzer='char' => WORST, DON'T USE
```
FINAL CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/04/2020 12:11:57 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/04/2020 12:11:57 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/04/2020 12:11:57 AM - INFO - |  2-1  |  BERNOULLI_NB using COUNT_VECTORIZER  |  13.32%  |  [0.14052143 0.14891737 0.13389306 0.14935926 0.15694076]  |  14.59 (+/- 1.59)  |  0.01517  |  0.008129  |
03/04/2020 12:11:57 AM - INFO - |  2-2  |  BERNOULLI_NB using HASHING_VECTORIZER  |  10.58%  |  [0.09942554 0.10914715 0.10251878 0.10826337 0.10698497]  |  10.53 (+/- 0.74)  |  0.7834  |  1.092  |
03/04/2020 12:11:57 AM - INFO - |  2-3  |  BERNOULLI_NB using TF_IDF_VECTORIZER  |  13.32%  |  [0.14052143 0.14891737 0.13389306 0.14935926 0.15694076]  |  14.59 (+/- 1.59)  |  0.01493  |  0.008015  |
03/04/2020 12:11:57 AM - INFO -


FINAL CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/04/2020 12:14:25 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/04/2020 12:14:25 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/04/2020 12:14:25 AM - INFO - |  2-1  |  BERNOULLI_NB using COUNT_VECTORIZER  |  27.33%  |  [0.2778 0.267  0.2694 0.275  0.2806]  |  27.40 (+/- 1.02)  |  0.1441  |  0.1433  |
03/04/2020 12:14:25 AM - INFO - |  2-2  |  BERNOULLI_NB using HASHING_VECTORIZER  |  26.74%  |  [0.2668 0.2688 0.2614 0.2582 0.2636]  |  26.38 (+/- 0.75)  |  0.5372  |  0.6365  |
03/04/2020 12:14:25 AM - INFO - |  2-3  |  BERNOULLI_NB using TF_IDF_VECTORIZER  |  27.33%  |  [0.2778 0.267  0.2694 0.275  0.2806]  |  27.40 (+/- 1.02)  |  0.1585  |  0.1358  |

```

analyzer='char_wb' => WORST, DON'T USE
```
FINAL CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/04/2020 12:16:37 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/04/2020 12:16:37 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/04/2020 12:16:37 AM - INFO - |  2-1  |  BERNOULLI_NB using COUNT_VECTORIZER  |  13.33%  |  [0.13787008 0.14847548 0.13654441 0.14891737 0.15738285]  |  14.58 (+/- 1.55)  |  0.01513  |  0.008473  |
03/04/2020 12:16:37 AM - INFO - |  2-2  |  BERNOULLI_NB using HASHING_VECTORIZER  |  10.58%  |  [0.09942554 0.10914715 0.10251878 0.10826337 0.10698497]  |  10.53 (+/- 0.74)  |  0.7708  |  1.087  |
03/04/2020 12:16:37 AM - INFO - |  2-3  |  BERNOULLI_NB using TF_IDF_VECTORIZER  |  13.32%  |  [0.14052143 0.14891737 0.13389306 0.14935926 0.15694076]  |  14.59 (+/- 1.59)  |  0.01346  |  0.006757  |
03/04/2020 12:16:37 AM - INFO - 


FINAL CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/04/2020 12:19:44 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/04/2020 12:19:44 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/04/2020 12:19:44 AM - INFO - |  2-1  |  BERNOULLI_NB using COUNT_VECTORIZER  |  27.37%  |  [0.2782 0.2666 0.2702 0.2762 0.2816]  |  27.46 (+/- 1.09)  |  0.1446  |  0.1463  |
03/04/2020 12:19:44 AM - INFO - |  2-2  |  BERNOULLI_NB using HASHING_VECTORIZER  |  26.74%  |  [0.2668 0.2688 0.2614 0.2582 0.2636]  |  26.38 (+/- 0.75)  |  0.5293  |  0.6407  |
03/04/2020 12:19:44 AM - INFO - |  2-3  |  BERNOULLI_NB using TF_IDF_VECTORIZER  |  27.33%  |  [0.2778 0.267  0.2694 0.275  0.2806]  |  27.40 (+/- 1.02)  |  0.1568  |  0.1304  |

```

analyzer='word' => BEST
````
FINAL CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/04/2020 12:21:54 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/04/2020 12:21:54 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/04/2020 12:21:54 AM - INFO - |  2-1  |  BERNOULLI_NB using COUNT_VECTORIZER  |  54.39%  |  [0.59611136 0.5687141  0.6040654  0.59743703 0.59902741]  |  59.31 (+/- 2.49)  |  0.1385  |  0.1225  |
03/04/2020 12:21:54 AM - INFO - |  2-2  |  BERNOULLI_NB using HASHING_VECTORIZER  |  46.55%  |  [0.49315068 0.48387097 0.49447636 0.49933716 0.49823165]  |  49.38 (+/- 1.09)  |  1.39  |  1.491  |
03/04/2020 12:21:54 AM - INFO - |  2-3  |  BERNOULLI_NB using TF_IDF_VECTORIZER  |  54.39%  |  [0.59611136 0.5687141  0.6040654  0.59743703 0.59902741]  |  59.31 (+/- 2.49)  |  0.154  |  0.1275  |
03/04/2020 12:21:54 AM - INFO - 


FINAL CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/04/2020 12:23:40 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/04/2020 12:23:40 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/04/2020 12:23:40 AM - INFO - |  2-1  |  BERNOULLI_NB using COUNT_VECTORIZER  |  37.48%  |  [0.376  0.3746 0.3738 0.3744 0.3752]  |  37.48 (+/- 0.15)  |  1.007  |  0.9325  |
03/04/2020 12:23:40 AM - INFO - |  2-2  |  BERNOULLI_NB using HASHING_VECTORIZER  |  33.89%  |  [0.3604 0.3568 0.3542 0.3596 0.3534]  |  35.69 (+/- 0.56)  |  0.6469  |  0.6706  |
03/04/2020 12:23:40 AM - INFO - |  2-3  |  BERNOULLI_NB using TF_IDF_VECTORIZER  |  37.48%  |  [0.376  0.3746 0.3738 0.3744 0.3752]  |  37.48 (+/- 0.15)  |  0.9449  |  0.888  |

````

#### Test 6: BernoulliNB using CountVectorizer(stop_words='english', strip_accents='unicode', analyzer='word'), HashingVectorizer(stop_words='english', strip_accents='unicode', analyzer='word'), TfidfVectorizer(stop_words='english', strip_accents='unicode', analyzer='word').  Using ngram_range=(1, 1) at TWENTY_NEWS_GROUPS and ngram_range=(1, 2) at IMDB_REVIEWS. Using binary=True at TWENTY_NEWS_GROUPS and binary=False at IMDB_REVIEWS.

binary=True
```
FINAL CLASSIFICATION TABLE: TWENTY_NEWS_GROUPS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/04/2020 12:25:57 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/04/2020 12:25:57 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/04/2020 12:25:57 AM - INFO - |  2-1  |  BERNOULLI_NB using COUNT_VECTORIZER  |  54.39%  |  [0.59611136 0.5687141  0.6040654  0.59743703 0.59902741]  |  59.31 (+/- 2.49)  |  0.1288  |  0.1242  |
03/04/2020 12:25:57 AM - INFO - |  2-2  |  BERNOULLI_NB using HASHING_VECTORIZER  |  52.23%  |  [0.54661953 0.53910738 0.55501547 0.55678303 0.56321839]  |  55.21 (+/- 1.68)  |  1.072  |  1.137  |
03/04/2020 12:25:57 AM - INFO - |  2-3  |  BERNOULLI_NB using TF_IDF_VECTORIZER  |  54.39%  |  [0.59611136 0.5687141  0.6040654  0.59743703 0.59902741]  |  59.31 (+/- 2.49)  |  0.1292  |  0.1233  |
03/04/2020 12:25:57 AM - INFO - 


FINAL CLASSIFICATION TABLE: IMDB_REVIEWS DATASET, CLASSIFIER WITH BEST PARAMETERS
03/04/2020 12:27:42 AM - INFO - | ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | Training time (seconds) | Test time (seconds) |
03/04/2020 12:27:42 AM - INFO - | -- | ------------ | ------------------ | ------------------------------------ | ----------------- |  ------------------ | ------------------ |
03/04/2020 12:27:42 AM - INFO - |  2-1  |  BERNOULLI_NB using COUNT_VECTORIZER  |  37.48%  |  [0.376  0.3746 0.3738 0.3744 0.3752]  |  37.48 (+/- 0.15)  |  0.9707  |  0.8794  |
03/04/2020 12:27:42 AM - INFO - |  2-2  |  BERNOULLI_NB using HASHING_VECTORIZER  |  34.88%  |  [0.3688 0.3692 0.3626 0.3688 0.3624]  |  36.64 (+/- 0.63)  |  0.6616  |  0.6616  |
03/04/2020 12:27:42 AM - INFO - |  2-3  |  BERNOULLI_NB using TF_IDF_VECTORIZER  |  37.48%  |  [0.376  0.3746 0.3738 0.3744 0.3752]  |  37.48 (+/- 0.15)  |  0.9679  |  0.9723  |

```