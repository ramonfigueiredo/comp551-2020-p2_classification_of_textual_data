## Ensemble results and ML algorithms selection

* Five best classifiers in TWENTY_NEWS_GROUPS dataset

| ID  | ML Algorithm                    | Accuracy Score (%) |
| --- | --------------------------------| ------------------ |
|  1  |  COMPLEMENT_NB                  |  71.22%            |
|  2  |  RIDGE_CLASSIFIER               |  70.02%            |
|  3  |  LINEAR_SVC                     |  69.82%            |
|  4  |  PASSIVE_AGGRESSIVE_CLASSIFIER  |  69.62%            |
|  5  |  LOGISTIC_REGRESSION            |  69.28%            |

* Five best classifiers in IMDB_REVIEWS dataset (Binary classification)

| ID  | ML Algorithm                    | Accuracy Score (%) |
| --- | --------------------------------| ------------------ |
|  1  |  PASSIVE_AGGRESSIVE_CLASSIFIER  | 88.07%             |
|  2  |  LOGISTIC_REGRESSION            | 87.75%             |
|  3  |  LINEAR_SVC                     | 87.13%             |
|  4  |  RIDGE_CLASSIFIER               | 86.90%             |
|  5  |  RANDOM_FOREST_CLASSIFIER       | 85.45%             |

* Five best classifiers in IMDB_REVIEWS dataset (Multi-class classification)

| ID  | ML Algorithm                    | Accuracy Score (%)           |
| --- | --------------------------------| ---------------------------- |
|  1  |  LOGISTIC_REGRESSION            | 42.04%                       |
|  2  |  PASSIVE_AGGRESSIVE_CLASSIFIER  | 41.81%                       |
|  3  |  LINEAR_SVC                     | 40.80%                       |
|  4  |  RIDGE_CLASSIFIER               | 38.55%                       |
|  5  |  ADA_BOOST_CLASSIFIER           | 38.02% (too low, not to use) |


Therefore, we decided to combine the following algorithms using VotingClassifier (voting='hard' and voting='soft') and StackingClassifier: 

* COMPLEMENT_NB
* LINEAR_SVC
* LOGISTIC_REGRESSION
* PASSIVE_AGGRESSIVE_CLASSIFIER
* RIDGE_CLASSIFIER