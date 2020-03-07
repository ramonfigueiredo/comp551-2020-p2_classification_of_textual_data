## Results using all classifiers and all datasets (classifier with default parameters)

### TWENTY_NEWS_GROUPS dataset (removing headers signatures and quoting)  

#### TWENTY_NEWS_GROUPS: Final classification table

| ID | ML Algorithm                     | Accuracy Score   | K-fold Cross Validation (CV) (k = 5)  |  CV (Mean +/- Std)       | Training time (seconds)  | Test time (seconds)  | 
| -- | -------------------------------  | ---------------- | ------------------------------------- |------------------------  | ------------------------ | -------------------  |
| 1  | ADA_BOOST_CLASSIFIER             | 36.54%           |                                       |                          | 4.842                    | 0.256                |
| 2  | BERNOULLI_NB                     | 45.84%           |                                       |                          | *0.062*                  | 0.053                |
| 3  | COMPLEMENT_NB                    | **71.34%**       |                                       |                          | **0.063**                | **0.010**            |
| 4  | DECISION_TREE_CLASSIFIER         | 43.92%           |                                       |                          | 10.921                   | *0.006*              |
| 5  | GRADIENT_BOOSTING_CLASSIFIER     | 59.68%           |                                       |                          | 337.842                  | 0.181                |
| 6  | K_NEIGHBORS_CLASSIFIER           | 07.01%           |                                       |                          | 0.002                    | 1.693                |
| 7  | LINEAR_SVC                       | 69.68%           |                                       |                          | 0.763                    | 0.009                |
| 8 | LOGISTIC_REGRESSION              | 69.46%           |                                       |                          | 17.369                   | 0.011                |
| 9 | MULTINOMIAL_NB                   | 67.13%           |                                       |                          | 0.083                    | 0.010                |
| 10 | NEAREST_CENTROID                 | 64.27%           |                                       |                          | 0.016                    | 0.013                |
| 11 | PASSIVE_AGGRESSIVE_CLASSIFIER    | 68.48%           |                                       |                          | 0.410                    | 0.013                |
| 12 | PERCEPTRON                       | 63.36%           |                                       |                          | 0.411                    | 0.013                |
| 13 | RANDOM_FOREST_CLASSIFIER         | 62.68%           |                                       |                          | 6.569                    | 0.305                |
| 14 | RIDGE_CLASSIFIER                 | 70.35%           |                                       |                          | 2.367                    | 0.021                |

#### TWENTY_NEWS_GROUPS: Plotting

* Accuracy score for the TWENTY_NEWS_GROUPS dataset (Removing headers signatures and quoting)
 
    ![TWENTY_NEWS_GROUPS](https://github.com/ramonfigueiredopessoa/comp551-2020-p2_classification_of_textual_data/blob/master/code/results/TWENTY_NEWS_GROUPS-ml_with_default_parameters.png)


### IMDB_REVIEWS dataset (Multi-class classification)  

#### IMDB_REVIEWS: Final classification table

| ID | ML Algorithm                     | Accuracy Score   | K-fold Cross Validation (CV) (k = 5)  |  CV (Mean +/- Std)       | Training time (seconds)  | Test time (seconds)  | 
| -- | -------------------------------  | ---------------- | ------------------------------------- |------------------------  | ------------------------ | -------------------  |
| 1  | ADA_BOOST_CLASSIFIER             | 35.86            |                                       |                          |  11.368                  | 0.717                |
| 2  | BERNOULLI_NB                     | 37.132           |                                       |                          |  0.039                   | 0.038                |
| 3  | COMPLEMENT_NB                    | 37.312           |                                       |                          |  **0.034**               | **0.019**                |
| 4  | DECISION_TREE_CLASSIFIER         | 25.764           |                                       |                          |  35.066                  | *0.014*                |
| 5  | GRADIENT_BOOSTING_CLASSIFIER     | 37.624           |                                       |                          |  397.786                 | 0.258                |
| 6  | K_NEIGHBORS_CLASSIFIER           | 26.352           |                                       |                          |  *0.006*                 | 12.872               |
| 7  | LINEAR_SVC                       | 37.328           |                                       |                          |  1.769                   | 0.018                |
| 8 | LOGISTIC_REGRESSION              | 42.084           |                                       |                          |  9.818                   | 0.021                |
| 9 | MULTINOMIAL_NB                   | 34.924           |                                       |                          |  0.067                   | 0.019                |
| 10 | NEAREST_CENTROID                 | 36.844           |                                       |                          |  0.023                   | 0.023                |
| 11 | PASSIVE_AGGRESSIVE_CLASSIFIER    | 33.112           |                                       |                          |  0.508                   | 0.028                |
| 12 | PERCEPTRON                       | 32.66            |                                       |                          |  0.314                   | 0.027                |
| 13 | RANDOM_FOREST_CLASSIFIER         | 37.54            |                                       |                          |  10.693                  | 0.417                |
| 14 | RIDGE_CLASSIFIER                 | 38.716           |                                       |                          |  3.194                   | 0.043                |

#### IMDB_REVIEWS: Plotting

* Accuracy score of IMDB_REVIEWS dataset (Multi-class classification)
 
    ![IMDB_REVIEWS](https://github.com/ramonfigueiredopessoa/comp551-2020-p2_classification_of_textual_data/blob/master/code/results/IMDB_REVIEWS-ml_with_default_parameters.png)

### Logs: TWENTY_NEWS_GROUPS and IMDB_REVIEWS (Multi-class classification)

Logs after run all classifiers using the TWENTY_NEWS_GROUPS dataset (removing headers signatures and quoting) and IMDB_REVIEWS dataset (Multi-class classification).

```

```