## Logs: Running RandomForestClassifier Grid Search

```
/comp551-2020-p2_classification_of_textual_data/code/grid_search/RandomForestClassifier.py
### Grid search for Random Forest Classifier: TWENTY_NEWS_GROUP Dataset
Performing grid search...
pipeline: ['classifier']
parameters:
{'classifier__max_features': [6, 11, 16, 21, 26, 31],
 'classifier__n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
Fitting 5 folds for each of 60 candidates, totalling 300 fits
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
[Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:  2.7min
[Parallel(n_jobs=-1)]: Done 184 tasks      | elapsed: 16.9min
[Parallel(n_jobs=-1)]: Done 300 out of 300 | elapsed: 28.3min finished
done in 1735.247s

Best score: 0.691
Best parameters set:
	classifier__max_features: 26
	classifier__n_estimators: 100
Running RandomForestClassifier with default values
________________________________________________________________________________
Training:
RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
train time: 37.497s
test time:  0.648s
accuracy:   0.622


===> Classification Report:

                          precision    recall  f1-score   support

             alt.atheism       0.49      0.41      0.45       319
           comp.graphics       0.60      0.59      0.60       389
 comp.os.ms-windows.misc       0.55      0.68      0.61       394
comp.sys.ibm.pc.hardware       0.62      0.57      0.59       392
   comp.sys.mac.hardware       0.64      0.64      0.64       385
          comp.windows.x       0.64      0.66      0.65       395
            misc.forsale       0.69      0.77      0.73       390
               rec.autos       0.41      0.66      0.50       396
         rec.motorcycles       0.66      0.68      0.67       398
      rec.sport.baseball       0.69      0.78      0.74       397
        rec.sport.hockey       0.81      0.80      0.81       399
               sci.crypt       0.79      0.66      0.72       396
         sci.electronics       0.50      0.41      0.45       393
                 sci.med       0.76      0.63      0.69       396
               sci.space       0.69      0.66      0.67       394
  soc.religion.christian       0.59      0.78      0.67       398
      talk.politics.guns       0.52      0.59      0.55       364
   talk.politics.mideast       0.81      0.72      0.76       376
      talk.politics.misc       0.54      0.34      0.42       310
      talk.religion.misc       0.32      0.09      0.14       251

                accuracy                           0.62      7532
               macro avg       0.62      0.61      0.60      7532
            weighted avg       0.62      0.62      0.62      7532

### Grid search for Random Forest Classifier: IMDB_REVIEWS Dataset

===> Reading files from /home/ets-crchum/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/train/neg

===> Reading files from /home/ets-crchum/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/train/pos

===> Reading files from /home/ets-crchum/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/test/neg

===> Reading files from /home/ets-crchum/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/test/pos
Performing grid search...
pipeline: ['classifier']
parameters:
{'classifier__max_features': [6, 11, 16, 21, 26, 31],
 'classifier__n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
Fitting 5 folds for each of 60 candidates, totalling 300 fits
[Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:  2.8min
[Parallel(n_jobs=-1)]: Done 184 tasks      | elapsed: 17.2min
[Parallel(n_jobs=-1)]: Done 300 out of 300 | elapsed: 29.4min finished
done in 1807.025s

Best score: 0.361
Best parameters set:
	classifier__max_features: 26
	classifier__n_estimators: 100
Running RandomForestClassifier with default values
________________________________________________________________________________
Training:
RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
train time: 72.125s
test time:  1.516s
accuracy:   0.375


===> Classification Report:

              precision    recall  f1-score   support

           1       0.38      0.91      0.54      5022
           2       0.49      0.01      0.02      2302
           3       0.38      0.02      0.04      2541
           4       0.37      0.07      0.12      2635
           7       0.31      0.05      0.08      2307
           8       0.24      0.08      0.12      2850
           9       0.41      0.01      0.01      2344
          10       0.38      0.84      0.53      4999

    accuracy                           0.38     25000
   macro avg       0.37      0.25      0.18     25000
weighted avg       0.37      0.38      0.25     25000
```