# https://towardsdatascience.com/logistic-regression-model-tuning-with-scikit-learn-part-1-425142e01af5
import os
from pprint import pprint
from time import time

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from datasets.load_dataset import load_twenty_news_groups, load_imdb_reviews
from utils.dataset_enum import Dataset


def random_forest_classifier_grid_search(dataset):

    if dataset == Dataset.TWENTY_NEWS_GROUP:
        remove = ('headers', 'footers', 'quotes')

        data_train = \
            load_twenty_news_groups(subset='train', categories=None, shuffle=True, random_state=0, remove=remove)

        data_test = \
            load_twenty_news_groups(subset='test', categories=None, shuffle=True, random_state=0, remove=remove)

        X_train, y_train = data_train.data, data_train.target
        X_test, y_test = data_test.data, data_test.target

        target_names = data_train.target_names

    elif dataset == Dataset.IMDB_REVIEWS:
        db_parent_path = os.getcwd()
        db_parent_path = db_parent_path.replace('grid_search', '')

        X_train, y_train = \
            load_imdb_reviews(subset='train', binary_labels=False, verbose=False, shuffle=True, random_state=0,
                              db_parent_path=db_parent_path)
        X_test, y_test = \
            load_imdb_reviews(subset='test', binary_labels=False, verbose=False, shuffle=True, random_state=0,
                              db_parent_path=db_parent_path)

        # IMDB reviews dataset
        # If binary classification: 0 = neg and 1 = pos.
        # If multi-class classification use the review scores: 1, 2, 3, 4, 7, 8, 9, 10
        target_names = ['1', '2', '3', '4', '7', '8', '9', '10']

    # Extracting features
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    # Create pipeline
    pipeline = Pipeline([('classifier', RandomForestClassifier())])

    # Create param grid.
    param_grid = [
        {
         'classifier__n_estimators': list(range(10, 101, 10)),
         'classifier__max_features': list(range(6, 32, 5))
        }
    ]

    # Create grid search object
    grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5, verbose=True, n_jobs=-1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(param_grid)
    t0 = time()
    grid_search.fit(X_train, y_train)
    print("done in %0.3fs" % (time() - t0))
    print()

    # Fit on data
    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    print("Running RandomForestClassifier with default values")
    clf = RandomForestClassifier()
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    y_pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, y_pred)
    print("accuracy:   %0.3f" % score)

    print("\n\n===> Classification Report:\n")
    print(metrics.classification_report(y_test, y_pred, target_names=target_names))

    # TODO: Use RandomForestClassifier with best parameters


if __name__ == '__main__':
    print("### Grid search for Random Forest Classifier: TWENTY_NEWS_GROUP Dataset")
    random_forest_classifier_grid_search(Dataset.TWENTY_NEWS_GROUP)

    print("### Grid search for Random Forest Classifier: IMDB_REVIEWS Dataset")
    random_forest_classifier_grid_search(Dataset.IMDB_REVIEWS)
