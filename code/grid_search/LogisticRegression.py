# Reference: https://chrisalbon.com/machine_learning/model_selection/hyperparameter_tuning_using_grid_search/
# Load libraries
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from datasets.load_dataset import load_twenty_news_groups, load_imdb_reviews
from utils.dataset_enum import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer


def grid_search_logistic_regression(dataset):

    if dataset == Dataset.TWENTY_NEWS_GROUP:
        remove = ('headers', 'footers', 'quotes')

        data_train = \
            load_twenty_news_groups(subset='train', categories=None, shuffle=True, random_state=0, remove=remove)

        X_train, y_train = data_train.data, data_train.target

    elif dataset == Dataset.IMDB_REVIEWS:
        X_train, y_train = \
            load_imdb_reviews(subset='train', binary_labels=False, verbose=False, shuffle=True, random_state=0)

    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
    X_train = vectorizer.fit_transform(X_train)

    # Create logistic regression
    logistic = linear_model.LogisticRegression(solver='liblinear')

    # Create regularization penalty space
    # penalty = ['l1', 'l2', 'elasticnet', None]
    penalty = ['l2']

    # Create regularization hyperparameter space
    C = np.logspace(0, 4, 10)

    # Create hyperparameter options
    hyperparameters = dict(C=C, penalty=penalty)

    # Create grid search using 5-fold cross validation
    clf = GridSearchCV(logistic, hyperparameters, cv=5, verbose=False, n_jobs=-1)

    # Fit grid search
    best_model = clf.fit(X_train, y_train)

    # View best hyperparameters
    print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
    print('Best C:', best_model.best_estimator_.get_params()['C'])


if __name__ == '__main__':
    print("### Grid search for Logistic Regression: TWENTY_NEWS_GROUP Dataset")
    grid_search_logistic_regression(Dataset.TWENTY_NEWS_GROUP)

    print("### Grid search for Logistic Regression: TWENTY_NEWS_GROUP Dataset")
    grid_search_logistic_regression(Dataset.IMDB_REVIEWS)
