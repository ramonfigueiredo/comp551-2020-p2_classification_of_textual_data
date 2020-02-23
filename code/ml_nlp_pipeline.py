import string
import time
from enum import Enum, unique

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.datasets import fetch_20newsgroups

nltk.download('punkt')
nltk.download('stopwords')

@unique
class Datasets(Enum):
    NEWS_GROUPS = 1
    NEWS_GROUPS_ADAPTED = 2


def load_dataset(path, header='infer', sep=',', x_col_indices=slice(-1), y_col_indices=-1):
    dataset = pd.read_csv(path, header=header, sep=sep)

    X = dataset.iloc[:, x_col_indices].values
    y = dataset.iloc[:, y_col_indices].values

    return X, y


def cross_validation():

    global n_folds, clf, scores
    n_folds = 5
    print('n_folds', n_folds)
    print('\nCross Validation')
    from sklearn.model_selection import cross_val_score

    start = time.time()
    print('\n\nMultinomialNB')
    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB(alpha=.01)
    print(clf)
    scores = cross_val_score(clf, vectors_train, y_train, cv=n_folds)
    print('scores', scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print("Cross Validation: MultinomialNB. It took {} seconds".format(time.time() - start))


def grid_search():
    start = time.time()
    global clf, scores
    print('Grid Search')
    from sklearn.model_selection import GridSearchCV
    from sklearn.naive_bayes import MultinomialNB
    tuned_parameters = [{'alpha': [0.01, 1, 0.5, 0.2, 0.1]}]
    print('tuned_parameters', tuned_parameters)
    clf = MultinomialNB()
    print(clf)
    clf = GridSearchCV(clf, tuned_parameters, cv=n_folds, refit=False)
    clf.fit(vectors_train, y_train)
    scores = clf.cv_results_['mean_test_score']
    print('scores:', scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    best_accuracy = clf.best_score_
    print("Grid Search: Best accuracy\n", best_accuracy)
    best_parameters = clf.best_params_
    print("Grid Search: Best parameters\n", best_parameters)
    print("Grid Search: MultinomialNB. It took {} seconds".format(time.time() - start))


if __name__ == '__main__':

    start = time.time()

    newsgroups = fetch_20newsgroups(subset='all')

    X = []
    for id_and_comment, target_name in zip(enumerate(newsgroups.data), newsgroups.target_names):
        comment = id_and_comment[1]

        # split into words
        tokens = word_tokenize(str(comment))

        # convert to lower case
        tokens = [w.lower() for w in tokens]

        # remove punctuation from each word
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]

        # remove remaining tokens that are not alphabetic
        words = [word for word in stripped if word.isalpha()]

        # filter out stop wordsstop_words = set(stopwords.words('english'))
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if not w in stop_words]
        comment = ' '.join(words)

        X.append(comment)

    # Pre-processing
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(newsgroups.data, newsgroups.target, train_size=0.8, test_size=0.2)

    print('\nlen(X_train)', len(X_train))
    print('\nlen(X_test)', len(X_test))
    print('\nlen(y_train)', len(y_train))
    print('\nlen(y_test)', len(y_test))

    print('\nX_train[1]\n', X_train[1])

    # Transform text into vectors
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(stop_words='english')
    # print(vectorizer)

    vectors_train = vectorizer.fit_transform(X_train)
    vectors_test = vectorizer.transform(X_test)

    print('\nvectors_train.shape', vectors_train.shape)

    print('\nvectors_test.shape', vectors_test.shape)

    # Tf-idf term weighting(only for textual features)
    from sklearn.feature_extraction.text import TfidfVectorizer

    tf_idf_vectorizer = TfidfVectorizer(stop_words='english')
    vectors_train_idf = tf_idf_vectorizer.fit_transform(X_train)
    vectors_test_idf = vectorizer.transform(X_test)

    print('\nX_train[1:2]', X_train[1:2])

    print('\nvectors_train_idf.shape', vectors_train_idf.shape)
    print('\nvectors_train_idf', vectors_train_idf)

    from sklearn.preprocessing import Normalizer

    normalizer_train = Normalizer().fit(X=vectors_train)
    vectors_train_normalized = normalizer_train.transform(vectors_train)
    vectors_test_normalized = normalizer_train.transform(vectors_test)

    print('\nvectors_train_normalized', vectors_train_normalized)
    print('\nvectors_test_normalized', vectors_test_normalized)

    # Training
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression()

    print('\nTraining (fit)')
    clf.fit(vectors_train, y_train)

    print(clf)

    print('\nPredict')
    print('\nvectors_test', vectors_test)
    y_pred = clf.predict(vectors_test)
    print('\ny_pred', y_pred)

    # Evaluation

    print('\nEvaluation')
    from sklearn import metrics
    print('accuracy_score', metrics.accuracy_score(y_test, y_pred))
    print('precision_score', metrics.precision_score(y_test, y_pred, average='macro'))
    print('f1_score', metrics.f1_score(y_test, y_pred, average='macro'))
    print('f1_score', metrics.f1_score(y_test, y_pred, average='macro'))

    print(metrics.classification_report(y_test, y_pred))

    cross_validation()

    grid_search()

    print('\n\nDONE!')

    print("Program finished. It took {} seconds".format(time.time() - start))
