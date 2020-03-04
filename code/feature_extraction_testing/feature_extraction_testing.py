import logging
import os
from time import time

from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline

from datasets.load_dataset import load_twenty_news_groups, load_imdb_reviews
from utils.dataset_enum import Dataset
from utils.ml_classifiers_enum import Classifier
from utils.vectorizer_enum import Vectorizer


def get_classifier_with_best_parameters(classifier_enum, best_parameters):

    if classifier_enum == Classifier.BERNOULLI_NB:
        return BernoulliNB(**best_parameters)


def run_classifier_grid_search(classifer, vectorizer_enum, classifier_enum, param_grid, dataset, final_classification_table_best_parameters):

    if param_grid is None:
        return

    if dataset == Dataset.TWENTY_NEWS_GROUPS:
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
        db_parent_path = db_parent_path.replace('feature_extraction_testing', '')

        X_train, y_train = \
            load_imdb_reviews(subset='train', binary_labels=False, verbose=False, shuffle=True, random_state=0,
                              db_parent_path=db_parent_path)
        X_test, y_test = \
            load_imdb_reviews(subset='test', binary_labels=False, verbose=False, shuffle=True, random_state=0,
                              db_parent_path=db_parent_path)

        # IMDB_REVIEWS dataset
        # If binary classification: 0 = neg and 1 = pos.
        # If multi-class classification use the review scores: 1, 2, 3, 4, 7, 8, 9, 10
        target_names = ['1', '2', '3', '4', '7', '8', '9', '10']

    try:
        # 'vectorizer__decode_error': ['strict', 'ignore', 'replace'], ==> No impact
        # 'vectorizer__strip_accents': ['ascii', 'unicode', None (default)], => Best: 'unicode'
        # 'vectorizer__stop_words': ['english', None (default)], => Best: 'english'
        # 'vectorizer__ngram_range': ((1, 1), (1, 2)),  # unigrams (default) or bigrams => unigrams better at TWENTY_NEWS_GROUPS, bigrams better at IMDB_REVIEWS
        # 'vectorizer__analyzer': ['word' (default), 'char', 'char_wb'], => Best: word (default)
        # 'vectorizer__binary': [False (default), True]
        # Extracting features
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

        # Create pipeline
        pipeline = Pipeline([
            ('classifier', classifer)
            # ,
            # ('vectorizer', vectorizer)
        ])

        # Create grid search object
        grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5, verbose=True, n_jobs=-1)

        logging.info("\n\nPerforming grid search...\n")
        logging.info("Parameters:")
        logging.info(param_grid)
        t0 = time()
        grid_search.fit(X_train, y_train)
        logging.info("\tDone in %0.3fs" % (time() - t0))

        # Get best parameters
        logging.info("\tBest score: %0.3f" % grid_search.best_score_)
        logging.info("\tBest parameters set:")
        best_parameters = grid_search.best_estimator_.get_params()
        new_parameters = {}
        for param_name in sorted(param_grid.keys()):
            logging.info("\t\t%s: %r" % (param_name, best_parameters[param_name]))
            key = param_name.replace('classifier__', '')
            value = best_parameters[param_name]
            new_parameters[key] = value

        logging.info('\n\n{}: USING {} WITH BEST PARAMETERS: {}'.format(vectorizer_enum.name, classifier_enum.name, new_parameters))
        clf = get_classifier_with_best_parameters(classifier_enum, new_parameters)
        final_classification_report(clf, X_train, y_train, X_test, y_test, target_names, classifier_enum, vectorizer_enum, final_classification_table_best_parameters)

    except MemoryError as error:
        # Output expected MemoryErrors.
        logging.error(error)

    except Exception as exception:
        # Output unexpected Exceptions.
        logging.error(exception)


def final_classification_report(clf, X_train, y_train,  X_test, y_test, target_names, classifier_enum, vectorizer_enum, final_classification_table):
    # Fit on data
    logging.info('_' * 80)
    logging.info("Training: ")
    logging.info(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    logging.info("Train time: %0.3fs" % train_time)
    # Predict
    t0 = time()
    y_pred = clf.predict(X_test)
    test_time = time() - t0
    logging.info("Test time:  %0.3fs" % test_time)
    accuracy_score = metrics.accuracy_score(y_test, y_pred)
    logging.info("Accuracy score:   %0.3f" % accuracy_score)
    logging.info("\n\n===> Classification Report:\n")
    logging.info(metrics.classification_report(y_test, y_pred, target_names=target_names))

    n_splits = 5
    logging.info("\n\nCross validation:")
    scoring = ['accuracy', 'precision_macro', 'precision_micro', 'precision_weighted', 'recall_macro', 'recall_micro',
               'recall_weighted', 'f1_macro', 'f1_micro', 'f1_weighted', 'jaccard_macro']
    cross_val_scores = cross_validate(clf, X_train, y_train, scoring=scoring, cv=n_splits,
                                      n_jobs=-1, verbose=True)
    cv_test_accuracy = cross_val_scores['test_accuracy']
    logging.info("\taccuracy: {}-fold cross validation: {}".format(5, cv_test_accuracy))
    cv_accuracy_score_mean_std = "%0.2f (+/- %0.2f)" % (cv_test_accuracy.mean() * 100, cv_test_accuracy.std() * 2 * 100)
    logging.info("\ttest accuracy: {}-fold cross validation accuracy: {}".format(n_splits, cv_accuracy_score_mean_std))

    key = str(classifier_enum.value) + "-" + str(vectorizer_enum.value)
    name = str(classifier_enum.name) + " using " + str(vectorizer_enum.name)
    final_classification_table[key] = name, format(accuracy_score, ".2%"), str(cv_test_accuracy), cv_accuracy_score_mean_std, format(train_time, ".4"), format(test_time, ".4")


def get_classifier_with_parameters(classifier_enum):

    '''
    BernoulliNB: Best parameters found using Grid Search:

    03/02/2020 01:48:20 AM - INFO - Classifier: BERNOULLI_NB, Dataset: TWENTY_NEWS_GROUP
    03/02/2020 01:48:23 AM - INFO -

    Performing grid search...

    03/02/2020 01:48:23 AM - INFO - 	Parameters:
    03/02/2020 01:48:23 AM - INFO - {'classifier__alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0], 'classifier__binarize': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0], 'classifier__fit_prior': [False, True]}
    03/02/2020 01:48:59 AM - INFO - 	Done in 36.482s
    03/02/2020 01:48:59 AM - INFO - 	Best score: 0.721
    03/02/2020 01:48:59 AM - INFO - 	Best parameters set:
    03/02/2020 01:48:59 AM - INFO - 		classifier__alpha: 0.1
    03/02/2020 01:48:59 AM - INFO - 		classifier__binarize: 0.1
    03/02/2020 01:48:59 AM - INFO - 		classifier__fit_prior: False

    03/02/2020 01:49:00 AM - INFO - Classifier: BERNOULLI_NB, Dataset: IMDB_REVIEWS
    03/02/2020 01:49:06 AM - INFO -

    Performing grid search...

    03/02/2020 01:49:06 AM - INFO - 	Parameters:
    03/02/2020 01:49:06 AM - INFO - {'classifier__alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0], 'classifier__binarize': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0], 'classifier__fit_prior': [False, True]}
    03/02/2020 01:49:33 AM - INFO - 	Done in 27.066s
    03/02/2020 01:49:33 AM - INFO - 	Best score: 0.380
    03/02/2020 01:49:33 AM - INFO - 	Best parameters set:
    03/02/2020 01:49:33 AM - INFO - 		classifier__alpha: 0.5
    03/02/2020 01:49:33 AM - INFO - 		classifier__binarize: 0.0001
    03/02/2020 01:49:33 AM - INFO - 		classifier__fit_prior: True
    '''


    '''
    TfidfVectorizer(input='content', encoding='utf-8', decode_error='strict', strip_accents=None, 
    lowercase=True, preprocessor=None, tokenizer=None, analyzer='word', stop_words=None, 
    token_pattern='(?u)\b\w\w+\b', ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None, 
    vocabulary=None, binary=False, dtype=<class 'numpy.float64'>, norm='l2', use_idf=True, smooth_idf=True, 
    sublinear_tf=False) = https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

    HashingVectorizer(input='content', encoding='utf-8', decode_error='strict', strip_accents=None, lowercase=True, 
    preprocessor=None, tokenizer=None, stop_words=None, token_pattern='(?u)\b\w\w+\b', ngram_range=(1, 1), analyzer='word', 
    n_features=1048576, binary=False, norm='l2', alternate_sign=True, dtype=<class 'numpy.float64'>)) = 
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html
    
    CountVectorizer(input='content', encoding='utf-8', decode_error='strict', strip_accents=None, lowercase=True, 
    preprocessor=None, tokenizer=None, stop_words=None, token_pattern='(?u)\b\w\w+\b', ngram_range=(1, 1), 
    analyzer='word', max_df=1.0, min_df=1, max_features=None, vocabulary=None, binary=False, dtype=<class 'numpy.int64'>) = 
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    '''

    if classifier_enum == Classifier.BERNOULLI_NB:
        clf = BernoulliNB()
        parameters = {
            'classifier__alpha': [0.1],
            'classifier__binarize': [0.0001],
            'classifier__fit_prior': [False]

            # 'classifier__alpha': [0.1, 0.5],
            # 'classifier__binarize': [0.0001, 0.1],
            # 'classifier__fit_prior': [False, True]
            # ,
            # 'vectorizer__decode_error': ['strict', 'ignore', 'replace'],
            # 'vectorizer__strip_accents': ['ascii', 'unicode', None],
            # 'vectorizer__stop_words': ['english', None],
            # 'vectorizer__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
            # 'vectorizer__analyzer': ['word', 'char', 'char_wb'],
            # 'vectorizer__binary': [False, True]

        }

    return clf, parameters


def run_grid_search(save_logs_in_file):
    if save_logs_in_file:
        if not os.path.exists('feature_extraction_testing'):
            os.mkdir('feature_extraction_testing')
        logging.basicConfig(filename='feature_extraction_testing/all.log', format='%(asctime)s - %(levelname)s - %(message)s',
                            level=logging.INFO, datefmt='%m/%d/%Y %I:%M:%S %p')
    else:
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO,
                            datefmt='%m/%d/%Y %I:%M:%S %p')

    classifier_list = [
        Classifier.BERNOULLI_NB
    ]

    dataset_list = [
        Dataset.TWENTY_NEWS_GROUPS,
        Dataset.IMDB_REVIEWS
    ]

    vectorizer_list = [
        Vectorizer.COUNT_VECTORIZER,
        Vectorizer.HASHING_VECTORIZER,
        Vectorizer.TF_IDF_VECTORIZER
    ]

    logging.info("\n>>> GRID SEARCH")
    for dataset in dataset_list:
        for classifier_enum in classifier_list:
            c_count = 1
            final_classification_table_best_parameters = {}
            for vectorizer_enum in vectorizer_list:

                logging.info("\n")
                logging.info("#" * 80)
                if save_logs_in_file:
                    print("#" * 80)
                logging.info("{})".format(c_count))

                clf, parameters = get_classifier_with_parameters(classifier_enum)

                logging.info("*" * 80)
                logging.info("Classifier: {}, Dataset: {}, Vectorizer: {}".format(classifier_enum.name, dataset.name, vectorizer_enum.name))
                logging.info("*" * 80)
                start = time()
                run_classifier_grid_search(clf, vectorizer_enum, classifier_enum, parameters, dataset, final_classification_table_best_parameters)
                end = time() - start
                logging.info("It took {} seconds".format(end))
                logging.info("*" * 80)

                if save_logs_in_file:
                    print("*" * 80)
                    print("Classifier: {}, Dataset: {}".format(classifier_enum.name, dataset.name))
                    print(clf)
                    print("It took {} seconds".format(end))
                    print("*" * 80)

                logging.info("#" * 80)
                if save_logs_in_file:
                    print("#" * 80)
                c_count = c_count + 1

        logging.info('\n\nFINAL CLASSIFICATION TABLE: {} DATASET, CLASSIFIER WITH BEST PARAMETERS'.format(dataset.name))
        print_final_classification_table(final_classification_table_best_parameters)


def print_final_classification_table(final_classification_table_default_parameters):
    logging.info(
        '| ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | '
        'Training time (seconds) | Test time (seconds) |')
    logging.info(
        '| -- | ------------ | ------------------ | ------------------------------------ | ----------------- | '
        ' ------------------ | ------------------ |')
    for key in sorted(final_classification_table_default_parameters.keys()):
        values = final_classification_table_default_parameters[key]
        logging.info(
            "|  {}  |  {}  |  {}  |  {}  |  {}  |  {}  |  {}  |".format(key, values[0], values[1], values[2],
                                                                        values[3], values[4], values[5]))


if __name__ == '__main__':
    run_grid_search(False)
