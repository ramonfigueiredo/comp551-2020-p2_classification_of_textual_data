from time import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_text_features(X_train, X_test, options, data_train_size_mb, data_test_size_mb):
    print("Extracting features from the training data using a vectorizer")
    t0 = time()

    if options.use_hashing:
        vectorizer = HashingVectorizer(stop_words='english', strip_accents='unicode', analyzer='word', binary=True)
        X_train = vectorizer.transform(X_train)

    elif options.use_count_vectorizer:
        vectorizer = CountVectorizer(stop_words='english', strip_accents='unicode', analyzer='word', binary=True)
        X_train = vectorizer.fit_transform(X_train)

    else:
        vectorizer = TfidfVectorizer(stop_words='english', strip_accents='unicode', analyzer='word', binary=True)
        X_train = vectorizer.fit_transform(X_train)

    duration = time() - t0
    print("done in %fs at %0.3fMB/s" % (duration, data_train_size_mb / duration))
    print("n_samples: %d, n_features: %d" % X_train.shape)
    print()

    print("Extracting features from the test data using the same vectorizer")
    t0 = time()
    X_test = vectorizer.transform(X_test)
    duration = time() - t0

    print("done in %fs at %0.3fMB/s" % (duration, data_test_size_mb / duration))
    print("n_samples: %d, n_features: %d" % X_test.shape)
    print()

    return vectorizer, X_train, X_test
