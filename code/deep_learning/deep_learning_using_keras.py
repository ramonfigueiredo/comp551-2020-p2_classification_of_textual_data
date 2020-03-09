# Ignore  the warnings
import warnings
from time import time

from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from feature_extraction.nltk_features_extraction import apply_nltk_feature_extraction

warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

from metrics.ml_metrics import print_classification_report, print_ml_metrics, print_confusion_matrix

import logging
import os
# Ignore  the warnings
import warnings

import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from datasets.load_dataset import load_dataset
from feature_extraction.vectorizer import extract_text_features
from feature_selection.select_k_best import select_k_best_using_chi2
from plotting.plot import plot_history
from utils.dataset_enum import Dataset

warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

'''
Referece: Practical Text Classification With Python and Keras
https://realpython.com/python-keras-text-classification/
'''

def run_deep_learning_KerasDL1(options):

    if options.save_logs_in_file:
        if not os.path.exists('logs'):
            os.mkdir('logs')
        logging.basicConfig(filename='logs/all.log', format='%(asctime)s - %(levelname)s - %(message)s',
                            level=logging.INFO, datefmt='%m/%d/%Y %I:%M:%S %p')
    else:
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO,
                            datefmt='%m/%d/%Y %I:%M:%S %p')

    logging.info("Program started...")

    dataset_list = []

    if options.dataset == Dataset.TWENTY_NEWS_GROUPS.name:
        dataset_list.append(Dataset.TWENTY_NEWS_GROUPS.name)
    elif options.dataset == Dataset.IMDB_REVIEWS.name:
        dataset_list.append(Dataset.IMDB_REVIEWS.name)
    else:
        dataset_list.append(Dataset.TWENTY_NEWS_GROUPS.name)
        dataset_list.append(Dataset.IMDB_REVIEWS.name)

    results = {}
    for dataset in dataset_list:

        X_train, y_train, X_test, y_test, target_names, data_train_size_mb, data_test_size_mb = load_dataset(dataset,
                                                                                                             options)

        if (dataset == Dataset.IMDB_REVIEWS.name and options.use_imdb_multi_class_labels) \
                or dataset == Dataset.TWENTY_NEWS_GROUPS.name:
            le_train = LabelEncoder()
            y_train = le_train.fit_transform(y_train)

            le_test = LabelEncoder()
            y_test = le_test.fit_transform(y_test)

            oh_train = OneHotEncoder(categories='auto', dtype=np.float, sparse=False, drop='first')
            y_train = y_train.reshape(len(y_train), 1)
            y_train = oh_train.fit_transform(y_train)

            oh_test = OneHotEncoder(categories='auto', dtype=np.float, sparse=False, drop='first')
            y_test = y_test.reshape(len(y_test), 1)
            y_test = oh_test.fit_transform(y_test)

        vectorizer, X_train, X_test = extract_text_features(X_train, X_test, options, data_train_size_mb,
                                                            data_test_size_mb)

        if options.use_hashing:
            feature_names = None
        else:
            feature_names = vectorizer.get_feature_names()

        if options.chi2_select:
            select_k_best_using_chi2(X_train, y_train, X_test, feature_names, options)

        print('=' * 80)
        if dataset == Dataset.IMDB_REVIEWS.name and options.use_imdb_multi_class_labels:
            layer_name = "Dense(7, activation='sigmoid')"
        elif dataset == Dataset.TWENTY_NEWS_GROUPS.name:
            layer_name = "Dense(19, activation='sigmoid')"
        else:
            layer_name = "Dense(1, activation='sigmoid')"

        print("KERAS DEEP LEARNING MODEL"
              "\nUsing layers:"
              "\n\t==> Dense(10, input_dim=input_dim, activation='relu')"
              "\n\t==> {}"
              "\nCompile option:"
              "\n\t==> model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])".format(layer_name)
              )

        input_dim = X_train.shape[1]  # Number of features

        model = Sequential()
        model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))

        if dataset == Dataset.IMDB_REVIEWS.name and options.use_imdb_multi_class_labels:
            model.add(layers.Dense(7, activation='sigmoid'))
        elif dataset == Dataset.TWENTY_NEWS_GROUPS.name:
            model.add(layers.Dense(19, activation='sigmoid'))
        else:
            model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        epochs=10
        start = time()
        history = model.fit(X_train, y_train, epochs=epochs, verbose=False, validation_data=(X_test, y_test),  batch_size=10)
        training_time = time() - start

        start = time()
        loss, training_accuracy = model.evaluate(X_train, y_train, verbose=False)
        test_time = time() - start

        algorithm_name = "Deep Learning using Keras 1 (KerasDL1)"

        print("\tDataset: {}".format(dataset))
        print("\tAlgorithm: {}".format(algorithm_name))
        print("\tLoss: {:.4f}".format(loss))
        print("\tTraining accuracy score: {:.2f}%".format(training_accuracy * 100))
        loss, test_accuracy = model.evaluate(X_test, y_test, verbose=False)
        print("\tTest Accuracy: {:.2f}%".format(test_accuracy * 100))
        print("\tTraining Time: {:.4f}".format(training_time))
        print("\tTest Time: {:.4f}".format(test_time))

        plt.style.use('ggplot')

        if dataset == Dataset.TWENTY_NEWS_GROUPS.name:
            plot_history(history, 'KerasDL1', '20 NEWS')
        elif dataset == Dataset.IMDB_REVIEWS.name:
            plot_history(history, 'KerasDL1', 'IMDB')

        print('\n')

        results[dataset] = dataset, algorithm_name, loss, training_accuracy, test_accuracy, training_time, test_time

    return results


'''
IMDB Review - Deep Model
Reference: https://www.kaggle.com/nilanml/imdb-review-deep-model-94-89-accuracy
'''
def run_deep_learning_KerasDL2(options):
    # To run and show all metrics: python main.py --run_deep_learning_using_keras --report --confusion_matrix --all_metrics
    if options.save_logs_in_file:
        if not os.path.exists('logs'):
            os.mkdir('logs')
        logging.basicConfig(filename='logs/all.log', format='%(asctime)s - %(levelname)s - %(message)s',
                            level=logging.INFO, datefmt='%m/%d/%Y %I:%M:%S %p')
    else:
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO,
                            datefmt='%m/%d/%Y %I:%M:%S %p')

    logging.info("Program started...")

    dataset_list = []

    if options.dataset == Dataset.TWENTY_NEWS_GROUPS.name:
        dataset_list.append(Dataset.TWENTY_NEWS_GROUPS.name)
    elif options.dataset == Dataset.IMDB_REVIEWS.name:
        dataset_list.append(Dataset.IMDB_REVIEWS.name)
    else:
        dataset_list.append(Dataset.TWENTY_NEWS_GROUPS.name)
        dataset_list.append(Dataset.IMDB_REVIEWS.name)

    results = {}
    for dataset in dataset_list:

        X_train, y_train, X_test, y_test, target_names, data_train_size_mb, data_test_size_mb = load_dataset(dataset, options)

        X_train = apply_nltk_feature_extraction(X_train, options, label='X_train')
        X_test = apply_nltk_feature_extraction(X_test, options, label='X_test')

        if (dataset == Dataset.IMDB_REVIEWS.name or options.use_imdb_multi_class_labels) \
                or dataset == Dataset.TWENTY_NEWS_GROUPS.name:
            le_train = LabelEncoder()
            y_train = le_train.fit_transform(y_train)

            le_test = LabelEncoder()
            y_test = le_test.fit_transform(y_test)

            oh_train = OneHotEncoder(categories='auto', dtype=np.float, sparse=False, drop='first')
            y_train = y_train.reshape(len(y_train), 1)
            y_train = oh_train.fit_transform(y_train)

            oh_test = OneHotEncoder(categories='auto', dtype=np.float, sparse=False, drop='first')
            y_test = y_test.reshape(len(y_test), 1)
            y_test = oh_test.fit_transform(y_test)

        print('=' * 80)
        print("KERAS DEEP LEARNING MODEL"
              "\nUsing layers:"
              "\n\t==> Embedding(max_features, embed_size)"
              "\n\t==> Bidirectional(LSTM(32, return_sequences = True)"
              "\n\t==> GlobalMaxPool1D()"
              "\n\t==> Dense(20, activation=\"relu\")"
              "\n\t==> Dropout(0.05)"
              "\n\t==> Dense(1, activation=\"sigmoid\")"
              "\n\t==> Dense(1, activation=\"sigmoid\")"
              "\nCompile option:"
              "\n\t==> model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])"
              )

        print('\t===> Tokenizer: fit_on_texts(X_train)')
        max_features = 6000
        tokenizer = Tokenizer(num_words=max_features)
        tokenizer.fit_on_texts(X_train)
        list_tokenized_train = tokenizer.texts_to_sequences(X_train)

        maxlen = 130
        print('\t===> X_train = pad_sequences(list_tokenized_train, maxlen={})'.format(max_features))
        X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
        y = y_train

        embed_size = 128
        print('\t===> Create Keras model')
        model = Sequential()
        model.add(Embedding(max_features, embed_size))
        model.add(Bidirectional(LSTM(32, return_sequences=True)))
        model.add(GlobalMaxPool1D())
        model.add(Dense(20, activation="relu"))
        model.add(Dropout(0.05))
        model.add(Dense(1, activation="sigmoid"))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        batch_size = 100

        if dataset == Dataset.TWENTY_NEWS_GROUPS.name:
            epochs = 10 #2  # 2, 3, 4, 5, 10
        elif dataset == Dataset.IMDB_REVIEWS.name:
            epochs = 10 #6

        print('\n\nNUMBER OF EPOCHS USED: {}\n'.format(epochs))

        if dataset == Dataset.IMDB_REVIEWS.name and options.use_imdb_multi_class_labels:
            model.add(layers.Dense(7, activation='sigmoid'))
        elif dataset == Dataset.TWENTY_NEWS_GROUPS.name:
            model.add(layers.Dense(19, activation='sigmoid'))
        else:
            model.add(layers.Dense(1, activation='sigmoid'))

        # Test the model
        print('\t===> Tokenizer: fit_on_texts(X_test)')
        list_sentences_test = X_test
        list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
        print('\t===> X_test = pad_sequences(list_sentences_test, maxlen={})'.format(max_features))
        X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)

        # Train the model
        # history = model.fit(X_t, y, batch_size=batch_size, epochs=epochs, validation_split=0.2)
        print('\t=====> Training the model: model.fit()')
        start = time()
        history = model.fit(X_t, y, batch_size=batch_size, epochs=epochs, validation_data=(X_te, y_test))
        training_time = time() - start

        print('\t=====> Test the model: model.predict()')
        prediction = model.predict(X_te)
        y_pred = (prediction > 0.5)

        print_classification_report(options, y_pred, y_test, target_names)

        print_ml_metrics(options, y_pred, y_test)

        print_confusion_matrix(options, y_pred, y_test)

        start = time()
        loss, training_accuracy = model.evaluate(X_t, y, verbose=False)
        test_time = time() - start
        algorithm_name = "Deep Learning using Keras 2 (KerasDL2)"

        print("\tDataset: {}".format(dataset))
        print("\tAlgorithm: {}".format(algorithm_name))
        print("\tLoss: {:.4f}".format(loss))
        print("\tTraining accuracy score: {:.2f}%".format(training_accuracy * 100))
        loss, test_accuracy = model.evaluate(X_te, y_test, verbose=False)
        print("\tTest Accuracy: {:.2f}%".format(test_accuracy * 100))
        print("\tTraining Time: {:.4f}".format(training_time))
        print("\tTest Time: {:.4f}".format(test_time))

        plt.style.use('ggplot')

        if dataset == Dataset.TWENTY_NEWS_GROUPS.name:
            plot_history(history, 'KerasDL2', '20 NEWS')
        elif dataset == Dataset.IMDB_REVIEWS.name:
            plot_history(history, 'KerasDL2', 'IMDB')

        print('\n')

        results[dataset] = dataset, algorithm_name, loss, training_accuracy, test_accuracy, training_time, test_time

    return results
