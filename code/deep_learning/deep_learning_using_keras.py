import logging
import os
import warnings
from time import time

import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from datasets.load_dataset import load_dataset
from feature_extraction.nltk_features_extraction import apply_nltk_feature_extraction
from feature_extraction.vectorizer import extract_text_features
from metrics.ml_metrics import print_classification_report, print_ml_metrics, print_confusion_matrix
from plotting.plot import plot_history
from utils.dataset_enum import Dataset

warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

'''
Referece: Practical Text Classification With Python and Keras
https://realpython.com/python-keras-text-classification/
'''


def manage_logs(options):
    if options.save_logs_in_file:
        if not os.path.exists('logs'):
            os.mkdir('logs')
        logging.basicConfig(filename='logs/all.log', format='%(asctime)s - %(levelname)s - %(message)s',
                            level=logging.INFO, datefmt='%m/%d/%Y %I:%M:%S %p')
    else:
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO,
                            datefmt='%m/%d/%Y %I:%M:%S %p')


def get_dataset_list(options):
    dataset_list = []
    if options.dataset == Dataset.TWENTY_NEWS_GROUPS.name:
        dataset_list.append(Dataset.TWENTY_NEWS_GROUPS.name)
    elif options.dataset == Dataset.IMDB_REVIEWS.name:
        dataset_list.append(Dataset.IMDB_REVIEWS.name)
    else:
        dataset_list.append(Dataset.TWENTY_NEWS_GROUPS.name)
        dataset_list.append(Dataset.IMDB_REVIEWS.name)
    return dataset_list


def one_hot_enconder(dataset, options, y_test, y_train):
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

    return y_test, y_train


def print_results(dataset, algorithm_name, training_loss, training_accuracy, test_loss, test_accuracy, training_time, test_time):
    print("\tDataset: {}".format(dataset))
    print("\tAlgorithm: {}".format(algorithm_name))
    print("\tTraining loss: {:.4f}".format(training_loss))
    print("\tTraining accuracy score: {:.2f}%".format(training_accuracy * 100))
    print("\tTest loss: {:.4f}".format(test_loss))
    print("\tTest accuracy score: {:.2f}%".format(test_accuracy * 100))
    print("\tTraining time: {:.4f}".format(training_time))
    print("\tTest time: {:.4f}".format(test_time))


def add_output_layer(dataset, model, options):
    if dataset == Dataset.IMDB_REVIEWS.name and options.use_imdb_multi_class_labels:
        model.add(layers.Dense(7, activation='sigmoid'))
    elif dataset == Dataset.TWENTY_NEWS_GROUPS.name:
        model.add(layers.Dense(19, activation='sigmoid'))
    else:
        model.add(layers.Dense(1, activation='sigmoid'))


def compile_model(model):
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())


def run_deep_learning_KerasDL1(options):
    manage_logs(options)

    logging.info("Program started...")

    dataset_list = get_dataset_list(options)

    results = {}
    for dataset in dataset_list:

        X_train, y_train, X_test, y_test, target_names, data_train_size_mb, data_test_size_mb = load_dataset(dataset, options)

        y_test, y_train = one_hot_enconder(dataset, options, y_test, y_train)

        vectorizer, X_train, X_test = extract_text_features(X_train, X_test, options, data_train_size_mb, data_test_size_mb)

        input_dim = X_train.shape[1]  # Number of features

        model = Sequential()
        model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
        add_output_layer(dataset, model, options)
        compile_model(model)

        if not options.epochs:
            if dataset == Dataset.TWENTY_NEWS_GROUPS.name:
                epochs = 12
            elif dataset == Dataset.IMDB_REVIEWS.name and options.use_imdb_multi_class_labels:
                epochs = 2
            else:  # IMDB_REVIEWS using binary classification
                epochs = 1

        start = time()
        if not options.epochs:
            print('\n\nNUMBER OF EPOCHS USED: {}\n'.format(epochs))
            history = model.fit(X_train, y_train, epochs=epochs, verbose=False, validation_data=(X_test, y_test), batch_size=10)
        else:
            print('\n\nNUMBER OF EPOCHS USED: {}\n'.format(options.epochs))
            history = model.fit(X_train, y_train, epochs=options.epochs, verbose=False, validation_data=(X_test, y_test), batch_size=10)
        training_time = time() - start


        training_loss, training_accuracy = model.evaluate(X_train, y_train, verbose=False)

        start = time()
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=False)
        test_time = time() - start

        algorithm_name = "Deep Learning using Keras 1 (KERAS_DL1)"

        print_results(dataset, algorithm_name, training_loss, training_accuracy, test_loss, test_accuracy, training_time, test_time)

        plt.style.use('ggplot')

        if dataset == Dataset.TWENTY_NEWS_GROUPS.name:
            plot_history(history, 'KERAS_DL1', '20 NEWS')
        elif dataset == Dataset.IMDB_REVIEWS.name:
            plot_history(history, 'KERAS_DL1', 'IMDB')

        print('\n')

        results[dataset] = dataset, algorithm_name, training_loss, training_accuracy, test_accuracy, training_time, test_time

    return results

'''
IMDB Review - Deep Model
Reference: https://www.kaggle.com/nilanml/imdb-review-deep-model-94-89-accuracy
'''
def run_deep_learning_KerasDL2(options):
    logging.info("Program started...")

    dataset_list = get_dataset_list(options)

    results = {}
    for dataset in dataset_list:

        X_train, y_train, X_test, y_test, target_names, data_train_size_mb, data_test_size_mb = load_dataset(dataset, options)

        X_train = apply_nltk_feature_extraction(X_train, options, label='X_train')
        X_test = apply_nltk_feature_extraction(X_test, options, label='X_test')

        y_test, y_train = one_hot_enconder(dataset, options, y_test, y_train)

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
        add_output_layer(dataset, model, options)
        compile_model(model)

        batch_size = 100

        if not options.epochs:
            if dataset == Dataset.TWENTY_NEWS_GROUPS.name:
                epochs = 15
            elif dataset == Dataset.IMDB_REVIEWS.name and options.use_imdb_multi_class_labels:
                epochs = 2
            else:  # IMDB_REVIEWS using binary classification
                epochs = 3

        # Test the model
        print('\t===> Tokenizer: fit_on_texts(X_test)')
        list_sentences_test = X_test
        list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
        print('\t===> X_test = pad_sequences(list_sentences_test, maxlen={})'.format(max_features))
        X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)

        # Train the model
        start = time()
        if not options.epochs:
            print('\n\nNUMBER OF EPOCHS USED: {}\n'.format(epochs))
            history = model.fit(X_t, y, batch_size=batch_size, epochs=epochs, validation_data=(X_te, y_test))
        else:
            print('\n\nNUMBER OF EPOCHS USED: {}\n'.format(options.epochs))
            history = model.fit(X_t, y, batch_size=batch_size, epochs=options.epochs, validation_data=(X_te, y_test))
        training_time = time() - start

        print('\t=====> Test the model: model.predict()')
        prediction = model.predict(X_te)
        y_pred = (prediction > 0.5)

        print_classification_report(options, y_pred, y_test, target_names)

        print_ml_metrics(options, y_pred, y_test)

        print_confusion_matrix(options, y_pred, y_test)

        training_loss, training_accuracy = model.evaluate(X_t, y, verbose=False)

        start = time()
        test_loss, test_accuracy = model.evaluate(X_te, y_test, verbose=False)
        test_time = time() - start

        algorithm_name = "Deep Learning using Keras 2 (KERAS_DL2)"

        print_results(dataset, algorithm_name, training_loss, training_accuracy, test_loss, test_accuracy, training_time, test_time)
        plt.style.use('ggplot')

        if dataset == Dataset.TWENTY_NEWS_GROUPS.name:
            plot_history(history, 'KERAS_DL2', '20 NEWS')
        elif dataset == Dataset.IMDB_REVIEWS.name:
            plot_history(history, 'KERAS_DL2', 'IMDB')

        print('\n')

        results[dataset] = dataset, algorithm_name, training_loss, training_accuracy, test_accuracy, training_time, test_time

    return results
