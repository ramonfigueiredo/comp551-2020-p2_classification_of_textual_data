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
from utils.dataset_enum import Dataset

warnings.filterwarnings('always')
warnings.filterwarnings('ignore')


def plot_history(history):
    print("Plotting the grapsh")
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy ({})'.format(dataset))
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss ({})'.format(dataset))
    plt.legend()
    plt.show()


def run_deep_learning_using_keras(options):

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
        print("Keras")

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

        history = model.fit(X_train, y_train, epochs=6, verbose=False, validation_data=(X_test, y_test), batch_size=10)

        loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
        print("Training Accuracy: {:.2f}%".format(accuracy*100))
        loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
        print("Testingf}%".format(accuracy*100))

        plt.style.use('ggplot')

        plot_history(history)
