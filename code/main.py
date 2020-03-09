import logging
import os
from time import time

import numpy as np

from argument_parser.argument_parser import get_options
from datasets.load_dataset import load_dataset
from deep_learning.deep_learning_using_keras import run_deep_learning_KerasDL1, run_deep_learning_KerasDL2
from feature_extraction.vectorizer import extract_text_features
from feature_selection.select_k_best import select_k_best_using_chi2
from machine_learning.ml_algorithms import run_all_classifiers
from machine_learning.ml_algorithms import run_just_miniproject_classifiers
from machine_learning.ml_algorithms import run_ml_algorithm_list
from metrics.ml_metrics import print_final_classification_report
from model_selection.grid_search_20newsgroups_and_imdb_using_binary_classification import \
    run_grid_search_20newsgroups_and_imdb_using_binary_classification
from model_selection.grid_search_imdb_using_multi_class_classification import \
    run_grid_search_imdb_using_multi_class_classification
from plotting.plot import plot_results
from utils.dataset_enum import Dataset
from utils.ml_classifiers_enum import validate_ml_list

if __name__ == '__main__':

    options = get_options()

    if options.save_logs_in_file:
        if not os.path.exists('logs'):
            os.mkdir('logs')
        logging.basicConfig(filename='logs/all.log', format='%(asctime)s - %(levelname)s - %(message)s',
                            level=logging.INFO, datefmt='%m/%d/%Y %I:%M:%S %p')
    else:
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO,
                            datefmt='%m/%d/%Y %I:%M:%S %p')

    logging.info("Program started...")

    start = time()

    if options.run_grid_search:
        run_grid_search_20newsgroups_and_imdb_using_binary_classification()
        run_grid_search_imdb_using_multi_class_classification()
    else:
        if options.run_deep_learning_using_keras:
            results_model1 = run_deep_learning_KerasDL1(options)
            results_model2 = run_deep_learning_KerasDL2(options)

            print('\n\nFINAL CLASSIFICATION TABLE:\n')
            print('| ID | Dataset | Algorithm | Loss | Training accuracy score (%) | Test accuracy score (%) | Training time (seconds) | Test time (seconds) |')
            print('| -- | ------- | --------- | ---- | --------------------------- | ----------------------- | ----------------------- | ------------------- |')
            count = 1
            for key in results_model1:
                dataset, algorithm_name, loss, training_accuracy, test_accuracy, training_time, test_time = results_model1[key]
                print("| {} | {} | {} | {:.4f} | {:.2f} | {:.2f} | {:.4f} | {:.4f} |".format(count, dataset, algorithm_name, loss, training_accuracy * 100, test_accuracy * 100, training_time, test_time))
                count = count + 1

            for key in results_model2:
                dataset, algorithm_name, loss, training_accuracy, test_accuracy, training_time, test_time = results_model2[key]
                print("| {} | {} | {} | {:.4f} | {:.2f} | {:.2f} | {:.4f} | {:.4f} |".format(count, dataset, algorithm_name, loss, training_accuracy * 100, test_accuracy * 100, training_time, test_time))
                count = count + 1

            print("\nMODEL 1: KERAS DEEP LEARNING MODEL"
                  "\nUsing layers:"
                  "\n\t==> Dense(10, input_dim=input_dim, activation='relu')"
                  "\n\t==> Dense(7=IMDB_REVIEWS multi-label or 19 = TWENTY_NEWS_GROUPS or 1 to IMDB_REVIEWS binary label, activation='sigmoid')"
                  "\nCompile option:"
                  "\n\t==> model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])"
                  )

            print("\nMODEL 2: KERAS DEEP LEARNING MODEL"
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

        else:
            dataset_list = []

            if options.dataset == Dataset.TWENTY_NEWS_GROUPS.name:
                dataset_list.append(Dataset.TWENTY_NEWS_GROUPS.name)
            elif options.dataset == Dataset.IMDB_REVIEWS.name:
                dataset_list.append(Dataset.IMDB_REVIEWS.name)
            else:
                dataset_list.append(Dataset.TWENTY_NEWS_GROUPS.name)
                dataset_list.append(Dataset.IMDB_REVIEWS.name)

            for dataset in dataset_list:

                X_train, y_train, X_test, y_test, target_names, data_train_size_mb, data_test_size_mb = load_dataset(dataset, options)

                vectorizer, X_train, X_test = extract_text_features(X_train, X_test, options, data_train_size_mb, data_test_size_mb)

                if options.use_hashing:
                    feature_names = None
                else:
                    feature_names = vectorizer.get_feature_names()

                if options.chi2_select:
                    select_k_best_using_chi2(X_train, y_train, X_test, feature_names, options)

                results = []
                if options.use_just_miniproject_classifiers:
                    results = run_just_miniproject_classifiers(options, X_train, y_train, X_test, y_test,
                                                               options.use_classifiers_with_default_parameters,
                                                               options.use_imdb_multi_class_labels, dataset,
                                                               feature_names, target_names, results)
                elif options.ml_algorithm_list:
                    validate_ml_list(options.ml_algorithm_list)
                    results = run_ml_algorithm_list(options, X_train, y_train, X_test, y_test, options.ml_algorithm_list,
                                                    options.use_classifiers_with_default_parameters,
                                                    options.use_imdb_multi_class_labels, dataset,
                                                    feature_names, target_names, results)
                else:
                    results = run_all_classifiers(options, X_train, y_train, X_test, y_test,
                                                  options.use_classifiers_with_default_parameters,
                                                  options.use_imdb_multi_class_labels, dataset,
                                                  feature_names, target_names, results)

                indices = np.arange(len(results))

                if options.run_cross_validation:
                    results = [[x[i] for x in results] for i in range(6)]
                    clf_name_list, accuracy_score_list, training_time_list, test_time_list, cross_val_score_list, cross_val_accuracy_score_mean_std_list = results
                else:
                    results = [[x[i] for x in results] for i in range(4)]
                    clf_name_list, accuracy_score_list, training_time_list, test_time_list = results

                training_time_list = np.array(training_time_list) / np.max(training_time_list)
                test_time_list = np.array(test_time_list) / np.max(test_time_list)

                title = plot_results(dataset, options, indices, clf_name_list, accuracy_score_list, training_time_list,
                                     test_time_list)

                print_final_classification_report(options, results, title)

    print('\n\nDONE!')

    print("Program finished. It took {} seconds".format(time() - start))
