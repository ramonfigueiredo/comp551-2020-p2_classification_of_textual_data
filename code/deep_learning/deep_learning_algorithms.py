from utils.ml_classifiers_enum import Classifier
from deep_learning.deep_learning_using_keras import run_deep_learning_KerasDL1, run_deep_learning_KerasDL2


def run_deep_learning(options):
    deep_learning_algorithm_list = []

    if not options.dl_algorithm_list:
        deep_learning_algorithm_list.append(Classifier.KERAS_DL1.name)
        deep_learning_algorithm_list.append(Classifier.KERAS_DL2.name)

    elif Classifier.KERAS_DL1.name in options.dl_algorithm_list:
        deep_learning_algorithm_list.append(Classifier.KERAS_DL1.name)

    elif Classifier.KERAS_DL2.name in options.dl_algorithm_list:
        deep_learning_algorithm_list.append(Classifier.KERAS_DL2.name)

    if Classifier.KERAS_DL1.name in deep_learning_algorithm_list:
        results_model1 = run_deep_learning_KerasDL1(options)

    if Classifier.KERAS_DL2.name in deep_learning_algorithm_list:
        results_model2 = run_deep_learning_KerasDL2(options)

    print('\n\nFINAL CLASSIFICATION TABLE:\n')
    print(
        '| ID | Dataset | Algorithm | Loss | Training accuracy score (%) | Test accuracy score (%) | Training time (seconds) | Test time (seconds) |')
    print(
        '| --- | ------- | --------- | ---- | --------------------------- | ----------------------- | ----------------------- | ------------------- |')

    count = 1

    if Classifier.KERAS_DL1.name in deep_learning_algorithm_list:
        for key in results_model1:
            dataset, algorithm_name, loss, training_accuracy, test_accuracy, training_time, test_time = results_model1[key]
            print(
                "| {} | {} | {} | {:.4f} | {:.2f} | {:.2f} | {:.4f} | {:.4f} |".format(count, dataset, algorithm_name, loss,
                                                                                       training_accuracy * 100,
                                                                                       test_accuracy * 100, training_time,
                                                                                       test_time))
            count = count + 1

    if Classifier.KERAS_DL2.name in deep_learning_algorithm_list:
        for key in results_model2:
            dataset, algorithm_name, loss, training_accuracy, test_accuracy, training_time, test_time = results_model2[key]
            print(
                "| {} | {} | {} | {:.4f} | {:.2f} | {:.2f} | {:.4f} | {:.4f} |".format(count, dataset, algorithm_name, loss,
                                                                                       training_accuracy * 100,
                                                                                       test_accuracy * 100, training_time,
                                                                                       test_time))
            count = count + 1
