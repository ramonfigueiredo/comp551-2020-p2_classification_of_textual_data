import operator

from sklearn import metrics


def print_confusion_matrix(options, y_pred, y_test):
    if options.print_cm:
        print("confusion matrix:")
        print(metrics.confusion_matrix(y_test, y_pred))



def print_ml_metrics(options, y_pred, y_test):
    if options.all_metrics:
        print("\n\n===> Classification Metrics:\n")
        print('accuracy classification score')
        print('\taccuracy score: ', metrics.accuracy_score(y_test, y_pred))
        print('\taccuracy score (normalize=False): ', metrics.accuracy_score(y_test, y_pred, normalize=False))
        print()
        print('compute the precision')
        print('\tprecision score (average=macro): ', metrics.precision_score(y_test, y_pred, average='macro'))
        print('\tprecision score (average=micro): ', metrics.precision_score(y_test, y_pred, average='micro'))
        print('\tprecision score (average=weighted): ', metrics.precision_score(y_test, y_pred, average='weighted'))
        print('\tprecision score (average=None): ', metrics.precision_score(y_test, y_pred, average=None))
        print('\tprecision score (average=None, zero_division=1): ',
              metrics.precision_score(y_test, y_pred, average=None, zero_division=1))
        print()
        print('compute the precision')
        print('\trecall score (average=macro): ', metrics.recall_score(y_test, y_pred, average='macro'))
        print('\trecall score (average=micro): ', metrics.recall_score(y_test, y_pred, average='micro'))
        print('\trecall score (average=weighted): ', metrics.recall_score(y_test, y_pred, average='weighted'))
        print('\trecall score (average=None): ', metrics.recall_score(y_test, y_pred, average=None))
        print('\trecall score (average=None, zero_division=1): ',
              metrics.recall_score(y_test, y_pred, average=None, zero_division=1))
        print()
        print('compute the F1 score, also known as balanced F-score or F-measure')
        print('\tf1 score (average=macro): ', metrics.f1_score(y_test, y_pred, average='macro'))
        print('\tf1 score (average=micro): ', metrics.f1_score(y_test, y_pred, average='micro'))
        print('\tf1 score (average=weighted): ', metrics.f1_score(y_test, y_pred, average='weighted'))
        print('\tf1 score (average=None): ', metrics.f1_score(y_test, y_pred, average=None))
        print()
        print('compute the F-beta score')
        print('\tf beta score (average=macro): ', metrics.fbeta_score(y_test, y_pred, average='macro', beta=0.5))
        print('\tf beta score (average=micro): ', metrics.fbeta_score(y_test, y_pred, average='micro', beta=0.5))
        print('\tf beta score (average=weighted): ', metrics.fbeta_score(y_test, y_pred, average='weighted', beta=0.5))
        print('\tf beta score (average=None): ', metrics.fbeta_score(y_test, y_pred, average=None, beta=0.5))
        print()
        print('compute the average Hamming loss')
        print('\thamming loss: ', metrics.hamming_loss(y_test, y_pred))
        print()
        print('jaccard similarity coefficient score')
        print('\tjaccard score (average=macro): ', metrics.jaccard_score(y_test, y_pred, average='macro'))
        print('\tjaccard score (average=None): ', metrics.jaccard_score(y_test, y_pred, average=None))
        print()


def print_classification_report(options, y_pred, y_test, target_names):
    if options.report:
        print("\n\n===> Classification Report:\n")
        print(metrics.classification_report(y_test, y_pred, target_names=target_names))


def accuracy_score(y_pred, y_test):
    score = metrics.accuracy_score(y_test, y_pred)
    print("accuracy:   %0.3f" % score)

    return score

def print_final_classification_report(options, results, title):
    print("FINAL CLASSIFICATION TABLE: {}".format(title))

    classifier_name_list = results[0]
    accuracy_score_list = results[1]
    train_time_list = results[2]
    test_time_list = results[3]

    if options.run_cross_validation:
        cross_val_scores = results[4]
        cross_val_accuracy_score_mean_std = results[5]

    if options.run_cross_validation:
        print('| ID | ML Algorithm | Accuracy Score (%) | K-fold Cross Validation (CV) (k = 5) | CV (Mean +/- Std) | '
              'Training time (seconds) | Test time (seconds) |')
        print(
            '| --- | ------------- | ------------------ | ------------------------------------ | ----------------- | '
            ' ------------------ | ------------------ |')
    else:
        print('| ID | ML Algorithm | Accuracy Score (%) | Training time (seconds) | Test time (seconds) |')
        print('| --- | ------------- | ------------------ | ----------------------- | ------------------- |')

    index = 1

    for classifier_name, accuracy_score, train_time, test_time in zip(classifier_name_list, accuracy_score_list,
                                                                      train_time_list, test_time_list):
        if classifier_name in ["Logistic Regression", "Decision Tree Classifier", "Linear SVC (penalty = L2)",
                               "Linear SVC (penalty = L1)", "Ada Boost Classifier", "Random forest"]:
            classifier_name = classifier_name + " [MANDATORY FOR COMP 551, ASSIGNMENT 2]"
        if options.run_cross_validation:
            print("|  {}  |  {}  |  {}  |  {}  |  {}  |  {}  |  {}  |".format(index, classifier_name,
                                                                              format(accuracy_score, ".2%"),
                                                                              cross_val_scores[index - 1],
                                                                              cross_val_accuracy_score_mean_std[
                                                                                  index - 1], format(train_time, ".4"),
                                                                              format(test_time, ".4")))
        else:
            print("|  {}  |  {}  |  {}  |  {}  |  {}  |".format(index, classifier_name,
                                                                format(accuracy_score, ".2%"),
                                                                format(train_time, ".4"),
                                                                format(test_time, ".4")))
        index = index + 1

    print("\n\nBest algorithm:")
    index_max_accuracy_score, accuracy_score = max(enumerate(accuracy_score_list), key=operator.itemgetter(1))
    print("===> {}) {}\n\t\tAccuracy score = {}\t\tTraining time = {}\t\tTest time = {}\n".format(
        index_max_accuracy_score + 1,
        classifier_name_list[index_max_accuracy_score],
        format(accuracy_score_list[index_max_accuracy_score], ".2%"),
        format(train_time_list[index_max_accuracy_score], ".4"),
        format(test_time_list[index_max_accuracy_score], ".4")))
