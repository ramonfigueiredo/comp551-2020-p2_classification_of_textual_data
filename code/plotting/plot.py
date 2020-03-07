import matplotlib.pyplot as plt

from utils.dataset_enum import Dataset


def plot_results(dataset, options, indices, clf_name_list, accuracy_score_list, training_time_list, test_time_list):
    plt.figure(figsize=(12, 8))
    title = ""
    if dataset == Dataset.TWENTY_NEWS_GROUPS.name:
        if options.twenty_news_with_no_filter:
            title = "{} dataset".format(Dataset.TWENTY_NEWS_GROUPS.name)
            plt.title()
        else:
            title = "{} dataset (removing headers signatures and quoting)".format(
                Dataset.TWENTY_NEWS_GROUPS.name)
            plt.title(title)

    elif dataset == Dataset.IMDB_REVIEWS.name:
        if options.use_imdb_multi_class_labels:
            imdb_classification_type = "Multi-class classification"
        else:
            imdb_classification_type = "Binary classification"

        title = "{} dataset ({})".format(Dataset.IMDB_REVIEWS.name, imdb_classification_type)
        plt.title(title)

    accuracy_score_list = [i*100 for i in accuracy_score_list]
    training_time_list = [i*100 for i in training_time_list]
    test_time_list = [i*100 for i in test_time_list]
    if options.plot_accurary_and_time_together:
        plt.barh(indices + .6, accuracy_score_list, .2, label="Accuracy score (%)", color='green')
        plt.barh(indices + .3, training_time_list, .2, label="Normalized training time (%)", color='c')
        plt.barh(indices, test_time_list, .2, label="Normalized test time (%)", color='darkorange')
    else:
        plt.barh(indices, accuracy_score_list, .75, label="Accuracy score (%)", color='green')

    plt.yticks(())
    plt.legend(loc='best')
    plt.subplots_adjust(left=.25)
    plt.subplots_adjust(top=.95)
    plt.subplots_adjust(bottom=.05)

    for i, c, s, tr, te in zip(indices, clf_name_list, accuracy_score_list, training_time_list, test_time_list):
        if options.plot_accurary_and_time_together:
            plt.text(-30, i + .3, c)
            plt.text(s + 5, i + .6, float("{0:.2f}".format(s)), ha='center', va='center', fontsize=12, weight='bold')
            plt.text(tr + 5, i + .3, float("{0:.2f}".format(tr)), ha='center', va='center', fontsize=12, weight='bold')
            plt.text(te + 5, i, float("{0:.2f}".format(te)), ha='center', va='center', fontsize=12, weight='bold')
        else:
            plt.text(-30, i, c)
            plt.text(s + 5, i, float("{0:.2f}".format(s)), ha='center', va='center', fontsize=12, weight='bold')

    plt.xlim(0, 110)
    plt.tight_layout()
    plt.show()

    return title
