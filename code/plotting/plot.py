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
    plt.barh(indices, accuracy_score_list, .2, label="score", color='navy')
    if options.plot_accurary_and_time_together:
        plt.barh(indices + .3, training_time_list, .2, label="training time", color='c')
        plt.barh(indices + .6, test_time_list, .2, label="test time", color='darkorange')
    plt.yticks(())
    plt.legend(loc='best')
    plt.subplots_adjust(left=.25)
    plt.subplots_adjust(top=.95)
    plt.subplots_adjust(bottom=.05)

    for i, c, s, tr, te in zip(indices, clf_name_list, accuracy_score_list, training_time_list, test_time_list):
        plt.text(-.3, i, c)
        plt.text(tr / 2, i + .3, round(tr, 2), ha='center', va='center', color='white')
        plt.text(te / 2, i + .6, round(te, 2), ha='center', va='center', color='white')
        plt.text(s / 2, i, round(s, 2), ha='center', va='center', color='white')

    plt.tight_layout()
    plt.show()

    return title
