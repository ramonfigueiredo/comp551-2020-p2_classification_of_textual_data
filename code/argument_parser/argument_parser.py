import argparse
import multiprocessing


def get_options():
    parser = argparse.ArgumentParser(prog='main.py',
                                     description='MiniProject 2: Classification of textual data. Authors: Ramon Figueiredo Pessoa, Rafael Gomes Braga, Ege Odaci',
                                     epilog='COMP 551 (001/002), Applied Machine Learning, Winter 2020, McGill University.')
    parser.add_argument("-d", "--dataset",
                        action="store", dest="dataset",
                        help="Dataset used (Options: TWENTY_NEWS_GROUP OR IMDB_REVIEWS). Default: ALL",
                        default='ALL')
    parser.add_argument("-ml", "--ml_algorithm_list",
                        action="append", dest="ml_algorithm_list",
                        help="List of machine learning algorithm to be executed. "
                             "This stores a list of ML algorithms, and appends each algorithm value to the list. "
                             "For example: -ml LINEAR_SVC -ml RANDOM_FOREST_CLASSIFIER, means "
                             "ml_algorithm_list = ['LINEAR_SVC', 'RANDOM_FOREST_CLASSIFIER']. "
                             "(Options of ML algorithms: "
                             "1) ADA_BOOST_CLASSIFIER, 2) BERNOULLI_NB, 3) COMPLEMENT_NB, 4) DECISION_TREE_CLASSIFIER, "
                             "5) GRADIENT_BOOSTING_CLASSIFIER, 6) K_NEIGHBORS_CLASSIFIER, 7) LINEAR_SVC, "
                             "8) LOGISTIC_REGRESSION, 9) MULTINOMIAL_NB, 10) NEAREST_CENTROID, "
                             "11) PASSIVE_AGGRESSIVE_CLASSIFIER, 12) PERCEPTRON, 13) RANDOM_FOREST_CLASSIFIER, "
                             "14) RIDGE_CLASSIFIER). "
                             "Default: None. If ml_algorithm_list is not provided, all ML algorithms will be executed.",
                        default=None)
    parser.add_argument("-use_default_parameters", "--use_classifiers_with_default_parameters",
                        action="store_true", default=False, dest="use_classifiers_with_default_parameters",
                        help="Use classifiers with default parameters. "
                             "Default: False = Use classifiers with best parameters found using grid search.")
    parser.add_argument("-not_shuffle", "--not_shuffle_dataset",
                        action="store_true", default=False, dest="not_shuffle_dataset",
                        help="Read dataset without shuffle data. Default: False")
    parser.add_argument("-n_jobs",
                        action="store", type=int, dest="n_jobs", default=-1,
                        help="The number of CPUs to use to do the computation. "
                             "If the provided number is negative or greater than the number of available CPUs, "
                             "the system will use all the available CPUs. Default: -1 (-1 == all CPUs)")
    parser.add_argument("-cv", "--run_cross_validation",
                        action="store_true", dest="run_cross_validation",
                        help="Run cross validation. Default: False")
    parser.add_argument("-n_splits",
                        action="store", type=int, dest="n_splits", default=5,
                        help="Number of cross validation folds. Default: 5. Must be at least 2. Default: 5")
    parser.add_argument("-required_classifiers", "--use_just_miniproject_classifiers",
                        action="store_true", dest="use_just_miniproject_classifiers",
                        help="Use just the miniproject classifiers (1. LogisticRegression, 2. DecisionTreeClassifier, "
                             "3. LinearSVC (L1), 4. LinearSVC (L2), 5. AdaBoostClassifier, 6. RandomForestClassifier). Default: False")
    parser.add_argument("-news_with_4_classes", "--twenty_news_using_four_categories",
                        action="store_true", default=False, dest="twenty_news_using_four_categories",
                        help="TWENTY_NEWS_GROUP dataset using some categories "
                             "('alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space'). "
                             "Default: False (use all categories). Default: False")
    parser.add_argument("-news_no_filter", "--twenty_news_with_no_filter",
                        action="store_true", default=False, dest="twenty_news_with_no_filter",
                        help="Do not remove newsgroup information that is easily overfit: "
                             "('headers', 'footers', 'quotes'). Default: False")
    parser.add_argument("-imdb_multi_class", "--use_imdb_multi_class_labels",
                        action="store_true", default=False, dest="use_imdb_multi_class_labels",
                        help="Use IMDB multi-class labels (review score: 1, 2, 3, 4, 7, 8, 9, 10)."
                             " If --use_imdb_multi_class_labels is False, the system uses binary classification. "
                             "(0 = neg and 1 = pos) Default: False")
    parser.add_argument("-show_reviews", "--show_imdb_reviews",
                        action="store_true", default=False, dest="show_imdb_reviews",
                        help="Show the IMDB_REVIEWS and respective labels while read the dataset. Default: False")
    parser.add_argument("-r", "--report",
                        action="store_true", dest="report",
                        help="Print a detailed classification report.")
    parser.add_argument("-m", "--all_metrics",
                        action="store_true", dest="all_metrics",
                        help="Print all classification metrics.")
    parser.add_argument("--chi2_select",
                        action="store", type=int, dest="chi2_select",
                        help="Select some number of features using a chi-squared test")
    parser.add_argument("-cm", "--confusion_matrix",
                        action="store_true", dest="print_cm",
                        help="Print the confusion matrix.")
    parser.add_argument("-use_hashing", "--use_hashing_vectorizer", dest="use_hashing",
                        action="store_true", default=False,
                        help="Use a hashing vectorizer. Default: False")
    parser.add_argument("-use_count", "--use_count_vectorizer", dest="use_count_vectorizer",
                        action="store_true", default=False,
                        help="Use a count vectorizer. Default: False")
    parser.add_argument("-n_features", "--n_features_using_hashing", dest="n_features",
                        action="store", type=int, default=2 ** 16,
                        help="n_features when using the hashing vectorizer. Default: 65536")
    parser.add_argument("-plot_time", "--plot_accurary_and_time_together",
                        action="store_true", default=False, dest="plot_accurary_and_time_together",
                        help="Plot training time and test time together with accuracy score. Default: False (Plot just accuracy)")
    parser.add_argument('-save_logs', '--save_logs_in_file', action='store_true', default=False,
                        dest='save_logs_in_file',
                        help='Save logs in a file. Default: False (show logs in the prompt)')
    parser.add_argument('-verbose', '--verbosity', action='store_true', default=False,
                        dest='verbose',
                        help='Increase output verbosity. Default: False')
    parser.add_argument("-random_state",
                        action="store", type=int, dest="random_state", default=0,
                        help="Seed used by the random number generator. Default: 0")
    parser.add_argument("-dl", "--run_deep_learning_using_keras",
                        action="store_true", default=False, dest="run_deep_learning_using_keras",
                        help="Run deep learning using keras. Default: False (Run scikit-learn algorithms)")
    parser.add_argument('-v', '--version', action='version', dest='version', version='%(prog)s 1.0')

    options = parser.parse_args()

    if options.n_jobs > multiprocessing.cpu_count() or (options.n_jobs != -1 and options.n_jobs < 1):
        options.n_jobs = -1  # use all available cpus

    options.dataset = options.dataset.upper().strip()

    show_option(options, parser)

    return options


def show_option(options, parser):

    print(parser.print_help())
    print('=' * 130)

    print('\nRunning with options: ')
    print('\tDataset =', options.dataset)
    print('\tML algorithm list (If ml_algorithm_list is not provided, all ML algorithms will be executed) =',
          options.ml_algorithm_list)
    print('\tUse classifiers with default parameters. '
          'Default: False = Use classifiers with best parameters found using grid search.',
          options.use_classifiers_with_default_parameters)
    print('\tRead dataset without shuffle data =', options.not_shuffle_dataset)
    print('\tThe number of CPUs to use to do the computation. '
          'If the provided number is negative or greater than the number of available CPUs, '
          'the system will use all the available CPUs. Default: -1 (-1 == all CPUs) =', options.n_jobs)
    print('\tRun cross validation. Default: False =', options.run_cross_validation)
    print('\tNumber of cross validation folds. Default: 5 =', options.n_splits)
    print('\tUse just the miniproject classifiers (1. LogisticRegression, 2. DecisionTreeClassifier, '
          '3. LinearSVC, 4. AdaBoostClassifier, 5. RandomForestClassifier) = ',
          options.use_just_miniproject_classifiers)
    print(
        '\tTWENTY_NEWS_GROUPS dataset using some categories (alt.atheism, talk.religion.misc, comp.graphics, sci.space) =',
        options.twenty_news_using_four_categories)
    print('\tDo not remove newsgroup information that is easily overfit (headers, footers, quotes) =',
          options.twenty_news_with_no_filter)
    print('\tUse IMDB multi-class labels (review score: 1, 2, 3, 4, 7, 8, 9, 10). '
          'If --use_imdb_multi_class_labels is False, the system uses binary classification (0 = neg and 1 = pos). '
          'Default: False =', options.use_imdb_multi_class_labels)
    print('\tShow the IMDB_REVIEWS and respective labels while read the dataset =', options.show_imdb_reviews)
    print('\tPrint classification report =', options.report)
    print('\tPrint all classification metrics (accuracy score, precision score, recall score, f1 score, f-beta score, jaccard score) = ', options.all_metrics)
    if options.chi2_select == None:
        chi2_select_option = "No number provided"
    else:
        chi2_select_option = str(options.chi2_select)
    print('\tSelect some number of features using a chi-squared test (For example: --chi2_select 10 = select 10 features using a chi-squared test) = ', chi2_select_option)
    print('\tPrint the confusion matrix =', options.print_cm)
    print('\tUse a hashing vectorizer =', options.use_hashing)
    print('\tUse a count vectorizer =', options.use_count_vectorizer)
    print('\tUse a tf-idf vectorizer =', (not options.use_hashing and not options.use_count_vectorizer))
    print('\tN features when using the hashing vectorizer =', options.n_features)
    print('\tPlot training time and test time together with accuracy score =', options.plot_accurary_and_time_together)
    print('\tSave logs in a file =', options.save_logs_in_file)
    print('\tSeed used by the random number generator (random_state) =', options.random_state)
    print('\tVerbose =', options.verbose)
    print('\tRun deep learning using keras. Default: False (Run scikit-learn algorithms) =', options.run_deep_learning_using_keras)
    print('=' * 130)
    print()