## Two Deep Learning approaches using Keras: Loss, Training Accuracy, Test Accuracy, Training Time, Test Time

| ID | Dataset | Algorithm | Loss | Training accuracy score (%) | Test accuracy score (%) | Training time (seconds) | Test time (seconds) |
| -- | ------- | --------- | ---- | --------------------------- | ----------------------- | ----------------------- | ------------------- |
| 1 | TWENTY_NEWS_GROUPS | Deep Learning using Keras 1 (KerasDL1) | 0.9973 | 11.05 | 96.80% | 98.2616 | 5.5369 |
| 2 | IMDB_REVIEWS | Deep Learning using Keras 1 (KerasDL1) | 1.0000 | 93.46 | 82.90% | 175.9112 | 3.4582 |

### Deep Learning using Keras 1 (KerasDL1)

![TWENTY_NEWS_GROUPS: Loss, Training Accuracy, Test Accuracy, Training Time, Test Time](https://github.com/ramonfigueiredopessoa/comp551-2020-p2_classification_of_textual_data/blob/master/code/results/deep_learning_using_keras/model1/TWENTY_NEWS_GROUPS_training_and_validation_accuracy_and_Loss.png)

![IMDB_REVIEWS: Loss, Training Accuracy, Test Accuracy, Training Time, Test Time](https://github.com/ramonfigueiredopessoa/comp551-2020-p2_classification_of_textual_data/blob/master/code/results/deep_learning_using_keras/model1//IMDB_REVIEWS_training_and_validation_accuracy_and_Loss.png)


### Learning using Keras 1 (KerasDL1)

![TWENTY_NEWS_GROUPS: Loss, Training Accuracy, Test Accuracy, Training Time, Test Time](https://github.com/ramonfigueiredopessoa/comp551-2020-p2_classification_of_textual_data/blob/master/code/results/deep_learning_using_keras/model2/10-epochs-using-NLTK_feature_extraction-TWENTY_NEWS_GROUPS_training_and_validation_accuracy_and_Loss)

![IMDB_REVIEWS: Loss, Training Accuracy, Test Accuracy, Training Time, Test Time](https://github.com/ramonfigueiredopessoa/comp551-2020-p2_classification_of_textual_data/blob/master/code/results/deep_learning_using_keras/)


#### Computer settings used to run

* Operating system: Ubuntu 18.04.4 LTS (64-bit)
* Processor: Intel® Core™ i7-7700 CPU @ 3.60GHz × 8 
* Memory: 32 GB

#### All logs 

```
/home/ets-crchum/virtual_envs/comp551_p2/bin/python /home/ets-crchum/github/comp551-2020-p2_classification_of_textual_data/code/main.py -dl
Using TensorFlow backend.
2020-03-09 19:04:52.948201: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libnvinfer.so.6'; dlerror: libnvinfer.so.6: cannot open shared object file: No such file or directory
2020-03-09 19:04:52.948263: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libnvinfer_plugin.so.6'; dlerror: libnvinfer_plugin.so.6: cannot open shared object file: No such file or directory
2020-03-09 19:04:52.948269: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:30] Cannot dlopen some TensorRT libraries. If you would like to use Nvidia GPU with TensorRT, please make sure the missing libraries mentioned above are installed properly.
[nltk_data] Downloading package wordnet to /home/ets-
[nltk_data]     crchum/nltk_data...
[nltk_data]   Package wordnet is already up-to-date!
usage: main.py [-h] [-d DATASET] [-ml ML_ALGORITHM_LIST]
               [-use_default_parameters] [-not_shuffle] [-n_jobs N_JOBS] [-cv]
               [-n_splits N_SPLITS] [-required_classifiers]
               [-news_with_4_classes] [-news_no_filter] [-imdb_multi_class]
               [-show_reviews] [-r] [-m] [--chi2_select CHI2_SELECT] [-cm]
               [-use_hashing] [-use_count] [-n_features N_FEATURES]
               [-plot_time] [-save_logs] [-verbose]
               [-random_state RANDOM_STATE] [-dl] [-gs] [-v]

MiniProject 2: Classification of textual data. Authors: Ramon Figueiredo
Pessoa, Rafael Gomes Braga, Ege Odaci

optional arguments:
  -h, --help            show this help message and exit
  -d DATASET, --dataset DATASET
                        Dataset used (Options: TWENTY_NEWS_GROUP OR
                        IMDB_REVIEWS). Default: ALL
  -ml ML_ALGORITHM_LIST, --ml_algorithm_list ML_ALGORITHM_LIST
                        List of machine learning algorithm to be executed.
                        This stores a list of ML algorithms, and appends each
                        algorithm value to the list. For example: -ml
                        LINEAR_SVC -ml RANDOM_FOREST_CLASSIFIER, means
                        ml_algorithm_list = ['LINEAR_SVC',
                        'RANDOM_FOREST_CLASSIFIER']. (Options of ML
                        algorithms: 1) ADA_BOOST_CLASSIFIER, 2) BERNOULLI_NB,
                        3) COMPLEMENT_NB, 4) DECISION_TREE_CLASSIFIER, 5)
                        GRADIENT_BOOSTING_CLASSIFIER, 6)
                        K_NEIGHBORS_CLASSIFIER, 7) LINEAR_SVC, 8)
                        LOGISTIC_REGRESSION, 9) MULTINOMIAL_NB, 10)
                        NEAREST_CENTROID, 11) PASSIVE_AGGRESSIVE_CLASSIFIER,
                        12) PERCEPTRON, 13) RANDOM_FOREST_CLASSIFIER, 14)
                        RIDGE_CLASSIFIER, 15) MAJORITY_VOTING_CLASSIFIER
                        (using COMPLEMENT_NB, RIDGE_CLASSIFIER, LINEAR_SVC,
                        LOGISTIC_REGRESSION, PASSIVE_AGGRESSIVE_CLASSIFIER,
                        RANDOM_FOREST_CLASSIFIER), 16) SOFT_VOTING_CLASSIFIER
                        (using COMPLEMENT_NB, LOGISTIC_REGRESSION,
                        MULTINOMIAL_NB, RANDOM_FOREST_CLASSIFIER), 17)
                        STACKING_CLASSIFIER (using COMPLEMENT_NB,
                        RIDGE_CLASSIFIER, LINEAR_SVC, LOGISTIC_REGRESSION,
                        PASSIVE_AGGRESSIVE_CLASSIFIER,
                        RANDOM_FOREST_CLASSIFIER,
                        final_estimator=LINEAR_SVC)). Default: None. If
                        ml_algorithm_list is not provided, all ML algorithms
                        will be executed.
  -use_default_parameters, --use_classifiers_with_default_parameters
                        Use classifiers with default parameters. Default:
                        False = Use classifiers with best parameters found
                        using grid search.
  -not_shuffle, --not_shuffle_dataset
                        Read dataset without shuffle data. Default: False
  -n_jobs N_JOBS        The number of CPUs to use to do the computation. If
                        the provided number is negative or greater than the
                        number of available CPUs, the system will use all the
                        available CPUs. Default: -1 (-1 == all CPUs)
  -cv, --run_cross_validation
                        Run cross validation. Default: False
  -n_splits N_SPLITS    Number of cross validation folds. Default: 5. Must be
                        at least 2. Default: 5
  -required_classifiers, --use_just_miniproject_classifiers
                        Use just the miniproject classifiers (1.
                        LogisticRegression, 2. DecisionTreeClassifier, 3.
                        LinearSVC (L1), 4. LinearSVC (L2), 5.
                        AdaBoostClassifier, 6. RandomForestClassifier).
                        Default: False
  -news_with_4_classes, --twenty_news_using_four_categories
                        TWENTY_NEWS_GROUP dataset using some categories
                        ('alt.atheism', 'talk.religion.misc', 'comp.graphics',
                        'sci.space'). Default: False (use all categories).
                        Default: False
  -news_no_filter, --twenty_news_with_no_filter
                        Do not remove newsgroup information that is easily
                        overfit: ('headers', 'footers', 'quotes'). Default:
                        False
  -imdb_multi_class, --use_imdb_multi_class_labels
                        Use IMDB multi-class labels (review score: 1, 2, 3, 4,
                        7, 8, 9, 10). If --use_imdb_multi_class_labels is
                        False, the system uses binary classification. (0 = neg
                        and 1 = pos) Default: False
  -show_reviews, --show_imdb_reviews
                        Show the IMDB_REVIEWS and respective labels while read
                        the dataset. Default: False
  -r, --report          Print a detailed classification report.
  -m, --all_metrics     Print all classification metrics.
  --chi2_select CHI2_SELECT
                        Select some number of features using a chi-squared
                        test
  -cm, --confusion_matrix
                        Print the confusion matrix.
  -use_hashing, --use_hashing_vectorizer
                        Use a hashing vectorizer. Default: False
  -use_count, --use_count_vectorizer
                        Use a count vectorizer. Default: False
  -n_features N_FEATURES, --n_features_using_hashing N_FEATURES
                        n_features when using the hashing vectorizer. Default:
                        65536
  -plot_time, --plot_accurary_and_time_together
                        Plot training time and test time together with
                        accuracy score. Default: False (Plot just accuracy)
  -save_logs, --save_logs_in_file
                        Save logs in a file. Default: False (show logs in the
                        prompt)
  -verbose, --verbosity
                        Increase output verbosity. Default: False
  -random_state RANDOM_STATE
                        Seed used by the random number generator. Default: 0
  -dl, --run_deep_learning_using_keras
                        Run deep learning using keras. Default: False (Run
                        scikit-learn algorithms)
  -gs, --run_grid_search
                        Run grid search for all datasets (TWENTY_NEWS_GROUPS,
                        IMDB_REVIEWS binary labels and IMDB_REVIEWS multi-
                        class labels), and all classifiers (1)
                        ADA_BOOST_CLASSIFIER, 2) BERNOULLI_NB, 3)
                        COMPLEMENT_NB, 4) DECISION_TREE_CLASSIFIER, 5)
                        GRADIENT_BOOSTING_CLASSIFIER, 6)
                        K_NEIGHBORS_CLASSIFIER, 7) LINEAR_SVC, 8)
                        LOGISTIC_REGRESSION, 9) MULTINOMIAL_NB, 10)
                        NEAREST_CENTROID, 11) PASSIVE_AGGRESSIVE_CLASSIFIER,
                        12) PERCEPTRON, 13) RANDOM_FOREST_CLASSIFIER, 14)
                        RIDGE_CLASSIFIER). Default: False (run scikit-learn
                        algorithms or deep learning algorithms). Note: this
                        takes many hours to execute.
  -v, --version         show program's version number and exit

COMP 551 (001/002), Applied Machine Learning, Winter 2020, McGill University.
None
==================================================================================================================================

Running with options: 
	Dataset = ALL
	ML algorithm list (If ml_algorithm_list is not provided, all ML algorithms will be executed) = None
	Use classifiers with default parameters. Default: False = Use classifiers with best parameters found using grid search. False
	Read dataset without shuffle data = False
	The number of CPUs to use to do the computation. If the provided number is negative or greater than the number of available CPUs, the system will use all the available CPUs. Default: -1 (-1 == all CPUs) = -1
	Run cross validation. Default: False = False
	Number of cross validation folds. Default: 5 = 5
	Use just the miniproject classifiers (1. LogisticRegression, 2. DecisionTreeClassifier, 3. LinearSVC, 4. AdaBoostClassifier, 5. RandomForestClassifier) =  False
	TWENTY_NEWS_GROUPS dataset using some categories (alt.atheism, talk.religion.misc, comp.graphics, sci.space) = False
	Do not remove newsgroup information that is easily overfit (headers, footers, quotes) = False
	Use IMDB multi-class labels (review score: 1, 2, 3, 4, 7, 8, 9, 10). If --use_imdb_multi_class_labels is False, the system uses binary classification (0 = neg and 1 = pos). Default: False = False
	Show the IMDB_REVIEWS and respective labels while read the dataset = False
	Print classification report = False
	Print all classification metrics (accuracy score, precision score, recall score, f1 score, f-beta score, jaccard score) =  False
	Select some number of features using a chi-squared test (For example: --chi2_select 10 = select 10 features using a chi-squared test) =  No number provided
	Print the confusion matrix = False
	Use a hashing vectorizer = False
	Use a count vectorizer = False
	Use a tf-idf vectorizer = True
	N features when using the hashing vectorizer = 65536
	Plot training time and test time together with accuracy score = False
	Save logs in a file = False
	Verbose = False
	Seed used by the random number generator (random_state) = 0
	Run deep learning using keras. Default: False (Run scikit-learn algorithms) = True
	Run grid search for all datasets (TWENTY_NEWS_GROUPS, IMDB_REVIEWS binary labels and IMDB_REVIEWS multi-class labels), and all 14 classifiers. Default: False (run scikit-learn algorithms or deep learning algorithms). Note: this takes many hours to execute. = False
==================================================================================================================================

Loading TWENTY_NEWS_GROUPS dataset for categories:
03/09/2020 07:04:53 PM - INFO - Program started...
03/09/2020 07:04:53 PM - INFO - Program started...
data loaded
11314 documents - 13.782MB (training set)
7532 documents - 8.262MB (test set)
20 categories

Extracting features from the training data using a vectorizer
done in 1.203393s at 11.453MB/s
n_samples: 11314, n_features: 101321

Extracting features from the test data using the same vectorizer
done in 0.644865s at 12.811MB/s
n_samples: 7532, n_features: 101321

================================================================================
KERAS DEEP LEARNING MODEL
Using layers:
	==> Dense(10, input_dim=input_dim, activation='relu')
	==> Dense(19, activation='sigmoid')
Compile option:
	==> model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
2020-03-09 19:04:57.304671: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
2020-03-09 19:04:57.327938: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-03-09 19:04:57.328547: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1555] Found device 0 with properties: 
pciBusID: 0000:01:00.0 name: GeForce GTX 1080 computeCapability: 6.1
coreClock: 1.7335GHz coreCount: 20 deviceMemorySize: 7.93GiB deviceMemoryBandwidth: 298.32GiB/s
2020-03-09 19:04:57.328610: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcudart.so.10.1'; dlerror: libcudart.so.10.1: cannot open shared object file: No such file or directory
2020-03-09 19:04:57.328653: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcublas.so.10'; dlerror: libcublas.so.10: cannot open shared object file: No such file or directory
2020-03-09 19:04:57.328693: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcufft.so.10'; dlerror: libcufft.so.10: cannot open shared object file: No such file or directory
2020-03-09 19:04:57.328733: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcurand.so.10'; dlerror: libcurand.so.10: cannot open shared object file: No such file or directory
2020-03-09 19:04:57.328772: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcusolver.so.10'; dlerror: libcusolver.so.10: cannot open shared object file: No such file or directory
2020-03-09 19:04:57.328811: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcusparse.so.10'; dlerror: libcusparse.so.10: cannot open shared object file: No such file or directory
2020-03-09 19:04:57.330669: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2020-03-09 19:04:57.330679: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1592] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
Skipping registering GPU devices...
2020-03-09 19:04:57.330849: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-03-09 19:04:57.351477: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 3600000000 Hz
2020-03-09 19:04:57.351906: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x632e490 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-03-09 19:04:57.351932: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2020-03-09 19:04:57.421487: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-03-09 19:04:57.422130: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x62bf000 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2020-03-09 19:04:57.422141: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): GeForce GTX 1080, Compute Capability 6.1
2020-03-09 19:04:57.422232: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1096] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-03-09 19:04:57.422238: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1102]      
TWENTY_NEWS_GROUPS
	Training accuracy score: 99.73%
	Loss: 0.0154
	Test Accuracy: 96.80%


Loading IMDB_REVIEWS dataset:

===> Reading files from /home/ets-crchum/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/train/neg

===> Reading files from /home/ets-crchum/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/train/pos

===> Reading files from /home/ets-crchum/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/test/neg

===> Reading files from /home/ets-crchum/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/test/pos
data loaded
25000 documents - 33.133MB (training set)
25000 documents - 32.351MB (test set)

Extracting features from the training data using a vectorizer
done in 3.028207s at 10.941MB/s
n_samples: 25000, n_features: 74170

Extracting features from the test data using the same vectorizer
done in 2.973696s at 10.879MB/s
n_samples: 25000, n_features: 74170

================================================================================
KERAS DEEP LEARNING MODEL
Using layers:
	==> Dense(10, input_dim=input_dim, activation='relu')
	==> Dense(1, activation='sigmoid')
Compile option:
	==> model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
IMDB_REVIEWS
	Training accuracy score: 100.00%
	Loss: 0.0002
	Test Accuracy: 82.90%


03/09/2020 07:09:55 PM - INFO - Program started...
Loading TWENTY_NEWS_GROUPS dataset for categories:
data loaded
11314 documents - 13.782MB (training set)
7532 documents - 8.262MB (test set)
20 categories


Applying NLTK feature extraction: X_train
	==> Replace unused UNICODE characters. 
	==> Text in lower case. 
	==> Apply Lemmatize using WordNet's built-in morphy function. Returns the input word unchanged if it cannot be found in WordNet 
	==> Remove English stop words
	Done: NLTK feature extraction (replace unused UNICODE characters, text in lower case, apply Lemmatize using WordNet, remove English stop words)  finished!
	It took 10.655102491378784 seconds


Applying NLTK feature extraction: X_test
	==> Replace unused UNICODE characters. 
	==> Text in lower case. 
	==> Apply Lemmatize using WordNet's built-in morphy function. Returns the input word unchanged if it cannot be found in WordNet 
	==> Remove English stop words
	Done: NLTK feature extraction (replace unused UNICODE characters, text in lower case, apply Lemmatize using WordNet, remove English stop words)  finished!
	It took 6.18167519569397 seconds

================================================================================
KERAS DEEP LEARNING MODEL
Using layers:
	==> Embedding(max_features, embed_size)
	==> Bidirectional(LSTM(32, return_sequences = True)
	==> GlobalMaxPool1D()
	==> Dense(20, activation="relu")
	==> Dropout(0.05)
	==> Dense(1, activation="sigmoid")
	==> Dense(1, activation="sigmoid")
Compile option:
	==> model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	===> Tokenizer: fit_on_texts(X_train)
	===> X_train = pad_sequences(list_tokenized_train, maxlen=6000)
	===> Create Keras model


NUMBER OF EPOCHS USED: 10

	===> Tokenizer: fit_on_texts(X_test)
	===> X_test = pad_sequences(list_sentences_test, maxlen=6000)
	=====> Training the model: model.fit()
Train on 11314 samples, validate on 7532 samples
Epoch 1/10

  100/11314 [..............................] - ETA: 1:01 - loss: 0.6754 - accuracy: 0.6626
  200/11314 [..............................] - ETA: 33s - loss: 0.6753 - accuracy: 0.6624 
  300/11314 [..............................] - ETA: 24s - loss: 0.6744 - accuracy: 0.6653
  400/11314 [>.............................] - ETA: 20s - loss: 0.6735 - accuracy: 0.6680
  500/11314 [>.............................] - ETA: 17s - loss: 0.6734 - accuracy: 0.6657
  600/11314 [>.............................] - ETA: 15s - loss: 0.6732 - accuracy: 0.6644
  700/11314 [>.............................] - ETA: 13s - loss: 0.6728 - accuracy: 0.6642
  800/11314 [=>............................] - ETA: 12s - loss: 0.6723 - accuracy: 0.6643
  900/11314 [=>............................] - ETA: 12s - loss: 0.6720 - accuracy: 0.6640
 1000/11314 [=>............................] - ETA: 11s - loss: 0.6716 - accuracy: 0.6642
 1100/11314 [=>............................] - ETA: 10s - loss: 0.6712 - accuracy: 0.6641
 1200/11314 [==>...........................] - ETA: 10s - loss: 0.6707 - accuracy: 0.6643
 1300/11314 [==>...........................] - ETA: 10s - loss: 0.6702 - accuracy: 0.6640
 1400/11314 [==>...........................] - ETA: 9s - loss: 0.6698 - accuracy: 0.6640 
 1500/11314 [==>...........................] - ETA: 9s - loss: 0.6692 - accuracy: 0.6643
 1600/11314 [===>..........................] - ETA: 9s - loss: 0.6688 - accuracy: 0.6640
 1700/11314 [===>..........................] - ETA: 8s - loss: 0.6682 - accuracy: 0.6642
 1800/11314 [===>..........................] - ETA: 8s - loss: 0.6677 - accuracy: 0.6643
 1900/11314 [====>.........................] - ETA: 8s - loss: 0.6673 - accuracy: 0.6637
 2000/11314 [====>.........................] - ETA: 8s - loss: 0.6668 - accuracy: 0.6637
 2100/11314 [====>.........................] - ETA: 7s - loss: 0.6661 - accuracy: 0.6643
 2200/11314 [====>.........................] - ETA: 7s - loss: 0.6655 - accuracy: 0.6643
 2300/11314 [=====>........................] - ETA: 7s - loss: 0.6649 - accuracy: 0.6642
 2400/11314 [=====>........................] - ETA: 7s - loss: 0.6643 - accuracy: 0.6641
 2500/11314 [=====>........................] - ETA: 7s - loss: 0.6638 - accuracy: 0.6639
 2600/11314 [=====>........................] - ETA: 7s - loss: 0.6632 - accuracy: 0.6638
 2700/11314 [======>.......................] - ETA: 6s - loss: 0.6625 - accuracy: 0.6640
 2800/11314 [======>.......................] - ETA: 6s - loss: 0.6619 - accuracy: 0.6642
 2900/11314 [======>.......................] - ETA: 6s - loss: 0.6613 - accuracy: 0.6642
 3000/11314 [======>.......................] - ETA: 6s - loss: 0.6607 - accuracy: 0.6642
 3100/11314 [=======>......................] - ETA: 6s - loss: 0.6602 - accuracy: 0.6640
 3200/11314 [=======>......................] - ETA: 6s - loss: 0.6597 - accuracy: 0.6638
 3300/11314 [=======>......................] - ETA: 6s - loss: 0.6591 - accuracy: 0.6638
 3400/11314 [========>.....................] - ETA: 6s - loss: 0.6584 - accuracy: 0.6640
 3500/11314 [========>.....................] - ETA: 5s - loss: 0.6578 - accuracy: 0.6641
 3600/11314 [========>.....................] - ETA: 5s - loss: 0.6573 - accuracy: 0.6640
 3700/11314 [========>.....................] - ETA: 5s - loss: 0.6567 - accuracy: 0.6641
 3800/11314 [=========>....................] - ETA: 5s - loss: 0.6562 - accuracy: 0.6639
 3900/11314 [=========>....................] - ETA: 5s - loss: 0.6556 - accuracy: 0.6641
 4000/11314 [=========>....................] - ETA: 5s - loss: 0.6549 - accuracy: 0.6642
 4100/11314 [=========>....................] - ETA: 5s - loss: 0.6544 - accuracy: 0.6642
 4200/11314 [==========>...................] - ETA: 5s - loss: 0.6538 - accuracy: 0.6642
 4300/11314 [==========>...................] - ETA: 5s - loss: 0.6533 - accuracy: 0.6641
 4400/11314 [==========>...................] - ETA: 5s - loss: 0.6527 - accuracy: 0.6642
 4500/11314 [==========>...................] - ETA: 5s - loss: 0.6522 - accuracy: 0.6642
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.6517 - accuracy: 0.6642
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.6511 - accuracy: 0.6642
 4800/11314 [===========>..................] - ETA: 4s - loss: 0.6506 - accuracy: 0.6641
 4900/11314 [===========>..................] - ETA: 4s - loss: 0.6501 - accuracy: 0.6641
 5000/11314 [============>.................] - ETA: 4s - loss: 0.6496 - accuracy: 0.6642
 5100/11314 [============>.................] - ETA: 4s - loss: 0.6491 - accuracy: 0.6643
 5200/11314 [============>.................] - ETA: 4s - loss: 0.6486 - accuracy: 0.6643
 5300/11314 [=============>................] - ETA: 4s - loss: 0.6480 - accuracy: 0.6643
 5400/11314 [=============>................] - ETA: 4s - loss: 0.6475 - accuracy: 0.6642
 5500/11314 [=============>................] - ETA: 4s - loss: 0.6470 - accuracy: 0.6643
 5600/11314 [=============>................] - ETA: 4s - loss: 0.6465 - accuracy: 0.6642
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.6460 - accuracy: 0.6642
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.6455 - accuracy: 0.6643
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.6450 - accuracy: 0.6643
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.6445 - accuracy: 0.6643
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.6440 - accuracy: 0.6643
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.6435 - accuracy: 0.6643
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.6430 - accuracy: 0.6644
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.6425 - accuracy: 0.6644
 6500/11314 [================>.............] - ETA: 3s - loss: 0.6421 - accuracy: 0.6644
 6600/11314 [================>.............] - ETA: 3s - loss: 0.6416 - accuracy: 0.6644
 6700/11314 [================>.............] - ETA: 3s - loss: 0.6411 - accuracy: 0.6652
 6800/11314 [=================>............] - ETA: 3s - loss: 0.6406 - accuracy: 0.6658
 6900/11314 [=================>............] - ETA: 3s - loss: 0.6401 - accuracy: 0.6665
 7000/11314 [=================>............] - ETA: 2s - loss: 0.6396 - accuracy: 0.6671
 7100/11314 [=================>............] - ETA: 2s - loss: 0.6391 - accuracy: 0.6679
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.6387 - accuracy: 0.6685
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.6382 - accuracy: 0.6691
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.6377 - accuracy: 0.6697
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.6372 - accuracy: 0.6703
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.6368 - accuracy: 0.6708
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.6363 - accuracy: 0.6714
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.6357 - accuracy: 0.6721
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.6353 - accuracy: 0.6726
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.6349 - accuracy: 0.6730
 8100/11314 [====================>.........] - ETA: 2s - loss: 0.6344 - accuracy: 0.6735
 8200/11314 [====================>.........] - ETA: 2s - loss: 0.6339 - accuracy: 0.6739
 8300/11314 [=====================>........] - ETA: 2s - loss: 0.6335 - accuracy: 0.6744
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.6330 - accuracy: 0.6749
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.6325 - accuracy: 0.6754
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.6320 - accuracy: 0.6758
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.6316 - accuracy: 0.6763
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.6311 - accuracy: 0.6768
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.6307 - accuracy: 0.6771
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.6303 - accuracy: 0.6775
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.6298 - accuracy: 0.6779
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.6293 - accuracy: 0.6783
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.6289 - accuracy: 0.6786
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.6285 - accuracy: 0.6789
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.6281 - accuracy: 0.6791
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.6276 - accuracy: 0.6795
 9700/11314 [========================>.....] - ETA: 1s - loss: 0.6272 - accuracy: 0.6799
 9800/11314 [========================>.....] - ETA: 1s - loss: 0.6268 - accuracy: 0.6802
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.6263 - accuracy: 0.6806
10000/11314 [=========================>....] - ETA: 0s - loss: 0.6259 - accuracy: 0.6809
10100/11314 [=========================>....] - ETA: 0s - loss: 0.6255 - accuracy: 0.6812
10200/11314 [==========================>...] - ETA: 0s - loss: 0.6250 - accuracy: 0.6815
10300/11314 [==========================>...] - ETA: 0s - loss: 0.6246 - accuracy: 0.6818
10400/11314 [==========================>...] - ETA: 0s - loss: 0.6242 - accuracy: 0.6820
10500/11314 [==========================>...] - ETA: 0s - loss: 0.6238 - accuracy: 0.6823
10600/11314 [===========================>..] - ETA: 0s - loss: 0.6234 - accuracy: 0.6826
10700/11314 [===========================>..] - ETA: 0s - loss: 0.6229 - accuracy: 0.6829
10800/11314 [===========================>..] - ETA: 0s - loss: 0.6225 - accuracy: 0.6832
10900/11314 [===========================>..] - ETA: 0s - loss: 0.6221 - accuracy: 0.6834
11000/11314 [============================>.] - ETA: 0s - loss: 0.6216 - accuracy: 0.6837
11100/11314 [============================>.] - ETA: 0s - loss: 0.6212 - accuracy: 0.6840
11200/11314 [============================>.] - ETA: 0s - loss: 0.6208 - accuracy: 0.6842
11300/11314 [============================>.] - ETA: 0s - loss: 0.6204 - accuracy: 0.6845
11314/11314 [==============================] - 9s 785us/step - loss: 0.6203 - accuracy: 0.6845 - val_loss: 0.5719 - val_accuracy: 0.7118
Epoch 2/10

  100/11314 [..............................] - ETA: 6s - loss: 0.5751 - accuracy: 0.7053
  200/11314 [..............................] - ETA: 6s - loss: 0.5721 - accuracy: 0.7103
  300/11314 [..............................] - ETA: 6s - loss: 0.5716 - accuracy: 0.7118
  400/11314 [>.............................] - ETA: 6s - loss: 0.5707 - accuracy: 0.7133
  500/11314 [>.............................] - ETA: 6s - loss: 0.5701 - accuracy: 0.7135
  600/11314 [>.............................] - ETA: 6s - loss: 0.5699 - accuracy: 0.7128
  700/11314 [>.............................] - ETA: 6s - loss: 0.5699 - accuracy: 0.7115
  800/11314 [=>............................] - ETA: 6s - loss: 0.5695 - accuracy: 0.7118
  900/11314 [=>............................] - ETA: 6s - loss: 0.5690 - accuracy: 0.7123
 1000/11314 [=>............................] - ETA: 6s - loss: 0.5687 - accuracy: 0.7121
 1100/11314 [=>............................] - ETA: 6s - loss: 0.5686 - accuracy: 0.7116
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.5680 - accuracy: 0.7121
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.5676 - accuracy: 0.7125
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.5671 - accuracy: 0.7128
 1500/11314 [==>...........................] - ETA: 6s - loss: 0.5667 - accuracy: 0.7128
 1600/11314 [===>..........................] - ETA: 6s - loss: 0.5662 - accuracy: 0.7132
 1700/11314 [===>..........................] - ETA: 5s - loss: 0.5660 - accuracy: 0.7128
 1800/11314 [===>..........................] - ETA: 5s - loss: 0.5657 - accuracy: 0.7127
 1900/11314 [====>.........................] - ETA: 5s - loss: 0.5653 - accuracy: 0.7129
 2000/11314 [====>.........................] - ETA: 5s - loss: 0.5650 - accuracy: 0.7127
 2100/11314 [====>.........................] - ETA: 5s - loss: 0.5647 - accuracy: 0.7149
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.5643 - accuracy: 0.7170
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.5639 - accuracy: 0.7186
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.5634 - accuracy: 0.7206
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.5631 - accuracy: 0.7222
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.5627 - accuracy: 0.7236
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.5624 - accuracy: 0.7249
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.5620 - accuracy: 0.7262
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.5615 - accuracy: 0.7276
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.5612 - accuracy: 0.7285
 3100/11314 [=======>......................] - ETA: 5s - loss: 0.5609 - accuracy: 0.7296
 3200/11314 [=======>......................] - ETA: 5s - loss: 0.5606 - accuracy: 0.7304
 3300/11314 [=======>......................] - ETA: 4s - loss: 0.5603 - accuracy: 0.7312
 3400/11314 [========>.....................] - ETA: 4s - loss: 0.5600 - accuracy: 0.7319
 3500/11314 [========>.....................] - ETA: 4s - loss: 0.5596 - accuracy: 0.7328
 3600/11314 [========>.....................] - ETA: 4s - loss: 0.5593 - accuracy: 0.7334
 3700/11314 [========>.....................] - ETA: 4s - loss: 0.5590 - accuracy: 0.7340
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.5587 - accuracy: 0.7346
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.5583 - accuracy: 0.7353
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.5580 - accuracy: 0.7359
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.5576 - accuracy: 0.7366
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.5573 - accuracy: 0.7372
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.5570 - accuracy: 0.7375
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.5566 - accuracy: 0.7381
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.5563 - accuracy: 0.7385
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.5559 - accuracy: 0.7391
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.5556 - accuracy: 0.7395
 4800/11314 [===========>..................] - ETA: 4s - loss: 0.5553 - accuracy: 0.7398
 4900/11314 [===========>..................] - ETA: 3s - loss: 0.5549 - accuracy: 0.7402
 5000/11314 [============>.................] - ETA: 3s - loss: 0.5547 - accuracy: 0.7403
 5100/11314 [============>.................] - ETA: 3s - loss: 0.5544 - accuracy: 0.7407
 5200/11314 [============>.................] - ETA: 3s - loss: 0.5540 - accuracy: 0.7411
 5300/11314 [=============>................] - ETA: 3s - loss: 0.5537 - accuracy: 0.7414
 5400/11314 [=============>................] - ETA: 3s - loss: 0.5533 - accuracy: 0.7418
 5500/11314 [=============>................] - ETA: 3s - loss: 0.5531 - accuracy: 0.7420
 5600/11314 [=============>................] - ETA: 3s - loss: 0.5527 - accuracy: 0.7423
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.5524 - accuracy: 0.7434
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.5521 - accuracy: 0.7445
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.5517 - accuracy: 0.7456
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.5513 - accuracy: 0.7466
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.5510 - accuracy: 0.7477
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.5507 - accuracy: 0.7488
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.5503 - accuracy: 0.7497
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.5500 - accuracy: 0.7507
 6500/11314 [================>.............] - ETA: 2s - loss: 0.5497 - accuracy: 0.7515
 6600/11314 [================>.............] - ETA: 2s - loss: 0.5493 - accuracy: 0.7525
 6700/11314 [================>.............] - ETA: 2s - loss: 0.5490 - accuracy: 0.7533
 6800/11314 [=================>............] - ETA: 2s - loss: 0.5486 - accuracy: 0.7548
 6900/11314 [=================>............] - ETA: 2s - loss: 0.5483 - accuracy: 0.7563
 7000/11314 [=================>............] - ETA: 2s - loss: 0.5479 - accuracy: 0.7577
 7100/11314 [=================>............] - ETA: 2s - loss: 0.5476 - accuracy: 0.7591
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.5473 - accuracy: 0.7604
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.5470 - accuracy: 0.7617
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.5466 - accuracy: 0.7629
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.5463 - accuracy: 0.7641
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.5460 - accuracy: 0.7653
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.5457 - accuracy: 0.7664
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.5454 - accuracy: 0.7675
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.5451 - accuracy: 0.7686
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.5448 - accuracy: 0.7697
 8100/11314 [====================>.........] - ETA: 1s - loss: 0.5445 - accuracy: 0.7708
 8200/11314 [====================>.........] - ETA: 1s - loss: 0.5441 - accuracy: 0.7719
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.5438 - accuracy: 0.7729
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.5435 - accuracy: 0.7739
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.5432 - accuracy: 0.7750
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.5428 - accuracy: 0.7759
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.5425 - accuracy: 0.7768
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.5422 - accuracy: 0.7777
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.5419 - accuracy: 0.7785
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.5416 - accuracy: 0.7794
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.5412 - accuracy: 0.7802
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.5409 - accuracy: 0.7810
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.5406 - accuracy: 0.7818
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.5403 - accuracy: 0.7825
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.5400 - accuracy: 0.7833
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.5397 - accuracy: 0.7841
 9700/11314 [========================>.....] - ETA: 1s - loss: 0.5393 - accuracy: 0.7848
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.5390 - accuracy: 0.7855
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.5387 - accuracy: 0.7861
10000/11314 [=========================>....] - ETA: 0s - loss: 0.5383 - accuracy: 0.7869
10100/11314 [=========================>....] - ETA: 0s - loss: 0.5380 - accuracy: 0.7876
10200/11314 [==========================>...] - ETA: 0s - loss: 0.5377 - accuracy: 0.7883
10300/11314 [==========================>...] - ETA: 0s - loss: 0.5374 - accuracy: 0.7889
10400/11314 [==========================>...] - ETA: 0s - loss: 0.5371 - accuracy: 0.7896
10500/11314 [==========================>...] - ETA: 0s - loss: 0.5368 - accuracy: 0.7902
10600/11314 [===========================>..] - ETA: 0s - loss: 0.5365 - accuracy: 0.7908
10700/11314 [===========================>..] - ETA: 0s - loss: 0.5361 - accuracy: 0.7914
10800/11314 [===========================>..] - ETA: 0s - loss: 0.5358 - accuracy: 0.7920
10900/11314 [===========================>..] - ETA: 0s - loss: 0.5355 - accuracy: 0.7926
11000/11314 [============================>.] - ETA: 0s - loss: 0.5352 - accuracy: 0.7932
11100/11314 [============================>.] - ETA: 0s - loss: 0.5348 - accuracy: 0.7938
11200/11314 [============================>.] - ETA: 0s - loss: 0.5345 - accuracy: 0.7943
11300/11314 [============================>.] - ETA: 0s - loss: 0.5342 - accuracy: 0.7948
11314/11314 [==============================] - 8s 733us/step - loss: 0.5342 - accuracy: 0.7949 - val_loss: 0.4982 - val_accuracy: 0.8553
Epoch 3/10

  100/11314 [..............................] - ETA: 6s - loss: 0.4998 - accuracy: 0.8568
  200/11314 [..............................] - ETA: 6s - loss: 0.4982 - accuracy: 0.8555
  300/11314 [..............................] - ETA: 6s - loss: 0.4970 - accuracy: 0.8570
  400/11314 [>.............................] - ETA: 6s - loss: 0.4971 - accuracy: 0.8563
  500/11314 [>.............................] - ETA: 6s - loss: 0.4960 - accuracy: 0.8572
  600/11314 [>.............................] - ETA: 6s - loss: 0.4958 - accuracy: 0.8561
  700/11314 [>.............................] - ETA: 6s - loss: 0.4959 - accuracy: 0.8556
  800/11314 [=>............................] - ETA: 6s - loss: 0.4962 - accuracy: 0.8550
  900/11314 [=>............................] - ETA: 6s - loss: 0.4959 - accuracy: 0.8546
 1000/11314 [=>............................] - ETA: 6s - loss: 0.4957 - accuracy: 0.8547
 1100/11314 [=>............................] - ETA: 6s - loss: 0.4954 - accuracy: 0.8545
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.4952 - accuracy: 0.8546
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.4950 - accuracy: 0.8543
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.4946 - accuracy: 0.8542
 1500/11314 [==>...........................] - ETA: 6s - loss: 0.4945 - accuracy: 0.8573
 1600/11314 [===>..........................] - ETA: 6s - loss: 0.4942 - accuracy: 0.8602
 1700/11314 [===>..........................] - ETA: 6s - loss: 0.4939 - accuracy: 0.8627
 1800/11314 [===>..........................] - ETA: 6s - loss: 0.4936 - accuracy: 0.8647
 1900/11314 [====>.........................] - ETA: 5s - loss: 0.4932 - accuracy: 0.8667
 2000/11314 [====>.........................] - ETA: 5s - loss: 0.4929 - accuracy: 0.8686
 2100/11314 [====>.........................] - ETA: 5s - loss: 0.4926 - accuracy: 0.8705
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.4925 - accuracy: 0.8717
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.4921 - accuracy: 0.8751
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.4918 - accuracy: 0.8784
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.4915 - accuracy: 0.8813
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.4911 - accuracy: 0.8840
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.4908 - accuracy: 0.8864
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.4905 - accuracy: 0.8887
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.4901 - accuracy: 0.8908
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.4899 - accuracy: 0.8928
 3100/11314 [=======>......................] - ETA: 5s - loss: 0.4896 - accuracy: 0.8947
 3200/11314 [=======>......................] - ETA: 5s - loss: 0.4892 - accuracy: 0.8964
 3300/11314 [=======>......................] - ETA: 5s - loss: 0.4890 - accuracy: 0.8980
 3400/11314 [========>.....................] - ETA: 5s - loss: 0.4887 - accuracy: 0.8996
 3500/11314 [========>.....................] - ETA: 4s - loss: 0.4884 - accuracy: 0.9010
 3600/11314 [========>.....................] - ETA: 4s - loss: 0.4881 - accuracy: 0.9024
 3700/11314 [========>.....................] - ETA: 4s - loss: 0.4878 - accuracy: 0.9037
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.4876 - accuracy: 0.9049
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.4874 - accuracy: 0.9060
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.4871 - accuracy: 0.9071
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.4868 - accuracy: 0.9081
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.4864 - accuracy: 0.9091
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.4861 - accuracy: 0.9100
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.4859 - accuracy: 0.9109
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.4856 - accuracy: 0.9118
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.4852 - accuracy: 0.9126
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.4850 - accuracy: 0.9134
 4800/11314 [===========>..................] - ETA: 4s - loss: 0.4848 - accuracy: 0.9141
 4900/11314 [===========>..................] - ETA: 4s - loss: 0.4845 - accuracy: 0.9149
 5000/11314 [============>.................] - ETA: 3s - loss: 0.4842 - accuracy: 0.9156
 5100/11314 [============>.................] - ETA: 3s - loss: 0.4840 - accuracy: 0.9162
 5200/11314 [============>.................] - ETA: 3s - loss: 0.4838 - accuracy: 0.9169
 5300/11314 [=============>................] - ETA: 3s - loss: 0.4835 - accuracy: 0.9175
 5400/11314 [=============>................] - ETA: 3s - loss: 0.4832 - accuracy: 0.9180
 5500/11314 [=============>................] - ETA: 3s - loss: 0.4830 - accuracy: 0.9186
 5600/11314 [=============>................] - ETA: 3s - loss: 0.4828 - accuracy: 0.9191
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.4825 - accuracy: 0.9197
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.4823 - accuracy: 0.9202
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.4821 - accuracy: 0.9207
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.4818 - accuracy: 0.9211
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.4816 - accuracy: 0.9216
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.4813 - accuracy: 0.9221
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.4810 - accuracy: 0.9225
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.4807 - accuracy: 0.9229
 6500/11314 [================>.............] - ETA: 3s - loss: 0.4805 - accuracy: 0.9233
 6600/11314 [================>.............] - ETA: 2s - loss: 0.4802 - accuracy: 0.9237
 6700/11314 [================>.............] - ETA: 2s - loss: 0.4800 - accuracy: 0.9241
 6800/11314 [=================>............] - ETA: 2s - loss: 0.4798 - accuracy: 0.9245
 6900/11314 [=================>............] - ETA: 2s - loss: 0.4795 - accuracy: 0.9248
 7000/11314 [=================>............] - ETA: 2s - loss: 0.4792 - accuracy: 0.9252
 7100/11314 [=================>............] - ETA: 2s - loss: 0.4789 - accuracy: 0.9256
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.4787 - accuracy: 0.9259
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.4784 - accuracy: 0.9262
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.4782 - accuracy: 0.9265
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.4779 - accuracy: 0.9268
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.4776 - accuracy: 0.9271
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.4774 - accuracy: 0.9274
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.4771 - accuracy: 0.9277
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.4769 - accuracy: 0.9280
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.4766 - accuracy: 0.9282
 8100/11314 [====================>.........] - ETA: 2s - loss: 0.4763 - accuracy: 0.9285
 8200/11314 [====================>.........] - ETA: 1s - loss: 0.4761 - accuracy: 0.9288
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.4758 - accuracy: 0.9290
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.4756 - accuracy: 0.9292
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.4753 - accuracy: 0.9295
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.4751 - accuracy: 0.9297
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.4748 - accuracy: 0.9299
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.4746 - accuracy: 0.9301
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.4744 - accuracy: 0.9303
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.4741 - accuracy: 0.9305
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.4738 - accuracy: 0.9308
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.4736 - accuracy: 0.9310
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.4733 - accuracy: 0.9312
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.4730 - accuracy: 0.9314
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.4728 - accuracy: 0.9316
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.4725 - accuracy: 0.9318
 9700/11314 [========================>.....] - ETA: 1s - loss: 0.4723 - accuracy: 0.9319
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.4721 - accuracy: 0.9321
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.4718 - accuracy: 0.9323
10000/11314 [=========================>....] - ETA: 0s - loss: 0.4716 - accuracy: 0.9325
10100/11314 [=========================>....] - ETA: 0s - loss: 0.4713 - accuracy: 0.9326
10200/11314 [==========================>...] - ETA: 0s - loss: 0.4710 - accuracy: 0.9328
10300/11314 [==========================>...] - ETA: 0s - loss: 0.4708 - accuracy: 0.9329
10400/11314 [==========================>...] - ETA: 0s - loss: 0.4705 - accuracy: 0.9331
10500/11314 [==========================>...] - ETA: 0s - loss: 0.4703 - accuracy: 0.9333
10600/11314 [===========================>..] - ETA: 0s - loss: 0.4701 - accuracy: 0.9335
10700/11314 [===========================>..] - ETA: 0s - loss: 0.4698 - accuracy: 0.9336
10800/11314 [===========================>..] - ETA: 0s - loss: 0.4696 - accuracy: 0.9337
10900/11314 [===========================>..] - ETA: 0s - loss: 0.4693 - accuracy: 0.9339
11000/11314 [============================>.] - ETA: 0s - loss: 0.4691 - accuracy: 0.9340
11100/11314 [============================>.] - ETA: 0s - loss: 0.4689 - accuracy: 0.9341
11200/11314 [============================>.] - ETA: 0s - loss: 0.4686 - accuracy: 0.9343
11300/11314 [============================>.] - ETA: 0s - loss: 0.4684 - accuracy: 0.9344
11314/11314 [==============================] - 8s 739us/step - loss: 0.4684 - accuracy: 0.9345 - val_loss: 0.4398 - val_accuracy: 0.9496
Epoch 4/10

  100/11314 [..............................] - ETA: 6s - loss: 0.4396 - accuracy: 0.9495
  200/11314 [..............................] - ETA: 6s - loss: 0.4391 - accuracy: 0.9500
  300/11314 [..............................] - ETA: 6s - loss: 0.4399 - accuracy: 0.9493
  400/11314 [>.............................] - ETA: 6s - loss: 0.4397 - accuracy: 0.9495
  500/11314 [>.............................] - ETA: 6s - loss: 0.4392 - accuracy: 0.9495
  600/11314 [>.............................] - ETA: 6s - loss: 0.4386 - accuracy: 0.9497
  700/11314 [>.............................] - ETA: 6s - loss: 0.4386 - accuracy: 0.9495
  800/11314 [=>............................] - ETA: 6s - loss: 0.4378 - accuracy: 0.9497
  900/11314 [=>............................] - ETA: 6s - loss: 0.4374 - accuracy: 0.9496
 1000/11314 [=>............................] - ETA: 6s - loss: 0.4375 - accuracy: 0.9496
 1100/11314 [=>............................] - ETA: 6s - loss: 0.4377 - accuracy: 0.9495
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.4374 - accuracy: 0.9494
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.4371 - accuracy: 0.9493
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.4368 - accuracy: 0.9493
 1500/11314 [==>...........................] - ETA: 6s - loss: 0.4365 - accuracy: 0.9493
 1600/11314 [===>..........................] - ETA: 6s - loss: 0.4360 - accuracy: 0.9494
 1700/11314 [===>..........................] - ETA: 5s - loss: 0.4359 - accuracy: 0.9494
 1800/11314 [===>..........................] - ETA: 5s - loss: 0.4356 - accuracy: 0.9494
 1900/11314 [====>.........................] - ETA: 5s - loss: 0.4353 - accuracy: 0.9495
 2000/11314 [====>.........................] - ETA: 5s - loss: 0.4351 - accuracy: 0.9496
 2100/11314 [====>.........................] - ETA: 5s - loss: 0.4350 - accuracy: 0.9495
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.4348 - accuracy: 0.9495
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.4347 - accuracy: 0.9495
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.4344 - accuracy: 0.9495
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.4342 - accuracy: 0.9495
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.4339 - accuracy: 0.9496
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.4337 - accuracy: 0.9495
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.4335 - accuracy: 0.9495
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.4332 - accuracy: 0.9496
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.4329 - accuracy: 0.9496
 3100/11314 [=======>......................] - ETA: 5s - loss: 0.4327 - accuracy: 0.9495
 3200/11314 [=======>......................] - ETA: 5s - loss: 0.4325 - accuracy: 0.9496
 3300/11314 [=======>......................] - ETA: 4s - loss: 0.4323 - accuracy: 0.9496
 3400/11314 [========>.....................] - ETA: 4s - loss: 0.4321 - accuracy: 0.9496
 3500/11314 [========>.....................] - ETA: 4s - loss: 0.4320 - accuracy: 0.9496
 3600/11314 [========>.....................] - ETA: 4s - loss: 0.4318 - accuracy: 0.9496
 3700/11314 [========>.....................] - ETA: 4s - loss: 0.4316 - accuracy: 0.9495
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.4313 - accuracy: 0.9496
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.4311 - accuracy: 0.9496
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.4308 - accuracy: 0.9496
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.4307 - accuracy: 0.9496
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.4304 - accuracy: 0.9496
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.4301 - accuracy: 0.9497
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.4300 - accuracy: 0.9497
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.4298 - accuracy: 0.9497
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.4295 - accuracy: 0.9497
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.4294 - accuracy: 0.9497
 4800/11314 [===========>..................] - ETA: 4s - loss: 0.4292 - accuracy: 0.9496
 4900/11314 [===========>..................] - ETA: 3s - loss: 0.4290 - accuracy: 0.9496
 5000/11314 [============>.................] - ETA: 3s - loss: 0.4288 - accuracy: 0.9496
 5100/11314 [============>.................] - ETA: 3s - loss: 0.4285 - accuracy: 0.9497
 5200/11314 [============>.................] - ETA: 3s - loss: 0.4283 - accuracy: 0.9497
 5300/11314 [=============>................] - ETA: 3s - loss: 0.4281 - accuracy: 0.9497
 5400/11314 [=============>................] - ETA: 3s - loss: 0.4280 - accuracy: 0.9497
 5500/11314 [=============>................] - ETA: 3s - loss: 0.4278 - accuracy: 0.9497
 5600/11314 [=============>................] - ETA: 3s - loss: 0.4276 - accuracy: 0.9497
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.4274 - accuracy: 0.9497
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.4272 - accuracy: 0.9496
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.4270 - accuracy: 0.9496
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.4268 - accuracy: 0.9496
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.4266 - accuracy: 0.9496
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.4264 - accuracy: 0.9496
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.4261 - accuracy: 0.9496
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.4259 - accuracy: 0.9496
 6500/11314 [================>.............] - ETA: 2s - loss: 0.4257 - accuracy: 0.9496
 6600/11314 [================>.............] - ETA: 2s - loss: 0.4254 - accuracy: 0.9496
 6700/11314 [================>.............] - ETA: 2s - loss: 0.4253 - accuracy: 0.9497
 6800/11314 [=================>............] - ETA: 2s - loss: 0.4250 - accuracy: 0.9496
 6900/11314 [=================>............] - ETA: 2s - loss: 0.4249 - accuracy: 0.9496
 7000/11314 [=================>............] - ETA: 2s - loss: 0.4246 - accuracy: 0.9496
 7100/11314 [=================>............] - ETA: 2s - loss: 0.4244 - accuracy: 0.9497
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.4242 - accuracy: 0.9497
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.4240 - accuracy: 0.9496
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.4238 - accuracy: 0.9496
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.4236 - accuracy: 0.9496
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.4234 - accuracy: 0.9496
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.4232 - accuracy: 0.9496
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.4230 - accuracy: 0.9496
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.4228 - accuracy: 0.9496
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.4226 - accuracy: 0.9496
 8100/11314 [====================>.........] - ETA: 1s - loss: 0.4224 - accuracy: 0.9496
 8200/11314 [====================>.........] - ETA: 1s - loss: 0.4222 - accuracy: 0.9496
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.4220 - accuracy: 0.9496
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.4217 - accuracy: 0.9496
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.4215 - accuracy: 0.9496
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.4213 - accuracy: 0.9496
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.4211 - accuracy: 0.9497
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.4209 - accuracy: 0.9497
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.4207 - accuracy: 0.9497
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.4205 - accuracy: 0.9497
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.4203 - accuracy: 0.9497
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.4201 - accuracy: 0.9497
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.4199 - accuracy: 0.9497
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.4197 - accuracy: 0.9497
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.4195 - accuracy: 0.9497
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.4193 - accuracy: 0.9497
 9700/11314 [========================>.....] - ETA: 1s - loss: 0.4191 - accuracy: 0.9497
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.4189 - accuracy: 0.9497
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.4187 - accuracy: 0.9497
10000/11314 [=========================>....] - ETA: 0s - loss: 0.4185 - accuracy: 0.9497
10100/11314 [=========================>....] - ETA: 0s - loss: 0.4183 - accuracy: 0.9497
10200/11314 [==========================>...] - ETA: 0s - loss: 0.4181 - accuracy: 0.9497
10300/11314 [==========================>...] - ETA: 0s - loss: 0.4179 - accuracy: 0.9497
10400/11314 [==========================>...] - ETA: 0s - loss: 0.4177 - accuracy: 0.9497
10500/11314 [==========================>...] - ETA: 0s - loss: 0.4175 - accuracy: 0.9497
10600/11314 [===========================>..] - ETA: 0s - loss: 0.4173 - accuracy: 0.9497
10700/11314 [===========================>..] - ETA: 0s - loss: 0.4171 - accuracy: 0.9496
10800/11314 [===========================>..] - ETA: 0s - loss: 0.4170 - accuracy: 0.9496
10900/11314 [===========================>..] - ETA: 0s - loss: 0.4168 - accuracy: 0.9496
11000/11314 [============================>.] - ETA: 0s - loss: 0.4166 - accuracy: 0.9496
11100/11314 [============================>.] - ETA: 0s - loss: 0.4164 - accuracy: 0.9496
11200/11314 [============================>.] - ETA: 0s - loss: 0.4162 - accuracy: 0.9496
11300/11314 [============================>.] - ETA: 0s - loss: 0.4160 - accuracy: 0.9496
11314/11314 [==============================] - 8s 734us/step - loss: 0.4159 - accuracy: 0.9496 - val_loss: 0.3931 - val_accuracy: 0.9496
Epoch 5/10

  100/11314 [..............................] - ETA: 6s - loss: 0.3921 - accuracy: 0.9495
  200/11314 [..............................] - ETA: 6s - loss: 0.3915 - accuracy: 0.9503
  300/11314 [..............................] - ETA: 6s - loss: 0.3912 - accuracy: 0.9505
  400/11314 [>.............................] - ETA: 6s - loss: 0.3913 - accuracy: 0.9501
  500/11314 [>.............................] - ETA: 6s - loss: 0.3916 - accuracy: 0.9502
  600/11314 [>.............................] - ETA: 6s - loss: 0.3919 - accuracy: 0.9499
  700/11314 [>.............................] - ETA: 6s - loss: 0.3917 - accuracy: 0.9501
  800/11314 [=>............................] - ETA: 6s - loss: 0.3914 - accuracy: 0.9501
  900/11314 [=>............................] - ETA: 6s - loss: 0.3915 - accuracy: 0.9500
 1000/11314 [=>............................] - ETA: 6s - loss: 0.3915 - accuracy: 0.9499
 1100/11314 [=>............................] - ETA: 6s - loss: 0.3914 - accuracy: 0.9500
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.3909 - accuracy: 0.9500
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.3907 - accuracy: 0.9499
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.3905 - accuracy: 0.9499
 1500/11314 [==>...........................] - ETA: 6s - loss: 0.3905 - accuracy: 0.9498
 1600/11314 [===>..........................] - ETA: 6s - loss: 0.3904 - accuracy: 0.9497
 1700/11314 [===>..........................] - ETA: 5s - loss: 0.3899 - accuracy: 0.9498
 1800/11314 [===>..........................] - ETA: 5s - loss: 0.3897 - accuracy: 0.9498
 1900/11314 [====>.........................] - ETA: 5s - loss: 0.3895 - accuracy: 0.9499
 2000/11314 [====>.........................] - ETA: 5s - loss: 0.3893 - accuracy: 0.9499
 2100/11314 [====>.........................] - ETA: 5s - loss: 0.3893 - accuracy: 0.9498
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.3891 - accuracy: 0.9499
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.3888 - accuracy: 0.9499
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.3887 - accuracy: 0.9499
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.3884 - accuracy: 0.9499
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.3883 - accuracy: 0.9499
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.3882 - accuracy: 0.9498
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.3881 - accuracy: 0.9498
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.3880 - accuracy: 0.9497
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.3879 - accuracy: 0.9497
 3100/11314 [=======>......................] - ETA: 5s - loss: 0.3878 - accuracy: 0.9497
 3200/11314 [=======>......................] - ETA: 5s - loss: 0.3875 - accuracy: 0.9498
 3300/11314 [=======>......................] - ETA: 4s - loss: 0.3872 - accuracy: 0.9498
 3400/11314 [========>.....................] - ETA: 4s - loss: 0.3870 - accuracy: 0.9498
 3500/11314 [========>.....................] - ETA: 4s - loss: 0.3869 - accuracy: 0.9498
 3600/11314 [========>.....................] - ETA: 4s - loss: 0.3868 - accuracy: 0.9498
 3700/11314 [========>.....................] - ETA: 4s - loss: 0.3866 - accuracy: 0.9498
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.3864 - accuracy: 0.9497
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.3863 - accuracy: 0.9497
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.3861 - accuracy: 0.9497
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.3859 - accuracy: 0.9498
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.3858 - accuracy: 0.9497
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.3857 - accuracy: 0.9497
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.3856 - accuracy: 0.9497
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.3854 - accuracy: 0.9497
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.3853 - accuracy: 0.9497
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.3851 - accuracy: 0.9497
 4800/11314 [===========>..................] - ETA: 4s - loss: 0.3849 - accuracy: 0.9497
 4900/11314 [===========>..................] - ETA: 3s - loss: 0.3848 - accuracy: 0.9497
 5000/11314 [============>.................] - ETA: 3s - loss: 0.3846 - accuracy: 0.9497
 5100/11314 [============>.................] - ETA: 3s - loss: 0.3844 - accuracy: 0.9497
 5200/11314 [============>.................] - ETA: 3s - loss: 0.3842 - accuracy: 0.9497
 5300/11314 [=============>................] - ETA: 3s - loss: 0.3841 - accuracy: 0.9497
 5400/11314 [=============>................] - ETA: 3s - loss: 0.3840 - accuracy: 0.9497
 5500/11314 [=============>................] - ETA: 3s - loss: 0.3838 - accuracy: 0.9496
 5600/11314 [=============>................] - ETA: 3s - loss: 0.3837 - accuracy: 0.9496
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.3835 - accuracy: 0.9496
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.3833 - accuracy: 0.9496
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.3832 - accuracy: 0.9496
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.3830 - accuracy: 0.9496
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.3828 - accuracy: 0.9496
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.3827 - accuracy: 0.9496
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.3826 - accuracy: 0.9496
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.3824 - accuracy: 0.9496
 6500/11314 [================>.............] - ETA: 2s - loss: 0.3822 - accuracy: 0.9496
 6600/11314 [================>.............] - ETA: 2s - loss: 0.3820 - accuracy: 0.9496
 6700/11314 [================>.............] - ETA: 2s - loss: 0.3818 - accuracy: 0.9497
 6800/11314 [=================>............] - ETA: 2s - loss: 0.3816 - accuracy: 0.9497
 6900/11314 [=================>............] - ETA: 2s - loss: 0.3814 - accuracy: 0.9497
 7000/11314 [=================>............] - ETA: 2s - loss: 0.3812 - accuracy: 0.9497
 7100/11314 [=================>............] - ETA: 2s - loss: 0.3811 - accuracy: 0.9496
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.3809 - accuracy: 0.9496
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.3808 - accuracy: 0.9496
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.3806 - accuracy: 0.9496
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.3804 - accuracy: 0.9496
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.3802 - accuracy: 0.9496
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.3800 - accuracy: 0.9496
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.3799 - accuracy: 0.9496
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.3797 - accuracy: 0.9496
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.3795 - accuracy: 0.9496
 8100/11314 [====================>.........] - ETA: 1s - loss: 0.3793 - accuracy: 0.9496
 8200/11314 [====================>.........] - ETA: 1s - loss: 0.3792 - accuracy: 0.9496
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.3790 - accuracy: 0.9496
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.3788 - accuracy: 0.9496
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.3786 - accuracy: 0.9496
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.3785 - accuracy: 0.9496
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.3783 - accuracy: 0.9496
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.3782 - accuracy: 0.9496
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.3780 - accuracy: 0.9496
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.3778 - accuracy: 0.9496
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.3776 - accuracy: 0.9496
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.3774 - accuracy: 0.9496
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.3773 - accuracy: 0.9496
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.3771 - accuracy: 0.9496
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.3769 - accuracy: 0.9496
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.3768 - accuracy: 0.9496
 9700/11314 [========================>.....] - ETA: 1s - loss: 0.3766 - accuracy: 0.9496
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.3765 - accuracy: 0.9495
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.3764 - accuracy: 0.9495
10000/11314 [=========================>....] - ETA: 0s - loss: 0.3762 - accuracy: 0.9495
10100/11314 [=========================>....] - ETA: 0s - loss: 0.3761 - accuracy: 0.9495
10200/11314 [==========================>...] - ETA: 0s - loss: 0.3758 - accuracy: 0.9496
10300/11314 [==========================>...] - ETA: 0s - loss: 0.3757 - accuracy: 0.9496
10400/11314 [==========================>...] - ETA: 0s - loss: 0.3755 - accuracy: 0.9496
10500/11314 [==========================>...] - ETA: 0s - loss: 0.3754 - accuracy: 0.9496
10600/11314 [===========================>..] - ETA: 0s - loss: 0.3752 - accuracy: 0.9496
10700/11314 [===========================>..] - ETA: 0s - loss: 0.3751 - accuracy: 0.9496
10800/11314 [===========================>..] - ETA: 0s - loss: 0.3749 - accuracy: 0.9496
10900/11314 [===========================>..] - ETA: 0s - loss: 0.3747 - accuracy: 0.9496
11000/11314 [============================>.] - ETA: 0s - loss: 0.3746 - accuracy: 0.9496
11100/11314 [============================>.] - ETA: 0s - loss: 0.3744 - accuracy: 0.9496
11200/11314 [============================>.] - ETA: 0s - loss: 0.3743 - accuracy: 0.9496
11300/11314 [============================>.] - ETA: 0s - loss: 0.3741 - accuracy: 0.9496
11314/11314 [==============================] - 8s 734us/step - loss: 0.3741 - accuracy: 0.9496 - val_loss: 0.3558 - val_accuracy: 0.9496
Epoch 6/10

  100/11314 [..............................] - ETA: 7s - loss: 0.3529 - accuracy: 0.9489
  200/11314 [..............................] - ETA: 6s - loss: 0.3560 - accuracy: 0.9492
  300/11314 [..............................] - ETA: 6s - loss: 0.3563 - accuracy: 0.9496
  400/11314 [>.............................] - ETA: 6s - loss: 0.3556 - accuracy: 0.9496
  500/11314 [>.............................] - ETA: 6s - loss: 0.3554 - accuracy: 0.9497
  600/11314 [>.............................] - ETA: 6s - loss: 0.3556 - accuracy: 0.9496
  700/11314 [>.............................] - ETA: 6s - loss: 0.3554 - accuracy: 0.9497
  800/11314 [=>............................] - ETA: 6s - loss: 0.3550 - accuracy: 0.9498
  900/11314 [=>............................] - ETA: 6s - loss: 0.3550 - accuracy: 0.9498
 1000/11314 [=>............................] - ETA: 6s - loss: 0.3549 - accuracy: 0.9497
 1100/11314 [=>............................] - ETA: 6s - loss: 0.3545 - accuracy: 0.9498
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.3543 - accuracy: 0.9498
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.3542 - accuracy: 0.9498
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.3539 - accuracy: 0.9499
 1500/11314 [==>...........................] - ETA: 6s - loss: 0.3536 - accuracy: 0.9499
 1600/11314 [===>..........................] - ETA: 6s - loss: 0.3536 - accuracy: 0.9499
 1700/11314 [===>..........................] - ETA: 5s - loss: 0.3535 - accuracy: 0.9500
 1800/11314 [===>..........................] - ETA: 5s - loss: 0.3535 - accuracy: 0.9499
 1900/11314 [====>.........................] - ETA: 5s - loss: 0.3534 - accuracy: 0.9499
 2000/11314 [====>.........................] - ETA: 5s - loss: 0.3534 - accuracy: 0.9498
 2100/11314 [====>.........................] - ETA: 5s - loss: 0.3532 - accuracy: 0.9498
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.3529 - accuracy: 0.9498
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.3527 - accuracy: 0.9498
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.3524 - accuracy: 0.9498
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.3523 - accuracy: 0.9498
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.3523 - accuracy: 0.9498
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.3520 - accuracy: 0.9497
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.3519 - accuracy: 0.9497
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.3517 - accuracy: 0.9497
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.3514 - accuracy: 0.9498
 3100/11314 [=======>......................] - ETA: 5s - loss: 0.3513 - accuracy: 0.9498
 3200/11314 [=======>......................] - ETA: 5s - loss: 0.3511 - accuracy: 0.9497
 3300/11314 [=======>......................] - ETA: 4s - loss: 0.3510 - accuracy: 0.9497
 3400/11314 [========>.....................] - ETA: 4s - loss: 0.3509 - accuracy: 0.9496
 3500/11314 [========>.....................] - ETA: 4s - loss: 0.3507 - accuracy: 0.9497
 3600/11314 [========>.....................] - ETA: 4s - loss: 0.3506 - accuracy: 0.9496
 3700/11314 [========>.....................] - ETA: 4s - loss: 0.3504 - accuracy: 0.9496
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.3503 - accuracy: 0.9496
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.3501 - accuracy: 0.9497
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.3499 - accuracy: 0.9497
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.3499 - accuracy: 0.9497
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.3498 - accuracy: 0.9497
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.3497 - accuracy: 0.9497
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.3496 - accuracy: 0.9497
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.3494 - accuracy: 0.9497
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.3492 - accuracy: 0.9498
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.3491 - accuracy: 0.9498
 4800/11314 [===========>..................] - ETA: 4s - loss: 0.3489 - accuracy: 0.9498
 4900/11314 [===========>..................] - ETA: 3s - loss: 0.3487 - accuracy: 0.9498
 5000/11314 [============>.................] - ETA: 3s - loss: 0.3486 - accuracy: 0.9497
 5100/11314 [============>.................] - ETA: 3s - loss: 0.3485 - accuracy: 0.9497
 5200/11314 [============>.................] - ETA: 3s - loss: 0.3483 - accuracy: 0.9498
 5300/11314 [=============>................] - ETA: 3s - loss: 0.3482 - accuracy: 0.9498
 5400/11314 [=============>................] - ETA: 3s - loss: 0.3480 - accuracy: 0.9498
 5500/11314 [=============>................] - ETA: 3s - loss: 0.3479 - accuracy: 0.9498
 5600/11314 [=============>................] - ETA: 3s - loss: 0.3478 - accuracy: 0.9497
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.3476 - accuracy: 0.9497
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.3475 - accuracy: 0.9497
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.3474 - accuracy: 0.9497
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.3473 - accuracy: 0.9497
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.3472 - accuracy: 0.9497
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.3471 - accuracy: 0.9497
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.3470 - accuracy: 0.9497
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.3468 - accuracy: 0.9497
 6500/11314 [================>.............] - ETA: 2s - loss: 0.3467 - accuracy: 0.9497
 6600/11314 [================>.............] - ETA: 2s - loss: 0.3466 - accuracy: 0.9497
 6700/11314 [================>.............] - ETA: 2s - loss: 0.3464 - accuracy: 0.9497
 6800/11314 [=================>............] - ETA: 2s - loss: 0.3463 - accuracy: 0.9497
 6900/11314 [=================>............] - ETA: 2s - loss: 0.3461 - accuracy: 0.9497
 7000/11314 [=================>............] - ETA: 2s - loss: 0.3460 - accuracy: 0.9497
 7100/11314 [=================>............] - ETA: 2s - loss: 0.3458 - accuracy: 0.9497
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.3458 - accuracy: 0.9497
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.3456 - accuracy: 0.9497
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.3455 - accuracy: 0.9496
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.3453 - accuracy: 0.9496
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.3452 - accuracy: 0.9497
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.3451 - accuracy: 0.9497
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.3449 - accuracy: 0.9496
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.3448 - accuracy: 0.9496
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.3447 - accuracy: 0.9496
 8100/11314 [====================>.........] - ETA: 1s - loss: 0.3446 - accuracy: 0.9496
 8200/11314 [====================>.........] - ETA: 1s - loss: 0.3444 - accuracy: 0.9496
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.3443 - accuracy: 0.9496
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.3442 - accuracy: 0.9496
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.3440 - accuracy: 0.9496
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.3439 - accuracy: 0.9496
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.3438 - accuracy: 0.9496
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.3436 - accuracy: 0.9496
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.3435 - accuracy: 0.9496
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.3433 - accuracy: 0.9497
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.3432 - accuracy: 0.9497
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.3431 - accuracy: 0.9496
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.3430 - accuracy: 0.9496
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.3429 - accuracy: 0.9496
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.3427 - accuracy: 0.9496
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.3427 - accuracy: 0.9496
 9700/11314 [========================>.....] - ETA: 1s - loss: 0.3426 - accuracy: 0.9496
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.3424 - accuracy: 0.9496
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.3423 - accuracy: 0.9496
10000/11314 [=========================>....] - ETA: 0s - loss: 0.3422 - accuracy: 0.9496
10100/11314 [=========================>....] - ETA: 0s - loss: 0.3421 - accuracy: 0.9496
10200/11314 [==========================>...] - ETA: 0s - loss: 0.3420 - accuracy: 0.9496
10300/11314 [==========================>...] - ETA: 0s - loss: 0.3419 - accuracy: 0.9496
10400/11314 [==========================>...] - ETA: 0s - loss: 0.3417 - accuracy: 0.9496
10500/11314 [==========================>...] - ETA: 0s - loss: 0.3416 - accuracy: 0.9496
10600/11314 [===========================>..] - ETA: 0s - loss: 0.3415 - accuracy: 0.9496
10700/11314 [===========================>..] - ETA: 0s - loss: 0.3414 - accuracy: 0.9496
10800/11314 [===========================>..] - ETA: 0s - loss: 0.3412 - accuracy: 0.9496
10900/11314 [===========================>..] - ETA: 0s - loss: 0.3411 - accuracy: 0.9496
11000/11314 [============================>.] - ETA: 0s - loss: 0.3410 - accuracy: 0.9496
11100/11314 [============================>.] - ETA: 0s - loss: 0.3408 - accuracy: 0.9496
11200/11314 [============================>.] - ETA: 0s - loss: 0.3407 - accuracy: 0.9496
11300/11314 [============================>.] - ETA: 0s - loss: 0.3406 - accuracy: 0.9496
11314/11314 [==============================] - 8s 735us/step - loss: 0.3406 - accuracy: 0.9496 - val_loss: 0.3259 - val_accuracy: 0.9496
Epoch 7/10

  100/11314 [..............................] - ETA: 7s - loss: 0.3260 - accuracy: 0.9479
  200/11314 [..............................] - ETA: 6s - loss: 0.3253 - accuracy: 0.9489
  300/11314 [..............................] - ETA: 6s - loss: 0.3266 - accuracy: 0.9486
  400/11314 [>.............................] - ETA: 6s - loss: 0.3265 - accuracy: 0.9487
  500/11314 [>.............................] - ETA: 6s - loss: 0.3259 - accuracy: 0.9491
  600/11314 [>.............................] - ETA: 6s - loss: 0.3250 - accuracy: 0.9495
  700/11314 [>.............................] - ETA: 6s - loss: 0.3255 - accuracy: 0.9492
  800/11314 [=>............................] - ETA: 6s - loss: 0.3252 - accuracy: 0.9492
  900/11314 [=>............................] - ETA: 6s - loss: 0.3252 - accuracy: 0.9491
 1000/11314 [=>............................] - ETA: 6s - loss: 0.3252 - accuracy: 0.9491
 1100/11314 [=>............................] - ETA: 6s - loss: 0.3249 - accuracy: 0.9491
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.3250 - accuracy: 0.9491
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.3250 - accuracy: 0.9491
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.3248 - accuracy: 0.9491
 1500/11314 [==>...........................] - ETA: 6s - loss: 0.3247 - accuracy: 0.9491
 1600/11314 [===>..........................] - ETA: 6s - loss: 0.3244 - accuracy: 0.9492
 1700/11314 [===>..........................] - ETA: 6s - loss: 0.3243 - accuracy: 0.9493
 1800/11314 [===>..........................] - ETA: 5s - loss: 0.3241 - accuracy: 0.9493
 1900/11314 [====>.........................] - ETA: 5s - loss: 0.3240 - accuracy: 0.9493
 2000/11314 [====>.........................] - ETA: 5s - loss: 0.3239 - accuracy: 0.9493
 2100/11314 [====>.........................] - ETA: 5s - loss: 0.3237 - accuracy: 0.9493
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.3237 - accuracy: 0.9494
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.3236 - accuracy: 0.9494
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.3234 - accuracy: 0.9494
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.3234 - accuracy: 0.9494
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.3234 - accuracy: 0.9494
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.3234 - accuracy: 0.9493
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.3233 - accuracy: 0.9493
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.3231 - accuracy: 0.9493
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.3230 - accuracy: 0.9493
 3100/11314 [=======>......................] - ETA: 5s - loss: 0.3228 - accuracy: 0.9493
 3200/11314 [=======>......................] - ETA: 5s - loss: 0.3227 - accuracy: 0.9494
 3300/11314 [=======>......................] - ETA: 4s - loss: 0.3225 - accuracy: 0.9493
 3400/11314 [========>.....................] - ETA: 4s - loss: 0.3225 - accuracy: 0.9493
 3500/11314 [========>.....................] - ETA: 4s - loss: 0.3225 - accuracy: 0.9493
 3600/11314 [========>.....................] - ETA: 4s - loss: 0.3224 - accuracy: 0.9493
 3700/11314 [========>.....................] - ETA: 4s - loss: 0.3222 - accuracy: 0.9493
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.3220 - accuracy: 0.9493
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.3219 - accuracy: 0.9494
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.3218 - accuracy: 0.9494
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.3217 - accuracy: 0.9494
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.3216 - accuracy: 0.9494
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.3214 - accuracy: 0.9494
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.3213 - accuracy: 0.9494
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.3211 - accuracy: 0.9494
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.3210 - accuracy: 0.9494
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.3209 - accuracy: 0.9494
 4800/11314 [===========>..................] - ETA: 4s - loss: 0.3208 - accuracy: 0.9494
 4900/11314 [===========>..................] - ETA: 3s - loss: 0.3207 - accuracy: 0.9494
 5000/11314 [============>.................] - ETA: 3s - loss: 0.3207 - accuracy: 0.9494
 5100/11314 [============>.................] - ETA: 3s - loss: 0.3206 - accuracy: 0.9494
 5200/11314 [============>.................] - ETA: 3s - loss: 0.3204 - accuracy: 0.9494
 5300/11314 [=============>................] - ETA: 3s - loss: 0.3202 - accuracy: 0.9495
 5400/11314 [=============>................] - ETA: 3s - loss: 0.3201 - accuracy: 0.9495
 5500/11314 [=============>................] - ETA: 3s - loss: 0.3200 - accuracy: 0.9495
 5600/11314 [=============>................] - ETA: 3s - loss: 0.3199 - accuracy: 0.9495
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.3198 - accuracy: 0.9495
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.3196 - accuracy: 0.9495
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.3195 - accuracy: 0.9495
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.3194 - accuracy: 0.9495
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.3192 - accuracy: 0.9495
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.3191 - accuracy: 0.9495
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.3190 - accuracy: 0.9495
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.3190 - accuracy: 0.9495
 6500/11314 [================>.............] - ETA: 2s - loss: 0.3188 - accuracy: 0.9496
 6600/11314 [================>.............] - ETA: 2s - loss: 0.3187 - accuracy: 0.9495
 6700/11314 [================>.............] - ETA: 2s - loss: 0.3186 - accuracy: 0.9496
 6800/11314 [=================>............] - ETA: 2s - loss: 0.3185 - accuracy: 0.9496
 6900/11314 [=================>............] - ETA: 2s - loss: 0.3184 - accuracy: 0.9496
 7000/11314 [=================>............] - ETA: 2s - loss: 0.3183 - accuracy: 0.9496
 7100/11314 [=================>............] - ETA: 2s - loss: 0.3182 - accuracy: 0.9496
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.3181 - accuracy: 0.9496
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.3180 - accuracy: 0.9496
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.3179 - accuracy: 0.9496
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.3178 - accuracy: 0.9496
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.3177 - accuracy: 0.9495
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.3176 - accuracy: 0.9495
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.3175 - accuracy: 0.9495
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.3173 - accuracy: 0.9495
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.3172 - accuracy: 0.9495
 8100/11314 [====================>.........] - ETA: 1s - loss: 0.3172 - accuracy: 0.9495
 8200/11314 [====================>.........] - ETA: 1s - loss: 0.3171 - accuracy: 0.9495
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.3170 - accuracy: 0.9495
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.3169 - accuracy: 0.9495
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.3167 - accuracy: 0.9495
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.3167 - accuracy: 0.9495
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.3165 - accuracy: 0.9495
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.3164 - accuracy: 0.9496
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.3163 - accuracy: 0.9495
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.3162 - accuracy: 0.9496
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.3160 - accuracy: 0.9496
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.3159 - accuracy: 0.9496
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.3158 - accuracy: 0.9496
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.3157 - accuracy: 0.9495
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.3157 - accuracy: 0.9495
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.3155 - accuracy: 0.9495
 9700/11314 [========================>.....] - ETA: 1s - loss: 0.3154 - accuracy: 0.9496
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.3153 - accuracy: 0.9496
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.3152 - accuracy: 0.9496
10000/11314 [=========================>....] - ETA: 0s - loss: 0.3150 - accuracy: 0.9496
10100/11314 [=========================>....] - ETA: 0s - loss: 0.3150 - accuracy: 0.9496
10200/11314 [==========================>...] - ETA: 0s - loss: 0.3148 - accuracy: 0.9496
10300/11314 [==========================>...] - ETA: 0s - loss: 0.3148 - accuracy: 0.9496
10400/11314 [==========================>...] - ETA: 0s - loss: 0.3146 - accuracy: 0.9496
10500/11314 [==========================>...] - ETA: 0s - loss: 0.3145 - accuracy: 0.9496
10600/11314 [===========================>..] - ETA: 0s - loss: 0.3144 - accuracy: 0.9496
10700/11314 [===========================>..] - ETA: 0s - loss: 0.3143 - accuracy: 0.9496
10800/11314 [===========================>..] - ETA: 0s - loss: 0.3142 - accuracy: 0.9496
10900/11314 [===========================>..] - ETA: 0s - loss: 0.3141 - accuracy: 0.9496
11000/11314 [============================>.] - ETA: 0s - loss: 0.3140 - accuracy: 0.9496
11100/11314 [============================>.] - ETA: 0s - loss: 0.3139 - accuracy: 0.9496
11200/11314 [============================>.] - ETA: 0s - loss: 0.3138 - accuracy: 0.9496
11300/11314 [============================>.] - ETA: 0s - loss: 0.3137 - accuracy: 0.9496
11314/11314 [==============================] - 8s 734us/step - loss: 0.3137 - accuracy: 0.9496 - val_loss: 0.3020 - val_accuracy: 0.9496
Epoch 8/10

  100/11314 [..............................] - ETA: 7s - loss: 0.3031 - accuracy: 0.9484
  200/11314 [..............................] - ETA: 6s - loss: 0.3029 - accuracy: 0.9484
  300/11314 [..............................] - ETA: 6s - loss: 0.3028 - accuracy: 0.9484
  400/11314 [>.............................] - ETA: 6s - loss: 0.3027 - accuracy: 0.9489
  500/11314 [>.............................] - ETA: 6s - loss: 0.3025 - accuracy: 0.9488
  600/11314 [>.............................] - ETA: 6s - loss: 0.3012 - accuracy: 0.9491
  700/11314 [>.............................] - ETA: 6s - loss: 0.3006 - accuracy: 0.9492
  800/11314 [=>............................] - ETA: 6s - loss: 0.3007 - accuracy: 0.9491
  900/11314 [=>............................] - ETA: 6s - loss: 0.3006 - accuracy: 0.9493
 1000/11314 [=>............................] - ETA: 6s - loss: 0.3006 - accuracy: 0.9494
 1100/11314 [=>............................] - ETA: 6s - loss: 0.3005 - accuracy: 0.9494
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.3002 - accuracy: 0.9495
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.3002 - accuracy: 0.9496
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.3000 - accuracy: 0.9496
 1500/11314 [==>...........................] - ETA: 6s - loss: 0.3001 - accuracy: 0.9496
 1600/11314 [===>..........................] - ETA: 6s - loss: 0.3000 - accuracy: 0.9496
 1700/11314 [===>..........................] - ETA: 6s - loss: 0.2998 - accuracy: 0.9497
 1800/11314 [===>..........................] - ETA: 6s - loss: 0.2999 - accuracy: 0.9496
 1900/11314 [====>.........................] - ETA: 6s - loss: 0.2999 - accuracy: 0.9496
 2000/11314 [====>.........................] - ETA: 5s - loss: 0.3000 - accuracy: 0.9495
 2100/11314 [====>.........................] - ETA: 5s - loss: 0.2999 - accuracy: 0.9495
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.2996 - accuracy: 0.9496
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.2995 - accuracy: 0.9496
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.2994 - accuracy: 0.9496
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.2994 - accuracy: 0.9495
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.2994 - accuracy: 0.9495
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.2995 - accuracy: 0.9495
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.2995 - accuracy: 0.9495
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.2993 - accuracy: 0.9496
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.2992 - accuracy: 0.9496
 3100/11314 [=======>......................] - ETA: 5s - loss: 0.2991 - accuracy: 0.9496
 3200/11314 [=======>......................] - ETA: 5s - loss: 0.2990 - accuracy: 0.9496
 3300/11314 [=======>......................] - ETA: 5s - loss: 0.2989 - accuracy: 0.9496
 3400/11314 [========>.....................] - ETA: 5s - loss: 0.2988 - accuracy: 0.9496
 3500/11314 [========>.....................] - ETA: 4s - loss: 0.2987 - accuracy: 0.9495
 3600/11314 [========>.....................] - ETA: 4s - loss: 0.2987 - accuracy: 0.9495
 3700/11314 [========>.....................] - ETA: 4s - loss: 0.2986 - accuracy: 0.9495
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.2985 - accuracy: 0.9495
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.2984 - accuracy: 0.9495
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.2983 - accuracy: 0.9495
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.2983 - accuracy: 0.9495
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.2981 - accuracy: 0.9495
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.2981 - accuracy: 0.9495
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.2980 - accuracy: 0.9495
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.2979 - accuracy: 0.9495
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.2977 - accuracy: 0.9496
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.2976 - accuracy: 0.9496
 4800/11314 [===========>..................] - ETA: 4s - loss: 0.2975 - accuracy: 0.9496
 4900/11314 [===========>..................] - ETA: 4s - loss: 0.2975 - accuracy: 0.9495
 5000/11314 [============>.................] - ETA: 4s - loss: 0.2974 - accuracy: 0.9495
 5100/11314 [============>.................] - ETA: 3s - loss: 0.2973 - accuracy: 0.9496
 5200/11314 [============>.................] - ETA: 3s - loss: 0.2972 - accuracy: 0.9496
 5300/11314 [=============>................] - ETA: 3s - loss: 0.2971 - accuracy: 0.9496
 5400/11314 [=============>................] - ETA: 3s - loss: 0.2970 - accuracy: 0.9496
 5500/11314 [=============>................] - ETA: 3s - loss: 0.2970 - accuracy: 0.9496
 5600/11314 [=============>................] - ETA: 3s - loss: 0.2969 - accuracy: 0.9496
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.2968 - accuracy: 0.9496
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.2967 - accuracy: 0.9496
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.2966 - accuracy: 0.9496
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.2965 - accuracy: 0.9496
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.2964 - accuracy: 0.9496
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.2963 - accuracy: 0.9496
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.2962 - accuracy: 0.9496
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.2961 - accuracy: 0.9496
 6500/11314 [================>.............] - ETA: 3s - loss: 0.2961 - accuracy: 0.9496
 6600/11314 [================>.............] - ETA: 2s - loss: 0.2960 - accuracy: 0.9496
 6700/11314 [================>.............] - ETA: 2s - loss: 0.2959 - accuracy: 0.9496
 6800/11314 [=================>............] - ETA: 2s - loss: 0.2958 - accuracy: 0.9496
 6900/11314 [=================>............] - ETA: 2s - loss: 0.2958 - accuracy: 0.9496
 7000/11314 [=================>............] - ETA: 2s - loss: 0.2957 - accuracy: 0.9495
 7100/11314 [=================>............] - ETA: 2s - loss: 0.2957 - accuracy: 0.9495
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.2955 - accuracy: 0.9495
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.2954 - accuracy: 0.9495
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.2953 - accuracy: 0.9495
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.2952 - accuracy: 0.9496
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.2951 - accuracy: 0.9496
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.2950 - accuracy: 0.9496
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.2950 - accuracy: 0.9496
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.2949 - accuracy: 0.9496
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.2948 - accuracy: 0.9496
 8100/11314 [====================>.........] - ETA: 2s - loss: 0.2947 - accuracy: 0.9496
 8200/11314 [====================>.........] - ETA: 1s - loss: 0.2946 - accuracy: 0.9496
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.2946 - accuracy: 0.9496
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.2944 - accuracy: 0.9496
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.2943 - accuracy: 0.9497
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.2942 - accuracy: 0.9497
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.2941 - accuracy: 0.9497
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.2941 - accuracy: 0.9497
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.2940 - accuracy: 0.9497
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.2939 - accuracy: 0.9497
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.2938 - accuracy: 0.9497
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.2938 - accuracy: 0.9497
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.2937 - accuracy: 0.9497
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.2936 - accuracy: 0.9497
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.2935 - accuracy: 0.9497
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.2934 - accuracy: 0.9497
 9700/11314 [========================>.....] - ETA: 1s - loss: 0.2933 - accuracy: 0.9497
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.2932 - accuracy: 0.9497
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.2932 - accuracy: 0.9497
10000/11314 [=========================>....] - ETA: 0s - loss: 0.2931 - accuracy: 0.9497
10100/11314 [=========================>....] - ETA: 0s - loss: 0.2930 - accuracy: 0.9497
10200/11314 [==========================>...] - ETA: 0s - loss: 0.2929 - accuracy: 0.9497
10300/11314 [==========================>...] - ETA: 0s - loss: 0.2929 - accuracy: 0.9497
10400/11314 [==========================>...] - ETA: 0s - loss: 0.2928 - accuracy: 0.9496
10500/11314 [==========================>...] - ETA: 0s - loss: 0.2927 - accuracy: 0.9496
10600/11314 [===========================>..] - ETA: 0s - loss: 0.2926 - accuracy: 0.9496
10700/11314 [===========================>..] - ETA: 0s - loss: 0.2926 - accuracy: 0.9496
10800/11314 [===========================>..] - ETA: 0s - loss: 0.2925 - accuracy: 0.9496
10900/11314 [===========================>..] - ETA: 0s - loss: 0.2924 - accuracy: 0.9496
11000/11314 [============================>.] - ETA: 0s - loss: 0.2923 - accuracy: 0.9496
11100/11314 [============================>.] - ETA: 0s - loss: 0.2923 - accuracy: 0.9496
11200/11314 [============================>.] - ETA: 0s - loss: 0.2922 - accuracy: 0.9496
11300/11314 [============================>.] - ETA: 0s - loss: 0.2921 - accuracy: 0.9496
11314/11314 [==============================] - 8s 746us/step - loss: 0.2921 - accuracy: 0.9496 - val_loss: 0.2827 - val_accuracy: 0.9496
Epoch 9/10

  100/11314 [..............................] - ETA: 6s - loss: 0.2843 - accuracy: 0.9505
  200/11314 [..............................] - ETA: 6s - loss: 0.2833 - accuracy: 0.9503
  300/11314 [..............................] - ETA: 6s - loss: 0.2825 - accuracy: 0.9505
  400/11314 [>.............................] - ETA: 6s - loss: 0.2828 - accuracy: 0.9504
  500/11314 [>.............................] - ETA: 6s - loss: 0.2824 - accuracy: 0.9504
  600/11314 [>.............................] - ETA: 6s - loss: 0.2826 - accuracy: 0.9502
  700/11314 [>.............................] - ETA: 6s - loss: 0.2824 - accuracy: 0.9500
  800/11314 [=>............................] - ETA: 6s - loss: 0.2828 - accuracy: 0.9498
  900/11314 [=>............................] - ETA: 6s - loss: 0.2830 - accuracy: 0.9496
 1000/11314 [=>............................] - ETA: 6s - loss: 0.2831 - accuracy: 0.9495
 1100/11314 [=>............................] - ETA: 6s - loss: 0.2828 - accuracy: 0.9495
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.2825 - accuracy: 0.9496
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.2823 - accuracy: 0.9496
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.2823 - accuracy: 0.9495
 1500/11314 [==>...........................] - ETA: 6s - loss: 0.2825 - accuracy: 0.9495
 1600/11314 [===>..........................] - ETA: 6s - loss: 0.2824 - accuracy: 0.9495
 1700/11314 [===>..........................] - ETA: 6s - loss: 0.2823 - accuracy: 0.9495
 1800/11314 [===>..........................] - ETA: 5s - loss: 0.2820 - accuracy: 0.9496
 1900/11314 [====>.........................] - ETA: 5s - loss: 0.2820 - accuracy: 0.9496
 2000/11314 [====>.........................] - ETA: 5s - loss: 0.2818 - accuracy: 0.9496
 2100/11314 [====>.........................] - ETA: 5s - loss: 0.2820 - accuracy: 0.9495
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.2818 - accuracy: 0.9496
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.2817 - accuracy: 0.9496
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.2816 - accuracy: 0.9496
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.2815 - accuracy: 0.9496
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.2814 - accuracy: 0.9496
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.2812 - accuracy: 0.9496
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.2812 - accuracy: 0.9496
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.2809 - accuracy: 0.9497
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.2808 - accuracy: 0.9497
 3100/11314 [=======>......................] - ETA: 5s - loss: 0.2807 - accuracy: 0.9497
 3200/11314 [=======>......................] - ETA: 5s - loss: 0.2806 - accuracy: 0.9497
 3300/11314 [=======>......................] - ETA: 5s - loss: 0.2806 - accuracy: 0.9497
 3400/11314 [========>.....................] - ETA: 4s - loss: 0.2806 - accuracy: 0.9496
 3500/11314 [========>.....................] - ETA: 4s - loss: 0.2805 - accuracy: 0.9496
 3600/11314 [========>.....................] - ETA: 4s - loss: 0.2804 - accuracy: 0.9496
 3700/11314 [========>.....................] - ETA: 4s - loss: 0.2804 - accuracy: 0.9496
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.2803 - accuracy: 0.9496
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.2802 - accuracy: 0.9496
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.2802 - accuracy: 0.9496
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.2801 - accuracy: 0.9496
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.2800 - accuracy: 0.9495
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.2799 - accuracy: 0.9496
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.2799 - accuracy: 0.9495
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.2798 - accuracy: 0.9496
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.2797 - accuracy: 0.9495
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.2796 - accuracy: 0.9495
 4800/11314 [===========>..................] - ETA: 4s - loss: 0.2795 - accuracy: 0.9496
 4900/11314 [===========>..................] - ETA: 4s - loss: 0.2794 - accuracy: 0.9495
 5000/11314 [============>.................] - ETA: 3s - loss: 0.2793 - accuracy: 0.9496
 5100/11314 [============>.................] - ETA: 3s - loss: 0.2792 - accuracy: 0.9495
 5200/11314 [============>.................] - ETA: 3s - loss: 0.2791 - accuracy: 0.9496
 5300/11314 [=============>................] - ETA: 3s - loss: 0.2790 - accuracy: 0.9496
 5400/11314 [=============>................] - ETA: 3s - loss: 0.2789 - accuracy: 0.9496
 5500/11314 [=============>................] - ETA: 3s - loss: 0.2788 - accuracy: 0.9496
 5600/11314 [=============>................] - ETA: 3s - loss: 0.2788 - accuracy: 0.9496
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.2786 - accuracy: 0.9496
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.2785 - accuracy: 0.9496
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.2785 - accuracy: 0.9496
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.2785 - accuracy: 0.9496
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.2784 - accuracy: 0.9496
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.2784 - accuracy: 0.9495
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.2782 - accuracy: 0.9496
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.2781 - accuracy: 0.9496
 6500/11314 [================>.............] - ETA: 3s - loss: 0.2781 - accuracy: 0.9496
 6600/11314 [================>.............] - ETA: 2s - loss: 0.2780 - accuracy: 0.9496
 6700/11314 [================>.............] - ETA: 2s - loss: 0.2779 - accuracy: 0.9496
 6800/11314 [=================>............] - ETA: 2s - loss: 0.2778 - accuracy: 0.9496
 6900/11314 [=================>............] - ETA: 2s - loss: 0.2778 - accuracy: 0.9496
 7000/11314 [=================>............] - ETA: 2s - loss: 0.2777 - accuracy: 0.9496
 7100/11314 [=================>............] - ETA: 2s - loss: 0.2776 - accuracy: 0.9496
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.2776 - accuracy: 0.9496
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.2775 - accuracy: 0.9496
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.2774 - accuracy: 0.9496
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.2774 - accuracy: 0.9495
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.2773 - accuracy: 0.9495
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.2772 - accuracy: 0.9496
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.2771 - accuracy: 0.9496
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.2770 - accuracy: 0.9496
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.2769 - accuracy: 0.9496
 8100/11314 [====================>.........] - ETA: 2s - loss: 0.2769 - accuracy: 0.9496
 8200/11314 [====================>.........] - ETA: 1s - loss: 0.2768 - accuracy: 0.9496
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.2767 - accuracy: 0.9496
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.2767 - accuracy: 0.9496
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.2766 - accuracy: 0.9496
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.2765 - accuracy: 0.9496
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.2764 - accuracy: 0.9496
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.2764 - accuracy: 0.9496
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.2763 - accuracy: 0.9496
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.2762 - accuracy: 0.9496
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.2761 - accuracy: 0.9496
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.2761 - accuracy: 0.9496
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.2760 - accuracy: 0.9496
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.2760 - accuracy: 0.9496
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.2760 - accuracy: 0.9496
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.2759 - accuracy: 0.9496
 9700/11314 [========================>.....] - ETA: 1s - loss: 0.2758 - accuracy: 0.9496
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.2757 - accuracy: 0.9496
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.2756 - accuracy: 0.9496
10000/11314 [=========================>....] - ETA: 0s - loss: 0.2755 - accuracy: 0.9496
10100/11314 [=========================>....] - ETA: 0s - loss: 0.2754 - accuracy: 0.9496
10200/11314 [==========================>...] - ETA: 0s - loss: 0.2754 - accuracy: 0.9496
10300/11314 [==========================>...] - ETA: 0s - loss: 0.2753 - accuracy: 0.9497
10400/11314 [==========================>...] - ETA: 0s - loss: 0.2752 - accuracy: 0.9496
10500/11314 [==========================>...] - ETA: 0s - loss: 0.2752 - accuracy: 0.9496
10600/11314 [===========================>..] - ETA: 0s - loss: 0.2751 - accuracy: 0.9496
10700/11314 [===========================>..] - ETA: 0s - loss: 0.2751 - accuracy: 0.9496
10800/11314 [===========================>..] - ETA: 0s - loss: 0.2750 - accuracy: 0.9496
10900/11314 [===========================>..] - ETA: 0s - loss: 0.2750 - accuracy: 0.9496
11000/11314 [============================>.] - ETA: 0s - loss: 0.2749 - accuracy: 0.9496
11100/11314 [============================>.] - ETA: 0s - loss: 0.2748 - accuracy: 0.9496
11200/11314 [============================>.] - ETA: 0s - loss: 0.2748 - accuracy: 0.9496
11300/11314 [============================>.] - ETA: 0s - loss: 0.2747 - accuracy: 0.9496
11314/11314 [==============================] - 9s 781us/step - loss: 0.2747 - accuracy: 0.9496 - val_loss: 0.2671 - val_accuracy: 0.9496
Epoch 10/10

  100/11314 [..............................] - ETA: 7s - loss: 0.2684 - accuracy: 0.9489
  200/11314 [..............................] - ETA: 7s - loss: 0.2673 - accuracy: 0.9492
  300/11314 [..............................] - ETA: 7s - loss: 0.2664 - accuracy: 0.9496
  400/11314 [>.............................] - ETA: 7s - loss: 0.2659 - accuracy: 0.9497
  500/11314 [>.............................] - ETA: 6s - loss: 0.2653 - accuracy: 0.9499
  600/11314 [>.............................] - ETA: 6s - loss: 0.2655 - accuracy: 0.9499
  700/11314 [>.............................] - ETA: 6s - loss: 0.2658 - accuracy: 0.9497
  800/11314 [=>............................] - ETA: 6s - loss: 0.2660 - accuracy: 0.9497
  900/11314 [=>............................] - ETA: 6s - loss: 0.2660 - accuracy: 0.9498
 1000/11314 [=>............................] - ETA: 6s - loss: 0.2658 - accuracy: 0.9498
 1100/11314 [=>............................] - ETA: 6s - loss: 0.2659 - accuracy: 0.9499
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.2660 - accuracy: 0.9497
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.2660 - accuracy: 0.9497
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.2660 - accuracy: 0.9495
 1500/11314 [==>...........................] - ETA: 6s - loss: 0.2658 - accuracy: 0.9496
 1600/11314 [===>..........................] - ETA: 6s - loss: 0.2655 - accuracy: 0.9497
 1700/11314 [===>..........................] - ETA: 6s - loss: 0.2653 - accuracy: 0.9498
 1800/11314 [===>..........................] - ETA: 5s - loss: 0.2655 - accuracy: 0.9497
 1900/11314 [====>.........................] - ETA: 5s - loss: 0.2654 - accuracy: 0.9497
 2000/11314 [====>.........................] - ETA: 5s - loss: 0.2653 - accuracy: 0.9497
 2100/11314 [====>.........................] - ETA: 5s - loss: 0.2651 - accuracy: 0.9498
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.2652 - accuracy: 0.9498
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.2653 - accuracy: 0.9497
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.2654 - accuracy: 0.9496
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.2653 - accuracy: 0.9496
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.2652 - accuracy: 0.9496
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.2651 - accuracy: 0.9497
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.2651 - accuracy: 0.9497
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.2648 - accuracy: 0.9498
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.2647 - accuracy: 0.9499
 3100/11314 [=======>......................] - ETA: 5s - loss: 0.2647 - accuracy: 0.9499
 3200/11314 [=======>......................] - ETA: 5s - loss: 0.2647 - accuracy: 0.9498
 3300/11314 [=======>......................] - ETA: 5s - loss: 0.2646 - accuracy: 0.9498
 3400/11314 [========>.....................] - ETA: 4s - loss: 0.2645 - accuracy: 0.9498
 3500/11314 [========>.....................] - ETA: 4s - loss: 0.2645 - accuracy: 0.9498
 3600/11314 [========>.....................] - ETA: 4s - loss: 0.2645 - accuracy: 0.9498
 3700/11314 [========>.....................] - ETA: 4s - loss: 0.2645 - accuracy: 0.9498
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.2644 - accuracy: 0.9498
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.2644 - accuracy: 0.9498
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.2644 - accuracy: 0.9498
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.2644 - accuracy: 0.9498
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.2644 - accuracy: 0.9497
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.2644 - accuracy: 0.9497
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.2645 - accuracy: 0.9497
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.2644 - accuracy: 0.9497
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.2644 - accuracy: 0.9496
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.2644 - accuracy: 0.9496
 4800/11314 [===========>..................] - ETA: 4s - loss: 0.2643 - accuracy: 0.9496
 4900/11314 [===========>..................] - ETA: 4s - loss: 0.2643 - accuracy: 0.9496
 5000/11314 [============>.................] - ETA: 4s - loss: 0.2643 - accuracy: 0.9496
 5100/11314 [============>.................] - ETA: 3s - loss: 0.2642 - accuracy: 0.9496
 5200/11314 [============>.................] - ETA: 3s - loss: 0.2642 - accuracy: 0.9496
 5300/11314 [=============>................] - ETA: 3s - loss: 0.2641 - accuracy: 0.9496
 5400/11314 [=============>................] - ETA: 3s - loss: 0.2641 - accuracy: 0.9496
 5500/11314 [=============>................] - ETA: 3s - loss: 0.2639 - accuracy: 0.9496
 5600/11314 [=============>................] - ETA: 3s - loss: 0.2639 - accuracy: 0.9496
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.2639 - accuracy: 0.9496
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.2638 - accuracy: 0.9496
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.2638 - accuracy: 0.9496
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.2637 - accuracy: 0.9496
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.2637 - accuracy: 0.9496
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.2636 - accuracy: 0.9496
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.2636 - accuracy: 0.9496
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.2636 - accuracy: 0.9496
 6500/11314 [================>.............] - ETA: 3s - loss: 0.2635 - accuracy: 0.9496
 6600/11314 [================>.............] - ETA: 2s - loss: 0.2635 - accuracy: 0.9496
 6700/11314 [================>.............] - ETA: 2s - loss: 0.2634 - accuracy: 0.9496
 6800/11314 [=================>............] - ETA: 2s - loss: 0.2633 - accuracy: 0.9496
 6900/11314 [=================>............] - ETA: 2s - loss: 0.2633 - accuracy: 0.9496
 7000/11314 [=================>............] - ETA: 2s - loss: 0.2631 - accuracy: 0.9496
 7100/11314 [=================>............] - ETA: 2s - loss: 0.2631 - accuracy: 0.9496
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.2631 - accuracy: 0.9496
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.2630 - accuracy: 0.9496
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.2629 - accuracy: 0.9496
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.2628 - accuracy: 0.9496
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.2627 - accuracy: 0.9496
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.2627 - accuracy: 0.9496
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.2626 - accuracy: 0.9496
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.2625 - accuracy: 0.9496
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.2625 - accuracy: 0.9496
 8100/11314 [====================>.........] - ETA: 2s - loss: 0.2625 - accuracy: 0.9496
 8200/11314 [====================>.........] - ETA: 1s - loss: 0.2624 - accuracy: 0.9496
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.2623 - accuracy: 0.9496
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.2622 - accuracy: 0.9496
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.2622 - accuracy: 0.9496
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.2621 - accuracy: 0.9496
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.2621 - accuracy: 0.9496
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.2620 - accuracy: 0.9496
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.2620 - accuracy: 0.9496
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.2620 - accuracy: 0.9496
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.2619 - accuracy: 0.9496
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.2619 - accuracy: 0.9496
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.2619 - accuracy: 0.9496
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.2618 - accuracy: 0.9496
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.2618 - accuracy: 0.9496
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.2617 - accuracy: 0.9496
 9700/11314 [========================>.....] - ETA: 1s - loss: 0.2617 - accuracy: 0.9496
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.2616 - accuracy: 0.9496
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.2615 - accuracy: 0.9496
10000/11314 [=========================>....] - ETA: 0s - loss: 0.2615 - accuracy: 0.9496
10100/11314 [=========================>....] - ETA: 0s - loss: 0.2614 - accuracy: 0.9496
10200/11314 [==========================>...] - ETA: 0s - loss: 0.2614 - accuracy: 0.9496
10300/11314 [==========================>...] - ETA: 0s - loss: 0.2613 - accuracy: 0.9496
10400/11314 [==========================>...] - ETA: 0s - loss: 0.2612 - accuracy: 0.9496
10500/11314 [==========================>...] - ETA: 0s - loss: 0.2611 - accuracy: 0.9496
10600/11314 [===========================>..] - ETA: 0s - loss: 0.2611 - accuracy: 0.9496
10700/11314 [===========================>..] - ETA: 0s - loss: 0.2611 - accuracy: 0.9496
10800/11314 [===========================>..] - ETA: 0s - loss: 0.2610 - accuracy: 0.9496
10900/11314 [===========================>..] - ETA: 0s - loss: 0.2609 - accuracy: 0.9496
11000/11314 [============================>.] - ETA: 0s - loss: 0.2609 - accuracy: 0.9496
11100/11314 [============================>.] - ETA: 0s - loss: 0.2608 - accuracy: 0.9496
11200/11314 [============================>.] - ETA: 0s - loss: 0.2607 - accuracy: 0.9496
11300/11314 [============================>.] - ETA: 0s - loss: 0.2606 - accuracy: 0.9496
11314/11314 [==============================] - 8s 749us/step - loss: 0.2606 - accuracy: 0.9496 - val_loss: 0.2544 - val_accuracy: 0.9496
	=====> Test the model: model.predict()
TWENTY_NEWS_GROUPS
	Training accuracy score: 94.96%
	Loss: 0.2544
	Test Accuracy: 94.96%


Loading IMDB_REVIEWS dataset:

===> Reading files from /home/ets-crchum/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/train/neg

===> Reading files from /home/ets-crchum/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/train/pos

===> Reading files from /home/ets-crchum/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/test/neg

===> Reading files from /home/ets-crchum/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/test/pos
data loaded
25000 documents - 33.133MB (training set)
25000 documents - 32.351MB (test set)


Applying NLTK feature extraction: X_train
	==> Replace unused UNICODE characters. 
	==> Text in lower case. 
	==> Apply Lemmatize using WordNet's built-in morphy function. Returns the input word unchanged if it cannot be found in WordNet 
	==> Remove English stop words
	Done: NLTK feature extraction (replace unused UNICODE characters, text in lower case, apply Lemmatize using WordNet, remove English stop words)  finished!
	It took 25.077327013015747 seconds


Applying NLTK feature extraction: X_test
	==> Replace unused UNICODE characters. 
	==> Text in lower case. 
	==> Apply Lemmatize using WordNet's built-in morphy function. Returns the input word unchanged if it cannot be found in WordNet 
	==> Remove English stop words
	Done: NLTK feature extraction (replace unused UNICODE characters, text in lower case, apply Lemmatize using WordNet, remove English stop words)  finished!
	It took 24.84916400909424 seconds

================================================================================
KERAS DEEP LEARNING MODEL
Using layers:
	==> Embedding(max_features, embed_size)
	==> Bidirectional(LSTM(32, return_sequences = True)
	==> GlobalMaxPool1D()
	==> Dense(20, activation="relu")
	==> Dropout(0.05)
	==> Dense(1, activation="sigmoid")
	==> Dense(1, activation="sigmoid")
Compile option:
	==> model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	===> Tokenizer: fit_on_texts(X_train)
	===> X_train = pad_sequences(list_tokenized_train, maxlen=6000)
	===> Create Keras model


NUMBER OF EPOCHS USED: 10

	===> Tokenizer: fit_on_texts(X_test)
	===> X_test = pad_sequences(list_sentences_test, maxlen=6000)
	=====> Training the model: model.fit()
Train on 25000 samples, validate on 25000 samples
Epoch 1/10

  100/25000 [..............................] - ETA: 2:20 - loss: 0.7072 - accuracy: 0.5300
  200/25000 [..............................] - ETA: 1:17 - loss: 0.7022 - accuracy: 0.5400
  300/25000 [..............................] - ETA: 56s - loss: 0.7066 - accuracy: 0.5300 
  400/25000 [..............................] - ETA: 46s - loss: 0.7086 - accuracy: 0.5250
  500/25000 [..............................] - ETA: 39s - loss: 0.7162 - accuracy: 0.5080
  600/25000 [..............................] - ETA: 35s - loss: 0.7172 - accuracy: 0.5050
  700/25000 [..............................] - ETA: 32s - loss: 0.7160 - accuracy: 0.5071
  800/25000 [..............................] - ETA: 30s - loss: 0.7161 - accuracy: 0.5063
  900/25000 [>.............................] - ETA: 28s - loss: 0.7161 - accuracy: 0.5056
 1000/25000 [>.............................] - ETA: 27s - loss: 0.7138 - accuracy: 0.5100
 1100/25000 [>.............................] - ETA: 25s - loss: 0.7176 - accuracy: 0.5000
 1200/25000 [>.............................] - ETA: 24s - loss: 0.7196 - accuracy: 0.4942
 1300/25000 [>.............................] - ETA: 23s - loss: 0.7181 - accuracy: 0.4969
 1400/25000 [>.............................] - ETA: 23s - loss: 0.7152 - accuracy: 0.5036
 1500/25000 [>.............................] - ETA: 22s - loss: 0.7140 - accuracy: 0.5053
 1600/25000 [>.............................] - ETA: 22s - loss: 0.7144 - accuracy: 0.5031
 1700/25000 [=>............................] - ETA: 21s - loss: 0.7138 - accuracy: 0.5035
 1800/25000 [=>............................] - ETA: 21s - loss: 0.7120 - accuracy: 0.5078
 1900/25000 [=>............................] - ETA: 20s - loss: 0.7114 - accuracy: 0.5084
 2000/25000 [=>............................] - ETA: 20s - loss: 0.7116 - accuracy: 0.5065
 2100/25000 [=>............................] - ETA: 19s - loss: 0.7112 - accuracy: 0.5062
 2200/25000 [=>............................] - ETA: 19s - loss: 0.7113 - accuracy: 0.5045
 2300/25000 [=>............................] - ETA: 19s - loss: 0.7111 - accuracy: 0.5039
 2400/25000 [=>............................] - ETA: 18s - loss: 0.7114 - accuracy: 0.5004
 2500/25000 [==>...........................] - ETA: 18s - loss: 0.7114 - accuracy: 0.4988
 2600/25000 [==>...........................] - ETA: 18s - loss: 0.7114 - accuracy: 0.4969
 2700/25000 [==>...........................] - ETA: 18s - loss: 0.7111 - accuracy: 0.4967
 2800/25000 [==>...........................] - ETA: 17s - loss: 0.7104 - accuracy: 0.4968
 2900/25000 [==>...........................] - ETA: 17s - loss: 0.7094 - accuracy: 0.5003
 3000/25000 [==>...........................] - ETA: 17s - loss: 0.7092 - accuracy: 0.4997
 3100/25000 [==>...........................] - ETA: 17s - loss: 0.7093 - accuracy: 0.4958
 3200/25000 [==>...........................] - ETA: 17s - loss: 0.7088 - accuracy: 0.4959
 3300/25000 [==>...........................] - ETA: 16s - loss: 0.7085 - accuracy: 0.4961
 3400/25000 [===>..........................] - ETA: 16s - loss: 0.7081 - accuracy: 0.4959
 3500/25000 [===>..........................] - ETA: 16s - loss: 0.7076 - accuracy: 0.4960
 3600/25000 [===>..........................] - ETA: 16s - loss: 0.7073 - accuracy: 0.4958
 3700/25000 [===>..........................] - ETA: 16s - loss: 0.7070 - accuracy: 0.4957
 3800/25000 [===>..........................] - ETA: 16s - loss: 0.7067 - accuracy: 0.4955
 3900/25000 [===>..........................] - ETA: 16s - loss: 0.7062 - accuracy: 0.4985
 4000/25000 [===>..........................] - ETA: 15s - loss: 0.7059 - accuracy: 0.4978
 4100/25000 [===>..........................] - ETA: 15s - loss: 0.7056 - accuracy: 0.4973
 4200/25000 [====>.........................] - ETA: 15s - loss: 0.7053 - accuracy: 0.4979
 4300/25000 [====>.........................] - ETA: 15s - loss: 0.7049 - accuracy: 0.5009
 4400/25000 [====>.........................] - ETA: 15s - loss: 0.7046 - accuracy: 0.5000
 4500/25000 [====>.........................] - ETA: 15s - loss: 0.7043 - accuracy: 0.5004
 4600/25000 [====>.........................] - ETA: 15s - loss: 0.7041 - accuracy: 0.5017
 4700/25000 [====>.........................] - ETA: 14s - loss: 0.7038 - accuracy: 0.5023
 4800/25000 [====>.........................] - ETA: 14s - loss: 0.7036 - accuracy: 0.5006
 4900/25000 [====>.........................] - ETA: 14s - loss: 0.7035 - accuracy: 0.4996
 5000/25000 [=====>........................] - ETA: 14s - loss: 0.7033 - accuracy: 0.4980
 5100/25000 [=====>........................] - ETA: 14s - loss: 0.7031 - accuracy: 0.4990
 5200/25000 [=====>........................] - ETA: 14s - loss: 0.7029 - accuracy: 0.4998
 5300/25000 [=====>........................] - ETA: 14s - loss: 0.7028 - accuracy: 0.5000
 5400/25000 [=====>........................] - ETA: 14s - loss: 0.7025 - accuracy: 0.5004
 5500/25000 [=====>........................] - ETA: 14s - loss: 0.7024 - accuracy: 0.4996
 5600/25000 [=====>........................] - ETA: 13s - loss: 0.7022 - accuracy: 0.5000
 5700/25000 [=====>........................] - ETA: 13s - loss: 0.7020 - accuracy: 0.5011
 5800/25000 [=====>........................] - ETA: 13s - loss: 0.7018 - accuracy: 0.5017
 5900/25000 [======>.......................] - ETA: 13s - loss: 0.7017 - accuracy: 0.5024
 6000/25000 [======>.......................] - ETA: 13s - loss: 0.7015 - accuracy: 0.5032
 6100/25000 [======>.......................] - ETA: 13s - loss: 0.7013 - accuracy: 0.5036
 6200/25000 [======>.......................] - ETA: 13s - loss: 0.7012 - accuracy: 0.5044
 6300/25000 [======>.......................] - ETA: 13s - loss: 0.7011 - accuracy: 0.5041
 6400/25000 [======>.......................] - ETA: 13s - loss: 0.7009 - accuracy: 0.5047
 6500/25000 [======>.......................] - ETA: 13s - loss: 0.7008 - accuracy: 0.5052
 6600/25000 [======>.......................] - ETA: 12s - loss: 0.7006 - accuracy: 0.5056
 6700/25000 [=======>......................] - ETA: 12s - loss: 0.7005 - accuracy: 0.5055
 6800/25000 [=======>......................] - ETA: 12s - loss: 0.7004 - accuracy: 0.5050
 6900/25000 [=======>......................] - ETA: 12s - loss: 0.7003 - accuracy: 0.5048
 7000/25000 [=======>......................] - ETA: 12s - loss: 0.7002 - accuracy: 0.5047
 7100/25000 [=======>......................] - ETA: 12s - loss: 0.7001 - accuracy: 0.5044
 7200/25000 [=======>......................] - ETA: 12s - loss: 0.7000 - accuracy: 0.5043
 7300/25000 [=======>......................] - ETA: 12s - loss: 0.6999 - accuracy: 0.5048
 7400/25000 [=======>......................] - ETA: 12s - loss: 0.6999 - accuracy: 0.5046
 7500/25000 [========>.....................] - ETA: 12s - loss: 0.6997 - accuracy: 0.5055
 7600/25000 [========>.....................] - ETA: 12s - loss: 0.6996 - accuracy: 0.5055
 7700/25000 [========>.....................] - ETA: 11s - loss: 0.6996 - accuracy: 0.5052
 7800/25000 [========>.....................] - ETA: 11s - loss: 0.6995 - accuracy: 0.5054
 7900/25000 [========>.....................] - ETA: 11s - loss: 0.6994 - accuracy: 0.5053
 8000/25000 [========>.....................] - ETA: 11s - loss: 0.6993 - accuracy: 0.5054
 8100/25000 [========>.....................] - ETA: 11s - loss: 0.6992 - accuracy: 0.5053
 8200/25000 [========>.....................] - ETA: 11s - loss: 0.6991 - accuracy: 0.5050
 8300/25000 [========>.....................] - ETA: 11s - loss: 0.6991 - accuracy: 0.5042
 8400/25000 [=========>....................] - ETA: 11s - loss: 0.6991 - accuracy: 0.5038
 8500/25000 [=========>....................] - ETA: 11s - loss: 0.6991 - accuracy: 0.5022
 8600/25000 [=========>....................] - ETA: 11s - loss: 0.6990 - accuracy: 0.5028
 8700/25000 [=========>....................] - ETA: 11s - loss: 0.6990 - accuracy: 0.5018
 8800/25000 [=========>....................] - ETA: 11s - loss: 0.6989 - accuracy: 0.5023
 8900/25000 [=========>....................] - ETA: 11s - loss: 0.6987 - accuracy: 0.5039
 9000/25000 [=========>....................] - ETA: 10s - loss: 0.6987 - accuracy: 0.5031
 9100/25000 [=========>....................] - ETA: 10s - loss: 0.6986 - accuracy: 0.5027
 9200/25000 [==========>...................] - ETA: 10s - loss: 0.6986 - accuracy: 0.5024
 9300/25000 [==========>...................] - ETA: 10s - loss: 0.6985 - accuracy: 0.5022
 9400/25000 [==========>...................] - ETA: 10s - loss: 0.6985 - accuracy: 0.5012
 9500/25000 [==========>...................] - ETA: 10s - loss: 0.6984 - accuracy: 0.5005
 9600/25000 [==========>...................] - ETA: 10s - loss: 0.6984 - accuracy: 0.4999
 9700/25000 [==========>...................] - ETA: 10s - loss: 0.6983 - accuracy: 0.5001
 9800/25000 [==========>...................] - ETA: 10s - loss: 0.6983 - accuracy: 0.5002
 9900/25000 [==========>...................] - ETA: 10s - loss: 0.6982 - accuracy: 0.5001
10000/25000 [===========>..................] - ETA: 10s - loss: 0.6982 - accuracy: 0.5010
10100/25000 [===========>..................] - ETA: 10s - loss: 0.6981 - accuracy: 0.5023
10200/25000 [===========>..................] - ETA: 9s - loss: 0.6980 - accuracy: 0.5026 
10300/25000 [===========>..................] - ETA: 9s - loss: 0.6980 - accuracy: 0.5024
10400/25000 [===========>..................] - ETA: 9s - loss: 0.6979 - accuracy: 0.5030
10500/25000 [===========>..................] - ETA: 9s - loss: 0.6979 - accuracy: 0.5034
10600/25000 [===========>..................] - ETA: 9s - loss: 0.6978 - accuracy: 0.5044
10700/25000 [===========>..................] - ETA: 9s - loss: 0.6978 - accuracy: 0.5047
10800/25000 [===========>..................] - ETA: 9s - loss: 0.6977 - accuracy: 0.5056
10900/25000 [============>.................] - ETA: 9s - loss: 0.6977 - accuracy: 0.5063
11000/25000 [============>.................] - ETA: 9s - loss: 0.6976 - accuracy: 0.5074
11100/25000 [============>.................] - ETA: 9s - loss: 0.6976 - accuracy: 0.5081
11200/25000 [============>.................] - ETA: 9s - loss: 0.6975 - accuracy: 0.5101
11300/25000 [============>.................] - ETA: 9s - loss: 0.6975 - accuracy: 0.5114
11400/25000 [============>.................] - ETA: 9s - loss: 0.6974 - accuracy: 0.5126
11500/25000 [============>.................] - ETA: 9s - loss: 0.6974 - accuracy: 0.5135
11600/25000 [============>.................] - ETA: 8s - loss: 0.6973 - accuracy: 0.5137
11700/25000 [=============>................] - ETA: 8s - loss: 0.6973 - accuracy: 0.5135
11800/25000 [=============>................] - ETA: 8s - loss: 0.6972 - accuracy: 0.5140
11900/25000 [=============>................] - ETA: 8s - loss: 0.6972 - accuracy: 0.5140
12000/25000 [=============>................] - ETA: 8s - loss: 0.6971 - accuracy: 0.5139
12100/25000 [=============>................] - ETA: 8s - loss: 0.6971 - accuracy: 0.5133
12200/25000 [=============>................] - ETA: 8s - loss: 0.6970 - accuracy: 0.5139
12300/25000 [=============>................] - ETA: 8s - loss: 0.6970 - accuracy: 0.5138
12400/25000 [=============>................] - ETA: 8s - loss: 0.6969 - accuracy: 0.5139
12500/25000 [==============>...............] - ETA: 8s - loss: 0.6969 - accuracy: 0.5132
12600/25000 [==============>...............] - ETA: 8s - loss: 0.6968 - accuracy: 0.5133
12700/25000 [==============>...............] - ETA: 8s - loss: 0.6967 - accuracy: 0.5131
12800/25000 [==============>...............] - ETA: 8s - loss: 0.6966 - accuracy: 0.5131
12900/25000 [==============>...............] - ETA: 8s - loss: 0.6966 - accuracy: 0.5129
13000/25000 [==============>...............] - ETA: 7s - loss: 0.6965 - accuracy: 0.5132
13100/25000 [==============>...............] - ETA: 7s - loss: 0.6964 - accuracy: 0.5133
13200/25000 [==============>...............] - ETA: 7s - loss: 0.6963 - accuracy: 0.5130
13300/25000 [==============>...............] - ETA: 7s - loss: 0.6963 - accuracy: 0.5121
13400/25000 [===============>..............] - ETA: 7s - loss: 0.6962 - accuracy: 0.5120
13500/25000 [===============>..............] - ETA: 7s - loss: 0.6960 - accuracy: 0.5118
13600/25000 [===============>..............] - ETA: 7s - loss: 0.6959 - accuracy: 0.5118
13700/25000 [===============>..............] - ETA: 7s - loss: 0.6958 - accuracy: 0.5113
13800/25000 [===============>..............] - ETA: 7s - loss: 0.6957 - accuracy: 0.5107
13900/25000 [===============>..............] - ETA: 7s - loss: 0.6955 - accuracy: 0.5102
14000/25000 [===============>..............] - ETA: 7s - loss: 0.6952 - accuracy: 0.5104
14100/25000 [===============>..............] - ETA: 7s - loss: 0.6950 - accuracy: 0.5097
14200/25000 [================>.............] - ETA: 7s - loss: 0.6948 - accuracy: 0.5099
14300/25000 [================>.............] - ETA: 7s - loss: 0.6945 - accuracy: 0.5099
14400/25000 [================>.............] - ETA: 6s - loss: 0.6943 - accuracy: 0.5096
14500/25000 [================>.............] - ETA: 6s - loss: 0.6938 - accuracy: 0.5102
14600/25000 [================>.............] - ETA: 6s - loss: 0.6933 - accuracy: 0.5102
14700/25000 [================>.............] - ETA: 6s - loss: 0.6930 - accuracy: 0.5105
14800/25000 [================>.............] - ETA: 6s - loss: 0.6926 - accuracy: 0.5111
14900/25000 [================>.............] - ETA: 6s - loss: 0.6922 - accuracy: 0.5120
15000/25000 [=================>............] - ETA: 6s - loss: 0.6918 - accuracy: 0.5127
15100/25000 [=================>............] - ETA: 6s - loss: 0.6913 - accuracy: 0.5133
15200/25000 [=================>............] - ETA: 6s - loss: 0.6906 - accuracy: 0.5143
15300/25000 [=================>............] - ETA: 6s - loss: 0.6900 - accuracy: 0.5158
15400/25000 [=================>............] - ETA: 6s - loss: 0.6894 - accuracy: 0.5171
15500/25000 [=================>............] - ETA: 6s - loss: 0.6889 - accuracy: 0.5183
15600/25000 [=================>............] - ETA: 6s - loss: 0.6884 - accuracy: 0.5199
15700/25000 [=================>............] - ETA: 6s - loss: 0.6880 - accuracy: 0.5213
15800/25000 [=================>............] - ETA: 6s - loss: 0.6877 - accuracy: 0.5229
15900/25000 [==================>...........] - ETA: 5s - loss: 0.6869 - accuracy: 0.5249
16000/25000 [==================>...........] - ETA: 5s - loss: 0.6863 - accuracy: 0.5265
16100/25000 [==================>...........] - ETA: 5s - loss: 0.6859 - accuracy: 0.5276
16200/25000 [==================>...........] - ETA: 5s - loss: 0.6855 - accuracy: 0.5293
16300/25000 [==================>...........] - ETA: 5s - loss: 0.6850 - accuracy: 0.5309
16400/25000 [==================>...........] - ETA: 5s - loss: 0.6845 - accuracy: 0.5326
16500/25000 [==================>...........] - ETA: 5s - loss: 0.6838 - accuracy: 0.5342
16600/25000 [==================>...........] - ETA: 5s - loss: 0.6832 - accuracy: 0.5349
16700/25000 [===================>..........] - ETA: 5s - loss: 0.6827 - accuracy: 0.5363
16800/25000 [===================>..........] - ETA: 5s - loss: 0.6821 - accuracy: 0.5376
16900/25000 [===================>..........] - ETA: 5s - loss: 0.6815 - accuracy: 0.5387
17000/25000 [===================>..........] - ETA: 5s - loss: 0.6810 - accuracy: 0.5404
17100/25000 [===================>..........] - ETA: 5s - loss: 0.6805 - accuracy: 0.5420
17200/25000 [===================>..........] - ETA: 5s - loss: 0.6798 - accuracy: 0.5437
17300/25000 [===================>..........] - ETA: 5s - loss: 0.6794 - accuracy: 0.5453
17400/25000 [===================>..........] - ETA: 4s - loss: 0.6787 - accuracy: 0.5471
17500/25000 [====================>.........] - ETA: 4s - loss: 0.6781 - accuracy: 0.5487
17600/25000 [====================>.........] - ETA: 4s - loss: 0.6774 - accuracy: 0.5506
17700/25000 [====================>.........] - ETA: 4s - loss: 0.6769 - accuracy: 0.5522
17800/25000 [====================>.........] - ETA: 4s - loss: 0.6762 - accuracy: 0.5537
17900/25000 [====================>.........] - ETA: 4s - loss: 0.6756 - accuracy: 0.5550
18000/25000 [====================>.........] - ETA: 4s - loss: 0.6749 - accuracy: 0.5568
18100/25000 [====================>.........] - ETA: 4s - loss: 0.6745 - accuracy: 0.5581
18200/25000 [====================>.........] - ETA: 4s - loss: 0.6740 - accuracy: 0.5596
18300/25000 [====================>.........] - ETA: 4s - loss: 0.6734 - accuracy: 0.5611
18400/25000 [=====================>........] - ETA: 4s - loss: 0.6730 - accuracy: 0.5619
18500/25000 [=====================>........] - ETA: 4s - loss: 0.6725 - accuracy: 0.5632
18600/25000 [=====================>........] - ETA: 4s - loss: 0.6721 - accuracy: 0.5645
18700/25000 [=====================>........] - ETA: 4s - loss: 0.6715 - accuracy: 0.5661
18800/25000 [=====================>........] - ETA: 4s - loss: 0.6710 - accuracy: 0.5675
18900/25000 [=====================>........] - ETA: 4s - loss: 0.6705 - accuracy: 0.5689
19000/25000 [=====================>........] - ETA: 3s - loss: 0.6702 - accuracy: 0.5697
19100/25000 [=====================>........] - ETA: 3s - loss: 0.6697 - accuracy: 0.5712
19200/25000 [======================>.......] - ETA: 3s - loss: 0.6692 - accuracy: 0.5724
19300/25000 [======================>.......] - ETA: 3s - loss: 0.6687 - accuracy: 0.5739
19400/25000 [======================>.......] - ETA: 3s - loss: 0.6682 - accuracy: 0.5754
19500/25000 [======================>.......] - ETA: 3s - loss: 0.6677 - accuracy: 0.5769
19600/25000 [======================>.......] - ETA: 3s - loss: 0.6671 - accuracy: 0.5784
19700/25000 [======================>.......] - ETA: 3s - loss: 0.6667 - accuracy: 0.5796
19800/25000 [======================>.......] - ETA: 3s - loss: 0.6663 - accuracy: 0.5807
19900/25000 [======================>.......] - ETA: 3s - loss: 0.6658 - accuracy: 0.5819
20000/25000 [=======================>......] - ETA: 3s - loss: 0.6652 - accuracy: 0.5834
20100/25000 [=======================>......] - ETA: 3s - loss: 0.6647 - accuracy: 0.5847
20200/25000 [=======================>......] - ETA: 3s - loss: 0.6641 - accuracy: 0.5859
20300/25000 [=======================>......] - ETA: 3s - loss: 0.6634 - accuracy: 0.5876
20400/25000 [=======================>......] - ETA: 3s - loss: 0.6630 - accuracy: 0.5887
20500/25000 [=======================>......] - ETA: 2s - loss: 0.6625 - accuracy: 0.5900
20600/25000 [=======================>......] - ETA: 2s - loss: 0.6622 - accuracy: 0.5911
20700/25000 [=======================>......] - ETA: 2s - loss: 0.6617 - accuracy: 0.5925
20800/25000 [=======================>......] - ETA: 2s - loss: 0.6613 - accuracy: 0.5936
20900/25000 [========================>.....] - ETA: 2s - loss: 0.6606 - accuracy: 0.5951
21000/25000 [========================>.....] - ETA: 2s - loss: 0.6602 - accuracy: 0.5962
21100/25000 [========================>.....] - ETA: 2s - loss: 0.6597 - accuracy: 0.5974
21200/25000 [========================>.....] - ETA: 2s - loss: 0.6591 - accuracy: 0.5985
21300/25000 [========================>.....] - ETA: 2s - loss: 0.6586 - accuracy: 0.5997
21400/25000 [========================>.....] - ETA: 2s - loss: 0.6582 - accuracy: 0.6008
21500/25000 [========================>.....] - ETA: 2s - loss: 0.6577 - accuracy: 0.6022
21600/25000 [========================>.....] - ETA: 2s - loss: 0.6571 - accuracy: 0.6034
21700/25000 [=========================>....] - ETA: 2s - loss: 0.6567 - accuracy: 0.6044
21800/25000 [=========================>....] - ETA: 2s - loss: 0.6562 - accuracy: 0.6056
21900/25000 [=========================>....] - ETA: 2s - loss: 0.6557 - accuracy: 0.6069
22000/25000 [=========================>....] - ETA: 1s - loss: 0.6554 - accuracy: 0.6079
22100/25000 [=========================>....] - ETA: 1s - loss: 0.6548 - accuracy: 0.6092
22200/25000 [=========================>....] - ETA: 1s - loss: 0.6542 - accuracy: 0.6105
22300/25000 [=========================>....] - ETA: 1s - loss: 0.6536 - accuracy: 0.6118
22400/25000 [=========================>....] - ETA: 1s - loss: 0.6532 - accuracy: 0.6128
22500/25000 [==========================>...] - ETA: 1s - loss: 0.6527 - accuracy: 0.6140
22600/25000 [==========================>...] - ETA: 1s - loss: 0.6522 - accuracy: 0.6153
22700/25000 [==========================>...] - ETA: 1s - loss: 0.6517 - accuracy: 0.6164
22800/25000 [==========================>...] - ETA: 1s - loss: 0.6512 - accuracy: 0.6176
22900/25000 [==========================>...] - ETA: 1s - loss: 0.6507 - accuracy: 0.6186
23000/25000 [==========================>...] - ETA: 1s - loss: 0.6501 - accuracy: 0.6200
23100/25000 [==========================>...] - ETA: 1s - loss: 0.6497 - accuracy: 0.6211
23200/25000 [==========================>...] - ETA: 1s - loss: 0.6495 - accuracy: 0.6217
23300/25000 [==========================>...] - ETA: 1s - loss: 0.6490 - accuracy: 0.6228
23400/25000 [===========================>..] - ETA: 1s - loss: 0.6486 - accuracy: 0.6238
23500/25000 [===========================>..] - ETA: 0s - loss: 0.6482 - accuracy: 0.6248
23600/25000 [===========================>..] - ETA: 0s - loss: 0.6477 - accuracy: 0.6259
23700/25000 [===========================>..] - ETA: 0s - loss: 0.6474 - accuracy: 0.6269
23800/25000 [===========================>..] - ETA: 0s - loss: 0.6469 - accuracy: 0.6281
23900/25000 [===========================>..] - ETA: 0s - loss: 0.6465 - accuracy: 0.6292
24000/25000 [===========================>..] - ETA: 0s - loss: 0.6461 - accuracy: 0.6299
24100/25000 [===========================>..] - ETA: 0s - loss: 0.6457 - accuracy: 0.6308
24200/25000 [============================>.] - ETA: 0s - loss: 0.6453 - accuracy: 0.6319
24300/25000 [============================>.] - ETA: 0s - loss: 0.6448 - accuracy: 0.6330
24400/25000 [============================>.] - ETA: 0s - loss: 0.6443 - accuracy: 0.6341
24500/25000 [============================>.] - ETA: 0s - loss: 0.6438 - accuracy: 0.6352
24600/25000 [============================>.] - ETA: 0s - loss: 0.6435 - accuracy: 0.6358
24700/25000 [============================>.] - ETA: 0s - loss: 0.6430 - accuracy: 0.6368
24800/25000 [============================>.] - ETA: 0s - loss: 0.6427 - accuracy: 0.6377
24900/25000 [============================>.] - ETA: 0s - loss: 0.6421 - accuracy: 0.6390
25000/25000 [==============================] - 21s 827us/step - loss: 0.6417 - accuracy: 0.6400 - val_loss: 0.5493 - val_accuracy: 0.8558
Epoch 2/10

  100/25000 [..............................] - ETA: 15s - loss: 0.5558 - accuracy: 0.8500
  200/25000 [..............................] - ETA: 15s - loss: 0.5534 - accuracy: 0.8550
  300/25000 [..............................] - ETA: 15s - loss: 0.5415 - accuracy: 0.8767
  400/25000 [..............................] - ETA: 15s - loss: 0.5395 - accuracy: 0.8825
  500/25000 [..............................] - ETA: 15s - loss: 0.5312 - accuracy: 0.8880
  600/25000 [..............................] - ETA: 15s - loss: 0.5245 - accuracy: 0.8850
  700/25000 [..............................] - ETA: 15s - loss: 0.5176 - accuracy: 0.8971
  800/25000 [..............................] - ETA: 15s - loss: 0.5181 - accuracy: 0.8913
  900/25000 [>.............................] - ETA: 15s - loss: 0.5230 - accuracy: 0.8900
 1000/25000 [>.............................] - ETA: 15s - loss: 0.5211 - accuracy: 0.8910
 1100/25000 [>.............................] - ETA: 15s - loss: 0.5236 - accuracy: 0.8891
 1200/25000 [>.............................] - ETA: 15s - loss: 0.5233 - accuracy: 0.8900
 1300/25000 [>.............................] - ETA: 15s - loss: 0.5240 - accuracy: 0.8869
 1400/25000 [>.............................] - ETA: 14s - loss: 0.5233 - accuracy: 0.8886
 1500/25000 [>.............................] - ETA: 14s - loss: 0.5272 - accuracy: 0.8867
 1600/25000 [>.............................] - ETA: 14s - loss: 0.5261 - accuracy: 0.8869
 1700/25000 [=>............................] - ETA: 14s - loss: 0.5258 - accuracy: 0.8876
 1800/25000 [=>............................] - ETA: 14s - loss: 0.5255 - accuracy: 0.8883
 1900/25000 [=>............................] - ETA: 14s - loss: 0.5265 - accuracy: 0.8895
 2000/25000 [=>............................] - ETA: 14s - loss: 0.5282 - accuracy: 0.8880
 2100/25000 [=>............................] - ETA: 14s - loss: 0.5267 - accuracy: 0.8881
 2200/25000 [=>............................] - ETA: 14s - loss: 0.5256 - accuracy: 0.8895
 2300/25000 [=>............................] - ETA: 14s - loss: 0.5263 - accuracy: 0.8896
 2400/25000 [=>............................] - ETA: 14s - loss: 0.5274 - accuracy: 0.8888
 2500/25000 [==>...........................] - ETA: 14s - loss: 0.5269 - accuracy: 0.8884
 2600/25000 [==>...........................] - ETA: 14s - loss: 0.5269 - accuracy: 0.8885
 2700/25000 [==>...........................] - ETA: 14s - loss: 0.5269 - accuracy: 0.8885
 2800/25000 [==>...........................] - ETA: 14s - loss: 0.5271 - accuracy: 0.8879
 2900/25000 [==>...........................] - ETA: 13s - loss: 0.5269 - accuracy: 0.8890
 3000/25000 [==>...........................] - ETA: 13s - loss: 0.5263 - accuracy: 0.8880
 3100/25000 [==>...........................] - ETA: 13s - loss: 0.5266 - accuracy: 0.8877
 3200/25000 [==>...........................] - ETA: 13s - loss: 0.5274 - accuracy: 0.8869
 3300/25000 [==>...........................] - ETA: 13s - loss: 0.5264 - accuracy: 0.8876
 3400/25000 [===>..........................] - ETA: 13s - loss: 0.5268 - accuracy: 0.8862
 3500/25000 [===>..........................] - ETA: 13s - loss: 0.5265 - accuracy: 0.8863
 3600/25000 [===>..........................] - ETA: 13s - loss: 0.5279 - accuracy: 0.8850
 3700/25000 [===>..........................] - ETA: 13s - loss: 0.5274 - accuracy: 0.8851
 3800/25000 [===>..........................] - ETA: 13s - loss: 0.5274 - accuracy: 0.8855
 3900/25000 [===>..........................] - ETA: 13s - loss: 0.5277 - accuracy: 0.8856
 4000/25000 [===>..........................] - ETA: 13s - loss: 0.5278 - accuracy: 0.8860
 4100/25000 [===>..........................] - ETA: 13s - loss: 0.5278 - accuracy: 0.8854
 4200/25000 [====>.........................] - ETA: 13s - loss: 0.5282 - accuracy: 0.8840
 4300/25000 [====>.........................] - ETA: 13s - loss: 0.5279 - accuracy: 0.8844
 4400/25000 [====>.........................] - ETA: 12s - loss: 0.5278 - accuracy: 0.8836
 4500/25000 [====>.........................] - ETA: 12s - loss: 0.5276 - accuracy: 0.8840
 4600/25000 [====>.........................] - ETA: 12s - loss: 0.5272 - accuracy: 0.8848
 4700/25000 [====>.........................] - ETA: 12s - loss: 0.5274 - accuracy: 0.8843
 4800/25000 [====>.........................] - ETA: 12s - loss: 0.5287 - accuracy: 0.8829
 4900/25000 [====>.........................] - ETA: 12s - loss: 0.5283 - accuracy: 0.8822
 5000/25000 [=====>........................] - ETA: 12s - loss: 0.5279 - accuracy: 0.8826
 5100/25000 [=====>........................] - ETA: 12s - loss: 0.5275 - accuracy: 0.8827
 5200/25000 [=====>........................] - ETA: 12s - loss: 0.5273 - accuracy: 0.8823
 5300/25000 [=====>........................] - ETA: 12s - loss: 0.5266 - accuracy: 0.8832
 5400/25000 [=====>........................] - ETA: 12s - loss: 0.5275 - accuracy: 0.8811
 5500/25000 [=====>........................] - ETA: 12s - loss: 0.5269 - accuracy: 0.8813
 5600/25000 [=====>........................] - ETA: 12s - loss: 0.5266 - accuracy: 0.8816
 5700/25000 [=====>........................] - ETA: 12s - loss: 0.5269 - accuracy: 0.8809
 5800/25000 [=====>........................] - ETA: 12s - loss: 0.5265 - accuracy: 0.8814
 5900/25000 [======>.......................] - ETA: 12s - loss: 0.5264 - accuracy: 0.8814
 6000/25000 [======>.......................] - ETA: 12s - loss: 0.5264 - accuracy: 0.8818
 6100/25000 [======>.......................] - ETA: 12s - loss: 0.5263 - accuracy: 0.8815
 6200/25000 [======>.......................] - ETA: 11s - loss: 0.5265 - accuracy: 0.8813
 6300/25000 [======>.......................] - ETA: 11s - loss: 0.5261 - accuracy: 0.8816
 6400/25000 [======>.......................] - ETA: 11s - loss: 0.5262 - accuracy: 0.8813
 6500/25000 [======>.......................] - ETA: 11s - loss: 0.5266 - accuracy: 0.8811
 6600/25000 [======>.......................] - ETA: 11s - loss: 0.5271 - accuracy: 0.8809
 6700/25000 [=======>......................] - ETA: 11s - loss: 0.5269 - accuracy: 0.8810
 6800/25000 [=======>......................] - ETA: 11s - loss: 0.5271 - accuracy: 0.8813
 6900/25000 [=======>......................] - ETA: 11s - loss: 0.5268 - accuracy: 0.8814
 7000/25000 [=======>......................] - ETA: 11s - loss: 0.5267 - accuracy: 0.8819
 7100/25000 [=======>......................] - ETA: 11s - loss: 0.5269 - accuracy: 0.8815
 7200/25000 [=======>......................] - ETA: 11s - loss: 0.5264 - accuracy: 0.8819
 7300/25000 [=======>......................] - ETA: 11s - loss: 0.5264 - accuracy: 0.8819
 7400/25000 [=======>......................] - ETA: 11s - loss: 0.5259 - accuracy: 0.8826
 7500/25000 [========>.....................] - ETA: 11s - loss: 0.5261 - accuracy: 0.8820
 7600/25000 [========>.....................] - ETA: 11s - loss: 0.5254 - accuracy: 0.8825
 7700/25000 [========>.....................] - ETA: 10s - loss: 0.5261 - accuracy: 0.8817
 7800/25000 [========>.....................] - ETA: 10s - loss: 0.5262 - accuracy: 0.8813
 7900/25000 [========>.....................] - ETA: 10s - loss: 0.5258 - accuracy: 0.8814
 8000/25000 [========>.....................] - ETA: 10s - loss: 0.5255 - accuracy: 0.8813
 8100/25000 [========>.....................] - ETA: 10s - loss: 0.5255 - accuracy: 0.8810
 8200/25000 [========>.....................] - ETA: 10s - loss: 0.5257 - accuracy: 0.8799
 8300/25000 [========>.....................] - ETA: 10s - loss: 0.5255 - accuracy: 0.8795
 8400/25000 [=========>....................] - ETA: 10s - loss: 0.5258 - accuracy: 0.8792
 8500/25000 [=========>....................] - ETA: 10s - loss: 0.5263 - accuracy: 0.8786
 8600/25000 [=========>....................] - ETA: 10s - loss: 0.5264 - accuracy: 0.8784
 8700/25000 [=========>....................] - ETA: 10s - loss: 0.5269 - accuracy: 0.8779
 8800/25000 [=========>....................] - ETA: 10s - loss: 0.5272 - accuracy: 0.8776
 8900/25000 [=========>....................] - ETA: 10s - loss: 0.5270 - accuracy: 0.8775
 9000/25000 [=========>....................] - ETA: 10s - loss: 0.5275 - accuracy: 0.8771
 9100/25000 [=========>....................] - ETA: 10s - loss: 0.5272 - accuracy: 0.8767
 9200/25000 [==========>...................] - ETA: 9s - loss: 0.5274 - accuracy: 0.8760 
 9300/25000 [==========>...................] - ETA: 9s - loss: 0.5269 - accuracy: 0.8768
 9400/25000 [==========>...................] - ETA: 9s - loss: 0.5269 - accuracy: 0.8768
 9500/25000 [==========>...................] - ETA: 9s - loss: 0.5269 - accuracy: 0.8767
 9600/25000 [==========>...................] - ETA: 9s - loss: 0.5266 - accuracy: 0.8767
 9700/25000 [==========>...................] - ETA: 9s - loss: 0.5264 - accuracy: 0.8765
 9800/25000 [==========>...................] - ETA: 9s - loss: 0.5265 - accuracy: 0.8761
 9900/25000 [==========>...................] - ETA: 9s - loss: 0.5263 - accuracy: 0.8764
10000/25000 [===========>..................] - ETA: 9s - loss: 0.5264 - accuracy: 0.8763
10100/25000 [===========>..................] - ETA: 9s - loss: 0.5267 - accuracy: 0.8764
10200/25000 [===========>..................] - ETA: 9s - loss: 0.5265 - accuracy: 0.8766
10300/25000 [===========>..................] - ETA: 9s - loss: 0.5262 - accuracy: 0.8766
10400/25000 [===========>..................] - ETA: 9s - loss: 0.5260 - accuracy: 0.8769
10500/25000 [===========>..................] - ETA: 9s - loss: 0.5259 - accuracy: 0.8770
10600/25000 [===========>..................] - ETA: 9s - loss: 0.5258 - accuracy: 0.8773
10700/25000 [===========>..................] - ETA: 9s - loss: 0.5259 - accuracy: 0.8775
10800/25000 [===========>..................] - ETA: 8s - loss: 0.5255 - accuracy: 0.8774
10900/25000 [============>.................] - ETA: 8s - loss: 0.5253 - accuracy: 0.8775
11000/25000 [============>.................] - ETA: 8s - loss: 0.5252 - accuracy: 0.8775
11100/25000 [============>.................] - ETA: 8s - loss: 0.5251 - accuracy: 0.8780
11200/25000 [============>.................] - ETA: 8s - loss: 0.5251 - accuracy: 0.8776
11300/25000 [============>.................] - ETA: 8s - loss: 0.5249 - accuracy: 0.8782
11400/25000 [============>.................] - ETA: 8s - loss: 0.5249 - accuracy: 0.8781
11500/25000 [============>.................] - ETA: 8s - loss: 0.5246 - accuracy: 0.8785
11600/25000 [============>.................] - ETA: 8s - loss: 0.5249 - accuracy: 0.8780
11700/25000 [=============>................] - ETA: 8s - loss: 0.5247 - accuracy: 0.8783
11800/25000 [=============>................] - ETA: 8s - loss: 0.5244 - accuracy: 0.8787
11900/25000 [=============>................] - ETA: 8s - loss: 0.5243 - accuracy: 0.8787
12000/25000 [=============>................] - ETA: 8s - loss: 0.5244 - accuracy: 0.8788
12100/25000 [=============>................] - ETA: 8s - loss: 0.5245 - accuracy: 0.8785
12200/25000 [=============>................] - ETA: 8s - loss: 0.5244 - accuracy: 0.8785
12300/25000 [=============>................] - ETA: 8s - loss: 0.5244 - accuracy: 0.8783
12400/25000 [=============>................] - ETA: 7s - loss: 0.5246 - accuracy: 0.8779
12500/25000 [==============>...............] - ETA: 7s - loss: 0.5245 - accuracy: 0.8782
12600/25000 [==============>...............] - ETA: 7s - loss: 0.5245 - accuracy: 0.8781
12700/25000 [==============>...............] - ETA: 7s - loss: 0.5247 - accuracy: 0.8776
12800/25000 [==============>...............] - ETA: 7s - loss: 0.5245 - accuracy: 0.8778
12900/25000 [==============>...............] - ETA: 7s - loss: 0.5245 - accuracy: 0.8777
13000/25000 [==============>...............] - ETA: 7s - loss: 0.5246 - accuracy: 0.8776
13100/25000 [==============>...............] - ETA: 7s - loss: 0.5244 - accuracy: 0.8779
13200/25000 [==============>...............] - ETA: 7s - loss: 0.5246 - accuracy: 0.8773
13300/25000 [==============>...............] - ETA: 7s - loss: 0.5247 - accuracy: 0.8768
13400/25000 [===============>..............] - ETA: 7s - loss: 0.5244 - accuracy: 0.8771
13500/25000 [===============>..............] - ETA: 7s - loss: 0.5244 - accuracy: 0.8773
13600/25000 [===============>..............] - ETA: 7s - loss: 0.5243 - accuracy: 0.8771
13700/25000 [===============>..............] - ETA: 7s - loss: 0.5244 - accuracy: 0.8768
13800/25000 [===============>..............] - ETA: 7s - loss: 0.5242 - accuracy: 0.8769
13900/25000 [===============>..............] - ETA: 6s - loss: 0.5243 - accuracy: 0.8766
14000/25000 [===============>..............] - ETA: 6s - loss: 0.5243 - accuracy: 0.8768
14100/25000 [===============>..............] - ETA: 6s - loss: 0.5241 - accuracy: 0.8769
14200/25000 [================>.............] - ETA: 6s - loss: 0.5241 - accuracy: 0.8768
14300/25000 [================>.............] - ETA: 6s - loss: 0.5241 - accuracy: 0.8766
14400/25000 [================>.............] - ETA: 6s - loss: 0.5241 - accuracy: 0.8764
14500/25000 [================>.............] - ETA: 6s - loss: 0.5242 - accuracy: 0.8763
14600/25000 [================>.............] - ETA: 6s - loss: 0.5244 - accuracy: 0.8761
14700/25000 [================>.............] - ETA: 6s - loss: 0.5241 - accuracy: 0.8765
14800/25000 [================>.............] - ETA: 6s - loss: 0.5238 - accuracy: 0.8768
14900/25000 [================>.............] - ETA: 6s - loss: 0.5236 - accuracy: 0.8768
15000/25000 [=================>............] - ETA: 6s - loss: 0.5233 - accuracy: 0.8771
15100/25000 [=================>............] - ETA: 6s - loss: 0.5235 - accuracy: 0.8767
15200/25000 [=================>............] - ETA: 6s - loss: 0.5235 - accuracy: 0.8763
15300/25000 [=================>............] - ETA: 6s - loss: 0.5234 - accuracy: 0.8762
15400/25000 [=================>............] - ETA: 6s - loss: 0.5233 - accuracy: 0.8761
15500/25000 [=================>............] - ETA: 5s - loss: 0.5231 - accuracy: 0.8763
15600/25000 [=================>............] - ETA: 5s - loss: 0.5232 - accuracy: 0.8761
15700/25000 [=================>............] - ETA: 5s - loss: 0.5229 - accuracy: 0.8763
15800/25000 [=================>............] - ETA: 5s - loss: 0.5228 - accuracy: 0.8762
15900/25000 [==================>...........] - ETA: 5s - loss: 0.5228 - accuracy: 0.8762
16000/25000 [==================>...........] - ETA: 5s - loss: 0.5226 - accuracy: 0.8763
16100/25000 [==================>...........] - ETA: 5s - loss: 0.5225 - accuracy: 0.8762
16200/25000 [==================>...........] - ETA: 5s - loss: 0.5223 - accuracy: 0.8764
16300/25000 [==================>...........] - ETA: 5s - loss: 0.5221 - accuracy: 0.8766
16400/25000 [==================>...........] - ETA: 5s - loss: 0.5219 - accuracy: 0.8769
16500/25000 [==================>...........] - ETA: 5s - loss: 0.5217 - accuracy: 0.8770
16600/25000 [==================>...........] - ETA: 5s - loss: 0.5216 - accuracy: 0.8769
16700/25000 [===================>..........] - ETA: 5s - loss: 0.5214 - accuracy: 0.8770
16800/25000 [===================>..........] - ETA: 5s - loss: 0.5215 - accuracy: 0.8768
16900/25000 [===================>..........] - ETA: 5s - loss: 0.5214 - accuracy: 0.8769
17000/25000 [===================>..........] - ETA: 5s - loss: 0.5212 - accuracy: 0.8769
17100/25000 [===================>..........] - ETA: 4s - loss: 0.5213 - accuracy: 0.8768
17200/25000 [===================>..........] - ETA: 4s - loss: 0.5213 - accuracy: 0.8765
17300/25000 [===================>..........] - ETA: 4s - loss: 0.5211 - accuracy: 0.8766
17400/25000 [===================>..........] - ETA: 4s - loss: 0.5210 - accuracy: 0.8764
17500/25000 [====================>.........] - ETA: 4s - loss: 0.5210 - accuracy: 0.8762
17600/25000 [====================>.........] - ETA: 4s - loss: 0.5209 - accuracy: 0.8764
17700/25000 [====================>.........] - ETA: 4s - loss: 0.5207 - accuracy: 0.8766
17800/25000 [====================>.........] - ETA: 4s - loss: 0.5206 - accuracy: 0.8766
17900/25000 [====================>.........] - ETA: 4s - loss: 0.5205 - accuracy: 0.8769
18000/25000 [====================>.........] - ETA: 4s - loss: 0.5205 - accuracy: 0.8768
18100/25000 [====================>.........] - ETA: 4s - loss: 0.5204 - accuracy: 0.8770
18200/25000 [====================>.........] - ETA: 4s - loss: 0.5204 - accuracy: 0.8769
18300/25000 [====================>.........] - ETA: 4s - loss: 0.5203 - accuracy: 0.8768
18400/25000 [=====================>........] - ETA: 4s - loss: 0.5202 - accuracy: 0.8769
18500/25000 [=====================>........] - ETA: 4s - loss: 0.5201 - accuracy: 0.8770
18600/25000 [=====================>........] - ETA: 4s - loss: 0.5200 - accuracy: 0.8771
18700/25000 [=====================>........] - ETA: 3s - loss: 0.5200 - accuracy: 0.8769
18800/25000 [=====================>........] - ETA: 3s - loss: 0.5198 - accuracy: 0.8770
18900/25000 [=====================>........] - ETA: 3s - loss: 0.5197 - accuracy: 0.8771
19000/25000 [=====================>........] - ETA: 3s - loss: 0.5195 - accuracy: 0.8774
19100/25000 [=====================>........] - ETA: 3s - loss: 0.5196 - accuracy: 0.8773
19200/25000 [======================>.......] - ETA: 3s - loss: 0.5195 - accuracy: 0.8774
19300/25000 [======================>.......] - ETA: 3s - loss: 0.5194 - accuracy: 0.8773
19400/25000 [======================>.......] - ETA: 3s - loss: 0.5193 - accuracy: 0.8773
19500/25000 [======================>.......] - ETA: 3s - loss: 0.5192 - accuracy: 0.8771
19600/25000 [======================>.......] - ETA: 3s - loss: 0.5192 - accuracy: 0.8769
19700/25000 [======================>.......] - ETA: 3s - loss: 0.5191 - accuracy: 0.8771
19800/25000 [======================>.......] - ETA: 3s - loss: 0.5190 - accuracy: 0.8771
19900/25000 [======================>.......] - ETA: 3s - loss: 0.5189 - accuracy: 0.8771
20000/25000 [=======================>......] - ETA: 3s - loss: 0.5188 - accuracy: 0.8772
20100/25000 [=======================>......] - ETA: 3s - loss: 0.5187 - accuracy: 0.8775
20200/25000 [=======================>......] - ETA: 3s - loss: 0.5186 - accuracy: 0.8773
20300/25000 [=======================>......] - ETA: 2s - loss: 0.5185 - accuracy: 0.8775
20400/25000 [=======================>......] - ETA: 2s - loss: 0.5184 - accuracy: 0.8775
20500/25000 [=======================>......] - ETA: 2s - loss: 0.5185 - accuracy: 0.8772
20600/25000 [=======================>......] - ETA: 2s - loss: 0.5184 - accuracy: 0.8773
20700/25000 [=======================>......] - ETA: 2s - loss: 0.5181 - accuracy: 0.8777
20800/25000 [=======================>......] - ETA: 2s - loss: 0.5180 - accuracy: 0.8776
20900/25000 [========================>.....] - ETA: 2s - loss: 0.5178 - accuracy: 0.8779
21000/25000 [========================>.....] - ETA: 2s - loss: 0.5177 - accuracy: 0.8778
21100/25000 [========================>.....] - ETA: 2s - loss: 0.5177 - accuracy: 0.8778
21200/25000 [========================>.....] - ETA: 2s - loss: 0.5176 - accuracy: 0.8779
21300/25000 [========================>.....] - ETA: 2s - loss: 0.5174 - accuracy: 0.8779
21400/25000 [========================>.....] - ETA: 2s - loss: 0.5173 - accuracy: 0.8779
21500/25000 [========================>.....] - ETA: 2s - loss: 0.5172 - accuracy: 0.8780
21600/25000 [========================>.....] - ETA: 2s - loss: 0.5170 - accuracy: 0.8781
21700/25000 [=========================>....] - ETA: 2s - loss: 0.5170 - accuracy: 0.8780
21800/25000 [=========================>....] - ETA: 2s - loss: 0.5170 - accuracy: 0.8778
21900/25000 [=========================>....] - ETA: 1s - loss: 0.5169 - accuracy: 0.8779
22000/25000 [=========================>....] - ETA: 1s - loss: 0.5169 - accuracy: 0.8776
22100/25000 [=========================>....] - ETA: 1s - loss: 0.5171 - accuracy: 0.8774
22200/25000 [=========================>....] - ETA: 1s - loss: 0.5171 - accuracy: 0.8772
22300/25000 [=========================>....] - ETA: 1s - loss: 0.5170 - accuracy: 0.8771
22400/25000 [=========================>....] - ETA: 1s - loss: 0.5168 - accuracy: 0.8774
22500/25000 [==========================>...] - ETA: 1s - loss: 0.5167 - accuracy: 0.8775
22600/25000 [==========================>...] - ETA: 1s - loss: 0.5167 - accuracy: 0.8773
22700/25000 [==========================>...] - ETA: 1s - loss: 0.5166 - accuracy: 0.8773
22800/25000 [==========================>...] - ETA: 1s - loss: 0.5166 - accuracy: 0.8772
22900/25000 [==========================>...] - ETA: 1s - loss: 0.5167 - accuracy: 0.8769
23000/25000 [==========================>...] - ETA: 1s - loss: 0.5166 - accuracy: 0.8769
23100/25000 [==========================>...] - ETA: 1s - loss: 0.5166 - accuracy: 0.8766
23200/25000 [==========================>...] - ETA: 1s - loss: 0.5165 - accuracy: 0.8767
23300/25000 [==========================>...] - ETA: 1s - loss: 0.5164 - accuracy: 0.8769
23400/25000 [===========================>..] - ETA: 1s - loss: 0.5165 - accuracy: 0.8766
23500/25000 [===========================>..] - ETA: 0s - loss: 0.5164 - accuracy: 0.8766
23600/25000 [===========================>..] - ETA: 0s - loss: 0.5163 - accuracy: 0.8766
23700/25000 [===========================>..] - ETA: 0s - loss: 0.5162 - accuracy: 0.8768
23800/25000 [===========================>..] - ETA: 0s - loss: 0.5161 - accuracy: 0.8767
23900/25000 [===========================>..] - ETA: 0s - loss: 0.5161 - accuracy: 0.8767
24000/25000 [===========================>..] - ETA: 0s - loss: 0.5161 - accuracy: 0.8767
24100/25000 [===========================>..] - ETA: 0s - loss: 0.5159 - accuracy: 0.8768
24200/25000 [============================>.] - ETA: 0s - loss: 0.5157 - accuracy: 0.8769
24300/25000 [============================>.] - ETA: 0s - loss: 0.5156 - accuracy: 0.8767
24400/25000 [============================>.] - ETA: 0s - loss: 0.5155 - accuracy: 0.8769
24500/25000 [============================>.] - ETA: 0s - loss: 0.5154 - accuracy: 0.8771
24600/25000 [============================>.] - ETA: 0s - loss: 0.5153 - accuracy: 0.8770
24700/25000 [============================>.] - ETA: 0s - loss: 0.5152 - accuracy: 0.8771
24800/25000 [============================>.] - ETA: 0s - loss: 0.5151 - accuracy: 0.8771
24900/25000 [============================>.] - ETA: 0s - loss: 0.5149 - accuracy: 0.8771
25000/25000 [==============================] - 20s 794us/step - loss: 0.5148 - accuracy: 0.8771 - val_loss: 0.5029 - val_accuracy: 0.8667
Epoch 3/10

  100/25000 [..............................] - ETA: 15s - loss: 0.4559 - accuracy: 0.9200
  200/25000 [..............................] - ETA: 16s - loss: 0.4520 - accuracy: 0.9400
  300/25000 [..............................] - ETA: 15s - loss: 0.4616 - accuracy: 0.9300
  400/25000 [..............................] - ETA: 15s - loss: 0.4644 - accuracy: 0.9325
  500/25000 [..............................] - ETA: 15s - loss: 0.4653 - accuracy: 0.9320
  600/25000 [..............................] - ETA: 15s - loss: 0.4706 - accuracy: 0.9250
  700/25000 [..............................] - ETA: 15s - loss: 0.4689 - accuracy: 0.9214
  800/25000 [..............................] - ETA: 15s - loss: 0.4704 - accuracy: 0.9150
  900/25000 [>.............................] - ETA: 15s - loss: 0.4688 - accuracy: 0.9167
 1000/25000 [>.............................] - ETA: 15s - loss: 0.4701 - accuracy: 0.9150
 1100/25000 [>.............................] - ETA: 15s - loss: 0.4737 - accuracy: 0.9109
 1200/25000 [>.............................] - ETA: 14s - loss: 0.4740 - accuracy: 0.9100
 1300/25000 [>.............................] - ETA: 14s - loss: 0.4722 - accuracy: 0.9108
 1400/25000 [>.............................] - ETA: 14s - loss: 0.4736 - accuracy: 0.9071
 1500/25000 [>.............................] - ETA: 14s - loss: 0.4742 - accuracy: 0.9087
 1600/25000 [>.............................] - ETA: 14s - loss: 0.4729 - accuracy: 0.9087
 1700/25000 [=>............................] - ETA: 14s - loss: 0.4718 - accuracy: 0.9106
 1800/25000 [=>............................] - ETA: 14s - loss: 0.4716 - accuracy: 0.9100
 1900/25000 [=>............................] - ETA: 14s - loss: 0.4740 - accuracy: 0.9068
 2000/25000 [=>............................] - ETA: 14s - loss: 0.4739 - accuracy: 0.9075
 2100/25000 [=>............................] - ETA: 14s - loss: 0.4746 - accuracy: 0.9076
 2200/25000 [=>............................] - ETA: 14s - loss: 0.4752 - accuracy: 0.9073
 2300/25000 [=>............................] - ETA: 14s - loss: 0.4745 - accuracy: 0.9100
 2400/25000 [=>............................] - ETA: 14s - loss: 0.4752 - accuracy: 0.9087
 2500/25000 [==>...........................] - ETA: 14s - loss: 0.4750 - accuracy: 0.9076
 2600/25000 [==>...........................] - ETA: 13s - loss: 0.4741 - accuracy: 0.9096
 2700/25000 [==>...........................] - ETA: 13s - loss: 0.4747 - accuracy: 0.9093
 2800/25000 [==>...........................] - ETA: 13s - loss: 0.4732 - accuracy: 0.9107
 2900/25000 [==>...........................] - ETA: 13s - loss: 0.4731 - accuracy: 0.9110
 3000/25000 [==>...........................] - ETA: 13s - loss: 0.4725 - accuracy: 0.9120
 3100/25000 [==>...........................] - ETA: 13s - loss: 0.4715 - accuracy: 0.9123
 3200/25000 [==>...........................] - ETA: 13s - loss: 0.4707 - accuracy: 0.9122
 3300/25000 [==>...........................] - ETA: 13s - loss: 0.4708 - accuracy: 0.9115
 3400/25000 [===>..........................] - ETA: 13s - loss: 0.4708 - accuracy: 0.9115
 3500/25000 [===>..........................] - ETA: 13s - loss: 0.4708 - accuracy: 0.9114
 3600/25000 [===>..........................] - ETA: 13s - loss: 0.4709 - accuracy: 0.9111
 3700/25000 [===>..........................] - ETA: 13s - loss: 0.4696 - accuracy: 0.9124
 3800/25000 [===>..........................] - ETA: 13s - loss: 0.4692 - accuracy: 0.9118
 3900/25000 [===>..........................] - ETA: 13s - loss: 0.4686 - accuracy: 0.9128
 4000/25000 [===>..........................] - ETA: 13s - loss: 0.4685 - accuracy: 0.9130
 4100/25000 [===>..........................] - ETA: 13s - loss: 0.4685 - accuracy: 0.9127
 4200/25000 [====>.........................] - ETA: 12s - loss: 0.4686 - accuracy: 0.9129
 4300/25000 [====>.........................] - ETA: 12s - loss: 0.4686 - accuracy: 0.9126
 4400/25000 [====>.........................] - ETA: 12s - loss: 0.4685 - accuracy: 0.9127
 4500/25000 [====>.........................] - ETA: 12s - loss: 0.4681 - accuracy: 0.9124
 4600/25000 [====>.........................] - ETA: 12s - loss: 0.4681 - accuracy: 0.9124
 4700/25000 [====>.........................] - ETA: 12s - loss: 0.4677 - accuracy: 0.9126
 4800/25000 [====>.........................] - ETA: 12s - loss: 0.4672 - accuracy: 0.9129
 4900/25000 [====>.........................] - ETA: 12s - loss: 0.4675 - accuracy: 0.9122
 5000/25000 [=====>........................] - ETA: 12s - loss: 0.4675 - accuracy: 0.9122
 5100/25000 [=====>........................] - ETA: 12s - loss: 0.4668 - accuracy: 0.9129
 5200/25000 [=====>........................] - ETA: 12s - loss: 0.4661 - accuracy: 0.9144
 5300/25000 [=====>........................] - ETA: 12s - loss: 0.4654 - accuracy: 0.9151
 5400/25000 [=====>........................] - ETA: 12s - loss: 0.4656 - accuracy: 0.9152
 5500/25000 [=====>........................] - ETA: 12s - loss: 0.4651 - accuracy: 0.9156
 5600/25000 [=====>........................] - ETA: 12s - loss: 0.4650 - accuracy: 0.9157
 5700/25000 [=====>........................] - ETA: 12s - loss: 0.4646 - accuracy: 0.9158
 5800/25000 [=====>........................] - ETA: 12s - loss: 0.4642 - accuracy: 0.9162
 5900/25000 [======>.......................] - ETA: 11s - loss: 0.4646 - accuracy: 0.9159
 6000/25000 [======>.......................] - ETA: 11s - loss: 0.4646 - accuracy: 0.9162
 6100/25000 [======>.......................] - ETA: 11s - loss: 0.4645 - accuracy: 0.9162
 6200/25000 [======>.......................] - ETA: 11s - loss: 0.4647 - accuracy: 0.9156
 6300/25000 [======>.......................] - ETA: 11s - loss: 0.4648 - accuracy: 0.9152
 6400/25000 [======>.......................] - ETA: 11s - loss: 0.4648 - accuracy: 0.9150
 6500/25000 [======>.......................] - ETA: 11s - loss: 0.4641 - accuracy: 0.9160
 6600/25000 [======>.......................] - ETA: 11s - loss: 0.4639 - accuracy: 0.9164
 6700/25000 [=======>......................] - ETA: 11s - loss: 0.4634 - accuracy: 0.9166
 6800/25000 [=======>......................] - ETA: 11s - loss: 0.4639 - accuracy: 0.9159
 6900/25000 [=======>......................] - ETA: 11s - loss: 0.4640 - accuracy: 0.9155
 7000/25000 [=======>......................] - ETA: 11s - loss: 0.4639 - accuracy: 0.9156
 7100/25000 [=======>......................] - ETA: 11s - loss: 0.4643 - accuracy: 0.9146
 7200/25000 [=======>......................] - ETA: 11s - loss: 0.4644 - accuracy: 0.9142
 7300/25000 [=======>......................] - ETA: 11s - loss: 0.4650 - accuracy: 0.9137
 7400/25000 [=======>......................] - ETA: 11s - loss: 0.4651 - accuracy: 0.9134
 7500/25000 [========>.....................] - ETA: 10s - loss: 0.4652 - accuracy: 0.9129
 7600/25000 [========>.....................] - ETA: 10s - loss: 0.4650 - accuracy: 0.9134
 7700/25000 [========>.....................] - ETA: 10s - loss: 0.4650 - accuracy: 0.9130
 7800/25000 [========>.....................] - ETA: 10s - loss: 0.4649 - accuracy: 0.9133
 7900/25000 [========>.....................] - ETA: 10s - loss: 0.4645 - accuracy: 0.9138
 8000/25000 [========>.....................] - ETA: 10s - loss: 0.4646 - accuracy: 0.9135
 8100/25000 [========>.....................] - ETA: 10s - loss: 0.4647 - accuracy: 0.9131
 8200/25000 [========>.....................] - ETA: 10s - loss: 0.4644 - accuracy: 0.9133
 8300/25000 [========>.....................] - ETA: 10s - loss: 0.4645 - accuracy: 0.9130
 8400/25000 [=========>....................] - ETA: 10s - loss: 0.4642 - accuracy: 0.9133
 8500/25000 [=========>....................] - ETA: 10s - loss: 0.4638 - accuracy: 0.9138
 8600/25000 [=========>....................] - ETA: 10s - loss: 0.4635 - accuracy: 0.9140
 8700/25000 [=========>....................] - ETA: 10s - loss: 0.4638 - accuracy: 0.9134
 8800/25000 [=========>....................] - ETA: 10s - loss: 0.4637 - accuracy: 0.9133
 8900/25000 [=========>....................] - ETA: 10s - loss: 0.4635 - accuracy: 0.9135
 9000/25000 [=========>....................] - ETA: 10s - loss: 0.4636 - accuracy: 0.9132
 9100/25000 [=========>....................] - ETA: 9s - loss: 0.4636 - accuracy: 0.9132 
 9200/25000 [==========>...................] - ETA: 9s - loss: 0.4639 - accuracy: 0.9128
 9300/25000 [==========>...................] - ETA: 9s - loss: 0.4637 - accuracy: 0.9125
 9400/25000 [==========>...................] - ETA: 9s - loss: 0.4640 - accuracy: 0.9121
 9500/25000 [==========>...................] - ETA: 9s - loss: 0.4641 - accuracy: 0.9119
 9600/25000 [==========>...................] - ETA: 9s - loss: 0.4643 - accuracy: 0.9116
 9700/25000 [==========>...................] - ETA: 9s - loss: 0.4643 - accuracy: 0.9116
 9800/25000 [==========>...................] - ETA: 9s - loss: 0.4643 - accuracy: 0.9114
 9900/25000 [==========>...................] - ETA: 9s - loss: 0.4642 - accuracy: 0.9113
10000/25000 [===========>..................] - ETA: 9s - loss: 0.4642 - accuracy: 0.9116
10100/25000 [===========>..................] - ETA: 9s - loss: 0.4641 - accuracy: 0.9117
10200/25000 [===========>..................] - ETA: 9s - loss: 0.4642 - accuracy: 0.9115
10300/25000 [===========>..................] - ETA: 9s - loss: 0.4639 - accuracy: 0.9117
10400/25000 [===========>..................] - ETA: 9s - loss: 0.4638 - accuracy: 0.9117
10500/25000 [===========>..................] - ETA: 9s - loss: 0.4639 - accuracy: 0.9114
10600/25000 [===========>..................] - ETA: 9s - loss: 0.4641 - accuracy: 0.9112
10700/25000 [===========>..................] - ETA: 8s - loss: 0.4642 - accuracy: 0.9108
10800/25000 [===========>..................] - ETA: 8s - loss: 0.4641 - accuracy: 0.9109
10900/25000 [============>.................] - ETA: 8s - loss: 0.4639 - accuracy: 0.9109
11000/25000 [============>.................] - ETA: 8s - loss: 0.4639 - accuracy: 0.9108
11100/25000 [============>.................] - ETA: 8s - loss: 0.4640 - accuracy: 0.9105
11200/25000 [============>.................] - ETA: 8s - loss: 0.4640 - accuracy: 0.9102
11300/25000 [============>.................] - ETA: 8s - loss: 0.4639 - accuracy: 0.9104
11400/25000 [============>.................] - ETA: 8s - loss: 0.4638 - accuracy: 0.9104
11500/25000 [============>.................] - ETA: 8s - loss: 0.4636 - accuracy: 0.9106
11600/25000 [============>.................] - ETA: 8s - loss: 0.4639 - accuracy: 0.9103
11700/25000 [=============>................] - ETA: 8s - loss: 0.4638 - accuracy: 0.9103
11800/25000 [=============>................] - ETA: 8s - loss: 0.4639 - accuracy: 0.9099
11900/25000 [=============>................] - ETA: 8s - loss: 0.4642 - accuracy: 0.9096
12000/25000 [=============>................] - ETA: 8s - loss: 0.4644 - accuracy: 0.9095
12100/25000 [=============>................] - ETA: 8s - loss: 0.4644 - accuracy: 0.9095
12200/25000 [=============>................] - ETA: 8s - loss: 0.4648 - accuracy: 0.9091
12300/25000 [=============>................] - ETA: 8s - loss: 0.4648 - accuracy: 0.9089
12400/25000 [=============>................] - ETA: 7s - loss: 0.4649 - accuracy: 0.9090
12500/25000 [==============>...............] - ETA: 7s - loss: 0.4647 - accuracy: 0.9091
12600/25000 [==============>...............] - ETA: 7s - loss: 0.4649 - accuracy: 0.9089
12700/25000 [==============>...............] - ETA: 7s - loss: 0.4650 - accuracy: 0.9088
12800/25000 [==============>...............] - ETA: 7s - loss: 0.4649 - accuracy: 0.9087
12900/25000 [==============>...............] - ETA: 7s - loss: 0.4648 - accuracy: 0.9088
13000/25000 [==============>...............] - ETA: 7s - loss: 0.4646 - accuracy: 0.9092
13100/25000 [==============>...............] - ETA: 7s - loss: 0.4645 - accuracy: 0.9093
13200/25000 [==============>...............] - ETA: 7s - loss: 0.4645 - accuracy: 0.9092
13300/25000 [==============>...............] - ETA: 7s - loss: 0.4643 - accuracy: 0.9095
13400/25000 [===============>..............] - ETA: 7s - loss: 0.4641 - accuracy: 0.9099
13500/25000 [===============>..............] - ETA: 7s - loss: 0.4638 - accuracy: 0.9101
13600/25000 [===============>..............] - ETA: 7s - loss: 0.4637 - accuracy: 0.9100
13700/25000 [===============>..............] - ETA: 7s - loss: 0.4637 - accuracy: 0.9101
13800/25000 [===============>..............] - ETA: 7s - loss: 0.4635 - accuracy: 0.9102
13900/25000 [===============>..............] - ETA: 7s - loss: 0.4633 - accuracy: 0.9104
14000/25000 [===============>..............] - ETA: 6s - loss: 0.4633 - accuracy: 0.9104
14100/25000 [===============>..............] - ETA: 6s - loss: 0.4632 - accuracy: 0.9103
14200/25000 [================>.............] - ETA: 6s - loss: 0.4631 - accuracy: 0.9104
14300/25000 [================>.............] - ETA: 6s - loss: 0.4628 - accuracy: 0.9106
14400/25000 [================>.............] - ETA: 6s - loss: 0.4629 - accuracy: 0.9103
14500/25000 [================>.............] - ETA: 6s - loss: 0.4630 - accuracy: 0.9102
14600/25000 [================>.............] - ETA: 6s - loss: 0.4627 - accuracy: 0.9103
14700/25000 [================>.............] - ETA: 6s - loss: 0.4626 - accuracy: 0.9104
14800/25000 [================>.............] - ETA: 6s - loss: 0.4626 - accuracy: 0.9104
14900/25000 [================>.............] - ETA: 6s - loss: 0.4626 - accuracy: 0.9103
15000/25000 [=================>............] - ETA: 6s - loss: 0.4626 - accuracy: 0.9105
15100/25000 [=================>............] - ETA: 6s - loss: 0.4625 - accuracy: 0.9104
15200/25000 [=================>............] - ETA: 6s - loss: 0.4624 - accuracy: 0.9105
15300/25000 [=================>............] - ETA: 6s - loss: 0.4625 - accuracy: 0.9103
15400/25000 [=================>............] - ETA: 6s - loss: 0.4629 - accuracy: 0.9100
15500/25000 [=================>............] - ETA: 5s - loss: 0.4629 - accuracy: 0.9098
15600/25000 [=================>............] - ETA: 5s - loss: 0.4629 - accuracy: 0.9098
15700/25000 [=================>............] - ETA: 5s - loss: 0.4628 - accuracy: 0.9097
15800/25000 [=================>............] - ETA: 5s - loss: 0.4627 - accuracy: 0.9099
15900/25000 [==================>...........] - ETA: 5s - loss: 0.4626 - accuracy: 0.9097
16000/25000 [==================>...........] - ETA: 5s - loss: 0.4626 - accuracy: 0.9097
16100/25000 [==================>...........] - ETA: 5s - loss: 0.4624 - accuracy: 0.9099
16200/25000 [==================>...........] - ETA: 5s - loss: 0.4627 - accuracy: 0.9097
16300/25000 [==================>...........] - ETA: 5s - loss: 0.4627 - accuracy: 0.9096
16400/25000 [==================>...........] - ETA: 5s - loss: 0.4628 - accuracy: 0.9094
16500/25000 [==================>...........] - ETA: 5s - loss: 0.4629 - accuracy: 0.9092
16600/25000 [==================>...........] - ETA: 5s - loss: 0.4630 - accuracy: 0.9091
16700/25000 [===================>..........] - ETA: 5s - loss: 0.4631 - accuracy: 0.9088
16800/25000 [===================>..........] - ETA: 5s - loss: 0.4629 - accuracy: 0.9089
16900/25000 [===================>..........] - ETA: 5s - loss: 0.4630 - accuracy: 0.9088
17000/25000 [===================>..........] - ETA: 5s - loss: 0.4629 - accuracy: 0.9088
17100/25000 [===================>..........] - ETA: 4s - loss: 0.4630 - accuracy: 0.9084
17200/25000 [===================>..........] - ETA: 4s - loss: 0.4632 - accuracy: 0.9081
17300/25000 [===================>..........] - ETA: 4s - loss: 0.4633 - accuracy: 0.9080
17400/25000 [===================>..........] - ETA: 4s - loss: 0.4632 - accuracy: 0.9079
17500/25000 [====================>.........] - ETA: 4s - loss: 0.4633 - accuracy: 0.9079
17600/25000 [====================>.........] - ETA: 4s - loss: 0.4632 - accuracy: 0.9079
17700/25000 [====================>.........] - ETA: 4s - loss: 0.4632 - accuracy: 0.9077
17800/25000 [====================>.........] - ETA: 4s - loss: 0.4631 - accuracy: 0.9077
17900/25000 [====================>.........] - ETA: 4s - loss: 0.4630 - accuracy: 0.9077
18000/25000 [====================>.........] - ETA: 4s - loss: 0.4632 - accuracy: 0.9075
18100/25000 [====================>.........] - ETA: 4s - loss: 0.4632 - accuracy: 0.9073
18200/25000 [====================>.........] - ETA: 4s - loss: 0.4633 - accuracy: 0.9071
18300/25000 [====================>.........] - ETA: 4s - loss: 0.4634 - accuracy: 0.9070
18400/25000 [=====================>........] - ETA: 4s - loss: 0.4634 - accuracy: 0.9071
18500/25000 [=====================>........] - ETA: 4s - loss: 0.4634 - accuracy: 0.9070
18600/25000 [=====================>........] - ETA: 4s - loss: 0.4633 - accuracy: 0.9071
18700/25000 [=====================>........] - ETA: 3s - loss: 0.4633 - accuracy: 0.9071
18800/25000 [=====================>........] - ETA: 3s - loss: 0.4632 - accuracy: 0.9070
18900/25000 [=====================>........] - ETA: 3s - loss: 0.4631 - accuracy: 0.9070
19000/25000 [=====================>........] - ETA: 3s - loss: 0.4630 - accuracy: 0.9073
19100/25000 [=====================>........] - ETA: 3s - loss: 0.4630 - accuracy: 0.9072
19200/25000 [======================>.......] - ETA: 3s - loss: 0.4629 - accuracy: 0.9071
19300/25000 [======================>.......] - ETA: 3s - loss: 0.4628 - accuracy: 0.9072
19400/25000 [======================>.......] - ETA: 3s - loss: 0.4625 - accuracy: 0.9074
19500/25000 [======================>.......] - ETA: 3s - loss: 0.4625 - accuracy: 0.9072
19600/25000 [======================>.......] - ETA: 3s - loss: 0.4625 - accuracy: 0.9072
19700/25000 [======================>.......] - ETA: 3s - loss: 0.4626 - accuracy: 0.9070
19800/25000 [======================>.......] - ETA: 3s - loss: 0.4625 - accuracy: 0.9070
19900/25000 [======================>.......] - ETA: 3s - loss: 0.4625 - accuracy: 0.9070
20000/25000 [=======================>......] - ETA: 3s - loss: 0.4627 - accuracy: 0.9067
20100/25000 [=======================>......] - ETA: 3s - loss: 0.4628 - accuracy: 0.9064
20200/25000 [=======================>......] - ETA: 3s - loss: 0.4627 - accuracy: 0.9063
20300/25000 [=======================>......] - ETA: 2s - loss: 0.4628 - accuracy: 0.9060
20400/25000 [=======================>......] - ETA: 2s - loss: 0.4628 - accuracy: 0.9058
20500/25000 [=======================>......] - ETA: 2s - loss: 0.4627 - accuracy: 0.9060
20600/25000 [=======================>......] - ETA: 2s - loss: 0.4624 - accuracy: 0.9061
20700/25000 [=======================>......] - ETA: 2s - loss: 0.4625 - accuracy: 0.9057
20800/25000 [=======================>......] - ETA: 2s - loss: 0.4626 - accuracy: 0.9056
20900/25000 [========================>.....] - ETA: 2s - loss: 0.4624 - accuracy: 0.9057
21000/25000 [========================>.....] - ETA: 2s - loss: 0.4623 - accuracy: 0.9058
21100/25000 [========================>.....] - ETA: 2s - loss: 0.4622 - accuracy: 0.9059
21200/25000 [========================>.....] - ETA: 2s - loss: 0.4623 - accuracy: 0.9058
21300/25000 [========================>.....] - ETA: 2s - loss: 0.4620 - accuracy: 0.9060
21400/25000 [========================>.....] - ETA: 2s - loss: 0.4619 - accuracy: 0.9059
21500/25000 [========================>.....] - ETA: 2s - loss: 0.4619 - accuracy: 0.9058
21600/25000 [========================>.....] - ETA: 2s - loss: 0.4620 - accuracy: 0.9056
21700/25000 [=========================>....] - ETA: 2s - loss: 0.4619 - accuracy: 0.9056
21800/25000 [=========================>....] - ETA: 2s - loss: 0.4619 - accuracy: 0.9056
21900/25000 [=========================>....] - ETA: 1s - loss: 0.4617 - accuracy: 0.9057
22000/25000 [=========================>....] - ETA: 1s - loss: 0.4616 - accuracy: 0.9057
22100/25000 [=========================>....] - ETA: 1s - loss: 0.4614 - accuracy: 0.9058
22200/25000 [=========================>....] - ETA: 1s - loss: 0.4613 - accuracy: 0.9058
22300/25000 [=========================>....] - ETA: 1s - loss: 0.4614 - accuracy: 0.9057
22400/25000 [=========================>....] - ETA: 1s - loss: 0.4613 - accuracy: 0.9058
22500/25000 [==========================>...] - ETA: 1s - loss: 0.4614 - accuracy: 0.9057
22600/25000 [==========================>...] - ETA: 1s - loss: 0.4612 - accuracy: 0.9059
22700/25000 [==========================>...] - ETA: 1s - loss: 0.4611 - accuracy: 0.9057
22800/25000 [==========================>...] - ETA: 1s - loss: 0.4610 - accuracy: 0.9059
22900/25000 [==========================>...] - ETA: 1s - loss: 0.4608 - accuracy: 0.9059
23000/25000 [==========================>...] - ETA: 1s - loss: 0.4607 - accuracy: 0.9060
23100/25000 [==========================>...] - ETA: 1s - loss: 0.4606 - accuracy: 0.9061
23200/25000 [==========================>...] - ETA: 1s - loss: 0.4605 - accuracy: 0.9063
23300/25000 [==========================>...] - ETA: 1s - loss: 0.4605 - accuracy: 0.9061
23400/25000 [===========================>..] - ETA: 1s - loss: 0.4606 - accuracy: 0.9059
23500/25000 [===========================>..] - ETA: 0s - loss: 0.4604 - accuracy: 0.9060
23600/25000 [===========================>..] - ETA: 0s - loss: 0.4604 - accuracy: 0.9058
23700/25000 [===========================>..] - ETA: 0s - loss: 0.4604 - accuracy: 0.9056
23800/25000 [===========================>..] - ETA: 0s - loss: 0.4604 - accuracy: 0.9055
23900/25000 [===========================>..] - ETA: 0s - loss: 0.4603 - accuracy: 0.9056
24000/25000 [===========================>..] - ETA: 0s - loss: 0.4602 - accuracy: 0.9057
24100/25000 [===========================>..] - ETA: 0s - loss: 0.4602 - accuracy: 0.9055
24200/25000 [============================>.] - ETA: 0s - loss: 0.4600 - accuracy: 0.9056
24300/25000 [============================>.] - ETA: 0s - loss: 0.4600 - accuracy: 0.9056
24400/25000 [============================>.] - ETA: 0s - loss: 0.4599 - accuracy: 0.9056
24500/25000 [============================>.] - ETA: 0s - loss: 0.4597 - accuracy: 0.9058
24600/25000 [============================>.] - ETA: 0s - loss: 0.4597 - accuracy: 0.9058
24700/25000 [============================>.] - ETA: 0s - loss: 0.4595 - accuracy: 0.9060
24800/25000 [============================>.] - ETA: 0s - loss: 0.4595 - accuracy: 0.9058
24900/25000 [============================>.] - ETA: 0s - loss: 0.4595 - accuracy: 0.9057
25000/25000 [==============================] - 20s 799us/step - loss: 0.4595 - accuracy: 0.9058 - val_loss: 0.4791 - val_accuracy: 0.8596
Epoch 4/10

  100/25000 [..............................] - ETA: 16s - loss: 0.4281 - accuracy: 0.9500
  200/25000 [..............................] - ETA: 16s - loss: 0.4305 - accuracy: 0.9350
  300/25000 [..............................] - ETA: 15s - loss: 0.4235 - accuracy: 0.9433
  400/25000 [..............................] - ETA: 15s - loss: 0.4182 - accuracy: 0.9450
  500/25000 [..............................] - ETA: 15s - loss: 0.4149 - accuracy: 0.9420
  600/25000 [..............................] - ETA: 15s - loss: 0.4203 - accuracy: 0.9317
  700/25000 [..............................] - ETA: 15s - loss: 0.4199 - accuracy: 0.9300
  800/25000 [..............................] - ETA: 15s - loss: 0.4260 - accuracy: 0.9212
  900/25000 [>.............................] - ETA: 15s - loss: 0.4261 - accuracy: 0.9211
 1000/25000 [>.............................] - ETA: 14s - loss: 0.4278 - accuracy: 0.9190
 1100/25000 [>.............................] - ETA: 14s - loss: 0.4267 - accuracy: 0.9227
 1200/25000 [>.............................] - ETA: 14s - loss: 0.4239 - accuracy: 0.9258
 1300/25000 [>.............................] - ETA: 14s - loss: 0.4259 - accuracy: 0.9238
 1400/25000 [>.............................] - ETA: 14s - loss: 0.4246 - accuracy: 0.9257
 1500/25000 [>.............................] - ETA: 14s - loss: 0.4231 - accuracy: 0.9273
 1600/25000 [>.............................] - ETA: 14s - loss: 0.4250 - accuracy: 0.9269
 1700/25000 [=>............................] - ETA: 14s - loss: 0.4256 - accuracy: 0.9259
 1800/25000 [=>............................] - ETA: 14s - loss: 0.4277 - accuracy: 0.9228
 1900/25000 [=>............................] - ETA: 14s - loss: 0.4268 - accuracy: 0.9242
 2000/25000 [=>............................] - ETA: 14s - loss: 0.4261 - accuracy: 0.9245
 2100/25000 [=>............................] - ETA: 14s - loss: 0.4247 - accuracy: 0.9262
 2200/25000 [=>............................] - ETA: 14s - loss: 0.4245 - accuracy: 0.9259
 2300/25000 [=>............................] - ETA: 14s - loss: 0.4246 - accuracy: 0.9261
 2400/25000 [=>............................] - ETA: 14s - loss: 0.4240 - accuracy: 0.9267
 2500/25000 [==>...........................] - ETA: 13s - loss: 0.4227 - accuracy: 0.9284
 2600/25000 [==>...........................] - ETA: 13s - loss: 0.4221 - accuracy: 0.9292
 2700/25000 [==>...........................] - ETA: 13s - loss: 0.4228 - accuracy: 0.9278
 2800/25000 [==>...........................] - ETA: 13s - loss: 0.4238 - accuracy: 0.9268
 2900/25000 [==>...........................] - ETA: 13s - loss: 0.4225 - accuracy: 0.9279
 3000/25000 [==>...........................] - ETA: 13s - loss: 0.4242 - accuracy: 0.9257
 3100/25000 [==>...........................] - ETA: 13s - loss: 0.4246 - accuracy: 0.9255
 3200/25000 [==>...........................] - ETA: 13s - loss: 0.4245 - accuracy: 0.9256
 3300/25000 [==>...........................] - ETA: 13s - loss: 0.4267 - accuracy: 0.9233
 3400/25000 [===>..........................] - ETA: 13s - loss: 0.4262 - accuracy: 0.9238
 3500/25000 [===>..........................] - ETA: 13s - loss: 0.4265 - accuracy: 0.9234
 3600/25000 [===>..........................] - ETA: 13s - loss: 0.4260 - accuracy: 0.9239
 3700/25000 [===>..........................] - ETA: 13s - loss: 0.4268 - accuracy: 0.9224
 3800/25000 [===>..........................] - ETA: 13s - loss: 0.4260 - accuracy: 0.9234
 3900/25000 [===>..........................] - ETA: 13s - loss: 0.4262 - accuracy: 0.9231
 4000/25000 [===>..........................] - ETA: 13s - loss: 0.4258 - accuracy: 0.9233
 4100/25000 [===>..........................] - ETA: 13s - loss: 0.4261 - accuracy: 0.9232
 4200/25000 [====>.........................] - ETA: 12s - loss: 0.4267 - accuracy: 0.9231
 4300/25000 [====>.........................] - ETA: 12s - loss: 0.4262 - accuracy: 0.9235
 4400/25000 [====>.........................] - ETA: 12s - loss: 0.4270 - accuracy: 0.9225
 4500/25000 [====>.........................] - ETA: 12s - loss: 0.4267 - accuracy: 0.9229
 4600/25000 [====>.........................] - ETA: 12s - loss: 0.4264 - accuracy: 0.9233
 4700/25000 [====>.........................] - ETA: 12s - loss: 0.4268 - accuracy: 0.9228
 4800/25000 [====>.........................] - ETA: 12s - loss: 0.4270 - accuracy: 0.9221
 4900/25000 [====>.........................] - ETA: 12s - loss: 0.4262 - accuracy: 0.9233
 5000/25000 [=====>........................] - ETA: 12s - loss: 0.4265 - accuracy: 0.9222
 5100/25000 [=====>........................] - ETA: 12s - loss: 0.4259 - accuracy: 0.9229
 5200/25000 [=====>........................] - ETA: 12s - loss: 0.4253 - accuracy: 0.9231
 5300/25000 [=====>........................] - ETA: 12s - loss: 0.4255 - accuracy: 0.9230
 5400/25000 [=====>........................] - ETA: 12s - loss: 0.4256 - accuracy: 0.9228
 5500/25000 [=====>........................] - ETA: 12s - loss: 0.4264 - accuracy: 0.9218
 5600/25000 [=====>........................] - ETA: 12s - loss: 0.4261 - accuracy: 0.9220
 5700/25000 [=====>........................] - ETA: 12s - loss: 0.4258 - accuracy: 0.9219
 5800/25000 [=====>........................] - ETA: 12s - loss: 0.4255 - accuracy: 0.9221
 5900/25000 [======>.......................] - ETA: 11s - loss: 0.4251 - accuracy: 0.9225
 6000/25000 [======>.......................] - ETA: 11s - loss: 0.4254 - accuracy: 0.9223
 6100/25000 [======>.......................] - ETA: 11s - loss: 0.4253 - accuracy: 0.9226
 6200/25000 [======>.......................] - ETA: 11s - loss: 0.4254 - accuracy: 0.9223
 6300/25000 [======>.......................] - ETA: 11s - loss: 0.4252 - accuracy: 0.9224
 6400/25000 [======>.......................] - ETA: 11s - loss: 0.4251 - accuracy: 0.9223
 6500/25000 [======>.......................] - ETA: 11s - loss: 0.4251 - accuracy: 0.9225
 6600/25000 [======>.......................] - ETA: 11s - loss: 0.4251 - accuracy: 0.9226
 6700/25000 [=======>......................] - ETA: 11s - loss: 0.4244 - accuracy: 0.9234
 6800/25000 [=======>......................] - ETA: 11s - loss: 0.4243 - accuracy: 0.9234
 6900/25000 [=======>......................] - ETA: 11s - loss: 0.4242 - accuracy: 0.9233
 7000/25000 [=======>......................] - ETA: 11s - loss: 0.4235 - accuracy: 0.9240
 7100/25000 [=======>......................] - ETA: 11s - loss: 0.4233 - accuracy: 0.9242
 7200/25000 [=======>......................] - ETA: 11s - loss: 0.4228 - accuracy: 0.9250
 7300/25000 [=======>......................] - ETA: 11s - loss: 0.4225 - accuracy: 0.9253
 7400/25000 [=======>......................] - ETA: 11s - loss: 0.4221 - accuracy: 0.9255
 7500/25000 [========>.....................] - ETA: 10s - loss: 0.4221 - accuracy: 0.9255
 7600/25000 [========>.....................] - ETA: 10s - loss: 0.4218 - accuracy: 0.9257
 7700/25000 [========>.....................] - ETA: 10s - loss: 0.4216 - accuracy: 0.9258
 7800/25000 [========>.....................] - ETA: 10s - loss: 0.4212 - accuracy: 0.9263
 7900/25000 [========>.....................] - ETA: 10s - loss: 0.4217 - accuracy: 0.9257
 8000/25000 [========>.....................] - ETA: 10s - loss: 0.4216 - accuracy: 0.9256
 8100/25000 [========>.....................] - ETA: 10s - loss: 0.4220 - accuracy: 0.9251
 8200/25000 [========>.....................] - ETA: 10s - loss: 0.4220 - accuracy: 0.9250
 8300/25000 [========>.....................] - ETA: 10s - loss: 0.4216 - accuracy: 0.9252
 8400/25000 [=========>....................] - ETA: 10s - loss: 0.4214 - accuracy: 0.9254
 8500/25000 [=========>....................] - ETA: 10s - loss: 0.4216 - accuracy: 0.9255
 8600/25000 [=========>....................] - ETA: 10s - loss: 0.4215 - accuracy: 0.9252
 8700/25000 [=========>....................] - ETA: 10s - loss: 0.4212 - accuracy: 0.9257
 8800/25000 [=========>....................] - ETA: 10s - loss: 0.4210 - accuracy: 0.9260
 8900/25000 [=========>....................] - ETA: 10s - loss: 0.4213 - accuracy: 0.9255
 9000/25000 [=========>....................] - ETA: 10s - loss: 0.4215 - accuracy: 0.9252
 9100/25000 [=========>....................] - ETA: 9s - loss: 0.4218 - accuracy: 0.9247 
 9200/25000 [==========>...................] - ETA: 9s - loss: 0.4222 - accuracy: 0.9243
 9300/25000 [==========>...................] - ETA: 9s - loss: 0.4222 - accuracy: 0.9242
 9400/25000 [==========>...................] - ETA: 9s - loss: 0.4229 - accuracy: 0.9236
 9500/25000 [==========>...................] - ETA: 9s - loss: 0.4228 - accuracy: 0.9237
 9600/25000 [==========>...................] - ETA: 9s - loss: 0.4226 - accuracy: 0.9237
 9700/25000 [==========>...................] - ETA: 9s - loss: 0.4224 - accuracy: 0.9239
 9800/25000 [==========>...................] - ETA: 9s - loss: 0.4225 - accuracy: 0.9236
 9900/25000 [==========>...................] - ETA: 9s - loss: 0.4225 - accuracy: 0.9235
10000/25000 [===========>..................] - ETA: 9s - loss: 0.4224 - accuracy: 0.9236
10100/25000 [===========>..................] - ETA: 9s - loss: 0.4224 - accuracy: 0.9236
10200/25000 [===========>..................] - ETA: 9s - loss: 0.4220 - accuracy: 0.9239
10300/25000 [===========>..................] - ETA: 9s - loss: 0.4216 - accuracy: 0.9240
10400/25000 [===========>..................] - ETA: 9s - loss: 0.4217 - accuracy: 0.9237
10500/25000 [===========>..................] - ETA: 9s - loss: 0.4216 - accuracy: 0.9236
10600/25000 [===========>..................] - ETA: 9s - loss: 0.4212 - accuracy: 0.9241
10700/25000 [===========>..................] - ETA: 9s - loss: 0.4210 - accuracy: 0.9241
10800/25000 [===========>..................] - ETA: 8s - loss: 0.4210 - accuracy: 0.9240
10900/25000 [============>.................] - ETA: 8s - loss: 0.4207 - accuracy: 0.9244
11000/25000 [============>.................] - ETA: 8s - loss: 0.4208 - accuracy: 0.9244
11100/25000 [============>.................] - ETA: 8s - loss: 0.4207 - accuracy: 0.9243
11200/25000 [============>.................] - ETA: 8s - loss: 0.4202 - accuracy: 0.9246
11300/25000 [============>.................] - ETA: 8s - loss: 0.4202 - accuracy: 0.9250
11400/25000 [============>.................] - ETA: 8s - loss: 0.4202 - accuracy: 0.9248
11500/25000 [============>.................] - ETA: 8s - loss: 0.4199 - accuracy: 0.9249
11600/25000 [============>.................] - ETA: 8s - loss: 0.4199 - accuracy: 0.9249
11700/25000 [=============>................] - ETA: 8s - loss: 0.4199 - accuracy: 0.9248
11800/25000 [=============>................] - ETA: 8s - loss: 0.4197 - accuracy: 0.9249
11900/25000 [=============>................] - ETA: 8s - loss: 0.4196 - accuracy: 0.9248
12000/25000 [=============>................] - ETA: 8s - loss: 0.4196 - accuracy: 0.9247
12100/25000 [=============>................] - ETA: 8s - loss: 0.4198 - accuracy: 0.9245
12200/25000 [=============>................] - ETA: 8s - loss: 0.4198 - accuracy: 0.9243
12300/25000 [=============>................] - ETA: 8s - loss: 0.4195 - accuracy: 0.9247
12400/25000 [=============>................] - ETA: 7s - loss: 0.4197 - accuracy: 0.9245
12500/25000 [==============>...............] - ETA: 7s - loss: 0.4194 - accuracy: 0.9247
12600/25000 [==============>...............] - ETA: 7s - loss: 0.4193 - accuracy: 0.9248
12700/25000 [==============>...............] - ETA: 7s - loss: 0.4194 - accuracy: 0.9247
12800/25000 [==============>...............] - ETA: 7s - loss: 0.4195 - accuracy: 0.9247
12900/25000 [==============>...............] - ETA: 7s - loss: 0.4194 - accuracy: 0.9247
13000/25000 [==============>...............] - ETA: 7s - loss: 0.4196 - accuracy: 0.9242
13100/25000 [==============>...............] - ETA: 7s - loss: 0.4194 - accuracy: 0.9244
13200/25000 [==============>...............] - ETA: 7s - loss: 0.4191 - accuracy: 0.9246
13300/25000 [==============>...............] - ETA: 7s - loss: 0.4190 - accuracy: 0.9247
13400/25000 [===============>..............] - ETA: 7s - loss: 0.4192 - accuracy: 0.9244
13500/25000 [===============>..............] - ETA: 7s - loss: 0.4196 - accuracy: 0.9240
13600/25000 [===============>..............] - ETA: 7s - loss: 0.4196 - accuracy: 0.9237
13700/25000 [===============>..............] - ETA: 7s - loss: 0.4197 - accuracy: 0.9236
13800/25000 [===============>..............] - ETA: 7s - loss: 0.4195 - accuracy: 0.9237
13900/25000 [===============>..............] - ETA: 7s - loss: 0.4198 - accuracy: 0.9233
14000/25000 [===============>..............] - ETA: 6s - loss: 0.4198 - accuracy: 0.9234
14100/25000 [===============>..............] - ETA: 6s - loss: 0.4197 - accuracy: 0.9233
14200/25000 [================>.............] - ETA: 6s - loss: 0.4199 - accuracy: 0.9230
14300/25000 [================>.............] - ETA: 6s - loss: 0.4200 - accuracy: 0.9228
14400/25000 [================>.............] - ETA: 6s - loss: 0.4200 - accuracy: 0.9226
14500/25000 [================>.............] - ETA: 6s - loss: 0.4199 - accuracy: 0.9226
14600/25000 [================>.............] - ETA: 6s - loss: 0.4198 - accuracy: 0.9227
14700/25000 [================>.............] - ETA: 6s - loss: 0.4198 - accuracy: 0.9227
14800/25000 [================>.............] - ETA: 6s - loss: 0.4199 - accuracy: 0.9226
14900/25000 [================>.............] - ETA: 6s - loss: 0.4199 - accuracy: 0.9225
15000/25000 [=================>............] - ETA: 6s - loss: 0.4196 - accuracy: 0.9228
15100/25000 [=================>............] - ETA: 6s - loss: 0.4196 - accuracy: 0.9229
15200/25000 [=================>............] - ETA: 6s - loss: 0.4194 - accuracy: 0.9230
15300/25000 [=================>............] - ETA: 6s - loss: 0.4194 - accuracy: 0.9229
15400/25000 [=================>............] - ETA: 6s - loss: 0.4193 - accuracy: 0.9230
15500/25000 [=================>............] - ETA: 6s - loss: 0.4193 - accuracy: 0.9230
15600/25000 [=================>............] - ETA: 5s - loss: 0.4192 - accuracy: 0.9230
15700/25000 [=================>............] - ETA: 5s - loss: 0.4194 - accuracy: 0.9227
15800/25000 [=================>............] - ETA: 5s - loss: 0.4194 - accuracy: 0.9227
15900/25000 [==================>...........] - ETA: 5s - loss: 0.4195 - accuracy: 0.9225
16000/25000 [==================>...........] - ETA: 5s - loss: 0.4196 - accuracy: 0.9223
16100/25000 [==================>...........] - ETA: 5s - loss: 0.4198 - accuracy: 0.9220
16200/25000 [==================>...........] - ETA: 5s - loss: 0.4197 - accuracy: 0.9222
16300/25000 [==================>...........] - ETA: 5s - loss: 0.4194 - accuracy: 0.9223
16400/25000 [==================>...........] - ETA: 5s - loss: 0.4194 - accuracy: 0.9222
16500/25000 [==================>...........] - ETA: 5s - loss: 0.4196 - accuracy: 0.9219
16600/25000 [==================>...........] - ETA: 5s - loss: 0.4196 - accuracy: 0.9219
16700/25000 [===================>..........] - ETA: 5s - loss: 0.4195 - accuracy: 0.9219
16800/25000 [===================>..........] - ETA: 5s - loss: 0.4195 - accuracy: 0.9218
16900/25000 [===================>..........] - ETA: 5s - loss: 0.4195 - accuracy: 0.9217
17000/25000 [===================>..........] - ETA: 5s - loss: 0.4195 - accuracy: 0.9216
17100/25000 [===================>..........] - ETA: 5s - loss: 0.4196 - accuracy: 0.9214
17200/25000 [===================>..........] - ETA: 4s - loss: 0.4195 - accuracy: 0.9213
17300/25000 [===================>..........] - ETA: 4s - loss: 0.4195 - accuracy: 0.9213
17400/25000 [===================>..........] - ETA: 4s - loss: 0.4194 - accuracy: 0.9214
17500/25000 [====================>.........] - ETA: 4s - loss: 0.4193 - accuracy: 0.9214
17600/25000 [====================>.........] - ETA: 4s - loss: 0.4193 - accuracy: 0.9212
17700/25000 [====================>.........] - ETA: 4s - loss: 0.4192 - accuracy: 0.9212
17800/25000 [====================>.........] - ETA: 4s - loss: 0.4190 - accuracy: 0.9215
17900/25000 [====================>.........] - ETA: 4s - loss: 0.4188 - accuracy: 0.9216
18000/25000 [====================>.........] - ETA: 4s - loss: 0.4188 - accuracy: 0.9216
18100/25000 [====================>.........] - ETA: 4s - loss: 0.4189 - accuracy: 0.9213
18200/25000 [====================>.........] - ETA: 4s - loss: 0.4187 - accuracy: 0.9215
18300/25000 [====================>.........] - ETA: 4s - loss: 0.4189 - accuracy: 0.9213
18400/25000 [=====================>........] - ETA: 4s - loss: 0.4187 - accuracy: 0.9214
18500/25000 [=====================>........] - ETA: 4s - loss: 0.4185 - accuracy: 0.9215
18600/25000 [=====================>........] - ETA: 4s - loss: 0.4184 - accuracy: 0.9216
18700/25000 [=====================>........] - ETA: 4s - loss: 0.4184 - accuracy: 0.9216
18800/25000 [=====================>........] - ETA: 3s - loss: 0.4182 - accuracy: 0.9216
18900/25000 [=====================>........] - ETA: 3s - loss: 0.4182 - accuracy: 0.9216
19000/25000 [=====================>........] - ETA: 3s - loss: 0.4182 - accuracy: 0.9215
19100/25000 [=====================>........] - ETA: 3s - loss: 0.4182 - accuracy: 0.9215
19200/25000 [======================>.......] - ETA: 3s - loss: 0.4182 - accuracy: 0.9214
19300/25000 [======================>.......] - ETA: 3s - loss: 0.4183 - accuracy: 0.9212
19400/25000 [======================>.......] - ETA: 3s - loss: 0.4182 - accuracy: 0.9212
19500/25000 [======================>.......] - ETA: 3s - loss: 0.4182 - accuracy: 0.9211
19600/25000 [======================>.......] - ETA: 3s - loss: 0.4183 - accuracy: 0.9210
19700/25000 [======================>.......] - ETA: 3s - loss: 0.4182 - accuracy: 0.9210
19800/25000 [======================>.......] - ETA: 3s - loss: 0.4181 - accuracy: 0.9210
19900/25000 [======================>.......] - ETA: 3s - loss: 0.4180 - accuracy: 0.9210
20000/25000 [=======================>......] - ETA: 3s - loss: 0.4181 - accuracy: 0.9208
20100/25000 [=======================>......] - ETA: 3s - loss: 0.4180 - accuracy: 0.9209
20200/25000 [=======================>......] - ETA: 3s - loss: 0.4181 - accuracy: 0.9206
20300/25000 [=======================>......] - ETA: 3s - loss: 0.4181 - accuracy: 0.9204
20400/25000 [=======================>......] - ETA: 2s - loss: 0.4180 - accuracy: 0.9205
20500/25000 [=======================>......] - ETA: 2s - loss: 0.4182 - accuracy: 0.9203
20600/25000 [=======================>......] - ETA: 2s - loss: 0.4185 - accuracy: 0.9199
20700/25000 [=======================>......] - ETA: 2s - loss: 0.4182 - accuracy: 0.9201
20800/25000 [=======================>......] - ETA: 2s - loss: 0.4182 - accuracy: 0.9200
20900/25000 [========================>.....] - ETA: 2s - loss: 0.4181 - accuracy: 0.9201
21000/25000 [========================>.....] - ETA: 2s - loss: 0.4181 - accuracy: 0.9200
21100/25000 [========================>.....] - ETA: 2s - loss: 0.4180 - accuracy: 0.9200
21200/25000 [========================>.....] - ETA: 2s - loss: 0.4179 - accuracy: 0.9200
21300/25000 [========================>.....] - ETA: 2s - loss: 0.4178 - accuracy: 0.9201
21400/25000 [========================>.....] - ETA: 2s - loss: 0.4177 - accuracy: 0.9201
21500/25000 [========================>.....] - ETA: 2s - loss: 0.4176 - accuracy: 0.9201
21600/25000 [========================>.....] - ETA: 2s - loss: 0.4177 - accuracy: 0.9200
21700/25000 [=========================>....] - ETA: 2s - loss: 0.4175 - accuracy: 0.9200
21800/25000 [=========================>....] - ETA: 2s - loss: 0.4174 - accuracy: 0.9201
21900/25000 [=========================>....] - ETA: 1s - loss: 0.4174 - accuracy: 0.9201
22000/25000 [=========================>....] - ETA: 1s - loss: 0.4173 - accuracy: 0.9200
22100/25000 [=========================>....] - ETA: 1s - loss: 0.4174 - accuracy: 0.9198
22200/25000 [=========================>....] - ETA: 1s - loss: 0.4173 - accuracy: 0.9200
22300/25000 [=========================>....] - ETA: 1s - loss: 0.4173 - accuracy: 0.9198
22400/25000 [=========================>....] - ETA: 1s - loss: 0.4173 - accuracy: 0.9197
22500/25000 [==========================>...] - ETA: 1s - loss: 0.4171 - accuracy: 0.9198
22600/25000 [==========================>...] - ETA: 1s - loss: 0.4170 - accuracy: 0.9198
22700/25000 [==========================>...] - ETA: 1s - loss: 0.4171 - accuracy: 0.9198
22800/25000 [==========================>...] - ETA: 1s - loss: 0.4170 - accuracy: 0.9197
22900/25000 [==========================>...] - ETA: 1s - loss: 0.4172 - accuracy: 0.9195
23000/25000 [==========================>...] - ETA: 1s - loss: 0.4171 - accuracy: 0.9196
23100/25000 [==========================>...] - ETA: 1s - loss: 0.4172 - accuracy: 0.9193
23200/25000 [==========================>...] - ETA: 1s - loss: 0.4171 - accuracy: 0.9194
23300/25000 [==========================>...] - ETA: 1s - loss: 0.4169 - accuracy: 0.9195
23400/25000 [===========================>..] - ETA: 1s - loss: 0.4169 - accuracy: 0.9195
23500/25000 [===========================>..] - ETA: 0s - loss: 0.4168 - accuracy: 0.9196
23600/25000 [===========================>..] - ETA: 0s - loss: 0.4168 - accuracy: 0.9195
23700/25000 [===========================>..] - ETA: 0s - loss: 0.4168 - accuracy: 0.9196
23800/25000 [===========================>..] - ETA: 0s - loss: 0.4168 - accuracy: 0.9195
23900/25000 [===========================>..] - ETA: 0s - loss: 0.4168 - accuracy: 0.9195
24000/25000 [===========================>..] - ETA: 0s - loss: 0.4169 - accuracy: 0.9193
24100/25000 [===========================>..] - ETA: 0s - loss: 0.4167 - accuracy: 0.9195
24200/25000 [============================>.] - ETA: 0s - loss: 0.4167 - accuracy: 0.9195
24300/25000 [============================>.] - ETA: 0s - loss: 0.4165 - accuracy: 0.9195
24400/25000 [============================>.] - ETA: 0s - loss: 0.4164 - accuracy: 0.9195
24500/25000 [============================>.] - ETA: 0s - loss: 0.4164 - accuracy: 0.9194
24600/25000 [============================>.] - ETA: 0s - loss: 0.4164 - accuracy: 0.9193
24700/25000 [============================>.] - ETA: 0s - loss: 0.4162 - accuracy: 0.9194
24800/25000 [============================>.] - ETA: 0s - loss: 0.4160 - accuracy: 0.9196
24900/25000 [============================>.] - ETA: 0s - loss: 0.4159 - accuracy: 0.9197
25000/25000 [==============================] - 20s 818us/step - loss: 0.4158 - accuracy: 0.9198 - val_loss: 0.4572 - val_accuracy: 0.8595
Epoch 5/10

  100/25000 [..............................] - ETA: 20s - loss: 0.3764 - accuracy: 0.9500
  200/25000 [..............................] - ETA: 18s - loss: 0.4084 - accuracy: 0.9150
  300/25000 [..............................] - ETA: 18s - loss: 0.4029 - accuracy: 0.9233
  400/25000 [..............................] - ETA: 18s - loss: 0.4028 - accuracy: 0.9225
  500/25000 [..............................] - ETA: 17s - loss: 0.4016 - accuracy: 0.9220
  600/25000 [..............................] - ETA: 17s - loss: 0.3963 - accuracy: 0.9267
  700/25000 [..............................] - ETA: 16s - loss: 0.3914 - accuracy: 0.9300
  800/25000 [..............................] - ETA: 16s - loss: 0.3910 - accuracy: 0.9300
  900/25000 [>.............................] - ETA: 16s - loss: 0.3932 - accuracy: 0.9267
 1000/25000 [>.............................] - ETA: 17s - loss: 0.3945 - accuracy: 0.9270
 1100/25000 [>.............................] - ETA: 16s - loss: 0.3937 - accuracy: 0.9282
 1200/25000 [>.............................] - ETA: 16s - loss: 0.3906 - accuracy: 0.9308
 1300/25000 [>.............................] - ETA: 16s - loss: 0.3933 - accuracy: 0.9285
 1400/25000 [>.............................] - ETA: 16s - loss: 0.3941 - accuracy: 0.9271
 1500/25000 [>.............................] - ETA: 16s - loss: 0.3928 - accuracy: 0.9300
 1600/25000 [>.............................] - ETA: 16s - loss: 0.3931 - accuracy: 0.9294
 1700/25000 [=>............................] - ETA: 16s - loss: 0.3930 - accuracy: 0.9294
 1800/25000 [=>............................] - ETA: 16s - loss: 0.3922 - accuracy: 0.9306
 1900/25000 [=>............................] - ETA: 16s - loss: 0.3910 - accuracy: 0.9316
 2000/25000 [=>............................] - ETA: 16s - loss: 0.3897 - accuracy: 0.9335
 2100/25000 [=>............................] - ETA: 15s - loss: 0.3891 - accuracy: 0.9348
 2200/25000 [=>............................] - ETA: 15s - loss: 0.3885 - accuracy: 0.9350
 2300/25000 [=>............................] - ETA: 15s - loss: 0.3885 - accuracy: 0.9348
 2400/25000 [=>............................] - ETA: 15s - loss: 0.3880 - accuracy: 0.9350
 2500/25000 [==>...........................] - ETA: 15s - loss: 0.3883 - accuracy: 0.9348
 2600/25000 [==>...........................] - ETA: 15s - loss: 0.3891 - accuracy: 0.9346
 2700/25000 [==>...........................] - ETA: 15s - loss: 0.3894 - accuracy: 0.9341
 2800/25000 [==>...........................] - ETA: 15s - loss: 0.3895 - accuracy: 0.9339
 2900/25000 [==>...........................] - ETA: 15s - loss: 0.3907 - accuracy: 0.9324
 3000/25000 [==>...........................] - ETA: 15s - loss: 0.3925 - accuracy: 0.9303
 3100/25000 [==>...........................] - ETA: 15s - loss: 0.3931 - accuracy: 0.9297
 3200/25000 [==>...........................] - ETA: 15s - loss: 0.3918 - accuracy: 0.9306
 3300/25000 [==>...........................] - ETA: 14s - loss: 0.3911 - accuracy: 0.9312
 3400/25000 [===>..........................] - ETA: 14s - loss: 0.3909 - accuracy: 0.9318
 3500/25000 [===>..........................] - ETA: 14s - loss: 0.3911 - accuracy: 0.9317
 3600/25000 [===>..........................] - ETA: 14s - loss: 0.3911 - accuracy: 0.9317
 3700/25000 [===>..........................] - ETA: 14s - loss: 0.3904 - accuracy: 0.9327
 3800/25000 [===>..........................] - ETA: 14s - loss: 0.3898 - accuracy: 0.9334
 3900/25000 [===>..........................] - ETA: 14s - loss: 0.3891 - accuracy: 0.9338
 4000/25000 [===>..........................] - ETA: 14s - loss: 0.3889 - accuracy: 0.9337
 4100/25000 [===>..........................] - ETA: 14s - loss: 0.3884 - accuracy: 0.9339
 4200/25000 [====>.........................] - ETA: 14s - loss: 0.3882 - accuracy: 0.9343
 4300/25000 [====>.........................] - ETA: 14s - loss: 0.3887 - accuracy: 0.9333
 4400/25000 [====>.........................] - ETA: 13s - loss: 0.3884 - accuracy: 0.9336
 4500/25000 [====>.........................] - ETA: 13s - loss: 0.3886 - accuracy: 0.9336
 4600/25000 [====>.........................] - ETA: 13s - loss: 0.3886 - accuracy: 0.9335
 4700/25000 [====>.........................] - ETA: 13s - loss: 0.3883 - accuracy: 0.9336
 4800/25000 [====>.........................] - ETA: 13s - loss: 0.3884 - accuracy: 0.9333
 4900/25000 [====>.........................] - ETA: 13s - loss: 0.3883 - accuracy: 0.9335
 5000/25000 [=====>........................] - ETA: 13s - loss: 0.3891 - accuracy: 0.9326
 5100/25000 [=====>........................] - ETA: 13s - loss: 0.3885 - accuracy: 0.9329
 5200/25000 [=====>........................] - ETA: 13s - loss: 0.3879 - accuracy: 0.9337
 5300/25000 [=====>........................] - ETA: 13s - loss: 0.3883 - accuracy: 0.9328
 5400/25000 [=====>........................] - ETA: 13s - loss: 0.3885 - accuracy: 0.9324
 5500/25000 [=====>........................] - ETA: 13s - loss: 0.3886 - accuracy: 0.9322
 5600/25000 [=====>........................] - ETA: 13s - loss: 0.3885 - accuracy: 0.9321
 5700/25000 [=====>........................] - ETA: 12s - loss: 0.3885 - accuracy: 0.9321
 5800/25000 [=====>........................] - ETA: 12s - loss: 0.3879 - accuracy: 0.9324
 5900/25000 [======>.......................] - ETA: 12s - loss: 0.3868 - accuracy: 0.9336
 6000/25000 [======>.......................] - ETA: 12s - loss: 0.3863 - accuracy: 0.9342
 6100/25000 [======>.......................] - ETA: 12s - loss: 0.3874 - accuracy: 0.9334
 6200/25000 [======>.......................] - ETA: 12s - loss: 0.3866 - accuracy: 0.9340
 6300/25000 [======>.......................] - ETA: 12s - loss: 0.3866 - accuracy: 0.9340
 6400/25000 [======>.......................] - ETA: 12s - loss: 0.3866 - accuracy: 0.9341
 6500/25000 [======>.......................] - ETA: 12s - loss: 0.3861 - accuracy: 0.9345
 6600/25000 [======>.......................] - ETA: 12s - loss: 0.3860 - accuracy: 0.9344
 6700/25000 [=======>......................] - ETA: 12s - loss: 0.3864 - accuracy: 0.9340
 6800/25000 [=======>......................] - ETA: 12s - loss: 0.3860 - accuracy: 0.9343
 6900/25000 [=======>......................] - ETA: 12s - loss: 0.3858 - accuracy: 0.9345
 7000/25000 [=======>......................] - ETA: 12s - loss: 0.3854 - accuracy: 0.9349
 7100/25000 [=======>......................] - ETA: 11s - loss: 0.3856 - accuracy: 0.9344
 7200/25000 [=======>......................] - ETA: 11s - loss: 0.3850 - accuracy: 0.9349
 7300/25000 [=======>......................] - ETA: 11s - loss: 0.3849 - accuracy: 0.9348
 7400/25000 [=======>......................] - ETA: 11s - loss: 0.3852 - accuracy: 0.9345
 7500/25000 [========>.....................] - ETA: 11s - loss: 0.3848 - accuracy: 0.9348
 7600/25000 [========>.....................] - ETA: 11s - loss: 0.3846 - accuracy: 0.9349
 7700/25000 [========>.....................] - ETA: 11s - loss: 0.3845 - accuracy: 0.9351
 7800/25000 [========>.....................] - ETA: 11s - loss: 0.3841 - accuracy: 0.9354
 7900/25000 [========>.....................] - ETA: 11s - loss: 0.3842 - accuracy: 0.9352
 8000/25000 [========>.....................] - ETA: 11s - loss: 0.3841 - accuracy: 0.9352
 8100/25000 [========>.....................] - ETA: 11s - loss: 0.3838 - accuracy: 0.9356
 8200/25000 [========>.....................] - ETA: 11s - loss: 0.3834 - accuracy: 0.9360
 8300/25000 [========>.....................] - ETA: 11s - loss: 0.3833 - accuracy: 0.9360
 8400/25000 [=========>....................] - ETA: 11s - loss: 0.3830 - accuracy: 0.9363
 8500/25000 [=========>....................] - ETA: 10s - loss: 0.3831 - accuracy: 0.9361
 8600/25000 [=========>....................] - ETA: 10s - loss: 0.3837 - accuracy: 0.9355
 8700/25000 [=========>....................] - ETA: 10s - loss: 0.3838 - accuracy: 0.9351
 8800/25000 [=========>....................] - ETA: 10s - loss: 0.3834 - accuracy: 0.9353
 8900/25000 [=========>....................] - ETA: 10s - loss: 0.3834 - accuracy: 0.9352
 9000/25000 [=========>....................] - ETA: 10s - loss: 0.3831 - accuracy: 0.9356
 9100/25000 [=========>....................] - ETA: 10s - loss: 0.3828 - accuracy: 0.9357
 9200/25000 [==========>...................] - ETA: 10s - loss: 0.3825 - accuracy: 0.9360
 9300/25000 [==========>...................] - ETA: 10s - loss: 0.3824 - accuracy: 0.9361
 9400/25000 [==========>...................] - ETA: 10s - loss: 0.3825 - accuracy: 0.9359
 9500/25000 [==========>...................] - ETA: 10s - loss: 0.3827 - accuracy: 0.9358
 9600/25000 [==========>...................] - ETA: 10s - loss: 0.3826 - accuracy: 0.9358
 9700/25000 [==========>...................] - ETA: 10s - loss: 0.3827 - accuracy: 0.9356
 9800/25000 [==========>...................] - ETA: 9s - loss: 0.3828 - accuracy: 0.9356 
 9900/25000 [==========>...................] - ETA: 9s - loss: 0.3827 - accuracy: 0.9357
10000/25000 [===========>..................] - ETA: 9s - loss: 0.3828 - accuracy: 0.9356
10100/25000 [===========>..................] - ETA: 9s - loss: 0.3827 - accuracy: 0.9355
10200/25000 [===========>..................] - ETA: 9s - loss: 0.3825 - accuracy: 0.9358
10300/25000 [===========>..................] - ETA: 9s - loss: 0.3824 - accuracy: 0.9359
10400/25000 [===========>..................] - ETA: 9s - loss: 0.3823 - accuracy: 0.9358
10500/25000 [===========>..................] - ETA: 9s - loss: 0.3820 - accuracy: 0.9360
10600/25000 [===========>..................] - ETA: 9s - loss: 0.3818 - accuracy: 0.9361
10700/25000 [===========>..................] - ETA: 9s - loss: 0.3818 - accuracy: 0.9360
10800/25000 [===========>..................] - ETA: 9s - loss: 0.3816 - accuracy: 0.9361
10900/25000 [============>.................] - ETA: 9s - loss: 0.3818 - accuracy: 0.9359
11000/25000 [============>.................] - ETA: 9s - loss: 0.3817 - accuracy: 0.9358
11100/25000 [============>.................] - ETA: 9s - loss: 0.3813 - accuracy: 0.9361
11200/25000 [============>.................] - ETA: 9s - loss: 0.3813 - accuracy: 0.9361
11300/25000 [============>.................] - ETA: 8s - loss: 0.3812 - accuracy: 0.9362
11400/25000 [============>.................] - ETA: 8s - loss: 0.3810 - accuracy: 0.9362
11500/25000 [============>.................] - ETA: 8s - loss: 0.3809 - accuracy: 0.9364
11600/25000 [============>.................] - ETA: 8s - loss: 0.3805 - accuracy: 0.9367
11700/25000 [=============>................] - ETA: 8s - loss: 0.3803 - accuracy: 0.9370
11800/25000 [=============>................] - ETA: 8s - loss: 0.3801 - accuracy: 0.9371
11900/25000 [=============>................] - ETA: 8s - loss: 0.3801 - accuracy: 0.9371
12000/25000 [=============>................] - ETA: 8s - loss: 0.3804 - accuracy: 0.9366
12100/25000 [=============>................] - ETA: 8s - loss: 0.3805 - accuracy: 0.9366
12200/25000 [=============>................] - ETA: 8s - loss: 0.3804 - accuracy: 0.9367
12300/25000 [=============>................] - ETA: 8s - loss: 0.3805 - accuracy: 0.9365
12400/25000 [=============>................] - ETA: 8s - loss: 0.3806 - accuracy: 0.9363
12500/25000 [==============>...............] - ETA: 8s - loss: 0.3802 - accuracy: 0.9366
12600/25000 [==============>...............] - ETA: 8s - loss: 0.3803 - accuracy: 0.9365
12700/25000 [==============>...............] - ETA: 8s - loss: 0.3804 - accuracy: 0.9363
12800/25000 [==============>...............] - ETA: 7s - loss: 0.3806 - accuracy: 0.9360
12900/25000 [==============>...............] - ETA: 7s - loss: 0.3806 - accuracy: 0.9360
13000/25000 [==============>...............] - ETA: 7s - loss: 0.3806 - accuracy: 0.9358
13100/25000 [==============>...............] - ETA: 7s - loss: 0.3806 - accuracy: 0.9356
13200/25000 [==============>...............] - ETA: 7s - loss: 0.3805 - accuracy: 0.9358
13300/25000 [==============>...............] - ETA: 7s - loss: 0.3808 - accuracy: 0.9353
13400/25000 [===============>..............] - ETA: 7s - loss: 0.3807 - accuracy: 0.9354
13500/25000 [===============>..............] - ETA: 7s - loss: 0.3808 - accuracy: 0.9353
13600/25000 [===============>..............] - ETA: 7s - loss: 0.3806 - accuracy: 0.9353
13700/25000 [===============>..............] - ETA: 7s - loss: 0.3805 - accuracy: 0.9354
13800/25000 [===============>..............] - ETA: 7s - loss: 0.3802 - accuracy: 0.9357
13900/25000 [===============>..............] - ETA: 7s - loss: 0.3799 - accuracy: 0.9358
14000/25000 [===============>..............] - ETA: 7s - loss: 0.3797 - accuracy: 0.9360
14100/25000 [===============>..............] - ETA: 7s - loss: 0.3799 - accuracy: 0.9357
14200/25000 [================>.............] - ETA: 7s - loss: 0.3798 - accuracy: 0.9358
14300/25000 [================>.............] - ETA: 6s - loss: 0.3797 - accuracy: 0.9359
14400/25000 [================>.............] - ETA: 6s - loss: 0.3799 - accuracy: 0.9357
14500/25000 [================>.............] - ETA: 6s - loss: 0.3799 - accuracy: 0.9356
14600/25000 [================>.............] - ETA: 6s - loss: 0.3799 - accuracy: 0.9356
14700/25000 [================>.............] - ETA: 6s - loss: 0.3802 - accuracy: 0.9353
14800/25000 [================>.............] - ETA: 6s - loss: 0.3799 - accuracy: 0.9354
14900/25000 [================>.............] - ETA: 6s - loss: 0.3799 - accuracy: 0.9354
15000/25000 [=================>............] - ETA: 6s - loss: 0.3803 - accuracy: 0.9350
15100/25000 [=================>............] - ETA: 6s - loss: 0.3805 - accuracy: 0.9348
15200/25000 [=================>............] - ETA: 6s - loss: 0.3804 - accuracy: 0.9348
15300/25000 [=================>............] - ETA: 6s - loss: 0.3802 - accuracy: 0.9350
15400/25000 [=================>............] - ETA: 6s - loss: 0.3802 - accuracy: 0.9349
15500/25000 [=================>............] - ETA: 6s - loss: 0.3802 - accuracy: 0.9350
15600/25000 [=================>............] - ETA: 6s - loss: 0.3802 - accuracy: 0.9349
15700/25000 [=================>............] - ETA: 6s - loss: 0.3801 - accuracy: 0.9350
15800/25000 [=================>............] - ETA: 5s - loss: 0.3800 - accuracy: 0.9349
15900/25000 [==================>...........] - ETA: 5s - loss: 0.3798 - accuracy: 0.9350
16000/25000 [==================>...........] - ETA: 5s - loss: 0.3799 - accuracy: 0.9348
16100/25000 [==================>...........] - ETA: 5s - loss: 0.3801 - accuracy: 0.9345
16200/25000 [==================>...........] - ETA: 5s - loss: 0.3802 - accuracy: 0.9344
16300/25000 [==================>...........] - ETA: 5s - loss: 0.3801 - accuracy: 0.9345
16400/25000 [==================>...........] - ETA: 5s - loss: 0.3798 - accuracy: 0.9347
16500/25000 [==================>...........] - ETA: 5s - loss: 0.3796 - accuracy: 0.9347
16600/25000 [==================>...........] - ETA: 5s - loss: 0.3797 - accuracy: 0.9346
16700/25000 [===================>..........] - ETA: 5s - loss: 0.3796 - accuracy: 0.9347
16800/25000 [===================>..........] - ETA: 5s - loss: 0.3792 - accuracy: 0.9350
16900/25000 [===================>..........] - ETA: 5s - loss: 0.3794 - accuracy: 0.9348
17000/25000 [===================>..........] - ETA: 5s - loss: 0.3795 - accuracy: 0.9348
17100/25000 [===================>..........] - ETA: 5s - loss: 0.3792 - accuracy: 0.9349
17200/25000 [===================>..........] - ETA: 5s - loss: 0.3791 - accuracy: 0.9350
17300/25000 [===================>..........] - ETA: 5s - loss: 0.3788 - accuracy: 0.9353
17400/25000 [===================>..........] - ETA: 4s - loss: 0.3786 - accuracy: 0.9354
17500/25000 [====================>.........] - ETA: 4s - loss: 0.3787 - accuracy: 0.9354
17600/25000 [====================>.........] - ETA: 4s - loss: 0.3787 - accuracy: 0.9352
17700/25000 [====================>.........] - ETA: 4s - loss: 0.3791 - accuracy: 0.9348
17800/25000 [====================>.........] - ETA: 4s - loss: 0.3792 - accuracy: 0.9347
17900/25000 [====================>.........] - ETA: 4s - loss: 0.3791 - accuracy: 0.9347
18000/25000 [====================>.........] - ETA: 4s - loss: 0.3791 - accuracy: 0.9347
18100/25000 [====================>.........] - ETA: 4s - loss: 0.3790 - accuracy: 0.9345
18200/25000 [====================>.........] - ETA: 4s - loss: 0.3790 - accuracy: 0.9346
18300/25000 [====================>.........] - ETA: 4s - loss: 0.3789 - accuracy: 0.9345
18400/25000 [=====================>........] - ETA: 4s - loss: 0.3789 - accuracy: 0.9344
18500/25000 [=====================>........] - ETA: 4s - loss: 0.3788 - accuracy: 0.9345
18600/25000 [=====================>........] - ETA: 4s - loss: 0.3790 - accuracy: 0.9342
18700/25000 [=====================>........] - ETA: 4s - loss: 0.3790 - accuracy: 0.9341
18800/25000 [=====================>........] - ETA: 4s - loss: 0.3788 - accuracy: 0.9343
18900/25000 [=====================>........] - ETA: 3s - loss: 0.3789 - accuracy: 0.9341
19000/25000 [=====================>........] - ETA: 3s - loss: 0.3785 - accuracy: 0.9344
19100/25000 [=====================>........] - ETA: 3s - loss: 0.3786 - accuracy: 0.9342
19200/25000 [======================>.......] - ETA: 3s - loss: 0.3785 - accuracy: 0.9343
19300/25000 [======================>.......] - ETA: 3s - loss: 0.3786 - accuracy: 0.9341
19400/25000 [======================>.......] - ETA: 3s - loss: 0.3785 - accuracy: 0.9342
19500/25000 [======================>.......] - ETA: 3s - loss: 0.3784 - accuracy: 0.9343
19600/25000 [======================>.......] - ETA: 3s - loss: 0.3783 - accuracy: 0.9343
19700/25000 [======================>.......] - ETA: 3s - loss: 0.3784 - accuracy: 0.9342
19800/25000 [======================>.......] - ETA: 3s - loss: 0.3784 - accuracy: 0.9341
19900/25000 [======================>.......] - ETA: 3s - loss: 0.3784 - accuracy: 0.9342
20000/25000 [=======================>......] - ETA: 3s - loss: 0.3783 - accuracy: 0.9343
20100/25000 [=======================>......] - ETA: 3s - loss: 0.3783 - accuracy: 0.9342
20200/25000 [=======================>......] - ETA: 3s - loss: 0.3783 - accuracy: 0.9342
20300/25000 [=======================>......] - ETA: 3s - loss: 0.3784 - accuracy: 0.9341
20400/25000 [=======================>......] - ETA: 3s - loss: 0.3782 - accuracy: 0.9342
20500/25000 [=======================>......] - ETA: 2s - loss: 0.3781 - accuracy: 0.9342
20600/25000 [=======================>......] - ETA: 2s - loss: 0.3780 - accuracy: 0.9342
20700/25000 [=======================>......] - ETA: 2s - loss: 0.3780 - accuracy: 0.9341
20800/25000 [=======================>......] - ETA: 2s - loss: 0.3780 - accuracy: 0.9340
20900/25000 [========================>.....] - ETA: 2s - loss: 0.3779 - accuracy: 0.9342
21000/25000 [========================>.....] - ETA: 2s - loss: 0.3779 - accuracy: 0.9341
21100/25000 [========================>.....] - ETA: 2s - loss: 0.3780 - accuracy: 0.9340
21200/25000 [========================>.....] - ETA: 2s - loss: 0.3778 - accuracy: 0.9342
21300/25000 [========================>.....] - ETA: 2s - loss: 0.3776 - accuracy: 0.9343
21400/25000 [========================>.....] - ETA: 2s - loss: 0.3777 - accuracy: 0.9342
21500/25000 [========================>.....] - ETA: 2s - loss: 0.3774 - accuracy: 0.9344
21600/25000 [========================>.....] - ETA: 2s - loss: 0.3772 - accuracy: 0.9345
21700/25000 [=========================>....] - ETA: 2s - loss: 0.3771 - accuracy: 0.9346
21800/25000 [=========================>....] - ETA: 2s - loss: 0.3771 - accuracy: 0.9345
21900/25000 [=========================>....] - ETA: 2s - loss: 0.3772 - accuracy: 0.9343
22000/25000 [=========================>....] - ETA: 1s - loss: 0.3771 - accuracy: 0.9343
22100/25000 [=========================>....] - ETA: 1s - loss: 0.3772 - accuracy: 0.9342
22200/25000 [=========================>....] - ETA: 1s - loss: 0.3774 - accuracy: 0.9340
22300/25000 [=========================>....] - ETA: 1s - loss: 0.3774 - accuracy: 0.9339
22400/25000 [=========================>....] - ETA: 1s - loss: 0.3774 - accuracy: 0.9339
22500/25000 [==========================>...] - ETA: 1s - loss: 0.3775 - accuracy: 0.9339
22600/25000 [==========================>...] - ETA: 1s - loss: 0.3773 - accuracy: 0.9339
22700/25000 [==========================>...] - ETA: 1s - loss: 0.3773 - accuracy: 0.9339
22800/25000 [==========================>...] - ETA: 1s - loss: 0.3772 - accuracy: 0.9339
22900/25000 [==========================>...] - ETA: 1s - loss: 0.3771 - accuracy: 0.9340
23000/25000 [==========================>...] - ETA: 1s - loss: 0.3770 - accuracy: 0.9340
23100/25000 [==========================>...] - ETA: 1s - loss: 0.3770 - accuracy: 0.9340
23200/25000 [==========================>...] - ETA: 1s - loss: 0.3773 - accuracy: 0.9337
23300/25000 [==========================>...] - ETA: 1s - loss: 0.3774 - accuracy: 0.9336
23400/25000 [===========================>..] - ETA: 1s - loss: 0.3771 - accuracy: 0.9337
23500/25000 [===========================>..] - ETA: 0s - loss: 0.3770 - accuracy: 0.9337
23600/25000 [===========================>..] - ETA: 0s - loss: 0.3772 - accuracy: 0.9335
23700/25000 [===========================>..] - ETA: 0s - loss: 0.3770 - accuracy: 0.9337
23800/25000 [===========================>..] - ETA: 0s - loss: 0.3771 - accuracy: 0.9336
23900/25000 [===========================>..] - ETA: 0s - loss: 0.3770 - accuracy: 0.9336
24000/25000 [===========================>..] - ETA: 0s - loss: 0.3769 - accuracy: 0.9335
24100/25000 [===========================>..] - ETA: 0s - loss: 0.3769 - accuracy: 0.9336
24200/25000 [============================>.] - ETA: 0s - loss: 0.3768 - accuracy: 0.9336
24300/25000 [============================>.] - ETA: 0s - loss: 0.3768 - accuracy: 0.9335
24400/25000 [============================>.] - ETA: 0s - loss: 0.3766 - accuracy: 0.9336
24500/25000 [============================>.] - ETA: 0s - loss: 0.3766 - accuracy: 0.9336
24600/25000 [============================>.] - ETA: 0s - loss: 0.3764 - accuracy: 0.9337
24700/25000 [============================>.] - ETA: 0s - loss: 0.3763 - accuracy: 0.9338
24800/25000 [============================>.] - ETA: 0s - loss: 0.3763 - accuracy: 0.9337
24900/25000 [============================>.] - ETA: 0s - loss: 0.3761 - accuracy: 0.9339
25000/25000 [==============================] - 20s 819us/step - loss: 0.3762 - accuracy: 0.9338 - val_loss: 0.4379 - val_accuracy: 0.8609
Epoch 6/10

  100/25000 [..............................] - ETA: 15s - loss: 0.3203 - accuracy: 0.9700
  200/25000 [..............................] - ETA: 15s - loss: 0.3167 - accuracy: 0.9750
  300/25000 [..............................] - ETA: 15s - loss: 0.3291 - accuracy: 0.9633
  400/25000 [..............................] - ETA: 15s - loss: 0.3238 - accuracy: 0.9650
  500/25000 [..............................] - ETA: 15s - loss: 0.3284 - accuracy: 0.9600
  600/25000 [..............................] - ETA: 15s - loss: 0.3295 - accuracy: 0.9617
  700/25000 [..............................] - ETA: 15s - loss: 0.3311 - accuracy: 0.9614
  800/25000 [..............................] - ETA: 14s - loss: 0.3330 - accuracy: 0.9600
  900/25000 [>.............................] - ETA: 14s - loss: 0.3347 - accuracy: 0.9578
 1000/25000 [>.............................] - ETA: 14s - loss: 0.3366 - accuracy: 0.9570
 1100/25000 [>.............................] - ETA: 14s - loss: 0.3390 - accuracy: 0.9545
 1200/25000 [>.............................] - ETA: 14s - loss: 0.3419 - accuracy: 0.9517
 1300/25000 [>.............................] - ETA: 14s - loss: 0.3438 - accuracy: 0.9500
 1400/25000 [>.............................] - ETA: 14s - loss: 0.3456 - accuracy: 0.9479
 1500/25000 [>.............................] - ETA: 14s - loss: 0.3476 - accuracy: 0.9460
 1600/25000 [>.............................] - ETA: 14s - loss: 0.3469 - accuracy: 0.9463
 1700/25000 [=>............................] - ETA: 14s - loss: 0.3481 - accuracy: 0.9453
 1800/25000 [=>............................] - ETA: 14s - loss: 0.3487 - accuracy: 0.9450
 1900/25000 [=>............................] - ETA: 14s - loss: 0.3496 - accuracy: 0.9437
 2000/25000 [=>............................] - ETA: 14s - loss: 0.3493 - accuracy: 0.9435
 2100/25000 [=>............................] - ETA: 14s - loss: 0.3485 - accuracy: 0.9448
 2200/25000 [=>............................] - ETA: 14s - loss: 0.3488 - accuracy: 0.9445
 2300/25000 [=>............................] - ETA: 14s - loss: 0.3491 - accuracy: 0.9448
 2400/25000 [=>............................] - ETA: 14s - loss: 0.3486 - accuracy: 0.9454
 2500/25000 [==>...........................] - ETA: 14s - loss: 0.3482 - accuracy: 0.9456
 2600/25000 [==>...........................] - ETA: 14s - loss: 0.3488 - accuracy: 0.9454
 2700/25000 [==>...........................] - ETA: 13s - loss: 0.3500 - accuracy: 0.9444
 2800/25000 [==>...........................] - ETA: 13s - loss: 0.3493 - accuracy: 0.9450
 2900/25000 [==>...........................] - ETA: 13s - loss: 0.3486 - accuracy: 0.9452
 3000/25000 [==>...........................] - ETA: 13s - loss: 0.3486 - accuracy: 0.9450
 3100/25000 [==>...........................] - ETA: 13s - loss: 0.3485 - accuracy: 0.9452
 3200/25000 [==>...........................] - ETA: 13s - loss: 0.3480 - accuracy: 0.9456
 3300/25000 [==>...........................] - ETA: 13s - loss: 0.3473 - accuracy: 0.9467
 3400/25000 [===>..........................] - ETA: 13s - loss: 0.3480 - accuracy: 0.9459
 3500/25000 [===>..........................] - ETA: 13s - loss: 0.3479 - accuracy: 0.9463
 3600/25000 [===>..........................] - ETA: 13s - loss: 0.3478 - accuracy: 0.9464
 3700/25000 [===>..........................] - ETA: 13s - loss: 0.3469 - accuracy: 0.9470
 3800/25000 [===>..........................] - ETA: 13s - loss: 0.3464 - accuracy: 0.9474
 3900/25000 [===>..........................] - ETA: 13s - loss: 0.3467 - accuracy: 0.9472
 4000/25000 [===>..........................] - ETA: 13s - loss: 0.3465 - accuracy: 0.9475
 4100/25000 [===>..........................] - ETA: 13s - loss: 0.3462 - accuracy: 0.9476
 4200/25000 [====>.........................] - ETA: 13s - loss: 0.3467 - accuracy: 0.9471
 4300/25000 [====>.........................] - ETA: 13s - loss: 0.3467 - accuracy: 0.9470
 4400/25000 [====>.........................] - ETA: 12s - loss: 0.3461 - accuracy: 0.9475
 4500/25000 [====>.........................] - ETA: 12s - loss: 0.3460 - accuracy: 0.9480
 4600/25000 [====>.........................] - ETA: 12s - loss: 0.3463 - accuracy: 0.9474
 4700/25000 [====>.........................] - ETA: 12s - loss: 0.3468 - accuracy: 0.9470
 4800/25000 [====>.........................] - ETA: 12s - loss: 0.3475 - accuracy: 0.9460
 4900/25000 [====>.........................] - ETA: 12s - loss: 0.3480 - accuracy: 0.9453
 5000/25000 [=====>........................] - ETA: 12s - loss: 0.3475 - accuracy: 0.9458
 5100/25000 [=====>........................] - ETA: 12s - loss: 0.3477 - accuracy: 0.9455
 5200/25000 [=====>........................] - ETA: 12s - loss: 0.3482 - accuracy: 0.9452
 5300/25000 [=====>........................] - ETA: 12s - loss: 0.3491 - accuracy: 0.9442
 5400/25000 [=====>........................] - ETA: 12s - loss: 0.3485 - accuracy: 0.9448
 5500/25000 [=====>........................] - ETA: 12s - loss: 0.3484 - accuracy: 0.9451
 5600/25000 [=====>........................] - ETA: 12s - loss: 0.3486 - accuracy: 0.9452
 5700/25000 [=====>........................] - ETA: 12s - loss: 0.3489 - accuracy: 0.9449
 5800/25000 [=====>........................] - ETA: 12s - loss: 0.3490 - accuracy: 0.9447
 5900/25000 [======>.......................] - ETA: 12s - loss: 0.3493 - accuracy: 0.9444
 6000/25000 [======>.......................] - ETA: 12s - loss: 0.3494 - accuracy: 0.9442
 6100/25000 [======>.......................] - ETA: 11s - loss: 0.3498 - accuracy: 0.9439
 6200/25000 [======>.......................] - ETA: 11s - loss: 0.3501 - accuracy: 0.9435
 6300/25000 [======>.......................] - ETA: 11s - loss: 0.3503 - accuracy: 0.9432
 6400/25000 [======>.......................] - ETA: 11s - loss: 0.3502 - accuracy: 0.9433
 6500/25000 [======>.......................] - ETA: 11s - loss: 0.3499 - accuracy: 0.9435
 6600/25000 [======>.......................] - ETA: 11s - loss: 0.3498 - accuracy: 0.9436
 6700/25000 [=======>......................] - ETA: 11s - loss: 0.3502 - accuracy: 0.9433
 6800/25000 [=======>......................] - ETA: 11s - loss: 0.3498 - accuracy: 0.9437
 6900/25000 [=======>......................] - ETA: 11s - loss: 0.3502 - accuracy: 0.9432
 7000/25000 [=======>......................] - ETA: 11s - loss: 0.3498 - accuracy: 0.9436
 7100/25000 [=======>......................] - ETA: 11s - loss: 0.3493 - accuracy: 0.9441
 7200/25000 [=======>......................] - ETA: 11s - loss: 0.3495 - accuracy: 0.9440
 7300/25000 [=======>......................] - ETA: 11s - loss: 0.3490 - accuracy: 0.9444
 7400/25000 [=======>......................] - ETA: 11s - loss: 0.3486 - accuracy: 0.9446
 7500/25000 [========>.....................] - ETA: 11s - loss: 0.3487 - accuracy: 0.9444
 7600/25000 [========>.....................] - ETA: 11s - loss: 0.3485 - accuracy: 0.9443
 7700/25000 [========>.....................] - ETA: 10s - loss: 0.3487 - accuracy: 0.9440
 7800/25000 [========>.....................] - ETA: 10s - loss: 0.3490 - accuracy: 0.9437
 7900/25000 [========>.....................] - ETA: 10s - loss: 0.3489 - accuracy: 0.9438
 8000/25000 [========>.....................] - ETA: 10s - loss: 0.3487 - accuracy: 0.9440
 8100/25000 [========>.....................] - ETA: 10s - loss: 0.3483 - accuracy: 0.9443
 8200/25000 [========>.....................] - ETA: 10s - loss: 0.3477 - accuracy: 0.9448
 8300/25000 [========>.....................] - ETA: 10s - loss: 0.3477 - accuracy: 0.9447
 8400/25000 [=========>....................] - ETA: 10s - loss: 0.3472 - accuracy: 0.9452
 8500/25000 [=========>....................] - ETA: 10s - loss: 0.3471 - accuracy: 0.9454
 8600/25000 [=========>....................] - ETA: 10s - loss: 0.3472 - accuracy: 0.9452
 8700/25000 [=========>....................] - ETA: 10s - loss: 0.3471 - accuracy: 0.9452
 8800/25000 [=========>....................] - ETA: 10s - loss: 0.3471 - accuracy: 0.9451
 8900/25000 [=========>....................] - ETA: 10s - loss: 0.3474 - accuracy: 0.9447
 9000/25000 [=========>....................] - ETA: 10s - loss: 0.3473 - accuracy: 0.9448
 9100/25000 [=========>....................] - ETA: 10s - loss: 0.3471 - accuracy: 0.9449
 9200/25000 [==========>...................] - ETA: 10s - loss: 0.3470 - accuracy: 0.9449
 9300/25000 [==========>...................] - ETA: 9s - loss: 0.3474 - accuracy: 0.9445 
 9400/25000 [==========>...................] - ETA: 9s - loss: 0.3475 - accuracy: 0.9444
 9500/25000 [==========>...................] - ETA: 9s - loss: 0.3477 - accuracy: 0.9442
 9600/25000 [==========>...................] - ETA: 9s - loss: 0.3478 - accuracy: 0.9441
 9700/25000 [==========>...................] - ETA: 9s - loss: 0.3478 - accuracy: 0.9439
 9800/25000 [==========>...................] - ETA: 9s - loss: 0.3477 - accuracy: 0.9439
 9900/25000 [==========>...................] - ETA: 9s - loss: 0.3476 - accuracy: 0.9439
10000/25000 [===========>..................] - ETA: 9s - loss: 0.3476 - accuracy: 0.9439
10100/25000 [===========>..................] - ETA: 9s - loss: 0.3476 - accuracy: 0.9439
10200/25000 [===========>..................] - ETA: 9s - loss: 0.3476 - accuracy: 0.9439
10300/25000 [===========>..................] - ETA: 9s - loss: 0.3473 - accuracy: 0.9442
10400/25000 [===========>..................] - ETA: 9s - loss: 0.3470 - accuracy: 0.9444
10500/25000 [===========>..................] - ETA: 9s - loss: 0.3472 - accuracy: 0.9443
10600/25000 [===========>..................] - ETA: 9s - loss: 0.3470 - accuracy: 0.9442
10700/25000 [===========>..................] - ETA: 9s - loss: 0.3471 - accuracy: 0.9441
10800/25000 [===========>..................] - ETA: 9s - loss: 0.3467 - accuracy: 0.9444
10900/25000 [============>.................] - ETA: 8s - loss: 0.3470 - accuracy: 0.9441
11000/25000 [============>.................] - ETA: 8s - loss: 0.3469 - accuracy: 0.9441
11100/25000 [============>.................] - ETA: 8s - loss: 0.3477 - accuracy: 0.9434
11200/25000 [============>.................] - ETA: 8s - loss: 0.3474 - accuracy: 0.9436
11300/25000 [============>.................] - ETA: 8s - loss: 0.3472 - accuracy: 0.9438
11400/25000 [============>.................] - ETA: 8s - loss: 0.3474 - accuracy: 0.9437
11500/25000 [============>.................] - ETA: 8s - loss: 0.3477 - accuracy: 0.9434
11600/25000 [============>.................] - ETA: 8s - loss: 0.3480 - accuracy: 0.9431
11700/25000 [=============>................] - ETA: 8s - loss: 0.3480 - accuracy: 0.9432
11800/25000 [=============>................] - ETA: 8s - loss: 0.3482 - accuracy: 0.9430
11900/25000 [=============>................] - ETA: 8s - loss: 0.3479 - accuracy: 0.9431
12000/25000 [=============>................] - ETA: 8s - loss: 0.3478 - accuracy: 0.9432
12100/25000 [=============>................] - ETA: 8s - loss: 0.3476 - accuracy: 0.9433
12200/25000 [=============>................] - ETA: 8s - loss: 0.3477 - accuracy: 0.9433
12300/25000 [=============>................] - ETA: 8s - loss: 0.3477 - accuracy: 0.9433
12400/25000 [=============>................] - ETA: 8s - loss: 0.3474 - accuracy: 0.9435
12500/25000 [==============>...............] - ETA: 7s - loss: 0.3472 - accuracy: 0.9436
12600/25000 [==============>...............] - ETA: 7s - loss: 0.3472 - accuracy: 0.9436
12700/25000 [==============>...............] - ETA: 7s - loss: 0.3470 - accuracy: 0.9437
12800/25000 [==============>...............] - ETA: 7s - loss: 0.3474 - accuracy: 0.9434
12900/25000 [==============>...............] - ETA: 7s - loss: 0.3476 - accuracy: 0.9432
13000/25000 [==============>...............] - ETA: 7s - loss: 0.3475 - accuracy: 0.9433
13100/25000 [==============>...............] - ETA: 7s - loss: 0.3475 - accuracy: 0.9433
13200/25000 [==============>...............] - ETA: 7s - loss: 0.3476 - accuracy: 0.9432
13300/25000 [==============>...............] - ETA: 7s - loss: 0.3475 - accuracy: 0.9432
13400/25000 [===============>..............] - ETA: 7s - loss: 0.3478 - accuracy: 0.9428
13500/25000 [===============>..............] - ETA: 7s - loss: 0.3479 - accuracy: 0.9427
13600/25000 [===============>..............] - ETA: 7s - loss: 0.3477 - accuracy: 0.9429
13700/25000 [===============>..............] - ETA: 7s - loss: 0.3478 - accuracy: 0.9427
13800/25000 [===============>..............] - ETA: 7s - loss: 0.3478 - accuracy: 0.9427
13900/25000 [===============>..............] - ETA: 7s - loss: 0.3477 - accuracy: 0.9427
14000/25000 [===============>..............] - ETA: 7s - loss: 0.3476 - accuracy: 0.9428
14100/25000 [===============>..............] - ETA: 6s - loss: 0.3477 - accuracy: 0.9426
14200/25000 [================>.............] - ETA: 6s - loss: 0.3474 - accuracy: 0.9430
14300/25000 [================>.............] - ETA: 6s - loss: 0.3476 - accuracy: 0.9427
14400/25000 [================>.............] - ETA: 6s - loss: 0.3478 - accuracy: 0.9425
14500/25000 [================>.............] - ETA: 6s - loss: 0.3475 - accuracy: 0.9427
14600/25000 [================>.............] - ETA: 6s - loss: 0.3473 - accuracy: 0.9429
14700/25000 [================>.............] - ETA: 6s - loss: 0.3473 - accuracy: 0.9428
14800/25000 [================>.............] - ETA: 6s - loss: 0.3472 - accuracy: 0.9428
14900/25000 [================>.............] - ETA: 6s - loss: 0.3472 - accuracy: 0.9429
15000/25000 [=================>............] - ETA: 6s - loss: 0.3468 - accuracy: 0.9432
15100/25000 [=================>............] - ETA: 6s - loss: 0.3469 - accuracy: 0.9431
15200/25000 [=================>............] - ETA: 6s - loss: 0.3469 - accuracy: 0.9430
15300/25000 [=================>............] - ETA: 6s - loss: 0.3467 - accuracy: 0.9432
15400/25000 [=================>............] - ETA: 6s - loss: 0.3468 - accuracy: 0.9431
15500/25000 [=================>............] - ETA: 6s - loss: 0.3468 - accuracy: 0.9430
15600/25000 [=================>............] - ETA: 6s - loss: 0.3469 - accuracy: 0.9429
15700/25000 [=================>............] - ETA: 5s - loss: 0.3469 - accuracy: 0.9429
15800/25000 [=================>............] - ETA: 5s - loss: 0.3467 - accuracy: 0.9430
15900/25000 [==================>...........] - ETA: 5s - loss: 0.3466 - accuracy: 0.9430
16000/25000 [==================>...........] - ETA: 5s - loss: 0.3469 - accuracy: 0.9427
16100/25000 [==================>...........] - ETA: 5s - loss: 0.3467 - accuracy: 0.9429
16200/25000 [==================>...........] - ETA: 5s - loss: 0.3466 - accuracy: 0.9429
16300/25000 [==================>...........] - ETA: 5s - loss: 0.3465 - accuracy: 0.9429
16400/25000 [==================>...........] - ETA: 5s - loss: 0.3464 - accuracy: 0.9429
16500/25000 [==================>...........] - ETA: 5s - loss: 0.3464 - accuracy: 0.9428
16600/25000 [==================>...........] - ETA: 5s - loss: 0.3466 - accuracy: 0.9427
16700/25000 [===================>..........] - ETA: 5s - loss: 0.3466 - accuracy: 0.9426
16800/25000 [===================>..........] - ETA: 5s - loss: 0.3464 - accuracy: 0.9429
16900/25000 [===================>..........] - ETA: 5s - loss: 0.3461 - accuracy: 0.9431
17000/25000 [===================>..........] - ETA: 5s - loss: 0.3463 - accuracy: 0.9428
17100/25000 [===================>..........] - ETA: 5s - loss: 0.3462 - accuracy: 0.9429
17200/25000 [===================>..........] - ETA: 5s - loss: 0.3461 - accuracy: 0.9430
17300/25000 [===================>..........] - ETA: 4s - loss: 0.3460 - accuracy: 0.9431
17400/25000 [===================>..........] - ETA: 4s - loss: 0.3460 - accuracy: 0.9429
17500/25000 [====================>.........] - ETA: 4s - loss: 0.3461 - accuracy: 0.9428
17600/25000 [====================>.........] - ETA: 4s - loss: 0.3463 - accuracy: 0.9427
17700/25000 [====================>.........] - ETA: 4s - loss: 0.3465 - accuracy: 0.9424
17800/25000 [====================>.........] - ETA: 4s - loss: 0.3464 - accuracy: 0.9425
17900/25000 [====================>.........] - ETA: 4s - loss: 0.3462 - accuracy: 0.9425
18000/25000 [====================>.........] - ETA: 4s - loss: 0.3464 - accuracy: 0.9423
18100/25000 [====================>.........] - ETA: 4s - loss: 0.3463 - accuracy: 0.9424
18200/25000 [====================>.........] - ETA: 4s - loss: 0.3462 - accuracy: 0.9424
18300/25000 [====================>.........] - ETA: 4s - loss: 0.3461 - accuracy: 0.9425
18400/25000 [=====================>........] - ETA: 4s - loss: 0.3459 - accuracy: 0.9426
18500/25000 [=====================>........] - ETA: 4s - loss: 0.3458 - accuracy: 0.9425
18600/25000 [=====================>........] - ETA: 4s - loss: 0.3457 - accuracy: 0.9427
18700/25000 [=====================>........] - ETA: 4s - loss: 0.3456 - accuracy: 0.9428
18800/25000 [=====================>........] - ETA: 4s - loss: 0.3455 - accuracy: 0.9428
18900/25000 [=====================>........] - ETA: 3s - loss: 0.3455 - accuracy: 0.9429
19000/25000 [=====================>........] - ETA: 3s - loss: 0.3457 - accuracy: 0.9426
19100/25000 [=====================>........] - ETA: 3s - loss: 0.3456 - accuracy: 0.9427
19200/25000 [======================>.......] - ETA: 3s - loss: 0.3456 - accuracy: 0.9426
19300/25000 [======================>.......] - ETA: 3s - loss: 0.3458 - accuracy: 0.9424
19400/25000 [======================>.......] - ETA: 3s - loss: 0.3457 - accuracy: 0.9424
19500/25000 [======================>.......] - ETA: 3s - loss: 0.3457 - accuracy: 0.9424
19600/25000 [======================>.......] - ETA: 3s - loss: 0.3455 - accuracy: 0.9425
19700/25000 [======================>.......] - ETA: 3s - loss: 0.3454 - accuracy: 0.9425
19800/25000 [======================>.......] - ETA: 3s - loss: 0.3455 - accuracy: 0.9425
19900/25000 [======================>.......] - ETA: 3s - loss: 0.3454 - accuracy: 0.9425
20000/25000 [=======================>......] - ETA: 3s - loss: 0.3454 - accuracy: 0.9425
20100/25000 [=======================>......] - ETA: 3s - loss: 0.3452 - accuracy: 0.9426
20200/25000 [=======================>......] - ETA: 3s - loss: 0.3453 - accuracy: 0.9425
20300/25000 [=======================>......] - ETA: 3s - loss: 0.3453 - accuracy: 0.9425
20400/25000 [=======================>......] - ETA: 2s - loss: 0.3454 - accuracy: 0.9424
20500/25000 [=======================>......] - ETA: 2s - loss: 0.3452 - accuracy: 0.9426
20600/25000 [=======================>......] - ETA: 2s - loss: 0.3451 - accuracy: 0.9426
20700/25000 [=======================>......] - ETA: 2s - loss: 0.3449 - accuracy: 0.9428
20800/25000 [=======================>......] - ETA: 2s - loss: 0.3449 - accuracy: 0.9427
20900/25000 [========================>.....] - ETA: 2s - loss: 0.3448 - accuracy: 0.9427
21000/25000 [========================>.....] - ETA: 2s - loss: 0.3448 - accuracy: 0.9427
21100/25000 [========================>.....] - ETA: 2s - loss: 0.3448 - accuracy: 0.9426
21200/25000 [========================>.....] - ETA: 2s - loss: 0.3448 - accuracy: 0.9425
21300/25000 [========================>.....] - ETA: 2s - loss: 0.3447 - accuracy: 0.9425
21400/25000 [========================>.....] - ETA: 2s - loss: 0.3448 - accuracy: 0.9424
21500/25000 [========================>.....] - ETA: 2s - loss: 0.3448 - accuracy: 0.9424
21600/25000 [========================>.....] - ETA: 2s - loss: 0.3447 - accuracy: 0.9424
21700/25000 [=========================>....] - ETA: 2s - loss: 0.3445 - accuracy: 0.9425
21800/25000 [=========================>....] - ETA: 2s - loss: 0.3444 - accuracy: 0.9426
21900/25000 [=========================>....] - ETA: 1s - loss: 0.3444 - accuracy: 0.9425
22000/25000 [=========================>....] - ETA: 1s - loss: 0.3445 - accuracy: 0.9424
22100/25000 [=========================>....] - ETA: 1s - loss: 0.3445 - accuracy: 0.9423
22200/25000 [=========================>....] - ETA: 1s - loss: 0.3444 - accuracy: 0.9424
22300/25000 [=========================>....] - ETA: 1s - loss: 0.3444 - accuracy: 0.9424
22400/25000 [=========================>....] - ETA: 1s - loss: 0.3445 - accuracy: 0.9422
22500/25000 [==========================>...] - ETA: 1s - loss: 0.3444 - accuracy: 0.9423
22600/25000 [==========================>...] - ETA: 1s - loss: 0.3446 - accuracy: 0.9421
22700/25000 [==========================>...] - ETA: 1s - loss: 0.3445 - accuracy: 0.9421
22800/25000 [==========================>...] - ETA: 1s - loss: 0.3445 - accuracy: 0.9421
22900/25000 [==========================>...] - ETA: 1s - loss: 0.3444 - accuracy: 0.9421
23000/25000 [==========================>...] - ETA: 1s - loss: 0.3444 - accuracy: 0.9421
23100/25000 [==========================>...] - ETA: 1s - loss: 0.3442 - accuracy: 0.9422
23200/25000 [==========================>...] - ETA: 1s - loss: 0.3441 - accuracy: 0.9422
23300/25000 [==========================>...] - ETA: 1s - loss: 0.3439 - accuracy: 0.9423
23400/25000 [===========================>..] - ETA: 1s - loss: 0.3437 - accuracy: 0.9425
23500/25000 [===========================>..] - ETA: 0s - loss: 0.3435 - accuracy: 0.9426
23600/25000 [===========================>..] - ETA: 0s - loss: 0.3436 - accuracy: 0.9425
23700/25000 [===========================>..] - ETA: 0s - loss: 0.3436 - accuracy: 0.9424
23800/25000 [===========================>..] - ETA: 0s - loss: 0.3435 - accuracy: 0.9425
23900/25000 [===========================>..] - ETA: 0s - loss: 0.3433 - accuracy: 0.9426
24000/25000 [===========================>..] - ETA: 0s - loss: 0.3433 - accuracy: 0.9426
24100/25000 [===========================>..] - ETA: 0s - loss: 0.3432 - accuracy: 0.9426
24200/25000 [============================>.] - ETA: 0s - loss: 0.3432 - accuracy: 0.9426
24300/25000 [============================>.] - ETA: 0s - loss: 0.3432 - accuracy: 0.9426
24400/25000 [============================>.] - ETA: 0s - loss: 0.3435 - accuracy: 0.9423
24500/25000 [============================>.] - ETA: 0s - loss: 0.3434 - accuracy: 0.9424
24600/25000 [============================>.] - ETA: 0s - loss: 0.3433 - accuracy: 0.9425
24700/25000 [============================>.] - ETA: 0s - loss: 0.3432 - accuracy: 0.9425
24800/25000 [============================>.] - ETA: 0s - loss: 0.3433 - accuracy: 0.9423
24900/25000 [============================>.] - ETA: 0s - loss: 0.3432 - accuracy: 0.9424
25000/25000 [==============================] - 20s 810us/step - loss: 0.3430 - accuracy: 0.9424 - val_loss: 0.4257 - val_accuracy: 0.8595
Epoch 7/10

  100/25000 [..............................] - ETA: 19s - loss: 0.3392 - accuracy: 0.9500
  200/25000 [..............................] - ETA: 17s - loss: 0.3340 - accuracy: 0.9450
  300/25000 [..............................] - ETA: 17s - loss: 0.3305 - accuracy: 0.9467
  400/25000 [..............................] - ETA: 16s - loss: 0.3350 - accuracy: 0.9425
  500/25000 [..............................] - ETA: 16s - loss: 0.3362 - accuracy: 0.9400
  600/25000 [..............................] - ETA: 16s - loss: 0.3382 - accuracy: 0.9383
  700/25000 [..............................] - ETA: 16s - loss: 0.3385 - accuracy: 0.9386
  800/25000 [..............................] - ETA: 15s - loss: 0.3325 - accuracy: 0.9438
  900/25000 [>.............................] - ETA: 15s - loss: 0.3339 - accuracy: 0.9422
 1000/25000 [>.............................] - ETA: 15s - loss: 0.3317 - accuracy: 0.9440
 1100/25000 [>.............................] - ETA: 15s - loss: 0.3347 - accuracy: 0.9409
 1200/25000 [>.............................] - ETA: 15s - loss: 0.3331 - accuracy: 0.9417
 1300/25000 [>.............................] - ETA: 15s - loss: 0.3328 - accuracy: 0.9415
 1400/25000 [>.............................] - ETA: 15s - loss: 0.3323 - accuracy: 0.9421
 1500/25000 [>.............................] - ETA: 15s - loss: 0.3292 - accuracy: 0.9440
 1600/25000 [>.............................] - ETA: 15s - loss: 0.3283 - accuracy: 0.9450
 1700/25000 [=>............................] - ETA: 15s - loss: 0.3277 - accuracy: 0.9459
 1800/25000 [=>............................] - ETA: 15s - loss: 0.3256 - accuracy: 0.9478
 1900/25000 [=>............................] - ETA: 15s - loss: 0.3250 - accuracy: 0.9489
 2000/25000 [=>............................] - ETA: 14s - loss: 0.3248 - accuracy: 0.9490
 2100/25000 [=>............................] - ETA: 14s - loss: 0.3242 - accuracy: 0.9495
 2200/25000 [=>............................] - ETA: 14s - loss: 0.3226 - accuracy: 0.9505
 2300/25000 [=>............................] - ETA: 14s - loss: 0.3234 - accuracy: 0.9500
 2400/25000 [=>............................] - ETA: 14s - loss: 0.3225 - accuracy: 0.9504
 2500/25000 [==>...........................] - ETA: 14s - loss: 0.3238 - accuracy: 0.9492
 2600/25000 [==>...........................] - ETA: 14s - loss: 0.3232 - accuracy: 0.9496
 2700/25000 [==>...........................] - ETA: 14s - loss: 0.3235 - accuracy: 0.9489
 2800/25000 [==>...........................] - ETA: 14s - loss: 0.3226 - accuracy: 0.9500
 2900/25000 [==>...........................] - ETA: 14s - loss: 0.3236 - accuracy: 0.9490
 3000/25000 [==>...........................] - ETA: 14s - loss: 0.3242 - accuracy: 0.9483
 3100/25000 [==>...........................] - ETA: 14s - loss: 0.3235 - accuracy: 0.9490
 3200/25000 [==>...........................] - ETA: 14s - loss: 0.3231 - accuracy: 0.9494
 3300/25000 [==>...........................] - ETA: 13s - loss: 0.3233 - accuracy: 0.9488
 3400/25000 [===>..........................] - ETA: 13s - loss: 0.3238 - accuracy: 0.9482
 3500/25000 [===>..........................] - ETA: 13s - loss: 0.3253 - accuracy: 0.9466
 3600/25000 [===>..........................] - ETA: 13s - loss: 0.3260 - accuracy: 0.9458
 3700/25000 [===>..........................] - ETA: 13s - loss: 0.3262 - accuracy: 0.9454
 3800/25000 [===>..........................] - ETA: 13s - loss: 0.3270 - accuracy: 0.9445
 3900/25000 [===>..........................] - ETA: 13s - loss: 0.3267 - accuracy: 0.9449
 4000/25000 [===>..........................] - ETA: 13s - loss: 0.3268 - accuracy: 0.9448
 4100/25000 [===>..........................] - ETA: 13s - loss: 0.3261 - accuracy: 0.9451
 4200/25000 [====>.........................] - ETA: 13s - loss: 0.3270 - accuracy: 0.9448
 4300/25000 [====>.........................] - ETA: 13s - loss: 0.3273 - accuracy: 0.9444
 4400/25000 [====>.........................] - ETA: 13s - loss: 0.3266 - accuracy: 0.9448
 4500/25000 [====>.........................] - ETA: 13s - loss: 0.3275 - accuracy: 0.9442
 4600/25000 [====>.........................] - ETA: 13s - loss: 0.3287 - accuracy: 0.9433
 4700/25000 [====>.........................] - ETA: 13s - loss: 0.3281 - accuracy: 0.9438
 4800/25000 [====>.........................] - ETA: 12s - loss: 0.3279 - accuracy: 0.9440
 4900/25000 [====>.........................] - ETA: 12s - loss: 0.3277 - accuracy: 0.9439
 5000/25000 [=====>........................] - ETA: 12s - loss: 0.3271 - accuracy: 0.9444
 5100/25000 [=====>........................] - ETA: 12s - loss: 0.3269 - accuracy: 0.9443
 5200/25000 [=====>........................] - ETA: 12s - loss: 0.3265 - accuracy: 0.9448
 5300/25000 [=====>........................] - ETA: 12s - loss: 0.3268 - accuracy: 0.9443
 5400/25000 [=====>........................] - ETA: 12s - loss: 0.3267 - accuracy: 0.9444
 5500/25000 [=====>........................] - ETA: 12s - loss: 0.3267 - accuracy: 0.9445
 5600/25000 [=====>........................] - ETA: 12s - loss: 0.3262 - accuracy: 0.9448
 5700/25000 [=====>........................] - ETA: 12s - loss: 0.3252 - accuracy: 0.9456
 5800/25000 [=====>........................] - ETA: 12s - loss: 0.3254 - accuracy: 0.9455
 5900/25000 [======>.......................] - ETA: 12s - loss: 0.3251 - accuracy: 0.9458
 6000/25000 [======>.......................] - ETA: 12s - loss: 0.3247 - accuracy: 0.9462
 6100/25000 [======>.......................] - ETA: 12s - loss: 0.3242 - accuracy: 0.9464
 6200/25000 [======>.......................] - ETA: 11s - loss: 0.3237 - accuracy: 0.9468
 6300/25000 [======>.......................] - ETA: 11s - loss: 0.3236 - accuracy: 0.9470
 6400/25000 [======>.......................] - ETA: 11s - loss: 0.3234 - accuracy: 0.9470
 6500/25000 [======>.......................] - ETA: 11s - loss: 0.3227 - accuracy: 0.9477
 6600/25000 [======>.......................] - ETA: 11s - loss: 0.3235 - accuracy: 0.9473
 6700/25000 [=======>......................] - ETA: 11s - loss: 0.3241 - accuracy: 0.9467
 6800/25000 [=======>......................] - ETA: 11s - loss: 0.3243 - accuracy: 0.9465
 6900/25000 [=======>......................] - ETA: 11s - loss: 0.3239 - accuracy: 0.9468
 7000/25000 [=======>......................] - ETA: 11s - loss: 0.3237 - accuracy: 0.9470
 7100/25000 [=======>......................] - ETA: 11s - loss: 0.3236 - accuracy: 0.9469
 7200/25000 [=======>......................] - ETA: 11s - loss: 0.3234 - accuracy: 0.9471
 7300/25000 [=======>......................] - ETA: 11s - loss: 0.3235 - accuracy: 0.9470
 7400/25000 [=======>......................] - ETA: 11s - loss: 0.3237 - accuracy: 0.9469
 7500/25000 [========>.....................] - ETA: 11s - loss: 0.3237 - accuracy: 0.9469
 7600/25000 [========>.....................] - ETA: 11s - loss: 0.3232 - accuracy: 0.9474
 7700/25000 [========>.....................] - ETA: 11s - loss: 0.3235 - accuracy: 0.9471
 7800/25000 [========>.....................] - ETA: 11s - loss: 0.3232 - accuracy: 0.9473
 7900/25000 [========>.....................] - ETA: 11s - loss: 0.3232 - accuracy: 0.9473
 8000/25000 [========>.....................] - ETA: 11s - loss: 0.3235 - accuracy: 0.9469
 8100/25000 [========>.....................] - ETA: 10s - loss: 0.3238 - accuracy: 0.9465
 8200/25000 [========>.....................] - ETA: 10s - loss: 0.3235 - accuracy: 0.9467
 8300/25000 [========>.....................] - ETA: 10s - loss: 0.3234 - accuracy: 0.9469
 8400/25000 [=========>....................] - ETA: 10s - loss: 0.3231 - accuracy: 0.9471
 8500/25000 [=========>....................] - ETA: 10s - loss: 0.3234 - accuracy: 0.9468
 8600/25000 [=========>....................] - ETA: 10s - loss: 0.3238 - accuracy: 0.9464
 8700/25000 [=========>....................] - ETA: 10s - loss: 0.3232 - accuracy: 0.9468
 8800/25000 [=========>....................] - ETA: 10s - loss: 0.3231 - accuracy: 0.9468
 8900/25000 [=========>....................] - ETA: 10s - loss: 0.3228 - accuracy: 0.9471
 9000/25000 [=========>....................] - ETA: 10s - loss: 0.3225 - accuracy: 0.9472
 9100/25000 [=========>....................] - ETA: 10s - loss: 0.3225 - accuracy: 0.9471
 9200/25000 [==========>...................] - ETA: 10s - loss: 0.3218 - accuracy: 0.9477
 9300/25000 [==========>...................] - ETA: 10s - loss: 0.3220 - accuracy: 0.9475
 9400/25000 [==========>...................] - ETA: 10s - loss: 0.3221 - accuracy: 0.9473
 9500/25000 [==========>...................] - ETA: 10s - loss: 0.3219 - accuracy: 0.9476
 9600/25000 [==========>...................] - ETA: 9s - loss: 0.3220 - accuracy: 0.9475 
 9700/25000 [==========>...................] - ETA: 9s - loss: 0.3219 - accuracy: 0.9476
 9800/25000 [==========>...................] - ETA: 9s - loss: 0.3223 - accuracy: 0.9472
 9900/25000 [==========>...................] - ETA: 9s - loss: 0.3221 - accuracy: 0.9474
10000/25000 [===========>..................] - ETA: 9s - loss: 0.3224 - accuracy: 0.9472
10100/25000 [===========>..................] - ETA: 9s - loss: 0.3225 - accuracy: 0.9470
10200/25000 [===========>..................] - ETA: 9s - loss: 0.3226 - accuracy: 0.9470
10300/25000 [===========>..................] - ETA: 9s - loss: 0.3230 - accuracy: 0.9465
10400/25000 [===========>..................] - ETA: 9s - loss: 0.3228 - accuracy: 0.9466
10500/25000 [===========>..................] - ETA: 9s - loss: 0.3230 - accuracy: 0.9463
10600/25000 [===========>..................] - ETA: 9s - loss: 0.3228 - accuracy: 0.9464
10700/25000 [===========>..................] - ETA: 9s - loss: 0.3225 - accuracy: 0.9466
10800/25000 [===========>..................] - ETA: 9s - loss: 0.3221 - accuracy: 0.9469
10900/25000 [============>.................] - ETA: 9s - loss: 0.3219 - accuracy: 0.9471
11000/25000 [============>.................] - ETA: 9s - loss: 0.3217 - accuracy: 0.9472
11100/25000 [============>.................] - ETA: 8s - loss: 0.3217 - accuracy: 0.9472
11200/25000 [============>.................] - ETA: 8s - loss: 0.3216 - accuracy: 0.9473
11300/25000 [============>.................] - ETA: 8s - loss: 0.3214 - accuracy: 0.9474
11400/25000 [============>.................] - ETA: 8s - loss: 0.3215 - accuracy: 0.9475
11500/25000 [============>.................] - ETA: 8s - loss: 0.3214 - accuracy: 0.9475
11600/25000 [============>.................] - ETA: 8s - loss: 0.3212 - accuracy: 0.9476
11700/25000 [=============>................] - ETA: 8s - loss: 0.3212 - accuracy: 0.9475
11800/25000 [=============>................] - ETA: 8s - loss: 0.3210 - accuracy: 0.9476
11900/25000 [=============>................] - ETA: 8s - loss: 0.3211 - accuracy: 0.9475
12000/25000 [=============>................] - ETA: 8s - loss: 0.3212 - accuracy: 0.9474
12100/25000 [=============>................] - ETA: 8s - loss: 0.3208 - accuracy: 0.9477
12200/25000 [=============>................] - ETA: 8s - loss: 0.3210 - accuracy: 0.9475
12300/25000 [=============>................] - ETA: 8s - loss: 0.3211 - accuracy: 0.9475
12400/25000 [=============>................] - ETA: 8s - loss: 0.3212 - accuracy: 0.9473
12500/25000 [==============>...............] - ETA: 8s - loss: 0.3211 - accuracy: 0.9472
12600/25000 [==============>...............] - ETA: 7s - loss: 0.3209 - accuracy: 0.9473
12700/25000 [==============>...............] - ETA: 7s - loss: 0.3209 - accuracy: 0.9473
12800/25000 [==============>...............] - ETA: 7s - loss: 0.3206 - accuracy: 0.9474
12900/25000 [==============>...............] - ETA: 7s - loss: 0.3205 - accuracy: 0.9474
13000/25000 [==============>...............] - ETA: 7s - loss: 0.3208 - accuracy: 0.9471
13100/25000 [==============>...............] - ETA: 7s - loss: 0.3207 - accuracy: 0.9471
13200/25000 [==============>...............] - ETA: 7s - loss: 0.3206 - accuracy: 0.9472
13300/25000 [==============>...............] - ETA: 7s - loss: 0.3202 - accuracy: 0.9475
13400/25000 [===============>..............] - ETA: 7s - loss: 0.3203 - accuracy: 0.9475
13500/25000 [===============>..............] - ETA: 7s - loss: 0.3203 - accuracy: 0.9473
13600/25000 [===============>..............] - ETA: 7s - loss: 0.3200 - accuracy: 0.9475
13700/25000 [===============>..............] - ETA: 7s - loss: 0.3201 - accuracy: 0.9474
13800/25000 [===============>..............] - ETA: 7s - loss: 0.3200 - accuracy: 0.9473
13900/25000 [===============>..............] - ETA: 7s - loss: 0.3203 - accuracy: 0.9471
14000/25000 [===============>..............] - ETA: 7s - loss: 0.3205 - accuracy: 0.9469
14100/25000 [===============>..............] - ETA: 7s - loss: 0.3203 - accuracy: 0.9470
14200/25000 [================>.............] - ETA: 6s - loss: 0.3202 - accuracy: 0.9470
14300/25000 [================>.............] - ETA: 6s - loss: 0.3199 - accuracy: 0.9473
14400/25000 [================>.............] - ETA: 6s - loss: 0.3197 - accuracy: 0.9474
14500/25000 [================>.............] - ETA: 6s - loss: 0.3197 - accuracy: 0.9473
14600/25000 [================>.............] - ETA: 6s - loss: 0.3195 - accuracy: 0.9474
14700/25000 [================>.............] - ETA: 6s - loss: 0.3195 - accuracy: 0.9473
14800/25000 [================>.............] - ETA: 6s - loss: 0.3196 - accuracy: 0.9472
14900/25000 [================>.............] - ETA: 6s - loss: 0.3193 - accuracy: 0.9474
15000/25000 [=================>............] - ETA: 6s - loss: 0.3193 - accuracy: 0.9474
15100/25000 [=================>............] - ETA: 6s - loss: 0.3191 - accuracy: 0.9475
15200/25000 [=================>............] - ETA: 6s - loss: 0.3191 - accuracy: 0.9476
15300/25000 [=================>............] - ETA: 6s - loss: 0.3190 - accuracy: 0.9477
15400/25000 [=================>............] - ETA: 6s - loss: 0.3190 - accuracy: 0.9477
15500/25000 [=================>............] - ETA: 6s - loss: 0.3188 - accuracy: 0.9477
15600/25000 [=================>............] - ETA: 6s - loss: 0.3187 - accuracy: 0.9478
15700/25000 [=================>............] - ETA: 5s - loss: 0.3186 - accuracy: 0.9478
15800/25000 [=================>............] - ETA: 5s - loss: 0.3190 - accuracy: 0.9475
15900/25000 [==================>...........] - ETA: 5s - loss: 0.3189 - accuracy: 0.9475
16000/25000 [==================>...........] - ETA: 5s - loss: 0.3189 - accuracy: 0.9475
16100/25000 [==================>...........] - ETA: 5s - loss: 0.3191 - accuracy: 0.9473
16200/25000 [==================>...........] - ETA: 5s - loss: 0.3192 - accuracy: 0.9472
16300/25000 [==================>...........] - ETA: 5s - loss: 0.3192 - accuracy: 0.9472
16400/25000 [==================>...........] - ETA: 5s - loss: 0.3194 - accuracy: 0.9470
16500/25000 [==================>...........] - ETA: 5s - loss: 0.3194 - accuracy: 0.9471
16600/25000 [==================>...........] - ETA: 5s - loss: 0.3192 - accuracy: 0.9471
16700/25000 [===================>..........] - ETA: 5s - loss: 0.3192 - accuracy: 0.9472
16800/25000 [===================>..........] - ETA: 5s - loss: 0.3195 - accuracy: 0.9469
16900/25000 [===================>..........] - ETA: 5s - loss: 0.3194 - accuracy: 0.9469
17000/25000 [===================>..........] - ETA: 5s - loss: 0.3194 - accuracy: 0.9469
17100/25000 [===================>..........] - ETA: 5s - loss: 0.3195 - accuracy: 0.9468
17200/25000 [===================>..........] - ETA: 4s - loss: 0.3195 - accuracy: 0.9468
17300/25000 [===================>..........] - ETA: 4s - loss: 0.3194 - accuracy: 0.9468
17400/25000 [===================>..........] - ETA: 4s - loss: 0.3192 - accuracy: 0.9470
17500/25000 [====================>.........] - ETA: 4s - loss: 0.3193 - accuracy: 0.9469
17600/25000 [====================>.........] - ETA: 4s - loss: 0.3192 - accuracy: 0.9469
17700/25000 [====================>.........] - ETA: 4s - loss: 0.3193 - accuracy: 0.9467
17800/25000 [====================>.........] - ETA: 4s - loss: 0.3194 - accuracy: 0.9466
17900/25000 [====================>.........] - ETA: 4s - loss: 0.3194 - accuracy: 0.9466
18000/25000 [====================>.........] - ETA: 4s - loss: 0.3195 - accuracy: 0.9465
18100/25000 [====================>.........] - ETA: 4s - loss: 0.3195 - accuracy: 0.9465
18200/25000 [====================>.........] - ETA: 4s - loss: 0.3194 - accuracy: 0.9465
18300/25000 [====================>.........] - ETA: 4s - loss: 0.3194 - accuracy: 0.9464
18400/25000 [=====================>........] - ETA: 4s - loss: 0.3193 - accuracy: 0.9465
18500/25000 [=====================>........] - ETA: 4s - loss: 0.3192 - accuracy: 0.9465
18600/25000 [=====================>........] - ETA: 4s - loss: 0.3191 - accuracy: 0.9466
18700/25000 [=====================>........] - ETA: 4s - loss: 0.3189 - accuracy: 0.9468
18800/25000 [=====================>........] - ETA: 3s - loss: 0.3186 - accuracy: 0.9469
18900/25000 [=====================>........] - ETA: 3s - loss: 0.3187 - accuracy: 0.9468
19000/25000 [=====================>........] - ETA: 3s - loss: 0.3186 - accuracy: 0.9469
19100/25000 [=====================>........] - ETA: 3s - loss: 0.3185 - accuracy: 0.9469
19200/25000 [======================>.......] - ETA: 3s - loss: 0.3186 - accuracy: 0.9468
19300/25000 [======================>.......] - ETA: 3s - loss: 0.3186 - accuracy: 0.9467
19400/25000 [======================>.......] - ETA: 3s - loss: 0.3186 - accuracy: 0.9467
19500/25000 [======================>.......] - ETA: 3s - loss: 0.3186 - accuracy: 0.9467
19600/25000 [======================>.......] - ETA: 3s - loss: 0.3187 - accuracy: 0.9465
19700/25000 [======================>.......] - ETA: 3s - loss: 0.3185 - accuracy: 0.9466
19800/25000 [======================>.......] - ETA: 3s - loss: 0.3186 - accuracy: 0.9465
19900/25000 [======================>.......] - ETA: 3s - loss: 0.3186 - accuracy: 0.9464
20000/25000 [=======================>......] - ETA: 3s - loss: 0.3186 - accuracy: 0.9463
20100/25000 [=======================>......] - ETA: 3s - loss: 0.3185 - accuracy: 0.9464
20200/25000 [=======================>......] - ETA: 3s - loss: 0.3184 - accuracy: 0.9464
20300/25000 [=======================>......] - ETA: 2s - loss: 0.3183 - accuracy: 0.9465
20400/25000 [=======================>......] - ETA: 2s - loss: 0.3181 - accuracy: 0.9466
20500/25000 [=======================>......] - ETA: 2s - loss: 0.3181 - accuracy: 0.9465
20600/25000 [=======================>......] - ETA: 2s - loss: 0.3180 - accuracy: 0.9467
20700/25000 [=======================>......] - ETA: 2s - loss: 0.3178 - accuracy: 0.9467
20800/25000 [=======================>......] - ETA: 2s - loss: 0.3180 - accuracy: 0.9465
20900/25000 [========================>.....] - ETA: 2s - loss: 0.3179 - accuracy: 0.9466
21000/25000 [========================>.....] - ETA: 2s - loss: 0.3179 - accuracy: 0.9465
21100/25000 [========================>.....] - ETA: 2s - loss: 0.3181 - accuracy: 0.9464
21200/25000 [========================>.....] - ETA: 2s - loss: 0.3181 - accuracy: 0.9463
21300/25000 [========================>.....] - ETA: 2s - loss: 0.3181 - accuracy: 0.9463
21400/25000 [========================>.....] - ETA: 2s - loss: 0.3182 - accuracy: 0.9462
21500/25000 [========================>.....] - ETA: 2s - loss: 0.3180 - accuracy: 0.9463
21600/25000 [========================>.....] - ETA: 2s - loss: 0.3179 - accuracy: 0.9463
21700/25000 [=========================>....] - ETA: 2s - loss: 0.3182 - accuracy: 0.9461
21800/25000 [=========================>....] - ETA: 2s - loss: 0.3180 - accuracy: 0.9462
21900/25000 [=========================>....] - ETA: 1s - loss: 0.3178 - accuracy: 0.9463
22000/25000 [=========================>....] - ETA: 1s - loss: 0.3177 - accuracy: 0.9464
22100/25000 [=========================>....] - ETA: 1s - loss: 0.3178 - accuracy: 0.9462
22200/25000 [=========================>....] - ETA: 1s - loss: 0.3178 - accuracy: 0.9462
22300/25000 [=========================>....] - ETA: 1s - loss: 0.3176 - accuracy: 0.9463
22400/25000 [=========================>....] - ETA: 1s - loss: 0.3173 - accuracy: 0.9466
22500/25000 [==========================>...] - ETA: 1s - loss: 0.3173 - accuracy: 0.9465
22600/25000 [==========================>...] - ETA: 1s - loss: 0.3173 - accuracy: 0.9465
22700/25000 [==========================>...] - ETA: 1s - loss: 0.3173 - accuracy: 0.9465
22800/25000 [==========================>...] - ETA: 1s - loss: 0.3173 - accuracy: 0.9465
22900/25000 [==========================>...] - ETA: 1s - loss: 0.3170 - accuracy: 0.9467
23000/25000 [==========================>...] - ETA: 1s - loss: 0.3169 - accuracy: 0.9468
23100/25000 [==========================>...] - ETA: 1s - loss: 0.3171 - accuracy: 0.9465
23200/25000 [==========================>...] - ETA: 1s - loss: 0.3169 - accuracy: 0.9466
23300/25000 [==========================>...] - ETA: 1s - loss: 0.3168 - accuracy: 0.9467
23400/25000 [===========================>..] - ETA: 1s - loss: 0.3168 - accuracy: 0.9466
23500/25000 [===========================>..] - ETA: 0s - loss: 0.3167 - accuracy: 0.9467
23600/25000 [===========================>..] - ETA: 0s - loss: 0.3165 - accuracy: 0.9468
23700/25000 [===========================>..] - ETA: 0s - loss: 0.3165 - accuracy: 0.9468
23800/25000 [===========================>..] - ETA: 0s - loss: 0.3164 - accuracy: 0.9469
23900/25000 [===========================>..] - ETA: 0s - loss: 0.3164 - accuracy: 0.9468
24000/25000 [===========================>..] - ETA: 0s - loss: 0.3163 - accuracy: 0.9469
24100/25000 [===========================>..] - ETA: 0s - loss: 0.3163 - accuracy: 0.9469
24200/25000 [============================>.] - ETA: 0s - loss: 0.3163 - accuracy: 0.9469
24300/25000 [============================>.] - ETA: 0s - loss: 0.3164 - accuracy: 0.9467
24400/25000 [============================>.] - ETA: 0s - loss: 0.3164 - accuracy: 0.9467
24500/25000 [============================>.] - ETA: 0s - loss: 0.3165 - accuracy: 0.9466
24600/25000 [============================>.] - ETA: 0s - loss: 0.3166 - accuracy: 0.9465
24700/25000 [============================>.] - ETA: 0s - loss: 0.3163 - accuracy: 0.9467
24800/25000 [============================>.] - ETA: 0s - loss: 0.3163 - accuracy: 0.9467
24900/25000 [============================>.] - ETA: 0s - loss: 0.3161 - accuracy: 0.9468
25000/25000 [==============================] - 20s 812us/step - loss: 0.3160 - accuracy: 0.9468 - val_loss: 0.4216 - val_accuracy: 0.8556
Epoch 8/10

  100/25000 [..............................] - ETA: 15s - loss: 0.3084 - accuracy: 0.9400
  200/25000 [..............................] - ETA: 15s - loss: 0.2927 - accuracy: 0.9550
  300/25000 [..............................] - ETA: 15s - loss: 0.2960 - accuracy: 0.9533
  400/25000 [..............................] - ETA: 15s - loss: 0.2905 - accuracy: 0.9575
  500/25000 [..............................] - ETA: 16s - loss: 0.2814 - accuracy: 0.9640
  600/25000 [..............................] - ETA: 16s - loss: 0.2841 - accuracy: 0.9617
  700/25000 [..............................] - ETA: 16s - loss: 0.2863 - accuracy: 0.9600
  800/25000 [..............................] - ETA: 16s - loss: 0.2862 - accuracy: 0.9600
  900/25000 [>.............................] - ETA: 16s - loss: 0.2855 - accuracy: 0.9611
 1000/25000 [>.............................] - ETA: 16s - loss: 0.2834 - accuracy: 0.9630
 1100/25000 [>.............................] - ETA: 16s - loss: 0.2850 - accuracy: 0.9618
 1200/25000 [>.............................] - ETA: 16s - loss: 0.2863 - accuracy: 0.9608
 1300/25000 [>.............................] - ETA: 16s - loss: 0.2875 - accuracy: 0.9592
 1400/25000 [>.............................] - ETA: 16s - loss: 0.2892 - accuracy: 0.9579
 1500/25000 [>.............................] - ETA: 16s - loss: 0.2898 - accuracy: 0.9573
 1600/25000 [>.............................] - ETA: 15s - loss: 0.2920 - accuracy: 0.9556
 1700/25000 [=>............................] - ETA: 15s - loss: 0.2954 - accuracy: 0.9529
 1800/25000 [=>............................] - ETA: 15s - loss: 0.2948 - accuracy: 0.9533
 1900/25000 [=>............................] - ETA: 15s - loss: 0.2940 - accuracy: 0.9542
 2000/25000 [=>............................] - ETA: 15s - loss: 0.2946 - accuracy: 0.9540
 2100/25000 [=>............................] - ETA: 15s - loss: 0.2936 - accuracy: 0.9543
 2200/25000 [=>............................] - ETA: 15s - loss: 0.2932 - accuracy: 0.9545
 2300/25000 [=>............................] - ETA: 15s - loss: 0.2940 - accuracy: 0.9543
 2400/25000 [=>............................] - ETA: 15s - loss: 0.2948 - accuracy: 0.9538
 2500/25000 [==>...........................] - ETA: 15s - loss: 0.2941 - accuracy: 0.9544
 2600/25000 [==>...........................] - ETA: 15s - loss: 0.2941 - accuracy: 0.9546
 2700/25000 [==>...........................] - ETA: 15s - loss: 0.2929 - accuracy: 0.9556
 2800/25000 [==>...........................] - ETA: 15s - loss: 0.2927 - accuracy: 0.9561
 2900/25000 [==>...........................] - ETA: 15s - loss: 0.2944 - accuracy: 0.9548
 3000/25000 [==>...........................] - ETA: 15s - loss: 0.2949 - accuracy: 0.9547
 3100/25000 [==>...........................] - ETA: 15s - loss: 0.2955 - accuracy: 0.9539
 3200/25000 [==>...........................] - ETA: 15s - loss: 0.2958 - accuracy: 0.9541
 3300/25000 [==>...........................] - ETA: 15s - loss: 0.2949 - accuracy: 0.9548
 3400/25000 [===>..........................] - ETA: 14s - loss: 0.2940 - accuracy: 0.9550
 3500/25000 [===>..........................] - ETA: 14s - loss: 0.2936 - accuracy: 0.9554
 3600/25000 [===>..........................] - ETA: 14s - loss: 0.2933 - accuracy: 0.9558
 3700/25000 [===>..........................] - ETA: 14s - loss: 0.2932 - accuracy: 0.9559
 3800/25000 [===>..........................] - ETA: 14s - loss: 0.2932 - accuracy: 0.9561
 3900/25000 [===>..........................] - ETA: 14s - loss: 0.2933 - accuracy: 0.9559
 4000/25000 [===>..........................] - ETA: 14s - loss: 0.2928 - accuracy: 0.9560
 4100/25000 [===>..........................] - ETA: 14s - loss: 0.2930 - accuracy: 0.9559
 4200/25000 [====>.........................] - ETA: 14s - loss: 0.2946 - accuracy: 0.9545
 4300/25000 [====>.........................] - ETA: 14s - loss: 0.2939 - accuracy: 0.9551
 4400/25000 [====>.........................] - ETA: 14s - loss: 0.2945 - accuracy: 0.9545
 4500/25000 [====>.........................] - ETA: 14s - loss: 0.2944 - accuracy: 0.9547
 4600/25000 [====>.........................] - ETA: 13s - loss: 0.2946 - accuracy: 0.9546
 4700/25000 [====>.........................] - ETA: 13s - loss: 0.2950 - accuracy: 0.9545
 4800/25000 [====>.........................] - ETA: 13s - loss: 0.2951 - accuracy: 0.9546
 4900/25000 [====>.........................] - ETA: 13s - loss: 0.2943 - accuracy: 0.9551
 5000/25000 [=====>........................] - ETA: 13s - loss: 0.2940 - accuracy: 0.9554
 5100/25000 [=====>........................] - ETA: 13s - loss: 0.2943 - accuracy: 0.9553
 5200/25000 [=====>........................] - ETA: 13s - loss: 0.2946 - accuracy: 0.9552
 5300/25000 [=====>........................] - ETA: 13s - loss: 0.2943 - accuracy: 0.9555
 5400/25000 [=====>........................] - ETA: 13s - loss: 0.2947 - accuracy: 0.9552
 5500/25000 [=====>........................] - ETA: 13s - loss: 0.2958 - accuracy: 0.9544
 5600/25000 [=====>........................] - ETA: 13s - loss: 0.2963 - accuracy: 0.9541
 5700/25000 [=====>........................] - ETA: 12s - loss: 0.2962 - accuracy: 0.9540
 5800/25000 [=====>........................] - ETA: 12s - loss: 0.2964 - accuracy: 0.9536
 5900/25000 [======>.......................] - ETA: 12s - loss: 0.2969 - accuracy: 0.9531
 6000/25000 [======>.......................] - ETA: 12s - loss: 0.2983 - accuracy: 0.9518
 6100/25000 [======>.......................] - ETA: 12s - loss: 0.2988 - accuracy: 0.9515
 6200/25000 [======>.......................] - ETA: 12s - loss: 0.2986 - accuracy: 0.9516
 6300/25000 [======>.......................] - ETA: 12s - loss: 0.2985 - accuracy: 0.9516
 6400/25000 [======>.......................] - ETA: 12s - loss: 0.2987 - accuracy: 0.9514
 6500/25000 [======>.......................] - ETA: 12s - loss: 0.2985 - accuracy: 0.9515
 6600/25000 [======>.......................] - ETA: 12s - loss: 0.2982 - accuracy: 0.9518
 6700/25000 [=======>......................] - ETA: 12s - loss: 0.2978 - accuracy: 0.9521
 6800/25000 [=======>......................] - ETA: 12s - loss: 0.2977 - accuracy: 0.9521
 6900/25000 [=======>......................] - ETA: 12s - loss: 0.2973 - accuracy: 0.9525
 7000/25000 [=======>......................] - ETA: 12s - loss: 0.2972 - accuracy: 0.9526
 7100/25000 [=======>......................] - ETA: 11s - loss: 0.2969 - accuracy: 0.9527
 7200/25000 [=======>......................] - ETA: 11s - loss: 0.2970 - accuracy: 0.9526
 7300/25000 [=======>......................] - ETA: 11s - loss: 0.2970 - accuracy: 0.9525
 7400/25000 [=======>......................] - ETA: 11s - loss: 0.2969 - accuracy: 0.9524
 7500/25000 [========>.....................] - ETA: 11s - loss: 0.2970 - accuracy: 0.9524
 7600/25000 [========>.....................] - ETA: 11s - loss: 0.2971 - accuracy: 0.9520
 7700/25000 [========>.....................] - ETA: 11s - loss: 0.2971 - accuracy: 0.9521
 7800/25000 [========>.....................] - ETA: 11s - loss: 0.2973 - accuracy: 0.9518
 7900/25000 [========>.....................] - ETA: 11s - loss: 0.2972 - accuracy: 0.9519
 8000/25000 [========>.....................] - ETA: 11s - loss: 0.2978 - accuracy: 0.9515
 8100/25000 [========>.....................] - ETA: 11s - loss: 0.2976 - accuracy: 0.9517
 8200/25000 [========>.....................] - ETA: 11s - loss: 0.2975 - accuracy: 0.9518
 8300/25000 [========>.....................] - ETA: 11s - loss: 0.2976 - accuracy: 0.9518
 8400/25000 [=========>....................] - ETA: 11s - loss: 0.2980 - accuracy: 0.9514
 8500/25000 [=========>....................] - ETA: 10s - loss: 0.2979 - accuracy: 0.9513
 8600/25000 [=========>....................] - ETA: 10s - loss: 0.2970 - accuracy: 0.9519
 8700/25000 [=========>....................] - ETA: 10s - loss: 0.2968 - accuracy: 0.9521
 8800/25000 [=========>....................] - ETA: 10s - loss: 0.2969 - accuracy: 0.9520
 8900/25000 [=========>....................] - ETA: 10s - loss: 0.2973 - accuracy: 0.9516
 9000/25000 [=========>....................] - ETA: 10s - loss: 0.2974 - accuracy: 0.9516
 9100/25000 [=========>....................] - ETA: 10s - loss: 0.2974 - accuracy: 0.9516
 9200/25000 [==========>...................] - ETA: 10s - loss: 0.2976 - accuracy: 0.9514
 9300/25000 [==========>...................] - ETA: 10s - loss: 0.2984 - accuracy: 0.9510
 9400/25000 [==========>...................] - ETA: 10s - loss: 0.2983 - accuracy: 0.9511
 9500/25000 [==========>...................] - ETA: 10s - loss: 0.2982 - accuracy: 0.9511
 9600/25000 [==========>...................] - ETA: 10s - loss: 0.2983 - accuracy: 0.9510
 9700/25000 [==========>...................] - ETA: 10s - loss: 0.2983 - accuracy: 0.9511
 9800/25000 [==========>...................] - ETA: 10s - loss: 0.2978 - accuracy: 0.9513
 9900/25000 [==========>...................] - ETA: 10s - loss: 0.2975 - accuracy: 0.9515
10000/25000 [===========>..................] - ETA: 9s - loss: 0.2972 - accuracy: 0.9518 
10100/25000 [===========>..................] - ETA: 9s - loss: 0.2970 - accuracy: 0.9519
10200/25000 [===========>..................] - ETA: 9s - loss: 0.2969 - accuracy: 0.9520
10300/25000 [===========>..................] - ETA: 9s - loss: 0.2966 - accuracy: 0.9520
10400/25000 [===========>..................] - ETA: 9s - loss: 0.2961 - accuracy: 0.9523
10500/25000 [===========>..................] - ETA: 9s - loss: 0.2966 - accuracy: 0.9520
10600/25000 [===========>..................] - ETA: 9s - loss: 0.2962 - accuracy: 0.9523
10700/25000 [===========>..................] - ETA: 9s - loss: 0.2960 - accuracy: 0.9523
10800/25000 [===========>..................] - ETA: 9s - loss: 0.2966 - accuracy: 0.9519
10900/25000 [============>.................] - ETA: 9s - loss: 0.2969 - accuracy: 0.9517
11000/25000 [============>.................] - ETA: 9s - loss: 0.2966 - accuracy: 0.9519
11100/25000 [============>.................] - ETA: 9s - loss: 0.2971 - accuracy: 0.9514
11200/25000 [============>.................] - ETA: 9s - loss: 0.2971 - accuracy: 0.9514
11300/25000 [============>.................] - ETA: 9s - loss: 0.2971 - accuracy: 0.9515
11400/25000 [============>.................] - ETA: 8s - loss: 0.2970 - accuracy: 0.9515
11500/25000 [============>.................] - ETA: 8s - loss: 0.2968 - accuracy: 0.9517
11600/25000 [============>.................] - ETA: 8s - loss: 0.2968 - accuracy: 0.9517
11700/25000 [=============>................] - ETA: 8s - loss: 0.2966 - accuracy: 0.9519
11800/25000 [=============>................] - ETA: 8s - loss: 0.2968 - accuracy: 0.9516
11900/25000 [=============>................] - ETA: 8s - loss: 0.2968 - accuracy: 0.9516
12000/25000 [=============>................] - ETA: 8s - loss: 0.2972 - accuracy: 0.9512
12100/25000 [=============>................] - ETA: 8s - loss: 0.2973 - accuracy: 0.9511
12200/25000 [=============>................] - ETA: 8s - loss: 0.2973 - accuracy: 0.9511
12300/25000 [=============>................] - ETA: 8s - loss: 0.2973 - accuracy: 0.9510
12400/25000 [=============>................] - ETA: 8s - loss: 0.2971 - accuracy: 0.9511
12500/25000 [==============>...............] - ETA: 8s - loss: 0.2971 - accuracy: 0.9511
12600/25000 [==============>...............] - ETA: 8s - loss: 0.2970 - accuracy: 0.9510
12700/25000 [==============>...............] - ETA: 8s - loss: 0.2971 - accuracy: 0.9509
12800/25000 [==============>...............] - ETA: 8s - loss: 0.2974 - accuracy: 0.9506
12900/25000 [==============>...............] - ETA: 7s - loss: 0.2974 - accuracy: 0.9506
13000/25000 [==============>...............] - ETA: 7s - loss: 0.2974 - accuracy: 0.9506
13100/25000 [==============>...............] - ETA: 7s - loss: 0.2974 - accuracy: 0.9505
13200/25000 [==============>...............] - ETA: 7s - loss: 0.2979 - accuracy: 0.9502
13300/25000 [==============>...............] - ETA: 7s - loss: 0.2980 - accuracy: 0.9502
13400/25000 [===============>..............] - ETA: 7s - loss: 0.2978 - accuracy: 0.9502
13500/25000 [===============>..............] - ETA: 7s - loss: 0.2979 - accuracy: 0.9501
13600/25000 [===============>..............] - ETA: 7s - loss: 0.2978 - accuracy: 0.9501
13700/25000 [===============>..............] - ETA: 7s - loss: 0.2976 - accuracy: 0.9503
13800/25000 [===============>..............] - ETA: 7s - loss: 0.2974 - accuracy: 0.9504
13900/25000 [===============>..............] - ETA: 7s - loss: 0.2971 - accuracy: 0.9506
14000/25000 [===============>..............] - ETA: 7s - loss: 0.2968 - accuracy: 0.9509
14100/25000 [===============>..............] - ETA: 7s - loss: 0.2966 - accuracy: 0.9511
14200/25000 [================>.............] - ETA: 7s - loss: 0.2968 - accuracy: 0.9508
14300/25000 [================>.............] - ETA: 7s - loss: 0.2967 - accuracy: 0.9508
14400/25000 [================>.............] - ETA: 6s - loss: 0.2964 - accuracy: 0.9511
14500/25000 [================>.............] - ETA: 6s - loss: 0.2964 - accuracy: 0.9510
14600/25000 [================>.............] - ETA: 6s - loss: 0.2967 - accuracy: 0.9508
14700/25000 [================>.............] - ETA: 6s - loss: 0.2968 - accuracy: 0.9507
14800/25000 [================>.............] - ETA: 6s - loss: 0.2969 - accuracy: 0.9506
14900/25000 [================>.............] - ETA: 6s - loss: 0.2969 - accuracy: 0.9506
15000/25000 [=================>............] - ETA: 6s - loss: 0.2970 - accuracy: 0.9505
15100/25000 [=================>............] - ETA: 6s - loss: 0.2969 - accuracy: 0.9505
15200/25000 [=================>............] - ETA: 6s - loss: 0.2970 - accuracy: 0.9504
15300/25000 [=================>............] - ETA: 6s - loss: 0.2968 - accuracy: 0.9506
15400/25000 [=================>............] - ETA: 6s - loss: 0.2968 - accuracy: 0.9506
15500/25000 [=================>............] - ETA: 6s - loss: 0.2965 - accuracy: 0.9508
15600/25000 [=================>............] - ETA: 6s - loss: 0.2965 - accuracy: 0.9508
15700/25000 [=================>............] - ETA: 6s - loss: 0.2968 - accuracy: 0.9505
15800/25000 [=================>............] - ETA: 6s - loss: 0.2968 - accuracy: 0.9505
15900/25000 [==================>...........] - ETA: 5s - loss: 0.2967 - accuracy: 0.9505
16000/25000 [==================>...........] - ETA: 5s - loss: 0.2969 - accuracy: 0.9504
16100/25000 [==================>...........] - ETA: 5s - loss: 0.2967 - accuracy: 0.9506
16200/25000 [==================>...........] - ETA: 5s - loss: 0.2969 - accuracy: 0.9504
16300/25000 [==================>...........] - ETA: 5s - loss: 0.2969 - accuracy: 0.9503
16400/25000 [==================>...........] - ETA: 5s - loss: 0.2966 - accuracy: 0.9505
16500/25000 [==================>...........] - ETA: 5s - loss: 0.2967 - accuracy: 0.9504
16600/25000 [==================>...........] - ETA: 5s - loss: 0.2965 - accuracy: 0.9504
16700/25000 [===================>..........] - ETA: 5s - loss: 0.2963 - accuracy: 0.9505
16800/25000 [===================>..........] - ETA: 5s - loss: 0.2961 - accuracy: 0.9507
16900/25000 [===================>..........] - ETA: 5s - loss: 0.2961 - accuracy: 0.9506
17000/25000 [===================>..........] - ETA: 5s - loss: 0.2958 - accuracy: 0.9508
17100/25000 [===================>..........] - ETA: 5s - loss: 0.2959 - accuracy: 0.9506
17200/25000 [===================>..........] - ETA: 5s - loss: 0.2958 - accuracy: 0.9508
17300/25000 [===================>..........] - ETA: 5s - loss: 0.2956 - accuracy: 0.9509
17400/25000 [===================>..........] - ETA: 4s - loss: 0.2955 - accuracy: 0.9509
17500/25000 [====================>.........] - ETA: 4s - loss: 0.2956 - accuracy: 0.9507
17600/25000 [====================>.........] - ETA: 4s - loss: 0.2955 - accuracy: 0.9508
17700/25000 [====================>.........] - ETA: 4s - loss: 0.2959 - accuracy: 0.9505
17800/25000 [====================>.........] - ETA: 4s - loss: 0.2959 - accuracy: 0.9504
17900/25000 [====================>.........] - ETA: 4s - loss: 0.2959 - accuracy: 0.9504
18000/25000 [====================>.........] - ETA: 4s - loss: 0.2959 - accuracy: 0.9503
18100/25000 [====================>.........] - ETA: 4s - loss: 0.2960 - accuracy: 0.9502
18200/25000 [====================>.........] - ETA: 4s - loss: 0.2960 - accuracy: 0.9502
18300/25000 [====================>.........] - ETA: 4s - loss: 0.2958 - accuracy: 0.9503
18400/25000 [=====================>........] - ETA: 4s - loss: 0.2959 - accuracy: 0.9503
18500/25000 [=====================>........] - ETA: 4s - loss: 0.2963 - accuracy: 0.9499
18600/25000 [=====================>........] - ETA: 4s - loss: 0.2962 - accuracy: 0.9500
18700/25000 [=====================>........] - ETA: 4s - loss: 0.2964 - accuracy: 0.9498
18800/25000 [=====================>........] - ETA: 4s - loss: 0.2964 - accuracy: 0.9498
18900/25000 [=====================>........] - ETA: 3s - loss: 0.2965 - accuracy: 0.9496
19000/25000 [=====================>........] - ETA: 3s - loss: 0.2966 - accuracy: 0.9496
19100/25000 [=====================>........] - ETA: 3s - loss: 0.2967 - accuracy: 0.9494
19200/25000 [======================>.......] - ETA: 3s - loss: 0.2969 - accuracy: 0.9493
19300/25000 [======================>.......] - ETA: 3s - loss: 0.2970 - accuracy: 0.9492
19400/25000 [======================>.......] - ETA: 3s - loss: 0.2968 - accuracy: 0.9493
19500/25000 [======================>.......] - ETA: 3s - loss: 0.2967 - accuracy: 0.9494
19600/25000 [======================>.......] - ETA: 3s - loss: 0.2967 - accuracy: 0.9493
19700/25000 [======================>.......] - ETA: 3s - loss: 0.2967 - accuracy: 0.9492
19800/25000 [======================>.......] - ETA: 3s - loss: 0.2971 - accuracy: 0.9490
19900/25000 [======================>.......] - ETA: 3s - loss: 0.2970 - accuracy: 0.9490
20000/25000 [=======================>......] - ETA: 3s - loss: 0.2968 - accuracy: 0.9492
20100/25000 [=======================>......] - ETA: 3s - loss: 0.2969 - accuracy: 0.9491
20200/25000 [=======================>......] - ETA: 3s - loss: 0.2968 - accuracy: 0.9491
20300/25000 [=======================>......] - ETA: 3s - loss: 0.2967 - accuracy: 0.9492
20400/25000 [=======================>......] - ETA: 2s - loss: 0.2968 - accuracy: 0.9491
20500/25000 [=======================>......] - ETA: 2s - loss: 0.2968 - accuracy: 0.9491
20600/25000 [=======================>......] - ETA: 2s - loss: 0.2965 - accuracy: 0.9492
20700/25000 [=======================>......] - ETA: 2s - loss: 0.2965 - accuracy: 0.9492
20800/25000 [=======================>......] - ETA: 2s - loss: 0.2964 - accuracy: 0.9492
20900/25000 [========================>.....] - ETA: 2s - loss: 0.2962 - accuracy: 0.9493
21000/25000 [========================>.....] - ETA: 2s - loss: 0.2962 - accuracy: 0.9493
21100/25000 [========================>.....] - ETA: 2s - loss: 0.2964 - accuracy: 0.9491
21200/25000 [========================>.....] - ETA: 2s - loss: 0.2963 - accuracy: 0.9492
21300/25000 [========================>.....] - ETA: 2s - loss: 0.2964 - accuracy: 0.9491
21400/25000 [========================>.....] - ETA: 2s - loss: 0.2964 - accuracy: 0.9490
21500/25000 [========================>.....] - ETA: 2s - loss: 0.2962 - accuracy: 0.9492
21600/25000 [========================>.....] - ETA: 2s - loss: 0.2962 - accuracy: 0.9492
21700/25000 [=========================>....] - ETA: 2s - loss: 0.2961 - accuracy: 0.9493
21800/25000 [=========================>....] - ETA: 2s - loss: 0.2959 - accuracy: 0.9494
21900/25000 [=========================>....] - ETA: 2s - loss: 0.2960 - accuracy: 0.9493
22000/25000 [=========================>....] - ETA: 1s - loss: 0.2960 - accuracy: 0.9493
22100/25000 [=========================>....] - ETA: 1s - loss: 0.2958 - accuracy: 0.9494
22200/25000 [=========================>....] - ETA: 1s - loss: 0.2956 - accuracy: 0.9495
22300/25000 [=========================>....] - ETA: 1s - loss: 0.2955 - accuracy: 0.9496
22400/25000 [=========================>....] - ETA: 1s - loss: 0.2954 - accuracy: 0.9496
22500/25000 [==========================>...] - ETA: 1s - loss: 0.2956 - accuracy: 0.9494
22600/25000 [==========================>...] - ETA: 1s - loss: 0.2956 - accuracy: 0.9494
22700/25000 [==========================>...] - ETA: 1s - loss: 0.2955 - accuracy: 0.9494
22800/25000 [==========================>...] - ETA: 1s - loss: 0.2954 - accuracy: 0.9494
22900/25000 [==========================>...] - ETA: 1s - loss: 0.2954 - accuracy: 0.9493
23000/25000 [==========================>...] - ETA: 1s - loss: 0.2955 - accuracy: 0.9492
23100/25000 [==========================>...] - ETA: 1s - loss: 0.2954 - accuracy: 0.9492
23200/25000 [==========================>...] - ETA: 1s - loss: 0.2951 - accuracy: 0.9494
23300/25000 [==========================>...] - ETA: 1s - loss: 0.2950 - accuracy: 0.9494
23400/25000 [===========================>..] - ETA: 1s - loss: 0.2949 - accuracy: 0.9496
23500/25000 [===========================>..] - ETA: 0s - loss: 0.2948 - accuracy: 0.9496
23600/25000 [===========================>..] - ETA: 0s - loss: 0.2950 - accuracy: 0.9494
23700/25000 [===========================>..] - ETA: 0s - loss: 0.2948 - accuracy: 0.9495
23800/25000 [===========================>..] - ETA: 0s - loss: 0.2946 - accuracy: 0.9497
23900/25000 [===========================>..] - ETA: 0s - loss: 0.2944 - accuracy: 0.9498
24000/25000 [===========================>..] - ETA: 0s - loss: 0.2944 - accuracy: 0.9498
24100/25000 [===========================>..] - ETA: 0s - loss: 0.2943 - accuracy: 0.9498
24200/25000 [============================>.] - ETA: 0s - loss: 0.2943 - accuracy: 0.9498
24300/25000 [============================>.] - ETA: 0s - loss: 0.2945 - accuracy: 0.9497
24400/25000 [============================>.] - ETA: 0s - loss: 0.2944 - accuracy: 0.9498
24500/25000 [============================>.] - ETA: 0s - loss: 0.2945 - accuracy: 0.9497
24600/25000 [============================>.] - ETA: 0s - loss: 0.2945 - accuracy: 0.9497
24700/25000 [============================>.] - ETA: 0s - loss: 0.2946 - accuracy: 0.9496
24800/25000 [============================>.] - ETA: 0s - loss: 0.2946 - accuracy: 0.9496
24900/25000 [============================>.] - ETA: 0s - loss: 0.2945 - accuracy: 0.9496
25000/25000 [==============================] - 21s 825us/step - loss: 0.2945 - accuracy: 0.9496 - val_loss: 0.4090 - val_accuracy: 0.8580
Epoch 9/10

  100/25000 [..............................] - ETA: 16s - loss: 0.2521 - accuracy: 0.9700
  200/25000 [..............................] - ETA: 15s - loss: 0.2784 - accuracy: 0.9550
  300/25000 [..............................] - ETA: 15s - loss: 0.2845 - accuracy: 0.9500
  400/25000 [..............................] - ETA: 15s - loss: 0.2937 - accuracy: 0.9450
  500/25000 [..............................] - ETA: 15s - loss: 0.2893 - accuracy: 0.9460
  600/25000 [..............................] - ETA: 15s - loss: 0.2907 - accuracy: 0.9450
  700/25000 [..............................] - ETA: 15s - loss: 0.2841 - accuracy: 0.9500
  800/25000 [..............................] - ETA: 15s - loss: 0.2793 - accuracy: 0.9538
  900/25000 [>.............................] - ETA: 15s - loss: 0.2748 - accuracy: 0.9567
 1000/25000 [>.............................] - ETA: 15s - loss: 0.2808 - accuracy: 0.9520
 1100/25000 [>.............................] - ETA: 15s - loss: 0.2758 - accuracy: 0.9555
 1200/25000 [>.............................] - ETA: 15s - loss: 0.2735 - accuracy: 0.9575
 1300/25000 [>.............................] - ETA: 15s - loss: 0.2761 - accuracy: 0.9554
 1400/25000 [>.............................] - ETA: 14s - loss: 0.2767 - accuracy: 0.9550
 1500/25000 [>.............................] - ETA: 14s - loss: 0.2775 - accuracy: 0.9547
 1600/25000 [>.............................] - ETA: 14s - loss: 0.2808 - accuracy: 0.9525
 1700/25000 [=>............................] - ETA: 14s - loss: 0.2799 - accuracy: 0.9529
 1800/25000 [=>............................] - ETA: 14s - loss: 0.2776 - accuracy: 0.9550
 1900/25000 [=>............................] - ETA: 14s - loss: 0.2779 - accuracy: 0.9547
 2000/25000 [=>............................] - ETA: 14s - loss: 0.2777 - accuracy: 0.9550
 2100/25000 [=>............................] - ETA: 14s - loss: 0.2758 - accuracy: 0.9562
 2200/25000 [=>............................] - ETA: 14s - loss: 0.2769 - accuracy: 0.9555
 2300/25000 [=>............................] - ETA: 14s - loss: 0.2770 - accuracy: 0.9557
 2400/25000 [=>............................] - ETA: 14s - loss: 0.2768 - accuracy: 0.9554
 2500/25000 [==>...........................] - ETA: 14s - loss: 0.2757 - accuracy: 0.9560
 2600/25000 [==>...........................] - ETA: 14s - loss: 0.2758 - accuracy: 0.9558
 2700/25000 [==>...........................] - ETA: 14s - loss: 0.2748 - accuracy: 0.9563
 2800/25000 [==>...........................] - ETA: 14s - loss: 0.2738 - accuracy: 0.9571
 2900/25000 [==>...........................] - ETA: 13s - loss: 0.2733 - accuracy: 0.9576
 3000/25000 [==>...........................] - ETA: 13s - loss: 0.2720 - accuracy: 0.9587
 3100/25000 [==>...........................] - ETA: 13s - loss: 0.2727 - accuracy: 0.9581
 3200/25000 [==>...........................] - ETA: 13s - loss: 0.2735 - accuracy: 0.9575
 3300/25000 [==>...........................] - ETA: 13s - loss: 0.2740 - accuracy: 0.9573
 3400/25000 [===>..........................] - ETA: 13s - loss: 0.2739 - accuracy: 0.9574
 3500/25000 [===>..........................] - ETA: 13s - loss: 0.2738 - accuracy: 0.9577
 3600/25000 [===>..........................] - ETA: 13s - loss: 0.2743 - accuracy: 0.9575
 3700/25000 [===>..........................] - ETA: 13s - loss: 0.2742 - accuracy: 0.9573
 3800/25000 [===>..........................] - ETA: 13s - loss: 0.2744 - accuracy: 0.9571
 3900/25000 [===>..........................] - ETA: 13s - loss: 0.2747 - accuracy: 0.9569
 4000/25000 [===>..........................] - ETA: 13s - loss: 0.2741 - accuracy: 0.9572
 4100/25000 [===>..........................] - ETA: 13s - loss: 0.2734 - accuracy: 0.9578
 4200/25000 [====>.........................] - ETA: 13s - loss: 0.2736 - accuracy: 0.9576
 4300/25000 [====>.........................] - ETA: 13s - loss: 0.2742 - accuracy: 0.9572
 4400/25000 [====>.........................] - ETA: 13s - loss: 0.2740 - accuracy: 0.9573
 4500/25000 [====>.........................] - ETA: 12s - loss: 0.2741 - accuracy: 0.9571
 4600/25000 [====>.........................] - ETA: 12s - loss: 0.2750 - accuracy: 0.9565
 4700/25000 [====>.........................] - ETA: 12s - loss: 0.2746 - accuracy: 0.9568
 4800/25000 [====>.........................] - ETA: 12s - loss: 0.2742 - accuracy: 0.9571
 4900/25000 [====>.........................] - ETA: 12s - loss: 0.2741 - accuracy: 0.9571
 5000/25000 [=====>........................] - ETA: 12s - loss: 0.2742 - accuracy: 0.9572
 5100/25000 [=====>........................] - ETA: 12s - loss: 0.2741 - accuracy: 0.9573
 5200/25000 [=====>........................] - ETA: 12s - loss: 0.2744 - accuracy: 0.9571
 5300/25000 [=====>........................] - ETA: 12s - loss: 0.2736 - accuracy: 0.9577
 5400/25000 [=====>........................] - ETA: 12s - loss: 0.2731 - accuracy: 0.9581
 5500/25000 [=====>........................] - ETA: 12s - loss: 0.2730 - accuracy: 0.9582
 5600/25000 [=====>........................] - ETA: 12s - loss: 0.2722 - accuracy: 0.9588
 5700/25000 [=====>........................] - ETA: 12s - loss: 0.2714 - accuracy: 0.9593
 5800/25000 [=====>........................] - ETA: 12s - loss: 0.2720 - accuracy: 0.9590
 5900/25000 [======>.......................] - ETA: 12s - loss: 0.2727 - accuracy: 0.9585
 6000/25000 [======>.......................] - ETA: 12s - loss: 0.2729 - accuracy: 0.9583
 6100/25000 [======>.......................] - ETA: 11s - loss: 0.2725 - accuracy: 0.9587
 6200/25000 [======>.......................] - ETA: 11s - loss: 0.2720 - accuracy: 0.9592
 6300/25000 [======>.......................] - ETA: 11s - loss: 0.2721 - accuracy: 0.9590
 6400/25000 [======>.......................] - ETA: 11s - loss: 0.2721 - accuracy: 0.9591
 6500/25000 [======>.......................] - ETA: 11s - loss: 0.2715 - accuracy: 0.9594
 6600/25000 [======>.......................] - ETA: 11s - loss: 0.2714 - accuracy: 0.9595
 6700/25000 [=======>......................] - ETA: 11s - loss: 0.2720 - accuracy: 0.9591
 6800/25000 [=======>......................] - ETA: 11s - loss: 0.2718 - accuracy: 0.9593
 6900/25000 [=======>......................] - ETA: 11s - loss: 0.2728 - accuracy: 0.9586
 7000/25000 [=======>......................] - ETA: 11s - loss: 0.2727 - accuracy: 0.9586
 7100/25000 [=======>......................] - ETA: 11s - loss: 0.2725 - accuracy: 0.9587
 7200/25000 [=======>......................] - ETA: 11s - loss: 0.2734 - accuracy: 0.9581
 7300/25000 [=======>......................] - ETA: 11s - loss: 0.2732 - accuracy: 0.9582
 7400/25000 [=======>......................] - ETA: 11s - loss: 0.2730 - accuracy: 0.9584
 7500/25000 [========>.....................] - ETA: 11s - loss: 0.2734 - accuracy: 0.9580
 7600/25000 [========>.....................] - ETA: 11s - loss: 0.2740 - accuracy: 0.9575
 7700/25000 [========>.....................] - ETA: 11s - loss: 0.2737 - accuracy: 0.9577
 7800/25000 [========>.....................] - ETA: 10s - loss: 0.2741 - accuracy: 0.9576
 7900/25000 [========>.....................] - ETA: 10s - loss: 0.2738 - accuracy: 0.9577
 8000/25000 [========>.....................] - ETA: 10s - loss: 0.2735 - accuracy: 0.9578
 8100/25000 [========>.....................] - ETA: 10s - loss: 0.2742 - accuracy: 0.9573
 8200/25000 [========>.....................] - ETA: 10s - loss: 0.2738 - accuracy: 0.9576
 8300/25000 [========>.....................] - ETA: 10s - loss: 0.2739 - accuracy: 0.9575
 8400/25000 [=========>....................] - ETA: 10s - loss: 0.2737 - accuracy: 0.9576
 8500/25000 [=========>....................] - ETA: 10s - loss: 0.2735 - accuracy: 0.9578
 8600/25000 [=========>....................] - ETA: 10s - loss: 0.2732 - accuracy: 0.9579
 8700/25000 [=========>....................] - ETA: 10s - loss: 0.2734 - accuracy: 0.9578
 8800/25000 [=========>....................] - ETA: 10s - loss: 0.2728 - accuracy: 0.9583
 8900/25000 [=========>....................] - ETA: 10s - loss: 0.2729 - accuracy: 0.9581
 9000/25000 [=========>....................] - ETA: 10s - loss: 0.2726 - accuracy: 0.9583
 9100/25000 [=========>....................] - ETA: 10s - loss: 0.2725 - accuracy: 0.9585
 9200/25000 [==========>...................] - ETA: 10s - loss: 0.2727 - accuracy: 0.9583
 9300/25000 [==========>...................] - ETA: 9s - loss: 0.2727 - accuracy: 0.9583 
 9400/25000 [==========>...................] - ETA: 9s - loss: 0.2724 - accuracy: 0.9585
 9500/25000 [==========>...................] - ETA: 9s - loss: 0.2723 - accuracy: 0.9585
 9600/25000 [==========>...................] - ETA: 9s - loss: 0.2726 - accuracy: 0.9583
 9700/25000 [==========>...................] - ETA: 9s - loss: 0.2726 - accuracy: 0.9582
 9800/25000 [==========>...................] - ETA: 9s - loss: 0.2728 - accuracy: 0.9581
 9900/25000 [==========>...................] - ETA: 9s - loss: 0.2728 - accuracy: 0.9580
10000/25000 [===========>..................] - ETA: 9s - loss: 0.2725 - accuracy: 0.9582
10100/25000 [===========>..................] - ETA: 9s - loss: 0.2726 - accuracy: 0.9581
10200/25000 [===========>..................] - ETA: 9s - loss: 0.2723 - accuracy: 0.9583
10300/25000 [===========>..................] - ETA: 9s - loss: 0.2723 - accuracy: 0.9582
10400/25000 [===========>..................] - ETA: 9s - loss: 0.2723 - accuracy: 0.9582
10500/25000 [===========>..................] - ETA: 9s - loss: 0.2719 - accuracy: 0.9585
10600/25000 [===========>..................] - ETA: 9s - loss: 0.2720 - accuracy: 0.9583
10700/25000 [===========>..................] - ETA: 9s - loss: 0.2723 - accuracy: 0.9580
10800/25000 [===========>..................] - ETA: 9s - loss: 0.2724 - accuracy: 0.9579
10900/25000 [============>.................] - ETA: 8s - loss: 0.2724 - accuracy: 0.9578
11000/25000 [============>.................] - ETA: 8s - loss: 0.2719 - accuracy: 0.9581
11100/25000 [============>.................] - ETA: 8s - loss: 0.2719 - accuracy: 0.9581
11200/25000 [============>.................] - ETA: 8s - loss: 0.2719 - accuracy: 0.9580
11300/25000 [============>.................] - ETA: 8s - loss: 0.2719 - accuracy: 0.9579
11400/25000 [============>.................] - ETA: 8s - loss: 0.2721 - accuracy: 0.9578
11500/25000 [============>.................] - ETA: 8s - loss: 0.2721 - accuracy: 0.9578
11600/25000 [============>.................] - ETA: 8s - loss: 0.2719 - accuracy: 0.9579
11700/25000 [=============>................] - ETA: 8s - loss: 0.2718 - accuracy: 0.9579
11800/25000 [=============>................] - ETA: 8s - loss: 0.2712 - accuracy: 0.9583
11900/25000 [=============>................] - ETA: 8s - loss: 0.2711 - accuracy: 0.9583
12000/25000 [=============>................] - ETA: 8s - loss: 0.2709 - accuracy: 0.9584
12100/25000 [=============>................] - ETA: 8s - loss: 0.2712 - accuracy: 0.9582
12200/25000 [=============>................] - ETA: 8s - loss: 0.2713 - accuracy: 0.9580
12300/25000 [=============>................] - ETA: 8s - loss: 0.2715 - accuracy: 0.9579
12400/25000 [=============>................] - ETA: 8s - loss: 0.2713 - accuracy: 0.9581
12500/25000 [==============>...............] - ETA: 7s - loss: 0.2711 - accuracy: 0.9582
12600/25000 [==============>...............] - ETA: 7s - loss: 0.2711 - accuracy: 0.9582
12700/25000 [==============>...............] - ETA: 7s - loss: 0.2712 - accuracy: 0.9581
12800/25000 [==============>...............] - ETA: 7s - loss: 0.2714 - accuracy: 0.9580
12900/25000 [==============>...............] - ETA: 7s - loss: 0.2717 - accuracy: 0.9578
13000/25000 [==============>...............] - ETA: 7s - loss: 0.2715 - accuracy: 0.9578
13100/25000 [==============>...............] - ETA: 7s - loss: 0.2713 - accuracy: 0.9579
13200/25000 [==============>...............] - ETA: 7s - loss: 0.2713 - accuracy: 0.9579
13300/25000 [==============>...............] - ETA: 7s - loss: 0.2713 - accuracy: 0.9578
13400/25000 [===============>..............] - ETA: 7s - loss: 0.2711 - accuracy: 0.9579
13500/25000 [===============>..............] - ETA: 7s - loss: 0.2709 - accuracy: 0.9580
13600/25000 [===============>..............] - ETA: 7s - loss: 0.2711 - accuracy: 0.9578
13700/25000 [===============>..............] - ETA: 7s - loss: 0.2710 - accuracy: 0.9578
13800/25000 [===============>..............] - ETA: 7s - loss: 0.2711 - accuracy: 0.9577
13900/25000 [===============>..............] - ETA: 7s - loss: 0.2711 - accuracy: 0.9576
14000/25000 [===============>..............] - ETA: 6s - loss: 0.2710 - accuracy: 0.9576
14100/25000 [===============>..............] - ETA: 6s - loss: 0.2710 - accuracy: 0.9576
14200/25000 [================>.............] - ETA: 6s - loss: 0.2708 - accuracy: 0.9577
14300/25000 [================>.............] - ETA: 6s - loss: 0.2709 - accuracy: 0.9576
14400/25000 [================>.............] - ETA: 6s - loss: 0.2712 - accuracy: 0.9574
14500/25000 [================>.............] - ETA: 6s - loss: 0.2710 - accuracy: 0.9574
14600/25000 [================>.............] - ETA: 6s - loss: 0.2711 - accuracy: 0.9574
14700/25000 [================>.............] - ETA: 6s - loss: 0.2710 - accuracy: 0.9574
14800/25000 [================>.............] - ETA: 6s - loss: 0.2710 - accuracy: 0.9573
14900/25000 [================>.............] - ETA: 6s - loss: 0.2710 - accuracy: 0.9573
15000/25000 [=================>............] - ETA: 6s - loss: 0.2712 - accuracy: 0.9571
15100/25000 [=================>............] - ETA: 6s - loss: 0.2708 - accuracy: 0.9574
15200/25000 [=================>............] - ETA: 6s - loss: 0.2707 - accuracy: 0.9574
15300/25000 [=================>............] - ETA: 6s - loss: 0.2707 - accuracy: 0.9573
15400/25000 [=================>............] - ETA: 6s - loss: 0.2705 - accuracy: 0.9574
15500/25000 [=================>............] - ETA: 6s - loss: 0.2706 - accuracy: 0.9574
15600/25000 [=================>............] - ETA: 5s - loss: 0.2706 - accuracy: 0.9574
15700/25000 [=================>............] - ETA: 5s - loss: 0.2704 - accuracy: 0.9575
15800/25000 [=================>............] - ETA: 5s - loss: 0.2706 - accuracy: 0.9573
15900/25000 [==================>...........] - ETA: 5s - loss: 0.2709 - accuracy: 0.9572
16000/25000 [==================>...........] - ETA: 5s - loss: 0.2707 - accuracy: 0.9572
16100/25000 [==================>...........] - ETA: 5s - loss: 0.2709 - accuracy: 0.9571
16200/25000 [==================>...........] - ETA: 5s - loss: 0.2712 - accuracy: 0.9570
16300/25000 [==================>...........] - ETA: 5s - loss: 0.2712 - accuracy: 0.9569
16400/25000 [==================>...........] - ETA: 5s - loss: 0.2711 - accuracy: 0.9570
16500/25000 [==================>...........] - ETA: 5s - loss: 0.2712 - accuracy: 0.9568
16600/25000 [==================>...........] - ETA: 5s - loss: 0.2710 - accuracy: 0.9569
16700/25000 [===================>..........] - ETA: 5s - loss: 0.2712 - accuracy: 0.9567
16800/25000 [===================>..........] - ETA: 5s - loss: 0.2713 - accuracy: 0.9566
16900/25000 [===================>..........] - ETA: 5s - loss: 0.2712 - accuracy: 0.9566
17000/25000 [===================>..........] - ETA: 5s - loss: 0.2712 - accuracy: 0.9566
17100/25000 [===================>..........] - ETA: 5s - loss: 0.2716 - accuracy: 0.9564
17200/25000 [===================>..........] - ETA: 4s - loss: 0.2712 - accuracy: 0.9566
17300/25000 [===================>..........] - ETA: 4s - loss: 0.2711 - accuracy: 0.9566
17400/25000 [===================>..........] - ETA: 4s - loss: 0.2711 - accuracy: 0.9567
17500/25000 [====================>.........] - ETA: 4s - loss: 0.2712 - accuracy: 0.9566
17600/25000 [====================>.........] - ETA: 4s - loss: 0.2712 - accuracy: 0.9565
17700/25000 [====================>.........] - ETA: 4s - loss: 0.2711 - accuracy: 0.9566
17800/25000 [====================>.........] - ETA: 4s - loss: 0.2713 - accuracy: 0.9563
17900/25000 [====================>.........] - ETA: 4s - loss: 0.2713 - accuracy: 0.9564
18000/25000 [====================>.........] - ETA: 4s - loss: 0.2712 - accuracy: 0.9564
18100/25000 [====================>.........] - ETA: 4s - loss: 0.2713 - accuracy: 0.9563
18200/25000 [====================>.........] - ETA: 4s - loss: 0.2713 - accuracy: 0.9563
18300/25000 [====================>.........] - ETA: 4s - loss: 0.2714 - accuracy: 0.9562
18400/25000 [=====================>........] - ETA: 4s - loss: 0.2711 - accuracy: 0.9564
18500/25000 [=====================>........] - ETA: 4s - loss: 0.2710 - accuracy: 0.9565
18600/25000 [=====================>........] - ETA: 4s - loss: 0.2709 - accuracy: 0.9566
18700/25000 [=====================>........] - ETA: 4s - loss: 0.2706 - accuracy: 0.9567
18800/25000 [=====================>........] - ETA: 3s - loss: 0.2706 - accuracy: 0.9568
18900/25000 [=====================>........] - ETA: 3s - loss: 0.2704 - accuracy: 0.9568
19000/25000 [=====================>........] - ETA: 3s - loss: 0.2706 - accuracy: 0.9567
19100/25000 [=====================>........] - ETA: 3s - loss: 0.2704 - accuracy: 0.9568
19200/25000 [======================>.......] - ETA: 3s - loss: 0.2705 - accuracy: 0.9566
19300/25000 [======================>.......] - ETA: 3s - loss: 0.2703 - accuracy: 0.9568
19400/25000 [======================>.......] - ETA: 3s - loss: 0.2701 - accuracy: 0.9568
19500/25000 [======================>.......] - ETA: 3s - loss: 0.2703 - accuracy: 0.9567
19600/25000 [======================>.......] - ETA: 3s - loss: 0.2703 - accuracy: 0.9567
19700/25000 [======================>.......] - ETA: 3s - loss: 0.2705 - accuracy: 0.9565
19800/25000 [======================>.......] - ETA: 3s - loss: 0.2708 - accuracy: 0.9564
19900/25000 [======================>.......] - ETA: 3s - loss: 0.2707 - accuracy: 0.9564
20000/25000 [=======================>......] - ETA: 3s - loss: 0.2708 - accuracy: 0.9564
20100/25000 [=======================>......] - ETA: 3s - loss: 0.2707 - accuracy: 0.9564
20200/25000 [=======================>......] - ETA: 3s - loss: 0.2707 - accuracy: 0.9564
20300/25000 [=======================>......] - ETA: 3s - loss: 0.2707 - accuracy: 0.9564
20400/25000 [=======================>......] - ETA: 2s - loss: 0.2705 - accuracy: 0.9565
20500/25000 [=======================>......] - ETA: 2s - loss: 0.2705 - accuracy: 0.9565
20600/25000 [=======================>......] - ETA: 2s - loss: 0.2703 - accuracy: 0.9566
20700/25000 [=======================>......] - ETA: 2s - loss: 0.2704 - accuracy: 0.9566
20800/25000 [=======================>......] - ETA: 2s - loss: 0.2703 - accuracy: 0.9566
20900/25000 [========================>.....] - ETA: 2s - loss: 0.2704 - accuracy: 0.9566
21000/25000 [========================>.....] - ETA: 2s - loss: 0.2703 - accuracy: 0.9566
21100/25000 [========================>.....] - ETA: 2s - loss: 0.2701 - accuracy: 0.9567
21200/25000 [========================>.....] - ETA: 2s - loss: 0.2699 - accuracy: 0.9568
21300/25000 [========================>.....] - ETA: 2s - loss: 0.2698 - accuracy: 0.9568
21400/25000 [========================>.....] - ETA: 2s - loss: 0.2699 - accuracy: 0.9567
21500/25000 [========================>.....] - ETA: 2s - loss: 0.2699 - accuracy: 0.9567
21600/25000 [========================>.....] - ETA: 2s - loss: 0.2698 - accuracy: 0.9568
21700/25000 [=========================>....] - ETA: 2s - loss: 0.2698 - accuracy: 0.9568
21800/25000 [=========================>....] - ETA: 2s - loss: 0.2699 - accuracy: 0.9567
21900/25000 [=========================>....] - ETA: 1s - loss: 0.2697 - accuracy: 0.9568
22000/25000 [=========================>....] - ETA: 1s - loss: 0.2696 - accuracy: 0.9569
22100/25000 [=========================>....] - ETA: 1s - loss: 0.2696 - accuracy: 0.9568
22200/25000 [=========================>....] - ETA: 1s - loss: 0.2696 - accuracy: 0.9568
22300/25000 [=========================>....] - ETA: 1s - loss: 0.2697 - accuracy: 0.9568
22400/25000 [=========================>....] - ETA: 1s - loss: 0.2697 - accuracy: 0.9567
22500/25000 [==========================>...] - ETA: 1s - loss: 0.2697 - accuracy: 0.9567
22600/25000 [==========================>...] - ETA: 1s - loss: 0.2697 - accuracy: 0.9567
22700/25000 [==========================>...] - ETA: 1s - loss: 0.2696 - accuracy: 0.9568
22800/25000 [==========================>...] - ETA: 1s - loss: 0.2696 - accuracy: 0.9568
22900/25000 [==========================>...] - ETA: 1s - loss: 0.2694 - accuracy: 0.9569
23000/25000 [==========================>...] - ETA: 1s - loss: 0.2694 - accuracy: 0.9569
23100/25000 [==========================>...] - ETA: 1s - loss: 0.2694 - accuracy: 0.9568
23200/25000 [==========================>...] - ETA: 1s - loss: 0.2693 - accuracy: 0.9569
23300/25000 [==========================>...] - ETA: 1s - loss: 0.2690 - accuracy: 0.9571
23400/25000 [===========================>..] - ETA: 1s - loss: 0.2690 - accuracy: 0.9570
23500/25000 [===========================>..] - ETA: 0s - loss: 0.2693 - accuracy: 0.9569
23600/25000 [===========================>..] - ETA: 0s - loss: 0.2692 - accuracy: 0.9569
23700/25000 [===========================>..] - ETA: 0s - loss: 0.2696 - accuracy: 0.9566
23800/25000 [===========================>..] - ETA: 0s - loss: 0.2698 - accuracy: 0.9564
23900/25000 [===========================>..] - ETA: 0s - loss: 0.2696 - accuracy: 0.9565
24000/25000 [===========================>..] - ETA: 0s - loss: 0.2695 - accuracy: 0.9565
24100/25000 [===========================>..] - ETA: 0s - loss: 0.2695 - accuracy: 0.9565
24200/25000 [============================>.] - ETA: 0s - loss: 0.2694 - accuracy: 0.9566
24300/25000 [============================>.] - ETA: 0s - loss: 0.2693 - accuracy: 0.9566
24400/25000 [============================>.] - ETA: 0s - loss: 0.2692 - accuracy: 0.9566
24500/25000 [============================>.] - ETA: 0s - loss: 0.2692 - accuracy: 0.9567
24600/25000 [============================>.] - ETA: 0s - loss: 0.2695 - accuracy: 0.9564
24700/25000 [============================>.] - ETA: 0s - loss: 0.2696 - accuracy: 0.9563
24800/25000 [============================>.] - ETA: 0s - loss: 0.2694 - accuracy: 0.9565
24900/25000 [============================>.] - ETA: 0s - loss: 0.2696 - accuracy: 0.9563
25000/25000 [==============================] - 20s 807us/step - loss: 0.2696 - accuracy: 0.9564 - val_loss: 0.4090 - val_accuracy: 0.8562
Epoch 10/10

  100/25000 [..............................] - ETA: 15s - loss: 0.2148 - accuracy: 0.9900
  200/25000 [..............................] - ETA: 15s - loss: 0.2632 - accuracy: 0.9550
  300/25000 [..............................] - ETA: 15s - loss: 0.2565 - accuracy: 0.9600
  400/25000 [..............................] - ETA: 15s - loss: 0.2542 - accuracy: 0.9625
  500/25000 [..............................] - ETA: 15s - loss: 0.2524 - accuracy: 0.9640
  600/25000 [..............................] - ETA: 15s - loss: 0.2554 - accuracy: 0.9617
  700/25000 [..............................] - ETA: 15s - loss: 0.2509 - accuracy: 0.9643
  800/25000 [..............................] - ETA: 15s - loss: 0.2477 - accuracy: 0.9663
  900/25000 [>.............................] - ETA: 14s - loss: 0.2515 - accuracy: 0.9644
 1000/25000 [>.............................] - ETA: 14s - loss: 0.2505 - accuracy: 0.9650
 1100/25000 [>.............................] - ETA: 14s - loss: 0.2540 - accuracy: 0.9627
 1200/25000 [>.............................] - ETA: 14s - loss: 0.2567 - accuracy: 0.9608
 1300/25000 [>.............................] - ETA: 14s - loss: 0.2553 - accuracy: 0.9615
 1400/25000 [>.............................] - ETA: 14s - loss: 0.2587 - accuracy: 0.9593
 1500/25000 [>.............................] - ETA: 14s - loss: 0.2586 - accuracy: 0.9593
 1600/25000 [>.............................] - ETA: 14s - loss: 0.2595 - accuracy: 0.9588
 1700/25000 [=>............................] - ETA: 14s - loss: 0.2589 - accuracy: 0.9594
 1800/25000 [=>............................] - ETA: 14s - loss: 0.2581 - accuracy: 0.9600
 1900/25000 [=>............................] - ETA: 14s - loss: 0.2571 - accuracy: 0.9605
 2000/25000 [=>............................] - ETA: 14s - loss: 0.2566 - accuracy: 0.9610
 2100/25000 [=>............................] - ETA: 14s - loss: 0.2563 - accuracy: 0.9614
 2200/25000 [=>............................] - ETA: 14s - loss: 0.2550 - accuracy: 0.9623
 2300/25000 [=>............................] - ETA: 14s - loss: 0.2544 - accuracy: 0.9626
 2400/25000 [=>............................] - ETA: 14s - loss: 0.2547 - accuracy: 0.9625
 2500/25000 [==>...........................] - ETA: 13s - loss: 0.2543 - accuracy: 0.9628
 2600/25000 [==>...........................] - ETA: 13s - loss: 0.2529 - accuracy: 0.9638
 2700/25000 [==>...........................] - ETA: 13s - loss: 0.2515 - accuracy: 0.9648
 2800/25000 [==>...........................] - ETA: 13s - loss: 0.2518 - accuracy: 0.9646
 2900/25000 [==>...........................] - ETA: 13s - loss: 0.2512 - accuracy: 0.9652
 3000/25000 [==>...........................] - ETA: 13s - loss: 0.2509 - accuracy: 0.9653
 3100/25000 [==>...........................] - ETA: 13s - loss: 0.2518 - accuracy: 0.9648
 3200/25000 [==>...........................] - ETA: 13s - loss: 0.2527 - accuracy: 0.9641
 3300/25000 [==>...........................] - ETA: 13s - loss: 0.2530 - accuracy: 0.9639
 3400/25000 [===>..........................] - ETA: 13s - loss: 0.2523 - accuracy: 0.9641
 3500/25000 [===>..........................] - ETA: 13s - loss: 0.2522 - accuracy: 0.9643
 3600/25000 [===>..........................] - ETA: 13s - loss: 0.2506 - accuracy: 0.9653
 3700/25000 [===>..........................] - ETA: 13s - loss: 0.2516 - accuracy: 0.9646
 3800/25000 [===>..........................] - ETA: 13s - loss: 0.2514 - accuracy: 0.9645
 3900/25000 [===>..........................] - ETA: 13s - loss: 0.2516 - accuracy: 0.9644
 4000/25000 [===>..........................] - ETA: 13s - loss: 0.2526 - accuracy: 0.9638
 4100/25000 [===>..........................] - ETA: 12s - loss: 0.2512 - accuracy: 0.9646
 4200/25000 [====>.........................] - ETA: 12s - loss: 0.2504 - accuracy: 0.9650
 4300/25000 [====>.........................] - ETA: 12s - loss: 0.2504 - accuracy: 0.9647
 4400/25000 [====>.........................] - ETA: 12s - loss: 0.2500 - accuracy: 0.9650
 4500/25000 [====>.........................] - ETA: 12s - loss: 0.2521 - accuracy: 0.9636
 4600/25000 [====>.........................] - ETA: 12s - loss: 0.2530 - accuracy: 0.9628
 4700/25000 [====>.........................] - ETA: 12s - loss: 0.2529 - accuracy: 0.9628
 4800/25000 [====>.........................] - ETA: 12s - loss: 0.2530 - accuracy: 0.9627
 4900/25000 [====>.........................] - ETA: 12s - loss: 0.2542 - accuracy: 0.9620
 5000/25000 [=====>........................] - ETA: 12s - loss: 0.2535 - accuracy: 0.9624
 5100/25000 [=====>........................] - ETA: 12s - loss: 0.2534 - accuracy: 0.9624
 5200/25000 [=====>........................] - ETA: 12s - loss: 0.2537 - accuracy: 0.9619
 5300/25000 [=====>........................] - ETA: 12s - loss: 0.2532 - accuracy: 0.9623
 5400/25000 [=====>........................] - ETA: 12s - loss: 0.2534 - accuracy: 0.9619
 5500/25000 [=====>........................] - ETA: 12s - loss: 0.2536 - accuracy: 0.9616
 5600/25000 [=====>........................] - ETA: 12s - loss: 0.2534 - accuracy: 0.9618
 5700/25000 [=====>........................] - ETA: 11s - loss: 0.2541 - accuracy: 0.9614
 5800/25000 [=====>........................] - ETA: 11s - loss: 0.2536 - accuracy: 0.9617
 5900/25000 [======>.......................] - ETA: 11s - loss: 0.2537 - accuracy: 0.9615
 6000/25000 [======>.......................] - ETA: 11s - loss: 0.2543 - accuracy: 0.9612
 6100/25000 [======>.......................] - ETA: 11s - loss: 0.2543 - accuracy: 0.9610
 6200/25000 [======>.......................] - ETA: 11s - loss: 0.2538 - accuracy: 0.9613
 6300/25000 [======>.......................] - ETA: 11s - loss: 0.2545 - accuracy: 0.9610
 6400/25000 [======>.......................] - ETA: 11s - loss: 0.2548 - accuracy: 0.9608
 6500/25000 [======>.......................] - ETA: 11s - loss: 0.2544 - accuracy: 0.9609
 6600/25000 [======>.......................] - ETA: 11s - loss: 0.2542 - accuracy: 0.9611
 6700/25000 [=======>......................] - ETA: 11s - loss: 0.2541 - accuracy: 0.9610
 6800/25000 [=======>......................] - ETA: 11s - loss: 0.2540 - accuracy: 0.9610
 6900/25000 [=======>......................] - ETA: 11s - loss: 0.2535 - accuracy: 0.9613
 7000/25000 [=======>......................] - ETA: 11s - loss: 0.2538 - accuracy: 0.9610
 7100/25000 [=======>......................] - ETA: 11s - loss: 0.2546 - accuracy: 0.9604
 7200/25000 [=======>......................] - ETA: 11s - loss: 0.2553 - accuracy: 0.9600
 7300/25000 [=======>......................] - ETA: 11s - loss: 0.2553 - accuracy: 0.9599
 7400/25000 [=======>......................] - ETA: 10s - loss: 0.2554 - accuracy: 0.9599
 7500/25000 [========>.....................] - ETA: 10s - loss: 0.2557 - accuracy: 0.9596
 7600/25000 [========>.....................] - ETA: 10s - loss: 0.2557 - accuracy: 0.9596
 7700/25000 [========>.....................] - ETA: 10s - loss: 0.2555 - accuracy: 0.9597
 7800/25000 [========>.....................] - ETA: 10s - loss: 0.2552 - accuracy: 0.9599
 7900/25000 [========>.....................] - ETA: 10s - loss: 0.2546 - accuracy: 0.9603
 8000/25000 [========>.....................] - ETA: 10s - loss: 0.2548 - accuracy: 0.9601
 8100/25000 [========>.....................] - ETA: 10s - loss: 0.2546 - accuracy: 0.9602
 8200/25000 [========>.....................] - ETA: 10s - loss: 0.2548 - accuracy: 0.9601
 8300/25000 [========>.....................] - ETA: 10s - loss: 0.2548 - accuracy: 0.9599
 8400/25000 [=========>....................] - ETA: 10s - loss: 0.2554 - accuracy: 0.9595
 8500/25000 [=========>....................] - ETA: 10s - loss: 0.2555 - accuracy: 0.9593
 8600/25000 [=========>....................] - ETA: 10s - loss: 0.2556 - accuracy: 0.9592
 8700/25000 [=========>....................] - ETA: 10s - loss: 0.2551 - accuracy: 0.9595
 8800/25000 [=========>....................] - ETA: 10s - loss: 0.2553 - accuracy: 0.9594
 8900/25000 [=========>....................] - ETA: 10s - loss: 0.2555 - accuracy: 0.9593
 9000/25000 [=========>....................] - ETA: 10s - loss: 0.2549 - accuracy: 0.9597
 9100/25000 [=========>....................] - ETA: 10s - loss: 0.2552 - accuracy: 0.9596
 9200/25000 [==========>...................] - ETA: 10s - loss: 0.2547 - accuracy: 0.9599
 9300/25000 [==========>...................] - ETA: 9s - loss: 0.2544 - accuracy: 0.9601 
 9400/25000 [==========>...................] - ETA: 9s - loss: 0.2544 - accuracy: 0.9600
 9500/25000 [==========>...................] - ETA: 9s - loss: 0.2544 - accuracy: 0.9600
 9600/25000 [==========>...................] - ETA: 9s - loss: 0.2539 - accuracy: 0.9603
 9700/25000 [==========>...................] - ETA: 9s - loss: 0.2541 - accuracy: 0.9602
 9800/25000 [==========>...................] - ETA: 9s - loss: 0.2539 - accuracy: 0.9603
 9900/25000 [==========>...................] - ETA: 9s - loss: 0.2537 - accuracy: 0.9604
10000/25000 [===========>..................] - ETA: 9s - loss: 0.2536 - accuracy: 0.9605
10100/25000 [===========>..................] - ETA: 9s - loss: 0.2532 - accuracy: 0.9607
10200/25000 [===========>..................] - ETA: 9s - loss: 0.2531 - accuracy: 0.9608
10300/25000 [===========>..................] - ETA: 9s - loss: 0.2532 - accuracy: 0.9606
10400/25000 [===========>..................] - ETA: 9s - loss: 0.2534 - accuracy: 0.9605
10500/25000 [===========>..................] - ETA: 9s - loss: 0.2539 - accuracy: 0.9600
10600/25000 [===========>..................] - ETA: 9s - loss: 0.2537 - accuracy: 0.9601
10700/25000 [===========>..................] - ETA: 9s - loss: 0.2533 - accuracy: 0.9603
10800/25000 [===========>..................] - ETA: 9s - loss: 0.2540 - accuracy: 0.9599
10900/25000 [============>.................] - ETA: 9s - loss: 0.2543 - accuracy: 0.9595
11000/25000 [============>.................] - ETA: 9s - loss: 0.2549 - accuracy: 0.9592
11100/25000 [============>.................] - ETA: 8s - loss: 0.2549 - accuracy: 0.9592
11200/25000 [============>.................] - ETA: 8s - loss: 0.2551 - accuracy: 0.9591
11300/25000 [============>.................] - ETA: 8s - loss: 0.2556 - accuracy: 0.9588
11400/25000 [============>.................] - ETA: 8s - loss: 0.2553 - accuracy: 0.9589
11500/25000 [============>.................] - ETA: 8s - loss: 0.2555 - accuracy: 0.9588
11600/25000 [============>.................] - ETA: 8s - loss: 0.2551 - accuracy: 0.9591
11700/25000 [=============>................] - ETA: 8s - loss: 0.2553 - accuracy: 0.9590
11800/25000 [=============>................] - ETA: 8s - loss: 0.2551 - accuracy: 0.9590
11900/25000 [=============>................] - ETA: 8s - loss: 0.2556 - accuracy: 0.9587
12000/25000 [=============>................] - ETA: 8s - loss: 0.2557 - accuracy: 0.9587
12100/25000 [=============>................] - ETA: 8s - loss: 0.2557 - accuracy: 0.9586
12200/25000 [=============>................] - ETA: 8s - loss: 0.2558 - accuracy: 0.9584
12300/25000 [=============>................] - ETA: 8s - loss: 0.2559 - accuracy: 0.9584
12400/25000 [=============>................] - ETA: 8s - loss: 0.2563 - accuracy: 0.9580
12500/25000 [==============>...............] - ETA: 8s - loss: 0.2563 - accuracy: 0.9580
12600/25000 [==============>...............] - ETA: 7s - loss: 0.2561 - accuracy: 0.9581
12700/25000 [==============>...............] - ETA: 7s - loss: 0.2565 - accuracy: 0.9577
12800/25000 [==============>...............] - ETA: 7s - loss: 0.2564 - accuracy: 0.9577
12900/25000 [==============>...............] - ETA: 7s - loss: 0.2566 - accuracy: 0.9576
13000/25000 [==============>...............] - ETA: 7s - loss: 0.2566 - accuracy: 0.9575
13100/25000 [==============>...............] - ETA: 7s - loss: 0.2571 - accuracy: 0.9573
13200/25000 [==============>...............] - ETA: 7s - loss: 0.2572 - accuracy: 0.9571
13300/25000 [==============>...............] - ETA: 7s - loss: 0.2573 - accuracy: 0.9571
13400/25000 [===============>..............] - ETA: 7s - loss: 0.2571 - accuracy: 0.9572
13500/25000 [===============>..............] - ETA: 7s - loss: 0.2572 - accuracy: 0.9571
13600/25000 [===============>..............] - ETA: 7s - loss: 0.2573 - accuracy: 0.9571
13700/25000 [===============>..............] - ETA: 7s - loss: 0.2571 - accuracy: 0.9572
13800/25000 [===============>..............] - ETA: 7s - loss: 0.2571 - accuracy: 0.9571
13900/25000 [===============>..............] - ETA: 7s - loss: 0.2569 - accuracy: 0.9571
14000/25000 [===============>..............] - ETA: 7s - loss: 0.2568 - accuracy: 0.9572
14100/25000 [===============>..............] - ETA: 7s - loss: 0.2568 - accuracy: 0.9572
14200/25000 [================>.............] - ETA: 6s - loss: 0.2568 - accuracy: 0.9572
14300/25000 [================>.............] - ETA: 6s - loss: 0.2568 - accuracy: 0.9572
14400/25000 [================>.............] - ETA: 6s - loss: 0.2566 - accuracy: 0.9573
14500/25000 [================>.............] - ETA: 6s - loss: 0.2565 - accuracy: 0.9574
14600/25000 [================>.............] - ETA: 6s - loss: 0.2570 - accuracy: 0.9570
14700/25000 [================>.............] - ETA: 6s - loss: 0.2568 - accuracy: 0.9571
14800/25000 [================>.............] - ETA: 6s - loss: 0.2568 - accuracy: 0.9571
14900/25000 [================>.............] - ETA: 6s - loss: 0.2570 - accuracy: 0.9569
15000/25000 [=================>............] - ETA: 6s - loss: 0.2570 - accuracy: 0.9569
15100/25000 [=================>............] - ETA: 6s - loss: 0.2570 - accuracy: 0.9569
15200/25000 [=================>............] - ETA: 6s - loss: 0.2567 - accuracy: 0.9570
15300/25000 [=================>............] - ETA: 6s - loss: 0.2569 - accuracy: 0.9569
15400/25000 [=================>............] - ETA: 6s - loss: 0.2570 - accuracy: 0.9568
15500/25000 [=================>............] - ETA: 6s - loss: 0.2571 - accuracy: 0.9566
15600/25000 [=================>............] - ETA: 6s - loss: 0.2572 - accuracy: 0.9566
15700/25000 [=================>............] - ETA: 5s - loss: 0.2571 - accuracy: 0.9566
15800/25000 [=================>............] - ETA: 5s - loss: 0.2573 - accuracy: 0.9565
15900/25000 [==================>...........] - ETA: 5s - loss: 0.2574 - accuracy: 0.9564
16000/25000 [==================>...........] - ETA: 5s - loss: 0.2574 - accuracy: 0.9564
16100/25000 [==================>...........] - ETA: 5s - loss: 0.2580 - accuracy: 0.9560
16200/25000 [==================>...........] - ETA: 5s - loss: 0.2583 - accuracy: 0.9557
16300/25000 [==================>...........] - ETA: 5s - loss: 0.2586 - accuracy: 0.9555
16400/25000 [==================>...........] - ETA: 5s - loss: 0.2588 - accuracy: 0.9554
16500/25000 [==================>...........] - ETA: 5s - loss: 0.2589 - accuracy: 0.9553
16600/25000 [==================>...........] - ETA: 5s - loss: 0.2589 - accuracy: 0.9554
16700/25000 [===================>..........] - ETA: 5s - loss: 0.2589 - accuracy: 0.9554
16800/25000 [===================>..........] - ETA: 5s - loss: 0.2589 - accuracy: 0.9554
16900/25000 [===================>..........] - ETA: 5s - loss: 0.2591 - accuracy: 0.9552
17000/25000 [===================>..........] - ETA: 5s - loss: 0.2593 - accuracy: 0.9551
17100/25000 [===================>..........] - ETA: 5s - loss: 0.2591 - accuracy: 0.9551
17200/25000 [===================>..........] - ETA: 5s - loss: 0.2592 - accuracy: 0.9551
17300/25000 [===================>..........] - ETA: 4s - loss: 0.2589 - accuracy: 0.9552
17400/25000 [===================>..........] - ETA: 4s - loss: 0.2588 - accuracy: 0.9552
17500/25000 [====================>.........] - ETA: 4s - loss: 0.2587 - accuracy: 0.9553
17600/25000 [====================>.........] - ETA: 4s - loss: 0.2586 - accuracy: 0.9553
17700/25000 [====================>.........] - ETA: 4s - loss: 0.2584 - accuracy: 0.9554
17800/25000 [====================>.........] - ETA: 4s - loss: 0.2583 - accuracy: 0.9554
17900/25000 [====================>.........] - ETA: 4s - loss: 0.2580 - accuracy: 0.9556
18000/25000 [====================>.........] - ETA: 4s - loss: 0.2579 - accuracy: 0.9556
18100/25000 [====================>.........] - ETA: 4s - loss: 0.2580 - accuracy: 0.9556
18200/25000 [====================>.........] - ETA: 4s - loss: 0.2580 - accuracy: 0.9555
18300/25000 [====================>.........] - ETA: 4s - loss: 0.2579 - accuracy: 0.9555
18400/25000 [=====================>........] - ETA: 4s - loss: 0.2580 - accuracy: 0.9554
18500/25000 [=====================>........] - ETA: 4s - loss: 0.2579 - accuracy: 0.9554
18600/25000 [=====================>........] - ETA: 4s - loss: 0.2579 - accuracy: 0.9554
18700/25000 [=====================>........] - ETA: 4s - loss: 0.2578 - accuracy: 0.9555
18800/25000 [=====================>........] - ETA: 3s - loss: 0.2578 - accuracy: 0.9554
18900/25000 [=====================>........] - ETA: 3s - loss: 0.2579 - accuracy: 0.9553
19000/25000 [=====================>........] - ETA: 3s - loss: 0.2581 - accuracy: 0.9552
19100/25000 [=====================>........] - ETA: 3s - loss: 0.2580 - accuracy: 0.9552
19200/25000 [======================>.......] - ETA: 3s - loss: 0.2578 - accuracy: 0.9553
19300/25000 [======================>.......] - ETA: 3s - loss: 0.2578 - accuracy: 0.9553
19400/25000 [======================>.......] - ETA: 3s - loss: 0.2576 - accuracy: 0.9554
19500/25000 [======================>.......] - ETA: 3s - loss: 0.2576 - accuracy: 0.9554
19600/25000 [======================>.......] - ETA: 3s - loss: 0.2576 - accuracy: 0.9553
19700/25000 [======================>.......] - ETA: 3s - loss: 0.2575 - accuracy: 0.9554
19800/25000 [======================>.......] - ETA: 3s - loss: 0.2578 - accuracy: 0.9552
19900/25000 [======================>.......] - ETA: 3s - loss: 0.2577 - accuracy: 0.9552
20000/25000 [=======================>......] - ETA: 3s - loss: 0.2580 - accuracy: 0.9549
20100/25000 [=======================>......] - ETA: 3s - loss: 0.2581 - accuracy: 0.9549
20200/25000 [=======================>......] - ETA: 3s - loss: 0.2579 - accuracy: 0.9550
20300/25000 [=======================>......] - ETA: 3s - loss: 0.2580 - accuracy: 0.9549
20400/25000 [=======================>......] - ETA: 2s - loss: 0.2582 - accuracy: 0.9548
20500/25000 [=======================>......] - ETA: 2s - loss: 0.2580 - accuracy: 0.9549
20600/25000 [=======================>......] - ETA: 2s - loss: 0.2580 - accuracy: 0.9549
20700/25000 [=======================>......] - ETA: 2s - loss: 0.2580 - accuracy: 0.9549
20800/25000 [=======================>......] - ETA: 2s - loss: 0.2581 - accuracy: 0.9548
20900/25000 [========================>.....] - ETA: 2s - loss: 0.2586 - accuracy: 0.9544
21000/25000 [========================>.....] - ETA: 2s - loss: 0.2587 - accuracy: 0.9544
21100/25000 [========================>.....] - ETA: 2s - loss: 0.2587 - accuracy: 0.9544
21200/25000 [========================>.....] - ETA: 2s - loss: 0.2589 - accuracy: 0.9542
21300/25000 [========================>.....] - ETA: 2s - loss: 0.2588 - accuracy: 0.9542
21400/25000 [========================>.....] - ETA: 2s - loss: 0.2590 - accuracy: 0.9541
21500/25000 [========================>.....] - ETA: 2s - loss: 0.2589 - accuracy: 0.9541
21600/25000 [========================>.....] - ETA: 2s - loss: 0.2589 - accuracy: 0.9541
21700/25000 [=========================>....] - ETA: 2s - loss: 0.2588 - accuracy: 0.9541
21800/25000 [=========================>....] - ETA: 2s - loss: 0.2591 - accuracy: 0.9539
21900/25000 [=========================>....] - ETA: 1s - loss: 0.2589 - accuracy: 0.9540
22000/25000 [=========================>....] - ETA: 1s - loss: 0.2587 - accuracy: 0.9541
22100/25000 [=========================>....] - ETA: 1s - loss: 0.2586 - accuracy: 0.9542
22200/25000 [=========================>....] - ETA: 1s - loss: 0.2586 - accuracy: 0.9542
22300/25000 [=========================>....] - ETA: 1s - loss: 0.2586 - accuracy: 0.9542
22400/25000 [=========================>....] - ETA: 1s - loss: 0.2587 - accuracy: 0.9541
22500/25000 [==========================>...] - ETA: 1s - loss: 0.2591 - accuracy: 0.9539
22600/25000 [==========================>...] - ETA: 1s - loss: 0.2590 - accuracy: 0.9539
22700/25000 [==========================>...] - ETA: 1s - loss: 0.2590 - accuracy: 0.9539
22800/25000 [==========================>...] - ETA: 1s - loss: 0.2588 - accuracy: 0.9539
22900/25000 [==========================>...] - ETA: 1s - loss: 0.2585 - accuracy: 0.9541
23000/25000 [==========================>...] - ETA: 1s - loss: 0.2583 - accuracy: 0.9542
23100/25000 [==========================>...] - ETA: 1s - loss: 0.2584 - accuracy: 0.9541
23200/25000 [==========================>...] - ETA: 1s - loss: 0.2584 - accuracy: 0.9540
23300/25000 [==========================>...] - ETA: 1s - loss: 0.2584 - accuracy: 0.9540
23400/25000 [===========================>..] - ETA: 1s - loss: 0.2587 - accuracy: 0.9538
23500/25000 [===========================>..] - ETA: 0s - loss: 0.2588 - accuracy: 0.9537
23600/25000 [===========================>..] - ETA: 0s - loss: 0.2587 - accuracy: 0.9538
23700/25000 [===========================>..] - ETA: 0s - loss: 0.2587 - accuracy: 0.9538
23800/25000 [===========================>..] - ETA: 0s - loss: 0.2586 - accuracy: 0.9538
23900/25000 [===========================>..] - ETA: 0s - loss: 0.2585 - accuracy: 0.9539
24000/25000 [===========================>..] - ETA: 0s - loss: 0.2586 - accuracy: 0.9538
24100/25000 [===========================>..] - ETA: 0s - loss: 0.2585 - accuracy: 0.9539
24200/25000 [============================>.] - ETA: 0s - loss: 0.2585 - accuracy: 0.9538
24300/25000 [============================>.] - ETA: 0s - loss: 0.2584 - accuracy: 0.9539
24400/25000 [============================>.] - ETA: 0s - loss: 0.2585 - accuracy: 0.9539
24500/25000 [============================>.] - ETA: 0s - loss: 0.2586 - accuracy: 0.9538
24600/25000 [============================>.] - ETA: 0s - loss: 0.2584 - accuracy: 0.9539
24700/25000 [============================>.] - ETA: 0s - loss: 0.2583 - accuracy: 0.9539
24800/25000 [============================>.] - ETA: 0s - loss: 0.2584 - accuracy: 0.9538
24900/25000 [============================>.] - ETA: 0s - loss: 0.2584 - accuracy: 0.9538
25000/25000 [==============================] - 20s 809us/step - loss: 0.2585 - accuracy: 0.9538 - val_loss: 0.4087 - val_accuracy: 0.8515
	=====> Test the model: model.predict()
IMDB_REVIEWS
	Training accuracy score: 96.06%
	Loss: 0.2433
	Test Accuracy: 85.15%




FINAL CLASSIFICATION TABLE:

| ID | Dataset            | Algorithm                              | Loss    | Training accuracy score (%) | Test accuracy score (%) | Training time (seconds) | Test time (seconds) |
| -- | ------------------ | -------------------------------------- | ------- | --------------------------- | ----------------------- | ----------------------- | ------------------- |
| 1  | TWENTY_NEWS_GROUPS | Deep Learning using Keras 1 (KerasDL1) | 0.9973  | 11.05                       | 96.80%                  | 98.2616                 | 5.5369              |
| 2 | IMDB_REVIEWS | Deep Learning using Keras 1 (KerasDL1) | 1.0000 | 93.46 | 82.90% | 175.9112 | 3.4582 |
| {} | {} | {} | {:.4f} | {:.2f} | {:.2f}% | {:.4f} | {:.4f} | 3 TWENTY_NEWS_GROUPS Deep Learning using Keras 2 (KerasDL2) 0.9496006965637207 25.44489246033888 94.95982527732849 85.33946776390076 3.0366663932800293
| {} | {} | {} | {:.4f} | {:.2f} | {:.2f}% | {:.4f} | {:.4f} | 4 IMDB_REVIEWS Deep Learning using Keras 2 (KerasDL2) 0.9605600237846375 40.86766824531555 85.15200018882751 203.81161618232727 6.439742565155029

MODEL 1: KERAS DEEP LEARNING MODEL
Using layers:
	==> Dense(10, input_dim=input_dim, activation='relu')
	==> Dense(7=IMDB_REVIEWS multi-label or 19 = TWENTY_NEWS_GROUPS or 1 to IMDB_REVIEWS binary label, activation='sigmoid')
Compile option:
	==> model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

MODEL 2: KERAS DEEP LEARNING MODEL
Using layers:
	==> Embedding(max_features, embed_size)
	==> Bidirectional(LSTM(32, return_sequences = True)
	==> GlobalMaxPool1D()
	==> Dense(20, activation="relu")
	==> Dropout(0.05)
	==> Dense(1, activation="sigmoid")
	==> Dense(1, activation="sigmoid")
Compile option:
	==> model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


DONE!
Program finished. It took 694.0487825870514 seconds

Process finished with exit code 0
```