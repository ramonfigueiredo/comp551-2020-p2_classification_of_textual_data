## Two Deep Learning approaches using Keras: Loss, Training Accuracy, Test Accuracy, Training Time, Test Time

| ID | Dataset | Algorithm | Loss | Training accuracy score (%) | Test accuracy score (%) | Training time (seconds) | Test time (seconds) |
| --- | ------- | --------- | ---- | --------------------------- | ----------------------- | ----------------------- | ------------------- |
| 1 | TWENTY_NEWS_GROUPS | Deep Learning using Keras 1 (KerasDL1) | 0.0301 | 99.42 | 96.56 | 95.1296 | 4.9434 |
| 2 | IMDB_REVIEWS | Deep Learning using Keras 1 (KerasDL1) | 0.0003 | 100.00 | 82.94 | 163.8357 | 3.2235 |
| 3 | TWENTY_NEWS_GROUPS | Deep Learning using Keras 2 (KerasDL2) | 0.2542 | 94.96 | 94.96 | 85.0631 | 2.8639 |
| 4 | IMDB_REVIEWS | Deep Learning using Keras 2 (KerasDL2) | 0.2784 | 95.44 | 85.30 | 200.2824 | 6.5434 |

### Deep Learning using Keras 1 (KerasDL1)

![TWENTY_NEWS_GROUPS: Loss, Training Accuracy, Test Accuracy, Training Time, Test Time](https://github.com/ramonfigueiredopessoa/comp551-2020-p2_classification_of_textual_data/blob/master/code/results/deep_learning_using_keras/KerasDL1_TWENTY_NEWS_GROUPS_training_and_validation_accuracy_and_Loss.png)

![IMDB_REVIEWS: Loss, Training Accuracy, Test Accuracy, Training Time, Test Time](https://github.com/ramonfigueiredopessoa/comp551-2020-p2_classification_of_textual_data/blob/master/code/results/deep_learning_using_keras/KerasDL1_IMDB_REVIEWS_training_and_validation_accuracy_and_Loss.png)


### Learning using Keras 1 (KerasDL1)

![TWENTY_NEWS_GROUPS: Loss, Training Accuracy, Test Accuracy, Training Time, Test Time](https://github.com/ramonfigueiredopessoa/comp551-2020-p2_classification_of_textual_data/blob/master/code/results/deep_learning_using_keras/KerasDL2_TWENTY_NEWS_GROUPS_training_and_validation_accuracy_and_Loss.png)

![IMDB_REVIEWS: Loss, Training Accuracy, Test Accuracy, Training Time, Test Time](https://github.com/ramonfigueiredopessoa/comp551-2020-p2_classification_of_textual_data/blob/master/code/results/deep_learning_using_keras/KerasDL2_IMDB_REVIEWS_training_and_validation_accuracy_and_Loss.png)


#### Computer settings used to run

* Operating system: Ubuntu 18.04.4 LTS (64-bit)
* Processor: Intel® Core™ i7-7700 CPU @ 3.60GHz × 8 
* Memory: 32 GB

#### All logs 

```
/home/ets-crchum/virtual_envs/comp551_p2/bin/python /home/ets-crchum/github/comp551-2020-p2_classification_of_textual_data/code/main.py -dl
Using TensorFlow backend.
2020-03-09 20:15:41.486696: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libnvinfer.so.6'; dlerror: libnvinfer.so.6: cannot open shared object file: No such file or directory
2020-03-09 20:15:41.486748: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libnvinfer_plugin.so.6'; dlerror: libnvinfer_plugin.so.6: cannot open shared object file: No such file or directory
2020-03-09 20:15:41.486753: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:30] Cannot dlopen some TensorRT libraries. If you would like to use Nvidia GPU with TensorRT, please make sure the missing libraries mentioned above are installed properly.
[nltk_data] Downloading package wordnet to /home/ets-
[nltk_data]     crchum/nltk_data...
[nltk_data]   Package wordnet is already up-to-date!
03/09/2020 08:15:42 PM - INFO - Program started...
03/09/2020 08:15:42 PM - INFO - Program started...
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
data loaded
11314 documents - 13.782MB (training set)
7532 documents - 8.262MB (test set)
20 categories

Extracting features from the training data using a vectorizer
done in 1.120154s at 12.304MB/s
n_samples: 11314, n_features: 101321

Extracting features from the test data using the same vectorizer
done in 0.598607s at 13.801MB/s
n_samples: 7532, n_features: 101321

================================================================================
KERAS DEEP LEARNING MODEL
Using layers:
	==> Dense(10, input_dim=input_dim, activation='relu')
	==> Dense(19, activation='sigmoid')
Compile option:
	==> model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
2020-03-09 20:15:45.538985: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
2020-03-09 20:15:45.560855: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-03-09 20:15:45.561430: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1555] Found device 0 with properties: 
pciBusID: 0000:01:00.0 name: GeForce GTX 1080 computeCapability: 6.1
coreClock: 1.7335GHz coreCount: 20 deviceMemorySize: 7.93GiB deviceMemoryBandwidth: 298.32GiB/s
2020-03-09 20:15:45.561494: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcudart.so.10.1'; dlerror: libcudart.so.10.1: cannot open shared object file: No such file or directory
2020-03-09 20:15:45.561537: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcublas.so.10'; dlerror: libcublas.so.10: cannot open shared object file: No such file or directory
2020-03-09 20:15:45.561575: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcufft.so.10'; dlerror: libcufft.so.10: cannot open shared object file: No such file or directory
2020-03-09 20:15:45.561627: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcurand.so.10'; dlerror: libcurand.so.10: cannot open shared object file: No such file or directory
2020-03-09 20:15:45.561666: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcusolver.so.10'; dlerror: libcusolver.so.10: cannot open shared object file: No such file or directory
2020-03-09 20:15:45.561717: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcusparse.so.10'; dlerror: libcusparse.so.10: cannot open shared object file: No such file or directory
2020-03-09 20:15:45.563634: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2020-03-09 20:15:45.563644: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1592] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
Skipping registering GPU devices...
2020-03-09 20:15:45.563814: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-03-09 20:15:45.587635: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 3600000000 Hz
2020-03-09 20:15:45.588108: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5fed940 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-03-09 20:15:45.588127: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2020-03-09 20:15:45.663180: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-03-09 20:15:45.663897: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5f7e190 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2020-03-09 20:15:45.663909: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): GeForce GTX 1080, Compute Capability 6.1
2020-03-09 20:15:45.664003: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1096] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-03-09 20:15:45.664008: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1102]      
	Dataset: TWENTY_NEWS_GROUPS
	Algorithm: Deep Learning using Keras 1 (KerasDL1)
	Training Loss: 0.0301
	Training accuracy score: 99.42%
	Test Accuracy: 96.56%
	Test Loss: 0.1162
	Training Time: 95.1296
	Test Time: 4.9434


Loading IMDB_REVIEWS dataset:

===> Reading files from /home/ets-crchum/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/train/neg

===> Reading files from /home/ets-crchum/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/train/pos

===> Reading files from /home/ets-crchum/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/test/neg

===> Reading files from /home/ets-crchum/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/test/pos
data loaded
25000 documents - 33.133MB (training set)
25000 documents - 32.351MB (test set)

Extracting features from the training data using a vectorizer
done in 2.928984s at 11.312MB/s
n_samples: 25000, n_features: 74170

Extracting features from the test data using the same vectorizer
done in 2.850790s at 11.348MB/s
n_samples: 25000, n_features: 74170

================================================================================
KERAS DEEP LEARNING MODEL
Using layers:
	==> Dense(10, input_dim=input_dim, activation='relu')
	==> Dense(1, activation='sigmoid')
Compile option:
	==> model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	Dataset: IMDB_REVIEWS
	Algorithm: Deep Learning using Keras 1 (KerasDL1)
	Training Loss: 0.0003
	Training accuracy score: 100.00%
	Test Accuracy: 82.94%
	Test Loss: 0.9399
	Training Time: 163.8357
	Test Time: 3.2235


Loading TWENTY_NEWS_GROUPS dataset for categories:
03/09/2020 08:20:26 PM - INFO - Program started...
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
	It took 10.08867359161377 seconds


Applying NLTK feature extraction: X_test
	==> Replace unused UNICODE characters. 
	==> Text in lower case. 
	==> Apply Lemmatize using WordNet's built-in morphy function. Returns the input word unchanged if it cannot be found in WordNet 
	==> Remove English stop words
	Done: NLTK feature extraction (replace unused UNICODE characters, text in lower case, apply Lemmatize using WordNet, remove English stop words)  finished!
	It took 5.839775323867798 seconds

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

  100/11314 [..............................] - ETA: 59s - loss: 0.6769 - accuracy: 0.6195
  200/11314 [..............................] - ETA: 32s - loss: 0.6764 - accuracy: 0.6171
  300/11314 [..............................] - ETA: 24s - loss: 0.6760 - accuracy: 0.6174
  400/11314 [>.............................] - ETA: 19s - loss: 0.6756 - accuracy: 0.6176
  500/11314 [>.............................] - ETA: 16s - loss: 0.6752 - accuracy: 0.6178
  600/11314 [>.............................] - ETA: 15s - loss: 0.6750 - accuracy: 0.6164
  700/11314 [>.............................] - ETA: 13s - loss: 0.6747 - accuracy: 0.6156
  800/11314 [=>............................] - ETA: 12s - loss: 0.6742 - accuracy: 0.6168
  900/11314 [=>............................] - ETA: 11s - loss: 0.6738 - accuracy: 0.6164
 1000/11314 [=>............................] - ETA: 11s - loss: 0.6736 - accuracy: 0.6158
 1100/11314 [=>............................] - ETA: 10s - loss: 0.6731 - accuracy: 0.6160
 1200/11314 [==>...........................] - ETA: 10s - loss: 0.6729 - accuracy: 0.6154
 1300/11314 [==>...........................] - ETA: 9s - loss: 0.6723 - accuracy: 0.6159 
 1400/11314 [==>...........................] - ETA: 9s - loss: 0.6717 - accuracy: 0.6168
 1500/11314 [==>...........................] - ETA: 9s - loss: 0.6714 - accuracy: 0.6166
 1600/11314 [===>..........................] - ETA: 8s - loss: 0.6708 - accuracy: 0.6172
 1700/11314 [===>..........................] - ETA: 8s - loss: 0.6704 - accuracy: 0.6170
 1800/11314 [===>..........................] - ETA: 8s - loss: 0.6699 - accuracy: 0.6170
 1900/11314 [====>.........................] - ETA: 8s - loss: 0.6694 - accuracy: 0.6168
 2000/11314 [====>.........................] - ETA: 7s - loss: 0.6689 - accuracy: 0.6166
 2100/11314 [====>.........................] - ETA: 7s - loss: 0.6685 - accuracy: 0.6163
 2200/11314 [====>.........................] - ETA: 7s - loss: 0.6679 - accuracy: 0.6168
 2300/11314 [=====>........................] - ETA: 7s - loss: 0.6673 - accuracy: 0.6170
 2400/11314 [=====>........................] - ETA: 7s - loss: 0.6668 - accuracy: 0.6172
 2500/11314 [=====>........................] - ETA: 7s - loss: 0.6663 - accuracy: 0.6172
 2600/11314 [=====>........................] - ETA: 7s - loss: 0.6657 - accuracy: 0.6174
 2700/11314 [======>.......................] - ETA: 6s - loss: 0.6650 - accuracy: 0.6177
 2800/11314 [======>.......................] - ETA: 6s - loss: 0.6645 - accuracy: 0.6178
 2900/11314 [======>.......................] - ETA: 6s - loss: 0.6639 - accuracy: 0.6181
 3000/11314 [======>.......................] - ETA: 6s - loss: 0.6633 - accuracy: 0.6181
 3100/11314 [=======>......................] - ETA: 6s - loss: 0.6627 - accuracy: 0.6183
 3200/11314 [=======>......................] - ETA: 6s - loss: 0.6622 - accuracy: 0.6183
 3300/11314 [=======>......................] - ETA: 6s - loss: 0.6616 - accuracy: 0.6188
 3400/11314 [========>.....................] - ETA: 6s - loss: 0.6609 - accuracy: 0.6204
 3500/11314 [========>.....................] - ETA: 5s - loss: 0.6604 - accuracy: 0.6215
 3600/11314 [========>.....................] - ETA: 5s - loss: 0.6598 - accuracy: 0.6226
 3700/11314 [========>.....................] - ETA: 5s - loss: 0.6592 - accuracy: 0.6236
 3800/11314 [=========>....................] - ETA: 5s - loss: 0.6586 - accuracy: 0.6248
 3900/11314 [=========>....................] - ETA: 5s - loss: 0.6580 - accuracy: 0.6258
 4000/11314 [=========>....................] - ETA: 5s - loss: 0.6575 - accuracy: 0.6269
 4100/11314 [=========>....................] - ETA: 5s - loss: 0.6569 - accuracy: 0.6277
 4200/11314 [==========>...................] - ETA: 5s - loss: 0.6564 - accuracy: 0.6286
 4300/11314 [==========>...................] - ETA: 5s - loss: 0.6558 - accuracy: 0.6295
 4400/11314 [==========>...................] - ETA: 5s - loss: 0.6553 - accuracy: 0.6299
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.6548 - accuracy: 0.6307
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.6542 - accuracy: 0.6316
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.6536 - accuracy: 0.6322
 4800/11314 [===========>..................] - ETA: 4s - loss: 0.6531 - accuracy: 0.6328
 4900/11314 [===========>..................] - ETA: 4s - loss: 0.6526 - accuracy: 0.6334
 5000/11314 [============>.................] - ETA: 4s - loss: 0.6520 - accuracy: 0.6341
 5100/11314 [============>.................] - ETA: 4s - loss: 0.6515 - accuracy: 0.6347
 5200/11314 [============>.................] - ETA: 4s - loss: 0.6509 - accuracy: 0.6353
 5300/11314 [=============>................] - ETA: 4s - loss: 0.6504 - accuracy: 0.6361
 5400/11314 [=============>................] - ETA: 4s - loss: 0.6498 - accuracy: 0.6366
 5500/11314 [=============>................] - ETA: 4s - loss: 0.6493 - accuracy: 0.6370
 5600/11314 [=============>................] - ETA: 4s - loss: 0.6488 - accuracy: 0.6375
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.6482 - accuracy: 0.6381
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.6477 - accuracy: 0.6386
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.6472 - accuracy: 0.6390
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.6467 - accuracy: 0.6393
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.6462 - accuracy: 0.6397
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.6457 - accuracy: 0.6402
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.6452 - accuracy: 0.6405
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.6447 - accuracy: 0.6409
 6500/11314 [================>.............] - ETA: 3s - loss: 0.6442 - accuracy: 0.6411
 6600/11314 [================>.............] - ETA: 3s - loss: 0.6438 - accuracy: 0.6413
 6700/11314 [================>.............] - ETA: 3s - loss: 0.6433 - accuracy: 0.6415
 6800/11314 [=================>............] - ETA: 3s - loss: 0.6428 - accuracy: 0.6418
 6900/11314 [=================>............] - ETA: 3s - loss: 0.6423 - accuracy: 0.6421
 7000/11314 [=================>............] - ETA: 2s - loss: 0.6418 - accuracy: 0.6425
 7100/11314 [=================>............] - ETA: 2s - loss: 0.6414 - accuracy: 0.6427
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.6408 - accuracy: 0.6432
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.6403 - accuracy: 0.6435
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.6398 - accuracy: 0.6439
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.6393 - accuracy: 0.6442
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.6389 - accuracy: 0.6445
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.6384 - accuracy: 0.6447
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.6379 - accuracy: 0.6451
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.6374 - accuracy: 0.6454
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.6369 - accuracy: 0.6456
 8100/11314 [====================>.........] - ETA: 2s - loss: 0.6365 - accuracy: 0.6458
 8200/11314 [====================>.........] - ETA: 2s - loss: 0.6360 - accuracy: 0.6460
 8300/11314 [=====================>........] - ETA: 2s - loss: 0.6355 - accuracy: 0.6463
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.6351 - accuracy: 0.6466
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.6346 - accuracy: 0.6468
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.6341 - accuracy: 0.6470
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.6337 - accuracy: 0.6472
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.6332 - accuracy: 0.6474
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.6328 - accuracy: 0.6476
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.6323 - accuracy: 0.6477
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.6319 - accuracy: 0.6480
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.6314 - accuracy: 0.6481
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.6310 - accuracy: 0.6484
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.6305 - accuracy: 0.6491
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.6301 - accuracy: 0.6498
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.6296 - accuracy: 0.6504
 9700/11314 [========================>.....] - ETA: 1s - loss: 0.6292 - accuracy: 0.6511
 9800/11314 [========================>.....] - ETA: 1s - loss: 0.6287 - accuracy: 0.6517
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.6283 - accuracy: 0.6524
10000/11314 [=========================>....] - ETA: 0s - loss: 0.6279 - accuracy: 0.6529
10100/11314 [=========================>....] - ETA: 0s - loss: 0.6274 - accuracy: 0.6534
10200/11314 [==========================>...] - ETA: 0s - loss: 0.6270 - accuracy: 0.6539
10300/11314 [==========================>...] - ETA: 0s - loss: 0.6266 - accuracy: 0.6545
10400/11314 [==========================>...] - ETA: 0s - loss: 0.6261 - accuracy: 0.6551
10500/11314 [==========================>...] - ETA: 0s - loss: 0.6257 - accuracy: 0.6562
10600/11314 [===========================>..] - ETA: 0s - loss: 0.6252 - accuracy: 0.6571
10700/11314 [===========================>..] - ETA: 0s - loss: 0.6248 - accuracy: 0.6581
10800/11314 [===========================>..] - ETA: 0s - loss: 0.6244 - accuracy: 0.6590
10900/11314 [===========================>..] - ETA: 0s - loss: 0.6240 - accuracy: 0.6599
11000/11314 [============================>.] - ETA: 0s - loss: 0.6236 - accuracy: 0.6608
11100/11314 [============================>.] - ETA: 0s - loss: 0.6231 - accuracy: 0.6617
11200/11314 [============================>.] - ETA: 0s - loss: 0.6227 - accuracy: 0.6625
11300/11314 [============================>.] - ETA: 0s - loss: 0.6223 - accuracy: 0.6635
11314/11314 [==============================] - 9s 781us/step - loss: 0.6222 - accuracy: 0.6636 - val_loss: 0.5728 - val_accuracy: 0.7592
Epoch 2/10

  100/11314 [..............................] - ETA: 6s - loss: 0.5753 - accuracy: 0.7479
  200/11314 [..............................] - ETA: 6s - loss: 0.5752 - accuracy: 0.7505
  300/11314 [..............................] - ETA: 6s - loss: 0.5732 - accuracy: 0.7551
  400/11314 [>.............................] - ETA: 6s - loss: 0.5724 - accuracy: 0.7563
  500/11314 [>.............................] - ETA: 6s - loss: 0.5718 - accuracy: 0.7572
  600/11314 [>.............................] - ETA: 6s - loss: 0.5713 - accuracy: 0.7569
  700/11314 [>.............................] - ETA: 6s - loss: 0.5708 - accuracy: 0.7581
  800/11314 [=>............................] - ETA: 6s - loss: 0.5705 - accuracy: 0.7580
  900/11314 [=>............................] - ETA: 6s - loss: 0.5699 - accuracy: 0.7586
 1000/11314 [=>............................] - ETA: 6s - loss: 0.5697 - accuracy: 0.7586
 1100/11314 [=>............................] - ETA: 6s - loss: 0.5694 - accuracy: 0.7583
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.5691 - accuracy: 0.7582
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.5685 - accuracy: 0.7590
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.5681 - accuracy: 0.7588
 1500/11314 [==>...........................] - ETA: 6s - loss: 0.5676 - accuracy: 0.7591
 1600/11314 [===>..........................] - ETA: 6s - loss: 0.5674 - accuracy: 0.7589
 1700/11314 [===>..........................] - ETA: 6s - loss: 0.5669 - accuracy: 0.7592
 1800/11314 [===>..........................] - ETA: 5s - loss: 0.5665 - accuracy: 0.7593
 1900/11314 [====>.........................] - ETA: 5s - loss: 0.5663 - accuracy: 0.7589
 2000/11314 [====>.........................] - ETA: 5s - loss: 0.5661 - accuracy: 0.7587
 2100/11314 [====>.........................] - ETA: 5s - loss: 0.5657 - accuracy: 0.7585
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.5653 - accuracy: 0.7585
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.5649 - accuracy: 0.7586
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.5647 - accuracy: 0.7584
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.5644 - accuracy: 0.7583
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.5640 - accuracy: 0.7584
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.5635 - accuracy: 0.7587
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.5632 - accuracy: 0.7586
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.5629 - accuracy: 0.7584
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.5625 - accuracy: 0.7584
 3100/11314 [=======>......................] - ETA: 5s - loss: 0.5621 - accuracy: 0.7602
 3200/11314 [=======>......................] - ETA: 5s - loss: 0.5619 - accuracy: 0.7614
 3300/11314 [=======>......................] - ETA: 4s - loss: 0.5616 - accuracy: 0.7626
 3400/11314 [========>.....................] - ETA: 4s - loss: 0.5612 - accuracy: 0.7638
 3500/11314 [========>.....................] - ETA: 4s - loss: 0.5609 - accuracy: 0.7650
 3600/11314 [========>.....................] - ETA: 4s - loss: 0.5606 - accuracy: 0.7661
 3700/11314 [========>.....................] - ETA: 4s - loss: 0.5603 - accuracy: 0.7671
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.5598 - accuracy: 0.7683
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.5595 - accuracy: 0.7692
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.5591 - accuracy: 0.7701
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.5588 - accuracy: 0.7708
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.5585 - accuracy: 0.7718
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.5582 - accuracy: 0.7726
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.5579 - accuracy: 0.7733
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.5575 - accuracy: 0.7740
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.5572 - accuracy: 0.7747
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.5569 - accuracy: 0.7764
 4800/11314 [===========>..................] - ETA: 4s - loss: 0.5566 - accuracy: 0.7780
 4900/11314 [===========>..................] - ETA: 3s - loss: 0.5562 - accuracy: 0.7796
 5000/11314 [============>.................] - ETA: 3s - loss: 0.5559 - accuracy: 0.7811
 5100/11314 [============>.................] - ETA: 3s - loss: 0.5555 - accuracy: 0.7825
 5200/11314 [============>.................] - ETA: 3s - loss: 0.5551 - accuracy: 0.7839
 5300/11314 [=============>................] - ETA: 3s - loss: 0.5548 - accuracy: 0.7853
 5400/11314 [=============>................] - ETA: 3s - loss: 0.5544 - accuracy: 0.7866
 5500/11314 [=============>................] - ETA: 3s - loss: 0.5541 - accuracy: 0.7877
 5600/11314 [=============>................] - ETA: 3s - loss: 0.5538 - accuracy: 0.7888
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.5535 - accuracy: 0.7900
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.5531 - accuracy: 0.7911
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.5528 - accuracy: 0.7923
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.5524 - accuracy: 0.7932
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.5521 - accuracy: 0.7943
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.5518 - accuracy: 0.7951
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.5514 - accuracy: 0.7961
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.5511 - accuracy: 0.7971
 6500/11314 [================>.............] - ETA: 2s - loss: 0.5507 - accuracy: 0.7980
 6600/11314 [================>.............] - ETA: 2s - loss: 0.5504 - accuracy: 0.7988
 6700/11314 [================>.............] - ETA: 2s - loss: 0.5500 - accuracy: 0.7996
 6800/11314 [=================>............] - ETA: 2s - loss: 0.5496 - accuracy: 0.8004
 6900/11314 [=================>............] - ETA: 2s - loss: 0.5493 - accuracy: 0.8019
 7000/11314 [=================>............] - ETA: 2s - loss: 0.5490 - accuracy: 0.8033
 7100/11314 [=================>............] - ETA: 2s - loss: 0.5486 - accuracy: 0.8047
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.5483 - accuracy: 0.8060
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.5479 - accuracy: 0.8073
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.5476 - accuracy: 0.8086
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.5473 - accuracy: 0.8098
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.5470 - accuracy: 0.8111
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.5466 - accuracy: 0.8123
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.5463 - accuracy: 0.8134
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.5460 - accuracy: 0.8145
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.5457 - accuracy: 0.8156
 8100/11314 [====================>.........] - ETA: 1s - loss: 0.5453 - accuracy: 0.8167
 8200/11314 [====================>.........] - ETA: 1s - loss: 0.5450 - accuracy: 0.8178
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.5446 - accuracy: 0.8188
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.5443 - accuracy: 0.8199
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.5439 - accuracy: 0.8208
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.5436 - accuracy: 0.8217
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.5433 - accuracy: 0.8227
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.5429 - accuracy: 0.8236
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.5426 - accuracy: 0.8244
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.5423 - accuracy: 0.8253
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.5420 - accuracy: 0.8262
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.5416 - accuracy: 0.8270
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.5413 - accuracy: 0.8278
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.5410 - accuracy: 0.8286
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.5406 - accuracy: 0.8294
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.5403 - accuracy: 0.8302
 9700/11314 [========================>.....] - ETA: 0s - loss: 0.5400 - accuracy: 0.8309
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.5397 - accuracy: 0.8317
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.5393 - accuracy: 0.8324
10000/11314 [=========================>....] - ETA: 0s - loss: 0.5390 - accuracy: 0.8331
10100/11314 [=========================>....] - ETA: 0s - loss: 0.5387 - accuracy: 0.8338
10200/11314 [==========================>...] - ETA: 0s - loss: 0.5384 - accuracy: 0.8344
10300/11314 [==========================>...] - ETA: 0s - loss: 0.5381 - accuracy: 0.8351
10400/11314 [==========================>...] - ETA: 0s - loss: 0.5378 - accuracy: 0.8358
10500/11314 [==========================>...] - ETA: 0s - loss: 0.5375 - accuracy: 0.8364
10600/11314 [===========================>..] - ETA: 0s - loss: 0.5371 - accuracy: 0.8370
10700/11314 [===========================>..] - ETA: 0s - loss: 0.5368 - accuracy: 0.8377
10800/11314 [===========================>..] - ETA: 0s - loss: 0.5365 - accuracy: 0.8383
10900/11314 [===========================>..] - ETA: 0s - loss: 0.5362 - accuracy: 0.8388
11000/11314 [============================>.] - ETA: 0s - loss: 0.5359 - accuracy: 0.8394
11100/11314 [============================>.] - ETA: 0s - loss: 0.5356 - accuracy: 0.8400
11200/11314 [============================>.] - ETA: 0s - loss: 0.5353 - accuracy: 0.8409
11300/11314 [============================>.] - ETA: 0s - loss: 0.5349 - accuracy: 0.8419
11314/11314 [==============================] - 8s 729us/step - loss: 0.5349 - accuracy: 0.8420 - val_loss: 0.4987 - val_accuracy: 0.9496
Epoch 3/10

  100/11314 [..............................] - ETA: 6s - loss: 0.5008 - accuracy: 0.9484
  200/11314 [..............................] - ETA: 6s - loss: 0.4995 - accuracy: 0.9497
  300/11314 [..............................] - ETA: 6s - loss: 0.4991 - accuracy: 0.9496
  400/11314 [>.............................] - ETA: 6s - loss: 0.4993 - accuracy: 0.9493
  500/11314 [>.............................] - ETA: 6s - loss: 0.4992 - accuracy: 0.9494
  600/11314 [>.............................] - ETA: 6s - loss: 0.4987 - accuracy: 0.9495
  700/11314 [>.............................] - ETA: 6s - loss: 0.4984 - accuracy: 0.9494
  800/11314 [=>............................] - ETA: 6s - loss: 0.4978 - accuracy: 0.9495
  900/11314 [=>............................] - ETA: 6s - loss: 0.4975 - accuracy: 0.9495
 1000/11314 [=>............................] - ETA: 6s - loss: 0.4968 - accuracy: 0.9496
 1100/11314 [=>............................] - ETA: 6s - loss: 0.4966 - accuracy: 0.9495
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.4963 - accuracy: 0.9496
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.4961 - accuracy: 0.9496
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.4956 - accuracy: 0.9496
 1500/11314 [==>...........................] - ETA: 6s - loss: 0.4953 - accuracy: 0.9495
 1600/11314 [===>..........................] - ETA: 6s - loss: 0.4948 - accuracy: 0.9496
 1700/11314 [===>..........................] - ETA: 5s - loss: 0.4944 - accuracy: 0.9495
 1800/11314 [===>..........................] - ETA: 5s - loss: 0.4940 - accuracy: 0.9496
 1900/11314 [====>.........................] - ETA: 5s - loss: 0.4937 - accuracy: 0.9497
 2000/11314 [====>.........................] - ETA: 5s - loss: 0.4934 - accuracy: 0.9496
 2100/11314 [====>.........................] - ETA: 5s - loss: 0.4931 - accuracy: 0.9497
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.4928 - accuracy: 0.9497
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.4924 - accuracy: 0.9497
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.4921 - accuracy: 0.9497
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.4919 - accuracy: 0.9497
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.4916 - accuracy: 0.9498
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.4913 - accuracy: 0.9497
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.4911 - accuracy: 0.9497
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.4909 - accuracy: 0.9497
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.4906 - accuracy: 0.9497
 3100/11314 [=======>......................] - ETA: 5s - loss: 0.4903 - accuracy: 0.9497
 3200/11314 [=======>......................] - ETA: 5s - loss: 0.4901 - accuracy: 0.9498
 3300/11314 [=======>......................] - ETA: 4s - loss: 0.4899 - accuracy: 0.9498
 3400/11314 [========>.....................] - ETA: 4s - loss: 0.4896 - accuracy: 0.9497
 3500/11314 [========>.....................] - ETA: 4s - loss: 0.4894 - accuracy: 0.9497
 3600/11314 [========>.....................] - ETA: 4s - loss: 0.4891 - accuracy: 0.9497
 3700/11314 [========>.....................] - ETA: 4s - loss: 0.4889 - accuracy: 0.9497
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.4885 - accuracy: 0.9496
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.4882 - accuracy: 0.9496
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.4879 - accuracy: 0.9497
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.4875 - accuracy: 0.9496
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.4873 - accuracy: 0.9497
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.4870 - accuracy: 0.9496
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.4867 - accuracy: 0.9496
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.4864 - accuracy: 0.9496
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.4862 - accuracy: 0.9496
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.4859 - accuracy: 0.9496
 4800/11314 [===========>..................] - ETA: 4s - loss: 0.4856 - accuracy: 0.9496
 4900/11314 [===========>..................] - ETA: 3s - loss: 0.4853 - accuracy: 0.9497
 5000/11314 [============>.................] - ETA: 3s - loss: 0.4850 - accuracy: 0.9497
 5100/11314 [============>.................] - ETA: 3s - loss: 0.4848 - accuracy: 0.9496
 5200/11314 [============>.................] - ETA: 3s - loss: 0.4845 - accuracy: 0.9496
 5300/11314 [=============>................] - ETA: 3s - loss: 0.4842 - accuracy: 0.9497
 5400/11314 [=============>................] - ETA: 3s - loss: 0.4840 - accuracy: 0.9497
 5500/11314 [=============>................] - ETA: 3s - loss: 0.4837 - accuracy: 0.9497
 5600/11314 [=============>................] - ETA: 3s - loss: 0.4835 - accuracy: 0.9497
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.4832 - accuracy: 0.9497
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.4829 - accuracy: 0.9497
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.4826 - accuracy: 0.9497
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.4823 - accuracy: 0.9497
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.4820 - accuracy: 0.9497
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.4818 - accuracy: 0.9497
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.4815 - accuracy: 0.9497
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.4812 - accuracy: 0.9496
 6500/11314 [================>.............] - ETA: 2s - loss: 0.4809 - accuracy: 0.9497
 6600/11314 [================>.............] - ETA: 2s - loss: 0.4807 - accuracy: 0.9497
 6700/11314 [================>.............] - ETA: 2s - loss: 0.4804 - accuracy: 0.9497
 6800/11314 [=================>............] - ETA: 2s - loss: 0.4802 - accuracy: 0.9497
 6900/11314 [=================>............] - ETA: 2s - loss: 0.4799 - accuracy: 0.9497
 7000/11314 [=================>............] - ETA: 2s - loss: 0.4797 - accuracy: 0.9496
 7100/11314 [=================>............] - ETA: 2s - loss: 0.4794 - accuracy: 0.9496
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.4792 - accuracy: 0.9496
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.4789 - accuracy: 0.9496
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.4787 - accuracy: 0.9496
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.4784 - accuracy: 0.9496
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.4782 - accuracy: 0.9496
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.4779 - accuracy: 0.9496
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.4776 - accuracy: 0.9496
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.4774 - accuracy: 0.9496
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.4771 - accuracy: 0.9496
 8100/11314 [====================>.........] - ETA: 1s - loss: 0.4769 - accuracy: 0.9496
 8200/11314 [====================>.........] - ETA: 1s - loss: 0.4766 - accuracy: 0.9496
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.4764 - accuracy: 0.9496
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.4761 - accuracy: 0.9496
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.4759 - accuracy: 0.9496
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.4756 - accuracy: 0.9496
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.4753 - accuracy: 0.9496
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.4751 - accuracy: 0.9496
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.4748 - accuracy: 0.9496
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.4745 - accuracy: 0.9496
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.4743 - accuracy: 0.9496
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.4741 - accuracy: 0.9496
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.4738 - accuracy: 0.9496
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.4736 - accuracy: 0.9496
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.4733 - accuracy: 0.9496
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.4731 - accuracy: 0.9496
 9700/11314 [========================>.....] - ETA: 1s - loss: 0.4728 - accuracy: 0.9496
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.4725 - accuracy: 0.9496
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.4723 - accuracy: 0.9496
10000/11314 [=========================>....] - ETA: 0s - loss: 0.4721 - accuracy: 0.9496
10100/11314 [=========================>....] - ETA: 0s - loss: 0.4718 - accuracy: 0.9496
10200/11314 [==========================>...] - ETA: 0s - loss: 0.4716 - accuracy: 0.9496
10300/11314 [==========================>...] - ETA: 0s - loss: 0.4713 - accuracy: 0.9496
10400/11314 [==========================>...] - ETA: 0s - loss: 0.4711 - accuracy: 0.9496
10500/11314 [==========================>...] - ETA: 0s - loss: 0.4708 - accuracy: 0.9496
10600/11314 [===========================>..] - ETA: 0s - loss: 0.4706 - accuracy: 0.9496
10700/11314 [===========================>..] - ETA: 0s - loss: 0.4703 - accuracy: 0.9496
10800/11314 [===========================>..] - ETA: 0s - loss: 0.4701 - accuracy: 0.9496
10900/11314 [===========================>..] - ETA: 0s - loss: 0.4698 - accuracy: 0.9496
11000/11314 [============================>.] - ETA: 0s - loss: 0.4695 - accuracy: 0.9496
11100/11314 [============================>.] - ETA: 0s - loss: 0.4693 - accuracy: 0.9496
11200/11314 [============================>.] - ETA: 0s - loss: 0.4690 - accuracy: 0.9496
11300/11314 [============================>.] - ETA: 0s - loss: 0.4687 - accuracy: 0.9496
11314/11314 [==============================] - 8s 732us/step - loss: 0.4687 - accuracy: 0.9496 - val_loss: 0.4400 - val_accuracy: 0.9496
Epoch 4/10

  100/11314 [..............................] - ETA: 6s - loss: 0.4423 - accuracy: 0.9495
  200/11314 [..............................] - ETA: 6s - loss: 0.4401 - accuracy: 0.9511
  300/11314 [..............................] - ETA: 6s - loss: 0.4391 - accuracy: 0.9502
  400/11314 [>.............................] - ETA: 6s - loss: 0.4390 - accuracy: 0.9501
  500/11314 [>.............................] - ETA: 6s - loss: 0.4389 - accuracy: 0.9503
  600/11314 [>.............................] - ETA: 6s - loss: 0.4385 - accuracy: 0.9503
  700/11314 [>.............................] - ETA: 6s - loss: 0.4380 - accuracy: 0.9504
  800/11314 [=>............................] - ETA: 6s - loss: 0.4379 - accuracy: 0.9502
  900/11314 [=>............................] - ETA: 6s - loss: 0.4377 - accuracy: 0.9502
 1000/11314 [=>............................] - ETA: 6s - loss: 0.4377 - accuracy: 0.9502
 1100/11314 [=>............................] - ETA: 6s - loss: 0.4375 - accuracy: 0.9502
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.4375 - accuracy: 0.9500
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.4373 - accuracy: 0.9500
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.4371 - accuracy: 0.9500
 1500/11314 [==>...........................] - ETA: 6s - loss: 0.4370 - accuracy: 0.9500
 1600/11314 [===>..........................] - ETA: 6s - loss: 0.4367 - accuracy: 0.9500
 1700/11314 [===>..........................] - ETA: 5s - loss: 0.4366 - accuracy: 0.9500
 1800/11314 [===>..........................] - ETA: 5s - loss: 0.4365 - accuracy: 0.9499
 1900/11314 [====>.........................] - ETA: 5s - loss: 0.4363 - accuracy: 0.9499
 2000/11314 [====>.........................] - ETA: 5s - loss: 0.4360 - accuracy: 0.9499
 2100/11314 [====>.........................] - ETA: 5s - loss: 0.4357 - accuracy: 0.9499
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.4354 - accuracy: 0.9499
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.4352 - accuracy: 0.9498
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.4349 - accuracy: 0.9498
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.4347 - accuracy: 0.9497
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.4345 - accuracy: 0.9497
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.4343 - accuracy: 0.9497
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.4341 - accuracy: 0.9497
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.4338 - accuracy: 0.9497
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.4337 - accuracy: 0.9497
 3100/11314 [=======>......................] - ETA: 5s - loss: 0.4335 - accuracy: 0.9497
 3200/11314 [=======>......................] - ETA: 5s - loss: 0.4332 - accuracy: 0.9497
 3300/11314 [=======>......................] - ETA: 4s - loss: 0.4331 - accuracy: 0.9496
 3400/11314 [========>.....................] - ETA: 4s - loss: 0.4328 - accuracy: 0.9496
 3500/11314 [========>.....................] - ETA: 4s - loss: 0.4326 - accuracy: 0.9496
 3600/11314 [========>.....................] - ETA: 4s - loss: 0.4325 - accuracy: 0.9496
 3700/11314 [========>.....................] - ETA: 4s - loss: 0.4323 - accuracy: 0.9496
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.4320 - accuracy: 0.9496
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.4318 - accuracy: 0.9495
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.4316 - accuracy: 0.9495
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.4315 - accuracy: 0.9495
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.4312 - accuracy: 0.9495
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.4310 - accuracy: 0.9495
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.4308 - accuracy: 0.9495
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.4305 - accuracy: 0.9495
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.4303 - accuracy: 0.9495
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.4301 - accuracy: 0.9495
 4800/11314 [===========>..................] - ETA: 4s - loss: 0.4299 - accuracy: 0.9495
 4900/11314 [===========>..................] - ETA: 3s - loss: 0.4297 - accuracy: 0.9494
 5000/11314 [============>.................] - ETA: 3s - loss: 0.4295 - accuracy: 0.9494
 5100/11314 [============>.................] - ETA: 3s - loss: 0.4292 - accuracy: 0.9495
 5200/11314 [============>.................] - ETA: 3s - loss: 0.4290 - accuracy: 0.9495
 5300/11314 [=============>................] - ETA: 3s - loss: 0.4287 - accuracy: 0.9495
 5400/11314 [=============>................] - ETA: 3s - loss: 0.4285 - accuracy: 0.9495
 5500/11314 [=============>................] - ETA: 3s - loss: 0.4283 - accuracy: 0.9495
 5600/11314 [=============>................] - ETA: 3s - loss: 0.4280 - accuracy: 0.9495
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.4278 - accuracy: 0.9495
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.4277 - accuracy: 0.9495
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.4274 - accuracy: 0.9495
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.4272 - accuracy: 0.9495
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.4270 - accuracy: 0.9496
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.4267 - accuracy: 0.9496
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.4265 - accuracy: 0.9496
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.4263 - accuracy: 0.9496
 6500/11314 [================>.............] - ETA: 2s - loss: 0.4261 - accuracy: 0.9496
 6600/11314 [================>.............] - ETA: 2s - loss: 0.4259 - accuracy: 0.9496
 6700/11314 [================>.............] - ETA: 2s - loss: 0.4256 - accuracy: 0.9496
 6800/11314 [=================>............] - ETA: 2s - loss: 0.4254 - accuracy: 0.9496
 6900/11314 [=================>............] - ETA: 2s - loss: 0.4252 - accuracy: 0.9496
 7000/11314 [=================>............] - ETA: 2s - loss: 0.4250 - accuracy: 0.9496
 7100/11314 [=================>............] - ETA: 2s - loss: 0.4248 - accuracy: 0.9496
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.4246 - accuracy: 0.9496
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.4243 - accuracy: 0.9496
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.4242 - accuracy: 0.9497
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.4239 - accuracy: 0.9497
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.4238 - accuracy: 0.9497
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.4236 - accuracy: 0.9496
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.4233 - accuracy: 0.9496
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.4232 - accuracy: 0.9496
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.4230 - accuracy: 0.9496
 8100/11314 [====================>.........] - ETA: 1s - loss: 0.4228 - accuracy: 0.9496
 8200/11314 [====================>.........] - ETA: 1s - loss: 0.4226 - accuracy: 0.9496
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.4224 - accuracy: 0.9496
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.4222 - accuracy: 0.9496
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.4220 - accuracy: 0.9496
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.4218 - accuracy: 0.9496
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.4216 - accuracy: 0.9495
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.4214 - accuracy: 0.9496
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.4211 - accuracy: 0.9496
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.4209 - accuracy: 0.9496
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.4207 - accuracy: 0.9496
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.4204 - accuracy: 0.9496
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.4202 - accuracy: 0.9496
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.4200 - accuracy: 0.9496
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.4198 - accuracy: 0.9496
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.4196 - accuracy: 0.9496
 9700/11314 [========================>.....] - ETA: 0s - loss: 0.4194 - accuracy: 0.9496
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.4192 - accuracy: 0.9496
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.4190 - accuracy: 0.9496
10000/11314 [=========================>....] - ETA: 0s - loss: 0.4187 - accuracy: 0.9496
10100/11314 [=========================>....] - ETA: 0s - loss: 0.4186 - accuracy: 0.9496
10200/11314 [==========================>...] - ETA: 0s - loss: 0.4184 - accuracy: 0.9496
10300/11314 [==========================>...] - ETA: 0s - loss: 0.4181 - accuracy: 0.9496
10400/11314 [==========================>...] - ETA: 0s - loss: 0.4179 - accuracy: 0.9496
10500/11314 [==========================>...] - ETA: 0s - loss: 0.4177 - accuracy: 0.9495
10600/11314 [===========================>..] - ETA: 0s - loss: 0.4175 - accuracy: 0.9496
10700/11314 [===========================>..] - ETA: 0s - loss: 0.4173 - accuracy: 0.9496
10800/11314 [===========================>..] - ETA: 0s - loss: 0.4171 - accuracy: 0.9496
10900/11314 [===========================>..] - ETA: 0s - loss: 0.4169 - accuracy: 0.9496
11000/11314 [============================>.] - ETA: 0s - loss: 0.4167 - accuracy: 0.9496
11100/11314 [============================>.] - ETA: 0s - loss: 0.4165 - accuracy: 0.9496
11200/11314 [============================>.] - ETA: 0s - loss: 0.4163 - accuracy: 0.9496
11300/11314 [============================>.] - ETA: 0s - loss: 0.4161 - accuracy: 0.9496
11314/11314 [==============================] - 8s 725us/step - loss: 0.4161 - accuracy: 0.9496 - val_loss: 0.3931 - val_accuracy: 0.9496
Epoch 5/10

  100/11314 [..............................] - ETA: 6s - loss: 0.3951 - accuracy: 0.9495
  200/11314 [..............................] - ETA: 6s - loss: 0.3933 - accuracy: 0.9500
  300/11314 [..............................] - ETA: 6s - loss: 0.3934 - accuracy: 0.9500
  400/11314 [>.............................] - ETA: 6s - loss: 0.3929 - accuracy: 0.9499
  500/11314 [>.............................] - ETA: 6s - loss: 0.3923 - accuracy: 0.9500
  600/11314 [>.............................] - ETA: 6s - loss: 0.3924 - accuracy: 0.9497
  700/11314 [>.............................] - ETA: 6s - loss: 0.3925 - accuracy: 0.9495
  800/11314 [=>............................] - ETA: 6s - loss: 0.3921 - accuracy: 0.9497
  900/11314 [=>............................] - ETA: 6s - loss: 0.3917 - accuracy: 0.9496
 1000/11314 [=>............................] - ETA: 6s - loss: 0.3916 - accuracy: 0.9495
 1100/11314 [=>............................] - ETA: 6s - loss: 0.3914 - accuracy: 0.9496
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.3910 - accuracy: 0.9496
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.3909 - accuracy: 0.9495
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.3906 - accuracy: 0.9495
 1500/11314 [==>...........................] - ETA: 5s - loss: 0.3903 - accuracy: 0.9496
 1600/11314 [===>..........................] - ETA: 5s - loss: 0.3900 - accuracy: 0.9496
 1700/11314 [===>..........................] - ETA: 5s - loss: 0.3901 - accuracy: 0.9496
 1800/11314 [===>..........................] - ETA: 5s - loss: 0.3899 - accuracy: 0.9496
 1900/11314 [====>.........................] - ETA: 5s - loss: 0.3895 - accuracy: 0.9496
 2000/11314 [====>.........................] - ETA: 5s - loss: 0.3894 - accuracy: 0.9496
 2100/11314 [====>.........................] - ETA: 5s - loss: 0.3892 - accuracy: 0.9496
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.3891 - accuracy: 0.9496
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.3889 - accuracy: 0.9495
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.3887 - accuracy: 0.9495
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.3885 - accuracy: 0.9495
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.3884 - accuracy: 0.9494
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.3882 - accuracy: 0.9494
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.3881 - accuracy: 0.9494
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.3879 - accuracy: 0.9494
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.3877 - accuracy: 0.9494
 3100/11314 [=======>......................] - ETA: 5s - loss: 0.3876 - accuracy: 0.9494
 3200/11314 [=======>......................] - ETA: 4s - loss: 0.3874 - accuracy: 0.9494
 3300/11314 [=======>......................] - ETA: 4s - loss: 0.3872 - accuracy: 0.9494
 3400/11314 [========>.....................] - ETA: 4s - loss: 0.3871 - accuracy: 0.9494
 3500/11314 [========>.....................] - ETA: 4s - loss: 0.3869 - accuracy: 0.9494
 3600/11314 [========>.....................] - ETA: 4s - loss: 0.3867 - accuracy: 0.9494
 3700/11314 [========>.....................] - ETA: 4s - loss: 0.3865 - accuracy: 0.9494
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.3863 - accuracy: 0.9494
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.3861 - accuracy: 0.9494
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.3859 - accuracy: 0.9494
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.3858 - accuracy: 0.9494
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.3856 - accuracy: 0.9495
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.3855 - accuracy: 0.9495
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.3852 - accuracy: 0.9495
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.3851 - accuracy: 0.9495
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.3850 - accuracy: 0.9495
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.3849 - accuracy: 0.9495
 4800/11314 [===========>..................] - ETA: 3s - loss: 0.3847 - accuracy: 0.9495
 4900/11314 [===========>..................] - ETA: 3s - loss: 0.3846 - accuracy: 0.9495
 5000/11314 [============>.................] - ETA: 3s - loss: 0.3844 - accuracy: 0.9495
 5100/11314 [============>.................] - ETA: 3s - loss: 0.3842 - accuracy: 0.9495
 5200/11314 [============>.................] - ETA: 3s - loss: 0.3841 - accuracy: 0.9495
 5300/11314 [=============>................] - ETA: 3s - loss: 0.3840 - accuracy: 0.9495
 5400/11314 [=============>................] - ETA: 3s - loss: 0.3838 - accuracy: 0.9495
 5500/11314 [=============>................] - ETA: 3s - loss: 0.3836 - accuracy: 0.9495
 5600/11314 [=============>................] - ETA: 3s - loss: 0.3835 - accuracy: 0.9495
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.3833 - accuracy: 0.9495
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.3831 - accuracy: 0.9495
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.3829 - accuracy: 0.9495
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.3827 - accuracy: 0.9495
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.3826 - accuracy: 0.9495
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.3824 - accuracy: 0.9495
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.3822 - accuracy: 0.9495
 6400/11314 [===============>..............] - ETA: 2s - loss: 0.3821 - accuracy: 0.9495
 6500/11314 [================>.............] - ETA: 2s - loss: 0.3819 - accuracy: 0.9495
 6600/11314 [================>.............] - ETA: 2s - loss: 0.3818 - accuracy: 0.9495
 6700/11314 [================>.............] - ETA: 2s - loss: 0.3816 - accuracy: 0.9495
 6800/11314 [=================>............] - ETA: 2s - loss: 0.3815 - accuracy: 0.9495
 6900/11314 [=================>............] - ETA: 2s - loss: 0.3813 - accuracy: 0.9496
 7000/11314 [=================>............] - ETA: 2s - loss: 0.3811 - accuracy: 0.9495
 7100/11314 [=================>............] - ETA: 2s - loss: 0.3809 - accuracy: 0.9495
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.3808 - accuracy: 0.9495
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.3806 - accuracy: 0.9495
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.3805 - accuracy: 0.9496
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.3803 - accuracy: 0.9496
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.3801 - accuracy: 0.9496
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.3799 - accuracy: 0.9496
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.3798 - accuracy: 0.9496
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.3796 - accuracy: 0.9496
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.3794 - accuracy: 0.9495
 8100/11314 [====================>.........] - ETA: 1s - loss: 0.3793 - accuracy: 0.9495
 8200/11314 [====================>.........] - ETA: 1s - loss: 0.3791 - accuracy: 0.9495
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.3789 - accuracy: 0.9495
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.3788 - accuracy: 0.9495
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.3786 - accuracy: 0.9495
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.3785 - accuracy: 0.9495
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.3783 - accuracy: 0.9495
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.3782 - accuracy: 0.9495
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.3780 - accuracy: 0.9496
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.3778 - accuracy: 0.9496
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.3776 - accuracy: 0.9496
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.3775 - accuracy: 0.9496
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.3773 - accuracy: 0.9496
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.3772 - accuracy: 0.9496
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.3770 - accuracy: 0.9496
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.3769 - accuracy: 0.9496
 9700/11314 [========================>.....] - ETA: 0s - loss: 0.3767 - accuracy: 0.9496
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.3766 - accuracy: 0.9496
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.3764 - accuracy: 0.9496
10000/11314 [=========================>....] - ETA: 0s - loss: 0.3763 - accuracy: 0.9495
10100/11314 [=========================>....] - ETA: 0s - loss: 0.3761 - accuracy: 0.9496
10200/11314 [==========================>...] - ETA: 0s - loss: 0.3759 - accuracy: 0.9496
10300/11314 [==========================>...] - ETA: 0s - loss: 0.3758 - accuracy: 0.9495
10400/11314 [==========================>...] - ETA: 0s - loss: 0.3756 - accuracy: 0.9495
10500/11314 [==========================>...] - ETA: 0s - loss: 0.3754 - accuracy: 0.9496
10600/11314 [===========================>..] - ETA: 0s - loss: 0.3752 - accuracy: 0.9496
10700/11314 [===========================>..] - ETA: 0s - loss: 0.3751 - accuracy: 0.9496
10800/11314 [===========================>..] - ETA: 0s - loss: 0.3749 - accuracy: 0.9496
10900/11314 [===========================>..] - ETA: 0s - loss: 0.3747 - accuracy: 0.9496
11000/11314 [============================>.] - ETA: 0s - loss: 0.3746 - accuracy: 0.9496
11100/11314 [============================>.] - ETA: 0s - loss: 0.3744 - accuracy: 0.9496
11200/11314 [============================>.] - ETA: 0s - loss: 0.3742 - accuracy: 0.9496
11300/11314 [============================>.] - ETA: 0s - loss: 0.3741 - accuracy: 0.9496
11314/11314 [==============================] - 8s 721us/step - loss: 0.3740 - accuracy: 0.9496 - val_loss: 0.3557 - val_accuracy: 0.9496
Epoch 6/10

  100/11314 [..............................] - ETA: 6s - loss: 0.3559 - accuracy: 0.9495
  200/11314 [..............................] - ETA: 6s - loss: 0.3547 - accuracy: 0.9492
  300/11314 [..............................] - ETA: 6s - loss: 0.3551 - accuracy: 0.9491
  400/11314 [>.............................] - ETA: 6s - loss: 0.3550 - accuracy: 0.9496
  500/11314 [>.............................] - ETA: 6s - loss: 0.3556 - accuracy: 0.9494
  600/11314 [>.............................] - ETA: 6s - loss: 0.3555 - accuracy: 0.9496
  700/11314 [>.............................] - ETA: 6s - loss: 0.3554 - accuracy: 0.9493
  800/11314 [=>............................] - ETA: 6s - loss: 0.3558 - accuracy: 0.9491
  900/11314 [=>............................] - ETA: 6s - loss: 0.3554 - accuracy: 0.9491
 1000/11314 [=>............................] - ETA: 6s - loss: 0.3552 - accuracy: 0.9492
 1100/11314 [=>............................] - ETA: 6s - loss: 0.3553 - accuracy: 0.9490
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.3551 - accuracy: 0.9491
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.3551 - accuracy: 0.9491
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.3551 - accuracy: 0.9491
 1500/11314 [==>...........................] - ETA: 6s - loss: 0.3550 - accuracy: 0.9492
 1600/11314 [===>..........................] - ETA: 5s - loss: 0.3549 - accuracy: 0.9491
 1700/11314 [===>..........................] - ETA: 5s - loss: 0.3547 - accuracy: 0.9492
 1800/11314 [===>..........................] - ETA: 5s - loss: 0.3546 - accuracy: 0.9492
 1900/11314 [====>.........................] - ETA: 5s - loss: 0.3545 - accuracy: 0.9491
 2000/11314 [====>.........................] - ETA: 5s - loss: 0.3542 - accuracy: 0.9492
 2100/11314 [====>.........................] - ETA: 5s - loss: 0.3540 - accuracy: 0.9492
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.3538 - accuracy: 0.9492
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.3536 - accuracy: 0.9493
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.3535 - accuracy: 0.9493
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.3532 - accuracy: 0.9494
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.3531 - accuracy: 0.9494
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.3529 - accuracy: 0.9495
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.3527 - accuracy: 0.9495
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.3526 - accuracy: 0.9495
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.3525 - accuracy: 0.9494
 3100/11314 [=======>......................] - ETA: 5s - loss: 0.3524 - accuracy: 0.9494
 3200/11314 [=======>......................] - ETA: 5s - loss: 0.3522 - accuracy: 0.9494
 3300/11314 [=======>......................] - ETA: 5s - loss: 0.3520 - accuracy: 0.9494
 3400/11314 [========>.....................] - ETA: 4s - loss: 0.3519 - accuracy: 0.9494
 3500/11314 [========>.....................] - ETA: 4s - loss: 0.3516 - accuracy: 0.9494
 3600/11314 [========>.....................] - ETA: 4s - loss: 0.3514 - accuracy: 0.9494
 3700/11314 [========>.....................] - ETA: 4s - loss: 0.3512 - accuracy: 0.9494
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.3510 - accuracy: 0.9494
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.3509 - accuracy: 0.9494
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.3507 - accuracy: 0.9494
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.3506 - accuracy: 0.9494
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.3504 - accuracy: 0.9494
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.3502 - accuracy: 0.9494
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.3500 - accuracy: 0.9494
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.3499 - accuracy: 0.9494
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.3498 - accuracy: 0.9494
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.3496 - accuracy: 0.9495
 4800/11314 [===========>..................] - ETA: 4s - loss: 0.3495 - accuracy: 0.9495
 4900/11314 [===========>..................] - ETA: 4s - loss: 0.3492 - accuracy: 0.9495
 5000/11314 [============>.................] - ETA: 3s - loss: 0.3491 - accuracy: 0.9495
 5100/11314 [============>.................] - ETA: 3s - loss: 0.3489 - accuracy: 0.9495
 5200/11314 [============>.................] - ETA: 3s - loss: 0.3488 - accuracy: 0.9495
 5300/11314 [=============>................] - ETA: 3s - loss: 0.3487 - accuracy: 0.9495
 5400/11314 [=============>................] - ETA: 3s - loss: 0.3485 - accuracy: 0.9495
 5500/11314 [=============>................] - ETA: 3s - loss: 0.3484 - accuracy: 0.9495
 5600/11314 [=============>................] - ETA: 3s - loss: 0.3482 - accuracy: 0.9495
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.3481 - accuracy: 0.9495
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.3478 - accuracy: 0.9495
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.3477 - accuracy: 0.9496
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.3475 - accuracy: 0.9496
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.3474 - accuracy: 0.9496
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.3473 - accuracy: 0.9496
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.3472 - accuracy: 0.9495
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.3471 - accuracy: 0.9495
 6500/11314 [================>.............] - ETA: 3s - loss: 0.3469 - accuracy: 0.9495
 6600/11314 [================>.............] - ETA: 2s - loss: 0.3468 - accuracy: 0.9495
 6700/11314 [================>.............] - ETA: 2s - loss: 0.3466 - accuracy: 0.9495
 6800/11314 [=================>............] - ETA: 2s - loss: 0.3465 - accuracy: 0.9495
 6900/11314 [=================>............] - ETA: 2s - loss: 0.3464 - accuracy: 0.9495
 7000/11314 [=================>............] - ETA: 2s - loss: 0.3462 - accuracy: 0.9495
 7100/11314 [=================>............] - ETA: 2s - loss: 0.3461 - accuracy: 0.9495
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.3459 - accuracy: 0.9495
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.3458 - accuracy: 0.9495
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.3457 - accuracy: 0.9495
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.3456 - accuracy: 0.9495
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.3454 - accuracy: 0.9496
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.3452 - accuracy: 0.9496
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.3451 - accuracy: 0.9496
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.3450 - accuracy: 0.9496
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.3448 - accuracy: 0.9496
 8100/11314 [====================>.........] - ETA: 2s - loss: 0.3447 - accuracy: 0.9496
 8200/11314 [====================>.........] - ETA: 1s - loss: 0.3445 - accuracy: 0.9496
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.3444 - accuracy: 0.9496
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.3442 - accuracy: 0.9496
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.3441 - accuracy: 0.9496
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.3439 - accuracy: 0.9496
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.3438 - accuracy: 0.9496
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.3436 - accuracy: 0.9496
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.3435 - accuracy: 0.9496
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.3433 - accuracy: 0.9496
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.3432 - accuracy: 0.9497
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.3431 - accuracy: 0.9497
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.3429 - accuracy: 0.9497
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.3428 - accuracy: 0.9497
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.3427 - accuracy: 0.9497
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.3425 - accuracy: 0.9497
 9700/11314 [========================>.....] - ETA: 1s - loss: 0.3424 - accuracy: 0.9497
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.3423 - accuracy: 0.9497
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.3421 - accuracy: 0.9497
10000/11314 [=========================>....] - ETA: 0s - loss: 0.3420 - accuracy: 0.9497
10100/11314 [=========================>....] - ETA: 0s - loss: 0.3419 - accuracy: 0.9497
10200/11314 [==========================>...] - ETA: 0s - loss: 0.3418 - accuracy: 0.9497
10300/11314 [==========================>...] - ETA: 0s - loss: 0.3416 - accuracy: 0.9497
10400/11314 [==========================>...] - ETA: 0s - loss: 0.3415 - accuracy: 0.9497
10500/11314 [==========================>...] - ETA: 0s - loss: 0.3414 - accuracy: 0.9496
10600/11314 [===========================>..] - ETA: 0s - loss: 0.3413 - accuracy: 0.9496
10700/11314 [===========================>..] - ETA: 0s - loss: 0.3412 - accuracy: 0.9496
10800/11314 [===========================>..] - ETA: 0s - loss: 0.3411 - accuracy: 0.9496
10900/11314 [===========================>..] - ETA: 0s - loss: 0.3409 - accuracy: 0.9496
11000/11314 [============================>.] - ETA: 0s - loss: 0.3408 - accuracy: 0.9496
11100/11314 [============================>.] - ETA: 0s - loss: 0.3407 - accuracy: 0.9496
11200/11314 [============================>.] - ETA: 0s - loss: 0.3406 - accuracy: 0.9496
11300/11314 [============================>.] - ETA: 0s - loss: 0.3405 - accuracy: 0.9496
11314/11314 [==============================] - 9s 760us/step - loss: 0.3404 - accuracy: 0.9496 - val_loss: 0.3258 - val_accuracy: 0.9496
Epoch 7/10

  100/11314 [..............................] - ETA: 6s - loss: 0.3271 - accuracy: 0.9495
  200/11314 [..............................] - ETA: 6s - loss: 0.3266 - accuracy: 0.9492
  300/11314 [..............................] - ETA: 6s - loss: 0.3268 - accuracy: 0.9493
  400/11314 [>.............................] - ETA: 6s - loss: 0.3263 - accuracy: 0.9493
  500/11314 [>.............................] - ETA: 6s - loss: 0.3258 - accuracy: 0.9496
  600/11314 [>.............................] - ETA: 6s - loss: 0.3251 - accuracy: 0.9497
  700/11314 [>.............................] - ETA: 6s - loss: 0.3248 - accuracy: 0.9498
  800/11314 [=>............................] - ETA: 6s - loss: 0.3247 - accuracy: 0.9497
  900/11314 [=>............................] - ETA: 6s - loss: 0.3243 - accuracy: 0.9499
 1000/11314 [=>............................] - ETA: 6s - loss: 0.3242 - accuracy: 0.9501
 1100/11314 [=>............................] - ETA: 6s - loss: 0.3240 - accuracy: 0.9501
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.3241 - accuracy: 0.9499
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.3240 - accuracy: 0.9499
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.3239 - accuracy: 0.9498
 1500/11314 [==>...........................] - ETA: 6s - loss: 0.3239 - accuracy: 0.9497
 1600/11314 [===>..........................] - ETA: 5s - loss: 0.3236 - accuracy: 0.9497
 1700/11314 [===>..........................] - ETA: 5s - loss: 0.3234 - accuracy: 0.9498
 1800/11314 [===>..........................] - ETA: 5s - loss: 0.3234 - accuracy: 0.9498
 1900/11314 [====>.........................] - ETA: 5s - loss: 0.3233 - accuracy: 0.9498
 2000/11314 [====>.........................] - ETA: 5s - loss: 0.3232 - accuracy: 0.9497
 2100/11314 [====>.........................] - ETA: 5s - loss: 0.3232 - accuracy: 0.9497
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.3231 - accuracy: 0.9496
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.3232 - accuracy: 0.9495
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.3233 - accuracy: 0.9495
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.3232 - accuracy: 0.9495
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.3229 - accuracy: 0.9495
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.3228 - accuracy: 0.9495
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.3227 - accuracy: 0.9495
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.3227 - accuracy: 0.9494
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.3226 - accuracy: 0.9495
 3100/11314 [=======>......................] - ETA: 5s - loss: 0.3224 - accuracy: 0.9495
 3200/11314 [=======>......................] - ETA: 4s - loss: 0.3222 - accuracy: 0.9496
 3300/11314 [=======>......................] - ETA: 4s - loss: 0.3221 - accuracy: 0.9496
 3400/11314 [========>.....................] - ETA: 4s - loss: 0.3221 - accuracy: 0.9495
 3500/11314 [========>.....................] - ETA: 4s - loss: 0.3220 - accuracy: 0.9496
 3600/11314 [========>.....................] - ETA: 4s - loss: 0.3219 - accuracy: 0.9495
 3700/11314 [========>.....................] - ETA: 4s - loss: 0.3218 - accuracy: 0.9495
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.3217 - accuracy: 0.9495
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.3217 - accuracy: 0.9495
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.3216 - accuracy: 0.9495
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.3214 - accuracy: 0.9495
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.3213 - accuracy: 0.9495
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.3212 - accuracy: 0.9495
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.3210 - accuracy: 0.9495
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.3209 - accuracy: 0.9495
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.3208 - accuracy: 0.9495
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.3207 - accuracy: 0.9496
 4800/11314 [===========>..................] - ETA: 4s - loss: 0.3205 - accuracy: 0.9496
 4900/11314 [===========>..................] - ETA: 3s - loss: 0.3204 - accuracy: 0.9496
 5000/11314 [============>.................] - ETA: 3s - loss: 0.3203 - accuracy: 0.9496
 5100/11314 [============>.................] - ETA: 3s - loss: 0.3201 - accuracy: 0.9496
 5200/11314 [============>.................] - ETA: 3s - loss: 0.3201 - accuracy: 0.9496
 5300/11314 [=============>................] - ETA: 3s - loss: 0.3200 - accuracy: 0.9496
 5400/11314 [=============>................] - ETA: 3s - loss: 0.3198 - accuracy: 0.9496
 5500/11314 [=============>................] - ETA: 3s - loss: 0.3197 - accuracy: 0.9496
 5600/11314 [=============>................] - ETA: 3s - loss: 0.3197 - accuracy: 0.9496
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.3196 - accuracy: 0.9497
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.3195 - accuracy: 0.9497
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.3193 - accuracy: 0.9497
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.3192 - accuracy: 0.9497
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.3191 - accuracy: 0.9497
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.3189 - accuracy: 0.9497
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.3187 - accuracy: 0.9497
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.3186 - accuracy: 0.9498
 6500/11314 [================>.............] - ETA: 2s - loss: 0.3185 - accuracy: 0.9497
 6600/11314 [================>.............] - ETA: 2s - loss: 0.3184 - accuracy: 0.9498
 6700/11314 [================>.............] - ETA: 2s - loss: 0.3183 - accuracy: 0.9498
 6800/11314 [=================>............] - ETA: 2s - loss: 0.3182 - accuracy: 0.9497
 6900/11314 [=================>............] - ETA: 2s - loss: 0.3181 - accuracy: 0.9497
 7000/11314 [=================>............] - ETA: 2s - loss: 0.3179 - accuracy: 0.9498
 7100/11314 [=================>............] - ETA: 2s - loss: 0.3178 - accuracy: 0.9498
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.3177 - accuracy: 0.9498
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.3176 - accuracy: 0.9497
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.3175 - accuracy: 0.9497
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.3174 - accuracy: 0.9497
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.3172 - accuracy: 0.9497
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.3171 - accuracy: 0.9497
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.3170 - accuracy: 0.9497
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.3169 - accuracy: 0.9497
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.3168 - accuracy: 0.9497
 8100/11314 [====================>.........] - ETA: 1s - loss: 0.3167 - accuracy: 0.9497
 8200/11314 [====================>.........] - ETA: 1s - loss: 0.3166 - accuracy: 0.9497
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.3165 - accuracy: 0.9497
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.3164 - accuracy: 0.9496
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.3164 - accuracy: 0.9496
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.3162 - accuracy: 0.9496
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.3162 - accuracy: 0.9496
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.3161 - accuracy: 0.9496
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.3160 - accuracy: 0.9496
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.3158 - accuracy: 0.9496
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.3157 - accuracy: 0.9496
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.3156 - accuracy: 0.9497
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.3155 - accuracy: 0.9496
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.3155 - accuracy: 0.9496
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.3154 - accuracy: 0.9496
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.3152 - accuracy: 0.9497
 9700/11314 [========================>.....] - ETA: 0s - loss: 0.3151 - accuracy: 0.9496
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.3150 - accuracy: 0.9496
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.3149 - accuracy: 0.9496
10000/11314 [=========================>....] - ETA: 0s - loss: 0.3148 - accuracy: 0.9496
10100/11314 [=========================>....] - ETA: 0s - loss: 0.3147 - accuracy: 0.9496
10200/11314 [==========================>...] - ETA: 0s - loss: 0.3146 - accuracy: 0.9496
10300/11314 [==========================>...] - ETA: 0s - loss: 0.3145 - accuracy: 0.9496
10400/11314 [==========================>...] - ETA: 0s - loss: 0.3144 - accuracy: 0.9496
10500/11314 [==========================>...] - ETA: 0s - loss: 0.3143 - accuracy: 0.9496
10600/11314 [===========================>..] - ETA: 0s - loss: 0.3142 - accuracy: 0.9497
10700/11314 [===========================>..] - ETA: 0s - loss: 0.3141 - accuracy: 0.9496
10800/11314 [===========================>..] - ETA: 0s - loss: 0.3140 - accuracy: 0.9496
10900/11314 [===========================>..] - ETA: 0s - loss: 0.3139 - accuracy: 0.9496
11000/11314 [============================>.] - ETA: 0s - loss: 0.3138 - accuracy: 0.9496
11100/11314 [============================>.] - ETA: 0s - loss: 0.3137 - accuracy: 0.9496
11200/11314 [============================>.] - ETA: 0s - loss: 0.3136 - accuracy: 0.9496
11300/11314 [============================>.] - ETA: 0s - loss: 0.3135 - accuracy: 0.9496
11314/11314 [==============================] - 8s 739us/step - loss: 0.3135 - accuracy: 0.9496 - val_loss: 0.3018 - val_accuracy: 0.9496
Epoch 8/10

  100/11314 [..............................] - ETA: 6s - loss: 0.3028 - accuracy: 0.9495
  200/11314 [..............................] - ETA: 6s - loss: 0.3029 - accuracy: 0.9492
  300/11314 [..............................] - ETA: 6s - loss: 0.3026 - accuracy: 0.9496
  400/11314 [>.............................] - ETA: 6s - loss: 0.3016 - accuracy: 0.9499
  500/11314 [>.............................] - ETA: 6s - loss: 0.3004 - accuracy: 0.9503
  600/11314 [>.............................] - ETA: 6s - loss: 0.3009 - accuracy: 0.9498
  700/11314 [>.............................] - ETA: 6s - loss: 0.3014 - accuracy: 0.9496
  800/11314 [=>............................] - ETA: 6s - loss: 0.3016 - accuracy: 0.9495
  900/11314 [=>............................] - ETA: 6s - loss: 0.3009 - accuracy: 0.9498
 1000/11314 [=>............................] - ETA: 6s - loss: 0.3007 - accuracy: 0.9499
 1100/11314 [=>............................] - ETA: 6s - loss: 0.3007 - accuracy: 0.9498
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.3008 - accuracy: 0.9496
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.3007 - accuracy: 0.9496
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.3010 - accuracy: 0.9495
 1500/11314 [==>...........................] - ETA: 6s - loss: 0.3011 - accuracy: 0.9495
 1600/11314 [===>..........................] - ETA: 6s - loss: 0.3012 - accuracy: 0.9494
 1700/11314 [===>..........................] - ETA: 6s - loss: 0.3011 - accuracy: 0.9494
 1800/11314 [===>..........................] - ETA: 6s - loss: 0.3010 - accuracy: 0.9494
 1900/11314 [====>.........................] - ETA: 6s - loss: 0.3008 - accuracy: 0.9495
 2000/11314 [====>.........................] - ETA: 6s - loss: 0.3007 - accuracy: 0.9494
 2100/11314 [====>.........................] - ETA: 5s - loss: 0.3006 - accuracy: 0.9495
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.3006 - accuracy: 0.9495
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.3005 - accuracy: 0.9495
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.3003 - accuracy: 0.9495
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.3001 - accuracy: 0.9495
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.3000 - accuracy: 0.9495
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.2999 - accuracy: 0.9496
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.2997 - accuracy: 0.9495
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.2995 - accuracy: 0.9496
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.2994 - accuracy: 0.9496
 3100/11314 [=======>......................] - ETA: 5s - loss: 0.2992 - accuracy: 0.9496
 3200/11314 [=======>......................] - ETA: 5s - loss: 0.2991 - accuracy: 0.9496
 3300/11314 [=======>......................] - ETA: 5s - loss: 0.2990 - accuracy: 0.9496
 3400/11314 [========>.....................] - ETA: 5s - loss: 0.2989 - accuracy: 0.9496
 3500/11314 [========>.....................] - ETA: 5s - loss: 0.2988 - accuracy: 0.9496
 3600/11314 [========>.....................] - ETA: 4s - loss: 0.2986 - accuracy: 0.9496
 3700/11314 [========>.....................] - ETA: 4s - loss: 0.2985 - accuracy: 0.9496
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.2985 - accuracy: 0.9495
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.2984 - accuracy: 0.9495
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.2983 - accuracy: 0.9496
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.2982 - accuracy: 0.9496
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.2981 - accuracy: 0.9496
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.2980 - accuracy: 0.9496
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.2979 - accuracy: 0.9496
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.2979 - accuracy: 0.9496
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.2978 - accuracy: 0.9496
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.2978 - accuracy: 0.9496
 4800/11314 [===========>..................] - ETA: 4s - loss: 0.2977 - accuracy: 0.9496
 4900/11314 [===========>..................] - ETA: 4s - loss: 0.2976 - accuracy: 0.9496
 5000/11314 [============>.................] - ETA: 4s - loss: 0.2975 - accuracy: 0.9496
 5100/11314 [============>.................] - ETA: 4s - loss: 0.2974 - accuracy: 0.9496
 5200/11314 [============>.................] - ETA: 3s - loss: 0.2973 - accuracy: 0.9496
 5300/11314 [=============>................] - ETA: 3s - loss: 0.2972 - accuracy: 0.9496
 5400/11314 [=============>................] - ETA: 3s - loss: 0.2970 - accuracy: 0.9496
 5500/11314 [=============>................] - ETA: 3s - loss: 0.2969 - accuracy: 0.9496
 5600/11314 [=============>................] - ETA: 3s - loss: 0.2969 - accuracy: 0.9496
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.2967 - accuracy: 0.9496
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.2966 - accuracy: 0.9496
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.2966 - accuracy: 0.9496
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.2965 - accuracy: 0.9496
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.2964 - accuracy: 0.9496
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.2962 - accuracy: 0.9497
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.2961 - accuracy: 0.9497
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.2960 - accuracy: 0.9497
 6500/11314 [================>.............] - ETA: 3s - loss: 0.2960 - accuracy: 0.9497
 6600/11314 [================>.............] - ETA: 3s - loss: 0.2959 - accuracy: 0.9497
 6700/11314 [================>.............] - ETA: 2s - loss: 0.2958 - accuracy: 0.9497
 6800/11314 [=================>............] - ETA: 2s - loss: 0.2957 - accuracy: 0.9497
 6900/11314 [=================>............] - ETA: 2s - loss: 0.2956 - accuracy: 0.9497
 7000/11314 [=================>............] - ETA: 2s - loss: 0.2956 - accuracy: 0.9497
 7100/11314 [=================>............] - ETA: 2s - loss: 0.2955 - accuracy: 0.9497
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.2954 - accuracy: 0.9497
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.2953 - accuracy: 0.9497
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.2951 - accuracy: 0.9497
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.2951 - accuracy: 0.9497
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.2950 - accuracy: 0.9497
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.2949 - accuracy: 0.9497
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.2948 - accuracy: 0.9497
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.2947 - accuracy: 0.9497
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.2946 - accuracy: 0.9497
 8100/11314 [====================>.........] - ETA: 2s - loss: 0.2945 - accuracy: 0.9497
 8200/11314 [====================>.........] - ETA: 2s - loss: 0.2944 - accuracy: 0.9497
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.2944 - accuracy: 0.9497
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.2943 - accuracy: 0.9497
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.2942 - accuracy: 0.9497
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.2940 - accuracy: 0.9497
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.2940 - accuracy: 0.9497
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.2939 - accuracy: 0.9497
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.2938 - accuracy: 0.9497
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.2937 - accuracy: 0.9497
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.2937 - accuracy: 0.9497
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.2936 - accuracy: 0.9497
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.2935 - accuracy: 0.9497
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.2934 - accuracy: 0.9497
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.2933 - accuracy: 0.9497
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.2932 - accuracy: 0.9497
 9700/11314 [========================>.....] - ETA: 1s - loss: 0.2932 - accuracy: 0.9496
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.2931 - accuracy: 0.9497
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.2930 - accuracy: 0.9497
10000/11314 [=========================>....] - ETA: 0s - loss: 0.2929 - accuracy: 0.9497
10100/11314 [=========================>....] - ETA: 0s - loss: 0.2928 - accuracy: 0.9496
10200/11314 [==========================>...] - ETA: 0s - loss: 0.2928 - accuracy: 0.9496
10300/11314 [==========================>...] - ETA: 0s - loss: 0.2927 - accuracy: 0.9496
10400/11314 [==========================>...] - ETA: 0s - loss: 0.2927 - accuracy: 0.9496
10500/11314 [==========================>...] - ETA: 0s - loss: 0.2926 - accuracy: 0.9496
10600/11314 [===========================>..] - ETA: 0s - loss: 0.2925 - accuracy: 0.9496
10700/11314 [===========================>..] - ETA: 0s - loss: 0.2924 - accuracy: 0.9496
10800/11314 [===========================>..] - ETA: 0s - loss: 0.2923 - accuracy: 0.9496
10900/11314 [===========================>..] - ETA: 0s - loss: 0.2922 - accuracy: 0.9496
11000/11314 [============================>.] - ETA: 0s - loss: 0.2922 - accuracy: 0.9496
11100/11314 [============================>.] - ETA: 0s - loss: 0.2921 - accuracy: 0.9496
11200/11314 [============================>.] - ETA: 0s - loss: 0.2920 - accuracy: 0.9496
11300/11314 [============================>.] - ETA: 0s - loss: 0.2919 - accuracy: 0.9496
11314/11314 [==============================] - 9s 762us/step - loss: 0.2919 - accuracy: 0.9496 - val_loss: 0.2824 - val_accuracy: 0.9496
Epoch 9/10

  100/11314 [..............................] - ETA: 7s - loss: 0.2813 - accuracy: 0.9505
  200/11314 [..............................] - ETA: 6s - loss: 0.2814 - accuracy: 0.9508
  300/11314 [..............................] - ETA: 6s - loss: 0.2825 - accuracy: 0.9498
  400/11314 [>.............................] - ETA: 6s - loss: 0.2826 - accuracy: 0.9495
  500/11314 [>.............................] - ETA: 6s - loss: 0.2818 - accuracy: 0.9498
  600/11314 [>.............................] - ETA: 6s - loss: 0.2815 - accuracy: 0.9500
  700/11314 [>.............................] - ETA: 6s - loss: 0.2817 - accuracy: 0.9498
  800/11314 [=>............................] - ETA: 6s - loss: 0.2819 - accuracy: 0.9499
  900/11314 [=>............................] - ETA: 6s - loss: 0.2818 - accuracy: 0.9498
 1000/11314 [=>............................] - ETA: 6s - loss: 0.2815 - accuracy: 0.9500
 1100/11314 [=>............................] - ETA: 6s - loss: 0.2818 - accuracy: 0.9499
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.2815 - accuracy: 0.9499
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.2817 - accuracy: 0.9497
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.2816 - accuracy: 0.9497
 1500/11314 [==>...........................] - ETA: 6s - loss: 0.2815 - accuracy: 0.9497
 1600/11314 [===>..........................] - ETA: 6s - loss: 0.2815 - accuracy: 0.9497
 1700/11314 [===>..........................] - ETA: 6s - loss: 0.2815 - accuracy: 0.9496
 1800/11314 [===>..........................] - ETA: 6s - loss: 0.2814 - accuracy: 0.9496
 1900/11314 [====>.........................] - ETA: 6s - loss: 0.2812 - accuracy: 0.9496
 2000/11314 [====>.........................] - ETA: 5s - loss: 0.2811 - accuracy: 0.9496
 2100/11314 [====>.........................] - ETA: 5s - loss: 0.2811 - accuracy: 0.9497
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.2810 - accuracy: 0.9497
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.2809 - accuracy: 0.9497
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.2808 - accuracy: 0.9497
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.2807 - accuracy: 0.9497
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.2804 - accuracy: 0.9498
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.2804 - accuracy: 0.9497
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.2805 - accuracy: 0.9497
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.2804 - accuracy: 0.9497
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.2803 - accuracy: 0.9497
 3100/11314 [=======>......................] - ETA: 5s - loss: 0.2802 - accuracy: 0.9497
 3200/11314 [=======>......................] - ETA: 5s - loss: 0.2801 - accuracy: 0.9497
 3300/11314 [=======>......................] - ETA: 5s - loss: 0.2800 - accuracy: 0.9497
 3400/11314 [========>.....................] - ETA: 4s - loss: 0.2800 - accuracy: 0.9497
 3500/11314 [========>.....................] - ETA: 4s - loss: 0.2799 - accuracy: 0.9496
 3600/11314 [========>.....................] - ETA: 4s - loss: 0.2799 - accuracy: 0.9496
 3700/11314 [========>.....................] - ETA: 4s - loss: 0.2797 - accuracy: 0.9496
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.2797 - accuracy: 0.9496
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.2797 - accuracy: 0.9496
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.2796 - accuracy: 0.9496
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.2795 - accuracy: 0.9496
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.2795 - accuracy: 0.9496
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.2794 - accuracy: 0.9496
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.2793 - accuracy: 0.9496
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.2793 - accuracy: 0.9496
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.2793 - accuracy: 0.9495
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.2791 - accuracy: 0.9496
 4800/11314 [===========>..................] - ETA: 4s - loss: 0.2791 - accuracy: 0.9496
 4900/11314 [===========>..................] - ETA: 4s - loss: 0.2790 - accuracy: 0.9496
 5000/11314 [============>.................] - ETA: 4s - loss: 0.2790 - accuracy: 0.9495
 5100/11314 [============>.................] - ETA: 3s - loss: 0.2789 - accuracy: 0.9495
 5200/11314 [============>.................] - ETA: 3s - loss: 0.2787 - accuracy: 0.9495
 5300/11314 [=============>................] - ETA: 3s - loss: 0.2787 - accuracy: 0.9495
 5400/11314 [=============>................] - ETA: 3s - loss: 0.2786 - accuracy: 0.9495
 5500/11314 [=============>................] - ETA: 3s - loss: 0.2785 - accuracy: 0.9495
 5600/11314 [=============>................] - ETA: 3s - loss: 0.2784 - accuracy: 0.9495
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.2784 - accuracy: 0.9495
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.2783 - accuracy: 0.9495
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.2782 - accuracy: 0.9495
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.2781 - accuracy: 0.9496
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.2780 - accuracy: 0.9496
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.2780 - accuracy: 0.9495
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.2779 - accuracy: 0.9495
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.2778 - accuracy: 0.9495
 6500/11314 [================>.............] - ETA: 3s - loss: 0.2778 - accuracy: 0.9496
 6600/11314 [================>.............] - ETA: 2s - loss: 0.2777 - accuracy: 0.9496
 6700/11314 [================>.............] - ETA: 2s - loss: 0.2776 - accuracy: 0.9496
 6800/11314 [=================>............] - ETA: 2s - loss: 0.2775 - accuracy: 0.9496
 6900/11314 [=================>............] - ETA: 2s - loss: 0.2775 - accuracy: 0.9496
 7000/11314 [=================>............] - ETA: 2s - loss: 0.2774 - accuracy: 0.9496
 7100/11314 [=================>............] - ETA: 2s - loss: 0.2774 - accuracy: 0.9496
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.2773 - accuracy: 0.9496
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.2772 - accuracy: 0.9496
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.2772 - accuracy: 0.9496
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.2771 - accuracy: 0.9496
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.2771 - accuracy: 0.9496
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.2770 - accuracy: 0.9496
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.2769 - accuracy: 0.9496
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.2768 - accuracy: 0.9496
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.2768 - accuracy: 0.9496
 8100/11314 [====================>.........] - ETA: 2s - loss: 0.2768 - accuracy: 0.9496
 8200/11314 [====================>.........] - ETA: 1s - loss: 0.2767 - accuracy: 0.9496
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.2766 - accuracy: 0.9496
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.2765 - accuracy: 0.9496
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.2764 - accuracy: 0.9496
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.2763 - accuracy: 0.9496
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.2762 - accuracy: 0.9496
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.2762 - accuracy: 0.9496
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.2762 - accuracy: 0.9496
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.2760 - accuracy: 0.9496
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.2760 - accuracy: 0.9496
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.2760 - accuracy: 0.9496
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.2759 - accuracy: 0.9496
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.2758 - accuracy: 0.9496
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.2757 - accuracy: 0.9496
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.2757 - accuracy: 0.9496
 9700/11314 [========================>.....] - ETA: 1s - loss: 0.2756 - accuracy: 0.9496
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.2756 - accuracy: 0.9496
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.2755 - accuracy: 0.9496
10000/11314 [=========================>....] - ETA: 0s - loss: 0.2754 - accuracy: 0.9496
10100/11314 [=========================>....] - ETA: 0s - loss: 0.2753 - accuracy: 0.9496
10200/11314 [==========================>...] - ETA: 0s - loss: 0.2753 - accuracy: 0.9496
10300/11314 [==========================>...] - ETA: 0s - loss: 0.2752 - accuracy: 0.9496
10400/11314 [==========================>...] - ETA: 0s - loss: 0.2751 - accuracy: 0.9496
10500/11314 [==========================>...] - ETA: 0s - loss: 0.2751 - accuracy: 0.9496
10600/11314 [===========================>..] - ETA: 0s - loss: 0.2750 - accuracy: 0.9496
10700/11314 [===========================>..] - ETA: 0s - loss: 0.2749 - accuracy: 0.9496
10800/11314 [===========================>..] - ETA: 0s - loss: 0.2748 - accuracy: 0.9496
10900/11314 [===========================>..] - ETA: 0s - loss: 0.2748 - accuracy: 0.9496
11000/11314 [============================>.] - ETA: 0s - loss: 0.2747 - accuracy: 0.9496
11100/11314 [============================>.] - ETA: 0s - loss: 0.2746 - accuracy: 0.9496
11200/11314 [============================>.] - ETA: 0s - loss: 0.2745 - accuracy: 0.9496
11300/11314 [============================>.] - ETA: 0s - loss: 0.2745 - accuracy: 0.9496
11314/11314 [==============================] - 9s 756us/step - loss: 0.2745 - accuracy: 0.9496 - val_loss: 0.2668 - val_accuracy: 0.9496
Epoch 10/10

  100/11314 [..............................] - ETA: 6s - loss: 0.2685 - accuracy: 0.9489
  200/11314 [..............................] - ETA: 6s - loss: 0.2683 - accuracy: 0.9489
  300/11314 [..............................] - ETA: 6s - loss: 0.2683 - accuracy: 0.9493
  400/11314 [>.............................] - ETA: 6s - loss: 0.2683 - accuracy: 0.9492
  500/11314 [>.............................] - ETA: 6s - loss: 0.2678 - accuracy: 0.9493
  600/11314 [>.............................] - ETA: 6s - loss: 0.2677 - accuracy: 0.9495
  700/11314 [>.............................] - ETA: 6s - loss: 0.2670 - accuracy: 0.9497
  800/11314 [=>............................] - ETA: 6s - loss: 0.2669 - accuracy: 0.9496
  900/11314 [=>............................] - ETA: 6s - loss: 0.2668 - accuracy: 0.9496
 1000/11314 [=>............................] - ETA: 6s - loss: 0.2663 - accuracy: 0.9498
 1100/11314 [=>............................] - ETA: 6s - loss: 0.2659 - accuracy: 0.9500
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.2660 - accuracy: 0.9500
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.2660 - accuracy: 0.9500
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.2659 - accuracy: 0.9500
 1500/11314 [==>...........................] - ETA: 6s - loss: 0.2660 - accuracy: 0.9499
 1600/11314 [===>..........................] - ETA: 6s - loss: 0.2658 - accuracy: 0.9499
 1700/11314 [===>..........................] - ETA: 6s - loss: 0.2658 - accuracy: 0.9498
 1800/11314 [===>..........................] - ETA: 6s - loss: 0.2658 - accuracy: 0.9498
 1900/11314 [====>.........................] - ETA: 6s - loss: 0.2660 - accuracy: 0.9497
 2000/11314 [====>.........................] - ETA: 6s - loss: 0.2659 - accuracy: 0.9497
 2100/11314 [====>.........................] - ETA: 5s - loss: 0.2660 - accuracy: 0.9496
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.2661 - accuracy: 0.9496
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.2659 - accuracy: 0.9496
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.2656 - accuracy: 0.9497
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.2655 - accuracy: 0.9498
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.2654 - accuracy: 0.9498
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.2655 - accuracy: 0.9497
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.2653 - accuracy: 0.9497
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.2654 - accuracy: 0.9497
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.2653 - accuracy: 0.9496
 3100/11314 [=======>......................] - ETA: 5s - loss: 0.2653 - accuracy: 0.9496
 3200/11314 [=======>......................] - ETA: 5s - loss: 0.2651 - accuracy: 0.9496
 3300/11314 [=======>......................] - ETA: 5s - loss: 0.2651 - accuracy: 0.9496
 3400/11314 [========>.....................] - ETA: 5s - loss: 0.2650 - accuracy: 0.9496
 3500/11314 [========>.....................] - ETA: 5s - loss: 0.2649 - accuracy: 0.9497
 3600/11314 [========>.....................] - ETA: 5s - loss: 0.2649 - accuracy: 0.9496
 3700/11314 [========>.....................] - ETA: 4s - loss: 0.2648 - accuracy: 0.9497
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.2647 - accuracy: 0.9497
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.2646 - accuracy: 0.9497
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.2645 - accuracy: 0.9497
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.2645 - accuracy: 0.9497
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.2645 - accuracy: 0.9497
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.2644 - accuracy: 0.9497
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.2644 - accuracy: 0.9497
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.2643 - accuracy: 0.9497
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.2642 - accuracy: 0.9497
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.2641 - accuracy: 0.9497
 4800/11314 [===========>..................] - ETA: 4s - loss: 0.2640 - accuracy: 0.9497
 4900/11314 [===========>..................] - ETA: 4s - loss: 0.2640 - accuracy: 0.9497
 5000/11314 [============>.................] - ETA: 4s - loss: 0.2639 - accuracy: 0.9497
 5100/11314 [============>.................] - ETA: 4s - loss: 0.2638 - accuracy: 0.9497
 5200/11314 [============>.................] - ETA: 3s - loss: 0.2638 - accuracy: 0.9497
 5300/11314 [=============>................] - ETA: 3s - loss: 0.2638 - accuracy: 0.9496
 5400/11314 [=============>................] - ETA: 3s - loss: 0.2637 - accuracy: 0.9496
 5500/11314 [=============>................] - ETA: 3s - loss: 0.2637 - accuracy: 0.9496
 5600/11314 [=============>................] - ETA: 3s - loss: 0.2636 - accuracy: 0.9496
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.2636 - accuracy: 0.9496
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.2636 - accuracy: 0.9496
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.2635 - accuracy: 0.9496
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.2634 - accuracy: 0.9496
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.2634 - accuracy: 0.9496
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.2633 - accuracy: 0.9496
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.2633 - accuracy: 0.9496
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.2633 - accuracy: 0.9496
 6500/11314 [================>.............] - ETA: 3s - loss: 0.2633 - accuracy: 0.9496
 6600/11314 [================>.............] - ETA: 3s - loss: 0.2632 - accuracy: 0.9496
 6700/11314 [================>.............] - ETA: 2s - loss: 0.2631 - accuracy: 0.9496
 6800/11314 [=================>............] - ETA: 2s - loss: 0.2631 - accuracy: 0.9496
 6900/11314 [=================>............] - ETA: 2s - loss: 0.2631 - accuracy: 0.9496
 7000/11314 [=================>............] - ETA: 2s - loss: 0.2630 - accuracy: 0.9496
 7100/11314 [=================>............] - ETA: 2s - loss: 0.2629 - accuracy: 0.9496
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.2628 - accuracy: 0.9496
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.2627 - accuracy: 0.9496
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.2626 - accuracy: 0.9496
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.2626 - accuracy: 0.9496
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.2625 - accuracy: 0.9496
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.2624 - accuracy: 0.9496
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.2623 - accuracy: 0.9496
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.2622 - accuracy: 0.9496
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.2622 - accuracy: 0.9496
 8100/11314 [====================>.........] - ETA: 2s - loss: 0.2622 - accuracy: 0.9496
 8200/11314 [====================>.........] - ETA: 1s - loss: 0.2621 - accuracy: 0.9496
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.2621 - accuracy: 0.9496
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.2620 - accuracy: 0.9496
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.2620 - accuracy: 0.9496
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.2619 - accuracy: 0.9496
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.2618 - accuracy: 0.9496
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.2618 - accuracy: 0.9496
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.2617 - accuracy: 0.9496
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.2617 - accuracy: 0.9496
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.2616 - accuracy: 0.9496
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.2615 - accuracy: 0.9496
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.2615 - accuracy: 0.9496
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.2614 - accuracy: 0.9496
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.2613 - accuracy: 0.9496
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.2613 - accuracy: 0.9496
 9700/11314 [========================>.....] - ETA: 1s - loss: 0.2612 - accuracy: 0.9496
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.2612 - accuracy: 0.9496
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.2611 - accuracy: 0.9496
10000/11314 [=========================>....] - ETA: 0s - loss: 0.2610 - accuracy: 0.9496
10100/11314 [=========================>....] - ETA: 0s - loss: 0.2610 - accuracy: 0.9496
10200/11314 [==========================>...] - ETA: 0s - loss: 0.2609 - accuracy: 0.9496
10300/11314 [==========================>...] - ETA: 0s - loss: 0.2609 - accuracy: 0.9496
10400/11314 [==========================>...] - ETA: 0s - loss: 0.2609 - accuracy: 0.9496
10500/11314 [==========================>...] - ETA: 0s - loss: 0.2609 - accuracy: 0.9496
10600/11314 [===========================>..] - ETA: 0s - loss: 0.2608 - accuracy: 0.9496
10700/11314 [===========================>..] - ETA: 0s - loss: 0.2607 - accuracy: 0.9496
10800/11314 [===========================>..] - ETA: 0s - loss: 0.2607 - accuracy: 0.9496
10900/11314 [===========================>..] - ETA: 0s - loss: 0.2607 - accuracy: 0.9496
11000/11314 [============================>.] - ETA: 0s - loss: 0.2606 - accuracy: 0.9496
11100/11314 [============================>.] - ETA: 0s - loss: 0.2605 - accuracy: 0.9496
11200/11314 [============================>.] - ETA: 0s - loss: 0.2605 - accuracy: 0.9496
11300/11314 [============================>.] - ETA: 0s - loss: 0.2604 - accuracy: 0.9496
11314/11314 [==============================] - 8s 743us/step - loss: 0.2604 - accuracy: 0.9496 - val_loss: 0.2542 - val_accuracy: 0.9496
	=====> Test the model: model.predict()
	Dataset: TWENTY_NEWS_GROUPS
	Algorithm: Deep Learning using Keras 2 (KerasDL2)
	Training Loss: 0.2542
	Training accuracy score: 94.96%
	Test Loss: 0.2542
	Test Accuracy: 94.96%
	Training Time: 85.0631
	Test Time: 2.8639


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
	It took 24.70145320892334 seconds


Applying NLTK feature extraction: X_test
	==> Replace unused UNICODE characters. 
	==> Text in lower case. 
	==> Apply Lemmatize using WordNet's built-in morphy function. Returns the input word unchanged if it cannot be found in WordNet 
	==> Remove English stop words
	Done: NLTK feature extraction (replace unused UNICODE characters, text in lower case, apply Lemmatize using WordNet, remove English stop words)  finished!
	It took 23.980381965637207 seconds

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

  100/25000 [..............................] - ETA: 2:15 - loss: 0.6859 - accuracy: 0.5600
  200/25000 [..............................] - ETA: 1:15 - loss: 0.6883 - accuracy: 0.5500
  300/25000 [..............................] - ETA: 55s - loss: 0.6820 - accuracy: 0.5767 
  400/25000 [..............................] - ETA: 45s - loss: 0.6884 - accuracy: 0.5500
  500/25000 [..............................] - ETA: 39s - loss: 0.6941 - accuracy: 0.5260
  600/25000 [..............................] - ETA: 35s - loss: 0.6990 - accuracy: 0.5050
  700/25000 [..............................] - ETA: 32s - loss: 0.7004 - accuracy: 0.4986
  800/25000 [..............................] - ETA: 29s - loss: 0.7017 - accuracy: 0.4925
  900/25000 [>.............................] - ETA: 28s - loss: 0.7012 - accuracy: 0.4944
 1000/25000 [>.............................] - ETA: 26s - loss: 0.7002 - accuracy: 0.4980
 1100/25000 [>.............................] - ETA: 25s - loss: 0.7007 - accuracy: 0.4955
 1200/25000 [>.............................] - ETA: 24s - loss: 0.7011 - accuracy: 0.4925
 1300/25000 [>.............................] - ETA: 23s - loss: 0.7007 - accuracy: 0.4946
 1400/25000 [>.............................] - ETA: 23s - loss: 0.6997 - accuracy: 0.4986
 1500/25000 [>.............................] - ETA: 22s - loss: 0.6998 - accuracy: 0.4980
 1600/25000 [>.............................] - ETA: 21s - loss: 0.7002 - accuracy: 0.4950
 1700/25000 [=>............................] - ETA: 21s - loss: 0.6998 - accuracy: 0.4965
 1800/25000 [=>............................] - ETA: 20s - loss: 0.6998 - accuracy: 0.4956
 1900/25000 [=>............................] - ETA: 20s - loss: 0.6999 - accuracy: 0.4942
 2000/25000 [=>............................] - ETA: 20s - loss: 0.6998 - accuracy: 0.4940
 2100/25000 [=>............................] - ETA: 19s - loss: 0.6990 - accuracy: 0.4981
 2200/25000 [=>............................] - ETA: 19s - loss: 0.6986 - accuracy: 0.5009
 2300/25000 [=>............................] - ETA: 19s - loss: 0.6982 - accuracy: 0.5026
 2400/25000 [=>............................] - ETA: 18s - loss: 0.6979 - accuracy: 0.5033
 2500/25000 [==>...........................] - ETA: 18s - loss: 0.6979 - accuracy: 0.5024
 2600/25000 [==>...........................] - ETA: 18s - loss: 0.6973 - accuracy: 0.5050
 2700/25000 [==>...........................] - ETA: 18s - loss: 0.6976 - accuracy: 0.5022
 2800/25000 [==>...........................] - ETA: 17s - loss: 0.6976 - accuracy: 0.5007
 2900/25000 [==>...........................] - ETA: 17s - loss: 0.6975 - accuracy: 0.5014
 3000/25000 [==>...........................] - ETA: 17s - loss: 0.6974 - accuracy: 0.5007
 3100/25000 [==>...........................] - ETA: 17s - loss: 0.6974 - accuracy: 0.4997
 3200/25000 [==>...........................] - ETA: 17s - loss: 0.6972 - accuracy: 0.5006
 3300/25000 [==>...........................] - ETA: 16s - loss: 0.6973 - accuracy: 0.5003
 3400/25000 [===>..........................] - ETA: 16s - loss: 0.6973 - accuracy: 0.4991
 3500/25000 [===>..........................] - ETA: 16s - loss: 0.6972 - accuracy: 0.4983
 3600/25000 [===>..........................] - ETA: 16s - loss: 0.6972 - accuracy: 0.4978
 3700/25000 [===>..........................] - ETA: 16s - loss: 0.6971 - accuracy: 0.4973
 3800/25000 [===>..........................] - ETA: 16s - loss: 0.6971 - accuracy: 0.4966
 3900/25000 [===>..........................] - ETA: 15s - loss: 0.6971 - accuracy: 0.4954
 4000/25000 [===>..........................] - ETA: 15s - loss: 0.6970 - accuracy: 0.4965
 4100/25000 [===>..........................] - ETA: 15s - loss: 0.6969 - accuracy: 0.4959
 4200/25000 [====>.........................] - ETA: 15s - loss: 0.6967 - accuracy: 0.4969
 4300/25000 [====>.........................] - ETA: 15s - loss: 0.6966 - accuracy: 0.4979
 4400/25000 [====>.........................] - ETA: 15s - loss: 0.6965 - accuracy: 0.4968
 4500/25000 [====>.........................] - ETA: 15s - loss: 0.6964 - accuracy: 0.4978
 4600/25000 [====>.........................] - ETA: 15s - loss: 0.6963 - accuracy: 0.4980
 4700/25000 [====>.........................] - ETA: 14s - loss: 0.6961 - accuracy: 0.4983
 4800/25000 [====>.........................] - ETA: 14s - loss: 0.6960 - accuracy: 0.4981
 4900/25000 [====>.........................] - ETA: 14s - loss: 0.6959 - accuracy: 0.4973
 5000/25000 [=====>........................] - ETA: 14s - loss: 0.6958 - accuracy: 0.4960
 5100/25000 [=====>........................] - ETA: 14s - loss: 0.6958 - accuracy: 0.4939
 5200/25000 [=====>........................] - ETA: 14s - loss: 0.6955 - accuracy: 0.4950
 5300/25000 [=====>........................] - ETA: 14s - loss: 0.6953 - accuracy: 0.4972
 5400/25000 [=====>........................] - ETA: 14s - loss: 0.6950 - accuracy: 0.5002
 5500/25000 [=====>........................] - ETA: 14s - loss: 0.6948 - accuracy: 0.5024
 5600/25000 [=====>........................] - ETA: 13s - loss: 0.6945 - accuracy: 0.5057
 5700/25000 [=====>........................] - ETA: 13s - loss: 0.6943 - accuracy: 0.5074
 5800/25000 [=====>........................] - ETA: 13s - loss: 0.6940 - accuracy: 0.5095
 5900/25000 [======>.......................] - ETA: 13s - loss: 0.6937 - accuracy: 0.5120
 6000/25000 [======>.......................] - ETA: 13s - loss: 0.6934 - accuracy: 0.5150
 6100/25000 [======>.......................] - ETA: 13s - loss: 0.6931 - accuracy: 0.5175
 6200/25000 [======>.......................] - ETA: 13s - loss: 0.6926 - accuracy: 0.5219
 6300/25000 [======>.......................] - ETA: 13s - loss: 0.6923 - accuracy: 0.5244
 6400/25000 [======>.......................] - ETA: 13s - loss: 0.6921 - accuracy: 0.5264
 6500/25000 [======>.......................] - ETA: 13s - loss: 0.6915 - accuracy: 0.5309
 6600/25000 [======>.......................] - ETA: 13s - loss: 0.6911 - accuracy: 0.5355
 6700/25000 [=======>......................] - ETA: 13s - loss: 0.6907 - accuracy: 0.5391
 6800/25000 [=======>......................] - ETA: 12s - loss: 0.6902 - accuracy: 0.5424
 6900/25000 [=======>......................] - ETA: 12s - loss: 0.6897 - accuracy: 0.5443
 7000/25000 [=======>......................] - ETA: 12s - loss: 0.6893 - accuracy: 0.5474
 7100/25000 [=======>......................] - ETA: 12s - loss: 0.6890 - accuracy: 0.5497
 7200/25000 [=======>......................] - ETA: 12s - loss: 0.6886 - accuracy: 0.5531
 7300/25000 [=======>......................] - ETA: 12s - loss: 0.6883 - accuracy: 0.5553
 7400/25000 [=======>......................] - ETA: 12s - loss: 0.6878 - accuracy: 0.5578
 7500/25000 [========>.....................] - ETA: 12s - loss: 0.6874 - accuracy: 0.5609
 7600/25000 [========>.....................] - ETA: 12s - loss: 0.6870 - accuracy: 0.5637
 7700/25000 [========>.....................] - ETA: 12s - loss: 0.6866 - accuracy: 0.5675
 7800/25000 [========>.....................] - ETA: 12s - loss: 0.6863 - accuracy: 0.5699
 7900/25000 [========>.....................] - ETA: 11s - loss: 0.6859 - accuracy: 0.5734
 8000/25000 [========>.....................] - ETA: 11s - loss: 0.6854 - accuracy: 0.5756
 8100/25000 [========>.....................] - ETA: 11s - loss: 0.6849 - accuracy: 0.5784
 8200/25000 [========>.....................] - ETA: 11s - loss: 0.6846 - accuracy: 0.5806
 8300/25000 [========>.....................] - ETA: 11s - loss: 0.6840 - accuracy: 0.5834
 8400/25000 [=========>....................] - ETA: 11s - loss: 0.6835 - accuracy: 0.5864
 8500/25000 [=========>....................] - ETA: 11s - loss: 0.6831 - accuracy: 0.5885
 8600/25000 [=========>....................] - ETA: 11s - loss: 0.6828 - accuracy: 0.5910
 8700/25000 [=========>....................] - ETA: 11s - loss: 0.6824 - accuracy: 0.5933
 8800/25000 [=========>....................] - ETA: 11s - loss: 0.6821 - accuracy: 0.5943
 8900/25000 [=========>....................] - ETA: 11s - loss: 0.6817 - accuracy: 0.5961
 9000/25000 [=========>....................] - ETA: 11s - loss: 0.6812 - accuracy: 0.5984
 9100/25000 [=========>....................] - ETA: 11s - loss: 0.6808 - accuracy: 0.6001
 9200/25000 [==========>...................] - ETA: 10s - loss: 0.6804 - accuracy: 0.6020
 9300/25000 [==========>...................] - ETA: 10s - loss: 0.6800 - accuracy: 0.6034
 9400/25000 [==========>...................] - ETA: 10s - loss: 0.6796 - accuracy: 0.6053
 9500/25000 [==========>...................] - ETA: 10s - loss: 0.6795 - accuracy: 0.6063
 9600/25000 [==========>...................] - ETA: 10s - loss: 0.6790 - accuracy: 0.6084
 9700/25000 [==========>...................] - ETA: 10s - loss: 0.6788 - accuracy: 0.6089
 9800/25000 [==========>...................] - ETA: 10s - loss: 0.6783 - accuracy: 0.6109
 9900/25000 [==========>...................] - ETA: 10s - loss: 0.6780 - accuracy: 0.6121
10000/25000 [===========>..................] - ETA: 10s - loss: 0.6776 - accuracy: 0.6138
10100/25000 [===========>..................] - ETA: 10s - loss: 0.6774 - accuracy: 0.6147
10200/25000 [===========>..................] - ETA: 10s - loss: 0.6772 - accuracy: 0.6156
10300/25000 [===========>..................] - ETA: 10s - loss: 0.6769 - accuracy: 0.6167
10400/25000 [===========>..................] - ETA: 10s - loss: 0.6766 - accuracy: 0.6175
10500/25000 [===========>..................] - ETA: 9s - loss: 0.6764 - accuracy: 0.6180 
10600/25000 [===========>..................] - ETA: 9s - loss: 0.6761 - accuracy: 0.6186
10700/25000 [===========>..................] - ETA: 9s - loss: 0.6759 - accuracy: 0.6198
10800/25000 [===========>..................] - ETA: 9s - loss: 0.6757 - accuracy: 0.6204
10900/25000 [============>.................] - ETA: 9s - loss: 0.6756 - accuracy: 0.6204
11000/25000 [============>.................] - ETA: 9s - loss: 0.6754 - accuracy: 0.6205
11100/25000 [============>.................] - ETA: 9s - loss: 0.6752 - accuracy: 0.6208
11200/25000 [============>.................] - ETA: 9s - loss: 0.6751 - accuracy: 0.6219
11300/25000 [============>.................] - ETA: 9s - loss: 0.6751 - accuracy: 0.6219
11400/25000 [============>.................] - ETA: 9s - loss: 0.6749 - accuracy: 0.6220
11500/25000 [============>.................] - ETA: 9s - loss: 0.6747 - accuracy: 0.6223
11600/25000 [============>.................] - ETA: 9s - loss: 0.6744 - accuracy: 0.6229
11700/25000 [=============>................] - ETA: 9s - loss: 0.6743 - accuracy: 0.6230
11800/25000 [=============>................] - ETA: 9s - loss: 0.6739 - accuracy: 0.6242
11900/25000 [=============>................] - ETA: 8s - loss: 0.6737 - accuracy: 0.6246
12000/25000 [=============>................] - ETA: 8s - loss: 0.6734 - accuracy: 0.6252
12100/25000 [=============>................] - ETA: 8s - loss: 0.6732 - accuracy: 0.6256
12200/25000 [=============>................] - ETA: 8s - loss: 0.6729 - accuracy: 0.6264
12300/25000 [=============>................] - ETA: 8s - loss: 0.6726 - accuracy: 0.6275
12400/25000 [=============>................] - ETA: 8s - loss: 0.6723 - accuracy: 0.6286
12500/25000 [==============>...............] - ETA: 8s - loss: 0.6722 - accuracy: 0.6294
12600/25000 [==============>...............] - ETA: 8s - loss: 0.6719 - accuracy: 0.6302
12700/25000 [==============>...............] - ETA: 8s - loss: 0.6717 - accuracy: 0.6310
12800/25000 [==============>...............] - ETA: 8s - loss: 0.6713 - accuracy: 0.6323
12900/25000 [==============>...............] - ETA: 8s - loss: 0.6710 - accuracy: 0.6335
13000/25000 [==============>...............] - ETA: 8s - loss: 0.6708 - accuracy: 0.6341
13100/25000 [==============>...............] - ETA: 8s - loss: 0.6705 - accuracy: 0.6348
13200/25000 [==============>...............] - ETA: 8s - loss: 0.6703 - accuracy: 0.6355
13300/25000 [==============>...............] - ETA: 7s - loss: 0.6699 - accuracy: 0.6370
13400/25000 [===============>..............] - ETA: 7s - loss: 0.6697 - accuracy: 0.6372
13500/25000 [===============>..............] - ETA: 7s - loss: 0.6695 - accuracy: 0.6379
13600/25000 [===============>..............] - ETA: 7s - loss: 0.6690 - accuracy: 0.6388
13700/25000 [===============>..............] - ETA: 7s - loss: 0.6690 - accuracy: 0.6386
13800/25000 [===============>..............] - ETA: 7s - loss: 0.6688 - accuracy: 0.6389
13900/25000 [===============>..............] - ETA: 7s - loss: 0.6684 - accuracy: 0.6405
14000/25000 [===============>..............] - ETA: 7s - loss: 0.6681 - accuracy: 0.6414
14100/25000 [===============>..............] - ETA: 7s - loss: 0.6676 - accuracy: 0.6425
14200/25000 [================>.............] - ETA: 7s - loss: 0.6673 - accuracy: 0.6433
14300/25000 [================>.............] - ETA: 7s - loss: 0.6670 - accuracy: 0.6444
14400/25000 [================>.............] - ETA: 7s - loss: 0.6668 - accuracy: 0.6451
14500/25000 [================>.............] - ETA: 7s - loss: 0.6666 - accuracy: 0.6457
14600/25000 [================>.............] - ETA: 7s - loss: 0.6663 - accuracy: 0.6465
14700/25000 [================>.............] - ETA: 6s - loss: 0.6660 - accuracy: 0.6481
14800/25000 [================>.............] - ETA: 6s - loss: 0.6659 - accuracy: 0.6485
14900/25000 [================>.............] - ETA: 6s - loss: 0.6657 - accuracy: 0.6494
15000/25000 [=================>............] - ETA: 6s - loss: 0.6654 - accuracy: 0.6503
15100/25000 [=================>............] - ETA: 6s - loss: 0.6651 - accuracy: 0.6515
15200/25000 [=================>............] - ETA: 6s - loss: 0.6648 - accuracy: 0.6526
15300/25000 [=================>............] - ETA: 6s - loss: 0.6646 - accuracy: 0.6538
15400/25000 [=================>............] - ETA: 6s - loss: 0.6646 - accuracy: 0.6542
15500/25000 [=================>............] - ETA: 6s - loss: 0.6642 - accuracy: 0.6549
15600/25000 [=================>............] - ETA: 6s - loss: 0.6640 - accuracy: 0.6554
15700/25000 [=================>............] - ETA: 6s - loss: 0.6637 - accuracy: 0.6564
15800/25000 [=================>............] - ETA: 6s - loss: 0.6635 - accuracy: 0.6570
15900/25000 [==================>...........] - ETA: 6s - loss: 0.6632 - accuracy: 0.6577
16000/25000 [==================>...........] - ETA: 6s - loss: 0.6630 - accuracy: 0.6587
16100/25000 [==================>...........] - ETA: 5s - loss: 0.6628 - accuracy: 0.6596
16200/25000 [==================>...........] - ETA: 5s - loss: 0.6625 - accuracy: 0.6604
16300/25000 [==================>...........] - ETA: 5s - loss: 0.6622 - accuracy: 0.6612
16400/25000 [==================>...........] - ETA: 5s - loss: 0.6618 - accuracy: 0.6624
16500/25000 [==================>...........] - ETA: 5s - loss: 0.6615 - accuracy: 0.6632
16600/25000 [==================>...........] - ETA: 5s - loss: 0.6611 - accuracy: 0.6643
16700/25000 [===================>..........] - ETA: 5s - loss: 0.6610 - accuracy: 0.6650
16800/25000 [===================>..........] - ETA: 5s - loss: 0.6608 - accuracy: 0.6657
16900/25000 [===================>..........] - ETA: 5s - loss: 0.6605 - accuracy: 0.6663
17000/25000 [===================>..........] - ETA: 5s - loss: 0.6604 - accuracy: 0.6668
17100/25000 [===================>..........] - ETA: 5s - loss: 0.6601 - accuracy: 0.6678
17200/25000 [===================>..........] - ETA: 5s - loss: 0.6599 - accuracy: 0.6683
17300/25000 [===================>..........] - ETA: 5s - loss: 0.6595 - accuracy: 0.6692
17400/25000 [===================>..........] - ETA: 5s - loss: 0.6594 - accuracy: 0.6698
17500/25000 [====================>.........] - ETA: 4s - loss: 0.6593 - accuracy: 0.6703
17600/25000 [====================>.........] - ETA: 4s - loss: 0.6593 - accuracy: 0.6704
17700/25000 [====================>.........] - ETA: 4s - loss: 0.6595 - accuracy: 0.6701
17800/25000 [====================>.........] - ETA: 4s - loss: 0.6594 - accuracy: 0.6707
17900/25000 [====================>.........] - ETA: 4s - loss: 0.6593 - accuracy: 0.6711
18000/25000 [====================>.........] - ETA: 4s - loss: 0.6592 - accuracy: 0.6716
18100/25000 [====================>.........] - ETA: 4s - loss: 0.6593 - accuracy: 0.6718
18200/25000 [====================>.........] - ETA: 4s - loss: 0.6593 - accuracy: 0.6719
18300/25000 [====================>.........] - ETA: 4s - loss: 0.6592 - accuracy: 0.6723
18400/25000 [=====================>........] - ETA: 4s - loss: 0.6591 - accuracy: 0.6729
18500/25000 [=====================>........] - ETA: 4s - loss: 0.6588 - accuracy: 0.6735
18600/25000 [=====================>........] - ETA: 4s - loss: 0.6585 - accuracy: 0.6745
18700/25000 [=====================>........] - ETA: 4s - loss: 0.6582 - accuracy: 0.6753
18800/25000 [=====================>........] - ETA: 4s - loss: 0.6580 - accuracy: 0.6760
18900/25000 [=====================>........] - ETA: 4s - loss: 0.6579 - accuracy: 0.6765
19000/25000 [=====================>........] - ETA: 3s - loss: 0.6575 - accuracy: 0.6775
19100/25000 [=====================>........] - ETA: 3s - loss: 0.6572 - accuracy: 0.6781
19200/25000 [======================>.......] - ETA: 3s - loss: 0.6568 - accuracy: 0.6790
19300/25000 [======================>.......] - ETA: 3s - loss: 0.6566 - accuracy: 0.6795
19400/25000 [======================>.......] - ETA: 3s - loss: 0.6564 - accuracy: 0.6796
19500/25000 [======================>.......] - ETA: 3s - loss: 0.6561 - accuracy: 0.6802
19600/25000 [======================>.......] - ETA: 3s - loss: 0.6559 - accuracy: 0.6809
19700/25000 [======================>.......] - ETA: 3s - loss: 0.6557 - accuracy: 0.6816
19800/25000 [======================>.......] - ETA: 3s - loss: 0.6555 - accuracy: 0.6823
19900/25000 [======================>.......] - ETA: 3s - loss: 0.6555 - accuracy: 0.6823
20000/25000 [=======================>......] - ETA: 3s - loss: 0.6555 - accuracy: 0.6822
20100/25000 [=======================>......] - ETA: 3s - loss: 0.6556 - accuracy: 0.6817
20200/25000 [=======================>......] - ETA: 3s - loss: 0.6556 - accuracy: 0.6814
20300/25000 [=======================>......] - ETA: 3s - loss: 0.6556 - accuracy: 0.6812
20400/25000 [=======================>......] - ETA: 3s - loss: 0.6556 - accuracy: 0.6811
20500/25000 [=======================>......] - ETA: 2s - loss: 0.6556 - accuracy: 0.6810
20600/25000 [=======================>......] - ETA: 2s - loss: 0.6557 - accuracy: 0.6803
20700/25000 [=======================>......] - ETA: 2s - loss: 0.6558 - accuracy: 0.6801
20800/25000 [=======================>......] - ETA: 2s - loss: 0.6556 - accuracy: 0.6803
20900/25000 [========================>.....] - ETA: 2s - loss: 0.6557 - accuracy: 0.6803
21000/25000 [========================>.....] - ETA: 2s - loss: 0.6556 - accuracy: 0.6807
21100/25000 [========================>.....] - ETA: 2s - loss: 0.6556 - accuracy: 0.6810
21200/25000 [========================>.....] - ETA: 2s - loss: 0.6554 - accuracy: 0.6814
21300/25000 [========================>.....] - ETA: 2s - loss: 0.6552 - accuracy: 0.6820
21400/25000 [========================>.....] - ETA: 2s - loss: 0.6550 - accuracy: 0.6824
21500/25000 [========================>.....] - ETA: 2s - loss: 0.6549 - accuracy: 0.6830
21600/25000 [========================>.....] - ETA: 2s - loss: 0.6547 - accuracy: 0.6835
21700/25000 [=========================>....] - ETA: 2s - loss: 0.6546 - accuracy: 0.6841
21800/25000 [=========================>....] - ETA: 2s - loss: 0.6543 - accuracy: 0.6844
21900/25000 [=========================>....] - ETA: 2s - loss: 0.6540 - accuracy: 0.6852
22000/25000 [=========================>....] - ETA: 1s - loss: 0.6539 - accuracy: 0.6856
22100/25000 [=========================>....] - ETA: 1s - loss: 0.6538 - accuracy: 0.6858
22200/25000 [=========================>....] - ETA: 1s - loss: 0.6537 - accuracy: 0.6861
22300/25000 [=========================>....] - ETA: 1s - loss: 0.6536 - accuracy: 0.6864
22400/25000 [=========================>....] - ETA: 1s - loss: 0.6534 - accuracy: 0.6869
22500/25000 [==========================>...] - ETA: 1s - loss: 0.6532 - accuracy: 0.6876
22600/25000 [==========================>...] - ETA: 1s - loss: 0.6530 - accuracy: 0.6881
22700/25000 [==========================>...] - ETA: 1s - loss: 0.6527 - accuracy: 0.6887
22800/25000 [==========================>...] - ETA: 1s - loss: 0.6526 - accuracy: 0.6890
22900/25000 [==========================>...] - ETA: 1s - loss: 0.6524 - accuracy: 0.6897
23000/25000 [==========================>...] - ETA: 1s - loss: 0.6522 - accuracy: 0.6903
23100/25000 [==========================>...] - ETA: 1s - loss: 0.6520 - accuracy: 0.6909
23200/25000 [==========================>...] - ETA: 1s - loss: 0.6518 - accuracy: 0.6913
23300/25000 [==========================>...] - ETA: 1s - loss: 0.6517 - accuracy: 0.6913
23400/25000 [===========================>..] - ETA: 1s - loss: 0.6516 - accuracy: 0.6917
23500/25000 [===========================>..] - ETA: 0s - loss: 0.6514 - accuracy: 0.6924
23600/25000 [===========================>..] - ETA: 0s - loss: 0.6511 - accuracy: 0.6930
23700/25000 [===========================>..] - ETA: 0s - loss: 0.6509 - accuracy: 0.6934
23800/25000 [===========================>..] - ETA: 0s - loss: 0.6507 - accuracy: 0.6939
23900/25000 [===========================>..] - ETA: 0s - loss: 0.6505 - accuracy: 0.6946
24000/25000 [===========================>..] - ETA: 0s - loss: 0.6502 - accuracy: 0.6949
24100/25000 [===========================>..] - ETA: 0s - loss: 0.6500 - accuracy: 0.6953
24200/25000 [============================>.] - ETA: 0s - loss: 0.6498 - accuracy: 0.6959
24300/25000 [============================>.] - ETA: 0s - loss: 0.6496 - accuracy: 0.6965
24400/25000 [============================>.] - ETA: 0s - loss: 0.6494 - accuracy: 0.6970
24500/25000 [============================>.] - ETA: 0s - loss: 0.6492 - accuracy: 0.6975
24600/25000 [============================>.] - ETA: 0s - loss: 0.6491 - accuracy: 0.6977
24700/25000 [============================>.] - ETA: 0s - loss: 0.6489 - accuracy: 0.6983
24800/25000 [============================>.] - ETA: 0s - loss: 0.6488 - accuracy: 0.6983
24900/25000 [============================>.] - ETA: 0s - loss: 0.6486 - accuracy: 0.6988
25000/25000 [==============================] - 21s 821us/step - loss: 0.6484 - accuracy: 0.6994 - val_loss: 0.5997 - val_accuracy: 0.8251
Epoch 2/10

  100/25000 [..............................] - ETA: 15s - loss: 0.5969 - accuracy: 0.8100
  200/25000 [..............................] - ETA: 15s - loss: 0.6079 - accuracy: 0.7800
  300/25000 [..............................] - ETA: 15s - loss: 0.6098 - accuracy: 0.7933
  400/25000 [..............................] - ETA: 15s - loss: 0.6013 - accuracy: 0.8175
  500/25000 [..............................] - ETA: 15s - loss: 0.5985 - accuracy: 0.8320
  600/25000 [..............................] - ETA: 15s - loss: 0.5970 - accuracy: 0.8383
  700/25000 [..............................] - ETA: 14s - loss: 0.5932 - accuracy: 0.8471
  800/25000 [..............................] - ETA: 14s - loss: 0.5972 - accuracy: 0.8350
  900/25000 [>.............................] - ETA: 14s - loss: 0.5966 - accuracy: 0.8333
 1000/25000 [>.............................] - ETA: 14s - loss: 0.5954 - accuracy: 0.8370
 1100/25000 [>.............................] - ETA: 14s - loss: 0.5969 - accuracy: 0.8355
 1200/25000 [>.............................] - ETA: 14s - loss: 0.5962 - accuracy: 0.8367
 1300/25000 [>.............................] - ETA: 14s - loss: 0.5950 - accuracy: 0.8415
 1400/25000 [>.............................] - ETA: 14s - loss: 0.5962 - accuracy: 0.8400
 1500/25000 [>.............................] - ETA: 14s - loss: 0.5950 - accuracy: 0.8413
 1600/25000 [>.............................] - ETA: 14s - loss: 0.5954 - accuracy: 0.8381
 1700/25000 [=>............................] - ETA: 14s - loss: 0.5949 - accuracy: 0.8394
 1800/25000 [=>............................] - ETA: 14s - loss: 0.5939 - accuracy: 0.8411
 1900/25000 [=>............................] - ETA: 14s - loss: 0.5938 - accuracy: 0.8405
 2000/25000 [=>............................] - ETA: 14s - loss: 0.5916 - accuracy: 0.8440
 2100/25000 [=>............................] - ETA: 14s - loss: 0.5918 - accuracy: 0.8433
 2200/25000 [=>............................] - ETA: 14s - loss: 0.5909 - accuracy: 0.8455
 2300/25000 [=>............................] - ETA: 14s - loss: 0.5907 - accuracy: 0.8457
 2400/25000 [=>............................] - ETA: 13s - loss: 0.5902 - accuracy: 0.8471
 2500/25000 [==>...........................] - ETA: 13s - loss: 0.5899 - accuracy: 0.8488
 2600/25000 [==>...........................] - ETA: 13s - loss: 0.5891 - accuracy: 0.8477
 2700/25000 [==>...........................] - ETA: 13s - loss: 0.5886 - accuracy: 0.8493
 2800/25000 [==>...........................] - ETA: 13s - loss: 0.5885 - accuracy: 0.8493
 2900/25000 [==>...........................] - ETA: 13s - loss: 0.5881 - accuracy: 0.8497
 3000/25000 [==>...........................] - ETA: 13s - loss: 0.5888 - accuracy: 0.8493
 3100/25000 [==>...........................] - ETA: 13s - loss: 0.5884 - accuracy: 0.8494
 3200/25000 [==>...........................] - ETA: 13s - loss: 0.5886 - accuracy: 0.8500
 3300/25000 [==>...........................] - ETA: 13s - loss: 0.5887 - accuracy: 0.8491
 3400/25000 [===>..........................] - ETA: 13s - loss: 0.5882 - accuracy: 0.8503
 3500/25000 [===>..........................] - ETA: 13s - loss: 0.5878 - accuracy: 0.8506
 3600/25000 [===>..........................] - ETA: 13s - loss: 0.5876 - accuracy: 0.8514
 3700/25000 [===>..........................] - ETA: 13s - loss: 0.5879 - accuracy: 0.8497
 3800/25000 [===>..........................] - ETA: 13s - loss: 0.5880 - accuracy: 0.8492
 3900/25000 [===>..........................] - ETA: 13s - loss: 0.5875 - accuracy: 0.8500
 4000/25000 [===>..........................] - ETA: 13s - loss: 0.5870 - accuracy: 0.8508
 4100/25000 [===>..........................] - ETA: 12s - loss: 0.5871 - accuracy: 0.8510
 4200/25000 [====>.........................] - ETA: 12s - loss: 0.5870 - accuracy: 0.8514
 4300/25000 [====>.........................] - ETA: 12s - loss: 0.5876 - accuracy: 0.8495
 4400/25000 [====>.........................] - ETA: 12s - loss: 0.5874 - accuracy: 0.8491
 4500/25000 [====>.........................] - ETA: 12s - loss: 0.5876 - accuracy: 0.8482
 4600/25000 [====>.........................] - ETA: 12s - loss: 0.5874 - accuracy: 0.8485
 4700/25000 [====>.........................] - ETA: 12s - loss: 0.5871 - accuracy: 0.8487
 4800/25000 [====>.........................] - ETA: 12s - loss: 0.5868 - accuracy: 0.8481
 4900/25000 [====>.........................] - ETA: 12s - loss: 0.5864 - accuracy: 0.8494
 5000/25000 [=====>........................] - ETA: 12s - loss: 0.5867 - accuracy: 0.8494
 5100/25000 [=====>........................] - ETA: 12s - loss: 0.5867 - accuracy: 0.8494
 5200/25000 [=====>........................] - ETA: 12s - loss: 0.5866 - accuracy: 0.8492
 5300/25000 [=====>........................] - ETA: 12s - loss: 0.5862 - accuracy: 0.8502
 5400/25000 [=====>........................] - ETA: 12s - loss: 0.5861 - accuracy: 0.8507
 5500/25000 [=====>........................] - ETA: 12s - loss: 0.5861 - accuracy: 0.8513
 5600/25000 [=====>........................] - ETA: 12s - loss: 0.5858 - accuracy: 0.8518
 5700/25000 [=====>........................] - ETA: 11s - loss: 0.5851 - accuracy: 0.8532
 5800/25000 [=====>........................] - ETA: 11s - loss: 0.5848 - accuracy: 0.8534
 5900/25000 [======>.......................] - ETA: 11s - loss: 0.5846 - accuracy: 0.8539
 6000/25000 [======>.......................] - ETA: 11s - loss: 0.5844 - accuracy: 0.8537
 6100/25000 [======>.......................] - ETA: 11s - loss: 0.5842 - accuracy: 0.8543
 6200/25000 [======>.......................] - ETA: 11s - loss: 0.5839 - accuracy: 0.8545
 6300/25000 [======>.......................] - ETA: 11s - loss: 0.5838 - accuracy: 0.8543
 6400/25000 [======>.......................] - ETA: 11s - loss: 0.5838 - accuracy: 0.8541
 6500/25000 [======>.......................] - ETA: 11s - loss: 0.5839 - accuracy: 0.8537
 6600/25000 [======>.......................] - ETA: 11s - loss: 0.5841 - accuracy: 0.8533
 6700/25000 [=======>......................] - ETA: 11s - loss: 0.5841 - accuracy: 0.8527
 6800/25000 [=======>......................] - ETA: 11s - loss: 0.5838 - accuracy: 0.8525
 6900/25000 [=======>......................] - ETA: 11s - loss: 0.5835 - accuracy: 0.8532
 7000/25000 [=======>......................] - ETA: 11s - loss: 0.5834 - accuracy: 0.8534
 7100/25000 [=======>......................] - ETA: 11s - loss: 0.5834 - accuracy: 0.8534
 7200/25000 [=======>......................] - ETA: 11s - loss: 0.5833 - accuracy: 0.8531
 7300/25000 [=======>......................] - ETA: 10s - loss: 0.5829 - accuracy: 0.8533
 7400/25000 [=======>......................] - ETA: 10s - loss: 0.5829 - accuracy: 0.8530
 7500/25000 [========>.....................] - ETA: 10s - loss: 0.5827 - accuracy: 0.8536
 7600/25000 [========>.....................] - ETA: 10s - loss: 0.5826 - accuracy: 0.8536
 7700/25000 [========>.....................] - ETA: 10s - loss: 0.5823 - accuracy: 0.8531
 7800/25000 [========>.....................] - ETA: 10s - loss: 0.5821 - accuracy: 0.8537
 7900/25000 [========>.....................] - ETA: 10s - loss: 0.5818 - accuracy: 0.8538
 8000/25000 [========>.....................] - ETA: 10s - loss: 0.5817 - accuracy: 0.8535
 8100/25000 [========>.....................] - ETA: 10s - loss: 0.5819 - accuracy: 0.8530
 8200/25000 [========>.....................] - ETA: 10s - loss: 0.5817 - accuracy: 0.8533
 8300/25000 [========>.....................] - ETA: 10s - loss: 0.5816 - accuracy: 0.8530
 8400/25000 [=========>....................] - ETA: 10s - loss: 0.5817 - accuracy: 0.8530
 8500/25000 [=========>....................] - ETA: 10s - loss: 0.5816 - accuracy: 0.8528
 8600/25000 [=========>....................] - ETA: 10s - loss: 0.5816 - accuracy: 0.8528
 8700/25000 [=========>....................] - ETA: 10s - loss: 0.5815 - accuracy: 0.8526
 8800/25000 [=========>....................] - ETA: 10s - loss: 0.5811 - accuracy: 0.8535
 8900/25000 [=========>....................] - ETA: 9s - loss: 0.5810 - accuracy: 0.8542 
 9000/25000 [=========>....................] - ETA: 9s - loss: 0.5808 - accuracy: 0.8544
 9100/25000 [=========>....................] - ETA: 9s - loss: 0.5809 - accuracy: 0.8540
 9200/25000 [==========>...................] - ETA: 9s - loss: 0.5807 - accuracy: 0.8541
 9300/25000 [==========>...................] - ETA: 9s - loss: 0.5807 - accuracy: 0.8540
 9400/25000 [==========>...................] - ETA: 9s - loss: 0.5807 - accuracy: 0.8537
 9500/25000 [==========>...................] - ETA: 9s - loss: 0.5808 - accuracy: 0.8533
 9600/25000 [==========>...................] - ETA: 9s - loss: 0.5808 - accuracy: 0.8532
 9700/25000 [==========>...................] - ETA: 9s - loss: 0.5809 - accuracy: 0.8527
 9800/25000 [==========>...................] - ETA: 9s - loss: 0.5807 - accuracy: 0.8532
 9900/25000 [==========>...................] - ETA: 9s - loss: 0.5807 - accuracy: 0.8529
10000/25000 [===========>..................] - ETA: 9s - loss: 0.5807 - accuracy: 0.8530
10100/25000 [===========>..................] - ETA: 9s - loss: 0.5805 - accuracy: 0.8533
10200/25000 [===========>..................] - ETA: 9s - loss: 0.5802 - accuracy: 0.8538
10300/25000 [===========>..................] - ETA: 9s - loss: 0.5800 - accuracy: 0.8542
10400/25000 [===========>..................] - ETA: 9s - loss: 0.5800 - accuracy: 0.8540
10500/25000 [===========>..................] - ETA: 8s - loss: 0.5800 - accuracy: 0.8535
10600/25000 [===========>..................] - ETA: 8s - loss: 0.5799 - accuracy: 0.8536
10700/25000 [===========>..................] - ETA: 8s - loss: 0.5797 - accuracy: 0.8538
10800/25000 [===========>..................] - ETA: 8s - loss: 0.5797 - accuracy: 0.8537
10900/25000 [============>.................] - ETA: 8s - loss: 0.5795 - accuracy: 0.8539
11000/25000 [============>.................] - ETA: 8s - loss: 0.5795 - accuracy: 0.8535
11100/25000 [============>.................] - ETA: 8s - loss: 0.5795 - accuracy: 0.8534
11200/25000 [============>.................] - ETA: 8s - loss: 0.5796 - accuracy: 0.8529
11300/25000 [============>.................] - ETA: 8s - loss: 0.5795 - accuracy: 0.8531
11400/25000 [============>.................] - ETA: 8s - loss: 0.5797 - accuracy: 0.8523
11500/25000 [============>.................] - ETA: 8s - loss: 0.5799 - accuracy: 0.8519
11600/25000 [============>.................] - ETA: 8s - loss: 0.5798 - accuracy: 0.8519
11700/25000 [=============>................] - ETA: 8s - loss: 0.5797 - accuracy: 0.8521
11800/25000 [=============>................] - ETA: 8s - loss: 0.5795 - accuracy: 0.8520
11900/25000 [=============>................] - ETA: 8s - loss: 0.5794 - accuracy: 0.8523
12000/25000 [=============>................] - ETA: 8s - loss: 0.5792 - accuracy: 0.8523
12100/25000 [=============>................] - ETA: 7s - loss: 0.5792 - accuracy: 0.8520
12200/25000 [=============>................] - ETA: 7s - loss: 0.5791 - accuracy: 0.8516
12300/25000 [=============>................] - ETA: 7s - loss: 0.5789 - accuracy: 0.8521
12400/25000 [=============>................] - ETA: 7s - loss: 0.5789 - accuracy: 0.8522
12500/25000 [==============>...............] - ETA: 7s - loss: 0.5789 - accuracy: 0.8519
12600/25000 [==============>...............] - ETA: 7s - loss: 0.5788 - accuracy: 0.8519
12700/25000 [==============>...............] - ETA: 7s - loss: 0.5787 - accuracy: 0.8519
12800/25000 [==============>...............] - ETA: 7s - loss: 0.5786 - accuracy: 0.8521
12900/25000 [==============>...............] - ETA: 7s - loss: 0.5787 - accuracy: 0.8519
13000/25000 [==============>...............] - ETA: 7s - loss: 0.5787 - accuracy: 0.8517
13100/25000 [==============>...............] - ETA: 7s - loss: 0.5786 - accuracy: 0.8515
13200/25000 [==============>...............] - ETA: 7s - loss: 0.5784 - accuracy: 0.8516
13300/25000 [==============>...............] - ETA: 7s - loss: 0.5784 - accuracy: 0.8514
13400/25000 [===============>..............] - ETA: 7s - loss: 0.5782 - accuracy: 0.8515
13500/25000 [===============>..............] - ETA: 7s - loss: 0.5780 - accuracy: 0.8518
13600/25000 [===============>..............] - ETA: 7s - loss: 0.5778 - accuracy: 0.8518
13700/25000 [===============>..............] - ETA: 7s - loss: 0.5777 - accuracy: 0.8521
13800/25000 [===============>..............] - ETA: 6s - loss: 0.5776 - accuracy: 0.8520
13900/25000 [===============>..............] - ETA: 6s - loss: 0.5776 - accuracy: 0.8522
14000/25000 [===============>..............] - ETA: 6s - loss: 0.5775 - accuracy: 0.8522
14100/25000 [===============>..............] - ETA: 6s - loss: 0.5773 - accuracy: 0.8523
14200/25000 [================>.............] - ETA: 6s - loss: 0.5773 - accuracy: 0.8520
14300/25000 [================>.............] - ETA: 6s - loss: 0.5772 - accuracy: 0.8521
14400/25000 [================>.............] - ETA: 6s - loss: 0.5772 - accuracy: 0.8519
14500/25000 [================>.............] - ETA: 6s - loss: 0.5770 - accuracy: 0.8521
14600/25000 [================>.............] - ETA: 6s - loss: 0.5768 - accuracy: 0.8524
14700/25000 [================>.............] - ETA: 6s - loss: 0.5767 - accuracy: 0.8524
14800/25000 [================>.............] - ETA: 6s - loss: 0.5766 - accuracy: 0.8525
14900/25000 [================>.............] - ETA: 6s - loss: 0.5765 - accuracy: 0.8528
15000/25000 [=================>............] - ETA: 6s - loss: 0.5764 - accuracy: 0.8529
15100/25000 [=================>............] - ETA: 6s - loss: 0.5761 - accuracy: 0.8530
15200/25000 [=================>............] - ETA: 6s - loss: 0.5760 - accuracy: 0.8532
15300/25000 [=================>............] - ETA: 6s - loss: 0.5756 - accuracy: 0.8534
15400/25000 [=================>............] - ETA: 5s - loss: 0.5757 - accuracy: 0.8529
15500/25000 [=================>............] - ETA: 5s - loss: 0.5756 - accuracy: 0.8529
15600/25000 [=================>............] - ETA: 5s - loss: 0.5756 - accuracy: 0.8529
15700/25000 [=================>............] - ETA: 5s - loss: 0.5754 - accuracy: 0.8530
15800/25000 [=================>............] - ETA: 5s - loss: 0.5754 - accuracy: 0.8529
15900/25000 [==================>...........] - ETA: 5s - loss: 0.5754 - accuracy: 0.8527
16000/25000 [==================>...........] - ETA: 5s - loss: 0.5752 - accuracy: 0.8531
16100/25000 [==================>...........] - ETA: 5s - loss: 0.5751 - accuracy: 0.8531
16200/25000 [==================>...........] - ETA: 5s - loss: 0.5752 - accuracy: 0.8527
16300/25000 [==================>...........] - ETA: 5s - loss: 0.5751 - accuracy: 0.8528
16400/25000 [==================>...........] - ETA: 5s - loss: 0.5752 - accuracy: 0.8524
16500/25000 [==================>...........] - ETA: 5s - loss: 0.5752 - accuracy: 0.8523
16600/25000 [==================>...........] - ETA: 5s - loss: 0.5752 - accuracy: 0.8522
16700/25000 [===================>..........] - ETA: 5s - loss: 0.5750 - accuracy: 0.8525
16800/25000 [===================>..........] - ETA: 5s - loss: 0.5750 - accuracy: 0.8524
16900/25000 [===================>..........] - ETA: 5s - loss: 0.5748 - accuracy: 0.8525
17000/25000 [===================>..........] - ETA: 4s - loss: 0.5747 - accuracy: 0.8524
17100/25000 [===================>..........] - ETA: 4s - loss: 0.5744 - accuracy: 0.8527
17200/25000 [===================>..........] - ETA: 4s - loss: 0.5744 - accuracy: 0.8524
17300/25000 [===================>..........] - ETA: 4s - loss: 0.5744 - accuracy: 0.8524
17400/25000 [===================>..........] - ETA: 4s - loss: 0.5743 - accuracy: 0.8525
17500/25000 [====================>.........] - ETA: 4s - loss: 0.5741 - accuracy: 0.8529
17600/25000 [====================>.........] - ETA: 4s - loss: 0.5739 - accuracy: 0.8531
17700/25000 [====================>.........] - ETA: 4s - loss: 0.5738 - accuracy: 0.8533
17800/25000 [====================>.........] - ETA: 4s - loss: 0.5737 - accuracy: 0.8533
17900/25000 [====================>.........] - ETA: 4s - loss: 0.5735 - accuracy: 0.8536
18000/25000 [====================>.........] - ETA: 4s - loss: 0.5734 - accuracy: 0.8535
18100/25000 [====================>.........] - ETA: 4s - loss: 0.5731 - accuracy: 0.8537
18200/25000 [====================>.........] - ETA: 4s - loss: 0.5730 - accuracy: 0.8537
18300/25000 [====================>.........] - ETA: 4s - loss: 0.5731 - accuracy: 0.8537
18400/25000 [=====================>........] - ETA: 4s - loss: 0.5732 - accuracy: 0.8534
18500/25000 [=====================>........] - ETA: 4s - loss: 0.5731 - accuracy: 0.8534
18600/25000 [=====================>........] - ETA: 3s - loss: 0.5731 - accuracy: 0.8533
18700/25000 [=====================>........] - ETA: 3s - loss: 0.5729 - accuracy: 0.8534
18800/25000 [=====================>........] - ETA: 3s - loss: 0.5729 - accuracy: 0.8531
18900/25000 [=====================>........] - ETA: 3s - loss: 0.5729 - accuracy: 0.8530
19000/25000 [=====================>........] - ETA: 3s - loss: 0.5727 - accuracy: 0.8532
19100/25000 [=====================>........] - ETA: 3s - loss: 0.5725 - accuracy: 0.8534
19200/25000 [======================>.......] - ETA: 3s - loss: 0.5724 - accuracy: 0.8532
19300/25000 [======================>.......] - ETA: 3s - loss: 0.5721 - accuracy: 0.8536
19400/25000 [======================>.......] - ETA: 3s - loss: 0.5719 - accuracy: 0.8538
19500/25000 [======================>.......] - ETA: 3s - loss: 0.5717 - accuracy: 0.8541
19600/25000 [======================>.......] - ETA: 3s - loss: 0.5716 - accuracy: 0.8541
19700/25000 [======================>.......] - ETA: 3s - loss: 0.5717 - accuracy: 0.8537
19800/25000 [======================>.......] - ETA: 3s - loss: 0.5716 - accuracy: 0.8536
19900/25000 [======================>.......] - ETA: 3s - loss: 0.5714 - accuracy: 0.8539
20000/25000 [=======================>......] - ETA: 3s - loss: 0.5714 - accuracy: 0.8539
20100/25000 [=======================>......] - ETA: 3s - loss: 0.5713 - accuracy: 0.8538
20200/25000 [=======================>......] - ETA: 2s - loss: 0.5712 - accuracy: 0.8540
20300/25000 [=======================>......] - ETA: 2s - loss: 0.5710 - accuracy: 0.8542
20400/25000 [=======================>......] - ETA: 2s - loss: 0.5709 - accuracy: 0.8544
20500/25000 [=======================>......] - ETA: 2s - loss: 0.5707 - accuracy: 0.8545
20600/25000 [=======================>......] - ETA: 2s - loss: 0.5706 - accuracy: 0.8545
20700/25000 [=======================>......] - ETA: 2s - loss: 0.5703 - accuracy: 0.8549
20800/25000 [=======================>......] - ETA: 2s - loss: 0.5702 - accuracy: 0.8551
20900/25000 [========================>.....] - ETA: 2s - loss: 0.5701 - accuracy: 0.8552
21000/25000 [========================>.....] - ETA: 2s - loss: 0.5699 - accuracy: 0.8553
21100/25000 [========================>.....] - ETA: 2s - loss: 0.5697 - accuracy: 0.8556
21200/25000 [========================>.....] - ETA: 2s - loss: 0.5696 - accuracy: 0.8556
21300/25000 [========================>.....] - ETA: 2s - loss: 0.5696 - accuracy: 0.8556
21400/25000 [========================>.....] - ETA: 2s - loss: 0.5695 - accuracy: 0.8557
21500/25000 [========================>.....] - ETA: 2s - loss: 0.5695 - accuracy: 0.8555
21600/25000 [========================>.....] - ETA: 2s - loss: 0.5694 - accuracy: 0.8555
21700/25000 [=========================>....] - ETA: 2s - loss: 0.5693 - accuracy: 0.8556
21800/25000 [=========================>....] - ETA: 1s - loss: 0.5693 - accuracy: 0.8554
21900/25000 [=========================>....] - ETA: 1s - loss: 0.5691 - accuracy: 0.8556
22000/25000 [=========================>....] - ETA: 1s - loss: 0.5691 - accuracy: 0.8555
22100/25000 [=========================>....] - ETA: 1s - loss: 0.5691 - accuracy: 0.8554
22200/25000 [=========================>....] - ETA: 1s - loss: 0.5690 - accuracy: 0.8555
22300/25000 [=========================>....] - ETA: 1s - loss: 0.5689 - accuracy: 0.8554
22400/25000 [=========================>....] - ETA: 1s - loss: 0.5688 - accuracy: 0.8553
22500/25000 [==========================>...] - ETA: 1s - loss: 0.5687 - accuracy: 0.8554
22600/25000 [==========================>...] - ETA: 1s - loss: 0.5685 - accuracy: 0.8555
22700/25000 [==========================>...] - ETA: 1s - loss: 0.5685 - accuracy: 0.8556
22800/25000 [==========================>...] - ETA: 1s - loss: 0.5685 - accuracy: 0.8555
22900/25000 [==========================>...] - ETA: 1s - loss: 0.5684 - accuracy: 0.8554
23000/25000 [==========================>...] - ETA: 1s - loss: 0.5683 - accuracy: 0.8556
23100/25000 [==========================>...] - ETA: 1s - loss: 0.5683 - accuracy: 0.8554
23200/25000 [==========================>...] - ETA: 1s - loss: 0.5682 - accuracy: 0.8556
23300/25000 [==========================>...] - ETA: 1s - loss: 0.5681 - accuracy: 0.8555
23400/25000 [===========================>..] - ETA: 0s - loss: 0.5680 - accuracy: 0.8558
23500/25000 [===========================>..] - ETA: 0s - loss: 0.5678 - accuracy: 0.8559
23600/25000 [===========================>..] - ETA: 0s - loss: 0.5676 - accuracy: 0.8561
23700/25000 [===========================>..] - ETA: 0s - loss: 0.5674 - accuracy: 0.8564
23800/25000 [===========================>..] - ETA: 0s - loss: 0.5673 - accuracy: 0.8566
23900/25000 [===========================>..] - ETA: 0s - loss: 0.5671 - accuracy: 0.8566
24000/25000 [===========================>..] - ETA: 0s - loss: 0.5671 - accuracy: 0.8565
24100/25000 [===========================>..] - ETA: 0s - loss: 0.5670 - accuracy: 0.8564
24200/25000 [============================>.] - ETA: 0s - loss: 0.5668 - accuracy: 0.8568
24300/25000 [============================>.] - ETA: 0s - loss: 0.5666 - accuracy: 0.8568
24400/25000 [============================>.] - ETA: 0s - loss: 0.5664 - accuracy: 0.8570
24500/25000 [============================>.] - ETA: 0s - loss: 0.5663 - accuracy: 0.8571
24600/25000 [============================>.] - ETA: 0s - loss: 0.5662 - accuracy: 0.8572
24700/25000 [============================>.] - ETA: 0s - loss: 0.5661 - accuracy: 0.8572
24800/25000 [============================>.] - ETA: 0s - loss: 0.5660 - accuracy: 0.8572
24900/25000 [============================>.] - ETA: 0s - loss: 0.5658 - accuracy: 0.8573
25000/25000 [==============================] - 20s 787us/step - loss: 0.5657 - accuracy: 0.8576 - val_loss: 0.5501 - val_accuracy: 0.8491
Epoch 3/10

  100/25000 [..............................] - ETA: 15s - loss: 0.5131 - accuracy: 0.9400
  200/25000 [..............................] - ETA: 15s - loss: 0.5226 - accuracy: 0.9200
  300/25000 [..............................] - ETA: 15s - loss: 0.5292 - accuracy: 0.9033
  400/25000 [..............................] - ETA: 15s - loss: 0.5320 - accuracy: 0.9000
  500/25000 [..............................] - ETA: 15s - loss: 0.5333 - accuracy: 0.8940
  600/25000 [..............................] - ETA: 15s - loss: 0.5366 - accuracy: 0.8850
  700/25000 [..............................] - ETA: 15s - loss: 0.5396 - accuracy: 0.8800
  800/25000 [..............................] - ETA: 15s - loss: 0.5381 - accuracy: 0.8838
  900/25000 [>.............................] - ETA: 15s - loss: 0.5360 - accuracy: 0.8867
 1000/25000 [>.............................] - ETA: 14s - loss: 0.5339 - accuracy: 0.8860
 1100/25000 [>.............................] - ETA: 14s - loss: 0.5336 - accuracy: 0.8873
 1200/25000 [>.............................] - ETA: 14s - loss: 0.5318 - accuracy: 0.8908
 1300/25000 [>.............................] - ETA: 14s - loss: 0.5315 - accuracy: 0.8908
 1400/25000 [>.............................] - ETA: 14s - loss: 0.5313 - accuracy: 0.8914
 1500/25000 [>.............................] - ETA: 14s - loss: 0.5299 - accuracy: 0.8940
 1600/25000 [>.............................] - ETA: 14s - loss: 0.5297 - accuracy: 0.8944
 1700/25000 [=>............................] - ETA: 14s - loss: 0.5281 - accuracy: 0.8965
 1800/25000 [=>............................] - ETA: 14s - loss: 0.5286 - accuracy: 0.8956
 1900/25000 [=>............................] - ETA: 14s - loss: 0.5299 - accuracy: 0.8932
 2000/25000 [=>............................] - ETA: 14s - loss: 0.5286 - accuracy: 0.8945
 2100/25000 [=>............................] - ETA: 14s - loss: 0.5267 - accuracy: 0.8971
 2200/25000 [=>............................] - ETA: 14s - loss: 0.5257 - accuracy: 0.8977
 2300/25000 [=>............................] - ETA: 14s - loss: 0.5269 - accuracy: 0.8952
 2400/25000 [=>............................] - ETA: 14s - loss: 0.5277 - accuracy: 0.8933
 2500/25000 [==>...........................] - ETA: 14s - loss: 0.5279 - accuracy: 0.8924
 2600/25000 [==>...........................] - ETA: 13s - loss: 0.5267 - accuracy: 0.8942
 2700/25000 [==>...........................] - ETA: 13s - loss: 0.5265 - accuracy: 0.8944
 2800/25000 [==>...........................] - ETA: 13s - loss: 0.5262 - accuracy: 0.8954
 2900/25000 [==>...........................] - ETA: 13s - loss: 0.5254 - accuracy: 0.8969
 3000/25000 [==>...........................] - ETA: 13s - loss: 0.5246 - accuracy: 0.8980
 3100/25000 [==>...........................] - ETA: 13s - loss: 0.5246 - accuracy: 0.8990
 3200/25000 [==>...........................] - ETA: 13s - loss: 0.5244 - accuracy: 0.8994
 3300/25000 [==>...........................] - ETA: 13s - loss: 0.5241 - accuracy: 0.9000
 3400/25000 [===>..........................] - ETA: 13s - loss: 0.5249 - accuracy: 0.8985
 3500/25000 [===>..........................] - ETA: 13s - loss: 0.5253 - accuracy: 0.8977
 3600/25000 [===>..........................] - ETA: 13s - loss: 0.5257 - accuracy: 0.8967
 3700/25000 [===>..........................] - ETA: 13s - loss: 0.5263 - accuracy: 0.8951
 3800/25000 [===>..........................] - ETA: 13s - loss: 0.5265 - accuracy: 0.8950
 3900/25000 [===>..........................] - ETA: 13s - loss: 0.5264 - accuracy: 0.8946
 4000/25000 [===>..........................] - ETA: 13s - loss: 0.5258 - accuracy: 0.8950
 4100/25000 [===>..........................] - ETA: 13s - loss: 0.5256 - accuracy: 0.8949
 4200/25000 [====>.........................] - ETA: 12s - loss: 0.5250 - accuracy: 0.8957
 4300/25000 [====>.........................] - ETA: 12s - loss: 0.5244 - accuracy: 0.8965
 4400/25000 [====>.........................] - ETA: 12s - loss: 0.5244 - accuracy: 0.8970
 4500/25000 [====>.........................] - ETA: 12s - loss: 0.5243 - accuracy: 0.8973
 4600/25000 [====>.........................] - ETA: 12s - loss: 0.5241 - accuracy: 0.8967
 4700/25000 [====>.........................] - ETA: 12s - loss: 0.5246 - accuracy: 0.8955
 4800/25000 [====>.........................] - ETA: 12s - loss: 0.5249 - accuracy: 0.8950
 4900/25000 [====>.........................] - ETA: 12s - loss: 0.5245 - accuracy: 0.8957
 5000/25000 [=====>........................] - ETA: 12s - loss: 0.5246 - accuracy: 0.8954
 5100/25000 [=====>........................] - ETA: 12s - loss: 0.5245 - accuracy: 0.8955
 5200/25000 [=====>........................] - ETA: 12s - loss: 0.5246 - accuracy: 0.8950
 5300/25000 [=====>........................] - ETA: 12s - loss: 0.5241 - accuracy: 0.8955
 5400/25000 [=====>........................] - ETA: 12s - loss: 0.5241 - accuracy: 0.8950
 5500/25000 [=====>........................] - ETA: 12s - loss: 0.5239 - accuracy: 0.8947
 5600/25000 [=====>........................] - ETA: 12s - loss: 0.5236 - accuracy: 0.8952
 5700/25000 [=====>........................] - ETA: 12s - loss: 0.5232 - accuracy: 0.8953
 5800/25000 [=====>........................] - ETA: 11s - loss: 0.5236 - accuracy: 0.8948
 5900/25000 [======>.......................] - ETA: 11s - loss: 0.5238 - accuracy: 0.8941
 6000/25000 [======>.......................] - ETA: 11s - loss: 0.5239 - accuracy: 0.8938
 6100/25000 [======>.......................] - ETA: 11s - loss: 0.5242 - accuracy: 0.8926
 6200/25000 [======>.......................] - ETA: 11s - loss: 0.5244 - accuracy: 0.8926
 6300/25000 [======>.......................] - ETA: 11s - loss: 0.5244 - accuracy: 0.8929
 6400/25000 [======>.......................] - ETA: 11s - loss: 0.5239 - accuracy: 0.8934
 6500/25000 [======>.......................] - ETA: 11s - loss: 0.5235 - accuracy: 0.8940
 6600/25000 [======>.......................] - ETA: 11s - loss: 0.5234 - accuracy: 0.8941
 6700/25000 [=======>......................] - ETA: 11s - loss: 0.5232 - accuracy: 0.8943
 6800/25000 [=======>......................] - ETA: 11s - loss: 0.5228 - accuracy: 0.8949
 6900/25000 [=======>......................] - ETA: 11s - loss: 0.5228 - accuracy: 0.8948
 7000/25000 [=======>......................] - ETA: 11s - loss: 0.5225 - accuracy: 0.8953
 7100/25000 [=======>......................] - ETA: 11s - loss: 0.5220 - accuracy: 0.8958
 7200/25000 [=======>......................] - ETA: 11s - loss: 0.5223 - accuracy: 0.8956
 7300/25000 [=======>......................] - ETA: 11s - loss: 0.5223 - accuracy: 0.8952
 7400/25000 [=======>......................] - ETA: 10s - loss: 0.5216 - accuracy: 0.8959
 7500/25000 [========>.....................] - ETA: 10s - loss: 0.5215 - accuracy: 0.8960
 7600/25000 [========>.....................] - ETA: 10s - loss: 0.5214 - accuracy: 0.8961
 7700/25000 [========>.....................] - ETA: 10s - loss: 0.5212 - accuracy: 0.8964
 7800/25000 [========>.....................] - ETA: 10s - loss: 0.5209 - accuracy: 0.8967
 7900/25000 [========>.....................] - ETA: 10s - loss: 0.5206 - accuracy: 0.8972
 8000/25000 [========>.....................] - ETA: 10s - loss: 0.5203 - accuracy: 0.8972
 8100/25000 [========>.....................] - ETA: 10s - loss: 0.5202 - accuracy: 0.8973
 8200/25000 [========>.....................] - ETA: 10s - loss: 0.5197 - accuracy: 0.8980
 8300/25000 [========>.....................] - ETA: 10s - loss: 0.5195 - accuracy: 0.8982
 8400/25000 [=========>....................] - ETA: 10s - loss: 0.5195 - accuracy: 0.8982
 8500/25000 [=========>....................] - ETA: 10s - loss: 0.5191 - accuracy: 0.8987
 8600/25000 [=========>....................] - ETA: 10s - loss: 0.5189 - accuracy: 0.8987
 8700/25000 [=========>....................] - ETA: 10s - loss: 0.5188 - accuracy: 0.8990
 8800/25000 [=========>....................] - ETA: 10s - loss: 0.5185 - accuracy: 0.8994
 8900/25000 [=========>....................] - ETA: 10s - loss: 0.5184 - accuracy: 0.8992
 9000/25000 [=========>....................] - ETA: 9s - loss: 0.5183 - accuracy: 0.8991 
 9100/25000 [=========>....................] - ETA: 9s - loss: 0.5184 - accuracy: 0.8989
 9200/25000 [==========>...................] - ETA: 9s - loss: 0.5182 - accuracy: 0.8991
 9300/25000 [==========>...................] - ETA: 9s - loss: 0.5179 - accuracy: 0.8996
 9400/25000 [==========>...................] - ETA: 9s - loss: 0.5176 - accuracy: 0.9000
 9500/25000 [==========>...................] - ETA: 9s - loss: 0.5175 - accuracy: 0.9001
 9600/25000 [==========>...................] - ETA: 9s - loss: 0.5177 - accuracy: 0.9000
 9700/25000 [==========>...................] - ETA: 9s - loss: 0.5177 - accuracy: 0.8996
 9800/25000 [==========>...................] - ETA: 9s - loss: 0.5177 - accuracy: 0.8997
 9900/25000 [==========>...................] - ETA: 9s - loss: 0.5176 - accuracy: 0.8994
10000/25000 [===========>..................] - ETA: 9s - loss: 0.5177 - accuracy: 0.8991
10100/25000 [===========>..................] - ETA: 9s - loss: 0.5177 - accuracy: 0.8989
10200/25000 [===========>..................] - ETA: 9s - loss: 0.5175 - accuracy: 0.8992
10300/25000 [===========>..................] - ETA: 9s - loss: 0.5173 - accuracy: 0.8992
10400/25000 [===========>..................] - ETA: 9s - loss: 0.5174 - accuracy: 0.8991
10500/25000 [===========>..................] - ETA: 9s - loss: 0.5172 - accuracy: 0.8993
10600/25000 [===========>..................] - ETA: 8s - loss: 0.5171 - accuracy: 0.8993
10700/25000 [===========>..................] - ETA: 8s - loss: 0.5169 - accuracy: 0.8996
10800/25000 [===========>..................] - ETA: 8s - loss: 0.5169 - accuracy: 0.8994
10900/25000 [============>.................] - ETA: 8s - loss: 0.5168 - accuracy: 0.8996
11000/25000 [============>.................] - ETA: 8s - loss: 0.5167 - accuracy: 0.8996
11100/25000 [============>.................] - ETA: 8s - loss: 0.5167 - accuracy: 0.8996
11200/25000 [============>.................] - ETA: 8s - loss: 0.5163 - accuracy: 0.9001
11300/25000 [============>.................] - ETA: 8s - loss: 0.5159 - accuracy: 0.9006
11400/25000 [============>.................] - ETA: 8s - loss: 0.5157 - accuracy: 0.9007
11500/25000 [============>.................] - ETA: 8s - loss: 0.5157 - accuracy: 0.9007
11600/25000 [============>.................] - ETA: 8s - loss: 0.5157 - accuracy: 0.9003
11700/25000 [=============>................] - ETA: 8s - loss: 0.5156 - accuracy: 0.9004
11800/25000 [=============>................] - ETA: 8s - loss: 0.5160 - accuracy: 0.8997
11900/25000 [=============>................] - ETA: 8s - loss: 0.5159 - accuracy: 0.8997
12000/25000 [=============>................] - ETA: 8s - loss: 0.5159 - accuracy: 0.8997
12100/25000 [=============>................] - ETA: 8s - loss: 0.5158 - accuracy: 0.8995
12200/25000 [=============>................] - ETA: 7s - loss: 0.5157 - accuracy: 0.8995
12300/25000 [=============>................] - ETA: 7s - loss: 0.5157 - accuracy: 0.8995
12400/25000 [=============>................] - ETA: 7s - loss: 0.5154 - accuracy: 0.8997
12500/25000 [==============>...............] - ETA: 7s - loss: 0.5153 - accuracy: 0.8996
12600/25000 [==============>...............] - ETA: 7s - loss: 0.5152 - accuracy: 0.8997
12700/25000 [==============>...............] - ETA: 7s - loss: 0.5150 - accuracy: 0.8998
12800/25000 [==============>...............] - ETA: 7s - loss: 0.5150 - accuracy: 0.8996
12900/25000 [==============>...............] - ETA: 7s - loss: 0.5150 - accuracy: 0.8996
13000/25000 [==============>...............] - ETA: 7s - loss: 0.5149 - accuracy: 0.8994
13100/25000 [==============>...............] - ETA: 7s - loss: 0.5147 - accuracy: 0.8995
13200/25000 [==============>...............] - ETA: 7s - loss: 0.5148 - accuracy: 0.8991
13300/25000 [==============>...............] - ETA: 7s - loss: 0.5144 - accuracy: 0.8995
13400/25000 [===============>..............] - ETA: 7s - loss: 0.5143 - accuracy: 0.8996
13500/25000 [===============>..............] - ETA: 7s - loss: 0.5141 - accuracy: 0.8993
13600/25000 [===============>..............] - ETA: 7s - loss: 0.5141 - accuracy: 0.8995
13700/25000 [===============>..............] - ETA: 7s - loss: 0.5140 - accuracy: 0.8997
13800/25000 [===============>..............] - ETA: 6s - loss: 0.5139 - accuracy: 0.8996
13900/25000 [===============>..............] - ETA: 6s - loss: 0.5139 - accuracy: 0.8996
14000/25000 [===============>..............] - ETA: 6s - loss: 0.5139 - accuracy: 0.8993
14100/25000 [===============>..............] - ETA: 6s - loss: 0.5138 - accuracy: 0.8994
14200/25000 [================>.............] - ETA: 6s - loss: 0.5137 - accuracy: 0.8993
14300/25000 [================>.............] - ETA: 6s - loss: 0.5136 - accuracy: 0.8994
14400/25000 [================>.............] - ETA: 6s - loss: 0.5134 - accuracy: 0.8997
14500/25000 [================>.............] - ETA: 6s - loss: 0.5132 - accuracy: 0.8999
14600/25000 [================>.............] - ETA: 6s - loss: 0.5131 - accuracy: 0.9000
14700/25000 [================>.............] - ETA: 6s - loss: 0.5132 - accuracy: 0.8998
14800/25000 [================>.............] - ETA: 6s - loss: 0.5132 - accuracy: 0.8998
14900/25000 [================>.............] - ETA: 6s - loss: 0.5132 - accuracy: 0.8997
15000/25000 [=================>............] - ETA: 6s - loss: 0.5130 - accuracy: 0.9000
15100/25000 [=================>............] - ETA: 6s - loss: 0.5130 - accuracy: 0.8999
15200/25000 [=================>............] - ETA: 6s - loss: 0.5129 - accuracy: 0.9001
15300/25000 [=================>............] - ETA: 6s - loss: 0.5131 - accuracy: 0.8996
15400/25000 [=================>............] - ETA: 5s - loss: 0.5130 - accuracy: 0.8997
15500/25000 [=================>............] - ETA: 5s - loss: 0.5128 - accuracy: 0.8996
15600/25000 [=================>............] - ETA: 5s - loss: 0.5128 - accuracy: 0.8997
15700/25000 [=================>............] - ETA: 5s - loss: 0.5127 - accuracy: 0.8998
15800/25000 [=================>............] - ETA: 5s - loss: 0.5125 - accuracy: 0.9001
15900/25000 [==================>...........] - ETA: 5s - loss: 0.5124 - accuracy: 0.9001
16000/25000 [==================>...........] - ETA: 5s - loss: 0.5124 - accuracy: 0.9001
16100/25000 [==================>...........] - ETA: 5s - loss: 0.5121 - accuracy: 0.9003
16200/25000 [==================>...........] - ETA: 5s - loss: 0.5119 - accuracy: 0.9005
16300/25000 [==================>...........] - ETA: 5s - loss: 0.5116 - accuracy: 0.9007
16400/25000 [==================>...........] - ETA: 5s - loss: 0.5116 - accuracy: 0.9004
16500/25000 [==================>...........] - ETA: 5s - loss: 0.5116 - accuracy: 0.9004
16600/25000 [==================>...........] - ETA: 5s - loss: 0.5115 - accuracy: 0.9005
16700/25000 [===================>..........] - ETA: 5s - loss: 0.5114 - accuracy: 0.9005
16800/25000 [===================>..........] - ETA: 5s - loss: 0.5114 - accuracy: 0.9005
16900/25000 [===================>..........] - ETA: 5s - loss: 0.5114 - accuracy: 0.9002
17000/25000 [===================>..........] - ETA: 4s - loss: 0.5113 - accuracy: 0.9003
17100/25000 [===================>..........] - ETA: 4s - loss: 0.5112 - accuracy: 0.9004
17200/25000 [===================>..........] - ETA: 4s - loss: 0.5113 - accuracy: 0.9001
17300/25000 [===================>..........] - ETA: 4s - loss: 0.5113 - accuracy: 0.8998
17400/25000 [===================>..........] - ETA: 4s - loss: 0.5111 - accuracy: 0.9000
17500/25000 [====================>.........] - ETA: 4s - loss: 0.5110 - accuracy: 0.8999
17600/25000 [====================>.........] - ETA: 4s - loss: 0.5111 - accuracy: 0.8997
17700/25000 [====================>.........] - ETA: 4s - loss: 0.5110 - accuracy: 0.8998
17800/25000 [====================>.........] - ETA: 4s - loss: 0.5110 - accuracy: 0.8998
17900/25000 [====================>.........] - ETA: 4s - loss: 0.5108 - accuracy: 0.8999
18000/25000 [====================>.........] - ETA: 4s - loss: 0.5107 - accuracy: 0.8999
18100/25000 [====================>.........] - ETA: 4s - loss: 0.5106 - accuracy: 0.9001
18200/25000 [====================>.........] - ETA: 4s - loss: 0.5103 - accuracy: 0.9004
18300/25000 [====================>.........] - ETA: 4s - loss: 0.5103 - accuracy: 0.9002
18400/25000 [=====================>........] - ETA: 4s - loss: 0.5103 - accuracy: 0.9001
18500/25000 [=====================>........] - ETA: 4s - loss: 0.5104 - accuracy: 0.8999
18600/25000 [=====================>........] - ETA: 3s - loss: 0.5102 - accuracy: 0.9001
18700/25000 [=====================>........] - ETA: 3s - loss: 0.5101 - accuracy: 0.9000
18800/25000 [=====================>........] - ETA: 3s - loss: 0.5100 - accuracy: 0.9001
18900/25000 [=====================>........] - ETA: 3s - loss: 0.5099 - accuracy: 0.9001
19000/25000 [=====================>........] - ETA: 3s - loss: 0.5098 - accuracy: 0.9002
19100/25000 [=====================>........] - ETA: 3s - loss: 0.5098 - accuracy: 0.9002
19200/25000 [======================>.......] - ETA: 3s - loss: 0.5100 - accuracy: 0.8995
19300/25000 [======================>.......] - ETA: 3s - loss: 0.5099 - accuracy: 0.8995
19400/25000 [======================>.......] - ETA: 3s - loss: 0.5100 - accuracy: 0.8994
19500/25000 [======================>.......] - ETA: 3s - loss: 0.5099 - accuracy: 0.8994
19600/25000 [======================>.......] - ETA: 3s - loss: 0.5098 - accuracy: 0.8994
19700/25000 [======================>.......] - ETA: 3s - loss: 0.5097 - accuracy: 0.8993
19800/25000 [======================>.......] - ETA: 3s - loss: 0.5096 - accuracy: 0.8994
19900/25000 [======================>.......] - ETA: 3s - loss: 0.5096 - accuracy: 0.8993
20000/25000 [=======================>......] - ETA: 3s - loss: 0.5094 - accuracy: 0.8994
20100/25000 [=======================>......] - ETA: 3s - loss: 0.5093 - accuracy: 0.8997
20200/25000 [=======================>......] - ETA: 2s - loss: 0.5090 - accuracy: 0.8999
20300/25000 [=======================>......] - ETA: 2s - loss: 0.5089 - accuracy: 0.8998
20400/25000 [=======================>......] - ETA: 2s - loss: 0.5090 - accuracy: 0.8996
20500/25000 [=======================>......] - ETA: 2s - loss: 0.5089 - accuracy: 0.8997
20600/25000 [=======================>......] - ETA: 2s - loss: 0.5089 - accuracy: 0.8996
20700/25000 [=======================>......] - ETA: 2s - loss: 0.5088 - accuracy: 0.8996
20800/25000 [=======================>......] - ETA: 2s - loss: 0.5089 - accuracy: 0.8994
20900/25000 [========================>.....] - ETA: 2s - loss: 0.5089 - accuracy: 0.8993
21000/25000 [========================>.....] - ETA: 2s - loss: 0.5088 - accuracy: 0.8993
21100/25000 [========================>.....] - ETA: 2s - loss: 0.5088 - accuracy: 0.8991
21200/25000 [========================>.....] - ETA: 2s - loss: 0.5087 - accuracy: 0.8991
21300/25000 [========================>.....] - ETA: 2s - loss: 0.5086 - accuracy: 0.8991
21400/25000 [========================>.....] - ETA: 2s - loss: 0.5084 - accuracy: 0.8993
21500/25000 [========================>.....] - ETA: 2s - loss: 0.5084 - accuracy: 0.8993
21600/25000 [========================>.....] - ETA: 2s - loss: 0.5082 - accuracy: 0.8994
21700/25000 [=========================>....] - ETA: 2s - loss: 0.5081 - accuracy: 0.8995
21800/25000 [=========================>....] - ETA: 1s - loss: 0.5080 - accuracy: 0.8995
21900/25000 [=========================>....] - ETA: 1s - loss: 0.5080 - accuracy: 0.8993
22000/25000 [=========================>....] - ETA: 1s - loss: 0.5081 - accuracy: 0.8991
22100/25000 [=========================>....] - ETA: 1s - loss: 0.5080 - accuracy: 0.8993
22200/25000 [=========================>....] - ETA: 1s - loss: 0.5078 - accuracy: 0.8995
22300/25000 [=========================>....] - ETA: 1s - loss: 0.5078 - accuracy: 0.8996
22400/25000 [=========================>....] - ETA: 1s - loss: 0.5077 - accuracy: 0.8996
22500/25000 [==========================>...] - ETA: 1s - loss: 0.5077 - accuracy: 0.8996
22600/25000 [==========================>...] - ETA: 1s - loss: 0.5077 - accuracy: 0.8995
22700/25000 [==========================>...] - ETA: 1s - loss: 0.5076 - accuracy: 0.8996
22800/25000 [==========================>...] - ETA: 1s - loss: 0.5074 - accuracy: 0.8996
22900/25000 [==========================>...] - ETA: 1s - loss: 0.5076 - accuracy: 0.8993
23000/25000 [==========================>...] - ETA: 1s - loss: 0.5076 - accuracy: 0.8991
23100/25000 [==========================>...] - ETA: 1s - loss: 0.5075 - accuracy: 0.8993
23200/25000 [==========================>...] - ETA: 1s - loss: 0.5075 - accuracy: 0.8992
23300/25000 [==========================>...] - ETA: 1s - loss: 0.5074 - accuracy: 0.8992
23400/25000 [===========================>..] - ETA: 0s - loss: 0.5072 - accuracy: 0.8993
23500/25000 [===========================>..] - ETA: 0s - loss: 0.5071 - accuracy: 0.8992
23600/25000 [===========================>..] - ETA: 0s - loss: 0.5071 - accuracy: 0.8991
23700/25000 [===========================>..] - ETA: 0s - loss: 0.5070 - accuracy: 0.8991
23800/25000 [===========================>..] - ETA: 0s - loss: 0.5068 - accuracy: 0.8993
23900/25000 [===========================>..] - ETA: 0s - loss: 0.5068 - accuracy: 0.8991
24000/25000 [===========================>..] - ETA: 0s - loss: 0.5067 - accuracy: 0.8991
24100/25000 [===========================>..] - ETA: 0s - loss: 0.5066 - accuracy: 0.8992
24200/25000 [============================>.] - ETA: 0s - loss: 0.5067 - accuracy: 0.8991
24300/25000 [============================>.] - ETA: 0s - loss: 0.5067 - accuracy: 0.8990
24400/25000 [============================>.] - ETA: 0s - loss: 0.5067 - accuracy: 0.8989
24500/25000 [============================>.] - ETA: 0s - loss: 0.5068 - accuracy: 0.8987
24600/25000 [============================>.] - ETA: 0s - loss: 0.5069 - accuracy: 0.8985
24700/25000 [============================>.] - ETA: 0s - loss: 0.5071 - accuracy: 0.8981
24800/25000 [============================>.] - ETA: 0s - loss: 0.5070 - accuracy: 0.8981
24900/25000 [============================>.] - ETA: 0s - loss: 0.5070 - accuracy: 0.8980
25000/25000 [==============================] - 20s 787us/step - loss: 0.5070 - accuracy: 0.8979 - val_loss: 0.5111 - val_accuracy: 0.8619
Epoch 4/10

  100/25000 [..............................] - ETA: 15s - loss: 0.4680 - accuracy: 0.9300
  200/25000 [..............................] - ETA: 15s - loss: 0.4622 - accuracy: 0.9400
  300/25000 [..............................] - ETA: 15s - loss: 0.4610 - accuracy: 0.9433
  400/25000 [..............................] - ETA: 15s - loss: 0.4624 - accuracy: 0.9425
  500/25000 [..............................] - ETA: 15s - loss: 0.4638 - accuracy: 0.9360
  600/25000 [..............................] - ETA: 15s - loss: 0.4652 - accuracy: 0.9317
  700/25000 [..............................] - ETA: 15s - loss: 0.4639 - accuracy: 0.9329
  800/25000 [..............................] - ETA: 15s - loss: 0.4657 - accuracy: 0.9300
  900/25000 [>.............................] - ETA: 14s - loss: 0.4664 - accuracy: 0.9300
 1000/25000 [>.............................] - ETA: 14s - loss: 0.4664 - accuracy: 0.9300
 1100/25000 [>.............................] - ETA: 14s - loss: 0.4672 - accuracy: 0.9291
 1200/25000 [>.............................] - ETA: 14s - loss: 0.4697 - accuracy: 0.9250
 1300/25000 [>.............................] - ETA: 14s - loss: 0.4699 - accuracy: 0.9238
 1400/25000 [>.............................] - ETA: 14s - loss: 0.4676 - accuracy: 0.9279
 1500/25000 [>.............................] - ETA: 14s - loss: 0.4686 - accuracy: 0.9267
 1600/25000 [>.............................] - ETA: 14s - loss: 0.4694 - accuracy: 0.9262
 1700/25000 [=>............................] - ETA: 14s - loss: 0.4706 - accuracy: 0.9241
 1800/25000 [=>............................] - ETA: 14s - loss: 0.4696 - accuracy: 0.9256
 1900/25000 [=>............................] - ETA: 14s - loss: 0.4714 - accuracy: 0.9226
 2000/25000 [=>............................] - ETA: 14s - loss: 0.4689 - accuracy: 0.9265
 2100/25000 [=>............................] - ETA: 14s - loss: 0.4708 - accuracy: 0.9238
 2200/25000 [=>............................] - ETA: 14s - loss: 0.4691 - accuracy: 0.9259
 2300/25000 [=>............................] - ETA: 14s - loss: 0.4684 - accuracy: 0.9265
 2400/25000 [=>............................] - ETA: 14s - loss: 0.4695 - accuracy: 0.9254
 2500/25000 [==>...........................] - ETA: 14s - loss: 0.4698 - accuracy: 0.9252
 2600/25000 [==>...........................] - ETA: 13s - loss: 0.4698 - accuracy: 0.9250
 2700/25000 [==>...........................] - ETA: 13s - loss: 0.4703 - accuracy: 0.9256
 2800/25000 [==>...........................] - ETA: 13s - loss: 0.4704 - accuracy: 0.9257
 2900/25000 [==>...........................] - ETA: 13s - loss: 0.4705 - accuracy: 0.9259
 3000/25000 [==>...........................] - ETA: 13s - loss: 0.4697 - accuracy: 0.9263
 3100/25000 [==>...........................] - ETA: 13s - loss: 0.4707 - accuracy: 0.9248
 3200/25000 [==>...........................] - ETA: 13s - loss: 0.4719 - accuracy: 0.9231
 3300/25000 [==>...........................] - ETA: 13s - loss: 0.4709 - accuracy: 0.9245
 3400/25000 [===>..........................] - ETA: 13s - loss: 0.4711 - accuracy: 0.9244
 3500/25000 [===>..........................] - ETA: 13s - loss: 0.4717 - accuracy: 0.9231
 3600/25000 [===>..........................] - ETA: 13s - loss: 0.4713 - accuracy: 0.9236
 3700/25000 [===>..........................] - ETA: 13s - loss: 0.4714 - accuracy: 0.9232
 3800/25000 [===>..........................] - ETA: 13s - loss: 0.4720 - accuracy: 0.9221
 3900/25000 [===>..........................] - ETA: 13s - loss: 0.4712 - accuracy: 0.9233
 4000/25000 [===>..........................] - ETA: 13s - loss: 0.4708 - accuracy: 0.9237
 4100/25000 [===>..........................] - ETA: 13s - loss: 0.4708 - accuracy: 0.9232
 4200/25000 [====>.........................] - ETA: 12s - loss: 0.4702 - accuracy: 0.9238
 4300/25000 [====>.........................] - ETA: 12s - loss: 0.4697 - accuracy: 0.9240
 4400/25000 [====>.........................] - ETA: 12s - loss: 0.4694 - accuracy: 0.9243
 4500/25000 [====>.........................] - ETA: 12s - loss: 0.4699 - accuracy: 0.9236
 4600/25000 [====>.........................] - ETA: 12s - loss: 0.4704 - accuracy: 0.9226
 4700/25000 [====>.........................] - ETA: 12s - loss: 0.4705 - accuracy: 0.9219
 4800/25000 [====>.........................] - ETA: 12s - loss: 0.4701 - accuracy: 0.9219
 4900/25000 [====>.........................] - ETA: 12s - loss: 0.4710 - accuracy: 0.9202
 5000/25000 [=====>........................] - ETA: 12s - loss: 0.4712 - accuracy: 0.9198
 5100/25000 [=====>........................] - ETA: 12s - loss: 0.4714 - accuracy: 0.9194
 5200/25000 [=====>........................] - ETA: 12s - loss: 0.4713 - accuracy: 0.9192
 5300/25000 [=====>........................] - ETA: 12s - loss: 0.4708 - accuracy: 0.9198
 5400/25000 [=====>........................] - ETA: 12s - loss: 0.4707 - accuracy: 0.9202
 5500/25000 [=====>........................] - ETA: 12s - loss: 0.4709 - accuracy: 0.9198
 5600/25000 [=====>........................] - ETA: 12s - loss: 0.4708 - accuracy: 0.9202
 5700/25000 [=====>........................] - ETA: 12s - loss: 0.4707 - accuracy: 0.9200
 5800/25000 [=====>........................] - ETA: 11s - loss: 0.4706 - accuracy: 0.9198
 5900/25000 [======>.......................] - ETA: 11s - loss: 0.4707 - accuracy: 0.9197
 6000/25000 [======>.......................] - ETA: 11s - loss: 0.4706 - accuracy: 0.9197
 6100/25000 [======>.......................] - ETA: 11s - loss: 0.4702 - accuracy: 0.9200
 6200/25000 [======>.......................] - ETA: 11s - loss: 0.4700 - accuracy: 0.9205
 6300/25000 [======>.......................] - ETA: 11s - loss: 0.4700 - accuracy: 0.9205
 6400/25000 [======>.......................] - ETA: 11s - loss: 0.4696 - accuracy: 0.9209
 6500/25000 [======>.......................] - ETA: 11s - loss: 0.4695 - accuracy: 0.9211
 6600/25000 [======>.......................] - ETA: 11s - loss: 0.4693 - accuracy: 0.9215
 6700/25000 [=======>......................] - ETA: 11s - loss: 0.4694 - accuracy: 0.9213
 6800/25000 [=======>......................] - ETA: 11s - loss: 0.4694 - accuracy: 0.9210
 6900/25000 [=======>......................] - ETA: 11s - loss: 0.4691 - accuracy: 0.9213
 7000/25000 [=======>......................] - ETA: 11s - loss: 0.4689 - accuracy: 0.9214
 7100/25000 [=======>......................] - ETA: 11s - loss: 0.4688 - accuracy: 0.9215
 7200/25000 [=======>......................] - ETA: 11s - loss: 0.4683 - accuracy: 0.9221
 7300/25000 [=======>......................] - ETA: 11s - loss: 0.4684 - accuracy: 0.9219
 7400/25000 [=======>......................] - ETA: 10s - loss: 0.4689 - accuracy: 0.9209
 7500/25000 [========>.....................] - ETA: 10s - loss: 0.4691 - accuracy: 0.9205
 7600/25000 [========>.....................] - ETA: 10s - loss: 0.4689 - accuracy: 0.9208
 7700/25000 [========>.....................] - ETA: 10s - loss: 0.4686 - accuracy: 0.9213
 7800/25000 [========>.....................] - ETA: 10s - loss: 0.4685 - accuracy: 0.9214
 7900/25000 [========>.....................] - ETA: 10s - loss: 0.4684 - accuracy: 0.9214
 8000/25000 [========>.....................] - ETA: 10s - loss: 0.4680 - accuracy: 0.9216
 8100/25000 [========>.....................] - ETA: 10s - loss: 0.4679 - accuracy: 0.9219
 8200/25000 [========>.....................] - ETA: 10s - loss: 0.4682 - accuracy: 0.9213
 8300/25000 [========>.....................] - ETA: 10s - loss: 0.4679 - accuracy: 0.9217
 8400/25000 [=========>....................] - ETA: 10s - loss: 0.4674 - accuracy: 0.9223
 8500/25000 [=========>....................] - ETA: 10s - loss: 0.4675 - accuracy: 0.9221
 8600/25000 [=========>....................] - ETA: 10s - loss: 0.4674 - accuracy: 0.9224
 8700/25000 [=========>....................] - ETA: 10s - loss: 0.4672 - accuracy: 0.9226
 8800/25000 [=========>....................] - ETA: 10s - loss: 0.4674 - accuracy: 0.9223
 8900/25000 [=========>....................] - ETA: 10s - loss: 0.4672 - accuracy: 0.9222
 9000/25000 [=========>....................] - ETA: 10s - loss: 0.4671 - accuracy: 0.9224
 9100/25000 [=========>....................] - ETA: 9s - loss: 0.4676 - accuracy: 0.9218 
 9200/25000 [==========>...................] - ETA: 9s - loss: 0.4676 - accuracy: 0.9215
 9300/25000 [==========>...................] - ETA: 9s - loss: 0.4676 - accuracy: 0.9216
 9400/25000 [==========>...................] - ETA: 9s - loss: 0.4673 - accuracy: 0.9219
 9500/25000 [==========>...................] - ETA: 9s - loss: 0.4669 - accuracy: 0.9224
 9600/25000 [==========>...................] - ETA: 9s - loss: 0.4668 - accuracy: 0.9224
 9700/25000 [==========>...................] - ETA: 9s - loss: 0.4667 - accuracy: 0.9222
 9800/25000 [==========>...................] - ETA: 9s - loss: 0.4665 - accuracy: 0.9223
 9900/25000 [==========>...................] - ETA: 9s - loss: 0.4663 - accuracy: 0.9226
10000/25000 [===========>..................] - ETA: 9s - loss: 0.4660 - accuracy: 0.9229
10100/25000 [===========>..................] - ETA: 9s - loss: 0.4659 - accuracy: 0.9229
10200/25000 [===========>..................] - ETA: 9s - loss: 0.4659 - accuracy: 0.9227
10300/25000 [===========>..................] - ETA: 9s - loss: 0.4659 - accuracy: 0.9226
10400/25000 [===========>..................] - ETA: 9s - loss: 0.4661 - accuracy: 0.9221
10500/25000 [===========>..................] - ETA: 9s - loss: 0.4660 - accuracy: 0.9223
10600/25000 [===========>..................] - ETA: 8s - loss: 0.4661 - accuracy: 0.9220
10700/25000 [===========>..................] - ETA: 8s - loss: 0.4660 - accuracy: 0.9220
10800/25000 [===========>..................] - ETA: 8s - loss: 0.4659 - accuracy: 0.9221
10900/25000 [============>.................] - ETA: 8s - loss: 0.4659 - accuracy: 0.9220
11000/25000 [============>.................] - ETA: 8s - loss: 0.4662 - accuracy: 0.9214
11100/25000 [============>.................] - ETA: 8s - loss: 0.4660 - accuracy: 0.9217
11200/25000 [============>.................] - ETA: 8s - loss: 0.4662 - accuracy: 0.9212
11300/25000 [============>.................] - ETA: 8s - loss: 0.4659 - accuracy: 0.9213
11400/25000 [============>.................] - ETA: 8s - loss: 0.4659 - accuracy: 0.9212
11500/25000 [============>.................] - ETA: 8s - loss: 0.4662 - accuracy: 0.9208
11600/25000 [============>.................] - ETA: 8s - loss: 0.4663 - accuracy: 0.9204
11700/25000 [=============>................] - ETA: 8s - loss: 0.4663 - accuracy: 0.9206
11800/25000 [=============>................] - ETA: 8s - loss: 0.4661 - accuracy: 0.9208
11900/25000 [=============>................] - ETA: 8s - loss: 0.4660 - accuracy: 0.9208
12000/25000 [=============>................] - ETA: 8s - loss: 0.4660 - accuracy: 0.9208
12100/25000 [=============>................] - ETA: 8s - loss: 0.4656 - accuracy: 0.9212
12200/25000 [=============>................] - ETA: 7s - loss: 0.4658 - accuracy: 0.9208
12300/25000 [=============>................] - ETA: 7s - loss: 0.4659 - accuracy: 0.9207
12400/25000 [=============>................] - ETA: 7s - loss: 0.4658 - accuracy: 0.9206
12500/25000 [==============>...............] - ETA: 7s - loss: 0.4659 - accuracy: 0.9203
12600/25000 [==============>...............] - ETA: 7s - loss: 0.4656 - accuracy: 0.9206
12700/25000 [==============>...............] - ETA: 7s - loss: 0.4654 - accuracy: 0.9209
12800/25000 [==============>...............] - ETA: 7s - loss: 0.4653 - accuracy: 0.9209
12900/25000 [==============>...............] - ETA: 7s - loss: 0.4652 - accuracy: 0.9209
13000/25000 [==============>...............] - ETA: 7s - loss: 0.4653 - accuracy: 0.9208
13100/25000 [==============>...............] - ETA: 7s - loss: 0.4652 - accuracy: 0.9208
13200/25000 [==============>...............] - ETA: 7s - loss: 0.4654 - accuracy: 0.9204
13300/25000 [==============>...............] - ETA: 7s - loss: 0.4653 - accuracy: 0.9205
13400/25000 [===============>..............] - ETA: 7s - loss: 0.4652 - accuracy: 0.9204
13500/25000 [===============>..............] - ETA: 7s - loss: 0.4651 - accuracy: 0.9203
13600/25000 [===============>..............] - ETA: 7s - loss: 0.4653 - accuracy: 0.9199
13700/25000 [===============>..............] - ETA: 7s - loss: 0.4654 - accuracy: 0.9194
13800/25000 [===============>..............] - ETA: 6s - loss: 0.4652 - accuracy: 0.9196
13900/25000 [===============>..............] - ETA: 6s - loss: 0.4653 - accuracy: 0.9194
14000/25000 [===============>..............] - ETA: 6s - loss: 0.4653 - accuracy: 0.9194
14100/25000 [===============>..............] - ETA: 6s - loss: 0.4650 - accuracy: 0.9198
14200/25000 [================>.............] - ETA: 6s - loss: 0.4651 - accuracy: 0.9196
14300/25000 [================>.............] - ETA: 6s - loss: 0.4651 - accuracy: 0.9194
14400/25000 [================>.............] - ETA: 6s - loss: 0.4650 - accuracy: 0.9195
14500/25000 [================>.............] - ETA: 6s - loss: 0.4650 - accuracy: 0.9194
14600/25000 [================>.............] - ETA: 6s - loss: 0.4650 - accuracy: 0.9194
14700/25000 [================>.............] - ETA: 6s - loss: 0.4648 - accuracy: 0.9195
14800/25000 [================>.............] - ETA: 6s - loss: 0.4648 - accuracy: 0.9194
14900/25000 [================>.............] - ETA: 6s - loss: 0.4648 - accuracy: 0.9193
15000/25000 [=================>............] - ETA: 6s - loss: 0.4648 - accuracy: 0.9193
15100/25000 [=================>............] - ETA: 6s - loss: 0.4646 - accuracy: 0.9195
15200/25000 [=================>............] - ETA: 6s - loss: 0.4648 - accuracy: 0.9192
15300/25000 [=================>............] - ETA: 6s - loss: 0.4647 - accuracy: 0.9192
15400/25000 [=================>............] - ETA: 5s - loss: 0.4648 - accuracy: 0.9190
15500/25000 [=================>............] - ETA: 5s - loss: 0.4647 - accuracy: 0.9192
15600/25000 [=================>............] - ETA: 5s - loss: 0.4649 - accuracy: 0.9187
15700/25000 [=================>............] - ETA: 5s - loss: 0.4651 - accuracy: 0.9185
15800/25000 [=================>............] - ETA: 5s - loss: 0.4651 - accuracy: 0.9184
15900/25000 [==================>...........] - ETA: 5s - loss: 0.4651 - accuracy: 0.9184
16000/25000 [==================>...........] - ETA: 5s - loss: 0.4649 - accuracy: 0.9185
16100/25000 [==================>...........] - ETA: 5s - loss: 0.4646 - accuracy: 0.9188
16200/25000 [==================>...........] - ETA: 5s - loss: 0.4646 - accuracy: 0.9188
16300/25000 [==================>...........] - ETA: 5s - loss: 0.4648 - accuracy: 0.9185
16400/25000 [==================>...........] - ETA: 5s - loss: 0.4648 - accuracy: 0.9185
16500/25000 [==================>...........] - ETA: 5s - loss: 0.4646 - accuracy: 0.9185
16600/25000 [==================>...........] - ETA: 5s - loss: 0.4643 - accuracy: 0.9188
16700/25000 [===================>..........] - ETA: 5s - loss: 0.4644 - accuracy: 0.9186
16800/25000 [===================>..........] - ETA: 5s - loss: 0.4642 - accuracy: 0.9187
16900/25000 [===================>..........] - ETA: 5s - loss: 0.4641 - accuracy: 0.9188
17000/25000 [===================>..........] - ETA: 4s - loss: 0.4640 - accuracy: 0.9188
17100/25000 [===================>..........] - ETA: 4s - loss: 0.4640 - accuracy: 0.9188
17200/25000 [===================>..........] - ETA: 4s - loss: 0.4640 - accuracy: 0.9188
17300/25000 [===================>..........] - ETA: 4s - loss: 0.4640 - accuracy: 0.9186
17400/25000 [===================>..........] - ETA: 4s - loss: 0.4640 - accuracy: 0.9183
17500/25000 [====================>.........] - ETA: 4s - loss: 0.4640 - accuracy: 0.9182
17600/25000 [====================>.........] - ETA: 4s - loss: 0.4639 - accuracy: 0.9182
17700/25000 [====================>.........] - ETA: 4s - loss: 0.4640 - accuracy: 0.9179
17800/25000 [====================>.........] - ETA: 4s - loss: 0.4638 - accuracy: 0.9180
17900/25000 [====================>.........] - ETA: 4s - loss: 0.4638 - accuracy: 0.9179
18000/25000 [====================>.........] - ETA: 4s - loss: 0.4639 - accuracy: 0.9178
18100/25000 [====================>.........] - ETA: 4s - loss: 0.4639 - accuracy: 0.9178
18200/25000 [====================>.........] - ETA: 4s - loss: 0.4639 - accuracy: 0.9176
18300/25000 [====================>.........] - ETA: 4s - loss: 0.4639 - accuracy: 0.9176
18400/25000 [=====================>........] - ETA: 4s - loss: 0.4640 - accuracy: 0.9174
18500/25000 [=====================>........] - ETA: 4s - loss: 0.4638 - accuracy: 0.9175
18600/25000 [=====================>........] - ETA: 3s - loss: 0.4637 - accuracy: 0.9175
18700/25000 [=====================>........] - ETA: 3s - loss: 0.4636 - accuracy: 0.9175
18800/25000 [=====================>........] - ETA: 3s - loss: 0.4635 - accuracy: 0.9176
18900/25000 [=====================>........] - ETA: 3s - loss: 0.4634 - accuracy: 0.9177
19000/25000 [=====================>........] - ETA: 3s - loss: 0.4634 - accuracy: 0.9177
19100/25000 [=====================>........] - ETA: 3s - loss: 0.4633 - accuracy: 0.9177
19200/25000 [======================>.......] - ETA: 3s - loss: 0.4632 - accuracy: 0.9177
19300/25000 [======================>.......] - ETA: 3s - loss: 0.4634 - accuracy: 0.9173
19400/25000 [======================>.......] - ETA: 3s - loss: 0.4632 - accuracy: 0.9173
19500/25000 [======================>.......] - ETA: 3s - loss: 0.4631 - accuracy: 0.9175
19600/25000 [======================>.......] - ETA: 3s - loss: 0.4629 - accuracy: 0.9176
19700/25000 [======================>.......] - ETA: 3s - loss: 0.4630 - accuracy: 0.9174
19800/25000 [======================>.......] - ETA: 3s - loss: 0.4628 - accuracy: 0.9176
19900/25000 [======================>.......] - ETA: 3s - loss: 0.4625 - accuracy: 0.9177
20000/25000 [=======================>......] - ETA: 3s - loss: 0.4625 - accuracy: 0.9176
20100/25000 [=======================>......] - ETA: 3s - loss: 0.4626 - accuracy: 0.9176
20200/25000 [=======================>......] - ETA: 2s - loss: 0.4625 - accuracy: 0.9176
20300/25000 [=======================>......] - ETA: 2s - loss: 0.4626 - accuracy: 0.9173
20400/25000 [=======================>......] - ETA: 2s - loss: 0.4627 - accuracy: 0.9171
20500/25000 [=======================>......] - ETA: 2s - loss: 0.4627 - accuracy: 0.9170
20600/25000 [=======================>......] - ETA: 2s - loss: 0.4626 - accuracy: 0.9171
20700/25000 [=======================>......] - ETA: 2s - loss: 0.4626 - accuracy: 0.9171
20800/25000 [=======================>......] - ETA: 2s - loss: 0.4627 - accuracy: 0.9167
20900/25000 [========================>.....] - ETA: 2s - loss: 0.4625 - accuracy: 0.9168
21000/25000 [========================>.....] - ETA: 2s - loss: 0.4627 - accuracy: 0.9164
21100/25000 [========================>.....] - ETA: 2s - loss: 0.4626 - accuracy: 0.9164
21200/25000 [========================>.....] - ETA: 2s - loss: 0.4625 - accuracy: 0.9166
21300/25000 [========================>.....] - ETA: 2s - loss: 0.4624 - accuracy: 0.9166
21400/25000 [========================>.....] - ETA: 2s - loss: 0.4623 - accuracy: 0.9167
21500/25000 [========================>.....] - ETA: 2s - loss: 0.4622 - accuracy: 0.9167
21600/25000 [========================>.....] - ETA: 2s - loss: 0.4622 - accuracy: 0.9166
21700/25000 [=========================>....] - ETA: 2s - loss: 0.4622 - accuracy: 0.9166
21800/25000 [=========================>....] - ETA: 1s - loss: 0.4622 - accuracy: 0.9166
21900/25000 [=========================>....] - ETA: 1s - loss: 0.4622 - accuracy: 0.9165
22000/25000 [=========================>....] - ETA: 1s - loss: 0.4621 - accuracy: 0.9165
22100/25000 [=========================>....] - ETA: 1s - loss: 0.4622 - accuracy: 0.9165
22200/25000 [=========================>....] - ETA: 1s - loss: 0.4620 - accuracy: 0.9167
22300/25000 [=========================>....] - ETA: 1s - loss: 0.4619 - accuracy: 0.9167
22400/25000 [=========================>....] - ETA: 1s - loss: 0.4620 - accuracy: 0.9164
22500/25000 [==========================>...] - ETA: 1s - loss: 0.4619 - accuracy: 0.9164
22600/25000 [==========================>...] - ETA: 1s - loss: 0.4620 - accuracy: 0.9163
22700/25000 [==========================>...] - ETA: 1s - loss: 0.4618 - accuracy: 0.9165
22800/25000 [==========================>...] - ETA: 1s - loss: 0.4616 - accuracy: 0.9165
22900/25000 [==========================>...] - ETA: 1s - loss: 0.4616 - accuracy: 0.9164
23000/25000 [==========================>...] - ETA: 1s - loss: 0.4616 - accuracy: 0.9163
23100/25000 [==========================>...] - ETA: 1s - loss: 0.4615 - accuracy: 0.9163
23200/25000 [==========================>...] - ETA: 1s - loss: 0.4615 - accuracy: 0.9162
23300/25000 [==========================>...] - ETA: 1s - loss: 0.4615 - accuracy: 0.9161
23400/25000 [===========================>..] - ETA: 0s - loss: 0.4614 - accuracy: 0.9162
23500/25000 [===========================>..] - ETA: 0s - loss: 0.4613 - accuracy: 0.9163
23600/25000 [===========================>..] - ETA: 0s - loss: 0.4614 - accuracy: 0.9161
23700/25000 [===========================>..] - ETA: 0s - loss: 0.4615 - accuracy: 0.9159
23800/25000 [===========================>..] - ETA: 0s - loss: 0.4614 - accuracy: 0.9159
23900/25000 [===========================>..] - ETA: 0s - loss: 0.4614 - accuracy: 0.9159
24000/25000 [===========================>..] - ETA: 0s - loss: 0.4611 - accuracy: 0.9161
24100/25000 [===========================>..] - ETA: 0s - loss: 0.4611 - accuracy: 0.9161
24200/25000 [============================>.] - ETA: 0s - loss: 0.4611 - accuracy: 0.9161
24300/25000 [============================>.] - ETA: 0s - loss: 0.4610 - accuracy: 0.9160
24400/25000 [============================>.] - ETA: 0s - loss: 0.4610 - accuracy: 0.9160
24500/25000 [============================>.] - ETA: 0s - loss: 0.4609 - accuracy: 0.9159
24600/25000 [============================>.] - ETA: 0s - loss: 0.4610 - accuracy: 0.9158
24700/25000 [============================>.] - ETA: 0s - loss: 0.4609 - accuracy: 0.9158
24800/25000 [============================>.] - ETA: 0s - loss: 0.4608 - accuracy: 0.9159
24900/25000 [============================>.] - ETA: 0s - loss: 0.4608 - accuracy: 0.9158
25000/25000 [==============================] - 20s 789us/step - loss: 0.4608 - accuracy: 0.9156 - val_loss: 0.4830 - val_accuracy: 0.8649
Epoch 5/10

  100/25000 [..............................] - ETA: 15s - loss: 0.4312 - accuracy: 0.9300
  200/25000 [..............................] - ETA: 16s - loss: 0.4165 - accuracy: 0.9500
  300/25000 [..............................] - ETA: 15s - loss: 0.4203 - accuracy: 0.9367
  400/25000 [..............................] - ETA: 15s - loss: 0.4168 - accuracy: 0.9450
  500/25000 [..............................] - ETA: 15s - loss: 0.4192 - accuracy: 0.9420
  600/25000 [..............................] - ETA: 15s - loss: 0.4195 - accuracy: 0.9417
  700/25000 [..............................] - ETA: 15s - loss: 0.4194 - accuracy: 0.9429
  800/25000 [..............................] - ETA: 15s - loss: 0.4183 - accuracy: 0.9450
  900/25000 [>.............................] - ETA: 15s - loss: 0.4200 - accuracy: 0.9422
 1000/25000 [>.............................] - ETA: 14s - loss: 0.4185 - accuracy: 0.9460
 1100/25000 [>.............................] - ETA: 14s - loss: 0.4224 - accuracy: 0.9400
 1200/25000 [>.............................] - ETA: 14s - loss: 0.4237 - accuracy: 0.9383
 1300/25000 [>.............................] - ETA: 14s - loss: 0.4228 - accuracy: 0.9400
 1400/25000 [>.............................] - ETA: 14s - loss: 0.4242 - accuracy: 0.9386
 1500/25000 [>.............................] - ETA: 14s - loss: 0.4254 - accuracy: 0.9373
 1600/25000 [>.............................] - ETA: 14s - loss: 0.4257 - accuracy: 0.9369
 1700/25000 [=>............................] - ETA: 14s - loss: 0.4243 - accuracy: 0.9382
 1800/25000 [=>............................] - ETA: 14s - loss: 0.4255 - accuracy: 0.9367
 1900/25000 [=>............................] - ETA: 14s - loss: 0.4251 - accuracy: 0.9379
 2000/25000 [=>............................] - ETA: 14s - loss: 0.4253 - accuracy: 0.9375
 2100/25000 [=>............................] - ETA: 14s - loss: 0.4253 - accuracy: 0.9381
 2200/25000 [=>............................] - ETA: 14s - loss: 0.4251 - accuracy: 0.9377
 2300/25000 [=>............................] - ETA: 14s - loss: 0.4251 - accuracy: 0.9378
 2400/25000 [=>............................] - ETA: 14s - loss: 0.4251 - accuracy: 0.9379
 2500/25000 [==>...........................] - ETA: 14s - loss: 0.4252 - accuracy: 0.9380
 2600/25000 [==>...........................] - ETA: 14s - loss: 0.4254 - accuracy: 0.9373
 2700/25000 [==>...........................] - ETA: 13s - loss: 0.4262 - accuracy: 0.9363
 2800/25000 [==>...........................] - ETA: 13s - loss: 0.4272 - accuracy: 0.9354
 2900/25000 [==>...........................] - ETA: 13s - loss: 0.4268 - accuracy: 0.9359
 3000/25000 [==>...........................] - ETA: 13s - loss: 0.4283 - accuracy: 0.9343
 3100/25000 [==>...........................] - ETA: 13s - loss: 0.4287 - accuracy: 0.9339
 3200/25000 [==>...........................] - ETA: 13s - loss: 0.4287 - accuracy: 0.9334
 3300/25000 [==>...........................] - ETA: 13s - loss: 0.4286 - accuracy: 0.9336
 3400/25000 [===>..........................] - ETA: 13s - loss: 0.4294 - accuracy: 0.9324
 3500/25000 [===>..........................] - ETA: 13s - loss: 0.4295 - accuracy: 0.9323
 3600/25000 [===>..........................] - ETA: 13s - loss: 0.4291 - accuracy: 0.9325
 3700/25000 [===>..........................] - ETA: 13s - loss: 0.4291 - accuracy: 0.9327
 3800/25000 [===>..........................] - ETA: 13s - loss: 0.4295 - accuracy: 0.9318
 3900/25000 [===>..........................] - ETA: 13s - loss: 0.4290 - accuracy: 0.9326
 4000/25000 [===>..........................] - ETA: 13s - loss: 0.4294 - accuracy: 0.9323
 4100/25000 [===>..........................] - ETA: 13s - loss: 0.4287 - accuracy: 0.9329
 4200/25000 [====>.........................] - ETA: 12s - loss: 0.4288 - accuracy: 0.9329
 4300/25000 [====>.........................] - ETA: 12s - loss: 0.4286 - accuracy: 0.9328
 4400/25000 [====>.........................] - ETA: 12s - loss: 0.4288 - accuracy: 0.9327
 4500/25000 [====>.........................] - ETA: 12s - loss: 0.4282 - accuracy: 0.9336
 4600/25000 [====>.........................] - ETA: 12s - loss: 0.4280 - accuracy: 0.9339
 4700/25000 [====>.........................] - ETA: 12s - loss: 0.4282 - accuracy: 0.9334
 4800/25000 [====>.........................] - ETA: 12s - loss: 0.4287 - accuracy: 0.9327
 4900/25000 [====>.........................] - ETA: 12s - loss: 0.4278 - accuracy: 0.9337
 5000/25000 [=====>........................] - ETA: 12s - loss: 0.4282 - accuracy: 0.9332
 5100/25000 [=====>........................] - ETA: 12s - loss: 0.4281 - accuracy: 0.9329
 5200/25000 [=====>........................] - ETA: 12s - loss: 0.4281 - accuracy: 0.9325
 5300/25000 [=====>........................] - ETA: 12s - loss: 0.4284 - accuracy: 0.9323
 5400/25000 [=====>........................] - ETA: 12s - loss: 0.4288 - accuracy: 0.9315
 5500/25000 [=====>........................] - ETA: 12s - loss: 0.4294 - accuracy: 0.9307
 5600/25000 [=====>........................] - ETA: 12s - loss: 0.4290 - accuracy: 0.9311
 5700/25000 [=====>........................] - ETA: 12s - loss: 0.4283 - accuracy: 0.9319
 5800/25000 [=====>........................] - ETA: 11s - loss: 0.4284 - accuracy: 0.9317
 5900/25000 [======>.......................] - ETA: 11s - loss: 0.4288 - accuracy: 0.9310
 6000/25000 [======>.......................] - ETA: 11s - loss: 0.4289 - accuracy: 0.9308
 6100/25000 [======>.......................] - ETA: 11s - loss: 0.4289 - accuracy: 0.9308
 6200/25000 [======>.......................] - ETA: 11s - loss: 0.4286 - accuracy: 0.9313
 6300/25000 [======>.......................] - ETA: 11s - loss: 0.4286 - accuracy: 0.9313
 6400/25000 [======>.......................] - ETA: 11s - loss: 0.4280 - accuracy: 0.9319
 6500/25000 [======>.......................] - ETA: 11s - loss: 0.4278 - accuracy: 0.9322
 6600/25000 [======>.......................] - ETA: 11s - loss: 0.4273 - accuracy: 0.9326
 6700/25000 [=======>......................] - ETA: 11s - loss: 0.4271 - accuracy: 0.9328
 6800/25000 [=======>......................] - ETA: 11s - loss: 0.4272 - accuracy: 0.9325
 6900/25000 [=======>......................] - ETA: 11s - loss: 0.4273 - accuracy: 0.9326
 7000/25000 [=======>......................] - ETA: 11s - loss: 0.4273 - accuracy: 0.9324
 7100/25000 [=======>......................] - ETA: 11s - loss: 0.4273 - accuracy: 0.9323
 7200/25000 [=======>......................] - ETA: 11s - loss: 0.4272 - accuracy: 0.9324
 7300/25000 [=======>......................] - ETA: 11s - loss: 0.4270 - accuracy: 0.9326
 7400/25000 [=======>......................] - ETA: 10s - loss: 0.4266 - accuracy: 0.9331
 7500/25000 [========>.....................] - ETA: 10s - loss: 0.4262 - accuracy: 0.9333
 7600/25000 [========>.....................] - ETA: 10s - loss: 0.4264 - accuracy: 0.9329
 7700/25000 [========>.....................] - ETA: 10s - loss: 0.4261 - accuracy: 0.9331
 7800/25000 [========>.....................] - ETA: 10s - loss: 0.4263 - accuracy: 0.9328
 7900/25000 [========>.....................] - ETA: 10s - loss: 0.4262 - accuracy: 0.9329
 8000/25000 [========>.....................] - ETA: 10s - loss: 0.4265 - accuracy: 0.9325
 8100/25000 [========>.....................] - ETA: 10s - loss: 0.4261 - accuracy: 0.9330
 8200/25000 [========>.....................] - ETA: 10s - loss: 0.4258 - accuracy: 0.9333
 8300/25000 [========>.....................] - ETA: 10s - loss: 0.4259 - accuracy: 0.9331
 8400/25000 [=========>....................] - ETA: 10s - loss: 0.4256 - accuracy: 0.9336
 8500/25000 [=========>....................] - ETA: 10s - loss: 0.4256 - accuracy: 0.9333
 8600/25000 [=========>....................] - ETA: 10s - loss: 0.4257 - accuracy: 0.9334
 8700/25000 [=========>....................] - ETA: 10s - loss: 0.4255 - accuracy: 0.9336
 8800/25000 [=========>....................] - ETA: 10s - loss: 0.4257 - accuracy: 0.9332
 8900/25000 [=========>....................] - ETA: 10s - loss: 0.4254 - accuracy: 0.9335
 9000/25000 [=========>....................] - ETA: 9s - loss: 0.4254 - accuracy: 0.9336 
 9100/25000 [=========>....................] - ETA: 9s - loss: 0.4254 - accuracy: 0.9334
 9200/25000 [==========>...................] - ETA: 9s - loss: 0.4252 - accuracy: 0.9336
 9300/25000 [==========>...................] - ETA: 9s - loss: 0.4252 - accuracy: 0.9335
 9400/25000 [==========>...................] - ETA: 9s - loss: 0.4255 - accuracy: 0.9330
 9500/25000 [==========>...................] - ETA: 9s - loss: 0.4254 - accuracy: 0.9332
 9600/25000 [==========>...................] - ETA: 9s - loss: 0.4256 - accuracy: 0.9328
 9700/25000 [==========>...................] - ETA: 9s - loss: 0.4253 - accuracy: 0.9332
 9800/25000 [==========>...................] - ETA: 9s - loss: 0.4254 - accuracy: 0.9330
 9900/25000 [==========>...................] - ETA: 9s - loss: 0.4256 - accuracy: 0.9325
10000/25000 [===========>..................] - ETA: 9s - loss: 0.4257 - accuracy: 0.9323
10100/25000 [===========>..................] - ETA: 9s - loss: 0.4257 - accuracy: 0.9323
10200/25000 [===========>..................] - ETA: 9s - loss: 0.4254 - accuracy: 0.9325
10300/25000 [===========>..................] - ETA: 9s - loss: 0.4251 - accuracy: 0.9328
10400/25000 [===========>..................] - ETA: 9s - loss: 0.4250 - accuracy: 0.9330
10500/25000 [===========>..................] - ETA: 9s - loss: 0.4249 - accuracy: 0.9330
10600/25000 [===========>..................] - ETA: 9s - loss: 0.4256 - accuracy: 0.9323
10700/25000 [===========>..................] - ETA: 8s - loss: 0.4256 - accuracy: 0.9321
10800/25000 [===========>..................] - ETA: 8s - loss: 0.4255 - accuracy: 0.9321
10900/25000 [============>.................] - ETA: 8s - loss: 0.4256 - accuracy: 0.9320
11000/25000 [============>.................] - ETA: 8s - loss: 0.4255 - accuracy: 0.9320
11100/25000 [============>.................] - ETA: 8s - loss: 0.4254 - accuracy: 0.9323
11200/25000 [============>.................] - ETA: 8s - loss: 0.4254 - accuracy: 0.9321
11300/25000 [============>.................] - ETA: 8s - loss: 0.4253 - accuracy: 0.9322
11400/25000 [============>.................] - ETA: 8s - loss: 0.4253 - accuracy: 0.9321
11500/25000 [============>.................] - ETA: 8s - loss: 0.4254 - accuracy: 0.9319
11600/25000 [============>.................] - ETA: 8s - loss: 0.4254 - accuracy: 0.9316
11700/25000 [=============>................] - ETA: 8s - loss: 0.4255 - accuracy: 0.9315
11800/25000 [=============>................] - ETA: 8s - loss: 0.4254 - accuracy: 0.9316
11900/25000 [=============>................] - ETA: 8s - loss: 0.4255 - accuracy: 0.9314
12000/25000 [=============>................] - ETA: 8s - loss: 0.4254 - accuracy: 0.9315
12100/25000 [=============>................] - ETA: 8s - loss: 0.4254 - accuracy: 0.9316
12200/25000 [=============>................] - ETA: 8s - loss: 0.4253 - accuracy: 0.9316
12300/25000 [=============>................] - ETA: 7s - loss: 0.4251 - accuracy: 0.9319
12400/25000 [=============>................] - ETA: 7s - loss: 0.4249 - accuracy: 0.9321
12500/25000 [==============>...............] - ETA: 7s - loss: 0.4249 - accuracy: 0.9322
12600/25000 [==============>...............] - ETA: 7s - loss: 0.4249 - accuracy: 0.9321
12700/25000 [==============>...............] - ETA: 7s - loss: 0.4249 - accuracy: 0.9320
12800/25000 [==============>...............] - ETA: 7s - loss: 0.4248 - accuracy: 0.9321
12900/25000 [==============>...............] - ETA: 7s - loss: 0.4248 - accuracy: 0.9319
13000/25000 [==============>...............] - ETA: 7s - loss: 0.4245 - accuracy: 0.9322
13100/25000 [==============>...............] - ETA: 7s - loss: 0.4243 - accuracy: 0.9324
13200/25000 [==============>...............] - ETA: 7s - loss: 0.4241 - accuracy: 0.9326
13300/25000 [==============>...............] - ETA: 7s - loss: 0.4243 - accuracy: 0.9321
13400/25000 [===============>..............] - ETA: 7s - loss: 0.4245 - accuracy: 0.9318
13500/25000 [===============>..............] - ETA: 7s - loss: 0.4245 - accuracy: 0.9317
13600/25000 [===============>..............] - ETA: 7s - loss: 0.4243 - accuracy: 0.9318
13700/25000 [===============>..............] - ETA: 7s - loss: 0.4243 - accuracy: 0.9318
13800/25000 [===============>..............] - ETA: 6s - loss: 0.4243 - accuracy: 0.9317
13900/25000 [===============>..............] - ETA: 6s - loss: 0.4244 - accuracy: 0.9317
14000/25000 [===============>..............] - ETA: 6s - loss: 0.4244 - accuracy: 0.9317
14100/25000 [===============>..............] - ETA: 6s - loss: 0.4241 - accuracy: 0.9319
14200/25000 [================>.............] - ETA: 6s - loss: 0.4241 - accuracy: 0.9319
14300/25000 [================>.............] - ETA: 6s - loss: 0.4241 - accuracy: 0.9318
14400/25000 [================>.............] - ETA: 6s - loss: 0.4240 - accuracy: 0.9319
14500/25000 [================>.............] - ETA: 6s - loss: 0.4239 - accuracy: 0.9319
14600/25000 [================>.............] - ETA: 6s - loss: 0.4239 - accuracy: 0.9318
14700/25000 [================>.............] - ETA: 6s - loss: 0.4241 - accuracy: 0.9316
14800/25000 [================>.............] - ETA: 6s - loss: 0.4241 - accuracy: 0.9316
14900/25000 [================>.............] - ETA: 6s - loss: 0.4243 - accuracy: 0.9311
15000/25000 [=================>............] - ETA: 6s - loss: 0.4244 - accuracy: 0.9308
15100/25000 [=================>............] - ETA: 6s - loss: 0.4242 - accuracy: 0.9310
15200/25000 [=================>............] - ETA: 6s - loss: 0.4244 - accuracy: 0.9307
15300/25000 [=================>............] - ETA: 6s - loss: 0.4243 - accuracy: 0.9307
15400/25000 [=================>............] - ETA: 5s - loss: 0.4242 - accuracy: 0.9308
15500/25000 [=================>............] - ETA: 5s - loss: 0.4242 - accuracy: 0.9307
15600/25000 [=================>............] - ETA: 5s - loss: 0.4241 - accuracy: 0.9307
15700/25000 [=================>............] - ETA: 5s - loss: 0.4240 - accuracy: 0.9308
15800/25000 [=================>............] - ETA: 5s - loss: 0.4242 - accuracy: 0.9306
15900/25000 [==================>...........] - ETA: 5s - loss: 0.4243 - accuracy: 0.9304
16000/25000 [==================>...........] - ETA: 5s - loss: 0.4244 - accuracy: 0.9301
16100/25000 [==================>...........] - ETA: 5s - loss: 0.4244 - accuracy: 0.9301
16200/25000 [==================>...........] - ETA: 5s - loss: 0.4244 - accuracy: 0.9299
16300/25000 [==================>...........] - ETA: 5s - loss: 0.4242 - accuracy: 0.9301
16400/25000 [==================>...........] - ETA: 5s - loss: 0.4242 - accuracy: 0.9301
16500/25000 [==================>...........] - ETA: 5s - loss: 0.4240 - accuracy: 0.9302
16600/25000 [==================>...........] - ETA: 5s - loss: 0.4239 - accuracy: 0.9302
16700/25000 [===================>..........] - ETA: 5s - loss: 0.4240 - accuracy: 0.9301
16800/25000 [===================>..........] - ETA: 5s - loss: 0.4239 - accuracy: 0.9301
16900/25000 [===================>..........] - ETA: 5s - loss: 0.4238 - accuracy: 0.9302
17000/25000 [===================>..........] - ETA: 5s - loss: 0.4238 - accuracy: 0.9301
17100/25000 [===================>..........] - ETA: 4s - loss: 0.4239 - accuracy: 0.9299
17200/25000 [===================>..........] - ETA: 4s - loss: 0.4239 - accuracy: 0.9298
17300/25000 [===================>..........] - ETA: 4s - loss: 0.4238 - accuracy: 0.9299
17400/25000 [===================>..........] - ETA: 4s - loss: 0.4239 - accuracy: 0.9297
17500/25000 [====================>.........] - ETA: 4s - loss: 0.4238 - accuracy: 0.9298
17600/25000 [====================>.........] - ETA: 4s - loss: 0.4237 - accuracy: 0.9298
17700/25000 [====================>.........] - ETA: 4s - loss: 0.4237 - accuracy: 0.9297
17800/25000 [====================>.........] - ETA: 4s - loss: 0.4237 - accuracy: 0.9295
17900/25000 [====================>.........] - ETA: 4s - loss: 0.4237 - accuracy: 0.9296
18000/25000 [====================>.........] - ETA: 4s - loss: 0.4236 - accuracy: 0.9296
18100/25000 [====================>.........] - ETA: 4s - loss: 0.4236 - accuracy: 0.9296
18200/25000 [====================>.........] - ETA: 4s - loss: 0.4234 - accuracy: 0.9297
18300/25000 [====================>.........] - ETA: 4s - loss: 0.4232 - accuracy: 0.9299
18400/25000 [=====================>........] - ETA: 4s - loss: 0.4230 - accuracy: 0.9300
18500/25000 [=====================>........] - ETA: 4s - loss: 0.4230 - accuracy: 0.9300
18600/25000 [=====================>........] - ETA: 4s - loss: 0.4230 - accuracy: 0.9299
18700/25000 [=====================>........] - ETA: 3s - loss: 0.4229 - accuracy: 0.9299
18800/25000 [=====================>........] - ETA: 3s - loss: 0.4230 - accuracy: 0.9298
18900/25000 [=====================>........] - ETA: 3s - loss: 0.4231 - accuracy: 0.9296
19000/25000 [=====================>........] - ETA: 3s - loss: 0.4229 - accuracy: 0.9298
19100/25000 [=====================>........] - ETA: 3s - loss: 0.4228 - accuracy: 0.9298
19200/25000 [======================>.......] - ETA: 3s - loss: 0.4227 - accuracy: 0.9298
19300/25000 [======================>.......] - ETA: 3s - loss: 0.4229 - accuracy: 0.9297
19400/25000 [======================>.......] - ETA: 3s - loss: 0.4229 - accuracy: 0.9295
19500/25000 [======================>.......] - ETA: 3s - loss: 0.4228 - accuracy: 0.9296
19600/25000 [======================>.......] - ETA: 3s - loss: 0.4227 - accuracy: 0.9297
19700/25000 [======================>.......] - ETA: 3s - loss: 0.4226 - accuracy: 0.9297
19800/25000 [======================>.......] - ETA: 3s - loss: 0.4225 - accuracy: 0.9297
19900/25000 [======================>.......] - ETA: 3s - loss: 0.4226 - accuracy: 0.9296
20000/25000 [=======================>......] - ETA: 3s - loss: 0.4224 - accuracy: 0.9298
20100/25000 [=======================>......] - ETA: 3s - loss: 0.4226 - accuracy: 0.9295
20200/25000 [=======================>......] - ETA: 3s - loss: 0.4226 - accuracy: 0.9294
20300/25000 [=======================>......] - ETA: 2s - loss: 0.4225 - accuracy: 0.9294
20400/25000 [=======================>......] - ETA: 2s - loss: 0.4225 - accuracy: 0.9293
20500/25000 [=======================>......] - ETA: 2s - loss: 0.4224 - accuracy: 0.9294
20600/25000 [=======================>......] - ETA: 2s - loss: 0.4223 - accuracy: 0.9294
20700/25000 [=======================>......] - ETA: 2s - loss: 0.4222 - accuracy: 0.9294
20800/25000 [=======================>......] - ETA: 2s - loss: 0.4222 - accuracy: 0.9294
20900/25000 [========================>.....] - ETA: 2s - loss: 0.4223 - accuracy: 0.9291
21000/25000 [========================>.....] - ETA: 2s - loss: 0.4222 - accuracy: 0.9292
21100/25000 [========================>.....] - ETA: 2s - loss: 0.4222 - accuracy: 0.9291
21200/25000 [========================>.....] - ETA: 2s - loss: 0.4222 - accuracy: 0.9290
21300/25000 [========================>.....] - ETA: 2s - loss: 0.4222 - accuracy: 0.9289
21400/25000 [========================>.....] - ETA: 2s - loss: 0.4222 - accuracy: 0.9289
21500/25000 [========================>.....] - ETA: 2s - loss: 0.4221 - accuracy: 0.9290
21600/25000 [========================>.....] - ETA: 2s - loss: 0.4220 - accuracy: 0.9291
21700/25000 [=========================>....] - ETA: 2s - loss: 0.4218 - accuracy: 0.9293
21800/25000 [=========================>....] - ETA: 2s - loss: 0.4218 - accuracy: 0.9293
21900/25000 [=========================>....] - ETA: 1s - loss: 0.4217 - accuracy: 0.9293
22000/25000 [=========================>....] - ETA: 1s - loss: 0.4216 - accuracy: 0.9294
22100/25000 [=========================>....] - ETA: 1s - loss: 0.4217 - accuracy: 0.9291
22200/25000 [=========================>....] - ETA: 1s - loss: 0.4218 - accuracy: 0.9289
22300/25000 [=========================>....] - ETA: 1s - loss: 0.4217 - accuracy: 0.9290
22400/25000 [=========================>....] - ETA: 1s - loss: 0.4216 - accuracy: 0.9291
22500/25000 [==========================>...] - ETA: 1s - loss: 0.4216 - accuracy: 0.9290
22600/25000 [==========================>...] - ETA: 1s - loss: 0.4215 - accuracy: 0.9291
22700/25000 [==========================>...] - ETA: 1s - loss: 0.4215 - accuracy: 0.9290
22800/25000 [==========================>...] - ETA: 1s - loss: 0.4214 - accuracy: 0.9290
22900/25000 [==========================>...] - ETA: 1s - loss: 0.4213 - accuracy: 0.9290
23000/25000 [==========================>...] - ETA: 1s - loss: 0.4212 - accuracy: 0.9291
23100/25000 [==========================>...] - ETA: 1s - loss: 0.4211 - accuracy: 0.9291
23200/25000 [==========================>...] - ETA: 1s - loss: 0.4210 - accuracy: 0.9291
23300/25000 [==========================>...] - ETA: 1s - loss: 0.4210 - accuracy: 0.9291
23400/25000 [===========================>..] - ETA: 1s - loss: 0.4210 - accuracy: 0.9291
23500/25000 [===========================>..] - ETA: 0s - loss: 0.4209 - accuracy: 0.9291
23600/25000 [===========================>..] - ETA: 0s - loss: 0.4207 - accuracy: 0.9293
23700/25000 [===========================>..] - ETA: 0s - loss: 0.4205 - accuracy: 0.9294
23800/25000 [===========================>..] - ETA: 0s - loss: 0.4204 - accuracy: 0.9295
23900/25000 [===========================>..] - ETA: 0s - loss: 0.4203 - accuracy: 0.9294
24000/25000 [===========================>..] - ETA: 0s - loss: 0.4203 - accuracy: 0.9293
24100/25000 [===========================>..] - ETA: 0s - loss: 0.4203 - accuracy: 0.9293
24200/25000 [============================>.] - ETA: 0s - loss: 0.4203 - accuracy: 0.9293
24300/25000 [============================>.] - ETA: 0s - loss: 0.4201 - accuracy: 0.9294
24400/25000 [============================>.] - ETA: 0s - loss: 0.4201 - accuracy: 0.9293
24500/25000 [============================>.] - ETA: 0s - loss: 0.4201 - accuracy: 0.9292
24600/25000 [============================>.] - ETA: 0s - loss: 0.4202 - accuracy: 0.9290
24700/25000 [============================>.] - ETA: 0s - loss: 0.4203 - accuracy: 0.9288
24800/25000 [============================>.] - ETA: 0s - loss: 0.4204 - accuracy: 0.9287
24900/25000 [============================>.] - ETA: 0s - loss: 0.4203 - accuracy: 0.9288
25000/25000 [==============================] - 20s 794us/step - loss: 0.4202 - accuracy: 0.9288 - val_loss: 0.4673 - val_accuracy: 0.8596
Epoch 6/10

  100/25000 [..............................] - ETA: 15s - loss: 0.4239 - accuracy: 0.9100
  200/25000 [..............................] - ETA: 15s - loss: 0.3949 - accuracy: 0.9450
  300/25000 [..............................] - ETA: 15s - loss: 0.3969 - accuracy: 0.9400
  400/25000 [..............................] - ETA: 15s - loss: 0.3979 - accuracy: 0.9375
  500/25000 [..............................] - ETA: 15s - loss: 0.4021 - accuracy: 0.9320
  600/25000 [..............................] - ETA: 15s - loss: 0.3994 - accuracy: 0.9333
  700/25000 [..............................] - ETA: 15s - loss: 0.4020 - accuracy: 0.9286
  800/25000 [..............................] - ETA: 15s - loss: 0.4044 - accuracy: 0.9250
  900/25000 [>.............................] - ETA: 15s - loss: 0.4093 - accuracy: 0.9211
 1000/25000 [>.............................] - ETA: 15s - loss: 0.4072 - accuracy: 0.9220
 1100/25000 [>.............................] - ETA: 14s - loss: 0.4076 - accuracy: 0.9227
 1200/25000 [>.............................] - ETA: 14s - loss: 0.4060 - accuracy: 0.9250
 1300/25000 [>.............................] - ETA: 14s - loss: 0.4056 - accuracy: 0.9254
 1400/25000 [>.............................] - ETA: 14s - loss: 0.4062 - accuracy: 0.9250
 1500/25000 [>.............................] - ETA: 14s - loss: 0.4080 - accuracy: 0.9227
 1600/25000 [>.............................] - ETA: 14s - loss: 0.4075 - accuracy: 0.9231
 1700/25000 [=>............................] - ETA: 14s - loss: 0.4087 - accuracy: 0.9218
 1800/25000 [=>............................] - ETA: 14s - loss: 0.4072 - accuracy: 0.9233
 1900/25000 [=>............................] - ETA: 14s - loss: 0.4059 - accuracy: 0.9253
 2000/25000 [=>............................] - ETA: 14s - loss: 0.4046 - accuracy: 0.9270
 2100/25000 [=>............................] - ETA: 14s - loss: 0.4048 - accuracy: 0.9271
 2200/25000 [=>............................] - ETA: 14s - loss: 0.4041 - accuracy: 0.9277
 2300/25000 [=>............................] - ETA: 14s - loss: 0.4041 - accuracy: 0.9274
 2400/25000 [=>............................] - ETA: 14s - loss: 0.4025 - accuracy: 0.9292
 2500/25000 [==>...........................] - ETA: 14s - loss: 0.4030 - accuracy: 0.9292
 2600/25000 [==>...........................] - ETA: 14s - loss: 0.4028 - accuracy: 0.9296
 2700/25000 [==>...........................] - ETA: 13s - loss: 0.4029 - accuracy: 0.9296
 2800/25000 [==>...........................] - ETA: 13s - loss: 0.4030 - accuracy: 0.9296
 2900/25000 [==>...........................] - ETA: 13s - loss: 0.4020 - accuracy: 0.9307
 3000/25000 [==>...........................] - ETA: 13s - loss: 0.4024 - accuracy: 0.9303
 3100/25000 [==>...........................] - ETA: 13s - loss: 0.4016 - accuracy: 0.9310
 3200/25000 [==>...........................] - ETA: 13s - loss: 0.4017 - accuracy: 0.9309
 3300/25000 [==>...........................] - ETA: 13s - loss: 0.4008 - accuracy: 0.9321
 3400/25000 [===>..........................] - ETA: 13s - loss: 0.3999 - accuracy: 0.9329
 3500/25000 [===>..........................] - ETA: 13s - loss: 0.3993 - accuracy: 0.9334
 3600/25000 [===>..........................] - ETA: 13s - loss: 0.3995 - accuracy: 0.9331
 3700/25000 [===>..........................] - ETA: 13s - loss: 0.3997 - accuracy: 0.9327
 3800/25000 [===>..........................] - ETA: 13s - loss: 0.3993 - accuracy: 0.9332
 3900/25000 [===>..........................] - ETA: 13s - loss: 0.3986 - accuracy: 0.9338
 4000/25000 [===>..........................] - ETA: 13s - loss: 0.3978 - accuracy: 0.9348
 4100/25000 [===>..........................] - ETA: 13s - loss: 0.3980 - accuracy: 0.9344
 4200/25000 [====>.........................] - ETA: 13s - loss: 0.3980 - accuracy: 0.9348
 4300/25000 [====>.........................] - ETA: 12s - loss: 0.3980 - accuracy: 0.9349
 4400/25000 [====>.........................] - ETA: 12s - loss: 0.3979 - accuracy: 0.9350
 4500/25000 [====>.........................] - ETA: 12s - loss: 0.3979 - accuracy: 0.9344
 4600/25000 [====>.........................] - ETA: 12s - loss: 0.3976 - accuracy: 0.9348
 4700/25000 [====>.........................] - ETA: 12s - loss: 0.3977 - accuracy: 0.9347
 4800/25000 [====>.........................] - ETA: 12s - loss: 0.3974 - accuracy: 0.9350
 4900/25000 [====>.........................] - ETA: 12s - loss: 0.3973 - accuracy: 0.9351
 5000/25000 [=====>........................] - ETA: 12s - loss: 0.3967 - accuracy: 0.9358
 5100/25000 [=====>........................] - ETA: 12s - loss: 0.3964 - accuracy: 0.9363
 5200/25000 [=====>........................] - ETA: 12s - loss: 0.3963 - accuracy: 0.9365
 5300/25000 [=====>........................] - ETA: 12s - loss: 0.3963 - accuracy: 0.9364
 5400/25000 [=====>........................] - ETA: 12s - loss: 0.3958 - accuracy: 0.9370
 5500/25000 [=====>........................] - ETA: 12s - loss: 0.3959 - accuracy: 0.9367
 5600/25000 [=====>........................] - ETA: 12s - loss: 0.3966 - accuracy: 0.9357
 5700/25000 [=====>........................] - ETA: 12s - loss: 0.3967 - accuracy: 0.9356
 5800/25000 [=====>........................] - ETA: 11s - loss: 0.3963 - accuracy: 0.9360
 5900/25000 [======>.......................] - ETA: 11s - loss: 0.3965 - accuracy: 0.9356
 6000/25000 [======>.......................] - ETA: 11s - loss: 0.3965 - accuracy: 0.9357
 6100/25000 [======>.......................] - ETA: 11s - loss: 0.3967 - accuracy: 0.9356
 6200/25000 [======>.......................] - ETA: 11s - loss: 0.3961 - accuracy: 0.9361
 6300/25000 [======>.......................] - ETA: 11s - loss: 0.3958 - accuracy: 0.9363
 6400/25000 [======>.......................] - ETA: 11s - loss: 0.3953 - accuracy: 0.9369
 6500/25000 [======>.......................] - ETA: 11s - loss: 0.3954 - accuracy: 0.9368
 6600/25000 [======>.......................] - ETA: 11s - loss: 0.3955 - accuracy: 0.9365
 6700/25000 [=======>......................] - ETA: 11s - loss: 0.3955 - accuracy: 0.9364
 6800/25000 [=======>......................] - ETA: 11s - loss: 0.3950 - accuracy: 0.9368
 6900/25000 [=======>......................] - ETA: 11s - loss: 0.3962 - accuracy: 0.9354
 7000/25000 [=======>......................] - ETA: 11s - loss: 0.3960 - accuracy: 0.9353
 7100/25000 [=======>......................] - ETA: 11s - loss: 0.3958 - accuracy: 0.9354
 7200/25000 [=======>......................] - ETA: 11s - loss: 0.3954 - accuracy: 0.9358
 7300/25000 [=======>......................] - ETA: 11s - loss: 0.3953 - accuracy: 0.9359
 7400/25000 [=======>......................] - ETA: 10s - loss: 0.3951 - accuracy: 0.9359
 7500/25000 [========>.....................] - ETA: 10s - loss: 0.3949 - accuracy: 0.9363
 7600/25000 [========>.....................] - ETA: 10s - loss: 0.3949 - accuracy: 0.9363
 7700/25000 [========>.....................] - ETA: 10s - loss: 0.3945 - accuracy: 0.9366
 7800/25000 [========>.....................] - ETA: 10s - loss: 0.3943 - accuracy: 0.9365
 7900/25000 [========>.....................] - ETA: 10s - loss: 0.3941 - accuracy: 0.9367
 8000/25000 [========>.....................] - ETA: 10s - loss: 0.3943 - accuracy: 0.9365
 8100/25000 [========>.....................] - ETA: 10s - loss: 0.3942 - accuracy: 0.9365
 8200/25000 [========>.....................] - ETA: 10s - loss: 0.3943 - accuracy: 0.9363
 8300/25000 [========>.....................] - ETA: 10s - loss: 0.3941 - accuracy: 0.9365
 8400/25000 [=========>....................] - ETA: 10s - loss: 0.3943 - accuracy: 0.9362
 8500/25000 [=========>....................] - ETA: 10s - loss: 0.3940 - accuracy: 0.9365
 8600/25000 [=========>....................] - ETA: 10s - loss: 0.3938 - accuracy: 0.9366
 8700/25000 [=========>....................] - ETA: 10s - loss: 0.3940 - accuracy: 0.9362
 8800/25000 [=========>....................] - ETA: 10s - loss: 0.3938 - accuracy: 0.9365
 8900/25000 [=========>....................] - ETA: 10s - loss: 0.3940 - accuracy: 0.9363
 9000/25000 [=========>....................] - ETA: 10s - loss: 0.3938 - accuracy: 0.9366
 9100/25000 [=========>....................] - ETA: 9s - loss: 0.3938 - accuracy: 0.9364 
 9200/25000 [==========>...................] - ETA: 9s - loss: 0.3938 - accuracy: 0.9363
 9300/25000 [==========>...................] - ETA: 9s - loss: 0.3939 - accuracy: 0.9361
 9400/25000 [==========>...................] - ETA: 9s - loss: 0.3941 - accuracy: 0.9360
 9500/25000 [==========>...................] - ETA: 9s - loss: 0.3938 - accuracy: 0.9361
 9600/25000 [==========>...................] - ETA: 9s - loss: 0.3938 - accuracy: 0.9360
 9700/25000 [==========>...................] - ETA: 9s - loss: 0.3938 - accuracy: 0.9361
 9800/25000 [==========>...................] - ETA: 9s - loss: 0.3936 - accuracy: 0.9362
 9900/25000 [==========>...................] - ETA: 9s - loss: 0.3936 - accuracy: 0.9362
10000/25000 [===========>..................] - ETA: 9s - loss: 0.3937 - accuracy: 0.9360
10100/25000 [===========>..................] - ETA: 9s - loss: 0.3937 - accuracy: 0.9360
10200/25000 [===========>..................] - ETA: 9s - loss: 0.3934 - accuracy: 0.9364
10300/25000 [===========>..................] - ETA: 9s - loss: 0.3936 - accuracy: 0.9360
10400/25000 [===========>..................] - ETA: 9s - loss: 0.3938 - accuracy: 0.9358
10500/25000 [===========>..................] - ETA: 9s - loss: 0.3937 - accuracy: 0.9358
10600/25000 [===========>..................] - ETA: 9s - loss: 0.3938 - accuracy: 0.9357
10700/25000 [===========>..................] - ETA: 8s - loss: 0.3937 - accuracy: 0.9356
10800/25000 [===========>..................] - ETA: 8s - loss: 0.3935 - accuracy: 0.9357
10900/25000 [============>.................] - ETA: 8s - loss: 0.3934 - accuracy: 0.9358
11000/25000 [============>.................] - ETA: 8s - loss: 0.3936 - accuracy: 0.9355
11100/25000 [============>.................] - ETA: 8s - loss: 0.3937 - accuracy: 0.9354
11200/25000 [============>.................] - ETA: 8s - loss: 0.3937 - accuracy: 0.9353
11300/25000 [============>.................] - ETA: 8s - loss: 0.3937 - accuracy: 0.9352
11400/25000 [============>.................] - ETA: 8s - loss: 0.3935 - accuracy: 0.9355
11500/25000 [============>.................] - ETA: 8s - loss: 0.3932 - accuracy: 0.9357
11600/25000 [============>.................] - ETA: 8s - loss: 0.3934 - accuracy: 0.9356
11700/25000 [=============>................] - ETA: 8s - loss: 0.3936 - accuracy: 0.9352
11800/25000 [=============>................] - ETA: 8s - loss: 0.3935 - accuracy: 0.9352
11900/25000 [=============>................] - ETA: 8s - loss: 0.3934 - accuracy: 0.9352
12000/25000 [=============>................] - ETA: 8s - loss: 0.3935 - accuracy: 0.9352
12100/25000 [=============>................] - ETA: 8s - loss: 0.3930 - accuracy: 0.9356
12200/25000 [=============>................] - ETA: 8s - loss: 0.3929 - accuracy: 0.9357
12300/25000 [=============>................] - ETA: 7s - loss: 0.3926 - accuracy: 0.9360
12400/25000 [=============>................] - ETA: 7s - loss: 0.3926 - accuracy: 0.9360
12500/25000 [==============>...............] - ETA: 7s - loss: 0.3926 - accuracy: 0.9359
12600/25000 [==============>...............] - ETA: 7s - loss: 0.3927 - accuracy: 0.9357
12700/25000 [==============>...............] - ETA: 7s - loss: 0.3924 - accuracy: 0.9360
12800/25000 [==============>...............] - ETA: 7s - loss: 0.3923 - accuracy: 0.9361
12900/25000 [==============>...............] - ETA: 7s - loss: 0.3922 - accuracy: 0.9361
13000/25000 [==============>...............] - ETA: 7s - loss: 0.3924 - accuracy: 0.9358
13100/25000 [==============>...............] - ETA: 7s - loss: 0.3926 - accuracy: 0.9356
13200/25000 [==============>...............] - ETA: 7s - loss: 0.3927 - accuracy: 0.9355
13300/25000 [==============>...............] - ETA: 7s - loss: 0.3928 - accuracy: 0.9353
13400/25000 [===============>..............] - ETA: 7s - loss: 0.3927 - accuracy: 0.9354
13500/25000 [===============>..............] - ETA: 7s - loss: 0.3924 - accuracy: 0.9356
13600/25000 [===============>..............] - ETA: 7s - loss: 0.3923 - accuracy: 0.9355
13700/25000 [===============>..............] - ETA: 7s - loss: 0.3923 - accuracy: 0.9356
13800/25000 [===============>..............] - ETA: 7s - loss: 0.3923 - accuracy: 0.9356
13900/25000 [===============>..............] - ETA: 6s - loss: 0.3920 - accuracy: 0.9358
14000/25000 [===============>..............] - ETA: 6s - loss: 0.3919 - accuracy: 0.9359
14100/25000 [===============>..............] - ETA: 6s - loss: 0.3919 - accuracy: 0.9357
14200/25000 [================>.............] - ETA: 6s - loss: 0.3918 - accuracy: 0.9357
14300/25000 [================>.............] - ETA: 6s - loss: 0.3918 - accuracy: 0.9357
14400/25000 [================>.............] - ETA: 6s - loss: 0.3919 - accuracy: 0.9356
14500/25000 [================>.............] - ETA: 6s - loss: 0.3919 - accuracy: 0.9356
14600/25000 [================>.............] - ETA: 6s - loss: 0.3917 - accuracy: 0.9357
14700/25000 [================>.............] - ETA: 6s - loss: 0.3916 - accuracy: 0.9358
14800/25000 [================>.............] - ETA: 6s - loss: 0.3916 - accuracy: 0.9358
14900/25000 [================>.............] - ETA: 6s - loss: 0.3917 - accuracy: 0.9356
15000/25000 [=================>............] - ETA: 6s - loss: 0.3915 - accuracy: 0.9358
15100/25000 [=================>............] - ETA: 6s - loss: 0.3914 - accuracy: 0.9359
15200/25000 [=================>............] - ETA: 6s - loss: 0.3913 - accuracy: 0.9360
15300/25000 [=================>............] - ETA: 6s - loss: 0.3911 - accuracy: 0.9361
15400/25000 [=================>............] - ETA: 6s - loss: 0.3911 - accuracy: 0.9362
15500/25000 [=================>............] - ETA: 5s - loss: 0.3911 - accuracy: 0.9360
15600/25000 [=================>............] - ETA: 5s - loss: 0.3910 - accuracy: 0.9361
15700/25000 [=================>............] - ETA: 5s - loss: 0.3908 - accuracy: 0.9363
15800/25000 [=================>............] - ETA: 5s - loss: 0.3907 - accuracy: 0.9363
15900/25000 [==================>...........] - ETA: 5s - loss: 0.3907 - accuracy: 0.9363
16000/25000 [==================>...........] - ETA: 5s - loss: 0.3907 - accuracy: 0.9362
16100/25000 [==================>...........] - ETA: 5s - loss: 0.3907 - accuracy: 0.9362
16200/25000 [==================>...........] - ETA: 5s - loss: 0.3905 - accuracy: 0.9364
16300/25000 [==================>...........] - ETA: 5s - loss: 0.3905 - accuracy: 0.9364
16400/25000 [==================>...........] - ETA: 5s - loss: 0.3905 - accuracy: 0.9362
16500/25000 [==================>...........] - ETA: 5s - loss: 0.3904 - accuracy: 0.9363
16600/25000 [==================>...........] - ETA: 5s - loss: 0.3903 - accuracy: 0.9363
16700/25000 [===================>..........] - ETA: 5s - loss: 0.3903 - accuracy: 0.9363
16800/25000 [===================>..........] - ETA: 5s - loss: 0.3901 - accuracy: 0.9364
16900/25000 [===================>..........] - ETA: 5s - loss: 0.3901 - accuracy: 0.9364
17000/25000 [===================>..........] - ETA: 5s - loss: 0.3899 - accuracy: 0.9365
17100/25000 [===================>..........] - ETA: 4s - loss: 0.3899 - accuracy: 0.9365
17200/25000 [===================>..........] - ETA: 4s - loss: 0.3898 - accuracy: 0.9366
17300/25000 [===================>..........] - ETA: 4s - loss: 0.3899 - accuracy: 0.9366
17400/25000 [===================>..........] - ETA: 4s - loss: 0.3898 - accuracy: 0.9366
17500/25000 [====================>.........] - ETA: 4s - loss: 0.3895 - accuracy: 0.9367
17600/25000 [====================>.........] - ETA: 4s - loss: 0.3893 - accuracy: 0.9369
17700/25000 [====================>.........] - ETA: 4s - loss: 0.3892 - accuracy: 0.9369
17800/25000 [====================>.........] - ETA: 4s - loss: 0.3892 - accuracy: 0.9369
17900/25000 [====================>.........] - ETA: 4s - loss: 0.3891 - accuracy: 0.9370
18000/25000 [====================>.........] - ETA: 4s - loss: 0.3888 - accuracy: 0.9372
18100/25000 [====================>.........] - ETA: 4s - loss: 0.3887 - accuracy: 0.9373
18200/25000 [====================>.........] - ETA: 4s - loss: 0.3886 - accuracy: 0.9373
18300/25000 [====================>.........] - ETA: 4s - loss: 0.3886 - accuracy: 0.9372
18400/25000 [=====================>........] - ETA: 4s - loss: 0.3885 - accuracy: 0.9373
18500/25000 [=====================>........] - ETA: 4s - loss: 0.3887 - accuracy: 0.9370
18600/25000 [=====================>........] - ETA: 4s - loss: 0.3885 - accuracy: 0.9372
18700/25000 [=====================>........] - ETA: 3s - loss: 0.3886 - accuracy: 0.9371
18800/25000 [=====================>........] - ETA: 3s - loss: 0.3886 - accuracy: 0.9370
18900/25000 [=====================>........] - ETA: 3s - loss: 0.3886 - accuracy: 0.9370
19000/25000 [=====================>........] - ETA: 3s - loss: 0.3886 - accuracy: 0.9369
19100/25000 [=====================>........] - ETA: 3s - loss: 0.3885 - accuracy: 0.9371
19200/25000 [======================>.......] - ETA: 3s - loss: 0.3884 - accuracy: 0.9371
19300/25000 [======================>.......] - ETA: 3s - loss: 0.3885 - accuracy: 0.9369
19400/25000 [======================>.......] - ETA: 3s - loss: 0.3883 - accuracy: 0.9370
19500/25000 [======================>.......] - ETA: 3s - loss: 0.3882 - accuracy: 0.9370
19600/25000 [======================>.......] - ETA: 3s - loss: 0.3881 - accuracy: 0.9370
19700/25000 [======================>.......] - ETA: 3s - loss: 0.3882 - accuracy: 0.9370
19800/25000 [======================>.......] - ETA: 3s - loss: 0.3880 - accuracy: 0.9371
19900/25000 [======================>.......] - ETA: 3s - loss: 0.3879 - accuracy: 0.9371
20000/25000 [=======================>......] - ETA: 3s - loss: 0.3879 - accuracy: 0.9370
20100/25000 [=======================>......] - ETA: 3s - loss: 0.3878 - accuracy: 0.9371
20200/25000 [=======================>......] - ETA: 3s - loss: 0.3878 - accuracy: 0.9369
20300/25000 [=======================>......] - ETA: 2s - loss: 0.3877 - accuracy: 0.9369
20400/25000 [=======================>......] - ETA: 2s - loss: 0.3875 - accuracy: 0.9370
20500/25000 [=======================>......] - ETA: 2s - loss: 0.3875 - accuracy: 0.9370
20600/25000 [=======================>......] - ETA: 2s - loss: 0.3875 - accuracy: 0.9370
20700/25000 [=======================>......] - ETA: 2s - loss: 0.3874 - accuracy: 0.9370
20800/25000 [=======================>......] - ETA: 2s - loss: 0.3874 - accuracy: 0.9370
20900/25000 [========================>.....] - ETA: 2s - loss: 0.3875 - accuracy: 0.9367
21000/25000 [========================>.....] - ETA: 2s - loss: 0.3874 - accuracy: 0.9369
21100/25000 [========================>.....] - ETA: 2s - loss: 0.3874 - accuracy: 0.9367
21200/25000 [========================>.....] - ETA: 2s - loss: 0.3874 - accuracy: 0.9367
21300/25000 [========================>.....] - ETA: 2s - loss: 0.3873 - accuracy: 0.9368
21400/25000 [========================>.....] - ETA: 2s - loss: 0.3872 - accuracy: 0.9369
21500/25000 [========================>.....] - ETA: 2s - loss: 0.3871 - accuracy: 0.9370
21600/25000 [========================>.....] - ETA: 2s - loss: 0.3870 - accuracy: 0.9369
21700/25000 [=========================>....] - ETA: 2s - loss: 0.3868 - accuracy: 0.9371
21800/25000 [=========================>....] - ETA: 2s - loss: 0.3868 - accuracy: 0.9370
21900/25000 [=========================>....] - ETA: 1s - loss: 0.3867 - accuracy: 0.9370
22000/25000 [=========================>....] - ETA: 1s - loss: 0.3867 - accuracy: 0.9369
22100/25000 [=========================>....] - ETA: 1s - loss: 0.3865 - accuracy: 0.9371
22200/25000 [=========================>....] - ETA: 1s - loss: 0.3863 - accuracy: 0.9372
22300/25000 [=========================>....] - ETA: 1s - loss: 0.3863 - accuracy: 0.9372
22400/25000 [=========================>....] - ETA: 1s - loss: 0.3862 - accuracy: 0.9373
22500/25000 [==========================>...] - ETA: 1s - loss: 0.3862 - accuracy: 0.9372
22600/25000 [==========================>...] - ETA: 1s - loss: 0.3862 - accuracy: 0.9371
22700/25000 [==========================>...] - ETA: 1s - loss: 0.3861 - accuracy: 0.9373
22800/25000 [==========================>...] - ETA: 1s - loss: 0.3860 - accuracy: 0.9373
22900/25000 [==========================>...] - ETA: 1s - loss: 0.3858 - accuracy: 0.9374
23000/25000 [==========================>...] - ETA: 1s - loss: 0.3859 - accuracy: 0.9372
23100/25000 [==========================>...] - ETA: 1s - loss: 0.3859 - accuracy: 0.9373
23200/25000 [==========================>...] - ETA: 1s - loss: 0.3858 - accuracy: 0.9373
23300/25000 [==========================>...] - ETA: 1s - loss: 0.3858 - accuracy: 0.9373
23400/25000 [===========================>..] - ETA: 1s - loss: 0.3859 - accuracy: 0.9371
23500/25000 [===========================>..] - ETA: 0s - loss: 0.3859 - accuracy: 0.9371
23600/25000 [===========================>..] - ETA: 0s - loss: 0.3858 - accuracy: 0.9371
23700/25000 [===========================>..] - ETA: 0s - loss: 0.3857 - accuracy: 0.9372
23800/25000 [===========================>..] - ETA: 0s - loss: 0.3856 - accuracy: 0.9372
23900/25000 [===========================>..] - ETA: 0s - loss: 0.3856 - accuracy: 0.9372
24000/25000 [===========================>..] - ETA: 0s - loss: 0.3856 - accuracy: 0.9370
24100/25000 [===========================>..] - ETA: 0s - loss: 0.3855 - accuracy: 0.9371
24200/25000 [============================>.] - ETA: 0s - loss: 0.3854 - accuracy: 0.9372
24300/25000 [============================>.] - ETA: 0s - loss: 0.3854 - accuracy: 0.9372
24400/25000 [============================>.] - ETA: 0s - loss: 0.3852 - accuracy: 0.9373
24500/25000 [============================>.] - ETA: 0s - loss: 0.3851 - accuracy: 0.9373
24600/25000 [============================>.] - ETA: 0s - loss: 0.3851 - accuracy: 0.9373
24700/25000 [============================>.] - ETA: 0s - loss: 0.3850 - accuracy: 0.9373
24800/25000 [============================>.] - ETA: 0s - loss: 0.3851 - accuracy: 0.9373
24900/25000 [============================>.] - ETA: 0s - loss: 0.3851 - accuracy: 0.9372
25000/25000 [==============================] - 20s 792us/step - loss: 0.3852 - accuracy: 0.9370 - val_loss: 0.4491 - val_accuracy: 0.8598
Epoch 7/10

  100/25000 [..............................] - ETA: 15s - loss: 0.3566 - accuracy: 0.9500
  200/25000 [..............................] - ETA: 15s - loss: 0.3612 - accuracy: 0.9500
  300/25000 [..............................] - ETA: 15s - loss: 0.3585 - accuracy: 0.9500
  400/25000 [..............................] - ETA: 15s - loss: 0.3551 - accuracy: 0.9575
  500/25000 [..............................] - ETA: 15s - loss: 0.3542 - accuracy: 0.9580
  600/25000 [..............................] - ETA: 15s - loss: 0.3570 - accuracy: 0.9550
  700/25000 [..............................] - ETA: 15s - loss: 0.3554 - accuracy: 0.9557
  800/25000 [..............................] - ETA: 16s - loss: 0.3523 - accuracy: 0.9588
  900/25000 [>.............................] - ETA: 16s - loss: 0.3493 - accuracy: 0.9622
 1000/25000 [>.............................] - ETA: 16s - loss: 0.3543 - accuracy: 0.9580
 1100/25000 [>.............................] - ETA: 16s - loss: 0.3518 - accuracy: 0.9609
 1200/25000 [>.............................] - ETA: 16s - loss: 0.3537 - accuracy: 0.9583
 1300/25000 [>.............................] - ETA: 16s - loss: 0.3547 - accuracy: 0.9569
 1400/25000 [>.............................] - ETA: 16s - loss: 0.3541 - accuracy: 0.9579
 1500/25000 [>.............................] - ETA: 16s - loss: 0.3535 - accuracy: 0.9587
 1600/25000 [>.............................] - ETA: 15s - loss: 0.3522 - accuracy: 0.9600
 1700/25000 [=>............................] - ETA: 15s - loss: 0.3516 - accuracy: 0.9606
 1800/25000 [=>............................] - ETA: 15s - loss: 0.3504 - accuracy: 0.9622
 1900/25000 [=>............................] - ETA: 15s - loss: 0.3494 - accuracy: 0.9626
 2000/25000 [=>............................] - ETA: 15s - loss: 0.3532 - accuracy: 0.9595
 2100/25000 [=>............................] - ETA: 15s - loss: 0.3528 - accuracy: 0.9595
 2200/25000 [=>............................] - ETA: 15s - loss: 0.3515 - accuracy: 0.9605
 2300/25000 [=>............................] - ETA: 15s - loss: 0.3532 - accuracy: 0.9587
 2400/25000 [=>............................] - ETA: 15s - loss: 0.3530 - accuracy: 0.9592
 2500/25000 [==>...........................] - ETA: 14s - loss: 0.3543 - accuracy: 0.9576
 2600/25000 [==>...........................] - ETA: 14s - loss: 0.3550 - accuracy: 0.9569
 2700/25000 [==>...........................] - ETA: 14s - loss: 0.3539 - accuracy: 0.9578
 2800/25000 [==>...........................] - ETA: 14s - loss: 0.3544 - accuracy: 0.9571
 2900/25000 [==>...........................] - ETA: 14s - loss: 0.3541 - accuracy: 0.9576
 3000/25000 [==>...........................] - ETA: 14s - loss: 0.3534 - accuracy: 0.9583
 3100/25000 [==>...........................] - ETA: 14s - loss: 0.3534 - accuracy: 0.9584
 3200/25000 [==>...........................] - ETA: 14s - loss: 0.3524 - accuracy: 0.9594
 3300/25000 [==>...........................] - ETA: 14s - loss: 0.3521 - accuracy: 0.9594
 3400/25000 [===>..........................] - ETA: 14s - loss: 0.3533 - accuracy: 0.9579
 3500/25000 [===>..........................] - ETA: 14s - loss: 0.3534 - accuracy: 0.9577
 3600/25000 [===>..........................] - ETA: 13s - loss: 0.3538 - accuracy: 0.9572
 3700/25000 [===>..........................] - ETA: 13s - loss: 0.3537 - accuracy: 0.9570
 3800/25000 [===>..........................] - ETA: 13s - loss: 0.3534 - accuracy: 0.9574
 3900/25000 [===>..........................] - ETA: 13s - loss: 0.3535 - accuracy: 0.9572
 4000/25000 [===>..........................] - ETA: 13s - loss: 0.3541 - accuracy: 0.9563
 4100/25000 [===>..........................] - ETA: 13s - loss: 0.3545 - accuracy: 0.9556
 4200/25000 [====>.........................] - ETA: 13s - loss: 0.3546 - accuracy: 0.9555
 4300/25000 [====>.........................] - ETA: 13s - loss: 0.3546 - accuracy: 0.9551
 4400/25000 [====>.........................] - ETA: 13s - loss: 0.3546 - accuracy: 0.9552
 4500/25000 [====>.........................] - ETA: 13s - loss: 0.3549 - accuracy: 0.9549
 4600/25000 [====>.........................] - ETA: 13s - loss: 0.3556 - accuracy: 0.9541
 4700/25000 [====>.........................] - ETA: 13s - loss: 0.3556 - accuracy: 0.9540
 4800/25000 [====>.........................] - ETA: 13s - loss: 0.3556 - accuracy: 0.9542
 4900/25000 [====>.........................] - ETA: 12s - loss: 0.3560 - accuracy: 0.9535
 5000/25000 [=====>........................] - ETA: 12s - loss: 0.3562 - accuracy: 0.9532
 5100/25000 [=====>........................] - ETA: 12s - loss: 0.3557 - accuracy: 0.9535
 5200/25000 [=====>........................] - ETA: 12s - loss: 0.3553 - accuracy: 0.9540
 5300/25000 [=====>........................] - ETA: 12s - loss: 0.3559 - accuracy: 0.9532
 5400/25000 [=====>........................] - ETA: 12s - loss: 0.3573 - accuracy: 0.9517
 5500/25000 [=====>........................] - ETA: 12s - loss: 0.3570 - accuracy: 0.9520
 5600/25000 [=====>........................] - ETA: 12s - loss: 0.3569 - accuracy: 0.9520
 5700/25000 [=====>........................] - ETA: 12s - loss: 0.3566 - accuracy: 0.9523
 5800/25000 [=====>........................] - ETA: 12s - loss: 0.3562 - accuracy: 0.9526
 5900/25000 [======>.......................] - ETA: 12s - loss: 0.3560 - accuracy: 0.9525
 6000/25000 [======>.......................] - ETA: 12s - loss: 0.3567 - accuracy: 0.9518
 6100/25000 [======>.......................] - ETA: 12s - loss: 0.3569 - accuracy: 0.9516
 6200/25000 [======>.......................] - ETA: 12s - loss: 0.3571 - accuracy: 0.9515
 6300/25000 [======>.......................] - ETA: 11s - loss: 0.3570 - accuracy: 0.9514
 6400/25000 [======>.......................] - ETA: 11s - loss: 0.3570 - accuracy: 0.9513
 6500/25000 [======>.......................] - ETA: 11s - loss: 0.3571 - accuracy: 0.9512
 6600/25000 [======>.......................] - ETA: 11s - loss: 0.3573 - accuracy: 0.9511
 6700/25000 [=======>......................] - ETA: 11s - loss: 0.3572 - accuracy: 0.9510
 6800/25000 [=======>......................] - ETA: 11s - loss: 0.3575 - accuracy: 0.9507
 6900/25000 [=======>......................] - ETA: 11s - loss: 0.3577 - accuracy: 0.9504
 7000/25000 [=======>......................] - ETA: 11s - loss: 0.3579 - accuracy: 0.9503
 7100/25000 [=======>......................] - ETA: 11s - loss: 0.3579 - accuracy: 0.9503
 7200/25000 [=======>......................] - ETA: 11s - loss: 0.3581 - accuracy: 0.9500
 7300/25000 [=======>......................] - ETA: 11s - loss: 0.3585 - accuracy: 0.9496
 7400/25000 [=======>......................] - ETA: 11s - loss: 0.3586 - accuracy: 0.9495
 7500/25000 [========>.....................] - ETA: 11s - loss: 0.3587 - accuracy: 0.9495
 7600/25000 [========>.....................] - ETA: 11s - loss: 0.3588 - accuracy: 0.9493
 7700/25000 [========>.....................] - ETA: 11s - loss: 0.3587 - accuracy: 0.9494
 7800/25000 [========>.....................] - ETA: 10s - loss: 0.3584 - accuracy: 0.9495
 7900/25000 [========>.....................] - ETA: 10s - loss: 0.3581 - accuracy: 0.9497
 8000/25000 [========>.....................] - ETA: 10s - loss: 0.3576 - accuracy: 0.9503
 8100/25000 [========>.....................] - ETA: 10s - loss: 0.3576 - accuracy: 0.9501
 8200/25000 [========>.....................] - ETA: 10s - loss: 0.3578 - accuracy: 0.9499
 8300/25000 [========>.....................] - ETA: 10s - loss: 0.3577 - accuracy: 0.9500
 8400/25000 [=========>....................] - ETA: 10s - loss: 0.3578 - accuracy: 0.9499
 8500/25000 [=========>....................] - ETA: 10s - loss: 0.3581 - accuracy: 0.9495
 8600/25000 [=========>....................] - ETA: 10s - loss: 0.3578 - accuracy: 0.9498
 8700/25000 [=========>....................] - ETA: 10s - loss: 0.3576 - accuracy: 0.9499
 8800/25000 [=========>....................] - ETA: 10s - loss: 0.3576 - accuracy: 0.9498
 8900/25000 [=========>....................] - ETA: 10s - loss: 0.3581 - accuracy: 0.9491
 9000/25000 [=========>....................] - ETA: 10s - loss: 0.3577 - accuracy: 0.9494
 9100/25000 [=========>....................] - ETA: 10s - loss: 0.3576 - accuracy: 0.9495
 9200/25000 [==========>...................] - ETA: 10s - loss: 0.3580 - accuracy: 0.9490
 9300/25000 [==========>...................] - ETA: 9s - loss: 0.3584 - accuracy: 0.9486 
 9400/25000 [==========>...................] - ETA: 9s - loss: 0.3587 - accuracy: 0.9481
 9500/25000 [==========>...................] - ETA: 9s - loss: 0.3590 - accuracy: 0.9478
 9600/25000 [==========>...................] - ETA: 9s - loss: 0.3591 - accuracy: 0.9477
 9700/25000 [==========>...................] - ETA: 9s - loss: 0.3586 - accuracy: 0.9481
 9800/25000 [==========>...................] - ETA: 9s - loss: 0.3588 - accuracy: 0.9479
 9900/25000 [==========>...................] - ETA: 9s - loss: 0.3587 - accuracy: 0.9479
10000/25000 [===========>..................] - ETA: 9s - loss: 0.3585 - accuracy: 0.9480
10100/25000 [===========>..................] - ETA: 9s - loss: 0.3584 - accuracy: 0.9480
10200/25000 [===========>..................] - ETA: 9s - loss: 0.3586 - accuracy: 0.9478
10300/25000 [===========>..................] - ETA: 9s - loss: 0.3585 - accuracy: 0.9480
10400/25000 [===========>..................] - ETA: 9s - loss: 0.3582 - accuracy: 0.9481
10500/25000 [===========>..................] - ETA: 9s - loss: 0.3578 - accuracy: 0.9485
10600/25000 [===========>..................] - ETA: 9s - loss: 0.3577 - accuracy: 0.9485
10700/25000 [===========>..................] - ETA: 9s - loss: 0.3582 - accuracy: 0.9481
10800/25000 [===========>..................] - ETA: 8s - loss: 0.3584 - accuracy: 0.9478
10900/25000 [============>.................] - ETA: 8s - loss: 0.3585 - accuracy: 0.9477
11000/25000 [============>.................] - ETA: 8s - loss: 0.3581 - accuracy: 0.9480
11100/25000 [============>.................] - ETA: 8s - loss: 0.3582 - accuracy: 0.9479
11200/25000 [============>.................] - ETA: 8s - loss: 0.3581 - accuracy: 0.9479
11300/25000 [============>.................] - ETA: 8s - loss: 0.3580 - accuracy: 0.9480
11400/25000 [============>.................] - ETA: 8s - loss: 0.3575 - accuracy: 0.9484
11500/25000 [============>.................] - ETA: 8s - loss: 0.3575 - accuracy: 0.9484
11600/25000 [============>.................] - ETA: 8s - loss: 0.3575 - accuracy: 0.9483
11700/25000 [=============>................] - ETA: 8s - loss: 0.3573 - accuracy: 0.9484
11800/25000 [=============>................] - ETA: 8s - loss: 0.3575 - accuracy: 0.9482
11900/25000 [=============>................] - ETA: 8s - loss: 0.3575 - accuracy: 0.9482
12000/25000 [=============>................] - ETA: 8s - loss: 0.3574 - accuracy: 0.9482
12100/25000 [=============>................] - ETA: 8s - loss: 0.3575 - accuracy: 0.9480
12200/25000 [=============>................] - ETA: 8s - loss: 0.3572 - accuracy: 0.9482
12300/25000 [=============>................] - ETA: 8s - loss: 0.3572 - accuracy: 0.9481
12400/25000 [=============>................] - ETA: 7s - loss: 0.3571 - accuracy: 0.9482
12500/25000 [==============>...............] - ETA: 7s - loss: 0.3569 - accuracy: 0.9483
12600/25000 [==============>...............] - ETA: 7s - loss: 0.3572 - accuracy: 0.9480
12700/25000 [==============>...............] - ETA: 7s - loss: 0.3569 - accuracy: 0.9482
12800/25000 [==============>...............] - ETA: 7s - loss: 0.3570 - accuracy: 0.9480
12900/25000 [==============>...............] - ETA: 7s - loss: 0.3570 - accuracy: 0.9481
13000/25000 [==============>...............] - ETA: 7s - loss: 0.3572 - accuracy: 0.9478
13100/25000 [==============>...............] - ETA: 7s - loss: 0.3572 - accuracy: 0.9478
13200/25000 [==============>...............] - ETA: 7s - loss: 0.3572 - accuracy: 0.9477
13300/25000 [==============>...............] - ETA: 7s - loss: 0.3573 - accuracy: 0.9475
13400/25000 [===============>..............] - ETA: 7s - loss: 0.3572 - accuracy: 0.9475
13500/25000 [===============>..............] - ETA: 7s - loss: 0.3574 - accuracy: 0.9473
13600/25000 [===============>..............] - ETA: 7s - loss: 0.3573 - accuracy: 0.9473
13700/25000 [===============>..............] - ETA: 7s - loss: 0.3574 - accuracy: 0.9472
13800/25000 [===============>..............] - ETA: 7s - loss: 0.3573 - accuracy: 0.9472
13900/25000 [===============>..............] - ETA: 7s - loss: 0.3571 - accuracy: 0.9472
14000/25000 [===============>..............] - ETA: 6s - loss: 0.3571 - accuracy: 0.9472
14100/25000 [===============>..............] - ETA: 6s - loss: 0.3570 - accuracy: 0.9472
14200/25000 [================>.............] - ETA: 6s - loss: 0.3570 - accuracy: 0.9473
14300/25000 [================>.............] - ETA: 6s - loss: 0.3570 - accuracy: 0.9471
14400/25000 [================>.............] - ETA: 6s - loss: 0.3570 - accuracy: 0.9471
14500/25000 [================>.............] - ETA: 6s - loss: 0.3570 - accuracy: 0.9470
14600/25000 [================>.............] - ETA: 6s - loss: 0.3571 - accuracy: 0.9468
14700/25000 [================>.............] - ETA: 6s - loss: 0.3569 - accuracy: 0.9470
14800/25000 [================>.............] - ETA: 6s - loss: 0.3571 - accuracy: 0.9468
14900/25000 [================>.............] - ETA: 6s - loss: 0.3571 - accuracy: 0.9468
15000/25000 [=================>............] - ETA: 6s - loss: 0.3571 - accuracy: 0.9468
15100/25000 [=================>............] - ETA: 6s - loss: 0.3570 - accuracy: 0.9469
15200/25000 [=================>............] - ETA: 6s - loss: 0.3570 - accuracy: 0.9468
15300/25000 [=================>............] - ETA: 6s - loss: 0.3568 - accuracy: 0.9468
15400/25000 [=================>............] - ETA: 6s - loss: 0.3566 - accuracy: 0.9469
15500/25000 [=================>............] - ETA: 5s - loss: 0.3568 - accuracy: 0.9467
15600/25000 [=================>............] - ETA: 5s - loss: 0.3567 - accuracy: 0.9467
15700/25000 [=================>............] - ETA: 5s - loss: 0.3566 - accuracy: 0.9468
15800/25000 [=================>............] - ETA: 5s - loss: 0.3564 - accuracy: 0.9469
15900/25000 [==================>...........] - ETA: 5s - loss: 0.3563 - accuracy: 0.9470
16000/25000 [==================>...........] - ETA: 5s - loss: 0.3565 - accuracy: 0.9468
16100/25000 [==================>...........] - ETA: 5s - loss: 0.3565 - accuracy: 0.9468
16200/25000 [==================>...........] - ETA: 5s - loss: 0.3566 - accuracy: 0.9465
16300/25000 [==================>...........] - ETA: 5s - loss: 0.3562 - accuracy: 0.9469
16400/25000 [==================>...........] - ETA: 5s - loss: 0.3563 - accuracy: 0.9468
16500/25000 [==================>...........] - ETA: 5s - loss: 0.3562 - accuracy: 0.9469
16600/25000 [==================>...........] - ETA: 5s - loss: 0.3562 - accuracy: 0.9467
16700/25000 [===================>..........] - ETA: 5s - loss: 0.3562 - accuracy: 0.9467
16800/25000 [===================>..........] - ETA: 5s - loss: 0.3562 - accuracy: 0.9467
16900/25000 [===================>..........] - ETA: 5s - loss: 0.3562 - accuracy: 0.9466
17000/25000 [===================>..........] - ETA: 5s - loss: 0.3563 - accuracy: 0.9465
17100/25000 [===================>..........] - ETA: 4s - loss: 0.3564 - accuracy: 0.9463
17200/25000 [===================>..........] - ETA: 4s - loss: 0.3563 - accuracy: 0.9463
17300/25000 [===================>..........] - ETA: 4s - loss: 0.3563 - accuracy: 0.9464
17400/25000 [===================>..........] - ETA: 4s - loss: 0.3560 - accuracy: 0.9465
17500/25000 [====================>.........] - ETA: 4s - loss: 0.3561 - accuracy: 0.9465
17600/25000 [====================>.........] - ETA: 4s - loss: 0.3561 - accuracy: 0.9464
17700/25000 [====================>.........] - ETA: 4s - loss: 0.3560 - accuracy: 0.9463
17800/25000 [====================>.........] - ETA: 4s - loss: 0.3562 - accuracy: 0.9462
17900/25000 [====================>.........] - ETA: 4s - loss: 0.3563 - accuracy: 0.9459
18000/25000 [====================>.........] - ETA: 4s - loss: 0.3561 - accuracy: 0.9461
18100/25000 [====================>.........] - ETA: 4s - loss: 0.3562 - accuracy: 0.9459
18200/25000 [====================>.........] - ETA: 4s - loss: 0.3562 - accuracy: 0.9458
18300/25000 [====================>.........] - ETA: 4s - loss: 0.3561 - accuracy: 0.9458
18400/25000 [=====================>........] - ETA: 4s - loss: 0.3560 - accuracy: 0.9459
18500/25000 [=====================>........] - ETA: 4s - loss: 0.3560 - accuracy: 0.9458
18600/25000 [=====================>........] - ETA: 4s - loss: 0.3558 - accuracy: 0.9459
18700/25000 [=====================>........] - ETA: 3s - loss: 0.3557 - accuracy: 0.9460
18800/25000 [=====================>........] - ETA: 3s - loss: 0.3556 - accuracy: 0.9460
18900/25000 [=====================>........] - ETA: 3s - loss: 0.3558 - accuracy: 0.9458
19000/25000 [=====================>........] - ETA: 3s - loss: 0.3564 - accuracy: 0.9453
19100/25000 [=====================>........] - ETA: 3s - loss: 0.3563 - accuracy: 0.9452
19200/25000 [======================>.......] - ETA: 3s - loss: 0.3562 - accuracy: 0.9453
19300/25000 [======================>.......] - ETA: 3s - loss: 0.3562 - accuracy: 0.9453
19400/25000 [======================>.......] - ETA: 3s - loss: 0.3564 - accuracy: 0.9451
19500/25000 [======================>.......] - ETA: 3s - loss: 0.3564 - accuracy: 0.9450
19600/25000 [======================>.......] - ETA: 3s - loss: 0.3566 - accuracy: 0.9447
19700/25000 [======================>.......] - ETA: 3s - loss: 0.3564 - accuracy: 0.9449
19800/25000 [======================>.......] - ETA: 3s - loss: 0.3563 - accuracy: 0.9451
19900/25000 [======================>.......] - ETA: 3s - loss: 0.3561 - accuracy: 0.9452
20000/25000 [=======================>......] - ETA: 3s - loss: 0.3564 - accuracy: 0.9449
20100/25000 [=======================>......] - ETA: 3s - loss: 0.3565 - accuracy: 0.9447
20200/25000 [=======================>......] - ETA: 3s - loss: 0.3564 - accuracy: 0.9448
20300/25000 [=======================>......] - ETA: 2s - loss: 0.3564 - accuracy: 0.9447
20400/25000 [=======================>......] - ETA: 2s - loss: 0.3563 - accuracy: 0.9448
20500/25000 [=======================>......] - ETA: 2s - loss: 0.3564 - accuracy: 0.9446
20600/25000 [=======================>......] - ETA: 2s - loss: 0.3564 - accuracy: 0.9446
20700/25000 [=======================>......] - ETA: 2s - loss: 0.3564 - accuracy: 0.9446
20800/25000 [=======================>......] - ETA: 2s - loss: 0.3565 - accuracy: 0.9444
20900/25000 [========================>.....] - ETA: 2s - loss: 0.3565 - accuracy: 0.9444
21000/25000 [========================>.....] - ETA: 2s - loss: 0.3564 - accuracy: 0.9445
21100/25000 [========================>.....] - ETA: 2s - loss: 0.3564 - accuracy: 0.9445
21200/25000 [========================>.....] - ETA: 2s - loss: 0.3562 - accuracy: 0.9446
21300/25000 [========================>.....] - ETA: 2s - loss: 0.3562 - accuracy: 0.9446
21400/25000 [========================>.....] - ETA: 2s - loss: 0.3559 - accuracy: 0.9448
21500/25000 [========================>.....] - ETA: 2s - loss: 0.3559 - accuracy: 0.9448
21600/25000 [========================>.....] - ETA: 2s - loss: 0.3558 - accuracy: 0.9448
21700/25000 [=========================>....] - ETA: 2s - loss: 0.3558 - accuracy: 0.9448
21800/25000 [=========================>....] - ETA: 2s - loss: 0.3556 - accuracy: 0.9450
21900/25000 [=========================>....] - ETA: 1s - loss: 0.3554 - accuracy: 0.9451
22000/25000 [=========================>....] - ETA: 1s - loss: 0.3552 - accuracy: 0.9451
22100/25000 [=========================>....] - ETA: 1s - loss: 0.3553 - accuracy: 0.9450
22200/25000 [=========================>....] - ETA: 1s - loss: 0.3553 - accuracy: 0.9450
22300/25000 [=========================>....] - ETA: 1s - loss: 0.3552 - accuracy: 0.9451
22400/25000 [=========================>....] - ETA: 1s - loss: 0.3552 - accuracy: 0.9450
22500/25000 [==========================>...] - ETA: 1s - loss: 0.3551 - accuracy: 0.9450
22600/25000 [==========================>...] - ETA: 1s - loss: 0.3552 - accuracy: 0.9449
22700/25000 [==========================>...] - ETA: 1s - loss: 0.3551 - accuracy: 0.9449
22800/25000 [==========================>...] - ETA: 1s - loss: 0.3551 - accuracy: 0.9450
22900/25000 [==========================>...] - ETA: 1s - loss: 0.3550 - accuracy: 0.9451
23000/25000 [==========================>...] - ETA: 1s - loss: 0.3549 - accuracy: 0.9451
23100/25000 [==========================>...] - ETA: 1s - loss: 0.3548 - accuracy: 0.9451
23200/25000 [==========================>...] - ETA: 1s - loss: 0.3549 - accuracy: 0.9450
23300/25000 [==========================>...] - ETA: 1s - loss: 0.3550 - accuracy: 0.9449
23400/25000 [===========================>..] - ETA: 1s - loss: 0.3550 - accuracy: 0.9449
23500/25000 [===========================>..] - ETA: 0s - loss: 0.3552 - accuracy: 0.9446
23600/25000 [===========================>..] - ETA: 0s - loss: 0.3552 - accuracy: 0.9446
23700/25000 [===========================>..] - ETA: 0s - loss: 0.3553 - accuracy: 0.9445
23800/25000 [===========================>..] - ETA: 0s - loss: 0.3553 - accuracy: 0.9444
23900/25000 [===========================>..] - ETA: 0s - loss: 0.3554 - accuracy: 0.9443
24000/25000 [===========================>..] - ETA: 0s - loss: 0.3553 - accuracy: 0.9443
24100/25000 [===========================>..] - ETA: 0s - loss: 0.3553 - accuracy: 0.9444
24200/25000 [============================>.] - ETA: 0s - loss: 0.3551 - accuracy: 0.9445
24300/25000 [============================>.] - ETA: 0s - loss: 0.3550 - accuracy: 0.9445
24400/25000 [============================>.] - ETA: 0s - loss: 0.3550 - accuracy: 0.9445
24500/25000 [============================>.] - ETA: 0s - loss: 0.3550 - accuracy: 0.9445
24600/25000 [============================>.] - ETA: 0s - loss: 0.3549 - accuracy: 0.9445
24700/25000 [============================>.] - ETA: 0s - loss: 0.3549 - accuracy: 0.9445
24800/25000 [============================>.] - ETA: 0s - loss: 0.3549 - accuracy: 0.9444
24900/25000 [============================>.] - ETA: 0s - loss: 0.3550 - accuracy: 0.9443
25000/25000 [==============================] - 20s 793us/step - loss: 0.3551 - accuracy: 0.9441 - val_loss: 0.4307 - val_accuracy: 0.8634
Epoch 8/10

  100/25000 [..............................] - ETA: 15s - loss: 0.3232 - accuracy: 0.9600
  200/25000 [..............................] - ETA: 15s - loss: 0.3161 - accuracy: 0.9700
  300/25000 [..............................] - ETA: 15s - loss: 0.3332 - accuracy: 0.9567
  400/25000 [..............................] - ETA: 15s - loss: 0.3314 - accuracy: 0.9575
  500/25000 [..............................] - ETA: 15s - loss: 0.3338 - accuracy: 0.9560
  600/25000 [..............................] - ETA: 15s - loss: 0.3355 - accuracy: 0.9550
  700/25000 [..............................] - ETA: 15s - loss: 0.3405 - accuracy: 0.9500
  800/25000 [..............................] - ETA: 15s - loss: 0.3393 - accuracy: 0.9500
  900/25000 [>.............................] - ETA: 15s - loss: 0.3440 - accuracy: 0.9456
 1000/25000 [>.............................] - ETA: 14s - loss: 0.3423 - accuracy: 0.9470
 1100/25000 [>.............................] - ETA: 14s - loss: 0.3448 - accuracy: 0.9445
 1200/25000 [>.............................] - ETA: 14s - loss: 0.3449 - accuracy: 0.9442
 1300/25000 [>.............................] - ETA: 14s - loss: 0.3439 - accuracy: 0.9446
 1400/25000 [>.............................] - ETA: 14s - loss: 0.3440 - accuracy: 0.9443
 1500/25000 [>.............................] - ETA: 14s - loss: 0.3430 - accuracy: 0.9453
 1600/25000 [>.............................] - ETA: 14s - loss: 0.3435 - accuracy: 0.9450
 1700/25000 [=>............................] - ETA: 14s - loss: 0.3432 - accuracy: 0.9453
 1800/25000 [=>............................] - ETA: 14s - loss: 0.3424 - accuracy: 0.9456
 1900/25000 [=>............................] - ETA: 14s - loss: 0.3416 - accuracy: 0.9463
 2000/25000 [=>............................] - ETA: 14s - loss: 0.3421 - accuracy: 0.9460
 2100/25000 [=>............................] - ETA: 14s - loss: 0.3415 - accuracy: 0.9467
 2200/25000 [=>............................] - ETA: 14s - loss: 0.3409 - accuracy: 0.9473
 2300/25000 [=>............................] - ETA: 14s - loss: 0.3410 - accuracy: 0.9470
 2400/25000 [=>............................] - ETA: 14s - loss: 0.3408 - accuracy: 0.9471
 2500/25000 [==>...........................] - ETA: 14s - loss: 0.3395 - accuracy: 0.9480
 2600/25000 [==>...........................] - ETA: 14s - loss: 0.3386 - accuracy: 0.9485
 2700/25000 [==>...........................] - ETA: 13s - loss: 0.3372 - accuracy: 0.9496
 2800/25000 [==>...........................] - ETA: 13s - loss: 0.3380 - accuracy: 0.9493
 2900/25000 [==>...........................] - ETA: 13s - loss: 0.3385 - accuracy: 0.9486
 3000/25000 [==>...........................] - ETA: 13s - loss: 0.3392 - accuracy: 0.9473
 3100/25000 [==>...........................] - ETA: 13s - loss: 0.3388 - accuracy: 0.9477
 3200/25000 [==>...........................] - ETA: 13s - loss: 0.3389 - accuracy: 0.9478
 3300/25000 [==>...........................] - ETA: 13s - loss: 0.3386 - accuracy: 0.9482
 3400/25000 [===>..........................] - ETA: 13s - loss: 0.3385 - accuracy: 0.9482
 3500/25000 [===>..........................] - ETA: 13s - loss: 0.3374 - accuracy: 0.9491
 3600/25000 [===>..........................] - ETA: 13s - loss: 0.3365 - accuracy: 0.9500
 3700/25000 [===>..........................] - ETA: 13s - loss: 0.3369 - accuracy: 0.9495
 3800/25000 [===>..........................] - ETA: 13s - loss: 0.3358 - accuracy: 0.9505
 3900/25000 [===>..........................] - ETA: 13s - loss: 0.3355 - accuracy: 0.9508
 4000/25000 [===>..........................] - ETA: 13s - loss: 0.3342 - accuracy: 0.9520
 4100/25000 [===>..........................] - ETA: 13s - loss: 0.3342 - accuracy: 0.9520
 4200/25000 [====>.........................] - ETA: 13s - loss: 0.3352 - accuracy: 0.9512
 4300/25000 [====>.........................] - ETA: 13s - loss: 0.3348 - accuracy: 0.9516
 4400/25000 [====>.........................] - ETA: 12s - loss: 0.3355 - accuracy: 0.9509
 4500/25000 [====>.........................] - ETA: 12s - loss: 0.3353 - accuracy: 0.9509
 4600/25000 [====>.........................] - ETA: 12s - loss: 0.3356 - accuracy: 0.9504
 4700/25000 [====>.........................] - ETA: 12s - loss: 0.3350 - accuracy: 0.9509
 4800/25000 [====>.........................] - ETA: 12s - loss: 0.3354 - accuracy: 0.9504
 4900/25000 [====>.........................] - ETA: 12s - loss: 0.3354 - accuracy: 0.9504
 5000/25000 [=====>........................] - ETA: 12s - loss: 0.3346 - accuracy: 0.9510
 5100/25000 [=====>........................] - ETA: 12s - loss: 0.3344 - accuracy: 0.9512
 5200/25000 [=====>........................] - ETA: 12s - loss: 0.3345 - accuracy: 0.9512
 5300/25000 [=====>........................] - ETA: 12s - loss: 0.3339 - accuracy: 0.9517
 5400/25000 [=====>........................] - ETA: 12s - loss: 0.3333 - accuracy: 0.9520
 5500/25000 [=====>........................] - ETA: 12s - loss: 0.3333 - accuracy: 0.9520
 5600/25000 [=====>........................] - ETA: 12s - loss: 0.3341 - accuracy: 0.9513
 5700/25000 [=====>........................] - ETA: 12s - loss: 0.3339 - accuracy: 0.9514
 5800/25000 [=====>........................] - ETA: 12s - loss: 0.3337 - accuracy: 0.9516
 5900/25000 [======>.......................] - ETA: 12s - loss: 0.3340 - accuracy: 0.9512
 6000/25000 [======>.......................] - ETA: 11s - loss: 0.3339 - accuracy: 0.9513
 6100/25000 [======>.......................] - ETA: 11s - loss: 0.3345 - accuracy: 0.9507
 6200/25000 [======>.......................] - ETA: 11s - loss: 0.3339 - accuracy: 0.9511
 6300/25000 [======>.......................] - ETA: 11s - loss: 0.3343 - accuracy: 0.9508
 6400/25000 [======>.......................] - ETA: 11s - loss: 0.3341 - accuracy: 0.9509
 6500/25000 [======>.......................] - ETA: 11s - loss: 0.3343 - accuracy: 0.9508
 6600/25000 [======>.......................] - ETA: 11s - loss: 0.3337 - accuracy: 0.9512
 6700/25000 [=======>......................] - ETA: 11s - loss: 0.3337 - accuracy: 0.9512
 6800/25000 [=======>......................] - ETA: 11s - loss: 0.3339 - accuracy: 0.9510
 6900/25000 [=======>......................] - ETA: 11s - loss: 0.3337 - accuracy: 0.9512
 7000/25000 [=======>......................] - ETA: 11s - loss: 0.3338 - accuracy: 0.9510
 7100/25000 [=======>......................] - ETA: 11s - loss: 0.3335 - accuracy: 0.9511
 7200/25000 [=======>......................] - ETA: 11s - loss: 0.3330 - accuracy: 0.9515
 7300/25000 [=======>......................] - ETA: 11s - loss: 0.3332 - accuracy: 0.9512
 7400/25000 [=======>......................] - ETA: 11s - loss: 0.3334 - accuracy: 0.9509
 7500/25000 [========>.....................] - ETA: 11s - loss: 0.3331 - accuracy: 0.9511
 7600/25000 [========>.....................] - ETA: 10s - loss: 0.3337 - accuracy: 0.9507
 7700/25000 [========>.....................] - ETA: 10s - loss: 0.3337 - accuracy: 0.9506
 7800/25000 [========>.....................] - ETA: 10s - loss: 0.3343 - accuracy: 0.9500
 7900/25000 [========>.....................] - ETA: 10s - loss: 0.3349 - accuracy: 0.9495
 8000/25000 [========>.....................] - ETA: 10s - loss: 0.3345 - accuracy: 0.9498
 8100/25000 [========>.....................] - ETA: 10s - loss: 0.3346 - accuracy: 0.9498
 8200/25000 [========>.....................] - ETA: 10s - loss: 0.3343 - accuracy: 0.9500
 8300/25000 [========>.....................] - ETA: 10s - loss: 0.3340 - accuracy: 0.9501
 8400/25000 [=========>....................] - ETA: 10s - loss: 0.3333 - accuracy: 0.9507
 8500/25000 [=========>....................] - ETA: 10s - loss: 0.3333 - accuracy: 0.9508
 8600/25000 [=========>....................] - ETA: 10s - loss: 0.3332 - accuracy: 0.9510
 8700/25000 [=========>....................] - ETA: 10s - loss: 0.3330 - accuracy: 0.9511
 8800/25000 [=========>....................] - ETA: 10s - loss: 0.3326 - accuracy: 0.9515
 8900/25000 [=========>....................] - ETA: 10s - loss: 0.3329 - accuracy: 0.9512
 9000/25000 [=========>....................] - ETA: 10s - loss: 0.3327 - accuracy: 0.9514
 9100/25000 [=========>....................] - ETA: 10s - loss: 0.3321 - accuracy: 0.9519
 9200/25000 [==========>...................] - ETA: 9s - loss: 0.3322 - accuracy: 0.9518 
 9300/25000 [==========>...................] - ETA: 9s - loss: 0.3320 - accuracy: 0.9518
 9400/25000 [==========>...................] - ETA: 9s - loss: 0.3320 - accuracy: 0.9517
 9500/25000 [==========>...................] - ETA: 9s - loss: 0.3317 - accuracy: 0.9520
 9600/25000 [==========>...................] - ETA: 9s - loss: 0.3319 - accuracy: 0.9517
 9700/25000 [==========>...................] - ETA: 9s - loss: 0.3316 - accuracy: 0.9519
 9800/25000 [==========>...................] - ETA: 9s - loss: 0.3314 - accuracy: 0.9519
 9900/25000 [==========>...................] - ETA: 9s - loss: 0.3309 - accuracy: 0.9522
10000/25000 [===========>..................] - ETA: 9s - loss: 0.3308 - accuracy: 0.9523
10100/25000 [===========>..................] - ETA: 9s - loss: 0.3306 - accuracy: 0.9525
10200/25000 [===========>..................] - ETA: 9s - loss: 0.3308 - accuracy: 0.9523
10300/25000 [===========>..................] - ETA: 9s - loss: 0.3307 - accuracy: 0.9523
10400/25000 [===========>..................] - ETA: 9s - loss: 0.3303 - accuracy: 0.9527
10500/25000 [===========>..................] - ETA: 9s - loss: 0.3302 - accuracy: 0.9528
10600/25000 [===========>..................] - ETA: 9s - loss: 0.3305 - accuracy: 0.9525
10700/25000 [===========>..................] - ETA: 8s - loss: 0.3302 - accuracy: 0.9526
10800/25000 [===========>..................] - ETA: 8s - loss: 0.3303 - accuracy: 0.9525
10900/25000 [============>.................] - ETA: 8s - loss: 0.3301 - accuracy: 0.9527
11000/25000 [============>.................] - ETA: 8s - loss: 0.3300 - accuracy: 0.9526
11100/25000 [============>.................] - ETA: 8s - loss: 0.3301 - accuracy: 0.9525
11200/25000 [============>.................] - ETA: 8s - loss: 0.3300 - accuracy: 0.9526
11300/25000 [============>.................] - ETA: 8s - loss: 0.3299 - accuracy: 0.9527
11400/25000 [============>.................] - ETA: 8s - loss: 0.3300 - accuracy: 0.9525
11500/25000 [============>.................] - ETA: 8s - loss: 0.3300 - accuracy: 0.9523
11600/25000 [============>.................] - ETA: 8s - loss: 0.3303 - accuracy: 0.9521
11700/25000 [=============>................] - ETA: 8s - loss: 0.3301 - accuracy: 0.9522
11800/25000 [=============>................] - ETA: 8s - loss: 0.3302 - accuracy: 0.9520
11900/25000 [=============>................] - ETA: 8s - loss: 0.3308 - accuracy: 0.9515
12000/25000 [=============>................] - ETA: 8s - loss: 0.3306 - accuracy: 0.9517
12100/25000 [=============>................] - ETA: 8s - loss: 0.3307 - accuracy: 0.9517
12200/25000 [=============>................] - ETA: 8s - loss: 0.3305 - accuracy: 0.9518
12300/25000 [=============>................] - ETA: 7s - loss: 0.3305 - accuracy: 0.9519
12400/25000 [=============>................] - ETA: 7s - loss: 0.3310 - accuracy: 0.9514
12500/25000 [==============>...............] - ETA: 7s - loss: 0.3308 - accuracy: 0.9515
12600/25000 [==============>...............] - ETA: 7s - loss: 0.3309 - accuracy: 0.9513
12700/25000 [==============>...............] - ETA: 7s - loss: 0.3306 - accuracy: 0.9516
12800/25000 [==============>...............] - ETA: 7s - loss: 0.3305 - accuracy: 0.9516
12900/25000 [==============>...............] - ETA: 7s - loss: 0.3305 - accuracy: 0.9516
13000/25000 [==============>...............] - ETA: 7s - loss: 0.3305 - accuracy: 0.9516
13100/25000 [==============>...............] - ETA: 7s - loss: 0.3305 - accuracy: 0.9516
13200/25000 [==============>...............] - ETA: 7s - loss: 0.3302 - accuracy: 0.9518
13300/25000 [==============>...............] - ETA: 7s - loss: 0.3302 - accuracy: 0.9517
13400/25000 [===============>..............] - ETA: 7s - loss: 0.3300 - accuracy: 0.9519
13500/25000 [===============>..............] - ETA: 7s - loss: 0.3301 - accuracy: 0.9517
13600/25000 [===============>..............] - ETA: 7s - loss: 0.3304 - accuracy: 0.9515
13700/25000 [===============>..............] - ETA: 7s - loss: 0.3302 - accuracy: 0.9515
13800/25000 [===============>..............] - ETA: 7s - loss: 0.3301 - accuracy: 0.9515
13900/25000 [===============>..............] - ETA: 6s - loss: 0.3301 - accuracy: 0.9515
14000/25000 [===============>..............] - ETA: 6s - loss: 0.3302 - accuracy: 0.9514
14100/25000 [===============>..............] - ETA: 6s - loss: 0.3299 - accuracy: 0.9516
14200/25000 [================>.............] - ETA: 6s - loss: 0.3299 - accuracy: 0.9517
14300/25000 [================>.............] - ETA: 6s - loss: 0.3297 - accuracy: 0.9518
14400/25000 [================>.............] - ETA: 6s - loss: 0.3298 - accuracy: 0.9517
14500/25000 [================>.............] - ETA: 6s - loss: 0.3297 - accuracy: 0.9519
14600/25000 [================>.............] - ETA: 6s - loss: 0.3297 - accuracy: 0.9518
14700/25000 [================>.............] - ETA: 6s - loss: 0.3296 - accuracy: 0.9519
14800/25000 [================>.............] - ETA: 6s - loss: 0.3294 - accuracy: 0.9520
14900/25000 [================>.............] - ETA: 6s - loss: 0.3293 - accuracy: 0.9521
15000/25000 [=================>............] - ETA: 6s - loss: 0.3293 - accuracy: 0.9520
15100/25000 [=================>............] - ETA: 6s - loss: 0.3291 - accuracy: 0.9522
15200/25000 [=================>............] - ETA: 6s - loss: 0.3288 - accuracy: 0.9524
15300/25000 [=================>............] - ETA: 6s - loss: 0.3288 - accuracy: 0.9523
15400/25000 [=================>............] - ETA: 6s - loss: 0.3288 - accuracy: 0.9523
15500/25000 [=================>............] - ETA: 5s - loss: 0.3292 - accuracy: 0.9519
15600/25000 [=================>............] - ETA: 5s - loss: 0.3290 - accuracy: 0.9519
15700/25000 [=================>............] - ETA: 5s - loss: 0.3288 - accuracy: 0.9520
15800/25000 [=================>............] - ETA: 5s - loss: 0.3289 - accuracy: 0.9520
15900/25000 [==================>...........] - ETA: 5s - loss: 0.3288 - accuracy: 0.9520
16000/25000 [==================>...........] - ETA: 5s - loss: 0.3289 - accuracy: 0.9518
16100/25000 [==================>...........] - ETA: 5s - loss: 0.3288 - accuracy: 0.9519
16200/25000 [==================>...........] - ETA: 5s - loss: 0.3286 - accuracy: 0.9519
16300/25000 [==================>...........] - ETA: 5s - loss: 0.3284 - accuracy: 0.9520
16400/25000 [==================>...........] - ETA: 5s - loss: 0.3287 - accuracy: 0.9518
16500/25000 [==================>...........] - ETA: 5s - loss: 0.3287 - accuracy: 0.9519
16600/25000 [==================>...........] - ETA: 5s - loss: 0.3285 - accuracy: 0.9520
16700/25000 [===================>..........] - ETA: 5s - loss: 0.3286 - accuracy: 0.9519
16800/25000 [===================>..........] - ETA: 5s - loss: 0.3289 - accuracy: 0.9517
16900/25000 [===================>..........] - ETA: 5s - loss: 0.3288 - accuracy: 0.9518
17000/25000 [===================>..........] - ETA: 5s - loss: 0.3289 - accuracy: 0.9517
17100/25000 [===================>..........] - ETA: 4s - loss: 0.3287 - accuracy: 0.9518
17200/25000 [===================>..........] - ETA: 4s - loss: 0.3287 - accuracy: 0.9517
17300/25000 [===================>..........] - ETA: 4s - loss: 0.3288 - accuracy: 0.9516
17400/25000 [===================>..........] - ETA: 4s - loss: 0.3287 - accuracy: 0.9516
17500/25000 [====================>.........] - ETA: 4s - loss: 0.3285 - accuracy: 0.9517
17600/25000 [====================>.........] - ETA: 4s - loss: 0.3286 - accuracy: 0.9516
17700/25000 [====================>.........] - ETA: 4s - loss: 0.3288 - accuracy: 0.9514
17800/25000 [====================>.........] - ETA: 4s - loss: 0.3287 - accuracy: 0.9515
17900/25000 [====================>.........] - ETA: 4s - loss: 0.3285 - accuracy: 0.9515
18000/25000 [====================>.........] - ETA: 4s - loss: 0.3288 - accuracy: 0.9512
18100/25000 [====================>.........] - ETA: 4s - loss: 0.3286 - accuracy: 0.9513
18200/25000 [====================>.........] - ETA: 4s - loss: 0.3285 - accuracy: 0.9514
18300/25000 [====================>.........] - ETA: 4s - loss: 0.3285 - accuracy: 0.9514
18400/25000 [=====================>........] - ETA: 4s - loss: 0.3286 - accuracy: 0.9513
18500/25000 [=====================>........] - ETA: 4s - loss: 0.3286 - accuracy: 0.9512
18600/25000 [=====================>........] - ETA: 4s - loss: 0.3287 - accuracy: 0.9511
18700/25000 [=====================>........] - ETA: 3s - loss: 0.3288 - accuracy: 0.9510
18800/25000 [=====================>........] - ETA: 3s - loss: 0.3288 - accuracy: 0.9510
18900/25000 [=====================>........] - ETA: 3s - loss: 0.3288 - accuracy: 0.9510
19000/25000 [=====================>........] - ETA: 3s - loss: 0.3286 - accuracy: 0.9511
19100/25000 [=====================>........] - ETA: 3s - loss: 0.3285 - accuracy: 0.9510
19200/25000 [======================>.......] - ETA: 3s - loss: 0.3286 - accuracy: 0.9510
19300/25000 [======================>.......] - ETA: 3s - loss: 0.3289 - accuracy: 0.9507
19400/25000 [======================>.......] - ETA: 3s - loss: 0.3290 - accuracy: 0.9506
19500/25000 [======================>.......] - ETA: 3s - loss: 0.3290 - accuracy: 0.9506
19600/25000 [======================>.......] - ETA: 3s - loss: 0.3291 - accuracy: 0.9505
19700/25000 [======================>.......] - ETA: 3s - loss: 0.3292 - accuracy: 0.9504
19800/25000 [======================>.......] - ETA: 3s - loss: 0.3289 - accuracy: 0.9506
19900/25000 [======================>.......] - ETA: 3s - loss: 0.3290 - accuracy: 0.9504
20000/25000 [=======================>......] - ETA: 3s - loss: 0.3288 - accuracy: 0.9505
20100/25000 [=======================>......] - ETA: 3s - loss: 0.3287 - accuracy: 0.9505
20200/25000 [=======================>......] - ETA: 3s - loss: 0.3290 - accuracy: 0.9502
20300/25000 [=======================>......] - ETA: 2s - loss: 0.3288 - accuracy: 0.9503
20400/25000 [=======================>......] - ETA: 2s - loss: 0.3287 - accuracy: 0.9503
20500/25000 [=======================>......] - ETA: 2s - loss: 0.3286 - accuracy: 0.9504
20600/25000 [=======================>......] - ETA: 2s - loss: 0.3285 - accuracy: 0.9504
20700/25000 [=======================>......] - ETA: 2s - loss: 0.3285 - accuracy: 0.9504
20800/25000 [=======================>......] - ETA: 2s - loss: 0.3285 - accuracy: 0.9504
20900/25000 [========================>.....] - ETA: 2s - loss: 0.3286 - accuracy: 0.9504
21000/25000 [========================>.....] - ETA: 2s - loss: 0.3287 - accuracy: 0.9502
21100/25000 [========================>.....] - ETA: 2s - loss: 0.3288 - accuracy: 0.9501
21200/25000 [========================>.....] - ETA: 2s - loss: 0.3287 - accuracy: 0.9501
21300/25000 [========================>.....] - ETA: 2s - loss: 0.3286 - accuracy: 0.9501
21400/25000 [========================>.....] - ETA: 2s - loss: 0.3285 - accuracy: 0.9501
21500/25000 [========================>.....] - ETA: 2s - loss: 0.3286 - accuracy: 0.9500
21600/25000 [========================>.....] - ETA: 2s - loss: 0.3287 - accuracy: 0.9499
21700/25000 [=========================>....] - ETA: 2s - loss: 0.3286 - accuracy: 0.9500
21800/25000 [=========================>....] - ETA: 2s - loss: 0.3286 - accuracy: 0.9500
21900/25000 [=========================>....] - ETA: 1s - loss: 0.3283 - accuracy: 0.9502
22000/25000 [=========================>....] - ETA: 1s - loss: 0.3281 - accuracy: 0.9502
22100/25000 [=========================>....] - ETA: 1s - loss: 0.3284 - accuracy: 0.9500
22200/25000 [=========================>....] - ETA: 1s - loss: 0.3282 - accuracy: 0.9501
22300/25000 [=========================>....] - ETA: 1s - loss: 0.3281 - accuracy: 0.9501
22400/25000 [=========================>....] - ETA: 1s - loss: 0.3279 - accuracy: 0.9502
22500/25000 [==========================>...] - ETA: 1s - loss: 0.3277 - accuracy: 0.9504
22600/25000 [==========================>...] - ETA: 1s - loss: 0.3275 - accuracy: 0.9506
22700/25000 [==========================>...] - ETA: 1s - loss: 0.3273 - accuracy: 0.9506
22800/25000 [==========================>...] - ETA: 1s - loss: 0.3272 - accuracy: 0.9507
22900/25000 [==========================>...] - ETA: 1s - loss: 0.3273 - accuracy: 0.9506
23000/25000 [==========================>...] - ETA: 1s - loss: 0.3275 - accuracy: 0.9504
23100/25000 [==========================>...] - ETA: 1s - loss: 0.3276 - accuracy: 0.9503
23200/25000 [==========================>...] - ETA: 1s - loss: 0.3274 - accuracy: 0.9504
23300/25000 [==========================>...] - ETA: 1s - loss: 0.3276 - accuracy: 0.9502
23400/25000 [===========================>..] - ETA: 1s - loss: 0.3275 - accuracy: 0.9502
23500/25000 [===========================>..] - ETA: 0s - loss: 0.3273 - accuracy: 0.9503
23600/25000 [===========================>..] - ETA: 0s - loss: 0.3273 - accuracy: 0.9503
23700/25000 [===========================>..] - ETA: 0s - loss: 0.3272 - accuracy: 0.9503
23800/25000 [===========================>..] - ETA: 0s - loss: 0.3270 - accuracy: 0.9504
23900/25000 [===========================>..] - ETA: 0s - loss: 0.3270 - accuracy: 0.9504
24000/25000 [===========================>..] - ETA: 0s - loss: 0.3269 - accuracy: 0.9504
24100/25000 [===========================>..] - ETA: 0s - loss: 0.3268 - accuracy: 0.9505
24200/25000 [============================>.] - ETA: 0s - loss: 0.3267 - accuracy: 0.9505
24300/25000 [============================>.] - ETA: 0s - loss: 0.3266 - accuracy: 0.9506
24400/25000 [============================>.] - ETA: 0s - loss: 0.3266 - accuracy: 0.9505
24500/25000 [============================>.] - ETA: 0s - loss: 0.3266 - accuracy: 0.9504
24600/25000 [============================>.] - ETA: 0s - loss: 0.3265 - accuracy: 0.9505
24700/25000 [============================>.] - ETA: 0s - loss: 0.3265 - accuracy: 0.9505
24800/25000 [============================>.] - ETA: 0s - loss: 0.3265 - accuracy: 0.9505
24900/25000 [============================>.] - ETA: 0s - loss: 0.3264 - accuracy: 0.9505
25000/25000 [==============================] - 20s 792us/step - loss: 0.3265 - accuracy: 0.9504 - val_loss: 0.4231 - val_accuracy: 0.8605
Epoch 9/10

  100/25000 [..............................] - ETA: 15s - loss: 0.3195 - accuracy: 0.9500
  200/25000 [..............................] - ETA: 15s - loss: 0.3104 - accuracy: 0.9550
  300/25000 [..............................] - ETA: 15s - loss: 0.3214 - accuracy: 0.9467
  400/25000 [..............................] - ETA: 15s - loss: 0.3126 - accuracy: 0.9550
  500/25000 [..............................] - ETA: 15s - loss: 0.3139 - accuracy: 0.9540
  600/25000 [..............................] - ETA: 14s - loss: 0.3085 - accuracy: 0.9583
  700/25000 [..............................] - ETA: 14s - loss: 0.3056 - accuracy: 0.9614
  800/25000 [..............................] - ETA: 14s - loss: 0.3059 - accuracy: 0.9613
  900/25000 [>.............................] - ETA: 14s - loss: 0.3054 - accuracy: 0.9611
 1000/25000 [>.............................] - ETA: 14s - loss: 0.3110 - accuracy: 0.9560
 1100/25000 [>.............................] - ETA: 14s - loss: 0.3134 - accuracy: 0.9536
 1200/25000 [>.............................] - ETA: 14s - loss: 0.3117 - accuracy: 0.9550
 1300/25000 [>.............................] - ETA: 14s - loss: 0.3157 - accuracy: 0.9523
 1400/25000 [>.............................] - ETA: 14s - loss: 0.3140 - accuracy: 0.9536
 1500/25000 [>.............................] - ETA: 14s - loss: 0.3131 - accuracy: 0.9547
 1600/25000 [>.............................] - ETA: 14s - loss: 0.3110 - accuracy: 0.9563
 1700/25000 [=>............................] - ETA: 14s - loss: 0.3087 - accuracy: 0.9576
 1800/25000 [=>............................] - ETA: 14s - loss: 0.3086 - accuracy: 0.9578
 1900/25000 [=>............................] - ETA: 14s - loss: 0.3066 - accuracy: 0.9589
 2000/25000 [=>............................] - ETA: 14s - loss: 0.3066 - accuracy: 0.9590
 2100/25000 [=>............................] - ETA: 14s - loss: 0.3059 - accuracy: 0.9595
 2200/25000 [=>............................] - ETA: 14s - loss: 0.3072 - accuracy: 0.9582
 2300/25000 [=>............................] - ETA: 14s - loss: 0.3068 - accuracy: 0.9587
 2400/25000 [=>............................] - ETA: 14s - loss: 0.3091 - accuracy: 0.9558
 2500/25000 [==>...........................] - ETA: 13s - loss: 0.3091 - accuracy: 0.9560
 2600/25000 [==>...........................] - ETA: 13s - loss: 0.3090 - accuracy: 0.9562
 2700/25000 [==>...........................] - ETA: 13s - loss: 0.3097 - accuracy: 0.9556
 2800/25000 [==>...........................] - ETA: 13s - loss: 0.3104 - accuracy: 0.9550
 2900/25000 [==>...........................] - ETA: 13s - loss: 0.3092 - accuracy: 0.9559
 3000/25000 [==>...........................] - ETA: 13s - loss: 0.3089 - accuracy: 0.9563
 3100/25000 [==>...........................] - ETA: 13s - loss: 0.3086 - accuracy: 0.9565
 3200/25000 [==>...........................] - ETA: 13s - loss: 0.3096 - accuracy: 0.9556
 3300/25000 [==>...........................] - ETA: 13s - loss: 0.3105 - accuracy: 0.9548
 3400/25000 [===>..........................] - ETA: 13s - loss: 0.3110 - accuracy: 0.9541
 3500/25000 [===>..........................] - ETA: 13s - loss: 0.3127 - accuracy: 0.9529
 3600/25000 [===>..........................] - ETA: 13s - loss: 0.3146 - accuracy: 0.9514
 3700/25000 [===>..........................] - ETA: 13s - loss: 0.3140 - accuracy: 0.9519
 3800/25000 [===>..........................] - ETA: 13s - loss: 0.3133 - accuracy: 0.9524
 3900/25000 [===>..........................] - ETA: 13s - loss: 0.3130 - accuracy: 0.9526
 4000/25000 [===>..........................] - ETA: 13s - loss: 0.3124 - accuracy: 0.9530
 4100/25000 [===>..........................] - ETA: 13s - loss: 0.3123 - accuracy: 0.9532
 4200/25000 [====>.........................] - ETA: 13s - loss: 0.3120 - accuracy: 0.9533
 4300/25000 [====>.........................] - ETA: 13s - loss: 0.3131 - accuracy: 0.9523
 4400/25000 [====>.........................] - ETA: 12s - loss: 0.3129 - accuracy: 0.9525
 4500/25000 [====>.........................] - ETA: 12s - loss: 0.3134 - accuracy: 0.9520
 4600/25000 [====>.........................] - ETA: 12s - loss: 0.3128 - accuracy: 0.9524
 4700/25000 [====>.........................] - ETA: 12s - loss: 0.3128 - accuracy: 0.9523
 4800/25000 [====>.........................] - ETA: 12s - loss: 0.3124 - accuracy: 0.9525
 4900/25000 [====>.........................] - ETA: 12s - loss: 0.3122 - accuracy: 0.9527
 5000/25000 [=====>........................] - ETA: 12s - loss: 0.3112 - accuracy: 0.9536
 5100/25000 [=====>........................] - ETA: 12s - loss: 0.3107 - accuracy: 0.9539
 5200/25000 [=====>........................] - ETA: 12s - loss: 0.3102 - accuracy: 0.9542
 5300/25000 [=====>........................] - ETA: 12s - loss: 0.3101 - accuracy: 0.9543
 5400/25000 [=====>........................] - ETA: 12s - loss: 0.3094 - accuracy: 0.9550
 5500/25000 [=====>........................] - ETA: 12s - loss: 0.3095 - accuracy: 0.9549
 5600/25000 [=====>........................] - ETA: 12s - loss: 0.3090 - accuracy: 0.9552
 5700/25000 [=====>........................] - ETA: 12s - loss: 0.3088 - accuracy: 0.9554
 5800/25000 [=====>........................] - ETA: 12s - loss: 0.3097 - accuracy: 0.9547
 5900/25000 [======>.......................] - ETA: 12s - loss: 0.3088 - accuracy: 0.9554
 6000/25000 [======>.......................] - ETA: 11s - loss: 0.3083 - accuracy: 0.9558
 6100/25000 [======>.......................] - ETA: 11s - loss: 0.3077 - accuracy: 0.9561
 6200/25000 [======>.......................] - ETA: 11s - loss: 0.3079 - accuracy: 0.9558
 6300/25000 [======>.......................] - ETA: 11s - loss: 0.3081 - accuracy: 0.9556
 6400/25000 [======>.......................] - ETA: 11s - loss: 0.3078 - accuracy: 0.9558
 6500/25000 [======>.......................] - ETA: 11s - loss: 0.3080 - accuracy: 0.9557
 6600/25000 [======>.......................] - ETA: 11s - loss: 0.3084 - accuracy: 0.9553
 6700/25000 [=======>......................] - ETA: 11s - loss: 0.3083 - accuracy: 0.9554
 6800/25000 [=======>......................] - ETA: 11s - loss: 0.3078 - accuracy: 0.9557
 6900/25000 [=======>......................] - ETA: 11s - loss: 0.3079 - accuracy: 0.9557
 7000/25000 [=======>......................] - ETA: 11s - loss: 0.3079 - accuracy: 0.9556
 7100/25000 [=======>......................] - ETA: 11s - loss: 0.3071 - accuracy: 0.9562
 7200/25000 [=======>......................] - ETA: 11s - loss: 0.3070 - accuracy: 0.9564
 7300/25000 [=======>......................] - ETA: 11s - loss: 0.3070 - accuracy: 0.9564
 7400/25000 [=======>......................] - ETA: 11s - loss: 0.3072 - accuracy: 0.9562
 7500/25000 [========>.....................] - ETA: 11s - loss: 0.3075 - accuracy: 0.9560
 7600/25000 [========>.....................] - ETA: 10s - loss: 0.3074 - accuracy: 0.9559
 7700/25000 [========>.....................] - ETA: 10s - loss: 0.3075 - accuracy: 0.9558
 7800/25000 [========>.....................] - ETA: 10s - loss: 0.3071 - accuracy: 0.9562
 7900/25000 [========>.....................] - ETA: 10s - loss: 0.3074 - accuracy: 0.9559
 8000/25000 [========>.....................] - ETA: 10s - loss: 0.3072 - accuracy: 0.9561
 8100/25000 [========>.....................] - ETA: 10s - loss: 0.3067 - accuracy: 0.9564
 8200/25000 [========>.....................] - ETA: 10s - loss: 0.3065 - accuracy: 0.9566
 8300/25000 [========>.....................] - ETA: 10s - loss: 0.3073 - accuracy: 0.9559
 8400/25000 [=========>....................] - ETA: 10s - loss: 0.3073 - accuracy: 0.9558
 8500/25000 [=========>....................] - ETA: 10s - loss: 0.3069 - accuracy: 0.9561
 8600/25000 [=========>....................] - ETA: 10s - loss: 0.3072 - accuracy: 0.9559
 8700/25000 [=========>....................] - ETA: 10s - loss: 0.3075 - accuracy: 0.9557
 8800/25000 [=========>....................] - ETA: 10s - loss: 0.3075 - accuracy: 0.9557
 8900/25000 [=========>....................] - ETA: 10s - loss: 0.3075 - accuracy: 0.9556
 9000/25000 [=========>....................] - ETA: 10s - loss: 0.3074 - accuracy: 0.9557
 9100/25000 [=========>....................] - ETA: 10s - loss: 0.3072 - accuracy: 0.9558
 9200/25000 [==========>...................] - ETA: 10s - loss: 0.3071 - accuracy: 0.9559
 9300/25000 [==========>...................] - ETA: 10s - loss: 0.3070 - accuracy: 0.9559
 9400/25000 [==========>...................] - ETA: 10s - loss: 0.3068 - accuracy: 0.9562
 9500/25000 [==========>...................] - ETA: 9s - loss: 0.3068 - accuracy: 0.9561 
 9600/25000 [==========>...................] - ETA: 9s - loss: 0.3068 - accuracy: 0.9560
 9700/25000 [==========>...................] - ETA: 9s - loss: 0.3067 - accuracy: 0.9562
 9800/25000 [==========>...................] - ETA: 9s - loss: 0.3063 - accuracy: 0.9564
 9900/25000 [==========>...................] - ETA: 9s - loss: 0.3059 - accuracy: 0.9568
10000/25000 [===========>..................] - ETA: 9s - loss: 0.3060 - accuracy: 0.9567
10100/25000 [===========>..................] - ETA: 9s - loss: 0.3058 - accuracy: 0.9568
10200/25000 [===========>..................] - ETA: 9s - loss: 0.3056 - accuracy: 0.9569
10300/25000 [===========>..................] - ETA: 9s - loss: 0.3053 - accuracy: 0.9571
10400/25000 [===========>..................] - ETA: 9s - loss: 0.3058 - accuracy: 0.9566
10500/25000 [===========>..................] - ETA: 9s - loss: 0.3059 - accuracy: 0.9566
10600/25000 [===========>..................] - ETA: 9s - loss: 0.3061 - accuracy: 0.9564
10700/25000 [===========>..................] - ETA: 9s - loss: 0.3064 - accuracy: 0.9562
10800/25000 [===========>..................] - ETA: 9s - loss: 0.3064 - accuracy: 0.9561
10900/25000 [============>.................] - ETA: 9s - loss: 0.3064 - accuracy: 0.9561
11000/25000 [============>.................] - ETA: 9s - loss: 0.3069 - accuracy: 0.9556
11100/25000 [============>.................] - ETA: 8s - loss: 0.3065 - accuracy: 0.9559
11200/25000 [============>.................] - ETA: 8s - loss: 0.3064 - accuracy: 0.9560
11300/25000 [============>.................] - ETA: 8s - loss: 0.3059 - accuracy: 0.9564
11400/25000 [============>.................] - ETA: 8s - loss: 0.3060 - accuracy: 0.9563
11500/25000 [============>.................] - ETA: 8s - loss: 0.3056 - accuracy: 0.9566
11600/25000 [============>.................] - ETA: 8s - loss: 0.3056 - accuracy: 0.9566
11700/25000 [=============>................] - ETA: 8s - loss: 0.3056 - accuracy: 0.9566
11800/25000 [=============>................] - ETA: 8s - loss: 0.3052 - accuracy: 0.9569
11900/25000 [=============>................] - ETA: 8s - loss: 0.3051 - accuracy: 0.9569
12000/25000 [=============>................] - ETA: 8s - loss: 0.3052 - accuracy: 0.9568
12100/25000 [=============>................] - ETA: 8s - loss: 0.3052 - accuracy: 0.9568
12200/25000 [=============>................] - ETA: 8s - loss: 0.3052 - accuracy: 0.9566
12300/25000 [=============>................] - ETA: 8s - loss: 0.3050 - accuracy: 0.9567
12400/25000 [=============>................] - ETA: 8s - loss: 0.3049 - accuracy: 0.9568
12500/25000 [==============>...............] - ETA: 8s - loss: 0.3050 - accuracy: 0.9567
12600/25000 [==============>...............] - ETA: 7s - loss: 0.3051 - accuracy: 0.9566
12700/25000 [==============>...............] - ETA: 7s - loss: 0.3048 - accuracy: 0.9568
12800/25000 [==============>...............] - ETA: 7s - loss: 0.3049 - accuracy: 0.9566
12900/25000 [==============>...............] - ETA: 7s - loss: 0.3049 - accuracy: 0.9567
13000/25000 [==============>...............] - ETA: 7s - loss: 0.3049 - accuracy: 0.9566
13100/25000 [==============>...............] - ETA: 7s - loss: 0.3050 - accuracy: 0.9565
13200/25000 [==============>...............] - ETA: 7s - loss: 0.3049 - accuracy: 0.9565
13300/25000 [==============>...............] - ETA: 7s - loss: 0.3050 - accuracy: 0.9565
13400/25000 [===============>..............] - ETA: 7s - loss: 0.3048 - accuracy: 0.9566
13500/25000 [===============>..............] - ETA: 7s - loss: 0.3047 - accuracy: 0.9567
13600/25000 [===============>..............] - ETA: 7s - loss: 0.3047 - accuracy: 0.9567
13700/25000 [===============>..............] - ETA: 7s - loss: 0.3050 - accuracy: 0.9564
13800/25000 [===============>..............] - ETA: 7s - loss: 0.3050 - accuracy: 0.9563
13900/25000 [===============>..............] - ETA: 7s - loss: 0.3050 - accuracy: 0.9562
14000/25000 [===============>..............] - ETA: 7s - loss: 0.3052 - accuracy: 0.9560
14100/25000 [===============>..............] - ETA: 7s - loss: 0.3049 - accuracy: 0.9562
14200/25000 [================>.............] - ETA: 6s - loss: 0.3047 - accuracy: 0.9563
14300/25000 [================>.............] - ETA: 6s - loss: 0.3049 - accuracy: 0.9562
14400/25000 [================>.............] - ETA: 6s - loss: 0.3050 - accuracy: 0.9561
14500/25000 [================>.............] - ETA: 6s - loss: 0.3047 - accuracy: 0.9563
14600/25000 [================>.............] - ETA: 6s - loss: 0.3047 - accuracy: 0.9562
14700/25000 [================>.............] - ETA: 6s - loss: 0.3050 - accuracy: 0.9560
14800/25000 [================>.............] - ETA: 6s - loss: 0.3051 - accuracy: 0.9559
14900/25000 [================>.............] - ETA: 6s - loss: 0.3047 - accuracy: 0.9561
15000/25000 [=================>............] - ETA: 6s - loss: 0.3048 - accuracy: 0.9561
15100/25000 [=================>............] - ETA: 6s - loss: 0.3051 - accuracy: 0.9558
15200/25000 [=================>............] - ETA: 6s - loss: 0.3052 - accuracy: 0.9557
15300/25000 [=================>............] - ETA: 6s - loss: 0.3051 - accuracy: 0.9557
15400/25000 [=================>............] - ETA: 6s - loss: 0.3052 - accuracy: 0.9555
15500/25000 [=================>............] - ETA: 6s - loss: 0.3056 - accuracy: 0.9551
15600/25000 [=================>............] - ETA: 6s - loss: 0.3057 - accuracy: 0.9550
15700/25000 [=================>............] - ETA: 5s - loss: 0.3058 - accuracy: 0.9550
15800/25000 [=================>............] - ETA: 5s - loss: 0.3059 - accuracy: 0.9548
15900/25000 [==================>...........] - ETA: 5s - loss: 0.3059 - accuracy: 0.9547
16000/25000 [==================>...........] - ETA: 5s - loss: 0.3059 - accuracy: 0.9547
16100/25000 [==================>...........] - ETA: 5s - loss: 0.3060 - accuracy: 0.9546
16200/25000 [==================>...........] - ETA: 5s - loss: 0.3064 - accuracy: 0.9544
16300/25000 [==================>...........] - ETA: 5s - loss: 0.3061 - accuracy: 0.9546
16400/25000 [==================>...........] - ETA: 5s - loss: 0.3061 - accuracy: 0.9546
16500/25000 [==================>...........] - ETA: 5s - loss: 0.3058 - accuracy: 0.9547
16600/25000 [==================>...........] - ETA: 5s - loss: 0.3059 - accuracy: 0.9546
16700/25000 [===================>..........] - ETA: 5s - loss: 0.3056 - accuracy: 0.9548
16800/25000 [===================>..........] - ETA: 5s - loss: 0.3055 - accuracy: 0.9549
16900/25000 [===================>..........] - ETA: 5s - loss: 0.3054 - accuracy: 0.9549
17000/25000 [===================>..........] - ETA: 5s - loss: 0.3054 - accuracy: 0.9548
17100/25000 [===================>..........] - ETA: 5s - loss: 0.3052 - accuracy: 0.9550
17200/25000 [===================>..........] - ETA: 4s - loss: 0.3050 - accuracy: 0.9551
17300/25000 [===================>..........] - ETA: 4s - loss: 0.3050 - accuracy: 0.9550
17400/25000 [===================>..........] - ETA: 4s - loss: 0.3050 - accuracy: 0.9550
17500/25000 [====================>.........] - ETA: 4s - loss: 0.3049 - accuracy: 0.9551
17600/25000 [====================>.........] - ETA: 4s - loss: 0.3049 - accuracy: 0.9551
17700/25000 [====================>.........] - ETA: 4s - loss: 0.3048 - accuracy: 0.9551
17800/25000 [====================>.........] - ETA: 4s - loss: 0.3049 - accuracy: 0.9549
17900/25000 [====================>.........] - ETA: 4s - loss: 0.3049 - accuracy: 0.9550
18000/25000 [====================>.........] - ETA: 4s - loss: 0.3051 - accuracy: 0.9549
18100/25000 [====================>.........] - ETA: 4s - loss: 0.3049 - accuracy: 0.9550
18200/25000 [====================>.........] - ETA: 4s - loss: 0.3049 - accuracy: 0.9549
18300/25000 [====================>.........] - ETA: 4s - loss: 0.3051 - accuracy: 0.9548
18400/25000 [=====================>........] - ETA: 4s - loss: 0.3049 - accuracy: 0.9549
18500/25000 [=====================>........] - ETA: 4s - loss: 0.3048 - accuracy: 0.9549
18600/25000 [=====================>........] - ETA: 4s - loss: 0.3049 - accuracy: 0.9549
18700/25000 [=====================>........] - ETA: 4s - loss: 0.3049 - accuracy: 0.9548
18800/25000 [=====================>........] - ETA: 3s - loss: 0.3050 - accuracy: 0.9547
18900/25000 [=====================>........] - ETA: 3s - loss: 0.3049 - accuracy: 0.9547
19000/25000 [=====================>........] - ETA: 3s - loss: 0.3048 - accuracy: 0.9548
19100/25000 [=====================>........] - ETA: 3s - loss: 0.3046 - accuracy: 0.9550
19200/25000 [======================>.......] - ETA: 3s - loss: 0.3045 - accuracy: 0.9549
19300/25000 [======================>.......] - ETA: 3s - loss: 0.3051 - accuracy: 0.9545
19400/25000 [======================>.......] - ETA: 3s - loss: 0.3048 - accuracy: 0.9547
19500/25000 [======================>.......] - ETA: 3s - loss: 0.3045 - accuracy: 0.9549
19600/25000 [======================>.......] - ETA: 3s - loss: 0.3043 - accuracy: 0.9550
19700/25000 [======================>.......] - ETA: 3s - loss: 0.3044 - accuracy: 0.9549
19800/25000 [======================>.......] - ETA: 3s - loss: 0.3043 - accuracy: 0.9549
19900/25000 [======================>.......] - ETA: 3s - loss: 0.3042 - accuracy: 0.9550
20000/25000 [=======================>......] - ETA: 3s - loss: 0.3041 - accuracy: 0.9550
20100/25000 [=======================>......] - ETA: 3s - loss: 0.3042 - accuracy: 0.9550
20200/25000 [=======================>......] - ETA: 3s - loss: 0.3042 - accuracy: 0.9550
20300/25000 [=======================>......] - ETA: 2s - loss: 0.3043 - accuracy: 0.9549
20400/25000 [=======================>......] - ETA: 2s - loss: 0.3044 - accuracy: 0.9548
20500/25000 [=======================>......] - ETA: 2s - loss: 0.3044 - accuracy: 0.9548
20600/25000 [=======================>......] - ETA: 2s - loss: 0.3043 - accuracy: 0.9548
20700/25000 [=======================>......] - ETA: 2s - loss: 0.3045 - accuracy: 0.9546
20800/25000 [=======================>......] - ETA: 2s - loss: 0.3044 - accuracy: 0.9547
20900/25000 [========================>.....] - ETA: 2s - loss: 0.3043 - accuracy: 0.9548
21000/25000 [========================>.....] - ETA: 2s - loss: 0.3045 - accuracy: 0.9546
21100/25000 [========================>.....] - ETA: 2s - loss: 0.3045 - accuracy: 0.9545
21200/25000 [========================>.....] - ETA: 2s - loss: 0.3046 - accuracy: 0.9544
21300/25000 [========================>.....] - ETA: 2s - loss: 0.3049 - accuracy: 0.9541
21400/25000 [========================>.....] - ETA: 2s - loss: 0.3047 - accuracy: 0.9543
21500/25000 [========================>.....] - ETA: 2s - loss: 0.3047 - accuracy: 0.9542
21600/25000 [========================>.....] - ETA: 2s - loss: 0.3046 - accuracy: 0.9543
21700/25000 [=========================>....] - ETA: 2s - loss: 0.3046 - accuracy: 0.9542
21800/25000 [=========================>....] - ETA: 2s - loss: 0.3046 - accuracy: 0.9542
21900/25000 [=========================>....] - ETA: 1s - loss: 0.3045 - accuracy: 0.9543
22000/25000 [=========================>....] - ETA: 1s - loss: 0.3044 - accuracy: 0.9544
22100/25000 [=========================>....] - ETA: 1s - loss: 0.3044 - accuracy: 0.9543
22200/25000 [=========================>....] - ETA: 1s - loss: 0.3045 - accuracy: 0.9542
22300/25000 [=========================>....] - ETA: 1s - loss: 0.3045 - accuracy: 0.9541
22400/25000 [=========================>....] - ETA: 1s - loss: 0.3044 - accuracy: 0.9542
22500/25000 [==========================>...] - ETA: 1s - loss: 0.3044 - accuracy: 0.9542
22600/25000 [==========================>...] - ETA: 1s - loss: 0.3045 - accuracy: 0.9540
22700/25000 [==========================>...] - ETA: 1s - loss: 0.3045 - accuracy: 0.9540
22800/25000 [==========================>...] - ETA: 1s - loss: 0.3045 - accuracy: 0.9540
22900/25000 [==========================>...] - ETA: 1s - loss: 0.3045 - accuracy: 0.9540
23000/25000 [==========================>...] - ETA: 1s - loss: 0.3044 - accuracy: 0.9540
23100/25000 [==========================>...] - ETA: 1s - loss: 0.3043 - accuracy: 0.9541
23200/25000 [==========================>...] - ETA: 1s - loss: 0.3042 - accuracy: 0.9541
23300/25000 [==========================>...] - ETA: 1s - loss: 0.3041 - accuracy: 0.9541
23400/25000 [===========================>..] - ETA: 1s - loss: 0.3040 - accuracy: 0.9542
23500/25000 [===========================>..] - ETA: 0s - loss: 0.3041 - accuracy: 0.9541
23600/25000 [===========================>..] - ETA: 0s - loss: 0.3041 - accuracy: 0.9541
23700/25000 [===========================>..] - ETA: 0s - loss: 0.3045 - accuracy: 0.9538
23800/25000 [===========================>..] - ETA: 0s - loss: 0.3044 - accuracy: 0.9538
23900/25000 [===========================>..] - ETA: 0s - loss: 0.3044 - accuracy: 0.9537
24000/25000 [===========================>..] - ETA: 0s - loss: 0.3043 - accuracy: 0.9538
24100/25000 [===========================>..] - ETA: 0s - loss: 0.3044 - accuracy: 0.9537
24200/25000 [============================>.] - ETA: 0s - loss: 0.3043 - accuracy: 0.9537
24300/25000 [============================>.] - ETA: 0s - loss: 0.3042 - accuracy: 0.9538
24400/25000 [============================>.] - ETA: 0s - loss: 0.3042 - accuracy: 0.9538
24500/25000 [============================>.] - ETA: 0s - loss: 0.3040 - accuracy: 0.9539
24600/25000 [============================>.] - ETA: 0s - loss: 0.3039 - accuracy: 0.9540
24700/25000 [============================>.] - ETA: 0s - loss: 0.3040 - accuracy: 0.9538
24800/25000 [============================>.] - ETA: 0s - loss: 0.3040 - accuracy: 0.9538
24900/25000 [============================>.] - ETA: 0s - loss: 0.3043 - accuracy: 0.9536
25000/25000 [==============================] - 20s 809us/step - loss: 0.3044 - accuracy: 0.9534 - val_loss: 0.4116 - val_accuracy: 0.8612
Epoch 10/10

  100/25000 [..............................] - ETA: 15s - loss: 0.2698 - accuracy: 0.9700
  200/25000 [..............................] - ETA: 15s - loss: 0.2747 - accuracy: 0.9700
  300/25000 [..............................] - ETA: 15s - loss: 0.2892 - accuracy: 0.9567
  400/25000 [..............................] - ETA: 15s - loss: 0.2976 - accuracy: 0.9500
  500/25000 [..............................] - ETA: 15s - loss: 0.2946 - accuracy: 0.9540
  600/25000 [..............................] - ETA: 15s - loss: 0.2981 - accuracy: 0.9517
  700/25000 [..............................] - ETA: 15s - loss: 0.3012 - accuracy: 0.9471
  800/25000 [..............................] - ETA: 15s - loss: 0.3019 - accuracy: 0.9475
  900/25000 [>.............................] - ETA: 15s - loss: 0.2974 - accuracy: 0.9511
 1000/25000 [>.............................] - ETA: 15s - loss: 0.2937 - accuracy: 0.9540
 1100/25000 [>.............................] - ETA: 15s - loss: 0.2926 - accuracy: 0.9555
 1200/25000 [>.............................] - ETA: 15s - loss: 0.2890 - accuracy: 0.9583
 1300/25000 [>.............................] - ETA: 15s - loss: 0.2907 - accuracy: 0.9569
 1400/25000 [>.............................] - ETA: 15s - loss: 0.2905 - accuracy: 0.9571
 1500/25000 [>.............................] - ETA: 15s - loss: 0.2926 - accuracy: 0.9560
 1600/25000 [>.............................] - ETA: 14s - loss: 0.2900 - accuracy: 0.9581
 1700/25000 [=>............................] - ETA: 14s - loss: 0.2879 - accuracy: 0.9600
 1800/25000 [=>............................] - ETA: 14s - loss: 0.2891 - accuracy: 0.9589
 1900/25000 [=>............................] - ETA: 14s - loss: 0.2913 - accuracy: 0.9574
 2000/25000 [=>............................] - ETA: 14s - loss: 0.2895 - accuracy: 0.9585
 2100/25000 [=>............................] - ETA: 14s - loss: 0.2882 - accuracy: 0.9590
 2200/25000 [=>............................] - ETA: 14s - loss: 0.2882 - accuracy: 0.9586
 2300/25000 [=>............................] - ETA: 14s - loss: 0.2886 - accuracy: 0.9583
 2400/25000 [=>............................] - ETA: 14s - loss: 0.2871 - accuracy: 0.9596
 2500/25000 [==>...........................] - ETA: 14s - loss: 0.2865 - accuracy: 0.9600
 2600/25000 [==>...........................] - ETA: 14s - loss: 0.2858 - accuracy: 0.9604
 2700/25000 [==>...........................] - ETA: 14s - loss: 0.2862 - accuracy: 0.9600
 2800/25000 [==>...........................] - ETA: 14s - loss: 0.2869 - accuracy: 0.9596
 2900/25000 [==>...........................] - ETA: 14s - loss: 0.2873 - accuracy: 0.9593
 3000/25000 [==>...........................] - ETA: 14s - loss: 0.2906 - accuracy: 0.9563
 3100/25000 [==>...........................] - ETA: 13s - loss: 0.2903 - accuracy: 0.9561
 3200/25000 [==>...........................] - ETA: 13s - loss: 0.2902 - accuracy: 0.9559
 3300/25000 [==>...........................] - ETA: 13s - loss: 0.2901 - accuracy: 0.9561
 3400/25000 [===>..........................] - ETA: 13s - loss: 0.2899 - accuracy: 0.9562
 3500/25000 [===>..........................] - ETA: 13s - loss: 0.2886 - accuracy: 0.9571
 3600/25000 [===>..........................] - ETA: 13s - loss: 0.2878 - accuracy: 0.9578
 3700/25000 [===>..........................] - ETA: 13s - loss: 0.2866 - accuracy: 0.9586
 3800/25000 [===>..........................] - ETA: 13s - loss: 0.2867 - accuracy: 0.9587
 3900/25000 [===>..........................] - ETA: 13s - loss: 0.2865 - accuracy: 0.9590
 4000/25000 [===>..........................] - ETA: 13s - loss: 0.2870 - accuracy: 0.9585
 4100/25000 [===>..........................] - ETA: 13s - loss: 0.2875 - accuracy: 0.9580
 4200/25000 [====>.........................] - ETA: 13s - loss: 0.2865 - accuracy: 0.9588
 4300/25000 [====>.........................] - ETA: 13s - loss: 0.2865 - accuracy: 0.9588
 4400/25000 [====>.........................] - ETA: 13s - loss: 0.2869 - accuracy: 0.9584
 4500/25000 [====>.........................] - ETA: 13s - loss: 0.2866 - accuracy: 0.9587
 4600/25000 [====>.........................] - ETA: 13s - loss: 0.2876 - accuracy: 0.9578
 4700/25000 [====>.........................] - ETA: 13s - loss: 0.2879 - accuracy: 0.9577
 4800/25000 [====>.........................] - ETA: 12s - loss: 0.2877 - accuracy: 0.9577
 4900/25000 [====>.........................] - ETA: 12s - loss: 0.2878 - accuracy: 0.9578
 5000/25000 [=====>........................] - ETA: 12s - loss: 0.2868 - accuracy: 0.9584
 5100/25000 [=====>........................] - ETA: 12s - loss: 0.2874 - accuracy: 0.9580
 5200/25000 [=====>........................] - ETA: 12s - loss: 0.2869 - accuracy: 0.9585
 5300/25000 [=====>........................] - ETA: 12s - loss: 0.2865 - accuracy: 0.9587
 5400/25000 [=====>........................] - ETA: 12s - loss: 0.2871 - accuracy: 0.9583
 5500/25000 [=====>........................] - ETA: 12s - loss: 0.2870 - accuracy: 0.9582
 5600/25000 [=====>........................] - ETA: 12s - loss: 0.2869 - accuracy: 0.9582
 5700/25000 [=====>........................] - ETA: 12s - loss: 0.2870 - accuracy: 0.9582
 5800/25000 [=====>........................] - ETA: 12s - loss: 0.2867 - accuracy: 0.9584
 5900/25000 [======>.......................] - ETA: 12s - loss: 0.2871 - accuracy: 0.9581
 6000/25000 [======>.......................] - ETA: 12s - loss: 0.2883 - accuracy: 0.9572
 6100/25000 [======>.......................] - ETA: 12s - loss: 0.2882 - accuracy: 0.9572
 6200/25000 [======>.......................] - ETA: 12s - loss: 0.2879 - accuracy: 0.9574
 6300/25000 [======>.......................] - ETA: 12s - loss: 0.2884 - accuracy: 0.9568
 6400/25000 [======>.......................] - ETA: 11s - loss: 0.2887 - accuracy: 0.9566
 6500/25000 [======>.......................] - ETA: 11s - loss: 0.2883 - accuracy: 0.9568
 6600/25000 [======>.......................] - ETA: 11s - loss: 0.2881 - accuracy: 0.9568
 6700/25000 [=======>......................] - ETA: 11s - loss: 0.2882 - accuracy: 0.9569
 6800/25000 [=======>......................] - ETA: 11s - loss: 0.2884 - accuracy: 0.9568
 6900/25000 [=======>......................] - ETA: 11s - loss: 0.2882 - accuracy: 0.9570
 7000/25000 [=======>......................] - ETA: 11s - loss: 0.2881 - accuracy: 0.9570
 7100/25000 [=======>......................] - ETA: 11s - loss: 0.2889 - accuracy: 0.9565
 7200/25000 [=======>......................] - ETA: 11s - loss: 0.2889 - accuracy: 0.9564
 7300/25000 [=======>......................] - ETA: 11s - loss: 0.2890 - accuracy: 0.9563
 7400/25000 [=======>......................] - ETA: 11s - loss: 0.2893 - accuracy: 0.9561
 7500/25000 [========>.....................] - ETA: 11s - loss: 0.2893 - accuracy: 0.9560
 7600/25000 [========>.....................] - ETA: 11s - loss: 0.2889 - accuracy: 0.9562
 7700/25000 [========>.....................] - ETA: 11s - loss: 0.2888 - accuracy: 0.9562
 7800/25000 [========>.....................] - ETA: 11s - loss: 0.2891 - accuracy: 0.9560
 7900/25000 [========>.....................] - ETA: 11s - loss: 0.2890 - accuracy: 0.9561
 8000/25000 [========>.....................] - ETA: 10s - loss: 0.2887 - accuracy: 0.9563
 8100/25000 [========>.....................] - ETA: 10s - loss: 0.2886 - accuracy: 0.9562
 8200/25000 [========>.....................] - ETA: 10s - loss: 0.2885 - accuracy: 0.9562
 8300/25000 [========>.....................] - ETA: 10s - loss: 0.2885 - accuracy: 0.9561
 8400/25000 [=========>....................] - ETA: 10s - loss: 0.2884 - accuracy: 0.9562
 8500/25000 [=========>....................] - ETA: 10s - loss: 0.2889 - accuracy: 0.9558
 8600/25000 [=========>....................] - ETA: 10s - loss: 0.2890 - accuracy: 0.9557
 8700/25000 [=========>....................] - ETA: 10s - loss: 0.2886 - accuracy: 0.9560
 8800/25000 [=========>....................] - ETA: 10s - loss: 0.2891 - accuracy: 0.9557
 8900/25000 [=========>....................] - ETA: 10s - loss: 0.2895 - accuracy: 0.9553
 9000/25000 [=========>....................] - ETA: 10s - loss: 0.2893 - accuracy: 0.9553
 9100/25000 [=========>....................] - ETA: 10s - loss: 0.2894 - accuracy: 0.9553
 9200/25000 [==========>...................] - ETA: 10s - loss: 0.2895 - accuracy: 0.9551
 9300/25000 [==========>...................] - ETA: 10s - loss: 0.2896 - accuracy: 0.9551
 9400/25000 [==========>...................] - ETA: 10s - loss: 0.2892 - accuracy: 0.9553
 9500/25000 [==========>...................] - ETA: 9s - loss: 0.2889 - accuracy: 0.9556 
 9600/25000 [==========>...................] - ETA: 9s - loss: 0.2886 - accuracy: 0.9558
 9700/25000 [==========>...................] - ETA: 9s - loss: 0.2885 - accuracy: 0.9559
 9800/25000 [==========>...................] - ETA: 9s - loss: 0.2885 - accuracy: 0.9559
 9900/25000 [==========>...................] - ETA: 9s - loss: 0.2885 - accuracy: 0.9559
10000/25000 [===========>..................] - ETA: 9s - loss: 0.2889 - accuracy: 0.9555
10100/25000 [===========>..................] - ETA: 9s - loss: 0.2892 - accuracy: 0.9551
10200/25000 [===========>..................] - ETA: 9s - loss: 0.2894 - accuracy: 0.9550
10300/25000 [===========>..................] - ETA: 9s - loss: 0.2894 - accuracy: 0.9550
10400/25000 [===========>..................] - ETA: 9s - loss: 0.2891 - accuracy: 0.9552
10500/25000 [===========>..................] - ETA: 9s - loss: 0.2893 - accuracy: 0.9550
10600/25000 [===========>..................] - ETA: 9s - loss: 0.2895 - accuracy: 0.9549
10700/25000 [===========>..................] - ETA: 9s - loss: 0.2895 - accuracy: 0.9550
10800/25000 [===========>..................] - ETA: 9s - loss: 0.2899 - accuracy: 0.9546
10900/25000 [============>.................] - ETA: 9s - loss: 0.2899 - accuracy: 0.9546
11000/25000 [============>.................] - ETA: 9s - loss: 0.2896 - accuracy: 0.9547
11100/25000 [============>.................] - ETA: 9s - loss: 0.2893 - accuracy: 0.9550
11200/25000 [============>.................] - ETA: 8s - loss: 0.2893 - accuracy: 0.9549
11300/25000 [============>.................] - ETA: 8s - loss: 0.2888 - accuracy: 0.9553
11400/25000 [============>.................] - ETA: 8s - loss: 0.2887 - accuracy: 0.9554
11500/25000 [============>.................] - ETA: 8s - loss: 0.2885 - accuracy: 0.9555
11600/25000 [============>.................] - ETA: 8s - loss: 0.2885 - accuracy: 0.9555
11700/25000 [=============>................] - ETA: 8s - loss: 0.2885 - accuracy: 0.9555
11800/25000 [=============>................] - ETA: 8s - loss: 0.2881 - accuracy: 0.9558
11900/25000 [=============>................] - ETA: 8s - loss: 0.2881 - accuracy: 0.9558
12000/25000 [=============>................] - ETA: 8s - loss: 0.2880 - accuracy: 0.9557
12100/25000 [=============>................] - ETA: 8s - loss: 0.2886 - accuracy: 0.9552
12200/25000 [=============>................] - ETA: 8s - loss: 0.2885 - accuracy: 0.9552
12300/25000 [=============>................] - ETA: 8s - loss: 0.2887 - accuracy: 0.9550
12400/25000 [=============>................] - ETA: 8s - loss: 0.2885 - accuracy: 0.9552
12500/25000 [==============>...............] - ETA: 8s - loss: 0.2881 - accuracy: 0.9555
12600/25000 [==============>...............] - ETA: 8s - loss: 0.2881 - accuracy: 0.9555
12700/25000 [==============>...............] - ETA: 7s - loss: 0.2878 - accuracy: 0.9557
12800/25000 [==============>...............] - ETA: 7s - loss: 0.2875 - accuracy: 0.9559
12900/25000 [==============>...............] - ETA: 7s - loss: 0.2872 - accuracy: 0.9561
13000/25000 [==============>...............] - ETA: 7s - loss: 0.2871 - accuracy: 0.9562
13100/25000 [==============>...............] - ETA: 7s - loss: 0.2872 - accuracy: 0.9560
13200/25000 [==============>...............] - ETA: 7s - loss: 0.2872 - accuracy: 0.9560
13300/25000 [==============>...............] - ETA: 7s - loss: 0.2874 - accuracy: 0.9559
13400/25000 [===============>..............] - ETA: 7s - loss: 0.2870 - accuracy: 0.9561
13500/25000 [===============>..............] - ETA: 7s - loss: 0.2869 - accuracy: 0.9561
13600/25000 [===============>..............] - ETA: 7s - loss: 0.2867 - accuracy: 0.9563
13700/25000 [===============>..............] - ETA: 7s - loss: 0.2866 - accuracy: 0.9564
13800/25000 [===============>..............] - ETA: 7s - loss: 0.2864 - accuracy: 0.9565
13900/25000 [===============>..............] - ETA: 7s - loss: 0.2866 - accuracy: 0.9563
14000/25000 [===============>..............] - ETA: 7s - loss: 0.2865 - accuracy: 0.9564
14100/25000 [===============>..............] - ETA: 7s - loss: 0.2864 - accuracy: 0.9565
14200/25000 [================>.............] - ETA: 7s - loss: 0.2863 - accuracy: 0.9565
14300/25000 [================>.............] - ETA: 6s - loss: 0.2860 - accuracy: 0.9568
14400/25000 [================>.............] - ETA: 6s - loss: 0.2860 - accuracy: 0.9567
14500/25000 [================>.............] - ETA: 6s - loss: 0.2860 - accuracy: 0.9568
14600/25000 [================>.............] - ETA: 6s - loss: 0.2860 - accuracy: 0.9567
14700/25000 [================>.............] - ETA: 6s - loss: 0.2858 - accuracy: 0.9568
14800/25000 [================>.............] - ETA: 6s - loss: 0.2857 - accuracy: 0.9568
14900/25000 [================>.............] - ETA: 6s - loss: 0.2857 - accuracy: 0.9568
15000/25000 [=================>............] - ETA: 6s - loss: 0.2856 - accuracy: 0.9569
15100/25000 [=================>............] - ETA: 6s - loss: 0.2857 - accuracy: 0.9568
15200/25000 [=================>............] - ETA: 6s - loss: 0.2857 - accuracy: 0.9567
15300/25000 [=================>............] - ETA: 6s - loss: 0.2859 - accuracy: 0.9566
15400/25000 [=================>............] - ETA: 6s - loss: 0.2858 - accuracy: 0.9567
15500/25000 [=================>............] - ETA: 6s - loss: 0.2856 - accuracy: 0.9568
15600/25000 [=================>............] - ETA: 6s - loss: 0.2857 - accuracy: 0.9567
15700/25000 [=================>............] - ETA: 6s - loss: 0.2854 - accuracy: 0.9569
15800/25000 [=================>............] - ETA: 5s - loss: 0.2854 - accuracy: 0.9569
15900/25000 [==================>...........] - ETA: 5s - loss: 0.2857 - accuracy: 0.9567
16000/25000 [==================>...........] - ETA: 5s - loss: 0.2857 - accuracy: 0.9567
16100/25000 [==================>...........] - ETA: 5s - loss: 0.2858 - accuracy: 0.9566
16200/25000 [==================>...........] - ETA: 5s - loss: 0.2858 - accuracy: 0.9565
16300/25000 [==================>...........] - ETA: 5s - loss: 0.2857 - accuracy: 0.9566
16400/25000 [==================>...........] - ETA: 5s - loss: 0.2855 - accuracy: 0.9568
16500/25000 [==================>...........] - ETA: 5s - loss: 0.2856 - accuracy: 0.9567
16600/25000 [==================>...........] - ETA: 5s - loss: 0.2855 - accuracy: 0.9568
16700/25000 [===================>..........] - ETA: 5s - loss: 0.2853 - accuracy: 0.9569
16800/25000 [===================>..........] - ETA: 5s - loss: 0.2854 - accuracy: 0.9568
16900/25000 [===================>..........] - ETA: 5s - loss: 0.2854 - accuracy: 0.9567
17000/25000 [===================>..........] - ETA: 5s - loss: 0.2856 - accuracy: 0.9566
17100/25000 [===================>..........] - ETA: 5s - loss: 0.2855 - accuracy: 0.9565
17200/25000 [===================>..........] - ETA: 5s - loss: 0.2855 - accuracy: 0.9565
17300/25000 [===================>..........] - ETA: 4s - loss: 0.2857 - accuracy: 0.9564
17400/25000 [===================>..........] - ETA: 4s - loss: 0.2856 - accuracy: 0.9564
17500/25000 [====================>.........] - ETA: 4s - loss: 0.2857 - accuracy: 0.9564
17600/25000 [====================>.........] - ETA: 4s - loss: 0.2857 - accuracy: 0.9564
17700/25000 [====================>.........] - ETA: 4s - loss: 0.2857 - accuracy: 0.9564
17800/25000 [====================>.........] - ETA: 4s - loss: 0.2856 - accuracy: 0.9564
17900/25000 [====================>.........] - ETA: 4s - loss: 0.2855 - accuracy: 0.9565
18000/25000 [====================>.........] - ETA: 4s - loss: 0.2855 - accuracy: 0.9564
18100/25000 [====================>.........] - ETA: 4s - loss: 0.2856 - accuracy: 0.9563
18200/25000 [====================>.........] - ETA: 4s - loss: 0.2858 - accuracy: 0.9562
18300/25000 [====================>.........] - ETA: 4s - loss: 0.2857 - accuracy: 0.9562
18400/25000 [=====================>........] - ETA: 4s - loss: 0.2857 - accuracy: 0.9561
18500/25000 [=====================>........] - ETA: 4s - loss: 0.2857 - accuracy: 0.9561
18600/25000 [=====================>........] - ETA: 4s - loss: 0.2857 - accuracy: 0.9561
18700/25000 [=====================>........] - ETA: 4s - loss: 0.2858 - accuracy: 0.9560
18800/25000 [=====================>........] - ETA: 3s - loss: 0.2858 - accuracy: 0.9560
18900/25000 [=====================>........] - ETA: 3s - loss: 0.2858 - accuracy: 0.9560
19000/25000 [=====================>........] - ETA: 3s - loss: 0.2856 - accuracy: 0.9561
19100/25000 [=====================>........] - ETA: 3s - loss: 0.2855 - accuracy: 0.9561
19200/25000 [======================>.......] - ETA: 3s - loss: 0.2854 - accuracy: 0.9562
19300/25000 [======================>.......] - ETA: 3s - loss: 0.2854 - accuracy: 0.9562
19400/25000 [======================>.......] - ETA: 3s - loss: 0.2853 - accuracy: 0.9562
19500/25000 [======================>.......] - ETA: 3s - loss: 0.2855 - accuracy: 0.9560
19600/25000 [======================>.......] - ETA: 3s - loss: 0.2855 - accuracy: 0.9560
19700/25000 [======================>.......] - ETA: 3s - loss: 0.2854 - accuracy: 0.9560
19800/25000 [======================>.......] - ETA: 3s - loss: 0.2854 - accuracy: 0.9560
19900/25000 [======================>.......] - ETA: 3s - loss: 0.2853 - accuracy: 0.9561
20000/25000 [=======================>......] - ETA: 3s - loss: 0.2854 - accuracy: 0.9560
20100/25000 [=======================>......] - ETA: 3s - loss: 0.2854 - accuracy: 0.9560
20200/25000 [=======================>......] - ETA: 3s - loss: 0.2853 - accuracy: 0.9560
20300/25000 [=======================>......] - ETA: 3s - loss: 0.2852 - accuracy: 0.9561
20400/25000 [=======================>......] - ETA: 2s - loss: 0.2852 - accuracy: 0.9560
20500/25000 [=======================>......] - ETA: 2s - loss: 0.2851 - accuracy: 0.9561
20600/25000 [=======================>......] - ETA: 2s - loss: 0.2849 - accuracy: 0.9563
20700/25000 [=======================>......] - ETA: 2s - loss: 0.2847 - accuracy: 0.9563
20800/25000 [=======================>......] - ETA: 2s - loss: 0.2847 - accuracy: 0.9563
20900/25000 [========================>.....] - ETA: 2s - loss: 0.2847 - accuracy: 0.9563
21000/25000 [========================>.....] - ETA: 2s - loss: 0.2848 - accuracy: 0.9562
21100/25000 [========================>.....] - ETA: 2s - loss: 0.2846 - accuracy: 0.9563
21200/25000 [========================>.....] - ETA: 2s - loss: 0.2847 - accuracy: 0.9562
21300/25000 [========================>.....] - ETA: 2s - loss: 0.2848 - accuracy: 0.9561
21400/25000 [========================>.....] - ETA: 2s - loss: 0.2847 - accuracy: 0.9562
21500/25000 [========================>.....] - ETA: 2s - loss: 0.2847 - accuracy: 0.9561
21600/25000 [========================>.....] - ETA: 2s - loss: 0.2847 - accuracy: 0.9561
21700/25000 [=========================>....] - ETA: 2s - loss: 0.2846 - accuracy: 0.9562
21800/25000 [=========================>....] - ETA: 2s - loss: 0.2846 - accuracy: 0.9562
21900/25000 [=========================>....] - ETA: 2s - loss: 0.2845 - accuracy: 0.9563
22000/25000 [=========================>....] - ETA: 1s - loss: 0.2844 - accuracy: 0.9563
22100/25000 [=========================>....] - ETA: 1s - loss: 0.2844 - accuracy: 0.9563
22200/25000 [=========================>....] - ETA: 1s - loss: 0.2842 - accuracy: 0.9565
22300/25000 [=========================>....] - ETA: 1s - loss: 0.2842 - accuracy: 0.9564
22400/25000 [=========================>....] - ETA: 1s - loss: 0.2841 - accuracy: 0.9565
22500/25000 [==========================>...] - ETA: 1s - loss: 0.2840 - accuracy: 0.9565
22600/25000 [==========================>...] - ETA: 1s - loss: 0.2842 - accuracy: 0.9563
22700/25000 [==========================>...] - ETA: 1s - loss: 0.2841 - accuracy: 0.9563
22800/25000 [==========================>...] - ETA: 1s - loss: 0.2839 - accuracy: 0.9564
22900/25000 [==========================>...] - ETA: 1s - loss: 0.2841 - accuracy: 0.9562
23000/25000 [==========================>...] - ETA: 1s - loss: 0.2843 - accuracy: 0.9560
23100/25000 [==========================>...] - ETA: 1s - loss: 0.2844 - accuracy: 0.9559
23200/25000 [==========================>...] - ETA: 1s - loss: 0.2844 - accuracy: 0.9559
23300/25000 [==========================>...] - ETA: 1s - loss: 0.2847 - accuracy: 0.9557
23400/25000 [===========================>..] - ETA: 1s - loss: 0.2848 - accuracy: 0.9556
23500/25000 [===========================>..] - ETA: 0s - loss: 0.2846 - accuracy: 0.9557
23600/25000 [===========================>..] - ETA: 0s - loss: 0.2844 - accuracy: 0.9559
23700/25000 [===========================>..] - ETA: 0s - loss: 0.2844 - accuracy: 0.9559
23800/25000 [===========================>..] - ETA: 0s - loss: 0.2843 - accuracy: 0.9559
23900/25000 [===========================>..] - ETA: 0s - loss: 0.2844 - accuracy: 0.9559
24000/25000 [===========================>..] - ETA: 0s - loss: 0.2842 - accuracy: 0.9559
24100/25000 [===========================>..] - ETA: 0s - loss: 0.2842 - accuracy: 0.9559
24200/25000 [============================>.] - ETA: 0s - loss: 0.2840 - accuracy: 0.9561
24300/25000 [============================>.] - ETA: 0s - loss: 0.2840 - accuracy: 0.9560
24400/25000 [============================>.] - ETA: 0s - loss: 0.2840 - accuracy: 0.9561
24500/25000 [============================>.] - ETA: 0s - loss: 0.2839 - accuracy: 0.9561
24600/25000 [============================>.] - ETA: 0s - loss: 0.2840 - accuracy: 0.9560
24700/25000 [============================>.] - ETA: 0s - loss: 0.2839 - accuracy: 0.9560
24800/25000 [============================>.] - ETA: 0s - loss: 0.2840 - accuracy: 0.9559
24900/25000 [============================>.] - ETA: 0s - loss: 0.2841 - accuracy: 0.9558
25000/25000 [==============================] - 20s 817us/step - loss: 0.2845 - accuracy: 0.9555 - val_loss: 0.4190 - val_accuracy: 0.8530
	=====> Test the model: model.predict()
	Dataset: IMDB_REVIEWS
	Algorithm: Deep Learning using Keras 2 (KerasDL2)
	Training Loss: 0.2784
	Training accuracy score: 95.44%
	Test Loss: 0.4190
	Test Accuracy: 85.30%
	Training Time: 200.2824
	Test Time: 6.5434




FINAL CLASSIFICATION TABLE:

| ID | Dataset | Algorithm | Loss | Training accuracy score (%) | Test accuracy score (%) | Training time (seconds) | Test time (seconds) |
| --- | ------- | --------- | ---- | --------------------------- | ----------------------- | ----------------------- | ------------------- |
| 1 | TWENTY_NEWS_GROUPS | Deep Learning using Keras 1 (KerasDL1) | 0.0301 | 99.42 | 96.56 | 95.1296 | 4.9434 |
| 2 | IMDB_REVIEWS | Deep Learning using Keras 1 (KerasDL1) | 0.0003 | 100.00 | 82.94 | 163.8357 | 3.2235 |
| 3 | TWENTY_NEWS_GROUPS | Deep Learning using Keras 2 (KerasDL2) | 0.2542 | 94.96 | 94.96 | 85.0631 | 2.8639 |
| 4 | IMDB_REVIEWS | Deep Learning using Keras 2 (KerasDL2) | 0.2784 | 95.44 | 85.30 | 200.2824 | 6.5434 |

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
Program finished. It took 671.1746361255646 seconds

Process finished with exit code 0
```