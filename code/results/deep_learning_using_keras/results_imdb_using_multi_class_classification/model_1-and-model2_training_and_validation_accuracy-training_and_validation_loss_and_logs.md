## Two Deep Learning approaches using Keras: Loss, Training Accuracy, Test Accuracy, Training Time, Test Time

| ID | Dataset | Algorithm | Loss | Training accuracy score (%) | Test accuracy score (%) | Training time (seconds) | Test time (seconds) |
| --- | ------- | --------- | ---- | --------------------------- | ----------------------- | ----------------------- | ------------------- |
| 1 | IMDB_REVIEWS (Multi-class classification) | Deep Learning using Keras 1 (KerasDL1) | 0.0103 | 99.94 | 87.07 | 178.0648 | 2.9642 |
| 2 | IMDB_REVIEWS (Multi-class classification) | Deep Learning using Keras 2 (KerasDL2) | 0.3497 | 88.63 | 88.58 | 201.2245 | 6.4431 |

### Deep Learning using Keras 1 (KerasDL1)

![IMDB_REVIEWS: Loss, Training Accuracy, Test Accuracy, Training Time, Test Time](https://github.com/ramonfigueiredopessoa/comp551-2020-p2_classification_of_textual_data/blob/master/code/results/deep_learning_using_keras/results_imdb_using_multi_class_classification/KerasDL1_IMDB_REVIEWS_training_and_validation_accuracy_and_Loss.png)


### Learning using Keras 1 (KerasDL1)

![IMDB_REVIEWS: Loss, Training Accuracy, Test Accuracy, Training Time, Test Time](https://github.com/ramonfigueiredopessoa/comp551-2020-p2_classification_of_textual_data/blob/master/code/results/deep_learning_using_keras/results_imdb_using_multi_class_classification/KerasDL2_IMDB_REVIEWS_training_and_validation_accuracy_and_Loss.png)


#### Computer settings used to run

* Operating system: Ubuntu 18.04.4 LTS (64-bit)
* Processor: Intel® Core™ i7-7700 CPU @ 3.60GHz × 8 
* Memory: 32 GB

#### All logs 

```
/home/ets-crchum/virtual_envs/comp551_p2/bin/python /home/ets-crchum/github/comp551-2020-p2_classification_of_textual_data/code/main.py -dl -d IMDB_REVIEWS --use_imdb_multi_class_labels
Using TensorFlow backend.
2020-03-09 21:09:30.046239: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libnvinfer.so.6'; dlerror: libnvinfer.so.6: cannot open shared object file: No such file or directory
2020-03-09 21:09:30.046289: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libnvinfer_plugin.so.6'; dlerror: libnvinfer_plugin.so.6: cannot open shared object file: No such file or directory
2020-03-09 21:09:30.046294: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:30] Cannot dlopen some TensorRT libraries. If you would like to use Nvidia GPU with TensorRT, please make sure the missing libraries mentioned above are installed properly.
[nltk_data] Downloading package wordnet to /home/ets-
[nltk_data]     crchum/nltk_data...
[nltk_data]   Package wordnet is already up-to-date!
03/09/2020 09:09:30 PM - INFO - Program started...
03/09/2020 09:09:30 PM - INFO - Program started...
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
	Dataset = IMDB_REVIEWS
	ML algorithm list (If ml_algorithm_list is not provided, all ML algorithms will be executed) = None
	Use classifiers with default parameters. Default: False = Use classifiers with best parameters found using grid search. False
	Read dataset without shuffle data = False
	The number of CPUs to use to do the computation. If the provided number is negative or greater than the number of available CPUs, the system will use all the available CPUs. Default: -1 (-1 == all CPUs) = -1
	Run cross validation. Default: False = False
	Number of cross validation folds. Default: 5 = 5
	Use just the miniproject classifiers (1. LogisticRegression, 2. DecisionTreeClassifier, 3. LinearSVC, 4. AdaBoostClassifier, 5. RandomForestClassifier) =  False
	TWENTY_NEWS_GROUPS dataset using some categories (alt.atheism, talk.religion.misc, comp.graphics, sci.space) = False
	Do not remove newsgroup information that is easily overfit (headers, footers, quotes) = False
	Use IMDB multi-class labels (review score: 1, 2, 3, 4, 7, 8, 9, 10). If --use_imdb_multi_class_labels is False, the system uses binary classification (0 = neg and 1 = pos). Default: False = True
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

Loading IMDB_REVIEWS dataset:

===> Reading files from /home/ets-crchum/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/train/neg

===> Reading files from /home/ets-crchum/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/train/pos

===> Reading files from /home/ets-crchum/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/test/neg

===> Reading files from /home/ets-crchum/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/test/pos
data loaded
25000 documents - 33.133MB (training set)
25000 documents - 32.351MB (test set)

Extracting features from the training data using a vectorizer
done in 2.949917s at 11.232MB/s
n_samples: 25000, n_features: 74170

Extracting features from the test data using the same vectorizer
done in 3.026076s at 10.691MB/s
n_samples: 25000, n_features: 74170

================================================================================
KERAS DEEP LEARNING MODEL
Using layers:
	==> Dense(10, input_dim=input_dim, activation='relu')
	==> Dense(7, activation='sigmoid')
Compile option:
	==> model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
2020-03-09 21:09:37.988238: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
2020-03-09 21:09:38.010372: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-03-09 21:09:38.010947: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1555] Found device 0 with properties: 
pciBusID: 0000:01:00.0 name: GeForce GTX 1080 computeCapability: 6.1
coreClock: 1.7335GHz coreCount: 20 deviceMemorySize: 7.93GiB deviceMemoryBandwidth: 298.32GiB/s
2020-03-09 21:09:38.011008: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcudart.so.10.1'; dlerror: libcudart.so.10.1: cannot open shared object file: No such file or directory
2020-03-09 21:09:38.011050: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcublas.so.10'; dlerror: libcublas.so.10: cannot open shared object file: No such file or directory
2020-03-09 21:09:38.011089: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcufft.so.10'; dlerror: libcufft.so.10: cannot open shared object file: No such file or directory
2020-03-09 21:09:38.011126: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcurand.so.10'; dlerror: libcurand.so.10: cannot open shared object file: No such file or directory
2020-03-09 21:09:38.011163: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcusolver.so.10'; dlerror: libcusolver.so.10: cannot open shared object file: No such file or directory
2020-03-09 21:09:38.011200: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcusparse.so.10'; dlerror: libcusparse.so.10: cannot open shared object file: No such file or directory
2020-03-09 21:09:38.013140: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2020-03-09 21:09:38.013150: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1592] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
Skipping registering GPU devices...
2020-03-09 21:09:38.013329: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-03-09 21:09:38.035691: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 3600000000 Hz
2020-03-09 21:09:38.036178: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x4d356d0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-03-09 21:09:38.036199: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2020-03-09 21:09:38.103879: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-03-09 21:09:38.104509: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x4cc8b20 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2020-03-09 21:09:38.104520: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): GeForce GTX 1080, Compute Capability 6.1
2020-03-09 21:09:38.104619: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1096] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-03-09 21:09:38.104625: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1102]      
	Dataset: IMDB_REVIEWS
	Algorithm: Deep Learning using Keras 1 (KerasDL1)
	Training Loss: 0.0103
	Training accuracy score: 99.94%
	Test Accuracy: 87.07%
	Test Loss: 0.5733
	Training Time: 178.0648
	Test Time: 2.9642


03/09/2020 09:12:42 PM - INFO - Program started...
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
	It took 25.725462913513184 seconds


Applying NLTK feature extraction: X_test
	==> Replace unused UNICODE characters. 
	==> Text in lower case. 
	==> Apply Lemmatize using WordNet's built-in morphy function. Returns the input word unchanged if it cannot be found in WordNet 
	==> Remove English stop words
	Done: NLTK feature extraction (replace unused UNICODE characters, text in lower case, apply Lemmatize using WordNet, remove English stop words)  finished!
	It took 24.440382719039917 seconds

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

  100/25000 [..............................] - ETA: 2:15 - loss: 0.6625 - accuracy: 0.6686
  200/25000 [..............................] - ETA: 1:15 - loss: 0.6605 - accuracy: 0.6807
  300/25000 [..............................] - ETA: 55s - loss: 0.6604 - accuracy: 0.6781 
  400/25000 [..............................] - ETA: 44s - loss: 0.6598 - accuracy: 0.6796
  500/25000 [..............................] - ETA: 38s - loss: 0.6595 - accuracy: 0.6783
  600/25000 [..............................] - ETA: 34s - loss: 0.6584 - accuracy: 0.6814
  700/25000 [..............................] - ETA: 31s - loss: 0.6582 - accuracy: 0.6808
  800/25000 [..............................] - ETA: 29s - loss: 0.6577 - accuracy: 0.6809
  900/25000 [>.............................] - ETA: 28s - loss: 0.6573 - accuracy: 0.6802
 1000/25000 [>.............................] - ETA: 26s - loss: 0.6564 - accuracy: 0.6826
 1100/25000 [>.............................] - ETA: 25s - loss: 0.6561 - accuracy: 0.6819
 1200/25000 [>.............................] - ETA: 24s - loss: 0.6558 - accuracy: 0.6812
 1300/25000 [>.............................] - ETA: 23s - loss: 0.6554 - accuracy: 0.6805
 1400/25000 [>.............................] - ETA: 22s - loss: 0.6550 - accuracy: 0.6799
 1500/25000 [>.............................] - ETA: 22s - loss: 0.6544 - accuracy: 0.6809
 1600/25000 [>.............................] - ETA: 21s - loss: 0.6540 - accuracy: 0.6807
 1700/25000 [=>............................] - ETA: 21s - loss: 0.6537 - accuracy: 0.6791
 1800/25000 [=>............................] - ETA: 20s - loss: 0.6531 - accuracy: 0.6793
 1900/25000 [=>............................] - ETA: 20s - loss: 0.6524 - accuracy: 0.6802
 2000/25000 [=>............................] - ETA: 19s - loss: 0.6519 - accuracy: 0.6798
 2100/25000 [=>............................] - ETA: 19s - loss: 0.6513 - accuracy: 0.6798
 2200/25000 [=>............................] - ETA: 19s - loss: 0.6507 - accuracy: 0.6800
 2300/25000 [=>............................] - ETA: 18s - loss: 0.6502 - accuracy: 0.6797
 2400/25000 [=>............................] - ETA: 18s - loss: 0.6495 - accuracy: 0.6805
 2500/25000 [==>...........................] - ETA: 18s - loss: 0.6489 - accuracy: 0.6802
 2600/25000 [==>...........................] - ETA: 18s - loss: 0.6483 - accuracy: 0.6799
 2700/25000 [==>...........................] - ETA: 17s - loss: 0.6475 - accuracy: 0.6808
 2800/25000 [==>...........................] - ETA: 17s - loss: 0.6468 - accuracy: 0.6809
 2900/25000 [==>...........................] - ETA: 17s - loss: 0.6461 - accuracy: 0.6809
 3000/25000 [==>...........................] - ETA: 17s - loss: 0.6455 - accuracy: 0.6804
 3100/25000 [==>...........................] - ETA: 16s - loss: 0.6447 - accuracy: 0.6807
 3200/25000 [==>...........................] - ETA: 16s - loss: 0.6440 - accuracy: 0.6808
 3300/25000 [==>...........................] - ETA: 16s - loss: 0.6432 - accuracy: 0.6809
 3400/25000 [===>..........................] - ETA: 16s - loss: 0.6425 - accuracy: 0.6807
 3500/25000 [===>..........................] - ETA: 16s - loss: 0.6416 - accuracy: 0.6811
 3600/25000 [===>..........................] - ETA: 16s - loss: 0.6407 - accuracy: 0.6816
 3700/25000 [===>..........................] - ETA: 15s - loss: 0.6400 - accuracy: 0.6814
 3800/25000 [===>..........................] - ETA: 15s - loss: 0.6394 - accuracy: 0.6811
 3900/25000 [===>..........................] - ETA: 15s - loss: 0.6386 - accuracy: 0.6814
 4000/25000 [===>..........................] - ETA: 15s - loss: 0.6379 - accuracy: 0.6811
 4100/25000 [===>..........................] - ETA: 15s - loss: 0.6372 - accuracy: 0.6811
 4200/25000 [====>.........................] - ETA: 15s - loss: 0.6366 - accuracy: 0.6809
 4300/25000 [====>.........................] - ETA: 15s - loss: 0.6361 - accuracy: 0.6806
 4400/25000 [====>.........................] - ETA: 15s - loss: 0.6354 - accuracy: 0.6807
 4500/25000 [====>.........................] - ETA: 14s - loss: 0.6348 - accuracy: 0.6810
 4600/25000 [====>.........................] - ETA: 14s - loss: 0.6340 - accuracy: 0.6838
 4700/25000 [====>.........................] - ETA: 14s - loss: 0.6333 - accuracy: 0.6860
 4800/25000 [====>.........................] - ETA: 14s - loss: 0.6327 - accuracy: 0.6882
 4900/25000 [====>.........................] - ETA: 14s - loss: 0.6321 - accuracy: 0.6904
 5000/25000 [=====>........................] - ETA: 14s - loss: 0.6315 - accuracy: 0.6926
 5100/25000 [=====>........................] - ETA: 14s - loss: 0.6308 - accuracy: 0.6948
 5200/25000 [=====>........................] - ETA: 14s - loss: 0.6302 - accuracy: 0.6969
 5300/25000 [=====>........................] - ETA: 14s - loss: 0.6293 - accuracy: 0.6994
 5400/25000 [=====>........................] - ETA: 13s - loss: 0.6288 - accuracy: 0.7010
 5500/25000 [=====>........................] - ETA: 13s - loss: 0.6282 - accuracy: 0.7029
 5600/25000 [=====>........................] - ETA: 13s - loss: 0.6276 - accuracy: 0.7046
 5700/25000 [=====>........................] - ETA: 13s - loss: 0.6271 - accuracy: 0.7062
 5800/25000 [=====>........................] - ETA: 13s - loss: 0.6266 - accuracy: 0.7080
 5900/25000 [======>.......................] - ETA: 13s - loss: 0.6262 - accuracy: 0.7091
 6000/25000 [======>.......................] - ETA: 13s - loss: 0.6258 - accuracy: 0.7103
 6100/25000 [======>.......................] - ETA: 13s - loss: 0.6252 - accuracy: 0.7118
 6200/25000 [======>.......................] - ETA: 13s - loss: 0.6247 - accuracy: 0.7129
 6300/25000 [======>.......................] - ETA: 13s - loss: 0.6242 - accuracy: 0.7143
 6400/25000 [======>.......................] - ETA: 12s - loss: 0.6235 - accuracy: 0.7160
 6500/25000 [======>.......................] - ETA: 12s - loss: 0.6230 - accuracy: 0.7171
 6600/25000 [======>.......................] - ETA: 12s - loss: 0.6224 - accuracy: 0.7184
 6700/25000 [=======>......................] - ETA: 12s - loss: 0.6220 - accuracy: 0.7197
 6800/25000 [=======>......................] - ETA: 12s - loss: 0.6215 - accuracy: 0.7209
 6900/25000 [=======>......................] - ETA: 12s - loss: 0.6211 - accuracy: 0.7217
 7000/25000 [=======>......................] - ETA: 12s - loss: 0.6206 - accuracy: 0.7230
 7100/25000 [=======>......................] - ETA: 12s - loss: 0.6201 - accuracy: 0.7239
 7200/25000 [=======>......................] - ETA: 12s - loss: 0.6197 - accuracy: 0.7250
 7300/25000 [=======>......................] - ETA: 12s - loss: 0.6193 - accuracy: 0.7259
 7400/25000 [=======>......................] - ETA: 12s - loss: 0.6188 - accuracy: 0.7268
 7500/25000 [========>.....................] - ETA: 12s - loss: 0.6183 - accuracy: 0.7280
 7600/25000 [========>.....................] - ETA: 11s - loss: 0.6178 - accuracy: 0.7289
 7700/25000 [========>.....................] - ETA: 11s - loss: 0.6173 - accuracy: 0.7300
 7800/25000 [========>.....................] - ETA: 11s - loss: 0.6169 - accuracy: 0.7308
 7900/25000 [========>.....................] - ETA: 11s - loss: 0.6164 - accuracy: 0.7316
 8000/25000 [========>.....................] - ETA: 11s - loss: 0.6159 - accuracy: 0.7323
 8100/25000 [========>.....................] - ETA: 11s - loss: 0.6155 - accuracy: 0.7330
 8200/25000 [========>.....................] - ETA: 11s - loss: 0.6151 - accuracy: 0.7339
 8300/25000 [========>.....................] - ETA: 11s - loss: 0.6146 - accuracy: 0.7347
 8400/25000 [=========>....................] - ETA: 11s - loss: 0.6142 - accuracy: 0.7352
 8500/25000 [=========>....................] - ETA: 11s - loss: 0.6137 - accuracy: 0.7359
 8600/25000 [=========>....................] - ETA: 11s - loss: 0.6134 - accuracy: 0.7366
 8700/25000 [=========>....................] - ETA: 11s - loss: 0.6129 - accuracy: 0.7373
 8800/25000 [=========>....................] - ETA: 10s - loss: 0.6125 - accuracy: 0.7381
 8900/25000 [=========>....................] - ETA: 10s - loss: 0.6121 - accuracy: 0.7385
 9000/25000 [=========>....................] - ETA: 10s - loss: 0.6117 - accuracy: 0.7390
 9100/25000 [=========>....................] - ETA: 10s - loss: 0.6114 - accuracy: 0.7396
 9200/25000 [==========>...................] - ETA: 10s - loss: 0.6109 - accuracy: 0.7402
 9300/25000 [==========>...................] - ETA: 10s - loss: 0.6105 - accuracy: 0.7410
 9400/25000 [==========>...................] - ETA: 10s - loss: 0.6101 - accuracy: 0.7414
 9500/25000 [==========>...................] - ETA: 10s - loss: 0.6098 - accuracy: 0.7419
 9600/25000 [==========>...................] - ETA: 10s - loss: 0.6094 - accuracy: 0.7425
 9700/25000 [==========>...................] - ETA: 10s - loss: 0.6090 - accuracy: 0.7431
 9800/25000 [==========>...................] - ETA: 10s - loss: 0.6085 - accuracy: 0.7437
 9900/25000 [==========>...................] - ETA: 10s - loss: 0.6082 - accuracy: 0.7442
10000/25000 [===========>..................] - ETA: 10s - loss: 0.6078 - accuracy: 0.7449
10100/25000 [===========>..................] - ETA: 10s - loss: 0.6073 - accuracy: 0.7454
10200/25000 [===========>..................] - ETA: 9s - loss: 0.6069 - accuracy: 0.7459 
10300/25000 [===========>..................] - ETA: 9s - loss: 0.6064 - accuracy: 0.7466
10400/25000 [===========>..................] - ETA: 9s - loss: 0.6061 - accuracy: 0.7469
10500/25000 [===========>..................] - ETA: 9s - loss: 0.6057 - accuracy: 0.7473
10600/25000 [===========>..................] - ETA: 9s - loss: 0.6053 - accuracy: 0.7476
10700/25000 [===========>..................] - ETA: 9s - loss: 0.6050 - accuracy: 0.7481
10800/25000 [===========>..................] - ETA: 9s - loss: 0.6046 - accuracy: 0.7485
10900/25000 [============>.................] - ETA: 9s - loss: 0.6042 - accuracy: 0.7490
11000/25000 [============>.................] - ETA: 9s - loss: 0.6038 - accuracy: 0.7496
11100/25000 [============>.................] - ETA: 9s - loss: 0.6034 - accuracy: 0.7500
11200/25000 [============>.................] - ETA: 9s - loss: 0.6030 - accuracy: 0.7504
11300/25000 [============>.................] - ETA: 9s - loss: 0.6026 - accuracy: 0.7510
11400/25000 [============>.................] - ETA: 9s - loss: 0.6022 - accuracy: 0.7515
11500/25000 [============>.................] - ETA: 8s - loss: 0.6018 - accuracy: 0.7519
11600/25000 [============>.................] - ETA: 8s - loss: 0.6015 - accuracy: 0.7521
11700/25000 [=============>................] - ETA: 8s - loss: 0.6012 - accuracy: 0.7524
11800/25000 [=============>................] - ETA: 8s - loss: 0.6008 - accuracy: 0.7528
11900/25000 [=============>................] - ETA: 8s - loss: 0.6005 - accuracy: 0.7531
12000/25000 [=============>................] - ETA: 8s - loss: 0.6001 - accuracy: 0.7536
12100/25000 [=============>................] - ETA: 8s - loss: 0.5998 - accuracy: 0.7540
12200/25000 [=============>................] - ETA: 8s - loss: 0.5995 - accuracy: 0.7542
12300/25000 [=============>................] - ETA: 8s - loss: 0.5991 - accuracy: 0.7546
12400/25000 [=============>................] - ETA: 8s - loss: 0.5987 - accuracy: 0.7550
12500/25000 [==============>...............] - ETA: 8s - loss: 0.5984 - accuracy: 0.7553
12600/25000 [==============>...............] - ETA: 8s - loss: 0.5980 - accuracy: 0.7555
12700/25000 [==============>...............] - ETA: 8s - loss: 0.5977 - accuracy: 0.7558
12800/25000 [==============>...............] - ETA: 8s - loss: 0.5975 - accuracy: 0.7559
12900/25000 [==============>...............] - ETA: 8s - loss: 0.5971 - accuracy: 0.7562
13000/25000 [==============>...............] - ETA: 7s - loss: 0.5967 - accuracy: 0.7566
13100/25000 [==============>...............] - ETA: 7s - loss: 0.5964 - accuracy: 0.7568
13200/25000 [==============>...............] - ETA: 7s - loss: 0.5960 - accuracy: 0.7572
13300/25000 [==============>...............] - ETA: 7s - loss: 0.5956 - accuracy: 0.7576
13400/25000 [===============>..............] - ETA: 7s - loss: 0.5953 - accuracy: 0.7580
13500/25000 [===============>..............] - ETA: 7s - loss: 0.5950 - accuracy: 0.7581
13600/25000 [===============>..............] - ETA: 7s - loss: 0.5947 - accuracy: 0.7584
13700/25000 [===============>..............] - ETA: 7s - loss: 0.5944 - accuracy: 0.7586
13800/25000 [===============>..............] - ETA: 7s - loss: 0.5940 - accuracy: 0.7589
13900/25000 [===============>..............] - ETA: 7s - loss: 0.5937 - accuracy: 0.7592
14000/25000 [===============>..............] - ETA: 7s - loss: 0.5933 - accuracy: 0.7594
14100/25000 [===============>..............] - ETA: 7s - loss: 0.5930 - accuracy: 0.7597
14200/25000 [================>.............] - ETA: 7s - loss: 0.5927 - accuracy: 0.7599
14300/25000 [================>.............] - ETA: 7s - loss: 0.5923 - accuracy: 0.7604
14400/25000 [================>.............] - ETA: 6s - loss: 0.5919 - accuracy: 0.7606
14500/25000 [================>.............] - ETA: 6s - loss: 0.5916 - accuracy: 0.7609
14600/25000 [================>.............] - ETA: 6s - loss: 0.5913 - accuracy: 0.7611
14700/25000 [================>.............] - ETA: 6s - loss: 0.5909 - accuracy: 0.7614
14800/25000 [================>.............] - ETA: 6s - loss: 0.5906 - accuracy: 0.7617
14900/25000 [================>.............] - ETA: 6s - loss: 0.5903 - accuracy: 0.7619
15000/25000 [=================>............] - ETA: 6s - loss: 0.5900 - accuracy: 0.7622
15100/25000 [=================>............] - ETA: 6s - loss: 0.5896 - accuracy: 0.7624
15200/25000 [=================>............] - ETA: 6s - loss: 0.5893 - accuracy: 0.7627
15300/25000 [=================>............] - ETA: 6s - loss: 0.5890 - accuracy: 0.7630
15400/25000 [=================>............] - ETA: 6s - loss: 0.5887 - accuracy: 0.7630
15500/25000 [=================>............] - ETA: 6s - loss: 0.5883 - accuracy: 0.7634
15600/25000 [=================>............] - ETA: 6s - loss: 0.5880 - accuracy: 0.7636
15700/25000 [=================>............] - ETA: 6s - loss: 0.5876 - accuracy: 0.7638
15800/25000 [=================>............] - ETA: 6s - loss: 0.5874 - accuracy: 0.7640
15900/25000 [==================>...........] - ETA: 5s - loss: 0.5871 - accuracy: 0.7642
16000/25000 [==================>...........] - ETA: 5s - loss: 0.5868 - accuracy: 0.7643
16100/25000 [==================>...........] - ETA: 5s - loss: 0.5865 - accuracy: 0.7645
16200/25000 [==================>...........] - ETA: 5s - loss: 0.5861 - accuracy: 0.7647
16300/25000 [==================>...........] - ETA: 5s - loss: 0.5858 - accuracy: 0.7649
16400/25000 [==================>...........] - ETA: 5s - loss: 0.5855 - accuracy: 0.7652
16500/25000 [==================>...........] - ETA: 5s - loss: 0.5851 - accuracy: 0.7655
16600/25000 [==================>...........] - ETA: 5s - loss: 0.5848 - accuracy: 0.7656
16700/25000 [===================>..........] - ETA: 5s - loss: 0.5845 - accuracy: 0.7657
16800/25000 [===================>..........] - ETA: 5s - loss: 0.5842 - accuracy: 0.7659
16900/25000 [===================>..........] - ETA: 5s - loss: 0.5839 - accuracy: 0.7660
17000/25000 [===================>..........] - ETA: 5s - loss: 0.5836 - accuracy: 0.7663
17100/25000 [===================>..........] - ETA: 5s - loss: 0.5833 - accuracy: 0.7664
17200/25000 [===================>..........] - ETA: 5s - loss: 0.5830 - accuracy: 0.7667
17300/25000 [===================>..........] - ETA: 5s - loss: 0.5826 - accuracy: 0.7675
17400/25000 [===================>..........] - ETA: 4s - loss: 0.5823 - accuracy: 0.7681
17500/25000 [====================>.........] - ETA: 4s - loss: 0.5820 - accuracy: 0.7688
17600/25000 [====================>.........] - ETA: 4s - loss: 0.5817 - accuracy: 0.7695
17700/25000 [====================>.........] - ETA: 4s - loss: 0.5814 - accuracy: 0.7702
17800/25000 [====================>.........] - ETA: 4s - loss: 0.5811 - accuracy: 0.7708
17900/25000 [====================>.........] - ETA: 4s - loss: 0.5808 - accuracy: 0.7714
18000/25000 [====================>.........] - ETA: 4s - loss: 0.5805 - accuracy: 0.7721
18100/25000 [====================>.........] - ETA: 4s - loss: 0.5802 - accuracy: 0.7727
18200/25000 [====================>.........] - ETA: 4s - loss: 0.5799 - accuracy: 0.7733
18300/25000 [====================>.........] - ETA: 4s - loss: 0.5797 - accuracy: 0.7739
18400/25000 [=====================>........] - ETA: 4s - loss: 0.5794 - accuracy: 0.7745
18500/25000 [=====================>........] - ETA: 4s - loss: 0.5791 - accuracy: 0.7750
18600/25000 [=====================>........] - ETA: 4s - loss: 0.5789 - accuracy: 0.7756
18700/25000 [=====================>........] - ETA: 4s - loss: 0.5786 - accuracy: 0.7762
18800/25000 [=====================>........] - ETA: 4s - loss: 0.5783 - accuracy: 0.7769
18900/25000 [=====================>........] - ETA: 3s - loss: 0.5780 - accuracy: 0.7774
19000/25000 [=====================>........] - ETA: 3s - loss: 0.5777 - accuracy: 0.7780
19100/25000 [=====================>........] - ETA: 3s - loss: 0.5774 - accuracy: 0.7785
19200/25000 [======================>.......] - ETA: 3s - loss: 0.5771 - accuracy: 0.7791
19300/25000 [======================>.......] - ETA: 3s - loss: 0.5769 - accuracy: 0.7797
19400/25000 [======================>.......] - ETA: 3s - loss: 0.5765 - accuracy: 0.7802
19500/25000 [======================>.......] - ETA: 3s - loss: 0.5763 - accuracy: 0.7808
19600/25000 [======================>.......] - ETA: 3s - loss: 0.5760 - accuracy: 0.7813
19700/25000 [======================>.......] - ETA: 3s - loss: 0.5757 - accuracy: 0.7819
19800/25000 [======================>.......] - ETA: 3s - loss: 0.5754 - accuracy: 0.7824
19900/25000 [======================>.......] - ETA: 3s - loss: 0.5751 - accuracy: 0.7829
20000/25000 [=======================>......] - ETA: 3s - loss: 0.5748 - accuracy: 0.7834
20100/25000 [=======================>......] - ETA: 3s - loss: 0.5745 - accuracy: 0.7839
20200/25000 [=======================>......] - ETA: 3s - loss: 0.5742 - accuracy: 0.7844
20300/25000 [=======================>......] - ETA: 3s - loss: 0.5740 - accuracy: 0.7849
20400/25000 [=======================>......] - ETA: 2s - loss: 0.5737 - accuracy: 0.7855
20500/25000 [=======================>......] - ETA: 2s - loss: 0.5734 - accuracy: 0.7859
20600/25000 [=======================>......] - ETA: 2s - loss: 0.5731 - accuracy: 0.7864
20700/25000 [=======================>......] - ETA: 2s - loss: 0.5729 - accuracy: 0.7869
20800/25000 [=======================>......] - ETA: 2s - loss: 0.5726 - accuracy: 0.7874
20900/25000 [========================>.....] - ETA: 2s - loss: 0.5723 - accuracy: 0.7878
21000/25000 [========================>.....] - ETA: 2s - loss: 0.5720 - accuracy: 0.7883
21100/25000 [========================>.....] - ETA: 2s - loss: 0.5718 - accuracy: 0.7888
21200/25000 [========================>.....] - ETA: 2s - loss: 0.5715 - accuracy: 0.7892
21300/25000 [========================>.....] - ETA: 2s - loss: 0.5712 - accuracy: 0.7897
21400/25000 [========================>.....] - ETA: 2s - loss: 0.5710 - accuracy: 0.7901
21500/25000 [========================>.....] - ETA: 2s - loss: 0.5707 - accuracy: 0.7906
21600/25000 [========================>.....] - ETA: 2s - loss: 0.5704 - accuracy: 0.7910
21700/25000 [=========================>....] - ETA: 2s - loss: 0.5702 - accuracy: 0.7915
21800/25000 [=========================>....] - ETA: 2s - loss: 0.5699 - accuracy: 0.7919
21900/25000 [=========================>....] - ETA: 2s - loss: 0.5696 - accuracy: 0.7923
22000/25000 [=========================>....] - ETA: 1s - loss: 0.5693 - accuracy: 0.7928
22100/25000 [=========================>....] - ETA: 1s - loss: 0.5691 - accuracy: 0.7932
22200/25000 [=========================>....] - ETA: 1s - loss: 0.5688 - accuracy: 0.7936
22300/25000 [=========================>....] - ETA: 1s - loss: 0.5685 - accuracy: 0.7940
22400/25000 [=========================>....] - ETA: 1s - loss: 0.5683 - accuracy: 0.7944
22500/25000 [==========================>...] - ETA: 1s - loss: 0.5680 - accuracy: 0.7948
22600/25000 [==========================>...] - ETA: 1s - loss: 0.5676 - accuracy: 0.7953
22700/25000 [==========================>...] - ETA: 1s - loss: 0.5674 - accuracy: 0.7957
22800/25000 [==========================>...] - ETA: 1s - loss: 0.5671 - accuracy: 0.7961
22900/25000 [==========================>...] - ETA: 1s - loss: 0.5669 - accuracy: 0.7965
23000/25000 [==========================>...] - ETA: 1s - loss: 0.5666 - accuracy: 0.7969
23100/25000 [==========================>...] - ETA: 1s - loss: 0.5663 - accuracy: 0.7973
23200/25000 [==========================>...] - ETA: 1s - loss: 0.5661 - accuracy: 0.7976
23300/25000 [==========================>...] - ETA: 1s - loss: 0.5658 - accuracy: 0.7980
23400/25000 [===========================>..] - ETA: 1s - loss: 0.5655 - accuracy: 0.7984
23500/25000 [===========================>..] - ETA: 0s - loss: 0.5652 - accuracy: 0.7988
23600/25000 [===========================>..] - ETA: 0s - loss: 0.5649 - accuracy: 0.7992
23700/25000 [===========================>..] - ETA: 0s - loss: 0.5646 - accuracy: 0.7996
23800/25000 [===========================>..] - ETA: 0s - loss: 0.5643 - accuracy: 0.8000
23900/25000 [===========================>..] - ETA: 0s - loss: 0.5640 - accuracy: 0.8004
24000/25000 [===========================>..] - ETA: 0s - loss: 0.5637 - accuracy: 0.8007
24100/25000 [===========================>..] - ETA: 0s - loss: 0.5634 - accuracy: 0.8012
24200/25000 [============================>.] - ETA: 0s - loss: 0.5632 - accuracy: 0.8015
24300/25000 [============================>.] - ETA: 0s - loss: 0.5629 - accuracy: 0.8019
24400/25000 [============================>.] - ETA: 0s - loss: 0.5626 - accuracy: 0.8022
24500/25000 [============================>.] - ETA: 0s - loss: 0.5624 - accuracy: 0.8026
24600/25000 [============================>.] - ETA: 0s - loss: 0.5621 - accuracy: 0.8029
24700/25000 [============================>.] - ETA: 0s - loss: 0.5619 - accuracy: 0.8032
24800/25000 [============================>.] - ETA: 0s - loss: 0.5616 - accuracy: 0.8035
24900/25000 [============================>.] - ETA: 0s - loss: 0.5614 - accuracy: 0.8038
25000/25000 [==============================] - 20s 809us/step - loss: 0.5611 - accuracy: 0.8041 - val_loss: 0.4952 - val_accuracy: 0.8858
Epoch 2/10

  100/25000 [..............................] - ETA: 15s - loss: 0.4953 - accuracy: 0.8886
  200/25000 [..............................] - ETA: 15s - loss: 0.4956 - accuracy: 0.8893
  300/25000 [..............................] - ETA: 15s - loss: 0.4945 - accuracy: 0.8890
  400/25000 [..............................] - ETA: 15s - loss: 0.4939 - accuracy: 0.8896
  500/25000 [..............................] - ETA: 15s - loss: 0.4943 - accuracy: 0.8889
  600/25000 [..............................] - ETA: 15s - loss: 0.4949 - accuracy: 0.8864
  700/25000 [..............................] - ETA: 15s - loss: 0.4940 - accuracy: 0.8871
  800/25000 [..............................] - ETA: 14s - loss: 0.4929 - accuracy: 0.8884
  900/25000 [>.............................] - ETA: 14s - loss: 0.4931 - accuracy: 0.8884
 1000/25000 [>.............................] - ETA: 14s - loss: 0.4924 - accuracy: 0.8879
 1100/25000 [>.............................] - ETA: 14s - loss: 0.4927 - accuracy: 0.8873
 1200/25000 [>.............................] - ETA: 15s - loss: 0.4931 - accuracy: 0.8865
 1300/25000 [>.............................] - ETA: 14s - loss: 0.4933 - accuracy: 0.8867
 1400/25000 [>.............................] - ETA: 14s - loss: 0.4932 - accuracy: 0.8866
 1500/25000 [>.............................] - ETA: 14s - loss: 0.4930 - accuracy: 0.8862
 1600/25000 [>.............................] - ETA: 14s - loss: 0.4929 - accuracy: 0.8858
 1700/25000 [=>............................] - ETA: 14s - loss: 0.4927 - accuracy: 0.8859
 1800/25000 [=>............................] - ETA: 14s - loss: 0.4923 - accuracy: 0.8862
 1900/25000 [=>............................] - ETA: 14s - loss: 0.4919 - accuracy: 0.8866
 2000/25000 [=>............................] - ETA: 14s - loss: 0.4919 - accuracy: 0.8866
 2100/25000 [=>............................] - ETA: 14s - loss: 0.4916 - accuracy: 0.8859
 2200/25000 [=>............................] - ETA: 14s - loss: 0.4919 - accuracy: 0.8858
 2300/25000 [=>............................] - ETA: 14s - loss: 0.4915 - accuracy: 0.8860
 2400/25000 [=>............................] - ETA: 14s - loss: 0.4916 - accuracy: 0.8859
 2500/25000 [==>...........................] - ETA: 14s - loss: 0.4914 - accuracy: 0.8860
 2600/25000 [==>...........................] - ETA: 14s - loss: 0.4913 - accuracy: 0.8859
 2700/25000 [==>...........................] - ETA: 13s - loss: 0.4911 - accuracy: 0.8859
 2800/25000 [==>...........................] - ETA: 13s - loss: 0.4906 - accuracy: 0.8859
 2900/25000 [==>...........................] - ETA: 13s - loss: 0.4908 - accuracy: 0.8857
 3000/25000 [==>...........................] - ETA: 13s - loss: 0.4907 - accuracy: 0.8857
 3100/25000 [==>...........................] - ETA: 13s - loss: 0.4908 - accuracy: 0.8858
 3200/25000 [==>...........................] - ETA: 13s - loss: 0.4906 - accuracy: 0.8855
 3300/25000 [==>...........................] - ETA: 13s - loss: 0.4905 - accuracy: 0.8856
 3400/25000 [===>..........................] - ETA: 13s - loss: 0.4902 - accuracy: 0.8858
 3500/25000 [===>..........................] - ETA: 13s - loss: 0.4903 - accuracy: 0.8856
 3600/25000 [===>..........................] - ETA: 13s - loss: 0.4900 - accuracy: 0.8856
 3700/25000 [===>..........................] - ETA: 13s - loss: 0.4898 - accuracy: 0.8855
 3800/25000 [===>..........................] - ETA: 13s - loss: 0.4899 - accuracy: 0.8853
 3900/25000 [===>..........................] - ETA: 13s - loss: 0.4897 - accuracy: 0.8854
 4000/25000 [===>..........................] - ETA: 13s - loss: 0.4892 - accuracy: 0.8857
 4100/25000 [===>..........................] - ETA: 13s - loss: 0.4889 - accuracy: 0.8859
 4200/25000 [====>.........................] - ETA: 13s - loss: 0.4888 - accuracy: 0.8857
 4300/25000 [====>.........................] - ETA: 12s - loss: 0.4887 - accuracy: 0.8857
 4400/25000 [====>.........................] - ETA: 12s - loss: 0.4886 - accuracy: 0.8856
 4500/25000 [====>.........................] - ETA: 12s - loss: 0.4883 - accuracy: 0.8857
 4600/25000 [====>.........................] - ETA: 12s - loss: 0.4880 - accuracy: 0.8856
 4700/25000 [====>.........................] - ETA: 12s - loss: 0.4879 - accuracy: 0.8857
 4800/25000 [====>.........................] - ETA: 12s - loss: 0.4877 - accuracy: 0.8857
 4900/25000 [====>.........................] - ETA: 12s - loss: 0.4875 - accuracy: 0.8860
 5000/25000 [=====>........................] - ETA: 12s - loss: 0.4874 - accuracy: 0.8859
 5100/25000 [=====>........................] - ETA: 12s - loss: 0.4871 - accuracy: 0.8859
 5200/25000 [=====>........................] - ETA: 12s - loss: 0.4871 - accuracy: 0.8858
 5300/25000 [=====>........................] - ETA: 12s - loss: 0.4870 - accuracy: 0.8859
 5400/25000 [=====>........................] - ETA: 12s - loss: 0.4868 - accuracy: 0.8859
 5500/25000 [=====>........................] - ETA: 12s - loss: 0.4867 - accuracy: 0.8858
 5600/25000 [=====>........................] - ETA: 12s - loss: 0.4865 - accuracy: 0.8858
 5700/25000 [=====>........................] - ETA: 12s - loss: 0.4861 - accuracy: 0.8859
 5800/25000 [=====>........................] - ETA: 12s - loss: 0.4860 - accuracy: 0.8858
 5900/25000 [======>.......................] - ETA: 12s - loss: 0.4859 - accuracy: 0.8857
 6000/25000 [======>.......................] - ETA: 12s - loss: 0.4857 - accuracy: 0.8857
 6100/25000 [======>.......................] - ETA: 12s - loss: 0.4856 - accuracy: 0.8856
 6200/25000 [======>.......................] - ETA: 11s - loss: 0.4854 - accuracy: 0.8856
 6300/25000 [======>.......................] - ETA: 11s - loss: 0.4853 - accuracy: 0.8857
 6400/25000 [======>.......................] - ETA: 11s - loss: 0.4852 - accuracy: 0.8856
 6500/25000 [======>.......................] - ETA: 11s - loss: 0.4851 - accuracy: 0.8855
 6600/25000 [======>.......................] - ETA: 11s - loss: 0.4850 - accuracy: 0.8853
 6700/25000 [=======>......................] - ETA: 11s - loss: 0.4849 - accuracy: 0.8853
 6800/25000 [=======>......................] - ETA: 11s - loss: 0.4846 - accuracy: 0.8854
 6900/25000 [=======>......................] - ETA: 11s - loss: 0.4843 - accuracy: 0.8855
 7000/25000 [=======>......................] - ETA: 11s - loss: 0.4840 - accuracy: 0.8856
 7100/25000 [=======>......................] - ETA: 11s - loss: 0.4838 - accuracy: 0.8856
 7200/25000 [=======>......................] - ETA: 11s - loss: 0.4837 - accuracy: 0.8856
 7300/25000 [=======>......................] - ETA: 11s - loss: 0.4834 - accuracy: 0.8857
 7400/25000 [=======>......................] - ETA: 11s - loss: 0.4833 - accuracy: 0.8856
 7500/25000 [========>.....................] - ETA: 11s - loss: 0.4830 - accuracy: 0.8856
 7600/25000 [========>.....................] - ETA: 11s - loss: 0.4830 - accuracy: 0.8856
 7700/25000 [========>.....................] - ETA: 10s - loss: 0.4828 - accuracy: 0.8855
 7800/25000 [========>.....................] - ETA: 10s - loss: 0.4826 - accuracy: 0.8856
 7900/25000 [========>.....................] - ETA: 10s - loss: 0.4824 - accuracy: 0.8856
 8000/25000 [========>.....................] - ETA: 10s - loss: 0.4822 - accuracy: 0.8856
 8100/25000 [========>.....................] - ETA: 10s - loss: 0.4821 - accuracy: 0.8856
 8200/25000 [========>.....................] - ETA: 10s - loss: 0.4819 - accuracy: 0.8856
 8300/25000 [========>.....................] - ETA: 10s - loss: 0.4818 - accuracy: 0.8856
 8400/25000 [=========>....................] - ETA: 10s - loss: 0.4816 - accuracy: 0.8856
 8500/25000 [=========>....................] - ETA: 10s - loss: 0.4815 - accuracy: 0.8856
 8600/25000 [=========>....................] - ETA: 10s - loss: 0.4814 - accuracy: 0.8856
 8700/25000 [=========>....................] - ETA: 10s - loss: 0.4813 - accuracy: 0.8856
 8800/25000 [=========>....................] - ETA: 10s - loss: 0.4811 - accuracy: 0.8856
 8900/25000 [=========>....................] - ETA: 10s - loss: 0.4808 - accuracy: 0.8857
 9000/25000 [=========>....................] - ETA: 10s - loss: 0.4807 - accuracy: 0.8856
 9100/25000 [=========>....................] - ETA: 10s - loss: 0.4805 - accuracy: 0.8857
 9200/25000 [==========>...................] - ETA: 10s - loss: 0.4803 - accuracy: 0.8858
 9300/25000 [==========>...................] - ETA: 9s - loss: 0.4801 - accuracy: 0.8859 
 9400/25000 [==========>...................] - ETA: 9s - loss: 0.4798 - accuracy: 0.8859
 9500/25000 [==========>...................] - ETA: 9s - loss: 0.4798 - accuracy: 0.8858
 9600/25000 [==========>...................] - ETA: 9s - loss: 0.4796 - accuracy: 0.8859
 9700/25000 [==========>...................] - ETA: 9s - loss: 0.4792 - accuracy: 0.8860
 9800/25000 [==========>...................] - ETA: 9s - loss: 0.4790 - accuracy: 0.8861
 9900/25000 [==========>...................] - ETA: 9s - loss: 0.4787 - accuracy: 0.8861
10000/25000 [===========>..................] - ETA: 9s - loss: 0.4786 - accuracy: 0.8861
10100/25000 [===========>..................] - ETA: 9s - loss: 0.4784 - accuracy: 0.8861
10200/25000 [===========>..................] - ETA: 9s - loss: 0.4782 - accuracy: 0.8861
10300/25000 [===========>..................] - ETA: 9s - loss: 0.4780 - accuracy: 0.8862
10400/25000 [===========>..................] - ETA: 9s - loss: 0.4778 - accuracy: 0.8862
10500/25000 [===========>..................] - ETA: 9s - loss: 0.4777 - accuracy: 0.8862
10600/25000 [===========>..................] - ETA: 9s - loss: 0.4775 - accuracy: 0.8862
10700/25000 [===========>..................] - ETA: 9s - loss: 0.4773 - accuracy: 0.8863
10800/25000 [===========>..................] - ETA: 9s - loss: 0.4772 - accuracy: 0.8863
10900/25000 [============>.................] - ETA: 8s - loss: 0.4770 - accuracy: 0.8863
11000/25000 [============>.................] - ETA: 8s - loss: 0.4769 - accuracy: 0.8863
11100/25000 [============>.................] - ETA: 8s - loss: 0.4767 - accuracy: 0.8863
11200/25000 [============>.................] - ETA: 8s - loss: 0.4765 - accuracy: 0.8864
11300/25000 [============>.................] - ETA: 8s - loss: 0.4764 - accuracy: 0.8864
11400/25000 [============>.................] - ETA: 8s - loss: 0.4761 - accuracy: 0.8864
11500/25000 [============>.................] - ETA: 8s - loss: 0.4759 - accuracy: 0.8864
11600/25000 [============>.................] - ETA: 8s - loss: 0.4758 - accuracy: 0.8864
11700/25000 [=============>................] - ETA: 8s - loss: 0.4757 - accuracy: 0.8863
11800/25000 [=============>................] - ETA: 8s - loss: 0.4756 - accuracy: 0.8863
11900/25000 [=============>................] - ETA: 8s - loss: 0.4754 - accuracy: 0.8863
12000/25000 [=============>................] - ETA: 8s - loss: 0.4752 - accuracy: 0.8863
12100/25000 [=============>................] - ETA: 8s - loss: 0.4751 - accuracy: 0.8863
12200/25000 [=============>................] - ETA: 8s - loss: 0.4749 - accuracy: 0.8863
12300/25000 [=============>................] - ETA: 8s - loss: 0.4748 - accuracy: 0.8864
12400/25000 [=============>................] - ETA: 8s - loss: 0.4746 - accuracy: 0.8864
12500/25000 [==============>...............] - ETA: 8s - loss: 0.4745 - accuracy: 0.8864
12600/25000 [==============>...............] - ETA: 7s - loss: 0.4743 - accuracy: 0.8864
12700/25000 [==============>...............] - ETA: 7s - loss: 0.4741 - accuracy: 0.8864
12800/25000 [==============>...............] - ETA: 7s - loss: 0.4740 - accuracy: 0.8864
12900/25000 [==============>...............] - ETA: 7s - loss: 0.4739 - accuracy: 0.8864
13000/25000 [==============>...............] - ETA: 7s - loss: 0.4737 - accuracy: 0.8864
13100/25000 [==============>...............] - ETA: 7s - loss: 0.4736 - accuracy: 0.8864
13200/25000 [==============>...............] - ETA: 7s - loss: 0.4734 - accuracy: 0.8864
13300/25000 [==============>...............] - ETA: 7s - loss: 0.4732 - accuracy: 0.8864
13400/25000 [===============>..............] - ETA: 7s - loss: 0.4731 - accuracy: 0.8863
13500/25000 [===============>..............] - ETA: 7s - loss: 0.4730 - accuracy: 0.8862
13600/25000 [===============>..............] - ETA: 7s - loss: 0.4729 - accuracy: 0.8862
13700/25000 [===============>..............] - ETA: 7s - loss: 0.4727 - accuracy: 0.8862
13800/25000 [===============>..............] - ETA: 7s - loss: 0.4726 - accuracy: 0.8862
13900/25000 [===============>..............] - ETA: 7s - loss: 0.4724 - accuracy: 0.8862
14000/25000 [===============>..............] - ETA: 7s - loss: 0.4722 - accuracy: 0.8862
14100/25000 [===============>..............] - ETA: 7s - loss: 0.4720 - accuracy: 0.8862
14200/25000 [================>.............] - ETA: 7s - loss: 0.4719 - accuracy: 0.8862
14300/25000 [================>.............] - ETA: 6s - loss: 0.4717 - accuracy: 0.8862
14400/25000 [================>.............] - ETA: 6s - loss: 0.4715 - accuracy: 0.8863
14500/25000 [================>.............] - ETA: 6s - loss: 0.4715 - accuracy: 0.8862
14600/25000 [================>.............] - ETA: 6s - loss: 0.4713 - accuracy: 0.8862
14700/25000 [================>.............] - ETA: 6s - loss: 0.4712 - accuracy: 0.8862
14800/25000 [================>.............] - ETA: 6s - loss: 0.4711 - accuracy: 0.8861
14900/25000 [================>.............] - ETA: 6s - loss: 0.4709 - accuracy: 0.8861
15000/25000 [=================>............] - ETA: 6s - loss: 0.4708 - accuracy: 0.8861
15100/25000 [=================>............] - ETA: 6s - loss: 0.4706 - accuracy: 0.8861
15200/25000 [=================>............] - ETA: 6s - loss: 0.4704 - accuracy: 0.8861
15300/25000 [=================>............] - ETA: 6s - loss: 0.4703 - accuracy: 0.8861
15400/25000 [=================>............] - ETA: 6s - loss: 0.4702 - accuracy: 0.8861
15500/25000 [=================>............] - ETA: 6s - loss: 0.4699 - accuracy: 0.8861
15600/25000 [=================>............] - ETA: 6s - loss: 0.4698 - accuracy: 0.8861
15700/25000 [=================>............] - ETA: 6s - loss: 0.4697 - accuracy: 0.8861
15800/25000 [=================>............] - ETA: 5s - loss: 0.4695 - accuracy: 0.8861
15900/25000 [==================>...........] - ETA: 5s - loss: 0.4694 - accuracy: 0.8860
16000/25000 [==================>...........] - ETA: 5s - loss: 0.4693 - accuracy: 0.8860
16100/25000 [==================>...........] - ETA: 5s - loss: 0.4692 - accuracy: 0.8860
16200/25000 [==================>...........] - ETA: 5s - loss: 0.4690 - accuracy: 0.8860
16300/25000 [==================>...........] - ETA: 5s - loss: 0.4688 - accuracy: 0.8860
16400/25000 [==================>...........] - ETA: 5s - loss: 0.4686 - accuracy: 0.8860
16500/25000 [==================>...........] - ETA: 5s - loss: 0.4685 - accuracy: 0.8860
16600/25000 [==================>...........] - ETA: 5s - loss: 0.4684 - accuracy: 0.8860
16700/25000 [===================>..........] - ETA: 5s - loss: 0.4683 - accuracy: 0.8860
16800/25000 [===================>..........] - ETA: 5s - loss: 0.4682 - accuracy: 0.8859
16900/25000 [===================>..........] - ETA: 5s - loss: 0.4680 - accuracy: 0.8860
17000/25000 [===================>..........] - ETA: 5s - loss: 0.4679 - accuracy: 0.8859
17100/25000 [===================>..........] - ETA: 5s - loss: 0.4677 - accuracy: 0.8859
17200/25000 [===================>..........] - ETA: 5s - loss: 0.4676 - accuracy: 0.8859
17300/25000 [===================>..........] - ETA: 5s - loss: 0.4674 - accuracy: 0.8860
17400/25000 [===================>..........] - ETA: 4s - loss: 0.4673 - accuracy: 0.8860
17500/25000 [====================>.........] - ETA: 4s - loss: 0.4671 - accuracy: 0.8860
17600/25000 [====================>.........] - ETA: 4s - loss: 0.4669 - accuracy: 0.8860
17700/25000 [====================>.........] - ETA: 4s - loss: 0.4668 - accuracy: 0.8860
17800/25000 [====================>.........] - ETA: 4s - loss: 0.4665 - accuracy: 0.8861
17900/25000 [====================>.........] - ETA: 4s - loss: 0.4664 - accuracy: 0.8860
18000/25000 [====================>.........] - ETA: 4s - loss: 0.4663 - accuracy: 0.8860
18100/25000 [====================>.........] - ETA: 4s - loss: 0.4662 - accuracy: 0.8860
18200/25000 [====================>.........] - ETA: 4s - loss: 0.4661 - accuracy: 0.8860
18300/25000 [====================>.........] - ETA: 4s - loss: 0.4659 - accuracy: 0.8860
18400/25000 [=====================>........] - ETA: 4s - loss: 0.4657 - accuracy: 0.8861
18500/25000 [=====================>........] - ETA: 4s - loss: 0.4656 - accuracy: 0.8860
18600/25000 [=====================>........] - ETA: 4s - loss: 0.4655 - accuracy: 0.8861
18700/25000 [=====================>........] - ETA: 4s - loss: 0.4654 - accuracy: 0.8861
18800/25000 [=====================>........] - ETA: 4s - loss: 0.4653 - accuracy: 0.8860
18900/25000 [=====================>........] - ETA: 3s - loss: 0.4651 - accuracy: 0.8860
19000/25000 [=====================>........] - ETA: 3s - loss: 0.4650 - accuracy: 0.8861
19100/25000 [=====================>........] - ETA: 3s - loss: 0.4648 - accuracy: 0.8860
19200/25000 [======================>.......] - ETA: 3s - loss: 0.4647 - accuracy: 0.8860
19300/25000 [======================>.......] - ETA: 3s - loss: 0.4645 - accuracy: 0.8861
19400/25000 [======================>.......] - ETA: 3s - loss: 0.4643 - accuracy: 0.8861
19500/25000 [======================>.......] - ETA: 3s - loss: 0.4642 - accuracy: 0.8861
19600/25000 [======================>.......] - ETA: 3s - loss: 0.4641 - accuracy: 0.8860
19700/25000 [======================>.......] - ETA: 3s - loss: 0.4639 - accuracy: 0.8861
19800/25000 [======================>.......] - ETA: 3s - loss: 0.4637 - accuracy: 0.8861
19900/25000 [======================>.......] - ETA: 3s - loss: 0.4636 - accuracy: 0.8861
20000/25000 [=======================>......] - ETA: 3s - loss: 0.4634 - accuracy: 0.8861
20100/25000 [=======================>......] - ETA: 3s - loss: 0.4633 - accuracy: 0.8861
20200/25000 [=======================>......] - ETA: 3s - loss: 0.4631 - accuracy: 0.8861
20300/25000 [=======================>......] - ETA: 3s - loss: 0.4629 - accuracy: 0.8862
20400/25000 [=======================>......] - ETA: 2s - loss: 0.4628 - accuracy: 0.8861
20500/25000 [=======================>......] - ETA: 2s - loss: 0.4627 - accuracy: 0.8861
20600/25000 [=======================>......] - ETA: 2s - loss: 0.4626 - accuracy: 0.8861
20700/25000 [=======================>......] - ETA: 2s - loss: 0.4625 - accuracy: 0.8861
20800/25000 [=======================>......] - ETA: 2s - loss: 0.4624 - accuracy: 0.8860
20900/25000 [========================>.....] - ETA: 2s - loss: 0.4622 - accuracy: 0.8860
21000/25000 [========================>.....] - ETA: 2s - loss: 0.4621 - accuracy: 0.8861
21100/25000 [========================>.....] - ETA: 2s - loss: 0.4619 - accuracy: 0.8861
21200/25000 [========================>.....] - ETA: 2s - loss: 0.4618 - accuracy: 0.8861
21300/25000 [========================>.....] - ETA: 2s - loss: 0.4616 - accuracy: 0.8862
21400/25000 [========================>.....] - ETA: 2s - loss: 0.4614 - accuracy: 0.8862
21500/25000 [========================>.....] - ETA: 2s - loss: 0.4613 - accuracy: 0.8862
21600/25000 [========================>.....] - ETA: 2s - loss: 0.4611 - accuracy: 0.8862
21700/25000 [=========================>....] - ETA: 2s - loss: 0.4610 - accuracy: 0.8862
21800/25000 [=========================>....] - ETA: 2s - loss: 0.4609 - accuracy: 0.8862
21900/25000 [=========================>....] - ETA: 2s - loss: 0.4607 - accuracy: 0.8862
22000/25000 [=========================>....] - ETA: 1s - loss: 0.4606 - accuracy: 0.8862
22100/25000 [=========================>....] - ETA: 1s - loss: 0.4604 - accuracy: 0.8862
22200/25000 [=========================>....] - ETA: 1s - loss: 0.4603 - accuracy: 0.8862
22300/25000 [=========================>....] - ETA: 1s - loss: 0.4601 - accuracy: 0.8862
22400/25000 [=========================>....] - ETA: 1s - loss: 0.4600 - accuracy: 0.8862
22500/25000 [==========================>...] - ETA: 1s - loss: 0.4599 - accuracy: 0.8862
22600/25000 [==========================>...] - ETA: 1s - loss: 0.4597 - accuracy: 0.8863
22700/25000 [==========================>...] - ETA: 1s - loss: 0.4596 - accuracy: 0.8863
22800/25000 [==========================>...] - ETA: 1s - loss: 0.4594 - accuracy: 0.8863
22900/25000 [==========================>...] - ETA: 1s - loss: 0.4593 - accuracy: 0.8863
23000/25000 [==========================>...] - ETA: 1s - loss: 0.4591 - accuracy: 0.8863
23100/25000 [==========================>...] - ETA: 1s - loss: 0.4590 - accuracy: 0.8863
23200/25000 [==========================>...] - ETA: 1s - loss: 0.4589 - accuracy: 0.8863
23300/25000 [==========================>...] - ETA: 1s - loss: 0.4588 - accuracy: 0.8863
23400/25000 [===========================>..] - ETA: 1s - loss: 0.4586 - accuracy: 0.8863
23500/25000 [===========================>..] - ETA: 0s - loss: 0.4585 - accuracy: 0.8863
23600/25000 [===========================>..] - ETA: 0s - loss: 0.4584 - accuracy: 0.8863
23700/25000 [===========================>..] - ETA: 0s - loss: 0.4583 - accuracy: 0.8862
23800/25000 [===========================>..] - ETA: 0s - loss: 0.4581 - accuracy: 0.8863
23900/25000 [===========================>..] - ETA: 0s - loss: 0.4580 - accuracy: 0.8863
24000/25000 [===========================>..] - ETA: 0s - loss: 0.4579 - accuracy: 0.8863
24100/25000 [===========================>..] - ETA: 0s - loss: 0.4578 - accuracy: 0.8863
24200/25000 [============================>.] - ETA: 0s - loss: 0.4576 - accuracy: 0.8863
24300/25000 [============================>.] - ETA: 0s - loss: 0.4575 - accuracy: 0.8863
24400/25000 [============================>.] - ETA: 0s - loss: 0.4574 - accuracy: 0.8863
24500/25000 [============================>.] - ETA: 0s - loss: 0.4573 - accuracy: 0.8863
24600/25000 [============================>.] - ETA: 0s - loss: 0.4571 - accuracy: 0.8863
24700/25000 [============================>.] - ETA: 0s - loss: 0.4570 - accuracy: 0.8863
24800/25000 [============================>.] - ETA: 0s - loss: 0.4569 - accuracy: 0.8863
24900/25000 [============================>.] - ETA: 0s - loss: 0.4568 - accuracy: 0.8863
25000/25000 [==============================] - 20s 807us/step - loss: 0.4567 - accuracy: 0.8863 - val_loss: 0.4235 - val_accuracy: 0.8858
Epoch 3/10

  100/25000 [..............................] - ETA: 15s - loss: 0.4244 - accuracy: 0.8829
  200/25000 [..............................] - ETA: 15s - loss: 0.4289 - accuracy: 0.8821
  300/25000 [..............................] - ETA: 15s - loss: 0.4267 - accuracy: 0.8819
  400/25000 [..............................] - ETA: 15s - loss: 0.4257 - accuracy: 0.8832
  500/25000 [..............................] - ETA: 15s - loss: 0.4245 - accuracy: 0.8843
  600/25000 [..............................] - ETA: 14s - loss: 0.4257 - accuracy: 0.8843
  700/25000 [..............................] - ETA: 14s - loss: 0.4250 - accuracy: 0.8857
  800/25000 [..............................] - ETA: 14s - loss: 0.4253 - accuracy: 0.8850
  900/25000 [>.............................] - ETA: 14s - loss: 0.4248 - accuracy: 0.8846
 1000/25000 [>.............................] - ETA: 14s - loss: 0.4247 - accuracy: 0.8847
 1100/25000 [>.............................] - ETA: 14s - loss: 0.4240 - accuracy: 0.8856
 1200/25000 [>.............................] - ETA: 14s - loss: 0.4243 - accuracy: 0.8858
 1300/25000 [>.............................] - ETA: 14s - loss: 0.4246 - accuracy: 0.8855
 1400/25000 [>.............................] - ETA: 14s - loss: 0.4242 - accuracy: 0.8857
 1500/25000 [>.............................] - ETA: 14s - loss: 0.4242 - accuracy: 0.8860
 1600/25000 [>.............................] - ETA: 14s - loss: 0.4237 - accuracy: 0.8865
 1700/25000 [=>............................] - ETA: 14s - loss: 0.4231 - accuracy: 0.8869
 1800/25000 [=>............................] - ETA: 14s - loss: 0.4231 - accuracy: 0.8867
 1900/25000 [=>............................] - ETA: 14s - loss: 0.4232 - accuracy: 0.8865
 2000/25000 [=>............................] - ETA: 14s - loss: 0.4230 - accuracy: 0.8864
 2100/25000 [=>............................] - ETA: 14s - loss: 0.4227 - accuracy: 0.8865
 2200/25000 [=>............................] - ETA: 14s - loss: 0.4225 - accuracy: 0.8866
 2300/25000 [=>............................] - ETA: 13s - loss: 0.4223 - accuracy: 0.8866
 2400/25000 [=>............................] - ETA: 13s - loss: 0.4223 - accuracy: 0.8864
 2500/25000 [==>...........................] - ETA: 13s - loss: 0.4220 - accuracy: 0.8864
 2600/25000 [==>...........................] - ETA: 13s - loss: 0.4220 - accuracy: 0.8863
 2700/25000 [==>...........................] - ETA: 13s - loss: 0.4218 - accuracy: 0.8865
 2800/25000 [==>...........................] - ETA: 13s - loss: 0.4212 - accuracy: 0.8869
 2900/25000 [==>...........................] - ETA: 13s - loss: 0.4211 - accuracy: 0.8869
 3000/25000 [==>...........................] - ETA: 13s - loss: 0.4209 - accuracy: 0.8869
 3100/25000 [==>...........................] - ETA: 13s - loss: 0.4208 - accuracy: 0.8871
 3200/25000 [==>...........................] - ETA: 13s - loss: 0.4206 - accuracy: 0.8871
 3300/25000 [==>...........................] - ETA: 13s - loss: 0.4207 - accuracy: 0.8869
 3400/25000 [===>..........................] - ETA: 13s - loss: 0.4204 - accuracy: 0.8870
 3500/25000 [===>..........................] - ETA: 13s - loss: 0.4205 - accuracy: 0.8867
 3600/25000 [===>..........................] - ETA: 13s - loss: 0.4204 - accuracy: 0.8868
 3700/25000 [===>..........................] - ETA: 13s - loss: 0.4203 - accuracy: 0.8867
 3800/25000 [===>..........................] - ETA: 13s - loss: 0.4204 - accuracy: 0.8868
 3900/25000 [===>..........................] - ETA: 12s - loss: 0.4202 - accuracy: 0.8870
 4000/25000 [===>..........................] - ETA: 12s - loss: 0.4204 - accuracy: 0.8868
 4100/25000 [===>..........................] - ETA: 12s - loss: 0.4199 - accuracy: 0.8871
 4200/25000 [====>.........................] - ETA: 12s - loss: 0.4196 - accuracy: 0.8873
 4300/25000 [====>.........................] - ETA: 12s - loss: 0.4195 - accuracy: 0.8874
 4400/25000 [====>.........................] - ETA: 12s - loss: 0.4192 - accuracy: 0.8875
 4500/25000 [====>.........................] - ETA: 12s - loss: 0.4191 - accuracy: 0.8874
 4600/25000 [====>.........................] - ETA: 12s - loss: 0.4190 - accuracy: 0.8874
 4700/25000 [====>.........................] - ETA: 12s - loss: 0.4188 - accuracy: 0.8873
 4800/25000 [====>.........................] - ETA: 12s - loss: 0.4186 - accuracy: 0.8874
 4900/25000 [====>.........................] - ETA: 12s - loss: 0.4186 - accuracy: 0.8873
 5000/25000 [=====>........................] - ETA: 12s - loss: 0.4186 - accuracy: 0.8873
 5100/25000 [=====>........................] - ETA: 12s - loss: 0.4185 - accuracy: 0.8873
 5200/25000 [=====>........................] - ETA: 12s - loss: 0.4183 - accuracy: 0.8873
 5300/25000 [=====>........................] - ETA: 12s - loss: 0.4182 - accuracy: 0.8873
 5400/25000 [=====>........................] - ETA: 12s - loss: 0.4180 - accuracy: 0.8874
 5500/25000 [=====>........................] - ETA: 12s - loss: 0.4176 - accuracy: 0.8876
 5600/25000 [=====>........................] - ETA: 11s - loss: 0.4176 - accuracy: 0.8874
 5700/25000 [=====>........................] - ETA: 11s - loss: 0.4175 - accuracy: 0.8874
 5800/25000 [=====>........................] - ETA: 11s - loss: 0.4174 - accuracy: 0.8874
 5900/25000 [======>.......................] - ETA: 11s - loss: 0.4172 - accuracy: 0.8876
 6000/25000 [======>.......................] - ETA: 11s - loss: 0.4170 - accuracy: 0.8876
 6100/25000 [======>.......................] - ETA: 11s - loss: 0.4170 - accuracy: 0.8876
 6200/25000 [======>.......................] - ETA: 11s - loss: 0.4169 - accuracy: 0.8876
 6300/25000 [======>.......................] - ETA: 11s - loss: 0.4167 - accuracy: 0.8878
 6400/25000 [======>.......................] - ETA: 11s - loss: 0.4165 - accuracy: 0.8878
 6500/25000 [======>.......................] - ETA: 11s - loss: 0.4165 - accuracy: 0.8877
 6600/25000 [======>.......................] - ETA: 11s - loss: 0.4163 - accuracy: 0.8877
 6700/25000 [=======>......................] - ETA: 11s - loss: 0.4163 - accuracy: 0.8878
 6800/25000 [=======>......................] - ETA: 11s - loss: 0.4162 - accuracy: 0.8877
 6900/25000 [=======>......................] - ETA: 11s - loss: 0.4163 - accuracy: 0.8875
 7000/25000 [=======>......................] - ETA: 11s - loss: 0.4162 - accuracy: 0.8874
 7100/25000 [=======>......................] - ETA: 11s - loss: 0.4163 - accuracy: 0.8872
 7200/25000 [=======>......................] - ETA: 10s - loss: 0.4164 - accuracy: 0.8871
 7300/25000 [=======>......................] - ETA: 10s - loss: 0.4162 - accuracy: 0.8871
 7400/25000 [=======>......................] - ETA: 10s - loss: 0.4162 - accuracy: 0.8871
 7500/25000 [========>.....................] - ETA: 10s - loss: 0.4162 - accuracy: 0.8869
 7600/25000 [========>.....................] - ETA: 10s - loss: 0.4160 - accuracy: 0.8871
 7700/25000 [========>.....................] - ETA: 10s - loss: 0.4160 - accuracy: 0.8870
 7800/25000 [========>.....................] - ETA: 10s - loss: 0.4160 - accuracy: 0.8870
 7900/25000 [========>.....................] - ETA: 10s - loss: 0.4159 - accuracy: 0.8870
 8000/25000 [========>.....................] - ETA: 10s - loss: 0.4159 - accuracy: 0.8869
 8100/25000 [========>.....................] - ETA: 10s - loss: 0.4159 - accuracy: 0.8868
 8200/25000 [========>.....................] - ETA: 10s - loss: 0.4158 - accuracy: 0.8868
 8300/25000 [========>.....................] - ETA: 10s - loss: 0.4158 - accuracy: 0.8868
 8400/25000 [=========>....................] - ETA: 10s - loss: 0.4157 - accuracy: 0.8868
 8500/25000 [=========>....................] - ETA: 10s - loss: 0.4155 - accuracy: 0.8869
 8600/25000 [=========>....................] - ETA: 10s - loss: 0.4154 - accuracy: 0.8869
 8700/25000 [=========>....................] - ETA: 10s - loss: 0.4153 - accuracy: 0.8869
 8800/25000 [=========>....................] - ETA: 10s - loss: 0.4152 - accuracy: 0.8868
 8900/25000 [=========>....................] - ETA: 9s - loss: 0.4150 - accuracy: 0.8869 
 9000/25000 [=========>....................] - ETA: 9s - loss: 0.4148 - accuracy: 0.8870
 9100/25000 [=========>....................] - ETA: 9s - loss: 0.4147 - accuracy: 0.8870
 9200/25000 [==========>...................] - ETA: 9s - loss: 0.4147 - accuracy: 0.8869
 9300/25000 [==========>...................] - ETA: 9s - loss: 0.4146 - accuracy: 0.8869
 9400/25000 [==========>...................] - ETA: 9s - loss: 0.4145 - accuracy: 0.8869
 9500/25000 [==========>...................] - ETA: 9s - loss: 0.4145 - accuracy: 0.8868
 9600/25000 [==========>...................] - ETA: 9s - loss: 0.4144 - accuracy: 0.8869
 9700/25000 [==========>...................] - ETA: 9s - loss: 0.4143 - accuracy: 0.8869
 9800/25000 [==========>...................] - ETA: 9s - loss: 0.4142 - accuracy: 0.8869
 9900/25000 [==========>...................] - ETA: 9s - loss: 0.4141 - accuracy: 0.8869
10000/25000 [===========>..................] - ETA: 9s - loss: 0.4141 - accuracy: 0.8868
10100/25000 [===========>..................] - ETA: 9s - loss: 0.4141 - accuracy: 0.8868
10200/25000 [===========>..................] - ETA: 9s - loss: 0.4140 - accuracy: 0.8868
10300/25000 [===========>..................] - ETA: 9s - loss: 0.4139 - accuracy: 0.8868
10400/25000 [===========>..................] - ETA: 9s - loss: 0.4138 - accuracy: 0.8868
10500/25000 [===========>..................] - ETA: 9s - loss: 0.4138 - accuracy: 0.8868
10600/25000 [===========>..................] - ETA: 9s - loss: 0.4138 - accuracy: 0.8867
10700/25000 [===========>..................] - ETA: 8s - loss: 0.4138 - accuracy: 0.8866
10800/25000 [===========>..................] - ETA: 8s - loss: 0.4138 - accuracy: 0.8866
10900/25000 [============>.................] - ETA: 8s - loss: 0.4136 - accuracy: 0.8867
11000/25000 [============>.................] - ETA: 8s - loss: 0.4136 - accuracy: 0.8866
11100/25000 [============>.................] - ETA: 8s - loss: 0.4135 - accuracy: 0.8866
11200/25000 [============>.................] - ETA: 8s - loss: 0.4135 - accuracy: 0.8865
11300/25000 [============>.................] - ETA: 8s - loss: 0.4135 - accuracy: 0.8864
11400/25000 [============>.................] - ETA: 8s - loss: 0.4135 - accuracy: 0.8864
11500/25000 [============>.................] - ETA: 8s - loss: 0.4133 - accuracy: 0.8865
11600/25000 [============>.................] - ETA: 8s - loss: 0.4133 - accuracy: 0.8865
11700/25000 [=============>................] - ETA: 8s - loss: 0.4131 - accuracy: 0.8866
11800/25000 [=============>................] - ETA: 8s - loss: 0.4130 - accuracy: 0.8865
11900/25000 [=============>................] - ETA: 8s - loss: 0.4130 - accuracy: 0.8864
12000/25000 [=============>................] - ETA: 8s - loss: 0.4129 - accuracy: 0.8864
12100/25000 [=============>................] - ETA: 8s - loss: 0.4128 - accuracy: 0.8864
12200/25000 [=============>................] - ETA: 8s - loss: 0.4128 - accuracy: 0.8864
12300/25000 [=============>................] - ETA: 7s - loss: 0.4127 - accuracy: 0.8864
12400/25000 [=============>................] - ETA: 7s - loss: 0.4126 - accuracy: 0.8864
12500/25000 [==============>...............] - ETA: 7s - loss: 0.4125 - accuracy: 0.8864
12600/25000 [==============>...............] - ETA: 7s - loss: 0.4125 - accuracy: 0.8864
12700/25000 [==============>...............] - ETA: 7s - loss: 0.4125 - accuracy: 0.8863
12800/25000 [==============>...............] - ETA: 7s - loss: 0.4125 - accuracy: 0.8863
12900/25000 [==============>...............] - ETA: 7s - loss: 0.4125 - accuracy: 0.8863
13000/25000 [==============>...............] - ETA: 7s - loss: 0.4124 - accuracy: 0.8862
13100/25000 [==============>...............] - ETA: 7s - loss: 0.4124 - accuracy: 0.8862
13200/25000 [==============>...............] - ETA: 7s - loss: 0.4123 - accuracy: 0.8863
13300/25000 [==============>...............] - ETA: 7s - loss: 0.4121 - accuracy: 0.8863
13400/25000 [===============>..............] - ETA: 7s - loss: 0.4120 - accuracy: 0.8863
13500/25000 [===============>..............] - ETA: 7s - loss: 0.4119 - accuracy: 0.8863
13600/25000 [===============>..............] - ETA: 7s - loss: 0.4118 - accuracy: 0.8863
13700/25000 [===============>..............] - ETA: 7s - loss: 0.4118 - accuracy: 0.8863
13800/25000 [===============>..............] - ETA: 7s - loss: 0.4116 - accuracy: 0.8864
13900/25000 [===============>..............] - ETA: 6s - loss: 0.4115 - accuracy: 0.8864
14000/25000 [===============>..............] - ETA: 6s - loss: 0.4113 - accuracy: 0.8865
14100/25000 [===============>..............] - ETA: 6s - loss: 0.4114 - accuracy: 0.8864
14200/25000 [================>.............] - ETA: 6s - loss: 0.4113 - accuracy: 0.8864
14300/25000 [================>.............] - ETA: 6s - loss: 0.4112 - accuracy: 0.8864
14400/25000 [================>.............] - ETA: 6s - loss: 0.4111 - accuracy: 0.8863
14500/25000 [================>.............] - ETA: 6s - loss: 0.4111 - accuracy: 0.8863
14600/25000 [================>.............] - ETA: 6s - loss: 0.4110 - accuracy: 0.8863
14700/25000 [================>.............] - ETA: 6s - loss: 0.4110 - accuracy: 0.8863
14800/25000 [================>.............] - ETA: 6s - loss: 0.4109 - accuracy: 0.8863
14900/25000 [================>.............] - ETA: 6s - loss: 0.4108 - accuracy: 0.8863
15000/25000 [=================>............] - ETA: 6s - loss: 0.4107 - accuracy: 0.8863
15100/25000 [=================>............] - ETA: 6s - loss: 0.4106 - accuracy: 0.8863
15200/25000 [=================>............] - ETA: 6s - loss: 0.4105 - accuracy: 0.8863
15300/25000 [=================>............] - ETA: 6s - loss: 0.4104 - accuracy: 0.8863
15400/25000 [=================>............] - ETA: 6s - loss: 0.4103 - accuracy: 0.8864
15500/25000 [=================>............] - ETA: 5s - loss: 0.4102 - accuracy: 0.8863
15600/25000 [=================>............] - ETA: 5s - loss: 0.4101 - accuracy: 0.8863
15700/25000 [=================>............] - ETA: 5s - loss: 0.4100 - accuracy: 0.8864
15800/25000 [=================>............] - ETA: 5s - loss: 0.4100 - accuracy: 0.8863
15900/25000 [==================>...........] - ETA: 5s - loss: 0.4099 - accuracy: 0.8864
16000/25000 [==================>...........] - ETA: 5s - loss: 0.4099 - accuracy: 0.8863
16100/25000 [==================>...........] - ETA: 5s - loss: 0.4100 - accuracy: 0.8862
16200/25000 [==================>...........] - ETA: 5s - loss: 0.4099 - accuracy: 0.8862
16300/25000 [==================>...........] - ETA: 5s - loss: 0.4098 - accuracy: 0.8863
16400/25000 [==================>...........] - ETA: 5s - loss: 0.4097 - accuracy: 0.8862
16500/25000 [==================>...........] - ETA: 5s - loss: 0.4097 - accuracy: 0.8863
16600/25000 [==================>...........] - ETA: 5s - loss: 0.4096 - accuracy: 0.8862
16700/25000 [===================>..........] - ETA: 5s - loss: 0.4095 - accuracy: 0.8863
16800/25000 [===================>..........] - ETA: 5s - loss: 0.4094 - accuracy: 0.8863
16900/25000 [===================>..........] - ETA: 5s - loss: 0.4093 - accuracy: 0.8863
17000/25000 [===================>..........] - ETA: 5s - loss: 0.4093 - accuracy: 0.8863
17100/25000 [===================>..........] - ETA: 4s - loss: 0.4093 - accuracy: 0.8862
17200/25000 [===================>..........] - ETA: 4s - loss: 0.4092 - accuracy: 0.8863
17300/25000 [===================>..........] - ETA: 4s - loss: 0.4090 - accuracy: 0.8864
17400/25000 [===================>..........] - ETA: 4s - loss: 0.4089 - accuracy: 0.8864
17500/25000 [====================>.........] - ETA: 4s - loss: 0.4089 - accuracy: 0.8863
17600/25000 [====================>.........] - ETA: 4s - loss: 0.4088 - accuracy: 0.8864
17700/25000 [====================>.........] - ETA: 4s - loss: 0.4087 - accuracy: 0.8863
17800/25000 [====================>.........] - ETA: 4s - loss: 0.4086 - accuracy: 0.8863
17900/25000 [====================>.........] - ETA: 4s - loss: 0.4085 - accuracy: 0.8863
18000/25000 [====================>.........] - ETA: 4s - loss: 0.4085 - accuracy: 0.8863
18100/25000 [====================>.........] - ETA: 4s - loss: 0.4085 - accuracy: 0.8863
18200/25000 [====================>.........] - ETA: 4s - loss: 0.4083 - accuracy: 0.8863
18300/25000 [====================>.........] - ETA: 4s - loss: 0.4082 - accuracy: 0.8864
18400/25000 [=====================>........] - ETA: 4s - loss: 0.4081 - accuracy: 0.8864
18500/25000 [=====================>........] - ETA: 4s - loss: 0.4080 - accuracy: 0.8864
18600/25000 [=====================>........] - ETA: 3s - loss: 0.4079 - accuracy: 0.8865
18700/25000 [=====================>........] - ETA: 3s - loss: 0.4077 - accuracy: 0.8866
18800/25000 [=====================>........] - ETA: 3s - loss: 0.4077 - accuracy: 0.8865
18900/25000 [=====================>........] - ETA: 3s - loss: 0.4076 - accuracy: 0.8865
19000/25000 [=====================>........] - ETA: 3s - loss: 0.4076 - accuracy: 0.8865
19100/25000 [=====================>........] - ETA: 3s - loss: 0.4074 - accuracy: 0.8865
19200/25000 [======================>.......] - ETA: 3s - loss: 0.4074 - accuracy: 0.8865
19300/25000 [======================>.......] - ETA: 3s - loss: 0.4073 - accuracy: 0.8865
19400/25000 [======================>.......] - ETA: 3s - loss: 0.4073 - accuracy: 0.8865
19500/25000 [======================>.......] - ETA: 3s - loss: 0.4072 - accuracy: 0.8865
19600/25000 [======================>.......] - ETA: 3s - loss: 0.4072 - accuracy: 0.8865
19700/25000 [======================>.......] - ETA: 3s - loss: 0.4070 - accuracy: 0.8865
19800/25000 [======================>.......] - ETA: 3s - loss: 0.4069 - accuracy: 0.8865
19900/25000 [======================>.......] - ETA: 3s - loss: 0.4070 - accuracy: 0.8865
20000/25000 [=======================>......] - ETA: 3s - loss: 0.4069 - accuracy: 0.8865
20100/25000 [=======================>......] - ETA: 3s - loss: 0.4068 - accuracy: 0.8865
20200/25000 [=======================>......] - ETA: 2s - loss: 0.4067 - accuracy: 0.8865
20300/25000 [=======================>......] - ETA: 2s - loss: 0.4066 - accuracy: 0.8865
20400/25000 [=======================>......] - ETA: 2s - loss: 0.4065 - accuracy: 0.8865
20500/25000 [=======================>......] - ETA: 2s - loss: 0.4064 - accuracy: 0.8865
20600/25000 [=======================>......] - ETA: 2s - loss: 0.4063 - accuracy: 0.8866
20700/25000 [=======================>......] - ETA: 2s - loss: 0.4062 - accuracy: 0.8865
20800/25000 [=======================>......] - ETA: 2s - loss: 0.4062 - accuracy: 0.8865
20900/25000 [========================>.....] - ETA: 2s - loss: 0.4061 - accuracy: 0.8865
21000/25000 [========================>.....] - ETA: 2s - loss: 0.4062 - accuracy: 0.8864
21100/25000 [========================>.....] - ETA: 2s - loss: 0.4061 - accuracy: 0.8864
21200/25000 [========================>.....] - ETA: 2s - loss: 0.4060 - accuracy: 0.8864
21300/25000 [========================>.....] - ETA: 2s - loss: 0.4059 - accuracy: 0.8864
21400/25000 [========================>.....] - ETA: 2s - loss: 0.4059 - accuracy: 0.8864
21500/25000 [========================>.....] - ETA: 2s - loss: 0.4059 - accuracy: 0.8864
21600/25000 [========================>.....] - ETA: 2s - loss: 0.4058 - accuracy: 0.8863
21700/25000 [=========================>....] - ETA: 2s - loss: 0.4057 - accuracy: 0.8864
21800/25000 [=========================>....] - ETA: 1s - loss: 0.4057 - accuracy: 0.8863
21900/25000 [=========================>....] - ETA: 1s - loss: 0.4056 - accuracy: 0.8864
22000/25000 [=========================>....] - ETA: 1s - loss: 0.4056 - accuracy: 0.8864
22100/25000 [=========================>....] - ETA: 1s - loss: 0.4055 - accuracy: 0.8863
22200/25000 [=========================>....] - ETA: 1s - loss: 0.4054 - accuracy: 0.8863
22300/25000 [=========================>....] - ETA: 1s - loss: 0.4053 - accuracy: 0.8863
22400/25000 [=========================>....] - ETA: 1s - loss: 0.4052 - accuracy: 0.8864
22500/25000 [==========================>...] - ETA: 1s - loss: 0.4051 - accuracy: 0.8864
22600/25000 [==========================>...] - ETA: 1s - loss: 0.4051 - accuracy: 0.8864
22700/25000 [==========================>...] - ETA: 1s - loss: 0.4050 - accuracy: 0.8864
22800/25000 [==========================>...] - ETA: 1s - loss: 0.4049 - accuracy: 0.8864
22900/25000 [==========================>...] - ETA: 1s - loss: 0.4049 - accuracy: 0.8864
23000/25000 [==========================>...] - ETA: 1s - loss: 0.4048 - accuracy: 0.8864
23100/25000 [==========================>...] - ETA: 1s - loss: 0.4047 - accuracy: 0.8864
23200/25000 [==========================>...] - ETA: 1s - loss: 0.4046 - accuracy: 0.8864
23300/25000 [==========================>...] - ETA: 1s - loss: 0.4045 - accuracy: 0.8864
23400/25000 [===========================>..] - ETA: 1s - loss: 0.4045 - accuracy: 0.8864
23500/25000 [===========================>..] - ETA: 0s - loss: 0.4045 - accuracy: 0.8864
23600/25000 [===========================>..] - ETA: 0s - loss: 0.4044 - accuracy: 0.8864
23700/25000 [===========================>..] - ETA: 0s - loss: 0.4044 - accuracy: 0.8864
23800/25000 [===========================>..] - ETA: 0s - loss: 0.4043 - accuracy: 0.8864
23900/25000 [===========================>..] - ETA: 0s - loss: 0.4043 - accuracy: 0.8864
24000/25000 [===========================>..] - ETA: 0s - loss: 0.4042 - accuracy: 0.8864
24100/25000 [===========================>..] - ETA: 0s - loss: 0.4042 - accuracy: 0.8864
24200/25000 [============================>.] - ETA: 0s - loss: 0.4041 - accuracy: 0.8863
24300/25000 [============================>.] - ETA: 0s - loss: 0.4040 - accuracy: 0.8864
24400/25000 [============================>.] - ETA: 0s - loss: 0.4039 - accuracy: 0.8864
24500/25000 [============================>.] - ETA: 0s - loss: 0.4039 - accuracy: 0.8864
24600/25000 [============================>.] - ETA: 0s - loss: 0.4039 - accuracy: 0.8863
24700/25000 [============================>.] - ETA: 0s - loss: 0.4039 - accuracy: 0.8863
24800/25000 [============================>.] - ETA: 0s - loss: 0.4038 - accuracy: 0.8863
24900/25000 [============================>.] - ETA: 0s - loss: 0.4037 - accuracy: 0.8863
25000/25000 [==============================] - 20s 790us/step - loss: 0.4037 - accuracy: 0.8863 - val_loss: 0.3863 - val_accuracy: 0.8858
Epoch 4/10

  100/25000 [..............................] - ETA: 15s - loss: 0.3942 - accuracy: 0.8814
  200/25000 [..............................] - ETA: 15s - loss: 0.3892 - accuracy: 0.8850
  300/25000 [..............................] - ETA: 15s - loss: 0.3859 - accuracy: 0.8862
  400/25000 [..............................] - ETA: 15s - loss: 0.3858 - accuracy: 0.8854
  500/25000 [..............................] - ETA: 14s - loss: 0.3864 - accuracy: 0.8857
  600/25000 [..............................] - ETA: 14s - loss: 0.3862 - accuracy: 0.8857
  700/25000 [..............................] - ETA: 14s - loss: 0.3864 - accuracy: 0.8863
  800/25000 [..............................] - ETA: 14s - loss: 0.3854 - accuracy: 0.8870
  900/25000 [>.............................] - ETA: 14s - loss: 0.3862 - accuracy: 0.8860
 1000/25000 [>.............................] - ETA: 14s - loss: 0.3859 - accuracy: 0.8863
 1100/25000 [>.............................] - ETA: 14s - loss: 0.3856 - accuracy: 0.8862
 1200/25000 [>.............................] - ETA: 14s - loss: 0.3851 - accuracy: 0.8867
 1300/25000 [>.............................] - ETA: 14s - loss: 0.3857 - accuracy: 0.8865
 1400/25000 [>.............................] - ETA: 14s - loss: 0.3862 - accuracy: 0.8862
 1500/25000 [>.............................] - ETA: 14s - loss: 0.3861 - accuracy: 0.8866
 1600/25000 [>.............................] - ETA: 14s - loss: 0.3860 - accuracy: 0.8865
 1700/25000 [=>............................] - ETA: 14s - loss: 0.3860 - accuracy: 0.8865
 1800/25000 [=>............................] - ETA: 14s - loss: 0.3864 - accuracy: 0.8863
 1900/25000 [=>............................] - ETA: 14s - loss: 0.3858 - accuracy: 0.8867
 2000/25000 [=>............................] - ETA: 14s - loss: 0.3852 - accuracy: 0.8870
 2100/25000 [=>............................] - ETA: 14s - loss: 0.3854 - accuracy: 0.8870
 2200/25000 [=>............................] - ETA: 14s - loss: 0.3854 - accuracy: 0.8870
 2300/25000 [=>............................] - ETA: 14s - loss: 0.3853 - accuracy: 0.8870
 2400/25000 [=>............................] - ETA: 13s - loss: 0.3851 - accuracy: 0.8870
 2500/25000 [==>...........................] - ETA: 13s - loss: 0.3845 - accuracy: 0.8874
 2600/25000 [==>...........................] - ETA: 13s - loss: 0.3848 - accuracy: 0.8872
 2700/25000 [==>...........................] - ETA: 13s - loss: 0.3852 - accuracy: 0.8869
 2800/25000 [==>...........................] - ETA: 13s - loss: 0.3850 - accuracy: 0.8870
 2900/25000 [==>...........................] - ETA: 13s - loss: 0.3850 - accuracy: 0.8869
 3000/25000 [==>...........................] - ETA: 13s - loss: 0.3853 - accuracy: 0.8867
 3100/25000 [==>...........................] - ETA: 13s - loss: 0.3851 - accuracy: 0.8870
 3200/25000 [==>...........................] - ETA: 13s - loss: 0.3852 - accuracy: 0.8869
 3300/25000 [==>...........................] - ETA: 13s - loss: 0.3850 - accuracy: 0.8871
 3400/25000 [===>..........................] - ETA: 13s - loss: 0.3850 - accuracy: 0.8871
 3500/25000 [===>..........................] - ETA: 13s - loss: 0.3847 - accuracy: 0.8871
 3600/25000 [===>..........................] - ETA: 13s - loss: 0.3848 - accuracy: 0.8870
 3700/25000 [===>..........................] - ETA: 13s - loss: 0.3845 - accuracy: 0.8872
 3800/25000 [===>..........................] - ETA: 13s - loss: 0.3846 - accuracy: 0.8872
 3900/25000 [===>..........................] - ETA: 13s - loss: 0.3843 - accuracy: 0.8874
 4000/25000 [===>..........................] - ETA: 13s - loss: 0.3840 - accuracy: 0.8875
 4100/25000 [===>..........................] - ETA: 13s - loss: 0.3840 - accuracy: 0.8875
 4200/25000 [====>.........................] - ETA: 12s - loss: 0.3840 - accuracy: 0.8874
 4300/25000 [====>.........................] - ETA: 12s - loss: 0.3839 - accuracy: 0.8875
 4400/25000 [====>.........................] - ETA: 12s - loss: 0.3839 - accuracy: 0.8874
 4500/25000 [====>.........................] - ETA: 12s - loss: 0.3839 - accuracy: 0.8874
 4600/25000 [====>.........................] - ETA: 12s - loss: 0.3840 - accuracy: 0.8873
 4700/25000 [====>.........................] - ETA: 12s - loss: 0.3839 - accuracy: 0.8872
 4800/25000 [====>.........................] - ETA: 12s - loss: 0.3842 - accuracy: 0.8870
 4900/25000 [====>.........................] - ETA: 12s - loss: 0.3841 - accuracy: 0.8869
 5000/25000 [=====>........................] - ETA: 12s - loss: 0.3837 - accuracy: 0.8871
 5100/25000 [=====>........................] - ETA: 12s - loss: 0.3837 - accuracy: 0.8871
 5200/25000 [=====>........................] - ETA: 12s - loss: 0.3835 - accuracy: 0.8872
 5300/25000 [=====>........................] - ETA: 12s - loss: 0.3835 - accuracy: 0.8871
 5400/25000 [=====>........................] - ETA: 12s - loss: 0.3837 - accuracy: 0.8869
 5500/25000 [=====>........................] - ETA: 12s - loss: 0.3834 - accuracy: 0.8871
 5600/25000 [=====>........................] - ETA: 12s - loss: 0.3832 - accuracy: 0.8871
 5700/25000 [=====>........................] - ETA: 11s - loss: 0.3832 - accuracy: 0.8870
 5800/25000 [=====>........................] - ETA: 11s - loss: 0.3830 - accuracy: 0.8871
 5900/25000 [======>.......................] - ETA: 11s - loss: 0.3830 - accuracy: 0.8869
 6000/25000 [======>.......................] - ETA: 11s - loss: 0.3830 - accuracy: 0.8870
 6100/25000 [======>.......................] - ETA: 11s - loss: 0.3830 - accuracy: 0.8869
 6200/25000 [======>.......................] - ETA: 11s - loss: 0.3830 - accuracy: 0.8868
 6300/25000 [======>.......................] - ETA: 11s - loss: 0.3830 - accuracy: 0.8869
 6400/25000 [======>.......................] - ETA: 11s - loss: 0.3831 - accuracy: 0.8868
 6500/25000 [======>.......................] - ETA: 11s - loss: 0.3834 - accuracy: 0.8866
 6600/25000 [======>.......................] - ETA: 11s - loss: 0.3833 - accuracy: 0.8866
 6700/25000 [=======>......................] - ETA: 11s - loss: 0.3834 - accuracy: 0.8865
 6800/25000 [=======>......................] - ETA: 11s - loss: 0.3834 - accuracy: 0.8865
 6900/25000 [=======>......................] - ETA: 11s - loss: 0.3834 - accuracy: 0.8865
 7000/25000 [=======>......................] - ETA: 11s - loss: 0.3833 - accuracy: 0.8865
 7100/25000 [=======>......................] - ETA: 11s - loss: 0.3831 - accuracy: 0.8865
 7200/25000 [=======>......................] - ETA: 11s - loss: 0.3833 - accuracy: 0.8865
 7300/25000 [=======>......................] - ETA: 10s - loss: 0.3832 - accuracy: 0.8865
 7400/25000 [=======>......................] - ETA: 10s - loss: 0.3830 - accuracy: 0.8865
 7500/25000 [========>.....................] - ETA: 10s - loss: 0.3830 - accuracy: 0.8865
 7600/25000 [========>.....................] - ETA: 10s - loss: 0.3830 - accuracy: 0.8865
 7700/25000 [========>.....................] - ETA: 10s - loss: 0.3829 - accuracy: 0.8865
 7800/25000 [========>.....................] - ETA: 10s - loss: 0.3829 - accuracy: 0.8865
 7900/25000 [========>.....................] - ETA: 10s - loss: 0.3829 - accuracy: 0.8865
 8000/25000 [========>.....................] - ETA: 10s - loss: 0.3829 - accuracy: 0.8864
 8100/25000 [========>.....................] - ETA: 10s - loss: 0.3830 - accuracy: 0.8863
 8200/25000 [========>.....................] - ETA: 10s - loss: 0.3829 - accuracy: 0.8864
 8300/25000 [========>.....................] - ETA: 10s - loss: 0.3829 - accuracy: 0.8864
 8400/25000 [=========>....................] - ETA: 10s - loss: 0.3828 - accuracy: 0.8864
 8500/25000 [=========>....................] - ETA: 10s - loss: 0.3828 - accuracy: 0.8864
 8600/25000 [=========>....................] - ETA: 10s - loss: 0.3826 - accuracy: 0.8864
 8700/25000 [=========>....................] - ETA: 10s - loss: 0.3827 - accuracy: 0.8864
 8800/25000 [=========>....................] - ETA: 10s - loss: 0.3828 - accuracy: 0.8863
 8900/25000 [=========>....................] - ETA: 9s - loss: 0.3827 - accuracy: 0.8863 
 9000/25000 [=========>....................] - ETA: 9s - loss: 0.3828 - accuracy: 0.8863
 9100/25000 [=========>....................] - ETA: 9s - loss: 0.3826 - accuracy: 0.8864
 9200/25000 [==========>...................] - ETA: 9s - loss: 0.3826 - accuracy: 0.8864
 9300/25000 [==========>...................] - ETA: 9s - loss: 0.3823 - accuracy: 0.8865
 9400/25000 [==========>...................] - ETA: 9s - loss: 0.3823 - accuracy: 0.8866
 9500/25000 [==========>...................] - ETA: 9s - loss: 0.3822 - accuracy: 0.8866
 9600/25000 [==========>...................] - ETA: 9s - loss: 0.3823 - accuracy: 0.8865
 9700/25000 [==========>...................] - ETA: 9s - loss: 0.3823 - accuracy: 0.8865
 9800/25000 [==========>...................] - ETA: 9s - loss: 0.3824 - accuracy: 0.8864
 9900/25000 [==========>...................] - ETA: 9s - loss: 0.3823 - accuracy: 0.8865
10000/25000 [===========>..................] - ETA: 9s - loss: 0.3823 - accuracy: 0.8865
10100/25000 [===========>..................] - ETA: 9s - loss: 0.3822 - accuracy: 0.8865
10200/25000 [===========>..................] - ETA: 9s - loss: 0.3821 - accuracy: 0.8865
10300/25000 [===========>..................] - ETA: 9s - loss: 0.3820 - accuracy: 0.8866
10400/25000 [===========>..................] - ETA: 9s - loss: 0.3819 - accuracy: 0.8866
10500/25000 [===========>..................] - ETA: 8s - loss: 0.3818 - accuracy: 0.8866
10600/25000 [===========>..................] - ETA: 8s - loss: 0.3818 - accuracy: 0.8866
10700/25000 [===========>..................] - ETA: 8s - loss: 0.3818 - accuracy: 0.8866
10800/25000 [===========>..................] - ETA: 8s - loss: 0.3817 - accuracy: 0.8866
10900/25000 [============>.................] - ETA: 8s - loss: 0.3816 - accuracy: 0.8866
11000/25000 [============>.................] - ETA: 8s - loss: 0.3816 - accuracy: 0.8866
11100/25000 [============>.................] - ETA: 8s - loss: 0.3817 - accuracy: 0.8866
11200/25000 [============>.................] - ETA: 8s - loss: 0.3815 - accuracy: 0.8866
11300/25000 [============>.................] - ETA: 8s - loss: 0.3816 - accuracy: 0.8865
11400/25000 [============>.................] - ETA: 8s - loss: 0.3815 - accuracy: 0.8866
11500/25000 [============>.................] - ETA: 8s - loss: 0.3814 - accuracy: 0.8867
11600/25000 [============>.................] - ETA: 8s - loss: 0.3814 - accuracy: 0.8867
11700/25000 [=============>................] - ETA: 8s - loss: 0.3814 - accuracy: 0.8866
11800/25000 [=============>................] - ETA: 8s - loss: 0.3813 - accuracy: 0.8866
11900/25000 [=============>................] - ETA: 8s - loss: 0.3813 - accuracy: 0.8866
12000/25000 [=============>................] - ETA: 8s - loss: 0.3813 - accuracy: 0.8866
12100/25000 [=============>................] - ETA: 8s - loss: 0.3813 - accuracy: 0.8866
12200/25000 [=============>................] - ETA: 7s - loss: 0.3813 - accuracy: 0.8865
12300/25000 [=============>................] - ETA: 7s - loss: 0.3811 - accuracy: 0.8866
12400/25000 [=============>................] - ETA: 7s - loss: 0.3811 - accuracy: 0.8866
12500/25000 [==============>...............] - ETA: 7s - loss: 0.3810 - accuracy: 0.8866
12600/25000 [==============>...............] - ETA: 7s - loss: 0.3811 - accuracy: 0.8866
12700/25000 [==============>...............] - ETA: 7s - loss: 0.3810 - accuracy: 0.8866
12800/25000 [==============>...............] - ETA: 7s - loss: 0.3810 - accuracy: 0.8866
12900/25000 [==============>...............] - ETA: 7s - loss: 0.3810 - accuracy: 0.8866
13000/25000 [==============>...............] - ETA: 7s - loss: 0.3809 - accuracy: 0.8865
13100/25000 [==============>...............] - ETA: 7s - loss: 0.3810 - accuracy: 0.8865
13200/25000 [==============>...............] - ETA: 7s - loss: 0.3809 - accuracy: 0.8865
13300/25000 [==============>...............] - ETA: 7s - loss: 0.3808 - accuracy: 0.8865
13400/25000 [===============>..............] - ETA: 7s - loss: 0.3809 - accuracy: 0.8865
13500/25000 [===============>..............] - ETA: 7s - loss: 0.3808 - accuracy: 0.8865
13600/25000 [===============>..............] - ETA: 7s - loss: 0.3807 - accuracy: 0.8865
13700/25000 [===============>..............] - ETA: 7s - loss: 0.3807 - accuracy: 0.8865
13800/25000 [===============>..............] - ETA: 6s - loss: 0.3806 - accuracy: 0.8865
13900/25000 [===============>..............] - ETA: 6s - loss: 0.3806 - accuracy: 0.8865
14000/25000 [===============>..............] - ETA: 6s - loss: 0.3804 - accuracy: 0.8866
14100/25000 [===============>..............] - ETA: 6s - loss: 0.3803 - accuracy: 0.8866
14200/25000 [================>.............] - ETA: 6s - loss: 0.3803 - accuracy: 0.8866
14300/25000 [================>.............] - ETA: 6s - loss: 0.3804 - accuracy: 0.8865
14400/25000 [================>.............] - ETA: 6s - loss: 0.3803 - accuracy: 0.8864
14500/25000 [================>.............] - ETA: 6s - loss: 0.3802 - accuracy: 0.8865
14600/25000 [================>.............] - ETA: 6s - loss: 0.3801 - accuracy: 0.8865
14700/25000 [================>.............] - ETA: 6s - loss: 0.3800 - accuracy: 0.8865
14800/25000 [================>.............] - ETA: 6s - loss: 0.3800 - accuracy: 0.8865
14900/25000 [================>.............] - ETA: 6s - loss: 0.3799 - accuracy: 0.8865
15000/25000 [=================>............] - ETA: 6s - loss: 0.3798 - accuracy: 0.8866
15100/25000 [=================>............] - ETA: 6s - loss: 0.3798 - accuracy: 0.8865
15200/25000 [=================>............] - ETA: 6s - loss: 0.3798 - accuracy: 0.8865
15300/25000 [=================>............] - ETA: 6s - loss: 0.3797 - accuracy: 0.8865
15400/25000 [=================>............] - ETA: 5s - loss: 0.3796 - accuracy: 0.8865
15500/25000 [=================>............] - ETA: 5s - loss: 0.3795 - accuracy: 0.8865
15600/25000 [=================>............] - ETA: 5s - loss: 0.3795 - accuracy: 0.8865
15700/25000 [=================>............] - ETA: 5s - loss: 0.3794 - accuracy: 0.8866
15800/25000 [=================>............] - ETA: 5s - loss: 0.3794 - accuracy: 0.8866
15900/25000 [==================>...........] - ETA: 5s - loss: 0.3794 - accuracy: 0.8866
16000/25000 [==================>...........] - ETA: 5s - loss: 0.3793 - accuracy: 0.8866
16100/25000 [==================>...........] - ETA: 5s - loss: 0.3793 - accuracy: 0.8866
16200/25000 [==================>...........] - ETA: 5s - loss: 0.3792 - accuracy: 0.8866
16300/25000 [==================>...........] - ETA: 5s - loss: 0.3793 - accuracy: 0.8865
16400/25000 [==================>...........] - ETA: 5s - loss: 0.3792 - accuracy: 0.8865
16500/25000 [==================>...........] - ETA: 5s - loss: 0.3791 - accuracy: 0.8866
16600/25000 [==================>...........] - ETA: 5s - loss: 0.3790 - accuracy: 0.8866
16700/25000 [===================>..........] - ETA: 5s - loss: 0.3790 - accuracy: 0.8866
16800/25000 [===================>..........] - ETA: 5s - loss: 0.3790 - accuracy: 0.8866
16900/25000 [===================>..........] - ETA: 5s - loss: 0.3789 - accuracy: 0.8866
17000/25000 [===================>..........] - ETA: 4s - loss: 0.3789 - accuracy: 0.8866
17100/25000 [===================>..........] - ETA: 4s - loss: 0.3789 - accuracy: 0.8866
17200/25000 [===================>..........] - ETA: 4s - loss: 0.3790 - accuracy: 0.8865
17300/25000 [===================>..........] - ETA: 4s - loss: 0.3788 - accuracy: 0.8866
17400/25000 [===================>..........] - ETA: 4s - loss: 0.3788 - accuracy: 0.8866
17500/25000 [====================>.........] - ETA: 4s - loss: 0.3788 - accuracy: 0.8865
17600/25000 [====================>.........] - ETA: 4s - loss: 0.3788 - accuracy: 0.8865
17700/25000 [====================>.........] - ETA: 4s - loss: 0.3788 - accuracy: 0.8865
17800/25000 [====================>.........] - ETA: 4s - loss: 0.3788 - accuracy: 0.8865
17900/25000 [====================>.........] - ETA: 4s - loss: 0.3788 - accuracy: 0.8865
18000/25000 [====================>.........] - ETA: 4s - loss: 0.3788 - accuracy: 0.8864
18100/25000 [====================>.........] - ETA: 4s - loss: 0.3788 - accuracy: 0.8864
18200/25000 [====================>.........] - ETA: 4s - loss: 0.3788 - accuracy: 0.8864
18300/25000 [====================>.........] - ETA: 4s - loss: 0.3787 - accuracy: 0.8865
18400/25000 [=====================>........] - ETA: 4s - loss: 0.3787 - accuracy: 0.8865
18500/25000 [=====================>........] - ETA: 4s - loss: 0.3786 - accuracy: 0.8865
18600/25000 [=====================>........] - ETA: 3s - loss: 0.3785 - accuracy: 0.8865
18700/25000 [=====================>........] - ETA: 3s - loss: 0.3785 - accuracy: 0.8865
18800/25000 [=====================>........] - ETA: 3s - loss: 0.3784 - accuracy: 0.8865
18900/25000 [=====================>........] - ETA: 3s - loss: 0.3784 - accuracy: 0.8865
19000/25000 [=====================>........] - ETA: 3s - loss: 0.3784 - accuracy: 0.8865
19100/25000 [=====================>........] - ETA: 3s - loss: 0.3784 - accuracy: 0.8864
19200/25000 [======================>.......] - ETA: 3s - loss: 0.3784 - accuracy: 0.8864
19300/25000 [======================>.......] - ETA: 3s - loss: 0.3784 - accuracy: 0.8864
19400/25000 [======================>.......] - ETA: 3s - loss: 0.3783 - accuracy: 0.8864
19500/25000 [======================>.......] - ETA: 3s - loss: 0.3782 - accuracy: 0.8864
19600/25000 [======================>.......] - ETA: 3s - loss: 0.3782 - accuracy: 0.8864
19700/25000 [======================>.......] - ETA: 3s - loss: 0.3782 - accuracy: 0.8864
19800/25000 [======================>.......] - ETA: 3s - loss: 0.3782 - accuracy: 0.8864
19900/25000 [======================>.......] - ETA: 3s - loss: 0.3782 - accuracy: 0.8864
20000/25000 [=======================>......] - ETA: 3s - loss: 0.3782 - accuracy: 0.8863
20100/25000 [=======================>......] - ETA: 3s - loss: 0.3782 - accuracy: 0.8863
20200/25000 [=======================>......] - ETA: 2s - loss: 0.3782 - accuracy: 0.8863
20300/25000 [=======================>......] - ETA: 2s - loss: 0.3781 - accuracy: 0.8863
20400/25000 [=======================>......] - ETA: 2s - loss: 0.3780 - accuracy: 0.8863
20500/25000 [=======================>......] - ETA: 2s - loss: 0.3780 - accuracy: 0.8863
20600/25000 [=======================>......] - ETA: 2s - loss: 0.3780 - accuracy: 0.8863
20700/25000 [=======================>......] - ETA: 2s - loss: 0.3780 - accuracy: 0.8863
20800/25000 [=======================>......] - ETA: 2s - loss: 0.3779 - accuracy: 0.8863
20900/25000 [========================>.....] - ETA: 2s - loss: 0.3778 - accuracy: 0.8863
21000/25000 [========================>.....] - ETA: 2s - loss: 0.3778 - accuracy: 0.8863
21100/25000 [========================>.....] - ETA: 2s - loss: 0.3779 - accuracy: 0.8863
21200/25000 [========================>.....] - ETA: 2s - loss: 0.3779 - accuracy: 0.8862
21300/25000 [========================>.....] - ETA: 2s - loss: 0.3778 - accuracy: 0.8862
21400/25000 [========================>.....] - ETA: 2s - loss: 0.3777 - accuracy: 0.8863
21500/25000 [========================>.....] - ETA: 2s - loss: 0.3776 - accuracy: 0.8863
21600/25000 [========================>.....] - ETA: 2s - loss: 0.3776 - accuracy: 0.8863
21700/25000 [=========================>....] - ETA: 2s - loss: 0.3775 - accuracy: 0.8863
21800/25000 [=========================>....] - ETA: 1s - loss: 0.3775 - accuracy: 0.8863
21900/25000 [=========================>....] - ETA: 1s - loss: 0.3775 - accuracy: 0.8862
22000/25000 [=========================>....] - ETA: 1s - loss: 0.3774 - accuracy: 0.8863
22100/25000 [=========================>....] - ETA: 1s - loss: 0.3773 - accuracy: 0.8863
22200/25000 [=========================>....] - ETA: 1s - loss: 0.3773 - accuracy: 0.8863
22300/25000 [=========================>....] - ETA: 1s - loss: 0.3773 - accuracy: 0.8863
22400/25000 [=========================>....] - ETA: 1s - loss: 0.3773 - accuracy: 0.8862
22500/25000 [==========================>...] - ETA: 1s - loss: 0.3773 - accuracy: 0.8862
22600/25000 [==========================>...] - ETA: 1s - loss: 0.3773 - accuracy: 0.8862
22700/25000 [==========================>...] - ETA: 1s - loss: 0.3772 - accuracy: 0.8863
22800/25000 [==========================>...] - ETA: 1s - loss: 0.3771 - accuracy: 0.8863
22900/25000 [==========================>...] - ETA: 1s - loss: 0.3771 - accuracy: 0.8863
23000/25000 [==========================>...] - ETA: 1s - loss: 0.3771 - accuracy: 0.8862
23100/25000 [==========================>...] - ETA: 1s - loss: 0.3770 - accuracy: 0.8863
23200/25000 [==========================>...] - ETA: 1s - loss: 0.3770 - accuracy: 0.8863
23300/25000 [==========================>...] - ETA: 1s - loss: 0.3769 - accuracy: 0.8863
23400/25000 [===========================>..] - ETA: 0s - loss: 0.3768 - accuracy: 0.8863
23500/25000 [===========================>..] - ETA: 0s - loss: 0.3768 - accuracy: 0.8863
23600/25000 [===========================>..] - ETA: 0s - loss: 0.3767 - accuracy: 0.8863
23700/25000 [===========================>..] - ETA: 0s - loss: 0.3767 - accuracy: 0.8863
23800/25000 [===========================>..] - ETA: 0s - loss: 0.3767 - accuracy: 0.8863
23900/25000 [===========================>..] - ETA: 0s - loss: 0.3767 - accuracy: 0.8863
24000/25000 [===========================>..] - ETA: 0s - loss: 0.3767 - accuracy: 0.8863
24100/25000 [===========================>..] - ETA: 0s - loss: 0.3766 - accuracy: 0.8863
24200/25000 [============================>.] - ETA: 0s - loss: 0.3766 - accuracy: 0.8863
24300/25000 [============================>.] - ETA: 0s - loss: 0.3765 - accuracy: 0.8864
24400/25000 [============================>.] - ETA: 0s - loss: 0.3764 - accuracy: 0.8863
24500/25000 [============================>.] - ETA: 0s - loss: 0.3764 - accuracy: 0.8863
24600/25000 [============================>.] - ETA: 0s - loss: 0.3764 - accuracy: 0.8863
24700/25000 [============================>.] - ETA: 0s - loss: 0.3764 - accuracy: 0.8863
24800/25000 [============================>.] - ETA: 0s - loss: 0.3764 - accuracy: 0.8863
24900/25000 [============================>.] - ETA: 0s - loss: 0.3764 - accuracy: 0.8863
25000/25000 [==============================] - 20s 789us/step - loss: 0.3764 - accuracy: 0.8863 - val_loss: 0.3674 - val_accuracy: 0.8858
Epoch 5/10

  100/25000 [..............................] - ETA: 15s - loss: 0.3636 - accuracy: 0.8900
  200/25000 [..............................] - ETA: 15s - loss: 0.3530 - accuracy: 0.8950
  300/25000 [..............................] - ETA: 15s - loss: 0.3572 - accuracy: 0.8938
  400/25000 [..............................] - ETA: 15s - loss: 0.3561 - accuracy: 0.8936
  500/25000 [..............................] - ETA: 15s - loss: 0.3557 - accuracy: 0.8931
  600/25000 [..............................] - ETA: 15s - loss: 0.3590 - accuracy: 0.8907
  700/25000 [..............................] - ETA: 15s - loss: 0.3604 - accuracy: 0.8902
  800/25000 [..............................] - ETA: 15s - loss: 0.3609 - accuracy: 0.8896
  900/25000 [>.............................] - ETA: 15s - loss: 0.3607 - accuracy: 0.8897
 1000/25000 [>.............................] - ETA: 15s - loss: 0.3624 - accuracy: 0.8890
 1100/25000 [>.............................] - ETA: 15s - loss: 0.3624 - accuracy: 0.8891
 1200/25000 [>.............................] - ETA: 15s - loss: 0.3617 - accuracy: 0.8895
 1300/25000 [>.............................] - ETA: 15s - loss: 0.3613 - accuracy: 0.8898
 1400/25000 [>.............................] - ETA: 15s - loss: 0.3624 - accuracy: 0.8890
 1500/25000 [>.............................] - ETA: 15s - loss: 0.3629 - accuracy: 0.8888
 1600/25000 [>.............................] - ETA: 15s - loss: 0.3630 - accuracy: 0.8885
 1700/25000 [=>............................] - ETA: 15s - loss: 0.3626 - accuracy: 0.8887
 1800/25000 [=>............................] - ETA: 14s - loss: 0.3624 - accuracy: 0.8887
 1900/25000 [=>............................] - ETA: 14s - loss: 0.3625 - accuracy: 0.8888
 2000/25000 [=>............................] - ETA: 14s - loss: 0.3622 - accuracy: 0.8889
 2100/25000 [=>............................] - ETA: 14s - loss: 0.3625 - accuracy: 0.8887
 2200/25000 [=>............................] - ETA: 14s - loss: 0.3626 - accuracy: 0.8885
 2300/25000 [=>............................] - ETA: 14s - loss: 0.3624 - accuracy: 0.8886
 2400/25000 [=>............................] - ETA: 14s - loss: 0.3627 - accuracy: 0.8883
 2500/25000 [==>...........................] - ETA: 14s - loss: 0.3632 - accuracy: 0.8880
 2600/25000 [==>...........................] - ETA: 14s - loss: 0.3640 - accuracy: 0.8876
 2700/25000 [==>...........................] - ETA: 14s - loss: 0.3640 - accuracy: 0.8876
 2800/25000 [==>...........................] - ETA: 14s - loss: 0.3646 - accuracy: 0.8873
 2900/25000 [==>...........................] - ETA: 14s - loss: 0.3642 - accuracy: 0.8874
 3000/25000 [==>...........................] - ETA: 14s - loss: 0.3639 - accuracy: 0.8876
 3100/25000 [==>...........................] - ETA: 13s - loss: 0.3638 - accuracy: 0.8877
 3200/25000 [==>...........................] - ETA: 13s - loss: 0.3639 - accuracy: 0.8875
 3300/25000 [==>...........................] - ETA: 13s - loss: 0.3636 - accuracy: 0.8876
 3400/25000 [===>..........................] - ETA: 13s - loss: 0.3632 - accuracy: 0.8878
 3500/25000 [===>..........................] - ETA: 13s - loss: 0.3630 - accuracy: 0.8880
 3600/25000 [===>..........................] - ETA: 13s - loss: 0.3630 - accuracy: 0.8880
 3700/25000 [===>..........................] - ETA: 13s - loss: 0.3632 - accuracy: 0.8879
 3800/25000 [===>..........................] - ETA: 13s - loss: 0.3633 - accuracy: 0.8878
 3900/25000 [===>..........................] - ETA: 13s - loss: 0.3635 - accuracy: 0.8878
 4000/25000 [===>..........................] - ETA: 13s - loss: 0.3637 - accuracy: 0.8878
 4100/25000 [===>..........................] - ETA: 13s - loss: 0.3639 - accuracy: 0.8877
 4200/25000 [====>.........................] - ETA: 13s - loss: 0.3642 - accuracy: 0.8875
 4300/25000 [====>.........................] - ETA: 13s - loss: 0.3641 - accuracy: 0.8876
 4400/25000 [====>.........................] - ETA: 13s - loss: 0.3641 - accuracy: 0.8876
 4500/25000 [====>.........................] - ETA: 13s - loss: 0.3642 - accuracy: 0.8875
 4600/25000 [====>.........................] - ETA: 12s - loss: 0.3645 - accuracy: 0.8873
 4700/25000 [====>.........................] - ETA: 12s - loss: 0.3645 - accuracy: 0.8873
 4800/25000 [====>.........................] - ETA: 12s - loss: 0.3646 - accuracy: 0.8873
 4900/25000 [====>.........................] - ETA: 12s - loss: 0.3646 - accuracy: 0.8873
 5000/25000 [=====>........................] - ETA: 12s - loss: 0.3649 - accuracy: 0.8871
 5100/25000 [=====>........................] - ETA: 12s - loss: 0.3649 - accuracy: 0.8871
 5200/25000 [=====>........................] - ETA: 12s - loss: 0.3650 - accuracy: 0.8870
 5300/25000 [=====>........................] - ETA: 12s - loss: 0.3651 - accuracy: 0.8869
 5400/25000 [=====>........................] - ETA: 12s - loss: 0.3651 - accuracy: 0.8869
 5500/25000 [=====>........................] - ETA: 12s - loss: 0.3652 - accuracy: 0.8869
 5600/25000 [=====>........................] - ETA: 12s - loss: 0.3653 - accuracy: 0.8868
 5700/25000 [=====>........................] - ETA: 12s - loss: 0.3654 - accuracy: 0.8868
 5800/25000 [=====>........................] - ETA: 12s - loss: 0.3654 - accuracy: 0.8867
 5900/25000 [======>.......................] - ETA: 12s - loss: 0.3653 - accuracy: 0.8868
 6000/25000 [======>.......................] - ETA: 11s - loss: 0.3654 - accuracy: 0.8867
 6100/25000 [======>.......................] - ETA: 11s - loss: 0.3652 - accuracy: 0.8868
 6200/25000 [======>.......................] - ETA: 11s - loss: 0.3650 - accuracy: 0.8870
 6300/25000 [======>.......................] - ETA: 11s - loss: 0.3649 - accuracy: 0.8870
 6400/25000 [======>.......................] - ETA: 11s - loss: 0.3650 - accuracy: 0.8869
 6500/25000 [======>.......................] - ETA: 11s - loss: 0.3647 - accuracy: 0.8871
 6600/25000 [======>.......................] - ETA: 11s - loss: 0.3647 - accuracy: 0.8870
 6700/25000 [=======>......................] - ETA: 11s - loss: 0.3649 - accuracy: 0.8869
 6800/25000 [=======>......................] - ETA: 11s - loss: 0.3651 - accuracy: 0.8867
 6900/25000 [=======>......................] - ETA: 11s - loss: 0.3653 - accuracy: 0.8867
 7000/25000 [=======>......................] - ETA: 11s - loss: 0.3652 - accuracy: 0.8867
 7100/25000 [=======>......................] - ETA: 11s - loss: 0.3655 - accuracy: 0.8865
 7200/25000 [=======>......................] - ETA: 11s - loss: 0.3653 - accuracy: 0.8866
 7300/25000 [=======>......................] - ETA: 11s - loss: 0.3654 - accuracy: 0.8865
 7400/25000 [=======>......................] - ETA: 11s - loss: 0.3656 - accuracy: 0.8864
 7500/25000 [========>.....................] - ETA: 10s - loss: 0.3655 - accuracy: 0.8864
 7600/25000 [========>.....................] - ETA: 10s - loss: 0.3655 - accuracy: 0.8864
 7700/25000 [========>.....................] - ETA: 10s - loss: 0.3654 - accuracy: 0.8865
 7800/25000 [========>.....................] - ETA: 10s - loss: 0.3653 - accuracy: 0.8864
 7900/25000 [========>.....................] - ETA: 10s - loss: 0.3653 - accuracy: 0.8864
 8000/25000 [========>.....................] - ETA: 10s - loss: 0.3654 - accuracy: 0.8864
 8100/25000 [========>.....................] - ETA: 10s - loss: 0.3653 - accuracy: 0.8864
 8200/25000 [========>.....................] - ETA: 10s - loss: 0.3654 - accuracy: 0.8864
 8300/25000 [========>.....................] - ETA: 10s - loss: 0.3654 - accuracy: 0.8864
 8400/25000 [=========>....................] - ETA: 10s - loss: 0.3652 - accuracy: 0.8864
 8500/25000 [=========>....................] - ETA: 10s - loss: 0.3651 - accuracy: 0.8865
 8600/25000 [=========>....................] - ETA: 10s - loss: 0.3652 - accuracy: 0.8865
 8700/25000 [=========>....................] - ETA: 10s - loss: 0.3651 - accuracy: 0.8865
 8800/25000 [=========>....................] - ETA: 10s - loss: 0.3650 - accuracy: 0.8865
 8900/25000 [=========>....................] - ETA: 10s - loss: 0.3652 - accuracy: 0.8864
 9000/25000 [=========>....................] - ETA: 10s - loss: 0.3652 - accuracy: 0.8864
 9100/25000 [=========>....................] - ETA: 9s - loss: 0.3651 - accuracy: 0.8864 
 9200/25000 [==========>...................] - ETA: 9s - loss: 0.3650 - accuracy: 0.8865
 9300/25000 [==========>...................] - ETA: 9s - loss: 0.3651 - accuracy: 0.8864
 9400/25000 [==========>...................] - ETA: 9s - loss: 0.3650 - accuracy: 0.8864
 9500/25000 [==========>...................] - ETA: 9s - loss: 0.3651 - accuracy: 0.8865
 9600/25000 [==========>...................] - ETA: 9s - loss: 0.3650 - accuracy: 0.8864
 9700/25000 [==========>...................] - ETA: 9s - loss: 0.3651 - accuracy: 0.8864
 9800/25000 [==========>...................] - ETA: 9s - loss: 0.3650 - accuracy: 0.8865
 9900/25000 [==========>...................] - ETA: 9s - loss: 0.3650 - accuracy: 0.8864
10000/25000 [===========>..................] - ETA: 9s - loss: 0.3652 - accuracy: 0.8863
10100/25000 [===========>..................] - ETA: 9s - loss: 0.3650 - accuracy: 0.8865
10200/25000 [===========>..................] - ETA: 9s - loss: 0.3650 - accuracy: 0.8864
10300/25000 [===========>..................] - ETA: 9s - loss: 0.3651 - accuracy: 0.8864
10400/25000 [===========>..................] - ETA: 9s - loss: 0.3649 - accuracy: 0.8865
10500/25000 [===========>..................] - ETA: 9s - loss: 0.3649 - accuracy: 0.8864
10600/25000 [===========>..................] - ETA: 9s - loss: 0.3647 - accuracy: 0.8865
10700/25000 [===========>..................] - ETA: 8s - loss: 0.3647 - accuracy: 0.8865
10800/25000 [===========>..................] - ETA: 8s - loss: 0.3649 - accuracy: 0.8864
10900/25000 [============>.................] - ETA: 8s - loss: 0.3650 - accuracy: 0.8864
11000/25000 [============>.................] - ETA: 8s - loss: 0.3650 - accuracy: 0.8863
11100/25000 [============>.................] - ETA: 8s - loss: 0.3650 - accuracy: 0.8863
11200/25000 [============>.................] - ETA: 8s - loss: 0.3650 - accuracy: 0.8863
11300/25000 [============>.................] - ETA: 8s - loss: 0.3651 - accuracy: 0.8863
11400/25000 [============>.................] - ETA: 8s - loss: 0.3650 - accuracy: 0.8863
11500/25000 [============>.................] - ETA: 8s - loss: 0.3650 - accuracy: 0.8862
11600/25000 [============>.................] - ETA: 8s - loss: 0.3649 - accuracy: 0.8863
11700/25000 [=============>................] - ETA: 8s - loss: 0.3648 - accuracy: 0.8863
11800/25000 [=============>................] - ETA: 8s - loss: 0.3649 - accuracy: 0.8863
11900/25000 [=============>................] - ETA: 8s - loss: 0.3648 - accuracy: 0.8864
12000/25000 [=============>................] - ETA: 8s - loss: 0.3648 - accuracy: 0.8864
12100/25000 [=============>................] - ETA: 8s - loss: 0.3649 - accuracy: 0.8863
12200/25000 [=============>................] - ETA: 8s - loss: 0.3648 - accuracy: 0.8863
12300/25000 [=============>................] - ETA: 7s - loss: 0.3648 - accuracy: 0.8863
12400/25000 [=============>................] - ETA: 7s - loss: 0.3648 - accuracy: 0.8863
12500/25000 [==============>...............] - ETA: 7s - loss: 0.3648 - accuracy: 0.8863
12600/25000 [==============>...............] - ETA: 7s - loss: 0.3648 - accuracy: 0.8863
12700/25000 [==============>...............] - ETA: 7s - loss: 0.3650 - accuracy: 0.8862
12800/25000 [==============>...............] - ETA: 7s - loss: 0.3650 - accuracy: 0.8862
12900/25000 [==============>...............] - ETA: 7s - loss: 0.3650 - accuracy: 0.8862
13000/25000 [==============>...............] - ETA: 7s - loss: 0.3649 - accuracy: 0.8862
13100/25000 [==============>...............] - ETA: 7s - loss: 0.3649 - accuracy: 0.8862
13200/25000 [==============>...............] - ETA: 7s - loss: 0.3650 - accuracy: 0.8862
13300/25000 [==============>...............] - ETA: 7s - loss: 0.3649 - accuracy: 0.8862
13400/25000 [===============>..............] - ETA: 7s - loss: 0.3647 - accuracy: 0.8863
13500/25000 [===============>..............] - ETA: 7s - loss: 0.3646 - accuracy: 0.8863
13600/25000 [===============>..............] - ETA: 7s - loss: 0.3646 - accuracy: 0.8863
13700/25000 [===============>..............] - ETA: 7s - loss: 0.3648 - accuracy: 0.8861
13800/25000 [===============>..............] - ETA: 7s - loss: 0.3647 - accuracy: 0.8862
13900/25000 [===============>..............] - ETA: 6s - loss: 0.3647 - accuracy: 0.8862
14000/25000 [===============>..............] - ETA: 6s - loss: 0.3647 - accuracy: 0.8861
14100/25000 [===============>..............] - ETA: 6s - loss: 0.3646 - accuracy: 0.8862
14200/25000 [================>.............] - ETA: 6s - loss: 0.3646 - accuracy: 0.8862
14300/25000 [================>.............] - ETA: 6s - loss: 0.3646 - accuracy: 0.8862
14400/25000 [================>.............] - ETA: 6s - loss: 0.3645 - accuracy: 0.8862
14500/25000 [================>.............] - ETA: 6s - loss: 0.3645 - accuracy: 0.8862
14600/25000 [================>.............] - ETA: 6s - loss: 0.3645 - accuracy: 0.8862
14700/25000 [================>.............] - ETA: 6s - loss: 0.3646 - accuracy: 0.8861
14800/25000 [================>.............] - ETA: 6s - loss: 0.3645 - accuracy: 0.8862
14900/25000 [================>.............] - ETA: 6s - loss: 0.3645 - accuracy: 0.8862
15000/25000 [=================>............] - ETA: 6s - loss: 0.3645 - accuracy: 0.8861
15100/25000 [=================>............] - ETA: 6s - loss: 0.3646 - accuracy: 0.8861
15200/25000 [=================>............] - ETA: 6s - loss: 0.3645 - accuracy: 0.8861
15300/25000 [=================>............] - ETA: 6s - loss: 0.3645 - accuracy: 0.8861
15400/25000 [=================>............] - ETA: 6s - loss: 0.3646 - accuracy: 0.8860
15500/25000 [=================>............] - ETA: 5s - loss: 0.3646 - accuracy: 0.8860
15600/25000 [=================>............] - ETA: 5s - loss: 0.3646 - accuracy: 0.8861
15700/25000 [=================>............] - ETA: 5s - loss: 0.3645 - accuracy: 0.8861
15800/25000 [=================>............] - ETA: 5s - loss: 0.3645 - accuracy: 0.8861
15900/25000 [==================>...........] - ETA: 5s - loss: 0.3646 - accuracy: 0.8860
16000/25000 [==================>...........] - ETA: 5s - loss: 0.3644 - accuracy: 0.8861
16100/25000 [==================>...........] - ETA: 5s - loss: 0.3643 - accuracy: 0.8861
16200/25000 [==================>...........] - ETA: 5s - loss: 0.3644 - accuracy: 0.8861
16300/25000 [==================>...........] - ETA: 5s - loss: 0.3644 - accuracy: 0.8860
16400/25000 [==================>...........] - ETA: 5s - loss: 0.3644 - accuracy: 0.8860
16500/25000 [==================>...........] - ETA: 5s - loss: 0.3645 - accuracy: 0.8860
16600/25000 [==================>...........] - ETA: 5s - loss: 0.3644 - accuracy: 0.8860
16700/25000 [===================>..........] - ETA: 5s - loss: 0.3644 - accuracy: 0.8860
16800/25000 [===================>..........] - ETA: 5s - loss: 0.3643 - accuracy: 0.8860
16900/25000 [===================>..........] - ETA: 5s - loss: 0.3643 - accuracy: 0.8860
17000/25000 [===================>..........] - ETA: 5s - loss: 0.3644 - accuracy: 0.8860
17100/25000 [===================>..........] - ETA: 4s - loss: 0.3644 - accuracy: 0.8860
17200/25000 [===================>..........] - ETA: 4s - loss: 0.3643 - accuracy: 0.8860
17300/25000 [===================>..........] - ETA: 4s - loss: 0.3642 - accuracy: 0.8861
17400/25000 [===================>..........] - ETA: 4s - loss: 0.3643 - accuracy: 0.8860
17500/25000 [====================>.........] - ETA: 4s - loss: 0.3643 - accuracy: 0.8860
17600/25000 [====================>.........] - ETA: 4s - loss: 0.3644 - accuracy: 0.8859
17700/25000 [====================>.........] - ETA: 4s - loss: 0.3643 - accuracy: 0.8860
17800/25000 [====================>.........] - ETA: 4s - loss: 0.3643 - accuracy: 0.8860
17900/25000 [====================>.........] - ETA: 4s - loss: 0.3642 - accuracy: 0.8860
18000/25000 [====================>.........] - ETA: 4s - loss: 0.3642 - accuracy: 0.8860
18100/25000 [====================>.........] - ETA: 4s - loss: 0.3643 - accuracy: 0.8859
18200/25000 [====================>.........] - ETA: 4s - loss: 0.3643 - accuracy: 0.8859
18300/25000 [====================>.........] - ETA: 4s - loss: 0.3644 - accuracy: 0.8859
18400/25000 [=====================>........] - ETA: 4s - loss: 0.3643 - accuracy: 0.8859
18500/25000 [=====================>........] - ETA: 4s - loss: 0.3641 - accuracy: 0.8860
18600/25000 [=====================>........] - ETA: 4s - loss: 0.3642 - accuracy: 0.8859
18700/25000 [=====================>........] - ETA: 3s - loss: 0.3642 - accuracy: 0.8859
18800/25000 [=====================>........] - ETA: 3s - loss: 0.3641 - accuracy: 0.8860
18900/25000 [=====================>........] - ETA: 3s - loss: 0.3641 - accuracy: 0.8860
19000/25000 [=====================>........] - ETA: 3s - loss: 0.3642 - accuracy: 0.8859
19100/25000 [=====================>........] - ETA: 3s - loss: 0.3643 - accuracy: 0.8859
19200/25000 [======================>.......] - ETA: 3s - loss: 0.3643 - accuracy: 0.8859
19300/25000 [======================>.......] - ETA: 3s - loss: 0.3643 - accuracy: 0.8859
19400/25000 [======================>.......] - ETA: 3s - loss: 0.3642 - accuracy: 0.8859
19500/25000 [======================>.......] - ETA: 3s - loss: 0.3643 - accuracy: 0.8859
19600/25000 [======================>.......] - ETA: 3s - loss: 0.3642 - accuracy: 0.8859
19700/25000 [======================>.......] - ETA: 3s - loss: 0.3641 - accuracy: 0.8859
19800/25000 [======================>.......] - ETA: 3s - loss: 0.3640 - accuracy: 0.8860
19900/25000 [======================>.......] - ETA: 3s - loss: 0.3640 - accuracy: 0.8860
20000/25000 [=======================>......] - ETA: 3s - loss: 0.3639 - accuracy: 0.8860
20100/25000 [=======================>......] - ETA: 3s - loss: 0.3640 - accuracy: 0.8860
20200/25000 [=======================>......] - ETA: 3s - loss: 0.3640 - accuracy: 0.8860
20300/25000 [=======================>......] - ETA: 2s - loss: 0.3639 - accuracy: 0.8859
20400/25000 [=======================>......] - ETA: 2s - loss: 0.3640 - accuracy: 0.8859
20500/25000 [=======================>......] - ETA: 2s - loss: 0.3639 - accuracy: 0.8860
20600/25000 [=======================>......] - ETA: 2s - loss: 0.3638 - accuracy: 0.8860
20700/25000 [=======================>......] - ETA: 2s - loss: 0.3638 - accuracy: 0.8860
20800/25000 [=======================>......] - ETA: 2s - loss: 0.3637 - accuracy: 0.8860
20900/25000 [========================>.....] - ETA: 2s - loss: 0.3637 - accuracy: 0.8860
21000/25000 [========================>.....] - ETA: 2s - loss: 0.3637 - accuracy: 0.8860
21100/25000 [========================>.....] - ETA: 2s - loss: 0.3637 - accuracy: 0.8861
21200/25000 [========================>.....] - ETA: 2s - loss: 0.3637 - accuracy: 0.8861
21300/25000 [========================>.....] - ETA: 2s - loss: 0.3637 - accuracy: 0.8860
21400/25000 [========================>.....] - ETA: 2s - loss: 0.3636 - accuracy: 0.8861
21500/25000 [========================>.....] - ETA: 2s - loss: 0.3636 - accuracy: 0.8861
21600/25000 [========================>.....] - ETA: 2s - loss: 0.3636 - accuracy: 0.8860
21700/25000 [=========================>....] - ETA: 2s - loss: 0.3636 - accuracy: 0.8861
21800/25000 [=========================>....] - ETA: 2s - loss: 0.3635 - accuracy: 0.8861
21900/25000 [=========================>....] - ETA: 1s - loss: 0.3635 - accuracy: 0.8861
22000/25000 [=========================>....] - ETA: 1s - loss: 0.3633 - accuracy: 0.8862
22100/25000 [=========================>....] - ETA: 1s - loss: 0.3633 - accuracy: 0.8862
22200/25000 [=========================>....] - ETA: 1s - loss: 0.3633 - accuracy: 0.8861
22300/25000 [=========================>....] - ETA: 1s - loss: 0.3633 - accuracy: 0.8862
22400/25000 [=========================>....] - ETA: 1s - loss: 0.3632 - accuracy: 0.8862
22500/25000 [==========================>...] - ETA: 1s - loss: 0.3632 - accuracy: 0.8862
22600/25000 [==========================>...] - ETA: 1s - loss: 0.3631 - accuracy: 0.8862
22700/25000 [==========================>...] - ETA: 1s - loss: 0.3630 - accuracy: 0.8863
22800/25000 [==========================>...] - ETA: 1s - loss: 0.3630 - accuracy: 0.8863
22900/25000 [==========================>...] - ETA: 1s - loss: 0.3631 - accuracy: 0.8862
23000/25000 [==========================>...] - ETA: 1s - loss: 0.3630 - accuracy: 0.8862
23100/25000 [==========================>...] - ETA: 1s - loss: 0.3631 - accuracy: 0.8862
23200/25000 [==========================>...] - ETA: 1s - loss: 0.3631 - accuracy: 0.8862
23300/25000 [==========================>...] - ETA: 1s - loss: 0.3631 - accuracy: 0.8862
23400/25000 [===========================>..] - ETA: 1s - loss: 0.3630 - accuracy: 0.8862
23500/25000 [===========================>..] - ETA: 0s - loss: 0.3630 - accuracy: 0.8862
23600/25000 [===========================>..] - ETA: 0s - loss: 0.3629 - accuracy: 0.8862
23700/25000 [===========================>..] - ETA: 0s - loss: 0.3629 - accuracy: 0.8862
23800/25000 [===========================>..] - ETA: 0s - loss: 0.3629 - accuracy: 0.8862
23900/25000 [===========================>..] - ETA: 0s - loss: 0.3628 - accuracy: 0.8863
24000/25000 [===========================>..] - ETA: 0s - loss: 0.3627 - accuracy: 0.8863
24100/25000 [===========================>..] - ETA: 0s - loss: 0.3627 - accuracy: 0.8863
24200/25000 [============================>.] - ETA: 0s - loss: 0.3627 - accuracy: 0.8863
24300/25000 [============================>.] - ETA: 0s - loss: 0.3627 - accuracy: 0.8863
24400/25000 [============================>.] - ETA: 0s - loss: 0.3627 - accuracy: 0.8862
24500/25000 [============================>.] - ETA: 0s - loss: 0.3627 - accuracy: 0.8862
24600/25000 [============================>.] - ETA: 0s - loss: 0.3627 - accuracy: 0.8863
24700/25000 [============================>.] - ETA: 0s - loss: 0.3627 - accuracy: 0.8862
24800/25000 [============================>.] - ETA: 0s - loss: 0.3626 - accuracy: 0.8863
24900/25000 [============================>.] - ETA: 0s - loss: 0.3626 - accuracy: 0.8863
25000/25000 [==============================] - 21s 829us/step - loss: 0.3625 - accuracy: 0.8863 - val_loss: 0.3580 - val_accuracy: 0.8858
Epoch 6/10

  100/25000 [..............................] - ETA: 15s - loss: 0.3551 - accuracy: 0.8900
  200/25000 [..............................] - ETA: 15s - loss: 0.3533 - accuracy: 0.8907
  300/25000 [..............................] - ETA: 15s - loss: 0.3556 - accuracy: 0.8900
  400/25000 [..............................] - ETA: 15s - loss: 0.3542 - accuracy: 0.8900
  500/25000 [..............................] - ETA: 15s - loss: 0.3588 - accuracy: 0.8871
  600/25000 [..............................] - ETA: 15s - loss: 0.3596 - accuracy: 0.8867
  700/25000 [..............................] - ETA: 15s - loss: 0.3578 - accuracy: 0.8878
  800/25000 [..............................] - ETA: 15s - loss: 0.3571 - accuracy: 0.8879
  900/25000 [>.............................] - ETA: 15s - loss: 0.3574 - accuracy: 0.8875
 1000/25000 [>.............................] - ETA: 15s - loss: 0.3575 - accuracy: 0.8871
 1100/25000 [>.............................] - ETA: 15s - loss: 0.3585 - accuracy: 0.8866
 1200/25000 [>.............................] - ETA: 15s - loss: 0.3607 - accuracy: 0.8855
 1300/25000 [>.............................] - ETA: 15s - loss: 0.3609 - accuracy: 0.8854
 1400/25000 [>.............................] - ETA: 15s - loss: 0.3597 - accuracy: 0.8858
 1500/25000 [>.............................] - ETA: 15s - loss: 0.3600 - accuracy: 0.8857
 1600/25000 [>.............................] - ETA: 14s - loss: 0.3598 - accuracy: 0.8858
 1700/25000 [=>............................] - ETA: 14s - loss: 0.3604 - accuracy: 0.8855
 1800/25000 [=>............................] - ETA: 14s - loss: 0.3592 - accuracy: 0.8861
 1900/25000 [=>............................] - ETA: 14s - loss: 0.3592 - accuracy: 0.8862
 2000/25000 [=>............................] - ETA: 14s - loss: 0.3590 - accuracy: 0.8861
 2100/25000 [=>............................] - ETA: 14s - loss: 0.3589 - accuracy: 0.8861
 2200/25000 [=>............................] - ETA: 14s - loss: 0.3587 - accuracy: 0.8861
 2300/25000 [=>............................] - ETA: 14s - loss: 0.3586 - accuracy: 0.8861
 2400/25000 [=>............................] - ETA: 14s - loss: 0.3584 - accuracy: 0.8861
 2500/25000 [==>...........................] - ETA: 14s - loss: 0.3587 - accuracy: 0.8859
 2600/25000 [==>...........................] - ETA: 14s - loss: 0.3590 - accuracy: 0.8858
 2700/25000 [==>...........................] - ETA: 14s - loss: 0.3590 - accuracy: 0.8858
 2800/25000 [==>...........................] - ETA: 14s - loss: 0.3589 - accuracy: 0.8859
 2900/25000 [==>...........................] - ETA: 14s - loss: 0.3587 - accuracy: 0.8860
 3000/25000 [==>...........................] - ETA: 13s - loss: 0.3585 - accuracy: 0.8861
 3100/25000 [==>...........................] - ETA: 13s - loss: 0.3588 - accuracy: 0.8859
 3200/25000 [==>...........................] - ETA: 13s - loss: 0.3584 - accuracy: 0.8861
 3300/25000 [==>...........................] - ETA: 13s - loss: 0.3584 - accuracy: 0.8861
 3400/25000 [===>..........................] - ETA: 13s - loss: 0.3584 - accuracy: 0.8861
 3500/25000 [===>..........................] - ETA: 13s - loss: 0.3582 - accuracy: 0.8863
 3600/25000 [===>..........................] - ETA: 13s - loss: 0.3579 - accuracy: 0.8864
 3700/25000 [===>..........................] - ETA: 13s - loss: 0.3581 - accuracy: 0.8864
 3800/25000 [===>..........................] - ETA: 13s - loss: 0.3582 - accuracy: 0.8864
 3900/25000 [===>..........................] - ETA: 13s - loss: 0.3578 - accuracy: 0.8866
 4000/25000 [===>..........................] - ETA: 13s - loss: 0.3578 - accuracy: 0.8865
 4100/25000 [===>..........................] - ETA: 13s - loss: 0.3579 - accuracy: 0.8865
 4200/25000 [====>.........................] - ETA: 13s - loss: 0.3579 - accuracy: 0.8864
 4300/25000 [====>.........................] - ETA: 13s - loss: 0.3579 - accuracy: 0.8863
 4400/25000 [====>.........................] - ETA: 12s - loss: 0.3582 - accuracy: 0.8861
 4500/25000 [====>.........................] - ETA: 12s - loss: 0.3583 - accuracy: 0.8861
 4600/25000 [====>.........................] - ETA: 12s - loss: 0.3582 - accuracy: 0.8861
 4700/25000 [====>.........................] - ETA: 12s - loss: 0.3584 - accuracy: 0.8860
 4800/25000 [====>.........................] - ETA: 12s - loss: 0.3581 - accuracy: 0.8861
 4900/25000 [====>.........................] - ETA: 12s - loss: 0.3579 - accuracy: 0.8862
 5000/25000 [=====>........................] - ETA: 12s - loss: 0.3580 - accuracy: 0.8862
 5100/25000 [=====>........................] - ETA: 12s - loss: 0.3582 - accuracy: 0.8860
 5200/25000 [=====>........................] - ETA: 12s - loss: 0.3582 - accuracy: 0.8860
 5300/25000 [=====>........................] - ETA: 12s - loss: 0.3581 - accuracy: 0.8861
 5400/25000 [=====>........................] - ETA: 12s - loss: 0.3582 - accuracy: 0.8860
 5500/25000 [=====>........................] - ETA: 12s - loss: 0.3579 - accuracy: 0.8861
 5600/25000 [=====>........................] - ETA: 12s - loss: 0.3581 - accuracy: 0.8861
 5700/25000 [=====>........................] - ETA: 12s - loss: 0.3577 - accuracy: 0.8862
 5800/25000 [=====>........................] - ETA: 12s - loss: 0.3576 - accuracy: 0.8863
 5900/25000 [======>.......................] - ETA: 11s - loss: 0.3577 - accuracy: 0.8863
 6000/25000 [======>.......................] - ETA: 11s - loss: 0.3577 - accuracy: 0.8863
 6100/25000 [======>.......................] - ETA: 11s - loss: 0.3576 - accuracy: 0.8863
 6200/25000 [======>.......................] - ETA: 11s - loss: 0.3576 - accuracy: 0.8863
 6300/25000 [======>.......................] - ETA: 11s - loss: 0.3577 - accuracy: 0.8863
 6400/25000 [======>.......................] - ETA: 11s - loss: 0.3575 - accuracy: 0.8864
 6500/25000 [======>.......................] - ETA: 11s - loss: 0.3574 - accuracy: 0.8865
 6600/25000 [======>.......................] - ETA: 11s - loss: 0.3570 - accuracy: 0.8866
 6700/25000 [=======>......................] - ETA: 11s - loss: 0.3572 - accuracy: 0.8865
 6800/25000 [=======>......................] - ETA: 11s - loss: 0.3572 - accuracy: 0.8865
 6900/25000 [=======>......................] - ETA: 11s - loss: 0.3572 - accuracy: 0.8865
 7000/25000 [=======>......................] - ETA: 11s - loss: 0.3573 - accuracy: 0.8864
 7100/25000 [=======>......................] - ETA: 11s - loss: 0.3571 - accuracy: 0.8866
 7200/25000 [=======>......................] - ETA: 11s - loss: 0.3570 - accuracy: 0.8866
 7300/25000 [=======>......................] - ETA: 11s - loss: 0.3571 - accuracy: 0.8865
 7400/25000 [=======>......................] - ETA: 11s - loss: 0.3570 - accuracy: 0.8865
 7500/25000 [========>.....................] - ETA: 10s - loss: 0.3571 - accuracy: 0.8865
 7600/25000 [========>.....................] - ETA: 10s - loss: 0.3572 - accuracy: 0.8864
 7700/25000 [========>.....................] - ETA: 10s - loss: 0.3569 - accuracy: 0.8866
 7800/25000 [========>.....................] - ETA: 10s - loss: 0.3567 - accuracy: 0.8867
 7900/25000 [========>.....................] - ETA: 10s - loss: 0.3570 - accuracy: 0.8865
 8000/25000 [========>.....................] - ETA: 10s - loss: 0.3570 - accuracy: 0.8865
 8100/25000 [========>.....................] - ETA: 10s - loss: 0.3568 - accuracy: 0.8866
 8200/25000 [========>.....................] - ETA: 10s - loss: 0.3569 - accuracy: 0.8866
 8300/25000 [========>.....................] - ETA: 10s - loss: 0.3569 - accuracy: 0.8865
 8400/25000 [=========>....................] - ETA: 10s - loss: 0.3569 - accuracy: 0.8864
 8500/25000 [=========>....................] - ETA: 10s - loss: 0.3571 - accuracy: 0.8864
 8600/25000 [=========>....................] - ETA: 10s - loss: 0.3569 - accuracy: 0.8864
 8700/25000 [=========>....................] - ETA: 10s - loss: 0.3570 - accuracy: 0.8864
 8800/25000 [=========>....................] - ETA: 10s - loss: 0.3569 - accuracy: 0.8864
 8900/25000 [=========>....................] - ETA: 10s - loss: 0.3569 - accuracy: 0.8864
 9000/25000 [=========>....................] - ETA: 10s - loss: 0.3569 - accuracy: 0.8864
 9100/25000 [=========>....................] - ETA: 9s - loss: 0.3567 - accuracy: 0.8865 
 9200/25000 [==========>...................] - ETA: 9s - loss: 0.3566 - accuracy: 0.8865
 9300/25000 [==========>...................] - ETA: 9s - loss: 0.3567 - accuracy: 0.8865
 9400/25000 [==========>...................] - ETA: 9s - loss: 0.3566 - accuracy: 0.8865
 9500/25000 [==========>...................] - ETA: 9s - loss: 0.3566 - accuracy: 0.8865
 9600/25000 [==========>...................] - ETA: 9s - loss: 0.3564 - accuracy: 0.8866
 9700/25000 [==========>...................] - ETA: 9s - loss: 0.3565 - accuracy: 0.8865
 9800/25000 [==========>...................] - ETA: 9s - loss: 0.3566 - accuracy: 0.8865
 9900/25000 [==========>...................] - ETA: 9s - loss: 0.3567 - accuracy: 0.8864
10000/25000 [===========>..................] - ETA: 9s - loss: 0.3568 - accuracy: 0.8864
10100/25000 [===========>..................] - ETA: 9s - loss: 0.3568 - accuracy: 0.8864
10200/25000 [===========>..................] - ETA: 9s - loss: 0.3569 - accuracy: 0.8863
10300/25000 [===========>..................] - ETA: 9s - loss: 0.3569 - accuracy: 0.8863
10400/25000 [===========>..................] - ETA: 9s - loss: 0.3569 - accuracy: 0.8863
10500/25000 [===========>..................] - ETA: 9s - loss: 0.3567 - accuracy: 0.8864
10600/25000 [===========>..................] - ETA: 9s - loss: 0.3568 - accuracy: 0.8863
10700/25000 [===========>..................] - ETA: 8s - loss: 0.3567 - accuracy: 0.8864
10800/25000 [===========>..................] - ETA: 8s - loss: 0.3567 - accuracy: 0.8864
10900/25000 [============>.................] - ETA: 8s - loss: 0.3568 - accuracy: 0.8863
11000/25000 [============>.................] - ETA: 8s - loss: 0.3568 - accuracy: 0.8863
11100/25000 [============>.................] - ETA: 8s - loss: 0.3568 - accuracy: 0.8863
11200/25000 [============>.................] - ETA: 8s - loss: 0.3567 - accuracy: 0.8864
11300/25000 [============>.................] - ETA: 8s - loss: 0.3568 - accuracy: 0.8863
11400/25000 [============>.................] - ETA: 8s - loss: 0.3568 - accuracy: 0.8863
11500/25000 [============>.................] - ETA: 8s - loss: 0.3569 - accuracy: 0.8863
11600/25000 [============>.................] - ETA: 8s - loss: 0.3571 - accuracy: 0.8862
11700/25000 [=============>................] - ETA: 8s - loss: 0.3572 - accuracy: 0.8861
11800/25000 [=============>................] - ETA: 8s - loss: 0.3572 - accuracy: 0.8861
11900/25000 [=============>................] - ETA: 8s - loss: 0.3573 - accuracy: 0.8860
12000/25000 [=============>................] - ETA: 8s - loss: 0.3573 - accuracy: 0.8860
12100/25000 [=============>................] - ETA: 8s - loss: 0.3573 - accuracy: 0.8860
12200/25000 [=============>................] - ETA: 7s - loss: 0.3573 - accuracy: 0.8860
12300/25000 [=============>................] - ETA: 7s - loss: 0.3573 - accuracy: 0.8861
12400/25000 [=============>................] - ETA: 7s - loss: 0.3573 - accuracy: 0.8860
12500/25000 [==============>...............] - ETA: 7s - loss: 0.3572 - accuracy: 0.8861
12600/25000 [==============>...............] - ETA: 7s - loss: 0.3571 - accuracy: 0.8861
12700/25000 [==============>...............] - ETA: 7s - loss: 0.3570 - accuracy: 0.8861
12800/25000 [==============>...............] - ETA: 7s - loss: 0.3569 - accuracy: 0.8862
12900/25000 [==============>...............] - ETA: 7s - loss: 0.3569 - accuracy: 0.8862
13000/25000 [==============>...............] - ETA: 7s - loss: 0.3569 - accuracy: 0.8861
13100/25000 [==============>...............] - ETA: 7s - loss: 0.3569 - accuracy: 0.8862
13200/25000 [==============>...............] - ETA: 7s - loss: 0.3570 - accuracy: 0.8861
13300/25000 [==============>...............] - ETA: 7s - loss: 0.3567 - accuracy: 0.8863
13400/25000 [===============>..............] - ETA: 7s - loss: 0.3568 - accuracy: 0.8862
13500/25000 [===============>..............] - ETA: 7s - loss: 0.3568 - accuracy: 0.8862
13600/25000 [===============>..............] - ETA: 7s - loss: 0.3567 - accuracy: 0.8863
13700/25000 [===============>..............] - ETA: 7s - loss: 0.3566 - accuracy: 0.8863
13800/25000 [===============>..............] - ETA: 6s - loss: 0.3567 - accuracy: 0.8863
13900/25000 [===============>..............] - ETA: 6s - loss: 0.3567 - accuracy: 0.8862
14000/25000 [===============>..............] - ETA: 6s - loss: 0.3567 - accuracy: 0.8862
14100/25000 [===============>..............] - ETA: 6s - loss: 0.3565 - accuracy: 0.8863
14200/25000 [================>.............] - ETA: 6s - loss: 0.3565 - accuracy: 0.8863
14300/25000 [================>.............] - ETA: 6s - loss: 0.3565 - accuracy: 0.8864
14400/25000 [================>.............] - ETA: 6s - loss: 0.3564 - accuracy: 0.8864
14500/25000 [================>.............] - ETA: 6s - loss: 0.3564 - accuracy: 0.8864
14600/25000 [================>.............] - ETA: 6s - loss: 0.3563 - accuracy: 0.8864
14700/25000 [================>.............] - ETA: 6s - loss: 0.3563 - accuracy: 0.8864
14800/25000 [================>.............] - ETA: 6s - loss: 0.3563 - accuracy: 0.8865
14900/25000 [================>.............] - ETA: 6s - loss: 0.3561 - accuracy: 0.8865
15000/25000 [=================>............] - ETA: 6s - loss: 0.3560 - accuracy: 0.8866
15100/25000 [=================>............] - ETA: 6s - loss: 0.3560 - accuracy: 0.8866
15200/25000 [=================>............] - ETA: 6s - loss: 0.3560 - accuracy: 0.8866
15300/25000 [=================>............] - ETA: 6s - loss: 0.3560 - accuracy: 0.8866
15400/25000 [=================>............] - ETA: 5s - loss: 0.3561 - accuracy: 0.8865
15500/25000 [=================>............] - ETA: 5s - loss: 0.3562 - accuracy: 0.8865
15600/25000 [=================>............] - ETA: 5s - loss: 0.3561 - accuracy: 0.8865
15700/25000 [=================>............] - ETA: 5s - loss: 0.3561 - accuracy: 0.8865
15800/25000 [=================>............] - ETA: 5s - loss: 0.3562 - accuracy: 0.8864
15900/25000 [==================>...........] - ETA: 5s - loss: 0.3563 - accuracy: 0.8864
16000/25000 [==================>...........] - ETA: 5s - loss: 0.3561 - accuracy: 0.8865
16100/25000 [==================>...........] - ETA: 5s - loss: 0.3562 - accuracy: 0.8865
16200/25000 [==================>...........] - ETA: 5s - loss: 0.3562 - accuracy: 0.8864
16300/25000 [==================>...........] - ETA: 5s - loss: 0.3563 - accuracy: 0.8864
16400/25000 [==================>...........] - ETA: 5s - loss: 0.3563 - accuracy: 0.8864
16500/25000 [==================>...........] - ETA: 5s - loss: 0.3562 - accuracy: 0.8864
16600/25000 [==================>...........] - ETA: 5s - loss: 0.3562 - accuracy: 0.8864
16700/25000 [===================>..........] - ETA: 5s - loss: 0.3561 - accuracy: 0.8865
16800/25000 [===================>..........] - ETA: 5s - loss: 0.3560 - accuracy: 0.8865
16900/25000 [===================>..........] - ETA: 5s - loss: 0.3560 - accuracy: 0.8865
17000/25000 [===================>..........] - ETA: 4s - loss: 0.3560 - accuracy: 0.8866
17100/25000 [===================>..........] - ETA: 4s - loss: 0.3559 - accuracy: 0.8866
17200/25000 [===================>..........] - ETA: 4s - loss: 0.3558 - accuracy: 0.8866
17300/25000 [===================>..........] - ETA: 4s - loss: 0.3558 - accuracy: 0.8866
17400/25000 [===================>..........] - ETA: 4s - loss: 0.3556 - accuracy: 0.8867
17500/25000 [====================>.........] - ETA: 4s - loss: 0.3556 - accuracy: 0.8867
17600/25000 [====================>.........] - ETA: 4s - loss: 0.3555 - accuracy: 0.8868
17700/25000 [====================>.........] - ETA: 4s - loss: 0.3554 - accuracy: 0.8868
17800/25000 [====================>.........] - ETA: 4s - loss: 0.3555 - accuracy: 0.8867
17900/25000 [====================>.........] - ETA: 4s - loss: 0.3554 - accuracy: 0.8868
18000/25000 [====================>.........] - ETA: 4s - loss: 0.3554 - accuracy: 0.8868
18100/25000 [====================>.........] - ETA: 4s - loss: 0.3555 - accuracy: 0.8867
18200/25000 [====================>.........] - ETA: 4s - loss: 0.3555 - accuracy: 0.8867
18300/25000 [====================>.........] - ETA: 4s - loss: 0.3556 - accuracy: 0.8867
18400/25000 [=====================>........] - ETA: 4s - loss: 0.3555 - accuracy: 0.8867
18500/25000 [=====================>........] - ETA: 4s - loss: 0.3555 - accuracy: 0.8867
18600/25000 [=====================>........] - ETA: 3s - loss: 0.3556 - accuracy: 0.8866
18700/25000 [=====================>........] - ETA: 3s - loss: 0.3556 - accuracy: 0.8867
18800/25000 [=====================>........] - ETA: 3s - loss: 0.3556 - accuracy: 0.8866
18900/25000 [=====================>........] - ETA: 3s - loss: 0.3555 - accuracy: 0.8867
19000/25000 [=====================>........] - ETA: 3s - loss: 0.3556 - accuracy: 0.8866
19100/25000 [=====================>........] - ETA: 3s - loss: 0.3556 - accuracy: 0.8866
19200/25000 [======================>.......] - ETA: 3s - loss: 0.3556 - accuracy: 0.8866
19300/25000 [======================>.......] - ETA: 3s - loss: 0.3556 - accuracy: 0.8866
19400/25000 [======================>.......] - ETA: 3s - loss: 0.3555 - accuracy: 0.8866
19500/25000 [======================>.......] - ETA: 3s - loss: 0.3556 - accuracy: 0.8866
19600/25000 [======================>.......] - ETA: 3s - loss: 0.3555 - accuracy: 0.8866
19700/25000 [======================>.......] - ETA: 3s - loss: 0.3555 - accuracy: 0.8866
19800/25000 [======================>.......] - ETA: 3s - loss: 0.3554 - accuracy: 0.8867
19900/25000 [======================>.......] - ETA: 3s - loss: 0.3555 - accuracy: 0.8866
20000/25000 [=======================>......] - ETA: 3s - loss: 0.3555 - accuracy: 0.8866
20100/25000 [=======================>......] - ETA: 3s - loss: 0.3555 - accuracy: 0.8866
20200/25000 [=======================>......] - ETA: 2s - loss: 0.3555 - accuracy: 0.8866
20300/25000 [=======================>......] - ETA: 2s - loss: 0.3555 - accuracy: 0.8866
20400/25000 [=======================>......] - ETA: 2s - loss: 0.3555 - accuracy: 0.8866
20500/25000 [=======================>......] - ETA: 2s - loss: 0.3555 - accuracy: 0.8866
20600/25000 [=======================>......] - ETA: 2s - loss: 0.3555 - accuracy: 0.8865
20700/25000 [=======================>......] - ETA: 2s - loss: 0.3555 - accuracy: 0.8865
20800/25000 [=======================>......] - ETA: 2s - loss: 0.3555 - accuracy: 0.8865
20900/25000 [========================>.....] - ETA: 2s - loss: 0.3555 - accuracy: 0.8865
21000/25000 [========================>.....] - ETA: 2s - loss: 0.3555 - accuracy: 0.8866
21100/25000 [========================>.....] - ETA: 2s - loss: 0.3555 - accuracy: 0.8865
21200/25000 [========================>.....] - ETA: 2s - loss: 0.3556 - accuracy: 0.8865
21300/25000 [========================>.....] - ETA: 2s - loss: 0.3555 - accuracy: 0.8865
21400/25000 [========================>.....] - ETA: 2s - loss: 0.3555 - accuracy: 0.8865
21500/25000 [========================>.....] - ETA: 2s - loss: 0.3555 - accuracy: 0.8865
21600/25000 [========================>.....] - ETA: 2s - loss: 0.3555 - accuracy: 0.8865
21700/25000 [=========================>....] - ETA: 2s - loss: 0.3555 - accuracy: 0.8865
21800/25000 [=========================>....] - ETA: 2s - loss: 0.3555 - accuracy: 0.8865
21900/25000 [=========================>....] - ETA: 1s - loss: 0.3555 - accuracy: 0.8865
22000/25000 [=========================>....] - ETA: 1s - loss: 0.3555 - accuracy: 0.8865
22100/25000 [=========================>....] - ETA: 1s - loss: 0.3554 - accuracy: 0.8865
22200/25000 [=========================>....] - ETA: 1s - loss: 0.3554 - accuracy: 0.8865
22300/25000 [=========================>....] - ETA: 1s - loss: 0.3555 - accuracy: 0.8865
22400/25000 [=========================>....] - ETA: 1s - loss: 0.3555 - accuracy: 0.8865
22500/25000 [==========================>...] - ETA: 1s - loss: 0.3556 - accuracy: 0.8864
22600/25000 [==========================>...] - ETA: 1s - loss: 0.3556 - accuracy: 0.8864
22700/25000 [==========================>...] - ETA: 1s - loss: 0.3556 - accuracy: 0.8864
22800/25000 [==========================>...] - ETA: 1s - loss: 0.3556 - accuracy: 0.8864
22900/25000 [==========================>...] - ETA: 1s - loss: 0.3556 - accuracy: 0.8864
23000/25000 [==========================>...] - ETA: 1s - loss: 0.3556 - accuracy: 0.8864
23100/25000 [==========================>...] - ETA: 1s - loss: 0.3556 - accuracy: 0.8864
23200/25000 [==========================>...] - ETA: 1s - loss: 0.3556 - accuracy: 0.8864
23300/25000 [==========================>...] - ETA: 1s - loss: 0.3556 - accuracy: 0.8864
23400/25000 [===========================>..] - ETA: 1s - loss: 0.3557 - accuracy: 0.8864
23500/25000 [===========================>..] - ETA: 0s - loss: 0.3557 - accuracy: 0.8864
23600/25000 [===========================>..] - ETA: 0s - loss: 0.3557 - accuracy: 0.8864
23700/25000 [===========================>..] - ETA: 0s - loss: 0.3556 - accuracy: 0.8864
23800/25000 [===========================>..] - ETA: 0s - loss: 0.3556 - accuracy: 0.8864
23900/25000 [===========================>..] - ETA: 0s - loss: 0.3556 - accuracy: 0.8864
24000/25000 [===========================>..] - ETA: 0s - loss: 0.3556 - accuracy: 0.8864
24100/25000 [===========================>..] - ETA: 0s - loss: 0.3556 - accuracy: 0.8863
24200/25000 [============================>.] - ETA: 0s - loss: 0.3556 - accuracy: 0.8863
24300/25000 [============================>.] - ETA: 0s - loss: 0.3557 - accuracy: 0.8863
24400/25000 [============================>.] - ETA: 0s - loss: 0.3557 - accuracy: 0.8863
24500/25000 [============================>.] - ETA: 0s - loss: 0.3557 - accuracy: 0.8863
24600/25000 [============================>.] - ETA: 0s - loss: 0.3557 - accuracy: 0.8863
24700/25000 [============================>.] - ETA: 0s - loss: 0.3557 - accuracy: 0.8863
24800/25000 [============================>.] - ETA: 0s - loss: 0.3557 - accuracy: 0.8863
24900/25000 [============================>.] - ETA: 0s - loss: 0.3557 - accuracy: 0.8863
25000/25000 [==============================] - 20s 788us/step - loss: 0.3557 - accuracy: 0.8863 - val_loss: 0.3534 - val_accuracy: 0.8858
Epoch 7/10

  100/25000 [..............................] - ETA: 15s - loss: 0.3678 - accuracy: 0.8786
  200/25000 [..............................] - ETA: 15s - loss: 0.3625 - accuracy: 0.8814
  300/25000 [..............................] - ETA: 15s - loss: 0.3547 - accuracy: 0.8848
  400/25000 [..............................] - ETA: 15s - loss: 0.3523 - accuracy: 0.8861
  500/25000 [..............................] - ETA: 15s - loss: 0.3530 - accuracy: 0.8854
  600/25000 [..............................] - ETA: 15s - loss: 0.3537 - accuracy: 0.8852
  700/25000 [..............................] - ETA: 15s - loss: 0.3548 - accuracy: 0.8845
  800/25000 [..............................] - ETA: 15s - loss: 0.3542 - accuracy: 0.8848
  900/25000 [>.............................] - ETA: 15s - loss: 0.3551 - accuracy: 0.8844
 1000/25000 [>.............................] - ETA: 15s - loss: 0.3555 - accuracy: 0.8843
 1100/25000 [>.............................] - ETA: 14s - loss: 0.3538 - accuracy: 0.8855
 1200/25000 [>.............................] - ETA: 15s - loss: 0.3534 - accuracy: 0.8855
 1300/25000 [>.............................] - ETA: 14s - loss: 0.3531 - accuracy: 0.8857
 1400/25000 [>.............................] - ETA: 14s - loss: 0.3535 - accuracy: 0.8858
 1500/25000 [>.............................] - ETA: 14s - loss: 0.3526 - accuracy: 0.8865
 1600/25000 [>.............................] - ETA: 14s - loss: 0.3530 - accuracy: 0.8864
 1700/25000 [=>............................] - ETA: 14s - loss: 0.3533 - accuracy: 0.8862
 1800/25000 [=>............................] - ETA: 14s - loss: 0.3533 - accuracy: 0.8861
 1900/25000 [=>............................] - ETA: 14s - loss: 0.3530 - accuracy: 0.8861
 2000/25000 [=>............................] - ETA: 14s - loss: 0.3523 - accuracy: 0.8865
 2100/25000 [=>............................] - ETA: 14s - loss: 0.3525 - accuracy: 0.8863
 2200/25000 [=>............................] - ETA: 14s - loss: 0.3527 - accuracy: 0.8863
 2300/25000 [=>............................] - ETA: 14s - loss: 0.3527 - accuracy: 0.8862
 2400/25000 [=>............................] - ETA: 14s - loss: 0.3529 - accuracy: 0.8860
 2500/25000 [==>...........................] - ETA: 14s - loss: 0.3528 - accuracy: 0.8861
 2600/25000 [==>...........................] - ETA: 14s - loss: 0.3527 - accuracy: 0.8862
 2700/25000 [==>...........................] - ETA: 14s - loss: 0.3530 - accuracy: 0.8859
 2800/25000 [==>...........................] - ETA: 14s - loss: 0.3534 - accuracy: 0.8858
 2900/25000 [==>...........................] - ETA: 13s - loss: 0.3532 - accuracy: 0.8859
 3000/25000 [==>...........................] - ETA: 13s - loss: 0.3531 - accuracy: 0.8859
 3100/25000 [==>...........................] - ETA: 13s - loss: 0.3528 - accuracy: 0.8861
 3200/25000 [==>...........................] - ETA: 13s - loss: 0.3529 - accuracy: 0.8861
 3300/25000 [==>...........................] - ETA: 13s - loss: 0.3530 - accuracy: 0.8860
 3400/25000 [===>..........................] - ETA: 13s - loss: 0.3525 - accuracy: 0.8863
 3500/25000 [===>..........................] - ETA: 13s - loss: 0.3529 - accuracy: 0.8860
 3600/25000 [===>..........................] - ETA: 13s - loss: 0.3531 - accuracy: 0.8860
 3700/25000 [===>..........................] - ETA: 13s - loss: 0.3534 - accuracy: 0.8859
 3800/25000 [===>..........................] - ETA: 13s - loss: 0.3535 - accuracy: 0.8858
 3900/25000 [===>..........................] - ETA: 13s - loss: 0.3538 - accuracy: 0.8857
 4000/25000 [===>..........................] - ETA: 13s - loss: 0.3535 - accuracy: 0.8858
 4100/25000 [===>..........................] - ETA: 13s - loss: 0.3532 - accuracy: 0.8860
 4200/25000 [====>.........................] - ETA: 13s - loss: 0.3530 - accuracy: 0.8861
 4300/25000 [====>.........................] - ETA: 13s - loss: 0.3530 - accuracy: 0.8861
 4400/25000 [====>.........................] - ETA: 13s - loss: 0.3530 - accuracy: 0.8861
 4500/25000 [====>.........................] - ETA: 13s - loss: 0.3528 - accuracy: 0.8862
 4600/25000 [====>.........................] - ETA: 12s - loss: 0.3523 - accuracy: 0.8864
 4700/25000 [====>.........................] - ETA: 12s - loss: 0.3520 - accuracy: 0.8866
 4800/25000 [====>.........................] - ETA: 12s - loss: 0.3523 - accuracy: 0.8865
 4900/25000 [====>.........................] - ETA: 12s - loss: 0.3524 - accuracy: 0.8865
 5000/25000 [=====>........................] - ETA: 12s - loss: 0.3525 - accuracy: 0.8864
 5100/25000 [=====>........................] - ETA: 12s - loss: 0.3526 - accuracy: 0.8864
 5200/25000 [=====>........................] - ETA: 12s - loss: 0.3526 - accuracy: 0.8864
 5300/25000 [=====>........................] - ETA: 12s - loss: 0.3524 - accuracy: 0.8865
 5400/25000 [=====>........................] - ETA: 12s - loss: 0.3524 - accuracy: 0.8864
 5500/25000 [=====>........................] - ETA: 12s - loss: 0.3525 - accuracy: 0.8864
 5600/25000 [=====>........................] - ETA: 12s - loss: 0.3528 - accuracy: 0.8862
 5700/25000 [=====>........................] - ETA: 12s - loss: 0.3528 - accuracy: 0.8863
 5800/25000 [=====>........................] - ETA: 12s - loss: 0.3531 - accuracy: 0.8861
 5900/25000 [======>.......................] - ETA: 12s - loss: 0.3530 - accuracy: 0.8861
 6000/25000 [======>.......................] - ETA: 12s - loss: 0.3533 - accuracy: 0.8860
 6100/25000 [======>.......................] - ETA: 12s - loss: 0.3531 - accuracy: 0.8861
 6200/25000 [======>.......................] - ETA: 11s - loss: 0.3531 - accuracy: 0.8861
 6300/25000 [======>.......................] - ETA: 11s - loss: 0.3529 - accuracy: 0.8862
 6400/25000 [======>.......................] - ETA: 11s - loss: 0.3528 - accuracy: 0.8863
 6500/25000 [======>.......................] - ETA: 11s - loss: 0.3530 - accuracy: 0.8861
 6600/25000 [======>.......................] - ETA: 11s - loss: 0.3531 - accuracy: 0.8861
 6700/25000 [=======>......................] - ETA: 11s - loss: 0.3532 - accuracy: 0.8861
 6800/25000 [=======>......................] - ETA: 11s - loss: 0.3532 - accuracy: 0.8861
 6900/25000 [=======>......................] - ETA: 11s - loss: 0.3533 - accuracy: 0.8860
 7000/25000 [=======>......................] - ETA: 11s - loss: 0.3533 - accuracy: 0.8860
 7100/25000 [=======>......................] - ETA: 11s - loss: 0.3535 - accuracy: 0.8860
 7200/25000 [=======>......................] - ETA: 11s - loss: 0.3535 - accuracy: 0.8860
 7300/25000 [=======>......................] - ETA: 11s - loss: 0.3537 - accuracy: 0.8859
 7400/25000 [=======>......................] - ETA: 11s - loss: 0.3535 - accuracy: 0.8860
 7500/25000 [========>.....................] - ETA: 11s - loss: 0.3535 - accuracy: 0.8860
 7600/25000 [========>.....................] - ETA: 11s - loss: 0.3536 - accuracy: 0.8859
 7700/25000 [========>.....................] - ETA: 10s - loss: 0.3535 - accuracy: 0.8860
 7800/25000 [========>.....................] - ETA: 10s - loss: 0.3534 - accuracy: 0.8861
 7900/25000 [========>.....................] - ETA: 10s - loss: 0.3536 - accuracy: 0.8859
 8000/25000 [========>.....................] - ETA: 10s - loss: 0.3538 - accuracy: 0.8859
 8100/25000 [========>.....................] - ETA: 10s - loss: 0.3537 - accuracy: 0.8860
 8200/25000 [========>.....................] - ETA: 10s - loss: 0.3537 - accuracy: 0.8860
 8300/25000 [========>.....................] - ETA: 10s - loss: 0.3537 - accuracy: 0.8860
 8400/25000 [=========>....................] - ETA: 10s - loss: 0.3538 - accuracy: 0.8859
 8500/25000 [=========>....................] - ETA: 10s - loss: 0.3538 - accuracy: 0.8860
 8600/25000 [=========>....................] - ETA: 10s - loss: 0.3537 - accuracy: 0.8860
 8700/25000 [=========>....................] - ETA: 10s - loss: 0.3539 - accuracy: 0.8859
 8800/25000 [=========>....................] - ETA: 10s - loss: 0.3536 - accuracy: 0.8860
 8900/25000 [=========>....................] - ETA: 10s - loss: 0.3538 - accuracy: 0.8859
 9000/25000 [=========>....................] - ETA: 10s - loss: 0.3537 - accuracy: 0.8859
 9100/25000 [=========>....................] - ETA: 10s - loss: 0.3538 - accuracy: 0.8860
 9200/25000 [==========>...................] - ETA: 10s - loss: 0.3538 - accuracy: 0.8859
 9300/25000 [==========>...................] - ETA: 9s - loss: 0.3540 - accuracy: 0.8859 
 9400/25000 [==========>...................] - ETA: 9s - loss: 0.3539 - accuracy: 0.8859
 9500/25000 [==========>...................] - ETA: 9s - loss: 0.3538 - accuracy: 0.8860
 9600/25000 [==========>...................] - ETA: 9s - loss: 0.3538 - accuracy: 0.8860
 9700/25000 [==========>...................] - ETA: 9s - loss: 0.3538 - accuracy: 0.8860
 9800/25000 [==========>...................] - ETA: 9s - loss: 0.3538 - accuracy: 0.8860
 9900/25000 [==========>...................] - ETA: 9s - loss: 0.3538 - accuracy: 0.8860
10000/25000 [===========>..................] - ETA: 9s - loss: 0.3536 - accuracy: 0.8861
10100/25000 [===========>..................] - ETA: 9s - loss: 0.3534 - accuracy: 0.8862
10200/25000 [===========>..................] - ETA: 9s - loss: 0.3533 - accuracy: 0.8862
10300/25000 [===========>..................] - ETA: 9s - loss: 0.3531 - accuracy: 0.8863
10400/25000 [===========>..................] - ETA: 9s - loss: 0.3530 - accuracy: 0.8863
10500/25000 [===========>..................] - ETA: 9s - loss: 0.3531 - accuracy: 0.8863
10600/25000 [===========>..................] - ETA: 9s - loss: 0.3530 - accuracy: 0.8864
10700/25000 [===========>..................] - ETA: 9s - loss: 0.3530 - accuracy: 0.8863
10800/25000 [===========>..................] - ETA: 8s - loss: 0.3528 - accuracy: 0.8864
10900/25000 [============>.................] - ETA: 8s - loss: 0.3528 - accuracy: 0.8864
11000/25000 [============>.................] - ETA: 8s - loss: 0.3528 - accuracy: 0.8864
11100/25000 [============>.................] - ETA: 8s - loss: 0.3526 - accuracy: 0.8865
11200/25000 [============>.................] - ETA: 8s - loss: 0.3527 - accuracy: 0.8865
11300/25000 [============>.................] - ETA: 8s - loss: 0.3528 - accuracy: 0.8864
11400/25000 [============>.................] - ETA: 8s - loss: 0.3528 - accuracy: 0.8864
11500/25000 [============>.................] - ETA: 8s - loss: 0.3527 - accuracy: 0.8865
11600/25000 [============>.................] - ETA: 8s - loss: 0.3526 - accuracy: 0.8865
11700/25000 [=============>................] - ETA: 8s - loss: 0.3524 - accuracy: 0.8866
11800/25000 [=============>................] - ETA: 8s - loss: 0.3524 - accuracy: 0.8866
11900/25000 [=============>................] - ETA: 8s - loss: 0.3524 - accuracy: 0.8867
12000/25000 [=============>................] - ETA: 8s - loss: 0.3523 - accuracy: 0.8867
12100/25000 [=============>................] - ETA: 8s - loss: 0.3524 - accuracy: 0.8866
12200/25000 [=============>................] - ETA: 8s - loss: 0.3523 - accuracy: 0.8867
12300/25000 [=============>................] - ETA: 8s - loss: 0.3523 - accuracy: 0.8866
12400/25000 [=============>................] - ETA: 7s - loss: 0.3524 - accuracy: 0.8866
12500/25000 [==============>...............] - ETA: 7s - loss: 0.3525 - accuracy: 0.8865
12600/25000 [==============>...............] - ETA: 7s - loss: 0.3524 - accuracy: 0.8866
12700/25000 [==============>...............] - ETA: 7s - loss: 0.3525 - accuracy: 0.8866
12800/25000 [==============>...............] - ETA: 7s - loss: 0.3526 - accuracy: 0.8865
12900/25000 [==============>...............] - ETA: 7s - loss: 0.3526 - accuracy: 0.8865
13000/25000 [==============>...............] - ETA: 7s - loss: 0.3527 - accuracy: 0.8864
13100/25000 [==============>...............] - ETA: 7s - loss: 0.3527 - accuracy: 0.8864
13200/25000 [==============>...............] - ETA: 7s - loss: 0.3527 - accuracy: 0.8865
13300/25000 [==============>...............] - ETA: 7s - loss: 0.3528 - accuracy: 0.8864
13400/25000 [===============>..............] - ETA: 7s - loss: 0.3528 - accuracy: 0.8863
13500/25000 [===============>..............] - ETA: 7s - loss: 0.3528 - accuracy: 0.8863
13600/25000 [===============>..............] - ETA: 7s - loss: 0.3527 - accuracy: 0.8864
13700/25000 [===============>..............] - ETA: 7s - loss: 0.3528 - accuracy: 0.8863
13800/25000 [===============>..............] - ETA: 7s - loss: 0.3527 - accuracy: 0.8864
13900/25000 [===============>..............] - ETA: 7s - loss: 0.3526 - accuracy: 0.8865
14000/25000 [===============>..............] - ETA: 6s - loss: 0.3527 - accuracy: 0.8864
14100/25000 [===============>..............] - ETA: 6s - loss: 0.3527 - accuracy: 0.8864
14200/25000 [================>.............] - ETA: 6s - loss: 0.3527 - accuracy: 0.8864
14300/25000 [================>.............] - ETA: 6s - loss: 0.3527 - accuracy: 0.8864
14400/25000 [================>.............] - ETA: 6s - loss: 0.3528 - accuracy: 0.8864
14500/25000 [================>.............] - ETA: 6s - loss: 0.3528 - accuracy: 0.8864
14600/25000 [================>.............] - ETA: 6s - loss: 0.3527 - accuracy: 0.8865
14700/25000 [================>.............] - ETA: 6s - loss: 0.3527 - accuracy: 0.8865
14800/25000 [================>.............] - ETA: 6s - loss: 0.3527 - accuracy: 0.8865
14900/25000 [================>.............] - ETA: 6s - loss: 0.3528 - accuracy: 0.8864
15000/25000 [=================>............] - ETA: 6s - loss: 0.3528 - accuracy: 0.8865
15100/25000 [=================>............] - ETA: 6s - loss: 0.3528 - accuracy: 0.8864
15200/25000 [=================>............] - ETA: 6s - loss: 0.3529 - accuracy: 0.8864
15300/25000 [=================>............] - ETA: 6s - loss: 0.3529 - accuracy: 0.8864
15400/25000 [=================>............] - ETA: 6s - loss: 0.3530 - accuracy: 0.8864
15500/25000 [=================>............] - ETA: 6s - loss: 0.3529 - accuracy: 0.8864
15600/25000 [=================>............] - ETA: 5s - loss: 0.3528 - accuracy: 0.8865
15700/25000 [=================>............] - ETA: 5s - loss: 0.3528 - accuracy: 0.8864
15800/25000 [=================>............] - ETA: 5s - loss: 0.3528 - accuracy: 0.8864
15900/25000 [==================>...........] - ETA: 5s - loss: 0.3528 - accuracy: 0.8864
16000/25000 [==================>...........] - ETA: 5s - loss: 0.3529 - accuracy: 0.8864
16100/25000 [==================>...........] - ETA: 5s - loss: 0.3530 - accuracy: 0.8863
16200/25000 [==================>...........] - ETA: 5s - loss: 0.3531 - accuracy: 0.8862
16300/25000 [==================>...........] - ETA: 5s - loss: 0.3532 - accuracy: 0.8862
16400/25000 [==================>...........] - ETA: 5s - loss: 0.3532 - accuracy: 0.8862
16500/25000 [==================>...........] - ETA: 5s - loss: 0.3532 - accuracy: 0.8862
16600/25000 [==================>...........] - ETA: 5s - loss: 0.3533 - accuracy: 0.8862
16700/25000 [===================>..........] - ETA: 5s - loss: 0.3532 - accuracy: 0.8862
16800/25000 [===================>..........] - ETA: 5s - loss: 0.3531 - accuracy: 0.8862
16900/25000 [===================>..........] - ETA: 5s - loss: 0.3531 - accuracy: 0.8862
17000/25000 [===================>..........] - ETA: 5s - loss: 0.3531 - accuracy: 0.8862
17100/25000 [===================>..........] - ETA: 4s - loss: 0.3530 - accuracy: 0.8862
17200/25000 [===================>..........] - ETA: 4s - loss: 0.3531 - accuracy: 0.8862
17300/25000 [===================>..........] - ETA: 4s - loss: 0.3530 - accuracy: 0.8863
17400/25000 [===================>..........] - ETA: 4s - loss: 0.3531 - accuracy: 0.8862
17500/25000 [====================>.........] - ETA: 4s - loss: 0.3531 - accuracy: 0.8862
17600/25000 [====================>.........] - ETA: 4s - loss: 0.3532 - accuracy: 0.8861
17700/25000 [====================>.........] - ETA: 4s - loss: 0.3532 - accuracy: 0.8861
17800/25000 [====================>.........] - ETA: 4s - loss: 0.3532 - accuracy: 0.8861
17900/25000 [====================>.........] - ETA: 4s - loss: 0.3533 - accuracy: 0.8861
18000/25000 [====================>.........] - ETA: 4s - loss: 0.3533 - accuracy: 0.8861
18100/25000 [====================>.........] - ETA: 4s - loss: 0.3533 - accuracy: 0.8861
18200/25000 [====================>.........] - ETA: 4s - loss: 0.3533 - accuracy: 0.8861
18300/25000 [====================>.........] - ETA: 4s - loss: 0.3533 - accuracy: 0.8861
18400/25000 [=====================>........] - ETA: 4s - loss: 0.3532 - accuracy: 0.8861
18500/25000 [=====================>........] - ETA: 4s - loss: 0.3532 - accuracy: 0.8861
18600/25000 [=====================>........] - ETA: 4s - loss: 0.3532 - accuracy: 0.8861
18700/25000 [=====================>........] - ETA: 3s - loss: 0.3532 - accuracy: 0.8861
18800/25000 [=====================>........] - ETA: 3s - loss: 0.3532 - accuracy: 0.8861
18900/25000 [=====================>........] - ETA: 3s - loss: 0.3533 - accuracy: 0.8861
19000/25000 [=====================>........] - ETA: 3s - loss: 0.3532 - accuracy: 0.8861
19100/25000 [=====================>........] - ETA: 3s - loss: 0.3532 - accuracy: 0.8861
19200/25000 [======================>.......] - ETA: 3s - loss: 0.3531 - accuracy: 0.8861
19300/25000 [======================>.......] - ETA: 3s - loss: 0.3532 - accuracy: 0.8861
19400/25000 [======================>.......] - ETA: 3s - loss: 0.3532 - accuracy: 0.8861
19500/25000 [======================>.......] - ETA: 3s - loss: 0.3533 - accuracy: 0.8861
19600/25000 [======================>.......] - ETA: 3s - loss: 0.3533 - accuracy: 0.8861
19700/25000 [======================>.......] - ETA: 3s - loss: 0.3532 - accuracy: 0.8861
19800/25000 [======================>.......] - ETA: 3s - loss: 0.3532 - accuracy: 0.8861
19900/25000 [======================>.......] - ETA: 3s - loss: 0.3532 - accuracy: 0.8860
20000/25000 [=======================>......] - ETA: 3s - loss: 0.3532 - accuracy: 0.8860
20100/25000 [=======================>......] - ETA: 3s - loss: 0.3531 - accuracy: 0.8861
20200/25000 [=======================>......] - ETA: 3s - loss: 0.3530 - accuracy: 0.8861
20300/25000 [=======================>......] - ETA: 2s - loss: 0.3530 - accuracy: 0.8861
20400/25000 [=======================>......] - ETA: 2s - loss: 0.3529 - accuracy: 0.8861
20500/25000 [=======================>......] - ETA: 2s - loss: 0.3529 - accuracy: 0.8862
20600/25000 [=======================>......] - ETA: 2s - loss: 0.3528 - accuracy: 0.8862
20700/25000 [=======================>......] - ETA: 2s - loss: 0.3529 - accuracy: 0.8862
20800/25000 [=======================>......] - ETA: 2s - loss: 0.3529 - accuracy: 0.8861
20900/25000 [========================>.....] - ETA: 2s - loss: 0.3529 - accuracy: 0.8861
21000/25000 [========================>.....] - ETA: 2s - loss: 0.3529 - accuracy: 0.8861
21100/25000 [========================>.....] - ETA: 2s - loss: 0.3529 - accuracy: 0.8861
21200/25000 [========================>.....] - ETA: 2s - loss: 0.3527 - accuracy: 0.8862
21300/25000 [========================>.....] - ETA: 2s - loss: 0.3526 - accuracy: 0.8862
21400/25000 [========================>.....] - ETA: 2s - loss: 0.3527 - accuracy: 0.8862
21500/25000 [========================>.....] - ETA: 2s - loss: 0.3526 - accuracy: 0.8862
21600/25000 [========================>.....] - ETA: 2s - loss: 0.3526 - accuracy: 0.8862
21700/25000 [=========================>....] - ETA: 2s - loss: 0.3525 - accuracy: 0.8863
21800/25000 [=========================>....] - ETA: 2s - loss: 0.3526 - accuracy: 0.8863
21900/25000 [=========================>....] - ETA: 1s - loss: 0.3525 - accuracy: 0.8863
22000/25000 [=========================>....] - ETA: 1s - loss: 0.3525 - accuracy: 0.8863
22100/25000 [=========================>....] - ETA: 1s - loss: 0.3525 - accuracy: 0.8863
22200/25000 [=========================>....] - ETA: 1s - loss: 0.3524 - accuracy: 0.8863
22300/25000 [=========================>....] - ETA: 1s - loss: 0.3524 - accuracy: 0.8863
22400/25000 [=========================>....] - ETA: 1s - loss: 0.3524 - accuracy: 0.8863
22500/25000 [==========================>...] - ETA: 1s - loss: 0.3524 - accuracy: 0.8863
22600/25000 [==========================>...] - ETA: 1s - loss: 0.3524 - accuracy: 0.8863
22700/25000 [==========================>...] - ETA: 1s - loss: 0.3524 - accuracy: 0.8863
22800/25000 [==========================>...] - ETA: 1s - loss: 0.3523 - accuracy: 0.8864
22900/25000 [==========================>...] - ETA: 1s - loss: 0.3524 - accuracy: 0.8864
23000/25000 [==========================>...] - ETA: 1s - loss: 0.3524 - accuracy: 0.8863
23100/25000 [==========================>...] - ETA: 1s - loss: 0.3524 - accuracy: 0.8863
23200/25000 [==========================>...] - ETA: 1s - loss: 0.3524 - accuracy: 0.8863
23300/25000 [==========================>...] - ETA: 1s - loss: 0.3523 - accuracy: 0.8864
23400/25000 [===========================>..] - ETA: 1s - loss: 0.3523 - accuracy: 0.8864
23500/25000 [===========================>..] - ETA: 0s - loss: 0.3523 - accuracy: 0.8864
23600/25000 [===========================>..] - ETA: 0s - loss: 0.3522 - accuracy: 0.8864
23700/25000 [===========================>..] - ETA: 0s - loss: 0.3521 - accuracy: 0.8865
23800/25000 [===========================>..] - ETA: 0s - loss: 0.3521 - accuracy: 0.8864
23900/25000 [===========================>..] - ETA: 0s - loss: 0.3520 - accuracy: 0.8865
24000/25000 [===========================>..] - ETA: 0s - loss: 0.3520 - accuracy: 0.8865
24100/25000 [===========================>..] - ETA: 0s - loss: 0.3521 - accuracy: 0.8864
24200/25000 [============================>.] - ETA: 0s - loss: 0.3522 - accuracy: 0.8864
24300/25000 [============================>.] - ETA: 0s - loss: 0.3522 - accuracy: 0.8864
24400/25000 [============================>.] - ETA: 0s - loss: 0.3522 - accuracy: 0.8864
24500/25000 [============================>.] - ETA: 0s - loss: 0.3523 - accuracy: 0.8863
24600/25000 [============================>.] - ETA: 0s - loss: 0.3523 - accuracy: 0.8863
24700/25000 [============================>.] - ETA: 0s - loss: 0.3524 - accuracy: 0.8863
24800/25000 [============================>.] - ETA: 0s - loss: 0.3523 - accuracy: 0.8863
24900/25000 [============================>.] - ETA: 0s - loss: 0.3523 - accuracy: 0.8863
25000/25000 [==============================] - 20s 796us/step - loss: 0.3524 - accuracy: 0.8863 - val_loss: 0.3513 - val_accuracy: 0.8858
Epoch 8/10

  100/25000 [..............................] - ETA: 15s - loss: 0.3569 - accuracy: 0.8814
  200/25000 [..............................] - ETA: 15s - loss: 0.3586 - accuracy: 0.8829
  300/25000 [..............................] - ETA: 15s - loss: 0.3575 - accuracy: 0.8848
  400/25000 [..............................] - ETA: 15s - loss: 0.3596 - accuracy: 0.8829
  500/25000 [..............................] - ETA: 15s - loss: 0.3543 - accuracy: 0.8851
  600/25000 [..............................] - ETA: 15s - loss: 0.3530 - accuracy: 0.8862
  700/25000 [..............................] - ETA: 15s - loss: 0.3517 - accuracy: 0.8865
  800/25000 [..............................] - ETA: 15s - loss: 0.3533 - accuracy: 0.8859
  900/25000 [>.............................] - ETA: 15s - loss: 0.3522 - accuracy: 0.8863
 1000/25000 [>.............................] - ETA: 15s - loss: 0.3508 - accuracy: 0.8870
 1100/25000 [>.............................] - ETA: 15s - loss: 0.3497 - accuracy: 0.8875
 1200/25000 [>.............................] - ETA: 15s - loss: 0.3516 - accuracy: 0.8867
 1300/25000 [>.............................] - ETA: 14s - loss: 0.3513 - accuracy: 0.8868
 1400/25000 [>.............................] - ETA: 14s - loss: 0.3505 - accuracy: 0.8870
 1500/25000 [>.............................] - ETA: 14s - loss: 0.3518 - accuracy: 0.8865
 1600/25000 [>.............................] - ETA: 14s - loss: 0.3526 - accuracy: 0.8862
 1700/25000 [=>............................] - ETA: 14s - loss: 0.3530 - accuracy: 0.8860
 1800/25000 [=>............................] - ETA: 14s - loss: 0.3527 - accuracy: 0.8860
 1900/25000 [=>............................] - ETA: 14s - loss: 0.3523 - accuracy: 0.8863
 2000/25000 [=>............................] - ETA: 14s - loss: 0.3525 - accuracy: 0.8862
 2100/25000 [=>............................] - ETA: 14s - loss: 0.3531 - accuracy: 0.8859
 2200/25000 [=>............................] - ETA: 14s - loss: 0.3525 - accuracy: 0.8861
 2300/25000 [=>............................] - ETA: 14s - loss: 0.3525 - accuracy: 0.8861
 2400/25000 [=>............................] - ETA: 14s - loss: 0.3524 - accuracy: 0.8862
 2500/25000 [==>...........................] - ETA: 14s - loss: 0.3525 - accuracy: 0.8861
 2600/25000 [==>...........................] - ETA: 14s - loss: 0.3525 - accuracy: 0.8860
 2700/25000 [==>...........................] - ETA: 13s - loss: 0.3526 - accuracy: 0.8859
 2800/25000 [==>...........................] - ETA: 13s - loss: 0.3525 - accuracy: 0.8859
 2900/25000 [==>...........................] - ETA: 13s - loss: 0.3521 - accuracy: 0.8862
 3000/25000 [==>...........................] - ETA: 13s - loss: 0.3521 - accuracy: 0.8862
 3100/25000 [==>...........................] - ETA: 13s - loss: 0.3520 - accuracy: 0.8861
 3200/25000 [==>...........................] - ETA: 13s - loss: 0.3520 - accuracy: 0.8861
 3300/25000 [==>...........................] - ETA: 13s - loss: 0.3515 - accuracy: 0.8864
 3400/25000 [===>..........................] - ETA: 13s - loss: 0.3512 - accuracy: 0.8866
 3500/25000 [===>..........................] - ETA: 13s - loss: 0.3505 - accuracy: 0.8869
 3600/25000 [===>..........................] - ETA: 13s - loss: 0.3504 - accuracy: 0.8869
 3700/25000 [===>..........................] - ETA: 13s - loss: 0.3505 - accuracy: 0.8869
 3800/25000 [===>..........................] - ETA: 13s - loss: 0.3506 - accuracy: 0.8868
 3900/25000 [===>..........................] - ETA: 13s - loss: 0.3507 - accuracy: 0.8867
 4000/25000 [===>..........................] - ETA: 13s - loss: 0.3512 - accuracy: 0.8865
 4100/25000 [===>..........................] - ETA: 13s - loss: 0.3510 - accuracy: 0.8866
 4200/25000 [====>.........................] - ETA: 13s - loss: 0.3509 - accuracy: 0.8866
 4300/25000 [====>.........................] - ETA: 13s - loss: 0.3510 - accuracy: 0.8866
 4400/25000 [====>.........................] - ETA: 13s - loss: 0.3511 - accuracy: 0.8865
 4500/25000 [====>.........................] - ETA: 12s - loss: 0.3507 - accuracy: 0.8867
 4600/25000 [====>.........................] - ETA: 12s - loss: 0.3506 - accuracy: 0.8867
 4700/25000 [====>.........................] - ETA: 12s - loss: 0.3509 - accuracy: 0.8866
 4800/25000 [====>.........................] - ETA: 12s - loss: 0.3507 - accuracy: 0.8866
 4900/25000 [====>.........................] - ETA: 12s - loss: 0.3507 - accuracy: 0.8866
 5000/25000 [=====>........................] - ETA: 12s - loss: 0.3510 - accuracy: 0.8865
 5100/25000 [=====>........................] - ETA: 12s - loss: 0.3511 - accuracy: 0.8864
 5200/25000 [=====>........................] - ETA: 12s - loss: 0.3513 - accuracy: 0.8863
 5300/25000 [=====>........................] - ETA: 12s - loss: 0.3516 - accuracy: 0.8862
 5400/25000 [=====>........................] - ETA: 12s - loss: 0.3514 - accuracy: 0.8863
 5500/25000 [=====>........................] - ETA: 12s - loss: 0.3517 - accuracy: 0.8861
 5600/25000 [=====>........................] - ETA: 12s - loss: 0.3517 - accuracy: 0.8861
 5700/25000 [=====>........................] - ETA: 12s - loss: 0.3515 - accuracy: 0.8861
 5800/25000 [=====>........................] - ETA: 12s - loss: 0.3517 - accuracy: 0.8860
 5900/25000 [======>.......................] - ETA: 12s - loss: 0.3515 - accuracy: 0.8861
 6000/25000 [======>.......................] - ETA: 11s - loss: 0.3515 - accuracy: 0.8861
 6100/25000 [======>.......................] - ETA: 11s - loss: 0.3513 - accuracy: 0.8862
 6200/25000 [======>.......................] - ETA: 11s - loss: 0.3513 - accuracy: 0.8863
 6300/25000 [======>.......................] - ETA: 11s - loss: 0.3510 - accuracy: 0.8864
 6400/25000 [======>.......................] - ETA: 11s - loss: 0.3509 - accuracy: 0.8865
 6500/25000 [======>.......................] - ETA: 11s - loss: 0.3508 - accuracy: 0.8865
 6600/25000 [======>.......................] - ETA: 11s - loss: 0.3509 - accuracy: 0.8865
 6700/25000 [=======>......................] - ETA: 11s - loss: 0.3508 - accuracy: 0.8865
 6800/25000 [=======>......................] - ETA: 11s - loss: 0.3510 - accuracy: 0.8864
 6900/25000 [=======>......................] - ETA: 11s - loss: 0.3509 - accuracy: 0.8865
 7000/25000 [=======>......................] - ETA: 11s - loss: 0.3511 - accuracy: 0.8864
 7100/25000 [=======>......................] - ETA: 11s - loss: 0.3509 - accuracy: 0.8866
 7200/25000 [=======>......................] - ETA: 11s - loss: 0.3509 - accuracy: 0.8865
 7300/25000 [=======>......................] - ETA: 11s - loss: 0.3509 - accuracy: 0.8865
 7400/25000 [=======>......................] - ETA: 11s - loss: 0.3509 - accuracy: 0.8865
 7500/25000 [========>.....................] - ETA: 11s - loss: 0.3509 - accuracy: 0.8864
 7600/25000 [========>.....................] - ETA: 11s - loss: 0.3510 - accuracy: 0.8864
 7700/25000 [========>.....................] - ETA: 10s - loss: 0.3509 - accuracy: 0.8864
 7800/25000 [========>.....................] - ETA: 10s - loss: 0.3505 - accuracy: 0.8866
 7900/25000 [========>.....................] - ETA: 10s - loss: 0.3506 - accuracy: 0.8866
 8000/25000 [========>.....................] - ETA: 10s - loss: 0.3507 - accuracy: 0.8865
 8100/25000 [========>.....................] - ETA: 10s - loss: 0.3509 - accuracy: 0.8864
 8200/25000 [========>.....................] - ETA: 10s - loss: 0.3512 - accuracy: 0.8863
 8300/25000 [========>.....................] - ETA: 10s - loss: 0.3510 - accuracy: 0.8864
 8400/25000 [=========>....................] - ETA: 10s - loss: 0.3511 - accuracy: 0.8863
 8500/25000 [=========>....................] - ETA: 10s - loss: 0.3508 - accuracy: 0.8865
 8600/25000 [=========>....................] - ETA: 10s - loss: 0.3509 - accuracy: 0.8864
 8700/25000 [=========>....................] - ETA: 10s - loss: 0.3509 - accuracy: 0.8864
 8800/25000 [=========>....................] - ETA: 10s - loss: 0.3508 - accuracy: 0.8864
 8900/25000 [=========>....................] - ETA: 10s - loss: 0.3506 - accuracy: 0.8865
 9000/25000 [=========>....................] - ETA: 10s - loss: 0.3505 - accuracy: 0.8866
 9100/25000 [=========>....................] - ETA: 10s - loss: 0.3504 - accuracy: 0.8867
 9200/25000 [==========>...................] - ETA: 9s - loss: 0.3502 - accuracy: 0.8867 
 9300/25000 [==========>...................] - ETA: 9s - loss: 0.3501 - accuracy: 0.8868
 9400/25000 [==========>...................] - ETA: 9s - loss: 0.3499 - accuracy: 0.8869
 9500/25000 [==========>...................] - ETA: 9s - loss: 0.3501 - accuracy: 0.8868
 9600/25000 [==========>...................] - ETA: 9s - loss: 0.3500 - accuracy: 0.8868
 9700/25000 [==========>...................] - ETA: 9s - loss: 0.3499 - accuracy: 0.8869
 9800/25000 [==========>...................] - ETA: 9s - loss: 0.3499 - accuracy: 0.8869
 9900/25000 [==========>...................] - ETA: 9s - loss: 0.3499 - accuracy: 0.8869
10000/25000 [===========>..................] - ETA: 9s - loss: 0.3500 - accuracy: 0.8869
10100/25000 [===========>..................] - ETA: 9s - loss: 0.3501 - accuracy: 0.8868
10200/25000 [===========>..................] - ETA: 9s - loss: 0.3501 - accuracy: 0.8868
10300/25000 [===========>..................] - ETA: 9s - loss: 0.3502 - accuracy: 0.8868
10400/25000 [===========>..................] - ETA: 9s - loss: 0.3503 - accuracy: 0.8867
10500/25000 [===========>..................] - ETA: 9s - loss: 0.3504 - accuracy: 0.8867
10600/25000 [===========>..................] - ETA: 9s - loss: 0.3504 - accuracy: 0.8867
10700/25000 [===========>..................] - ETA: 9s - loss: 0.3505 - accuracy: 0.8866
10800/25000 [===========>..................] - ETA: 8s - loss: 0.3505 - accuracy: 0.8866
10900/25000 [============>.................] - ETA: 8s - loss: 0.3506 - accuracy: 0.8866
11000/25000 [============>.................] - ETA: 8s - loss: 0.3505 - accuracy: 0.8866
11100/25000 [============>.................] - ETA: 8s - loss: 0.3504 - accuracy: 0.8866
11200/25000 [============>.................] - ETA: 8s - loss: 0.3505 - accuracy: 0.8866
11300/25000 [============>.................] - ETA: 8s - loss: 0.3505 - accuracy: 0.8866
11400/25000 [============>.................] - ETA: 8s - loss: 0.3505 - accuracy: 0.8866
11500/25000 [============>.................] - ETA: 8s - loss: 0.3506 - accuracy: 0.8866
11600/25000 [============>.................] - ETA: 8s - loss: 0.3507 - accuracy: 0.8865
11700/25000 [=============>................] - ETA: 8s - loss: 0.3507 - accuracy: 0.8865
11800/25000 [=============>................] - ETA: 8s - loss: 0.3506 - accuracy: 0.8865
11900/25000 [=============>................] - ETA: 8s - loss: 0.3507 - accuracy: 0.8865
12000/25000 [=============>................] - ETA: 8s - loss: 0.3507 - accuracy: 0.8865
12100/25000 [=============>................] - ETA: 8s - loss: 0.3506 - accuracy: 0.8866
12200/25000 [=============>................] - ETA: 8s - loss: 0.3507 - accuracy: 0.8865
12300/25000 [=============>................] - ETA: 8s - loss: 0.3507 - accuracy: 0.8865
12400/25000 [=============>................] - ETA: 7s - loss: 0.3507 - accuracy: 0.8865
12500/25000 [==============>...............] - ETA: 7s - loss: 0.3507 - accuracy: 0.8865
12600/25000 [==============>...............] - ETA: 7s - loss: 0.3508 - accuracy: 0.8864
12700/25000 [==============>...............] - ETA: 7s - loss: 0.3508 - accuracy: 0.8864
12800/25000 [==============>...............] - ETA: 7s - loss: 0.3506 - accuracy: 0.8865
12900/25000 [==============>...............] - ETA: 7s - loss: 0.3506 - accuracy: 0.8865
13000/25000 [==============>...............] - ETA: 7s - loss: 0.3507 - accuracy: 0.8865
13100/25000 [==============>...............] - ETA: 7s - loss: 0.3507 - accuracy: 0.8865
13200/25000 [==============>...............] - ETA: 7s - loss: 0.3507 - accuracy: 0.8865
13300/25000 [==============>...............] - ETA: 7s - loss: 0.3507 - accuracy: 0.8865
13400/25000 [===============>..............] - ETA: 7s - loss: 0.3507 - accuracy: 0.8865
13500/25000 [===============>..............] - ETA: 7s - loss: 0.3506 - accuracy: 0.8865
13600/25000 [===============>..............] - ETA: 7s - loss: 0.3506 - accuracy: 0.8865
13700/25000 [===============>..............] - ETA: 7s - loss: 0.3507 - accuracy: 0.8865
13800/25000 [===============>..............] - ETA: 7s - loss: 0.3507 - accuracy: 0.8865
13900/25000 [===============>..............] - ETA: 7s - loss: 0.3506 - accuracy: 0.8865
14000/25000 [===============>..............] - ETA: 6s - loss: 0.3506 - accuracy: 0.8865
14100/25000 [===============>..............] - ETA: 6s - loss: 0.3506 - accuracy: 0.8865
14200/25000 [================>.............] - ETA: 6s - loss: 0.3506 - accuracy: 0.8864
14300/25000 [================>.............] - ETA: 6s - loss: 0.3508 - accuracy: 0.8864
14400/25000 [================>.............] - ETA: 6s - loss: 0.3507 - accuracy: 0.8864
14500/25000 [================>.............] - ETA: 6s - loss: 0.3506 - accuracy: 0.8864
14600/25000 [================>.............] - ETA: 6s - loss: 0.3505 - accuracy: 0.8865
14700/25000 [================>.............] - ETA: 6s - loss: 0.3505 - accuracy: 0.8865
14800/25000 [================>.............] - ETA: 6s - loss: 0.3505 - accuracy: 0.8865
14900/25000 [================>.............] - ETA: 6s - loss: 0.3505 - accuracy: 0.8865
15000/25000 [=================>............] - ETA: 6s - loss: 0.3505 - accuracy: 0.8865
15100/25000 [=================>............] - ETA: 6s - loss: 0.3504 - accuracy: 0.8865
15200/25000 [=================>............] - ETA: 6s - loss: 0.3503 - accuracy: 0.8866
15300/25000 [=================>............] - ETA: 6s - loss: 0.3503 - accuracy: 0.8866
15400/25000 [=================>............] - ETA: 6s - loss: 0.3503 - accuracy: 0.8866
15500/25000 [=================>............] - ETA: 6s - loss: 0.3504 - accuracy: 0.8865
15600/25000 [=================>............] - ETA: 5s - loss: 0.3505 - accuracy: 0.8865
15700/25000 [=================>............] - ETA: 5s - loss: 0.3505 - accuracy: 0.8865
15800/25000 [=================>............] - ETA: 5s - loss: 0.3505 - accuracy: 0.8865
15900/25000 [==================>...........] - ETA: 5s - loss: 0.3505 - accuracy: 0.8865
16000/25000 [==================>...........] - ETA: 5s - loss: 0.3505 - accuracy: 0.8865
16100/25000 [==================>...........] - ETA: 5s - loss: 0.3505 - accuracy: 0.8865
16200/25000 [==================>...........] - ETA: 5s - loss: 0.3506 - accuracy: 0.8864
16300/25000 [==================>...........] - ETA: 5s - loss: 0.3505 - accuracy: 0.8865
16400/25000 [==================>...........] - ETA: 5s - loss: 0.3505 - accuracy: 0.8865
16500/25000 [==================>...........] - ETA: 5s - loss: 0.3505 - accuracy: 0.8865
16600/25000 [==================>...........] - ETA: 5s - loss: 0.3504 - accuracy: 0.8866
16700/25000 [===================>..........] - ETA: 5s - loss: 0.3503 - accuracy: 0.8866
16800/25000 [===================>..........] - ETA: 5s - loss: 0.3504 - accuracy: 0.8866
16900/25000 [===================>..........] - ETA: 5s - loss: 0.3504 - accuracy: 0.8866
17000/25000 [===================>..........] - ETA: 5s - loss: 0.3504 - accuracy: 0.8865
17100/25000 [===================>..........] - ETA: 5s - loss: 0.3504 - accuracy: 0.8865
17200/25000 [===================>..........] - ETA: 4s - loss: 0.3505 - accuracy: 0.8865
17300/25000 [===================>..........] - ETA: 4s - loss: 0.3506 - accuracy: 0.8865
17400/25000 [===================>..........] - ETA: 4s - loss: 0.3506 - accuracy: 0.8864
17500/25000 [====================>.........] - ETA: 4s - loss: 0.3508 - accuracy: 0.8864
17600/25000 [====================>.........] - ETA: 4s - loss: 0.3508 - accuracy: 0.8863
17700/25000 [====================>.........] - ETA: 4s - loss: 0.3508 - accuracy: 0.8863
17800/25000 [====================>.........] - ETA: 4s - loss: 0.3507 - accuracy: 0.8864
17900/25000 [====================>.........] - ETA: 4s - loss: 0.3508 - accuracy: 0.8863
18000/25000 [====================>.........] - ETA: 4s - loss: 0.3509 - accuracy: 0.8863
18100/25000 [====================>.........] - ETA: 4s - loss: 0.3509 - accuracy: 0.8863
18200/25000 [====================>.........] - ETA: 4s - loss: 0.3508 - accuracy: 0.8863
18300/25000 [====================>.........] - ETA: 4s - loss: 0.3508 - accuracy: 0.8863
18400/25000 [=====================>........] - ETA: 4s - loss: 0.3507 - accuracy: 0.8864
18500/25000 [=====================>........] - ETA: 4s - loss: 0.3508 - accuracy: 0.8864
18600/25000 [=====================>........] - ETA: 4s - loss: 0.3508 - accuracy: 0.8863
18700/25000 [=====================>........] - ETA: 3s - loss: 0.3509 - accuracy: 0.8863
18800/25000 [=====================>........] - ETA: 3s - loss: 0.3509 - accuracy: 0.8863
18900/25000 [=====================>........] - ETA: 3s - loss: 0.3508 - accuracy: 0.8863
19000/25000 [=====================>........] - ETA: 3s - loss: 0.3508 - accuracy: 0.8864
19100/25000 [=====================>........] - ETA: 3s - loss: 0.3508 - accuracy: 0.8863
19200/25000 [======================>.......] - ETA: 3s - loss: 0.3509 - accuracy: 0.8863
19300/25000 [======================>.......] - ETA: 3s - loss: 0.3508 - accuracy: 0.8863
19400/25000 [======================>.......] - ETA: 3s - loss: 0.3508 - accuracy: 0.8864
19500/25000 [======================>.......] - ETA: 3s - loss: 0.3507 - accuracy: 0.8864
19600/25000 [======================>.......] - ETA: 3s - loss: 0.3507 - accuracy: 0.8864
19700/25000 [======================>.......] - ETA: 3s - loss: 0.3506 - accuracy: 0.8864
19800/25000 [======================>.......] - ETA: 3s - loss: 0.3506 - accuracy: 0.8865
19900/25000 [======================>.......] - ETA: 3s - loss: 0.3506 - accuracy: 0.8865
20000/25000 [=======================>......] - ETA: 3s - loss: 0.3506 - accuracy: 0.8864
20100/25000 [=======================>......] - ETA: 3s - loss: 0.3506 - accuracy: 0.8864
20200/25000 [=======================>......] - ETA: 3s - loss: 0.3506 - accuracy: 0.8864
20300/25000 [=======================>......] - ETA: 2s - loss: 0.3507 - accuracy: 0.8864
20400/25000 [=======================>......] - ETA: 2s - loss: 0.3505 - accuracy: 0.8865
20500/25000 [=======================>......] - ETA: 2s - loss: 0.3506 - accuracy: 0.8864
20600/25000 [=======================>......] - ETA: 2s - loss: 0.3507 - accuracy: 0.8864
20700/25000 [=======================>......] - ETA: 2s - loss: 0.3507 - accuracy: 0.8864
20800/25000 [=======================>......] - ETA: 2s - loss: 0.3507 - accuracy: 0.8864
20900/25000 [========================>.....] - ETA: 2s - loss: 0.3506 - accuracy: 0.8864
21000/25000 [========================>.....] - ETA: 2s - loss: 0.3506 - accuracy: 0.8864
21100/25000 [========================>.....] - ETA: 2s - loss: 0.3507 - accuracy: 0.8864
21200/25000 [========================>.....] - ETA: 2s - loss: 0.3506 - accuracy: 0.8865
21300/25000 [========================>.....] - ETA: 2s - loss: 0.3506 - accuracy: 0.8864
21400/25000 [========================>.....] - ETA: 2s - loss: 0.3504 - accuracy: 0.8865
21500/25000 [========================>.....] - ETA: 2s - loss: 0.3504 - accuracy: 0.8865
21600/25000 [========================>.....] - ETA: 2s - loss: 0.3504 - accuracy: 0.8865
21700/25000 [=========================>....] - ETA: 2s - loss: 0.3503 - accuracy: 0.8865
21800/25000 [=========================>....] - ETA: 2s - loss: 0.3504 - accuracy: 0.8865
21900/25000 [=========================>....] - ETA: 1s - loss: 0.3503 - accuracy: 0.8865
22000/25000 [=========================>....] - ETA: 1s - loss: 0.3503 - accuracy: 0.8866
22100/25000 [=========================>....] - ETA: 1s - loss: 0.3504 - accuracy: 0.8865
22200/25000 [=========================>....] - ETA: 1s - loss: 0.3504 - accuracy: 0.8866
22300/25000 [=========================>....] - ETA: 1s - loss: 0.3503 - accuracy: 0.8866
22400/25000 [=========================>....] - ETA: 1s - loss: 0.3504 - accuracy: 0.8865
22500/25000 [==========================>...] - ETA: 1s - loss: 0.3504 - accuracy: 0.8865
22600/25000 [==========================>...] - ETA: 1s - loss: 0.3504 - accuracy: 0.8865
22700/25000 [==========================>...] - ETA: 1s - loss: 0.3504 - accuracy: 0.8865
22800/25000 [==========================>...] - ETA: 1s - loss: 0.3504 - accuracy: 0.8865
22900/25000 [==========================>...] - ETA: 1s - loss: 0.3504 - accuracy: 0.8865
23000/25000 [==========================>...] - ETA: 1s - loss: 0.3505 - accuracy: 0.8865
23100/25000 [==========================>...] - ETA: 1s - loss: 0.3504 - accuracy: 0.8865
23200/25000 [==========================>...] - ETA: 1s - loss: 0.3504 - accuracy: 0.8865
23300/25000 [==========================>...] - ETA: 1s - loss: 0.3503 - accuracy: 0.8865
23400/25000 [===========================>..] - ETA: 1s - loss: 0.3503 - accuracy: 0.8865
23500/25000 [===========================>..] - ETA: 0s - loss: 0.3504 - accuracy: 0.8865
23600/25000 [===========================>..] - ETA: 0s - loss: 0.3504 - accuracy: 0.8865
23700/25000 [===========================>..] - ETA: 0s - loss: 0.3504 - accuracy: 0.8865
23800/25000 [===========================>..] - ETA: 0s - loss: 0.3504 - accuracy: 0.8865
23900/25000 [===========================>..] - ETA: 0s - loss: 0.3503 - accuracy: 0.8865
24000/25000 [===========================>..] - ETA: 0s - loss: 0.3504 - accuracy: 0.8865
24100/25000 [===========================>..] - ETA: 0s - loss: 0.3504 - accuracy: 0.8865
24200/25000 [============================>.] - ETA: 0s - loss: 0.3505 - accuracy: 0.8864
24300/25000 [============================>.] - ETA: 0s - loss: 0.3505 - accuracy: 0.8864
24400/25000 [============================>.] - ETA: 0s - loss: 0.3506 - accuracy: 0.8864
24500/25000 [============================>.] - ETA: 0s - loss: 0.3506 - accuracy: 0.8863
24600/25000 [============================>.] - ETA: 0s - loss: 0.3507 - accuracy: 0.8863
24700/25000 [============================>.] - ETA: 0s - loss: 0.3507 - accuracy: 0.8863
24800/25000 [============================>.] - ETA: 0s - loss: 0.3508 - accuracy: 0.8863
24900/25000 [============================>.] - ETA: 0s - loss: 0.3507 - accuracy: 0.8863
25000/25000 [==============================] - 20s 796us/step - loss: 0.3508 - accuracy: 0.8863 - val_loss: 0.3503 - val_accuracy: 0.8858
Epoch 9/10

  100/25000 [..............................] - ETA: 15s - loss: 0.3403 - accuracy: 0.8943
  200/25000 [..............................] - ETA: 15s - loss: 0.3460 - accuracy: 0.8900
  300/25000 [..............................] - ETA: 15s - loss: 0.3500 - accuracy: 0.8881
  400/25000 [..............................] - ETA: 15s - loss: 0.3503 - accuracy: 0.8868
  500/25000 [..............................] - ETA: 15s - loss: 0.3506 - accuracy: 0.8863
  600/25000 [..............................] - ETA: 15s - loss: 0.3490 - accuracy: 0.8871
  700/25000 [..............................] - ETA: 15s - loss: 0.3477 - accuracy: 0.8878
  800/25000 [..............................] - ETA: 15s - loss: 0.3482 - accuracy: 0.8873
  900/25000 [>.............................] - ETA: 15s - loss: 0.3493 - accuracy: 0.8868
 1000/25000 [>.............................] - ETA: 14s - loss: 0.3507 - accuracy: 0.8860
 1100/25000 [>.............................] - ETA: 15s - loss: 0.3516 - accuracy: 0.8852
 1200/25000 [>.............................] - ETA: 15s - loss: 0.3506 - accuracy: 0.8855
 1300/25000 [>.............................] - ETA: 14s - loss: 0.3519 - accuracy: 0.8849
 1400/25000 [>.............................] - ETA: 14s - loss: 0.3520 - accuracy: 0.8850
 1500/25000 [>.............................] - ETA: 14s - loss: 0.3521 - accuracy: 0.8850
 1600/25000 [>.............................] - ETA: 14s - loss: 0.3514 - accuracy: 0.8854
 1700/25000 [=>............................] - ETA: 14s - loss: 0.3521 - accuracy: 0.8852
 1800/25000 [=>............................] - ETA: 14s - loss: 0.3519 - accuracy: 0.8854
 1900/25000 [=>............................] - ETA: 14s - loss: 0.3515 - accuracy: 0.8855
 2000/25000 [=>............................] - ETA: 14s - loss: 0.3520 - accuracy: 0.8853
 2100/25000 [=>............................] - ETA: 14s - loss: 0.3520 - accuracy: 0.8852
 2200/25000 [=>............................] - ETA: 14s - loss: 0.3517 - accuracy: 0.8853
 2300/25000 [=>............................] - ETA: 14s - loss: 0.3526 - accuracy: 0.8848
 2400/25000 [=>............................] - ETA: 14s - loss: 0.3521 - accuracy: 0.8851
 2500/25000 [==>...........................] - ETA: 14s - loss: 0.3526 - accuracy: 0.8849
 2600/25000 [==>...........................] - ETA: 14s - loss: 0.3523 - accuracy: 0.8851
 2700/25000 [==>...........................] - ETA: 13s - loss: 0.3528 - accuracy: 0.8849
 2800/25000 [==>...........................] - ETA: 13s - loss: 0.3521 - accuracy: 0.8852
 2900/25000 [==>...........................] - ETA: 13s - loss: 0.3523 - accuracy: 0.8852
 3000/25000 [==>...........................] - ETA: 13s - loss: 0.3517 - accuracy: 0.8855
 3100/25000 [==>...........................] - ETA: 13s - loss: 0.3519 - accuracy: 0.8854
 3200/25000 [==>...........................] - ETA: 13s - loss: 0.3518 - accuracy: 0.8855
 3300/25000 [==>...........................] - ETA: 13s - loss: 0.3518 - accuracy: 0.8856
 3400/25000 [===>..........................] - ETA: 13s - loss: 0.3515 - accuracy: 0.8857
 3500/25000 [===>..........................] - ETA: 13s - loss: 0.3516 - accuracy: 0.8856
 3600/25000 [===>..........................] - ETA: 13s - loss: 0.3513 - accuracy: 0.8858
 3700/25000 [===>..........................] - ETA: 13s - loss: 0.3515 - accuracy: 0.8858
 3800/25000 [===>..........................] - ETA: 13s - loss: 0.3513 - accuracy: 0.8858
 3900/25000 [===>..........................] - ETA: 13s - loss: 0.3513 - accuracy: 0.8858
 4000/25000 [===>..........................] - ETA: 13s - loss: 0.3511 - accuracy: 0.8860
 4100/25000 [===>..........................] - ETA: 13s - loss: 0.3510 - accuracy: 0.8861
 4200/25000 [====>.........................] - ETA: 13s - loss: 0.3508 - accuracy: 0.8862
 4300/25000 [====>.........................] - ETA: 12s - loss: 0.3503 - accuracy: 0.8863
 4400/25000 [====>.........................] - ETA: 12s - loss: 0.3504 - accuracy: 0.8862
 4500/25000 [====>.........................] - ETA: 12s - loss: 0.3508 - accuracy: 0.8859
 4600/25000 [====>.........................] - ETA: 12s - loss: 0.3508 - accuracy: 0.8859
 4700/25000 [====>.........................] - ETA: 12s - loss: 0.3509 - accuracy: 0.8859
 4800/25000 [====>.........................] - ETA: 12s - loss: 0.3507 - accuracy: 0.8859
 4900/25000 [====>.........................] - ETA: 12s - loss: 0.3507 - accuracy: 0.8859
 5000/25000 [=====>........................] - ETA: 12s - loss: 0.3506 - accuracy: 0.8860
 5100/25000 [=====>........................] - ETA: 12s - loss: 0.3508 - accuracy: 0.8859
 5200/25000 [=====>........................] - ETA: 12s - loss: 0.3510 - accuracy: 0.8858
 5300/25000 [=====>........................] - ETA: 12s - loss: 0.3516 - accuracy: 0.8854
 5400/25000 [=====>........................] - ETA: 12s - loss: 0.3513 - accuracy: 0.8856
 5500/25000 [=====>........................] - ETA: 12s - loss: 0.3514 - accuracy: 0.8855
 5600/25000 [=====>........................] - ETA: 12s - loss: 0.3515 - accuracy: 0.8856
 5700/25000 [=====>........................] - ETA: 12s - loss: 0.3515 - accuracy: 0.8855
 5800/25000 [=====>........................] - ETA: 12s - loss: 0.3515 - accuracy: 0.8856
 5900/25000 [======>.......................] - ETA: 11s - loss: 0.3513 - accuracy: 0.8857
 6000/25000 [======>.......................] - ETA: 11s - loss: 0.3514 - accuracy: 0.8855
 6100/25000 [======>.......................] - ETA: 11s - loss: 0.3516 - accuracy: 0.8855
 6200/25000 [======>.......................] - ETA: 11s - loss: 0.3516 - accuracy: 0.8855
 6300/25000 [======>.......................] - ETA: 11s - loss: 0.3517 - accuracy: 0.8855
 6400/25000 [======>.......................] - ETA: 11s - loss: 0.3516 - accuracy: 0.8855
 6500/25000 [======>.......................] - ETA: 11s - loss: 0.3518 - accuracy: 0.8854
 6600/25000 [======>.......................] - ETA: 11s - loss: 0.3517 - accuracy: 0.8854
 6700/25000 [=======>......................] - ETA: 11s - loss: 0.3516 - accuracy: 0.8855
 6800/25000 [=======>......................] - ETA: 11s - loss: 0.3514 - accuracy: 0.8855
 6900/25000 [=======>......................] - ETA: 11s - loss: 0.3513 - accuracy: 0.8856
 7000/25000 [=======>......................] - ETA: 11s - loss: 0.3515 - accuracy: 0.8856
 7100/25000 [=======>......................] - ETA: 11s - loss: 0.3515 - accuracy: 0.8856
 7200/25000 [=======>......................] - ETA: 11s - loss: 0.3515 - accuracy: 0.8856
 7300/25000 [=======>......................] - ETA: 11s - loss: 0.3516 - accuracy: 0.8855
 7400/25000 [=======>......................] - ETA: 10s - loss: 0.3515 - accuracy: 0.8855
 7500/25000 [========>.....................] - ETA: 10s - loss: 0.3514 - accuracy: 0.8856
 7600/25000 [========>.....................] - ETA: 10s - loss: 0.3513 - accuracy: 0.8857
 7700/25000 [========>.....................] - ETA: 10s - loss: 0.3514 - accuracy: 0.8856
 7800/25000 [========>.....................] - ETA: 10s - loss: 0.3515 - accuracy: 0.8856
 7900/25000 [========>.....................] - ETA: 10s - loss: 0.3518 - accuracy: 0.8855
 8000/25000 [========>.....................] - ETA: 10s - loss: 0.3515 - accuracy: 0.8856
 8100/25000 [========>.....................] - ETA: 10s - loss: 0.3516 - accuracy: 0.8856
 8200/25000 [========>.....................] - ETA: 10s - loss: 0.3514 - accuracy: 0.8857
 8300/25000 [========>.....................] - ETA: 10s - loss: 0.3516 - accuracy: 0.8856
 8400/25000 [=========>....................] - ETA: 10s - loss: 0.3514 - accuracy: 0.8856
 8500/25000 [=========>....................] - ETA: 10s - loss: 0.3516 - accuracy: 0.8855
 8600/25000 [=========>....................] - ETA: 10s - loss: 0.3516 - accuracy: 0.8855
 8700/25000 [=========>....................] - ETA: 10s - loss: 0.3515 - accuracy: 0.8856
 8800/25000 [=========>....................] - ETA: 10s - loss: 0.3514 - accuracy: 0.8856
 8900/25000 [=========>....................] - ETA: 10s - loss: 0.3517 - accuracy: 0.8855
 9000/25000 [=========>....................] - ETA: 10s - loss: 0.3518 - accuracy: 0.8854
 9100/25000 [=========>....................] - ETA: 9s - loss: 0.3517 - accuracy: 0.8855 
 9200/25000 [==========>...................] - ETA: 9s - loss: 0.3518 - accuracy: 0.8855
 9300/25000 [==========>...................] - ETA: 9s - loss: 0.3518 - accuracy: 0.8855
 9400/25000 [==========>...................] - ETA: 9s - loss: 0.3520 - accuracy: 0.8854
 9500/25000 [==========>...................] - ETA: 9s - loss: 0.3519 - accuracy: 0.8855
 9600/25000 [==========>...................] - ETA: 9s - loss: 0.3520 - accuracy: 0.8854
 9700/25000 [==========>...................] - ETA: 9s - loss: 0.3519 - accuracy: 0.8854
 9800/25000 [==========>...................] - ETA: 9s - loss: 0.3515 - accuracy: 0.8856
 9900/25000 [==========>...................] - ETA: 9s - loss: 0.3516 - accuracy: 0.8856
10000/25000 [===========>..................] - ETA: 9s - loss: 0.3517 - accuracy: 0.8856
10100/25000 [===========>..................] - ETA: 9s - loss: 0.3518 - accuracy: 0.8855
10200/25000 [===========>..................] - ETA: 9s - loss: 0.3518 - accuracy: 0.8855
10300/25000 [===========>..................] - ETA: 9s - loss: 0.3517 - accuracy: 0.8856
10400/25000 [===========>..................] - ETA: 9s - loss: 0.3515 - accuracy: 0.8857
10500/25000 [===========>..................] - ETA: 9s - loss: 0.3516 - accuracy: 0.8856
10600/25000 [===========>..................] - ETA: 9s - loss: 0.3517 - accuracy: 0.8856
10700/25000 [===========>..................] - ETA: 8s - loss: 0.3518 - accuracy: 0.8856
10800/25000 [===========>..................] - ETA: 8s - loss: 0.3516 - accuracy: 0.8856
10900/25000 [============>.................] - ETA: 8s - loss: 0.3516 - accuracy: 0.8856
11000/25000 [============>.................] - ETA: 8s - loss: 0.3517 - accuracy: 0.8856
11100/25000 [============>.................] - ETA: 8s - loss: 0.3516 - accuracy: 0.8856
11200/25000 [============>.................] - ETA: 8s - loss: 0.3517 - accuracy: 0.8856
11300/25000 [============>.................] - ETA: 8s - loss: 0.3517 - accuracy: 0.8856
11400/25000 [============>.................] - ETA: 8s - loss: 0.3518 - accuracy: 0.8855
11500/25000 [============>.................] - ETA: 8s - loss: 0.3516 - accuracy: 0.8857
11600/25000 [============>.................] - ETA: 8s - loss: 0.3515 - accuracy: 0.8857
11700/25000 [=============>................] - ETA: 8s - loss: 0.3516 - accuracy: 0.8857
11800/25000 [=============>................] - ETA: 8s - loss: 0.3515 - accuracy: 0.8858
11900/25000 [=============>................] - ETA: 8s - loss: 0.3515 - accuracy: 0.8858
12000/25000 [=============>................] - ETA: 8s - loss: 0.3514 - accuracy: 0.8858
12100/25000 [=============>................] - ETA: 8s - loss: 0.3514 - accuracy: 0.8858
12200/25000 [=============>................] - ETA: 8s - loss: 0.3514 - accuracy: 0.8858
12300/25000 [=============>................] - ETA: 7s - loss: 0.3514 - accuracy: 0.8858
12400/25000 [=============>................] - ETA: 7s - loss: 0.3513 - accuracy: 0.8859
12500/25000 [==============>...............] - ETA: 7s - loss: 0.3512 - accuracy: 0.8859
12600/25000 [==============>...............] - ETA: 7s - loss: 0.3512 - accuracy: 0.8859
12700/25000 [==============>...............] - ETA: 7s - loss: 0.3512 - accuracy: 0.8859
12800/25000 [==============>...............] - ETA: 7s - loss: 0.3512 - accuracy: 0.8859
12900/25000 [==============>...............] - ETA: 7s - loss: 0.3511 - accuracy: 0.8859
13000/25000 [==============>...............] - ETA: 7s - loss: 0.3509 - accuracy: 0.8860
13100/25000 [==============>...............] - ETA: 7s - loss: 0.3508 - accuracy: 0.8860
13200/25000 [==============>...............] - ETA: 7s - loss: 0.3508 - accuracy: 0.8860
13300/25000 [==============>...............] - ETA: 7s - loss: 0.3508 - accuracy: 0.8860
13400/25000 [===============>..............] - ETA: 7s - loss: 0.3506 - accuracy: 0.8861
13500/25000 [===============>..............] - ETA: 7s - loss: 0.3506 - accuracy: 0.8861
13600/25000 [===============>..............] - ETA: 7s - loss: 0.3506 - accuracy: 0.8861
13700/25000 [===============>..............] - ETA: 7s - loss: 0.3505 - accuracy: 0.8862
13800/25000 [===============>..............] - ETA: 7s - loss: 0.3505 - accuracy: 0.8861
13900/25000 [===============>..............] - ETA: 6s - loss: 0.3507 - accuracy: 0.8861
14000/25000 [===============>..............] - ETA: 6s - loss: 0.3505 - accuracy: 0.8862
14100/25000 [===============>..............] - ETA: 6s - loss: 0.3503 - accuracy: 0.8862
14200/25000 [================>.............] - ETA: 6s - loss: 0.3503 - accuracy: 0.8862
14300/25000 [================>.............] - ETA: 6s - loss: 0.3504 - accuracy: 0.8862
14400/25000 [================>.............] - ETA: 6s - loss: 0.3503 - accuracy: 0.8862
14500/25000 [================>.............] - ETA: 6s - loss: 0.3503 - accuracy: 0.8862
14600/25000 [================>.............] - ETA: 6s - loss: 0.3504 - accuracy: 0.8861
14700/25000 [================>.............] - ETA: 6s - loss: 0.3503 - accuracy: 0.8861
14800/25000 [================>.............] - ETA: 6s - loss: 0.3504 - accuracy: 0.8861
14900/25000 [================>.............] - ETA: 6s - loss: 0.3503 - accuracy: 0.8861
15000/25000 [=================>............] - ETA: 6s - loss: 0.3503 - accuracy: 0.8862
15100/25000 [=================>............] - ETA: 6s - loss: 0.3504 - accuracy: 0.8861
15200/25000 [=================>............] - ETA: 6s - loss: 0.3503 - accuracy: 0.8862
15300/25000 [=================>............] - ETA: 6s - loss: 0.3503 - accuracy: 0.8862
15400/25000 [=================>............] - ETA: 6s - loss: 0.3503 - accuracy: 0.8862
15500/25000 [=================>............] - ETA: 5s - loss: 0.3502 - accuracy: 0.8862
15600/25000 [=================>............] - ETA: 5s - loss: 0.3502 - accuracy: 0.8862
15700/25000 [=================>............] - ETA: 5s - loss: 0.3502 - accuracy: 0.8862
15800/25000 [=================>............] - ETA: 5s - loss: 0.3503 - accuracy: 0.8862
15900/25000 [==================>...........] - ETA: 5s - loss: 0.3502 - accuracy: 0.8862
16000/25000 [==================>...........] - ETA: 5s - loss: 0.3502 - accuracy: 0.8862
16100/25000 [==================>...........] - ETA: 5s - loss: 0.3502 - accuracy: 0.8862
16200/25000 [==================>...........] - ETA: 5s - loss: 0.3502 - accuracy: 0.8862
16300/25000 [==================>...........] - ETA: 5s - loss: 0.3502 - accuracy: 0.8862
16400/25000 [==================>...........] - ETA: 5s - loss: 0.3502 - accuracy: 0.8862
16500/25000 [==================>...........] - ETA: 5s - loss: 0.3501 - accuracy: 0.8862
16600/25000 [==================>...........] - ETA: 5s - loss: 0.3501 - accuracy: 0.8862
16700/25000 [===================>..........] - ETA: 5s - loss: 0.3501 - accuracy: 0.8862
16800/25000 [===================>..........] - ETA: 5s - loss: 0.3501 - accuracy: 0.8862
16900/25000 [===================>..........] - ETA: 5s - loss: 0.3500 - accuracy: 0.8863
17000/25000 [===================>..........] - ETA: 5s - loss: 0.3501 - accuracy: 0.8862
17100/25000 [===================>..........] - ETA: 4s - loss: 0.3500 - accuracy: 0.8862
17200/25000 [===================>..........] - ETA: 4s - loss: 0.3500 - accuracy: 0.8863
17300/25000 [===================>..........] - ETA: 4s - loss: 0.3500 - accuracy: 0.8863
17400/25000 [===================>..........] - ETA: 4s - loss: 0.3500 - accuracy: 0.8863
17500/25000 [====================>.........] - ETA: 4s - loss: 0.3499 - accuracy: 0.8863
17600/25000 [====================>.........] - ETA: 4s - loss: 0.3499 - accuracy: 0.8863
17700/25000 [====================>.........] - ETA: 4s - loss: 0.3498 - accuracy: 0.8863
17800/25000 [====================>.........] - ETA: 4s - loss: 0.3499 - accuracy: 0.8863
17900/25000 [====================>.........] - ETA: 4s - loss: 0.3499 - accuracy: 0.8863
18000/25000 [====================>.........] - ETA: 4s - loss: 0.3498 - accuracy: 0.8863
18100/25000 [====================>.........] - ETA: 4s - loss: 0.3497 - accuracy: 0.8864
18200/25000 [====================>.........] - ETA: 4s - loss: 0.3497 - accuracy: 0.8864
18300/25000 [====================>.........] - ETA: 4s - loss: 0.3496 - accuracy: 0.8864
18400/25000 [=====================>........] - ETA: 4s - loss: 0.3497 - accuracy: 0.8864
18500/25000 [=====================>........] - ETA: 4s - loss: 0.3496 - accuracy: 0.8864
18600/25000 [=====================>........] - ETA: 4s - loss: 0.3496 - accuracy: 0.8864
18700/25000 [=====================>........] - ETA: 3s - loss: 0.3497 - accuracy: 0.8864
18800/25000 [=====================>........] - ETA: 3s - loss: 0.3497 - accuracy: 0.8864
18900/25000 [=====================>........] - ETA: 3s - loss: 0.3497 - accuracy: 0.8864
19000/25000 [=====================>........] - ETA: 3s - loss: 0.3497 - accuracy: 0.8865
19100/25000 [=====================>........] - ETA: 3s - loss: 0.3497 - accuracy: 0.8865
19200/25000 [======================>.......] - ETA: 3s - loss: 0.3497 - accuracy: 0.8864
19300/25000 [======================>.......] - ETA: 3s - loss: 0.3497 - accuracy: 0.8865
19400/25000 [======================>.......] - ETA: 3s - loss: 0.3497 - accuracy: 0.8864
19500/25000 [======================>.......] - ETA: 3s - loss: 0.3497 - accuracy: 0.8864
19600/25000 [======================>.......] - ETA: 3s - loss: 0.3497 - accuracy: 0.8864
19700/25000 [======================>.......] - ETA: 3s - loss: 0.3497 - accuracy: 0.8864
19800/25000 [======================>.......] - ETA: 3s - loss: 0.3497 - accuracy: 0.8864
19900/25000 [======================>.......] - ETA: 3s - loss: 0.3496 - accuracy: 0.8865
20000/25000 [=======================>......] - ETA: 3s - loss: 0.3497 - accuracy: 0.8864
20100/25000 [=======================>......] - ETA: 3s - loss: 0.3497 - accuracy: 0.8864
20200/25000 [=======================>......] - ETA: 3s - loss: 0.3497 - accuracy: 0.8864
20300/25000 [=======================>......] - ETA: 2s - loss: 0.3497 - accuracy: 0.8864
20400/25000 [=======================>......] - ETA: 2s - loss: 0.3496 - accuracy: 0.8865
20500/25000 [=======================>......] - ETA: 2s - loss: 0.3496 - accuracy: 0.8865
20600/25000 [=======================>......] - ETA: 2s - loss: 0.3495 - accuracy: 0.8865
20700/25000 [=======================>......] - ETA: 2s - loss: 0.3495 - accuracy: 0.8865
20800/25000 [=======================>......] - ETA: 2s - loss: 0.3495 - accuracy: 0.8865
20900/25000 [========================>.....] - ETA: 2s - loss: 0.3495 - accuracy: 0.8865
21000/25000 [========================>.....] - ETA: 2s - loss: 0.3495 - accuracy: 0.8865
21100/25000 [========================>.....] - ETA: 2s - loss: 0.3495 - accuracy: 0.8865
21200/25000 [========================>.....] - ETA: 2s - loss: 0.3495 - accuracy: 0.8865
21300/25000 [========================>.....] - ETA: 2s - loss: 0.3495 - accuracy: 0.8865
21400/25000 [========================>.....] - ETA: 2s - loss: 0.3496 - accuracy: 0.8865
21500/25000 [========================>.....] - ETA: 2s - loss: 0.3496 - accuracy: 0.8865
21600/25000 [========================>.....] - ETA: 2s - loss: 0.3496 - accuracy: 0.8865
21700/25000 [=========================>....] - ETA: 2s - loss: 0.3496 - accuracy: 0.8865
21800/25000 [=========================>....] - ETA: 2s - loss: 0.3496 - accuracy: 0.8865
21900/25000 [=========================>....] - ETA: 1s - loss: 0.3496 - accuracy: 0.8865
22000/25000 [=========================>....] - ETA: 1s - loss: 0.3496 - accuracy: 0.8865
22100/25000 [=========================>....] - ETA: 1s - loss: 0.3497 - accuracy: 0.8865
22200/25000 [=========================>....] - ETA: 1s - loss: 0.3497 - accuracy: 0.8864
22300/25000 [=========================>....] - ETA: 1s - loss: 0.3497 - accuracy: 0.8864
22400/25000 [=========================>....] - ETA: 1s - loss: 0.3498 - accuracy: 0.8864
22500/25000 [==========================>...] - ETA: 1s - loss: 0.3498 - accuracy: 0.8864
22600/25000 [==========================>...] - ETA: 1s - loss: 0.3497 - accuracy: 0.8864
22700/25000 [==========================>...] - ETA: 1s - loss: 0.3498 - accuracy: 0.8864
22800/25000 [==========================>...] - ETA: 1s - loss: 0.3498 - accuracy: 0.8864
22900/25000 [==========================>...] - ETA: 1s - loss: 0.3498 - accuracy: 0.8864
23000/25000 [==========================>...] - ETA: 1s - loss: 0.3499 - accuracy: 0.8864
23100/25000 [==========================>...] - ETA: 1s - loss: 0.3499 - accuracy: 0.8864
23200/25000 [==========================>...] - ETA: 1s - loss: 0.3500 - accuracy: 0.8863
23300/25000 [==========================>...] - ETA: 1s - loss: 0.3500 - accuracy: 0.8863
23400/25000 [===========================>..] - ETA: 1s - loss: 0.3500 - accuracy: 0.8863
23500/25000 [===========================>..] - ETA: 0s - loss: 0.3500 - accuracy: 0.8863
23600/25000 [===========================>..] - ETA: 0s - loss: 0.3500 - accuracy: 0.8863
23700/25000 [===========================>..] - ETA: 0s - loss: 0.3501 - accuracy: 0.8863
23800/25000 [===========================>..] - ETA: 0s - loss: 0.3500 - accuracy: 0.8863
23900/25000 [===========================>..] - ETA: 0s - loss: 0.3500 - accuracy: 0.8863
24000/25000 [===========================>..] - ETA: 0s - loss: 0.3500 - accuracy: 0.8863
24100/25000 [===========================>..] - ETA: 0s - loss: 0.3500 - accuracy: 0.8863
24200/25000 [============================>.] - ETA: 0s - loss: 0.3501 - accuracy: 0.8862
24300/25000 [============================>.] - ETA: 0s - loss: 0.3502 - accuracy: 0.8862
24400/25000 [============================>.] - ETA: 0s - loss: 0.3501 - accuracy: 0.8862
24500/25000 [============================>.] - ETA: 0s - loss: 0.3501 - accuracy: 0.8863
24600/25000 [============================>.] - ETA: 0s - loss: 0.3501 - accuracy: 0.8863
24700/25000 [============================>.] - ETA: 0s - loss: 0.3500 - accuracy: 0.8863
24800/25000 [============================>.] - ETA: 0s - loss: 0.3501 - accuracy: 0.8863
24900/25000 [============================>.] - ETA: 0s - loss: 0.3501 - accuracy: 0.8863
25000/25000 [==============================] - 20s 793us/step - loss: 0.3501 - accuracy: 0.8863 - val_loss: 0.3499 - val_accuracy: 0.8858
Epoch 10/10

  100/25000 [..............................] - ETA: 16s - loss: 0.3486 - accuracy: 0.8857
  200/25000 [..............................] - ETA: 16s - loss: 0.3530 - accuracy: 0.8843
  300/25000 [..............................] - ETA: 15s - loss: 0.3550 - accuracy: 0.8833
  400/25000 [..............................] - ETA: 15s - loss: 0.3540 - accuracy: 0.8843
  500/25000 [..............................] - ETA: 15s - loss: 0.3550 - accuracy: 0.8843
  600/25000 [..............................] - ETA: 15s - loss: 0.3523 - accuracy: 0.8855
  700/25000 [..............................] - ETA: 15s - loss: 0.3528 - accuracy: 0.8847
  800/25000 [..............................] - ETA: 15s - loss: 0.3523 - accuracy: 0.8848
  900/25000 [>.............................] - ETA: 15s - loss: 0.3505 - accuracy: 0.8857
 1000/25000 [>.............................] - ETA: 15s - loss: 0.3517 - accuracy: 0.8854
 1100/25000 [>.............................] - ETA: 15s - loss: 0.3522 - accuracy: 0.8852
 1200/25000 [>.............................] - ETA: 15s - loss: 0.3520 - accuracy: 0.8854
 1300/25000 [>.............................] - ETA: 14s - loss: 0.3530 - accuracy: 0.8849
 1400/25000 [>.............................] - ETA: 14s - loss: 0.3534 - accuracy: 0.8848
 1500/25000 [>.............................] - ETA: 14s - loss: 0.3535 - accuracy: 0.8847
 1600/25000 [>.............................] - ETA: 14s - loss: 0.3535 - accuracy: 0.8846
 1700/25000 [=>............................] - ETA: 14s - loss: 0.3533 - accuracy: 0.8847
 1800/25000 [=>............................] - ETA: 14s - loss: 0.3539 - accuracy: 0.8845
 1900/25000 [=>............................] - ETA: 14s - loss: 0.3532 - accuracy: 0.8847
 2000/25000 [=>............................] - ETA: 14s - loss: 0.3531 - accuracy: 0.8846
 2100/25000 [=>............................] - ETA: 14s - loss: 0.3526 - accuracy: 0.8846
 2200/25000 [=>............................] - ETA: 14s - loss: 0.3524 - accuracy: 0.8847
 2300/25000 [=>............................] - ETA: 14s - loss: 0.3530 - accuracy: 0.8843
 2400/25000 [=>............................] - ETA: 14s - loss: 0.3525 - accuracy: 0.8846
 2500/25000 [==>...........................] - ETA: 14s - loss: 0.3527 - accuracy: 0.8845
 2600/25000 [==>...........................] - ETA: 14s - loss: 0.3525 - accuracy: 0.8845
 2700/25000 [==>...........................] - ETA: 13s - loss: 0.3529 - accuracy: 0.8843
 2800/25000 [==>...........................] - ETA: 13s - loss: 0.3526 - accuracy: 0.8844
 2900/25000 [==>...........................] - ETA: 13s - loss: 0.3524 - accuracy: 0.8845
 3000/25000 [==>...........................] - ETA: 13s - loss: 0.3523 - accuracy: 0.8847
 3100/25000 [==>...........................] - ETA: 13s - loss: 0.3522 - accuracy: 0.8849
 3200/25000 [==>...........................] - ETA: 13s - loss: 0.3520 - accuracy: 0.8850
 3300/25000 [==>...........................] - ETA: 13s - loss: 0.3523 - accuracy: 0.8848
 3400/25000 [===>..........................] - ETA: 13s - loss: 0.3528 - accuracy: 0.8847
 3500/25000 [===>..........................] - ETA: 13s - loss: 0.3525 - accuracy: 0.8848
 3600/25000 [===>..........................] - ETA: 13s - loss: 0.3527 - accuracy: 0.8848
 3700/25000 [===>..........................] - ETA: 13s - loss: 0.3522 - accuracy: 0.8850
 3800/25000 [===>..........................] - ETA: 13s - loss: 0.3529 - accuracy: 0.8847
 3900/25000 [===>..........................] - ETA: 13s - loss: 0.3523 - accuracy: 0.8849
 4000/25000 [===>..........................] - ETA: 13s - loss: 0.3524 - accuracy: 0.8849
 4100/25000 [===>..........................] - ETA: 13s - loss: 0.3521 - accuracy: 0.8851
 4200/25000 [====>.........................] - ETA: 13s - loss: 0.3520 - accuracy: 0.8851
 4300/25000 [====>.........................] - ETA: 12s - loss: 0.3517 - accuracy: 0.8853
 4400/25000 [====>.........................] - ETA: 12s - loss: 0.3517 - accuracy: 0.8853
 4500/25000 [====>.........................] - ETA: 12s - loss: 0.3522 - accuracy: 0.8850
 4600/25000 [====>.........................] - ETA: 12s - loss: 0.3521 - accuracy: 0.8851
 4700/25000 [====>.........................] - ETA: 12s - loss: 0.3526 - accuracy: 0.8849
 4800/25000 [====>.........................] - ETA: 12s - loss: 0.3523 - accuracy: 0.8850
 4900/25000 [====>.........................] - ETA: 12s - loss: 0.3518 - accuracy: 0.8852
 5000/25000 [=====>........................] - ETA: 12s - loss: 0.3514 - accuracy: 0.8854
 5100/25000 [=====>........................] - ETA: 12s - loss: 0.3518 - accuracy: 0.8853
 5200/25000 [=====>........................] - ETA: 12s - loss: 0.3515 - accuracy: 0.8854
 5300/25000 [=====>........................] - ETA: 12s - loss: 0.3517 - accuracy: 0.8852
 5400/25000 [=====>........................] - ETA: 12s - loss: 0.3519 - accuracy: 0.8851
 5500/25000 [=====>........................] - ETA: 12s - loss: 0.3517 - accuracy: 0.8852
 5600/25000 [=====>........................] - ETA: 12s - loss: 0.3519 - accuracy: 0.8852
 5700/25000 [=====>........................] - ETA: 12s - loss: 0.3521 - accuracy: 0.8851
 5800/25000 [=====>........................] - ETA: 12s - loss: 0.3517 - accuracy: 0.8853
 5900/25000 [======>.......................] - ETA: 11s - loss: 0.3516 - accuracy: 0.8853
 6000/25000 [======>.......................] - ETA: 11s - loss: 0.3516 - accuracy: 0.8854
 6100/25000 [======>.......................] - ETA: 11s - loss: 0.3515 - accuracy: 0.8854
 6200/25000 [======>.......................] - ETA: 11s - loss: 0.3513 - accuracy: 0.8855
 6300/25000 [======>.......................] - ETA: 11s - loss: 0.3514 - accuracy: 0.8854
 6400/25000 [======>.......................] - ETA: 11s - loss: 0.3511 - accuracy: 0.8856
 6500/25000 [======>.......................] - ETA: 11s - loss: 0.3512 - accuracy: 0.8855
 6600/25000 [======>.......................] - ETA: 11s - loss: 0.3511 - accuracy: 0.8856
 6700/25000 [=======>......................] - ETA: 11s - loss: 0.3510 - accuracy: 0.8857
 6800/25000 [=======>......................] - ETA: 11s - loss: 0.3511 - accuracy: 0.8856
 6900/25000 [=======>......................] - ETA: 11s - loss: 0.3512 - accuracy: 0.8855
 7000/25000 [=======>......................] - ETA: 11s - loss: 0.3511 - accuracy: 0.8856
 7100/25000 [=======>......................] - ETA: 11s - loss: 0.3510 - accuracy: 0.8856
 7200/25000 [=======>......................] - ETA: 11s - loss: 0.3515 - accuracy: 0.8854
 7300/25000 [=======>......................] - ETA: 11s - loss: 0.3515 - accuracy: 0.8854
 7400/25000 [=======>......................] - ETA: 11s - loss: 0.3514 - accuracy: 0.8855
 7500/25000 [========>.....................] - ETA: 10s - loss: 0.3514 - accuracy: 0.8854
 7600/25000 [========>.....................] - ETA: 10s - loss: 0.3513 - accuracy: 0.8855
 7700/25000 [========>.....................] - ETA: 10s - loss: 0.3511 - accuracy: 0.8856
 7800/25000 [========>.....................] - ETA: 10s - loss: 0.3512 - accuracy: 0.8856
 7900/25000 [========>.....................] - ETA: 10s - loss: 0.3512 - accuracy: 0.8856
 8000/25000 [========>.....................] - ETA: 10s - loss: 0.3511 - accuracy: 0.8856
 8100/25000 [========>.....................] - ETA: 10s - loss: 0.3509 - accuracy: 0.8857
 8200/25000 [========>.....................] - ETA: 10s - loss: 0.3509 - accuracy: 0.8857
 8300/25000 [========>.....................] - ETA: 10s - loss: 0.3513 - accuracy: 0.8855
 8400/25000 [=========>....................] - ETA: 10s - loss: 0.3512 - accuracy: 0.8855
 8500/25000 [=========>....................] - ETA: 10s - loss: 0.3513 - accuracy: 0.8855
 8600/25000 [=========>....................] - ETA: 10s - loss: 0.3513 - accuracy: 0.8855
 8700/25000 [=========>....................] - ETA: 10s - loss: 0.3512 - accuracy: 0.8855
 8800/25000 [=========>....................] - ETA: 10s - loss: 0.3513 - accuracy: 0.8855
 8900/25000 [=========>....................] - ETA: 10s - loss: 0.3511 - accuracy: 0.8856
 9000/25000 [=========>....................] - ETA: 10s - loss: 0.3511 - accuracy: 0.8855
 9100/25000 [=========>....................] - ETA: 9s - loss: 0.3511 - accuracy: 0.8856 
 9200/25000 [==========>...................] - ETA: 9s - loss: 0.3512 - accuracy: 0.8855
 9300/25000 [==========>...................] - ETA: 9s - loss: 0.3512 - accuracy: 0.8855
 9400/25000 [==========>...................] - ETA: 9s - loss: 0.3513 - accuracy: 0.8855
 9500/25000 [==========>...................] - ETA: 9s - loss: 0.3513 - accuracy: 0.8855
 9600/25000 [==========>...................] - ETA: 9s - loss: 0.3511 - accuracy: 0.8856
 9700/25000 [==========>...................] - ETA: 9s - loss: 0.3511 - accuracy: 0.8856
 9800/25000 [==========>...................] - ETA: 9s - loss: 0.3511 - accuracy: 0.8856
 9900/25000 [==========>...................] - ETA: 9s - loss: 0.3511 - accuracy: 0.8856
10000/25000 [===========>..................] - ETA: 9s - loss: 0.3513 - accuracy: 0.8855
10100/25000 [===========>..................] - ETA: 9s - loss: 0.3512 - accuracy: 0.8856
10200/25000 [===========>..................] - ETA: 9s - loss: 0.3513 - accuracy: 0.8855
10300/25000 [===========>..................] - ETA: 9s - loss: 0.3512 - accuracy: 0.8856
10400/25000 [===========>..................] - ETA: 9s - loss: 0.3513 - accuracy: 0.8856
10500/25000 [===========>..................] - ETA: 9s - loss: 0.3512 - accuracy: 0.8856
10600/25000 [===========>..................] - ETA: 9s - loss: 0.3513 - accuracy: 0.8856
10700/25000 [===========>..................] - ETA: 8s - loss: 0.3514 - accuracy: 0.8856
10800/25000 [===========>..................] - ETA: 8s - loss: 0.3513 - accuracy: 0.8856
10900/25000 [============>.................] - ETA: 8s - loss: 0.3512 - accuracy: 0.8856
11000/25000 [============>.................] - ETA: 8s - loss: 0.3512 - accuracy: 0.8857
11100/25000 [============>.................] - ETA: 8s - loss: 0.3511 - accuracy: 0.8857
11200/25000 [============>.................] - ETA: 8s - loss: 0.3511 - accuracy: 0.8857
11300/25000 [============>.................] - ETA: 8s - loss: 0.3512 - accuracy: 0.8857
11400/25000 [============>.................] - ETA: 8s - loss: 0.3513 - accuracy: 0.8856
11500/25000 [============>.................] - ETA: 8s - loss: 0.3513 - accuracy: 0.8856
11600/25000 [============>.................] - ETA: 8s - loss: 0.3513 - accuracy: 0.8857
11700/25000 [=============>................] - ETA: 8s - loss: 0.3513 - accuracy: 0.8857
11800/25000 [=============>................] - ETA: 8s - loss: 0.3513 - accuracy: 0.8857
11900/25000 [=============>................] - ETA: 8s - loss: 0.3512 - accuracy: 0.8857
12000/25000 [=============>................] - ETA: 8s - loss: 0.3511 - accuracy: 0.8857
12100/25000 [=============>................] - ETA: 8s - loss: 0.3510 - accuracy: 0.8858
12200/25000 [=============>................] - ETA: 8s - loss: 0.3512 - accuracy: 0.8857
12300/25000 [=============>................] - ETA: 8s - loss: 0.3512 - accuracy: 0.8857
12400/25000 [=============>................] - ETA: 7s - loss: 0.3512 - accuracy: 0.8857
12500/25000 [==============>...............] - ETA: 7s - loss: 0.3513 - accuracy: 0.8857
12600/25000 [==============>...............] - ETA: 7s - loss: 0.3510 - accuracy: 0.8858
12700/25000 [==============>...............] - ETA: 7s - loss: 0.3508 - accuracy: 0.8859
12800/25000 [==============>...............] - ETA: 7s - loss: 0.3508 - accuracy: 0.8859
12900/25000 [==============>...............] - ETA: 7s - loss: 0.3506 - accuracy: 0.8859
13000/25000 [==============>...............] - ETA: 7s - loss: 0.3507 - accuracy: 0.8859
13100/25000 [==============>...............] - ETA: 7s - loss: 0.3506 - accuracy: 0.8860
13200/25000 [==============>...............] - ETA: 7s - loss: 0.3505 - accuracy: 0.8860
13300/25000 [==============>...............] - ETA: 7s - loss: 0.3506 - accuracy: 0.8860
13400/25000 [===============>..............] - ETA: 7s - loss: 0.3507 - accuracy: 0.8859
13500/25000 [===============>..............] - ETA: 7s - loss: 0.3507 - accuracy: 0.8860
13600/25000 [===============>..............] - ETA: 7s - loss: 0.3505 - accuracy: 0.8861
13700/25000 [===============>..............] - ETA: 7s - loss: 0.3506 - accuracy: 0.8860
13800/25000 [===============>..............] - ETA: 7s - loss: 0.3506 - accuracy: 0.8860
13900/25000 [===============>..............] - ETA: 7s - loss: 0.3505 - accuracy: 0.8861
14000/25000 [===============>..............] - ETA: 7s - loss: 0.3505 - accuracy: 0.8860
14100/25000 [===============>..............] - ETA: 6s - loss: 0.3505 - accuracy: 0.8860
14200/25000 [================>.............] - ETA: 6s - loss: 0.3505 - accuracy: 0.8860
14300/25000 [================>.............] - ETA: 6s - loss: 0.3505 - accuracy: 0.8860
14400/25000 [================>.............] - ETA: 6s - loss: 0.3506 - accuracy: 0.8860
14500/25000 [================>.............] - ETA: 6s - loss: 0.3506 - accuracy: 0.8860
14600/25000 [================>.............] - ETA: 6s - loss: 0.3505 - accuracy: 0.8860
14700/25000 [================>.............] - ETA: 6s - loss: 0.3505 - accuracy: 0.8860
14800/25000 [================>.............] - ETA: 6s - loss: 0.3505 - accuracy: 0.8860
14900/25000 [================>.............] - ETA: 6s - loss: 0.3504 - accuracy: 0.8860
15000/25000 [=================>............] - ETA: 6s - loss: 0.3504 - accuracy: 0.8860
15100/25000 [=================>............] - ETA: 6s - loss: 0.3503 - accuracy: 0.8861
15200/25000 [=================>............] - ETA: 6s - loss: 0.3503 - accuracy: 0.8861
15300/25000 [=================>............] - ETA: 6s - loss: 0.3502 - accuracy: 0.8861
15400/25000 [=================>............] - ETA: 6s - loss: 0.3502 - accuracy: 0.8862
15500/25000 [=================>............] - ETA: 6s - loss: 0.3503 - accuracy: 0.8861
15600/25000 [=================>............] - ETA: 6s - loss: 0.3503 - accuracy: 0.8861
15700/25000 [=================>............] - ETA: 6s - loss: 0.3502 - accuracy: 0.8861
15800/25000 [=================>............] - ETA: 5s - loss: 0.3502 - accuracy: 0.8861
15900/25000 [==================>...........] - ETA: 5s - loss: 0.3503 - accuracy: 0.8861
16000/25000 [==================>...........] - ETA: 5s - loss: 0.3502 - accuracy: 0.8861
16100/25000 [==================>...........] - ETA: 5s - loss: 0.3502 - accuracy: 0.8861
16200/25000 [==================>...........] - ETA: 5s - loss: 0.3502 - accuracy: 0.8861
16300/25000 [==================>...........] - ETA: 5s - loss: 0.3501 - accuracy: 0.8862
16400/25000 [==================>...........] - ETA: 5s - loss: 0.3500 - accuracy: 0.8862
16500/25000 [==================>...........] - ETA: 5s - loss: 0.3500 - accuracy: 0.8862
16600/25000 [==================>...........] - ETA: 5s - loss: 0.3499 - accuracy: 0.8862
16700/25000 [===================>..........] - ETA: 5s - loss: 0.3499 - accuracy: 0.8862
16800/25000 [===================>..........] - ETA: 5s - loss: 0.3500 - accuracy: 0.8862
16900/25000 [===================>..........] - ETA: 5s - loss: 0.3500 - accuracy: 0.8862
17000/25000 [===================>..........] - ETA: 5s - loss: 0.3498 - accuracy: 0.8863
17100/25000 [===================>..........] - ETA: 5s - loss: 0.3498 - accuracy: 0.8863
17200/25000 [===================>..........] - ETA: 5s - loss: 0.3498 - accuracy: 0.8863
17300/25000 [===================>..........] - ETA: 5s - loss: 0.3499 - accuracy: 0.8862
17400/25000 [===================>..........] - ETA: 4s - loss: 0.3499 - accuracy: 0.8862
17500/25000 [====================>.........] - ETA: 4s - loss: 0.3498 - accuracy: 0.8863
17600/25000 [====================>.........] - ETA: 4s - loss: 0.3498 - accuracy: 0.8863
17700/25000 [====================>.........] - ETA: 4s - loss: 0.3498 - accuracy: 0.8863
17800/25000 [====================>.........] - ETA: 4s - loss: 0.3498 - accuracy: 0.8863
17900/25000 [====================>.........] - ETA: 4s - loss: 0.3498 - accuracy: 0.8862
18000/25000 [====================>.........] - ETA: 4s - loss: 0.3498 - accuracy: 0.8863
18100/25000 [====================>.........] - ETA: 4s - loss: 0.3498 - accuracy: 0.8863
18200/25000 [====================>.........] - ETA: 4s - loss: 0.3498 - accuracy: 0.8863
18300/25000 [====================>.........] - ETA: 4s - loss: 0.3498 - accuracy: 0.8863
18400/25000 [=====================>........] - ETA: 4s - loss: 0.3498 - accuracy: 0.8862
18500/25000 [=====================>........] - ETA: 4s - loss: 0.3498 - accuracy: 0.8862
18600/25000 [=====================>........] - ETA: 4s - loss: 0.3498 - accuracy: 0.8862
18700/25000 [=====================>........] - ETA: 4s - loss: 0.3498 - accuracy: 0.8863
18800/25000 [=====================>........] - ETA: 4s - loss: 0.3498 - accuracy: 0.8863
18900/25000 [=====================>........] - ETA: 3s - loss: 0.3498 - accuracy: 0.8863
19000/25000 [=====================>........] - ETA: 3s - loss: 0.3498 - accuracy: 0.8862
19100/25000 [=====================>........] - ETA: 3s - loss: 0.3498 - accuracy: 0.8863
19200/25000 [======================>.......] - ETA: 3s - loss: 0.3497 - accuracy: 0.8863
19300/25000 [======================>.......] - ETA: 3s - loss: 0.3497 - accuracy: 0.8863
19400/25000 [======================>.......] - ETA: 3s - loss: 0.3497 - accuracy: 0.8863
19500/25000 [======================>.......] - ETA: 3s - loss: 0.3497 - accuracy: 0.8863
19600/25000 [======================>.......] - ETA: 3s - loss: 0.3497 - accuracy: 0.8863
19700/25000 [======================>.......] - ETA: 3s - loss: 0.3497 - accuracy: 0.8863
19800/25000 [======================>.......] - ETA: 3s - loss: 0.3496 - accuracy: 0.8864
19900/25000 [======================>.......] - ETA: 3s - loss: 0.3497 - accuracy: 0.8863
20000/25000 [=======================>......] - ETA: 3s - loss: 0.3496 - accuracy: 0.8864
20100/25000 [=======================>......] - ETA: 3s - loss: 0.3495 - accuracy: 0.8864
20200/25000 [=======================>......] - ETA: 3s - loss: 0.3495 - accuracy: 0.8864
20300/25000 [=======================>......] - ETA: 3s - loss: 0.3496 - accuracy: 0.8864
20400/25000 [=======================>......] - ETA: 2s - loss: 0.3496 - accuracy: 0.8864
20500/25000 [=======================>......] - ETA: 2s - loss: 0.3497 - accuracy: 0.8863
20600/25000 [=======================>......] - ETA: 2s - loss: 0.3496 - accuracy: 0.8864
20700/25000 [=======================>......] - ETA: 2s - loss: 0.3497 - accuracy: 0.8863
20800/25000 [=======================>......] - ETA: 2s - loss: 0.3497 - accuracy: 0.8863
20900/25000 [========================>.....] - ETA: 2s - loss: 0.3495 - accuracy: 0.8864
21000/25000 [========================>.....] - ETA: 2s - loss: 0.3494 - accuracy: 0.8864
21100/25000 [========================>.....] - ETA: 2s - loss: 0.3495 - accuracy: 0.8864
21200/25000 [========================>.....] - ETA: 2s - loss: 0.3495 - accuracy: 0.8864
21300/25000 [========================>.....] - ETA: 2s - loss: 0.3496 - accuracy: 0.8864
21400/25000 [========================>.....] - ETA: 2s - loss: 0.3496 - accuracy: 0.8864
21500/25000 [========================>.....] - ETA: 2s - loss: 0.3496 - accuracy: 0.8864
21600/25000 [========================>.....] - ETA: 2s - loss: 0.3497 - accuracy: 0.8863
21700/25000 [=========================>....] - ETA: 2s - loss: 0.3496 - accuracy: 0.8864
21800/25000 [=========================>....] - ETA: 2s - loss: 0.3497 - accuracy: 0.8864
21900/25000 [=========================>....] - ETA: 2s - loss: 0.3496 - accuracy: 0.8864
22000/25000 [=========================>....] - ETA: 1s - loss: 0.3497 - accuracy: 0.8863
22100/25000 [=========================>....] - ETA: 1s - loss: 0.3497 - accuracy: 0.8863
22200/25000 [=========================>....] - ETA: 1s - loss: 0.3497 - accuracy: 0.8863
22300/25000 [=========================>....] - ETA: 1s - loss: 0.3497 - accuracy: 0.8863
22400/25000 [=========================>....] - ETA: 1s - loss: 0.3498 - accuracy: 0.8863
22500/25000 [==========================>...] - ETA: 1s - loss: 0.3497 - accuracy: 0.8863
22600/25000 [==========================>...] - ETA: 1s - loss: 0.3497 - accuracy: 0.8863
22700/25000 [==========================>...] - ETA: 1s - loss: 0.3496 - accuracy: 0.8863
22800/25000 [==========================>...] - ETA: 1s - loss: 0.3497 - accuracy: 0.8863
22900/25000 [==========================>...] - ETA: 1s - loss: 0.3496 - accuracy: 0.8864
23000/25000 [==========================>...] - ETA: 1s - loss: 0.3497 - accuracy: 0.8863
23100/25000 [==========================>...] - ETA: 1s - loss: 0.3497 - accuracy: 0.8863
23200/25000 [==========================>...] - ETA: 1s - loss: 0.3499 - accuracy: 0.8862
23300/25000 [==========================>...] - ETA: 1s - loss: 0.3499 - accuracy: 0.8862
23400/25000 [===========================>..] - ETA: 1s - loss: 0.3499 - accuracy: 0.8862
23500/25000 [===========================>..] - ETA: 0s - loss: 0.3499 - accuracy: 0.8862
23600/25000 [===========================>..] - ETA: 0s - loss: 0.3500 - accuracy: 0.8862
23700/25000 [===========================>..] - ETA: 0s - loss: 0.3499 - accuracy: 0.8862
23800/25000 [===========================>..] - ETA: 0s - loss: 0.3498 - accuracy: 0.8863
23900/25000 [===========================>..] - ETA: 0s - loss: 0.3498 - accuracy: 0.8863
24000/25000 [===========================>..] - ETA: 0s - loss: 0.3497 - accuracy: 0.8863
24100/25000 [===========================>..] - ETA: 0s - loss: 0.3498 - accuracy: 0.8863
24200/25000 [============================>.] - ETA: 0s - loss: 0.3497 - accuracy: 0.8863
24300/25000 [============================>.] - ETA: 0s - loss: 0.3497 - accuracy: 0.8863
24400/25000 [============================>.] - ETA: 0s - loss: 0.3497 - accuracy: 0.8863
24500/25000 [============================>.] - ETA: 0s - loss: 0.3497 - accuracy: 0.8863
24600/25000 [============================>.] - ETA: 0s - loss: 0.3497 - accuracy: 0.8863
24700/25000 [============================>.] - ETA: 0s - loss: 0.3497 - accuracy: 0.8863
24800/25000 [============================>.] - ETA: 0s - loss: 0.3497 - accuracy: 0.8863
24900/25000 [============================>.] - ETA: 0s - loss: 0.3498 - accuracy: 0.8863
25000/25000 [==============================] - 20s 816us/step - loss: 0.3498 - accuracy: 0.8863 - val_loss: 0.3498 - val_accuracy: 0.8858
	=====> Test the model: model.predict()
	Dataset: IMDB_REVIEWS
	Algorithm: Deep Learning using Keras 2 (KerasDL2)
	Training Loss: 0.3497
	Training accuracy score: 88.63%
	Test Loss: 0.3498
	Test Accuracy: 88.58%
	Training Time: 201.2245
	Test Time: 6.4431




FINAL CLASSIFICATION TABLE:

| ID | Dataset | Algorithm | Loss | Training accuracy score (%) | Test accuracy score (%) | Training time (seconds) | Test time (seconds) |
| --- | ------- | --------- | ---- | --------------------------- | ----------------------- | ----------------------- | ------------------- |
| 1 | IMDB_REVIEWS | Deep Learning using Keras 1 (KerasDL1) | 0.0103 | 99.94 | 87.07 | 178.0648 | 2.9642 |
| 2 | IMDB_REVIEWS | Deep Learning using Keras 2 (KerasDL2) | 0.3497 | 88.63 | 88.58 | 201.2245 | 6.4431 |

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
Program finished. It took 469.06244015693665 seconds

Process finished with exit code 0
```