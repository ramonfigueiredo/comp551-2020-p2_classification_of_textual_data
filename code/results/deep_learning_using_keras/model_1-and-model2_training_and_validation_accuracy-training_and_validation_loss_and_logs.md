## Two Deep Learning approaches using Keras: Loss, Training Accuracy, Test Accuracy, Training Time, Test Time

| ID | Dataset | Algorithm | Loss | Training accuracy score (%) | Test accuracy score (%) | Training time (seconds) | Test time (seconds) |
| --- | ------- | --------- | ---- | --------------------------- | ----------------------- | ----------------------- | ------------------- |
| 1 | TWENTY_NEWS_GROUPS | Deep Learning using Keras 1 (KerasDL1) | 0.1164 | 99.47 | 96.66 | 95.7168 | 1.6525 |
| 2 | IMDB_REVIEWS | Deep Learning using Keras 1 (KerasDL1) | 0.9344 | 99.98 | 83.16 | 165.9841 | 3.2173 |
| 3 | TWENTY_NEWS_GROUPS | Deep Learning using Keras 2 (KerasDL2) | 0.2558 | 94.96 | 94.96 | 82.9633 | 2.8831 |
| 4 | IMDB_REVIEWS | Deep Learning using Keras 2 (KerasDL2) | 0.4072 | 95.71 | 85.54 | 194.0329 | 6.2445 |

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
2020-03-09 19:39:31.429045: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libnvinfer.so.6'; dlerror: libnvinfer.so.6: cannot open shared object file: No such file or directory
2020-03-09 19:39:31.429098: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libnvinfer_plugin.so.6'; dlerror: libnvinfer_plugin.so.6: cannot open shared object file: No such file or directory
2020-03-09 19:39:31.429103: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:30] Cannot dlopen some TensorRT libraries. If you would like to use Nvidia GPU with TensorRT, please make sure the missing libraries mentioned above are installed properly.
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
03/09/2020 07:39:32 PM - INFO - Program started...
03/09/2020 07:39:32 PM - INFO - Program started...
data loaded
11314 documents - 13.782MB (training set)
7532 documents - 8.262MB (test set)
20 categories

Extracting features from the training data using a vectorizer
done in 1.263750s at 10.906MB/s
n_samples: 11314, n_features: 101321

Extracting features from the test data using the same vectorizer
done in 0.628999s at 13.135MB/s
n_samples: 7532, n_features: 101321

================================================================================
KERAS DEEP LEARNING MODEL
Using layers:
	==> Dense(10, input_dim=input_dim, activation='relu')
	==> Dense(19, activation='sigmoid')
Compile option:
	==> model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
2020-03-09 19:39:35.713299: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
2020-03-09 19:39:35.736910: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-03-09 19:39:35.737635: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1555] Found device 0 with properties: 
pciBusID: 0000:01:00.0 name: GeForce GTX 1080 computeCapability: 6.1
coreClock: 1.7335GHz coreCount: 20 deviceMemorySize: 7.93GiB deviceMemoryBandwidth: 298.32GiB/s
2020-03-09 19:39:35.737697: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcudart.so.10.1'; dlerror: libcudart.so.10.1: cannot open shared object file: No such file or directory
2020-03-09 19:39:35.737740: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcublas.so.10'; dlerror: libcublas.so.10: cannot open shared object file: No such file or directory
2020-03-09 19:39:35.737780: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcufft.so.10'; dlerror: libcufft.so.10: cannot open shared object file: No such file or directory
2020-03-09 19:39:35.737820: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcurand.so.10'; dlerror: libcurand.so.10: cannot open shared object file: No such file or directory
2020-03-09 19:39:35.737859: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcusolver.so.10'; dlerror: libcusolver.so.10: cannot open shared object file: No such file or directory
2020-03-09 19:39:35.737899: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcusparse.so.10'; dlerror: libcusparse.so.10: cannot open shared object file: No such file or directory
2020-03-09 19:39:35.739771: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2020-03-09 19:39:35.739782: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1592] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
Skipping registering GPU devices...
2020-03-09 19:39:35.739958: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-03-09 19:39:35.763712: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 3600000000 Hz
2020-03-09 19:39:35.764144: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5e56070 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-03-09 19:39:35.764167: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2020-03-09 19:39:35.839410: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-03-09 19:39:35.840062: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5df3460 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2020-03-09 19:39:35.840073: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): GeForce GTX 1080, Compute Capability 6.1
2020-03-09 19:39:35.840168: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1096] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-03-09 19:39:35.840173: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1102]      
	Dataset: TWENTY_NEWS_GROUPS
	Algorithm: Deep Learning using Keras 1 (KerasDL1)
	Loss: 0.0241
	Training accuracy score: 99.47%
	Test Accuracy: 96.66%
	Training Time: 95.7168
	Test Time: 1.6525


Loading IMDB_REVIEWS dataset:

===> Reading files from /home/ets-crchum/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/train/neg

===> Reading files from /home/ets-crchum/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/train/pos

===> Reading files from /home/ets-crchum/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/test/neg

===> Reading files from /home/ets-crchum/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/test/pos
data loaded
25000 documents - 33.133MB (training set)
25000 documents - 32.351MB (test set)

Extracting features from the training data using a vectorizer
done in 2.931770s at 11.301MB/s
n_samples: 25000, n_features: 74170

Extracting features from the test data using the same vectorizer
done in 2.892533s at 11.184MB/s
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
	Loss: 0.0003
	Training accuracy score: 99.98%
	Test Accuracy: 83.16%
	Training Time: 165.9841
	Test Time: 3.2173


Loading TWENTY_NEWS_GROUPS dataset for categories:
03/09/2020 07:44:14 PM - INFO - Program started...
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
	It took 10.15224003791809 seconds


Applying NLTK feature extraction: X_test
	==> Replace unused UNICODE characters. 
	==> Text in lower case. 
	==> Apply Lemmatize using WordNet's built-in morphy function. Returns the input word unchanged if it cannot be found in WordNet 
	==> Remove English stop words
	Done: NLTK feature extraction (replace unused UNICODE characters, text in lower case, apply Lemmatize using WordNet, remove English stop words)  finished!
	It took 5.918877601623535 seconds

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

  100/11314 [..............................] - ETA: 59s - loss: 0.6846 - accuracy: 0.5684
  200/11314 [..............................] - ETA: 33s - loss: 0.6841 - accuracy: 0.5703
  300/11314 [..............................] - ETA: 24s - loss: 0.6839 - accuracy: 0.5698
  400/11314 [>.............................] - ETA: 19s - loss: 0.6836 - accuracy: 0.5692
  500/11314 [>.............................] - ETA: 16s - loss: 0.6835 - accuracy: 0.5686
  600/11314 [>.............................] - ETA: 14s - loss: 0.6831 - accuracy: 0.5691
  700/11314 [>.............................] - ETA: 13s - loss: 0.6827 - accuracy: 0.5695
  800/11314 [=>............................] - ETA: 12s - loss: 0.6823 - accuracy: 0.5704
  900/11314 [=>............................] - ETA: 11s - loss: 0.6820 - accuracy: 0.5704
 1000/11314 [=>............................] - ETA: 11s - loss: 0.6818 - accuracy: 0.5695
 1100/11314 [=>............................] - ETA: 10s - loss: 0.6814 - accuracy: 0.5695
 1200/11314 [==>...........................] - ETA: 10s - loss: 0.6812 - accuracy: 0.5686
 1300/11314 [==>...........................] - ETA: 9s - loss: 0.6809 - accuracy: 0.5688 
 1400/11314 [==>...........................] - ETA: 9s - loss: 0.6804 - accuracy: 0.5695
 1500/11314 [==>...........................] - ETA: 9s - loss: 0.6800 - accuracy: 0.5696
 1600/11314 [===>..........................] - ETA: 8s - loss: 0.6797 - accuracy: 0.5692
 1700/11314 [===>..........................] - ETA: 8s - loss: 0.6793 - accuracy: 0.5696
 1800/11314 [===>..........................] - ETA: 8s - loss: 0.6790 - accuracy: 0.5692
 1900/11314 [====>.........................] - ETA: 8s - loss: 0.6785 - accuracy: 0.5700
 2000/11314 [====>.........................] - ETA: 7s - loss: 0.6781 - accuracy: 0.5702
 2100/11314 [====>.........................] - ETA: 7s - loss: 0.6777 - accuracy: 0.5703
 2200/11314 [====>.........................] - ETA: 7s - loss: 0.6772 - accuracy: 0.5706
 2300/11314 [=====>........................] - ETA: 7s - loss: 0.6768 - accuracy: 0.5709
 2400/11314 [=====>........................] - ETA: 7s - loss: 0.6765 - accuracy: 0.5708
 2500/11314 [=====>........................] - ETA: 7s - loss: 0.6761 - accuracy: 0.5708
 2600/11314 [=====>........................] - ETA: 6s - loss: 0.6757 - accuracy: 0.5707
 2700/11314 [======>.......................] - ETA: 6s - loss: 0.6753 - accuracy: 0.5708
 2800/11314 [======>.......................] - ETA: 6s - loss: 0.6748 - accuracy: 0.5710
 2900/11314 [======>.......................] - ETA: 6s - loss: 0.6744 - accuracy: 0.5710
 3000/11314 [======>.......................] - ETA: 6s - loss: 0.6739 - accuracy: 0.5713
 3100/11314 [=======>......................] - ETA: 6s - loss: 0.6735 - accuracy: 0.5712
 3200/11314 [=======>......................] - ETA: 6s - loss: 0.6730 - accuracy: 0.5716
 3300/11314 [=======>......................] - ETA: 6s - loss: 0.6726 - accuracy: 0.5714
 3400/11314 [========>.....................] - ETA: 5s - loss: 0.6721 - accuracy: 0.5714
 3500/11314 [========>.....................] - ETA: 5s - loss: 0.6717 - accuracy: 0.5714
 3600/11314 [========>.....................] - ETA: 5s - loss: 0.6712 - accuracy: 0.5716
 3700/11314 [========>.....................] - ETA: 5s - loss: 0.6707 - accuracy: 0.5716
 3800/11314 [=========>....................] - ETA: 5s - loss: 0.6701 - accuracy: 0.5720
 3900/11314 [=========>....................] - ETA: 5s - loss: 0.6696 - accuracy: 0.5720
 4000/11314 [=========>....................] - ETA: 5s - loss: 0.6692 - accuracy: 0.5718
 4100/11314 [=========>....................] - ETA: 5s - loss: 0.6688 - accuracy: 0.5715
 4200/11314 [==========>...................] - ETA: 5s - loss: 0.6684 - accuracy: 0.5714
 4300/11314 [==========>...................] - ETA: 5s - loss: 0.6679 - accuracy: 0.5713
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.6674 - accuracy: 0.5714
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.6669 - accuracy: 0.5714
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.6664 - accuracy: 0.5715
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.6658 - accuracy: 0.5716
 4800/11314 [===========>..................] - ETA: 4s - loss: 0.6653 - accuracy: 0.5719
 4900/11314 [===========>..................] - ETA: 4s - loss: 0.6648 - accuracy: 0.5721
 5000/11314 [============>.................] - ETA: 4s - loss: 0.6643 - accuracy: 0.5720
 5100/11314 [============>.................] - ETA: 4s - loss: 0.6638 - accuracy: 0.5719
 5200/11314 [============>.................] - ETA: 4s - loss: 0.6633 - accuracy: 0.5718
 5300/11314 [=============>................] - ETA: 4s - loss: 0.6628 - accuracy: 0.5718
 5400/11314 [=============>................] - ETA: 4s - loss: 0.6623 - accuracy: 0.5718
 5500/11314 [=============>................] - ETA: 4s - loss: 0.6618 - accuracy: 0.5719
 5600/11314 [=============>................] - ETA: 3s - loss: 0.6613 - accuracy: 0.5720
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.6608 - accuracy: 0.5728
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.6602 - accuracy: 0.5737
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.6597 - accuracy: 0.5746
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.6593 - accuracy: 0.5753
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.6588 - accuracy: 0.5760
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.6583 - accuracy: 0.5767
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.6578 - accuracy: 0.5774
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.6573 - accuracy: 0.5781
 6500/11314 [================>.............] - ETA: 3s - loss: 0.6568 - accuracy: 0.5786
 6600/11314 [================>.............] - ETA: 3s - loss: 0.6563 - accuracy: 0.5793
 6700/11314 [================>.............] - ETA: 3s - loss: 0.6558 - accuracy: 0.5798
 6800/11314 [=================>............] - ETA: 3s - loss: 0.6553 - accuracy: 0.5805
 6900/11314 [=================>............] - ETA: 3s - loss: 0.6549 - accuracy: 0.5810
 7000/11314 [=================>............] - ETA: 2s - loss: 0.6544 - accuracy: 0.5815
 7100/11314 [=================>............] - ETA: 2s - loss: 0.6539 - accuracy: 0.5821
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.6534 - accuracy: 0.5826
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.6529 - accuracy: 0.5832
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.6525 - accuracy: 0.5836
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.6520 - accuracy: 0.5847
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.6515 - accuracy: 0.5857
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.6510 - accuracy: 0.5868
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.6505 - accuracy: 0.5878
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.6501 - accuracy: 0.5888
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.6496 - accuracy: 0.5897
 8100/11314 [====================>.........] - ETA: 2s - loss: 0.6491 - accuracy: 0.5907
 8200/11314 [====================>.........] - ETA: 2s - loss: 0.6486 - accuracy: 0.5917
 8300/11314 [=====================>........] - ETA: 2s - loss: 0.6481 - accuracy: 0.5927
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.6477 - accuracy: 0.5935
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.6472 - accuracy: 0.5943
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.6468 - accuracy: 0.5951
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.6463 - accuracy: 0.5960
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.6459 - accuracy: 0.5968
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.6454 - accuracy: 0.5976
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.6449 - accuracy: 0.5984
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.6445 - accuracy: 0.5991
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.6441 - accuracy: 0.5998
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.6436 - accuracy: 0.6004
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.6432 - accuracy: 0.6010
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.6428 - accuracy: 0.6016
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.6423 - accuracy: 0.6023
 9700/11314 [========================>.....] - ETA: 1s - loss: 0.6418 - accuracy: 0.6030
 9800/11314 [========================>.....] - ETA: 1s - loss: 0.6414 - accuracy: 0.6041
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.6409 - accuracy: 0.6053
10000/11314 [=========================>....] - ETA: 0s - loss: 0.6405 - accuracy: 0.6068
10100/11314 [=========================>....] - ETA: 0s - loss: 0.6400 - accuracy: 0.6084
10200/11314 [==========================>...] - ETA: 0s - loss: 0.6396 - accuracy: 0.6099
10300/11314 [==========================>...] - ETA: 0s - loss: 0.6391 - accuracy: 0.6114
10400/11314 [==========================>...] - ETA: 0s - loss: 0.6387 - accuracy: 0.6128
10500/11314 [==========================>...] - ETA: 0s - loss: 0.6382 - accuracy: 0.6142
10600/11314 [===========================>..] - ETA: 0s - loss: 0.6378 - accuracy: 0.6155
10700/11314 [===========================>..] - ETA: 0s - loss: 0.6374 - accuracy: 0.6169
10800/11314 [===========================>..] - ETA: 0s - loss: 0.6369 - accuracy: 0.6187
10900/11314 [===========================>..] - ETA: 0s - loss: 0.6365 - accuracy: 0.6204
11000/11314 [============================>.] - ETA: 0s - loss: 0.6360 - accuracy: 0.6221
11100/11314 [============================>.] - ETA: 0s - loss: 0.6356 - accuracy: 0.6237
11200/11314 [============================>.] - ETA: 0s - loss: 0.6352 - accuracy: 0.6253
11300/11314 [============================>.] - ETA: 0s - loss: 0.6347 - accuracy: 0.6269
11314/11314 [==============================] - 9s 772us/step - loss: 0.6347 - accuracy: 0.6272 - val_loss: 0.5840 - val_accuracy: 0.8071
Epoch 2/10

  100/11314 [..............................] - ETA: 6s - loss: 0.5850 - accuracy: 0.8047
  200/11314 [..............................] - ETA: 6s - loss: 0.5834 - accuracy: 0.8061
  300/11314 [..............................] - ETA: 6s - loss: 0.5835 - accuracy: 0.8061
  400/11314 [>.............................] - ETA: 6s - loss: 0.5836 - accuracy: 0.8061
  500/11314 [>.............................] - ETA: 6s - loss: 0.5826 - accuracy: 0.8066
  600/11314 [>.............................] - ETA: 6s - loss: 0.5821 - accuracy: 0.8071
  700/11314 [>.............................] - ETA: 6s - loss: 0.5816 - accuracy: 0.8070
  800/11314 [=>............................] - ETA: 6s - loss: 0.5816 - accuracy: 0.8061
  900/11314 [=>............................] - ETA: 6s - loss: 0.5810 - accuracy: 0.8063
 1000/11314 [=>............................] - ETA: 6s - loss: 0.5805 - accuracy: 0.8067
 1100/11314 [=>............................] - ETA: 6s - loss: 0.5801 - accuracy: 0.8071
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.5798 - accuracy: 0.8072
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.5794 - accuracy: 0.8077
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.5790 - accuracy: 0.8080
 1500/11314 [==>...........................] - ETA: 6s - loss: 0.5786 - accuracy: 0.8081
 1600/11314 [===>..........................] - ETA: 5s - loss: 0.5783 - accuracy: 0.8078
 1700/11314 [===>..........................] - ETA: 5s - loss: 0.5780 - accuracy: 0.8078
 1800/11314 [===>..........................] - ETA: 5s - loss: 0.5777 - accuracy: 0.8075
 1900/11314 [====>.........................] - ETA: 5s - loss: 0.5772 - accuracy: 0.8075
 2000/11314 [====>.........................] - ETA: 5s - loss: 0.5767 - accuracy: 0.8079
 2100/11314 [====>.........................] - ETA: 5s - loss: 0.5765 - accuracy: 0.8076
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.5762 - accuracy: 0.8073
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.5759 - accuracy: 0.8071
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.5756 - accuracy: 0.8071
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.5752 - accuracy: 0.8072
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.5748 - accuracy: 0.8074
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.5743 - accuracy: 0.8077
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.5738 - accuracy: 0.8079
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.5734 - accuracy: 0.8079
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.5731 - accuracy: 0.8078
 3100/11314 [=======>......................] - ETA: 5s - loss: 0.5728 - accuracy: 0.8077
 3200/11314 [=======>......................] - ETA: 4s - loss: 0.5724 - accuracy: 0.8075
 3300/11314 [=======>......................] - ETA: 4s - loss: 0.5720 - accuracy: 0.8075
 3400/11314 [========>.....................] - ETA: 4s - loss: 0.5717 - accuracy: 0.8074
 3500/11314 [========>.....................] - ETA: 4s - loss: 0.5713 - accuracy: 0.8074
 3600/11314 [========>.....................] - ETA: 4s - loss: 0.5710 - accuracy: 0.8072
 3700/11314 [========>.....................] - ETA: 4s - loss: 0.5707 - accuracy: 0.8070
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.5703 - accuracy: 0.8070
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.5700 - accuracy: 0.8070
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.5697 - accuracy: 0.8070
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.5693 - accuracy: 0.8069
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.5690 - accuracy: 0.8068
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.5686 - accuracy: 0.8068
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.5683 - accuracy: 0.8069
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.5679 - accuracy: 0.8071
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.5675 - accuracy: 0.8072
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.5672 - accuracy: 0.8070
 4800/11314 [===========>..................] - ETA: 3s - loss: 0.5668 - accuracy: 0.8071
 4900/11314 [===========>..................] - ETA: 3s - loss: 0.5664 - accuracy: 0.8072
 5000/11314 [============>.................] - ETA: 3s - loss: 0.5661 - accuracy: 0.8072
 5100/11314 [============>.................] - ETA: 3s - loss: 0.5657 - accuracy: 0.8072
 5200/11314 [============>.................] - ETA: 3s - loss: 0.5653 - accuracy: 0.8072
 5300/11314 [=============>................] - ETA: 3s - loss: 0.5650 - accuracy: 0.8073
 5400/11314 [=============>................] - ETA: 3s - loss: 0.5646 - accuracy: 0.8074
 5500/11314 [=============>................] - ETA: 3s - loss: 0.5643 - accuracy: 0.8073
 5600/11314 [=============>................] - ETA: 3s - loss: 0.5639 - accuracy: 0.8072
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.5636 - accuracy: 0.8072
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.5632 - accuracy: 0.8071
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.5629 - accuracy: 0.8071
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.5625 - accuracy: 0.8071
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.5622 - accuracy: 0.8071
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.5618 - accuracy: 0.8072
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.5614 - accuracy: 0.8072
 6400/11314 [===============>..............] - ETA: 2s - loss: 0.5611 - accuracy: 0.8072
 6500/11314 [================>.............] - ETA: 2s - loss: 0.5607 - accuracy: 0.8073
 6600/11314 [================>.............] - ETA: 2s - loss: 0.5604 - accuracy: 0.8072
 6700/11314 [================>.............] - ETA: 2s - loss: 0.5600 - accuracy: 0.8073
 6800/11314 [=================>............] - ETA: 2s - loss: 0.5597 - accuracy: 0.8073
 6900/11314 [=================>............] - ETA: 2s - loss: 0.5593 - accuracy: 0.8073
 7000/11314 [=================>............] - ETA: 2s - loss: 0.5590 - accuracy: 0.8073
 7100/11314 [=================>............] - ETA: 2s - loss: 0.5587 - accuracy: 0.8072
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.5584 - accuracy: 0.8080
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.5580 - accuracy: 0.8092
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.5576 - accuracy: 0.8105
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.5573 - accuracy: 0.8118
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.5569 - accuracy: 0.8130
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.5566 - accuracy: 0.8142
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.5563 - accuracy: 0.8159
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.5559 - accuracy: 0.8176
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.5556 - accuracy: 0.8192
 8100/11314 [====================>.........] - ETA: 1s - loss: 0.5553 - accuracy: 0.8208
 8200/11314 [====================>.........] - ETA: 1s - loss: 0.5550 - accuracy: 0.8224
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.5546 - accuracy: 0.8239
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.5543 - accuracy: 0.8255
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.5539 - accuracy: 0.8269
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.5536 - accuracy: 0.8283
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.5532 - accuracy: 0.8297
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.5529 - accuracy: 0.8311
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.5525 - accuracy: 0.8324
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.5521 - accuracy: 0.8337
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.5519 - accuracy: 0.8350
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.5515 - accuracy: 0.8362
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.5512 - accuracy: 0.8374
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.5508 - accuracy: 0.8386
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.5505 - accuracy: 0.8398
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.5501 - accuracy: 0.8410
 9700/11314 [========================>.....] - ETA: 0s - loss: 0.5498 - accuracy: 0.8421
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.5494 - accuracy: 0.8432
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.5492 - accuracy: 0.8443
10000/11314 [=========================>....] - ETA: 0s - loss: 0.5488 - accuracy: 0.8453
10100/11314 [=========================>....] - ETA: 0s - loss: 0.5485 - accuracy: 0.8463
10200/11314 [==========================>...] - ETA: 0s - loss: 0.5482 - accuracy: 0.8473
10300/11314 [==========================>...] - ETA: 0s - loss: 0.5479 - accuracy: 0.8483
10400/11314 [==========================>...] - ETA: 0s - loss: 0.5475 - accuracy: 0.8493
10500/11314 [==========================>...] - ETA: 0s - loss: 0.5472 - accuracy: 0.8503
10600/11314 [===========================>..] - ETA: 0s - loss: 0.5469 - accuracy: 0.8512
10700/11314 [===========================>..] - ETA: 0s - loss: 0.5466 - accuracy: 0.8521
10800/11314 [===========================>..] - ETA: 0s - loss: 0.5462 - accuracy: 0.8530
10900/11314 [===========================>..] - ETA: 0s - loss: 0.5459 - accuracy: 0.8539
11000/11314 [============================>.] - ETA: 0s - loss: 0.5456 - accuracy: 0.8548
11100/11314 [============================>.] - ETA: 0s - loss: 0.5453 - accuracy: 0.8556
11200/11314 [============================>.] - ETA: 0s - loss: 0.5450 - accuracy: 0.8564
11300/11314 [============================>.] - ETA: 0s - loss: 0.5446 - accuracy: 0.8573
11314/11314 [==============================] - 8s 719us/step - loss: 0.5446 - accuracy: 0.8574 - val_loss: 0.5072 - val_accuracy: 0.9496
Epoch 3/10

  100/11314 [..............................] - ETA: 6s - loss: 0.5060 - accuracy: 0.9489
  200/11314 [..............................] - ETA: 6s - loss: 0.5055 - accuracy: 0.9489
  300/11314 [..............................] - ETA: 6s - loss: 0.5057 - accuracy: 0.9496
  400/11314 [>.............................] - ETA: 6s - loss: 0.5058 - accuracy: 0.9503
  500/11314 [>.............................] - ETA: 6s - loss: 0.5062 - accuracy: 0.9500
  600/11314 [>.............................] - ETA: 6s - loss: 0.5056 - accuracy: 0.9501
  700/11314 [>.............................] - ETA: 6s - loss: 0.5057 - accuracy: 0.9499
  800/11314 [=>............................] - ETA: 6s - loss: 0.5053 - accuracy: 0.9500
  900/11314 [=>............................] - ETA: 6s - loss: 0.5049 - accuracy: 0.9499
 1000/11314 [=>............................] - ETA: 6s - loss: 0.5048 - accuracy: 0.9498
 1100/11314 [=>............................] - ETA: 6s - loss: 0.5046 - accuracy: 0.9498
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.5042 - accuracy: 0.9498
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.5039 - accuracy: 0.9498
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.5034 - accuracy: 0.9499
 1500/11314 [==>...........................] - ETA: 6s - loss: 0.5034 - accuracy: 0.9498
 1600/11314 [===>..........................] - ETA: 5s - loss: 0.5031 - accuracy: 0.9497
 1700/11314 [===>..........................] - ETA: 5s - loss: 0.5027 - accuracy: 0.9497
 1800/11314 [===>..........................] - ETA: 5s - loss: 0.5025 - accuracy: 0.9496
 1900/11314 [====>.........................] - ETA: 5s - loss: 0.5022 - accuracy: 0.9496
 2000/11314 [====>.........................] - ETA: 5s - loss: 0.5020 - accuracy: 0.9497
 2100/11314 [====>.........................] - ETA: 5s - loss: 0.5017 - accuracy: 0.9496
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.5014 - accuracy: 0.9496
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.5012 - accuracy: 0.9495
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.5009 - accuracy: 0.9496
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.5006 - accuracy: 0.9495
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.5002 - accuracy: 0.9496
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.4999 - accuracy: 0.9496
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.4997 - accuracy: 0.9496
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.4992 - accuracy: 0.9497
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.4990 - accuracy: 0.9497
 3100/11314 [=======>......................] - ETA: 5s - loss: 0.4987 - accuracy: 0.9497
 3200/11314 [=======>......................] - ETA: 4s - loss: 0.4985 - accuracy: 0.9497
 3300/11314 [=======>......................] - ETA: 4s - loss: 0.4982 - accuracy: 0.9497
 3400/11314 [========>.....................] - ETA: 4s - loss: 0.4979 - accuracy: 0.9497
 3500/11314 [========>.....................] - ETA: 4s - loss: 0.4976 - accuracy: 0.9497
 3600/11314 [========>.....................] - ETA: 4s - loss: 0.4973 - accuracy: 0.9497
 3700/11314 [========>.....................] - ETA: 4s - loss: 0.4970 - accuracy: 0.9497
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.4967 - accuracy: 0.9497
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.4965 - accuracy: 0.9497
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.4962 - accuracy: 0.9497
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.4959 - accuracy: 0.9497
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.4956 - accuracy: 0.9497
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.4953 - accuracy: 0.9497
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.4950 - accuracy: 0.9497
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.4947 - accuracy: 0.9497
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.4944 - accuracy: 0.9497
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.4941 - accuracy: 0.9497
 4800/11314 [===========>..................] - ETA: 3s - loss: 0.4938 - accuracy: 0.9497
 4900/11314 [===========>..................] - ETA: 3s - loss: 0.4935 - accuracy: 0.9497
 5000/11314 [============>.................] - ETA: 3s - loss: 0.4932 - accuracy: 0.9497
 5100/11314 [============>.................] - ETA: 3s - loss: 0.4929 - accuracy: 0.9497
 5200/11314 [============>.................] - ETA: 3s - loss: 0.4927 - accuracy: 0.9497
 5300/11314 [=============>................] - ETA: 3s - loss: 0.4924 - accuracy: 0.9497
 5400/11314 [=============>................] - ETA: 3s - loss: 0.4921 - accuracy: 0.9497
 5500/11314 [=============>................] - ETA: 3s - loss: 0.4919 - accuracy: 0.9497
 5600/11314 [=============>................] - ETA: 3s - loss: 0.4916 - accuracy: 0.9497
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.4913 - accuracy: 0.9497
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.4910 - accuracy: 0.9497
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.4907 - accuracy: 0.9497
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.4905 - accuracy: 0.9497
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.4902 - accuracy: 0.9497
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.4899 - accuracy: 0.9497
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.4896 - accuracy: 0.9497
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.4894 - accuracy: 0.9497
 6500/11314 [================>.............] - ETA: 2s - loss: 0.4891 - accuracy: 0.9497
 6600/11314 [================>.............] - ETA: 2s - loss: 0.4889 - accuracy: 0.9497
 6700/11314 [================>.............] - ETA: 2s - loss: 0.4886 - accuracy: 0.9496
 6800/11314 [=================>............] - ETA: 2s - loss: 0.4884 - accuracy: 0.9497
 6900/11314 [=================>............] - ETA: 2s - loss: 0.4881 - accuracy: 0.9497
 7000/11314 [=================>............] - ETA: 2s - loss: 0.4878 - accuracy: 0.9497
 7100/11314 [=================>............] - ETA: 2s - loss: 0.4875 - accuracy: 0.9497
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.4872 - accuracy: 0.9497
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.4869 - accuracy: 0.9497
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.4867 - accuracy: 0.9497
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.4864 - accuracy: 0.9496
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.4861 - accuracy: 0.9496
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.4858 - accuracy: 0.9496
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.4856 - accuracy: 0.9496
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.4853 - accuracy: 0.9496
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.4850 - accuracy: 0.9496
 8100/11314 [====================>.........] - ETA: 1s - loss: 0.4847 - accuracy: 0.9496
 8200/11314 [====================>.........] - ETA: 1s - loss: 0.4845 - accuracy: 0.9496
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.4842 - accuracy: 0.9496
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.4839 - accuracy: 0.9496
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.4837 - accuracy: 0.9496
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.4834 - accuracy: 0.9496
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.4831 - accuracy: 0.9496
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.4828 - accuracy: 0.9496
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.4826 - accuracy: 0.9496
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.4823 - accuracy: 0.9496
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.4821 - accuracy: 0.9496
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.4818 - accuracy: 0.9496
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.4815 - accuracy: 0.9496
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.4812 - accuracy: 0.9496
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.4809 - accuracy: 0.9496
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.4807 - accuracy: 0.9496
 9700/11314 [========================>.....] - ETA: 0s - loss: 0.4804 - accuracy: 0.9496
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.4802 - accuracy: 0.9496
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.4799 - accuracy: 0.9496
10000/11314 [=========================>....] - ETA: 0s - loss: 0.4797 - accuracy: 0.9496
10100/11314 [=========================>....] - ETA: 0s - loss: 0.4794 - accuracy: 0.9496
10200/11314 [==========================>...] - ETA: 0s - loss: 0.4791 - accuracy: 0.9496
10300/11314 [==========================>...] - ETA: 0s - loss: 0.4789 - accuracy: 0.9496
10400/11314 [==========================>...] - ETA: 0s - loss: 0.4786 - accuracy: 0.9496
10500/11314 [==========================>...] - ETA: 0s - loss: 0.4784 - accuracy: 0.9496
10600/11314 [===========================>..] - ETA: 0s - loss: 0.4781 - accuracy: 0.9496
10700/11314 [===========================>..] - ETA: 0s - loss: 0.4778 - accuracy: 0.9496
10800/11314 [===========================>..] - ETA: 0s - loss: 0.4775 - accuracy: 0.9496
10900/11314 [===========================>..] - ETA: 0s - loss: 0.4773 - accuracy: 0.9496
11000/11314 [============================>.] - ETA: 0s - loss: 0.4770 - accuracy: 0.9496
11100/11314 [============================>.] - ETA: 0s - loss: 0.4768 - accuracy: 0.9496
11200/11314 [============================>.] - ETA: 0s - loss: 0.4765 - accuracy: 0.9496
11300/11314 [============================>.] - ETA: 0s - loss: 0.4763 - accuracy: 0.9496
11314/11314 [==============================] - 8s 721us/step - loss: 0.4762 - accuracy: 0.9496 - val_loss: 0.4467 - val_accuracy: 0.9496
Epoch 4/10

  100/11314 [..............................] - ETA: 6s - loss: 0.4493 - accuracy: 0.9484
  200/11314 [..............................] - ETA: 6s - loss: 0.4464 - accuracy: 0.9503
  300/11314 [..............................] - ETA: 6s - loss: 0.4453 - accuracy: 0.9505
  400/11314 [>.............................] - ETA: 6s - loss: 0.4451 - accuracy: 0.9501
  500/11314 [>.............................] - ETA: 6s - loss: 0.4446 - accuracy: 0.9503
  600/11314 [>.............................] - ETA: 6s - loss: 0.4445 - accuracy: 0.9502
  700/11314 [>.............................] - ETA: 6s - loss: 0.4442 - accuracy: 0.9501
  800/11314 [=>............................] - ETA: 6s - loss: 0.4441 - accuracy: 0.9498
  900/11314 [=>............................] - ETA: 6s - loss: 0.4438 - accuracy: 0.9498
 1000/11314 [=>............................] - ETA: 6s - loss: 0.4438 - accuracy: 0.9498
 1100/11314 [=>............................] - ETA: 6s - loss: 0.4435 - accuracy: 0.9499
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.4434 - accuracy: 0.9497
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.4432 - accuracy: 0.9497
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.4429 - accuracy: 0.9497
 1500/11314 [==>...........................] - ETA: 5s - loss: 0.4427 - accuracy: 0.9497
 1600/11314 [===>..........................] - ETA: 5s - loss: 0.4428 - accuracy: 0.9497
 1700/11314 [===>..........................] - ETA: 5s - loss: 0.4427 - accuracy: 0.9497
 1800/11314 [===>..........................] - ETA: 5s - loss: 0.4425 - accuracy: 0.9497
 1900/11314 [====>.........................] - ETA: 5s - loss: 0.4426 - accuracy: 0.9496
 2000/11314 [====>.........................] - ETA: 5s - loss: 0.4424 - accuracy: 0.9496
 2100/11314 [====>.........................] - ETA: 5s - loss: 0.4422 - accuracy: 0.9496
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.4421 - accuracy: 0.9495
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.4419 - accuracy: 0.9495
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.4416 - accuracy: 0.9495
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.4414 - accuracy: 0.9495
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.4410 - accuracy: 0.9496
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.4408 - accuracy: 0.9496
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.4405 - accuracy: 0.9496
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.4404 - accuracy: 0.9496
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.4401 - accuracy: 0.9496
 3100/11314 [=======>......................] - ETA: 5s - loss: 0.4398 - accuracy: 0.9496
 3200/11314 [=======>......................] - ETA: 4s - loss: 0.4397 - accuracy: 0.9495
 3300/11314 [=======>......................] - ETA: 4s - loss: 0.4394 - accuracy: 0.9496
 3400/11314 [========>.....................] - ETA: 4s - loss: 0.4391 - accuracy: 0.9496
 3500/11314 [========>.....................] - ETA: 4s - loss: 0.4390 - accuracy: 0.9495
 3600/11314 [========>.....................] - ETA: 4s - loss: 0.4387 - accuracy: 0.9496
 3700/11314 [========>.....................] - ETA: 4s - loss: 0.4385 - accuracy: 0.9496
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.4383 - accuracy: 0.9496
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.4381 - accuracy: 0.9496
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.4378 - accuracy: 0.9496
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.4376 - accuracy: 0.9496
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.4374 - accuracy: 0.9496
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.4371 - accuracy: 0.9496
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.4369 - accuracy: 0.9496
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.4367 - accuracy: 0.9496
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.4364 - accuracy: 0.9496
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.4362 - accuracy: 0.9496
 4800/11314 [===========>..................] - ETA: 3s - loss: 0.4359 - accuracy: 0.9496
 4900/11314 [===========>..................] - ETA: 3s - loss: 0.4357 - accuracy: 0.9496
 5000/11314 [============>.................] - ETA: 3s - loss: 0.4355 - accuracy: 0.9496
 5100/11314 [============>.................] - ETA: 3s - loss: 0.4353 - accuracy: 0.9495
 5200/11314 [============>.................] - ETA: 3s - loss: 0.4351 - accuracy: 0.9495
 5300/11314 [=============>................] - ETA: 3s - loss: 0.4349 - accuracy: 0.9496
 5400/11314 [=============>................] - ETA: 3s - loss: 0.4346 - accuracy: 0.9496
 5500/11314 [=============>................] - ETA: 3s - loss: 0.4344 - accuracy: 0.9496
 5600/11314 [=============>................] - ETA: 3s - loss: 0.4342 - accuracy: 0.9496
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.4339 - accuracy: 0.9496
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.4337 - accuracy: 0.9496
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.4335 - accuracy: 0.9496
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.4333 - accuracy: 0.9495
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.4331 - accuracy: 0.9496
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.4329 - accuracy: 0.9496
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.4327 - accuracy: 0.9496
 6400/11314 [===============>..............] - ETA: 2s - loss: 0.4325 - accuracy: 0.9496
 6500/11314 [================>.............] - ETA: 2s - loss: 0.4323 - accuracy: 0.9496
 6600/11314 [================>.............] - ETA: 2s - loss: 0.4321 - accuracy: 0.9496
 6700/11314 [================>.............] - ETA: 2s - loss: 0.4318 - accuracy: 0.9496
 6800/11314 [=================>............] - ETA: 2s - loss: 0.4316 - accuracy: 0.9496
 6900/11314 [=================>............] - ETA: 2s - loss: 0.4313 - accuracy: 0.9496
 7000/11314 [=================>............] - ETA: 2s - loss: 0.4311 - accuracy: 0.9496
 7100/11314 [=================>............] - ETA: 2s - loss: 0.4309 - accuracy: 0.9496
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.4307 - accuracy: 0.9496
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.4304 - accuracy: 0.9496
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.4302 - accuracy: 0.9496
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.4300 - accuracy: 0.9496
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.4298 - accuracy: 0.9496
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.4296 - accuracy: 0.9496
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.4294 - accuracy: 0.9497
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.4291 - accuracy: 0.9497
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.4289 - accuracy: 0.9497
 8100/11314 [====================>.........] - ETA: 1s - loss: 0.4287 - accuracy: 0.9497
 8200/11314 [====================>.........] - ETA: 1s - loss: 0.4285 - accuracy: 0.9497
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.4283 - accuracy: 0.9497
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.4281 - accuracy: 0.9497
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.4279 - accuracy: 0.9496
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.4276 - accuracy: 0.9497
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.4274 - accuracy: 0.9497
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.4272 - accuracy: 0.9497
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.4270 - accuracy: 0.9497
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.4268 - accuracy: 0.9497
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.4266 - accuracy: 0.9497
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.4264 - accuracy: 0.9496
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.4262 - accuracy: 0.9496
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.4260 - accuracy: 0.9497
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.4258 - accuracy: 0.9496
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.4256 - accuracy: 0.9496
 9700/11314 [========================>.....] - ETA: 0s - loss: 0.4254 - accuracy: 0.9496
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.4252 - accuracy: 0.9496
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.4250 - accuracy: 0.9496
10000/11314 [=========================>....] - ETA: 0s - loss: 0.4248 - accuracy: 0.9496
10100/11314 [=========================>....] - ETA: 0s - loss: 0.4246 - accuracy: 0.9496
10200/11314 [==========================>...] - ETA: 0s - loss: 0.4243 - accuracy: 0.9496
10300/11314 [==========================>...] - ETA: 0s - loss: 0.4241 - accuracy: 0.9496
10400/11314 [==========================>...] - ETA: 0s - loss: 0.4239 - accuracy: 0.9496
10500/11314 [==========================>...] - ETA: 0s - loss: 0.4237 - accuracy: 0.9496
10600/11314 [===========================>..] - ETA: 0s - loss: 0.4235 - accuracy: 0.9496
10700/11314 [===========================>..] - ETA: 0s - loss: 0.4233 - accuracy: 0.9496
10800/11314 [===========================>..] - ETA: 0s - loss: 0.4231 - accuracy: 0.9496
10900/11314 [===========================>..] - ETA: 0s - loss: 0.4229 - accuracy: 0.9496
11000/11314 [============================>.] - ETA: 0s - loss: 0.4227 - accuracy: 0.9496
11100/11314 [============================>.] - ETA: 0s - loss: 0.4225 - accuracy: 0.9496
11200/11314 [============================>.] - ETA: 0s - loss: 0.4223 - accuracy: 0.9496
11300/11314 [============================>.] - ETA: 0s - loss: 0.4221 - accuracy: 0.9496
11314/11314 [==============================] - 8s 719us/step - loss: 0.4221 - accuracy: 0.9496 - val_loss: 0.3985 - val_accuracy: 0.9496
Epoch 5/10

  100/11314 [..............................] - ETA: 6s - loss: 0.3976 - accuracy: 0.9500
  200/11314 [..............................] - ETA: 6s - loss: 0.3982 - accuracy: 0.9500
  300/11314 [..............................] - ETA: 6s - loss: 0.3969 - accuracy: 0.9504
  400/11314 [>.............................] - ETA: 6s - loss: 0.3971 - accuracy: 0.9503
  500/11314 [>.............................] - ETA: 6s - loss: 0.3974 - accuracy: 0.9498
  600/11314 [>.............................] - ETA: 6s - loss: 0.3972 - accuracy: 0.9498
  700/11314 [>.............................] - ETA: 6s - loss: 0.3972 - accuracy: 0.9496
  800/11314 [=>............................] - ETA: 6s - loss: 0.3970 - accuracy: 0.9495
  900/11314 [=>............................] - ETA: 6s - loss: 0.3969 - accuracy: 0.9496
 1000/11314 [=>............................] - ETA: 6s - loss: 0.3968 - accuracy: 0.9496
 1100/11314 [=>............................] - ETA: 6s - loss: 0.3966 - accuracy: 0.9495
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.3966 - accuracy: 0.9493
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.3964 - accuracy: 0.9494
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.3962 - accuracy: 0.9494
 1500/11314 [==>...........................] - ETA: 5s - loss: 0.3961 - accuracy: 0.9493
 1600/11314 [===>..........................] - ETA: 5s - loss: 0.3959 - accuracy: 0.9492
 1700/11314 [===>..........................] - ETA: 5s - loss: 0.3957 - accuracy: 0.9493
 1800/11314 [===>..........................] - ETA: 5s - loss: 0.3953 - accuracy: 0.9494
 1900/11314 [====>.........................] - ETA: 5s - loss: 0.3951 - accuracy: 0.9495
 2000/11314 [====>.........................] - ETA: 5s - loss: 0.3948 - accuracy: 0.9495
 2100/11314 [====>.........................] - ETA: 5s - loss: 0.3946 - accuracy: 0.9495
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.3946 - accuracy: 0.9495
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.3944 - accuracy: 0.9495
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.3942 - accuracy: 0.9494
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.3940 - accuracy: 0.9494
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.3938 - accuracy: 0.9494
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.3936 - accuracy: 0.9494
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.3934 - accuracy: 0.9494
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.3933 - accuracy: 0.9494
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.3932 - accuracy: 0.9494
 3100/11314 [=======>......................] - ETA: 4s - loss: 0.3930 - accuracy: 0.9494
 3200/11314 [=======>......................] - ETA: 4s - loss: 0.3928 - accuracy: 0.9494
 3300/11314 [=======>......................] - ETA: 4s - loss: 0.3927 - accuracy: 0.9494
 3400/11314 [========>.....................] - ETA: 4s - loss: 0.3925 - accuracy: 0.9494
 3500/11314 [========>.....................] - ETA: 4s - loss: 0.3924 - accuracy: 0.9494
 3600/11314 [========>.....................] - ETA: 4s - loss: 0.3921 - accuracy: 0.9494
 3700/11314 [========>.....................] - ETA: 4s - loss: 0.3920 - accuracy: 0.9494
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.3918 - accuracy: 0.9494
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.3916 - accuracy: 0.9494
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.3914 - accuracy: 0.9495
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.3912 - accuracy: 0.9495
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.3911 - accuracy: 0.9494
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.3909 - accuracy: 0.9494
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.3908 - accuracy: 0.9494
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.3906 - accuracy: 0.9494
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.3904 - accuracy: 0.9494
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.3902 - accuracy: 0.9494
 4800/11314 [===========>..................] - ETA: 3s - loss: 0.3900 - accuracy: 0.9495
 4900/11314 [===========>..................] - ETA: 3s - loss: 0.3899 - accuracy: 0.9495
 5000/11314 [============>.................] - ETA: 3s - loss: 0.3897 - accuracy: 0.9494
 5100/11314 [============>.................] - ETA: 3s - loss: 0.3895 - accuracy: 0.9494
 5200/11314 [============>.................] - ETA: 3s - loss: 0.3894 - accuracy: 0.9494
 5300/11314 [=============>................] - ETA: 3s - loss: 0.3893 - accuracy: 0.9494
 5400/11314 [=============>................] - ETA: 3s - loss: 0.3891 - accuracy: 0.9494
 5500/11314 [=============>................] - ETA: 3s - loss: 0.3889 - accuracy: 0.9494
 5600/11314 [=============>................] - ETA: 3s - loss: 0.3887 - accuracy: 0.9494
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.3885 - accuracy: 0.9495
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.3883 - accuracy: 0.9495
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.3880 - accuracy: 0.9495
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.3879 - accuracy: 0.9495
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.3877 - accuracy: 0.9495
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.3876 - accuracy: 0.9495
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.3874 - accuracy: 0.9495
 6400/11314 [===============>..............] - ETA: 2s - loss: 0.3872 - accuracy: 0.9495
 6500/11314 [================>.............] - ETA: 2s - loss: 0.3870 - accuracy: 0.9495
 6600/11314 [================>.............] - ETA: 2s - loss: 0.3869 - accuracy: 0.9495
 6700/11314 [================>.............] - ETA: 2s - loss: 0.3867 - accuracy: 0.9495
 6800/11314 [=================>............] - ETA: 2s - loss: 0.3865 - accuracy: 0.9495
 6900/11314 [=================>............] - ETA: 2s - loss: 0.3863 - accuracy: 0.9495
 7000/11314 [=================>............] - ETA: 2s - loss: 0.3862 - accuracy: 0.9495
 7100/11314 [=================>............] - ETA: 2s - loss: 0.3860 - accuracy: 0.9495
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.3858 - accuracy: 0.9495
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.3857 - accuracy: 0.9495
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.3855 - accuracy: 0.9495
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.3853 - accuracy: 0.9495
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.3851 - accuracy: 0.9495
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.3849 - accuracy: 0.9495
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.3848 - accuracy: 0.9495
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.3846 - accuracy: 0.9495
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.3845 - accuracy: 0.9495
 8100/11314 [====================>.........] - ETA: 1s - loss: 0.3843 - accuracy: 0.9495
 8200/11314 [====================>.........] - ETA: 1s - loss: 0.3841 - accuracy: 0.9495
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.3839 - accuracy: 0.9495
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.3838 - accuracy: 0.9495
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.3836 - accuracy: 0.9495
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.3834 - accuracy: 0.9495
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.3832 - accuracy: 0.9495
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.3830 - accuracy: 0.9495
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.3829 - accuracy: 0.9495
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.3827 - accuracy: 0.9495
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.3825 - accuracy: 0.9495
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.3823 - accuracy: 0.9495
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.3822 - accuracy: 0.9496
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.3820 - accuracy: 0.9495
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.3818 - accuracy: 0.9496
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.3816 - accuracy: 0.9496
 9700/11314 [========================>.....] - ETA: 0s - loss: 0.3814 - accuracy: 0.9496
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.3812 - accuracy: 0.9496
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.3811 - accuracy: 0.9496
10000/11314 [=========================>....] - ETA: 0s - loss: 0.3809 - accuracy: 0.9496
10100/11314 [=========================>....] - ETA: 0s - loss: 0.3808 - accuracy: 0.9496
10200/11314 [==========================>...] - ETA: 0s - loss: 0.3806 - accuracy: 0.9496
10300/11314 [==========================>...] - ETA: 0s - loss: 0.3805 - accuracy: 0.9496
10400/11314 [==========================>...] - ETA: 0s - loss: 0.3804 - accuracy: 0.9496
10500/11314 [==========================>...] - ETA: 0s - loss: 0.3802 - accuracy: 0.9496
10600/11314 [===========================>..] - ETA: 0s - loss: 0.3800 - accuracy: 0.9496
10700/11314 [===========================>..] - ETA: 0s - loss: 0.3799 - accuracy: 0.9496
10800/11314 [===========================>..] - ETA: 0s - loss: 0.3797 - accuracy: 0.9496
10900/11314 [===========================>..] - ETA: 0s - loss: 0.3796 - accuracy: 0.9496
11000/11314 [============================>.] - ETA: 0s - loss: 0.3794 - accuracy: 0.9496
11100/11314 [============================>.] - ETA: 0s - loss: 0.3792 - accuracy: 0.9496
11200/11314 [============================>.] - ETA: 0s - loss: 0.3791 - accuracy: 0.9496
11300/11314 [============================>.] - ETA: 0s - loss: 0.3789 - accuracy: 0.9496
11314/11314 [==============================] - 8s 721us/step - loss: 0.3789 - accuracy: 0.9496 - val_loss: 0.3601 - val_accuracy: 0.9496
Epoch 6/10

  100/11314 [..............................] - ETA: 6s - loss: 0.3590 - accuracy: 0.9495
  200/11314 [..............................] - ETA: 6s - loss: 0.3591 - accuracy: 0.9497
  300/11314 [..............................] - ETA: 6s - loss: 0.3594 - accuracy: 0.9493
  400/11314 [>.............................] - ETA: 6s - loss: 0.3596 - accuracy: 0.9492
  500/11314 [>.............................] - ETA: 6s - loss: 0.3597 - accuracy: 0.9491
  600/11314 [>.............................] - ETA: 6s - loss: 0.3602 - accuracy: 0.9489
  700/11314 [>.............................] - ETA: 6s - loss: 0.3602 - accuracy: 0.9489
  800/11314 [=>............................] - ETA: 6s - loss: 0.3598 - accuracy: 0.9489
  900/11314 [=>............................] - ETA: 6s - loss: 0.3596 - accuracy: 0.9489
 1000/11314 [=>............................] - ETA: 6s - loss: 0.3594 - accuracy: 0.9488
 1100/11314 [=>............................] - ETA: 6s - loss: 0.3591 - accuracy: 0.9490
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.3587 - accuracy: 0.9493
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.3582 - accuracy: 0.9496
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.3582 - accuracy: 0.9495
 1500/11314 [==>...........................] - ETA: 5s - loss: 0.3581 - accuracy: 0.9495
 1600/11314 [===>..........................] - ETA: 5s - loss: 0.3580 - accuracy: 0.9495
 1700/11314 [===>..........................] - ETA: 5s - loss: 0.3579 - accuracy: 0.9495
 1800/11314 [===>..........................] - ETA: 5s - loss: 0.3578 - accuracy: 0.9495
 1900/11314 [====>.........................] - ETA: 5s - loss: 0.3575 - accuracy: 0.9496
 2000/11314 [====>.........................] - ETA: 5s - loss: 0.3574 - accuracy: 0.9495
 2100/11314 [====>.........................] - ETA: 5s - loss: 0.3573 - accuracy: 0.9495
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.3573 - accuracy: 0.9495
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.3572 - accuracy: 0.9495
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.3570 - accuracy: 0.9495
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.3569 - accuracy: 0.9495
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.3567 - accuracy: 0.9495
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.3565 - accuracy: 0.9495
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.3563 - accuracy: 0.9495
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.3562 - accuracy: 0.9495
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.3561 - accuracy: 0.9495
 3100/11314 [=======>......................] - ETA: 4s - loss: 0.3559 - accuracy: 0.9495
 3200/11314 [=======>......................] - ETA: 4s - loss: 0.3557 - accuracy: 0.9495
 3300/11314 [=======>......................] - ETA: 4s - loss: 0.3556 - accuracy: 0.9495
 3400/11314 [========>.....................] - ETA: 4s - loss: 0.3555 - accuracy: 0.9495
 3500/11314 [========>.....................] - ETA: 4s - loss: 0.3554 - accuracy: 0.9495
 3600/11314 [========>.....................] - ETA: 4s - loss: 0.3552 - accuracy: 0.9495
 3700/11314 [========>.....................] - ETA: 4s - loss: 0.3550 - accuracy: 0.9495
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.3549 - accuracy: 0.9495
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.3548 - accuracy: 0.9495
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.3546 - accuracy: 0.9495
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.3544 - accuracy: 0.9495
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.3543 - accuracy: 0.9495
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.3541 - accuracy: 0.9495
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.3539 - accuracy: 0.9495
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.3538 - accuracy: 0.9495
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.3537 - accuracy: 0.9495
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.3536 - accuracy: 0.9495
 4800/11314 [===========>..................] - ETA: 3s - loss: 0.3534 - accuracy: 0.9495
 4900/11314 [===========>..................] - ETA: 3s - loss: 0.3532 - accuracy: 0.9495
 5000/11314 [============>.................] - ETA: 3s - loss: 0.3531 - accuracy: 0.9495
 5100/11314 [============>.................] - ETA: 3s - loss: 0.3529 - accuracy: 0.9496
 5200/11314 [============>.................] - ETA: 3s - loss: 0.3527 - accuracy: 0.9496
 5300/11314 [=============>................] - ETA: 3s - loss: 0.3525 - accuracy: 0.9496
 5400/11314 [=============>................] - ETA: 3s - loss: 0.3524 - accuracy: 0.9496
 5500/11314 [=============>................] - ETA: 3s - loss: 0.3523 - accuracy: 0.9496
 5600/11314 [=============>................] - ETA: 3s - loss: 0.3522 - accuracy: 0.9496
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.3520 - accuracy: 0.9495
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.3519 - accuracy: 0.9495
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.3517 - accuracy: 0.9495
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.3516 - accuracy: 0.9495
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.3514 - accuracy: 0.9496
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.3513 - accuracy: 0.9496
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.3511 - accuracy: 0.9496
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.3510 - accuracy: 0.9496
 6500/11314 [================>.............] - ETA: 2s - loss: 0.3508 - accuracy: 0.9496
 6600/11314 [================>.............] - ETA: 2s - loss: 0.3507 - accuracy: 0.9496
 6700/11314 [================>.............] - ETA: 2s - loss: 0.3506 - accuracy: 0.9495
 6800/11314 [=================>............] - ETA: 2s - loss: 0.3505 - accuracy: 0.9495
 6900/11314 [=================>............] - ETA: 2s - loss: 0.3504 - accuracy: 0.9495
 7000/11314 [=================>............] - ETA: 2s - loss: 0.3502 - accuracy: 0.9495
 7100/11314 [=================>............] - ETA: 2s - loss: 0.3501 - accuracy: 0.9495
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.3499 - accuracy: 0.9495
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.3498 - accuracy: 0.9495
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.3496 - accuracy: 0.9495
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.3495 - accuracy: 0.9495
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.3494 - accuracy: 0.9495
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.3493 - accuracy: 0.9495
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.3492 - accuracy: 0.9495
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.3490 - accuracy: 0.9495
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.3489 - accuracy: 0.9495
 8100/11314 [====================>.........] - ETA: 1s - loss: 0.3488 - accuracy: 0.9495
 8200/11314 [====================>.........] - ETA: 1s - loss: 0.3487 - accuracy: 0.9495
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.3485 - accuracy: 0.9495
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.3483 - accuracy: 0.9495
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.3482 - accuracy: 0.9495
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.3480 - accuracy: 0.9495
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.3479 - accuracy: 0.9495
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.3478 - accuracy: 0.9495
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.3476 - accuracy: 0.9495
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.3475 - accuracy: 0.9495
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.3473 - accuracy: 0.9495
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.3472 - accuracy: 0.9495
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.3471 - accuracy: 0.9495
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.3469 - accuracy: 0.9495
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.3468 - accuracy: 0.9495
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.3467 - accuracy: 0.9496
 9700/11314 [========================>.....] - ETA: 0s - loss: 0.3466 - accuracy: 0.9495
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.3464 - accuracy: 0.9496
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.3463 - accuracy: 0.9496
10000/11314 [=========================>....] - ETA: 0s - loss: 0.3461 - accuracy: 0.9496
10100/11314 [=========================>....] - ETA: 0s - loss: 0.3460 - accuracy: 0.9496
10200/11314 [==========================>...] - ETA: 0s - loss: 0.3459 - accuracy: 0.9496
10300/11314 [==========================>...] - ETA: 0s - loss: 0.3457 - accuracy: 0.9496
10400/11314 [==========================>...] - ETA: 0s - loss: 0.3456 - accuracy: 0.9496
10500/11314 [==========================>...] - ETA: 0s - loss: 0.3455 - accuracy: 0.9495
10600/11314 [===========================>..] - ETA: 0s - loss: 0.3454 - accuracy: 0.9495
10700/11314 [===========================>..] - ETA: 0s - loss: 0.3452 - accuracy: 0.9496
10800/11314 [===========================>..] - ETA: 0s - loss: 0.3451 - accuracy: 0.9496
10900/11314 [===========================>..] - ETA: 0s - loss: 0.3449 - accuracy: 0.9496
11000/11314 [============================>.] - ETA: 0s - loss: 0.3448 - accuracy: 0.9496
11100/11314 [============================>.] - ETA: 0s - loss: 0.3447 - accuracy: 0.9496
11200/11314 [============================>.] - ETA: 0s - loss: 0.3445 - accuracy: 0.9496
11300/11314 [============================>.] - ETA: 0s - loss: 0.3444 - accuracy: 0.9496
11314/11314 [==============================] - 8s 726us/step - loss: 0.3444 - accuracy: 0.9496 - val_loss: 0.3293 - val_accuracy: 0.9496
Epoch 7/10

  100/11314 [..............................] - ETA: 6s - loss: 0.3303 - accuracy: 0.9495
  200/11314 [..............................] - ETA: 6s - loss: 0.3303 - accuracy: 0.9489
  300/11314 [..............................] - ETA: 6s - loss: 0.3308 - accuracy: 0.9488
  400/11314 [>.............................] - ETA: 6s - loss: 0.3298 - accuracy: 0.9488
  500/11314 [>.............................] - ETA: 6s - loss: 0.3295 - accuracy: 0.9492
  600/11314 [>.............................] - ETA: 6s - loss: 0.3293 - accuracy: 0.9490
  700/11314 [>.............................] - ETA: 6s - loss: 0.3289 - accuracy: 0.9491
  800/11314 [=>............................] - ETA: 6s - loss: 0.3287 - accuracy: 0.9491
  900/11314 [=>............................] - ETA: 6s - loss: 0.3288 - accuracy: 0.9491
 1000/11314 [=>............................] - ETA: 6s - loss: 0.3287 - accuracy: 0.9491
 1100/11314 [=>............................] - ETA: 6s - loss: 0.3285 - accuracy: 0.9492
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.3282 - accuracy: 0.9493
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.3280 - accuracy: 0.9494
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.3279 - accuracy: 0.9494
 1500/11314 [==>...........................] - ETA: 5s - loss: 0.3279 - accuracy: 0.9493
 1600/11314 [===>..........................] - ETA: 5s - loss: 0.3277 - accuracy: 0.9493
 1700/11314 [===>..........................] - ETA: 5s - loss: 0.3274 - accuracy: 0.9495
 1800/11314 [===>..........................] - ETA: 5s - loss: 0.3274 - accuracy: 0.9494
 1900/11314 [====>.........................] - ETA: 5s - loss: 0.3273 - accuracy: 0.9494
 2000/11314 [====>.........................] - ETA: 5s - loss: 0.3270 - accuracy: 0.9494
 2100/11314 [====>.........................] - ETA: 5s - loss: 0.3270 - accuracy: 0.9494
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.3269 - accuracy: 0.9494
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.3266 - accuracy: 0.9495
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.3264 - accuracy: 0.9495
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.3262 - accuracy: 0.9496
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.3261 - accuracy: 0.9497
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.3261 - accuracy: 0.9496
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.3259 - accuracy: 0.9497
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.3258 - accuracy: 0.9497
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.3257 - accuracy: 0.9497
 3100/11314 [=======>......................] - ETA: 5s - loss: 0.3256 - accuracy: 0.9497
 3200/11314 [=======>......................] - ETA: 4s - loss: 0.3255 - accuracy: 0.9497
 3300/11314 [=======>......................] - ETA: 4s - loss: 0.3254 - accuracy: 0.9497
 3400/11314 [========>.....................] - ETA: 4s - loss: 0.3251 - accuracy: 0.9497
 3500/11314 [========>.....................] - ETA: 4s - loss: 0.3249 - accuracy: 0.9498
 3600/11314 [========>.....................] - ETA: 4s - loss: 0.3249 - accuracy: 0.9498
 3700/11314 [========>.....................] - ETA: 4s - loss: 0.3248 - accuracy: 0.9497
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.3247 - accuracy: 0.9498
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.3246 - accuracy: 0.9498
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.3244 - accuracy: 0.9498
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.3244 - accuracy: 0.9498
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.3243 - accuracy: 0.9498
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.3242 - accuracy: 0.9498
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.3241 - accuracy: 0.9498
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.3240 - accuracy: 0.9498
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.3239 - accuracy: 0.9497
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.3238 - accuracy: 0.9497
 4800/11314 [===========>..................] - ETA: 3s - loss: 0.3237 - accuracy: 0.9497
 4900/11314 [===========>..................] - ETA: 3s - loss: 0.3237 - accuracy: 0.9497
 5000/11314 [============>.................] - ETA: 3s - loss: 0.3235 - accuracy: 0.9497
 5100/11314 [============>.................] - ETA: 3s - loss: 0.3234 - accuracy: 0.9497
 5200/11314 [============>.................] - ETA: 3s - loss: 0.3232 - accuracy: 0.9497
 5300/11314 [=============>................] - ETA: 3s - loss: 0.3231 - accuracy: 0.9497
 5400/11314 [=============>................] - ETA: 3s - loss: 0.3231 - accuracy: 0.9497
 5500/11314 [=============>................] - ETA: 3s - loss: 0.3229 - accuracy: 0.9497
 5600/11314 [=============>................] - ETA: 3s - loss: 0.3229 - accuracy: 0.9496
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.3228 - accuracy: 0.9496
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.3227 - accuracy: 0.9496
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.3226 - accuracy: 0.9496
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.3225 - accuracy: 0.9496
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.3224 - accuracy: 0.9496
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.3223 - accuracy: 0.9496
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.3222 - accuracy: 0.9496
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.3221 - accuracy: 0.9496
 6500/11314 [================>.............] - ETA: 3s - loss: 0.3220 - accuracy: 0.9496
 6600/11314 [================>.............] - ETA: 2s - loss: 0.3219 - accuracy: 0.9495
 6700/11314 [================>.............] - ETA: 2s - loss: 0.3218 - accuracy: 0.9496
 6800/11314 [=================>............] - ETA: 2s - loss: 0.3217 - accuracy: 0.9495
 6900/11314 [=================>............] - ETA: 2s - loss: 0.3216 - accuracy: 0.9495
 7000/11314 [=================>............] - ETA: 2s - loss: 0.3214 - accuracy: 0.9495
 7100/11314 [=================>............] - ETA: 2s - loss: 0.3213 - accuracy: 0.9495
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.3212 - accuracy: 0.9495
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.3211 - accuracy: 0.9496
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.3210 - accuracy: 0.9495
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.3209 - accuracy: 0.9495
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.3208 - accuracy: 0.9495
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.3207 - accuracy: 0.9495
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.3206 - accuracy: 0.9495
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.3205 - accuracy: 0.9495
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.3204 - accuracy: 0.9495
 8100/11314 [====================>.........] - ETA: 1s - loss: 0.3203 - accuracy: 0.9495
 8200/11314 [====================>.........] - ETA: 1s - loss: 0.3202 - accuracy: 0.9495
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.3201 - accuracy: 0.9496
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.3199 - accuracy: 0.9496
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.3198 - accuracy: 0.9496
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.3197 - accuracy: 0.9496
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.3196 - accuracy: 0.9496
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.3195 - accuracy: 0.9496
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.3194 - accuracy: 0.9496
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.3193 - accuracy: 0.9495
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.3192 - accuracy: 0.9495
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.3191 - accuracy: 0.9496
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.3190 - accuracy: 0.9495
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.3189 - accuracy: 0.9495
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.3187 - accuracy: 0.9496
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.3186 - accuracy: 0.9496
 9700/11314 [========================>.....] - ETA: 0s - loss: 0.3185 - accuracy: 0.9496
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.3184 - accuracy: 0.9496
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.3182 - accuracy: 0.9496
10000/11314 [=========================>....] - ETA: 0s - loss: 0.3181 - accuracy: 0.9496
10100/11314 [=========================>....] - ETA: 0s - loss: 0.3180 - accuracy: 0.9496
10200/11314 [==========================>...] - ETA: 0s - loss: 0.3179 - accuracy: 0.9496
10300/11314 [==========================>...] - ETA: 0s - loss: 0.3178 - accuracy: 0.9496
10400/11314 [==========================>...] - ETA: 0s - loss: 0.3177 - accuracy: 0.9496
10500/11314 [==========================>...] - ETA: 0s - loss: 0.3176 - accuracy: 0.9496
10600/11314 [===========================>..] - ETA: 0s - loss: 0.3175 - accuracy: 0.9496
10700/11314 [===========================>..] - ETA: 0s - loss: 0.3174 - accuracy: 0.9496
10800/11314 [===========================>..] - ETA: 0s - loss: 0.3172 - accuracy: 0.9496
10900/11314 [===========================>..] - ETA: 0s - loss: 0.3172 - accuracy: 0.9496
11000/11314 [============================>.] - ETA: 0s - loss: 0.3171 - accuracy: 0.9496
11100/11314 [============================>.] - ETA: 0s - loss: 0.3170 - accuracy: 0.9496
11200/11314 [============================>.] - ETA: 0s - loss: 0.3168 - accuracy: 0.9496
11300/11314 [============================>.] - ETA: 0s - loss: 0.3167 - accuracy: 0.9496
11314/11314 [==============================] - 8s 728us/step - loss: 0.3167 - accuracy: 0.9496 - val_loss: 0.3047 - val_accuracy: 0.9496
Epoch 8/10

  100/11314 [..............................] - ETA: 6s - loss: 0.3063 - accuracy: 0.9484
  200/11314 [..............................] - ETA: 6s - loss: 0.3056 - accuracy: 0.9489
  300/11314 [..............................] - ETA: 6s - loss: 0.3056 - accuracy: 0.9491
  400/11314 [>.............................] - ETA: 6s - loss: 0.3048 - accuracy: 0.9492
  500/11314 [>.............................] - ETA: 6s - loss: 0.3048 - accuracy: 0.9493
  600/11314 [>.............................] - ETA: 6s - loss: 0.3048 - accuracy: 0.9495
  700/11314 [>.............................] - ETA: 6s - loss: 0.3045 - accuracy: 0.9495
  800/11314 [=>............................] - ETA: 6s - loss: 0.3046 - accuracy: 0.9493
  900/11314 [=>............................] - ETA: 6s - loss: 0.3048 - accuracy: 0.9493
 1000/11314 [=>............................] - ETA: 6s - loss: 0.3042 - accuracy: 0.9495
 1100/11314 [=>............................] - ETA: 6s - loss: 0.3039 - accuracy: 0.9495
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.3037 - accuracy: 0.9495
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.3036 - accuracy: 0.9496
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.3035 - accuracy: 0.9495
 1500/11314 [==>...........................] - ETA: 5s - loss: 0.3036 - accuracy: 0.9495
 1600/11314 [===>..........................] - ETA: 5s - loss: 0.3032 - accuracy: 0.9496
 1700/11314 [===>..........................] - ETA: 5s - loss: 0.3031 - accuracy: 0.9497
 1800/11314 [===>..........................] - ETA: 5s - loss: 0.3029 - accuracy: 0.9497
 1900/11314 [====>.........................] - ETA: 5s - loss: 0.3027 - accuracy: 0.9497
 2000/11314 [====>.........................] - ETA: 5s - loss: 0.3026 - accuracy: 0.9497
 2100/11314 [====>.........................] - ETA: 5s - loss: 0.3026 - accuracy: 0.9496
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.3024 - accuracy: 0.9496
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.3023 - accuracy: 0.9497
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.3022 - accuracy: 0.9497
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.3022 - accuracy: 0.9497
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.3020 - accuracy: 0.9497
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.3020 - accuracy: 0.9497
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.3019 - accuracy: 0.9497
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.3019 - accuracy: 0.9496
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.3018 - accuracy: 0.9496
 3100/11314 [=======>......................] - ETA: 4s - loss: 0.3017 - accuracy: 0.9496
 3200/11314 [=======>......................] - ETA: 4s - loss: 0.3014 - accuracy: 0.9497
 3300/11314 [=======>......................] - ETA: 4s - loss: 0.3013 - accuracy: 0.9498
 3400/11314 [========>.....................] - ETA: 4s - loss: 0.3011 - accuracy: 0.9498
 3500/11314 [========>.....................] - ETA: 4s - loss: 0.3011 - accuracy: 0.9498
 3600/11314 [========>.....................] - ETA: 4s - loss: 0.3010 - accuracy: 0.9498
 3700/11314 [========>.....................] - ETA: 4s - loss: 0.3009 - accuracy: 0.9498
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.3008 - accuracy: 0.9498
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.3008 - accuracy: 0.9497
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.3007 - accuracy: 0.9498
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.3007 - accuracy: 0.9497
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.3006 - accuracy: 0.9497
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.3005 - accuracy: 0.9497
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.3004 - accuracy: 0.9498
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.3003 - accuracy: 0.9498
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.3002 - accuracy: 0.9498
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.3001 - accuracy: 0.9497
 4800/11314 [===========>..................] - ETA: 3s - loss: 0.3001 - accuracy: 0.9497
 4900/11314 [===========>..................] - ETA: 3s - loss: 0.3000 - accuracy: 0.9497
 5000/11314 [============>.................] - ETA: 3s - loss: 0.2998 - accuracy: 0.9498
 5100/11314 [============>.................] - ETA: 3s - loss: 0.2997 - accuracy: 0.9497
 5200/11314 [============>.................] - ETA: 3s - loss: 0.2997 - accuracy: 0.9497
 5300/11314 [=============>................] - ETA: 3s - loss: 0.2997 - accuracy: 0.9497
 5400/11314 [=============>................] - ETA: 3s - loss: 0.2996 - accuracy: 0.9497
 5500/11314 [=============>................] - ETA: 3s - loss: 0.2995 - accuracy: 0.9497
 5600/11314 [=============>................] - ETA: 3s - loss: 0.2994 - accuracy: 0.9497
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.2993 - accuracy: 0.9497
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.2992 - accuracy: 0.9497
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.2992 - accuracy: 0.9497
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.2991 - accuracy: 0.9497
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.2990 - accuracy: 0.9497
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.2989 - accuracy: 0.9497
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.2988 - accuracy: 0.9497
 6400/11314 [===============>..............] - ETA: 2s - loss: 0.2987 - accuracy: 0.9497
 6500/11314 [================>.............] - ETA: 2s - loss: 0.2986 - accuracy: 0.9497
 6600/11314 [================>.............] - ETA: 2s - loss: 0.2985 - accuracy: 0.9497
 6700/11314 [================>.............] - ETA: 2s - loss: 0.2984 - accuracy: 0.9497
 6800/11314 [=================>............] - ETA: 2s - loss: 0.2983 - accuracy: 0.9497
 6900/11314 [=================>............] - ETA: 2s - loss: 0.2982 - accuracy: 0.9497
 7000/11314 [=================>............] - ETA: 2s - loss: 0.2981 - accuracy: 0.9497
 7100/11314 [=================>............] - ETA: 2s - loss: 0.2980 - accuracy: 0.9497
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.2979 - accuracy: 0.9497
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.2978 - accuracy: 0.9497
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.2977 - accuracy: 0.9497
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.2976 - accuracy: 0.9497
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.2975 - accuracy: 0.9497
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.2974 - accuracy: 0.9497
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.2973 - accuracy: 0.9497
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.2972 - accuracy: 0.9497
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.2971 - accuracy: 0.9497
 8100/11314 [====================>.........] - ETA: 1s - loss: 0.2971 - accuracy: 0.9497
 8200/11314 [====================>.........] - ETA: 1s - loss: 0.2970 - accuracy: 0.9497
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.2969 - accuracy: 0.9497
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.2968 - accuracy: 0.9496
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.2967 - accuracy: 0.9497
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.2966 - accuracy: 0.9497
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.2966 - accuracy: 0.9496
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.2965 - accuracy: 0.9496
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.2964 - accuracy: 0.9496
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.2964 - accuracy: 0.9496
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.2963 - accuracy: 0.9496
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.2962 - accuracy: 0.9496
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.2962 - accuracy: 0.9496
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.2961 - accuracy: 0.9496
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.2960 - accuracy: 0.9496
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.2959 - accuracy: 0.9496
 9700/11314 [========================>.....] - ETA: 0s - loss: 0.2958 - accuracy: 0.9496
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.2958 - accuracy: 0.9496
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.2957 - accuracy: 0.9496
10000/11314 [=========================>....] - ETA: 0s - loss: 0.2956 - accuracy: 0.9496
10100/11314 [=========================>....] - ETA: 0s - loss: 0.2955 - accuracy: 0.9496
10200/11314 [==========================>...] - ETA: 0s - loss: 0.2954 - accuracy: 0.9496
10300/11314 [==========================>...] - ETA: 0s - loss: 0.2953 - accuracy: 0.9496
10400/11314 [==========================>...] - ETA: 0s - loss: 0.2952 - accuracy: 0.9496
10500/11314 [==========================>...] - ETA: 0s - loss: 0.2952 - accuracy: 0.9496
10600/11314 [===========================>..] - ETA: 0s - loss: 0.2951 - accuracy: 0.9496
10700/11314 [===========================>..] - ETA: 0s - loss: 0.2951 - accuracy: 0.9496
10800/11314 [===========================>..] - ETA: 0s - loss: 0.2950 - accuracy: 0.9496
10900/11314 [===========================>..] - ETA: 0s - loss: 0.2949 - accuracy: 0.9496
11000/11314 [============================>.] - ETA: 0s - loss: 0.2948 - accuracy: 0.9496
11100/11314 [============================>.] - ETA: 0s - loss: 0.2947 - accuracy: 0.9496
11200/11314 [============================>.] - ETA: 0s - loss: 0.2946 - accuracy: 0.9496
11300/11314 [============================>.] - ETA: 0s - loss: 0.2945 - accuracy: 0.9496
11314/11314 [==============================] - 8s 718us/step - loss: 0.2945 - accuracy: 0.9496 - val_loss: 0.2848 - val_accuracy: 0.9496
Epoch 9/10

  100/11314 [..............................] - ETA: 6s - loss: 0.2855 - accuracy: 0.9505
  200/11314 [..............................] - ETA: 6s - loss: 0.2840 - accuracy: 0.9508
  300/11314 [..............................] - ETA: 6s - loss: 0.2851 - accuracy: 0.9502
  400/11314 [>.............................] - ETA: 6s - loss: 0.2857 - accuracy: 0.9497
  500/11314 [>.............................] - ETA: 6s - loss: 0.2846 - accuracy: 0.9499
  600/11314 [>.............................] - ETA: 6s - loss: 0.2843 - accuracy: 0.9498
  700/11314 [>.............................] - ETA: 6s - loss: 0.2838 - accuracy: 0.9500
  800/11314 [=>............................] - ETA: 6s - loss: 0.2834 - accuracy: 0.9501
  900/11314 [=>............................] - ETA: 6s - loss: 0.2836 - accuracy: 0.9499
 1000/11314 [=>............................] - ETA: 6s - loss: 0.2839 - accuracy: 0.9497
 1100/11314 [=>............................] - ETA: 6s - loss: 0.2838 - accuracy: 0.9498
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.2837 - accuracy: 0.9498
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.2837 - accuracy: 0.9497
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.2837 - accuracy: 0.9497
 1500/11314 [==>...........................] - ETA: 5s - loss: 0.2833 - accuracy: 0.9498
 1600/11314 [===>..........................] - ETA: 5s - loss: 0.2832 - accuracy: 0.9498
 1700/11314 [===>..........................] - ETA: 5s - loss: 0.2832 - accuracy: 0.9498
 1800/11314 [===>..........................] - ETA: 5s - loss: 0.2831 - accuracy: 0.9498
 1900/11314 [====>.........................] - ETA: 5s - loss: 0.2830 - accuracy: 0.9498
 2000/11314 [====>.........................] - ETA: 5s - loss: 0.2830 - accuracy: 0.9498
 2100/11314 [====>.........................] - ETA: 5s - loss: 0.2829 - accuracy: 0.9497
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.2828 - accuracy: 0.9498
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.2828 - accuracy: 0.9497
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.2827 - accuracy: 0.9497
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.2826 - accuracy: 0.9496
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.2826 - accuracy: 0.9496
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.2825 - accuracy: 0.9496
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.2824 - accuracy: 0.9497
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.2823 - accuracy: 0.9497
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.2823 - accuracy: 0.9496
 3100/11314 [=======>......................] - ETA: 4s - loss: 0.2823 - accuracy: 0.9496
 3200/11314 [=======>......................] - ETA: 4s - loss: 0.2822 - accuracy: 0.9497
 3300/11314 [=======>......................] - ETA: 4s - loss: 0.2821 - accuracy: 0.9497
 3400/11314 [========>.....................] - ETA: 4s - loss: 0.2820 - accuracy: 0.9497
 3500/11314 [========>.....................] - ETA: 4s - loss: 0.2820 - accuracy: 0.9496
 3600/11314 [========>.....................] - ETA: 4s - loss: 0.2820 - accuracy: 0.9496
 3700/11314 [========>.....................] - ETA: 4s - loss: 0.2819 - accuracy: 0.9496
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.2818 - accuracy: 0.9497
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.2817 - accuracy: 0.9497
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.2816 - accuracy: 0.9496
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.2815 - accuracy: 0.9496
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.2814 - accuracy: 0.9496
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.2814 - accuracy: 0.9496
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.2813 - accuracy: 0.9497
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.2812 - accuracy: 0.9496
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.2812 - accuracy: 0.9496
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.2811 - accuracy: 0.9496
 4800/11314 [===========>..................] - ETA: 3s - loss: 0.2811 - accuracy: 0.9496
 4900/11314 [===========>..................] - ETA: 3s - loss: 0.2811 - accuracy: 0.9496
 5000/11314 [============>.................] - ETA: 3s - loss: 0.2810 - accuracy: 0.9496
 5100/11314 [============>.................] - ETA: 3s - loss: 0.2808 - accuracy: 0.9496
 5200/11314 [============>.................] - ETA: 3s - loss: 0.2808 - accuracy: 0.9496
 5300/11314 [=============>................] - ETA: 3s - loss: 0.2807 - accuracy: 0.9496
 5400/11314 [=============>................] - ETA: 3s - loss: 0.2806 - accuracy: 0.9496
 5500/11314 [=============>................] - ETA: 3s - loss: 0.2806 - accuracy: 0.9496
 5600/11314 [=============>................] - ETA: 3s - loss: 0.2806 - accuracy: 0.9496
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.2804 - accuracy: 0.9496
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.2803 - accuracy: 0.9496
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.2803 - accuracy: 0.9496
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.2802 - accuracy: 0.9496
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.2801 - accuracy: 0.9496
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.2800 - accuracy: 0.9497
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.2800 - accuracy: 0.9496
 6400/11314 [===============>..............] - ETA: 2s - loss: 0.2799 - accuracy: 0.9496
 6500/11314 [================>.............] - ETA: 2s - loss: 0.2798 - accuracy: 0.9497
 6600/11314 [================>.............] - ETA: 2s - loss: 0.2797 - accuracy: 0.9497
 6700/11314 [================>.............] - ETA: 2s - loss: 0.2796 - accuracy: 0.9497
 6800/11314 [=================>............] - ETA: 2s - loss: 0.2795 - accuracy: 0.9497
 6900/11314 [=================>............] - ETA: 2s - loss: 0.2795 - accuracy: 0.9497
 7000/11314 [=================>............] - ETA: 2s - loss: 0.2794 - accuracy: 0.9496
 7100/11314 [=================>............] - ETA: 2s - loss: 0.2794 - accuracy: 0.9497
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.2793 - accuracy: 0.9496
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.2792 - accuracy: 0.9497
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.2792 - accuracy: 0.9496
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.2791 - accuracy: 0.9496
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.2791 - accuracy: 0.9496
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.2790 - accuracy: 0.9497
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.2789 - accuracy: 0.9497
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.2789 - accuracy: 0.9496
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.2788 - accuracy: 0.9496
 8100/11314 [====================>.........] - ETA: 1s - loss: 0.2788 - accuracy: 0.9496
 8200/11314 [====================>.........] - ETA: 1s - loss: 0.2787 - accuracy: 0.9496
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.2787 - accuracy: 0.9496
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.2786 - accuracy: 0.9496
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.2785 - accuracy: 0.9497
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.2784 - accuracy: 0.9497
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.2784 - accuracy: 0.9496
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.2783 - accuracy: 0.9496
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.2782 - accuracy: 0.9496
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.2782 - accuracy: 0.9496
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.2781 - accuracy: 0.9496
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.2780 - accuracy: 0.9496
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.2780 - accuracy: 0.9496
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.2779 - accuracy: 0.9497
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.2778 - accuracy: 0.9496
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.2778 - accuracy: 0.9496
 9700/11314 [========================>.....] - ETA: 0s - loss: 0.2777 - accuracy: 0.9496
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.2776 - accuracy: 0.9496
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.2776 - accuracy: 0.9496
10000/11314 [=========================>....] - ETA: 0s - loss: 0.2775 - accuracy: 0.9496
10100/11314 [=========================>....] - ETA: 0s - loss: 0.2775 - accuracy: 0.9496
10200/11314 [==========================>...] - ETA: 0s - loss: 0.2774 - accuracy: 0.9496
10300/11314 [==========================>...] - ETA: 0s - loss: 0.2773 - accuracy: 0.9496
10400/11314 [==========================>...] - ETA: 0s - loss: 0.2773 - accuracy: 0.9496
10500/11314 [==========================>...] - ETA: 0s - loss: 0.2772 - accuracy: 0.9496
10600/11314 [===========================>..] - ETA: 0s - loss: 0.2771 - accuracy: 0.9496
10700/11314 [===========================>..] - ETA: 0s - loss: 0.2771 - accuracy: 0.9496
10800/11314 [===========================>..] - ETA: 0s - loss: 0.2770 - accuracy: 0.9496
10900/11314 [===========================>..] - ETA: 0s - loss: 0.2769 - accuracy: 0.9496
11000/11314 [============================>.] - ETA: 0s - loss: 0.2769 - accuracy: 0.9496
11100/11314 [============================>.] - ETA: 0s - loss: 0.2768 - accuracy: 0.9496
11200/11314 [============================>.] - ETA: 0s - loss: 0.2767 - accuracy: 0.9496
11300/11314 [============================>.] - ETA: 0s - loss: 0.2766 - accuracy: 0.9496
11314/11314 [==============================] - 8s 720us/step - loss: 0.2766 - accuracy: 0.9496 - val_loss: 0.2688 - val_accuracy: 0.9496
Epoch 10/10

  100/11314 [..............................] - ETA: 6s - loss: 0.2679 - accuracy: 0.9495
  200/11314 [..............................] - ETA: 6s - loss: 0.2700 - accuracy: 0.9489
  300/11314 [..............................] - ETA: 6s - loss: 0.2698 - accuracy: 0.9489
  400/11314 [>.............................] - ETA: 6s - loss: 0.2684 - accuracy: 0.9497
  500/11314 [>.............................] - ETA: 6s - loss: 0.2691 - accuracy: 0.9494
  600/11314 [>.............................] - ETA: 6s - loss: 0.2692 - accuracy: 0.9492
  700/11314 [>.............................] - ETA: 6s - loss: 0.2687 - accuracy: 0.9496
  800/11314 [=>............................] - ETA: 6s - loss: 0.2685 - accuracy: 0.9496
  900/11314 [=>............................] - ETA: 6s - loss: 0.2688 - accuracy: 0.9495
 1000/11314 [=>............................] - ETA: 6s - loss: 0.2688 - accuracy: 0.9494
 1100/11314 [=>............................] - ETA: 6s - loss: 0.2689 - accuracy: 0.9493
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.2689 - accuracy: 0.9493
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.2686 - accuracy: 0.9495
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.2686 - accuracy: 0.9495
 1500/11314 [==>...........................] - ETA: 5s - loss: 0.2685 - accuracy: 0.9495
 1600/11314 [===>..........................] - ETA: 5s - loss: 0.2685 - accuracy: 0.9494
 1700/11314 [===>..........................] - ETA: 5s - loss: 0.2683 - accuracy: 0.9495
 1800/11314 [===>..........................] - ETA: 5s - loss: 0.2683 - accuracy: 0.9494
 1900/11314 [====>.........................] - ETA: 5s - loss: 0.2683 - accuracy: 0.9494
 2000/11314 [====>.........................] - ETA: 5s - loss: 0.2681 - accuracy: 0.9494
 2100/11314 [====>.........................] - ETA: 5s - loss: 0.2680 - accuracy: 0.9493
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.2680 - accuracy: 0.9493
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.2680 - accuracy: 0.9493
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.2680 - accuracy: 0.9493
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.2680 - accuracy: 0.9492
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.2679 - accuracy: 0.9492
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.2680 - accuracy: 0.9492
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.2681 - accuracy: 0.9491
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.2680 - accuracy: 0.9491
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.2679 - accuracy: 0.9491
 3100/11314 [=======>......................] - ETA: 4s - loss: 0.2676 - accuracy: 0.9492
 3200/11314 [=======>......................] - ETA: 4s - loss: 0.2676 - accuracy: 0.9492
 3300/11314 [=======>......................] - ETA: 4s - loss: 0.2674 - accuracy: 0.9493
 3400/11314 [========>.....................] - ETA: 4s - loss: 0.2672 - accuracy: 0.9493
 3500/11314 [========>.....................] - ETA: 4s - loss: 0.2673 - accuracy: 0.9493
 3600/11314 [========>.....................] - ETA: 4s - loss: 0.2672 - accuracy: 0.9493
 3700/11314 [========>.....................] - ETA: 4s - loss: 0.2671 - accuracy: 0.9493
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.2671 - accuracy: 0.9493
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.2669 - accuracy: 0.9493
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.2668 - accuracy: 0.9494
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.2667 - accuracy: 0.9494
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.2666 - accuracy: 0.9494
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.2665 - accuracy: 0.9494
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.2664 - accuracy: 0.9494
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.2663 - accuracy: 0.9495
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.2662 - accuracy: 0.9495
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.2661 - accuracy: 0.9495
 4800/11314 [===========>..................] - ETA: 3s - loss: 0.2661 - accuracy: 0.9495
 4900/11314 [===========>..................] - ETA: 3s - loss: 0.2660 - accuracy: 0.9495
 5000/11314 [============>.................] - ETA: 3s - loss: 0.2660 - accuracy: 0.9495
 5100/11314 [============>.................] - ETA: 3s - loss: 0.2660 - accuracy: 0.9495
 5200/11314 [============>.................] - ETA: 3s - loss: 0.2660 - accuracy: 0.9495
 5300/11314 [=============>................] - ETA: 3s - loss: 0.2659 - accuracy: 0.9495
 5400/11314 [=============>................] - ETA: 3s - loss: 0.2657 - accuracy: 0.9495
 5500/11314 [=============>................] - ETA: 3s - loss: 0.2657 - accuracy: 0.9495
 5600/11314 [=============>................] - ETA: 3s - loss: 0.2657 - accuracy: 0.9495
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.2656 - accuracy: 0.9495
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.2655 - accuracy: 0.9495
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.2655 - accuracy: 0.9495
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.2654 - accuracy: 0.9495
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.2654 - accuracy: 0.9495
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.2653 - accuracy: 0.9495
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.2652 - accuracy: 0.9495
 6400/11314 [===============>..............] - ETA: 2s - loss: 0.2651 - accuracy: 0.9495
 6500/11314 [================>.............] - ETA: 2s - loss: 0.2650 - accuracy: 0.9495
 6600/11314 [================>.............] - ETA: 2s - loss: 0.2649 - accuracy: 0.9496
 6700/11314 [================>.............] - ETA: 2s - loss: 0.2648 - accuracy: 0.9496
 6800/11314 [=================>............] - ETA: 2s - loss: 0.2647 - accuracy: 0.9496
 6900/11314 [=================>............] - ETA: 2s - loss: 0.2647 - accuracy: 0.9495
 7000/11314 [=================>............] - ETA: 2s - loss: 0.2646 - accuracy: 0.9495
 7100/11314 [=================>............] - ETA: 2s - loss: 0.2646 - accuracy: 0.9496
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.2645 - accuracy: 0.9496
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.2644 - accuracy: 0.9496
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.2644 - accuracy: 0.9496
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.2643 - accuracy: 0.9496
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.2643 - accuracy: 0.9496
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.2642 - accuracy: 0.9496
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.2641 - accuracy: 0.9496
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.2641 - accuracy: 0.9496
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.2640 - accuracy: 0.9496
 8100/11314 [====================>.........] - ETA: 1s - loss: 0.2640 - accuracy: 0.9496
 8200/11314 [====================>.........] - ETA: 1s - loss: 0.2639 - accuracy: 0.9496
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.2639 - accuracy: 0.9496
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.2638 - accuracy: 0.9496
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.2638 - accuracy: 0.9496
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.2637 - accuracy: 0.9496
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.2636 - accuracy: 0.9496
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.2636 - accuracy: 0.9496
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.2635 - accuracy: 0.9496
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.2634 - accuracy: 0.9496
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.2634 - accuracy: 0.9496
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.2633 - accuracy: 0.9496
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.2633 - accuracy: 0.9496
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.2632 - accuracy: 0.9496
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.2632 - accuracy: 0.9496
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.2631 - accuracy: 0.9496
 9700/11314 [========================>.....] - ETA: 0s - loss: 0.2631 - accuracy: 0.9496
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.2630 - accuracy: 0.9496
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.2629 - accuracy: 0.9496
10000/11314 [=========================>....] - ETA: 0s - loss: 0.2629 - accuracy: 0.9496
10100/11314 [=========================>....] - ETA: 0s - loss: 0.2629 - accuracy: 0.9496
10200/11314 [==========================>...] - ETA: 0s - loss: 0.2629 - accuracy: 0.9496
10300/11314 [==========================>...] - ETA: 0s - loss: 0.2628 - accuracy: 0.9496
10400/11314 [==========================>...] - ETA: 0s - loss: 0.2627 - accuracy: 0.9496
10500/11314 [==========================>...] - ETA: 0s - loss: 0.2627 - accuracy: 0.9496
10600/11314 [===========================>..] - ETA: 0s - loss: 0.2626 - accuracy: 0.9496
10700/11314 [===========================>..] - ETA: 0s - loss: 0.2626 - accuracy: 0.9496
10800/11314 [===========================>..] - ETA: 0s - loss: 0.2625 - accuracy: 0.9496
10900/11314 [===========================>..] - ETA: 0s - loss: 0.2624 - accuracy: 0.9496
11000/11314 [============================>.] - ETA: 0s - loss: 0.2624 - accuracy: 0.9496
11100/11314 [============================>.] - ETA: 0s - loss: 0.2623 - accuracy: 0.9496
11200/11314 [============================>.] - ETA: 0s - loss: 0.2622 - accuracy: 0.9496
11300/11314 [============================>.] - ETA: 0s - loss: 0.2622 - accuracy: 0.9496
11314/11314 [==============================] - 8s 718us/step - loss: 0.2622 - accuracy: 0.9496 - val_loss: 0.2558 - val_accuracy: 0.9496
	=====> Test the model: model.predict()
	Dataset: TWENTY_NEWS_GROUPS
	Algorithm: Deep Learning using Keras 2 (KerasDL2)
	Loss: 0.2558
	Training accuracy score: 94.96%
	Test Accuracy: 94.96%
	Training Time: 82.9633
	Test Time: 2.8831


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
	It took 24.363157272338867 seconds


Applying NLTK feature extraction: X_test
	==> Replace unused UNICODE characters. 
	==> Text in lower case. 
	==> Apply Lemmatize using WordNet's built-in morphy function. Returns the input word unchanged if it cannot be found in WordNet 
	==> Remove English stop words
	Done: NLTK feature extraction (replace unused UNICODE characters, text in lower case, apply Lemmatize using WordNet, remove English stop words)  finished!
	It took 23.813104152679443 seconds

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

  100/25000 [..............................] - ETA: 2:14 - loss: 0.6990 - accuracy: 0.5100
  200/25000 [..............................] - ETA: 1:14 - loss: 0.6963 - accuracy: 0.5200
  300/25000 [..............................] - ETA: 54s - loss: 0.6911 - accuracy: 0.5400 
  400/25000 [..............................] - ETA: 44s - loss: 0.6923 - accuracy: 0.5350
  500/25000 [..............................] - ETA: 38s - loss: 0.6925 - accuracy: 0.5340
  600/25000 [..............................] - ETA: 34s - loss: 0.6934 - accuracy: 0.5300
  700/25000 [..............................] - ETA: 31s - loss: 0.6954 - accuracy: 0.5214
  800/25000 [..............................] - ETA: 29s - loss: 0.6944 - accuracy: 0.5250
  900/25000 [>.............................] - ETA: 27s - loss: 0.6950 - accuracy: 0.5222
 1000/25000 [>.............................] - ETA: 26s - loss: 0.6943 - accuracy: 0.5250
 1100/25000 [>.............................] - ETA: 25s - loss: 0.6941 - accuracy: 0.5255
 1200/25000 [>.............................] - ETA: 24s - loss: 0.6946 - accuracy: 0.5225
 1300/25000 [>.............................] - ETA: 23s - loss: 0.6944 - accuracy: 0.5231
 1400/25000 [>.............................] - ETA: 22s - loss: 0.6955 - accuracy: 0.5179
 1500/25000 [>.............................] - ETA: 22s - loss: 0.6958 - accuracy: 0.5160
 1600/25000 [>.............................] - ETA: 21s - loss: 0.6955 - accuracy: 0.5169
 1700/25000 [=>............................] - ETA: 21s - loss: 0.6956 - accuracy: 0.5159
 1800/25000 [=>............................] - ETA: 20s - loss: 0.6965 - accuracy: 0.5100
 1900/25000 [=>............................] - ETA: 20s - loss: 0.6965 - accuracy: 0.5089
 2000/25000 [=>............................] - ETA: 19s - loss: 0.6973 - accuracy: 0.5040
 2100/25000 [=>............................] - ETA: 19s - loss: 0.6975 - accuracy: 0.5014
 2200/25000 [=>............................] - ETA: 19s - loss: 0.6972 - accuracy: 0.5023
 2300/25000 [=>............................] - ETA: 18s - loss: 0.6970 - accuracy: 0.5030
 2400/25000 [=>............................] - ETA: 18s - loss: 0.6970 - accuracy: 0.5021
 2500/25000 [==>...........................] - ETA: 18s - loss: 0.6973 - accuracy: 0.4988
 2600/25000 [==>...........................] - ETA: 17s - loss: 0.6973 - accuracy: 0.4977
 2700/25000 [==>...........................] - ETA: 17s - loss: 0.6974 - accuracy: 0.4948
 2800/25000 [==>...........................] - ETA: 17s - loss: 0.6973 - accuracy: 0.4939
 2900/25000 [==>...........................] - ETA: 17s - loss: 0.6974 - accuracy: 0.4910
 3000/25000 [==>...........................] - ETA: 17s - loss: 0.6974 - accuracy: 0.4903
 3100/25000 [==>...........................] - ETA: 16s - loss: 0.6973 - accuracy: 0.4906
 3200/25000 [==>...........................] - ETA: 16s - loss: 0.6971 - accuracy: 0.4922
 3300/25000 [==>...........................] - ETA: 16s - loss: 0.6970 - accuracy: 0.4912
 3400/25000 [===>..........................] - ETA: 16s - loss: 0.6969 - accuracy: 0.4918
 3500/25000 [===>..........................] - ETA: 16s - loss: 0.6968 - accuracy: 0.4946
 3600/25000 [===>..........................] - ETA: 16s - loss: 0.6966 - accuracy: 0.4972
 3700/25000 [===>..........................] - ETA: 15s - loss: 0.6965 - accuracy: 0.4968
 3800/25000 [===>..........................] - ETA: 15s - loss: 0.6964 - accuracy: 0.4982
 3900/25000 [===>..........................] - ETA: 15s - loss: 0.6963 - accuracy: 0.5003
 4000/25000 [===>..........................] - ETA: 15s - loss: 0.6962 - accuracy: 0.5023
 4100/25000 [===>..........................] - ETA: 15s - loss: 0.6961 - accuracy: 0.5039
 4200/25000 [====>.........................] - ETA: 15s - loss: 0.6960 - accuracy: 0.5052
 4300/25000 [====>.........................] - ETA: 15s - loss: 0.6959 - accuracy: 0.5077
 4400/25000 [====>.........................] - ETA: 14s - loss: 0.6958 - accuracy: 0.5064
 4500/25000 [====>.........................] - ETA: 14s - loss: 0.6957 - accuracy: 0.5069
 4600/25000 [====>.........................] - ETA: 14s - loss: 0.6956 - accuracy: 0.5080
 4700/25000 [====>.........................] - ETA: 14s - loss: 0.6956 - accuracy: 0.5083
 4800/25000 [====>.........................] - ETA: 14s - loss: 0.6955 - accuracy: 0.5100
 4900/25000 [====>.........................] - ETA: 14s - loss: 0.6954 - accuracy: 0.5102
 5000/25000 [=====>........................] - ETA: 14s - loss: 0.6953 - accuracy: 0.5102
 5100/25000 [=====>........................] - ETA: 14s - loss: 0.6953 - accuracy: 0.5094
 5200/25000 [=====>........................] - ETA: 14s - loss: 0.6953 - accuracy: 0.5088
 5300/25000 [=====>........................] - ETA: 14s - loss: 0.6952 - accuracy: 0.5096
 5400/25000 [=====>........................] - ETA: 14s - loss: 0.6951 - accuracy: 0.5091
 5500/25000 [=====>........................] - ETA: 13s - loss: 0.6951 - accuracy: 0.5089
 5600/25000 [=====>........................] - ETA: 13s - loss: 0.6950 - accuracy: 0.5100
 5700/25000 [=====>........................] - ETA: 13s - loss: 0.6950 - accuracy: 0.5118
 5800/25000 [=====>........................] - ETA: 13s - loss: 0.6949 - accuracy: 0.5143
 5900/25000 [======>.......................] - ETA: 13s - loss: 0.6948 - accuracy: 0.5173
 6000/25000 [======>.......................] - ETA: 13s - loss: 0.6947 - accuracy: 0.5188
 6100/25000 [======>.......................] - ETA: 13s - loss: 0.6946 - accuracy: 0.5231
 6200/25000 [======>.......................] - ETA: 13s - loss: 0.6944 - accuracy: 0.5258
 6300/25000 [======>.......................] - ETA: 13s - loss: 0.6942 - accuracy: 0.5284
 6400/25000 [======>.......................] - ETA: 13s - loss: 0.6941 - accuracy: 0.5300
 6500/25000 [======>.......................] - ETA: 12s - loss: 0.6939 - accuracy: 0.5328
 6600/25000 [======>.......................] - ETA: 12s - loss: 0.6938 - accuracy: 0.5342
 6700/25000 [=======>......................] - ETA: 12s - loss: 0.6935 - accuracy: 0.5364
 6800/25000 [=======>......................] - ETA: 12s - loss: 0.6934 - accuracy: 0.5376
 6900/25000 [=======>......................] - ETA: 12s - loss: 0.6930 - accuracy: 0.5406
 7000/25000 [=======>......................] - ETA: 12s - loss: 0.6928 - accuracy: 0.5427
 7100/25000 [=======>......................] - ETA: 12s - loss: 0.6926 - accuracy: 0.5444
 7200/25000 [=======>......................] - ETA: 12s - loss: 0.6923 - accuracy: 0.5462
 7300/25000 [=======>......................] - ETA: 12s - loss: 0.6921 - accuracy: 0.5482
 7400/25000 [=======>......................] - ETA: 12s - loss: 0.6918 - accuracy: 0.5505
 7500/25000 [========>.....................] - ETA: 12s - loss: 0.6915 - accuracy: 0.5532
 7600/25000 [========>.....................] - ETA: 11s - loss: 0.6912 - accuracy: 0.5557
 7700/25000 [========>.....................] - ETA: 11s - loss: 0.6909 - accuracy: 0.5577
 7800/25000 [========>.....................] - ETA: 11s - loss: 0.6904 - accuracy: 0.5606
 7900/25000 [========>.....................] - ETA: 11s - loss: 0.6900 - accuracy: 0.5625
 8000/25000 [========>.....................] - ETA: 11s - loss: 0.6897 - accuracy: 0.5645
 8100/25000 [========>.....................] - ETA: 11s - loss: 0.6892 - accuracy: 0.5673
 8200/25000 [========>.....................] - ETA: 11s - loss: 0.6889 - accuracy: 0.5690
 8300/25000 [========>.....................] - ETA: 11s - loss: 0.6886 - accuracy: 0.5708
 8400/25000 [=========>....................] - ETA: 11s - loss: 0.6882 - accuracy: 0.5735
 8500/25000 [=========>....................] - ETA: 11s - loss: 0.6878 - accuracy: 0.5758
 8600/25000 [=========>....................] - ETA: 11s - loss: 0.6874 - accuracy: 0.5783
 8700/25000 [=========>....................] - ETA: 11s - loss: 0.6869 - accuracy: 0.5809
 8800/25000 [=========>....................] - ETA: 10s - loss: 0.6866 - accuracy: 0.5825
 8900/25000 [=========>....................] - ETA: 10s - loss: 0.6864 - accuracy: 0.5842
 9000/25000 [=========>....................] - ETA: 10s - loss: 0.6861 - accuracy: 0.5856
 9100/25000 [=========>....................] - ETA: 10s - loss: 0.6857 - accuracy: 0.5869
 9200/25000 [==========>...................] - ETA: 10s - loss: 0.6853 - accuracy: 0.5893
 9300/25000 [==========>...................] - ETA: 10s - loss: 0.6849 - accuracy: 0.5910
 9400/25000 [==========>...................] - ETA: 10s - loss: 0.6846 - accuracy: 0.5928
 9500/25000 [==========>...................] - ETA: 10s - loss: 0.6844 - accuracy: 0.5943
 9600/25000 [==========>...................] - ETA: 10s - loss: 0.6839 - accuracy: 0.5959
 9700/25000 [==========>...................] - ETA: 10s - loss: 0.6833 - accuracy: 0.5988
 9800/25000 [==========>...................] - ETA: 10s - loss: 0.6830 - accuracy: 0.6004
 9900/25000 [==========>...................] - ETA: 10s - loss: 0.6826 - accuracy: 0.6024
10000/25000 [===========>..................] - ETA: 10s - loss: 0.6822 - accuracy: 0.6045
10100/25000 [===========>..................] - ETA: 9s - loss: 0.6818 - accuracy: 0.6065 
10200/25000 [===========>..................] - ETA: 9s - loss: 0.6814 - accuracy: 0.6083
10300/25000 [===========>..................] - ETA: 9s - loss: 0.6810 - accuracy: 0.6104
10400/25000 [===========>..................] - ETA: 9s - loss: 0.6806 - accuracy: 0.6123
10500/25000 [===========>..................] - ETA: 9s - loss: 0.6801 - accuracy: 0.6142
10600/25000 [===========>..................] - ETA: 9s - loss: 0.6797 - accuracy: 0.6155
10700/25000 [===========>..................] - ETA: 9s - loss: 0.6792 - accuracy: 0.6176
10800/25000 [===========>..................] - ETA: 9s - loss: 0.6786 - accuracy: 0.6193
10900/25000 [============>.................] - ETA: 9s - loss: 0.6783 - accuracy: 0.6207
11000/25000 [============>.................] - ETA: 9s - loss: 0.6778 - accuracy: 0.6225
11100/25000 [============>.................] - ETA: 9s - loss: 0.6774 - accuracy: 0.6242
11200/25000 [============>.................] - ETA: 9s - loss: 0.6770 - accuracy: 0.6256
11300/25000 [============>.................] - ETA: 9s - loss: 0.6765 - accuracy: 0.6271
11400/25000 [============>.................] - ETA: 9s - loss: 0.6761 - accuracy: 0.6287
11500/25000 [============>.................] - ETA: 8s - loss: 0.6757 - accuracy: 0.6297
11600/25000 [============>.................] - ETA: 8s - loss: 0.6751 - accuracy: 0.6316
11700/25000 [=============>................] - ETA: 8s - loss: 0.6746 - accuracy: 0.6337
11800/25000 [=============>................] - ETA: 8s - loss: 0.6742 - accuracy: 0.6352
11900/25000 [=============>................] - ETA: 8s - loss: 0.6738 - accuracy: 0.6361
12000/25000 [=============>................] - ETA: 8s - loss: 0.6734 - accuracy: 0.6378
12100/25000 [=============>................] - ETA: 8s - loss: 0.6731 - accuracy: 0.6394
12200/25000 [=============>................] - ETA: 8s - loss: 0.6725 - accuracy: 0.6410
12300/25000 [=============>................] - ETA: 8s - loss: 0.6721 - accuracy: 0.6418
12400/25000 [=============>................] - ETA: 8s - loss: 0.6717 - accuracy: 0.6430
12500/25000 [==============>...............] - ETA: 8s - loss: 0.6715 - accuracy: 0.6442
12600/25000 [==============>...............] - ETA: 8s - loss: 0.6709 - accuracy: 0.6457
12700/25000 [==============>...............] - ETA: 8s - loss: 0.6707 - accuracy: 0.6463
12800/25000 [==============>...............] - ETA: 8s - loss: 0.6702 - accuracy: 0.6477
12900/25000 [==============>...............] - ETA: 7s - loss: 0.6696 - accuracy: 0.6498
13000/25000 [==============>...............] - ETA: 7s - loss: 0.6691 - accuracy: 0.6511
13100/25000 [==============>...............] - ETA: 7s - loss: 0.6689 - accuracy: 0.6521
13200/25000 [==============>...............] - ETA: 7s - loss: 0.6685 - accuracy: 0.6535
13300/25000 [==============>...............] - ETA: 7s - loss: 0.6682 - accuracy: 0.6544
13400/25000 [===============>..............] - ETA: 7s - loss: 0.6677 - accuracy: 0.6555
13500/25000 [===============>..............] - ETA: 7s - loss: 0.6673 - accuracy: 0.6564
13600/25000 [===============>..............] - ETA: 7s - loss: 0.6670 - accuracy: 0.6576
13700/25000 [===============>..............] - ETA: 7s - loss: 0.6667 - accuracy: 0.6584
13800/25000 [===============>..............] - ETA: 7s - loss: 0.6664 - accuracy: 0.6592
13900/25000 [===============>..............] - ETA: 7s - loss: 0.6660 - accuracy: 0.6603
14000/25000 [===============>..............] - ETA: 7s - loss: 0.6658 - accuracy: 0.6610
14100/25000 [===============>..............] - ETA: 7s - loss: 0.6657 - accuracy: 0.6615
14200/25000 [================>.............] - ETA: 7s - loss: 0.6654 - accuracy: 0.6623
14300/25000 [================>.............] - ETA: 6s - loss: 0.6650 - accuracy: 0.6636
14400/25000 [================>.............] - ETA: 6s - loss: 0.6646 - accuracy: 0.6647
14500/25000 [================>.............] - ETA: 6s - loss: 0.6643 - accuracy: 0.6657
14600/25000 [================>.............] - ETA: 6s - loss: 0.6638 - accuracy: 0.6667
14700/25000 [================>.............] - ETA: 6s - loss: 0.6635 - accuracy: 0.6674
14800/25000 [================>.............] - ETA: 6s - loss: 0.6634 - accuracy: 0.6678
14900/25000 [================>.............] - ETA: 6s - loss: 0.6631 - accuracy: 0.6686
15000/25000 [=================>............] - ETA: 6s - loss: 0.6629 - accuracy: 0.6695
15100/25000 [=================>............] - ETA: 6s - loss: 0.6628 - accuracy: 0.6691
15200/25000 [=================>............] - ETA: 6s - loss: 0.6630 - accuracy: 0.6686
15300/25000 [=================>............] - ETA: 6s - loss: 0.6632 - accuracy: 0.6675
15400/25000 [=================>............] - ETA: 6s - loss: 0.6633 - accuracy: 0.6667
15500/25000 [=================>............] - ETA: 6s - loss: 0.6633 - accuracy: 0.6663
15600/25000 [=================>............] - ETA: 6s - loss: 0.6634 - accuracy: 0.6653
15700/25000 [=================>............] - ETA: 6s - loss: 0.6635 - accuracy: 0.6644
15800/25000 [=================>............] - ETA: 5s - loss: 0.6634 - accuracy: 0.6644
15900/25000 [==================>...........] - ETA: 5s - loss: 0.6634 - accuracy: 0.6636
16000/25000 [==================>...........] - ETA: 5s - loss: 0.6635 - accuracy: 0.6631
16100/25000 [==================>...........] - ETA: 5s - loss: 0.6635 - accuracy: 0.6627
16200/25000 [==================>...........] - ETA: 5s - loss: 0.6634 - accuracy: 0.6623
16300/25000 [==================>...........] - ETA: 5s - loss: 0.6635 - accuracy: 0.6615
16400/25000 [==================>...........] - ETA: 5s - loss: 0.6636 - accuracy: 0.6612
16500/25000 [==================>...........] - ETA: 5s - loss: 0.6636 - accuracy: 0.6607
16600/25000 [==================>...........] - ETA: 5s - loss: 0.6636 - accuracy: 0.6606
16700/25000 [===================>..........] - ETA: 5s - loss: 0.6636 - accuracy: 0.6599
16800/25000 [===================>..........] - ETA: 5s - loss: 0.6637 - accuracy: 0.6595
16900/25000 [===================>..........] - ETA: 5s - loss: 0.6637 - accuracy: 0.6589
17000/25000 [===================>..........] - ETA: 5s - loss: 0.6637 - accuracy: 0.6586
17100/25000 [===================>..........] - ETA: 5s - loss: 0.6638 - accuracy: 0.6584
17200/25000 [===================>..........] - ETA: 5s - loss: 0.6639 - accuracy: 0.6581
17300/25000 [===================>..........] - ETA: 4s - loss: 0.6639 - accuracy: 0.6578
17400/25000 [===================>..........] - ETA: 4s - loss: 0.6639 - accuracy: 0.6578
17500/25000 [====================>.........] - ETA: 4s - loss: 0.6639 - accuracy: 0.6576
17600/25000 [====================>.........] - ETA: 4s - loss: 0.6640 - accuracy: 0.6576
17700/25000 [====================>.........] - ETA: 4s - loss: 0.6638 - accuracy: 0.6585
17800/25000 [====================>.........] - ETA: 4s - loss: 0.6638 - accuracy: 0.6594
17900/25000 [====================>.........] - ETA: 4s - loss: 0.6637 - accuracy: 0.6597
18000/25000 [====================>.........] - ETA: 4s - loss: 0.6636 - accuracy: 0.6599
18100/25000 [====================>.........] - ETA: 4s - loss: 0.6636 - accuracy: 0.6603
18200/25000 [====================>.........] - ETA: 4s - loss: 0.6634 - accuracy: 0.6607
18300/25000 [====================>.........] - ETA: 4s - loss: 0.6633 - accuracy: 0.6609
18400/25000 [=====================>........] - ETA: 4s - loss: 0.6634 - accuracy: 0.6609
18500/25000 [=====================>........] - ETA: 4s - loss: 0.6632 - accuracy: 0.6612
18600/25000 [=====================>........] - ETA: 4s - loss: 0.6629 - accuracy: 0.6620
18700/25000 [=====================>........] - ETA: 4s - loss: 0.6628 - accuracy: 0.6624
18800/25000 [=====================>........] - ETA: 3s - loss: 0.6628 - accuracy: 0.6623
18900/25000 [=====================>........] - ETA: 3s - loss: 0.6628 - accuracy: 0.6624
19000/25000 [=====================>........] - ETA: 3s - loss: 0.6628 - accuracy: 0.6626
19100/25000 [=====================>........] - ETA: 3s - loss: 0.6626 - accuracy: 0.6631
19200/25000 [======================>.......] - ETA: 3s - loss: 0.6625 - accuracy: 0.6633
19300/25000 [======================>.......] - ETA: 3s - loss: 0.6623 - accuracy: 0.6639
19400/25000 [======================>.......] - ETA: 3s - loss: 0.6622 - accuracy: 0.6641
19500/25000 [======================>.......] - ETA: 3s - loss: 0.6622 - accuracy: 0.6643
19600/25000 [======================>.......] - ETA: 3s - loss: 0.6621 - accuracy: 0.6646
19700/25000 [======================>.......] - ETA: 3s - loss: 0.6620 - accuracy: 0.6648
19800/25000 [======================>.......] - ETA: 3s - loss: 0.6619 - accuracy: 0.6653
19900/25000 [======================>.......] - ETA: 3s - loss: 0.6617 - accuracy: 0.6657
20000/25000 [=======================>......] - ETA: 3s - loss: 0.6617 - accuracy: 0.6658
20100/25000 [=======================>......] - ETA: 3s - loss: 0.6616 - accuracy: 0.6661
20200/25000 [=======================>......] - ETA: 3s - loss: 0.6615 - accuracy: 0.6660
20300/25000 [=======================>......] - ETA: 3s - loss: 0.6613 - accuracy: 0.6664
20400/25000 [=======================>......] - ETA: 2s - loss: 0.6613 - accuracy: 0.6663
20500/25000 [=======================>......] - ETA: 2s - loss: 0.6612 - accuracy: 0.6667
20600/25000 [=======================>......] - ETA: 2s - loss: 0.6612 - accuracy: 0.6668
20700/25000 [=======================>......] - ETA: 2s - loss: 0.6610 - accuracy: 0.6673
20800/25000 [=======================>......] - ETA: 2s - loss: 0.6608 - accuracy: 0.6680
20900/25000 [========================>.....] - ETA: 2s - loss: 0.6606 - accuracy: 0.6687
21000/25000 [========================>.....] - ETA: 2s - loss: 0.6606 - accuracy: 0.6690
21100/25000 [========================>.....] - ETA: 2s - loss: 0.6606 - accuracy: 0.6689
21200/25000 [========================>.....] - ETA: 2s - loss: 0.6604 - accuracy: 0.6692
21300/25000 [========================>.....] - ETA: 2s - loss: 0.6604 - accuracy: 0.6694
21400/25000 [========================>.....] - ETA: 2s - loss: 0.6603 - accuracy: 0.6700
21500/25000 [========================>.....] - ETA: 2s - loss: 0.6600 - accuracy: 0.6707
21600/25000 [========================>.....] - ETA: 2s - loss: 0.6599 - accuracy: 0.6711
21700/25000 [=========================>....] - ETA: 2s - loss: 0.6597 - accuracy: 0.6718
21800/25000 [=========================>....] - ETA: 2s - loss: 0.6596 - accuracy: 0.6720
21900/25000 [=========================>....] - ETA: 1s - loss: 0.6596 - accuracy: 0.6725
22000/25000 [=========================>....] - ETA: 1s - loss: 0.6594 - accuracy: 0.6730
22100/25000 [=========================>....] - ETA: 1s - loss: 0.6593 - accuracy: 0.6730
22200/25000 [=========================>....] - ETA: 1s - loss: 0.6592 - accuracy: 0.6733
22300/25000 [=========================>....] - ETA: 1s - loss: 0.6591 - accuracy: 0.6735
22400/25000 [=========================>....] - ETA: 1s - loss: 0.6590 - accuracy: 0.6738
22500/25000 [==========================>...] - ETA: 1s - loss: 0.6590 - accuracy: 0.6740
22600/25000 [==========================>...] - ETA: 1s - loss: 0.6590 - accuracy: 0.6742
22700/25000 [==========================>...] - ETA: 1s - loss: 0.6588 - accuracy: 0.6748
22800/25000 [==========================>...] - ETA: 1s - loss: 0.6587 - accuracy: 0.6747
22900/25000 [==========================>...] - ETA: 1s - loss: 0.6586 - accuracy: 0.6751
23000/25000 [==========================>...] - ETA: 1s - loss: 0.6586 - accuracy: 0.6753
23100/25000 [==========================>...] - ETA: 1s - loss: 0.6584 - accuracy: 0.6759
23200/25000 [==========================>...] - ETA: 1s - loss: 0.6583 - accuracy: 0.6762
23300/25000 [==========================>...] - ETA: 1s - loss: 0.6582 - accuracy: 0.6765
23400/25000 [===========================>..] - ETA: 1s - loss: 0.6581 - accuracy: 0.6771
23500/25000 [===========================>..] - ETA: 0s - loss: 0.6579 - accuracy: 0.6780
23600/25000 [===========================>..] - ETA: 0s - loss: 0.6577 - accuracy: 0.6786
23700/25000 [===========================>..] - ETA: 0s - loss: 0.6576 - accuracy: 0.6789
23800/25000 [===========================>..] - ETA: 0s - loss: 0.6575 - accuracy: 0.6795
23900/25000 [===========================>..] - ETA: 0s - loss: 0.6573 - accuracy: 0.6800
24000/25000 [===========================>..] - ETA: 0s - loss: 0.6571 - accuracy: 0.6805
24100/25000 [===========================>..] - ETA: 0s - loss: 0.6570 - accuracy: 0.6812
24200/25000 [============================>.] - ETA: 0s - loss: 0.6568 - accuracy: 0.6816
24300/25000 [============================>.] - ETA: 0s - loss: 0.6565 - accuracy: 0.6823
24400/25000 [============================>.] - ETA: 0s - loss: 0.6564 - accuracy: 0.6830
24500/25000 [============================>.] - ETA: 0s - loss: 0.6562 - accuracy: 0.6834
24600/25000 [============================>.] - ETA: 0s - loss: 0.6559 - accuracy: 0.6841
24700/25000 [============================>.] - ETA: 0s - loss: 0.6557 - accuracy: 0.6847
24800/25000 [============================>.] - ETA: 0s - loss: 0.6555 - accuracy: 0.6853
24900/25000 [============================>.] - ETA: 0s - loss: 0.6552 - accuracy: 0.6859
25000/25000 [==============================] - 20s 798us/step - loss: 0.6551 - accuracy: 0.6862 - val_loss: 0.6044 - val_accuracy: 0.8098
Epoch 2/10

  100/25000 [..............................] - ETA: 15s - loss: 0.6189 - accuracy: 0.8300
  200/25000 [..............................] - ETA: 15s - loss: 0.6115 - accuracy: 0.8250
  300/25000 [..............................] - ETA: 15s - loss: 0.6040 - accuracy: 0.8333
  400/25000 [..............................] - ETA: 14s - loss: 0.6024 - accuracy: 0.8250
  500/25000 [..............................] - ETA: 14s - loss: 0.5987 - accuracy: 0.8260
  600/25000 [..............................] - ETA: 14s - loss: 0.5959 - accuracy: 0.8283
  700/25000 [..............................] - ETA: 14s - loss: 0.5960 - accuracy: 0.8314
  800/25000 [..............................] - ETA: 14s - loss: 0.5957 - accuracy: 0.8375
  900/25000 [>.............................] - ETA: 14s - loss: 0.5940 - accuracy: 0.8456
 1000/25000 [>.............................] - ETA: 14s - loss: 0.5922 - accuracy: 0.8470
 1100/25000 [>.............................] - ETA: 14s - loss: 0.5930 - accuracy: 0.8482
 1200/25000 [>.............................] - ETA: 14s - loss: 0.5945 - accuracy: 0.8433
 1300/25000 [>.............................] - ETA: 14s - loss: 0.5936 - accuracy: 0.8462
 1400/25000 [>.............................] - ETA: 14s - loss: 0.5934 - accuracy: 0.8457
 1500/25000 [>.............................] - ETA: 14s - loss: 0.5910 - accuracy: 0.8493
 1600/25000 [>.............................] - ETA: 14s - loss: 0.5899 - accuracy: 0.8506
 1700/25000 [=>............................] - ETA: 14s - loss: 0.5892 - accuracy: 0.8506
 1800/25000 [=>............................] - ETA: 14s - loss: 0.5896 - accuracy: 0.8506
 1900/25000 [=>............................] - ETA: 14s - loss: 0.5891 - accuracy: 0.8521
 2000/25000 [=>............................] - ETA: 14s - loss: 0.5897 - accuracy: 0.8505
 2100/25000 [=>............................] - ETA: 13s - loss: 0.5896 - accuracy: 0.8514
 2200/25000 [=>............................] - ETA: 13s - loss: 0.5895 - accuracy: 0.8509
 2300/25000 [=>............................] - ETA: 13s - loss: 0.5889 - accuracy: 0.8504
 2400/25000 [=>............................] - ETA: 13s - loss: 0.5892 - accuracy: 0.8496
 2500/25000 [==>...........................] - ETA: 13s - loss: 0.5893 - accuracy: 0.8496
 2600/25000 [==>...........................] - ETA: 13s - loss: 0.5891 - accuracy: 0.8488
 2700/25000 [==>...........................] - ETA: 13s - loss: 0.5899 - accuracy: 0.8478
 2800/25000 [==>...........................] - ETA: 13s - loss: 0.5889 - accuracy: 0.8504
 2900/25000 [==>...........................] - ETA: 13s - loss: 0.5893 - accuracy: 0.8486
 3000/25000 [==>...........................] - ETA: 13s - loss: 0.5892 - accuracy: 0.8483
 3100/25000 [==>...........................] - ETA: 13s - loss: 0.5893 - accuracy: 0.8481
 3200/25000 [==>...........................] - ETA: 13s - loss: 0.5894 - accuracy: 0.8469
 3300/25000 [==>...........................] - ETA: 13s - loss: 0.5895 - accuracy: 0.8470
 3400/25000 [===>..........................] - ETA: 13s - loss: 0.5895 - accuracy: 0.8465
 3500/25000 [===>..........................] - ETA: 13s - loss: 0.5894 - accuracy: 0.8454
 3600/25000 [===>..........................] - ETA: 13s - loss: 0.5897 - accuracy: 0.8442
 3700/25000 [===>..........................] - ETA: 12s - loss: 0.5901 - accuracy: 0.8424
 3800/25000 [===>..........................] - ETA: 12s - loss: 0.5901 - accuracy: 0.8418
 3900/25000 [===>..........................] - ETA: 12s - loss: 0.5902 - accuracy: 0.8410
 4000/25000 [===>..........................] - ETA: 12s - loss: 0.5907 - accuracy: 0.8388
 4100/25000 [===>..........................] - ETA: 12s - loss: 0.5910 - accuracy: 0.8395
 4200/25000 [====>.........................] - ETA: 12s - loss: 0.5911 - accuracy: 0.8390
 4300/25000 [====>.........................] - ETA: 12s - loss: 0.5913 - accuracy: 0.8386
 4400/25000 [====>.........................] - ETA: 12s - loss: 0.5911 - accuracy: 0.8393
 4500/25000 [====>.........................] - ETA: 12s - loss: 0.5910 - accuracy: 0.8404
 4600/25000 [====>.........................] - ETA: 12s - loss: 0.5912 - accuracy: 0.8396
 4700/25000 [====>.........................] - ETA: 12s - loss: 0.5917 - accuracy: 0.8385
 4800/25000 [====>.........................] - ETA: 12s - loss: 0.5914 - accuracy: 0.8388
 4900/25000 [====>.........................] - ETA: 12s - loss: 0.5916 - accuracy: 0.8376
 5000/25000 [=====>........................] - ETA: 12s - loss: 0.5913 - accuracy: 0.8384
 5100/25000 [=====>........................] - ETA: 12s - loss: 0.5910 - accuracy: 0.8384
 5200/25000 [=====>........................] - ETA: 12s - loss: 0.5910 - accuracy: 0.8390
 5300/25000 [=====>........................] - ETA: 12s - loss: 0.5909 - accuracy: 0.8389
 5400/25000 [=====>........................] - ETA: 11s - loss: 0.5909 - accuracy: 0.8389
 5500/25000 [=====>........................] - ETA: 11s - loss: 0.5908 - accuracy: 0.8389
 5600/25000 [=====>........................] - ETA: 11s - loss: 0.5906 - accuracy: 0.8393
 5700/25000 [=====>........................] - ETA: 11s - loss: 0.5906 - accuracy: 0.8391
 5800/25000 [=====>........................] - ETA: 11s - loss: 0.5903 - accuracy: 0.8393
 5900/25000 [======>.......................] - ETA: 11s - loss: 0.5903 - accuracy: 0.8390
 6000/25000 [======>.......................] - ETA: 11s - loss: 0.5905 - accuracy: 0.8380
 6100/25000 [======>.......................] - ETA: 11s - loss: 0.5902 - accuracy: 0.8390
 6200/25000 [======>.......................] - ETA: 11s - loss: 0.5902 - accuracy: 0.8389
 6300/25000 [======>.......................] - ETA: 11s - loss: 0.5902 - accuracy: 0.8390
 6400/25000 [======>.......................] - ETA: 11s - loss: 0.5903 - accuracy: 0.8381
 6500/25000 [======>.......................] - ETA: 11s - loss: 0.5903 - accuracy: 0.8382
 6600/25000 [======>.......................] - ETA: 11s - loss: 0.5902 - accuracy: 0.8386
 6700/25000 [=======>......................] - ETA: 11s - loss: 0.5901 - accuracy: 0.8388
 6800/25000 [=======>......................] - ETA: 11s - loss: 0.5900 - accuracy: 0.8384
 6900/25000 [=======>......................] - ETA: 11s - loss: 0.5898 - accuracy: 0.8387
 7000/25000 [=======>......................] - ETA: 10s - loss: 0.5897 - accuracy: 0.8387
 7100/25000 [=======>......................] - ETA: 10s - loss: 0.5896 - accuracy: 0.8387
 7200/25000 [=======>......................] - ETA: 10s - loss: 0.5896 - accuracy: 0.8382
 7300/25000 [=======>......................] - ETA: 10s - loss: 0.5894 - accuracy: 0.8384
 7400/25000 [=======>......................] - ETA: 10s - loss: 0.5894 - accuracy: 0.8382
 7500/25000 [========>.....................] - ETA: 10s - loss: 0.5892 - accuracy: 0.8385
 7600/25000 [========>.....................] - ETA: 10s - loss: 0.5890 - accuracy: 0.8389
 7700/25000 [========>.....................] - ETA: 10s - loss: 0.5888 - accuracy: 0.8394
 7800/25000 [========>.....................] - ETA: 10s - loss: 0.5888 - accuracy: 0.8396
 7900/25000 [========>.....................] - ETA: 10s - loss: 0.5885 - accuracy: 0.8409
 8000/25000 [========>.....................] - ETA: 10s - loss: 0.5882 - accuracy: 0.8414
 8100/25000 [========>.....................] - ETA: 10s - loss: 0.5879 - accuracy: 0.8417
 8200/25000 [========>.....................] - ETA: 10s - loss: 0.5877 - accuracy: 0.8420
 8300/25000 [========>.....................] - ETA: 10s - loss: 0.5873 - accuracy: 0.8424
 8400/25000 [=========>....................] - ETA: 10s - loss: 0.5873 - accuracy: 0.8419
 8500/25000 [=========>....................] - ETA: 10s - loss: 0.5870 - accuracy: 0.8424
 8600/25000 [=========>....................] - ETA: 10s - loss: 0.5873 - accuracy: 0.8417
 8700/25000 [=========>....................] - ETA: 9s - loss: 0.5873 - accuracy: 0.8413 
 8800/25000 [=========>....................] - ETA: 9s - loss: 0.5873 - accuracy: 0.8409
 8900/25000 [=========>....................] - ETA: 9s - loss: 0.5873 - accuracy: 0.8410
 9000/25000 [=========>....................] - ETA: 9s - loss: 0.5870 - accuracy: 0.8413
 9100/25000 [=========>....................] - ETA: 9s - loss: 0.5871 - accuracy: 0.8409
 9200/25000 [==========>...................] - ETA: 9s - loss: 0.5868 - accuracy: 0.8409
 9300/25000 [==========>...................] - ETA: 9s - loss: 0.5866 - accuracy: 0.8409
 9400/25000 [==========>...................] - ETA: 9s - loss: 0.5863 - accuracy: 0.8406
 9500/25000 [==========>...................] - ETA: 9s - loss: 0.5861 - accuracy: 0.8411
 9600/25000 [==========>...................] - ETA: 9s - loss: 0.5857 - accuracy: 0.8418
 9700/25000 [==========>...................] - ETA: 9s - loss: 0.5858 - accuracy: 0.8413
 9800/25000 [==========>...................] - ETA: 9s - loss: 0.5857 - accuracy: 0.8410
 9900/25000 [==========>...................] - ETA: 9s - loss: 0.5860 - accuracy: 0.8403
10000/25000 [===========>..................] - ETA: 9s - loss: 0.5864 - accuracy: 0.8390
10100/25000 [===========>..................] - ETA: 9s - loss: 0.5865 - accuracy: 0.8386
10200/25000 [===========>..................] - ETA: 9s - loss: 0.5865 - accuracy: 0.8387
10300/25000 [===========>..................] - ETA: 8s - loss: 0.5864 - accuracy: 0.8387
10400/25000 [===========>..................] - ETA: 8s - loss: 0.5863 - accuracy: 0.8388
10500/25000 [===========>..................] - ETA: 8s - loss: 0.5861 - accuracy: 0.8391
10600/25000 [===========>..................] - ETA: 8s - loss: 0.5859 - accuracy: 0.8393
10700/25000 [===========>..................] - ETA: 8s - loss: 0.5857 - accuracy: 0.8399
10800/25000 [===========>..................] - ETA: 8s - loss: 0.5857 - accuracy: 0.8398
10900/25000 [============>.................] - ETA: 8s - loss: 0.5857 - accuracy: 0.8396
11000/25000 [============>.................] - ETA: 8s - loss: 0.5856 - accuracy: 0.8395
11100/25000 [============>.................] - ETA: 8s - loss: 0.5855 - accuracy: 0.8396
11200/25000 [============>.................] - ETA: 8s - loss: 0.5854 - accuracy: 0.8396
11300/25000 [============>.................] - ETA: 8s - loss: 0.5856 - accuracy: 0.8394
11400/25000 [============>.................] - ETA: 8s - loss: 0.5854 - accuracy: 0.8401
11500/25000 [============>.................] - ETA: 8s - loss: 0.5851 - accuracy: 0.8403
11600/25000 [============>.................] - ETA: 8s - loss: 0.5850 - accuracy: 0.8401
11700/25000 [=============>................] - ETA: 8s - loss: 0.5849 - accuracy: 0.8402
11800/25000 [=============>................] - ETA: 8s - loss: 0.5848 - accuracy: 0.8403
11900/25000 [=============>................] - ETA: 7s - loss: 0.5847 - accuracy: 0.8400
12000/25000 [=============>................] - ETA: 7s - loss: 0.5846 - accuracy: 0.8400
12100/25000 [=============>................] - ETA: 7s - loss: 0.5846 - accuracy: 0.8399
12200/25000 [=============>................] - ETA: 7s - loss: 0.5845 - accuracy: 0.8399
12300/25000 [=============>................] - ETA: 7s - loss: 0.5844 - accuracy: 0.8403
12400/25000 [=============>................] - ETA: 7s - loss: 0.5842 - accuracy: 0.8402
12500/25000 [==============>...............] - ETA: 7s - loss: 0.5841 - accuracy: 0.8399
12600/25000 [==============>...............] - ETA: 7s - loss: 0.5839 - accuracy: 0.8400
12700/25000 [==============>...............] - ETA: 7s - loss: 0.5838 - accuracy: 0.8401
12800/25000 [==============>...............] - ETA: 7s - loss: 0.5837 - accuracy: 0.8402
12900/25000 [==============>...............] - ETA: 7s - loss: 0.5836 - accuracy: 0.8405
13000/25000 [==============>...............] - ETA: 7s - loss: 0.5833 - accuracy: 0.8408
13100/25000 [==============>...............] - ETA: 7s - loss: 0.5832 - accuracy: 0.8409
13200/25000 [==============>...............] - ETA: 7s - loss: 0.5829 - accuracy: 0.8414
13300/25000 [==============>...............] - ETA: 7s - loss: 0.5830 - accuracy: 0.8411
13400/25000 [===============>..............] - ETA: 7s - loss: 0.5828 - accuracy: 0.8411
13500/25000 [===============>..............] - ETA: 7s - loss: 0.5826 - accuracy: 0.8413
13600/25000 [===============>..............] - ETA: 6s - loss: 0.5826 - accuracy: 0.8411
13700/25000 [===============>..............] - ETA: 6s - loss: 0.5825 - accuracy: 0.8411
13800/25000 [===============>..............] - ETA: 6s - loss: 0.5823 - accuracy: 0.8414
13900/25000 [===============>..............] - ETA: 6s - loss: 0.5821 - accuracy: 0.8415
14000/25000 [===============>..............] - ETA: 6s - loss: 0.5820 - accuracy: 0.8416
14100/25000 [===============>..............] - ETA: 6s - loss: 0.5820 - accuracy: 0.8413
14200/25000 [================>.............] - ETA: 6s - loss: 0.5820 - accuracy: 0.8415
14300/25000 [================>.............] - ETA: 6s - loss: 0.5819 - accuracy: 0.8413
14400/25000 [================>.............] - ETA: 6s - loss: 0.5818 - accuracy: 0.8413
14500/25000 [================>.............] - ETA: 6s - loss: 0.5820 - accuracy: 0.8407
14600/25000 [================>.............] - ETA: 6s - loss: 0.5819 - accuracy: 0.8407
14700/25000 [================>.............] - ETA: 6s - loss: 0.5816 - accuracy: 0.8410
14800/25000 [================>.............] - ETA: 6s - loss: 0.5815 - accuracy: 0.8410
14900/25000 [================>.............] - ETA: 6s - loss: 0.5813 - accuracy: 0.8413
15000/25000 [=================>............] - ETA: 6s - loss: 0.5812 - accuracy: 0.8415
15100/25000 [=================>............] - ETA: 6s - loss: 0.5810 - accuracy: 0.8417
15200/25000 [=================>............] - ETA: 5s - loss: 0.5807 - accuracy: 0.8420
15300/25000 [=================>............] - ETA: 5s - loss: 0.5805 - accuracy: 0.8422
15400/25000 [=================>............] - ETA: 5s - loss: 0.5804 - accuracy: 0.8421
15500/25000 [=================>............] - ETA: 5s - loss: 0.5802 - accuracy: 0.8425
15600/25000 [=================>............] - ETA: 5s - loss: 0.5800 - accuracy: 0.8429
15700/25000 [=================>............] - ETA: 5s - loss: 0.5799 - accuracy: 0.8429
15800/25000 [=================>............] - ETA: 5s - loss: 0.5799 - accuracy: 0.8426
15900/25000 [==================>...........] - ETA: 5s - loss: 0.5797 - accuracy: 0.8428
16000/25000 [==================>...........] - ETA: 5s - loss: 0.5796 - accuracy: 0.8430
16100/25000 [==================>...........] - ETA: 5s - loss: 0.5795 - accuracy: 0.8431
16200/25000 [==================>...........] - ETA: 5s - loss: 0.5793 - accuracy: 0.8432
16300/25000 [==================>...........] - ETA: 5s - loss: 0.5792 - accuracy: 0.8435
16400/25000 [==================>...........] - ETA: 5s - loss: 0.5791 - accuracy: 0.8435
16500/25000 [==================>...........] - ETA: 5s - loss: 0.5790 - accuracy: 0.8435
16600/25000 [==================>...........] - ETA: 5s - loss: 0.5789 - accuracy: 0.8435
16700/25000 [===================>..........] - ETA: 5s - loss: 0.5787 - accuracy: 0.8437
16800/25000 [===================>..........] - ETA: 4s - loss: 0.5788 - accuracy: 0.8432
16900/25000 [===================>..........] - ETA: 4s - loss: 0.5787 - accuracy: 0.8432
17000/25000 [===================>..........] - ETA: 4s - loss: 0.5785 - accuracy: 0.8434
17100/25000 [===================>..........] - ETA: 4s - loss: 0.5782 - accuracy: 0.8437
17200/25000 [===================>..........] - ETA: 4s - loss: 0.5779 - accuracy: 0.8440
17300/25000 [===================>..........] - ETA: 4s - loss: 0.5776 - accuracy: 0.8445
17400/25000 [===================>..........] - ETA: 4s - loss: 0.5775 - accuracy: 0.8445
17500/25000 [====================>.........] - ETA: 4s - loss: 0.5775 - accuracy: 0.8442
17600/25000 [====================>.........] - ETA: 4s - loss: 0.5774 - accuracy: 0.8441
17700/25000 [====================>.........] - ETA: 4s - loss: 0.5773 - accuracy: 0.8441
17800/25000 [====================>.........] - ETA: 4s - loss: 0.5777 - accuracy: 0.8434
17900/25000 [====================>.........] - ETA: 4s - loss: 0.5774 - accuracy: 0.8436
18000/25000 [====================>.........] - ETA: 4s - loss: 0.5775 - accuracy: 0.8433
18100/25000 [====================>.........] - ETA: 4s - loss: 0.5775 - accuracy: 0.8431
18200/25000 [====================>.........] - ETA: 4s - loss: 0.5775 - accuracy: 0.8432
18300/25000 [====================>.........] - ETA: 4s - loss: 0.5774 - accuracy: 0.8433
18400/25000 [=====================>........] - ETA: 4s - loss: 0.5772 - accuracy: 0.8433
18500/25000 [=====================>........] - ETA: 3s - loss: 0.5772 - accuracy: 0.8430
18600/25000 [=====================>........] - ETA: 3s - loss: 0.5772 - accuracy: 0.8429
18700/25000 [=====================>........] - ETA: 3s - loss: 0.5772 - accuracy: 0.8426
18800/25000 [=====================>........] - ETA: 3s - loss: 0.5770 - accuracy: 0.8425
18900/25000 [=====================>........] - ETA: 3s - loss: 0.5770 - accuracy: 0.8425
19000/25000 [=====================>........] - ETA: 3s - loss: 0.5770 - accuracy: 0.8423
19100/25000 [=====================>........] - ETA: 3s - loss: 0.5767 - accuracy: 0.8426
19200/25000 [======================>.......] - ETA: 3s - loss: 0.5766 - accuracy: 0.8423
19300/25000 [======================>.......] - ETA: 3s - loss: 0.5765 - accuracy: 0.8425
19400/25000 [======================>.......] - ETA: 3s - loss: 0.5764 - accuracy: 0.8423
19500/25000 [======================>.......] - ETA: 3s - loss: 0.5764 - accuracy: 0.8423
19600/25000 [======================>.......] - ETA: 3s - loss: 0.5762 - accuracy: 0.8426
19700/25000 [======================>.......] - ETA: 3s - loss: 0.5759 - accuracy: 0.8429
19800/25000 [======================>.......] - ETA: 3s - loss: 0.5759 - accuracy: 0.8430
19900/25000 [======================>.......] - ETA: 3s - loss: 0.5757 - accuracy: 0.8432
20000/25000 [=======================>......] - ETA: 3s - loss: 0.5756 - accuracy: 0.8432
20100/25000 [=======================>......] - ETA: 2s - loss: 0.5759 - accuracy: 0.8426
20200/25000 [=======================>......] - ETA: 2s - loss: 0.5757 - accuracy: 0.8428
20300/25000 [=======================>......] - ETA: 2s - loss: 0.5756 - accuracy: 0.8427
20400/25000 [=======================>......] - ETA: 2s - loss: 0.5757 - accuracy: 0.8425
20500/25000 [=======================>......] - ETA: 2s - loss: 0.5760 - accuracy: 0.8419
20600/25000 [=======================>......] - ETA: 2s - loss: 0.5760 - accuracy: 0.8417
20700/25000 [=======================>......] - ETA: 2s - loss: 0.5760 - accuracy: 0.8417
20800/25000 [=======================>......] - ETA: 2s - loss: 0.5759 - accuracy: 0.8417
20900/25000 [========================>.....] - ETA: 2s - loss: 0.5758 - accuracy: 0.8417
21000/25000 [========================>.....] - ETA: 2s - loss: 0.5759 - accuracy: 0.8415
21100/25000 [========================>.....] - ETA: 2s - loss: 0.5757 - accuracy: 0.8418
21200/25000 [========================>.....] - ETA: 2s - loss: 0.5755 - accuracy: 0.8420
21300/25000 [========================>.....] - ETA: 2s - loss: 0.5754 - accuracy: 0.8420
21400/25000 [========================>.....] - ETA: 2s - loss: 0.5755 - accuracy: 0.8417
21500/25000 [========================>.....] - ETA: 2s - loss: 0.5753 - accuracy: 0.8419
21600/25000 [========================>.....] - ETA: 2s - loss: 0.5752 - accuracy: 0.8422
21700/25000 [=========================>....] - ETA: 2s - loss: 0.5750 - accuracy: 0.8424
21800/25000 [=========================>....] - ETA: 1s - loss: 0.5748 - accuracy: 0.8425
21900/25000 [=========================>....] - ETA: 1s - loss: 0.5748 - accuracy: 0.8423
22000/25000 [=========================>....] - ETA: 1s - loss: 0.5747 - accuracy: 0.8423
22100/25000 [=========================>....] - ETA: 1s - loss: 0.5747 - accuracy: 0.8421
22200/25000 [=========================>....] - ETA: 1s - loss: 0.5747 - accuracy: 0.8420
22300/25000 [=========================>....] - ETA: 1s - loss: 0.5744 - accuracy: 0.8423
22400/25000 [=========================>....] - ETA: 1s - loss: 0.5744 - accuracy: 0.8422
22500/25000 [==========================>...] - ETA: 1s - loss: 0.5742 - accuracy: 0.8425
22600/25000 [==========================>...] - ETA: 1s - loss: 0.5740 - accuracy: 0.8427
22700/25000 [==========================>...] - ETA: 1s - loss: 0.5739 - accuracy: 0.8428
22800/25000 [==========================>...] - ETA: 1s - loss: 0.5738 - accuracy: 0.8429
22900/25000 [==========================>...] - ETA: 1s - loss: 0.5736 - accuracy: 0.8428
23000/25000 [==========================>...] - ETA: 1s - loss: 0.5737 - accuracy: 0.8426
23100/25000 [==========================>...] - ETA: 1s - loss: 0.5737 - accuracy: 0.8426
23200/25000 [==========================>...] - ETA: 1s - loss: 0.5735 - accuracy: 0.8427
23300/25000 [==========================>...] - ETA: 1s - loss: 0.5734 - accuracy: 0.8427
23400/25000 [===========================>..] - ETA: 0s - loss: 0.5734 - accuracy: 0.8426
23500/25000 [===========================>..] - ETA: 0s - loss: 0.5734 - accuracy: 0.8426
23600/25000 [===========================>..] - ETA: 0s - loss: 0.5734 - accuracy: 0.8426
23700/25000 [===========================>..] - ETA: 0s - loss: 0.5732 - accuracy: 0.8428
23800/25000 [===========================>..] - ETA: 0s - loss: 0.5731 - accuracy: 0.8430
23900/25000 [===========================>..] - ETA: 0s - loss: 0.5729 - accuracy: 0.8431
24000/25000 [===========================>..] - ETA: 0s - loss: 0.5729 - accuracy: 0.8430
24100/25000 [===========================>..] - ETA: 0s - loss: 0.5727 - accuracy: 0.8432
24200/25000 [============================>.] - ETA: 0s - loss: 0.5726 - accuracy: 0.8432
24300/25000 [============================>.] - ETA: 0s - loss: 0.5725 - accuracy: 0.8434
24400/25000 [============================>.] - ETA: 0s - loss: 0.5724 - accuracy: 0.8435
24500/25000 [============================>.] - ETA: 0s - loss: 0.5723 - accuracy: 0.8436
24600/25000 [============================>.] - ETA: 0s - loss: 0.5722 - accuracy: 0.8437
24700/25000 [============================>.] - ETA: 0s - loss: 0.5720 - accuracy: 0.8438
24800/25000 [============================>.] - ETA: 0s - loss: 0.5720 - accuracy: 0.8437
24900/25000 [============================>.] - ETA: 0s - loss: 0.5719 - accuracy: 0.8437
25000/25000 [==============================] - 19s 769us/step - loss: 0.5717 - accuracy: 0.8439 - val_loss: 0.5530 - val_accuracy: 0.8443
Epoch 3/10

  100/25000 [..............................] - ETA: 15s - loss: 0.5407 - accuracy: 0.8800
  200/25000 [..............................] - ETA: 15s - loss: 0.5358 - accuracy: 0.8850
  300/25000 [..............................] - ETA: 15s - loss: 0.5301 - accuracy: 0.8967
  400/25000 [..............................] - ETA: 15s - loss: 0.5309 - accuracy: 0.8975
  500/25000 [..............................] - ETA: 14s - loss: 0.5351 - accuracy: 0.8840
  600/25000 [..............................] - ETA: 14s - loss: 0.5350 - accuracy: 0.8833
  700/25000 [..............................] - ETA: 14s - loss: 0.5368 - accuracy: 0.8786
  800/25000 [..............................] - ETA: 14s - loss: 0.5359 - accuracy: 0.8800
  900/25000 [>.............................] - ETA: 14s - loss: 0.5345 - accuracy: 0.8822
 1000/25000 [>.............................] - ETA: 14s - loss: 0.5340 - accuracy: 0.8830
 1100/25000 [>.............................] - ETA: 14s - loss: 0.5343 - accuracy: 0.8845
 1200/25000 [>.............................] - ETA: 14s - loss: 0.5327 - accuracy: 0.8875
 1300/25000 [>.............................] - ETA: 14s - loss: 0.5321 - accuracy: 0.8892
 1400/25000 [>.............................] - ETA: 14s - loss: 0.5318 - accuracy: 0.8886
 1500/25000 [>.............................] - ETA: 14s - loss: 0.5309 - accuracy: 0.8893
 1600/25000 [>.............................] - ETA: 14s - loss: 0.5338 - accuracy: 0.8844
 1700/25000 [=>............................] - ETA: 14s - loss: 0.5358 - accuracy: 0.8812
 1800/25000 [=>............................] - ETA: 14s - loss: 0.5362 - accuracy: 0.8794
 1900/25000 [=>............................] - ETA: 14s - loss: 0.5380 - accuracy: 0.8753
 2000/25000 [=>............................] - ETA: 13s - loss: 0.5374 - accuracy: 0.8765
 2100/25000 [=>............................] - ETA: 13s - loss: 0.5376 - accuracy: 0.8757
 2200/25000 [=>............................] - ETA: 13s - loss: 0.5369 - accuracy: 0.8768
 2300/25000 [=>............................] - ETA: 13s - loss: 0.5355 - accuracy: 0.8787
 2400/25000 [=>............................] - ETA: 13s - loss: 0.5348 - accuracy: 0.8808
 2500/25000 [==>...........................] - ETA: 13s - loss: 0.5340 - accuracy: 0.8824
 2600/25000 [==>...........................] - ETA: 13s - loss: 0.5337 - accuracy: 0.8819
 2700/25000 [==>...........................] - ETA: 13s - loss: 0.5335 - accuracy: 0.8819
 2800/25000 [==>...........................] - ETA: 13s - loss: 0.5331 - accuracy: 0.8825
 2900/25000 [==>...........................] - ETA: 13s - loss: 0.5330 - accuracy: 0.8838
 3000/25000 [==>...........................] - ETA: 13s - loss: 0.5322 - accuracy: 0.8853
 3100/25000 [==>...........................] - ETA: 13s - loss: 0.5315 - accuracy: 0.8868
 3200/25000 [==>...........................] - ETA: 13s - loss: 0.5307 - accuracy: 0.8888
 3300/25000 [==>...........................] - ETA: 13s - loss: 0.5309 - accuracy: 0.8885
 3400/25000 [===>..........................] - ETA: 13s - loss: 0.5307 - accuracy: 0.8888
 3500/25000 [===>..........................] - ETA: 13s - loss: 0.5311 - accuracy: 0.8871
 3600/25000 [===>..........................] - ETA: 12s - loss: 0.5304 - accuracy: 0.8883
 3700/25000 [===>..........................] - ETA: 12s - loss: 0.5303 - accuracy: 0.8881
 3800/25000 [===>..........................] - ETA: 12s - loss: 0.5299 - accuracy: 0.8879
 3900/25000 [===>..........................] - ETA: 12s - loss: 0.5306 - accuracy: 0.8862
 4000/25000 [===>..........................] - ETA: 12s - loss: 0.5302 - accuracy: 0.8870
 4100/25000 [===>..........................] - ETA: 12s - loss: 0.5295 - accuracy: 0.8885
 4200/25000 [====>.........................] - ETA: 12s - loss: 0.5297 - accuracy: 0.8881
 4300/25000 [====>.........................] - ETA: 12s - loss: 0.5293 - accuracy: 0.8884
 4400/25000 [====>.........................] - ETA: 12s - loss: 0.5292 - accuracy: 0.8880
 4500/25000 [====>.........................] - ETA: 12s - loss: 0.5292 - accuracy: 0.8884
 4600/25000 [====>.........................] - ETA: 12s - loss: 0.5290 - accuracy: 0.8885
 4700/25000 [====>.........................] - ETA: 12s - loss: 0.5290 - accuracy: 0.8879
 4800/25000 [====>.........................] - ETA: 12s - loss: 0.5279 - accuracy: 0.8894
 4900/25000 [====>.........................] - ETA: 12s - loss: 0.5274 - accuracy: 0.8900
 5000/25000 [=====>........................] - ETA: 12s - loss: 0.5284 - accuracy: 0.8886
 5100/25000 [=====>........................] - ETA: 12s - loss: 0.5280 - accuracy: 0.8894
 5200/25000 [=====>........................] - ETA: 12s - loss: 0.5281 - accuracy: 0.8892
 5300/25000 [=====>........................] - ETA: 11s - loss: 0.5279 - accuracy: 0.8894
 5400/25000 [=====>........................] - ETA: 11s - loss: 0.5279 - accuracy: 0.8891
 5500/25000 [=====>........................] - ETA: 11s - loss: 0.5275 - accuracy: 0.8896
 5600/25000 [=====>........................] - ETA: 11s - loss: 0.5272 - accuracy: 0.8902
 5700/25000 [=====>........................] - ETA: 11s - loss: 0.5271 - accuracy: 0.8898
 5800/25000 [=====>........................] - ETA: 11s - loss: 0.5270 - accuracy: 0.8895
 5900/25000 [======>.......................] - ETA: 11s - loss: 0.5269 - accuracy: 0.8893
 6000/25000 [======>.......................] - ETA: 11s - loss: 0.5268 - accuracy: 0.8892
 6100/25000 [======>.......................] - ETA: 11s - loss: 0.5265 - accuracy: 0.8890
 6200/25000 [======>.......................] - ETA: 11s - loss: 0.5263 - accuracy: 0.8894
 6300/25000 [======>.......................] - ETA: 11s - loss: 0.5262 - accuracy: 0.8894
 6400/25000 [======>.......................] - ETA: 11s - loss: 0.5258 - accuracy: 0.8897
 6500/25000 [======>.......................] - ETA: 11s - loss: 0.5255 - accuracy: 0.8902
 6600/25000 [======>.......................] - ETA: 11s - loss: 0.5254 - accuracy: 0.8898
 6700/25000 [=======>......................] - ETA: 11s - loss: 0.5251 - accuracy: 0.8903
 6800/25000 [=======>......................] - ETA: 11s - loss: 0.5250 - accuracy: 0.8904
 6900/25000 [=======>......................] - ETA: 10s - loss: 0.5252 - accuracy: 0.8897
 7000/25000 [=======>......................] - ETA: 10s - loss: 0.5248 - accuracy: 0.8901
 7100/25000 [=======>......................] - ETA: 10s - loss: 0.5245 - accuracy: 0.8904
 7200/25000 [=======>......................] - ETA: 10s - loss: 0.5248 - accuracy: 0.8897
 7300/25000 [=======>......................] - ETA: 10s - loss: 0.5245 - accuracy: 0.8901
 7400/25000 [=======>......................] - ETA: 10s - loss: 0.5244 - accuracy: 0.8901
 7500/25000 [========>.....................] - ETA: 10s - loss: 0.5243 - accuracy: 0.8900
 7600/25000 [========>.....................] - ETA: 10s - loss: 0.5240 - accuracy: 0.8904
 7700/25000 [========>.....................] - ETA: 10s - loss: 0.5236 - accuracy: 0.8909
 7800/25000 [========>.....................] - ETA: 10s - loss: 0.5235 - accuracy: 0.8913
 7900/25000 [========>.....................] - ETA: 10s - loss: 0.5232 - accuracy: 0.8915
 8000/25000 [========>.....................] - ETA: 10s - loss: 0.5234 - accuracy: 0.8914
 8100/25000 [========>.....................] - ETA: 10s - loss: 0.5235 - accuracy: 0.8910
 8200/25000 [========>.....................] - ETA: 10s - loss: 0.5238 - accuracy: 0.8902
 8300/25000 [========>.....................] - ETA: 10s - loss: 0.5235 - accuracy: 0.8907
 8400/25000 [=========>....................] - ETA: 10s - loss: 0.5235 - accuracy: 0.8905
 8500/25000 [=========>....................] - ETA: 10s - loss: 0.5236 - accuracy: 0.8905
 8600/25000 [=========>....................] - ETA: 9s - loss: 0.5238 - accuracy: 0.8897 
 8700/25000 [=========>....................] - ETA: 9s - loss: 0.5239 - accuracy: 0.8893
 8800/25000 [=========>....................] - ETA: 9s - loss: 0.5242 - accuracy: 0.8888
 8900/25000 [=========>....................] - ETA: 9s - loss: 0.5240 - accuracy: 0.8890
 9000/25000 [=========>....................] - ETA: 9s - loss: 0.5239 - accuracy: 0.8891
 9100/25000 [=========>....................] - ETA: 9s - loss: 0.5235 - accuracy: 0.8896
 9200/25000 [==========>...................] - ETA: 9s - loss: 0.5237 - accuracy: 0.8892
 9300/25000 [==========>...................] - ETA: 9s - loss: 0.5237 - accuracy: 0.8891
 9400/25000 [==========>...................] - ETA: 9s - loss: 0.5235 - accuracy: 0.8895
 9500/25000 [==========>...................] - ETA: 9s - loss: 0.5233 - accuracy: 0.8895
 9600/25000 [==========>...................] - ETA: 9s - loss: 0.5235 - accuracy: 0.8892
 9700/25000 [==========>...................] - ETA: 9s - loss: 0.5235 - accuracy: 0.8891
 9800/25000 [==========>...................] - ETA: 9s - loss: 0.5236 - accuracy: 0.8887
 9900/25000 [==========>...................] - ETA: 9s - loss: 0.5239 - accuracy: 0.8882
10000/25000 [===========>..................] - ETA: 9s - loss: 0.5238 - accuracy: 0.8884
10100/25000 [===========>..................] - ETA: 9s - loss: 0.5237 - accuracy: 0.8887
10200/25000 [===========>..................] - ETA: 9s - loss: 0.5237 - accuracy: 0.8884
10300/25000 [===========>..................] - ETA: 8s - loss: 0.5236 - accuracy: 0.8882
10400/25000 [===========>..................] - ETA: 8s - loss: 0.5233 - accuracy: 0.8887
10500/25000 [===========>..................] - ETA: 8s - loss: 0.5235 - accuracy: 0.8880
10600/25000 [===========>..................] - ETA: 8s - loss: 0.5233 - accuracy: 0.8883
10700/25000 [===========>..................] - ETA: 8s - loss: 0.5234 - accuracy: 0.8881
10800/25000 [===========>..................] - ETA: 8s - loss: 0.5234 - accuracy: 0.8876
10900/25000 [============>.................] - ETA: 8s - loss: 0.5233 - accuracy: 0.8877
11000/25000 [============>.................] - ETA: 8s - loss: 0.5235 - accuracy: 0.8875
11100/25000 [============>.................] - ETA: 8s - loss: 0.5233 - accuracy: 0.8877
11200/25000 [============>.................] - ETA: 8s - loss: 0.5232 - accuracy: 0.8877
11300/25000 [============>.................] - ETA: 8s - loss: 0.5233 - accuracy: 0.8874
11400/25000 [============>.................] - ETA: 8s - loss: 0.5231 - accuracy: 0.8875
11500/25000 [============>.................] - ETA: 8s - loss: 0.5229 - accuracy: 0.8878
11600/25000 [============>.................] - ETA: 8s - loss: 0.5228 - accuracy: 0.8875
11700/25000 [=============>................] - ETA: 8s - loss: 0.5231 - accuracy: 0.8869
11800/25000 [=============>................] - ETA: 8s - loss: 0.5231 - accuracy: 0.8865
11900/25000 [=============>................] - ETA: 7s - loss: 0.5230 - accuracy: 0.8865
12000/25000 [=============>................] - ETA: 7s - loss: 0.5231 - accuracy: 0.8861
12100/25000 [=============>................] - ETA: 7s - loss: 0.5229 - accuracy: 0.8862
12200/25000 [=============>................] - ETA: 7s - loss: 0.5226 - accuracy: 0.8866
12300/25000 [=============>................] - ETA: 7s - loss: 0.5227 - accuracy: 0.8862
12400/25000 [=============>................] - ETA: 7s - loss: 0.5225 - accuracy: 0.8864
12500/25000 [==============>...............] - ETA: 7s - loss: 0.5223 - accuracy: 0.8866
12600/25000 [==============>...............] - ETA: 7s - loss: 0.5223 - accuracy: 0.8864
12700/25000 [==============>...............] - ETA: 7s - loss: 0.5221 - accuracy: 0.8868
12800/25000 [==============>...............] - ETA: 7s - loss: 0.5221 - accuracy: 0.8864
12900/25000 [==============>...............] - ETA: 7s - loss: 0.5220 - accuracy: 0.8864
13000/25000 [==============>...............] - ETA: 7s - loss: 0.5220 - accuracy: 0.8864
13100/25000 [==============>...............] - ETA: 7s - loss: 0.5220 - accuracy: 0.8863
13200/25000 [==============>...............] - ETA: 7s - loss: 0.5221 - accuracy: 0.8859
13300/25000 [==============>...............] - ETA: 7s - loss: 0.5222 - accuracy: 0.8856
13400/25000 [===============>..............] - ETA: 7s - loss: 0.5220 - accuracy: 0.8859
13500/25000 [===============>..............] - ETA: 7s - loss: 0.5222 - accuracy: 0.8856
13600/25000 [===============>..............] - ETA: 6s - loss: 0.5221 - accuracy: 0.8854
13700/25000 [===============>..............] - ETA: 6s - loss: 0.5219 - accuracy: 0.8858
13800/25000 [===============>..............] - ETA: 6s - loss: 0.5215 - accuracy: 0.8864
13900/25000 [===============>..............] - ETA: 6s - loss: 0.5215 - accuracy: 0.8862
14000/25000 [===============>..............] - ETA: 6s - loss: 0.5216 - accuracy: 0.8859
14100/25000 [===============>..............] - ETA: 6s - loss: 0.5211 - accuracy: 0.8865
14200/25000 [================>.............] - ETA: 6s - loss: 0.5213 - accuracy: 0.8861
14300/25000 [================>.............] - ETA: 6s - loss: 0.5214 - accuracy: 0.8858
14400/25000 [================>.............] - ETA: 6s - loss: 0.5216 - accuracy: 0.8854
14500/25000 [================>.............] - ETA: 6s - loss: 0.5219 - accuracy: 0.8848
14600/25000 [================>.............] - ETA: 6s - loss: 0.5217 - accuracy: 0.8850
14700/25000 [================>.............] - ETA: 6s - loss: 0.5216 - accuracy: 0.8850
14800/25000 [================>.............] - ETA: 6s - loss: 0.5219 - accuracy: 0.8845
14900/25000 [================>.............] - ETA: 6s - loss: 0.5221 - accuracy: 0.8842
15000/25000 [=================>............] - ETA: 6s - loss: 0.5221 - accuracy: 0.8840
15100/25000 [=================>............] - ETA: 6s - loss: 0.5219 - accuracy: 0.8842
15200/25000 [=================>............] - ETA: 5s - loss: 0.5222 - accuracy: 0.8838
15300/25000 [=================>............] - ETA: 5s - loss: 0.5221 - accuracy: 0.8837
15400/25000 [=================>............] - ETA: 5s - loss: 0.5221 - accuracy: 0.8836
15500/25000 [=================>............] - ETA: 5s - loss: 0.5220 - accuracy: 0.8835
15600/25000 [=================>............] - ETA: 5s - loss: 0.5220 - accuracy: 0.8835
15700/25000 [=================>............] - ETA: 5s - loss: 0.5221 - accuracy: 0.8832
15800/25000 [=================>............] - ETA: 5s - loss: 0.5221 - accuracy: 0.8831
15900/25000 [==================>...........] - ETA: 5s - loss: 0.5221 - accuracy: 0.8829
16000/25000 [==================>...........] - ETA: 5s - loss: 0.5221 - accuracy: 0.8827
16100/25000 [==================>...........] - ETA: 5s - loss: 0.5225 - accuracy: 0.8819
16200/25000 [==================>...........] - ETA: 5s - loss: 0.5226 - accuracy: 0.8815
16300/25000 [==================>...........] - ETA: 5s - loss: 0.5226 - accuracy: 0.8814
16400/25000 [==================>...........] - ETA: 5s - loss: 0.5224 - accuracy: 0.8813
16500/25000 [==================>...........] - ETA: 5s - loss: 0.5224 - accuracy: 0.8813
16600/25000 [==================>...........] - ETA: 5s - loss: 0.5222 - accuracy: 0.8814
16700/25000 [===================>..........] - ETA: 5s - loss: 0.5222 - accuracy: 0.8813
16800/25000 [===================>..........] - ETA: 5s - loss: 0.5220 - accuracy: 0.8814
16900/25000 [===================>..........] - ETA: 4s - loss: 0.5221 - accuracy: 0.8811
17000/25000 [===================>..........] - ETA: 4s - loss: 0.5219 - accuracy: 0.8813
17100/25000 [===================>..........] - ETA: 4s - loss: 0.5221 - accuracy: 0.8808
17200/25000 [===================>..........] - ETA: 4s - loss: 0.5223 - accuracy: 0.8803
17300/25000 [===================>..........] - ETA: 4s - loss: 0.5223 - accuracy: 0.8803
17400/25000 [===================>..........] - ETA: 4s - loss: 0.5224 - accuracy: 0.8800
17500/25000 [====================>.........] - ETA: 4s - loss: 0.5223 - accuracy: 0.8802
17600/25000 [====================>.........] - ETA: 4s - loss: 0.5223 - accuracy: 0.8802
17700/25000 [====================>.........] - ETA: 4s - loss: 0.5222 - accuracy: 0.8801
17800/25000 [====================>.........] - ETA: 4s - loss: 0.5223 - accuracy: 0.8798
17900/25000 [====================>.........] - ETA: 4s - loss: 0.5223 - accuracy: 0.8798
18000/25000 [====================>.........] - ETA: 4s - loss: 0.5222 - accuracy: 0.8799
18100/25000 [====================>.........] - ETA: 4s - loss: 0.5222 - accuracy: 0.8798
18200/25000 [====================>.........] - ETA: 4s - loss: 0.5222 - accuracy: 0.8796
18300/25000 [====================>.........] - ETA: 4s - loss: 0.5222 - accuracy: 0.8795
18400/25000 [=====================>........] - ETA: 4s - loss: 0.5222 - accuracy: 0.8793
18500/25000 [=====================>........] - ETA: 3s - loss: 0.5222 - accuracy: 0.8792
18600/25000 [=====================>........] - ETA: 3s - loss: 0.5223 - accuracy: 0.8790
18700/25000 [=====================>........] - ETA: 3s - loss: 0.5221 - accuracy: 0.8792
18800/25000 [=====================>........] - ETA: 3s - loss: 0.5223 - accuracy: 0.8787
18900/25000 [=====================>........] - ETA: 3s - loss: 0.5222 - accuracy: 0.8788
19000/25000 [=====================>........] - ETA: 3s - loss: 0.5222 - accuracy: 0.8788
19100/25000 [=====================>........] - ETA: 3s - loss: 0.5219 - accuracy: 0.8791
19200/25000 [======================>.......] - ETA: 3s - loss: 0.5220 - accuracy: 0.8788
19300/25000 [======================>.......] - ETA: 3s - loss: 0.5218 - accuracy: 0.8789
19400/25000 [======================>.......] - ETA: 3s - loss: 0.5217 - accuracy: 0.8791
19500/25000 [======================>.......] - ETA: 3s - loss: 0.5217 - accuracy: 0.8789
19600/25000 [======================>.......] - ETA: 3s - loss: 0.5215 - accuracy: 0.8790
19700/25000 [======================>.......] - ETA: 3s - loss: 0.5214 - accuracy: 0.8791
19800/25000 [======================>.......] - ETA: 3s - loss: 0.5214 - accuracy: 0.8789
19900/25000 [======================>.......] - ETA: 3s - loss: 0.5214 - accuracy: 0.8789
20000/25000 [=======================>......] - ETA: 3s - loss: 0.5214 - accuracy: 0.8788
20100/25000 [=======================>......] - ETA: 3s - loss: 0.5212 - accuracy: 0.8790
20200/25000 [=======================>......] - ETA: 2s - loss: 0.5209 - accuracy: 0.8793
20300/25000 [=======================>......] - ETA: 2s - loss: 0.5208 - accuracy: 0.8794
20400/25000 [=======================>......] - ETA: 2s - loss: 0.5207 - accuracy: 0.8795
20500/25000 [=======================>......] - ETA: 2s - loss: 0.5207 - accuracy: 0.8795
20600/25000 [=======================>......] - ETA: 2s - loss: 0.5204 - accuracy: 0.8798
20700/25000 [=======================>......] - ETA: 2s - loss: 0.5204 - accuracy: 0.8797
20800/25000 [=======================>......] - ETA: 2s - loss: 0.5204 - accuracy: 0.8796
20900/25000 [========================>.....] - ETA: 2s - loss: 0.5203 - accuracy: 0.8796
21000/25000 [========================>.....] - ETA: 2s - loss: 0.5201 - accuracy: 0.8797
21100/25000 [========================>.....] - ETA: 2s - loss: 0.5198 - accuracy: 0.8800
21200/25000 [========================>.....] - ETA: 2s - loss: 0.5197 - accuracy: 0.8801
21300/25000 [========================>.....] - ETA: 2s - loss: 0.5196 - accuracy: 0.8802
21400/25000 [========================>.....] - ETA: 2s - loss: 0.5197 - accuracy: 0.8800
21500/25000 [========================>.....] - ETA: 2s - loss: 0.5195 - accuracy: 0.8801
21600/25000 [========================>.....] - ETA: 2s - loss: 0.5194 - accuracy: 0.8802
21700/25000 [=========================>....] - ETA: 2s - loss: 0.5193 - accuracy: 0.8801
21800/25000 [=========================>....] - ETA: 1s - loss: 0.5192 - accuracy: 0.8803
21900/25000 [=========================>....] - ETA: 1s - loss: 0.5191 - accuracy: 0.8803
22000/25000 [=========================>....] - ETA: 1s - loss: 0.5190 - accuracy: 0.8802
22100/25000 [=========================>....] - ETA: 1s - loss: 0.5188 - accuracy: 0.8804
22200/25000 [=========================>....] - ETA: 1s - loss: 0.5187 - accuracy: 0.8805
22300/25000 [=========================>....] - ETA: 1s - loss: 0.5185 - accuracy: 0.8807
22400/25000 [=========================>....] - ETA: 1s - loss: 0.5183 - accuracy: 0.8808
22500/25000 [==========================>...] - ETA: 1s - loss: 0.5184 - accuracy: 0.8806
22600/25000 [==========================>...] - ETA: 1s - loss: 0.5182 - accuracy: 0.8808
22700/25000 [==========================>...] - ETA: 1s - loss: 0.5181 - accuracy: 0.8809
22800/25000 [==========================>...] - ETA: 1s - loss: 0.5179 - accuracy: 0.8810
22900/25000 [==========================>...] - ETA: 1s - loss: 0.5177 - accuracy: 0.8813
23000/25000 [==========================>...] - ETA: 1s - loss: 0.5176 - accuracy: 0.8813
23100/25000 [==========================>...] - ETA: 1s - loss: 0.5175 - accuracy: 0.8814
23200/25000 [==========================>...] - ETA: 1s - loss: 0.5175 - accuracy: 0.8814
23300/25000 [==========================>...] - ETA: 1s - loss: 0.5174 - accuracy: 0.8814
23400/25000 [===========================>..] - ETA: 0s - loss: 0.5174 - accuracy: 0.8815
23500/25000 [===========================>..] - ETA: 0s - loss: 0.5173 - accuracy: 0.8816
23600/25000 [===========================>..] - ETA: 0s - loss: 0.5173 - accuracy: 0.8814
23700/25000 [===========================>..] - ETA: 0s - loss: 0.5171 - accuracy: 0.8815
23800/25000 [===========================>..] - ETA: 0s - loss: 0.5172 - accuracy: 0.8813
23900/25000 [===========================>..] - ETA: 0s - loss: 0.5173 - accuracy: 0.8811
24000/25000 [===========================>..] - ETA: 0s - loss: 0.5171 - accuracy: 0.8813
24100/25000 [===========================>..] - ETA: 0s - loss: 0.5170 - accuracy: 0.8813
24200/25000 [============================>.] - ETA: 0s - loss: 0.5170 - accuracy: 0.8812
24300/25000 [============================>.] - ETA: 0s - loss: 0.5169 - accuracy: 0.8813
24400/25000 [============================>.] - ETA: 0s - loss: 0.5168 - accuracy: 0.8814
24500/25000 [============================>.] - ETA: 0s - loss: 0.5167 - accuracy: 0.8816
24600/25000 [============================>.] - ETA: 0s - loss: 0.5166 - accuracy: 0.8814
24700/25000 [============================>.] - ETA: 0s - loss: 0.5167 - accuracy: 0.8811
24800/25000 [============================>.] - ETA: 0s - loss: 0.5166 - accuracy: 0.8811
24900/25000 [============================>.] - ETA: 0s - loss: 0.5167 - accuracy: 0.8810
25000/25000 [==============================] - 19s 772us/step - loss: 0.5166 - accuracy: 0.8810 - val_loss: 0.5152 - val_accuracy: 0.8538
Epoch 4/10

  100/25000 [..............................] - ETA: 15s - loss: 0.4796 - accuracy: 0.9100
  200/25000 [..............................] - ETA: 15s - loss: 0.4889 - accuracy: 0.9000
  300/25000 [..............................] - ETA: 15s - loss: 0.4981 - accuracy: 0.8867
  400/25000 [..............................] - ETA: 14s - loss: 0.4931 - accuracy: 0.8950
  500/25000 [..............................] - ETA: 14s - loss: 0.4847 - accuracy: 0.9080
  600/25000 [..............................] - ETA: 14s - loss: 0.4776 - accuracy: 0.9150
  700/25000 [..............................] - ETA: 14s - loss: 0.4842 - accuracy: 0.9043
  800/25000 [..............................] - ETA: 14s - loss: 0.4826 - accuracy: 0.9075
  900/25000 [>.............................] - ETA: 14s - loss: 0.4803 - accuracy: 0.9111
 1000/25000 [>.............................] - ETA: 14s - loss: 0.4778 - accuracy: 0.9150
 1100/25000 [>.............................] - ETA: 14s - loss: 0.4792 - accuracy: 0.9136
 1200/25000 [>.............................] - ETA: 14s - loss: 0.4788 - accuracy: 0.9133
 1300/25000 [>.............................] - ETA: 14s - loss: 0.4794 - accuracy: 0.9108
 1400/25000 [>.............................] - ETA: 14s - loss: 0.4792 - accuracy: 0.9107
 1500/25000 [>.............................] - ETA: 14s - loss: 0.4789 - accuracy: 0.9120
 1600/25000 [>.............................] - ETA: 14s - loss: 0.4781 - accuracy: 0.9131
 1700/25000 [=>............................] - ETA: 14s - loss: 0.4775 - accuracy: 0.9135
 1800/25000 [=>............................] - ETA: 13s - loss: 0.4800 - accuracy: 0.9106
 1900/25000 [=>............................] - ETA: 13s - loss: 0.4804 - accuracy: 0.9095
 2000/25000 [=>............................] - ETA: 13s - loss: 0.4789 - accuracy: 0.9115
 2100/25000 [=>............................] - ETA: 13s - loss: 0.4781 - accuracy: 0.9124
 2200/25000 [=>............................] - ETA: 13s - loss: 0.4787 - accuracy: 0.9114
 2300/25000 [=>............................] - ETA: 13s - loss: 0.4791 - accuracy: 0.9113
 2400/25000 [=>............................] - ETA: 13s - loss: 0.4794 - accuracy: 0.9112
 2500/25000 [==>...........................] - ETA: 13s - loss: 0.4791 - accuracy: 0.9116
 2600/25000 [==>...........................] - ETA: 13s - loss: 0.4797 - accuracy: 0.9096
 2700/25000 [==>...........................] - ETA: 13s - loss: 0.4805 - accuracy: 0.9081
 2800/25000 [==>...........................] - ETA: 13s - loss: 0.4798 - accuracy: 0.9093
 2900/25000 [==>...........................] - ETA: 13s - loss: 0.4791 - accuracy: 0.9107
 3000/25000 [==>...........................] - ETA: 13s - loss: 0.4789 - accuracy: 0.9110
 3100/25000 [==>...........................] - ETA: 13s - loss: 0.4782 - accuracy: 0.9119
 3200/25000 [==>...........................] - ETA: 13s - loss: 0.4803 - accuracy: 0.9084
 3300/25000 [==>...........................] - ETA: 13s - loss: 0.4804 - accuracy: 0.9082
 3400/25000 [===>..........................] - ETA: 13s - loss: 0.4805 - accuracy: 0.9079
 3500/25000 [===>..........................] - ETA: 13s - loss: 0.4810 - accuracy: 0.9069
 3600/25000 [===>..........................] - ETA: 13s - loss: 0.4812 - accuracy: 0.9067
 3700/25000 [===>..........................] - ETA: 12s - loss: 0.4814 - accuracy: 0.9062
 3800/25000 [===>..........................] - ETA: 12s - loss: 0.4816 - accuracy: 0.9063
 3900/25000 [===>..........................] - ETA: 12s - loss: 0.4814 - accuracy: 0.9062
 4000/25000 [===>..........................] - ETA: 12s - loss: 0.4805 - accuracy: 0.9072
 4100/25000 [===>..........................] - ETA: 12s - loss: 0.4810 - accuracy: 0.9061
 4200/25000 [====>.........................] - ETA: 12s - loss: 0.4814 - accuracy: 0.9052
 4300/25000 [====>.........................] - ETA: 12s - loss: 0.4816 - accuracy: 0.9051
 4400/25000 [====>.........................] - ETA: 12s - loss: 0.4829 - accuracy: 0.9030
 4500/25000 [====>.........................] - ETA: 12s - loss: 0.4831 - accuracy: 0.9029
 4600/25000 [====>.........................] - ETA: 12s - loss: 0.4836 - accuracy: 0.9022
 4700/25000 [====>.........................] - ETA: 12s - loss: 0.4838 - accuracy: 0.9017
 4800/25000 [====>.........................] - ETA: 12s - loss: 0.4845 - accuracy: 0.9006
 4900/25000 [====>.........................] - ETA: 12s - loss: 0.4843 - accuracy: 0.9006
 5000/25000 [=====>........................] - ETA: 12s - loss: 0.4846 - accuracy: 0.9000
 5100/25000 [=====>........................] - ETA: 12s - loss: 0.4846 - accuracy: 0.8996
 5200/25000 [=====>........................] - ETA: 12s - loss: 0.4850 - accuracy: 0.8987
 5300/25000 [=====>........................] - ETA: 12s - loss: 0.4855 - accuracy: 0.8974
 5400/25000 [=====>........................] - ETA: 11s - loss: 0.4854 - accuracy: 0.8974
 5500/25000 [=====>........................] - ETA: 11s - loss: 0.4860 - accuracy: 0.8962
 5600/25000 [=====>........................] - ETA: 11s - loss: 0.4861 - accuracy: 0.8957
 5700/25000 [=====>........................] - ETA: 11s - loss: 0.4863 - accuracy: 0.8953
 5800/25000 [=====>........................] - ETA: 11s - loss: 0.4863 - accuracy: 0.8950
 5900/25000 [======>.......................] - ETA: 11s - loss: 0.4862 - accuracy: 0.8953
 6000/25000 [======>.......................] - ETA: 11s - loss: 0.4859 - accuracy: 0.8958
 6100/25000 [======>.......................] - ETA: 11s - loss: 0.4863 - accuracy: 0.8954
 6200/25000 [======>.......................] - ETA: 11s - loss: 0.4860 - accuracy: 0.8956
 6300/25000 [======>.......................] - ETA: 11s - loss: 0.4860 - accuracy: 0.8956
 6400/25000 [======>.......................] - ETA: 11s - loss: 0.4858 - accuracy: 0.8956
 6500/25000 [======>.......................] - ETA: 11s - loss: 0.4865 - accuracy: 0.8946
 6600/25000 [======>.......................] - ETA: 11s - loss: 0.4860 - accuracy: 0.8952
 6700/25000 [=======>......................] - ETA: 11s - loss: 0.4857 - accuracy: 0.8954
 6800/25000 [=======>......................] - ETA: 11s - loss: 0.4861 - accuracy: 0.8950
 6900/25000 [=======>......................] - ETA: 11s - loss: 0.4861 - accuracy: 0.8948
 7000/25000 [=======>......................] - ETA: 11s - loss: 0.4857 - accuracy: 0.8951
 7100/25000 [=======>......................] - ETA: 10s - loss: 0.4857 - accuracy: 0.8952
 7200/25000 [=======>......................] - ETA: 10s - loss: 0.4853 - accuracy: 0.8956
 7300/25000 [=======>......................] - ETA: 10s - loss: 0.4848 - accuracy: 0.8962
 7400/25000 [=======>......................] - ETA: 10s - loss: 0.4847 - accuracy: 0.8962
 7500/25000 [========>.....................] - ETA: 10s - loss: 0.4843 - accuracy: 0.8967
 7600/25000 [========>.....................] - ETA: 10s - loss: 0.4845 - accuracy: 0.8963
 7700/25000 [========>.....................] - ETA: 10s - loss: 0.4844 - accuracy: 0.8965
 7800/25000 [========>.....................] - ETA: 10s - loss: 0.4846 - accuracy: 0.8963
 7900/25000 [========>.....................] - ETA: 10s - loss: 0.4843 - accuracy: 0.8963
 8000/25000 [========>.....................] - ETA: 10s - loss: 0.4845 - accuracy: 0.8961
 8100/25000 [========>.....................] - ETA: 10s - loss: 0.4843 - accuracy: 0.8964
 8200/25000 [========>.....................] - ETA: 10s - loss: 0.4841 - accuracy: 0.8966
 8300/25000 [========>.....................] - ETA: 10s - loss: 0.4841 - accuracy: 0.8964
 8400/25000 [=========>....................] - ETA: 10s - loss: 0.4841 - accuracy: 0.8961
 8500/25000 [=========>....................] - ETA: 10s - loss: 0.4837 - accuracy: 0.8965
 8600/25000 [=========>....................] - ETA: 10s - loss: 0.4837 - accuracy: 0.8964
 8700/25000 [=========>....................] - ETA: 9s - loss: 0.4839 - accuracy: 0.8961 
 8800/25000 [=========>....................] - ETA: 9s - loss: 0.4839 - accuracy: 0.8958
 8900/25000 [=========>....................] - ETA: 9s - loss: 0.4838 - accuracy: 0.8960
 9000/25000 [=========>....................] - ETA: 9s - loss: 0.4832 - accuracy: 0.8967
 9100/25000 [=========>....................] - ETA: 9s - loss: 0.4832 - accuracy: 0.8963
 9200/25000 [==========>...................] - ETA: 9s - loss: 0.4832 - accuracy: 0.8959
 9300/25000 [==========>...................] - ETA: 9s - loss: 0.4832 - accuracy: 0.8957
 9400/25000 [==========>...................] - ETA: 9s - loss: 0.4829 - accuracy: 0.8961
 9500/25000 [==========>...................] - ETA: 9s - loss: 0.4831 - accuracy: 0.8959
 9600/25000 [==========>...................] - ETA: 9s - loss: 0.4833 - accuracy: 0.8955
 9700/25000 [==========>...................] - ETA: 9s - loss: 0.4830 - accuracy: 0.8958
 9800/25000 [==========>...................] - ETA: 9s - loss: 0.4830 - accuracy: 0.8957
 9900/25000 [==========>...................] - ETA: 9s - loss: 0.4827 - accuracy: 0.8961
10000/25000 [===========>..................] - ETA: 9s - loss: 0.4829 - accuracy: 0.8956
10100/25000 [===========>..................] - ETA: 9s - loss: 0.4828 - accuracy: 0.8956
10200/25000 [===========>..................] - ETA: 9s - loss: 0.4828 - accuracy: 0.8958
10300/25000 [===========>..................] - ETA: 8s - loss: 0.4825 - accuracy: 0.8959
10400/25000 [===========>..................] - ETA: 8s - loss: 0.4822 - accuracy: 0.8963
10500/25000 [===========>..................] - ETA: 8s - loss: 0.4825 - accuracy: 0.8957
10600/25000 [===========>..................] - ETA: 8s - loss: 0.4828 - accuracy: 0.8950
10700/25000 [===========>..................] - ETA: 8s - loss: 0.4827 - accuracy: 0.8952
10800/25000 [===========>..................] - ETA: 8s - loss: 0.4828 - accuracy: 0.8954
10900/25000 [============>.................] - ETA: 8s - loss: 0.4832 - accuracy: 0.8948
11000/25000 [============>.................] - ETA: 8s - loss: 0.4833 - accuracy: 0.8946
11100/25000 [============>.................] - ETA: 8s - loss: 0.4833 - accuracy: 0.8944
11200/25000 [============>.................] - ETA: 8s - loss: 0.4836 - accuracy: 0.8938
11300/25000 [============>.................] - ETA: 8s - loss: 0.4833 - accuracy: 0.8942
11400/25000 [============>.................] - ETA: 8s - loss: 0.4832 - accuracy: 0.8943
11500/25000 [============>.................] - ETA: 8s - loss: 0.4833 - accuracy: 0.8943
11600/25000 [============>.................] - ETA: 8s - loss: 0.4832 - accuracy: 0.8943
11700/25000 [=============>................] - ETA: 8s - loss: 0.4835 - accuracy: 0.8938
11800/25000 [=============>................] - ETA: 8s - loss: 0.4835 - accuracy: 0.8937
11900/25000 [=============>................] - ETA: 8s - loss: 0.4836 - accuracy: 0.8936
12000/25000 [=============>................] - ETA: 7s - loss: 0.4834 - accuracy: 0.8938
12100/25000 [=============>................] - ETA: 7s - loss: 0.4836 - accuracy: 0.8936
12200/25000 [=============>................] - ETA: 7s - loss: 0.4839 - accuracy: 0.8931
12300/25000 [=============>................] - ETA: 7s - loss: 0.4839 - accuracy: 0.8931
12400/25000 [=============>................] - ETA: 7s - loss: 0.4835 - accuracy: 0.8934
12500/25000 [==============>...............] - ETA: 7s - loss: 0.4834 - accuracy: 0.8934
12600/25000 [==============>...............] - ETA: 7s - loss: 0.4831 - accuracy: 0.8937
12700/25000 [==============>...............] - ETA: 7s - loss: 0.4831 - accuracy: 0.8935
12800/25000 [==============>...............] - ETA: 7s - loss: 0.4835 - accuracy: 0.8928
12900/25000 [==============>...............] - ETA: 7s - loss: 0.4838 - accuracy: 0.8922
13000/25000 [==============>...............] - ETA: 7s - loss: 0.4837 - accuracy: 0.8921
13100/25000 [==============>...............] - ETA: 7s - loss: 0.4838 - accuracy: 0.8918
13200/25000 [==============>...............] - ETA: 7s - loss: 0.4838 - accuracy: 0.8917
13300/25000 [==============>...............] - ETA: 7s - loss: 0.4839 - accuracy: 0.8916
13400/25000 [===============>..............] - ETA: 7s - loss: 0.4839 - accuracy: 0.8915
13500/25000 [===============>..............] - ETA: 7s - loss: 0.4839 - accuracy: 0.8913
13600/25000 [===============>..............] - ETA: 6s - loss: 0.4841 - accuracy: 0.8909
13700/25000 [===============>..............] - ETA: 6s - loss: 0.4845 - accuracy: 0.8901
13800/25000 [===============>..............] - ETA: 6s - loss: 0.4845 - accuracy: 0.8900
13900/25000 [===============>..............] - ETA: 6s - loss: 0.4848 - accuracy: 0.8894
14000/25000 [===============>..............] - ETA: 6s - loss: 0.4848 - accuracy: 0.8893
14100/25000 [===============>..............] - ETA: 6s - loss: 0.4849 - accuracy: 0.8890
14200/25000 [================>.............] - ETA: 6s - loss: 0.4848 - accuracy: 0.8890
14300/25000 [================>.............] - ETA: 6s - loss: 0.4848 - accuracy: 0.8889
14400/25000 [================>.............] - ETA: 6s - loss: 0.4848 - accuracy: 0.8889
14500/25000 [================>.............] - ETA: 6s - loss: 0.4848 - accuracy: 0.8888
14600/25000 [================>.............] - ETA: 6s - loss: 0.4851 - accuracy: 0.8883
14700/25000 [================>.............] - ETA: 6s - loss: 0.4852 - accuracy: 0.8882
14800/25000 [================>.............] - ETA: 6s - loss: 0.4853 - accuracy: 0.8881
14900/25000 [================>.............] - ETA: 6s - loss: 0.4852 - accuracy: 0.8881
15000/25000 [=================>............] - ETA: 6s - loss: 0.4850 - accuracy: 0.8880
15100/25000 [=================>............] - ETA: 6s - loss: 0.4852 - accuracy: 0.8877
15200/25000 [=================>............] - ETA: 5s - loss: 0.4850 - accuracy: 0.8878
15300/25000 [=================>............] - ETA: 5s - loss: 0.4849 - accuracy: 0.8879
15400/25000 [=================>............] - ETA: 5s - loss: 0.4851 - accuracy: 0.8873
15500/25000 [=================>............] - ETA: 5s - loss: 0.4848 - accuracy: 0.8877
15600/25000 [=================>............] - ETA: 5s - loss: 0.4848 - accuracy: 0.8877
15700/25000 [=================>............] - ETA: 5s - loss: 0.4847 - accuracy: 0.8878
15800/25000 [=================>............] - ETA: 5s - loss: 0.4847 - accuracy: 0.8878
15900/25000 [==================>...........] - ETA: 5s - loss: 0.4846 - accuracy: 0.8879
16000/25000 [==================>...........] - ETA: 5s - loss: 0.4845 - accuracy: 0.8879
16100/25000 [==================>...........] - ETA: 5s - loss: 0.4845 - accuracy: 0.8879
16200/25000 [==================>...........] - ETA: 5s - loss: 0.4843 - accuracy: 0.8880
16300/25000 [==================>...........] - ETA: 5s - loss: 0.4842 - accuracy: 0.8882
16400/25000 [==================>...........] - ETA: 5s - loss: 0.4842 - accuracy: 0.8882
16500/25000 [==================>...........] - ETA: 5s - loss: 0.4839 - accuracy: 0.8885
16600/25000 [==================>...........] - ETA: 5s - loss: 0.4841 - accuracy: 0.8883
16700/25000 [===================>..........] - ETA: 5s - loss: 0.4841 - accuracy: 0.8881
16800/25000 [===================>..........] - ETA: 5s - loss: 0.4838 - accuracy: 0.8884
16900/25000 [===================>..........] - ETA: 4s - loss: 0.4837 - accuracy: 0.8885
17000/25000 [===================>..........] - ETA: 4s - loss: 0.4833 - accuracy: 0.8888
17100/25000 [===================>..........] - ETA: 4s - loss: 0.4834 - accuracy: 0.8887
17200/25000 [===================>..........] - ETA: 4s - loss: 0.4831 - accuracy: 0.8890
17300/25000 [===================>..........] - ETA: 4s - loss: 0.4834 - accuracy: 0.8886
17400/25000 [===================>..........] - ETA: 4s - loss: 0.4833 - accuracy: 0.8887
17500/25000 [====================>.........] - ETA: 4s - loss: 0.4833 - accuracy: 0.8886
17600/25000 [====================>.........] - ETA: 4s - loss: 0.4833 - accuracy: 0.8886
17700/25000 [====================>.........] - ETA: 4s - loss: 0.4832 - accuracy: 0.8886
17800/25000 [====================>.........] - ETA: 4s - loss: 0.4833 - accuracy: 0.8884
17900/25000 [====================>.........] - ETA: 4s - loss: 0.4833 - accuracy: 0.8883
18000/25000 [====================>.........] - ETA: 4s - loss: 0.4833 - accuracy: 0.8881
18100/25000 [====================>.........] - ETA: 4s - loss: 0.4834 - accuracy: 0.8878
18200/25000 [====================>.........] - ETA: 4s - loss: 0.4834 - accuracy: 0.8876
18300/25000 [====================>.........] - ETA: 4s - loss: 0.4835 - accuracy: 0.8875
18400/25000 [=====================>........] - ETA: 4s - loss: 0.4834 - accuracy: 0.8875
18500/25000 [=====================>........] - ETA: 3s - loss: 0.4833 - accuracy: 0.8876
18600/25000 [=====================>........] - ETA: 3s - loss: 0.4831 - accuracy: 0.8878
18700/25000 [=====================>........] - ETA: 3s - loss: 0.4830 - accuracy: 0.8879
18800/25000 [=====================>........] - ETA: 3s - loss: 0.4828 - accuracy: 0.8880
18900/25000 [=====================>........] - ETA: 3s - loss: 0.4827 - accuracy: 0.8880
19000/25000 [=====================>........] - ETA: 3s - loss: 0.4827 - accuracy: 0.8882
19100/25000 [=====================>........] - ETA: 3s - loss: 0.4825 - accuracy: 0.8884
19200/25000 [======================>.......] - ETA: 3s - loss: 0.4826 - accuracy: 0.8882
19300/25000 [======================>.......] - ETA: 3s - loss: 0.4825 - accuracy: 0.8884
19400/25000 [======================>.......] - ETA: 3s - loss: 0.4823 - accuracy: 0.8887
19500/25000 [======================>.......] - ETA: 3s - loss: 0.4823 - accuracy: 0.8885
19600/25000 [======================>.......] - ETA: 3s - loss: 0.4822 - accuracy: 0.8885
19700/25000 [======================>.......] - ETA: 3s - loss: 0.4821 - accuracy: 0.8886
19800/25000 [======================>.......] - ETA: 3s - loss: 0.4821 - accuracy: 0.8886
19900/25000 [======================>.......] - ETA: 3s - loss: 0.4823 - accuracy: 0.8882
20000/25000 [=======================>......] - ETA: 3s - loss: 0.4822 - accuracy: 0.8882
20100/25000 [=======================>......] - ETA: 2s - loss: 0.4822 - accuracy: 0.8882
20200/25000 [=======================>......] - ETA: 2s - loss: 0.4821 - accuracy: 0.8883
20300/25000 [=======================>......] - ETA: 2s - loss: 0.4819 - accuracy: 0.8886
20400/25000 [=======================>......] - ETA: 2s - loss: 0.4818 - accuracy: 0.8885
20500/25000 [=======================>......] - ETA: 2s - loss: 0.4819 - accuracy: 0.8884
20600/25000 [=======================>......] - ETA: 2s - loss: 0.4817 - accuracy: 0.8886
20700/25000 [=======================>......] - ETA: 2s - loss: 0.4815 - accuracy: 0.8887
20800/25000 [=======================>......] - ETA: 2s - loss: 0.4814 - accuracy: 0.8888
20900/25000 [========================>.....] - ETA: 2s - loss: 0.4814 - accuracy: 0.8889
21000/25000 [========================>.....] - ETA: 2s - loss: 0.4811 - accuracy: 0.8890
21100/25000 [========================>.....] - ETA: 2s - loss: 0.4812 - accuracy: 0.8888
21200/25000 [========================>.....] - ETA: 2s - loss: 0.4809 - accuracy: 0.8890
21300/25000 [========================>.....] - ETA: 2s - loss: 0.4808 - accuracy: 0.8891
21400/25000 [========================>.....] - ETA: 2s - loss: 0.4805 - accuracy: 0.8894
21500/25000 [========================>.....] - ETA: 2s - loss: 0.4806 - accuracy: 0.8893
21600/25000 [========================>.....] - ETA: 2s - loss: 0.4806 - accuracy: 0.8892
21700/25000 [=========================>....] - ETA: 2s - loss: 0.4804 - accuracy: 0.8894
21800/25000 [=========================>....] - ETA: 1s - loss: 0.4802 - accuracy: 0.8896
21900/25000 [=========================>....] - ETA: 1s - loss: 0.4802 - accuracy: 0.8896
22000/25000 [=========================>....] - ETA: 1s - loss: 0.4802 - accuracy: 0.8895
22100/25000 [=========================>....] - ETA: 1s - loss: 0.4801 - accuracy: 0.8895
22200/25000 [=========================>....] - ETA: 1s - loss: 0.4801 - accuracy: 0.8895
22300/25000 [=========================>....] - ETA: 1s - loss: 0.4801 - accuracy: 0.8896
22400/25000 [=========================>....] - ETA: 1s - loss: 0.4800 - accuracy: 0.8896
22500/25000 [==========================>...] - ETA: 1s - loss: 0.4799 - accuracy: 0.8897
22600/25000 [==========================>...] - ETA: 1s - loss: 0.4797 - accuracy: 0.8899
22700/25000 [==========================>...] - ETA: 1s - loss: 0.4795 - accuracy: 0.8902
22800/25000 [==========================>...] - ETA: 1s - loss: 0.4795 - accuracy: 0.8901
22900/25000 [==========================>...] - ETA: 1s - loss: 0.4794 - accuracy: 0.8901
23000/25000 [==========================>...] - ETA: 1s - loss: 0.4792 - accuracy: 0.8903
23100/25000 [==========================>...] - ETA: 1s - loss: 0.4792 - accuracy: 0.8902
23200/25000 [==========================>...] - ETA: 1s - loss: 0.4791 - accuracy: 0.8903
23300/25000 [==========================>...] - ETA: 1s - loss: 0.4790 - accuracy: 0.8904
23400/25000 [===========================>..] - ETA: 0s - loss: 0.4790 - accuracy: 0.8903
23500/25000 [===========================>..] - ETA: 0s - loss: 0.4788 - accuracy: 0.8905
23600/25000 [===========================>..] - ETA: 0s - loss: 0.4787 - accuracy: 0.8906
23700/25000 [===========================>..] - ETA: 0s - loss: 0.4788 - accuracy: 0.8904
23800/25000 [===========================>..] - ETA: 0s - loss: 0.4788 - accuracy: 0.8904
23900/25000 [===========================>..] - ETA: 0s - loss: 0.4789 - accuracy: 0.8902
24000/25000 [===========================>..] - ETA: 0s - loss: 0.4789 - accuracy: 0.8903
24100/25000 [===========================>..] - ETA: 0s - loss: 0.4788 - accuracy: 0.8902
24200/25000 [============================>.] - ETA: 0s - loss: 0.4787 - accuracy: 0.8902
24300/25000 [============================>.] - ETA: 0s - loss: 0.4787 - accuracy: 0.8902
24400/25000 [============================>.] - ETA: 0s - loss: 0.4786 - accuracy: 0.8902
24500/25000 [============================>.] - ETA: 0s - loss: 0.4788 - accuracy: 0.8899
24600/25000 [============================>.] - ETA: 0s - loss: 0.4789 - accuracy: 0.8896
24700/25000 [============================>.] - ETA: 0s - loss: 0.4788 - accuracy: 0.8896
24800/25000 [============================>.] - ETA: 0s - loss: 0.4788 - accuracy: 0.8896
24900/25000 [============================>.] - ETA: 0s - loss: 0.4786 - accuracy: 0.8898
25000/25000 [==============================] - 19s 771us/step - loss: 0.4784 - accuracy: 0.8899 - val_loss: 0.4897 - val_accuracy: 0.8555
Epoch 5/10

  100/25000 [..............................] - ETA: 15s - loss: 0.4413 - accuracy: 0.9100
  200/25000 [..............................] - ETA: 15s - loss: 0.4460 - accuracy: 0.9050
  300/25000 [..............................] - ETA: 14s - loss: 0.4566 - accuracy: 0.8967
  400/25000 [..............................] - ETA: 14s - loss: 0.4545 - accuracy: 0.9050
  500/25000 [..............................] - ETA: 14s - loss: 0.4524 - accuracy: 0.9080
  600/25000 [..............................] - ETA: 14s - loss: 0.4499 - accuracy: 0.9100
  700/25000 [..............................] - ETA: 14s - loss: 0.4483 - accuracy: 0.9129
  800/25000 [..............................] - ETA: 14s - loss: 0.4508 - accuracy: 0.9100
  900/25000 [>.............................] - ETA: 14s - loss: 0.4500 - accuracy: 0.9089
 1000/25000 [>.............................] - ETA: 14s - loss: 0.4449 - accuracy: 0.9150
 1100/25000 [>.............................] - ETA: 14s - loss: 0.4464 - accuracy: 0.9127
 1200/25000 [>.............................] - ETA: 14s - loss: 0.4474 - accuracy: 0.9117
 1300/25000 [>.............................] - ETA: 14s - loss: 0.4460 - accuracy: 0.9131
 1400/25000 [>.............................] - ETA: 14s - loss: 0.4460 - accuracy: 0.9129
 1500/25000 [>.............................] - ETA: 14s - loss: 0.4466 - accuracy: 0.9133
 1600/25000 [>.............................] - ETA: 14s - loss: 0.4475 - accuracy: 0.9125
 1700/25000 [=>............................] - ETA: 14s - loss: 0.4459 - accuracy: 0.9147
 1800/25000 [=>............................] - ETA: 14s - loss: 0.4476 - accuracy: 0.9128
 1900/25000 [=>............................] - ETA: 13s - loss: 0.4474 - accuracy: 0.9126
 2000/25000 [=>............................] - ETA: 13s - loss: 0.4471 - accuracy: 0.9125
 2100/25000 [=>............................] - ETA: 13s - loss: 0.4472 - accuracy: 0.9124
 2200/25000 [=>............................] - ETA: 13s - loss: 0.4460 - accuracy: 0.9141
 2300/25000 [=>............................] - ETA: 13s - loss: 0.4451 - accuracy: 0.9148
 2400/25000 [=>............................] - ETA: 13s - loss: 0.4430 - accuracy: 0.9175
 2500/25000 [==>...........................] - ETA: 13s - loss: 0.4440 - accuracy: 0.9160
 2600/25000 [==>...........................] - ETA: 13s - loss: 0.4435 - accuracy: 0.9165
 2700/25000 [==>...........................] - ETA: 13s - loss: 0.4438 - accuracy: 0.9159
 2800/25000 [==>...........................] - ETA: 13s - loss: 0.4454 - accuracy: 0.9139
 2900/25000 [==>...........................] - ETA: 13s - loss: 0.4475 - accuracy: 0.9114
 3000/25000 [==>...........................] - ETA: 13s - loss: 0.4475 - accuracy: 0.9117
 3100/25000 [==>...........................] - ETA: 13s - loss: 0.4486 - accuracy: 0.9097
 3200/25000 [==>...........................] - ETA: 13s - loss: 0.4498 - accuracy: 0.9081
 3300/25000 [==>...........................] - ETA: 13s - loss: 0.4508 - accuracy: 0.9064
 3400/25000 [===>..........................] - ETA: 13s - loss: 0.4525 - accuracy: 0.9038
 3500/25000 [===>..........................] - ETA: 13s - loss: 0.4529 - accuracy: 0.9023
 3600/25000 [===>..........................] - ETA: 13s - loss: 0.4534 - accuracy: 0.9014
 3700/25000 [===>..........................] - ETA: 12s - loss: 0.4545 - accuracy: 0.9000
 3800/25000 [===>..........................] - ETA: 12s - loss: 0.4561 - accuracy: 0.8971
 3900/25000 [===>..........................] - ETA: 12s - loss: 0.4568 - accuracy: 0.8959
 4000/25000 [===>..........................] - ETA: 12s - loss: 0.4574 - accuracy: 0.8950
 4100/25000 [===>..........................] - ETA: 12s - loss: 0.4582 - accuracy: 0.8937
 4200/25000 [====>.........................] - ETA: 12s - loss: 0.4601 - accuracy: 0.8907
 4300/25000 [====>.........................] - ETA: 12s - loss: 0.4610 - accuracy: 0.8893
 4400/25000 [====>.........................] - ETA: 12s - loss: 0.4610 - accuracy: 0.8893
 4500/25000 [====>.........................] - ETA: 12s - loss: 0.4618 - accuracy: 0.8882
 4600/25000 [====>.........................] - ETA: 12s - loss: 0.4621 - accuracy: 0.8883
 4700/25000 [====>.........................] - ETA: 12s - loss: 0.4623 - accuracy: 0.8877
 4800/25000 [====>.........................] - ETA: 12s - loss: 0.4628 - accuracy: 0.8871
 4900/25000 [====>.........................] - ETA: 12s - loss: 0.4640 - accuracy: 0.8855
 5000/25000 [=====>........................] - ETA: 12s - loss: 0.4655 - accuracy: 0.8838
 5100/25000 [=====>........................] - ETA: 12s - loss: 0.4660 - accuracy: 0.8833
 5200/25000 [=====>........................] - ETA: 12s - loss: 0.4676 - accuracy: 0.8815
 5300/25000 [=====>........................] - ETA: 12s - loss: 0.4686 - accuracy: 0.8804
 5400/25000 [=====>........................] - ETA: 11s - loss: 0.4683 - accuracy: 0.8809
 5500/25000 [=====>........................] - ETA: 11s - loss: 0.4685 - accuracy: 0.8811
 5600/25000 [=====>........................] - ETA: 11s - loss: 0.4694 - accuracy: 0.8800
 5700/25000 [=====>........................] - ETA: 11s - loss: 0.4696 - accuracy: 0.8796
 5800/25000 [=====>........................] - ETA: 11s - loss: 0.4701 - accuracy: 0.8793
 5900/25000 [======>.......................] - ETA: 11s - loss: 0.4702 - accuracy: 0.8790
 6000/25000 [======>.......................] - ETA: 11s - loss: 0.4701 - accuracy: 0.8793
 6100/25000 [======>.......................] - ETA: 11s - loss: 0.4703 - accuracy: 0.8792
 6200/25000 [======>.......................] - ETA: 11s - loss: 0.4711 - accuracy: 0.8784
 6300/25000 [======>.......................] - ETA: 11s - loss: 0.4710 - accuracy: 0.8783
 6400/25000 [======>.......................] - ETA: 11s - loss: 0.4713 - accuracy: 0.8781
 6500/25000 [======>.......................] - ETA: 11s - loss: 0.4712 - accuracy: 0.8783
 6600/25000 [======>.......................] - ETA: 11s - loss: 0.4714 - accuracy: 0.8782
 6700/25000 [=======>......................] - ETA: 11s - loss: 0.4719 - accuracy: 0.8775
 6800/25000 [=======>......................] - ETA: 11s - loss: 0.4720 - accuracy: 0.8775
 6900/25000 [=======>......................] - ETA: 11s - loss: 0.4723 - accuracy: 0.8772
 7000/25000 [=======>......................] - ETA: 10s - loss: 0.4725 - accuracy: 0.8769
 7100/25000 [=======>......................] - ETA: 10s - loss: 0.4725 - accuracy: 0.8770
 7200/25000 [=======>......................] - ETA: 10s - loss: 0.4728 - accuracy: 0.8768
 7300/25000 [=======>......................] - ETA: 10s - loss: 0.4724 - accuracy: 0.8773
 7400/25000 [=======>......................] - ETA: 10s - loss: 0.4724 - accuracy: 0.8770
 7500/25000 [========>.....................] - ETA: 10s - loss: 0.4738 - accuracy: 0.8755
 7600/25000 [========>.....................] - ETA: 10s - loss: 0.4741 - accuracy: 0.8751
 7700/25000 [========>.....................] - ETA: 10s - loss: 0.4741 - accuracy: 0.8753
 7800/25000 [========>.....................] - ETA: 10s - loss: 0.4746 - accuracy: 0.8746
 7900/25000 [========>.....................] - ETA: 10s - loss: 0.4749 - accuracy: 0.8741
 8000/25000 [========>.....................] - ETA: 10s - loss: 0.4753 - accuracy: 0.8735
 8100/25000 [========>.....................] - ETA: 10s - loss: 0.4747 - accuracy: 0.8742
 8200/25000 [========>.....................] - ETA: 10s - loss: 0.4749 - accuracy: 0.8739
 8300/25000 [========>.....................] - ETA: 10s - loss: 0.4746 - accuracy: 0.8741
 8400/25000 [=========>....................] - ETA: 10s - loss: 0.4750 - accuracy: 0.8735
 8500/25000 [=========>....................] - ETA: 10s - loss: 0.4749 - accuracy: 0.8738
 8600/25000 [=========>....................] - ETA: 10s - loss: 0.4746 - accuracy: 0.8741
 8700/25000 [=========>....................] - ETA: 9s - loss: 0.4745 - accuracy: 0.8741 
 8800/25000 [=========>....................] - ETA: 9s - loss: 0.4745 - accuracy: 0.8741
 8900/25000 [=========>....................] - ETA: 9s - loss: 0.4744 - accuracy: 0.8742
 9000/25000 [=========>....................] - ETA: 9s - loss: 0.4744 - accuracy: 0.8740
 9100/25000 [=========>....................] - ETA: 9s - loss: 0.4740 - accuracy: 0.8743
 9200/25000 [==========>...................] - ETA: 9s - loss: 0.4736 - accuracy: 0.8749
 9300/25000 [==========>...................] - ETA: 9s - loss: 0.4730 - accuracy: 0.8756
 9400/25000 [==========>...................] - ETA: 9s - loss: 0.4728 - accuracy: 0.8759
 9500/25000 [==========>...................] - ETA: 9s - loss: 0.4729 - accuracy: 0.8756
 9600/25000 [==========>...................] - ETA: 9s - loss: 0.4728 - accuracy: 0.8755
 9700/25000 [==========>...................] - ETA: 9s - loss: 0.4726 - accuracy: 0.8759
 9800/25000 [==========>...................] - ETA: 9s - loss: 0.4723 - accuracy: 0.8761
 9900/25000 [==========>...................] - ETA: 9s - loss: 0.4723 - accuracy: 0.8761
10000/25000 [===========>..................] - ETA: 9s - loss: 0.4720 - accuracy: 0.8764
10100/25000 [===========>..................] - ETA: 9s - loss: 0.4721 - accuracy: 0.8763
10200/25000 [===========>..................] - ETA: 9s - loss: 0.4721 - accuracy: 0.8763
10300/25000 [===========>..................] - ETA: 8s - loss: 0.4715 - accuracy: 0.8770
10400/25000 [===========>..................] - ETA: 8s - loss: 0.4714 - accuracy: 0.8768
10500/25000 [===========>..................] - ETA: 8s - loss: 0.4716 - accuracy: 0.8767
10600/25000 [===========>..................] - ETA: 8s - loss: 0.4714 - accuracy: 0.8771
10700/25000 [===========>..................] - ETA: 8s - loss: 0.4709 - accuracy: 0.8777
10800/25000 [===========>..................] - ETA: 8s - loss: 0.4705 - accuracy: 0.8781
10900/25000 [============>.................] - ETA: 8s - loss: 0.4706 - accuracy: 0.8776
11000/25000 [============>.................] - ETA: 8s - loss: 0.4704 - accuracy: 0.8778
11100/25000 [============>.................] - ETA: 8s - loss: 0.4701 - accuracy: 0.8781
11200/25000 [============>.................] - ETA: 8s - loss: 0.4698 - accuracy: 0.8784
11300/25000 [============>.................] - ETA: 8s - loss: 0.4698 - accuracy: 0.8783
11400/25000 [============>.................] - ETA: 8s - loss: 0.4698 - accuracy: 0.8782
11500/25000 [============>.................] - ETA: 8s - loss: 0.4700 - accuracy: 0.8779
11600/25000 [============>.................] - ETA: 8s - loss: 0.4699 - accuracy: 0.8778
11700/25000 [=============>................] - ETA: 8s - loss: 0.4697 - accuracy: 0.8780
11800/25000 [=============>................] - ETA: 8s - loss: 0.4698 - accuracy: 0.8780
11900/25000 [=============>................] - ETA: 7s - loss: 0.4696 - accuracy: 0.8782
12000/25000 [=============>................] - ETA: 7s - loss: 0.4696 - accuracy: 0.8782
12100/25000 [=============>................] - ETA: 7s - loss: 0.4696 - accuracy: 0.8779
12200/25000 [=============>................] - ETA: 7s - loss: 0.4692 - accuracy: 0.8784
12300/25000 [=============>................] - ETA: 7s - loss: 0.4691 - accuracy: 0.8784
12400/25000 [=============>................] - ETA: 7s - loss: 0.4689 - accuracy: 0.8784
12500/25000 [==============>...............] - ETA: 7s - loss: 0.4692 - accuracy: 0.8779
12600/25000 [==============>...............] - ETA: 7s - loss: 0.4693 - accuracy: 0.8778
12700/25000 [==============>...............] - ETA: 7s - loss: 0.4692 - accuracy: 0.8779
12800/25000 [==============>...............] - ETA: 7s - loss: 0.4695 - accuracy: 0.8774
12900/25000 [==============>...............] - ETA: 7s - loss: 0.4692 - accuracy: 0.8778
13000/25000 [==============>...............] - ETA: 7s - loss: 0.4691 - accuracy: 0.8778
13100/25000 [==============>...............] - ETA: 7s - loss: 0.4688 - accuracy: 0.8782
13200/25000 [==============>...............] - ETA: 7s - loss: 0.4684 - accuracy: 0.8785
13300/25000 [==============>...............] - ETA: 7s - loss: 0.4682 - accuracy: 0.8788
13400/25000 [===============>..............] - ETA: 7s - loss: 0.4678 - accuracy: 0.8792
13500/25000 [===============>..............] - ETA: 7s - loss: 0.4676 - accuracy: 0.8794
13600/25000 [===============>..............] - ETA: 6s - loss: 0.4674 - accuracy: 0.8795
13700/25000 [===============>..............] - ETA: 6s - loss: 0.4672 - accuracy: 0.8796
13800/25000 [===============>..............] - ETA: 6s - loss: 0.4670 - accuracy: 0.8798
13900/25000 [===============>..............] - ETA: 6s - loss: 0.4667 - accuracy: 0.8801
14000/25000 [===============>..............] - ETA: 6s - loss: 0.4663 - accuracy: 0.8806
14100/25000 [===============>..............] - ETA: 6s - loss: 0.4661 - accuracy: 0.8808
14200/25000 [================>.............] - ETA: 6s - loss: 0.4658 - accuracy: 0.8811
14300/25000 [================>.............] - ETA: 6s - loss: 0.4660 - accuracy: 0.8808
14400/25000 [================>.............] - ETA: 6s - loss: 0.4660 - accuracy: 0.8808
14500/25000 [================>.............] - ETA: 6s - loss: 0.4658 - accuracy: 0.8809
14600/25000 [================>.............] - ETA: 6s - loss: 0.4656 - accuracy: 0.8812
14700/25000 [================>.............] - ETA: 6s - loss: 0.4654 - accuracy: 0.8814
14800/25000 [================>.............] - ETA: 6s - loss: 0.4652 - accuracy: 0.8817
14900/25000 [================>.............] - ETA: 6s - loss: 0.4651 - accuracy: 0.8818
15000/25000 [=================>............] - ETA: 6s - loss: 0.4651 - accuracy: 0.8817
15100/25000 [=================>............] - ETA: 6s - loss: 0.4647 - accuracy: 0.8821
15200/25000 [=================>............] - ETA: 5s - loss: 0.4648 - accuracy: 0.8819
15300/25000 [=================>............] - ETA: 5s - loss: 0.4645 - accuracy: 0.8823
15400/25000 [=================>............] - ETA: 5s - loss: 0.4645 - accuracy: 0.8820
15500/25000 [=================>............] - ETA: 5s - loss: 0.4645 - accuracy: 0.8820
15600/25000 [=================>............] - ETA: 5s - loss: 0.4642 - accuracy: 0.8824
15700/25000 [=================>............] - ETA: 5s - loss: 0.4642 - accuracy: 0.8822
15800/25000 [=================>............] - ETA: 5s - loss: 0.4642 - accuracy: 0.8822
15900/25000 [==================>...........] - ETA: 5s - loss: 0.4642 - accuracy: 0.8820
16000/25000 [==================>...........] - ETA: 5s - loss: 0.4639 - accuracy: 0.8824
16100/25000 [==================>...........] - ETA: 5s - loss: 0.4636 - accuracy: 0.8827
16200/25000 [==================>...........] - ETA: 5s - loss: 0.4634 - accuracy: 0.8828
16300/25000 [==================>...........] - ETA: 5s - loss: 0.4632 - accuracy: 0.8831
16400/25000 [==================>...........] - ETA: 5s - loss: 0.4631 - accuracy: 0.8830
16500/25000 [==================>...........] - ETA: 5s - loss: 0.4629 - accuracy: 0.8832
16600/25000 [==================>...........] - ETA: 5s - loss: 0.4627 - accuracy: 0.8835
16700/25000 [===================>..........] - ETA: 5s - loss: 0.4627 - accuracy: 0.8834
16800/25000 [===================>..........] - ETA: 4s - loss: 0.4625 - accuracy: 0.8835
16900/25000 [===================>..........] - ETA: 4s - loss: 0.4624 - accuracy: 0.8837
17000/25000 [===================>..........] - ETA: 4s - loss: 0.4625 - accuracy: 0.8835
17100/25000 [===================>..........] - ETA: 4s - loss: 0.4624 - accuracy: 0.8836
17200/25000 [===================>..........] - ETA: 4s - loss: 0.4624 - accuracy: 0.8834
17300/25000 [===================>..........] - ETA: 4s - loss: 0.4623 - accuracy: 0.8836
17400/25000 [===================>..........] - ETA: 4s - loss: 0.4620 - accuracy: 0.8838
17500/25000 [====================>.........] - ETA: 4s - loss: 0.4620 - accuracy: 0.8837
17600/25000 [====================>.........] - ETA: 4s - loss: 0.4619 - accuracy: 0.8838
17700/25000 [====================>.........] - ETA: 4s - loss: 0.4618 - accuracy: 0.8840
17800/25000 [====================>.........] - ETA: 4s - loss: 0.4616 - accuracy: 0.8841
17900/25000 [====================>.........] - ETA: 4s - loss: 0.4617 - accuracy: 0.8840
18000/25000 [====================>.........] - ETA: 4s - loss: 0.4614 - accuracy: 0.8843
18100/25000 [====================>.........] - ETA: 4s - loss: 0.4612 - accuracy: 0.8845
18200/25000 [====================>.........] - ETA: 4s - loss: 0.4611 - accuracy: 0.8847
18300/25000 [====================>.........] - ETA: 4s - loss: 0.4609 - accuracy: 0.8849
18400/25000 [=====================>........] - ETA: 4s - loss: 0.4607 - accuracy: 0.8849
18500/25000 [=====================>........] - ETA: 3s - loss: 0.4607 - accuracy: 0.8849
18600/25000 [=====================>........] - ETA: 3s - loss: 0.4605 - accuracy: 0.8850
18700/25000 [=====================>........] - ETA: 3s - loss: 0.4607 - accuracy: 0.8847
18800/25000 [=====================>........] - ETA: 3s - loss: 0.4605 - accuracy: 0.8849
18900/25000 [=====================>........] - ETA: 3s - loss: 0.4604 - accuracy: 0.8849
19000/25000 [=====================>........] - ETA: 3s - loss: 0.4601 - accuracy: 0.8852
19100/25000 [=====================>........] - ETA: 3s - loss: 0.4602 - accuracy: 0.8851
19200/25000 [======================>.......] - ETA: 3s - loss: 0.4602 - accuracy: 0.8850
19300/25000 [======================>.......] - ETA: 3s - loss: 0.4600 - accuracy: 0.8853
19400/25000 [======================>.......] - ETA: 3s - loss: 0.4601 - accuracy: 0.8850
19500/25000 [======================>.......] - ETA: 3s - loss: 0.4602 - accuracy: 0.8848
19600/25000 [======================>.......] - ETA: 3s - loss: 0.4600 - accuracy: 0.8848
19700/25000 [======================>.......] - ETA: 3s - loss: 0.4599 - accuracy: 0.8850
19800/25000 [======================>.......] - ETA: 3s - loss: 0.4597 - accuracy: 0.8851
19900/25000 [======================>.......] - ETA: 3s - loss: 0.4598 - accuracy: 0.8849
20000/25000 [=======================>......] - ETA: 3s - loss: 0.4598 - accuracy: 0.8849
20100/25000 [=======================>......] - ETA: 2s - loss: 0.4596 - accuracy: 0.8850
20200/25000 [=======================>......] - ETA: 2s - loss: 0.4595 - accuracy: 0.8850
20300/25000 [=======================>......] - ETA: 2s - loss: 0.4593 - accuracy: 0.8852
20400/25000 [=======================>......] - ETA: 2s - loss: 0.4591 - accuracy: 0.8854
20500/25000 [=======================>......] - ETA: 2s - loss: 0.4588 - accuracy: 0.8857
20600/25000 [=======================>......] - ETA: 2s - loss: 0.4588 - accuracy: 0.8857
20700/25000 [=======================>......] - ETA: 2s - loss: 0.4585 - accuracy: 0.8860
20800/25000 [=======================>......] - ETA: 2s - loss: 0.4584 - accuracy: 0.8862
20900/25000 [========================>.....] - ETA: 2s - loss: 0.4581 - accuracy: 0.8865
21000/25000 [========================>.....] - ETA: 2s - loss: 0.4579 - accuracy: 0.8867
21100/25000 [========================>.....] - ETA: 2s - loss: 0.4579 - accuracy: 0.8865
21200/25000 [========================>.....] - ETA: 2s - loss: 0.4579 - accuracy: 0.8866
21300/25000 [========================>.....] - ETA: 2s - loss: 0.4578 - accuracy: 0.8865
21400/25000 [========================>.....] - ETA: 2s - loss: 0.4577 - accuracy: 0.8865
21500/25000 [========================>.....] - ETA: 2s - loss: 0.4577 - accuracy: 0.8865
21600/25000 [========================>.....] - ETA: 2s - loss: 0.4579 - accuracy: 0.8863
21700/25000 [=========================>....] - ETA: 2s - loss: 0.4579 - accuracy: 0.8861
21800/25000 [=========================>....] - ETA: 1s - loss: 0.4578 - accuracy: 0.8861
21900/25000 [=========================>....] - ETA: 1s - loss: 0.4577 - accuracy: 0.8862
22000/25000 [=========================>....] - ETA: 1s - loss: 0.4577 - accuracy: 0.8862
22100/25000 [=========================>....] - ETA: 1s - loss: 0.4575 - accuracy: 0.8862
22200/25000 [=========================>....] - ETA: 1s - loss: 0.4572 - accuracy: 0.8865
22300/25000 [=========================>....] - ETA: 1s - loss: 0.4574 - accuracy: 0.8862
22400/25000 [=========================>....] - ETA: 1s - loss: 0.4572 - accuracy: 0.8863
22500/25000 [==========================>...] - ETA: 1s - loss: 0.4571 - accuracy: 0.8865
22600/25000 [==========================>...] - ETA: 1s - loss: 0.4571 - accuracy: 0.8865
22700/25000 [==========================>...] - ETA: 1s - loss: 0.4568 - accuracy: 0.8867
22800/25000 [==========================>...] - ETA: 1s - loss: 0.4566 - accuracy: 0.8869
22900/25000 [==========================>...] - ETA: 1s - loss: 0.4565 - accuracy: 0.8871
23000/25000 [==========================>...] - ETA: 1s - loss: 0.4564 - accuracy: 0.8870
23100/25000 [==========================>...] - ETA: 1s - loss: 0.4564 - accuracy: 0.8870
23200/25000 [==========================>...] - ETA: 1s - loss: 0.4564 - accuracy: 0.8869
23300/25000 [==========================>...] - ETA: 1s - loss: 0.4564 - accuracy: 0.8868
23400/25000 [===========================>..] - ETA: 0s - loss: 0.4563 - accuracy: 0.8868
23500/25000 [===========================>..] - ETA: 0s - loss: 0.4561 - accuracy: 0.8870
23600/25000 [===========================>..] - ETA: 0s - loss: 0.4562 - accuracy: 0.8867
23700/25000 [===========================>..] - ETA: 0s - loss: 0.4561 - accuracy: 0.8868
23800/25000 [===========================>..] - ETA: 0s - loss: 0.4558 - accuracy: 0.8869
23900/25000 [===========================>..] - ETA: 0s - loss: 0.4557 - accuracy: 0.8871
24000/25000 [===========================>..] - ETA: 0s - loss: 0.4554 - accuracy: 0.8874
24100/25000 [===========================>..] - ETA: 0s - loss: 0.4553 - accuracy: 0.8875
24200/25000 [============================>.] - ETA: 0s - loss: 0.4555 - accuracy: 0.8872
24300/25000 [============================>.] - ETA: 0s - loss: 0.4553 - accuracy: 0.8873
24400/25000 [============================>.] - ETA: 0s - loss: 0.4551 - accuracy: 0.8875
24500/25000 [============================>.] - ETA: 0s - loss: 0.4552 - accuracy: 0.8873
24600/25000 [============================>.] - ETA: 0s - loss: 0.4552 - accuracy: 0.8873
24700/25000 [============================>.] - ETA: 0s - loss: 0.4551 - accuracy: 0.8874
24800/25000 [============================>.] - ETA: 0s - loss: 0.4550 - accuracy: 0.8874
24900/25000 [============================>.] - ETA: 0s - loss: 0.4548 - accuracy: 0.8876
25000/25000 [==============================] - 19s 769us/step - loss: 0.4547 - accuracy: 0.8878 - val_loss: 0.4658 - val_accuracy: 0.8594
Epoch 6/10

  100/25000 [..............................] - ETA: 15s - loss: 0.4152 - accuracy: 0.9300
  200/25000 [..............................] - ETA: 14s - loss: 0.4088 - accuracy: 0.9350
  300/25000 [..............................] - ETA: 14s - loss: 0.4088 - accuracy: 0.9333
  400/25000 [..............................] - ETA: 14s - loss: 0.4099 - accuracy: 0.9325
  500/25000 [..............................] - ETA: 14s - loss: 0.4129 - accuracy: 0.9280
  600/25000 [..............................] - ETA: 14s - loss: 0.4124 - accuracy: 0.9267
  700/25000 [..............................] - ETA: 14s - loss: 0.4129 - accuracy: 0.9257
  800/25000 [..............................] - ETA: 14s - loss: 0.4154 - accuracy: 0.9225
  900/25000 [>.............................] - ETA: 14s - loss: 0.4114 - accuracy: 0.9256
 1000/25000 [>.............................] - ETA: 14s - loss: 0.4141 - accuracy: 0.9230
 1100/25000 [>.............................] - ETA: 14s - loss: 0.4171 - accuracy: 0.9182
 1200/25000 [>.............................] - ETA: 14s - loss: 0.4183 - accuracy: 0.9158
 1300/25000 [>.............................] - ETA: 14s - loss: 0.4174 - accuracy: 0.9169
 1400/25000 [>.............................] - ETA: 14s - loss: 0.4190 - accuracy: 0.9143
 1500/25000 [>.............................] - ETA: 14s - loss: 0.4188 - accuracy: 0.9147
 1600/25000 [>.............................] - ETA: 14s - loss: 0.4202 - accuracy: 0.9131
 1700/25000 [=>............................] - ETA: 14s - loss: 0.4200 - accuracy: 0.9129
 1800/25000 [=>............................] - ETA: 14s - loss: 0.4182 - accuracy: 0.9144
 1900/25000 [=>............................] - ETA: 14s - loss: 0.4202 - accuracy: 0.9111
 2000/25000 [=>............................] - ETA: 14s - loss: 0.4214 - accuracy: 0.9105
 2100/25000 [=>............................] - ETA: 13s - loss: 0.4198 - accuracy: 0.9119
 2200/25000 [=>............................] - ETA: 13s - loss: 0.4197 - accuracy: 0.9114
 2300/25000 [=>............................] - ETA: 13s - loss: 0.4202 - accuracy: 0.9109
 2400/25000 [=>............................] - ETA: 13s - loss: 0.4200 - accuracy: 0.9104
 2500/25000 [==>...........................] - ETA: 13s - loss: 0.4195 - accuracy: 0.9108
 2600/25000 [==>...........................] - ETA: 13s - loss: 0.4198 - accuracy: 0.9108
 2700/25000 [==>...........................] - ETA: 13s - loss: 0.4193 - accuracy: 0.9107
 2800/25000 [==>...........................] - ETA: 13s - loss: 0.4181 - accuracy: 0.9121
 2900/25000 [==>...........................] - ETA: 13s - loss: 0.4177 - accuracy: 0.9128
 3000/25000 [==>...........................] - ETA: 13s - loss: 0.4173 - accuracy: 0.9133
 3100/25000 [==>...........................] - ETA: 13s - loss: 0.4171 - accuracy: 0.9129
 3200/25000 [==>...........................] - ETA: 13s - loss: 0.4174 - accuracy: 0.9122
 3300/25000 [==>...........................] - ETA: 13s - loss: 0.4173 - accuracy: 0.9121
 3400/25000 [===>..........................] - ETA: 13s - loss: 0.4169 - accuracy: 0.9126
 3500/25000 [===>..........................] - ETA: 13s - loss: 0.4163 - accuracy: 0.9137
 3600/25000 [===>..........................] - ETA: 13s - loss: 0.4164 - accuracy: 0.9142
 3700/25000 [===>..........................] - ETA: 13s - loss: 0.4164 - accuracy: 0.9143
 3800/25000 [===>..........................] - ETA: 12s - loss: 0.4180 - accuracy: 0.9126
 3900/25000 [===>..........................] - ETA: 12s - loss: 0.4185 - accuracy: 0.9121
 4000/25000 [===>..........................] - ETA: 12s - loss: 0.4196 - accuracy: 0.9107
 4100/25000 [===>..........................] - ETA: 12s - loss: 0.4194 - accuracy: 0.9110
 4200/25000 [====>.........................] - ETA: 12s - loss: 0.4192 - accuracy: 0.9114
 4300/25000 [====>.........................] - ETA: 12s - loss: 0.4197 - accuracy: 0.9105
 4400/25000 [====>.........................] - ETA: 12s - loss: 0.4203 - accuracy: 0.9100
 4500/25000 [====>.........................] - ETA: 12s - loss: 0.4203 - accuracy: 0.9102
 4600/25000 [====>.........................] - ETA: 12s - loss: 0.4210 - accuracy: 0.9096
 4700/25000 [====>.........................] - ETA: 12s - loss: 0.4215 - accuracy: 0.9091
 4800/25000 [====>.........................] - ETA: 12s - loss: 0.4221 - accuracy: 0.9083
 4900/25000 [====>.........................] - ETA: 12s - loss: 0.4220 - accuracy: 0.9082
 5000/25000 [=====>........................] - ETA: 12s - loss: 0.4212 - accuracy: 0.9092
 5100/25000 [=====>........................] - ETA: 12s - loss: 0.4210 - accuracy: 0.9092
 5200/25000 [=====>........................] - ETA: 12s - loss: 0.4205 - accuracy: 0.9096
 5300/25000 [=====>........................] - ETA: 12s - loss: 0.4213 - accuracy: 0.9089
 5400/25000 [=====>........................] - ETA: 11s - loss: 0.4212 - accuracy: 0.9089
 5500/25000 [=====>........................] - ETA: 11s - loss: 0.4213 - accuracy: 0.9089
 5600/25000 [=====>........................] - ETA: 11s - loss: 0.4210 - accuracy: 0.9093
 5700/25000 [=====>........................] - ETA: 11s - loss: 0.4209 - accuracy: 0.9095
 5800/25000 [=====>........................] - ETA: 11s - loss: 0.4210 - accuracy: 0.9095
 5900/25000 [======>.......................] - ETA: 11s - loss: 0.4215 - accuracy: 0.9088
 6000/25000 [======>.......................] - ETA: 11s - loss: 0.4216 - accuracy: 0.9087
 6100/25000 [======>.......................] - ETA: 11s - loss: 0.4215 - accuracy: 0.9089
 6200/25000 [======>.......................] - ETA: 11s - loss: 0.4216 - accuracy: 0.9087
 6300/25000 [======>.......................] - ETA: 11s - loss: 0.4219 - accuracy: 0.9081
 6400/25000 [======>.......................] - ETA: 11s - loss: 0.4216 - accuracy: 0.9087
 6500/25000 [======>.......................] - ETA: 11s - loss: 0.4216 - accuracy: 0.9086
 6600/25000 [======>.......................] - ETA: 11s - loss: 0.4218 - accuracy: 0.9083
 6700/25000 [=======>......................] - ETA: 11s - loss: 0.4213 - accuracy: 0.9090
 6800/25000 [=======>......................] - ETA: 11s - loss: 0.4206 - accuracy: 0.9097
 6900/25000 [=======>......................] - ETA: 11s - loss: 0.4209 - accuracy: 0.9091
 7000/25000 [=======>......................] - ETA: 10s - loss: 0.4210 - accuracy: 0.9091
 7100/25000 [=======>......................] - ETA: 10s - loss: 0.4208 - accuracy: 0.9094
 7200/25000 [=======>......................] - ETA: 10s - loss: 0.4210 - accuracy: 0.9092
 7300/25000 [=======>......................] - ETA: 10s - loss: 0.4213 - accuracy: 0.9088
 7400/25000 [=======>......................] - ETA: 10s - loss: 0.4210 - accuracy: 0.9092
 7500/25000 [========>.....................] - ETA: 10s - loss: 0.4209 - accuracy: 0.9092
 7600/25000 [========>.....................] - ETA: 10s - loss: 0.4204 - accuracy: 0.9097
 7700/25000 [========>.....................] - ETA: 10s - loss: 0.4200 - accuracy: 0.9103
 7800/25000 [========>.....................] - ETA: 10s - loss: 0.4201 - accuracy: 0.9101
 7900/25000 [========>.....................] - ETA: 10s - loss: 0.4198 - accuracy: 0.9103
 8000/25000 [========>.....................] - ETA: 10s - loss: 0.4196 - accuracy: 0.9104
 8100/25000 [========>.....................] - ETA: 10s - loss: 0.4199 - accuracy: 0.9101
 8200/25000 [========>.....................] - ETA: 10s - loss: 0.4196 - accuracy: 0.9106
 8300/25000 [========>.....................] - ETA: 10s - loss: 0.4200 - accuracy: 0.9101
 8400/25000 [=========>....................] - ETA: 10s - loss: 0.4197 - accuracy: 0.9105
 8500/25000 [=========>....................] - ETA: 10s - loss: 0.4197 - accuracy: 0.9104
 8600/25000 [=========>....................] - ETA: 10s - loss: 0.4194 - accuracy: 0.9106
 8700/25000 [=========>....................] - ETA: 9s - loss: 0.4192 - accuracy: 0.9109 
 8800/25000 [=========>....................] - ETA: 9s - loss: 0.4191 - accuracy: 0.9110
 8900/25000 [=========>....................] - ETA: 9s - loss: 0.4189 - accuracy: 0.9111
 9000/25000 [=========>....................] - ETA: 9s - loss: 0.4187 - accuracy: 0.9114
 9100/25000 [=========>....................] - ETA: 9s - loss: 0.4184 - accuracy: 0.9116
 9200/25000 [==========>...................] - ETA: 9s - loss: 0.4181 - accuracy: 0.9120
 9300/25000 [==========>...................] - ETA: 9s - loss: 0.4181 - accuracy: 0.9120
 9400/25000 [==========>...................] - ETA: 9s - loss: 0.4177 - accuracy: 0.9123
 9500/25000 [==========>...................] - ETA: 9s - loss: 0.4173 - accuracy: 0.9127
 9600/25000 [==========>...................] - ETA: 9s - loss: 0.4172 - accuracy: 0.9129
 9700/25000 [==========>...................] - ETA: 9s - loss: 0.4171 - accuracy: 0.9130
 9800/25000 [==========>...................] - ETA: 9s - loss: 0.4172 - accuracy: 0.9129
 9900/25000 [==========>...................] - ETA: 9s - loss: 0.4168 - accuracy: 0.9131
10000/25000 [===========>..................] - ETA: 9s - loss: 0.4169 - accuracy: 0.9130
10100/25000 [===========>..................] - ETA: 9s - loss: 0.4170 - accuracy: 0.9128
10200/25000 [===========>..................] - ETA: 9s - loss: 0.4165 - accuracy: 0.9131
10300/25000 [===========>..................] - ETA: 8s - loss: 0.4163 - accuracy: 0.9131
10400/25000 [===========>..................] - ETA: 8s - loss: 0.4160 - accuracy: 0.9135
10500/25000 [===========>..................] - ETA: 8s - loss: 0.4158 - accuracy: 0.9137
10600/25000 [===========>..................] - ETA: 8s - loss: 0.4155 - accuracy: 0.9138
10700/25000 [===========>..................] - ETA: 8s - loss: 0.4159 - accuracy: 0.9132
10800/25000 [===========>..................] - ETA: 8s - loss: 0.4157 - accuracy: 0.9133
10900/25000 [============>.................] - ETA: 8s - loss: 0.4155 - accuracy: 0.9136
11000/25000 [============>.................] - ETA: 8s - loss: 0.4155 - accuracy: 0.9135
11100/25000 [============>.................] - ETA: 8s - loss: 0.4156 - accuracy: 0.9133
11200/25000 [============>.................] - ETA: 8s - loss: 0.4158 - accuracy: 0.9129
11300/25000 [============>.................] - ETA: 8s - loss: 0.4158 - accuracy: 0.9128
11400/25000 [============>.................] - ETA: 8s - loss: 0.4158 - accuracy: 0.9128
11500/25000 [============>.................] - ETA: 8s - loss: 0.4155 - accuracy: 0.9131
11600/25000 [============>.................] - ETA: 8s - loss: 0.4153 - accuracy: 0.9133
11700/25000 [=============>................] - ETA: 8s - loss: 0.4154 - accuracy: 0.9132
11800/25000 [=============>................] - ETA: 8s - loss: 0.4155 - accuracy: 0.9131
11900/25000 [=============>................] - ETA: 7s - loss: 0.4150 - accuracy: 0.9135
12000/25000 [=============>................] - ETA: 7s - loss: 0.4146 - accuracy: 0.9139
12100/25000 [=============>................] - ETA: 7s - loss: 0.4143 - accuracy: 0.9143
12200/25000 [=============>................] - ETA: 7s - loss: 0.4141 - accuracy: 0.9143
12300/25000 [=============>................] - ETA: 7s - loss: 0.4140 - accuracy: 0.9146
12400/25000 [=============>................] - ETA: 7s - loss: 0.4142 - accuracy: 0.9143
12500/25000 [==============>...............] - ETA: 7s - loss: 0.4142 - accuracy: 0.9142
12600/25000 [==============>...............] - ETA: 7s - loss: 0.4144 - accuracy: 0.9137
12700/25000 [==============>...............] - ETA: 7s - loss: 0.4143 - accuracy: 0.9138
12800/25000 [==============>...............] - ETA: 7s - loss: 0.4141 - accuracy: 0.9138
12900/25000 [==============>...............] - ETA: 7s - loss: 0.4138 - accuracy: 0.9141
13000/25000 [==============>...............] - ETA: 7s - loss: 0.4136 - accuracy: 0.9142
13100/25000 [==============>...............] - ETA: 7s - loss: 0.4137 - accuracy: 0.9142
13200/25000 [==============>...............] - ETA: 7s - loss: 0.4139 - accuracy: 0.9139
13300/25000 [==============>...............] - ETA: 7s - loss: 0.4138 - accuracy: 0.9140
13400/25000 [===============>..............] - ETA: 7s - loss: 0.4138 - accuracy: 0.9140
13500/25000 [===============>..............] - ETA: 7s - loss: 0.4138 - accuracy: 0.9139
13600/25000 [===============>..............] - ETA: 6s - loss: 0.4138 - accuracy: 0.9138
13700/25000 [===============>..............] - ETA: 6s - loss: 0.4139 - accuracy: 0.9137
13800/25000 [===============>..............] - ETA: 6s - loss: 0.4139 - accuracy: 0.9137
13900/25000 [===============>..............] - ETA: 6s - loss: 0.4136 - accuracy: 0.9140
14000/25000 [===============>..............] - ETA: 6s - loss: 0.4134 - accuracy: 0.9141
14100/25000 [===============>..............] - ETA: 6s - loss: 0.4135 - accuracy: 0.9140
14200/25000 [================>.............] - ETA: 6s - loss: 0.4131 - accuracy: 0.9143
14300/25000 [================>.............] - ETA: 6s - loss: 0.4131 - accuracy: 0.9142
14400/25000 [================>.............] - ETA: 6s - loss: 0.4128 - accuracy: 0.9145
14500/25000 [================>.............] - ETA: 6s - loss: 0.4129 - accuracy: 0.9145
14600/25000 [================>.............] - ETA: 6s - loss: 0.4130 - accuracy: 0.9143
14700/25000 [================>.............] - ETA: 6s - loss: 0.4129 - accuracy: 0.9144
14800/25000 [================>.............] - ETA: 6s - loss: 0.4129 - accuracy: 0.9143
14900/25000 [================>.............] - ETA: 6s - loss: 0.4127 - accuracy: 0.9146
15000/25000 [=================>............] - ETA: 6s - loss: 0.4126 - accuracy: 0.9147
15100/25000 [=================>............] - ETA: 6s - loss: 0.4126 - accuracy: 0.9146
15200/25000 [=================>............] - ETA: 5s - loss: 0.4126 - accuracy: 0.9147
15300/25000 [=================>............] - ETA: 5s - loss: 0.4127 - accuracy: 0.9144
15400/25000 [=================>............] - ETA: 5s - loss: 0.4127 - accuracy: 0.9144
15500/25000 [=================>............] - ETA: 5s - loss: 0.4125 - accuracy: 0.9145
15600/25000 [=================>............] - ETA: 5s - loss: 0.4124 - accuracy: 0.9146
15700/25000 [=================>............] - ETA: 5s - loss: 0.4122 - accuracy: 0.9147
15800/25000 [=================>............] - ETA: 5s - loss: 0.4121 - accuracy: 0.9147
15900/25000 [==================>...........] - ETA: 5s - loss: 0.4122 - accuracy: 0.9145
16000/25000 [==================>...........] - ETA: 5s - loss: 0.4120 - accuracy: 0.9146
16100/25000 [==================>...........] - ETA: 5s - loss: 0.4119 - accuracy: 0.9147
16200/25000 [==================>...........] - ETA: 5s - loss: 0.4118 - accuracy: 0.9148
16300/25000 [==================>...........] - ETA: 5s - loss: 0.4120 - accuracy: 0.9145
16400/25000 [==================>...........] - ETA: 5s - loss: 0.4118 - accuracy: 0.9147
16500/25000 [==================>...........] - ETA: 5s - loss: 0.4117 - accuracy: 0.9147
16600/25000 [==================>...........] - ETA: 5s - loss: 0.4116 - accuracy: 0.9147
16700/25000 [===================>..........] - ETA: 5s - loss: 0.4114 - accuracy: 0.9149
16800/25000 [===================>..........] - ETA: 4s - loss: 0.4112 - accuracy: 0.9151
16900/25000 [===================>..........] - ETA: 4s - loss: 0.4111 - accuracy: 0.9152
17000/25000 [===================>..........] - ETA: 4s - loss: 0.4109 - accuracy: 0.9154
17100/25000 [===================>..........] - ETA: 4s - loss: 0.4110 - accuracy: 0.9153
17200/25000 [===================>..........] - ETA: 4s - loss: 0.4111 - accuracy: 0.9152
17300/25000 [===================>..........] - ETA: 4s - loss: 0.4109 - accuracy: 0.9153
17400/25000 [===================>..........] - ETA: 4s - loss: 0.4109 - accuracy: 0.9153
17500/25000 [====================>.........] - ETA: 4s - loss: 0.4106 - accuracy: 0.9157
17600/25000 [====================>.........] - ETA: 4s - loss: 0.4108 - accuracy: 0.9153
17700/25000 [====================>.........] - ETA: 4s - loss: 0.4108 - accuracy: 0.9153
17800/25000 [====================>.........] - ETA: 4s - loss: 0.4109 - accuracy: 0.9152
17900/25000 [====================>.........] - ETA: 4s - loss: 0.4109 - accuracy: 0.9150
18000/25000 [====================>.........] - ETA: 4s - loss: 0.4109 - accuracy: 0.9151
18100/25000 [====================>.........] - ETA: 4s - loss: 0.4108 - accuracy: 0.9151
18200/25000 [====================>.........] - ETA: 4s - loss: 0.4108 - accuracy: 0.9152
18300/25000 [====================>.........] - ETA: 4s - loss: 0.4107 - accuracy: 0.9153
18400/25000 [=====================>........] - ETA: 4s - loss: 0.4105 - accuracy: 0.9153
18500/25000 [=====================>........] - ETA: 3s - loss: 0.4105 - accuracy: 0.9154
18600/25000 [=====================>........] - ETA: 3s - loss: 0.4106 - accuracy: 0.9152
18700/25000 [=====================>........] - ETA: 3s - loss: 0.4107 - accuracy: 0.9151
18800/25000 [=====================>........] - ETA: 3s - loss: 0.4104 - accuracy: 0.9153
18900/25000 [=====================>........] - ETA: 3s - loss: 0.4103 - accuracy: 0.9153
19000/25000 [=====================>........] - ETA: 3s - loss: 0.4102 - accuracy: 0.9154
19100/25000 [=====================>........] - ETA: 3s - loss: 0.4101 - accuracy: 0.9154
19200/25000 [======================>.......] - ETA: 3s - loss: 0.4102 - accuracy: 0.9153
19300/25000 [======================>.......] - ETA: 3s - loss: 0.4103 - accuracy: 0.9152
19400/25000 [======================>.......] - ETA: 3s - loss: 0.4103 - accuracy: 0.9151
19500/25000 [======================>.......] - ETA: 3s - loss: 0.4102 - accuracy: 0.9151
19600/25000 [======================>.......] - ETA: 3s - loss: 0.4105 - accuracy: 0.9148
19700/25000 [======================>.......] - ETA: 3s - loss: 0.4104 - accuracy: 0.9149
19800/25000 [======================>.......] - ETA: 3s - loss: 0.4104 - accuracy: 0.9148
19900/25000 [======================>.......] - ETA: 3s - loss: 0.4102 - accuracy: 0.9150
20000/25000 [=======================>......] - ETA: 3s - loss: 0.4102 - accuracy: 0.9150
20100/25000 [=======================>......] - ETA: 2s - loss: 0.4100 - accuracy: 0.9151
20200/25000 [=======================>......] - ETA: 2s - loss: 0.4100 - accuracy: 0.9150
20300/25000 [=======================>......] - ETA: 2s - loss: 0.4098 - accuracy: 0.9152
20400/25000 [=======================>......] - ETA: 2s - loss: 0.4096 - accuracy: 0.9153
20500/25000 [=======================>......] - ETA: 2s - loss: 0.4096 - accuracy: 0.9153
20600/25000 [=======================>......] - ETA: 2s - loss: 0.4095 - accuracy: 0.9153
20700/25000 [=======================>......] - ETA: 2s - loss: 0.4093 - accuracy: 0.9155
20800/25000 [=======================>......] - ETA: 2s - loss: 0.4093 - accuracy: 0.9155
20900/25000 [========================>.....] - ETA: 2s - loss: 0.4094 - accuracy: 0.9152
21000/25000 [========================>.....] - ETA: 2s - loss: 0.4093 - accuracy: 0.9153
21100/25000 [========================>.....] - ETA: 2s - loss: 0.4093 - accuracy: 0.9152
21200/25000 [========================>.....] - ETA: 2s - loss: 0.4090 - accuracy: 0.9155
21300/25000 [========================>.....] - ETA: 2s - loss: 0.4090 - accuracy: 0.9154
21400/25000 [========================>.....] - ETA: 2s - loss: 0.4088 - accuracy: 0.9156
21500/25000 [========================>.....] - ETA: 2s - loss: 0.4087 - accuracy: 0.9155
21600/25000 [========================>.....] - ETA: 2s - loss: 0.4086 - accuracy: 0.9155
21700/25000 [=========================>....] - ETA: 2s - loss: 0.4087 - accuracy: 0.9154
21800/25000 [=========================>....] - ETA: 1s - loss: 0.4086 - accuracy: 0.9155
21900/25000 [=========================>....] - ETA: 1s - loss: 0.4087 - accuracy: 0.9153
22000/25000 [=========================>....] - ETA: 1s - loss: 0.4085 - accuracy: 0.9154
22100/25000 [=========================>....] - ETA: 1s - loss: 0.4085 - accuracy: 0.9154
22200/25000 [=========================>....] - ETA: 1s - loss: 0.4084 - accuracy: 0.9154
22300/25000 [=========================>....] - ETA: 1s - loss: 0.4082 - accuracy: 0.9154
22400/25000 [=========================>....] - ETA: 1s - loss: 0.4082 - accuracy: 0.9154
22500/25000 [==========================>...] - ETA: 1s - loss: 0.4082 - accuracy: 0.9153
22600/25000 [==========================>...] - ETA: 1s - loss: 0.4079 - accuracy: 0.9156
22700/25000 [==========================>...] - ETA: 1s - loss: 0.4080 - accuracy: 0.9154
22800/25000 [==========================>...] - ETA: 1s - loss: 0.4079 - accuracy: 0.9155
22900/25000 [==========================>...] - ETA: 1s - loss: 0.4079 - accuracy: 0.9155
23000/25000 [==========================>...] - ETA: 1s - loss: 0.4078 - accuracy: 0.9155
23100/25000 [==========================>...] - ETA: 1s - loss: 0.4077 - accuracy: 0.9156
23200/25000 [==========================>...] - ETA: 1s - loss: 0.4076 - accuracy: 0.9156
23300/25000 [==========================>...] - ETA: 1s - loss: 0.4076 - accuracy: 0.9156
23400/25000 [===========================>..] - ETA: 0s - loss: 0.4076 - accuracy: 0.9156
23500/25000 [===========================>..] - ETA: 0s - loss: 0.4076 - accuracy: 0.9155
23600/25000 [===========================>..] - ETA: 0s - loss: 0.4076 - accuracy: 0.9155
23700/25000 [===========================>..] - ETA: 0s - loss: 0.4076 - accuracy: 0.9154
23800/25000 [===========================>..] - ETA: 0s - loss: 0.4075 - accuracy: 0.9154
23900/25000 [===========================>..] - ETA: 0s - loss: 0.4074 - accuracy: 0.9154
24000/25000 [===========================>..] - ETA: 0s - loss: 0.4074 - accuracy: 0.9153
24100/25000 [===========================>..] - ETA: 0s - loss: 0.4075 - accuracy: 0.9152
24200/25000 [============================>.] - ETA: 0s - loss: 0.4073 - accuracy: 0.9153
24300/25000 [============================>.] - ETA: 0s - loss: 0.4071 - accuracy: 0.9156
24400/25000 [============================>.] - ETA: 0s - loss: 0.4072 - accuracy: 0.9153
24500/25000 [============================>.] - ETA: 0s - loss: 0.4072 - accuracy: 0.9153
24600/25000 [============================>.] - ETA: 0s - loss: 0.4071 - accuracy: 0.9153
24700/25000 [============================>.] - ETA: 0s - loss: 0.4070 - accuracy: 0.9153
24800/25000 [============================>.] - ETA: 0s - loss: 0.4069 - accuracy: 0.9154
24900/25000 [============================>.] - ETA: 0s - loss: 0.4071 - accuracy: 0.9151
25000/25000 [==============================] - 19s 770us/step - loss: 0.4071 - accuracy: 0.9151 - val_loss: 0.4463 - val_accuracy: 0.8618
Epoch 7/10

  100/25000 [..............................] - ETA: 14s - loss: 0.3706 - accuracy: 0.9500
  200/25000 [..............................] - ETA: 14s - loss: 0.3870 - accuracy: 0.9350
  300/25000 [..............................] - ETA: 14s - loss: 0.3872 - accuracy: 0.9333
  400/25000 [..............................] - ETA: 14s - loss: 0.3944 - accuracy: 0.9200
  500/25000 [..............................] - ETA: 14s - loss: 0.3870 - accuracy: 0.9280
  600/25000 [..............................] - ETA: 14s - loss: 0.3799 - accuracy: 0.9350
  700/25000 [..............................] - ETA: 14s - loss: 0.3779 - accuracy: 0.9386
  800/25000 [..............................] - ETA: 14s - loss: 0.3730 - accuracy: 0.9425
  900/25000 [>.............................] - ETA: 14s - loss: 0.3715 - accuracy: 0.9444
 1000/25000 [>.............................] - ETA: 14s - loss: 0.3727 - accuracy: 0.9430
 1100/25000 [>.............................] - ETA: 14s - loss: 0.3750 - accuracy: 0.9391
 1200/25000 [>.............................] - ETA: 14s - loss: 0.3715 - accuracy: 0.9433
 1300/25000 [>.............................] - ETA: 14s - loss: 0.3711 - accuracy: 0.9438
 1400/25000 [>.............................] - ETA: 14s - loss: 0.3715 - accuracy: 0.9429
 1500/25000 [>.............................] - ETA: 14s - loss: 0.3711 - accuracy: 0.9433
 1600/25000 [>.............................] - ETA: 14s - loss: 0.3720 - accuracy: 0.9425
 1700/25000 [=>............................] - ETA: 14s - loss: 0.3736 - accuracy: 0.9406
 1800/25000 [=>............................] - ETA: 14s - loss: 0.3731 - accuracy: 0.9411
 1900/25000 [=>............................] - ETA: 13s - loss: 0.3718 - accuracy: 0.9426
 2000/25000 [=>............................] - ETA: 13s - loss: 0.3726 - accuracy: 0.9420
 2100/25000 [=>............................] - ETA: 13s - loss: 0.3736 - accuracy: 0.9410
 2200/25000 [=>............................] - ETA: 13s - loss: 0.3738 - accuracy: 0.9405
 2300/25000 [=>............................] - ETA: 13s - loss: 0.3733 - accuracy: 0.9400
 2400/25000 [=>............................] - ETA: 13s - loss: 0.3734 - accuracy: 0.9396
 2500/25000 [==>...........................] - ETA: 13s - loss: 0.3719 - accuracy: 0.9412
 2600/25000 [==>...........................] - ETA: 13s - loss: 0.3708 - accuracy: 0.9419
 2700/25000 [==>...........................] - ETA: 13s - loss: 0.3703 - accuracy: 0.9422
 2800/25000 [==>...........................] - ETA: 13s - loss: 0.3698 - accuracy: 0.9425
 2900/25000 [==>...........................] - ETA: 13s - loss: 0.3702 - accuracy: 0.9414
 3000/25000 [==>...........................] - ETA: 13s - loss: 0.3698 - accuracy: 0.9423
 3100/25000 [==>...........................] - ETA: 13s - loss: 0.3693 - accuracy: 0.9429
 3200/25000 [==>...........................] - ETA: 13s - loss: 0.3688 - accuracy: 0.9431
 3300/25000 [==>...........................] - ETA: 13s - loss: 0.3696 - accuracy: 0.9424
 3400/25000 [===>..........................] - ETA: 13s - loss: 0.3704 - accuracy: 0.9418
 3500/25000 [===>..........................] - ETA: 13s - loss: 0.3700 - accuracy: 0.9423
 3600/25000 [===>..........................] - ETA: 13s - loss: 0.3700 - accuracy: 0.9419
 3700/25000 [===>..........................] - ETA: 12s - loss: 0.3705 - accuracy: 0.9416
 3800/25000 [===>..........................] - ETA: 12s - loss: 0.3715 - accuracy: 0.9405
 3900/25000 [===>..........................] - ETA: 12s - loss: 0.3715 - accuracy: 0.9405
 4000/25000 [===>..........................] - ETA: 12s - loss: 0.3704 - accuracy: 0.9415
 4100/25000 [===>..........................] - ETA: 12s - loss: 0.3703 - accuracy: 0.9417
 4200/25000 [====>.........................] - ETA: 12s - loss: 0.3697 - accuracy: 0.9421
 4300/25000 [====>.........................] - ETA: 12s - loss: 0.3701 - accuracy: 0.9416
 4400/25000 [====>.........................] - ETA: 12s - loss: 0.3701 - accuracy: 0.9416
 4500/25000 [====>.........................] - ETA: 12s - loss: 0.3703 - accuracy: 0.9409
 4600/25000 [====>.........................] - ETA: 12s - loss: 0.3709 - accuracy: 0.9400
 4700/25000 [====>.........................] - ETA: 12s - loss: 0.3713 - accuracy: 0.9394
 4800/25000 [====>.........................] - ETA: 12s - loss: 0.3713 - accuracy: 0.9390
 4900/25000 [====>.........................] - ETA: 12s - loss: 0.3715 - accuracy: 0.9390
 5000/25000 [=====>........................] - ETA: 12s - loss: 0.3713 - accuracy: 0.9388
 5100/25000 [=====>........................] - ETA: 12s - loss: 0.3712 - accuracy: 0.9388
 5200/25000 [=====>........................] - ETA: 12s - loss: 0.3719 - accuracy: 0.9383
 5300/25000 [=====>........................] - ETA: 12s - loss: 0.3724 - accuracy: 0.9379
 5400/25000 [=====>........................] - ETA: 11s - loss: 0.3726 - accuracy: 0.9376
 5500/25000 [=====>........................] - ETA: 11s - loss: 0.3736 - accuracy: 0.9369
 5600/25000 [=====>........................] - ETA: 11s - loss: 0.3743 - accuracy: 0.9361
 5700/25000 [=====>........................] - ETA: 11s - loss: 0.3744 - accuracy: 0.9360
 5800/25000 [=====>........................] - ETA: 11s - loss: 0.3748 - accuracy: 0.9355
 5900/25000 [======>.......................] - ETA: 11s - loss: 0.3747 - accuracy: 0.9356
 6000/25000 [======>.......................] - ETA: 11s - loss: 0.3738 - accuracy: 0.9365
 6100/25000 [======>.......................] - ETA: 11s - loss: 0.3736 - accuracy: 0.9367
 6200/25000 [======>.......................] - ETA: 11s - loss: 0.3741 - accuracy: 0.9360
 6300/25000 [======>.......................] - ETA: 11s - loss: 0.3735 - accuracy: 0.9367
 6400/25000 [======>.......................] - ETA: 11s - loss: 0.3734 - accuracy: 0.9366
 6500/25000 [======>.......................] - ETA: 11s - loss: 0.3733 - accuracy: 0.9366
 6600/25000 [======>.......................] - ETA: 11s - loss: 0.3725 - accuracy: 0.9373
 6700/25000 [=======>......................] - ETA: 11s - loss: 0.3728 - accuracy: 0.9370
 6800/25000 [=======>......................] - ETA: 11s - loss: 0.3733 - accuracy: 0.9365
 6900/25000 [=======>......................] - ETA: 11s - loss: 0.3734 - accuracy: 0.9362
 7000/25000 [=======>......................] - ETA: 10s - loss: 0.3733 - accuracy: 0.9363
 7100/25000 [=======>......................] - ETA: 10s - loss: 0.3734 - accuracy: 0.9362
 7200/25000 [=======>......................] - ETA: 10s - loss: 0.3730 - accuracy: 0.9367
 7300/25000 [=======>......................] - ETA: 10s - loss: 0.3725 - accuracy: 0.9371
 7400/25000 [=======>......................] - ETA: 10s - loss: 0.3719 - accuracy: 0.9376
 7500/25000 [========>.....................] - ETA: 10s - loss: 0.3718 - accuracy: 0.9377
 7600/25000 [========>.....................] - ETA: 10s - loss: 0.3718 - accuracy: 0.9376
 7700/25000 [========>.....................] - ETA: 10s - loss: 0.3717 - accuracy: 0.9377
 7800/25000 [========>.....................] - ETA: 10s - loss: 0.3720 - accuracy: 0.9373
 7900/25000 [========>.....................] - ETA: 10s - loss: 0.3712 - accuracy: 0.9381
 8000/25000 [========>.....................] - ETA: 10s - loss: 0.3712 - accuracy: 0.9381
 8100/25000 [========>.....................] - ETA: 10s - loss: 0.3708 - accuracy: 0.9384
 8200/25000 [========>.....................] - ETA: 10s - loss: 0.3708 - accuracy: 0.9384
 8300/25000 [========>.....................] - ETA: 10s - loss: 0.3713 - accuracy: 0.9380
 8400/25000 [=========>....................] - ETA: 10s - loss: 0.3712 - accuracy: 0.9380
 8500/25000 [=========>....................] - ETA: 10s - loss: 0.3713 - accuracy: 0.9378
 8600/25000 [=========>....................] - ETA: 9s - loss: 0.3714 - accuracy: 0.9377 
 8700/25000 [=========>....................] - ETA: 9s - loss: 0.3719 - accuracy: 0.9371
 8800/25000 [=========>....................] - ETA: 9s - loss: 0.3723 - accuracy: 0.9367
 8900/25000 [=========>....................] - ETA: 9s - loss: 0.3725 - accuracy: 0.9365
 9000/25000 [=========>....................] - ETA: 9s - loss: 0.3727 - accuracy: 0.9363
 9100/25000 [=========>....................] - ETA: 9s - loss: 0.3728 - accuracy: 0.9363
 9200/25000 [==========>...................] - ETA: 9s - loss: 0.3727 - accuracy: 0.9363
 9300/25000 [==========>...................] - ETA: 9s - loss: 0.3724 - accuracy: 0.9366
 9400/25000 [==========>...................] - ETA: 9s - loss: 0.3721 - accuracy: 0.9369
 9500/25000 [==========>...................] - ETA: 9s - loss: 0.3722 - accuracy: 0.9367
 9600/25000 [==========>...................] - ETA: 9s - loss: 0.3720 - accuracy: 0.9369
 9700/25000 [==========>...................] - ETA: 9s - loss: 0.3718 - accuracy: 0.9370
 9800/25000 [==========>...................] - ETA: 9s - loss: 0.3719 - accuracy: 0.9369
 9900/25000 [==========>...................] - ETA: 9s - loss: 0.3719 - accuracy: 0.9369
10000/25000 [===========>..................] - ETA: 9s - loss: 0.3725 - accuracy: 0.9362
10100/25000 [===========>..................] - ETA: 9s - loss: 0.3726 - accuracy: 0.9360
10200/25000 [===========>..................] - ETA: 9s - loss: 0.3726 - accuracy: 0.9361
10300/25000 [===========>..................] - ETA: 8s - loss: 0.3724 - accuracy: 0.9362
10400/25000 [===========>..................] - ETA: 8s - loss: 0.3723 - accuracy: 0.9362
10500/25000 [===========>..................] - ETA: 8s - loss: 0.3722 - accuracy: 0.9364
10600/25000 [===========>..................] - ETA: 8s - loss: 0.3720 - accuracy: 0.9366
10700/25000 [===========>..................] - ETA: 8s - loss: 0.3719 - accuracy: 0.9364
10800/25000 [===========>..................] - ETA: 8s - loss: 0.3718 - accuracy: 0.9365
10900/25000 [============>.................] - ETA: 8s - loss: 0.3722 - accuracy: 0.9360
11000/25000 [============>.................] - ETA: 8s - loss: 0.3727 - accuracy: 0.9355
11100/25000 [============>.................] - ETA: 8s - loss: 0.3727 - accuracy: 0.9354
11200/25000 [============>.................] - ETA: 8s - loss: 0.3724 - accuracy: 0.9356
11300/25000 [============>.................] - ETA: 8s - loss: 0.3723 - accuracy: 0.9357
11400/25000 [============>.................] - ETA: 8s - loss: 0.3721 - accuracy: 0.9358
11500/25000 [============>.................] - ETA: 8s - loss: 0.3722 - accuracy: 0.9356
11600/25000 [============>.................] - ETA: 8s - loss: 0.3726 - accuracy: 0.9352
11700/25000 [=============>................] - ETA: 8s - loss: 0.3725 - accuracy: 0.9352
11800/25000 [=============>................] - ETA: 8s - loss: 0.3727 - accuracy: 0.9350
11900/25000 [=============>................] - ETA: 7s - loss: 0.3728 - accuracy: 0.9348
12000/25000 [=============>................] - ETA: 7s - loss: 0.3727 - accuracy: 0.9348
12100/25000 [=============>................] - ETA: 7s - loss: 0.3726 - accuracy: 0.9348
12200/25000 [=============>................] - ETA: 7s - loss: 0.3725 - accuracy: 0.9349
12300/25000 [=============>................] - ETA: 7s - loss: 0.3724 - accuracy: 0.9350
12400/25000 [=============>................] - ETA: 7s - loss: 0.3723 - accuracy: 0.9351
12500/25000 [==============>...............] - ETA: 7s - loss: 0.3722 - accuracy: 0.9351
12600/25000 [==============>...............] - ETA: 7s - loss: 0.3719 - accuracy: 0.9354
12700/25000 [==============>...............] - ETA: 7s - loss: 0.3721 - accuracy: 0.9351
12800/25000 [==============>...............] - ETA: 7s - loss: 0.3720 - accuracy: 0.9352
12900/25000 [==============>...............] - ETA: 7s - loss: 0.3722 - accuracy: 0.9350
13000/25000 [==============>...............] - ETA: 7s - loss: 0.3721 - accuracy: 0.9351
13100/25000 [==============>...............] - ETA: 7s - loss: 0.3721 - accuracy: 0.9350
13200/25000 [==============>...............] - ETA: 7s - loss: 0.3721 - accuracy: 0.9349
13300/25000 [==============>...............] - ETA: 7s - loss: 0.3721 - accuracy: 0.9349
13400/25000 [===============>..............] - ETA: 7s - loss: 0.3721 - accuracy: 0.9349
13500/25000 [===============>..............] - ETA: 7s - loss: 0.3720 - accuracy: 0.9350
13600/25000 [===============>..............] - ETA: 6s - loss: 0.3721 - accuracy: 0.9349
13700/25000 [===============>..............] - ETA: 6s - loss: 0.3722 - accuracy: 0.9348
13800/25000 [===============>..............] - ETA: 6s - loss: 0.3723 - accuracy: 0.9345
13900/25000 [===============>..............] - ETA: 6s - loss: 0.3723 - accuracy: 0.9345
14000/25000 [===============>..............] - ETA: 6s - loss: 0.3721 - accuracy: 0.9347
14100/25000 [===============>..............] - ETA: 6s - loss: 0.3720 - accuracy: 0.9348
14200/25000 [================>.............] - ETA: 6s - loss: 0.3719 - accuracy: 0.9349
14300/25000 [================>.............] - ETA: 6s - loss: 0.3720 - accuracy: 0.9348
14400/25000 [================>.............] - ETA: 6s - loss: 0.3721 - accuracy: 0.9347
14500/25000 [================>.............] - ETA: 6s - loss: 0.3717 - accuracy: 0.9349
14600/25000 [================>.............] - ETA: 6s - loss: 0.3718 - accuracy: 0.9347
14700/25000 [================>.............] - ETA: 6s - loss: 0.3718 - accuracy: 0.9347
14800/25000 [================>.............] - ETA: 6s - loss: 0.3718 - accuracy: 0.9347
14900/25000 [================>.............] - ETA: 6s - loss: 0.3719 - accuracy: 0.9345
15000/25000 [=================>............] - ETA: 6s - loss: 0.3719 - accuracy: 0.9344
15100/25000 [=================>............] - ETA: 6s - loss: 0.3718 - accuracy: 0.9344
15200/25000 [=================>............] - ETA: 5s - loss: 0.3717 - accuracy: 0.9345
15300/25000 [=================>............] - ETA: 5s - loss: 0.3716 - accuracy: 0.9344
15400/25000 [=================>............] - ETA: 5s - loss: 0.3715 - accuracy: 0.9345
15500/25000 [=================>............] - ETA: 5s - loss: 0.3715 - accuracy: 0.9345
15600/25000 [=================>............] - ETA: 5s - loss: 0.3713 - accuracy: 0.9347
15700/25000 [=================>............] - ETA: 5s - loss: 0.3710 - accuracy: 0.9349
15800/25000 [=================>............] - ETA: 5s - loss: 0.3711 - accuracy: 0.9348
15900/25000 [==================>...........] - ETA: 5s - loss: 0.3714 - accuracy: 0.9345
16000/25000 [==================>...........] - ETA: 5s - loss: 0.3712 - accuracy: 0.9346
16100/25000 [==================>...........] - ETA: 5s - loss: 0.3712 - accuracy: 0.9346
16200/25000 [==================>...........] - ETA: 5s - loss: 0.3713 - accuracy: 0.9346
16300/25000 [==================>...........] - ETA: 5s - loss: 0.3713 - accuracy: 0.9344
16400/25000 [==================>...........] - ETA: 5s - loss: 0.3714 - accuracy: 0.9343
16500/25000 [==================>...........] - ETA: 5s - loss: 0.3712 - accuracy: 0.9344
16600/25000 [==================>...........] - ETA: 5s - loss: 0.3713 - accuracy: 0.9343
16700/25000 [===================>..........] - ETA: 5s - loss: 0.3713 - accuracy: 0.9342
16800/25000 [===================>..........] - ETA: 5s - loss: 0.3714 - accuracy: 0.9341
16900/25000 [===================>..........] - ETA: 4s - loss: 0.3711 - accuracy: 0.9344
17000/25000 [===================>..........] - ETA: 4s - loss: 0.3712 - accuracy: 0.9342
17100/25000 [===================>..........] - ETA: 4s - loss: 0.3713 - accuracy: 0.9340
17200/25000 [===================>..........] - ETA: 4s - loss: 0.3715 - accuracy: 0.9337
17300/25000 [===================>..........] - ETA: 4s - loss: 0.3713 - accuracy: 0.9340
17400/25000 [===================>..........] - ETA: 4s - loss: 0.3713 - accuracy: 0.9340
17500/25000 [====================>.........] - ETA: 4s - loss: 0.3714 - accuracy: 0.9338
17600/25000 [====================>.........] - ETA: 4s - loss: 0.3714 - accuracy: 0.9337
17700/25000 [====================>.........] - ETA: 4s - loss: 0.3715 - accuracy: 0.9336
17800/25000 [====================>.........] - ETA: 4s - loss: 0.3716 - accuracy: 0.9335
17900/25000 [====================>.........] - ETA: 4s - loss: 0.3715 - accuracy: 0.9335
18000/25000 [====================>.........] - ETA: 4s - loss: 0.3713 - accuracy: 0.9336
18100/25000 [====================>.........] - ETA: 4s - loss: 0.3713 - accuracy: 0.9336
18200/25000 [====================>.........] - ETA: 4s - loss: 0.3713 - accuracy: 0.9336
18300/25000 [====================>.........] - ETA: 4s - loss: 0.3716 - accuracy: 0.9332
18400/25000 [=====================>........] - ETA: 4s - loss: 0.3715 - accuracy: 0.9333
18500/25000 [=====================>........] - ETA: 3s - loss: 0.3713 - accuracy: 0.9334
18600/25000 [=====================>........] - ETA: 3s - loss: 0.3712 - accuracy: 0.9335
18700/25000 [=====================>........] - ETA: 3s - loss: 0.3710 - accuracy: 0.9335
18800/25000 [=====================>........] - ETA: 3s - loss: 0.3709 - accuracy: 0.9336
18900/25000 [=====================>........] - ETA: 3s - loss: 0.3708 - accuracy: 0.9336
19000/25000 [=====================>........] - ETA: 3s - loss: 0.3710 - accuracy: 0.9334
19100/25000 [=====================>........] - ETA: 3s - loss: 0.3710 - accuracy: 0.9332
19200/25000 [======================>.......] - ETA: 3s - loss: 0.3711 - accuracy: 0.9331
19300/25000 [======================>.......] - ETA: 3s - loss: 0.3710 - accuracy: 0.9332
19400/25000 [======================>.......] - ETA: 3s - loss: 0.3710 - accuracy: 0.9331
19500/25000 [======================>.......] - ETA: 3s - loss: 0.3712 - accuracy: 0.9329
19600/25000 [======================>.......] - ETA: 3s - loss: 0.3710 - accuracy: 0.9329
19700/25000 [======================>.......] - ETA: 3s - loss: 0.3709 - accuracy: 0.9329
19800/25000 [======================>.......] - ETA: 3s - loss: 0.3706 - accuracy: 0.9331
19900/25000 [======================>.......] - ETA: 3s - loss: 0.3705 - accuracy: 0.9331
20000/25000 [=======================>......] - ETA: 3s - loss: 0.3702 - accuracy: 0.9334
20100/25000 [=======================>......] - ETA: 2s - loss: 0.3703 - accuracy: 0.9333
20200/25000 [=======================>......] - ETA: 2s - loss: 0.3703 - accuracy: 0.9333
20300/25000 [=======================>......] - ETA: 2s - loss: 0.3706 - accuracy: 0.9330
20400/25000 [=======================>......] - ETA: 2s - loss: 0.3705 - accuracy: 0.9330
20500/25000 [=======================>......] - ETA: 2s - loss: 0.3707 - accuracy: 0.9328
20600/25000 [=======================>......] - ETA: 2s - loss: 0.3707 - accuracy: 0.9327
20700/25000 [=======================>......] - ETA: 2s - loss: 0.3709 - accuracy: 0.9325
20800/25000 [=======================>......] - ETA: 2s - loss: 0.3708 - accuracy: 0.9325
20900/25000 [========================>.....] - ETA: 2s - loss: 0.3710 - accuracy: 0.9323
21000/25000 [========================>.....] - ETA: 2s - loss: 0.3708 - accuracy: 0.9324
21100/25000 [========================>.....] - ETA: 2s - loss: 0.3705 - accuracy: 0.9326
21200/25000 [========================>.....] - ETA: 2s - loss: 0.3705 - accuracy: 0.9326
21300/25000 [========================>.....] - ETA: 2s - loss: 0.3705 - accuracy: 0.9325
21400/25000 [========================>.....] - ETA: 2s - loss: 0.3704 - accuracy: 0.9326
21500/25000 [========================>.....] - ETA: 2s - loss: 0.3705 - accuracy: 0.9325
21600/25000 [========================>.....] - ETA: 2s - loss: 0.3706 - accuracy: 0.9323
21700/25000 [=========================>....] - ETA: 2s - loss: 0.3708 - accuracy: 0.9321
21800/25000 [=========================>....] - ETA: 1s - loss: 0.3709 - accuracy: 0.9319
21900/25000 [=========================>....] - ETA: 1s - loss: 0.3708 - accuracy: 0.9320
22000/25000 [=========================>....] - ETA: 1s - loss: 0.3707 - accuracy: 0.9320
22100/25000 [=========================>....] - ETA: 1s - loss: 0.3705 - accuracy: 0.9321
22200/25000 [=========================>....] - ETA: 1s - loss: 0.3704 - accuracy: 0.9322
22300/25000 [=========================>....] - ETA: 1s - loss: 0.3704 - accuracy: 0.9322
22400/25000 [=========================>....] - ETA: 1s - loss: 0.3705 - accuracy: 0.9320
22500/25000 [==========================>...] - ETA: 1s - loss: 0.3706 - accuracy: 0.9319
22600/25000 [==========================>...] - ETA: 1s - loss: 0.3704 - accuracy: 0.9320
22700/25000 [==========================>...] - ETA: 1s - loss: 0.3707 - accuracy: 0.9318
22800/25000 [==========================>...] - ETA: 1s - loss: 0.3708 - accuracy: 0.9315
22900/25000 [==========================>...] - ETA: 1s - loss: 0.3710 - accuracy: 0.9313
23000/25000 [==========================>...] - ETA: 1s - loss: 0.3710 - accuracy: 0.9313
23100/25000 [==========================>...] - ETA: 1s - loss: 0.3708 - accuracy: 0.9314
23200/25000 [==========================>...] - ETA: 1s - loss: 0.3708 - accuracy: 0.9314
23300/25000 [==========================>...] - ETA: 1s - loss: 0.3710 - accuracy: 0.9311
23400/25000 [===========================>..] - ETA: 0s - loss: 0.3709 - accuracy: 0.9312
23500/25000 [===========================>..] - ETA: 0s - loss: 0.3708 - accuracy: 0.9312
23600/25000 [===========================>..] - ETA: 0s - loss: 0.3708 - accuracy: 0.9313
23700/25000 [===========================>..] - ETA: 0s - loss: 0.3709 - accuracy: 0.9311
23800/25000 [===========================>..] - ETA: 0s - loss: 0.3708 - accuracy: 0.9312
23900/25000 [===========================>..] - ETA: 0s - loss: 0.3708 - accuracy: 0.9311
24000/25000 [===========================>..] - ETA: 0s - loss: 0.3707 - accuracy: 0.9312
24100/25000 [===========================>..] - ETA: 0s - loss: 0.3708 - accuracy: 0.9310
24200/25000 [============================>.] - ETA: 0s - loss: 0.3708 - accuracy: 0.9310
24300/25000 [============================>.] - ETA: 0s - loss: 0.3710 - accuracy: 0.9308
24400/25000 [============================>.] - ETA: 0s - loss: 0.3711 - accuracy: 0.9306
24500/25000 [============================>.] - ETA: 0s - loss: 0.3711 - accuracy: 0.9306
24600/25000 [============================>.] - ETA: 0s - loss: 0.3709 - accuracy: 0.9307
24700/25000 [============================>.] - ETA: 0s - loss: 0.3708 - accuracy: 0.9307
24800/25000 [============================>.] - ETA: 0s - loss: 0.3708 - accuracy: 0.9306
24900/25000 [============================>.] - ETA: 0s - loss: 0.3708 - accuracy: 0.9306
25000/25000 [==============================] - 19s 770us/step - loss: 0.3708 - accuracy: 0.9306 - val_loss: 0.4405 - val_accuracy: 0.8503
Epoch 8/10

  100/25000 [..............................] - ETA: 15s - loss: 0.3246 - accuracy: 0.9600
  200/25000 [..............................] - ETA: 15s - loss: 0.3472 - accuracy: 0.9450
  300/25000 [..............................] - ETA: 15s - loss: 0.3531 - accuracy: 0.9367
  400/25000 [..............................] - ETA: 15s - loss: 0.3634 - accuracy: 0.9275
  500/25000 [..............................] - ETA: 14s - loss: 0.3592 - accuracy: 0.9300
  600/25000 [..............................] - ETA: 14s - loss: 0.3565 - accuracy: 0.9333
  700/25000 [..............................] - ETA: 14s - loss: 0.3587 - accuracy: 0.9314
  800/25000 [..............................] - ETA: 14s - loss: 0.3613 - accuracy: 0.9300
  900/25000 [>.............................] - ETA: 14s - loss: 0.3573 - accuracy: 0.9322
 1000/25000 [>.............................] - ETA: 14s - loss: 0.3590 - accuracy: 0.9310
 1100/25000 [>.............................] - ETA: 14s - loss: 0.3553 - accuracy: 0.9345
 1200/25000 [>.............................] - ETA: 14s - loss: 0.3604 - accuracy: 0.9300
 1300/25000 [>.............................] - ETA: 14s - loss: 0.3602 - accuracy: 0.9300
 1400/25000 [>.............................] - ETA: 14s - loss: 0.3615 - accuracy: 0.9279
 1500/25000 [>.............................] - ETA: 14s - loss: 0.3597 - accuracy: 0.9293
 1600/25000 [>.............................] - ETA: 14s - loss: 0.3606 - accuracy: 0.9287
 1700/25000 [=>............................] - ETA: 14s - loss: 0.3589 - accuracy: 0.9306
 1800/25000 [=>............................] - ETA: 14s - loss: 0.3595 - accuracy: 0.9300
 1900/25000 [=>............................] - ETA: 14s - loss: 0.3588 - accuracy: 0.9300
 2000/25000 [=>............................] - ETA: 14s - loss: 0.3604 - accuracy: 0.9285
 2100/25000 [=>............................] - ETA: 13s - loss: 0.3583 - accuracy: 0.9305
 2200/25000 [=>............................] - ETA: 13s - loss: 0.3587 - accuracy: 0.9300
 2300/25000 [=>............................] - ETA: 13s - loss: 0.3612 - accuracy: 0.9278
 2400/25000 [=>............................] - ETA: 13s - loss: 0.3610 - accuracy: 0.9283
 2500/25000 [==>...........................] - ETA: 13s - loss: 0.3604 - accuracy: 0.9288
 2600/25000 [==>...........................] - ETA: 13s - loss: 0.3580 - accuracy: 0.9312
 2700/25000 [==>...........................] - ETA: 13s - loss: 0.3578 - accuracy: 0.9315
 2800/25000 [==>...........................] - ETA: 13s - loss: 0.3563 - accuracy: 0.9329
 2900/25000 [==>...........................] - ETA: 13s - loss: 0.3557 - accuracy: 0.9338
 3000/25000 [==>...........................] - ETA: 13s - loss: 0.3560 - accuracy: 0.9337
 3100/25000 [==>...........................] - ETA: 13s - loss: 0.3555 - accuracy: 0.9339
 3200/25000 [==>...........................] - ETA: 13s - loss: 0.3552 - accuracy: 0.9341
 3300/25000 [==>...........................] - ETA: 13s - loss: 0.3556 - accuracy: 0.9339
 3400/25000 [===>..........................] - ETA: 13s - loss: 0.3548 - accuracy: 0.9344
 3500/25000 [===>..........................] - ETA: 13s - loss: 0.3557 - accuracy: 0.9334
 3600/25000 [===>..........................] - ETA: 13s - loss: 0.3544 - accuracy: 0.9344
 3700/25000 [===>..........................] - ETA: 12s - loss: 0.3544 - accuracy: 0.9346
 3800/25000 [===>..........................] - ETA: 12s - loss: 0.3535 - accuracy: 0.9353
 3900/25000 [===>..........................] - ETA: 12s - loss: 0.3531 - accuracy: 0.9354
 4000/25000 [===>..........................] - ETA: 12s - loss: 0.3529 - accuracy: 0.9355
 4100/25000 [===>..........................] - ETA: 12s - loss: 0.3521 - accuracy: 0.9363
 4200/25000 [====>.........................] - ETA: 12s - loss: 0.3519 - accuracy: 0.9367
 4300/25000 [====>.........................] - ETA: 12s - loss: 0.3516 - accuracy: 0.9370
 4400/25000 [====>.........................] - ETA: 12s - loss: 0.3522 - accuracy: 0.9366
 4500/25000 [====>.........................] - ETA: 12s - loss: 0.3516 - accuracy: 0.9371
 4600/25000 [====>.........................] - ETA: 12s - loss: 0.3510 - accuracy: 0.9376
 4700/25000 [====>.........................] - ETA: 12s - loss: 0.3504 - accuracy: 0.9381
 4800/25000 [====>.........................] - ETA: 12s - loss: 0.3501 - accuracy: 0.9383
 4900/25000 [====>.........................] - ETA: 12s - loss: 0.3499 - accuracy: 0.9386
 5000/25000 [=====>........................] - ETA: 12s - loss: 0.3494 - accuracy: 0.9390
 5100/25000 [=====>........................] - ETA: 12s - loss: 0.3497 - accuracy: 0.9388
 5200/25000 [=====>........................] - ETA: 12s - loss: 0.3498 - accuracy: 0.9387
 5300/25000 [=====>........................] - ETA: 12s - loss: 0.3500 - accuracy: 0.9385
 5400/25000 [=====>........................] - ETA: 11s - loss: 0.3501 - accuracy: 0.9383
 5500/25000 [=====>........................] - ETA: 11s - loss: 0.3495 - accuracy: 0.9387
 5600/25000 [=====>........................] - ETA: 11s - loss: 0.3494 - accuracy: 0.9388
 5700/25000 [=====>........................] - ETA: 11s - loss: 0.3496 - accuracy: 0.9386
 5800/25000 [=====>........................] - ETA: 11s - loss: 0.3492 - accuracy: 0.9391
 5900/25000 [======>.......................] - ETA: 11s - loss: 0.3488 - accuracy: 0.9395
 6000/25000 [======>.......................] - ETA: 11s - loss: 0.3484 - accuracy: 0.9398
 6100/25000 [======>.......................] - ETA: 11s - loss: 0.3485 - accuracy: 0.9398
 6200/25000 [======>.......................] - ETA: 11s - loss: 0.3480 - accuracy: 0.9402
 6300/25000 [======>.......................] - ETA: 11s - loss: 0.3479 - accuracy: 0.9403
 6400/25000 [======>.......................] - ETA: 11s - loss: 0.3480 - accuracy: 0.9403
 6500/25000 [======>.......................] - ETA: 11s - loss: 0.3488 - accuracy: 0.9397
 6600/25000 [======>.......................] - ETA: 11s - loss: 0.3485 - accuracy: 0.9398
 6700/25000 [=======>......................] - ETA: 11s - loss: 0.3483 - accuracy: 0.9401
 6800/25000 [=======>......................] - ETA: 11s - loss: 0.3479 - accuracy: 0.9404
 6900/25000 [=======>......................] - ETA: 11s - loss: 0.3479 - accuracy: 0.9404
 7000/25000 [=======>......................] - ETA: 11s - loss: 0.3477 - accuracy: 0.9404
 7100/25000 [=======>......................] - ETA: 10s - loss: 0.3476 - accuracy: 0.9404
 7200/25000 [=======>......................] - ETA: 10s - loss: 0.3475 - accuracy: 0.9406
 7300/25000 [=======>......................] - ETA: 10s - loss: 0.3474 - accuracy: 0.9405
 7400/25000 [=======>......................] - ETA: 10s - loss: 0.3473 - accuracy: 0.9407
 7500/25000 [========>.....................] - ETA: 10s - loss: 0.3468 - accuracy: 0.9409
 7600/25000 [========>.....................] - ETA: 10s - loss: 0.3469 - accuracy: 0.9408
 7700/25000 [========>.....................] - ETA: 10s - loss: 0.3463 - accuracy: 0.9412
 7800/25000 [========>.....................] - ETA: 10s - loss: 0.3460 - accuracy: 0.9414
 7900/25000 [========>.....................] - ETA: 10s - loss: 0.3464 - accuracy: 0.9410
 8000/25000 [========>.....................] - ETA: 10s - loss: 0.3462 - accuracy: 0.9411
 8100/25000 [========>.....................] - ETA: 10s - loss: 0.3462 - accuracy: 0.9412
 8200/25000 [========>.....................] - ETA: 10s - loss: 0.3459 - accuracy: 0.9413
 8300/25000 [========>.....................] - ETA: 10s - loss: 0.3458 - accuracy: 0.9414
 8400/25000 [=========>....................] - ETA: 10s - loss: 0.3458 - accuracy: 0.9414
 8500/25000 [=========>....................] - ETA: 10s - loss: 0.3456 - accuracy: 0.9415
 8600/25000 [=========>....................] - ETA: 10s - loss: 0.3459 - accuracy: 0.9413
 8700/25000 [=========>....................] - ETA: 9s - loss: 0.3457 - accuracy: 0.9414 
 8800/25000 [=========>....................] - ETA: 9s - loss: 0.3457 - accuracy: 0.9414
 8900/25000 [=========>....................] - ETA: 9s - loss: 0.3458 - accuracy: 0.9412
 9000/25000 [=========>....................] - ETA: 9s - loss: 0.3456 - accuracy: 0.9413
 9100/25000 [=========>....................] - ETA: 9s - loss: 0.3453 - accuracy: 0.9415
 9200/25000 [==========>...................] - ETA: 9s - loss: 0.3451 - accuracy: 0.9416
 9300/25000 [==========>...................] - ETA: 9s - loss: 0.3450 - accuracy: 0.9416
 9400/25000 [==========>...................] - ETA: 9s - loss: 0.3452 - accuracy: 0.9415
 9500/25000 [==========>...................] - ETA: 9s - loss: 0.3452 - accuracy: 0.9415
 9600/25000 [==========>...................] - ETA: 9s - loss: 0.3452 - accuracy: 0.9415
 9700/25000 [==========>...................] - ETA: 9s - loss: 0.3450 - accuracy: 0.9414
 9800/25000 [==========>...................] - ETA: 9s - loss: 0.3446 - accuracy: 0.9418
 9900/25000 [==========>...................] - ETA: 9s - loss: 0.3447 - accuracy: 0.9418
10000/25000 [===========>..................] - ETA: 9s - loss: 0.3443 - accuracy: 0.9421
10100/25000 [===========>..................] - ETA: 9s - loss: 0.3447 - accuracy: 0.9418
10200/25000 [===========>..................] - ETA: 9s - loss: 0.3446 - accuracy: 0.9420
10300/25000 [===========>..................] - ETA: 8s - loss: 0.3448 - accuracy: 0.9417
10400/25000 [===========>..................] - ETA: 8s - loss: 0.3445 - accuracy: 0.9418
10500/25000 [===========>..................] - ETA: 8s - loss: 0.3444 - accuracy: 0.9419
10600/25000 [===========>..................] - ETA: 8s - loss: 0.3446 - accuracy: 0.9416
10700/25000 [===========>..................] - ETA: 8s - loss: 0.3444 - accuracy: 0.9418
10800/25000 [===========>..................] - ETA: 8s - loss: 0.3440 - accuracy: 0.9421
10900/25000 [============>.................] - ETA: 8s - loss: 0.3435 - accuracy: 0.9425
11000/25000 [============>.................] - ETA: 8s - loss: 0.3435 - accuracy: 0.9425
11100/25000 [============>.................] - ETA: 8s - loss: 0.3437 - accuracy: 0.9423
11200/25000 [============>.................] - ETA: 8s - loss: 0.3433 - accuracy: 0.9425
11300/25000 [============>.................] - ETA: 8s - loss: 0.3430 - accuracy: 0.9427
11400/25000 [============>.................] - ETA: 8s - loss: 0.3433 - accuracy: 0.9425
11500/25000 [============>.................] - ETA: 8s - loss: 0.3434 - accuracy: 0.9423
11600/25000 [============>.................] - ETA: 8s - loss: 0.3432 - accuracy: 0.9424
11700/25000 [=============>................] - ETA: 8s - loss: 0.3430 - accuracy: 0.9426
11800/25000 [=============>................] - ETA: 8s - loss: 0.3434 - accuracy: 0.9422
11900/25000 [=============>................] - ETA: 8s - loss: 0.3435 - accuracy: 0.9420
12000/25000 [=============>................] - ETA: 7s - loss: 0.3437 - accuracy: 0.9417
12100/25000 [=============>................] - ETA: 7s - loss: 0.3439 - accuracy: 0.9416
12200/25000 [=============>................] - ETA: 7s - loss: 0.3437 - accuracy: 0.9416
12300/25000 [=============>................] - ETA: 7s - loss: 0.3434 - accuracy: 0.9419
12400/25000 [=============>................] - ETA: 7s - loss: 0.3434 - accuracy: 0.9418
12500/25000 [==============>...............] - ETA: 7s - loss: 0.3437 - accuracy: 0.9415
12600/25000 [==============>...............] - ETA: 7s - loss: 0.3435 - accuracy: 0.9416
12700/25000 [==============>...............] - ETA: 7s - loss: 0.3434 - accuracy: 0.9416
12800/25000 [==============>...............] - ETA: 7s - loss: 0.3435 - accuracy: 0.9415
12900/25000 [==============>...............] - ETA: 7s - loss: 0.3434 - accuracy: 0.9416
13000/25000 [==============>...............] - ETA: 7s - loss: 0.3433 - accuracy: 0.9416
13100/25000 [==============>...............] - ETA: 7s - loss: 0.3431 - accuracy: 0.9417
13200/25000 [==============>...............] - ETA: 7s - loss: 0.3429 - accuracy: 0.9418
13300/25000 [==============>...............] - ETA: 7s - loss: 0.3434 - accuracy: 0.9413
13400/25000 [===============>..............] - ETA: 7s - loss: 0.3437 - accuracy: 0.9410
13500/25000 [===============>..............] - ETA: 7s - loss: 0.3439 - accuracy: 0.9407
13600/25000 [===============>..............] - ETA: 6s - loss: 0.3436 - accuracy: 0.9410
13700/25000 [===============>..............] - ETA: 6s - loss: 0.3436 - accuracy: 0.9409
13800/25000 [===============>..............] - ETA: 6s - loss: 0.3436 - accuracy: 0.9409
13900/25000 [===============>..............] - ETA: 6s - loss: 0.3434 - accuracy: 0.9410
14000/25000 [===============>..............] - ETA: 6s - loss: 0.3432 - accuracy: 0.9412
14100/25000 [===============>..............] - ETA: 6s - loss: 0.3433 - accuracy: 0.9411
14200/25000 [================>.............] - ETA: 6s - loss: 0.3433 - accuracy: 0.9411
14300/25000 [================>.............] - ETA: 6s - loss: 0.3432 - accuracy: 0.9411
14400/25000 [================>.............] - ETA: 6s - loss: 0.3430 - accuracy: 0.9413
14500/25000 [================>.............] - ETA: 6s - loss: 0.3429 - accuracy: 0.9413
14600/25000 [================>.............] - ETA: 6s - loss: 0.3430 - accuracy: 0.9412
14700/25000 [================>.............] - ETA: 6s - loss: 0.3431 - accuracy: 0.9410
14800/25000 [================>.............] - ETA: 6s - loss: 0.3430 - accuracy: 0.9411
14900/25000 [================>.............] - ETA: 6s - loss: 0.3430 - accuracy: 0.9411
15000/25000 [=================>............] - ETA: 6s - loss: 0.3435 - accuracy: 0.9407
15100/25000 [=================>............] - ETA: 6s - loss: 0.3436 - accuracy: 0.9406
15200/25000 [=================>............] - ETA: 5s - loss: 0.3433 - accuracy: 0.9409
15300/25000 [=================>............] - ETA: 5s - loss: 0.3435 - accuracy: 0.9407
15400/25000 [=================>............] - ETA: 5s - loss: 0.3437 - accuracy: 0.9405
15500/25000 [=================>............] - ETA: 5s - loss: 0.3435 - accuracy: 0.9406
15600/25000 [=================>............] - ETA: 5s - loss: 0.3435 - accuracy: 0.9406
15700/25000 [=================>............] - ETA: 5s - loss: 0.3437 - accuracy: 0.9404
15800/25000 [=================>............] - ETA: 5s - loss: 0.3435 - accuracy: 0.9406
15900/25000 [==================>...........] - ETA: 5s - loss: 0.3432 - accuracy: 0.9408
16000/25000 [==================>...........] - ETA: 5s - loss: 0.3432 - accuracy: 0.9408
16100/25000 [==================>...........] - ETA: 5s - loss: 0.3433 - accuracy: 0.9406
16200/25000 [==================>...........] - ETA: 5s - loss: 0.3430 - accuracy: 0.9408
16300/25000 [==================>...........] - ETA: 5s - loss: 0.3430 - accuracy: 0.9407
16400/25000 [==================>...........] - ETA: 5s - loss: 0.3428 - accuracy: 0.9409
16500/25000 [==================>...........] - ETA: 5s - loss: 0.3425 - accuracy: 0.9411
16600/25000 [==================>...........] - ETA: 5s - loss: 0.3423 - accuracy: 0.9412
16700/25000 [===================>..........] - ETA: 5s - loss: 0.3422 - accuracy: 0.9413
16800/25000 [===================>..........] - ETA: 5s - loss: 0.3421 - accuracy: 0.9414
16900/25000 [===================>..........] - ETA: 4s - loss: 0.3420 - accuracy: 0.9415
17000/25000 [===================>..........] - ETA: 4s - loss: 0.3420 - accuracy: 0.9414
17100/25000 [===================>..........] - ETA: 4s - loss: 0.3420 - accuracy: 0.9413
17200/25000 [===================>..........] - ETA: 4s - loss: 0.3420 - accuracy: 0.9413
17300/25000 [===================>..........] - ETA: 4s - loss: 0.3421 - accuracy: 0.9413
17400/25000 [===================>..........] - ETA: 4s - loss: 0.3420 - accuracy: 0.9412
17500/25000 [====================>.........] - ETA: 4s - loss: 0.3419 - accuracy: 0.9413
17600/25000 [====================>.........] - ETA: 4s - loss: 0.3420 - accuracy: 0.9411
17700/25000 [====================>.........] - ETA: 4s - loss: 0.3420 - accuracy: 0.9410
17800/25000 [====================>.........] - ETA: 4s - loss: 0.3420 - accuracy: 0.9410
17900/25000 [====================>.........] - ETA: 4s - loss: 0.3422 - accuracy: 0.9408
18000/25000 [====================>.........] - ETA: 4s - loss: 0.3419 - accuracy: 0.9410
18100/25000 [====================>.........] - ETA: 4s - loss: 0.3417 - accuracy: 0.9411
18200/25000 [====================>.........] - ETA: 4s - loss: 0.3417 - accuracy: 0.9410
18300/25000 [====================>.........] - ETA: 4s - loss: 0.3415 - accuracy: 0.9411
18400/25000 [=====================>........] - ETA: 4s - loss: 0.3416 - accuracy: 0.9410
18500/25000 [=====================>........] - ETA: 3s - loss: 0.3419 - accuracy: 0.9407
18600/25000 [=====================>........] - ETA: 3s - loss: 0.3420 - accuracy: 0.9406
18700/25000 [=====================>........] - ETA: 3s - loss: 0.3418 - accuracy: 0.9407
18800/25000 [=====================>........] - ETA: 3s - loss: 0.3416 - accuracy: 0.9409
18900/25000 [=====================>........] - ETA: 3s - loss: 0.3415 - accuracy: 0.9409
19000/25000 [=====================>........] - ETA: 3s - loss: 0.3415 - accuracy: 0.9408
19100/25000 [=====================>........] - ETA: 3s - loss: 0.3413 - accuracy: 0.9409
19200/25000 [======================>.......] - ETA: 3s - loss: 0.3411 - accuracy: 0.9410
19300/25000 [======================>.......] - ETA: 3s - loss: 0.3411 - accuracy: 0.9410
19400/25000 [======================>.......] - ETA: 3s - loss: 0.3411 - accuracy: 0.9410
19500/25000 [======================>.......] - ETA: 3s - loss: 0.3412 - accuracy: 0.9409
19600/25000 [======================>.......] - ETA: 3s - loss: 0.3410 - accuracy: 0.9410
19700/25000 [======================>.......] - ETA: 3s - loss: 0.3410 - accuracy: 0.9410
19800/25000 [======================>.......] - ETA: 3s - loss: 0.3411 - accuracy: 0.9409
19900/25000 [======================>.......] - ETA: 3s - loss: 0.3412 - accuracy: 0.9407
20000/25000 [=======================>......] - ETA: 3s - loss: 0.3413 - accuracy: 0.9406
20100/25000 [=======================>......] - ETA: 2s - loss: 0.3412 - accuracy: 0.9406
20200/25000 [=======================>......] - ETA: 2s - loss: 0.3411 - accuracy: 0.9406
20300/25000 [=======================>......] - ETA: 2s - loss: 0.3408 - accuracy: 0.9409
20400/25000 [=======================>......] - ETA: 2s - loss: 0.3408 - accuracy: 0.9408
20500/25000 [=======================>......] - ETA: 2s - loss: 0.3409 - accuracy: 0.9408
20600/25000 [=======================>......] - ETA: 2s - loss: 0.3409 - accuracy: 0.9407
20700/25000 [=======================>......] - ETA: 2s - loss: 0.3407 - accuracy: 0.9409
20800/25000 [=======================>......] - ETA: 2s - loss: 0.3407 - accuracy: 0.9409
20900/25000 [========================>.....] - ETA: 2s - loss: 0.3408 - accuracy: 0.9408
21000/25000 [========================>.....] - ETA: 2s - loss: 0.3406 - accuracy: 0.9409
21100/25000 [========================>.....] - ETA: 2s - loss: 0.3407 - accuracy: 0.9408
21200/25000 [========================>.....] - ETA: 2s - loss: 0.3409 - accuracy: 0.9406
21300/25000 [========================>.....] - ETA: 2s - loss: 0.3411 - accuracy: 0.9405
21400/25000 [========================>.....] - ETA: 2s - loss: 0.3408 - accuracy: 0.9406
21500/25000 [========================>.....] - ETA: 2s - loss: 0.3407 - accuracy: 0.9407
21600/25000 [========================>.....] - ETA: 2s - loss: 0.3406 - accuracy: 0.9407
21700/25000 [=========================>....] - ETA: 2s - loss: 0.3407 - accuracy: 0.9406
21800/25000 [=========================>....] - ETA: 1s - loss: 0.3407 - accuracy: 0.9406
21900/25000 [=========================>....] - ETA: 1s - loss: 0.3407 - accuracy: 0.9405
22000/25000 [=========================>....] - ETA: 1s - loss: 0.3407 - accuracy: 0.9405
22100/25000 [=========================>....] - ETA: 1s - loss: 0.3409 - accuracy: 0.9403
22200/25000 [=========================>....] - ETA: 1s - loss: 0.3408 - accuracy: 0.9404
22300/25000 [=========================>....] - ETA: 1s - loss: 0.3407 - accuracy: 0.9404
22400/25000 [=========================>....] - ETA: 1s - loss: 0.3406 - accuracy: 0.9405
22500/25000 [==========================>...] - ETA: 1s - loss: 0.3407 - accuracy: 0.9404
22600/25000 [==========================>...] - ETA: 1s - loss: 0.3408 - accuracy: 0.9402
22700/25000 [==========================>...] - ETA: 1s - loss: 0.3408 - accuracy: 0.9402
22800/25000 [==========================>...] - ETA: 1s - loss: 0.3407 - accuracy: 0.9402
22900/25000 [==========================>...] - ETA: 1s - loss: 0.3407 - accuracy: 0.9402
23000/25000 [==========================>...] - ETA: 1s - loss: 0.3409 - accuracy: 0.9400
23100/25000 [==========================>...] - ETA: 1s - loss: 0.3409 - accuracy: 0.9400
23200/25000 [==========================>...] - ETA: 1s - loss: 0.3408 - accuracy: 0.9400
23300/25000 [==========================>...] - ETA: 1s - loss: 0.3407 - accuracy: 0.9401
23400/25000 [===========================>..] - ETA: 0s - loss: 0.3406 - accuracy: 0.9401
23500/25000 [===========================>..] - ETA: 0s - loss: 0.3406 - accuracy: 0.9400
23600/25000 [===========================>..] - ETA: 0s - loss: 0.3404 - accuracy: 0.9401
23700/25000 [===========================>..] - ETA: 0s - loss: 0.3404 - accuracy: 0.9400
23800/25000 [===========================>..] - ETA: 0s - loss: 0.3407 - accuracy: 0.9397
23900/25000 [===========================>..] - ETA: 0s - loss: 0.3409 - accuracy: 0.9395
24000/25000 [===========================>..] - ETA: 0s - loss: 0.3410 - accuracy: 0.9395
24100/25000 [===========================>..] - ETA: 0s - loss: 0.3411 - accuracy: 0.9393
24200/25000 [============================>.] - ETA: 0s - loss: 0.3410 - accuracy: 0.9395
24300/25000 [============================>.] - ETA: 0s - loss: 0.3409 - accuracy: 0.9395
24400/25000 [============================>.] - ETA: 0s - loss: 0.3410 - accuracy: 0.9394
24500/25000 [============================>.] - ETA: 0s - loss: 0.3411 - accuracy: 0.9392
24600/25000 [============================>.] - ETA: 0s - loss: 0.3413 - accuracy: 0.9391
24700/25000 [============================>.] - ETA: 0s - loss: 0.3410 - accuracy: 0.9392
24800/25000 [============================>.] - ETA: 0s - loss: 0.3410 - accuracy: 0.9392
24900/25000 [============================>.] - ETA: 0s - loss: 0.3411 - accuracy: 0.9390
25000/25000 [==============================] - 19s 770us/step - loss: 0.3410 - accuracy: 0.9391 - val_loss: 0.4173 - val_accuracy: 0.8628
Epoch 9/10

  100/25000 [..............................] - ETA: 14s - loss: 0.3588 - accuracy: 0.9200
  200/25000 [..............................] - ETA: 15s - loss: 0.3507 - accuracy: 0.9250
  300/25000 [..............................] - ETA: 14s - loss: 0.3264 - accuracy: 0.9467
  400/25000 [..............................] - ETA: 14s - loss: 0.3400 - accuracy: 0.9375
  500/25000 [..............................] - ETA: 14s - loss: 0.3326 - accuracy: 0.9400
  600/25000 [..............................] - ETA: 14s - loss: 0.3275 - accuracy: 0.9450
  700/25000 [..............................] - ETA: 14s - loss: 0.3263 - accuracy: 0.9457
  800/25000 [..............................] - ETA: 14s - loss: 0.3348 - accuracy: 0.9388
  900/25000 [>.............................] - ETA: 14s - loss: 0.3380 - accuracy: 0.9356
 1000/25000 [>.............................] - ETA: 14s - loss: 0.3351 - accuracy: 0.9380
 1100/25000 [>.............................] - ETA: 14s - loss: 0.3341 - accuracy: 0.9382
 1200/25000 [>.............................] - ETA: 14s - loss: 0.3309 - accuracy: 0.9408
 1300/25000 [>.............................] - ETA: 14s - loss: 0.3278 - accuracy: 0.9431
 1400/25000 [>.............................] - ETA: 14s - loss: 0.3271 - accuracy: 0.9436
 1500/25000 [>.............................] - ETA: 14s - loss: 0.3280 - accuracy: 0.9433
 1600/25000 [>.............................] - ETA: 14s - loss: 0.3276 - accuracy: 0.9431
 1700/25000 [=>............................] - ETA: 14s - loss: 0.3260 - accuracy: 0.9441
 1800/25000 [=>............................] - ETA: 14s - loss: 0.3273 - accuracy: 0.9422
 1900/25000 [=>............................] - ETA: 14s - loss: 0.3254 - accuracy: 0.9437
 2000/25000 [=>............................] - ETA: 14s - loss: 0.3278 - accuracy: 0.9415
 2100/25000 [=>............................] - ETA: 13s - loss: 0.3256 - accuracy: 0.9433
 2200/25000 [=>............................] - ETA: 13s - loss: 0.3228 - accuracy: 0.9455
 2300/25000 [=>............................] - ETA: 13s - loss: 0.3227 - accuracy: 0.9452
 2400/25000 [=>............................] - ETA: 13s - loss: 0.3243 - accuracy: 0.9438
 2500/25000 [==>...........................] - ETA: 13s - loss: 0.3256 - accuracy: 0.9424
 2600/25000 [==>...........................] - ETA: 13s - loss: 0.3252 - accuracy: 0.9427
 2700/25000 [==>...........................] - ETA: 13s - loss: 0.3262 - accuracy: 0.9419
 2800/25000 [==>...........................] - ETA: 13s - loss: 0.3256 - accuracy: 0.9425
 2900/25000 [==>...........................] - ETA: 13s - loss: 0.3259 - accuracy: 0.9424
 3000/25000 [==>...........................] - ETA: 13s - loss: 0.3249 - accuracy: 0.9430
 3100/25000 [==>...........................] - ETA: 13s - loss: 0.3238 - accuracy: 0.9439
 3200/25000 [==>...........................] - ETA: 13s - loss: 0.3239 - accuracy: 0.9438
 3300/25000 [==>...........................] - ETA: 13s - loss: 0.3235 - accuracy: 0.9439
 3400/25000 [===>..........................] - ETA: 13s - loss: 0.3235 - accuracy: 0.9438
 3500/25000 [===>..........................] - ETA: 13s - loss: 0.3224 - accuracy: 0.9446
 3600/25000 [===>..........................] - ETA: 13s - loss: 0.3225 - accuracy: 0.9442
 3700/25000 [===>..........................] - ETA: 13s - loss: 0.3227 - accuracy: 0.9435
 3800/25000 [===>..........................] - ETA: 12s - loss: 0.3222 - accuracy: 0.9439
 3900/25000 [===>..........................] - ETA: 12s - loss: 0.3232 - accuracy: 0.9433
 4000/25000 [===>..........................] - ETA: 12s - loss: 0.3224 - accuracy: 0.9440
 4100/25000 [===>..........................] - ETA: 12s - loss: 0.3216 - accuracy: 0.9446
 4200/25000 [====>.........................] - ETA: 12s - loss: 0.3222 - accuracy: 0.9443
 4300/25000 [====>.........................] - ETA: 12s - loss: 0.3219 - accuracy: 0.9444
 4400/25000 [====>.........................] - ETA: 12s - loss: 0.3210 - accuracy: 0.9452
 4500/25000 [====>.........................] - ETA: 12s - loss: 0.3218 - accuracy: 0.9447
 4600/25000 [====>.........................] - ETA: 12s - loss: 0.3215 - accuracy: 0.9448
 4700/25000 [====>.........................] - ETA: 12s - loss: 0.3207 - accuracy: 0.9455
 4800/25000 [====>.........................] - ETA: 12s - loss: 0.3208 - accuracy: 0.9454
 4900/25000 [====>.........................] - ETA: 12s - loss: 0.3212 - accuracy: 0.9449
 5000/25000 [=====>........................] - ETA: 12s - loss: 0.3220 - accuracy: 0.9444
 5100/25000 [=====>........................] - ETA: 12s - loss: 0.3226 - accuracy: 0.9441
 5200/25000 [=====>........................] - ETA: 12s - loss: 0.3224 - accuracy: 0.9442
 5300/25000 [=====>........................] - ETA: 12s - loss: 0.3217 - accuracy: 0.9447
 5400/25000 [=====>........................] - ETA: 11s - loss: 0.3219 - accuracy: 0.9446
 5500/25000 [=====>........................] - ETA: 11s - loss: 0.3222 - accuracy: 0.9444
 5600/25000 [=====>........................] - ETA: 11s - loss: 0.3217 - accuracy: 0.9448
 5700/25000 [=====>........................] - ETA: 11s - loss: 0.3211 - accuracy: 0.9453
 5800/25000 [=====>........................] - ETA: 11s - loss: 0.3214 - accuracy: 0.9450
 5900/25000 [======>.......................] - ETA: 11s - loss: 0.3205 - accuracy: 0.9456
 6000/25000 [======>.......................] - ETA: 11s - loss: 0.3199 - accuracy: 0.9460
 6100/25000 [======>.......................] - ETA: 11s - loss: 0.3193 - accuracy: 0.9466
 6200/25000 [======>.......................] - ETA: 11s - loss: 0.3186 - accuracy: 0.9471
 6300/25000 [======>.......................] - ETA: 11s - loss: 0.3179 - accuracy: 0.9478
 6400/25000 [======>.......................] - ETA: 11s - loss: 0.3182 - accuracy: 0.9473
 6500/25000 [======>.......................] - ETA: 11s - loss: 0.3175 - accuracy: 0.9478
 6600/25000 [======>.......................] - ETA: 11s - loss: 0.3177 - accuracy: 0.9477
 6700/25000 [=======>......................] - ETA: 11s - loss: 0.3171 - accuracy: 0.9482
 6800/25000 [=======>......................] - ETA: 11s - loss: 0.3170 - accuracy: 0.9482
 6900/25000 [=======>......................] - ETA: 11s - loss: 0.3163 - accuracy: 0.9488
 7000/25000 [=======>......................] - ETA: 10s - loss: 0.3165 - accuracy: 0.9487
 7100/25000 [=======>......................] - ETA: 10s - loss: 0.3168 - accuracy: 0.9485
 7200/25000 [=======>......................] - ETA: 10s - loss: 0.3167 - accuracy: 0.9485
 7300/25000 [=======>......................] - ETA: 10s - loss: 0.3169 - accuracy: 0.9484
 7400/25000 [=======>......................] - ETA: 10s - loss: 0.3167 - accuracy: 0.9484
 7500/25000 [========>.....................] - ETA: 10s - loss: 0.3162 - accuracy: 0.9488
 7600/25000 [========>.....................] - ETA: 10s - loss: 0.3162 - accuracy: 0.9487
 7700/25000 [========>.....................] - ETA: 10s - loss: 0.3161 - accuracy: 0.9488
 7800/25000 [========>.....................] - ETA: 10s - loss: 0.3161 - accuracy: 0.9488
 7900/25000 [========>.....................] - ETA: 10s - loss: 0.3161 - accuracy: 0.9489
 8000/25000 [========>.....................] - ETA: 10s - loss: 0.3162 - accuracy: 0.9489
 8100/25000 [========>.....................] - ETA: 10s - loss: 0.3163 - accuracy: 0.9488
 8200/25000 [========>.....................] - ETA: 10s - loss: 0.3163 - accuracy: 0.9488
 8300/25000 [========>.....................] - ETA: 10s - loss: 0.3163 - accuracy: 0.9487
 8400/25000 [=========>....................] - ETA: 10s - loss: 0.3165 - accuracy: 0.9485
 8500/25000 [=========>....................] - ETA: 10s - loss: 0.3168 - accuracy: 0.9482
 8600/25000 [=========>....................] - ETA: 9s - loss: 0.3171 - accuracy: 0.9479 
 8700/25000 [=========>....................] - ETA: 9s - loss: 0.3172 - accuracy: 0.9477
 8800/25000 [=========>....................] - ETA: 9s - loss: 0.3170 - accuracy: 0.9477
 8900/25000 [=========>....................] - ETA: 9s - loss: 0.3173 - accuracy: 0.9475
 9000/25000 [=========>....................] - ETA: 9s - loss: 0.3172 - accuracy: 0.9476
 9100/25000 [=========>....................] - ETA: 9s - loss: 0.3172 - accuracy: 0.9475
 9200/25000 [==========>...................] - ETA: 9s - loss: 0.3173 - accuracy: 0.9474
 9300/25000 [==========>...................] - ETA: 9s - loss: 0.3169 - accuracy: 0.9477
 9400/25000 [==========>...................] - ETA: 9s - loss: 0.3169 - accuracy: 0.9478
 9500/25000 [==========>...................] - ETA: 9s - loss: 0.3166 - accuracy: 0.9480
 9600/25000 [==========>...................] - ETA: 9s - loss: 0.3165 - accuracy: 0.9480
 9700/25000 [==========>...................] - ETA: 9s - loss: 0.3162 - accuracy: 0.9482
 9800/25000 [==========>...................] - ETA: 9s - loss: 0.3165 - accuracy: 0.9481
 9900/25000 [==========>...................] - ETA: 9s - loss: 0.3164 - accuracy: 0.9481
10000/25000 [===========>..................] - ETA: 9s - loss: 0.3161 - accuracy: 0.9484
10100/25000 [===========>..................] - ETA: 9s - loss: 0.3161 - accuracy: 0.9484
10200/25000 [===========>..................] - ETA: 9s - loss: 0.3162 - accuracy: 0.9481
10300/25000 [===========>..................] - ETA: 8s - loss: 0.3162 - accuracy: 0.9482
10400/25000 [===========>..................] - ETA: 8s - loss: 0.3168 - accuracy: 0.9477
10500/25000 [===========>..................] - ETA: 8s - loss: 0.3171 - accuracy: 0.9474
10600/25000 [===========>..................] - ETA: 8s - loss: 0.3168 - accuracy: 0.9475
10700/25000 [===========>..................] - ETA: 8s - loss: 0.3169 - accuracy: 0.9475
10800/25000 [===========>..................] - ETA: 8s - loss: 0.3165 - accuracy: 0.9477
10900/25000 [============>.................] - ETA: 8s - loss: 0.3166 - accuracy: 0.9476
11000/25000 [============>.................] - ETA: 8s - loss: 0.3170 - accuracy: 0.9473
11100/25000 [============>.................] - ETA: 8s - loss: 0.3174 - accuracy: 0.9469
11200/25000 [============>.................] - ETA: 8s - loss: 0.3176 - accuracy: 0.9467
11300/25000 [============>.................] - ETA: 8s - loss: 0.3186 - accuracy: 0.9458
11400/25000 [============>.................] - ETA: 8s - loss: 0.3185 - accuracy: 0.9459
11500/25000 [============>.................] - ETA: 8s - loss: 0.3184 - accuracy: 0.9460
11600/25000 [============>.................] - ETA: 8s - loss: 0.3183 - accuracy: 0.9460
11700/25000 [=============>................] - ETA: 8s - loss: 0.3184 - accuracy: 0.9460
11800/25000 [=============>................] - ETA: 8s - loss: 0.3187 - accuracy: 0.9458
11900/25000 [=============>................] - ETA: 7s - loss: 0.3184 - accuracy: 0.9460
12000/25000 [=============>................] - ETA: 7s - loss: 0.3181 - accuracy: 0.9463
12100/25000 [=============>................] - ETA: 7s - loss: 0.3184 - accuracy: 0.9460
12200/25000 [=============>................] - ETA: 7s - loss: 0.3182 - accuracy: 0.9461
12300/25000 [=============>................] - ETA: 7s - loss: 0.3182 - accuracy: 0.9460
12400/25000 [=============>................] - ETA: 7s - loss: 0.3180 - accuracy: 0.9461
12500/25000 [==============>...............] - ETA: 7s - loss: 0.3181 - accuracy: 0.9462
12600/25000 [==============>...............] - ETA: 7s - loss: 0.3180 - accuracy: 0.9462
12700/25000 [==============>...............] - ETA: 7s - loss: 0.3181 - accuracy: 0.9462
12800/25000 [==============>...............] - ETA: 7s - loss: 0.3182 - accuracy: 0.9462
12900/25000 [==============>...............] - ETA: 7s - loss: 0.3179 - accuracy: 0.9464
13000/25000 [==============>...............] - ETA: 7s - loss: 0.3182 - accuracy: 0.9461
13100/25000 [==============>...............] - ETA: 7s - loss: 0.3184 - accuracy: 0.9459
13200/25000 [==============>...............] - ETA: 7s - loss: 0.3181 - accuracy: 0.9460
13300/25000 [==============>...............] - ETA: 7s - loss: 0.3179 - accuracy: 0.9461
13400/25000 [===============>..............] - ETA: 7s - loss: 0.3180 - accuracy: 0.9460
13500/25000 [===============>..............] - ETA: 7s - loss: 0.3181 - accuracy: 0.9459
13600/25000 [===============>..............] - ETA: 6s - loss: 0.3181 - accuracy: 0.9458
13700/25000 [===============>..............] - ETA: 6s - loss: 0.3181 - accuracy: 0.9458
13800/25000 [===============>..............] - ETA: 6s - loss: 0.3181 - accuracy: 0.9458
13900/25000 [===============>..............] - ETA: 6s - loss: 0.3180 - accuracy: 0.9458
14000/25000 [===============>..............] - ETA: 6s - loss: 0.3177 - accuracy: 0.9461
14100/25000 [===============>..............] - ETA: 6s - loss: 0.3176 - accuracy: 0.9461
14200/25000 [================>.............] - ETA: 6s - loss: 0.3174 - accuracy: 0.9463
14300/25000 [================>.............] - ETA: 6s - loss: 0.3172 - accuracy: 0.9464
14400/25000 [================>.............] - ETA: 6s - loss: 0.3178 - accuracy: 0.9460
14500/25000 [================>.............] - ETA: 6s - loss: 0.3181 - accuracy: 0.9457
14600/25000 [================>.............] - ETA: 6s - loss: 0.3182 - accuracy: 0.9456
14700/25000 [================>.............] - ETA: 6s - loss: 0.3182 - accuracy: 0.9456
14800/25000 [================>.............] - ETA: 6s - loss: 0.3179 - accuracy: 0.9457
14900/25000 [================>.............] - ETA: 6s - loss: 0.3178 - accuracy: 0.9459
15000/25000 [=================>............] - ETA: 6s - loss: 0.3174 - accuracy: 0.9462
15100/25000 [=================>............] - ETA: 6s - loss: 0.3175 - accuracy: 0.9461
15200/25000 [=================>............] - ETA: 5s - loss: 0.3175 - accuracy: 0.9460
15300/25000 [=================>............] - ETA: 5s - loss: 0.3175 - accuracy: 0.9460
15400/25000 [=================>............] - ETA: 5s - loss: 0.3177 - accuracy: 0.9458
15500/25000 [=================>............] - ETA: 5s - loss: 0.3177 - accuracy: 0.9459
15600/25000 [=================>............] - ETA: 5s - loss: 0.3176 - accuracy: 0.9459
15700/25000 [=================>............] - ETA: 5s - loss: 0.3177 - accuracy: 0.9458
15800/25000 [=================>............] - ETA: 5s - loss: 0.3178 - accuracy: 0.9457
15900/25000 [==================>...........] - ETA: 5s - loss: 0.3176 - accuracy: 0.9458
16000/25000 [==================>...........] - ETA: 5s - loss: 0.3177 - accuracy: 0.9457
16100/25000 [==================>...........] - ETA: 5s - loss: 0.3177 - accuracy: 0.9457
16200/25000 [==================>...........] - ETA: 5s - loss: 0.3176 - accuracy: 0.9457
16300/25000 [==================>...........] - ETA: 5s - loss: 0.3179 - accuracy: 0.9455
16400/25000 [==================>...........] - ETA: 5s - loss: 0.3179 - accuracy: 0.9455
16500/25000 [==================>...........] - ETA: 5s - loss: 0.3179 - accuracy: 0.9455
16600/25000 [==================>...........] - ETA: 5s - loss: 0.3178 - accuracy: 0.9455
16700/25000 [===================>..........] - ETA: 5s - loss: 0.3176 - accuracy: 0.9456
16800/25000 [===================>..........] - ETA: 4s - loss: 0.3178 - accuracy: 0.9455
16900/25000 [===================>..........] - ETA: 4s - loss: 0.3179 - accuracy: 0.9454
17000/25000 [===================>..........] - ETA: 4s - loss: 0.3180 - accuracy: 0.9453
17100/25000 [===================>..........] - ETA: 4s - loss: 0.3182 - accuracy: 0.9451
17200/25000 [===================>..........] - ETA: 4s - loss: 0.3180 - accuracy: 0.9452
17300/25000 [===================>..........] - ETA: 4s - loss: 0.3181 - accuracy: 0.9451
17400/25000 [===================>..........] - ETA: 4s - loss: 0.3179 - accuracy: 0.9453
17500/25000 [====================>.........] - ETA: 4s - loss: 0.3177 - accuracy: 0.9454
17600/25000 [====================>.........] - ETA: 4s - loss: 0.3177 - accuracy: 0.9455
17700/25000 [====================>.........] - ETA: 4s - loss: 0.3177 - accuracy: 0.9454
17800/25000 [====================>.........] - ETA: 4s - loss: 0.3175 - accuracy: 0.9455
17900/25000 [====================>.........] - ETA: 4s - loss: 0.3179 - accuracy: 0.9452
18000/25000 [====================>.........] - ETA: 4s - loss: 0.3177 - accuracy: 0.9453
18100/25000 [====================>.........] - ETA: 4s - loss: 0.3179 - accuracy: 0.9451
18200/25000 [====================>.........] - ETA: 4s - loss: 0.3179 - accuracy: 0.9451
18300/25000 [====================>.........] - ETA: 4s - loss: 0.3181 - accuracy: 0.9449
18400/25000 [=====================>........] - ETA: 4s - loss: 0.3181 - accuracy: 0.9449
18500/25000 [=====================>........] - ETA: 3s - loss: 0.3180 - accuracy: 0.9449
18600/25000 [=====================>........] - ETA: 3s - loss: 0.3179 - accuracy: 0.9449
18700/25000 [=====================>........] - ETA: 3s - loss: 0.3177 - accuracy: 0.9450
18800/25000 [=====================>........] - ETA: 3s - loss: 0.3175 - accuracy: 0.9452
18900/25000 [=====================>........] - ETA: 3s - loss: 0.3174 - accuracy: 0.9453
19000/25000 [=====================>........] - ETA: 3s - loss: 0.3174 - accuracy: 0.9452
19100/25000 [=====================>........] - ETA: 3s - loss: 0.3176 - accuracy: 0.9451
19200/25000 [======================>.......] - ETA: 3s - loss: 0.3177 - accuracy: 0.9449
19300/25000 [======================>.......] - ETA: 3s - loss: 0.3176 - accuracy: 0.9450
19400/25000 [======================>.......] - ETA: 3s - loss: 0.3176 - accuracy: 0.9450
19500/25000 [======================>.......] - ETA: 3s - loss: 0.3174 - accuracy: 0.9450
19600/25000 [======================>.......] - ETA: 3s - loss: 0.3175 - accuracy: 0.9450
19700/25000 [======================>.......] - ETA: 3s - loss: 0.3175 - accuracy: 0.9449
19800/25000 [======================>.......] - ETA: 3s - loss: 0.3173 - accuracy: 0.9451
19900/25000 [======================>.......] - ETA: 3s - loss: 0.3172 - accuracy: 0.9451
20000/25000 [=======================>......] - ETA: 3s - loss: 0.3171 - accuracy: 0.9451
20100/25000 [=======================>......] - ETA: 2s - loss: 0.3169 - accuracy: 0.9453
20200/25000 [=======================>......] - ETA: 2s - loss: 0.3168 - accuracy: 0.9453
20300/25000 [=======================>......] - ETA: 2s - loss: 0.3166 - accuracy: 0.9455
20400/25000 [=======================>......] - ETA: 2s - loss: 0.3168 - accuracy: 0.9453
20500/25000 [=======================>......] - ETA: 2s - loss: 0.3169 - accuracy: 0.9452
20600/25000 [=======================>......] - ETA: 2s - loss: 0.3169 - accuracy: 0.9452
20700/25000 [=======================>......] - ETA: 2s - loss: 0.3167 - accuracy: 0.9454
20800/25000 [=======================>......] - ETA: 2s - loss: 0.3168 - accuracy: 0.9452
20900/25000 [========================>.....] - ETA: 2s - loss: 0.3168 - accuracy: 0.9452
21000/25000 [========================>.....] - ETA: 2s - loss: 0.3168 - accuracy: 0.9452
21100/25000 [========================>.....] - ETA: 2s - loss: 0.3168 - accuracy: 0.9451
21200/25000 [========================>.....] - ETA: 2s - loss: 0.3168 - accuracy: 0.9451
21300/25000 [========================>.....] - ETA: 2s - loss: 0.3169 - accuracy: 0.9450
21400/25000 [========================>.....] - ETA: 2s - loss: 0.3169 - accuracy: 0.9450
21500/25000 [========================>.....] - ETA: 2s - loss: 0.3170 - accuracy: 0.9448
21600/25000 [========================>.....] - ETA: 2s - loss: 0.3168 - accuracy: 0.9450
21700/25000 [=========================>....] - ETA: 2s - loss: 0.3168 - accuracy: 0.9449
21800/25000 [=========================>....] - ETA: 1s - loss: 0.3168 - accuracy: 0.9449
21900/25000 [=========================>....] - ETA: 1s - loss: 0.3168 - accuracy: 0.9448
22000/25000 [=========================>....] - ETA: 1s - loss: 0.3168 - accuracy: 0.9448
22100/25000 [=========================>....] - ETA: 1s - loss: 0.3167 - accuracy: 0.9449
22200/25000 [=========================>....] - ETA: 1s - loss: 0.3164 - accuracy: 0.9451
22300/25000 [=========================>....] - ETA: 1s - loss: 0.3164 - accuracy: 0.9451
22400/25000 [=========================>....] - ETA: 1s - loss: 0.3163 - accuracy: 0.9451
22500/25000 [==========================>...] - ETA: 1s - loss: 0.3164 - accuracy: 0.9450
22600/25000 [==========================>...] - ETA: 1s - loss: 0.3164 - accuracy: 0.9450
22700/25000 [==========================>...] - ETA: 1s - loss: 0.3165 - accuracy: 0.9449
22800/25000 [==========================>...] - ETA: 1s - loss: 0.3164 - accuracy: 0.9449
22900/25000 [==========================>...] - ETA: 1s - loss: 0.3164 - accuracy: 0.9449
23000/25000 [==========================>...] - ETA: 1s - loss: 0.3163 - accuracy: 0.9449
23100/25000 [==========================>...] - ETA: 1s - loss: 0.3164 - accuracy: 0.9448
23200/25000 [==========================>...] - ETA: 1s - loss: 0.3167 - accuracy: 0.9446
23300/25000 [==========================>...] - ETA: 1s - loss: 0.3166 - accuracy: 0.9445
23400/25000 [===========================>..] - ETA: 0s - loss: 0.3166 - accuracy: 0.9446
23500/25000 [===========================>..] - ETA: 0s - loss: 0.3163 - accuracy: 0.9448
23600/25000 [===========================>..] - ETA: 0s - loss: 0.3162 - accuracy: 0.9448
23700/25000 [===========================>..] - ETA: 0s - loss: 0.3159 - accuracy: 0.9450
23800/25000 [===========================>..] - ETA: 0s - loss: 0.3159 - accuracy: 0.9450
23900/25000 [===========================>..] - ETA: 0s - loss: 0.3159 - accuracy: 0.9450
24000/25000 [===========================>..] - ETA: 0s - loss: 0.3157 - accuracy: 0.9452
24100/25000 [===========================>..] - ETA: 0s - loss: 0.3155 - accuracy: 0.9453
24200/25000 [============================>.] - ETA: 0s - loss: 0.3155 - accuracy: 0.9452
24300/25000 [============================>.] - ETA: 0s - loss: 0.3155 - accuracy: 0.9451
24400/25000 [============================>.] - ETA: 0s - loss: 0.3156 - accuracy: 0.9451
24500/25000 [============================>.] - ETA: 0s - loss: 0.3154 - accuracy: 0.9452
24600/25000 [============================>.] - ETA: 0s - loss: 0.3154 - accuracy: 0.9452
24700/25000 [============================>.] - ETA: 0s - loss: 0.3152 - accuracy: 0.9453
24800/25000 [============================>.] - ETA: 0s - loss: 0.3151 - accuracy: 0.9454
24900/25000 [============================>.] - ETA: 0s - loss: 0.3149 - accuracy: 0.9455
25000/25000 [==============================] - 19s 770us/step - loss: 0.3152 - accuracy: 0.9453 - val_loss: 0.4123 - val_accuracy: 0.8577
Epoch 10/10

  100/25000 [..............................] - ETA: 15s - loss: 0.2791 - accuracy: 0.9600
  200/25000 [..............................] - ETA: 15s - loss: 0.2980 - accuracy: 0.9500
  300/25000 [..............................] - ETA: 14s - loss: 0.3001 - accuracy: 0.9500
  400/25000 [..............................] - ETA: 15s - loss: 0.2953 - accuracy: 0.9525
  500/25000 [..............................] - ETA: 14s - loss: 0.3035 - accuracy: 0.9480
  600/25000 [..............................] - ETA: 14s - loss: 0.2956 - accuracy: 0.9517
  700/25000 [..............................] - ETA: 14s - loss: 0.2963 - accuracy: 0.9514
  800/25000 [..............................] - ETA: 14s - loss: 0.2943 - accuracy: 0.9538
  900/25000 [>.............................] - ETA: 14s - loss: 0.2977 - accuracy: 0.9511
 1000/25000 [>.............................] - ETA: 14s - loss: 0.2979 - accuracy: 0.9510
 1100/25000 [>.............................] - ETA: 14s - loss: 0.2984 - accuracy: 0.9509
 1200/25000 [>.............................] - ETA: 14s - loss: 0.2943 - accuracy: 0.9542
 1300/25000 [>.............................] - ETA: 14s - loss: 0.2938 - accuracy: 0.9554
 1400/25000 [>.............................] - ETA: 14s - loss: 0.2957 - accuracy: 0.9536
 1500/25000 [>.............................] - ETA: 14s - loss: 0.2970 - accuracy: 0.9527
 1600/25000 [>.............................] - ETA: 14s - loss: 0.2973 - accuracy: 0.9525
 1700/25000 [=>............................] - ETA: 14s - loss: 0.2972 - accuracy: 0.9524
 1800/25000 [=>............................] - ETA: 14s - loss: 0.2967 - accuracy: 0.9528
 1900/25000 [=>............................] - ETA: 13s - loss: 0.2943 - accuracy: 0.9547
 2000/25000 [=>............................] - ETA: 13s - loss: 0.2956 - accuracy: 0.9535
 2100/25000 [=>............................] - ETA: 13s - loss: 0.2944 - accuracy: 0.9543
 2200/25000 [=>............................] - ETA: 13s - loss: 0.2942 - accuracy: 0.9541
 2300/25000 [=>............................] - ETA: 13s - loss: 0.2945 - accuracy: 0.9539
 2400/25000 [=>............................] - ETA: 13s - loss: 0.2935 - accuracy: 0.9546
 2500/25000 [==>...........................] - ETA: 13s - loss: 0.2919 - accuracy: 0.9556
 2600/25000 [==>...........................] - ETA: 13s - loss: 0.2918 - accuracy: 0.9558
 2700/25000 [==>...........................] - ETA: 13s - loss: 0.2913 - accuracy: 0.9563
 2800/25000 [==>...........................] - ETA: 13s - loss: 0.2935 - accuracy: 0.9546
 2900/25000 [==>...........................] - ETA: 13s - loss: 0.2939 - accuracy: 0.9545
 3000/25000 [==>...........................] - ETA: 13s - loss: 0.2937 - accuracy: 0.9547
 3100/25000 [==>...........................] - ETA: 13s - loss: 0.2948 - accuracy: 0.9539
 3200/25000 [==>...........................] - ETA: 13s - loss: 0.2953 - accuracy: 0.9538
 3300/25000 [==>...........................] - ETA: 13s - loss: 0.2954 - accuracy: 0.9536
 3400/25000 [===>..........................] - ETA: 13s - loss: 0.2958 - accuracy: 0.9532
 3500/25000 [===>..........................] - ETA: 13s - loss: 0.2967 - accuracy: 0.9526
 3600/25000 [===>..........................] - ETA: 12s - loss: 0.2954 - accuracy: 0.9533
 3700/25000 [===>..........................] - ETA: 12s - loss: 0.2949 - accuracy: 0.9535
 3800/25000 [===>..........................] - ETA: 12s - loss: 0.2948 - accuracy: 0.9537
 3900/25000 [===>..........................] - ETA: 12s - loss: 0.2947 - accuracy: 0.9538
 4000/25000 [===>..........................] - ETA: 12s - loss: 0.2954 - accuracy: 0.9535
 4100/25000 [===>..........................] - ETA: 12s - loss: 0.2947 - accuracy: 0.9539
 4200/25000 [====>.........................] - ETA: 12s - loss: 0.2939 - accuracy: 0.9545
 4300/25000 [====>.........................] - ETA: 12s - loss: 0.2941 - accuracy: 0.9542
 4400/25000 [====>.........................] - ETA: 12s - loss: 0.2940 - accuracy: 0.9541
 4500/25000 [====>.........................] - ETA: 12s - loss: 0.2947 - accuracy: 0.9536
 4600/25000 [====>.........................] - ETA: 12s - loss: 0.2948 - accuracy: 0.9535
 4700/25000 [====>.........................] - ETA: 12s - loss: 0.2953 - accuracy: 0.9532
 4800/25000 [====>.........................] - ETA: 12s - loss: 0.2958 - accuracy: 0.9529
 4900/25000 [====>.........................] - ETA: 12s - loss: 0.2974 - accuracy: 0.9516
 5000/25000 [=====>........................] - ETA: 12s - loss: 0.2970 - accuracy: 0.9520
 5100/25000 [=====>........................] - ETA: 12s - loss: 0.2978 - accuracy: 0.9514
 5200/25000 [=====>........................] - ETA: 12s - loss: 0.2980 - accuracy: 0.9513
 5300/25000 [=====>........................] - ETA: 11s - loss: 0.2977 - accuracy: 0.9515
 5400/25000 [=====>........................] - ETA: 11s - loss: 0.2980 - accuracy: 0.9513
 5500/25000 [=====>........................] - ETA: 11s - loss: 0.2977 - accuracy: 0.9515
 5600/25000 [=====>........................] - ETA: 11s - loss: 0.2976 - accuracy: 0.9513
 5700/25000 [=====>........................] - ETA: 11s - loss: 0.2971 - accuracy: 0.9516
 5800/25000 [=====>........................] - ETA: 11s - loss: 0.2969 - accuracy: 0.9516
 5900/25000 [======>.......................] - ETA: 11s - loss: 0.2969 - accuracy: 0.9515
 6000/25000 [======>.......................] - ETA: 11s - loss: 0.2964 - accuracy: 0.9520
 6100/25000 [======>.......................] - ETA: 11s - loss: 0.2961 - accuracy: 0.9521
 6200/25000 [======>.......................] - ETA: 11s - loss: 0.2962 - accuracy: 0.9521
 6300/25000 [======>.......................] - ETA: 11s - loss: 0.2961 - accuracy: 0.9522
 6400/25000 [======>.......................] - ETA: 11s - loss: 0.2961 - accuracy: 0.9522
 6500/25000 [======>.......................] - ETA: 11s - loss: 0.2959 - accuracy: 0.9523
 6600/25000 [======>.......................] - ETA: 11s - loss: 0.2962 - accuracy: 0.9521
 6700/25000 [=======>......................] - ETA: 11s - loss: 0.2963 - accuracy: 0.9521
 6800/25000 [=======>......................] - ETA: 11s - loss: 0.2964 - accuracy: 0.9521
 6900/25000 [=======>......................] - ETA: 10s - loss: 0.2961 - accuracy: 0.9523
 7000/25000 [=======>......................] - ETA: 10s - loss: 0.2967 - accuracy: 0.9519
 7100/25000 [=======>......................] - ETA: 10s - loss: 0.2969 - accuracy: 0.9517
 7200/25000 [=======>......................] - ETA: 10s - loss: 0.2963 - accuracy: 0.9521
 7300/25000 [=======>......................] - ETA: 10s - loss: 0.2963 - accuracy: 0.9521
 7400/25000 [=======>......................] - ETA: 10s - loss: 0.2959 - accuracy: 0.9523
 7500/25000 [========>.....................] - ETA: 10s - loss: 0.2965 - accuracy: 0.9520
 7600/25000 [========>.....................] - ETA: 10s - loss: 0.2961 - accuracy: 0.9522
 7700/25000 [========>.....................] - ETA: 10s - loss: 0.2963 - accuracy: 0.9521
 7800/25000 [========>.....................] - ETA: 10s - loss: 0.2978 - accuracy: 0.9510
 7900/25000 [========>.....................] - ETA: 10s - loss: 0.2976 - accuracy: 0.9511
 8000/25000 [========>.....................] - ETA: 10s - loss: 0.2981 - accuracy: 0.9506
 8100/25000 [========>.....................] - ETA: 10s - loss: 0.2985 - accuracy: 0.9502
 8200/25000 [========>.....................] - ETA: 10s - loss: 0.2982 - accuracy: 0.9504
 8300/25000 [========>.....................] - ETA: 10s - loss: 0.2981 - accuracy: 0.9505
 8400/25000 [=========>....................] - ETA: 10s - loss: 0.2978 - accuracy: 0.9507
 8500/25000 [=========>....................] - ETA: 10s - loss: 0.2976 - accuracy: 0.9508
 8600/25000 [=========>....................] - ETA: 9s - loss: 0.2972 - accuracy: 0.9512 
 8700/25000 [=========>....................] - ETA: 9s - loss: 0.2972 - accuracy: 0.9511
 8800/25000 [=========>....................] - ETA: 9s - loss: 0.2971 - accuracy: 0.9511
 8900/25000 [=========>....................] - ETA: 9s - loss: 0.2974 - accuracy: 0.9508
 9000/25000 [=========>....................] - ETA: 9s - loss: 0.2971 - accuracy: 0.9509
 9100/25000 [=========>....................] - ETA: 9s - loss: 0.2970 - accuracy: 0.9509
 9200/25000 [==========>...................] - ETA: 9s - loss: 0.2975 - accuracy: 0.9505
 9300/25000 [==========>...................] - ETA: 9s - loss: 0.2975 - accuracy: 0.9505
 9400/25000 [==========>...................] - ETA: 9s - loss: 0.2982 - accuracy: 0.9500
 9500/25000 [==========>...................] - ETA: 9s - loss: 0.2978 - accuracy: 0.9503
 9600/25000 [==========>...................] - ETA: 9s - loss: 0.2977 - accuracy: 0.9502
 9700/25000 [==========>...................] - ETA: 9s - loss: 0.2974 - accuracy: 0.9504
 9800/25000 [==========>...................] - ETA: 9s - loss: 0.2982 - accuracy: 0.9499
 9900/25000 [==========>...................] - ETA: 9s - loss: 0.2980 - accuracy: 0.9501
10000/25000 [===========>..................] - ETA: 9s - loss: 0.2978 - accuracy: 0.9503
10100/25000 [===========>..................] - ETA: 9s - loss: 0.2975 - accuracy: 0.9505
10200/25000 [===========>..................] - ETA: 9s - loss: 0.2972 - accuracy: 0.9507
10300/25000 [===========>..................] - ETA: 8s - loss: 0.2973 - accuracy: 0.9506
10400/25000 [===========>..................] - ETA: 8s - loss: 0.2973 - accuracy: 0.9506
10500/25000 [===========>..................] - ETA: 8s - loss: 0.2970 - accuracy: 0.9508
10600/25000 [===========>..................] - ETA: 8s - loss: 0.2970 - accuracy: 0.9508
10700/25000 [===========>..................] - ETA: 8s - loss: 0.2964 - accuracy: 0.9511
10800/25000 [===========>..................] - ETA: 8s - loss: 0.2961 - accuracy: 0.9514
10900/25000 [============>.................] - ETA: 8s - loss: 0.2961 - accuracy: 0.9514
11000/25000 [============>.................] - ETA: 8s - loss: 0.2966 - accuracy: 0.9509
11100/25000 [============>.................] - ETA: 8s - loss: 0.2964 - accuracy: 0.9511
11200/25000 [============>.................] - ETA: 8s - loss: 0.2962 - accuracy: 0.9512
11300/25000 [============>.................] - ETA: 8s - loss: 0.2960 - accuracy: 0.9512
11400/25000 [============>.................] - ETA: 8s - loss: 0.2960 - accuracy: 0.9512
11500/25000 [============>.................] - ETA: 8s - loss: 0.2956 - accuracy: 0.9515
11600/25000 [============>.................] - ETA: 8s - loss: 0.2955 - accuracy: 0.9515
11700/25000 [=============>................] - ETA: 8s - loss: 0.2954 - accuracy: 0.9515
11800/25000 [=============>................] - ETA: 8s - loss: 0.2956 - accuracy: 0.9514
11900/25000 [=============>................] - ETA: 7s - loss: 0.2960 - accuracy: 0.9510
12000/25000 [=============>................] - ETA: 7s - loss: 0.2959 - accuracy: 0.9511
12100/25000 [=============>................] - ETA: 7s - loss: 0.2962 - accuracy: 0.9509
12200/25000 [=============>................] - ETA: 7s - loss: 0.2964 - accuracy: 0.9507
12300/25000 [=============>................] - ETA: 7s - loss: 0.2963 - accuracy: 0.9507
12400/25000 [=============>................] - ETA: 7s - loss: 0.2962 - accuracy: 0.9508
12500/25000 [==============>...............] - ETA: 7s - loss: 0.2964 - accuracy: 0.9506
12600/25000 [==============>...............] - ETA: 7s - loss: 0.2963 - accuracy: 0.9506
12700/25000 [==============>...............] - ETA: 7s - loss: 0.2962 - accuracy: 0.9507
12800/25000 [==============>...............] - ETA: 7s - loss: 0.2963 - accuracy: 0.9506
12900/25000 [==============>...............] - ETA: 7s - loss: 0.2965 - accuracy: 0.9504
13000/25000 [==============>...............] - ETA: 7s - loss: 0.2964 - accuracy: 0.9505
13100/25000 [==============>...............] - ETA: 7s - loss: 0.2964 - accuracy: 0.9505
13200/25000 [==============>...............] - ETA: 7s - loss: 0.2961 - accuracy: 0.9507
13300/25000 [==============>...............] - ETA: 7s - loss: 0.2961 - accuracy: 0.9506
13400/25000 [===============>..............] - ETA: 7s - loss: 0.2957 - accuracy: 0.9509
13500/25000 [===============>..............] - ETA: 7s - loss: 0.2957 - accuracy: 0.9508
13600/25000 [===============>..............] - ETA: 6s - loss: 0.2955 - accuracy: 0.9509
13700/25000 [===============>..............] - ETA: 6s - loss: 0.2955 - accuracy: 0.9509
13800/25000 [===============>..............] - ETA: 6s - loss: 0.2954 - accuracy: 0.9509
13900/25000 [===============>..............] - ETA: 6s - loss: 0.2951 - accuracy: 0.9512
14000/25000 [===============>..............] - ETA: 6s - loss: 0.2952 - accuracy: 0.9511
14100/25000 [===============>..............] - ETA: 6s - loss: 0.2954 - accuracy: 0.9510
14200/25000 [================>.............] - ETA: 6s - loss: 0.2955 - accuracy: 0.9509
14300/25000 [================>.............] - ETA: 6s - loss: 0.2954 - accuracy: 0.9509
14400/25000 [================>.............] - ETA: 6s - loss: 0.2953 - accuracy: 0.9509
14500/25000 [================>.............] - ETA: 6s - loss: 0.2954 - accuracy: 0.9508
14600/25000 [================>.............] - ETA: 6s - loss: 0.2952 - accuracy: 0.9508
14700/25000 [================>.............] - ETA: 6s - loss: 0.2951 - accuracy: 0.9509
14800/25000 [================>.............] - ETA: 6s - loss: 0.2949 - accuracy: 0.9510
14900/25000 [================>.............] - ETA: 6s - loss: 0.2948 - accuracy: 0.9511
15000/25000 [=================>............] - ETA: 6s - loss: 0.2949 - accuracy: 0.9510
15100/25000 [=================>............] - ETA: 6s - loss: 0.2947 - accuracy: 0.9511
15200/25000 [=================>............] - ETA: 5s - loss: 0.2950 - accuracy: 0.9509
15300/25000 [=================>............] - ETA: 5s - loss: 0.2952 - accuracy: 0.9507
15400/25000 [=================>............] - ETA: 5s - loss: 0.2953 - accuracy: 0.9506
15500/25000 [=================>............] - ETA: 5s - loss: 0.2951 - accuracy: 0.9508
15600/25000 [=================>............] - ETA: 5s - loss: 0.2952 - accuracy: 0.9506
15700/25000 [=================>............] - ETA: 5s - loss: 0.2951 - accuracy: 0.9506
15800/25000 [=================>............] - ETA: 5s - loss: 0.2952 - accuracy: 0.9504
15900/25000 [==================>...........] - ETA: 5s - loss: 0.2951 - accuracy: 0.9505
16000/25000 [==================>...........] - ETA: 5s - loss: 0.2951 - accuracy: 0.9505
16100/25000 [==================>...........] - ETA: 5s - loss: 0.2949 - accuracy: 0.9506
16200/25000 [==================>...........] - ETA: 5s - loss: 0.2946 - accuracy: 0.9508
16300/25000 [==================>...........] - ETA: 5s - loss: 0.2946 - accuracy: 0.9509
16400/25000 [==================>...........] - ETA: 5s - loss: 0.2947 - accuracy: 0.9508
16500/25000 [==================>...........] - ETA: 5s - loss: 0.2947 - accuracy: 0.9508
16600/25000 [==================>...........] - ETA: 5s - loss: 0.2945 - accuracy: 0.9509
16700/25000 [===================>..........] - ETA: 5s - loss: 0.2945 - accuracy: 0.9508
16800/25000 [===================>..........] - ETA: 4s - loss: 0.2946 - accuracy: 0.9508
16900/25000 [===================>..........] - ETA: 4s - loss: 0.2944 - accuracy: 0.9509
17000/25000 [===================>..........] - ETA: 4s - loss: 0.2942 - accuracy: 0.9510
17100/25000 [===================>..........] - ETA: 4s - loss: 0.2942 - accuracy: 0.9510
17200/25000 [===================>..........] - ETA: 4s - loss: 0.2941 - accuracy: 0.9510
17300/25000 [===================>..........] - ETA: 4s - loss: 0.2940 - accuracy: 0.9511
17400/25000 [===================>..........] - ETA: 4s - loss: 0.2939 - accuracy: 0.9511
17500/25000 [====================>.........] - ETA: 4s - loss: 0.2939 - accuracy: 0.9510
17600/25000 [====================>.........] - ETA: 4s - loss: 0.2938 - accuracy: 0.9511
17700/25000 [====================>.........] - ETA: 4s - loss: 0.2939 - accuracy: 0.9510
17800/25000 [====================>.........] - ETA: 4s - loss: 0.2937 - accuracy: 0.9511
17900/25000 [====================>.........] - ETA: 4s - loss: 0.2937 - accuracy: 0.9511
18000/25000 [====================>.........] - ETA: 4s - loss: 0.2938 - accuracy: 0.9510
18100/25000 [====================>.........] - ETA: 4s - loss: 0.2940 - accuracy: 0.9508
18200/25000 [====================>.........] - ETA: 4s - loss: 0.2938 - accuracy: 0.9509
18300/25000 [====================>.........] - ETA: 4s - loss: 0.2936 - accuracy: 0.9511
18400/25000 [=====================>........] - ETA: 4s - loss: 0.2935 - accuracy: 0.9511
18500/25000 [=====================>........] - ETA: 3s - loss: 0.2933 - accuracy: 0.9512
18600/25000 [=====================>........] - ETA: 3s - loss: 0.2931 - accuracy: 0.9513
18700/25000 [=====================>........] - ETA: 3s - loss: 0.2932 - accuracy: 0.9512
18800/25000 [=====================>........] - ETA: 3s - loss: 0.2934 - accuracy: 0.9511
18900/25000 [=====================>........] - ETA: 3s - loss: 0.2933 - accuracy: 0.9511
19000/25000 [=====================>........] - ETA: 3s - loss: 0.2933 - accuracy: 0.9511
19100/25000 [=====================>........] - ETA: 3s - loss: 0.2936 - accuracy: 0.9508
19200/25000 [======================>.......] - ETA: 3s - loss: 0.2934 - accuracy: 0.9510
19300/25000 [======================>.......] - ETA: 3s - loss: 0.2933 - accuracy: 0.9510
19400/25000 [======================>.......] - ETA: 3s - loss: 0.2931 - accuracy: 0.9511
19500/25000 [======================>.......] - ETA: 3s - loss: 0.2930 - accuracy: 0.9512
19600/25000 [======================>.......] - ETA: 3s - loss: 0.2932 - accuracy: 0.9511
19700/25000 [======================>.......] - ETA: 3s - loss: 0.2930 - accuracy: 0.9512
19800/25000 [======================>.......] - ETA: 3s - loss: 0.2930 - accuracy: 0.9512
19900/25000 [======================>.......] - ETA: 3s - loss: 0.2929 - accuracy: 0.9513
20000/25000 [=======================>......] - ETA: 3s - loss: 0.2929 - accuracy: 0.9513
20100/25000 [=======================>......] - ETA: 2s - loss: 0.2928 - accuracy: 0.9512
20200/25000 [=======================>......] - ETA: 2s - loss: 0.2927 - accuracy: 0.9513
20300/25000 [=======================>......] - ETA: 2s - loss: 0.2926 - accuracy: 0.9513
20400/25000 [=======================>......] - ETA: 2s - loss: 0.2924 - accuracy: 0.9515
20500/25000 [=======================>......] - ETA: 2s - loss: 0.2923 - accuracy: 0.9515
20600/25000 [=======================>......] - ETA: 2s - loss: 0.2925 - accuracy: 0.9514
20700/25000 [=======================>......] - ETA: 2s - loss: 0.2924 - accuracy: 0.9514
20800/25000 [=======================>......] - ETA: 2s - loss: 0.2923 - accuracy: 0.9514
20900/25000 [========================>.....] - ETA: 2s - loss: 0.2924 - accuracy: 0.9512
21000/25000 [========================>.....] - ETA: 2s - loss: 0.2926 - accuracy: 0.9511
21100/25000 [========================>.....] - ETA: 2s - loss: 0.2924 - accuracy: 0.9512
21200/25000 [========================>.....] - ETA: 2s - loss: 0.2925 - accuracy: 0.9512
21300/25000 [========================>.....] - ETA: 2s - loss: 0.2924 - accuracy: 0.9512
21400/25000 [========================>.....] - ETA: 2s - loss: 0.2925 - accuracy: 0.9510
21500/25000 [========================>.....] - ETA: 2s - loss: 0.2925 - accuracy: 0.9510
21600/25000 [========================>.....] - ETA: 2s - loss: 0.2923 - accuracy: 0.9511
21700/25000 [=========================>....] - ETA: 2s - loss: 0.2922 - accuracy: 0.9512
21800/25000 [=========================>....] - ETA: 1s - loss: 0.2921 - accuracy: 0.9512
21900/25000 [=========================>....] - ETA: 1s - loss: 0.2921 - accuracy: 0.9512
22000/25000 [=========================>....] - ETA: 1s - loss: 0.2922 - accuracy: 0.9511
22100/25000 [=========================>....] - ETA: 1s - loss: 0.2922 - accuracy: 0.9511
22200/25000 [=========================>....] - ETA: 1s - loss: 0.2921 - accuracy: 0.9511
22300/25000 [=========================>....] - ETA: 1s - loss: 0.2920 - accuracy: 0.9512
22400/25000 [=========================>....] - ETA: 1s - loss: 0.2920 - accuracy: 0.9512
22500/25000 [==========================>...] - ETA: 1s - loss: 0.2920 - accuracy: 0.9512
22600/25000 [==========================>...] - ETA: 1s - loss: 0.2920 - accuracy: 0.9511
22700/25000 [==========================>...] - ETA: 1s - loss: 0.2920 - accuracy: 0.9511
22800/25000 [==========================>...] - ETA: 1s - loss: 0.2919 - accuracy: 0.9511
22900/25000 [==========================>...] - ETA: 1s - loss: 0.2920 - accuracy: 0.9510
23000/25000 [==========================>...] - ETA: 1s - loss: 0.2920 - accuracy: 0.9510
23100/25000 [==========================>...] - ETA: 1s - loss: 0.2921 - accuracy: 0.9509
23200/25000 [==========================>...] - ETA: 1s - loss: 0.2920 - accuracy: 0.9509
23300/25000 [==========================>...] - ETA: 1s - loss: 0.2921 - accuracy: 0.9509
23400/25000 [===========================>..] - ETA: 0s - loss: 0.2919 - accuracy: 0.9510
23500/25000 [===========================>..] - ETA: 0s - loss: 0.2921 - accuracy: 0.9508
23600/25000 [===========================>..] - ETA: 0s - loss: 0.2921 - accuracy: 0.9507
23700/25000 [===========================>..] - ETA: 0s - loss: 0.2922 - accuracy: 0.9506
23800/25000 [===========================>..] - ETA: 0s - loss: 0.2923 - accuracy: 0.9505
23900/25000 [===========================>..] - ETA: 0s - loss: 0.2924 - accuracy: 0.9505
24000/25000 [===========================>..] - ETA: 0s - loss: 0.2925 - accuracy: 0.9503
24100/25000 [===========================>..] - ETA: 0s - loss: 0.2925 - accuracy: 0.9502
24200/25000 [============================>.] - ETA: 0s - loss: 0.2925 - accuracy: 0.9502
24300/25000 [============================>.] - ETA: 0s - loss: 0.2923 - accuracy: 0.9504
24400/25000 [============================>.] - ETA: 0s - loss: 0.2921 - accuracy: 0.9505
24500/25000 [============================>.] - ETA: 0s - loss: 0.2921 - accuracy: 0.9505
24600/25000 [============================>.] - ETA: 0s - loss: 0.2921 - accuracy: 0.9504
24700/25000 [============================>.] - ETA: 0s - loss: 0.2920 - accuracy: 0.9504
24800/25000 [============================>.] - ETA: 0s - loss: 0.2919 - accuracy: 0.9505
24900/25000 [============================>.] - ETA: 0s - loss: 0.2919 - accuracy: 0.9505
25000/25000 [==============================] - 19s 769us/step - loss: 0.2920 - accuracy: 0.9504 - val_loss: 0.4072 - val_accuracy: 0.8554
	=====> Test the model: model.predict()
	Dataset: IMDB_REVIEWS
	Algorithm: Deep Learning using Keras 2 (KerasDL2)
	Loss: 0.2761
	Training accuracy score: 95.71%
	Test Accuracy: 85.54%
	Training Time: 194.0329
	Test Time: 6.2445




FINAL CLASSIFICATION TABLE:

| ID | Dataset | Algorithm | Loss | Training accuracy score (%) | Test accuracy score (%) | Training time (seconds) | Test time (seconds) |
| -- | ------- | --------- | ---- | --------------------------- | ----------------------- | ----------------------- | ------------------- |
| 1 | TWENTY_NEWS_GROUPS | Deep Learning using Keras 1 (KerasDL1) | 0.1164 | 99.47 | 96.66 | 95.7168 | 1.6525 |
| 2 | IMDB_REVIEWS | Deep Learning using Keras 1 (KerasDL1) | 0.9344 | 99.98 | 83.16 | 165.9841 | 3.2173 |
| 3 | TWENTY_NEWS_GROUPS | Deep Learning using Keras 2 (KerasDL2) | 0.2558 | 94.96 | 94.96 | 82.9633 | 2.8831 |
| 4 | IMDB_REVIEWS | Deep Learning using Keras 2 (KerasDL2) | 0.4072 | 95.71 | 85.54 | 194.0329 | 6.2445 |

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
Program finished. It took 658.675763130188 seconds

Process finished with exit code 0
```