#### Deep Learning using Keras: Training and Validation Accuracy and Loss

| Dataset             | Training Accuracy (%) | Test Accuracy (%) | 
| ------------------- | --------------------- | ----------------- |
|  TWENTY_NEWS_GROUPS | 97.56%                | 95.98%            |
|  IMDB_REVIEWS       | 99.96%                | 84.17%            |

![TWENTY_NEWS_GROUPS: Training and Validation Accuracy, Training and Validation Loss](https://github.com/ramonfigueiredopessoa/comp551-2020-p2_classification_of_textual_data/blob/master/code/results/deep_learning_using_keras/model1/TWENTY_NEWS_GROUPS_training_and_validation_accuracy_and_Loss.png)

![IMDB_REVIEWS: Training and Validation Accuracy, Training and Validation Loss](https://github.com/ramonfigueiredopessoa/comp551-2020-p2_classification_of_textual_data/blob/master/code/results/deep_learning_using_keras/model1//IMDB_REVIEWS_training_and_validation_accuracy_and_Loss.png)


## Model 2: Deep Learning using Keras (TWENTY_NEWS_GROUP and IMDB_REVIEWS)

| Dataset             | Training Accuracy (%) | Loss   |Test Accuracy (%)  | 
| ------------------- | --------------------- | ------ | ----------------- |
|  TWENTY_NEWS_GROUPS | 94.96%                | 0.5998 | **94.96%**        |
|  IMDB_REVIEWS       | 94.94%                | 0.3113 | **86.16%**        |

![TWENTY_NEWS_GROUPS: Training and Validation Accuracy, Training and Validation Loss](https://github.com/ramonfigueiredopessoa/comp551-2020-p2_classification_of_textual_data/blob/master/code/results/deep_learning_using_keras/model2/10-epochs-using-NLTK_feature_extraction-TWENTY_NEWS_GROUPS_training_and_validation_accuracy_and_Loss)

![IMDB_REVIEWS: Training and Validation Accuracy, Training and Validation Loss](https://github.com/ramonfigueiredopessoa/comp551-2020-p2_classification_of_textual_data/blob/master/code/results/deep_learning_using_keras/model2/10-epochs-using-NLTK_feature_extraction-IMDB_REVIEWS_training_and_validation_accuracy_and_Loss)


#### Computer settings used to run

* Operating system: Ubuntu 18.04.4 LTS (64-bit)
* Processor: Intel® Core™ i7-7700 CPU @ 3.60GHz × 8 
* Memory: 32 GB

#### Logs Keras Deep Learning model 2 (with layers Embedding > Bidirectional LSTM > GlobalMaxPool1D > Dense RELU > Dropout (0.05) > Dense SIGMOID) using NLTK text, and 10 epochs 

* TWENTY_NEWS_GROUPS
	- Training accuracy score: 94.96%
	- Loss: 0.3695
	- Test Accuracy: 94.96%


* IMDB_REVIEWS
	* Training accuracy score: 96.36%
	* Loss: 0.2090
	* Test Accuracy: 85.66%

```
/home/ets-crchum/virtual_envs/comp551_p2/bin/python /home/ets-crchum/github/comp551-2020-p2_classification_of_textual_data/code/main.py -dl
Using TensorFlow backend.
2020-03-09 14:01:59.660977: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libnvinfer.so.6'; dlerror: libnvinfer.so.6: cannot open shared object file: No such file or directory
2020-03-09 14:01:59.661030: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libnvinfer_plugin.so.6'; dlerror: libnvinfer_plugin.so.6: cannot open shared object file: No such file or directory
2020-03-09 14:01:59.661036: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:30] Cannot dlopen some TensorRT libraries. If you would like to use Nvidia GPU with TensorRT, please make sure the missing libraries mentioned above are installed properly.
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
03/09/2020 02:02:00 PM - INFO - Program started...
03/09/2020 02:02:00 PM - INFO - Program started...
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
	It took 10.142449617385864 seconds


Applying NLTK feature extraction: X_test
	==> Replace unused UNICODE characters. 
	==> Text in lower case. 
	==> Apply Lemmatize using WordNet's built-in morphy function. Returns the input word unchanged if it cannot be found in WordNet 
	==> Remove English stop words
	Done: NLTK feature extraction (replace unused UNICODE characters, text in lower case, apply Lemmatize using WordNet, remove English stop words)  finished!
	It took 5.891373634338379 seconds

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
2020-03-09 14:02:19.241122: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
2020-03-09 14:02:19.247305: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-03-09 14:02:19.247845: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1555] Found device 0 with properties: 
pciBusID: 0000:01:00.0 name: GeForce GTX 1080 computeCapability: 6.1
coreClock: 1.7335GHz coreCount: 20 deviceMemorySize: 7.93GiB deviceMemoryBandwidth: 298.32GiB/s
2020-03-09 14:02:19.247910: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcudart.so.10.1'; dlerror: libcudart.so.10.1: cannot open shared object file: No such file or directory
2020-03-09 14:02:19.247953: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcublas.so.10'; dlerror: libcublas.so.10: cannot open shared object file: No such file or directory
2020-03-09 14:02:19.247993: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcufft.so.10'; dlerror: libcufft.so.10: cannot open shared object file: No such file or directory
2020-03-09 14:02:19.248033: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcurand.so.10'; dlerror: libcurand.so.10: cannot open shared object file: No such file or directory
2020-03-09 14:02:19.248073: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcusolver.so.10'; dlerror: libcusolver.so.10: cannot open shared object file: No such file or directory
2020-03-09 14:02:19.248112: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcusparse.so.10'; dlerror: libcusparse.so.10: cannot open shared object file: No such file or directory
2020-03-09 14:02:19.250066: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2020-03-09 14:02:19.250076: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1592] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
Skipping registering GPU devices...
2020-03-09 14:02:19.250279: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-03-09 14:02:19.271948: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 3600000000 Hz
2020-03-09 14:02:19.272589: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5741010 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-03-09 14:02:19.272608: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2020-03-09 14:02:19.338328: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-03-09 14:02:19.338983: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0xab0f860 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2020-03-09 14:02:19.338995: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): GeForce GTX 1080, Compute Capability 6.1
2020-03-09 14:02:19.339087: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1096] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-03-09 14:02:19.339092: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1102]      
	===> Tokenizer: fit_on_texts(X_test)
	===> X_test = pad_sequences(list_sentences_test, maxlen=6000)
	=====> Training the model: model.fit()
Train on 11314 samples, validate on 7532 samples
Epoch 1/10

  100/11314 [..............................] - ETA: 1:03 - loss: 0.7170 - accuracy: 0.3889
  200/11314 [..............................] - ETA: 34s - loss: 0.7173 - accuracy: 0.3832 
  300/11314 [..............................] - ETA: 25s - loss: 0.7169 - accuracy: 0.3832
  400/11314 [>.............................] - ETA: 20s - loss: 0.7167 - accuracy: 0.3816
  500/11314 [>.............................] - ETA: 17s - loss: 0.7166 - accuracy: 0.3801
  600/11314 [>.............................] - ETA: 15s - loss: 0.7161 - accuracy: 0.3808
  700/11314 [>.............................] - ETA: 14s - loss: 0.7156 - accuracy: 0.3812
  800/11314 [=>............................] - ETA: 12s - loss: 0.7152 - accuracy: 0.3807
  900/11314 [=>............................] - ETA: 12s - loss: 0.7147 - accuracy: 0.3806
 1000/11314 [=>............................] - ETA: 11s - loss: 0.7142 - accuracy: 0.3811
 1100/11314 [=>............................] - ETA: 10s - loss: 0.7138 - accuracy: 0.3804
 1200/11314 [==>...........................] - ETA: 10s - loss: 0.7134 - accuracy: 0.3800
 1300/11314 [==>...........................] - ETA: 9s - loss: 0.7129 - accuracy: 0.3802 
 1400/11314 [==>...........................] - ETA: 9s - loss: 0.7124 - accuracy: 0.3806
 1500/11314 [==>...........................] - ETA: 9s - loss: 0.7118 - accuracy: 0.3807
 1600/11314 [===>..........................] - ETA: 8s - loss: 0.7113 - accuracy: 0.3812
 1700/11314 [===>..........................] - ETA: 8s - loss: 0.7107 - accuracy: 0.3815
 1800/11314 [===>..........................] - ETA: 8s - loss: 0.7102 - accuracy: 0.3815
 1900/11314 [====>.........................] - ETA: 8s - loss: 0.7096 - accuracy: 0.3818
 2000/11314 [====>.........................] - ETA: 8s - loss: 0.7090 - accuracy: 0.3819
 2100/11314 [====>.........................] - ETA: 7s - loss: 0.7084 - accuracy: 0.3817
 2200/11314 [====>.........................] - ETA: 7s - loss: 0.7079 - accuracy: 0.3813
 2300/11314 [=====>........................] - ETA: 7s - loss: 0.7073 - accuracy: 0.3827
 2400/11314 [=====>........................] - ETA: 7s - loss: 0.7066 - accuracy: 0.3846
 2500/11314 [=====>........................] - ETA: 7s - loss: 0.7060 - accuracy: 0.3862
 2600/11314 [=====>........................] - ETA: 7s - loss: 0.7054 - accuracy: 0.3901
 2700/11314 [======>.......................] - ETA: 6s - loss: 0.7048 - accuracy: 0.3943
 2800/11314 [======>.......................] - ETA: 6s - loss: 0.7041 - accuracy: 0.3985
 2900/11314 [======>.......................] - ETA: 6s - loss: 0.7035 - accuracy: 0.4032
 3000/11314 [======>.......................] - ETA: 6s - loss: 0.7029 - accuracy: 0.4087
 3100/11314 [=======>......................] - ETA: 6s - loss: 0.7023 - accuracy: 0.4153
 3200/11314 [=======>......................] - ETA: 6s - loss: 0.7017 - accuracy: 0.4254
 3300/11314 [=======>......................] - ETA: 6s - loss: 0.7011 - accuracy: 0.4366
 3400/11314 [========>.....................] - ETA: 6s - loss: 0.7005 - accuracy: 0.4490
 3500/11314 [========>.....................] - ETA: 5s - loss: 0.6999 - accuracy: 0.4613
 3600/11314 [========>.....................] - ETA: 5s - loss: 0.6994 - accuracy: 0.4739
 3700/11314 [========>.....................] - ETA: 5s - loss: 0.6988 - accuracy: 0.4858
 3800/11314 [=========>....................] - ETA: 5s - loss: 0.6983 - accuracy: 0.4976
 3900/11314 [=========>....................] - ETA: 5s - loss: 0.6977 - accuracy: 0.5086
 4000/11314 [=========>....................] - ETA: 5s - loss: 0.6972 - accuracy: 0.5193
 4100/11314 [=========>....................] - ETA: 5s - loss: 0.6967 - accuracy: 0.5298
 4200/11314 [==========>...................] - ETA: 5s - loss: 0.6962 - accuracy: 0.5397
 4300/11314 [==========>...................] - ETA: 5s - loss: 0.6957 - accuracy: 0.5493
 4400/11314 [==========>...................] - ETA: 5s - loss: 0.6953 - accuracy: 0.5583
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.6948 - accuracy: 0.5671
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.6943 - accuracy: 0.5754
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.6939 - accuracy: 0.5834
 4800/11314 [===========>..................] - ETA: 4s - loss: 0.6935 - accuracy: 0.5910
 4900/11314 [===========>..................] - ETA: 4s - loss: 0.6930 - accuracy: 0.5983
 5000/11314 [============>.................] - ETA: 4s - loss: 0.6926 - accuracy: 0.6053
 5100/11314 [============>.................] - ETA: 4s - loss: 0.6922 - accuracy: 0.6120
 5200/11314 [============>.................] - ETA: 4s - loss: 0.6918 - accuracy: 0.6186
 5300/11314 [=============>................] - ETA: 4s - loss: 0.6914 - accuracy: 0.6248
 5400/11314 [=============>................] - ETA: 4s - loss: 0.6910 - accuracy: 0.6308
 5500/11314 [=============>................] - ETA: 4s - loss: 0.6906 - accuracy: 0.6366
 5600/11314 [=============>................] - ETA: 4s - loss: 0.6902 - accuracy: 0.6422
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.6899 - accuracy: 0.6476
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.6895 - accuracy: 0.6528
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.6891 - accuracy: 0.6578
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.6888 - accuracy: 0.6626
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.6884 - accuracy: 0.6674
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.6881 - accuracy: 0.6719
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.6877 - accuracy: 0.6763
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.6874 - accuracy: 0.6806
 6500/11314 [================>.............] - ETA: 3s - loss: 0.6870 - accuracy: 0.6847
 6600/11314 [================>.............] - ETA: 3s - loss: 0.6867 - accuracy: 0.6888
 6700/11314 [================>.............] - ETA: 3s - loss: 0.6864 - accuracy: 0.6927
 6800/11314 [=================>............] - ETA: 3s - loss: 0.6860 - accuracy: 0.6964
 6900/11314 [=================>............] - ETA: 3s - loss: 0.6857 - accuracy: 0.7001
 7000/11314 [=================>............] - ETA: 2s - loss: 0.6854 - accuracy: 0.7036
 7100/11314 [=================>............] - ETA: 2s - loss: 0.6851 - accuracy: 0.7071
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.6848 - accuracy: 0.7105
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.6844 - accuracy: 0.7137
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.6841 - accuracy: 0.7169
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.6838 - accuracy: 0.7201
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.6835 - accuracy: 0.7231
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.6832 - accuracy: 0.7260
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.6829 - accuracy: 0.7289
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.6826 - accuracy: 0.7316
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.6823 - accuracy: 0.7344
 8100/11314 [====================>.........] - ETA: 2s - loss: 0.6820 - accuracy: 0.7370
 8200/11314 [====================>.........] - ETA: 2s - loss: 0.6817 - accuracy: 0.7396
 8300/11314 [=====================>........] - ETA: 2s - loss: 0.6815 - accuracy: 0.7421
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.6812 - accuracy: 0.7446
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.6809 - accuracy: 0.7470
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.6806 - accuracy: 0.7494
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.6803 - accuracy: 0.7517
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.6800 - accuracy: 0.7539
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.6797 - accuracy: 0.7561
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.6795 - accuracy: 0.7583
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.6792 - accuracy: 0.7604
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.6789 - accuracy: 0.7624
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.6786 - accuracy: 0.7644
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.6784 - accuracy: 0.7664
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.6781 - accuracy: 0.7683
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.6778 - accuracy: 0.7702
 9700/11314 [========================>.....] - ETA: 1s - loss: 0.6776 - accuracy: 0.7721
 9800/11314 [========================>.....] - ETA: 1s - loss: 0.6773 - accuracy: 0.7739
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.6770 - accuracy: 0.7757
10000/11314 [=========================>....] - ETA: 0s - loss: 0.6768 - accuracy: 0.7774
10100/11314 [=========================>....] - ETA: 0s - loss: 0.6765 - accuracy: 0.7791
10200/11314 [==========================>...] - ETA: 0s - loss: 0.6762 - accuracy: 0.7808
10300/11314 [==========================>...] - ETA: 0s - loss: 0.6760 - accuracy: 0.7824
10400/11314 [==========================>...] - ETA: 0s - loss: 0.6757 - accuracy: 0.7840
10500/11314 [==========================>...] - ETA: 0s - loss: 0.6755 - accuracy: 0.7856
10600/11314 [===========================>..] - ETA: 0s - loss: 0.6752 - accuracy: 0.7871
10700/11314 [===========================>..] - ETA: 0s - loss: 0.6749 - accuracy: 0.7886
10800/11314 [===========================>..] - ETA: 0s - loss: 0.6747 - accuracy: 0.7901
10900/11314 [===========================>..] - ETA: 0s - loss: 0.6744 - accuracy: 0.7916
11000/11314 [============================>.] - ETA: 0s - loss: 0.6742 - accuracy: 0.7930
11100/11314 [============================>.] - ETA: 0s - loss: 0.6739 - accuracy: 0.7944
11200/11314 [============================>.] - ETA: 0s - loss: 0.6737 - accuracy: 0.7958
11300/11314 [============================>.] - ETA: 0s - loss: 0.6734 - accuracy: 0.7972
11314/11314 [==============================] - 9s 775us/step - loss: 0.6734 - accuracy: 0.7974 - val_loss: 0.6444 - val_accuracy: 0.9496
Epoch 2/10

  100/11314 [..............................] - ETA: 6s - loss: 0.6444 - accuracy: 0.9495
  200/11314 [..............................] - ETA: 6s - loss: 0.6443 - accuracy: 0.9487
  300/11314 [..............................] - ETA: 6s - loss: 0.6440 - accuracy: 0.9493
  400/11314 [>.............................] - ETA: 6s - loss: 0.6437 - accuracy: 0.9499
  500/11314 [>.............................] - ETA: 6s - loss: 0.6435 - accuracy: 0.9499
  600/11314 [>.............................] - ETA: 6s - loss: 0.6433 - accuracy: 0.9497
  700/11314 [>.............................] - ETA: 6s - loss: 0.6431 - accuracy: 0.9496
  800/11314 [=>............................] - ETA: 6s - loss: 0.6429 - accuracy: 0.9495
  900/11314 [=>............................] - ETA: 6s - loss: 0.6427 - accuracy: 0.9495
 1000/11314 [=>............................] - ETA: 6s - loss: 0.6425 - accuracy: 0.9496
 1100/11314 [=>............................] - ETA: 6s - loss: 0.6423 - accuracy: 0.9496
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.6421 - accuracy: 0.9495
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.6419 - accuracy: 0.9495
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.6417 - accuracy: 0.9494
 1500/11314 [==>...........................] - ETA: 5s - loss: 0.6415 - accuracy: 0.9494
 1600/11314 [===>..........................] - ETA: 5s - loss: 0.6413 - accuracy: 0.9493
 1700/11314 [===>..........................] - ETA: 5s - loss: 0.6411 - accuracy: 0.9493
 1800/11314 [===>..........................] - ETA: 5s - loss: 0.6409 - accuracy: 0.9492
 1900/11314 [====>.........................] - ETA: 5s - loss: 0.6407 - accuracy: 0.9493
 2000/11314 [====>.........................] - ETA: 5s - loss: 0.6405 - accuracy: 0.9493
 2100/11314 [====>.........................] - ETA: 5s - loss: 0.6403 - accuracy: 0.9493
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.6401 - accuracy: 0.9494
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.6399 - accuracy: 0.9494
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.6397 - accuracy: 0.9494
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.6395 - accuracy: 0.9494
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.6393 - accuracy: 0.9494
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.6391 - accuracy: 0.9493
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.6389 - accuracy: 0.9494
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.6387 - accuracy: 0.9494
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.6385 - accuracy: 0.9495
 3100/11314 [=======>......................] - ETA: 5s - loss: 0.6383 - accuracy: 0.9494
 3200/11314 [=======>......................] - ETA: 4s - loss: 0.6381 - accuracy: 0.9494
 3300/11314 [=======>......................] - ETA: 4s - loss: 0.6379 - accuracy: 0.9494
 3400/11314 [========>.....................] - ETA: 4s - loss: 0.6377 - accuracy: 0.9495
 3500/11314 [========>.....................] - ETA: 4s - loss: 0.6375 - accuracy: 0.9495
 3600/11314 [========>.....................] - ETA: 4s - loss: 0.6373 - accuracy: 0.9495
 3700/11314 [========>.....................] - ETA: 4s - loss: 0.6371 - accuracy: 0.9495
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.6369 - accuracy: 0.9495
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.6367 - accuracy: 0.9495
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.6365 - accuracy: 0.9496
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.6363 - accuracy: 0.9496
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.6361 - accuracy: 0.9496
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.6359 - accuracy: 0.9496
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.6356 - accuracy: 0.9497
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.6354 - accuracy: 0.9497
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.6353 - accuracy: 0.9496
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.6350 - accuracy: 0.9497
 4800/11314 [===========>..................] - ETA: 3s - loss: 0.6348 - accuracy: 0.9497
 4900/11314 [===========>..................] - ETA: 3s - loss: 0.6346 - accuracy: 0.9497
 5000/11314 [============>.................] - ETA: 3s - loss: 0.6344 - accuracy: 0.9497
 5100/11314 [============>.................] - ETA: 3s - loss: 0.6343 - accuracy: 0.9497
 5200/11314 [============>.................] - ETA: 3s - loss: 0.6341 - accuracy: 0.9496
 5300/11314 [=============>................] - ETA: 3s - loss: 0.6339 - accuracy: 0.9496
 5400/11314 [=============>................] - ETA: 3s - loss: 0.6337 - accuracy: 0.9496
 5500/11314 [=============>................] - ETA: 3s - loss: 0.6335 - accuracy: 0.9496
 5600/11314 [=============>................] - ETA: 3s - loss: 0.6333 - accuracy: 0.9496
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.6331 - accuracy: 0.9496
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.6329 - accuracy: 0.9496
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.6327 - accuracy: 0.9496
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.6325 - accuracy: 0.9495
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.6323 - accuracy: 0.9496
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.6321 - accuracy: 0.9496
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.6319 - accuracy: 0.9496
 6400/11314 [===============>..............] - ETA: 2s - loss: 0.6317 - accuracy: 0.9496
 6500/11314 [================>.............] - ETA: 2s - loss: 0.6315 - accuracy: 0.9496
 6600/11314 [================>.............] - ETA: 2s - loss: 0.6313 - accuracy: 0.9496
 6700/11314 [================>.............] - ETA: 2s - loss: 0.6311 - accuracy: 0.9496
 6800/11314 [=================>............] - ETA: 2s - loss: 0.6309 - accuracy: 0.9496
 6900/11314 [=================>............] - ETA: 2s - loss: 0.6307 - accuracy: 0.9496
 7000/11314 [=================>............] - ETA: 2s - loss: 0.6305 - accuracy: 0.9496
 7100/11314 [=================>............] - ETA: 2s - loss: 0.6303 - accuracy: 0.9496
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.6301 - accuracy: 0.9496
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.6299 - accuracy: 0.9496
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.6297 - accuracy: 0.9496
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.6295 - accuracy: 0.9496
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.6293 - accuracy: 0.9496
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.6291 - accuracy: 0.9496
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.6289 - accuracy: 0.9496
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.6287 - accuracy: 0.9496
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.6285 - accuracy: 0.9496
 8100/11314 [====================>.........] - ETA: 1s - loss: 0.6283 - accuracy: 0.9496
 8200/11314 [====================>.........] - ETA: 1s - loss: 0.6281 - accuracy: 0.9496
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.6280 - accuracy: 0.9496
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.6278 - accuracy: 0.9496
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.6276 - accuracy: 0.9496
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.6274 - accuracy: 0.9496
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.6272 - accuracy: 0.9496
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.6270 - accuracy: 0.9496
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.6268 - accuracy: 0.9496
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.6266 - accuracy: 0.9496
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.6264 - accuracy: 0.9496
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.6262 - accuracy: 0.9496
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.6260 - accuracy: 0.9496
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.6258 - accuracy: 0.9496
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.6256 - accuracy: 0.9496
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.6254 - accuracy: 0.9496
 9700/11314 [========================>.....] - ETA: 0s - loss: 0.6252 - accuracy: 0.9496
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.6250 - accuracy: 0.9496
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.6248 - accuracy: 0.9496
10000/11314 [=========================>....] - ETA: 0s - loss: 0.6246 - accuracy: 0.9496
10100/11314 [=========================>....] - ETA: 0s - loss: 0.6245 - accuracy: 0.9496
10200/11314 [==========================>...] - ETA: 0s - loss: 0.6243 - accuracy: 0.9496
10300/11314 [==========================>...] - ETA: 0s - loss: 0.6241 - accuracy: 0.9496
10400/11314 [==========================>...] - ETA: 0s - loss: 0.6239 - accuracy: 0.9496
10500/11314 [==========================>...] - ETA: 0s - loss: 0.6237 - accuracy: 0.9496
10600/11314 [===========================>..] - ETA: 0s - loss: 0.6235 - accuracy: 0.9496
10700/11314 [===========================>..] - ETA: 0s - loss: 0.6233 - accuracy: 0.9496
10800/11314 [===========================>..] - ETA: 0s - loss: 0.6231 - accuracy: 0.9496
10900/11314 [===========================>..] - ETA: 0s - loss: 0.6229 - accuracy: 0.9496
11000/11314 [============================>.] - ETA: 0s - loss: 0.6227 - accuracy: 0.9496
11100/11314 [============================>.] - ETA: 0s - loss: 0.6225 - accuracy: 0.9496
11200/11314 [============================>.] - ETA: 0s - loss: 0.6223 - accuracy: 0.9496
11300/11314 [============================>.] - ETA: 0s - loss: 0.6221 - accuracy: 0.9496
11314/11314 [==============================] - 8s 718us/step - loss: 0.6221 - accuracy: 0.9496 - val_loss: 0.5999 - val_accuracy: 0.9496
Epoch 3/10

  100/11314 [..............................] - ETA: 6s - loss: 0.6001 - accuracy: 0.9484
  200/11314 [..............................] - ETA: 6s - loss: 0.5997 - accuracy: 0.9497
  300/11314 [..............................] - ETA: 6s - loss: 0.5993 - accuracy: 0.9505
  400/11314 [>.............................] - ETA: 6s - loss: 0.5991 - accuracy: 0.9508
  500/11314 [>.............................] - ETA: 6s - loss: 0.5990 - accuracy: 0.9504
  600/11314 [>.............................] - ETA: 6s - loss: 0.5988 - accuracy: 0.9504
  700/11314 [>.............................] - ETA: 6s - loss: 0.5986 - accuracy: 0.9502
  800/11314 [=>............................] - ETA: 6s - loss: 0.5985 - accuracy: 0.9499
  900/11314 [=>............................] - ETA: 6s - loss: 0.5984 - accuracy: 0.9497
 1000/11314 [=>............................] - ETA: 6s - loss: 0.5982 - accuracy: 0.9497
 1100/11314 [=>............................] - ETA: 6s - loss: 0.5980 - accuracy: 0.9496
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.5979 - accuracy: 0.9495
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.5977 - accuracy: 0.9496
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.5975 - accuracy: 0.9496
 1500/11314 [==>...........................] - ETA: 5s - loss: 0.5973 - accuracy: 0.9496
 1600/11314 [===>..........................] - ETA: 5s - loss: 0.5971 - accuracy: 0.9497
 1700/11314 [===>..........................] - ETA: 5s - loss: 0.5969 - accuracy: 0.9496
 1800/11314 [===>..........................] - ETA: 5s - loss: 0.5967 - accuracy: 0.9496
 1900/11314 [====>.........................] - ETA: 5s - loss: 0.5965 - accuracy: 0.9496
 2000/11314 [====>.........................] - ETA: 5s - loss: 0.5964 - accuracy: 0.9496
 2100/11314 [====>.........................] - ETA: 5s - loss: 0.5962 - accuracy: 0.9495
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.5960 - accuracy: 0.9495
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.5959 - accuracy: 0.9495
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.5957 - accuracy: 0.9494
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.5955 - accuracy: 0.9494
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.5953 - accuracy: 0.9494
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.5951 - accuracy: 0.9495
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.5949 - accuracy: 0.9495
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.5947 - accuracy: 0.9495
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.5946 - accuracy: 0.9495
 3100/11314 [=======>......................] - ETA: 4s - loss: 0.5944 - accuracy: 0.9495
 3200/11314 [=======>......................] - ETA: 4s - loss: 0.5942 - accuracy: 0.9495
 3300/11314 [=======>......................] - ETA: 4s - loss: 0.5940 - accuracy: 0.9495
 3400/11314 [========>.....................] - ETA: 4s - loss: 0.5938 - accuracy: 0.9495
 3500/11314 [========>.....................] - ETA: 4s - loss: 0.5937 - accuracy: 0.9495
 3600/11314 [========>.....................] - ETA: 4s - loss: 0.5935 - accuracy: 0.9495
 3700/11314 [========>.....................] - ETA: 4s - loss: 0.5933 - accuracy: 0.9495
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.5931 - accuracy: 0.9495
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.5929 - accuracy: 0.9494
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.5928 - accuracy: 0.9495
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.5926 - accuracy: 0.9495
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.5924 - accuracy: 0.9495
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.5922 - accuracy: 0.9495
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.5920 - accuracy: 0.9495
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.5919 - accuracy: 0.9495
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.5917 - accuracy: 0.9495
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.5915 - accuracy: 0.9495
 4800/11314 [===========>..................] - ETA: 3s - loss: 0.5913 - accuracy: 0.9495
 4900/11314 [===========>..................] - ETA: 3s - loss: 0.5911 - accuracy: 0.9495
 5000/11314 [============>.................] - ETA: 3s - loss: 0.5909 - accuracy: 0.9495
 5100/11314 [============>.................] - ETA: 3s - loss: 0.5908 - accuracy: 0.9495
 5200/11314 [============>.................] - ETA: 3s - loss: 0.5906 - accuracy: 0.9496
 5300/11314 [=============>................] - ETA: 3s - loss: 0.5904 - accuracy: 0.9496
 5400/11314 [=============>................] - ETA: 3s - loss: 0.5902 - accuracy: 0.9496
 5500/11314 [=============>................] - ETA: 3s - loss: 0.5900 - accuracy: 0.9496
 5600/11314 [=============>................] - ETA: 3s - loss: 0.5898 - accuracy: 0.9496
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.5897 - accuracy: 0.9496
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.5895 - accuracy: 0.9496
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.5893 - accuracy: 0.9496
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.5891 - accuracy: 0.9496
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.5890 - accuracy: 0.9496
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.5888 - accuracy: 0.9496
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.5886 - accuracy: 0.9496
 6400/11314 [===============>..............] - ETA: 2s - loss: 0.5884 - accuracy: 0.9496
 6500/11314 [================>.............] - ETA: 2s - loss: 0.5882 - accuracy: 0.9496
 6600/11314 [================>.............] - ETA: 2s - loss: 0.5880 - accuracy: 0.9496
 6700/11314 [================>.............] - ETA: 2s - loss: 0.5879 - accuracy: 0.9496
 6800/11314 [=================>............] - ETA: 2s - loss: 0.5877 - accuracy: 0.9497
 6900/11314 [=================>............] - ETA: 2s - loss: 0.5875 - accuracy: 0.9496
 7000/11314 [=================>............] - ETA: 2s - loss: 0.5873 - accuracy: 0.9496
 7100/11314 [=================>............] - ETA: 2s - loss: 0.5871 - accuracy: 0.9496
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.5870 - accuracy: 0.9496
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.5868 - accuracy: 0.9496
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.5866 - accuracy: 0.9496
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.5864 - accuracy: 0.9496
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.5863 - accuracy: 0.9496
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.5861 - accuracy: 0.9496
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.5859 - accuracy: 0.9496
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.5857 - accuracy: 0.9496
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.5856 - accuracy: 0.9496
 8100/11314 [====================>.........] - ETA: 1s - loss: 0.5854 - accuracy: 0.9496
 8200/11314 [====================>.........] - ETA: 1s - loss: 0.5852 - accuracy: 0.9497
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.5850 - accuracy: 0.9497
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.5848 - accuracy: 0.9497
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.5847 - accuracy: 0.9497
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.5845 - accuracy: 0.9497
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.5843 - accuracy: 0.9496
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.5841 - accuracy: 0.9496
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.5840 - accuracy: 0.9496
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.5838 - accuracy: 0.9496
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.5836 - accuracy: 0.9496
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.5834 - accuracy: 0.9496
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.5833 - accuracy: 0.9496
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.5831 - accuracy: 0.9496
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.5829 - accuracy: 0.9496
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.5827 - accuracy: 0.9496
 9700/11314 [========================>.....] - ETA: 0s - loss: 0.5826 - accuracy: 0.9496
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.5824 - accuracy: 0.9496
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.5822 - accuracy: 0.9496
10000/11314 [=========================>....] - ETA: 0s - loss: 0.5820 - accuracy: 0.9496
10100/11314 [=========================>....] - ETA: 0s - loss: 0.5819 - accuracy: 0.9496
10200/11314 [==========================>...] - ETA: 0s - loss: 0.5817 - accuracy: 0.9497
10300/11314 [==========================>...] - ETA: 0s - loss: 0.5815 - accuracy: 0.9496
10400/11314 [==========================>...] - ETA: 0s - loss: 0.5813 - accuracy: 0.9496
10500/11314 [==========================>...] - ETA: 0s - loss: 0.5812 - accuracy: 0.9496
10600/11314 [===========================>..] - ETA: 0s - loss: 0.5810 - accuracy: 0.9496
10700/11314 [===========================>..] - ETA: 0s - loss: 0.5808 - accuracy: 0.9496
10800/11314 [===========================>..] - ETA: 0s - loss: 0.5806 - accuracy: 0.9496
10900/11314 [===========================>..] - ETA: 0s - loss: 0.5805 - accuracy: 0.9496
11000/11314 [============================>.] - ETA: 0s - loss: 0.5803 - accuracy: 0.9496
11100/11314 [============================>.] - ETA: 0s - loss: 0.5801 - accuracy: 0.9496
11200/11314 [============================>.] - ETA: 0s - loss: 0.5800 - accuracy: 0.9496
11300/11314 [============================>.] - ETA: 0s - loss: 0.5798 - accuracy: 0.9496
11314/11314 [==============================] - 8s 727us/step - loss: 0.5798 - accuracy: 0.9496 - val_loss: 0.5596 - val_accuracy: 0.9496
Epoch 4/10

  100/11314 [..............................] - ETA: 6s - loss: 0.5600 - accuracy: 0.9484
  200/11314 [..............................] - ETA: 6s - loss: 0.5597 - accuracy: 0.9489
  300/11314 [..............................] - ETA: 6s - loss: 0.5595 - accuracy: 0.9491
  400/11314 [>.............................] - ETA: 6s - loss: 0.5592 - accuracy: 0.9493
  500/11314 [>.............................] - ETA: 6s - loss: 0.5590 - accuracy: 0.9494
  600/11314 [>.............................] - ETA: 6s - loss: 0.5588 - accuracy: 0.9496
  700/11314 [>.............................] - ETA: 6s - loss: 0.5587 - accuracy: 0.9495
  800/11314 [=>............................] - ETA: 6s - loss: 0.5585 - accuracy: 0.9494
  900/11314 [=>............................] - ETA: 6s - loss: 0.5584 - accuracy: 0.9494
 1000/11314 [=>............................] - ETA: 6s - loss: 0.5582 - accuracy: 0.9494
 1100/11314 [=>............................] - ETA: 6s - loss: 0.5580 - accuracy: 0.9494
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.5579 - accuracy: 0.9494
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.5577 - accuracy: 0.9494
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.5576 - accuracy: 0.9493
 1500/11314 [==>...........................] - ETA: 6s - loss: 0.5574 - accuracy: 0.9494
 1600/11314 [===>..........................] - ETA: 5s - loss: 0.5572 - accuracy: 0.9494
 1700/11314 [===>..........................] - ETA: 5s - loss: 0.5570 - accuracy: 0.9494
 1800/11314 [===>..........................] - ETA: 5s - loss: 0.5569 - accuracy: 0.9493
 1900/11314 [====>.........................] - ETA: 5s - loss: 0.5567 - accuracy: 0.9494
 2000/11314 [====>.........................] - ETA: 5s - loss: 0.5566 - accuracy: 0.9493
 2100/11314 [====>.........................] - ETA: 5s - loss: 0.5564 - accuracy: 0.9494
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.5562 - accuracy: 0.9494
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.5560 - accuracy: 0.9494
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.5559 - accuracy: 0.9494
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.5557 - accuracy: 0.9494
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.5556 - accuracy: 0.9493
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.5554 - accuracy: 0.9494
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.5552 - accuracy: 0.9493
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.5551 - accuracy: 0.9493
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.5549 - accuracy: 0.9493
 3100/11314 [=======>......................] - ETA: 5s - loss: 0.5548 - accuracy: 0.9493
 3200/11314 [=======>......................] - ETA: 4s - loss: 0.5546 - accuracy: 0.9493
 3300/11314 [=======>......................] - ETA: 4s - loss: 0.5544 - accuracy: 0.9493
 3400/11314 [========>.....................] - ETA: 4s - loss: 0.5542 - accuracy: 0.9493
 3500/11314 [========>.....................] - ETA: 4s - loss: 0.5541 - accuracy: 0.9494
 3600/11314 [========>.....................] - ETA: 4s - loss: 0.5539 - accuracy: 0.9494
 3700/11314 [========>.....................] - ETA: 4s - loss: 0.5537 - accuracy: 0.9494
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.5536 - accuracy: 0.9494
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.5534 - accuracy: 0.9494
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.5533 - accuracy: 0.9494
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.5531 - accuracy: 0.9494
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.5529 - accuracy: 0.9494
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.5528 - accuracy: 0.9494
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.5526 - accuracy: 0.9494
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.5525 - accuracy: 0.9493
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.5523 - accuracy: 0.9493
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.5521 - accuracy: 0.9493
 4800/11314 [===========>..................] - ETA: 3s - loss: 0.5519 - accuracy: 0.9494
 4900/11314 [===========>..................] - ETA: 3s - loss: 0.5518 - accuracy: 0.9494
 5000/11314 [============>.................] - ETA: 3s - loss: 0.5516 - accuracy: 0.9494
 5100/11314 [============>.................] - ETA: 3s - loss: 0.5515 - accuracy: 0.9494
 5200/11314 [============>.................] - ETA: 3s - loss: 0.5513 - accuracy: 0.9495
 5300/11314 [=============>................] - ETA: 3s - loss: 0.5511 - accuracy: 0.9495
 5400/11314 [=============>................] - ETA: 3s - loss: 0.5509 - accuracy: 0.9495
 5500/11314 [=============>................] - ETA: 3s - loss: 0.5508 - accuracy: 0.9495
 5600/11314 [=============>................] - ETA: 3s - loss: 0.5506 - accuracy: 0.9495
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.5504 - accuracy: 0.9495
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.5503 - accuracy: 0.9495
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.5501 - accuracy: 0.9496
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.5499 - accuracy: 0.9496
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.5498 - accuracy: 0.9496
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.5496 - accuracy: 0.9496
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.5495 - accuracy: 0.9495
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.5493 - accuracy: 0.9495
 6500/11314 [================>.............] - ETA: 2s - loss: 0.5491 - accuracy: 0.9495
 6600/11314 [================>.............] - ETA: 2s - loss: 0.5490 - accuracy: 0.9496
 6700/11314 [================>.............] - ETA: 2s - loss: 0.5488 - accuracy: 0.9496
 6800/11314 [=================>............] - ETA: 2s - loss: 0.5486 - accuracy: 0.9496
 6900/11314 [=================>............] - ETA: 2s - loss: 0.5485 - accuracy: 0.9496
 7000/11314 [=================>............] - ETA: 2s - loss: 0.5483 - accuracy: 0.9496
 7100/11314 [=================>............] - ETA: 2s - loss: 0.5482 - accuracy: 0.9496
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.5480 - accuracy: 0.9496
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.5478 - accuracy: 0.9496
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.5477 - accuracy: 0.9496
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.5475 - accuracy: 0.9496
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.5474 - accuracy: 0.9496
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.5472 - accuracy: 0.9496
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.5470 - accuracy: 0.9496
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.5469 - accuracy: 0.9496
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.5467 - accuracy: 0.9496
 8100/11314 [====================>.........] - ETA: 1s - loss: 0.5466 - accuracy: 0.9496
 8200/11314 [====================>.........] - ETA: 1s - loss: 0.5464 - accuracy: 0.9496
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.5462 - accuracy: 0.9496
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.5461 - accuracy: 0.9496
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.5459 - accuracy: 0.9495
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.5458 - accuracy: 0.9495
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.5456 - accuracy: 0.9496
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.5454 - accuracy: 0.9496
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.5453 - accuracy: 0.9496
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.5451 - accuracy: 0.9496
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.5450 - accuracy: 0.9496
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.5448 - accuracy: 0.9496
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.5446 - accuracy: 0.9496
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.5445 - accuracy: 0.9496
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.5443 - accuracy: 0.9496
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.5442 - accuracy: 0.9496
 9700/11314 [========================>.....] - ETA: 0s - loss: 0.5440 - accuracy: 0.9496
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.5438 - accuracy: 0.9496
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.5437 - accuracy: 0.9496
10000/11314 [=========================>....] - ETA: 0s - loss: 0.5435 - accuracy: 0.9496
10100/11314 [=========================>....] - ETA: 0s - loss: 0.5434 - accuracy: 0.9496
10200/11314 [==========================>...] - ETA: 0s - loss: 0.5432 - accuracy: 0.9496
10300/11314 [==========================>...] - ETA: 0s - loss: 0.5431 - accuracy: 0.9496
10400/11314 [==========================>...] - ETA: 0s - loss: 0.5429 - accuracy: 0.9496
10500/11314 [==========================>...] - ETA: 0s - loss: 0.5427 - accuracy: 0.9496
10600/11314 [===========================>..] - ETA: 0s - loss: 0.5426 - accuracy: 0.9496
10700/11314 [===========================>..] - ETA: 0s - loss: 0.5424 - accuracy: 0.9496
10800/11314 [===========================>..] - ETA: 0s - loss: 0.5423 - accuracy: 0.9496
10900/11314 [===========================>..] - ETA: 0s - loss: 0.5421 - accuracy: 0.9496
11000/11314 [============================>.] - ETA: 0s - loss: 0.5419 - accuracy: 0.9496
11100/11314 [============================>.] - ETA: 0s - loss: 0.5418 - accuracy: 0.9496
11200/11314 [============================>.] - ETA: 0s - loss: 0.5416 - accuracy: 0.9496
11300/11314 [============================>.] - ETA: 0s - loss: 0.5415 - accuracy: 0.9496
11314/11314 [==============================] - 8s 724us/step - loss: 0.5415 - accuracy: 0.9496 - val_loss: 0.5233 - val_accuracy: 0.9496
Epoch 5/10

  100/11314 [..............................] - ETA: 6s - loss: 0.5230 - accuracy: 0.9505
  200/11314 [..............................] - ETA: 6s - loss: 0.5231 - accuracy: 0.9497
  300/11314 [..............................] - ETA: 6s - loss: 0.5231 - accuracy: 0.9495
  400/11314 [>.............................] - ETA: 6s - loss: 0.5230 - accuracy: 0.9493
  500/11314 [>.............................] - ETA: 6s - loss: 0.5227 - accuracy: 0.9496
  600/11314 [>.............................] - ETA: 6s - loss: 0.5225 - accuracy: 0.9496
  700/11314 [>.............................] - ETA: 6s - loss: 0.5224 - accuracy: 0.9495
  800/11314 [=>............................] - ETA: 6s - loss: 0.5221 - accuracy: 0.9498
  900/11314 [=>............................] - ETA: 6s - loss: 0.5220 - accuracy: 0.9497
 1000/11314 [=>............................] - ETA: 6s - loss: 0.5218 - accuracy: 0.9499
 1100/11314 [=>............................] - ETA: 6s - loss: 0.5218 - accuracy: 0.9497
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.5215 - accuracy: 0.9498
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.5214 - accuracy: 0.9497
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.5213 - accuracy: 0.9497
 1500/11314 [==>...........................] - ETA: 6s - loss: 0.5212 - accuracy: 0.9496
 1600/11314 [===>..........................] - ETA: 5s - loss: 0.5210 - accuracy: 0.9496
 1700/11314 [===>..........................] - ETA: 5s - loss: 0.5209 - accuracy: 0.9496
 1800/11314 [===>..........................] - ETA: 5s - loss: 0.5207 - accuracy: 0.9496
 1900/11314 [====>.........................] - ETA: 5s - loss: 0.5206 - accuracy: 0.9497
 2000/11314 [====>.........................] - ETA: 5s - loss: 0.5204 - accuracy: 0.9497
 2100/11314 [====>.........................] - ETA: 5s - loss: 0.5202 - accuracy: 0.9497
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.5201 - accuracy: 0.9497
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.5200 - accuracy: 0.9497
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.5198 - accuracy: 0.9496
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.5197 - accuracy: 0.9496
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.5195 - accuracy: 0.9496
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.5194 - accuracy: 0.9496
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.5193 - accuracy: 0.9496
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.5191 - accuracy: 0.9496
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.5190 - accuracy: 0.9496
 3100/11314 [=======>......................] - ETA: 5s - loss: 0.5188 - accuracy: 0.9496
 3200/11314 [=======>......................] - ETA: 4s - loss: 0.5187 - accuracy: 0.9496
 3300/11314 [=======>......................] - ETA: 4s - loss: 0.5185 - accuracy: 0.9496
 3400/11314 [========>.....................] - ETA: 4s - loss: 0.5183 - accuracy: 0.9496
 3500/11314 [========>.....................] - ETA: 4s - loss: 0.5182 - accuracy: 0.9496
 3600/11314 [========>.....................] - ETA: 4s - loss: 0.5181 - accuracy: 0.9496
 3700/11314 [========>.....................] - ETA: 4s - loss: 0.5179 - accuracy: 0.9496
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.5178 - accuracy: 0.9496
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.5176 - accuracy: 0.9496
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.5175 - accuracy: 0.9496
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.5173 - accuracy: 0.9496
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.5172 - accuracy: 0.9495
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.5171 - accuracy: 0.9495
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.5169 - accuracy: 0.9495
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.5168 - accuracy: 0.9496
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.5166 - accuracy: 0.9496
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.5164 - accuracy: 0.9496
 4800/11314 [===========>..................] - ETA: 3s - loss: 0.5163 - accuracy: 0.9496
 4900/11314 [===========>..................] - ETA: 3s - loss: 0.5161 - accuracy: 0.9496
 5000/11314 [============>.................] - ETA: 3s - loss: 0.5160 - accuracy: 0.9496
 5100/11314 [============>.................] - ETA: 3s - loss: 0.5158 - accuracy: 0.9496
 5200/11314 [============>.................] - ETA: 3s - loss: 0.5157 - accuracy: 0.9496
 5300/11314 [=============>................] - ETA: 3s - loss: 0.5155 - accuracy: 0.9496
 5400/11314 [=============>................] - ETA: 3s - loss: 0.5154 - accuracy: 0.9497
 5500/11314 [=============>................] - ETA: 3s - loss: 0.5152 - accuracy: 0.9496
 5600/11314 [=============>................] - ETA: 3s - loss: 0.5151 - accuracy: 0.9496
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.5150 - accuracy: 0.9496
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.5148 - accuracy: 0.9496
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.5147 - accuracy: 0.9496
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.5145 - accuracy: 0.9496
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.5144 - accuracy: 0.9496
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.5143 - accuracy: 0.9496
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.5141 - accuracy: 0.9496
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.5140 - accuracy: 0.9496
 6500/11314 [================>.............] - ETA: 2s - loss: 0.5138 - accuracy: 0.9496
 6600/11314 [================>.............] - ETA: 2s - loss: 0.5137 - accuracy: 0.9496
 6700/11314 [================>.............] - ETA: 2s - loss: 0.5135 - accuracy: 0.9495
 6800/11314 [=================>............] - ETA: 2s - loss: 0.5134 - accuracy: 0.9495
 6900/11314 [=================>............] - ETA: 2s - loss: 0.5132 - accuracy: 0.9496
 7000/11314 [=================>............] - ETA: 2s - loss: 0.5131 - accuracy: 0.9495
 7100/11314 [=================>............] - ETA: 2s - loss: 0.5130 - accuracy: 0.9495
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.5128 - accuracy: 0.9496
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.5127 - accuracy: 0.9495
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.5125 - accuracy: 0.9496
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.5124 - accuracy: 0.9496
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.5122 - accuracy: 0.9496
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.5121 - accuracy: 0.9496
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.5119 - accuracy: 0.9496
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.5118 - accuracy: 0.9496
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.5116 - accuracy: 0.9496
 8100/11314 [====================>.........] - ETA: 1s - loss: 0.5115 - accuracy: 0.9496
 8200/11314 [====================>.........] - ETA: 1s - loss: 0.5113 - accuracy: 0.9496
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.5112 - accuracy: 0.9496
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.5111 - accuracy: 0.9496
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.5109 - accuracy: 0.9496
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.5108 - accuracy: 0.9496
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.5106 - accuracy: 0.9496
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.5105 - accuracy: 0.9496
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.5103 - accuracy: 0.9496
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.5102 - accuracy: 0.9496
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.5100 - accuracy: 0.9496
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.5099 - accuracy: 0.9496
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.5097 - accuracy: 0.9496
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.5096 - accuracy: 0.9496
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.5095 - accuracy: 0.9496
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.5093 - accuracy: 0.9496
 9700/11314 [========================>.....] - ETA: 0s - loss: 0.5092 - accuracy: 0.9496
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.5090 - accuracy: 0.9496
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.5089 - accuracy: 0.9496
10000/11314 [=========================>....] - ETA: 0s - loss: 0.5087 - accuracy: 0.9496
10100/11314 [=========================>....] - ETA: 0s - loss: 0.5086 - accuracy: 0.9496
10200/11314 [==========================>...] - ETA: 0s - loss: 0.5085 - accuracy: 0.9496
10300/11314 [==========================>...] - ETA: 0s - loss: 0.5083 - accuracy: 0.9496
10400/11314 [==========================>...] - ETA: 0s - loss: 0.5082 - accuracy: 0.9496
10500/11314 [==========================>...] - ETA: 0s - loss: 0.5080 - accuracy: 0.9496
10600/11314 [===========================>..] - ETA: 0s - loss: 0.5079 - accuracy: 0.9496
10700/11314 [===========================>..] - ETA: 0s - loss: 0.5078 - accuracy: 0.9496
10800/11314 [===========================>..] - ETA: 0s - loss: 0.5076 - accuracy: 0.9496
10900/11314 [===========================>..] - ETA: 0s - loss: 0.5075 - accuracy: 0.9496
11000/11314 [============================>.] - ETA: 0s - loss: 0.5073 - accuracy: 0.9496
11100/11314 [============================>.] - ETA: 0s - loss: 0.5072 - accuracy: 0.9496
11200/11314 [============================>.] - ETA: 0s - loss: 0.5071 - accuracy: 0.9496
11300/11314 [============================>.] - ETA: 0s - loss: 0.5069 - accuracy: 0.9496
11314/11314 [==============================] - 8s 721us/step - loss: 0.5069 - accuracy: 0.9496 - val_loss: 0.4905 - val_accuracy: 0.9496
Epoch 6/10

  100/11314 [..............................] - ETA: 6s - loss: 0.4906 - accuracy: 0.9495
  200/11314 [..............................] - ETA: 6s - loss: 0.4907 - accuracy: 0.9489
  300/11314 [..............................] - ETA: 6s - loss: 0.4904 - accuracy: 0.9493
  400/11314 [>.............................] - ETA: 6s - loss: 0.4903 - accuracy: 0.9492
  500/11314 [>.............................] - ETA: 6s - loss: 0.4900 - accuracy: 0.9495
  600/11314 [>.............................] - ETA: 6s - loss: 0.4900 - accuracy: 0.9494
  700/11314 [>.............................] - ETA: 6s - loss: 0.4898 - accuracy: 0.9495
  800/11314 [=>............................] - ETA: 6s - loss: 0.4897 - accuracy: 0.9494
  900/11314 [=>............................] - ETA: 6s - loss: 0.4894 - accuracy: 0.9496
 1000/11314 [=>............................] - ETA: 6s - loss: 0.4893 - accuracy: 0.9496
 1100/11314 [=>............................] - ETA: 6s - loss: 0.4891 - accuracy: 0.9498
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.4889 - accuracy: 0.9498
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.4888 - accuracy: 0.9498
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.4886 - accuracy: 0.9498
 1500/11314 [==>...........................] - ETA: 6s - loss: 0.4885 - accuracy: 0.9497
 1600/11314 [===>..........................] - ETA: 5s - loss: 0.4884 - accuracy: 0.9497
 1700/11314 [===>..........................] - ETA: 5s - loss: 0.4883 - accuracy: 0.9497
 1800/11314 [===>..........................] - ETA: 5s - loss: 0.4882 - accuracy: 0.9496
 1900/11314 [====>.........................] - ETA: 5s - loss: 0.4881 - accuracy: 0.9496
 2000/11314 [====>.........................] - ETA: 5s - loss: 0.4879 - accuracy: 0.9496
 2100/11314 [====>.........................] - ETA: 5s - loss: 0.4878 - accuracy: 0.9496
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.4877 - accuracy: 0.9496
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.4875 - accuracy: 0.9497
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.4873 - accuracy: 0.9497
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.4872 - accuracy: 0.9497
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.4871 - accuracy: 0.9497
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.4869 - accuracy: 0.9497
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.4867 - accuracy: 0.9498
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.4866 - accuracy: 0.9498
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.4865 - accuracy: 0.9498
 3100/11314 [=======>......................] - ETA: 5s - loss: 0.4864 - accuracy: 0.9498
 3200/11314 [=======>......................] - ETA: 5s - loss: 0.4862 - accuracy: 0.9498
 3300/11314 [=======>......................] - ETA: 4s - loss: 0.4861 - accuracy: 0.9497
 3400/11314 [========>.....................] - ETA: 4s - loss: 0.4860 - accuracy: 0.9497
 3500/11314 [========>.....................] - ETA: 4s - loss: 0.4858 - accuracy: 0.9497
 3600/11314 [========>.....................] - ETA: 4s - loss: 0.4857 - accuracy: 0.9497
 3700/11314 [========>.....................] - ETA: 4s - loss: 0.4856 - accuracy: 0.9497
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.4855 - accuracy: 0.9497
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.4854 - accuracy: 0.9496
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.4852 - accuracy: 0.9497
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.4851 - accuracy: 0.9497
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.4850 - accuracy: 0.9497
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.4848 - accuracy: 0.9497
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.4847 - accuracy: 0.9497
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.4845 - accuracy: 0.9497
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.4844 - accuracy: 0.9497
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.4843 - accuracy: 0.9497
 4800/11314 [===========>..................] - ETA: 4s - loss: 0.4841 - accuracy: 0.9497
 4900/11314 [===========>..................] - ETA: 3s - loss: 0.4840 - accuracy: 0.9497
 5000/11314 [============>.................] - ETA: 3s - loss: 0.4839 - accuracy: 0.9497
 5100/11314 [============>.................] - ETA: 3s - loss: 0.4837 - accuracy: 0.9497
 5200/11314 [============>.................] - ETA: 3s - loss: 0.4836 - accuracy: 0.9497
 5300/11314 [=============>................] - ETA: 3s - loss: 0.4835 - accuracy: 0.9497
 5400/11314 [=============>................] - ETA: 3s - loss: 0.4833 - accuracy: 0.9497
 5500/11314 [=============>................] - ETA: 3s - loss: 0.4832 - accuracy: 0.9497
 5600/11314 [=============>................] - ETA: 3s - loss: 0.4831 - accuracy: 0.9497
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.4829 - accuracy: 0.9497
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.4828 - accuracy: 0.9497
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.4827 - accuracy: 0.9497
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.4825 - accuracy: 0.9498
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.4824 - accuracy: 0.9498
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.4822 - accuracy: 0.9498
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.4821 - accuracy: 0.9498
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.4820 - accuracy: 0.9498
 6500/11314 [================>.............] - ETA: 2s - loss: 0.4819 - accuracy: 0.9497
 6600/11314 [================>.............] - ETA: 2s - loss: 0.4817 - accuracy: 0.9497
 6700/11314 [================>.............] - ETA: 2s - loss: 0.4816 - accuracy: 0.9497
 6800/11314 [=================>............] - ETA: 2s - loss: 0.4815 - accuracy: 0.9497
 6900/11314 [=================>............] - ETA: 2s - loss: 0.4814 - accuracy: 0.9497
 7000/11314 [=================>............] - ETA: 2s - loss: 0.4812 - accuracy: 0.9497
 7100/11314 [=================>............] - ETA: 2s - loss: 0.4811 - accuracy: 0.9497
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.4810 - accuracy: 0.9497
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.4808 - accuracy: 0.9497
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.4807 - accuracy: 0.9497
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.4806 - accuracy: 0.9497
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.4804 - accuracy: 0.9497
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.4803 - accuracy: 0.9497
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.4802 - accuracy: 0.9497
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.4801 - accuracy: 0.9497
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.4800 - accuracy: 0.9497
 8100/11314 [====================>.........] - ETA: 1s - loss: 0.4798 - accuracy: 0.9496
 8200/11314 [====================>.........] - ETA: 1s - loss: 0.4797 - accuracy: 0.9497
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.4796 - accuracy: 0.9496
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.4795 - accuracy: 0.9496
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.4793 - accuracy: 0.9496
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.4792 - accuracy: 0.9496
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.4791 - accuracy: 0.9496
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.4789 - accuracy: 0.9496
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.4788 - accuracy: 0.9496
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.4787 - accuracy: 0.9496
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.4786 - accuracy: 0.9496
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.4784 - accuracy: 0.9496
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.4783 - accuracy: 0.9496
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.4782 - accuracy: 0.9496
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.4780 - accuracy: 0.9496
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.4779 - accuracy: 0.9496
 9700/11314 [========================>.....] - ETA: 0s - loss: 0.4778 - accuracy: 0.9496
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.4777 - accuracy: 0.9496
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.4775 - accuracy: 0.9496
10000/11314 [=========================>....] - ETA: 0s - loss: 0.4774 - accuracy: 0.9496
10100/11314 [=========================>....] - ETA: 0s - loss: 0.4773 - accuracy: 0.9496
10200/11314 [==========================>...] - ETA: 0s - loss: 0.4772 - accuracy: 0.9496
10300/11314 [==========================>...] - ETA: 0s - loss: 0.4770 - accuracy: 0.9496
10400/11314 [==========================>...] - ETA: 0s - loss: 0.4769 - accuracy: 0.9496
10500/11314 [==========================>...] - ETA: 0s - loss: 0.4768 - accuracy: 0.9496
10600/11314 [===========================>..] - ETA: 0s - loss: 0.4766 - accuracy: 0.9496
10700/11314 [===========================>..] - ETA: 0s - loss: 0.4765 - accuracy: 0.9496
10800/11314 [===========================>..] - ETA: 0s - loss: 0.4764 - accuracy: 0.9496
10900/11314 [===========================>..] - ETA: 0s - loss: 0.4762 - accuracy: 0.9496
11000/11314 [============================>.] - ETA: 0s - loss: 0.4761 - accuracy: 0.9496
11100/11314 [============================>.] - ETA: 0s - loss: 0.4760 - accuracy: 0.9496
11200/11314 [============================>.] - ETA: 0s - loss: 0.4759 - accuracy: 0.9496
11300/11314 [============================>.] - ETA: 0s - loss: 0.4757 - accuracy: 0.9496
11314/11314 [==============================] - 8s 725us/step - loss: 0.4757 - accuracy: 0.9496 - val_loss: 0.4610 - val_accuracy: 0.9496
Epoch 7/10

  100/11314 [..............................] - ETA: 6s - loss: 0.4611 - accuracy: 0.9495
  200/11314 [..............................] - ETA: 6s - loss: 0.4611 - accuracy: 0.9492
  300/11314 [..............................] - ETA: 6s - loss: 0.4610 - accuracy: 0.9493
  400/11314 [>.............................] - ETA: 6s - loss: 0.4606 - accuracy: 0.9496
  500/11314 [>.............................] - ETA: 6s - loss: 0.4605 - accuracy: 0.9496
  600/11314 [>.............................] - ETA: 6s - loss: 0.4603 - accuracy: 0.9496
  700/11314 [>.............................] - ETA: 6s - loss: 0.4603 - accuracy: 0.9495
  800/11314 [=>............................] - ETA: 6s - loss: 0.4602 - accuracy: 0.9495
  900/11314 [=>............................] - ETA: 6s - loss: 0.4600 - accuracy: 0.9496
 1000/11314 [=>............................] - ETA: 6s - loss: 0.4599 - accuracy: 0.9495
 1100/11314 [=>............................] - ETA: 6s - loss: 0.4598 - accuracy: 0.9495
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.4598 - accuracy: 0.9493
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.4596 - accuracy: 0.9494
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.4595 - accuracy: 0.9494
 1500/11314 [==>...........................] - ETA: 6s - loss: 0.4593 - accuracy: 0.9495
 1600/11314 [===>..........................] - ETA: 6s - loss: 0.4592 - accuracy: 0.9495
 1700/11314 [===>..........................] - ETA: 5s - loss: 0.4591 - accuracy: 0.9494
 1800/11314 [===>..........................] - ETA: 5s - loss: 0.4591 - accuracy: 0.9494
 1900/11314 [====>.........................] - ETA: 5s - loss: 0.4589 - accuracy: 0.9494
 2000/11314 [====>.........................] - ETA: 5s - loss: 0.4588 - accuracy: 0.9494
 2100/11314 [====>.........................] - ETA: 5s - loss: 0.4587 - accuracy: 0.9493
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.4586 - accuracy: 0.9493
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.4585 - accuracy: 0.9493
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.4583 - accuracy: 0.9493
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.4582 - accuracy: 0.9494
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.4581 - accuracy: 0.9494
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.4579 - accuracy: 0.9494
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.4578 - accuracy: 0.9494
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.4577 - accuracy: 0.9494
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.4576 - accuracy: 0.9494
 3100/11314 [=======>......................] - ETA: 5s - loss: 0.4575 - accuracy: 0.9494
 3200/11314 [=======>......................] - ETA: 4s - loss: 0.4573 - accuracy: 0.9494
 3300/11314 [=======>......................] - ETA: 4s - loss: 0.4572 - accuracy: 0.9494
 3400/11314 [========>.....................] - ETA: 4s - loss: 0.4571 - accuracy: 0.9494
 3500/11314 [========>.....................] - ETA: 4s - loss: 0.4570 - accuracy: 0.9494
 3600/11314 [========>.....................] - ETA: 4s - loss: 0.4568 - accuracy: 0.9494
 3700/11314 [========>.....................] - ETA: 4s - loss: 0.4567 - accuracy: 0.9494
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.4566 - accuracy: 0.9494
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.4565 - accuracy: 0.9494
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.4564 - accuracy: 0.9494
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.4562 - accuracy: 0.9494
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.4561 - accuracy: 0.9494
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.4560 - accuracy: 0.9494
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.4558 - accuracy: 0.9495
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.4557 - accuracy: 0.9495
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.4556 - accuracy: 0.9495
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.4555 - accuracy: 0.9495
 4800/11314 [===========>..................] - ETA: 3s - loss: 0.4554 - accuracy: 0.9495
 4900/11314 [===========>..................] - ETA: 3s - loss: 0.4553 - accuracy: 0.9494
 5000/11314 [============>.................] - ETA: 3s - loss: 0.4552 - accuracy: 0.9494
 5100/11314 [============>.................] - ETA: 3s - loss: 0.4550 - accuracy: 0.9494
 5200/11314 [============>.................] - ETA: 3s - loss: 0.4549 - accuracy: 0.9494
 5300/11314 [=============>................] - ETA: 3s - loss: 0.4548 - accuracy: 0.9494
 5400/11314 [=============>................] - ETA: 3s - loss: 0.4547 - accuracy: 0.9494
 5500/11314 [=============>................] - ETA: 3s - loss: 0.4546 - accuracy: 0.9494
 5600/11314 [=============>................] - ETA: 3s - loss: 0.4544 - accuracy: 0.9494
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.4543 - accuracy: 0.9494
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.4542 - accuracy: 0.9494
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.4541 - accuracy: 0.9494
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.4540 - accuracy: 0.9494
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.4539 - accuracy: 0.9494
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.4538 - accuracy: 0.9494
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.4536 - accuracy: 0.9494
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.4535 - accuracy: 0.9494
 6500/11314 [================>.............] - ETA: 2s - loss: 0.4534 - accuracy: 0.9493
 6600/11314 [================>.............] - ETA: 2s - loss: 0.4533 - accuracy: 0.9494
 6700/11314 [================>.............] - ETA: 2s - loss: 0.4532 - accuracy: 0.9493
 6800/11314 [=================>............] - ETA: 2s - loss: 0.4531 - accuracy: 0.9493
 6900/11314 [=================>............] - ETA: 2s - loss: 0.4530 - accuracy: 0.9494
 7000/11314 [=================>............] - ETA: 2s - loss: 0.4528 - accuracy: 0.9494
 7100/11314 [=================>............] - ETA: 2s - loss: 0.4527 - accuracy: 0.9494
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.4526 - accuracy: 0.9494
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.4524 - accuracy: 0.9494
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.4523 - accuracy: 0.9494
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.4522 - accuracy: 0.9494
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.4521 - accuracy: 0.9495
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.4519 - accuracy: 0.9495
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.4518 - accuracy: 0.9495
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.4517 - accuracy: 0.9495
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.4516 - accuracy: 0.9495
 8100/11314 [====================>.........] - ETA: 1s - loss: 0.4514 - accuracy: 0.9495
 8200/11314 [====================>.........] - ETA: 1s - loss: 0.4513 - accuracy: 0.9495
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.4512 - accuracy: 0.9495
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.4511 - accuracy: 0.9495
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.4510 - accuracy: 0.9495
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.4509 - accuracy: 0.9495
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.4507 - accuracy: 0.9495
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.4506 - accuracy: 0.9495
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.4505 - accuracy: 0.9495
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.4504 - accuracy: 0.9495
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.4503 - accuracy: 0.9495
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.4502 - accuracy: 0.9495
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.4500 - accuracy: 0.9495
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.4499 - accuracy: 0.9495
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.4498 - accuracy: 0.9495
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.4497 - accuracy: 0.9495
 9700/11314 [========================>.....] - ETA: 0s - loss: 0.4496 - accuracy: 0.9495
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.4494 - accuracy: 0.9495
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.4493 - accuracy: 0.9495
10000/11314 [=========================>....] - ETA: 0s - loss: 0.4492 - accuracy: 0.9495
10100/11314 [=========================>....] - ETA: 0s - loss: 0.4491 - accuracy: 0.9495
10200/11314 [==========================>...] - ETA: 0s - loss: 0.4490 - accuracy: 0.9495
10300/11314 [==========================>...] - ETA: 0s - loss: 0.4489 - accuracy: 0.9496
10400/11314 [==========================>...] - ETA: 0s - loss: 0.4488 - accuracy: 0.9495
10500/11314 [==========================>...] - ETA: 0s - loss: 0.4486 - accuracy: 0.9495
10600/11314 [===========================>..] - ETA: 0s - loss: 0.4485 - accuracy: 0.9495
10700/11314 [===========================>..] - ETA: 0s - loss: 0.4484 - accuracy: 0.9496
10800/11314 [===========================>..] - ETA: 0s - loss: 0.4483 - accuracy: 0.9496
10900/11314 [===========================>..] - ETA: 0s - loss: 0.4482 - accuracy: 0.9496
11000/11314 [============================>.] - ETA: 0s - loss: 0.4480 - accuracy: 0.9496
11100/11314 [============================>.] - ETA: 0s - loss: 0.4479 - accuracy: 0.9496
11200/11314 [============================>.] - ETA: 0s - loss: 0.4478 - accuracy: 0.9496
11300/11314 [============================>.] - ETA: 0s - loss: 0.4477 - accuracy: 0.9496
11314/11314 [==============================] - 8s 723us/step - loss: 0.4477 - accuracy: 0.9496 - val_loss: 0.4344 - val_accuracy: 0.9496
Epoch 8/10

  100/11314 [..............................] - ETA: 6s - loss: 0.4345 - accuracy: 0.9495
  200/11314 [..............................] - ETA: 6s - loss: 0.4342 - accuracy: 0.9497
  300/11314 [..............................] - ETA: 6s - loss: 0.4341 - accuracy: 0.9496
  400/11314 [>.............................] - ETA: 6s - loss: 0.4336 - accuracy: 0.9503
  500/11314 [>.............................] - ETA: 6s - loss: 0.4336 - accuracy: 0.9500
  600/11314 [>.............................] - ETA: 6s - loss: 0.4335 - accuracy: 0.9500
  700/11314 [>.............................] - ETA: 6s - loss: 0.4334 - accuracy: 0.9500
  800/11314 [=>............................] - ETA: 6s - loss: 0.4334 - accuracy: 0.9499
  900/11314 [=>............................] - ETA: 6s - loss: 0.4332 - accuracy: 0.9500
 1000/11314 [=>............................] - ETA: 6s - loss: 0.4332 - accuracy: 0.9498
 1100/11314 [=>............................] - ETA: 6s - loss: 0.4331 - accuracy: 0.9498
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.4330 - accuracy: 0.9498
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.4329 - accuracy: 0.9498
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.4328 - accuracy: 0.9498
 1500/11314 [==>...........................] - ETA: 6s - loss: 0.4327 - accuracy: 0.9498
 1600/11314 [===>..........................] - ETA: 5s - loss: 0.4326 - accuracy: 0.9498
 1700/11314 [===>..........................] - ETA: 5s - loss: 0.4326 - accuracy: 0.9497
 1800/11314 [===>..........................] - ETA: 5s - loss: 0.4325 - accuracy: 0.9496
 1900/11314 [====>.........................] - ETA: 5s - loss: 0.4324 - accuracy: 0.9496
 2000/11314 [====>.........................] - ETA: 5s - loss: 0.4323 - accuracy: 0.9497
 2100/11314 [====>.........................] - ETA: 5s - loss: 0.4322 - accuracy: 0.9496
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.4321 - accuracy: 0.9496
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.4320 - accuracy: 0.9496
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.4319 - accuracy: 0.9495
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.4318 - accuracy: 0.9495
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.4317 - accuracy: 0.9495
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.4316 - accuracy: 0.9495
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.4315 - accuracy: 0.9495
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.4314 - accuracy: 0.9495
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.4313 - accuracy: 0.9495
 3100/11314 [=======>......................] - ETA: 5s - loss: 0.4312 - accuracy: 0.9495
 3200/11314 [=======>......................] - ETA: 4s - loss: 0.4310 - accuracy: 0.9495
 3300/11314 [=======>......................] - ETA: 4s - loss: 0.4309 - accuracy: 0.9495
 3400/11314 [========>.....................] - ETA: 4s - loss: 0.4308 - accuracy: 0.9495
 3500/11314 [========>.....................] - ETA: 4s - loss: 0.4307 - accuracy: 0.9495
 3600/11314 [========>.....................] - ETA: 4s - loss: 0.4306 - accuracy: 0.9495
 3700/11314 [========>.....................] - ETA: 4s - loss: 0.4305 - accuracy: 0.9495
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.4304 - accuracy: 0.9495
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.4303 - accuracy: 0.9495
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.4302 - accuracy: 0.9495
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.4301 - accuracy: 0.9495
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.4300 - accuracy: 0.9495
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.4299 - accuracy: 0.9495
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.4297 - accuracy: 0.9496
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.4296 - accuracy: 0.9495
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.4295 - accuracy: 0.9496
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.4294 - accuracy: 0.9496
 4800/11314 [===========>..................] - ETA: 3s - loss: 0.4293 - accuracy: 0.9496
 4900/11314 [===========>..................] - ETA: 3s - loss: 0.4291 - accuracy: 0.9496
 5000/11314 [============>.................] - ETA: 3s - loss: 0.4291 - accuracy: 0.9496
 5100/11314 [============>.................] - ETA: 3s - loss: 0.4289 - accuracy: 0.9496
 5200/11314 [============>.................] - ETA: 3s - loss: 0.4288 - accuracy: 0.9496
 5300/11314 [=============>................] - ETA: 3s - loss: 0.4287 - accuracy: 0.9497
 5400/11314 [=============>................] - ETA: 3s - loss: 0.4285 - accuracy: 0.9497
 5500/11314 [=============>................] - ETA: 3s - loss: 0.4284 - accuracy: 0.9497
 5600/11314 [=============>................] - ETA: 3s - loss: 0.4283 - accuracy: 0.9497
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.4282 - accuracy: 0.9497
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.4281 - accuracy: 0.9497
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.4280 - accuracy: 0.9497
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.4279 - accuracy: 0.9497
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.4278 - accuracy: 0.9497
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.4277 - accuracy: 0.9497
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.4276 - accuracy: 0.9497
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.4275 - accuracy: 0.9497
 6500/11314 [================>.............] - ETA: 2s - loss: 0.4274 - accuracy: 0.9497
 6600/11314 [================>.............] - ETA: 2s - loss: 0.4273 - accuracy: 0.9497
 6700/11314 [================>.............] - ETA: 2s - loss: 0.4272 - accuracy: 0.9497
 6800/11314 [=================>............] - ETA: 2s - loss: 0.4270 - accuracy: 0.9497
 6900/11314 [=================>............] - ETA: 2s - loss: 0.4270 - accuracy: 0.9497
 7000/11314 [=================>............] - ETA: 2s - loss: 0.4269 - accuracy: 0.9497
 7100/11314 [=================>............] - ETA: 2s - loss: 0.4267 - accuracy: 0.9497
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.4266 - accuracy: 0.9497
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.4265 - accuracy: 0.9497
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.4264 - accuracy: 0.9497
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.4263 - accuracy: 0.9497
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.4262 - accuracy: 0.9497
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.4261 - accuracy: 0.9496
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.4260 - accuracy: 0.9497
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.4259 - accuracy: 0.9497
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.4258 - accuracy: 0.9497
 8100/11314 [====================>.........] - ETA: 1s - loss: 0.4257 - accuracy: 0.9497
 8200/11314 [====================>.........] - ETA: 1s - loss: 0.4256 - accuracy: 0.9497
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.4255 - accuracy: 0.9496
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.4254 - accuracy: 0.9496
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.4253 - accuracy: 0.9497
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.4252 - accuracy: 0.9497
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.4251 - accuracy: 0.9497
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.4250 - accuracy: 0.9497
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.4248 - accuracy: 0.9497
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.4247 - accuracy: 0.9497
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.4246 - accuracy: 0.9497
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.4245 - accuracy: 0.9497
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.4244 - accuracy: 0.9497
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.4243 - accuracy: 0.9497
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.4242 - accuracy: 0.9497
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.4241 - accuracy: 0.9497
 9700/11314 [========================>.....] - ETA: 0s - loss: 0.4240 - accuracy: 0.9497
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.4239 - accuracy: 0.9497
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.4238 - accuracy: 0.9496
10000/11314 [=========================>....] - ETA: 0s - loss: 0.4237 - accuracy: 0.9496
10100/11314 [=========================>....] - ETA: 0s - loss: 0.4236 - accuracy: 0.9496
10200/11314 [==========================>...] - ETA: 0s - loss: 0.4235 - accuracy: 0.9496
10300/11314 [==========================>...] - ETA: 0s - loss: 0.4234 - accuracy: 0.9496
10400/11314 [==========================>...] - ETA: 0s - loss: 0.4233 - accuracy: 0.9496
10500/11314 [==========================>...] - ETA: 0s - loss: 0.4232 - accuracy: 0.9496
10600/11314 [===========================>..] - ETA: 0s - loss: 0.4231 - accuracy: 0.9496
10700/11314 [===========================>..] - ETA: 0s - loss: 0.4230 - accuracy: 0.9496
10800/11314 [===========================>..] - ETA: 0s - loss: 0.4229 - accuracy: 0.9496
10900/11314 [===========================>..] - ETA: 0s - loss: 0.4228 - accuracy: 0.9496
11000/11314 [============================>.] - ETA: 0s - loss: 0.4227 - accuracy: 0.9496
11100/11314 [============================>.] - ETA: 0s - loss: 0.4226 - accuracy: 0.9496
11200/11314 [============================>.] - ETA: 0s - loss: 0.4225 - accuracy: 0.9496
11300/11314 [============================>.] - ETA: 0s - loss: 0.4224 - accuracy: 0.9496
11314/11314 [==============================] - 8s 721us/step - loss: 0.4224 - accuracy: 0.9496 - val_loss: 0.4104 - val_accuracy: 0.9496
Epoch 9/10

  100/11314 [..............................] - ETA: 6s - loss: 0.4092 - accuracy: 0.9511
  200/11314 [..............................] - ETA: 6s - loss: 0.4096 - accuracy: 0.9505
  300/11314 [..............................] - ETA: 6s - loss: 0.4093 - accuracy: 0.9507
  400/11314 [>.............................] - ETA: 6s - loss: 0.4094 - accuracy: 0.9505
  500/11314 [>.............................] - ETA: 6s - loss: 0.4095 - accuracy: 0.9502
  600/11314 [>.............................] - ETA: 6s - loss: 0.4095 - accuracy: 0.9502
  700/11314 [>.............................] - ETA: 6s - loss: 0.4094 - accuracy: 0.9502
  800/11314 [=>............................] - ETA: 6s - loss: 0.4090 - accuracy: 0.9505
  900/11314 [=>............................] - ETA: 6s - loss: 0.4088 - accuracy: 0.9506
 1000/11314 [=>............................] - ETA: 6s - loss: 0.4089 - accuracy: 0.9504
 1100/11314 [=>............................] - ETA: 6s - loss: 0.4089 - accuracy: 0.9502
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.4089 - accuracy: 0.9501
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.4089 - accuracy: 0.9500
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.4089 - accuracy: 0.9499
 1500/11314 [==>...........................] - ETA: 5s - loss: 0.4088 - accuracy: 0.9499
 1600/11314 [===>..........................] - ETA: 5s - loss: 0.4088 - accuracy: 0.9498
 1700/11314 [===>..........................] - ETA: 5s - loss: 0.4087 - accuracy: 0.9498
 1800/11314 [===>..........................] - ETA: 5s - loss: 0.4086 - accuracy: 0.9498
 1900/11314 [====>.........................] - ETA: 5s - loss: 0.4086 - accuracy: 0.9497
 2000/11314 [====>.........................] - ETA: 5s - loss: 0.4085 - accuracy: 0.9497
 2100/11314 [====>.........................] - ETA: 5s - loss: 0.4085 - accuracy: 0.9496
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.4084 - accuracy: 0.9496
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.4083 - accuracy: 0.9496
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.4081 - accuracy: 0.9496
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.4080 - accuracy: 0.9496
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.4080 - accuracy: 0.9496
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.4078 - accuracy: 0.9496
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.4078 - accuracy: 0.9496
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.4077 - accuracy: 0.9495
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.4077 - accuracy: 0.9495
 3100/11314 [=======>......................] - ETA: 4s - loss: 0.4076 - accuracy: 0.9495
 3200/11314 [=======>......................] - ETA: 4s - loss: 0.4074 - accuracy: 0.9496
 3300/11314 [=======>......................] - ETA: 4s - loss: 0.4073 - accuracy: 0.9496
 3400/11314 [========>.....................] - ETA: 4s - loss: 0.4072 - accuracy: 0.9496
 3500/11314 [========>.....................] - ETA: 4s - loss: 0.4071 - accuracy: 0.9496
 3600/11314 [========>.....................] - ETA: 4s - loss: 0.4069 - accuracy: 0.9496
 3700/11314 [========>.....................] - ETA: 4s - loss: 0.4069 - accuracy: 0.9496
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.4068 - accuracy: 0.9497
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.4066 - accuracy: 0.9497
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.4065 - accuracy: 0.9497
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.4065 - accuracy: 0.9497
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.4064 - accuracy: 0.9497
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.4062 - accuracy: 0.9497
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.4062 - accuracy: 0.9497
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.4061 - accuracy: 0.9497
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.4059 - accuracy: 0.9497
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.4059 - accuracy: 0.9497
 4800/11314 [===========>..................] - ETA: 3s - loss: 0.4058 - accuracy: 0.9496
 4900/11314 [===========>..................] - ETA: 3s - loss: 0.4057 - accuracy: 0.9496
 5000/11314 [============>.................] - ETA: 3s - loss: 0.4056 - accuracy: 0.9497
 5100/11314 [============>.................] - ETA: 3s - loss: 0.4055 - accuracy: 0.9497
 5200/11314 [============>.................] - ETA: 3s - loss: 0.4054 - accuracy: 0.9497
 5300/11314 [=============>................] - ETA: 3s - loss: 0.4053 - accuracy: 0.9497
 5400/11314 [=============>................] - ETA: 3s - loss: 0.4051 - accuracy: 0.9497
 5500/11314 [=============>................] - ETA: 3s - loss: 0.4051 - accuracy: 0.9497
 5600/11314 [=============>................] - ETA: 3s - loss: 0.4050 - accuracy: 0.9497
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.4049 - accuracy: 0.9497
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.4048 - accuracy: 0.9497
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.4047 - accuracy: 0.9497
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.4046 - accuracy: 0.9497
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.4045 - accuracy: 0.9497
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.4044 - accuracy: 0.9497
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.4043 - accuracy: 0.9497
 6400/11314 [===============>..............] - ETA: 2s - loss: 0.4042 - accuracy: 0.9497
 6500/11314 [================>.............] - ETA: 2s - loss: 0.4041 - accuracy: 0.9497
 6600/11314 [================>.............] - ETA: 2s - loss: 0.4041 - accuracy: 0.9496
 6700/11314 [================>.............] - ETA: 2s - loss: 0.4040 - accuracy: 0.9496
 6800/11314 [=================>............] - ETA: 2s - loss: 0.4039 - accuracy: 0.9496
 6900/11314 [=================>............] - ETA: 2s - loss: 0.4038 - accuracy: 0.9496
 7000/11314 [=================>............] - ETA: 2s - loss: 0.4037 - accuracy: 0.9496
 7100/11314 [=================>............] - ETA: 2s - loss: 0.4036 - accuracy: 0.9496
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.4035 - accuracy: 0.9496
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.4034 - accuracy: 0.9496
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.4033 - accuracy: 0.9496
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.4032 - accuracy: 0.9496
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.4031 - accuracy: 0.9496
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.4031 - accuracy: 0.9496
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.4030 - accuracy: 0.9496
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.4029 - accuracy: 0.9496
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.4028 - accuracy: 0.9496
 8100/11314 [====================>.........] - ETA: 1s - loss: 0.4027 - accuracy: 0.9496
 8200/11314 [====================>.........] - ETA: 1s - loss: 0.4026 - accuracy: 0.9496
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.4025 - accuracy: 0.9496
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.4024 - accuracy: 0.9496
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.4023 - accuracy: 0.9496
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.4022 - accuracy: 0.9496
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.4021 - accuracy: 0.9496
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.4020 - accuracy: 0.9496
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.4019 - accuracy: 0.9496
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.4018 - accuracy: 0.9496
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.4017 - accuracy: 0.9496
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.4017 - accuracy: 0.9496
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.4016 - accuracy: 0.9496
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.4015 - accuracy: 0.9495
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.4014 - accuracy: 0.9495
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.4013 - accuracy: 0.9495
 9700/11314 [========================>.....] - ETA: 0s - loss: 0.4012 - accuracy: 0.9495
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.4011 - accuracy: 0.9495
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.4010 - accuracy: 0.9495
10000/11314 [=========================>....] - ETA: 0s - loss: 0.4009 - accuracy: 0.9495
10100/11314 [=========================>....] - ETA: 0s - loss: 0.4009 - accuracy: 0.9495
10200/11314 [==========================>...] - ETA: 0s - loss: 0.4008 - accuracy: 0.9495
10300/11314 [==========================>...] - ETA: 0s - loss: 0.4007 - accuracy: 0.9495
10400/11314 [==========================>...] - ETA: 0s - loss: 0.4006 - accuracy: 0.9495
10500/11314 [==========================>...] - ETA: 0s - loss: 0.4004 - accuracy: 0.9496
10600/11314 [===========================>..] - ETA: 0s - loss: 0.4003 - accuracy: 0.9496
10700/11314 [===========================>..] - ETA: 0s - loss: 0.4002 - accuracy: 0.9496
10800/11314 [===========================>..] - ETA: 0s - loss: 0.4001 - accuracy: 0.9496
10900/11314 [===========================>..] - ETA: 0s - loss: 0.4001 - accuracy: 0.9496
11000/11314 [============================>.] - ETA: 0s - loss: 0.4000 - accuracy: 0.9496
11100/11314 [============================>.] - ETA: 0s - loss: 0.3999 - accuracy: 0.9496
11200/11314 [============================>.] - ETA: 0s - loss: 0.3998 - accuracy: 0.9496
11300/11314 [============================>.] - ETA: 0s - loss: 0.3997 - accuracy: 0.9496
11314/11314 [==============================] - 8s 721us/step - loss: 0.3997 - accuracy: 0.9496 - val_loss: 0.3889 - val_accuracy: 0.9496
Epoch 10/10

  100/11314 [..............................] - ETA: 6s - loss: 0.3886 - accuracy: 0.9500
  200/11314 [..............................] - ETA: 6s - loss: 0.3887 - accuracy: 0.9497
  300/11314 [..............................] - ETA: 6s - loss: 0.3888 - accuracy: 0.9495
  400/11314 [>.............................] - ETA: 6s - loss: 0.3890 - accuracy: 0.9492
  500/11314 [>.............................] - ETA: 6s - loss: 0.3886 - accuracy: 0.9496
  600/11314 [>.............................] - ETA: 6s - loss: 0.3886 - accuracy: 0.9495
  700/11314 [>.............................] - ETA: 6s - loss: 0.3886 - accuracy: 0.9493
  800/11314 [=>............................] - ETA: 6s - loss: 0.3886 - accuracy: 0.9493
  900/11314 [=>............................] - ETA: 6s - loss: 0.3886 - accuracy: 0.9492
 1000/11314 [=>............................] - ETA: 6s - loss: 0.3884 - accuracy: 0.9493
 1100/11314 [=>............................] - ETA: 6s - loss: 0.3884 - accuracy: 0.9492
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.3884 - accuracy: 0.9491
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.3884 - accuracy: 0.9490
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.3883 - accuracy: 0.9490
 1500/11314 [==>...........................] - ETA: 5s - loss: 0.3881 - accuracy: 0.9491
 1600/11314 [===>..........................] - ETA: 5s - loss: 0.3880 - accuracy: 0.9491
 1700/11314 [===>..........................] - ETA: 5s - loss: 0.3880 - accuracy: 0.9490
 1800/11314 [===>..........................] - ETA: 5s - loss: 0.3878 - accuracy: 0.9492
 1900/11314 [====>.........................] - ETA: 5s - loss: 0.3877 - accuracy: 0.9492
 2000/11314 [====>.........................] - ETA: 5s - loss: 0.3876 - accuracy: 0.9491
 2100/11314 [====>.........................] - ETA: 5s - loss: 0.3875 - accuracy: 0.9491
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.3874 - accuracy: 0.9492
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.3874 - accuracy: 0.9491
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.3873 - accuracy: 0.9491
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.3872 - accuracy: 0.9492
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.3870 - accuracy: 0.9493
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.3869 - accuracy: 0.9492
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.3868 - accuracy: 0.9492
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.3868 - accuracy: 0.9492
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.3866 - accuracy: 0.9493
 3100/11314 [=======>......................] - ETA: 5s - loss: 0.3866 - accuracy: 0.9492
 3200/11314 [=======>......................] - ETA: 4s - loss: 0.3865 - accuracy: 0.9492
 3300/11314 [=======>......................] - ETA: 4s - loss: 0.3864 - accuracy: 0.9493
 3400/11314 [========>.....................] - ETA: 4s - loss: 0.3863 - accuracy: 0.9493
 3500/11314 [========>.....................] - ETA: 4s - loss: 0.3862 - accuracy: 0.9493
 3600/11314 [========>.....................] - ETA: 4s - loss: 0.3860 - accuracy: 0.9493
 3700/11314 [========>.....................] - ETA: 4s - loss: 0.3859 - accuracy: 0.9494
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.3858 - accuracy: 0.9494
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.3857 - accuracy: 0.9494
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.3856 - accuracy: 0.9494
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.3855 - accuracy: 0.9494
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.3854 - accuracy: 0.9494
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.3853 - accuracy: 0.9494
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.3853 - accuracy: 0.9494
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.3852 - accuracy: 0.9494
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.3851 - accuracy: 0.9494
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.3850 - accuracy: 0.9494
 4800/11314 [===========>..................] - ETA: 3s - loss: 0.3849 - accuracy: 0.9494
 4900/11314 [===========>..................] - ETA: 3s - loss: 0.3848 - accuracy: 0.9495
 5000/11314 [============>.................] - ETA: 3s - loss: 0.3847 - accuracy: 0.9495
 5100/11314 [============>.................] - ETA: 3s - loss: 0.3846 - accuracy: 0.9494
 5200/11314 [============>.................] - ETA: 3s - loss: 0.3845 - accuracy: 0.9495
 5300/11314 [=============>................] - ETA: 3s - loss: 0.3844 - accuracy: 0.9495
 5400/11314 [=============>................] - ETA: 3s - loss: 0.3843 - accuracy: 0.9495
 5500/11314 [=============>................] - ETA: 3s - loss: 0.3842 - accuracy: 0.9495
 5600/11314 [=============>................] - ETA: 3s - loss: 0.3842 - accuracy: 0.9495
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.3841 - accuracy: 0.9495
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.3840 - accuracy: 0.9495
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.3839 - accuracy: 0.9495
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.3838 - accuracy: 0.9495
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.3838 - accuracy: 0.9495
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.3837 - accuracy: 0.9494
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.3836 - accuracy: 0.9494
 6400/11314 [===============>..............] - ETA: 2s - loss: 0.3835 - accuracy: 0.9495
 6500/11314 [================>.............] - ETA: 2s - loss: 0.3834 - accuracy: 0.9495
 6600/11314 [================>.............] - ETA: 2s - loss: 0.3833 - accuracy: 0.9495
 6700/11314 [================>.............] - ETA: 2s - loss: 0.3832 - accuracy: 0.9495
 6800/11314 [=================>............] - ETA: 2s - loss: 0.3831 - accuracy: 0.9495
 6900/11314 [=================>............] - ETA: 2s - loss: 0.3831 - accuracy: 0.9495
 7000/11314 [=================>............] - ETA: 2s - loss: 0.3830 - accuracy: 0.9495
 7100/11314 [=================>............] - ETA: 2s - loss: 0.3829 - accuracy: 0.9495
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.3828 - accuracy: 0.9495
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.3827 - accuracy: 0.9495
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.3826 - accuracy: 0.9495
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.3825 - accuracy: 0.9495
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.3824 - accuracy: 0.9495
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.3824 - accuracy: 0.9495
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.3823 - accuracy: 0.9495
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.3822 - accuracy: 0.9495
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.3821 - accuracy: 0.9495
 8100/11314 [====================>.........] - ETA: 1s - loss: 0.3820 - accuracy: 0.9495
 8200/11314 [====================>.........] - ETA: 1s - loss: 0.3819 - accuracy: 0.9495
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.3818 - accuracy: 0.9495
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.3817 - accuracy: 0.9496
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.3816 - accuracy: 0.9496
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.3815 - accuracy: 0.9496
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.3814 - accuracy: 0.9496
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.3814 - accuracy: 0.9495
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.3813 - accuracy: 0.9495
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.3812 - accuracy: 0.9495
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.3811 - accuracy: 0.9495
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.3811 - accuracy: 0.9495
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.3810 - accuracy: 0.9495
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.3809 - accuracy: 0.9495
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.3808 - accuracy: 0.9495
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.3807 - accuracy: 0.9495
 9700/11314 [========================>.....] - ETA: 0s - loss: 0.3806 - accuracy: 0.9495
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.3806 - accuracy: 0.9495
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.3805 - accuracy: 0.9495
10000/11314 [=========================>....] - ETA: 0s - loss: 0.3804 - accuracy: 0.9495
10100/11314 [=========================>....] - ETA: 0s - loss: 0.3803 - accuracy: 0.9496
10200/11314 [==========================>...] - ETA: 0s - loss: 0.3802 - accuracy: 0.9495
10300/11314 [==========================>...] - ETA: 0s - loss: 0.3801 - accuracy: 0.9496
10400/11314 [==========================>...] - ETA: 0s - loss: 0.3800 - accuracy: 0.9496
10500/11314 [==========================>...] - ETA: 0s - loss: 0.3799 - accuracy: 0.9496
10600/11314 [===========================>..] - ETA: 0s - loss: 0.3798 - accuracy: 0.9496
10700/11314 [===========================>..] - ETA: 0s - loss: 0.3797 - accuracy: 0.9496
10800/11314 [===========================>..] - ETA: 0s - loss: 0.3796 - accuracy: 0.9496
10900/11314 [===========================>..] - ETA: 0s - loss: 0.3795 - accuracy: 0.9496
11000/11314 [============================>.] - ETA: 0s - loss: 0.3795 - accuracy: 0.9496
11100/11314 [============================>.] - ETA: 0s - loss: 0.3794 - accuracy: 0.9496
11200/11314 [============================>.] - ETA: 0s - loss: 0.3793 - accuracy: 0.9496
11300/11314 [============================>.] - ETA: 0s - loss: 0.3792 - accuracy: 0.9496
11314/11314 [==============================] - 8s 721us/step - loss: 0.3792 - accuracy: 0.9496 - val_loss: 0.3695 - val_accuracy: 0.9496
	=====> Test the model: model.predict()
TWENTY_NEWS_GROUPS
	Training accuracy score: 94.96%
	Loss: 0.3695
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
	It took 24.420624494552612 seconds


Applying NLTK feature extraction: X_test
	==> Replace unused UNICODE characters. 
	==> Text in lower case. 
	==> Apply Lemmatize using WordNet's built-in morphy function. Returns the input word unchanged if it cannot be found in WordNet 
	==> Remove English stop words
	Done: NLTK feature extraction (replace unused UNICODE characters, text in lower case, apply Lemmatize using WordNet, remove English stop words)  finished!
	It took 23.824209928512573 seconds

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
	===> Tokenizer: fit_on_texts(X_test)
	===> X_test = pad_sequences(list_sentences_test, maxlen=6000)
	=====> Training the model: model.fit()
Train on 25000 samples, validate on 25000 samples
Epoch 1/10

  100/25000 [..............................] - ETA: 2:13 - loss: 0.8173 - accuracy: 0.4400
  200/25000 [..............................] - ETA: 1:14 - loss: 0.7889 - accuracy: 0.4750
  300/25000 [..............................] - ETA: 54s - loss: 0.7944 - accuracy: 0.4667 
  400/25000 [..............................] - ETA: 44s - loss: 0.7966 - accuracy: 0.4625
  500/25000 [..............................] - ETA: 38s - loss: 0.7869 - accuracy: 0.4740
  600/25000 [..............................] - ETA: 34s - loss: 0.7727 - accuracy: 0.4917
  700/25000 [..............................] - ETA: 31s - loss: 0.7645 - accuracy: 0.5014
  800/25000 [..............................] - ETA: 29s - loss: 0.7628 - accuracy: 0.5025
  900/25000 [>.............................] - ETA: 27s - loss: 0.7651 - accuracy: 0.4978
 1000/25000 [>.............................] - ETA: 26s - loss: 0.7604 - accuracy: 0.5030
 1100/25000 [>.............................] - ETA: 25s - loss: 0.7613 - accuracy: 0.5000
 1200/25000 [>.............................] - ETA: 24s - loss: 0.7625 - accuracy: 0.4967
 1300/25000 [>.............................] - ETA: 23s - loss: 0.7619 - accuracy: 0.4962
 1400/25000 [>.............................] - ETA: 22s - loss: 0.7573 - accuracy: 0.5014
 1500/25000 [>.............................] - ETA: 22s - loss: 0.7568 - accuracy: 0.5007
 1600/25000 [>.............................] - ETA: 21s - loss: 0.7564 - accuracy: 0.4994
 1700/25000 [=>............................] - ETA: 20s - loss: 0.7548 - accuracy: 0.5006
 1800/25000 [=>............................] - ETA: 20s - loss: 0.7549 - accuracy: 0.4983
 1900/25000 [=>............................] - ETA: 20s - loss: 0.7516 - accuracy: 0.5016
 2000/25000 [=>............................] - ETA: 19s - loss: 0.7508 - accuracy: 0.5005
 2100/25000 [=>............................] - ETA: 19s - loss: 0.7501 - accuracy: 0.4990
 2200/25000 [=>............................] - ETA: 19s - loss: 0.7497 - accuracy: 0.4968
 2300/25000 [=>............................] - ETA: 18s - loss: 0.7483 - accuracy: 0.4970
 2400/25000 [=>............................] - ETA: 18s - loss: 0.7473 - accuracy: 0.4954
 2500/25000 [==>...........................] - ETA: 18s - loss: 0.7453 - accuracy: 0.4968
 2600/25000 [==>...........................] - ETA: 18s - loss: 0.7431 - accuracy: 0.4981
 2700/25000 [==>...........................] - ETA: 17s - loss: 0.7413 - accuracy: 0.4978
 2800/25000 [==>...........................] - ETA: 17s - loss: 0.7403 - accuracy: 0.4961
 2900/25000 [==>...........................] - ETA: 17s - loss: 0.7389 - accuracy: 0.4955
 3000/25000 [==>...........................] - ETA: 17s - loss: 0.7372 - accuracy: 0.4980
 3100/25000 [==>...........................] - ETA: 17s - loss: 0.7356 - accuracy: 0.4990
 3200/25000 [==>...........................] - ETA: 16s - loss: 0.7340 - accuracy: 0.5019
 3300/25000 [==>...........................] - ETA: 16s - loss: 0.7330 - accuracy: 0.4997
 3400/25000 [===>..........................] - ETA: 16s - loss: 0.7318 - accuracy: 0.4991
 3500/25000 [===>..........................] - ETA: 16s - loss: 0.7308 - accuracy: 0.4991
 3600/25000 [===>..........................] - ETA: 16s - loss: 0.7299 - accuracy: 0.4964
 3700/25000 [===>..........................] - ETA: 16s - loss: 0.7289 - accuracy: 0.4959
 3800/25000 [===>..........................] - ETA: 15s - loss: 0.7280 - accuracy: 0.4961
 3900/25000 [===>..........................] - ETA: 15s - loss: 0.7273 - accuracy: 0.4949
 4000/25000 [===>..........................] - ETA: 15s - loss: 0.7265 - accuracy: 0.4925
 4100/25000 [===>..........................] - ETA: 15s - loss: 0.7255 - accuracy: 0.4954
 4200/25000 [====>.........................] - ETA: 15s - loss: 0.7248 - accuracy: 0.4960
 4300/25000 [====>.........................] - ETA: 15s - loss: 0.7241 - accuracy: 0.4947
 4400/25000 [====>.........................] - ETA: 15s - loss: 0.7234 - accuracy: 0.4934
 4500/25000 [====>.........................] - ETA: 14s - loss: 0.7227 - accuracy: 0.4936
 4600/25000 [====>.........................] - ETA: 14s - loss: 0.7221 - accuracy: 0.4935
 4700/25000 [====>.........................] - ETA: 14s - loss: 0.7214 - accuracy: 0.4938
 4800/25000 [====>.........................] - ETA: 14s - loss: 0.7209 - accuracy: 0.4931
 4900/25000 [====>.........................] - ETA: 14s - loss: 0.7204 - accuracy: 0.4933
 5000/25000 [=====>........................] - ETA: 14s - loss: 0.7198 - accuracy: 0.4930
 5100/25000 [=====>........................] - ETA: 14s - loss: 0.7193 - accuracy: 0.4924
 5200/25000 [=====>........................] - ETA: 14s - loss: 0.7188 - accuracy: 0.4927
 5300/25000 [=====>........................] - ETA: 14s - loss: 0.7184 - accuracy: 0.4934
 5400/25000 [=====>........................] - ETA: 13s - loss: 0.7179 - accuracy: 0.4931
 5500/25000 [=====>........................] - ETA: 13s - loss: 0.7175 - accuracy: 0.4925
 5600/25000 [=====>........................] - ETA: 13s - loss: 0.7171 - accuracy: 0.4921
 5700/25000 [=====>........................] - ETA: 13s - loss: 0.7167 - accuracy: 0.4921
 5800/25000 [=====>........................] - ETA: 13s - loss: 0.7162 - accuracy: 0.4926
 5900/25000 [======>.......................] - ETA: 13s - loss: 0.7158 - accuracy: 0.4937
 6000/25000 [======>.......................] - ETA: 13s - loss: 0.7155 - accuracy: 0.4938
 6100/25000 [======>.......................] - ETA: 13s - loss: 0.7151 - accuracy: 0.4941
 6200/25000 [======>.......................] - ETA: 13s - loss: 0.7147 - accuracy: 0.4940
 6300/25000 [======>.......................] - ETA: 13s - loss: 0.7144 - accuracy: 0.4941
 6400/25000 [======>.......................] - ETA: 12s - loss: 0.7141 - accuracy: 0.4928
 6500/25000 [======>.......................] - ETA: 12s - loss: 0.7138 - accuracy: 0.4932
 6600/25000 [======>.......................] - ETA: 12s - loss: 0.7134 - accuracy: 0.4942
 6700/25000 [=======>......................] - ETA: 12s - loss: 0.7131 - accuracy: 0.4936
 6800/25000 [=======>......................] - ETA: 12s - loss: 0.7128 - accuracy: 0.4931
 6900/25000 [=======>......................] - ETA: 12s - loss: 0.7126 - accuracy: 0.4922
 7000/25000 [=======>......................] - ETA: 12s - loss: 0.7123 - accuracy: 0.4930
 7100/25000 [=======>......................] - ETA: 12s - loss: 0.7120 - accuracy: 0.4939
 7200/25000 [=======>......................] - ETA: 12s - loss: 0.7117 - accuracy: 0.4940
 7300/25000 [=======>......................] - ETA: 12s - loss: 0.7115 - accuracy: 0.4934
 7400/25000 [=======>......................] - ETA: 12s - loss: 0.7112 - accuracy: 0.4938
 7500/25000 [========>.....................] - ETA: 12s - loss: 0.7110 - accuracy: 0.4935
 7600/25000 [========>.....................] - ETA: 12s - loss: 0.7108 - accuracy: 0.4938
 7700/25000 [========>.....................] - ETA: 11s - loss: 0.7105 - accuracy: 0.4939
 7800/25000 [========>.....................] - ETA: 11s - loss: 0.7103 - accuracy: 0.4935
 7900/25000 [========>.....................] - ETA: 11s - loss: 0.7101 - accuracy: 0.4944
 8000/25000 [========>.....................] - ETA: 11s - loss: 0.7099 - accuracy: 0.4944
 8100/25000 [========>.....................] - ETA: 11s - loss: 0.7097 - accuracy: 0.4948
 8200/25000 [========>.....................] - ETA: 11s - loss: 0.7095 - accuracy: 0.4955
 8300/25000 [========>.....................] - ETA: 11s - loss: 0.7093 - accuracy: 0.4965
 8400/25000 [=========>....................] - ETA: 11s - loss: 0.7091 - accuracy: 0.4960
 8500/25000 [=========>....................] - ETA: 11s - loss: 0.7089 - accuracy: 0.4959
 8600/25000 [=========>....................] - ETA: 11s - loss: 0.7087 - accuracy: 0.4965
 8700/25000 [=========>....................] - ETA: 11s - loss: 0.7085 - accuracy: 0.4956
 8800/25000 [=========>....................] - ETA: 11s - loss: 0.7083 - accuracy: 0.4963
 8900/25000 [=========>....................] - ETA: 11s - loss: 0.7082 - accuracy: 0.4971
 9000/25000 [=========>....................] - ETA: 10s - loss: 0.7080 - accuracy: 0.4966
 9100/25000 [=========>....................] - ETA: 10s - loss: 0.7078 - accuracy: 0.4971
 9200/25000 [==========>...................] - ETA: 10s - loss: 0.7077 - accuracy: 0.4979
 9300/25000 [==========>...................] - ETA: 10s - loss: 0.7075 - accuracy: 0.4972
 9400/25000 [==========>...................] - ETA: 10s - loss: 0.7073 - accuracy: 0.4977
 9500/25000 [==========>...................] - ETA: 10s - loss: 0.7072 - accuracy: 0.4976
 9600/25000 [==========>...................] - ETA: 10s - loss: 0.7070 - accuracy: 0.4978
 9700/25000 [==========>...................] - ETA: 10s - loss: 0.7069 - accuracy: 0.4979
 9800/25000 [==========>...................] - ETA: 10s - loss: 0.7068 - accuracy: 0.4988
 9900/25000 [==========>...................] - ETA: 10s - loss: 0.7066 - accuracy: 0.4994
10000/25000 [===========>..................] - ETA: 10s - loss: 0.7065 - accuracy: 0.4997
10100/25000 [===========>..................] - ETA: 10s - loss: 0.7064 - accuracy: 0.5003
10200/25000 [===========>..................] - ETA: 9s - loss: 0.7062 - accuracy: 0.5016 
10300/25000 [===========>..................] - ETA: 9s - loss: 0.7061 - accuracy: 0.5017
10400/25000 [===========>..................] - ETA: 9s - loss: 0.7060 - accuracy: 0.5017
10500/25000 [===========>..................] - ETA: 9s - loss: 0.7058 - accuracy: 0.5020
10600/25000 [===========>..................] - ETA: 9s - loss: 0.7057 - accuracy: 0.5022
10700/25000 [===========>..................] - ETA: 9s - loss: 0.7055 - accuracy: 0.5027
10800/25000 [===========>..................] - ETA: 9s - loss: 0.7054 - accuracy: 0.5024
10900/25000 [============>.................] - ETA: 9s - loss: 0.7053 - accuracy: 0.5023
11000/25000 [============>.................] - ETA: 9s - loss: 0.7052 - accuracy: 0.5021
11100/25000 [============>.................] - ETA: 9s - loss: 0.7051 - accuracy: 0.5024
11200/25000 [============>.................] - ETA: 9s - loss: 0.7050 - accuracy: 0.5028
11300/25000 [============>.................] - ETA: 9s - loss: 0.7049 - accuracy: 0.5026
11400/25000 [============>.................] - ETA: 9s - loss: 0.7047 - accuracy: 0.5040
11500/25000 [============>.................] - ETA: 9s - loss: 0.7046 - accuracy: 0.5052
11600/25000 [============>.................] - ETA: 8s - loss: 0.7045 - accuracy: 0.5059
11700/25000 [=============>................] - ETA: 8s - loss: 0.7045 - accuracy: 0.5060
11800/25000 [=============>................] - ETA: 8s - loss: 0.7044 - accuracy: 0.5062
11900/25000 [=============>................] - ETA: 8s - loss: 0.7043 - accuracy: 0.5066
12000/25000 [=============>................] - ETA: 8s - loss: 0.7042 - accuracy: 0.5065
12100/25000 [=============>................] - ETA: 8s - loss: 0.7041 - accuracy: 0.5069
12200/25000 [=============>................] - ETA: 8s - loss: 0.7040 - accuracy: 0.5082
12300/25000 [=============>................] - ETA: 8s - loss: 0.7039 - accuracy: 0.5084
12400/25000 [=============>................] - ETA: 8s - loss: 0.7038 - accuracy: 0.5092
12500/25000 [==============>...............] - ETA: 8s - loss: 0.7037 - accuracy: 0.5102
12600/25000 [==============>...............] - ETA: 8s - loss: 0.7036 - accuracy: 0.5104
12700/25000 [==============>...............] - ETA: 8s - loss: 0.7035 - accuracy: 0.5111
12800/25000 [==============>...............] - ETA: 8s - loss: 0.7034 - accuracy: 0.5125
12900/25000 [==============>...............] - ETA: 8s - loss: 0.7033 - accuracy: 0.5139
13000/25000 [==============>...............] - ETA: 7s - loss: 0.7032 - accuracy: 0.5142
13100/25000 [==============>...............] - ETA: 7s - loss: 0.7031 - accuracy: 0.5156
13200/25000 [==============>...............] - ETA: 7s - loss: 0.7030 - accuracy: 0.5156
13300/25000 [==============>...............] - ETA: 7s - loss: 0.7030 - accuracy: 0.5149
13400/25000 [===============>..............] - ETA: 7s - loss: 0.7029 - accuracy: 0.5147
13500/25000 [===============>..............] - ETA: 7s - loss: 0.7028 - accuracy: 0.5138
13600/25000 [===============>..............] - ETA: 7s - loss: 0.7027 - accuracy: 0.5140
13700/25000 [===============>..............] - ETA: 7s - loss: 0.7026 - accuracy: 0.5147
13800/25000 [===============>..............] - ETA: 7s - loss: 0.7025 - accuracy: 0.5146
13900/25000 [===============>..............] - ETA: 7s - loss: 0.7024 - accuracy: 0.5146
14000/25000 [===============>..............] - ETA: 7s - loss: 0.7023 - accuracy: 0.5144
14100/25000 [===============>..............] - ETA: 7s - loss: 0.7022 - accuracy: 0.5135
14200/25000 [================>.............] - ETA: 7s - loss: 0.7021 - accuracy: 0.5139
14300/25000 [================>.............] - ETA: 7s - loss: 0.7019 - accuracy: 0.5141
14400/25000 [================>.............] - ETA: 6s - loss: 0.7018 - accuracy: 0.5144
14500/25000 [================>.............] - ETA: 6s - loss: 0.7017 - accuracy: 0.5144
14600/25000 [================>.............] - ETA: 6s - loss: 0.7016 - accuracy: 0.5146
14700/25000 [================>.............] - ETA: 6s - loss: 0.7014 - accuracy: 0.5142
14800/25000 [================>.............] - ETA: 6s - loss: 0.7014 - accuracy: 0.5136
14900/25000 [================>.............] - ETA: 6s - loss: 0.7012 - accuracy: 0.5137
15000/25000 [=================>............] - ETA: 6s - loss: 0.7011 - accuracy: 0.5137
15100/25000 [=================>............] - ETA: 6s - loss: 0.7010 - accuracy: 0.5138
15200/25000 [=================>............] - ETA: 6s - loss: 0.7008 - accuracy: 0.5141
15300/25000 [=================>............] - ETA: 6s - loss: 0.7007 - accuracy: 0.5141
15400/25000 [=================>............] - ETA: 6s - loss: 0.7005 - accuracy: 0.5143
15500/25000 [=================>............] - ETA: 6s - loss: 0.7004 - accuracy: 0.5142
15600/25000 [=================>............] - ETA: 6s - loss: 0.7001 - accuracy: 0.5144
15700/25000 [=================>............] - ETA: 6s - loss: 0.7001 - accuracy: 0.5138
15800/25000 [=================>............] - ETA: 6s - loss: 0.6998 - accuracy: 0.5138
15900/25000 [==================>...........] - ETA: 5s - loss: 0.6996 - accuracy: 0.5136
16000/25000 [==================>...........] - ETA: 5s - loss: 0.6993 - accuracy: 0.5140
16100/25000 [==================>...........] - ETA: 5s - loss: 0.6990 - accuracy: 0.5142
16200/25000 [==================>...........] - ETA: 5s - loss: 0.6989 - accuracy: 0.5136
16300/25000 [==================>...........] - ETA: 5s - loss: 0.6987 - accuracy: 0.5136
16400/25000 [==================>...........] - ETA: 5s - loss: 0.6985 - accuracy: 0.5132
16500/25000 [==================>...........] - ETA: 5s - loss: 0.6983 - accuracy: 0.5136
16600/25000 [==================>...........] - ETA: 5s - loss: 0.6981 - accuracy: 0.5135
16700/25000 [===================>..........] - ETA: 5s - loss: 0.6980 - accuracy: 0.5135
16800/25000 [===================>..........] - ETA: 5s - loss: 0.6977 - accuracy: 0.5134
16900/25000 [===================>..........] - ETA: 5s - loss: 0.6976 - accuracy: 0.5126
17000/25000 [===================>..........] - ETA: 5s - loss: 0.6973 - accuracy: 0.5122
17100/25000 [===================>..........] - ETA: 5s - loss: 0.6969 - accuracy: 0.5124
17200/25000 [===================>..........] - ETA: 5s - loss: 0.6967 - accuracy: 0.5119
17300/25000 [===================>..........] - ETA: 5s - loss: 0.6964 - accuracy: 0.5116
17400/25000 [===================>..........] - ETA: 4s - loss: 0.6960 - accuracy: 0.5117
17500/25000 [====================>.........] - ETA: 4s - loss: 0.6956 - accuracy: 0.5117
17600/25000 [====================>.........] - ETA: 4s - loss: 0.6953 - accuracy: 0.5115
17700/25000 [====================>.........] - ETA: 4s - loss: 0.6948 - accuracy: 0.5116
17800/25000 [====================>.........] - ETA: 4s - loss: 0.6945 - accuracy: 0.5113
17900/25000 [====================>.........] - ETA: 4s - loss: 0.6940 - accuracy: 0.5116
18000/25000 [====================>.........] - ETA: 4s - loss: 0.6935 - accuracy: 0.5117
18100/25000 [====================>.........] - ETA: 4s - loss: 0.6932 - accuracy: 0.5114
18200/25000 [====================>.........] - ETA: 4s - loss: 0.6927 - accuracy: 0.5117
18300/25000 [====================>.........] - ETA: 4s - loss: 0.6921 - accuracy: 0.5120
18400/25000 [=====================>........] - ETA: 4s - loss: 0.6917 - accuracy: 0.5120
18500/25000 [=====================>........] - ETA: 4s - loss: 0.6911 - accuracy: 0.5124
18600/25000 [=====================>........] - ETA: 4s - loss: 0.6909 - accuracy: 0.5123
18700/25000 [=====================>........] - ETA: 4s - loss: 0.6902 - accuracy: 0.5129
18800/25000 [=====================>........] - ETA: 4s - loss: 0.6897 - accuracy: 0.5137
18900/25000 [=====================>........] - ETA: 3s - loss: 0.6893 - accuracy: 0.5138
19000/25000 [=====================>........] - ETA: 3s - loss: 0.6889 - accuracy: 0.5141
19100/25000 [=====================>........] - ETA: 3s - loss: 0.6883 - accuracy: 0.5146
19200/25000 [======================>.......] - ETA: 3s - loss: 0.6880 - accuracy: 0.5146
19300/25000 [======================>.......] - ETA: 3s - loss: 0.6873 - accuracy: 0.5150
19400/25000 [======================>.......] - ETA: 3s - loss: 0.6866 - accuracy: 0.5159
19500/25000 [======================>.......] - ETA: 3s - loss: 0.6860 - accuracy: 0.5164
19600/25000 [======================>.......] - ETA: 3s - loss: 0.6854 - accuracy: 0.5171
19700/25000 [======================>.......] - ETA: 3s - loss: 0.6848 - accuracy: 0.5176
19800/25000 [======================>.......] - ETA: 3s - loss: 0.6841 - accuracy: 0.5183
19900/25000 [======================>.......] - ETA: 3s - loss: 0.6836 - accuracy: 0.5188
20000/25000 [=======================>......] - ETA: 3s - loss: 0.6834 - accuracy: 0.5193
20100/25000 [=======================>......] - ETA: 3s - loss: 0.6828 - accuracy: 0.5204
20200/25000 [=======================>......] - ETA: 3s - loss: 0.6822 - accuracy: 0.5216
20300/25000 [=======================>......] - ETA: 3s - loss: 0.6817 - accuracy: 0.5231
20400/25000 [=======================>......] - ETA: 2s - loss: 0.6812 - accuracy: 0.5243
20500/25000 [=======================>......] - ETA: 2s - loss: 0.6807 - accuracy: 0.5254
20600/25000 [=======================>......] - ETA: 2s - loss: 0.6799 - accuracy: 0.5268
20700/25000 [=======================>......] - ETA: 2s - loss: 0.6794 - accuracy: 0.5278
20800/25000 [=======================>......] - ETA: 2s - loss: 0.6791 - accuracy: 0.5286
20900/25000 [========================>.....] - ETA: 2s - loss: 0.6786 - accuracy: 0.5298
21000/25000 [========================>.....] - ETA: 2s - loss: 0.6781 - accuracy: 0.5312
21100/25000 [========================>.....] - ETA: 2s - loss: 0.6776 - accuracy: 0.5324
21200/25000 [========================>.....] - ETA: 2s - loss: 0.6768 - accuracy: 0.5339
21300/25000 [========================>.....] - ETA: 2s - loss: 0.6764 - accuracy: 0.5349
21400/25000 [========================>.....] - ETA: 2s - loss: 0.6761 - accuracy: 0.5357
21500/25000 [========================>.....] - ETA: 2s - loss: 0.6757 - accuracy: 0.5365
21600/25000 [========================>.....] - ETA: 2s - loss: 0.6751 - accuracy: 0.5379
21700/25000 [=========================>....] - ETA: 2s - loss: 0.6746 - accuracy: 0.5390
21800/25000 [=========================>....] - ETA: 2s - loss: 0.6739 - accuracy: 0.5405
21900/25000 [=========================>....] - ETA: 1s - loss: 0.6734 - accuracy: 0.5418
22000/25000 [=========================>....] - ETA: 1s - loss: 0.6729 - accuracy: 0.5430
22100/25000 [=========================>....] - ETA: 1s - loss: 0.6723 - accuracy: 0.5444
22200/25000 [=========================>....] - ETA: 1s - loss: 0.6717 - accuracy: 0.5458
22300/25000 [=========================>....] - ETA: 1s - loss: 0.6712 - accuracy: 0.5469
22400/25000 [=========================>....] - ETA: 1s - loss: 0.6706 - accuracy: 0.5481
22500/25000 [==========================>...] - ETA: 1s - loss: 0.6701 - accuracy: 0.5492
22600/25000 [==========================>...] - ETA: 1s - loss: 0.6699 - accuracy: 0.5501
22700/25000 [==========================>...] - ETA: 1s - loss: 0.6692 - accuracy: 0.5515
22800/25000 [==========================>...] - ETA: 1s - loss: 0.6689 - accuracy: 0.5528
22900/25000 [==========================>...] - ETA: 1s - loss: 0.6685 - accuracy: 0.5539
23000/25000 [==========================>...] - ETA: 1s - loss: 0.6681 - accuracy: 0.5550
23100/25000 [==========================>...] - ETA: 1s - loss: 0.6675 - accuracy: 0.5561
23200/25000 [==========================>...] - ETA: 1s - loss: 0.6673 - accuracy: 0.5570
23300/25000 [==========================>...] - ETA: 1s - loss: 0.6666 - accuracy: 0.5581
23400/25000 [===========================>..] - ETA: 1s - loss: 0.6661 - accuracy: 0.5593
23500/25000 [===========================>..] - ETA: 0s - loss: 0.6659 - accuracy: 0.5602
23600/25000 [===========================>..] - ETA: 0s - loss: 0.6656 - accuracy: 0.5609
23700/25000 [===========================>..] - ETA: 0s - loss: 0.6652 - accuracy: 0.5622
23800/25000 [===========================>..] - ETA: 0s - loss: 0.6646 - accuracy: 0.5632
23900/25000 [===========================>..] - ETA: 0s - loss: 0.6640 - accuracy: 0.5642
24000/25000 [===========================>..] - ETA: 0s - loss: 0.6634 - accuracy: 0.5653
24100/25000 [===========================>..] - ETA: 0s - loss: 0.6628 - accuracy: 0.5664
24200/25000 [============================>.] - ETA: 0s - loss: 0.6622 - accuracy: 0.5674
24300/25000 [============================>.] - ETA: 0s - loss: 0.6617 - accuracy: 0.5686
24400/25000 [============================>.] - ETA: 0s - loss: 0.6609 - accuracy: 0.5701
24500/25000 [============================>.] - ETA: 0s - loss: 0.6607 - accuracy: 0.5711
24600/25000 [============================>.] - ETA: 0s - loss: 0.6602 - accuracy: 0.5721
24700/25000 [============================>.] - ETA: 0s - loss: 0.6597 - accuracy: 0.5732
24800/25000 [============================>.] - ETA: 0s - loss: 0.6592 - accuracy: 0.5743
24900/25000 [============================>.] - ETA: 0s - loss: 0.6587 - accuracy: 0.5751
25000/25000 [==============================] - 20s 803us/step - loss: 0.6582 - accuracy: 0.5763 - val_loss: 0.5404 - val_accuracy: 0.8320
Epoch 2/10

  100/25000 [..............................] - ETA: 15s - loss: 0.4999 - accuracy: 0.8900
  200/25000 [..............................] - ETA: 14s - loss: 0.4785 - accuracy: 0.8900
  300/25000 [..............................] - ETA: 14s - loss: 0.4952 - accuracy: 0.8733
  400/25000 [..............................] - ETA: 15s - loss: 0.4991 - accuracy: 0.8650
  500/25000 [..............................] - ETA: 15s - loss: 0.5140 - accuracy: 0.8640
  600/25000 [..............................] - ETA: 15s - loss: 0.5126 - accuracy: 0.8583
  700/25000 [..............................] - ETA: 14s - loss: 0.5138 - accuracy: 0.8614
  800/25000 [..............................] - ETA: 14s - loss: 0.5175 - accuracy: 0.8587
  900/25000 [>.............................] - ETA: 14s - loss: 0.5148 - accuracy: 0.8644
 1000/25000 [>.............................] - ETA: 14s - loss: 0.5159 - accuracy: 0.8640
 1100/25000 [>.............................] - ETA: 14s - loss: 0.5117 - accuracy: 0.8645
 1200/25000 [>.............................] - ETA: 14s - loss: 0.5170 - accuracy: 0.8592
 1300/25000 [>.............................] - ETA: 14s - loss: 0.5201 - accuracy: 0.8531
 1400/25000 [>.............................] - ETA: 14s - loss: 0.5243 - accuracy: 0.8507
 1500/25000 [>.............................] - ETA: 14s - loss: 0.5250 - accuracy: 0.8493
 1600/25000 [>.............................] - ETA: 14s - loss: 0.5279 - accuracy: 0.8444
 1700/25000 [=>............................] - ETA: 14s - loss: 0.5254 - accuracy: 0.8476
 1800/25000 [=>............................] - ETA: 14s - loss: 0.5226 - accuracy: 0.8494
 1900/25000 [=>............................] - ETA: 14s - loss: 0.5257 - accuracy: 0.8479
 2000/25000 [=>............................] - ETA: 14s - loss: 0.5265 - accuracy: 0.8485
 2100/25000 [=>............................] - ETA: 14s - loss: 0.5278 - accuracy: 0.8481
 2200/25000 [=>............................] - ETA: 13s - loss: 0.5294 - accuracy: 0.8455
 2300/25000 [=>............................] - ETA: 13s - loss: 0.5302 - accuracy: 0.8426
 2400/25000 [=>............................] - ETA: 13s - loss: 0.5278 - accuracy: 0.8446
 2500/25000 [==>...........................] - ETA: 13s - loss: 0.5268 - accuracy: 0.8456
 2600/25000 [==>...........................] - ETA: 13s - loss: 0.5283 - accuracy: 0.8454
 2700/25000 [==>...........................] - ETA: 13s - loss: 0.5285 - accuracy: 0.8459
 2800/25000 [==>...........................] - ETA: 13s - loss: 0.5290 - accuracy: 0.8446
 2900/25000 [==>...........................] - ETA: 13s - loss: 0.5263 - accuracy: 0.8469
 3000/25000 [==>...........................] - ETA: 13s - loss: 0.5259 - accuracy: 0.8483
 3100/25000 [==>...........................] - ETA: 13s - loss: 0.5254 - accuracy: 0.8497
 3200/25000 [==>...........................] - ETA: 13s - loss: 0.5256 - accuracy: 0.8512
 3300/25000 [==>...........................] - ETA: 13s - loss: 0.5249 - accuracy: 0.8521
 3400/25000 [===>..........................] - ETA: 13s - loss: 0.5276 - accuracy: 0.8500
 3500/25000 [===>..........................] - ETA: 13s - loss: 0.5296 - accuracy: 0.8471
 3600/25000 [===>..........................] - ETA: 13s - loss: 0.5309 - accuracy: 0.8458
 3700/25000 [===>..........................] - ETA: 13s - loss: 0.5295 - accuracy: 0.8459
 3800/25000 [===>..........................] - ETA: 13s - loss: 0.5303 - accuracy: 0.8450
 3900/25000 [===>..........................] - ETA: 12s - loss: 0.5321 - accuracy: 0.8444
 4000/25000 [===>..........................] - ETA: 12s - loss: 0.5327 - accuracy: 0.8440
 4100/25000 [===>..........................] - ETA: 12s - loss: 0.5323 - accuracy: 0.8437
 4200/25000 [====>.........................] - ETA: 12s - loss: 0.5317 - accuracy: 0.8443
 4300/25000 [====>.........................] - ETA: 12s - loss: 0.5325 - accuracy: 0.8440
 4400/25000 [====>.........................] - ETA: 12s - loss: 0.5330 - accuracy: 0.8436
 4500/25000 [====>.........................] - ETA: 12s - loss: 0.5328 - accuracy: 0.8442
 4600/25000 [====>.........................] - ETA: 12s - loss: 0.5334 - accuracy: 0.8424
 4700/25000 [====>.........................] - ETA: 12s - loss: 0.5332 - accuracy: 0.8421
 4800/25000 [====>.........................] - ETA: 12s - loss: 0.5317 - accuracy: 0.8433
 4900/25000 [====>.........................] - ETA: 12s - loss: 0.5318 - accuracy: 0.8441
 5000/25000 [=====>........................] - ETA: 12s - loss: 0.5313 - accuracy: 0.8444
 5100/25000 [=====>........................] - ETA: 12s - loss: 0.5299 - accuracy: 0.8455
 5200/25000 [=====>........................] - ETA: 12s - loss: 0.5299 - accuracy: 0.8462
 5300/25000 [=====>........................] - ETA: 12s - loss: 0.5297 - accuracy: 0.8457
 5400/25000 [=====>........................] - ETA: 12s - loss: 0.5289 - accuracy: 0.8459
 5500/25000 [=====>........................] - ETA: 11s - loss: 0.5281 - accuracy: 0.8456
 5600/25000 [=====>........................] - ETA: 11s - loss: 0.5282 - accuracy: 0.8448
 5700/25000 [=====>........................] - ETA: 11s - loss: 0.5263 - accuracy: 0.8468
 5800/25000 [=====>........................] - ETA: 11s - loss: 0.5254 - accuracy: 0.8474
 5900/25000 [======>.......................] - ETA: 11s - loss: 0.5252 - accuracy: 0.8471
 6000/25000 [======>.......................] - ETA: 11s - loss: 0.5248 - accuracy: 0.8475
 6100/25000 [======>.......................] - ETA: 11s - loss: 0.5252 - accuracy: 0.8475
 6200/25000 [======>.......................] - ETA: 11s - loss: 0.5253 - accuracy: 0.8476
 6300/25000 [======>.......................] - ETA: 11s - loss: 0.5249 - accuracy: 0.8479
 6400/25000 [======>.......................] - ETA: 11s - loss: 0.5245 - accuracy: 0.8483
 6500/25000 [======>.......................] - ETA: 11s - loss: 0.5239 - accuracy: 0.8488
 6600/25000 [======>.......................] - ETA: 11s - loss: 0.5239 - accuracy: 0.8486
 6700/25000 [=======>......................] - ETA: 11s - loss: 0.5231 - accuracy: 0.8488
 6800/25000 [=======>......................] - ETA: 11s - loss: 0.5234 - accuracy: 0.8487
 6900/25000 [=======>......................] - ETA: 11s - loss: 0.5226 - accuracy: 0.8483
 7000/25000 [=======>......................] - ETA: 11s - loss: 0.5222 - accuracy: 0.8483
 7100/25000 [=======>......................] - ETA: 10s - loss: 0.5223 - accuracy: 0.8476
 7200/25000 [=======>......................] - ETA: 10s - loss: 0.5220 - accuracy: 0.8478
 7300/25000 [=======>......................] - ETA: 10s - loss: 0.5219 - accuracy: 0.8473
 7400/25000 [=======>......................] - ETA: 10s - loss: 0.5221 - accuracy: 0.8472
 7500/25000 [========>.....................] - ETA: 10s - loss: 0.5222 - accuracy: 0.8467
 7600/25000 [========>.....................] - ETA: 10s - loss: 0.5222 - accuracy: 0.8459
 7700/25000 [========>.....................] - ETA: 10s - loss: 0.5223 - accuracy: 0.8462
 7800/25000 [========>.....................] - ETA: 10s - loss: 0.5221 - accuracy: 0.8463
 7900/25000 [========>.....................] - ETA: 10s - loss: 0.5218 - accuracy: 0.8468
 8000/25000 [========>.....................] - ETA: 10s - loss: 0.5215 - accuracy: 0.8470
 8100/25000 [========>.....................] - ETA: 10s - loss: 0.5214 - accuracy: 0.8469
 8200/25000 [========>.....................] - ETA: 10s - loss: 0.5214 - accuracy: 0.8467
 8300/25000 [========>.....................] - ETA: 10s - loss: 0.5212 - accuracy: 0.8469
 8400/25000 [=========>....................] - ETA: 10s - loss: 0.5208 - accuracy: 0.8465
 8500/25000 [=========>....................] - ETA: 10s - loss: 0.5211 - accuracy: 0.8465
 8600/25000 [=========>....................] - ETA: 10s - loss: 0.5204 - accuracy: 0.8466
 8700/25000 [=========>....................] - ETA: 10s - loss: 0.5200 - accuracy: 0.8468
 8800/25000 [=========>....................] - ETA: 9s - loss: 0.5194 - accuracy: 0.8472 
 8900/25000 [=========>....................] - ETA: 9s - loss: 0.5192 - accuracy: 0.8470
 9000/25000 [=========>....................] - ETA: 9s - loss: 0.5195 - accuracy: 0.8467
 9100/25000 [=========>....................] - ETA: 9s - loss: 0.5191 - accuracy: 0.8468
 9200/25000 [==========>...................] - ETA: 9s - loss: 0.5192 - accuracy: 0.8461
 9300/25000 [==========>...................] - ETA: 9s - loss: 0.5193 - accuracy: 0.8457
 9400/25000 [==========>...................] - ETA: 9s - loss: 0.5189 - accuracy: 0.8460
 9500/25000 [==========>...................] - ETA: 9s - loss: 0.5187 - accuracy: 0.8462
 9600/25000 [==========>...................] - ETA: 9s - loss: 0.5185 - accuracy: 0.8462
 9700/25000 [==========>...................] - ETA: 9s - loss: 0.5185 - accuracy: 0.8462
 9800/25000 [==========>...................] - ETA: 9s - loss: 0.5185 - accuracy: 0.8460
 9900/25000 [==========>...................] - ETA: 9s - loss: 0.5182 - accuracy: 0.8460
10000/25000 [===========>..................] - ETA: 9s - loss: 0.5180 - accuracy: 0.8467
10100/25000 [===========>..................] - ETA: 9s - loss: 0.5178 - accuracy: 0.8470
10200/25000 [===========>..................] - ETA: 9s - loss: 0.5177 - accuracy: 0.8473
10300/25000 [===========>..................] - ETA: 9s - loss: 0.5175 - accuracy: 0.8475
10400/25000 [===========>..................] - ETA: 8s - loss: 0.5176 - accuracy: 0.8469
10500/25000 [===========>..................] - ETA: 8s - loss: 0.5175 - accuracy: 0.8467
10600/25000 [===========>..................] - ETA: 8s - loss: 0.5172 - accuracy: 0.8466
10700/25000 [===========>..................] - ETA: 8s - loss: 0.5173 - accuracy: 0.8461
10800/25000 [===========>..................] - ETA: 8s - loss: 0.5174 - accuracy: 0.8466
10900/25000 [============>.................] - ETA: 8s - loss: 0.5172 - accuracy: 0.8468
11000/25000 [============>.................] - ETA: 8s - loss: 0.5170 - accuracy: 0.8467
11100/25000 [============>.................] - ETA: 8s - loss: 0.5166 - accuracy: 0.8468
11200/25000 [============>.................] - ETA: 8s - loss: 0.5166 - accuracy: 0.8471
11300/25000 [============>.................] - ETA: 8s - loss: 0.5160 - accuracy: 0.8475
11400/25000 [============>.................] - ETA: 8s - loss: 0.5158 - accuracy: 0.8476
11500/25000 [============>.................] - ETA: 8s - loss: 0.5158 - accuracy: 0.8475
11600/25000 [============>.................] - ETA: 8s - loss: 0.5158 - accuracy: 0.8476
11700/25000 [=============>................] - ETA: 8s - loss: 0.5154 - accuracy: 0.8483
11800/25000 [=============>................] - ETA: 8s - loss: 0.5155 - accuracy: 0.8481
11900/25000 [=============>................] - ETA: 8s - loss: 0.5156 - accuracy: 0.8482
12000/25000 [=============>................] - ETA: 8s - loss: 0.5155 - accuracy: 0.8482
12100/25000 [=============>................] - ETA: 7s - loss: 0.5153 - accuracy: 0.8483
12200/25000 [=============>................] - ETA: 7s - loss: 0.5149 - accuracy: 0.8486
12300/25000 [=============>................] - ETA: 7s - loss: 0.5148 - accuracy: 0.8484
12400/25000 [=============>................] - ETA: 7s - loss: 0.5147 - accuracy: 0.8486
12500/25000 [==============>...............] - ETA: 7s - loss: 0.5145 - accuracy: 0.8485
12600/25000 [==============>...............] - ETA: 7s - loss: 0.5145 - accuracy: 0.8484
12700/25000 [==============>...............] - ETA: 7s - loss: 0.5144 - accuracy: 0.8483
12800/25000 [==============>...............] - ETA: 7s - loss: 0.5142 - accuracy: 0.8487
12900/25000 [==============>...............] - ETA: 7s - loss: 0.5140 - accuracy: 0.8489
13000/25000 [==============>...............] - ETA: 7s - loss: 0.5140 - accuracy: 0.8487
13100/25000 [==============>...............] - ETA: 7s - loss: 0.5135 - accuracy: 0.8488
13200/25000 [==============>...............] - ETA: 7s - loss: 0.5134 - accuracy: 0.8488
13300/25000 [==============>...............] - ETA: 7s - loss: 0.5128 - accuracy: 0.8493
13400/25000 [===============>..............] - ETA: 7s - loss: 0.5127 - accuracy: 0.8493
13500/25000 [===============>..............] - ETA: 7s - loss: 0.5123 - accuracy: 0.8496
13600/25000 [===============>..............] - ETA: 7s - loss: 0.5122 - accuracy: 0.8492
13700/25000 [===============>..............] - ETA: 6s - loss: 0.5120 - accuracy: 0.8493
13800/25000 [===============>..............] - ETA: 6s - loss: 0.5119 - accuracy: 0.8494
13900/25000 [===============>..............] - ETA: 6s - loss: 0.5118 - accuracy: 0.8495
14000/25000 [===============>..............] - ETA: 6s - loss: 0.5117 - accuracy: 0.8492
14100/25000 [===============>..............] - ETA: 6s - loss: 0.5121 - accuracy: 0.8491
14200/25000 [================>.............] - ETA: 6s - loss: 0.5120 - accuracy: 0.8492
14300/25000 [================>.............] - ETA: 6s - loss: 0.5119 - accuracy: 0.8490
14400/25000 [================>.............] - ETA: 6s - loss: 0.5115 - accuracy: 0.8497
14500/25000 [================>.............] - ETA: 6s - loss: 0.5113 - accuracy: 0.8499
14600/25000 [================>.............] - ETA: 6s - loss: 0.5114 - accuracy: 0.8501
14700/25000 [================>.............] - ETA: 6s - loss: 0.5111 - accuracy: 0.8501
14800/25000 [================>.............] - ETA: 6s - loss: 0.5110 - accuracy: 0.8504
14900/25000 [================>.............] - ETA: 6s - loss: 0.5109 - accuracy: 0.8504
15000/25000 [=================>............] - ETA: 6s - loss: 0.5106 - accuracy: 0.8507
15100/25000 [=================>............] - ETA: 6s - loss: 0.5102 - accuracy: 0.8509
15200/25000 [=================>............] - ETA: 6s - loss: 0.5099 - accuracy: 0.8507
15300/25000 [=================>............] - ETA: 5s - loss: 0.5098 - accuracy: 0.8511
15400/25000 [=================>............] - ETA: 5s - loss: 0.5098 - accuracy: 0.8512
15500/25000 [=================>............] - ETA: 5s - loss: 0.5094 - accuracy: 0.8516
15600/25000 [=================>............] - ETA: 5s - loss: 0.5092 - accuracy: 0.8517
15700/25000 [=================>............] - ETA: 5s - loss: 0.5093 - accuracy: 0.8516
15800/25000 [=================>............] - ETA: 5s - loss: 0.5092 - accuracy: 0.8517
15900/25000 [==================>...........] - ETA: 5s - loss: 0.5092 - accuracy: 0.8519
16000/25000 [==================>...........] - ETA: 5s - loss: 0.5091 - accuracy: 0.8519
16100/25000 [==================>...........] - ETA: 5s - loss: 0.5087 - accuracy: 0.8522
16200/25000 [==================>...........] - ETA: 5s - loss: 0.5082 - accuracy: 0.8525
16300/25000 [==================>...........] - ETA: 5s - loss: 0.5080 - accuracy: 0.8527
16400/25000 [==================>...........] - ETA: 5s - loss: 0.5077 - accuracy: 0.8527
16500/25000 [==================>...........] - ETA: 5s - loss: 0.5073 - accuracy: 0.8529
16600/25000 [==================>...........] - ETA: 5s - loss: 0.5071 - accuracy: 0.8531
16700/25000 [===================>..........] - ETA: 5s - loss: 0.5070 - accuracy: 0.8531
16800/25000 [===================>..........] - ETA: 5s - loss: 0.5071 - accuracy: 0.8532
16900/25000 [===================>..........] - ETA: 4s - loss: 0.5072 - accuracy: 0.8531
17000/25000 [===================>..........] - ETA: 4s - loss: 0.5070 - accuracy: 0.8532
17100/25000 [===================>..........] - ETA: 4s - loss: 0.5070 - accuracy: 0.8531
17200/25000 [===================>..........] - ETA: 4s - loss: 0.5071 - accuracy: 0.8530
17300/25000 [===================>..........] - ETA: 4s - loss: 0.5067 - accuracy: 0.8534
17400/25000 [===================>..........] - ETA: 4s - loss: 0.5067 - accuracy: 0.8532
17500/25000 [====================>.........] - ETA: 4s - loss: 0.5068 - accuracy: 0.8529
17600/25000 [====================>.........] - ETA: 4s - loss: 0.5066 - accuracy: 0.8532
17700/25000 [====================>.........] - ETA: 4s - loss: 0.5065 - accuracy: 0.8530
17800/25000 [====================>.........] - ETA: 4s - loss: 0.5063 - accuracy: 0.8531
17900/25000 [====================>.........] - ETA: 4s - loss: 0.5062 - accuracy: 0.8530
18000/25000 [====================>.........] - ETA: 4s - loss: 0.5061 - accuracy: 0.8531
18100/25000 [====================>.........] - ETA: 4s - loss: 0.5059 - accuracy: 0.8534
18200/25000 [====================>.........] - ETA: 4s - loss: 0.5057 - accuracy: 0.8534
18300/25000 [====================>.........] - ETA: 4s - loss: 0.5058 - accuracy: 0.8533
18400/25000 [=====================>........] - ETA: 4s - loss: 0.5057 - accuracy: 0.8532
18500/25000 [=====================>........] - ETA: 4s - loss: 0.5057 - accuracy: 0.8532
18600/25000 [=====================>........] - ETA: 3s - loss: 0.5056 - accuracy: 0.8533
18700/25000 [=====================>........] - ETA: 3s - loss: 0.5053 - accuracy: 0.8536
18800/25000 [=====================>........] - ETA: 3s - loss: 0.5051 - accuracy: 0.8539
18900/25000 [=====================>........] - ETA: 3s - loss: 0.5048 - accuracy: 0.8542
19000/25000 [=====================>........] - ETA: 3s - loss: 0.5048 - accuracy: 0.8542
19100/25000 [=====================>........] - ETA: 3s - loss: 0.5045 - accuracy: 0.8543
19200/25000 [======================>.......] - ETA: 3s - loss: 0.5044 - accuracy: 0.8543
19300/25000 [======================>.......] - ETA: 3s - loss: 0.5047 - accuracy: 0.8540
19400/25000 [======================>.......] - ETA: 3s - loss: 0.5043 - accuracy: 0.8543
19500/25000 [======================>.......] - ETA: 3s - loss: 0.5040 - accuracy: 0.8544
19600/25000 [======================>.......] - ETA: 3s - loss: 0.5040 - accuracy: 0.8545
19700/25000 [======================>.......] - ETA: 3s - loss: 0.5041 - accuracy: 0.8544
19800/25000 [======================>.......] - ETA: 3s - loss: 0.5041 - accuracy: 0.8542
19900/25000 [======================>.......] - ETA: 3s - loss: 0.5042 - accuracy: 0.8540
20000/25000 [=======================>......] - ETA: 3s - loss: 0.5038 - accuracy: 0.8542
20100/25000 [=======================>......] - ETA: 3s - loss: 0.5036 - accuracy: 0.8544
20200/25000 [=======================>......] - ETA: 2s - loss: 0.5032 - accuracy: 0.8549
20300/25000 [=======================>......] - ETA: 2s - loss: 0.5029 - accuracy: 0.8551
20400/25000 [=======================>......] - ETA: 2s - loss: 0.5027 - accuracy: 0.8553
20500/25000 [=======================>......] - ETA: 2s - loss: 0.5025 - accuracy: 0.8555
20600/25000 [=======================>......] - ETA: 2s - loss: 0.5022 - accuracy: 0.8558
20700/25000 [=======================>......] - ETA: 2s - loss: 0.5019 - accuracy: 0.8561
20800/25000 [=======================>......] - ETA: 2s - loss: 0.5017 - accuracy: 0.8563
20900/25000 [========================>.....] - ETA: 2s - loss: 0.5014 - accuracy: 0.8565
21000/25000 [========================>.....] - ETA: 2s - loss: 0.5013 - accuracy: 0.8565
21100/25000 [========================>.....] - ETA: 2s - loss: 0.5011 - accuracy: 0.8566
21200/25000 [========================>.....] - ETA: 2s - loss: 0.5007 - accuracy: 0.8568
21300/25000 [========================>.....] - ETA: 2s - loss: 0.5009 - accuracy: 0.8564
21400/25000 [========================>.....] - ETA: 2s - loss: 0.5008 - accuracy: 0.8565
21500/25000 [========================>.....] - ETA: 2s - loss: 0.5006 - accuracy: 0.8568
21600/25000 [========================>.....] - ETA: 2s - loss: 0.5002 - accuracy: 0.8571
21700/25000 [=========================>....] - ETA: 2s - loss: 0.5001 - accuracy: 0.8571
21800/25000 [=========================>....] - ETA: 1s - loss: 0.5001 - accuracy: 0.8571
21900/25000 [=========================>....] - ETA: 1s - loss: 0.5000 - accuracy: 0.8573
22000/25000 [=========================>....] - ETA: 1s - loss: 0.4999 - accuracy: 0.8573
22100/25000 [=========================>....] - ETA: 1s - loss: 0.5000 - accuracy: 0.8571
22200/25000 [=========================>....] - ETA: 1s - loss: 0.4999 - accuracy: 0.8571
22300/25000 [=========================>....] - ETA: 1s - loss: 0.4997 - accuracy: 0.8570
22400/25000 [=========================>....] - ETA: 1s - loss: 0.4995 - accuracy: 0.8572
22500/25000 [==========================>...] - ETA: 1s - loss: 0.4994 - accuracy: 0.8573
22600/25000 [==========================>...] - ETA: 1s - loss: 0.4992 - accuracy: 0.8576
22700/25000 [==========================>...] - ETA: 1s - loss: 0.4992 - accuracy: 0.8574
22800/25000 [==========================>...] - ETA: 1s - loss: 0.4994 - accuracy: 0.8573
22900/25000 [==========================>...] - ETA: 1s - loss: 0.4993 - accuracy: 0.8572
23000/25000 [==========================>...] - ETA: 1s - loss: 0.4991 - accuracy: 0.8574
23100/25000 [==========================>...] - ETA: 1s - loss: 0.4993 - accuracy: 0.8571
23200/25000 [==========================>...] - ETA: 1s - loss: 0.4991 - accuracy: 0.8573
23300/25000 [==========================>...] - ETA: 1s - loss: 0.4988 - accuracy: 0.8576
23400/25000 [===========================>..] - ETA: 0s - loss: 0.4986 - accuracy: 0.8577
23500/25000 [===========================>..] - ETA: 0s - loss: 0.4984 - accuracy: 0.8579
23600/25000 [===========================>..] - ETA: 0s - loss: 0.4983 - accuracy: 0.8578
23700/25000 [===========================>..] - ETA: 0s - loss: 0.4982 - accuracy: 0.8578
23800/25000 [===========================>..] - ETA: 0s - loss: 0.4980 - accuracy: 0.8580
23900/25000 [===========================>..] - ETA: 0s - loss: 0.4980 - accuracy: 0.8580
24000/25000 [===========================>..] - ETA: 0s - loss: 0.4977 - accuracy: 0.8583
24100/25000 [===========================>..] - ETA: 0s - loss: 0.4975 - accuracy: 0.8583
24200/25000 [============================>.] - ETA: 0s - loss: 0.4976 - accuracy: 0.8580
24300/25000 [============================>.] - ETA: 0s - loss: 0.4976 - accuracy: 0.8581
24400/25000 [============================>.] - ETA: 0s - loss: 0.4975 - accuracy: 0.8582
24500/25000 [============================>.] - ETA: 0s - loss: 0.4976 - accuracy: 0.8578
24600/25000 [============================>.] - ETA: 0s - loss: 0.4974 - accuracy: 0.8580
24700/25000 [============================>.] - ETA: 0s - loss: 0.4973 - accuracy: 0.8581
24800/25000 [============================>.] - ETA: 0s - loss: 0.4972 - accuracy: 0.8582
24900/25000 [============================>.] - ETA: 0s - loss: 0.4970 - accuracy: 0.8583
25000/25000 [==============================] - 19s 776us/step - loss: 0.4970 - accuracy: 0.8581 - val_loss: 0.4795 - val_accuracy: 0.8575
Epoch 3/10

  100/25000 [..............................] - ETA: 14s - loss: 0.4313 - accuracy: 0.8800
  200/25000 [..............................] - ETA: 15s - loss: 0.4428 - accuracy: 0.8900
  300/25000 [..............................] - ETA: 14s - loss: 0.4305 - accuracy: 0.9033
  400/25000 [..............................] - ETA: 15s - loss: 0.4182 - accuracy: 0.9125
  500/25000 [..............................] - ETA: 15s - loss: 0.4319 - accuracy: 0.8940
  600/25000 [..............................] - ETA: 15s - loss: 0.4304 - accuracy: 0.9017
  700/25000 [..............................] - ETA: 15s - loss: 0.4290 - accuracy: 0.9000
  800/25000 [..............................] - ETA: 15s - loss: 0.4341 - accuracy: 0.8950
  900/25000 [>.............................] - ETA: 15s - loss: 0.4378 - accuracy: 0.8956
 1000/25000 [>.............................] - ETA: 15s - loss: 0.4375 - accuracy: 0.8960
 1100/25000 [>.............................] - ETA: 14s - loss: 0.4363 - accuracy: 0.8964
 1200/25000 [>.............................] - ETA: 14s - loss: 0.4372 - accuracy: 0.8983
 1300/25000 [>.............................] - ETA: 14s - loss: 0.4325 - accuracy: 0.9031
 1400/25000 [>.............................] - ETA: 14s - loss: 0.4330 - accuracy: 0.9007
 1500/25000 [>.............................] - ETA: 14s - loss: 0.4364 - accuracy: 0.9007
 1600/25000 [>.............................] - ETA: 14s - loss: 0.4372 - accuracy: 0.9019
 1700/25000 [=>............................] - ETA: 14s - loss: 0.4376 - accuracy: 0.9018
 1800/25000 [=>............................] - ETA: 14s - loss: 0.4366 - accuracy: 0.9017
 1900/25000 [=>............................] - ETA: 14s - loss: 0.4381 - accuracy: 0.9016
 2000/25000 [=>............................] - ETA: 14s - loss: 0.4365 - accuracy: 0.9010
 2100/25000 [=>............................] - ETA: 14s - loss: 0.4377 - accuracy: 0.9014
 2200/25000 [=>............................] - ETA: 14s - loss: 0.4370 - accuracy: 0.9023
 2300/25000 [=>............................] - ETA: 14s - loss: 0.4370 - accuracy: 0.9043
 2400/25000 [=>............................] - ETA: 13s - loss: 0.4395 - accuracy: 0.9021
 2500/25000 [==>...........................] - ETA: 13s - loss: 0.4404 - accuracy: 0.9024
 2600/25000 [==>...........................] - ETA: 13s - loss: 0.4406 - accuracy: 0.9038
 2700/25000 [==>...........................] - ETA: 13s - loss: 0.4399 - accuracy: 0.9044
 2800/25000 [==>...........................] - ETA: 13s - loss: 0.4379 - accuracy: 0.9057
 2900/25000 [==>...........................] - ETA: 13s - loss: 0.4387 - accuracy: 0.9048
 3000/25000 [==>...........................] - ETA: 13s - loss: 0.4381 - accuracy: 0.9033
 3100/25000 [==>...........................] - ETA: 13s - loss: 0.4382 - accuracy: 0.9019
 3200/25000 [==>...........................] - ETA: 13s - loss: 0.4368 - accuracy: 0.9025
 3300/25000 [==>...........................] - ETA: 13s - loss: 0.4367 - accuracy: 0.9024
 3400/25000 [===>..........................] - ETA: 13s - loss: 0.4362 - accuracy: 0.9029
 3500/25000 [===>..........................] - ETA: 13s - loss: 0.4363 - accuracy: 0.9029
 3600/25000 [===>..........................] - ETA: 13s - loss: 0.4376 - accuracy: 0.9017
 3700/25000 [===>..........................] - ETA: 13s - loss: 0.4367 - accuracy: 0.9022
 3800/25000 [===>..........................] - ETA: 13s - loss: 0.4365 - accuracy: 0.9013
 3900/25000 [===>..........................] - ETA: 12s - loss: 0.4375 - accuracy: 0.9010
 4000/25000 [===>..........................] - ETA: 12s - loss: 0.4365 - accuracy: 0.9013
 4100/25000 [===>..........................] - ETA: 12s - loss: 0.4368 - accuracy: 0.9012
 4200/25000 [====>.........................] - ETA: 12s - loss: 0.4363 - accuracy: 0.9021
 4300/25000 [====>.........................] - ETA: 12s - loss: 0.4356 - accuracy: 0.9030
 4400/25000 [====>.........................] - ETA: 12s - loss: 0.4363 - accuracy: 0.9018
 4500/25000 [====>.........................] - ETA: 12s - loss: 0.4375 - accuracy: 0.9007
 4600/25000 [====>.........................] - ETA: 12s - loss: 0.4371 - accuracy: 0.9017
 4700/25000 [====>.........................] - ETA: 12s - loss: 0.4358 - accuracy: 0.9019
 4800/25000 [====>.........................] - ETA: 12s - loss: 0.4358 - accuracy: 0.9019
 4900/25000 [====>.........................] - ETA: 12s - loss: 0.4355 - accuracy: 0.9027
 5000/25000 [=====>........................] - ETA: 12s - loss: 0.4357 - accuracy: 0.9028
 5100/25000 [=====>........................] - ETA: 12s - loss: 0.4354 - accuracy: 0.9027
 5200/25000 [=====>........................] - ETA: 12s - loss: 0.4361 - accuracy: 0.9027
 5300/25000 [=====>........................] - ETA: 12s - loss: 0.4358 - accuracy: 0.9036
 5400/25000 [=====>........................] - ETA: 12s - loss: 0.4361 - accuracy: 0.9031
 5500/25000 [=====>........................] - ETA: 12s - loss: 0.4364 - accuracy: 0.9025
 5600/25000 [=====>........................] - ETA: 11s - loss: 0.4360 - accuracy: 0.9032
 5700/25000 [=====>........................] - ETA: 11s - loss: 0.4366 - accuracy: 0.9023
 5800/25000 [=====>........................] - ETA: 11s - loss: 0.4366 - accuracy: 0.9029
 5900/25000 [======>.......................] - ETA: 11s - loss: 0.4364 - accuracy: 0.9032
 6000/25000 [======>.......................] - ETA: 11s - loss: 0.4367 - accuracy: 0.9023
 6100/25000 [======>.......................] - ETA: 11s - loss: 0.4364 - accuracy: 0.9025
 6200/25000 [======>.......................] - ETA: 11s - loss: 0.4365 - accuracy: 0.9027
 6300/25000 [======>.......................] - ETA: 11s - loss: 0.4362 - accuracy: 0.9029
 6400/25000 [======>.......................] - ETA: 11s - loss: 0.4361 - accuracy: 0.9030
 6500/25000 [======>.......................] - ETA: 11s - loss: 0.4355 - accuracy: 0.9037
 6600/25000 [======>.......................] - ETA: 11s - loss: 0.4351 - accuracy: 0.9042
 6700/25000 [=======>......................] - ETA: 11s - loss: 0.4349 - accuracy: 0.9037
 6800/25000 [=======>......................] - ETA: 11s - loss: 0.4346 - accuracy: 0.9035
 6900/25000 [=======>......................] - ETA: 11s - loss: 0.4345 - accuracy: 0.9039
 7000/25000 [=======>......................] - ETA: 11s - loss: 0.4344 - accuracy: 0.9041
 7100/25000 [=======>......................] - ETA: 11s - loss: 0.4338 - accuracy: 0.9045
 7200/25000 [=======>......................] - ETA: 10s - loss: 0.4340 - accuracy: 0.9047
 7300/25000 [=======>......................] - ETA: 10s - loss: 0.4345 - accuracy: 0.9037
 7400/25000 [=======>......................] - ETA: 10s - loss: 0.4354 - accuracy: 0.9024
 7500/25000 [========>.....................] - ETA: 10s - loss: 0.4351 - accuracy: 0.9024
 7600/25000 [========>.....................] - ETA: 10s - loss: 0.4350 - accuracy: 0.9025
 7700/25000 [========>.....................] - ETA: 10s - loss: 0.4352 - accuracy: 0.9023
 7800/25000 [========>.....................] - ETA: 10s - loss: 0.4354 - accuracy: 0.9019
 7900/25000 [========>.....................] - ETA: 10s - loss: 0.4351 - accuracy: 0.9023
 8000/25000 [========>.....................] - ETA: 10s - loss: 0.4348 - accuracy: 0.9024
 8100/25000 [========>.....................] - ETA: 10s - loss: 0.4348 - accuracy: 0.9023
 8200/25000 [========>.....................] - ETA: 10s - loss: 0.4354 - accuracy: 0.9017
 8300/25000 [========>.....................] - ETA: 10s - loss: 0.4360 - accuracy: 0.9010
 8400/25000 [=========>....................] - ETA: 10s - loss: 0.4363 - accuracy: 0.9004
 8500/25000 [=========>....................] - ETA: 10s - loss: 0.4365 - accuracy: 0.9001
 8600/25000 [=========>....................] - ETA: 10s - loss: 0.4362 - accuracy: 0.9006
 8700/25000 [=========>....................] - ETA: 10s - loss: 0.4361 - accuracy: 0.9007
 8800/25000 [=========>....................] - ETA: 10s - loss: 0.4363 - accuracy: 0.9003
 8900/25000 [=========>....................] - ETA: 9s - loss: 0.4369 - accuracy: 0.8996 
 9000/25000 [=========>....................] - ETA: 9s - loss: 0.4369 - accuracy: 0.8994
 9100/25000 [=========>....................] - ETA: 9s - loss: 0.4369 - accuracy: 0.8995
 9200/25000 [==========>...................] - ETA: 9s - loss: 0.4368 - accuracy: 0.8992
 9300/25000 [==========>...................] - ETA: 9s - loss: 0.4364 - accuracy: 0.8996
 9400/25000 [==========>...................] - ETA: 9s - loss: 0.4369 - accuracy: 0.8993
 9500/25000 [==========>...................] - ETA: 9s - loss: 0.4369 - accuracy: 0.8994
 9600/25000 [==========>...................] - ETA: 9s - loss: 0.4368 - accuracy: 0.8995
 9700/25000 [==========>...................] - ETA: 9s - loss: 0.4367 - accuracy: 0.8993
 9800/25000 [==========>...................] - ETA: 9s - loss: 0.4362 - accuracy: 0.8994
 9900/25000 [==========>...................] - ETA: 9s - loss: 0.4356 - accuracy: 0.9001
10000/25000 [===========>..................] - ETA: 9s - loss: 0.4355 - accuracy: 0.9002
10100/25000 [===========>..................] - ETA: 9s - loss: 0.4352 - accuracy: 0.9003
10200/25000 [===========>..................] - ETA: 9s - loss: 0.4352 - accuracy: 0.9005
10300/25000 [===========>..................] - ETA: 9s - loss: 0.4350 - accuracy: 0.9006
10400/25000 [===========>..................] - ETA: 9s - loss: 0.4352 - accuracy: 0.9004
10500/25000 [===========>..................] - ETA: 8s - loss: 0.4354 - accuracy: 0.9004
10600/25000 [===========>..................] - ETA: 8s - loss: 0.4351 - accuracy: 0.9003
10700/25000 [===========>..................] - ETA: 8s - loss: 0.4347 - accuracy: 0.9006
10800/25000 [===========>..................] - ETA: 8s - loss: 0.4348 - accuracy: 0.9004
10900/25000 [============>.................] - ETA: 8s - loss: 0.4344 - accuracy: 0.9007
11000/25000 [============>.................] - ETA: 8s - loss: 0.4338 - accuracy: 0.9011
11100/25000 [============>.................] - ETA: 8s - loss: 0.4336 - accuracy: 0.9011
11200/25000 [============>.................] - ETA: 8s - loss: 0.4338 - accuracy: 0.9009
11300/25000 [============>.................] - ETA: 8s - loss: 0.4337 - accuracy: 0.9009
11400/25000 [============>.................] - ETA: 8s - loss: 0.4332 - accuracy: 0.9011
11500/25000 [============>.................] - ETA: 8s - loss: 0.4330 - accuracy: 0.9011
11600/25000 [============>.................] - ETA: 8s - loss: 0.4333 - accuracy: 0.9009
11700/25000 [=============>................] - ETA: 8s - loss: 0.4331 - accuracy: 0.9011
11800/25000 [=============>................] - ETA: 8s - loss: 0.4337 - accuracy: 0.9003
11900/25000 [=============>................] - ETA: 8s - loss: 0.4336 - accuracy: 0.9000
12000/25000 [=============>................] - ETA: 8s - loss: 0.4337 - accuracy: 0.8998
12100/25000 [=============>................] - ETA: 7s - loss: 0.4333 - accuracy: 0.9000
12200/25000 [=============>................] - ETA: 7s - loss: 0.4331 - accuracy: 0.9001
12300/25000 [=============>................] - ETA: 7s - loss: 0.4329 - accuracy: 0.9004
12400/25000 [=============>................] - ETA: 7s - loss: 0.4327 - accuracy: 0.9005
12500/25000 [==============>...............] - ETA: 7s - loss: 0.4328 - accuracy: 0.9003
12600/25000 [==============>...............] - ETA: 7s - loss: 0.4329 - accuracy: 0.9004
12700/25000 [==============>...............] - ETA: 7s - loss: 0.4331 - accuracy: 0.9001
12800/25000 [==============>...............] - ETA: 7s - loss: 0.4326 - accuracy: 0.9004
12900/25000 [==============>...............] - ETA: 7s - loss: 0.4325 - accuracy: 0.9005
13000/25000 [==============>...............] - ETA: 7s - loss: 0.4324 - accuracy: 0.9005
13100/25000 [==============>...............] - ETA: 7s - loss: 0.4326 - accuracy: 0.9003
13200/25000 [==============>...............] - ETA: 7s - loss: 0.4324 - accuracy: 0.9005
13300/25000 [==============>...............] - ETA: 7s - loss: 0.4324 - accuracy: 0.9005
13400/25000 [===============>..............] - ETA: 7s - loss: 0.4323 - accuracy: 0.9004
13500/25000 [===============>..............] - ETA: 7s - loss: 0.4321 - accuracy: 0.9005
13600/25000 [===============>..............] - ETA: 7s - loss: 0.4321 - accuracy: 0.9006
13700/25000 [===============>..............] - ETA: 6s - loss: 0.4324 - accuracy: 0.9001
13800/25000 [===============>..............] - ETA: 6s - loss: 0.4321 - accuracy: 0.9001
13900/25000 [===============>..............] - ETA: 6s - loss: 0.4325 - accuracy: 0.8997
14000/25000 [===============>..............] - ETA: 6s - loss: 0.4324 - accuracy: 0.8996
14100/25000 [===============>..............] - ETA: 6s - loss: 0.4322 - accuracy: 0.8998
14200/25000 [================>.............] - ETA: 6s - loss: 0.4320 - accuracy: 0.9001
14300/25000 [================>.............] - ETA: 6s - loss: 0.4316 - accuracy: 0.9006
14400/25000 [================>.............] - ETA: 6s - loss: 0.4313 - accuracy: 0.9008
14500/25000 [================>.............] - ETA: 6s - loss: 0.4311 - accuracy: 0.9010
14600/25000 [================>.............] - ETA: 6s - loss: 0.4308 - accuracy: 0.9012
14700/25000 [================>.............] - ETA: 6s - loss: 0.4306 - accuracy: 0.9012
14800/25000 [================>.............] - ETA: 6s - loss: 0.4307 - accuracy: 0.9010
14900/25000 [================>.............] - ETA: 6s - loss: 0.4305 - accuracy: 0.9009
15000/25000 [=================>............] - ETA: 6s - loss: 0.4306 - accuracy: 0.9009
15100/25000 [=================>............] - ETA: 6s - loss: 0.4304 - accuracy: 0.9011
15200/25000 [=================>............] - ETA: 6s - loss: 0.4303 - accuracy: 0.9012
15300/25000 [=================>............] - ETA: 5s - loss: 0.4305 - accuracy: 0.9008
15400/25000 [=================>............] - ETA: 5s - loss: 0.4303 - accuracy: 0.9008
15500/25000 [=================>............] - ETA: 5s - loss: 0.4299 - accuracy: 0.9012
15600/25000 [=================>............] - ETA: 5s - loss: 0.4298 - accuracy: 0.9013
15700/25000 [=================>............] - ETA: 5s - loss: 0.4301 - accuracy: 0.9011
15800/25000 [=================>............] - ETA: 5s - loss: 0.4299 - accuracy: 0.9011
15900/25000 [==================>...........] - ETA: 5s - loss: 0.4297 - accuracy: 0.9014
16000/25000 [==================>...........] - ETA: 5s - loss: 0.4297 - accuracy: 0.9014
16100/25000 [==================>...........] - ETA: 5s - loss: 0.4297 - accuracy: 0.9013
16200/25000 [==================>...........] - ETA: 5s - loss: 0.4298 - accuracy: 0.9011
16300/25000 [==================>...........] - ETA: 5s - loss: 0.4299 - accuracy: 0.9010
16400/25000 [==================>...........] - ETA: 5s - loss: 0.4295 - accuracy: 0.9012
16500/25000 [==================>...........] - ETA: 5s - loss: 0.4297 - accuracy: 0.9012
16600/25000 [==================>...........] - ETA: 5s - loss: 0.4295 - accuracy: 0.9012
16700/25000 [===================>..........] - ETA: 5s - loss: 0.4294 - accuracy: 0.9013
16800/25000 [===================>..........] - ETA: 5s - loss: 0.4293 - accuracy: 0.9014
16900/25000 [===================>..........] - ETA: 5s - loss: 0.4296 - accuracy: 0.9011
17000/25000 [===================>..........] - ETA: 4s - loss: 0.4295 - accuracy: 0.9011
17100/25000 [===================>..........] - ETA: 4s - loss: 0.4293 - accuracy: 0.9012
17200/25000 [===================>..........] - ETA: 4s - loss: 0.4291 - accuracy: 0.9015
17300/25000 [===================>..........] - ETA: 4s - loss: 0.4290 - accuracy: 0.9014
17400/25000 [===================>..........] - ETA: 4s - loss: 0.4291 - accuracy: 0.9011
17500/25000 [====================>.........] - ETA: 4s - loss: 0.4292 - accuracy: 0.9010
17600/25000 [====================>.........] - ETA: 4s - loss: 0.4293 - accuracy: 0.9008
17700/25000 [====================>.........] - ETA: 4s - loss: 0.4293 - accuracy: 0.9006
17800/25000 [====================>.........] - ETA: 4s - loss: 0.4293 - accuracy: 0.9004
17900/25000 [====================>.........] - ETA: 4s - loss: 0.4293 - accuracy: 0.9004
18000/25000 [====================>.........] - ETA: 4s - loss: 0.4294 - accuracy: 0.9004
18100/25000 [====================>.........] - ETA: 4s - loss: 0.4295 - accuracy: 0.9001
18200/25000 [====================>.........] - ETA: 4s - loss: 0.4296 - accuracy: 0.8998
18300/25000 [====================>.........] - ETA: 4s - loss: 0.4294 - accuracy: 0.8998
18400/25000 [=====================>........] - ETA: 4s - loss: 0.4295 - accuracy: 0.8998
18500/25000 [=====================>........] - ETA: 4s - loss: 0.4292 - accuracy: 0.8999
18600/25000 [=====================>........] - ETA: 3s - loss: 0.4294 - accuracy: 0.8997
18700/25000 [=====================>........] - ETA: 3s - loss: 0.4292 - accuracy: 0.8999
18800/25000 [=====================>........] - ETA: 3s - loss: 0.4291 - accuracy: 0.8998
18900/25000 [=====================>........] - ETA: 3s - loss: 0.4291 - accuracy: 0.8997
19000/25000 [=====================>........] - ETA: 3s - loss: 0.4292 - accuracy: 0.8994
19100/25000 [=====================>........] - ETA: 3s - loss: 0.4289 - accuracy: 0.8996
19200/25000 [======================>.......] - ETA: 3s - loss: 0.4287 - accuracy: 0.8997
19300/25000 [======================>.......] - ETA: 3s - loss: 0.4284 - accuracy: 0.8999
19400/25000 [======================>.......] - ETA: 3s - loss: 0.4284 - accuracy: 0.8999
19500/25000 [======================>.......] - ETA: 3s - loss: 0.4283 - accuracy: 0.9001
19600/25000 [======================>.......] - ETA: 3s - loss: 0.4284 - accuracy: 0.8999
19700/25000 [======================>.......] - ETA: 3s - loss: 0.4284 - accuracy: 0.8999
19800/25000 [======================>.......] - ETA: 3s - loss: 0.4282 - accuracy: 0.9001
19900/25000 [======================>.......] - ETA: 3s - loss: 0.4281 - accuracy: 0.9002
20000/25000 [=======================>......] - ETA: 3s - loss: 0.4281 - accuracy: 0.8999
20100/25000 [=======================>......] - ETA: 3s - loss: 0.4278 - accuracy: 0.9000
20200/25000 [=======================>......] - ETA: 2s - loss: 0.4277 - accuracy: 0.9000
20300/25000 [=======================>......] - ETA: 2s - loss: 0.4277 - accuracy: 0.9000
20400/25000 [=======================>......] - ETA: 2s - loss: 0.4276 - accuracy: 0.9000
20500/25000 [=======================>......] - ETA: 2s - loss: 0.4278 - accuracy: 0.8998
20600/25000 [=======================>......] - ETA: 2s - loss: 0.4279 - accuracy: 0.8997
20700/25000 [=======================>......] - ETA: 2s - loss: 0.4278 - accuracy: 0.8998
20800/25000 [=======================>......] - ETA: 2s - loss: 0.4277 - accuracy: 0.8998
20900/25000 [========================>.....] - ETA: 2s - loss: 0.4275 - accuracy: 0.8999
21000/25000 [========================>.....] - ETA: 2s - loss: 0.4272 - accuracy: 0.9000
21100/25000 [========================>.....] - ETA: 2s - loss: 0.4272 - accuracy: 0.9000
21200/25000 [========================>.....] - ETA: 2s - loss: 0.4271 - accuracy: 0.9000
21300/25000 [========================>.....] - ETA: 2s - loss: 0.4269 - accuracy: 0.9000
21400/25000 [========================>.....] - ETA: 2s - loss: 0.4268 - accuracy: 0.8999
21500/25000 [========================>.....] - ETA: 2s - loss: 0.4269 - accuracy: 0.8998
21600/25000 [========================>.....] - ETA: 2s - loss: 0.4268 - accuracy: 0.8999
21700/25000 [=========================>....] - ETA: 2s - loss: 0.4268 - accuracy: 0.9000
21800/25000 [=========================>....] - ETA: 1s - loss: 0.4271 - accuracy: 0.8995
21900/25000 [=========================>....] - ETA: 1s - loss: 0.4277 - accuracy: 0.8990
22000/25000 [=========================>....] - ETA: 1s - loss: 0.4277 - accuracy: 0.8990
22100/25000 [=========================>....] - ETA: 1s - loss: 0.4277 - accuracy: 0.8988
22200/25000 [=========================>....] - ETA: 1s - loss: 0.4274 - accuracy: 0.8990
22300/25000 [=========================>....] - ETA: 1s - loss: 0.4275 - accuracy: 0.8989
22400/25000 [=========================>....] - ETA: 1s - loss: 0.4273 - accuracy: 0.8989
22500/25000 [==========================>...] - ETA: 1s - loss: 0.4273 - accuracy: 0.8988
22600/25000 [==========================>...] - ETA: 1s - loss: 0.4274 - accuracy: 0.8986
22700/25000 [==========================>...] - ETA: 1s - loss: 0.4275 - accuracy: 0.8985
22800/25000 [==========================>...] - ETA: 1s - loss: 0.4274 - accuracy: 0.8985
22900/25000 [==========================>...] - ETA: 1s - loss: 0.4274 - accuracy: 0.8984
23000/25000 [==========================>...] - ETA: 1s - loss: 0.4275 - accuracy: 0.8982
23100/25000 [==========================>...] - ETA: 1s - loss: 0.4273 - accuracy: 0.8983
23200/25000 [==========================>...] - ETA: 1s - loss: 0.4273 - accuracy: 0.8984
23300/25000 [==========================>...] - ETA: 1s - loss: 0.4273 - accuracy: 0.8982
23400/25000 [===========================>..] - ETA: 0s - loss: 0.4274 - accuracy: 0.8981
23500/25000 [===========================>..] - ETA: 0s - loss: 0.4273 - accuracy: 0.8981
23600/25000 [===========================>..] - ETA: 0s - loss: 0.4271 - accuracy: 0.8983
23700/25000 [===========================>..] - ETA: 0s - loss: 0.4270 - accuracy: 0.8984
23800/25000 [===========================>..] - ETA: 0s - loss: 0.4270 - accuracy: 0.8984
23900/25000 [===========================>..] - ETA: 0s - loss: 0.4268 - accuracy: 0.8984
24000/25000 [===========================>..] - ETA: 0s - loss: 0.4269 - accuracy: 0.8982
24100/25000 [===========================>..] - ETA: 0s - loss: 0.4269 - accuracy: 0.8982
24200/25000 [============================>.] - ETA: 0s - loss: 0.4268 - accuracy: 0.8982
24300/25000 [============================>.] - ETA: 0s - loss: 0.4269 - accuracy: 0.8982
24400/25000 [============================>.] - ETA: 0s - loss: 0.4268 - accuracy: 0.8982
24500/25000 [============================>.] - ETA: 0s - loss: 0.4270 - accuracy: 0.8979
24600/25000 [============================>.] - ETA: 0s - loss: 0.4268 - accuracy: 0.8980
24700/25000 [============================>.] - ETA: 0s - loss: 0.4267 - accuracy: 0.8979
24800/25000 [============================>.] - ETA: 0s - loss: 0.4266 - accuracy: 0.8979
24900/25000 [============================>.] - ETA: 0s - loss: 0.4267 - accuracy: 0.8978
25000/25000 [==============================] - 19s 777us/step - loss: 0.4266 - accuracy: 0.8978 - val_loss: 0.4472 - val_accuracy: 0.8605
Epoch 4/10

  100/25000 [..............................] - ETA: 15s - loss: 0.3935 - accuracy: 0.9100
  200/25000 [..............................] - ETA: 15s - loss: 0.3885 - accuracy: 0.9200
  300/25000 [..............................] - ETA: 14s - loss: 0.3790 - accuracy: 0.9267
  400/25000 [..............................] - ETA: 15s - loss: 0.3718 - accuracy: 0.9350
  500/25000 [..............................] - ETA: 15s - loss: 0.3705 - accuracy: 0.9360
  600/25000 [..............................] - ETA: 15s - loss: 0.3841 - accuracy: 0.9200
  700/25000 [..............................] - ETA: 15s - loss: 0.3850 - accuracy: 0.9186
  800/25000 [..............................] - ETA: 15s - loss: 0.3849 - accuracy: 0.9212
  900/25000 [>.............................] - ETA: 15s - loss: 0.3844 - accuracy: 0.9256
 1000/25000 [>.............................] - ETA: 14s - loss: 0.3823 - accuracy: 0.9270
 1100/25000 [>.............................] - ETA: 14s - loss: 0.3825 - accuracy: 0.9264
 1200/25000 [>.............................] - ETA: 14s - loss: 0.3808 - accuracy: 0.9267
 1300/25000 [>.............................] - ETA: 14s - loss: 0.3763 - accuracy: 0.9300
 1400/25000 [>.............................] - ETA: 14s - loss: 0.3782 - accuracy: 0.9264
 1500/25000 [>.............................] - ETA: 14s - loss: 0.3809 - accuracy: 0.9260
 1600/25000 [>.............................] - ETA: 14s - loss: 0.3798 - accuracy: 0.9281
 1700/25000 [=>............................] - ETA: 14s - loss: 0.3785 - accuracy: 0.9294
 1800/25000 [=>............................] - ETA: 14s - loss: 0.3779 - accuracy: 0.9306
 1900/25000 [=>............................] - ETA: 14s - loss: 0.3761 - accuracy: 0.9316
 2000/25000 [=>............................] - ETA: 14s - loss: 0.3775 - accuracy: 0.9315
 2100/25000 [=>............................] - ETA: 14s - loss: 0.3762 - accuracy: 0.9314
 2200/25000 [=>............................] - ETA: 14s - loss: 0.3770 - accuracy: 0.9309
 2300/25000 [=>............................] - ETA: 14s - loss: 0.3772 - accuracy: 0.9313
 2400/25000 [=>............................] - ETA: 14s - loss: 0.3773 - accuracy: 0.9317
 2500/25000 [==>...........................] - ETA: 14s - loss: 0.3778 - accuracy: 0.9304
 2600/25000 [==>...........................] - ETA: 13s - loss: 0.3773 - accuracy: 0.9308
 2700/25000 [==>...........................] - ETA: 13s - loss: 0.3769 - accuracy: 0.9311
 2800/25000 [==>...........................] - ETA: 13s - loss: 0.3786 - accuracy: 0.9289
 2900/25000 [==>...........................] - ETA: 13s - loss: 0.3779 - accuracy: 0.9300
 3000/25000 [==>...........................] - ETA: 13s - loss: 0.3770 - accuracy: 0.9300
 3100/25000 [==>...........................] - ETA: 13s - loss: 0.3780 - accuracy: 0.9284
 3200/25000 [==>...........................] - ETA: 13s - loss: 0.3789 - accuracy: 0.9281
 3300/25000 [==>...........................] - ETA: 13s - loss: 0.3798 - accuracy: 0.9270
 3400/25000 [===>..........................] - ETA: 13s - loss: 0.3801 - accuracy: 0.9265
 3500/25000 [===>..........................] - ETA: 13s - loss: 0.3808 - accuracy: 0.9266
 3600/25000 [===>..........................] - ETA: 13s - loss: 0.3807 - accuracy: 0.9261
 3700/25000 [===>..........................] - ETA: 13s - loss: 0.3799 - accuracy: 0.9268
 3800/25000 [===>..........................] - ETA: 13s - loss: 0.3802 - accuracy: 0.9261
 3900/25000 [===>..........................] - ETA: 13s - loss: 0.3810 - accuracy: 0.9246
 4000/25000 [===>..........................] - ETA: 13s - loss: 0.3813 - accuracy: 0.9250
 4100/25000 [===>..........................] - ETA: 12s - loss: 0.3829 - accuracy: 0.9239
 4200/25000 [====>.........................] - ETA: 12s - loss: 0.3827 - accuracy: 0.9245
 4300/25000 [====>.........................] - ETA: 12s - loss: 0.3823 - accuracy: 0.9253
 4400/25000 [====>.........................] - ETA: 12s - loss: 0.3828 - accuracy: 0.9245
 4500/25000 [====>.........................] - ETA: 12s - loss: 0.3837 - accuracy: 0.9242
 4600/25000 [====>.........................] - ETA: 12s - loss: 0.3837 - accuracy: 0.9239
 4700/25000 [====>.........................] - ETA: 12s - loss: 0.3842 - accuracy: 0.9234
 4800/25000 [====>.........................] - ETA: 12s - loss: 0.3844 - accuracy: 0.9229
 4900/25000 [====>.........................] - ETA: 12s - loss: 0.3847 - accuracy: 0.9224
 5000/25000 [=====>........................] - ETA: 12s - loss: 0.3864 - accuracy: 0.9210
 5100/25000 [=====>........................] - ETA: 12s - loss: 0.3861 - accuracy: 0.9208
 5200/25000 [=====>........................] - ETA: 12s - loss: 0.3857 - accuracy: 0.9208
 5300/25000 [=====>........................] - ETA: 12s - loss: 0.3850 - accuracy: 0.9211
 5400/25000 [=====>........................] - ETA: 12s - loss: 0.3848 - accuracy: 0.9217
 5500/25000 [=====>........................] - ETA: 12s - loss: 0.3851 - accuracy: 0.9213
 5600/25000 [=====>........................] - ETA: 11s - loss: 0.3855 - accuracy: 0.9207
 5700/25000 [=====>........................] - ETA: 11s - loss: 0.3863 - accuracy: 0.9200
 5800/25000 [=====>........................] - ETA: 11s - loss: 0.3866 - accuracy: 0.9195
 5900/25000 [======>.......................] - ETA: 11s - loss: 0.3860 - accuracy: 0.9200
 6000/25000 [======>.......................] - ETA: 11s - loss: 0.3853 - accuracy: 0.9207
 6100/25000 [======>.......................] - ETA: 11s - loss: 0.3846 - accuracy: 0.9215
 6200/25000 [======>.......................] - ETA: 11s - loss: 0.3852 - accuracy: 0.9206
 6300/25000 [======>.......................] - ETA: 11s - loss: 0.3848 - accuracy: 0.9208
 6400/25000 [======>.......................] - ETA: 11s - loss: 0.3847 - accuracy: 0.9206
 6500/25000 [======>.......................] - ETA: 11s - loss: 0.3849 - accuracy: 0.9203
 6600/25000 [======>.......................] - ETA: 11s - loss: 0.3847 - accuracy: 0.9205
 6700/25000 [=======>......................] - ETA: 11s - loss: 0.3842 - accuracy: 0.9204
 6800/25000 [=======>......................] - ETA: 11s - loss: 0.3840 - accuracy: 0.9206
 6900/25000 [=======>......................] - ETA: 11s - loss: 0.3839 - accuracy: 0.9204
 7000/25000 [=======>......................] - ETA: 11s - loss: 0.3846 - accuracy: 0.9197
 7100/25000 [=======>......................] - ETA: 11s - loss: 0.3847 - accuracy: 0.9193
 7200/25000 [=======>......................] - ETA: 10s - loss: 0.3845 - accuracy: 0.9190
 7300/25000 [=======>......................] - ETA: 10s - loss: 0.3844 - accuracy: 0.9190
 7400/25000 [=======>......................] - ETA: 10s - loss: 0.3846 - accuracy: 0.9185
 7500/25000 [========>.....................] - ETA: 10s - loss: 0.3844 - accuracy: 0.9187
 7600/25000 [========>.....................] - ETA: 10s - loss: 0.3844 - accuracy: 0.9189
 7700/25000 [========>.....................] - ETA: 10s - loss: 0.3845 - accuracy: 0.9187
 7800/25000 [========>.....................] - ETA: 10s - loss: 0.3845 - accuracy: 0.9187
 7900/25000 [========>.....................] - ETA: 10s - loss: 0.3848 - accuracy: 0.9182
 8000/25000 [========>.....................] - ETA: 10s - loss: 0.3850 - accuracy: 0.9183
 8100/25000 [========>.....................] - ETA: 10s - loss: 0.3845 - accuracy: 0.9186
 8200/25000 [========>.....................] - ETA: 10s - loss: 0.3840 - accuracy: 0.9191
 8300/25000 [========>.....................] - ETA: 10s - loss: 0.3833 - accuracy: 0.9199
 8400/25000 [=========>....................] - ETA: 10s - loss: 0.3826 - accuracy: 0.9204
 8500/25000 [=========>....................] - ETA: 10s - loss: 0.3826 - accuracy: 0.9204
 8600/25000 [=========>....................] - ETA: 10s - loss: 0.3825 - accuracy: 0.9205
 8700/25000 [=========>....................] - ETA: 10s - loss: 0.3824 - accuracy: 0.9205
 8800/25000 [=========>....................] - ETA: 9s - loss: 0.3830 - accuracy: 0.9197 
 8900/25000 [=========>....................] - ETA: 9s - loss: 0.3832 - accuracy: 0.9196
 9000/25000 [=========>....................] - ETA: 9s - loss: 0.3832 - accuracy: 0.9196
 9100/25000 [=========>....................] - ETA: 9s - loss: 0.3834 - accuracy: 0.9192
 9200/25000 [==========>...................] - ETA: 9s - loss: 0.3831 - accuracy: 0.9193
 9300/25000 [==========>...................] - ETA: 9s - loss: 0.3830 - accuracy: 0.9196
 9400/25000 [==========>...................] - ETA: 9s - loss: 0.3828 - accuracy: 0.9197
 9500/25000 [==========>...................] - ETA: 9s - loss: 0.3829 - accuracy: 0.9199
 9600/25000 [==========>...................] - ETA: 9s - loss: 0.3827 - accuracy: 0.9199
 9700/25000 [==========>...................] - ETA: 9s - loss: 0.3827 - accuracy: 0.9198
 9800/25000 [==========>...................] - ETA: 9s - loss: 0.3824 - accuracy: 0.9200
 9900/25000 [==========>...................] - ETA: 9s - loss: 0.3825 - accuracy: 0.9200
10000/25000 [===========>..................] - ETA: 9s - loss: 0.3825 - accuracy: 0.9200
10100/25000 [===========>..................] - ETA: 9s - loss: 0.3824 - accuracy: 0.9200
10200/25000 [===========>..................] - ETA: 9s - loss: 0.3827 - accuracy: 0.9198
10300/25000 [===========>..................] - ETA: 9s - loss: 0.3828 - accuracy: 0.9195
10400/25000 [===========>..................] - ETA: 8s - loss: 0.3827 - accuracy: 0.9196
10500/25000 [===========>..................] - ETA: 8s - loss: 0.3827 - accuracy: 0.9195
10600/25000 [===========>..................] - ETA: 8s - loss: 0.3830 - accuracy: 0.9192
10700/25000 [===========>..................] - ETA: 8s - loss: 0.3828 - accuracy: 0.9194
10800/25000 [===========>..................] - ETA: 8s - loss: 0.3833 - accuracy: 0.9191
10900/25000 [============>.................] - ETA: 8s - loss: 0.3831 - accuracy: 0.9191
11000/25000 [============>.................] - ETA: 8s - loss: 0.3827 - accuracy: 0.9194
11100/25000 [============>.................] - ETA: 8s - loss: 0.3822 - accuracy: 0.9196
11200/25000 [============>.................] - ETA: 8s - loss: 0.3822 - accuracy: 0.9195
11300/25000 [============>.................] - ETA: 8s - loss: 0.3819 - accuracy: 0.9197
11400/25000 [============>.................] - ETA: 8s - loss: 0.3818 - accuracy: 0.9199
11500/25000 [============>.................] - ETA: 8s - loss: 0.3818 - accuracy: 0.9199
11600/25000 [============>.................] - ETA: 8s - loss: 0.3820 - accuracy: 0.9197
11700/25000 [=============>................] - ETA: 8s - loss: 0.3821 - accuracy: 0.9197
11800/25000 [=============>................] - ETA: 8s - loss: 0.3818 - accuracy: 0.9199
11900/25000 [=============>................] - ETA: 8s - loss: 0.3816 - accuracy: 0.9199
12000/25000 [=============>................] - ETA: 8s - loss: 0.3814 - accuracy: 0.9200
12100/25000 [=============>................] - ETA: 7s - loss: 0.3815 - accuracy: 0.9199
12200/25000 [=============>................] - ETA: 7s - loss: 0.3815 - accuracy: 0.9201
12300/25000 [=============>................] - ETA: 7s - loss: 0.3814 - accuracy: 0.9201
12400/25000 [=============>................] - ETA: 7s - loss: 0.3812 - accuracy: 0.9203
12500/25000 [==============>...............] - ETA: 7s - loss: 0.3814 - accuracy: 0.9201
12600/25000 [==============>...............] - ETA: 7s - loss: 0.3813 - accuracy: 0.9202
12700/25000 [==============>...............] - ETA: 7s - loss: 0.3818 - accuracy: 0.9198
12800/25000 [==============>...............] - ETA: 7s - loss: 0.3817 - accuracy: 0.9199
12900/25000 [==============>...............] - ETA: 7s - loss: 0.3818 - accuracy: 0.9200
13000/25000 [==============>...............] - ETA: 7s - loss: 0.3813 - accuracy: 0.9203
13100/25000 [==============>...............] - ETA: 7s - loss: 0.3813 - accuracy: 0.9202
13200/25000 [==============>...............] - ETA: 7s - loss: 0.3815 - accuracy: 0.9201
13300/25000 [==============>...............] - ETA: 7s - loss: 0.3816 - accuracy: 0.9198
13400/25000 [===============>..............] - ETA: 7s - loss: 0.3815 - accuracy: 0.9200
13500/25000 [===============>..............] - ETA: 7s - loss: 0.3815 - accuracy: 0.9200
13600/25000 [===============>..............] - ETA: 7s - loss: 0.3816 - accuracy: 0.9198
13700/25000 [===============>..............] - ETA: 6s - loss: 0.3815 - accuracy: 0.9198
13800/25000 [===============>..............] - ETA: 6s - loss: 0.3812 - accuracy: 0.9200
13900/25000 [===============>..............] - ETA: 6s - loss: 0.3810 - accuracy: 0.9201
14000/25000 [===============>..............] - ETA: 6s - loss: 0.3810 - accuracy: 0.9201
14100/25000 [===============>..............] - ETA: 6s - loss: 0.3808 - accuracy: 0.9203
14200/25000 [================>.............] - ETA: 6s - loss: 0.3806 - accuracy: 0.9205
14300/25000 [================>.............] - ETA: 6s - loss: 0.3802 - accuracy: 0.9208
14400/25000 [================>.............] - ETA: 6s - loss: 0.3802 - accuracy: 0.9208
14500/25000 [================>.............] - ETA: 6s - loss: 0.3799 - accuracy: 0.9208
14600/25000 [================>.............] - ETA: 6s - loss: 0.3798 - accuracy: 0.9208
14700/25000 [================>.............] - ETA: 6s - loss: 0.3798 - accuracy: 0.9206
14800/25000 [================>.............] - ETA: 6s - loss: 0.3795 - accuracy: 0.9209
14900/25000 [================>.............] - ETA: 6s - loss: 0.3798 - accuracy: 0.9208
15000/25000 [=================>............] - ETA: 6s - loss: 0.3798 - accuracy: 0.9207
15100/25000 [=================>............] - ETA: 6s - loss: 0.3796 - accuracy: 0.9209
15200/25000 [=================>............] - ETA: 6s - loss: 0.3797 - accuracy: 0.9207
15300/25000 [=================>............] - ETA: 5s - loss: 0.3798 - accuracy: 0.9205
15400/25000 [=================>............] - ETA: 5s - loss: 0.3796 - accuracy: 0.9206
15500/25000 [=================>............] - ETA: 5s - loss: 0.3795 - accuracy: 0.9205
15600/25000 [=================>............] - ETA: 5s - loss: 0.3794 - accuracy: 0.9207
15700/25000 [=================>............] - ETA: 5s - loss: 0.3793 - accuracy: 0.9206
15800/25000 [=================>............] - ETA: 5s - loss: 0.3793 - accuracy: 0.9206
15900/25000 [==================>...........] - ETA: 5s - loss: 0.3793 - accuracy: 0.9204
16000/25000 [==================>...........] - ETA: 5s - loss: 0.3789 - accuracy: 0.9205
16100/25000 [==================>...........] - ETA: 5s - loss: 0.3790 - accuracy: 0.9202
16200/25000 [==================>...........] - ETA: 5s - loss: 0.3791 - accuracy: 0.9200
16300/25000 [==================>...........] - ETA: 5s - loss: 0.3790 - accuracy: 0.9200
16400/25000 [==================>...........] - ETA: 5s - loss: 0.3789 - accuracy: 0.9199
16500/25000 [==================>...........] - ETA: 5s - loss: 0.3787 - accuracy: 0.9200
16600/25000 [==================>...........] - ETA: 5s - loss: 0.3783 - accuracy: 0.9202
16700/25000 [===================>..........] - ETA: 5s - loss: 0.3782 - accuracy: 0.9204
16800/25000 [===================>..........] - ETA: 5s - loss: 0.3786 - accuracy: 0.9198
16900/25000 [===================>..........] - ETA: 4s - loss: 0.3782 - accuracy: 0.9199
17000/25000 [===================>..........] - ETA: 4s - loss: 0.3784 - accuracy: 0.9196
17100/25000 [===================>..........] - ETA: 4s - loss: 0.3783 - accuracy: 0.9197
17200/25000 [===================>..........] - ETA: 4s - loss: 0.3784 - accuracy: 0.9196
17300/25000 [===================>..........] - ETA: 4s - loss: 0.3786 - accuracy: 0.9194
17400/25000 [===================>..........] - ETA: 4s - loss: 0.3784 - accuracy: 0.9196
17500/25000 [====================>.........] - ETA: 4s - loss: 0.3785 - accuracy: 0.9195
17600/25000 [====================>.........] - ETA: 4s - loss: 0.3783 - accuracy: 0.9195
17700/25000 [====================>.........] - ETA: 4s - loss: 0.3782 - accuracy: 0.9195
17800/25000 [====================>.........] - ETA: 4s - loss: 0.3782 - accuracy: 0.9195
17900/25000 [====================>.........] - ETA: 4s - loss: 0.3779 - accuracy: 0.9197
18000/25000 [====================>.........] - ETA: 4s - loss: 0.3779 - accuracy: 0.9196
18100/25000 [====================>.........] - ETA: 4s - loss: 0.3779 - accuracy: 0.9195
18200/25000 [====================>.........] - ETA: 4s - loss: 0.3778 - accuracy: 0.9196
18300/25000 [====================>.........] - ETA: 4s - loss: 0.3778 - accuracy: 0.9195
18400/25000 [=====================>........] - ETA: 4s - loss: 0.3778 - accuracy: 0.9194
18500/25000 [=====================>........] - ETA: 4s - loss: 0.3778 - accuracy: 0.9195
18600/25000 [=====================>........] - ETA: 3s - loss: 0.3778 - accuracy: 0.9194
18700/25000 [=====================>........] - ETA: 3s - loss: 0.3778 - accuracy: 0.9195
18800/25000 [=====================>........] - ETA: 3s - loss: 0.3777 - accuracy: 0.9194
18900/25000 [=====================>........] - ETA: 3s - loss: 0.3780 - accuracy: 0.9191
19000/25000 [=====================>........] - ETA: 3s - loss: 0.3780 - accuracy: 0.9191
19100/25000 [=====================>........] - ETA: 3s - loss: 0.3782 - accuracy: 0.9188
19200/25000 [======================>.......] - ETA: 3s - loss: 0.3780 - accuracy: 0.9191
19300/25000 [======================>.......] - ETA: 3s - loss: 0.3781 - accuracy: 0.9190
19400/25000 [======================>.......] - ETA: 3s - loss: 0.3781 - accuracy: 0.9190
19500/25000 [======================>.......] - ETA: 3s - loss: 0.3781 - accuracy: 0.9191
19600/25000 [======================>.......] - ETA: 3s - loss: 0.3784 - accuracy: 0.9188
19700/25000 [======================>.......] - ETA: 3s - loss: 0.3783 - accuracy: 0.9189
19800/25000 [======================>.......] - ETA: 3s - loss: 0.3781 - accuracy: 0.9190
19900/25000 [======================>.......] - ETA: 3s - loss: 0.3783 - accuracy: 0.9189
20000/25000 [=======================>......] - ETA: 3s - loss: 0.3784 - accuracy: 0.9189
20100/25000 [=======================>......] - ETA: 3s - loss: 0.3785 - accuracy: 0.9186
20200/25000 [=======================>......] - ETA: 2s - loss: 0.3785 - accuracy: 0.9186
20300/25000 [=======================>......] - ETA: 2s - loss: 0.3783 - accuracy: 0.9186
20400/25000 [=======================>......] - ETA: 2s - loss: 0.3783 - accuracy: 0.9186
20500/25000 [=======================>......] - ETA: 2s - loss: 0.3783 - accuracy: 0.9186
20600/25000 [=======================>......] - ETA: 2s - loss: 0.3784 - accuracy: 0.9185
20700/25000 [=======================>......] - ETA: 2s - loss: 0.3781 - accuracy: 0.9187
20800/25000 [=======================>......] - ETA: 2s - loss: 0.3783 - accuracy: 0.9186
20900/25000 [========================>.....] - ETA: 2s - loss: 0.3781 - accuracy: 0.9188
21000/25000 [========================>.....] - ETA: 2s - loss: 0.3781 - accuracy: 0.9187
21100/25000 [========================>.....] - ETA: 2s - loss: 0.3780 - accuracy: 0.9187
21200/25000 [========================>.....] - ETA: 2s - loss: 0.3777 - accuracy: 0.9190
21300/25000 [========================>.....] - ETA: 2s - loss: 0.3776 - accuracy: 0.9189
21400/25000 [========================>.....] - ETA: 2s - loss: 0.3777 - accuracy: 0.9189
21500/25000 [========================>.....] - ETA: 2s - loss: 0.3777 - accuracy: 0.9188
21600/25000 [========================>.....] - ETA: 2s - loss: 0.3776 - accuracy: 0.9188
21700/25000 [=========================>....] - ETA: 2s - loss: 0.3775 - accuracy: 0.9188
21800/25000 [=========================>....] - ETA: 1s - loss: 0.3774 - accuracy: 0.9189
21900/25000 [=========================>....] - ETA: 1s - loss: 0.3774 - accuracy: 0.9187
22000/25000 [=========================>....] - ETA: 1s - loss: 0.3771 - accuracy: 0.9190
22100/25000 [=========================>....] - ETA: 1s - loss: 0.3773 - accuracy: 0.9188
22200/25000 [=========================>....] - ETA: 1s - loss: 0.3773 - accuracy: 0.9186
22300/25000 [=========================>....] - ETA: 1s - loss: 0.3773 - accuracy: 0.9186
22400/25000 [=========================>....] - ETA: 1s - loss: 0.3774 - accuracy: 0.9184
22500/25000 [==========================>...] - ETA: 1s - loss: 0.3771 - accuracy: 0.9186
22600/25000 [==========================>...] - ETA: 1s - loss: 0.3768 - accuracy: 0.9188
22700/25000 [==========================>...] - ETA: 1s - loss: 0.3767 - accuracy: 0.9188
22800/25000 [==========================>...] - ETA: 1s - loss: 0.3767 - accuracy: 0.9188
22900/25000 [==========================>...] - ETA: 1s - loss: 0.3765 - accuracy: 0.9190
23000/25000 [==========================>...] - ETA: 1s - loss: 0.3763 - accuracy: 0.9190
23100/25000 [==========================>...] - ETA: 1s - loss: 0.3761 - accuracy: 0.9192
23200/25000 [==========================>...] - ETA: 1s - loss: 0.3761 - accuracy: 0.9193
23300/25000 [==========================>...] - ETA: 1s - loss: 0.3761 - accuracy: 0.9193
23400/25000 [===========================>..] - ETA: 0s - loss: 0.3761 - accuracy: 0.9193
23500/25000 [===========================>..] - ETA: 0s - loss: 0.3762 - accuracy: 0.9192
23600/25000 [===========================>..] - ETA: 0s - loss: 0.3762 - accuracy: 0.9193
23700/25000 [===========================>..] - ETA: 0s - loss: 0.3762 - accuracy: 0.9193
23800/25000 [===========================>..] - ETA: 0s - loss: 0.3761 - accuracy: 0.9193
23900/25000 [===========================>..] - ETA: 0s - loss: 0.3762 - accuracy: 0.9193
24000/25000 [===========================>..] - ETA: 0s - loss: 0.3763 - accuracy: 0.9191
24100/25000 [===========================>..] - ETA: 0s - loss: 0.3764 - accuracy: 0.9190
24200/25000 [============================>.] - ETA: 0s - loss: 0.3765 - accuracy: 0.9189
24300/25000 [============================>.] - ETA: 0s - loss: 0.3764 - accuracy: 0.9189
24400/25000 [============================>.] - ETA: 0s - loss: 0.3763 - accuracy: 0.9190
24500/25000 [============================>.] - ETA: 0s - loss: 0.3765 - accuracy: 0.9188
24600/25000 [============================>.] - ETA: 0s - loss: 0.3764 - accuracy: 0.9188
24700/25000 [============================>.] - ETA: 0s - loss: 0.3765 - accuracy: 0.9187
24800/25000 [============================>.] - ETA: 0s - loss: 0.3764 - accuracy: 0.9187
24900/25000 [============================>.] - ETA: 0s - loss: 0.3766 - accuracy: 0.9185
25000/25000 [==============================] - 19s 777us/step - loss: 0.3765 - accuracy: 0.9184 - val_loss: 0.4283 - val_accuracy: 0.8601
Epoch 5/10

  100/25000 [..............................] - ETA: 14s - loss: 0.3187 - accuracy: 0.9800
  200/25000 [..............................] - ETA: 15s - loss: 0.3566 - accuracy: 0.9300
  300/25000 [..............................] - ETA: 14s - loss: 0.3536 - accuracy: 0.9333
  400/25000 [..............................] - ETA: 15s - loss: 0.3441 - accuracy: 0.9375
  500/25000 [..............................] - ETA: 15s - loss: 0.3461 - accuracy: 0.9380
  600/25000 [..............................] - ETA: 15s - loss: 0.3497 - accuracy: 0.9333
  700/25000 [..............................] - ETA: 15s - loss: 0.3446 - accuracy: 0.9371
  800/25000 [..............................] - ETA: 15s - loss: 0.3464 - accuracy: 0.9350
  900/25000 [>.............................] - ETA: 14s - loss: 0.3448 - accuracy: 0.9367
 1000/25000 [>.............................] - ETA: 14s - loss: 0.3492 - accuracy: 0.9330
 1100/25000 [>.............................] - ETA: 14s - loss: 0.3444 - accuracy: 0.9373
 1200/25000 [>.............................] - ETA: 14s - loss: 0.3449 - accuracy: 0.9375
 1300/25000 [>.............................] - ETA: 14s - loss: 0.3447 - accuracy: 0.9377
 1400/25000 [>.............................] - ETA: 14s - loss: 0.3414 - accuracy: 0.9400
 1500/25000 [>.............................] - ETA: 14s - loss: 0.3422 - accuracy: 0.9387
 1600/25000 [>.............................] - ETA: 14s - loss: 0.3439 - accuracy: 0.9381
 1700/25000 [=>............................] - ETA: 14s - loss: 0.3435 - accuracy: 0.9382
 1800/25000 [=>............................] - ETA: 14s - loss: 0.3421 - accuracy: 0.9394
 1900/25000 [=>............................] - ETA: 14s - loss: 0.3412 - accuracy: 0.9411
 2000/25000 [=>............................] - ETA: 14s - loss: 0.3402 - accuracy: 0.9420
 2100/25000 [=>............................] - ETA: 14s - loss: 0.3388 - accuracy: 0.9429
 2200/25000 [=>............................] - ETA: 14s - loss: 0.3402 - accuracy: 0.9414
 2300/25000 [=>............................] - ETA: 13s - loss: 0.3398 - accuracy: 0.9400
 2400/25000 [=>............................] - ETA: 13s - loss: 0.3382 - accuracy: 0.9413
 2500/25000 [==>...........................] - ETA: 13s - loss: 0.3367 - accuracy: 0.9428
 2600/25000 [==>...........................] - ETA: 13s - loss: 0.3372 - accuracy: 0.9427
 2700/25000 [==>...........................] - ETA: 13s - loss: 0.3375 - accuracy: 0.9426
 2800/25000 [==>...........................] - ETA: 13s - loss: 0.3362 - accuracy: 0.9436
 2900/25000 [==>...........................] - ETA: 13s - loss: 0.3357 - accuracy: 0.9441
 3000/25000 [==>...........................] - ETA: 13s - loss: 0.3357 - accuracy: 0.9447
 3100/25000 [==>...........................] - ETA: 13s - loss: 0.3355 - accuracy: 0.9445
 3200/25000 [==>...........................] - ETA: 13s - loss: 0.3366 - accuracy: 0.9434
 3300/25000 [==>...........................] - ETA: 13s - loss: 0.3362 - accuracy: 0.9436
 3400/25000 [===>..........................] - ETA: 13s - loss: 0.3368 - accuracy: 0.9429
 3500/25000 [===>..........................] - ETA: 13s - loss: 0.3376 - accuracy: 0.9423
 3600/25000 [===>..........................] - ETA: 13s - loss: 0.3394 - accuracy: 0.9408
 3700/25000 [===>..........................] - ETA: 13s - loss: 0.3402 - accuracy: 0.9400
 3800/25000 [===>..........................] - ETA: 13s - loss: 0.3405 - accuracy: 0.9397
 3900/25000 [===>..........................] - ETA: 12s - loss: 0.3410 - accuracy: 0.9390
 4000/25000 [===>..........................] - ETA: 12s - loss: 0.3413 - accuracy: 0.9388
 4100/25000 [===>..........................] - ETA: 12s - loss: 0.3415 - accuracy: 0.9390
 4200/25000 [====>.........................] - ETA: 12s - loss: 0.3413 - accuracy: 0.9388
 4300/25000 [====>.........................] - ETA: 12s - loss: 0.3409 - accuracy: 0.9391
 4400/25000 [====>.........................] - ETA: 12s - loss: 0.3416 - accuracy: 0.9382
 4500/25000 [====>.........................] - ETA: 12s - loss: 0.3415 - accuracy: 0.9382
 4600/25000 [====>.........................] - ETA: 12s - loss: 0.3409 - accuracy: 0.9387
 4700/25000 [====>.........................] - ETA: 12s - loss: 0.3413 - accuracy: 0.9383
 4800/25000 [====>.........................] - ETA: 12s - loss: 0.3405 - accuracy: 0.9390
 4900/25000 [====>.........................] - ETA: 12s - loss: 0.3402 - accuracy: 0.9392
 5000/25000 [=====>........................] - ETA: 12s - loss: 0.3401 - accuracy: 0.9388
 5100/25000 [=====>........................] - ETA: 12s - loss: 0.3407 - accuracy: 0.9382
 5200/25000 [=====>........................] - ETA: 12s - loss: 0.3401 - accuracy: 0.9387
 5300/25000 [=====>........................] - ETA: 12s - loss: 0.3397 - accuracy: 0.9387
 5400/25000 [=====>........................] - ETA: 12s - loss: 0.3398 - accuracy: 0.9389
 5500/25000 [=====>........................] - ETA: 11s - loss: 0.3402 - accuracy: 0.9382
 5600/25000 [=====>........................] - ETA: 11s - loss: 0.3410 - accuracy: 0.9373
 5700/25000 [=====>........................] - ETA: 11s - loss: 0.3406 - accuracy: 0.9377
 5800/25000 [=====>........................] - ETA: 11s - loss: 0.3399 - accuracy: 0.9379
 5900/25000 [======>.......................] - ETA: 11s - loss: 0.3406 - accuracy: 0.9373
 6000/25000 [======>.......................] - ETA: 11s - loss: 0.3409 - accuracy: 0.9372
 6100/25000 [======>.......................] - ETA: 11s - loss: 0.3404 - accuracy: 0.9375
 6200/25000 [======>.......................] - ETA: 11s - loss: 0.3405 - accuracy: 0.9373
 6300/25000 [======>.......................] - ETA: 11s - loss: 0.3411 - accuracy: 0.9368
 6400/25000 [======>.......................] - ETA: 11s - loss: 0.3409 - accuracy: 0.9370
 6500/25000 [======>.......................] - ETA: 11s - loss: 0.3407 - accuracy: 0.9369
 6600/25000 [======>.......................] - ETA: 11s - loss: 0.3404 - accuracy: 0.9370
 6700/25000 [=======>......................] - ETA: 11s - loss: 0.3412 - accuracy: 0.9363
 6800/25000 [=======>......................] - ETA: 11s - loss: 0.3407 - accuracy: 0.9365
 6900/25000 [=======>......................] - ETA: 11s - loss: 0.3403 - accuracy: 0.9367
 7000/25000 [=======>......................] - ETA: 11s - loss: 0.3403 - accuracy: 0.9366
 7100/25000 [=======>......................] - ETA: 11s - loss: 0.3404 - accuracy: 0.9365
 7200/25000 [=======>......................] - ETA: 10s - loss: 0.3401 - accuracy: 0.9367
 7300/25000 [=======>......................] - ETA: 10s - loss: 0.3401 - accuracy: 0.9366
 7400/25000 [=======>......................] - ETA: 10s - loss: 0.3403 - accuracy: 0.9364
 7500/25000 [========>.....................] - ETA: 10s - loss: 0.3405 - accuracy: 0.9361
 7600/25000 [========>.....................] - ETA: 10s - loss: 0.3395 - accuracy: 0.9368
 7700/25000 [========>.....................] - ETA: 10s - loss: 0.3387 - accuracy: 0.9374
 7800/25000 [========>.....................] - ETA: 10s - loss: 0.3384 - accuracy: 0.9373
 7900/25000 [========>.....................] - ETA: 10s - loss: 0.3387 - accuracy: 0.9367
 8000/25000 [========>.....................] - ETA: 10s - loss: 0.3385 - accuracy: 0.9366
 8100/25000 [========>.....................] - ETA: 10s - loss: 0.3390 - accuracy: 0.9362
 8200/25000 [========>.....................] - ETA: 10s - loss: 0.3391 - accuracy: 0.9362
 8300/25000 [========>.....................] - ETA: 10s - loss: 0.3387 - accuracy: 0.9365
 8400/25000 [=========>....................] - ETA: 10s - loss: 0.3389 - accuracy: 0.9363
 8500/25000 [=========>....................] - ETA: 10s - loss: 0.3389 - accuracy: 0.9361
 8600/25000 [=========>....................] - ETA: 10s - loss: 0.3387 - accuracy: 0.9360
 8700/25000 [=========>....................] - ETA: 10s - loss: 0.3383 - accuracy: 0.9364
 8800/25000 [=========>....................] - ETA: 9s - loss: 0.3380 - accuracy: 0.9364 
 8900/25000 [=========>....................] - ETA: 9s - loss: 0.3378 - accuracy: 0.9365
 9000/25000 [=========>....................] - ETA: 9s - loss: 0.3385 - accuracy: 0.9354
 9100/25000 [=========>....................] - ETA: 9s - loss: 0.3388 - accuracy: 0.9353
 9200/25000 [==========>...................] - ETA: 9s - loss: 0.3388 - accuracy: 0.9352
 9300/25000 [==========>...................] - ETA: 9s - loss: 0.3388 - accuracy: 0.9353
 9400/25000 [==========>...................] - ETA: 9s - loss: 0.3392 - accuracy: 0.9349
 9500/25000 [==========>...................] - ETA: 9s - loss: 0.3397 - accuracy: 0.9346
 9600/25000 [==========>...................] - ETA: 9s - loss: 0.3393 - accuracy: 0.9350
 9700/25000 [==========>...................] - ETA: 9s - loss: 0.3397 - accuracy: 0.9345
 9800/25000 [==========>...................] - ETA: 9s - loss: 0.3397 - accuracy: 0.9342
 9900/25000 [==========>...................] - ETA: 9s - loss: 0.3398 - accuracy: 0.9341
10000/25000 [===========>..................] - ETA: 9s - loss: 0.3406 - accuracy: 0.9335
10100/25000 [===========>..................] - ETA: 9s - loss: 0.3405 - accuracy: 0.9336
10200/25000 [===========>..................] - ETA: 9s - loss: 0.3407 - accuracy: 0.9335
10300/25000 [===========>..................] - ETA: 9s - loss: 0.3410 - accuracy: 0.9332
10400/25000 [===========>..................] - ETA: 8s - loss: 0.3410 - accuracy: 0.9331
10500/25000 [===========>..................] - ETA: 8s - loss: 0.3416 - accuracy: 0.9326
10600/25000 [===========>..................] - ETA: 8s - loss: 0.3413 - accuracy: 0.9328
10700/25000 [===========>..................] - ETA: 8s - loss: 0.3410 - accuracy: 0.9330
10800/25000 [===========>..................] - ETA: 8s - loss: 0.3413 - accuracy: 0.9327
10900/25000 [============>.................] - ETA: 8s - loss: 0.3418 - accuracy: 0.9321
11000/25000 [============>.................] - ETA: 8s - loss: 0.3414 - accuracy: 0.9324
11100/25000 [============>.................] - ETA: 8s - loss: 0.3416 - accuracy: 0.9322
11200/25000 [============>.................] - ETA: 8s - loss: 0.3420 - accuracy: 0.9318
11300/25000 [============>.................] - ETA: 8s - loss: 0.3419 - accuracy: 0.9319
11400/25000 [============>.................] - ETA: 8s - loss: 0.3420 - accuracy: 0.9318
11500/25000 [============>.................] - ETA: 8s - loss: 0.3419 - accuracy: 0.9320
11600/25000 [============>.................] - ETA: 8s - loss: 0.3420 - accuracy: 0.9318
11700/25000 [=============>................] - ETA: 8s - loss: 0.3416 - accuracy: 0.9321
11800/25000 [=============>................] - ETA: 8s - loss: 0.3411 - accuracy: 0.9324
11900/25000 [=============>................] - ETA: 8s - loss: 0.3409 - accuracy: 0.9327
12000/25000 [=============>................] - ETA: 8s - loss: 0.3410 - accuracy: 0.9326
12100/25000 [=============>................] - ETA: 7s - loss: 0.3413 - accuracy: 0.9325
12200/25000 [=============>................] - ETA: 7s - loss: 0.3411 - accuracy: 0.9325
12300/25000 [=============>................] - ETA: 7s - loss: 0.3409 - accuracy: 0.9328
12400/25000 [=============>................] - ETA: 7s - loss: 0.3410 - accuracy: 0.9327
12500/25000 [==============>...............] - ETA: 7s - loss: 0.3415 - accuracy: 0.9323
12600/25000 [==============>...............] - ETA: 7s - loss: 0.3413 - accuracy: 0.9325
12700/25000 [==============>...............] - ETA: 7s - loss: 0.3411 - accuracy: 0.9326
12800/25000 [==============>...............] - ETA: 7s - loss: 0.3416 - accuracy: 0.9322
12900/25000 [==============>...............] - ETA: 7s - loss: 0.3415 - accuracy: 0.9324
13000/25000 [==============>...............] - ETA: 7s - loss: 0.3418 - accuracy: 0.9322
13100/25000 [==============>...............] - ETA: 7s - loss: 0.3420 - accuracy: 0.9321
13200/25000 [==============>...............] - ETA: 7s - loss: 0.3421 - accuracy: 0.9322
13300/25000 [==============>...............] - ETA: 7s - loss: 0.3423 - accuracy: 0.9320
13400/25000 [===============>..............] - ETA: 7s - loss: 0.3420 - accuracy: 0.9321
13500/25000 [===============>..............] - ETA: 7s - loss: 0.3419 - accuracy: 0.9321
13600/25000 [===============>..............] - ETA: 7s - loss: 0.3421 - accuracy: 0.9319
13700/25000 [===============>..............] - ETA: 6s - loss: 0.3418 - accuracy: 0.9320
13800/25000 [===============>..............] - ETA: 6s - loss: 0.3415 - accuracy: 0.9322
13900/25000 [===============>..............] - ETA: 6s - loss: 0.3413 - accuracy: 0.9323
14000/25000 [===============>..............] - ETA: 6s - loss: 0.3414 - accuracy: 0.9321
14100/25000 [===============>..............] - ETA: 6s - loss: 0.3416 - accuracy: 0.9321
14200/25000 [================>.............] - ETA: 6s - loss: 0.3413 - accuracy: 0.9322
14300/25000 [================>.............] - ETA: 6s - loss: 0.3409 - accuracy: 0.9325
14400/25000 [================>.............] - ETA: 6s - loss: 0.3409 - accuracy: 0.9324
14500/25000 [================>.............] - ETA: 6s - loss: 0.3410 - accuracy: 0.9323
14600/25000 [================>.............] - ETA: 6s - loss: 0.3410 - accuracy: 0.9324
14700/25000 [================>.............] - ETA: 6s - loss: 0.3410 - accuracy: 0.9322
14800/25000 [================>.............] - ETA: 6s - loss: 0.3415 - accuracy: 0.9319
14900/25000 [================>.............] - ETA: 6s - loss: 0.3416 - accuracy: 0.9318
15000/25000 [=================>............] - ETA: 6s - loss: 0.3413 - accuracy: 0.9320
15100/25000 [=================>............] - ETA: 6s - loss: 0.3408 - accuracy: 0.9325
15200/25000 [=================>............] - ETA: 6s - loss: 0.3407 - accuracy: 0.9324
15300/25000 [=================>............] - ETA: 5s - loss: 0.3407 - accuracy: 0.9322
15400/25000 [=================>............] - ETA: 5s - loss: 0.3408 - accuracy: 0.9321
15500/25000 [=================>............] - ETA: 5s - loss: 0.3407 - accuracy: 0.9321
15600/25000 [=================>............] - ETA: 5s - loss: 0.3405 - accuracy: 0.9323
15700/25000 [=================>............] - ETA: 5s - loss: 0.3406 - accuracy: 0.9322
15800/25000 [=================>............] - ETA: 5s - loss: 0.3405 - accuracy: 0.9323
15900/25000 [==================>...........] - ETA: 5s - loss: 0.3405 - accuracy: 0.9323
16000/25000 [==================>...........] - ETA: 5s - loss: 0.3403 - accuracy: 0.9324
16100/25000 [==================>...........] - ETA: 5s - loss: 0.3406 - accuracy: 0.9322
16200/25000 [==================>...........] - ETA: 5s - loss: 0.3407 - accuracy: 0.9321
16300/25000 [==================>...........] - ETA: 5s - loss: 0.3405 - accuracy: 0.9322
16400/25000 [==================>...........] - ETA: 5s - loss: 0.3406 - accuracy: 0.9320
16500/25000 [==================>...........] - ETA: 5s - loss: 0.3408 - accuracy: 0.9318
16600/25000 [==================>...........] - ETA: 5s - loss: 0.3405 - accuracy: 0.9320
16700/25000 [===================>..........] - ETA: 5s - loss: 0.3401 - accuracy: 0.9323
16800/25000 [===================>..........] - ETA: 5s - loss: 0.3400 - accuracy: 0.9323
16900/25000 [===================>..........] - ETA: 4s - loss: 0.3399 - accuracy: 0.9322
17000/25000 [===================>..........] - ETA: 4s - loss: 0.3395 - accuracy: 0.9324
17100/25000 [===================>..........] - ETA: 4s - loss: 0.3396 - accuracy: 0.9323
17200/25000 [===================>..........] - ETA: 4s - loss: 0.3394 - accuracy: 0.9324
17300/25000 [===================>..........] - ETA: 4s - loss: 0.3393 - accuracy: 0.9325
17400/25000 [===================>..........] - ETA: 4s - loss: 0.3393 - accuracy: 0.9325
17500/25000 [====================>.........] - ETA: 4s - loss: 0.3392 - accuracy: 0.9325
17600/25000 [====================>.........] - ETA: 4s - loss: 0.3392 - accuracy: 0.9326
17700/25000 [====================>.........] - ETA: 4s - loss: 0.3393 - accuracy: 0.9325
17800/25000 [====================>.........] - ETA: 4s - loss: 0.3393 - accuracy: 0.9325
17900/25000 [====================>.........] - ETA: 4s - loss: 0.3395 - accuracy: 0.9323
18000/25000 [====================>.........] - ETA: 4s - loss: 0.3394 - accuracy: 0.9324
18100/25000 [====================>.........] - ETA: 4s - loss: 0.3393 - accuracy: 0.9325
18200/25000 [====================>.........] - ETA: 4s - loss: 0.3393 - accuracy: 0.9326
18300/25000 [====================>.........] - ETA: 4s - loss: 0.3394 - accuracy: 0.9323
18400/25000 [=====================>........] - ETA: 4s - loss: 0.3394 - accuracy: 0.9323
18500/25000 [=====================>........] - ETA: 4s - loss: 0.3394 - accuracy: 0.9324
18600/25000 [=====================>........] - ETA: 3s - loss: 0.3393 - accuracy: 0.9324
18700/25000 [=====================>........] - ETA: 3s - loss: 0.3394 - accuracy: 0.9323
18800/25000 [=====================>........] - ETA: 3s - loss: 0.3394 - accuracy: 0.9323
18900/25000 [=====================>........] - ETA: 3s - loss: 0.3396 - accuracy: 0.9321
19000/25000 [=====================>........] - ETA: 3s - loss: 0.3396 - accuracy: 0.9322
19100/25000 [=====================>........] - ETA: 3s - loss: 0.3395 - accuracy: 0.9321
19200/25000 [======================>.......] - ETA: 3s - loss: 0.3395 - accuracy: 0.9320
19300/25000 [======================>.......] - ETA: 3s - loss: 0.3393 - accuracy: 0.9322
19400/25000 [======================>.......] - ETA: 3s - loss: 0.3393 - accuracy: 0.9322
19500/25000 [======================>.......] - ETA: 3s - loss: 0.3394 - accuracy: 0.9321
19600/25000 [======================>.......] - ETA: 3s - loss: 0.3392 - accuracy: 0.9322
19700/25000 [======================>.......] - ETA: 3s - loss: 0.3394 - accuracy: 0.9321
19800/25000 [======================>.......] - ETA: 3s - loss: 0.3394 - accuracy: 0.9321
19900/25000 [======================>.......] - ETA: 3s - loss: 0.3395 - accuracy: 0.9320
20000/25000 [=======================>......] - ETA: 3s - loss: 0.3393 - accuracy: 0.9322
20100/25000 [=======================>......] - ETA: 3s - loss: 0.3394 - accuracy: 0.9319
20200/25000 [=======================>......] - ETA: 2s - loss: 0.3394 - accuracy: 0.9319
20300/25000 [=======================>......] - ETA: 2s - loss: 0.3394 - accuracy: 0.9318
20400/25000 [=======================>......] - ETA: 2s - loss: 0.3391 - accuracy: 0.9319
20500/25000 [=======================>......] - ETA: 2s - loss: 0.3392 - accuracy: 0.9318
20600/25000 [=======================>......] - ETA: 2s - loss: 0.3391 - accuracy: 0.9318
20700/25000 [=======================>......] - ETA: 2s - loss: 0.3390 - accuracy: 0.9318
20800/25000 [=======================>......] - ETA: 2s - loss: 0.3391 - accuracy: 0.9316
20900/25000 [========================>.....] - ETA: 2s - loss: 0.3389 - accuracy: 0.9317
21000/25000 [========================>.....] - ETA: 2s - loss: 0.3390 - accuracy: 0.9315
21100/25000 [========================>.....] - ETA: 2s - loss: 0.3390 - accuracy: 0.9314
21200/25000 [========================>.....] - ETA: 2s - loss: 0.3388 - accuracy: 0.9315
21300/25000 [========================>.....] - ETA: 2s - loss: 0.3385 - accuracy: 0.9317
21400/25000 [========================>.....] - ETA: 2s - loss: 0.3387 - accuracy: 0.9316
21500/25000 [========================>.....] - ETA: 2s - loss: 0.3387 - accuracy: 0.9315
21600/25000 [========================>.....] - ETA: 2s - loss: 0.3384 - accuracy: 0.9317
21700/25000 [=========================>....] - ETA: 2s - loss: 0.3384 - accuracy: 0.9318
21800/25000 [=========================>....] - ETA: 1s - loss: 0.3384 - accuracy: 0.9317
21900/25000 [=========================>....] - ETA: 1s - loss: 0.3384 - accuracy: 0.9316
22000/25000 [=========================>....] - ETA: 1s - loss: 0.3384 - accuracy: 0.9315
22100/25000 [=========================>....] - ETA: 1s - loss: 0.3385 - accuracy: 0.9315
22200/25000 [=========================>....] - ETA: 1s - loss: 0.3387 - accuracy: 0.9314
22300/25000 [=========================>....] - ETA: 1s - loss: 0.3389 - accuracy: 0.9312
22400/25000 [=========================>....] - ETA: 1s - loss: 0.3389 - accuracy: 0.9312
22500/25000 [==========================>...] - ETA: 1s - loss: 0.3389 - accuracy: 0.9311
22600/25000 [==========================>...] - ETA: 1s - loss: 0.3389 - accuracy: 0.9311
22700/25000 [==========================>...] - ETA: 1s - loss: 0.3388 - accuracy: 0.9311
22800/25000 [==========================>...] - ETA: 1s - loss: 0.3389 - accuracy: 0.9311
22900/25000 [==========================>...] - ETA: 1s - loss: 0.3390 - accuracy: 0.9310
23000/25000 [==========================>...] - ETA: 1s - loss: 0.3388 - accuracy: 0.9311
23100/25000 [==========================>...] - ETA: 1s - loss: 0.3386 - accuracy: 0.9313
23200/25000 [==========================>...] - ETA: 1s - loss: 0.3386 - accuracy: 0.9312
23300/25000 [==========================>...] - ETA: 1s - loss: 0.3385 - accuracy: 0.9311
23400/25000 [===========================>..] - ETA: 0s - loss: 0.3383 - accuracy: 0.9312
23500/25000 [===========================>..] - ETA: 0s - loss: 0.3381 - accuracy: 0.9313
23600/25000 [===========================>..] - ETA: 0s - loss: 0.3380 - accuracy: 0.9314
23700/25000 [===========================>..] - ETA: 0s - loss: 0.3378 - accuracy: 0.9314
23800/25000 [===========================>..] - ETA: 0s - loss: 0.3377 - accuracy: 0.9316
23900/25000 [===========================>..] - ETA: 0s - loss: 0.3377 - accuracy: 0.9315
24000/25000 [===========================>..] - ETA: 0s - loss: 0.3379 - accuracy: 0.9313
24100/25000 [===========================>..] - ETA: 0s - loss: 0.3377 - accuracy: 0.9315
24200/25000 [============================>.] - ETA: 0s - loss: 0.3380 - accuracy: 0.9312
24300/25000 [============================>.] - ETA: 0s - loss: 0.3381 - accuracy: 0.9312
24400/25000 [============================>.] - ETA: 0s - loss: 0.3379 - accuracy: 0.9313
24500/25000 [============================>.] - ETA: 0s - loss: 0.3380 - accuracy: 0.9312
24600/25000 [============================>.] - ETA: 0s - loss: 0.3380 - accuracy: 0.9311
24700/25000 [============================>.] - ETA: 0s - loss: 0.3382 - accuracy: 0.9310
24800/25000 [============================>.] - ETA: 0s - loss: 0.3382 - accuracy: 0.9310
24900/25000 [============================>.] - ETA: 0s - loss: 0.3379 - accuracy: 0.9312
25000/25000 [==============================] - 19s 777us/step - loss: 0.3378 - accuracy: 0.9312 - val_loss: 0.4133 - val_accuracy: 0.8573
Epoch 6/10

  100/25000 [..............................] - ETA: 14s - loss: 0.2929 - accuracy: 0.9700
  200/25000 [..............................] - ETA: 15s - loss: 0.3046 - accuracy: 0.9500
  300/25000 [..............................] - ETA: 15s - loss: 0.3138 - accuracy: 0.9400
  400/25000 [..............................] - ETA: 15s - loss: 0.3223 - accuracy: 0.9375
  500/25000 [..............................] - ETA: 15s - loss: 0.3148 - accuracy: 0.9420
  600/25000 [..............................] - ETA: 15s - loss: 0.3162 - accuracy: 0.9383
  700/25000 [..............................] - ETA: 15s - loss: 0.3178 - accuracy: 0.9386
  800/25000 [..............................] - ETA: 15s - loss: 0.3154 - accuracy: 0.9413
  900/25000 [>.............................] - ETA: 14s - loss: 0.3162 - accuracy: 0.9411
 1000/25000 [>.............................] - ETA: 14s - loss: 0.3181 - accuracy: 0.9400
 1100/25000 [>.............................] - ETA: 14s - loss: 0.3153 - accuracy: 0.9427
 1200/25000 [>.............................] - ETA: 14s - loss: 0.3155 - accuracy: 0.9433
 1300/25000 [>.............................] - ETA: 14s - loss: 0.3154 - accuracy: 0.9431
 1400/25000 [>.............................] - ETA: 14s - loss: 0.3178 - accuracy: 0.9407
 1500/25000 [>.............................] - ETA: 14s - loss: 0.3172 - accuracy: 0.9413
 1600/25000 [>.............................] - ETA: 14s - loss: 0.3164 - accuracy: 0.9419
 1700/25000 [=>............................] - ETA: 14s - loss: 0.3152 - accuracy: 0.9412
 1800/25000 [=>............................] - ETA: 14s - loss: 0.3157 - accuracy: 0.9400
 1900/25000 [=>............................] - ETA: 14s - loss: 0.3130 - accuracy: 0.9421
 2000/25000 [=>............................] - ETA: 14s - loss: 0.3149 - accuracy: 0.9400
 2100/25000 [=>............................] - ETA: 14s - loss: 0.3159 - accuracy: 0.9386
 2200/25000 [=>............................] - ETA: 14s - loss: 0.3137 - accuracy: 0.9405
 2300/25000 [=>............................] - ETA: 13s - loss: 0.3126 - accuracy: 0.9409
 2400/25000 [=>............................] - ETA: 13s - loss: 0.3125 - accuracy: 0.9413
 2500/25000 [==>...........................] - ETA: 13s - loss: 0.3123 - accuracy: 0.9416
 2600/25000 [==>...........................] - ETA: 13s - loss: 0.3113 - accuracy: 0.9423
 2700/25000 [==>...........................] - ETA: 13s - loss: 0.3125 - accuracy: 0.9411
 2800/25000 [==>...........................] - ETA: 13s - loss: 0.3131 - accuracy: 0.9407
 2900/25000 [==>...........................] - ETA: 13s - loss: 0.3125 - accuracy: 0.9410
 3000/25000 [==>...........................] - ETA: 13s - loss: 0.3125 - accuracy: 0.9410
 3100/25000 [==>...........................] - ETA: 13s - loss: 0.3121 - accuracy: 0.9410
 3200/25000 [==>...........................] - ETA: 13s - loss: 0.3122 - accuracy: 0.9403
 3300/25000 [==>...........................] - ETA: 13s - loss: 0.3120 - accuracy: 0.9406
 3400/25000 [===>..........................] - ETA: 13s - loss: 0.3115 - accuracy: 0.9409
 3500/25000 [===>..........................] - ETA: 13s - loss: 0.3128 - accuracy: 0.9394
 3600/25000 [===>..........................] - ETA: 13s - loss: 0.3120 - accuracy: 0.9406
 3700/25000 [===>..........................] - ETA: 13s - loss: 0.3117 - accuracy: 0.9408
 3800/25000 [===>..........................] - ETA: 13s - loss: 0.3111 - accuracy: 0.9413
 3900/25000 [===>..........................] - ETA: 13s - loss: 0.3109 - accuracy: 0.9415
 4000/25000 [===>..........................] - ETA: 12s - loss: 0.3106 - accuracy: 0.9417
 4100/25000 [===>..........................] - ETA: 12s - loss: 0.3111 - accuracy: 0.9410
 4200/25000 [====>.........................] - ETA: 12s - loss: 0.3111 - accuracy: 0.9412
 4300/25000 [====>.........................] - ETA: 12s - loss: 0.3112 - accuracy: 0.9409
 4400/25000 [====>.........................] - ETA: 12s - loss: 0.3121 - accuracy: 0.9405
 4500/25000 [====>.........................] - ETA: 12s - loss: 0.3123 - accuracy: 0.9407
 4600/25000 [====>.........................] - ETA: 12s - loss: 0.3122 - accuracy: 0.9411
 4700/25000 [====>.........................] - ETA: 12s - loss: 0.3121 - accuracy: 0.9415
 4800/25000 [====>.........................] - ETA: 12s - loss: 0.3116 - accuracy: 0.9423
 4900/25000 [====>.........................] - ETA: 12s - loss: 0.3103 - accuracy: 0.9433
 5000/25000 [=====>........................] - ETA: 12s - loss: 0.3097 - accuracy: 0.9436
 5100/25000 [=====>........................] - ETA: 12s - loss: 0.3101 - accuracy: 0.9431
 5200/25000 [=====>........................] - ETA: 12s - loss: 0.3094 - accuracy: 0.9437
 5300/25000 [=====>........................] - ETA: 12s - loss: 0.3091 - accuracy: 0.9440
 5400/25000 [=====>........................] - ETA: 12s - loss: 0.3095 - accuracy: 0.9435
 5500/25000 [=====>........................] - ETA: 12s - loss: 0.3094 - accuracy: 0.9435
 5600/25000 [=====>........................] - ETA: 11s - loss: 0.3095 - accuracy: 0.9436
 5700/25000 [=====>........................] - ETA: 11s - loss: 0.3102 - accuracy: 0.9430
 5800/25000 [=====>........................] - ETA: 11s - loss: 0.3099 - accuracy: 0.9429
 5900/25000 [======>.......................] - ETA: 11s - loss: 0.3102 - accuracy: 0.9425
 6000/25000 [======>.......................] - ETA: 11s - loss: 0.3108 - accuracy: 0.9418
 6100/25000 [======>.......................] - ETA: 11s - loss: 0.3111 - accuracy: 0.9416
 6200/25000 [======>.......................] - ETA: 11s - loss: 0.3107 - accuracy: 0.9418
 6300/25000 [======>.......................] - ETA: 11s - loss: 0.3105 - accuracy: 0.9417
 6400/25000 [======>.......................] - ETA: 11s - loss: 0.3103 - accuracy: 0.9420
 6500/25000 [======>.......................] - ETA: 11s - loss: 0.3108 - accuracy: 0.9418
 6600/25000 [======>.......................] - ETA: 11s - loss: 0.3115 - accuracy: 0.9414
 6700/25000 [=======>......................] - ETA: 11s - loss: 0.3110 - accuracy: 0.9415
 6800/25000 [=======>......................] - ETA: 11s - loss: 0.3107 - accuracy: 0.9416
 6900/25000 [=======>......................] - ETA: 11s - loss: 0.3111 - accuracy: 0.9412
 7000/25000 [=======>......................] - ETA: 11s - loss: 0.3112 - accuracy: 0.9410
 7100/25000 [=======>......................] - ETA: 11s - loss: 0.3110 - accuracy: 0.9408
 7200/25000 [=======>......................] - ETA: 10s - loss: 0.3106 - accuracy: 0.9411
 7300/25000 [=======>......................] - ETA: 10s - loss: 0.3107 - accuracy: 0.9410
 7400/25000 [=======>......................] - ETA: 10s - loss: 0.3102 - accuracy: 0.9415
 7500/25000 [========>.....................] - ETA: 10s - loss: 0.3103 - accuracy: 0.9415
 7600/25000 [========>.....................] - ETA: 10s - loss: 0.3096 - accuracy: 0.9420
 7700/25000 [========>.....................] - ETA: 10s - loss: 0.3097 - accuracy: 0.9417
 7800/25000 [========>.....................] - ETA: 10s - loss: 0.3098 - accuracy: 0.9414
 7900/25000 [========>.....................] - ETA: 10s - loss: 0.3097 - accuracy: 0.9414
 8000/25000 [========>.....................] - ETA: 10s - loss: 0.3098 - accuracy: 0.9411
 8100/25000 [========>.....................] - ETA: 10s - loss: 0.3102 - accuracy: 0.9409
 8200/25000 [========>.....................] - ETA: 10s - loss: 0.3101 - accuracy: 0.9410
 8300/25000 [========>.....................] - ETA: 10s - loss: 0.3099 - accuracy: 0.9412
 8400/25000 [=========>....................] - ETA: 10s - loss: 0.3098 - accuracy: 0.9412
 8500/25000 [=========>....................] - ETA: 10s - loss: 0.3098 - accuracy: 0.9412
 8600/25000 [=========>....................] - ETA: 10s - loss: 0.3102 - accuracy: 0.9407
 8700/25000 [=========>....................] - ETA: 10s - loss: 0.3105 - accuracy: 0.9403
 8800/25000 [=========>....................] - ETA: 9s - loss: 0.3103 - accuracy: 0.9405 
 8900/25000 [=========>....................] - ETA: 9s - loss: 0.3101 - accuracy: 0.9406
 9000/25000 [=========>....................] - ETA: 9s - loss: 0.3095 - accuracy: 0.9411
 9100/25000 [=========>....................] - ETA: 9s - loss: 0.3094 - accuracy: 0.9411
 9200/25000 [==========>...................] - ETA: 9s - loss: 0.3091 - accuracy: 0.9413
 9300/25000 [==========>...................] - ETA: 9s - loss: 0.3093 - accuracy: 0.9413
 9400/25000 [==========>...................] - ETA: 9s - loss: 0.3090 - accuracy: 0.9414
 9500/25000 [==========>...................] - ETA: 9s - loss: 0.3087 - accuracy: 0.9415
 9600/25000 [==========>...................] - ETA: 9s - loss: 0.3093 - accuracy: 0.9409
 9700/25000 [==========>...................] - ETA: 9s - loss: 0.3095 - accuracy: 0.9407
 9800/25000 [==========>...................] - ETA: 9s - loss: 0.3094 - accuracy: 0.9408
 9900/25000 [==========>...................] - ETA: 9s - loss: 0.3094 - accuracy: 0.9407
10000/25000 [===========>..................] - ETA: 9s - loss: 0.3095 - accuracy: 0.9406
10100/25000 [===========>..................] - ETA: 9s - loss: 0.3097 - accuracy: 0.9405
10200/25000 [===========>..................] - ETA: 9s - loss: 0.3100 - accuracy: 0.9402
10300/25000 [===========>..................] - ETA: 9s - loss: 0.3102 - accuracy: 0.9399
10400/25000 [===========>..................] - ETA: 8s - loss: 0.3103 - accuracy: 0.9399
10500/25000 [===========>..................] - ETA: 8s - loss: 0.3102 - accuracy: 0.9401
10600/25000 [===========>..................] - ETA: 8s - loss: 0.3106 - accuracy: 0.9399
10700/25000 [===========>..................] - ETA: 8s - loss: 0.3106 - accuracy: 0.9398
10800/25000 [===========>..................] - ETA: 8s - loss: 0.3106 - accuracy: 0.9398
10900/25000 [============>.................] - ETA: 8s - loss: 0.3103 - accuracy: 0.9400
11000/25000 [============>.................] - ETA: 8s - loss: 0.3105 - accuracy: 0.9398
11100/25000 [============>.................] - ETA: 8s - loss: 0.3104 - accuracy: 0.9398
11200/25000 [============>.................] - ETA: 8s - loss: 0.3102 - accuracy: 0.9400
11300/25000 [============>.................] - ETA: 8s - loss: 0.3100 - accuracy: 0.9400
11400/25000 [============>.................] - ETA: 8s - loss: 0.3101 - accuracy: 0.9399
11500/25000 [============>.................] - ETA: 8s - loss: 0.3097 - accuracy: 0.9403
11600/25000 [============>.................] - ETA: 8s - loss: 0.3099 - accuracy: 0.9401
11700/25000 [=============>................] - ETA: 8s - loss: 0.3097 - accuracy: 0.9402
11800/25000 [=============>................] - ETA: 8s - loss: 0.3099 - accuracy: 0.9402
11900/25000 [=============>................] - ETA: 8s - loss: 0.3098 - accuracy: 0.9403
12000/25000 [=============>................] - ETA: 8s - loss: 0.3095 - accuracy: 0.9404
12100/25000 [=============>................] - ETA: 7s - loss: 0.3100 - accuracy: 0.9402
12200/25000 [=============>................] - ETA: 7s - loss: 0.3098 - accuracy: 0.9402
12300/25000 [=============>................] - ETA: 7s - loss: 0.3097 - accuracy: 0.9403
12400/25000 [=============>................] - ETA: 7s - loss: 0.3099 - accuracy: 0.9403
12500/25000 [==============>...............] - ETA: 7s - loss: 0.3098 - accuracy: 0.9404
12600/25000 [==============>...............] - ETA: 7s - loss: 0.3098 - accuracy: 0.9405
12700/25000 [==============>...............] - ETA: 7s - loss: 0.3099 - accuracy: 0.9405
12800/25000 [==============>...............] - ETA: 7s - loss: 0.3098 - accuracy: 0.9405
12900/25000 [==============>...............] - ETA: 7s - loss: 0.3096 - accuracy: 0.9406
13000/25000 [==============>...............] - ETA: 7s - loss: 0.3097 - accuracy: 0.9405
13100/25000 [==============>...............] - ETA: 7s - loss: 0.3093 - accuracy: 0.9409
13200/25000 [==============>...............] - ETA: 7s - loss: 0.3094 - accuracy: 0.9407
13300/25000 [==============>...............] - ETA: 7s - loss: 0.3092 - accuracy: 0.9409
13400/25000 [===============>..............] - ETA: 7s - loss: 0.3092 - accuracy: 0.9409
13500/25000 [===============>..............] - ETA: 7s - loss: 0.3089 - accuracy: 0.9410
13600/25000 [===============>..............] - ETA: 7s - loss: 0.3089 - accuracy: 0.9410
13700/25000 [===============>..............] - ETA: 6s - loss: 0.3089 - accuracy: 0.9408
13800/25000 [===============>..............] - ETA: 6s - loss: 0.3089 - accuracy: 0.9409
13900/25000 [===============>..............] - ETA: 6s - loss: 0.3089 - accuracy: 0.9409
14000/25000 [===============>..............] - ETA: 6s - loss: 0.3089 - accuracy: 0.9409
14100/25000 [===============>..............] - ETA: 6s - loss: 0.3090 - accuracy: 0.9406
14200/25000 [================>.............] - ETA: 6s - loss: 0.3090 - accuracy: 0.9407
14300/25000 [================>.............] - ETA: 6s - loss: 0.3088 - accuracy: 0.9410
14400/25000 [================>.............] - ETA: 6s - loss: 0.3086 - accuracy: 0.9410
14500/25000 [================>.............] - ETA: 6s - loss: 0.3084 - accuracy: 0.9412
14600/25000 [================>.............] - ETA: 6s - loss: 0.3086 - accuracy: 0.9412
14700/25000 [================>.............] - ETA: 6s - loss: 0.3083 - accuracy: 0.9414
14800/25000 [================>.............] - ETA: 6s - loss: 0.3084 - accuracy: 0.9413
14900/25000 [================>.............] - ETA: 6s - loss: 0.3083 - accuracy: 0.9414
15000/25000 [=================>............] - ETA: 6s - loss: 0.3084 - accuracy: 0.9414
15100/25000 [=================>............] - ETA: 6s - loss: 0.3083 - accuracy: 0.9414
15200/25000 [=================>............] - ETA: 6s - loss: 0.3083 - accuracy: 0.9413
15300/25000 [=================>............] - ETA: 5s - loss: 0.3080 - accuracy: 0.9414
15400/25000 [=================>............] - ETA: 5s - loss: 0.3080 - accuracy: 0.9414
15500/25000 [=================>............] - ETA: 5s - loss: 0.3079 - accuracy: 0.9413
15600/25000 [=================>............] - ETA: 5s - loss: 0.3078 - accuracy: 0.9413
15700/25000 [=================>............] - ETA: 5s - loss: 0.3079 - accuracy: 0.9413
15800/25000 [=================>............] - ETA: 5s - loss: 0.3079 - accuracy: 0.9413
15900/25000 [==================>...........] - ETA: 5s - loss: 0.3082 - accuracy: 0.9409
16000/25000 [==================>...........] - ETA: 5s - loss: 0.3080 - accuracy: 0.9409
16100/25000 [==================>...........] - ETA: 5s - loss: 0.3079 - accuracy: 0.9410
16200/25000 [==================>...........] - ETA: 5s - loss: 0.3083 - accuracy: 0.9407
16300/25000 [==================>...........] - ETA: 5s - loss: 0.3085 - accuracy: 0.9404
16400/25000 [==================>...........] - ETA: 5s - loss: 0.3083 - accuracy: 0.9405
16500/25000 [==================>...........] - ETA: 5s - loss: 0.3082 - accuracy: 0.9407
16600/25000 [==================>...........] - ETA: 5s - loss: 0.3079 - accuracy: 0.9408
16700/25000 [===================>..........] - ETA: 5s - loss: 0.3077 - accuracy: 0.9410
16800/25000 [===================>..........] - ETA: 5s - loss: 0.3074 - accuracy: 0.9412
16900/25000 [===================>..........] - ETA: 4s - loss: 0.3073 - accuracy: 0.9412
17000/25000 [===================>..........] - ETA: 4s - loss: 0.3076 - accuracy: 0.9409
17100/25000 [===================>..........] - ETA: 4s - loss: 0.3075 - accuracy: 0.9409
17200/25000 [===================>..........] - ETA: 4s - loss: 0.3072 - accuracy: 0.9412
17300/25000 [===================>..........] - ETA: 4s - loss: 0.3073 - accuracy: 0.9410
17400/25000 [===================>..........] - ETA: 4s - loss: 0.3074 - accuracy: 0.9410
17500/25000 [====================>.........] - ETA: 4s - loss: 0.3074 - accuracy: 0.9409
17600/25000 [====================>.........] - ETA: 4s - loss: 0.3074 - accuracy: 0.9409
17700/25000 [====================>.........] - ETA: 4s - loss: 0.3074 - accuracy: 0.9408
17800/25000 [====================>.........] - ETA: 4s - loss: 0.3074 - accuracy: 0.9407
17900/25000 [====================>.........] - ETA: 4s - loss: 0.3075 - accuracy: 0.9407
18000/25000 [====================>.........] - ETA: 4s - loss: 0.3074 - accuracy: 0.9407
18100/25000 [====================>.........] - ETA: 4s - loss: 0.3072 - accuracy: 0.9408
18200/25000 [====================>.........] - ETA: 4s - loss: 0.3072 - accuracy: 0.9408
18300/25000 [====================>.........] - ETA: 4s - loss: 0.3072 - accuracy: 0.9409
18400/25000 [=====================>........] - ETA: 4s - loss: 0.3070 - accuracy: 0.9409
18500/25000 [=====================>........] - ETA: 4s - loss: 0.3068 - accuracy: 0.9410
18600/25000 [=====================>........] - ETA: 3s - loss: 0.3067 - accuracy: 0.9410
18700/25000 [=====================>........] - ETA: 3s - loss: 0.3065 - accuracy: 0.9411
18800/25000 [=====================>........] - ETA: 3s - loss: 0.3062 - accuracy: 0.9414
18900/25000 [=====================>........] - ETA: 3s - loss: 0.3061 - accuracy: 0.9414
19000/25000 [=====================>........] - ETA: 3s - loss: 0.3060 - accuracy: 0.9415
19100/25000 [=====================>........] - ETA: 3s - loss: 0.3060 - accuracy: 0.9415
19200/25000 [======================>.......] - ETA: 3s - loss: 0.3059 - accuracy: 0.9416
19300/25000 [======================>.......] - ETA: 3s - loss: 0.3058 - accuracy: 0.9417
19400/25000 [======================>.......] - ETA: 3s - loss: 0.3061 - accuracy: 0.9413
19500/25000 [======================>.......] - ETA: 3s - loss: 0.3062 - accuracy: 0.9413
19600/25000 [======================>.......] - ETA: 3s - loss: 0.3061 - accuracy: 0.9413
19700/25000 [======================>.......] - ETA: 3s - loss: 0.3062 - accuracy: 0.9413
19800/25000 [======================>.......] - ETA: 3s - loss: 0.3061 - accuracy: 0.9414
19900/25000 [======================>.......] - ETA: 3s - loss: 0.3060 - accuracy: 0.9414
20000/25000 [=======================>......] - ETA: 3s - loss: 0.3061 - accuracy: 0.9412
20100/25000 [=======================>......] - ETA: 3s - loss: 0.3061 - accuracy: 0.9412
20200/25000 [=======================>......] - ETA: 2s - loss: 0.3060 - accuracy: 0.9412
20300/25000 [=======================>......] - ETA: 2s - loss: 0.3059 - accuracy: 0.9413
20400/25000 [=======================>......] - ETA: 2s - loss: 0.3057 - accuracy: 0.9414
20500/25000 [=======================>......] - ETA: 2s - loss: 0.3059 - accuracy: 0.9412
20600/25000 [=======================>......] - ETA: 2s - loss: 0.3060 - accuracy: 0.9411
20700/25000 [=======================>......] - ETA: 2s - loss: 0.3056 - accuracy: 0.9414
20800/25000 [=======================>......] - ETA: 2s - loss: 0.3056 - accuracy: 0.9413
20900/25000 [========================>.....] - ETA: 2s - loss: 0.3057 - accuracy: 0.9411
21000/25000 [========================>.....] - ETA: 2s - loss: 0.3058 - accuracy: 0.9409
21100/25000 [========================>.....] - ETA: 2s - loss: 0.3057 - accuracy: 0.9409
21200/25000 [========================>.....] - ETA: 2s - loss: 0.3057 - accuracy: 0.9409
21300/25000 [========================>.....] - ETA: 2s - loss: 0.3056 - accuracy: 0.9410
21400/25000 [========================>.....] - ETA: 2s - loss: 0.3056 - accuracy: 0.9409
21500/25000 [========================>.....] - ETA: 2s - loss: 0.3057 - accuracy: 0.9409
21600/25000 [========================>.....] - ETA: 2s - loss: 0.3055 - accuracy: 0.9409
21700/25000 [=========================>....] - ETA: 2s - loss: 0.3055 - accuracy: 0.9410
21800/25000 [=========================>....] - ETA: 1s - loss: 0.3055 - accuracy: 0.9410
21900/25000 [=========================>....] - ETA: 1s - loss: 0.3055 - accuracy: 0.9409
22000/25000 [=========================>....] - ETA: 1s - loss: 0.3054 - accuracy: 0.9410
22100/25000 [=========================>....] - ETA: 1s - loss: 0.3055 - accuracy: 0.9408
22200/25000 [=========================>....] - ETA: 1s - loss: 0.3053 - accuracy: 0.9409
22300/25000 [=========================>....] - ETA: 1s - loss: 0.3054 - accuracy: 0.9409
22400/25000 [=========================>....] - ETA: 1s - loss: 0.3051 - accuracy: 0.9409
22500/25000 [==========================>...] - ETA: 1s - loss: 0.3052 - accuracy: 0.9408
22600/25000 [==========================>...] - ETA: 1s - loss: 0.3054 - accuracy: 0.9408
22700/25000 [==========================>...] - ETA: 1s - loss: 0.3053 - accuracy: 0.9407
22800/25000 [==========================>...] - ETA: 1s - loss: 0.3055 - accuracy: 0.9405
22900/25000 [==========================>...] - ETA: 1s - loss: 0.3056 - accuracy: 0.9404
23000/25000 [==========================>...] - ETA: 1s - loss: 0.3058 - accuracy: 0.9403
23100/25000 [==========================>...] - ETA: 1s - loss: 0.3058 - accuracy: 0.9402
23200/25000 [==========================>...] - ETA: 1s - loss: 0.3060 - accuracy: 0.9400
23300/25000 [==========================>...] - ETA: 1s - loss: 0.3060 - accuracy: 0.9400
23400/25000 [===========================>..] - ETA: 0s - loss: 0.3059 - accuracy: 0.9400
23500/25000 [===========================>..] - ETA: 0s - loss: 0.3058 - accuracy: 0.9400
23600/25000 [===========================>..] - ETA: 0s - loss: 0.3057 - accuracy: 0.9401
23700/25000 [===========================>..] - ETA: 0s - loss: 0.3056 - accuracy: 0.9401
23800/25000 [===========================>..] - ETA: 0s - loss: 0.3057 - accuracy: 0.9400
23900/25000 [===========================>..] - ETA: 0s - loss: 0.3057 - accuracy: 0.9400
24000/25000 [===========================>..] - ETA: 0s - loss: 0.3056 - accuracy: 0.9400
24100/25000 [===========================>..] - ETA: 0s - loss: 0.3056 - accuracy: 0.9400
24200/25000 [============================>.] - ETA: 0s - loss: 0.3055 - accuracy: 0.9400
24300/25000 [============================>.] - ETA: 0s - loss: 0.3054 - accuracy: 0.9401
24400/25000 [============================>.] - ETA: 0s - loss: 0.3053 - accuracy: 0.9402
24500/25000 [============================>.] - ETA: 0s - loss: 0.3052 - accuracy: 0.9403
24600/25000 [============================>.] - ETA: 0s - loss: 0.3054 - accuracy: 0.9402
24700/25000 [============================>.] - ETA: 0s - loss: 0.3054 - accuracy: 0.9401
24800/25000 [============================>.] - ETA: 0s - loss: 0.3055 - accuracy: 0.9400
24900/25000 [============================>.] - ETA: 0s - loss: 0.3054 - accuracy: 0.9401
25000/25000 [==============================] - 19s 776us/step - loss: 0.3054 - accuracy: 0.9400 - val_loss: 0.4108 - val_accuracy: 0.8613
Epoch 7/10

  100/25000 [..............................] - ETA: 15s - loss: 0.2799 - accuracy: 0.9500
  200/25000 [..............................] - ETA: 14s - loss: 0.2858 - accuracy: 0.9400
  300/25000 [..............................] - ETA: 14s - loss: 0.3037 - accuracy: 0.9333
  400/25000 [..............................] - ETA: 14s - loss: 0.2891 - accuracy: 0.9425
  500/25000 [..............................] - ETA: 15s - loss: 0.2982 - accuracy: 0.9360
  600/25000 [..............................] - ETA: 15s - loss: 0.2924 - accuracy: 0.9417
  700/25000 [..............................] - ETA: 14s - loss: 0.2941 - accuracy: 0.9400
  800/25000 [..............................] - ETA: 14s - loss: 0.2922 - accuracy: 0.9438
  900/25000 [>.............................] - ETA: 14s - loss: 0.2927 - accuracy: 0.9433
 1000/25000 [>.............................] - ETA: 14s - loss: 0.2887 - accuracy: 0.9470
 1100/25000 [>.............................] - ETA: 14s - loss: 0.2896 - accuracy: 0.9464
 1200/25000 [>.............................] - ETA: 14s - loss: 0.2872 - accuracy: 0.9483
 1300/25000 [>.............................] - ETA: 14s - loss: 0.2879 - accuracy: 0.9485
 1400/25000 [>.............................] - ETA: 14s - loss: 0.2868 - accuracy: 0.9500
 1500/25000 [>.............................] - ETA: 14s - loss: 0.2883 - accuracy: 0.9480
 1600/25000 [>.............................] - ETA: 14s - loss: 0.2860 - accuracy: 0.9494
 1700/25000 [=>............................] - ETA: 14s - loss: 0.2865 - accuracy: 0.9488
 1800/25000 [=>............................] - ETA: 14s - loss: 0.2863 - accuracy: 0.9483
 1900/25000 [=>............................] - ETA: 14s - loss: 0.2868 - accuracy: 0.9479
 2000/25000 [=>............................] - ETA: 14s - loss: 0.2846 - accuracy: 0.9495
 2100/25000 [=>............................] - ETA: 14s - loss: 0.2849 - accuracy: 0.9495
 2200/25000 [=>............................] - ETA: 14s - loss: 0.2866 - accuracy: 0.9477
 2300/25000 [=>............................] - ETA: 13s - loss: 0.2885 - accuracy: 0.9457
 2400/25000 [=>............................] - ETA: 13s - loss: 0.2915 - accuracy: 0.9446
 2500/25000 [==>...........................] - ETA: 13s - loss: 0.2900 - accuracy: 0.9456
 2600/25000 [==>...........................] - ETA: 13s - loss: 0.2874 - accuracy: 0.9477
 2700/25000 [==>...........................] - ETA: 13s - loss: 0.2861 - accuracy: 0.9489
 2800/25000 [==>...........................] - ETA: 13s - loss: 0.2869 - accuracy: 0.9479
 2900/25000 [==>...........................] - ETA: 13s - loss: 0.2860 - accuracy: 0.9483
 3000/25000 [==>...........................] - ETA: 13s - loss: 0.2842 - accuracy: 0.9493
 3100/25000 [==>...........................] - ETA: 13s - loss: 0.2841 - accuracy: 0.9497
 3200/25000 [==>...........................] - ETA: 13s - loss: 0.2826 - accuracy: 0.9506
 3300/25000 [==>...........................] - ETA: 13s - loss: 0.2808 - accuracy: 0.9518
 3400/25000 [===>..........................] - ETA: 13s - loss: 0.2793 - accuracy: 0.9529
 3500/25000 [===>..........................] - ETA: 13s - loss: 0.2821 - accuracy: 0.9514
 3600/25000 [===>..........................] - ETA: 13s - loss: 0.2832 - accuracy: 0.9506
 3700/25000 [===>..........................] - ETA: 13s - loss: 0.2844 - accuracy: 0.9500
 3800/25000 [===>..........................] - ETA: 13s - loss: 0.2837 - accuracy: 0.9503
 3900/25000 [===>..........................] - ETA: 13s - loss: 0.2828 - accuracy: 0.9510
 4000/25000 [===>..........................] - ETA: 13s - loss: 0.2831 - accuracy: 0.9505
 4100/25000 [===>..........................] - ETA: 12s - loss: 0.2833 - accuracy: 0.9507
 4200/25000 [====>.........................] - ETA: 12s - loss: 0.2823 - accuracy: 0.9512
 4300/25000 [====>.........................] - ETA: 12s - loss: 0.2812 - accuracy: 0.9521
 4400/25000 [====>.........................] - ETA: 12s - loss: 0.2805 - accuracy: 0.9525
 4500/25000 [====>.........................] - ETA: 12s - loss: 0.2797 - accuracy: 0.9531
 4600/25000 [====>.........................] - ETA: 12s - loss: 0.2793 - accuracy: 0.9533
 4700/25000 [====>.........................] - ETA: 12s - loss: 0.2787 - accuracy: 0.9536
 4800/25000 [====>.........................] - ETA: 12s - loss: 0.2782 - accuracy: 0.9542
 4900/25000 [====>.........................] - ETA: 12s - loss: 0.2794 - accuracy: 0.9533
 5000/25000 [=====>........................] - ETA: 12s - loss: 0.2798 - accuracy: 0.9532
 5100/25000 [=====>........................] - ETA: 12s - loss: 0.2798 - accuracy: 0.9531
 5200/25000 [=====>........................] - ETA: 12s - loss: 0.2793 - accuracy: 0.9535
 5300/25000 [=====>........................] - ETA: 12s - loss: 0.2794 - accuracy: 0.9532
 5400/25000 [=====>........................] - ETA: 12s - loss: 0.2795 - accuracy: 0.9531
 5500/25000 [=====>........................] - ETA: 12s - loss: 0.2799 - accuracy: 0.9531
 5600/25000 [=====>........................] - ETA: 12s - loss: 0.2796 - accuracy: 0.9534
 5700/25000 [=====>........................] - ETA: 11s - loss: 0.2805 - accuracy: 0.9525
 5800/25000 [=====>........................] - ETA: 11s - loss: 0.2810 - accuracy: 0.9521
 5900/25000 [======>.......................] - ETA: 11s - loss: 0.2808 - accuracy: 0.9522
 6000/25000 [======>.......................] - ETA: 11s - loss: 0.2813 - accuracy: 0.9518
 6100/25000 [======>.......................] - ETA: 11s - loss: 0.2804 - accuracy: 0.9523
 6200/25000 [======>.......................] - ETA: 11s - loss: 0.2813 - accuracy: 0.9518
 6300/25000 [======>.......................] - ETA: 11s - loss: 0.2806 - accuracy: 0.9522
 6400/25000 [======>.......................] - ETA: 11s - loss: 0.2802 - accuracy: 0.9525
 6500/25000 [======>.......................] - ETA: 11s - loss: 0.2798 - accuracy: 0.9528
 6600/25000 [======>.......................] - ETA: 11s - loss: 0.2791 - accuracy: 0.9532
 6700/25000 [=======>......................] - ETA: 11s - loss: 0.2793 - accuracy: 0.9527
 6800/25000 [=======>......................] - ETA: 11s - loss: 0.2802 - accuracy: 0.9524
 6900/25000 [=======>......................] - ETA: 11s - loss: 0.2803 - accuracy: 0.9520
 7000/25000 [=======>......................] - ETA: 11s - loss: 0.2799 - accuracy: 0.9523
 7100/25000 [=======>......................] - ETA: 11s - loss: 0.2795 - accuracy: 0.9525
 7200/25000 [=======>......................] - ETA: 11s - loss: 0.2793 - accuracy: 0.9526
 7300/25000 [=======>......................] - ETA: 10s - loss: 0.2787 - accuracy: 0.9532
 7400/25000 [=======>......................] - ETA: 10s - loss: 0.2789 - accuracy: 0.9528
 7500/25000 [========>.....................] - ETA: 10s - loss: 0.2789 - accuracy: 0.9528
 7600/25000 [========>.....................] - ETA: 10s - loss: 0.2786 - accuracy: 0.9530
 7700/25000 [========>.....................] - ETA: 10s - loss: 0.2787 - accuracy: 0.9529
 7800/25000 [========>.....................] - ETA: 10s - loss: 0.2793 - accuracy: 0.9524
 7900/25000 [========>.....................] - ETA: 10s - loss: 0.2795 - accuracy: 0.9523
 8000/25000 [========>.....................] - ETA: 10s - loss: 0.2792 - accuracy: 0.9524
 8100/25000 [========>.....................] - ETA: 10s - loss: 0.2793 - accuracy: 0.9523
 8200/25000 [========>.....................] - ETA: 10s - loss: 0.2796 - accuracy: 0.9522
 8300/25000 [========>.....................] - ETA: 10s - loss: 0.2797 - accuracy: 0.9520
 8400/25000 [=========>....................] - ETA: 10s - loss: 0.2796 - accuracy: 0.9521
 8500/25000 [=========>....................] - ETA: 10s - loss: 0.2794 - accuracy: 0.9520
 8600/25000 [=========>....................] - ETA: 10s - loss: 0.2791 - accuracy: 0.9521
 8700/25000 [=========>....................] - ETA: 10s - loss: 0.2791 - accuracy: 0.9521
 8800/25000 [=========>....................] - ETA: 10s - loss: 0.2792 - accuracy: 0.9519
 8900/25000 [=========>....................] - ETA: 9s - loss: 0.2788 - accuracy: 0.9521 
 9000/25000 [=========>....................] - ETA: 9s - loss: 0.2792 - accuracy: 0.9517
 9100/25000 [=========>....................] - ETA: 9s - loss: 0.2792 - accuracy: 0.9516
 9200/25000 [==========>...................] - ETA: 9s - loss: 0.2796 - accuracy: 0.9513
 9300/25000 [==========>...................] - ETA: 9s - loss: 0.2790 - accuracy: 0.9515
 9400/25000 [==========>...................] - ETA: 9s - loss: 0.2788 - accuracy: 0.9516
 9500/25000 [==========>...................] - ETA: 9s - loss: 0.2786 - accuracy: 0.9517
 9600/25000 [==========>...................] - ETA: 9s - loss: 0.2784 - accuracy: 0.9519
 9700/25000 [==========>...................] - ETA: 9s - loss: 0.2788 - accuracy: 0.9516
 9800/25000 [==========>...................] - ETA: 9s - loss: 0.2789 - accuracy: 0.9514
 9900/25000 [==========>...................] - ETA: 9s - loss: 0.2786 - accuracy: 0.9516
10000/25000 [===========>..................] - ETA: 9s - loss: 0.2787 - accuracy: 0.9516
10100/25000 [===========>..................] - ETA: 9s - loss: 0.2786 - accuracy: 0.9516
10200/25000 [===========>..................] - ETA: 9s - loss: 0.2786 - accuracy: 0.9516
10300/25000 [===========>..................] - ETA: 9s - loss: 0.2789 - accuracy: 0.9514
10400/25000 [===========>..................] - ETA: 9s - loss: 0.2792 - accuracy: 0.9511
10500/25000 [===========>..................] - ETA: 8s - loss: 0.2791 - accuracy: 0.9511
10600/25000 [===========>..................] - ETA: 8s - loss: 0.2792 - accuracy: 0.9510
10700/25000 [===========>..................] - ETA: 8s - loss: 0.2792 - accuracy: 0.9511
10800/25000 [===========>..................] - ETA: 8s - loss: 0.2792 - accuracy: 0.9510
10900/25000 [============>.................] - ETA: 8s - loss: 0.2796 - accuracy: 0.9506
11000/25000 [============>.................] - ETA: 8s - loss: 0.2800 - accuracy: 0.9502
11100/25000 [============>.................] - ETA: 8s - loss: 0.2801 - accuracy: 0.9499
11200/25000 [============>.................] - ETA: 8s - loss: 0.2802 - accuracy: 0.9497
11300/25000 [============>.................] - ETA: 8s - loss: 0.2799 - accuracy: 0.9499
11400/25000 [============>.................] - ETA: 8s - loss: 0.2798 - accuracy: 0.9499
11500/25000 [============>.................] - ETA: 8s - loss: 0.2796 - accuracy: 0.9501
11600/25000 [============>.................] - ETA: 8s - loss: 0.2796 - accuracy: 0.9502
11700/25000 [=============>................] - ETA: 8s - loss: 0.2793 - accuracy: 0.9503
11800/25000 [=============>................] - ETA: 8s - loss: 0.2791 - accuracy: 0.9505
11900/25000 [=============>................] - ETA: 8s - loss: 0.2788 - accuracy: 0.9507
12000/25000 [=============>................] - ETA: 8s - loss: 0.2791 - accuracy: 0.9504
12100/25000 [=============>................] - ETA: 7s - loss: 0.2790 - accuracy: 0.9505
12200/25000 [=============>................] - ETA: 7s - loss: 0.2789 - accuracy: 0.9506
12300/25000 [=============>................] - ETA: 7s - loss: 0.2785 - accuracy: 0.9508
12400/25000 [=============>................] - ETA: 7s - loss: 0.2782 - accuracy: 0.9510
12500/25000 [==============>...............] - ETA: 7s - loss: 0.2779 - accuracy: 0.9512
12600/25000 [==============>...............] - ETA: 7s - loss: 0.2777 - accuracy: 0.9512
12700/25000 [==============>...............] - ETA: 7s - loss: 0.2776 - accuracy: 0.9513
12800/25000 [==============>...............] - ETA: 7s - loss: 0.2777 - accuracy: 0.9512
12900/25000 [==============>...............] - ETA: 7s - loss: 0.2775 - accuracy: 0.9513
13000/25000 [==============>...............] - ETA: 7s - loss: 0.2779 - accuracy: 0.9508
13100/25000 [==============>...............] - ETA: 7s - loss: 0.2783 - accuracy: 0.9505
13200/25000 [==============>...............] - ETA: 7s - loss: 0.2784 - accuracy: 0.9505
13300/25000 [==============>...............] - ETA: 7s - loss: 0.2784 - accuracy: 0.9505
13400/25000 [===============>..............] - ETA: 7s - loss: 0.2781 - accuracy: 0.9505
13500/25000 [===============>..............] - ETA: 7s - loss: 0.2780 - accuracy: 0.9506
13600/25000 [===============>..............] - ETA: 7s - loss: 0.2775 - accuracy: 0.9509
13700/25000 [===============>..............] - ETA: 6s - loss: 0.2774 - accuracy: 0.9509
13800/25000 [===============>..............] - ETA: 6s - loss: 0.2769 - accuracy: 0.9512
13900/25000 [===============>..............] - ETA: 6s - loss: 0.2767 - accuracy: 0.9512
14000/25000 [===============>..............] - ETA: 6s - loss: 0.2764 - accuracy: 0.9514
14100/25000 [===============>..............] - ETA: 6s - loss: 0.2767 - accuracy: 0.9511
14200/25000 [================>.............] - ETA: 6s - loss: 0.2767 - accuracy: 0.9511
14300/25000 [================>.............] - ETA: 6s - loss: 0.2764 - accuracy: 0.9513
14400/25000 [================>.............] - ETA: 6s - loss: 0.2766 - accuracy: 0.9511
14500/25000 [================>.............] - ETA: 6s - loss: 0.2767 - accuracy: 0.9510
14600/25000 [================>.............] - ETA: 6s - loss: 0.2770 - accuracy: 0.9508
14700/25000 [================>.............] - ETA: 6s - loss: 0.2770 - accuracy: 0.9507
14800/25000 [================>.............] - ETA: 6s - loss: 0.2770 - accuracy: 0.9507
14900/25000 [================>.............] - ETA: 6s - loss: 0.2766 - accuracy: 0.9509
15000/25000 [=================>............] - ETA: 6s - loss: 0.2765 - accuracy: 0.9510
15100/25000 [=================>............] - ETA: 6s - loss: 0.2762 - accuracy: 0.9512
15200/25000 [=================>............] - ETA: 6s - loss: 0.2763 - accuracy: 0.9511
15300/25000 [=================>............] - ETA: 6s - loss: 0.2762 - accuracy: 0.9510
15400/25000 [=================>............] - ETA: 5s - loss: 0.2764 - accuracy: 0.9510
15500/25000 [=================>............] - ETA: 5s - loss: 0.2764 - accuracy: 0.9510
15600/25000 [=================>............] - ETA: 5s - loss: 0.2761 - accuracy: 0.9512
15700/25000 [=================>............] - ETA: 5s - loss: 0.2761 - accuracy: 0.9511
15800/25000 [=================>............] - ETA: 5s - loss: 0.2762 - accuracy: 0.9511
15900/25000 [==================>...........] - ETA: 5s - loss: 0.2761 - accuracy: 0.9511
16000/25000 [==================>...........] - ETA: 5s - loss: 0.2758 - accuracy: 0.9513
16100/25000 [==================>...........] - ETA: 5s - loss: 0.2759 - accuracy: 0.9512
16200/25000 [==================>...........] - ETA: 5s - loss: 0.2759 - accuracy: 0.9512
16300/25000 [==================>...........] - ETA: 5s - loss: 0.2757 - accuracy: 0.9513
16400/25000 [==================>...........] - ETA: 5s - loss: 0.2758 - accuracy: 0.9513
16500/25000 [==================>...........] - ETA: 5s - loss: 0.2759 - accuracy: 0.9512
16600/25000 [==================>...........] - ETA: 5s - loss: 0.2757 - accuracy: 0.9513
16700/25000 [===================>..........] - ETA: 5s - loss: 0.2757 - accuracy: 0.9513
16800/25000 [===================>..........] - ETA: 5s - loss: 0.2755 - accuracy: 0.9514
16900/25000 [===================>..........] - ETA: 5s - loss: 0.2756 - accuracy: 0.9514
17000/25000 [===================>..........] - ETA: 4s - loss: 0.2756 - accuracy: 0.9513
17100/25000 [===================>..........] - ETA: 4s - loss: 0.2757 - accuracy: 0.9511
17200/25000 [===================>..........] - ETA: 4s - loss: 0.2756 - accuracy: 0.9511
17300/25000 [===================>..........] - ETA: 4s - loss: 0.2757 - accuracy: 0.9509
17400/25000 [===================>..........] - ETA: 4s - loss: 0.2757 - accuracy: 0.9510
17500/25000 [====================>.........] - ETA: 4s - loss: 0.2757 - accuracy: 0.9509
17600/25000 [====================>.........] - ETA: 4s - loss: 0.2759 - accuracy: 0.9507
17700/25000 [====================>.........] - ETA: 4s - loss: 0.2759 - accuracy: 0.9507
17800/25000 [====================>.........] - ETA: 4s - loss: 0.2759 - accuracy: 0.9507
17900/25000 [====================>.........] - ETA: 4s - loss: 0.2759 - accuracy: 0.9508
18000/25000 [====================>.........] - ETA: 4s - loss: 0.2761 - accuracy: 0.9507
18100/25000 [====================>.........] - ETA: 4s - loss: 0.2758 - accuracy: 0.9508
18200/25000 [====================>.........] - ETA: 4s - loss: 0.2757 - accuracy: 0.9509
18300/25000 [====================>.........] - ETA: 4s - loss: 0.2754 - accuracy: 0.9510
18400/25000 [=====================>........] - ETA: 4s - loss: 0.2753 - accuracy: 0.9510
18500/25000 [=====================>........] - ETA: 4s - loss: 0.2758 - accuracy: 0.9507
18600/25000 [=====================>........] - ETA: 3s - loss: 0.2760 - accuracy: 0.9504
18700/25000 [=====================>........] - ETA: 3s - loss: 0.2757 - accuracy: 0.9506
18800/25000 [=====================>........] - ETA: 3s - loss: 0.2758 - accuracy: 0.9506
18900/25000 [=====================>........] - ETA: 3s - loss: 0.2757 - accuracy: 0.9506
19000/25000 [=====================>........] - ETA: 3s - loss: 0.2760 - accuracy: 0.9504
19100/25000 [=====================>........] - ETA: 3s - loss: 0.2759 - accuracy: 0.9504
19200/25000 [======================>.......] - ETA: 3s - loss: 0.2762 - accuracy: 0.9502
19300/25000 [======================>.......] - ETA: 3s - loss: 0.2762 - accuracy: 0.9502
19400/25000 [======================>.......] - ETA: 3s - loss: 0.2761 - accuracy: 0.9502
19500/25000 [======================>.......] - ETA: 3s - loss: 0.2763 - accuracy: 0.9500
19600/25000 [======================>.......] - ETA: 3s - loss: 0.2762 - accuracy: 0.9501
19700/25000 [======================>.......] - ETA: 3s - loss: 0.2762 - accuracy: 0.9501
19800/25000 [======================>.......] - ETA: 3s - loss: 0.2763 - accuracy: 0.9500
19900/25000 [======================>.......] - ETA: 3s - loss: 0.2763 - accuracy: 0.9500
20000/25000 [=======================>......] - ETA: 3s - loss: 0.2763 - accuracy: 0.9499
20100/25000 [=======================>......] - ETA: 3s - loss: 0.2764 - accuracy: 0.9498
20200/25000 [=======================>......] - ETA: 2s - loss: 0.2764 - accuracy: 0.9498
20300/25000 [=======================>......] - ETA: 2s - loss: 0.2762 - accuracy: 0.9499
20400/25000 [=======================>......] - ETA: 2s - loss: 0.2761 - accuracy: 0.9500
20500/25000 [=======================>......] - ETA: 2s - loss: 0.2762 - accuracy: 0.9499
20600/25000 [=======================>......] - ETA: 2s - loss: 0.2761 - accuracy: 0.9500
20700/25000 [=======================>......] - ETA: 2s - loss: 0.2761 - accuracy: 0.9500
20800/25000 [=======================>......] - ETA: 2s - loss: 0.2763 - accuracy: 0.9499
20900/25000 [========================>.....] - ETA: 2s - loss: 0.2763 - accuracy: 0.9499
21000/25000 [========================>.....] - ETA: 2s - loss: 0.2766 - accuracy: 0.9497
21100/25000 [========================>.....] - ETA: 2s - loss: 0.2765 - accuracy: 0.9498
21200/25000 [========================>.....] - ETA: 2s - loss: 0.2765 - accuracy: 0.9497
21300/25000 [========================>.....] - ETA: 2s - loss: 0.2763 - accuracy: 0.9498
21400/25000 [========================>.....] - ETA: 2s - loss: 0.2765 - accuracy: 0.9497
21500/25000 [========================>.....] - ETA: 2s - loss: 0.2764 - accuracy: 0.9497
21600/25000 [========================>.....] - ETA: 2s - loss: 0.2761 - accuracy: 0.9498
21700/25000 [=========================>....] - ETA: 2s - loss: 0.2760 - accuracy: 0.9499
21800/25000 [=========================>....] - ETA: 1s - loss: 0.2758 - accuracy: 0.9500
21900/25000 [=========================>....] - ETA: 1s - loss: 0.2759 - accuracy: 0.9499
22000/25000 [=========================>....] - ETA: 1s - loss: 0.2761 - accuracy: 0.9498
22100/25000 [=========================>....] - ETA: 1s - loss: 0.2760 - accuracy: 0.9498
22200/25000 [=========================>....] - ETA: 1s - loss: 0.2761 - accuracy: 0.9497
22300/25000 [=========================>....] - ETA: 1s - loss: 0.2760 - accuracy: 0.9497
22400/25000 [=========================>....] - ETA: 1s - loss: 0.2761 - accuracy: 0.9496
22500/25000 [==========================>...] - ETA: 1s - loss: 0.2761 - accuracy: 0.9496
22600/25000 [==========================>...] - ETA: 1s - loss: 0.2760 - accuracy: 0.9496
22700/25000 [==========================>...] - ETA: 1s - loss: 0.2762 - accuracy: 0.9495
22800/25000 [==========================>...] - ETA: 1s - loss: 0.2763 - accuracy: 0.9494
22900/25000 [==========================>...] - ETA: 1s - loss: 0.2763 - accuracy: 0.9493
23000/25000 [==========================>...] - ETA: 1s - loss: 0.2763 - accuracy: 0.9493
23100/25000 [==========================>...] - ETA: 1s - loss: 0.2761 - accuracy: 0.9494
23200/25000 [==========================>...] - ETA: 1s - loss: 0.2762 - accuracy: 0.9493
23300/25000 [==========================>...] - ETA: 1s - loss: 0.2760 - accuracy: 0.9494
23400/25000 [===========================>..] - ETA: 0s - loss: 0.2759 - accuracy: 0.9495
23500/25000 [===========================>..] - ETA: 0s - loss: 0.2760 - accuracy: 0.9494
23600/25000 [===========================>..] - ETA: 0s - loss: 0.2758 - accuracy: 0.9494
23700/25000 [===========================>..] - ETA: 0s - loss: 0.2757 - accuracy: 0.9494
23800/25000 [===========================>..] - ETA: 0s - loss: 0.2760 - accuracy: 0.9493
23900/25000 [===========================>..] - ETA: 0s - loss: 0.2759 - accuracy: 0.9493
24000/25000 [===========================>..] - ETA: 0s - loss: 0.2763 - accuracy: 0.9491
24100/25000 [===========================>..] - ETA: 0s - loss: 0.2764 - accuracy: 0.9490
24200/25000 [============================>.] - ETA: 0s - loss: 0.2766 - accuracy: 0.9489
24300/25000 [============================>.] - ETA: 0s - loss: 0.2767 - accuracy: 0.9488
24400/25000 [============================>.] - ETA: 0s - loss: 0.2770 - accuracy: 0.9486
24500/25000 [============================>.] - ETA: 0s - loss: 0.2771 - accuracy: 0.9484
24600/25000 [============================>.] - ETA: 0s - loss: 0.2770 - accuracy: 0.9485
24700/25000 [============================>.] - ETA: 0s - loss: 0.2772 - accuracy: 0.9484
24800/25000 [============================>.] - ETA: 0s - loss: 0.2773 - accuracy: 0.9482
24900/25000 [============================>.] - ETA: 0s - loss: 0.2776 - accuracy: 0.9480
25000/25000 [==============================] - 19s 779us/step - loss: 0.2774 - accuracy: 0.9481 - val_loss: 0.4008 - val_accuracy: 0.8584
Epoch 8/10

  100/25000 [..............................] - ETA: 16s - loss: 0.3221 - accuracy: 0.9200
  200/25000 [..............................] - ETA: 15s - loss: 0.2981 - accuracy: 0.9300
  300/25000 [..............................] - ETA: 15s - loss: 0.2992 - accuracy: 0.9300
  400/25000 [..............................] - ETA: 15s - loss: 0.2945 - accuracy: 0.9350
  500/25000 [..............................] - ETA: 15s - loss: 0.2953 - accuracy: 0.9320
  600/25000 [..............................] - ETA: 15s - loss: 0.2872 - accuracy: 0.9383
  700/25000 [..............................] - ETA: 15s - loss: 0.2797 - accuracy: 0.9429
  800/25000 [..............................] - ETA: 15s - loss: 0.2724 - accuracy: 0.9475
  900/25000 [>.............................] - ETA: 15s - loss: 0.2711 - accuracy: 0.9489
 1000/25000 [>.............................] - ETA: 15s - loss: 0.2713 - accuracy: 0.9480
 1100/25000 [>.............................] - ETA: 16s - loss: 0.2717 - accuracy: 0.9473
 1200/25000 [>.............................] - ETA: 15s - loss: 0.2688 - accuracy: 0.9492
 1300/25000 [>.............................] - ETA: 15s - loss: 0.2663 - accuracy: 0.9508
 1400/25000 [>.............................] - ETA: 15s - loss: 0.2645 - accuracy: 0.9521
 1500/25000 [>.............................] - ETA: 15s - loss: 0.2652 - accuracy: 0.9520
 1600/25000 [>.............................] - ETA: 15s - loss: 0.2618 - accuracy: 0.9538
 1700/25000 [=>............................] - ETA: 15s - loss: 0.2590 - accuracy: 0.9559
 1800/25000 [=>............................] - ETA: 15s - loss: 0.2602 - accuracy: 0.9556
 1900/25000 [=>............................] - ETA: 15s - loss: 0.2599 - accuracy: 0.9563
 2000/25000 [=>............................] - ETA: 15s - loss: 0.2583 - accuracy: 0.9570
 2100/25000 [=>............................] - ETA: 15s - loss: 0.2579 - accuracy: 0.9571
 2200/25000 [=>............................] - ETA: 15s - loss: 0.2573 - accuracy: 0.9577
 2300/25000 [=>............................] - ETA: 14s - loss: 0.2590 - accuracy: 0.9561
 2400/25000 [=>............................] - ETA: 14s - loss: 0.2619 - accuracy: 0.9542
 2500/25000 [==>...........................] - ETA: 14s - loss: 0.2611 - accuracy: 0.9548
 2600/25000 [==>...........................] - ETA: 14s - loss: 0.2640 - accuracy: 0.9527
 2700/25000 [==>...........................] - ETA: 14s - loss: 0.2647 - accuracy: 0.9522
 2800/25000 [==>...........................] - ETA: 14s - loss: 0.2651 - accuracy: 0.9521
 2900/25000 [==>...........................] - ETA: 14s - loss: 0.2650 - accuracy: 0.9521
 3000/25000 [==>...........................] - ETA: 14s - loss: 0.2663 - accuracy: 0.9507
 3100/25000 [==>...........................] - ETA: 14s - loss: 0.2667 - accuracy: 0.9503
 3200/25000 [==>...........................] - ETA: 14s - loss: 0.2648 - accuracy: 0.9516
 3300/25000 [==>...........................] - ETA: 14s - loss: 0.2666 - accuracy: 0.9500
 3400/25000 [===>..........................] - ETA: 14s - loss: 0.2662 - accuracy: 0.9500
 3500/25000 [===>..........................] - ETA: 14s - loss: 0.2658 - accuracy: 0.9500
 3600/25000 [===>..........................] - ETA: 14s - loss: 0.2639 - accuracy: 0.9511
 3700/25000 [===>..........................] - ETA: 14s - loss: 0.2628 - accuracy: 0.9519
 3800/25000 [===>..........................] - ETA: 14s - loss: 0.2625 - accuracy: 0.9521
 3900/25000 [===>..........................] - ETA: 13s - loss: 0.2627 - accuracy: 0.9521
 4000/25000 [===>..........................] - ETA: 13s - loss: 0.2635 - accuracy: 0.9515
 4100/25000 [===>..........................] - ETA: 13s - loss: 0.2627 - accuracy: 0.9522
 4200/25000 [====>.........................] - ETA: 13s - loss: 0.2629 - accuracy: 0.9519
 4300/25000 [====>.........................] - ETA: 13s - loss: 0.2620 - accuracy: 0.9526
 4400/25000 [====>.........................] - ETA: 13s - loss: 0.2616 - accuracy: 0.9527
 4500/25000 [====>.........................] - ETA: 13s - loss: 0.2618 - accuracy: 0.9527
 4600/25000 [====>.........................] - ETA: 13s - loss: 0.2607 - accuracy: 0.9533
 4700/25000 [====>.........................] - ETA: 13s - loss: 0.2611 - accuracy: 0.9530
 4800/25000 [====>.........................] - ETA: 13s - loss: 0.2612 - accuracy: 0.9529
 4900/25000 [====>.........................] - ETA: 13s - loss: 0.2608 - accuracy: 0.9533
 5000/25000 [=====>........................] - ETA: 13s - loss: 0.2610 - accuracy: 0.9530
 5100/25000 [=====>........................] - ETA: 12s - loss: 0.2616 - accuracy: 0.9529
 5200/25000 [=====>........................] - ETA: 12s - loss: 0.2620 - accuracy: 0.9525
 5300/25000 [=====>........................] - ETA: 12s - loss: 0.2617 - accuracy: 0.9526
 5400/25000 [=====>........................] - ETA: 12s - loss: 0.2614 - accuracy: 0.9528
 5500/25000 [=====>........................] - ETA: 12s - loss: 0.2605 - accuracy: 0.9533
 5600/25000 [=====>........................] - ETA: 12s - loss: 0.2600 - accuracy: 0.9532
 5700/25000 [=====>........................] - ETA: 12s - loss: 0.2606 - accuracy: 0.9530
 5800/25000 [=====>........................] - ETA: 12s - loss: 0.2607 - accuracy: 0.9528
 5900/25000 [======>.......................] - ETA: 12s - loss: 0.2605 - accuracy: 0.9529
 6000/25000 [======>.......................] - ETA: 12s - loss: 0.2606 - accuracy: 0.9528
 6100/25000 [======>.......................] - ETA: 12s - loss: 0.2604 - accuracy: 0.9531
 6200/25000 [======>.......................] - ETA: 12s - loss: 0.2597 - accuracy: 0.9535
 6300/25000 [======>.......................] - ETA: 12s - loss: 0.2588 - accuracy: 0.9541
 6400/25000 [======>.......................] - ETA: 11s - loss: 0.2596 - accuracy: 0.9534
 6500/25000 [======>.......................] - ETA: 11s - loss: 0.2597 - accuracy: 0.9534
 6600/25000 [======>.......................] - ETA: 11s - loss: 0.2600 - accuracy: 0.9530
 6700/25000 [=======>......................] - ETA: 11s - loss: 0.2596 - accuracy: 0.9533
 6800/25000 [=======>......................] - ETA: 11s - loss: 0.2589 - accuracy: 0.9535
 6900/25000 [=======>......................] - ETA: 11s - loss: 0.2587 - accuracy: 0.9536
 7000/25000 [=======>......................] - ETA: 11s - loss: 0.2585 - accuracy: 0.9537
 7100/25000 [=======>......................] - ETA: 11s - loss: 0.2592 - accuracy: 0.9534
 7200/25000 [=======>......................] - ETA: 11s - loss: 0.2584 - accuracy: 0.9539
 7300/25000 [=======>......................] - ETA: 11s - loss: 0.2581 - accuracy: 0.9540
 7400/25000 [=======>......................] - ETA: 11s - loss: 0.2587 - accuracy: 0.9538
 7500/25000 [========>.....................] - ETA: 11s - loss: 0.2582 - accuracy: 0.9541
 7600/25000 [========>.....................] - ETA: 11s - loss: 0.2581 - accuracy: 0.9542
 7700/25000 [========>.....................] - ETA: 11s - loss: 0.2582 - accuracy: 0.9543
 7800/25000 [========>.....................] - ETA: 11s - loss: 0.2584 - accuracy: 0.9540
 7900/25000 [========>.....................] - ETA: 10s - loss: 0.2585 - accuracy: 0.9539
 8000/25000 [========>.....................] - ETA: 10s - loss: 0.2576 - accuracy: 0.9545
 8100/25000 [========>.....................] - ETA: 10s - loss: 0.2574 - accuracy: 0.9546
 8200/25000 [========>.....................] - ETA: 10s - loss: 0.2576 - accuracy: 0.9544
 8300/25000 [========>.....................] - ETA: 10s - loss: 0.2580 - accuracy: 0.9541
 8400/25000 [=========>....................] - ETA: 10s - loss: 0.2575 - accuracy: 0.9544
 8500/25000 [=========>....................] - ETA: 10s - loss: 0.2578 - accuracy: 0.9544
 8600/25000 [=========>....................] - ETA: 10s - loss: 0.2580 - accuracy: 0.9542
 8700/25000 [=========>....................] - ETA: 10s - loss: 0.2578 - accuracy: 0.9544
 8800/25000 [=========>....................] - ETA: 10s - loss: 0.2581 - accuracy: 0.9543
 8900/25000 [=========>....................] - ETA: 10s - loss: 0.2577 - accuracy: 0.9545
 9000/25000 [=========>....................] - ETA: 10s - loss: 0.2573 - accuracy: 0.9548
 9100/25000 [=========>....................] - ETA: 10s - loss: 0.2573 - accuracy: 0.9548
 9200/25000 [==========>...................] - ETA: 10s - loss: 0.2572 - accuracy: 0.9550
 9300/25000 [==========>...................] - ETA: 9s - loss: 0.2567 - accuracy: 0.9553 
 9400/25000 [==========>...................] - ETA: 9s - loss: 0.2570 - accuracy: 0.9551
 9500/25000 [==========>...................] - ETA: 9s - loss: 0.2568 - accuracy: 0.9551
 9600/25000 [==========>...................] - ETA: 9s - loss: 0.2573 - accuracy: 0.9548
 9700/25000 [==========>...................] - ETA: 9s - loss: 0.2573 - accuracy: 0.9548
 9800/25000 [==========>...................] - ETA: 9s - loss: 0.2572 - accuracy: 0.9548
 9900/25000 [==========>...................] - ETA: 9s - loss: 0.2573 - accuracy: 0.9546
10000/25000 [===========>..................] - ETA: 9s - loss: 0.2573 - accuracy: 0.9546
10100/25000 [===========>..................] - ETA: 9s - loss: 0.2579 - accuracy: 0.9542
10200/25000 [===========>..................] - ETA: 9s - loss: 0.2579 - accuracy: 0.9541
10300/25000 [===========>..................] - ETA: 9s - loss: 0.2580 - accuracy: 0.9541
10400/25000 [===========>..................] - ETA: 9s - loss: 0.2584 - accuracy: 0.9538
10500/25000 [===========>..................] - ETA: 9s - loss: 0.2587 - accuracy: 0.9535
10600/25000 [===========>..................] - ETA: 9s - loss: 0.2588 - accuracy: 0.9535
10700/25000 [===========>..................] - ETA: 9s - loss: 0.2585 - accuracy: 0.9536
10800/25000 [===========>..................] - ETA: 8s - loss: 0.2584 - accuracy: 0.9537
10900/25000 [============>.................] - ETA: 8s - loss: 0.2583 - accuracy: 0.9538
11000/25000 [============>.................] - ETA: 8s - loss: 0.2583 - accuracy: 0.9537
11100/25000 [============>.................] - ETA: 8s - loss: 0.2581 - accuracy: 0.9539
11200/25000 [============>.................] - ETA: 8s - loss: 0.2589 - accuracy: 0.9534
11300/25000 [============>.................] - ETA: 8s - loss: 0.2588 - accuracy: 0.9534
11400/25000 [============>.................] - ETA: 8s - loss: 0.2583 - accuracy: 0.9537
11500/25000 [============>.................] - ETA: 8s - loss: 0.2580 - accuracy: 0.9537
11600/25000 [============>.................] - ETA: 8s - loss: 0.2580 - accuracy: 0.9536
11700/25000 [=============>................] - ETA: 8s - loss: 0.2578 - accuracy: 0.9538
11800/25000 [=============>................] - ETA: 8s - loss: 0.2580 - accuracy: 0.9536
11900/25000 [=============>................] - ETA: 8s - loss: 0.2578 - accuracy: 0.9537
12000/25000 [=============>................] - ETA: 8s - loss: 0.2578 - accuracy: 0.9538
12100/25000 [=============>................] - ETA: 8s - loss: 0.2580 - accuracy: 0.9536
12200/25000 [=============>................] - ETA: 8s - loss: 0.2579 - accuracy: 0.9537
12300/25000 [=============>................] - ETA: 8s - loss: 0.2581 - accuracy: 0.9536
12400/25000 [=============>................] - ETA: 7s - loss: 0.2582 - accuracy: 0.9535
12500/25000 [==============>...............] - ETA: 7s - loss: 0.2578 - accuracy: 0.9536
12600/25000 [==============>...............] - ETA: 7s - loss: 0.2578 - accuracy: 0.9536
12700/25000 [==============>...............] - ETA: 7s - loss: 0.2576 - accuracy: 0.9536
12800/25000 [==============>...............] - ETA: 7s - loss: 0.2579 - accuracy: 0.9534
12900/25000 [==============>...............] - ETA: 7s - loss: 0.2578 - accuracy: 0.9535
13000/25000 [==============>...............] - ETA: 7s - loss: 0.2579 - accuracy: 0.9534
13100/25000 [==============>...............] - ETA: 7s - loss: 0.2578 - accuracy: 0.9534
13200/25000 [==============>...............] - ETA: 7s - loss: 0.2579 - accuracy: 0.9533
13300/25000 [==============>...............] - ETA: 7s - loss: 0.2579 - accuracy: 0.9532
13400/25000 [===============>..............] - ETA: 7s - loss: 0.2579 - accuracy: 0.9533
13500/25000 [===============>..............] - ETA: 7s - loss: 0.2577 - accuracy: 0.9533
13600/25000 [===============>..............] - ETA: 7s - loss: 0.2580 - accuracy: 0.9532
13700/25000 [===============>..............] - ETA: 7s - loss: 0.2580 - accuracy: 0.9531
13800/25000 [===============>..............] - ETA: 7s - loss: 0.2579 - accuracy: 0.9533
13900/25000 [===============>..............] - ETA: 6s - loss: 0.2577 - accuracy: 0.9534
14000/25000 [===============>..............] - ETA: 6s - loss: 0.2575 - accuracy: 0.9535
14100/25000 [===============>..............] - ETA: 6s - loss: 0.2576 - accuracy: 0.9534
14200/25000 [================>.............] - ETA: 6s - loss: 0.2574 - accuracy: 0.9535
14300/25000 [================>.............] - ETA: 6s - loss: 0.2574 - accuracy: 0.9534
14400/25000 [================>.............] - ETA: 6s - loss: 0.2573 - accuracy: 0.9535
14500/25000 [================>.............] - ETA: 6s - loss: 0.2572 - accuracy: 0.9535
14600/25000 [================>.............] - ETA: 6s - loss: 0.2574 - accuracy: 0.9534
14700/25000 [================>.............] - ETA: 6s - loss: 0.2571 - accuracy: 0.9535
14800/25000 [================>.............] - ETA: 6s - loss: 0.2567 - accuracy: 0.9537
14900/25000 [================>.............] - ETA: 6s - loss: 0.2567 - accuracy: 0.9536
15000/25000 [=================>............] - ETA: 6s - loss: 0.2573 - accuracy: 0.9533
15100/25000 [=================>............] - ETA: 6s - loss: 0.2570 - accuracy: 0.9534
15200/25000 [=================>............] - ETA: 6s - loss: 0.2572 - accuracy: 0.9532
15300/25000 [=================>............] - ETA: 6s - loss: 0.2574 - accuracy: 0.9531
15400/25000 [=================>............] - ETA: 6s - loss: 0.2576 - accuracy: 0.9529
15500/25000 [=================>............] - ETA: 5s - loss: 0.2575 - accuracy: 0.9529
15600/25000 [=================>............] - ETA: 5s - loss: 0.2574 - accuracy: 0.9529
15700/25000 [=================>............] - ETA: 5s - loss: 0.2572 - accuracy: 0.9531
15800/25000 [=================>............] - ETA: 5s - loss: 0.2569 - accuracy: 0.9532
15900/25000 [==================>...........] - ETA: 5s - loss: 0.2568 - accuracy: 0.9533
16000/25000 [==================>...........] - ETA: 5s - loss: 0.2570 - accuracy: 0.9531
16100/25000 [==================>...........] - ETA: 5s - loss: 0.2574 - accuracy: 0.9528
16200/25000 [==================>...........] - ETA: 5s - loss: 0.2570 - accuracy: 0.9530
16300/25000 [==================>...........] - ETA: 5s - loss: 0.2569 - accuracy: 0.9530
16400/25000 [==================>...........] - ETA: 5s - loss: 0.2570 - accuracy: 0.9529
16500/25000 [==================>...........] - ETA: 5s - loss: 0.2573 - accuracy: 0.9527
16600/25000 [==================>...........] - ETA: 5s - loss: 0.2571 - accuracy: 0.9528
16700/25000 [===================>..........] - ETA: 5s - loss: 0.2574 - accuracy: 0.9526
16800/25000 [===================>..........] - ETA: 5s - loss: 0.2570 - accuracy: 0.9529
16900/25000 [===================>..........] - ETA: 5s - loss: 0.2568 - accuracy: 0.9530
17000/25000 [===================>..........] - ETA: 5s - loss: 0.2567 - accuracy: 0.9530
17100/25000 [===================>..........] - ETA: 4s - loss: 0.2565 - accuracy: 0.9532
17200/25000 [===================>..........] - ETA: 4s - loss: 0.2563 - accuracy: 0.9532
17300/25000 [===================>..........] - ETA: 4s - loss: 0.2566 - accuracy: 0.9530
17400/25000 [===================>..........] - ETA: 4s - loss: 0.2567 - accuracy: 0.9529
17500/25000 [====================>.........] - ETA: 4s - loss: 0.2569 - accuracy: 0.9528
17600/25000 [====================>.........] - ETA: 4s - loss: 0.2568 - accuracy: 0.9528
17700/25000 [====================>.........] - ETA: 4s - loss: 0.2567 - accuracy: 0.9529
17800/25000 [====================>.........] - ETA: 4s - loss: 0.2567 - accuracy: 0.9529
17900/25000 [====================>.........] - ETA: 4s - loss: 0.2568 - accuracy: 0.9527
18000/25000 [====================>.........] - ETA: 4s - loss: 0.2567 - accuracy: 0.9528
18100/25000 [====================>.........] - ETA: 4s - loss: 0.2567 - accuracy: 0.9528
18200/25000 [====================>.........] - ETA: 4s - loss: 0.2568 - accuracy: 0.9526
18300/25000 [====================>.........] - ETA: 4s - loss: 0.2565 - accuracy: 0.9528
18400/25000 [=====================>........] - ETA: 4s - loss: 0.2565 - accuracy: 0.9528
18500/25000 [=====================>........] - ETA: 4s - loss: 0.2565 - accuracy: 0.9528
18600/25000 [=====================>........] - ETA: 4s - loss: 0.2562 - accuracy: 0.9530
18700/25000 [=====================>........] - ETA: 3s - loss: 0.2564 - accuracy: 0.9527
18800/25000 [=====================>........] - ETA: 3s - loss: 0.2564 - accuracy: 0.9528
18900/25000 [=====================>........] - ETA: 3s - loss: 0.2563 - accuracy: 0.9529
19000/25000 [=====================>........] - ETA: 3s - loss: 0.2569 - accuracy: 0.9524
19100/25000 [=====================>........] - ETA: 3s - loss: 0.2571 - accuracy: 0.9522
19200/25000 [======================>.......] - ETA: 3s - loss: 0.2569 - accuracy: 0.9522
19300/25000 [======================>.......] - ETA: 3s - loss: 0.2568 - accuracy: 0.9523
19400/25000 [======================>.......] - ETA: 3s - loss: 0.2569 - accuracy: 0.9523
19500/25000 [======================>.......] - ETA: 3s - loss: 0.2566 - accuracy: 0.9524
19600/25000 [======================>.......] - ETA: 3s - loss: 0.2568 - accuracy: 0.9523
19700/25000 [======================>.......] - ETA: 3s - loss: 0.2571 - accuracy: 0.9522
19800/25000 [======================>.......] - ETA: 3s - loss: 0.2575 - accuracy: 0.9519
19900/25000 [======================>.......] - ETA: 3s - loss: 0.2575 - accuracy: 0.9519
20000/25000 [=======================>......] - ETA: 3s - loss: 0.2577 - accuracy: 0.9518
20100/25000 [=======================>......] - ETA: 3s - loss: 0.2576 - accuracy: 0.9517
20200/25000 [=======================>......] - ETA: 3s - loss: 0.2576 - accuracy: 0.9517
20300/25000 [=======================>......] - ETA: 2s - loss: 0.2573 - accuracy: 0.9518
20400/25000 [=======================>......] - ETA: 2s - loss: 0.2574 - accuracy: 0.9518
20500/25000 [=======================>......] - ETA: 2s - loss: 0.2575 - accuracy: 0.9517
20600/25000 [=======================>......] - ETA: 2s - loss: 0.2574 - accuracy: 0.9518
20700/25000 [=======================>......] - ETA: 2s - loss: 0.2573 - accuracy: 0.9519
20800/25000 [=======================>......] - ETA: 2s - loss: 0.2576 - accuracy: 0.9517
20900/25000 [========================>.....] - ETA: 2s - loss: 0.2575 - accuracy: 0.9518
21000/25000 [========================>.....] - ETA: 2s - loss: 0.2577 - accuracy: 0.9517
21100/25000 [========================>.....] - ETA: 2s - loss: 0.2575 - accuracy: 0.9518
21200/25000 [========================>.....] - ETA: 2s - loss: 0.2576 - accuracy: 0.9517
21300/25000 [========================>.....] - ETA: 2s - loss: 0.2577 - accuracy: 0.9516
21400/25000 [========================>.....] - ETA: 2s - loss: 0.2574 - accuracy: 0.9518
21500/25000 [========================>.....] - ETA: 2s - loss: 0.2573 - accuracy: 0.9519
21600/25000 [========================>.....] - ETA: 2s - loss: 0.2570 - accuracy: 0.9520
21700/25000 [=========================>....] - ETA: 2s - loss: 0.2570 - accuracy: 0.9521
21800/25000 [=========================>....] - ETA: 2s - loss: 0.2570 - accuracy: 0.9521
21900/25000 [=========================>....] - ETA: 1s - loss: 0.2570 - accuracy: 0.9520
22000/25000 [=========================>....] - ETA: 1s - loss: 0.2570 - accuracy: 0.9520
22100/25000 [=========================>....] - ETA: 1s - loss: 0.2571 - accuracy: 0.9519
22200/25000 [=========================>....] - ETA: 1s - loss: 0.2571 - accuracy: 0.9520
22300/25000 [=========================>....] - ETA: 1s - loss: 0.2568 - accuracy: 0.9521
22400/25000 [=========================>....] - ETA: 1s - loss: 0.2566 - accuracy: 0.9523
22500/25000 [==========================>...] - ETA: 1s - loss: 0.2566 - accuracy: 0.9522
22600/25000 [==========================>...] - ETA: 1s - loss: 0.2568 - accuracy: 0.9521
22700/25000 [==========================>...] - ETA: 1s - loss: 0.2566 - accuracy: 0.9522
22800/25000 [==========================>...] - ETA: 1s - loss: 0.2568 - accuracy: 0.9521
22900/25000 [==========================>...] - ETA: 1s - loss: 0.2569 - accuracy: 0.9521
23000/25000 [==========================>...] - ETA: 1s - loss: 0.2569 - accuracy: 0.9520
23100/25000 [==========================>...] - ETA: 1s - loss: 0.2571 - accuracy: 0.9519
23200/25000 [==========================>...] - ETA: 1s - loss: 0.2569 - accuracy: 0.9520
23300/25000 [==========================>...] - ETA: 1s - loss: 0.2568 - accuracy: 0.9521
23400/25000 [===========================>..] - ETA: 1s - loss: 0.2570 - accuracy: 0.9520
23500/25000 [===========================>..] - ETA: 0s - loss: 0.2571 - accuracy: 0.9519
23600/25000 [===========================>..] - ETA: 0s - loss: 0.2571 - accuracy: 0.9519
23700/25000 [===========================>..] - ETA: 0s - loss: 0.2571 - accuracy: 0.9519
23800/25000 [===========================>..] - ETA: 0s - loss: 0.2572 - accuracy: 0.9518
23900/25000 [===========================>..] - ETA: 0s - loss: 0.2573 - accuracy: 0.9517
24000/25000 [===========================>..] - ETA: 0s - loss: 0.2573 - accuracy: 0.9517
24100/25000 [===========================>..] - ETA: 0s - loss: 0.2573 - accuracy: 0.9517
24200/25000 [============================>.] - ETA: 0s - loss: 0.2575 - accuracy: 0.9515
24300/25000 [============================>.] - ETA: 0s - loss: 0.2575 - accuracy: 0.9515
24400/25000 [============================>.] - ETA: 0s - loss: 0.2575 - accuracy: 0.9515
24500/25000 [============================>.] - ETA: 0s - loss: 0.2575 - accuracy: 0.9515
24600/25000 [============================>.] - ETA: 0s - loss: 0.2574 - accuracy: 0.9515
24700/25000 [============================>.] - ETA: 0s - loss: 0.2572 - accuracy: 0.9516
24800/25000 [============================>.] - ETA: 0s - loss: 0.2570 - accuracy: 0.9518
24900/25000 [============================>.] - ETA: 0s - loss: 0.2570 - accuracy: 0.9517
25000/25000 [==============================] - 20s 785us/step - loss: 0.2570 - accuracy: 0.9517 - val_loss: 0.3963 - val_accuracy: 0.8586
Epoch 9/10

  100/25000 [..............................] - ETA: 14s - loss: 0.1880 - accuracy: 0.9700
  200/25000 [..............................] - ETA: 14s - loss: 0.2104 - accuracy: 0.9650
  300/25000 [..............................] - ETA: 14s - loss: 0.2084 - accuracy: 0.9700
  400/25000 [..............................] - ETA: 15s - loss: 0.2149 - accuracy: 0.9675
  500/25000 [..............................] - ETA: 15s - loss: 0.2192 - accuracy: 0.9660
  600/25000 [..............................] - ETA: 15s - loss: 0.2168 - accuracy: 0.9683
  700/25000 [..............................] - ETA: 14s - loss: 0.2203 - accuracy: 0.9671
  800/25000 [..............................] - ETA: 14s - loss: 0.2221 - accuracy: 0.9650
  900/25000 [>.............................] - ETA: 14s - loss: 0.2194 - accuracy: 0.9667
 1000/25000 [>.............................] - ETA: 14s - loss: 0.2208 - accuracy: 0.9670
 1100/25000 [>.............................] - ETA: 14s - loss: 0.2206 - accuracy: 0.9682
 1200/25000 [>.............................] - ETA: 14s - loss: 0.2199 - accuracy: 0.9692
 1300/25000 [>.............................] - ETA: 14s - loss: 0.2221 - accuracy: 0.9669
 1400/25000 [>.............................] - ETA: 14s - loss: 0.2209 - accuracy: 0.9679
 1500/25000 [>.............................] - ETA: 14s - loss: 0.2236 - accuracy: 0.9667
 1600/25000 [>.............................] - ETA: 14s - loss: 0.2219 - accuracy: 0.9681
 1700/25000 [=>............................] - ETA: 14s - loss: 0.2214 - accuracy: 0.9682
 1800/25000 [=>............................] - ETA: 14s - loss: 0.2223 - accuracy: 0.9683
 1900/25000 [=>............................] - ETA: 14s - loss: 0.2233 - accuracy: 0.9684
 2000/25000 [=>............................] - ETA: 14s - loss: 0.2229 - accuracy: 0.9680
 2100/25000 [=>............................] - ETA: 14s - loss: 0.2256 - accuracy: 0.9662
 2200/25000 [=>............................] - ETA: 14s - loss: 0.2257 - accuracy: 0.9664
 2300/25000 [=>............................] - ETA: 13s - loss: 0.2275 - accuracy: 0.9652
 2400/25000 [=>............................] - ETA: 13s - loss: 0.2283 - accuracy: 0.9646
 2500/25000 [==>...........................] - ETA: 13s - loss: 0.2285 - accuracy: 0.9648
 2600/25000 [==>...........................] - ETA: 13s - loss: 0.2286 - accuracy: 0.9646
 2700/25000 [==>...........................] - ETA: 13s - loss: 0.2286 - accuracy: 0.9644
 2800/25000 [==>...........................] - ETA: 13s - loss: 0.2283 - accuracy: 0.9646
 2900/25000 [==>...........................] - ETA: 13s - loss: 0.2275 - accuracy: 0.9652
 3000/25000 [==>...........................] - ETA: 13s - loss: 0.2272 - accuracy: 0.9650
 3100/25000 [==>...........................] - ETA: 13s - loss: 0.2274 - accuracy: 0.9645
 3200/25000 [==>...........................] - ETA: 13s - loss: 0.2278 - accuracy: 0.9644
 3300/25000 [==>...........................] - ETA: 13s - loss: 0.2271 - accuracy: 0.9648
 3400/25000 [===>..........................] - ETA: 13s - loss: 0.2272 - accuracy: 0.9650
 3500/25000 [===>..........................] - ETA: 13s - loss: 0.2285 - accuracy: 0.9643
 3600/25000 [===>..........................] - ETA: 13s - loss: 0.2295 - accuracy: 0.9636
 3700/25000 [===>..........................] - ETA: 13s - loss: 0.2297 - accuracy: 0.9635
 3800/25000 [===>..........................] - ETA: 12s - loss: 0.2298 - accuracy: 0.9634
 3900/25000 [===>..........................] - ETA: 12s - loss: 0.2301 - accuracy: 0.9633
 4000/25000 [===>..........................] - ETA: 12s - loss: 0.2310 - accuracy: 0.9628
 4100/25000 [===>..........................] - ETA: 12s - loss: 0.2305 - accuracy: 0.9632
 4200/25000 [====>.........................] - ETA: 12s - loss: 0.2311 - accuracy: 0.9626
 4300/25000 [====>.........................] - ETA: 12s - loss: 0.2322 - accuracy: 0.9619
 4400/25000 [====>.........................] - ETA: 12s - loss: 0.2313 - accuracy: 0.9623
 4500/25000 [====>.........................] - ETA: 12s - loss: 0.2313 - accuracy: 0.9622
 4600/25000 [====>.........................] - ETA: 12s - loss: 0.2310 - accuracy: 0.9624
 4700/25000 [====>.........................] - ETA: 12s - loss: 0.2309 - accuracy: 0.9623
 4800/25000 [====>.........................] - ETA: 12s - loss: 0.2306 - accuracy: 0.9625
 4900/25000 [====>.........................] - ETA: 12s - loss: 0.2307 - accuracy: 0.9624
 5000/25000 [=====>........................] - ETA: 12s - loss: 0.2316 - accuracy: 0.9618
 5100/25000 [=====>........................] - ETA: 12s - loss: 0.2320 - accuracy: 0.9618
 5200/25000 [=====>........................] - ETA: 12s - loss: 0.2324 - accuracy: 0.9615
 5300/25000 [=====>........................] - ETA: 12s - loss: 0.2324 - accuracy: 0.9617
 5400/25000 [=====>........................] - ETA: 12s - loss: 0.2321 - accuracy: 0.9620
 5500/25000 [=====>........................] - ETA: 11s - loss: 0.2322 - accuracy: 0.9618
 5600/25000 [=====>........................] - ETA: 11s - loss: 0.2330 - accuracy: 0.9616
 5700/25000 [=====>........................] - ETA: 11s - loss: 0.2328 - accuracy: 0.9618
 5800/25000 [=====>........................] - ETA: 11s - loss: 0.2324 - accuracy: 0.9621
 5900/25000 [======>.......................] - ETA: 11s - loss: 0.2325 - accuracy: 0.9620
 6000/25000 [======>.......................] - ETA: 11s - loss: 0.2320 - accuracy: 0.9623
 6100/25000 [======>.......................] - ETA: 11s - loss: 0.2319 - accuracy: 0.9623
 6200/25000 [======>.......................] - ETA: 11s - loss: 0.2314 - accuracy: 0.9626
 6300/25000 [======>.......................] - ETA: 11s - loss: 0.2317 - accuracy: 0.9622
 6400/25000 [======>.......................] - ETA: 11s - loss: 0.2326 - accuracy: 0.9616
 6500/25000 [======>.......................] - ETA: 11s - loss: 0.2325 - accuracy: 0.9615
 6600/25000 [======>.......................] - ETA: 11s - loss: 0.2325 - accuracy: 0.9615
 6700/25000 [=======>......................] - ETA: 11s - loss: 0.2321 - accuracy: 0.9618
 6800/25000 [=======>......................] - ETA: 11s - loss: 0.2325 - accuracy: 0.9616
 6900/25000 [=======>......................] - ETA: 11s - loss: 0.2328 - accuracy: 0.9616
 7000/25000 [=======>......................] - ETA: 11s - loss: 0.2325 - accuracy: 0.9617
 7100/25000 [=======>......................] - ETA: 10s - loss: 0.2326 - accuracy: 0.9615
 7200/25000 [=======>......................] - ETA: 10s - loss: 0.2323 - accuracy: 0.9618
 7300/25000 [=======>......................] - ETA: 10s - loss: 0.2316 - accuracy: 0.9622
 7400/25000 [=======>......................] - ETA: 10s - loss: 0.2328 - accuracy: 0.9615
 7500/25000 [========>.....................] - ETA: 10s - loss: 0.2321 - accuracy: 0.9619
 7600/25000 [========>.....................] - ETA: 10s - loss: 0.2329 - accuracy: 0.9614
 7700/25000 [========>.....................] - ETA: 10s - loss: 0.2325 - accuracy: 0.9617
 7800/25000 [========>.....................] - ETA: 10s - loss: 0.2330 - accuracy: 0.9614
 7900/25000 [========>.....................] - ETA: 10s - loss: 0.2330 - accuracy: 0.9613
 8000/25000 [========>.....................] - ETA: 10s - loss: 0.2327 - accuracy: 0.9615
 8100/25000 [========>.....................] - ETA: 10s - loss: 0.2326 - accuracy: 0.9615
 8200/25000 [========>.....................] - ETA: 10s - loss: 0.2322 - accuracy: 0.9618
 8300/25000 [========>.....................] - ETA: 10s - loss: 0.2327 - accuracy: 0.9616
 8400/25000 [=========>....................] - ETA: 10s - loss: 0.2327 - accuracy: 0.9617
 8500/25000 [=========>....................] - ETA: 10s - loss: 0.2328 - accuracy: 0.9615
 8600/25000 [=========>....................] - ETA: 10s - loss: 0.2331 - accuracy: 0.9615
 8700/25000 [=========>....................] - ETA: 9s - loss: 0.2324 - accuracy: 0.9620 
 8800/25000 [=========>....................] - ETA: 9s - loss: 0.2318 - accuracy: 0.9623
 8900/25000 [=========>....................] - ETA: 9s - loss: 0.2316 - accuracy: 0.9624
 9000/25000 [=========>....................] - ETA: 9s - loss: 0.2314 - accuracy: 0.9623
 9100/25000 [=========>....................] - ETA: 9s - loss: 0.2315 - accuracy: 0.9622
 9200/25000 [==========>...................] - ETA: 9s - loss: 0.2316 - accuracy: 0.9623
 9300/25000 [==========>...................] - ETA: 9s - loss: 0.2321 - accuracy: 0.9619
 9400/25000 [==========>...................] - ETA: 9s - loss: 0.2318 - accuracy: 0.9621
 9500/25000 [==========>...................] - ETA: 9s - loss: 0.2324 - accuracy: 0.9618
 9600/25000 [==========>...................] - ETA: 9s - loss: 0.2325 - accuracy: 0.9617
 9700/25000 [==========>...................] - ETA: 9s - loss: 0.2327 - accuracy: 0.9614
 9800/25000 [==========>...................] - ETA: 9s - loss: 0.2334 - accuracy: 0.9610
 9900/25000 [==========>...................] - ETA: 9s - loss: 0.2331 - accuracy: 0.9612
10000/25000 [===========>..................] - ETA: 9s - loss: 0.2333 - accuracy: 0.9611
10100/25000 [===========>..................] - ETA: 9s - loss: 0.2332 - accuracy: 0.9611
10200/25000 [===========>..................] - ETA: 9s - loss: 0.2328 - accuracy: 0.9614
10300/25000 [===========>..................] - ETA: 9s - loss: 0.2327 - accuracy: 0.9614
10400/25000 [===========>..................] - ETA: 8s - loss: 0.2329 - accuracy: 0.9613
10500/25000 [===========>..................] - ETA: 8s - loss: 0.2335 - accuracy: 0.9610
10600/25000 [===========>..................] - ETA: 8s - loss: 0.2335 - accuracy: 0.9608
10700/25000 [===========>..................] - ETA: 8s - loss: 0.2338 - accuracy: 0.9607
10800/25000 [===========>..................] - ETA: 8s - loss: 0.2337 - accuracy: 0.9606
10900/25000 [============>.................] - ETA: 8s - loss: 0.2333 - accuracy: 0.9608
11000/25000 [============>.................] - ETA: 8s - loss: 0.2333 - accuracy: 0.9608
11100/25000 [============>.................] - ETA: 8s - loss: 0.2332 - accuracy: 0.9608
11200/25000 [============>.................] - ETA: 8s - loss: 0.2328 - accuracy: 0.9609
11300/25000 [============>.................] - ETA: 8s - loss: 0.2329 - accuracy: 0.9609
11400/25000 [============>.................] - ETA: 8s - loss: 0.2325 - accuracy: 0.9611
11500/25000 [============>.................] - ETA: 8s - loss: 0.2330 - accuracy: 0.9608
11600/25000 [============>.................] - ETA: 8s - loss: 0.2331 - accuracy: 0.9606
11700/25000 [=============>................] - ETA: 8s - loss: 0.2332 - accuracy: 0.9606
11800/25000 [=============>................] - ETA: 8s - loss: 0.2330 - accuracy: 0.9607
11900/25000 [=============>................] - ETA: 8s - loss: 0.2329 - accuracy: 0.9607
12000/25000 [=============>................] - ETA: 7s - loss: 0.2334 - accuracy: 0.9603
12100/25000 [=============>................] - ETA: 7s - loss: 0.2333 - accuracy: 0.9604
12200/25000 [=============>................] - ETA: 7s - loss: 0.2335 - accuracy: 0.9602
12300/25000 [=============>................] - ETA: 7s - loss: 0.2335 - accuracy: 0.9602
12400/25000 [=============>................] - ETA: 7s - loss: 0.2341 - accuracy: 0.9599
12500/25000 [==============>...............] - ETA: 7s - loss: 0.2346 - accuracy: 0.9595
12600/25000 [==============>...............] - ETA: 7s - loss: 0.2346 - accuracy: 0.9595
12700/25000 [==============>...............] - ETA: 7s - loss: 0.2347 - accuracy: 0.9594
12800/25000 [==============>...............] - ETA: 7s - loss: 0.2350 - accuracy: 0.9592
12900/25000 [==============>...............] - ETA: 7s - loss: 0.2354 - accuracy: 0.9590
13000/25000 [==============>...............] - ETA: 7s - loss: 0.2356 - accuracy: 0.9588
13100/25000 [==============>...............] - ETA: 7s - loss: 0.2355 - accuracy: 0.9589
13200/25000 [==============>...............] - ETA: 7s - loss: 0.2354 - accuracy: 0.9589
13300/25000 [==============>...............] - ETA: 7s - loss: 0.2354 - accuracy: 0.9589
13400/25000 [===============>..............] - ETA: 7s - loss: 0.2353 - accuracy: 0.9590
13500/25000 [===============>..............] - ETA: 7s - loss: 0.2353 - accuracy: 0.9590
13600/25000 [===============>..............] - ETA: 7s - loss: 0.2354 - accuracy: 0.9590
13700/25000 [===============>..............] - ETA: 6s - loss: 0.2352 - accuracy: 0.9591
13800/25000 [===============>..............] - ETA: 6s - loss: 0.2352 - accuracy: 0.9591
13900/25000 [===============>..............] - ETA: 6s - loss: 0.2351 - accuracy: 0.9591
14000/25000 [===============>..............] - ETA: 6s - loss: 0.2354 - accuracy: 0.9590
14100/25000 [===============>..............] - ETA: 6s - loss: 0.2352 - accuracy: 0.9590
14200/25000 [================>.............] - ETA: 6s - loss: 0.2352 - accuracy: 0.9590
14300/25000 [================>.............] - ETA: 6s - loss: 0.2355 - accuracy: 0.9589
14400/25000 [================>.............] - ETA: 6s - loss: 0.2357 - accuracy: 0.9587
14500/25000 [================>.............] - ETA: 6s - loss: 0.2357 - accuracy: 0.9587
14600/25000 [================>.............] - ETA: 6s - loss: 0.2359 - accuracy: 0.9585
14700/25000 [================>.............] - ETA: 6s - loss: 0.2356 - accuracy: 0.9586
14800/25000 [================>.............] - ETA: 6s - loss: 0.2352 - accuracy: 0.9589
14900/25000 [================>.............] - ETA: 6s - loss: 0.2355 - accuracy: 0.9587
15000/25000 [=================>............] - ETA: 6s - loss: 0.2352 - accuracy: 0.9589
15100/25000 [=================>............] - ETA: 6s - loss: 0.2354 - accuracy: 0.9587
15200/25000 [=================>............] - ETA: 6s - loss: 0.2357 - accuracy: 0.9586
15300/25000 [=================>............] - ETA: 5s - loss: 0.2357 - accuracy: 0.9585
15400/25000 [=================>............] - ETA: 5s - loss: 0.2359 - accuracy: 0.9584
15500/25000 [=================>............] - ETA: 5s - loss: 0.2357 - accuracy: 0.9585
15600/25000 [=================>............] - ETA: 5s - loss: 0.2355 - accuracy: 0.9587
15700/25000 [=================>............] - ETA: 5s - loss: 0.2357 - accuracy: 0.9586
15800/25000 [=================>............] - ETA: 5s - loss: 0.2358 - accuracy: 0.9585
15900/25000 [==================>...........] - ETA: 5s - loss: 0.2359 - accuracy: 0.9584
16000/25000 [==================>...........] - ETA: 5s - loss: 0.2356 - accuracy: 0.9586
16100/25000 [==================>...........] - ETA: 5s - loss: 0.2354 - accuracy: 0.9586
16200/25000 [==================>...........] - ETA: 5s - loss: 0.2354 - accuracy: 0.9586
16300/25000 [==================>...........] - ETA: 5s - loss: 0.2352 - accuracy: 0.9587
16400/25000 [==================>...........] - ETA: 5s - loss: 0.2350 - accuracy: 0.9588
16500/25000 [==================>...........] - ETA: 5s - loss: 0.2352 - accuracy: 0.9587
16600/25000 [==================>...........] - ETA: 5s - loss: 0.2352 - accuracy: 0.9587
16700/25000 [===================>..........] - ETA: 5s - loss: 0.2352 - accuracy: 0.9587
16800/25000 [===================>..........] - ETA: 5s - loss: 0.2351 - accuracy: 0.9588
16900/25000 [===================>..........] - ETA: 5s - loss: 0.2351 - accuracy: 0.9588
17000/25000 [===================>..........] - ETA: 4s - loss: 0.2352 - accuracy: 0.9587
17100/25000 [===================>..........] - ETA: 4s - loss: 0.2350 - accuracy: 0.9588
17200/25000 [===================>..........] - ETA: 4s - loss: 0.2351 - accuracy: 0.9588
17300/25000 [===================>..........] - ETA: 4s - loss: 0.2354 - accuracy: 0.9585
17400/25000 [===================>..........] - ETA: 4s - loss: 0.2355 - accuracy: 0.9584
17500/25000 [====================>.........] - ETA: 4s - loss: 0.2356 - accuracy: 0.9584
17600/25000 [====================>.........] - ETA: 4s - loss: 0.2355 - accuracy: 0.9584
17700/25000 [====================>.........] - ETA: 4s - loss: 0.2353 - accuracy: 0.9585
17800/25000 [====================>.........] - ETA: 4s - loss: 0.2354 - accuracy: 0.9585
17900/25000 [====================>.........] - ETA: 4s - loss: 0.2357 - accuracy: 0.9583
18000/25000 [====================>.........] - ETA: 4s - loss: 0.2357 - accuracy: 0.9583
18100/25000 [====================>.........] - ETA: 4s - loss: 0.2357 - accuracy: 0.9582
18200/25000 [====================>.........] - ETA: 4s - loss: 0.2359 - accuracy: 0.9581
18300/25000 [====================>.........] - ETA: 4s - loss: 0.2357 - accuracy: 0.9581
18400/25000 [=====================>........] - ETA: 4s - loss: 0.2360 - accuracy: 0.9580
18500/25000 [=====================>........] - ETA: 4s - loss: 0.2359 - accuracy: 0.9579
18600/25000 [=====================>........] - ETA: 3s - loss: 0.2359 - accuracy: 0.9580
18700/25000 [=====================>........] - ETA: 3s - loss: 0.2357 - accuracy: 0.9581
18800/25000 [=====================>........] - ETA: 3s - loss: 0.2359 - accuracy: 0.9579
18900/25000 [=====================>........] - ETA: 3s - loss: 0.2361 - accuracy: 0.9578
19000/25000 [=====================>........] - ETA: 3s - loss: 0.2362 - accuracy: 0.9578
19100/25000 [=====================>........] - ETA: 3s - loss: 0.2361 - accuracy: 0.9579
19200/25000 [======================>.......] - ETA: 3s - loss: 0.2360 - accuracy: 0.9579
19300/25000 [======================>.......] - ETA: 3s - loss: 0.2364 - accuracy: 0.9576
19400/25000 [======================>.......] - ETA: 3s - loss: 0.2363 - accuracy: 0.9576
19500/25000 [======================>.......] - ETA: 3s - loss: 0.2363 - accuracy: 0.9576
19600/25000 [======================>.......] - ETA: 3s - loss: 0.2361 - accuracy: 0.9577
19700/25000 [======================>.......] - ETA: 3s - loss: 0.2362 - accuracy: 0.9577
19800/25000 [======================>.......] - ETA: 3s - loss: 0.2364 - accuracy: 0.9576
19900/25000 [======================>.......] - ETA: 3s - loss: 0.2363 - accuracy: 0.9576
20000/25000 [=======================>......] - ETA: 3s - loss: 0.2363 - accuracy: 0.9575
20100/25000 [=======================>......] - ETA: 3s - loss: 0.2362 - accuracy: 0.9576
20200/25000 [=======================>......] - ETA: 2s - loss: 0.2364 - accuracy: 0.9575
20300/25000 [=======================>......] - ETA: 2s - loss: 0.2366 - accuracy: 0.9573
20400/25000 [=======================>......] - ETA: 2s - loss: 0.2364 - accuracy: 0.9575
20500/25000 [=======================>......] - ETA: 2s - loss: 0.2366 - accuracy: 0.9573
20600/25000 [=======================>......] - ETA: 2s - loss: 0.2370 - accuracy: 0.9571
20700/25000 [=======================>......] - ETA: 2s - loss: 0.2373 - accuracy: 0.9569
20800/25000 [=======================>......] - ETA: 2s - loss: 0.2375 - accuracy: 0.9568
20900/25000 [========================>.....] - ETA: 2s - loss: 0.2375 - accuracy: 0.9567
21000/25000 [========================>.....] - ETA: 2s - loss: 0.2374 - accuracy: 0.9568
21100/25000 [========================>.....] - ETA: 2s - loss: 0.2371 - accuracy: 0.9569
21200/25000 [========================>.....] - ETA: 2s - loss: 0.2373 - accuracy: 0.9568
21300/25000 [========================>.....] - ETA: 2s - loss: 0.2374 - accuracy: 0.9568
21400/25000 [========================>.....] - ETA: 2s - loss: 0.2374 - accuracy: 0.9567
21500/25000 [========================>.....] - ETA: 2s - loss: 0.2375 - accuracy: 0.9567
21600/25000 [========================>.....] - ETA: 2s - loss: 0.2375 - accuracy: 0.9567
21700/25000 [=========================>....] - ETA: 2s - loss: 0.2375 - accuracy: 0.9566
21800/25000 [=========================>....] - ETA: 1s - loss: 0.2377 - accuracy: 0.9565
21900/25000 [=========================>....] - ETA: 1s - loss: 0.2376 - accuracy: 0.9566
22000/25000 [=========================>....] - ETA: 1s - loss: 0.2375 - accuracy: 0.9566
22100/25000 [=========================>....] - ETA: 1s - loss: 0.2373 - accuracy: 0.9567
22200/25000 [=========================>....] - ETA: 1s - loss: 0.2372 - accuracy: 0.9567
22300/25000 [=========================>....] - ETA: 1s - loss: 0.2377 - accuracy: 0.9564
22400/25000 [=========================>....] - ETA: 1s - loss: 0.2378 - accuracy: 0.9563
22500/25000 [==========================>...] - ETA: 1s - loss: 0.2379 - accuracy: 0.9562
22600/25000 [==========================>...] - ETA: 1s - loss: 0.2378 - accuracy: 0.9562
22700/25000 [==========================>...] - ETA: 1s - loss: 0.2377 - accuracy: 0.9563
22800/25000 [==========================>...] - ETA: 1s - loss: 0.2377 - accuracy: 0.9563
22900/25000 [==========================>...] - ETA: 1s - loss: 0.2381 - accuracy: 0.9560
23000/25000 [==========================>...] - ETA: 1s - loss: 0.2381 - accuracy: 0.9560
23100/25000 [==========================>...] - ETA: 1s - loss: 0.2382 - accuracy: 0.9559
23200/25000 [==========================>...] - ETA: 1s - loss: 0.2383 - accuracy: 0.9559
23300/25000 [==========================>...] - ETA: 1s - loss: 0.2383 - accuracy: 0.9559
23400/25000 [===========================>..] - ETA: 0s - loss: 0.2383 - accuracy: 0.9558
23500/25000 [===========================>..] - ETA: 0s - loss: 0.2381 - accuracy: 0.9560
23600/25000 [===========================>..] - ETA: 0s - loss: 0.2383 - accuracy: 0.9558
23700/25000 [===========================>..] - ETA: 0s - loss: 0.2383 - accuracy: 0.9558
23800/25000 [===========================>..] - ETA: 0s - loss: 0.2385 - accuracy: 0.9557
23900/25000 [===========================>..] - ETA: 0s - loss: 0.2388 - accuracy: 0.9555
24000/25000 [===========================>..] - ETA: 0s - loss: 0.2388 - accuracy: 0.9555
24100/25000 [===========================>..] - ETA: 0s - loss: 0.2389 - accuracy: 0.9554
24200/25000 [============================>.] - ETA: 0s - loss: 0.2388 - accuracy: 0.9555
24300/25000 [============================>.] - ETA: 0s - loss: 0.2387 - accuracy: 0.9555
24400/25000 [============================>.] - ETA: 0s - loss: 0.2386 - accuracy: 0.9555
24500/25000 [============================>.] - ETA: 0s - loss: 0.2384 - accuracy: 0.9557
24600/25000 [============================>.] - ETA: 0s - loss: 0.2387 - accuracy: 0.9554
24700/25000 [============================>.] - ETA: 0s - loss: 0.2387 - accuracy: 0.9555
24800/25000 [============================>.] - ETA: 0s - loss: 0.2389 - accuracy: 0.9552
24900/25000 [============================>.] - ETA: 0s - loss: 0.2388 - accuracy: 0.9553
25000/25000 [==============================] - 19s 777us/step - loss: 0.2386 - accuracy: 0.9554 - val_loss: 0.4015 - val_accuracy: 0.8564
Epoch 10/10

  100/25000 [..............................] - ETA: 15s - loss: 0.2452 - accuracy: 0.9500
  200/25000 [..............................] - ETA: 15s - loss: 0.2711 - accuracy: 0.9350
  300/25000 [..............................] - ETA: 14s - loss: 0.2378 - accuracy: 0.9533
  400/25000 [..............................] - ETA: 15s - loss: 0.2334 - accuracy: 0.9550
  500/25000 [..............................] - ETA: 15s - loss: 0.2426 - accuracy: 0.9500
  600/25000 [..............................] - ETA: 15s - loss: 0.2463 - accuracy: 0.9500
  700/25000 [..............................] - ETA: 15s - loss: 0.2372 - accuracy: 0.9557
  800/25000 [..............................] - ETA: 15s - loss: 0.2358 - accuracy: 0.9563
  900/25000 [>.............................] - ETA: 15s - loss: 0.2336 - accuracy: 0.9578
 1000/25000 [>.............................] - ETA: 15s - loss: 0.2289 - accuracy: 0.9600
 1100/25000 [>.............................] - ETA: 15s - loss: 0.2280 - accuracy: 0.9609
 1200/25000 [>.............................] - ETA: 15s - loss: 0.2312 - accuracy: 0.9592
 1300/25000 [>.............................] - ETA: 14s - loss: 0.2298 - accuracy: 0.9600
 1400/25000 [>.............................] - ETA: 14s - loss: 0.2285 - accuracy: 0.9607
 1500/25000 [>.............................] - ETA: 14s - loss: 0.2244 - accuracy: 0.9633
 1600/25000 [>.............................] - ETA: 14s - loss: 0.2228 - accuracy: 0.9638
 1700/25000 [=>............................] - ETA: 14s - loss: 0.2212 - accuracy: 0.9647
 1800/25000 [=>............................] - ETA: 14s - loss: 0.2214 - accuracy: 0.9650
 1900/25000 [=>............................] - ETA: 14s - loss: 0.2230 - accuracy: 0.9642
 2000/25000 [=>............................] - ETA: 14s - loss: 0.2219 - accuracy: 0.9645
 2100/25000 [=>............................] - ETA: 14s - loss: 0.2212 - accuracy: 0.9643
 2200/25000 [=>............................] - ETA: 14s - loss: 0.2214 - accuracy: 0.9641
 2300/25000 [=>............................] - ETA: 14s - loss: 0.2202 - accuracy: 0.9648
 2400/25000 [=>............................] - ETA: 14s - loss: 0.2218 - accuracy: 0.9638
 2500/25000 [==>...........................] - ETA: 14s - loss: 0.2230 - accuracy: 0.9628
 2600/25000 [==>...........................] - ETA: 13s - loss: 0.2232 - accuracy: 0.9627
 2700/25000 [==>...........................] - ETA: 13s - loss: 0.2225 - accuracy: 0.9630
 2800/25000 [==>...........................] - ETA: 13s - loss: 0.2218 - accuracy: 0.9636
 2900/25000 [==>...........................] - ETA: 13s - loss: 0.2215 - accuracy: 0.9634
 3000/25000 [==>...........................] - ETA: 13s - loss: 0.2213 - accuracy: 0.9633
 3100/25000 [==>...........................] - ETA: 13s - loss: 0.2217 - accuracy: 0.9626
 3200/25000 [==>...........................] - ETA: 13s - loss: 0.2227 - accuracy: 0.9619
 3300/25000 [==>...........................] - ETA: 13s - loss: 0.2234 - accuracy: 0.9615
 3400/25000 [===>..........................] - ETA: 13s - loss: 0.2240 - accuracy: 0.9612
 3500/25000 [===>..........................] - ETA: 13s - loss: 0.2240 - accuracy: 0.9611
 3600/25000 [===>..........................] - ETA: 13s - loss: 0.2227 - accuracy: 0.9617
 3700/25000 [===>..........................] - ETA: 13s - loss: 0.2231 - accuracy: 0.9614
 3800/25000 [===>..........................] - ETA: 13s - loss: 0.2242 - accuracy: 0.9608
 3900/25000 [===>..........................] - ETA: 13s - loss: 0.2241 - accuracy: 0.9605
 4000/25000 [===>..........................] - ETA: 12s - loss: 0.2238 - accuracy: 0.9607
 4100/25000 [===>..........................] - ETA: 12s - loss: 0.2248 - accuracy: 0.9600
 4200/25000 [====>.........................] - ETA: 12s - loss: 0.2249 - accuracy: 0.9600
 4300/25000 [====>.........................] - ETA: 12s - loss: 0.2250 - accuracy: 0.9600
 4400/25000 [====>.........................] - ETA: 12s - loss: 0.2240 - accuracy: 0.9607
 4500/25000 [====>.........................] - ETA: 12s - loss: 0.2242 - accuracy: 0.9604
 4600/25000 [====>.........................] - ETA: 12s - loss: 0.2250 - accuracy: 0.9600
 4700/25000 [====>.........................] - ETA: 12s - loss: 0.2251 - accuracy: 0.9600
 4800/25000 [====>.........................] - ETA: 12s - loss: 0.2252 - accuracy: 0.9600
 4900/25000 [====>.........................] - ETA: 12s - loss: 0.2250 - accuracy: 0.9600
 5000/25000 [=====>........................] - ETA: 12s - loss: 0.2265 - accuracy: 0.9592
 5100/25000 [=====>........................] - ETA: 12s - loss: 0.2259 - accuracy: 0.9596
 5200/25000 [=====>........................] - ETA: 12s - loss: 0.2245 - accuracy: 0.9604
 5300/25000 [=====>........................] - ETA: 12s - loss: 0.2245 - accuracy: 0.9604
 5400/25000 [=====>........................] - ETA: 12s - loss: 0.2245 - accuracy: 0.9606
 5500/25000 [=====>........................] - ETA: 12s - loss: 0.2245 - accuracy: 0.9605
 5600/25000 [=====>........................] - ETA: 11s - loss: 0.2241 - accuracy: 0.9609
 5700/25000 [=====>........................] - ETA: 11s - loss: 0.2240 - accuracy: 0.9609
 5800/25000 [=====>........................] - ETA: 11s - loss: 0.2237 - accuracy: 0.9610
 5900/25000 [======>.......................] - ETA: 11s - loss: 0.2234 - accuracy: 0.9612
 6000/25000 [======>.......................] - ETA: 11s - loss: 0.2226 - accuracy: 0.9617
 6100/25000 [======>.......................] - ETA: 11s - loss: 0.2228 - accuracy: 0.9615
 6200/25000 [======>.......................] - ETA: 11s - loss: 0.2230 - accuracy: 0.9613
 6300/25000 [======>.......................] - ETA: 11s - loss: 0.2247 - accuracy: 0.9602
 6400/25000 [======>.......................] - ETA: 11s - loss: 0.2249 - accuracy: 0.9602
 6500/25000 [======>.......................] - ETA: 11s - loss: 0.2244 - accuracy: 0.9605
 6600/25000 [======>.......................] - ETA: 11s - loss: 0.2247 - accuracy: 0.9603
 6700/25000 [=======>......................] - ETA: 11s - loss: 0.2242 - accuracy: 0.9606
 6800/25000 [=======>......................] - ETA: 11s - loss: 0.2234 - accuracy: 0.9610
 6900/25000 [=======>......................] - ETA: 11s - loss: 0.2236 - accuracy: 0.9609
 7000/25000 [=======>......................] - ETA: 11s - loss: 0.2245 - accuracy: 0.9604
 7100/25000 [=======>......................] - ETA: 11s - loss: 0.2254 - accuracy: 0.9600
 7200/25000 [=======>......................] - ETA: 10s - loss: 0.2250 - accuracy: 0.9603
 7300/25000 [=======>......................] - ETA: 10s - loss: 0.2252 - accuracy: 0.9601
 7400/25000 [=======>......................] - ETA: 10s - loss: 0.2244 - accuracy: 0.9605
 7500/25000 [========>.....................] - ETA: 10s - loss: 0.2245 - accuracy: 0.9605
 7600/25000 [========>.....................] - ETA: 10s - loss: 0.2245 - accuracy: 0.9605
 7700/25000 [========>.....................] - ETA: 10s - loss: 0.2252 - accuracy: 0.9601
 7800/25000 [========>.....................] - ETA: 10s - loss: 0.2253 - accuracy: 0.9600
 7900/25000 [========>.....................] - ETA: 10s - loss: 0.2245 - accuracy: 0.9604
 8000/25000 [========>.....................] - ETA: 10s - loss: 0.2239 - accuracy: 0.9607
 8100/25000 [========>.....................] - ETA: 10s - loss: 0.2236 - accuracy: 0.9609
 8200/25000 [========>.....................] - ETA: 10s - loss: 0.2236 - accuracy: 0.9609
 8300/25000 [========>.....................] - ETA: 10s - loss: 0.2235 - accuracy: 0.9608
 8400/25000 [=========>....................] - ETA: 10s - loss: 0.2231 - accuracy: 0.9611
 8500/25000 [=========>....................] - ETA: 10s - loss: 0.2229 - accuracy: 0.9612
 8600/25000 [=========>....................] - ETA: 10s - loss: 0.2226 - accuracy: 0.9614
 8700/25000 [=========>....................] - ETA: 10s - loss: 0.2224 - accuracy: 0.9615
 8800/25000 [=========>....................] - ETA: 9s - loss: 0.2226 - accuracy: 0.9614 
 8900/25000 [=========>....................] - ETA: 9s - loss: 0.2223 - accuracy: 0.9615
 9000/25000 [=========>....................] - ETA: 9s - loss: 0.2228 - accuracy: 0.9611
 9100/25000 [=========>....................] - ETA: 9s - loss: 0.2233 - accuracy: 0.9609
 9200/25000 [==========>...................] - ETA: 9s - loss: 0.2232 - accuracy: 0.9609
 9300/25000 [==========>...................] - ETA: 9s - loss: 0.2236 - accuracy: 0.9605
 9400/25000 [==========>...................] - ETA: 9s - loss: 0.2234 - accuracy: 0.9606
 9500/25000 [==========>...................] - ETA: 9s - loss: 0.2230 - accuracy: 0.9608
 9600/25000 [==========>...................] - ETA: 9s - loss: 0.2237 - accuracy: 0.9605
 9700/25000 [==========>...................] - ETA: 9s - loss: 0.2236 - accuracy: 0.9605
 9800/25000 [==========>...................] - ETA: 9s - loss: 0.2234 - accuracy: 0.9606
 9900/25000 [==========>...................] - ETA: 9s - loss: 0.2230 - accuracy: 0.9608
10000/25000 [===========>..................] - ETA: 9s - loss: 0.2233 - accuracy: 0.9606
10100/25000 [===========>..................] - ETA: 9s - loss: 0.2235 - accuracy: 0.9605
10200/25000 [===========>..................] - ETA: 9s - loss: 0.2233 - accuracy: 0.9606
10300/25000 [===========>..................] - ETA: 9s - loss: 0.2231 - accuracy: 0.9607
10400/25000 [===========>..................] - ETA: 9s - loss: 0.2224 - accuracy: 0.9611
10500/25000 [===========>..................] - ETA: 8s - loss: 0.2222 - accuracy: 0.9612
10600/25000 [===========>..................] - ETA: 8s - loss: 0.2221 - accuracy: 0.9612
10700/25000 [===========>..................] - ETA: 8s - loss: 0.2223 - accuracy: 0.9611
10800/25000 [===========>..................] - ETA: 8s - loss: 0.2228 - accuracy: 0.9608
10900/25000 [============>.................] - ETA: 8s - loss: 0.2227 - accuracy: 0.9609
11000/25000 [============>.................] - ETA: 8s - loss: 0.2223 - accuracy: 0.9611
11100/25000 [============>.................] - ETA: 8s - loss: 0.2226 - accuracy: 0.9608
11200/25000 [============>.................] - ETA: 8s - loss: 0.2228 - accuracy: 0.9607
11300/25000 [============>.................] - ETA: 8s - loss: 0.2223 - accuracy: 0.9610
11400/25000 [============>.................] - ETA: 8s - loss: 0.2225 - accuracy: 0.9608
11500/25000 [============>.................] - ETA: 8s - loss: 0.2224 - accuracy: 0.9609
11600/25000 [============>.................] - ETA: 8s - loss: 0.2228 - accuracy: 0.9606
11700/25000 [=============>................] - ETA: 8s - loss: 0.2225 - accuracy: 0.9608
11800/25000 [=============>................] - ETA: 8s - loss: 0.2221 - accuracy: 0.9610
11900/25000 [=============>................] - ETA: 8s - loss: 0.2223 - accuracy: 0.9609
12000/25000 [=============>................] - ETA: 8s - loss: 0.2223 - accuracy: 0.9608
12100/25000 [=============>................] - ETA: 8s - loss: 0.2226 - accuracy: 0.9607
12200/25000 [=============>................] - ETA: 8s - loss: 0.2225 - accuracy: 0.9607
12300/25000 [=============>................] - ETA: 7s - loss: 0.2232 - accuracy: 0.9602
12400/25000 [=============>................] - ETA: 7s - loss: 0.2233 - accuracy: 0.9602
12500/25000 [==============>...............] - ETA: 7s - loss: 0.2230 - accuracy: 0.9602
12600/25000 [==============>...............] - ETA: 7s - loss: 0.2232 - accuracy: 0.9601
12700/25000 [==============>...............] - ETA: 7s - loss: 0.2237 - accuracy: 0.9598
12800/25000 [==============>...............] - ETA: 7s - loss: 0.2239 - accuracy: 0.9596
12900/25000 [==============>...............] - ETA: 7s - loss: 0.2244 - accuracy: 0.9594
13000/25000 [==============>...............] - ETA: 7s - loss: 0.2242 - accuracy: 0.9595
13100/25000 [==============>...............] - ETA: 7s - loss: 0.2247 - accuracy: 0.9592
13200/25000 [==============>...............] - ETA: 7s - loss: 0.2251 - accuracy: 0.9589
13300/25000 [==============>...............] - ETA: 7s - loss: 0.2246 - accuracy: 0.9592
13400/25000 [===============>..............] - ETA: 7s - loss: 0.2244 - accuracy: 0.9593
13500/25000 [===============>..............] - ETA: 7s - loss: 0.2239 - accuracy: 0.9596
13600/25000 [===============>..............] - ETA: 7s - loss: 0.2238 - accuracy: 0.9596
13700/25000 [===============>..............] - ETA: 7s - loss: 0.2239 - accuracy: 0.9594
13800/25000 [===============>..............] - ETA: 7s - loss: 0.2236 - accuracy: 0.9596
13900/25000 [===============>..............] - ETA: 6s - loss: 0.2236 - accuracy: 0.9596
14000/25000 [===============>..............] - ETA: 6s - loss: 0.2236 - accuracy: 0.9596
14100/25000 [===============>..............] - ETA: 6s - loss: 0.2234 - accuracy: 0.9597
14200/25000 [================>.............] - ETA: 6s - loss: 0.2232 - accuracy: 0.9599
14300/25000 [================>.............] - ETA: 6s - loss: 0.2233 - accuracy: 0.9598
14400/25000 [================>.............] - ETA: 6s - loss: 0.2234 - accuracy: 0.9597
14500/25000 [================>.............] - ETA: 6s - loss: 0.2231 - accuracy: 0.9598
14600/25000 [================>.............] - ETA: 6s - loss: 0.2228 - accuracy: 0.9599
14700/25000 [================>.............] - ETA: 6s - loss: 0.2229 - accuracy: 0.9599
14800/25000 [================>.............] - ETA: 6s - loss: 0.2227 - accuracy: 0.9600
14900/25000 [================>.............] - ETA: 6s - loss: 0.2223 - accuracy: 0.9602
15000/25000 [=================>............] - ETA: 6s - loss: 0.2221 - accuracy: 0.9603
15100/25000 [=================>............] - ETA: 6s - loss: 0.2218 - accuracy: 0.9605
15200/25000 [=================>............] - ETA: 6s - loss: 0.2221 - accuracy: 0.9603
15300/25000 [=================>............] - ETA: 6s - loss: 0.2224 - accuracy: 0.9601
15400/25000 [=================>............] - ETA: 6s - loss: 0.2223 - accuracy: 0.9601
15500/25000 [=================>............] - ETA: 5s - loss: 0.2225 - accuracy: 0.9600
15600/25000 [=================>............] - ETA: 5s - loss: 0.2229 - accuracy: 0.9599
15700/25000 [=================>............] - ETA: 5s - loss: 0.2231 - accuracy: 0.9597
15800/25000 [=================>............] - ETA: 5s - loss: 0.2235 - accuracy: 0.9596
15900/25000 [==================>...........] - ETA: 5s - loss: 0.2235 - accuracy: 0.9596
16000/25000 [==================>...........] - ETA: 5s - loss: 0.2233 - accuracy: 0.9596
16100/25000 [==================>...........] - ETA: 5s - loss: 0.2234 - accuracy: 0.9596
16200/25000 [==================>...........] - ETA: 5s - loss: 0.2235 - accuracy: 0.9594
16300/25000 [==================>...........] - ETA: 5s - loss: 0.2234 - accuracy: 0.9595
16400/25000 [==================>...........] - ETA: 5s - loss: 0.2233 - accuracy: 0.9595
16500/25000 [==================>...........] - ETA: 5s - loss: 0.2229 - accuracy: 0.9597
16600/25000 [==================>...........] - ETA: 5s - loss: 0.2229 - accuracy: 0.9598
16700/25000 [===================>..........] - ETA: 5s - loss: 0.2233 - accuracy: 0.9595
16800/25000 [===================>..........] - ETA: 5s - loss: 0.2234 - accuracy: 0.9594
16900/25000 [===================>..........] - ETA: 5s - loss: 0.2236 - accuracy: 0.9592
17000/25000 [===================>..........] - ETA: 4s - loss: 0.2236 - accuracy: 0.9593
17100/25000 [===================>..........] - ETA: 4s - loss: 0.2235 - accuracy: 0.9593
17200/25000 [===================>..........] - ETA: 4s - loss: 0.2236 - accuracy: 0.9592
17300/25000 [===================>..........] - ETA: 4s - loss: 0.2235 - accuracy: 0.9592
17400/25000 [===================>..........] - ETA: 4s - loss: 0.2237 - accuracy: 0.9591
17500/25000 [====================>.........] - ETA: 4s - loss: 0.2238 - accuracy: 0.9590
17600/25000 [====================>.........] - ETA: 4s - loss: 0.2239 - accuracy: 0.9589
17700/25000 [====================>.........] - ETA: 4s - loss: 0.2239 - accuracy: 0.9588
17800/25000 [====================>.........] - ETA: 4s - loss: 0.2237 - accuracy: 0.9590
17900/25000 [====================>.........] - ETA: 4s - loss: 0.2236 - accuracy: 0.9590
18000/25000 [====================>.........] - ETA: 4s - loss: 0.2234 - accuracy: 0.9591
18100/25000 [====================>.........] - ETA: 4s - loss: 0.2235 - accuracy: 0.9591
18200/25000 [====================>.........] - ETA: 4s - loss: 0.2235 - accuracy: 0.9591
18300/25000 [====================>.........] - ETA: 4s - loss: 0.2237 - accuracy: 0.9589
18400/25000 [=====================>........] - ETA: 4s - loss: 0.2236 - accuracy: 0.9589
18500/25000 [=====================>........] - ETA: 4s - loss: 0.2234 - accuracy: 0.9590
18600/25000 [=====================>........] - ETA: 3s - loss: 0.2235 - accuracy: 0.9590
18700/25000 [=====================>........] - ETA: 3s - loss: 0.2235 - accuracy: 0.9589
18800/25000 [=====================>........] - ETA: 3s - loss: 0.2236 - accuracy: 0.9589
18900/25000 [=====================>........] - ETA: 3s - loss: 0.2235 - accuracy: 0.9589
19000/25000 [=====================>........] - ETA: 3s - loss: 0.2236 - accuracy: 0.9588
19100/25000 [=====================>........] - ETA: 3s - loss: 0.2238 - accuracy: 0.9587
19200/25000 [======================>.......] - ETA: 3s - loss: 0.2239 - accuracy: 0.9586
19300/25000 [======================>.......] - ETA: 3s - loss: 0.2239 - accuracy: 0.9586
19400/25000 [======================>.......] - ETA: 3s - loss: 0.2238 - accuracy: 0.9586
19500/25000 [======================>.......] - ETA: 3s - loss: 0.2239 - accuracy: 0.9585
19600/25000 [======================>.......] - ETA: 3s - loss: 0.2240 - accuracy: 0.9585
19700/25000 [======================>.......] - ETA: 3s - loss: 0.2241 - accuracy: 0.9583
19800/25000 [======================>.......] - ETA: 3s - loss: 0.2242 - accuracy: 0.9583
19900/25000 [======================>.......] - ETA: 3s - loss: 0.2243 - accuracy: 0.9582
20000/25000 [=======================>......] - ETA: 3s - loss: 0.2246 - accuracy: 0.9581
20100/25000 [=======================>......] - ETA: 3s - loss: 0.2246 - accuracy: 0.9580
20200/25000 [=======================>......] - ETA: 2s - loss: 0.2249 - accuracy: 0.9578
20300/25000 [=======================>......] - ETA: 2s - loss: 0.2251 - accuracy: 0.9576
20400/25000 [=======================>......] - ETA: 2s - loss: 0.2255 - accuracy: 0.9575
20500/25000 [=======================>......] - ETA: 2s - loss: 0.2255 - accuracy: 0.9575
20600/25000 [=======================>......] - ETA: 2s - loss: 0.2255 - accuracy: 0.9575
20700/25000 [=======================>......] - ETA: 2s - loss: 0.2258 - accuracy: 0.9573
20800/25000 [=======================>......] - ETA: 2s - loss: 0.2256 - accuracy: 0.9573
20900/25000 [========================>.....] - ETA: 2s - loss: 0.2255 - accuracy: 0.9574
21000/25000 [========================>.....] - ETA: 2s - loss: 0.2254 - accuracy: 0.9574
21100/25000 [========================>.....] - ETA: 2s - loss: 0.2253 - accuracy: 0.9575
21200/25000 [========================>.....] - ETA: 2s - loss: 0.2252 - accuracy: 0.9575
21300/25000 [========================>.....] - ETA: 2s - loss: 0.2248 - accuracy: 0.9577
21400/25000 [========================>.....] - ETA: 2s - loss: 0.2248 - accuracy: 0.9576
21500/25000 [========================>.....] - ETA: 2s - loss: 0.2249 - accuracy: 0.9575
21600/25000 [========================>.....] - ETA: 2s - loss: 0.2250 - accuracy: 0.9574
21700/25000 [=========================>....] - ETA: 2s - loss: 0.2247 - accuracy: 0.9576
21800/25000 [=========================>....] - ETA: 1s - loss: 0.2247 - accuracy: 0.9575
21900/25000 [=========================>....] - ETA: 1s - loss: 0.2248 - accuracy: 0.9574
22000/25000 [=========================>....] - ETA: 1s - loss: 0.2249 - accuracy: 0.9574
22100/25000 [=========================>....] - ETA: 1s - loss: 0.2250 - accuracy: 0.9574
22200/25000 [=========================>....] - ETA: 1s - loss: 0.2249 - accuracy: 0.9573
22300/25000 [=========================>....] - ETA: 1s - loss: 0.2252 - accuracy: 0.9572
22400/25000 [=========================>....] - ETA: 1s - loss: 0.2253 - accuracy: 0.9571
22500/25000 [==========================>...] - ETA: 1s - loss: 0.2254 - accuracy: 0.9570
22600/25000 [==========================>...] - ETA: 1s - loss: 0.2257 - accuracy: 0.9569
22700/25000 [==========================>...] - ETA: 1s - loss: 0.2257 - accuracy: 0.9569
22800/25000 [==========================>...] - ETA: 1s - loss: 0.2256 - accuracy: 0.9569
22900/25000 [==========================>...] - ETA: 1s - loss: 0.2256 - accuracy: 0.9569
23000/25000 [==========================>...] - ETA: 1s - loss: 0.2258 - accuracy: 0.9567
23100/25000 [==========================>...] - ETA: 1s - loss: 0.2258 - accuracy: 0.9567
23200/25000 [==========================>...] - ETA: 1s - loss: 0.2257 - accuracy: 0.9567
23300/25000 [==========================>...] - ETA: 1s - loss: 0.2257 - accuracy: 0.9567
23400/25000 [===========================>..] - ETA: 0s - loss: 0.2257 - accuracy: 0.9568
23500/25000 [===========================>..] - ETA: 0s - loss: 0.2255 - accuracy: 0.9569
23600/25000 [===========================>..] - ETA: 0s - loss: 0.2254 - accuracy: 0.9569
23700/25000 [===========================>..] - ETA: 0s - loss: 0.2253 - accuracy: 0.9570
23800/25000 [===========================>..] - ETA: 0s - loss: 0.2251 - accuracy: 0.9570
23900/25000 [===========================>..] - ETA: 0s - loss: 0.2251 - accuracy: 0.9570
24000/25000 [===========================>..] - ETA: 0s - loss: 0.2251 - accuracy: 0.9570
24100/25000 [===========================>..] - ETA: 0s - loss: 0.2250 - accuracy: 0.9571
24200/25000 [============================>.] - ETA: 0s - loss: 0.2250 - accuracy: 0.9571
24300/25000 [============================>.] - ETA: 0s - loss: 0.2247 - accuracy: 0.9572
24400/25000 [============================>.] - ETA: 0s - loss: 0.2248 - accuracy: 0.9572
24500/25000 [============================>.] - ETA: 0s - loss: 0.2248 - accuracy: 0.9571
24600/25000 [============================>.] - ETA: 0s - loss: 0.2248 - accuracy: 0.9572
24700/25000 [============================>.] - ETA: 0s - loss: 0.2248 - accuracy: 0.9571
24800/25000 [============================>.] - ETA: 0s - loss: 0.2245 - accuracy: 0.9573
24900/25000 [============================>.] - ETA: 0s - loss: 0.2245 - accuracy: 0.9572
25000/25000 [==============================] - 20s 782us/step - loss: 0.2246 - accuracy: 0.9572 - val_loss: 0.3948 - val_accuracy: 0.8566
	=====> Test the model: model.predict()
IMDB_REVIEWS
	Training accuracy score: 96.36%
	Loss: 0.2090
	Test Accuracy: 85.66%




DONE!
Program finished. It took 379.1327602863312 seconds

Process finished with exit code 0
```