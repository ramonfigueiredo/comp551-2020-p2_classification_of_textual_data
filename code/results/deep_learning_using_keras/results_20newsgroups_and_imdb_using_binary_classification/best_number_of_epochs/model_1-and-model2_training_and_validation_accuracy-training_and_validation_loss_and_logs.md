## Two Deep Learning approaches using Keras: Loss, Training Accuracy, Test Accuracy, Training Time, Test Time

| ID | Dataset | Algorithm | Loss | Training accuracy score (%) | Test accuracy score (%) | Training time (seconds) | Test time (seconds) |
| --- | ------- | --------- | ---- | --------------------------- | ----------------------- | ----------------------- | ------------------- |
| 1 | TWENTY_NEWS_GROUPS | Deep Learning using Keras 1 (KERAS_DL1) | 0.0179 | 99.70 | 96.69 | 94.4457 | 3.5685 |
| 2 | IMDB_REVIEWS | Deep Learning using Keras 1 (KERAS_DL1) | 0.1362 | 96.15 | 88.57 | 17.2855 | 3.3037 |
| 3 | TWENTY_NEWS_GROUPS | Deep Learning using Keras 2 (KERAS_DL2) | 0.0822 | 96.96 | 95.66 | 127.3021 | 1.9339 |
| 4 | IMDB_REVIEWS | Deep Learning using Keras 2 (KERAS_DL2) | 0.1185 | 96.68 | 86.25 | 60.0973 | 6.2492 |

### Deep Learning using Keras 1 (KERAS_DL1)

![TWENTY_NEWS_GROUPS: Loss, Training Accuracy, Test Accuracy, Training Time, Test Time](https://github.com/ramonfigueiredopessoa/comp551-2020-p2_classification_of_textual_data/blob/master/code/results/deep_learning_using_keras/results_20newsgroups_and_imdb_using_binary_classification/best_number_of_epochs/KERAS_DL1_TWENTY_NEWS_GROUPS_training_and_validation_accuracy_and_Loss.png)

![IMDB_REVIEWS: Loss, Training Accuracy, Test Accuracy, Training Time, Test Time](https://github.com/ramonfigueiredopessoa/comp551-2020-p2_classification_of_textual_data/blob/master/code/results/deep_learning_using_keras/results_20newsgroups_and_imdb_using_binary_classification/best_number_of_epochs/KERAS_DL1_IMDB_REVIEWS_training_and_validation_accuracy_and_Loss.png)


### Deep Learning using Keras 2 (KERAS_DL2)

![TWENTY_NEWS_GROUPS: Loss, Training Accuracy, Test Accuracy, Training Time, Test Time](https://github.com/ramonfigueiredopessoa/comp551-2020-p2_classification_of_textual_data/blob/master/code/results/deep_learning_using_keras/results_20newsgroups_and_imdb_using_binary_classification/best_number_of_epochs/KERAS_DL2_TWENTY_NEWS_GROUPS_training_and_validation_accuracy_and_Loss.png)

![IMDB_REVIEWS: Loss, Training Accuracy, Test Accuracy, Training Time, Test Time](https://github.com/ramonfigueiredopessoa/comp551-2020-p2_classification_of_textual_data/blob/master/code/results/deep_learning_using_keras/results_20newsgroups_and_imdb_using_binary_classification/best_number_of_epochs/KERAS_DL2_IMDB_REVIEWS_training_and_validation_accuracy_and_Loss.png)

## Comparing test accuracy score using best number of epochs and using 20 epochs

| Deep Learning	| Dataset					 | Best number of epochs | Test accuracy with best epochs | Test accuracy with 20 epochs |
| ------------- | -------------------------- | --------------------- |------------------------------- | ---------------------------- |
| KERAS DL 1	| 20 NEWS GROUPS 			 | 10					 | 96.69						  | 96.66						 |
| KERAS DL 1	| IMDB REVIEWS (binary)		 | 1					 | 88.36						  | 83.23						 |
| KERAS DL 1	| IMDB REVIEWS (multi-class) | 2					 | 89.10						  | 86.61						 |
| KERAS DL 2	| 20 NEWS GROUPS 			 | 15					 | 96.08						  | 95.98						 |
| KERAS DL 2	| IMDB REVIEWS (binary)		 | 3					 | 86.33						  | 83.98						 |
| KERAS DL 2	| IMDB REVIEWS (multi-class) | 2					 | 89.07						  | 86.15						 |

#### Computer settings used to run

* Operating system: Ubuntu 18.04.4 LTS (64-bit)
* Processor: Intel® Core™ i7-7700 CPU @ 3.60GHz × 8 
* Memory: 32 GB

#### All logs 

```
python /comp551-2020-p2_classification_of_textual_data/code/main.py -dl
Using TensorFlow backend.
2020-03-12 15:19:16.290634: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libnvinfer.so.6'; dlerror: libnvinfer.so.6: cannot open shared object file: No such file or directory
2020-03-12 15:19:16.290718: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libnvinfer_plugin.so.6'; dlerror: libnvinfer_plugin.so.6: cannot open shared object file: No such file or directory
2020-03-12 15:19:16.290724: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:30] Cannot dlopen some TensorRT libraries. If you would like to use Nvidia GPU with TensorRT, please make sure the missing libraries mentioned above are installed properly.
[nltk_data] Downloading package wordnet to /home/ets-
[nltk_data]     crchum/nltk_data...
[nltk_data]   Package wordnet is already up-to-date!
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
	Number of epochs used by the deep learning approach. Default: None = use the best number of epochs for each dataset. =  None
	Deep Learning algorithm list (If dl_algorithm_list is not provided, all Deep Learning algorithms will be executed). Options of Deep Learning algorithms: 1) KERAS_DL1, 2) KERAS_DL2. = None
	Run grid search for all datasets (TWENTY_NEWS_GROUPS, IMDB_REVIEWS binary labels and IMDB_REVIEWS multi-class labels), and all 14 classifiers. Default: False (run scikit-learn algorithms or deep learning algorithms). Note: this takes many hours to execute. = False
==================================================================================================================================

Loading TWENTY_NEWS_GROUPS dataset for categories:
03/12/2020 03:19:16 PM - INFO - Program started...
03/12/2020 03:19:16 PM - INFO - Program started...
data loaded
11314 documents - 13.782MB (training set)
7532 documents - 8.262MB (test set)
20 categories

Extracting features from the training data using a vectorizer
done in 1.180811s at 11.672MB/s
n_samples: 11314, n_features: 101321

Extracting features from the test data using the same vectorizer
done in 0.629623s at 13.121MB/s
n_samples: 7532, n_features: 101321

2020-03-12 15:19:20.177359: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
2020-03-12 15:19:20.190794: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-03-12 15:19:20.191377: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1555] Found device 0 with properties: 
pciBusID: 0000:01:00.0 name: GeForce GTX 1080 computeCapability: 6.1
coreClock: 1.7335GHz coreCount: 20 deviceMemorySize: 7.93GiB deviceMemoryBandwidth: 298.32GiB/s
2020-03-12 15:19:20.191436: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcudart.so.10.1'; dlerror: libcudart.so.10.1: cannot open shared object file: No such file or directory
2020-03-12 15:19:20.191477: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcublas.so.10'; dlerror: libcublas.so.10: cannot open shared object file: No such file or directory
2020-03-12 15:19:20.191516: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcufft.so.10'; dlerror: libcufft.so.10: cannot open shared object file: No such file or directory
2020-03-12 15:19:20.191556: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcurand.so.10'; dlerror: libcurand.so.10: cannot open shared object file: No such file or directory
2020-03-12 15:19:20.191597: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcusolver.so.10'; dlerror: libcusolver.so.10: cannot open shared object file: No such file or directory
2020-03-12 15:19:20.191636: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcusparse.so.10'; dlerror: libcusparse.so.10: cannot open shared object file: No such file or directory
2020-03-12 15:19:20.193522: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2020-03-12 15:19:20.193532: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1592] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
Skipping registering GPU devices...
2020-03-12 15:19:20.193733: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-03-12 15:19:20.216120: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 3600000000 Hz
2020-03-12 15:19:20.216865: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x9665cb0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-03-12 15:19:20.216883: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2020-03-12 15:19:20.284448: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-03-12 15:19:20.285060: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x972fb10 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2020-03-12 15:19:20.285069: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): GeForce GTX 1080, Compute Capability 6.1
2020-03-12 15:19:20.285155: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1096] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-03-12 15:19:20.285160: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1102]      
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 10)                1013220   
_________________________________________________________________
dense_2 (Dense)              (None, 19)                209       
=================================================================
Total params: 1,013,429
Trainable params: 1,013,429
Non-trainable params: 0
_________________________________________________________________
None


NUMBER OF EPOCHS USED: 10

	Dataset: TWENTY_NEWS_GROUPS
	Algorithm: Deep Learning using Keras 1 (KERAS_DL1)
	Training loss: 0.0179
	Training accuracy score: 99.70%
	Test loss: 0.1175
	Test accuracy score: 96.69%
	Training time: 94.4457
	Test time: 3.5685


Loading IMDB_REVIEWS dataset:

===> Reading files from /home/ets-crchum/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/train/neg

===> Reading files from /home/ets-crchum/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/train/pos

===> Reading files from /home/ets-crchum/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/test/neg

===> Reading files from /home/ets-crchum/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/test/pos
data loaded
25000 documents - 33.133MB (training set)
25000 documents - 32.351MB (test set)

Extracting features from the training data using a vectorizer
done in 2.970234s at 11.155MB/s
n_samples: 25000, n_features: 74170

Extracting features from the test data using the same vectorizer
done in 2.896089s at 11.170MB/s
n_samples: 25000, n_features: 74170

Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_3 (Dense)              (None, 10)                741710    
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
Total params: 741,721
Trainable params: 741,721
Non-trainable params: 0
_________________________________________________________________
None


NUMBER OF EPOCHS USED: 1

	Dataset: IMDB_REVIEWS
	Algorithm: Deep Learning using Keras 1 (KERAS_DL1)
	Training loss: 0.1362
	Training accuracy score: 96.15%
	Test loss: 0.2801
	Test accuracy score: 88.57%
	Training time: 17.2855
	Test time: 3.3037


03/12/2020 03:21:35 PM - INFO - Program started...
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
	It took 10.418174743652344 seconds


Applying NLTK feature extraction: X_test
	==> Replace unused UNICODE characters. 
	==> Text in lower case. 
	==> Apply Lemmatize using WordNet's built-in morphy function. Returns the input word unchanged if it cannot be found in WordNet 
	==> Remove English stop words
	Done: NLTK feature extraction (replace unused UNICODE characters, text in lower case, apply Lemmatize using WordNet, remove English stop words)  finished!
	It took 6.234265327453613 seconds

	===> Tokenizer: fit_on_texts(X_train)
	===> X_train = pad_sequences(list_tokenized_train, maxlen=6000)
	===> Create Keras model
Model: "sequential_3"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, None, 128)         768000    
_________________________________________________________________
bidirectional_1 (Bidirection (None, None, 64)          41216     
_________________________________________________________________
global_max_pooling1d_1 (Glob (None, 64)                0         
_________________________________________________________________
dense_5 (Dense)              (None, 20)                1300      
_________________________________________________________________
dropout_1 (Dropout)          (None, 20)                0         
_________________________________________________________________
dense_6 (Dense)              (None, 19)                399       
=================================================================
Total params: 810,915
Trainable params: 810,915
Non-trainable params: 0
_________________________________________________________________
None
	===> Tokenizer: fit_on_texts(X_test)
	===> X_test = pad_sequences(list_sentences_test, maxlen=6000)


NUMBER OF EPOCHS USED: 15

Train on 11314 samples, validate on 7532 samples
Epoch 1/15

  100/11314 [..............................] - ETA: 1:02 - loss: 0.6964 - accuracy: 0.4689
  200/11314 [..............................] - ETA: 34s - loss: 0.6952 - accuracy: 0.4947 
  300/11314 [..............................] - ETA: 25s - loss: 0.6942 - accuracy: 0.5186
  400/11314 [>.............................] - ETA: 20s - loss: 0.6933 - accuracy: 0.5459
  500/11314 [>.............................] - ETA: 17s - loss: 0.6925 - accuracy: 0.5695
  600/11314 [>.............................] - ETA: 15s - loss: 0.6918 - accuracy: 0.5911
  700/11314 [>.............................] - ETA: 14s - loss: 0.6911 - accuracy: 0.6101
  800/11314 [=>............................] - ETA: 13s - loss: 0.6904 - accuracy: 0.6265
  900/11314 [=>............................] - ETA: 12s - loss: 0.6897 - accuracy: 0.6377
 1000/11314 [=>............................] - ETA: 11s - loss: 0.6889 - accuracy: 0.6476
 1100/11314 [=>............................] - ETA: 11s - loss: 0.6880 - accuracy: 0.6545
 1200/11314 [==>...........................] - ETA: 10s - loss: 0.6871 - accuracy: 0.6625
 1300/11314 [==>...........................] - ETA: 10s - loss: 0.6861 - accuracy: 0.6693
 1400/11314 [==>...........................] - ETA: 10s - loss: 0.6851 - accuracy: 0.6741
 1500/11314 [==>...........................] - ETA: 9s - loss: 0.6840 - accuracy: 0.6782 
 1600/11314 [===>..........................] - ETA: 9s - loss: 0.6827 - accuracy: 0.6816
 1700/11314 [===>..........................] - ETA: 9s - loss: 0.6815 - accuracy: 0.6844
 1800/11314 [===>..........................] - ETA: 8s - loss: 0.6802 - accuracy: 0.6866
 1900/11314 [====>.........................] - ETA: 8s - loss: 0.6786 - accuracy: 0.6888
 2000/11314 [====>.........................] - ETA: 8s - loss: 0.6771 - accuracy: 0.6906
 2100/11314 [====>.........................] - ETA: 8s - loss: 0.6752 - accuracy: 0.6923
 2200/11314 [====>.........................] - ETA: 8s - loss: 0.6734 - accuracy: 0.6934
 2300/11314 [=====>........................] - ETA: 7s - loss: 0.6714 - accuracy: 0.6946
 2400/11314 [=====>........................] - ETA: 7s - loss: 0.6694 - accuracy: 0.6957
 2500/11314 [=====>........................] - ETA: 7s - loss: 0.6672 - accuracy: 0.6966
 2600/11314 [=====>........................] - ETA: 7s - loss: 0.6648 - accuracy: 0.6974
 2700/11314 [======>.......................] - ETA: 7s - loss: 0.6623 - accuracy: 0.6982
 2800/11314 [======>.......................] - ETA: 7s - loss: 0.6599 - accuracy: 0.6987
 2900/11314 [======>.......................] - ETA: 6s - loss: 0.6571 - accuracy: 0.6995
 3000/11314 [======>.......................] - ETA: 6s - loss: 0.6543 - accuracy: 0.7002
 3100/11314 [=======>......................] - ETA: 6s - loss: 0.6514 - accuracy: 0.7010
 3200/11314 [=======>......................] - ETA: 6s - loss: 0.6485 - accuracy: 0.7014
 3300/11314 [=======>......................] - ETA: 6s - loss: 0.6455 - accuracy: 0.7019
 3400/11314 [========>.....................] - ETA: 6s - loss: 0.6426 - accuracy: 0.7022
 3500/11314 [========>.....................] - ETA: 6s - loss: 0.6393 - accuracy: 0.7026
 3600/11314 [========>.....................] - ETA: 6s - loss: 0.6360 - accuracy: 0.7033
 3700/11314 [========>.....................] - ETA: 5s - loss: 0.6326 - accuracy: 0.7040
 3800/11314 [=========>....................] - ETA: 5s - loss: 0.6294 - accuracy: 0.7046
 3900/11314 [=========>....................] - ETA: 5s - loss: 0.6259 - accuracy: 0.7067
 4000/11314 [=========>....................] - ETA: 5s - loss: 0.6226 - accuracy: 0.7090
 4100/11314 [=========>....................] - ETA: 5s - loss: 0.6192 - accuracy: 0.7114
 4200/11314 [==========>...................] - ETA: 5s - loss: 0.6159 - accuracy: 0.7142
 4300/11314 [==========>...................] - ETA: 5s - loss: 0.6124 - accuracy: 0.7172
 4400/11314 [==========>...................] - ETA: 5s - loss: 0.6089 - accuracy: 0.7202
 4500/11314 [==========>...................] - ETA: 5s - loss: 0.6052 - accuracy: 0.7237
 4600/11314 [===========>..................] - ETA: 5s - loss: 0.6016 - accuracy: 0.7271
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.5980 - accuracy: 0.7303
 4800/11314 [===========>..................] - ETA: 4s - loss: 0.5943 - accuracy: 0.7337
 4900/11314 [===========>..................] - ETA: 4s - loss: 0.5908 - accuracy: 0.7365
 5000/11314 [============>.................] - ETA: 4s - loss: 0.5869 - accuracy: 0.7397
 5100/11314 [============>.................] - ETA: 4s - loss: 0.5831 - accuracy: 0.7426
 5200/11314 [============>.................] - ETA: 4s - loss: 0.5793 - accuracy: 0.7453
 5300/11314 [=============>................] - ETA: 4s - loss: 0.5755 - accuracy: 0.7482
 5400/11314 [=============>................] - ETA: 4s - loss: 0.5716 - accuracy: 0.7511
 5500/11314 [=============>................] - ETA: 4s - loss: 0.5677 - accuracy: 0.7544
 5600/11314 [=============>................] - ETA: 4s - loss: 0.5640 - accuracy: 0.7575
 5700/11314 [==============>...............] - ETA: 4s - loss: 0.5604 - accuracy: 0.7602
 5800/11314 [==============>...............] - ETA: 4s - loss: 0.5565 - accuracy: 0.7633
 5900/11314 [==============>...............] - ETA: 4s - loss: 0.5527 - accuracy: 0.7662
 6000/11314 [==============>...............] - ETA: 4s - loss: 0.5489 - accuracy: 0.7689
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.5452 - accuracy: 0.7718
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.5414 - accuracy: 0.7744
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.5378 - accuracy: 0.7769
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.5341 - accuracy: 0.7795
 6500/11314 [================>.............] - ETA: 3s - loss: 0.5305 - accuracy: 0.7817
 6600/11314 [================>.............] - ETA: 3s - loss: 0.5271 - accuracy: 0.7841
 6700/11314 [================>.............] - ETA: 3s - loss: 0.5235 - accuracy: 0.7863
 6800/11314 [=================>............] - ETA: 3s - loss: 0.5202 - accuracy: 0.7884
 6900/11314 [=================>............] - ETA: 3s - loss: 0.5167 - accuracy: 0.7905
 7000/11314 [=================>............] - ETA: 3s - loss: 0.5133 - accuracy: 0.7926
 7100/11314 [=================>............] - ETA: 3s - loss: 0.5099 - accuracy: 0.7947
 7200/11314 [==================>...........] - ETA: 3s - loss: 0.5066 - accuracy: 0.7967
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.5032 - accuracy: 0.7988
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.5000 - accuracy: 0.8006
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.4967 - accuracy: 0.8025
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.4935 - accuracy: 0.8043
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.4903 - accuracy: 0.8061
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.4872 - accuracy: 0.8079
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.4841 - accuracy: 0.8096
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.4811 - accuracy: 0.8113
 8100/11314 [====================>.........] - ETA: 2s - loss: 0.4781 - accuracy: 0.8129
 8200/11314 [====================>.........] - ETA: 2s - loss: 0.4752 - accuracy: 0.8145
 8300/11314 [=====================>........] - ETA: 2s - loss: 0.4723 - accuracy: 0.8161
 8400/11314 [=====================>........] - ETA: 2s - loss: 0.4697 - accuracy: 0.8176
 8500/11314 [=====================>........] - ETA: 2s - loss: 0.4669 - accuracy: 0.8191
 8600/11314 [=====================>........] - ETA: 2s - loss: 0.4642 - accuracy: 0.8205
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.4616 - accuracy: 0.8219
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.4590 - accuracy: 0.8233
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.4563 - accuracy: 0.8246
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.4537 - accuracy: 0.8260
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.4513 - accuracy: 0.8272
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.4489 - accuracy: 0.8285
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.4465 - accuracy: 0.8298
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.4441 - accuracy: 0.8310
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.4418 - accuracy: 0.8322
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.4397 - accuracy: 0.8334
 9700/11314 [========================>.....] - ETA: 1s - loss: 0.4375 - accuracy: 0.8345
 9800/11314 [========================>.....] - ETA: 1s - loss: 0.4354 - accuracy: 0.8356
 9900/11314 [=========================>....] - ETA: 1s - loss: 0.4333 - accuracy: 0.8367
10000/11314 [=========================>....] - ETA: 0s - loss: 0.4312 - accuracy: 0.8378
10100/11314 [=========================>....] - ETA: 0s - loss: 0.4291 - accuracy: 0.8389
10200/11314 [==========================>...] - ETA: 0s - loss: 0.4270 - accuracy: 0.8400
10300/11314 [==========================>...] - ETA: 0s - loss: 0.4250 - accuracy: 0.8410
10400/11314 [==========================>...] - ETA: 0s - loss: 0.4231 - accuracy: 0.8420
10500/11314 [==========================>...] - ETA: 0s - loss: 0.4212 - accuracy: 0.8430
10600/11314 [===========================>..] - ETA: 0s - loss: 0.4195 - accuracy: 0.8440
10700/11314 [===========================>..] - ETA: 0s - loss: 0.4176 - accuracy: 0.8450
10800/11314 [===========================>..] - ETA: 0s - loss: 0.4158 - accuracy: 0.8459
10900/11314 [===========================>..] - ETA: 0s - loss: 0.4139 - accuracy: 0.8468
11000/11314 [============================>.] - ETA: 0s - loss: 0.4122 - accuracy: 0.8477
11100/11314 [============================>.] - ETA: 0s - loss: 0.4105 - accuracy: 0.8486
11200/11314 [============================>.] - ETA: 0s - loss: 0.4087 - accuracy: 0.8495
11300/11314 [============================>.] - ETA: 0s - loss: 0.4069 - accuracy: 0.8503
11314/11314 [==============================] - 10s 852us/step - loss: 0.4067 - accuracy: 0.8505 - val_loss: 0.2049 - val_accuracy: 0.9496
Epoch 2/15

  100/11314 [..............................] - ETA: 6s - loss: 0.2111 - accuracy: 0.9468
  200/11314 [..............................] - ETA: 6s - loss: 0.2158 - accuracy: 0.9450
  300/11314 [..............................] - ETA: 6s - loss: 0.2159 - accuracy: 0.9451
  400/11314 [>.............................] - ETA: 6s - loss: 0.2170 - accuracy: 0.9447
  500/11314 [>.............................] - ETA: 6s - loss: 0.2168 - accuracy: 0.9452
  600/11314 [>.............................] - ETA: 6s - loss: 0.2166 - accuracy: 0.9454
  700/11314 [>.............................] - ETA: 6s - loss: 0.2168 - accuracy: 0.9450
  800/11314 [=>............................] - ETA: 6s - loss: 0.2165 - accuracy: 0.9451
  900/11314 [=>............................] - ETA: 6s - loss: 0.2163 - accuracy: 0.9454
 1000/11314 [=>............................] - ETA: 6s - loss: 0.2158 - accuracy: 0.9456
 1100/11314 [=>............................] - ETA: 6s - loss: 0.2164 - accuracy: 0.9456
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.2170 - accuracy: 0.9456
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.2166 - accuracy: 0.9456
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.2170 - accuracy: 0.9458
 1500/11314 [==>...........................] - ETA: 6s - loss: 0.2171 - accuracy: 0.9459
 1600/11314 [===>..........................] - ETA: 6s - loss: 0.2164 - accuracy: 0.9462
 1700/11314 [===>..........................] - ETA: 6s - loss: 0.2159 - accuracy: 0.9464
 1800/11314 [===>..........................] - ETA: 6s - loss: 0.2159 - accuracy: 0.9466
 1900/11314 [====>.........................] - ETA: 5s - loss: 0.2156 - accuracy: 0.9468
 2000/11314 [====>.........................] - ETA: 5s - loss: 0.2158 - accuracy: 0.9468
 2100/11314 [====>.........................] - ETA: 5s - loss: 0.2155 - accuracy: 0.9469
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.2154 - accuracy: 0.9470
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.2154 - accuracy: 0.9470
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.2156 - accuracy: 0.9468
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.2156 - accuracy: 0.9469
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.2156 - accuracy: 0.9469
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.2153 - accuracy: 0.9471
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.2152 - accuracy: 0.9471
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.2151 - accuracy: 0.9472
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.2148 - accuracy: 0.9473
 3100/11314 [=======>......................] - ETA: 5s - loss: 0.2146 - accuracy: 0.9473
 3200/11314 [=======>......................] - ETA: 5s - loss: 0.2146 - accuracy: 0.9473
 3300/11314 [=======>......................] - ETA: 5s - loss: 0.2145 - accuracy: 0.9473
 3400/11314 [========>.....................] - ETA: 5s - loss: 0.2141 - accuracy: 0.9474
 3500/11314 [========>.....................] - ETA: 5s - loss: 0.2142 - accuracy: 0.9473
 3600/11314 [========>.....................] - ETA: 4s - loss: 0.2141 - accuracy: 0.9474
 3700/11314 [========>.....................] - ETA: 4s - loss: 0.2139 - accuracy: 0.9475
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.2138 - accuracy: 0.9475
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.2137 - accuracy: 0.9475
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.2136 - accuracy: 0.9475
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.2136 - accuracy: 0.9475
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.2134 - accuracy: 0.9475
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.2134 - accuracy: 0.9476
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.2131 - accuracy: 0.9477
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.2130 - accuracy: 0.9477
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.2128 - accuracy: 0.9477
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.2129 - accuracy: 0.9476
 4800/11314 [===========>..................] - ETA: 4s - loss: 0.2129 - accuracy: 0.9477
 4900/11314 [===========>..................] - ETA: 4s - loss: 0.2128 - accuracy: 0.9477
 5000/11314 [============>.................] - ETA: 4s - loss: 0.2129 - accuracy: 0.9477
 5100/11314 [============>.................] - ETA: 4s - loss: 0.2130 - accuracy: 0.9477
 5200/11314 [============>.................] - ETA: 4s - loss: 0.2131 - accuracy: 0.9477
 5300/11314 [=============>................] - ETA: 3s - loss: 0.2130 - accuracy: 0.9477
 5400/11314 [=============>................] - ETA: 3s - loss: 0.2129 - accuracy: 0.9477
 5500/11314 [=============>................] - ETA: 3s - loss: 0.2129 - accuracy: 0.9477
 5600/11314 [=============>................] - ETA: 3s - loss: 0.2128 - accuracy: 0.9477
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.2128 - accuracy: 0.9477
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.2128 - accuracy: 0.9477
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.2128 - accuracy: 0.9477
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.2127 - accuracy: 0.9477
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.2127 - accuracy: 0.9477
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.2126 - accuracy: 0.9477
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.2125 - accuracy: 0.9477
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.2123 - accuracy: 0.9478
 6500/11314 [================>.............] - ETA: 3s - loss: 0.2122 - accuracy: 0.9478
 6600/11314 [================>.............] - ETA: 3s - loss: 0.2121 - accuracy: 0.9478
 6700/11314 [================>.............] - ETA: 2s - loss: 0.2121 - accuracy: 0.9479
 6800/11314 [=================>............] - ETA: 2s - loss: 0.2121 - accuracy: 0.9479
 6900/11314 [=================>............] - ETA: 2s - loss: 0.2121 - accuracy: 0.9479
 7000/11314 [=================>............] - ETA: 2s - loss: 0.2121 - accuracy: 0.9479
 7100/11314 [=================>............] - ETA: 2s - loss: 0.2120 - accuracy: 0.9479
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.2120 - accuracy: 0.9479
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.2120 - accuracy: 0.9479
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.2120 - accuracy: 0.9479
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.2119 - accuracy: 0.9479
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.2119 - accuracy: 0.9479
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.2118 - accuracy: 0.9479
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.2117 - accuracy: 0.9479
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.2116 - accuracy: 0.9480
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.2117 - accuracy: 0.9479
 8100/11314 [====================>.........] - ETA: 2s - loss: 0.2116 - accuracy: 0.9479
 8200/11314 [====================>.........] - ETA: 2s - loss: 0.2116 - accuracy: 0.9480
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.2115 - accuracy: 0.9480
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.2114 - accuracy: 0.9480
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.2114 - accuracy: 0.9480
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.2113 - accuracy: 0.9480
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.2114 - accuracy: 0.9480
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.2114 - accuracy: 0.9480
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.2113 - accuracy: 0.9480
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.2112 - accuracy: 0.9480
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.2111 - accuracy: 0.9480
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.2111 - accuracy: 0.9480
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.2110 - accuracy: 0.9481
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.2110 - accuracy: 0.9481
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.2109 - accuracy: 0.9481
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.2109 - accuracy: 0.9481
 9700/11314 [========================>.....] - ETA: 1s - loss: 0.2108 - accuracy: 0.9481
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.2107 - accuracy: 0.9481
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.2107 - accuracy: 0.9481
10000/11314 [=========================>....] - ETA: 0s - loss: 0.2107 - accuracy: 0.9481
10100/11314 [=========================>....] - ETA: 0s - loss: 0.2107 - accuracy: 0.9481
10200/11314 [==========================>...] - ETA: 0s - loss: 0.2105 - accuracy: 0.9481
10300/11314 [==========================>...] - ETA: 0s - loss: 0.2106 - accuracy: 0.9481
10400/11314 [==========================>...] - ETA: 0s - loss: 0.2106 - accuracy: 0.9481
10500/11314 [==========================>...] - ETA: 0s - loss: 0.2106 - accuracy: 0.9481
10600/11314 [===========================>..] - ETA: 0s - loss: 0.2105 - accuracy: 0.9481
10700/11314 [===========================>..] - ETA: 0s - loss: 0.2104 - accuracy: 0.9480
10800/11314 [===========================>..] - ETA: 0s - loss: 0.2105 - accuracy: 0.9481
10900/11314 [===========================>..] - ETA: 0s - loss: 0.2104 - accuracy: 0.9481
11000/11314 [============================>.] - ETA: 0s - loss: 0.2104 - accuracy: 0.9481
11100/11314 [============================>.] - ETA: 0s - loss: 0.2103 - accuracy: 0.9481
11200/11314 [============================>.] - ETA: 0s - loss: 0.2103 - accuracy: 0.9481
11300/11314 [============================>.] - ETA: 0s - loss: 0.2103 - accuracy: 0.9481
11314/11314 [==============================] - 8s 750us/step - loss: 0.2103 - accuracy: 0.9481 - val_loss: 0.1968 - val_accuracy: 0.9496
Epoch 3/15

  100/11314 [..............................] - ETA: 6s - loss: 0.2045 - accuracy: 0.9495
  200/11314 [..............................] - ETA: 7s - loss: 0.2053 - accuracy: 0.9482
  300/11314 [..............................] - ETA: 6s - loss: 0.2065 - accuracy: 0.9472
  400/11314 [>.............................] - ETA: 6s - loss: 0.2072 - accuracy: 0.9475
  500/11314 [>.............................] - ETA: 6s - loss: 0.2071 - accuracy: 0.9474
  600/11314 [>.............................] - ETA: 6s - loss: 0.2059 - accuracy: 0.9476
  700/11314 [>.............................] - ETA: 6s - loss: 0.2052 - accuracy: 0.9480
  800/11314 [=>............................] - ETA: 6s - loss: 0.2048 - accuracy: 0.9483
  900/11314 [=>............................] - ETA: 6s - loss: 0.2048 - accuracy: 0.9482
 1000/11314 [=>............................] - ETA: 6s - loss: 0.2053 - accuracy: 0.9482
 1100/11314 [=>............................] - ETA: 6s - loss: 0.2055 - accuracy: 0.9482
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.2061 - accuracy: 0.9481
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.2067 - accuracy: 0.9481
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.2067 - accuracy: 0.9479
 1500/11314 [==>...........................] - ETA: 6s - loss: 0.2067 - accuracy: 0.9478
 1600/11314 [===>..........................] - ETA: 6s - loss: 0.2065 - accuracy: 0.9480
 1700/11314 [===>..........................] - ETA: 5s - loss: 0.2065 - accuracy: 0.9480
 1800/11314 [===>..........................] - ETA: 5s - loss: 0.2060 - accuracy: 0.9482
 1900/11314 [====>.........................] - ETA: 5s - loss: 0.2061 - accuracy: 0.9482
 2000/11314 [====>.........................] - ETA: 5s - loss: 0.2056 - accuracy: 0.9482
 2100/11314 [====>.........................] - ETA: 5s - loss: 0.2058 - accuracy: 0.9482
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.2056 - accuracy: 0.9482
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.2054 - accuracy: 0.9483
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.2051 - accuracy: 0.9483
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.2051 - accuracy: 0.9483
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.2049 - accuracy: 0.9483
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.2048 - accuracy: 0.9483
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.2047 - accuracy: 0.9483
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.2047 - accuracy: 0.9484
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.2048 - accuracy: 0.9483
 3100/11314 [=======>......................] - ETA: 5s - loss: 0.2046 - accuracy: 0.9484
 3200/11314 [=======>......................] - ETA: 5s - loss: 0.2046 - accuracy: 0.9483
 3300/11314 [=======>......................] - ETA: 4s - loss: 0.2045 - accuracy: 0.9483
 3400/11314 [========>.....................] - ETA: 4s - loss: 0.2045 - accuracy: 0.9483
 3500/11314 [========>.....................] - ETA: 4s - loss: 0.2044 - accuracy: 0.9483
 3600/11314 [========>.....................] - ETA: 4s - loss: 0.2042 - accuracy: 0.9484
 3700/11314 [========>.....................] - ETA: 4s - loss: 0.2043 - accuracy: 0.9483
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.2042 - accuracy: 0.9484
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.2042 - accuracy: 0.9484
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.2042 - accuracy: 0.9484
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.2041 - accuracy: 0.9484
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.2041 - accuracy: 0.9484
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.2039 - accuracy: 0.9485
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.2038 - accuracy: 0.9485
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.2038 - accuracy: 0.9485
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.2038 - accuracy: 0.9484
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.2039 - accuracy: 0.9485
 4800/11314 [===========>..................] - ETA: 4s - loss: 0.2038 - accuracy: 0.9485
 4900/11314 [===========>..................] - ETA: 3s - loss: 0.2037 - accuracy: 0.9485
 5000/11314 [============>.................] - ETA: 3s - loss: 0.2037 - accuracy: 0.9484
 5100/11314 [============>.................] - ETA: 3s - loss: 0.2035 - accuracy: 0.9485
 5200/11314 [============>.................] - ETA: 3s - loss: 0.2035 - accuracy: 0.9485
 5300/11314 [=============>................] - ETA: 3s - loss: 0.2034 - accuracy: 0.9485
 5400/11314 [=============>................] - ETA: 3s - loss: 0.2034 - accuracy: 0.9485
 5500/11314 [=============>................] - ETA: 3s - loss: 0.2033 - accuracy: 0.9485
 5600/11314 [=============>................] - ETA: 3s - loss: 0.2034 - accuracy: 0.9485
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.2032 - accuracy: 0.9486
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.2031 - accuracy: 0.9486
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.2030 - accuracy: 0.9486
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.2029 - accuracy: 0.9486
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.2029 - accuracy: 0.9487
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.2028 - accuracy: 0.9487
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.2028 - accuracy: 0.9487
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.2028 - accuracy: 0.9487
 6500/11314 [================>.............] - ETA: 2s - loss: 0.2027 - accuracy: 0.9487
 6600/11314 [================>.............] - ETA: 2s - loss: 0.2027 - accuracy: 0.9487
 6700/11314 [================>.............] - ETA: 2s - loss: 0.2026 - accuracy: 0.9487
 6800/11314 [=================>............] - ETA: 2s - loss: 0.2025 - accuracy: 0.9487
 6900/11314 [=================>............] - ETA: 2s - loss: 0.2025 - accuracy: 0.9488
 7000/11314 [=================>............] - ETA: 2s - loss: 0.2024 - accuracy: 0.9488
 7100/11314 [=================>............] - ETA: 2s - loss: 0.2023 - accuracy: 0.9488
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.2022 - accuracy: 0.9488
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.2023 - accuracy: 0.9487
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.2022 - accuracy: 0.9488
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.2022 - accuracy: 0.9488
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.2023 - accuracy: 0.9488
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.2022 - accuracy: 0.9488
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.2021 - accuracy: 0.9488
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.2021 - accuracy: 0.9488
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.2020 - accuracy: 0.9488
 8100/11314 [====================>.........] - ETA: 2s - loss: 0.2020 - accuracy: 0.9488
 8200/11314 [====================>.........] - ETA: 1s - loss: 0.2019 - accuracy: 0.9488
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.2019 - accuracy: 0.9488
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.2020 - accuracy: 0.9488
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.2019 - accuracy: 0.9488
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.2020 - accuracy: 0.9488
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.2021 - accuracy: 0.9488
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.2021 - accuracy: 0.9488
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.2021 - accuracy: 0.9488
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.2020 - accuracy: 0.9488
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.2020 - accuracy: 0.9488
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.2019 - accuracy: 0.9488
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.2019 - accuracy: 0.9488
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.2018 - accuracy: 0.9488
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.2018 - accuracy: 0.9488
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.2018 - accuracy: 0.9488
 9700/11314 [========================>.....] - ETA: 1s - loss: 0.2018 - accuracy: 0.9488
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.2018 - accuracy: 0.9488
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.2019 - accuracy: 0.9488
10000/11314 [=========================>....] - ETA: 0s - loss: 0.2018 - accuracy: 0.9488
10100/11314 [=========================>....] - ETA: 0s - loss: 0.2018 - accuracy: 0.9488
10200/11314 [==========================>...] - ETA: 0s - loss: 0.2017 - accuracy: 0.9489
10300/11314 [==========================>...] - ETA: 0s - loss: 0.2017 - accuracy: 0.9488
10400/11314 [==========================>...] - ETA: 0s - loss: 0.2017 - accuracy: 0.9489
10500/11314 [==========================>...] - ETA: 0s - loss: 0.2017 - accuracy: 0.9489
10600/11314 [===========================>..] - ETA: 0s - loss: 0.2016 - accuracy: 0.9489
10700/11314 [===========================>..] - ETA: 0s - loss: 0.2016 - accuracy: 0.9489
10800/11314 [===========================>..] - ETA: 0s - loss: 0.2016 - accuracy: 0.9489
10900/11314 [===========================>..] - ETA: 0s - loss: 0.2015 - accuracy: 0.9489
11000/11314 [============================>.] - ETA: 0s - loss: 0.2015 - accuracy: 0.9489
11100/11314 [============================>.] - ETA: 0s - loss: 0.2014 - accuracy: 0.9489
11200/11314 [============================>.] - ETA: 0s - loss: 0.2014 - accuracy: 0.9489
11300/11314 [============================>.] - ETA: 0s - loss: 0.2014 - accuracy: 0.9489
11314/11314 [==============================] - 8s 746us/step - loss: 0.2014 - accuracy: 0.9489 - val_loss: 0.1942 - val_accuracy: 0.9496
Epoch 4/15

  100/11314 [..............................] - ETA: 7s - loss: 0.1975 - accuracy: 0.9479
  200/11314 [..............................] - ETA: 7s - loss: 0.2001 - accuracy: 0.9474
  300/11314 [..............................] - ETA: 6s - loss: 0.1997 - accuracy: 0.9479
  400/11314 [>.............................] - ETA: 7s - loss: 0.1983 - accuracy: 0.9483
  500/11314 [>.............................] - ETA: 6s - loss: 0.1987 - accuracy: 0.9485
  600/11314 [>.............................] - ETA: 6s - loss: 0.1980 - accuracy: 0.9485
  700/11314 [>.............................] - ETA: 6s - loss: 0.1957 - accuracy: 0.9492
  800/11314 [=>............................] - ETA: 6s - loss: 0.1959 - accuracy: 0.9489
  900/11314 [=>............................] - ETA: 6s - loss: 0.1972 - accuracy: 0.9487
 1000/11314 [=>............................] - ETA: 6s - loss: 0.1975 - accuracy: 0.9486
 1100/11314 [=>............................] - ETA: 6s - loss: 0.1972 - accuracy: 0.9487
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.1978 - accuracy: 0.9485
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.1978 - accuracy: 0.9485
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.1987 - accuracy: 0.9485
 1500/11314 [==>...........................] - ETA: 6s - loss: 0.1989 - accuracy: 0.9484
 1600/11314 [===>..........................] - ETA: 6s - loss: 0.1985 - accuracy: 0.9485
 1700/11314 [===>..........................] - ETA: 5s - loss: 0.1987 - accuracy: 0.9485
 1800/11314 [===>..........................] - ETA: 5s - loss: 0.1986 - accuracy: 0.9486
 1900/11314 [====>.........................] - ETA: 5s - loss: 0.1981 - accuracy: 0.9487
 2000/11314 [====>.........................] - ETA: 5s - loss: 0.1980 - accuracy: 0.9488
 2100/11314 [====>.........................] - ETA: 5s - loss: 0.1980 - accuracy: 0.9488
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.1981 - accuracy: 0.9488
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.1981 - accuracy: 0.9487
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.1977 - accuracy: 0.9487
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.1977 - accuracy: 0.9487
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.1976 - accuracy: 0.9486
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.1975 - accuracy: 0.9486
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.1973 - accuracy: 0.9486
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.1975 - accuracy: 0.9485
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.1976 - accuracy: 0.9486
 3100/11314 [=======>......................] - ETA: 5s - loss: 0.1976 - accuracy: 0.9485
 3200/11314 [=======>......................] - ETA: 5s - loss: 0.1974 - accuracy: 0.9486
 3300/11314 [=======>......................] - ETA: 4s - loss: 0.1972 - accuracy: 0.9486
 3400/11314 [========>.....................] - ETA: 4s - loss: 0.1971 - accuracy: 0.9487
 3500/11314 [========>.....................] - ETA: 4s - loss: 0.1972 - accuracy: 0.9486
 3600/11314 [========>.....................] - ETA: 4s - loss: 0.1970 - accuracy: 0.9487
 3700/11314 [========>.....................] - ETA: 4s - loss: 0.1969 - accuracy: 0.9487
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.1967 - accuracy: 0.9488
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.1967 - accuracy: 0.9488
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.1967 - accuracy: 0.9488
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.1965 - accuracy: 0.9488
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.1965 - accuracy: 0.9488
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.1965 - accuracy: 0.9488
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.1966 - accuracy: 0.9488
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.1966 - accuracy: 0.9488
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.1965 - accuracy: 0.9488
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.1964 - accuracy: 0.9489
 4800/11314 [===========>..................] - ETA: 4s - loss: 0.1964 - accuracy: 0.9489
 4900/11314 [===========>..................] - ETA: 3s - loss: 0.1964 - accuracy: 0.9489
 5000/11314 [============>.................] - ETA: 3s - loss: 0.1964 - accuracy: 0.9488
 5100/11314 [============>.................] - ETA: 3s - loss: 0.1964 - accuracy: 0.9488
 5200/11314 [============>.................] - ETA: 3s - loss: 0.1964 - accuracy: 0.9488
 5300/11314 [=============>................] - ETA: 3s - loss: 0.1963 - accuracy: 0.9488
 5400/11314 [=============>................] - ETA: 3s - loss: 0.1964 - accuracy: 0.9488
 5500/11314 [=============>................] - ETA: 3s - loss: 0.1965 - accuracy: 0.9488
 5600/11314 [=============>................] - ETA: 3s - loss: 0.1966 - accuracy: 0.9488
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.1966 - accuracy: 0.9488
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.1965 - accuracy: 0.9488
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.1964 - accuracy: 0.9489
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.1964 - accuracy: 0.9489
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.1964 - accuracy: 0.9489
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.1964 - accuracy: 0.9489
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.1962 - accuracy: 0.9489
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.1961 - accuracy: 0.9489
 6500/11314 [================>.............] - ETA: 2s - loss: 0.1961 - accuracy: 0.9489
 6600/11314 [================>.............] - ETA: 2s - loss: 0.1961 - accuracy: 0.9489
 6700/11314 [================>.............] - ETA: 2s - loss: 0.1961 - accuracy: 0.9489
 6800/11314 [=================>............] - ETA: 2s - loss: 0.1961 - accuracy: 0.9489
 6900/11314 [=================>............] - ETA: 2s - loss: 0.1960 - accuracy: 0.9489
 7000/11314 [=================>............] - ETA: 2s - loss: 0.1959 - accuracy: 0.9489
 7100/11314 [=================>............] - ETA: 2s - loss: 0.1960 - accuracy: 0.9489
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.1960 - accuracy: 0.9489
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.1960 - accuracy: 0.9489
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.1959 - accuracy: 0.9489
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.1958 - accuracy: 0.9490
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.1959 - accuracy: 0.9489
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.1958 - accuracy: 0.9489
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.1959 - accuracy: 0.9489
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.1959 - accuracy: 0.9489
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.1959 - accuracy: 0.9489
 8100/11314 [====================>.........] - ETA: 1s - loss: 0.1958 - accuracy: 0.9490
 8200/11314 [====================>.........] - ETA: 1s - loss: 0.1957 - accuracy: 0.9490
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.1956 - accuracy: 0.9490
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.1956 - accuracy: 0.9490
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.1955 - accuracy: 0.9490
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.1955 - accuracy: 0.9490
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.1954 - accuracy: 0.9490
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.1954 - accuracy: 0.9491
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.1954 - accuracy: 0.9491
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.1955 - accuracy: 0.9491
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.1955 - accuracy: 0.9491
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.1955 - accuracy: 0.9491
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.1955 - accuracy: 0.9491
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.1954 - accuracy: 0.9491
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.1954 - accuracy: 0.9491
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.1954 - accuracy: 0.9491
 9700/11314 [========================>.....] - ETA: 0s - loss: 0.1954 - accuracy: 0.9491
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.1953 - accuracy: 0.9491
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.1951 - accuracy: 0.9491
10000/11314 [=========================>....] - ETA: 0s - loss: 0.1952 - accuracy: 0.9491
10100/11314 [=========================>....] - ETA: 0s - loss: 0.1952 - accuracy: 0.9491
10200/11314 [==========================>...] - ETA: 0s - loss: 0.1952 - accuracy: 0.9491
10300/11314 [==========================>...] - ETA: 0s - loss: 0.1952 - accuracy: 0.9491
10400/11314 [==========================>...] - ETA: 0s - loss: 0.1952 - accuracy: 0.9491
10500/11314 [==========================>...] - ETA: 0s - loss: 0.1952 - accuracy: 0.9491
10600/11314 [===========================>..] - ETA: 0s - loss: 0.1952 - accuracy: 0.9491
10700/11314 [===========================>..] - ETA: 0s - loss: 0.1951 - accuracy: 0.9491
10800/11314 [===========================>..] - ETA: 0s - loss: 0.1951 - accuracy: 0.9491
10900/11314 [===========================>..] - ETA: 0s - loss: 0.1951 - accuracy: 0.9491
11000/11314 [============================>.] - ETA: 0s - loss: 0.1952 - accuracy: 0.9491
11100/11314 [============================>.] - ETA: 0s - loss: 0.1951 - accuracy: 0.9491
11200/11314 [============================>.] - ETA: 0s - loss: 0.1951 - accuracy: 0.9491
11300/11314 [============================>.] - ETA: 0s - loss: 0.1950 - accuracy: 0.9491
11314/11314 [==============================] - 8s 730us/step - loss: 0.1950 - accuracy: 0.9491 - val_loss: 0.1913 - val_accuracy: 0.9496
Epoch 5/15

  100/11314 [..............................] - ETA: 7s - loss: 0.2014 - accuracy: 0.9479
  200/11314 [..............................] - ETA: 7s - loss: 0.1932 - accuracy: 0.9487
  300/11314 [..............................] - ETA: 6s - loss: 0.1951 - accuracy: 0.9482
  400/11314 [>.............................] - ETA: 6s - loss: 0.1942 - accuracy: 0.9483
  500/11314 [>.............................] - ETA: 6s - loss: 0.1927 - accuracy: 0.9487
  600/11314 [>.............................] - ETA: 6s - loss: 0.1928 - accuracy: 0.9483
  700/11314 [>.............................] - ETA: 6s - loss: 0.1914 - accuracy: 0.9484
  800/11314 [=>............................] - ETA: 6s - loss: 0.1912 - accuracy: 0.9484
  900/11314 [=>............................] - ETA: 6s - loss: 0.1917 - accuracy: 0.9483
 1000/11314 [=>............................] - ETA: 6s - loss: 0.1917 - accuracy: 0.9484
 1100/11314 [=>............................] - ETA: 6s - loss: 0.1912 - accuracy: 0.9484
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.1913 - accuracy: 0.9483
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.1901 - accuracy: 0.9487
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.1899 - accuracy: 0.9487
 1500/11314 [==>...........................] - ETA: 6s - loss: 0.1897 - accuracy: 0.9487
 1600/11314 [===>..........................] - ETA: 6s - loss: 0.1894 - accuracy: 0.9488
 1700/11314 [===>..........................] - ETA: 5s - loss: 0.1898 - accuracy: 0.9487
 1800/11314 [===>..........................] - ETA: 5s - loss: 0.1897 - accuracy: 0.9488
 1900/11314 [====>.........................] - ETA: 5s - loss: 0.1897 - accuracy: 0.9489
 2000/11314 [====>.........................] - ETA: 5s - loss: 0.1896 - accuracy: 0.9489
 2100/11314 [====>.........................] - ETA: 5s - loss: 0.1898 - accuracy: 0.9488
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.1900 - accuracy: 0.9488
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.1898 - accuracy: 0.9489
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.1896 - accuracy: 0.9489
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.1895 - accuracy: 0.9489
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.1895 - accuracy: 0.9488
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.1893 - accuracy: 0.9489
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.1893 - accuracy: 0.9489
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.1893 - accuracy: 0.9489
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.1893 - accuracy: 0.9489
 3100/11314 [=======>......................] - ETA: 5s - loss: 0.1894 - accuracy: 0.9489
 3200/11314 [=======>......................] - ETA: 5s - loss: 0.1894 - accuracy: 0.9489
 3300/11314 [=======>......................] - ETA: 4s - loss: 0.1894 - accuracy: 0.9489
 3400/11314 [========>.....................] - ETA: 4s - loss: 0.1895 - accuracy: 0.9489
 3500/11314 [========>.....................] - ETA: 4s - loss: 0.1896 - accuracy: 0.9489
 3600/11314 [========>.....................] - ETA: 4s - loss: 0.1898 - accuracy: 0.9488
 3700/11314 [========>.....................] - ETA: 4s - loss: 0.1895 - accuracy: 0.9489
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.1895 - accuracy: 0.9489
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.1895 - accuracy: 0.9490
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.1893 - accuracy: 0.9490
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.1892 - accuracy: 0.9490
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.1892 - accuracy: 0.9491
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.1891 - accuracy: 0.9491
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.1889 - accuracy: 0.9491
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.1890 - accuracy: 0.9491
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.1889 - accuracy: 0.9490
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.1888 - accuracy: 0.9490
 4800/11314 [===========>..................] - ETA: 4s - loss: 0.1889 - accuracy: 0.9490
 4900/11314 [===========>..................] - ETA: 3s - loss: 0.1888 - accuracy: 0.9490
 5000/11314 [============>.................] - ETA: 3s - loss: 0.1887 - accuracy: 0.9490
 5100/11314 [============>.................] - ETA: 3s - loss: 0.1886 - accuracy: 0.9491
 5200/11314 [============>.................] - ETA: 3s - loss: 0.1886 - accuracy: 0.9491
 5300/11314 [=============>................] - ETA: 3s - loss: 0.1885 - accuracy: 0.9491
 5400/11314 [=============>................] - ETA: 3s - loss: 0.1885 - accuracy: 0.9492
 5500/11314 [=============>................] - ETA: 3s - loss: 0.1884 - accuracy: 0.9492
 5600/11314 [=============>................] - ETA: 3s - loss: 0.1885 - accuracy: 0.9491
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.1884 - accuracy: 0.9491
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.1884 - accuracy: 0.9491
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.1884 - accuracy: 0.9491
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.1883 - accuracy: 0.9491
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.1884 - accuracy: 0.9491
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.1883 - accuracy: 0.9491
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.1884 - accuracy: 0.9491
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.1885 - accuracy: 0.9491
 6500/11314 [================>.............] - ETA: 2s - loss: 0.1885 - accuracy: 0.9491
 6600/11314 [================>.............] - ETA: 2s - loss: 0.1886 - accuracy: 0.9491
 6700/11314 [================>.............] - ETA: 2s - loss: 0.1886 - accuracy: 0.9491
 6800/11314 [=================>............] - ETA: 2s - loss: 0.1887 - accuracy: 0.9491
 6900/11314 [=================>............] - ETA: 2s - loss: 0.1887 - accuracy: 0.9491
 7000/11314 [=================>............] - ETA: 2s - loss: 0.1888 - accuracy: 0.9490
 7100/11314 [=================>............] - ETA: 2s - loss: 0.1888 - accuracy: 0.9490
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.1889 - accuracy: 0.9490
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.1889 - accuracy: 0.9490
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.1890 - accuracy: 0.9490
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.1891 - accuracy: 0.9490
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.1892 - accuracy: 0.9490
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.1893 - accuracy: 0.9490
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.1893 - accuracy: 0.9490
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.1894 - accuracy: 0.9490
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.1895 - accuracy: 0.9490
 8100/11314 [====================>.........] - ETA: 1s - loss: 0.1896 - accuracy: 0.9490
 8200/11314 [====================>.........] - ETA: 1s - loss: 0.1897 - accuracy: 0.9490
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.1897 - accuracy: 0.9490
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.1897 - accuracy: 0.9490
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.1898 - accuracy: 0.9490
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.1898 - accuracy: 0.9490
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.1898 - accuracy: 0.9490
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.1897 - accuracy: 0.9490
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.1898 - accuracy: 0.9490
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.1899 - accuracy: 0.9490
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.1899 - accuracy: 0.9490
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.1900 - accuracy: 0.9490
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.1901 - accuracy: 0.9490
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.1901 - accuracy: 0.9490
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.1901 - accuracy: 0.9490
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.1901 - accuracy: 0.9490
 9700/11314 [========================>.....] - ETA: 1s - loss: 0.1901 - accuracy: 0.9490
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.1900 - accuracy: 0.9490
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.1901 - accuracy: 0.9490
10000/11314 [=========================>....] - ETA: 0s - loss: 0.1900 - accuracy: 0.9490
10100/11314 [=========================>....] - ETA: 0s - loss: 0.1900 - accuracy: 0.9490
10200/11314 [==========================>...] - ETA: 0s - loss: 0.1900 - accuracy: 0.9490
10300/11314 [==========================>...] - ETA: 0s - loss: 0.1899 - accuracy: 0.9490
10400/11314 [==========================>...] - ETA: 0s - loss: 0.1899 - accuracy: 0.9490
10500/11314 [==========================>...] - ETA: 0s - loss: 0.1899 - accuracy: 0.9490
10600/11314 [===========================>..] - ETA: 0s - loss: 0.1899 - accuracy: 0.9490
10700/11314 [===========================>..] - ETA: 0s - loss: 0.1898 - accuracy: 0.9490
10800/11314 [===========================>..] - ETA: 0s - loss: 0.1898 - accuracy: 0.9490
10900/11314 [===========================>..] - ETA: 0s - loss: 0.1898 - accuracy: 0.9490
11000/11314 [============================>.] - ETA: 0s - loss: 0.1897 - accuracy: 0.9491
11100/11314 [============================>.] - ETA: 0s - loss: 0.1898 - accuracy: 0.9490
11200/11314 [============================>.] - ETA: 0s - loss: 0.1898 - accuracy: 0.9491
11300/11314 [============================>.] - ETA: 0s - loss: 0.1898 - accuracy: 0.9491
11314/11314 [==============================] - 8s 735us/step - loss: 0.1898 - accuracy: 0.9491 - val_loss: 0.1875 - val_accuracy: 0.9496
Epoch 6/15

  100/11314 [..............................] - ETA: 6s - loss: 0.1880 - accuracy: 0.9489
  200/11314 [..............................] - ETA: 6s - loss: 0.1892 - accuracy: 0.9492
  300/11314 [..............................] - ETA: 6s - loss: 0.1867 - accuracy: 0.9495
  400/11314 [>.............................] - ETA: 6s - loss: 0.1849 - accuracy: 0.9496
  500/11314 [>.............................] - ETA: 6s - loss: 0.1842 - accuracy: 0.9500
  600/11314 [>.............................] - ETA: 6s - loss: 0.1851 - accuracy: 0.9496
  700/11314 [>.............................] - ETA: 6s - loss: 0.1849 - accuracy: 0.9497
  800/11314 [=>............................] - ETA: 6s - loss: 0.1847 - accuracy: 0.9497
  900/11314 [=>............................] - ETA: 6s - loss: 0.1851 - accuracy: 0.9496
 1000/11314 [=>............................] - ETA: 6s - loss: 0.1849 - accuracy: 0.9497
 1100/11314 [=>............................] - ETA: 6s - loss: 0.1853 - accuracy: 0.9496
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.1856 - accuracy: 0.9496
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.1854 - accuracy: 0.9496
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.1853 - accuracy: 0.9496
 1500/11314 [==>...........................] - ETA: 6s - loss: 0.1854 - accuracy: 0.9496
 1600/11314 [===>..........................] - ETA: 6s - loss: 0.1857 - accuracy: 0.9495
 1700/11314 [===>..........................] - ETA: 5s - loss: 0.1854 - accuracy: 0.9496
 1800/11314 [===>..........................] - ETA: 5s - loss: 0.1854 - accuracy: 0.9495
 1900/11314 [====>.........................] - ETA: 5s - loss: 0.1852 - accuracy: 0.9496
 2000/11314 [====>.........................] - ETA: 5s - loss: 0.1851 - accuracy: 0.9496
 2100/11314 [====>.........................] - ETA: 5s - loss: 0.1852 - accuracy: 0.9496
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.1852 - accuracy: 0.9495
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.1849 - accuracy: 0.9496
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.1850 - accuracy: 0.9496
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.1850 - accuracy: 0.9496
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.1850 - accuracy: 0.9496
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.1849 - accuracy: 0.9496
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.1849 - accuracy: 0.9496
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.1848 - accuracy: 0.9496
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.1848 - accuracy: 0.9496
 3100/11314 [=======>......................] - ETA: 5s - loss: 0.1849 - accuracy: 0.9495
 3200/11314 [=======>......................] - ETA: 5s - loss: 0.1849 - accuracy: 0.9495
 3300/11314 [=======>......................] - ETA: 4s - loss: 0.1847 - accuracy: 0.9495
 3400/11314 [========>.....................] - ETA: 4s - loss: 0.1848 - accuracy: 0.9495
 3500/11314 [========>.....................] - ETA: 4s - loss: 0.1847 - accuracy: 0.9495
 3600/11314 [========>.....................] - ETA: 4s - loss: 0.1846 - accuracy: 0.9495
 3700/11314 [========>.....................] - ETA: 4s - loss: 0.1846 - accuracy: 0.9495
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.1846 - accuracy: 0.9494
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.1844 - accuracy: 0.9495
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.1843 - accuracy: 0.9495
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.1842 - accuracy: 0.9495
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.1843 - accuracy: 0.9494
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.1843 - accuracy: 0.9494
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.1843 - accuracy: 0.9494
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.1842 - accuracy: 0.9494
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.1842 - accuracy: 0.9495
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.1841 - accuracy: 0.9494
 4800/11314 [===========>..................] - ETA: 4s - loss: 0.1840 - accuracy: 0.9494
 4900/11314 [===========>..................] - ETA: 3s - loss: 0.1839 - accuracy: 0.9494
 5000/11314 [============>.................] - ETA: 3s - loss: 0.1837 - accuracy: 0.9494
 5100/11314 [============>.................] - ETA: 3s - loss: 0.1837 - accuracy: 0.9494
 5200/11314 [============>.................] - ETA: 3s - loss: 0.1837 - accuracy: 0.9494
 5300/11314 [=============>................] - ETA: 3s - loss: 0.1836 - accuracy: 0.9494
 5400/11314 [=============>................] - ETA: 3s - loss: 0.1834 - accuracy: 0.9495
 5500/11314 [=============>................] - ETA: 3s - loss: 0.1833 - accuracy: 0.9494
 5600/11314 [=============>................] - ETA: 3s - loss: 0.1833 - accuracy: 0.9494
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.1832 - accuracy: 0.9494
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.1833 - accuracy: 0.9494
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.1832 - accuracy: 0.9494
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.1831 - accuracy: 0.9494
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.1832 - accuracy: 0.9494
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.1832 - accuracy: 0.9494
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.1830 - accuracy: 0.9494
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.1830 - accuracy: 0.9494
 6500/11314 [================>.............] - ETA: 2s - loss: 0.1831 - accuracy: 0.9494
 6600/11314 [================>.............] - ETA: 2s - loss: 0.1830 - accuracy: 0.9494
 6700/11314 [================>.............] - ETA: 2s - loss: 0.1830 - accuracy: 0.9494
 6800/11314 [=================>............] - ETA: 2s - loss: 0.1829 - accuracy: 0.9494
 6900/11314 [=================>............] - ETA: 2s - loss: 0.1829 - accuracy: 0.9494
 7000/11314 [=================>............] - ETA: 2s - loss: 0.1829 - accuracy: 0.9493
 7100/11314 [=================>............] - ETA: 2s - loss: 0.1829 - accuracy: 0.9493
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.1830 - accuracy: 0.9493
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.1830 - accuracy: 0.9493
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.1831 - accuracy: 0.9493
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.1829 - accuracy: 0.9493
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.1830 - accuracy: 0.9493
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.1830 - accuracy: 0.9493
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.1831 - accuracy: 0.9493
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.1830 - accuracy: 0.9493
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.1830 - accuracy: 0.9493
 8100/11314 [====================>.........] - ETA: 2s - loss: 0.1830 - accuracy: 0.9493
 8200/11314 [====================>.........] - ETA: 1s - loss: 0.1830 - accuracy: 0.9493
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.1830 - accuracy: 0.9493
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.1830 - accuracy: 0.9492
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.1830 - accuracy: 0.9492
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.1830 - accuracy: 0.9492
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.1829 - accuracy: 0.9492
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.1829 - accuracy: 0.9492
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.1828 - accuracy: 0.9492
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.1828 - accuracy: 0.9492
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.1827 - accuracy: 0.9492
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.1827 - accuracy: 0.9492
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.1826 - accuracy: 0.9492
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.1826 - accuracy: 0.9492
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.1825 - accuracy: 0.9492
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.1824 - accuracy: 0.9493
 9700/11314 [========================>.....] - ETA: 1s - loss: 0.1824 - accuracy: 0.9493
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.1823 - accuracy: 0.9493
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.1822 - accuracy: 0.9493
10000/11314 [=========================>....] - ETA: 0s - loss: 0.1822 - accuracy: 0.9493
10100/11314 [=========================>....] - ETA: 0s - loss: 0.1822 - accuracy: 0.9493
10200/11314 [==========================>...] - ETA: 0s - loss: 0.1821 - accuracy: 0.9493
10300/11314 [==========================>...] - ETA: 0s - loss: 0.1819 - accuracy: 0.9493
10400/11314 [==========================>...] - ETA: 0s - loss: 0.1818 - accuracy: 0.9493
10500/11314 [==========================>...] - ETA: 0s - loss: 0.1818 - accuracy: 0.9493
10600/11314 [===========================>..] - ETA: 0s - loss: 0.1818 - accuracy: 0.9493
10700/11314 [===========================>..] - ETA: 0s - loss: 0.1816 - accuracy: 0.9493
10800/11314 [===========================>..] - ETA: 0s - loss: 0.1815 - accuracy: 0.9493
10900/11314 [===========================>..] - ETA: 0s - loss: 0.1814 - accuracy: 0.9493
11000/11314 [============================>.] - ETA: 0s - loss: 0.1814 - accuracy: 0.9493
11100/11314 [============================>.] - ETA: 0s - loss: 0.1813 - accuracy: 0.9493
11200/11314 [============================>.] - ETA: 0s - loss: 0.1813 - accuracy: 0.9493
11300/11314 [============================>.] - ETA: 0s - loss: 0.1812 - accuracy: 0.9493
11314/11314 [==============================] - 8s 735us/step - loss: 0.1812 - accuracy: 0.9493 - val_loss: 0.1793 - val_accuracy: 0.9496
Epoch 7/15

  100/11314 [..............................] - ETA: 6s - loss: 0.1687 - accuracy: 0.9495
  200/11314 [..............................] - ETA: 6s - loss: 0.1695 - accuracy: 0.9495
  300/11314 [..............................] - ETA: 6s - loss: 0.1718 - accuracy: 0.9489
  400/11314 [>.............................] - ETA: 6s - loss: 0.1705 - accuracy: 0.9488
  500/11314 [>.............................] - ETA: 6s - loss: 0.1718 - accuracy: 0.9489
  600/11314 [>.............................] - ETA: 6s - loss: 0.1705 - accuracy: 0.9490
  700/11314 [>.............................] - ETA: 6s - loss: 0.1707 - accuracy: 0.9491
  800/11314 [=>............................] - ETA: 6s - loss: 0.1707 - accuracy: 0.9491
  900/11314 [=>............................] - ETA: 6s - loss: 0.1711 - accuracy: 0.9489
 1000/11314 [=>............................] - ETA: 6s - loss: 0.1709 - accuracy: 0.9490
 1100/11314 [=>............................] - ETA: 6s - loss: 0.1704 - accuracy: 0.9491
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.1706 - accuracy: 0.9492
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.1705 - accuracy: 0.9492
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.1699 - accuracy: 0.9492
 1500/11314 [==>...........................] - ETA: 6s - loss: 0.1696 - accuracy: 0.9493
 1600/11314 [===>..........................] - ETA: 5s - loss: 0.1700 - accuracy: 0.9492
 1700/11314 [===>..........................] - ETA: 5s - loss: 0.1700 - accuracy: 0.9492
 1800/11314 [===>..........................] - ETA: 5s - loss: 0.1698 - accuracy: 0.9493
 1900/11314 [====>.........................] - ETA: 5s - loss: 0.1699 - accuracy: 0.9493
 2000/11314 [====>.........................] - ETA: 5s - loss: 0.1700 - accuracy: 0.9493
 2100/11314 [====>.........................] - ETA: 5s - loss: 0.1698 - accuracy: 0.9493
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.1696 - accuracy: 0.9494
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.1698 - accuracy: 0.9493
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.1697 - accuracy: 0.9493
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.1698 - accuracy: 0.9494
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.1697 - accuracy: 0.9494
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.1698 - accuracy: 0.9494
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.1699 - accuracy: 0.9494
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.1699 - accuracy: 0.9494
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.1702 - accuracy: 0.9493
 3100/11314 [=======>......................] - ETA: 5s - loss: 0.1703 - accuracy: 0.9493
 3200/11314 [=======>......................] - ETA: 5s - loss: 0.1702 - accuracy: 0.9493
 3300/11314 [=======>......................] - ETA: 4s - loss: 0.1702 - accuracy: 0.9493
 3400/11314 [========>.....................] - ETA: 4s - loss: 0.1699 - accuracy: 0.9494
 3500/11314 [========>.....................] - ETA: 4s - loss: 0.1698 - accuracy: 0.9494
 3600/11314 [========>.....................] - ETA: 4s - loss: 0.1697 - accuracy: 0.9494
 3700/11314 [========>.....................] - ETA: 4s - loss: 0.1698 - accuracy: 0.9494
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.1697 - accuracy: 0.9494
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.1697 - accuracy: 0.9495
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.1698 - accuracy: 0.9495
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.1698 - accuracy: 0.9495
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.1696 - accuracy: 0.9496
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.1697 - accuracy: 0.9496
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.1698 - accuracy: 0.9496
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.1697 - accuracy: 0.9496
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.1697 - accuracy: 0.9496
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.1697 - accuracy: 0.9496
 4800/11314 [===========>..................] - ETA: 4s - loss: 0.1697 - accuracy: 0.9496
 4900/11314 [===========>..................] - ETA: 4s - loss: 0.1697 - accuracy: 0.9496
 5000/11314 [============>.................] - ETA: 4s - loss: 0.1697 - accuracy: 0.9496
 5100/11314 [============>.................] - ETA: 3s - loss: 0.1698 - accuracy: 0.9495
 5200/11314 [============>.................] - ETA: 3s - loss: 0.1699 - accuracy: 0.9495
 5300/11314 [=============>................] - ETA: 3s - loss: 0.1700 - accuracy: 0.9495
 5400/11314 [=============>................] - ETA: 3s - loss: 0.1700 - accuracy: 0.9495
 5500/11314 [=============>................] - ETA: 3s - loss: 0.1700 - accuracy: 0.9495
 5600/11314 [=============>................] - ETA: 3s - loss: 0.1698 - accuracy: 0.9495
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.1698 - accuracy: 0.9495
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.1697 - accuracy: 0.9495
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.1695 - accuracy: 0.9495
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.1695 - accuracy: 0.9495
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.1694 - accuracy: 0.9495
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.1694 - accuracy: 0.9495
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.1693 - accuracy: 0.9495
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.1692 - accuracy: 0.9495
 6500/11314 [================>.............] - ETA: 3s - loss: 0.1693 - accuracy: 0.9495
 6600/11314 [================>.............] - ETA: 2s - loss: 0.1694 - accuracy: 0.9495
 6700/11314 [================>.............] - ETA: 2s - loss: 0.1694 - accuracy: 0.9495
 6800/11314 [=================>............] - ETA: 2s - loss: 0.1694 - accuracy: 0.9495
 6900/11314 [=================>............] - ETA: 2s - loss: 0.1695 - accuracy: 0.9495
 7000/11314 [=================>............] - ETA: 2s - loss: 0.1694 - accuracy: 0.9495
 7100/11314 [=================>............] - ETA: 2s - loss: 0.1693 - accuracy: 0.9495
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.1693 - accuracy: 0.9494
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.1692 - accuracy: 0.9494
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.1692 - accuracy: 0.9494
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.1693 - accuracy: 0.9494
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.1693 - accuracy: 0.9494
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.1693 - accuracy: 0.9494
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.1693 - accuracy: 0.9494
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.1694 - accuracy: 0.9494
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.1693 - accuracy: 0.9493
 8100/11314 [====================>.........] - ETA: 2s - loss: 0.1692 - accuracy: 0.9493
 8200/11314 [====================>.........] - ETA: 1s - loss: 0.1691 - accuracy: 0.9493
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.1691 - accuracy: 0.9493
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.1691 - accuracy: 0.9493
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.1691 - accuracy: 0.9493
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.1691 - accuracy: 0.9493
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.1691 - accuracy: 0.9493
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.1691 - accuracy: 0.9493
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.1691 - accuracy: 0.9493
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.1691 - accuracy: 0.9493
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.1692 - accuracy: 0.9493
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.1692 - accuracy: 0.9493
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.1691 - accuracy: 0.9493
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.1691 - accuracy: 0.9493
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.1690 - accuracy: 0.9493
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.1691 - accuracy: 0.9493
 9700/11314 [========================>.....] - ETA: 1s - loss: 0.1690 - accuracy: 0.9493
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.1690 - accuracy: 0.9494
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.1691 - accuracy: 0.9494
10000/11314 [=========================>....] - ETA: 0s - loss: 0.1691 - accuracy: 0.9493
10100/11314 [=========================>....] - ETA: 0s - loss: 0.1690 - accuracy: 0.9493
10200/11314 [==========================>...] - ETA: 0s - loss: 0.1691 - accuracy: 0.9493
10300/11314 [==========================>...] - ETA: 0s - loss: 0.1691 - accuracy: 0.9493
10400/11314 [==========================>...] - ETA: 0s - loss: 0.1690 - accuracy: 0.9494
10500/11314 [==========================>...] - ETA: 0s - loss: 0.1690 - accuracy: 0.9494
10600/11314 [===========================>..] - ETA: 0s - loss: 0.1691 - accuracy: 0.9494
10700/11314 [===========================>..] - ETA: 0s - loss: 0.1691 - accuracy: 0.9494
10800/11314 [===========================>..] - ETA: 0s - loss: 0.1692 - accuracy: 0.9493
10900/11314 [===========================>..] - ETA: 0s - loss: 0.1691 - accuracy: 0.9493
11000/11314 [============================>.] - ETA: 0s - loss: 0.1692 - accuracy: 0.9493
11100/11314 [============================>.] - ETA: 0s - loss: 0.1691 - accuracy: 0.9493
11200/11314 [============================>.] - ETA: 0s - loss: 0.1692 - accuracy: 0.9493
11300/11314 [============================>.] - ETA: 0s - loss: 0.1690 - accuracy: 0.9493
11314/11314 [==============================] - 8s 739us/step - loss: 0.1690 - accuracy: 0.9493 - val_loss: 0.1765 - val_accuracy: 0.9498
Epoch 8/15

  100/11314 [..............................] - ETA: 6s - loss: 0.1646 - accuracy: 0.9484
  200/11314 [..............................] - ETA: 6s - loss: 0.1654 - accuracy: 0.9492
  300/11314 [..............................] - ETA: 6s - loss: 0.1652 - accuracy: 0.9488
  400/11314 [>.............................] - ETA: 6s - loss: 0.1664 - accuracy: 0.9488
  500/11314 [>.............................] - ETA: 6s - loss: 0.1627 - accuracy: 0.9491
  600/11314 [>.............................] - ETA: 6s - loss: 0.1619 - accuracy: 0.9494
  700/11314 [>.............................] - ETA: 6s - loss: 0.1621 - accuracy: 0.9492
  800/11314 [=>............................] - ETA: 6s - loss: 0.1621 - accuracy: 0.9493
  900/11314 [=>............................] - ETA: 6s - loss: 0.1625 - accuracy: 0.9491
 1000/11314 [=>............................] - ETA: 6s - loss: 0.1620 - accuracy: 0.9489
 1100/11314 [=>............................] - ETA: 6s - loss: 0.1614 - accuracy: 0.9490
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.1611 - accuracy: 0.9490
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.1609 - accuracy: 0.9489
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.1607 - accuracy: 0.9489
 1500/11314 [==>...........................] - ETA: 6s - loss: 0.1607 - accuracy: 0.9490
 1600/11314 [===>..........................] - ETA: 6s - loss: 0.1608 - accuracy: 0.9491
 1700/11314 [===>..........................] - ETA: 5s - loss: 0.1617 - accuracy: 0.9490
 1800/11314 [===>..........................] - ETA: 5s - loss: 0.1617 - accuracy: 0.9490
 1900/11314 [====>.........................] - ETA: 5s - loss: 0.1621 - accuracy: 0.9490
 2000/11314 [====>.........................] - ETA: 5s - loss: 0.1619 - accuracy: 0.9490
 2100/11314 [====>.........................] - ETA: 5s - loss: 0.1618 - accuracy: 0.9490
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.1617 - accuracy: 0.9491
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.1614 - accuracy: 0.9492
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.1613 - accuracy: 0.9493
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.1612 - accuracy: 0.9493
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.1608 - accuracy: 0.9494
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.1607 - accuracy: 0.9494
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.1609 - accuracy: 0.9494
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.1609 - accuracy: 0.9493
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.1610 - accuracy: 0.9493
 3100/11314 [=======>......................] - ETA: 5s - loss: 0.1609 - accuracy: 0.9493
 3200/11314 [=======>......................] - ETA: 5s - loss: 0.1609 - accuracy: 0.9493
 3300/11314 [=======>......................] - ETA: 4s - loss: 0.1608 - accuracy: 0.9493
 3400/11314 [========>.....................] - ETA: 4s - loss: 0.1609 - accuracy: 0.9493
 3500/11314 [========>.....................] - ETA: 4s - loss: 0.1610 - accuracy: 0.9493
 3600/11314 [========>.....................] - ETA: 4s - loss: 0.1610 - accuracy: 0.9493
 3700/11314 [========>.....................] - ETA: 4s - loss: 0.1610 - accuracy: 0.9493
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.1612 - accuracy: 0.9493
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.1614 - accuracy: 0.9493
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.1614 - accuracy: 0.9493
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.1615 - accuracy: 0.9492
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.1616 - accuracy: 0.9493
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.1615 - accuracy: 0.9493
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.1614 - accuracy: 0.9492
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.1613 - accuracy: 0.9492
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.1614 - accuracy: 0.9492
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.1615 - accuracy: 0.9491
 4800/11314 [===========>..................] - ETA: 4s - loss: 0.1613 - accuracy: 0.9492
 4900/11314 [===========>..................] - ETA: 3s - loss: 0.1613 - accuracy: 0.9492
 5000/11314 [============>.................] - ETA: 3s - loss: 0.1612 - accuracy: 0.9492
 5100/11314 [============>.................] - ETA: 3s - loss: 0.1613 - accuracy: 0.9491
 5200/11314 [============>.................] - ETA: 3s - loss: 0.1612 - accuracy: 0.9492
 5300/11314 [=============>................] - ETA: 3s - loss: 0.1613 - accuracy: 0.9492
 5400/11314 [=============>................] - ETA: 3s - loss: 0.1613 - accuracy: 0.9491
 5500/11314 [=============>................] - ETA: 3s - loss: 0.1614 - accuracy: 0.9491
 5600/11314 [=============>................] - ETA: 3s - loss: 0.1613 - accuracy: 0.9491
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.1614 - accuracy: 0.9491
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.1614 - accuracy: 0.9491
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.1613 - accuracy: 0.9491
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.1612 - accuracy: 0.9491
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.1610 - accuracy: 0.9492
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.1611 - accuracy: 0.9492
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.1611 - accuracy: 0.9492
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.1609 - accuracy: 0.9492
 6500/11314 [================>.............] - ETA: 2s - loss: 0.1609 - accuracy: 0.9492
 6600/11314 [================>.............] - ETA: 2s - loss: 0.1609 - accuracy: 0.9492
 6700/11314 [================>.............] - ETA: 2s - loss: 0.1610 - accuracy: 0.9492
 6800/11314 [=================>............] - ETA: 2s - loss: 0.1609 - accuracy: 0.9492
 6900/11314 [=================>............] - ETA: 2s - loss: 0.1610 - accuracy: 0.9492
 7000/11314 [=================>............] - ETA: 2s - loss: 0.1611 - accuracy: 0.9492
 7100/11314 [=================>............] - ETA: 2s - loss: 0.1610 - accuracy: 0.9492
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.1610 - accuracy: 0.9492
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.1609 - accuracy: 0.9491
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.1608 - accuracy: 0.9492
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.1607 - accuracy: 0.9492
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.1607 - accuracy: 0.9492
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.1607 - accuracy: 0.9492
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.1606 - accuracy: 0.9492
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.1606 - accuracy: 0.9492
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.1605 - accuracy: 0.9492
 8100/11314 [====================>.........] - ETA: 1s - loss: 0.1604 - accuracy: 0.9492
 8200/11314 [====================>.........] - ETA: 1s - loss: 0.1604 - accuracy: 0.9492
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.1604 - accuracy: 0.9492
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.1603 - accuracy: 0.9492
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.1603 - accuracy: 0.9492
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.1602 - accuracy: 0.9492
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.1602 - accuracy: 0.9492
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.1602 - accuracy: 0.9492
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.1602 - accuracy: 0.9492
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.1600 - accuracy: 0.9492
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.1600 - accuracy: 0.9492
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.1600 - accuracy: 0.9492
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.1600 - accuracy: 0.9492
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.1599 - accuracy: 0.9492
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.1599 - accuracy: 0.9493
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.1599 - accuracy: 0.9492
 9700/11314 [========================>.....] - ETA: 1s - loss: 0.1599 - accuracy: 0.9492
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.1598 - accuracy: 0.9492
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.1598 - accuracy: 0.9493
10000/11314 [=========================>....] - ETA: 0s - loss: 0.1597 - accuracy: 0.9493
10100/11314 [=========================>....] - ETA: 0s - loss: 0.1597 - accuracy: 0.9493
10200/11314 [==========================>...] - ETA: 0s - loss: 0.1597 - accuracy: 0.9493
10300/11314 [==========================>...] - ETA: 0s - loss: 0.1596 - accuracy: 0.9493
10400/11314 [==========================>...] - ETA: 0s - loss: 0.1596 - accuracy: 0.9493
10500/11314 [==========================>...] - ETA: 0s - loss: 0.1596 - accuracy: 0.9493
10600/11314 [===========================>..] - ETA: 0s - loss: 0.1595 - accuracy: 0.9493
10700/11314 [===========================>..] - ETA: 0s - loss: 0.1594 - accuracy: 0.9493
10800/11314 [===========================>..] - ETA: 0s - loss: 0.1594 - accuracy: 0.9493
10900/11314 [===========================>..] - ETA: 0s - loss: 0.1594 - accuracy: 0.9493
11000/11314 [============================>.] - ETA: 0s - loss: 0.1594 - accuracy: 0.9493
11100/11314 [============================>.] - ETA: 0s - loss: 0.1594 - accuracy: 0.9493
11200/11314 [============================>.] - ETA: 0s - loss: 0.1594 - accuracy: 0.9493
11300/11314 [============================>.] - ETA: 0s - loss: 0.1593 - accuracy: 0.9493
11314/11314 [==============================] - 8s 735us/step - loss: 0.1593 - accuracy: 0.9493 - val_loss: 0.1679 - val_accuracy: 0.9498
Epoch 9/15

  100/11314 [..............................] - ETA: 6s - loss: 0.1560 - accuracy: 0.9489
  200/11314 [..............................] - ETA: 6s - loss: 0.1503 - accuracy: 0.9489
  300/11314 [..............................] - ETA: 6s - loss: 0.1496 - accuracy: 0.9491
  400/11314 [>.............................] - ETA: 6s - loss: 0.1497 - accuracy: 0.9497
  500/11314 [>.............................] - ETA: 6s - loss: 0.1489 - accuracy: 0.9499
  600/11314 [>.............................] - ETA: 6s - loss: 0.1493 - accuracy: 0.9497
  700/11314 [>.............................] - ETA: 6s - loss: 0.1493 - accuracy: 0.9497
  800/11314 [=>............................] - ETA: 6s - loss: 0.1503 - accuracy: 0.9495
  900/11314 [=>............................] - ETA: 6s - loss: 0.1507 - accuracy: 0.9494
 1000/11314 [=>............................] - ETA: 6s - loss: 0.1505 - accuracy: 0.9495
 1100/11314 [=>............................] - ETA: 6s - loss: 0.1508 - accuracy: 0.9496
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.1514 - accuracy: 0.9493
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.1515 - accuracy: 0.9493
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.1512 - accuracy: 0.9494
 1500/11314 [==>...........................] - ETA: 6s - loss: 0.1513 - accuracy: 0.9494
 1600/11314 [===>..........................] - ETA: 6s - loss: 0.1514 - accuracy: 0.9493
 1700/11314 [===>..........................] - ETA: 6s - loss: 0.1512 - accuracy: 0.9496
 1800/11314 [===>..........................] - ETA: 5s - loss: 0.1514 - accuracy: 0.9495
 1900/11314 [====>.........................] - ETA: 5s - loss: 0.1522 - accuracy: 0.9494
 2000/11314 [====>.........................] - ETA: 5s - loss: 0.1522 - accuracy: 0.9494
 2100/11314 [====>.........................] - ETA: 5s - loss: 0.1521 - accuracy: 0.9495
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.1517 - accuracy: 0.9495
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.1519 - accuracy: 0.9496
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.1516 - accuracy: 0.9497
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.1513 - accuracy: 0.9498
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.1517 - accuracy: 0.9498
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.1518 - accuracy: 0.9498
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.1513 - accuracy: 0.9499
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.1509 - accuracy: 0.9500
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.1510 - accuracy: 0.9499
 3100/11314 [=======>......................] - ETA: 5s - loss: 0.1510 - accuracy: 0.9500
 3200/11314 [=======>......................] - ETA: 5s - loss: 0.1512 - accuracy: 0.9499
 3300/11314 [=======>......................] - ETA: 5s - loss: 0.1510 - accuracy: 0.9500
 3400/11314 [========>.....................] - ETA: 4s - loss: 0.1508 - accuracy: 0.9500
 3500/11314 [========>.....................] - ETA: 4s - loss: 0.1510 - accuracy: 0.9500
 3600/11314 [========>.....................] - ETA: 4s - loss: 0.1511 - accuracy: 0.9500
 3700/11314 [========>.....................] - ETA: 4s - loss: 0.1513 - accuracy: 0.9499
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.1516 - accuracy: 0.9499
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.1516 - accuracy: 0.9499
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.1516 - accuracy: 0.9499
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.1517 - accuracy: 0.9498
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.1518 - accuracy: 0.9497
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.1523 - accuracy: 0.9497
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.1523 - accuracy: 0.9497
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.1522 - accuracy: 0.9497
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.1522 - accuracy: 0.9497
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.1523 - accuracy: 0.9497
 4800/11314 [===========>..................] - ETA: 4s - loss: 0.1524 - accuracy: 0.9497
 4900/11314 [===========>..................] - ETA: 3s - loss: 0.1523 - accuracy: 0.9496
 5000/11314 [============>.................] - ETA: 3s - loss: 0.1524 - accuracy: 0.9497
 5100/11314 [============>.................] - ETA: 3s - loss: 0.1526 - accuracy: 0.9497
 5200/11314 [============>.................] - ETA: 3s - loss: 0.1525 - accuracy: 0.9497
 5300/11314 [=============>................] - ETA: 3s - loss: 0.1525 - accuracy: 0.9496
 5400/11314 [=============>................] - ETA: 3s - loss: 0.1527 - accuracy: 0.9496
 5500/11314 [=============>................] - ETA: 3s - loss: 0.1526 - accuracy: 0.9496
 5600/11314 [=============>................] - ETA: 3s - loss: 0.1526 - accuracy: 0.9496
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.1526 - accuracy: 0.9496
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.1526 - accuracy: 0.9496
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.1525 - accuracy: 0.9496
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.1525 - accuracy: 0.9496
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.1525 - accuracy: 0.9496
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.1526 - accuracy: 0.9496
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.1525 - accuracy: 0.9496
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.1525 - accuracy: 0.9496
 6500/11314 [================>.............] - ETA: 3s - loss: 0.1524 - accuracy: 0.9496
 6600/11314 [================>.............] - ETA: 2s - loss: 0.1523 - accuracy: 0.9496
 6700/11314 [================>.............] - ETA: 2s - loss: 0.1522 - accuracy: 0.9496
 6800/11314 [=================>............] - ETA: 2s - loss: 0.1522 - accuracy: 0.9496
 6900/11314 [=================>............] - ETA: 2s - loss: 0.1521 - accuracy: 0.9495
 7000/11314 [=================>............] - ETA: 2s - loss: 0.1522 - accuracy: 0.9496
 7100/11314 [=================>............] - ETA: 2s - loss: 0.1522 - accuracy: 0.9495
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.1521 - accuracy: 0.9496
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.1520 - accuracy: 0.9496
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.1521 - accuracy: 0.9496
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.1522 - accuracy: 0.9496
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.1521 - accuracy: 0.9495
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.1522 - accuracy: 0.9495
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.1522 - accuracy: 0.9495
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.1522 - accuracy: 0.9495
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.1522 - accuracy: 0.9495
 8100/11314 [====================>.........] - ETA: 2s - loss: 0.1522 - accuracy: 0.9495
 8200/11314 [====================>.........] - ETA: 1s - loss: 0.1521 - accuracy: 0.9495
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.1520 - accuracy: 0.9495
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.1520 - accuracy: 0.9495
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.1519 - accuracy: 0.9495
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.1518 - accuracy: 0.9495
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.1518 - accuracy: 0.9495
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.1518 - accuracy: 0.9494
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.1517 - accuracy: 0.9494
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.1517 - accuracy: 0.9495
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.1517 - accuracy: 0.9495
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.1516 - accuracy: 0.9495
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.1516 - accuracy: 0.9495
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.1516 - accuracy: 0.9495
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.1517 - accuracy: 0.9495
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.1516 - accuracy: 0.9495
 9700/11314 [========================>.....] - ETA: 1s - loss: 0.1516 - accuracy: 0.9495
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.1516 - accuracy: 0.9495
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.1515 - accuracy: 0.9495
10000/11314 [=========================>....] - ETA: 0s - loss: 0.1515 - accuracy: 0.9495
10100/11314 [=========================>....] - ETA: 0s - loss: 0.1515 - accuracy: 0.9496
10200/11314 [==========================>...] - ETA: 0s - loss: 0.1514 - accuracy: 0.9496
10300/11314 [==========================>...] - ETA: 0s - loss: 0.1515 - accuracy: 0.9496
10400/11314 [==========================>...] - ETA: 0s - loss: 0.1515 - accuracy: 0.9496
10500/11314 [==========================>...] - ETA: 0s - loss: 0.1515 - accuracy: 0.9496
10600/11314 [===========================>..] - ETA: 0s - loss: 0.1515 - accuracy: 0.9496
10700/11314 [===========================>..] - ETA: 0s - loss: 0.1515 - accuracy: 0.9496
10800/11314 [===========================>..] - ETA: 0s - loss: 0.1516 - accuracy: 0.9496
10900/11314 [===========================>..] - ETA: 0s - loss: 0.1515 - accuracy: 0.9496
11000/11314 [============================>.] - ETA: 0s - loss: 0.1514 - accuracy: 0.9496
11100/11314 [============================>.] - ETA: 0s - loss: 0.1514 - accuracy: 0.9496
11200/11314 [============================>.] - ETA: 0s - loss: 0.1514 - accuracy: 0.9496
11300/11314 [============================>.] - ETA: 0s - loss: 0.1514 - accuracy: 0.9496
11314/11314 [==============================] - 8s 739us/step - loss: 0.1514 - accuracy: 0.9496 - val_loss: 0.1668 - val_accuracy: 0.9500
Epoch 10/15

  100/11314 [..............................] - ETA: 6s - loss: 0.1450 - accuracy: 0.9516
  200/11314 [..............................] - ETA: 6s - loss: 0.1448 - accuracy: 0.9516
  300/11314 [..............................] - ETA: 6s - loss: 0.1463 - accuracy: 0.9505
  400/11314 [>.............................] - ETA: 6s - loss: 0.1428 - accuracy: 0.9505
  500/11314 [>.............................] - ETA: 6s - loss: 0.1441 - accuracy: 0.9499
  600/11314 [>.............................] - ETA: 6s - loss: 0.1449 - accuracy: 0.9501
  700/11314 [>.............................] - ETA: 6s - loss: 0.1447 - accuracy: 0.9502
  800/11314 [=>............................] - ETA: 6s - loss: 0.1447 - accuracy: 0.9501
  900/11314 [=>............................] - ETA: 6s - loss: 0.1447 - accuracy: 0.9501
 1000/11314 [=>............................] - ETA: 6s - loss: 0.1443 - accuracy: 0.9501
 1100/11314 [=>............................] - ETA: 6s - loss: 0.1442 - accuracy: 0.9500
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.1443 - accuracy: 0.9503
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.1439 - accuracy: 0.9505
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.1439 - accuracy: 0.9504
 1500/11314 [==>...........................] - ETA: 6s - loss: 0.1441 - accuracy: 0.9503
 1600/11314 [===>..........................] - ETA: 6s - loss: 0.1434 - accuracy: 0.9504
 1700/11314 [===>..........................] - ETA: 6s - loss: 0.1438 - accuracy: 0.9502
 1800/11314 [===>..........................] - ETA: 6s - loss: 0.1442 - accuracy: 0.9502
 1900/11314 [====>.........................] - ETA: 6s - loss: 0.1441 - accuracy: 0.9502
 2000/11314 [====>.........................] - ETA: 6s - loss: 0.1440 - accuracy: 0.9501
 2100/11314 [====>.........................] - ETA: 5s - loss: 0.1436 - accuracy: 0.9499
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.1436 - accuracy: 0.9500
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.1436 - accuracy: 0.9500
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.1434 - accuracy: 0.9500
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.1433 - accuracy: 0.9500
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.1433 - accuracy: 0.9501
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.1433 - accuracy: 0.9501
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.1432 - accuracy: 0.9502
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.1435 - accuracy: 0.9502
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.1434 - accuracy: 0.9503
 3100/11314 [=======>......................] - ETA: 5s - loss: 0.1433 - accuracy: 0.9503
 3200/11314 [=======>......................] - ETA: 5s - loss: 0.1435 - accuracy: 0.9503
 3300/11314 [=======>......................] - ETA: 5s - loss: 0.1431 - accuracy: 0.9504
 3400/11314 [========>.....................] - ETA: 5s - loss: 0.1430 - accuracy: 0.9504
 3500/11314 [========>.....................] - ETA: 5s - loss: 0.1433 - accuracy: 0.9503
 3600/11314 [========>.....................] - ETA: 4s - loss: 0.1432 - accuracy: 0.9504
 3700/11314 [========>.....................] - ETA: 4s - loss: 0.1432 - accuracy: 0.9504
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.1432 - accuracy: 0.9504
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.1432 - accuracy: 0.9504
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.1431 - accuracy: 0.9503
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.1431 - accuracy: 0.9503
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.1431 - accuracy: 0.9503
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.1431 - accuracy: 0.9503
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.1430 - accuracy: 0.9503
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.1431 - accuracy: 0.9502
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.1430 - accuracy: 0.9502
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.1431 - accuracy: 0.9502
 4800/11314 [===========>..................] - ETA: 4s - loss: 0.1430 - accuracy: 0.9502
 4900/11314 [===========>..................] - ETA: 4s - loss: 0.1430 - accuracy: 0.9502
 5000/11314 [============>.................] - ETA: 4s - loss: 0.1427 - accuracy: 0.9502
 5100/11314 [============>.................] - ETA: 3s - loss: 0.1425 - accuracy: 0.9503
 5200/11314 [============>.................] - ETA: 3s - loss: 0.1425 - accuracy: 0.9503
 5300/11314 [=============>................] - ETA: 3s - loss: 0.1423 - accuracy: 0.9503
 5400/11314 [=============>................] - ETA: 3s - loss: 0.1424 - accuracy: 0.9503
 5500/11314 [=============>................] - ETA: 3s - loss: 0.1424 - accuracy: 0.9503
 5600/11314 [=============>................] - ETA: 3s - loss: 0.1425 - accuracy: 0.9503
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.1423 - accuracy: 0.9503
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.1422 - accuracy: 0.9503
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.1422 - accuracy: 0.9503
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.1422 - accuracy: 0.9504
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.1424 - accuracy: 0.9503
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.1424 - accuracy: 0.9503
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.1424 - accuracy: 0.9503
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.1425 - accuracy: 0.9503
 6500/11314 [================>.............] - ETA: 3s - loss: 0.1424 - accuracy: 0.9503
 6600/11314 [================>.............] - ETA: 3s - loss: 0.1425 - accuracy: 0.9503
 6700/11314 [================>.............] - ETA: 2s - loss: 0.1424 - accuracy: 0.9503
 6800/11314 [=================>............] - ETA: 2s - loss: 0.1424 - accuracy: 0.9503
 6900/11314 [=================>............] - ETA: 2s - loss: 0.1424 - accuracy: 0.9503
 7000/11314 [=================>............] - ETA: 2s - loss: 0.1425 - accuracy: 0.9503
 7100/11314 [=================>............] - ETA: 2s - loss: 0.1425 - accuracy: 0.9503
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.1425 - accuracy: 0.9503
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.1425 - accuracy: 0.9503
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.1425 - accuracy: 0.9503
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.1424 - accuracy: 0.9503
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.1423 - accuracy: 0.9503
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.1423 - accuracy: 0.9503
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.1423 - accuracy: 0.9503
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.1423 - accuracy: 0.9503
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.1423 - accuracy: 0.9503
 8100/11314 [====================>.........] - ETA: 2s - loss: 0.1423 - accuracy: 0.9503
 8200/11314 [====================>.........] - ETA: 1s - loss: 0.1423 - accuracy: 0.9503
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.1423 - accuracy: 0.9503
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.1422 - accuracy: 0.9503
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.1422 - accuracy: 0.9503
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.1421 - accuracy: 0.9503
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.1422 - accuracy: 0.9503
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.1422 - accuracy: 0.9503
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.1422 - accuracy: 0.9503
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.1422 - accuracy: 0.9503
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.1422 - accuracy: 0.9503
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.1422 - accuracy: 0.9503
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.1422 - accuracy: 0.9503
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.1423 - accuracy: 0.9503
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.1423 - accuracy: 0.9504
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.1422 - accuracy: 0.9504
 9700/11314 [========================>.....] - ETA: 1s - loss: 0.1421 - accuracy: 0.9505
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.1422 - accuracy: 0.9505
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.1421 - accuracy: 0.9505
10000/11314 [=========================>....] - ETA: 0s - loss: 0.1421 - accuracy: 0.9504
10100/11314 [=========================>....] - ETA: 0s - loss: 0.1421 - accuracy: 0.9505
10200/11314 [==========================>...] - ETA: 0s - loss: 0.1421 - accuracy: 0.9505
10300/11314 [==========================>...] - ETA: 0s - loss: 0.1421 - accuracy: 0.9505
10400/11314 [==========================>...] - ETA: 0s - loss: 0.1421 - accuracy: 0.9505
10500/11314 [==========================>...] - ETA: 0s - loss: 0.1421 - accuracy: 0.9505
10600/11314 [===========================>..] - ETA: 0s - loss: 0.1421 - accuracy: 0.9505
10700/11314 [===========================>..] - ETA: 0s - loss: 0.1421 - accuracy: 0.9505
10800/11314 [===========================>..] - ETA: 0s - loss: 0.1421 - accuracy: 0.9505
10900/11314 [===========================>..] - ETA: 0s - loss: 0.1421 - accuracy: 0.9506
11000/11314 [============================>.] - ETA: 0s - loss: 0.1420 - accuracy: 0.9506
11100/11314 [============================>.] - ETA: 0s - loss: 0.1421 - accuracy: 0.9505
11200/11314 [============================>.] - ETA: 0s - loss: 0.1420 - accuracy: 0.9506
11300/11314 [============================>.] - ETA: 0s - loss: 0.1420 - accuracy: 0.9505
11314/11314 [==============================] - 8s 749us/step - loss: 0.1419 - accuracy: 0.9506 - val_loss: 0.1631 - val_accuracy: 0.9506
Epoch 11/15

  100/11314 [..............................] - ETA: 6s - loss: 0.1345 - accuracy: 0.9521
  200/11314 [..............................] - ETA: 6s - loss: 0.1304 - accuracy: 0.9526
  300/11314 [..............................] - ETA: 6s - loss: 0.1305 - accuracy: 0.9523
  400/11314 [>.............................] - ETA: 6s - loss: 0.1282 - accuracy: 0.9524
  500/11314 [>.............................] - ETA: 6s - loss: 0.1277 - accuracy: 0.9531
  600/11314 [>.............................] - ETA: 6s - loss: 0.1303 - accuracy: 0.9525
  700/11314 [>.............................] - ETA: 6s - loss: 0.1312 - accuracy: 0.9514
  800/11314 [=>............................] - ETA: 6s - loss: 0.1306 - accuracy: 0.9516
  900/11314 [=>............................] - ETA: 6s - loss: 0.1310 - accuracy: 0.9516
 1000/11314 [=>............................] - ETA: 6s - loss: 0.1320 - accuracy: 0.9515
 1100/11314 [=>............................] - ETA: 6s - loss: 0.1330 - accuracy: 0.9512
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.1330 - accuracy: 0.9515
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.1327 - accuracy: 0.9517
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.1327 - accuracy: 0.9517
 1500/11314 [==>...........................] - ETA: 6s - loss: 0.1332 - accuracy: 0.9515
 1600/11314 [===>..........................] - ETA: 6s - loss: 0.1335 - accuracy: 0.9515
 1700/11314 [===>..........................] - ETA: 6s - loss: 0.1335 - accuracy: 0.9516
 1800/11314 [===>..........................] - ETA: 5s - loss: 0.1331 - accuracy: 0.9520
 1900/11314 [====>.........................] - ETA: 5s - loss: 0.1334 - accuracy: 0.9519
 2000/11314 [====>.........................] - ETA: 5s - loss: 0.1335 - accuracy: 0.9518
 2100/11314 [====>.........................] - ETA: 5s - loss: 0.1337 - accuracy: 0.9519
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.1336 - accuracy: 0.9521
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.1331 - accuracy: 0.9523
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.1331 - accuracy: 0.9523
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.1328 - accuracy: 0.9522
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.1331 - accuracy: 0.9520
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.1331 - accuracy: 0.9521
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.1331 - accuracy: 0.9521
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.1331 - accuracy: 0.9521
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.1331 - accuracy: 0.9521
 3100/11314 [=======>......................] - ETA: 5s - loss: 0.1331 - accuracy: 0.9520
 3200/11314 [=======>......................] - ETA: 5s - loss: 0.1330 - accuracy: 0.9520
 3300/11314 [=======>......................] - ETA: 5s - loss: 0.1331 - accuracy: 0.9519
 3400/11314 [========>.....................] - ETA: 4s - loss: 0.1337 - accuracy: 0.9519
 3500/11314 [========>.....................] - ETA: 4s - loss: 0.1335 - accuracy: 0.9520
 3600/11314 [========>.....................] - ETA: 4s - loss: 0.1336 - accuracy: 0.9520
 3700/11314 [========>.....................] - ETA: 4s - loss: 0.1335 - accuracy: 0.9520
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.1334 - accuracy: 0.9520
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.1333 - accuracy: 0.9520
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.1332 - accuracy: 0.9520
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.1332 - accuracy: 0.9521
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.1333 - accuracy: 0.9520
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.1334 - accuracy: 0.9520
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.1334 - accuracy: 0.9521
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.1334 - accuracy: 0.9521
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.1334 - accuracy: 0.9521
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.1335 - accuracy: 0.9520
 4800/11314 [===========>..................] - ETA: 4s - loss: 0.1334 - accuracy: 0.9521
 4900/11314 [===========>..................] - ETA: 4s - loss: 0.1334 - accuracy: 0.9521
 5000/11314 [============>.................] - ETA: 3s - loss: 0.1334 - accuracy: 0.9521
 5100/11314 [============>.................] - ETA: 3s - loss: 0.1335 - accuracy: 0.9521
 5200/11314 [============>.................] - ETA: 3s - loss: 0.1334 - accuracy: 0.9520
 5300/11314 [=============>................] - ETA: 3s - loss: 0.1333 - accuracy: 0.9521
 5400/11314 [=============>................] - ETA: 3s - loss: 0.1333 - accuracy: 0.9522
 5500/11314 [=============>................] - ETA: 3s - loss: 0.1333 - accuracy: 0.9523
 5600/11314 [=============>................] - ETA: 3s - loss: 0.1333 - accuracy: 0.9523
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.1332 - accuracy: 0.9524
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.1332 - accuracy: 0.9525
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.1332 - accuracy: 0.9524
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.1331 - accuracy: 0.9524
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.1332 - accuracy: 0.9524
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.1331 - accuracy: 0.9524
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.1331 - accuracy: 0.9524
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.1330 - accuracy: 0.9525
 6500/11314 [================>.............] - ETA: 3s - loss: 0.1330 - accuracy: 0.9525
 6600/11314 [================>.............] - ETA: 2s - loss: 0.1328 - accuracy: 0.9525
 6700/11314 [================>.............] - ETA: 2s - loss: 0.1328 - accuracy: 0.9526
 6800/11314 [=================>............] - ETA: 2s - loss: 0.1327 - accuracy: 0.9526
 6900/11314 [=================>............] - ETA: 2s - loss: 0.1327 - accuracy: 0.9526
 7000/11314 [=================>............] - ETA: 2s - loss: 0.1327 - accuracy: 0.9527
 7100/11314 [=================>............] - ETA: 2s - loss: 0.1326 - accuracy: 0.9527
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.1326 - accuracy: 0.9527
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.1325 - accuracy: 0.9527
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.1325 - accuracy: 0.9527
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.1326 - accuracy: 0.9527
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.1326 - accuracy: 0.9527
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.1325 - accuracy: 0.9527
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.1326 - accuracy: 0.9527
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.1325 - accuracy: 0.9527
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.1324 - accuracy: 0.9527
 8100/11314 [====================>.........] - ETA: 2s - loss: 0.1323 - accuracy: 0.9527
 8200/11314 [====================>.........] - ETA: 1s - loss: 0.1322 - accuracy: 0.9527
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.1321 - accuracy: 0.9528
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.1321 - accuracy: 0.9528
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.1320 - accuracy: 0.9528
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.1320 - accuracy: 0.9528
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.1319 - accuracy: 0.9527
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.1319 - accuracy: 0.9527
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.1318 - accuracy: 0.9528
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.1318 - accuracy: 0.9527
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.1318 - accuracy: 0.9527
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.1317 - accuracy: 0.9528
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.1317 - accuracy: 0.9528
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.1318 - accuracy: 0.9528
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.1318 - accuracy: 0.9528
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.1318 - accuracy: 0.9528
 9700/11314 [========================>.....] - ETA: 1s - loss: 0.1318 - accuracy: 0.9528
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.1317 - accuracy: 0.9528
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.1317 - accuracy: 0.9528
10000/11314 [=========================>....] - ETA: 0s - loss: 0.1317 - accuracy: 0.9528
10100/11314 [=========================>....] - ETA: 0s - loss: 0.1316 - accuracy: 0.9528
10200/11314 [==========================>...] - ETA: 0s - loss: 0.1316 - accuracy: 0.9528
10300/11314 [==========================>...] - ETA: 0s - loss: 0.1316 - accuracy: 0.9528
10400/11314 [==========================>...] - ETA: 0s - loss: 0.1315 - accuracy: 0.9528
10500/11314 [==========================>...] - ETA: 0s - loss: 0.1316 - accuracy: 0.9527
10600/11314 [===========================>..] - ETA: 0s - loss: 0.1315 - accuracy: 0.9527
10700/11314 [===========================>..] - ETA: 0s - loss: 0.1315 - accuracy: 0.9527
10800/11314 [===========================>..] - ETA: 0s - loss: 0.1314 - accuracy: 0.9527
10900/11314 [===========================>..] - ETA: 0s - loss: 0.1314 - accuracy: 0.9528
11000/11314 [============================>.] - ETA: 0s - loss: 0.1313 - accuracy: 0.9528
11100/11314 [============================>.] - ETA: 0s - loss: 0.1314 - accuracy: 0.9528
11200/11314 [============================>.] - ETA: 0s - loss: 0.1314 - accuracy: 0.9528
11300/11314 [============================>.] - ETA: 0s - loss: 0.1313 - accuracy: 0.9528
11314/11314 [==============================] - 8s 737us/step - loss: 0.1313 - accuracy: 0.9528 - val_loss: 0.1592 - val_accuracy: 0.9519
Epoch 12/15

  100/11314 [..............................] - ETA: 7s - loss: 0.1201 - accuracy: 0.9563
  200/11314 [..............................] - ETA: 6s - loss: 0.1223 - accuracy: 0.9553
  300/11314 [..............................] - ETA: 6s - loss: 0.1236 - accuracy: 0.9549
  400/11314 [>.............................] - ETA: 6s - loss: 0.1235 - accuracy: 0.9545
  500/11314 [>.............................] - ETA: 6s - loss: 0.1235 - accuracy: 0.9543
  600/11314 [>.............................] - ETA: 6s - loss: 0.1224 - accuracy: 0.9546
  700/11314 [>.............................] - ETA: 6s - loss: 0.1232 - accuracy: 0.9544
  800/11314 [=>............................] - ETA: 6s - loss: 0.1243 - accuracy: 0.9539
  900/11314 [=>............................] - ETA: 6s - loss: 0.1261 - accuracy: 0.9537
 1000/11314 [=>............................] - ETA: 6s - loss: 0.1259 - accuracy: 0.9533
 1100/11314 [=>............................] - ETA: 6s - loss: 0.1259 - accuracy: 0.9535
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.1255 - accuracy: 0.9537
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.1252 - accuracy: 0.9537
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.1248 - accuracy: 0.9540
 1500/11314 [==>...........................] - ETA: 6s - loss: 0.1244 - accuracy: 0.9541
 1600/11314 [===>..........................] - ETA: 5s - loss: 0.1245 - accuracy: 0.9540
 1700/11314 [===>..........................] - ETA: 5s - loss: 0.1249 - accuracy: 0.9540
 1800/11314 [===>..........................] - ETA: 5s - loss: 0.1247 - accuracy: 0.9542
 1900/11314 [====>.........................] - ETA: 5s - loss: 0.1249 - accuracy: 0.9543
 2000/11314 [====>.........................] - ETA: 5s - loss: 0.1252 - accuracy: 0.9541
 2100/11314 [====>.........................] - ETA: 5s - loss: 0.1262 - accuracy: 0.9540
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.1269 - accuracy: 0.9538
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.1278 - accuracy: 0.9539
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.1279 - accuracy: 0.9540
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.1282 - accuracy: 0.9540
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.1286 - accuracy: 0.9539
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.1286 - accuracy: 0.9539
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.1292 - accuracy: 0.9537
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.1295 - accuracy: 0.9537
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.1296 - accuracy: 0.9537
 3100/11314 [=======>......................] - ETA: 5s - loss: 0.1291 - accuracy: 0.9539
 3200/11314 [=======>......................] - ETA: 5s - loss: 0.1294 - accuracy: 0.9538
 3300/11314 [=======>......................] - ETA: 4s - loss: 0.1293 - accuracy: 0.9538
 3400/11314 [========>.....................] - ETA: 4s - loss: 0.1292 - accuracy: 0.9538
 3500/11314 [========>.....................] - ETA: 4s - loss: 0.1292 - accuracy: 0.9538
 3600/11314 [========>.....................] - ETA: 4s - loss: 0.1292 - accuracy: 0.9540
 3700/11314 [========>.....................] - ETA: 4s - loss: 0.1291 - accuracy: 0.9541
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.1291 - accuracy: 0.9541
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.1292 - accuracy: 0.9542
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.1294 - accuracy: 0.9542
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.1295 - accuracy: 0.9542
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.1296 - accuracy: 0.9542
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.1297 - accuracy: 0.9542
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.1297 - accuracy: 0.9542
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.1298 - accuracy: 0.9542
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.1298 - accuracy: 0.9542
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.1298 - accuracy: 0.9542
 4800/11314 [===========>..................] - ETA: 4s - loss: 0.1297 - accuracy: 0.9543
 4900/11314 [===========>..................] - ETA: 3s - loss: 0.1296 - accuracy: 0.9543
 5000/11314 [============>.................] - ETA: 3s - loss: 0.1295 - accuracy: 0.9544
 5100/11314 [============>.................] - ETA: 3s - loss: 0.1294 - accuracy: 0.9544
 5200/11314 [============>.................] - ETA: 3s - loss: 0.1293 - accuracy: 0.9544
 5300/11314 [=============>................] - ETA: 3s - loss: 0.1295 - accuracy: 0.9545
 5400/11314 [=============>................] - ETA: 3s - loss: 0.1292 - accuracy: 0.9545
 5500/11314 [=============>................] - ETA: 3s - loss: 0.1289 - accuracy: 0.9546
 5600/11314 [=============>................] - ETA: 3s - loss: 0.1289 - accuracy: 0.9547
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.1286 - accuracy: 0.9547
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.1288 - accuracy: 0.9547
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.1287 - accuracy: 0.9547
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.1285 - accuracy: 0.9547
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.1285 - accuracy: 0.9547
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.1285 - accuracy: 0.9547
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.1285 - accuracy: 0.9547
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.1284 - accuracy: 0.9547
 6500/11314 [================>.............] - ETA: 2s - loss: 0.1284 - accuracy: 0.9547
 6600/11314 [================>.............] - ETA: 2s - loss: 0.1284 - accuracy: 0.9546
 6700/11314 [================>.............] - ETA: 2s - loss: 0.1284 - accuracy: 0.9547
 6800/11314 [=================>............] - ETA: 2s - loss: 0.1284 - accuracy: 0.9546
 6900/11314 [=================>............] - ETA: 2s - loss: 0.1284 - accuracy: 0.9547
 7000/11314 [=================>............] - ETA: 2s - loss: 0.1284 - accuracy: 0.9546
 7100/11314 [=================>............] - ETA: 2s - loss: 0.1283 - accuracy: 0.9546
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.1282 - accuracy: 0.9547
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.1282 - accuracy: 0.9547
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.1281 - accuracy: 0.9548
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.1280 - accuracy: 0.9548
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.1280 - accuracy: 0.9549
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.1278 - accuracy: 0.9549
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.1278 - accuracy: 0.9549
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.1280 - accuracy: 0.9549
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.1280 - accuracy: 0.9549
 8100/11314 [====================>.........] - ETA: 1s - loss: 0.1281 - accuracy: 0.9548
 8200/11314 [====================>.........] - ETA: 1s - loss: 0.1282 - accuracy: 0.9548
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.1282 - accuracy: 0.9548
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.1282 - accuracy: 0.9548
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.1282 - accuracy: 0.9549
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.1283 - accuracy: 0.9549
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.1283 - accuracy: 0.9549
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.1283 - accuracy: 0.9549
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.1284 - accuracy: 0.9549
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.1284 - accuracy: 0.9549
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.1284 - accuracy: 0.9549
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.1284 - accuracy: 0.9549
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.1285 - accuracy: 0.9548
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.1285 - accuracy: 0.9548
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.1285 - accuracy: 0.9548
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.1284 - accuracy: 0.9548
 9700/11314 [========================>.....] - ETA: 1s - loss: 0.1284 - accuracy: 0.9548
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.1284 - accuracy: 0.9548
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.1285 - accuracy: 0.9547
10000/11314 [=========================>....] - ETA: 0s - loss: 0.1285 - accuracy: 0.9548
10100/11314 [=========================>....] - ETA: 0s - loss: 0.1284 - accuracy: 0.9548
10200/11314 [==========================>...] - ETA: 0s - loss: 0.1284 - accuracy: 0.9547
10300/11314 [==========================>...] - ETA: 0s - loss: 0.1284 - accuracy: 0.9547
10400/11314 [==========================>...] - ETA: 0s - loss: 0.1283 - accuracy: 0.9547
10500/11314 [==========================>...] - ETA: 0s - loss: 0.1284 - accuracy: 0.9547
10600/11314 [===========================>..] - ETA: 0s - loss: 0.1283 - accuracy: 0.9547
10700/11314 [===========================>..] - ETA: 0s - loss: 0.1283 - accuracy: 0.9547
10800/11314 [===========================>..] - ETA: 0s - loss: 0.1283 - accuracy: 0.9548
10900/11314 [===========================>..] - ETA: 0s - loss: 0.1282 - accuracy: 0.9548
11000/11314 [============================>.] - ETA: 0s - loss: 0.1281 - accuracy: 0.9549
11100/11314 [============================>.] - ETA: 0s - loss: 0.1280 - accuracy: 0.9549
11200/11314 [============================>.] - ETA: 0s - loss: 0.1280 - accuracy: 0.9549
11300/11314 [============================>.] - ETA: 0s - loss: 0.1280 - accuracy: 0.9549
11314/11314 [==============================] - 8s 734us/step - loss: 0.1280 - accuracy: 0.9549 - val_loss: 0.1562 - val_accuracy: 0.9535
Epoch 13/15

  100/11314 [..............................] - ETA: 6s - loss: 0.1177 - accuracy: 0.9595
  200/11314 [..............................] - ETA: 6s - loss: 0.1209 - accuracy: 0.9584
  300/11314 [..............................] - ETA: 6s - loss: 0.1191 - accuracy: 0.9589
  400/11314 [>.............................] - ETA: 6s - loss: 0.1190 - accuracy: 0.9575
  500/11314 [>.............................] - ETA: 6s - loss: 0.1176 - accuracy: 0.9572
  600/11314 [>.............................] - ETA: 6s - loss: 0.1185 - accuracy: 0.9573
  700/11314 [>.............................] - ETA: 6s - loss: 0.1186 - accuracy: 0.9573
  800/11314 [=>............................] - ETA: 6s - loss: 0.1183 - accuracy: 0.9571
  900/11314 [=>............................] - ETA: 6s - loss: 0.1188 - accuracy: 0.9573
 1000/11314 [=>............................] - ETA: 6s - loss: 0.1196 - accuracy: 0.9569
 1100/11314 [=>............................] - ETA: 6s - loss: 0.1196 - accuracy: 0.9570
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.1190 - accuracy: 0.9571
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.1193 - accuracy: 0.9566
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.1194 - accuracy: 0.9567
 1500/11314 [==>...........................] - ETA: 6s - loss: 0.1194 - accuracy: 0.9567
 1600/11314 [===>..........................] - ETA: 6s - loss: 0.1194 - accuracy: 0.9568
 1700/11314 [===>..........................] - ETA: 5s - loss: 0.1194 - accuracy: 0.9568
 1800/11314 [===>..........................] - ETA: 5s - loss: 0.1191 - accuracy: 0.9569
 1900/11314 [====>.........................] - ETA: 5s - loss: 0.1195 - accuracy: 0.9566
 2000/11314 [====>.........................] - ETA: 5s - loss: 0.1198 - accuracy: 0.9565
 2100/11314 [====>.........................] - ETA: 5s - loss: 0.1191 - accuracy: 0.9565
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.1195 - accuracy: 0.9566
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.1199 - accuracy: 0.9564
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.1198 - accuracy: 0.9563
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.1200 - accuracy: 0.9561
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.1198 - accuracy: 0.9563
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.1198 - accuracy: 0.9564
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.1196 - accuracy: 0.9566
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.1197 - accuracy: 0.9567
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.1196 - accuracy: 0.9568
 3100/11314 [=======>......................] - ETA: 5s - loss: 0.1194 - accuracy: 0.9569
 3200/11314 [=======>......................] - ETA: 5s - loss: 0.1192 - accuracy: 0.9569
 3300/11314 [=======>......................] - ETA: 4s - loss: 0.1189 - accuracy: 0.9571
 3400/11314 [========>.....................] - ETA: 4s - loss: 0.1190 - accuracy: 0.9572
 3500/11314 [========>.....................] - ETA: 4s - loss: 0.1189 - accuracy: 0.9572
 3600/11314 [========>.....................] - ETA: 4s - loss: 0.1189 - accuracy: 0.9573
 3700/11314 [========>.....................] - ETA: 4s - loss: 0.1187 - accuracy: 0.9573
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.1186 - accuracy: 0.9573
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.1184 - accuracy: 0.9574
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.1182 - accuracy: 0.9574
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.1180 - accuracy: 0.9575
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.1177 - accuracy: 0.9576
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.1177 - accuracy: 0.9575
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.1178 - accuracy: 0.9575
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.1178 - accuracy: 0.9575
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.1178 - accuracy: 0.9575
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.1178 - accuracy: 0.9574
 4800/11314 [===========>..................] - ETA: 4s - loss: 0.1175 - accuracy: 0.9575
 4900/11314 [===========>..................] - ETA: 3s - loss: 0.1173 - accuracy: 0.9576
 5000/11314 [============>.................] - ETA: 3s - loss: 0.1174 - accuracy: 0.9576
 5100/11314 [============>.................] - ETA: 3s - loss: 0.1174 - accuracy: 0.9576
 5200/11314 [============>.................] - ETA: 3s - loss: 0.1174 - accuracy: 0.9575
 5300/11314 [=============>................] - ETA: 3s - loss: 0.1174 - accuracy: 0.9575
 5400/11314 [=============>................] - ETA: 3s - loss: 0.1174 - accuracy: 0.9575
 5500/11314 [=============>................] - ETA: 3s - loss: 0.1174 - accuracy: 0.9574
 5600/11314 [=============>................] - ETA: 3s - loss: 0.1174 - accuracy: 0.9574
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.1174 - accuracy: 0.9574
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.1174 - accuracy: 0.9574
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.1174 - accuracy: 0.9574
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.1173 - accuracy: 0.9573
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.1172 - accuracy: 0.9573
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.1171 - accuracy: 0.9574
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.1170 - accuracy: 0.9574
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.1170 - accuracy: 0.9575
 6500/11314 [================>.............] - ETA: 2s - loss: 0.1169 - accuracy: 0.9575
 6600/11314 [================>.............] - ETA: 2s - loss: 0.1168 - accuracy: 0.9575
 6700/11314 [================>.............] - ETA: 2s - loss: 0.1169 - accuracy: 0.9575
 6800/11314 [=================>............] - ETA: 2s - loss: 0.1169 - accuracy: 0.9575
 6900/11314 [=================>............] - ETA: 2s - loss: 0.1168 - accuracy: 0.9575
 7000/11314 [=================>............] - ETA: 2s - loss: 0.1167 - accuracy: 0.9575
 7100/11314 [=================>............] - ETA: 2s - loss: 0.1167 - accuracy: 0.9575
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.1166 - accuracy: 0.9575
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.1165 - accuracy: 0.9575
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.1165 - accuracy: 0.9575
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.1165 - accuracy: 0.9576
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.1164 - accuracy: 0.9576
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.1164 - accuracy: 0.9576
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.1162 - accuracy: 0.9576
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.1161 - accuracy: 0.9578
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.1162 - accuracy: 0.9578
 8100/11314 [====================>.........] - ETA: 1s - loss: 0.1161 - accuracy: 0.9578
 8200/11314 [====================>.........] - ETA: 1s - loss: 0.1161 - accuracy: 0.9578
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.1160 - accuracy: 0.9578
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.1159 - accuracy: 0.9579
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.1158 - accuracy: 0.9580
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.1158 - accuracy: 0.9580
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.1158 - accuracy: 0.9580
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.1158 - accuracy: 0.9580
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.1159 - accuracy: 0.9580
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.1159 - accuracy: 0.9580
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.1160 - accuracy: 0.9580
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.1159 - accuracy: 0.9580
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.1159 - accuracy: 0.9580
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.1159 - accuracy: 0.9580
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.1159 - accuracy: 0.9580
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.1159 - accuracy: 0.9580
 9700/11314 [========================>.....] - ETA: 1s - loss: 0.1158 - accuracy: 0.9580
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.1157 - accuracy: 0.9580
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.1158 - accuracy: 0.9580
10000/11314 [=========================>....] - ETA: 0s - loss: 0.1157 - accuracy: 0.9581
10100/11314 [=========================>....] - ETA: 0s - loss: 0.1155 - accuracy: 0.9582
10200/11314 [==========================>...] - ETA: 0s - loss: 0.1156 - accuracy: 0.9581
10300/11314 [==========================>...] - ETA: 0s - loss: 0.1155 - accuracy: 0.9582
10400/11314 [==========================>...] - ETA: 0s - loss: 0.1155 - accuracy: 0.9582
10500/11314 [==========================>...] - ETA: 0s - loss: 0.1155 - accuracy: 0.9582
10600/11314 [===========================>..] - ETA: 0s - loss: 0.1155 - accuracy: 0.9582
10700/11314 [===========================>..] - ETA: 0s - loss: 0.1154 - accuracy: 0.9582
10800/11314 [===========================>..] - ETA: 0s - loss: 0.1153 - accuracy: 0.9582
10900/11314 [===========================>..] - ETA: 0s - loss: 0.1153 - accuracy: 0.9582
11000/11314 [============================>.] - ETA: 0s - loss: 0.1152 - accuracy: 0.9582
11100/11314 [============================>.] - ETA: 0s - loss: 0.1153 - accuracy: 0.9582
11200/11314 [============================>.] - ETA: 0s - loss: 0.1152 - accuracy: 0.9583
11300/11314 [============================>.] - ETA: 0s - loss: 0.1152 - accuracy: 0.9583
11314/11314 [==============================] - 8s 734us/step - loss: 0.1152 - accuracy: 0.9583 - val_loss: 0.1515 - val_accuracy: 0.9549
Epoch 14/15

  100/11314 [..............................] - ETA: 6s - loss: 0.1072 - accuracy: 0.9574
  200/11314 [..............................] - ETA: 7s - loss: 0.1056 - accuracy: 0.9608
  300/11314 [..............................] - ETA: 6s - loss: 0.1056 - accuracy: 0.9602
  400/11314 [>.............................] - ETA: 6s - loss: 0.1055 - accuracy: 0.9596
  500/11314 [>.............................] - ETA: 6s - loss: 0.1056 - accuracy: 0.9600
  600/11314 [>.............................] - ETA: 6s - loss: 0.1057 - accuracy: 0.9603
  700/11314 [>.............................] - ETA: 6s - loss: 0.1055 - accuracy: 0.9602
  800/11314 [=>............................] - ETA: 6s - loss: 0.1058 - accuracy: 0.9598
  900/11314 [=>............................] - ETA: 6s - loss: 0.1061 - accuracy: 0.9596
 1000/11314 [=>............................] - ETA: 6s - loss: 0.1056 - accuracy: 0.9598
 1100/11314 [=>............................] - ETA: 6s - loss: 0.1050 - accuracy: 0.9600
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.1051 - accuracy: 0.9599
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.1051 - accuracy: 0.9600
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.1049 - accuracy: 0.9599
 1500/11314 [==>...........................] - ETA: 6s - loss: 0.1052 - accuracy: 0.9602
 1600/11314 [===>..........................] - ETA: 6s - loss: 0.1060 - accuracy: 0.9602
 1700/11314 [===>..........................] - ETA: 5s - loss: 0.1062 - accuracy: 0.9603
 1800/11314 [===>..........................] - ETA: 5s - loss: 0.1062 - accuracy: 0.9601
 1900/11314 [====>.........................] - ETA: 5s - loss: 0.1064 - accuracy: 0.9601
 2000/11314 [====>.........................] - ETA: 5s - loss: 0.1065 - accuracy: 0.9600
 2100/11314 [====>.........................] - ETA: 5s - loss: 0.1064 - accuracy: 0.9599
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.1067 - accuracy: 0.9600
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.1068 - accuracy: 0.9600
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.1067 - accuracy: 0.9599
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.1066 - accuracy: 0.9601
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.1062 - accuracy: 0.9602
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.1057 - accuracy: 0.9605
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.1057 - accuracy: 0.9606
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.1057 - accuracy: 0.9605
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.1057 - accuracy: 0.9605
 3100/11314 [=======>......................] - ETA: 5s - loss: 0.1055 - accuracy: 0.9606
 3200/11314 [=======>......................] - ETA: 5s - loss: 0.1060 - accuracy: 0.9605
 3300/11314 [=======>......................] - ETA: 4s - loss: 0.1060 - accuracy: 0.9604
 3400/11314 [========>.....................] - ETA: 4s - loss: 0.1061 - accuracy: 0.9603
 3500/11314 [========>.....................] - ETA: 4s - loss: 0.1058 - accuracy: 0.9605
 3600/11314 [========>.....................] - ETA: 4s - loss: 0.1058 - accuracy: 0.9605
 3700/11314 [========>.....................] - ETA: 4s - loss: 0.1058 - accuracy: 0.9605
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.1058 - accuracy: 0.9606
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.1057 - accuracy: 0.9606
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.1057 - accuracy: 0.9607
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.1056 - accuracy: 0.9609
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.1059 - accuracy: 0.9607
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.1059 - accuracy: 0.9608
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.1061 - accuracy: 0.9608
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.1058 - accuracy: 0.9607
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.1059 - accuracy: 0.9607
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.1059 - accuracy: 0.9606
 4800/11314 [===========>..................] - ETA: 4s - loss: 0.1058 - accuracy: 0.9607
 4900/11314 [===========>..................] - ETA: 3s - loss: 0.1056 - accuracy: 0.9608
 5000/11314 [============>.................] - ETA: 3s - loss: 0.1058 - accuracy: 0.9607
 5100/11314 [============>.................] - ETA: 3s - loss: 0.1059 - accuracy: 0.9607
 5200/11314 [============>.................] - ETA: 3s - loss: 0.1059 - accuracy: 0.9606
 5300/11314 [=============>................] - ETA: 3s - loss: 0.1059 - accuracy: 0.9607
 5400/11314 [=============>................] - ETA: 3s - loss: 0.1059 - accuracy: 0.9606
 5500/11314 [=============>................] - ETA: 3s - loss: 0.1058 - accuracy: 0.9607
 5600/11314 [=============>................] - ETA: 3s - loss: 0.1058 - accuracy: 0.9607
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.1058 - accuracy: 0.9607
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.1058 - accuracy: 0.9607
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.1056 - accuracy: 0.9608
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.1055 - accuracy: 0.9608
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.1055 - accuracy: 0.9609
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.1052 - accuracy: 0.9610
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.1054 - accuracy: 0.9610
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.1054 - accuracy: 0.9609
 6500/11314 [================>.............] - ETA: 2s - loss: 0.1053 - accuracy: 0.9610
 6600/11314 [================>.............] - ETA: 2s - loss: 0.1052 - accuracy: 0.9610
 6700/11314 [================>.............] - ETA: 2s - loss: 0.1051 - accuracy: 0.9610
 6800/11314 [=================>............] - ETA: 2s - loss: 0.1051 - accuracy: 0.9609
 6900/11314 [=================>............] - ETA: 2s - loss: 0.1050 - accuracy: 0.9610
 7000/11314 [=================>............] - ETA: 2s - loss: 0.1050 - accuracy: 0.9609
 7100/11314 [=================>............] - ETA: 2s - loss: 0.1051 - accuracy: 0.9609
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.1052 - accuracy: 0.9609
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.1053 - accuracy: 0.9609
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.1054 - accuracy: 0.9610
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.1054 - accuracy: 0.9610
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.1056 - accuracy: 0.9610
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.1057 - accuracy: 0.9610
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.1059 - accuracy: 0.9609
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.1058 - accuracy: 0.9609
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.1057 - accuracy: 0.9610
 8100/11314 [====================>.........] - ETA: 1s - loss: 0.1057 - accuracy: 0.9609
 8200/11314 [====================>.........] - ETA: 1s - loss: 0.1057 - accuracy: 0.9609
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.1055 - accuracy: 0.9610
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.1055 - accuracy: 0.9610
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.1054 - accuracy: 0.9610
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.1054 - accuracy: 0.9611
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.1054 - accuracy: 0.9611
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.1055 - accuracy: 0.9610
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.1053 - accuracy: 0.9611
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.1052 - accuracy: 0.9611
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.1053 - accuracy: 0.9611
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.1054 - accuracy: 0.9612
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.1054 - accuracy: 0.9611
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.1054 - accuracy: 0.9611
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.1055 - accuracy: 0.9611
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.1055 - accuracy: 0.9611
 9700/11314 [========================>.....] - ETA: 1s - loss: 0.1055 - accuracy: 0.9611
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.1054 - accuracy: 0.9611
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.1054 - accuracy: 0.9611
10000/11314 [=========================>....] - ETA: 0s - loss: 0.1054 - accuracy: 0.9612
10100/11314 [=========================>....] - ETA: 0s - loss: 0.1053 - accuracy: 0.9612
10200/11314 [==========================>...] - ETA: 0s - loss: 0.1052 - accuracy: 0.9613
10300/11314 [==========================>...] - ETA: 0s - loss: 0.1052 - accuracy: 0.9613
10400/11314 [==========================>...] - ETA: 0s - loss: 0.1052 - accuracy: 0.9613
10500/11314 [==========================>...] - ETA: 0s - loss: 0.1052 - accuracy: 0.9613
10600/11314 [===========================>..] - ETA: 0s - loss: 0.1051 - accuracy: 0.9613
10700/11314 [===========================>..] - ETA: 0s - loss: 0.1051 - accuracy: 0.9613
10800/11314 [===========================>..] - ETA: 0s - loss: 0.1051 - accuracy: 0.9613
10900/11314 [===========================>..] - ETA: 0s - loss: 0.1050 - accuracy: 0.9614
11000/11314 [============================>.] - ETA: 0s - loss: 0.1050 - accuracy: 0.9614
11100/11314 [============================>.] - ETA: 0s - loss: 0.1050 - accuracy: 0.9614
11200/11314 [============================>.] - ETA: 0s - loss: 0.1050 - accuracy: 0.9614
11300/11314 [============================>.] - ETA: 0s - loss: 0.1050 - accuracy: 0.9614
11314/11314 [==============================] - 8s 733us/step - loss: 0.1050 - accuracy: 0.9614 - val_loss: 0.1496 - val_accuracy: 0.9559
Epoch 15/15

  100/11314 [..............................] - ETA: 6s - loss: 0.0989 - accuracy: 0.9637
  200/11314 [..............................] - ETA: 6s - loss: 0.1009 - accuracy: 0.9637
  300/11314 [..............................] - ETA: 6s - loss: 0.0997 - accuracy: 0.9640
  400/11314 [>.............................] - ETA: 6s - loss: 0.0994 - accuracy: 0.9632
  500/11314 [>.............................] - ETA: 6s - loss: 0.0994 - accuracy: 0.9627
  600/11314 [>.............................] - ETA: 6s - loss: 0.0988 - accuracy: 0.9628
  700/11314 [>.............................] - ETA: 6s - loss: 0.0988 - accuracy: 0.9630
  800/11314 [=>............................] - ETA: 6s - loss: 0.0994 - accuracy: 0.9632
  900/11314 [=>............................] - ETA: 6s - loss: 0.0988 - accuracy: 0.9633
 1000/11314 [=>............................] - ETA: 6s - loss: 0.0993 - accuracy: 0.9636
 1100/11314 [=>............................] - ETA: 6s - loss: 0.0991 - accuracy: 0.9634
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.0985 - accuracy: 0.9638
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.0986 - accuracy: 0.9640
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.0980 - accuracy: 0.9641
 1500/11314 [==>...........................] - ETA: 6s - loss: 0.0972 - accuracy: 0.9644
 1600/11314 [===>..........................] - ETA: 5s - loss: 0.0976 - accuracy: 0.9642
 1700/11314 [===>..........................] - ETA: 5s - loss: 0.0972 - accuracy: 0.9644
 1800/11314 [===>..........................] - ETA: 5s - loss: 0.0970 - accuracy: 0.9644
 1900/11314 [====>.........................] - ETA: 5s - loss: 0.0970 - accuracy: 0.9642
 2000/11314 [====>.........................] - ETA: 5s - loss: 0.0968 - accuracy: 0.9643
 2100/11314 [====>.........................] - ETA: 5s - loss: 0.0966 - accuracy: 0.9643
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.0969 - accuracy: 0.9640
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.0967 - accuracy: 0.9639
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.0965 - accuracy: 0.9639
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.0961 - accuracy: 0.9639
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.0961 - accuracy: 0.9640
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.0959 - accuracy: 0.9640
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.0959 - accuracy: 0.9641
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.0959 - accuracy: 0.9641
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.0964 - accuracy: 0.9639
 3100/11314 [=======>......................] - ETA: 5s - loss: 0.0964 - accuracy: 0.9640
 3200/11314 [=======>......................] - ETA: 5s - loss: 0.0962 - accuracy: 0.9641
 3300/11314 [=======>......................] - ETA: 4s - loss: 0.0963 - accuracy: 0.9641
 3400/11314 [========>.....................] - ETA: 4s - loss: 0.0965 - accuracy: 0.9641
 3500/11314 [========>.....................] - ETA: 4s - loss: 0.0964 - accuracy: 0.9641
 3600/11314 [========>.....................] - ETA: 4s - loss: 0.0963 - accuracy: 0.9641
 3700/11314 [========>.....................] - ETA: 4s - loss: 0.0962 - accuracy: 0.9641
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.0966 - accuracy: 0.9641
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.0967 - accuracy: 0.9641
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.0964 - accuracy: 0.9642
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.0965 - accuracy: 0.9642
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.0965 - accuracy: 0.9641
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.0966 - accuracy: 0.9641
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.0967 - accuracy: 0.9640
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.0967 - accuracy: 0.9640
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.0966 - accuracy: 0.9641
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.0967 - accuracy: 0.9641
 4800/11314 [===========>..................] - ETA: 4s - loss: 0.0967 - accuracy: 0.9641
 4900/11314 [===========>..................] - ETA: 3s - loss: 0.0968 - accuracy: 0.9639
 5000/11314 [============>.................] - ETA: 3s - loss: 0.0966 - accuracy: 0.9640
 5100/11314 [============>.................] - ETA: 3s - loss: 0.0967 - accuracy: 0.9640
 5200/11314 [============>.................] - ETA: 3s - loss: 0.0969 - accuracy: 0.9640
 5300/11314 [=============>................] - ETA: 3s - loss: 0.0969 - accuracy: 0.9640
 5400/11314 [=============>................] - ETA: 3s - loss: 0.0969 - accuracy: 0.9640
 5500/11314 [=============>................] - ETA: 3s - loss: 0.0970 - accuracy: 0.9640
 5600/11314 [=============>................] - ETA: 3s - loss: 0.0970 - accuracy: 0.9640
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.0970 - accuracy: 0.9640
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.0971 - accuracy: 0.9640
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.0970 - accuracy: 0.9640
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.0972 - accuracy: 0.9640
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.0971 - accuracy: 0.9640
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.0971 - accuracy: 0.9641
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.0973 - accuracy: 0.9640
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.0974 - accuracy: 0.9640
 6500/11314 [================>.............] - ETA: 2s - loss: 0.0974 - accuracy: 0.9640
 6600/11314 [================>.............] - ETA: 2s - loss: 0.0974 - accuracy: 0.9640
 6700/11314 [================>.............] - ETA: 2s - loss: 0.0974 - accuracy: 0.9640
 6800/11314 [=================>............] - ETA: 2s - loss: 0.0975 - accuracy: 0.9640
 6900/11314 [=================>............] - ETA: 2s - loss: 0.0974 - accuracy: 0.9641
 7000/11314 [=================>............] - ETA: 2s - loss: 0.0972 - accuracy: 0.9641
 7100/11314 [=================>............] - ETA: 2s - loss: 0.0972 - accuracy: 0.9641
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.0973 - accuracy: 0.9640
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.0975 - accuracy: 0.9640
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.0974 - accuracy: 0.9640
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.0973 - accuracy: 0.9641
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.0972 - accuracy: 0.9641
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.0972 - accuracy: 0.9641
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.0973 - accuracy: 0.9641
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.0973 - accuracy: 0.9641
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.0973 - accuracy: 0.9641
 8100/11314 [====================>.........] - ETA: 1s - loss: 0.0972 - accuracy: 0.9641
 8200/11314 [====================>.........] - ETA: 1s - loss: 0.0970 - accuracy: 0.9641
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.0970 - accuracy: 0.9641
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.0970 - accuracy: 0.9641
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.0970 - accuracy: 0.9641
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.0972 - accuracy: 0.9641
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.0972 - accuracy: 0.9640
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.0972 - accuracy: 0.9640
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.0971 - accuracy: 0.9640
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.0970 - accuracy: 0.9641
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.0969 - accuracy: 0.9642
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.0969 - accuracy: 0.9641
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.0969 - accuracy: 0.9642
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.0968 - accuracy: 0.9642
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.0967 - accuracy: 0.9642
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.0967 - accuracy: 0.9642
 9700/11314 [========================>.....] - ETA: 1s - loss: 0.0967 - accuracy: 0.9642
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.0966 - accuracy: 0.9642
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.0966 - accuracy: 0.9642
10000/11314 [=========================>....] - ETA: 0s - loss: 0.0966 - accuracy: 0.9642
10100/11314 [=========================>....] - ETA: 0s - loss: 0.0967 - accuracy: 0.9642
10200/11314 [==========================>...] - ETA: 0s - loss: 0.0966 - accuracy: 0.9642
10300/11314 [==========================>...] - ETA: 0s - loss: 0.0966 - accuracy: 0.9642
10400/11314 [==========================>...] - ETA: 0s - loss: 0.0967 - accuracy: 0.9642
10500/11314 [==========================>...] - ETA: 0s - loss: 0.0966 - accuracy: 0.9643
10600/11314 [===========================>..] - ETA: 0s - loss: 0.0968 - accuracy: 0.9642
10700/11314 [===========================>..] - ETA: 0s - loss: 0.0968 - accuracy: 0.9642
10800/11314 [===========================>..] - ETA: 0s - loss: 0.0969 - accuracy: 0.9641
10900/11314 [===========================>..] - ETA: 0s - loss: 0.0969 - accuracy: 0.9641
11000/11314 [============================>.] - ETA: 0s - loss: 0.0968 - accuracy: 0.9641
11100/11314 [============================>.] - ETA: 0s - loss: 0.0968 - accuracy: 0.9641
11200/11314 [============================>.] - ETA: 0s - loss: 0.0968 - accuracy: 0.9641
11300/11314 [============================>.] - ETA: 0s - loss: 0.0968 - accuracy: 0.9641
11314/11314 [==============================] - 8s 734us/step - loss: 0.0968 - accuracy: 0.9641 - val_loss: 0.1496 - val_accuracy: 0.9566
	=====> Test the model: model.predict()
	Dataset: TWENTY_NEWS_GROUPS
	Algorithm: Deep Learning using Keras 2 (KERAS_DL2)
	Training loss: 0.0822
	Training accuracy score: 96.96%
	Test loss: 0.1496
	Test accuracy score: 95.66%
	Training time: 127.3021
	Test time: 1.9339


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
	It took 25.186416625976562 seconds


Applying NLTK feature extraction: X_test
	==> Replace unused UNICODE characters. 
	==> Text in lower case. 
	==> Apply Lemmatize using WordNet's built-in morphy function. Returns the input word unchanged if it cannot be found in WordNet 
	==> Remove English stop words
	Done: NLTK feature extraction (replace unused UNICODE characters, text in lower case, apply Lemmatize using WordNet, remove English stop words)  finished!
	It took 24.592084407806396 seconds

	===> Tokenizer: fit_on_texts(X_train)
	===> X_train = pad_sequences(list_tokenized_train, maxlen=6000)
	===> Create Keras model
Model: "sequential_4"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_2 (Embedding)      (None, None, 128)         768000    
_________________________________________________________________
bidirectional_2 (Bidirection (None, None, 64)          41216     
_________________________________________________________________
global_max_pooling1d_2 (Glob (None, 64)                0         
_________________________________________________________________
dense_7 (Dense)              (None, 20)                1300      
_________________________________________________________________
dropout_2 (Dropout)          (None, 20)                0         
_________________________________________________________________
dense_8 (Dense)              (None, 1)                 21        
=================================================================
Total params: 810,537
Trainable params: 810,537
Non-trainable params: 0
_________________________________________________________________
None
	===> Tokenizer: fit_on_texts(X_test)
	===> X_test = pad_sequences(list_sentences_test, maxlen=6000)


NUMBER OF EPOCHS USED: 3

Train on 25000 samples, validate on 25000 samples
Epoch 1/3

  100/25000 [..............................] - ETA: 2:11 - loss: 0.6942 - accuracy: 0.3900
  200/25000 [..............................] - ETA: 1:13 - loss: 0.6943 - accuracy: 0.4200
  300/25000 [..............................] - ETA: 54s - loss: 0.6941 - accuracy: 0.4467 
  400/25000 [..............................] - ETA: 44s - loss: 0.6941 - accuracy: 0.4350
  500/25000 [..............................] - ETA: 38s - loss: 0.6941 - accuracy: 0.4360
  600/25000 [..............................] - ETA: 34s - loss: 0.6938 - accuracy: 0.4650
  700/25000 [..............................] - ETA: 31s - loss: 0.6936 - accuracy: 0.4757
  800/25000 [..............................] - ETA: 29s - loss: 0.6934 - accuracy: 0.4863
  900/25000 [>.............................] - ETA: 27s - loss: 0.6934 - accuracy: 0.4867
 1000/25000 [>.............................] - ETA: 26s - loss: 0.6934 - accuracy: 0.4840
 1100/25000 [>.............................] - ETA: 25s - loss: 0.6932 - accuracy: 0.4900
 1200/25000 [>.............................] - ETA: 24s - loss: 0.6930 - accuracy: 0.4975
 1300/25000 [>.............................] - ETA: 23s - loss: 0.6929 - accuracy: 0.5008
 1400/25000 [>.............................] - ETA: 22s - loss: 0.6929 - accuracy: 0.5029
 1500/25000 [>.............................] - ETA: 22s - loss: 0.6927 - accuracy: 0.5080
 1600/25000 [>.............................] - ETA: 21s - loss: 0.6925 - accuracy: 0.5144
 1700/25000 [=>............................] - ETA: 21s - loss: 0.6925 - accuracy: 0.5171
 1800/25000 [=>............................] - ETA: 21s - loss: 0.6923 - accuracy: 0.5222
 1900/25000 [=>............................] - ETA: 20s - loss: 0.6923 - accuracy: 0.5205
 2000/25000 [=>............................] - ETA: 20s - loss: 0.6921 - accuracy: 0.5260
 2100/25000 [=>............................] - ETA: 20s - loss: 0.6920 - accuracy: 0.5267
 2200/25000 [=>............................] - ETA: 19s - loss: 0.6919 - accuracy: 0.5323
 2300/25000 [=>............................] - ETA: 19s - loss: 0.6917 - accuracy: 0.5374
 2400/25000 [=>............................] - ETA: 19s - loss: 0.6915 - accuracy: 0.5454
 2500/25000 [==>...........................] - ETA: 18s - loss: 0.6913 - accuracy: 0.5524
 2600/25000 [==>...........................] - ETA: 18s - loss: 0.6912 - accuracy: 0.5608
 2700/25000 [==>...........................] - ETA: 18s - loss: 0.6910 - accuracy: 0.5656
 2800/25000 [==>...........................] - ETA: 18s - loss: 0.6909 - accuracy: 0.5711
 2900/25000 [==>...........................] - ETA: 17s - loss: 0.6908 - accuracy: 0.5748
 3000/25000 [==>...........................] - ETA: 17s - loss: 0.6905 - accuracy: 0.5800
 3100/25000 [==>...........................] - ETA: 17s - loss: 0.6903 - accuracy: 0.5858
 3200/25000 [==>...........................] - ETA: 17s - loss: 0.6900 - accuracy: 0.5931
 3300/25000 [==>...........................] - ETA: 17s - loss: 0.6897 - accuracy: 0.5985
 3400/25000 [===>..........................] - ETA: 16s - loss: 0.6894 - accuracy: 0.6032
 3500/25000 [===>..........................] - ETA: 16s - loss: 0.6891 - accuracy: 0.6060
 3600/25000 [===>..........................] - ETA: 16s - loss: 0.6887 - accuracy: 0.6097
 3700/25000 [===>..........................] - ETA: 16s - loss: 0.6882 - accuracy: 0.6151
 3800/25000 [===>..........................] - ETA: 16s - loss: 0.6878 - accuracy: 0.6182
 3900/25000 [===>..........................] - ETA: 16s - loss: 0.6872 - accuracy: 0.6221
 4000/25000 [===>..........................] - ETA: 15s - loss: 0.6866 - accuracy: 0.6260
 4100/25000 [===>..........................] - ETA: 15s - loss: 0.6858 - accuracy: 0.6293
 4200/25000 [====>.........................] - ETA: 15s - loss: 0.6851 - accuracy: 0.6319
 4300/25000 [====>.........................] - ETA: 15s - loss: 0.6839 - accuracy: 0.6363
 4400/25000 [====>.........................] - ETA: 15s - loss: 0.6828 - accuracy: 0.6395
 4500/25000 [====>.........................] - ETA: 15s - loss: 0.6811 - accuracy: 0.6427
 4600/25000 [====>.........................] - ETA: 15s - loss: 0.6788 - accuracy: 0.6474
 4700/25000 [====>.........................] - ETA: 14s - loss: 0.6765 - accuracy: 0.6500
 4800/25000 [====>.........................] - ETA: 14s - loss: 0.6737 - accuracy: 0.6540
 4900/25000 [====>.........................] - ETA: 14s - loss: 0.6706 - accuracy: 0.6571
 5000/25000 [=====>........................] - ETA: 14s - loss: 0.6672 - accuracy: 0.6596
 5100/25000 [=====>........................] - ETA: 14s - loss: 0.6633 - accuracy: 0.6627
 5200/25000 [=====>........................] - ETA: 14s - loss: 0.6605 - accuracy: 0.6640
 5300/25000 [=====>........................] - ETA: 14s - loss: 0.6570 - accuracy: 0.6672
 5400/25000 [=====>........................] - ETA: 14s - loss: 0.6526 - accuracy: 0.6698
 5500/25000 [=====>........................] - ETA: 14s - loss: 0.6492 - accuracy: 0.6729
 5600/25000 [=====>........................] - ETA: 13s - loss: 0.6470 - accuracy: 0.6746
 5700/25000 [=====>........................] - ETA: 13s - loss: 0.6440 - accuracy: 0.6768
 5800/25000 [=====>........................] - ETA: 13s - loss: 0.6409 - accuracy: 0.6790
 5900/25000 [======>.......................] - ETA: 13s - loss: 0.6377 - accuracy: 0.6805
 6000/25000 [======>.......................] - ETA: 13s - loss: 0.6350 - accuracy: 0.6823
 6100/25000 [======>.......................] - ETA: 13s - loss: 0.6307 - accuracy: 0.6849
 6200/25000 [======>.......................] - ETA: 13s - loss: 0.6276 - accuracy: 0.6873
 6300/25000 [======>.......................] - ETA: 13s - loss: 0.6230 - accuracy: 0.6906
 6400/25000 [======>.......................] - ETA: 13s - loss: 0.6194 - accuracy: 0.6933
 6500/25000 [======>.......................] - ETA: 13s - loss: 0.6167 - accuracy: 0.6954
 6600/25000 [======>.......................] - ETA: 12s - loss: 0.6127 - accuracy: 0.6979
 6700/25000 [=======>......................] - ETA: 12s - loss: 0.6093 - accuracy: 0.7000
 6800/25000 [=======>......................] - ETA: 12s - loss: 0.6063 - accuracy: 0.7018
 6900/25000 [=======>......................] - ETA: 12s - loss: 0.6034 - accuracy: 0.7035
 7000/25000 [=======>......................] - ETA: 12s - loss: 0.6021 - accuracy: 0.7043
 7100/25000 [=======>......................] - ETA: 12s - loss: 0.6008 - accuracy: 0.7055
 7200/25000 [=======>......................] - ETA: 12s - loss: 0.5981 - accuracy: 0.7071
 7300/25000 [=======>......................] - ETA: 12s - loss: 0.5953 - accuracy: 0.7085
 7400/25000 [=======>......................] - ETA: 12s - loss: 0.5927 - accuracy: 0.7104
 7500/25000 [========>.....................] - ETA: 12s - loss: 0.5913 - accuracy: 0.7115
 7600/25000 [========>.....................] - ETA: 12s - loss: 0.5891 - accuracy: 0.7130
 7700/25000 [========>.....................] - ETA: 11s - loss: 0.5858 - accuracy: 0.7148
 7800/25000 [========>.....................] - ETA: 11s - loss: 0.5835 - accuracy: 0.7163
 7900/25000 [========>.....................] - ETA: 11s - loss: 0.5807 - accuracy: 0.7181
 8000/25000 [========>.....................] - ETA: 11s - loss: 0.5786 - accuracy: 0.7195
 8100/25000 [========>.....................] - ETA: 11s - loss: 0.5762 - accuracy: 0.7211
 8200/25000 [========>.....................] - ETA: 11s - loss: 0.5749 - accuracy: 0.7223
 8300/25000 [========>.....................] - ETA: 11s - loss: 0.5711 - accuracy: 0.7246
 8400/25000 [=========>....................] - ETA: 11s - loss: 0.5684 - accuracy: 0.7261
 8500/25000 [=========>....................] - ETA: 11s - loss: 0.5669 - accuracy: 0.7267
 8600/25000 [=========>....................] - ETA: 11s - loss: 0.5659 - accuracy: 0.7273
 8700/25000 [=========>....................] - ETA: 11s - loss: 0.5637 - accuracy: 0.7289
 8800/25000 [=========>....................] - ETA: 11s - loss: 0.5619 - accuracy: 0.7297
 8900/25000 [=========>....................] - ETA: 11s - loss: 0.5600 - accuracy: 0.7306
 9000/25000 [=========>....................] - ETA: 10s - loss: 0.5582 - accuracy: 0.7316
 9100/25000 [=========>....................] - ETA: 10s - loss: 0.5565 - accuracy: 0.7330
 9200/25000 [==========>...................] - ETA: 10s - loss: 0.5554 - accuracy: 0.7335
 9300/25000 [==========>...................] - ETA: 10s - loss: 0.5542 - accuracy: 0.7345
 9400/25000 [==========>...................] - ETA: 10s - loss: 0.5525 - accuracy: 0.7356
 9500/25000 [==========>...................] - ETA: 10s - loss: 0.5502 - accuracy: 0.7372
 9600/25000 [==========>...................] - ETA: 10s - loss: 0.5488 - accuracy: 0.7383
 9700/25000 [==========>...................] - ETA: 10s - loss: 0.5462 - accuracy: 0.7401
 9800/25000 [==========>...................] - ETA: 10s - loss: 0.5440 - accuracy: 0.7412
 9900/25000 [==========>...................] - ETA: 10s - loss: 0.5419 - accuracy: 0.7425
10000/25000 [===========>..................] - ETA: 10s - loss: 0.5395 - accuracy: 0.7441
10100/25000 [===========>..................] - ETA: 10s - loss: 0.5386 - accuracy: 0.7443
10200/25000 [===========>..................] - ETA: 10s - loss: 0.5378 - accuracy: 0.7448
10300/25000 [===========>..................] - ETA: 9s - loss: 0.5369 - accuracy: 0.7452 
10400/25000 [===========>..................] - ETA: 9s - loss: 0.5346 - accuracy: 0.7465
10500/25000 [===========>..................] - ETA: 9s - loss: 0.5329 - accuracy: 0.7473
10600/25000 [===========>..................] - ETA: 9s - loss: 0.5314 - accuracy: 0.7483
10700/25000 [===========>..................] - ETA: 9s - loss: 0.5292 - accuracy: 0.7494
10800/25000 [===========>..................] - ETA: 9s - loss: 0.5279 - accuracy: 0.7504
10900/25000 [============>.................] - ETA: 9s - loss: 0.5264 - accuracy: 0.7510
11000/25000 [============>.................] - ETA: 9s - loss: 0.5246 - accuracy: 0.7523
11100/25000 [============>.................] - ETA: 9s - loss: 0.5236 - accuracy: 0.7530
11200/25000 [============>.................] - ETA: 9s - loss: 0.5223 - accuracy: 0.7538
11300/25000 [============>.................] - ETA: 9s - loss: 0.5218 - accuracy: 0.7538
11400/25000 [============>.................] - ETA: 9s - loss: 0.5200 - accuracy: 0.7549
11500/25000 [============>.................] - ETA: 9s - loss: 0.5195 - accuracy: 0.7556
11600/25000 [============>.................] - ETA: 8s - loss: 0.5178 - accuracy: 0.7566
11700/25000 [=============>................] - ETA: 8s - loss: 0.5162 - accuracy: 0.7572
11800/25000 [=============>................] - ETA: 8s - loss: 0.5151 - accuracy: 0.7577
11900/25000 [=============>................] - ETA: 8s - loss: 0.5147 - accuracy: 0.7577
12000/25000 [=============>................] - ETA: 8s - loss: 0.5134 - accuracy: 0.7588
12100/25000 [=============>................] - ETA: 8s - loss: 0.5127 - accuracy: 0.7592
12200/25000 [=============>................] - ETA: 8s - loss: 0.5109 - accuracy: 0.7602
12300/25000 [=============>................] - ETA: 8s - loss: 0.5106 - accuracy: 0.7602
12400/25000 [=============>................] - ETA: 8s - loss: 0.5088 - accuracy: 0.7613
12500/25000 [==============>...............] - ETA: 8s - loss: 0.5075 - accuracy: 0.7622
12600/25000 [==============>...............] - ETA: 8s - loss: 0.5067 - accuracy: 0.7626
12700/25000 [==============>...............] - ETA: 8s - loss: 0.5046 - accuracy: 0.7639
12800/25000 [==============>...............] - ETA: 8s - loss: 0.5033 - accuracy: 0.7648
12900/25000 [==============>...............] - ETA: 8s - loss: 0.5020 - accuracy: 0.7656
13000/25000 [==============>...............] - ETA: 7s - loss: 0.5011 - accuracy: 0.7662
13100/25000 [==============>...............] - ETA: 7s - loss: 0.4993 - accuracy: 0.7669
13200/25000 [==============>...............] - ETA: 7s - loss: 0.4979 - accuracy: 0.7675
13300/25000 [==============>...............] - ETA: 7s - loss: 0.4963 - accuracy: 0.7685
13400/25000 [===============>..............] - ETA: 7s - loss: 0.4951 - accuracy: 0.7690
13500/25000 [===============>..............] - ETA: 7s - loss: 0.4947 - accuracy: 0.7691
13600/25000 [===============>..............] - ETA: 7s - loss: 0.4937 - accuracy: 0.7697
13700/25000 [===============>..............] - ETA: 7s - loss: 0.4931 - accuracy: 0.7699
13800/25000 [===============>..............] - ETA: 7s - loss: 0.4921 - accuracy: 0.7704
13900/25000 [===============>..............] - ETA: 7s - loss: 0.4902 - accuracy: 0.7715
14000/25000 [===============>..............] - ETA: 7s - loss: 0.4887 - accuracy: 0.7723
14100/25000 [===============>..............] - ETA: 7s - loss: 0.4880 - accuracy: 0.7728
14200/25000 [================>.............] - ETA: 7s - loss: 0.4869 - accuracy: 0.7733
14300/25000 [================>.............] - ETA: 7s - loss: 0.4858 - accuracy: 0.7738
14400/25000 [================>.............] - ETA: 7s - loss: 0.4840 - accuracy: 0.7747
14500/25000 [================>.............] - ETA: 6s - loss: 0.4825 - accuracy: 0.7756
14600/25000 [================>.............] - ETA: 6s - loss: 0.4815 - accuracy: 0.7762
14700/25000 [================>.............] - ETA: 6s - loss: 0.4807 - accuracy: 0.7767
14800/25000 [================>.............] - ETA: 6s - loss: 0.4792 - accuracy: 0.7776
14900/25000 [================>.............] - ETA: 6s - loss: 0.4780 - accuracy: 0.7783
15000/25000 [=================>............] - ETA: 6s - loss: 0.4783 - accuracy: 0.7783
15100/25000 [=================>............] - ETA: 6s - loss: 0.4773 - accuracy: 0.7789
15200/25000 [=================>............] - ETA: 6s - loss: 0.4759 - accuracy: 0.7796
15300/25000 [=================>............] - ETA: 6s - loss: 0.4751 - accuracy: 0.7799
15400/25000 [=================>............] - ETA: 6s - loss: 0.4740 - accuracy: 0.7806
15500/25000 [=================>............] - ETA: 6s - loss: 0.4725 - accuracy: 0.7816
15600/25000 [=================>............] - ETA: 6s - loss: 0.4712 - accuracy: 0.7822
15700/25000 [=================>............] - ETA: 6s - loss: 0.4700 - accuracy: 0.7830
15800/25000 [=================>............] - ETA: 6s - loss: 0.4693 - accuracy: 0.7834
15900/25000 [==================>...........] - ETA: 5s - loss: 0.4686 - accuracy: 0.7837
16000/25000 [==================>...........] - ETA: 5s - loss: 0.4680 - accuracy: 0.7841
16100/25000 [==================>...........] - ETA: 5s - loss: 0.4674 - accuracy: 0.7845
16200/25000 [==================>...........] - ETA: 5s - loss: 0.4660 - accuracy: 0.7854
16300/25000 [==================>...........] - ETA: 5s - loss: 0.4654 - accuracy: 0.7858
16400/25000 [==================>...........] - ETA: 5s - loss: 0.4651 - accuracy: 0.7858
16500/25000 [==================>...........] - ETA: 5s - loss: 0.4644 - accuracy: 0.7863
16600/25000 [==================>...........] - ETA: 5s - loss: 0.4636 - accuracy: 0.7868
16700/25000 [===================>..........] - ETA: 5s - loss: 0.4628 - accuracy: 0.7872
16800/25000 [===================>..........] - ETA: 5s - loss: 0.4621 - accuracy: 0.7878
16900/25000 [===================>..........] - ETA: 5s - loss: 0.4615 - accuracy: 0.7882
17000/25000 [===================>..........] - ETA: 5s - loss: 0.4607 - accuracy: 0.7885
17100/25000 [===================>..........] - ETA: 5s - loss: 0.4597 - accuracy: 0.7892
17200/25000 [===================>..........] - ETA: 5s - loss: 0.4584 - accuracy: 0.7899
17300/25000 [===================>..........] - ETA: 5s - loss: 0.4577 - accuracy: 0.7902
17400/25000 [===================>..........] - ETA: 4s - loss: 0.4569 - accuracy: 0.7907
17500/25000 [====================>.........] - ETA: 4s - loss: 0.4561 - accuracy: 0.7910
17600/25000 [====================>.........] - ETA: 4s - loss: 0.4554 - accuracy: 0.7914
17700/25000 [====================>.........] - ETA: 4s - loss: 0.4548 - accuracy: 0.7918
17800/25000 [====================>.........] - ETA: 4s - loss: 0.4535 - accuracy: 0.7925
17900/25000 [====================>.........] - ETA: 4s - loss: 0.4523 - accuracy: 0.7931
18000/25000 [====================>.........] - ETA: 4s - loss: 0.4513 - accuracy: 0.7938
18100/25000 [====================>.........] - ETA: 4s - loss: 0.4512 - accuracy: 0.7940
18200/25000 [====================>.........] - ETA: 4s - loss: 0.4502 - accuracy: 0.7943
18300/25000 [====================>.........] - ETA: 4s - loss: 0.4499 - accuracy: 0.7945
18400/25000 [=====================>........] - ETA: 4s - loss: 0.4492 - accuracy: 0.7951
18500/25000 [=====================>........] - ETA: 4s - loss: 0.4490 - accuracy: 0.7952
18600/25000 [=====================>........] - ETA: 4s - loss: 0.4483 - accuracy: 0.7956
18700/25000 [=====================>........] - ETA: 4s - loss: 0.4471 - accuracy: 0.7963
18800/25000 [=====================>........] - ETA: 4s - loss: 0.4465 - accuracy: 0.7964
18900/25000 [=====================>........] - ETA: 3s - loss: 0.4454 - accuracy: 0.7970
19000/25000 [=====================>........] - ETA: 3s - loss: 0.4453 - accuracy: 0.7972
19100/25000 [=====================>........] - ETA: 3s - loss: 0.4446 - accuracy: 0.7974
19200/25000 [======================>.......] - ETA: 3s - loss: 0.4445 - accuracy: 0.7976
19300/25000 [======================>.......] - ETA: 3s - loss: 0.4437 - accuracy: 0.7980
19400/25000 [======================>.......] - ETA: 3s - loss: 0.4435 - accuracy: 0.7982
19500/25000 [======================>.......] - ETA: 3s - loss: 0.4428 - accuracy: 0.7984
19600/25000 [======================>.......] - ETA: 3s - loss: 0.4417 - accuracy: 0.7991
19700/25000 [======================>.......] - ETA: 3s - loss: 0.4407 - accuracy: 0.7995
19800/25000 [======================>.......] - ETA: 3s - loss: 0.4400 - accuracy: 0.7998
19900/25000 [======================>.......] - ETA: 3s - loss: 0.4389 - accuracy: 0.8004
20000/25000 [=======================>......] - ETA: 3s - loss: 0.4384 - accuracy: 0.8007
20100/25000 [=======================>......] - ETA: 3s - loss: 0.4372 - accuracy: 0.8015
20200/25000 [=======================>......] - ETA: 3s - loss: 0.4365 - accuracy: 0.8018
20300/25000 [=======================>......] - ETA: 3s - loss: 0.4358 - accuracy: 0.8022
20400/25000 [=======================>......] - ETA: 2s - loss: 0.4355 - accuracy: 0.8023
20500/25000 [=======================>......] - ETA: 2s - loss: 0.4345 - accuracy: 0.8029
20600/25000 [=======================>......] - ETA: 2s - loss: 0.4337 - accuracy: 0.8033
20700/25000 [=======================>......] - ETA: 2s - loss: 0.4332 - accuracy: 0.8035
20800/25000 [=======================>......] - ETA: 2s - loss: 0.4326 - accuracy: 0.8039
20900/25000 [========================>.....] - ETA: 2s - loss: 0.4318 - accuracy: 0.8044
21000/25000 [========================>.....] - ETA: 2s - loss: 0.4313 - accuracy: 0.8045
21100/25000 [========================>.....] - ETA: 2s - loss: 0.4310 - accuracy: 0.8045
21200/25000 [========================>.....] - ETA: 2s - loss: 0.4304 - accuracy: 0.8047
21300/25000 [========================>.....] - ETA: 2s - loss: 0.4299 - accuracy: 0.8049
21400/25000 [========================>.....] - ETA: 2s - loss: 0.4291 - accuracy: 0.8053
21500/25000 [========================>.....] - ETA: 2s - loss: 0.4286 - accuracy: 0.8057
21600/25000 [========================>.....] - ETA: 2s - loss: 0.4280 - accuracy: 0.8060
21700/25000 [=========================>....] - ETA: 2s - loss: 0.4280 - accuracy: 0.8061
21800/25000 [=========================>....] - ETA: 2s - loss: 0.4276 - accuracy: 0.8064
21900/25000 [=========================>....] - ETA: 2s - loss: 0.4270 - accuracy: 0.8067
22000/25000 [=========================>....] - ETA: 1s - loss: 0.4261 - accuracy: 0.8070
22100/25000 [=========================>....] - ETA: 1s - loss: 0.4258 - accuracy: 0.8071
22200/25000 [=========================>....] - ETA: 1s - loss: 0.4247 - accuracy: 0.8077
22300/25000 [=========================>....] - ETA: 1s - loss: 0.4244 - accuracy: 0.8079
22400/25000 [=========================>....] - ETA: 1s - loss: 0.4241 - accuracy: 0.8082
22500/25000 [==========================>...] - ETA: 1s - loss: 0.4237 - accuracy: 0.8083
22600/25000 [==========================>...] - ETA: 1s - loss: 0.4233 - accuracy: 0.8085
22700/25000 [==========================>...] - ETA: 1s - loss: 0.4228 - accuracy: 0.8086
22800/25000 [==========================>...] - ETA: 1s - loss: 0.4223 - accuracy: 0.8090
22900/25000 [==========================>...] - ETA: 1s - loss: 0.4216 - accuracy: 0.8093
23000/25000 [==========================>...] - ETA: 1s - loss: 0.4205 - accuracy: 0.8100
23100/25000 [==========================>...] - ETA: 1s - loss: 0.4203 - accuracy: 0.8101
23200/25000 [==========================>...] - ETA: 1s - loss: 0.4202 - accuracy: 0.8101
23300/25000 [==========================>...] - ETA: 1s - loss: 0.4198 - accuracy: 0.8104
23400/25000 [===========================>..] - ETA: 1s - loss: 0.4196 - accuracy: 0.8104
23500/25000 [===========================>..] - ETA: 0s - loss: 0.4187 - accuracy: 0.8109
23600/25000 [===========================>..] - ETA: 0s - loss: 0.4185 - accuracy: 0.8110
23700/25000 [===========================>..] - ETA: 0s - loss: 0.4177 - accuracy: 0.8115
23800/25000 [===========================>..] - ETA: 0s - loss: 0.4172 - accuracy: 0.8117
23900/25000 [===========================>..] - ETA: 0s - loss: 0.4163 - accuracy: 0.8122
24000/25000 [===========================>..] - ETA: 0s - loss: 0.4155 - accuracy: 0.8126
24100/25000 [===========================>..] - ETA: 0s - loss: 0.4150 - accuracy: 0.8129
24200/25000 [============================>.] - ETA: 0s - loss: 0.4142 - accuracy: 0.8133
24300/25000 [============================>.] - ETA: 0s - loss: 0.4135 - accuracy: 0.8137
24400/25000 [============================>.] - ETA: 0s - loss: 0.4127 - accuracy: 0.8141
24500/25000 [============================>.] - ETA: 0s - loss: 0.4125 - accuracy: 0.8143
24600/25000 [============================>.] - ETA: 0s - loss: 0.4125 - accuracy: 0.8145
24700/25000 [============================>.] - ETA: 0s - loss: 0.4123 - accuracy: 0.8147
24800/25000 [============================>.] - ETA: 0s - loss: 0.4119 - accuracy: 0.8150
24900/25000 [============================>.] - ETA: 0s - loss: 0.4113 - accuracy: 0.8153
25000/25000 [==============================] - 20s 809us/step - loss: 0.4108 - accuracy: 0.8157 - val_loss: 0.3092 - val_accuracy: 0.8662
Epoch 2/3

  100/25000 [..............................] - ETA: 14s - loss: 0.2067 - accuracy: 0.9300
  200/25000 [..............................] - ETA: 15s - loss: 0.1996 - accuracy: 0.9300
  300/25000 [..............................] - ETA: 15s - loss: 0.2044 - accuracy: 0.9267
  400/25000 [..............................] - ETA: 14s - loss: 0.2048 - accuracy: 0.9225
  500/25000 [..............................] - ETA: 14s - loss: 0.2151 - accuracy: 0.9220
  600/25000 [..............................] - ETA: 15s - loss: 0.2136 - accuracy: 0.9233
  700/25000 [..............................] - ETA: 15s - loss: 0.2137 - accuracy: 0.9243
  800/25000 [..............................] - ETA: 14s - loss: 0.2240 - accuracy: 0.9200
  900/25000 [>.............................] - ETA: 14s - loss: 0.2224 - accuracy: 0.9189
 1000/25000 [>.............................] - ETA: 14s - loss: 0.2213 - accuracy: 0.9170
 1100/25000 [>.............................] - ETA: 14s - loss: 0.2256 - accuracy: 0.9136
 1200/25000 [>.............................] - ETA: 14s - loss: 0.2222 - accuracy: 0.9142
 1300/25000 [>.............................] - ETA: 14s - loss: 0.2153 - accuracy: 0.9169
 1400/25000 [>.............................] - ETA: 14s - loss: 0.2175 - accuracy: 0.9150
 1500/25000 [>.............................] - ETA: 14s - loss: 0.2201 - accuracy: 0.9153
 1600/25000 [>.............................] - ETA: 14s - loss: 0.2213 - accuracy: 0.9150
 1700/25000 [=>............................] - ETA: 14s - loss: 0.2181 - accuracy: 0.9176
 1800/25000 [=>............................] - ETA: 14s - loss: 0.2148 - accuracy: 0.9200
 1900/25000 [=>............................] - ETA: 14s - loss: 0.2122 - accuracy: 0.9221
 2000/25000 [=>............................] - ETA: 14s - loss: 0.2103 - accuracy: 0.9230
 2100/25000 [=>............................] - ETA: 14s - loss: 0.2102 - accuracy: 0.9229
 2200/25000 [=>............................] - ETA: 14s - loss: 0.2076 - accuracy: 0.9245
 2300/25000 [=>............................] - ETA: 14s - loss: 0.2096 - accuracy: 0.9243
 2400/25000 [=>............................] - ETA: 14s - loss: 0.2119 - accuracy: 0.9233
 2500/25000 [==>...........................] - ETA: 13s - loss: 0.2102 - accuracy: 0.9244
 2600/25000 [==>...........................] - ETA: 13s - loss: 0.2110 - accuracy: 0.9238
 2700/25000 [==>...........................] - ETA: 13s - loss: 0.2091 - accuracy: 0.9241
 2800/25000 [==>...........................] - ETA: 13s - loss: 0.2149 - accuracy: 0.9218
 2900/25000 [==>...........................] - ETA: 13s - loss: 0.2129 - accuracy: 0.9224
 3000/25000 [==>...........................] - ETA: 13s - loss: 0.2135 - accuracy: 0.9223
 3100/25000 [==>...........................] - ETA: 13s - loss: 0.2140 - accuracy: 0.9213
 3200/25000 [==>...........................] - ETA: 13s - loss: 0.2149 - accuracy: 0.9200
 3300/25000 [==>...........................] - ETA: 13s - loss: 0.2118 - accuracy: 0.9209
 3400/25000 [===>..........................] - ETA: 13s - loss: 0.2159 - accuracy: 0.9182
 3500/25000 [===>..........................] - ETA: 13s - loss: 0.2157 - accuracy: 0.9189
 3600/25000 [===>..........................] - ETA: 13s - loss: 0.2157 - accuracy: 0.9186
 3700/25000 [===>..........................] - ETA: 13s - loss: 0.2175 - accuracy: 0.9184
 3800/25000 [===>..........................] - ETA: 13s - loss: 0.2192 - accuracy: 0.9176
 3900/25000 [===>..........................] - ETA: 13s - loss: 0.2206 - accuracy: 0.9169
 4000/25000 [===>..........................] - ETA: 13s - loss: 0.2214 - accuracy: 0.9170
 4100/25000 [===>..........................] - ETA: 12s - loss: 0.2214 - accuracy: 0.9166
 4200/25000 [====>.........................] - ETA: 12s - loss: 0.2206 - accuracy: 0.9176
 4300/25000 [====>.........................] - ETA: 12s - loss: 0.2224 - accuracy: 0.9172
 4400/25000 [====>.........................] - ETA: 12s - loss: 0.2239 - accuracy: 0.9161
 4500/25000 [====>.........................] - ETA: 12s - loss: 0.2236 - accuracy: 0.9158
 4600/25000 [====>.........................] - ETA: 12s - loss: 0.2235 - accuracy: 0.9159
 4700/25000 [====>.........................] - ETA: 12s - loss: 0.2241 - accuracy: 0.9155
 4800/25000 [====>.........................] - ETA: 12s - loss: 0.2231 - accuracy: 0.9165
 4900/25000 [====>.........................] - ETA: 12s - loss: 0.2256 - accuracy: 0.9159
 5000/25000 [=====>........................] - ETA: 12s - loss: 0.2256 - accuracy: 0.9154
 5100/25000 [=====>........................] - ETA: 12s - loss: 0.2238 - accuracy: 0.9165
 5200/25000 [=====>........................] - ETA: 12s - loss: 0.2250 - accuracy: 0.9158
 5300/25000 [=====>........................] - ETA: 12s - loss: 0.2247 - accuracy: 0.9157
 5400/25000 [=====>........................] - ETA: 12s - loss: 0.2270 - accuracy: 0.9144
 5500/25000 [=====>........................] - ETA: 12s - loss: 0.2276 - accuracy: 0.9144
 5600/25000 [=====>........................] - ETA: 12s - loss: 0.2255 - accuracy: 0.9155
 5700/25000 [=====>........................] - ETA: 11s - loss: 0.2253 - accuracy: 0.9160
 5800/25000 [=====>........................] - ETA: 11s - loss: 0.2251 - accuracy: 0.9162
 5900/25000 [======>.......................] - ETA: 11s - loss: 0.2253 - accuracy: 0.9163
 6000/25000 [======>.......................] - ETA: 11s - loss: 0.2264 - accuracy: 0.9157
 6100/25000 [======>.......................] - ETA: 11s - loss: 0.2252 - accuracy: 0.9162
 6200/25000 [======>.......................] - ETA: 11s - loss: 0.2260 - accuracy: 0.9160
 6300/25000 [======>.......................] - ETA: 11s - loss: 0.2256 - accuracy: 0.9159
 6400/25000 [======>.......................] - ETA: 11s - loss: 0.2273 - accuracy: 0.9144
 6500/25000 [======>.......................] - ETA: 11s - loss: 0.2276 - accuracy: 0.9140
 6600/25000 [======>.......................] - ETA: 11s - loss: 0.2277 - accuracy: 0.9138
 6700/25000 [=======>......................] - ETA: 11s - loss: 0.2264 - accuracy: 0.9143
 6800/25000 [=======>......................] - ETA: 11s - loss: 0.2260 - accuracy: 0.9147
 6900/25000 [=======>......................] - ETA: 11s - loss: 0.2258 - accuracy: 0.9139
 7000/25000 [=======>......................] - ETA: 11s - loss: 0.2252 - accuracy: 0.9141
 7100/25000 [=======>......................] - ETA: 11s - loss: 0.2252 - accuracy: 0.9139
 7200/25000 [=======>......................] - ETA: 11s - loss: 0.2265 - accuracy: 0.9136
 7300/25000 [=======>......................] - ETA: 10s - loss: 0.2258 - accuracy: 0.9138
 7400/25000 [=======>......................] - ETA: 10s - loss: 0.2259 - accuracy: 0.9139
 7500/25000 [========>.....................] - ETA: 10s - loss: 0.2261 - accuracy: 0.9139
 7600/25000 [========>.....................] - ETA: 10s - loss: 0.2260 - accuracy: 0.9139
 7700/25000 [========>.....................] - ETA: 10s - loss: 0.2267 - accuracy: 0.9138
 7800/25000 [========>.....................] - ETA: 10s - loss: 0.2255 - accuracy: 0.9144
 7900/25000 [========>.....................] - ETA: 10s - loss: 0.2256 - accuracy: 0.9144
 8000/25000 [========>.....................] - ETA: 10s - loss: 0.2258 - accuracy: 0.9145
 8100/25000 [========>.....................] - ETA: 10s - loss: 0.2260 - accuracy: 0.9144
 8200/25000 [========>.....................] - ETA: 10s - loss: 0.2266 - accuracy: 0.9141
 8300/25000 [========>.....................] - ETA: 10s - loss: 0.2270 - accuracy: 0.9139
 8400/25000 [=========>....................] - ETA: 10s - loss: 0.2266 - accuracy: 0.9137
 8500/25000 [=========>....................] - ETA: 10s - loss: 0.2262 - accuracy: 0.9136
 8600/25000 [=========>....................] - ETA: 10s - loss: 0.2265 - accuracy: 0.9134
 8700/25000 [=========>....................] - ETA: 10s - loss: 0.2272 - accuracy: 0.9132
 8800/25000 [=========>....................] - ETA: 10s - loss: 0.2265 - accuracy: 0.9134
 8900/25000 [=========>....................] - ETA: 9s - loss: 0.2264 - accuracy: 0.9135 
 9000/25000 [=========>....................] - ETA: 9s - loss: 0.2260 - accuracy: 0.9133
 9100/25000 [=========>....................] - ETA: 9s - loss: 0.2268 - accuracy: 0.9130
 9200/25000 [==========>...................] - ETA: 9s - loss: 0.2270 - accuracy: 0.9132
 9300/25000 [==========>...................] - ETA: 9s - loss: 0.2262 - accuracy: 0.9138
 9400/25000 [==========>...................] - ETA: 9s - loss: 0.2262 - accuracy: 0.9137
 9500/25000 [==========>...................] - ETA: 9s - loss: 0.2259 - accuracy: 0.9139
 9600/25000 [==========>...................] - ETA: 9s - loss: 0.2245 - accuracy: 0.9145
 9700/25000 [==========>...................] - ETA: 9s - loss: 0.2245 - accuracy: 0.9144
 9800/25000 [==========>...................] - ETA: 9s - loss: 0.2237 - accuracy: 0.9146
 9900/25000 [==========>...................] - ETA: 9s - loss: 0.2236 - accuracy: 0.9146
10000/25000 [===========>..................] - ETA: 9s - loss: 0.2237 - accuracy: 0.9145
10100/25000 [===========>..................] - ETA: 9s - loss: 0.2240 - accuracy: 0.9144
10200/25000 [===========>..................] - ETA: 9s - loss: 0.2243 - accuracy: 0.9141
10300/25000 [===========>..................] - ETA: 9s - loss: 0.2256 - accuracy: 0.9138
10400/25000 [===========>..................] - ETA: 9s - loss: 0.2254 - accuracy: 0.9137
10500/25000 [===========>..................] - ETA: 9s - loss: 0.2254 - accuracy: 0.9137
10600/25000 [===========>..................] - ETA: 8s - loss: 0.2257 - accuracy: 0.9137
10700/25000 [===========>..................] - ETA: 8s - loss: 0.2256 - accuracy: 0.9139
10800/25000 [===========>..................] - ETA: 8s - loss: 0.2252 - accuracy: 0.9143
10900/25000 [============>.................] - ETA: 8s - loss: 0.2259 - accuracy: 0.9140
11000/25000 [============>.................] - ETA: 8s - loss: 0.2262 - accuracy: 0.9137
11100/25000 [============>.................] - ETA: 8s - loss: 0.2260 - accuracy: 0.9137
11200/25000 [============>.................] - ETA: 8s - loss: 0.2259 - accuracy: 0.9137
11300/25000 [============>.................] - ETA: 8s - loss: 0.2263 - accuracy: 0.9137
11400/25000 [============>.................] - ETA: 8s - loss: 0.2269 - accuracy: 0.9132
11500/25000 [============>.................] - ETA: 8s - loss: 0.2279 - accuracy: 0.9129
11600/25000 [============>.................] - ETA: 8s - loss: 0.2282 - accuracy: 0.9126
11700/25000 [=============>................] - ETA: 8s - loss: 0.2275 - accuracy: 0.9130
11800/25000 [=============>................] - ETA: 8s - loss: 0.2283 - accuracy: 0.9125
11900/25000 [=============>................] - ETA: 8s - loss: 0.2288 - accuracy: 0.9121
12000/25000 [=============>................] - ETA: 8s - loss: 0.2290 - accuracy: 0.9121
12100/25000 [=============>................] - ETA: 7s - loss: 0.2287 - accuracy: 0.9122
12200/25000 [=============>................] - ETA: 7s - loss: 0.2288 - accuracy: 0.9123
12300/25000 [=============>................] - ETA: 7s - loss: 0.2288 - accuracy: 0.9121
12400/25000 [=============>................] - ETA: 7s - loss: 0.2283 - accuracy: 0.9125
12500/25000 [==============>...............] - ETA: 7s - loss: 0.2287 - accuracy: 0.9121
12600/25000 [==============>...............] - ETA: 7s - loss: 0.2298 - accuracy: 0.9117
12700/25000 [==============>...............] - ETA: 7s - loss: 0.2294 - accuracy: 0.9120
12800/25000 [==============>...............] - ETA: 7s - loss: 0.2289 - accuracy: 0.9121
12900/25000 [==============>...............] - ETA: 7s - loss: 0.2282 - accuracy: 0.9125
13000/25000 [==============>...............] - ETA: 7s - loss: 0.2284 - accuracy: 0.9123
13100/25000 [==============>...............] - ETA: 7s - loss: 0.2291 - accuracy: 0.9121
13200/25000 [==============>...............] - ETA: 7s - loss: 0.2292 - accuracy: 0.9121
13300/25000 [==============>...............] - ETA: 7s - loss: 0.2286 - accuracy: 0.9123
13400/25000 [===============>..............] - ETA: 7s - loss: 0.2288 - accuracy: 0.9121
13500/25000 [===============>..............] - ETA: 7s - loss: 0.2289 - accuracy: 0.9120
13600/25000 [===============>..............] - ETA: 7s - loss: 0.2284 - accuracy: 0.9122
13700/25000 [===============>..............] - ETA: 7s - loss: 0.2283 - accuracy: 0.9122
13800/25000 [===============>..............] - ETA: 6s - loss: 0.2280 - accuracy: 0.9122
13900/25000 [===============>..............] - ETA: 6s - loss: 0.2285 - accuracy: 0.9120
14000/25000 [===============>..............] - ETA: 6s - loss: 0.2285 - accuracy: 0.9120
14100/25000 [===============>..............] - ETA: 6s - loss: 0.2281 - accuracy: 0.9121
14200/25000 [================>.............] - ETA: 6s - loss: 0.2281 - accuracy: 0.9120
14300/25000 [================>.............] - ETA: 6s - loss: 0.2295 - accuracy: 0.9116
14400/25000 [================>.............] - ETA: 6s - loss: 0.2296 - accuracy: 0.9115
14500/25000 [================>.............] - ETA: 6s - loss: 0.2295 - accuracy: 0.9114
14600/25000 [================>.............] - ETA: 6s - loss: 0.2291 - accuracy: 0.9116
14700/25000 [================>.............] - ETA: 6s - loss: 0.2286 - accuracy: 0.9118
14800/25000 [================>.............] - ETA: 6s - loss: 0.2283 - accuracy: 0.9120
14900/25000 [================>.............] - ETA: 6s - loss: 0.2276 - accuracy: 0.9123
15000/25000 [=================>............] - ETA: 6s - loss: 0.2277 - accuracy: 0.9123
15100/25000 [=================>............] - ETA: 6s - loss: 0.2275 - accuracy: 0.9124
15200/25000 [=================>............] - ETA: 6s - loss: 0.2281 - accuracy: 0.9123
15300/25000 [=================>............] - ETA: 6s - loss: 0.2283 - accuracy: 0.9123
15400/25000 [=================>............] - ETA: 5s - loss: 0.2279 - accuracy: 0.9124
15500/25000 [=================>............] - ETA: 5s - loss: 0.2278 - accuracy: 0.9124
15600/25000 [=================>............] - ETA: 5s - loss: 0.2272 - accuracy: 0.9126
15700/25000 [=================>............] - ETA: 5s - loss: 0.2274 - accuracy: 0.9125
15800/25000 [=================>............] - ETA: 5s - loss: 0.2269 - accuracy: 0.9128
15900/25000 [==================>...........] - ETA: 5s - loss: 0.2272 - accuracy: 0.9127
16000/25000 [==================>...........] - ETA: 5s - loss: 0.2270 - accuracy: 0.9129
16100/25000 [==================>...........] - ETA: 5s - loss: 0.2272 - accuracy: 0.9130
16200/25000 [==================>...........] - ETA: 5s - loss: 0.2273 - accuracy: 0.9131
16300/25000 [==================>...........] - ETA: 5s - loss: 0.2275 - accuracy: 0.9129
16400/25000 [==================>...........] - ETA: 5s - loss: 0.2270 - accuracy: 0.9132
16500/25000 [==================>...........] - ETA: 5s - loss: 0.2271 - accuracy: 0.9130
16600/25000 [==================>...........] - ETA: 5s - loss: 0.2270 - accuracy: 0.9130
16700/25000 [===================>..........] - ETA: 5s - loss: 0.2268 - accuracy: 0.9130
16800/25000 [===================>..........] - ETA: 5s - loss: 0.2263 - accuracy: 0.9132
16900/25000 [===================>..........] - ETA: 5s - loss: 0.2259 - accuracy: 0.9133
17000/25000 [===================>..........] - ETA: 4s - loss: 0.2262 - accuracy: 0.9133
17100/25000 [===================>..........] - ETA: 4s - loss: 0.2260 - accuracy: 0.9133
17200/25000 [===================>..........] - ETA: 4s - loss: 0.2269 - accuracy: 0.9131
17300/25000 [===================>..........] - ETA: 4s - loss: 0.2268 - accuracy: 0.9132
17400/25000 [===================>..........] - ETA: 4s - loss: 0.2266 - accuracy: 0.9133
17500/25000 [====================>.........] - ETA: 4s - loss: 0.2270 - accuracy: 0.9130
17600/25000 [====================>.........] - ETA: 4s - loss: 0.2271 - accuracy: 0.9129
17700/25000 [====================>.........] - ETA: 4s - loss: 0.2271 - accuracy: 0.9128
17800/25000 [====================>.........] - ETA: 4s - loss: 0.2277 - accuracy: 0.9122
17900/25000 [====================>.........] - ETA: 4s - loss: 0.2278 - accuracy: 0.9122
18000/25000 [====================>.........] - ETA: 4s - loss: 0.2279 - accuracy: 0.9122
18100/25000 [====================>.........] - ETA: 4s - loss: 0.2275 - accuracy: 0.9123
18200/25000 [====================>.........] - ETA: 4s - loss: 0.2277 - accuracy: 0.9121
18300/25000 [====================>.........] - ETA: 4s - loss: 0.2278 - accuracy: 0.9122
18400/25000 [=====================>........] - ETA: 4s - loss: 0.2283 - accuracy: 0.9121
18500/25000 [=====================>........] - ETA: 4s - loss: 0.2285 - accuracy: 0.9119
18600/25000 [=====================>........] - ETA: 3s - loss: 0.2293 - accuracy: 0.9114
18700/25000 [=====================>........] - ETA: 3s - loss: 0.2291 - accuracy: 0.9114
18800/25000 [=====================>........] - ETA: 3s - loss: 0.2293 - accuracy: 0.9113
18900/25000 [=====================>........] - ETA: 3s - loss: 0.2291 - accuracy: 0.9114
19000/25000 [=====================>........] - ETA: 3s - loss: 0.2289 - accuracy: 0.9115
19100/25000 [=====================>........] - ETA: 3s - loss: 0.2293 - accuracy: 0.9114
19200/25000 [======================>.......] - ETA: 3s - loss: 0.2302 - accuracy: 0.9110
19300/25000 [======================>.......] - ETA: 3s - loss: 0.2302 - accuracy: 0.9107
19400/25000 [======================>.......] - ETA: 3s - loss: 0.2306 - accuracy: 0.9104
19500/25000 [======================>.......] - ETA: 3s - loss: 0.2304 - accuracy: 0.9106
19600/25000 [======================>.......] - ETA: 3s - loss: 0.2312 - accuracy: 0.9102
19700/25000 [======================>.......] - ETA: 3s - loss: 0.2321 - accuracy: 0.9099
19800/25000 [======================>.......] - ETA: 3s - loss: 0.2322 - accuracy: 0.9097
19900/25000 [======================>.......] - ETA: 3s - loss: 0.2319 - accuracy: 0.9098
20000/25000 [=======================>......] - ETA: 3s - loss: 0.2315 - accuracy: 0.9099
20100/25000 [=======================>......] - ETA: 3s - loss: 0.2314 - accuracy: 0.9099
20200/25000 [=======================>......] - ETA: 2s - loss: 0.2317 - accuracy: 0.9099
20300/25000 [=======================>......] - ETA: 2s - loss: 0.2320 - accuracy: 0.9096
20400/25000 [=======================>......] - ETA: 2s - loss: 0.2317 - accuracy: 0.9097
20500/25000 [=======================>......] - ETA: 2s - loss: 0.2317 - accuracy: 0.9096
20600/25000 [=======================>......] - ETA: 2s - loss: 0.2322 - accuracy: 0.9096
20700/25000 [=======================>......] - ETA: 2s - loss: 0.2326 - accuracy: 0.9096
20800/25000 [=======================>......] - ETA: 2s - loss: 0.2326 - accuracy: 0.9096
20900/25000 [========================>.....] - ETA: 2s - loss: 0.2321 - accuracy: 0.9098
21000/25000 [========================>.....] - ETA: 2s - loss: 0.2324 - accuracy: 0.9099
21100/25000 [========================>.....] - ETA: 2s - loss: 0.2325 - accuracy: 0.9097
21200/25000 [========================>.....] - ETA: 2s - loss: 0.2327 - accuracy: 0.9095
21300/25000 [========================>.....] - ETA: 2s - loss: 0.2328 - accuracy: 0.9093
21400/25000 [========================>.....] - ETA: 2s - loss: 0.2328 - accuracy: 0.9093
21500/25000 [========================>.....] - ETA: 2s - loss: 0.2328 - accuracy: 0.9093
21600/25000 [========================>.....] - ETA: 2s - loss: 0.2324 - accuracy: 0.9095
21700/25000 [=========================>....] - ETA: 2s - loss: 0.2322 - accuracy: 0.9095
21800/25000 [=========================>....] - ETA: 1s - loss: 0.2328 - accuracy: 0.9092
21900/25000 [=========================>....] - ETA: 1s - loss: 0.2333 - accuracy: 0.9089
22000/25000 [=========================>....] - ETA: 1s - loss: 0.2336 - accuracy: 0.9088
22100/25000 [=========================>....] - ETA: 1s - loss: 0.2340 - accuracy: 0.9087
22200/25000 [=========================>....] - ETA: 1s - loss: 0.2339 - accuracy: 0.9086
22300/25000 [=========================>....] - ETA: 1s - loss: 0.2337 - accuracy: 0.9086
22400/25000 [=========================>....] - ETA: 1s - loss: 0.2335 - accuracy: 0.9087
22500/25000 [==========================>...] - ETA: 1s - loss: 0.2337 - accuracy: 0.9088
22600/25000 [==========================>...] - ETA: 1s - loss: 0.2337 - accuracy: 0.9088
22700/25000 [==========================>...] - ETA: 1s - loss: 0.2336 - accuracy: 0.9088
22800/25000 [==========================>...] - ETA: 1s - loss: 0.2337 - accuracy: 0.9087
22900/25000 [==========================>...] - ETA: 1s - loss: 0.2339 - accuracy: 0.9086
23000/25000 [==========================>...] - ETA: 1s - loss: 0.2340 - accuracy: 0.9085
23100/25000 [==========================>...] - ETA: 1s - loss: 0.2345 - accuracy: 0.9081
23200/25000 [==========================>...] - ETA: 1s - loss: 0.2349 - accuracy: 0.9079
23300/25000 [==========================>...] - ETA: 1s - loss: 0.2350 - accuracy: 0.9079
23400/25000 [===========================>..] - ETA: 0s - loss: 0.2350 - accuracy: 0.9079
23500/25000 [===========================>..] - ETA: 0s - loss: 0.2348 - accuracy: 0.9080
23600/25000 [===========================>..] - ETA: 0s - loss: 0.2348 - accuracy: 0.9080
23700/25000 [===========================>..] - ETA: 0s - loss: 0.2351 - accuracy: 0.9078
23800/25000 [===========================>..] - ETA: 0s - loss: 0.2353 - accuracy: 0.9077
23900/25000 [===========================>..] - ETA: 0s - loss: 0.2354 - accuracy: 0.9076
24000/25000 [===========================>..] - ETA: 0s - loss: 0.2359 - accuracy: 0.9074
24100/25000 [===========================>..] - ETA: 0s - loss: 0.2358 - accuracy: 0.9074
24200/25000 [============================>.] - ETA: 0s - loss: 0.2357 - accuracy: 0.9075
24300/25000 [============================>.] - ETA: 0s - loss: 0.2359 - accuracy: 0.9074
24400/25000 [============================>.] - ETA: 0s - loss: 0.2362 - accuracy: 0.9073
24500/25000 [============================>.] - ETA: 0s - loss: 0.2361 - accuracy: 0.9073
24600/25000 [============================>.] - ETA: 0s - loss: 0.2364 - accuracy: 0.9073
24700/25000 [============================>.] - ETA: 0s - loss: 0.2365 - accuracy: 0.9072
24800/25000 [============================>.] - ETA: 0s - loss: 0.2363 - accuracy: 0.9072
24900/25000 [============================>.] - ETA: 0s - loss: 0.2367 - accuracy: 0.9071
25000/25000 [==============================] - 20s 783us/step - loss: 0.2366 - accuracy: 0.9073 - val_loss: 0.3082 - val_accuracy: 0.8666
Epoch 3/3

  100/25000 [..............................] - ETA: 15s - loss: 0.1982 - accuracy: 0.9400
  200/25000 [..............................] - ETA: 15s - loss: 0.1621 - accuracy: 0.9600
  300/25000 [..............................] - ETA: 15s - loss: 0.1841 - accuracy: 0.9500
  400/25000 [..............................] - ETA: 15s - loss: 0.1782 - accuracy: 0.9400
  500/25000 [..............................] - ETA: 15s - loss: 0.1776 - accuracy: 0.9380
  600/25000 [..............................] - ETA: 15s - loss: 0.1837 - accuracy: 0.9350
  700/25000 [..............................] - ETA: 14s - loss: 0.1883 - accuracy: 0.9314
  800/25000 [..............................] - ETA: 14s - loss: 0.1832 - accuracy: 0.9337
  900/25000 [>.............................] - ETA: 14s - loss: 0.1817 - accuracy: 0.9356
 1000/25000 [>.............................] - ETA: 14s - loss: 0.1778 - accuracy: 0.9380
 1100/25000 [>.............................] - ETA: 14s - loss: 0.1774 - accuracy: 0.9382
 1200/25000 [>.............................] - ETA: 14s - loss: 0.1744 - accuracy: 0.9392
 1300/25000 [>.............................] - ETA: 14s - loss: 0.1786 - accuracy: 0.9377
 1400/25000 [>.............................] - ETA: 14s - loss: 0.1787 - accuracy: 0.9364
 1500/25000 [>.............................] - ETA: 14s - loss: 0.1730 - accuracy: 0.9380
 1600/25000 [>.............................] - ETA: 14s - loss: 0.1742 - accuracy: 0.9381
 1700/25000 [=>............................] - ETA: 14s - loss: 0.1770 - accuracy: 0.9365
 1800/25000 [=>............................] - ETA: 14s - loss: 0.1773 - accuracy: 0.9344
 1900/25000 [=>............................] - ETA: 14s - loss: 0.1763 - accuracy: 0.9358
 2000/25000 [=>............................] - ETA: 14s - loss: 0.1742 - accuracy: 0.9365
 2100/25000 [=>............................] - ETA: 14s - loss: 0.1723 - accuracy: 0.9376
 2200/25000 [=>............................] - ETA: 14s - loss: 0.1712 - accuracy: 0.9386
 2300/25000 [=>............................] - ETA: 14s - loss: 0.1679 - accuracy: 0.9396
 2400/25000 [=>............................] - ETA: 13s - loss: 0.1684 - accuracy: 0.9388
 2500/25000 [==>...........................] - ETA: 13s - loss: 0.1661 - accuracy: 0.9400
 2600/25000 [==>...........................] - ETA: 13s - loss: 0.1639 - accuracy: 0.9412
 2700/25000 [==>...........................] - ETA: 13s - loss: 0.1625 - accuracy: 0.9422
 2800/25000 [==>...........................] - ETA: 13s - loss: 0.1587 - accuracy: 0.9439
 2900/25000 [==>...........................] - ETA: 13s - loss: 0.1607 - accuracy: 0.9434
 3000/25000 [==>...........................] - ETA: 13s - loss: 0.1617 - accuracy: 0.9420
 3100/25000 [==>...........................] - ETA: 13s - loss: 0.1610 - accuracy: 0.9426
 3200/25000 [==>...........................] - ETA: 13s - loss: 0.1612 - accuracy: 0.9425
 3300/25000 [==>...........................] - ETA: 13s - loss: 0.1616 - accuracy: 0.9421
 3400/25000 [===>..........................] - ETA: 13s - loss: 0.1598 - accuracy: 0.9432
 3500/25000 [===>..........................] - ETA: 13s - loss: 0.1618 - accuracy: 0.9423
 3600/25000 [===>..........................] - ETA: 13s - loss: 0.1619 - accuracy: 0.9419
 3700/25000 [===>..........................] - ETA: 13s - loss: 0.1628 - accuracy: 0.9422
 3800/25000 [===>..........................] - ETA: 13s - loss: 0.1606 - accuracy: 0.9434
 3900/25000 [===>..........................] - ETA: 13s - loss: 0.1605 - accuracy: 0.9433
 4000/25000 [===>..........................] - ETA: 12s - loss: 0.1605 - accuracy: 0.9435
 4100/25000 [===>..........................] - ETA: 12s - loss: 0.1593 - accuracy: 0.9439
 4200/25000 [====>.........................] - ETA: 12s - loss: 0.1612 - accuracy: 0.9436
 4300/25000 [====>.........................] - ETA: 12s - loss: 0.1616 - accuracy: 0.9437
 4400/25000 [====>.........................] - ETA: 12s - loss: 0.1629 - accuracy: 0.9432
 4500/25000 [====>.........................] - ETA: 12s - loss: 0.1622 - accuracy: 0.9429
 4600/25000 [====>.........................] - ETA: 12s - loss: 0.1630 - accuracy: 0.9420
 4700/25000 [====>.........................] - ETA: 12s - loss: 0.1634 - accuracy: 0.9417
 4800/25000 [====>.........................] - ETA: 12s - loss: 0.1623 - accuracy: 0.9417
 4900/25000 [====>.........................] - ETA: 12s - loss: 0.1629 - accuracy: 0.9406
 5000/25000 [=====>........................] - ETA: 12s - loss: 0.1632 - accuracy: 0.9398
 5100/25000 [=====>........................] - ETA: 12s - loss: 0.1625 - accuracy: 0.9404
 5200/25000 [=====>........................] - ETA: 12s - loss: 0.1612 - accuracy: 0.9413
 5300/25000 [=====>........................] - ETA: 12s - loss: 0.1629 - accuracy: 0.9406
 5400/25000 [=====>........................] - ETA: 12s - loss: 0.1638 - accuracy: 0.9394
 5500/25000 [=====>........................] - ETA: 12s - loss: 0.1630 - accuracy: 0.9398
 5600/25000 [=====>........................] - ETA: 11s - loss: 0.1636 - accuracy: 0.9396
 5700/25000 [=====>........................] - ETA: 11s - loss: 0.1641 - accuracy: 0.9396
 5800/25000 [=====>........................] - ETA: 11s - loss: 0.1650 - accuracy: 0.9398
 5900/25000 [======>.......................] - ETA: 11s - loss: 0.1636 - accuracy: 0.9405
 6000/25000 [======>.......................] - ETA: 11s - loss: 0.1634 - accuracy: 0.9408
 6100/25000 [======>.......................] - ETA: 11s - loss: 0.1634 - accuracy: 0.9405
 6200/25000 [======>.......................] - ETA: 11s - loss: 0.1636 - accuracy: 0.9403
 6300/25000 [======>.......................] - ETA: 11s - loss: 0.1634 - accuracy: 0.9405
 6400/25000 [======>.......................] - ETA: 11s - loss: 0.1632 - accuracy: 0.9405
 6500/25000 [======>.......................] - ETA: 11s - loss: 0.1619 - accuracy: 0.9409
 6600/25000 [======>.......................] - ETA: 11s - loss: 0.1618 - accuracy: 0.9409
 6700/25000 [=======>......................] - ETA: 11s - loss: 0.1628 - accuracy: 0.9403
 6800/25000 [=======>......................] - ETA: 11s - loss: 0.1639 - accuracy: 0.9403
 6900/25000 [=======>......................] - ETA: 11s - loss: 0.1635 - accuracy: 0.9404
 7000/25000 [=======>......................] - ETA: 11s - loss: 0.1652 - accuracy: 0.9397
 7100/25000 [=======>......................] - ETA: 11s - loss: 0.1650 - accuracy: 0.9397
 7200/25000 [=======>......................] - ETA: 10s - loss: 0.1647 - accuracy: 0.9399
 7300/25000 [=======>......................] - ETA: 10s - loss: 0.1653 - accuracy: 0.9397
 7400/25000 [=======>......................] - ETA: 10s - loss: 0.1648 - accuracy: 0.9396
 7500/25000 [========>.....................] - ETA: 10s - loss: 0.1643 - accuracy: 0.9399
 7600/25000 [========>.....................] - ETA: 10s - loss: 0.1639 - accuracy: 0.9400
 7700/25000 [========>.....................] - ETA: 10s - loss: 0.1642 - accuracy: 0.9397
 7800/25000 [========>.....................] - ETA: 10s - loss: 0.1642 - accuracy: 0.9399
 7900/25000 [========>.....................] - ETA: 10s - loss: 0.1649 - accuracy: 0.9396
 8000/25000 [========>.....................] - ETA: 10s - loss: 0.1643 - accuracy: 0.9395
 8100/25000 [========>.....................] - ETA: 10s - loss: 0.1644 - accuracy: 0.9396
 8200/25000 [========>.....................] - ETA: 10s - loss: 0.1654 - accuracy: 0.9391
 8300/25000 [========>.....................] - ETA: 10s - loss: 0.1662 - accuracy: 0.9388
 8400/25000 [=========>....................] - ETA: 10s - loss: 0.1671 - accuracy: 0.9383
 8500/25000 [=========>....................] - ETA: 10s - loss: 0.1666 - accuracy: 0.9387
 8600/25000 [=========>....................] - ETA: 10s - loss: 0.1662 - accuracy: 0.9387
 8700/25000 [=========>....................] - ETA: 10s - loss: 0.1665 - accuracy: 0.9386
 8800/25000 [=========>....................] - ETA: 9s - loss: 0.1668 - accuracy: 0.9384 
 8900/25000 [=========>....................] - ETA: 9s - loss: 0.1668 - accuracy: 0.9387
 9000/25000 [=========>....................] - ETA: 9s - loss: 0.1667 - accuracy: 0.9387
 9100/25000 [=========>....................] - ETA: 9s - loss: 0.1666 - accuracy: 0.9385
 9200/25000 [==========>...................] - ETA: 9s - loss: 0.1667 - accuracy: 0.9386
 9300/25000 [==========>...................] - ETA: 9s - loss: 0.1667 - accuracy: 0.9387
 9400/25000 [==========>...................] - ETA: 9s - loss: 0.1671 - accuracy: 0.9386
 9500/25000 [==========>...................] - ETA: 9s - loss: 0.1673 - accuracy: 0.9385
 9600/25000 [==========>...................] - ETA: 9s - loss: 0.1671 - accuracy: 0.9385
 9700/25000 [==========>...................] - ETA: 9s - loss: 0.1664 - accuracy: 0.9388
 9800/25000 [==========>...................] - ETA: 9s - loss: 0.1663 - accuracy: 0.9390
 9900/25000 [==========>...................] - ETA: 9s - loss: 0.1673 - accuracy: 0.9386
10000/25000 [===========>..................] - ETA: 9s - loss: 0.1676 - accuracy: 0.9386
10100/25000 [===========>..................] - ETA: 9s - loss: 0.1672 - accuracy: 0.9388
10200/25000 [===========>..................] - ETA: 9s - loss: 0.1671 - accuracy: 0.9387
10300/25000 [===========>..................] - ETA: 9s - loss: 0.1675 - accuracy: 0.9383
10400/25000 [===========>..................] - ETA: 9s - loss: 0.1675 - accuracy: 0.9384
10500/25000 [===========>..................] - ETA: 8s - loss: 0.1678 - accuracy: 0.9382
10600/25000 [===========>..................] - ETA: 8s - loss: 0.1679 - accuracy: 0.9384
10700/25000 [===========>..................] - ETA: 8s - loss: 0.1685 - accuracy: 0.9379
10800/25000 [===========>..................] - ETA: 8s - loss: 0.1681 - accuracy: 0.9381
10900/25000 [============>.................] - ETA: 8s - loss: 0.1679 - accuracy: 0.9381
11000/25000 [============>.................] - ETA: 8s - loss: 0.1681 - accuracy: 0.9378
11100/25000 [============>.................] - ETA: 8s - loss: 0.1681 - accuracy: 0.9377
11200/25000 [============>.................] - ETA: 8s - loss: 0.1683 - accuracy: 0.9377
11300/25000 [============>.................] - ETA: 8s - loss: 0.1687 - accuracy: 0.9373
11400/25000 [============>.................] - ETA: 8s - loss: 0.1694 - accuracy: 0.9372
11500/25000 [============>.................] - ETA: 8s - loss: 0.1692 - accuracy: 0.9372
11600/25000 [============>.................] - ETA: 8s - loss: 0.1690 - accuracy: 0.9374
11700/25000 [=============>................] - ETA: 8s - loss: 0.1694 - accuracy: 0.9373
11800/25000 [=============>................] - ETA: 8s - loss: 0.1693 - accuracy: 0.9375
11900/25000 [=============>................] - ETA: 8s - loss: 0.1693 - accuracy: 0.9375
12000/25000 [=============>................] - ETA: 8s - loss: 0.1693 - accuracy: 0.9375
12100/25000 [=============>................] - ETA: 7s - loss: 0.1696 - accuracy: 0.9374
12200/25000 [=============>................] - ETA: 7s - loss: 0.1699 - accuracy: 0.9373
12300/25000 [=============>................] - ETA: 7s - loss: 0.1697 - accuracy: 0.9375
12400/25000 [=============>................] - ETA: 7s - loss: 0.1696 - accuracy: 0.9376
12500/25000 [==============>...............] - ETA: 7s - loss: 0.1697 - accuracy: 0.9375
12600/25000 [==============>...............] - ETA: 7s - loss: 0.1700 - accuracy: 0.9375
12700/25000 [==============>...............] - ETA: 7s - loss: 0.1696 - accuracy: 0.9376
12800/25000 [==============>...............] - ETA: 7s - loss: 0.1696 - accuracy: 0.9375
12900/25000 [==============>...............] - ETA: 7s - loss: 0.1705 - accuracy: 0.9371
13000/25000 [==============>...............] - ETA: 7s - loss: 0.1701 - accuracy: 0.9371
13100/25000 [==============>...............] - ETA: 7s - loss: 0.1701 - accuracy: 0.9371
13200/25000 [==============>...............] - ETA: 7s - loss: 0.1705 - accuracy: 0.9368
13300/25000 [==============>...............] - ETA: 7s - loss: 0.1704 - accuracy: 0.9369
13400/25000 [===============>..............] - ETA: 7s - loss: 0.1703 - accuracy: 0.9370
13500/25000 [===============>..............] - ETA: 7s - loss: 0.1703 - accuracy: 0.9369
13600/25000 [===============>..............] - ETA: 7s - loss: 0.1698 - accuracy: 0.9372
13700/25000 [===============>..............] - ETA: 6s - loss: 0.1696 - accuracy: 0.9372
13800/25000 [===============>..............] - ETA: 6s - loss: 0.1698 - accuracy: 0.9372
13900/25000 [===============>..............] - ETA: 6s - loss: 0.1694 - accuracy: 0.9375
14000/25000 [===============>..............] - ETA: 6s - loss: 0.1697 - accuracy: 0.9374
14100/25000 [===============>..............] - ETA: 6s - loss: 0.1696 - accuracy: 0.9374
14200/25000 [================>.............] - ETA: 6s - loss: 0.1695 - accuracy: 0.9375
14300/25000 [================>.............] - ETA: 6s - loss: 0.1701 - accuracy: 0.9370
14400/25000 [================>.............] - ETA: 6s - loss: 0.1704 - accuracy: 0.9369
14500/25000 [================>.............] - ETA: 6s - loss: 0.1705 - accuracy: 0.9369
14600/25000 [================>.............] - ETA: 6s - loss: 0.1714 - accuracy: 0.9366
14700/25000 [================>.............] - ETA: 6s - loss: 0.1711 - accuracy: 0.9367
14800/25000 [================>.............] - ETA: 6s - loss: 0.1707 - accuracy: 0.9368
14900/25000 [================>.............] - ETA: 6s - loss: 0.1701 - accuracy: 0.9370
15000/25000 [=================>............] - ETA: 6s - loss: 0.1700 - accuracy: 0.9371
15100/25000 [=================>............] - ETA: 6s - loss: 0.1696 - accuracy: 0.9374
15200/25000 [=================>............] - ETA: 6s - loss: 0.1697 - accuracy: 0.9372
15300/25000 [=================>............] - ETA: 6s - loss: 0.1701 - accuracy: 0.9371
15400/25000 [=================>............] - ETA: 5s - loss: 0.1705 - accuracy: 0.9368
15500/25000 [=================>............] - ETA: 5s - loss: 0.1705 - accuracy: 0.9365
15600/25000 [=================>............] - ETA: 5s - loss: 0.1705 - accuracy: 0.9365
15700/25000 [=================>............] - ETA: 5s - loss: 0.1707 - accuracy: 0.9364
15800/25000 [=================>............] - ETA: 5s - loss: 0.1707 - accuracy: 0.9365
15900/25000 [==================>...........] - ETA: 5s - loss: 0.1713 - accuracy: 0.9362
16000/25000 [==================>...........] - ETA: 5s - loss: 0.1708 - accuracy: 0.9364
16100/25000 [==================>...........] - ETA: 5s - loss: 0.1712 - accuracy: 0.9361
16200/25000 [==================>...........] - ETA: 5s - loss: 0.1715 - accuracy: 0.9359
16300/25000 [==================>...........] - ETA: 5s - loss: 0.1714 - accuracy: 0.9358
16400/25000 [==================>...........] - ETA: 5s - loss: 0.1712 - accuracy: 0.9359
16500/25000 [==================>...........] - ETA: 5s - loss: 0.1713 - accuracy: 0.9359
16600/25000 [==================>...........] - ETA: 5s - loss: 0.1715 - accuracy: 0.9359
16700/25000 [===================>..........] - ETA: 5s - loss: 0.1721 - accuracy: 0.9356
16800/25000 [===================>..........] - ETA: 5s - loss: 0.1725 - accuracy: 0.9355
16900/25000 [===================>..........] - ETA: 5s - loss: 0.1728 - accuracy: 0.9354
17000/25000 [===================>..........] - ETA: 4s - loss: 0.1730 - accuracy: 0.9354
17100/25000 [===================>..........] - ETA: 4s - loss: 0.1724 - accuracy: 0.9356
17200/25000 [===================>..........] - ETA: 4s - loss: 0.1721 - accuracy: 0.9358
17300/25000 [===================>..........] - ETA: 4s - loss: 0.1719 - accuracy: 0.9358
17400/25000 [===================>..........] - ETA: 4s - loss: 0.1722 - accuracy: 0.9357
17500/25000 [====================>.........] - ETA: 4s - loss: 0.1725 - accuracy: 0.9357
17600/25000 [====================>.........] - ETA: 4s - loss: 0.1729 - accuracy: 0.9354
17700/25000 [====================>.........] - ETA: 4s - loss: 0.1727 - accuracy: 0.9355
17800/25000 [====================>.........] - ETA: 4s - loss: 0.1724 - accuracy: 0.9355
17900/25000 [====================>.........] - ETA: 4s - loss: 0.1725 - accuracy: 0.9355
18000/25000 [====================>.........] - ETA: 4s - loss: 0.1723 - accuracy: 0.9357
18100/25000 [====================>.........] - ETA: 4s - loss: 0.1730 - accuracy: 0.9355
18200/25000 [====================>.........] - ETA: 4s - loss: 0.1729 - accuracy: 0.9355
18300/25000 [====================>.........] - ETA: 4s - loss: 0.1726 - accuracy: 0.9357
18400/25000 [=====================>........] - ETA: 4s - loss: 0.1731 - accuracy: 0.9356
18500/25000 [=====================>........] - ETA: 4s - loss: 0.1736 - accuracy: 0.9355
18600/25000 [=====================>........] - ETA: 3s - loss: 0.1738 - accuracy: 0.9354
18700/25000 [=====================>........] - ETA: 3s - loss: 0.1738 - accuracy: 0.9353
18800/25000 [=====================>........] - ETA: 3s - loss: 0.1738 - accuracy: 0.9353
18900/25000 [=====================>........] - ETA: 3s - loss: 0.1746 - accuracy: 0.9349
19000/25000 [=====================>........] - ETA: 3s - loss: 0.1752 - accuracy: 0.9346
19100/25000 [=====================>........] - ETA: 3s - loss: 0.1755 - accuracy: 0.9345
19200/25000 [======================>.......] - ETA: 3s - loss: 0.1754 - accuracy: 0.9345
19300/25000 [======================>.......] - ETA: 3s - loss: 0.1759 - accuracy: 0.9344
19400/25000 [======================>.......] - ETA: 3s - loss: 0.1760 - accuracy: 0.9344
19500/25000 [======================>.......] - ETA: 3s - loss: 0.1760 - accuracy: 0.9344
19600/25000 [======================>.......] - ETA: 3s - loss: 0.1760 - accuracy: 0.9345
19700/25000 [======================>.......] - ETA: 3s - loss: 0.1758 - accuracy: 0.9347
19800/25000 [======================>.......] - ETA: 3s - loss: 0.1758 - accuracy: 0.9345
19900/25000 [======================>.......] - ETA: 3s - loss: 0.1759 - accuracy: 0.9344
20000/25000 [=======================>......] - ETA: 3s - loss: 0.1759 - accuracy: 0.9343
20100/25000 [=======================>......] - ETA: 3s - loss: 0.1759 - accuracy: 0.9343
20200/25000 [=======================>......] - ETA: 2s - loss: 0.1758 - accuracy: 0.9342
20300/25000 [=======================>......] - ETA: 2s - loss: 0.1758 - accuracy: 0.9342
20400/25000 [=======================>......] - ETA: 2s - loss: 0.1757 - accuracy: 0.9341
20500/25000 [=======================>......] - ETA: 2s - loss: 0.1756 - accuracy: 0.9341
20600/25000 [=======================>......] - ETA: 2s - loss: 0.1758 - accuracy: 0.9341
20700/25000 [=======================>......] - ETA: 2s - loss: 0.1758 - accuracy: 0.9341
20800/25000 [=======================>......] - ETA: 2s - loss: 0.1764 - accuracy: 0.9337
20900/25000 [========================>.....] - ETA: 2s - loss: 0.1764 - accuracy: 0.9336
21000/25000 [========================>.....] - ETA: 2s - loss: 0.1763 - accuracy: 0.9338
21100/25000 [========================>.....] - ETA: 2s - loss: 0.1762 - accuracy: 0.9338
21200/25000 [========================>.....] - ETA: 2s - loss: 0.1766 - accuracy: 0.9336
21300/25000 [========================>.....] - ETA: 2s - loss: 0.1762 - accuracy: 0.9338
21400/25000 [========================>.....] - ETA: 2s - loss: 0.1761 - accuracy: 0.9338
21500/25000 [========================>.....] - ETA: 2s - loss: 0.1759 - accuracy: 0.9339
21600/25000 [========================>.....] - ETA: 2s - loss: 0.1761 - accuracy: 0.9338
21700/25000 [=========================>....] - ETA: 2s - loss: 0.1759 - accuracy: 0.9339
21800/25000 [=========================>....] - ETA: 1s - loss: 0.1759 - accuracy: 0.9339
21900/25000 [=========================>....] - ETA: 1s - loss: 0.1760 - accuracy: 0.9338
22000/25000 [=========================>....] - ETA: 1s - loss: 0.1760 - accuracy: 0.9339
22100/25000 [=========================>....] - ETA: 1s - loss: 0.1757 - accuracy: 0.9341
22200/25000 [=========================>....] - ETA: 1s - loss: 0.1759 - accuracy: 0.9340
22300/25000 [=========================>....] - ETA: 1s - loss: 0.1766 - accuracy: 0.9336
22400/25000 [=========================>....] - ETA: 1s - loss: 0.1765 - accuracy: 0.9336
22500/25000 [==========================>...] - ETA: 1s - loss: 0.1771 - accuracy: 0.9335
22600/25000 [==========================>...] - ETA: 1s - loss: 0.1771 - accuracy: 0.9335
22700/25000 [==========================>...] - ETA: 1s - loss: 0.1776 - accuracy: 0.9335
22800/25000 [==========================>...] - ETA: 1s - loss: 0.1777 - accuracy: 0.9335
22900/25000 [==========================>...] - ETA: 1s - loss: 0.1776 - accuracy: 0.9334
23000/25000 [==========================>...] - ETA: 1s - loss: 0.1784 - accuracy: 0.9333
23100/25000 [==========================>...] - ETA: 1s - loss: 0.1782 - accuracy: 0.9333
23200/25000 [==========================>...] - ETA: 1s - loss: 0.1781 - accuracy: 0.9334
23300/25000 [==========================>...] - ETA: 1s - loss: 0.1783 - accuracy: 0.9333
23400/25000 [===========================>..] - ETA: 0s - loss: 0.1785 - accuracy: 0.9332
23500/25000 [===========================>..] - ETA: 0s - loss: 0.1785 - accuracy: 0.9333
23600/25000 [===========================>..] - ETA: 0s - loss: 0.1784 - accuracy: 0.9333
23700/25000 [===========================>..] - ETA: 0s - loss: 0.1787 - accuracy: 0.9332
23800/25000 [===========================>..] - ETA: 0s - loss: 0.1786 - accuracy: 0.9332
23900/25000 [===========================>..] - ETA: 0s - loss: 0.1792 - accuracy: 0.9330
24000/25000 [===========================>..] - ETA: 0s - loss: 0.1792 - accuracy: 0.9330
24100/25000 [===========================>..] - ETA: 0s - loss: 0.1790 - accuracy: 0.9330
24200/25000 [============================>.] - ETA: 0s - loss: 0.1794 - accuracy: 0.9330
24300/25000 [============================>.] - ETA: 0s - loss: 0.1796 - accuracy: 0.9328
24400/25000 [============================>.] - ETA: 0s - loss: 0.1797 - accuracy: 0.9330
24500/25000 [============================>.] - ETA: 0s - loss: 0.1799 - accuracy: 0.9329
24600/25000 [============================>.] - ETA: 0s - loss: 0.1803 - accuracy: 0.9328
24700/25000 [============================>.] - ETA: 0s - loss: 0.1807 - accuracy: 0.9326
24800/25000 [============================>.] - ETA: 0s - loss: 0.1806 - accuracy: 0.9326
24900/25000 [============================>.] - ETA: 0s - loss: 0.1810 - accuracy: 0.9324
25000/25000 [==============================] - 20s 782us/step - loss: 0.1813 - accuracy: 0.9323 - val_loss: 0.3343 - val_accuracy: 0.8625
	=====> Test the model: model.predict()
	Dataset: IMDB_REVIEWS
	Algorithm: Deep Learning using Keras 2 (KERAS_DL2)
	Training loss: 0.1185
	Training accuracy score: 96.68%
	Test loss: 0.3343
	Test accuracy score: 86.25%
	Training time: 60.0973
	Test time: 6.2492




FINAL CLASSIFICATION TABLE:

| ID | Dataset | Algorithm | Loss | Training accuracy score (%) | Test accuracy score (%) | Training time (seconds) | Test time (seconds) |
| --- | ------- | --------- | ---- | --------------------------- | ----------------------- | ----------------------- | ------------------- |
| 1 | TWENTY_NEWS_GROUPS | Deep Learning using Keras 1 (KERAS_DL1) | 0.0179 | 99.70 | 96.69 | 94.4457 | 3.5685 |
| 2 | IMDB_REVIEWS | Deep Learning using Keras 1 (KERAS_DL1) | 0.1362 | 96.15 | 88.57 | 17.2855 | 3.3037 |
| 3 | TWENTY_NEWS_GROUPS | Deep Learning using Keras 2 (KERAS_DL2) | 0.0822 | 96.96 | 95.66 | 127.3021 | 1.9339 |
| 4 | IMDB_REVIEWS | Deep Learning using Keras 2 (KERAS_DL2) | 0.1185 | 96.68 | 86.25 | 60.0973 | 6.2492 |


DONE!
Program finished. It took 428.02609753608704 seconds

Process finished with exit code 0
```