## Two Deep Learning approaches using Keras: Loss, Training Accuracy, Test Accuracy, Training Time, Test Time

| ID | Dataset | Algorithm | Loss | Training accuracy score (%) | Test accuracy score (%) | Training time (seconds) | Test time (seconds) |
| --- | ------- | --------- | ---- | --------------------------- | ----------------------- | ----------------------- | ------------------- |
| 1 | TWENTY_NEWS_GROUPS | Deep Learning using Keras 1 (KERAS_DL1) | 0.0158 | 99.73 | 96.63 | 113.0123 | 3.1087 |
| 2 | IMDB_REVIEWS | Deep Learning using Keras 1 (KERAS_DL1) | 0.1381 | 96.22 | 88.36 | 17.0448 | 3.3398 |
| 3 | TWENTY_NEWS_GROUPS | Deep Learning using Keras 2 (KERAS_DL2) | 0.0595 | 97.89 | 96.08 | 133.2070 | 1.9305 |
| 4 | IMDB_REVIEWS | Deep Learning using Keras 2 (KERAS_DL2) | 0.1437 | 95.68 | 86.33 | 61.3531 | 6.3131 |

### Deep Learning using Keras 1 (KERAS_DL1)

![TWENTY_NEWS_GROUPS: Loss, Training Accuracy, Test Accuracy, Training Time, Test Time](https://github.com/ramonfigueiredopessoa/comp551-2020-p2_classification_of_textual_data/blob/master/code/results/deep_learning_using_keras/results_20newsgroups_and_imdb_using_binary_classification/20_epochs/KERAS_DL1_TWENTY_NEWS_GROUPS_training_and_validation_accuracy_and_Loss.png)

![IMDB_REVIEWS: Loss, Training Accuracy, Test Accuracy, Training Time, Test Time](https://github.com/ramonfigueiredopessoa/comp551-2020-p2_classification_of_textual_data/blob/master/code/results/deep_learning_using_keras/results_20newsgroups_and_imdb_using_binary_classification/20_epochs/KERAS_DL1_IMDB_REVIEWS_training_and_validation_accuracy_and_Loss.png)


### Deep Learning using Keras 2 (KERAS_DL2)

![TWENTY_NEWS_GROUPS: Loss, Training Accuracy, Test Accuracy, Training Time, Test Time](https://github.com/ramonfigueiredopessoa/comp551-2020-p2_classification_of_textual_data/blob/master/code/results/deep_learning_using_keras/results_20newsgroups_and_imdb_using_binary_classification/20_epochs/KERAS_DL2_TWENTY_NEWS_GROUPS_training_and_validation_accuracy_and_Loss.png)

![IMDB_REVIEWS: Loss, Training Accuracy, Test Accuracy, Training Time, Test Time](https://github.com/ramonfigueiredopessoa/comp551-2020-p2_classification_of_textual_data/blob/master/code/results/deep_learning_using_keras/results_20newsgroups_and_imdb_using_binary_classification/20_epochs/KERAS_DL2_IMDB_REVIEWS_training_and_validation_accuracy_and_Loss.png)


#### Computer settings used to run

* Operating system: Ubuntu 18.04.4 LTS (64-bit)
* Processor: Intel® Core™ i7-7700 CPU @ 3.60GHz × 8 
* Memory: 32 GB

#### All logs 

```
python /comp551-2020-p2_classification_of_textual_data/code/main.py -dl
Using TensorFlow backend.
2020-03-12 14:23:54.895909: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libnvinfer.so.6'; dlerror: libnvinfer.so.6: cannot open shared object file: No such file or directory
2020-03-12 14:23:54.895979: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libnvinfer_plugin.so.6'; dlerror: libnvinfer_plugin.so.6: cannot open shared object file: No such file or directory
2020-03-12 14:23:54.895984: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:30] Cannot dlopen some TensorRT libraries. If you would like to use Nvidia GPU with TensorRT, please make sure the missing libraries mentioned above are installed properly.
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
03/12/2020 02:23:55 PM - INFO - Program started...
03/12/2020 02:23:55 PM - INFO - Program started...
data loaded
11314 documents - 13.782MB (training set)
7532 documents - 8.262MB (test set)
20 categories

Extracting features from the training data using a vectorizer
done in 1.163751s at 11.843MB/s
n_samples: 11314, n_features: 101321

Extracting features from the test data using the same vectorizer
done in 0.624041s at 13.239MB/s
n_samples: 7532, n_features: 101321

2020-03-12 14:23:58.784432: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
2020-03-12 14:23:58.798542: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-03-12 14:23:58.799123: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1555] Found device 0 with properties: 
pciBusID: 0000:01:00.0 name: GeForce GTX 1080 computeCapability: 6.1
coreClock: 1.7335GHz coreCount: 20 deviceMemorySize: 7.93GiB deviceMemoryBandwidth: 298.32GiB/s
2020-03-12 14:23:58.799184: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcudart.so.10.1'; dlerror: libcudart.so.10.1: cannot open shared object file: No such file or directory
2020-03-12 14:23:58.799225: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcublas.so.10'; dlerror: libcublas.so.10: cannot open shared object file: No such file or directory
2020-03-12 14:23:58.799264: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcufft.so.10'; dlerror: libcufft.so.10: cannot open shared object file: No such file or directory
2020-03-12 14:23:58.799303: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcurand.so.10'; dlerror: libcurand.so.10: cannot open shared object file: No such file or directory
2020-03-12 14:23:58.799341: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcusolver.so.10'; dlerror: libcusolver.so.10: cannot open shared object file: No such file or directory
2020-03-12 14:23:58.799380: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcusparse.so.10'; dlerror: libcusparse.so.10: cannot open shared object file: No such file or directory
2020-03-12 14:23:58.801336: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2020-03-12 14:23:58.801346: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1592] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
Skipping registering GPU devices...
2020-03-12 14:23:58.801535: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-03-12 14:23:58.824074: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 3600000000 Hz
2020-03-12 14:23:58.824506: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x9153890 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-03-12 14:23:58.824516: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2020-03-12 14:23:58.891941: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-03-12 14:23:58.892575: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x9c44760 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2020-03-12 14:23:58.892587: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): GeForce GTX 1080, Compute Capability 6.1
2020-03-12 14:23:58.892683: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1096] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-03-12 14:23:58.892688: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1102]      
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


NUMBER OF EPOCHS USED: 12

	Dataset: TWENTY_NEWS_GROUPS
	Algorithm: Deep Learning using Keras 1 (KERAS_DL1)
	Training loss: 0.0158
	Training accuracy score: 99.73%
	Test loss: 0.1312
	Test accuracy score: 96.63%
	Training time: 113.0123
	Test time: 3.1087


Loading IMDB_REVIEWS dataset:

===> Reading files from /home/ets-crchum/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/train/neg

===> Reading files from /home/ets-crchum/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/train/pos

===> Reading files from /home/ets-crchum/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/test/neg

===> Reading files from /home/ets-crchum/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/test/pos
data loaded
25000 documents - 33.133MB (training set)
25000 documents - 32.351MB (test set)

Extracting features from the training data using a vectorizer
done in 2.946164s at 11.246MB/s
n_samples: 25000, n_features: 74170

Extracting features from the test data using the same vectorizer
done in 2.871963s at 11.264MB/s
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
	Training loss: 0.1381
	Training accuracy score: 96.22%
	Test loss: 0.2823
	Test accuracy score: 88.36%
	Training time: 17.0448
	Test time: 3.3398


Loading TWENTY_NEWS_GROUPS dataset for categories:
03/12/2020 02:26:31 PM - INFO - Program started...
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
	It took 10.314192295074463 seconds


Applying NLTK feature extraction: X_test
	==> Replace unused UNICODE characters. 
	==> Text in lower case. 
	==> Apply Lemmatize using WordNet's built-in morphy function. Returns the input word unchanged if it cannot be found in WordNet 
	==> Remove English stop words
	Done: NLTK feature extraction (replace unused UNICODE characters, text in lower case, apply Lemmatize using WordNet, remove English stop words)  finished!
	It took 5.991779088973999 seconds

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

  100/11314 [..............................] - ETA: 59s - loss: 0.6901 - accuracy: 0.5737
  200/11314 [..............................] - ETA: 33s - loss: 0.6881 - accuracy: 0.6408
  300/11314 [..............................] - ETA: 24s - loss: 0.6862 - accuracy: 0.6872
  400/11314 [>.............................] - ETA: 19s - loss: 0.6841 - accuracy: 0.7161
  500/11314 [>.............................] - ETA: 17s - loss: 0.6819 - accuracy: 0.7326
  600/11314 [>.............................] - ETA: 15s - loss: 0.6793 - accuracy: 0.7449
  700/11314 [>.............................] - ETA: 13s - loss: 0.6768 - accuracy: 0.7545
  800/11314 [=>............................] - ETA: 12s - loss: 0.6740 - accuracy: 0.7624
  900/11314 [=>............................] - ETA: 12s - loss: 0.6708 - accuracy: 0.7680
 1000/11314 [=>............................] - ETA: 11s - loss: 0.6677 - accuracy: 0.7738
 1100/11314 [=>............................] - ETA: 10s - loss: 0.6643 - accuracy: 0.7780
 1200/11314 [==>...........................] - ETA: 10s - loss: 0.6604 - accuracy: 0.7820
 1300/11314 [==>...........................] - ETA: 10s - loss: 0.6566 - accuracy: 0.7855
 1400/11314 [==>...........................] - ETA: 9s - loss: 0.6518 - accuracy: 0.7883 
 1500/11314 [==>...........................] - ETA: 9s - loss: 0.6471 - accuracy: 0.7906
 1600/11314 [===>..........................] - ETA: 9s - loss: 0.6417 - accuracy: 0.7935
 1700/11314 [===>..........................] - ETA: 8s - loss: 0.6365 - accuracy: 0.7964
 1800/11314 [===>..........................] - ETA: 8s - loss: 0.6315 - accuracy: 0.7995
 1900/11314 [====>.........................] - ETA: 8s - loss: 0.6262 - accuracy: 0.8023
 2000/11314 [====>.........................] - ETA: 8s - loss: 0.6205 - accuracy: 0.8048
 2100/11314 [====>.........................] - ETA: 8s - loss: 0.6148 - accuracy: 0.8067
 2200/11314 [====>.........................] - ETA: 7s - loss: 0.6091 - accuracy: 0.8088
 2300/11314 [=====>........................] - ETA: 7s - loss: 0.6038 - accuracy: 0.8105
 2400/11314 [=====>........................] - ETA: 7s - loss: 0.5977 - accuracy: 0.8121
 2500/11314 [=====>........................] - ETA: 7s - loss: 0.5918 - accuracy: 0.8139
 2600/11314 [=====>........................] - ETA: 7s - loss: 0.5857 - accuracy: 0.8156
 2700/11314 [======>.......................] - ETA: 7s - loss: 0.5795 - accuracy: 0.8184
 2800/11314 [======>.......................] - ETA: 6s - loss: 0.5734 - accuracy: 0.8210
 2900/11314 [======>.......................] - ETA: 6s - loss: 0.5670 - accuracy: 0.8236
 3000/11314 [======>.......................] - ETA: 6s - loss: 0.5609 - accuracy: 0.8259
 3100/11314 [=======>......................] - ETA: 6s - loss: 0.5547 - accuracy: 0.8281
 3200/11314 [=======>......................] - ETA: 6s - loss: 0.5487 - accuracy: 0.8301
 3300/11314 [=======>......................] - ETA: 6s - loss: 0.5428 - accuracy: 0.8319
 3400/11314 [========>.....................] - ETA: 6s - loss: 0.5367 - accuracy: 0.8339
 3500/11314 [========>.....................] - ETA: 6s - loss: 0.5308 - accuracy: 0.8358
 3600/11314 [========>.....................] - ETA: 5s - loss: 0.5250 - accuracy: 0.8376
 3700/11314 [========>.....................] - ETA: 5s - loss: 0.5191 - accuracy: 0.8396
 3800/11314 [=========>....................] - ETA: 5s - loss: 0.5134 - accuracy: 0.8413
 3900/11314 [=========>....................] - ETA: 5s - loss: 0.5076 - accuracy: 0.8429
 4000/11314 [=========>....................] - ETA: 5s - loss: 0.5020 - accuracy: 0.8445
 4100/11314 [=========>....................] - ETA: 5s - loss: 0.4964 - accuracy: 0.8463
 4200/11314 [==========>...................] - ETA: 5s - loss: 0.4913 - accuracy: 0.8479
 4300/11314 [==========>...................] - ETA: 5s - loss: 0.4861 - accuracy: 0.8500
 4400/11314 [==========>...................] - ETA: 5s - loss: 0.4811 - accuracy: 0.8520
 4500/11314 [==========>...................] - ETA: 5s - loss: 0.4764 - accuracy: 0.8537
 4600/11314 [===========>..................] - ETA: 5s - loss: 0.4718 - accuracy: 0.8556
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.4671 - accuracy: 0.8575
 4800/11314 [===========>..................] - ETA: 4s - loss: 0.4626 - accuracy: 0.8593
 4900/11314 [===========>..................] - ETA: 4s - loss: 0.4582 - accuracy: 0.8611
 5000/11314 [============>.................] - ETA: 4s - loss: 0.4539 - accuracy: 0.8627
 5100/11314 [============>.................] - ETA: 4s - loss: 0.4497 - accuracy: 0.8643
 5200/11314 [============>.................] - ETA: 4s - loss: 0.4457 - accuracy: 0.8658
 5300/11314 [=============>................] - ETA: 4s - loss: 0.4417 - accuracy: 0.8674
 5400/11314 [=============>................] - ETA: 4s - loss: 0.4377 - accuracy: 0.8688
 5500/11314 [=============>................] - ETA: 4s - loss: 0.4340 - accuracy: 0.8702
 5600/11314 [=============>................] - ETA: 4s - loss: 0.4303 - accuracy: 0.8716
 5700/11314 [==============>...............] - ETA: 4s - loss: 0.4268 - accuracy: 0.8729
 5800/11314 [==============>...............] - ETA: 4s - loss: 0.4234 - accuracy: 0.8742
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.4200 - accuracy: 0.8754
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.4166 - accuracy: 0.8765
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.4134 - accuracy: 0.8777
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.4102 - accuracy: 0.8788
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.4072 - accuracy: 0.8799
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.4042 - accuracy: 0.8809
 6500/11314 [================>.............] - ETA: 3s - loss: 0.4013 - accuracy: 0.8820
 6600/11314 [================>.............] - ETA: 3s - loss: 0.3985 - accuracy: 0.8830
 6700/11314 [================>.............] - ETA: 3s - loss: 0.3958 - accuracy: 0.8840
 6800/11314 [=================>............] - ETA: 3s - loss: 0.3933 - accuracy: 0.8849
 6900/11314 [=================>............] - ETA: 3s - loss: 0.3908 - accuracy: 0.8858
 7000/11314 [=================>............] - ETA: 3s - loss: 0.3882 - accuracy: 0.8867
 7100/11314 [=================>............] - ETA: 3s - loss: 0.3858 - accuracy: 0.8876
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.3835 - accuracy: 0.8884
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.3812 - accuracy: 0.8892
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.3790 - accuracy: 0.8900
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.3768 - accuracy: 0.8908
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.3747 - accuracy: 0.8915
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.3726 - accuracy: 0.8923
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.3706 - accuracy: 0.8930
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.3685 - accuracy: 0.8937
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.3666 - accuracy: 0.8944
 8100/11314 [====================>.........] - ETA: 2s - loss: 0.3647 - accuracy: 0.8951
 8200/11314 [====================>.........] - ETA: 2s - loss: 0.3629 - accuracy: 0.8957
 8300/11314 [=====================>........] - ETA: 2s - loss: 0.3609 - accuracy: 0.8964
 8400/11314 [=====================>........] - ETA: 2s - loss: 0.3591 - accuracy: 0.8970
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.3573 - accuracy: 0.8977
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.3556 - accuracy: 0.8983
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.3539 - accuracy: 0.8989
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.3522 - accuracy: 0.8994
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.3506 - accuracy: 0.9000
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.3491 - accuracy: 0.9005
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.3476 - accuracy: 0.9011
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.3460 - accuracy: 0.9016
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.3446 - accuracy: 0.9021
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.3431 - accuracy: 0.9026
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.3417 - accuracy: 0.9031
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.3403 - accuracy: 0.9036
 9700/11314 [========================>.....] - ETA: 1s - loss: 0.3389 - accuracy: 0.9041
 9800/11314 [========================>.....] - ETA: 1s - loss: 0.3375 - accuracy: 0.9046
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.3362 - accuracy: 0.9050
10000/11314 [=========================>....] - ETA: 0s - loss: 0.3350 - accuracy: 0.9055
10100/11314 [=========================>....] - ETA: 0s - loss: 0.3337 - accuracy: 0.9059
10200/11314 [==========================>...] - ETA: 0s - loss: 0.3324 - accuracy: 0.9063
10300/11314 [==========================>...] - ETA: 0s - loss: 0.3312 - accuracy: 0.9067
10400/11314 [==========================>...] - ETA: 0s - loss: 0.3301 - accuracy: 0.9071
10500/11314 [==========================>...] - ETA: 0s - loss: 0.3289 - accuracy: 0.9075
10600/11314 [===========================>..] - ETA: 0s - loss: 0.3278 - accuracy: 0.9079
10700/11314 [===========================>..] - ETA: 0s - loss: 0.3267 - accuracy: 0.9083
10800/11314 [===========================>..] - ETA: 0s - loss: 0.3256 - accuracy: 0.9087
10900/11314 [===========================>..] - ETA: 0s - loss: 0.3246 - accuracy: 0.9091
11000/11314 [============================>.] - ETA: 0s - loss: 0.3235 - accuracy: 0.9094
11100/11314 [============================>.] - ETA: 0s - loss: 0.3225 - accuracy: 0.9098
11200/11314 [============================>.] - ETA: 0s - loss: 0.3214 - accuracy: 0.9102
11300/11314 [============================>.] - ETA: 0s - loss: 0.3204 - accuracy: 0.9105
11314/11314 [==============================] - 9s 806us/step - loss: 0.3203 - accuracy: 0.9105 - val_loss: 0.2000 - val_accuracy: 0.9496
Epoch 2/15

  100/11314 [..............................] - ETA: 6s - loss: 0.2131 - accuracy: 0.9495
  200/11314 [..............................] - ETA: 6s - loss: 0.2070 - accuracy: 0.9500
  300/11314 [..............................] - ETA: 6s - loss: 0.2096 - accuracy: 0.9493
  400/11314 [>.............................] - ETA: 6s - loss: 0.2091 - accuracy: 0.9492
  500/11314 [>.............................] - ETA: 7s - loss: 0.2083 - accuracy: 0.9492
  600/11314 [>.............................] - ETA: 7s - loss: 0.2087 - accuracy: 0.9489
  700/11314 [>.............................] - ETA: 6s - loss: 0.2086 - accuracy: 0.9488
  800/11314 [=>............................] - ETA: 6s - loss: 0.2082 - accuracy: 0.9489
  900/11314 [=>............................] - ETA: 6s - loss: 0.2075 - accuracy: 0.9492
 1000/11314 [=>............................] - ETA: 6s - loss: 0.2081 - accuracy: 0.9491
 1100/11314 [=>............................] - ETA: 6s - loss: 0.2080 - accuracy: 0.9491
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.2076 - accuracy: 0.9492
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.2077 - accuracy: 0.9492
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.2078 - accuracy: 0.9492
 1500/11314 [==>...........................] - ETA: 6s - loss: 0.2080 - accuracy: 0.9492
 1600/11314 [===>..........................] - ETA: 6s - loss: 0.2082 - accuracy: 0.9492
 1700/11314 [===>..........................] - ETA: 6s - loss: 0.2084 - accuracy: 0.9492
 1800/11314 [===>..........................] - ETA: 6s - loss: 0.2085 - accuracy: 0.9492
 1900/11314 [====>.........................] - ETA: 6s - loss: 0.2081 - accuracy: 0.9493
 2000/11314 [====>.........................] - ETA: 6s - loss: 0.2085 - accuracy: 0.9492
 2100/11314 [====>.........................] - ETA: 5s - loss: 0.2083 - accuracy: 0.9492
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.2086 - accuracy: 0.9491
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.2091 - accuracy: 0.9490
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.2088 - accuracy: 0.9491
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.2088 - accuracy: 0.9491
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.2087 - accuracy: 0.9491
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.2086 - accuracy: 0.9491
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.2086 - accuracy: 0.9491
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.2083 - accuracy: 0.9492
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.2082 - accuracy: 0.9492
 3100/11314 [=======>......................] - ETA: 5s - loss: 0.2083 - accuracy: 0.9492
 3200/11314 [=======>......................] - ETA: 5s - loss: 0.2082 - accuracy: 0.9492
 3300/11314 [=======>......................] - ETA: 5s - loss: 0.2081 - accuracy: 0.9493
 3400/11314 [========>.....................] - ETA: 5s - loss: 0.2080 - accuracy: 0.9493
 3500/11314 [========>.....................] - ETA: 5s - loss: 0.2080 - accuracy: 0.9493
 3600/11314 [========>.....................] - ETA: 5s - loss: 0.2080 - accuracy: 0.9493
 3700/11314 [========>.....................] - ETA: 5s - loss: 0.2079 - accuracy: 0.9493
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.2077 - accuracy: 0.9493
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.2077 - accuracy: 0.9493
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.2076 - accuracy: 0.9494
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.2076 - accuracy: 0.9494
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.2076 - accuracy: 0.9494
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.2074 - accuracy: 0.9494
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.2073 - accuracy: 0.9494
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.2072 - accuracy: 0.9494
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.2073 - accuracy: 0.9494
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.2073 - accuracy: 0.9494
 4800/11314 [===========>..................] - ETA: 4s - loss: 0.2072 - accuracy: 0.9494
 4900/11314 [===========>..................] - ETA: 4s - loss: 0.2072 - accuracy: 0.9494
 5000/11314 [============>.................] - ETA: 4s - loss: 0.2071 - accuracy: 0.9494
 5100/11314 [============>.................] - ETA: 4s - loss: 0.2070 - accuracy: 0.9494
 5200/11314 [============>.................] - ETA: 4s - loss: 0.2068 - accuracy: 0.9495
 5300/11314 [=============>................] - ETA: 3s - loss: 0.2067 - accuracy: 0.9495
 5400/11314 [=============>................] - ETA: 3s - loss: 0.2066 - accuracy: 0.9495
 5500/11314 [=============>................] - ETA: 3s - loss: 0.2067 - accuracy: 0.9495
 5600/11314 [=============>................] - ETA: 3s - loss: 0.2065 - accuracy: 0.9495
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.2065 - accuracy: 0.9495
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.2066 - accuracy: 0.9495
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.2065 - accuracy: 0.9495
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.2065 - accuracy: 0.9495
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.2064 - accuracy: 0.9495
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.2064 - accuracy: 0.9495
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.2064 - accuracy: 0.9495
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.2064 - accuracy: 0.9495
 6500/11314 [================>.............] - ETA: 3s - loss: 0.2065 - accuracy: 0.9495
 6600/11314 [================>.............] - ETA: 3s - loss: 0.2065 - accuracy: 0.9495
 6700/11314 [================>.............] - ETA: 3s - loss: 0.2065 - accuracy: 0.9495
 6800/11314 [=================>............] - ETA: 2s - loss: 0.2064 - accuracy: 0.9495
 6900/11314 [=================>............] - ETA: 2s - loss: 0.2064 - accuracy: 0.9495
 7000/11314 [=================>............] - ETA: 2s - loss: 0.2063 - accuracy: 0.9495
 7100/11314 [=================>............] - ETA: 2s - loss: 0.2064 - accuracy: 0.9495
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.2064 - accuracy: 0.9495
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.2064 - accuracy: 0.9495
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.2063 - accuracy: 0.9495
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.2063 - accuracy: 0.9495
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.2063 - accuracy: 0.9495
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.2064 - accuracy: 0.9495
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.2064 - accuracy: 0.9495
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.2064 - accuracy: 0.9495
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.2063 - accuracy: 0.9495
 8100/11314 [====================>.........] - ETA: 2s - loss: 0.2063 - accuracy: 0.9495
 8200/11314 [====================>.........] - ETA: 2s - loss: 0.2062 - accuracy: 0.9495
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.2062 - accuracy: 0.9495
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.2062 - accuracy: 0.9495
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.2062 - accuracy: 0.9495
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.2062 - accuracy: 0.9495
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.2063 - accuracy: 0.9495
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.2061 - accuracy: 0.9495
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.2062 - accuracy: 0.9495
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.2063 - accuracy: 0.9495
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.2063 - accuracy: 0.9495
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.2063 - accuracy: 0.9495
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.2062 - accuracy: 0.9495
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.2063 - accuracy: 0.9494
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.2063 - accuracy: 0.9494
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.2062 - accuracy: 0.9494
 9700/11314 [========================>.....] - ETA: 1s - loss: 0.2062 - accuracy: 0.9494
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.2062 - accuracy: 0.9494
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.2061 - accuracy: 0.9494
10000/11314 [=========================>....] - ETA: 0s - loss: 0.2061 - accuracy: 0.9494
10100/11314 [=========================>....] - ETA: 0s - loss: 0.2060 - accuracy: 0.9495
10200/11314 [==========================>...] - ETA: 0s - loss: 0.2060 - accuracy: 0.9495
10300/11314 [==========================>...] - ETA: 0s - loss: 0.2060 - accuracy: 0.9495
10400/11314 [==========================>...] - ETA: 0s - loss: 0.2060 - accuracy: 0.9495
10500/11314 [==========================>...] - ETA: 0s - loss: 0.2060 - accuracy: 0.9495
10600/11314 [===========================>..] - ETA: 0s - loss: 0.2060 - accuracy: 0.9495
10700/11314 [===========================>..] - ETA: 0s - loss: 0.2060 - accuracy: 0.9495
10800/11314 [===========================>..] - ETA: 0s - loss: 0.2060 - accuracy: 0.9495
10900/11314 [===========================>..] - ETA: 0s - loss: 0.2060 - accuracy: 0.9495
11000/11314 [============================>.] - ETA: 0s - loss: 0.2060 - accuracy: 0.9495
11100/11314 [============================>.] - ETA: 0s - loss: 0.2060 - accuracy: 0.9495
11200/11314 [============================>.] - ETA: 0s - loss: 0.2060 - accuracy: 0.9494
11300/11314 [============================>.] - ETA: 0s - loss: 0.2060 - accuracy: 0.9494
11314/11314 [==============================] - 9s 761us/step - loss: 0.2060 - accuracy: 0.9494 - val_loss: 0.1990 - val_accuracy: 0.9496
Epoch 3/15

  100/11314 [..............................] - ETA: 6s - loss: 0.2049 - accuracy: 0.9500
  200/11314 [..............................] - ETA: 6s - loss: 0.2074 - accuracy: 0.9492
  300/11314 [..............................] - ETA: 6s - loss: 0.2071 - accuracy: 0.9491
  400/11314 [>.............................] - ETA: 6s - loss: 0.2059 - accuracy: 0.9493
  500/11314 [>.............................] - ETA: 6s - loss: 0.2064 - accuracy: 0.9491
  600/11314 [>.............................] - ETA: 6s - loss: 0.2061 - accuracy: 0.9492
  700/11314 [>.............................] - ETA: 6s - loss: 0.2061 - accuracy: 0.9491
  800/11314 [=>............................] - ETA: 6s - loss: 0.2062 - accuracy: 0.9490
  900/11314 [=>............................] - ETA: 6s - loss: 0.2051 - accuracy: 0.9493
 1000/11314 [=>............................] - ETA: 6s - loss: 0.2045 - accuracy: 0.9494
 1100/11314 [=>............................] - ETA: 6s - loss: 0.2047 - accuracy: 0.9493
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.2045 - accuracy: 0.9493
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.2047 - accuracy: 0.9493
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.2046 - accuracy: 0.9494
 1500/11314 [==>...........................] - ETA: 6s - loss: 0.2048 - accuracy: 0.9493
 1600/11314 [===>..........................] - ETA: 6s - loss: 0.2045 - accuracy: 0.9494
 1700/11314 [===>..........................] - ETA: 6s - loss: 0.2045 - accuracy: 0.9494
 1800/11314 [===>..........................] - ETA: 6s - loss: 0.2047 - accuracy: 0.9493
 1900/11314 [====>.........................] - ETA: 6s - loss: 0.2049 - accuracy: 0.9493
 2000/11314 [====>.........................] - ETA: 6s - loss: 0.2048 - accuracy: 0.9494
 2100/11314 [====>.........................] - ETA: 5s - loss: 0.2050 - accuracy: 0.9493
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.2052 - accuracy: 0.9492
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.2050 - accuracy: 0.9493
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.2049 - accuracy: 0.9493
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.2049 - accuracy: 0.9493
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.2047 - accuracy: 0.9494
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.2046 - accuracy: 0.9494
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.2046 - accuracy: 0.9494
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.2048 - accuracy: 0.9494
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.2048 - accuracy: 0.9494
 3100/11314 [=======>......................] - ETA: 5s - loss: 0.2047 - accuracy: 0.9494
 3200/11314 [=======>......................] - ETA: 5s - loss: 0.2048 - accuracy: 0.9494
 3300/11314 [=======>......................] - ETA: 5s - loss: 0.2050 - accuracy: 0.9493
 3400/11314 [========>.....................] - ETA: 5s - loss: 0.2049 - accuracy: 0.9493
 3500/11314 [========>.....................] - ETA: 5s - loss: 0.2049 - accuracy: 0.9494
 3600/11314 [========>.....................] - ETA: 4s - loss: 0.2048 - accuracy: 0.9494
 3700/11314 [========>.....................] - ETA: 4s - loss: 0.2047 - accuracy: 0.9494
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.2048 - accuracy: 0.9494
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.2044 - accuracy: 0.9495
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.2042 - accuracy: 0.9495
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.2040 - accuracy: 0.9496
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.2039 - accuracy: 0.9496
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.2038 - accuracy: 0.9496
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.2037 - accuracy: 0.9496
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.2037 - accuracy: 0.9496
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.2038 - accuracy: 0.9496
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.2038 - accuracy: 0.9496
 4800/11314 [===========>..................] - ETA: 4s - loss: 0.2038 - accuracy: 0.9496
 4900/11314 [===========>..................] - ETA: 4s - loss: 0.2038 - accuracy: 0.9496
 5000/11314 [============>.................] - ETA: 4s - loss: 0.2037 - accuracy: 0.9496
 5100/11314 [============>.................] - ETA: 3s - loss: 0.2036 - accuracy: 0.9496
 5200/11314 [============>.................] - ETA: 3s - loss: 0.2037 - accuracy: 0.9496
 5300/11314 [=============>................] - ETA: 3s - loss: 0.2037 - accuracy: 0.9496
 5400/11314 [=============>................] - ETA: 3s - loss: 0.2038 - accuracy: 0.9496
 5500/11314 [=============>................] - ETA: 3s - loss: 0.2036 - accuracy: 0.9497
 5600/11314 [=============>................] - ETA: 3s - loss: 0.2036 - accuracy: 0.9497
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.2036 - accuracy: 0.9497
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.2036 - accuracy: 0.9497
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.2035 - accuracy: 0.9497
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.2036 - accuracy: 0.9497
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.2036 - accuracy: 0.9497
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.2035 - accuracy: 0.9497
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.2035 - accuracy: 0.9497
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.2035 - accuracy: 0.9497
 6500/11314 [================>.............] - ETA: 3s - loss: 0.2033 - accuracy: 0.9497
 6600/11314 [================>.............] - ETA: 3s - loss: 0.2034 - accuracy: 0.9497
 6700/11314 [================>.............] - ETA: 2s - loss: 0.2035 - accuracy: 0.9497
 6800/11314 [=================>............] - ETA: 2s - loss: 0.2034 - accuracy: 0.9497
 6900/11314 [=================>............] - ETA: 2s - loss: 0.2033 - accuracy: 0.9497
 7000/11314 [=================>............] - ETA: 2s - loss: 0.2033 - accuracy: 0.9496
 7100/11314 [=================>............] - ETA: 2s - loss: 0.2033 - accuracy: 0.9496
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.2034 - accuracy: 0.9496
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.2033 - accuracy: 0.9496
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.2033 - accuracy: 0.9496
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.2032 - accuracy: 0.9496
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.2032 - accuracy: 0.9496
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.2032 - accuracy: 0.9496
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.2031 - accuracy: 0.9496
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.2031 - accuracy: 0.9496
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.2031 - accuracy: 0.9496
 8100/11314 [====================>.........] - ETA: 2s - loss: 0.2032 - accuracy: 0.9496
 8200/11314 [====================>.........] - ETA: 2s - loss: 0.2031 - accuracy: 0.9496
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.2031 - accuracy: 0.9496
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.2031 - accuracy: 0.9496
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.2030 - accuracy: 0.9496
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.2029 - accuracy: 0.9496
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.2029 - accuracy: 0.9496
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.2030 - accuracy: 0.9496
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.2030 - accuracy: 0.9496
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.2029 - accuracy: 0.9496
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.2029 - accuracy: 0.9496
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.2029 - accuracy: 0.9496
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.2029 - accuracy: 0.9496
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.2029 - accuracy: 0.9496
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.2029 - accuracy: 0.9496
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.2029 - accuracy: 0.9496
 9700/11314 [========================>.....] - ETA: 1s - loss: 0.2029 - accuracy: 0.9496
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.2029 - accuracy: 0.9496
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.2029 - accuracy: 0.9496
10000/11314 [=========================>....] - ETA: 0s - loss: 0.2029 - accuracy: 0.9496
10100/11314 [=========================>....] - ETA: 0s - loss: 0.2029 - accuracy: 0.9496
10200/11314 [==========================>...] - ETA: 0s - loss: 0.2029 - accuracy: 0.9496
10300/11314 [==========================>...] - ETA: 0s - loss: 0.2028 - accuracy: 0.9496
10400/11314 [==========================>...] - ETA: 0s - loss: 0.2028 - accuracy: 0.9496
10500/11314 [==========================>...] - ETA: 0s - loss: 0.2027 - accuracy: 0.9496
10600/11314 [===========================>..] - ETA: 0s - loss: 0.2027 - accuracy: 0.9496
10700/11314 [===========================>..] - ETA: 0s - loss: 0.2027 - accuracy: 0.9496
10800/11314 [===========================>..] - ETA: 0s - loss: 0.2028 - accuracy: 0.9496
10900/11314 [===========================>..] - ETA: 0s - loss: 0.2028 - accuracy: 0.9496
11000/11314 [============================>.] - ETA: 0s - loss: 0.2027 - accuracy: 0.9496
11100/11314 [============================>.] - ETA: 0s - loss: 0.2027 - accuracy: 0.9495
11200/11314 [============================>.] - ETA: 0s - loss: 0.2027 - accuracy: 0.9496
11300/11314 [============================>.] - ETA: 0s - loss: 0.2027 - accuracy: 0.9496
11314/11314 [==============================] - 9s 757us/step - loss: 0.2027 - accuracy: 0.9496 - val_loss: 0.1959 - val_accuracy: 0.9496
Epoch 4/15

  100/11314 [..............................] - ETA: 6s - loss: 0.2004 - accuracy: 0.9500
  200/11314 [..............................] - ETA: 6s - loss: 0.1962 - accuracy: 0.9503
  300/11314 [..............................] - ETA: 6s - loss: 0.1967 - accuracy: 0.9504
  400/11314 [>.............................] - ETA: 6s - loss: 0.1974 - accuracy: 0.9501
  500/11314 [>.............................] - ETA: 7s - loss: 0.1984 - accuracy: 0.9497
  600/11314 [>.............................] - ETA: 6s - loss: 0.1981 - accuracy: 0.9498
  700/11314 [>.............................] - ETA: 6s - loss: 0.1980 - accuracy: 0.9499
  800/11314 [=>............................] - ETA: 6s - loss: 0.1980 - accuracy: 0.9499
  900/11314 [=>............................] - ETA: 6s - loss: 0.1982 - accuracy: 0.9498
 1000/11314 [=>............................] - ETA: 6s - loss: 0.1976 - accuracy: 0.9500
 1100/11314 [=>............................] - ETA: 6s - loss: 0.1977 - accuracy: 0.9499
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.1979 - accuracy: 0.9498
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.1978 - accuracy: 0.9499
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.1978 - accuracy: 0.9498
 1500/11314 [==>...........................] - ETA: 6s - loss: 0.1980 - accuracy: 0.9497
 1600/11314 [===>..........................] - ETA: 6s - loss: 0.1979 - accuracy: 0.9497
 1700/11314 [===>..........................] - ETA: 6s - loss: 0.1976 - accuracy: 0.9497
 1800/11314 [===>..........................] - ETA: 6s - loss: 0.1974 - accuracy: 0.9498
 1900/11314 [====>.........................] - ETA: 6s - loss: 0.1971 - accuracy: 0.9499
 2000/11314 [====>.........................] - ETA: 5s - loss: 0.1969 - accuracy: 0.9499
 2100/11314 [====>.........................] - ETA: 5s - loss: 0.1965 - accuracy: 0.9500
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.1966 - accuracy: 0.9500
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.1968 - accuracy: 0.9499
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.1968 - accuracy: 0.9498
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.1967 - accuracy: 0.9498
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.1966 - accuracy: 0.9498
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.1964 - accuracy: 0.9499
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.1963 - accuracy: 0.9499
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.1964 - accuracy: 0.9499
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.1965 - accuracy: 0.9498
 3100/11314 [=======>......................] - ETA: 5s - loss: 0.1965 - accuracy: 0.9498
 3200/11314 [=======>......................] - ETA: 5s - loss: 0.1962 - accuracy: 0.9499
 3300/11314 [=======>......................] - ETA: 5s - loss: 0.1960 - accuracy: 0.9499
 3400/11314 [========>.....................] - ETA: 5s - loss: 0.1962 - accuracy: 0.9498
 3500/11314 [========>.....................] - ETA: 4s - loss: 0.1962 - accuracy: 0.9498
 3600/11314 [========>.....................] - ETA: 4s - loss: 0.1960 - accuracy: 0.9498
 3700/11314 [========>.....................] - ETA: 4s - loss: 0.1957 - accuracy: 0.9499
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.1956 - accuracy: 0.9499
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.1955 - accuracy: 0.9499
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.1956 - accuracy: 0.9498
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.1956 - accuracy: 0.9498
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.1955 - accuracy: 0.9499
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.1955 - accuracy: 0.9499
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.1957 - accuracy: 0.9498
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.1957 - accuracy: 0.9498
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.1956 - accuracy: 0.9498
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.1956 - accuracy: 0.9498
 4800/11314 [===========>..................] - ETA: 4s - loss: 0.1957 - accuracy: 0.9498
 4900/11314 [===========>..................] - ETA: 4s - loss: 0.1957 - accuracy: 0.9497
 5000/11314 [============>.................] - ETA: 4s - loss: 0.1958 - accuracy: 0.9497
 5100/11314 [============>.................] - ETA: 3s - loss: 0.1959 - accuracy: 0.9497
 5200/11314 [============>.................] - ETA: 3s - loss: 0.1958 - accuracy: 0.9497
 5300/11314 [=============>................] - ETA: 3s - loss: 0.1958 - accuracy: 0.9497
 5400/11314 [=============>................] - ETA: 3s - loss: 0.1958 - accuracy: 0.9497
 5500/11314 [=============>................] - ETA: 3s - loss: 0.1958 - accuracy: 0.9497
 5600/11314 [=============>................] - ETA: 3s - loss: 0.1958 - accuracy: 0.9497
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.1955 - accuracy: 0.9497
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.1955 - accuracy: 0.9497
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.1956 - accuracy: 0.9497
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.1956 - accuracy: 0.9497
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.1955 - accuracy: 0.9497
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.1955 - accuracy: 0.9496
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.1955 - accuracy: 0.9496
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.1953 - accuracy: 0.9497
 6500/11314 [================>.............] - ETA: 3s - loss: 0.1953 - accuracy: 0.9496
 6600/11314 [================>.............] - ETA: 3s - loss: 0.1953 - accuracy: 0.9496
 6700/11314 [================>.............] - ETA: 2s - loss: 0.1952 - accuracy: 0.9496
 6800/11314 [=================>............] - ETA: 2s - loss: 0.1951 - accuracy: 0.9496
 6900/11314 [=================>............] - ETA: 2s - loss: 0.1950 - accuracy: 0.9496
 7000/11314 [=================>............] - ETA: 2s - loss: 0.1948 - accuracy: 0.9496
 7100/11314 [=================>............] - ETA: 2s - loss: 0.1947 - accuracy: 0.9496
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.1945 - accuracy: 0.9496
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.1945 - accuracy: 0.9496
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.1944 - accuracy: 0.9496
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.1944 - accuracy: 0.9496
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.1943 - accuracy: 0.9496
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.1941 - accuracy: 0.9496
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.1941 - accuracy: 0.9496
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.1941 - accuracy: 0.9495
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.1940 - accuracy: 0.9495
 8100/11314 [====================>.........] - ETA: 2s - loss: 0.1940 - accuracy: 0.9495
 8200/11314 [====================>.........] - ETA: 1s - loss: 0.1939 - accuracy: 0.9495
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.1938 - accuracy: 0.9495
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.1938 - accuracy: 0.9495
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.1938 - accuracy: 0.9495
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.1937 - accuracy: 0.9495
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.1936 - accuracy: 0.9495
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.1936 - accuracy: 0.9495
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.1935 - accuracy: 0.9495
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.1933 - accuracy: 0.9496
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.1932 - accuracy: 0.9496
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.1931 - accuracy: 0.9495
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.1931 - accuracy: 0.9496
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.1930 - accuracy: 0.9496
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.1930 - accuracy: 0.9495
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.1929 - accuracy: 0.9495
 9700/11314 [========================>.....] - ETA: 1s - loss: 0.1928 - accuracy: 0.9495
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.1928 - accuracy: 0.9495
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.1927 - accuracy: 0.9495
10000/11314 [=========================>....] - ETA: 0s - loss: 0.1927 - accuracy: 0.9495
10100/11314 [=========================>....] - ETA: 0s - loss: 0.1925 - accuracy: 0.9495
10200/11314 [==========================>...] - ETA: 0s - loss: 0.1925 - accuracy: 0.9495
10300/11314 [==========================>...] - ETA: 0s - loss: 0.1924 - accuracy: 0.9495
10400/11314 [==========================>...] - ETA: 0s - loss: 0.1923 - accuracy: 0.9496
10500/11314 [==========================>...] - ETA: 0s - loss: 0.1922 - accuracy: 0.9495
10600/11314 [===========================>..] - ETA: 0s - loss: 0.1921 - accuracy: 0.9495
10700/11314 [===========================>..] - ETA: 0s - loss: 0.1920 - accuracy: 0.9496
10800/11314 [===========================>..] - ETA: 0s - loss: 0.1919 - accuracy: 0.9496
10900/11314 [===========================>..] - ETA: 0s - loss: 0.1918 - accuracy: 0.9496
11000/11314 [============================>.] - ETA: 0s - loss: 0.1917 - accuracy: 0.9496
11100/11314 [============================>.] - ETA: 0s - loss: 0.1916 - accuracy: 0.9496
11200/11314 [============================>.] - ETA: 0s - loss: 0.1915 - accuracy: 0.9496
11300/11314 [============================>.] - ETA: 0s - loss: 0.1914 - accuracy: 0.9496
11314/11314 [==============================] - 9s 752us/step - loss: 0.1914 - accuracy: 0.9496 - val_loss: 0.1799 - val_accuracy: 0.9496
Epoch 5/15

  100/11314 [..............................] - ETA: 6s - loss: 0.1818 - accuracy: 0.9479
  200/11314 [..............................] - ETA: 6s - loss: 0.1816 - accuracy: 0.9484
  300/11314 [..............................] - ETA: 6s - loss: 0.1794 - accuracy: 0.9489
  400/11314 [>.............................] - ETA: 6s - loss: 0.1787 - accuracy: 0.9489
  500/11314 [>.............................] - ETA: 7s - loss: 0.1783 - accuracy: 0.9492
  600/11314 [>.............................] - ETA: 6s - loss: 0.1777 - accuracy: 0.9493
  700/11314 [>.............................] - ETA: 6s - loss: 0.1779 - accuracy: 0.9492
  800/11314 [=>............................] - ETA: 6s - loss: 0.1776 - accuracy: 0.9490
  900/11314 [=>............................] - ETA: 6s - loss: 0.1769 - accuracy: 0.9491
 1000/11314 [=>............................] - ETA: 6s - loss: 0.1760 - accuracy: 0.9491
 1100/11314 [=>............................] - ETA: 6s - loss: 0.1759 - accuracy: 0.9491
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.1760 - accuracy: 0.9490
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.1752 - accuracy: 0.9492
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.1751 - accuracy: 0.9493
 1500/11314 [==>...........................] - ETA: 6s - loss: 0.1750 - accuracy: 0.9493
 1600/11314 [===>..........................] - ETA: 6s - loss: 0.1745 - accuracy: 0.9494
 1700/11314 [===>..........................] - ETA: 6s - loss: 0.1748 - accuracy: 0.9493
 1800/11314 [===>..........................] - ETA: 6s - loss: 0.1749 - accuracy: 0.9494
 1900/11314 [====>.........................] - ETA: 6s - loss: 0.1748 - accuracy: 0.9495
 2000/11314 [====>.........................] - ETA: 5s - loss: 0.1742 - accuracy: 0.9496
 2100/11314 [====>.........................] - ETA: 5s - loss: 0.1740 - accuracy: 0.9496
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.1741 - accuracy: 0.9496
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.1742 - accuracy: 0.9496
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.1741 - accuracy: 0.9497
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.1740 - accuracy: 0.9498
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.1740 - accuracy: 0.9497
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.1740 - accuracy: 0.9496
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.1739 - accuracy: 0.9496
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.1739 - accuracy: 0.9496
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.1739 - accuracy: 0.9496
 3100/11314 [=======>......................] - ETA: 5s - loss: 0.1736 - accuracy: 0.9497
 3200/11314 [=======>......................] - ETA: 5s - loss: 0.1735 - accuracy: 0.9496
 3300/11314 [=======>......................] - ETA: 5s - loss: 0.1734 - accuracy: 0.9496
 3400/11314 [========>.....................] - ETA: 5s - loss: 0.1733 - accuracy: 0.9496
 3500/11314 [========>.....................] - ETA: 4s - loss: 0.1730 - accuracy: 0.9496
 3600/11314 [========>.....................] - ETA: 4s - loss: 0.1730 - accuracy: 0.9496
 3700/11314 [========>.....................] - ETA: 4s - loss: 0.1729 - accuracy: 0.9496
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.1727 - accuracy: 0.9496
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.1726 - accuracy: 0.9496
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.1726 - accuracy: 0.9496
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.1726 - accuracy: 0.9496
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.1726 - accuracy: 0.9496
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.1724 - accuracy: 0.9496
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.1726 - accuracy: 0.9496
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.1725 - accuracy: 0.9496
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.1725 - accuracy: 0.9496
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.1724 - accuracy: 0.9496
 4800/11314 [===========>..................] - ETA: 4s - loss: 0.1723 - accuracy: 0.9496
 4900/11314 [===========>..................] - ETA: 4s - loss: 0.1720 - accuracy: 0.9497
 5000/11314 [============>.................] - ETA: 4s - loss: 0.1719 - accuracy: 0.9497
 5100/11314 [============>.................] - ETA: 3s - loss: 0.1717 - accuracy: 0.9497
 5200/11314 [============>.................] - ETA: 3s - loss: 0.1717 - accuracy: 0.9497
 5300/11314 [=============>................] - ETA: 3s - loss: 0.1717 - accuracy: 0.9497
 5400/11314 [=============>................] - ETA: 3s - loss: 0.1715 - accuracy: 0.9497
 5500/11314 [=============>................] - ETA: 3s - loss: 0.1715 - accuracy: 0.9497
 5600/11314 [=============>................] - ETA: 3s - loss: 0.1715 - accuracy: 0.9497
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.1715 - accuracy: 0.9497
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.1714 - accuracy: 0.9497
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.1712 - accuracy: 0.9498
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.1713 - accuracy: 0.9497
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.1713 - accuracy: 0.9497
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.1713 - accuracy: 0.9497
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.1712 - accuracy: 0.9497
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.1712 - accuracy: 0.9497
 6500/11314 [================>.............] - ETA: 3s - loss: 0.1712 - accuracy: 0.9497
 6600/11314 [================>.............] - ETA: 3s - loss: 0.1710 - accuracy: 0.9497
 6700/11314 [================>.............] - ETA: 2s - loss: 0.1710 - accuracy: 0.9497
 6800/11314 [=================>............] - ETA: 2s - loss: 0.1710 - accuracy: 0.9497
 6900/11314 [=================>............] - ETA: 2s - loss: 0.1709 - accuracy: 0.9497
 7000/11314 [=================>............] - ETA: 2s - loss: 0.1708 - accuracy: 0.9497
 7100/11314 [=================>............] - ETA: 2s - loss: 0.1707 - accuracy: 0.9497
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.1706 - accuracy: 0.9497
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.1707 - accuracy: 0.9497
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.1706 - accuracy: 0.9497
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.1705 - accuracy: 0.9497
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.1704 - accuracy: 0.9497
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.1703 - accuracy: 0.9497
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.1704 - accuracy: 0.9497
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.1703 - accuracy: 0.9497
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.1703 - accuracy: 0.9497
 8100/11314 [====================>.........] - ETA: 2s - loss: 0.1703 - accuracy: 0.9497
 8200/11314 [====================>.........] - ETA: 1s - loss: 0.1703 - accuracy: 0.9497
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.1702 - accuracy: 0.9497
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.1702 - accuracy: 0.9497
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.1702 - accuracy: 0.9497
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.1700 - accuracy: 0.9497
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.1699 - accuracy: 0.9497
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.1699 - accuracy: 0.9497
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.1698 - accuracy: 0.9497
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.1698 - accuracy: 0.9497
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.1697 - accuracy: 0.9497
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.1696 - accuracy: 0.9497
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.1696 - accuracy: 0.9497
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.1695 - accuracy: 0.9497
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.1695 - accuracy: 0.9497
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.1694 - accuracy: 0.9497
 9700/11314 [========================>.....] - ETA: 1s - loss: 0.1693 - accuracy: 0.9497
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.1692 - accuracy: 0.9497
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.1692 - accuracy: 0.9497
10000/11314 [=========================>....] - ETA: 0s - loss: 0.1692 - accuracy: 0.9497
10100/11314 [=========================>....] - ETA: 0s - loss: 0.1691 - accuracy: 0.9497
10200/11314 [==========================>...] - ETA: 0s - loss: 0.1691 - accuracy: 0.9497
10300/11314 [==========================>...] - ETA: 0s - loss: 0.1690 - accuracy: 0.9497
10400/11314 [==========================>...] - ETA: 0s - loss: 0.1689 - accuracy: 0.9497
10500/11314 [==========================>...] - ETA: 0s - loss: 0.1689 - accuracy: 0.9497
10600/11314 [===========================>..] - ETA: 0s - loss: 0.1690 - accuracy: 0.9497
10700/11314 [===========================>..] - ETA: 0s - loss: 0.1689 - accuracy: 0.9497
10800/11314 [===========================>..] - ETA: 0s - loss: 0.1689 - accuracy: 0.9497
10900/11314 [===========================>..] - ETA: 0s - loss: 0.1688 - accuracy: 0.9497
11000/11314 [============================>.] - ETA: 0s - loss: 0.1687 - accuracy: 0.9497
11100/11314 [============================>.] - ETA: 0s - loss: 0.1688 - accuracy: 0.9497
11200/11314 [============================>.] - ETA: 0s - loss: 0.1687 - accuracy: 0.9497
11300/11314 [============================>.] - ETA: 0s - loss: 0.1687 - accuracy: 0.9497
11314/11314 [==============================] - 9s 759us/step - loss: 0.1686 - accuracy: 0.9497 - val_loss: 0.1664 - val_accuracy: 0.9498
Epoch 6/15

  100/11314 [..............................] - ETA: 7s - loss: 0.1620 - accuracy: 0.9511
  200/11314 [..............................] - ETA: 6s - loss: 0.1615 - accuracy: 0.9500
  300/11314 [..............................] - ETA: 6s - loss: 0.1569 - accuracy: 0.9505
  400/11314 [>.............................] - ETA: 7s - loss: 0.1568 - accuracy: 0.9503
  500/11314 [>.............................] - ETA: 7s - loss: 0.1579 - accuracy: 0.9501
  600/11314 [>.............................] - ETA: 6s - loss: 0.1579 - accuracy: 0.9502
  700/11314 [>.............................] - ETA: 6s - loss: 0.1574 - accuracy: 0.9505
  800/11314 [=>............................] - ETA: 6s - loss: 0.1568 - accuracy: 0.9506
  900/11314 [=>............................] - ETA: 6s - loss: 0.1560 - accuracy: 0.9505
 1000/11314 [=>............................] - ETA: 6s - loss: 0.1569 - accuracy: 0.9502
 1100/11314 [=>............................] - ETA: 6s - loss: 0.1559 - accuracy: 0.9502
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.1556 - accuracy: 0.9500
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.1564 - accuracy: 0.9499
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.1570 - accuracy: 0.9498
 1500/11314 [==>...........................] - ETA: 6s - loss: 0.1566 - accuracy: 0.9498
 1600/11314 [===>..........................] - ETA: 6s - loss: 0.1567 - accuracy: 0.9499
 1700/11314 [===>..........................] - ETA: 6s - loss: 0.1564 - accuracy: 0.9498
 1800/11314 [===>..........................] - ETA: 6s - loss: 0.1563 - accuracy: 0.9499
 1900/11314 [====>.........................] - ETA: 6s - loss: 0.1564 - accuracy: 0.9499
 2000/11314 [====>.........................] - ETA: 6s - loss: 0.1563 - accuracy: 0.9501
 2100/11314 [====>.........................] - ETA: 6s - loss: 0.1564 - accuracy: 0.9501
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.1564 - accuracy: 0.9500
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.1567 - accuracy: 0.9500
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.1570 - accuracy: 0.9499
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.1571 - accuracy: 0.9499
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.1570 - accuracy: 0.9499
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.1570 - accuracy: 0.9499
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.1571 - accuracy: 0.9499
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.1571 - accuracy: 0.9499
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.1569 - accuracy: 0.9500
 3100/11314 [=======>......................] - ETA: 5s - loss: 0.1567 - accuracy: 0.9500
 3200/11314 [=======>......................] - ETA: 5s - loss: 0.1567 - accuracy: 0.9499
 3300/11314 [=======>......................] - ETA: 5s - loss: 0.1565 - accuracy: 0.9499
 3400/11314 [========>.....................] - ETA: 5s - loss: 0.1566 - accuracy: 0.9499
 3500/11314 [========>.....................] - ETA: 5s - loss: 0.1562 - accuracy: 0.9499
 3600/11314 [========>.....................] - ETA: 4s - loss: 0.1561 - accuracy: 0.9500
 3700/11314 [========>.....................] - ETA: 4s - loss: 0.1560 - accuracy: 0.9500
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.1561 - accuracy: 0.9500
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.1563 - accuracy: 0.9500
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.1561 - accuracy: 0.9500
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.1561 - accuracy: 0.9500
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.1561 - accuracy: 0.9500
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.1562 - accuracy: 0.9500
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.1560 - accuracy: 0.9500
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.1558 - accuracy: 0.9500
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.1557 - accuracy: 0.9501
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.1556 - accuracy: 0.9501
 4800/11314 [===========>..................] - ETA: 4s - loss: 0.1556 - accuracy: 0.9501
 4900/11314 [===========>..................] - ETA: 4s - loss: 0.1554 - accuracy: 0.9501
 5000/11314 [============>.................] - ETA: 4s - loss: 0.1553 - accuracy: 0.9502
 5100/11314 [============>.................] - ETA: 4s - loss: 0.1553 - accuracy: 0.9502
 5200/11314 [============>.................] - ETA: 4s - loss: 0.1553 - accuracy: 0.9502
 5300/11314 [=============>................] - ETA: 3s - loss: 0.1551 - accuracy: 0.9502
 5400/11314 [=============>................] - ETA: 3s - loss: 0.1550 - accuracy: 0.9502
 5500/11314 [=============>................] - ETA: 3s - loss: 0.1549 - accuracy: 0.9501
 5600/11314 [=============>................] - ETA: 3s - loss: 0.1549 - accuracy: 0.9502
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.1549 - accuracy: 0.9502
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.1548 - accuracy: 0.9502
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.1549 - accuracy: 0.9502
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.1549 - accuracy: 0.9502
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.1549 - accuracy: 0.9502
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.1548 - accuracy: 0.9502
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.1548 - accuracy: 0.9502
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.1547 - accuracy: 0.9502
 6500/11314 [================>.............] - ETA: 3s - loss: 0.1546 - accuracy: 0.9503
 6600/11314 [================>.............] - ETA: 3s - loss: 0.1546 - accuracy: 0.9503
 6700/11314 [================>.............] - ETA: 3s - loss: 0.1547 - accuracy: 0.9503
 6800/11314 [=================>............] - ETA: 3s - loss: 0.1546 - accuracy: 0.9503
 6900/11314 [=================>............] - ETA: 2s - loss: 0.1545 - accuracy: 0.9504
 7000/11314 [=================>............] - ETA: 2s - loss: 0.1544 - accuracy: 0.9504
 7100/11314 [=================>............] - ETA: 2s - loss: 0.1542 - accuracy: 0.9504
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.1541 - accuracy: 0.9504
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.1542 - accuracy: 0.9504
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.1541 - accuracy: 0.9504
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.1541 - accuracy: 0.9504
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.1541 - accuracy: 0.9504
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.1542 - accuracy: 0.9504
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.1541 - accuracy: 0.9504
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.1540 - accuracy: 0.9504
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.1540 - accuracy: 0.9503
 8100/11314 [====================>.........] - ETA: 2s - loss: 0.1540 - accuracy: 0.9503
 8200/11314 [====================>.........] - ETA: 2s - loss: 0.1539 - accuracy: 0.9503
 8300/11314 [=====================>........] - ETA: 2s - loss: 0.1539 - accuracy: 0.9503
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.1539 - accuracy: 0.9503
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.1539 - accuracy: 0.9503
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.1539 - accuracy: 0.9503
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.1538 - accuracy: 0.9503
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.1539 - accuracy: 0.9503
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.1539 - accuracy: 0.9503
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.1539 - accuracy: 0.9503
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.1538 - accuracy: 0.9503
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.1538 - accuracy: 0.9503
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.1538 - accuracy: 0.9503
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.1537 - accuracy: 0.9503
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.1537 - accuracy: 0.9503
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.1538 - accuracy: 0.9503
 9700/11314 [========================>.....] - ETA: 1s - loss: 0.1538 - accuracy: 0.9503
 9800/11314 [========================>.....] - ETA: 1s - loss: 0.1537 - accuracy: 0.9503
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.1538 - accuracy: 0.9503
10000/11314 [=========================>....] - ETA: 0s - loss: 0.1537 - accuracy: 0.9503
10100/11314 [=========================>....] - ETA: 0s - loss: 0.1536 - accuracy: 0.9504
10200/11314 [==========================>...] - ETA: 0s - loss: 0.1536 - accuracy: 0.9504
10300/11314 [==========================>...] - ETA: 0s - loss: 0.1535 - accuracy: 0.9504
10400/11314 [==========================>...] - ETA: 0s - loss: 0.1535 - accuracy: 0.9504
10500/11314 [==========================>...] - ETA: 0s - loss: 0.1534 - accuracy: 0.9504
10600/11314 [===========================>..] - ETA: 0s - loss: 0.1535 - accuracy: 0.9504
10700/11314 [===========================>..] - ETA: 0s - loss: 0.1535 - accuracy: 0.9504
10800/11314 [===========================>..] - ETA: 0s - loss: 0.1535 - accuracy: 0.9504
10900/11314 [===========================>..] - ETA: 0s - loss: 0.1535 - accuracy: 0.9504
11000/11314 [============================>.] - ETA: 0s - loss: 0.1536 - accuracy: 0.9504
11100/11314 [============================>.] - ETA: 0s - loss: 0.1535 - accuracy: 0.9504
11200/11314 [============================>.] - ETA: 0s - loss: 0.1535 - accuracy: 0.9504
11300/11314 [============================>.] - ETA: 0s - loss: 0.1534 - accuracy: 0.9504
11314/11314 [==============================] - 9s 783us/step - loss: 0.1534 - accuracy: 0.9504 - val_loss: 0.1608 - val_accuracy: 0.9504
Epoch 7/15

  100/11314 [..............................] - ETA: 7s - loss: 0.1508 - accuracy: 0.9500
  200/11314 [..............................] - ETA: 7s - loss: 0.1444 - accuracy: 0.9526
  300/11314 [..............................] - ETA: 7s - loss: 0.1472 - accuracy: 0.9514
  400/11314 [>.............................] - ETA: 7s - loss: 0.1443 - accuracy: 0.9518
  500/11314 [>.............................] - ETA: 7s - loss: 0.1451 - accuracy: 0.9517
  600/11314 [>.............................] - ETA: 7s - loss: 0.1442 - accuracy: 0.9518
  700/11314 [>.............................] - ETA: 7s - loss: 0.1437 - accuracy: 0.9515
  800/11314 [=>............................] - ETA: 7s - loss: 0.1440 - accuracy: 0.9514
  900/11314 [=>............................] - ETA: 6s - loss: 0.1444 - accuracy: 0.9513
 1000/11314 [=>............................] - ETA: 6s - loss: 0.1446 - accuracy: 0.9513
 1100/11314 [=>............................] - ETA: 6s - loss: 0.1449 - accuracy: 0.9513
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.1450 - accuracy: 0.9511
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.1448 - accuracy: 0.9511
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.1450 - accuracy: 0.9510
 1500/11314 [==>...........................] - ETA: 6s - loss: 0.1449 - accuracy: 0.9510
 1600/11314 [===>..........................] - ETA: 6s - loss: 0.1451 - accuracy: 0.9510
 1700/11314 [===>..........................] - ETA: 6s - loss: 0.1449 - accuracy: 0.9510
 1800/11314 [===>..........................] - ETA: 6s - loss: 0.1446 - accuracy: 0.9509
 1900/11314 [====>.........................] - ETA: 6s - loss: 0.1442 - accuracy: 0.9510
 2000/11314 [====>.........................] - ETA: 6s - loss: 0.1443 - accuracy: 0.9509
 2100/11314 [====>.........................] - ETA: 6s - loss: 0.1443 - accuracy: 0.9509
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.1444 - accuracy: 0.9509
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.1440 - accuracy: 0.9510
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.1439 - accuracy: 0.9510
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.1436 - accuracy: 0.9510
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.1434 - accuracy: 0.9510
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.1437 - accuracy: 0.9510
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.1437 - accuracy: 0.9511
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.1436 - accuracy: 0.9511
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.1435 - accuracy: 0.9511
 3100/11314 [=======>......................] - ETA: 5s - loss: 0.1434 - accuracy: 0.9511
 3200/11314 [=======>......................] - ETA: 5s - loss: 0.1433 - accuracy: 0.9511
 3300/11314 [=======>......................] - ETA: 5s - loss: 0.1432 - accuracy: 0.9512
 3400/11314 [========>.....................] - ETA: 5s - loss: 0.1433 - accuracy: 0.9511
 3500/11314 [========>.....................] - ETA: 5s - loss: 0.1433 - accuracy: 0.9510
 3600/11314 [========>.....................] - ETA: 4s - loss: 0.1432 - accuracy: 0.9511
 3700/11314 [========>.....................] - ETA: 4s - loss: 0.1436 - accuracy: 0.9510
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.1434 - accuracy: 0.9510
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.1432 - accuracy: 0.9511
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.1431 - accuracy: 0.9511
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.1429 - accuracy: 0.9512
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.1431 - accuracy: 0.9511
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.1430 - accuracy: 0.9512
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.1428 - accuracy: 0.9512
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.1428 - accuracy: 0.9512
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.1428 - accuracy: 0.9512
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.1427 - accuracy: 0.9512
 4800/11314 [===========>..................] - ETA: 4s - loss: 0.1427 - accuracy: 0.9512
 4900/11314 [===========>..................] - ETA: 4s - loss: 0.1428 - accuracy: 0.9512
 5000/11314 [============>.................] - ETA: 4s - loss: 0.1428 - accuracy: 0.9512
 5100/11314 [============>.................] - ETA: 4s - loss: 0.1428 - accuracy: 0.9512
 5200/11314 [============>.................] - ETA: 3s - loss: 0.1429 - accuracy: 0.9512
 5300/11314 [=============>................] - ETA: 3s - loss: 0.1428 - accuracy: 0.9513
 5400/11314 [=============>................] - ETA: 3s - loss: 0.1427 - accuracy: 0.9513
 5500/11314 [=============>................] - ETA: 3s - loss: 0.1427 - accuracy: 0.9513
 5600/11314 [=============>................] - ETA: 3s - loss: 0.1427 - accuracy: 0.9513
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.1427 - accuracy: 0.9512
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.1427 - accuracy: 0.9513
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.1427 - accuracy: 0.9513
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.1427 - accuracy: 0.9513
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.1427 - accuracy: 0.9513
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.1426 - accuracy: 0.9513
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.1425 - accuracy: 0.9513
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.1426 - accuracy: 0.9513
 6500/11314 [================>.............] - ETA: 3s - loss: 0.1427 - accuracy: 0.9513
 6600/11314 [================>.............] - ETA: 3s - loss: 0.1426 - accuracy: 0.9513
 6700/11314 [================>.............] - ETA: 3s - loss: 0.1425 - accuracy: 0.9513
 6800/11314 [=================>............] - ETA: 3s - loss: 0.1426 - accuracy: 0.9513
 6900/11314 [=================>............] - ETA: 2s - loss: 0.1424 - accuracy: 0.9514
 7000/11314 [=================>............] - ETA: 2s - loss: 0.1424 - accuracy: 0.9513
 7100/11314 [=================>............] - ETA: 2s - loss: 0.1423 - accuracy: 0.9514
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.1422 - accuracy: 0.9514
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.1422 - accuracy: 0.9514
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.1421 - accuracy: 0.9514
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.1423 - accuracy: 0.9514
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.1422 - accuracy: 0.9514
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.1422 - accuracy: 0.9514
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.1423 - accuracy: 0.9514
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.1422 - accuracy: 0.9514
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.1422 - accuracy: 0.9514
 8100/11314 [====================>.........] - ETA: 2s - loss: 0.1423 - accuracy: 0.9514
 8200/11314 [====================>.........] - ETA: 2s - loss: 0.1422 - accuracy: 0.9514
 8300/11314 [=====================>........] - ETA: 2s - loss: 0.1423 - accuracy: 0.9514
 8400/11314 [=====================>........] - ETA: 2s - loss: 0.1422 - accuracy: 0.9514
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.1423 - accuracy: 0.9513
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.1423 - accuracy: 0.9513
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.1423 - accuracy: 0.9513
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.1422 - accuracy: 0.9514
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.1421 - accuracy: 0.9514
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.1421 - accuracy: 0.9514
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.1420 - accuracy: 0.9514
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.1421 - accuracy: 0.9514
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.1421 - accuracy: 0.9514
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.1421 - accuracy: 0.9514
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.1421 - accuracy: 0.9513
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.1421 - accuracy: 0.9513
 9700/11314 [========================>.....] - ETA: 1s - loss: 0.1421 - accuracy: 0.9513
 9800/11314 [========================>.....] - ETA: 1s - loss: 0.1422 - accuracy: 0.9513
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.1422 - accuracy: 0.9513
10000/11314 [=========================>....] - ETA: 0s - loss: 0.1421 - accuracy: 0.9514
10100/11314 [=========================>....] - ETA: 0s - loss: 0.1422 - accuracy: 0.9513
10200/11314 [==========================>...] - ETA: 0s - loss: 0.1422 - accuracy: 0.9513
10300/11314 [==========================>...] - ETA: 0s - loss: 0.1421 - accuracy: 0.9513
10400/11314 [==========================>...] - ETA: 0s - loss: 0.1420 - accuracy: 0.9514
10500/11314 [==========================>...] - ETA: 0s - loss: 0.1421 - accuracy: 0.9514
10600/11314 [===========================>..] - ETA: 0s - loss: 0.1420 - accuracy: 0.9514
10700/11314 [===========================>..] - ETA: 0s - loss: 0.1421 - accuracy: 0.9514
10800/11314 [===========================>..] - ETA: 0s - loss: 0.1420 - accuracy: 0.9514
10900/11314 [===========================>..] - ETA: 0s - loss: 0.1420 - accuracy: 0.9514
11000/11314 [============================>.] - ETA: 0s - loss: 0.1420 - accuracy: 0.9514
11100/11314 [============================>.] - ETA: 0s - loss: 0.1419 - accuracy: 0.9514
11200/11314 [============================>.] - ETA: 0s - loss: 0.1420 - accuracy: 0.9514
11300/11314 [============================>.] - ETA: 0s - loss: 0.1420 - accuracy: 0.9514
11314/11314 [==============================] - 9s 800us/step - loss: 0.1420 - accuracy: 0.9514 - val_loss: 0.1534 - val_accuracy: 0.9510
Epoch 8/15

  100/11314 [..............................] - ETA: 9s - loss: 0.1317 - accuracy: 0.9542
  200/11314 [..............................] - ETA: 8s - loss: 0.1340 - accuracy: 0.9534
  300/11314 [..............................] - ETA: 7s - loss: 0.1308 - accuracy: 0.9539
  400/11314 [>.............................] - ETA: 7s - loss: 0.1304 - accuracy: 0.9541
  500/11314 [>.............................] - ETA: 7s - loss: 0.1319 - accuracy: 0.9538
  600/11314 [>.............................] - ETA: 7s - loss: 0.1329 - accuracy: 0.9537
  700/11314 [>.............................] - ETA: 7s - loss: 0.1330 - accuracy: 0.9533
  800/11314 [=>............................] - ETA: 7s - loss: 0.1337 - accuracy: 0.9534
  900/11314 [=>............................] - ETA: 6s - loss: 0.1340 - accuracy: 0.9534
 1000/11314 [=>............................] - ETA: 6s - loss: 0.1347 - accuracy: 0.9534
 1100/11314 [=>............................] - ETA: 6s - loss: 0.1344 - accuracy: 0.9537
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.1352 - accuracy: 0.9535
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.1344 - accuracy: 0.9536
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.1345 - accuracy: 0.9537
 1500/11314 [==>...........................] - ETA: 6s - loss: 0.1339 - accuracy: 0.9536
 1600/11314 [===>..........................] - ETA: 6s - loss: 0.1333 - accuracy: 0.9538
 1700/11314 [===>..........................] - ETA: 6s - loss: 0.1335 - accuracy: 0.9537
 1800/11314 [===>..........................] - ETA: 6s - loss: 0.1330 - accuracy: 0.9535
 1900/11314 [====>.........................] - ETA: 6s - loss: 0.1334 - accuracy: 0.9534
 2000/11314 [====>.........................] - ETA: 6s - loss: 0.1330 - accuracy: 0.9535
 2100/11314 [====>.........................] - ETA: 6s - loss: 0.1334 - accuracy: 0.9533
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.1335 - accuracy: 0.9533
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.1332 - accuracy: 0.9535
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.1331 - accuracy: 0.9535
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.1336 - accuracy: 0.9533
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.1332 - accuracy: 0.9534
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.1335 - accuracy: 0.9535
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.1334 - accuracy: 0.9535
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.1335 - accuracy: 0.9534
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.1334 - accuracy: 0.9534
 3100/11314 [=======>......................] - ETA: 5s - loss: 0.1337 - accuracy: 0.9534
 3200/11314 [=======>......................] - ETA: 5s - loss: 0.1337 - accuracy: 0.9534
 3300/11314 [=======>......................] - ETA: 5s - loss: 0.1337 - accuracy: 0.9535
 3400/11314 [========>.....................] - ETA: 5s - loss: 0.1334 - accuracy: 0.9537
 3500/11314 [========>.....................] - ETA: 5s - loss: 0.1333 - accuracy: 0.9537
 3600/11314 [========>.....................] - ETA: 5s - loss: 0.1334 - accuracy: 0.9537
 3700/11314 [========>.....................] - ETA: 5s - loss: 0.1333 - accuracy: 0.9537
 3800/11314 [=========>....................] - ETA: 5s - loss: 0.1333 - accuracy: 0.9537
 3900/11314 [=========>....................] - ETA: 5s - loss: 0.1329 - accuracy: 0.9538
 4000/11314 [=========>....................] - ETA: 5s - loss: 0.1328 - accuracy: 0.9538
 4100/11314 [=========>....................] - ETA: 5s - loss: 0.1328 - accuracy: 0.9538
 4200/11314 [==========>...................] - ETA: 5s - loss: 0.1326 - accuracy: 0.9537
 4300/11314 [==========>...................] - ETA: 5s - loss: 0.1328 - accuracy: 0.9537
 4400/11314 [==========>...................] - ETA: 5s - loss: 0.1327 - accuracy: 0.9536
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.1327 - accuracy: 0.9537
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.1329 - accuracy: 0.9536
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.1329 - accuracy: 0.9537
 4800/11314 [===========>..................] - ETA: 4s - loss: 0.1328 - accuracy: 0.9536
 4900/11314 [===========>..................] - ETA: 4s - loss: 0.1327 - accuracy: 0.9536
 5000/11314 [============>.................] - ETA: 4s - loss: 0.1327 - accuracy: 0.9535
 5100/11314 [============>.................] - ETA: 4s - loss: 0.1326 - accuracy: 0.9535
 5200/11314 [============>.................] - ETA: 4s - loss: 0.1325 - accuracy: 0.9535
 5300/11314 [=============>................] - ETA: 4s - loss: 0.1324 - accuracy: 0.9534
 5400/11314 [=============>................] - ETA: 4s - loss: 0.1324 - accuracy: 0.9534
 5500/11314 [=============>................] - ETA: 4s - loss: 0.1323 - accuracy: 0.9534
 5600/11314 [=============>................] - ETA: 4s - loss: 0.1323 - accuracy: 0.9535
 5700/11314 [==============>...............] - ETA: 4s - loss: 0.1321 - accuracy: 0.9536
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.1321 - accuracy: 0.9536
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.1321 - accuracy: 0.9535
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.1320 - accuracy: 0.9535
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.1320 - accuracy: 0.9535
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.1320 - accuracy: 0.9535
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.1319 - accuracy: 0.9535
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.1319 - accuracy: 0.9535
 6500/11314 [================>.............] - ETA: 3s - loss: 0.1319 - accuracy: 0.9536
 6600/11314 [================>.............] - ETA: 3s - loss: 0.1319 - accuracy: 0.9536
 6700/11314 [================>.............] - ETA: 3s - loss: 0.1319 - accuracy: 0.9536
 6800/11314 [=================>............] - ETA: 3s - loss: 0.1319 - accuracy: 0.9535
 6900/11314 [=================>............] - ETA: 3s - loss: 0.1319 - accuracy: 0.9535
 7000/11314 [=================>............] - ETA: 3s - loss: 0.1319 - accuracy: 0.9536
 7100/11314 [=================>............] - ETA: 2s - loss: 0.1317 - accuracy: 0.9536
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.1317 - accuracy: 0.9536
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.1317 - accuracy: 0.9535
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.1318 - accuracy: 0.9535
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.1317 - accuracy: 0.9535
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.1317 - accuracy: 0.9535
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.1317 - accuracy: 0.9535
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.1316 - accuracy: 0.9535
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.1316 - accuracy: 0.9535
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.1317 - accuracy: 0.9536
 8100/11314 [====================>.........] - ETA: 2s - loss: 0.1316 - accuracy: 0.9536
 8200/11314 [====================>.........] - ETA: 2s - loss: 0.1316 - accuracy: 0.9536
 8300/11314 [=====================>........] - ETA: 2s - loss: 0.1314 - accuracy: 0.9536
 8400/11314 [=====================>........] - ETA: 2s - loss: 0.1315 - accuracy: 0.9536
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.1315 - accuracy: 0.9536
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.1314 - accuracy: 0.9536
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.1313 - accuracy: 0.9536
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.1313 - accuracy: 0.9536
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.1313 - accuracy: 0.9536
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.1312 - accuracy: 0.9537
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.1312 - accuracy: 0.9537
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.1312 - accuracy: 0.9537
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.1312 - accuracy: 0.9537
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.1314 - accuracy: 0.9537
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.1315 - accuracy: 0.9536
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.1316 - accuracy: 0.9536
 9700/11314 [========================>.....] - ETA: 1s - loss: 0.1317 - accuracy: 0.9536
 9800/11314 [========================>.....] - ETA: 1s - loss: 0.1317 - accuracy: 0.9536
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.1319 - accuracy: 0.9536
10000/11314 [=========================>....] - ETA: 0s - loss: 0.1320 - accuracy: 0.9536
10100/11314 [=========================>....] - ETA: 0s - loss: 0.1321 - accuracy: 0.9536
10200/11314 [==========================>...] - ETA: 0s - loss: 0.1321 - accuracy: 0.9536
10300/11314 [==========================>...] - ETA: 0s - loss: 0.1322 - accuracy: 0.9536
10400/11314 [==========================>...] - ETA: 0s - loss: 0.1324 - accuracy: 0.9536
10500/11314 [==========================>...] - ETA: 0s - loss: 0.1324 - accuracy: 0.9536
10600/11314 [===========================>..] - ETA: 0s - loss: 0.1325 - accuracy: 0.9536
10700/11314 [===========================>..] - ETA: 0s - loss: 0.1325 - accuracy: 0.9536
10800/11314 [===========================>..] - ETA: 0s - loss: 0.1325 - accuracy: 0.9536
10900/11314 [===========================>..] - ETA: 0s - loss: 0.1327 - accuracy: 0.9536
11000/11314 [============================>.] - ETA: 0s - loss: 0.1328 - accuracy: 0.9536
11100/11314 [============================>.] - ETA: 0s - loss: 0.1328 - accuracy: 0.9535
11200/11314 [============================>.] - ETA: 0s - loss: 0.1330 - accuracy: 0.9535
11300/11314 [============================>.] - ETA: 0s - loss: 0.1330 - accuracy: 0.9535
11314/11314 [==============================] - 9s 806us/step - loss: 0.1330 - accuracy: 0.9535 - val_loss: 0.1518 - val_accuracy: 0.9519
Epoch 9/15

  100/11314 [..............................] - ETA: 8s - loss: 0.1368 - accuracy: 0.9521
  200/11314 [..............................] - ETA: 7s - loss: 0.1377 - accuracy: 0.9505
  300/11314 [..............................] - ETA: 7s - loss: 0.1357 - accuracy: 0.9514
  400/11314 [>.............................] - ETA: 7s - loss: 0.1368 - accuracy: 0.9517
  500/11314 [>.............................] - ETA: 7s - loss: 0.1380 - accuracy: 0.9516
  600/11314 [>.............................] - ETA: 7s - loss: 0.1375 - accuracy: 0.9519
  700/11314 [>.............................] - ETA: 7s - loss: 0.1370 - accuracy: 0.9523
  800/11314 [=>............................] - ETA: 7s - loss: 0.1366 - accuracy: 0.9522
  900/11314 [=>............................] - ETA: 7s - loss: 0.1350 - accuracy: 0.9525
 1000/11314 [=>............................] - ETA: 7s - loss: 0.1336 - accuracy: 0.9528
 1100/11314 [=>............................] - ETA: 7s - loss: 0.1337 - accuracy: 0.9528
 1200/11314 [==>...........................] - ETA: 7s - loss: 0.1328 - accuracy: 0.9529
 1300/11314 [==>...........................] - ETA: 7s - loss: 0.1325 - accuracy: 0.9530
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.1326 - accuracy: 0.9530
 1500/11314 [==>...........................] - ETA: 6s - loss: 0.1330 - accuracy: 0.9529
 1600/11314 [===>..........................] - ETA: 6s - loss: 0.1333 - accuracy: 0.9529
 1700/11314 [===>..........................] - ETA: 6s - loss: 0.1332 - accuracy: 0.9532
 1800/11314 [===>..........................] - ETA: 6s - loss: 0.1327 - accuracy: 0.9533
 1900/11314 [====>.........................] - ETA: 6s - loss: 0.1324 - accuracy: 0.9534
 2000/11314 [====>.........................] - ETA: 6s - loss: 0.1328 - accuracy: 0.9534
 2100/11314 [====>.........................] - ETA: 6s - loss: 0.1332 - accuracy: 0.9532
 2200/11314 [====>.........................] - ETA: 6s - loss: 0.1327 - accuracy: 0.9531
 2300/11314 [=====>........................] - ETA: 6s - loss: 0.1325 - accuracy: 0.9531
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.1322 - accuracy: 0.9532
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.1322 - accuracy: 0.9532
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.1322 - accuracy: 0.9531
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.1320 - accuracy: 0.9532
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.1319 - accuracy: 0.9531
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.1318 - accuracy: 0.9532
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.1317 - accuracy: 0.9532
 3100/11314 [=======>......................] - ETA: 5s - loss: 0.1314 - accuracy: 0.9532
 3200/11314 [=======>......................] - ETA: 5s - loss: 0.1313 - accuracy: 0.9533
 3300/11314 [=======>......................] - ETA: 5s - loss: 0.1313 - accuracy: 0.9533
 3400/11314 [========>.....................] - ETA: 5s - loss: 0.1311 - accuracy: 0.9533
 3500/11314 [========>.....................] - ETA: 5s - loss: 0.1311 - accuracy: 0.9533
 3600/11314 [========>.....................] - ETA: 5s - loss: 0.1315 - accuracy: 0.9533
 3700/11314 [========>.....................] - ETA: 5s - loss: 0.1310 - accuracy: 0.9534
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.1306 - accuracy: 0.9534
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.1306 - accuracy: 0.9535
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.1307 - accuracy: 0.9534
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.1305 - accuracy: 0.9535
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.1305 - accuracy: 0.9535
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.1305 - accuracy: 0.9536
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.1305 - accuracy: 0.9537
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.1304 - accuracy: 0.9538
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.1305 - accuracy: 0.9537
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.1304 - accuracy: 0.9537
 4800/11314 [===========>..................] - ETA: 4s - loss: 0.1303 - accuracy: 0.9537
 4900/11314 [===========>..................] - ETA: 4s - loss: 0.1301 - accuracy: 0.9538
 5000/11314 [============>.................] - ETA: 4s - loss: 0.1301 - accuracy: 0.9537
 5100/11314 [============>.................] - ETA: 4s - loss: 0.1300 - accuracy: 0.9538
 5200/11314 [============>.................] - ETA: 3s - loss: 0.1299 - accuracy: 0.9538
 5300/11314 [=============>................] - ETA: 3s - loss: 0.1299 - accuracy: 0.9537
 5400/11314 [=============>................] - ETA: 3s - loss: 0.1299 - accuracy: 0.9537
 5500/11314 [=============>................] - ETA: 3s - loss: 0.1299 - accuracy: 0.9537
 5600/11314 [=============>................] - ETA: 3s - loss: 0.1299 - accuracy: 0.9537
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.1297 - accuracy: 0.9538
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.1298 - accuracy: 0.9538
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.1297 - accuracy: 0.9538
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.1298 - accuracy: 0.9537
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.1297 - accuracy: 0.9538
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.1297 - accuracy: 0.9538
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.1298 - accuracy: 0.9539
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.1297 - accuracy: 0.9539
 6500/11314 [================>.............] - ETA: 3s - loss: 0.1297 - accuracy: 0.9539
 6600/11314 [================>.............] - ETA: 3s - loss: 0.1296 - accuracy: 0.9539
 6700/11314 [================>.............] - ETA: 3s - loss: 0.1296 - accuracy: 0.9539
 6800/11314 [=================>............] - ETA: 2s - loss: 0.1295 - accuracy: 0.9540
 6900/11314 [=================>............] - ETA: 2s - loss: 0.1294 - accuracy: 0.9539
 7000/11314 [=================>............] - ETA: 2s - loss: 0.1292 - accuracy: 0.9540
 7100/11314 [=================>............] - ETA: 2s - loss: 0.1290 - accuracy: 0.9540
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.1288 - accuracy: 0.9541
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.1288 - accuracy: 0.9541
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.1287 - accuracy: 0.9542
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.1288 - accuracy: 0.9541
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.1288 - accuracy: 0.9541
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.1287 - accuracy: 0.9541
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.1287 - accuracy: 0.9541
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.1286 - accuracy: 0.9541
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.1285 - accuracy: 0.9541
 8100/11314 [====================>.........] - ETA: 2s - loss: 0.1284 - accuracy: 0.9542
 8200/11314 [====================>.........] - ETA: 2s - loss: 0.1284 - accuracy: 0.9542
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.1283 - accuracy: 0.9542
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.1283 - accuracy: 0.9542
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.1281 - accuracy: 0.9542
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.1281 - accuracy: 0.9542
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.1279 - accuracy: 0.9543
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.1279 - accuracy: 0.9543
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.1279 - accuracy: 0.9543
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.1279 - accuracy: 0.9543
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.1279 - accuracy: 0.9543
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.1277 - accuracy: 0.9543
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.1277 - accuracy: 0.9543
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.1277 - accuracy: 0.9543
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.1276 - accuracy: 0.9543
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.1276 - accuracy: 0.9543
 9700/11314 [========================>.....] - ETA: 1s - loss: 0.1276 - accuracy: 0.9543
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.1275 - accuracy: 0.9544
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.1275 - accuracy: 0.9544
10000/11314 [=========================>....] - ETA: 0s - loss: 0.1274 - accuracy: 0.9544
10100/11314 [=========================>....] - ETA: 0s - loss: 0.1273 - accuracy: 0.9544
10200/11314 [==========================>...] - ETA: 0s - loss: 0.1273 - accuracy: 0.9544
10300/11314 [==========================>...] - ETA: 0s - loss: 0.1273 - accuracy: 0.9544
10400/11314 [==========================>...] - ETA: 0s - loss: 0.1272 - accuracy: 0.9544
10500/11314 [==========================>...] - ETA: 0s - loss: 0.1272 - accuracy: 0.9545
10600/11314 [===========================>..] - ETA: 0s - loss: 0.1272 - accuracy: 0.9545
10700/11314 [===========================>..] - ETA: 0s - loss: 0.1272 - accuracy: 0.9544
10800/11314 [===========================>..] - ETA: 0s - loss: 0.1271 - accuracy: 0.9544
10900/11314 [===========================>..] - ETA: 0s - loss: 0.1270 - accuracy: 0.9545
11000/11314 [============================>.] - ETA: 0s - loss: 0.1270 - accuracy: 0.9545
11100/11314 [============================>.] - ETA: 0s - loss: 0.1270 - accuracy: 0.9545
11200/11314 [============================>.] - ETA: 0s - loss: 0.1269 - accuracy: 0.9545
11300/11314 [============================>.] - ETA: 0s - loss: 0.1269 - accuracy: 0.9545
11314/11314 [==============================] - 9s 791us/step - loss: 0.1269 - accuracy: 0.9545 - val_loss: 0.1458 - val_accuracy: 0.9535
Epoch 10/15

  100/11314 [..............................] - ETA: 6s - loss: 0.1142 - accuracy: 0.9605
  200/11314 [..............................] - ETA: 6s - loss: 0.1150 - accuracy: 0.9595
  300/11314 [..............................] - ETA: 6s - loss: 0.1109 - accuracy: 0.9598
  400/11314 [>.............................] - ETA: 6s - loss: 0.1097 - accuracy: 0.9600
  500/11314 [>.............................] - ETA: 6s - loss: 0.1111 - accuracy: 0.9595
  600/11314 [>.............................] - ETA: 6s - loss: 0.1126 - accuracy: 0.9590
  700/11314 [>.............................] - ETA: 6s - loss: 0.1129 - accuracy: 0.9586
  800/11314 [=>............................] - ETA: 6s - loss: 0.1137 - accuracy: 0.9580
  900/11314 [=>............................] - ETA: 6s - loss: 0.1139 - accuracy: 0.9578
 1000/11314 [=>............................] - ETA: 6s - loss: 0.1140 - accuracy: 0.9578
 1100/11314 [=>............................] - ETA: 6s - loss: 0.1145 - accuracy: 0.9576
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.1149 - accuracy: 0.9576
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.1159 - accuracy: 0.9573
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.1162 - accuracy: 0.9570
 1500/11314 [==>...........................] - ETA: 6s - loss: 0.1161 - accuracy: 0.9572
 1600/11314 [===>..........................] - ETA: 6s - loss: 0.1162 - accuracy: 0.9571
 1700/11314 [===>..........................] - ETA: 6s - loss: 0.1164 - accuracy: 0.9570
 1800/11314 [===>..........................] - ETA: 5s - loss: 0.1163 - accuracy: 0.9568
 1900/11314 [====>.........................] - ETA: 5s - loss: 0.1163 - accuracy: 0.9569
 2000/11314 [====>.........................] - ETA: 5s - loss: 0.1157 - accuracy: 0.9571
 2100/11314 [====>.........................] - ETA: 5s - loss: 0.1158 - accuracy: 0.9571
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.1158 - accuracy: 0.9571
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.1157 - accuracy: 0.9569
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.1158 - accuracy: 0.9568
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.1157 - accuracy: 0.9568
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.1159 - accuracy: 0.9568
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.1161 - accuracy: 0.9566
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.1161 - accuracy: 0.9566
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.1162 - accuracy: 0.9565
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.1159 - accuracy: 0.9566
 3100/11314 [=======>......................] - ETA: 5s - loss: 0.1160 - accuracy: 0.9566
 3200/11314 [=======>......................] - ETA: 5s - loss: 0.1161 - accuracy: 0.9565
 3300/11314 [=======>......................] - ETA: 5s - loss: 0.1161 - accuracy: 0.9565
 3400/11314 [========>.....................] - ETA: 4s - loss: 0.1162 - accuracy: 0.9564
 3500/11314 [========>.....................] - ETA: 4s - loss: 0.1166 - accuracy: 0.9564
 3600/11314 [========>.....................] - ETA: 4s - loss: 0.1168 - accuracy: 0.9564
 3700/11314 [========>.....................] - ETA: 4s - loss: 0.1165 - accuracy: 0.9565
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.1166 - accuracy: 0.9564
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.1166 - accuracy: 0.9565
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.1164 - accuracy: 0.9565
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.1164 - accuracy: 0.9566
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.1164 - accuracy: 0.9566
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.1163 - accuracy: 0.9566
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.1164 - accuracy: 0.9565
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.1163 - accuracy: 0.9566
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.1163 - accuracy: 0.9567
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.1162 - accuracy: 0.9566
 4800/11314 [===========>..................] - ETA: 4s - loss: 0.1160 - accuracy: 0.9567
 4900/11314 [===========>..................] - ETA: 4s - loss: 0.1162 - accuracy: 0.9566
 5000/11314 [============>.................] - ETA: 3s - loss: 0.1163 - accuracy: 0.9565
 5100/11314 [============>.................] - ETA: 3s - loss: 0.1164 - accuracy: 0.9566
 5200/11314 [============>.................] - ETA: 3s - loss: 0.1164 - accuracy: 0.9566
 5300/11314 [=============>................] - ETA: 3s - loss: 0.1165 - accuracy: 0.9566
 5400/11314 [=============>................] - ETA: 3s - loss: 0.1164 - accuracy: 0.9566
 5500/11314 [=============>................] - ETA: 3s - loss: 0.1164 - accuracy: 0.9566
 5600/11314 [=============>................] - ETA: 3s - loss: 0.1162 - accuracy: 0.9566
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.1163 - accuracy: 0.9566
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.1163 - accuracy: 0.9566
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.1163 - accuracy: 0.9567
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.1163 - accuracy: 0.9566
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.1162 - accuracy: 0.9567
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.1162 - accuracy: 0.9567
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.1161 - accuracy: 0.9567
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.1161 - accuracy: 0.9567
 6500/11314 [================>.............] - ETA: 3s - loss: 0.1161 - accuracy: 0.9567
 6600/11314 [================>.............] - ETA: 2s - loss: 0.1162 - accuracy: 0.9567
 6700/11314 [================>.............] - ETA: 2s - loss: 0.1163 - accuracy: 0.9566
 6800/11314 [=================>............] - ETA: 2s - loss: 0.1164 - accuracy: 0.9566
 6900/11314 [=================>............] - ETA: 2s - loss: 0.1163 - accuracy: 0.9566
 7000/11314 [=================>............] - ETA: 2s - loss: 0.1163 - accuracy: 0.9567
 7100/11314 [=================>............] - ETA: 2s - loss: 0.1163 - accuracy: 0.9567
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.1162 - accuracy: 0.9567
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.1161 - accuracy: 0.9567
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.1162 - accuracy: 0.9568
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.1162 - accuracy: 0.9568
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.1162 - accuracy: 0.9568
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.1161 - accuracy: 0.9567
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.1162 - accuracy: 0.9567
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.1162 - accuracy: 0.9566
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.1162 - accuracy: 0.9567
 8100/11314 [====================>.........] - ETA: 2s - loss: 0.1161 - accuracy: 0.9567
 8200/11314 [====================>.........] - ETA: 1s - loss: 0.1160 - accuracy: 0.9567
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.1160 - accuracy: 0.9568
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.1161 - accuracy: 0.9567
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.1161 - accuracy: 0.9567
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.1159 - accuracy: 0.9567
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.1159 - accuracy: 0.9568
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.1160 - accuracy: 0.9567
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.1160 - accuracy: 0.9567
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.1159 - accuracy: 0.9568
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.1159 - accuracy: 0.9568
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.1159 - accuracy: 0.9568
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.1160 - accuracy: 0.9567
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.1160 - accuracy: 0.9567
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.1161 - accuracy: 0.9567
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.1161 - accuracy: 0.9567
 9700/11314 [========================>.....] - ETA: 1s - loss: 0.1160 - accuracy: 0.9567
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.1159 - accuracy: 0.9567
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.1159 - accuracy: 0.9568
10000/11314 [=========================>....] - ETA: 0s - loss: 0.1159 - accuracy: 0.9568
10100/11314 [=========================>....] - ETA: 0s - loss: 0.1159 - accuracy: 0.9568
10200/11314 [==========================>...] - ETA: 0s - loss: 0.1158 - accuracy: 0.9568
10300/11314 [==========================>...] - ETA: 0s - loss: 0.1158 - accuracy: 0.9568
10400/11314 [==========================>...] - ETA: 0s - loss: 0.1158 - accuracy: 0.9568
10500/11314 [==========================>...] - ETA: 0s - loss: 0.1157 - accuracy: 0.9568
10600/11314 [===========================>..] - ETA: 0s - loss: 0.1157 - accuracy: 0.9568
10700/11314 [===========================>..] - ETA: 0s - loss: 0.1156 - accuracy: 0.9568
10800/11314 [===========================>..] - ETA: 0s - loss: 0.1157 - accuracy: 0.9568
10900/11314 [===========================>..] - ETA: 0s - loss: 0.1156 - accuracy: 0.9568
11000/11314 [============================>.] - ETA: 0s - loss: 0.1157 - accuracy: 0.9568
11100/11314 [============================>.] - ETA: 0s - loss: 0.1156 - accuracy: 0.9568
11200/11314 [============================>.] - ETA: 0s - loss: 0.1155 - accuracy: 0.9569
11300/11314 [============================>.] - ETA: 0s - loss: 0.1155 - accuracy: 0.9569
11314/11314 [==============================] - 9s 753us/step - loss: 0.1156 - accuracy: 0.9568 - val_loss: 0.1413 - val_accuracy: 0.9548
Epoch 11/15

  100/11314 [..............................] - ETA: 6s - loss: 0.1002 - accuracy: 0.9605
  200/11314 [..............................] - ETA: 6s - loss: 0.1019 - accuracy: 0.9616
  300/11314 [..............................] - ETA: 6s - loss: 0.1040 - accuracy: 0.9612
  400/11314 [>.............................] - ETA: 6s - loss: 0.1048 - accuracy: 0.9609
  500/11314 [>.............................] - ETA: 6s - loss: 0.1044 - accuracy: 0.9612
  600/11314 [>.............................] - ETA: 6s - loss: 0.1053 - accuracy: 0.9604
  700/11314 [>.............................] - ETA: 6s - loss: 0.1050 - accuracy: 0.9602
  800/11314 [=>............................] - ETA: 6s - loss: 0.1058 - accuracy: 0.9599
  900/11314 [=>............................] - ETA: 6s - loss: 0.1064 - accuracy: 0.9593
 1000/11314 [=>............................] - ETA: 6s - loss: 0.1065 - accuracy: 0.9595
 1100/11314 [=>............................] - ETA: 6s - loss: 0.1059 - accuracy: 0.9596
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.1069 - accuracy: 0.9591
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.1063 - accuracy: 0.9593
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.1067 - accuracy: 0.9593
 1500/11314 [==>...........................] - ETA: 6s - loss: 0.1071 - accuracy: 0.9592
 1600/11314 [===>..........................] - ETA: 6s - loss: 0.1072 - accuracy: 0.9592
 1700/11314 [===>..........................] - ETA: 6s - loss: 0.1074 - accuracy: 0.9591
 1800/11314 [===>..........................] - ETA: 6s - loss: 0.1074 - accuracy: 0.9590
 1900/11314 [====>.........................] - ETA: 5s - loss: 0.1073 - accuracy: 0.9589
 2000/11314 [====>.........................] - ETA: 5s - loss: 0.1071 - accuracy: 0.9590
 2100/11314 [====>.........................] - ETA: 5s - loss: 0.1073 - accuracy: 0.9591
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.1077 - accuracy: 0.9589
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.1077 - accuracy: 0.9590
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.1078 - accuracy: 0.9590
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.1080 - accuracy: 0.9589
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.1076 - accuracy: 0.9590
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.1077 - accuracy: 0.9589
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.1077 - accuracy: 0.9588
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.1075 - accuracy: 0.9588
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.1073 - accuracy: 0.9589
 3100/11314 [=======>......................] - ETA: 5s - loss: 0.1075 - accuracy: 0.9589
 3200/11314 [=======>......................] - ETA: 5s - loss: 0.1075 - accuracy: 0.9588
 3300/11314 [=======>......................] - ETA: 5s - loss: 0.1077 - accuracy: 0.9587
 3400/11314 [========>.....................] - ETA: 5s - loss: 0.1077 - accuracy: 0.9587
 3500/11314 [========>.....................] - ETA: 5s - loss: 0.1079 - accuracy: 0.9587
 3600/11314 [========>.....................] - ETA: 5s - loss: 0.1080 - accuracy: 0.9586
 3700/11314 [========>.....................] - ETA: 4s - loss: 0.1081 - accuracy: 0.9585
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.1081 - accuracy: 0.9584
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.1081 - accuracy: 0.9585
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.1079 - accuracy: 0.9585
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.1078 - accuracy: 0.9585
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.1078 - accuracy: 0.9585
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.1076 - accuracy: 0.9585
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.1077 - accuracy: 0.9585
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.1077 - accuracy: 0.9585
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.1079 - accuracy: 0.9585
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.1077 - accuracy: 0.9585
 4800/11314 [===========>..................] - ETA: 4s - loss: 0.1075 - accuracy: 0.9585
 4900/11314 [===========>..................] - ETA: 4s - loss: 0.1077 - accuracy: 0.9585
 5000/11314 [============>.................] - ETA: 4s - loss: 0.1078 - accuracy: 0.9585
 5100/11314 [============>.................] - ETA: 4s - loss: 0.1075 - accuracy: 0.9586
 5200/11314 [============>.................] - ETA: 4s - loss: 0.1075 - accuracy: 0.9587
 5300/11314 [=============>................] - ETA: 4s - loss: 0.1074 - accuracy: 0.9587
 5400/11314 [=============>................] - ETA: 4s - loss: 0.1073 - accuracy: 0.9588
 5500/11314 [=============>................] - ETA: 4s - loss: 0.1074 - accuracy: 0.9587
 5600/11314 [=============>................] - ETA: 4s - loss: 0.1074 - accuracy: 0.9587
 5700/11314 [==============>...............] - ETA: 4s - loss: 0.1073 - accuracy: 0.9588
 5800/11314 [==============>...............] - ETA: 4s - loss: 0.1073 - accuracy: 0.9588
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.1073 - accuracy: 0.9588
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.1073 - accuracy: 0.9587
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.1073 - accuracy: 0.9588
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.1072 - accuracy: 0.9588
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.1070 - accuracy: 0.9589
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.1069 - accuracy: 0.9589
 6500/11314 [================>.............] - ETA: 3s - loss: 0.1069 - accuracy: 0.9590
 6600/11314 [================>.............] - ETA: 3s - loss: 0.1068 - accuracy: 0.9589
 6700/11314 [================>.............] - ETA: 3s - loss: 0.1068 - accuracy: 0.9589
 6800/11314 [=================>............] - ETA: 3s - loss: 0.1068 - accuracy: 0.9589
 6900/11314 [=================>............] - ETA: 3s - loss: 0.1068 - accuracy: 0.9589
 7000/11314 [=================>............] - ETA: 3s - loss: 0.1067 - accuracy: 0.9590
 7100/11314 [=================>............] - ETA: 3s - loss: 0.1069 - accuracy: 0.9589
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.1068 - accuracy: 0.9589
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.1068 - accuracy: 0.9589
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.1068 - accuracy: 0.9589
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.1070 - accuracy: 0.9589
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.1070 - accuracy: 0.9589
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.1068 - accuracy: 0.9590
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.1069 - accuracy: 0.9590
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.1068 - accuracy: 0.9590
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.1068 - accuracy: 0.9590
 8100/11314 [====================>.........] - ETA: 2s - loss: 0.1068 - accuracy: 0.9590
 8200/11314 [====================>.........] - ETA: 2s - loss: 0.1067 - accuracy: 0.9590
 8300/11314 [=====================>........] - ETA: 2s - loss: 0.1068 - accuracy: 0.9590
 8400/11314 [=====================>........] - ETA: 2s - loss: 0.1067 - accuracy: 0.9590
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.1066 - accuracy: 0.9590
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.1066 - accuracy: 0.9589
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.1066 - accuracy: 0.9590
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.1065 - accuracy: 0.9590
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.1065 - accuracy: 0.9591
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.1064 - accuracy: 0.9591
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.1064 - accuracy: 0.9591
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.1065 - accuracy: 0.9591
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.1064 - accuracy: 0.9591
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.1063 - accuracy: 0.9591
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.1064 - accuracy: 0.9591
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.1064 - accuracy: 0.9591
 9700/11314 [========================>.....] - ETA: 1s - loss: 0.1062 - accuracy: 0.9592
 9800/11314 [========================>.....] - ETA: 1s - loss: 0.1062 - accuracy: 0.9592
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.1061 - accuracy: 0.9592
10000/11314 [=========================>....] - ETA: 0s - loss: 0.1062 - accuracy: 0.9592
10100/11314 [=========================>....] - ETA: 0s - loss: 0.1062 - accuracy: 0.9592
10200/11314 [==========================>...] - ETA: 0s - loss: 0.1060 - accuracy: 0.9593
10300/11314 [==========================>...] - ETA: 0s - loss: 0.1061 - accuracy: 0.9593
10400/11314 [==========================>...] - ETA: 0s - loss: 0.1061 - accuracy: 0.9593
10500/11314 [==========================>...] - ETA: 0s - loss: 0.1060 - accuracy: 0.9593
10600/11314 [===========================>..] - ETA: 0s - loss: 0.1058 - accuracy: 0.9594
10700/11314 [===========================>..] - ETA: 0s - loss: 0.1058 - accuracy: 0.9594
10800/11314 [===========================>..] - ETA: 0s - loss: 0.1057 - accuracy: 0.9594
10900/11314 [===========================>..] - ETA: 0s - loss: 0.1057 - accuracy: 0.9594
11000/11314 [============================>.] - ETA: 0s - loss: 0.1057 - accuracy: 0.9594
11100/11314 [============================>.] - ETA: 0s - loss: 0.1057 - accuracy: 0.9594
11200/11314 [============================>.] - ETA: 0s - loss: 0.1058 - accuracy: 0.9594
11300/11314 [============================>.] - ETA: 0s - loss: 0.1057 - accuracy: 0.9594
11314/11314 [==============================] - 9s 821us/step - loss: 0.1056 - accuracy: 0.9594 - val_loss: 0.1395 - val_accuracy: 0.9559
Epoch 12/15

  100/11314 [..............................] - ETA: 7s - loss: 0.0969 - accuracy: 0.9647
  200/11314 [..............................] - ETA: 7s - loss: 0.0957 - accuracy: 0.9653
  300/11314 [..............................] - ETA: 7s - loss: 0.1001 - accuracy: 0.9630
  400/11314 [>.............................] - ETA: 7s - loss: 0.1001 - accuracy: 0.9608
  500/11314 [>.............................] - ETA: 7s - loss: 0.0992 - accuracy: 0.9617
  600/11314 [>.............................] - ETA: 7s - loss: 0.1008 - accuracy: 0.9611
  700/11314 [>.............................] - ETA: 7s - loss: 0.1000 - accuracy: 0.9618
  800/11314 [=>............................] - ETA: 7s - loss: 0.1027 - accuracy: 0.9612
  900/11314 [=>............................] - ETA: 7s - loss: 0.1025 - accuracy: 0.9608
 1000/11314 [=>............................] - ETA: 7s - loss: 0.1017 - accuracy: 0.9613
 1100/11314 [=>............................] - ETA: 7s - loss: 0.1020 - accuracy: 0.9610
 1200/11314 [==>...........................] - ETA: 7s - loss: 0.1020 - accuracy: 0.9609
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.1012 - accuracy: 0.9611
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.1012 - accuracy: 0.9613
 1500/11314 [==>...........................] - ETA: 6s - loss: 0.1006 - accuracy: 0.9616
 1600/11314 [===>..........................] - ETA: 6s - loss: 0.0997 - accuracy: 0.9619
 1700/11314 [===>..........................] - ETA: 6s - loss: 0.0996 - accuracy: 0.9617
 1800/11314 [===>..........................] - ETA: 6s - loss: 0.0993 - accuracy: 0.9618
 1900/11314 [====>.........................] - ETA: 6s - loss: 0.0991 - accuracy: 0.9619
 2000/11314 [====>.........................] - ETA: 6s - loss: 0.0994 - accuracy: 0.9619
 2100/11314 [====>.........................] - ETA: 6s - loss: 0.0994 - accuracy: 0.9619
 2200/11314 [====>.........................] - ETA: 6s - loss: 0.0992 - accuracy: 0.9621
 2300/11314 [=====>........................] - ETA: 6s - loss: 0.0989 - accuracy: 0.9622
 2400/11314 [=====>........................] - ETA: 6s - loss: 0.0986 - accuracy: 0.9623
 2500/11314 [=====>........................] - ETA: 6s - loss: 0.0987 - accuracy: 0.9623
 2600/11314 [=====>........................] - ETA: 6s - loss: 0.0984 - accuracy: 0.9622
 2700/11314 [======>.......................] - ETA: 6s - loss: 0.0982 - accuracy: 0.9622
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.0981 - accuracy: 0.9622
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.0980 - accuracy: 0.9622
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.0978 - accuracy: 0.9622
 3100/11314 [=======>......................] - ETA: 5s - loss: 0.0975 - accuracy: 0.9625
 3200/11314 [=======>......................] - ETA: 5s - loss: 0.0976 - accuracy: 0.9623
 3300/11314 [=======>......................] - ETA: 5s - loss: 0.0977 - accuracy: 0.9622
 3400/11314 [========>.....................] - ETA: 5s - loss: 0.0975 - accuracy: 0.9623
 3500/11314 [========>.....................] - ETA: 5s - loss: 0.0972 - accuracy: 0.9625
 3600/11314 [========>.....................] - ETA: 5s - loss: 0.0970 - accuracy: 0.9625
 3700/11314 [========>.....................] - ETA: 5s - loss: 0.0970 - accuracy: 0.9625
 3800/11314 [=========>....................] - ETA: 5s - loss: 0.0968 - accuracy: 0.9625
 3900/11314 [=========>....................] - ETA: 5s - loss: 0.0967 - accuracy: 0.9626
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.0965 - accuracy: 0.9627
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.0966 - accuracy: 0.9627
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.0965 - accuracy: 0.9627
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.0964 - accuracy: 0.9629
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.0966 - accuracy: 0.9628
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.0965 - accuracy: 0.9629
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.0967 - accuracy: 0.9628
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.0967 - accuracy: 0.9628
 4800/11314 [===========>..................] - ETA: 4s - loss: 0.0966 - accuracy: 0.9628
 4900/11314 [===========>..................] - ETA: 4s - loss: 0.0965 - accuracy: 0.9629
 5000/11314 [============>.................] - ETA: 4s - loss: 0.0963 - accuracy: 0.9629
 5100/11314 [============>.................] - ETA: 4s - loss: 0.0962 - accuracy: 0.9629
 5200/11314 [============>.................] - ETA: 4s - loss: 0.0962 - accuracy: 0.9630
 5300/11314 [=============>................] - ETA: 4s - loss: 0.0962 - accuracy: 0.9629
 5400/11314 [=============>................] - ETA: 3s - loss: 0.0964 - accuracy: 0.9628
 5500/11314 [=============>................] - ETA: 3s - loss: 0.0963 - accuracy: 0.9628
 5600/11314 [=============>................] - ETA: 3s - loss: 0.0962 - accuracy: 0.9629
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.0962 - accuracy: 0.9629
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.0962 - accuracy: 0.9629
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.0961 - accuracy: 0.9629
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.0962 - accuracy: 0.9629
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.0963 - accuracy: 0.9629
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.0962 - accuracy: 0.9630
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.0961 - accuracy: 0.9630
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.0961 - accuracy: 0.9630
 6500/11314 [================>.............] - ETA: 3s - loss: 0.0960 - accuracy: 0.9630
 6600/11314 [================>.............] - ETA: 3s - loss: 0.0960 - accuracy: 0.9630
 6700/11314 [================>.............] - ETA: 3s - loss: 0.0960 - accuracy: 0.9630
 6800/11314 [=================>............] - ETA: 2s - loss: 0.0961 - accuracy: 0.9631
 6900/11314 [=================>............] - ETA: 2s - loss: 0.0961 - accuracy: 0.9630
 7000/11314 [=================>............] - ETA: 2s - loss: 0.0961 - accuracy: 0.9630
 7100/11314 [=================>............] - ETA: 2s - loss: 0.0962 - accuracy: 0.9630
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.0962 - accuracy: 0.9630
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.0962 - accuracy: 0.9629
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.0963 - accuracy: 0.9629
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.0962 - accuracy: 0.9629
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.0964 - accuracy: 0.9629
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.0963 - accuracy: 0.9629
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.0963 - accuracy: 0.9629
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.0962 - accuracy: 0.9629
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.0962 - accuracy: 0.9629
 8100/11314 [====================>.........] - ETA: 2s - loss: 0.0962 - accuracy: 0.9629
 8200/11314 [====================>.........] - ETA: 2s - loss: 0.0962 - accuracy: 0.9629
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.0963 - accuracy: 0.9629
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.0964 - accuracy: 0.9628
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.0964 - accuracy: 0.9628
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.0965 - accuracy: 0.9628
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.0966 - accuracy: 0.9628
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.0966 - accuracy: 0.9628
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.0966 - accuracy: 0.9628
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.0965 - accuracy: 0.9629
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.0964 - accuracy: 0.9629
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.0963 - accuracy: 0.9630
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.0964 - accuracy: 0.9629
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.0964 - accuracy: 0.9630
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.0963 - accuracy: 0.9630
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.0963 - accuracy: 0.9630
 9700/11314 [========================>.....] - ETA: 1s - loss: 0.0962 - accuracy: 0.9630
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.0963 - accuracy: 0.9630
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.0963 - accuracy: 0.9630
10000/11314 [=========================>....] - ETA: 0s - loss: 0.0963 - accuracy: 0.9630
10100/11314 [=========================>....] - ETA: 0s - loss: 0.0964 - accuracy: 0.9630
10200/11314 [==========================>...] - ETA: 0s - loss: 0.0963 - accuracy: 0.9630
10300/11314 [==========================>...] - ETA: 0s - loss: 0.0963 - accuracy: 0.9630
10400/11314 [==========================>...] - ETA: 0s - loss: 0.0963 - accuracy: 0.9630
10500/11314 [==========================>...] - ETA: 0s - loss: 0.0962 - accuracy: 0.9630
10600/11314 [===========================>..] - ETA: 0s - loss: 0.0963 - accuracy: 0.9630
10700/11314 [===========================>..] - ETA: 0s - loss: 0.0962 - accuracy: 0.9630
10800/11314 [===========================>..] - ETA: 0s - loss: 0.0962 - accuracy: 0.9630
10900/11314 [===========================>..] - ETA: 0s - loss: 0.0962 - accuracy: 0.9630
11000/11314 [============================>.] - ETA: 0s - loss: 0.0963 - accuracy: 0.9630
11100/11314 [============================>.] - ETA: 0s - loss: 0.0963 - accuracy: 0.9630
11200/11314 [============================>.] - ETA: 0s - loss: 0.0963 - accuracy: 0.9630
11300/11314 [============================>.] - ETA: 0s - loss: 0.0963 - accuracy: 0.9630
11314/11314 [==============================] - 9s 786us/step - loss: 0.0963 - accuracy: 0.9630 - val_loss: 0.1362 - val_accuracy: 0.9574
Epoch 13/15

  100/11314 [..............................] - ETA: 8s - loss: 0.0960 - accuracy: 0.9668
  200/11314 [..............................] - ETA: 8s - loss: 0.0890 - accuracy: 0.9674
  300/11314 [..............................] - ETA: 8s - loss: 0.0946 - accuracy: 0.9656
  400/11314 [>.............................] - ETA: 8s - loss: 0.0931 - accuracy: 0.9655
  500/11314 [>.............................] - ETA: 7s - loss: 0.0931 - accuracy: 0.9654
  600/11314 [>.............................] - ETA: 7s - loss: 0.0916 - accuracy: 0.9652
  700/11314 [>.............................] - ETA: 7s - loss: 0.0904 - accuracy: 0.9658
  800/11314 [=>............................] - ETA: 7s - loss: 0.0898 - accuracy: 0.9660
  900/11314 [=>............................] - ETA: 7s - loss: 0.0894 - accuracy: 0.9662
 1000/11314 [=>............................] - ETA: 7s - loss: 0.0900 - accuracy: 0.9659
 1100/11314 [=>............................] - ETA: 7s - loss: 0.0904 - accuracy: 0.9656
 1200/11314 [==>...........................] - ETA: 7s - loss: 0.0903 - accuracy: 0.9653
 1300/11314 [==>...........................] - ETA: 7s - loss: 0.0902 - accuracy: 0.9654
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.0901 - accuracy: 0.9652
 1500/11314 [==>...........................] - ETA: 6s - loss: 0.0904 - accuracy: 0.9649
 1600/11314 [===>..........................] - ETA: 6s - loss: 0.0898 - accuracy: 0.9651
 1700/11314 [===>..........................] - ETA: 6s - loss: 0.0894 - accuracy: 0.9652
 1800/11314 [===>..........................] - ETA: 6s - loss: 0.0896 - accuracy: 0.9650
 1900/11314 [====>.........................] - ETA: 6s - loss: 0.0896 - accuracy: 0.9651
 2000/11314 [====>.........................] - ETA: 6s - loss: 0.0898 - accuracy: 0.9650
 2100/11314 [====>.........................] - ETA: 6s - loss: 0.0899 - accuracy: 0.9651
 2200/11314 [====>.........................] - ETA: 6s - loss: 0.0896 - accuracy: 0.9653
 2300/11314 [=====>........................] - ETA: 6s - loss: 0.0896 - accuracy: 0.9652
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.0894 - accuracy: 0.9653
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.0896 - accuracy: 0.9654
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.0896 - accuracy: 0.9653
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.0891 - accuracy: 0.9654
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.0891 - accuracy: 0.9655
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.0892 - accuracy: 0.9653
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.0891 - accuracy: 0.9654
 3100/11314 [=======>......................] - ETA: 5s - loss: 0.0887 - accuracy: 0.9656
 3200/11314 [=======>......................] - ETA: 5s - loss: 0.0885 - accuracy: 0.9656
 3300/11314 [=======>......................] - ETA: 5s - loss: 0.0886 - accuracy: 0.9655
 3400/11314 [========>.....................] - ETA: 5s - loss: 0.0887 - accuracy: 0.9654
 3500/11314 [========>.....................] - ETA: 5s - loss: 0.0888 - accuracy: 0.9655
 3600/11314 [========>.....................] - ETA: 5s - loss: 0.0889 - accuracy: 0.9655
 3700/11314 [========>.....................] - ETA: 5s - loss: 0.0890 - accuracy: 0.9654
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.0888 - accuracy: 0.9656
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.0887 - accuracy: 0.9657
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.0887 - accuracy: 0.9658
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.0887 - accuracy: 0.9659
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.0886 - accuracy: 0.9658
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.0886 - accuracy: 0.9658
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.0886 - accuracy: 0.9659
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.0888 - accuracy: 0.9658
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.0886 - accuracy: 0.9659
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.0884 - accuracy: 0.9660
 4800/11314 [===========>..................] - ETA: 4s - loss: 0.0883 - accuracy: 0.9660
 4900/11314 [===========>..................] - ETA: 4s - loss: 0.0884 - accuracy: 0.9660
 5000/11314 [============>.................] - ETA: 4s - loss: 0.0882 - accuracy: 0.9661
 5100/11314 [============>.................] - ETA: 4s - loss: 0.0879 - accuracy: 0.9662
 5200/11314 [============>.................] - ETA: 3s - loss: 0.0879 - accuracy: 0.9661
 5300/11314 [=============>................] - ETA: 3s - loss: 0.0879 - accuracy: 0.9662
 5400/11314 [=============>................] - ETA: 3s - loss: 0.0877 - accuracy: 0.9662
 5500/11314 [=============>................] - ETA: 3s - loss: 0.0878 - accuracy: 0.9662
 5600/11314 [=============>................] - ETA: 3s - loss: 0.0877 - accuracy: 0.9662
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.0878 - accuracy: 0.9661
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.0878 - accuracy: 0.9661
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.0878 - accuracy: 0.9661
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.0879 - accuracy: 0.9660
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.0879 - accuracy: 0.9660
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.0878 - accuracy: 0.9661
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.0877 - accuracy: 0.9661
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.0876 - accuracy: 0.9662
 6500/11314 [================>.............] - ETA: 3s - loss: 0.0876 - accuracy: 0.9661
 6600/11314 [================>.............] - ETA: 3s - loss: 0.0877 - accuracy: 0.9661
 6700/11314 [================>.............] - ETA: 2s - loss: 0.0877 - accuracy: 0.9662
 6800/11314 [=================>............] - ETA: 2s - loss: 0.0876 - accuracy: 0.9662
 6900/11314 [=================>............] - ETA: 2s - loss: 0.0876 - accuracy: 0.9662
 7000/11314 [=================>............] - ETA: 2s - loss: 0.0874 - accuracy: 0.9662
 7100/11314 [=================>............] - ETA: 2s - loss: 0.0875 - accuracy: 0.9662
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.0874 - accuracy: 0.9662
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.0874 - accuracy: 0.9663
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.0874 - accuracy: 0.9662
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.0873 - accuracy: 0.9663
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.0873 - accuracy: 0.9663
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.0874 - accuracy: 0.9663
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.0874 - accuracy: 0.9663
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.0874 - accuracy: 0.9662
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.0873 - accuracy: 0.9663
 8100/11314 [====================>.........] - ETA: 2s - loss: 0.0873 - accuracy: 0.9663
 8200/11314 [====================>.........] - ETA: 2s - loss: 0.0873 - accuracy: 0.9663
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.0874 - accuracy: 0.9663
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.0873 - accuracy: 0.9663
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.0874 - accuracy: 0.9663
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.0874 - accuracy: 0.9664
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.0874 - accuracy: 0.9663
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.0874 - accuracy: 0.9664
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.0874 - accuracy: 0.9663
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.0875 - accuracy: 0.9663
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.0876 - accuracy: 0.9663
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.0876 - accuracy: 0.9664
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.0877 - accuracy: 0.9663
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.0878 - accuracy: 0.9663
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.0878 - accuracy: 0.9663
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.0879 - accuracy: 0.9662
 9700/11314 [========================>.....] - ETA: 1s - loss: 0.0879 - accuracy: 0.9663
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.0879 - accuracy: 0.9662
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.0879 - accuracy: 0.9662
10000/11314 [=========================>....] - ETA: 0s - loss: 0.0879 - accuracy: 0.9662
10100/11314 [=========================>....] - ETA: 0s - loss: 0.0879 - accuracy: 0.9662
10200/11314 [==========================>...] - ETA: 0s - loss: 0.0879 - accuracy: 0.9662
10300/11314 [==========================>...] - ETA: 0s - loss: 0.0878 - accuracy: 0.9663
10400/11314 [==========================>...] - ETA: 0s - loss: 0.0878 - accuracy: 0.9663
10500/11314 [==========================>...] - ETA: 0s - loss: 0.0878 - accuracy: 0.9663
10600/11314 [===========================>..] - ETA: 0s - loss: 0.0878 - accuracy: 0.9664
10700/11314 [===========================>..] - ETA: 0s - loss: 0.0879 - accuracy: 0.9663
10800/11314 [===========================>..] - ETA: 0s - loss: 0.0879 - accuracy: 0.9664
10900/11314 [===========================>..] - ETA: 0s - loss: 0.0879 - accuracy: 0.9664
11000/11314 [============================>.] - ETA: 0s - loss: 0.0878 - accuracy: 0.9664
11100/11314 [============================>.] - ETA: 0s - loss: 0.0878 - accuracy: 0.9664
11200/11314 [============================>.] - ETA: 0s - loss: 0.0877 - accuracy: 0.9665
11300/11314 [============================>.] - ETA: 0s - loss: 0.0877 - accuracy: 0.9665
11314/11314 [==============================] - 9s 758us/step - loss: 0.0878 - accuracy: 0.9665 - val_loss: 0.1359 - val_accuracy: 0.9587
Epoch 14/15

  100/11314 [..............................] - ETA: 7s - loss: 0.0774 - accuracy: 0.9663
  200/11314 [..............................] - ETA: 6s - loss: 0.0788 - accuracy: 0.9674
  300/11314 [..............................] - ETA: 6s - loss: 0.0794 - accuracy: 0.9681
  400/11314 [>.............................] - ETA: 7s - loss: 0.0808 - accuracy: 0.9678
  500/11314 [>.............................] - ETA: 7s - loss: 0.0808 - accuracy: 0.9675
  600/11314 [>.............................] - ETA: 7s - loss: 0.0805 - accuracy: 0.9682
  700/11314 [>.............................] - ETA: 7s - loss: 0.0806 - accuracy: 0.9684
  800/11314 [=>............................] - ETA: 6s - loss: 0.0802 - accuracy: 0.9688
  900/11314 [=>............................] - ETA: 6s - loss: 0.0807 - accuracy: 0.9687
 1000/11314 [=>............................] - ETA: 6s - loss: 0.0812 - accuracy: 0.9684
 1100/11314 [=>............................] - ETA: 6s - loss: 0.0813 - accuracy: 0.9686
 1200/11314 [==>...........................] - ETA: 6s - loss: 0.0814 - accuracy: 0.9686
 1300/11314 [==>...........................] - ETA: 6s - loss: 0.0805 - accuracy: 0.9690
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.0810 - accuracy: 0.9689
 1500/11314 [==>...........................] - ETA: 6s - loss: 0.0817 - accuracy: 0.9688
 1600/11314 [===>..........................] - ETA: 6s - loss: 0.0820 - accuracy: 0.9687
 1700/11314 [===>..........................] - ETA: 6s - loss: 0.0820 - accuracy: 0.9686
 1800/11314 [===>..........................] - ETA: 6s - loss: 0.0817 - accuracy: 0.9687
 1900/11314 [====>.........................] - ETA: 6s - loss: 0.0815 - accuracy: 0.9689
 2000/11314 [====>.........................] - ETA: 6s - loss: 0.0813 - accuracy: 0.9689
 2100/11314 [====>.........................] - ETA: 5s - loss: 0.0812 - accuracy: 0.9690
 2200/11314 [====>.........................] - ETA: 5s - loss: 0.0812 - accuracy: 0.9688
 2300/11314 [=====>........................] - ETA: 5s - loss: 0.0811 - accuracy: 0.9690
 2400/11314 [=====>........................] - ETA: 5s - loss: 0.0809 - accuracy: 0.9692
 2500/11314 [=====>........................] - ETA: 5s - loss: 0.0807 - accuracy: 0.9693
 2600/11314 [=====>........................] - ETA: 5s - loss: 0.0807 - accuracy: 0.9695
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.0806 - accuracy: 0.9694
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.0804 - accuracy: 0.9694
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.0802 - accuracy: 0.9694
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.0801 - accuracy: 0.9694
 3100/11314 [=======>......................] - ETA: 5s - loss: 0.0801 - accuracy: 0.9695
 3200/11314 [=======>......................] - ETA: 5s - loss: 0.0802 - accuracy: 0.9696
 3300/11314 [=======>......................] - ETA: 5s - loss: 0.0805 - accuracy: 0.9694
 3400/11314 [========>.....................] - ETA: 5s - loss: 0.0809 - accuracy: 0.9692
 3500/11314 [========>.....................] - ETA: 5s - loss: 0.0809 - accuracy: 0.9691
 3600/11314 [========>.....................] - ETA: 5s - loss: 0.0808 - accuracy: 0.9691
 3700/11314 [========>.....................] - ETA: 5s - loss: 0.0809 - accuracy: 0.9690
 3800/11314 [=========>....................] - ETA: 4s - loss: 0.0812 - accuracy: 0.9688
 3900/11314 [=========>....................] - ETA: 4s - loss: 0.0812 - accuracy: 0.9689
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.0811 - accuracy: 0.9690
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.0814 - accuracy: 0.9689
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.0813 - accuracy: 0.9689
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.0814 - accuracy: 0.9689
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.0811 - accuracy: 0.9690
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.0811 - accuracy: 0.9692
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.0808 - accuracy: 0.9692
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.0810 - accuracy: 0.9691
 4800/11314 [===========>..................] - ETA: 4s - loss: 0.0808 - accuracy: 0.9692
 4900/11314 [===========>..................] - ETA: 4s - loss: 0.0808 - accuracy: 0.9692
 5000/11314 [============>.................] - ETA: 4s - loss: 0.0807 - accuracy: 0.9693
 5100/11314 [============>.................] - ETA: 4s - loss: 0.0807 - accuracy: 0.9693
 5200/11314 [============>.................] - ETA: 4s - loss: 0.0805 - accuracy: 0.9694
 5300/11314 [=============>................] - ETA: 3s - loss: 0.0804 - accuracy: 0.9694
 5400/11314 [=============>................] - ETA: 3s - loss: 0.0804 - accuracy: 0.9694
 5500/11314 [=============>................] - ETA: 3s - loss: 0.0803 - accuracy: 0.9694
 5600/11314 [=============>................] - ETA: 3s - loss: 0.0802 - accuracy: 0.9695
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.0805 - accuracy: 0.9694
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.0803 - accuracy: 0.9695
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.0804 - accuracy: 0.9695
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.0804 - accuracy: 0.9694
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.0804 - accuracy: 0.9694
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.0804 - accuracy: 0.9694
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.0803 - accuracy: 0.9694
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.0803 - accuracy: 0.9694
 6500/11314 [================>.............] - ETA: 3s - loss: 0.0804 - accuracy: 0.9694
 6600/11314 [================>.............] - ETA: 3s - loss: 0.0802 - accuracy: 0.9695
 6700/11314 [================>.............] - ETA: 3s - loss: 0.0802 - accuracy: 0.9696
 6800/11314 [=================>............] - ETA: 2s - loss: 0.0801 - accuracy: 0.9697
 6900/11314 [=================>............] - ETA: 2s - loss: 0.0803 - accuracy: 0.9696
 7000/11314 [=================>............] - ETA: 2s - loss: 0.0803 - accuracy: 0.9696
 7100/11314 [=================>............] - ETA: 2s - loss: 0.0803 - accuracy: 0.9696
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.0805 - accuracy: 0.9696
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.0805 - accuracy: 0.9695
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.0803 - accuracy: 0.9696
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.0802 - accuracy: 0.9696
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.0800 - accuracy: 0.9697
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.0802 - accuracy: 0.9696
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.0801 - accuracy: 0.9696
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.0801 - accuracy: 0.9696
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.0802 - accuracy: 0.9696
 8100/11314 [====================>.........] - ETA: 2s - loss: 0.0802 - accuracy: 0.9696
 8200/11314 [====================>.........] - ETA: 2s - loss: 0.0802 - accuracy: 0.9696
 8300/11314 [=====================>........] - ETA: 1s - loss: 0.0802 - accuracy: 0.9697
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.0801 - accuracy: 0.9697
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.0801 - accuracy: 0.9697
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.0801 - accuracy: 0.9698
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.0800 - accuracy: 0.9698
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.0801 - accuracy: 0.9698
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.0800 - accuracy: 0.9698
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.0800 - accuracy: 0.9698
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.0800 - accuracy: 0.9698
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.0800 - accuracy: 0.9698
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.0800 - accuracy: 0.9698
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.0800 - accuracy: 0.9697
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.0799 - accuracy: 0.9698
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.0800 - accuracy: 0.9697
 9700/11314 [========================>.....] - ETA: 1s - loss: 0.0800 - accuracy: 0.9697
 9800/11314 [========================>.....] - ETA: 0s - loss: 0.0800 - accuracy: 0.9697
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.0800 - accuracy: 0.9697
10000/11314 [=========================>....] - ETA: 0s - loss: 0.0799 - accuracy: 0.9697
10100/11314 [=========================>....] - ETA: 0s - loss: 0.0800 - accuracy: 0.9697
10200/11314 [==========================>...] - ETA: 0s - loss: 0.0800 - accuracy: 0.9696
10300/11314 [==========================>...] - ETA: 0s - loss: 0.0800 - accuracy: 0.9697
10400/11314 [==========================>...] - ETA: 0s - loss: 0.0800 - accuracy: 0.9697
10500/11314 [==========================>...] - ETA: 0s - loss: 0.0800 - accuracy: 0.9697
10600/11314 [===========================>..] - ETA: 0s - loss: 0.0801 - accuracy: 0.9697
10700/11314 [===========================>..] - ETA: 0s - loss: 0.0800 - accuracy: 0.9697
10800/11314 [===========================>..] - ETA: 0s - loss: 0.0800 - accuracy: 0.9698
10900/11314 [===========================>..] - ETA: 0s - loss: 0.0801 - accuracy: 0.9697
11000/11314 [============================>.] - ETA: 0s - loss: 0.0801 - accuracy: 0.9697
11100/11314 [============================>.] - ETA: 0s - loss: 0.0801 - accuracy: 0.9697
11200/11314 [============================>.] - ETA: 0s - loss: 0.0801 - accuracy: 0.9697
11300/11314 [============================>.] - ETA: 0s - loss: 0.0800 - accuracy: 0.9698
11314/11314 [==============================] - 9s 771us/step - loss: 0.0800 - accuracy: 0.9698 - val_loss: 0.1356 - val_accuracy: 0.9599
Epoch 15/15

  100/11314 [..............................] - ETA: 8s - loss: 0.0732 - accuracy: 0.9732
  200/11314 [..............................] - ETA: 8s - loss: 0.0770 - accuracy: 0.9708
  300/11314 [..............................] - ETA: 8s - loss: 0.0770 - accuracy: 0.9704
  400/11314 [>.............................] - ETA: 8s - loss: 0.0757 - accuracy: 0.9721
  500/11314 [>.............................] - ETA: 8s - loss: 0.0750 - accuracy: 0.9714
  600/11314 [>.............................] - ETA: 7s - loss: 0.0755 - accuracy: 0.9710
  700/11314 [>.............................] - ETA: 7s - loss: 0.0746 - accuracy: 0.9710
  800/11314 [=>............................] - ETA: 7s - loss: 0.0748 - accuracy: 0.9706
  900/11314 [=>............................] - ETA: 7s - loss: 0.0751 - accuracy: 0.9705
 1000/11314 [=>............................] - ETA: 7s - loss: 0.0748 - accuracy: 0.9709
 1100/11314 [=>............................] - ETA: 7s - loss: 0.0759 - accuracy: 0.9704
 1200/11314 [==>...........................] - ETA: 7s - loss: 0.0763 - accuracy: 0.9703
 1300/11314 [==>...........................] - ETA: 7s - loss: 0.0761 - accuracy: 0.9706
 1400/11314 [==>...........................] - ETA: 6s - loss: 0.0762 - accuracy: 0.9708
 1500/11314 [==>...........................] - ETA: 6s - loss: 0.0756 - accuracy: 0.9711
 1600/11314 [===>..........................] - ETA: 6s - loss: 0.0744 - accuracy: 0.9717
 1700/11314 [===>..........................] - ETA: 6s - loss: 0.0743 - accuracy: 0.9719
 1800/11314 [===>..........................] - ETA: 6s - loss: 0.0739 - accuracy: 0.9720
 1900/11314 [====>.........................] - ETA: 6s - loss: 0.0737 - accuracy: 0.9723
 2000/11314 [====>.........................] - ETA: 6s - loss: 0.0738 - accuracy: 0.9722
 2100/11314 [====>.........................] - ETA: 6s - loss: 0.0737 - accuracy: 0.9720
 2200/11314 [====>.........................] - ETA: 6s - loss: 0.0733 - accuracy: 0.9721
 2300/11314 [=====>........................] - ETA: 6s - loss: 0.0737 - accuracy: 0.9720
 2400/11314 [=====>........................] - ETA: 6s - loss: 0.0734 - accuracy: 0.9721
 2500/11314 [=====>........................] - ETA: 6s - loss: 0.0734 - accuracy: 0.9720
 2600/11314 [=====>........................] - ETA: 6s - loss: 0.0733 - accuracy: 0.9721
 2700/11314 [======>.......................] - ETA: 5s - loss: 0.0731 - accuracy: 0.9721
 2800/11314 [======>.......................] - ETA: 5s - loss: 0.0729 - accuracy: 0.9722
 2900/11314 [======>.......................] - ETA: 5s - loss: 0.0731 - accuracy: 0.9720
 3000/11314 [======>.......................] - ETA: 5s - loss: 0.0731 - accuracy: 0.9721
 3100/11314 [=======>......................] - ETA: 5s - loss: 0.0727 - accuracy: 0.9723
 3200/11314 [=======>......................] - ETA: 5s - loss: 0.0727 - accuracy: 0.9723
 3300/11314 [=======>......................] - ETA: 5s - loss: 0.0729 - accuracy: 0.9722
 3400/11314 [========>.....................] - ETA: 5s - loss: 0.0732 - accuracy: 0.9721
 3500/11314 [========>.....................] - ETA: 5s - loss: 0.0733 - accuracy: 0.9721
 3600/11314 [========>.....................] - ETA: 5s - loss: 0.0734 - accuracy: 0.9720
 3700/11314 [========>.....................] - ETA: 5s - loss: 0.0733 - accuracy: 0.9721
 3800/11314 [=========>....................] - ETA: 5s - loss: 0.0733 - accuracy: 0.9722
 3900/11314 [=========>....................] - ETA: 5s - loss: 0.0735 - accuracy: 0.9722
 4000/11314 [=========>....................] - ETA: 4s - loss: 0.0735 - accuracy: 0.9723
 4100/11314 [=========>....................] - ETA: 4s - loss: 0.0733 - accuracy: 0.9725
 4200/11314 [==========>...................] - ETA: 4s - loss: 0.0733 - accuracy: 0.9725
 4300/11314 [==========>...................] - ETA: 4s - loss: 0.0733 - accuracy: 0.9725
 4400/11314 [==========>...................] - ETA: 4s - loss: 0.0733 - accuracy: 0.9726
 4500/11314 [==========>...................] - ETA: 4s - loss: 0.0735 - accuracy: 0.9726
 4600/11314 [===========>..................] - ETA: 4s - loss: 0.0735 - accuracy: 0.9726
 4700/11314 [===========>..................] - ETA: 4s - loss: 0.0735 - accuracy: 0.9726
 4800/11314 [===========>..................] - ETA: 4s - loss: 0.0736 - accuracy: 0.9726
 4900/11314 [===========>..................] - ETA: 4s - loss: 0.0736 - accuracy: 0.9726
 5000/11314 [============>.................] - ETA: 4s - loss: 0.0735 - accuracy: 0.9727
 5100/11314 [============>.................] - ETA: 4s - loss: 0.0734 - accuracy: 0.9727
 5200/11314 [============>.................] - ETA: 4s - loss: 0.0735 - accuracy: 0.9727
 5300/11314 [=============>................] - ETA: 4s - loss: 0.0735 - accuracy: 0.9727
 5400/11314 [=============>................] - ETA: 4s - loss: 0.0735 - accuracy: 0.9728
 5500/11314 [=============>................] - ETA: 3s - loss: 0.0734 - accuracy: 0.9727
 5600/11314 [=============>................] - ETA: 3s - loss: 0.0734 - accuracy: 0.9727
 5700/11314 [==============>...............] - ETA: 3s - loss: 0.0735 - accuracy: 0.9726
 5800/11314 [==============>...............] - ETA: 3s - loss: 0.0734 - accuracy: 0.9727
 5900/11314 [==============>...............] - ETA: 3s - loss: 0.0734 - accuracy: 0.9728
 6000/11314 [==============>...............] - ETA: 3s - loss: 0.0734 - accuracy: 0.9728
 6100/11314 [===============>..............] - ETA: 3s - loss: 0.0734 - accuracy: 0.9727
 6200/11314 [===============>..............] - ETA: 3s - loss: 0.0734 - accuracy: 0.9728
 6300/11314 [===============>..............] - ETA: 3s - loss: 0.0734 - accuracy: 0.9728
 6400/11314 [===============>..............] - ETA: 3s - loss: 0.0734 - accuracy: 0.9728
 6500/11314 [================>.............] - ETA: 3s - loss: 0.0735 - accuracy: 0.9728
 6600/11314 [================>.............] - ETA: 3s - loss: 0.0736 - accuracy: 0.9728
 6700/11314 [================>.............] - ETA: 3s - loss: 0.0735 - accuracy: 0.9729
 6800/11314 [=================>............] - ETA: 3s - loss: 0.0735 - accuracy: 0.9728
 6900/11314 [=================>............] - ETA: 2s - loss: 0.0735 - accuracy: 0.9728
 7000/11314 [=================>............] - ETA: 2s - loss: 0.0735 - accuracy: 0.9728
 7100/11314 [=================>............] - ETA: 2s - loss: 0.0735 - accuracy: 0.9728
 7200/11314 [==================>...........] - ETA: 2s - loss: 0.0734 - accuracy: 0.9729
 7300/11314 [==================>...........] - ETA: 2s - loss: 0.0733 - accuracy: 0.9729
 7400/11314 [==================>...........] - ETA: 2s - loss: 0.0735 - accuracy: 0.9729
 7500/11314 [==================>...........] - ETA: 2s - loss: 0.0733 - accuracy: 0.9730
 7600/11314 [===================>..........] - ETA: 2s - loss: 0.0734 - accuracy: 0.9728
 7700/11314 [===================>..........] - ETA: 2s - loss: 0.0734 - accuracy: 0.9729
 7800/11314 [===================>..........] - ETA: 2s - loss: 0.0734 - accuracy: 0.9729
 7900/11314 [===================>..........] - ETA: 2s - loss: 0.0734 - accuracy: 0.9729
 8000/11314 [====================>.........] - ETA: 2s - loss: 0.0735 - accuracy: 0.9729
 8100/11314 [====================>.........] - ETA: 2s - loss: 0.0735 - accuracy: 0.9729
 8200/11314 [====================>.........] - ETA: 2s - loss: 0.0736 - accuracy: 0.9729
 8300/11314 [=====================>........] - ETA: 2s - loss: 0.0737 - accuracy: 0.9729
 8400/11314 [=====================>........] - ETA: 1s - loss: 0.0736 - accuracy: 0.9729
 8500/11314 [=====================>........] - ETA: 1s - loss: 0.0736 - accuracy: 0.9729
 8600/11314 [=====================>........] - ETA: 1s - loss: 0.0735 - accuracy: 0.9729
 8700/11314 [======================>.......] - ETA: 1s - loss: 0.0734 - accuracy: 0.9729
 8800/11314 [======================>.......] - ETA: 1s - loss: 0.0734 - accuracy: 0.9728
 8900/11314 [======================>.......] - ETA: 1s - loss: 0.0734 - accuracy: 0.9728
 9000/11314 [======================>.......] - ETA: 1s - loss: 0.0734 - accuracy: 0.9728
 9100/11314 [=======================>......] - ETA: 1s - loss: 0.0734 - accuracy: 0.9728
 9200/11314 [=======================>......] - ETA: 1s - loss: 0.0734 - accuracy: 0.9728
 9300/11314 [=======================>......] - ETA: 1s - loss: 0.0733 - accuracy: 0.9728
 9400/11314 [=======================>......] - ETA: 1s - loss: 0.0732 - accuracy: 0.9728
 9500/11314 [========================>.....] - ETA: 1s - loss: 0.0732 - accuracy: 0.9729
 9600/11314 [========================>.....] - ETA: 1s - loss: 0.0732 - accuracy: 0.9728
 9700/11314 [========================>.....] - ETA: 1s - loss: 0.0732 - accuracy: 0.9728
 9800/11314 [========================>.....] - ETA: 1s - loss: 0.0732 - accuracy: 0.9728
 9900/11314 [=========================>....] - ETA: 0s - loss: 0.0733 - accuracy: 0.9728
10000/11314 [=========================>....] - ETA: 0s - loss: 0.0733 - accuracy: 0.9728
10100/11314 [=========================>....] - ETA: 0s - loss: 0.0733 - accuracy: 0.9728
10200/11314 [==========================>...] - ETA: 0s - loss: 0.0733 - accuracy: 0.9728
10300/11314 [==========================>...] - ETA: 0s - loss: 0.0733 - accuracy: 0.9728
10400/11314 [==========================>...] - ETA: 0s - loss: 0.0733 - accuracy: 0.9728
10500/11314 [==========================>...] - ETA: 0s - loss: 0.0732 - accuracy: 0.9729
10600/11314 [===========================>..] - ETA: 0s - loss: 0.0732 - accuracy: 0.9729
10700/11314 [===========================>..] - ETA: 0s - loss: 0.0733 - accuracy: 0.9728
10800/11314 [===========================>..] - ETA: 0s - loss: 0.0732 - accuracy: 0.9729
10900/11314 [===========================>..] - ETA: 0s - loss: 0.0733 - accuracy: 0.9728
11000/11314 [============================>.] - ETA: 0s - loss: 0.0732 - accuracy: 0.9729
11100/11314 [============================>.] - ETA: 0s - loss: 0.0731 - accuracy: 0.9729
11200/11314 [============================>.] - ETA: 0s - loss: 0.0731 - accuracy: 0.9729
11300/11314 [============================>.] - ETA: 0s - loss: 0.0730 - accuracy: 0.9730
11314/11314 [==============================] - 9s 802us/step - loss: 0.0730 - accuracy: 0.9730 - val_loss: 0.1363 - val_accuracy: 0.9608
	=====> Test the model: model.predict()
	Dataset: TWENTY_NEWS_GROUPS
	Algorithm: Deep Learning using Keras 2 (KERAS_DL2)
	Training loss: 0.0595
	Training accuracy score: 97.89%
	Test loss: 0.1363
	Test accuracy score: 96.08%
	Training time: 133.2070
	Test time: 1.9305


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
	It took 25.5984308719635 seconds


Applying NLTK feature extraction: X_test
	==> Replace unused UNICODE characters. 
	==> Text in lower case. 
	==> Apply Lemmatize using WordNet's built-in morphy function. Returns the input word unchanged if it cannot be found in WordNet 
	==> Remove English stop words
	Done: NLTK feature extraction (replace unused UNICODE characters, text in lower case, apply Lemmatize using WordNet, remove English stop words)  finished!
	It took 24.894062280654907 seconds

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

  100/25000 [..............................] - ETA: 2:14 - loss: 0.6929 - accuracy: 0.5100
  200/25000 [..............................] - ETA: 1:15 - loss: 0.6932 - accuracy: 0.4700
  300/25000 [..............................] - ETA: 54s - loss: 0.6929 - accuracy: 0.5167 
  400/25000 [..............................] - ETA: 44s - loss: 0.6930 - accuracy: 0.5050
  500/25000 [..............................] - ETA: 39s - loss: 0.6930 - accuracy: 0.5000
  600/25000 [..............................] - ETA: 35s - loss: 0.6928 - accuracy: 0.5083
  700/25000 [..............................] - ETA: 32s - loss: 0.6922 - accuracy: 0.5257
  800/25000 [..............................] - ETA: 29s - loss: 0.6922 - accuracy: 0.5213
  900/25000 [>.............................] - ETA: 28s - loss: 0.6920 - accuracy: 0.5244
 1000/25000 [>.............................] - ETA: 26s - loss: 0.6920 - accuracy: 0.5200
 1100/25000 [>.............................] - ETA: 25s - loss: 0.6922 - accuracy: 0.5182
 1200/25000 [>.............................] - ETA: 24s - loss: 0.6923 - accuracy: 0.5150
 1300/25000 [>.............................] - ETA: 24s - loss: 0.6919 - accuracy: 0.5200
 1400/25000 [>.............................] - ETA: 23s - loss: 0.6916 - accuracy: 0.5243
 1500/25000 [>.............................] - ETA: 22s - loss: 0.6915 - accuracy: 0.5233
 1600/25000 [>.............................] - ETA: 22s - loss: 0.6913 - accuracy: 0.5231
 1700/25000 [=>............................] - ETA: 21s - loss: 0.6915 - accuracy: 0.5212
 1800/25000 [=>............................] - ETA: 21s - loss: 0.6915 - accuracy: 0.5189
 1900/25000 [=>............................] - ETA: 20s - loss: 0.6914 - accuracy: 0.5179
 2000/25000 [=>............................] - ETA: 20s - loss: 0.6913 - accuracy: 0.5165
 2100/25000 [=>............................] - ETA: 20s - loss: 0.6912 - accuracy: 0.5171
 2200/25000 [=>............................] - ETA: 19s - loss: 0.6906 - accuracy: 0.5227
 2300/25000 [=>............................] - ETA: 19s - loss: 0.6903 - accuracy: 0.5235
 2400/25000 [=>............................] - ETA: 19s - loss: 0.6904 - accuracy: 0.5217
 2500/25000 [==>...........................] - ETA: 19s - loss: 0.6905 - accuracy: 0.5200
 2600/25000 [==>...........................] - ETA: 18s - loss: 0.6903 - accuracy: 0.5208
 2700/25000 [==>...........................] - ETA: 18s - loss: 0.6900 - accuracy: 0.5237
 2800/25000 [==>...........................] - ETA: 18s - loss: 0.6901 - accuracy: 0.5232
 2900/25000 [==>...........................] - ETA: 18s - loss: 0.6900 - accuracy: 0.5228
 3000/25000 [==>...........................] - ETA: 18s - loss: 0.6899 - accuracy: 0.5220
 3100/25000 [==>...........................] - ETA: 17s - loss: 0.6898 - accuracy: 0.5223
 3200/25000 [==>...........................] - ETA: 17s - loss: 0.6897 - accuracy: 0.5219
 3300/25000 [==>...........................] - ETA: 17s - loss: 0.6896 - accuracy: 0.5239
 3400/25000 [===>..........................] - ETA: 17s - loss: 0.6894 - accuracy: 0.5259
 3500/25000 [===>..........................] - ETA: 17s - loss: 0.6893 - accuracy: 0.5263
 3600/25000 [===>..........................] - ETA: 16s - loss: 0.6890 - accuracy: 0.5325
 3700/25000 [===>..........................] - ETA: 16s - loss: 0.6887 - accuracy: 0.5368
 3800/25000 [===>..........................] - ETA: 16s - loss: 0.6885 - accuracy: 0.5416
 3900/25000 [===>..........................] - ETA: 16s - loss: 0.6883 - accuracy: 0.5438
 4000/25000 [===>..........................] - ETA: 16s - loss: 0.6881 - accuracy: 0.5472
 4100/25000 [===>..........................] - ETA: 16s - loss: 0.6879 - accuracy: 0.5485
 4200/25000 [====>.........................] - ETA: 16s - loss: 0.6875 - accuracy: 0.5512
 4300/25000 [====>.........................] - ETA: 15s - loss: 0.6871 - accuracy: 0.5547
 4400/25000 [====>.........................] - ETA: 15s - loss: 0.6867 - accuracy: 0.5561
 4500/25000 [====>.........................] - ETA: 15s - loss: 0.6864 - accuracy: 0.5562
 4600/25000 [====>.........................] - ETA: 15s - loss: 0.6859 - accuracy: 0.5578
 4700/25000 [====>.........................] - ETA: 15s - loss: 0.6854 - accuracy: 0.5600
 4800/25000 [====>.........................] - ETA: 15s - loss: 0.6848 - accuracy: 0.5635
 4900/25000 [====>.........................] - ETA: 15s - loss: 0.6838 - accuracy: 0.5690
 5000/25000 [=====>........................] - ETA: 15s - loss: 0.6828 - accuracy: 0.5738
 5100/25000 [=====>........................] - ETA: 15s - loss: 0.6816 - accuracy: 0.5786
 5200/25000 [=====>........................] - ETA: 15s - loss: 0.6809 - accuracy: 0.5800
 5300/25000 [=====>........................] - ETA: 14s - loss: 0.6793 - accuracy: 0.5845
 5400/25000 [=====>........................] - ETA: 14s - loss: 0.6777 - accuracy: 0.5887
 5500/25000 [=====>........................] - ETA: 14s - loss: 0.6757 - accuracy: 0.5916
 5600/25000 [=====>........................] - ETA: 14s - loss: 0.6727 - accuracy: 0.5959
 5700/25000 [=====>........................] - ETA: 14s - loss: 0.6701 - accuracy: 0.5984
 5800/25000 [=====>........................] - ETA: 14s - loss: 0.6671 - accuracy: 0.6021
 5900/25000 [======>.......................] - ETA: 14s - loss: 0.6642 - accuracy: 0.6059
 6000/25000 [======>.......................] - ETA: 14s - loss: 0.6621 - accuracy: 0.6082
 6100/25000 [======>.......................] - ETA: 14s - loss: 0.6595 - accuracy: 0.6111
 6200/25000 [======>.......................] - ETA: 14s - loss: 0.6564 - accuracy: 0.6142
 6300/25000 [======>.......................] - ETA: 13s - loss: 0.6544 - accuracy: 0.6165
 6400/25000 [======>.......................] - ETA: 13s - loss: 0.6511 - accuracy: 0.6194
 6500/25000 [======>.......................] - ETA: 13s - loss: 0.6477 - accuracy: 0.6228
 6600/25000 [======>.......................] - ETA: 13s - loss: 0.6441 - accuracy: 0.6259
 6700/25000 [=======>......................] - ETA: 13s - loss: 0.6419 - accuracy: 0.6282
 6800/25000 [=======>......................] - ETA: 13s - loss: 0.6379 - accuracy: 0.6322
 6900/25000 [=======>......................] - ETA: 13s - loss: 0.6353 - accuracy: 0.6352
 7000/25000 [=======>......................] - ETA: 13s - loss: 0.6313 - accuracy: 0.6387
 7100/25000 [=======>......................] - ETA: 13s - loss: 0.6269 - accuracy: 0.6423
 7200/25000 [=======>......................] - ETA: 13s - loss: 0.6234 - accuracy: 0.6447
 7300/25000 [=======>......................] - ETA: 12s - loss: 0.6207 - accuracy: 0.6470
 7400/25000 [=======>......................] - ETA: 12s - loss: 0.6185 - accuracy: 0.6489
 7500/25000 [========>.....................] - ETA: 12s - loss: 0.6150 - accuracy: 0.6519
 7600/25000 [========>.....................] - ETA: 12s - loss: 0.6109 - accuracy: 0.6553
 7700/25000 [========>.....................] - ETA: 12s - loss: 0.6102 - accuracy: 0.6561
 7800/25000 [========>.....................] - ETA: 12s - loss: 0.6078 - accuracy: 0.6579
 7900/25000 [========>.....................] - ETA: 12s - loss: 0.6044 - accuracy: 0.6601
 8000/25000 [========>.....................] - ETA: 12s - loss: 0.6032 - accuracy: 0.6618
 8100/25000 [========>.....................] - ETA: 12s - loss: 0.6003 - accuracy: 0.6640
 8200/25000 [========>.....................] - ETA: 12s - loss: 0.5975 - accuracy: 0.6665
 8300/25000 [========>.....................] - ETA: 12s - loss: 0.5940 - accuracy: 0.6689
 8400/25000 [=========>....................] - ETA: 11s - loss: 0.5914 - accuracy: 0.6708
 8500/25000 [=========>....................] - ETA: 11s - loss: 0.5888 - accuracy: 0.6731
 8600/25000 [=========>....................] - ETA: 11s - loss: 0.5870 - accuracy: 0.6747
 8700/25000 [=========>....................] - ETA: 11s - loss: 0.5853 - accuracy: 0.6762
 8800/25000 [=========>....................] - ETA: 11s - loss: 0.5844 - accuracy: 0.6768
 8900/25000 [=========>....................] - ETA: 11s - loss: 0.5832 - accuracy: 0.6781
 9000/25000 [=========>....................] - ETA: 11s - loss: 0.5814 - accuracy: 0.6797
 9100/25000 [=========>....................] - ETA: 11s - loss: 0.5797 - accuracy: 0.6810
 9200/25000 [==========>...................] - ETA: 11s - loss: 0.5773 - accuracy: 0.6832
 9300/25000 [==========>...................] - ETA: 11s - loss: 0.5765 - accuracy: 0.6840
 9400/25000 [==========>...................] - ETA: 11s - loss: 0.5749 - accuracy: 0.6850
 9500/25000 [==========>...................] - ETA: 11s - loss: 0.5728 - accuracy: 0.6865
 9600/25000 [==========>...................] - ETA: 10s - loss: 0.5716 - accuracy: 0.6876
 9700/25000 [==========>...................] - ETA: 10s - loss: 0.5708 - accuracy: 0.6885
 9800/25000 [==========>...................] - ETA: 10s - loss: 0.5691 - accuracy: 0.6895
 9900/25000 [==========>...................] - ETA: 10s - loss: 0.5673 - accuracy: 0.6910
10000/25000 [===========>..................] - ETA: 10s - loss: 0.5656 - accuracy: 0.6929
10100/25000 [===========>..................] - ETA: 10s - loss: 0.5638 - accuracy: 0.6943
10200/25000 [===========>..................] - ETA: 10s - loss: 0.5629 - accuracy: 0.6950
10300/25000 [===========>..................] - ETA: 10s - loss: 0.5610 - accuracy: 0.6966
10400/25000 [===========>..................] - ETA: 10s - loss: 0.5590 - accuracy: 0.6983
10500/25000 [===========>..................] - ETA: 10s - loss: 0.5585 - accuracy: 0.6989
10600/25000 [===========>..................] - ETA: 10s - loss: 0.5568 - accuracy: 0.7003
10700/25000 [===========>..................] - ETA: 10s - loss: 0.5552 - accuracy: 0.7017
10800/25000 [===========>..................] - ETA: 9s - loss: 0.5540 - accuracy: 0.7027 
10900/25000 [============>.................] - ETA: 9s - loss: 0.5529 - accuracy: 0.7037
11000/25000 [============>.................] - ETA: 9s - loss: 0.5514 - accuracy: 0.7050
11100/25000 [============>.................] - ETA: 9s - loss: 0.5500 - accuracy: 0.7059
11200/25000 [============>.................] - ETA: 9s - loss: 0.5480 - accuracy: 0.7075
11300/25000 [============>.................] - ETA: 9s - loss: 0.5463 - accuracy: 0.7089
11400/25000 [============>.................] - ETA: 9s - loss: 0.5443 - accuracy: 0.7106
11500/25000 [============>.................] - ETA: 9s - loss: 0.5419 - accuracy: 0.7122
11600/25000 [============>.................] - ETA: 9s - loss: 0.5409 - accuracy: 0.7133
11700/25000 [=============>................] - ETA: 9s - loss: 0.5395 - accuracy: 0.7144
11800/25000 [=============>................] - ETA: 9s - loss: 0.5378 - accuracy: 0.7156
11900/25000 [=============>................] - ETA: 9s - loss: 0.5368 - accuracy: 0.7162
12000/25000 [=============>................] - ETA: 9s - loss: 0.5349 - accuracy: 0.7178
12100/25000 [=============>................] - ETA: 8s - loss: 0.5346 - accuracy: 0.7184
12200/25000 [=============>................] - ETA: 8s - loss: 0.5334 - accuracy: 0.7193
12300/25000 [=============>................] - ETA: 8s - loss: 0.5323 - accuracy: 0.7200
12400/25000 [=============>................] - ETA: 8s - loss: 0.5306 - accuracy: 0.7213
12500/25000 [==============>...............] - ETA: 8s - loss: 0.5290 - accuracy: 0.7226
12600/25000 [==============>...............] - ETA: 8s - loss: 0.5276 - accuracy: 0.7234
12700/25000 [==============>...............] - ETA: 8s - loss: 0.5260 - accuracy: 0.7244
12800/25000 [==============>...............] - ETA: 8s - loss: 0.5246 - accuracy: 0.7255
12900/25000 [==============>...............] - ETA: 8s - loss: 0.5236 - accuracy: 0.7261
13000/25000 [==============>...............] - ETA: 8s - loss: 0.5222 - accuracy: 0.7270
13100/25000 [==============>...............] - ETA: 8s - loss: 0.5210 - accuracy: 0.7280
13200/25000 [==============>...............] - ETA: 8s - loss: 0.5197 - accuracy: 0.7292
13300/25000 [==============>...............] - ETA: 8s - loss: 0.5184 - accuracy: 0.7300
13400/25000 [===============>..............] - ETA: 8s - loss: 0.5166 - accuracy: 0.7313
13500/25000 [===============>..............] - ETA: 7s - loss: 0.5150 - accuracy: 0.7325
13600/25000 [===============>..............] - ETA: 7s - loss: 0.5132 - accuracy: 0.7338
13700/25000 [===============>..............] - ETA: 7s - loss: 0.5122 - accuracy: 0.7347
13800/25000 [===============>..............] - ETA: 7s - loss: 0.5104 - accuracy: 0.7358
13900/25000 [===============>..............] - ETA: 7s - loss: 0.5092 - accuracy: 0.7365
14000/25000 [===============>..............] - ETA: 7s - loss: 0.5079 - accuracy: 0.7376
14100/25000 [===============>..............] - ETA: 7s - loss: 0.5062 - accuracy: 0.7387
14200/25000 [================>.............] - ETA: 7s - loss: 0.5054 - accuracy: 0.7393
14300/25000 [================>.............] - ETA: 7s - loss: 0.5040 - accuracy: 0.7404
14400/25000 [================>.............] - ETA: 7s - loss: 0.5037 - accuracy: 0.7410
14500/25000 [================>.............] - ETA: 7s - loss: 0.5034 - accuracy: 0.7415
14600/25000 [================>.............] - ETA: 7s - loss: 0.5022 - accuracy: 0.7423
14700/25000 [================>.............] - ETA: 7s - loss: 0.5009 - accuracy: 0.7431
14800/25000 [================>.............] - ETA: 7s - loss: 0.4997 - accuracy: 0.7440
14900/25000 [================>.............] - ETA: 6s - loss: 0.4983 - accuracy: 0.7450
15000/25000 [=================>............] - ETA: 6s - loss: 0.4967 - accuracy: 0.7459
15100/25000 [=================>............] - ETA: 6s - loss: 0.4962 - accuracy: 0.7464
15200/25000 [=================>............] - ETA: 6s - loss: 0.4950 - accuracy: 0.7472
15300/25000 [=================>............] - ETA: 6s - loss: 0.4940 - accuracy: 0.7478
15400/25000 [=================>............] - ETA: 6s - loss: 0.4925 - accuracy: 0.7489
15500/25000 [=================>............] - ETA: 6s - loss: 0.4916 - accuracy: 0.7494
15600/25000 [=================>............] - ETA: 6s - loss: 0.4905 - accuracy: 0.7500
15700/25000 [=================>............] - ETA: 6s - loss: 0.4892 - accuracy: 0.7508
15800/25000 [=================>............] - ETA: 6s - loss: 0.4884 - accuracy: 0.7515
15900/25000 [==================>...........] - ETA: 6s - loss: 0.4868 - accuracy: 0.7526
16000/25000 [==================>...........] - ETA: 6s - loss: 0.4858 - accuracy: 0.7534
16100/25000 [==================>...........] - ETA: 6s - loss: 0.4846 - accuracy: 0.7542
16200/25000 [==================>...........] - ETA: 6s - loss: 0.4838 - accuracy: 0.7546
16300/25000 [==================>...........] - ETA: 5s - loss: 0.4825 - accuracy: 0.7554
16400/25000 [==================>...........] - ETA: 5s - loss: 0.4816 - accuracy: 0.7561
16500/25000 [==================>...........] - ETA: 5s - loss: 0.4807 - accuracy: 0.7568
16600/25000 [==================>...........] - ETA: 5s - loss: 0.4800 - accuracy: 0.7574
16700/25000 [===================>..........] - ETA: 5s - loss: 0.4788 - accuracy: 0.7581
16800/25000 [===================>..........] - ETA: 5s - loss: 0.4776 - accuracy: 0.7589
16900/25000 [===================>..........] - ETA: 5s - loss: 0.4765 - accuracy: 0.7596
17000/25000 [===================>..........] - ETA: 5s - loss: 0.4753 - accuracy: 0.7604
17100/25000 [===================>..........] - ETA: 5s - loss: 0.4749 - accuracy: 0.7609
17200/25000 [===================>..........] - ETA: 5s - loss: 0.4743 - accuracy: 0.7613
17300/25000 [===================>..........] - ETA: 5s - loss: 0.4737 - accuracy: 0.7617
17400/25000 [===================>..........] - ETA: 5s - loss: 0.4726 - accuracy: 0.7625
17500/25000 [====================>.........] - ETA: 5s - loss: 0.4721 - accuracy: 0.7630
17600/25000 [====================>.........] - ETA: 5s - loss: 0.4711 - accuracy: 0.7636
17700/25000 [====================>.........] - ETA: 4s - loss: 0.4700 - accuracy: 0.7644
17800/25000 [====================>.........] - ETA: 4s - loss: 0.4688 - accuracy: 0.7652
17900/25000 [====================>.........] - ETA: 4s - loss: 0.4677 - accuracy: 0.7660
18000/25000 [====================>.........] - ETA: 4s - loss: 0.4667 - accuracy: 0.7667
18100/25000 [====================>.........] - ETA: 4s - loss: 0.4656 - accuracy: 0.7674
18200/25000 [====================>.........] - ETA: 4s - loss: 0.4645 - accuracy: 0.7681
18300/25000 [====================>.........] - ETA: 4s - loss: 0.4637 - accuracy: 0.7687
18400/25000 [=====================>........] - ETA: 4s - loss: 0.4627 - accuracy: 0.7692
18500/25000 [=====================>........] - ETA: 4s - loss: 0.4622 - accuracy: 0.7695
18600/25000 [=====================>........] - ETA: 4s - loss: 0.4614 - accuracy: 0.7701
18700/25000 [=====================>........] - ETA: 4s - loss: 0.4603 - accuracy: 0.7707
18800/25000 [=====================>........] - ETA: 4s - loss: 0.4591 - accuracy: 0.7714
18900/25000 [=====================>........] - ETA: 4s - loss: 0.4584 - accuracy: 0.7721
19000/25000 [=====================>........] - ETA: 4s - loss: 0.4574 - accuracy: 0.7727
19100/25000 [=====================>........] - ETA: 4s - loss: 0.4564 - accuracy: 0.7732
19200/25000 [======================>.......] - ETA: 3s - loss: 0.4556 - accuracy: 0.7739
19300/25000 [======================>.......] - ETA: 3s - loss: 0.4548 - accuracy: 0.7744
19400/25000 [======================>.......] - ETA: 3s - loss: 0.4542 - accuracy: 0.7747
19500/25000 [======================>.......] - ETA: 3s - loss: 0.4536 - accuracy: 0.7751
19600/25000 [======================>.......] - ETA: 3s - loss: 0.4531 - accuracy: 0.7756
19700/25000 [======================>.......] - ETA: 3s - loss: 0.4524 - accuracy: 0.7761
19800/25000 [======================>.......] - ETA: 3s - loss: 0.4519 - accuracy: 0.7766
19900/25000 [======================>.......] - ETA: 3s - loss: 0.4513 - accuracy: 0.7769
20000/25000 [=======================>......] - ETA: 3s - loss: 0.4503 - accuracy: 0.7776
20100/25000 [=======================>......] - ETA: 3s - loss: 0.4496 - accuracy: 0.7781
20200/25000 [=======================>......] - ETA: 3s - loss: 0.4484 - accuracy: 0.7790
20300/25000 [=======================>......] - ETA: 3s - loss: 0.4475 - accuracy: 0.7796
20400/25000 [=======================>......] - ETA: 3s - loss: 0.4466 - accuracy: 0.7801
20500/25000 [=======================>......] - ETA: 3s - loss: 0.4460 - accuracy: 0.7805
20600/25000 [=======================>......] - ETA: 2s - loss: 0.4450 - accuracy: 0.7811
20700/25000 [=======================>......] - ETA: 2s - loss: 0.4443 - accuracy: 0.7816
20800/25000 [=======================>......] - ETA: 2s - loss: 0.4434 - accuracy: 0.7821
20900/25000 [========================>.....] - ETA: 2s - loss: 0.4430 - accuracy: 0.7824
21000/25000 [========================>.....] - ETA: 2s - loss: 0.4424 - accuracy: 0.7828
21100/25000 [========================>.....] - ETA: 2s - loss: 0.4419 - accuracy: 0.7831
21200/25000 [========================>.....] - ETA: 2s - loss: 0.4416 - accuracy: 0.7833
21300/25000 [========================>.....] - ETA: 2s - loss: 0.4409 - accuracy: 0.7838
21400/25000 [========================>.....] - ETA: 2s - loss: 0.4403 - accuracy: 0.7842
21500/25000 [========================>.....] - ETA: 2s - loss: 0.4395 - accuracy: 0.7846
21600/25000 [========================>.....] - ETA: 2s - loss: 0.4389 - accuracy: 0.7849
21700/25000 [=========================>....] - ETA: 2s - loss: 0.4389 - accuracy: 0.7853
21800/25000 [=========================>....] - ETA: 2s - loss: 0.4382 - accuracy: 0.7858
21900/25000 [=========================>....] - ETA: 2s - loss: 0.4379 - accuracy: 0.7859
22000/25000 [=========================>....] - ETA: 2s - loss: 0.4373 - accuracy: 0.7862
22100/25000 [=========================>....] - ETA: 1s - loss: 0.4368 - accuracy: 0.7866
22200/25000 [=========================>....] - ETA: 1s - loss: 0.4359 - accuracy: 0.7871
22300/25000 [=========================>....] - ETA: 1s - loss: 0.4355 - accuracy: 0.7873
22400/25000 [=========================>....] - ETA: 1s - loss: 0.4349 - accuracy: 0.7877
22500/25000 [==========================>...] - ETA: 1s - loss: 0.4345 - accuracy: 0.7879
22600/25000 [==========================>...] - ETA: 1s - loss: 0.4343 - accuracy: 0.7882
22700/25000 [==========================>...] - ETA: 1s - loss: 0.4335 - accuracy: 0.7887
22800/25000 [==========================>...] - ETA: 1s - loss: 0.4328 - accuracy: 0.7890
22900/25000 [==========================>...] - ETA: 1s - loss: 0.4325 - accuracy: 0.7891
23000/25000 [==========================>...] - ETA: 1s - loss: 0.4319 - accuracy: 0.7895
23100/25000 [==========================>...] - ETA: 1s - loss: 0.4317 - accuracy: 0.7898
23200/25000 [==========================>...] - ETA: 1s - loss: 0.4313 - accuracy: 0.7901
23300/25000 [==========================>...] - ETA: 1s - loss: 0.4307 - accuracy: 0.7904
23400/25000 [===========================>..] - ETA: 1s - loss: 0.4305 - accuracy: 0.7906
23500/25000 [===========================>..] - ETA: 1s - loss: 0.4299 - accuracy: 0.7909
23600/25000 [===========================>..] - ETA: 0s - loss: 0.4293 - accuracy: 0.7912
23700/25000 [===========================>..] - ETA: 0s - loss: 0.4288 - accuracy: 0.7915
23800/25000 [===========================>..] - ETA: 0s - loss: 0.4287 - accuracy: 0.7918
23900/25000 [===========================>..] - ETA: 0s - loss: 0.4282 - accuracy: 0.7920
24000/25000 [===========================>..] - ETA: 0s - loss: 0.4281 - accuracy: 0.7921
24100/25000 [===========================>..] - ETA: 0s - loss: 0.4275 - accuracy: 0.7924
24200/25000 [============================>.] - ETA: 0s - loss: 0.4271 - accuracy: 0.7926
24300/25000 [============================>.] - ETA: 0s - loss: 0.4263 - accuracy: 0.7930
24400/25000 [============================>.] - ETA: 0s - loss: 0.4258 - accuracy: 0.7934
24500/25000 [============================>.] - ETA: 0s - loss: 0.4256 - accuracy: 0.7936
24600/25000 [============================>.] - ETA: 0s - loss: 0.4256 - accuracy: 0.7936
24700/25000 [============================>.] - ETA: 0s - loss: 0.4253 - accuracy: 0.7936
24800/25000 [============================>.] - ETA: 0s - loss: 0.4251 - accuracy: 0.7938
24900/25000 [============================>.] - ETA: 0s - loss: 0.4246 - accuracy: 0.7941
25000/25000 [==============================] - 21s 834us/step - loss: 0.4242 - accuracy: 0.7944 - val_loss: 0.3155 - val_accuracy: 0.8626
Epoch 2/3

  100/25000 [..............................] - ETA: 15s - loss: 0.1888 - accuracy: 0.9400
  200/25000 [..............................] - ETA: 15s - loss: 0.2492 - accuracy: 0.9250
  300/25000 [..............................] - ETA: 15s - loss: 0.2415 - accuracy: 0.9167
  400/25000 [..............................] - ETA: 15s - loss: 0.2405 - accuracy: 0.9150
  500/25000 [..............................] - ETA: 15s - loss: 0.2289 - accuracy: 0.9240
  600/25000 [..............................] - ETA: 16s - loss: 0.2337 - accuracy: 0.9233
  700/25000 [..............................] - ETA: 15s - loss: 0.2369 - accuracy: 0.9200
  800/25000 [..............................] - ETA: 15s - loss: 0.2385 - accuracy: 0.9212
  900/25000 [>.............................] - ETA: 15s - loss: 0.2405 - accuracy: 0.9178
 1000/25000 [>.............................] - ETA: 15s - loss: 0.2395 - accuracy: 0.9170
 1100/25000 [>.............................] - ETA: 15s - loss: 0.2374 - accuracy: 0.9173
 1200/25000 [>.............................] - ETA: 15s - loss: 0.2355 - accuracy: 0.9175
 1300/25000 [>.............................] - ETA: 15s - loss: 0.2344 - accuracy: 0.9169
 1400/25000 [>.............................] - ETA: 15s - loss: 0.2309 - accuracy: 0.9179
 1500/25000 [>.............................] - ETA: 15s - loss: 0.2323 - accuracy: 0.9193
 1600/25000 [>.............................] - ETA: 15s - loss: 0.2327 - accuracy: 0.9175
 1700/25000 [=>............................] - ETA: 15s - loss: 0.2299 - accuracy: 0.9182
 1800/25000 [=>............................] - ETA: 14s - loss: 0.2316 - accuracy: 0.9183
 1900/25000 [=>............................] - ETA: 14s - loss: 0.2303 - accuracy: 0.9195
 2000/25000 [=>............................] - ETA: 14s - loss: 0.2302 - accuracy: 0.9190
 2100/25000 [=>............................] - ETA: 14s - loss: 0.2320 - accuracy: 0.9171
 2200/25000 [=>............................] - ETA: 14s - loss: 0.2354 - accuracy: 0.9145
 2300/25000 [=>............................] - ETA: 14s - loss: 0.2408 - accuracy: 0.9122
 2400/25000 [=>............................] - ETA: 14s - loss: 0.2396 - accuracy: 0.9104
 2500/25000 [==>...........................] - ETA: 14s - loss: 0.2389 - accuracy: 0.9108
 2600/25000 [==>...........................] - ETA: 14s - loss: 0.2367 - accuracy: 0.9115
 2700/25000 [==>...........................] - ETA: 14s - loss: 0.2349 - accuracy: 0.9122
 2800/25000 [==>...........................] - ETA: 14s - loss: 0.2366 - accuracy: 0.9111
 2900/25000 [==>...........................] - ETA: 14s - loss: 0.2383 - accuracy: 0.9107
 3000/25000 [==>...........................] - ETA: 14s - loss: 0.2359 - accuracy: 0.9110
 3100/25000 [==>...........................] - ETA: 13s - loss: 0.2358 - accuracy: 0.9119
 3200/25000 [==>...........................] - ETA: 13s - loss: 0.2357 - accuracy: 0.9122
 3300/25000 [==>...........................] - ETA: 13s - loss: 0.2351 - accuracy: 0.9130
 3400/25000 [===>..........................] - ETA: 13s - loss: 0.2335 - accuracy: 0.9135
 3500/25000 [===>..........................] - ETA: 13s - loss: 0.2364 - accuracy: 0.9123
 3600/25000 [===>..........................] - ETA: 13s - loss: 0.2361 - accuracy: 0.9131
 3700/25000 [===>..........................] - ETA: 13s - loss: 0.2351 - accuracy: 0.9127
 3800/25000 [===>..........................] - ETA: 13s - loss: 0.2347 - accuracy: 0.9124
 3900/25000 [===>..........................] - ETA: 13s - loss: 0.2329 - accuracy: 0.9136
 4000/25000 [===>..........................] - ETA: 13s - loss: 0.2344 - accuracy: 0.9128
 4100/25000 [===>..........................] - ETA: 13s - loss: 0.2362 - accuracy: 0.9110
 4200/25000 [====>.........................] - ETA: 13s - loss: 0.2380 - accuracy: 0.9102
 4300/25000 [====>.........................] - ETA: 13s - loss: 0.2388 - accuracy: 0.9105
 4400/25000 [====>.........................] - ETA: 13s - loss: 0.2378 - accuracy: 0.9105
 4500/25000 [====>.........................] - ETA: 13s - loss: 0.2387 - accuracy: 0.9104
 4600/25000 [====>.........................] - ETA: 12s - loss: 0.2370 - accuracy: 0.9111
 4700/25000 [====>.........................] - ETA: 12s - loss: 0.2370 - accuracy: 0.9115
 4800/25000 [====>.........................] - ETA: 12s - loss: 0.2374 - accuracy: 0.9110
 4900/25000 [====>.........................] - ETA: 12s - loss: 0.2368 - accuracy: 0.9118
 5000/25000 [=====>........................] - ETA: 12s - loss: 0.2367 - accuracy: 0.9114
 5100/25000 [=====>........................] - ETA: 12s - loss: 0.2354 - accuracy: 0.9124
 5200/25000 [=====>........................] - ETA: 12s - loss: 0.2347 - accuracy: 0.9125
 5300/25000 [=====>........................] - ETA: 12s - loss: 0.2354 - accuracy: 0.9121
 5400/25000 [=====>........................] - ETA: 12s - loss: 0.2344 - accuracy: 0.9124
 5500/25000 [=====>........................] - ETA: 12s - loss: 0.2363 - accuracy: 0.9124
 5600/25000 [=====>........................] - ETA: 12s - loss: 0.2374 - accuracy: 0.9112
 5700/25000 [=====>........................] - ETA: 12s - loss: 0.2375 - accuracy: 0.9111
 5800/25000 [=====>........................] - ETA: 12s - loss: 0.2385 - accuracy: 0.9098
 5900/25000 [======>.......................] - ETA: 12s - loss: 0.2371 - accuracy: 0.9105
 6000/25000 [======>.......................] - ETA: 12s - loss: 0.2357 - accuracy: 0.9107
 6100/25000 [======>.......................] - ETA: 11s - loss: 0.2363 - accuracy: 0.9103
 6200/25000 [======>.......................] - ETA: 11s - loss: 0.2358 - accuracy: 0.9105
 6300/25000 [======>.......................] - ETA: 11s - loss: 0.2358 - accuracy: 0.9106
 6400/25000 [======>.......................] - ETA: 11s - loss: 0.2374 - accuracy: 0.9105
 6500/25000 [======>.......................] - ETA: 11s - loss: 0.2364 - accuracy: 0.9109
 6600/25000 [======>.......................] - ETA: 11s - loss: 0.2361 - accuracy: 0.9115
 6700/25000 [=======>......................] - ETA: 11s - loss: 0.2355 - accuracy: 0.9116
 6800/25000 [=======>......................] - ETA: 11s - loss: 0.2347 - accuracy: 0.9125
 6900/25000 [=======>......................] - ETA: 11s - loss: 0.2339 - accuracy: 0.9130
 7000/25000 [=======>......................] - ETA: 11s - loss: 0.2341 - accuracy: 0.9126
 7100/25000 [=======>......................] - ETA: 11s - loss: 0.2347 - accuracy: 0.9124
 7200/25000 [=======>......................] - ETA: 11s - loss: 0.2344 - accuracy: 0.9125
 7300/25000 [=======>......................] - ETA: 11s - loss: 0.2343 - accuracy: 0.9123
 7400/25000 [=======>......................] - ETA: 11s - loss: 0.2351 - accuracy: 0.9118
 7500/25000 [========>.....................] - ETA: 11s - loss: 0.2341 - accuracy: 0.9123
 7600/25000 [========>.....................] - ETA: 11s - loss: 0.2348 - accuracy: 0.9121
 7700/25000 [========>.....................] - ETA: 10s - loss: 0.2352 - accuracy: 0.9121
 7800/25000 [========>.....................] - ETA: 10s - loss: 0.2352 - accuracy: 0.9121
 7900/25000 [========>.....................] - ETA: 10s - loss: 0.2354 - accuracy: 0.9115
 8000/25000 [========>.....................] - ETA: 10s - loss: 0.2342 - accuracy: 0.9118
 8100/25000 [========>.....................] - ETA: 10s - loss: 0.2349 - accuracy: 0.9115
 8200/25000 [========>.....................] - ETA: 10s - loss: 0.2352 - accuracy: 0.9113
 8300/25000 [========>.....................] - ETA: 10s - loss: 0.2355 - accuracy: 0.9113
 8400/25000 [=========>....................] - ETA: 10s - loss: 0.2365 - accuracy: 0.9112
 8500/25000 [=========>....................] - ETA: 10s - loss: 0.2362 - accuracy: 0.9112
 8600/25000 [=========>....................] - ETA: 10s - loss: 0.2358 - accuracy: 0.9112
 8700/25000 [=========>....................] - ETA: 10s - loss: 0.2354 - accuracy: 0.9111
 8800/25000 [=========>....................] - ETA: 10s - loss: 0.2354 - accuracy: 0.9111
 8900/25000 [=========>....................] - ETA: 10s - loss: 0.2362 - accuracy: 0.9108
 9000/25000 [=========>....................] - ETA: 10s - loss: 0.2357 - accuracy: 0.9110
 9100/25000 [=========>....................] - ETA: 10s - loss: 0.2347 - accuracy: 0.9114
 9200/25000 [==========>...................] - ETA: 10s - loss: 0.2348 - accuracy: 0.9113
 9300/25000 [==========>...................] - ETA: 9s - loss: 0.2356 - accuracy: 0.9113 
 9400/25000 [==========>...................] - ETA: 9s - loss: 0.2352 - accuracy: 0.9115
 9500/25000 [==========>...................] - ETA: 9s - loss: 0.2348 - accuracy: 0.9117
 9600/25000 [==========>...................] - ETA: 9s - loss: 0.2360 - accuracy: 0.9110
 9700/25000 [==========>...................] - ETA: 9s - loss: 0.2359 - accuracy: 0.9110
 9800/25000 [==========>...................] - ETA: 9s - loss: 0.2360 - accuracy: 0.9109
 9900/25000 [==========>...................] - ETA: 9s - loss: 0.2359 - accuracy: 0.9109
10000/25000 [===========>..................] - ETA: 9s - loss: 0.2355 - accuracy: 0.9110
10100/25000 [===========>..................] - ETA: 9s - loss: 0.2357 - accuracy: 0.9108
10200/25000 [===========>..................] - ETA: 9s - loss: 0.2356 - accuracy: 0.9107
10300/25000 [===========>..................] - ETA: 9s - loss: 0.2355 - accuracy: 0.9107
10400/25000 [===========>..................] - ETA: 9s - loss: 0.2355 - accuracy: 0.9110
10500/25000 [===========>..................] - ETA: 9s - loss: 0.2365 - accuracy: 0.9107
10600/25000 [===========>..................] - ETA: 9s - loss: 0.2367 - accuracy: 0.9105
10700/25000 [===========>..................] - ETA: 9s - loss: 0.2363 - accuracy: 0.9107
10800/25000 [===========>..................] - ETA: 9s - loss: 0.2370 - accuracy: 0.9101
10900/25000 [============>.................] - ETA: 8s - loss: 0.2370 - accuracy: 0.9099
11000/25000 [============>.................] - ETA: 8s - loss: 0.2373 - accuracy: 0.9097
11100/25000 [============>.................] - ETA: 8s - loss: 0.2368 - accuracy: 0.9101
11200/25000 [============>.................] - ETA: 8s - loss: 0.2373 - accuracy: 0.9100
11300/25000 [============>.................] - ETA: 8s - loss: 0.2371 - accuracy: 0.9097
11400/25000 [============>.................] - ETA: 8s - loss: 0.2366 - accuracy: 0.9102
11500/25000 [============>.................] - ETA: 8s - loss: 0.2368 - accuracy: 0.9102
11600/25000 [============>.................] - ETA: 8s - loss: 0.2370 - accuracy: 0.9103
11700/25000 [=============>................] - ETA: 8s - loss: 0.2368 - accuracy: 0.9104
11800/25000 [=============>................] - ETA: 8s - loss: 0.2366 - accuracy: 0.9104
11900/25000 [=============>................] - ETA: 8s - loss: 0.2372 - accuracy: 0.9103
12000/25000 [=============>................] - ETA: 8s - loss: 0.2365 - accuracy: 0.9107
12100/25000 [=============>................] - ETA: 8s - loss: 0.2364 - accuracy: 0.9107
12200/25000 [=============>................] - ETA: 8s - loss: 0.2364 - accuracy: 0.9106
12300/25000 [=============>................] - ETA: 8s - loss: 0.2372 - accuracy: 0.9102
12400/25000 [=============>................] - ETA: 8s - loss: 0.2378 - accuracy: 0.9095
12500/25000 [==============>...............] - ETA: 7s - loss: 0.2379 - accuracy: 0.9093
12600/25000 [==============>...............] - ETA: 7s - loss: 0.2378 - accuracy: 0.9093
12700/25000 [==============>...............] - ETA: 7s - loss: 0.2382 - accuracy: 0.9091
12800/25000 [==============>...............] - ETA: 7s - loss: 0.2384 - accuracy: 0.9091
12900/25000 [==============>...............] - ETA: 7s - loss: 0.2386 - accuracy: 0.9090
13000/25000 [==============>...............] - ETA: 7s - loss: 0.2383 - accuracy: 0.9092
13100/25000 [==============>...............] - ETA: 7s - loss: 0.2383 - accuracy: 0.9092
13200/25000 [==============>...............] - ETA: 7s - loss: 0.2379 - accuracy: 0.9092
13300/25000 [==============>...............] - ETA: 7s - loss: 0.2386 - accuracy: 0.9088
13400/25000 [===============>..............] - ETA: 7s - loss: 0.2383 - accuracy: 0.9089
13500/25000 [===============>..............] - ETA: 7s - loss: 0.2388 - accuracy: 0.9087
13600/25000 [===============>..............] - ETA: 7s - loss: 0.2390 - accuracy: 0.9087
13700/25000 [===============>..............] - ETA: 7s - loss: 0.2396 - accuracy: 0.9086
13800/25000 [===============>..............] - ETA: 7s - loss: 0.2399 - accuracy: 0.9086
13900/25000 [===============>..............] - ETA: 7s - loss: 0.2401 - accuracy: 0.9083
14000/25000 [===============>..............] - ETA: 6s - loss: 0.2399 - accuracy: 0.9084
14100/25000 [===============>..............] - ETA: 6s - loss: 0.2399 - accuracy: 0.9085
14200/25000 [================>.............] - ETA: 6s - loss: 0.2405 - accuracy: 0.9082
14300/25000 [================>.............] - ETA: 6s - loss: 0.2400 - accuracy: 0.9084
14400/25000 [================>.............] - ETA: 6s - loss: 0.2399 - accuracy: 0.9084
14500/25000 [================>.............] - ETA: 6s - loss: 0.2401 - accuracy: 0.9083
14600/25000 [================>.............] - ETA: 6s - loss: 0.2402 - accuracy: 0.9082
14700/25000 [================>.............] - ETA: 6s - loss: 0.2405 - accuracy: 0.9080
14800/25000 [================>.............] - ETA: 6s - loss: 0.2404 - accuracy: 0.9082
14900/25000 [================>.............] - ETA: 6s - loss: 0.2403 - accuracy: 0.9083
15000/25000 [=================>............] - ETA: 6s - loss: 0.2408 - accuracy: 0.9081
15100/25000 [=================>............] - ETA: 6s - loss: 0.2414 - accuracy: 0.9078
15200/25000 [=================>............] - ETA: 6s - loss: 0.2417 - accuracy: 0.9076
15300/25000 [=================>............] - ETA: 6s - loss: 0.2417 - accuracy: 0.9075
15400/25000 [=================>............] - ETA: 6s - loss: 0.2423 - accuracy: 0.9073
15500/25000 [=================>............] - ETA: 6s - loss: 0.2424 - accuracy: 0.9074
15600/25000 [=================>............] - ETA: 5s - loss: 0.2425 - accuracy: 0.9075
15700/25000 [=================>............] - ETA: 5s - loss: 0.2426 - accuracy: 0.9075
15800/25000 [=================>............] - ETA: 5s - loss: 0.2422 - accuracy: 0.9076
15900/25000 [==================>...........] - ETA: 5s - loss: 0.2421 - accuracy: 0.9077
16000/25000 [==================>...........] - ETA: 5s - loss: 0.2432 - accuracy: 0.9072
16100/25000 [==================>...........] - ETA: 5s - loss: 0.2427 - accuracy: 0.9072
16200/25000 [==================>...........] - ETA: 5s - loss: 0.2421 - accuracy: 0.9075
16300/25000 [==================>...........] - ETA: 5s - loss: 0.2423 - accuracy: 0.9074
16400/25000 [==================>...........] - ETA: 5s - loss: 0.2422 - accuracy: 0.9073
16500/25000 [==================>...........] - ETA: 5s - loss: 0.2424 - accuracy: 0.9075
16600/25000 [==================>...........] - ETA: 5s - loss: 0.2427 - accuracy: 0.9073
16700/25000 [===================>..........] - ETA: 5s - loss: 0.2428 - accuracy: 0.9072
16800/25000 [===================>..........] - ETA: 5s - loss: 0.2426 - accuracy: 0.9074
16900/25000 [===================>..........] - ETA: 5s - loss: 0.2427 - accuracy: 0.9072
17000/25000 [===================>..........] - ETA: 5s - loss: 0.2428 - accuracy: 0.9073
17100/25000 [===================>..........] - ETA: 5s - loss: 0.2425 - accuracy: 0.9075
17200/25000 [===================>..........] - ETA: 4s - loss: 0.2425 - accuracy: 0.9075
17300/25000 [===================>..........] - ETA: 4s - loss: 0.2437 - accuracy: 0.9069
17400/25000 [===================>..........] - ETA: 4s - loss: 0.2441 - accuracy: 0.9067
17500/25000 [====================>.........] - ETA: 4s - loss: 0.2440 - accuracy: 0.9066
17600/25000 [====================>.........] - ETA: 4s - loss: 0.2440 - accuracy: 0.9065
17700/25000 [====================>.........] - ETA: 4s - loss: 0.2440 - accuracy: 0.9064
17800/25000 [====================>.........] - ETA: 4s - loss: 0.2445 - accuracy: 0.9062
17900/25000 [====================>.........] - ETA: 4s - loss: 0.2447 - accuracy: 0.9062
18000/25000 [====================>.........] - ETA: 4s - loss: 0.2447 - accuracy: 0.9062
18100/25000 [====================>.........] - ETA: 4s - loss: 0.2450 - accuracy: 0.9061
18200/25000 [====================>.........] - ETA: 4s - loss: 0.2445 - accuracy: 0.9063
18300/25000 [====================>.........] - ETA: 4s - loss: 0.2449 - accuracy: 0.9061
18400/25000 [=====================>........] - ETA: 4s - loss: 0.2449 - accuracy: 0.9061
18500/25000 [=====================>........] - ETA: 4s - loss: 0.2453 - accuracy: 0.9059
18600/25000 [=====================>........] - ETA: 4s - loss: 0.2452 - accuracy: 0.9060
18700/25000 [=====================>........] - ETA: 4s - loss: 0.2456 - accuracy: 0.9056
18800/25000 [=====================>........] - ETA: 3s - loss: 0.2457 - accuracy: 0.9056
18900/25000 [=====================>........] - ETA: 3s - loss: 0.2456 - accuracy: 0.9056
19000/25000 [=====================>........] - ETA: 3s - loss: 0.2456 - accuracy: 0.9057
19100/25000 [=====================>........] - ETA: 3s - loss: 0.2453 - accuracy: 0.9059
19200/25000 [======================>.......] - ETA: 3s - loss: 0.2455 - accuracy: 0.9057
19300/25000 [======================>.......] - ETA: 3s - loss: 0.2452 - accuracy: 0.9060
19400/25000 [======================>.......] - ETA: 3s - loss: 0.2454 - accuracy: 0.9057
19500/25000 [======================>.......] - ETA: 3s - loss: 0.2456 - accuracy: 0.9055
19600/25000 [======================>.......] - ETA: 3s - loss: 0.2456 - accuracy: 0.9055
19700/25000 [======================>.......] - ETA: 3s - loss: 0.2458 - accuracy: 0.9053
19800/25000 [======================>.......] - ETA: 3s - loss: 0.2460 - accuracy: 0.9052
19900/25000 [======================>.......] - ETA: 3s - loss: 0.2455 - accuracy: 0.9055
20000/25000 [=======================>......] - ETA: 3s - loss: 0.2457 - accuracy: 0.9054
20100/25000 [=======================>......] - ETA: 3s - loss: 0.2457 - accuracy: 0.9053
20200/25000 [=======================>......] - ETA: 3s - loss: 0.2456 - accuracy: 0.9053
20300/25000 [=======================>......] - ETA: 2s - loss: 0.2454 - accuracy: 0.9054
20400/25000 [=======================>......] - ETA: 2s - loss: 0.2454 - accuracy: 0.9054
20500/25000 [=======================>......] - ETA: 2s - loss: 0.2452 - accuracy: 0.9056
20600/25000 [=======================>......] - ETA: 2s - loss: 0.2454 - accuracy: 0.9054
20700/25000 [=======================>......] - ETA: 2s - loss: 0.2462 - accuracy: 0.9051
20800/25000 [=======================>......] - ETA: 2s - loss: 0.2461 - accuracy: 0.9051
20900/25000 [========================>.....] - ETA: 2s - loss: 0.2463 - accuracy: 0.9050
21000/25000 [========================>.....] - ETA: 2s - loss: 0.2462 - accuracy: 0.9050
21100/25000 [========================>.....] - ETA: 2s - loss: 0.2461 - accuracy: 0.9051
21200/25000 [========================>.....] - ETA: 2s - loss: 0.2462 - accuracy: 0.9049
21300/25000 [========================>.....] - ETA: 2s - loss: 0.2463 - accuracy: 0.9049
21400/25000 [========================>.....] - ETA: 2s - loss: 0.2462 - accuracy: 0.9049
21500/25000 [========================>.....] - ETA: 2s - loss: 0.2455 - accuracy: 0.9052
21600/25000 [========================>.....] - ETA: 2s - loss: 0.2456 - accuracy: 0.9051
21700/25000 [=========================>....] - ETA: 2s - loss: 0.2457 - accuracy: 0.9051
21800/25000 [=========================>....] - ETA: 2s - loss: 0.2455 - accuracy: 0.9052
21900/25000 [=========================>....] - ETA: 1s - loss: 0.2453 - accuracy: 0.9052
22000/25000 [=========================>....] - ETA: 1s - loss: 0.2461 - accuracy: 0.9049
22100/25000 [=========================>....] - ETA: 1s - loss: 0.2466 - accuracy: 0.9047
22200/25000 [=========================>....] - ETA: 1s - loss: 0.2466 - accuracy: 0.9047
22300/25000 [=========================>....] - ETA: 1s - loss: 0.2468 - accuracy: 0.9047
22400/25000 [=========================>....] - ETA: 1s - loss: 0.2468 - accuracy: 0.9046
22500/25000 [==========================>...] - ETA: 1s - loss: 0.2469 - accuracy: 0.9046
22600/25000 [==========================>...] - ETA: 1s - loss: 0.2471 - accuracy: 0.9046
22700/25000 [==========================>...] - ETA: 1s - loss: 0.2471 - accuracy: 0.9046
22800/25000 [==========================>...] - ETA: 1s - loss: 0.2469 - accuracy: 0.9046
22900/25000 [==========================>...] - ETA: 1s - loss: 0.2470 - accuracy: 0.9044
23000/25000 [==========================>...] - ETA: 1s - loss: 0.2469 - accuracy: 0.9045
23100/25000 [==========================>...] - ETA: 1s - loss: 0.2466 - accuracy: 0.9046
23200/25000 [==========================>...] - ETA: 1s - loss: 0.2470 - accuracy: 0.9044
23300/25000 [==========================>...] - ETA: 1s - loss: 0.2467 - accuracy: 0.9046
23400/25000 [===========================>..] - ETA: 1s - loss: 0.2468 - accuracy: 0.9046
23500/25000 [===========================>..] - ETA: 0s - loss: 0.2468 - accuracy: 0.9045
23600/25000 [===========================>..] - ETA: 0s - loss: 0.2463 - accuracy: 0.9047
23700/25000 [===========================>..] - ETA: 0s - loss: 0.2467 - accuracy: 0.9046
23800/25000 [===========================>..] - ETA: 0s - loss: 0.2464 - accuracy: 0.9047
23900/25000 [===========================>..] - ETA: 0s - loss: 0.2462 - accuracy: 0.9047
24000/25000 [===========================>..] - ETA: 0s - loss: 0.2464 - accuracy: 0.9046
24100/25000 [===========================>..] - ETA: 0s - loss: 0.2467 - accuracy: 0.9045
24200/25000 [============================>.] - ETA: 0s - loss: 0.2464 - accuracy: 0.9046
24300/25000 [============================>.] - ETA: 0s - loss: 0.2464 - accuracy: 0.9046
24400/25000 [============================>.] - ETA: 0s - loss: 0.2464 - accuracy: 0.9045
24500/25000 [============================>.] - ETA: 0s - loss: 0.2463 - accuracy: 0.9045
24600/25000 [============================>.] - ETA: 0s - loss: 0.2465 - accuracy: 0.9044
24700/25000 [============================>.] - ETA: 0s - loss: 0.2463 - accuracy: 0.9045
24800/25000 [============================>.] - ETA: 0s - loss: 0.2463 - accuracy: 0.9045
24900/25000 [============================>.] - ETA: 0s - loss: 0.2461 - accuracy: 0.9045
25000/25000 [==============================] - 20s 800us/step - loss: 0.2460 - accuracy: 0.9046 - val_loss: 0.3114 - val_accuracy: 0.8660
Epoch 3/3

  100/25000 [..............................] - ETA: 15s - loss: 0.1694 - accuracy: 0.9700
  200/25000 [..............................] - ETA: 16s - loss: 0.1353 - accuracy: 0.9750
  300/25000 [..............................] - ETA: 16s - loss: 0.1542 - accuracy: 0.9667
  400/25000 [..............................] - ETA: 15s - loss: 0.1661 - accuracy: 0.9600
  500/25000 [..............................] - ETA: 15s - loss: 0.1629 - accuracy: 0.9580
  600/25000 [..............................] - ETA: 15s - loss: 0.1647 - accuracy: 0.9583
  700/25000 [..............................] - ETA: 15s - loss: 0.1659 - accuracy: 0.9514
  800/25000 [..............................] - ETA: 15s - loss: 0.1657 - accuracy: 0.9488
  900/25000 [>.............................] - ETA: 15s - loss: 0.1658 - accuracy: 0.9489
 1000/25000 [>.............................] - ETA: 15s - loss: 0.1583 - accuracy: 0.9520
 1100/25000 [>.............................] - ETA: 15s - loss: 0.1551 - accuracy: 0.9527
 1200/25000 [>.............................] - ETA: 15s - loss: 0.1599 - accuracy: 0.9483
 1300/25000 [>.............................] - ETA: 15s - loss: 0.1585 - accuracy: 0.9485
 1400/25000 [>.............................] - ETA: 15s - loss: 0.1575 - accuracy: 0.9479
 1500/25000 [>.............................] - ETA: 14s - loss: 0.1550 - accuracy: 0.9493
 1600/25000 [>.............................] - ETA: 14s - loss: 0.1539 - accuracy: 0.9500
 1700/25000 [=>............................] - ETA: 14s - loss: 0.1512 - accuracy: 0.9506
 1800/25000 [=>............................] - ETA: 14s - loss: 0.1490 - accuracy: 0.9511
 1900/25000 [=>............................] - ETA: 14s - loss: 0.1532 - accuracy: 0.9495
 2000/25000 [=>............................] - ETA: 14s - loss: 0.1557 - accuracy: 0.9485
 2100/25000 [=>............................] - ETA: 14s - loss: 0.1588 - accuracy: 0.9467
 2200/25000 [=>............................] - ETA: 14s - loss: 0.1585 - accuracy: 0.9459
 2300/25000 [=>............................] - ETA: 14s - loss: 0.1628 - accuracy: 0.9443
 2400/25000 [=>............................] - ETA: 14s - loss: 0.1604 - accuracy: 0.9450
 2500/25000 [==>...........................] - ETA: 14s - loss: 0.1587 - accuracy: 0.9448
 2600/25000 [==>...........................] - ETA: 14s - loss: 0.1585 - accuracy: 0.9442
 2700/25000 [==>...........................] - ETA: 14s - loss: 0.1636 - accuracy: 0.9426
 2800/25000 [==>...........................] - ETA: 14s - loss: 0.1667 - accuracy: 0.9411
 2900/25000 [==>...........................] - ETA: 14s - loss: 0.1647 - accuracy: 0.9414
 3000/25000 [==>...........................] - ETA: 13s - loss: 0.1637 - accuracy: 0.9423
 3100/25000 [==>...........................] - ETA: 13s - loss: 0.1657 - accuracy: 0.9410
 3200/25000 [==>...........................] - ETA: 13s - loss: 0.1666 - accuracy: 0.9413
 3300/25000 [==>...........................] - ETA: 13s - loss: 0.1671 - accuracy: 0.9409
 3400/25000 [===>..........................] - ETA: 13s - loss: 0.1662 - accuracy: 0.9412
 3500/25000 [===>..........................] - ETA: 13s - loss: 0.1662 - accuracy: 0.9414
 3600/25000 [===>..........................] - ETA: 13s - loss: 0.1655 - accuracy: 0.9417
 3700/25000 [===>..........................] - ETA: 13s - loss: 0.1645 - accuracy: 0.9419
 3800/25000 [===>..........................] - ETA: 13s - loss: 0.1662 - accuracy: 0.9416
 3900/25000 [===>..........................] - ETA: 13s - loss: 0.1697 - accuracy: 0.9403
 4000/25000 [===>..........................] - ETA: 13s - loss: 0.1691 - accuracy: 0.9405
 4100/25000 [===>..........................] - ETA: 13s - loss: 0.1700 - accuracy: 0.9405
 4200/25000 [====>.........................] - ETA: 13s - loss: 0.1699 - accuracy: 0.9405
 4300/25000 [====>.........................] - ETA: 13s - loss: 0.1708 - accuracy: 0.9400
 4400/25000 [====>.........................] - ETA: 12s - loss: 0.1702 - accuracy: 0.9400
 4500/25000 [====>.........................] - ETA: 12s - loss: 0.1694 - accuracy: 0.9407
 4600/25000 [====>.........................] - ETA: 12s - loss: 0.1684 - accuracy: 0.9411
 4700/25000 [====>.........................] - ETA: 12s - loss: 0.1693 - accuracy: 0.9406
 4800/25000 [====>.........................] - ETA: 12s - loss: 0.1679 - accuracy: 0.9410
 4900/25000 [====>.........................] - ETA: 12s - loss: 0.1680 - accuracy: 0.9410
 5000/25000 [=====>........................] - ETA: 12s - loss: 0.1685 - accuracy: 0.9404
 5100/25000 [=====>........................] - ETA: 12s - loss: 0.1672 - accuracy: 0.9410
 5200/25000 [=====>........................] - ETA: 12s - loss: 0.1668 - accuracy: 0.9408
 5300/25000 [=====>........................] - ETA: 12s - loss: 0.1685 - accuracy: 0.9406
 5400/25000 [=====>........................] - ETA: 12s - loss: 0.1682 - accuracy: 0.9404
 5500/25000 [=====>........................] - ETA: 12s - loss: 0.1676 - accuracy: 0.9405
 5600/25000 [=====>........................] - ETA: 12s - loss: 0.1677 - accuracy: 0.9407
 5700/25000 [=====>........................] - ETA: 12s - loss: 0.1672 - accuracy: 0.9407
 5800/25000 [=====>........................] - ETA: 12s - loss: 0.1663 - accuracy: 0.9410
 5900/25000 [======>.......................] - ETA: 12s - loss: 0.1658 - accuracy: 0.9414
 6000/25000 [======>.......................] - ETA: 11s - loss: 0.1666 - accuracy: 0.9413
 6100/25000 [======>.......................] - ETA: 11s - loss: 0.1655 - accuracy: 0.9418
 6200/25000 [======>.......................] - ETA: 11s - loss: 0.1653 - accuracy: 0.9421
 6300/25000 [======>.......................] - ETA: 11s - loss: 0.1650 - accuracy: 0.9421
 6400/25000 [======>.......................] - ETA: 11s - loss: 0.1650 - accuracy: 0.9423
 6500/25000 [======>.......................] - ETA: 11s - loss: 0.1662 - accuracy: 0.9420
 6600/25000 [======>.......................] - ETA: 11s - loss: 0.1662 - accuracy: 0.9421
 6700/25000 [=======>......................] - ETA: 11s - loss: 0.1656 - accuracy: 0.9421
 6800/25000 [=======>......................] - ETA: 11s - loss: 0.1666 - accuracy: 0.9416
 6900/25000 [=======>......................] - ETA: 11s - loss: 0.1673 - accuracy: 0.9412
 7000/25000 [=======>......................] - ETA: 11s - loss: 0.1659 - accuracy: 0.9419
 7100/25000 [=======>......................] - ETA: 11s - loss: 0.1669 - accuracy: 0.9415
 7200/25000 [=======>......................] - ETA: 11s - loss: 0.1667 - accuracy: 0.9415
 7300/25000 [=======>......................] - ETA: 11s - loss: 0.1668 - accuracy: 0.9414
 7400/25000 [=======>......................] - ETA: 11s - loss: 0.1669 - accuracy: 0.9415
 7500/25000 [========>.....................] - ETA: 10s - loss: 0.1659 - accuracy: 0.9420
 7600/25000 [========>.....................] - ETA: 10s - loss: 0.1657 - accuracy: 0.9422
 7700/25000 [========>.....................] - ETA: 10s - loss: 0.1668 - accuracy: 0.9418
 7800/25000 [========>.....................] - ETA: 10s - loss: 0.1670 - accuracy: 0.9417
 7900/25000 [========>.....................] - ETA: 10s - loss: 0.1665 - accuracy: 0.9415
 8000/25000 [========>.....................] - ETA: 10s - loss: 0.1669 - accuracy: 0.9411
 8100/25000 [========>.....................] - ETA: 10s - loss: 0.1680 - accuracy: 0.9406
 8200/25000 [========>.....................] - ETA: 10s - loss: 0.1690 - accuracy: 0.9401
 8300/25000 [========>.....................] - ETA: 10s - loss: 0.1691 - accuracy: 0.9404
 8400/25000 [=========>....................] - ETA: 10s - loss: 0.1694 - accuracy: 0.9406
 8500/25000 [=========>....................] - ETA: 10s - loss: 0.1702 - accuracy: 0.9401
 8600/25000 [=========>....................] - ETA: 10s - loss: 0.1712 - accuracy: 0.9400
 8700/25000 [=========>....................] - ETA: 10s - loss: 0.1717 - accuracy: 0.9395
 8800/25000 [=========>....................] - ETA: 10s - loss: 0.1709 - accuracy: 0.9398
 8900/25000 [=========>....................] - ETA: 10s - loss: 0.1712 - accuracy: 0.9396
 9000/25000 [=========>....................] - ETA: 10s - loss: 0.1717 - accuracy: 0.9392
 9100/25000 [=========>....................] - ETA: 9s - loss: 0.1716 - accuracy: 0.9390 
 9200/25000 [==========>...................] - ETA: 9s - loss: 0.1728 - accuracy: 0.9386
 9300/25000 [==========>...................] - ETA: 9s - loss: 0.1729 - accuracy: 0.9384
 9400/25000 [==========>...................] - ETA: 9s - loss: 0.1730 - accuracy: 0.9382
 9500/25000 [==========>...................] - ETA: 9s - loss: 0.1735 - accuracy: 0.9380
 9600/25000 [==========>...................] - ETA: 9s - loss: 0.1737 - accuracy: 0.9380
 9700/25000 [==========>...................] - ETA: 9s - loss: 0.1730 - accuracy: 0.9384
 9800/25000 [==========>...................] - ETA: 9s - loss: 0.1727 - accuracy: 0.9384
 9900/25000 [==========>...................] - ETA: 9s - loss: 0.1732 - accuracy: 0.9381
10000/25000 [===========>..................] - ETA: 9s - loss: 0.1734 - accuracy: 0.9379
10100/25000 [===========>..................] - ETA: 9s - loss: 0.1741 - accuracy: 0.9375
10200/25000 [===========>..................] - ETA: 9s - loss: 0.1732 - accuracy: 0.9379
10300/25000 [===========>..................] - ETA: 9s - loss: 0.1749 - accuracy: 0.9376
10400/25000 [===========>..................] - ETA: 9s - loss: 0.1760 - accuracy: 0.9373
10500/25000 [===========>..................] - ETA: 9s - loss: 0.1759 - accuracy: 0.9374
10600/25000 [===========>..................] - ETA: 9s - loss: 0.1766 - accuracy: 0.9373
10700/25000 [===========>..................] - ETA: 8s - loss: 0.1767 - accuracy: 0.9372
10800/25000 [===========>..................] - ETA: 8s - loss: 0.1774 - accuracy: 0.9368
10900/25000 [============>.................] - ETA: 8s - loss: 0.1779 - accuracy: 0.9364
11000/25000 [============>.................] - ETA: 8s - loss: 0.1787 - accuracy: 0.9361
11100/25000 [============>.................] - ETA: 8s - loss: 0.1788 - accuracy: 0.9360
11200/25000 [============>.................] - ETA: 8s - loss: 0.1794 - accuracy: 0.9359
11300/25000 [============>.................] - ETA: 8s - loss: 0.1796 - accuracy: 0.9360
11400/25000 [============>.................] - ETA: 8s - loss: 0.1792 - accuracy: 0.9361
11500/25000 [============>.................] - ETA: 8s - loss: 0.1791 - accuracy: 0.9362
11600/25000 [============>.................] - ETA: 8s - loss: 0.1793 - accuracy: 0.9361
11700/25000 [=============>................] - ETA: 8s - loss: 0.1796 - accuracy: 0.9358
11800/25000 [=============>................] - ETA: 8s - loss: 0.1795 - accuracy: 0.9358
11900/25000 [=============>................] - ETA: 8s - loss: 0.1792 - accuracy: 0.9360
12000/25000 [=============>................] - ETA: 8s - loss: 0.1798 - accuracy: 0.9357
12100/25000 [=============>................] - ETA: 8s - loss: 0.1802 - accuracy: 0.9355
12200/25000 [=============>................] - ETA: 8s - loss: 0.1804 - accuracy: 0.9353
12300/25000 [=============>................] - ETA: 7s - loss: 0.1801 - accuracy: 0.9354
12400/25000 [=============>................] - ETA: 7s - loss: 0.1806 - accuracy: 0.9349
12500/25000 [==============>...............] - ETA: 7s - loss: 0.1808 - accuracy: 0.9349
12600/25000 [==============>...............] - ETA: 7s - loss: 0.1809 - accuracy: 0.9349
12700/25000 [==============>...............] - ETA: 7s - loss: 0.1808 - accuracy: 0.9348
12800/25000 [==============>...............] - ETA: 7s - loss: 0.1810 - accuracy: 0.9348
12900/25000 [==============>...............] - ETA: 7s - loss: 0.1816 - accuracy: 0.9347
13000/25000 [==============>...............] - ETA: 7s - loss: 0.1818 - accuracy: 0.9345
13100/25000 [==============>...............] - ETA: 7s - loss: 0.1822 - accuracy: 0.9341
13200/25000 [==============>...............] - ETA: 7s - loss: 0.1819 - accuracy: 0.9342
13300/25000 [==============>...............] - ETA: 7s - loss: 0.1817 - accuracy: 0.9343
13400/25000 [===============>..............] - ETA: 7s - loss: 0.1818 - accuracy: 0.9344
13500/25000 [===============>..............] - ETA: 7s - loss: 0.1822 - accuracy: 0.9341
13600/25000 [===============>..............] - ETA: 7s - loss: 0.1819 - accuracy: 0.9343
13700/25000 [===============>..............] - ETA: 7s - loss: 0.1817 - accuracy: 0.9344
13800/25000 [===============>..............] - ETA: 7s - loss: 0.1820 - accuracy: 0.9343
13900/25000 [===============>..............] - ETA: 6s - loss: 0.1820 - accuracy: 0.9341
14000/25000 [===============>..............] - ETA: 6s - loss: 0.1815 - accuracy: 0.9343
14100/25000 [===============>..............] - ETA: 6s - loss: 0.1816 - accuracy: 0.9343
14200/25000 [================>.............] - ETA: 6s - loss: 0.1826 - accuracy: 0.9340
14300/25000 [================>.............] - ETA: 6s - loss: 0.1822 - accuracy: 0.9341
14400/25000 [================>.............] - ETA: 6s - loss: 0.1822 - accuracy: 0.9339
14500/25000 [================>.............] - ETA: 6s - loss: 0.1824 - accuracy: 0.9337
14600/25000 [================>.............] - ETA: 6s - loss: 0.1819 - accuracy: 0.9339
14700/25000 [================>.............] - ETA: 6s - loss: 0.1827 - accuracy: 0.9335
14800/25000 [================>.............] - ETA: 6s - loss: 0.1827 - accuracy: 0.9334
14900/25000 [================>.............] - ETA: 6s - loss: 0.1823 - accuracy: 0.9334
15000/25000 [=================>............] - ETA: 6s - loss: 0.1825 - accuracy: 0.9333
15100/25000 [=================>............] - ETA: 6s - loss: 0.1823 - accuracy: 0.9333
15200/25000 [=================>............] - ETA: 6s - loss: 0.1822 - accuracy: 0.9334
15300/25000 [=================>............] - ETA: 6s - loss: 0.1824 - accuracy: 0.9333
15400/25000 [=================>............] - ETA: 6s - loss: 0.1821 - accuracy: 0.9334
15500/25000 [=================>............] - ETA: 5s - loss: 0.1823 - accuracy: 0.9333
15600/25000 [=================>............] - ETA: 5s - loss: 0.1829 - accuracy: 0.9331
15700/25000 [=================>............] - ETA: 5s - loss: 0.1829 - accuracy: 0.9329
15800/25000 [=================>............] - ETA: 5s - loss: 0.1832 - accuracy: 0.9327
15900/25000 [==================>...........] - ETA: 5s - loss: 0.1831 - accuracy: 0.9329
16000/25000 [==================>...........] - ETA: 5s - loss: 0.1836 - accuracy: 0.9326
16100/25000 [==================>...........] - ETA: 5s - loss: 0.1834 - accuracy: 0.9326
16200/25000 [==================>...........] - ETA: 5s - loss: 0.1830 - accuracy: 0.9327
16300/25000 [==================>...........] - ETA: 5s - loss: 0.1831 - accuracy: 0.9325
16400/25000 [==================>...........] - ETA: 5s - loss: 0.1830 - accuracy: 0.9324
16500/25000 [==================>...........] - ETA: 5s - loss: 0.1827 - accuracy: 0.9324
16600/25000 [==================>...........] - ETA: 5s - loss: 0.1831 - accuracy: 0.9323
16700/25000 [===================>..........] - ETA: 5s - loss: 0.1825 - accuracy: 0.9325
16800/25000 [===================>..........] - ETA: 5s - loss: 0.1829 - accuracy: 0.9324
16900/25000 [===================>..........] - ETA: 5s - loss: 0.1831 - accuracy: 0.9324
17000/25000 [===================>..........] - ETA: 5s - loss: 0.1836 - accuracy: 0.9322
17100/25000 [===================>..........] - ETA: 4s - loss: 0.1836 - accuracy: 0.9323
17200/25000 [===================>..........] - ETA: 4s - loss: 0.1844 - accuracy: 0.9322
17300/25000 [===================>..........] - ETA: 4s - loss: 0.1846 - accuracy: 0.9321
17400/25000 [===================>..........] - ETA: 4s - loss: 0.1846 - accuracy: 0.9320
17500/25000 [====================>.........] - ETA: 4s - loss: 0.1850 - accuracy: 0.9319
17600/25000 [====================>.........] - ETA: 4s - loss: 0.1852 - accuracy: 0.9317
17700/25000 [====================>.........] - ETA: 4s - loss: 0.1854 - accuracy: 0.9316
17800/25000 [====================>.........] - ETA: 4s - loss: 0.1855 - accuracy: 0.9317
17900/25000 [====================>.........] - ETA: 4s - loss: 0.1854 - accuracy: 0.9317
18000/25000 [====================>.........] - ETA: 4s - loss: 0.1852 - accuracy: 0.9317
18100/25000 [====================>.........] - ETA: 4s - loss: 0.1851 - accuracy: 0.9317
18200/25000 [====================>.........] - ETA: 4s - loss: 0.1855 - accuracy: 0.9315
18300/25000 [====================>.........] - ETA: 4s - loss: 0.1855 - accuracy: 0.9315
18400/25000 [=====================>........] - ETA: 4s - loss: 0.1855 - accuracy: 0.9314
18500/25000 [=====================>........] - ETA: 4s - loss: 0.1852 - accuracy: 0.9315
18600/25000 [=====================>........] - ETA: 4s - loss: 0.1853 - accuracy: 0.9316
18700/25000 [=====================>........] - ETA: 3s - loss: 0.1850 - accuracy: 0.9317
18800/25000 [=====================>........] - ETA: 3s - loss: 0.1851 - accuracy: 0.9315
18900/25000 [=====================>........] - ETA: 3s - loss: 0.1855 - accuracy: 0.9314
19000/25000 [=====================>........] - ETA: 3s - loss: 0.1853 - accuracy: 0.9315
19100/25000 [=====================>........] - ETA: 3s - loss: 0.1859 - accuracy: 0.9313
19200/25000 [======================>.......] - ETA: 3s - loss: 0.1857 - accuracy: 0.9314
19300/25000 [======================>.......] - ETA: 3s - loss: 0.1863 - accuracy: 0.9312
19400/25000 [======================>.......] - ETA: 3s - loss: 0.1861 - accuracy: 0.9312
19500/25000 [======================>.......] - ETA: 3s - loss: 0.1865 - accuracy: 0.9310
19600/25000 [======================>.......] - ETA: 3s - loss: 0.1865 - accuracy: 0.9309
19700/25000 [======================>.......] - ETA: 3s - loss: 0.1867 - accuracy: 0.9308
19800/25000 [======================>.......] - ETA: 3s - loss: 0.1868 - accuracy: 0.9308
19900/25000 [======================>.......] - ETA: 3s - loss: 0.1873 - accuracy: 0.9305
20000/25000 [=======================>......] - ETA: 3s - loss: 0.1872 - accuracy: 0.9305
20100/25000 [=======================>......] - ETA: 3s - loss: 0.1875 - accuracy: 0.9303
20200/25000 [=======================>......] - ETA: 3s - loss: 0.1874 - accuracy: 0.9304
20300/25000 [=======================>......] - ETA: 2s - loss: 0.1876 - accuracy: 0.9303
20400/25000 [=======================>......] - ETA: 2s - loss: 0.1874 - accuracy: 0.9302
20500/25000 [=======================>......] - ETA: 2s - loss: 0.1876 - accuracy: 0.9303
20600/25000 [=======================>......] - ETA: 2s - loss: 0.1878 - accuracy: 0.9300
20700/25000 [=======================>......] - ETA: 2s - loss: 0.1879 - accuracy: 0.9300
20800/25000 [=======================>......] - ETA: 2s - loss: 0.1879 - accuracy: 0.9300
20900/25000 [========================>.....] - ETA: 2s - loss: 0.1881 - accuracy: 0.9298
21000/25000 [========================>.....] - ETA: 2s - loss: 0.1880 - accuracy: 0.9300
21100/25000 [========================>.....] - ETA: 2s - loss: 0.1876 - accuracy: 0.9301
21200/25000 [========================>.....] - ETA: 2s - loss: 0.1873 - accuracy: 0.9301
21300/25000 [========================>.....] - ETA: 2s - loss: 0.1873 - accuracy: 0.9301
21400/25000 [========================>.....] - ETA: 2s - loss: 0.1871 - accuracy: 0.9300
21500/25000 [========================>.....] - ETA: 2s - loss: 0.1870 - accuracy: 0.9301
21600/25000 [========================>.....] - ETA: 2s - loss: 0.1868 - accuracy: 0.9302
21700/25000 [=========================>....] - ETA: 2s - loss: 0.1881 - accuracy: 0.9298
21800/25000 [=========================>....] - ETA: 2s - loss: 0.1889 - accuracy: 0.9295
21900/25000 [=========================>....] - ETA: 1s - loss: 0.1888 - accuracy: 0.9295
22000/25000 [=========================>....] - ETA: 1s - loss: 0.1889 - accuracy: 0.9294
22100/25000 [=========================>....] - ETA: 1s - loss: 0.1893 - accuracy: 0.9294
22200/25000 [=========================>....] - ETA: 1s - loss: 0.1894 - accuracy: 0.9292
22300/25000 [=========================>....] - ETA: 1s - loss: 0.1894 - accuracy: 0.9293
22400/25000 [=========================>....] - ETA: 1s - loss: 0.1891 - accuracy: 0.9294
22500/25000 [==========================>...] - ETA: 1s - loss: 0.1893 - accuracy: 0.9292
22600/25000 [==========================>...] - ETA: 1s - loss: 0.1891 - accuracy: 0.9293
22700/25000 [==========================>...] - ETA: 1s - loss: 0.1891 - accuracy: 0.9293
22800/25000 [==========================>...] - ETA: 1s - loss: 0.1894 - accuracy: 0.9292
22900/25000 [==========================>...] - ETA: 1s - loss: 0.1895 - accuracy: 0.9292
23000/25000 [==========================>...] - ETA: 1s - loss: 0.1897 - accuracy: 0.9290
23100/25000 [==========================>...] - ETA: 1s - loss: 0.1895 - accuracy: 0.9292
23200/25000 [==========================>...] - ETA: 1s - loss: 0.1896 - accuracy: 0.9291
23300/25000 [==========================>...] - ETA: 1s - loss: 0.1896 - accuracy: 0.9291
23400/25000 [===========================>..] - ETA: 1s - loss: 0.1899 - accuracy: 0.9290
23500/25000 [===========================>..] - ETA: 0s - loss: 0.1903 - accuracy: 0.9289
23600/25000 [===========================>..] - ETA: 0s - loss: 0.1899 - accuracy: 0.9291
23700/25000 [===========================>..] - ETA: 0s - loss: 0.1899 - accuracy: 0.9291
23800/25000 [===========================>..] - ETA: 0s - loss: 0.1897 - accuracy: 0.9292
23900/25000 [===========================>..] - ETA: 0s - loss: 0.1898 - accuracy: 0.9291
24000/25000 [===========================>..] - ETA: 0s - loss: 0.1903 - accuracy: 0.9290
24100/25000 [===========================>..] - ETA: 0s - loss: 0.1907 - accuracy: 0.9288
24200/25000 [============================>.] - ETA: 0s - loss: 0.1906 - accuracy: 0.9289
24300/25000 [============================>.] - ETA: 0s - loss: 0.1907 - accuracy: 0.9289
24400/25000 [============================>.] - ETA: 0s - loss: 0.1908 - accuracy: 0.9287
24500/25000 [============================>.] - ETA: 0s - loss: 0.1910 - accuracy: 0.9286
24600/25000 [============================>.] - ETA: 0s - loss: 0.1913 - accuracy: 0.9285
24700/25000 [============================>.] - ETA: 0s - loss: 0.1917 - accuracy: 0.9282
24800/25000 [============================>.] - ETA: 0s - loss: 0.1919 - accuracy: 0.9281
24900/25000 [============================>.] - ETA: 0s - loss: 0.1918 - accuracy: 0.9281
25000/25000 [==============================] - 20s 789us/step - loss: 0.1920 - accuracy: 0.9281 - val_loss: 0.3305 - val_accuracy: 0.8633
	=====> Test the model: model.predict()
	Dataset: IMDB_REVIEWS
	Algorithm: Deep Learning using Keras 2 (KERAS_DL2)
	Training loss: 0.1437
	Training accuracy score: 95.68%
	Test loss: 0.3305
	Test accuracy score: 86.33%
	Training time: 61.3531
	Test time: 6.3131




FINAL CLASSIFICATION TABLE:

| ID | Dataset | Algorithm | Loss | Training accuracy score (%) | Test accuracy score (%) | Training time (seconds) | Test time (seconds) |
| --- | ------- | --------- | ---- | --------------------------- | ----------------------- | ----------------------- | ------------------- |
| 1 | TWENTY_NEWS_GROUPS | Deep Learning using Keras 1 (KERAS_DL1) | 0.0158 | 99.73 | 96.63 | 113.0123 | 3.1087 |
| 2 | IMDB_REVIEWS | Deep Learning using Keras 1 (KERAS_DL1) | 0.1381 | 96.22 | 88.36 | 17.0448 | 3.3398 |
| 3 | TWENTY_NEWS_GROUPS | Deep Learning using Keras 2 (KERAS_DL2) | 0.0595 | 97.89 | 96.08 | 133.2070 | 1.9305 |
| 4 | IMDB_REVIEWS | Deep Learning using Keras 2 (KERAS_DL2) | 0.1437 | 95.68 | 86.33 | 61.3531 | 6.3131 |


DONE!
Program finished. It took 453.72250390052795 seconds

Process finished with exit code 0
```