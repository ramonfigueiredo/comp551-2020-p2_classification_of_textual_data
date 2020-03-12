## Two Deep Learning approaches using Keras: Loss, Training Accuracy, Test Accuracy, Training Time, Test Time

| ID | Dataset | Algorithm | Loss | Training accuracy score (%) | Test accuracy score (%) | Training time (seconds) | Test time (seconds) |
| --- | ------- | --------- | ---- | --------------------------- | ----------------------- | ----------------------- | ------------------- |
| 1 | IMDB_REVIEWS | Deep Learning using Keras 1 (KERAS_DL1) | 0.2374 | 90.11 | 89.10 | 33.8532 | 3.3294 |
| 2 | IMDB_REVIEWS | Deep Learning using Keras 2 (KERAS_DL2) | 0.2865 | 89.35 | 89.07 | 40.8770 | 6.0731 |

### Deep Learning using Keras 1 (KERAS_DL1)

![IMDB_REVIEWS: Loss, Training Accuracy, Test Accuracy, Training Time, Test Time](https://github.com/ramonfigueiredopessoa/comp551-2020-p2_classification_of_textual_data/blob/master/code/results/deep_learning_using_keras/results_imdb_using_multi_class_classification/best_number_of_epochs/KERAS_DL1_IMDB_REVIEWS_training_and_validation_accuracy_and_Loss.png)


### Learning using Keras 2 (KERAS_DL2)

![IMDB_REVIEWS: Loss, Training Accuracy, Test Accuracy, Training Time, Test Time](https://github.com/ramonfigueiredopessoa/comp551-2020-p2_classification_of_textual_data/blob/master/code/results/deep_learning_using_keras/results_imdb_using_multi_class_classification/best_number_of_epochs/KERAS_DL2_IMDB_REVIEWS_training_and_validation_accuracy_and_Loss.png)

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
python /comp551-2020-p2_classification_of_textual_data/code/main.py -dl -d IMDB_REVIEWS -imdb_multi_class
Using TensorFlow backend.
2020-03-12 14:51:25.404903: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libnvinfer.so.6'; dlerror: libnvinfer.so.6: cannot open shared object file: No such file or directory
2020-03-12 14:51:25.404956: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libnvinfer_plugin.so.6'; dlerror: libnvinfer_plugin.so.6: cannot open shared object file: No such file or directory
2020-03-12 14:51:25.404962: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:30] Cannot dlopen some TensorRT libraries. If you would like to use Nvidia GPU with TensorRT, please make sure the missing libraries mentioned above are installed properly.
[nltk_data] Downloading package wordnet to /home/ets-
[nltk_data]     crchum/nltk_data...
[nltk_data]   Package wordnet is already up-to-date!
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
	Number of epochs used by the deep learning approach. Default: None = use the best number of epochs for each dataset. =  None
	Deep Learning algorithm list (If dl_algorithm_list is not provided, all Deep Learning algorithms will be executed). Options of Deep Learning algorithms: 1) KERAS_DL1, 2) KERAS_DL2. = None
	Run grid search for all datasets (TWENTY_NEWS_GROUPS, IMDB_REVIEWS binary labels and IMDB_REVIEWS multi-class labels), and all 14 classifiers. Default: False (run scikit-learn algorithms or deep learning algorithms). Note: this takes many hours to execute. = False
==================================================================================================================================

Loading IMDB_REVIEWS dataset:
03/12/2020 02:51:25 PM - INFO - Program started...
03/12/2020 02:51:25 PM - INFO - Program started...

===> Reading files from /home/ets-crchum/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/train/neg

===> Reading files from /home/ets-crchum/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/train/pos

===> Reading files from /home/ets-crchum/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/test/neg

===> Reading files from /home/ets-crchum/github/comp551-2020-p2_classification_of_textual_data/code/datasets/imdb_reviews/aclImdb/test/pos
data loaded
25000 documents - 33.133MB (training set)
25000 documents - 32.351MB (test set)

Extracting features from the training data using a vectorizer
done in 2.941622s at 11.263MB/s
n_samples: 25000, n_features: 74170

Extracting features from the test data using the same vectorizer
done in 2.890129s at 11.194MB/s
n_samples: 25000, n_features: 74170

2020-03-12 14:51:32.986635: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
2020-03-12 14:51:33.000511: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-03-12 14:51:33.001091: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1555] Found device 0 with properties: 
pciBusID: 0000:01:00.0 name: GeForce GTX 1080 computeCapability: 6.1
coreClock: 1.7335GHz coreCount: 20 deviceMemorySize: 7.93GiB deviceMemoryBandwidth: 298.32GiB/s
2020-03-12 14:51:33.001152: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcudart.so.10.1'; dlerror: libcudart.so.10.1: cannot open shared object file: No such file or directory
2020-03-12 14:51:33.001194: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcublas.so.10'; dlerror: libcublas.so.10: cannot open shared object file: No such file or directory
2020-03-12 14:51:33.001233: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcufft.so.10'; dlerror: libcufft.so.10: cannot open shared object file: No such file or directory
2020-03-12 14:51:33.001272: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcurand.so.10'; dlerror: libcurand.so.10: cannot open shared object file: No such file or directory
2020-03-12 14:51:33.001311: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcusolver.so.10'; dlerror: libcusolver.so.10: cannot open shared object file: No such file or directory
2020-03-12 14:51:33.001349: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcusparse.so.10'; dlerror: libcusparse.so.10: cannot open shared object file: No such file or directory
2020-03-12 14:51:33.003291: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2020-03-12 14:51:33.003300: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1592] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
Skipping registering GPU devices...
2020-03-12 14:51:33.003471: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-03-12 14:51:33.024127: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 3600000000 Hz
2020-03-12 14:51:33.024645: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x8db4330 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-03-12 14:51:33.024663: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2020-03-12 14:51:33.095105: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-03-12 14:51:33.095811: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x8db5720 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2020-03-12 14:51:33.095822: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): GeForce GTX 1080, Compute Capability 6.1
2020-03-12 14:51:33.095919: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1096] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-03-12 14:51:33.095924: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1102]      
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 10)                741710    
_________________________________________________________________
dense_2 (Dense)              (None, 7)                 77        
=================================================================
Total params: 741,787
Trainable params: 741,787
Non-trainable params: 0
_________________________________________________________________
None


NUMBER OF EPOCHS USED: 2

	Dataset: IMDB_REVIEWS
	Algorithm: Deep Learning using Keras 1 (KERAS_DL1)
	Training loss: 0.2374
	Training accuracy score: 90.11%
	Test loss: 0.2937
	Test accuracy score: 89.10%
	Training time: 33.8532
	Test time: 3.3294


03/12/2020 02:52:13 PM - INFO - Program started...
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
	It took 26.19129753112793 seconds


Applying NLTK feature extraction: X_test
	==> Replace unused UNICODE characters. 
	==> Text in lower case. 
	==> Apply Lemmatize using WordNet's built-in morphy function. Returns the input word unchanged if it cannot be found in WordNet 
	==> Remove English stop words
	Done: NLTK feature extraction (replace unused UNICODE characters, text in lower case, apply Lemmatize using WordNet, remove English stop words)  finished!
	It took 24.12926197052002 seconds

	===> Tokenizer: fit_on_texts(X_train)
	===> X_train = pad_sequences(list_tokenized_train, maxlen=6000)
	===> Create Keras model
Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, None, 128)         768000    
_________________________________________________________________
bidirectional_1 (Bidirection (None, None, 64)          41216     
_________________________________________________________________
global_max_pooling1d_1 (Glob (None, 64)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 20)                1300      
_________________________________________________________________
dropout_1 (Dropout)          (None, 20)                0         
_________________________________________________________________
dense_4 (Dense)              (None, 7)                 147       
=================================================================
Total params: 810,663
Trainable params: 810,663
Non-trainable params: 0
_________________________________________________________________
None
	===> Tokenizer: fit_on_texts(X_test)
	===> X_test = pad_sequences(list_sentences_test, maxlen=6000)


NUMBER OF EPOCHS USED: 2

Train on 25000 samples, validate on 25000 samples
Epoch 1/2

  100/25000 [..............................] - ETA: 2:14 - loss: 0.7002 - accuracy: 0.3071
  200/25000 [..............................] - ETA: 1:15 - loss: 0.6977 - accuracy: 0.3321
  300/25000 [..............................] - ETA: 55s - loss: 0.6954 - accuracy: 0.3929 
  400/25000 [..............................] - ETA: 44s - loss: 0.6936 - accuracy: 0.4518
  500/25000 [..............................] - ETA: 38s - loss: 0.6915 - accuracy: 0.5057
  600/25000 [..............................] - ETA: 34s - loss: 0.6894 - accuracy: 0.5498
  700/25000 [..............................] - ETA: 31s - loss: 0.6874 - accuracy: 0.5820
  800/25000 [..............................] - ETA: 29s - loss: 0.6851 - accuracy: 0.6082
  900/25000 [>.............................] - ETA: 27s - loss: 0.6830 - accuracy: 0.6265
 1000/25000 [>.............................] - ETA: 26s - loss: 0.6807 - accuracy: 0.6421
 1100/25000 [>.............................] - ETA: 25s - loss: 0.6783 - accuracy: 0.6561
 1200/25000 [>.............................] - ETA: 24s - loss: 0.6763 - accuracy: 0.6646
 1300/25000 [>.............................] - ETA: 23s - loss: 0.6741 - accuracy: 0.6730
 1400/25000 [>.............................] - ETA: 22s - loss: 0.6719 - accuracy: 0.6797
 1500/25000 [>.............................] - ETA: 22s - loss: 0.6695 - accuracy: 0.6867
 1600/25000 [>.............................] - ETA: 21s - loss: 0.6668 - accuracy: 0.6924
 1700/25000 [=>............................] - ETA: 21s - loss: 0.6644 - accuracy: 0.6971
 1800/25000 [=>............................] - ETA: 20s - loss: 0.6618 - accuracy: 0.7010
 1900/25000 [=>............................] - ETA: 20s - loss: 0.6593 - accuracy: 0.7047
 2000/25000 [=>............................] - ETA: 20s - loss: 0.6562 - accuracy: 0.7086
 2100/25000 [=>............................] - ETA: 19s - loss: 0.6527 - accuracy: 0.7123
 2200/25000 [=>............................] - ETA: 19s - loss: 0.6497 - accuracy: 0.7148
 2300/25000 [=>............................] - ETA: 19s - loss: 0.6466 - accuracy: 0.7167
 2400/25000 [=>............................] - ETA: 18s - loss: 0.6431 - accuracy: 0.7188
 2500/25000 [==>...........................] - ETA: 18s - loss: 0.6401 - accuracy: 0.7202
 2600/25000 [==>...........................] - ETA: 18s - loss: 0.6362 - accuracy: 0.7225
 2700/25000 [==>...........................] - ETA: 18s - loss: 0.6323 - accuracy: 0.7243
 2800/25000 [==>...........................] - ETA: 17s - loss: 0.6285 - accuracy: 0.7261
 2900/25000 [==>...........................] - ETA: 17s - loss: 0.6250 - accuracy: 0.7270
 3000/25000 [==>...........................] - ETA: 17s - loss: 0.6211 - accuracy: 0.7285
 3100/25000 [==>...........................] - ETA: 17s - loss: 0.6170 - accuracy: 0.7298
 3200/25000 [==>...........................] - ETA: 17s - loss: 0.6130 - accuracy: 0.7314
 3300/25000 [==>...........................] - ETA: 16s - loss: 0.6090 - accuracy: 0.7331
 3400/25000 [===>..........................] - ETA: 16s - loss: 0.6044 - accuracy: 0.7347
 3500/25000 [===>..........................] - ETA: 16s - loss: 0.6011 - accuracy: 0.7356
 3600/25000 [===>..........................] - ETA: 16s - loss: 0.5972 - accuracy: 0.7367
 3700/25000 [===>..........................] - ETA: 16s - loss: 0.5933 - accuracy: 0.7380
 3800/25000 [===>..........................] - ETA: 16s - loss: 0.5894 - accuracy: 0.7391
 3900/25000 [===>..........................] - ETA: 15s - loss: 0.5852 - accuracy: 0.7407
 4000/25000 [===>..........................] - ETA: 15s - loss: 0.5812 - accuracy: 0.7420
 4100/25000 [===>..........................] - ETA: 15s - loss: 0.5776 - accuracy: 0.7450
 4200/25000 [====>.........................] - ETA: 15s - loss: 0.5739 - accuracy: 0.7484
 4300/25000 [====>.........................] - ETA: 15s - loss: 0.5702 - accuracy: 0.7513
 4400/25000 [====>.........................] - ETA: 15s - loss: 0.5668 - accuracy: 0.7540
 4500/25000 [====>.........................] - ETA: 15s - loss: 0.5632 - accuracy: 0.7567
 4600/25000 [====>.........................] - ETA: 15s - loss: 0.5598 - accuracy: 0.7593
 4700/25000 [====>.........................] - ETA: 14s - loss: 0.5566 - accuracy: 0.7617
 4800/25000 [====>.........................] - ETA: 14s - loss: 0.5536 - accuracy: 0.7639
 4900/25000 [====>.........................] - ETA: 14s - loss: 0.5497 - accuracy: 0.7665
 5000/25000 [=====>........................] - ETA: 14s - loss: 0.5469 - accuracy: 0.7682
 5100/25000 [=====>........................] - ETA: 14s - loss: 0.5440 - accuracy: 0.7703
 5200/25000 [=====>........................] - ETA: 14s - loss: 0.5409 - accuracy: 0.7723
 5300/25000 [=====>........................] - ETA: 14s - loss: 0.5381 - accuracy: 0.7743
 5400/25000 [=====>........................] - ETA: 14s - loss: 0.5351 - accuracy: 0.7761
 5500/25000 [=====>........................] - ETA: 14s - loss: 0.5322 - accuracy: 0.7780
 5600/25000 [=====>........................] - ETA: 13s - loss: 0.5300 - accuracy: 0.7795
 5700/25000 [=====>........................] - ETA: 13s - loss: 0.5274 - accuracy: 0.7810
 5800/25000 [=====>........................] - ETA: 13s - loss: 0.5249 - accuracy: 0.7826
 5900/25000 [======>.......................] - ETA: 13s - loss: 0.5223 - accuracy: 0.7844
 6000/25000 [======>.......................] - ETA: 13s - loss: 0.5199 - accuracy: 0.7860
 6100/25000 [======>.......................] - ETA: 13s - loss: 0.5177 - accuracy: 0.7876
 6200/25000 [======>.......................] - ETA: 13s - loss: 0.5157 - accuracy: 0.7890
 6300/25000 [======>.......................] - ETA: 13s - loss: 0.5132 - accuracy: 0.7906
 6400/25000 [======>.......................] - ETA: 13s - loss: 0.5108 - accuracy: 0.7921
 6500/25000 [======>.......................] - ETA: 13s - loss: 0.5088 - accuracy: 0.7935
 6600/25000 [======>.......................] - ETA: 12s - loss: 0.5066 - accuracy: 0.7950
 6700/25000 [=======>......................] - ETA: 12s - loss: 0.5045 - accuracy: 0.7964
 6800/25000 [=======>......................] - ETA: 12s - loss: 0.5024 - accuracy: 0.7978
 6900/25000 [=======>......................] - ETA: 12s - loss: 0.5003 - accuracy: 0.7992
 7000/25000 [=======>......................] - ETA: 12s - loss: 0.4986 - accuracy: 0.8004
 7100/25000 [=======>......................] - ETA: 12s - loss: 0.4968 - accuracy: 0.8016
 7200/25000 [=======>......................] - ETA: 12s - loss: 0.4950 - accuracy: 0.8027
 7300/25000 [=======>......................] - ETA: 12s - loss: 0.4932 - accuracy: 0.8039
 7400/25000 [=======>......................] - ETA: 12s - loss: 0.4916 - accuracy: 0.8050
 7500/25000 [========>.....................] - ETA: 12s - loss: 0.4903 - accuracy: 0.8059
 7600/25000 [========>.....................] - ETA: 12s - loss: 0.4888 - accuracy: 0.8068
 7700/25000 [========>.....................] - ETA: 12s - loss: 0.4873 - accuracy: 0.8078
 7800/25000 [========>.....................] - ETA: 11s - loss: 0.4860 - accuracy: 0.8087
 7900/25000 [========>.....................] - ETA: 11s - loss: 0.4845 - accuracy: 0.8096
 8000/25000 [========>.....................] - ETA: 11s - loss: 0.4828 - accuracy: 0.8106
 8100/25000 [========>.....................] - ETA: 11s - loss: 0.4812 - accuracy: 0.8116
 8200/25000 [========>.....................] - ETA: 11s - loss: 0.4797 - accuracy: 0.8126
 8300/25000 [========>.....................] - ETA: 11s - loss: 0.4780 - accuracy: 0.8136
 8400/25000 [=========>....................] - ETA: 11s - loss: 0.4768 - accuracy: 0.8143
 8500/25000 [=========>....................] - ETA: 11s - loss: 0.4755 - accuracy: 0.8151
 8600/25000 [=========>....................] - ETA: 11s - loss: 0.4743 - accuracy: 0.8159
 8700/25000 [=========>....................] - ETA: 11s - loss: 0.4729 - accuracy: 0.8167
 8800/25000 [=========>....................] - ETA: 11s - loss: 0.4717 - accuracy: 0.8174
 8900/25000 [=========>....................] - ETA: 11s - loss: 0.4702 - accuracy: 0.8182
 9000/25000 [=========>....................] - ETA: 10s - loss: 0.4690 - accuracy: 0.8189
 9100/25000 [=========>....................] - ETA: 10s - loss: 0.4678 - accuracy: 0.8196
 9200/25000 [==========>...................] - ETA: 10s - loss: 0.4667 - accuracy: 0.8203
 9300/25000 [==========>...................] - ETA: 10s - loss: 0.4654 - accuracy: 0.8211
 9400/25000 [==========>...................] - ETA: 10s - loss: 0.4644 - accuracy: 0.8218
 9500/25000 [==========>...................] - ETA: 10s - loss: 0.4632 - accuracy: 0.8225
 9600/25000 [==========>...................] - ETA: 10s - loss: 0.4619 - accuracy: 0.8232
 9700/25000 [==========>...................] - ETA: 10s - loss: 0.4612 - accuracy: 0.8237
 9800/25000 [==========>...................] - ETA: 10s - loss: 0.4600 - accuracy: 0.8243
 9900/25000 [==========>...................] - ETA: 10s - loss: 0.4591 - accuracy: 0.8249
10000/25000 [===========>..................] - ETA: 10s - loss: 0.4582 - accuracy: 0.8254
10100/25000 [===========>..................] - ETA: 10s - loss: 0.4571 - accuracy: 0.8261
10200/25000 [===========>..................] - ETA: 10s - loss: 0.4564 - accuracy: 0.8265
10300/25000 [===========>..................] - ETA: 9s - loss: 0.4554 - accuracy: 0.8271 
10400/25000 [===========>..................] - ETA: 9s - loss: 0.4545 - accuracy: 0.8277
10500/25000 [===========>..................] - ETA: 9s - loss: 0.4533 - accuracy: 0.8283
10600/25000 [===========>..................] - ETA: 9s - loss: 0.4525 - accuracy: 0.8288
10700/25000 [===========>..................] - ETA: 9s - loss: 0.4517 - accuracy: 0.8294
10800/25000 [===========>..................] - ETA: 9s - loss: 0.4509 - accuracy: 0.8299
10900/25000 [============>.................] - ETA: 9s - loss: 0.4499 - accuracy: 0.8304
11000/25000 [============>.................] - ETA: 9s - loss: 0.4492 - accuracy: 0.8308
11100/25000 [============>.................] - ETA: 9s - loss: 0.4484 - accuracy: 0.8313
11200/25000 [============>.................] - ETA: 9s - loss: 0.4476 - accuracy: 0.8317
11300/25000 [============>.................] - ETA: 9s - loss: 0.4469 - accuracy: 0.8322
11400/25000 [============>.................] - ETA: 9s - loss: 0.4461 - accuracy: 0.8327
11500/25000 [============>.................] - ETA: 9s - loss: 0.4451 - accuracy: 0.8332
11600/25000 [============>.................] - ETA: 9s - loss: 0.4443 - accuracy: 0.8337
11700/25000 [=============>................] - ETA: 8s - loss: 0.4439 - accuracy: 0.8341
11800/25000 [=============>................] - ETA: 8s - loss: 0.4430 - accuracy: 0.8346
11900/25000 [=============>................] - ETA: 8s - loss: 0.4424 - accuracy: 0.8350
12000/25000 [=============>................] - ETA: 8s - loss: 0.4416 - accuracy: 0.8354
12100/25000 [=============>................] - ETA: 8s - loss: 0.4409 - accuracy: 0.8359
12200/25000 [=============>................] - ETA: 8s - loss: 0.4404 - accuracy: 0.8362
12300/25000 [=============>................] - ETA: 8s - loss: 0.4398 - accuracy: 0.8366
12400/25000 [=============>................] - ETA: 8s - loss: 0.4390 - accuracy: 0.8371
12500/25000 [==============>...............] - ETA: 8s - loss: 0.4384 - accuracy: 0.8375
12600/25000 [==============>...............] - ETA: 8s - loss: 0.4378 - accuracy: 0.8378
12700/25000 [==============>...............] - ETA: 8s - loss: 0.4371 - accuracy: 0.8382
12800/25000 [==============>...............] - ETA: 8s - loss: 0.4366 - accuracy: 0.8386
12900/25000 [==============>...............] - ETA: 8s - loss: 0.4359 - accuracy: 0.8389
13000/25000 [==============>...............] - ETA: 8s - loss: 0.4352 - accuracy: 0.8394
13100/25000 [==============>...............] - ETA: 7s - loss: 0.4346 - accuracy: 0.8397
13200/25000 [==============>...............] - ETA: 7s - loss: 0.4340 - accuracy: 0.8401
13300/25000 [==============>...............] - ETA: 7s - loss: 0.4334 - accuracy: 0.8405
13400/25000 [===============>..............] - ETA: 7s - loss: 0.4329 - accuracy: 0.8408
13500/25000 [===============>..............] - ETA: 7s - loss: 0.4323 - accuracy: 0.8411
13600/25000 [===============>..............] - ETA: 7s - loss: 0.4317 - accuracy: 0.8415
13700/25000 [===============>..............] - ETA: 7s - loss: 0.4311 - accuracy: 0.8419
13800/25000 [===============>..............] - ETA: 7s - loss: 0.4308 - accuracy: 0.8421
13900/25000 [===============>..............] - ETA: 7s - loss: 0.4303 - accuracy: 0.8424
14000/25000 [===============>..............] - ETA: 7s - loss: 0.4297 - accuracy: 0.8427
14100/25000 [===============>..............] - ETA: 7s - loss: 0.4293 - accuracy: 0.8430
14200/25000 [================>.............] - ETA: 7s - loss: 0.4287 - accuracy: 0.8433
14300/25000 [================>.............] - ETA: 7s - loss: 0.4284 - accuracy: 0.8435
14400/25000 [================>.............] - ETA: 7s - loss: 0.4279 - accuracy: 0.8438
14500/25000 [================>.............] - ETA: 6s - loss: 0.4274 - accuracy: 0.8441
14600/25000 [================>.............] - ETA: 6s - loss: 0.4270 - accuracy: 0.8444
14700/25000 [================>.............] - ETA: 6s - loss: 0.4265 - accuracy: 0.8446
14800/25000 [================>.............] - ETA: 6s - loss: 0.4260 - accuracy: 0.8449
14900/25000 [================>.............] - ETA: 6s - loss: 0.4256 - accuracy: 0.8451
15000/25000 [=================>............] - ETA: 6s - loss: 0.4254 - accuracy: 0.8453
15100/25000 [=================>............] - ETA: 6s - loss: 0.4249 - accuracy: 0.8456
15200/25000 [=================>............] - ETA: 6s - loss: 0.4244 - accuracy: 0.8459
15300/25000 [=================>............] - ETA: 6s - loss: 0.4241 - accuracy: 0.8461
15400/25000 [=================>............] - ETA: 6s - loss: 0.4238 - accuracy: 0.8462
15500/25000 [=================>............] - ETA: 6s - loss: 0.4233 - accuracy: 0.8465
15600/25000 [=================>............] - ETA: 6s - loss: 0.4230 - accuracy: 0.8467
15700/25000 [=================>............] - ETA: 6s - loss: 0.4225 - accuracy: 0.8470
15800/25000 [=================>............] - ETA: 6s - loss: 0.4221 - accuracy: 0.8473
15900/25000 [==================>...........] - ETA: 6s - loss: 0.4217 - accuracy: 0.8475
16000/25000 [==================>...........] - ETA: 5s - loss: 0.4213 - accuracy: 0.8477
16100/25000 [==================>...........] - ETA: 5s - loss: 0.4210 - accuracy: 0.8479
16200/25000 [==================>...........] - ETA: 5s - loss: 0.4206 - accuracy: 0.8481
16300/25000 [==================>...........] - ETA: 5s - loss: 0.4201 - accuracy: 0.8484
16400/25000 [==================>...........] - ETA: 5s - loss: 0.4195 - accuracy: 0.8488
16500/25000 [==================>...........] - ETA: 5s - loss: 0.4192 - accuracy: 0.8490
16600/25000 [==================>...........] - ETA: 5s - loss: 0.4187 - accuracy: 0.8492
16700/25000 [===================>..........] - ETA: 5s - loss: 0.4184 - accuracy: 0.8494
16800/25000 [===================>..........] - ETA: 5s - loss: 0.4180 - accuracy: 0.8496
16900/25000 [===================>..........] - ETA: 5s - loss: 0.4176 - accuracy: 0.8498
17000/25000 [===================>..........] - ETA: 5s - loss: 0.4172 - accuracy: 0.8500
17100/25000 [===================>..........] - ETA: 5s - loss: 0.4169 - accuracy: 0.8502
17200/25000 [===================>..........] - ETA: 5s - loss: 0.4165 - accuracy: 0.8505
17300/25000 [===================>..........] - ETA: 5s - loss: 0.4162 - accuracy: 0.8507
17400/25000 [===================>..........] - ETA: 4s - loss: 0.4159 - accuracy: 0.8509
17500/25000 [====================>.........] - ETA: 4s - loss: 0.4156 - accuracy: 0.8511
17600/25000 [====================>.........] - ETA: 4s - loss: 0.4154 - accuracy: 0.8512
17700/25000 [====================>.........] - ETA: 4s - loss: 0.4151 - accuracy: 0.8514
17800/25000 [====================>.........] - ETA: 4s - loss: 0.4147 - accuracy: 0.8516
17900/25000 [====================>.........] - ETA: 4s - loss: 0.4144 - accuracy: 0.8518
18000/25000 [====================>.........] - ETA: 4s - loss: 0.4140 - accuracy: 0.8519
18100/25000 [====================>.........] - ETA: 4s - loss: 0.4137 - accuracy: 0.8521
18200/25000 [====================>.........] - ETA: 4s - loss: 0.4135 - accuracy: 0.8523
18300/25000 [====================>.........] - ETA: 4s - loss: 0.4131 - accuracy: 0.8525
18400/25000 [=====================>........] - ETA: 4s - loss: 0.4128 - accuracy: 0.8527
18500/25000 [=====================>........] - ETA: 4s - loss: 0.4124 - accuracy: 0.8528
18600/25000 [=====================>........] - ETA: 4s - loss: 0.4122 - accuracy: 0.8530
18700/25000 [=====================>........] - ETA: 4s - loss: 0.4119 - accuracy: 0.8531
18800/25000 [=====================>........] - ETA: 4s - loss: 0.4115 - accuracy: 0.8533
18900/25000 [=====================>........] - ETA: 3s - loss: 0.4111 - accuracy: 0.8535
19000/25000 [=====================>........] - ETA: 3s - loss: 0.4108 - accuracy: 0.8537
19100/25000 [=====================>........] - ETA: 3s - loss: 0.4104 - accuracy: 0.8539
19200/25000 [======================>.......] - ETA: 3s - loss: 0.4102 - accuracy: 0.8541
19300/25000 [======================>.......] - ETA: 3s - loss: 0.4098 - accuracy: 0.8543
19400/25000 [======================>.......] - ETA: 3s - loss: 0.4094 - accuracy: 0.8545
19500/25000 [======================>.......] - ETA: 3s - loss: 0.4091 - accuracy: 0.8547
19600/25000 [======================>.......] - ETA: 3s - loss: 0.4088 - accuracy: 0.8548
19700/25000 [======================>.......] - ETA: 3s - loss: 0.4085 - accuracy: 0.8550
19800/25000 [======================>.......] - ETA: 3s - loss: 0.4082 - accuracy: 0.8551
19900/25000 [======================>.......] - ETA: 3s - loss: 0.4078 - accuracy: 0.8554
20000/25000 [=======================>......] - ETA: 3s - loss: 0.4076 - accuracy: 0.8555
20100/25000 [=======================>......] - ETA: 3s - loss: 0.4074 - accuracy: 0.8556
20200/25000 [=======================>......] - ETA: 3s - loss: 0.4070 - accuracy: 0.8558
20300/25000 [=======================>......] - ETA: 3s - loss: 0.4067 - accuracy: 0.8560
20400/25000 [=======================>......] - ETA: 3s - loss: 0.4064 - accuracy: 0.8561
20500/25000 [=======================>......] - ETA: 2s - loss: 0.4063 - accuracy: 0.8562
20600/25000 [=======================>......] - ETA: 2s - loss: 0.4061 - accuracy: 0.8563
20700/25000 [=======================>......] - ETA: 2s - loss: 0.4058 - accuracy: 0.8565
20800/25000 [=======================>......] - ETA: 2s - loss: 0.4054 - accuracy: 0.8567
20900/25000 [========================>.....] - ETA: 2s - loss: 0.4051 - accuracy: 0.8569
21000/25000 [========================>.....] - ETA: 2s - loss: 0.4049 - accuracy: 0.8570
21100/25000 [========================>.....] - ETA: 2s - loss: 0.4046 - accuracy: 0.8571
21200/25000 [========================>.....] - ETA: 2s - loss: 0.4044 - accuracy: 0.8572
21300/25000 [========================>.....] - ETA: 2s - loss: 0.4041 - accuracy: 0.8574
21400/25000 [========================>.....] - ETA: 2s - loss: 0.4038 - accuracy: 0.8575
21500/25000 [========================>.....] - ETA: 2s - loss: 0.4035 - accuracy: 0.8576
21600/25000 [========================>.....] - ETA: 2s - loss: 0.4033 - accuracy: 0.8578
21700/25000 [=========================>....] - ETA: 2s - loss: 0.4031 - accuracy: 0.8579
21800/25000 [=========================>....] - ETA: 2s - loss: 0.4029 - accuracy: 0.8580
21900/25000 [=========================>....] - ETA: 2s - loss: 0.4027 - accuracy: 0.8581
22000/25000 [=========================>....] - ETA: 1s - loss: 0.4024 - accuracy: 0.8583
22100/25000 [=========================>....] - ETA: 1s - loss: 0.4022 - accuracy: 0.8584
22200/25000 [=========================>....] - ETA: 1s - loss: 0.4020 - accuracy: 0.8585
22300/25000 [=========================>....] - ETA: 1s - loss: 0.4018 - accuracy: 0.8586
22400/25000 [=========================>....] - ETA: 1s - loss: 0.4017 - accuracy: 0.8587
22500/25000 [==========================>...] - ETA: 1s - loss: 0.4015 - accuracy: 0.8588
22600/25000 [==========================>...] - ETA: 1s - loss: 0.4012 - accuracy: 0.8589
22700/25000 [==========================>...] - ETA: 1s - loss: 0.4010 - accuracy: 0.8591
22800/25000 [==========================>...] - ETA: 1s - loss: 0.4009 - accuracy: 0.8591
22900/25000 [==========================>...] - ETA: 1s - loss: 0.4006 - accuracy: 0.8593
23000/25000 [==========================>...] - ETA: 1s - loss: 0.4004 - accuracy: 0.8594
23100/25000 [==========================>...] - ETA: 1s - loss: 0.4002 - accuracy: 0.8595
23200/25000 [==========================>...] - ETA: 1s - loss: 0.4000 - accuracy: 0.8596
23300/25000 [==========================>...] - ETA: 1s - loss: 0.3998 - accuracy: 0.8597
23400/25000 [===========================>..] - ETA: 1s - loss: 0.3995 - accuracy: 0.8598
23500/25000 [===========================>..] - ETA: 0s - loss: 0.3992 - accuracy: 0.8599
23600/25000 [===========================>..] - ETA: 0s - loss: 0.3990 - accuracy: 0.8600
23700/25000 [===========================>..] - ETA: 0s - loss: 0.3988 - accuracy: 0.8602
23800/25000 [===========================>..] - ETA: 0s - loss: 0.3986 - accuracy: 0.8603
23900/25000 [===========================>..] - ETA: 0s - loss: 0.3984 - accuracy: 0.8604
24000/25000 [===========================>..] - ETA: 0s - loss: 0.3982 - accuracy: 0.8605
24100/25000 [===========================>..] - ETA: 0s - loss: 0.3980 - accuracy: 0.8606
24200/25000 [============================>.] - ETA: 0s - loss: 0.3978 - accuracy: 0.8607
24300/25000 [============================>.] - ETA: 0s - loss: 0.3975 - accuracy: 0.8609
24400/25000 [============================>.] - ETA: 0s - loss: 0.3972 - accuracy: 0.8610
24500/25000 [============================>.] - ETA: 0s - loss: 0.3970 - accuracy: 0.8611
24600/25000 [============================>.] - ETA: 0s - loss: 0.3968 - accuracy: 0.8612
24700/25000 [============================>.] - ETA: 0s - loss: 0.3966 - accuracy: 0.8613
24800/25000 [============================>.] - ETA: 0s - loss: 0.3965 - accuracy: 0.8614
24900/25000 [============================>.] - ETA: 0s - loss: 0.3963 - accuracy: 0.8615
25000/25000 [==============================] - 20s 813us/step - loss: 0.3961 - accuracy: 0.8615 - val_loss: 0.3440 - val_accuracy: 0.8858
Epoch 2/2

  100/25000 [..............................] - ETA: 15s - loss: 0.3691 - accuracy: 0.8757
  200/25000 [..............................] - ETA: 15s - loss: 0.3657 - accuracy: 0.8807
  300/25000 [..............................] - ETA: 15s - loss: 0.3553 - accuracy: 0.8843
  400/25000 [..............................] - ETA: 15s - loss: 0.3553 - accuracy: 0.8843
  500/25000 [..............................] - ETA: 15s - loss: 0.3577 - accuracy: 0.8831
  600/25000 [..............................] - ETA: 15s - loss: 0.3594 - accuracy: 0.8821
  700/25000 [..............................] - ETA: 15s - loss: 0.3589 - accuracy: 0.8829
  800/25000 [..............................] - ETA: 15s - loss: 0.3592 - accuracy: 0.8829
  900/25000 [>.............................] - ETA: 15s - loss: 0.3568 - accuracy: 0.8840
 1000/25000 [>.............................] - ETA: 15s - loss: 0.3581 - accuracy: 0.8836
 1100/25000 [>.............................] - ETA: 15s - loss: 0.3605 - accuracy: 0.8822
 1200/25000 [>.............................] - ETA: 15s - loss: 0.3601 - accuracy: 0.8824
 1300/25000 [>.............................] - ETA: 14s - loss: 0.3593 - accuracy: 0.8824
 1400/25000 [>.............................] - ETA: 14s - loss: 0.3581 - accuracy: 0.8830
 1500/25000 [>.............................] - ETA: 14s - loss: 0.3562 - accuracy: 0.8834
 1600/25000 [>.............................] - ETA: 14s - loss: 0.3537 - accuracy: 0.8842
 1700/25000 [=>............................] - ETA: 14s - loss: 0.3525 - accuracy: 0.8845
 1800/25000 [=>............................] - ETA: 14s - loss: 0.3526 - accuracy: 0.8845
 1900/25000 [=>............................] - ETA: 14s - loss: 0.3529 - accuracy: 0.8846
 2000/25000 [=>............................] - ETA: 14s - loss: 0.3517 - accuracy: 0.8851
 2100/25000 [=>............................] - ETA: 14s - loss: 0.3520 - accuracy: 0.8848
 2200/25000 [=>............................] - ETA: 14s - loss: 0.3510 - accuracy: 0.8851
 2300/25000 [=>............................] - ETA: 14s - loss: 0.3510 - accuracy: 0.8852
 2400/25000 [=>............................] - ETA: 14s - loss: 0.3517 - accuracy: 0.8847
 2500/25000 [==>...........................] - ETA: 14s - loss: 0.3513 - accuracy: 0.8846
 2600/25000 [==>...........................] - ETA: 14s - loss: 0.3513 - accuracy: 0.8846
 2700/25000 [==>...........................] - ETA: 13s - loss: 0.3502 - accuracy: 0.8850
 2800/25000 [==>...........................] - ETA: 13s - loss: 0.3505 - accuracy: 0.8848
 2900/25000 [==>...........................] - ETA: 13s - loss: 0.3506 - accuracy: 0.8846
 3000/25000 [==>...........................] - ETA: 13s - loss: 0.3500 - accuracy: 0.8849
 3100/25000 [==>...........................] - ETA: 13s - loss: 0.3501 - accuracy: 0.8847
 3200/25000 [==>...........................] - ETA: 13s - loss: 0.3496 - accuracy: 0.8850
 3300/25000 [==>...........................] - ETA: 13s - loss: 0.3501 - accuracy: 0.8846
 3400/25000 [===>..........................] - ETA: 13s - loss: 0.3503 - accuracy: 0.8845
 3500/25000 [===>..........................] - ETA: 13s - loss: 0.3495 - accuracy: 0.8849
 3600/25000 [===>..........................] - ETA: 13s - loss: 0.3489 - accuracy: 0.8852
 3700/25000 [===>..........................] - ETA: 13s - loss: 0.3489 - accuracy: 0.8852
 3800/25000 [===>..........................] - ETA: 13s - loss: 0.3487 - accuracy: 0.8852
 3900/25000 [===>..........................] - ETA: 13s - loss: 0.3484 - accuracy: 0.8853
 4000/25000 [===>..........................] - ETA: 13s - loss: 0.3477 - accuracy: 0.8855
 4100/25000 [===>..........................] - ETA: 13s - loss: 0.3478 - accuracy: 0.8854
 4200/25000 [====>.........................] - ETA: 13s - loss: 0.3477 - accuracy: 0.8853
 4300/25000 [====>.........................] - ETA: 12s - loss: 0.3474 - accuracy: 0.8854
 4400/25000 [====>.........................] - ETA: 12s - loss: 0.3474 - accuracy: 0.8854
 4500/25000 [====>.........................] - ETA: 12s - loss: 0.3476 - accuracy: 0.8853
 4600/25000 [====>.........................] - ETA: 12s - loss: 0.3473 - accuracy: 0.8855
 4700/25000 [====>.........................] - ETA: 12s - loss: 0.3473 - accuracy: 0.8854
 4800/25000 [====>.........................] - ETA: 12s - loss: 0.3471 - accuracy: 0.8855
 4900/25000 [====>.........................] - ETA: 12s - loss: 0.3471 - accuracy: 0.8855
 5000/25000 [=====>........................] - ETA: 12s - loss: 0.3473 - accuracy: 0.8855
 5100/25000 [=====>........................] - ETA: 12s - loss: 0.3470 - accuracy: 0.8854
 5200/25000 [=====>........................] - ETA: 12s - loss: 0.3472 - accuracy: 0.8854
 5300/25000 [=====>........................] - ETA: 12s - loss: 0.3471 - accuracy: 0.8854
 5400/25000 [=====>........................] - ETA: 12s - loss: 0.3467 - accuracy: 0.8856
 5500/25000 [=====>........................] - ETA: 12s - loss: 0.3467 - accuracy: 0.8856
 5600/25000 [=====>........................] - ETA: 12s - loss: 0.3467 - accuracy: 0.8855
 5700/25000 [=====>........................] - ETA: 12s - loss: 0.3465 - accuracy: 0.8856
 5800/25000 [=====>........................] - ETA: 12s - loss: 0.3460 - accuracy: 0.8857
 5900/25000 [======>.......................] - ETA: 12s - loss: 0.3458 - accuracy: 0.8857
 6000/25000 [======>.......................] - ETA: 11s - loss: 0.3457 - accuracy: 0.8856
 6100/25000 [======>.......................] - ETA: 11s - loss: 0.3458 - accuracy: 0.8856
 6200/25000 [======>.......................] - ETA: 11s - loss: 0.3458 - accuracy: 0.8856
 6300/25000 [======>.......................] - ETA: 11s - loss: 0.3459 - accuracy: 0.8856
 6400/25000 [======>.......................] - ETA: 11s - loss: 0.3458 - accuracy: 0.8856
 6500/25000 [======>.......................] - ETA: 11s - loss: 0.3453 - accuracy: 0.8857
 6600/25000 [======>.......................] - ETA: 11s - loss: 0.3451 - accuracy: 0.8857
 6700/25000 [=======>......................] - ETA: 11s - loss: 0.3450 - accuracy: 0.8858
 6800/25000 [=======>......................] - ETA: 11s - loss: 0.3449 - accuracy: 0.8858
 6900/25000 [=======>......................] - ETA: 11s - loss: 0.3449 - accuracy: 0.8857
 7000/25000 [=======>......................] - ETA: 11s - loss: 0.3449 - accuracy: 0.8857
 7100/25000 [=======>......................] - ETA: 11s - loss: 0.3450 - accuracy: 0.8856
 7200/25000 [=======>......................] - ETA: 11s - loss: 0.3449 - accuracy: 0.8856
 7300/25000 [=======>......................] - ETA: 11s - loss: 0.3449 - accuracy: 0.8856
 7400/25000 [=======>......................] - ETA: 11s - loss: 0.3447 - accuracy: 0.8856
 7500/25000 [========>.....................] - ETA: 10s - loss: 0.3447 - accuracy: 0.8856
 7600/25000 [========>.....................] - ETA: 10s - loss: 0.3447 - accuracy: 0.8855
 7700/25000 [========>.....................] - ETA: 10s - loss: 0.3441 - accuracy: 0.8857
 7800/25000 [========>.....................] - ETA: 10s - loss: 0.3440 - accuracy: 0.8858
 7900/25000 [========>.....................] - ETA: 10s - loss: 0.3439 - accuracy: 0.8857
 8000/25000 [========>.....................] - ETA: 10s - loss: 0.3437 - accuracy: 0.8857
 8100/25000 [========>.....................] - ETA: 10s - loss: 0.3435 - accuracy: 0.8857
 8200/25000 [========>.....................] - ETA: 10s - loss: 0.3432 - accuracy: 0.8858
 8300/25000 [========>.....................] - ETA: 10s - loss: 0.3431 - accuracy: 0.8859
 8400/25000 [=========>....................] - ETA: 10s - loss: 0.3433 - accuracy: 0.8859
 8500/25000 [=========>....................] - ETA: 10s - loss: 0.3434 - accuracy: 0.8859
 8600/25000 [=========>....................] - ETA: 10s - loss: 0.3433 - accuracy: 0.8859
 8700/25000 [=========>....................] - ETA: 10s - loss: 0.3430 - accuracy: 0.8859
 8800/25000 [=========>....................] - ETA: 10s - loss: 0.3430 - accuracy: 0.8858
 8900/25000 [=========>....................] - ETA: 10s - loss: 0.3427 - accuracy: 0.8859
 9000/25000 [=========>....................] - ETA: 10s - loss: 0.3428 - accuracy: 0.8858
 9100/25000 [=========>....................] - ETA: 9s - loss: 0.3427 - accuracy: 0.8858 
 9200/25000 [==========>...................] - ETA: 9s - loss: 0.3425 - accuracy: 0.8858
 9300/25000 [==========>...................] - ETA: 9s - loss: 0.3427 - accuracy: 0.8858
 9400/25000 [==========>...................] - ETA: 9s - loss: 0.3425 - accuracy: 0.8858
 9500/25000 [==========>...................] - ETA: 9s - loss: 0.3426 - accuracy: 0.8858
 9600/25000 [==========>...................] - ETA: 9s - loss: 0.3426 - accuracy: 0.8857
 9700/25000 [==========>...................] - ETA: 9s - loss: 0.3425 - accuracy: 0.8858
 9800/25000 [==========>...................] - ETA: 9s - loss: 0.3424 - accuracy: 0.8858
 9900/25000 [==========>...................] - ETA: 9s - loss: 0.3424 - accuracy: 0.8857
10000/25000 [===========>..................] - ETA: 9s - loss: 0.3426 - accuracy: 0.8857
10100/25000 [===========>..................] - ETA: 9s - loss: 0.3425 - accuracy: 0.8856
10200/25000 [===========>..................] - ETA: 9s - loss: 0.3424 - accuracy: 0.8856
10300/25000 [===========>..................] - ETA: 9s - loss: 0.3424 - accuracy: 0.8856
10400/25000 [===========>..................] - ETA: 9s - loss: 0.3423 - accuracy: 0.8856
10500/25000 [===========>..................] - ETA: 9s - loss: 0.3420 - accuracy: 0.8856
10600/25000 [===========>..................] - ETA: 9s - loss: 0.3420 - accuracy: 0.8857
10700/25000 [===========>..................] - ETA: 8s - loss: 0.3422 - accuracy: 0.8856
10800/25000 [===========>..................] - ETA: 8s - loss: 0.3419 - accuracy: 0.8856
10900/25000 [============>.................] - ETA: 8s - loss: 0.3417 - accuracy: 0.8856
11000/25000 [============>.................] - ETA: 8s - loss: 0.3417 - accuracy: 0.8856
11100/25000 [============>.................] - ETA: 8s - loss: 0.3416 - accuracy: 0.8856
11200/25000 [============>.................] - ETA: 8s - loss: 0.3415 - accuracy: 0.8856
11300/25000 [============>.................] - ETA: 8s - loss: 0.3413 - accuracy: 0.8856
11400/25000 [============>.................] - ETA: 8s - loss: 0.3411 - accuracy: 0.8856
11500/25000 [============>.................] - ETA: 8s - loss: 0.3410 - accuracy: 0.8856
11600/25000 [============>.................] - ETA: 8s - loss: 0.3411 - accuracy: 0.8856
11700/25000 [=============>................] - ETA: 8s - loss: 0.3409 - accuracy: 0.8857
11800/25000 [=============>................] - ETA: 8s - loss: 0.3407 - accuracy: 0.8857
11900/25000 [=============>................] - ETA: 8s - loss: 0.3405 - accuracy: 0.8858
12000/25000 [=============>................] - ETA: 8s - loss: 0.3402 - accuracy: 0.8858
12100/25000 [=============>................] - ETA: 8s - loss: 0.3402 - accuracy: 0.8858
12200/25000 [=============>................] - ETA: 8s - loss: 0.3402 - accuracy: 0.8858
12300/25000 [=============>................] - ETA: 7s - loss: 0.3401 - accuracy: 0.8858
12400/25000 [=============>................] - ETA: 7s - loss: 0.3400 - accuracy: 0.8858
12500/25000 [==============>...............] - ETA: 7s - loss: 0.3399 - accuracy: 0.8858
12600/25000 [==============>...............] - ETA: 7s - loss: 0.3398 - accuracy: 0.8859
12700/25000 [==============>...............] - ETA: 7s - loss: 0.3398 - accuracy: 0.8859
12800/25000 [==============>...............] - ETA: 7s - loss: 0.3396 - accuracy: 0.8858
12900/25000 [==============>...............] - ETA: 7s - loss: 0.3395 - accuracy: 0.8858
13000/25000 [==============>...............] - ETA: 7s - loss: 0.3395 - accuracy: 0.8858
13100/25000 [==============>...............] - ETA: 7s - loss: 0.3396 - accuracy: 0.8857
13200/25000 [==============>...............] - ETA: 7s - loss: 0.3395 - accuracy: 0.8857
13300/25000 [==============>...............] - ETA: 7s - loss: 0.3394 - accuracy: 0.8857
13400/25000 [===============>..............] - ETA: 7s - loss: 0.3394 - accuracy: 0.8857
13500/25000 [===============>..............] - ETA: 7s - loss: 0.3392 - accuracy: 0.8858
13600/25000 [===============>..............] - ETA: 7s - loss: 0.3390 - accuracy: 0.8858
13700/25000 [===============>..............] - ETA: 7s - loss: 0.3389 - accuracy: 0.8858
13800/25000 [===============>..............] - ETA: 7s - loss: 0.3388 - accuracy: 0.8858
13900/25000 [===============>..............] - ETA: 6s - loss: 0.3388 - accuracy: 0.8858
14000/25000 [===============>..............] - ETA: 6s - loss: 0.3388 - accuracy: 0.8857
14100/25000 [===============>..............] - ETA: 6s - loss: 0.3386 - accuracy: 0.8858
14200/25000 [================>.............] - ETA: 6s - loss: 0.3386 - accuracy: 0.8858
14300/25000 [================>.............] - ETA: 6s - loss: 0.3387 - accuracy: 0.8857
14400/25000 [================>.............] - ETA: 6s - loss: 0.3386 - accuracy: 0.8857
14500/25000 [================>.............] - ETA: 6s - loss: 0.3386 - accuracy: 0.8857
14600/25000 [================>.............] - ETA: 6s - loss: 0.3383 - accuracy: 0.8857
14700/25000 [================>.............] - ETA: 6s - loss: 0.3383 - accuracy: 0.8857
14800/25000 [================>.............] - ETA: 6s - loss: 0.3382 - accuracy: 0.8857
14900/25000 [================>.............] - ETA: 6s - loss: 0.3380 - accuracy: 0.8858
15000/25000 [=================>............] - ETA: 6s - loss: 0.3380 - accuracy: 0.8857
15100/25000 [=================>............] - ETA: 6s - loss: 0.3379 - accuracy: 0.8858
15200/25000 [=================>............] - ETA: 6s - loss: 0.3378 - accuracy: 0.8858
15300/25000 [=================>............] - ETA: 6s - loss: 0.3377 - accuracy: 0.8858
15400/25000 [=================>............] - ETA: 6s - loss: 0.3376 - accuracy: 0.8858
15500/25000 [=================>............] - ETA: 5s - loss: 0.3375 - accuracy: 0.8857
15600/25000 [=================>............] - ETA: 5s - loss: 0.3374 - accuracy: 0.8857
15700/25000 [=================>............] - ETA: 5s - loss: 0.3372 - accuracy: 0.8858
15800/25000 [=================>............] - ETA: 5s - loss: 0.3370 - accuracy: 0.8858
15900/25000 [==================>...........] - ETA: 5s - loss: 0.3367 - accuracy: 0.8859
16000/25000 [==================>...........] - ETA: 5s - loss: 0.3367 - accuracy: 0.8859
16100/25000 [==================>...........] - ETA: 5s - loss: 0.3366 - accuracy: 0.8859
16200/25000 [==================>...........] - ETA: 5s - loss: 0.3365 - accuracy: 0.8859
16300/25000 [==================>...........] - ETA: 5s - loss: 0.3365 - accuracy: 0.8858
16400/25000 [==================>...........] - ETA: 5s - loss: 0.3362 - accuracy: 0.8859
16500/25000 [==================>...........] - ETA: 5s - loss: 0.3362 - accuracy: 0.8858
16600/25000 [==================>...........] - ETA: 5s - loss: 0.3361 - accuracy: 0.8858
16700/25000 [===================>..........] - ETA: 5s - loss: 0.3360 - accuracy: 0.8858
16800/25000 [===================>..........] - ETA: 5s - loss: 0.3359 - accuracy: 0.8858
16900/25000 [===================>..........] - ETA: 5s - loss: 0.3359 - accuracy: 0.8858
17000/25000 [===================>..........] - ETA: 5s - loss: 0.3357 - accuracy: 0.8858
17100/25000 [===================>..........] - ETA: 4s - loss: 0.3355 - accuracy: 0.8858
17200/25000 [===================>..........] - ETA: 4s - loss: 0.3355 - accuracy: 0.8858
17300/25000 [===================>..........] - ETA: 4s - loss: 0.3355 - accuracy: 0.8858
17400/25000 [===================>..........] - ETA: 4s - loss: 0.3353 - accuracy: 0.8858
17500/25000 [====================>.........] - ETA: 4s - loss: 0.3351 - accuracy: 0.8858
17600/25000 [====================>.........] - ETA: 4s - loss: 0.3348 - accuracy: 0.8859
17700/25000 [====================>.........] - ETA: 4s - loss: 0.3346 - accuracy: 0.8859
17800/25000 [====================>.........] - ETA: 4s - loss: 0.3344 - accuracy: 0.8860
17900/25000 [====================>.........] - ETA: 4s - loss: 0.3344 - accuracy: 0.8860
18000/25000 [====================>.........] - ETA: 4s - loss: 0.3340 - accuracy: 0.8861
18100/25000 [====================>.........] - ETA: 4s - loss: 0.3338 - accuracy: 0.8862
18200/25000 [====================>.........] - ETA: 4s - loss: 0.3337 - accuracy: 0.8862
18300/25000 [====================>.........] - ETA: 4s - loss: 0.3335 - accuracy: 0.8863
18400/25000 [=====================>........] - ETA: 4s - loss: 0.3335 - accuracy: 0.8863
18500/25000 [=====================>........] - ETA: 4s - loss: 0.3334 - accuracy: 0.8863
18600/25000 [=====================>........] - ETA: 4s - loss: 0.3331 - accuracy: 0.8863
18700/25000 [=====================>........] - ETA: 3s - loss: 0.3330 - accuracy: 0.8864
18800/25000 [=====================>........] - ETA: 3s - loss: 0.3329 - accuracy: 0.8864
18900/25000 [=====================>........] - ETA: 3s - loss: 0.3328 - accuracy: 0.8864
19000/25000 [=====================>........] - ETA: 3s - loss: 0.3327 - accuracy: 0.8864
19100/25000 [=====================>........] - ETA: 3s - loss: 0.3326 - accuracy: 0.8864
19200/25000 [======================>.......] - ETA: 3s - loss: 0.3326 - accuracy: 0.8864
19300/25000 [======================>.......] - ETA: 3s - loss: 0.3326 - accuracy: 0.8864
19400/25000 [======================>.......] - ETA: 3s - loss: 0.3325 - accuracy: 0.8864
19500/25000 [======================>.......] - ETA: 3s - loss: 0.3324 - accuracy: 0.8864
19600/25000 [======================>.......] - ETA: 3s - loss: 0.3322 - accuracy: 0.8864
19700/25000 [======================>.......] - ETA: 3s - loss: 0.3321 - accuracy: 0.8864
19800/25000 [======================>.......] - ETA: 3s - loss: 0.3320 - accuracy: 0.8864
19900/25000 [======================>.......] - ETA: 3s - loss: 0.3319 - accuracy: 0.8864
20000/25000 [=======================>......] - ETA: 3s - loss: 0.3318 - accuracy: 0.8864
20100/25000 [=======================>......] - ETA: 3s - loss: 0.3317 - accuracy: 0.8864
20200/25000 [=======================>......] - ETA: 3s - loss: 0.3317 - accuracy: 0.8864
20300/25000 [=======================>......] - ETA: 2s - loss: 0.3315 - accuracy: 0.8864
20400/25000 [=======================>......] - ETA: 2s - loss: 0.3315 - accuracy: 0.8864
20500/25000 [=======================>......] - ETA: 2s - loss: 0.3314 - accuracy: 0.8864
20600/25000 [=======================>......] - ETA: 2s - loss: 0.3312 - accuracy: 0.8864
20700/25000 [=======================>......] - ETA: 2s - loss: 0.3311 - accuracy: 0.8865
20800/25000 [=======================>......] - ETA: 2s - loss: 0.3310 - accuracy: 0.8865
20900/25000 [========================>.....] - ETA: 2s - loss: 0.3309 - accuracy: 0.8865
21000/25000 [========================>.....] - ETA: 2s - loss: 0.3308 - accuracy: 0.8865
21100/25000 [========================>.....] - ETA: 2s - loss: 0.3307 - accuracy: 0.8865
21200/25000 [========================>.....] - ETA: 2s - loss: 0.3307 - accuracy: 0.8864
21300/25000 [========================>.....] - ETA: 2s - loss: 0.3306 - accuracy: 0.8865
21400/25000 [========================>.....] - ETA: 2s - loss: 0.3304 - accuracy: 0.8865
21500/25000 [========================>.....] - ETA: 2s - loss: 0.3304 - accuracy: 0.8865
21600/25000 [========================>.....] - ETA: 2s - loss: 0.3303 - accuracy: 0.8866
21700/25000 [=========================>....] - ETA: 2s - loss: 0.3302 - accuracy: 0.8866
21800/25000 [=========================>....] - ETA: 2s - loss: 0.3301 - accuracy: 0.8866
21900/25000 [=========================>....] - ETA: 1s - loss: 0.3301 - accuracy: 0.8866
22000/25000 [=========================>....] - ETA: 1s - loss: 0.3299 - accuracy: 0.8866
22100/25000 [=========================>....] - ETA: 1s - loss: 0.3298 - accuracy: 0.8866
22200/25000 [=========================>....] - ETA: 1s - loss: 0.3296 - accuracy: 0.8866
22300/25000 [=========================>....] - ETA: 1s - loss: 0.3295 - accuracy: 0.8867
22400/25000 [=========================>....] - ETA: 1s - loss: 0.3293 - accuracy: 0.8867
22500/25000 [==========================>...] - ETA: 1s - loss: 0.3292 - accuracy: 0.8867
22600/25000 [==========================>...] - ETA: 1s - loss: 0.3291 - accuracy: 0.8868
22700/25000 [==========================>...] - ETA: 1s - loss: 0.3289 - accuracy: 0.8868
22800/25000 [==========================>...] - ETA: 1s - loss: 0.3288 - accuracy: 0.8868
22900/25000 [==========================>...] - ETA: 1s - loss: 0.3286 - accuracy: 0.8869
23000/25000 [==========================>...] - ETA: 1s - loss: 0.3285 - accuracy: 0.8869
23100/25000 [==========================>...] - ETA: 1s - loss: 0.3284 - accuracy: 0.8869
23200/25000 [==========================>...] - ETA: 1s - loss: 0.3283 - accuracy: 0.8869
23300/25000 [==========================>...] - ETA: 1s - loss: 0.3282 - accuracy: 0.8869
23400/25000 [===========================>..] - ETA: 1s - loss: 0.3280 - accuracy: 0.8869
23500/25000 [===========================>..] - ETA: 0s - loss: 0.3280 - accuracy: 0.8869
23600/25000 [===========================>..] - ETA: 0s - loss: 0.3278 - accuracy: 0.8870
23700/25000 [===========================>..] - ETA: 0s - loss: 0.3277 - accuracy: 0.8870
23800/25000 [===========================>..] - ETA: 0s - loss: 0.3275 - accuracy: 0.8871
23900/25000 [===========================>..] - ETA: 0s - loss: 0.3273 - accuracy: 0.8871
24000/25000 [===========================>..] - ETA: 0s - loss: 0.3272 - accuracy: 0.8872
24100/25000 [===========================>..] - ETA: 0s - loss: 0.3271 - accuracy: 0.8872
24200/25000 [============================>.] - ETA: 0s - loss: 0.3270 - accuracy: 0.8872
24300/25000 [============================>.] - ETA: 0s - loss: 0.3268 - accuracy: 0.8873
24400/25000 [============================>.] - ETA: 0s - loss: 0.3268 - accuracy: 0.8872
24500/25000 [============================>.] - ETA: 0s - loss: 0.3267 - accuracy: 0.8872
24600/25000 [============================>.] - ETA: 0s - loss: 0.3266 - accuracy: 0.8872
24700/25000 [============================>.] - ETA: 0s - loss: 0.3265 - accuracy: 0.8872
24800/25000 [============================>.] - ETA: 0s - loss: 0.3263 - accuracy: 0.8873
24900/25000 [============================>.] - ETA: 0s - loss: 0.3262 - accuracy: 0.8873
25000/25000 [==============================] - 20s 791us/step - loss: 0.3261 - accuracy: 0.8873 - val_loss: 0.2985 - val_accuracy: 0.8907
	=====> Test the model: model.predict()
	Dataset: IMDB_REVIEWS
	Algorithm: Deep Learning using Keras 2 (KERAS_DL2)
	Training loss: 0.2865
	Training accuracy score: 89.35%
	Test loss: 0.2985
	Test accuracy score: 89.07%
	Training time: 40.8770
	Test time: 6.0731




FINAL CLASSIFICATION TABLE:

| ID | Dataset | Algorithm | Loss | Training accuracy score (%) | Test accuracy score (%) | Training time (seconds) | Test time (seconds) |
| --- | ------- | --------- | ---- | --------------------------- | ----------------------- | ----------------------- | ------------------- |
| 1 | IMDB_REVIEWS | Deep Learning using Keras 1 (KERAS_DL1) | 0.2374 | 90.11 | 89.10 | 33.8532 | 3.3294 |
| 2 | IMDB_REVIEWS | Deep Learning using Keras 2 (KERAS_DL2) | 0.2865 | 89.35 | 89.07 | 40.8770 | 6.0731 |


DONE!
Program finished. It took 164.16750502586365 seconds

Process finished with exit code 0
```