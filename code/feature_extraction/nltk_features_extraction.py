import re
from time import time

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

nltk.download("wordnet")


def clean_text(text):
    text = re.sub(r'[^\w\s]','',text, re.UNICODE)
    text = text.lower()
    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    text = [lemmatizer.lemmatize(token, "v") for token in text]
    text = [word for word in text if not word in stop_words]
    text = " ".join(text)
    return text


def apply_nltk_feature_extraction(X, options, label=''):
    start = time()
    print()
    print("Applying NLTK feature extraction: {}"
          "\n\t==> Replace unused UNICODE characters. "
          "\n\t==> Text in lower case. "
          "\n\t==> Apply Lemmatize using WordNet's built-in morphy function. "
          "Returns the input word unchanged if it cannot be found in WordNet "
          "\n\t==> Remove English stop words".format(label)
          )
    for i, text in enumerate(X):
        if options.verbose:
            print("TEXT BEFORE:\n", text)
        cleaned_text = clean_text(text)
        if options.verbose:
            print("TEXT BEFORE:\n", cleaned_text)
        X[i] = cleaned_text
    print("\tDone: NLTK feature extraction (replace unused UNICODE characters, text in lower case, "
          "apply Lemmatize using WordNet, remove English stop words)  finished!")
    print("\tIt took {} seconds".format(time() - start))
    print()
    return X
