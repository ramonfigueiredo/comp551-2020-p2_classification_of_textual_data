from time import time

import numpy as np
from sklearn.feature_selection import SelectKBest, chi2


def select_k_best_using_chi2(X_train, y_train, X_test, feature_names, options):

    print("Extracting %d best features using the chi-squared test" %
          options.chi2_select)
    t0 = time()
    ch2 = SelectKBest(chi2, k=options.chi2_select)

    X_train = ch2.fit_transform(X_train, y_train)
    X_test = ch2.transform(X_test)

    if feature_names:
        # keep selected feature names
        feature_names = [feature_names[i] for i
                         in ch2.get_support(indices=True)]
    print("done in %fs" % (time() - t0))
    print()

    if feature_names:
        feature_names = np.asarray(feature_names)

    return X_train, X_test, feature_names
