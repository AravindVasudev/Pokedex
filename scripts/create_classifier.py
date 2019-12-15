#!/usr/bin/env python3

'''
This script creates the classifier, trains it, and dumps it to
`classifier.pickle` file.
'''
import pickle
import numpy as np
from time import time
from sklearn.svm import SVC
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
from load_dataset import load_dataset

i = 0
scores = 0

# Load the dataset
dataset = load_dataset()
X = dataset.x
y = dataset.y

# Split the train and test set using KFolds cross validation method
t = time()
kf = KFold(n_splits=4, shuffle=True, random_state=42)
for train_index, test_index in kf.split(X):
    i += 1
    print('KFold Iteration: {}'.format(i))
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Apply PCA and extract 80% of components
    print('  Computing PCA...')
    t0 = time()
    pca = PCA(n_components=.6, whiten=True).fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca  = pca.transform(X_test)
    print('  PCA Computed: {}'.format(time() - t0))

    # create multiple classifiers using GridSearchCV
    param_grid = {
        'kernel': ['rbf', 'linear'],
        'C': [1e3, 5e3, 1e4, 5e4, 1e5],
        'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]
    }
    clf = GridSearchCV(SVC(class_weight='balanced'), param_grid)

    # train the classifier
    print('  training...')
    t0 = time()
    clf = clf.fit(X_train_pca, y_train)
    print('  training time: {:0.3f}'.format(time() - t0))

    # Evaluate the classifier
    y_pred = clf.predict(X_test_pca)
    print(classification_report(y_test, y_pred))
    # print(confusion_matrix(y_test, y_pred))
    scores += clf.score(X_test_pca, y_test)

# dump the classifier to `classifier.pickle`
print('Dumping the classifier')
with open('classifier.pickle', 'wb') as clfFile:
    pickle.dump(clf, clfFile)

print('Total Average Score: {}'.format(scores / 4))
print('Done! Total Time Taken: {}'.format(time() - t))
