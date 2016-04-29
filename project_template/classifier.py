from collections import defaultdict
from collections import Counter
import csv
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF
from sklearn.datasets import fetch_20newsgroups
import re
import random
import numpy as np

num_test = 40
news = [];
with open('./project_template/data/news.txt', 'r') as f:
    news = f.read().splitlines();
random.shuffle(news)
    
opinions = [];
with open('./project_template/data/opinions.txt', 'r') as f:
    opinions = f.read().splitlines();
random.shuffle(opinions)
    
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(news + opinions)
y_train = [1] * len(news) + [2] * len(opinions)

###############################################################################
# Benchmark classifiers
def benchmark(clf_class, params, name):
    print("parameters:", params)
    t0 = time()
    clf = clf_class(**params).fit(X_train, y_train)
    print("done in %fs" % (time() - t0))

    if hasattr(clf, 'coef_'):
        print("Percentage of non zeros coef: %f"
              % (np.mean(clf.coef_ != 0) * 100))
    return clf
    #print("Predicting the outcomes of the testing set")
    #t0 = time()
    #pred = clf.predict(X_test)
    #print("done in %fs" % (time() - t0))

    #print("Classification report on test set for classifier:")
    #print(clf)
    #print()
    #print(classification_report(y_test, pred,
    #                            target_names=["news", "opinion"]))

    #cm = confusion_matrix(y_test, pred)
    #print("Confusion matrix:")
    #print(cm)

    # Show confusion matrix
    #pl.matshow(cm)
    #pl.title('Confusion matrix of the %s classifier' % name)
    #pl.colorbar()


#print("Testbenching a linear classifier...")
parameters = {
    'loss': 'hinge',
    'penalty': 'l2',
    'n_iter': 50,
    'alpha': 0.00001,
    'fit_intercept': True,
}

#benchmark(SGDClassifier, parameters, 'SGD')

print("Testbenching a MultinomialNB classifier...")
parameters = {'alpha': 0.01}

clf = benchmark(MultinomialNB, parameters, 'MultinomialNB')

def predict(ls):
    vec = vectorizer.transform([x["tweet"]["text"] for x in ls])
    return clf.predict(vec)

#pl.show()
