__author__ = 'manabchetia'

import matplotlib.pyplot as plt
import csv
from textblob import TextBlob
import pandas
import sklearn
import cPickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.learning_curve import learning_curve



def split_into_lemmas(message):
    message = unicode(message, 'utf8').lower()
    words = TextBlob(message).words
    return [word.lemma for word in words]

if __name__ == "__main__":
    filename = 'data/smsspamcollection/SMSSpamCollection'
    messages = pandas.read_csv(filename, sep='\t', names=["label", "message"])

    bow_transformer = CountVectorizer(analyzer=split_into_lemmas).fit(messages['message'])
    messages_bow = bow_transformer.transform(messages['message'])
    tfidf_transformer = TfidfTransformer().fit(messages_bow)
    messages_tfidf = tfidf_transformer.transform(messages_bow)

    spam_detector = MultinomialNB().fit(messages_tfidf, messages['label'])
    all_predictions = spam_detector.predict(messages_tfidf)
    print all_predictions

