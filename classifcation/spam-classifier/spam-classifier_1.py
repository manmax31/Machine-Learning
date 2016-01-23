__author__ = 'manabchetia'

# Spam classification using SVMs and Multinomial NBs

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

def get_data(filename):
    return [line.rstrip() for line in open(filename)]


def create_tfidf_training_data(data):
    y = []
    corpus = []
    for datum in data:
        sentence = datum.split('\t')
        y.append(sentence[0])
        corpus.append(sentence[1])
    vectoriser = TfidfVectorizer(min_df=1)
    X = vectoriser.fit_transform(corpus)
    return X, y


def train_svm(X, y):
    """
    Create and train the Support Vector Machine.
    """
    svm = SVC(C=1000000.0, gamma=0.0, kernel='rbf')
    svm.fit(X, y)
    return svm

def train_NB(X, y):
    naive_bayes = MultinomialNB()
    naive_bayes.fit(X, y)
    return naive_bayes

def train_KNN(X, y):
     knn = KNeighborsClassifier()
     knn.fit(X, y)
     return knn


def train_random_forest(X, y):
    rf = RandomForestClassifier()
    rf.fit(X, y)
    return rf

if __name__ == "__main__":
    filename = 'data/smsspamcollection/SMSSpamCollection'
    messages = get_data(filename)

    X, y = create_tfidf_training_data(messages)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training
    svm = train_svm(X_train, y_train)
    multin_nb = train_NB(X_train, y_train)
    knn = train_KNN(X, y)
    rf = train_random_forest(X, y)

    # Testing
    pred_svm = svm.predict(X_test)
    pred_nb  = multin_nb.predict(X_test)
    pred_knn = knn.predict(X_test)
    pref_rf = rf.predict(X_test)


    print("SVM:")
    print("Test accuracy: {}".format(svm.score(X_test, y_test)))
    print("Confusion Matrix: \n{}".format(confusion_matrix(pred_svm, y_test)))

    print("\nNaive Bayes:")
    print("Test accuracy: {}".format(accuracy_score(y_test, pred_nb)))
    print("Confusion Matrix: \n{}".format(confusion_matrix(pred_nb, y_test)))

    print("\nkNN:")
    print("Test accuracy: {}".format(accuracy_score(y_test, pred_knn)))
    print("Confusion Matrix: \n{}".format(confusion_matrix(pred_knn, y_test)))

    print("\nRandom Forest:")
    print("Test accuracy: {}".format(accuracy_score(y_test, pred_rf)))
    print("Confusion Matrix: \n{}".format(confusion_matrix(pred_rf, y_test)))