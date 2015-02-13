__author__ = 'manabchetia'

# http://radimrehurek.com/data_science_python/?utm_source=Python+Weekly+Newsletter&utm_campaign=6fd5cf3287-Python_Weekly_Issue_178_February_12_2015&utm_medium=email&utm_term=0_9e26887fc5-6fd5cf3287-312712733

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


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


if __name__ == "__main__":
    filename = 'data/smsspamcollection/SMSSpamCollection'
    data = get_data(filename)

    X, y = create_tfidf_training_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    svm = train_svm(X_train, y_train)

    pred = svm.predict(X_test)

    print(X_test)

    print("Test accuracy: {}".format(svm.score(X_test, y_test)))
    print("Confusion Matrix: \n{}".format(confusion_matrix(pred, y_test)))