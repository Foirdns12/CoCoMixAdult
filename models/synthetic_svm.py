import os

import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from data.synthetic import data, labels

PATH = os.path.dirname(os.path.abspath(__file__))


def train_model():
    clf = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', LinearSVC())
    ])
    clf.fit(data, labels)
    print(clf.score(data, labels))

    joblib.dump(clf, os.path.join(PATH, "synthetic_svm.joblib"))


def load_model():
    return joblib.load(os.path.join(PATH, "synthetic_svm.joblib"))


if __name__ == "__main__":
    train_model()
