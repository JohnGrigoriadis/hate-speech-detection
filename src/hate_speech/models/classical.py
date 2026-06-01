"""Classical ML models: Logistic Regression, SVM, and Naive Bayes."""
import numpy as np
from scipy.sparse import spmatrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from hate_speech.config import LR_CONFIG, NB_CONFIG, SVM_CONFIG


def train_conventional_models(
    X_train: spmatrix, y_train: np.ndarray
) -> dict[str, LogisticRegression | SVC | MultinomialNB]:
    """Train Logistic Regression, SVM, and Naive Bayes on TF-IDF features.

    Returns a dict mapping model name → fitted estimator.
    """
    print("Training Logistic Regression...")
    lr = LogisticRegression(**LR_CONFIG)
    lr.fit(X_train, y_train)

    print("Training SVM...")
    svm = SVC(**SVM_CONFIG)
    svm.fit(X_train, y_train)

    print("Training Naive Bayes...")
    nb = MultinomialNB(**NB_CONFIG)
    nb.fit(X_train, y_train)

    return {
        "Logistic Regression": lr,
        "SVM":                 svm,
        "Naive Bayes":         nb,
    }
