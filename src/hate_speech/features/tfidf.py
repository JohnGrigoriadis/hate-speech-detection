"""TF-IDF feature extraction."""
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer

from hate_speech.config import TFIDF_CONFIG


def create_tfidf_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[spmatrix, spmatrix, TfidfVectorizer]:
    """Fit a TF-IDF vectorizer on train_df and transform both splits.

    Returns (X_train, X_test, fitted_vectorizer).
    """
    tfidf = TfidfVectorizer(**TFIDF_CONFIG)
    X_train = tfidf.fit_transform(train_df["text_clean"])
    X_test  = tfidf.transform(test_df["text_clean"])
    return X_train, X_test, tfidf
