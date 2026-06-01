"""Hybrid model: TF-IDF features combined with lexicon features."""
import pandas as pd
from scipy.sparse import hstack, spmatrix
from sklearn.linear_model import LogisticRegression

from hate_speech.config import LEXICON_FEATURE_COLS, RANDOM_SEED
from hate_speech.features.lexicon import create_lexicon_features
from hate_speech.features.tfidf import create_tfidf_features


def train_lexicon_enhanced_model(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[LogisticRegression, spmatrix]:
    """Train a Logistic Regression on stacked TF-IDF + lexicon features.

    Returns (fitted_model, X_test_combined).
    """
    train_enh = create_lexicon_features(train_df)
    test_enh  = create_lexicon_features(test_df)

    X_train_tfidf, X_test_tfidf, _ = create_tfidf_features(train_df, test_df)

    lex_train = train_enh[LEXICON_FEATURE_COLS].values
    lex_test  = test_enh[LEXICON_FEATURE_COLS].values

    X_train = hstack([X_train_tfidf, lex_train])
    X_test  = hstack([X_test_tfidf,  lex_test])

    model = LogisticRegression(
        random_state=RANDOM_SEED, max_iter=1000, class_weight="balanced"
    )
    model.fit(X_train, train_df["labels"])

    return model, X_test
