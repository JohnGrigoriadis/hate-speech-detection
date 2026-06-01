"""Lexicon-based feature extraction."""
import pandas as pd

from hate_speech.config import HATE_LEXICON, LEXICON_FEATURE_COLS


def create_lexicon_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 7 hand-crafted lexicon features to a copy of df."""
    out = df.copy()

    out["hate_word_count"] = out["text_clean"].apply(
        lambda x: sum(1 for word in HATE_LEXICON if word in x.lower().split())
    )
    out["word_count"] = out["text_clean"].apply(lambda x: len(x.split()))
    out["hate_word_ratio"] = out["hate_word_count"] / (out["word_count"] + 1)
    out["contains_hate_words"] = (out["hate_word_count"] > 0).astype(int)

    out["text_length"] = out["text_clean"].str.len()
    out["text_length_normalized"] = out["text_length"] / out["text_length"].max()

    out["exclamation_count"] = out["text_clean"].str.count("!")
    out["question_count"]    = out["text_clean"].str.count(r"\?")
    out["caps_ratio"] = out["text"].apply(
        lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1)
    )

    return out


__all__ = ["create_lexicon_features", "HATE_LEXICON", "LEXICON_FEATURE_COLS"]
