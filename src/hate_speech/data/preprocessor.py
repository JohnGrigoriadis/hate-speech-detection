"""Text preprocessing for hate speech detection."""
import re

import pandas as pd


def preprocess_text(text: str) -> str:
    """Clean a single tweet: remove URLs, @USER mentions, special chars, lowercase."""
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@USER", " ", text)
    text = re.sub(r"[^\w\s.,!?@#]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.lower()
    return text


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply preprocessing to the 'text' column and drop empty results."""
    df_clean = df.copy()
    df_clean["text_clean"] = df_clean["text"].apply(preprocess_text)
    df_clean = df_clean[df_clean["text_clean"].str.len() > 0].reset_index(drop=True)
    return df_clean
