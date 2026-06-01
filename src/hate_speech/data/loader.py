"""Dataset loading and exploration."""
import pandas as pd

from hate_speech.config import HASOC_TRAIN_PATH, OLID_TEST_PATH, OLID_TRAIN_PATH


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all three datasets from the data/ directory."""
    olid_train = pd.read_csv(OLID_TRAIN_PATH)
    olid_test  = pd.read_csv(OLID_TEST_PATH)
    hasoc_train = pd.read_csv(HASOC_TRAIN_PATH)
    return olid_train, olid_test, hasoc_train


def explore_data(df: pd.DataFrame, name: str) -> None:
    """Print comprehensive statistics for a dataset."""
    print(f"=== {name} Dataset ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nLabel distribution:")
    print(df["labels"].value_counts())
    print(f"Label proportions:")
    print(df["labels"].value_counts(normalize=True))

    text_lengths = df["text"].str.len()
    print(f"\nText length statistics:")
    print(f"  Average: {text_lengths.mean():.2f}")
    print(f"  Median:  {text_lengths.median():.2f}")
    print(f"  Min:     {text_lengths.min()}")
    print(f"  Max:     {text_lengths.max()}")

    print(f"\nSample texts:")
    for i, text in enumerate(df["text"].head(3).values):
        label = df["labels"].iloc[i]
        print(f"  Label {label}: {text}")
    print("-" * 70)
