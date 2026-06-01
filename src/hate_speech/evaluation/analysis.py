"""Error analysis and dataset characteristic comparisons."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from hate_speech.config import FIGURES_DIR
from hate_speech.features.lexicon import create_lexicon_features


def detailed_error_analysis(
    y_true: np.ndarray,
    predictions_dict: dict[str, np.ndarray],
    test_df: pd.DataFrame,
    analysis_name: str,
) -> None:
    """Print false-positive / false-negative counts and examples for every model."""
    print(f"\n{'=' * 60}")
    print(f"ERROR ANALYSIS: {analysis_name}")
    print(f"{'=' * 60}")

    for model_name, preds in predictions_dict.items():
        if preds is None:
            continue

        print(f"\n--- {model_name} ---")
        errors_df = test_df.copy()
        errors_df["predicted"] = preds
        errors_df["actual"]    = y_true
        errors_df["correct"]   = errors_df["predicted"] == errors_df["actual"]

        total_errors = (~errors_df["correct"]).sum()
        fp = ((errors_df["predicted"] == 1) & (errors_df["actual"] == 0)).sum()
        fn = ((errors_df["predicted"] == 0) & (errors_df["actual"] == 1)).sum()
        print(f"Total errors: {total_errors} ({total_errors / len(errors_df) * 100:.1f}%)")
        print(f"False positives: {fp}   False negatives: {fn}")

        if fp > 0:
            fp_rows = errors_df[(errors_df["predicted"] == 1) & (errors_df["actual"] == 0)]
            print("False Positive Examples (predicted offensive, actually not):")
            for i, (_, row) in enumerate(fp_rows.head(2).iterrows()):
                print(f"  {i+1}. {row['text'][:100]}…")

        if fn > 0:
            fn_rows = errors_df[(errors_df["predicted"] == 0) & (errors_df["actual"] == 1)]
            print("False Negative Examples (predicted not offensive, actually offensive):")
            for i, (_, row) in enumerate(fn_rows.head(2).iterrows()):
                print(f"  {i+1}. {row['text'][:100]}…")


def analyze_dataset_differences(
    olid_train: pd.DataFrame,
    hasoc_train: pd.DataFrame,
    olid_test: pd.DataFrame,
    save_dir=FIGURES_DIR,
) -> None:
    """Print vocabulary overlap and lexicon statistics; save comparison plots."""
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("DATASET CHARACTERISTICS ANALYSIS")
    print("=" * 80)

    datasets = {
        "OLID Train":  olid_train,
        "HASOC Train": hasoc_train,
        "OLID Test":   olid_test,
    }

    for name, df in datasets.items():
        vocab = set(" ".join(df["text_clean"]).lower().split())
        print(f"{name}: size={len(df)}  offensive={( df['labels']==1).mean()*100:.1f}%  "
              f"avg_len={df['text_clean'].str.len().mean():.1f}  vocab={len(vocab)}")

    olid_vocab  = set(" ".join(olid_train["text_clean"]).lower().split())
    hasoc_vocab = set(" ".join(hasoc_train["text_clean"]).lower().split())
    test_vocab  = set(" ".join(olid_test["text_clean"]).lower().split())

    overlap_oh = len(olid_vocab & hasoc_vocab)
    overlap_ot = len(olid_vocab & test_vocab)
    print(f"\nVocabulary Overlap:")
    print(f"  OLID–HASOC:      {overlap_oh} words "
          f"({overlap_oh / len(olid_vocab | hasoc_vocab) * 100:.1f}%)")
    print(f"  OLID Train–Test: {overlap_ot} words "
          f"({overlap_ot / len(olid_vocab | test_vocab) * 100:.1f}%)")

    # Lexicon feature comparison chart
    olid_lex  = create_lexicon_features(olid_train)
    hasoc_lex = create_lexicon_features(hasoc_train)

    x = [0, 1]
    plt.figure(figsize=(10, 6))
    plt.bar([i - 0.2 for i in x],
            [olid_lex["hate_word_count"].mean(), hasoc_lex["hate_word_count"].mean()],
            width=0.4, label="Avg Hate Words")
    plt.bar([i + 0.2 for i in x],
            [(olid_lex["hate_word_count"] > 0).mean() * 100,
             (hasoc_lex["hate_word_count"] > 0).mean() * 100],
            width=0.4, label="Texts with Hate Words (%)")
    plt.xticks(x, ["OLID Train", "HASOC Train"])
    plt.title("Lexicon Feature Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "lexicon_feature_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Lexicon feature comparison saved.")
