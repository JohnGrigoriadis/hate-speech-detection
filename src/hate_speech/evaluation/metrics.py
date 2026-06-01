"""Evaluation metrics and results aggregation."""
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from hate_speech.config import METRICS_TO_ANALYZE


def comprehensive_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    setup_type: str,
) -> tuple[dict, np.ndarray, dict]:
    """Compute macro/per-class metrics and confusion matrix.

    Returns (metrics_dict, confusion_matrix, full_report_dict).
    """
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm     = confusion_matrix(y_true, y_pred)

    metrics = {
        "model":            model_name,
        "setup":            setup_type,
        "macro_precision":  report["macro avg"]["precision"],
        "macro_recall":     report["macro avg"]["recall"],
        "macro_f1":         report["macro avg"]["f1-score"],
        "class_0_precision": report["0"]["precision"],
        "class_0_recall":    report["0"]["recall"],
        "class_0_f1":        report["0"]["f1-score"],
        "class_1_precision": report["1"]["precision"],
        "class_1_recall":    report["1"]["recall"],
        "class_1_f1":        report["1"]["f1-score"],
        "accuracy":          report["accuracy"],
        "support_0":         report["0"]["support"],
        "support_1":         report["1"]["support"],
    }

    return metrics, cm, report


def create_results_summary(all_results: list[dict]) -> pd.DataFrame:
    """Convert a list of metrics dicts into a tidy DataFrame and print it."""
    results_df = pd.DataFrame(all_results)
    display_cols = ["model", "setup", "accuracy", "macro_f1", "macro_precision", "macro_recall"]
    print("Results Summary:")
    print("=" * 100)
    print(results_df[display_cols].round(4).to_string(index=False))
    print("=" * 100)
    return results_df


def compute_performance_drops(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compute absolute and relative performance drops from in-domain to cross-domain."""
    in_domain    = results_df[results_df["setup"] == "in-domain"].copy()
    cross_domain = results_df[results_df["setup"] == "cross-domain"].copy()

    rows = []
    for model in in_domain["model"].unique():
        if model not in cross_domain["model"].values:
            continue
        in_row    = in_domain[in_domain["model"] == model].iloc[0]
        cross_row = cross_domain[cross_domain["model"] == model].iloc[0]
        row = {"model": model}
        for metric in METRICS_TO_ANALYZE:
            iv = in_row[metric]
            cv = cross_row[metric]
            row[f"{metric}_drop"]     = iv - cv
            row[f"{metric}_drop_pct"] = ((iv - cv) / iv * 100) if iv != 0 else 0
        rows.append(row)

    return pd.DataFrame(rows)
