"""Plotting functions — confusion matrices, performance comparisons, drop charts."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from hate_speech.config import FIGURES_DIR, METRICS_TO_ANALYZE

plt.style.use("seaborn-v0_8")


def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    setup_type: str,
    save_dir: Path = FIGURES_DIR,
) -> None:
    """Save a labelled confusion-matrix heatmap as a high-resolution PNG."""
    save_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Non-Offensive (0)", "Offensive (1)"],
        yticklabels=["Non-Offensive (0)", "Offensive (1)"],
        cbar_kws={"label": "Count"},
    )
    plt.title(f"Confusion Matrix: {model_name} ({setup_type})", fontsize=14, pad=20)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()

    slug = f"{model_name.replace(' ', '_')}_{setup_type.replace('-', '_')}"
    path = save_dir / f"confusion_matrix_{slug}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    tn, fp, fn, tp = cm.ravel()
    print(f"[{model_name} | {setup_type}] TN={tn} FP={fp} FN={fn} TP={tp}  → saved {path.name}")


def plot_performance_comparison(
    results_df: pd.DataFrame,
    metric: str,
    save_dir: Path = FIGURES_DIR,
) -> None:
    """Bar chart comparing in-domain vs cross-domain for a single metric."""
    save_dir.mkdir(parents=True, exist_ok=True)

    title = metric.replace("_", " ").title()
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(data=results_df, x="model", y=metric, hue="setup")
    plt.title(f"{title} Comparison — In-Domain vs Cross-Domain", fontsize=16, pad=20)
    plt.xlabel("Model", fontsize=14)
    plt.ylabel(title, fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Setup", fontsize=12, title_fontsize=12)
    plt.grid(True, alpha=0.3, axis="y")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=10, padding=3)
    plt.tight_layout()

    path = save_dir / f"performance_comparison_{metric}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Performance comparison ({metric}) saved → {path.name}")


def plot_performance_drops(
    drop_df: pd.DataFrame,
    metric: str,
    save_dir: Path = FIGURES_DIR,
) -> None:
    """Bar chart of performance drop (in-domain → cross-domain) for one metric."""
    save_dir.mkdir(parents=True, exist_ok=True)

    col   = f"{metric}_drop"
    pct_col = f"{metric}_drop_pct"
    title = metric.replace("_", " ").title()

    plt.figure(figsize=(12, 8))
    ax = sns.barplot(data=drop_df, x="model", y=col)
    plt.title(f"{title} Drop (In-Domain → Cross-Domain)", fontsize=16, pad=20)
    plt.xlabel("Model", fontsize=14)
    plt.ylabel("Performance Drop", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, alpha=0.3, axis="y")

    for j, (_, row) in enumerate(drop_df.iterrows()):
        bar_h = row[col]
        offset = max(drop_df[col]) * 0.01
        plt.text(j, bar_h + offset, f"{row[pct_col]:.1f}%",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()
    path = save_dir / f"performance_drop_{metric}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Performance drop ({metric}) saved → {path.name}")


def plot_all_comparisons(
    results_df: pd.DataFrame, save_dir: Path = FIGURES_DIR
) -> None:
    """2×2 combined overview chart for all four metrics."""
    save_dir.mkdir(parents=True, exist_ok=True)

    metrics_info = [
        ("macro_f1",        "Macro F1-Score"),
        ("accuracy",        "Accuracy"),
        ("macro_precision", "Macro Precision"),
        ("macro_recall",    "Macro Recall"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    for idx, (metric, title) in enumerate(metrics_info):
        row, col = divmod(idx, 2)
        sns.barplot(data=results_df, x="model", y=metric, hue="setup", ax=axes[row][col])
        axes[row][col].set_title(f"{title} Comparison", fontsize=14)
        axes[row][col].set_xlabel("Model", fontsize=12)
        axes[row][col].set_ylabel(title, fontsize=12)
        axes[row][col].tick_params(axis="x", rotation=45)
        axes[row][col].grid(True, alpha=0.3, axis="y")
        for container in axes[row][col].containers:
            axes[row][col].bar_label(container, fmt="%.3f", fontsize=8, padding=2)

    plt.suptitle(
        "Comprehensive Performance Comparison: In-Domain vs Cross-Domain",
        fontsize=18, y=0.98,
    )
    plt.tight_layout()
    path = save_dir / "comprehensive_performance_comparison.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Comprehensive comparison saved → {path.name}")


def plot_dataset_distributions(
    datasets: dict[str, pd.DataFrame], save_dir: Path = FIGURES_DIR
) -> None:
    """Overlay text-length histograms and a label-distribution bar chart."""
    save_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    for name, df in datasets.items():
        plt.hist(df["text_clean"].str.len(), alpha=0.7, label=name, bins=30)
    plt.xlabel("Text Length (characters)")
    plt.ylabel("Frequency")
    plt.title("Text Length Distribution Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "text_length_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    label_rows = []
    for name, df in datasets.items():
        for label in [0, 1]:
            count = (df["labels"] == label).sum()
            label_rows.append({
                "Dataset": name, "Label": f"Label {label}",
                "Percentage": count / len(df) * 100,
            })

    plt.figure(figsize=(10, 6))
    sns.barplot(data=pd.DataFrame(label_rows), x="Dataset", y="Percentage", hue="Label")
    plt.title("Label Distribution Comparison")
    plt.ylabel("Percentage")
    plt.tight_layout()
    plt.savefig(save_dir / "label_distribution_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Dataset distribution plots saved.")
