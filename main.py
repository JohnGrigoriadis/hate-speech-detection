"""Entry point — runs both experiments and saves aggregated results."""
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

from hate_speech.config import DROPS_CSV, FIGURES_DIR, METRICS_TO_ANALYZE, RESULTS_CSV, RESULTS_DIR
from hate_speech.evaluation.metrics import compute_performance_drops
from hate_speech.evaluation.visualization import (
    plot_all_comparisons,
    plot_performance_comparison,
    plot_performance_drops,
)
from hate_speech.experiments.cross_domain import run_cross_domain
from hate_speech.experiments.in_domain import run_in_domain


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    in_df,    in_preds    = run_in_domain()
    cross_df, cross_preds = run_cross_domain()

    all_results = pd.concat([in_df, cross_df], ignore_index=True)
    all_results.to_csv(RESULTS_CSV, index=False)
    print(f"\nAll results saved → {RESULTS_CSV}")

    drop_df = compute_performance_drops(all_results)
    drop_df.to_csv(DROPS_CSV, index=False)
    print(f"Performance drops saved → {DROPS_CSV}")

    for metric in METRICS_TO_ANALYZE:
        plot_performance_comparison(all_results, metric, FIGURES_DIR)
        plot_performance_drops(drop_df, metric, FIGURES_DIR)

    plot_all_comparisons(all_results, FIGURES_DIR)

    print("\nDone. Outputs written to outputs/")


if __name__ == "__main__":
    main()
