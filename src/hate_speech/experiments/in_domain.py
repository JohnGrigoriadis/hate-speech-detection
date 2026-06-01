"""In-domain experiment: train on OLID, test on OLID."""
import pandas as pd

from hate_speech.config import FIGURES_DIR, RANDOM_SEED, RESULTS_CSV, RESULTS_DIR, TRANSFORMER_MODELS
from hate_speech.data.loader import load_data
from hate_speech.data.preprocessor import preprocess_dataset
from hate_speech.evaluation.analysis import analyze_dataset_differences, detailed_error_analysis
from hate_speech.evaluation.metrics import comprehensive_evaluation, create_results_summary
from hate_speech.evaluation.visualization import plot_confusion_matrix
from hate_speech.features.tfidf import create_tfidf_features
from hate_speech.models.classical import train_conventional_models
from hate_speech.models.hybrid import train_lexicon_enhanced_model
from hate_speech.models.transformer import evaluate_transformer_model, train_transformer_model

import numpy as np


def run_in_domain() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Run all 6 models on the in-domain setup (OLID → OLID).

    Returns (results_df, predictions_dict).
    """
    print("=" * 80)
    print("IN-DOMAIN EXPERIMENT: Train on OLID, Test on OLID")
    print("=" * 80)

    olid_train, olid_test, _ = load_data()
    train = preprocess_dataset(olid_train)
    test  = preprocess_dataset(olid_test)

    all_results: list[dict] = []
    all_preds:   dict[str, np.ndarray] = {}

    # ── Transformer models ────────────────────────────────────────────────────
    for model_type, model_name in TRANSFORMER_MODELS.items():
        try:
            model = train_transformer_model(train, model_type, model_name)
            preds = evaluate_transformer_model(model, test)
        except Exception as exc:
            print(f"Skipping {model_name}: {exc}")
            continue

        label = model_name.split("-")[0].upper()
        metrics, cm, _ = comprehensive_evaluation(test["labels"], preds, label, "in-domain")
        all_results.append(metrics)
        all_preds[f"{label}_in_domain"] = preds
        plot_confusion_matrix(cm, label, "In-Domain", FIGURES_DIR)

        print(f"{label} In-Domain — Accuracy: {metrics['accuracy']:.4f}  "
              f"Macro F1: {metrics['macro_f1']:.4f}")

    # ── Classical ML models ───────────────────────────────────────────────────
    X_train, X_test, _ = create_tfidf_features(train, test)
    y_train = train["labels"]
    y_test  = test["labels"]

    conv_models = train_conventional_models(X_train, y_train)
    for name, model in conv_models.items():
        preds   = model.predict(X_test)
        metrics, cm, _ = comprehensive_evaluation(y_test, preds, name, "in-domain")
        all_results.append(metrics)
        all_preds[f"{name}_in_domain"] = preds
        plot_confusion_matrix(cm, name, "In-Domain", FIGURES_DIR)
        print(f"{name} In-Domain — Accuracy: {metrics['accuracy']:.4f}  "
              f"Macro F1: {metrics['macro_f1']:.4f}")

    # ── Lexicon-enhanced model ────────────────────────────────────────────────
    print("Training Lexicon-Enhanced model …")
    lex_model, X_test_lex = train_lexicon_enhanced_model(train, test)
    preds   = lex_model.predict(X_test_lex)
    metrics, cm, _ = comprehensive_evaluation(y_test, preds, "Lexicon-Enhanced", "in-domain")
    all_results.append(metrics)
    all_preds["Lexicon-Enhanced_in_domain"] = preds
    plot_confusion_matrix(cm, "Lexicon-Enhanced", "In-Domain", FIGURES_DIR)
    print(f"Lexicon-Enhanced In-Domain — Accuracy: {metrics['accuracy']:.4f}  "
          f"Macro F1: {metrics['macro_f1']:.4f}")

    # ── Error analysis ────────────────────────────────────────────────────────
    detailed_error_analysis(y_test, all_preds, test, "IN-DOMAIN SETUP")

    return create_results_summary(all_results), all_preds


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    results_df, _ = run_in_domain()
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"\nResults saved → {RESULTS_CSV}")


if __name__ == "__main__":
    main()
