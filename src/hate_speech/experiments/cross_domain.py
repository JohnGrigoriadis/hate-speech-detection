"""Cross-domain experiment: train on HASOC, test on OLID."""
import numpy as np
import pandas as pd

from hate_speech.config import FIGURES_DIR, RESULTS_DIR, TRANSFORMER_MODELS
from hate_speech.data.loader import load_data
from hate_speech.data.preprocessor import preprocess_dataset
from hate_speech.evaluation.analysis import detailed_error_analysis
from hate_speech.evaluation.metrics import comprehensive_evaluation, create_results_summary
from hate_speech.evaluation.visualization import plot_confusion_matrix
from hate_speech.features.tfidf import create_tfidf_features
from hate_speech.models.classical import train_conventional_models
from hate_speech.models.hybrid import train_lexicon_enhanced_model
from hate_speech.models.transformer import evaluate_transformer_model, train_transformer_model


def run_cross_domain() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Run all 6 models on the cross-domain setup (HASOC → OLID).

    Returns (results_df, predictions_dict).
    """
    print("=" * 80)
    print("CROSS-DOMAIN EXPERIMENT: Train on HASOC, Test on OLID")
    print("=" * 80)

    _, olid_test, hasoc_train = load_data()
    train = preprocess_dataset(hasoc_train)
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
        metrics, cm, _ = comprehensive_evaluation(test["labels"], preds, label, "cross-domain")
        all_results.append(metrics)
        all_preds[f"{label}_cross_domain"] = preds
        plot_confusion_matrix(cm, label, "Cross-Domain", FIGURES_DIR)
        print(f"{label} Cross-Domain — Accuracy: {metrics['accuracy']:.4f}  "
              f"Macro F1: {metrics['macro_f1']:.4f}")

    # ── Classical ML models ───────────────────────────────────────────────────
    X_train, X_test, _ = create_tfidf_features(train, test)
    y_train = train["labels"]
    y_test  = test["labels"]

    conv_models = train_conventional_models(X_train, y_train)
    for name, model in conv_models.items():
        preds   = model.predict(X_test)
        metrics, cm, _ = comprehensive_evaluation(y_test, preds, name, "cross-domain")
        all_results.append(metrics)
        all_preds[f"{name}_cross_domain"] = preds
        plot_confusion_matrix(cm, name, "Cross-Domain", FIGURES_DIR)
        print(f"{name} Cross-Domain — Accuracy: {metrics['accuracy']:.4f}  "
              f"Macro F1: {metrics['macro_f1']:.4f}")

    # ── Lexicon-enhanced model ────────────────────────────────────────────────
    print("Training Lexicon-Enhanced model …")
    lex_model, X_test_lex = train_lexicon_enhanced_model(train, test)
    preds   = lex_model.predict(X_test_lex)
    metrics, cm, _ = comprehensive_evaluation(y_test, preds, "Lexicon-Enhanced", "cross-domain")
    all_results.append(metrics)
    all_preds["Lexicon-Enhanced_cross_domain"] = preds
    plot_confusion_matrix(cm, "Lexicon-Enhanced", "Cross-Domain", FIGURES_DIR)
    print(f"Lexicon-Enhanced Cross-Domain — Accuracy: {metrics['accuracy']:.4f}  "
          f"Macro F1: {metrics['macro_f1']:.4f}")

    # ── Error analysis ────────────────────────────────────────────────────────
    detailed_error_analysis(y_test, all_preds, test, "CROSS-DOMAIN SETUP")

    return create_results_summary(all_results), all_preds


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    results_df, _ = run_cross_domain()
    out = RESULTS_DIR / "cross_domain_results.csv"
    results_df.to_csv(out, index=False)
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    main()
