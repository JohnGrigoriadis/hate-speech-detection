"""Central configuration — all paths, hyperparameters, and constants live here."""
from pathlib import Path

# ── Directory layout ──────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parents[2]
DATA_DIR    = ROOT_DIR / "data"
OUTPUTS_DIR = ROOT_DIR / "outputs"
RESULTS_DIR = OUTPUTS_DIR / "results"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# ── Dataset paths ─────────────────────────────────────────────────────────────
OLID_TRAIN_PATH  = DATA_DIR / "olid-train-small.csv"
OLID_TEST_PATH   = DATA_DIR / "olid-test.csv"
HASOC_TRAIN_PATH = DATA_DIR / "hasoc-train.csv"

# ── Output file paths ─────────────────────────────────────────────────────────
RESULTS_CSV = RESULTS_DIR / "results.csv"
DROPS_CSV   = RESULTS_DIR / "performance_drops.csv"

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── TF-IDF hyperparameters ────────────────────────────────────────────────────
TFIDF_CONFIG: dict = {
    "max_features": 10_000,
    "ngram_range":  (1, 2),
    "stop_words":   "english",
    "lowercase":    True,
    "max_df":       0.8,
    "min_df":       2,
    "sublinear_tf": True,
}

# ── Classical model hyperparameters ───────────────────────────────────────────
LR_CONFIG: dict = {
    "random_state": RANDOM_SEED,
    "max_iter":     1000,
    "class_weight": "balanced",
    "C":            1.0,
}

SVM_CONFIG: dict = {
    "kernel":       "linear",
    "random_state": RANDOM_SEED,
    "class_weight": "balanced",
    "probability":  True,
    "C":            1.0,
}

NB_CONFIG: dict = {
    "alpha": 1.0,
}

# ── Transformer model hyperparameters ─────────────────────────────────────────
TRANSFORMER_MODELS: dict[str, str] = {
    "bert":    "bert-base-uncased",
    "roberta": "roberta-base",
}

TRANSFORMER_ARGS: dict = {
    "num_train_epochs":                   3,
    "learning_rate":                      2e-5,
    "train_batch_size":                   32,
    "eval_batch_size":                    64,
    "max_seq_length":                     128,
    "gradient_accumulation_steps":        2,
    "warmup_steps":                       100,
    "save_model_at_epoch":                False,
    "save_steps":                         -1,
    "evaluate_during_training":           False,
    "fp16":                               True,   # overridden at runtime if no CUDA
    "dataloader_num_workers":             4,
    "reprocess_input_data":               True,
    "overwrite_output_dir":               True,
    "silent":                             True,
    "no_save":                            True,
    "use_multiprocessing":                True,
    "use_multiprocessing_for_evaluation": True,
    "multiprocessing_chunksize":          500,
    "process_count":                      2,
    "logging_steps":                      50,
    "manual_seed":                        RANDOM_SEED,
}

# ── Hate speech lexicon ───────────────────────────────────────────────────────
HATE_LEXICON: list[str] = [
    "hate", "stupid", "idiot", "kill", "die", "ugly", "fat", "loser", "dumb",
    "moron", "retard", "bitch", "ass", "damn", "hell", "shit", "fuck", "bloody",
    "crap", "suck", "worst", "terrible", "awful", "disgusting", "pathetic",
    "worthless", "trash", "garbage", "scum", "pig", "dog", "freak", "weirdo",
    "creep",
]

LEXICON_FEATURE_COLS: list[str] = [
    "hate_word_count",
    "hate_word_ratio",
    "contains_hate_words",
    "text_length_normalized",
    "exclamation_count",
    "question_count",
    "caps_ratio",
]

# ── Metrics reported in summary tables ────────────────────────────────────────
METRICS_TO_ANALYZE: list[str] = [
    "accuracy",
    "macro_f1",
    "macro_precision",
    "macro_recall",
]
