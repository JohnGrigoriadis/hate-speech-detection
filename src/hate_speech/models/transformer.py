"""Transformer-based models (BERT, RoBERTa) via simpletransformers."""
import logging
import multiprocessing as mp

import numpy as np
import pandas as pd
import psutil
import torch
from simpletransformers.classification import ClassificationModel

from hate_speech.config import TRANSFORMER_ARGS

logging.basicConfig(level=logging.INFO)
logging.getLogger("transformers").setLevel(logging.WARNING)


def _configure_args() -> dict:
    """Adjust multiprocessing settings based on available RAM and CPU cores."""
    args = TRANSFORMER_ARGS.copy()
    args["use_cuda"] = torch.cuda.is_available()
    args["fp16"]     = torch.cuda.is_available()

    cpu_count = mp.cpu_count()
    ram_gb    = psutil.virtual_memory().total / 1024 ** 3

    if ram_gb >= 12 and cpu_count >= 4:
        args["use_multiprocessing"]                = True
        args["use_multiprocessing_for_evaluation"] = True
        args["process_count"]                      = min(4, cpu_count - 1)
        args["dataloader_num_workers"]             = min(6, cpu_count)
        args["multiprocessing_chunksize"]          = 1000
    elif ram_gb >= 8 and cpu_count >= 2:
        args["use_multiprocessing"]                = True
        args["use_multiprocessing_for_evaluation"] = True
        args["process_count"]                      = 2
        args["dataloader_num_workers"]             = 4
        args["multiprocessing_chunksize"]          = 500
    else:
        args["use_multiprocessing"]                = False
        args["use_multiprocessing_for_evaluation"] = False
        args["process_count"]                      = 1
        args["dataloader_num_workers"]             = 2

    return args


def train_transformer_model(
    train_df: pd.DataFrame,
    model_type: str = "bert",
    model_name: str = "bert-base-uncased",
) -> ClassificationModel:
    """Train a transformer classification model.

    Falls back to single-process mode automatically if multiprocessing fails.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    args = _configure_args()

    train_data = pd.DataFrame({
        "text":   train_df["text_clean"].values,
        "labels": train_df["labels"].values,
    })

    print(f"Training {model_type} ({model_name}) on {len(train_data)} samples …")

    model = ClassificationModel(
        model_type, model_name, args=args, use_cuda=torch.cuda.is_available()
    )

    try:
        model.train_model(train_data)
    except Exception as exc:
        if args["use_multiprocessing"]:
            print(f"Multiprocessing failed ({str(exc)[:80]}…), retrying single-process …")
            args["use_multiprocessing"]                = False
            args["use_multiprocessing_for_evaluation"] = False
            args["dataloader_num_workers"]             = 0
            model = ClassificationModel(
                model_type, model_name, args=args, use_cuda=torch.cuda.is_available()
            )
            model.train_model(train_data)
        else:
            raise

    print(f"{model_type} training complete.")
    return model


def evaluate_transformer_model(
    model: ClassificationModel, test_df: pd.DataFrame
) -> np.ndarray:
    """Return label predictions for test_df."""
    predictions, _ = model.predict(test_df["text_clean"].tolist())
    return predictions
