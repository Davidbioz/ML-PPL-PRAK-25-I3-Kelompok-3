from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from ..config import SETTINGS
from ..dataset import build_and_save_splits
from ..features import normalize_batch
from ..plots import plot_confusion_matrix, save_classification_report


# --------------------------
# Metrics helpers
# --------------------------


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}


# --------------------------
# FastText Baseline
# --------------------------


def train_fasttext(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Dict:
    import fasttext

    # Prepare supervised format: __label__<label> text
    def to_ft(df: pd.DataFrame, path: str):
        texts = normalize_batch(df["text"].tolist())
        labels = df["label"].astype(int).tolist()
        with open(path, "w", encoding="utf-8") as f:
            for t, y in zip(texts, labels):
                f.write(f"__label__{y} {t}\n")

    with tempfile.TemporaryDirectory() as td:
        tr_path = os.path.join(td, "train.txt")
        va_path = os.path.join(td, "val.txt")
        to_ft(train_df, tr_path)
        to_ft(val_df, va_path)

        model = fasttext.train_supervised(
            input=tr_path,
            lr=0.5,
            epoch=25,
            wordNgrams=2,
            dim=100,
            loss="softmax",
            verbose=2,
        )

        # Evaluate on validation
        def predict_labels(df: pd.DataFrame):
            texts = normalize_batch(df["text"].tolist())
            preds = [
                int(model.predict(t)[0][0].replace("__label__", "")) for t in texts
            ]
            return np.array(preds)

        val_pred = predict_labels(val_df)
        val_metrics = compute_metrics(val_df["label"].values, val_pred)

        # Save model
        model.save_model(SETTINGS.fasttext_model_path)

    # Test evaluation with saved model (reload)
    import fasttext

    model = fasttext.load_model(SETTINGS.fasttext_model_path)
    test_texts = normalize_batch(test_df["text"].tolist())
    test_pred = [
        int(model.predict(t)[0][0].replace("__label__", "")) for t in test_texts
    ]
    y_true = test_df["label"].values
    y_pred = np.array(test_pred)
    test_metrics = compute_metrics(y_true, y_pred)
    # Artifacts
    plot_confusion_matrix(
        y_true, y_pred, title="FastText Test CM", fname="cm_fasttext.png"
    )
    save_classification_report(y_true, y_pred, fname="report_fasttext.txt")
    return {"val": val_metrics, "test": test_metrics}


# --------------------------
# IndoBERT Fine-tuning
# --------------------------


@dataclass
class BertParams:
    max_length: int = 256
    batch_size: int = 16
    epochs: int = 3
    lr: float = 2e-5


def train_indobert(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    params: BertParams = BertParams(),
) -> Dict:
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    model_name = SETTINGS.indobert_checkpoint
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    class TextDataset(Dataset):
        def __init__(self, df: pd.DataFrame):
            self.texts = df["text"].tolist()
            self.labels = df["label"].astype(int).tolist()

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            enc = tokenizer(
                self.texts[idx],
                truncation=True,
                max_length=params.max_length,
                return_tensors="pt",
            )
            item = {k: v.squeeze(0) for k, v in enc.items()}
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    train_ds = TextDataset(train_df)
    val_ds = TextDataset(val_df)
    test_ds = TextDataset(test_df)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    collator = DataCollatorWithPadding(tokenizer)

    # Mixed precision selection (CUDA only)
    use_fp16 = False
    use_bf16 = False
    if torch.cuda.is_available():
        try:
            if (
                hasattr(torch.cuda, "is_bf16_supported")
                and torch.cuda.is_bf16_supported()
            ):
                use_bf16 = True
            else:
                use_fp16 = True
        except Exception:
            use_fp16 = True

    training_args = TrainingArguments(
        output_dir=SETTINGS.indobert_model_dir,
        overwrite_output_dir=True,
        learning_rate=params.lr,
        per_device_train_batch_size=params.batch_size,
        per_device_eval_batch_size=params.batch_size,
        num_train_epochs=params.epochs,
        logging_steps=50,
        seed=SETTINGS.random_seed,
        report_to=[],
        fp16=use_fp16,
        bf16=use_bf16,
    )

    def hf_compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        p, r, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=0
        )
        return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=hf_compute_metrics,
    )

    trainer.train()
    trainer.save_model(SETTINGS.indobert_model_dir)

    # Evaluate and save artifacts
    val_metrics = trainer.evaluate(val_ds)
    test_metrics = trainer.evaluate(test_ds)
    preds = np.argmax(trainer.predict(test_ds).predictions, axis=-1)
    y_true = test_df["label"].values
    plot_confusion_matrix(
        y_true, preds, title="IndoBERT Test CM", fname="cm_indobert.png"
    )
    save_classification_report(y_true, preds, fname="report_indobert.txt")
    return {"val": val_metrics, "test": test_metrics}


def main(model: str = "both") -> Tuple[Dict, Dict]:
    paths = build_and_save_splits()
    train_df = pd.read_csv(paths["train"])  # noqa
    val_df = pd.read_csv(paths["val"])  # noqa
    test_df = pd.read_csv(paths["test"])  # noqa

    results = {}
    if model in ("fasttext", "both"):
        results["fasttext"] = train_fasttext(train_df, val_df, test_df)
    if model in ("indobert", "both"):
        results["indobert"] = train_indobert(train_df, val_df, test_df)
    return results


def run_indobert_and_compare_barplot(epochs: int = 1) -> str:
    paths = build_and_save_splits()
    train_df = pd.read_csv(paths["train"])  # noqa
    val_df = pd.read_csv(paths["val"])  # noqa
    test_df = pd.read_csv(paths["test"])  # noqa

    # FastText: reload metrics from a quick evaluation using saved model
    # If model not trained yet, do a short training
    try:
        from .predict import predict_fasttext  # noqa: F401

        fasttext_available = os.path.exists(SETTINGS.fasttext_model_path)
    except Exception:
        fasttext_available = False
    if not fasttext_available:
        ft_res = train_fasttext(train_df, val_df, test_df)
        ft_test = ft_res["test"]
    else:
        import fasttext

        model = fasttext.load_model(SETTINGS.fasttext_model_path)
        texts = normalize_batch(test_df["text"].tolist())
        preds = [int(model.predict(t)[0][0].replace("__label__", "")) for t in texts]
        ft_test = compute_metrics(test_df["label"].values, np.array(preds))

    # IndoBERT short run
    bp = BertParams(epochs=epochs)
    ind_res = train_indobert(train_df, val_df, test_df, params=bp)
    id_test = ind_res["test"]

    models_metrics = {
        "FastText": {
            "accuracy": ft_test.get("accuracy"),
            "precision": ft_test.get("precision"),
            "recall": ft_test.get("recall"),
            "f1": ft_test.get("f1"),
        },
        "IndoBERT": {
            "accuracy": id_test.get("eval_accuracy", id_test.get("accuracy")),
            "precision": id_test.get("eval_precision", id_test.get("precision")),
            "recall": id_test.get("eval_recall", id_test.get("recall")),
            "f1": id_test.get("eval_f1", id_test.get("f1")),
        },
    }
    from ..plots import plot_metrics_comparison

    out_path = plot_metrics_comparison(models_metrics, fname="metrics_compare.png")
    return out_path


if __name__ == "__main__":
    out = main("both")
    print(out)
