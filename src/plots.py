from __future__ import annotations
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from .config import FIGURES_DIR


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, title: str, fname: str
) -> str:
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["non-hoax", "hoax"], cmap="Blues", ax=ax
    )
    ax.set_title(title)
    out_path = os.path.join(FIGURES_DIR, fname)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def save_classification_report(
    y_true: np.ndarray, y_pred: np.ndarray, fname: str
) -> str:
    report = classification_report(
        y_true, y_pred, target_names=["non-hoax", "hoax"], digits=4
    )
    out_path = os.path.join(FIGURES_DIR, fname)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    return out_path


def plot_metrics_comparison(
    models_metrics: dict,
    fname: str = "metrics_compare.png",
    title: str = "Model Comparison (Test)",
) -> str:
    # Expected shape: {"FastText": {"accuracy": float, "precision": float, "recall": float, "f1": float}, "IndoBERT": {...}}
    metrics = ["accuracy", "precision", "recall", "f1"]
    model_names = list(models_metrics.keys())
    x = np.arange(len(metrics))
    width = 0.35 if len(model_names) == 2 else 0.8 / max(1, len(model_names))

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, model in enumerate(model_names):
        vals = [models_metrics[model].get(m, np.nan) for m in metrics]
        ax.bar(x + (i - (len(model_names) - 1) / 2) * width, vals, width, label=model)

    ax.set_xticks(x)
    ax.set_xticklabels([m.title() for m in metrics])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    out_path = os.path.join(FIGURES_DIR, fname)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
