from __future__ import annotations

import os
from dataclasses import dataclass


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
INTERIM_DIR = os.path.join(DATA_DIR, "interim")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")


os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


@dataclass(frozen=True)
class Settings:
    random_seed: int = 42
    test_size: float = 0.2
    val_size: float = 0.1  # out of train
    text_priority: tuple[str, ...] = (
        "Clean Narasi",
        "Narasi",
        "isi_berita",
        "summary",
    )
    label_col: str = "hoax"
    source_map: dict[str, str] = None  # filled at runtime
    fasttext_model_path: str = os.path.join(MODELS_DIR, "fasttext_model.bin")
    indobert_model_dir: str = os.path.join(MODELS_DIR, "indobert")
    indobert_checkpoint: str = "indobenchmark/indobert-base-p1"  # HF model


SETTINGS = Settings()
