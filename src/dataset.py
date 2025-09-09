from __future__ import annotations

import os
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import RAW_DIR, PROCESSED_DIR, SETTINGS


RAW_FILES = {
    "CNN": os.path.join(RAW_DIR, "Summarized_CNN.csv"),
    "Kompas": os.path.join(RAW_DIR, "Summarized_Kompas.csv"),
    "Detik": os.path.join(RAW_DIR, "Summarized_Detik.csv"),
    "TurnBackHoax": os.path.join(RAW_DIR, "Summarized_TurnBackHoax.csv"),
}


def _select_text_column(df: pd.DataFrame) -> pd.Series:
    for col in SETTINGS.text_priority:
        if col in df.columns:
            return df[col]
    # fallback: try any string-like column
    text_cols = [c for c in df.columns if df[c].dtype == object]
    if not text_cols:
        raise ValueError("No text column found in dataframe")
    return df[text_cols[0]]


def load_and_merge() -> pd.DataFrame:
    frames = []
    for source, path in RAW_FILES.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing dataset: {path}")
        df = pd.read_csv(path)
        if SETTINGS.label_col not in df.columns:
            raise KeyError(f"Label column '{SETTINGS.label_col}' missing in {path}")
        text = _select_text_column(df).astype(str)
        label = df[SETTINGS.label_col].astype(int)
        frames.append(pd.DataFrame({"text": text, "label": label, "source": source}))
    data = pd.concat(frames, ignore_index=True)
    # Drop blanks/NaNs
    data["text"] = data["text"].fillna("").str.strip()
    data = data[data["text"].str.len() > 0]
    data = data.reset_index(drop=True)
    return data


def split_dataset(
    data: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(
        data,
        test_size=SETTINGS.test_size,
        random_state=SETTINGS.random_seed,
        stratify=data["label"],
    )
    train, val = train_test_split(
        train,
        test_size=SETTINGS.val_size,
        random_state=SETTINGS.random_seed,
        stratify=train["label"],
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def build_and_save_splits() -> dict:
    data = load_and_merge()
    train, val, test = split_dataset(data)
    out_paths = {
        "train": os.path.join(PROCESSED_DIR, "train.csv"),
        "val": os.path.join(PROCESSED_DIR, "val.csv"),
        "test": os.path.join(PROCESSED_DIR, "test.csv"),
        "all": os.path.join(PROCESSED_DIR, "all.csv"),
    }
    train.to_csv(out_paths["train"], index=False)
    val.to_csv(out_paths["val"], index=False)
    test.to_csv(out_paths["test"], index=False)
    pd.concat([train, val, test]).to_csv(out_paths["all"], index=False)
    return out_paths


if __name__ == "__main__":
    paths = build_and_save_splits()
    print("Saved:", paths)
