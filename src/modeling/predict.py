from __future__ import annotations

from typing import Iterable, List

from ..config import SETTINGS
from ..features import normalize_batch


def predict_fasttext(texts: Iterable[str]) -> List[int]:
    import fasttext

    model = fasttext.load_model(SETTINGS.fasttext_model_path)
    norm = normalize_batch(texts)
    preds = [int(model.predict(t)[0][0].replace("__label__", "")) for t in norm]
    return preds


def predict_indobert(texts: Iterable[str]) -> List[int]:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    import numpy as np

    tokenizer = AutoTokenizer.from_pretrained(SETTINGS.indobert_model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        SETTINGS.indobert_model_dir
    )
    model.eval()

    preds: List[int] = []
    for t in texts:
        enc = tokenizer(t, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits
            pred = int(np.argmax(logits.cpu().numpy(), axis=-1)[0])
            preds.append(pred)
    return preds
