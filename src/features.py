from __future__ import annotations

import re
from typing import Iterable, List


URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@[\w_]+|#[\w_]+")
NON_TEXT_RE = re.compile(r"[^\w\s.,!?;:\-()'\"]", re.UNICODE)
MULTI_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    t = text.lower()
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    # Keep common punctuation; remove emojis and symbols
    t = NON_TEXT_RE.sub(" ", t)
    t = MULTI_WS_RE.sub(" ", t).strip()
    return t


def normalize_batch(texts: Iterable[str]) -> List[str]:
    return [normalize_text(t) for t in texts]
