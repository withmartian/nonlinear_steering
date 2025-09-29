from __future__ import annotations

from typing import Any, List, Optional, Tuple
import torch


_MODEL = None
_TOKENIZER = None
_DEVICE: Optional[str] = None
_DATASET = None
_UNIQUE_LABELS: Optional[List[str]] = None


def register_model_tokenizer(model: Any, tokenizer: Any, device: Optional[str] = None) -> None:
    global _MODEL, _TOKENIZER, _DEVICE
    _MODEL = model
    _TOKENIZER = tokenizer
    _DEVICE = device or ("cuda" if torch.cuda.is_available() else "cpu")


def get_model_tokenizer() -> Tuple[Any, Any, str]:
    if _MODEL is None or _TOKENIZER is None:
        raise RuntimeError("Model/tokenizer not registered. Call register_model_tokenizer() after loading.")
    return _MODEL, _TOKENIZER, (_DEVICE or ("cuda" if torch.cuda.is_available() else "cpu"))


def register_dataset(dataset: Any, unique_labels: List[str]) -> None:
    global _DATASET, _UNIQUE_LABELS
    _DATASET = dataset
    _UNIQUE_LABELS = list(unique_labels)


def get_dataset() -> Any:
    if _DATASET is None:
        raise RuntimeError("Dataset not registered. Call register_dataset(dataset, unique_labels).")
    return _DATASET


def get_unique_labels() -> List[str]:
    if _UNIQUE_LABELS is None:
        raise RuntimeError("Unique labels not registered. Call register_dataset(dataset, unique_labels).")
    return _UNIQUE_LABELS



