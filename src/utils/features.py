from __future__ import annotations

from typing import List

import numpy as np
import torch


def get_hidden_cached(
    texts: List[str],
    *,
    tokenizer,
    model,
    layer_idx: int,
    device: str,
    batch_size: int = 64,
) -> np.ndarray:
    all_vectors: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        tokenized = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        with torch.no_grad():
            outputs = model(**tokenized, output_hidden_states=True)
        hidden = outputs.hidden_states[layer_idx]
        mask = tokenized["attention_mask"]
        lengths = mask.sum(dim=1) - 1

        for i, idx in enumerate(lengths):
            all_vectors.append(hidden[i, int(idx), :].detach().cpu().float().numpy())

    return np.stack(all_vectors, axis=0)



