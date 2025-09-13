from __future__ import annotations

from collections import defaultdict
from typing import List, Optional

import numpy as np

from .activations import get_hidden_cached


def compute_caa_vectors(
    dataset,
    unique_labels: List[str],
    *,
    steer_layer: int,
    tokenizer,
    model,
    device: str,
    max_pairs: Optional[int] = None,
) -> np.ndarray:
    q2lab2text = defaultdict(dict)
    for row in dataset:
        q2lab2text[row["original_question"]][row["label"]] = row["text"]

    pos, neg = defaultdict(list), defaultdict(list)
    for _, lab_map in q2lab2text.items():
        labs = set(lab_map)
        for tgt in labs:
            for other in labs - {tgt}:
                pos[tgt].append(lab_map[tgt])
                neg[tgt].append(lab_map[other])

    caa_vecs = []
    for lbl in unique_labels:
        pairs = len(pos[lbl])
        if max_pairs and pairs > max_pairs:
            import random
            keep = random.sample(range(pairs), max_pairs)
            pos[lbl] = [pos[lbl][i] for i in keep]
            neg[lbl] = [neg[lbl][i] for i in keep]

        if not pos[lbl]:
            caa_vecs.append(np.zeros(model.config.hidden_size, dtype=np.float32))
            continue

        X_pos = get_hidden_cached(pos[lbl], tokenizer=tokenizer, model=model, layer_idx=steer_layer, device=device)
        X_neg = get_hidden_cached(neg[lbl], tokenizer=tokenizer, model=model, layer_idx=steer_layer, device=device)
        caa_vecs.append((X_pos - X_neg).mean(0))

    return np.stack(caa_vecs, axis=0)



