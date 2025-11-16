from __future__ import annotations

import numpy as np
import torch

from src.steering.models import ActivationSteering, one_hot


def build_eval_classifier(model, tokenizer, eval_prompts, *, eval_layer: int, unique_labels):
    from src.utils.features import get_hidden_cached
    device = str(model.device)
    X_eval = get_hidden_cached(
        eval_prompts, tokenizer=tokenizer, model=model, layer_idx=eval_layer, device=device
    )
    y = np.zeros(len(X_eval), dtype=np.int64)
    clf = ActivationSteering(input_dim=X_eval.shape[1], num_labels=len(unique_labels))
    idx = np.arange(len(X_eval))
    np.random.shuffle(idx)
    mid = len(idx) // 2
    if mid == 0:
        mid = len(idx)
    clf.fit(X_eval[idx[mid:]], one_hot(y[idx[mid:]], len(unique_labels)), epochs=1, batch_size=64)
    return clf



