from __future__ import annotations

import copy
import torch


def make_slice(base_model, start, end, *, dtype):
    m = copy.deepcopy(base_model).to(dtype=dtype)
    m.model.layers = m.model.layers[start:end]
    return m


def get_hidden(model, tok, texts, *, max_len=48, layer_idx=-1, device="cpu"):
    ids = tok(
        texts, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        h = model(**ids, use_cache=False, output_hidden_states=True).hidden_states
    return h[layer_idx]



