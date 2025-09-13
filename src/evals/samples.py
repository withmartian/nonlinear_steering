from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from src.utils.generation import batch_generate
from src.steering.hooks import get_gradient_hook, get_additive_hook


def sample_steered_responses(
    *,
    prompts: List[str],
    target_labels: List[str],
    unique_labels: List[str],
    steer_model_k,
    layer_k: int,
    alpha_grad: float,
    caa_vectors,
    layer_caa: int,
    alpha_caa: float,
    model,
    tokenizer,
    max_new_tokens: int = 32,
    batch_size: int = 500,
    save_path: Path | None = None,
):
    tone2idx = {t: i for i, t in enumerate(unique_labels)}
    tgt_idx = [tone2idx[t] for t in target_labels]

    grad_hook = get_gradient_hook(steer_model_k, target_labels=tgt_idx, avoid_labels=[], alpha=alpha_grad)
    caa_vec = caa_vectors[tgt_idx].mean(axis=0)
    caa_hook = get_additive_hook(caa_vec, alpha=alpha_caa)

    unsteered = batch_generate(model, tokenizer, prompts, layer_idx=layer_caa, hook_fn=None, max_new_tokens=max_new_tokens, batch_size=batch_size)
    ksteer = batch_generate(model, tokenizer, prompts, layer_idx=layer_k, hook_fn=grad_hook, max_new_tokens=max_new_tokens, batch_size=batch_size)
    caa_out = batch_generate(model, tokenizer, prompts, layer_idx=layer_caa, hook_fn=caa_hook, max_new_tokens=max_new_tokens, batch_size=batch_size)

    def _strip(gen: str, prompt: str) -> str:
        return gen[len(prompt) :].lstrip() if gen.startswith(prompt) else gen

    rows = []
    for p, base, ktxt, ctxt in zip(prompts, unsteered, ksteer, caa_out):
        rows.append(
            {
                "prompt": p,
                "unsteered": _strip(base, p),
                "k_steering": _strip(ktxt, p),
                "caa": _strip(ctxt, p),
                "layer_k": layer_k,
                "layer_caa": layer_caa,
                "alpha_grad": alpha_grad,
                "alpha_caa": alpha_caa,
                "targets": ", ".join(target_labels),
            }
        )
    df = pd.DataFrame(rows)
    if save_path is not None:
        if save_path.suffix.lower() == ".json":
            df.to_json(save_path, orient="records", indent=2)
        else:
            df.to_csv(save_path, index=False)
    return df



