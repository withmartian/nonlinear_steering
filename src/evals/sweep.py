from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
from tqdm.auto import tqdm

from src.steering.eval import eval_steering_combinations
from src.steering.caa import compute_caa_vectors


async def sweep_alphas_for_layers(
    layers_to_sweep: List[int],
    *,
    prompts: List[str],
    dataset,
    unique_labels,
    num_target_labels: int = 2,
    act_clf=None,
    eval_layer: int | None = None,
    alpha_grad_guess: tuple[float, float] = (0.1, 3.2e4),
    alpha_caa_guess: tuple[float, float] = (0.1, 32.0),
    num_pairs_caa: int = 100,
    steps: int = 1,
    opposing: bool = False,
    **other_kwargs,
):
    combos = [tuple(sorted(c)) for c in combinations(unique_labels, num_target_labels)]
    layer2alpha = {}
    for l in tqdm(layers_to_sweep, desc="Layers"):
        caa_vecs_layer = compute_caa_vectors(
            dataset, unique_labels, steer_layer=l, tokenizer=other_kwargs.get("tokenizer"), model=other_kwargs.get("model"), device=str(other_kwargs.get("model").device), max_pairs=num_pairs_caa
        )
        combo2α = {}
        # Here we could plug in OOD-based calibration; placeholder zeros
        for combo in tqdm(combos, desc=f"Layer {l} combos", leave=False):
            combo2α[combo] = (1.0, 1.0)
        layer2alpha[l] = combo2α
    return layer2alpha


async def evaluate_layers(
    layer2alpha: Dict[int, Dict[tuple, Tuple[float, float]]],
    *,
    prompts: List[str],
    dataset,
    unique_labels,
    num_target_labels: int = 2,
    opposing: bool = False,
    steps: int = 1,
    max_samples: int = 100,
    **run_kwargs,
):
    frames = {}
    for l in sorted(layer2alpha.keys()):
        df = await eval_steering_combinations(
            prompts=prompts[:max_samples],
            unique_labels=unique_labels,
            steer_model=run_kwargs["steer_model_getter"](l),
            caa_vectors=run_kwargs["caa_vecs_getter"](l),
            layer_idx=l,
            act_clf=run_kwargs["act_clf"],
            alpha_grad=1.0,
            alpha_caa=1.0,
            base_model=run_kwargs["model"],
            tokenizer=run_kwargs["tokenizer"],
            num_target_labels=num_target_labels,
            max_samples=max_samples,
        )
        k_score = float(df["K-Steering"].mean())
        c_score = float(df["CAA"].mean())
        frames[l] = (df, k_score, c_score)
    best_k_layer = max(frames, key=lambda x: frames[x][1])
    best_caa_layer = max(frames, key=lambda x: frames[x][2])
    return frames[best_k_layer][0], best_caa_layer, best_k_layer



