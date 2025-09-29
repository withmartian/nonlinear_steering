from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from src.utils.features import get_hidden_cached
from src.utils.generation import batch_generate
from src.steering.hooks import get_additive_hook
from .ood import calibrate_alpha_ood_only, is_ood, OpenAiJudge


async def map_dct_vectors_to_labels(
    *,
    dct_vectors: np.ndarray,
    prompts: List[str],
    act_clf,
    layer_idx: int,
    base_act: Optional[np.ndarray],
    unique_labels: List[str],
) -> Mapping[str, List[int]]:
    base_module = getattr(act_clf, "classifier", act_clf)
    base_module.eval()
    device = next(base_module.parameters()).device

    if base_act is None:
        base_act = get_hidden_cached(prompts, tokenizer=None, model=None, layer_idx=layer_idx, device=str(device))
        # Note: callers should pass base_act for efficiency; this branch is a fallback

    base_act = np.asarray(base_act, dtype=np.float32)
    with torch.no_grad():
        t0 = torch.tensor(base_act, dtype=torch.float32, device=device)
        P0 = base_module(t0).sigmoid().cpu().numpy()

    tone2dct = defaultdict(list)
    for i_vec, vec in enumerate(dct_vectors):
        vec = np.asarray(vec, dtype=np.float32)
        with torch.no_grad():
            t1 = torch.tensor(base_act + vec, dtype=torch.float32, device=device)
            P1 = base_module(t1).sigmoid().cpu().numpy()

        delta_p = P1.mean(axis=0) - P0.mean(axis=0)
        best_idx = int(np.argmax(delta_p))
        best_lbl = unique_labels[best_idx]
        tone2dct[best_lbl].append(i_vec)

    return tone2dct


async def sweep_alphas_for_dct(
    *,
    prompts: List[str],
    unique_labels: List[str],
    tone2dct: Mapping[str, List[int]],
    dct_vectors: np.ndarray,
    layer_idx: int,
    model,
    tokenizer,
    judge: Optional[OpenAiJudge] = None,
    alpha_dct_guess: Tuple[float, float] = (1.0, 32.0),
    tol: float = 0.05,
    max_iters: int = 10,
    num_labels: int = 2,
) -> Dict[Tuple[str, ...], float]:
    combo2alpha: Dict[Tuple[str, ...], float] = {}
    for combo in combinations(unique_labels, num_labels):
        vec_ids = [vid for lbl in combo for vid in tone2dct.get(lbl, [])]
        if not vec_ids:
            combo2alpha[combo] = 0.0
            continue

        vec = dct_vectors[vec_ids].mean(axis=0)

        async def _ood_check(alpha: float) -> bool:
            hook = get_additive_hook(vec, alpha=alpha)
            gens = batch_generate(
                model,
                tokenizer,
                prompts,
                layer_idx=layer_idx,
                hook_fn=hook,
                max_new_tokens=24,
            )
            return await is_ood(gens, judge=judge) if judge is not None else False

        if judge is None:
            best_alpha = alpha_dct_guess[0]
        else:
            best_alpha = await calibrate_alpha_ood_only(
                _ood_check,
                min_alpha=alpha_dct_guess[0],
                max_alpha=alpha_dct_guess[1],
                tol=tol,
                max_iters=max_iters,
            )
        combo2alpha[combo] = float(best_alpha)

    return combo2alpha



