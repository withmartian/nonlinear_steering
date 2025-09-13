from __future__ import annotations

from collections import defaultdict
from itertools import combinations, permutations
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from src.utils.generation import batch_generate
from src.utils.features import get_hidden_cached
from .models import ActivationSteering, one_hot
from .hooks import get_additive_hook, get_gradient_hook


def get_or_train_layer_clf(
    *,
    layer_idx: int,
    X: np.ndarray,
    y: np.ndarray,
    unique_labels: Sequence[str],
    hidden_dim: int = 128,
    epochs: int = 5,
    batch_size: int = 32,
    cache_dir: str = "layer_clfs",
):
    import os
    from pathlib import Path

    if y.dtype.kind not in ("i", "u"):
        lbl2idx = {lbl: i for i, lbl in enumerate(unique_labels)}
        y = np.asarray([lbl2idx[lbl] for lbl in y], dtype=np.int64)

    f = Path(cache_dir) / f"layer{layer_idx}.pt"
    f.parent.mkdir(exist_ok=True, parents=True)
    if f.exists():
        sd = torch.load(f, map_location="cpu", weights_only=False)
        clf = ActivationSteering(input_dim=X.shape[1], num_labels=len(unique_labels), hidden_dim=hidden_dim)
        clf.classifier.load_state_dict(sd["state_dict"])
        return clf, sd["acc"]

    idx_A, idx_B = train_test_split(np.arange(len(X)), test_size=0.5, random_state=42, stratify=y)
    X_A, X_B, y_A, y_B = X[idx_A], X[idx_B], y[idx_A], y[idx_B]

    clf = ActivationSteering(input_dim=X.shape[1], num_labels=len(unique_labels), hidden_dim=hidden_dim)
    clf.fit(X_A, one_hot(y_A, len(unique_labels)), epochs=epochs, batch_size=batch_size)

    with torch.no_grad():
        acc = (
            torch.argmax(
                clf.classifier(torch.tensor(X_B, dtype=torch.float32, device=clf.device)), dim=1
            ).cpu().numpy()
            == y_B
        ).mean()

    torch.save({"state_dict": clf.classifier.state_dict(), "acc": acc}, f)
    return clf, acc


def get_or_train_eval_clf(
    *,
    X: np.ndarray,
    y: np.ndarray,
    unique_labels: Sequence[str],
    hidden_dim: int = 128,
    epochs: int = 5,
    batch_size: int = 32,
    cache_dir: str = "layer_clfs",
):
    from pathlib import Path
    cache_f = Path(cache_dir) / "final_layer_eval.pt"
    cache_f.parent.mkdir(exist_ok=True, parents=True)

    if cache_f.exists():
        sd = torch.load(cache_f, map_location="cpu", weights_only=False)
        clf = ActivationSteering(input_dim=X.shape[1], num_labels=len(unique_labels), hidden_dim=hidden_dim)
        clf.classifier.load_state_dict(sd["state_dict"])
        return clf, sd.get("acc_on_A", None)

    idx_A, idx_B = train_test_split(np.arange(len(X)), test_size=0.5, random_state=42, stratify=y)
    X_A, X_B, y_A, y_B = X[idx_A], X[idx_B], y[idx_A], y[idx_B]

    clf = ActivationSteering(input_dim=X.shape[1], num_labels=len(unique_labels), hidden_dim=hidden_dim)
    clf.fit(X_B, one_hot(y_B, len(unique_labels)), epochs=epochs, batch_size=batch_size)

    with torch.no_grad():
        preds = clf.classifier(torch.tensor(X_A, dtype=torch.float32, device=clf.device))
        acc_A = (torch.argmax(preds, dim=1).cpu().numpy() == y_A).mean()

    torch.save({"state_dict": clf.classifier.state_dict(), "acc_on_A": acc_A}, cache_f)
    return clf, acc_A


def evaluate_combo(
    *,
    tgt_idx: List[int],
    avoid_idx: Optional[List[int]] = None,
    tgt_names: List[str],
    base_act: np.ndarray,
    steer_model,
    caa_vectors: np.ndarray,
    act_clf,
    alpha_grad: float,
    alpha_caa: float,
) -> Dict[str, object]:
    device = next(act_clf.parameters()).device
    steered_list = []
    for i in range(base_act.shape[0]):
        x = base_act[i : i + 1]
        if avoid_idx:
            t = steer_model.steer_activations(x, tgt_idx, avoid_idx=avoid_idx, alpha=alpha_grad, steps=1)
        else:
            t = steer_model.steer_activations(x, tgt_idx, alpha=alpha_grad, steps=1)
        steered_list.append(t.detach().cpu().numpy())
    grad_act = np.concatenate(steered_list, axis=0)

    caa_vec = caa_vectors[tgt_idx].mean(axis=0)
    caa_act = base_act + alpha_caa * caa_vec[None, :]

    with torch.no_grad():
        base_logits = act_clf(torch.tensor(base_act, dtype=torch.float32, device=device))
        grad_logits = act_clf(torch.tensor(grad_act, dtype=torch.float32, device=device))
        caa_logits = act_clf(torch.tensor(caa_act, dtype=torch.float32, device=device))

    base_prob = torch.sigmoid(base_logits).cpu().numpy()
    grad_prob = torch.sigmoid(grad_logits).cpu().numpy()
    caa_prob = torch.sigmoid(caa_logits).cpu().numpy()

    delta_k = (grad_prob[:, tgt_idx] - base_prob[:, tgt_idx]).mean()
    delta_c = (caa_prob[:, tgt_idx] - base_prob[:, tgt_idx]).mean()
    return {"Targets": ", ".join(tgt_names), "K-Steering": float(delta_k), "CAA": float(delta_c)}


def eval_steering_combinations(
    *,
    prompts: List[str],
    unique_labels: List[str],
    caa_vectors: np.ndarray,
    steer_model,
    act_clf,
    base_model,
    tokenizer,
    layer_idx: int,
    alpha_grad: float = 1.0,
    alpha_caa: float = 1.0,
    alpha_table: Optional[Dict[Tuple[str, ...], Tuple[float, float]]] = None,
    num_target_labels: int = 2,
    opposing: bool = False,
    steps: int = 1,
    max_samples: Optional[int] = None,
) -> pd.DataFrame:
    if max_samples is not None:
        prompts = prompts[:max_samples]

    base_ans = batch_generate(base_model, tokenizer, prompts, layer_idx=layer_idx, hook_fn=None)
    base_act = get_hidden_cached(prompts, tokenizer=tokenizer, model=base_model, layer_idx=layer_idx, device=str(base_model.device))
    tone2idx = {t: i for i, t in enumerate(unique_labels)}

    combos = [tuple(sorted(c)) for c in combinations(unique_labels, num_target_labels)]
    model_device = next(act_clf.parameters()).device

    rows = []
    for combo in combos:
        tgt_idx = [tone2idx[label] for label in combo]
        # pick alphas
        if alpha_table is not None and combo in alpha_table:
            a_g, a_c = alpha_table[combo]
        else:
            a_g, a_c = alpha_grad, alpha_caa

        avoid_idx = None
        if opposing and len(tgt_idx) == 2:
            avoid_idx = [tgt_idx[1]]
            tgt_eval_idx = [tgt_idx[0]]
        else:
            tgt_eval_idx = tgt_idx

        row = evaluate_combo(
            tgt_idx=tgt_idx,
            avoid_idx=avoid_idx,
            tgt_names=list(combo),
            base_act=base_act,
            steer_model=steer_model,
            caa_vectors=caa_vectors,
            act_clf=act_clf,
            alpha_grad=a_g,
            alpha_caa=a_c,
        )
        rows.append(row)
    return pd.DataFrame(rows)


