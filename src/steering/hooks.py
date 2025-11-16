from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch


def get_gradient_hook(
    *,
    steer_module,
    target_labels: List[int] | torch.Tensor | None = None,
    avoid_labels: List[int] | torch.Tensor | None = None,
    alpha: float = 1.0,
    steps: int = 1,
    step_size_decay: float = 1.0,
):
    target_labels = torch.as_tensor(target_labels or [], device=steer_module.device)
    avoid_labels = torch.as_tensor(avoid_labels or [], device=steer_module.device)

    @torch.inference_mode(False)
    def fwd_hook(module, inp, out):
        # Support both tuple outputs and tensor outputs from decoder layers
        if isinstance(out, tuple):
            h_fp16 = out[0]
            rest = out[1:]
        else:
            h_fp16 = out
            rest = None

        B, S, D = h_fp16.shape
        h_current = h_fp16.reshape(-1, D).float()
        for step in range(steps):
            h_step = h_current.clone()
            h_step.requires_grad_(True)
            logits = steer_module.classifier(h_step)
            logits = logits.view(B, S, -1).mean(dim=1)
            loss_vec = _compute_steering_loss(
                logits, target_idx=target_labels, avoid_idx=avoid_labels
            )
            if loss_vec.numel() > 0:
                grad = torch.autograd.grad(
                    outputs=loss_vec,
                    inputs=h_step,
                    grad_outputs=torch.ones_like(loss_vec),
                    retain_graph=False,
                    create_graph=False,
                )[0]
                current_alpha = alpha * (step_size_decay ** step)
                grad = grad.view(B * S, D)
                h_current = (h_step - current_alpha * grad).detach()
            else:
                h_current = h_step.detach()
        h_new = h_current.reshape(B, S, D).to(h_fp16.dtype)
        if rest is None:
            return h_new
        return (h_new,) + rest

    return fwd_hook


def _compute_steering_loss(
    logits: torch.Tensor,
    *,
    target_idx,
    avoid_idx,
) -> torch.Tensor:
    if not torch.is_tensor(target_idx):
        target_idx = torch.as_tensor(target_idx, device=logits.device)
    else:
        target_idx = target_idx.to(logits.device)
    if not torch.is_tensor(avoid_idx):
        avoid_idx = torch.as_tensor(avoid_idx, device=logits.device)
    else:
        avoid_idx = avoid_idx.to(logits.device)
    B, _ = logits.shape
    avoid_term = logits[:, avoid_idx].mean(dim=1) if avoid_idx.numel() > 0 else torch.zeros(B, device=logits.device)
    target_term = logits[:, target_idx].mean(dim=1) if target_idx.numel() > 0 else torch.zeros(B, device=logits.device)
    return avoid_term - target_term


def get_additive_hook(vector: torch.Tensor | np.ndarray, *, alpha: float = 1.0):
    if not torch.is_tensor(vector):
        vector = torch.as_tensor(vector, dtype=torch.float16)

    def fwd_hook(module, inp, out):
        if isinstance(out, tuple):
            h = out[0]
            rest = out[1:]
            return (h + alpha * vector.to(h.device),) + rest
        else:
            h = out
            return h + alpha * vector.to(h.device)

    return fwd_hook



