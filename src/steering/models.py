from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def one_hot(indices: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((len(indices), num_classes), dtype=np.float32)
    out[np.arange(len(indices)), indices] = 1.0
    return out


class MultiLabelSteeringModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_labels: int,
        *,
        linear: bool = False,
    ) -> None:
        super().__init__()
        if linear:
            self.net = nn.Linear(input_dim, num_labels)
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_labels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActivationSteering:
    def __init__(self, input_dim: int, num_labels: int, *, hidden_dim: int = 128, lr: float = 1e-3, device: Optional[str] = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_labels = num_labels
        self.classifier = MultiLabelSteeringModel(
            input_dim, hidden_dim, num_labels
        ).to(self.device)
        self.optimizer = optim.Adam(self.classifier.parameters(), lr=lr)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def fit(self, X: np.ndarray, Y: np.ndarray, *, epochs: int = 10, batch_size: int = 32) -> None:
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        Y_t = torch.tensor(Y, dtype=torch.float32, device=self.device)
        dataset = torch.utils.data.TensorDataset(X_t, Y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for _ in range(epochs):
            for bx, by in loader:
                self.optimizer.zero_grad()
                logits = self.classifier(bx)
                loss = self.loss_fn(logits, by)
                loss.backward()
                self.optimizer.step()

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.classifier.eval()
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        logits = self.classifier(X_t)
        probs = torch.sigmoid(logits)
        return probs.cpu().numpy()

    def steer_activations(
        self,
        acts: Union[np.ndarray, torch.Tensor],
        target_idx: List[int],
        avoid_idx: List[int] | None = None,
        *,
        alpha: float = 1.0,
        steps: int = 1,
        step_size_decay: float = 1.0,
    ) -> torch.Tensor:
        if avoid_idx is None:
            avoid_idx = []
        if isinstance(acts, np.ndarray):
            acts_t = torch.as_tensor(acts, dtype=torch.float32, device=self.device)
        else:
            acts_t = acts.to(self.device, dtype=torch.float32)

        steered = acts_t.detach().clone()
        for step in range(steps):
            curr = steered.clone().requires_grad_(True)
            logits = self.classifier(curr)
            loss_vec = _compute_steering_loss(
                logits, target_idx=target_idx, avoid_idx=avoid_idx
            )
            loss = loss_vec.mean()
            grads = torch.autograd.grad(loss, curr, retain_graph=False)[0]
            current_alpha = alpha * (step_size_decay ** step)
            steered = (curr - current_alpha * grads).detach()
        return steered


def _compute_steering_loss(
    logits: torch.Tensor,
    *,
    target_idx: List[int] | torch.Tensor,
    avoid_idx: List[int] | torch.Tensor,
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
    if avoid_idx.numel() > 0:
        avoid_term = logits[:, avoid_idx].mean(dim=1)
    else:
        avoid_term = torch.zeros(B, device=logits.device)
    if target_idx.numel() > 0:
        target_term = logits[:, target_idx].mean(dim=1)
    else:
        target_term = torch.zeros(B, device=logits.device)
    return avoid_term - target_term



