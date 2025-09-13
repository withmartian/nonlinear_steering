from __future__ import annotations

from pathlib import Path
from typing import List, Mapping, Tuple

import numpy as np
import torch

from src.utils.deps import load_dct_module
from src.utils.model_ops import make_slice, get_hidden


def compute_dct_vectors_for_layers(
    *,
    model,
    tokenizer,
    dataset,
    source_layer: int,
    target_layer: int,
    num_samples: int = 8,
    num_factors: int = 256,
    max_seq_len: int = 48,
    device: str = "cpu",
):
    dct = load_dct_module()

    import random
    prompts = random.sample([row["text"] for row in dataset], k=num_samples)

    source_h = get_hidden(
        model,
        tokenizer,
        prompts,
        max_len=max_seq_len,
        layer_idx=source_layer,
        device=device,
    ).to(dtype=torch.float32)

    slice_model = make_slice(model, source_layer, target_layer, dtype=torch.float32)
    # Ensure we always get hidden states as a structured output
    try:
        slice_model.config.output_hidden_states = True
        slice_model.config.use_return_dict = True
        slice_model.config.use_cache = False
    except Exception:
        pass
    last_layer_idx = len(slice_model.model.layers) - 1

    sliced = dct.SlicedModel(
        slice_model,
        start_layer=0,
        end_layer=last_layer_idx,
        layers_name="model.layers",
    )

    target_h = sliced(source_h).to(dtype=torch.float32)
    delta_single = dct.DeltaActivations(sliced, target_position_indices=slice(-3, None))

    calibrator = dct.SteeringCalibrator(target_ratio=0.5)
    try:
        input_scale = calibrator.calibrate(delta_single, source_h, target_h, factor_batch_size=64)
    except ValueError:
        input_scale = 1.0

    exp_dct = dct.ExponentialDCT(num_factors=num_factors)
    U, V = exp_dct.fit(
        delta_single,
        source_h,
        target_h,
        batch_size=2,
        factor_batch_size=128,
        d_proj=48,
        input_scale=input_scale,
        max_iters=6,
    )
    return V.cpu().detach().numpy().T


