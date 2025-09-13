from __future__ import annotations

from typing import Callable, List, Optional

import torch


def batch_generate(
    model,
    tokenizer,
    prompts: List[str],
    *,
    layer_idx: int,
    hook_fn: Optional[Callable] = None,
    max_new_tokens: int = 24,
    batch_size: int = 512,
) -> List[str]:
    device = model.device
    target_layer = model.model.layers[layer_idx]
    outputs: List[str] = []

    saved_hooks = target_layer._forward_hooks.copy()
    target_layer._forward_hooks.clear()

    handle = None
    if hook_fn is not None:
        handle = target_layer.register_forward_hook(hook_fn)

    try:
        for start in range(0, len(prompts), batch_size):
            sub_prompts = prompts[start : start + batch_size]
            tokenized_inputs = tokenizer(
                sub_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)

            with torch.no_grad():
                gen_ids = model.generate(
                    **tokenized_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            outputs.extend(
                tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            )
    finally:
        if handle is not None:
            handle.remove()
        target_layer._forward_hooks.clear()
        target_layer._forward_hooks.update(saved_hooks)

    return outputs



