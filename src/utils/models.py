from __future__ import annotations

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_hf_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        _attn_implementation="eager",
        output_hidden_states=True,
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    return model, tokenizer



