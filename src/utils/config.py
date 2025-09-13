from __future__ import annotations

from typing import Any, Dict
import yaml


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def pick(args_ns, cfg: Dict[str, Any], key: str, default=None):
    val = getattr(args_ns, key.replace("-", "_"), None)
    if val not in (None, []):
        return val
    return cfg.get(key, default)



