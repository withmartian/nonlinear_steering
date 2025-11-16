from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
GENERATIONS_DIR = DATA_DIR / "generations"
RESULTS_DIR = PROJECT_ROOT / "results"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

for p in (DATA_DIR, GENERATIONS_DIR, RESULTS_DIR, NOTEBOOKS_DIR):
    p.mkdir(parents=True, exist_ok=True)


def result_path(name: str) -> Path:
    return RESULTS_DIR / name


def generation_path(name: str) -> Path:
    return GENERATIONS_DIR / name



