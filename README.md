# Beyond Linear Steering: Unified Multi-Attribute Control for Language Models

This repository accompanies the paper [Beyond Linear Steering: Unified Multi-Attribute Control for Language Models](https://www.arxiv.org/abs/2505.24535), allowing for the results to be replicated. The multi-layer evaluation from Section 5 of the paper is implemented in `src/`, and the code for other experiments is accessible in `notebooks/`.

## Installation

- Python 3.10+
- Install dependencies:

```bash
pip install -r requirements.txt
```

Some features (plot export, judge-based calibration) require optional deps and API keys:
- Static image export: `kaleido`
- OpenAI judge calibration and tone-judge: `openai` + an API key, and `tiktoken`

## Directory layout

- `src/steering/` – core steering primitives
  - `models.py` – activation classifier and steering model
  - `hooks.py` – gradient/additive forward hooks
  - `caa.py` – CAA vector computation
  - `dct_vectors.py` – DCT steering vectors
  - `eval.py` – activation-space evaluation helpers
- `src/utils/` – utilities and helpers
  - `data.py` – task/dataset loading (tones)
  - `models.py` – HF model loader
  - `features.py` – hidden state extraction (`get_hidden_cached`)
  - `generation.py` – batched generation with hooks
  - `paths.py` – `results/`, `data/`, `data/generations` paths
  - `config.py`, `tasks.py`, `deps.py`, `model_ops.py`, `state.py`
- `src/evals/` – evaluation flows
  - `ood.py` – judge-based OOD scoring & alpha calibration
  - `dct_eval.py` – DCT-to-label mapping and alpha sweep
  - `plotting.py` – bar chart plotting
  - `samples.py` – sampling steered responses and saving
- `src/judges/` – judge integrations
  - `tone.py` – tone comparison judge (tiktoken-based first-token mapping)
- `configs/` – example YAML configs
- `data/` – local datasets and generated artifacts (created on first run)
  - `generations/` – generated samples
- `results/` – evaluation results (CSVs, figures)

## Quickstart

1) Set a yaml config in `configs/`. There is an example, minimal config that runs quickly available.

2) Run the full pipeline (calibrate alphas → find the optimal layer and alpha combination → save results/plots):

```bash
python -m src --config configs/bench.example.yaml --full
```

3) Enable LLM-judge OOD calibration (requires OpenAI key):

```bash
python -m src --config configs/bench.example.yaml --full \
  --judge-enabled --judge-model gpt-4o-mini --judge-api-key YOUR_OPENAI_KEY
```

Outputs will be written to `results/` (CSVs and figures), and generated samples (if any) to `data/generations/`.

## CLI overview

- `--config` – path to YAML config (see example above)
- `--model` – HF model id (overrides YAML)
- `--task` – `tones`
- `--methods` – any of `k-steering`, `caa`, `dct` (quick benchmark mode)
- `--layers` – list of layer indices to benchmark
- `--eval-layer` – layer used for the activation classifier
- `--num-attributes` / `--target-labels` – target labels selection
- `--max-samples` – cap prompts used in evaluation
- `--full` – run the full notebook-equivalent pipeline
- Judge flags: `--judge-enabled`, `--judge-model`, `--judge-api-key`