# Nonlinear Steering (Refactored)

This repository explores and benchmarks nonlinear steering methods for LLMs, including:
- K-Steering (gradient-based activation steering)
- CAA (Contrastive Activation Additions)
- DCT-based steering vectors (Exponential DCT factors)

The codebase has been modularized under `src/` with clear separation between steering primitives, utilities, evaluation flows, and judges.

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

1) Prepare a config (edit `configs/bench.example.yaml` as needed):

```yaml
model: unsloth/Llama-3.2-3B-Instruct
task: tones
methods: [k-steering, caa, dct]
layers: [-2, -1]
eval_layer: -1
num_attributes: 1
# target_labels: [casual, empathetic]
max_samples: 100
# DCT-specific params
dct:
  offset: 4
  num_samples: 8
  num_factors: 128
  max_seq_len: 48
```

2) Run a quick intrinsic benchmark (activation deltas):

```bash
python -m src --config configs/bench.example.yaml
```

3) Run the full pipeline (calibration → evaluation → DCT → save results/plots):

```bash
python -m src --config configs/bench.example.yaml --full
```

4) Enable LLM-judge OOD calibration (requires OpenAI key):

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

## Notes and tips

- The tones task is implemented. If you need the debates task, open an issue or extend `src/utils/data.py` similarly to tones.
- For faster experimentation, limit `layers` and `max_samples` in the YAML.
- Static figure export requires `kaleido`.

## Development

- Code is structured for clarity and testability. Core steering logic lives under `src/steering/`; high-level orchestration and utilities are in `src/evals/` and `src/utils/`.
- Feel free to contribute additional tasks, judges, or evaluation metrics.
