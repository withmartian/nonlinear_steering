import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.benchmarks import MMLU

# If you use transformers/torch for something else, add them back.
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch

APPEND_STRING = "Output A, B, C, or D. Full answer not needed. Answer:"


def after_x(s: str, x: str) -> str:
    parts = s.split(x, 1)
    return parts[1] if len(parts) > 1 else ""


class CacheReaderModel(DeepEvalBaseLLM):
    """
    Minimal LLM adapter that returns the first character of the cached answer
    (A/B/C/D) for exactly-matching prompts from the CSV.
    """
    def __init__(self, df_dict: Dict[str, str], name: str):
        self.df_dict = df_dict
        self.cache_hits = 0
        self._name = name

    def load_model(self):
        pass

    def generate(self, prompt: str) -> str:
        if prompt in self.df_dict:
            self.cache_hits += 1
        final_result = str(self.df_dict.get(prompt, "")).strip()
        # If your prompts included APPEND_STRING during generation and you want to
        # slice after it, uncomment the next line:
        # final_result = after_x(final_result, APPEND_STRING).strip()
        return final_result[:1] if final_result else ""

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def batch_generate(self, prompts: List[str]) -> List[str]:
        return [self.generate(p) for p in prompts]

    def get_model_name(self):
        return self._name


def parse_args():
    p = argparse.ArgumentParser(
        description="Run MMLU over cached CSV generations (CAA and K-Steering), aggregate per-file and averages."
    )
    p.add_argument(
        "--model_name",
        type=str,
        default="llama-3.2-3b",
        help="Model folder name (e.g., 'llama-3.2-3b', 'mistral-7b'). Must match the generation output folder.",
    )
    p.add_argument(
        "--combo_num",
        type=int,
        default=1,
        help="Combo size folder to scan under model_name/ (e.g., 1, 2, 3).",
    )
    p.add_argument(
        "--n_problems_per_task",
        type=int,
        default=3,
        help="MMLU: number of problems per task."
    )
    p.add_argument(
        "--n_shots",
        type=int,
        default=3,
        help="MMLU: number of few-shot examples."
    )
    return p.parse_args()


def load_df_dict(csv_path: Path, column_name: str) -> Dict[str, str]:
    df = pd.read_csv(csv_path)
    if "prompt" not in df.columns:
        raise ValueError(f"{csv_path} missing required 'prompt' column")
    if column_name not in df.columns:
        raise ValueError(f"{csv_path} missing required '{column_name}' column")
    return dict(zip(df["prompt"], df[column_name]))


def run_mmlu_with_cache(
    df_dict: Dict[str, str],
    run_name: str,
    n_problems_per_task: int,
    n_shots: int
) -> float:
    llm = CacheReaderModel(df_dict=df_dict, name=run_name)
    benchmark = MMLU(
        n_problems_per_task=n_problems_per_task,
        n_shots=n_shots,
        confinement_instructions=APPEND_STRING,
    )
    result = benchmark.evaluate(model=llm)

    # Try to extract a numeric score from result robustly.
    # Depending on deepeval version, result may be an object or a dict.
    score = None

    print(f"Result is {result}")

    for key in ("score", "overall_score", "accuracy", "overall_accuracy"):
        if hasattr(result, key):
            score = getattr(result, key)
            break
        if isinstance(result, dict) and key in result:
            score = result[key]
            break

    if score is None:
        # Fallback: Some versions may store results in nested structures.
        # Adjust here if your local deepeval exposes it differently.
        raise RuntimeError(
            "Could not extract a numeric score from MMLU result. Inspect the 'result' object/API."
        )

    return float(score)


def main():
    args = parse_args()
    csv_dir = Path(args.model_name) / str(args.combo_num)
    if not csv_dir.exists():
        raise FileNotFoundError(
            f"Folder not found: {csv_dir} (expected CSVs under {args.model_name}/{args.combo_num}/)"
        )

    csv_paths = sorted(csv_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under {csv_dir}")

    target_columns = ["unsteered", "caa", "k_steering"]
    results: Dict[str, float] = {}

    for col in target_columns:
        print(f"Processing column {col}")
        per_file_scores: List[float] = []
        for csv_path in csv_paths:
            print(f"Processing path {csv_path}")
            df_dict = load_df_dict(csv_path, col)
            run_name = f"{col}:{csv_path.name}"

            score = run_mmlu_with_cache(
                df_dict=df_dict,
                run_name=run_name,
                n_problems_per_task=args.n_problems_per_task,
                n_shots=args.n_shots,
            )
            results[f"{col}_{csv_path.name}"] = score
            per_file_scores.append(score)
            print(f"[{col}] {csv_path.name} => {score:.4f}")

        avg = sum(per_file_scores) / len(per_file_scores) if per_file_scores else None
        results[f"{col}_avg"] = avg
        if avg is not None:
            print(f"[{col}] AVERAGE over {len(per_file_scores)} files => {avg:.4f}")
        else:
            print(f"[{col}] No scores computed.")

    out_json = f"{args.model_name}_{args.combo_num}.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_json}")


if __name__ == "__main__":
    main()
