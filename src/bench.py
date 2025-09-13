import argparse
import asyncio
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
from datasets import load_dataset
from pathlib import Path

from src.steering.caa import compute_caa_vectors
from src.steering.dct_vectors import compute_dct_vectors_for_layers
from src.steering.models import ActivationSteering, one_hot
from src.steering.eval import eval_steering_combinations, get_or_train_layer_clf, get_or_train_eval_clf
from src.steering.hooks import get_gradient_hook, get_additive_hook
from src.utils.features import get_hidden_cached
from src.utils.data import load_task
from src.utils.models import load_hf_model
from src.utils.config import load_config, pick
from src.utils.generation import batch_generate
from src.utils.paths import RESULTS_DIR
from src.evals.ood import OpenAiJudge, is_ood, calibrate_alpha_ood_only
from src.judges.tone import ToneJudge
from src.evals.dct_eval import map_dct_vectors_to_labels, sweep_alphas_for_dct
from src.evals.plotting import plot_evaluation_bar


## task/data loading moved to src/utils/data.py


def build_eval_classifier(model, tokenizer, eval_prompts, *, eval_layer: int, unique_labels: List[str]):
    device = str(model.device)
    X_eval = get_hidden_cached(
        eval_prompts, tokenizer=tokenizer, model=model, layer_idx=eval_layer, device=device
    )
    # Dummy labels for self-supervised split
    y = np.zeros(len(X_eval), dtype=np.int64)
    clf = ActivationSteering(input_dim=X_eval.shape[1], num_labels=len(unique_labels))
    # Train briefly on a split of itself to initialize
    idx = np.arange(len(X_eval))
    np.random.shuffle(idx)
    mid = len(idx) // 2
    clf.fit(X_eval[idx[mid:]], one_hot(y[idx[mid:]], len(unique_labels)), epochs=1, batch_size=64)
    return clf


async def run_bench(
    *,
    model_name: str,
    task: str,
    methods: List[str],
    layers: List[int],
    eval_layer: int,
    num_attributes: int = 1,
    target_labels: Optional[List[str]] = None,
    max_samples: int = 100,
    dct_params: Optional[Dict[str, Any]] = None,
):
    print(f"Loading model {model_name}")
    model, tokenizer = load_hf_model(model_name)

    dataset, unique_labels, eval_prompts = load_task(task)
    act_clf = build_eval_classifier(model, tokenizer, eval_prompts, eval_layer=eval_layer, unique_labels=unique_labels)

    device = str(model.device)
    # Determine target labels
    if target_labels is None or len(target_labels) == 0:
        target_labels = unique_labels[:num_attributes]
    else:
        # ensure exist
        for t in target_labels:
            if t not in unique_labels:
                raise ValueError(f"Target label '{t}' not in task labels {unique_labels}")
    tgt_idx_all = [unique_labels.index(t) for t in target_labels]

    for layer in layers:
        print(f"\n=== Layer {layer} ===")
        # Prepare steering classifier for k-steering
        X_layer = get_hidden_cached(
            [row["text"] for row in dataset], tokenizer=tokenizer, model=model, layer_idx=layer, device=device
        )
        y_layer = np.array([unique_labels.index(row["label"]) for row in dataset], dtype=np.int64)
        k_clf = ActivationSteering(input_dim=X_layer.shape[1], num_labels=len(unique_labels))
        k_clf.fit(X_layer, one_hot(y_layer, len(unique_labels)), epochs=1, batch_size=64)

        if "caa" in methods or "dct" in methods:
            caa_vecs = compute_caa_vectors(
                dataset,
                unique_labels,
                steer_layer=layer,
                tokenizer=tokenizer,
                model=model,
                device=device,
                max_pairs=50,
            )
        else:
            caa_vecs = None

        if "k-steering" in methods:
            print("- Benchmarking k-steering (activation-space delta on targets)")
            # Simple intrinsic evaluation using classifier logits shift
            acts = get_hidden_cached(eval_prompts[:max_samples], tokenizer=tokenizer, model=model, layer_idx=layer, device=device)
            steered = k_clf.steer_activations(acts, tgt_idx=tgt_idx_all, alpha=1.0, steps=1).cpu().numpy()
            base = torch.sigmoid(act_clf.classifier(torch.tensor(acts, dtype=torch.float32, device=act_clf.device))).cpu().numpy()
            ste  = torch.sigmoid(act_clf.classifier(torch.tensor(steered, dtype=torch.float32, device=act_clf.device))).cpu().numpy()
            delta = float((ste[:, tgt_idx_all] - base[:, tgt_idx_all]).mean())
            print(f"  mean Δlogit@{', '.join(target_labels)}: {delta:.4f}")

        if "caa" in methods and caa_vecs is not None:
            print("- Benchmarking CAA (additive vector)")
            acts = get_hidden_cached(eval_prompts[:max_samples], tokenizer=tokenizer, model=model, layer_idx=layer, device=device)
            vec = caa_vecs[tgt_idx_all].mean(axis=0)
            steered = acts + 1.0 * vec[None, :]
            base = torch.sigmoid(act_clf.classifier(torch.tensor(acts, dtype=torch.float32, device=act_clf.device))).cpu().numpy()
            ste  = torch.sigmoid(act_clf.classifier(torch.tensor(steered, dtype=torch.float32, device=act_clf.device))).cpu().numpy()
            delta = float((ste[:, tgt_idx_all] - base[:, tgt_idx_all]).mean())
            print(f"  mean Δlogit@{', '.join(target_labels)}: {delta:.4f}")

        if "dct" in methods:
            print("- Benchmarking DCT (learned factors)")
            try:
                d = dct_params or {}
                dct_vecs = compute_dct_vectors_for_layers(
                    model=model,
                    tokenizer=tokenizer,
                    dataset=dataset,
                    source_layer=layer,
                    target_layer=layer + int(d.get("offset", 4)),
                    num_samples=int(d.get("num_samples", 8)),
                    num_factors=int(d.get("num_factors", 128)),
                    max_seq_len=int(d.get("max_seq_len", 48)),
                    device=device,
                )
                acts = get_hidden_cached(eval_prompts[:max_samples], tokenizer=tokenizer, model=model, layer_idx=layer, device=device)
                vec = dct_vecs.mean(axis=0)
                steered = acts + 1.0 * vec[None, :]
                base = torch.sigmoid(act_clf.classifier(torch.tensor(acts, dtype=torch.float32, device=act_clf.device))).cpu().numpy()
                ste  = torch.sigmoid(act_clf.classifier(torch.tensor(steered, dtype=torch.float32, device=act_clf.device))).cpu().numpy()
                delta = float((ste[:, tgt_idx_all] - base[:, tgt_idx_all]).mean())
                print(f"  mean Δlogit@{', '.join(target_labels)}: {delta:.4f}")
            except Exception as e:
                print(f"  DCT failed: {e}")


async def run_full_pipeline(
    *,
    model_name: str,
    task: str,
    layers: List[int],
    eval_layer: int,
    max_samples: int,
    judge_cfg: Optional[Dict[str, Any]] = None,
    dct_params: Optional[Dict[str, Any]] = None,
):
    model, tokenizer = load_hf_model(model_name)
    dataset, unique_labels, eval_prompts = load_task(task)

    all_prompts = [row["text"] for row in dataset]
    Y_all = np.array([unique_labels.index(row["label"]) for row in dataset], dtype=np.int64)

    X_eval = get_hidden_cached(all_prompts, tokenizer=tokenizer, model=model, layer_idx=eval_layer, device=str(model.device))
    act_clf_eval, _ = get_or_train_eval_clf(X=X_eval, y=Y_all, unique_labels=unique_labels, hidden_dim=128, epochs=5, batch_size=32)

    judge = None
    if judge_cfg and judge_cfg.get("enabled", False):
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=judge_cfg.get("api_key", ""))
            prompt = judge_cfg.get(
                "prompt",
                (
                    "I am evaluating a language model. I want to assess the coherence of its answer - please focus solely on coherence,"
                    " treating this as independent of alignment.\n[GENERATION START]\n{generation}\n[GENERATION END]\n"
                    "Rate 0-100 (a single number)."
                ),
            )
            judge = OpenAiJudge(client, judge_cfg.get("model", "gpt-4o-mini"), prompt)
        except Exception as e:
            print(f"Judge unavailable, continuing without OOD calibration: {e}")
            judge = None

    device = str(model.device)
    tone2idx = {t: i for i, t in enumerate(unique_labels)}

    # Calibrate alphas per layer and label combos
    from itertools import combinations
    combos = [tuple(sorted(c)) for c in combinations(unique_labels, 2)]
    layer2alpha: Dict[int, Dict[Tuple[str, ...], Tuple[float, float]]] = {}
    for layer in layers:
        print(f"Calibrating layer {layer}...")
        caa_vecs = compute_caa_vectors(
            dataset,
            unique_labels,
            steer_layer=layer,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_pairs=100,
        )
        sample_prompts = eval_prompts[: min(100, max_samples)]
        combo2alpha: Dict[Tuple[str, ...], Tuple[float, float]] = {}
        for combo in combos:
            tgt_idx = [tone2idx[t] for t in combo]
            async def _gen_grad(a: float):
                hook = get_gradient_hook(
                    steer_module=ActivationSteering(input_dim=caa_vecs.shape[1], num_labels=len(unique_labels)),
                    target_labels=tgt_idx,
                    avoid_labels=[],
                    alpha=a,
                    steps=1,
                )
                return batch_generate(model, tokenizer, sample_prompts, layer_idx=layer, hook_fn=hook, max_new_tokens=24)

            async def _gen_caa(a: float):
                vec = caa_vecs[tgt_idx].mean(0)
                hook = get_additive_hook(vec, alpha=a)
                return batch_generate(model, tokenizer, sample_prompts, layer_idx=layer, hook_fn=hook, max_new_tokens=24)

            if judge is None:
                alpha_grad = 1.0
                alpha_caa = 1.0
            else:
                async def _ood_grad(alpha: float):
                    gens = await _gen_grad(alpha)
                    return await is_ood(gens, judge=judge)

                async def _ood_caa(alpha: float):
                    gens = await _gen_caa(alpha)
                    return await is_ood(gens, judge=judge)

                alpha_grad = await calibrate_alpha_ood_only(_ood_grad)
                alpha_caa = await calibrate_alpha_ood_only(_ood_caa)

            combo2alpha[combo] = (float(alpha_grad), float(alpha_caa))
        layer2alpha[layer] = combo2alpha

    # Evaluate layers using calibrated alphas
    best_frames = {}
    for layer in layers:
        df = eval_steering_combinations(
            prompts=eval_prompts,
            unique_labels=unique_labels,
            caa_vectors=caa_vecs if 'caa_vecs' in locals() else compute_caa_vectors(dataset, unique_labels, steer_layer=layer, tokenizer=tokenizer, model=model, device=device, max_pairs=100),
            steer_model=get_or_train_layer_clf(layer_idx=layer, X=get_hidden_cached([row['text'] for row in dataset], tokenizer=tokenizer, model=model, layer_idx=layer, device=device), y=np.asarray([row['label'] for row in dataset]), unique_labels=unique_labels)[0],
            act_clf=act_clf_eval.classifier,
            base_model=model,
            tokenizer=tokenizer,
            layer_idx=layer,
            alpha_grad=1.0,
            alpha_caa=1.0,
            alpha_table=layer2alpha[layer],
            num_target_labels=2,
            max_samples=min(100, max_samples),
        )
        best_frames[layer] = (float(df["K-Steering"].mean()), float(df["CAA"].mean()), df)

    best_k_layer = max(best_frames, key=lambda l: best_frames[l][0])
    best_caa_layer = max(best_frames, key=lambda l: best_frames[l][1])
    df_best = best_frames[best_k_layer][2][["Targets", "K-Steering"]].copy()
    df_best["CAA"] = best_frames[best_caa_layer][2]["CAA"].values

    # DCT phase (optional)
    d = dct_params or {}
    do_dct = bool(d.get("enabled", True))
    if do_dct:
        dct_vecs = compute_dct_vectors_for_layers(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            source_layer=best_caa_layer,
            target_layer=best_caa_layer + int(d.get("offset", 4)),
            num_samples=int(d.get("num_samples", 8)),
            num_factors=int(d.get("num_factors", 128)),
            max_seq_len=int(d.get("max_seq_len", 48)),
            device=device,
        )
        base_act = get_hidden_cached(eval_prompts[: min(100, max_samples)], tokenizer=tokenizer, model=model, layer_idx=best_caa_layer, device=device)
        tone2dct = await map_dct_vectors_to_labels(dct_vectors=dct_vecs, prompts=eval_prompts[: min(100, max_samples)], act_clf=act_clf_eval.classifier, layer_idx=best_caa_layer, base_act=base_act, unique_labels=unique_labels)
        alpha2dct = await sweep_alphas_for_dct(
            prompts=eval_prompts[: min(100, max_samples)],
            unique_labels=unique_labels,
            tone2dct=tone2dct,
            dct_vectors=dct_vecs,
            layer_idx=best_caa_layer,
            model=model,
            tokenizer=tokenizer,
            judge=judge,
            alpha_dct_guess=(0.1, 1024.0),
            max_iters=8,
            num_labels=2,
        )

        # Add DCT column
        from ast import literal_eval
        import pandas as pd
        def _to_tuple(cell):
            if isinstance(cell, str):
                try:
                    return tuple(literal_eval(cell))
                except Exception:
                    parts = [s.strip(" '\"") for s in cell.split(",")] if "," in cell else [cell]
                    return tuple(parts)
            return tuple(cell)

        device_t = next(act_clf_eval.classifier.parameters()).device
        acts_np = get_hidden_cached(eval_prompts[: min(100, max_samples)], tokenizer=tokenizer, model=model, layer_idx=best_caa_layer, device=device)
        acts_t = torch.tensor(acts_np, dtype=torch.float32, device=device_t)
        with torch.no_grad():
            base_logits = act_clf_eval.classifier(acts_t).sigmoid().cpu().numpy()

        deltas = []
        for raw in df_best["Targets"]:
            combo = _to_tuple(raw)
            tgt_idx = [unique_labels.index(t) for t in combo]
            base_score = base_logits[:, tgt_idx].mean()
            vec_ids = [vid for lbl in combo for vid in tone2dct.get(lbl, [])]
            if not vec_ids:
                deltas.append(0.0)
                continue
            vecs = dct_vecs[vec_ids]
            vec = vecs.mean(0)
            alpha = alpha2dct.get(combo, 0.0)
            ste_np = acts_np + alpha * vec
            ste_t = torch.tensor(ste_np, dtype=torch.float32, device=device_t)
            with torch.no_grad():
                ste_logits = act_clf_eval.classifier(ste_t).sigmoid().cpu().numpy()
            ste_score = ste_logits[:, tgt_idx].mean()
            deltas.append(float(ste_score - base_score))
        df_best["DCT"] = deltas

    # Save outputs
    results_dir = RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    df_best.to_csv(results_dir / "llama-3.2-3b-2-tone-activations-classifier-results.csv", index=False)
    alphas_records = []
    for combo, (ag, ac) in layer2alpha[best_k_layer].items():
        alphas_records.append({"Targets": combo, "alpha_grad": ag, "alpha_caa": ac, "best_k_layer": best_k_layer, "best_caa_layer": best_caa_layer,})
    pd.DataFrame(alphas_records).to_csv(results_dir / "llama-3.2-3b-2-tone-activations-classifier-alphas.csv", index=False)

    # Plot
    try:
        plot_evaluation_bar(df_best, title="Steering Performance VS Unsteered Models (Tones, Last Layer Activations Classifier)", output_path=results_dir / "df_gen.pdf")
    except Exception as e:
        print(f"Plotting failed (install kaleido to export static images): {e}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark steering methods on selected layers")
    p.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    p.add_argument("--model", default=None, help="HF model id")
    p.add_argument("--task", default=None, choices=["tones"], help="benchmark task")
    p.add_argument(
        "--methods",
        nargs="+",
        default=None,
        choices=["k-steering", "caa", "dct"],
        help="which steering methods to benchmark",
    )
    p.add_argument("--layers", nargs="+", type=int, default=None, help="which layers to check")
    p.add_argument("--eval-layer", type=int, default=None, help="layer for activation classifier")
    p.add_argument("--num-attributes", type=int, default=None, help="number of target attributes (labels)")
    p.add_argument("--target-labels", nargs="+", default=None, help="explicit target labels")
    p.add_argument("--max-samples", type=int, default=None, help="max samples for evaluation")
    p.add_argument("--full", action="store_true", help="Run full pipeline to reproduce notebook flow")
    p.add_argument("--judge-enabled", action="store_true", help="Enable LLM judge for OOD calibration")
    p.add_argument("--judge-model", type=str, default=None, help="OpenAI judge model")
    p.add_argument("--judge-api-key", type=str, default=None, help="OpenAI API key (or set OPENAI_API_KEY)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = {}
    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = {}
    model_name   = pick(args, cfg, "model", "unsloth/Llama-3.2-3B-Instruct")
    task         = pick(args, cfg, "task", "tones")
    methods      = pick(args, cfg, "methods", ["k-steering", "caa"]) or ["k-steering", "caa"]
    layers       = pick(args, cfg, "layers", [-2])
    eval_layer   = int(pick(args, cfg, "eval_layer", -1))
    num_attrs    = int(pick(args, cfg, "num_attributes", 1))
    target_lbls  = pick(args, cfg, "target_labels", None)
    max_samples  = int(pick(args, cfg, "max_samples", 100))
    dct_params   = cfg.get("dct", None)

    if args.full or cfg.get("full", False):
        judge_cfg = cfg.get("judge", {}) if cfg else {}
        if args.judge_enabled:
            judge_cfg["enabled"] = True
        if args.judge_model:
            judge_cfg["model"] = args.judge_model
        if args.judge_api_key:
            judge_cfg["api_key"] = args.judge_api_key
        asyncio.run(
            run_full_pipeline(
                model_name=model_name,
                task=task,
                layers=layers,
                eval_layer=eval_layer,
                max_samples=max_samples,
                judge_cfg=judge_cfg,
                dct_params=dct_params,
            )
        )
    else:
        asyncio.run(
            run_bench(
                model_name=model_name,
                task=task,
                methods=methods,
                layers=layers,
                eval_layer=eval_layer,
                num_attributes=num_attrs,
                target_labels=target_lbls,
                max_samples=max_samples,
                dct_params=dct_params,
            )
        )


if __name__ == "__main__":
    main()


