import argparse
import functools
import json
import torch
from datetime import datetime
import numpy as np
import random
from torch import load as torch_load
from transformer_lens import HookedTransformer, utils
from transformers import AutoTokenizer
from judges.debates_judge import DebateJudge
from your_hooks_module import actadd_debate_steering_hook
from your_generation_module import get_generations, tokenize_instructions_fn, strawman_inst_test
from your_steering_model import MultiToneActivationSteering

steps_to_try = list(range(1, 11))
N_INST_TEST = 20
intervention_layers = list(range(28))  # Adjust to match your model
act_names = ['resid_pre', 'resid_mid', 'resid_post']


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_alpha(alpha: float, target_style: str, avoid_style: str):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    print(f"[INFO] Loading model and judge on device {device}...")
    model = HookedTransformer.from_pretrained_no_processing(
        'meta-llama/Llama-3.2-3B-Instruct',
        device=device,
        dtype=torch.float16,
        default_padding_side='left'
    )
    model.tokenizer.padding_side = 'left'
    model.tokenizer.pad_token = model.tokenizer.eos_token

    judge = DebateJudge(api_key="your_key")

    all_activations = torch_load("debate_activations.pt", weights_only=False)
    debate_steer = MultiToneActivationSteering(input_dim=all_activations['reductio'].shape[1], num_classes=10)
    debate_steer.fit({
        'reductio': all_activations['reductio'],
        'precedent': all_activations['precedent'],
        'strawman': all_activations['strawman'],
        'burden': all_activations['burden'],
        'analogy': all_activations['analogy'],
        'concession': all_activations['concession'],
        'empirical': all_activations['empirical'],
        'moral': all_activations['moral'],
        'refutation': all_activations['refutation'],
        'circular': all_activations['circular']
    }, epochs=30, batch_size=32)

    # Baseline generations
    baseline_generations = get_generations(
        model,
        strawman_inst_test[:N_INST_TEST],
        tokenize_instructions_fn,
        fwd_hooks=[]
    )

    alpha_log = {
        "meta": {
            "run_time": datetime.now().isoformat(),
            "alpha": alpha,
            "target_style": target_style,
            "avoid_style": avoid_style,
            "n_instances": N_INST_TEST,
            "intervention_layers": intervention_layers,
            "activation_points": act_names
        },
        "baseline_generations": baseline_generations,
        "step_runs": []
    }

    for steps in steps_to_try:
        print(f"\n[INFO] Alpha {alpha} | Steps = {steps}")

        hook_fn = functools.partial(
            actadd_debate_steering_hook,
            target_styles=[target_style],
            avoid_styles=[avoid_style],
            alpha=alpha,
            steps=steps
        )

        fwd_hooks = [
            (utils.get_act_name(act_name, l), hook_fn)
            for l in intervention_layers
            for act_name in act_names
        ]

        steered_generations = get_generations(
            model,
            strawman_inst_test[:N_INST_TEST],
            tokenize_instructions_fn,
            fwd_hooks=fwd_hooks
        )

        result = judge.evaluate_batch(
            baseline_generations,
            steered_generations,
            target_style.title(),
            avoid_style.title()
        )

        print(f"Success Rate: {result['success_rate']}")
        print(f"Average Strength: {result['average_strength']}")

        alpha_log["step_runs"].append({
            "steps": steps,
            "success_rate": result["success_rate"],
            "average_strength": result["average_strength"],
            "steered_generations": steered_generations
        })

        if result["success_rate"] == 0 and result["average_strength"] == 0:
            print(f"[WARN] Early stopping at alpha={alpha}, step={steps}")
            break

    safe_alpha = str(round(alpha, 2)).replace(".", "_")
    output_path = f"steering_{target_style}_minus_{avoid_style}_alpha_{safe_alpha}.json"
    with open(output_path, "w") as f:
        json.dump(alpha_log, f, indent=2)

    print(f"[DONE] Saved results to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, required=True, help="Alpha value for steering strength")
    parser.add_argument("--target", type=str, required=True, help="Target debate style")
    parser.add_argument("--avoid", type=str, required=True, help="Debate style to avoid")
    args = parser.parse_args()

    run_alpha(args.alpha, args.target, args.avoid)
