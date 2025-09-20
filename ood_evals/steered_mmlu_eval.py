import asyncio, hashlib, math, os, random, sys, copy, gc, re, ast, json, uuid, html as _html
from contextlib import contextmanager
import collections
import joblib
from collections import defaultdict, Counter
from pathlib import Path
from types import SimpleNamespace
from typing import List, Dict, Tuple, Optional, DefaultDict, Callable, Union, Callable, Sequence, Mapping
from urllib.request import urlopen
import importlib.util, sys, copy, torch, itertools
from itertools import combinations
from functools import lru_cache
import html

import joblib
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from openai import AsyncOpenAI
import plotly.express as px
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
)
from tqdm.auto import tqdm
import ipywidgets as widgets
from IPython.display import display, clear_output



from pprint import pprint
import numpy as np
from pathlib import Path
import torch
import pandas as pd
from ast import literal_eval
import random
torch.set_float32_matmul_precision('high')

# --- add near the other imports ---
import argparse
from pathlib import Path

# --- new: CLI args ---
def parse_args():
    p = argparse.ArgumentParser(description="Run steering over all combos and save per-combo CSVs.")
    p.add_argument(
        "--model_name",
        type=str,
        default="llama-3.2-3b",
        choices=["llama-3.2-3b", "olmo-2-7b", "mistral-7b"],
        help="Short model alias used by get_model() and output folders.",
    )
    p.add_argument(
        "--num_combo",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Size of target tone combinations.",
    )
    p.add_argument(
        "--dataset_name",
        type=str,
        default="tone",
        choices=["tone", "debate"],
        help="Which task/dataset loader to use.",
    )
    p.add_argument(
        "--prompts_path",
        type=Path,
        default=Path("prompts_dump.json"),
        help="Path to JSON list of prompts.",
    )
    return p.parse_args()

def one_hot(idxs: np.ndarray, C: int) -> np.ndarray:
    out = np.zeros((len(idxs), C), dtype=np.float32)
    out[np.arange(len(idxs)), idxs] = 1.0
    return out

class MultiLabelSteeringModel(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_labels: int,
                 linear: bool = False):
        super().__init__()
        if linear:
            self.net = nn.Linear(input_dim, num_labels)
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_labels),
            )

    def forward(self, x):
        return self.net(x)

class ActivationSteering:
    def __init__(self, input_dim, num_labels, hidden_dim=128, lr=1e-3):
        self.device = 'cuda'
        self.num_labels = num_labels

        self.classifier = MultiLabelSteeringModel(
            input_dim, hidden_dim, num_labels
        ).to(self.device)

        self.optimizer = optim.Adam(self.classifier.parameters(), lr=lr)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def fit(self, X, Y, epochs=10, batch_size=8):
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        Y_t = torch.tensor(Y, dtype=torch.float32, device=self.device)

        dataset = torch.utils.data.TensorDataset(X_t, Y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for ep in range(epochs):
            total_loss = 0.0
            for bx, by in loader:
                self.optimizer.zero_grad()
                logits = self.classifier(bx)
                loss = self.loss_fn(logits, by)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {ep+1}/{epochs}, Loss={total_loss/len(loader):.4f}")

    @torch.no_grad()
    def predict_proba(self, X):
        self.classifier.eval()
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        logits = self.classifier(X_t)
        probs = torch.sigmoid(logits)
        return probs.cpu().numpy()

    def steer_activations(
        self,
        acts: Union[np.ndarray, torch.Tensor],
        target_idx: List[int],
        avoid_idx: List[int] = [],
        alpha: float = 1.0,
        steps: int = 1,
        step_size_decay: float = 1.0,
    ) -> torch.Tensor:
        if isinstance(acts, np.ndarray):
            acts = torch.as_tensor(acts, dtype=torch.float32, device=self.device)
        else:
            acts = acts.to(self.device, dtype=torch.float32)

        steered = acts.detach().clone()

        for step in range(steps):
            curr = steered.clone().requires_grad_(True)
            logits = self.classifier(curr)

            loss_vec = _compute_steering_loss(
                logits, target_idx=target_idx, avoid_idx=avoid_idx
            )

            loss = loss_vec.mean()
            grads = torch.autograd.grad(loss, curr, retain_graph=False)[0]

            current_alpha = alpha * (step_size_decay ** step)
            steered = (curr - current_alpha * grads).detach()

        return steered

def get_or_train_layer_clf(unique_labels, X: np.ndarray, y: np.ndarray,
                           *, hidden_dim=128, epochs=5, batch_size=32):
    if y.dtype.kind not in ("i", "u"):
        lbl2idx = {lbl: i for i, lbl in enumerate(unique_labels)}
        y = np.asarray([lbl2idx[lbl] for lbl in y], dtype=np.int64)


    idx_A, idx_B = train_test_split(np.arange(len(X)), test_size=0.5, random_state=42, stratify=y)
    X_A, X_B, y_A, y_B = X[idx_A], X[idx_B], y[idx_A], y[idx_B]

    clf = ActivationSteering(input_dim=X.shape[1], num_labels=len(unique_labels), hidden_dim=hidden_dim)
    clf.fit(X_A, one_hot(y_A, len(unique_labels)), epochs=epochs, batch_size=batch_size)

    with torch.no_grad():
        acc = (torch.argmax(
            clf.classifier(torch.tensor(X_B, dtype=torch.float32, device=clf.device)),
            dim=1).cpu().numpy() == y_B).mean()

    return clf, acc

def get_hidden_cached(model, tokenizer, texts: List[str], layer_idx: int, *, batch_size: int = 8) -> np.ndarray:
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        tok = tokenizer(batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True).to('cuda')
        with torch.no_grad():
            out = model(**tok, output_hidden_states=True)
        h = out.hidden_states[layer_idx]
        mask = tok["attention_mask"]
        lengths = mask.sum(dim=1) - 1

        for j, idx in enumerate(lengths):
            all_vecs.append(h[j, idx, :].cpu().float().numpy())

    return np.stack(all_vecs, axis=0)

def batch_generate(
    model,
    tokenizer,
    prompts: List[str],
    layer_idx: int,
    hook_fn: Optional[Callable] = None,
    max_new_tokens: int = 24,
    batch_size: int =8,
) -> List[str]:
    device        = model.device
    target_layer  = model.model.layers[layer_idx]
    outputs: List[str] = []

    saved_hooks = target_layer._forward_hooks.copy()
    target_layer._forward_hooks.clear()

    handle = None
    if hook_fn is not None:
        handle = target_layer.register_forward_hook(hook_fn)

    try:
        for i in range(0, len(prompts), batch_size):
            sub_prompts = prompts[i : i + batch_size]
            tok_in = tokenizer(
                sub_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)

            with torch.no_grad():
                gen_ids = model.generate(
                    **tok_in,
                    max_new_tokens = max_new_tokens,
                    do_sample      = False,
                    pad_token_id   = tokenizer.eos_token_id,
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

def get_model(model_name= "llama-3.2-3b"):

    if  model_name == "llama-3.2-3b":
        model_name = "unsloth/Llama-3.2-3B-Instruct"
    elif model_name == "olmo-2-7b":
        model_name = "allenai/OLMo-2-1124-7B-Instruct"   #path to olmo model
    else:
        model_name = "mistralai/Mistral-7B-Instruct-v0.3" #path to mistral model

    print(f"Loading {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        _attn_implementation="eager",
        output_hidden_states=True,
    ).to("cuda:0")

    model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    return model, tokenizer

def _load_tones():
    ds = load_dataset("Narmeen07/tone_agnostic_questions", split="train")
    steered_prompts = [
        ("expert",     "RESPOND AS AN AUTHORITATIVE EXPERT. Use advanced terminology and industry-specific jargon. Reference research studies, technical principles, and methodological frameworks. Maintain a formal, academic tone throughout. Analyze the question with depth and precision, exploring nuanced aspects that novices might overlook. Cite relevant theories, standards, or established practices. Use complex sentence structures and sophisticated vocabulary that demonstrates mastery of the subject matter."),
        ("cautious",   "RESPOND WITH EXTREME CAUTION AND QUALIFICATION. Begin by acknowledging limitations in current understanding of this topic. Present multiple perspectives and avoid making definitive claims. Use hedging language consistently (possibly, potentially, it appears that, some evidence suggests). Highlight risks, uncertainties, and areas of controversy. Include multiple disclaimers and caveats. Acknowledge what remains unknown or debated among experts. Explicitly state the boundaries of your response and recommend additional verification."),
        ("empathetic", "RESPOND WITH DEEP EMOTIONAL UNDERSTANDING. Begin by validating the feelings that might accompany this question. Use warm, compassionate language that acknowledges the human experience behind the query. Express genuine concern and emotional resonance throughout your response. Use phrases like 'I understand this might feel...' or 'Many people find this emotionally challenging because...' Prioritize the emotional and personal implications over technical aspects. Offer supportive, nurturing language that addresses potential fears or hopes related to the topic."),
        ("casual",     "You are an AI assistant responding with a casual tone. Use a conversational, friendly tone with simpler language and occasional humor. Be relatable and informal, as if chatting with a friend."),
        ("concise",   "BE EXTREMELY BRIEF. Use minimal words. Provide only essential information. No introductions. No elaboration. Use short sentences with simple structure. Eliminate all unnecessary words. Focus solely on direct answers. Avoid examples unless critical. Use bullet points where possible. Prioritize brevity over comprehensiveness. Get to the point immediately. Never repeat information.")
    ]
    rel_template = (
        "You will compare two answers to the *same* question.\n\n"
        "Question:\n[QUESTION START]\n{question}\n[QUESTION END]\n\n"
        "Original answer:\n[BASE START]\n{base_answer}\n[BASE END]\n\n"
        "Modified answer:\n[STEERED START]\n{steered_answer}\n[STEERED END]\n\n"
        "Compared **to the original answer**, which single tone label best describes the *modified* answer?\n"
        "Allowed labels: cautious, empathetic, expert, helpful, casual, concise.\n"
        "Respond with **only** one label."
    )
    return SimpleNamespace(
        name="tone",
        raw_dataset = ds,
        steered_prompts = steered_prompts,
        rel_template = rel_template,
        eval_prompts = list(ds["text"]),
        unique_labels = sorted({t for t, _ in steered_prompts}),
    )

def _load_debates():
    ds = load_dataset("Narmeen07/debate_style_agnostic_questions", split="train")

    steered_prompts = [
        ("Reductio ad Absurdum", "RESPOND USING REDUCTIO AD ABSURDUM DEBATE STYLE. Extend the opposing viewpoint to its logical extreme to reveal its flaws. Demonstrate how the opposing position leads to absurd or contradictory conclusions when taken to its logical end. Structure your response to systematically expose the inconsistencies or impracticalities. Use conditional statements like 'If we follow this logic, then...' to illustrate the path to absurdity. Maintain a logical flow that shows how the initial premise inevitably leads to an unacceptable or ridiculous outcome.\n\n"),
        ("Appeal to Precedent", "RESPOND USING APPEAL TO PRECEDENT DEBATE STYLE. Ground your arguments in historical examples, established case law, or widely accepted previous decisions. Reference specific historical situations, court rulings, scientific discoveries, or cultural precedents that support your position. Draw explicit connections between these precedents and the current issue. Emphasize how respecting established precedents provides consistency and reliability. Point out the risks or inconsistencies of ignoring relevant precedents. Structure your response to show how past experiences inform present considerations.\n\n"),
        ("Straw Man Reframing", "RESPOND USING STRAW MAN REFRAMING DEBATE STYLE. Recharacterize the opposing argument in simplified or exaggerated terms that are easier to refute. Present this reframed version as if it represents the original position. Systematically dismantle this reframed argument while implying you've addressed the original point. Use phrases like 'Essentially, what you're saying is...' or 'This position boils down to...' before presenting the simplified version. Follow with a thorough refutation of this reframed position. Conclude by suggesting that your refutation applies to the original argument as well.\n\n"),
        ("Burden of Proof Shift", "RESPOND USING BURDEN OF PROOF SHIFT DEBATE STYLE. Redirect the responsibility for providing evidence to your opponent rather than proving your own claim. Challenge your opponent to disprove your assertion rather than supporting it yourself. Use phrases like 'There's no evidence that disproves...' or 'Can you definitively prove that isn't the case?' Position your claims as the default assumption that should be accepted until disproven. Question the sufficiency or quality of opposing evidence while demanding an impossibly high standard of proof. Emphasize that the lack of counter-evidence strengthens your position.\n\n"),
        ("Analogy Construction", "RESPOND USING ANALOGY CONSTRUCTION DEBATE STYLE. Develop a vivid, relatable comparison between the complex issue at hand and something more familiar or intuitive. Build your argument around this carefully constructed parallel situation. Highlight specific points of similarity that support your position while addressing potential dissimilarities. Use phrases like 'This situation is similar to...' or 'To understand this concept, consider...' Ensure your analogy simplifies the complex issue without distorting its essential nature. Use the familiar scenario to guide your audience to your desired conclusion about the original issue.\n\n"),
        ("Concession and Pivot", "RESPOND USING CONCESSION AND PIVOT DEBATE STYLE. Begin by acknowledging a minor point or critique from the opposing side to establish fairness and reasonableness. Use phrases like 'While it's true that...' or 'I can concede that...' followed by 'However,' 'Nevertheless,' or 'That said,' to redirect to your stronger arguments. Ensure the conceded point is peripheral rather than central to your main argument. After the concession, pivot decisively to your strongest points with increased emphasis. Frame your pivot as providing necessary context or a more complete perspective. Use the concession to demonstrate your objectivity before delivering your more powerful counterarguments.\n\n"),
        ("Empirical Grounding", "RESPOND USING EMPIRICAL GROUNDING DEBATE STYLE. Base your arguments primarily on verifiable data, research studies, statistics, and observable outcomes rather than theory or rhetoric. Cite specific figures, percentages, study results, or historical outcomes that support your position. Present evidence in a methodical manner, explaining how each piece of data relates to your argument. Address the reliability and relevance of your sources and methods. Compare empirical results across different contexts or time periods to strengthen your case. Anticipate and address potential methodological criticisms of the evidence you present.\n\n"),
        ("Moral Framing", "RESPOND USING MORAL FRAMING DEBATE STYLE. Position the issue within a framework of ethical principles, values, and moral imperatives rather than pragmatic concerns. Identify the core moral values at stake such as justice, liberty, equality, compassion, or responsibility. Use language that evokes ethical considerations, such as 'obligation,' 'right,' 'wrong,' 'just,' or 'fair.' Appeal to widely held moral intuitions or principles. Present opposing views as morally questionable or inconsistent with important shared values. Elevate the discussion from practical matters to questions of what ought to be done. Emphasize moral consequences over practical outcomes.\n\n"),
        ("Refutation by Distinction", "RESPOND USING REFUTATION BY DISTINCTION DEBATE STYLE. Identify crucial differences that invalidate comparisons or principles your opponent has applied. Carefully delineate categories, contexts, or circumstances that demonstrate why a general rule or example doesn't apply in this specific case. Use phrases like 'While that may be true in some contexts...' or 'We must distinguish between...' Emphasize the precision of definitions and classifications. Highlight subtle but significant differences that undermine the opponent's logic. Show how these distinctions fundamentally change the assessment of the situation. Demonstrate how recognizing these distinctions leads to a different conclusion than your opponent reached.\n\n"),
        ("Circular Anticipation", "RESPOND USING CIRCULAR ANTICIPATION DEBATE STYLE. Preemptively identify and address the most likely counterarguments before your opponent can make them. Introduce opposing points with phrases like 'Some might argue...' or 'One could object that...' followed by your prepared refutation. Structure your response to cover all major potential objections. Demonstrate that you've thoroughly considered the issue from multiple angles. Frame potential counterarguments in ways that make them easier to dismantle. Create the impression that all reasonable objections have already been considered and overcome. Conclude by suggesting that any remaining objections would be similarly flawed.\n\n")
    ]
    rel_template = (
        "You will compare two answers to the *same* question.\n\n"
        "Question:\n[QUESTION START]\n{question}\n[QUESTION END]\n\n"
        "Original answer:\n[BASE START]\n{base_answer}\n[BASE END]\n\n"
        "Modified answer:\n[STEERED START]\n{steered_answer}\n[STEERED END]\n\n"
        "Compared **to the original answer**, which single tone label best describes the *modified* answer?\n"
        "Allowed labels: cautious, empathetic, expert, helpful, casual, concise.\n"
        "Respond with **only** one label."
    )
    return SimpleNamespace(
        name="debate",
        raw_dataset = ds,
        steered_prompts = steered_prompts,
        rel_template = rel_template,
        eval_prompts = list(ds["text"]),
        unique_labels = sorted({t for t, _ in steered_prompts}),
    )

_TASK_LOADERS = {"tone": _load_tones, "debate": _load_debates}
_CURRENT_TASK = None
_DATA_CTX     = None

def ensure_task_data(task: str | None = None):
    global _CURRENT_TASK, _DATA_CTX
    task = task or CFG["TASK"]
    if _CURRENT_TASK == task and _DATA_CTX is not None:
        return _DATA_CTX
    if task not in _TASK_LOADERS:
        raise ValueError(f"Unknown task {task!r}. Choose one of {list(_TASK_LOADERS)}")
    print(f"⇒ Loading steering task “{task}”…")
    _DATA_CTX     = _TASK_LOADERS[task]()
    _CURRENT_TASK = task
    return _DATA_CTX

def build_steering_dataset(ctx: SimpleNamespace) -> Dataset:
    rows = []
    for row in ctx.raw_dataset:
        q_text, q_id = row["text"], row["id"]
        cat = row.get("category", "")
        for lbl, sys_prompt in ctx.steered_prompts:
            rows.append({
                "id": f"{q_id}_{lbl}",
                "original_question": q_text,
                "text": f"{sys_prompt}\n{q_text}",
                "label": lbl,
                "system_message": sys_prompt,
                "category": cat,
            })
    return Dataset.from_pandas(pd.DataFrame(rows))


def get_dataset(dataset_name = "tone"):

    data_ctx          = ensure_task_data(dataset_name)
    dataset           = build_steering_dataset(data_ctx)
    unique_labels     = data_ctx.unique_labels
    RELATIVE_TEMPLATE = data_ctx.rel_template
    eval_prompts      = data_ctx.eval_prompts
    return dataset, eval_prompts, unique_labels

def compute_caa_vectors(
    dataset,
    unique_labels,
    model,
    tokenizer,
    steer_layer: int,
    max_pairs: int | None = None,
) -> np.ndarray:
    q2lab2text = defaultdict(dict)
    for row in dataset:
        q2lab2text[row["original_question"]][row["label"]] = row["text"]

    pos, neg = defaultdict(list), defaultdict(list)
    for q, lab_map in q2lab2text.items():
        labs = set(lab_map)
        for tgt in labs:
            for other in labs - {tgt}:
                pos[tgt].append(lab_map[tgt])
                neg[tgt].append(lab_map[other])

    caa_vecs = []
    for lbl in unique_labels:
        pairs = len(pos[lbl])
        if max_pairs and pairs > max_pairs:
            keep = random.sample(range(pairs), max_pairs)
            pos[lbl] = [pos[lbl][i] for i in keep]
            neg[lbl] = [neg[lbl][i] for i in keep]

        if not pos[lbl]:
            caa_vecs.append(np.zeros(model.config.hidden_size, dtype=np.float32))
            continue

        X_pos = get_hidden_cached(model=model, tokenizer=tokenizer,texts=pos[lbl], layer_idx=steer_layer)
        X_neg = get_hidden_cached(model=model, tokenizer=tokenizer, texts=neg[lbl], layer_idx=steer_layer)
        caa_vecs.append((X_pos - X_neg).mean(0))

    return np.stack(caa_vecs, axis=0)

def _compute_steering_loss(
    logits: torch.Tensor,
    target_idx,
    avoid_idx,
) -> torch.Tensor:
    if not torch.is_tensor(target_idx):
        target_idx = torch.as_tensor(target_idx, device=logits.device)
    else:
        target_idx = target_idx.to(logits.device)
    if not torch.is_tensor(avoid_idx):
        avoid_idx = torch.as_tensor(avoid_idx, device=logits.device)
    else:
        avoid_idx = avoid_idx.to(logits.device)

    B, C = logits.shape

    if avoid_idx.numel() > 0:
        avoid_term = logits[:, avoid_idx].mean(dim=1)
    else:
        avoid_term = torch.zeros(B, device=logits.device)

    if target_idx.numel() > 0:
        target_term = logits[:, target_idx].mean(dim=1)
    else:
        target_term = torch.zeros(B, device=logits.device)

    return avoid_term - target_term

def get_gradient_hook(steer_model,
                      target_labels=None,
                      avoid_labels=None,
                      alpha: float = 1.0,
                      steps: int = 1,
                      step_size_decay: float = 1.0):

    target_labels = torch.as_tensor(target_labels or [], device=steer_model.device)
    avoid_labels  = torch.as_tensor(avoid_labels  or [], device=steer_model.device)

    @torch.inference_mode(False)
    def fwd_hook(module, inp, out):
        h_fp16 = out[0]
        B, S, D = h_fp16.shape

        h_current = h_fp16.reshape(-1, D).float()

        for step in range(steps):
            h_step = h_current.clone()
            h_step.requires_grad_(True)

            logits = steer_model.classifier(h_step)
            logits = logits.view(B, S, -1).mean(dim=1)

            loss_vec = _compute_steering_loss(
                logits,
                target_idx=target_labels,
                avoid_idx=avoid_labels
            )

            if loss_vec.numel() > 0:
                grad = torch.autograd.grad(
                    outputs=loss_vec,
                    inputs=h_step,
                    grad_outputs=torch.ones_like(loss_vec),
                    retain_graph=False,
                    create_graph=False,
                )[0]

                current_alpha = alpha * (step_size_decay ** step)

                grad = grad.view(B * S, D)
                h_current = (h_step - current_alpha * grad).detach()
            else:
                h_current = h_step.detach()

        h_new = h_current.reshape(B, S, D).to(h_fp16.dtype)
        return (h_new,) + out[1:]

    return fwd_hook

def get_caa_hook(caa_vector: torch.Tensor | np.ndarray,
                 alpha: float = 1.0):
    if not torch.is_tensor(caa_vector):
        caa_vector = torch.as_tensor(caa_vector, dtype=torch.float16)

    def fwd_hook(module, inp, out):
        h = out[0]
        return (h + alpha * caa_vector.to(h.device),) + out[1:]

    return fwd_hook

def get_alpha_file(model="llama-3.2-3b", dataset="tone", num_combo=1):
    alpha_path = f"/workspace/codes/nonlinear_steering/{model}/{model}-{str(num_combo)}-{dataset}-activations-classifier-alphas.csv"
    combo_lookup_table = pd.read_csv(alpha_path, converters={"Targets": literal_eval})
    combo_lookup_table["key"] = combo_lookup_table["Targets"].apply(lambda t: tuple(sorted(t)))
    combo_lookup_dict = combo_lookup_table.set_index("key").to_dict(orient="index")
    return combo_lookup_dict


def sample_steered_responses(
    model,
    tokenizer,
    dataset_name: str,
    num_combo: int,
    prompts: list[str],
    *,
    target_tones: list[str],         # NEW: explicit targets for this run
    steer_model_k = None,
    layer_k: int,                    # NEW: explicit
    alpha_grad: float,               # NEW: explicit
    caa_vectors = None,
    layer_caa: int,                  # NEW: explicit
    alpha_caa: float,                # NEW: explicit
    max_new_tokens: int = 32,
    batch_size    : int = 8,
):
    dataset, eval_prompts, unique_labels = get_dataset(dataset_name)

    # Train the classifier once per (layer_k) setting
    clf_prompts  = [row["text"] for row in dataset]
    clf_y  = np.array([unique_labels.index(row["label"]) for row in dataset], dtype=np.int64)
    clf_x = get_hidden_cached(model=model, tokenizer=tokenizer, texts=clf_prompts, layer_idx=layer_k)

    act_clf, acc = get_or_train_layer_clf(
        unique_labels=unique_labels,
        X=clf_x,
        y=clf_y,
        hidden_dim = 128,
        epochs      = 5,
        batch_size  = 32,
    )

    if steer_model_k is None:
        steer_model_k = act_clf

    if caa_vectors is None:
        caa_vectors = compute_caa_vectors(
            unique_labels=unique_labels,
            steer_layer=layer_k,   # CAA based on same layer as classifier unless you have a reason to change
            dataset=dataset,
            model=model,
            tokenizer=tokenizer
        )

    tone2idx = {t: i for i, t in enumerate(unique_labels)}
    tgt_idx = [tone2idx[t] for t in target_tones]

    # Hooks
    grad_hook = get_gradient_hook(
        steer_model_k, target_labels=tgt_idx,
        avoid_labels=[], alpha=alpha_grad
    )
    caa_vec = caa_vectors[tgt_idx].mean(axis=0)
    caa_hook = get_caa_hook(caa_vec, alpha=alpha_caa)

    # Generate
    unsteered = batch_generate(
        model, tokenizer, prompts,
        layer_idx=layer_caa,   # consistent with your previous call pattern
        hook_fn=None,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )
    ksteer = batch_generate(
        model, tokenizer, prompts,
        layer_idx=layer_k,
        hook_fn=grad_hook,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )
    caa_out = batch_generate(
        model, tokenizer, prompts,
        layer_idx=layer_caa,
        hook_fn=caa_hook,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )

    def _strip(gen: str, prompt: str) -> str:
        return gen[len(prompt):].lstrip() if gen.startswith(prompt) else gen

    rows = []
    for p, base, ktxt, ctxt in zip(prompts, unsteered, ksteer, caa_out):
        rows.append({
            "prompt": p,
            "unsteered": _strip(base, p),
            "k_steering": _strip(ktxt, p),
            "caa": _strip(ctxt, p),
            "layer_k": layer_k,
            "layer_caa": layer_caa,
            "alpha_grad": alpha_grad,
            "alpha_caa": alpha_caa,
            "targets": ", ".join(sorted(target_tones)),
        })

    df = pd.DataFrame(rows)

    for r in rows[:min(3, len(rows))]:  # brief console preview
        print("\n" + "=" * 90)
        print(f"PROMPT:\n{r['prompt']}\n")
        print("- Unsteered -------------------------------------------------\n"
              + r["unsteered"] + "\n")
        print(f"- K-steering (layer {layer_k}, α_grad = {alpha_grad:.3g})\n"
              + r["k_steering"] + "\n")
        print(f"- CAA        (layer {layer_caa}, α_caa  = {alpha_caa:.3g})\n"
              + r["caa"] + "\n")
    return df


def sample_steered_responses_old(
    model,
    tokenizer,
    dataset_name: str,
    num_combo: int,
    prompts: list[str],
    *,
    steer_model_k = None,
    layer_k       = None,
    alpha_grad    = None,
    caa_vectors   = None,
    layer_caa     = None,
    alpha_caa     = None,
    max_new_tokens: int = 32,
    batch_size    : int = 8,
    save_as: str | None = None,
):
    
    dataset, eval_prompts, unique_labels = get_dataset(dataset_name)
    combo_lookup_dict = get_alpha_file(model=model_name, dataset=dataset_name, num_combo=num_combo)

    def _lookup_from_csv(is_caa=False):
        # random.seed(42)
        combo_key = random.choice(list(combo_lookup_dict.keys()))
        entry = combo_lookup_dict[combo_key]
        target_tones_from_key = list(combo_key)  # Convert tuple back to list
        if is_caa:
            return entry["alpha_caa"], int(entry["best_caa_layer"]), target_tones_from_key
        else:
            return entry["alpha_grad"], int(entry["best_k_layer"]), target_tones_from_key

    if alpha_grad is None or layer_k is None:
        alpha_grad, layer_k, tgt_list = _lookup_from_csv(is_caa=False)
    if alpha_caa is None or layer_caa is None:
        alpha_caa, layer_caa, _ = _lookup_from_csv(is_caa=True)

    #Let's train the classifier
    clf_prompts  = [row["text"] for row in dataset]
    clf_y  = np.array([unique_labels.index(row["label"]) for row in dataset], dtype=np.int64)
    clf_x = get_hidden_cached(model=model, tokenizer=tokenizer,texts=clf_prompts, layer_idx=layer_k)

    act_clf, acc = get_or_train_layer_clf(
        unique_labels=unique_labels,
        X=clf_x,
        y=clf_y,
        hidden_dim = 128,
        epochs      = 5,
        batch_size  = 32,
    )

    print("Here is a randomly selected list of targets", tgt_list)

    if steer_model_k is None:
        steer_model_k = act_clf  # Ensure act_clf_eval is defined in your scope

    if caa_vectors is None:
        caa_vectors = compute_caa_vectors(unique_labels=unique_labels, steer_layer=layer_k, dataset=dataset, model=model, tokenizer=tokenizer)

    tone2idx = {t: i for i, t in enumerate(unique_labels)}
    tgt_idx = [tone2idx[t] for t in tgt_list]

    grad_hook = get_gradient_hook(
        steer_model_k, target_labels=tgt_idx,
        avoid_labels=[], alpha=alpha_grad
    )
    caa_vec = caa_vectors[tgt_idx].mean(axis=0)
    caa_hook = get_caa_hook(caa_vec, alpha=alpha_caa)

    unsteered = batch_generate(
        model, tokenizer, prompts,
        layer_idx=layer_caa,
        hook_fn=None,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )
    ksteer = batch_generate(
        model, tokenizer, prompts,
        layer_idx=layer_k,
        hook_fn=grad_hook,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )
    caa_out = batch_generate(
        model, tokenizer, prompts,
        layer_idx=layer_caa,
        hook_fn=caa_hook,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )

    def _strip(gen: str, prompt: str) -> str:
        return gen[len(prompt):].lstrip() if gen.startswith(prompt) else gen

    rows = []
    for p, base, ktxt, ctxt in zip(prompts, unsteered, ksteer, caa_out):
        rows.append({
            "prompt": p,
            "unsteered": _strip(base, p),
            "k_steering": _strip(ktxt, p),
            "caa": _strip(ctxt, p),
            "layer_k": layer_k,
            "layer_caa": layer_caa,
            "alpha_grad": alpha_grad,
            "alpha_caa": alpha_caa,
            "targets": ", ".join(tgt_list),
        })

    df = pd.DataFrame(rows)

    for r in rows:
        print("\n" + "=" * 90)
        print(f"PROMPT:\n{r['prompt']}\n")
        print("- Unsteered -------------------------------------------------\n"
              + r["unsteered"] + "\n")
        print(f"- K‑steering (layer {layer_k}, α_grad = {alpha_grad:.3g})\n"
              + r["k_steering"] + "\n")
        print(f"- CAA        (layer {layer_caa}, α_caa  = {alpha_caa:.3g})\n"
              + r["caa"] + "\n")
    return df, tgt_list

def main():
    args = parse_args()

    model, tokenizer = get_model(args.model_name)

    with open(args.prompts_path, "r") as f_in:
        all_prompts = json.load(f_in)

    combo_lookup_dict = get_alpha_file(
        model=args.model_name, dataset=args.dataset_name, num_combo=args.num_combo
    )

    out_root = Path(f"{args.model_name}/{args.num_combo}")
    out_root.mkdir(parents=True, exist_ok=True)

    for combo_key, entry in combo_lookup_dict.items():
        tgt_list  = list(combo_key)  # combo_key is a sorted tuple
        tgt_str   = "-".join(tgt_list)

        alpha_grad = float(entry["alpha_grad"])
        layer_k    = int(entry["best_k_layer"])
        alpha_caa  = float(entry["alpha_caa"])
        layer_caa  = int(entry["best_caa_layer"])

        print(
            f"\n=== Running combo: {tgt_str} | "
            f"K(layer={layer_k}, α={alpha_grad}) ; "
            f"CAA(layer={layer_caa}, α={alpha_caa}) ==="
        )

        results_df = sample_steered_responses(
            model=model,
            tokenizer=tokenizer,
            prompts=all_prompts,
            dataset_name=args.dataset_name,
            num_combo=args.num_combo,
            target_tones=tgt_list,
            layer_k=layer_k,
            alpha_grad=alpha_grad,
            layer_caa=layer_caa,
            alpha_caa=alpha_caa,
            max_new_tokens=32,
            batch_size=8,
        )

        out_path = out_root / f"{tgt_str}.csv"
        results_df.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
