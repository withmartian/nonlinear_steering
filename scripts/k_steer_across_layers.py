# File: run_layer_steering.py
import torch
import functools
import json
import os
import multiprocessing as mp
import random
import numpy as np
from datetime import datetime
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from torch import Tensor
from typing import List
from tqdm import tqdm
import torch
import functools
import einops
import requests
import pandas as pd
import io
import textwrap
import gc

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch import Tensor
from typing import List, Callable
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
from transformers import AutoTokenizer
from jaxtyping import Float, Int
from colorama import Fore
from sklearn.model_selection import train_test_split
import random
import numpy as np
import sys
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append("/home/ubuntu/nonlinear_steering")
from judges.debates_judge import DebateJudge

judge = DebateJudge(api_key="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")


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class MultiToneActivationSteering:
    def __init__(self, input_dim, num_classes=6, hidden_dim=256, lr=1e-3, device='cuda'):
        """
        Parameters:
          - input_dim: dimensionality of your raw activations.
          - num_classes: number of tone classes to classify.
          - hidden_dim: size of the hidden layer in the MLP.
          - lr: learning rate for training the classifier.
          - device: 'cuda' or 'cpu'.
        """
        self.device = device
        self.num_classes = num_classes
        
        # Define a multi-class MLP classifier
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)  # outputs logits for each tone class
        ).to(device)
        
        self.optimizer = optim.Adam(self.classifier.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Store class names for reference
        self.class_names = None
        

    def fit(self, activations_dict, epochs=30, batch_size=32, class_weights=None):
        """
        Train the multiclass classifier on raw activations.

        Parameters:
          - activations_dict: Dictionary mapping class names to NumPy arrays of activations.
          - epochs: number of training epochs.
          - batch_size: training batch size.
          - class_weights: Optional tensor of weights for each class to handle imbalance.
        """
        self.class_names = list(activations_dict.keys())
        assert len(self.class_names) == self.num_classes, f"Expected {self.num_classes} classes, got {len(self.class_names)}"
        
        # Prepare training data
        X_list = []
        y_list = []
        
        for i, class_name in enumerate(self.class_names):
            X_class = torch.tensor(activations_dict[class_name], dtype=torch.float32, device=self.device)
            y_class = torch.full((X_class.size(0),), i, dtype=torch.long, device=self.device)
            X_list.append(X_class)
            y_list.append(y_class)
        
        X = torch.cat(X_list, dim=0)
        y = torch.cat(y_list, dim=0)
        
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Set class weights for loss function if provided
        if class_weights is not None:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        
        # Train the classifier
        for epoch in range(epochs):
            self.classifier.train()
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for batch_X, batch_y in loader:
                self.optimizer.zero_grad()
                logits = self.classifier(batch_X)  # shape: (batch, num_classes)
                loss = self.loss_fn(logits, batch_y)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(logits.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            accuracy = 100 * correct / total
            print(f"Epoch {epoch}: loss = {epoch_loss/len(loader):.4f}, accuracy = {accuracy:.2f}%")
            
        # Evaluate on each class
        self.classifier.eval()
        class_accuracies = {}
        
        with torch.no_grad():
            for i, class_name in enumerate(self.class_names):
                X_class = torch.tensor(activations_dict[class_name], dtype=torch.float32, device=self.device)
                y_class = torch.full((X_class.size(0),), i, dtype=torch.long, device=self.device)
                
                logits = self.classifier(X_class)
                _, predicted = torch.max(logits.data, 1)
                accuracy = 100 * (predicted == y_class).sum().item() / y_class.size(0)
                class_accuracies[class_name] = accuracy
                print(f"Accuracy for {class_name}: {accuracy:.2f}%")

    def steer_toward_tones(self, activation, target_tones=None, avoid_tones=None, alpha=0.1, steps=1, decay_rate=0.95):
        """
        Steer the given activation toward target tones and away from tones to avoid.
        
        Parameters:
        - activation: a NumPy array of shape (D,) or (N, D).
        - target_tones: list of tone names to steer toward.
        - avoid_tones: list of tone names to steer away from.
        - alpha: scaling factor for the update.
        - steps: number of gradient descent steps to take.
        
        Returns:
        - The modified activation as a NumPy array.
        """
        
        # Validate input tone names
        if target_tones:
            for tone in target_tones:
                assert tone in self.class_names, f"Unknown target tone: {tone}"
        
        if avoid_tones:
            for tone in avoid_tones:
                assert tone in self.class_names, f"Unknown avoid tone: {tone}"
        
        # Get indices for target and avoid tones
        target_indices = [self.class_names.index(tone) for tone in target_tones] if target_tones else []
        avoid_indices = [self.class_names.index(tone) for tone in avoid_tones] if avoid_tones else []
        
        # If a single activation vector is passed, add a batch dimension
        single_input = False
        if activation.ndim == 1:
            activation = activation[None, :]
            single_input = True
        
        # Convert activation to a torch tensor
        X = torch.tensor(activation, dtype=torch.float32, device=self.device)
        
        current_alpha = alpha
        
        # Perform multiple steps of gradient descent
        for step in range(steps):
            if step > 0:
                current_alpha *= decay_rate
            # Need to create a new tensor that requires gradients for each step
            X = X.detach().requires_grad_(True)
            
            # Forward pass: compute classifier output (logits)
            self.classifier.eval()
            logits = self.classifier(X)  # shape: [N, num_classes]
            
            # Create custom loss function to maximize target tone scores and minimize avoid tone scores
            loss = 0
            
            if target_indices:
                target_logits = logits[:, target_indices]
                # Negative because we want to maximize these logits (gradient descent will minimize)
                loss = loss - target_logits.mean()
            
            if avoid_indices:
                avoid_logits = logits[:, avoid_indices]
                # Positive because we want to minimize these logits
                loss = loss + avoid_logits.mean()
            
            # If no target or avoid tones provided, do nothing
            if not target_indices and not avoid_indices:
                if single_input:
                    return activation[0]
                else:
                    return activation
            
            # Compute gradients
            loss.backward()
            
            # Get the gradient with respect to the input activation
            grad = X.grad.data
            
            # Update the activation by moving in the loss gradient direction
            # Negative gradient because we're trying to minimize the loss

            X = X - current_alpha * grad
        
        # Convert back to NumPy
        X_new_np = X.detach().cpu().numpy()
        if single_input:
            return X_new_np[0]
        else:
            return X_new_np
    
    def remove_tone_projection(self, activation, target_tones=None, avoid_tones=None):
        """
        Remove the projection of the activation onto the tone steering vector.
        
        Parameters:
        - activation: a NumPy array of shape (D,) or (N, D).
        - target_tones: list of tone names to steer toward.
        - avoid_tones: list of tone names to steer away from.
        
        Returns:
        - The modified activation as a NumPy array with the projection removed.
        """
        assert self.class_names is not None, "Model must be trained before steering"
        
        # Validate input tone names
        if target_tones:
            for tone in target_tones:
                assert tone in self.class_names, f"Unknown target tone: {tone}"
        
        if avoid_tones:
            for tone in avoid_tones:
                assert tone in self.class_names, f"Unknown avoid tone: {tone}"
        
        # Get indices for target and avoid tones
        target_indices = [self.class_names.index(tone) for tone in target_tones] if target_tones else []
        avoid_indices = [self.class_names.index(tone) for tone in avoid_tones] if avoid_tones else []
        
        # If a single activation vector is passed, add a batch dimension
        single_input = False
        if activation.ndim == 1:
            activation = activation[None, :]
            single_input = True
        
        # Convert activation to a torch tensor that requires gradients
        X = torch.tensor(activation, dtype=torch.float32, device=self.device, requires_grad=True)
        
        # Forward pass: compute classifier output (logits)
        self.classifier.eval()
        logits = self.classifier(X)  # shape: [N, num_classes]
        
        # Create custom loss function to maximize target tone scores and minimize avoid tone scores
        loss = 0
        
        if target_indices:
            target_logits = logits[:, target_indices]
            # Negative because we want to maximize these logits (gradient descent will minimize)
            loss = loss - target_logits.mean()
        
        if avoid_indices:
            avoid_logits = logits[:, avoid_indices]
            # Positive because we want to minimize these logits
            loss = loss + avoid_logits.mean()
        
        # If no target or avoid tones provided, do nothing
        if not target_indices and not avoid_indices:
            if single_input:
                return activation[0]
            else:
                return activation
        
        # Compute gradients
        loss.backward()
        
        # Get the gradient with respect to the input activation
        grad = X.grad.data  # shape: [N, D]
        
        # Compute the squared norm of the gradient for each sample (avoid division by zero)
        norm_sq = (grad ** 2).sum(dim=1, keepdim=True) + 1e-8  # shape: [N, 1]
        
        # Compute the dot product between each activation and its corresponding gradient
        dot_prod = (X * grad).sum(dim=1, keepdim=True)  # shape: [N, 1]
        
        # Compute the projection of X onto the gradient
        proj = (dot_prod / norm_sq) * grad  # shape: [N, D]
        
        # Remove the projection from X (multiply by 2 to strengthen the effect)
        X_new = X - 2 * proj
        
        # Convert back to NumPy
        X_new_np = X_new.detach().cpu().numpy()
        if single_input:
            return X_new_np[0]
        else:
            return X_new_np
        


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

QWEN_CHAT_TEMPLATE = """<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""



def get_k_ary_instruction(style="Empirical Grounding", repo_name="Narmeen07/debate_style_agnostic_questions"):
    # Define highly distinctive debate style instructions that will produce clearly different responses
    debate_style_instructions = {
        "Reductio ad Absurdum": "RESPOND USING REDUCTIO AD ABSURDUM DEBATE STYLE. Extend the opposing viewpoint to its logical extreme to reveal its flaws. Demonstrate how the opposing position leads to absurd or contradictory conclusions when taken to its logical end. Structure your response to systematically expose the inconsistencies or impracticalities. Use conditional statements like 'If we follow this logic, then...' to illustrate the path to absurdity. Maintain a logical flow that shows how the initial premise inevitably leads to an unacceptable or ridiculous outcome.\n\n",
        
        "Appeal to Precedent": "RESPOND USING APPEAL TO PRECEDENT DEBATE STYLE. Ground your arguments in historical examples, established case law, or widely accepted previous decisions. Reference specific historical situations, court rulings, scientific discoveries, or cultural precedents that support your position. Draw explicit connections between these precedents and the current issue. Emphasize how respecting established precedents provides consistency and reliability. Point out the risks or inconsistencies of ignoring relevant precedents. Structure your response to show how past experiences inform present considerations.\n\n",
        
        "Straw Man Reframing": "RESPOND USING STRAW MAN REFRAMING DEBATE STYLE. Recharacterize the opposing argument in simplified or exaggerated terms that are easier to refute. Present this reframed version as if it represents the original position. Systematically dismantle this reframed argument while implying you've addressed the original point. Use phrases like 'Essentially, what you're saying is...' or 'This position boils down to...' before presenting the simplified version. Follow with a thorough refutation of this reframed position. Conclude by suggesting that your refutation applies to the original argument as well.\n\n",
        
        "Burden of Proof Shift": "RESPOND USING BURDEN OF PROOF SHIFT DEBATE STYLE. Redirect the responsibility for providing evidence to your opponent rather than proving your own claim. Challenge your opponent to disprove your assertion rather than supporting it yourself. Use phrases like 'There's no evidence that disproves...' or 'Can you definitively prove that isn't the case?' Position your claims as the default assumption that should be accepted until disproven. Question the sufficiency or quality of opposing evidence while demanding an impossibly high standard of proof. Emphasize that the lack of counter-evidence strengthens your position.\n\n",
        
        "Analogy Construction": "RESPOND USING ANALOGY CONSTRUCTION DEBATE STYLE. Develop a vivid, relatable comparison between the complex issue at hand and something more familiar or intuitive. Build your argument around this carefully constructed parallel situation. Highlight specific points of similarity that support your position while addressing potential dissimilarities. Use phrases like 'This situation is similar to...' or 'To understand this concept, consider...' Ensure your analogy simplifies the complex issue without distorting its essential nature. Use the familiar scenario to guide your audience to your desired conclusion about the original issue.\n\n",
        
        "Concession and Pivot": "RESPOND USING CONCESSION AND PIVOT DEBATE STYLE. Begin by acknowledging a minor point or critique from the opposing side to establish fairness and reasonableness. Use phrases like 'While it's true that...' or 'I can concede that...' followed by 'However,' 'Nevertheless,' or 'That said,' to redirect to your stronger arguments. Ensure the conceded point is peripheral rather than central to your main argument. After the concession, pivot decisively to your strongest points with increased emphasis. Frame your pivot as providing necessary context or a more complete perspective. Use the concession to demonstrate your objectivity before delivering your more powerful counterarguments.\n\n",
        
        "Empirical Grounding": "RESPOND USING EMPIRICAL GROUNDING DEBATE STYLE. Base your arguments primarily on verifiable data, research studies, statistics, and observable outcomes rather than theory or rhetoric. Cite specific figures, percentages, study results, or historical outcomes that support your position. Present evidence in a methodical manner, explaining how each piece of data relates to your argument. Address the reliability and relevance of your sources and methods. Compare empirical results across different contexts or time periods to strengthen your case. Anticipate and address potential methodological criticisms of the evidence you present.\n\n",
        
        "Moral Framing": "RESPOND USING MORAL FRAMING DEBATE STYLE. Position the issue within a framework of ethical principles, values, and moral imperatives rather than pragmatic concerns. Identify the core moral values at stake such as justice, liberty, equality, compassion, or responsibility. Use language that evokes ethical considerations, such as 'obligation,' 'right,' 'wrong,' 'just,' or 'fair.' Appeal to widely held moral intuitions or principles. Present opposing views as morally questionable or inconsistent with important shared values. Elevate the discussion from practical matters to questions of what ought to be done. Emphasize moral consequences over practical outcomes.\n\n",
        
        "Refutation by Distinction": "RESPOND USING REFUTATION BY DISTINCTION DEBATE STYLE. Identify crucial differences that invalidate comparisons or principles your opponent has applied. Carefully delineate categories, contexts, or circumstances that demonstrate why a general rule or example doesn't apply in this specific case. Use phrases like 'While that may be true in some contexts...' or 'We must distinguish between...' Emphasize the precision of definitions and classifications. Highlight subtle but significant differences that undermine the opponent's logic. Show how these distinctions fundamentally change the assessment of the situation. Demonstrate how recognizing these distinctions leads to a different conclusion than your opponent reached.\n\n",
        
        "Circular Anticipation": "RESPOND USING CIRCULAR ANTICIPATION DEBATE STYLE. Preemptively identify and address the most likely counterarguments before your opponent can make them. Introduce opposing points with phrases like 'Some might argue...' or 'One could object that...' followed by your prepared refutation. Structure your response to cover all major potential objections. Demonstrate that you've thoroughly considered the issue from multiple angles. Frame potential counterarguments in ways that make them easier to dismantle. Create the impression that all reasonable objections have already been considered and overcome. Conclude by suggesting that any remaining objections would be similarly flawed.\n\n"
    }
    dataset = load_dataset(repo_name, split="train")
    def format_prompt(example):
        return {"formatted_text": f"{debate_style_instructions[style]}{example['text']}"}
    formatted_dataset = dataset.map(format_prompt)
    train_data, test_data = train_test_split(formatted_dataset["formatted_text"], test_size=0.2, random_state=42)
    return train_data, test_data



style_label_map = {
    "reductio": "Reductio ad Absurdum",
    "precedent": "Appeal to Precedent",
    "strawman": "Straw Man Reframing",
    "burden": "Burden of Proof Shift",
    "analogy": "Analogy Construction",
    "concession": "Concession and Pivot",
    "empirical": "Empirical Grounding",
    "moral": "Moral Framing",
    "refutation": "Refutation by Distinction",
    "circular": "Circular Anticipation"
}

MODEL_PATH = 'meta-llama/Llama-3.2-3B-Instruct'
N_INST_TEST = 10
TOTAL_SAMPLES = 600
act_names = ['resid_pre', 'resid_mid', 'resid_post']

def run_layer(layer: int, device: int):

    
    model = HookedTransformer.from_pretrained_no_processing(
        MODEL_PATH,
        device=f"cuda:{device}",
        dtype=torch.float16,
        default_padding_side='left',
    )
    model = model.to(f"cuda:{device}")
    model.tokenizer.padding_side = 'left'
    model.tokenizer.pad_token = model.tokenizer.eos_token


    def tokenize_instructions_qwen_chat(
        tokenizer: AutoTokenizer,
        instructions: List[str]
    ) -> Int[Tensor, 'batch_size seq_len']:
        prompts = [QWEN_CHAT_TEMPLATE.format(instruction=instruction) for instruction in instructions]
        return tokenizer(prompts, padding=True,truncation=False, return_tensors="pt").input_ids

    def tokenize_fn(instructions):
        return tokenize_instructions_qwen_chat(model.tokenizer, instructions)
    
    def _generate_with_hooks(model: HookedTransformer, toks: Tensor, max_tokens_generated: int = 64, fwd_hooks=[]) -> List[str]:
        all_toks = torch.zeros((toks.shape[0], toks.shape[1] + max_tokens_generated), dtype=torch.long, device=model.cfg.device)
        all_toks[:, :toks.shape[1]] = toks
        for i in range(max_tokens_generated):
            with model.hooks(fwd_hooks=fwd_hooks):
                logits = model(all_toks[:, :-max_tokens_generated + i])
                next_tokens = logits[:, -1, :].argmax(dim=-1)
                all_toks[:, -max_tokens_generated + i] = next_tokens
        return model.tokenizer.batch_decode(all_toks[:, toks.shape[1]:], skip_special_tokens=True)

    def get_generations(model: HookedTransformer, instructions: List[str], tokenize_fn, fwd_hooks=[], max_tokens_generated: int = 64, batch_size: int = 1) -> List[str]:
        generations = []
        for i in tqdm(range(0, len(instructions), batch_size)):
            toks = tokenize_fn(instructions=instructions[i:i + batch_size])
            generation = _generate_with_hooks(model, toks, max_tokens_generated=max_tokens_generated, fwd_hooks=fwd_hooks)
            generations.extend(generation)
        return generations
    
    def batch_process_activations(instructions, model, tokenize_fn, layer=14, pos=-1, batch_size=16):
        acts_list = []
        for start_idx in range(0, len(instructions), batch_size):
            end_idx = min(start_idx + batch_size, len(instructions))
            toks = tokenize_fn(instructions[start_idx:end_idx]).to(model.cfg.device)

            logits, cache = model.run_with_cache(toks, names_filter=lambda hook_name: 'resid' in hook_name)
            batch_acts = cache['resid_pre', layer][:, pos, :].detach().cpu().numpy()
            acts_list.append(batch_acts)
            del cache, logits
            torch.cuda.empty_cache()
        return np.concatenate(acts_list, axis=0)
    # Load data + activations
    datasets = {s: get_k_ary_instruction(style_label_map[s])[0][:TOTAL_SAMPLES] for s in style_label_map}
    layer_activations = {
        name: batch_process_activations(instructions=data, model=model, tokenize_fn=tokenize_fn, layer=layer)
        for name, data in datasets.items()
    }

    # Train steering classifier
    debate_steer = MultiToneActivationSteering(input_dim=layer_activations['reductio'].shape[1], num_classes=10, device=model.cfg.device)
    debate_steer.fit(layer_activations, epochs=30, batch_size=32)
    torch.cuda.empty_cache()


    alphas_to_try = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,2.0,2.5,3.0,3.5,4.0, 4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5,9.0,9.5,10.0,11.0,12.0, 13.0,14.0, 15.0,20.0,25.0,30.0]


    layer_log = {
        "meta": {
            "layer": layer,
            "alphas": alphas_to_try,
            "timestamp": datetime.now().isoformat()
        },
        "runs": []
    }

    for alpha in alphas_to_try:
    # Steering hook
        def actadd_hook(activation, hook, target_styles=['precedent'], avoid_styles=['reductio'], alpha=alpha, steps=1):
            shape = activation.shape
            act_2d = activation.view(-1, shape[-1]).detach().cpu().numpy()
            steered_2d = debate_steer.steer_toward_tones(act_2d, target_tones=target_styles, avoid_tones=avoid_styles, alpha=alpha, steps=steps)
            return torch.from_numpy(steered_2d).to(activation.device).view(*shape)

        # Run and compare generations
        reductio_inst_train, reductio_inst_test = get_k_ary_instruction(style_label_map["reductio"])
        baseline = get_generations(model, reductio_inst_test[:N_INST_TEST], tokenize_fn, [])
        intervention_layers = list(range(model.cfg.n_layers)) 
        fwd_hooks = [
            (utils.get_act_name(act_name, l), functools.partial(actadd_hook))
            for l in intervention_layers
            for act_name in ['resid_pre', 'resid_mid', 'resid_post']
        ]

        steered = get_generations(model, reductio_inst_test[:N_INST_TEST], tokenize_fn, fwd_hooks)
        result = judge.evaluate_batch(baseline, steered, style_label_map['precedent'], style_label_map['reductio'])
        # Print results
        print(result)

        layer_log["runs"].append({
            "alpha": alpha,
            "success_rate": result.get("success_rate", None),
            "average_strength": result.get("average_strength", None),
            "details": result
        })

        if result.get("success_rate", 0) == 0 and result.get("average_strength", 0) == 0:
            print(f"Early stopping at alpha={alpha}: both success and strength are 0.")
            break

    os.makedirs("results", exist_ok=True)
    with open(f"results/layer_{layer}_results.json", "w") as f:
        json.dump(layer_log, f, indent=2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=7, help="Layer number to run")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to use")
    args = parser.parse_args()

    run_layer(layer=args.layer, device=args.gpu)

