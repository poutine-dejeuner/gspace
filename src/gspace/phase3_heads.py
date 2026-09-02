"""
Phase 3: Finding Addition Heads via QK Twist Analysis.

Core method from the paper:
1. Train probes p_a (59 vectors) for operand A and q_b (59 vectors) for operand B.
2. Multiply through W_Q and W_K of each attention head.
3. Compute cosine similarity between all (a,b) pairs in QK space.
4. For an addition head: max similarity when b = (target_sum - a) mod 59,
   i.e., the head detects specific sums.
5. Multiple heads tile the residue space for resolution.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA


@dataclass
class Phase3Config:
    model_id: str = "Qwen/Qwen2.5-0.5B"
    modulus: int = 59
    op: str = "+"
    n_problems: int = 1000  # for training operand probes
    best_layer: int = 4   # from Phase 1 (will override after loading results)
    output_dir: str = "results/phase3"
    seed: int = 42
    dtype: str = "bfloat16"
    batch_size: int = 2


def train_operand_probes(
    activations: np.ndarray,
    operand_labels: np.ndarray,  # 0..58
    C: float = 1.0,
    max_iter: int = 500,
) -> LogisticRegression:
    """Train a 59-way logistic probe for an operand."""
    n = len(activations)
    if n > 20000:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, 20000, replace=False)
        activations = activations[idx]
        operand_labels = operand_labels[idx]

    probe = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=max_iter,
        C=C,
    )
    probe.fit(activations, operand_labels)
    return probe


def get_head_qk_weights(model, layer: int, head: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract W_Q and W_K for a specific attention head.

    Returns (W_Q_head, W_K_head) each [d_model, d_head]
    """
    attn = model.model.layers[layer].self_attn
    d_model = model.config.hidden_size
    n_heads = model.config.num_attention_heads
    d_head = d_model // n_heads

    # Get full weight matrices
    W_Q_full = attn.q_proj.weight.data  # [d_model, d_model]
    W_K_full = attn.k_proj.weight.data  # [d_model, d_model]

    # Extract head slice
    start = head * d_head
    end = start + d_head
    W_Q_head = W_Q_full[start:end, :]  # [d_head, d_model]
    W_K_head = W_K_full[start:end, :]

    return W_Q_head.T, W_K_head.T  # [d_model, d_head]


def compute_qk_cosine(
    probe_a_weights: np.ndarray,  # [modulus, d_model]
    probe_b_weights: np.ndarray,  # [modulus, d_model]
    W_Q: torch.Tensor,            # [d_model, d_head]
    W_K: torch.Tensor,            # [d_model, d_head]
    device: torch.device,
) -> np.ndarray:
    """Compute cosine similarity between (p_a @ W_Q) and (q_b @ W_K).

    Returns matrix [modulus, modulus] where entry (i,j) = cos_sim(A=i, B=j)
    """
    modulus = probe_a_weights.shape[0]
    p_a = torch.tensor(probe_a_weights, dtype=torch.float32, device=device)  # [M, d_model]
    p_b = torch.tensor(probe_b_weights, dtype=torch.float32, device=device)

    W_Q = W_Q.to(dtype=torch.float32, device=device)
    W_K = W_K.to(dtype=torch.float32, device=device)

    # Project through QK
    q_space = p_a @ W_Q  # [M, d_head]
    k_space = p_b @ W_K  # [M, d_head]

    # Cosine similarity
    q_norm = torch.nn.functional.normalize(q_space, dim=1)
    k_norm = torch.nn.functional.normalize(k_space, dim=1)
    cos_sim = (q_norm @ k_norm.T).cpu().numpy()  # [M, M]

    return cos_sim


def find_addition_offset(cos_sim: np.ndarray, modulus: int = 59) -> dict:
    """For an addition head, find the offset where max alignment occurs.

    For addition: the head should align A=i with B=j when (i+j) mod M = constant.
    This means max on anti-diagonals: i + j ≡ target_sum (mod M).

    Returns best offset (sum detected) and strength.
    """
    best_sum = 0
    best_val = -1
    sum_scores = {}

    for s in range(modulus):
        # Collect cosine values for all (i, j) where (i+j) % M == s
        vals = []
        for i in range(modulus):
            j = (s - i) % modulus
            vals.append(cos_sim[i, j])
        avg = np.mean(vals)
        sum_scores[s] = float(avg)
        if avg > best_val:
            best_val = avg
            best_sum = s

    # Also check: is the diagonal (i=j) strong? That means identity, not addition
    diagonal_val = float(np.mean([cos_sim[i, i] for i in range(modulus)]))
    off_diagonal_val = best_val

    return {
        "best_sum": best_sum,
        "best_sum_cosine": best_val,
        "diagonal_cosine": diagonal_val,
        "is_addition_head": best_val > diagonal_val + 0.05 and best_val > 0.3,
        "sum_scores": sum_scores,
    }


def analyze_all_heads(
    model,
    probe_a: LogisticRegression,
    probe_b: LogisticRegression,
    modulus: int,
    layers: list[int],
    device: torch.device,
) -> dict:
    """Analyze QK twist for all heads in specified layers."""
    n_heads = model.config.num_attention_heads
    results = {}

    probe_a_w = probe_a.coef_  # [59, d_model]
    probe_b_w = probe_b.coef_

    for layer in tqdm(layers, desc="Analyzing heads"):
        results[layer] = {}
        for head in range(n_heads):
            W_Q, W_K = get_head_qk_weights(model, layer, head)
            cos_sim = compute_qk_cosine(probe_a_w, probe_b_w, W_Q, W_K, device)
            offset = find_addition_offset(cos_sim, modulus)

            results[layer][head] = {
                "cosine_matrix": cos_sim.tolist() if layer == layers[0] and head < 3
                    else None,  # Only save full matrix for first few heads (huge)
                "best_sum": offset["best_sum"],
                "best_sum_cosine": offset["best_sum_cosine"],
                "diagonal_cosine": offset["diagonal_cosine"],
                "is_addition_head": offset["is_addition_head"],
            }

    return results


def compute_head_output_geometry(
    model,
    tokenizer,
    problems: list[dict],
    device: torch.device,
    head_list: list[tuple[int, int]],  # [(layer, head), ...]
    modulus: int = 59,
) -> dict:
    """Project each head's output into PCA space of residue probes.

    For each head, cache output at the operator position and compute per-residue means.
    """
    # Register hooks
    head_outputs = {}

    def make_hook(layer, head):
        def hook(module, input, output):
            # output is the attention output tensor [batch, seq_len, d_model]
            # We need just this head's contribution
            attn_output = output[0]  # [batch, seq_len, d_model]
            head_outputs[(layer, head)] = attn_output.detach().cpu()
        return hook

    hooks = []
    for layer, head in head_list:
        # Hook onto the attention output projection for just this head
        # We need to intercept the head output before the O projection
        # Actually, we hook the full attention output and decompose later
        hook = model.model.layers[layer].self_attn.o_proj.register_forward_hook(
            make_hook(layer, head)
        )
        hooks.append(hook)
        head_outputs[(layer, head)] = None  # placeholder

    # For now, use a simpler approach: just hook layer output and attribute to heads
    # Full per-head decomposition requires modifying attention computation.
    # We'll capture per-layer output and note the limitation.
    layer_outputs = {}
    def make_layer_hook(l):
        def hook(module, input, output):
            layer_outputs[l] = output[0].detach().cpu()
        return hook

    for l in sorted(set(h[0] for h in head_list)):
        hook = model.model.layers[l].register_forward_hook(make_layer_hook(l))
        hooks.append(hook)

    all_acts = {l: [] for l in sorted(set(h[0] for h in head_list))}
    all_labels = []

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(problems), 4), desc="Head output geometry"):
            batch = problems[i:i+4]
            prompts = [p["prompt"] for p in batch]
            labels = [p["result"] for p in batch]

            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
            _ = model(**inputs)

            for l in layer_outputs:
                # last token from layer output
                last_pos = inputs["attention_mask"].sum(dim=1) - 1
                acts = layer_outputs[l][range(len(last_pos)), last_pos].numpy()
                all_acts[l].append(acts)
            all_labels.extend(labels)

    for hook in hooks:
        hook.remove()

    # Per-residue means per layer
    geometry = {}
    for l, act_list in all_acts.items():
        acts = np.concatenate(act_list, axis=0)
        labels = np.array(all_labels)

        means = {}
        for r in range(modulus):
            mask = labels == r
            if mask.sum() > 0:
                means[r] = acts[mask].mean(axis=0)

        residues = sorted(means.keys())
        mean_vecs = np.stack([means[r] for r in residues])
        pca = PCA(n_components=min(20, mean_vecs.shape[0]-1))
        proj = pca.fit_transform(mean_vecs)

        geometry[l] = {
            "pca_projection": proj.tolist(),
            "pca_variance": pca.explained_variance_ratio_.tolist(),
        }

    return geometry


def main():
    config = Phase3Config()
    os.makedirs(config.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load Phase 1 results to get best_layer
    p1_path = "results/phase1/phase1_results.json"
    if os.path.exists(p1_path):
        with open(p1_path) as f:
            p1_results = json.load(f)
        config.best_layer = p1_results["best_layer"]
        config.model_id = p1_results["config"]["model_id"]
        print(f"Loaded Phase 1 results: best_layer={config.best_layer}")

    print(f"Model: {config.model_id}")

    # Generate data
    from gspace.phase1_probes import generate_split_problems
    train_problems, test_problems = generate_split_problems(
        config.modulus, config.op, 2000, 500, config.seed
    )

    # Load model
    print("Loading model...")
    dtype = getattr(torch, config.dtype)
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        torch_dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
    )
    if device.type == "cpu":
        model = model.to(device)
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    d_model = model.config.hidden_size
    print(f"Layers: {n_layers}, Heads: {n_heads}, d_model: {d_model}")

    # Cache activations for operand probe training
    # We need activations at the position BEFORE the answer
    # Strategy: probe on the last token (where answer is predicted)
    # Operand A is at position of first number, B at second number — but
    # for simplicity, probe operands at the last prompt token
    from gspace.phase1_probes import cache_activations
    print("Caching activations for operand probes...")

    small_train = train_problems[:min(1000, len(train_problems))]
    act_data = cache_activations(model, tokenizer, small_train, device)

    best_layer_acts = act_data["activations"][config.best_layer]
    # For operand labels: use a and b from problem metadata
    a_labels = np.array([m["a"] for m in act_data["metadata"]])
    b_labels = np.array([m["b"] for m in act_data["metadata"]])

    # Train operand probes
    print("Training operand probes...")
    probe_a = train_operand_probes(best_layer_acts, a_labels)
    probe_b = train_operand_probes(best_layer_acts, b_labels)

    a_preds = probe_a.predict(best_layer_acts)
    b_preds = probe_b.predict(best_layer_acts)
    print(f"Probe A accuracy: {np.mean(a_preds == a_labels):.4f}")
    print(f"Probe B accuracy: {np.mean(b_preds == b_labels):.4f}")

    # Analyze QK twist for all heads in a band of layers around best_layer
    # Focus on early/mid layers where computation happens
    start_layer = max(0, config.best_layer - 2)
    end_layer = min(n_layers, config.best_layer + 2)
    analyze_layers = list(range(start_layer, end_layer))
    analyze_layers = [l for l in analyze_layers if l < n_layers]
    print(f"Analyzing QK twist in layers {analyze_layers}")

    head_results = analyze_all_heads(
        model, probe_a, probe_b, config.modulus, analyze_layers, device
    )

    # Find candidate addition heads
    addition_heads = []
    for layer, heads in head_results.items():
        for head, info in heads.items():
            if info["is_addition_head"]:
                addition_heads.append({
                    "layer": layer,
                    "head": head,
                    "best_sum": info["best_sum"],
                    "cosine": info["best_sum_cosine"],
                })
    addition_heads.sort(key=lambda h: h["cosine"], reverse=True)
    print(f"\nFound {len(addition_heads)} candidate addition heads")
    for h in addition_heads[:10]:
        print(f"  L{h['layer']}H{h['head']}: sum={h['best_sum']}, cos={h['cosine']:.4f}")

    # Save results
    results = {
        "config": {
            "model_id": config.model_id,
            "modulus": config.modulus,
            "op": config.op,
            "best_layer": config.best_layer,
            "analyzed_layers": analyze_layers,
        },
        "head_results": {
            str(l): {
                str(h): {k: v for k, v in info.items() if k != "cosine_matrix"}
                for h, info in heads.items()
            }
            for l, heads in head_results.items()
        },
        "addition_heads": addition_heads,
    }

    out_path = os.path.join(config.output_dir, "phase3_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    del model, tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
