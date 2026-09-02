"""
Phase 1: Behavioral Confirmation & Linear Probing.

1. Cache residual stream activations for all layers on the benchmark.
2. Train 59-way logistic probes at each layer to predict residue.
3. Produce layer-wise accuracy curve → find where the residue manifold is sharpest.
4. Compute PCA of probe weights and cosine similarity matrix.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score


@dataclass
class Phase1Config:
    model_id: str = "Qwen/Qwen2.5-0.5B"
    modulus: int = 59
    op: str = "+"
    n_train: int = 2400  # ~70% of 3481 total pairs
    n_test: int = 1081    # ~30%
    output_dir: str = "results/phase1"
    seed: int = 42
    dtype: str = "bfloat16"
    # Which token position to probe (relative to prompt end)
    probe_position: int = -1  # last token
    max_train_probe: int = 50000  # max samples for probe training


def generate_split_problems(modulus: int, op: str, n_train: int, n_test: int, seed: int):
    """Generate train/test split of arithmetic problems."""
    import random
    rng = random.Random(seed)
    all_pairs = [(a, b) for a in range(modulus) for b in range(modulus)]
    rng.shuffle(all_pairs)

    test_pairs = all_pairs[:n_test]
    train_pairs = all_pairs[n_test:n_test + n_train]

    def make_prompt(a, b):
        return f"Compute {a} {op} {b} mod {modulus}: {a} {op} {b} mod {modulus} ="

    result_fn = {"+": lambda a, b: (a + b) % modulus,
                 "-": lambda a, b: (a - b) % modulus,
                 "*": lambda a, b: (a * b) % modulus}[op]

    def make_problems(pairs):
        return [{"a": a, "b": b, "result": result_fn(a, b),
                 "prompt": make_prompt(a, b)} for a, b in pairs]

    return make_problems(train_pairs), make_problems(test_pairs)


def cache_activations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    problems: list[dict],
    device: torch.device,
    batch_size: int = 4,
    layers: Optional[list[int]] = None,
) -> dict:
    """Cache residual stream activations for all specified layers at target position.

    Returns dict with:
        activations: dict[layer_idx] -> np.ndarray [n_samples, d_model]
        labels: np.ndarray [n_samples] of target residues
        metadata: dict with a, b per sample
    """
    n_layers = model.config.num_hidden_layers
    if layers is None:
        layers = list(range(n_layers))

    # Register hooks to capture hidden states
    hidden_states = {}

    def make_hook(layer_idx):
        def hook(module, input, output):
            # output is a tuple, first element is the hidden states tensor
            hidden_states[layer_idx] = output[0].detach().cpu()
        return hook

    hooks = []
    for l in layers:
        hook = model.model.layers[l].register_forward_hook(make_hook(l))
        hooks.append(hook)

    all_activations = {l: [] for l in layers}
    all_labels = []
    all_metadata = []

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(problems), batch_size), desc="Caching activations"):
            batch = problems[i:i + batch_size]
            prompts = [p["prompt"] for p in batch]
            labels = [p["result"] for p in batch]

            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
            _ = model(**inputs, output_hidden_states=False)

            for l in layers:
                # Get last token position for each sample
                hs = hidden_states[l]  # [batch, seq_len, d_model]
                # Position: last non-padding token
                last_pos = inputs["attention_mask"].sum(dim=1) - 1  # [batch]
                batch_acts = hs[range(len(last_pos)), last_pos].numpy()  # [batch, d_model]
                all_activations[l].append(batch_acts)

            all_labels.extend(labels)
            all_metadata.extend([{"a": p["a"], "b": p["b"]} for p in batch])

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Concatenate
    activations = {l: np.concatenate(all_activations[l], axis=0) for l in layers}
    labels_arr = np.array(all_labels)

    return {"activations": activations, "labels": labels_arr, "metadata": all_metadata}


def train_probes(
    activations: np.ndarray,
    labels: np.ndarray,
    C: float = 1.0,
    max_iter: int = 500,
) -> tuple[LogisticRegression, float]:
    """Train a multinomial logistic probe and return (model, accuracy)."""
    # Subsample if too large
    n = len(activations)
    if n > 50000:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, 50000, replace=False)
        activations = activations[idx]
        labels = labels[idx]

    probe = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=max_iter,
        C=C,
    )
    probe.fit(activations, labels)
    preds = probe.predict(activations)
    acc = accuracy_score(labels, preds)

    # Also compute within-±5 accuracy
    within_5 = np.mean(np.abs(preds - labels) <= 5)
    return probe, acc, within_5


def probe_layer_sweep(
    activations_dict: dict[int, np.ndarray],
    labels: np.ndarray,
    test_activations_dict: dict[int, np.ndarray],
    test_labels: np.ndarray,
) -> dict:
    """Train probes at each layer, report train + test accuracy."""
    results = {}
    for layer in tqdm(sorted(activations_dict.keys()), desc="Probing layers"):
        train_acts = activations_dict[layer]
        test_acts = test_activations_dict[layer]

        probe, train_acc, train_within5 = train_probes(train_acts, labels)
        test_preds = probe.predict(test_acts)
        test_acc = accuracy_score(test_labels, test_preds)
        test_within5 = np.mean(np.abs(test_preds - test_labels) <= 5)

        results[layer] = {
            "train_acc": float(train_acc),
            "test_acc": float(test_acc),
            "train_within5": float(train_within5),
            "test_within5": float(test_within5),
            "probe_weights": probe.coef_,  # [59, d_model]
            "probe_intercepts": probe.intercept_,
        }
    return results


def analyze_probe_geometry(probe_weights: np.ndarray, modulus: int) -> dict:
    """Analyze geometry of probe weight vectors.

    probe_weights: [modulus, d_model]
    Returns PCA variance, cosine similarity matrix, etc.
    """
    # PCA of probe weight vectors
    pca = PCA()
    pca.fit(probe_weights)
    explained_var = pca.cumulative_explained_variance_ratio_

    # Cosine similarity matrix
    norms = np.linalg.norm(probe_weights, axis=1, keepdims=True)
    normalized = probe_weights / (norms + 1e-8)
    cos_sim = normalized @ normalized.T  # [modulus, modulus]

    return {
        "pca_explained_variance": explained_var.tolist(),
        "cosine_similarity": cos_sim.tolist(),
        "probe_weights_pca": pca.transform(probe_weights).tolist(),  # top components
    }


def main():
    config = Phase1Config()
    os.makedirs(config.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {config.model_id}")
    print(f"Modulus: {config.modulus}, Op: {config.op}")

    # Generate data
    train_problems, test_problems = generate_split_problems(
        config.modulus, config.op, config.n_train, config.n_test, config.seed
    )
    print(f"Train: {len(train_problems)}, Test: {len(test_problems)}")

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
    d_model = model.config.hidden_size
    print(f"Layers: {n_layers}, d_model: {d_model}")

    # Cache activations
    print("Caching train activations...")
    train_data = cache_activations(model, tokenizer, train_problems, device)
    print("Caching test activations...")
    test_data = cache_activations(model, tokenizer, test_problems, device)

    # Save cached activations
    cache_path = os.path.join(config.output_dir, "activations.npz")
    np.savez_compressed(
        cache_path,
        **{f"train_l{l}": train_data["activations"][l] for l in range(n_layers)},
        **{f"test_l{l}": test_data["activations"][l] for l in range(n_layers)},
        train_labels=train_data["labels"],
        test_labels=test_data["labels"],
    )
    print(f"Cached activations saved to {cache_path}")

    # Probe each layer
    print("Training probes...")
    probe_results = probe_layer_sweep(
        train_data["activations"], train_data["labels"],
        test_data["activations"], test_data["labels"],
    )

    # Find best layer
    best_layer = max(probe_results, key=lambda l: probe_results[l]["test_acc"])
    print(f"\nBest layer: {best_layer} (test acc: {probe_results[best_layer]['test_acc']:.4f})")

    # Analyze probe geometry at best layer and a few others
    key_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1, best_layer]
    key_layers = sorted(set(l for l in key_layers if l in probe_results))

    geometry = {}
    for l in key_layers:
        pw = probe_results[l]["probe_weights"]
        geometry[l] = analyze_probe_geometry(pw, config.modulus)

    # Compute PCA of per-residue mean activations at best layer
    print("\nComputing per-residue mean activations...")
    acts = train_data["activations"][best_layer]
    labels = train_data["labels"]

    per_residue_means = {}
    for r in range(config.modulus):
        mask = labels == r
        if mask.sum() > 0:
            per_residue_means[r] = acts[mask].mean(axis=0)

    # Sort by residue, stack
    residues = sorted(per_residue_means.keys())
    mean_vectors = np.stack([per_residue_means[r] for r in residues])  # [59, d_model]

    # PCA
    pca = PCA()
    pca.fit(mean_vectors)
    projected = pca.transform(mean_vectors)

    # Cosine similarity
    norms = np.linalg.norm(mean_vectors, axis=1, keepdims=True)
    normalized = mean_vectors / (norms + 1e-8)
    cos_sim = normalized @ normalized.T

    manifold_results = {
        "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
        "pca_projection": projected.tolist(),
        "cosine_similarity": cos_sim.tolist(),
        "r2_by_k": [],  # R²(residue | top-k PCs) for k=1..10
    }

    # R² by k
    from sklearn.linear_model import LinearRegression
    for k in range(1, min(11, d_model)):
        X = projected[:, :k]
        y = np.array(residues, dtype=float)
        reg = LinearRegression().fit(X, y)
        r2 = reg.score(X, y)
        manifold_results["r2_by_k"].append(float(r2))
    print(f"R² by k: {[f'{v:.4f}' for v in manifold_results['r2_by_k']]}")

    # Save all results
    results = {
        "config": {
            "model_id": config.model_id,
            "modulus": config.modulus,
            "op": config.op,
            "n_layers": n_layers,
            "d_model": d_model,
        },
        "best_layer": best_layer,
        "probe_results": {str(l): {k: v for k, v in r.items()
                                    if k not in ("probe_weights", "probe_intercepts")}
                           for l, r in probe_results.items()},
        "probe_geometry": {str(l): g for l, g in geometry.items()},
        "manifold": manifold_results,
    }

    out_path = os.path.join(config.output_dir, "phase1_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Cleanup
    del model, tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
