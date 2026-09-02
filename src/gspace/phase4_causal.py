"""
Phase 4: Causal Validation via Subspace Ablation and Activation Patching.

Two experiments (from the paper):
1. Ablation: zero out the k-dimensional residue subspace → loss spikes on answer tokens.
2. Patching: substitute mean activations to manipulate perceived residue → behavior shifts.
"""

import json
import os
from dataclasses import dataclass, field

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from sklearn.decomposition import PCA


@dataclass
class Phase4Config:
    model_id: str = "Qwen/Qwen2.5-0.5B"
    modulus: int = 59
    op: str = "+"
    best_layer: int = 4
    manifold_dim: int = 6  # from Phase 2
    n_problems: int = 500
    output_dir: str = "results/phase4"
    seed: int = 42
    dtype: str = "bfloat16"


def compute_per_residue_means(
    activations: np.ndarray,
    labels: np.ndarray,
    modulus: int,
) -> dict[int, np.ndarray]:
    """Compute mean activation vector per residue value."""
    means = {}
    for r in range(modulus):
        mask = labels == r
        if mask.sum() > 0:
            means[r] = activations[mask].mean(axis=0)
        else:
            means[r] = np.zeros(activations.shape[1])
    return means


def run_ablation(
    model,
    tokenizer,
    problems: list[dict],
    device: torch.device,
    subspace_basis: np.ndarray,  # [k, d_model] — top-k PCA components of residue manifold
    target_layer: int,
    control: bool = False,
) -> dict:
    """Ablate the k-dimensional subspace and measure loss change.

    If control=True, ablate a random k-dimensional subspace instead.
    """
    k = subspace_basis.shape[0]
    d_model = subspace_basis.shape[1]
    basis_tensor = torch.tensor(subspace_basis, dtype=torch.float32, device=device)

    if control:
        # Generate random orthonormal k-dim subspace
        random_basis = torch.randn(k, d_model, device=device)
        Q, _ = torch.linalg.qr(random_basis.T)
        basis_tensor = Q[:, :k].T

    def ablation_hook(module, input, output):
        """Zero out components in the target subspace."""
        hidden = output[0]  # [batch, seq_len, d_model]
        # Project onto subspace and subtract
        coeffs = hidden @ basis_tensor.T  # [batch, seq_len, k]
        projection = coeffs @ basis_tensor  # [batch, seq_len, d_model]
        modified = hidden - projection  # zero out subspace
        return (modified,) + output[1:]

    # Register hook
    hook = model.model.layers[target_layer].register_forward_hook(ablation_hook)

    total_loss = 0.0
    total_newline_loss = 0.0  # for us: "answer token" loss
    n_tokens = 0

    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    with torch.no_grad():
        for i in tqdm(range(0, len(problems), 2), desc="Ablation eval"):
            batch = problems[i:i+2]
            prompts = [p["prompt"] for p in batch]
            targets = [int(p["result"]) for p in batch]

            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
            target_tensor = torch.tensor(targets, device=device)

            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :]   # last position logits
            loss = loss_fn(logits, target_tensor)

            total_loss += loss.sum().item()
            n_tokens += len(batch)

    hook.remove()

    avg_loss = total_loss / n_tokens if n_tokens > 0 else 0
    return {"avg_loss": avg_loss, "total_loss": total_loss, "n_tokens": n_tokens}


def run_patching(
    model,
    tokenizer,
    problems: list[dict],
    device: torch.device,
    per_residue_means: dict[int, np.ndarray],
    target_layer: int,
    k: int = 6,
) -> dict:
    """Patch activations to manipulate perceived residue at target layer.

    For each problem, substitute a_patched = a_original - μ_original + μ_target
    This makes the model think operand A has value 'target' instead of 'original'.

    We vary target over all residues and measure P(predicted answer).
    """
    # For simplicity: patch only on the last token, replacing the operand A representation
    # Actually we need to identify where operand A is represented.
    # Strategy: patch the last token's activation to specific residue means.

    # First compute PCA basis for rank-k patching
    mean_vecs = np.stack([per_residue_means[r] for r in range(len(per_residue_means))])
    pca = PCA(n_components=k)
    pca.fit(mean_vecs)

    results = []
    for orig_residue in tqdm(range(len(per_residue_means)), desc="Patching sweep"):
        for target_residue in range(len(per_residue_means)):
            # Find problems where the true operand A equals orig_residue
            matching = [p for p in problems if p["a"] == orig_residue]
            if not matching:
                continue

            mu_orig = per_residue_means[orig_residue]
            mu_target = per_residue_means[target_residue]
            delta = mu_target - mu_orig  # full dimension

            # Rank-k delta
            delta_proj = pca.inverse_transform(pca.transform(delta.reshape(1, -1))).flatten()
            delta_tensor = torch.tensor(delta_proj, dtype=torch.float32, device=device)

            def patching_hook(module, input, output):
                hidden = output[0]  # [batch, seq_len, d_model]
                # Add delta at last position
                last_pos = (hidden.shape[1] - 1)  # simplified: assume last position
                # Actually we need per-sample last positions; simplify for now
                hidden[:, -1, :] += delta_tensor
                return (hidden,) + output[1:]

            hook = model.model.layers[target_layer].register_forward_hook(patching_hook)

            correct = 0
            total = 0
            model.eval()
            with torch.no_grad():
                for p in matching[:10]:  # limit per (orig, target) pair
                    inputs = tokenizer(p["prompt"], return_tensors="pt").to(device)
                    outputs = model(**inputs)
                    pred_id = outputs.logits[0, -1, :].argmax().item()
                    pred_token = tokenizer.decode(pred_id).strip()
                    try:
                        pred_val = int(pred_token)
                    except ValueError:
                        pred_val = None
                    if pred_val == ((target_residue + p["b"]) % len(per_residue_means)):
                        correct += 1
                    total += 1

            hook.remove()

            if total > 0:
                results.append({
                    "original_a": orig_residue,
                    "target_a": target_residue,
                    "accuracy": correct / total,
                    "n": total,
                })

    return {
        "patching_results": results,
        "summary": "see patching_results for (original, target, accuracy) tuples"
    }


def main():
    config = Phase4Config()
    os.makedirs(config.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load Phase 1+2 results
    p1_path = "results/phase1/phase1_results.json"
    if os.path.exists(p1_path):
        with open(p1_path) as f:
            p1 = json.load(f)
        config.best_layer = p1["best_layer"]
        config.model_id = p1["config"]["model_id"]
        config.manifold_dim = min(6, next(
            (i+1 for i, r2 in enumerate(p1["manifold"]["r2_by_k"]) if r2 > 0.95), 6
        ))
        print(f"Loaded: best_layer={config.best_layer}, manifold_dim={config.manifold_dim}")

    # Generate data
    from gspace.phase1_probes import generate_split_problems
    train_problems, test_problems = generate_split_problems(
        config.modulus, config.op, 1000, config.n_problems, config.seed
    )

    # Load model
    print(f"Loading {config.model_id}...")
    dtype = getattr(torch, config.dtype)
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        torch_dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
    )
    if device.type == "cpu":
        model = model.to(device)

    # Cache activations for per-residue means
    from gspace.phase1_probes import cache_activations
    print("Caching activations...")
    act_data = cache_activations(model, tokenizer, train_problems[:500], device)
    best_acts = act_data["activations"][config.best_layer]

    # Get per-residue means for result
    result_labels = np.array([m["result"] for m in act_data["metadata"]])
    per_residue_means = compute_per_residue_means(best_acts, result_labels, config.modulus)

    # Compute manifold PCA basis
    residues = sorted(per_residue_means.keys())
    mean_vecs = np.stack([per_residue_means[r] for r in residues])
    pca = PCA(n_components=config.manifold_dim)
    pca.fit(mean_vecs)
    basis = pca.components_  # [k, d_model]
    print(f"Manifold PCA: top-{config.manifold_dim} explains {pca.explained_variance_ratio_.sum():.4f} variance")

    # EXPERIMENT 1: Ablation
    print("\n=== Experiment 1: Subspace Ablation ===")
    test_set = test_problems[:min(200, len(test_problems))]

    # Baseline (no ablation)
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    baseline_loss = 0
    with torch.no_grad():
        for p in tqdm(test_set, desc="Baseline"):
            inputs = tokenizer(p["prompt"], return_tensors="pt").to(device)
            target = torch.tensor([int(p["result"])], device=device)
            outputs = model(**inputs)
            loss = loss_fn(outputs.logits[:, -1, :], target)
            baseline_loss += loss.item()
    baseline_loss /= len(test_set)
    print(f"Baseline loss: {baseline_loss:.4f}")

    # Real ablation
    ablate_result = run_ablation(model, tokenizer, test_set, device, basis, config.best_layer)
    print(f"Ablation loss: {ablate_result['avg_loss']:.4f}")

    # Control ablation (random subspace)
    ctrl_result = run_ablation(model, tokenizer, test_set, device, basis, config.best_layer, control=True)
    print(f"Control (random) ablation loss: {ctrl_result['avg_loss']:.4f}")

    # Dimensionality sweep
    dim_sweep = {}
    for k in range(1, min(config.manifold_dim + 6, basis.shape[1] + 1)):
        k_basis = basis[:k] if k <= config.manifold_dim else pca.components_[:k]
        r = run_ablation(model, tokenizer, test_set[:50], device, k_basis, config.best_layer)
        dim_sweep[k] = r["avg_loss"]
    print(f"Dimensionality sweep: {dim_sweep}")

    # EXPERIMENT 2: Patching
    print("\n=== Experiment 2: Activation Patching ===")
    patch_results = run_patching(
        model, tokenizer, test_set, device, per_residue_means,
        config.best_layer, k=config.manifold_dim
    )

    # Save
    results = {
        "config": {"model_id": config.model_id, "best_layer": config.best_layer,
                   "manifold_dim": config.manifold_dim},
        "baseline_loss": baseline_loss,
        "ablation_loss": ablate_result["avg_loss"],
        "control_ablation_loss": ctrl_result["avg_loss"],
        "ablation_delta": ablate_result["avg_loss"] - baseline_loss,
        "dimensionality_sweep": dim_sweep,
        "patching": patch_results,
    }

    out_path = os.path.join(config.output_dir, "phase4_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    del model, tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
