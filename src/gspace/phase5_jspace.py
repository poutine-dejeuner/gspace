"""
Phase 5: J-Space Comparison.

Compare the geometric subspace (g-space) with the Jacobian-lens workspace (j-space).

J-space: the dominant singular vectors of the input-output Jacobian at a layer.
∂(output_logits) / ∂(residual_at_layer)

The hypothesis: g-space is active in layers l < l_0 (computation),
and j-space captures the result in layers l >= l_0 (readout).
"""

import json
import os
from dataclasses import dataclass, field

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


@dataclass
class Phase5Config:
    model_id: str = "Qwen/Qwen2.5-0.5B"
    modulus: int = 59
    op: str = "+"
    best_layer: int = 4
    manifold_dim: int = 6
    output_dir: str = "results/phase5"
    seed: int = 42
    dtype: str = "bfloat16"
    n_jacobian_samples: int = 100
    jacobian_layers: list = None  # set in main


def compute_jspace(
    model,
    tokenizer,
    problems: list[dict],
    device: torch.device,
    layers: list[int],
    n_samples: int = 100,
    top_k: int = 50,
) -> dict[int, dict]:
    """Compute J-space (top singular vectors of input-output Jacobian) per layer.

    For each layer, compute ∂(logit_of_correct_answer)/∂(residual_at_layer)
    using backward-mode differentiation. Stack gradients across samples,
    then SVD the gradient matrix.

    Returns dict[layer] -> {svals, left_vecs, right_vecs}
    """
    d_model = model.config.hidden_size

    results = {}

    for layer in tqdm(layers, desc="J-space layers"):
        gradients = []
        correct_logit = []

        # Register hook to capture residual stream at this layer
        residual = {}

        def make_hook(l):
            def hook(module, input, output):
                residual[l] = output[0]  # keep gradient graph
            return hook

        hook = model.model.layers[layer].register_forward_hook(make_hook(layer))

        # Forward + backward for each sample
        model.train()  # need grad
        subset = problems[:min(n_samples, len(problems))]

        for p in tqdm(subset, desc=f"  L{layer}", leave=False):
            inputs = tokenizer(p["prompt"], return_tensors="pt").to(device)
            target_token_id = tokenizer.encode(str(p["result"]), add_special_tokens=False)[0]

            model.zero_grad()

            # We need the residual with grad. Use output_hidden_states + retain_grad
            outputs = model(**inputs, output_hidden_states=True)

            # Alternative: use the hook to get residual with grad
            # The hook captured residual at forward pass; now backprop
            logits = outputs.logits[0, -1, :]
            target_logit = logits[target_token_id]
            target_logit.backward(retain_graph=False)

            # The residual should have .grad now
            if hasattr(residual[layer], 'grad') and residual[layer].grad is not None:
                grad = residual[layer].grad[0, -1, :].detach().cpu().numpy()  # [d_model]
                gradients.append(grad)
            else:
                # Fallback: use finite differences or skip
                pass

            model.zero_grad()

        hook.remove()
        model.eval()

        if len(gradients) == 0:
            print(f"  WARNING: no gradients collected for layer {layer}")
            results[layer] = {"error": "no gradients"}
            continue

        # Stack gradients [n_samples, d_model]
        G = np.stack(gradients, axis=0)

        # SVD
        G_centered = G - G.mean(axis=0)
        U, S, Vt = np.linalg.svd(G_centered, full_matrices=False)

        results[layer] = {
            "singular_values": S[:top_k].tolist(),
            "right_singular_vectors": Vt[:top_k].tolist(),  # j-space basis [top_k, d_model]
            "n_samples": len(gradients),
        }

    return results


def compute_subspace_alignment(
    gspace_basis: np.ndarray,  # [k_g, d_model]
    jspace_basis: np.ndarray,  # [k_j, d_model]
) -> float:
    """Compute alignment between g-space and j-space.

    Uses the principal angles / Grassmann distance:
    alignment = ||P_G @ P_J||_F^2 / sqrt(k_g * k_j)

    Where P_G, P_J are projection matrices onto the respective subspaces.
    """
    # Normalize basis vectors
    g_norm = gspace_basis / (np.linalg.norm(gspace_basis, axis=1, keepdims=True) + 1e-8)
    j_norm = jspace_basis / (np.linalg.norm(jspace_basis, axis=1, keepdims=True) + 1e-8)

    # Cross-projection
    cross = g_norm @ j_norm.T  # [k_g, k_j]
    # Average squared cosine similarity
    alignment = np.mean(cross ** 2)

    # Normalize: max possible is 1.0 (identical subspaces)
    # For subspaces with different dimensions, we use:
    # alignment = trace(P_G @ P_J) / min(k_g, k_j)
    # Which is: sum of squared cosines of principal angles / min(k_g, k_j)
    k_min = min(gspace_basis.shape[0], jspace_basis.shape[0])
    svals = np.linalg.svd(cross, compute_uv=False)
    canon_corr = np.sum(svals ** 2) / k_min

    return float(canon_corr)


def main():
    config = Phase5Config()
    os.makedirs(config.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load Phase 1 + 2 results
    p1_path = "results/phase1/phase1_results.json"
    manifold_basis = None
    if os.path.exists(p1_path):
        with open(p1_path) as f:
            p1 = json.load(f)
        config.best_layer = p1["best_layer"]
        config.model_id = p1["config"]["model_id"]
        config.manifold_dim = min(6, next(
            (i+1 for i, r2 in enumerate(p1["manifold"]["r2_by_k"]) if r2 > 0.95), 6
        ))
        print(f"Loaded: best_layer={config.best_layer}, manifold_dim={config.manifold_dim}")

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
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    print(f"Layers: {n_layers}, d_model: {d_model}")

    # Generate problems
    from gspace.phase1_probes import generate_split_problems
    _, test_problems = generate_split_problems(
        config.modulus, config.op, 500, config.n_jacobian_samples * 2, config.seed
    )

    # Select layers to analyze (evenly spaced across model)
    if config.jacobian_layers is None:
        # Sample ~8 layers across the model
        step = max(1, n_layers // 8)
        config.jacobian_layers = list(range(0, n_layers, step))
    print(f"Analyzing J-space at layers: {config.jacobian_layers}")

    # Compute J-space
    print("Computing J-space (this may take a while)...")
    jspace = compute_jspace(
        model, tokenizer, test_problems, device,
        config.jacobian_layers, n_samples=config.n_jacobian_samples
    )

    # Build g-space basis from Phase 2 manifold results
    # g-space = top-k PCA components of per-residue mean vectors
    if manifold_basis is None:
        # Reconstruct from cached activations
        from gspace.phase1_probes import cache_activations
        print("Recomputing g-space basis...")
        subset = test_problems[:min(500, len(test_problems))]
        act_data = cache_activations(model, tokenizer, subset, device)
        best_acts = act_data["activations"][config.best_layer]

        from gspace.phase4_causal import compute_per_residue_means
        result_labels = np.array([m["result"] for m in act_data["metadata"]])
        means = compute_per_residue_means(best_acts, result_labels, config.modulus)

        residues = sorted(means.keys())
        mean_vecs = np.stack([means[r] for r in residues])
        from sklearn.decomposition import PCA
        pca = PCA(n_components=config.manifold_dim)
        pca.fit(mean_vecs)
        gspace_basis_arr = pca.components_  # [k, d_model]
        print(f"G-space: top-{config.manifold_dim} PCs explain {pca.explained_variance_ratio_.sum():.4f}")
    else:
        gspace_basis_arr = np.array(manifold_basis)

    # Compute alignment per layer
    alignments = {}
    for layer in sorted(jspace.keys()):
        if "error" in jspace[layer]:
            alignments[layer] = {"error": jspace[layer]["error"]}
            continue

        j_basis = np.array(jspace[layer]["right_singular_vectors"])  # [top_k, d_model]
        # Use same k as g-space for fair comparison
        j_basis = j_basis[:config.manifold_dim, :]

        alignment = compute_subspace_alignment(gspace_basis_arr, j_basis)
        alignments[layer] = {
            "alignment": alignment,
            "jspace_top_svals": jspace[layer]["singular_values"][:5],
        }
        print(f"  Layer {layer}: alignment = {alignment:.4f}")

    # Check hypothesis: alignment should be low in early layers (g-space ≠ j-space)
    # and increase after computation is done
    early_align = np.mean([v["alignment"] for l, v in alignments.items()
                           if isinstance(v, dict) and "alignment" in v
                           and l < config.best_layer])
    late_align = np.mean([v["alignment"] for l, v in alignments.items()
                          if isinstance(v, dict) and "alignment" in v
                          and l >= config.best_layer])
    print(f"\nEarly layers (l < {config.best_layer}) mean alignment: {early_align:.4f}")
    print(f"Late layers (l >= {config.best_layer}) mean alignment: {late_align:.4f}")
    print(f"Hypothesis check: {'SUPPORTED' if late_align > early_align else 'REJECTED' if late_align <= early_align else 'INCONCLUSIVE'}")

    # Save
    results = {
        "config": {
            "model_id": config.model_id,
            "best_layer": config.best_layer,
            "manifold_dim": config.manifold_dim,
            "jacobian_layers": config.jacobian_layers,
        },
        "jspace": {str(l): v for l, v in jspace.items()},
        "alignments": {str(l): v for l, v in alignments.items()},
        "early_mean_alignment": float(early_align),
        "late_mean_alignment": float(late_align),
        "hypothesis": "supported" if late_align > early_align else "rejected",
    }

    out_path = os.path.join(config.output_dir, "phase5_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    del model, tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
