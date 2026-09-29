"""
Intervention experiments à la "Global Workspace" paper.

Key insight from the paper: interventions must be applied at ALL token positions
and across the FULL workspace layer range, not just a single layer.

Our approach for Qwen3.5-27B (64 layers, workspace ≈ 48-63):
  For each layer in workspace_range:
    For each token position:
      Compute h's projection onto J-lens vectors v_true and v_wrong
      Replace v_true component with v_wrong component
  This is the "patching in lens coordinates" method: h += V(σ(c) - c)
  where V = [v_true, v_wrong], c = V† h, σ swaps the two entries.

"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from safetensors.torch import load_file as safe_load
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


MODULUS = 59
# Workspace boundaries identified by identify_workspace.py:
#   Sensory: L0-L20, Workspace: L21-L60, Motor: L61-L63
SENSORY_RANGE = (0, 20)
WORKSPACE_RANGE = (21, 60)
MOTOR_RANGE = (61, 63)
D_MODEL = 5120


# ═════════════════════════════════════════════════════════════════════
# Loading
# ═════════════════════════════════════════════════════════════════════

def load_jlens(path):
    print(f"Loading J-lens from {path}")
    d = torch.load(path, map_location="cpu", weights_only=True)
    J_dict = d["J"]
    J_matrices = {}
    for layer_name, J in J_dict.items():
        l = int(layer_name)
        J_matrices[l] = J.to(torch.float32).numpy()
    print(f"  {len(J_matrices)} layers")
    return J_matrices


def load_wu(cache_dir=None):
    if cache_dir is None:
        cache_dir = os.path.expanduser(
            "~/.cache/huggingface/hub/models--Qwen--Qwen3.5-27B/"
            "snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654"
        )
    print(f"Loading W_U from {cache_dir}")
    for fname in sorted(os.listdir(cache_dir)):
        if fname.endswith(".safetensors"):
            tensors = safe_load(os.path.join(cache_dir, fname))
            if "lm_head.weight" in tensors:
                wu = tensors["lm_head.weight"].to(torch.float32).numpy()
                return wu
    raise FileNotFoundError("lm_head.weight not found")


# ═════════════════════════════════════════════════════════════════════
# Token mapping for Qwen
# ═════════════════════════════════════════════════════════════════════

def build_digit_tokens(tokenizer):
    """Map digit 0-9 to their single token IDs in Qwen."""
    digit_tokens = {}
    for d in range(10):
        for fmt in [str(d), f" {d}"]:
            tok_id = tokenizer.encode(fmt, add_special_tokens=False)
            if len(tok_id) == 1:
                digit_tokens[d] = tok_id[0]
                break
    return digit_tokens


def get_answer_tokens(ans, digit_tokens):
    """Get token IDs for a number. Qwen tokenizes by digit."""
    s = str(ans)
    toks = []
    for ch in s:
        d = int(ch)
        if d in digit_tokens:
            toks.append(digit_tokens[d])
    return toks if toks else [digit_tokens[0]]


# ═════════════════════════════════════════════════════════════════════
# J-lens coordinate patching
# ═════════════════════════════════════════════════════════════════════

class JSpacePatcher:
    """
    Patches J-lens coordinates across multiple layers.

    For each intervened layer:
      h_patched = h + V @ (σ(c) - c) * alpha
      where V = [v_true, v_wrong], c = V† h, σ swaps the two entries.

    Memory optimization: We compute (w_true @ J_ℓ) and (w_wrong @ J_ℓ) on-the-fly
    per layer instead of precomputing all (W_U @ J_ℓ) for all 248K tokens.
    """

    def __init__(self, J_matrices, W_U, true_tok, wrong_tok,
                 layers_to_intervene, alpha=1.0, all_positions=True):
        self.J_matrices = J_matrices  # {layer: np array (d_model, d_model)}
        self.W_U = W_U  # numpy (vocab, d_model)
        self.layers = layers_to_intervene
        self.alpha = alpha
        self.all_positions = all_positions

        # Extract single-token row vectors (1, d_model)
        self.w_true = W_U[true_tok].astype(np.float32)   # (d_model,)
        self.w_wrong = W_U[wrong_tok].astype(np.float32)  # (d_model,)

        # Pre-compute V and V† for each layer
        self.V_per_layer = {}
        self.Vinv_per_layer = {}
        for l in layers_to_intervene:
            # v_token = w_token @ J_ℓ  (d_model, d_model) @ (d_model,) -> (d_model,)
            v_t = (self.w_true.astype(np.float64) @ J_matrices[l].astype(np.float64)).astype(np.float32)
            v_w = (self.w_wrong.astype(np.float64) @ J_matrices[l].astype(np.float64)).astype(np.float32)
            # V = [v_t, v_w] -> (d_model, 2)
            V = np.stack([v_t, v_w], axis=1).astype(np.float32)
            # Pseudoinverse via (V^T V)^{-1} V^T
            VTV = V.T @ V  # (2, 2)
            Vinv = np.linalg.solve(VTV + 1e-8 * np.eye(2), V.T).astype(np.float32)  # (2, d_model)
            self.V_per_layer[l] = torch.from_numpy(V)
            self.Vinv_per_layer[l] = torch.from_numpy(Vinv)

    def hook_fn(self, module, input, output, layer_idx):
        if layer_idx not in self.layers:
            return output

        # Handle tuple output from some layer implementations
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output

        V = self.V_per_layer[layer_idx].to(h.device, dtype=h.dtype)
        Vinv = self.Vinv_per_layer[layer_idx].to(h.device, dtype=h.dtype)

        # h: [batch, seq, d_model]
        # c = Vinv @ h: [batch, seq, 2]
        c = torch.einsum("bsd,cd->bsc", h, Vinv)  # [B, S, 2]

        # Swapped coordinates
        sigma_c = c[..., [1, 0]]  # [B, S, 2]

        # Interpolate: c_new = c + alpha * (sigma_c - c)
        delta_c = self.alpha * (sigma_c - c)  # [B, S, 2]

        # h_patched = h + V @ delta_c
        delta_h = torch.einsum("bsc,dc->bsd", delta_c, V)  # [B, S, d_model]

        if not self.all_positions:
            # Only intervene at last position
            mask = torch.zeros_like(h)
            mask[:, -1:, :] = 1.0
            delta_h = delta_h * mask

        h_out = h + delta_h

        if isinstance(output, tuple):
            return (h_out,) + output[1:]
        return h_out


# ═════════════════════════════════════════════════════════════════════
# Run intervention
# ═════════════════════════════════════════════════════════════════════

def run_patching_intervention(model, tokenizer, J_matrices, W_U,
                              digit_tokens, device,
                              n_samples=100, alpha=1.0,
                              layers_to_intervene=None,
                              all_positions=True):
    """Full workspace patching intervention.

    Evaluates via rank shift of true vs wrong answer tokens at the post-'=' position.
    Rank-based flip: wrong token overtakes true token in the ranked list.
    """
    if layers_to_intervene is None:
        layers_to_intervene = list(range(WORKSPACE_RANGE[0], WORKSPACE_RANGE[1] + 1))

    # Generate test problems
    rng = random.Random(42)
    all_pairs = [(a, b) for a in range(MODULUS) for b in range(MODULUS)]
    rng.shuffle(all_pairs)
    test_pairs = all_pairs[:n_samples]

    n_rank_flipped = 0  # wrong rank < true rank after intervention
    rank_changes = []

    for a, b in tqdm(test_pairs, desc=f"α={alpha} L{layers_to_intervene[0]}-{layers_to_intervene[-1]}"):
        true_ans = (a + b) % MODULUS
        wrong_ans = (true_ans + 1) % MODULUS

        prompt = f"Compute {a} + {b} mod {MODULUS}: {a} + {b} mod {MODULUS} ="
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        true_tok = get_answer_tokens(true_ans, digit_tokens)[0]
        wrong_tok = get_answer_tokens(wrong_ans, digit_tokens)[0]

        # --- Clean forward pass ---
        with torch.no_grad():
            out_clean = model(**inputs)
            logits_clean = out_clean.logits[0, -1, :].float().cpu().numpy()

        # --- Intervened forward pass with hooks ---
        patcher = JSpacePatcher(
            J_matrices, W_U, true_tok, wrong_tok,
            layers_to_intervene, alpha=alpha,
            all_positions=all_positions,
        )
        hooks = []
        for l in layers_to_intervene:
            target = model.model.layers[l]
            def make_fn(layer_idx):
                def fn(module, inp, out):
                    return patcher.hook_fn(module, inp, out, layer_idx)
                return fn
            hook_handle = target.register_forward_hook(make_fn(l))
            hooks.append(hook_handle)

        with torch.no_grad():
            out_interv = model(**inputs)
            logits_interv = out_interv.logits[0, -1, :].float().cpu().numpy()

        for h in hooks:
            h.remove()

        # --- Ranks ---
        rank_clean_true = int((logits_clean > logits_clean[true_tok]).sum() + 1)
        rank_clean_wrong = int((logits_clean > logits_clean[wrong_tok]).sum() + 1)
        rank_interv_true = int((logits_interv > logits_interv[true_tok]).sum() + 1)
        rank_interv_wrong = int((logits_interv > logits_interv[wrong_tok]).sum() + 1)

        # Rank flip: wrong now ranks BETTER (lower number) than true
        clean_true_better = rank_clean_true < rank_clean_wrong
        interv_wrong_better = rank_interv_wrong < rank_interv_true
        rank_flipped = clean_true_better and interv_wrong_better

        if rank_flipped:
            n_rank_flipped += 1

        rank_changes.append({
            "clean_true": rank_clean_true,
            "clean_wrong": rank_clean_wrong,
            "interv_true": rank_interv_true,
            "interv_wrong": rank_interv_wrong,
            "rank_flipped": rank_flipped,
        })

    results = {
        "alpha": alpha,
        "layers": layers_to_intervene,
        "n_samples": len(test_pairs),
        "n_rank_flipped": n_rank_flipped,
        "rank_flip_rate": n_rank_flipped / max(len(test_pairs), 1),
        "avg_clean_true_rank": float(np.mean([r["clean_true"] for r in rank_changes])),
        "avg_clean_wrong_rank": float(np.mean([r["clean_wrong"] for r in rank_changes])),
        "avg_interv_true_rank": float(np.mean([r["interv_true"] for r in rank_changes])),
        "avg_interv_wrong_rank": float(np.mean([r["interv_wrong"] for r in rank_changes])),
    }

    print(f"    true rank: {results['avg_clean_true_rank']:.2f} → {results['avg_interv_true_rank']:.2f}")
    print(f"    wrong rank: {results['avg_clean_wrong_rank']:.2f} → {results['avg_interv_wrong_rank']:.2f}")
    print(f"    rank-flipped: {n_rank_flipped}/{len(test_pairs)} "
          f"({results['rank_flip_rate']*100:.1f}%)")

    return results


# ═════════════════════════════════════════════════════════════════════
# Sweep
# ═════════════════════════════════════════════════════════════════════

def sweep_intervention(model, tokenizer, J_matrices, W_U, digit_tokens, device,
                       n_samples=100):
    """Sweep over alphas and layer ranges."""
    print("\n" + "=" * 60)
    print("SWEEP: Alpha and layer range")
    print("=" * 60)

    alphas = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
    layer_configs = {
        # Single layer baselines
        "L61 only (motor start)": [61],
        "L62 only (peak J-space)": [62],
        # Actual identified boundaries
        "L21-60 (full workspace)": list(range(21, 61)),  # 40 layers
        "L30-60 (mid workspace)": list(range(30, 61)),   # 31 layers
        # Motor only
        "L61-62 (motor)": list(range(61, 63)),           # 2 layers
        # Extended: workspace + motor
        "L21-62 (workspace+motor)": list(range(21, 63)), # 42 layers
        # Control: early layers
        "L0-20 (sensory)": list(range(0, 21)),          # 21 layers
    }

    all_results = {}
    for name, layers in layer_configs.items():
        for alpha in alphas:
            key = f"{name}_alpha{alpha}"
            print(f"\n  {name}, α={alpha}")
            r = run_patching_intervention(
                model, tokenizer, J_matrices, W_U, digit_tokens, device,
                n_samples=n_samples, alpha=alpha,
                layers_to_intervene=layers,
                all_positions=True,
            )
            all_results[key] = r

    return all_results


# ═════════════════════════════════════════════════════════════════════
# Visualization
# ═════════════════════════════════════════════════════════════════════

def plot_sweep(all_results, fig_dir):
    alphas = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
    configs = sorted(set(k.rsplit("_alpha", 1)[0] for k in all_results.keys()),
                     key=lambda c: len(c))

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel 1: True answer rank vs alpha
    ax = axes[0, 0]
    for cfg in configs:
        ranks = [all_results[f"{cfg}_alpha{a}"]["avg_interv_true_rank"] for a in alphas]
        ax.plot(alphas, ranks, "o-", linewidth=2, markersize=6, label=cfg)
    ax.set_xlabel("α (intervention strength)")
    ax.set_ylabel("Mean rank of true answer")
    ax.set_title("True answer rank degradation")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    # Panel 2: Wrong answer rank vs alpha
    ax = axes[0, 1]
    for cfg in configs:
        ranks = [all_results[f"{cfg}_alpha{a}"]["avg_interv_wrong_rank"] for a in alphas]
        ax.plot(alphas, ranks, "s-", linewidth=2, markersize=6, label=cfg)
    ax.set_xlabel("α (intervention strength)")
    ax.set_ylabel("Mean rank of wrong answer")
    ax.set_title("Wrong answer rank improvement")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    # Panel 3: Rank flip rate vs alpha
    ax = axes[1, 0]
    for cfg in configs:
        rates = [all_results[f"{cfg}_alpha{a}"]["rank_flip_rate"] * 100 for a in alphas]
        ax.plot(alphas, rates, "o-", linewidth=2, markersize=6, label=cfg)
    ax.set_xlabel("α (intervention strength)")
    ax.set_ylabel("Rank flip rate (%)")
    ax.set_title("Rank flip rate (wrong overtakes true)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: Best rank flip rate per config
    ax = axes[1, 1]
    best_rates = {}
    best_alphas = {}
    for cfg in configs:
        rates = [(all_results[f"{cfg}_alpha{a}"]["rank_flip_rate"], a) for a in alphas]
        best_rate, best_a = max(rates, key=lambda x: x[0])
        best_rates[cfg] = best_rate * 100
        best_alphas[cfg] = best_a

    x_labels = [c.replace(" (full workspace)", "\n(full workspace)").replace(" (extended)", "\n(extended)")
                for c in configs]
    bars = ax.bar(range(len(configs)), [best_rates[c] for c in configs],
                  color="tab:blue", alpha=0.7)
    for i, cfg in enumerate(configs):
        ax.text(i, best_rates[cfg] + 0.5, f"α={best_alphas[cfg]}",
                ha="center", fontsize=8)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_ylabel("Best rank flip rate (%)")
    ax.set_title("Maximum rank flip rate per layer range")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("J-space Intervention — Full Workspace Patching",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig_path = fig_dir / "intervention_sweep.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fig_path}")


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Workspace intervention experiments")
    parser.add_argument("--output-dir", default="results/intervention")
    parser.add_argument("--j-path",
                        default="results/jspace_cache/qwen3.5-27b/jlens/"
                                "Salesforce-wikitext/Qwen3.5-27B_jacobian_lens.pt")
    parser.add_argument("--model-path",
                        default=os.path.expanduser(
                            "~/.cache/huggingface/hub/models--Qwen--Qwen3.5-27B/"
                            "snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654"))
    parser.add_argument("--smoke-test", type=int, default=0)
    parser.add_argument("--n-samples", type=int, default=100,
                        help="Number of samples per experiment (default: 100)")
    args = parser.parse_args()

    OUTPUT_DIR = Path(args.output_dir)
    FIG_DIR = Path("figures") / OUTPUT_DIR.name
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load J-lens
    J_matrices = load_jlens(args.j_path)
    W_U = load_wu(args.model_path)
    print(f"Loaded J-lens ({len(J_matrices)} layers) and W_U ({W_U.shape})", flush=True)

    # Load model
    print(f"Loading model from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        output_hidden_states=True,
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"  Model on {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

    digit_tokens = build_digit_tokens(tokenizer)

    n_samples = args.smoke_test if args.smoke_test > 0 else args.n_samples

    # Run sweep
    sweep_results = sweep_intervention(
        model, tokenizer, J_matrices, W_U, digit_tokens, device,
        n_samples=n_samples,
    )

    # Save results
    json_path = OUTPUT_DIR / "intervention_sweep.json"
    # Convert numpy types for JSON
    clean_results = {}
    for k, v in sweep_results.items():
        clean_results[k] = {
            kk: (float(vv) if isinstance(vv, (np.floating,)) else vv)
            for kk, vv in v.items()
            if kk != "layers"
        }
        clean_results[k]["layers"] = list(v["layers"])
    with open(json_path, "w") as f:
        json.dump(clean_results, f, indent=2)
    print(f"\nSaved {json_path}")

    # Plot
    plot_sweep(clean_results, FIG_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
