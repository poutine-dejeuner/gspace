"""
Layerwise G-space vs token-response activation plot.

For each layer, compute:
1. G-space R²: crosscoder features → answer (linear regression)
2. Token-response cosine: mean cosine between residual activation
   and J-lens vectors of ALL answer tokens (0-117)
3. J-space first-token rank (existing)
4. J-space full-answer rank (existing)

Plot them all on one figure.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.linear_model import LinearRegression
from safetensors.torch import load_file as safe_load
from transformers import AutoTokenizer


MODULUS = 59
D_MODEL = 5120
LAYERS = list(range(48, 64))  # 48-63


def load_jlens(path):
    """Load pre-computed Jacobian lens."""
    print(f"Loading J-lens from {path}")
    d = torch.load(path, map_location="cpu", weights_only=True)
    J_dict = d["J"]  # {layer_idx: [d_model, d_model]}
    n_prompts = d.get("n_prompts", "?")
    J_matrices = {}
    for layer_name, J in J_dict.items():
        l = int(layer_name)
        J_matrices[l] = J.to(torch.float32).numpy()
    print(f"  {len(J_matrices)} layers: {sorted(J_matrices.keys())[:5]}... (fitted on {n_prompts} prompts)")
    return J_matrices


def load_wu(model_path):
    """Load W_U from safetensors model files."""
    print(f"Loading W_U from {model_path}")
    for fname in sorted(os.listdir(model_path)):
        if fname.endswith(".safetensors"):
            fp = os.path.join(model_path, fname)
            tensors = safe_load(fp)
            if "lm_head.weight" in tensors:
                wu = tensors["lm_head.weight"].to(torch.float32).numpy()
                print(f"  W_U in: {fname}, shape={wu.shape}")
                return wu
    raise FileNotFoundError("lm_head.weight not found")


def load_activations(path):
    """Load cached activations."""
    print(f"Loading activations from {path}")
    data = np.load(path, allow_pickle=True)
    acts = {}
    for key in data.files:
        if key.startswith("train_l") and not key.startswith("train_labels"):
            l = int(key.split("train_l")[1])
            acts[l] = data[key]
    labels = data["train_labels"]
    print(f"  {len(acts)} layers, {len(labels)} samples")
    return acts, labels


def load_crosscoder(path):
    print(f"Loading crosscoder from {path}")
    from safetensors.torch import load_file
    with open(os.path.join(path, "config.json")) as f:
        config = json.load(f)
    n_feat = config.get("n_features", config.get("expansion_factor", 4096))
    n_layers_cc = config.get("n_layers", 16)

    ckpt_path = os.path.join(path, "checkpoints", "crosscoder_step0024413.safetensors")
    ckpt = load_file(ckpt_path)
    W_enc = ckpt["W_enc"].to(torch.float32).numpy()
    W_dec = ckpt["W_dec"].to(torch.float32).numpy()
    b_enc = ckpt.get("b_enc", torch.zeros(n_feat)).to(torch.float32).numpy()

    print(f"  {n_feat} features, {n_layers_cc} layers")
    return n_feat, n_layers_cc, W_enc, W_dec, b_enc


def get_answer_token_ids(tokenizer):
    """Map answer (0..117) -> token ID."""
    answer_ids = {}
    for i in range(MODULUS * 2):
        toks = tokenizer.encode(str(i), add_special_tokens=False)
        if len(toks) == 1:
            answer_ids[i] = toks[0]
    print(f"  {len(answer_ids)} single-token answers found")
    return answer_ids


def compute_jlens_vectors(J_matrices, W_U, l):
    """Compute J-lens vectors for layer l: W_U @ J_ℓ"""
    import numpy as np
    J = J_matrices[l]
    return (W_U.astype(np.float64) @ J.astype(np.float64)).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jlens_path", default="/data/jlens/qwen3.5-27b/jlens/Salesforce-wikitext/Qwen3.5-27B_jacobian_lens.pt")
    parser.add_argument("--model_id", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--activations", default="results/phase1_q35/activations.npz")
    parser.add_argument("--crosscoder", default="results/crosscoder")
    parser.add_argument("--output_dir", default="results/layerwise_plot")
    args = parser.parse_args()

    OUTPUT_DIR = Path(args.output_dir)
    FIG_DIR = Path("figures") / OUTPUT_DIR.name
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load everything
    J_matrices = load_jlens(args.jlens_path)
    W_U = load_wu(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3.5-27B/snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654"))
    acts_dict, labels = load_activations(args.activations)
    n_feat, n_layers_cc, W_enc, W_dec, b_enc = load_crosscoder(args.crosscoder)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    answer_token_ids = get_answer_token_ids(tokenizer)

    n = len(labels)
    K = 32

    # Precompute J-lens vectors for all answer tokens (0-117, all sub-tokens)
    # and the full answer token set for ranking
    answer_full_tokens = {}
    all_answer_sub_tokens = set()
    for ans in range(MODULUS * 2):
        toks = tokenizer.encode(str(ans), add_special_tokens=False)
        answer_full_tokens[ans] = toks
        for t in toks:
            all_answer_sub_tokens.add(t)
    # Also include the single-token answer ids
    for tok_id in answer_token_ids.values():
        all_answer_sub_tokens.add(tok_id)
    all_answer_sub_tokens = sorted(all_answer_sub_tokens)
    print(f"  {len(all_answer_sub_tokens)} unique sub-tokens in answers 0-{MODULUS*2-1}")
    print(f"  Sample: {[tokenizer.decode([t]) for t in all_answer_sub_tokens[:15]]}")

    # Metrics per layer
    gspace_r2 = {}
    token_cos_mean = {}
    token_cos_ans_mean = {}  # cosine with ONLY the correct answer's sub-tokens
    jrank_first = {}
    jrank_full = {}

    for l in LAYERS:
        if l not in acts_dict or l not in J_matrices:
            continue

        acts = acts_dict[l].astype(np.float32)

        # ── G-space R² ──
        l_idx = l - LAYERS[0]
        enc_slice = W_enc[l_idx * D_MODEL:(l_idx + 1) * D_MODEL, :]
        pre_acts = acts @ enc_slice + b_enc
        # TopK
        topk_idx = np.argpartition(-pre_acts, K, axis=1)[:, :K]
        features = np.zeros_like(pre_acts)
        for i in range(n):
            features[i, topk_idx[i]] = pre_acts[i, topk_idx[i]]

        reg = LinearRegression()
        reg.fit(features, labels)
        r2 = reg.score(features, labels)
        gspace_r2[l] = float(r2)

        # ── Token-response cosine ──
        # Compute J-lens for all answer sub-tokens
        j_full = compute_jlens_vectors(J_matrices, W_U, l).astype(np.float32)
        j_answer = j_full[all_answer_sub_tokens]  # [n_tokens, d]

        # Normalize
        j_norms = np.linalg.norm(j_answer, axis=1, keepdims=True)
        j_normed = j_answer / np.where(j_norms > 1e-8, j_norms, 1.0)
        act_norms = np.linalg.norm(acts, axis=1, keepdims=True)
        act_normed = acts / np.where(act_norms > 1e-8, act_norms, 1.0)

        # Cosine of each activation with each answer token → take max
        all_cos = np.abs(act_normed @ j_normed.T)  # [n, n_tokens]
        token_cos_mean[l] = float(np.mean(np.max(all_cos, axis=1)))

        # ── Cosine with correct answer's tokens only ──
        ans_cosines = []
        for i in range(n):
            ans = int(labels[i])
            sub_toks = answer_full_tokens[ans]
            indices = [all_answer_sub_tokens.index(t) for t in sub_toks if t in all_answer_sub_tokens]
            if indices:
                cos_val = np.max(all_cos[i, indices])
                ans_cosines.append(cos_val)
        token_cos_ans_mean[l] = float(np.mean(ans_cosines)) if ans_cosines else float("nan")

        # ── J-space first-token rank ──
        # Filter to single-token answers only
        single_tok_answers = {i for i in range(MODULUS * 2) if i in answer_token_ids}
        single_indices = sorted(set(answer_token_ids[i] for i in single_tok_answers))
        j_single = j_full[single_indices]
        scores_single = acts @ j_single.T
        ans_to_idx_single = {tok: idx for idx, tok in enumerate(single_indices)}

        ranks_first = []
        for i in range(n):
            ans = int(labels[i])
            if ans not in single_tok_answers:
                continue
            tok = answer_token_ids[ans]
            idx = ans_to_idx_single[tok]
            score = scores_single[i, idx]
            rank = int((scores_single[i] > score).sum() + 1)
            ranks_first.append(rank)
        jrank_first[l] = float(np.mean(ranks_first)) if ranks_first else float("nan")

        # ── J-space full-answer rank ──
        scores_all = acts @ j_answer.T
        tok_to_idx = {tok: idx for idx, tok in enumerate(all_answer_sub_tokens)}

        ranks_full = []
        for i in range(n):
            ans = int(labels[i])
            sub_toks = answer_full_tokens[ans]
            max_rank = 0
            for st in sub_toks:
                if st not in tok_to_idx:
                    max_rank = max(max_rank, len(all_answer_sub_tokens))
                    continue
                idx = tok_to_idx[st]
                score = scores_all[i, idx]
                rank = int((scores_all[i] > score).sum() + 1)
                max_rank = max(max_rank, rank)
            ranks_full.append(max_rank)
        jrank_full[l] = float(np.mean(ranks_full)) if ranks_full else float("nan")

        print(f"  Layer {l:2d}: G-R²={gspace_r2[l]:.3f}, cos_all={token_cos_mean[l]:.4f}, "
              f"cos_ans={token_cos_ans_mean[l]:.4f}, "
              f"rank_1st={jrank_first[l]:.1f}, rank_full={jrank_full[l]:.1f}")

    # ═════════════════════════════════════════════════════════════════
    # PLOT: 4 panels
    # ═════════════════════════════════════════════════════════════════

    valid_layers = sorted(gspace_r2.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: G-space R²
    ax = axes[0, 0]
    r2_vals = [gspace_r2[l] for l in valid_layers]
    peak_g = max(valid_layers, key=lambda l: gspace_r2[l])
    ax.plot(valid_layers, r2_vals, "o-", color="tab:blue", markersize=6, linewidth=2)
    ax.axvline(peak_g, color="tab:blue", linestyle="--", alpha=0.5,
               label=f"Peak L{peak_g} (R²={gspace_r2[peak_g]:.3f})")
    ax.set_xlabel("Layer")
    ax.set_ylabel("R²")
    ax.set_title("G-space: Crosscoder features → answer")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Token-response cosine (all answers vs correct answer)
    ax = axes[0, 1]
    cos_all = [token_cos_mean[l] for l in valid_layers]
    cos_ans = [token_cos_ans_mean[l] for l in valid_layers]
    ax.plot(valid_layers, cos_all, "o-", color="tab:green", markersize=6, linewidth=2,
            label="Max cosine with ANY answer token")
    ax.plot(valid_layers, cos_ans, "s--", color="tab:olive", markersize=5, linewidth=1.5,
            label="Cosine with CORRECT answer tokens")
    ax.set_xlabel("Layer")
    ax.set_ylabel("|cosine similarity|")
    ax.set_title("Token-response alignment: activation · J-lens(answer tokens)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: J-space rank (first token vs full answer)
    ax = axes[1, 0]
    rank1 = [jrank_first[l] for l in valid_layers]
    rank_full = [jrank_full[l] for l in valid_layers]
    peak_j = min(valid_layers, key=lambda l: jrank_first[l])
    ax.plot(valid_layers, rank1, "o-", color="tab:red", markersize=6, linewidth=2,
            label="First token rank")
    ax.plot(valid_layers, rank_full, "s--", color="tab:orange", markersize=5, linewidth=1.5,
            label="Full answer rank (max over sub-tokens)")
    ax.axvline(peak_j, color="tab:red", linestyle="--", alpha=0.5,
               label=f"Peak L{peak_j}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean rank (lower = better)")
    ax.set_title("J-space: Answer token rank via J-lens")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # Panel 4: Combined normalized view
    ax = axes[1, 1]

    def normalize(vals):
        v = np.array(vals)
        return (v - np.min(v)) / (np.max(v) - np.min(v) + 1e-8)

    r2_norm = normalize(r2_vals)
    cos_norm = normalize(cos_ans)
    rank_inv = 1.0 / np.array(rank_full)
    rank_norm = normalize(rank_inv)

    ax.plot(valid_layers, r2_norm, "o-", color="tab:blue", markersize=5, linewidth=2,
            label="G-space R² (norm)")
    ax.plot(valid_layers, cos_norm, "s-", color="tab:green", markersize=5, linewidth=2,
            label="Token cos (correct ans, norm)")
    ax.plot(valid_layers, rank_norm, "D-", color="tab:red", markersize=5, linewidth=2,
            label="J-space 1/rank (full, norm)")

    ax.axvline(peak_g, color="tab:blue", linestyle="--", alpha=0.4)
    ax.axvline(peak_j, color="tab:red", linestyle="--", alpha=0.4)

    offset = peak_j - peak_g
    ax.annotate(f"Δ = {offset:+d} layers",
                xy=((peak_g + peak_j) / 2, 0.5),
                fontsize=11, ha="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))
    ax.set_xlabel("Layer")
    ax.set_ylabel("Normalized score")
    ax.set_title(f"Combined: G-space, token cosine, J-space rank (Δ = {offset:+d})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"Layerwise G-space vs Token-response Activation\n"
                 f"Qwen3.5-27B, mod {MODULUS} addition, {n} samples",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    plot_path = FIG_DIR / "layerwise_gspace_vs_tokens.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved {plot_path}")

    # Save data
    data = {
        "layers": valid_layers,
        "gspace_r2": gspace_r2,
        "token_cos_all_mean": token_cos_mean,
        "token_cos_ans_mean": token_cos_ans_mean,
        "jrank_first_mean": jrank_first,
        "jrank_full_mean": jrank_full,
    }
    data_path = OUTPUT_DIR / "layerwise_data.json"
    with open(data_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {data_path}")


if __name__ == "__main__":
    main()
