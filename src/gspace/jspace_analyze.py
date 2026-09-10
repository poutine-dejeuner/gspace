"""
J-space analysis using pre-computed Jacobian lens from Neuronpedia for Qwen3.5-27B.

Memory-efficient: computes W_U @ J_ℓ per layer on the fly, processes each layer
independently, never keeps all [vocab, d_model] matrices in RAM simultaneously.

1. G-space (crosscoder) → logit answer prediction per layer (R² curve)
2. J-space answer rank per layer (rank curve)
3. Direct g-space vs J-space comparison at peak layers:
   - Cosine alignment between crosscoder decoder directions and J-lens vectors
   - Gradient pursuit decomposition (k=25)

Usage:
    python -m gspace.jspace_analyze \
        --jlens_path /data/jlens/qwen3.5-27b/jlens/Salesforce-wikitext/Qwen3.5-27B_jacobian_lens.pt \
        --model_id Qwen/Qwen3.5-27B \
        --activations results/phase1_q35/activations.npz \
        --crosscoder results/crosscoder \
        --output_dir results/jspace_analysis \
        --k_sparse 25
"""

import argparse
import json
import math
import os
import re
import gc
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import safetensors.torch as st

parser = argparse.ArgumentParser()
parser.add_argument("--jlens_path", type=str,
    default="/data/jlens/qwen3.5-27b/jlens/Salesforce-wikitext/Qwen3.5-27B_jacobian_lens.pt")
parser.add_argument("--model_id", type=str, default="Qwen/Qwen3.5-27B")
parser.add_argument("--activations", type=str, default="results/phase1_q35/activations.npz")
parser.add_argument("--crosscoder", type=str, default="results/crosscoder")
parser.add_argument("--output_dir", type=str, default="results/jspace_analysis")
parser.add_argument("--k_sparse", type=int, default=25)
parser.add_argument("--smoke_test", type=int, default=0,
    help="If > 0, use only N samples for quick validation")
args = parser.parse_args()

OUTPUT_DIR = Path(args.output_dir)
FIG_DIR = Path("figures") / OUTPUT_DIR.name
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODULUS = 59
D_MODEL = 5120
LAYERS = list(range(48, 64))  # analysis layers
N_LAYERS = len(LAYERS)

# ─────────────────────────────────────────────────────────────────────
# Load pre-computed J-lens (keep J_matrices in dict for lazy access)
# ─────────────────────────────────────────────────────────────────────

print("=" * 60)
print("Loading pre-computed Jacobian lens")
print("=" * 60)

jlens_data = torch.load(args.jlens_path, map_location="cpu")
J_matrices = jlens_data["J"]  # dict: int layer -> [5120, 5120] float16
jlens_layers = sorted(J_matrices.keys())
print(f"  J-lens has {len(jlens_layers)} layers: {jlens_layers[0]}-{jlens_layers[-1]}")
print(f"  n_prompts used for fitting: {jlens_data['n_prompts']}")

# ─────────────────────────────────────────────────────────────────────
# Load W_U from safetensors (lm_head only, no GPU needed)
# ─────────────────────────────────────────────────────────────────────

print("=" * 60)
print("Loading W_U from safetensors")
print("=" * 60)

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(args.model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

index_path = hf_hub_download(args.model_id, "model.safetensors.index.json")
with open(index_path) as f:
    index = json.load(f)
lm_head_file = index["weight_map"]["lm_head.weight"]
print(f"  lm_head.weight in: {lm_head_file}")

lm_head_path = hf_hub_download(args.model_id, lm_head_file)
with st.safe_open(lm_head_path, framework="pt", device="cpu") as f_sf:
    W_U = f_sf.get_tensor("lm_head.weight").float()  # [vocab, 5120]
vocab_size = W_U.shape[0]
print(f"  W_U: {W_U.shape}, vocab: {vocab_size}")

# Pre-compute answer token IDs
answer_token_ids = {}
for ans in range(MODULUS * 2):
    tok = tokenizer.encode(str(ans), add_special_tokens=False)
    if len(tok) == 1:
        answer_token_ids[ans] = tok[0]
n_answer_tokens = len(answer_token_ids)
print(f"  {n_answer_tokens} single-token answers found")

# ─────────────────────────────────────────────────────────────────────
# J-lens helper: compute W_U @ J_ℓ per layer, one at a time
# ─────────────────────────────────────────────────────────────────────

_JLENS_CACHE = {}

def compute_jlens_vectors(l):
    """Return J-lens vectors [vocab, d_model] for layer l (cached once)."""
    if l not in _JLENS_CACHE:
        J_l = J_matrices[l].float()
        _JLENS_CACHE[l] = (W_U @ J_l).numpy()  # [vocab, d_model]
    return _JLENS_CACHE[l]

def free_jlens_cache():
    _JLENS_CACHE.clear()
    gc.collect()

# ─────────────────────────────────────────────────────────────────────
# Load cached activations from phase1_q35
# ─────────────────────────────────────────────────────────────────────

print("=" * 60)
print("Loading cached activations")
print("=" * 60)

act_data = np.load(args.activations)
train_labels = act_data["train_labels"]
n_total = len(train_labels)

if args.smoke_test > 0:
    n = min(args.smoke_test, n_total)
    print(f"  [SMOKE TEST] Using {n} samples")
    train_labels = train_labels[:n]
else:
    n = n_total

train_acts = {}
for key in sorted(act_data.keys()):
    m = re.match(r'^train_l(\d+)$', key)
    if m:
        l = int(m.group(1))
        if l in LAYERS and l in J_matrices:
            train_acts[l] = act_data[key][:n]
print(f"  Loaded activations for {len(train_acts)} layers, {n} samples")

# ─────────────────────────────────────────────────────────────────────
# Load crosscoder
# ─────────────────────────────────────────────────────────────────────

print("=" * 60)
print("Loading crosscoder")
print("=" * 60)

CKPT_PATH = Path(args.crosscoder) / "checkpoints" / "crosscoder_step0024413.safetensors"
sd = st.load_file(str(CKPT_PATH))
W_enc = sd["W_enc"].numpy()  # [L*D, N_feat]
b_enc = sd["b_enc"].numpy()
W_dec = sd["W_dec"].numpy()  # [N_feat, L*D]
N_FEAT = W_dec.shape[0]
print(f"  {N_FEAT} features, {N_LAYERS} layers in crosscoder")

# ═════════════════════════════════════════════════════════════════════
# ANALYSIS 1: G-space prediction of answer per layer
# ═════════════════════════════════════════════════════════════════════

print("=" * 60)
print("ANALYSIS 1: G-space prediction per layer")
print("=" * 60)

from sklearn.linear_model import LinearRegression

gspace_r2 = {}
gspace_mse = {}

for l_idx, l in enumerate(LAYERS):
    if l not in train_acts or l not in J_matrices:
        continue

    acts = train_acts[l].astype(np.float32)  # [n, d]

    # Encode via crosscoder
    enc_slice = W_enc[l_idx * D_MODEL:(l_idx + 1) * D_MODEL, :]
    pre_acts = acts @ enc_slice + b_enc  # [n, N_feat]

    # TopK sparsity (K=32)
    K = 32
    # Use argpartition to get top-K indices efficiently
    topk_idx = np.argpartition(-pre_acts, K, axis=1)[:, :K]
    topk_vals = np.take_along_axis(pre_acts, topk_idx, axis=1)
    features = np.zeros_like(pre_acts)
    for i in range(n):
        features[i, topk_idx[i]] = topk_vals[i]

    reg = LinearRegression()
    reg.fit(features, train_labels)
    r2 = reg.score(features, train_labels)
    preds = reg.predict(features)
    mse = np.mean((preds - train_labels) ** 2)

    gspace_r2[l] = float(r2)
    gspace_mse[l] = float(mse)
    print(f"  Layer {l:2d}: R²={r2:.4f}, MSE={mse:.1f}")

if gspace_r2:
    gspace_peak = max(gspace_r2, key=lambda l: gspace_r2[l])
    print(f"\n  Peak G-space: layer {gspace_peak} (R²={gspace_r2[gspace_peak]:.4f})")

# ═════════════════════════════════════════════════════════════════════
# ANALYSIS 2: J-space answer rank per layer
# (compute J-lens vectors one layer at a time to save RAM)
# ═════════════════════════════════════════════════════════════════════

print("=" * 60)
print("ANALYSIS 2: J-space answer rank per layer")
print("=" * 60)

jspace_ranks = {}

for l in LAYERS:
    if l not in train_acts or l not in J_matrices:
        continue

    # Compute J-lens vectors for this layer (cached)
    # BUT we only need answer token vectors, not all 248K.
    # Extract only the answer-eligible rows.
    j_vecs_full = compute_jlens_vectors(l).astype(np.float32)  # [vocab, d]
    answer_indices = sorted(set(
        answer_token_ids[i] for i in range(MODULUS * 2) if i in answer_token_ids
    ))
    j_vecs = j_vecs_full[answer_indices]  # [n_answers, d]
    acts = train_acts[l].astype(np.float32)  # [n, d]

    # Score: s_i[idx] = j_vecs[idx] @ acts[i]
    # We only need rank of correct answer among answer tokens
    scores = acts @ j_vecs.T  # [n, n_answers] — fast, n_answers ~ 118
    n_answers = len(answer_indices)
    ans_to_idx = {tok: i for i, tok in enumerate(answer_indices)}

    ranks = []
    top1_count = 0
    n_valid = 0

    for i in range(n):
        ans = int(train_labels[i])
        ans_tok = answer_token_ids.get(ans)
        if ans_tok is None or ans_tok not in ans_to_idx:
            continue
        ans_idx = ans_to_idx[ans_tok]
        ans_score = scores[i, ans_idx]
        rank = int((scores[i] > ans_score).sum() + 1)
        ranks.append(rank)
        if rank == 1:
            top1_count += 1
        n_valid += 1

    if ranks:
        mean_rank = float(np.mean(ranks))
        median_rank = float(np.median(ranks))
        top1_pct = top1_count / len(ranks) * 100
    else:
        mean_rank = float("nan")
        median_rank = float("nan")
        top1_pct = float("nan")

    jspace_ranks[l] = {
        "mean_rank": mean_rank,
        "median_rank": median_rank,
        "top1_pct": top1_pct,
        "n_valid": len(ranks),
    }
    print(f"  Layer {l:2d}: mean_rank={mean_rank:.1f}, median_rank={median_rank:.1f}, "
          f"top1={top1_pct:.1f}% (n={len(ranks)})")

    # Free this layer's J-lens vectors from cache after use
    # (keep in _JLENS_CACHE for potential reuse in analysis 3)
    # Actually, keep them — analysis 3 needs them. But we'll free selectively.

valid_j_layers = [l for l in LAYERS if l in jspace_ranks and not np.isnan(jspace_ranks[l]["mean_rank"])]
if valid_j_layers:
    jspace_peak = min(valid_j_layers, key=lambda l: jspace_ranks[l]["mean_rank"])
    print(f"\n  Peak J-space: layer {jspace_peak} (mean_rank={jspace_ranks[jspace_peak]['mean_rank']:.1f})")
    for k in [1, 5, 10, 25, 50]:
        count_k = sum(1 for l in valid_j_layers if jspace_ranks[l]["mean_rank"] <= k)
        print(f"  Layers with mean_rank ≤ {k}: {count_k}")

# ═════════════════════════════════════════════════════════════════════
# ANALYSIS 2b: J-space rank for FULL answer (all sub-tokens)
# ═════════════════════════════════════════════════════════════════════

print("=" * 60)
print("ANALYSIS 2b: J-space rank for FULL answer (all sub-tokens)")
print("=" * 60)

# For each answer, tokenize fully and compute product of ranks
jspace_full_ranks = {}

for l in LAYERS:
    if l not in train_acts or l not in J_matrices:
        continue

    j_vecs_full = compute_jlens_vectors(l).astype(np.float32)  # [vocab, d]
    answer_indices = sorted(set(
        answer_token_ids[i] for i in range(MODULUS * 2) if i in answer_token_ids
    ))
    j_vecs = j_vecs_full[answer_indices]  # [n_answers, d]
    acts = train_acts[l].astype(np.float32)  # [n, d]

    scores = acts @ j_vecs.T  # [n, n_answers]
    ans_to_idx = {tok: i for i, tok in enumerate(answer_indices)}

    # Also precompute tokenization for all answers 0-117
    answer_full_tokens = {}
    for ans in range(MODULUS * 2):
        toks = tokenizer.encode(str(ans), add_special_tokens=False)
        answer_full_tokens[ans] = toks

    ranks = []
    top1_count = 0
    n_valid = 0

    for i in range(n):
        ans = int(train_labels[i])
        sub_tokens = answer_full_tokens.get(ans, [])
        if not sub_tokens:
            continue
        # For each sub-token, compute its rank among answer tokens
        # Use max rank across sub-tokens (all sub-tokens must be ranked well)
        max_rank = 0
        all_top1 = True
        for st in sub_tokens:
            if st not in ans_to_idx:
                all_top1 = False
                max_rank = max(max_rank, len(answer_indices))
                continue
            st_idx = ans_to_idx[st]
            st_score = scores[i, st_idx]
            st_rank = int((scores[i] > st_score).sum() + 1)
            max_rank = max(max_rank, st_rank)
            if st_rank > 1:
                all_top1 = False
        ranks.append(max_rank)
        if all_top1:
            top1_count += 1
        n_valid += 1

    if ranks:
        mean_rank = float(np.mean(ranks))
        median_rank = float(np.median(ranks))
        top1_pct = top1_count / len(ranks) * 100
    else:
        mean_rank = float("nan")
        median_rank = float("nan")
        top1_pct = float("nan")

    jspace_full_ranks[l] = {
        "mean_rank": mean_rank,
        "median_rank": median_rank,
        "top1_pct": top1_pct,
        "n_valid": len(ranks),
    }
    print(f"  Layer {l:2d}: mean_rank={mean_rank:.1f}, median_rank={median_rank:.1f}, "
          f"top1={top1_pct:.1f}% (n={len(ranks)})")

if jspace_full_ranks:
    valid_full = [l for l in LAYERS if l in jspace_full_ranks and not np.isnan(jspace_full_ranks[l]["mean_rank"])]
    if valid_full:
        jspace_full_peak = min(valid_full, key=lambda l: jspace_full_ranks[l]["mean_rank"])
        print(f"\n  Peak J-space (full): layer {jspace_full_peak} "
              f"(mean_rank={jspace_full_ranks[jspace_full_peak]['mean_rank']:.1f})")

# ═════════════════════════════════════════════════════════════════════
# ANALYSIS 3: Direct g-space vs J-space comparison
# ═════════════════════════════════════════════════════════════════════

print("=" * 60)
print("ANALYSIS 3: G-space vs J-space direct comparison")
print("=" * 60)

# 3a: Per-layer cosine alignment between CC decoder directions and J-lens vectors (answer tokens only)
print("\n3a: Layer-wise alignment CC decoder ↔ J-lens answer vectors")

alignments_per_layer = {}

for l_idx, l in enumerate(LAYERS):
    if l not in J_matrices:
        continue

    j_vecs = compute_jlens_vectors(l).astype(np.float32)  # [vocab, d]

    # Only use answer tokens for alignment computation
    answer_indices = [answer_token_ids[i] for i in range(MODULUS * 2)
                      if i in answer_token_ids]
    j_answer = j_vecs[answer_indices].astype(np.float32)  # [n_answers, d]

    # Normalize
    j_norms = np.linalg.norm(j_answer, axis=1, keepdims=True)
    j_normed = j_answer / np.where(j_norms > 1e-8, j_norms, 1.0)

    # CC decoder slice for this layer
    dec_slice = W_dec[:, l_idx * D_MODEL:(l_idx + 1) * D_MODEL]  # [N_feat, d]
    dec_norms = np.linalg.norm(dec_slice, axis=1, keepdims=True)
    dec_normed = dec_slice / np.where(dec_norms > 1e-8, dec_norms, 1.0)

    # For each CC feature, max alignment with any answer J-lens vector
    align_cc_to_j = np.max(np.abs(j_normed @ dec_normed.T), axis=0)  # [N_feat]
    # For each answer J-lens vector, max alignment with any CC feature
    align_j_to_cc = np.max(np.abs(dec_normed @ j_normed.T), axis=0)  # [n_answers]

    mean_cc = float(np.mean(align_cc_to_j))
    max_cc = float(np.max(align_cc_to_j))
    mean_token = float(np.mean(align_j_to_cc))
    max_token = float(np.max(align_j_to_cc))

    alignments_per_layer[l] = {
        "mean_cc_to_j": mean_cc,
        "max_cc_to_j": max_cc,
        "mean_answer_to_cc": mean_token,
        "max_answer_to_cc": max_token,
    }
    print(f"  Layer {l:2d}: CC→J mean={mean_cc:.4f} max={max_cc:.4f}, "
          f"answer→CC mean={mean_token:.4f} max={max_token:.4f}")

# Free J-lens cache — no longer needed
free_jlens_cache()

# 3b: Sparse decomposition at peak J-space layer
print(f"\n3b: Sparse decomposition at peak J-space layer ({jspace_peak})")

try:
    from scipy.optimize import nnls

    l_peak = jspace_peak
    # Recompute J-lens vectors for peak layer (they were freed)
    j_vecs_peak = compute_jlens_vectors(l_peak).astype(np.float32)
    acts_peak = train_acts[l_peak].astype(np.float32)

    # Normalize peak J-lens vectors
    # Use only numeric tokens (digits 0-9 and common multi-digit numbers)
    # + all tokens from the answer set for the task (0 through 117)
    numeric_token_ids = set()
    # Single digits 0-9
    for d in range(10):
        t = tokenizer.encode(str(d), add_special_tokens=False)
        if len(t) == 1:
            numeric_token_ids.add(t[0])
    # All answers 0-117 (even if multi-token, include all sub-tokens)
    for ans in range(MODULUS * 2):
        t = tokenizer.encode(str(ans), add_special_tokens=False)
        for tid in t:
            numeric_token_ids.add(tid)
    # Also include answer token ids from answer_token_ids
    for tok_id in answer_token_ids.values():
        numeric_token_ids.add(tok_id)
    
    numeric_indices = sorted(numeric_token_ids)
    j_sub = j_vecs_peak[numeric_indices].astype(np.float64)  # [n_num, d]
    j_sub_norms = np.linalg.norm(j_sub, axis=1, keepdims=True)
    j_normed = j_sub / np.where(j_sub_norms > 1e-8, j_sub_norms, 1.0)
    print(f"  Using {len(numeric_indices)} numeric/answer J-lens vectors for sparse decomposition")

    n_decomp = min(100, n)
    k = args.k_sparse

    def gradient_pursuit(h, V, k, n_iter=200):
        """Sparse nonnegative decomposition: h ≈ V^T @ alpha, ||alpha||_0 ≤ k, alpha ≥ 0."""
        n_vocab = V.shape[0]
        alpha = np.zeros(n_vocab)
        residual = h.copy().astype(np.float64)
        support = set()

        for _ in range(n_iter):
            if len(support) >= k:
                break
            corrs = V.astype(np.float64) @ residual
            for s in support:
                corrs[s] = -np.inf
            best = int(np.argmax(corrs))
            if corrs[best] <= 0:
                break
            support.add(best)
            active = sorted(support)
            V_active = V[active].astype(np.float64)
            alpha_active, _ = nnls(V_active.T, h.astype(np.float64))
            alpha[active] = alpha_active
            residual = h.astype(np.float64) - V_active.T @ alpha_active

        return alpha, residual

    token_usage = np.zeros(len(numeric_indices))
    jspace_variance_fraction = []

    for i in range(n_decomp):
        h = acts_peak[i].astype(np.float64)
        alpha, resid = gradient_pursuit(h, j_normed, k)
        token_usage += (alpha > 0).astype(float)

        j_comp = j_normed.T @ alpha
        var_total = np.var(h)
        var_resid = np.var(resid)
        if var_total > 1e-8:
            jspace_variance_fraction.append(1 - var_resid / var_total)

    # Map back to global token IDs
    top_tokens_global = [(int(numeric_indices[i]), token_usage[i] / n_decomp)
                         for i in np.argsort(token_usage)[-30:][::-1]]

    print(f"  J-space variance fraction: mean={np.mean(jspace_variance_fraction):.3f}, "
          f"std={np.std(jspace_variance_fraction):.3f}")
    print(f"  Top 20 tokens in sparse decomposition:")
    for tok_id, freq in top_tokens_global[:20]:
        tok_str = tokenizer.decode([tok_id])
        print(f"    {tok_str:20s} (id={tok_id:5d}): {freq:.3f}")

    # 3c: CC features projected onto J-space subspace
    l_idx_peak = LAYERS.index(l_peak)
    dec_slice_peak = W_dec[:, l_idx_peak * D_MODEL:(l_idx_peak + 1) * D_MODEL]
    dec_norms_peak = np.linalg.norm(dec_slice_peak, axis=1, keepdims=True)
    dec_normed_peak = dec_slice_peak / np.where(dec_norms_peak > 1e-8, dec_norms_peak, 1.0)

    k_subspace = min(100, len(top_tokens_global))
    top_j_indices = sorted([idx for idx, _ in top_tokens_global[:k_subspace]])
    # Map global token indices to positions in j_normed
    idx_map = {gid: pos for pos, gid in enumerate(numeric_indices)}
    V_subspace = j_normed[[idx_map[gid] for gid in top_j_indices]].astype(np.float64)

    cc_in_subspace = []
    for f in range(N_FEAT):
        d = dec_normed_peak[f].astype(np.float64)
        try:
            coeffs = np.linalg.lstsq(V_subspace.T, d, rcond=None)[0]
            proj = V_subspace.T @ coeffs
            frac = np.linalg.norm(proj) / (np.linalg.norm(d) + 1e-8)
        except np.linalg.LinAlgError:
            frac = 0.0
        cc_in_subspace.append(frac)

    print(f"\n  CC features in top-{k_subspace} J-space: mean={np.mean(cc_in_subspace):.4f}, "
          f"max={np.max(cc_in_subspace):.4f}")

    sparse_results = {
        "jspace_variance_fraction_mean": float(np.mean(jspace_variance_fraction)),
        "jspace_variance_fraction_std": float(np.std(jspace_variance_fraction)),
        "top_tokens": [(int(idx), float(freq)) for idx, freq in top_tokens_global[:30]],
        "cc_in_j_subspace_mean": float(np.mean(cc_in_subspace)),
        "cc_in_j_subspace_max": float(np.max(cc_in_subspace)),
    }

except ImportError:
    print("  scipy not available, skipping sparse decomposition")
    sparse_results = {"error": "scipy not available"}

# ═════════════════════════════════════════════════════════════════════
# PLOTS
# ═════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Generating plots")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: G-space R² per layer
ax1 = axes[0]
layers_g = sorted(gspace_r2.keys())
r2_vals = [gspace_r2[l] for l in layers_g]
ax1.plot(layers_g, r2_vals, "o-", color="tab:blue", markersize=5, linewidth=2)
ax1.axvline(gspace_peak, color="tab:blue", linestyle="--", alpha=0.5,
            label=f"Peak: L{gspace_peak} (R²={gspace_r2[gspace_peak]:.3f})")
ax1.set_xlabel("Layer")
ax1.set_ylabel("R² (features → answer)")
ax1.set_title("G-space: Crosscoder features → answer")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Plot 2: J-space answer rank per layer
ax2 = axes[1]
layers_j = sorted(jspace_ranks.keys())
mean_ranks = [jspace_ranks[l]["mean_rank"] for l in layers_j]
ax2.plot(layers_j, mean_ranks, "o-", color="tab:red", markersize=5, linewidth=2)
ax2.axvline(jspace_peak, color="tab:red", linestyle="--", alpha=0.5,
            label=f"Peak: L{jspace_peak} (rank={jspace_ranks[jspace_peak]['mean_rank']:.1f})")
ax2.set_xlabel("Layer")
ax2.set_ylabel("Mean rank of correct answer")
ax2.set_title("J-space: Answer rank via J-lens")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_yscale("log")

# Plot 3: Alignment CC decoder ↔ J-lens vectors
ax3 = axes[2]
layers_a = sorted(alignments_per_layer.keys())
mean_align = [alignments_per_layer[l]["mean_cc_to_j"] for l in layers_a]
max_align = [alignments_per_layer[l]["max_cc_to_j"] for l in layers_a]
ax3.plot(layers_a, mean_align, "o-", color="tab:purple", markersize=5,
         linewidth=2, label="Mean CC→J alignment")
ax3.plot(layers_a, max_align, "s--", color="tab:orange", markersize=4,
         linewidth=1.5, label="Max CC→J alignment")
ax3.set_xlabel("Layer")
ax3.set_ylabel("|cosine similarity|")
ax3.set_title("G-space ↔ J-space alignment")
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = FIG_DIR / "jspace_analysis.png"
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"  Saved {plot_path}")

# Combined peak comparison
fig2, ax = plt.subplots(figsize=(10, 6))

r2_norm = (np.array(r2_vals) - np.min(r2_vals)) / (np.max(r2_vals) - np.min(r2_vals) + 1e-8)
rank_inv = 1.0 / np.array(mean_ranks)
rank_inv_norm = (rank_inv - np.min(rank_inv)) / (np.max(rank_inv) - np.min(rank_inv) + 1e-8)

ax.plot(layers_g, r2_norm, "o-", color="tab:blue", markersize=5, linewidth=2,
        label="G-space (R², normalized)")
ax.plot(layers_j, rank_inv_norm, "s-", color="tab:red", markersize=5, linewidth=2,
        label="J-space (1/rank, normalized)")

ax.axvline(gspace_peak, color="tab:blue", linestyle="--", alpha=0.5)
ax.axvline(jspace_peak, color="tab:red", linestyle="--", alpha=0.5)

peak_offset = jspace_peak - gspace_peak
ax.annotate(f"Δ = {peak_offset:+d} layers",
            xy=((gspace_peak + jspace_peak) / 2, 0.5),
            fontsize=12, ha="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))

ax.set_xlabel("Layer")
ax.set_ylabel("Normalized score")
ax.set_title(f"G-space vs J-space: peak comparison (Δ = {peak_offset:+d} layers)")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plot_path2 = FIG_DIR / "gspace_vs_jspace_peaks.png"
fig2.savefig(plot_path2, dpi=150, bbox_inches="tight")
print(f"  Saved {plot_path2}")

# ═════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ═════════════════════════════════════════════════════════════════════

summary = {
    "jlens_path": args.jlens_path,
    "n_prompts_jlens": jlens_data["n_prompts"],
    "layers_analyzed": LAYERS,
    "gspace_peak": int(gspace_peak),
    "gspace_peak_r2": gspace_r2[gspace_peak],
    "jspace_peak": int(jspace_peak),
    "jspace_peak_mean_rank": jspace_ranks[jspace_peak]["mean_rank"],
    "jspace_peak_top1_pct": jspace_ranks[jspace_peak]["top1_pct"],
    "peak_offset": int(jspace_peak - gspace_peak),
    "gspace_r2": gspace_r2,
    "gspace_mse": gspace_mse,
    "jspace_ranks": {str(l): v for l, v in jspace_ranks.items()},
    "alignments_per_layer": {str(l): v for l, v in alignments_per_layer.items()},
    "sparse_decomposition": sparse_results,
}

summary_path = OUTPUT_DIR / "analysis_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nResults saved to {summary_path}")

print("\nDone!")
