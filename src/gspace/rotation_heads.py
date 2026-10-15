"""
Rotation Head Discovery: Find attention heads that rotate the modular result circle.

Inspired by Anthropic's "When Models Manipulate Manifolds" paper, this script:

1. Trains 59 linear probes (one per mod-59 result value) at the best G-space layer
2. Passes probe weights through each attention head's QK matrices
3. Measures which heads produce cyclic-shift patterns in probe similarity

A head that rotates the mod-59 circle by offset k will show:
- Cosine similarity[i][j] peaked at i = (j + k) % 59

Usage:
    python -m gspace.rotation_heads --smoke_test 10
    python -m gspace.rotation_heads --modulus 59 --output_dir results/rotation_heads
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Project paths ──────────────────────────────────────────────────
PROJ = Path(__file__).resolve().parents[2]
FIG_DIR = PROJ / "figures" / "rotation_heads"
RES_DIR = PROJ / "results" / "rotation_heads"
MODEL_ID = "Qwen/Qwen3.5-27B"

# ── Config ─────────────────────────────────────────────────────────
MODULUS = 59
D_MODEL = 5120
PROBE_LAYER = 61  # Best G-space layer from Phase 1
N_TRAIN = 2400
N_TEST = 1081
BATCH_SIZE = 8
SEED = 42
QK_LAYER_START = 48
QK_LAYER_END = 63  # exclusive (layers 48-62)


# ═════════════════════════════════════════════════════════════════════
# 1. Data generation & activation caching
# ═════════════════════════════════════════════════════════════════════

def generate_problems(modulus: int, n_train: int, n_test: int, seed: int):
    rng = np.random.default_rng(seed)
    all_pairs = [(a, b) for a in range(modulus) for b in range(modulus)]
    rng.shuffle(all_pairs)
    test_pairs = all_pairs[:n_test]
    train_pairs = all_pairs[n_test:n_test + n_train]

    def make(pairs):
        out = []
        for a, b in pairs:
            result = (a + b) % modulus
            prompt = f"Compute {a} + {b} mod {modulus}: {a} + {b} mod {modulus} ="
            out.append({"a": a, "b": b, "result": result, "prompt": prompt})
        return out
    return make(train_pairs), make(test_pairs)


def cache_activations(model, tokenizer, problems, device, save_path):
    if os.path.exists(save_path):
        print(f"Loading cached activations from {save_path}")
        data = np.load(save_path, allow_pickle=True)
        return data["activations"], data["labels"]

    hidden_states = {}
    def make_hook(l):
        def hook(_mod, _inp, out):
            o = out[0] if isinstance(out, tuple) else out
            hidden_states[l] = o.detach().cpu().float()
        return hook

    h = model.model.layers[PROBE_LAYER].register_forward_hook(make_hook(PROBE_LAYER))

    all_acts, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(problems), BATCH_SIZE), desc="Caching"):
            batch = problems[i:i + BATCH_SIZE]
            prompts = [p["prompt"] for p in batch]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
            _ = model(**inputs)
            hs = hidden_states[PROBE_LAYER]  # already float32 on CPU
            last_pos = (inputs["attention_mask"].sum(dim=1) - 1).cpu()
            batch_acts = hs[range(len(last_pos)), last_pos].cpu().numpy()
            all_acts.append(batch_acts)
            all_labels.extend([p["result"] for p in batch])

    h.remove()
    acts = np.concatenate(all_acts, axis=0).astype(np.float32)
    labels = np.array(all_labels, dtype=np.int64)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path, activations=acts, labels=labels)
    print(f"Saved activations: {acts.shape} -> {save_path}")
    return acts, labels


# ═════════════════════════════════════════════════════════════════════
# 2. Per-result-value probes
# ═════════════════════════════════════════════════════════════════════

def train_result_probes(acts, labels):
    """Train 59 binary one-vs-rest probes at PROBE_LAYER.
    
    For classes with no samples (possible in smoke tests), uses zero vector.
    """
    probes = []
    n_skipped = 0
    for c in tqdm(range(MODULUS), desc="Training probes"):
        y = (labels == c).astype(int)
        if y.sum() == 0:
            # No samples for this class — use zeros
            probes.append(np.zeros(acts.shape[1], dtype=np.float32))
            n_skipped += 1
            continue
        if y.sum() == len(y):
            # All samples are this class — trivial classifier
            probes.append(np.zeros(acts.shape[1], dtype=np.float32))
            n_skipped += 1
            continue
        clf = LogisticRegression(
            penalty="l2", C=1.0, max_iter=1000, random_state=SEED, solver="lbfgs"
        )
        clf.fit(acts, y)
        probes.append(clf.coef_[0])
    if n_skipped:
        print(f"  Warning: {n_skipped}/{MODULUS} probes skipped (no samples)")
    return np.stack(probes, axis=0)  # (59, d_model)


# ═════════════════════════════════════════════════════════════════════
# 3. QK analysis
# ═════════════════════════════════════════════════════════════════════

def get_head_qk(layer_module):
    """Extract per-head Q/K weight splits from an attention layer.
    
    Qwen3.5 has two types:
    - full_attention: self_attn with q_proj, k_proj (GQA)
    - linear_attention: linear_attn (don't have standard QK head access)
    """
    # Try full attention first
    if hasattr(layer_module, 'self_attn'):
        attn = layer_module.self_attn
        # Head info is in the config, not the module itself
        n_heads = attn.config.num_attention_heads
        n_kv_heads = attn.config.num_key_value_heads
        head_dim = attn.head_dim
        W_Q = attn.q_proj.weight.detach().float().cpu()
        W_K = attn.k_proj.weight.detach().float().cpu()
        W_Q_list = [W_Q[i * head_dim:(i + 1) * head_dim] for i in range(n_heads)]
        W_K_list = [W_K[i * head_dim:(i + 1) * head_dim] for i in range(n_kv_heads)]
        return W_Q_list, W_K_list, {
            "n_heads": n_heads, "n_kv_heads": n_kv_heads,
            "head_dim": head_dim, "attn_type": "full"
        }
    elif hasattr(layer_module, 'linear_attn'):
        # Linear attention — skip for now (no per-head QK access)
        return [], [], {"attn_type": "linear"}
    else:
        print(f"  WARNING: Unknown attention module: {type(layer_module)}")
        return [], [], {}


def compute_qk_probe_similarity(probe_weights, W_Q_list, W_K_list, head_info):
    """Compute cosine similarity between probes through Q and K per head."""
    n_q = head_info["n_heads"]
    n_kv = head_info["n_kv_heads"]
    q_to_kv = [qi * n_kv // n_q for qi in range(n_q)]
    results = {}

    for qi in range(n_q):
        ki = q_to_kv[qi]
        Qp = probe_weights @ W_Q_list[qi].numpy().T   # (59, head_dim)
        Kp = probe_weights @ W_K_list[ki].numpy().T   # (59, head_dim)
        Qn = Qp / (np.linalg.norm(Qp, axis=1, keepdims=True) + 1e-8)
        Kn = Kp / (np.linalg.norm(Kp, axis=1, keepdims=True) + 1e-8)
        results[(qi, ki)] = Qn @ Kn.T  # (59, 59)

    return results


def detect_cyclic_shift(sim_matrix, modulus=MODULUS):
    """Quantify cyclic shift pattern in similarity matrix."""
    M = modulus
    max_j = np.argmax(sim_matrix, axis=1)
    offsets = (max_j - np.arange(M)) % M
    counts = np.bincount(offsets, minlength=M)
    best_offset = int(np.argmax(counts))
    best_count = int(counts[best_offset])
    cyclic_score = best_count / M
    mean_sim = float(np.mean([sim_matrix[i, (i + best_offset) % M]
                               for i in range(M)]))
    diag_mean = float(np.mean(np.diag(sim_matrix)))
    off_diag = sim_matrix[~np.eye(M, dtype=bool)]
    off_diag_mean = float(np.mean(off_diag))
    contrast = diag_mean / (off_diag_mean + 1e-8)

    # spread_80
    sorted_counts = np.sort(counts)[::-1]
    cumsum = np.cumsum(sorted_counts) / M
    spread_80 = int(np.searchsorted(cumsum, 0.8) + 1)

    return {
        "best_offset": best_offset,
        "cyclic_score": cyclic_score,
        "mean_sim_at_offset": mean_sim,
        "spread_80": spread_80,
        "diag_mean": diag_mean,
        "off_diag_mean": off_diag_mean,
        "contrast_ratio": contrast,
    }


# ═════════════════════════════════════════════════════════════════════
# 4. Main analysis
# ═════════════════════════════════════════════════════════════════════

def analyze_layer(layer_idx, model, probe_weights, layer_type, qk_start, qk_end):
    """Analyze all heads in a given layer."""
    layer_mod = model.model.layers[layer_idx]
    W_Q_list, W_K_list, head_info = get_head_qk(layer_mod)

    if not W_Q_list:
        return None

    # Skip linear attention layers (only analyze full-attention layers)
    if head_info.get("attn_type") == "linear":
        return None

    head_sims = compute_qk_probe_similarity(probe_weights, W_Q_list, W_K_list, head_info)

    results = []
    for (q_head, kv_head), sim_matrix in sorted(head_sims.items()):
        metrics = detect_cyclic_shift(sim_matrix)
        metrics["layer"] = layer_idx
        metrics["q_head"] = q_head
        metrics["kv_head"] = kv_head
        metrics["attn_type"] = layer_type
        metrics["sim_matrix"] = sim_matrix
        results.append(metrics)

    return results


def find_rotation_heads(model, probe_weights, qk_start, qk_end):
    """Analyze QK cyclic shift across attention heads."""
    all_results = []
    # Get layer types from config - may be flat or nested
    c = model.config
    if hasattr(c, 'text_config'):
        try:
            layer_types = c.text_config.layer_types
        except Exception:
            layer_types = c.layer_types
    else:
        layer_types = c.layer_types

    for l in tqdm(range(qk_start, qk_end), desc="Analyzing layers"):
        ltype = layer_types[l] if l < len(layer_types) else "unknown"
        try:
            res = analyze_layer(l, model, probe_weights, ltype, qk_start, qk_end)
            if res:
                all_results.extend(res)
            tqdm.write(f"  Layer {l} ({ltype}): {len(res) if res else 0} heads")
        except Exception as e:
            tqdm.write(f"  Layer {l}: ERROR - {e}")

    return all_results


# ═════════════════════════════════════════════════════════════════════
# 5. Visualization
# ═════════════════════════════════════════════════════════════════════

def plot_baseline(probe_weights, fig_dir):
    """Baseline probe similarity (no QK transformation)."""
    Pn = probe_weights / (np.linalg.norm(probe_weights, axis=1, keepdims=True) + 1e-8)
    sim = Pn @ Pn.T
    M = MODULUS

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    im = axes[0].imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1,
                         extent=[0, M, M, 0])
    axes[0].set_xlabel("Probe j"); axes[0].set_ylabel("Probe i")
    axes[0].set_title(f"Baseline Probe Cosine Similarity (Layer {PROBE_LAYER})")
    plt.colorbar(im, ax=axes[0], label="Cosine similarity")

    offsets = (np.argmax(sim, axis=1) - np.arange(M)) % M
    axes[1].bar(range(M), np.bincount(offsets, minlength=M))
    axes[1].set_xlabel("Offset k"); axes[1].set_ylabel("Count")
    axes[1].set_title("Dominant offset distribution (baseline)")
    axes[1].axhline(M, color="red", linestyle="--", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(fig_dir, "baseline_similarity.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_top_heads(head_results, fig_dir, top_n=12):
    """Similarity matrices for top cyclic-shift heads."""
    sorted_heads = sorted(head_results, key=lambda h: h["cyclic_score"], reverse=True)[:top_n]
    ncols = 4
    nrows = (len(sorted_heads) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * max(1, nrows)))
    axes = np.atleast_1d(axes).flatten()

    M = MODULUS
    for idx, head in enumerate(sorted_heads):
        ax = axes[idx]
        im = ax.imshow(head["sim_matrix"], cmap="RdBu_r", vmin=-1, vmax=1,
                        extent=[0, M, M, 0])
        ax.set_title(f"L{head['layer']} H{head['q_head']} "
                      f"cyclic={head['cyclic_score']:.2f} offset={head['best_offset']}")
        ax.set_xlabel("K probe j"); ax.set_ylabel("Q probe i")
        plt.colorbar(im, ax=ax)

    for idx in range(len(sorted_heads), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Top Cyclic-Shift Heads: Probe Similarity through QK", fontsize=14, y=1.01)
    fig.tight_layout()
    path = os.path.join(fig_dir, "top_cyclic_heads.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_summary(head_results, fig_dir):
    """Layer × head grid colored by cyclic score and offset."""
    layers = sorted(set(h["layer"] for h in head_results))
    max_heads = max(h["q_head"] for h in head_results) + 1

    score_mat = np.full((len(layers), max_heads), np.nan)
    offset_mat = np.full((len(layers), max_heads), np.nan)
    for h in head_results:
        li = layers.index(h["layer"])
        score_mat[li, h["q_head"]] = h["cyclic_score"]
        offset_mat[li, h["q_head"]] = h["best_offset"]

    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, len(layers) * 0.35)))

    im0 = axes[0].imshow(score_mat, aspect="auto", cmap="viridis",
                          vmin=0, vmax=max(0.5, np.nanmax(score_mat)))
    axes[0].set_xticks(range(max_heads))
    axes[0].set_xticklabels([str(i) for i in range(max_heads)], fontsize=7)
    axes[0].set_yticks(range(len(layers)))
    axes[0].set_yticklabels([str(l) for l in layers])
    axes[0].set_xlabel("Head index"); axes[0].set_ylabel("Layer")
    axes[0].set_title("Cyclic Shift Score")
    plt.colorbar(im0, ax=axes[0], label="Cyclic score")

    im1 = axes[1].imshow(offset_mat, aspect="auto", cmap="twilight",
                          vmin=0, vmax=MODULUS - 1)
    axes[1].set_xticks(range(max_heads))
    axes[1].set_xticklabels([str(i) for i in range(max_heads)], fontsize=7)
    axes[1].set_yticks(range(len(layers)))
    axes[1].set_yticklabels([str(l) for l in layers])
    axes[1].set_xlabel("Head index")
    axes[1].set_title("Best Offset")
    plt.colorbar(im1, ax=axes[1], label="Offset")

    fig.suptitle("Rotation Head Discovery: QK Cyclic Shift Analysis", fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(fig_dir, "summary_grid.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_offset_hist(head_results, fig_dir):
    """Histogram of best_offset across all heads."""
    offsets = [h["best_offset"] for h in head_results]
    scores = [h["cyclic_score"] for h in head_results]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.hist(offsets, bins=MODULUS, range=(-0.5, MODULUS - 0.5),
            weights=scores, alpha=0.7, label="Weighted by cyclic score")
    ax.hist(offsets, bins=MODULUS, range=(-0.5, MODULUS - 0.5),
            alpha=0.3, label="Unweighted")
    ax.set_xlabel("Cyclic offset k"); ax.set_ylabel("Count")
    ax.set_title("Offset distribution across all heads")
    ax.legend(); ax.axvline(0, color="red", linestyle="--", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(fig_dir, "offset_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_probe_geometry(probe_weights, fig_dir):
    """PCA of 59 probe vectors to show circle topology."""
    pca = PCA(n_components=6).fit(probe_weights)
    proj = pca.transform(probe_weights)
    ev = pca.explained_variance_ratio_

    fig = plt.figure(figsize=(14, 10))

    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    colors = np.arange(MODULUS)
    xs, ys, zs = proj[:, 0], proj[:, 1], proj[:, 2]
    ax1.scatter(xs, ys, zs, c=colors, cmap="twilight", s=30)
    for i in range(MODULUS):
        j = (i + 1) % MODULUS
        ax1.plot([xs[i], xs[j]], [ys[i], ys[j]], [zs[i], zs[j]],
                 color="gray", alpha=0.3, lw=0.5)
    ax1.set_title(f"PC1-3 ({sum(ev[:3])*100:.1f}% var)")
    ax1.set_xlabel("PC1"); ax1.set_ylabel("PC2"); ax1.set_zlabel("PC3")

    ax2 = fig.add_subplot(2, 2, 2)
    sc = ax2.scatter(proj[:, 0], proj[:, 1], c=colors, cmap="twilight", s=40)
    for i in range(MODULUS):
        j = (i + 1) % MODULUS
        ax2.plot([proj[i, 0], proj[j, 0]], [proj[i, 1], proj[j, 1]],
                 color="gray", alpha=0.4, lw=0.5)
    ax2.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)"); ax2.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
    ax2.set_title("Probe PCA"); plt.colorbar(sc, ax=ax2, label="Result")

    ax3 = fig.add_subplot(2, 2, 3)
    Pn = probe_weights / (np.linalg.norm(probe_weights, axis=1, keepdims=True) + 1e-8)
    im = ax3.imshow(Pn @ Pn.T, cmap="RdBu_r", vmin=-1, vmax=1,
                     extent=[0, MODULUS, MODULUS, 0])
    ax3.set_xlabel("Probe j"); ax3.set_ylabel("Probe i")
    ax3.set_title("Probe Cosine Similarity"); plt.colorbar(im, ax=ax3)

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(range(1, len(ev) + 1), np.cumsum(ev), "o-", color="steelblue")
    ax4.axhline(0.9, color="red", linestyle="--", alpha=0.5)
    ax4.set_xlabel("PCs"); ax4.set_ylabel("Cumulative variance")
    ax4.set_title("Scree Plot"); ax4.grid(True, alpha=0.3)

    fig.suptitle(f"Probe Geometry at Layer {PROBE_LAYER}: Mod {MODULUS} Circle", fontsize=14)
    fig.tight_layout()
    path = os.path.join(fig_dir, "probe_geometry.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_head_layer_scores(head_results, fig_dir):
    """Also plot contrast_ratio as a metric."""
    layers = sorted(set(h["layer"] for h in head_results))
    max_heads = max(h["q_head"] for h in head_results) + 1
    contrast_mat = np.full((len(layers), max_heads), np.nan)
    for h in head_results:
        li = layers.index(h["layer"])
        contrast_mat[li, h["q_head"]] = h["contrast_ratio"]

    fig, ax = plt.subplots(figsize=(12, max(5, len(layers) * 0.35)))
    im = ax.imshow(contrast_mat, aspect="auto", cmap="inferno")
    ax.set_xticks(range(max_heads))
    ax.set_xticklabels([str(i) for i in range(max_heads)], fontsize=7)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([str(l) for l in layers])
    ax.set_xlabel("Head index"); ax.set_ylabel("Layer")
    ax.set_title("Contrast Ratio (diag / off-diag)")
    plt.colorbar(im, ax=ax, label="Contrast")
    fig.tight_layout()
    path = os.path.join(fig_dir, "contrast_grid.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ═════════════════════════════════════════════════════════════════════
# 6. Main
# ═════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Rotation head discovery")
    parser.add_argument("--modulus", type=int, default=MODULUS)
    parser.add_argument("--probe_layer", type=int, default=PROBE_LAYER)
    parser.add_argument("--qk_start", type=int, default=QK_LAYER_START)
    parser.add_argument("--qk_end", type=int, default=QK_LAYER_END)
    parser.add_argument("--n_train", type=int, default=N_TRAIN)
    parser.add_argument("--n_test", type=int, default=N_TEST)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--output_dir", type=str, default=str(RES_DIR))
    parser.add_argument("--fig_dir", type=str, default=str(FIG_DIR))
    parser.add_argument("--smoke_test", type=int, default=0,
                        help="If >0, use N problems for a quick test")
    args = parser.parse_args()

    modulus = args.modulus
    probe_layer = args.probe_layer
    qk_start = args.qk_start
    qk_end = args.qk_end
    n_train = args.n_train
    n_test = args.n_test
    batch_size = args.batch_size

    is_smoke = args.smoke_test > 0
    if is_smoke:
        smoke_n = args.smoke_test
        n_train = min(n_train, smoke_n)
        n_test = min(n_test, smoke_n)
        print(f"SMOKE TEST: using {n_train} train, {n_test} test samples")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    # 1. Load model
    print(f"Loading {MODEL_ID}...")
    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=dtype, device_map="auto",
        trust_remote_code=True, attn_implementation="eager"
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    device = next(model.parameters()).device
    print(f"Device: {device}")

    # 2. Generate problems & cache activations
    train_probs, test_probs = generate_problems(modulus, n_train, n_test, SEED)
    act_path = os.path.join(args.output_dir, f"acts_l{probe_layer}.npz")
    acts, labels = cache_activations(model, tokenizer, train_probs, device, act_path)

    # 3. Train per-value probes
    print("Training 1-vs-rest probes...")
    probe_weights = train_result_probes(acts, labels)
    print(f"Probe weights: {probe_weights.shape}")
    np.save(os.path.join(args.output_dir, "probe_weights.npy"), probe_weights)

    # 4. Baseline
    print("Computing baseline...")
    plot_baseline(probe_weights, args.fig_dir)
    plot_probe_geometry(probe_weights, args.fig_dir)

    # 5. QK analysis
    print(f"Analyzing QK cyclic shift: layers {qk_start}-{qk_end-1}...")
    head_results = find_rotation_heads(model, probe_weights, qk_start, qk_end)

    # 6. Save results
    print(f"Saving {len(head_results)} head results...")
    serializable = []
    for h in head_results:
        d = {k: v for k, v in h.items() if k != "sim_matrix"}
        serializable.append(d)
    with open(os.path.join(args.output_dir, "head_results.json"), "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    # Also save sim matrices as npz
    sim_matrices = {}
    for i, h in enumerate(head_results):
        key = f"L{h['layer']}_H{h['q_head']}"
        sim_matrices[key] = h["sim_matrix"]
    np.savez_compressed(os.path.join(args.output_dir, "sim_matrices.npz"), **sim_matrices)

    # 7. Visualize
    print("Plotting...")
    if head_results:
        plot_top_heads(head_results, args.fig_dir)
        plot_summary(head_results, args.fig_dir)
        plot_offset_hist(head_results, args.fig_dir)
        plot_head_layer_scores(head_results, args.fig_dir)
    else:
        print("  No head results to plot (0 heads analyzed)")

    # 8. Print top heads
    if head_results:
        sorted_heads = sorted(head_results, key=lambda h: h["cyclic_score"], reverse=True)[:10]
        print("\n═══════════════════════════════════════════════")
        print("Top 10 Rotation Heads by Cyclic Score:")
        print(f"{'Layer':>6} {'Head':>5} {'Score':>7} {'Offset':>7} {'Contrast':>9}")
        print("─" * 40)
        for h in sorted_heads:
            print(f"{h['layer']:>6} {h['q_head']:>5} {h['cyclic_score']:>7.3f} "
                  f"{h['best_offset']:>7} {h['contrast_ratio']:>9.3f}")
    else:
        print("\nNo head results to display.")

    print("\nDone!")


if __name__ == "__main__":
    main()
