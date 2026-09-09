"""
Crosscoder analysis: (1) per-layer feature organization,
(2) geometric structure (circular/helical features),
(4) separation from J-space.

Usage:
    python crosscoder_analyze.py [--crosscoder results/crosscoder] [--activations results/phase1_q35/activations.npz]
"""
import json, math, argparse, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Config ───
D_MODEL = 5120
LAYERS = list(range(48, 64))  # 48-63
N_LAYERS = len(LAYERS)
MODULUS = 59

parser = argparse.ArgumentParser()
parser.add_argument("--crosscoder", type=str, default="results/crosscoder")
parser.add_argument("--activations", type=str, default="results/phase1_q35/activations.npz")
parser.add_argument("--output_dir", type=str, default="results/crosscoder_analysis")
parser.add_argument("--jspace_path", type=str, default=None,
    help="Path to precomputed Jacobian (if available for Qwen3.5-27B)")
args = parser.parse_args()

OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CKPT_PATH = Path(args.crosscoder) / "checkpoints" / "crosscoder_step0024413.safetensors"

# ─── Load crosscoder ───
print("Loading crosscoder weights...")
sd = load_file(str(CKPT_PATH))
W_dec = sd["W_dec"].numpy()  # (N_feat, L*D)
W_enc = sd["W_enc"].numpy()  # (L*D, N_feat)
b_dec = sd["b_dec"].numpy()
b_enc = sd["b_enc"].numpy()
N_FEAT = W_dec.shape[0]
print(f"  {N_FEAT} features, {N_LAYERS} layers")

# ─── Load activations (from phase 1) ───
print("Loading activations...")
act_data = np.load(args.activations)

# activations.npz has keys like 'train_l0'..'train_l63', 'test_l0'..'test_l63',
# plus 'train_labels', 'test_labels'.
# We use train activations for analysis.
import re
all_layers = sorted([k for k in act_data if re.match(r'^train_l\d+$', k)],
                    key=lambda x: int(x.split("_l")[1]))
n_layers_full = len(all_layers)
n_samples = act_data["train_l0"].shape[0]
d_model = act_data["train_l0"].shape[1]

# Stack into (n_samples, n_layers, d_model)
act_arr = np.stack([act_data[k] for k in all_layers], axis=1)
print(f"  Activation shape: {act_arr.shape}")

# Labels
labels = act_data["train_labels"]  # (n_samples,)
result_vals = labels  # these are the (a+b) mod 59 results
has_labels = True
print(f"  Labels: {labels.shape}, range=[{labels.min()}, {labels.max()}]")

# Only keep layers 48-63 (indices 48..63)
if n_layers_full > N_LAYERS:
    act_arr = act_arr[:, 48:64, :]  # layers 0..63 in full, take 48..63
    print(f"  Trimmed to layers 48-63: {act_arr.shape}")
elif n_layers_full == 64:
    # Assume layers 0-63, trim
    act_arr = act_arr[:, 48:64, :]
    print(f"  Trimmed to layers 48-63: {act_arr.shape}")
else:
    N_LAYERS = n_layers_full
    LAYERS = list(range(48, 48 + N_LAYERS))

# Flatten: (n_samples, L*D)
act_flat = act_arr.reshape(n_samples, -1).astype(np.float32)

# ─── 1. Per-layer feature analysis ───
print("\n=== 1. Per-layer feature organization ===")

# Reshape decoder weights: (N_feat, L, D)
W_dec_ld = W_dec.reshape(N_FEAT, N_LAYERS, D_MODEL)

# Per-layer L2 norm of each feature's decoder vector
layer_norms = np.linalg.norm(W_dec_ld, axis=-1)  # (N_feat, L)

# Dominant layer per feature (argmax norm)
dominant_layer = np.argmax(layer_norms, axis=1)  # (N_feat,)

# Distribution of features across layers
layer_counts = np.bincount(dominant_layer, minlength=N_LAYERS)

# Top features per layer (features most active in each layer)
top_k = 10
top_features_per_layer = {}
for l_idx in range(N_LAYERS):
    layer_norm = layer_norms[:, l_idx]
    top_indices = np.argsort(layer_norm)[-top_k:][::-1]
    top_features_per_layer[LAYERS[l_idx]] = top_indices.tolist()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1a: Feature distribution across layers
ax = axes[0]
ax.bar(range(N_LAYERS), layer_counts, color="steelblue", alpha=0.8)
ax.set_xlabel("Layer index"); ax.set_ylabel("Feature count")
ax.set_title("Features by dominant layer")
ax.set_xticks(range(0, N_LAYERS, 2))
ax.set_xticklabels([str(LAYERS[i]) for i in range(0, N_LAYERS, 2)], rotation=45)
ax.grid(True, alpha=0.3)

# 1b: Per-layer norm heatmap (first 200 features)
ax = axes[1]
im = ax.imshow(layer_norms[:200].T, aspect="auto", cmap="viridis",
               extent=[0, 200, N_LAYERS-0.5, -0.5])
ax.set_xlabel("Feature index"); ax.set_ylabel("Layer index")
ax.set_title("Decoder norm per (feature, layer)")
ax.set_yticks(range(N_LAYERS))
ax.set_yticklabels([str(LAYERS[l]) for l in range(N_LAYERS)])
plt.colorbar(im, ax=ax, shrink=0.8)

# 1c: Norm concentration (how much each feature concentrates in one layer)
ax = axes[2]
concentration = layer_norms.max(axis=1) / (layer_norms.sum(axis=1) + 1e-8)
# Handle edge case where all norms are 1.0 (after renorm_decoder)
if concentration.std() < 1e-6:
    ax.text(0.5, 0.5, f"All norms = 1.0 (renormed)\nConcentration = {concentration[0]:.4f}",
            transform=ax.transAxes, ha="center", va="center", fontsize=12)
    ax.set_title("Feature concentration (uniform — decoder renormed)")
else:
    ax.hist(concentration, bins=20, color="coral", alpha=0.8)
    ax.axvline(x=1.0/N_LAYERS, color="red", linestyle="--", label=f"Uniform={1/N_LAYERS:.3f}")
    ax.legend()
ax.set_xlabel("Max norm / total norm"); ax.set_ylabel("Feature count")
ax.grid(True, alpha=0.3)

plt.suptitle("Crosscoder analysis (1): Per-layer feature organization", fontsize=13)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "analysis1_per_layer.png", dpi=150, bbox_inches="tight")
print(f"Saved {OUTPUT_DIR / 'analysis1_per_layer.png'}")

# ─── 2. Geometric structure: circular/helical features ───
print("\n=== 2. Geometric structure ===")

# For each layer, compute activations and project them through
# the top features of that layer to look for circular structure.
# We need to run a forward pass of the crosscoder on the activations.

# Crosscoder forward on all activations (batch to avoid OOM)
print("  Running crosscoder forward on activations...")
batch_size = 256
act_t = torch.from_numpy(act_flat).float()
W_enc_t = torch.from_numpy(W_enc).float()
b_enc_t = torch.from_numpy(b_enc).float()
b_dec_t = torch.from_numpy(b_dec).float()

all_z = []
for i in range(0, n_samples, batch_size):
    x = act_t[i:i+batch_size]
    pre = (x - b_dec_t) @ W_enc_t + b_enc_t
    z = F.relu(pre)
    # TopK sparsity
    K = 32
    vals, idx = z.topk(K, dim=-1)
    z_sparse = torch.zeros_like(z)
    z_sparse.scatter_(-1, idx, vals)
    all_z.append(z_sparse.numpy())
z_all = np.concatenate(all_z, axis=0)  # (n_samples, N_feat)
print(f"  Latent shape: {z_all.shape}")

# Frequency: how often each feature is active
feature_freq = (z_all > 1e-6).mean(axis=0)  # (N_feat,)

# ─── 2. Geometric structure: circular/helical features ───
print("\n=== 2. Geometric structure ===")

# Crosscoder forward on all activations (batch to avoid OOM)
print("  Running crosscoder forward on activations...")
batch_size = 256
act_t = torch.from_numpy(act_flat).float()
W_enc_t = torch.from_numpy(W_enc).float()
b_enc_t = torch.from_numpy(b_enc).float()
b_dec_t = torch.from_numpy(b_dec).float()

all_z = []
for i in range(0, n_samples, batch_size):
    x = act_t[i:i+batch_size]
    pre = (x - b_dec_t) @ W_enc_t + b_enc_t
    z = F.relu(pre)
    K = 32
    vals, idx = z.topk(K, dim=-1)
    z_sparse = torch.zeros_like(z)
    z_sparse.scatter_(-1, idx, vals)
    all_z.append(z_sparse.numpy())
z_all = np.concatenate(all_z, axis=0)
print(f"  Latent shape: {z_all.shape}")

# Frequency: how often each feature is active
feature_freq = (z_all > 1e-6).mean(axis=0)
active_features = np.where(feature_freq > 0.1)[0]
print(f"  Features active > 10%: {len(active_features)}/{N_FEAT}")

# Top features vs result correlation
if has_labels:
    correlations = []
    for f_idx in active_features[:200]:
        z_f = z_all[:, f_idx]
        r, p = spearmanr(z_f, result_vals)
        correlations.append((f_idx, abs(r), r))
    correlations.sort(key=lambda x: x[1], reverse=True)

    print(f"  Top 5 features correlated with result: "
          f"{[(int(f), float(r)) for f, ar, r in correlations[:5]]}")

    # Plot top 6 features
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))
    for plot_i, (f_idx, _, r_val) in enumerate(correlations[:6]):
        ax = axes2[plot_i // 3, plot_i % 3]
        z_f = z_all[:, f_idx]
        sc = ax.scatter(result_vals, z_f, c=result_vals, cmap="hsv", alpha=0.5, s=10)
        ax.set_xlabel("(a+b) mod 59")
        ax.set_ylabel(f"Feature {f_idx} activation")
        ax.set_title(f"Feature {f_idx} (ρ={r_val:.3f})")
        plt.colorbar(sc, ax=ax)
        ax.grid(True, alpha=0.3)
    plt.suptitle("Crosscoder (2): Top features vs modular result", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "analysis2_features_vs_result.png", dpi=150, bbox_inches="tight")
    print(f"Saved {OUTPUT_DIR / 'analysis2_features_vs_result.png'}")

# PCA of feature space
print("  Computing feature-space PCA...")
z_active = z_all[:, active_features[:200]]
z_centered = z_active - z_active.mean(axis=0)
pca_feat = PCA(n_components=min(50, z_centered.shape[1]))
z_pca = pca_feat.fit_transform(z_centered)

fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
ax = axes3[0]
ax.plot(np.cumsum(pca_feat.explained_variance_ratio_)[:50], "b.-")
ax.set_xlabel("PC"); ax.set_ylabel("Cumulative var explained")
ax.set_title("Feature space PCA variance")
ax.axhline(y=0.5, color="r", linestyle="--", label="50%")
ax.axhline(y=0.8, color="orange", linestyle="--", label="80%")
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes3[1]
if has_labels:
    sc = ax.scatter(z_pca[:, 0], z_pca[:, 1], c=result_vals, cmap="hsv", alpha=0.4, s=5)
    plt.colorbar(sc, ax=ax, label="(a+b) mod 59")
    ax.set_title("Feature PC1 vs PC2, colored by result")

    # Circularity check
    theta = np.arctan2(z_pca[:, 1], z_pca[:, 0])
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    X_circ = np.stack([cos_t, sin_t], axis=1)
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression().fit(X_circ, result_vals)
    pred = lr.predict(X_circ)
    r2 = 1 - np.var(result_vals - pred) / np.var(result_vals)
    ax.annotate(f"Circular fit R²={r2:.3f}", xy=(0.05, 0.95),
                xycoords="axes fraction", fontsize=12, va="top",
                bbox=dict(boxstyle="round", fc="wheat", alpha=0.5))
else:
    ax.scatter(z_pca[:, 0], z_pca[:, 1], alpha=0.4, s=5)
    ax.set_title("Feature PC1 vs PC2")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
ax.grid(True, alpha=0.3)

plt.suptitle("Crosscoder (2): Feature space geometry", fontsize=13)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "analysis2_geometry.png", dpi=150, bbox_inches="tight")
print(f"Saved {OUTPUT_DIR / 'analysis2_geometry.png'}")

# ─── 3. Cross-layer feature relationships ───
print("\n=== 3. Cross-layer feature structure ===")

# Check if top features from different layers share structure
# Correlation between dominant features of different layers
top_features_global = []
for l_idx in range(N_LAYERS):
    top_indices = np.argsort(layer_norms[:, l_idx])[-5:][::-1]
    top_features_global.extend([(LAYERS[l_idx], f) for f in top_indices])

# List top features per layer with their dominant layer
print("  Top 5 features per layer:")
for l_idx in range(N_LAYERS):
    top = np.argsort(layer_norms[:, l_idx])[-5:][::-1]
    doms = [dominant_layer[f] for f in top]
    print(f"    Layer {LAYERS[l_idx]:2d}: features {top}, dominant layers {[LAYERS[d] for d in doms]}")

# ─── 4. J-space comparison ───
print("\n=== 4. J-space separation analysis ===")

if args.jspace_path and Path(args.jspace_path).exists():
    print(f"  Loading J-space from {args.jspace_path}")
    # TODO: load jacobian data
    # For now, placeholder
else:
    print("  No J-space data provided. Computing approximate J-space...")
    # Approximate J-space: for each sample, compute the output prediction
    # We don't have the model loaded, so we use activations as proxy
    # The J-space is the gradient of output wrt residual stream.
    # Without the model, we approximate: J-space ≈ the subspace in the final layers
    # that best predicts the output.

    # Simple proxy: PCA of activations at the last layer (63) vs earlier layers
    act_last = act_arr[:, -1, :]  # layer 63
    act_early = act_arr[:, 0, :]   # layer 48
    act_mid = act_arr[:, 7, :]     # layer 55

    # Compute alignment with crosscoder
    if has_labels:
        # J-space proxy: the direction in each layer's activation space
        # that best predicts the result
        j_directions = []
        for l_idx in range(N_LAYERS):
            lr = LinearRegression().fit(act_arr[:, l_idx, :], result_vals)
            j_vec = lr.coef_  # (D,)
            j_vec = j_vec / (np.linalg.norm(j_vec) + 1e-8)
            j_directions.append(j_vec)
        j_directions = np.array(j_directions)  # (L, D)

        # Compare with crosscoder decoder vectors
        # For each layer, compute alignment between J-direction and decoder vectors
        alignments = []
        for l_idx in range(N_LAYERS):
            j_vec = j_directions[l_idx]
            # Crosscoder decoder for this layer's top features
            cc_dec_l = W_dec_ld[:, l_idx, :]  # (N_feat, D)
            # Cosine similarity with J-direction
            cc_norms = np.linalg.norm(cc_dec_l, axis=-1) + 1e-8
            cos_sim = np.abs(cc_dec_l @ j_vec) / cc_norms
            top_align = np.sort(cos_sim)[-10:][::-1]
            alignments.append(top_align.mean())
        alignments = np.array(alignments)

        # Plot: alignment per layer
        fig4, axes4 = plt.subplots(1, 3, figsize=(18, 5))

        ax = axes4[0]
        ax.bar(range(N_LAYERS), alignments, color="teal", alpha=0.8)
        ax.set_xlabel("Layer index"); ax.set_ylabel("Mean cosine similarity")
        ax.set_title("Crosscoder ↔ J-space alignment per layer")
        ax.set_xticks(range(0, N_LAYERS, 2))
        ax.set_xticklabels([str(LAYERS[l]) for l in range(0, N_LAYERS, 2)], rotation=45)
        ax.grid(True, alpha=0.3)

        # Heatmap: J-direction vs crosscoder features per layer
        ax = axes4[1]
        heatmap = np.zeros((N_LAYERS, min(100, N_FEAT)))
        for l_idx in range(N_LAYERS):
            j_vec = j_directions[l_idx]
            cc_dec_l = W_dec_ld[:100, l_idx, :]  # first 100 features
            cc_norms = np.linalg.norm(cc_dec_l, axis=-1) + 1e-8
            cos_sim = np.abs(cc_dec_l @ j_vec) / cc_norms
            heatmap[l_idx] = cos_sim
        im = ax.imshow(heatmap, aspect="auto", cmap="RdBu_r",
                       extent=[0, 100, N_LAYERS-0.5, -0.5])
        ax.set_xlabel("Feature index"); ax.set_ylabel("Layer index")
        ax.set_title("|cos(J-dir, CC feat)| per (layer, feature)")
        ax.set_yticks(range(N_LAYERS))
        ax.set_yticklabels([str(LAYERS[l]) for l in range(N_LAYERS)])
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Layer-wise correlation: J-direction norms vs CC feature norms
        ax = axes4[2]
        j_norms = np.linalg.norm(j_directions, axis=-1)
        cc_layer_norms = layer_norms.mean(axis=0)  # mean norm per layer
        ax.scatter(range(N_LAYERS), alignments, c="teal", s=80, label="Alignment", zorder=5)
        ax2 = ax.twinx()
        ax2.plot(range(N_LAYERS), cc_layer_norms / cc_layer_norms.max(), "orange",
                 marker="o", label="CC norm (norm.)")
        ax.set_xlabel("Layer index"); ax.set_ylabel("Cos similarity")
        ax2.set_ylabel("CC norm (normalized)")
        ax.set_title("J-alignment vs CC dominance per layer")
        ax.set_xticks(range(0, N_LAYERS, 2))
        ax.set_xticklabels([str(LAYERS[l]) for l in range(0, N_LAYERS, 2)], rotation=45)
        fig4.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        plt.suptitle("Crosscoder analysis (4): J-space vs Crosscoder separation", fontsize=13)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "analysis4_jspace.png", dpi=150, bbox_inches="tight")
        print(f"Saved {OUTPUT_DIR / 'analysis4_jspace.png'}")

        # Summary metric: overall alignment
        mean_align = alignments.mean()
        print(f"  Mean J↔CC alignment: {mean_align:.4f}")
        print(f"  Layer with max alignment: {LAYERS[alignments.argmax()]} ({alignments.max():.4f})")
        print(f"  Layer with min alignment: {LAYERS[alignments.argmin()]} ({alignments.min():.4f})")
    else:
        print("  No labels — skipping J-space analysis")

# ─── Summary ───
print("\n=== Summary ===")
print(f"  Features: {N_FEAT}")
print(f"  Layers: {LAYERS[0]}-{LAYERS[-1]}")
print(f"  Layer with most dominant features: {LAYERS[np.argmax(layer_counts)]} ({layer_counts.max()} features)")
print(f"  Top correlated features with result: {correlations[:5] if has_labels else 'N/A'}")

# Save summary JSON
summary = {
    "n_features": N_FEAT,
    "n_layers": N_LAYERS,
    "layers": LAYERS,
    "layer_feature_counts": layer_counts.tolist(),
    "top_features_per_layer": top_features_per_layer,
    "active_features_pct": float(len(active_features) / N_FEAT * 100),
}
if has_labels:
    summary["top_result_correlated_features"] = [(int(f), float(r), float(raw)) for f, r, raw in correlations[:10]]
    summary["circular_fit_r2"] = float(r2) if "r2" in dir() else None
    summary["mean_j_cc_alignment"] = float(alignments.mean()) if "alignments" in dir() else None

with open(OUTPUT_DIR / "analysis_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nAll results saved to {OUTPUT_DIR}")
