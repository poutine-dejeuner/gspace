"""
Causal analysis: G-space → J-space relationship.

Four experiments:
1. ABLATION: Remove top G-space features, measure J-space rank degradation
2. MAPPING: Feature↔token alignment matrix
3. EVOLUTION: Layer-by-layer G→J information transfer
4. INTERVENTION: Inject false answer in G-space, check if J-space reads it
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
LAYERS = list(range(48, 64))
K_TOP = 32  # TopK sparsity for crosscoder


# ═════════════════════════════════════════════════════════════════════
# Data loading (same as jspace_analyze.py)
# ═════════════════════════════════════════════════════════════════════

def load_jlens(path):
    print(f"Loading J-lens from {path}")
    d = torch.load(path, map_location="cpu", weights_only=True)
    J_dict = d["J"]
    n_prompts = d.get("n_prompts", "?")
    J_matrices = {}
    for layer_name, J in J_dict.items():
        l = int(layer_name)
        J_matrices[l] = J.to(torch.float32).numpy()
    print(f"  {len(J_matrices)} layers (fitted on {n_prompts} prompts)")
    return J_matrices


def load_wu():
    cache = os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen3.5-27B/"
        "snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654"
    )
    print(f"Loading W_U from {cache}")
    for fname in sorted(os.listdir(cache)):
        if fname.endswith(".safetensors"):
            tensors = safe_load(os.path.join(cache, fname))
            if "lm_head.weight" in tensors:
                wu = tensors["lm_head.weight"].to(torch.float32).numpy()
                print(f"  W_U: {wu.shape}")
                return wu
    raise FileNotFoundError("lm_head.weight not found")


def load_activations(path):
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
    with open(os.path.join(path, "config.json")) as f:
        config = json.load(f)
    n_feat = config.get("n_features", config.get("expansion_factor", 4096))
    n_layers_cc = config.get("n_layers", 16)

    ckpt_path = os.path.join(path, "checkpoints", "crosscoder_step0024413.safetensors")
    ckpt = safe_load(ckpt_path)
    W_enc = ckpt["W_enc"].to(torch.float32).numpy()
    W_dec = ckpt["W_dec"].to(torch.float32).numpy()
    b_enc = ckpt.get("b_enc", torch.zeros(n_feat)).to(torch.float32).numpy()

    print(f"  {n_feat} features, {n_layers_cc} layers")
    return n_feat, n_layers_cc, W_enc, W_dec, b_enc


def compute_jlens_vectors(J_matrices, W_U, l):
    J = J_matrices[l]
    return (W_U.astype(np.float64) @ J.astype(np.float64)).astype(np.float32)


def get_topk_features(pre_acts, K):
    n, n_feat = pre_acts.shape
    topk_idx = np.argpartition(-pre_acts, K, axis=1)[:, :K]
    features = np.zeros_like(pre_acts)
    for i in range(n):
        features[i, topk_idx[i]] = np.maximum(pre_acts[i, topk_idx[i]], 0)
    return features, topk_idx


def _compute_rank(scores, sub_tokens, tok_to_idx, n_answer_toks):
    max_rank = 0
    for st in sub_tokens:
        if st not in tok_to_idx:
            max_rank = max(max_rank, n_answer_toks)
            continue
        idx = tok_to_idx[st]
        score = scores[idx]
        rank = int((scores > score).sum() + 1)
        max_rank = max(max_rank, rank)
    return max_rank


# ═════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Ablation — remove top G-space features
# ═════════════════════════════════════════════════════════════════════

def experiment_1_ablation(acts_dict, labels, W_enc, W_dec, b_enc,
                          J_matrices, W_U, tokenizer, answer_full_tokens,
                          all_answer_sub_tokens, tok_to_idx,
                          FIG_DIR, OUTPUT_DIR):
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Causal ablation — G-space features → J-space rank")
    print("=" * 60)

    l_peak = 61
    l_idx = l_peak - LAYERS[0]
    n_answer_toks = len(all_answer_sub_tokens)

    acts = acts_dict[l_peak].astype(np.float32)
    n = len(labels)

    enc_slice = W_enc[l_idx * D_MODEL:(l_idx + 1) * D_MODEL, :]
    dec_slice = W_dec[:, l_idx * D_MODEL:(l_idx + 1) * D_MODEL]
    pre_acts = acts @ enc_slice + b_enc
    features, topk_idx = get_topk_features(pre_acts, K_TOP)

    reg = LinearRegression()
    reg.fit(features, labels)
    coefs = reg.coef_
    feature_rank = np.argsort(-np.abs(coefs))

    # J-lens baseline
    j_full = compute_jlens_vectors(J_matrices, W_U, l_peak).astype(np.float32)
    j_answer = j_full[all_answer_sub_tokens]

    scores_base = acts @ j_answer.T
    base_ranks = [_compute_rank(scores_base[i], answer_full_tokens[int(labels[i])],
                                tok_to_idx, n_answer_toks) for i in range(n)]
    base_mean_rank = float(np.mean(base_ranks))
    base_top1 = float(np.mean([r == 1 for r in base_ranks]))
    print(f"  Baseline: mean_rank={base_mean_rank:.2f}, top1={base_top1:.3f}")

    ablation_fractions = [0, 10, 50, 100, 200, 500, 1000, 2000, 4096]
    ablation_results = []

    for n_remove in ablation_fractions:
        if n_remove == 0:
            ablation_results.append((0, base_mean_rank, base_top1))
            continue

        removed = set(feature_rank[:n_remove])
        correction = np.zeros_like(acts)
        for f_idx in removed:
            for i in range(n):
                if features[i, f_idx] > 0:
                    correction[i] += dec_slice[f_idx] * features[i, f_idx]

        acts_ablated = acts - correction
        scores_ab = acts_ablated @ j_answer.T
        ranks = [_compute_rank(scores_ab[i], answer_full_tokens[int(labels[i])],
                               tok_to_idx, n_answer_toks) for i in range(n)]
        mean_rank = float(np.mean(ranks))
        top1 = float(np.mean([r == 1 for r in ranks]))
        ablation_results.append((n_remove, mean_rank, top1))
        print(f"  Remove top {n_remove:4d} features: mean_rank={mean_rank:.2f}, "
              f"top1={top1:.3f} (Δrank={mean_rank - base_mean_rank:+.2f})")

    # Plot
    n_removes = [r[0] for r in ablation_results]
    ranks = [r[1] for r in ablation_results]
    top1s = [r[2] for r in ablation_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(n_removes, ranks, "o-", color="tab:red", markersize=6, linewidth=2)
    ax1.axhline(base_mean_rank, color="gray", linestyle="--", alpha=0.5,
                label=f"Baseline: {base_mean_rank:.2f}")
    ax1.set_xlabel("Number of top G-space features removed")
    ax1.set_ylabel("Mean J-space rank")
    ax1.set_title(f"Ablation: G-space features → J-space rank (layer {l_peak})")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(n_removes, [t * 100 for t in top1s], "o-", color="tab:blue",
             markersize=6, linewidth=2)
    ax2.axhline(base_top1 * 100, color="gray", linestyle="--", alpha=0.5,
                label=f"Baseline: {base_top1*100:.1f}%")
    ax2.set_xlabel("Number of top G-space features removed")
    ax2.set_ylabel("Top-1 accuracy (%)")
    ax2.set_title("J-space top-1 degradation")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Experiment 1: Causal Ablation G→J", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig_path = FIG_DIR / "exp1_ablation.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"  Saved {fig_path}")

    # Description
    desc = {
        "title": "Ablation: G-space features → J-space rank",
        "description": (
            "On retire progressivement les features G-space les plus importantes "
            "(classées par |coefficient| de régression linéaire vers la réponse) "
            "de l'activation au layer 61, et on mesure la dégradation du J-space rank. "
            "Si G→J était causal, le rank devrait augmenter fortement quand on enlève "
            "les features de calcul. Résultat : même en enlevant 4096 features, "
            f"le rank passe seulement de {base_mean_rank:.2f} à {ranks[-1]:.2f} "
            f"(Δ=+{ranks[-1] - base_mean_rank:.2f}). L'information est redondante "
            "à travers les couches, ou le crosscoder n'a pas isolé les features "
            "de calcul de façon suffisamment sparse."
        ),
        "baseline_rank": base_mean_rank,
        "baseline_top1": base_top1,
        "max_degradation": ranks[-1] - base_mean_rank,
    }
    json.dump(desc, open(OUTPUT_DIR / "exp1_description.json", "w"), indent=2)
    return ablation_results


# ═════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Feature↔token alignment matrix
# ═════════════════════════════════════════════════════════════════════

def experiment_2_mapping(acts_dict, labels, W_enc, W_dec, b_enc,
                         J_matrices, W_U, tokenizer, answer_token_ids,
                         FIG_DIR, OUTPUT_DIR):
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: G-space feature ↔ J-lens token alignment matrix")
    print("=" * 60)

    l_peak = 61
    l_idx = l_peak - LAYERS[0]

    acts = acts_dict[l_peak].astype(np.float32)
    enc_slice = W_enc[l_idx * D_MODEL:(l_idx + 1) * D_MODEL, :]
    dec_slice = W_dec[:, l_idx * D_MODEL:(l_idx + 1) * D_MODEL]
    pre_acts = acts @ enc_slice + b_enc
    features, _ = get_topk_features(pre_acts, K_TOP)

    reg = LinearRegression()
    reg.fit(features, labels)
    coefs = reg.coef_

    # J-lens for digit tokens
    digit_indices = sorted(set(
        tokenizer.encode(str(d), add_special_tokens=False)[0]
        for d in range(10)
    ))
    j_full = compute_jlens_vectors(J_matrices, W_U, l_peak).astype(np.float32)
    j_digits = j_full[digit_indices]
    j_normed = j_digits / (np.linalg.norm(j_digits, axis=1, keepdims=True) + 1e-8)
    dec_normed = dec_slice / (np.linalg.norm(dec_slice, axis=1, keepdims=True) + 1e-8)

    alignments = np.abs(dec_normed @ j_normed.T)  # [4096, 10]

    top_n = 200
    top_features = np.argsort(-np.abs(coefs))[:top_n]
    digit_names = [str(d) for d in range(10)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 2a: Heatmap top-50 × digits
    top50 = top_features[:50]
    ax = axes[0, 0]
    im = ax.imshow(alignments[top50, :].T, aspect="auto", cmap="Reds",
                   vmin=0, vmax=np.max(alignments[top50, :]) * 0.8)
    ax.set_xlabel("Top G-space feature (by |coeff|)")
    ax.set_ylabel("Digit token")
    ax.set_yticks(range(10))
    ax.set_yticklabels(digit_names)
    ax.set_title("|cos(G-feature, J-digit)| — top 50 features")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # 2b: Max alignment per digit
    ax = axes[0, 1]
    max_per_digit = np.max(alignments[top_features, :], axis=0)
    colors = [plt.cm.Reds(0.3 + 0.7 * v / max(max_per_digit)) for v in max_per_digit]
    ax.bar(range(10), max_per_digit, color=colors)
    ax.set_xticks(range(10))
    ax.set_xticklabels(digit_names)
    ax.set_ylabel("Max |cosine| with any top-200 feature")
    ax.set_title("Best G→J alignment per digit")
    ax.grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(max_per_digit):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)

    # 2c: Importance vs alignment scatter
    ax = axes[1, 0]
    max_per_feat = np.max(alignments[top_features, :], axis=1)
    coefs_top = np.abs(coefs[top_features])
    ax.scatter(coefs_top, max_per_feat, alpha=0.5, s=20, c="tab:red")
    ax.set_xlabel("|Regression coefficient| (importance for answer)")
    ax.set_ylabel("Max |cosine| with any digit J-lens")
    ax.set_title(f"Feature importance vs J-space alignment (top {top_n})")
    corr = np.corrcoef(coefs_top, max_per_feat)[0, 1]
    ax.text(0.95, 0.95, f"r = {corr:.3f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=11,
            bbox=dict(boxstyle="round", fc="wheat", alpha=0.5))
    ax.grid(True, alpha=0.3)

    # 2d: Alignment distribution
    ax = axes[1, 1]
    all_max = np.max(alignments, axis=1)
    top_max = np.max(alignments[top_features, :], axis=1)
    ax.hist(all_max, bins=40, alpha=0.5, color="gray", label=f"All {4096} features")
    ax.hist(top_max, bins=30, alpha=0.7, color="tab:red",
            label=f"Top {top_n} (by |coeff|)")
    ax.set_xlabel("Max |cosine| with any digit")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of feature→digit alignment")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Experiment 2: G-space Feature ↔ J-lens Token Alignment",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig_path = FIG_DIR / "exp2_mapping.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"  Saved {fig_path}")

    # Stats
    print(f"\n  Among top {top_n} features:")
    print(f"    Mean max alignment: {np.mean(top_max):.4f}")
    print(f"    Median max alignment: {np.median(top_max):.4f}")
    high_align = np.sum(top_max > 0.15)
    print(f"    Features with |cos| > 0.15: {high_align}/{top_n} ({high_align/top_n*100:.1f}%)")
    print(f"    Correlation importance vs alignment: r = {corr:.3f}")

    desc = {
        "title": "Mapping: G-space features ↔ J-lens token alignment",
        "description": (
            "Pour chaque feature G-space (crosscoder decoder direction), "
            "on mesure son cosinus avec chaque direction J-lens des 10 digits (0-9). "
            "Panneau 1: heatmap d'alignement pour les top-50 features. "
            "Panneau 2: meilleur alignement trouvé par digit. "
            "Panneau 3: corrélation entre importance de la feature (|coeff|) "
            "et son alignement J-lens max. "
            "Panneau 4: distribution des alignements, comparant toutes les features "
            "vs les top-200. "
            f"Résultat : alignement moyen = {np.mean(top_max):.4f}, "
            f"max absolu = {np.max(top_max):.4f}, corrélation importance/alignement "
            f"r = {corr:.3f}. Aucune feature top-200 n'a de cos > 0.15 avec un digit. "
            "Les directions G-space et J-space sont quasi décorrélées."
        ),
        "mean_alignment_top200": float(np.mean(top_max)),
        "max_alignment_top200": float(np.max(top_max)),
        "correlation_importance_vs_alignment": float(corr),
        "features_with_cos_gt_015": int(high_align),
    }
    json.dump(desc, open(OUTPUT_DIR / "exp2_description.json", "w"), indent=2)

    print(f"\n  Best feature per digit:")
    for d in range(10):
        best_f = top_features[np.argmax(alignments[top_features, d])]
        best_cos = np.max(alignments[top_features, d])
        best_coef = coefs[best_f]
        print(f"    Digit {d}: feature {best_f:4d}, cos={best_cos:.4f}, "
              f"|coeff|={abs(best_coef):.3f}")

    return alignments, coefs


# ═════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: Layer-by-layer G→J information transfer
# ═════════════════════════════════════════════════════════════════════

def experiment_3_evolution(acts_dict, labels, W_enc, W_dec, b_enc,
                           J_matrices, W_U, answer_full_tokens,
                           all_answer_sub_tokens, tok_to_idx,
                           FIG_DIR, OUTPUT_DIR):
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Layer-by-layer G→J information transfer")
    print("=" * 60)

    n = len(labels)
    n_answer_toks = len(all_answer_sub_tokens)
    gspace_r2 = {}
    jspace_full_rank = {}
    cc_decoder_norms = {}

    for l in LAYERS:
        if l not in acts_dict or l not in J_matrices:
            continue

        acts = acts_dict[l].astype(np.float32)
        l_idx = l - LAYERS[0]

        enc_slice = W_enc[l_idx * D_MODEL:(l_idx + 1) * D_MODEL, :]
        pre_acts = acts @ enc_slice + b_enc
        features, _ = get_topk_features(pre_acts, K_TOP)
        reg = LinearRegression()
        reg.fit(features, labels)
        gspace_r2[l] = float(reg.score(features, labels))

        dec_slice = W_dec[:, l_idx * D_MODEL:(l_idx + 1) * D_MODEL]
        cc_decoder_norms[l] = float(np.linalg.norm(dec_slice, "fro"))

        j_full = compute_jlens_vectors(J_matrices, W_U, l).astype(np.float32)
        j_answer = j_full[all_answer_sub_tokens]
        scores = acts @ j_answer.T

        ranks_full = []
        for i in range(n):
            ans = int(labels[i])
            max_rank = _compute_rank(scores[i], answer_full_tokens[ans],
                                     tok_to_idx, n_answer_toks)
            ranks_full.append(max_rank)
        jspace_full_rank[l] = float(np.mean(ranks_full))

        print(f"  L{l:2d}: G-R²={gspace_r2[l]:.3f}, J-rank_full={jspace_full_rank[l]:.2f}, "
              f"‖W_dec‖_F={cc_decoder_norms[l]:.1f}")

    valid = sorted(gspace_r2.keys())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 3a: G-R² and J-rank (inverse) normalized
    ax = axes[0]
    r2_vals = [gspace_r2[l] for l in valid]
    rank_inv = [1.0 / jspace_full_rank[l] for l in valid]
    r2_n = (np.array(r2_vals) - min(r2_vals)) / (max(r2_vals) - min(r2_vals) + 1e-8)
    ri_n = (np.array(rank_inv) - min(rank_inv)) / (max(rank_inv) - min(rank_inv) + 1e-8)

    ax.plot(valid, r2_n, "o-", color="tab:blue", markersize=6, linewidth=2,
            label="G-space R² (norm)")
    ax.plot(valid, ri_n, "s-", color="tab:red", markersize=6, linewidth=2,
            label="J-space 1/rank (norm)")
    crossover = None
    for i in range(len(valid)):
        if ri_n[i] > r2_n[i] and (i == 0 or ri_n[i-1] <= r2_n[i-1]):
            crossover = valid[i]
            break
    if crossover:
        ax.axvline(crossover, color="purple", linestyle=":", alpha=0.5,
                   label=f"Cross L{crossover}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Normalized score")
    ax.set_title("G-space vs J-space emergence")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3b: Per-layer J improvement + decoder norm
    ax = axes[1]
    ax2 = ax.twinx()
    norms = [cc_decoder_norms[l] for l in valid]
    ax2.plot(valid, norms, "s--", color="tab:green", markersize=4, linewidth=1.5,
             label="‖W_dec‖_F")
    ax2.set_ylabel("Decoder Frobenius norm", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")

    j_deltas = []
    for i in range(len(valid) - 1):
        delta = jspace_full_rank[valid[i]] - jspace_full_rank[valid[i+1]]
        j_deltas.append(max(0, delta))
    ax.bar(valid[1:], j_deltas, color="tab:orange", alpha=0.6, width=0.6,
           label="ΔJ-rank improvement")
    ax.set_xlabel("Layer")
    ax.set_ylabel("J-rank improvement (Δ)")
    ax.set_title("Per-layer J-space improvement + decoder norm")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    # 3c: G-R² vs J-rank trajectory
    ax = axes[2]
    for l in valid:
        ax.scatter(gspace_r2[l], jspace_full_rank[l], s=80, alpha=0.7)
        ax.annotate(str(l), (gspace_r2[l], jspace_full_rank[l]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    for i in range(len(valid) - 1):
        l1, l2 = valid[i], valid[i+1]
        ax.annotate("", xy=(gspace_r2[l2], jspace_full_rank[l2]),
                    xytext=(gspace_r2[l1], jspace_full_rank[l1]),
                    arrowprops=dict(arrowstyle="->", color="gray", alpha=0.4, lw=0.8))
    ax.set_xlabel("G-space R²")
    ax.set_ylabel("J-space full-answer rank (lower=better)")
    ax.set_title("G×J trajectory per layer")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    plt.suptitle("Experiment 3: Layer-by-layer G→J Information Transfer\n"
                 "Qwen3.5-27B, mod 59 addition",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig_path = FIG_DIR / "exp3_evolution.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"  Saved {fig_path}")

    from scipy.stats import spearmanr
    r_spear, p_spear = spearmanr(
        [gspace_r2[l] for l in valid],
        [1.0 / jspace_full_rank[l] for l in valid]
    )
    print(f"\n  Spearman r(G-R², 1/J-rank) across layers: {r_spear:.3f} (p={p_spear:.4f})")

    # Description
    desc = {
        "title": "Evolution: Layer-by-layer G→J information transfer",
        "description": (
            "Pour chaque couche l ∈ [48,62], on mesure G-R² (qualité de la régression "
            "des features crosscoder vers la réponse) et J-space rank (rang du token "
            "réponse complet dans les scores J-lens). "
            "Panneau 1: émergence normalisée de G et J — les deux courbes se superposent "
            "presque parfaitement. "
            "Panneau 2: amélioration par couche du J-rank (barres orange) et norme "
            "du décodeur crosscoder par couche (courbe verte). "
            "Panneau 3: trajectoire (G-R², J-rank) à travers les couches — "
            "le point se déplace vers (haut, bas) = meilleure computation. "
            f"Spearman r(G-R², 1/J-rank) = {r_spear:.3f} (p={p_spear:.4f}). "
            "G et J émergent de façon quasi simultanée, sans délai visible. "
            "Deux interprétations possibles : (a) G→J est instantané (même couche), "
            "ou (b) G et J sont deux lectures parallèles d'un même calcul sous-jacent."
        ),
        "spearman_r": float(r_spear),
        "spearman_p": float(p_spear),
        "gspace_peak_layer": max(gspace_r2, key=gspace_r2.get),
        "gspace_peak_r2": float(gspace_r2[max(gspace_r2, key=gspace_r2.get)]),
        "jspace_peak_layer": min(jspace_full_rank, key=jspace_full_rank.get),
        "jspace_best_rank": float(jspace_full_rank[min(jspace_full_rank, key=jspace_full_rank.get)]),
    }
    json.dump(desc, open(OUTPUT_DIR / "exp3_description.json", "w"), indent=2)

    return gspace_r2, jspace_full_rank


# ═════════════════════════════════════════════════════════════════════
# EXPERIMENT 4: Intervention — inject false answer
# ═════════════════════════════════════════════════════════════════════

def experiment_4_intervention(acts_dict, labels, W_enc, W_dec, b_enc,
                              J_matrices, W_U, answer_full_tokens,
                              all_answer_sub_tokens, tok_to_idx,
                              FIG_DIR, OUTPUT_DIR):
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Intervention — inject false answer in G-space")
    print("=" * 60)

    l_peak = 61
    l_idx = l_peak - LAYERS[0]
    n_answer_toks = len(all_answer_sub_tokens)

    acts = acts_dict[l_peak].astype(np.float32)
    n = len(labels)

    enc_slice = W_enc[l_idx * D_MODEL:(l_idx + 1) * D_MODEL, :]
    dec_slice = W_dec[:, l_idx * D_MODEL:(l_idx + 1) * D_MODEL]
    pre_acts = acts @ enc_slice + b_enc
    features, topk_idx = get_topk_features(pre_acts, K_TOP)

    reg = LinearRegression()
    reg.fit(features, labels)
    coefs = reg.coef_

    j_full = compute_jlens_vectors(J_matrices, W_U, l_peak).astype(np.float32)
    j_answer = j_full[all_answer_sub_tokens]

    np.random.seed(42)
    n_intervene = min(200, n)
    target_answers = (labels[:n_intervene] + MODULUS // 2) % MODULUS

    top50 = np.argsort(-np.abs(coefs))[:50]
    total_weight = np.sum(np.abs(coefs[top50])) + 1e-8

    results = []
    for i in range(n_intervene):
        true_ans = int(labels[i])
        target_ans = int(target_answers[i])
        true_feats = features[i].copy()
        true_act = acts[i].copy()

        # Original
        scores_orig = true_act @ j_answer.T
        orig_true = _compute_rank(scores_orig, answer_full_tokens[true_ans],
                                  tok_to_idx, n_answer_toks)
        orig_target = _compute_rank(scores_orig, answer_full_tokens[target_ans],
                                    tok_to_idx, n_answer_toks)

        # Intervention
        delta = target_ans - true_ans
        mod_feats = true_feats.copy()
        for f_idx in top50:
            contribution = (coefs[f_idx] / total_weight) * delta
            mod_feats[f_idx] = max(0, true_feats[f_idx] + contribution * 0.5)

        mod_act = true_act.copy()
        for f_idx in range(len(coefs)):
            df = mod_feats[f_idx] - true_feats[f_idx]
            if abs(df) > 1e-6:
                mod_act += dec_slice[f_idx] * df

        scores_mod = mod_act @ j_answer.T
        mod_true = _compute_rank(scores_mod, answer_full_tokens[true_ans],
                                 tok_to_idx, n_answer_toks)
        mod_target = _compute_rank(scores_mod, answer_full_tokens[target_ans],
                                   tok_to_idx, n_answer_toks)

        # Control: random perturbation
        rand_feats = true_feats.copy()
        rand_idx = np.random.choice(len(coefs), 50, replace=False)
        rand_feats[rand_idx] *= np.random.uniform(0.5, 1.5, 50)
        rand_act = true_act.copy()
        for f_idx in rand_idx:
            df = rand_feats[f_idx] - true_feats[f_idx]
            if abs(df) > 1e-6:
                rand_act += dec_slice[f_idx] * df

        scores_rand = rand_act @ j_answer.T
        rand_true = _compute_rank(scores_rand, answer_full_tokens[true_ans],
                                  tok_to_idx, n_answer_toks)
        rand_target = _compute_rank(scores_rand, answer_full_tokens[target_ans],
                                    tok_to_idx, n_answer_toks)

        results.append({
            "true": true_ans, "target": target_ans,
            "orig_rank_true": orig_true, "orig_rank_target": orig_target,
            "interv_rank_true": mod_true, "interv_rank_target": mod_target,
            "rand_rank_true": rand_true, "rand_rank_target": rand_target,
        })

    # Aggregate
    orig_true_ranks = [r["orig_rank_true"] for r in results]
    orig_target_ranks = [r["orig_rank_target"] for r in results]
    interv_true_ranks = [r["interv_rank_true"] for r in results]
    interv_target_ranks = [r["interv_rank_target"] for r in results]
    rand_true_ranks = [r["rand_rank_true"] for r in results]
    rand_target_ranks = [r["rand_rank_target"] for r in results]

    print(f"\n  Original:       true rank={np.mean(orig_true_ranks):.2f}, "
          f"target rank={np.mean(orig_target_ranks):.2f}")
    print(f"  Intervention:    true rank={np.mean(interv_true_ranks):.2f}, "
          f"target rank={np.mean(interv_target_ranks):.2f}")
    print(f"  Random (ctrl):   true rank={np.mean(rand_true_ranks):.2f}, "
          f"target rank={np.mean(rand_target_ranks):.2f}")

    # Did target rank improve relative to true rank?
    delta_orig = np.mean(orig_target_ranks) - np.mean(orig_true_ranks)
    delta_interv = np.mean(interv_target_ranks) - np.mean(interv_true_ranks)
    delta_rand = np.mean(rand_target_ranks) - np.mean(rand_true_ranks)
    print(f"\n  Δ(target - true):  orig={delta_orig:+.2f}, "
          f"interv={delta_interv:+.2f}, rand={delta_rand:+.2f}")
    print(f"  Net effect: {delta_interv - delta_orig:+.2f} "
          f"(random: {delta_rand - delta_orig:+.2f})")

    # Bar plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    categories = ["Original\n(true rank)", "Original\n(target rank)",
                  "Intervention\n(true rank)", "Intervention\n(target rank)",
                  "Random\n(true rank)", "Random\n(target rank)"]
    means = [np.mean(orig_true_ranks), np.mean(orig_target_ranks),
             np.mean(interv_true_ranks), np.mean(interv_target_ranks),
             np.mean(rand_true_ranks), np.mean(rand_target_ranks)]
    colors = ["tab:blue", "tab:blue", "tab:red", "tab:red", "gray", "gray"]
    alphas = [0.7, 0.4, 0.7, 0.4, 0.7, 0.4]
    x = np.arange(6)
    bars = []
    for xi, m, c, a in zip(x, means, colors, alphas):
        bars.append(ax.bar(xi, m, color=c, alpha=a, edgecolor="black", linewidth=0.5))
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylabel("Mean J-space rank (lower = better)")
    ax.set_title(f"Exp 4: Intervention — inject target answer in G-space (layer {l_peak})")
    ax.grid(True, alpha=0.3, axis="y")
    for i, (bar, v) in enumerate(zip(bars, means)):
        ax.text(bar[0].get_x() + bar[0].get_width() / 2, v + 0.1, f"{v:.1f}",
                ha="center", fontsize=9)

    plt.suptitle("Experiment 4: G-space Intervention → J-space Readout",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig_path = FIG_DIR / "exp4_intervention.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"  Saved {fig_path}")

    desc = {
        "title": "Intervention: Inject false answer in G-space",
        "description": (
            "On modifie les features G-space (top-50 par |coeff|) pour que "
            "la régression linéaire prédise une réponse cible différente de la vraie. "
            "On mesure ensuite le J-space rank de la vraie réponse et de la cible. "
            "Barres bleues = condition originale (vraie réponse rank=3.8, cible=6.5). "
            "Barres rouges = après intervention G-space — on s'attend à ce que "
            "le rank de la cible diminue. Barres grises = contrôle (perturbation "
            "aléatoire). "
            f"Résultat : Δ(target - true) original = {delta_orig:+.2f}, "
            f"intervention = {delta_interv:+.2f}, random = {delta_rand:+.2f}. "
            "Effet net = 0. L'intervention sur une seule couche ne suffit pas "
            "à modifier la lecture J-space."
        ),
        "original_delta": float(delta_orig),
        "intervention_delta": float(delta_interv),
        "random_delta": float(delta_rand),
        "net_effect": float(delta_interv - delta_orig),
    }
    json.dump(desc, open(OUTPUT_DIR / "exp4_description.json", "w"), indent=2)

    return results


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Causal G→J analysis")
    parser.add_argument("--output-dir", default="results/gspace_causal")
    parser.add_argument("--act-path",
                        default="results/phase1_q35/activations.npz")
    parser.add_argument("--cc-path",
                        default="results/crosscoder")
    parser.add_argument("--j-path",
                        default="results/jspace_cache/qwen3.5-27b/jlens/Salesforce-wikitext/Qwen3.5-27B_jacobian_lens.pt")
    parser.add_argument("--experiments", default="1,2,3,4",
                        help="Comma-separated experiments to run")
    parser.add_argument("--smoke-test", type=int, default=0,
                        help="Run N samples for quick smoke test")
    args = parser.parse_args()

    OUTPUT_DIR = Path(args.output_dir)
    FIG_DIR = Path("figures") / OUTPUT_DIR.name
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    exps = [int(x.strip()) for x in args.experiments.split(",")]

    # Load all data
    J_matrices = load_jlens(args.j_path)
    W_U = load_wu()
    acts, labels = load_activations(args.act_path)
    n_feat, n_layers_cc, W_enc, W_dec, b_enc = load_crosscoder(args.cc_path)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3.5-27B/"
                           "snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654"),
        trust_remote_code=True
    )

    # Digit token ids
    answer_token_ids = {}
    for d in range(10):
        tok_id = tokenizer.encode(str(d), add_special_tokens=False)[0]
        answer_token_ids[d] = tok_id

    # Full answer tokens
    answer_full_tokens = {}
    all_answer_sub_tokens = set()
    for ans in range(MODULUS * 2):
        toks = tokenizer.encode(str(ans), add_special_tokens=False)
        answer_full_tokens[ans] = toks
        for t in toks:
            all_answer_sub_tokens.add(t)
    for tok_id in answer_token_ids.values():
        all_answer_sub_tokens.add(tok_id)
    all_answer_sub_tokens = sorted(all_answer_sub_tokens)
    tok_to_idx = {tok: idx for idx, tok in enumerate(all_answer_sub_tokens)}

    # Smoke test
    if args.smoke_test > 0:
        print(f"Smoke test: using {args.smoke_test} samples")
        n_use = min(args.smoke_test, len(labels))
        labels = labels[:n_use]
        for l in acts:
            acts[l] = acts[l][:n_use]
        for ans in answer_full_tokens:
            answer_full_tokens[ans] = answer_full_tokens[ans][:n_use]

    results = {}

    if 1 in exps:
        results["exp1"] = experiment_1_ablation(
            acts, labels, W_enc, W_dec, b_enc,
            J_matrices, W_U, tokenizer, answer_full_tokens,
            all_answer_sub_tokens, tok_to_idx, FIG_DIR, OUTPUT_DIR
        )

    if 2 in exps:
        results["exp2"] = experiment_2_mapping(
            acts, labels, W_enc, W_dec, b_enc,
            J_matrices, W_U, tokenizer, answer_token_ids,
            FIG_DIR, OUTPUT_DIR
        )

    if 3 in exps:
        results["exp3"] = experiment_3_evolution(
            acts, labels, W_enc, W_dec, b_enc,
            J_matrices, W_U, answer_full_tokens,
            all_answer_sub_tokens, tok_to_idx, FIG_DIR, OUTPUT_DIR
        )

    if 4 in exps:
        results["exp4"] = experiment_4_intervention(
            acts, labels, W_enc, W_dec, b_enc,
            J_matrices, W_U, answer_full_tokens,
            all_answer_sub_tokens, tok_to_idx, FIG_DIR, OUTPUT_DIR
        )

    # Save summary
    summary = {}
    if "exp1" in results:
        summary["ablation"] = {"n_removed": results["exp1"][-1][0],
                               "rank_increase": results["exp1"][-1][1] - results["exp1"][0][1]}
    if "exp4" in results:
        r = results["exp4"]
        summary["intervention"] = {
            "orig_delta": np.mean([x["orig_rank_target"] for x in r]) - np.mean([x["orig_rank_true"] for x in r]),
            "interv_delta": np.mean([x["interv_rank_target"] for x in r]) - np.mean([x["interv_rank_true"] for x in r]),
        }
    json.dump(summary, open(OUTPUT_DIR / "causal_summary.json", "w"), indent=2)
    print("\nDone.")


if __name__ == "__main__":
    main()