"""
Measure rotation of the modular arithmetic circle across linear-attention layers.

For each problem (a+b mod 59), extract the residual stream at layers 59,60,61,62,63
at the last token position. Find the circular subspace via PCA on per-class means,
then measure angular rotation between adjacent layers.

If the model rotates the circle, we expect:
- A low-D subspace (~2D) where class means form a circle at each layer
- Systematic angular offset between the same class at adjacent layers
  (e.g., class c at layer 61 is rotated by Δθ relative to class c at layer 60)
- The rotation angle should be consistent across classes within a layer pair
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJ = Path(__file__).resolve().parents[2]
FIG_DIR = PROJ / "figures" / "rotation_measure"
RES_DIR = PROJ / "results" / "rotation_measure"
MODEL_ID = "Qwen/Qwen3.5-27B"

MODULUS = 59
D_MODEL = 5120
N_TRAIN = 2400
N_TEST = 1081
BATCH_SIZE = 8
SEED = 42
# Focus on layers around the G/J peaks (59,60,61,62,63) 
# where 60,61,62 are linear attention
MEASURE_LAYERS = list(range(59, 64))  # 59-63


def make_problems(n_total):
    """Generate modular addition problems: a + b mod 59."""
    rng = np.random.default_rng(SEED)
    problems = []
    for _ in range(n_total):
        a = int(rng.integers(0, MODULUS))
        b = int(rng.integers(0, MODULUS))
        result = (a + b) % MODULUS
        prompt = f"Compute {a} + {b} mod {MODULUS}. The answer is"
        problems.append({"a": a, "b": b, "result": result, "prompt": prompt})
    return problems


def cache_activations(model, tokenizer, problems, layers, save_path, device):
    """Extract residual stream at last token for each problem across specified layers."""
    save_path = Path(save_path)
    if save_path.exists():
        print(f"Loading cached activations from {save_path}")
        data = np.load(save_path)
        acts = {l: data[f"l{l}"] for l in layers}
        labels = data["labels"]
        return acts, labels

    print(f"Caching activations for layers {layers}...")
    hidden_states = {}
    
    def make_hook(l):
        def hook(_mod, _inp, out):
            o = out[0] if isinstance(out, tuple) else out
            hidden_states[l] = o.detach().cpu().float()
        return hook

    hooks = []
    for l in layers:
        hooks.append(model.model.layers[l].register_forward_hook(make_hook(l)))

    all_acts = {l: [] for l in layers}
    all_labels = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(problems), BATCH_SIZE), desc="Caching"):
            batch = problems[i:i + BATCH_SIZE]
            prompts = [p["prompt"] for p in batch]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
            _ = model(**inputs)
            last_pos = (inputs["attention_mask"].sum(dim=1) - 1).cpu()
            for l in layers:
                hs = hidden_states[l]  # already float32 on CPU
                batch_acts = hs[range(len(last_pos)), last_pos].cpu().numpy()
                all_acts[l].append(batch_acts)
            all_labels.extend([p["result"] for p in batch])

    for h in hooks:
        h.remove()

    # Concatenate
    acts = {l: np.concatenate(all_acts[l], axis=0) for l in layers}
    labels = np.array(all_labels, dtype=np.int64)
    
    os.makedirs(save_path.parent, exist_ok=True)
    save_dict = {f"l{l}": acts[l] for l in layers}
    save_dict["labels"] = labels
    np.savez_compressed(save_path, **save_dict)
    print(f"Saved: {save_path}")
    for l in layers:
        print(f"  Layer {l}: {acts[l].shape}")
    return acts, labels


def compute_class_means(acts, labels, layer):
    """Compute mean activation for each result class (0..58) at a layer."""
    acts_l = acts[layer]
    means = np.zeros((MODULUS, D_MODEL), dtype=np.float32)
    counts = np.zeros(MODULUS, dtype=np.int32)
    for c in range(MODULUS):
        mask = labels == c
        if mask.sum() > 0:
            means[c] = acts_l[mask].mean(axis=0)
            counts[c] = mask.sum()
    return means, counts


def find_circular_subspace(class_means, n_components=3):
    """PCA on class means to find the subspace where they form a circle."""
    pca = PCA(n_components=n_components, random_state=SEED)
    projected = pca.fit_transform(class_means)  # (59, n_components)
    return projected, pca


def compute_angular_offset(proj_a, proj_b):
    """
    Align two circular embeddings via optimal rotation in 2D and measure per-class offset.
    
    Finds rotation R that minimizes ||proj_a - R @ proj_b||² (Procrustes in 2D),
    then computes per-class angular differences.
    """
    # Use first 2 PCs
    A = proj_a[:, :2]  # (59, 2)
    B = proj_b[:, :2]  # (59, 2)
    
    # Optimal rotation via SVD: find R such that A ≈ B @ R^T
    # Actually: find R that rotates B to match A: A ≈ R @ B
    # Procrustes: min ||A - B @ R^T||_F → R = U @ V^T  where U,S,V^T = SVD(A^T @ B)
    M = A.T @ B  # (2, 2)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt  # rotation matrix
    # Ensure proper rotation (det = +1)
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = U @ Vt
    
    # Rotate B to align with A
    B_aligned = B @ R.T  # (59, 2)
    
    # Compute per-class angles
    angles_a = np.arctan2(A[:, 1], A[:, 0])
    angles_b = np.arctan2(B_aligned[:, 1], B_aligned[:, 0])
    
    # Angular difference (mod 2π)
    diffs = (angles_a - angles_b) % (2 * np.pi)
    diffs[diffs > np.pi] -= 2 * np.pi  # wrap to [-π, π]
    
    # Global rotation angle (mean offset)
    # Use circular mean
    mean_sin = np.mean(np.sin(diffs))
    mean_cos = np.mean(np.cos(diffs))
    global_angle = np.arctan2(mean_sin, mean_cos)  # in radians
    
    return diffs, global_angle, R


def measure_layer_rotation(acts, labels, layers):
    """Main analysis: find circle in each layer and measure rotation between them."""
    print("\n" + "=" * 60)
    print("Measuring inter-layer rotation of the modular circle")
    print("=" * 60)
    
    # 1. Compute class means for each layer
    all_means = {}
    all_counts = {}
    for l in layers:
        means, counts = compute_class_means(acts, labels, l)
        all_means[l] = means
        all_counts[l] = counts
        print(f"Layer {l}: class means computed, min count={counts.min()}, mean count={counts.mean():.1f}")
    
    # 2. PCA on class means for each layer
    all_projections = {}
    all_pcas = {}
    for l in layers:
        proj, pca = find_circular_subspace(all_means[l], n_components=3)
        all_projections[l] = proj
        all_pcas[l] = pca
        evr = pca.explained_variance_ratio_
        print(f"Layer {l}: PCA var explained = PC1:{evr[0]:.3f} PC2:{evr[1]:.3f} PC3:{evr[2]:.3f} sum:{sum(evr[:3]):.3f}")
    
    # 3. Measure rotation between adjacent layers
    results = []
    layer_pairs = [(layers[i], layers[i+1]) for i in range(len(layers)-1)]
    
    for la, lb in layer_pairs:
        proj_a = all_projections[la]
        proj_b = all_projections[lb]
        diffs, global_angle, R_mat = compute_angular_offset(proj_a, proj_b)
        
        # Circular statistics
        mean_sin = np.mean(np.sin(diffs))
        mean_cos = np.mean(np.cos(diffs))
        r_bar = np.sqrt(mean_sin**2 + mean_cos**2)  # circular concentration (0-1)
        global_angle_deg = np.degrees(global_angle)
        std_deg = np.degrees(np.std(diffs))
        
        # Correlation between class index and angular offset (should be near 0 if rotation is class-independent)
        # If circle rotates as a whole, all classes have similar offset
        # Spearman between class and absolute offset
        r_spear, p_spear = stats.spearmanr(range(MODULUS), np.abs(diffs))
        
        result = {
            "layer_from": la,
            "layer_to": lb,
            "global_angle_rad": float(global_angle),
            "global_angle_deg": float(global_angle_deg),
            "circular_concentration": float(r_bar),
            "std_deg": float(std_deg),
            "spearman_r": float(r_spear),
            "spearman_p": float(p_spear),
            "per_class_diffs_deg": [float(np.degrees(d)) for d in diffs],
        }
        results.append(result)
        
        print(f"\nL{la} → L{lb}:")
        print(f"  Global rotation: {global_angle_deg:.1f}°  (concentration r={r_bar:.3f})")
        print(f"  Std of offsets: {std_deg:.1f}°")
        print(f"  Spearman r(class, offset): {r_spear:.4f} (p={p_spear:.4f})")
        
        # Also measure rotation relative to "perfect circle" — 
        # how much of the class ordering is preserved?
        # In a circle, class ordering around the circle should match modular ordering
        # Compute correlation between class index and angular position
        for ta, name in [(proj_a, f"L{la}"), (proj_b, f"L{lb}")]:
            angles = np.arctan2(ta[:, 1], ta[:, 0])
            r_circ, p_circ = stats.spearmanr(range(MODULUS), angles)
            # Also test circular-linear correlation
            cos_ang = np.cos(2 * np.pi * np.arange(MODULUS) / MODULUS)
            sin_ang = np.sin(2 * np.pi * np.arange(MODULUS) / MODULUS)
            r_cos = np.corrcoef(cos_ang, ta[:, 0])[0,1]
            r_sin = np.corrcoef(sin_ang, ta[:, 1])[0,1]
            print(f"  {name} circularity: r_cos={r_cos:.4f} r_sin={r_sin:.4f}")
    
    return results, all_projections, all_pcas


def plot_results(results, all_projections, all_pcas, args):
    """Visualize the rotation measurement."""
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)
    
    layers = MEASURE_LAYERS
    
    # Figure 1: PC1 vs PC2 for each layer (circle visualization)
    n_layers = len(layers)
    cols = min(5, n_layers)
    rows = (n_layers + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (l, proj) in enumerate(all_projections.items()):
        ax = axes[idx // cols][idx % cols]
        pc1, pc2 = proj[:, 0], proj[:, 1]
        
        # Color by class (mod 59 → hue)
        colors = plt.cm.hsv(np.arange(MODULUS) / MODULUS)
        ax.scatter(pc1, pc2, c=colors, s=60, edgecolors='black', linewidth=0.5)
        
        # Add class labels for a subset
        for c in range(0, MODULUS, 10):
            ax.annotate(str(c), (pc1[c], pc2[c]), fontsize=7, 
                       ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.7))
        
        evr = all_pcas.get(l, None)
        title = f"Layer {l}"
        if evr is not None:
            title += f"\nPC1:{evr.explained_variance_ratio_[0]:.2%} PC2:{evr.explained_variance_ratio_[1]:.2%}"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_aspect('equal')
    
    # Hide empty subplots
    for idx in range(n_layers, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)
    
    plt.suptitle("Mod 59 Circle: PC1 vs PC2 per Layer", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "circles_per_layer.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {FIG_DIR / 'circles_per_layer.png'}")
    
    # Figure 2: Rotation between adjacent layers (quiver plot)
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 5))
    if len(results) == 1:
        axes = [axes]
    
    for idx, (res, (la, lb)) in enumerate(zip(results, 
        [(MEASURE_LAYERS[i], MEASURE_LAYERS[i+1]) for i in range(len(MEASURE_LAYERS)-1)])):
        ax = axes[idx]
        proj_a = all_projections[la][:, :2]
        proj_b = all_projections[lb][:, :2]
        
        # Show the movement of each class
        colors = plt.cm.hsv(np.arange(MODULUS) / MODULUS)
        for c in range(MODULUS):
            ax.arrow(proj_a[c, 0], proj_a[c, 1],
                    proj_b[c, 0] - proj_a[c, 0],
                    proj_b[c, 1] - proj_a[c, 1],
                    head_width=0.02, head_length=0.03, fc=colors[c], ec=colors[c],
                    alpha=0.6, length_includes_head=True)
        
        ax.set_title(f"L{la} → L{lb}\nRotation: {res['global_angle_deg']:.1f}°\n"
                    f"Concentration r={res['circular_concentration']:.3f}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_aspect('equal')
    
    plt.suptitle("Class Movement Between Adjacent Layers", fontsize=14)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "layer_transitions.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {FIG_DIR / 'layer_transitions.png'}")
    
    # Figure 3: Angular offset per class for each layer pair
    n_pairs = len(results)
    cols = min(3, n_pairs)
    rows = (n_pairs + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, res in enumerate(results):
        ax = axes[idx // cols][idx % cols]
        diffs_deg = res["per_class_diffs_deg"]
        colors = plt.cm.hsv(np.arange(MODULUS) / MODULUS)
        ax.bar(range(MODULUS), diffs_deg, color=colors, edgecolor='black', linewidth=0.3)
        ax.axhline(y=res["global_angle_deg"], color='red', linestyle='--', 
                   label=f"Mean: {res['global_angle_deg']:.1f}°")
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax.set_title(f"L{res['layer_from']} → L{res['layer_to']}\n"
                    f"r={res['circular_concentration']:.3f}, std={res['std_deg']:.1f}°")
        ax.set_xlabel("Result class")
        ax.set_ylabel("Angular offset (°)")
        ax.legend(fontsize=8)
    
    for idx in range(n_pairs, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)
    
    plt.suptitle("Angular Offset per Class: Evidence for Rotation", fontsize=14)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "angular_offsets.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {FIG_DIR / 'angular_offsets.png'}")
    
    # Figure 4: Summary bar chart — global rotation angle per layer pair
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [f"L{r['layer_from']}→L{r['layer_to']}" for r in results]
    angles = [r["global_angle_deg"] for r in results]
    concentrations = [r["circular_concentration"] for r in results]
    
    bars = ax.bar(range(len(results)), angles, color=plt.cm.RdYlGn(np.array(concentrations)))
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Global rotation angle (°)")
    ax.set_title("Inter-layer Rotation of the Modular Circle")
    
    # Colorbar for concentration
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label="Circular concentration r")
    
    plt.tight_layout()
    fig.savefig(FIG_DIR / "rotation_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {FIG_DIR / 'rotation_summary.png'}")
    
    # Save numerical results
    out_path = RES_DIR / "rotation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke_test", type=int, default=0,
                       help="Use N samples for quick test (default: 0 = full run)")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    print("=" * 60)
    print("Inter-layer Rotation Measurement")
    print(f"Layers: {MEASURE_LAYERS}")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Generate problems
    n_total = N_TRAIN + N_TEST
    if args.smoke_test:
        n_total = args.smoke_test * 2
        print(f"SMOKE TEST: using {args.smoke_test} train, {args.smoke_test} test")
    problems = make_problems(n_total)

    # Load model
    print(f"Loading {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True,
        dtype=torch.bfloat16, device_map=device,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Cache activations
    cache_path = RES_DIR / "acts_multilayer.npz"
    acts, labels = cache_activations(model, tokenizer, problems, MEASURE_LAYERS, cache_path, device)

    # Measure rotation
    results, all_projections, all_pcas = measure_layer_rotation(acts, labels, MEASURE_LAYERS)

    # Plot
    plot_results(results, all_projections, all_pcas, args)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
