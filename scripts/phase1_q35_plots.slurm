#!/bin/bash
#SBATCH --job-name=gspace-p1p
#SBATCH --output=logs/phase1_plots_%j.out
#SBATCH --error=logs/phase1_plots_%j.err
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00

source /mnt/home/jacob/vl/repos/gspace/.venv/bin/activate
cd /mnt/home/jacob/vl/repos/gspace
mkdir -p logs results/phase1_q35

echo "=== Phase 1 plots: Qwen3.5-27B ==="

python3 << 'EOF'
import json, os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

modulus = 59
output_dir = 'results/phase1_q35'

# Load data
data = np.load(f'{output_dir}/activations.npz')
n_layers = 64
d_model = 5120
BEST_LAYER = 61

train_acts = data[f'train_l{BEST_LAYER}']
train_labels = data['train_labels']
print(f'Train acts shape: {train_acts.shape}')

# Per-residue means
per_residue_means = {}
for r in range(modulus):
    mask = train_labels == r
    if mask.sum() > 0:
        per_residue_means[r] = train_acts[mask].mean(axis=0)
residues = sorted(per_residue_means.keys())
mean_vectors = np.stack([per_residue_means[r] for r in residues])

# PCA full
pca_full = PCA()
pca_full.fit(mean_vectors)
projected_full = pca_full.transform(mean_vectors)

# Cosine similarity
norms = np.linalg.norm(mean_vectors, axis=1, keepdims=True)
normalized = mean_vectors / (norms + 1e-8)
cos_sim = normalized @ normalized.T

# R2 by k
r2_by_k = []
for k in range(1, min(21, d_model)):
    X = projected_full[:, :k]
    y = np.array(residues, dtype=float)
    reg = LinearRegression().fit(X, y)
    r2_by_k.append(float(reg.score(X, y)))

# Fixed b=0 and avg over b from train pairs
# Reconstruct pairs: load from the probe results or regenerate
# The activations don't store (a,b) pairs, so let's load from the original train script logic
# Actually, let's use the cached activations differently.
# We need to know which (a,b) each activation corresponds to.
# The train_pairs were generated with rng seed 42. Let's reconstruct.

import random
rng = random.Random(42)
all_pairs = [(a, b) for a in range(modulus) for b in range(modulus)]
test_pairs = rng.sample(all_pairs, 1000)
remaining = [(a, b) for a, b in all_pairs if (a, b) not in set(test_pairs)]
train_pairs = rng.sample(remaining, 1200)

fixed_b = {}
avg_b = {}
by_a_fixed = {a: [] for a in range(modulus)}
by_a_all = {a: [] for a in range(modulus)}

for idx, ((a, b), act) in enumerate(zip(train_pairs, train_acts)):
    if b == 0:
        by_a_fixed[a].append(act)
    by_a_all[a].append(act)

for a in range(modulus):
    fixed_b[a] = np.mean(by_a_fixed[a], axis=0) if by_a_fixed[a] else np.zeros(d_model)
    avg_b[a] = np.mean(by_a_all[a], axis=0) if by_a_all[a] else np.zeros(d_model)

# PCA plotting function
def make_pca_plot(embeddings, title, ax):
    residues = sorted(embeddings.keys())
    vectors = np.stack([embeddings[r] for r in residues])
    pca = PCA(n_components=2)
    proj = pca.fit_transform(vectors)
    v1, v2 = pca.explained_variance_ratio_ * 100
    
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=residues, cmap='twilight', s=80,
                    edgecolors='k', linewidth=0.5, zorder=5)
    for i, r in enumerate(residues):
        ax.annotate(str(r), (proj[i, 0], proj[i, 1]), fontsize=7, ha='center', va='center',
                    bbox=dict(boxstyle='circle,pad=0.15', fc='white', alpha=0.85))
    ax.plot(proj[:, 0], proj[:, 1], 'gray', alpha=0.3, linewidth=0.8, zorder=1)
    ax.plot([proj[-1, 0], proj[0, 0]], [proj[-1, 1], proj[0, 1]], 'gray', alpha=0.3, linewidth=0.8, zorder=1)
    ax.set_xlabel(f'PC1 ({v1:.1f}%)')
    ax.set_ylabel(f'PC2 ({v2:.1f}%)')
    ax.set_title(title)
    ax.set_aspect('equal')
    plt.colorbar(sc, ax=ax, label='Residue', shrink=0.8)
    return proj, v1, v2

# Load probe results for layer accuracy
with open(f'{output_dir}/layer_accs.json') as f:
    layer_accs = json.load(f)

# Plots
fig, axes = plt.subplots(2, 3, figsize=(21, 12))

# Layer accuracy
ax = axes[0, 0]
layers = list(range(n_layers))
test_accs = [a['test_acc'] for a in layer_accs]
test_w5 = [a['test_within5'] for a in layer_accs]
ax.plot(layers, test_accs, 'b-', lw=1.5, label='Test acc')
ax.plot(layers, test_w5, 'r-', lw=1.5, label='Test within-5')
ax.axvline(x=BEST_LAYER, color='green', ls='--', alpha=0.5, label=f'Best L{BEST_LAYER}')
ax.set_xlabel('Layer'); ax.set_ylabel('Accuracy')
ax.set_title('Linear probe accuracy by layer')
ax.legend(); ax.grid(True, alpha=0.3)

# Fixed b=0 PCA
proj_fixed, v1f, v2f = make_pca_plot(fixed_b, 'Fixed b=0: embedding of a', axes[0, 1])
axes[0, 1].set_title(f'Fixed b=0: embedding of a\nPC1+PC2 = {v1f+v2f:.1f}%')

# Averaged b PCA
proj_avg, v1a, v2a = make_pca_plot(avg_b, 'Averaged over b: mean embedding of a', axes[0, 2])
axes[0, 2].set_title(f'Averaged over b: mean embedding of a\nPC1+PC2 = {v1a+v2a:.1f}%')

# Cosine similarity
ax = axes[1, 0]
im = ax.imshow(cos_sim, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
ax.set_xlabel('Residue'); ax.set_ylabel('Residue')
ax.set_title('Cosine similarity (per-residue means)')
plt.colorbar(im, ax=ax)

# R2 by k
ax = axes[1, 1]
ks = list(range(1, len(r2_by_k)+1))
ax.plot(ks, r2_by_k, 'o-', markersize=4)
ax.axhline(y=0.5, color='gray', ls='--', alpha=0.5)
ax.axhline(y=0.9, color='gray', ls='--', alpha=0.5)
ax.set_xlabel('Number of PCs'); ax.set_ylabel('R²')
ax.set_title('Linear regression: residue = f(top-k PCs)')
ax.grid(True, alpha=0.3)

# Full PCA projection
ax = axes[1, 2]
sc = ax.scatter(projected_full[:, 0], projected_full[:, 1], c=residues, cmap='twilight', s=80,
                edgecolors='k', linewidth=0.5, zorder=5)
for i, r in enumerate(residues):
    ax.annotate(str(r), (projected_full[i, 0], projected_full[i, 1]), fontsize=7, ha='center', va='center',
                bbox=dict(boxstyle='circle,pad=0.15', fc='white', alpha=0.85))
ax.plot(projected_full[:, 0], projected_full[:, 1], 'gray', alpha=0.3, linewidth=0.8, zorder=1)
ax.plot([projected_full[-1, 0], projected_full[0, 0]], [projected_full[-1, 1], projected_full[0, 1]], 'gray', alpha=0.3, linewidth=0.8, zorder=1)
ax.set_xlabel(f'PC1 ({pca_full.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca_full.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title('Per-residue mean — PCA projection')
ax.set_aspect('equal')
plt.colorbar(sc, ax=ax, label='Residue', shrink=0.8)

plt.suptitle(f'Qwen3.5-27B — Linear Probes & Geometry (mod 59, best layer {BEST_LAYER})', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(f'{output_dir}/phase1_summary.png', dpi=150, bbox_inches='tight')
print(f'Saved to {output_dir}/phase1_summary.png')

# Circularity
from scipy.optimize import leastsq
from scipy.stats import spearmanr

def fit_circle(x, y):
    def resid(params):
        cx, cy, r = params
        return np.sqrt((x-cx)**2 + (y-cy)**2) - r
    cx0, cy0 = x.mean(), y.mean()
    r0 = np.sqrt((x-cx0)**2 + (y-cy0)**2).mean()
    params, _ = leastsq(resid, [cx0, cy0, r0])
    cx, cy, r = params
    err = np.mean(np.abs(resid(params)))
    return cx, cy, r, err

for name, proj in [('Fixed b=0', proj_fixed), ('Avg over b', proj_avg), 
                    ('Per-residue mean', projected_full)]:
    cx, cy, r, err = fit_circle(proj[:, 0], proj[:, 1])
    angles = np.arctan2(proj[:, 1]-cy, proj[:, 0]-cx)
    corr, p = spearmanr(range(modulus), np.argsort(angles))
    print(f'\n{name}:')
    print(f'  circle center=({cx:.1f},{cy:.1f}) r={r:.1f} err={err:.2f}')
    print(f'  Spearman r = {corr:.3f} (p={p:.4f})')

# Save full results
results = {
    'model_id': 'Qwen/Qwen3.5-27B', 'modulus': modulus, 'n_layers': n_layers, 'd_model': d_model,
    'best_layer': BEST_LAYER,
    'manifold': {
        'pca_explained_variance': pca_full.explained_variance_ratio_.tolist(),
        'pca_projection': projected_full.tolist(),
        'cosine_similarity': cos_sim.tolist(),
        'r2_by_k': r2_by_k,
    },
}
with open(f'{output_dir}/phase1_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f'\nR² at k=2: {r2_by_k[1]:.4f}, k=5: {r2_by_k[4]:.4f}, k=10: {r2_by_k[9]:.4f}')
print('Done!')
EOF
echo "Finished at $(date)"
