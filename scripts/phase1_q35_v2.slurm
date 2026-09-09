#!/bin/bash
#SBATCH --job-name=gspace-p1v2
#SBATCH --output=logs/phase1_q35_v2_%j.out
#SBATCH --error=logs/phase1_q35_v2_%j.err
#SBATCH --partition=batch
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2:00:00

source /mnt/home/jacob/vl/repos/gspace/.venv/bin/activate
cd /mnt/home/jacob/vl/repos/gspace
mkdir -p logs results/phase1_q35

echo "=== Phase 1 v2: probes + plots from cached activations ==="

python3 << 'EOF'
import json, os, random
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

modulus = 59
n_layers = 64
d_model = 5120
output_dir = 'results/phase1_q35'

# Load cached activations
data = np.load(f'{output_dir}/activations.npz')
train_acts = {l: data[f'train_l{l}'] for l in range(n_layers)}
test_acts = {l: data[f'test_l{l}'] for l in range(n_layers)}
train_labels = data['train_labels']
test_labels = data['test_labels']
print(f'Train: {train_acts[0].shape[0]}, Test: {test_acts[0].shape[0]}')

# ------ Train probes ------
print('Training probes...')
probe_results = {}
for l in tqdm(range(n_layers)):
    X_train, y_train = train_acts[l], train_labels
    X_test, y_test = test_acts[l], test_labels
    probe = LogisticRegression(solver='lbfgs', max_iter=500, C=1.0)
    probe.fit(X_train, y_train)
    train_preds = probe.predict(X_train)
    test_preds = probe.predict(X_test)
    probe_results[l] = {
        'train_acc': float(accuracy_score(y_train, train_preds)),
        'test_acc': float(accuracy_score(y_test, test_preds)),
        'train_within5': float(np.mean(np.abs(train_preds - y_train) <= 5)),
        'test_within5': float(np.mean(np.abs(test_preds - y_test) <= 5)),
    }

best_layer = max(probe_results, key=lambda l: probe_results[l]['test_acc'])
print(f'Best layer: {best_layer} (test acc: {probe_results[best_layer]["test_acc"]:.4f})')

# Save layer accs
layer_accs = [{'layer': l, **probe_results[l]} for l in range(n_layers)]
with open(f'{output_dir}/layer_accs.json', 'w') as f:
    json.dump(layer_accs, f, indent=2)

# ------ Embedding geometry ------
acts = train_acts[best_layer]

# Reconstruct pairs (same seed as extraction)
rng = random.Random(42)
all_pairs = [(a, b) for a in range(modulus) for b in range(modulus)]
test_pairs_set = set(rng.sample(all_pairs, 1000))
remaining = [(a, b) for a, b in all_pairs if (a, b) not in test_pairs_set]
train_pairs = rng.sample(remaining, 1200)

# Fixed b=0 and avg over b
by_a_fixed = {a: [] for a in range(modulus)}
by_a_all = {a: [] for a in range(modulus)}
for (a, b), act in zip(train_pairs, acts):
    if b == 0:
        by_a_fixed[a].append(act)
    by_a_all[a].append(act)

fixed_b = {a: np.mean(by_a_fixed[a], axis=0) if by_a_fixed[a] else np.zeros(d_model) for a in range(modulus)}
avg_b = {a: np.mean(by_a_all[a], axis=0) if by_a_all[a] else np.zeros(d_model) for a in range(modulus)}

# Per-residue means
per_residue_means = {}
for r in range(modulus):
    mask = train_labels == r
    if mask.sum() > 0:
        per_residue_means[r] = acts[mask].mean(axis=0)
residues = sorted(per_residue_means.keys())
mean_vectors = np.stack([per_residue_means[r] for r in residues])

# PCA
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
    reg = LinearRegression().fit(projected_full[:, :k], np.array(residues, dtype=float))
    r2_by_k.append(float(reg.score(projected_full[:, :k], np.array(residues, dtype=float))))

# ------ Plots ------
def make_pca_plot(embeddings, title, ax):
    residues_sorted = sorted(embeddings.keys())
    vectors = np.stack([embeddings[r] for r in residues_sorted])
    pca = PCA(n_components=2)
    proj = pca.fit_transform(vectors)
    v1, v2 = pca.explained_variance_ratio_ * 100
    
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=residues_sorted, cmap='twilight', s=80,
                    edgecolors='k', linewidth=0.5, zorder=5)
    for i, r in enumerate(residues_sorted):
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

fig, axes = plt.subplots(2, 3, figsize=(21, 12))

# 1) Layer accuracy
ax = axes[0, 0]
layers = list(range(n_layers))
ax.plot(layers, [probe_results[l]['test_acc'] for l in layers], 'b-', lw=1.5, label='Test acc')
ax.plot(layers, [probe_results[l]['test_within5'] for l in layers], 'r-', lw=1.5, label='Test within-5')
ax.axvline(x=best_layer, color='green', ls='--', alpha=0.5, label=f'Best L{best_layer}')
ax.set_xlabel('Layer'); ax.set_ylabel('Accuracy')
ax.set_title('Linear probe accuracy by layer')
ax.legend(); ax.grid(True, alpha=0.3)

# 2) Fixed b=0
proj_fixed, v1f, v2f = make_pca_plot(fixed_b, 'Fixed b=0: embedding of a', axes[0, 1])
axes[0, 1].set_title(f'Fixed b=0: embedding of a\nPC1+PC2 = {v1f+v2f:.1f}%')

# 3) Averaged b
proj_avg, v1a, v2a = make_pca_plot(avg_b, 'Averaged over b: mean embedding of a', axes[0, 2])
axes[0, 2].set_title(f'Averaged over b: mean embedding of a\nPC1+PC2 = {v1a+v2a:.1f}%')

# 4) Cosine similarity
ax = axes[1, 0]
im = ax.imshow(cos_sim, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
ax.set_xlabel('Residue'); ax.set_ylabel('Residue')
ax.set_title('Cosine similarity (per-residue means)')
plt.colorbar(im, ax=ax)

# 5) R2 by k
ax = axes[1, 1]
ax.plot(range(1, len(r2_by_k)+1), r2_by_k, 'o-', markersize=4)
ax.axhline(y=0.5, color='gray', ls='--', alpha=0.5)
ax.axhline(y=0.9, color='gray', ls='--', alpha=0.5)
ax.set_xlabel('Number of PCs'); ax.set_ylabel('R²')
ax.set_title('Linear regression: residue = f(top-k PCs)')
ax.grid(True, alpha=0.3)

# 6) Per-residue PCA
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

plt.suptitle(f'Qwen3.5-27B — Linear Probes & Geometry (mod 59, best layer {best_layer})', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(f'{output_dir}/phase1_summary.png', dpi=150, bbox_inches='tight')
print(f'\nSaved to {output_dir}/phase1_summary.png')

# ------ Circularity ------
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
    return cx, cy, r, np.mean(np.abs(resid(params)))

for name, proj in [('Fixed b=0', proj_fixed), ('Avg over b', proj_avg), 
                    ('Per-residue mean', projected_full)]:
    cx, cy, r, err = fit_circle(proj[:, 0], proj[:, 1])
    angles = np.arctan2(proj[:, 1]-cy, proj[:, 0]-cx)
    corr, p = spearmanr(range(modulus), np.argsort(angles))
    print(f'\n{name}:')
    print(f'  center=({cx:.1f},{cy:.1f}) r={r:.1f} err={err:.2f}')
    print(f'  Spearman r = {corr:.3f} (p={p:.4f})')

# ------ Layer summary ------
print(f'\n=== Layer Accuracy Summary ===')
for l in [0, 16, 32, 48, 56, best_layer, 63]:
    print(f'  L{l:3d}: test={probe_results[l]["test_acc"]:.4f}  w5={probe_results[l]["test_within5"]:.4f}')

print(f'\nBest probe layer: {best_layer}')
print(f'R²: k=2: {r2_by_k[1]:.4f}, k=5: {r2_by_k[4]:.4f}, k=10: {r2_by_k[9]:.4f}')

# Save results
results = {
    'model_id': 'Qwen/Qwen3.5-27B', 'modulus': modulus,
    'best_layer': best_layer,
    'probe_results': {str(l): probe_results[l] for l in range(n_layers)},
    'manifold': {
        'pca_explained_variance': pca_full.explained_variance_ratio_.tolist(),
        'pca_projection': projected_full.tolist(),
        'cosine_similarity': cos_sim.tolist(),
        'r2_by_k': r2_by_k,
    },
}
with open(f'{output_dir}/phase1_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('Done!')
EOF
echo "Finished at $(date)"
