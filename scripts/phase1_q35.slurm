#!/bin/bash
#SBATCH --job-name=gspace-p1b
#SBATCH --output=logs/phase1_q35_%j.out
#SBATCH --error=logs/phase1_q35_%j.err
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=4:00:00

source /mnt/home/jacob/vl/repos/gspace/.venv/bin/activate
cd /mnt/home/jacob/vl/repos/gspace
mkdir -p logs results/phase1_q35

echo "=== Phase 1: Qwen3.5-27B — activation extraction + probes ==="
echo "Job: $SLURM_JOB_ID | $(hostname) | $(date)"

python3 << 'EOF'
import json, os, random
import torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

model_id = 'Qwen/Qwen3.5-27B'
modulus = 59
n_train = 1200
n_test = 1000
output_dir = 'results/phase1_q35'
os.makedirs(output_dir, exist_ok=True)
rng = random.Random(42)

print(f'Loading {model_id}...')
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map='auto',
    attn_implementation='eager',
)
model.config.output_hidden_states = True
n_layers = model.config.num_hidden_layers
d_model = model.config.hidden_size
print(f'Layers: {n_layers}, d_model: {d_model}')

# Generate problems
all_pairs = [(a, b) for a in range(modulus) for b in range(modulus)]
test_pairs = rng.sample(all_pairs, n_test)
remaining = [(a, b) for (a, b) in all_pairs if (a, b) not in set(test_pairs)]
train_pairs = rng.sample(remaining, n_train)

def extract_activations(pairs, desc='extract'):
    """Extract last-token residual stream at ALL layers simultaneously."""
    all_acts = {l: [] for l in range(n_layers)}
    all_labels = []
    
    batch_size = 4
    for i in tqdm(range(0, len(pairs), batch_size), desc=desc):
        batch_pairs = pairs[i:i+batch_size]
        prompts = []
        labels = []
        for a, b in batch_pairs:
            result = (a + b) % modulus
            prompts.append(f'Compute ({a} + {b}) mod 59. Answer with only the number.')
            labels.append(result)
        
        texts = [tok.apply_chat_template(
            [{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        ) for p in prompts]
        inputs = tok(texts, return_tensors='pt', padding=True).to('cuda')
        
        with torch.no_grad():
            outputs = model(**inputs)
            hs = outputs.hidden_states  # tuple of (n_layers+1) tensors [batch, seq, d_model]
            last_pos = inputs['attention_mask'].sum(dim=1) - 1
            
            for l in range(n_layers):
                acts = hs[l + 1][range(len(last_pos)), last_pos].cpu().float().numpy()
                all_acts[l].append(acts)
            all_labels.extend(labels)
    
    # Concatenate
    for l in range(n_layers):
        all_acts[l] = np.concatenate(all_acts[l], axis=0)
    return all_acts, np.array(all_labels, dtype=int)

print(f'\nExtracting train activations ({n_train} samples)...')
train_acts, train_labels = extract_activations(train_pairs, '  train')
print(f'Extracting test activations ({n_test} samples)...')
test_acts, test_labels = extract_activations(test_pairs, '  test')

del model; torch.cuda.empty_cache()

# Save raw
print(f'Saving activations...')
np.savez_compressed(f'{output_dir}/activations.npz',
    **{f'train_l{l}': train_acts[l] for l in range(n_layers)},
    **{f'test_l{l}': test_acts[l] for l in range(n_layers)},
    train_labels=train_labels, test_labels=test_labels,
)

# ------ Train probes at each layer ------
print('\nTraining logistic probes at each layer...')
probe_results = {}

for l in tqdm(range(n_layers), desc='  probes'):
    X_train = train_acts[l]
    y_train = train_labels
    X_test = test_acts[l]
    y_test = test_labels
    
    probe = LogisticRegression(solver='lbfgs', max_iter=500, C=1.0)
    probe.fit(X_train, y_train)
    train_preds = probe.predict(X_train)
    test_preds = probe.predict(X_test)
    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    train_w5 = np.mean(np.abs(train_preds - y_train) <= 5)
    test_w5 = np.mean(np.abs(test_preds - y_test) <= 5)
    
    probe_results[l] = {
        'train_acc': float(train_acc), 'test_acc': float(test_acc),
        'train_within5': float(train_w5), 'test_within5': float(test_w5),
    }

# Best layer
best_layer = max(probe_results, key=lambda l: probe_results[l]['test_acc'])
print(f'\nBest probe layer: {best_layer} (test acc: {probe_results[best_layer]["test_acc"]:.4f})')

# ------ Embedding geometry at best layer (fixed b=0 AND averaged) ------
print(f'\n=== Embedding geometry at layer {best_layer} ===')

# We already have train activations where b varies — let's extract proper slices
# But we computed per-pair, not per-a-fixed-b. Let's do targeted extraction.
# Strategy: use the benchmark already done. The activations are per (a,b).
# For fixed b=0: filter pairs where b=0
# For avg over b: group by a and average

# Fixed b=0
fixed_b = {}
avg_b = {}
for l in range(n_layers):
    if l != best_layer: continue
    acts = train_acts[l]
    by_a_fixed = {a: [] for a in range(modulus)}
    by_a_all = {a: [] for a in range(modulus)}
    for idx, ((a, b), act) in enumerate(zip(train_pairs, acts)):
        if b == 0:
            by_a_fixed[a].append(act)
        by_a_all[a].append(act)
    
    for a in range(modulus):
        if by_a_fixed[a]:
            fixed_b[a] = np.mean(by_a_fixed[a], axis=0)
        else:
            fixed_b[a] = np.zeros(d_model)  # fallback
        if by_a_all[a]:
            avg_b[a] = np.mean(by_a_all[a], axis=0)
        else:
            avg_b[a] = np.zeros(d_model)

# PCA plots
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

# Full PCA on BOTH methods
# Also do per-residue mean (all pairs) for cosine/R2 analysis
per_residue_means = {}
for r in range(modulus):
    mask = train_labels == r
    if mask.sum() > 0:
        per_residue_means[r] = train_acts[best_layer][mask].mean(axis=0)
residues = sorted(per_residue_means.keys())
mean_vectors = np.stack([per_residue_means[r] for r in residues])

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

# ------ Plots ------
fig, axes = plt.subplots(2, 3, figsize=(21, 12))

# Layer accuracy
ax = axes[0, 0]
layers = list(range(n_layers))
ax.plot(layers, [probe_results[l]['test_acc'] for l in layers], 'b-', lw=1.5, label='Test acc')
ax.plot(layers, [probe_results[l]['test_within5'] for l in layers], 'r-', lw=1.5, label='Test within-5')
ax.axvline(x=best_layer, color='green', ls='--', alpha=0.5, label=f'Best L{best_layer}')
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

# Full PCA projection (per-residue means)
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

plt.suptitle(f'Qwen3.5-27B — Linear Probes & Geometry (mod 59)', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(f'{output_dir}/phase1_summary.png', dpi=150, bbox_inches='tight')
print(f'Saved to {output_dir}/phase1_summary.png')

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

# ------ Save ------
results = {
    'model_id': model_id, 'modulus': modulus, 'n_layers': n_layers, 'd_model': d_model,
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

# Layer accuracies
layer_accs = [{'layer': l, **probe_results[l]} for l in range(n_layers)]
with open(f'{output_dir}/layer_accs.json', 'w') as f:
    json.dump(layer_accs, f, indent=2)

# Summary
print(f'\n=== Layer Accuracy Summary ===')
for l in [0, n_layers//4, n_layers//2, 3*n_layers//4, best_layer, n_layers-1]:
    if l in probe_results:
        print(f'  L{l:3d}: test={probe_results[l]["test_acc"]:.4f}  w5={probe_results[l]["test_within5"]:.4f}')

print(f'Best probe layer: {best_layer} (test acc: {probe_results[best_layer]["test_acc"]:.4f})')
print(f'R² at k=2: {r2_by_k[1]:.4f}, k=5: {r2_by_k[4]:.4f}, k=10: {r2_by_k[9]:.4f}')
print('Done!')
EOF

echo "Finished at $(date)"
