#!/bin/bash
#SBATCH --job-name=gspace-ph1b
#SBATCH --output=logs/phase1b_%j.out
#SBATCH --error=logs/phase1b_%j.err
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=2:00:00

source /mnt/home/jacob/vl/repos/gspace/.venv/bin/activate
cd /mnt/home/jacob/vl/repos/gspace
mkdir -p logs results/phase1

echo "=== Phase 1b: Probes only (activations cached) ==="
echo "Job: $SLURM_JOB_ID | $(hostname) | $(date)"

python3 -c "
import json, os, random
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

# Load cached activations
cache_path = 'results/phase1/activations.npz'
print(f'Loading cached activations from {cache_path}')
data = np.load(cache_path)

n_layers = 64
d_model = 5120
modulus = 59

train_acts = {l: data[f'train_l{l}'] for l in range(n_layers)}
test_acts = {l: data[f'test_l{l}'] for l in range(n_layers)}
train_labels = data['train_labels']
test_labels = data['test_labels']

print(f'Train: {train_acts[0].shape[0]} samples, Test: {test_acts[0].shape[0]} samples')

# ------ Train probes at each layer ------
print('Training logistic probes at each layer...')
probe_results = {}
rng = random.Random(42)

for l in tqdm(range(n_layers), desc='  probes'):
    X_train = train_acts[l]
    y_train = train_labels
    X_test = test_acts[l]
    y_test = test_labels
    
    # Subsample if needed
    if len(X_train) > 30000:
        idx = rng.sample(range(len(X_train)), 30000)
        X_train = X_train[idx]
        y_train = y_train[idx]
    
    probe = LogisticRegression(solver='lbfgs', max_iter=300, C=1.0)
    probe.fit(X_train, y_train)
    train_preds = probe.predict(X_train)
    test_preds = probe.predict(X_test)
    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    train_w5 = np.mean(np.abs(train_preds - y_train) <= 5)
    test_w5 = np.mean(np.abs(test_preds - y_test) <= 5)
    
    probe_results[l] = {
        'train_acc': float(train_acc),
        'test_acc': float(test_acc),
        'train_within5': float(train_w5),
        'test_within5': float(test_w5),
        'probe_weights': probe.coef_.tolist(),
        'probe_intercepts': probe.intercept_.tolist(),
    }

# ------ Find best layer ------
best_layer = max(probe_results, key=lambda l: probe_results[l]['test_acc'])
print(f'\nBest layer: {best_layer} (test acc: {probe_results[best_layer][\"test_acc\"]:.4f})')

# ------ Per-residue mean activations at best layer ------
acts = train_acts[best_layer]
per_residue_means = {}
for r in range(modulus):
    mask = train_labels == r
    if mask.sum() > 0:
        per_residue_means[r] = acts[mask].mean(axis=0)

residues = sorted(per_residue_means.keys())
mean_vectors = np.stack([per_residue_means[r] for r in residues])

# PCA
pca = PCA()
pca.fit(mean_vectors)
projected = pca.transform(mean_vectors)

# Cosine similarity
norms = np.linalg.norm(mean_vectors, axis=1, keepdims=True)
normalized = mean_vectors / (norms + 1e-8)
cos_sim = normalized @ normalized.T

# R² by k
from sklearn.linear_model import LinearRegression
r2_by_k = []
for k in range(1, min(21, d_model)):
    X = projected[:, :k]
    y = np.array(residues, dtype=float)
    reg = LinearRegression().fit(X, y)
    r2 = reg.score(X, y)
    r2_by_k.append(float(r2))
print(f'R2 by k (first 10): {[f\"{v:.4f}\" for v in r2_by_k[:10]]}')
print(f'R2 by k (last 10): {[f\"{v:.4f}\" for v in r2_by_k[-10:]]}')

# ------ Save results ------
results = {
    'config': {'model_id': 'Qwen/Qwen2.5-32B-Instruct', 'modulus': 59, 'op': '+', 'n_layers': 64, 'd_model': 5120},
    'best_layer': best_layer,
    'probe_results': {str(l): {k: v for k, v in r.items() if k not in ('probe_weights', 'probe_intercepts')}
                       for l, r in probe_results.items()},
    'manifold': {
        'pca_explained_variance': pca.explained_variance_ratio_.tolist(),
        'pca_projection': projected.tolist(),
        'cosine_similarity': cos_sim.tolist(),
        'r2_by_k': r2_by_k,
    },
}

out_path = 'results/phase1/phase1_results.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)

# Layer accuracies
layer_accs = [{'layer': l, 'train_acc': probe_results[l]['train_acc'], 'test_acc': probe_results[l]['test_acc'],
               'train_within5': probe_results[l]['train_within5'], 'test_within5': probe_results[l]['test_within5']}
              for l in range(n_layers)]
with open('results/phase1/layer_accs.json', 'w') as f:
    json.dump(layer_accs, f, indent=2)

# Quick summary
print(f'\n=== Layer Accuracy Summary ===')
for l in [0, 8, 16, 24, 32, 40, 48, 56, 63]:
    if l in probe_results:
        print(f'  Layer {l:3d}: train={probe_results[l][\"train_acc\"]:.4f}  test={probe_results[l][\"test_acc\"]:.4f}  w5={probe_results[l][\"test_within5\"]:.4f}')

print(f'\nDone!')
" 2>&1

echo "Finished at $(date)"
