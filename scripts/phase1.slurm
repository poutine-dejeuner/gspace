#!/bin/bash
#SBATCH --job-name=gspace-ph1
#SBATCH --output=logs/phase1_%j.out
#SBATCH --error=logs/phase1_%j.err
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=4:00:00

source /mnt/home/jacob/vl/repos/gspace/.venv/bin/activate
cd /mnt/home/jacob/vl/repos/gspace
mkdir -p logs results/phase1

echo "=== Phase 1: Activation Extraction + Linear Probes ==="
echo "Job: $SLURM_JOB_ID | $(hostname) | $(date)"
echo "Model: Qwen/Qwen2.5-32B-Instruct"

python3 -c "
import json, os, random, re
import torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import gc

# ------ Config ------
model_id = 'Qwen/Qwen2.5-32B-Instruct'
modulus = 59
op = '+'
n_train = 2400
n_test = 1081
output_dir = 'results/phase1'
seed = 42
batch_size = 8

os.makedirs(output_dir, exist_ok=True)
rng = random.Random(seed)

print(f'Loading {model_id}...')
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
device_map = 'auto'

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map=device_map,
    attn_implementation='eager',
)
n_layers = model.config.num_hidden_layers
d_model = model.config.hidden_size
print(f'Layers: {n_layers}, d_model: {d_model}, device: {model.device}')

# ------ Generate problems ------
all_pairs = [(a, b) for a in range(modulus) for b in range(modulus)]
rng.shuffle(all_pairs)
test_pairs = all_pairs[:n_test]
train_pairs = all_pairs[n_test:n_test + n_train]

def make_chat_prompt(a, b):
    return f'Compute ({a} + {b}) mod 59. Answer with only the number.'

def result_fn(a, b):
    return (a + b) % modulus

# ------ Cache all layers at once (fewer forward passes) ------
# Strategy: run each problem, capture ALL hidden states via output_hidden_states=True
print('Caching train activations (all layers via output_hidden_states)...')

all_train_acts = {l: [] for l in range(n_layers)}
all_train_labels = []

for i in tqdm(range(0, min(n_train, 1200), batch_size), desc='  train acts'):  # 1200 should be enough
    batch_pairs = train_pairs[i:i+batch_size]
    prompts = [make_chat_prompt(a, b) for a, b in batch_pairs]
    labels = [result_fn(a, b) for a, b in batch_pairs]
    
    # Use chat template
    texts = []
    for p in prompts:
        msgs = [{'role': 'user', 'content': p}]
        texts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    
    inputs = tok(texts, return_tensors='pt', padding=True).to('cuda')
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hs = outputs.hidden_states  # tuple of (n_layers+1) tensors [batch, seq, d_model]
        last_pos = inputs['attention_mask'].sum(dim=1) - 1
        for l in range(n_layers):
            all_train_acts[l].append(hs[l+1][range(len(last_pos)), last_pos].cpu().float().numpy())
    all_train_labels.extend(labels)

train_acts = {l: np.concatenate(all_train_acts[l], axis=0) for l in range(n_layers)}
train_labels_arr = np.array(all_train_labels)
print(f'Train activations: {train_acts[0].shape[0]} samples x {d_model} dims')

# ------ Cache test activations ------
print('Caching test activations...')
all_test_acts = {l: [] for l in range(n_layers)}
all_test_labels = []

for i in tqdm(range(0, n_test, batch_size), desc='  test acts'):
    batch_pairs = test_pairs[i:i+batch_size]
    prompts = [make_chat_prompt(a, b) for a, b in batch_pairs]
    labels = [result_fn(a, b) for a, b in batch_pairs]
    
    texts = []
    for p in prompts:
        msgs = [{'role': 'user', 'content': p}]
        texts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    
    inputs = tok(texts, return_tensors='pt', padding=True).to('cuda')
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hs = outputs.hidden_states
        last_pos = inputs['attention_mask'].sum(dim=1) - 1
        for l in range(n_layers):
            all_test_acts[l].append(hs[l+1][range(len(last_pos)), last_pos].cpu().float().numpy())
    all_test_labels.extend(labels)

test_acts = {l: np.concatenate(all_test_acts[l], axis=0) for l in range(n_layers)}
test_labels_arr = np.array(all_test_labels)
print(f'Test activations: {test_acts[0].shape[0]} samples x {d_model} dims')

# Free GPU memory
del model, outputs, hs
gc.collect()
torch.cuda.empty_cache()

# Save activations
cache_path = os.path.join(output_dir, 'activations.npz')
np.savez_compressed(
    cache_path,
    **{f'train_l{l}': train_acts[l] for l in range(n_layers)},
    **{f'test_l{l}': test_acts[l] for l in range(n_layers)},
    train_labels=train_labels_arr,
    test_labels=test_labels_arr,
)
print(f'Cached to {cache_path}')

# ------ Train probes at each layer ------
print('Training logistic probes at each layer...')
probe_results = {}

for l in tqdm(range(n_layers), desc='  probes'):
    X_train = train_acts[l]
    y_train = train_labels_arr
    X_test = test_acts[l]
    y_test = test_labels_arr
    
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
        'probe_weights': probe.coef_.tolist(),  # [59, d_model]
        'probe_intercepts': probe.intercept_.tolist(),
    }

# ------ Find best layer and analyze geometry ------
best_layer = max(probe_results, key=lambda l: probe_results[l]['test_acc'])
print(f'\nBest layer: {best_layer} (test acc: {probe_results[best_layer][\"test_acc\"]:.4f})')

# Per-residue mean activations at best layer
acts = train_acts[best_layer]
per_residue_means = {}
for r in range(modulus):
    mask = train_labels_arr == r
    if mask.sum() > 0:
        per_residue_means[r] = acts[mask].mean(axis=0)

residues = sorted(per_residue_means.keys())
mean_vectors = np.stack([per_residue_means[r] for r in residues])

# PCA
pca = PCA()
pca.fit(mean_vectors)
projected = pca.transform(mean_vectors)

# Cosine similarity of per-residue mean vectors
norms = np.linalg.norm(mean_vectors, axis=1, keepdims=True)
normalized = mean_vectors / (norms + 1e-8)
cos_sim = normalized @ normalized.T

# R² by k
from sklearn.linear_model import LinearRegression
r2_by_k = []
for k in range(1, min(11, d_model)):
    X = projected[:, :k]
    y = np.array(residues, dtype=float)
    reg = LinearRegression().fit(X, y)
    r2 = reg.score(X, y)
    r2_by_k.append(float(r2))
print(f'R2 by k: {[f\"{v:.4f}\" for v in r2_by_k[:5]]}...')

# ------ Save results ------
results = {
    'config': {'model_id': model_id, 'modulus': modulus, 'op': op, 'n_layers': n_layers, 'd_model': d_model},
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

out_path = os.path.join(output_dir, 'phase1_results.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)

# Save per-layer accuracies for plotting
layer_accs = [{'layer': l, **{k: v for k, v in probe_results[l].items() if k != 'probe_weights' and k != 'probe_intercepts'}} for l in range(n_layers)]
with open(os.path.join(output_dir, 'layer_accs.json'), 'w') as f:
    json.dump(layer_accs, f, indent=2)

print(f'\nDone! Results saved to {out_path}')
print(f'Layer accuracies: {os.path.join(output_dir, \"layer_accs.json\")}')
" 2>&1

echo "Finished at $(date)"
