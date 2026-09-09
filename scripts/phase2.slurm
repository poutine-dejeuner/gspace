#!/bin/bash
#SBATCH --job-name=gspace-ph2
#SBATCH --output=logs/phase2_%j.out
#SBATCH --error=logs/phase2_%j.err
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=2:00:00

source /mnt/home/jacob/vl/repos/gspace/.venv/bin/activate
cd /mnt/home/jacob/vl/repos/gspace
mkdir -p logs results/phase2

echo "=== Phase 2: Fixed-b and averaged-b embedding geometry ==="
echo "Job: $SLURM_JOB_ID | $(hostname) | $(date)"

python3 << 'EOF'
import json, os, random
import torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

model_id = 'Qwen/Qwen2.5-32B-Instruct'
modulus = 59
output_dir = 'results/phase2'
os.makedirs(output_dir, exist_ok=True)

print(f'Loading {model_id}...')
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map='auto',
    attn_implementation='eager',
)
BEST_LAYER = 61
device = 'cuda'
print(f'Using layer {BEST_LAYER}')

def make_chat_prompt(a, b):
    return f'Compute ({a} + {b}) mod 59. Answer with only the number.'

def get_activation(prompts):
    texts = [tok.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True)
             for p in prompts]
    inputs = tok(texts, return_tensors='pt', padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hs = outputs.hidden_states[BEST_LAYER + 1]
        last_pos = inputs['attention_mask'].sum(dim=1) - 1
        acts = hs[range(len(last_pos)), last_pos].cpu().float().numpy()
    return acts

# 1) Fixed b=0
print('\n=== Fixed b=0 ===')
fixed_b = {}
for a in tqdm(range(modulus), desc='  fixed b=0'):
    fixed_b[a] = get_activation([make_chat_prompt(a, 0)])[0]

# 2) Averaged over b
print('\n=== Averaged over b ===')
avg_b = {}
rng = random.Random(42)
for a in tqdm(range(modulus), desc='  avg over b'):
    b_vals = rng.sample(range(modulus), min(20, modulus))
    acts = get_activation([make_chat_prompt(a, b) for b in b_vals])
    avg_b[a] = acts.mean(axis=0)

del model; torch.cuda.empty_cache()

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
    return proj

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
proj_fixed = make_pca_plot(fixed_b, 'Fixed b=0: embedding of a', axes[0])
proj_avg = make_pca_plot(avg_b, 'Averaged over b: mean embedding of a', axes[1])
plt.suptitle(f'Qwen2.5-32B-Instruct, Layer {BEST_LAYER}, mod 59', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(f'{output_dir}/embedding_pca.png', dpi=150, bbox_inches='tight')
print(f'\nSaved to {output_dir}/embedding_pca.png')

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

for name, proj in [('Fixed b=0', proj_fixed), ('Averaged over b', proj_avg)]:
    cx, cy, r, err = fit_circle(proj[:, 0], proj[:, 1])
    print(f'\n{name}: center=({cx:.1f},{cy:.1f}) r={r:.1f} circle_err={err:.2f}')
    angles = np.arctan2(proj[:, 1]-cy, proj[:, 0]-cx)
    corr, p = spearmanr(range(modulus), np.argsort(angles))
    print(f'  Spearman r (residue vs angular order): {corr:.3f} (p={p:.4f})')

# Save embeddings
np.savez_compressed(f'{output_dir}/embeddings.npz',
    fixed_b=np.stack([fixed_b[r] for r in range(modulus)]),
    avg_b=np.stack([avg_b[r] for r in range(modulus)]),
)
print(f'\nDone! Embeddings saved to {output_dir}/embeddings.npz')
EOF

echo "Finished at $(date)"
