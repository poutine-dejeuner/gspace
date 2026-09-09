#!/bin/bash
#SBATCH --job-name=gspace-js3
#SBATCH --output=logs/jspace_%j.out
#SBATCH --error=logs/jspace_%j.err
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=4:00:00

source /mnt/home/jacob/vl/repos/gspace/.venv/bin/activate
cd /mnt/home/jacob/vl/repos/gspace
mkdir -p logs results/jspace

echo "=== J-space sweep v3: generate, then measure at answer token ==="
echo "Job: $SLURM_JOB_ID | $(hostname) | $(date)"

python3 << 'EOF'
import json, os, random
import torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

model_id = 'Qwen/Qwen2.5-32B-Instruct'
modulus = 59
n_samples = 100
output_dir = 'results/jspace'
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
lm_head = model.lm_head
print(f'Layers: {n_layers}')

# Generate problems
all_pairs = [(a, b) for a in range(modulus) for b in range(modulus)]
test_pairs = rng.sample(all_pairs, n_samples)

results = {l: {'top1': 0, 'top5': 0, 'top10': 0, 'ranks': [], 'probs': []} for l in range(n_layers)}
n_valid = 0

for a, b in tqdm(test_pairs, desc='  J-space'):
    true_result = (a + b) % modulus
    prompt = f'Compute ({a} + {b}) mod 59. Answer with only the number.'
    text = tok.apply_chat_template(
        [{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True
    )
    inputs = tok(text, return_tensors='pt').to('cuda')
    prompt_len = inputs['input_ids'].shape[1]
    
    # Generate 5 tokens and capture hidden states at each generation step
    with torch.no_grad():
        gen_out = model.generate(
            **inputs, max_new_tokens=5, do_sample=False,
            output_hidden_states=True, return_dict_in_generate=True,
            pad_token_id=tok.eos_token_id,
        )
    
    # gen_out.hidden_states: tuple of n_generated_tokens tuples
    # Each inner tuple has (n_layers+1) tensors [batch, 1, d_model]
    if not gen_out.hidden_states or len(gen_out.hidden_states) == 0:
        continue
    
    # First generated token: this is where the model commits to its answer
    first_hs = gen_out.hidden_states[0]  # tuple of (n_layers+1) [1, 1, d_model]
    
    # Decode what model actually generated (first few tokens)
    generated_ids = gen_out.sequences[0, prompt_len:]
    # Just use the first generated token as the "answer token"
    answer_tid = generated_ids[0].item()
    
    for l in range(n_layers):
        h = first_hs[l + 1][0, 0]  # [d_model]
        logits = lm_head(h)  # [vocab]
        probs = torch.softmax(logits, dim=-1)
        
        # Rank of the model's OWN first generated token
        sorted_ids = logits.argsort(descending=True)
        rank = (sorted_ids == answer_tid).nonzero(as_tuple=True)[0].item()
        prob = probs[answer_tid].item()
        
        results[l]['ranks'].append(rank)
        results[l]['probs'].append(prob)
        
        # Also check: was it the CORRECT answer?
        # We'll filter by correctness separately
    
    n_valid += 1

del model; torch.cuda.empty_cache()

# Compute metrics
for l in range(n_layers):
    ranks = results[l]['ranks']
    results[l]['mean_rank'] = float(np.mean(ranks))
    results[l]['mean_prob'] = float(np.mean(results[l]['probs']))
    results[l]['top1'] = sum(1 for r in ranks if r == 0) / n_valid
    results[l]['top5'] = sum(1 for r in ranks if r < 5) / n_valid
    results[l]['top10'] = sum(1 for r in ranks if r < 10) / n_valid
    del results[l]['ranks']
    del results[l]['probs']

# Emergence
threshold = 0.3
emergence = None
for l in range(n_layers):
    if results[l]['top1'] >= threshold:
        emergence = l
        break

print(f'\nValid samples: {n_valid}')
print(f'Emergence (top1 > {threshold:.0%}): layer {emergence}')
print(f'L61 top1={results[61]["top1"]:.3f} top5={results[61]["top5"]:.3f} mean_rank={results[61]["mean_rank"]:.1f}')
print(f'L30 top1={results[30]["top1"]:.3f}')
print(f'L40 top1={results[40]["top1"]:.3f}')
print(f'L50 top1={results[50]["top1"]:.3f}')

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
layers = list(range(n_layers))

ax = axes[0, 0]
ax.plot(layers, [results[l]['top1'] for l in layers], 'b-', lw=1.5)
ax.axhline(y=0.54, color='gray', ls='--', alpha=0.5, label='Direct acc (54%)')
ax.axvline(x=61, color='red', ls='--', alpha=0.5, label='Probe best L61')
if emergence: ax.axvline(x=emergence, color='green', ls='--', alpha=0.5, label=f'Emergence L{emergence}')
ax.set_xlabel('Layer'); ax.set_ylabel('Top-1 (first gen token rank=0)')
ax.set_title('J-space: how often the generated token is top-1')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(layers, [results[l]['mean_rank'] for l in layers], 'r-', lw=1.5)
ax.set_xlabel('Layer'); ax.set_ylabel('Mean rank')
ax.set_title('J-space: mean rank of generated token')
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.plot(layers, [results[l]['top5'] for l in layers], 'orange', lw=1.5, label='Top-5')
ax.plot(layers, [results[l]['top10'] for l in layers], 'green', lw=1.5, label='Top-10')
ax.axvline(x=61, color='red', ls='--', alpha=0.5)
ax.set_xlabel('Layer'); ax.set_ylabel('Fraction')
ax.set_title('J-space: generated token in top-5 / top-10')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.plot(layers, [results[l]['mean_prob'] for l in layers], 'purple', lw=1.5)
ax.set_xlabel('Layer'); ax.set_ylabel('Mean prob')
ax.set_title('J-space: mean softmax prob of generated token')
ax.grid(True, alpha=0.3)

plt.suptitle(f'Qwen2.5-32B-Instruct — J-space at 1st generated token (n={n_valid})', fontsize=13)
plt.tight_layout()
plt.savefig(f'{output_dir}/jspace_sweep.png', dpi=150, bbox_inches='tight')
print(f'Saved to {output_dir}/jspace_sweep.png')

with open(f'{output_dir}/jspace_results.json', 'w') as f:
    json.dump({str(l): results[l] for l in range(n_layers)}, f, indent=2)

print(f'\n=== COMPARISON ===')
print(f'  J-space emergence: L{emergence}')
print(f'  Probe best:        L61')
print('Done!')
EOF
echo "Finished at $(date)"
