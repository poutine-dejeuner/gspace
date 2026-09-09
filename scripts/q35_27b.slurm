#!/bin/bash
#SBATCH --job-name=gspace-q35
#SBATCH --output=logs/q35_27b_%j.out
#SBATCH --error=logs/q35_27b_%j.err
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=1:00:00

source /mnt/home/jacob/vl/repos/gspace/.venv/bin/activate
cd /mnt/home/jacob/vl/repos/gspace
mkdir -p logs results

echo "=== Qwen3.5-27B modular addition benchmark ==="
echo "Job: $SLURM_JOB_ID | $(hostname) | $(date)"

python3 << 'EOF'
import json, random, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

model_id = 'Qwen/Qwen3.5-27B'
moduli = [23, 43, 59]
n_trials = 100
rng = random.Random(42)
output_dir = 'results'
os.makedirs(output_dir, exist_ok=True)

print(f'Loading {model_id}...')
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map='auto',
    attn_implementation='eager',
)
device = 'cuda'

def test_mod(m, n=100):
    pairs = rng.sample([(a, b) for a in range(m) for b in range(m)], n)
    correct = 0
    for a, b in pairs:
        true_res = (a + b) % m
        prompt = f'Compute ({a} + {b}) mod {m}. Answer with only the number.'
        text = tok.apply_chat_template(
            [{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tok(text, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=5, do_sample=False, pad_token_id=tok.eos_token_id)
        response = tok.decode(out[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        try:
            pred = int(response)
            if pred == true_res:
                correct += 1
        except:
            pass
    return correct / n

results = {}
for mod in moduli:
    acc = test_mod(mod, n_trials)
    results[str(mod)] = acc
    print(f'  mod {mod}: {acc:.2%}')

results['model'] = model_id
results['n_trials'] = n_trials

out_path = f'{output_dir}/q35_27b.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nSaved to {out_path}')

del model
torch.cuda.empty_cache()
print('Done!')
EOF

echo "Finished at $(date)"
