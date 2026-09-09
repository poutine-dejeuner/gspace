#!/bin/bash
#SBATCH --job-name=gspace-q38
#SBATCH --output=logs/q38_%j.out
#SBATCH --error=logs/q38_%j.err
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=3:00:00

source /mnt/home/jacob/vl/repos/gspace/.venv/bin/activate
cd /mnt/home/jacob/vl/repos/gspace
mkdir -p logs results

sudo apt-get update -qq 2>/dev/null
sudo apt-get install -y -qq build-essential 2>&1 | tail -1

echo "=== Qwen3.8-27B modular addition eval ==="
echo "Job: $SLURM_JOB_ID | $(hostname) | $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

python -c "
import torch, random, json, re
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

model_name = 'Qwen/Qwen3.8-27B'
print(f'Loading {model_name}...')
tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map='auto',
    attn_implementation='eager', trust_remote_code=True
)

moduli = [16, 23, 31, 37, 43, 47, 53, 59, 61, 64]
rng = random.Random(0)
all_results = {}

for mod in moduli:
    all_pairs = [(a, b) for a in range(mod) for b in range(mod)]
    test_pairs = rng.sample(all_pairs, min(200, len(all_pairs)))
    
    # 0-shot direct
    correct = 0
    for a, b in tqdm(test_pairs, desc=f'  mod{mod} direct', leave=False):
        result = (a + b) % mod
        prompt = f'Compute ({a} + {b}) mod {mod}. Answer with only the number.'
        messages = [{'role': 'user', 'content': prompt}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors='pt').to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=tok.eos_token_id)
            full = tok.decode(out[0], skip_special_tokens=True)
            response = full[len(text):].strip() if text in full else full.split('assistant')[-1].strip()
            nums = re.findall(r'-?\d+', response)
            pred = int(nums[0]) if nums else None
            if pred == result:
                correct += 1
    acc_dir = correct / len(test_pairs)
    
    # Few-shot (5 examples)
    fs_examples = rng.sample(all_pairs, 5)
    correct_fs = 0
    for a, b in tqdm(test_pairs, desc=f'  mod{mod} fewshot', leave=False):
        result = (a + b) % mod
        parts = [f'Compute ({ea} + {eb}) mod {mod}. Answer: {(ea+eb)%mod}' for ea, eb in fs_examples]
        parts.append(f'Compute ({a} + {b}) mod {mod}. Answer:')
        prompt = '\n'.join(parts)
        messages = [{'role': 'user', 'content': prompt}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors='pt').to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=tok.eos_token_id)
            full = tok.decode(out[0], skip_special_tokens=True)
            response = full[len(text):].strip() if text in full else full.split('assistant')[-1].strip()
            nums = re.findall(r'-?\d+', response)
            pred = int(nums[0]) if nums else None
            if pred == result:
                correct_fs += 1
    acc_fs = correct_fs / len(test_pairs)
    
    # Teacher-force
    ranks = []
    for a, b in tqdm(test_pairs, desc=f'  mod{mod} tf', leave=False):
        result = (a + b) % mod
        prompt = f'{a} + {b} ='
        inputs = tok(prompt, return_tensors='pt').to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]
            ans_tokens = tok.encode(str(result), add_special_tokens=False)
            if len(ans_tokens) != 1:
                continue
            tid = ans_tokens[0]
            sorted_ids = logits.argsort(descending=True)
            rank = (sorted_ids == tid).nonzero(as_tuple=True)[0].item()
            ranks.append(rank)
    
    all_results[f'mod{mod}'] = {
        'direct': acc_dir, 'dir_n': len(test_pairs),
        'fewshot': acc_fs, 'fs_n': len(test_pairs),
        'tf_rank': sum(ranks)/len(ranks) if ranks else None,
        'tf_n': len(ranks),
    }
    print(f'  mod={mod:3d} | direct={acc_dir:.3f} ({correct}/{len(test_pairs)}) | fewshot={acc_fs:.3f} ({correct_fs}/{len(test_pairs)}) | TF rank={all_results[f\"mod{mod}\"][\"tf_rank\"]:.1f}')

with open('results/q38_sweep.json', 'w') as f:
    json.dump(all_results, f, indent=2)

# Compare with 32B and 72B
try:
    with open('results/mod_sweep.json') as f: d32 = json.load(f)
except: d32 = {}
try:
    with open('results/72b_sweep.json') as f: d72 = json.load(f)
except: d72 = {}

print(f'\n{\"=\"*80}')
print('Qwen3.8-27B vs Qwen2.5-32B vs Qwen2.5-72B (0-shot direct)')
print(f'{\"Modulus\":<10} {\"Q3.8-27B\":>10} {\"Q2.5-32B\":>10} {\"Q2.5-72B\":>10}')
print('-' * 50)
for mod in moduli:
    k = f'mod{mod}'
    v27 = all_results[k]['direct']
    v32 = d32.get(k, {}).get('direct_acc', d32.get(k, {}).get('direct', 0))
    v72 = d72.get(k, {}).get('direct', 0)
    print(f'{mod:<10} {v27:9.1%} {v32:9.1%} {v72:9.1%}')

print(f'Done at $(date)')
" 2>&1

echo "Finished at $(date)"
