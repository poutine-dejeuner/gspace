#!/bin/bash
#SBATCH --job-name=gspace-mod
#SBATCH --output=logs/mod_%j.out
#SBATCH --error=logs/mod_%j.err
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2:00:00

source /mnt/home/jacob/vl/repos/gspace/.venv/bin/activate
cd /mnt/home/jacob/vl/repos/gspace
mkdir -p logs results

echo "Installing build-essential..."
sudo apt-get update -qq 2>/dev/null
sudo apt-get install -y -qq build-essential 2>&1 | tail -1

echo "=== Mod Addition: modulus sweep on Qwen2.5-32B-Instruct ==="
echo "Job: $SLURM_JOB_ID | Node: $(hostname) | $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

python -c "
import torch, random, json, os, re
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

print('Loading Qwen2.5-32B-Instruct...')
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-32B-Instruct')
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-32B-Instruct', torch_dtype=torch.bfloat16, device_map='auto',
    attn_implementation='eager'
)

moduli = [16, 23, 31, 37, 43, 47, 53, 59, 61, 64]
n_test = 100  # per modulus
rng = random.Random(0)

results = {}

for mod in moduli:
    all_pairs = [(a, b) for a in range(mod) for b in range(mod)]
    test_pairs = rng.sample(all_pairs, min(n_test, len(all_pairs)))
    
    # 0-shot direct
    correct = 0
    for a, b in tqdm(test_pairs[:50], desc=f'  mod{mod} direct', leave=False):
        result = (a + b) % mod
        prompt = f'Compute ({a} + {b}) mod {mod}. Answer with only the number.'
        messages = [{'role': 'user', 'content': prompt}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors='pt').to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tok.eos_token_id)
            full = tok.decode(out[0], skip_special_tokens=True)
            response = full[len(text):].strip() if text in full else full.split('assistant')[-1].strip()
            nums = re.findall(r'\d+', response)
            pred = int(nums[0]) if nums else None
            if pred == result:
                correct += 1
    acc_direct = correct / 50
    
    # Few-shot (5 examples)
    few_shot_examples = rng.sample(all_pairs, 5)
    correct_fs = 0
    for a, b in tqdm(test_pairs, desc=f'  mod{mod} fewshot', leave=False):
        result = (a + b) % mod
        parts = [f'Compute ({ea} + {eb}) mod {mod}. Answer: {(ea+eb)%mod}' for ea, eb in few_shot_examples]
        parts.append(f'Compute ({a} + {b}) mod {mod}. Answer:')
        prompt = '\n'.join(parts)
        messages = [{'role': 'user', 'content': prompt}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors='pt').to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tok.eos_token_id)
            full = tok.decode(out[0], skip_special_tokens=True)
            response = full[len(text):].strip() if text in full else full.split('assistant')[-1].strip()
            nums = re.findall(r'\d+', response)
            pred = int(nums[0]) if nums else None
            if pred == result:
                correct_fs += 1
    acc_fewshot = correct_fs / len(test_pairs)
    
    # Teacher-force (bare format)
    ranks, probs = [], []
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
            prob = torch.softmax(logits, dim=-1)[tid].item()
            ranks.append(rank)
            probs.append(prob)
    
    results[f'mod{mod}'] = {
        'direct_acc': acc_direct,
        'fewshot_acc': acc_fewshot,
        'tf_mean_rank': sum(ranks)/len(ranks) if ranks else None,
        'tf_mean_prob': sum(probs)/len(probs) if probs else None,
        'tf_top10': sum(1 for r in ranks if r < 10) if ranks else 0,
        'tf_n': len(ranks),
    }
    
    print(f'  mod={mod:3d} | direct={acc_direct:.3f} ({correct}/50) | fewshot={acc_fewshot:.3f} ({correct_fs}/{len(test_pairs)}) | TF rank={results[f\"mod{mod}\"][\"tf_mean_rank\"]:.1f} prob={results[f\"mod{mod}\"][\"tf_mean_prob\"]:.6f}')

# Save
os.makedirs('results', exist_ok=True)
with open('results/mod_sweep.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f'\n{\"=\"*70}')
print('SWEEP SUMMARY:')
print(f'{\"Modulus\":<10} {\"Direct\":<10} {\"Few-shot\":<10} {\"TF rank\":<10} {\"TF prob\":<10}')
print('-' * 60)
for mod in moduli:
    r = results[f'mod{mod}']
    print(f'{mod:<10} {r[\"direct_acc\"]:<10.4f} {r[\"fewshot_acc\"]:<10.4f} {r[\"tf_mean_rank\"] or \"-\":<10} {r[\"tf_mean_prob\"] or \"-\":<10}')

print(f'Done at $(date)')
" 2>&1

echo "Finished at $(date)"
