#!/bin/bash
#SBATCH --job-name=gspace-mid
#SBATCH --output=logs/mid_%j.out
#SBATCH --error=logs/mid_%j.err
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

echo "=== Mid-size model sweep (14B-30B range) ==="
echo "Job: $SLURM_JOB_ID | $(hostname) | $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

python -c "
import torch, random, json, os, re, gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Mid-size models: find the threshold where modular addition emerges
models = [
    ('Qwen/Qwen2.5-14B-Instruct', 'Qwen2.5-14B'),
    ('Qwen/Qwen3-14B-Base', 'Qwen3-14B-Base'),
    ('Qwen/Qwen3-30B-A3B-Instruct', 'Qwen3-30B-A3B'),
]

moduli = [23, 43, 59, 64]
n_direct = 50
n_fs = 100
rng = random.Random(0)
all_results = {}

for model_name, label in models:
    print(f'\n{\"=\"*60}')
    print(f'Testing {label} ({model_name})')
    print('='*60)
    
    try:
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map='auto',
            attn_implementation='eager', trust_remote_code=True
        )
    except Exception as e:
        print(f'  SKIP: failed to load ({e})')
        all_results[label] = {'error': str(e)}
        continue
    
    mod_results = {}
    for mod in moduli:
        all_pairs = [(a, b) for a in range(mod) for b in range(mod)]
        test_pairs = rng.sample(all_pairs, min(n_direct, len(all_pairs)))
        fs_pairs = rng.sample(all_pairs, min(n_fs, len(all_pairs)))
        
        # Direct (0-shot)
        correct = 0
        for a, b in tqdm(test_pairs, desc=f'  mod{mod} direct', leave=False):
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
        acc_dir = correct / len(test_pairs)
        
        # Few-shot (5 examples)
        fs_examples = rng.sample(all_pairs, 5)
        correct_fs = 0
        for a, b in tqdm(fs_pairs, desc=f'  mod{mod} fewshot', leave=False):
            result = (a + b) % mod
            parts = [f'Compute ({ea} + {eb}) mod {mod}. Answer: {(ea+eb)%mod}' for ea, eb in fs_examples]
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
        acc_fs = correct_fs / len(fs_pairs)
        
        # Teacher-force
        ranks = []
        for a, b in tqdm(fs_pairs, desc=f'  mod{mod} tf', leave=False):
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
        
        mod_results[f'mod{mod}'] = {
            'direct': acc_dir, 'dir_n': len(test_pairs),
            'fewshot': acc_fs, 'fs_n': len(fs_pairs),
            'tf_rank': sum(ranks)/len(ranks) if ranks else None,
        }
        print(f'  mod={mod:3d} | direct={acc_dir:.3f} | fewshot={acc_fs:.3f} | TF rank={mod_results[f\"mod{mod}\"][\"tf_rank\"]}')
    
    all_results[label] = mod_results
    
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()

# Also load previous 32B results for comparison
with open('results/mod_sweep.json') as f:
    all_results['Qwen2.5-32B'] = json.load(f)
with open('results/bench_7b_plus.json') as f:
    prev = json.load(f)
    if 'Qwen2.5-7B-Instruct' in prev:
        all_results['Qwen2.5-7B'] = prev['Qwen2.5-7B-Instruct']

# Save
with open('results/mid_sweep.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print(f'\n{\"=\"*80}')
print('COMPARISON: Where does modular addition emerge?')
print(f'{\"Model\":<22} {\"mod23\":>8} {\"mod43\":>8} {\"mod59\":>8} {\"mod64\":>8}')
print('-' * 65)
for label in ['Qwen2.5-7B', 'Qwen2.5-14B', 'Qwen3-14B-Base', 'Qwen3-30B-A3B', 'Qwen2.5-32B']:
    r = all_results.get(label, {})
    if isinstance(r, dict) and 'error' not in r:
        d23 = r.get('mod23', r.get('mod23', {})).get('direct', 0) if isinstance(r.get('mod23', {}), dict) else 0
        d43 = r.get('mod43', r.get('mod43', {})).get('direct', 0) if isinstance(r.get('mod43', {}), dict) else 0
        d59 = r.get('mod59', r.get('mod59', {})).get('direct', 0) if isinstance(r.get('mod59', {}), dict) else 0
        d64 = r.get('mod64', r.get('mod64', {})).get('direct', 0) if isinstance(r.get('mod64', {}), dict) else 0
        print(f'{label:<22} {d23:7.1%} {d43:7.1%} {d59:7.1%} {d64:7.1%}')
    elif isinstance(r, dict) and 'error' in r:
        print(f'{label:<22} {\"ERR\":>8}')

print(f'Done at $(date)')
" 2>&1

echo "Finished at $(date)"
