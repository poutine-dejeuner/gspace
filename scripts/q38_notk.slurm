#!/bin/bash
#SBATCH --job-name=gspace-q38b
#SBATCH --output=logs/q38b_%j.out
#SBATCH --error=logs/q38b_%j.err
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=1:30:00

source /mnt/home/jacob/vl/repos/gspace/.venv/bin/activate
cd /mnt/home/jacob/vl/repos/gspace
mkdir -p logs results

echo "=== Qwen3.8-27B NON-THINKING retry ==="
echo "Job: $SLURM_JOB_ID | $(hostname) | $(date)"

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

moduli = [23, 43, 59]
rng = random.Random(0)
all_results = {}

for mod in moduli:
    all_pairs = [(a, b) for a in range(mod) for b in range(mod)]
    test_pairs = rng.sample(all_pairs, min(100, len(all_pairs)))
    
    # 0-shot with enable_thinking=False
    correct = 0
    for a, b in tqdm(test_pairs, desc=f'  mod{mod}', leave=False):
        result = (a + b) % mod
        prompt = f'Compute ({a} + {b}) mod {mod}. Answer with only the number.'
        messages = [{'role': 'user', 'content': prompt}]
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False
        )
        inputs = tok(text, return_tensors='pt').to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tok.eos_token_id)
            full = tok.decode(out[0], skip_special_tokens=True)
            response = full[len(text):].strip() if text in full else full.split('assistant')[-1].strip()
            nums = re.findall(r'-?\d+', response)
            pred = int(nums[0]) if nums else None
            if pred == result:
                correct += 1
    acc = correct / len(test_pairs)
    
    # TF
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
        'direct': acc, 'dir_n': len(test_pairs),
        'tf_rank': sum(ranks)/len(ranks) if ranks else None,
        'tf_n': len(ranks),
    }
    print(f'  mod={mod:3d} | direct={acc:.3f} ({correct}/{len(test_pairs)}) | TF rank={all_results[f\"mod{mod}\"][\"tf_rank\"]:.1f}')

print(f'\n=== COMPARISON ===')
print(f'{\"Model\":<22} {\"mod23\":>8} {\"mod43\":>8} {\"mod59\":>8}')
for mod in moduli:
    k = f'mod{mod}'
    print(f'{\"Qwen3.8-27B (think)\":<22}  -  -  -')  # from previous run
print(f'Done.')
" 2>&1

echo "Finished at $(date)"
