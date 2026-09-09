#!/bin/bash
#SBATCH --job-name=gspace-xl
#SBATCH --output=logs/xl_%j.out
#SBATCH --error=logs/xl_%j.err
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=4:00:00

source /mnt/home/jacob/vl/repos/gspace/.venv/bin/activate
cd /mnt/home/jacob/vl/repos/gspace
mkdir -p logs results

# Install C compiler for Triton JIT
echo "Installing build-essential..."
sudo apt-get update -qq 2>/dev/null
sudo apt-get install -y -qq build-essential 2>&1 | tail -1

echo "=== Mod 59 Addition: XL Models (32B-72B) ==="
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"

python -c "
import torch, random, json, os, re, sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

modulus = 59
n_test = 200
rng = random.Random(0)
all_pairs = [(a, b) for a in range(modulus) for b in range(modulus)]
test_pairs = rng.sample(all_pairs, n_test)
few_shot_examples = [(0,0),(1,1),(10,20),(30,30),(58,1),(5,50),(25,25),(40,15),(7,13),(45,10)]

models_to_test = [
    ('Qwen/Qwen2.5-32B-Instruct', 'Qwen2.5-32B-Inst'),
    ('Qwen/QwQ-32B', 'QwQ-32B'),
    ('Qwen/Qwen2.5-72B-Instruct', 'Qwen2.5-72B-Inst'),
]

results = {}

for hf_id, name in models_to_test:
    print(f'\n{\"=\"*70}')
    print(f'{name} ({hf_id})')
    
    try:
        tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, torch_dtype=torch.bfloat16, device_map='auto',
            attn_implementation='eager'
        )
    except Exception as e:
        print(f'  LOAD FAILED: {e}')
        results[name] = {'error': str(e)}
        continue
    
    # Strategy 1: Direct instruction (0-shot)
    correct_direct = 0
    print(f'  Direct (0-shot)...')
    for a, b in tqdm(test_pairs[:50], desc='  Direct', leave=False):
        result = (a + b) % modulus
        prompt = f'Compute ({a} + {b}) mod {modulus}. Answer with only the number.'
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
                correct_direct += 1
    acc_direct = correct_direct / 50
    
    # Strategy 2: Few-shot (10 examples)
    correct_fewshot = 0
    print(f'  Few-shot (10-example)...')
    for a, b in tqdm(test_pairs, desc='  Few-shot', leave=False):
        result = (a + b) % modulus
        parts = [f'Compute ({ea} + {eb}) mod {modulus}. Answer: {(ea+eb)%modulus}' for ea, eb in few_shot_examples]
        parts.append(f'Compute ({a} + {b}) mod {modulus}. Answer:')
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
                correct_fewshot += 1
    acc_fewshot = correct_fewshot / n_test
    
    # Strategy 3: Teacher-force on bare format (logit rank)
    print(f'  Teacher-force...')
    ranks, probs = [], []
    for a, b in tqdm(test_pairs, desc='  TF', leave=False):
        result = (a + b) % modulus
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
    
    res = {
        'direct_acc': acc_direct,
        'fewshot_acc': acc_fewshot,
        'direct_n': 50, 'fewshot_n': n_test,
    }
    if ranks:
        res.update({
            'tf_mean_rank': sum(ranks)/len(ranks),
            'tf_mean_prob': sum(probs)/len(probs),
            'tf_top10': sum(1 for r in ranks if r < 10),
            'tf_top50': sum(1 for r in ranks if r < 50),
            'tf_n': len(ranks),
        })
        print(f'  TF: rank={res[\"tf_mean_rank\"]:.0f} prob={res[\"tf_mean_prob\"]:.6f} top10={res[\"tf_top10\"]}/{res[\"tf_n\"]}')
    else:
        res['tf_error'] = 'all multi-token'
        print(f'  TF: all multi-token')
    
    print(f'  Direct: {acc_direct:.4f} ({correct_direct}/50)')
    print(f'  Few-shot: {acc_fewshot:.4f} ({correct_fewshot}/{n_test})')
    
    results[name] = res
    torch.cuda.empty_cache()
    del model, tok

# Save
os.makedirs('results', exist_ok=True)
with open('results/xl_bench.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f'\n{\"=\"*70}')
print('FINAL SUMMARY:')
for name, r in results.items():
    print(f'\n{name}:')
    for k, v in r.items():
        if isinstance(v, float):
            print(f'  {k}: {v:.4f}')
        else:
            print(f'  {k}: {v}')

print(f'\nDone at $(date)')
" 2>&1

echo "Finished at $(date)"
