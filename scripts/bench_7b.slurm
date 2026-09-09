#!/bin/bash
#SBATCH --job-name=gspace-bench
#SBATCH --output=logs/bench_%j.out
#SBATCH --error=logs/bench_%j.err
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00

source /mnt/home/jacob/vl/repos/gspace/.venv/bin/activate
cd /mnt/home/jacob/vl/repos/gspace
mkdir -p logs results

# Install C compiler for Triton JIT
echo "Installing build-essential..."
sudo apt-get update -qq && sudo apt-get install -y -qq build-essential 2>&1 | tail -1
echo "gcc: $(which gcc)"

echo "=== Mod 59 Addition Benchmark on >7B Models ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"

python -c "
import torch, random, json, os, re
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

modulus = 59
n_test = 100

# Candidate models (>7B, non-gated)
models_to_test = [
    ('Qwen/Qwen2.5-7B-Instruct', 'Qwen2.5-7B-Inst'),
    ('Qwen/Qwen2.5-Math-7B-Instruct', 'Qwen2.5-Math-7B-Inst'),
    ('deepseek-ai/deepseek-math-7b-instruct', 'DeepSeek-Math-7B-Inst'),
    ('mistralai/Mistral-7B-Instruct-v0.3', 'Mistral-7B-Inst-v0.3'),
    ('microsoft/phi-4', 'Phi-4'),
]

rng = random.Random(0)
all_pairs = [(a, b) for a in range(modulus) for b in range(modulus)]
test_pairs = rng.sample(all_pairs, n_test)

# Few-shot examples (fixed)
few_shot_examples = [(0,0),(1,1),(10,20),(30,30),(58,1),(5,50),(25,25),(40,15),(7,13),(45,10)]

results = {}

for hf_id, name in models_to_test:
    print(f'\n{\"=\"*60}')
    print(f'Testing {name} ({hf_id})')
    
    try:
        tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        # Use eager attention to avoid Triton JIT on B300
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, torch_dtype=torch.bfloat16, device_map='auto',
            attn_implementation='eager'
        )
    except Exception as e:
        print(f'  LOAD FAILED: {e}')
        results[name] = {'error': str(e)}
        continue
    
    # Strategy 1: Direct instruction
    correct_direct = 0
    print(f'\n  --- Direct instruction ---')
    for a, b in tqdm(test_pairs[:30], desc='  Direct'):
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
    
    acc_direct = correct_direct / 30
    print(f'  Direct accuracy: {acc_direct:.4f}')
    
    # Strategy 2: Few-shot in-context
    correct_fewshot = 0
    print(f'\n  --- Few-shot (10 examples) ---')
    
    def make_few_shot(a, b, examples):
        parts = [f'Compute ({ea} + {eb}) mod {modulus}. Answer: {(ea+eb)%modulus}' for ea, eb in examples]
        parts.append(f'Compute ({a} + {b}) mod {modulus}. Answer:')
        return '\n'.join(parts)
    
    for a, b in tqdm(test_pairs, desc='  Few-shot'):
        result = (a + b) % modulus
        prompt = make_few_shot(a, b, few_shot_examples)
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
    print(f'  Few-shot accuracy: {acc_fewshot:.4f}')
    
    # Strategy 3: Teacher-force (logit rank of correct answer)
    print(f'\n  --- Teacher-force ---')
    ranks = []
    probs = []
    for a, b in tqdm(test_pairs, desc='  TF'):
        result = (a + b) % modulus
        # Use bare format for teacher-force
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
    
    if ranks:
        results[name] = {
            'direct_acc': acc_direct,
            'fewshot_acc': acc_fewshot,
            'tf_mean_rank': sum(ranks)/len(ranks),
            'tf_mean_prob': sum(probs)/len(probs),
            'tf_top10': sum(1 for r in ranks if r < 10),
            'tf_top50': sum(1 for r in ranks if r < 50),
            'tf_n': len(ranks),
        }
        print(f'  TF: mean_rank={results[name][\"tf_mean_rank\"]:.0f}, mean_prob={results[name][\"tf_mean_prob\"]:.6f}, '
              f'top10={results[name][\"tf_top10\"]}/{results[name][\"tf_n\"]}')
    else:
        results[name] = {'direct_acc': acc_direct, 'fewshot_acc': acc_fewshot, 'tf_error': 'all multi-token'}
        print(f'  TF: all multi-token')
    
    del model, tok
    torch.cuda.empty_cache()

# Save results
os.makedirs('results', exist_ok=True)
with open('results/bench_7b_plus.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f'\n{\"=\"*60}')
print('SUMMARY:')
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
