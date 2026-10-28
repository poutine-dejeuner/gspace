"""Test if enable_thinking=False matters for sequential addition."""
import torch, sys
sys.path.insert(0, '.')
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = 'Qwen/Qwen3.5-27B'
MODULUS = 59

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, trust_remote_code=True,
    dtype=torch.bfloat16, device_map='cuda:0',
    attn_implementation='eager',
)
model.eval()
model.generation_config.update(enable_thinking=False)  # Force thinking off

# Try multiple formats
formats = [
    'Modular addition: {} modulo {}. Result:',
    'Compute the sum modulo {} of {}. The answer is',
    'What is {} modulo {}?',
]

cases = [
    ([3, 15], 18),
    ([3, 15, 8], 26),
    ([15, 42, 10], 8),
    ([30, 30], 1),
    ([58, 1], 0),
    ([5, 7], 12),
    ([30, 40], 11),
]

print("Testing with enable_thinking=False")
for fmt in formats:
    fmt_name = fmt.split('{')[0][:30]
    print(f"\n--- Format: '{fmt_name}...' ---")
    for nums, expected in cases:
        n = len(nums)
        if '{}' in fmt:
            # Check how many {} placeholders
            count = fmt.count('{}')
            if count == 1:
                # Single placeholder - put numbers and modulus
                nums_str = ' + '.join(str(n) for n in nums)
                prompt = fmt.format(nums_str, MODULUS) if count == 2 else fmt.format(nums_str)
            elif count == 2:
                nums_str = ' '.join(str(n) for n in nums)
                prompt = fmt.format(MODULUS, nums_str)
        else:
            prompt = fmt.format(*(nums + [MODULUS]))
        
        encoded = tokenizer(prompt, return_tensors='pt').to('cuda:0')
        with torch.no_grad():
            generated = model.generate(**encoded, max_new_tokens=6, do_sample=False)
        
        full = tokenizer.decode(generated[0], skip_special_tokens=True)
        answer = full[len(prompt):].strip().split()[0] if len(full) > len(prompt) else ''
        # Check for thinking tokens
        has_thinking = 'thinking' in full
        
        correct = expected == 0 or (answer and str(expected) in answer)
        if not correct and answer:
            correct = answer.rstrip('.') == str(expected) or answer == str(expected)
        
        mark = 'OK' if correct else 'XX'
        print(f'  {str(nums):20s} -> "{answer:10s}" exp={expected:3d} {mark} {"(thinking)" if has_thinking else ""}')
