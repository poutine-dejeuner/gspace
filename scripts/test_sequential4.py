import torch, sys, re
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

# Format that worked in earlier test (thinking ON):
fmt_seq = 'Modular addition: {} modulo {}. Result:'
fmt_pair = 'Compute {} + {} mod {}: {} + {} mod {} ='

def extract_answer(full_text, prompt, max_answer_tokens=5):
    """Extract just the answer after prompt, handling thinking tokens."""
    answer_part = full_text[len(prompt):].strip()
    # Remove thinking blocks: "thinking...</think>" blocks
    # Qwen thinking:  content between "thinking" and "response"
    # Actually it might be: "thinking\n...\nresponse\n..."
    lines = answer_part.split('\n')
    clean_lines = []
    in_think = False
    for line in lines:
        if 'thinking' in line.lower() and not in_think:
            in_think = True
            continue
        if 'response' in line.lower() and in_think:
            in_think = False
            continue
        if not in_think and line.strip():
            clean_lines.append(line.strip())
    
    # Take first token from cleaned output
    for cl in clean_lines:
        tokens = cl.split()
        if tokens:
            return tokens[0].rstrip('.')
    return None

cases = [
    # Sequential (3+ numbers)
    ([3, 15], 18),
    ([3, 15, 8], 26),
    ([3, 15, 8, 22], 48),
    ([3, 15, 8, 22, 9], 57),
    ([15, 42, 10], 8),
    ([30, 30], 1),
    ([58, 1], 0),
    ([0, 0, 0, 0, 0], 0),
    ([50, 20], 11),
    ([7, 14, 21, 28, 35], (7+14+21+28+35)%59),  # 46
    ([1, 1, 1, 1, 1], 5),
    # Large numbers that would trigger mod reduction
    ([40, 30], 70 % 59),  # 11
    ([58, 58, 58], (58*3) % 59),  # 174%59 = 56
]

print("=== SEQUENTIAL FORMAT (3+ numbers) ===")
for nums, expected in cases:
    if len(nums) < 3:
        continue
    nums_str = ' '.join(str(n) for n in nums)
    prompt = fmt_seq.format(nums_str, MODULUS)
    encoded = tokenizer(prompt, return_tensors='pt').to('cuda:0')
    with torch.no_grad():
        generated = model.generate(**encoded, max_new_tokens=10, do_sample=False)
    full = tokenizer.decode(generated[0], skip_special_tokens=True)
    answer = extract_answer(full, prompt)
    
    correct = answer is not None and (str(expected) == answer or answer.rstrip('.') == str(expected))
    mark = 'OK' if correct else 'XX'
    print(f'  {str(nums):25s} -> "{answer:10s}" exp={expected:3d} {mark}')

print()
print("=== SEQUENTIAL FORMAT (2 numbers) ===")
for nums, expected in cases:
    if len(nums) != 2:
        continue
    nums_str = ' '.join(str(n) for n in nums)
    prompt = fmt_seq.format(nums_str, MODULUS)
    encoded = tokenizer(prompt, return_tensors='pt').to('cuda:0')
    with torch.no_grad():
        generated = model.generate(**encoded, max_new_tokens=10, do_sample=False)
    full = tokenizer.decode(generated[0], skip_special_tokens=True)
    answer = extract_answer(full, prompt)
    
    correct = answer is not None and (str(expected) == answer or answer.rstrip('.') == str(expected))
    mark = 'OK' if correct else 'XX'
    print(f'  {str(nums):25s} -> "{answer:10s}" exp={expected:3d} {mark}')

print()
print("=== PAIR FORMAT (2 numbers) ===")
for nums, expected in cases:
    if len(nums) != 2:
        continue
    a, b = nums[0], nums[1]
    prompt = fmt_pair.format(a, b, MODULUS, a, b, MODULUS)
    encoded = tokenizer(prompt, return_tensors='pt').to('cuda:0')
    with torch.no_grad():
        generated = model.generate(**encoded, max_new_tokens=10, do_sample=False)
    full = tokenizer.decode(generated[0], skip_special_tokens=True)
    answer = extract_answer(full, prompt)
    
    correct = answer is not None and (str(expected) == answer or answer.rstrip('.') == str(expected))
    mark = 'OK' if correct else 'XX'
    print(f'  {str(nums):25s} -> "{answer:10s}" exp={expected:3d} {mark}')
