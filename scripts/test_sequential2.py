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

format3 = 'Modular addition: {} modulo {}. Result:'
prompts_2num = 'Compute {} + {} mod {}: {} + {} mod {} ='

cases = [
    ([5, 7], 12),
    ([30, 40], 11),
    ([58, 3], 2),
    ([0, 0], 0),
    ([3, 15, 8, 22, 9], 57),
    ([15, 42, 10], 8),
    ([30, 30], 1),  # 30+30=60 mod 59 = 1
    ([58, 1], 0),  # 58+1=59 mod 59 = 0
    ([0, 0, 0, 0, 0], 0),
    ([50, 20], 11),
    ([7, 14, 21, 28, 35], (7+14+21+28+35)%59),
    ([1, 1, 1, 1, 1], 5),
]

print('=== 2-NUMBER (standard prompt) ===')
for nums, expected in cases:
    if len(nums) != 2:
        continue
    a, b = nums[0], nums[1]
    prompt = prompts_2num.format(a, b, MODULUS, a, b, MODULUS)
    encoded = tokenizer(prompt, return_tensors='pt').to('cuda:0')
    with torch.no_grad():
        generated = model.generate(**encoded, max_new_tokens=5, do_sample=False)
    result = tokenizer.decode(generated[0], skip_special_tokens=True)
    answer = result[len(prompt):].strip().split()[0] if len(result) > len(prompt) else ''
    correct = str(expected) in answer or answer.startswith(str(expected))
    mark = 'OK' if correct else 'XX'
    print(f'  {nums} -> "{answer}" exp={expected} {mark}')

print()
print('===  SEQUENTIAL (format 3)  ===')
for nums, expected in cases:
    if len(nums) < 3:
        continue
    nums_str = ' '.join(str(n) for n in nums)
    prompt = format3.format(nums_str, MODULUS)
    encoded = tokenizer(prompt, return_tensors='pt').to('cuda:0')
    with torch.no_grad():
        generated = model.generate(**encoded, max_new_tokens=5, do_sample=False)
    result = tokenizer.decode(generated[0], skip_special_tokens=True)
    answer = result[len(prompt):].strip().split()[0] if len(result) > len(prompt) else ''
    correct = str(expected) in answer or answer.startswith(str(expected))
    mark = 'OK' if correct else 'XX'
    print(f'  {nums} -> "{answer}" exp={expected} {mark}')

print()
print('=== 2-NUMBER in sequential format (for comparison) ===')
for nums, expected in cases:
    if len(nums) != 2:
        continue
    nums_str = ' '.join(str(n) for n in nums)
    prompt = format3.format(nums_str, MODULUS)
    encoded = tokenizer(prompt, return_tensors='pt').to('cuda:0')
    with torch.no_grad():
        generated = model.generate(**encoded, max_new_tokens=5, do_sample=False)
    result = tokenizer.decode(generated[0], skip_special_tokens=True)
    answer = result[len(prompt):].strip().split()[0] if len(result) > len(prompt) else ''
    correct = str(expected) in answer or answer.startswith(str(expected))
    mark = 'OK' if correct else 'XX'
    print(f'  {nums} -> "{answer}" exp={expected} {mark}')
