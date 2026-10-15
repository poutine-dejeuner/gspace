import torch, sys
sys.path.insert(0, '.')
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-27B", trust_remote_code=True,
    dtype=torch.bfloat16, device_map="cuda:0",
    attn_implementation="eager",
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B", trust_remote_code=True)

prompt = "Compute 5 + 7 mod 59: 5 + 7 mod 59 ="
encoded = tokenizer(prompt, return_tensors="pt").to("cuda:0")
print(f"Tokens: {encoded['input_ids'][0].tolist()}")
print(f"Prompt: {prompt}")

captured = {}
def make_hook(l):
    def hook(_m, _i, output):
        h = output[0] if isinstance(output, tuple) else output
        captured[l] = h.detach()
    return hook

for l in [59, 60, 61, 62, 63]:
    model.model.layers[l].register_forward_hook(make_hook(l))

with torch.no_grad():
    out = model(**encoded)

for l in [59, 60, 61, 62, 63]:
    h = captured[l]
    print(f"\nL{l}: shape={h.shape}")
    print(f"  last_token (pos -1): norm={h[0,-1].norm().item():.4f}, mean={h[0,-1].mean().item():.4f}")
    print(f"  prev_token (pos -2): norm={h[0,-2].norm().item():.4f}")
    print(f"  diff ||last - prev|| = {(h[0,-1]-h[0,-2]).norm().item():.4f}")
    print(f"  all_positions mean={h.mean().item():.4f}, std={h.std().item():.4f}")

generated = model.generate(**encoded, max_new_tokens=5, do_sample=False)
result = tokenizer.decode(generated[0])
print(f"\nGenerated: {result}")
answer_part = result.split("=")[-1].strip()
print(f"Answer digits: '{answer_part}'")
