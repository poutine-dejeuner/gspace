"""Diagnose why Qwen3.5-27B gets 0/100 on mod 59 in intervention script."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = '/mnt/home/jacob/.cache/huggingface/hub/models--Qwen--Qwen3.5-27B/snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654'

tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map='auto',
    trust_remote_code=True,
)

# Build digit tokens like the intervention script
digit_tokens = {}
for d in range(10):
    for fmt in [str(d), f" {d}"]:
        tok_id = tok.encode(fmt, add_special_tokens=False)
        if len(tok_id) == 1:
            digit_tokens[d] = tok_id[0]
            break
print(f"digit_tokens: {digit_tokens}")

MODULUS = 59

import random
random.seed(42)
test_cases = [(random.randint(0, MODULUS-1), random.randint(0, MODULUS-1)) for _ in range(20)]

correct = 0
for i, (a, b) in enumerate(test_cases):
    true_ans = (a + b) % MODULUS
    prompt = f"Compute {a} + {b} mod {MODULUS}: {a} + {b} mod {MODULUS} ="

    # Tokenize
    inputs = tok(prompt, return_tensors='pt').to(model.device)

    with torch.no_grad():
        out = model(**inputs)
    logits = out.logits[0, -1, :].float().cpu().numpy()

    top1_id = int(logits.argmax())
    top1_text = tok.decode([top1_id])
    top5_ids = logits.argsort()[-5:][::-1].tolist()
    top5_text = [tok.decode([t]) for t in top5_ids]

    # Check correctness
    ans_tokens = []
    ans_str = str(true_ans)
    for ch in ans_str:
        d = int(ch)
        if d in digit_tokens:
            ans_tokens.append(digit_tokens[d])
    is_correct = top1_id in ans_tokens
    if is_correct:
        correct += 1

    # Also try generate
    gen_out = model.generate(**inputs, max_new_tokens=5, do_sample=False, 
                             pad_token_id=tok.eos_token_id)
    gen_text = tok.decode(gen_out[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)

    print(f"{a}+{b}={true_ans}: top1={top1_id}('{top1_text}') top5={top5_text} "
          f"correct={is_correct} ans_tokens={ans_tokens} gen='{gen_text}'")

print(f"\nCorrect (top1): {correct}/{len(test_cases)}")
