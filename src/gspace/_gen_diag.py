"""Diagnostic: what does model.generate() actually produce?"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3.5-27B"
MODULUS = 59

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)

gen_kwargs = dict(max_new_tokens=5, do_sample=False, pad_token_id=tok.eos_token_id)

test_pairs = [(5, 7), (23, 19), (40, 7), (58, 1), (0, 0), (30, 30)]

for a, b in test_pairs:
    true_ans = (a + b) % MODULUS
    prompt = f"Compute {a} + {b} mod {MODULUS}: {a} + {b} mod {MODULUS} ="
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]
    
    with torch.no_grad():
        gen = model.generate(**inputs, **gen_kwargs)
    
    # Full generated sequence
    full_text = tok.decode(gen[0], skip_special_tokens=True)
    # Just the generated part
    gen_tokens = gen[0, prompt_len:]
    gen_text = tok.decode(gen_tokens, skip_special_tokens=True)
    
    print(f"\n{a}+{b}≡{true_ans} mod {MODULUS}")
    print(f"  Prompt: \"{prompt}\"")
    print(f"  Gen tokens: {gen_tokens.tolist()}")
    print(f"  Gen text: \"{gen_text}\"")
    print(f"  Full: \"{full_text}\"")
    
    # Parse
    def parse_generated_number(text):
        text = text.strip()
        for token in text.split():
            try:
                return int(token)
            except ValueError:
                continue
        return None
    
    parsed = parse_generated_number(gen_text)
    print(f"  Parsed: {parsed}, match={parsed == true_ans}")
    
    # Also check logits at position -1
    with torch.no_grad():
        out = model(**inputs)
        logits = out.logits[0, -1, :].float().cpu()
    topk = logits.topk(5)
    print(f"  Top-5 logits at '=': {[(tok.decode([t.item()]), t.item(), f'{v.item():.2f}') for t, v in zip(topk.indices, topk.values)]}")

model.cpu()
del model
torch.cuda.empty_cache()
print("\nDone")
