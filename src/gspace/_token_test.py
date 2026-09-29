"""Quick tokenization test for Qwen3.5-27B digit encoding."""
import torch
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(
    '/mnt/home/jacob/.cache/huggingface/hub/models--Qwen--Qwen3.5-27B/snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654',
    trust_remote_code=True,
)
print("Digit tokenization:")
for d in range(10):
    ids = tok.encode(str(d), add_special_tokens=False)
    ids_sp = tok.encode(f" {d}", add_special_tokens=False)
    txt = tok.decode(ids)
    txt_sp = tok.decode(ids_sp)
    print(f"  {d}: '{str(d)}'->{ids}({txt}), ' {d}'->{ids_sp}({txt_sp})")

# Test modular arithmetic prompt
prompt = "Compute 5 + 10 mod 59: 5 + 10 mod 59 ="
toks = tok.encode(prompt, add_special_tokens=False)
print(f"\nPrompt: {prompt}")
print(f"Tokens ({len(toks)}): {toks}")
for t in toks:
    print(f"  {t}: '{tok.decode([t])}'")

# Test answers as they appear in training data
for ans in [15, 0, 5, 37, 58]:
    ans_str = str(ans)
    ans_tok = tok.encode(ans_str, add_special_tokens=False)
    ans_tok_sp = tok.encode(f" {ans_str}", add_special_tokens=False)
    print(f"\nAnswer '{ans_str}': {ans_tok} -> '{tok.decode(ans_tok)}'")
    print(f"Answer ' {ans_str}': {ans_tok_sp} -> '{tok.decode(ans_tok_sp)}'")

# Check logits for a specific prompt
print("\n--- Top-10 tokens after '=' ---")
print("(run on compute node with model)")
