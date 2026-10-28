"""Test whether Qwen3.5-27B can do sequential modular addition.
Prompts like: "answer in one word: what is the sum modulo 59 of 3 15 8 22 9. The sum is:"
"""
import torch, sys, json, os
sys.path.insert(0, '.')
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3.5-27B"
MODULUS = 59

def test_sequential():
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True,
        dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="eager",
    )
    model.eval()
    
    # Test cases
    cases = [
        ([3], 3),
        ([3, 15], 18),
        ([3, 15, 8], 26),
        ([3, 15, 8, 22], 48),
        ([3, 15, 8, 22, 9], 57),
        ([15, 42, 10], 67 % 59),  # = 8
        ([30, 30], 60 % 59),      # = 1
        ([58, 1], 0),
        ([0, 0, 0, 0, 0], 0),
        ([50, 20], 70 % 59),      # = 11
    ]
    
    for nums, expected in cases:
        nums_str = " ".join(str(n) for n in nums)
        # Try different prompt formats
        prompts = [
            f"answer in one word: what is the sum modulo {MODULUS} of {nums_str}. The sum is:",
            f"What is ({nums_str}) mod {MODULUS}? Answer with just a number.",
            f"Compute: {' + '.join(str(n) for n in nums)} mod {MODULUS}. The answer is",
            f"Modular addition: {nums_str} modulo {MODULUS}. Result:",
        ]
        
        for fmt_idx, prompt in enumerate(prompts):
            encoded = tokenizer(prompt, return_tensors="pt").to("cuda:0")
            with torch.no_grad():
                generated = model.generate(**encoded, max_new_tokens=8, do_sample=False)
            result = tokenizer.decode(generated[0], skip_special_tokens=True)
            answer = result[len(prompt):].strip()
            
            # Check if expected appears in answer
            correct = str(expected) in answer.split()[:5]
            
            print(f"Format {fmt_idx}: {nums} → '{answer}' expected={expected} {'✓' if correct else '✗'}")

def test_simple_addition():
    """Test if the standard prompt format still works."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True,
        dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="eager",
    )
    model.eval()
    
    for a, b in [(5, 7), (30, 40), (58, 3), (0, 0)]:
        expected = (a + b) % MODULUS
        prompt = f"Compute {a} + {b} mod {MODULUS}: {a} + {b} mod {MODULUS} ="
        encoded = tokenizer(prompt, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            generated = model.generate(**encoded, max_new_tokens=5, do_sample=False)
        result = tokenizer.decode(generated[0], skip_special_tokens=True)
        answer = result[len(prompt):].strip()
        correct = str(expected) in answer.split()[:3]
        print(f"{a}+{b}=? expected={expected} '{answer}' {'✓' if correct else '✗'}")

if __name__ == "__main__":
    print("=== SIMPLE ADDITION ===")
    test_simple_addition()
    print("\n=== SEQUENTIAL ADDITION ===")
    test_sequential()
