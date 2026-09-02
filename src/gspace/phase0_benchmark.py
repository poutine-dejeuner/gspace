"""
Phase 0: Model Selection via Modular Arithmetic Benchmark.

Generate a benchmark for mod-59 arithmetic, test small open-weight models,
and select the smallest one with perfect accuracy.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


@dataclass
class BenchmarkConfig:
    modulus: int = 59
    operations: list[str] = field(default_factory=lambda: ["+"])
    n_test: int = 200  # test pairs
    output_dir: str = "benchmark"
    seed: int = 42


@dataclass
class ModelCandidate:
    name: str
    hf_id: str
    params: str  # human-readable
    dtype: torch.dtype = torch.bfloat16


CANDIDATES: list[ModelCandidate] = [
    ModelCandidate("Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B", "0.5B"),
    ModelCandidate("SmolLM2-1.7B", "HuggingFaceTB/SmolLM2-1.7B-Instruct", "1.7B"),
    ModelCandidate("Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B", "1.5B"),
    ModelCandidate("Gemma-3-4B", "google/gemma-3-4b-it", "4B"),
    ModelCandidate("Llama-3.2-3B", "meta-llama/Llama-3.2-3B-Instruct", "3B"),
]


def generate_problems(modulus: int, op: str, n: int, seed: int) -> list[dict]:
    """Generate n arithmetic problems for mod-59."""
    import random
    rng = random.Random(seed)
    # generate all pairs, hold out test set
    all_pairs = [(a, b) for a in range(modulus) for b in range(modulus)]
    rng.shuffle(all_pairs)
    test_pairs = all_pairs[:n]

    problems = []
    for a, b in test_pairs:
        if op == "+":
            result = (a + b) % modulus
        elif op == "-":
            result = (a - b) % modulus
        elif op == "*":
            result = (a * b) % modulus
        else:
            raise ValueError(f"Unknown op: {op}")

        prompt = f"Compute {a} {op} {b} mod {modulus}: {a} {op} {b} mod {modulus} ="
        problems.append({
            "a": a,
            "b": b,
            "op": op,
            "modulus": modulus,
            "result": result,
            "prompt": prompt,
        })
    return problems


def evaluate_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    problems: list[dict],
    device: torch.device,
    batch_size: int = 8,
) -> dict:
    """Evaluate model accuracy on benchmark problems.

    For each problem, extract the next-token prediction and check
    if it matches the correct result.
    """
    model.eval()
    correct = 0
    total = len(problems)
    per_position_accuracy = []  # (a_true, b_true, pred, correct)

    # Results token: we need to check that '0'..'58' are single tokens
    result_tokens = set()
    for r in range(59):
        tok = tokenizer.encode(str(r), add_special_tokens=False)
        if len(tok) == 1:
            result_tokens.add(r)

    for i in tqdm(range(0, total, batch_size), desc="Evaluating"):
        batch = problems[i : i + batch_size]
        prompts = [p["prompt"] for p in batch]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            # Get logits for the last token position
            last_logits = outputs.logits[:, -1, :]
            # Get top prediction
            pred_token_ids = last_logits.argmax(dim=-1)

        for j, p in enumerate(batch):
            pred_token = tokenizer.decode(pred_token_ids[j]).strip()
            try:
                pred_val = int(pred_token)
            except ValueError:
                pred_val = None

            per_position_accuracy.append({
                "a": p["a"],
                "b": p["b"],
                "op": p["op"],
                "target": p["result"],
                "pred_token": pred_token,
                "pred": pred_val,
                "correct": pred_val == p["result"],
            })
            if pred_val == p["result"]:
                correct += 1

    accuracy = correct / total
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "details": per_position_accuracy,
        "single_token_results": len(result_tokens) == 59,
    }


def main():
    config = BenchmarkConfig()
    os.makedirs(config.output_dir, exist_ok=True)

    # Generate benchmark
    all_problems = {}
    for op in config.operations:
        problems = generate_problems(config.modulus, op, config.n_test, config.seed)
        all_problems[op] = problems
        out_path = os.path.join(config.output_dir, f"mod{config.modulus}_{op}.json")
        with open(out_path, "w") as f:
            json.dump(problems, f, indent=2)
        print(f"Generated {len(problems)} {op} problems → {out_path}")

    # Evaluate candidates
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    results = {}
    for candidate in CANDIDATES:
        print(f"\n{'='*50}")
        print(f"Testing {candidate.name} ({candidate.hf_id})...")

        try:
            tokenizer = AutoTokenizer.from_pretrained(candidate.hf_id)
            model = AutoModelForCausalLM.from_pretrained(
                candidate.hf_id,
                torch_dtype=candidate.dtype,
                device_map="auto" if device.type == "cuda" else None,
            )
            if device.type == "cpu":
                model = model.to(device)
        except Exception as e:
            print(f"  FAILED to load: {e}")
            results[candidate.name] = {"error": str(e)}
            continue

        # Quick tokenizer check: are residues 0-58 single tokens?
        token_check = {}
        for r in range(config.modulus):
            tok = tokenizer.encode(str(r), add_special_tokens=False)
            token_check[r] = len(tok) == 1
        single_token_count = sum(token_check.values())
        print(f"  Single-token residues: {single_token_count}/{config.modulus}")

        model_results = {}
        for op, problems in all_problems.items():
            print(f"  Evaluating {op}...")
            result = evaluate_model(model, tokenizer, problems, device)
            print(f"    Accuracy: {result['accuracy']:.4f} ({result['correct']}/{result['total']})")
            model_results[op] = {
                "accuracy": result["accuracy"],
                "correct": result["correct"],
                "total": result["total"],
            }

        model_results["single_token_residues"] = single_token_count
        results[candidate.name] = model_results

        # Free memory
        del model, tokenizer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for name, r in results.items():
        if "error" in r:
            print(f"  {name}: ERROR — {r['error']}")
        else:
            accs = {op: f"{v['accuracy']:.4f}" for op, v in r.items() if isinstance(v, dict) and "accuracy" in v}
            print(f"  {name}: {accs}  (single-token: {r.get('single_token_residues', '?')}/{config.modulus})")

    # Save results
    out_path = os.path.join(config.output_dir, "model_selection_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Find best smallest
    perfect = [(name, r) for name, r in results.items()
               if "error" not in r
               and all(isinstance(v, dict) and v.get("accuracy", 0) == 1.0
                       for k, v in r.items() if k != "single_token_residues")]
    if perfect:
        print(f"\nModels with perfect accuracy: {[p[0] for p in perfect]}")
    else:
        best = max(
            [(name, r) for name, r in results.items() if "error" not in r],
            key=lambda x: sum(
                v.get("accuracy", 0) for k, v in x[1].items()
                if k != "single_token_residues"
            ),
        )
        print(f"\nBest model: {best[0]}")


if __name__ == "__main__":
    main()
