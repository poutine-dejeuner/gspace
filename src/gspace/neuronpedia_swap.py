"""
J-lens SWAP intervention — matching Neuronpedia's approach.

Neuronpedia's swap formula (from their inference code):
  For each steered layer l and each token position:
    1. Compute J-lens direction for source token: s = w_src @ J_bar[l]
    2. Compute J-lens direction for target token: t = w_tgt @ J_bar[l]
    3. Normalize both to unit vectors: ŝ = s/||s||, t̂ = t/||t||
    4. h ← h - (h·ŝ)ŝ + (h·ŝ)t̂

This removes the source-token readout component and replaces it with
the target-token component, keeping the same coefficient (magnitude).

Key differences from our prior coordinate-swapping approach:
- Uses J-lens DIRECTIONS (w_t @ J_bar), not pseudoinverse coordinates
- Applied at ALL token positions, not just last token
- Direction is PER-LAYER (different at each layer because J_bar changes)
- Simple projection removal + re-addition, not coordinate-basis swap

We test this on modular addition: swap the correct digit answer's J-lens
readout for a wrong digit's, across layers 59-63 (G/J peak region).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file as safe_load
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

PROJ = Path(__file__).resolve().parents[2]
FIG_DIR = PROJ / "figures" / "intervention"
RES_DIR = PROJ / "results" / "intervention"
MODEL_ID = "Qwen/Qwen3.5-27B"

MODULUS = 59
N_PROBLEMS = 200
BATCH_SIZE = 4
SEED = 42


def load_jlens(path):
    """Load J-lens matrices: {layer: J[d_model, d_model]}."""
    print(f"Loading J-lens from {path}")
    d = torch.load(path, map_location="cpu", weights_only=True)
    J_dict = d["J"]
    J_matrices = {}
    for layer_name, J in J_dict.items():
        l = int(layer_name)
        J_matrices[l] = J.to(torch.float32)
    print(f"  {len(J_matrices)} layers: {sorted(J_matrices.keys())}")
    return J_matrices


def load_wu():
    cache_dir = os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen3.5-27B/"
        "snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654"
    )
    print(f"Loading W_U from {cache_dir}")
    for fname in sorted(os.listdir(cache_dir)):
        if fname.endswith(".safetensors"):
            tensors = safe_load(os.path.join(cache_dir, fname))
            if "lm_head.weight" in tensors:
                return tensors["lm_head.weight"].to(torch.float32)
    raise FileNotFoundError("lm_head.weight not found")


def get_token_id(tokenizer, s):
    """Get token ID for a string, trying with and without leading space."""
    tid = tokenizer.convert_tokens_to_ids(s)
    if tid == tokenizer.unk_token_id:
        tid = tokenizer.convert_tokens_to_ids(" " + s)
    return tid


def get_token_ids_for_two_digit(tokenizer, n):
    """For a two-digit number n (0-58), get the two token IDs.
    
    Qwen tokenizes e.g. "15" as ["1", "5"].
    For single-digit: "7" → just one token.
    """
    s = str(n)
    ids = []
    for ch in s:
        tid = get_token_id(tokenizer, ch)
        ids.append(tid)
    return ids


def make_problems(n):
    """Generate modular addition problems with the prompt format the model expects."""
    rng = np.random.default_rng(SEED)
    problems = []
    for _ in range(n):
        a = int(rng.integers(0, MODULUS))
        b = int(rng.integers(0, MODULUS))
        result = (a + b) % MODULUS
        prompt = f"Compute {a} + {b} mod {MODULUS}.\nThe answer is"
        problems.append({"a": a, "b": b, "result": result, "prompt": prompt})
    return problems


def build_jlens_direction(w_t, J_bar, device):
    """Compute residual-space J-lens direction for token w_t at a layer.
    
    Formula: direction = w_t @ J_bar  (equivalent to J_bar^T @ w_t)
    where w_t is the unembedding vector [d_model] and J_bar is [d_model, d_model].
    """
    return (w_t.to(torch.float32).to(device) @ J_bar.to(torch.float32).to(device))


class JLensSwapper:
    """Apply Neuronpedia-style J-lens swap during forward pass.
    
    For each steered layer:
      h ← h - (h·ŝ)ŝ + (h·ŝ)t̂
    where ŝ, t̂ are unit J-lens directions for source/target tokens.
    """
    
    def __init__(self, model, j_matrices, wu, tokenizer, token_id_src, token_id_tgt, layers):
        self.model = model
        self.layers = layers
        self.tokenizer = tokenizer
        self.hooks = []
        self.device = next(model.parameters()).device
        
        # Precompute per-layer swap directions
        # s_l = w_src @ J[l],  t_l = w_tgt @ J[l]
        w_src = wu[token_id_src]  # [d_model]
        w_tgt = wu[token_id_tgt]  # [d_model]
        
        self.src_hats = {}  # unit source directions per layer
        self.tgt_hats = {}  # unit target directions per layer
        
        for l in layers:
            s_dir = build_jlens_direction(w_src, j_matrices[l], "cpu")
            t_dir = build_jlens_direction(w_tgt, j_matrices[l], "cpu")
            s_norm = torch.linalg.vector_norm(s_dir)
            t_norm = torch.linalg.vector_norm(t_dir)
            if s_norm > 0 and t_norm > 0:
                self.src_hats[l] = s_dir / s_norm
                self.tgt_hats[l] = t_dir / t_norm
    
    def _make_swap_hook(self, l):
        s_hat = self.src_hats[l].to(self.device).to(torch.bfloat16)
        t_hat = self.tgt_hats[l].to(self.device).to(torch.bfloat16)
        
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            coeff = (hidden * s_hat).sum(dim=-1, keepdim=True)
            swapped = hidden - coeff * s_hat + coeff * t_hat
            if isinstance(output, tuple):
                return (swapped,) + output[1:]
            return swapped
        return hook
    
    def attach(self):
        for l in self.layers:
            if l in self.src_hats:
                h = self.model.model.layers[l].register_forward_hook(self._make_swap_hook(l))
                self.hooks.append(h)
        print(f"  Attached swap hooks at layers {sorted(self.src_hats.keys())}")
    
    def detach(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def evaluate_swap(model, tokenizer, problems, swapper, token_id_correct_first,
                  token_id_wrong_first):
    """Run forward pass with swap hooks and measure effect on ranks.
    
    For two-digit answers, we track the rank of the FIRST digit token only
    (since that's what the model predicts first).
    """
    device = next(model.parameters()).device
    model.eval()
    
    n_correct_top1 = 0
    n_swapped = 0  # wrong token is ranked above correct
    rank_correct_list = []
    rank_wrong_list = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(problems), BATCH_SIZE), desc="Swap eval"):
            batch = problems[i:i + BATCH_SIZE]
            prompts = [p["prompt"] for p in batch]
            
            # Tokenize
            encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
            
            # Forward pass (hooks fire automatically)
            with swapper.attach() if False else _null_context():
                pass
            
            # We need to manually manage hooks
            # Run with hooks attached
            swapper.attach()
            outputs = model(**encoded)
            swapper.detach()
            
            logits = outputs.logits  # [batch, seq, vocab]
            last_pos = (encoded["attention_mask"].sum(dim=1) - 1)
            
            for j in range(len(batch)):
                p = batch[j]
                lp = last_pos[j].item()
                logits_last = logits[j, lp]  # [vocab]
                
                # Get rank of the first digit of the correct answer
                correct_digit = str(p["result"])
                correct_first_char = correct_digit[0]
                tid_correct = get_token_id(tokenizer, correct_first_char)
                
                # Get rank of first digit of wrong answer (correct + 1 mod 59)
                wrong_result = (p["result"] + 1) % MODULUS
                wrong_first_char = str(wrong_result)[0]
                tid_wrong = get_token_id(tokenizer, wrong_first_char)
                
                _, sorted_ids = torch.sort(logits_last, descending=True)
                rank_correct = (sorted_ids == tid_correct).nonzero(as_tuple=True)[0].item()
                rank_wrong = (sorted_ids == tid_wrong).nonzero(as_tuple=True)[0].item()
                
                if rank_correct == 0:
                    n_correct_top1 += 1
                if rank_wrong < rank_correct:
                    n_swapped += 1
                
                rank_correct_list.append(rank_correct)
                rank_wrong_list.append(rank_wrong)
    
    n = len(problems)
    return {
        "n": n,
        "correct_top1": n_correct_top1 / n,
        "swapped_fraction": n_swapped / n,
        "rank_correct_mean": np.mean(rank_correct_list),
        "rank_wrong_mean": np.mean(rank_wrong_list),
        "rank_correct_list": rank_correct_list,
        "rank_wrong_list": rank_wrong_list,
    }


def _null_context():
    """No-op context manager for the unused case above."""
    class _Noop:
        def __enter__(self): return self
        def __exit__(self, *a): pass
    return _Noop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke_test", type=int, default=0)
    parser.add_argument("--n_problems", type=int, default=N_PROBLEMS)
    args = parser.parse_args()
    
    if args.smoke_test:
        n_problems = min(args.smoke_test, 20)
        print(f"SMOKE TEST: {n_problems} problems")
    else:
        n_problems = args.n_problems
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load J-lens
    jlens_path = PROJ / "results" / "jspace_cache" / "qwen3.5-27b" / "jlens" / \
                 "Salesforce-wikitext" / "Qwen3.5-27B_jacobian_lens.pt"
    j_matrices = load_jlens(str(jlens_path))
    
    # Load model
    print(f"Loading {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True,
        dtype=torch.bfloat16, device_map=device,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    wu = load_wu()
    
    # All digit token IDs (0-9)
    digit_ids = {}
    for d in range(10):
        tid = get_token_id(tokenizer, str(d))
        if tid != tokenizer.unk_token_id:
            digit_ids[d] = tid
    print(f"Digit tokens: {digit_ids}")
    
    # Generate problems
    problems = make_problems(n_problems)
    
    # Configurations to test
    layer_configs = {
        "L59-63 (peak region)": list(range(59, 64)),
        "L48-63 (full crosscoder)": list(range(48, 64)),
        "L18-63 (Neuronpedia-style wide)": list(range(18, 64)),
        "L61 only (G peak)": [61],
        "L62 only (J peak)": [62],
        "L61-62 (G+J peaks)": [61, 62],
        "L60-62 (linear attn)": [60, 61, 62],
    }
    
    results = []
    
    for config_name, layers in layer_configs.items():
        # Filter to layers where J-lens is available
        layers = [l for l in layers if l in j_matrices]
        if not layers:
            print(f"\n{config_name}: no valid layers, skipping")
            continue
        
        print(f"\n{'='*60}")
        print(f"Config: {config_name} ({len(layers)} layers: {layers[0]}-{layers[-1]})")
        
        # For each problem, build swapper swapping first digit of correct → first digit of wrong
        # We evaluate problem-by-problem since swap directions depend on the specific tokens
        
        n_flipped = 0
        rank_correct_all = []
        rank_wrong_all = []
        rank_correct_clean_all = []
        rank_wrong_clean_all = []
        
        for idx in tqdm(range(len(problems)), desc=f"  {config_name}"):
            p = problems[idx]
            
            # Source token: first digit of correct answer
            correct_str = str(p["result"])
            src_char = correct_str[0]
            src_id = digit_ids.get(int(src_char))
            
            # Target token: first digit of (correct + 1) mod 59
            wrong = (p["result"] + 1) % MODULUS
            wrong_str = str(wrong)
            tgt_char = wrong_str[0]
            tgt_id = digit_ids.get(int(tgt_char))
            
            if src_id is None or tgt_id is None or src_id == tgt_id:
                continue
            
            # Build swapper
            swapper = JLensSwapper(model, j_matrices, wu, tokenizer, src_id, tgt_id, layers)
            
            # Run single forward pass with swap hooks
            swapper.attach()
            prompt = p["prompt"]
            encoded = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model(**encoded)
            swapper.detach()
            
            logits = outputs.logits[0, -1]  # last token position
            
            # Ranks
            _, sorted_ids = torch.sort(logits, descending=True)
            
            rank_correct = (sorted_ids == src_id).nonzero(as_tuple=True)[0].item()
            rank_wrong = (sorted_ids == tgt_id).nonzero(as_tuple=True)[0].item()
            
            if rank_wrong < rank_correct:
                n_flipped += 1
            
            rank_correct_all.append(rank_correct)
            rank_wrong_all.append(rank_wrong)
            
            # Also measure clean (no swap) ranks for reference
            # (only do this on first config to save time)
            if idx == 0 and len(results) == 0:
                # Quick clean baseline
                pass
        
        n_eval = len(rank_correct_all)
        flip_rate = n_flipped / n_eval if n_eval > 0 else 0
        
        result = {
            "config": config_name,
            "layers": layers,
            "n_layers": len(layers),
            "n_eval": n_eval,
            "n_flipped": n_flipped,
            "flip_rate": flip_rate,
            "rank_correct_mean": float(np.mean(rank_correct_all)),
            "rank_wrong_mean": float(np.mean(rank_wrong_all)),
        }
        results.append(result)
        print(f"  Flip rate: {n_flipped}/{n_eval} = {flip_rate:.3f}")
        print(f"  Rank correct: {result['rank_correct_mean']:.1f}, wrong: {result['rank_wrong_mean']:.1f}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"  {r['config']:35s}: flip={r['flip_rate']:.3f} "
              f"rank_c={r['rank_correct_mean']:.1f} rank_w={r['rank_wrong_mean']:.1f}")
    
    # Save
    os.makedirs(RES_DIR, exist_ok=True)
    out_path = RES_DIR / "neuronpedia_swap.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
