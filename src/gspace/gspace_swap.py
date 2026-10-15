"""
G-space swap intervention — analogous to Neuronpedia's J-lens swap.

Instead of using J-lens readout directions (w_t @ J_bar[l]), we use per-layer
linear probe weights trained to predict modular arithmetic results from
residual stream activations.

For G-space:  direction = g_class[layer]  (probe weight for that class)
Swap:  h <- h - (h.g_hat_correct)g_hat_correct + (h.g_hat_wrong)g_hat_wrong

We compare this against J-lens swap to see if G-probe directions are as
effective as J-lens directions for swapping model outputs.
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
RES_DIR = PROJ / "results" / "intervention"
MODEL_ID = "Qwen/Qwen3.5-27B"

MODULUS = 59
SEED = 42


# ═════════════════════════════════════════════════════════════════════
# Utilities
# ═════════════════════════════════════════════════════════════════════

def get_token_id(tokenizer, s):
    tid = tokenizer.convert_tokens_to_ids(s)
    if tid == tokenizer.unk_token_id:
        tid = tokenizer.convert_tokens_to_ids(" " + s)
    return tid


def make_problems(n, seed=SEED):
    rng = np.random.default_rng(seed)
    problems = []
    for _ in range(n):
        a = int(rng.integers(0, MODULUS))
        b = int(rng.integers(0, MODULUS))
        result = (a + b) % MODULUS
        prompt = f"Compute {a} + {b} mod {MODULUS}: {a} + {b} mod {MODULUS} ="
        problems.append({"a": a, "b": b, "result": result, "prompt": prompt})
    return problems


def load_model_and_tokenizer(device):
    print(f"Loading {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True,
        dtype=torch.bfloat16, device_map=device,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_wu():
    cache_dir = os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen3.5-27B/"
        "snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654"
    )
    for fname in sorted(os.listdir(cache_dir)):
        if fname.endswith(".safetensors"):
            tensors = safe_load(os.path.join(cache_dir, fname))
            if "lm_head.weight" in tensors:
                return tensors["lm_head.weight"].to(torch.float32)
    raise FileNotFoundError("lm_head.weight not found")


# ═════════════════════════════════════════════════════════════════════
# Train per-layer G-space probes (one weight vector per class per layer)
# ═════════════════════════════════════════════════════════════════════

def collect_activations(model, tokenizer, problems, layers, device):
    """Collect residual stream activations at each layer for each problem."""
    model.eval()
    
    # activations[layer] = list of (activation[d_model], result_label)
    activations = {l: [] for l in layers}
    
    with torch.no_grad():
        for p in tqdm(problems, desc="Collecting activations"):
            encoded = tokenizer(p["prompt"], return_tensors="pt").to(device)
            
            # We need to capture intermediate layer outputs
            # Register hooks for the layers we want
            handles = []
            captured = {}
            
            def make_hook(l):
                def hook(_module, _inputs, output):
                    h = output[0] if isinstance(output, tuple) else output
                    captured[l] = h[0, -1].detach().cpu().to(torch.float32)
                return hook
            
            for l in layers:
                h = model.model.layers[l].register_forward_hook(make_hook(l))
                handles.append(h)
            
            _ = model(**encoded)
            
            for h in handles:
                h.remove()
            
            for l in layers:
                activations[l].append((captured[l], p["result"]))
    
    return activations


def train_gspace_probes(activations, modulus, layers, device):
    """Train per-layer multiclass linear probes using PyTorch SGD.
    
    Returns:
        g_vectors: {layer: [modulus, d_model]} — weight per class
        biases: {layer: [modulus]} — bias per class
        accs: {layer: float} — test accuracy
    """
    
    g_vectors = {}
    biases = {}
    accs = {}
    
    # First, do a quick diagnostic on one layer
    diagnostic_layer = layers[len(layers)//2]
    acts_list, labels_list = zip(*activations[diagnostic_layer])
    X_diag = torch.stack(acts_list)
    y_diag = torch.tensor(labels_list)
    print(f"\nDiagnostic (L{diagnostic_layer}): X shape={X_diag.shape}, y range=[{y_diag.min().item()},{y_diag.max().item()}]")
    print(f"  X mean={X_diag.mean().item():.4f}, std={X_diag.std().item():.4f}")
    print(f"  Unique labels: {len(torch.unique(y_diag))}")
    # Check class separability: mean distance between random pairs vs same-class pairs
    by_class = {}
    for i in range(len(y_diag)):
        c = y_diag[i].item()
        by_class.setdefault(c, []).append(X_diag[i])
    if len(by_class) >= 2:
        c0, c1 = list(by_class.keys())[:2]
        x0 = torch.stack(by_class[c0])
        x1 = torch.stack(by_class[c1])
        inter_dist = (x0.mean(0) - x1.mean(0)).norm().item()
        intra_std = (x0.std(0).norm() + x1.std(0).norm()).item() / 2
        print(f"  Inter-class dist (c{c0} vs c{c1}): {inter_dist:.4f}")
        print(f"  Intra-class std (avg): {intra_std:.4f}")
        print(f"  SNR: {inter_dist / (intra_std + 1e-8):.2f}")
    
    for l in tqdm(layers, desc="Training G-probes"):
        acts_list, labels_list = zip(*activations[l])
        X_all = torch.stack(acts_list).numpy()  # [n, d_model]
        y_all = torch.tensor(labels_list).numpy()  # [n]
        
        n = len(labels_list)
        perm = np.random.RandomState(l).permutation(n)
        n_train = int(n * 0.8)
        X_train = X_all[perm[:n_train]]
        y_train = y_all[perm[:n_train]]
        X_test = X_all[perm[n_train:]]
        y_test = y_all[perm[n_train:]]
        
        # Use sklearn LogisticRegression (proven to work in Phase 1)
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        probe = LogisticRegression(
            solver="lbfgs", max_iter=500, C=1.0,
        )
        probe.fit(X_train_s, y_train)
        
        acc = probe.score(X_test_s, y_test)
        
        # Convert coef_ from numpy to torch tensors (float32)
        # probe.coef_ shape: [59, d_model] for 59-class OvR
        g_vectors[l] = torch.from_numpy(probe.coef_.astype(np.float32))
        biases[l] = torch.from_numpy(probe.intercept_.astype(np.float32))
        accs[l] = acc
        print(f"  Layer {l}: test_acc={acc:.4f}")
    
    return g_vectors, biases, accs


# ═════════════════════════════════════════════════════════════════════
# G-space swap
# ═════════════════════════════════════════════════════════════════════

class GSpaceSwapper:
    """Apply G-space swap: exchange the correct result's direction for wrong result's."""
    
    def __init__(self, model, g_vectors, result_true, result_wrong, layers):
        self.model = model
        self.layers = layers
        self.hooks = []
        self.device = next(model.parameters()).device
        
        self.src_hats = {}
        self.tgt_hats = {}
        
        for l in layers:
            g_l = g_vectors[l].to(torch.float32)  # [modulus, d_model]
            s = g_l[result_true]   # [d_model]
            t = g_l[result_wrong]  # [d_model]
            s_norm = torch.linalg.vector_norm(s)
            t_norm = torch.linalg.vector_norm(t)
            if s_norm > 0 and t_norm > 0:
                self.src_hats[l] = (s / s_norm).to(torch.bfloat16)
                self.tgt_hats[l] = (t / t_norm).to(torch.bfloat16)
    
    def _make_swap_hook(self, l):
        s_hat = self.src_hats[l].to(self.device)
        t_hat = self.tgt_hats[l].to(self.device)
        
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
    
    def detach(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ═════════════════════════════════════════════════════════════════════
# J-lens swap (for comparison)
# ═════════════════════════════════════════════════════════════════════

def load_jlens(path):
    print(f"Loading J-lens from {path}")
    d = torch.load(path, map_location="cpu", weights_only=True)
    J_dict = d["J"]
    J_matrices = {}
    for layer_name, J in J_dict.items():
        l = int(layer_name)
        J_matrices[l] = J.to(torch.float32)
    print(f"  {len(J_matrices)} layers: {sorted(J_matrices.keys())}")
    return J_matrices


class JLensSwapper:
    def __init__(self, model, j_matrices, wu, token_id_src, token_id_tgt, layers):
        self.model = model
        self.layers = layers
        self.hooks = []
        self.device = next(model.parameters()).device
        
        w_src = wu[token_id_src]
        w_tgt = wu[token_id_tgt]
        
        self.src_hats = {}
        self.tgt_hats = {}
        
        for l in layers:
            s_dir = (w_src.to(torch.float32) @ j_matrices[l].to(torch.float32))
            t_dir = (w_tgt.to(torch.float32) @ j_matrices[l].to(torch.float32))
            s_norm = torch.linalg.vector_norm(s_dir)
            t_norm = torch.linalg.vector_norm(t_dir)
            if s_norm > 0 and t_norm > 0:
                self.src_hats[l] = (s_dir / s_norm).to(torch.bfloat16)
                self.tgt_hats[l] = (t_dir / t_norm).to(torch.bfloat16)
    
    def _make_swap_hook(self, l):
        s_hat = self.src_hats[l].to(self.device)
        t_hat = self.tgt_hats[l].to(self.device)
        
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
    
    def detach(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ═════════════════════════════════════════════════════════════════════
# Evaluation
# ═════════════════════════════════════════════════════════════════════

def evaluate_swapper(model, tokenizer, problems, make_swapper_fn, config_name,
                     digit_ids, n_problems):
    """Apply swap and measure whether the wrong digit outranks the correct one."""
    device = next(model.parameters()).device
    model.eval()
    
    n_flipped = 0
    n_attempted = 0
    rank_correct_list = []
    rank_wrong_list = []
    
    for p in tqdm(problems, desc=f"  {config_name}", leave=False):
        # Correct result's first digit
        correct_str = str(p["result"])
        src_char = correct_str[0]
        src_id = digit_ids.get(int(src_char))
        
        # Wrong result = (correct + 1) mod 59, first digit
        wrong = (p["result"] + 1) % MODULUS
        wrong_str = str(wrong)
        tgt_char = wrong_str[0]
        tgt_id = digit_ids.get(int(tgt_char))
        
        if src_id is None or tgt_id is None or src_id == tgt_id:
            continue
        
        swapper = make_swapper_fn(src_id, tgt_id, p["result"], wrong)
        
        if swapper is None:
            continue  # No valid layers for swap
        
        swapper.attach()
        encoded = tokenizer(p["prompt"], return_tensors="pt").to(device)
        outputs = model(**encoded)
        swapper.detach()
        
        logits = outputs.logits[0, -1]
        _, sorted_ids = torch.sort(logits, descending=True)
        
        rank_correct = (sorted_ids == src_id).nonzero(as_tuple=True)[0].item()
        rank_wrong = (sorted_ids == tgt_id).nonzero(as_tuple=True)[0].item()
        
        if rank_wrong < rank_correct:
            n_flipped += 1
        n_attempted += 1
        rank_correct_list.append(rank_correct)
        rank_wrong_list.append(rank_wrong)
    
    flip_rate = n_flipped / n_attempted if n_attempted > 0 else 0
    return {
        "config": config_name,
        "n_attempted": n_attempted,
        "n_flipped": n_flipped,
        "flip_rate": flip_rate,
        "rank_correct_mean": float(np.mean(rank_correct_list)) if rank_correct_list else 0,
        "rank_wrong_mean": float(np.mean(rank_wrong_list)) if rank_wrong_list else 0,
    }


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke_test", type=int, default=0,
                       help="Use N training examples and N test problems")
    parser.add_argument("--n_train", type=int, default=500,
                       help="Number of problems for training G-probes")
    parser.add_argument("--n_test", type=int, default=200,
                       help="Number of problems for swap evaluation")
    parser.add_argument("--probe_layers", type=str, default=None,
                       help="Layers to train probes on (e.g. '48,50,55,59,61,62,63')")
    args = parser.parse_args()
    
    if args.smoke_test:
        n_train = max(args.smoke_test, 300)  # Need at least ~5 per class for 59 classes
        n_test = min(args.smoke_test, 40)
        print(f"SMOKE TEST: n_train={n_train}, n_test={n_test}")
    else:
        n_train = args.n_train
        n_test = args.n_test
    
    # Probe layers: pick key layers across the spectrum
    if args.probe_layers:
        probe_layers = [int(x.strip()) for x in args.probe_layers.split(",")]
    else:
        probe_layers = sorted(set([0, 10, 20, 30, 40, 45, 48, 50, 52, 55, 58, 59, 60, 61, 62, 63]))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_and_tokenizer(device)
    
    # Load digit token IDs
    digit_ids = {}
    for d in range(10):
        tid = get_token_id(tokenizer, str(d))
        if tid != tokenizer.unk_token_id:
            digit_ids[d] = tid
    print(f"Digit tokens: {digit_ids}")
    
    # Step 1: Collect activations for training G-probes
    print(f"\nStep 1: Collecting activations for {n_train} problems at {len(probe_layers)} layers")
    train_problems = make_problems(n_train, seed=SEED)
    activations = collect_activations(model, tokenizer, train_problems, probe_layers, device)
    
    # Step 2: Train per-layer G-probes
    print(f"\nStep 2: Training G-space probes")
    g_vectors, biases, accs = train_gspace_probes(activations, MODULUS, probe_layers, device)
    
    # Print probe accuracies
    print("\nG-space probe accuracies:")
    for l in sorted(accs.keys()):
        print(f"  L{l}: {accs[l]:.4f}")
    
    # Step 3: Load J-lens for comparison
    print("\nStep 3: Loading J-lens")
    jlens_path = PROJ / "results" / "jspace_cache" / "qwen3.5-27b" / "jlens" / \
                 "Salesforce-wikitext" / "Qwen3.5-27B_jacobian_lens.pt"
    j_matrices = load_jlens(str(jlens_path))
    wu = load_wu()
    
    # Step 4: Evaluate both G-space and J-space swaps
    print(f"\nStep 4: Evaluating swaps on {n_test} problems")
    test_problems = make_problems(n_test, seed=SEED + 1)
    
    # Configs: which layers to swap at
    configs = [
        ("L61 only", [61]),
        ("L62 only", [62]),
        ("L61-62", [61, 62]),
        ("L48-63", list(range(48, 64))),
        ("L59-63", list(range(59, 64))),
    ]
    
    all_results = []
    
    # --- G-space swap ---
    print("\n--- G-SPACE SWAP ---")
    for config_name, layers in configs:
        valid_layers = [l for l in layers if l in g_vectors and accs[l] > 0.3]
        if not valid_layers:
            print(f"  Skipping {config_name}: no valid layers")
            continue
        
        def make_gswap_fn(valid_layers):
            def fn(src_id, tgt_id, result_true, result_wrong):
                return GSpaceSwapper(model, g_vectors, result_true, result_wrong, valid_layers)
            return fn
        
        result = evaluate_swapper(
            model, tokenizer, test_problems,
            make_gswap_fn(valid_layers),
            f"G-space {config_name}", digit_ids, n_test
        )
        result["swap_type"] = "G-space"
        all_results.append(result)
        print(f"  {config_name}: flip={result['flip_rate']:.3f} "
              f"rank_c={result['rank_correct_mean']:.1f} rank_w={result['rank_wrong_mean']:.1f}")
    
    # --- J-space swap ---
    print("\n--- J-SPACE SWAP ---")
    for config_name, layers in configs:
        valid_layers = [l for l in layers if l in j_matrices]
        if not valid_layers:
            print(f"  Skipping {config_name}: no valid layers")
            continue
        
        def make_jswap_fn(valid_layers):
            def fn(src_id, tgt_id, result_true, result_wrong):
                return JLensSwapper(model, j_matrices, wu, src_id, tgt_id, valid_layers)
            return fn
        
        result = evaluate_swapper(
            model, tokenizer, test_problems,
            make_jswap_fn(valid_layers),
            f"J-space {config_name}", digit_ids, n_test
        )
        result["swap_type"] = "J-space"
        all_results.append(result)
        print(f"  {config_name}: flip={result['flip_rate']:.3f} "
              f"rank_c={result['rank_correct_mean']:.1f} rank_w={result['rank_wrong_mean']:.1f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    for r in all_results:
        print(f"  {r['config']:30s} ({r['swap_type']:8s}): "
              f"flip={r['flip_rate']:.3f} "
              f"rank_c={r['rank_correct_mean']:.1f} rank_w={r['rank_wrong_mean']:.1f}")
    
    # Save
    os.makedirs(RES_DIR, exist_ok=True)
    out_path = RES_DIR / "gspace_swap.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
