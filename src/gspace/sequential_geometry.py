"""Replicate Anthropic's geometry analysis for modular addition.
Uses the sequential format "Modular addition: a b modulo 59. Result:" which
forces the model to compute modular reduction correctly (unlike the pair format).

Steps:
1. For all 59x59 problems, collect residual stream at last prompt token
2. Train linear probes for result (0-58) per layer
3. PCA on class mean activations → test for circular topology
4. For attention heads: pass probe vectors through QK → look for cyclic shift
5. For sequential multi-number addition: trace activation trajectory position-by-position

Usage:
  python src/gspace/sequential_geometry.py [--smoke_test N] [--n_vals N] [--layers L0,L1,...]
"""

import torch, sys, os, json, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
from tqdm import tqdm
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA

sys.path.insert(0, '.')

MODEL = "Qwen/Qwen3.5-27B"
MODULUS = 59
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path("results/sequential_geometry")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = Path("figures/sequential_geometry")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Prompt format that forces modular reduction
PROMPT_SEQ = "Modular addition: {} {}. Result:"
PROMPT_PAIR = "Compute {} + {} mod {}: {} + {} mod {} ="

def make_problems(n, seed=42):
    """Generate all 59x59 pairs with the sequential prompt format."""
    rng = np.random.default_rng(seed)
    problems = []
    # Grid: all pairs
    for a in range(MODULUS):
        for b in range(MODULUS):
            result = (a + b) % MODULUS
            prompt = PROMPT_SEQ.format(a, b, MODULUS)
            problems.append({
                "a": a, "b": b, "result": result, "prompt": prompt,
                "n_tokens": None  # filled after tokenization
            })
    
    if n < len(problems):
        rng.shuffle(problems)
        return problems[:n]
    return problems

def extract_answer(full_text, prompt):
    """Extract answer token, handling Qwen thinking blocks."""
    answer_part = full_text[len(prompt):].strip()
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
    for cl in clean_lines:
        tokens = cl.split()
        if tokens:
            return tokens[0].rstrip('.')
    return None

def collect_activations(model, tokenizer, problems, layers):
    """Collect residual stream activations at the last prompt token for each problem."""
    # Tokenize all prompts
    prompts = [p["prompt"] for p in problems]
    encoded = tokenizer(prompts, padding=True, return_tensors="pt")
    
    # Get last non-padding token for each example
    last_pos = encoded["attention_mask"].sum(dim=1) - 1  # [batch]
    
    # Collect activations
    all_activations = {l: [] for l in layers}
    
    # Process in batches
    batch_size = 16  # Keep this small to avoid OOM
    n = len(problems)
    
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_encoded = {k: v[start:end].to(DEVICE) for k, v in encoded.items()}
        batch_last_pos = last_pos[start:end]
        
        # Register hooks for each layer
        activations = {}
        def make_hook(l):
            def hook(_m, _i, output):
                h = output[0] if isinstance(output, tuple) else output
                activations[l] = h.detach().cpu()
            return hook
        
        handles = []
        for l in layers:
            handle = model.model.layers[l].register_forward_hook(make_hook(l))
            handles.append(handle)
        
        with torch.no_grad():
            _ = model(**batch_encoded, output_hidden_states=False)
        
        for handle in handles:
            handle.remove()
        
        for l in layers:
            h = activations[l]  # [batch, seq_len, d_model]
            batch_acts = h[range(len(batch_last_pos)), batch_last_pos]  # [batch, d_model]
            for i in range(len(batch_last_pos)):
                all_activations[l].append((batch_acts[i].clone(), problems[start + i]["result"]))
    
    return all_activations

def train_probes(activations, layers):
    """Train linear probes per layer using sklearn LogisticRegression."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    
    accuracies = {}
    probe_weights = {}  # layer -> [59, d_model] torch tensor
    probe_biases = {}   # layer -> [59]
    class_means = {}    # layer -> [59, d_model] mean activation per class
    
    for l in tqdm(layers, desc="Training probes"):
        acts_list, labels_list = zip(*activations[l])
        X = torch.stack(acts_list).float().numpy()
        y = np.array(labels_list)
        
        # Compute class means
        means = np.zeros((MODULUS, X.shape[1]))
        for c in range(MODULUS):
            mask = y == c
            if mask.sum() > 0:
                means[c] = X[mask].mean(0)
        class_means[l] = torch.from_numpy(means.astype(np.float32))
        
        # Split
        perm = np.random.RandomState(42).permutation(len(y))
        n_train = int(len(y) * 0.8)
        X_train, X_test = X[perm[:n_train]], X[perm[n_train:]]
        y_train, y_test = y[perm[:n_train]], y[perm[n_train:]]
        
        # Standardize
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        # Train multiclass logistic regression (no multi_class param in sklearn 1.9.0)
        probe = LogisticRegression(solver="lbfgs", max_iter=1000, C=1.0)
        probe.fit(X_train_s, y_train)
        acc = probe.score(X_test_s, y_test)
        accuracies[l] = acc
        
        # Convert coefficients: coef_ shape (59, d_model)
        probe_weights[l] = torch.from_numpy(probe.coef_.astype(np.float32))
        probe_biases[l] = torch.from_numpy(probe.intercept_.astype(np.float32))
        
        print(f"  Layer {l}: test_acc={acc:.4f}")
    
    return accuracies, probe_weights, probe_biases, class_means

def analyze_circular_topology(class_means, accuracies, layers):
    """Check if class means form a circle (as expected for mod 59).
    
    For a circular topology in d dimensions:
    - PC1 and PC2 should capture large variance
    - The top 2 PCs should show a circular/elliptical arrangement
    - r_cos (cosine alignment with angular neighbors) should be high
    """
    results = {}
    
    for l in layers:
        if l not in class_means:
            continue
        means = class_means[l].numpy()  # [59, d_model]
        
        # PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(6, means.shape[0]))
        means_pca = pca.fit_transform(means)
        
        # Variance explained
        var_ratio = pca.explained_variance_ratio_
        cum_var = np.cumsum(var_ratio)
        
        # Check circularity: consecutive classes should be close
        # Cosine similarity between adjacent class means
        cos_sim = np.zeros(MODULUS)
        for c in range(MODULUS):
            v = means_pca[c]
            v_next = means_pca[(c + 1) % MODULUS]
            cos_sim[c] = np.dot(v, v_next) / (np.linalg.norm(v) * np.linalg.norm(v_next) + 1e-8)
        
        r_cos = cos_sim.mean()
        
        # Check if PC1-PC2 projection looks circular
        # For a circle, variance in PC1 ≈ variance in PC2
        pc_ratio = var_ratio[1] / (var_ratio[0] + 1e-8) if len(var_ratio) > 1 else 0
        
        results[l] = {
            "var_ratio": var_ratio.tolist(),
            "cum_var_3": float(cum_var[2]) if len(cum_var) > 2 else 0,
            "cum_var_6": float(cum_var[min(5, len(cum_var)-1)]),
            "r_cos": float(r_cos),
            "pc_ratio": float(pc_ratio),
            "accuracy": accuracies.get(l, 0),
        }
    
    return results

def compute_qk_rotation(probe_weights, model, layers, attention_layers):
    """Pass probe vectors through QK matrices to check for cyclic shift.
    
    Similar to Anthropic's boundary head detection:
    - Compute Q(probe_i) · K(probe_j) for each attention head
    - Look for cyclic shift: highest similarity at i-j = offset
    """
    results = {}
    
    for l in tqdm(attention_layers, desc="Analyzing QK rotation"):
        if l not in probe_weights:
            continue
        
        W = probe_weights[l]  # [59, d_model]
        
        # Get QKV weights for this layer
        layer = model.model.layers[l]
        attn = layer.self_attn
        head_dim = attn.head_dim
        # Read config from the inner text config (model.config.text_config is Qwen3_5TextConfig)
        if hasattr(model.config, "hidden_size"):
            hsz = model.config.hidden_size
            nh = model.config.num_attention_heads
            nkv = model.config.num_key_value_heads
        else:
            hsz = attn.q_proj.weight.shape[1]
            nh = attn.q_proj.weight.shape[0] // (2 * head_dim)  # q_proj is doubled (gate + query)
            nkv = attn.k_proj.weight.shape[0] // head_dim
        
        # Qwen3.5 q_proj interleaves gate+query (doubled) for full-attn layers
        qw_full = attn.q_proj.weight.data.float().cpu()  # [?, hsz]
        kw_full = attn.k_proj.weight.data.float().cpu()  # [nkv * head_dim, hsz]
        
        # For full-attention layers, first nh*head_dim dims = query, next nh*head_dim dims = gate
        # We only want the query part
        qw = qw_full[:nh * head_dim]  # [nh * head_dim, hsz]
        kw = kw_full  # [nkv * head_dim, hsz]
        
        for h in range(nh):
            W_q = qw[h * head_dim:(h + 1) * head_dim]  # [head_dim, hsz]
            W_k = kw[(h % nkv) * head_dim:((h % nkv) + 1) * head_dim]  # [head_dim, hsz]
            
            # Project probes through QK: Q_probe = W @ W_q^T, K_probe = W @ W_k^T
            Q_probes = W @ W_q.T  # [59, head_dim]
            K_probes = W @ W_k.T  # [59, head_dim]
            
            # Normalize
            Q_norm = Q_probes / (Q_probes.norm(dim=-1, keepdim=True) + 1e-8)
            K_norm = K_probes / (K_probes.norm(dim=-1, keepdim=True) + 1e-8)
            
            # Cosine similarity matrix [59, 59]
            sim = Q_norm @ K_norm.T  # [59, 59]
            
            # Check for cyclic shift: similarity should peak at offsets
            # For each row i, the max should be at column i+offset
            cyclic_scores = {}
            for offset in range(MODULUS):
                aligned = torch.diag(sim.roll(-offset, dims=1)).mean().item()
                cyclic_scores[offset] = aligned
            
            # Find best offset
            best_offset = max(cyclic_scores, key=cyclic_scores.get)  # type: ignore
            best_score = cyclic_scores[best_offset]
            
            # Check if offset=0 dominates (no rotation)
            offset_0_score = cyclic_scores[0]
            
            results[(l, h)] = {
                "layer": l,
                "head": h,
                "best_offset": best_offset,
                "best_score": best_score,
                "offset_0_score": offset_0_score,
                "delta": best_score - offset_0_score,
                "cyclic_scores": cyclic_scores,
            }
    
    return results

def analyze_position_trajectory(model, tokenizer, problems, layers):
    """For problems with 3+ numbers, trace activation trajectory after each number.
    
    Prompt: "Modular addition: 3 15 8 modulo 59. Result:"
    We collect activations at:
    - Position after "3" (first number)
    - Position after "15" (second number)  
    - Position after "8" (third number)
    - Position after "59." (modulus)
    - Position after "Result:" (last token)
    
    Then check if positions trace a circle in PCA space.
    """
    # Select multi-number problems
    multi_problems = [p for p in problems if len(p["prompt"].split()) > 10]
    multi_problems = multi_problems[:50]  # limit
    
    trajectories = {}
    
    for p in tqdm(multi_problems, desc="Collecting trajectories"):
        prompt = p["prompt"]
        encoded = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        input_ids = encoded["input_ids"][0]
        tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())
        
        # Find positions of numbers in the prompt
        # Tokenized: ["Modular", "addition", ":", "3", "15", "8", "modulo", "59", ".", "Result", ":"]
        # We want positions right after each number token
        
        num_positions = []
        for i, tok in enumerate(tokens):
            # Check if this token is a number (0-58)
            if tok and tok.strip().lstrip('Ġ').isdigit():
                val = int(tok.strip().lstrip('Ġ'))
                if 0 <= val < MODULUS:
                    num_positions.append(i)
        
        if len(num_positions) >= 3:
            # Register hooks for our layers
            activations = {}
            def make_hook(l):
                def hook(_m, _i, output):
                    h = output[0] if isinstance(output, tuple) else output
                    activations[l] = h.detach().cpu()
                return hook
            
            handles = []
            for l in layers:
                handles.append(
                    model.model.layers[l].register_forward_hook(make_hook(l))
                )
            
            with torch.no_grad():
                _ = model(**encoded, output_hidden_states=False)
            
            for handle in handles:
                handle.remove()
            
            for l in layers:
                if l not in activations:
                    continue
                h = activations[l][0]  # [seq_len, d_model]
                
                if l not in trajectories:
                    trajectories[l] = []
                
                # Collect activations at each number position
                traj = []
                for pos in num_positions:
                    traj.append(h[pos].clone())
                trajectories[l].append(traj)
    
    return trajectories

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke_test", type=int, default=0,
                       help="Run with N problems for debugging")
    parser.add_argument("--layers", type=str, default="",
                       help="Comma-separated layer list (default: 10-step sweep)")
    parser.add_argument("--n_vals", type=int, default=3481,
                       help="Number of problems (max 3481 for full 59x59 grid)")
    parser.add_argument("--attn_layers", type=str, default="51,55,59",
                       help="Layers with full attention to check QK rotation")
    args = parser.parse_args()
    
    smoke_test = args.smoke_test
    print(f"Smoke test: {smoke_test} problems" if smoke_test else "Full grid")
    
    # Determine layers
    if args.layers:
        layers = [int(l) for l in args.layers.split(",")]
    else:
        # Sweep every 10 layers + dense around motor layers
        layers = list(range(0, 64, 10)) + list(range(55, 64))
        layers = sorted(set(layers))
    
    attn_layers = [int(l) for l in args.attn_layers.split(",")]
    
    print(f"Layers: {layers}")
    print(f"Attention layers: {attn_layers}")
    
    # Load model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True,
        dtype=torch.bfloat16, device_map=DEVICE,
        attn_implementation="eager",
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    print(f"Layers: {model.config.num_hidden_layers}, d_model: {model.config.hidden_size}")
    
    # Generate problems
    n = smoke_test if smoke_test else min(args.n_vals, 3481)
    problems = make_problems(n)
    print(f"Generated {len(problems)} problems")
    
    # Collect activations
    print("\nCollecting activations...")
    activations = collect_activations(model, tokenizer, problems, layers)
    
    # Verify probe quality
    print(f"\nChecking probe quality for L61...")
    if 61 in activations:
        acts, labels = zip(*activations[61])
        X = torch.stack(acts).float().numpy()
        y = np.array(labels)
        unique, counts = np.unique(y, return_counts=True)
        print(f"  L61: {len(acts)} samples, {len(unique)} classes, "
              f"X mean={X.mean():.4f}, X std={X.std():.4f}")
        per_class = [len(acts) // MODULUS, (len(acts) + MODULUS - 1) // MODULUS]
        print(f"  Per class: {per_class[0]}-{per_class[1]} samples")
    
    # Train probes
    print("\nTraining probes...")
    accuracies, probe_weights, probe_biases, class_means = train_probes(activations, layers)
    
    # Analyze circular topology
    print("\nAnalyzing circular topology...")
    circular_results = analyze_circular_topology(class_means, accuracies, layers)
    
    for l in sorted(circular_results.keys()):
        r = circular_results[l]
        print(f"  L{l:2d}: acc={r['accuracy']:.2f} r_cos={r['r_cos']:.3f} "
              f"pc_ratio={r['pc_ratio']:.3f} cum_var_3={r['cum_var_3']:.3f} "
              f"cum_var_6={r['cum_var_6']:.3f}")
    
    # QK rotation analysis
    print("\nAnalyzing QK rotation...")
    qk_results = compute_qk_rotation(probe_weights, model, layers, attn_layers)
    
    # Summarize QK results
    if qk_results:
        sorted_qk = sorted(qk_results.values(), key=lambda x: -x["delta"])
        print(f"\nTop 10 QK rotation heads:")
        for i, r in enumerate(sorted_qk[:10]):
            print(f"  {i}: L{r['layer']:2d} H{r['head']:2d}: "
                  f"delta={r['delta']:.4f} best_off={r['best_offset']:2d} "
                  f"best={r['best_score']:.4f} offset0={r['offset_0_score']:.4f}")
        
        # Count how many have non-zero best offset
        non_zero = sum(1 for r in sorted_qk if r["best_offset"] != 0 and r["delta"] > 0.01)
        print(f"\nHeads with non-zero offset (delta>0.01): {non_zero}/{len(sorted_qk)}")
    
    # Position-by-position trajectory analysis (for multi-number problems)
    if not smoke_test or smoke_test > 50:
        print("\nAnalyzing position-by-position trajectories...")
        trajectories = analyze_position_trajectory(
            model, tokenizer, problems[:50], layers
        )
        
        if trajectories:
            for l in sorted(trajectories.keys()):
                traj_list = trajectories[l]
                if len(traj_list) < 2:
                    continue
                # Stack trajectories: [n_problems, n_positions, d_model]
                traj = torch.stack([torch.stack(t) for t in traj_list if len(t) >= 3])
                if traj.shape[0] < 2:
                    continue
                
                # PCA on the concatenated activations
                n, np_pos, d = traj.shape
                flat = traj.reshape(-1, d).numpy()
                pca_obj = PCA(n_components=3)
                flat_pca = pca_obj.fit_transform(flat)

                # Reshape back
                traj_pca = flat_pca.reshape(n, np_pos, 3)

                # For each problem, check if trajectory moves along a consistent direction
                # We want: delta between successive positions should be consistent
                deltas = traj_pca[:, 1:] - traj_pca[:, :-1]  # [n, np_pos-1, 3]
                
                # Cosine similarity between consecutive deltas
                cos_deltas = []
                for i in range(n):
                    for j in range(np_pos - 2):
                        d1 = deltas[i, j]
                        d2 = deltas[i, j + 1]
                        cs = np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-8)
                        cos_deltas.append(cs)
                
                mean_cos = np.mean(cos_deltas) if cos_deltas else 0
                print(f"  L{l:2d}: {n} trajectories, {np_pos} positions, "
                      f"mean_step_cos={mean_cos:.3f} var_ratio={pca_obj.explained_variance_ratio_[:3]}")
    
    # Save results
    results = {
        "model": MODEL,
        "modulus": MODULUS,
        "prompt_format": PROMPT_SEQ,
        "n_problems": len(problems),
        "n_layers": len(layers),
        "accuracies": {str(k): float(v) for k, v in accuracies.items()},
        "circular_topology": {
            str(k): v for k, v in circular_results.items()
        },
        "qk_rotation_top_10": [
            {
                "layer": r["layer"],
                "head": r["head"],
                "best_offset": r["best_offset"],
                "best_score": round(r["best_score"], 4),
                "delta": round(r["delta"], 4),
            }
            for r in sorted_qk[:10]
        ] if qk_results else [],
        "qk_rotation_stats": {
            "total_heads": len(sorted_qk),
            "non_zero_offset": non_zero,
        } if qk_results else {},
    }
    
    output_path = RESULTS_DIR / "results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_path}")
    
    # Generate plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        
        n_layers = len(layers)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 1. Probe accuracies
        ax = axes[0, 0]
        l_list = sorted(accuracies.keys())
        a_list = [accuracies[l] for l in l_list]
        ax.plot(l_list, a_list, "o-", color="blue")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Probe Accuracy")
        ax.set_title("Linear Probe Accuracy per Layer")
        ax.axhline(1/MODULUS, color="gray", linestyle="--", label="random")
        ax.legend()
        
        # 2. Circular topology scores
        ax = axes[0, 1]
        l_list = sorted(circular_results.keys())
        r_cos_list = [circular_results[l]["r_cos"] for l in l_list]
        cum_var = [circular_results[l]["cum_var_3"] for l in l_list]
        ax.plot(l_list, r_cos_list, "o-", label="r_cos (circularity)")
        ax.plot(l_list, cum_var, "s-", label="cum_var_3 (PCA)")
        ax.set_xlabel("Layer")
        ax.set_title("Circular Topology Metrics")
        ax.legend()
        
        # 3. PCA of class means (best layer)
        best_layer = max(accuracies, key=accuracies.get) if accuracies else layers[0]
        if best_layer in class_means:
            ax = axes[0, 2]
            means = class_means[best_layer].numpy()
            pca = PCA(n_components=2)
            means_2d = pca.fit_transform(means)
            colors = np.arange(MODULUS)
            scatter = ax.scatter(means_2d[:, 0], means_2d[:, 1], c=colors, cmap="twilight", s=30)
            plt.colorbar(scatter, ax=ax, label="class")
            # Connect consecutive classes
            ax.plot(means_2d[:, 0], means_2d[:, 1], "k-", alpha=0.3)
            ax.set_title(f"PC1-PC2 of Class Means (L{best_layer})")
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
        
        # 4. QK rotation: delta scores
        if qk_results:
            ax = axes[1, 0]
            deltas = [r["delta"] for r in sorted_qk]
            offsets = [r["best_offset"] for r in sorted_qk]
            ax.scatter(range(len(sorted_qk)), deltas, c=offsets, cmap="viridis", alpha=0.6, s=10)
            ax.set_xlabel("Head index (sorted by delta)")
            ax.set_ylabel("delta = best_score - offset_0_score")
            ax.set_title("QK Rotation Signal per Head")
        
        # 5. Best QK head similarity matrix
        if qk_results:
            ax = axes[1, 1]
            best_head = sorted_qk[0]
            l, h = best_head["layer"], best_head["head"]
            # Recompute similarity matrix
            W = probe_weights[l]
            attn = model.model.layers[l].self_attn
            hdim = attn.head_dim
            nh = 24
            nkv = 4
            qw = attn.q_proj.weight.data[:nh * hdim].float().cpu()
            kw = attn.k_proj.weight.data.float().cpu()
            W_q = qw[h * hdim:(h + 1) * hdim]
            W_k = kw[(h % nkv) * hdim:((h % nkv) + 1) * hdim]
            Q = W @ W_q.T
            K = W @ W_k.T
            Q_n = Q / (Q.norm(dim=-1, keepdim=True) + 1e-8)
            K_n = K / (K.norm(dim=-1, keepdim=True) + 1e-8)
            sim = Q_n @ K_n.T
            im = ax.imshow(sim.numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
            plt.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title(f"QK sim matrix L{l} H{h}")
            ax.set_xlabel("K probe")
            ax.set_ylabel("Q probe")
        
        # 6. Trajectory plot (if available)
        ax = axes[1, 2]
        if trajectories and best_layer in trajectories and len(trajectories[best_layer]) > 0:
            traj_list = trajectories[best_layer][:5]  # first 5
            for i, traj in enumerate(traj_list):
                if len(traj) >= 3:
                    t = torch.stack(traj).numpy()
                    # Just use first 3 dimensions for visualization
                    t_3d = t[:, :3]
                    ax.plot(t_3d[:, 0], t_3d[:, 1], "o-", alpha=0.7, label=f"prob {i}")
            ax.set_title(f"Position Trajectories (L{best_layer}, first 3 dims)")
            ax.set_xlabel("dim 0")
            ax.set_ylabel("dim 1")
        else:
            ax.text(0.5, 0.5, "(no trajectory data)", transform=ax.transAxes, 
                    ha="center", va="center", alpha=0.5)
        
        
        plt.tight_layout()
        fig_path = FIGURES_DIR / "sequential_analysis.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {fig_path}")
        plt.close()
        
    except Exception as e:
        print(f"Plotting error: {e}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
