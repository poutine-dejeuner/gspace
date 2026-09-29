"""
Identify workspace layers for Qwen3.5-27B using the exact same metrics
as the "Global Workspace" paper (Section: "In which layers does the J-space
act as a workspace?", Figure 27/28).

Metrics:
  1. CKA (Centered Kernel Alignment) between pairs of layers on J-lens vectors
  2. Top-k next-token prediction accuracy of J-lens
  3. Excess kurtosis of J-lens readout distributions
  4. Autocorrelation of top-1 J-lens token across positions (Δ log prob)
  5. Effective rank of J-lens vectors
"""

import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import kurtosis
from tqdm import tqdm
import torch
from safetensors.torch import load_file as safe_load
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure unbuffered output
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

D_MODEL = 5120
N_LAYERS = 64
CKA_SUBSAMPLE = 2  # every Nth layer for CKA

# ─── Loading ────────────────────────────────────────────────────────

def load_jlens(path):
    print(f"Loading J-lens from {path}...", flush=True)
    t0 = time.time()
    d = torch.load(path, map_location="cpu", weights_only=True)
    print(f"  Loaded .pt file in {time.time()-t0:.0f}s, converting...", flush=True)
    result = {}
    for k, v in tqdm(d["J"].items(), desc="Converting layers"):
        result[int(k)] = v.float().numpy()
    print(f"  {len(result)} layers loaded in {time.time()-t0:.0f}s", flush=True)
    return result

def load_wu(cache_dir):
    print(f"Loading W_U from {cache_dir}...", flush=True)
    t0 = time.time()
    # Qwen 3.5 stores lm_head.weight in model-00001-of-00011.safetensors (first shard)
    # Check first few shards only
    for f in sorted(os.listdir(cache_dir)):
        if f.endswith(".safetensors"):
            tensors = safe_load(os.path.join(cache_dir, f))
            if "lm_head.weight" in tensors:
                wu = tensors["lm_head.weight"].float().numpy()
                print(f"  W_U loaded in {time.time()-t0:.0f}s, shape={wu.shape}", flush=True)
                return wu
    raise FileNotFoundError("lm_head.weight not found")

def compute_jlens_vectors(J_matrices, W_U):
    print(f"Computing J-lens vectors W_U @ J for {len(J_matrices)} layers...", flush=True)
    t0 = time.time()
    result = {}
    for l in tqdm(sorted(J_matrices.keys()), desc="J-lens vecs"):
        result[l] = (W_U.astype(np.float64) @ J_matrices[l].astype(np.float64)).astype(np.float32)
    print(f"  Done in {time.time()-t0:.0f}s", flush=True)
    return result

# ═════════════════════════════════════════════════════════════════════
# METRIC 1: CKA
# ═════════════════════════════════════════════════════════════════════

def compute_cka(jlens_vecs, layers, n_tokens=5000):
    """CKA(i,j) = ||Y_i^T Y_j||_F^2 / (||Y_i^T Y_i||_F * ||Y_j^T Y_j||_F)"""
    n_layers = len(layers)
    cka = np.zeros((n_layers, n_layers))
    grams = {}
    for l in tqdm(layers, desc="CKA grams"):
        V = jlens_vecs[l][:n_tokens].astype(np.float64)
        V -= V.mean(axis=0, keepdims=True)
        grams[l] = V @ V.T
    for ii, i in enumerate(layers):
        Gi = grams[i]; ni = np.trace(Gi @ Gi)**0.5
        for jj, j in enumerate(layers):
            if jj < ii: continue
            Gj = grams[j]; nj = np.trace(Gj @ Gj)**0.5
            cka[ii,jj] = cka[jj,ii] = np.trace(Gi @ Gj) / (ni*nj + 1e-10)
    return cka, layers

# ═════════════════════════════════════════════════════════════════════
# METRICS 2-4: Forward passes on WikiText
# ═════════════════════════════════════════════════════════════════════

def get_wikitext_samples(tokenizer, n_samples=50, max_len=256):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test", trust_remote_code=True)
    texts, cur = [], ""
    for item in ds:
        t = item["text"].strip()
        if not t or t.startswith("="):
            if len(cur) > 100: texts.append(cur)
            cur = ""
        else: cur += " " + t
    if len(cur) > 100: texts.append(cur)
    rng = np.random.RandomState(42)
    idxs = rng.choice(len(texts), min(n_samples, len(texts)), replace=False)
    samples = []
    for idx in idxs:
        toks = tokenizer.encode(texts[idx], add_special_tokens=True)
        if len(toks) > max_len:
            start = rng.randint(0, len(toks)-max_len)
            toks = toks[start:start+max_len]
        samples.append(toks)
    return samples

@torch.no_grad()
def collect_readouts(model, jlens_vecs_available, W_U, samples,
                     layers_to_analyze, device, max_samples=20, max_seq_len=128):
    """For each (layer, pos): readout ≈ softmax(W_U @ J_ℓ @ h_ℓ)
    We use jlens_vecs = W_U @ J_ℓ (precomputed), so logits = jlens_vecs @ h.
    The paper uses softmax(W_U @ rms_norm(J_ℓ @ h_ℓ)) but notes the
    approximation without norm is good for top-k metrics."""
    data = {l: {"positions": 0, "correct_top1": 0, "correct_top5": 0,
                "correct_top10": 0, "probs": [], "top1_tokens": []}
            for l in layers_to_analyze}
    sample_every = 20

    for tokens in tqdm(samples[:max_samples], desc="WikiText readouts"):
        if len(tokens) > max_seq_len: tokens = tokens[:max_seq_len]
        input_ids = torch.tensor([tokens]).to(device)
        out = model(input_ids, output_hidden_states=True)
        hs = out.hidden_states
        seq_len = input_ids.shape[1]
        next_tokens = input_ids[0, 1:]

        for l in layers_to_analyze:
            if l not in jlens_vecs_available: continue
            h = hs[l][0]  # [S, D]
            J = torch.from_numpy(jlens_vecs_available[l]).to(device, dtype=h.dtype)
            # logits = W_U @ J_ℓ @ h = jlens_vecs @ h
            # jlens_vecs: [vocab, D], h: [S, D] → logits: [S, vocab]
            logits = (J @ h.T).T  # [S, vocab]
            probs = torch.softmax(logits, dim=-1)
            top1 = logits.argmax(dim=-1)
            n_pos = seq_len - 1
            data[l]["positions"] += n_pos
            data[l]["correct_top1"] += (top1[:-1] == next_tokens).sum().item()
            top5 = logits[:-1].topk(5, dim=-1).indices
            top10 = logits[:-1].topk(10, dim=-1).indices
            nt = next_tokens.unsqueeze(1)
            data[l]["correct_top5"] += (top5 == nt).any(dim=1).sum().item()
            data[l]["correct_top10"] += (top10 == nt).any(dim=1).sum().item()
            for pos in range(0, seq_len-1, sample_every):
                data[l]["probs"].append(probs[pos].cpu().float().numpy())
            data[l]["top1_tokens"].extend(
                (pos, int(top1[pos].item())) for pos in range(seq_len-1))

    metrics = {}
    for l in layers_to_analyze:
        d = data[l]; n = max(d["positions"], 1)
        m = {"top1_acc": d["correct_top1"]/n, "top5_acc": d["correct_top5"]/n,
             "top10_acc": d["correct_top10"]/n}
        if d["probs"]:
            ps = np.stack(d["probs"])
            kp = kurtosis(ps, axis=0, fisher=True)
            m["excess_kurtosis"] = float(np.mean(kp))
        else:
            m["excess_kurtosis"] = 0.0
        metrics[l] = m
    return metrics, data

def compute_autocorrelation(layer_data, layers_to_analyze):
    """Δ log probability: log(P(adjacent same top1) / P(random same))"""
    results = {}
    for l in layers_to_analyze:
        top1 = np.array([t for _,t in layer_data[l]["top1_tokens"]])
        if len(top1) < 2:
            results[l] = {"lag1_delta": 0.0, "lag5_delta": 0.0}
            continue
        same1 = max((top1[:-1] == top1[1:]).mean(), 1e-10)
        _, cnts = np.unique(top1, return_counts=True)
        probs = cnts / cnts.sum(); null = max((probs**2).sum(), 1e-10)
        d1 = np.log(same1 / null)
        d5 = 0.0
        if len(top1) > 5:
            same5 = max((top1[:-5] == top1[5:]).mean(), 1e-10)
            d5 = np.log(same5 / null)
        results[l] = {"lag1_delta": float(d1), "lag5_delta": float(d5)}
    return results

# ═════════════════════════════════════════════════════════════════════
# METRIC 5: Effective rank
# ═════════════════════════════════════════════════════════════════════

def compute_effective_rank(jlens_vecs, layers, n_tokens=5000):
    results = {}
    for l in layers:
        V = jlens_vecs[l][:n_tokens].astype(np.float64)
        V -= V.mean(axis=0, keepdims=True)
        _, S, _ = np.linalg.svd(V, full_matrices=False)
        cumvar = np.cumsum(S**2) / np.sum(S**2)
        results[l] = {
            "eff50": int(np.searchsorted(cumvar, 0.50)+1),
            "eff70": int(np.searchsorted(cumvar, 0.70)+1),
            "eff90": int(np.searchsorted(cumvar, 0.90)+1),
            "eff50_norm": float(np.searchsorted(cumvar, 0.50)/D_MODEL),
            "eff70_norm": float(np.searchsorted(cumvar, 0.70)/D_MODEL),
            "eff90_norm": float(np.searchsorted(cumvar, 0.90)/D_MODEL),
        }
    return results

# ═════════════════════════════════════════════════════════════════════
# Identify boundaries
# ═════════════════════════════════════════════════════════════════════

def identify_boundaries(metrics_234, autocorr, eff_rank, cka_matrix, cka_layers):
    layers = sorted(metrics_234.keys())

    # Workspace END: steepest jump in top-1 accuracy (motor transition)
    top1s = np.array([metrics_234[l]["top1_acc"] for l in layers])
    motor_idx = np.argmax(np.diff(top1s))
    motor_layer = layers[motor_idx + 1]

    # Workspace START: excess kurtosis rises > 10% of max
    kurts = np.array([metrics_234[l]["excess_kurtosis"] for l in layers])
    thresh = 0.10 * max(kurts.max(), 1e-10)
    cands = np.where(kurts > thresh)[0]
    start_layer = layers[cands[0]] if len(cands) > 0 else layers[len(layers)//3]

    # CKA refinement: find where CKA with late layers transitions
    n_cka = len(cka_layers)
    late_cka = cka_matrix[int(n_cka*0.85), :]
    mid = (late_cka.max() + late_cka.min()) / 2
    trans = np.where(np.diff(late_cka > mid))[0]
    sensory_end = cka_layers[trans[0]] if len(trans) > 0 else cka_layers[n_cka//3]

    return {
        "sensory": [0, max(0, start_layer - 1)],
        "workspace": [start_layer, motor_layer - 1],
        "motor": [motor_layer, N_LAYERS - 1],
        "start_layer": start_layer,
        "motor_layer": motor_layer,
        "sensory_end_cka": sensory_end,
    }

# ═════════════════════════════════════════════════════════════════════
# Plot
# ═════════════════════════════════════════════════════════════════════

def plot_all(cka_matrix, cka_layers, metrics, autocorr, eff_rank,
             boundaries, fig_dir):
    import matplotlib.pyplot as plt

    layers = sorted(metrics.keys())
    ws, we = boundaries["workspace"]
    ms = boundaries["motor"][0]

    fig, axes = plt.subplots(2, 3, figsize=(21, 14))

    # CKA
    ax = axes[0,0]
    im = ax.imshow(cka_matrix, cmap="RdBu_r", aspect="auto", origin="lower", vmin=0, vmax=1)
    step = max(1, len(cka_layers)//8)
    ax.set_xticks(range(0, len(cka_layers), step))
    ax.set_xticklabels([str(l) for l in cka_layers[::step]], fontsize=7)
    ax.set_yticks(range(0, len(cka_layers), step))
    ax.set_yticklabels([str(l) for l in cka_layers[::step]], fontsize=7)
    for b_layer, color, label in [(boundaries["start_layer"],"green","Workspace start"),
                                    (boundaries["motor_layer"],"red","Motor start")]:
        idx = min(range(len(cka_layers)), key=lambda i: abs(cka_layers[i]-b_layer))
        ax.axhline(idx, color=color, ls="--", lw=1.5, alpha=0.7)
        ax.axvline(idx, color=color, ls="--", lw=1.5, alpha=0.7)
    ax.set_title("CKA: J-lens vector similarity")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Top-k accuracy
    ax = axes[0,1]
    ax.axvspan(ws, we, alpha=0.1, color="blue", label=f"Workspace L{ws}-L{we}")
    ax.axvspan(ms, N_LAYERS-1, alpha=0.1, color="red", label=f"Motor L{ms}-L{N_LAYERS-1}")
    ax.plot(layers, [metrics[l]["top1_acc"]*100 for l in layers], "o-", c="tab:blue", lw=2, label="Top-1")
    ax.plot(layers, [metrics[l]["top5_acc"]*100 for l in layers], "s-", c="tab:orange", lw=2, label="Top-5")
    ax.plot(layers, [metrics[l]["top10_acc"]*100 for l in layers], "^-", c="tab:green", lw=2, label="Top-10")
    ax.set_xlabel("Layer"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("J-lens next-token prediction accuracy")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Excess kurtosis
    ax = axes[0,2]
    ax.axvspan(ws, we, alpha=0.1, color="blue")
    ax.axvspan(ms, N_LAYERS-1, alpha=0.1, color="red")
    ax.plot(layers, [metrics[l]["excess_kurtosis"] for l in layers], "o-", c="tab:purple", lw=2, ms=5)
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.set_xlabel("Layer"); ax.set_ylabel("Excess kurtosis")
    ax.set_title("J-lens readout excess kurtosis")
    ax.grid(True, alpha=0.3)

    # Autocorrelation
    ax = axes[1,0]
    ax.axvspan(ws, we, alpha=0.1, color="blue")
    ax.axvspan(ms, N_LAYERS-1, alpha=0.1, color="red")
    ax.plot(layers, [autocorr[l]["lag1_delta"] for l in layers], "o-", c="tab:brown", lw=2, ms=5, label="Lag-1")
    ax.plot(layers, [autocorr[l]["lag5_delta"] for l in layers], "s-", c="tab:pink", lw=2, ms=5, label="Lag-5")
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.set_xlabel("Layer"); ax.set_ylabel("Δ log probability")
    ax.set_title("Autocorrelation of top-1 J-lens token")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Effective rank
    ax = axes[1,1]
    ax.axvspan(ws, we, alpha=0.1, color="blue")
    ax.axvspan(ms, N_LAYERS-1, alpha=0.1, color="red")
    eff_l = sorted(eff_rank.keys())
    ax.plot(eff_l, [eff_rank[l]["eff50"]/D_MODEL*100 for l in eff_l], "o-", c="tab:cyan", lw=2, label="@50%")
    ax.plot(eff_l, [eff_rank[l]["eff70"]/D_MODEL*100 for l in eff_l], "s-", c="tab:olive", lw=2, label="@70%")
    ax.plot(eff_l, [eff_rank[l]["eff90"]/D_MODEL*100 for l in eff_l], "^-", c="tab:gray", lw=2, label="@90%")
    ax.set_xlabel("Layer"); ax.set_ylabel("% of d_model")
    ax.set_title("Effective rank of J-lens vectors")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Summary
    ax = axes[1,2]
    ax.axis("off")
    b = boundaries
    txt = f"""
Qwen3.5-27B Workspace Boundaries
{'='*32}

SENSORY:  L{b['sensory'][0]} – L{b['sensory'][1]}
  J-space carries little content

WORKSPACE: L{b['workspace'][0]} – L{b['workspace'][1]}
  ({b['workspace'][1]-b['workspace'][0]+1} layers)
  Abstract concepts, broadcast format

MOTOR:    L{b['motor'][0]} – L{b['motor'][1]}
  ({b['motor'][1]-b['motor'][0]+1} layers)
  Next-token prediction regime

Detection method (per paper):
• Start: excess kurtosis rises
• End:   top-1 accuracy jumps steeply
• CKA confirms block structure
"""
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=10,
            fontfamily="monospace", verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle("Workspace Layer Identification — Qwen3.5-27B",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    path = fig_dir / "workspace_identification.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}", flush=True)

# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Identify workspace layers")
    p.add_argument("--j-path", default="results/jspace_cache/qwen3.5-27b/jlens/"
                   "Salesforce-wikitext/Qwen3.5-27B_jacobian_lens.pt")
    p.add_argument("--model-path", default=os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen3.5-27B/"
        "snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654"))
    p.add_argument("--output-dir", default="results/workspace_id")
    p.add_argument("--smoke-test", type=int, default=0)
    p.add_argument("--n-wikitext", type=int, default=20,
                   help="WikiText samples for readout metrics")
    p.add_argument("--max-seq-len", type=int, default=128)
    args = p.parse_args()

    OUT = Path(args.output_dir)
    FIG = Path("figures") / OUT.name
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    print("="*60, flush=True)
    print("WORKSPACE LAYER IDENTIFICATION", flush=True)
    print("="*60, flush=True)

    # Load
    J_matrices = load_jlens(args.j_path)
    print("J-lens loaded, now loading W_U...", flush=True)
    W_U = load_wu(args.model_path)
    print("Computing J-lens vectors...", flush=True)
    jlens_vecs = compute_jlens_vectors(J_matrices, W_U)
    layers_available = sorted(jlens_vecs.keys())
    print(f"J-lens available for layers: {layers_available[0]}-{layers_available[-1]} "
          f"({len(layers_available)} layers)")

    # Layer sets
    cka_layers = list(range(0, N_LAYERS, CKA_SUBSAMPLE))
    # For metrics 2-5, use all available J-lens layers
    metric_layers = layers_available

    # Metric 1: CKA
    cka_mat, cka_l = compute_cka(jlens_vecs, cka_layers)

    # Metrics 2-4: Need model + WikiText
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, output_hidden_states=True)
    model.eval()
    device = next(model.parameters()).device
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    n_wiki = min(args.n_wikitext, 5) if args.smoke_test > 0 else args.n_wikitext
    samples = get_wikitext_samples(tokenizer, n_samples=max(50, n_wiki*3),
                                   max_len=args.max_seq_len)
    readout_metrics, readout_data = collect_readouts(
        model, jlens_vecs, W_U, samples, metric_layers, device,
        max_samples=n_wiki, max_seq_len=args.max_seq_len)
    autocorr = compute_autocorrelation(readout_data, metric_layers)

    # Metric 5: Effective rank
    eff_rank = compute_effective_rank(jlens_vecs, metric_layers)

    # Identify
    boundaries = identify_boundaries(readout_metrics, autocorr, eff_rank,
                                     cka_mat, cka_l)

    # Save
    results = {
        "cka_layers": cka_l,
        "cka_matrix": cka_mat.tolist(),
        "readout_metrics": {str(l): v for l, v in readout_metrics.items()},
        "autocorrelation": {str(l): v for l, v in autocorr.items()},
        "effective_rank": {str(l): v for l, v in eff_rank.items()},
        "boundaries": {k: [int(x) for x in v] if isinstance(v, list) else int(v)
                       for k, v in boundaries.items()},
    }
    with open(OUT / "workspace_boundaries.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {OUT / 'workspace_boundaries.json'}")

    # Plot
    plot_all(cka_mat, cka_l, readout_metrics, autocorr, eff_rank, boundaries, FIG)

    print("\n" + "="*60)
    print("WORKSPACE BOUNDARIES:")
    print(f"  Sensory:  L{boundaries['sensory'][0]} – L{boundaries['sensory'][1]}")
    print(f"  Workspace: L{boundaries['workspace'][0]} – L{boundaries['workspace'][1]}")
    print(f"  Motor:    L{boundaries['motor'][0]} – L{boundaries['motor'][1]}")
    print("="*60)

if __name__ == "__main__":
    main()
