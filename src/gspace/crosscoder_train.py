"""
Train a TopK Sparse Crosscoder on Qwen3.5-27B residual stream activations
for modular addition (a+b mod 59).

Based on OpenInterpretability notebook 17_train_crosscoder.ipynb
(Anthropic 2024 crosscoder recipe: TopK sparsity, geometric median init,
cosine LR schedule with warmup).

Usage:
    python crosscoder_train.py [--modulus 59] [--n_features 4096] [--k_topk 32] \
        [--token_budget 50000000] [--batch_size 2048] [--layers 48-63]
"""
import os, math, json, time, hashlib, argparse, random, sys
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# ─── Config ───
MODEL_ID = "Qwen/Qwen3.5-27B"
DEVICE = "cuda:0"
DTYPE = torch.bfloat16

# ─── CLI args ───
parser = argparse.ArgumentParser()
parser.add_argument("--modulus", type=int, default=59)
parser.add_argument("--n_features", type=int, default=4096, help="Dict size")
parser.add_argument("--k_topk", type=int, default=32, help="TopK sparsity")
parser.add_argument("--token_budget", type=int, default=50_000_000)
parser.add_argument("--batch_size", type=int, default=2048)
parser.add_argument("--fwd_batch", type=int, default=4, help="Seqs per fwd pass")
parser.add_argument("--seq_len", type=int, default=256)
parser.add_argument("--layer_start", type=int, default=48)
parser.add_argument("--layer_end", type=int, default=63)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--warmup", type=int, default=500)
parser.add_argument("--output_dir", type=str, default="results/crosscoder")
parser.add_argument("--log_every", type=int, default=200)
parser.add_argument("--save_every", type=int, default=5000)
parser.add_argument("--smoke_test", type=int, default=0,
    help="Run N training steps only, verify save/resume, then exit")
args = parser.parse_args()

LAYERS = list(range(args.layer_start, args.layer_end + 1))
N_FEATURES = args.n_features
K_TOPK = args.k_topk
N_LAYERS = len(LAYERS)
D_MODEL = 5120
MODULUS = args.modulus
LR = args.lr
WARMUP = args.warmup
SEQ_LEN = args.seq_len
FWD_BATCH = args.fwd_batch
BATCH_SIZE = args.batch_size

OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Checkpoint paths ───
CKPT_DIR = OUTPUT_DIR / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)

# ─── Print config ───
print(f"Model: {MODEL_ID}, device: {DEVICE}, dtype: {DTYPE}")
print(f"Layers: {LAYERS} ({N_LAYERS} layers, d_model={D_MODEL})")
print(f"Crosscoder: {N_FEATURES} features, K={K_TOPK}")
print(f"Training: {args.token_budget} tokens, batch={BATCH_SIZE}, lr={LR}")
print(f"Output: {OUTPUT_DIR}")


# ====================================================================
# 1. Data generator: modular addition prompts
# ====================================================================

def generate_mod_prompt(a, b, modulus):
    """Generate a natural-language prompt for modular addition."""
    templates = [
        f"Compute {a} + {b} modulo {modulus}. The answer is",
        f"What is ({a} + {b}) mod {modulus}? The result is",
        f"Calculate {a} plus {b} in Z/{modulus}Z. Answer:",
        f"Find ({a} + {b}) % {modulus}. Solution:",
        f"In modulo {modulus} arithmetic, {a} + {b} =",
    ]
    return random.choice(templates)


def mod_text_stream():
    """Infinite generator of modular-addition text strings."""
    modulus = MODULUS
    while True:
        # Generate a multi-example context
        n_examples = random.randint(3, 8)
        parts = []
        for _ in range(n_examples):
            a, b = random.randrange(modulus), random.randrange(modulus)
            prompt = generate_mod_prompt(a, b, modulus)
            answer = str((a + b) % modulus)
            parts.append(f"{prompt} {answer}.")
        yield " ".join(parts)


# ====================================================================
# 2. Activation extraction
# ====================================================================

class MultiLayerHook:
    """Capture residual-stream output at specified layers during forward pass."""

    def __init__(self, blocks, layers):
        self.layers = layers
        self.bufs = {l: None for l in layers}
        self.handles = []
        for l in layers:
            h = blocks[l].register_forward_hook(self._make(l))
            self.handles.append(h)

    def _make(self, l):
        def hook(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            self.bufs[l] = h.detach()
        return hook

    def pop_stack(self):
        stacked = torch.stack([self.bufs[l] for l in self.layers], dim=0)
        for l in self.layers:
            self.bufs[l] = None
        return stacked

    def close(self):
        for h in self.handles:
            h.remove()


def activation_stream(model, tok, blocks, layers, d_model,
                      seq_len=SEQ_LEN, fwd_batch=FWD_BATCH, batch_size=BATCH_SIZE):
    """Yields float32 tensors of shape (BATCH_SIZE, L*D_MODEL) forever."""
    hooker = MultiLayerHook(blocks, layers)
    L = len(layers)
    text_iter = mod_text_stream()
    pending = []
    try:
        while True:
            batch_texts = []
            while len(batch_texts) < fwd_batch:
                batch_texts.append(next(text_iter))
            enc = tok(batch_texts, return_tensors="pt", max_length=seq_len,
                      truncation=True, padding="max_length")
            ids = enc["input_ids"].to(model.device)
            with torch.no_grad():
                model(ids)
            stacked = hooker.pop_stack()          # (L, B, T, D) bf16
            stacked = stacked.permute(1, 2, 0, 3).contiguous()  # (B, T, L, D)
            flat = stacked.reshape(-1, L * d_model).float()     # (B*T, L*D)
            pending.append(flat.cpu())
            rows = sum(p.shape[0] for p in pending)
            while rows >= batch_size:
                pool = torch.cat(pending, dim=0)
                perm = torch.randperm(pool.shape[0])
                pool = pool[perm]
                out = pool[:batch_size]
                leftover = pool[batch_size:]
                pending = [leftover] if leftover.shape[0] else []
                rows = leftover.shape[0]
                yield out.to(DEVICE, non_blocking=True)
    finally:
        hooker.close()


# ====================================================================
# 3. Crosscoder model (TopK, like Anthropic 2024)
# ====================================================================

class CrossCoder(nn.Module):
    def __init__(self, n_features: int, n_layers: int, d_model: int, k_topk: int):
        super().__init__()
        self.N = n_features
        self.L = n_layers
        self.D = d_model
        self.K = k_topk
        in_dim = n_layers * d_model

        self.W_enc = nn.Parameter(torch.empty(in_dim, n_features, dtype=torch.float32))
        self.W_dec = nn.Parameter(torch.empty(n_features, in_dim, dtype=torch.float32))
        self.b_enc = nn.Parameter(torch.zeros(n_features, dtype=torch.float32))
        self.b_dec = nn.Parameter(torch.zeros(in_dim, dtype=torch.float32))

        nn.init.xavier_uniform_(self.W_enc, gain=1.0 / math.sqrt(n_layers))
        nn.init.normal_(self.W_dec, std=1.0 / math.sqrt(n_features))
        with torch.no_grad():
            self.renorm_decoder_()

    @torch.no_grad()
    def renorm_decoder_(self):
        """Unit L2 norm per (feature, layer) slice."""
        W = self.W_dec.data.view(self.N, self.L, self.D)
        norms = W.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        W.div_(norms)

    @torch.no_grad()
    def geometric_median_init_b_dec_(self, x_samples, iters=64):
        y = x_samples.mean(dim=0).float()
        for _ in range(iters):
            d = (x_samples - y).norm(dim=-1).clamp_min(1e-6)
            w = 1.0 / d
            y = (w[:, None] * x_samples).sum(0) / w.sum()
        self.b_dec.data.copy_(y)

    def encode(self, x):
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        acts = F.relu(pre)
        vals, idx = acts.topk(self.K, dim=-1)
        z = torch.zeros_like(acts)
        z.scatter_(-1, idx, vals)
        return z

    def decode(self, z):
        return z @ self.W_dec + self.b_dec

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def per_layer_decoder_norms(self):
        W = self.W_dec.detach().view(self.N, self.L, self.D)
        return W.norm(dim=-1).cpu()


# ====================================================================
# 4. Training
# ====================================================================

def lr_at(step, total_steps, warmup=WARMUP, base=LR):
    if step < warmup:
        return base * (step + 1) / warmup
    prog = (step - warmup) / max(1, total_steps - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * prog))


def load_checkpoint(cc, opt, path):
    """Load crosscoder weights and optimizer state from a safetensors checkpoint."""
    from safetensors.torch import load_file
    sd = load_file(str(path))
    cc.W_enc.data.copy_(sd["W_enc"])
    cc.W_dec.data.copy_(sd["W_dec"])
    cc.b_enc.data.copy_(sd["b_enc"])
    cc.b_dec.data.copy_(sd["b_dec"])
    # Also try to load optimizer state
    opt_path = str(path).replace(".safetensors", "_opt.pt")
    start_step = 0
    if os.path.exists(opt_path):
        opt_state = torch.load(opt_path, map_location=DEVICE)
        opt.load_state_dict(opt_state["optimizer"])
        start_step = opt_state["step"] + 1
        print(f"Resumed optimizer from step {start_step}")
    return start_step


def save_checkpoint(cc, opt, step, losses):
    """Save crosscoder weights + optimizer state + losses."""
    ckpt_path = CKPT_DIR / f"crosscoder_step{step:07d}.safetensors"
    save_file({
        "W_enc": cc.W_enc.data,
        "W_dec": cc.W_dec.data,
        "b_enc": cc.b_enc.data,
        "b_dec": cc.b_dec.data,
    }, str(ckpt_path))
    # Save optimizer state
    opt_path = CKPT_DIR / f"crosscoder_step{step:07d}_opt.pt"
    torch.save({"optimizer": opt.state_dict(), "step": step}, str(opt_path))
    # Save losses
    with open(OUTPUT_DIR / "training_losses.json", "w") as f:
        json.dump(losses, f)
    print(f"  Saved checkpoint at step {step}")


def find_latest_checkpoint():
    """Find the latest checkpoint by step number."""
    if not CKPT_DIR.exists():
        return None, 0
    ckpts = list(CKPT_DIR.glob("crosscoder_step*.safetensors"))
    if not ckpts:
        return None, 0
    # Extract step number from filename
    def step_num(p):
        import re
        m = re.search(r'step(\d+)', p.name)
        return int(m.group(1)) if m else 0
    latest = max(ckpts, key=step_num)
    return latest, step_num(latest)


def train():
    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, device_map=DEVICE,
        trust_remote_code=True,
    )
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    blocks = model.model.layers  # Qwen architecture

    # Initialize crosscoder
    print(f"Initializing crosscoder ({N_FEATURES} features, {N_LAYERS} layers)...")
    cc = CrossCoder(N_FEATURES, N_LAYERS, D_MODEL, K_TOPK).to(DEVICE)

    # Try to resume from checkpoint
    latest_ckpt, resume_step = find_latest_checkpoint()
    
    stream = activation_stream(model, tok, blocks, LAYERS, D_MODEL)
    
    # Optimizer
    opt = torch.optim.Adam(cc.parameters(), lr=LR, betas=(0.9, 0.999))

    if latest_ckpt is None:
        # Fresh init: geometric median for b_dec
        with torch.no_grad():
            probe = next(stream).float()
            cc.geometric_median_init_b_dec_(probe, iters=64)
        del probe
        torch.cuda.empty_cache()
        losses = []
        start_step = 0
    else:
        print(f"Resuming from {latest_ckpt.name} (step {resume_step})")
        losses = []
        if (OUTPUT_DIR / "training_losses.json").exists():
            with open(OUTPUT_DIR / "training_losses.json") as f:
                losses = json.load(f)
        loaded_step = load_checkpoint(cc, opt, latest_ckpt)
        start_step = max(resume_step, loaded_step) + 1

    n_params = sum(p.numel() for p in cc.parameters())
    print(f"Crosscoder params: {n_params/1e6:.2f}M (fp32)")
    
    total_steps = args.token_budget // BATCH_SIZE

    # Smoke test: override budget and save_every for fast validation
    if args.smoke_test > 0:
        smoke_steps = min(args.smoke_test, total_steps - start_step)
        total_steps = start_step + smoke_steps
        args.save_every = max(1, smoke_steps // 3)  # save at ~1/3 and ~2/3 and final
        args.log_every = 1
        print(f"SMOKE TEST MODE: {smoke_steps} steps, save every {args.save_every}")

    remaining = total_steps - start_step
    print(f"Training from step {start_step} to {total_steps} ({remaining:,} steps remaining).")

    # Training loop
    pbar = tqdm(range(start_step, total_steps), desc="Training crosscoder", initial=start_step, total=total_steps)
    try:
        for step in pbar:
            lr = lr_at(step, total_steps)
            for g in opt.param_groups:
                g["lr"] = lr

            # Get batch
            batch = next(stream).float()

            # Forward
            x_hat, z = cc(batch)
            mse = F.mse_loss(x_hat, batch)
            loss = mse

            # Backward
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cc.parameters(), max_norm=1.0)
            opt.step()

            # Renorm decoder
            with torch.no_grad():
                cc.renorm_decoder_()

            # Track
            l0 = (z > 1e-6).float().sum(dim=-1).mean().item()
            losses.append({"step": step, "mse": mse.item(), "l0": l0, "lr": lr})

            if step % args.log_every == 0:
                pbar.set_postfix(mse=f"{mse.item():.4f}", l0=f"{l0:.1f}", lr=f"{lr:.2e}")

            if (step + 1) % args.save_every == 0:
                save_checkpoint(cc, opt, step, losses)

    except KeyboardInterrupt:
        print("Interrupted, saving...")
    finally:
        save_checkpoint(cc, opt, step, losses)
        stream_hooker = None  # will be closed by finally in activation_stream
        print(f"Final checkpoint saved to {OUTPUT_DIR}")

    # Smoke test: verify resume works from the last checkpoint
    if args.smoke_test > 0:
        print("\n=== Smoke test: verifying resume... ===")
        latest, latest_step = find_latest_checkpoint()
        if latest is not None:
            cc2 = CrossCoder(N_FEATURES, N_LAYERS, D_MODEL, K_TOPK).to(DEVICE)
            opt2 = torch.optim.Adam(cc2.parameters(), lr=LR, betas=(0.9, 0.999))
            loaded = load_checkpoint(cc2, opt2, latest)
            # Run 3 more steps to verify no crash
            for i in range(3):
                batch = next(stream).float()
                x_hat, z = cc2(batch)
                loss = F.mse_loss(x_hat, batch)
                loss.backward()
                opt2.step()
                opt2.zero_grad()
            print(f"Resume OK: loaded step {loaded}, ran 3 more steps, final MSE={loss.item():.4f}")
            del cc2, opt2
        torch.cuda.empty_cache()
        print("=== Smoke test PASSED ===\n")

    return cc, losses


def save_weights_only(cc, step):
    ckpt_path = CKPT_DIR / f"crosscoder_step{step:07d}.safetensors"
    save_file({
        "W_enc": cc.W_enc.data,
        "W_dec": cc.W_dec.data,
        "b_enc": cc.b_enc.data,
        "b_dec": cc.b_dec.data,
    }, str(ckpt_path))
    print(f"  Saved weights at step {step}")


# ====================================================================
# 5. Analysis
# ====================================================================

def analyze(cc, losses, output_dir):
    """Post-training analysis: per-layer feature contribution, fit to modular data."""
    print("\n=== Post-training analysis ===")

    # Per-layer decoder norms
    norms = cc.per_layer_decoder_norms()  # (N, L)
    top_features_per_layer = norms.argmax(dim=1)  # which layer each feature is strongest in

    # Distribution of features across layers
    layer_counts = np.bincount(top_features_per_layer.numpy(), minlength=N_LAYERS)

    # Plot 1: Training loss curve
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    steps = [d["step"] for d in losses]
    mses = [d["mse"] for d in losses]
    ax.plot(steps, mses, alpha=0.3, linewidth=0.5, label="raw")
    # Rolling average
    window = max(1, len(mses) // 200)
    smoothed = np.convolve(mses, np.ones(window)/window, mode="valid")
    ax.plot(steps[window-1:], smoothed, "r-", linewidth=1.5, label=f"avg({window})")
    ax.set_xlabel("Step"); ax.set_ylabel("MSE")
    ax.set_title("Training MSE"); ax.legend(); ax.grid(True, alpha=0.3)

    # Plot 2: L0 (active features)
    ax = axes[1]
    l0s = [d["l0"] for d in losses]
    ax.plot(steps, l0s, alpha=0.3, linewidth=0.5)
    smoothed_l0 = np.convolve(l0s, np.ones(window)/window, mode="valid")
    ax.plot(steps[window-1:], smoothed_l0, "g-", linewidth=1.5)
    ax.set_xlabel("Step"); ax.set_ylabel("L0")
    ax.set_title("Active features (L0)"); ax.grid(True, alpha=0.3)

    # Plot 3: Feature distribution across layers
    ax = axes[2]
    ax.bar(range(N_LAYERS), layer_counts, color="steelblue", alpha=0.8)
    ax.set_xlabel("Layer index"); ax.set_ylabel("Feature count")
    ax.set_title("Features by dominant layer")
    ax.set_xticks(range(0, N_LAYERS, 4))
    ax.set_xticklabels([str(LAYERS[i]) for i in range(0, N_LAYERS, 4)])
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"Crosscoder ({N_FEATURES} feats, K={K_TOPK}) — {MODEL_ID} layers {LAYERS[0]}-{LAYERS[-1]}", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "crosscoder_training.png", dpi=150, bbox_inches="tight")
    print(f"Saved {output_dir / 'crosscoder_training.png'}")

    # Save config
    config = {
        "model_id": MODEL_ID, "modulus": MODULUS,
        "n_features": N_FEATURES, "k_topk": K_TOPK,
        "layers": LAYERS, "d_model": D_MODEL,
        "token_budget": args.token_budget, "batch_size": BATCH_SIZE,
        "lr": LR, "warmup": WARMUP,
        "final_mse": losses[-1]["mse"] if losses else None,
        "final_l0": losses[-1]["l0"] if losses else None,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    return norms


# ====================================================================
# Main
# ====================================================================

if __name__ == "__main__":
    cc, losses = train()
    norms = analyze(cc, losses, OUTPUT_DIR)
    print("Done!")
