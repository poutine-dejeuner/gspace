"""
True J-space computation + g-space vs J-space comparison.

1. Compute J_ℓ = E[∂h_final/∂h_ℓ] averaged over token positions and promp=ts
2. For each layer: g-space (crosscoder) → logit regression → R² curve
3. For each layer: J-lens answer rank → rank curve
4. At peak J-space layer: decompose activations via sparse nonnegative gradient pursuit,
   compare crosscoder decoder directions to J-lens vectors.

Usage:
    python -m gspace.jspace [--n_prompts 500] [--n_layers 16] [--k_sparse 25]
"""

import argparse
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Config ───
MODULUS = 59
D_MODEL = 5120  # Qwen3.5-27B

parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, default="Qwen/Qwen3.5-27B")
parser.add_argument("--crosscoder", type=str, default="results/crosscoder")
parser.add_argument("--activations_path", type=str, default="results/phase1_q35/activations.npz")
parser.add_argument("--output_dir", type=str, default="results/jspace")
parser.add_argument("--n_prompts", type=int, default=500,
    help="Number of mod59 prompts for Jacobian averaging")
parser.add_argument("--layers", type=int, nargs="+", default=list(range(48, 64)),
    help="Layers to analyze (default: 48-63)")
parser.add_argument("--k_sparse", type=int, default=25,
    help="Sparsity level for gradient pursuit decomposition")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--dtype", type=str, default="bfloat16")
parser.add_argument("--load_in_8bit", action="store_true",
    help="Load model in 8-bit for memory savings")
parser.add_argument("--smoke_test", type=int, default=0,
    help="If > 0, run only N steps to verify pipeline works")
args = parser.parse_args()

LAYERS = list(args.layers)
N_LAYERS = len(LAYERS)
OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Smoke test override ───
if args.smoke_test > 0:
    args.n_prompts = min(args.n_prompts, args.smoke_test)
    print(f"[SMOKE TEST] n_prompts={args.n_prompts}")

# ─────────────────────────────────────────────────────────────────────
# STEP 0: Load model and cached activations
# ─────────────────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 0: Loading model and data")
print("=" * 60)

from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = getattr(torch, args.dtype)

tokenizer = AutoTokenizer.from_pretrained(args.model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Loading {args.model_id}...")
if args.load_in_8bit:
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map="auto",
        load_in_8bit=True,
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map="auto",
    )
model.eval()

n_total_layers = model.config.num_hidden_layers
vocab_size = model.config.vocab_size
W_U = model.lm_head.weight.detach()  # [vocab, d_model]

print(f"  Total layers: {n_total_layers}, d_model: {D_MODEL}, vocab: {vocab_size}")
print(f"  Analyzing layers: {LAYERS}")

# Load cached activations (for crosscoder features and probing)
print("Loading cached activations...")
act_data = np.load(args.activations_path)
train_labels = act_data["train_labels"]
train_acts = {}
for key in sorted(act_data.keys()):
    m = re.match(r'^train_l(\d+)$', key)
    if m:
        l = int(m.group(1))
        if l in LAYERS:
            train_acts[l] = act_data[key]
print(f"  Loaded activations for {len(train_acts)} layers, {len(train_labels)} samples")

# ─────────────────────────────────────────────────────────────────────
# STEP 1: Compute true J-lens vectors via averaged Jacobian
# ─────────────────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 1: Computing J-lens vectors")
print("=" * 60)

# For efficiency, we hook into each target layer during backward pass.
# We need: J_ℓ = E[ ∂h_final / ∂h_ℓ ] over (t, t'≥t, prompt)
# h_final is the residual stream at position after the prompt.
# Since we have mod59 prompts "Compute a + b mod 59: a + b mod 59 =",
# we take h_final at the last token position (t' = -1).
# For ∂h_final / ∂h_ℓ, we only need the Jacobian at the last position
# with respect to the last position (causal attention ensures earlier
# positions don't affect this for the same or future positions).

# Generate a fresh set of mod59 prompts for unbiased Jacobian estimation
import random
rng = random.Random(42)
all_pairs = [(a, b) for a in range(MODULUS) for b in range(MODULUS)]
rng.shuffle(all_pairs)
j_pairs = all_pairs[:args.n_prompts]

def make_prompt(a, b):
    return f"Compute {a} + {b} mod 59: {a} + {b} mod 59 ="

# Pre-compute the correct answer token IDs
# The answer is a number 0-116, tokenized as decimal string
# We need the token ID for each answer string
answer_token_ids = {}
for ans in range(MODULUS * 2):
    tok = tokenizer.encode(str(ans), add_special_tokens=False)
    if len(tok) == 1:
        answer_token_ids[ans] = tok[0]
    else:
        # Multi-token answer (shouldn't happen for 0-116 in Qwen)
        answer_token_ids[ans] = tok  # store list

print(f"  Answer token IDs: {len(answer_token_ids)} unique")

# We'll accumulate Jacobian averages
# For each layer ℓ, J_ℓ is (d_model, d_model), computed via
# ∂h_final / ∂h_ℓ = Jacobian of final residual at last pos w.r.t layer ℓ residual at last pos
jac_accum = {l: np.zeros((D_MODEL, D_MODEL), dtype=np.float32) for l in LAYERS}
jac_counts = {l: 0 for l in LAYERS}

# To compute Jacobians efficiently, we'll use torch's autograd per batch

# Register hooks to capture activations at each target layer
layer_activations = {}

def make_hook(l):
    def hook(module, input, output):
        # output[0] is [batch, seq, d_model]
        layer_activations[l] = output[0]
    return hook

hooks = []
for l in LAYERS:
    hook = model.model.layers[l].register_forward_hook(make_hook(l))
    hooks.append(hook)

print(f"  Computing Jacobians over {args.n_prompts} prompts...")

with torch.no_grad():
    for i in tqdm(range(0, args.n_prompts, args.batch_size), desc="Jacobian"):
        batch_pairs = j_pairs[i:i + args.batch_size]
        prompts = [make_prompt(a, b) for a, b in batch_pairs]
        answers = [(a + b) % MODULUS for a, b in batch_pairs]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        seq_len = inputs["input_ids"].shape[1]

        # We need the final residual at the LAST token position
        # Hack: we do a forward pass and capture final hidden states
        # Then for each layer, we compute the Jacobian via backward

        # First, get the model output to capture hidden states
        outputs = model(**inputs, output_hidden_states=True)
        final_hidden = outputs.hidden_states[-1]  # [batch, seq, d_model]

        # For each sample in batch, get last position
        last_positions = inputs["attention_mask"].sum(dim=1) - 1  # [batch]

        for b_idx in range(len(batch_pairs)):
            ans = answers[b_idx]
            ans_tok = answer_token_ids.get(ans)
            if ans_tok is None or isinstance(ans_tok, list):
                continue  # skip multi-token answers

            last_pos = last_positions[b_idx].item()

            # Compute logit for correct answer at last position
            h_final_sample = final_hidden[b_idx, last_pos]  # [d_model]

            # We need ∂(logit_correct) / ∂h_ℓ for each ℓ
            # We'll do a per-layer backward from h_final into h_ℓ
            # More efficient: do a single backward from logit through the whole model

            # Actually let's do it more efficiently:
            # Backward from logit_answer w.r.t. _all_ layer activations simultaneously
            # by using the captured activations and letting autograd trace from there.

            # But autograd doesn't trace through hooks. We need to re-run with requires_grad.
            # Strategy: re-run the specific layers with autograd enabled.

            # Actually, the most efficient approach for Jacobians:
            # Use torch.autograd.grad for each layer separately.
            # Re-run from layer ℓ to final for each ℓ? No, too expensive.

            # Better: compute the Jacobian of the output w.r.t. the input of each transformer
            # layer block. The transformer block is sequential: h_{ℓ+1} = h_ℓ + f_ℓ(h_ℓ).
            # But since f_ℓ involves attention across positions, it's not per-position independent.

            # For a causal LM, ∂h_final[b, last_pos] / ∂h_ℓ[b, last_pos] is well-defined
            # (attention only looks at current and past positions).

            # Simplest correct approach: For each sample, run forward with autograd
            # from layer ℓ to output. We cache the forward input to layer ℓ, then
            # compute gradient of logit_answer w.r.t. that input.

    # This approach is O(n_layers * n_prompts) forward passes which is too expensive.
    # Let's use a more efficient method.

# Remove hooks
for hook in hooks:
    hook.remove()

# ─── Efficient Jacobian computation ───
# Key insight: we don't need the full (d_model, d_model) Jacobian.
# We need the J-lens vectors, which are rows of W_U · J_ℓ.
# For each token v, the J-lens vector is:
#   j_v^ℓ = (W_U · J_ℓ)[v, :] = J_ℓ^T · W_U[v, :]
#
# But J_ℓ^T is the VJP (vector-Jacobian product) of h_ℓ w.r.t. h_final.
# Specifically, for a vector u in final space:
#   J_ℓ^T u = ∂(u^T h_final) / ∂h_ℓ
# Setting u = W_U[v, :] (the unembedding vector for token v):
#   j_v^ℓ = ∂(logit_v) / ∂h_ℓ  averaged over t, t', prompts
#
# So we can compute J-lens vectors directly via gradient of logit w.r.t. h_ℓ,
# averaged over positions and prompts!

print("  Computing J-lens vectors via gradient accumulation...")

# For efficiency, we compute J-lens vectors for ALL tokens simultaneously
# by accumulating ∂logit_v / ∂h_ℓ for each v, but that's vocab_size * d_model.
# Instead, we compute ∂h_final / ∂h_ℓ as a matrix and then multiply by W_U.

# Even better: compute J-lens vectors on the fly via backward through the model.
# We'll hook the gradient of h_final w.r.t. h_ℓ by using autograd.

# Strategy for efficient computation per batch:
# 1. Forward pass with output_hidden_states=True to get h_ℓ for all ℓ
# 2. For each (sample, position), select h_final at that position
# 3. Compute logit for correct answer, backward to get ∂logit/∂h_ℓ
# 4. Accumulate

jac_lens_accum = {l: np.zeros((vocab_size, D_MODEL), dtype=np.float32) for l in LAYERS}
jac_counts2 = {l: 0 for l in LAYERS}

# We'll process one sample at a time for correctness (small batches with autograd)
for i in tqdm(range(args.n_prompts), desc="J-lens vectors"):
    a, b = j_pairs[i]
    prompt = make_prompt(a, b)
    ans = (a + b) % MODULUS
    ans_tok = answer_token_ids.get(ans)
    if ans_tok is None or isinstance(ans_tok, list):
        continue

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    seq_len = inputs["input_ids"].shape[1]
    last_pos = inputs["attention_mask"].sum(dim=1).item() - 1

    # Run forward with output_hidden_states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # For each target layer, compute ∂logit_correct / ∂h_ℓ at last position
    for l_idx, l in enumerate(LAYERS):
        h_l = outputs.hidden_states[l][0, last_pos].detach().clone()
        h_l.requires_grad_(True)

        # We need to run from layer l to final:
        # This requires re-running the forward pass from h_l, which is expensive.
        pass

    break  # DEBUG: just one iteration for now


# ─── CLEANER APPROACH ───
# Instead of per-layer re-forward, run once and capture gradients
# via torch.autograd.grad on the hidden states.

# The key: model(**inputs, output_hidden_states=True) with retain_graph
# and then grad of logit w.r.t. each hidden state.

print("  Computing J-lens via autograd.grad on hidden states...")

# Process in small batches
all_jac_lens = {l: np.zeros((vocab_size, D_MODEL), dtype=np.float32) for l in LAYERS}
all_counts = {l: 0 for l in LAYERS}

for b_start in tqdm(range(0, args.n_prompts, args.batch_size), desc="Gradient batches"):
    b_end = min(b_start + args.batch_size, args.n_prompts)
    batch_pairs = j_pairs[b_start:b_end]
    prompts = [make_prompt(a, b) for a, b in batch_pairs]

    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    bsz = inputs["input_ids"].shape[0]
    last_positions = inputs["attention_mask"].sum(dim=1) - 1  # [batch]

    # Forward pass with autograd enabled
    # We need gradients of output w.r.t. hidden states.
    # transformers output_hidden_states returns detached tensors by default.
    # We need to manually trigger gradient computation.

    # Approach: use register_full_backward_hook on each layer
    # Then do a backward pass from logit to accumulate VJP.

    embeds = model.model.embed_tokens(inputs["input_ids"])

    # Manual forward through layers, tracking gradients
    # (This is complex. Let's use an alternative.)

# ─── SIMPLEST CORRECT APPROACH ───
# Run forward, then for each layer, wrap h_ℓ in a tensor with grad,
# and re-run from that layer. But we can optimize:
#
#   logit_correct = W_U[ans_tok] @ h_final
#                = W_U[ans_tok] @ (h_ℓ + f_{ℓ+1} + ... + f_L)
#
# ∂logit / ∂h_ℓ = W_U[ans_tok] · ∂h_final/∂h_ℓ
#
# The Jacobian ∂h_final/∂h_ℓ across layers accounts for attention
# and MLP transformations. We can compute it by running autograd
# from h_ℓ to logit for each layer.

# Let's use a practical approach: run the model, save hidden states,
# then compute the gradient of logit_answer w.r.t. each hidden state
# using torch.autograd.grad. We need hidden states to be leaf tensors
# in the computation graph.

# Critical trick: we manually build the computation from h_ℓ onward.
# For each layer ℓ, we need the mapping h_ℓ → h_final.
# We can get this by calling just the later layers.

# Let's use the model's internal layer structure.
# Qwen3.5: model.model.layers[l], model.model.norm, lm_head
# We can call model.model(...) with start_layer parameter? No.

# Simplest: re-run forward for each layer, but only once per sample
# using torch's autograd. We'll use the hidden states from a full forward
# and compute the gradient of logit w.r.t. each.

# Here's the clean PyTorch way:
print("  Running gradient computation...")

for b_start in tqdm(range(0, args.n_prompts, args.batch_size), desc="Gradient batches"):
    b_end = min(b_start + args.batch_size, args.n_prompts)
    batch_pairs = j_pairs[b_start:b_end]

    prompts = [make_prompt(a, b) for a, b in batch_pairs]
    answers = [(a + b) % MODULUS for a, b in batch_pairs]

    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

    # Run forward, capturing hidden states that require grad
    # We need to make the hidden states leaf nodes in the graph.
    # Strategy: run forward with no_grad, then for each layer,
    # create a clone that requires grad and manually propagate through
    # later layers.

    # Actually, the simplest approach that works:
    # Run the entire forward pass with all layers requiring grad,
    # then compute gradient of logit w.r.t. each intermediate.

    # But the transformer forward uses .detach()? Let's test.
    # The standard output_hidden_states returns tensors that ARE
    # part of the computation graph if the forward was in grad mode.

    # Let's just run the model forward and use autograd:
    model.zero_grad()
    outputs = model(**inputs, output_hidden_states=True)
    # hidden_states is a tuple of tensors, each [batch, seq, d_model]

    for b_idx in range(len(batch_pairs)):
        ans = answers[b_idx]
        ans_tok = answer_token_ids.get(ans)
        if ans_tok is None or isinstance(ans_tok, list):
            continue

        last_pos = last_positions[b_idx].item()

        h_final = outputs.hidden_states[-1][b_idx, last_pos]  # [d_model]
        logit_correct = (model.lm_head.weight[ans_tok] @ h_final).squeeze()

        for l in LAYERS:
            h_l = outputs.hidden_states[l][b_idx, last_pos]  # [d_model]
            # This h_l should be part of the computation graph
            grad = torch.autograd.grad(logit_correct, h_l, retain_graph=True)[0]
            if grad is not None:
                # grad is ∂logit_correct/∂h_l, which is the J-lens vector for ans_tok
                all_jac_lens[l][ans_tok] += grad.detach().cpu().float().numpy()
                all_counts[l] += 1

        # Clear graph to free memory (PyTorch accumulates by default)
        model.zero_grad()

# Average
for l in LAYERS:
    if all_counts[l] > 0:
        all_jac_lens[l] /= all_counts[l]

print("  J-lens vectors computed.")
for l in LAYERS:
    nnz = np.count_nonzero(np.linalg.norm(all_jac_lens[l], axis=1))
    print(f"    Layer {l}: {all_counts[l]} samples, {nnz} non-zero vectors")

# Save J-lens vectors
jlens_path = OUTPUT_DIR / "jlens_vectors.npz"
np.savez_compressed(jlens_path,
    **{f"l{l}": all_jac_lens[l] for l in LAYERS},
    layers=np.array(LAYERS),
    vocab_size=vocab_size,
    d_model=D_MODEL,
)
print(f"  Saved to {jlens_path}")

# ─────────────────────────────────────────────────────────────────────
# STEP 2: Load crosscoder, compute g-space prediction per layer
# ─────────────────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 2: G-space prediction of answer per layer")
print("=" * 60)

CKPT_PATH = Path(args.crosscoder) / "checkpoints" / "crosscoder_step0024413.safetensors"
sd = load_file(str(CKPT_PATH))
W_enc = sd["W_enc"].numpy()  # [L*D, N_feat]
b_enc = sd["b_enc"].numpy()  # [N_feat]
W_dec = sd["W_dec"].numpy()  # [N_feat, L*D]
N_FEAT = W_dec.shape[0]
print(f"  {N_FEAT} features")

# Compute crosscoder features for each layer
# For each layer l, decoder slice: W_dec[:, l_idx * D_MODEL : (l_idx+1)*D_MODEL]
gspace_r2 = {}

for l_idx, l in enumerate(LAYERS):
    if l not in train_acts:
        continue

    # Get activations at this layer
    acts = train_acts[l]  # [N, D_MODEL]
    n_samples = len(acts)

    # Crosscoder: encode → features
    # Acts are in float16/bfloat16 from npz, convert to float32
    acts_f32 = acts.astype(np.float32)

    # Encoder: pre-activation = acts @ W_enc[l_slice] + b_enc
    enc_slice = W_enc[l_idx * D_MODEL : (l_idx + 1) * D_MODEL, :]  # [D, N_feat]
    pre_acts = acts_f32 @ enc_slice + b_enc  # [N, N_feat]

    # TopK activation (same as training): keep top K=32
    K = 32
    topk_vals, topk_idx = np.topk(pre_acts, K, axis=1)
    features = np.zeros_like(pre_acts)
    for i in range(n_samples):
        features[i, topk_idx[i]] = topk_vals[i]

    # Decoder: reconstruction = features @ W_dec[...]
    dec_slice = W_dec[:, l_idx * D_MODEL : (l_idx + 1) * D_MODEL]

    # Predict answer from features via linear regression
    reg = LinearRegression()
    reg.fit(features, train_labels)
    r2 = reg.score(features, train_labels)

    gspace_r2[l] = float(r2)
    print(f"  Layer {l}: R² = {r2:.4f}")

# ─────────────────────────────────────────────────────────────────────
# STEP 3: J-space answer rank per layer
# ─────────────────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 3: J-space answer rank per layer")
print("=" * 60)

jspace_ranks = {}

for l_idx, l in enumerate(LAYERS):
    if l not in train_acts:
        continue

    j_vectors = all_jac_lens[l]  # [vocab, d_model]
    acts = train_acts[l].astype(np.float32)  # [N, d_model]

    # For each sample, compute J-lens scores = acts @ j_vectors^T
    # Then rank of correct answer token
    scores = acts @ j_vectors.T  # [N, vocab]

    all_ranks = []
    for i in range(len(acts)):
        ans = train_labels[i]
        ans_tok = answer_token_ids.get(int(ans))
        if ans_tok is None or isinstance(ans_tok, list):
            continue
        # Rank: how many tokens have higher score than answer
        ans_score = scores[i, ans_tok]
        rank = (scores[i] > ans_score).sum() + 1
        all_ranks.append(rank)

    if all_ranks:
        mean_rank = np.mean(all_ranks)
        median_rank = np.median(all_ranks)
        jspace_ranks[l] = {
            "mean_rank": float(mean_rank),
            "median_rank": float(median_rank),
            "n_valid": len(all_ranks),
        }
        print(f"  Layer {l}: mean_rank={mean_rank:.1f}, median_rank={median_rank:.1f}")
    else:
        jspace_ranks[l] = {"mean_rank": float('nan'), "median_rank": float('nan'), "n_valid": 0}

# ─────────────────────────────────────────────────────────────────────
# STEP 4: Direct g-space vs J-space comparison
# ─────────────────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 4: Direct g-space vs J-space comparison")
print("=" * 60)

# At peak J-space layer (lowest rank), do gradient pursuit decomposition
# and compare crosscoder decoder directions to J-lens vectors

# Find peak J-space layer
valid_layers = [l for l in LAYERS if l in jspace_ranks and not np.isnan(jspace_ranks[l]["mean_rank"])]
peak_j_layer = min(valid_layers, key=lambda l: jspace_ranks[l]["mean_rank"])
print(f"  Peak J-space layer: {peak_j_layer} (mean_rank={jspace_ranks[peak_j_layer]['mean_rank']:.1f})")

# Find peak g-space layer
gspace_peak = max(gspace_r2, key=lambda l: gspace_r2[l])
print(f"  Peak g-space layer: {gspace_peak} (R²={gspace_r2[gspace_peak]:.4f})")

# ─── Gradient pursuit decomposition ───
# Given activation h and J-lens vectors V = [j_1, ..., j_vocab]^T (each row is a lens vector),
# find sparse nonnegative combination minimizing ||h - V^T α||² with ||α||₀ ≤ k

def gradient_pursuit(h, V, k, max_iter=100):
    """Sparse nonnegative decomposition of h into J-lens vectors.

    Args:
        h: activation vector [d_model]
        V: J-lens vectors [n_vocab, d_model]
        k: sparsity target
        max_iter: maximum iterations

    Returns:
        alpha: sparse coefficients [n_vocab]
        residual: h - V^T @ alpha
    """
    n_vocab, d_model = V.shape
    alpha = np.zeros(n_vocab)
    residual = h.copy()
    support = set()

    for _ in range(k * max_iter // k):  # max total iterations
        if len(support) >= k:
            break

        # Compute correlation of each J-lens vector with residual
        corrs = V @ residual  # [n_vocab]
        # Exclude already-selected
        for s in support:
            corrs[s] = -np.inf
        # Only positive correlations (nonnegative constraint)
        best = np.argmax(corrs)
        if corrs[best] <= 0:
            break

        support.add(best)

        # Re-optimize all active coefficients (nonnegative least squares)
        active = list(support)
        V_active = V[active]  # [|S|, d_model]
        # Solve nonnegative least squares: min ||V_active^T α - h||², α ≥ 0
        from scipy.optimize import nnls
        alpha_active, _ = nnls(V_active.T, h)
        alpha[active] = alpha_active
        residual = h - V_active.T @ alpha_active

    return alpha, residual

# Only do gradient pursuit if scipy is available
try:
    from scipy.optimize import nnls

    print("  Running gradient pursuit at peak J-space layer...")
    j_vectors_peak = all_jac_lens[peak_j_layer]  # [vocab, d_model]
    acts_peak = train_acts[peak_j_layer][:100].astype(np.float32)  # first 100 for speed

    # For each of the top 20 crosscoder features at this layer, compute alignment
    # with J-lens vectors
    l_idx = LAYERS.index(peak_j_layer)
    dec_slice_peak = W_dec[:, l_idx * D_MODEL : (l_idx + 1) * D_MODEL]  # [N_feat, D_MODEL]

    # Compute alignment: for each J-lens vector, cosine sim with each crosscoder decoder direction
    # Then take max over CC features for each J-lens token
    j_norms = np.linalg.norm(j_vectors_peak, axis=1, keepdims=True)
    j_normalized = j_vectors_peak / (j_norms + 1e-8)

    dec_norms = np.linalg.norm(dec_slice_peak, axis=1, keepdims=True)
    dec_normalized = dec_slice_peak / (dec_norms + 1e-8)

    # Alignment matrix: [n_vocab, N_feat]
    alignment = j_normalized @ dec_normalized.T

    # Top aligned pairs
    top_alignments = []
    for feat_idx in range(N_FEAT):
        max_j_idx = np.argmax(np.abs(alignment[:, feat_idx]))
        max_val = alignment[max_j_idx, feat_idx]
        top_alignments.append((feat_idx, max_j_idx, float(max_val)))

    top_alignments.sort(key=lambda x: abs(x[2]), reverse=True)
    print(f"  Top 10 alignments (CC feat ↔ J-lens token):")
    for feat, jtok, val in top_alignments[:10]:
        tok_str = tokenizer.decode([jtok])
        print(f"    Feat {feat} ↔ token '{tok_str}' (id={jtok}): cos={val:.4f}")

    # Also compute per-token alignment: for each vocab token, max over CC features
    per_token_align = np.max(np.abs(alignment), axis=1)
    mean_align = float(np.mean(per_token_align[per_token_align >