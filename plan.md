# Implementation Plan: Geometric Subspaces for Modular Arithmetic

## Hypothesis

Over a subset of layers, models represent modular arithmetic operands as **circles (S¹)** embedded in a low-dimensional **g-space (G)**, and rotate those circles through the layers to compute the result. The answer becomes visible in **j-space** (Jacobian-lens workspace) after g-space operations complete.

---

## Phase 0: Model Selection via Benchmark

**Goal**: Find the smallest open-weight model that achieves perfect accuracy on modular arithmetic.

### 0.1 Benchmark Design

Generate a test set of modular arithmetic problems:

```
"Compute a + b mod 59: a=23, b=41 →"
"Compute a − b mod 59: a=12, b=48 →"
"Compute a × b mod 59: a=7, b=13 →"
```

**Coverage**:
- Operations: `+`, `−`, `×` (start with `+`; expand if time)
- All `a, b ∈ [0, 58]` → 59² = 3,481 pairs per operation
- Hold out a validation set (e.g., 20% of pairs) for probing; use the rest for analysis

**Expected output**: The correct residue `(a op b) mod 59`. Tokenization matters — ensure residue 0–58 is a single token or predictable token sequence.

### 0.2 Candidate Models (small, open-weight)

Start smallest and work up:

| Model | Params | Notes |
|-------|--------|-------|
| Qwen2.5-0.5B | 0.5B | Smallest; likely fails |
| SmolLM2-1.7B | 1.7B | |
| Gemma-3-1B | 1B | |
| Qwen2.5-1.5B | 1.5B | |
| Gemma-3-4B | 4B | Reproduction paper used this |
| Llama-3.2-3B | 3B | |
| Qwen2.5-7B | 7B | Fallback if smaller ones fail |

### 0.3 Selection Criterion

**Perfect accuracy** on the held-out test set for `a + b mod 59`. If no small model achieves perfect accuracy, relax to:
- >95% for `+`, then investigate whether errors cluster (e.g., near modulus boundary, specific residues).
- If errors are systematic, the model may still have the geometric structure but fail at edge cases.

### 0.4 Output of Phase 0

- Selected model name and size
- Benchmark results for all tested models
- Confirmation that residues 0–58 are tokenized as single tokens (or a plan to handle multi-token)

---

## Phase 1: Behavioral & Representation Confirmation

**Goal**: Verify the model actually does the task and that residue is linearly decodable.

### 1.1 Run the model

- Teacher-force with correct answers (for analysis) or let it generate (for behavioral metrics).
- Cache residual stream activations at **all layers** for all token positions.
- Use `transformer_lens` or direct `transformers` + hooks.

### 1.2 Behavioral metrics

- Per-position accuracy: at which token position does the answer become decodable?
- Confusion matrix: which residues are confused with which?
- Does accuracy improve across layers? (Measure via probe at each layer.)

### 1.3 Linear probe for residue

Train a 59-way logistic regression probe at each layer:

```
probe(residual_stream[layer][position]) → residue ∈ [0, 58]
```

- Find the layer where accuracy peaks → this is where the residue manifold is sharpest.
- Compute PCA of the 59 probe weight vectors → how many components capture >90% variance?
- Cosine similarity matrix of probe weights: expect a **circulant banded** pattern (nearby residues similar, distant residues dissimilar, wrapping around at 58↔0).

### 1.4 Output of Phase 1

- Layer-wise probe accuracy curve
- Probe weight PCA dimensionality
- Cosine similarity heatmap (should show circular structure if S¹ manifold exists)

---

## Phase 2: Manifold Discovery (G-Space)

**Goal**: Find the low-dimensional curved manifold representing residues.

### 2.1 Per-residue mean activations

At the best probe layer (from Phase 1), for each residue `r ∈ [0, 58]`:
- Average the residual stream across all tokens where the **true** operand/result equals `r`.
- This gives 59 mean vectors `μ₀, μ₁, ..., μ₅₈`.

**Important controls** (from Gemma reproduction):
- Z-score residual dimensions before PCA.
- Control for fixed-modulus confound: restrict to one operand value at a time if needed.

### 2.2 PCA and visualization

- PCA on the 59 mean vectors.
- Plot in top 2–3 PCs. Color by residue.
- **Expected**: A **circle** (S¹) or rippled circle — residues arranged in cyclic order with possible helical lift into 3D.
- Compute: `R²(residue | top-k PCs)` for k = 1..10.
- Find the **extrinsic dimension** where R² saturates (paper: 6D; we expect 2–6D).

### 2.3 Ringing / cosine similarity

- Pairwise cosine similarity of μᵢ vs μⱼ.
- **Expected**: banded diagonal (neighbors similar), negative band further out, positive again even further (ringing), **with cyclic wrap-around** at 0↔58.
- This is the signature of a rippled circle embedding.

### 2.4 Feature-manifold duality (optional, if SAEs available)

If SAEs exist for the chosen model (Gemma Scope, etc.):
- Find SAE features that activate for specific residues.
- Plot feature decoder vectors alongside the mean-activation manifold.
- Expected: feature decoders lie along the manifold, discretizing the circle (place-cell-like tiling).

### 2.5 Output of Phase 2

- PCA plots (PC1–PC2, PC1–PC3 colored by residue)
- R² vs k plot
- Cosine similarity heatmap with cyclic boundary
- Extrinsic dimension estimate

---

## Phase 3: Operand Representation & G-Space Operations

**Goal**: Find where operands are represented and how they combine.

### 3.1 Dual-operand probes

At the computation layer(s), train **separate** probes for:
- Operand `a` (from residual at position of `a` or at the `=` token)
- Operand `b`
- Result `r = (a+b) mod 59`

Find the layers where:
- `a` is decodable (probe accuracy peaks)
- `b` is decodable
- `r` is decodable

### 3.2 Joint geometry of operands

For the computation token (e.g., `=` or operator token):
- PCA on the union of per-residue means for `a`, `b`, and `r`.
- Are `a` and `b` arranged in **orthogonal subspaces**? (This would make linear separation of results easy, like the newline decision.)
- Does `r` lie at a specific geometric relationship to `a` and `b`? (e.g., vector sum, rotation)

### 3.3 Search for "addition heads" (QK Twist)

This is the crux. For each attention head at the computation layers:

1. Train probes `p_a` (59 vectors, one per a-value) and `q_b` (59 vectors, one per b-value).
2. Multiply through `W_Q` and `W_K`: `p_a^T W_Q`, `q_b^T W_K`.
3. Compute cosine similarity matrix between all (a, b) pairs in QK space.
4. **Expected for an addition head**: max cosine similarity when `b = (target − a) mod 59` (i.e., the head attends to token `b` when it would combine with `a` to produce the correct result). Alternatively: max when `(a + b) mod 59 = constant` — the head detects a specific sum.
5. Multiple heads may tile the residue space (different heads detect different sum ranges).
6. **Control**: random heads show no structure.

### 3.4 Distributed computation across layers

- Project each head's output into the PCA basis of residue probes.
- Layer L heads: individual outputs are ~1D curves; their **sum** produces the circular manifold.
- Layer L+1 heads: each outputs a curve; they sharpen the representation.
- Compute R² progression: how many heads needed for good residue prediction?

### 3.5 Output of Phase 3

- Joint PCA of a, b, r representations
- QK cosine similarity matrices for candidate addition heads (with controls)
- Head tiling visualization
- R² by head count

---

## Phase 4: Causal Validation

**Goal**: Prove the g-space causally matters.

### 4.1 Subspace Ablation (Necessity)

- Identify the k-dimensional residue subspace from Phase 2 PCA.
- Zero-ablate it at the computation layer(s).
- Measure loss change **only on the answer token**.
- **Control**: ablate random k-D subspace → no effect.
- Sweep k → effect should saturate at the manifold dimension.

### 4.2 Activation Patching (Sufficiency)

- For residue `r`, use mean activation `μ_r` from Phase 2.
- Patch: `a_patched = a_original − μ_a_original + μ_a_target`
- Apply at the computation layer(s) on the operand position(s).
- **Expected**: P(correct answer) shifts as if the operand actually changed.
- **Rank-k check**: patch only within the k-PCA subspace → should match full-vector patch.

### 4.3 Steering to specific residues

- Sweep target residue across 0–58.
- Plot P(predicted answer = (a_target + b) mod 59) vs a_target.
- Monotonic relationship = strong causal evidence.

### 4.4 Output of Phase 4

- Ablation loss by token position (answer token vs others)
- Patching P(answer) vs target residue curve
- Rank-k vs full-vector patch correlation

---

## Phase 5: J-Space Comparison

**Goal**: Compare g-space with Jacobian-lens workspace.

### 5.1 Compute J-space

For the selected model:
- Compute input-output Jacobian at the computation layers.
- Find the dominant singular vectors → this is j-space.
- [Detail: use the Jacobian lens method — SVD of ∂(output)/∂(residual at layer L)]

### 5.2 Compare G-space and J-space

- Cosine similarity between g-space basis (from Phase 2 PCA) and j-space basis.
- Layer alignment: does j-space align with g-space early, or only after g-space operations complete?
- **Hypothesis**: g-space is active in layers l < l₀; j-space captures the result in layers l ≥ l₀.

### 5.3 Cross-task comparison

(Stretch goal) Repeat for different moduli (mod 7, mod 13, mod 97) or different operations (×). Does g-space dimension depend on modulus? Does j-space?

### 5.4 Output of Phase 5

- J-space singular value spectrum
- G–J subspace cosine similarity by layer
- Confirmation/rejection of the l₀ hypothesis

---

## Phase 6: Generalization (if time)

- **Multiplication mod 59**: Does the same manifold serve both + and ×? Or different subspaces?
- **Other groups**: Permutation composition? Small dihedral groups?
- **Different moduli**: Does the extrinsic dimension scale with modulus?

---

## Implementation Notes

### Tooling
- `transformer_lens` for hooking and activation caching
- `nnsight` as alternative for intervention-heavy workflows
- `scikit-learn` for probes and PCA
- SAEs: use model-specific dictionaries (Gemma Scope, etc.) if available

### Compute
- Target: single GPU (RTX 4090 24GB or similar).
- Cache all layer activations → ~(n_layers × d_model × n_tokens × 2 bytes) VRAM.
- For 4B model (d_model ~ 2560, ~36 layers, ~100K tokens): ~18 GB — tight but feasible.

### Fast Iteration
- Start with a small subset of the benchmark (100–200 examples) for development.
- Scale to full dataset only for final results.
- Cache activations to disk to avoid recomputation.

### File Structure
```
gspace/
├── plan.md                          # This file
├── methods_summary.md               # Paper methods reference
├── AGENTS.md                        # Project spec
├── benchmark/
│   └── mod59_benchmark.py           # Generate + evaluate benchmark
├── src/
│   ├── model_selection.py           # Phase 0
│   ├── behavioral.py                # Phase 1
│   ├── manifold.py                  # Phase 2 (per-residue means, PCA)
│   ├── probes.py                    # Phase 1+3 (linear/logistic probes)
│   ├── addition_heads.py            # Phase 3 (QK twist search)
│   ├── causal.py                    # Phase 4 (ablation + patching)
│   └── jspace.py                    # Phase 5 (Jacobian lens)
├── notebooks/
│   └── phase*_summary.ipynb         # One notebook per phase
└── results/
    ├── phase0_model_selection.md
    ├── phase1_behavioral.md
    ├── phase2_manifold.md
    ├── phase3_operations.md
    ├── phase4_causal.md
    └── phase5_jspace.md
```

---

## Timeline (rough)

| Phase | What | Effort |
|-------|------|--------|
| 0 | Model selection + benchmark | 1–2 sessions |
| 1 | Behavioral + probe confirmation | 1 session |
| 2 | Manifold discovery (PCA, cosine) | 1–2 sessions |
| 3 | Addition heads (QK twist) | 2–3 sessions |
| 4 | Causal validation | 1 session |
| 5 | J-space comparison | 1–2 sessions |
| 6 | Generalization (stretch) | ? |
