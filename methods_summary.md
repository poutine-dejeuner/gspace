# Methods Summary: "When Models Manipulate Manifolds"

A practical reference from [Gurnee et al. 2025](https://transformer-circuits.pub/2025/linebreaks/index.html) and the [open-weight reproduction](https://github.com/GuanchunLi/manifold-counting-task).

---

## Quick Summary

The paper reverse-engineers how Claude 3.5 Haiku counts characters to decide line breaks. The **core method pipeline** (4 steps):

1. **Collect mean activations** per scalar value (e.g., character count 1–150) at each layer.
2. **PCA** on those mean vectors → reveals a low-dimensional (6-D) **curved manifold** (helix/rippled circle). The scalar is encoded as **position along the curve**.
3. **Probes** (logistic regression) confirm the scalar is linearly decodable. **Multiply probes through W_Q/W_K** to find heads that geometrically "twist" manifolds (e.g., rotating count to align with line width at an offset = boundary detection).
4. **Causal validation**: (a) ablate the top-k PCA subspace → loss spikes only on relevant tokens; (b) patch activations `a − μ_old + μ_target` → model's behavior shifts as if the scalar changed.

**Key insight**: Scalar quantities are represented as **1D feature manifolds embedded with high curvature in low-D subspaces**. Computation = linear transformations (rotations) of these manifolds via attention heads. Multiple heads with different offsets "tile" the space for fine resolution.

---

## Adaptation to Modular Arithmetic (mod 59)

### Direct mapping of each method:

| Paper | Our Task |
|-------|----------|
| Character count manifold | **Residue `x mod 59`** manifold (expect a **circle** S¹, not open interval) |
| Line width manifold | **Modulus 59** (likely fixed in weights, not inferred) |
| QK twist: rotate count to align with width at offset | **Addition heads**: rotate operand A manifold to align with operand B at offset = (a+b) mod 59 |
| Characters remaining = k − count | **Modular sum/difference** `(a+b) mod 59` or `(a−b) mod 59` |
| Distributed counting (L0→L1 heads) | **Distributed modular addition** across heads/layers |
| Newline decision (orthogonal subspaces, linear separation) | **Answer readout**: operands arranged orthogonally so correct sum is linearly separable |
| Causal ablation of count subspace | Ablate mod-59 subspace → loss on arithmetic tokens |
| Activation patching `a − μ_c + μ_target` | Patch operand → predicted result shifts to match patched value |
| SAE/crosscoder feature families | SAE features that tile residues 0–58 (place-cell-like tuning curves) |

### Key differences from linebreaking:
- **Cyclic topology**: mod 59 wraps around → expect a **circle** (S¹), not an open curve. The paper's "rippled circle" toy model (§2.5) is directly applicable.
- **Binary operation** vs unary accumulation: need to track **two** operand manifolds and how an attention head rotates one against the other.
- **Fixed modulus** (59) vs inferred line width: modulus is likely in weights, simplifying width detection.

### Concrete recipe:
1. Create synthetic dataset: `"Compute a + b mod 59: a=23, b=41 →"` with ground-truth labels for a, b, (a+b) mod 59 at each token position.
2. Cache residual stream activations. For each residue r ∈ [0,58], compute the **mean activation vector** at each layer.
3. PCA on the 59 mean vectors. Look for a **circular/helical structure** in top PCs. Compute R²(residue | top-k PCs).
4. Train 59-way logistic probes at each layer. Find where accuracy peaks (= where the manifold is sharpest).
5. For addition: find the layer where **both operands** are represented, then search for attention heads whose QK circuit aligns operand A at offset = target sum against operand B.
6. Causal: ablate the residue subspace → loss on "=" token; patch a's representation → P(answer) follows patched value.
7. Repeat for different group operations (addition, multiplication, permutation composition).

---

## 1. Task Setup

**Linebreaking task**: Given fixed-width text (implicit constraint `k`), predict whether the next token fits on the line or a newline is needed.

### Synthetic Dataset Creation
1. Take a diverse text corpus, strip all newlines.
2. Reinsert newlines every `k` characters (to nearest word boundary ≤ `k`), for `k ∈ {15, 20, …, 150}`.
3. Use **teacher-forcing** — feed correctly-wrapped strings and analyze per-position predictions (avoids compounding errors from free generation).
4. Compute per-token labels:
   - **Character count**: total characters since last newline (on detokenized text, not token indices).
   - **Line width `k`**: characters between adjacent newlines.
   - **Characters remaining**: `k − count`.
   - **Next token length**: character length of the true next word.
5. The wrapping constraint is **implicit** — the model infers `k` from context, not from explicit instructions.

### Behavioral Gate
Before any mechanistic analysis, verify the model can actually do the task:
- Within-line newline AUC (teacher-forced).
- P(newline) at true breaks vs mid-line.
- P(newline) as a function of characters remaining (should spike near 0).

---

## 2. Unsupervised Feature Discovery

**Tool**: Weakly Causal Crosscoder (WCC) with ~10M features trained on Claude 3.5 Haiku.  
**Open-model alternative**: Gemma Scope 2 SAEs.

### Method
1. Run the model on prompts; collect feature activations.
2. **Attribution graph** [Ameisen et al. 2025]: Trace which features excite/inhibit each other to produce the output logit. This surfaces the computational graph for a specific prediction.
3. Identify features whose activation varies smoothly with character count:
   - Bin activations by line character count.
   - Look for features with smooth profiles and large between-count variance.
4. **Feature families**: Groups of features with offset, overlapping tuning curves — like biological place cells. Characteristic property: **dilation** (later features activate over wider ranges).

### Key Finding
~10 features form a "place cell" family that tiles the character count range. At any count, 2–3 features are co-active, locally parametrizing the manifold.

---

## 3. Manifold Discovery (Geometric View)

> This is the core method for our purposes. It requires **no** feature dictionary — just cached activations + PCA.

### Step-by-Step
1. **Collect mean activations**: For each possible character count `c ∈ [1, 150]`, average the residual stream vectors at a target layer across all tokens with that count.
2. **PCA**: Compute PCA of these 150 mean vectors. The top 6 PCs capture ~95% of the variance (Haiku) / ~93% (Gemma-3-4B).
3. **Visualize**: Project the mean vectors into the top 3 PCs. The points trace a **smooth, count-ordered, rippled curve** — a helix in PCs 1–3 and a more complex twist in PCs 4–6.
4. **Validate dimensionality**: `R²(count | top-k PCs)` should saturate at the manifold dimension (e.g., 6).
5. **Feature-manifold duality**: The SAE/crosscoder feature decoder vectors lie along this same curve, discretizing it. Interpolating between 2–3 active features reconstructs the continuous manifold.

### Important: Z-scoring Before PCA (Gemma Reproduction)
Raw PCA can be dominated by a few massive-activation "scale" dimensions. **Z-score residual dimensions** before PCA to recover the count axis (raw PC1↔count = +0.57 → z-scored PC1↔count = +0.94).

### The Ringing Pattern
- Cosine similarity matrix of per-count mean vectors: diagonal band (neighbors similar), off-diagonal negative bands, then positive again further out.
- This "ringing" is a **necessary consequence** of embedding a high-curvature 1D manifold into low dimensions — it's optimal for trading off capacity (dimensionality) vs distinguishability (curvature).
- **Physical model**: Points on a hypersphere with attractive forces to k nearest neighbors and repulsive forces to all others produce the same ringing + rippled helix.

---

## 4. Probe-Based Analysis

### Logistic Probes for Scalar Quantities
Train n-way multinomial logistic regression probes to predict character count / line width / characters remaining from the residual stream at a given layer.

**Useful analyses**:
- **Probe quality**: RMSE, within-±N accuracy.
- **PCA of probe weight vectors**: Their top components span the same subspace as the mean-activation PCA. 6 components capture ~82% of probe weight variance.
- **Probe response matrix**: For each probe `i` and each true count `j`, compute the probe's response. Shows diagonal band (correct classification) + off-diagonal ringing stripes — the same ringing as in cosine similarity.
- **Layer sweep**: Compute probe accuracy at every layer to find where the count representation is sharpest. The manifold is present-but-diffuse early, sharp mid-network, then consumed late.

### Cosine Similarity Matrix of Probe Directions
The cosine similarity between probe weight vectors for different counts forms a smooth, count-ordered band — nearby counts aligned, distant counts opposed. This is the readout-space echo of the manifold geometry.

---

## 5. Boundary Detection: The QK "Twist"

> Key insight: attention heads perform geometric transformations on manifolds via their `W_Q` and `W_K` matrices.

### Method
1. Train probes `p_i` for character count `i` and `q_k` for line width `k`.
2. Multiply probes through the QK weights of a candidate attention head:
   - Query space: `p_i^T W_Q`
   - Key space: `q_k^T W_K`
3. Compute cosine similarity matrix between all `(p_i^T W_Q)` and `(q_k^T W_K)`.
4. Compare to the same matrix in the residual stream (identity transform) and through a random head's QK.

### Expected Pattern (Boundary Head)
- **Residual stream**: max similarity on diagonal `i = k`, but max cosine only ~0.25 (representations are weakly aligned).
- **Boundary head QK space**: max similarity on **off-diagonal** `i < k`, with near-perfect alignment (cosine ≈ 1.0). The head "twists" the count manifold to align with line width at a specific offset.
- **Random head QK space**: no structure.

### Multiple Boundary Heads
Different boundary heads have different offsets. Together they tile the "characters remaining" range:
- Each head's output varies most in a specific range.
- Their **sum** produces an evenly-spaced, high-resolution representation across all values.
- Individual head outputs are ~1D; the sum is a 2D curve in PCA space.

### Why Multidimensional?
- 1D encoding: linear operations reduce to scaling/translation → cannot create a threshold.
- 2D+: can **rotate** the manifold via linear transformations → dot product creates natural threshold when counts align.
- >2D: additional dimensions pack more curvature → finer resolution.

---

## 6. Newline Decision Geometry

### Method
1. For all combinations of characters remaining `i` and next-token length `j`, average the residual stream at the decision layer (~90% depth).
2. PCA on the union of these mean vectors.
3. Visualize: characters remaining on one axis, next-token length on an orthogonal axis.
4. The pairwise sum `v_i + w_j` for `i − j ≥ 0` (break) vs `i − j < 0` (no break) should be **linearly separable**.
5. Validate: train a separating hyperplane on the PCA embeddings → should achieve high AUC on ground-truth newline prediction.

---

## 7. Distributed Character Counting Algorithm

### How the Count is Built (Layers 0–1)
1. **Embedding analysis**: Average `W_E` vectors by token character length → PCA → circular pattern with oscillating component. Token length is linearly decodable from static embeddings (R² ~0.96).
2. **Per-head decomposition**:
   - Project each head's output into the PCA basis of character count probes.
   - QK circuit: each head uses the previous newline as an "attention sink" for `s_h` tokens, then smears over receptive field up to `r_h` tokens.
   - OV circuit: `s_h × μ_c` (sink size × avg token length) + correction for above/below-average tokens.
3. **Layer 0 heads**: Each output is ~1D (a ray). Their **sum** produces the curved manifold.
4. **Layer 1 heads**: Each outputs a curve. They sharpen Layer 0's estimate.
5. R² progression: 5 Layer 0 heads → R² = 0.93; + 6 Layer 1 heads → R² = 0.97.

### Line Width Computation
Similar distributed counting, but on newline tokens (counting characters between newlines). Uses partially disjoint head sets.

---

## 8. Causal Validation

> Every geometric claim is validated with two experiments.

### Experiment 1: Subspace Ablation (Necessity)
1. Identify the k-dimensional subspace (top-k PCs of per-count means, or probe weights).
2. **Zero-ablate** that subspace from the residual stream at the target layer.
3. Measure loss increase, **broken down by newline vs non-newline tokens**.
4. **Control**: ablate a random k-dimensional subspace → should show no effect.
5. **Dimensionality check**: sweep k; effect should **saturate** at the manifold dimension.

### Experiment 2: Activation Patching (Sufficiency)
1. For each count `c`, compute the mean activation vector `μ_c` across all tokens with that count.
2. Patch: `a_patched = a_original − μ_original + μ_target`
   - This subtracts the "perceived" count and adds a target count.
3. Apply across the count-bearing band (multiple adjacent layers, last few tokens).
4. Measure P(newline) as a function of the **patched** (not real) count.
5. **Rank-k sufficiency**: patch only within the k-dimensional PCA subspace → should match full-vector patch (corr ≈ 1.0).

### Experiment 3: Characters-Remaining Variant
Same procedure but on `rem = k − count`. Steer perceived remaining distance → P(newline) should go from high (line full) to low (plenty of room).

---

## 9. Visual Illusions

A validation technique: use mechanistic understanding to construct inputs that hijack specific components.

### Method
1. Identify what **other** roles the important attention heads play (on wider data distributions).
2. Find sequences that cause the head to attend differently (e.g., `@@` in git diffs makes heads attend from newline to `@@` instead of previous newline).
3. Insert these "illusion" tokens into the prompt **without changing the line length**.
4. Verify: (a) the head's attention pattern is disrupted, (b) newline prediction is modulated.
5. Test specificity: sweep many 2-character inserts; most should have moderate impact; a few delimiter-like sequences (`@@`, `}}`, `||`, `>>`) should strongly disrupt.

---

## 10. Open-Model Reproduction Pipeline (Gemma-3-4B)

The Gemma reproduction uses a deliberately simpler backbone — no proprietary transcoder needed.

### Design Rules
| Rule | Why |
|------|-----|
| Wrap width `k` is **implicit** (inferred from context) | Explicit instructions change the task |
| Count = characters since newline on **detokenized** text | Variable-length tokens are the point |
| Teacher-force over correctly-wrapped strings | Free gen compounds errors |
| Backbone = **probes + PCA**; SAEs are optional confirmation | Anthropic's 10M-feature transcoder is unavailable |
| Rediscover layers/heads empirically | Haiku's layer indices don't transfer |
| Control for count↔width confound (fixed-width checks) | Per-count means secretly mix widths |
| **Z-score** residual dims before PCA | Massive-activation scale dims dominate raw PCA |

### Phase Structure
1. **Phase 0**: Environment, model loading, correctness checks (logits-match, tokenizer/offset)
2. **Phase 1**: Behavioral gate (can the model do the task?)
3. **Phase 2**: Count manifold (PCA, probes, causal ablation + steering, SAE place cells)
4. **Phase 3**: Line-width manifold, newline readout, characters-remaining causal

### Pre-committed Gates
Each phase has pass/fail thresholds declared **before** running experiments. This prevents p-hacking and post-hoc storytelling.

---

## 11. Applicability to Modular Arithmetic (mod 59)

Mapping each method to our task:

| Paper Method | Our Analog |
|-------------|-----------|
| Character count manifold | **Residue class manifold** (result of `x mod 59`) |
| Line width manifold | **Modulus** (59) — possibly fixed, not inferred |
| Boundary heads (QK twist) | **Addition heads**: heads that rotate one operand manifold to align with another at offset = sum |
| Characters remaining `k − count` | **Modular difference** (result of operation) |
| Newline decision (orthogonal subspaces) | **Output readout** in orthogonal subspace |
| Distributed counting (L0→L1) | **Distributed modular addition** across heads/layers |
| Causal ablation | Ablate mod-59 subspace → loss on arithmetic tokens |
| Activation patching | Patch operand → change predicted result |
| Visual illusions | Construct inputs that hijack the mod-59 counter |
| Probes + PCA backbone | Probes for residues 0–58 + PCA of per-residue means |
| Z-score before PCA | Same — critical for open-weight models |

### Key Differences to Account For
- **Modular arithmetic is cyclic** (mod 59 wraps around), while character count is linear [1, 150]. Expect the manifold to be a **circle** (S¹) rather than an open interval — the "rippled circle" toy model from §2.5 is directly applicable.
- The operation is binary (two operands) rather than unary accumulation. Need to track **two** operand manifolds and how they combine.
- The modulus (59) is likely stored in weights rather than inferred from context.

---

## 12. References

- Gurnee, W., Ameisen, E., Kauvar, I., Tarng, J., Pearce, A., Olah, C., & Batson, J. (2025). *When Models Manipulate Manifolds: The Geometry of a Counting Task.* Transformer Circuits Thread. arXiv:2601.04480.
- Li, G. (2026). *Reproducing When Models Manipulate Manifolds on an open-weight model.* github.com/GuanchunLi/manifold-counting-task.
- Ameisen, E. et al. (2025). *Circuit Tracing: Revealing Computational Graphs in Language Models.* Transformer Circuits Thread.
- Modell, et al. *Feature Manifolds.* (on cosine similarity encoding intrinsic geometry of features).
