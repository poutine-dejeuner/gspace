# References

## Primary reference

### When Models Manipulate Manifolds: The Geometry of a Counting Task

- **Authors:** Wes Gurnee*, Emmanuel Ameisen*, Isaac Kauvar, Julius Tarng, Adam Pearce, Chris Olah, Joshua Batson*‡
- **Affiliation:** Anthropic
- **Published:** October 21st, 2025
- **URL:** https://transformer-circuits.pub/2025/linebreaks/index.html
- **Local copy:** `refs/linebreaks.md` (cleaned markdown), `refs/linebreaks.html` (raw HTML)

**Summary:** The authors investigate how Claude 3.5 Haiku performs linebreaking in fixed-width text — a natural perceptual task common in pretraining corpora. The model must count characters in the current line, infer the line width constraint, and predict whether the next word fits or requires a newline.

**Key findings:**
1. **Character count** is represented on a 1D feature manifold (a rippled helix/curve) embedded in a low-dimensional (~6D) subspace of the residual stream, with 10 sparse crosscoder features discretizing this manifold.
2. **Boundary detection** is done by attention heads whose QK matrices "twist" the character count manifold to align it with the line width manifold at a specific offset, allowing detection of approaching line boundaries.
3. **The newline decision** combines characters-remaining with next-token-length in near-orthogonal subspaces, making the break/no-break decision linearly separable.
4. **The manifold is constructed distributively** by many attention heads across layers 0-1, each contributing a piece of curvature.
5. **Visual illusions** can be constructed by inserting distractor tokens (e.g., `@@`) that hijack counting attention heads.
6. **Rippled representations are optimal** — the ringing/rippling is a natural consequence of packing a high-curvature 1D manifold into low dimensions, with connections to Fourier features and space-filling curves.

**Model:** Claude 3.5 Haiku, using a 10M-feature Weakly Causal Crosscoder (WCC) dictionary.
