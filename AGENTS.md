You are a research assistant. Question my approaches, clarify my ideas to the
maximum, suggest and do experiments to test hypotheses.

Identify geometric computations subspace and manifold of an LLM when performing
computations.

## Motivation

In [When Models Manipulate Manifolds: The Geometry of a Counting Task](https://transformer-circuits.pub/2025/linebreaks/index.html), the authors discover a geometric space being acted upon by geometric transformation in order to perorm computations required for a task.
Specifically, to perform the end of line prediction taks, the llm represents
the line-positions as an interval embedded as a helix, and this helix is
rotated to keep track of the current position.

We want to identify the geometric subspace in which a language model performs
algebraic computation (any group operation but cyclic groups are the simplest), and compare the emergence of this
subspace with the Jacobian-lens workspace (J-space). The hypothesis: algebraic
computations are carried out in a low-dimensional subspace of the residual
stream, in a subset of layers $l\<l_0$, and the answer will be visible in the j-space in subsequent layers.

## Question

Identify the computation geometric subspace and layers participating in the
   computations (probably different for different task and model). Not clear how to do this,
   see [When models manipulate manifolds] for inspiration.
Possible approaches:


## Task

- Modular arithmetic mod 59

## Hypothesis

- Over a subset of layers, the models will represent modular arithmetic data as circles in g-space G and rotate that circle through the
   layers to get to the answer.
- The answer will be visible in the j-space after the g-space operations.

## Implementation details

 - To produce results fast, a smaller model is prefered
 - The model must be large enough that it reliably gets modular arithmetic
   questions right. We first need to identify these models. We make a modular arithmetic benchmark to test
this. Starting with
smaller open weights models, we discard those that dont have a perfect score on
the benchmark. Experiments will continue with smallest model with perfect
score.

## Project conventions
- Use test-based development
- see SLURM.md to run code
- Use uv to maintain the venv. uv add, not uv pip install.
- **Smoke test before SLURM**: Every training script must support `--smoke_test N`
  (default N=10) that runs the full pipeline (train → save → resume → analyze)
  for N steps and exits cleanly. Run it locally before submitting to SLURM.
  This catches signature mismatches, missing imports, save/load bugs, etc.
  in ~30s instead of after hours on the cluster.

## Job monitoring
Whenever you launch a SLURM job or any long-running background process, stay in the loop until it finishes. Do NOT drop the conversation or move on to unrelated work. Poll periodically with squeue, tail the output/error logs, and report progress. If a job fails, diagnose the error immediately and propose a fix. When it succeeds, report the results.

