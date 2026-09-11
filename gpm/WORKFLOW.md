# Phase 4 — improving GPM

One change at a time, each with a stated hypothesis and a pre-registered way to
be wrong.

## Why not all at once

This project has already paid for that answer twice. The v1→v2 sweep changed the
task list *and* the GPU, and the resulting Δrho could not be attributed to
either. GPM's own seed noise is ±0.039 — larger than most of the improvements we
expect — so two simultaneous changes leave you unable to say which one moved
anything, or whether either did.

Config-only changes are the exception: they can share a sweep because the grid
already separates them into distinct rows.

## What is detectable, before designing anything

GPM baseline: `rho 0.073, sd 0.039, n=5` (Track A, eps=0.8).

| seeds | minimum detectable Δrho | i.e. rho must reach |
|---|---|---|
| 3 | 0.063 | 0.136 |
| 5 | 0.049 | 0.122 |
| 10 | 0.035 | 0.108 |
| 20 | 0.024 | 0.098 |

**At 5 seeds, any improvement smaller than ~0.05 is invisible.** Budget seeds to
the effect you expect, or you will run an experiment that cannot answer its own
question. This table is the reason each stage below names a seed count.

---

## Stage 0 · Instrumentation — local, free

Basis saturation is the quantity the whole phase turns on, and right now it is
computed post-hoc from `basis_entries`. Promote it to a signature probe so every
run reports it directly:

- **new probe C3** — `fraction of available gradient directions consumed`
- fails when the basis exceeds ~90% of `max_bases_frac`, because past that point
  the mechanism is measurably degrading (rho 0.073 → −0.047 as saturation went
  59.5% → 99.2%)

**Done when:** a GPM run prints its saturation, and a deliberately over-saturated
config fails the probe. No GPU needed.

---

## Stage 1 · Invert `eps_growth` — one sweep, 5 seeds

**Hypothesis.** `eps = eps_base + 0.005 * tasks_seen` raises the retention
threshold as the basis fills, so later tasks consume *more* directions exactly
when least affordable. Reversing it should delay saturation.

**Change.** Config only — sweep `eps_growth` over `[-0.01, -0.005, 0, +0.005]`.
No new code, so all four are grid points in a single sweep.

**Expected.** Modest. This slows saturation, it does not remove it. If Δrho is
under 0.05 it will not be distinguishable at 5 seeds, and that is an acceptable
answer: it tells you the growth schedule is not the binding constraint.

**Cost.** 4 configs × 5 seeds ≈ 20 runs ≈ 2h.

---

## Stage 1 · OUTCOME — refuted, and it changed the protocol

At 5 seeds, `eps_growth=-0.01` measured rho 0.124 vs the baseline's 0.058:
Delta +0.066 against a pre-registered MDE of 0.049, monotone across all four
levels. It was adopted provisionally.

At 10 seeds the effect **reversed and vanished**: Delta -0.005, t = -0.18, and
the negative arm won in exactly 5 of 10 seeds. The first result was noise —

```
eps_growth=-0.01,  seeds 0-4: [0.080 0.094 0.031 0.151 0.251]  mean 0.121
                   seeds 5-9: [0.034 0.011 0.079 0.046 0.097]  mean 0.053
eps_growth=+0.005, seeds 0-4: [0.016 0.077 0.057 0.066 0.062]  mean 0.056
                   seeds 5-9: [0.130 0.110 0.061 0.143 0.203]  mean 0.129
```

Same config, same GPU, same code; both arms swung by more than the claimed
effect. **What survives is the memory result**: 31.5MB vs 58.5MB, a 46%
reduction, deterministic rather than statistical. Adopt `eps_growth=-0.01` for
that, and make no accuracy claim.

### Four attempts to reduce the variance, all refuted

1. **More collection batches** — relative rank spread stayed 22-25% at
   `collect_batches` 4, 16 and 64. Not estimation noise.
2. **GPM is not the noisy part.** `control_joint` has CV 0.4%; `control_sequential`
   — no mechanism at all — has CV 30.7%, same as GPM's 26-35%. The chaos is in
   sequential training on a hard stream, not in the mechanism.
3. **Paired estimator** (each seed against its own floor) made it *worse*, sd
   0.070 -> 0.084. Floor-to-mechanism correlation is -0.234: there is no shared
   "unlucky seed" factor to cancel.
4. **More eval batches** — 4x the eval data moved sd by +3%. Measurement noise
   is not the bottleneck.

### What actually fixes it: develop on Task-IL, validate on Class-IL

| benchmark | rho | sd | SNR | seeds to detect a 50% gain |
|---|---|---|---|---|
| Class-IL | 0.093 | 0.054 | 1.7 | ~11 |
| Task-IL | 0.227 | 0.050 | 4.5 | ~2 |

The noise floor is the same in both — the *signal* is 2.4x larger in Task-IL.
Iterating in Class-IL means paying 5x the compute to see the same change.

**Revised protocol for every stage below:**

1. Develop and screen on **Task-IL**, 5 seeds. Cheap, and effects are visible.
2. Anything that survives gets **one confirmation run on Class-IL**, 10 seeds.
3. Only the Class-IL number is ever reported as a result.

The second step is not optional: the scenarios reorder mechanisms (EWC went
0.335 in Task-IL to 0.104 in Class-IL; LwF moved the other way), so a Task-IL
gain is a hypothesis, not a finding. But it is a hypothesis you can generate five
times more cheaply.

---

## Stage 2 · Aging basis — `gpm_aging`, 5 seeds Task-IL then 10 Class-IL

**Hypothesis.** Monotonic growth is the binding failure. A basis that evicts
directions — by age, or by how little the recent stream projects onto them —
converts unbounded consumption into a steady state.

**Change.** A new registered mechanism, not an edit to `gpm`. Registering it
separately means the existing harness compares them head-to-head automatically,
with signatures, costs and the report, and it keeps the published GPM result
intact.

Sketch: track a usage score per basis column, decay it each task, evict the
lowest-scoring columns once the basis exceeds a target occupancy.

**Expected.** This is the direction with the most headroom and the most
uncertainty. 10 seeds because a real effect here should exceed 0.035 and
anything smaller is not worth building on.

**Pre-registered failure.** If saturation drops but rho does not improve, the
hypothesis is wrong: the problem is not *how many* directions are consumed but
*which*. That would redirect the phase toward Stage 3 rather than tuning eviction
rates.

**Cost.** ~3 grid points × 10 seeds ≈ 30 runs ≈ 3h.

---

## Stage 3 · Soft projection — `gpm_soft`, 10 seeds

**Hypothesis.** A direction is currently binary: free or frozen. Weighting the
projector,

```
    g  <-  g ( I - M diag(lambda) Mᵀ ),   lambda_i in [0, 1]
```

means no direction is ever fully lost, so the free space never collapses.

**Change.** New registered mechanism. `lambda_i` derived from each direction's
share of retained variance.

**This deliberately gives up the exactness guarantee.** Hard projection makes
`dW·M = 0` exactly; soft projection makes it small. That guarantee is what
distinguishes GPM from a regulariser, so trading it needs measuring rather than
assuming — including whether the result is still meaningfully different from EWC,
which is also "penalise movement in important directions".

**Cost.** ~3 grid points × 10 seeds ≈ 30 runs ≈ 3h.

---

## Stage 4 · Head-to-head and hybrid — 10 seeds

Best variant vs baseline GPM vs `gpm + replay@capacity=50`.

The hybrid is worth a row because GPM buys −0.124 forgetting for zero stored
data, and a buffer far too small to work alone may cover what projection cannot
reach. Neither is close to sufficient by itself.

**Cost.** ~4 configs × 10 seeds ≈ 40 runs ≈ 4h.

---

## The loop, per stage

1. Implement locally; `pytest` green before anything else
2. Smoke run at `nano` — does it engage, does its signature fire
3. One sweep on the fixed 12-task Class-IL benchmark, seeds per the table above
4. `mechanism_report.py runs_stageN --compare runs_A`
5. Write down what happened **including when nothing did** — a stage that fails
   its pre-registered check is a result, not a wasted run

Nothing moves to the next stage until the current one has an answer. The
benchmark, controls, seeds and task list stay fixed throughout; if any of them
change, every comparison across stages breaks.

---

## What would count as success

GPM sits at rho 0.073. Replay is at 1.004. A variant reaching **rho 0.15** would
double the best rehearsal-free result on this benchmark and be worth writing up;
it would still be a seventh of what a replay buffer achieves.

Worth holding both facts at once: the gap is the reason this is research rather
than engineering, and it is also the reason not to over-invest before Stage 2
reports.


---

# PHASE OUTCOME — three stages, and what they establish

| stage | hypothesis | result |
|---|---|---|
| 1 · `eps_growth` | the rising threshold accelerates saturation | **null.** +0.066 at 5 seeds reversed to -0.005 at 10. Memory -46% is real |
| 2 · aging basis | monotonic growth is the binding constraint | **null.** Saturation 0.584 -> 0.250 bought rho +0.024 +/- 0.117, below the 0.05 bar |
| 3 · soft projection | the binary freeze is the binding constraint | **refuted, strongly.** rho 0.287 -> 0.108 as strength falls. t ~ 5.6 |

Stage 3 is the one that carries information. Every softening of the projector
made things monotonically worse, which means **the exactness guarantee is
load-bearing**: `dW.M = 0` is not incidental to GPM, it is the mechanism. Any
variant that relaxes it loses more than it gains. That is a positive finding
about why GPM works, obtained by breaking it.

## Four explanations ruled out by diagnostics

1. **Insufficient coverage** — GPM projects **99.7%** of parameters. Only the
   embeddings and norm scales (0.3%) are unprotected.
2. **Stale basis** — the stored subspace keeps capturing 80-92% of an old task's
   activation energy as training proceeds, and the fraction *rises* rather than
   decaying. The guarantee stays relevant.
3. **Plasticity cost** — `LA` is 100% at every strength and 96-101% at every
   occupancy. GPM has never blocked learning; that was never the problem.
4. **Rank-threshold noise** — real, documented, and not the dominant variance
   source: the sequential control has CV 30.7% with no mechanism at all.

## What remains

GPM holds an exact guarantee over 99.7% of the model, on a basis that stays
relevant, at no cost to plasticity — and still only cuts forgetting from the
control's 0.75 to 0.60. The residual is the ~15% of activation energy outside
the retained subspace, and on exact-match tasks that is enough to break an
answer.

Raising `eps` to capture more of it is the obvious move and was already
measured: eps 0.97 scored rho **-0.047**, worse than doing nothing. Capture more
and saturation destroys the score; capture less and protection is too thin. GPM
sits between two failure modes with no room between them.

**Recommendation: stop improving GPM.** Three pre-registered stages and four
ruled-out explanations put its ceiling near rho 0.29 at five tasks and 0.09 at
twelve. Closing the remaining gap to replay's 1.004 needs a different mechanism,
not a better GPM — and this phase is the evidence for that claim rather than an
assumption behind it.
