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

## Stage 2 · Aging basis — `gpm_aging`, 10 seeds

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
