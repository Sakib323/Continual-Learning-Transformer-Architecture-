# GPM — Full Progress Report

**Repository:** `/Users/macbook/Documents/Continual learning architecture`
**Status at time of writing:** GPM selected as the mechanism to develop; three
improvement stages completed, all negative; four alternative explanations for
its ceiling ruled out by measurement.

This report is written for someone with **no background in machine learning or
statistics**. Every symbol and piece of jargon is defined at first use. It is
also the handoff document for continuing the work, so it records not just
results but the tests that produced them, the bugs found along the way, and the
reasoning behind each decision.

Other mechanisms (EWC, replay, DER and the rest) are covered only briefly —
enough to explain why GPM was chosen. The bulk concerns GPM itself.

---

## Part 0 · The problem, from zero

### What a neural network is, for our purposes

A **model** (or **neural network**) is a large collection of numbers called
**weights** or **parameters**. Ours holds **23,673,344** of them. Feeding it an
input produces an output; **training** means adjusting those numbers until the
outputs are correct.

Training works by **gradient descent**. For each weight the system computes a
**gradient** — a number saying "nudge this up or down to reduce the error." The
procedure that computes all gradients at once is called **backpropagation**
(backprop). An **optimizer** turns gradients into actual weight changes; we use
**AdamW**, a standard choice.

Useful vocabulary:

- **batch** — a group of examples processed together. Ours: 32 at a time.
- **step** — one batch in, one weight update out. Each task gets 400 steps.
- **learning rate** — how large each update is.
- **loss** (`L`) — a number measuring how wrong the model currently is. Training
  minimises it.
- **inference** — using a trained model to answer, without changing weights.

### Catastrophic forgetting

Train a network on Task A, then on Task B. The weight changes that make B work
**destroy** the arrangement that made A work. Not gradual decay — collapse. In
our measurements a plain model drops from 100% accuracy on the first task to
near zero after a few more.

**Continual learning (CL)** is the research field trying to fix this. A
**mechanism** is one specific proposed fix — a small piece of code that attaches
to the training process.

Two properties matter and pull against each other:

- **Stability** — keeping old knowledge.
- **Plasticity** — still being able to learn new things.

A mechanism can stop forgetting entirely by freezing the model, but then it
learns nothing new. That is not a solution, and measuring the two separately is
how you catch it. This tension is the **stability–plasticity trade-off**, and it
recurs throughout this report.

### The larger goal

The project's eventual aim is a system that can keep learning after deployment —
absorbing new information without destroying what it already knows. Addressing
catastrophic forgetting is the first step. The plan has five phases:

| phase | content | status |
|---|---|---|
| 0 | scaffolding, mechanism contract, controls | complete |
| 1 | vertical slice — one mechanism per injection surface | complete |
| 2 | build out 15 mechanisms | complete |
| 3 | ablation — measure which ones work | complete |
| 4 | improve the winner | **three stages done, all negative** |
| 5 | port to a real pretrained model at scale | not started |

---

## Part 1 · What was built

### The model — `olmo2_cl/`

A from-scratch implementation of the **OLMo 2** transformer architecture
(**292 lines**, `modeling_olmo2.py`). A **transformer** is the architecture
behind modern language models. Ours reproduces OLMo 2's three distinctive
choices:

- **Reordered norm** — normalisation applied to each sublayer's *output* inside
  the residual connection, not to its input.
- **QK-Norm before RoPE** — normalising the query and key projections before
  applying rotary position embeddings.
- **No biases** anywhere.

Plus SwiGLU feed-forward blocks, grouped-query attention, and RoPE. It is
architecture-faithful but **not** a copy of AI2's source — `VENDORED_FROM.md`
records the provenance. Parameter names match HuggingFace's `Olmo2` convention
exactly, so a real OLMo 2 checkpoint would load into it module-for-module.

**Size presets** used in this work:

| preset | parameters |
|---|---|
| `nano` | ~1M |
| `tiny` | 4.5M |
| `small` | 23.7M |

### The injection system

The model exposes an `Injector` protocol threaded through every module. There
are two ways a mechanism attaches:

1. **`observe(site, layer_idx, tensor)`** — read or transform an activation
   mid-forward. Six named sites: `embed`, `mlp_hidden`, `attn_out`,
   `resid_mid`, `resid_out`, `final`.
2. **`build_attention(...)` / `build_mlp(...)`** — replace a submodule at
   construction time.

With the default `NullInjector`, the model is a plain OLMo 2 and no mechanism
code runs. **The model has no idea continual learning exists** — eight one-line
hooks are the entire interface.

### The mechanism harness — `Continual Learning Mechanism Stack (CLMS)/`

Every mechanism implements a common contract (`base.py`) with hooks for
`setup`, `on_task_start`, `on_batch`, `compute_loss`, `before_step`,
`after_step`, `on_task_end`, plus `signature`, `cost_report` and
`state_dict`/`load_state_dict` for checkpointing.

A **`Composer`** validates and runs a set of mechanisms together, enforcing
declared conflicts and required stream capabilities.

### The test suite — `tests/test_contract.py`

**174 tests, 57 test functions.** These are not decoration; several caught real
bugs described later in this report. GPM-specific coverage:

```
test_training_randomness_is_seeded          (parametrised over gpm, shrink_perturb, cbp)
test_gpm_saturation_probe_fires_before_the_cap
test_gpm_records_its_saturation_trajectory
test_aging_basis_stops_growing
test_aging_basis_survives_a_resume
test_soft_projection_leaves_exactly_the_stated_fraction
test_soft_projection_probe_detects_a_miswired_strength
```

---

## Part 2 · The benchmark

### Tasks

Small algorithmic puzzles where the correct answer is unambiguous:

| task | what it asks |
|---|---|
| `copy` | repeat an 8-symbol sequence back |
| `reverse` | reverse it |
| `sort` | sort it ascending |
| `sortdesc` | sort it descending |
| `modadd7/13/23/31` | add two numbers modulo 7 / 13 / 23 / 31 |
| `induction6/8` | the final symbol repeats an earlier one; answer with what followed that earlier occurrence |
| `copy12`, `reverse12` | 12-symbol variants |

The current 12-task stream:

```
copy, modadd7, reverse, sort, modadd13, induction6,
copy12, sortdesc, modadd23, reverse12, induction8, modadd31
```

Families are **interleaved deliberately** — consecutive tasks from the same
family transfer to each other, which would suppress the interference the
benchmark exists to measure. A test enforces this.

**Chance baseline.** What you would score by guessing. Essential context: a task
with two possible answers gives 50% for free, so "50% accuracy" can mean zero
learning. Every task declares its own chance level, and tests enforce that
retrieval and arithmetic tasks have a non-trivial one.

### The two scenarios

This distinction changes results more than any mechanism does.

- **Task-IL (Task-Incremental Learning)** — the model is *told* which task each
  input belongs to, via a special token. Easier.
- **Class-IL (Class-Incremental Learning)** — the model receives only the input
  and must infer the task itself. This is what a deployed system faces, so it is
  the headline benchmark.

The same mechanism scores very differently: GPM reaches **0.227 in Task-IL** on
five tasks and **0.073 in Class-IL** on twelve.

Related terms:

- **Task boundary** — a signal saying "a new task starts now." Seven of the
  fifteen mechanisms require one. GPM is among them.
- **Task ID at inference** — needing to be told *which* task a question belongs
  to when answering. Only XdG requires this; it is disqualifying for a deployed
  system, and the harness refuses to run XdG in Class-IL rather than emit a
  meaningless number.

### The controls

Every measurement is relative to two reference runs:

- **`control_sequential` — the floor.** Train on the stream with no mechanism at
  all. This is what forgetting costs you.
- **`control_joint` — the ceiling.** Train on all tasks *simultaneously*,
  shuffled together, so no forgetting is possible. The best any mechanism could
  match.
- **`control_independent`** — a fresh model per task. Tells you whether a task
  is learnable at all, independent of forgetting. Used to catch broken tasks
  before spending money.

**Span = ceiling − floor.** The room a mechanism has to work in. If the span is
tiny, no mechanism can look different from any other and the benchmark is
useless — the report warns below 0.05.

### The metrics

- **AA (Average Accuracy)** — after finishing the whole stream, accuracy
  averaged over every task. The headline number.

- **rho (ρ, recovery ratio)** — AA rescaled to be comparable across benchmarks:

  ```
  rho = (AA − floor) / (ceiling − floor)
  ```

  `rho = 0` means no better than doing nothing; `rho = 1` means as good as joint
  training; negative means actively worse than no mechanism. Because it is
  normalised against controls from the **same sweep**, rho stays comparable
  across different hardware where raw accuracy does not.

- **FM (Forgetting Measure)** — for each task, its best-ever accuracy minus its
  final accuracy. High means it forgot a lot. The plain control sits at 0.758.

- **LA (Learning Accuracy)** — how well each task was learned *at the moment it
  was trained*, before later tasks could interfere. Reported as a percentage of
  the control's LA. This separates two failures that look identical in AA:
  learned-then-forgot (LA high, AA low) versus never-learned (LA low). We
  measured empirically that **AA ≈ LA − FM**, correlation 0.987–0.999.

- **FWT (Forward Transfer)** — whether earlier tasks help you learn later ones
  *faster*: accuracy on a task before ever training on it, above chance.

- **BWT (Backward Transfer)** — whether later learning helped or hurt earlier
  tasks. Negative is the normal case.

### Statistical vocabulary

- **seed** — a number fixing all randomness in a run. Same seed, same code, same
  hardware gives an identical result; a different seed gives a different but
  equally valid one.
- **sd (standard deviation)** — how spread out results are across seeds.
- **SE (standard error)** — uncertainty in the *mean*, = sd / √n. Shrinks with
  more seeds, which is why seed count is a budget decision.
- **CV (coefficient of variation)** — sd ÷ mean, as a percentage.
- **Δrho** — the difference in rho between two configurations.
- **t-statistic** — the difference divided by its uncertainty. |t| above ~2
  suggests the difference is real.
- **p-value** — probability of seeing a difference this large if there were
  really none. Below 0.05 is the conventional bar.
- **MDE (Minimum Detectable Effect)** — the smallest difference an experiment
  can reliably find at its seed count:

  | seeds | MDE (rho) |
  |---|---|
  | 3 | 0.063 |
  | 5 | 0.049 |
  | 10 | 0.035 |
  | 20 | 0.024 |

  **Running an experiment whose MDE exceeds the effect you expect wastes money** —
  it cannot answer its own question.

### Experimental vocabulary

- **run** — one training of one model with one configuration and one seed.
- **configuration** — one specific setting, e.g. `gpm[eps_base=0.8]`.
- **preset** — a named bundle of settings.
- **grid / grid point** — a list of values to try for one parameter.
- **sweep** — running every grid point at every seed.
- **ablation** — testing many mechanisms under identical conditions.
- **signature probe** — an independent check that a mechanism did the *internal*
  thing its paper describes, not merely that it scored well. **This caught four
  bugs that accuracy alone hid.**
- **inert** — a mechanism that attached without error and then did nothing.
  Detected and reported.
- **gate** — a cheap check run *before* an expensive one.
- **pre-registration** — writing down what you expect and what would count as
  being wrong, *before* running.

---

## Part 3 · The ablation — how GPM was chosen

Fifteen mechanisms were implemented and measured. One line each:

| mechanism | idea |
|---|---|
| **replay** | keep a buffer of old examples, mix them into new training |
| **DER** | replay plus matching the model's own earlier outputs |
| **EWC** | estimate which weights matter for old tasks, penalise changing them |
| **SI** | like EWC, importance computed during training |
| **LwF** | keep a copy of the old model, require agreement with it |
| **GPM** | project gradients away from directions old tasks used |
| **LoRA / O-LoRA** | add small per-task weight sets, freeze the rest |
| **L2P** | learn a pool of prompts, select per input |
| **kWTA / XdG** | let only some units activate per task |
| **CBP / Shrink-Perturb** | continually refresh units to restore plasticity |
| **Memory layer / Sparse update** | add addressable memory; update few parameters |

### Task-IL results, 5 tasks, 23.7M parameters

| mechanism | rho | LA% | verdict |
|---|---|---|---|
| **replay** | **1.004** | 101% | matches joint training |
| **DER** | **0.982** | 100% | matches joint training |
| EWC | 0.335 | **53%** | partial — buys retention with plasticity |
| **GPM** | **0.227** | **101%** | **partial — keeps plasticity** |
| everything else | ≤0.077 | | no effect or harmful |

### Class-IL results, 12 tasks — the realistic setting

```
floor 0.0813   ceiling 0.8753   span +0.7940

ewc[lam=10]        0.104 ± 0.065   LA 42%
gpm[eps_base=0.8]  0.073 ± 0.039   LA 96%
lwf[lam=0.1]       0.060 ± 0.050   LA 90%
everything else   ≤0.051
```

**No rehearsal-free mechanism works on decidable Class-IL.** This replicates a
known finding — the paper the registry already cites for EWC and SI
(arXiv 1904.07734, van de Ven & Tolias) reports exactly this collapse from
Task-IL to Class-IL. Independently reproducing it is strong evidence the harness
measures the right thing.

### Why GPM was selected

Not the highest rho. The selection criteria came from the eventual deployment
architecture:

| requirement | why | GPM |
|---|---|---|
| stores no raw user data | privacy; the objection to replay | ✅ stores a derived subspace |
| preserves plasticity | the system runs indefinitely | ✅ **LA 96–101%** |
| consolidates into base weights | no adapter routing at inference | ✅ |

Among mechanisms keeping LA ≥ 90%, GPM reduces forgetting the most:

```
sequential control: FM 0.758

gpm[eps=0.9]        LA  91%   FM 0.634   −0.124 vs control
gpm[eps=0.8]        LA  96%   FM 0.664   −0.094
memory_sparse       LA  98%   FM 0.699   −0.058
shrink_perturb      LA 100%   FM 0.720   −0.038
```

EWC scores higher on rho but at LA 42% — it retains by refusing to learn. At
λ=10000 it reaches FM **exactly 0.000** with LA 19%: a frozen model, which is
the endpoint of the trade rather than a solution.

Replay and DER were set aside deliberately — not because they fail (they are
the only two that work) but because storing raw examples forever conflicts with
the deployment design.

---

## Part 4 · GPM in detail

### The idea

Learn each new task **only in directions the old tasks never used**, so old
behaviour is preserved by geometry rather than by penalty.

### The symbols

**Scalars**

| symbol | meaning | our value |
|---|---|---|
| `L` | the loss | — |
| `d_in`, `d_out` | layer input / output width | 512 / 512 at `small` |
| `k` | rank — how many directions are stored | ~0.5·d_in after 12 tasks |
| `eps` (ε) | fraction of activation variance retained | 0.8–0.97 |
| `sigma_i` (σᵢ) | variance along the i-th input direction | spans ~200× |

**Vectors**

| symbol | meaning |
|---|---|
| `x` | input activation to a layer |
| `x_par` | the part of `x` inside the protected subspace |
| `x_perp` | the residual, `x − x_par` |
| `y = Wx` | the layer's output |

**Matrices**

| symbol | meaning | shape |
|---|---|---|
| `W` | layer weights | d_out × d_in |
| `dW` (also `g`) | the proposed update — the raw gradient | d_out × d_in |
| `R` | stacked input activations collected from old tasks | n_samples × d_in |
| `C = RᵀR` | input covariance | d_in × d_in |
| `M` | orthonormal basis of the protected subspace | d_in × k |
| `P = I − MMᵀ` | projector onto the free complement | d_in × d_in |

**Jargon in those definitions**

- **subspace** — a set of directions within the space of possible inputs.
- **basis** — a minimal set of directions describing a subspace, like axes on a
  graph.
- **orthogonal** — at right angles; independent.
- **orthonormal** — orthogonal *and* each direction has length 1. This gives
  `MᵀM = I`, the single fact that makes the guarantee exact.
- **projection** — splitting a quantity into a part inside a subspace and a part
  outside it.
- **rank** — how many independent directions something spans.
- **`Mᵀ`** — the transpose of M (rows and columns swapped).
- **`I`** — the identity matrix; multiplying by it changes nothing.

### The update rule

```
dW  ←  dW − (dW M) Mᵀ   =   dW (I − M Mᵀ)   =   dW P
```

In the code (`projection.py`, `before_step`):

```python
g = module.weight.grad          # (out, in)
Md = M.to(g.device, g.dtype)    # (in, k)
module.weight.grad = g - (g @ Md) @ Md.T
```

### Why it works — the guarantee

Because `M` is orthonormal, `MᵀM = I`, so:

```
(dW P) M  =  dW M − dW M (Mᵀ M)  =  dW M − dW M  =  0
```

Therefore for any input `x = Mc` lying in the protected subspace:

```
(W + dW P) x  =  Wx + (dW P M) c  =  Wx
```

The layer's response to every stored input is **exactly** unchanged — not
penalised, not discouraged, algebraically identical. This is the difference in
kind between GPM (a constraint) and EWC (a penalty).

**Verified property:** the projection acts **row-wise**. Row *i* of the result
depends only on row *i* of `dW` — confirmed numerically to 1.9e-07. So the
operation on one row is the whole story, repeated `d_out` times.

### Where the basis comes from

At each task boundary (`on_task_end`):

1. **Collect.** Run a few batches forward with hooks on every linear layer,
   recording the *input* activations into `R` (n_samples × d_in).
2. **Remove what is already covered:** `R ← R − (R M_old) M_oldᵀ`. Only the
   genuinely new part of the representation matters.
3. **Find principal directions** via **SVD (Singular Value Decomposition)**, a
   standard procedure that finds the dominant directions in a pile of data and
   how important each is:
   ```
   U, S, _ = svd(Rᵀ R)
   ```
   Since `RᵀR` is symmetric, `U` holds its **eigenvectors** (the directions) and
   `S` the **eigenvalues** (variance along each).
4. **Keep enough to explain `eps` of the variance:**
   ```
   csum = cumsum(S) / sum(S)
   k    = #{ i : csum_i < eps } + 1
   ```
5. **Append:** `M ← [ M_old | U[:, :k] ]`

### Every parameter

| parameter | default | meaning |
|---|---|---|
| `eps_base` | 0.9 | starting fraction of variance to retain |
| `eps_growth` | 0.005 | how `eps` changes per task |
| `eps_floor` | 0.50 | lower clamp (added in Stage 1; without it, negative growth drives eps below zero on a long stream) |
| `max_bases_frac` | 0.75 | cap: never freeze more than this fraction of a layer |
| `collect_batches` | 4 | batches used to build the basis |
| `min_features` | 8 | skip layers narrower than this |
| `saturation_warn_frac` | 0.85 | where probe C2 fires (added Stage 0) |

`eps` is computed as:

```python
eps = min(0.99, max(eps_floor, eps_base + eps_growth * tasks_seen))
```

### Coverage

GPM projects **99.7%** of all parameters (23,592,960 of 23,673,344). Only the
embedding table and normalisation scales — 0.3% — are unprotected.

### Signature probes

- **C2** — fraction of gradient directions consumed. Fires at 85% of
  `max_bases_frac`.
- **C4** (on the soft variant) — error between the measured surviving gradient
  fraction and the value the `strength` parameter predicts.

---

## Part 5 · Every sweep run, chronologically

| sweep | scenario | tasks | GPU | what it was for |
|---|---|---|---|---|
| v1 | Task-IL | 6 | RTX 4060 Ti | first full 15-mechanism ablation |
| v2 | Task-IL | 5 | RTX 2060 | after benchmark fixes |
| v3 | Task-IL | 5 | RTX 2060 | re-run of 3 changed mechanisms |
| v4 | Task-IL | 5 | RTX 2060 | with forward transfer wired up |
| v5 | Task-IL | 5 | RTX 2060 | 5-seed re-run after the RNG fix |
| runs_A/B | **Class-IL** | 12 | RTX 3060 | the decisive ablation, 227 runs, zero crashes |
| runs_s1 | Class-IL | 12 | GTX 1660S | Stage 1 — `eps_growth` |
| runs_s2 | Task-IL | 12 | RTX 3060 | Stage 2 — aging basis |
| runs_s3 | Task-IL | 5 | RTX 3060 | Stage 3 — soft projection |

All on rented GPUs via vast.ai, roughly $0.06–0.12/hour.

---

## Part 6 · Bugs found — and how

Each of these produced a plausible-looking number that meant something other
than it appeared to. None were visible in accuracy alone.

### 1. EWC's Fisher collapsed to 1e-11

The **Fisher information** is EWC's measure of "how much does this weight
matter." The *empirical* Fisher uses squared gradients on ground-truth labels —
but once a task is solved those gradients vanish, so the Fisher collapses and no
penalty strength can do anything. Switching to the *model* Fisher gained ~1000×
and still landed at 1e-9. Normalising each task's Fisher to unit mean is what
makes the strength parameter scale-free.

### 2. O-LoRA was frozen, and the bug flattered it

`set_active(0)` left adapters 1..N with `requires_grad=False`, and
`build_optimizer` skipped exactly those — so they never entered the optimizer.
From task 1 the model was frozen solid: FM exactly 0.0, representation
similarity exactly 1.0, identical results at every parameter value.

After the fix O-LoRA's score got **worse** (rho +0.107 → −0.178): a model frozen
on task 0 outscores one that actually trains. The bug had been making it look
better.

### 3. `induction` was ill-posed

The cue symbol was not guaranteed unique, so **11.4%** of sequences had an
answer undecidable from the input, regardless of model quality.

### 4. Gradient masking does not freeze parameters under AdamW

`sparse_update` expressed "only update these slots" by multiplying the rest of
the gradient by zero. **Adam's momentum and AdamW's weight decay kept moving
them anyway** — measured at 67% of a normal update. Zeroing a gradient does not
freeze a weight; only `grad = None` or restoring the value after the step does.

The fix did **not** rescue the mechanism: it went from 1/6 signature checks to
6/6 while rho stayed at 0.019 ± 0.066. Provably correct, and still useless —
a far stronger negative result than one confounded by a bug.

### 5. The Class-IL benchmark was undecidable

The most expensive bug. Without a task token the model sees only the input, and:

- `modadd7/13/23/31` all took **2 symbols** from overlapping ranges
- `copy`, `reverse`, `sort`, `sortdesc`, `induction8` all took **8 random
  symbols** from the same vocabulary

Eleven of twelve tasks sat in a collision group demanding different answers from
identical inputs.

```
oracle ceiling if forced to guess:  0.3333
measured joint ceiling:             0.4513
```

The ceiling sat just above the pure-guessing bound. **A full day of compute
measured how well the model guessed which task it had been handed.**

Fixed by giving each task a disjoint symbol range within its input shape —
realistic, since different topics genuinely use different vocabulary. Required
raising the symbol count from 64 to 108. Oracle ceiling after the fix: **1.0000**.
Two tests now enforce decidability.

### 6. Three mechanisms were not reproducible at a fixed seed

`gpm`, `shrink_perturb` and `continual_backprop` drew training randomness from
the **global** random number generator. Same seed, same config:

```
gpm run a: AA = 0.243056
gpm run b: AA = 0.538194     difference 0.295
```

That spread is larger than most mechanisms' entire effect — and **seed-to-seed
error bars cannot see it**, because the variance lives *within* a seed. Fixed
with a per-mechanism seeded generator; verified bit-exact on CPU.

A residual remains on GPU and is *not* our code: GPM selects its rank by
thresholding, so float noise near the threshold flips the retained rank by one —
a discrete change that compounds.

### 7. Two signature probes were measuring the wrong thing

Probes need the same scrutiny as mechanisms.

- **A4** counted slots accessed *cumulatively*, which reaches 1.0 over any long
  run however sparse each step is. It failed 5 of 6 seeds while the mechanism
  worked correctly. Rewritten as concentration.
- **B2** was scoped to the whole model while the mechanism masks only part of
  it, burying the signal. Rescoped.
- **D1** (replay) first passed at 0.5028 against a 0.5 threshold — one seed from
  a false alarm. Rethresholded to 0.2, where the two regimes actually separate.

### 8. Uneven tuning grids gave some mechanisms a free advantage

The report shows each mechanism's **best** grid point. Three mechanisms had 4
points, nine had 3, and two had **1**. Best-of-4 versus best-of-1 is worth
roughly half a standard deviation — the same size as the gaps between
mechanisms. Equalised, with a test.

### 9. Silent CPU fallback after a GPU fault

A GPU faulted mid-sweep with `unspecified launch failure`. Every subsequent run
**silently fell back to CPU** and completed normally — 20× slower, and recorded
with a different device than its neighbours. `pick_device` now aborts instead,
and the report detects mixed hardware.

---

## Part 7 · The improvement phase

### Methodology

One change at a time, each with a stated hypothesis and a **pre-registered way
to be wrong**. Config-only changes may share a sweep because the grid separates
them into distinct rows.

### Stage 0 · Instrumentation (no GPU)

Saturation — how full the basis is — is the quantity the whole phase turns on.

Probe **C2** measured the right thing but fired only at the cap. At 98% of cap
GPM scored rho **−0.047** (actively harmful) and the probe passed 5/5.
Recalibrated to 85% of cap, against the measured data:

```
 72% of cap → rho +0.073   healthy     ← only this passes now
 97% of cap → rho +0.058   degrading
 98% of cap → rho −0.047   harmful
100% of cap → rho  0.001   plasticity gone
```

Also added per-task **saturation trajectory** recording, surviving
checkpoint/resume.

### Stage 1 · Invert `eps_growth`

**Hypothesis.** `eps` rises with task count, so later tasks retain *more*
variance and consume *more* directions exactly when least affordable. Reversing
it should delay saturation.

**At 5 seeds:** rho 0.124 vs baseline 0.058, Δ **+0.066**, monotone across all
four levels, exceeding the pre-registered MDE of 0.049. Adopted provisionally.

**At 10 seeds it reversed and vanished:** Δ **−0.005**, t = −0.18, and the
negative arm won in exactly 5 of 10 seeds.

```
eps_growth=-0.01,  seeds 0-4: [0.080 0.094 0.031 0.151 0.251]  mean 0.121
                   seeds 5-9: [0.034 0.011 0.079 0.046 0.097]  mean 0.053
eps_growth=+0.005, seeds 0-4: [0.016 0.077 0.057 0.066 0.062]  mean 0.056
                   seeds 5-9: [0.130 0.110 0.061 0.143 0.203]  mean 0.129
```

Same config, same GPU, same code — **both arms swung by more than the claimed
effect.** One seed scoring 0.251 dragged a five-seed average from 0.053 to
0.121. This is the project's cautionary example on seed counts.

**What survives:** memory 31.5 MB vs 58.5 MB, a **46% reduction**, deterministic
rather than statistical. Adopt `eps_growth = −0.01` for that, with no accuracy
claim.

### Stage 2 · Aging basis (`gpm_aging`)

**Hypothesis.** Monotonic growth is the binding constraint. A basis that evicts
directions converts unbounded consumption into a steady state.

**Implementation.** Keeps a usage score per basis column, measuring how much the
current task's activations project onto each existing direction; decays scores
by `usage_decay = 0.7` each task; evicts the lowest-scoring columns once
occupancy exceeds `target_occupancy`. New directions start at the mean of
surviving scores rather than zero, so they are not evicted before proving
useful.

**Eviction worked perfectly:**

```
target   final sat  evicted   trajectory
0.25         0.250    8,929   [.01 .03 .06 .09 .14 .22 .24 .24 .25 .25 .25 .25]  flat from task 7
0.40         0.386    5,345   [.01 .03 .06 .09 .14 .22 .29 .32 .37 .38 .39 .39]
0.55         0.504    2,088   [.01 .03 .06 .09 .14 .22 .29 .33 .43 .46 .50 .52]
0.75         0.584        0   [.01 .03 .06 .09 .14 .22 .29 .33 .43 .47 .53 .61]  baseline, still climbing
```

**And rho barely moved:**

```
target=0.25   rho 0.111 ± 0.117   ← best
target=0.55   rho 0.090 ± 0.085
target=0.75   rho 0.087 ± 0.081   ← baseline
target=0.40   rho 0.080 ± 0.088   ← breaks any trend
```

Δrho **+0.024** against a pre-registered bar of 0.05, with sd five times the
effect and no monotonic ordering.

**This is the pre-registered failure, exactly as written.** Saturation dropped
57% and bought nothing. Monotonic basis growth was **not** the binding
constraint.

### Stage 3 · Soft projection (`gpm_soft`)

**Hypothesis.** A direction is currently binary — fully frozen or fully free.
Perhaps the binary nature itself is the constraint.

**The equation:**

```
dW  ←  dW − strength · (dW M) Mᵀ
```

`strength = 1.0` reproduces baseline GPM exactly. Below that, every stored
direction keeps a `(1 − strength)` share of its gradient.

**This deliberately gives up the exactness guarantee.** Probe **C4** measures
the surviving fraction directly, verified against the arithmetic to 1e-7.

**Result — refuted, with a strong effect in the wrong direction:**

```
strength=1.0   rho 0.287 ± 0.061   FM 0.602   LA 100%   ← baseline
strength=0.85  rho 0.157 ± 0.140   FM 0.716   LA 100%
strength=0.50  rho 0.109 ± 0.067   FM 0.756   LA 100%
strength=0.70  rho 0.108 ± 0.038   FM 0.755   LA 100%
```

Δrho **−0.179**, t ≈ 5.6. Softening makes it monotonically worse. At strength
0.5–0.7 the forgetting (0.755) equals the control's — the protection vanishes
entirely.

**The exactness guarantee is load-bearing.** `dW·M = 0` is not incidental to
GPM; it *is* the mechanism. This is the one positive finding of the phase,
obtained by breaking it.

---

## Part 8 · Diagnostics — four explanations ruled out

Each of these was proposed, tested, and refuted. All ran locally at zero GPU
cost.

### 1. Insufficient coverage — refuted

GPM projects **99.7%** of parameters. Only embeddings and norm scales are
unprotected.

### 2. Stale basis — refuted

Hypothesis: as the network changes, the stored subspace stops describing the
activations it was built from. Measured the fraction of an old task's activation
energy still captured:

```
after copy       0.812
after modadd7    0.840
after reverse    0.786
after sort       0.866
after modadd13   0.917
```

It **rises**, because the basis grows. No staleness.

### 3. Plasticity cost — refuted

**LA is 100% at every strength and 96–101% at every occupancy.** GPM has never
blocked learning. Both Stage 2 and Stage 3 were premised on recovering plasticity
that was never lost — the evidence was in the LA column the whole time.

### 4. Rank-threshold noise dominating the variance — refuted

Real and documented, but not the dominant source. The **sequential control has
CV 30.7% with no mechanism at all**, while `control_joint` has 0.4%. The chaos
is in sequential training on a hard stream, not in GPM.

Further variance attempts, all refuted:

- **More collection batches** — relative rank spread stayed 22–25% at 4, 16, 64.
- **Paired estimator** (each seed against its own floor) — made it *worse*,
  sd 0.070 → 0.084. Floor-to-mechanism correlation is **−0.234**; there is no
  shared "unlucky seed" factor to cancel.
- **More eval batches** — 4× the eval data moved sd by **+3%**.

### A correction to the protocol

After Stage 1 the recommendation was "develop on Task-IL, it gives 4× the
signal-to-noise." That comparison was **confounded**:

| benchmark | tasks | rho | sd | SNR |
|---|---|---|---|---|
| Task-IL | 5 | 0.227 | 0.050 | **4.5** |
| Class-IL | 12 | 0.093 | 0.054 | 1.7 |
| **Task-IL** | **12** | **0.087** | **0.081** | **1.07** |

It compared 5-task Task-IL against 12-task Class-IL and attributed the
difference to the scenario. The real driver is **task count**. At matched 12
tasks, Task-IL is *worse*. Stage 2 ran on Task-IL expecting better precision and
got sd 0.081–0.117 against Class-IL's 0.054.

---

## Part 9 · What the three stages establish

| stage | hypothesis | result |
|---|---|---|
| 1 · `eps_growth` | the rising threshold accelerates saturation | **null** (memory −46% is real) |
| 2 · aging basis | monotonic growth is the binding constraint | **null** (Δrho +0.024 ± 0.117) |
| 3 · soft projection | the binary freeze is the binding constraint | **refuted, strongly** (t ≈ 5.6) |

**Where GPM actually stands.** It holds an exact guarantee over 99.7% of the
model, on a basis that stays relevant, at zero cost to plasticity — and still
only cuts forgetting from the control's 0.758 to 0.602.

The residual is the **~15% of activation energy outside the retained subspace**
(mean capture 0.849 per layer). On exact-match tasks that is enough to break an
answer. Raising `eps` to capture more was already measured: eps 0.97 scored rho
**−0.047**, worse than doing nothing, because saturation takes over.

**GPM sits between two failure modes with no room between them.** Capture more
and saturation destroys the score; capture less and protection is too thin.

Its measured ceiling: **rho ≈ 0.29 at five tasks (Task-IL), rho ≈ 0.09 at twelve
(Class-IL)**.

---

## Part 10 · Current state of the repository

```
olmo2_cl/                  292-line OLMo 2 transformer + config
Continual Learning Mechanism Stack (CLMS)/clms/
    base.py                the Mechanism contract
    compose.py             validation and orchestration
    config.py              presets, tuning grids, defaults
    data/synthetic.py      tasks, streams, chance baselines, symbol ranges
    eval/metrics.py        AA, rho, FM, LA, FWT, BWT
    eval/probes.py         diagnostic probes
    mechanisms/            15 mechanisms + 2 GPM variants
train.py                   the training loop
scripts/sweep.py           grid sweeps
scripts/mechanism_report.py  the per-mechanism report
scripts/run_benchmark.sh   unattended two-track benchmark
scripts/vast_setup.sh      GPU provisioning with a real kernel-launch check
tests/test_contract.py     174 tests, 57 functions
RESULTS.md                 the ablation write-up
GLOSSARY.md                every term, for newcomers
RUNBOOK.md                 how to run a sweep on vast.ai
gpm/README.md              GPM mechanism explainer
gpm/WORKFLOW.md            the improvement phase, with outcomes
```

**Registered GPM variants**

| name | class | grid |
|---|---|---|
| `gpm` | `GradientProjectionMemory` | `eps_base ∈ [0.8, 0.9, 0.97]` |
| `gpm_growth` | (preset over `gpm`) | `eps_growth ∈ [−0.01, −0.005, 0, 0.005]` |
| `gpm_aging` | `AgingGradientProjectionMemory` | `target_occupancy ∈ [0.25, 0.4, 0.55, 0.75]` |
| `gpm_soft` | `SoftGradientProjectionMemory` | `strength ∈ [0.5, 0.7, 0.85, 1.0]` |

All three variants are registered, tested, and have their conflict with
`continual_backprop` declared symmetrically.

---

## Part 11 · Open questions

1. **Is GPM's ceiling real or an artefact of scale?** Everything is measured at
   23.7M parameters on synthetic algorithmic tasks. Phase 5 (a real pretrained
   model) has never been run.

2. **Does the 15% residual explain the gap quantitatively?** The per-layer
   capture of 0.849 compounds through depth — `0.849⁴ = 0.519` over four
   blocks. This is suggestive, not established.

3. **No mechanism produces forward transfer.** FWT is ≈ −0.055 across every
   mechanism and both scenarios. These methods preserve; none of them compound.

4. **`si` and `shrink_perturb` still lack signature probes** — scores with no
   independent check that the mechanism engaged.

5. **`induction8` is marginal** — 0.266 against a chance baseline of 0.1429,
   only 1.9× chance. It compresses the span slightly. Kept because the span is
   healthy, but it is the row to distrust first.

---

## Appendix · What makes a result trustworthy here

Three habits produced most of the value, and abandoning them is the main risk in
continuing:

**Signature probes.** Four bugs were invisible in accuracy and visible in the
probes. One of them was making a mechanism look *better* than its fixed version.

**Pre-registration.** Stage 2's failure was written down before it ran. Without
that, "saturation dropped 57%" would have been reported as progress.

**Gates before spending.** Two cheap checks caught broken benchmarks that would
have wasted a full day each. One was caught too late, and did.

And the most expensive lesson, from Stage 1: **a clean-looking effect at five
seeds reversed sign at ten.** The MDE table exists because of it.
