# Glossary — every term used in this project

Written for someone with no machine-learning or statistics background. Terms are
grouped by what they are *for*, not alphabetically, because the groups explain
each other. Real numbers from this project are used throughout, so the
definitions and the results stay consistent.

---

## 1 · The problem we are solving

**Neural network / model.** A large collection of numbers ("weights" or
"parameters") that turns an input into an output. Ours has 23.7 million of them.
Training means adjusting those numbers until the outputs are right.

**Task.** One thing we want the model to do. Ours are small algorithmic puzzles:
`copy` (repeat a sequence back), `reverse`, `sort`, `modadd7` (add two numbers
modulo 7), `induction6` (spot a repeated symbol and predict what followed it
last time). Twelve of them in the current benchmark.

**Stream.** The tasks presented one after another — learn `copy`, then
`modadd7`, then `reverse`, and so on. The model never sees an earlier task again
once it has moved on.

**Catastrophic forgetting.** The central problem. When a network learns task 2,
the weight changes that make task 2 work *destroy* the arrangement that made task
1 work. Not gradual decay — collapse. In our measurements a plain model drops
from 100% on `copy` to near zero after a few more tasks.

**Continual learning (CL).** The research field trying to fix that: learn new
things without wrecking the old ones.

**Mechanism.** One specific proposed fix. We implemented fifteen from the
literature — GPM, EWC, replay, and so on. Each is a small piece of code that
attaches to the training process.

**Plasticity.** The ability to still learn *new* things. A mechanism can stop
forgetting by freezing the model, but then it learns nothing new either. That is
not a solution, and measuring it separately is how you catch it.

**Stability–plasticity trade-off.** The core tension. Stability = keeping old
knowledge; plasticity = acquiring new. Every mechanism we measured trades one
for the other; none escaped the exchange rate.

---

## 2 · How training works (the mechanical bits)

**Gradient.** For each weight, a number saying "nudge this up or down to reduce
the error." Computed by **backpropagation** ("backprop"). The gradient is the
whole reason a network can learn.

**Optimizer.** The rule that turns gradients into actual weight changes. We use
**AdamW**, a standard choice.

**Momentum.** AdamW remembers recent gradients and keeps moving in that
direction, like a rolling ball. This caused a real bug in our project: setting a
gradient to zero does *not* stop the weight moving, because momentum carries it.

**Weight decay.** AdamW also shrinks every weight slightly on each step,
independently of the gradient. Same consequence — a "frozen" weight still drifts.

**Batch.** A group of examples processed together. Ours are 32 at a time.

**Step.** One batch in, one weight update out. Each task gets 400 steps.

**Learning rate.** How big each update is. **Warm-up** means starting small and
ramping up, which stabilises early training.

**Checkpoint.** A saved copy of all the weights, so an interrupted run can
resume rather than restart.

**Inference.** Using a trained model to answer, without changing any weights. The
distinction matters: some mechanisms need extra information *at inference*, which
a deployed system usually cannot supply.

---

## 3 · The scoreboard — how we measure a mechanism

### The two controls

Every measurement is relative to two reference runs.

**`control_sequential` — the floor.** Train on the stream with *no* mechanism at
all. This is what forgetting costs you. Currently **AA 0.0902**.

**`control_joint` — the ceiling.** Train on all twelve tasks *simultaneously*,
shuffled together. No forgetting is possible because nothing is sequential. This
is the best any mechanism could hope to match. Currently **AA 0.8784**.

**`control_independent`.** A third control: train a *fresh* model per task. Tells
you whether a task is learnable at all, independent of forgetting. Used to catch
broken tasks before spending money.

**Span = ceiling − floor.** Currently **0.7882**. The room a mechanism has to
work in. If the span is tiny, no mechanism can look different from any other and
the benchmark is useless — the report warns below 0.05.

### The metrics

**AA (Average Accuracy).** The headline number: after finishing the whole
stream, how accurate is the model, averaged over every task? Currently
GPM = 0.1632.

**rho (ρ, "recovery ratio").** AA rescaled so it is comparable across
benchmarks:

```
rho = (AA − floor) / (ceiling − floor)
```

- **rho = 0** — no better than doing nothing
- **rho = 1** — as good as joint training, the practical maximum
- **rho < 0** — actively worse than no mechanism at all

GPM is currently at **rho 0.093**. Replay reached **1.004**.

Why rescale? Because raw accuracy depends on the benchmark and the hardware,
while rho is measured against controls from the *same* run, so it stays
comparable.

**FM (Forgetting Measure).** How much was lost: for each task, its best-ever
accuracy minus its final accuracy. High = forgot a lot. The plain control sits
at 0.758.

**LA (Learning Accuracy).** How well each task was learned *at the moment it was
trained*, before later tasks could interfere. This separates two failures that
look identical in AA:

- learned it, then forgot it → LA high, AA low
- never learned it at all → LA low

Reported as a percentage of the control's LA. EWC scores 42%, meaning it retains
by refusing to learn. Roughly, **AA ≈ LA − FM** (we measured the correlation at
0.99).

**BWT (Backward Transfer).** Whether learning later tasks helped or hurt earlier
ones. Negative = hurt, which is the normal case.

**FWT (Forward Transfer).** Whether earlier tasks help you learn later ones
*faster*. Measured as accuracy on a task before ever training on it, above that
task's chance level. **Every mechanism we tested scores about −0.055**: none of
them compound knowledge. They preserve; they do not accelerate.

**Chance baseline.** What you would score by guessing. Essential context: a task
with 2 possible answers gives 50% for free, so "50% accuracy" can mean zero
learning. Every task in our benchmark declares its own chance level.

---

## 4 · The two scenarios

The difference between them changes results more than any mechanism does.

**Task-IL (Task-Incremental Learning).** The model is *told* which task each
input belongs to — like being handed a question with "this is a sorting problem"
written on it. Easier.

**Class-IL (Class-Incremental Learning).** The model gets only the input and must
work out for itself which task it is. This is what a real deployed system faces,
so it is our headline benchmark.

The same mechanism scores very differently: GPM gets **0.227 in Task-IL** and
**0.093 in Class-IL**. Rankings reorder entirely between them, so a Task-IL
result never transfers automatically.

**Task boundary.** A signal saying "a new task starts now." Seven of our fifteen
mechanisms require it. A continuous stream of real user interaction has no such
markers.

**Task ID at inference.** Stronger still — needing to be told *which* task a
question belongs to when answering it. Only XdG requires this, and it is
disqualifying for a deployed system, so the harness refuses to run it in Class-IL
rather than print a meaningless number.

**Decidability.** Whether the answer is determined by the input at all. We
discovered our Class-IL benchmark was **undecidable**: `copy`, `reverse` and
`sort` all received the same eight random symbols and demanded three different
answers. No model can win that. It capped joint training at 0.45 against a
guessing bound of 0.33, and one full sweep measured nothing but guessing. Fixed
by giving each task its own vocabulary.

---

## 5 · Statistics — why one number is never enough

**Seed.** A number that fixes all the randomness in a run: initial weights, data
order, everything. Same seed, same code, same hardware → identical result. A
different seed gives a *different but equally valid* result.

**Why run many seeds.** Because a single run is one sample from a distribution.
Ours vary a lot: the same GPM configuration produced rho values from 0.011 to
0.251 across ten seeds.

**Mean.** The average across seeds.

**sd (standard deviation).** How spread out the results are. GPM's is 0.054 —
which is **larger than most of the improvements we are hunting for**, and that is
the central difficulty of this phase.

**Variance.** sd squared. Same idea, different units.

**CV (coefficient of variation).** sd ÷ mean, as a percentage. Lets you compare
spread between things of different sizes. Our `control_joint` has CV 0.4%
(reliable); `control_sequential` has 30.7% (chaotic).

**SE (standard error).** How uncertain the *mean* is, = sd ÷ √n. Crucially it
shrinks with more seeds: four times the seeds halves the uncertainty. This is
why seed count is a budget decision.

**Δrho ("delta rho").** The difference in rho between two configurations. The
thing we are usually trying to measure.

**t-statistic.** The difference divided by its uncertainty: `t = Δrho / SE`.
Roughly, |t| above ~2 means the difference is probably real; below that it could
easily be luck.

**p-value.** The probability of seeing a difference this large if there were
really no difference at all. Below 0.05 is the conventional bar for "real."

**Significant.** Shorthand for "p below 0.05." Not a synonym for "important" — a
tiny difference can be significant with enough data, and a large one can be
non-significant with too little.

**MDE (Minimum Detectable Effect).** The smallest difference an experiment can
reliably find, given its seed count. Ours:

| seeds | MDE (rho) |
|---|---|
| 3 | 0.063 |
| 5 | 0.049 |
| 10 | 0.035 |
| 20 | 0.024 |

**Running an experiment whose MDE exceeds the effect you expect is wasted
money** — it cannot answer its own question no matter what comes back. This
table is why every stage names a seed count in advance.

**Statistical power.** The chance an experiment finds a real effect. Low power
means real effects get missed.

**Correlation.** Whether two quantities move together. +1 = perfectly together,
0 = unrelated, −1 = perfectly opposite. We tested whether a seed's floor
predicted its GPM score and got **−0.234** — essentially unrelated, which is why
a paired comparison did not help.

**Paired vs unpaired comparison.** Paired compares like with like (each seed
against its own control) and is more powerful *when* the pairs are correlated.
Ours were not, so pairing made things worse.

**Dose–response / monotonic trend.** When a bigger dose gives a bigger effect,
consistently. Stronger evidence than a single comparison — but not proof.
Stage 1 showed a perfect monotonic trend across four settings that **vanished
entirely** when we added five more seeds.

**Outlier.** A single unusually extreme value. One seed scoring 0.251 dragged a
five-seed average from 0.053 to 0.121 and produced a result that did not
replicate.

**Replication.** Re-running to see whether a result holds. Stage 1 is the
project's cautionary example: a clean-looking effect at five seeds reversed sign
at ten.

---

## 6 · How experiments are organised

**Run.** One training of one model with one configuration and one seed.

**Configuration ("config").** One specific setting, e.g.
`gpm[eps_base=0.8]`.

**Preset.** A named bundle of settings — `gpm`, `replay`, `control_joint`.

**Grid / grid point.** A list of values to try for one setting. `eps_base` has
grid `[0.8, 0.9, 0.97]`, so three grid points.

**Sweep.** Running every grid point at every seed and collecting the results.

**Ablation.** Testing many mechanisms under identical conditions to see which
matters. Our Phase 3 was a fifteen-mechanism ablation.

**Selection advantage.** A subtle unfairness: if we report each mechanism's
*best* grid point, a mechanism with four grid points gets four chances and keeps
the luckiest, while a mechanism with one gets one. Worth about half a standard
deviation — the same size as the differences between mechanisms. We found two
mechanisms running with a single grid point against everyone else's three or
four, and fixed it.

**Signature probe.** An independent check that a mechanism did the *internal*
thing its paper describes, not merely that it scored well. Example: EWC claims
to move "important" weights less, so probe B1 measures whether it actually does.
**This caught four bugs that accuracy alone hid**, including a mechanism whose
bug was making it look *better* than the fixed version.

**Inert.** A mechanism that attached without error and then did nothing. Detected
and reported, because otherwise it produces a full table of meaningless numbers.

**Gate.** A cheap check run *before* an expensive one. Two of ours caught broken
benchmarks that would have wasted a full day of compute.

**Pre-registration.** Writing down what you expect, and what would count as being
wrong, *before* running. Prevents reinterpreting a null result as a success after
the fact.

**Sanity check.** A quick verification that something behaves as expected before
trusting a bigger result.

---

## 7 · GPM-specific terms

GPM = **Gradient Projection Memory**, the mechanism we selected.

**Linear layer.** The basic building block: multiply the input by a matrix of
weights. Written `y = Wx`.

**Vector / matrix.** A list of numbers; a grid of numbers. Weights are matrices.

**Subspace.** A set of directions within the space of possible inputs. If old
tasks only ever produced inputs pointing in certain directions, those directions
form a subspace.

**Basis.** A minimal set of directions that describes a subspace — like axes on a
graph. GPM stores one basis per layer, written `M`.

**Orthogonal.** At right angles; independent. Moving along one direction does not
affect the other.

**Orthonormal.** Orthogonal *and* each direction has length 1. Makes the maths
clean: `MᵀM = I`.

**Projection.** Splitting a quantity into a part inside a subspace and a part
outside it. GPM keeps only the outside part of each gradient:

```
g  ←  g − (g M) Mᵀ
```

This makes the weight update *exactly* orthogonal to everything the old tasks
used, so their behaviour is mathematically unchanged. That exactness is what
distinguishes GPM from a penalty-based method like EWC, which merely discourages
movement.

**SVD (Singular Value Decomposition).** A standard procedure that finds the
principal directions in a pile of data and how important each one is.

**Eigenvector / eigenvalue, singular value.** The directions SVD finds, and the
amount of variation along each.

**Variance explained.** How much of the data's variation a set of directions
accounts for. GPM keeps enough directions to explain a fraction `eps` of it.

**Rank (k).** How many directions are kept. The core quantity.

**`eps_base`.** The starting fraction of variance to retain — 0.8 means keep
enough directions to explain 80%.

**`eps_growth`.** How `eps` changes per task. It was **+0.005**, meaning later
tasks kept *more* — consuming more directions exactly when least affordable.
Stage 1 tested reversing this.

**`max_bases_frac`.** A cap: never freeze more than this fraction of a layer's
directions. Set to 0.75.

**Saturation.** How full the basis is. **This is GPM's binding failure.** Every
task consumes directions and none are ever returned, so the model eventually has
nowhere left to learn. Measured:

| basis used | rho |
|---|---|
| 59.5% of cap | **+0.073** |
| 97.8% | +0.058 |
| 98.2% | **−0.047** |
| 100% | ~0.001 |

**Probe C2.** GPM's signature check, reporting the fraction of directions
consumed. Originally it only fired at the cap — so at 98% saturation, with the
mechanism actively harmful, it still reported healthy. Stage 0 recalibrated it to
warn at 85% of the cap.

---

## 8 · The other fourteen mechanisms, in one line each

| mechanism | idea |
|---|---|
| **replay** | Keep a small buffer of old examples and mix them into new training. Won: rho 1.004 |
| **DER** | Replay, but also store the model's old *outputs* and match them |
| **EWC** | Estimate which weights matter for old tasks; penalise changing those |
| **SI** | Like EWC, but computes importance during training instead of afterwards |
| **LwF** | Keep a copy of the old model; require the new one to agree with it |
| **GPM** | Project gradients away from directions old tasks used (our pick) |
| **LoRA** | Add a small separate set of weights per task; freeze the rest |
| **O-LoRA** | LoRA with the per-task additions forced to be mutually orthogonal |
| **L2P** | Learn a pool of "prompts" and pick the right one per input |
| **kWTA** | Let only the top-k units activate, so tasks use different ones |
| **XdG** | Switch off a random subset of units per task |
| **CBP** | Continually reinitialise the least useful units to restore plasticity |
| **Shrink-Perturb** | Periodically shrink all weights and add noise |
| **Memory layer** | Add a large addressable memory to the network |
| **Sparse update** | Update only the few parameters most specific to the current batch |

**Fisher information.** EWC's measure of "how much does this weight matter."
Ours initially collapsed to 1e-11 — mathematically correct, practically useless,
because once a task is solved its gradients vanish.

---

## 9 · Infrastructure

**GPU.** The processor that makes training fast. We rent them by the hour.

**vast.ai.** The marketplace we rent from — roughly $0.06–0.36/hour.

**CUDA.** NVIDIA's GPU software layer.

**Compute capability / `sm_75`, `sm_86`.** A GPU's generation code. Our software
supports `sm_75` and newer, so older cards (GTX 10-series and earlier) fail —
worth checking before renting.

**tmux.** Keeps a job running after you disconnect. Essential for overnight runs.

**`result.json`.** One file per run holding every metric, cost, signature and the
hardware it ran on. All analysis reads these.

**`runs_A` / `runs_B` etc.** Directories holding sweeps. Kept separate so a
corrected re-run does not silently reuse old results.

**Mixed hardware.** Running some configurations on one GPU and others on a
different one. Since rho is measured against controls, mixing invalidates the
comparison. The report detects and warns.

---

## 10 · Where we are

| | rho | meaning |
|---|---|---|
| doing nothing | 0.000 | the floor, by definition |
| **GPM today (Class-IL)** | **0.093 ± 0.054** | our starting point |
| GPM (Task-IL, easier) | 0.227 ± 0.050 | same mechanism, easier scenario |
| best rehearsal-free rival (EWC) | 0.104 | and it destroys 58% of plasticity |
| replay | 1.004 | works, but stores raw data forever |
| joint training | 1.000 | the ceiling, by definition |

The goal of the GPM development phase is to move that 0.093 up without storing
raw data and without sacrificing plasticity.
