# Continual Learning Mechanism Stack

Continual-learning mechanisms that plug into an OLMo 2 style transformer through
a single contract, so comparing them is a config change rather than a rewrite.

**Repository:** https://github.com/Sakib323/Continual-Learning-Transformer-Architecture-

```
olmo2_cl/                                  OLMo 2 architecture, from scratch
Continual Learning Mechanism Stack (CLMS)/
└── clms/                                  the mechanism library
configs/                                   one YAML per experimental stack
scripts/sweep.py                           ablation runner + ranking report
train.py                                   entry point
design/                                    planning documents
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
python train.py --list
```

The display folder name contains spaces and parentheses, so it can never be a
Python package name. The importable package lives inside it and `pyproject.toml`
maps it; `import clms` works from anywhere after `pip install -e .`.

## Running

```bash
python train.py --preset control_sequential     # the forgetting floor
python train.py --preset control_joint          # the ceiling
python train.py --set mech.ewc.enabled=true --set mech.ewc.lam=5000
python train.py --preset stack_rsp --seed 1

python scripts/pilot.py --tune                  # what will this cost?
python scripts/sweep.py --preset ablation --tune --seeds 0,1,2
```

### Tune before you rank

A ranking at one arbitrary hyperparameter is not a fair test. The first nano
ablation ran EWC at `lam=1000`, where its own signature probe reported the
penalty was not biting at all — that result says nothing about EWC. `--tune`
sweeps each mechanism over the parameter that most controls its strength
(`clms.config.TUNING_GRIDS`) and scores it at its best setting.

```bash
python scripts/sweep.py --preset ablation --tune --seeds 0,1,2 --dry-run
# 47 configurations x 3 seeds = 141 runs

python scripts/pilot.py --tune --set model.size_preset=small
# times one segment on this machine, extrapolates to GPU-hours and dollars
```

Run `pilot.py` on the instance type you intend to rent — it is the cheapest
de-risking available, and a plan that turns out to be 300 GPU-hours is much
better discovered before the rental than three hours into it.

Any config key can be overridden with `--set a.b.c=value`; `mech.` is an alias
for `mechanisms.`. Unknown keys are rejected with the valid options listed, so a
typo fails in a second rather than after an hour of training.

### Three run modes

`--mode sequential` trains task after task — the continual setting, and the
floor when nothing is enabled. `joint` trains on all tasks mixed, giving the
ceiling. `independent` trains a fresh model per task, a second ceiling that
isolates whether sharing helps at all.

You need all three. Turning every flag off gives only a floor, and a floor alone
lets you say a mechanism beats doing nothing — true of almost everything, and it
ranks nothing. With both bounds, mechanisms become comparable on one number:

```
        AA(mechanism) − AA(sequential)
rho = ─────────────────────────────────     0 = no better than nothing
        AA(joint)     − AA(sequential)      1 = matches joint training
```

## The contract

Five surfaces. Most mechanisms never touch the model at all, which is why they
can be a library rather than a fork.

| | Surface | Hook |
|---|---|---|
| **A** | architecture | `build_mlp`, `build_attention`, `observe` |
| **L** | loss | `compute_loss` |
| **O** | optimizer | `before_step`, `after_step` |
| **S** | state | buffers; no model contact |
| **D** | data | `on_batch` |

Adding a mechanism is one file in `clms/mechanisms/` with an `@register`
decorator. CLI flags, config schema and presets all follow from the registry —
there is no argument parser to edit.

### Two mandatory methods

```python
def self_test(self, model, batch, ctx) -> tuple[bool, str]:
def signature(self, model, ctx) -> SignatureCheck | None:
```

`self_test` runs on step 0 of every run and aborts if a mechanism is inert. A
mechanism that runs but does nothing looks exactly like one that legitimately
doesn't help, and that failure is invisible without an assertion — it is the
most dangerous bug class in this project. Writing these six mechanisms surfaced
three real bugs this way before any of them reached a training run.

`signature` measures the internal quantity the mechanism's paper claims should
move — EWC's displacement ratio on high-Fisher parameters, k-WTA's active
fraction, continual backprop's effective rank. A FAIL means the mechanism is not
doing what it claims, whatever the accuracy says.

### Validation before any GPU time

Conflicts (two mechanisms that cannot co-run), stream capabilities (EWC needs
task boundaries; `xdg` needs task identity, so it is rejected under `class_il`),
and hook ordering are all checked at config-parse time.

## Registered so far

| Mechanism | Surfaces | Family | Paper |
|---|---|---|---|
| `replay` | S·D | F01 | 2004.07211 |
| `der` | S·L | F01 | 2004.07211 |
| `lwf` | L | F02 | 1606.09282 |
| `ewc` | L | F02 | 1904.07734 |
| `si` | L·O | F02 | 1904.07734 |
| `gpm` | O | F03 | 2103.09762 |
| `olora` | A·L | F03 | survey |
| `xdg` | A | F04 | 1904.07734 |
| `kwta` | A | F04 | 1903.11257 |
| `memory_layer` | A | F04 | 2510.15103 |
| `sparse_update` | O | F04 | 2510.15103 |
| `lora` | A·O | F05 | 2510.15103 |
| `l2p` | A·L·O | F05 | 2112.08654 |
| `continual_backprop` | A·O | F12 | 2306.13812 |
| `shrink_perturb` | O | F12 | 2306.13812 |

That is the tier-1 set: 15 mechanisms across 6 families, covering all five
surfaces. Roughly 80 more are catalogued in `design/mechanism-registry.html`,
tiered by build priority.

`--preset ablation` expands to controls plus each of these one at a time.

## The benchmark

Synthetic algorithmic tasks — copy, reverse, sort, modular arithmetic,
key-value recall, induction — rather than TRACE. At 20–50M parameters TRACE's
tasks sit near chance on both sides of training, which collapses the
floor-ceiling gap every comparison depends on. These give exact ground truth,
tunable inter-task overlap, and minutes per run.

Evaluation batches come from a generator seeded independently of training, so
the probe set is frozen across every run and configuration.

Sanity check, `tiny` (4.5M), copy → reverse:

```
control_sequential   AA=0.500   FM=1.000   BWT=-1.000    copy: 100% -> 0%
replay               AA=1.000   FM=0.000   BWT=+0.000    copy: 100% -> 100%
```

Textbook catastrophic forgetting, and a mechanism that removes it.

### Reading the report

Two failure modes are easy to confuse, and `sweep.py` calls both out explicitly.

**Retained but did not learn.** `FM=0.000` with `rho≈0` is not a success — it
means the model froze on task 0 and never acquired task 1. On a forgetting-only
metric that reads as a perfect score. This is intransigence, the opposite
failure, and it is why the ceiling exists.

**Signature failures.** `B1:FAIL` next to a flat score means the mechanism did
not do the internal thing its paper claims, so the number reflects a
misconfiguration rather than the method. Fix the setting and re-run before
concluding anything.

#### A worked example of why this matters

The first tuned sweep ran EWC at four values of lambda. All four scored rho ~ 0,
and all four reported `B1:FAIL` — high-Fisher parameters were moving as much as
low-Fisher ones. Accuracy alone would have concluded "EWC does not help on this
benchmark." The signature said something more specific: EWC was never applying
pressure at all.

Two causes, both real:

1. The Fisher was the *empirical* one — squared gradients of the loss on
   ground-truth labels. Once a task is solved those gradients vanish, so the
   Fisher collapses and no lambda can recover it. Measured at 1e-11.
2. Switching to the *model* Fisher (sampling from the network's own predictive
   distribution) helped by ~1000x but still landed at ~1e-9, because a network
   that has memorised a deterministic task has near-zero likelihood curvature
   in the directions it uses.

The fix is normalising each task's Fisher to unit mean, which is what makes
lambda scale-free. `B1` passes afterwards, and lambda finally controls something.

Both knobs are exposed (`fisher_type`, `normalize`) so the comparison is
reproducible rather than buried in a commit.

## Model

`olmo2_cl` is an architecture-faithful OLMo 2 reimplementation: reordered norm
(RMSNorm on sublayer *outputs*, inside the residual), QK-Norm before RoPE,
SwiGLU, GQA, no biases. Self-contained, so there is no `transformers` internals
dependency to break. See `olmo2_cl/VENDORED_FROM.md` for reference sources and
what to check when rebasing.

| preset | params | use |
|---|---|---|
| `nano` | ~0.9M | CPU smoke tests |
| `tiny` | ~4.5M | fast iteration |
| `small` | ~24M | default sweep size |
| `base` | ~46M | upper end of the sweep ladder |
| `mid` | ~200M | rank confirmation |

Trained from scratch, never from a checkpoint. Several mechanisms only exist if
present at initialization — memory layers, k-WTA, XdG, MoE routing — so grafting
them onto a finished model would measure grafting damage instead of the
mechanism.

## Working on vast.ai

One scripted path, which ends by telling you what your sweep will cost on the
card you just rented:

```bash
git clone https://github.com/Sakib323/Continual-Learning-Transformer-Architecture-.git
cd Continual-Learning-Transformer-Architecture-
bash scripts/vast_setup.sh
```

`vast_setup.sh` creates the venv, installs the package editable, **fails loudly
if CUDA is missing** rather than silently billing you for a CPU run, executes the
contract tests and a nano smoke run, then reports pilot timings.

Manual equivalent:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .          # makes `import clms` / `import olmo2_cl` work anywhere
python train.py --list
```

VS Code: Remote-SSH to the instance, open the repo folder, select `.venv` as the
interpreter. Add the instance to `~/.ssh/config` so it is one click.

### Using CLMS as a module

`pip install -e .` puts both packages on the path, so the library is importable
from any script or notebook on the instance — you are not restricted to
`train.py`:

```python
from clms import Composer, RunContext, registry
from clms.data import TaskStream, build_task_sequence
from clms.eval import AccuracyMatrix, ProbeRunner, sequence_accuracy
from olmo2_cl import Olmo2Config, build_model

ctx   = RunContext(num_tasks=6, device="cuda", seed=0)
comp  = Composer.from_config({"replay": {"enabled": True, "ratio": 0.25}}, ctx)

mcfg  = Olmo2Config(size_preset="small", vocab_size=128)
comp.set_model_config(mcfg)              # surface-A mechanisms size themselves
model = build_model(mcfg, injector=comp).to("cuda")
comp.setup(model, mcfg)

# then drive your own loop through the hooks:
#   comp.on_batch / comp.compute_loss / comp.before_step / comp.after_step
#   comp.on_task_start / comp.on_task_end
```

The order matters in one place only: `set_model_config` must run *before*
`build_model`, because architecture-surface mechanisms decide their shape from
the model config. `train.py` is a reference driver, not a requirement.

Three things specific to rented instances:

**Instances get interrupted.** Every task boundary writes `checkpoint.pt`
containing model, optimizer *and mechanism* state. A resumed run whose Fisher
matrix or replay buffer was lost is silently a different experiment — that is
why `state_dict()` is on the `Mechanism` contract.

**Disk fills faster than you expect.** A checkpoint is ~280MB at 24M params, and
a 141-run tuned sweep would leave ~40GB behind — enough to fill an instance
mid-sweep. So the checkpoint is deleted once `result.json` lands (`run.keep_checkpoint`
to override), leaving ~4KB per run. `run.skip_completed` also makes re-running
the same sweep command a no-op for finished configurations, so an interrupted
sweep resumes at the sweep level, not just within a run.

**Pin the GPU type for a whole sweep.** Different cards mean different numerics
and kernel paths, and that difference shows up in results looking exactly like a
mechanism effect. Every `result.json` records the card, torch version and
device; treat a mid-sweep hardware change as grounds for re-running.

**Measure before you commit.** Run one complete config end to end on the
instance type you intend to rent and time it, then multiply by
(mechanisms × seeds + controls). That pilot is the cheapest de-risking here.

## Statistical hygiene

Continual-learning results are high variance — Dohare et al. ran 30 seeds, and
rankings routinely flip between them. Minimum 3 seeds; `scripts/sweep.py`
reports mean ± sd and prints which mechanisms are indistinguishable at the seed
count you ran. Never rank two whose intervals overlap.

Cost is a measured output, not a footnote. A mechanism that wins by buffering
10% of the training data is not comparable to one that adds nothing, so every
run logs added parameters, buffer bytes, and wall time per mechanism.

## Documents

| | |
|---|---|
| `design/mechanism-registry.html` | all ~95 mechanisms, tagged by surface and tier |
| `design/build-workflow.html` | phases, risks, repo layout |
| `design/diagnostics-suite.html` | the 17 internal probes and what each family claims |
| `design/memory-hierarchy-plan.html` | the four-level architecture proposal |
