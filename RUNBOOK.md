# Runbook — Phase 3 ablation on vast.ai

The order below is deliberate: each step is cheap and can invalidate the next
one. The gate in step 5 exists because a broken benchmark and a broken mechanism
produce the same table, and you would rather find that out for a few cents than
after 141 runs.

---

## 1 · Push

```bash
git add -A
git commit -m "Phase 3: tuning grids, pilot timing, checkpoint policy"
git push
```

Nothing runs on the instance that is not committed. The sweep clones the repo.

---

## 2 · Rent the instance

| | Choose | Why |
|---|---|---|
| GPU | **1x RTX 4090** | 24M params does not need an A100. A 4090 at $0.30-0.55/hr is the right price-performance point here, and the sweep is single-GPU by design. |
| Disk | **≥ 30 GB** | ~5GB image + torch, the rest headroom. Checkpoints are deleted on success, so results are ~4KB per run. |
| Image | a **PyTorch/CUDA template** | Skips a ~3GB torch download on every rental. `pip install -e .` then sees torch already satisfied. |
| Type | **on-demand** for the first run | Interruptible is cheaper and resume handles it, but debug on stable hardware first. Switch to interruptible once the sweep is known-good. |

**Record the exact GPU model.** Different cards mean different numerics and
kernel paths, and that difference shows up in results looking exactly like a
mechanism effect. Every `result.json` stores the device; a mid-sweep hardware
change is grounds for re-running the affected configurations.

---

## 3 · Connect

```bash
ssh -p <PORT> root@<HOST>
```

Add it to `~/.ssh/config` so VS Code Remote-SSH is one click:

```
Host vast-clms
    HostName <HOST>
    Port <PORT>
    User root
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
```

VS Code: **Remote-SSH: Connect to Host** → `vast-clms` → open the repo folder →
select `.venv/bin/python` as the interpreter.

---

## 4 · Provision

```bash
git clone https://github.com/Sakib323/Continual-Learning-Transformer-Architecture-.git
cd Continual-Learning-Transformer-Architecture-
bash scripts/vast_setup.sh
```

`vast_setup.sh` creates the venv, installs editable, **fails loudly if CUDA is
missing** rather than silently billing you for a CPU run, executes the 118
contract tests and a nano smoke run, then reports pilot timings for that card.

Expect the pilot to report meaningfully faster than the MPS baseline of 26
GPU-hours. Use its number, not that one.

---

## 5 · The gate — controls only

**Do not skip this.** Roughly 6 runs, a few minutes, and it decides whether the
other 141 are worth paying for.

```bash
source .venv/bin/activate
python scripts/sweep.py --preset control_sequential --seeds 0,1,2 \
    --set model.size_preset=small
```

That schedules both controls automatically. Read the header:

```
floor (sequential)      0.xxxx
ceiling (joint)         0.xxxx   span=+0.xxxx
```

| What you see | What it means | Do |
|---|---|---|
| **span > 0.25** | Healthy. Forgetting is large and measurable. | Proceed to step 6. |
| **span < 0.05** | The report warns explicitly. Either the tasks are too easy (no forgetting to fix) or too hard (nothing learned). Every mechanism will look identical. | Stop. Fix the benchmark first. |
| **ceiling well below 1.0** | Some task is not being learned even jointly. | Check per-task numbers in `result.json` → `matrix`, drop or replace the unlearnable task, or raise `steps_per_task`. |

A useful diagnostic when the ceiling looks wrong:

```bash
python train.py --preset control_independent --set model.size_preset=small
```

That trains a fresh model per task, so its per-task accuracies are the clean
learnability ceiling — it tells you *which* task is the problem.

### What this gate already caught

Run before any rental, fresh model per task:

| task | ceiling | chance | verdict |
|---|---|---|---|
| `copy` | 1.00 | ~0 | learned |
| `reverse` | 1.00 | ~0 | learned |
| `sort` | 0.83 | ~0 | learned |
| `modadd23` | 1.00 | 0.043 | learned |
| `induction6` | 0.48 | 0.200 | learning — 2.4x chance |
| `kvrecall` | 0.20 | 0.250 | **at chance** |
| `kvrecall2` | 0.50 | 0.500 | **at chance** |

Two things worth internalising here.

**Always compare against chance, not against zero.** `kvrecall2` scoring 0.50
looks like "half right" and is in fact zero learning: with two pairs, guessing
between the two values present in the prompt gives exactly 0.50. Every task now
declares a `chance` baseline and the tests enforce it.

**`kvrecall` is excluded at every difficulty.** It scored at its chance baseline
at 4 pairs and at 2, at 4.5M and at 24M. Key-value retrieval needs an induction
head that does not form at this scale, and a task scoring at chance contributes
only noise to every mechanism's rho.

Separately, `induction` was *ill-posed*: the cue symbol was not guaranteed
unique, so 11.4% of sequences had an undecidable answer regardless of model
quality. Fixed, and covered by a test.

The default sequence is now the five tasks that clear chance:

```
copy, reverse, sort, modadd23, induction6
```

If `induction6` fails the gate on your hardware, fall back to the four with a
decisive ceiling:

```bash
--set stream.tasks=copy,reverse,sort,modadd23
```

---

## 6 · The ablation

```bash
tmux new -s sweep
source .venv/bin/activate

python scripts/sweep.py --preset ablation --tune --seeds 0,1,2 \
    --set model.size_preset=small \
    2>&1 | tee sweep.log
```

Detach with `Ctrl-B D`, reattach with `tmux attach -t sweep`. **Use tmux.** An
SSH drop kills a foreground process, and vast.ai connections do drop.

Check the plan before committing:

```bash
python scripts/sweep.py --preset ablation --tune --seeds 0,1,2 --dry-run
# 47 configurations x 3 seeds = 141 runs
```

If the pilot says that is more hours than you want:

- drop to `--seeds 0,1` — but never to one seed; CL rankings flip between seeds
- `--set model.size_preset=tiny` — rankings survive scaling down further than
  absolute numbers do
- `--set stream.steps_per_task=250`
- run without `--tune` first to find the promising few, then tune only those

---

## 7 · Re-running after a code change — read this first

`run.skip_completed` skips any configuration whose `result.json` already exists.
That is what makes an interrupted sweep resumable, and it is a **trap** when you
re-run after fixing a bug: the old results are still there, so the sweep skips
everything and reports the numbers you were trying to replace.

Point a corrected re-run at a fresh directory:

```bash
python scripts/sweep.py --preset ablation --tune --seeds 0,1,2 \
    --set model.size_preset=small \
    --set run.out_dir=runs_v2
```

Keeping the old directory rather than deleting it also lets you diff the two
sweeps, which is the only way to see what a fix actually changed.

---

## 8 · If the instance dies

Re-run the identical command.

- Finished configurations are skipped (`run.skip_completed`)
- An interrupted run resumes from its last task boundary, **mechanism state
  included** — Fisher matrices, replay buffers, GPM bases, utility counters. A
  resume that lost those would silently be a different experiment.

---

## 9 · Results

```bash
python scripts/sweep.py --report-only            # full ranking
python scripts/sweep.py --report-only --all-points  # every grid point
```

Pull them back — they are small, since checkpoints are already deleted:

```bash
# from your laptop
scp -P <PORT> -r root@<HOST>:~/Continual-Learning-Transformer-Architecture-/runs ./runs
```

### Reading the table

Three things the report calls out explicitly, all of which are easy to misread:

- **`rho`** — 0 means no better than sequential fine-tuning, 1 means matching
  joint training. Negative means actively harmful, which happens.
- **RETAINED BUT DID NOT LEARN** — `FM≈0` with `rho≈0` is *not* a success. The
  model froze on task 0 and never acquired task 1. On a forgetting-only metric
  that reads as a perfect score.
- **SIGNATURE FAILURES** — the mechanism did not do the internal thing its paper
  claims, so its score reflects a misconfiguration rather than the method. Fix
  and re-run before concluding anything. This is not hypothetical: it is how the
  EWC Fisher collapse was found.

Do not rank two mechanisms whose `mean±sd` intervals overlap — the report lists
those pairs for you.

---

## 10 · Then

- **Phase 4** — curated stacks over the winners, plus a small factorial over the
  top 3-4 to catch interactions.
- **Phase 5** — promote the best stack to `base` (46M), then 0.5B. Where the
  small-scale ranking does *not* hold, that discrepancy is itself a result.

---

## Cost discipline

- `nvidia-smi` while the sweep runs — if utilisation is low the bottleneck is
  data or CPU, and you are paying for an idle GPU.
- Destroy the instance when the sweep finishes. vast.ai bills while it exists,
  running or not.
- Copy `runs/` off before destroying. It is only a few MB.
