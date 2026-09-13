# Success metrics — what we measure, where we stand, what we are aiming for

Written 2026-09-13, alongside the audit (`AUDIT.md`) and the `gpm_v2`
implementation. Every metric is defined in `clms/eval/metrics.py` or in the
mechanism's own probe; every "current" number below has a file it came from.

The one-line version: **the headline number is rho on the 12-task Class-IL
benchmark, and GPM's pre-audit best is 0.073.** Everything else is either a
guard-rail (it must not get worse while rho improves) or a diagnostic (it
tells us *why* rho moved).

---

## 1 · The headline: rho (recovery ratio)

```
rho = (AA_mechanism − AA_sequential_control) / (AA_joint_control − AA_sequential_control)
```

- **0** = no better than doing nothing. **1** = as good as training on all
  tasks at once, which is the best any mechanism could do. **Negative** =
  actively harmful.
- It is normalised against controls from the *same* sweep, so it is
  comparable across hardware, model sizes and task counts where raw AA is not.
- Its noise floor is set by seed count, not by the mechanism. Minimum
  detectable difference: 0.063 at 3 seeds, 0.049 at 5, 0.035 at 10, 0.024
  at 20. A claim smaller than the MDE of the run that made it is not a result.

**Where we stand (all pre-audit, i.e. with defects D1–D7 in place):**

| benchmark | tasks | model | GPM best rho | seeds | source |
|---|---|---|---|---|---|
| **Class-IL** | **12** | small 23.7M | **0.073 ± 0.039** (eps 0.8) | 5 | `runs_A`, report_A |
| Class-IL | 12 | nano 0.9M | 0.026 ± 0.014 (eps 0.8) | 5 | `runs_B`, report_B |
| Class-IL | 12 | small | 0.112 (eps_growth −0.005) | 5 | `runs_s1` (Stage 1; reversed at 10 seeds) |
| Task-IL | 12 | small | 0.087 (baseline) / 0.111 (aging 0.25) | 5 | `runs_s2` (Stage 2) |
| Task-IL | 5 | small | 0.287 ± 0.061 (strength 1.0) | 5 | `runs_s3` (Stage 3) |
| Task-IL | 5 | small | 0.227 ± 0.050 | 5 | v5 sweep, `RESULTS.md` |

For scale: EWC's best on the same Class-IL benchmark is 0.104 (at 42 %
plasticity, so not a real competitor), and replay reaches **1.004** on Task-IL
5-task. Replay is the number to beat; it is excluded from the target only
because it stores raw user data.

**Post-audit, diagnostic scale only** (`AUDIT.md` §5; tiny 4.5M, first six
tasks, 300 steps, 2 seeds; floor 0.100, joint ceiling 0.837):

| variant | AA | rho |
|---|---|---|
| gpm as shipped, eps 0.8 | 0.247 | 0.20 |
| gpm as shipped, eps 0.9 | 0.258 | 0.21 |
| gpm with all fixes, eps 0.9 | 0.264 | 0.22 |
| **gpm with all fixes, eps 0.97** | **0.542** | **0.60** (seeds: 0.49, 0.71) |

Two seeds and a small model: this is the reason to run the real sweep, not a
result in itself.

**Post-audit, on the actual 12-task Class-IL benchmark** (`AUDIT.md` §5b; tiny
4.5M, 300 steps per task, harness fixes on, one seed; floor 0.113, joint
ceiling 0.856; `audit_logs/tiny12_class_il.log`):

| variant | AA | LA % | FM | occupancy | C5 | rho |
|---|---|---|---|---|---|---|
| gpm as shipped, eps 0.8 | 0.079 | 98 % | 0.821 | 0.63 | — | **−0.046** |
| gpm_v2, eps 0.97 | 0.340 | 103 % | 0.587 | 0.57 | 3.8e-5 | **0.305** |
| gpm_v2, eps 0.99 | 0.458 | 90 % | 0.332 | 0.76 | 4.3e-5 | **0.465** |

One seed at 4.5M parameters. It clears the first two tiers below and sits at
the guard-rails (LA 90 %, occupancy 0.76) at eps 0.99, which is exactly the
trade the paper describes. The 5-seed `small` sweep decides whether it holds.

**Targets, in tiers, all on the 12-task Class-IL benchmark at `small`, 5 seeds:**

| tier | rho | what it would mean |
|---|---|---|
| floor to clear | **≥ 0.15** | double the pre-audit best; exceeds the 5-seed MDE from 0.073 by 1.5× |
| target | **≥ 0.30** | the fixes carry to 12 tasks at real scale; GPM becomes the rehearsal-free mechanism to build the memory stack on |
| stretch | **≥ 0.50** | within reach of what a small replay buffer does; the 15 % residual story is closed |
| end state (phase 5) | **≈ 1.0 without stored data** | the project's actual requirement; no rehearsal-free method has ever reached it |

---

## 2 · The guard-rails — must not get worse

### LA (learning accuracy) and LA %

Mean of the accuracy-matrix diagonal: how well each task was learned the
moment it was trained, before anything could interfere. Reported as a
percentage of the sequential control's LA. Below ~90 % the mechanism is
buying retention by refusing to learn, which is intransigence, not continual
learning. EWC at λ = 10 000 has FM 0.000 and LA 19 %: a frozen model.

- **Current:** GPM 96 % (eps 0.8) and 91 % (eps 0.9) on Class-IL 12; 100 % at
  every strength in Stage 3. Corrected GPM on the diagnostic: 0.879 vs the
  control's 0.851, i.e. **103 %**.
- **Target:** **≥ 90 %** at every eps in the sweep. If the paper-range eps
  (0.99) drives LA under 90 % on twelve tasks, that is GPM's genuine capacity
  limit and the number to report, not to tune away.
- **Caveat fixed by the harness change:** LA was confounded by the global LR
  schedule (D6). Post-fix LA values are not comparable with pre-fix ones.

### FM (forgetting measure)

Per task, best-ever accuracy minus final accuracy, averaged over old tasks.
Because AA ≈ LA − FM (measured correlation 0.987–0.999), FM is where rho
gains have to come from once LA is held.

- **Current:** control 0.758; GPM 0.664 (eps 0.8), 0.634 (eps 0.9) on
  Class-IL 12. Corrected GPM on the diagnostic: 0.404 vs control 0.902.
- **Target:** **≤ 0.40** at `small`, 12 tasks (roughly halving the control),
  with LA held. Stretch: ≤ 0.20.

### C5 — weight-step leak (new, `gpm_v2` only)

`||ΔW·M||_F / ||ΔW||_F` measured on the *actual* optimizer step, every 25
steps. This is the guarantee itself, checked rather than assumed. AdamW
without the correction leaves 9–17 % of the step inside the protected
subspace (0.217 in the smoke run).

- **Current:** 2.6e-5 after correction at occupancy 0.59 (CPU and MPS).
  Probe passes below 0.01.
- **Target:** stays **< 1e-3** in every run of every sweep. If it ever rises,
  something upstream (a new optimizer, a new parameter group, a numerical
  routine) has broken the mechanism and the rho of that run is not GPM's.
- **Track record:** on its first 12-task run it read 0.93 and exposed D7, a
  Gram-matrix SVD returning non-orthogonal vectors once a basis was more than
  half full — a defect the shipped `gpm` shares and that no accuracy metric
  would have shown.

### C2 — basis occupancy

Mean over projected layers of (directions frozen) / (layer width). GPM's
capacity meter. The paper's known limit: dissimilar tasks fill it fast and
"after which no new learning will be possible".

- **Current:** as-shipped GPM reached 0.518–0.744 of a 0.75 cap after 12
  tasks (i.e. 69–99 % of cap). Corrected rule: 0.39 at eps 0.97 after 6 tasks
  with no cap.
- **Target:** **< 0.85** after 12 tasks at the eps that wins on rho. Above
  that, LA will show the cost and the honest report is "GPM's capacity ends
  around task N on this stream".
- **Aim:** occupancy per task should *fall* along the stream as tasks share
  representation with earlier ones (the paper rule adds k = 0 for a covered
  task). If it stays linear, tasks are not sharing and GPM's premise is weak
  on this stream.

### Memory (MB of persistent state)

The mechanism's stored state: bases for GPM, Fisher for EWC, buffer for
replay. Reported per run in `costs`, and as rho/MB in the report.

- **Current:** GPM 54.6 MB (eps 0.8) to 84.6 MB (eps 0.97) at `small`.
  Sharing q/k/v and gate/up bases should cut ~43 %; the paper rule should cut
  more by adding fewer directions.
- **Target:** **≤ 55 MB at `small` for the winning eps**, i.e. no worse than
  the pre-audit configuration while scoring higher. At 7B this extrapolates
  to ~16–25 GB pre-audit; the sharing and rank fixes are the first two of the
  compressions the scale-up will need.

### Wall time

- **Current:** as-shipped GPM 2.1–2.4× the control (647–753 s vs 305 s per
  run), most of it moving bases host→device every step.
- **Target:** **≤ 1.3× control** once bases are cached on device. The step
  correction adds one clone and one matmul per layer per step, which should
  be within that.

---

## 3 · Diagnostics — explain the number, do not target it

| metric | what it says | current | what we want to see |
|---|---|---|---|
| **BWT** (backward transfer) | change in old-task accuracy caused by later learning; ≈ −FM here | −0.63 to −0.77 | tracks FM upward; positive BWT would be transfer, which no mechanism shows |
| **FWT** (forward transfer) | accuracy on a task *before* training on it, above chance | −0.047 to −0.058 for every mechanism | any positive value would be new. Not a target for GPM; projection preserves, it does not compound |
| **AIA** (average incremental accuracy) | AA averaged over the whole stream, not just the end | reported per run | should rise with rho; a divergence means early tasks are retained and late ones are not, or the reverse |
| **span** (ceiling − floor) | room the benchmark gives mechanisms to differ | 0.794 (Class-IL 12, small) | stays > 0.5 after the harness fixes; the report warns below 0.05 |
| **A2 CKA / D2 logit drift** | how much task-0 representations and outputs moved | CKA 0.30–0.69, KL 96–281 under GPM | should move toward CKA 1.0 / KL 0 on protected tasks as rho rises; if rho rises and these do not, the gain is not coming from protection |
| **seed sd of rho** | noise floor | 0.039 (GPM) vs 0.007 (eps 0.97 saturated) | ≤ 0.05 at 5 seeds; the sequential control's own CV is 31 %, so this is mostly benchmark noise |

---

## 4 · What "done" looks like for this phase

1. `gpm_v2` sweep on Class-IL 12, `small`, 5 seeds, eps ∈ {0.95, 0.97, 0.99},
   with the harness fixes, alongside fresh controls. **Pass:** best rho ≥
   0.15 with LA ≥ 90 %, C5 < 1e-3 in every run, occupancy < 0.85.
2. Attribution: one ablation sweep turning the fixes off one at a time
   (`project_step`, `protect_readout`, `freeze_norms`, and eps back at 0.8),
   3 seeds each. Tells us which fix carries the effect and which are free.
3. Only after 1 and 2: the EQUATION.md directions (damped spectrum first),
   compared against a GPM that works.
4. A design note on the readout before Phase 5. The tied embedding is the
   same single matrix in a real LM, and D5 will not go away at scale.
