# GPM implementation audit — 2026-09-13

**Verdict.** The "stop improving GPM" conclusion in `GPM_PROGRESS_REPORT.md`
is premature. The mechanism as implemented does not deliver the guarantee the
documents describe, and the harness measures it in a way that both flatters
and confounds it. Six concrete defects were found, five of them reproduced by
running the code locally on CPU. Three of the four "ruled-out explanations" in
Part 8 of the progress report were ruled out with instruments that inherit the
same defects.

Everything below was produced by `scripts/gpm_diagnostics.py` (sections A–G).
Re-run it to reproduce any number in this file.

---

## 1 · The six defects

### D1 · The weight step is not orthogonal to the basis (AdamW)

The docs claim `dW·M = 0` "exactly, algebraically identical". That holds for
the **gradient** after `before_step`. The **weight update** is then produced by
AdamW, whose per-element division by `sqrt(v)` does not commute with the
projection: `(m ⊘ sqrt(v))·M ≠ 0` even when `m·M = 0`. The GPM paper trains
with plain SGD (Section 6, "Training Details"), where the two coincide.

Measured, `||ΔW·M||_F / ||ΔW||_F` per optimizer step, tiny model, basis built
after `copy`:

| optimizer | during task 2 | during task 3 |
|---|---|---|
| AdamW (as in `train.py`) | **0.091** mean, 0.148 max | **0.171** mean, 0.288 max |
| Adam, weight_decay=0 | 0.091 | 0.176 |
| SGD + momentum | 0.006 | 0.003 |

So 9–17 % of every step (by norm) lands inside the "protected" subspace, and
weight decay is not the cause. Projecting the *step* after `optimizer.step()`
(restore `W ← W_pre + ΔW(I − MMᵀ)`) removes it at the cost of one extra matmul
per layer; measured FM 0.738 → 0.648 on the 3-task probe.

This also re-reads Stage 3. `strength = 1.0` was never exact under AdamW, so
"softening makes it monotonically worse" is evidence that the mechanism is
*under*-protecting, not that the current projection is the ceiling.

### D2 · Rank selection thresholds the residual, not the total — 3.6× over-consumption

`projection.py::_extend_basis`:

```python
R = R - (R @ existing) @ existing.T          # residual
csum = cumsum(S) / sum(S)                     # fraction of the RESIDUAL spectrum
k = int((csum < eps).sum()) + 1               # always >= 1
```

The paper's criterion (Eq. 9) is on the **total** energy, with the part already
captured by `M` counted:

```
||R_proj||²_F + ||R̂_k||²_F  >=  eps · ||R||²_F        k may be 0
```

The difference is not cosmetic. If the existing basis already explains 95 % of
the new task's activations, the paper adds nothing; this code still adds enough
directions to explain `eps` of the remaining 5 %, every task, so consumption is
geometric in the free space. Measured on the first six tasks of the long
stream (tiny, eps = 0.9):

| after task | new dirs, current rule | new dirs, paper rule | cumulative occupancy |
|---|---|---|---|
| copy | 5.0 % of widths | 5.0 % | 5.0 % vs 5.0 % |
| modadd7 | 12.0 % | 4.3 % | 17.0 % vs 9.3 % |
| reverse | 8.3 % | 2.6 % | 25.3 % vs 11.9 % |
| sort | 12.8 % | 3.6 % | 38.1 % vs 15.5 % |
| modadd13 | 18.5 % | 2.7 % | 56.7 % vs 18.2 % |
| induction6 | 14.7 % | 1.7 % | **71.4 % vs 19.9 %** |

After six tasks the current rule is at 95 % of the 0.75 cap; the paper's rule
is at a quarter of it. This single defect explains the "saturation cliff" at
eps = 0.97, the −0.047 rho, the rank-flip reproducibility complaint, and why
Stage 2's eviction bought nothing (evicting junk directions from an
over-full basis does not change what is protected).

It has a second, worse consequence, shown in §4 below: because 0.97 saturated,
every sweep ran at eps 0.8–0.9, and at those values the protected subspace
does not contain the task at all. D2 did not just waste memory; it pushed the
whole programme into the one regime where GPM cannot work.

### D3 · The basis and the Fisher are built on the evaluation set

`train.py` line 356: `ctx.scratch["fisher_batches"] = list(stream.eval_batches(task, 8))`.
`evaluate_all` scores on `stream.eval_batches(task, eval_batches)`. Both use
`Generator().manual_seed(seed + 10_000 + task_id)`, so with `eval_batches=2`
the two eval batches are the first two collection batches. Verified identical
(section A). GPM (`collect_batches=4`) and EWC therefore build their protection
from the exact inputs they are later scored on. The paper collects from random
*training* samples. This inflates every reported retention number for GPM and
EWC by an unknown amount, and it makes the "stale basis — refuted, capture
rises to 0.917" diagnostic circular: capture was measured on data that is in
the basis by construction.

### D4 · Up to 79 % of the collected activation rows are PAD positions

Every sequence is padded to `stream.max_len = 29`; there is no attention mask;
the collection hook records every position. Fraction of PAD rows per task:

| task | PAD rows | answer rows |
|---|---|---|
| modadd7/13/23/31 | **79.3 %** | 6.9 % |
| induction6 / induction8 | 65.5 % / 58.6 % | 6.9 % |
| copy / reverse / sort / sortdesc | 34.5 % | 31.0 % |
| copy12 / reverse12 | 6.9 % | 44.8 % |

For the four modadd tasks, four fifths of the basis is spent protecting the
network's response to post-EOS padding, which contributes nothing to the loss
or the answer. The `eps` budget is then spent on the wrong subspace, and the
"15 % residual" for the positions that matter is larger than the reported
per-layer mean suggests.

### D5 · The readout and the norm gains are unprotected, and they move a lot

`setup()` skips `lm_head` ("tied to the embedding; projecting it is unstable").
With `tie_word_embeddings=True` that single matrix is both the input embedding
and the output classifier. Relative L2 movement during one task, tiny model,
GPM on:

| group | during task 2 | during task 3 |
|---|---|---|
| embed / lm_head | **0.63** | **0.50** |
| norm gains | 0.003 | 0.006 |
| projected linears | 0.15 | 0.19 |

The one parameter GPM leaves free moves three to four times more than the
ones it constrains. This is the class-incremental readout-bias problem in its
purest form: every step on a new task pushes down the logits of every symbol
not in the current answer. Freezing embeddings and norm gains after task 0
took `modadd7` retention through `reverse` from 0.52 to **0.97** (section D).

The paper's setting hides this: it uses a separate per-task head with no
constraint (multi-head), and it freezes normalisation parameters after task 1
(Section 6, "Network Architecture"). Neither is done here.

A principled version instead of freezing: project `lm_head.weight.grad` on the
right by `(I − M_final M_finalᵀ)` using the final-norm output basis (this is
just GPM applied to the layer that was skipped), and zero the embedding rows of
symbols already seen. Both are implemented in `GPMFixed` in the diagnostics.

### D6 · The learning-rate schedule is global, so late tasks barely train

`rewarm_per_task: False` in every sweep. The cosine runs over all
`12 × 400 = 4800` steps, so the LR each task actually sees is:

| task | 0 | 3 | 6 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|
| % of peak LR at start → end | 2→99 | 86→76 | 51→38 | 25→15 | 15→7 | 7→2 | **2→0** |

On the 5-task Task-IL stream task 4 trains at 10 % → 0 %. This is why
`modadd31` and `induction8` sit at 0.06–0.31 on the diagonal in *every*
sequential run while `modadd7/13/23` reach 0.98–1.00: the last tasks are not
harder, they are trained at a dead LR. Consequences: LA is confounded with
position in the stream; late tasks cause almost no forgetting so FM is
dominated by mid-stream tasks; and the `independent` control uses a per-task
schedule while `sequential` does not, so the two ceilings are not comparable.
Continual-learning benchmarks (the GPM paper included) reset the schedule per
task.

---

## 2 · What the "four ruled-out explanations" actually established

| Part 8 claim | status after audit |
|---|---|
| 1 · coverage is 99.7 % — refuted | The missing 0.3 % is the tied embedding/readout, which moves 3–4× more than anything protected. Coverage by parameter count is the wrong measure. |
| 2 · stale basis — refuted | Measured on the eval set, which is inside the basis by construction (D3). Not informative. |
| 3 · plasticity cost — refuted | Plausible, but LA is confounded by D6 and the over-consumption in D2 was masking a plasticity problem that would appear on a longer stream. |
| 4 · rank-threshold noise — refuted as dominant | Correct as stated, but the threshold rule itself is wrong (D2), which is a larger effect than its noise. |

The compounding argument in `EQUATION.md` §4 (`0.849^depth`) is a real
property of per-layer projection, but the 0.849 was measured with D3 and D4 in
place, so its true value on held-out, non-PAD positions is unknown.

---

## 3 · Smaller things noticed, not measured

- **Basis is moved host→device every step for every layer**
  (`M.to(g.device)` in `before_step`, 56 layers, ~50 MB per step). GPM runs
  take 2.1–2.4× the control's wall time (647–753 s vs 305 s). Cache the bases
  on the device once per task boundary.
- **q/k/v share an input and get three separate bases**; so do gate/up. Sharing
  cuts basis memory by ~43 % with no change in behaviour.
- **Adam state carries across task boundaries.** The momentum from task t lies
  in the directions task t+1 will freeze; the first ~20 steps of each task
  move inside the new constraint (visible as the 0.21–0.24 `max` in the SGD
  row of D1). Resetting optimizer state at each boundary is standard and cheap.
- **Momentum from task 0 is unprojected by construction** and decays with
  β₁ = 0.9, so this is a ~20-step effect, not a first-order one.
- **`self_test` advances the mechanism RNG** before training starts. Harmless
  but means the "seeded" collection is offset by one `randperm`.
- **`k` is never 0** in the current rule, so every task adds at least one
  direction per layer even when its representation is already covered.

---

## 4 · Section F — does the unprotected energy carry the answer?

This is the test the whole method rests on. Train `copy` alone (tiny model,
300 steps), build the basis, then evaluate `copy` with a forward pre-hook on
every projected layer that replaces its input `x` by `MMᵀx` — i.e. run the
network on exactly the components GPM promises to preserve and nothing else.
If the guarantee protects what matters, accuracy should survive.

| eps | occupancy after 1 task | k at layer-0 q_proj | acc(copy) | acc(copy) restricted to span(M) |
|---|---|---|---|---|
| 0.80 | 3.0 % | 16 / 256 | 1.00 | **0.00** |
| 0.90 | 5.6 % | 20 / 256 | 1.00 | **0.00** |
| 0.97 | 9.4 % | 23 / 256 | 1.00 | **0.00** |
| 0.99 | 12.5 % | 24 / 256 | 1.00 | **1.00** |

Identical pattern with PAD rows dropped and train-batch collection (3.4 / 5.7 /
9.3 / 11.9 % occupancy; 0.00 / 0.00 / 0.00 / 1.00). The top singular direction
holds only ~10 % of the energy, so this is not a single massive-activation
outlier; the representation is spread, and the last 1–3 % of energy is where
an exact-match answer lives.

**Reading.** At every `eps` the sweeps used (0.8, 0.9, 0.97), the subspace GPM
protects does not contain the task. The `dW·M = 0` guarantee is delivered
(modulo D1) on a subspace the answer is not in. Only at `eps ≈ 0.99` does the
protected subspace carry the function — which is the range the paper uses
(0.95–1.0, Appendix C.5) and the range this project could never reach,
because D2 makes 0.97 saturate the cap. The "15 % residual" in `EQUATION.md`
§3 is therefore not a suggestive per-layer statistic; it is the whole task.

This also explains the accuracy matrices in `runs_A`: `copy` goes 1.00 → 0.00
after a single projected task in every GPM seed, exactly as the control does.
GPM at eps = 0.8 is not partially protecting `copy`; it is not protecting it.

---

## 5 · Section G — current GPM vs corrected GPM, six tasks

Tiny model, first six tasks of the long stream (`copy, modadd7, reverse, sort,
modadd13, induction6`), Class-IL, 300 steps per task, per-task cosine LR,
AdamW as in `train.py`, two seeds. No joint ceiling was run, so this reports
AA / LA / FM rather than rho. `gpm FIXED` is `GPMFixed` from the diagnostics:
paper rank rule (D2), PAD rows masked and train-batch collection (D3, D4),
post-step projection (D1), `lm_head` projected with the final-norm basis plus
seen-symbol embedding rows and norm gains frozen (D5).

| variant | AA | LA | FM | occupancy after 6 tasks |
|---|---|---|---|---|
| control, no mechanism | 0.100 | 0.851 | 0.902 | — |
| gpm current, eps 0.8 | 0.247 | 0.885 | 0.766 | 0.370 |
| gpm current, eps 0.9 | 0.258 | 0.877 | 0.742 | 0.612 |
| gpm fixed, eps 0.9 | 0.264 | 0.875 | 0.734 | 0.207 |
| **gpm fixed, eps 0.97** | **0.542** | 0.879 | **0.404** | 0.386 |

Per-seed final rows for the last variant (`copy, modadd7, reverse, sort,
modadd13, induction6`):

```
seed0: [0.01, 0.98, 0.37, 0.12, 0.93, 0.27]   AA 0.447  FM 0.519
seed1: [0.98, 1.00, 0.34, 0.21, 1.00, 0.30]   AA 0.638  FM 0.289
```

Three things to read off this table:

1. **Fixed at eps 0.9 ≈ current at eps 0.9 in accuracy, at a third of the
   occupancy.** The fixes do not change what eps 0.9 protects (§4 already
   said it protects the wrong subspace); they change what it *costs*.
2. **That saved budget is what lets eps 0.97 run at all.** Under the current
   rule 0.97 saturates the cap; under the fixed rule it sits at 39 % after six
   tasks, well under the 0.612 the current rule spends on eps 0.9. And 0.97 is
   where the protected subspace starts to carry the task.
3. **AA 0.258 → 0.542, FM 0.742 → 0.404, LA unchanged.** The gap between the
   two seeds is large (0.45 vs 0.64), so this is a two-seed diagnostic, not a
   result; but the effect is roughly five times the seed spread, which is not
   the situation Stage 1 was in. `copy` surviving five later tasks at 0.98 in
   seed 1 has no precedent in `runs_A`, `runs_s1`, `runs_s2` or `runs_s3`.

What this does *not* establish: which of the six fixes carries the effect
(they were applied together, deliberately, to answer "is the ceiling real"
before spending on attribution); whether it holds at `small` on all twelve
tasks; whether eps 0.99 is affordable on twelve tasks even under the paper's
rule. Those are the next sweep's questions, in that order.

---

## 6 · What to do next, in order

The next thing is **not** another mechanism variant and not Phase 5. It is to
make the existing GPM deliver its stated guarantee, then re-measure. Each step
below is small, testable, and stays inside the one-change-at-a-time rule.

1. **Fix the harness first**, because it affects every mechanism's number:
   - `train.py`: draw `fisher_batches` from a separate training-distribution
     generator, never from `eval_batches` (D3).
   - `train.py`: default `optim.rewarm_per_task: true` for sequential mode
     (D6). Keep the global schedule available as an explicit opt-in.
   - `train.py`: reset optimizer state at task boundaries (or make it an
     option and measure it).
   - Add a test that `fisher_batches` and `eval_batches` are disjoint.
2. **Fix GPM to the paper's algorithm**, as a new registered mechanism
   (`gpm_v2`) so `gpm` stays as the published baseline:
   - paper rank rule, `k ≥ 0` (D2);
   - mask PAD positions when collecting (D4);
   - project the *step* after `optimizer.step()` so `ΔW·M = 0` under AdamW
     (D1), or switch GPM runs to SGD and re-tune the LR — the post-step
     projection is the cheaper path and keeps the optimizer comparable to
     every other mechanism;
   - include `lm_head` in the projected set with the final-norm basis, and
     freeze embedding rows of seen symbols (D5);
   - freeze norm gains after task 0, as the paper does (D5);
   - cache bases on-device; share q/k/v and gate/up bases (§3).
   - Add a signature probe that measures `||ΔW·M||/||ΔW||` on the actual
     weight step, with a `< 0.01` threshold. That is the probe that would
     have caught D1 on day one.
3. **Re-run the ablation controls and `gpm_v2`** on the 12-task Class-IL
   benchmark, 5 seeds, with the harness fixes. `eps` should be re-swept in the
   paper's range (0.95–0.99) because D2 is what made 0.97 fail. Budget:
   ~5 configs × 5 seeds ≈ 25 runs ≈ 3 h on the usual instance.
4. **Only then** decide between the EQUATION.md directions. Direction A
   (damped spectrum / natural-gradient form) is still the most attractive
   follow-up, but it should be compared against a GPM that actually works,
   not against the current one.
5. The class-incremental readout problem (D5) will not go away at scale: a
   real LM's tied embedding is the same single matrix. Whatever mechanism
   wins, the readout needs its own protection story. That is worth a design
   note before Phase 5.

---

## 7 · What does not change

- The ablation ranking of the *other* mechanisms is probably still right in
  order, since D3 and D6 affect them all similarly. Replay and DER's rho ≈ 1
  is unaffected by any of this.
- The benchmark decidability fix, the RNG seeding fix, the LA metric, the
  MDE table, and the pre-registration habit are all sound and should be kept.
- The 46 % memory reduction from `eps_growth = −0.01` becomes moot once D2 is
  fixed; drop the growth parameter and revisit if the paper's schedule
  (`+0.003` per task) is ever needed.
