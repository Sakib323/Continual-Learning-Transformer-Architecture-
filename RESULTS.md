# Phase 3 — which mechanisms work

> **Headline, Class-IL (the realistic setting): none of the thirteen
> rehearsal-free mechanisms work.** Best is EWC at rho 0.104 +/- 0.065 over
> twelve tasks, and it gets there by going rigid — 42% of the control's
> plasticity. See "Class-IL: the decisive negative result" below. The Task-IL
> table that follows is the easier scenario, where the model is told which task
> each input belongs to.

Fifteen continual-learning mechanisms across five injection surfaces, measured on
a five-task algorithmic stream at 23.7M parameters, three seeds each, with each
mechanism tuned over its own grid.

**Setup.** `copy → reverse → sort → modadd23 → induction6`, 400 steps per task,
RTX 2060. `floor 0.1495` (sequential fine-tuning), `ceiling 0.8336` (joint
training), `span 0.6841`.

`gpm`, `shrink_perturb` and `cbp` are reported from a later five-seed sweep
(`floor 0.1330`, `ceiling 0.8428`, `span 0.7098`) run after their randomness was
seeded — see the reproducibility section. rho is normalised within each sweep,
so the two are comparable; raw accuracy is not.

**Reading the columns.**

- **rho** — recovery ratio. 0 = no better than sequential fine-tuning, 1 =
  matching joint training. Negative means worse than doing nothing.
- **LA%** — learning accuracy relative to the control: did the model still learn
  each task when it was trained? Below 100% means the mechanism cost plasticity.
- **FM** — forgetting.
- **signatures** — independent probes that the mechanism did the internal thing
  its paper claims, not merely that it scored well.

---

## Results

| mechanism | rho | ±sd | LA% | FM | sig | verdict |
|---|---|---|---|---|---|---|
| **replay** | **1.004** | 0.019 | 101% | 0.007 | 3/3 | **matches joint training** |
| **der** | **0.982** | 0.008 | 100% | 0.017 | 3/3 | **matches joint training** |
| ewc | 0.335 | 0.136 | **53%** | 0.073 | 3/3 | partial — buys retention with plasticity |
| gpm | 0.227 | 0.050 | 101% | 0.673 | 5/5 | partial — keeps plasticity, weak retention |
| lwf | 0.077 | 0.028 | **56%** | 0.334 | 3/3 | marginal, and costs plasticity |
| kwta | 0.043 | 0.076 | 99% | 0.808 | 3/3 | no effect |
| memory_sparse | 0.019 | 0.066 | 100% | 0.833 | 6/6 | no effect |
| si | 0.001 | 0.058 | 100% | 0.854 | none | no effect |
| shrink_perturb | 0.027 | 0.046 | 100% | 0.847 | none | no effect |
| xdg | −0.037 | 0.018 | 100% | 0.883 | 3/3 | harmful |
| cbp | −0.034 | 0.037 | 98% | 0.879 | 5/5 | no effect |
| lora | −0.129 | 0.009 | **51%** | 0.456 | 3/3 | harmful |
| olora | −0.178 | 0.011 | **38%** | 0.359 | 3/3 | harmful |
| l2p | −0.207 | 0.003 | **26%** | 0.257 | 3/3 | harmful |

Two mechanisms work. Two are partial. The rest do nothing or hurt.

---

## What the LA column buys

Average accuracy alone cannot distinguish *learned then forgot* from *never
learned*. Both produce a low score, and the second looks like partial success on
any forgetting-only metric. Splitting them out sorts the field into three
regimes that AA collapses into one:

**Rehearsal keeps both.** replay and DER learn every task as well as the control
does and then hold onto it. FM near zero with LA at 100%.

**Regularization trades one for the other.** EWC retains what it learns and then
goes rigid. Its diagonal:

```
control_sequential   1.00  1.00  0.87  1.00  0.31   <- all five learnable
ewc[lam=100]         1.00  0.99  0.00  0.02  0.16   <- rigid after task 1
ewc[lam=10000]       1.00  0.00  0.00  0.02  0.00   <- rigid after task 0
```

EWC's rho of 0.335 is entirely retention of copy and reverse. It never learns
`sort`, `modadd23` or `induction6` at all — the accumulated Fisher penalty locks
the model. LwF, LoRA, O-LoRA and L2P show the same trade on worse terms; L2P
ends at 26% of the control's plasticity.

**Sparsity and projection keep plasticity and retain nothing.** kWTA, XdG, SI,
shrink-perturb and CBP all sit at LA ≈ 100% with FM ≈ 0.85 — the same forgetting
as doing nothing.

---

## Cost

| | rho | buffer | at 7B |
|---|---|---|---|
| replay | 1.004 | **0.7 MB** | trivial |
| der | 0.982 | 22 MB | fine |
| ewc | 0.335 | 189 MB (8x the model) | **~56 GB — blocker** |

The best mechanism is also the cheapest, and the one that scales worst is a
partial result.

---

## Four bugs the signature probes caught

None of these were visible in accuracy. Each produced a plausible-looking number
that meant something other than what it appeared to.

**EWC's Fisher collapsed to 1e-11.** The empirical Fisher uses squared gradients
of the loss on ground-truth labels; once a task is solved those vanish, so no
lambda could do anything. Switching to the model Fisher gained ~1000x and still
landed at 1e-9. Normalizing each task's Fisher to unit mean is what makes lambda
scale-free. Without the fix EWC scored rho≈0 at every lambda with `B1:FAIL`.

**O-LoRA was frozen and the bug flattered it.** `set_active(0)` left adapters
1..N with `requires_grad=False`, and `build_optimizer` skipped those, so they
never entered the optimizer. From task 1 the model was frozen solid — `FM`
exactly 0.0, `CKA` exactly 1.0, identical results at every lambda. Fixed, and
O-LoRA's score got *worse*: a model frozen on task 0 scores better than one that
actually trains. rho went +0.107 → −0.178.

**`induction` was ill-posed.** The cue symbol was not guaranteed unique, so
11.4% of sequences had an answer undecidable from the input regardless of model
quality.

**Gradient masking does not freeze parameters under AdamW.** `sparse_update`
expressed "only update these slots" by multiplying the rest of the gradient by
zero. Adam's momentum and AdamW's decoupled weight decay kept moving them
anyway — measured at 67% of a normal update. `after_step` now restores the
masked values, and the freeze is exact (max displacement 0.0, not merely small).

That last fix did **not** rescue the mechanism: memory_sparse went from 1/6
signature checks to 6/6 while rho stayed at 0.019 ± 0.066. The implementation is
now provably correct and the idea still does not help here — which is a far
stronger negative result than the version confounded by a bug.

---

## Two probes that were measuring the wrong thing

Signature probes need the same scrutiny as mechanisms.

**A4 counted cumulatively.** "Fraction of slots ever accessed" reaches 1.0 over
any long run no matter how sparse each step is. It failed in 5 of 6 seeds while
the mechanism worked correctly. Rewritten as concentration — the fraction of
slots absorbing 90% of accesses — it reads 0.047.

**B2 was scoped to the whole model.** `sparse_update` masks memory slots and
leaves the backbone to train densely, so averaging over all parameters buried
the signal near 0.5. Rescoped to the masked units and reformulated as the
per-step claim, it now tracks the knob directly:

```
slot_frac=0.01  ->  B2 = 0.00977
slot_frac=0.03  ->  B2 = 0.02973
slot_frac=0.1   ->  B2 = 0.08849
```

A probe sitting on its own threshold is also a defect: replay's D1 first passed
at 0.5028 against a 0.5 bar, one seed from a false alarm. Rethresholded to 0.2,
where the regimes actually separate — a healthy reservoir sits at 0.4–0.7, a
buffer that drops data at the task boundary reads ~0. It now measures 0.650
across every ratio, which is the first independent confirmation that the
headline result rehearses earlier tasks rather than merely scoring well.

---

## A reproducibility bug in three mechanisms

`gpm`, `shrink_perturb` and `continual_backprop` drew training randomness from
the **global** RNG — GPM subsampling activations for its projection basis,
shrink-perturb injecting noise, CBP deciding stochastically which units to
reinitialise. Runs were therefore not reproducible at a fixed seed:

```
gpm, identical seed and config, before the fix
  run a: AA=0.243056   run b: AA=0.538194    dAA = 0.295
```

That spread is roughly 0.43 of span — larger than most mechanisms' entire
effect — and seed-to-seed error bars cannot see it, because the variance lives
*within* a seed. Every reported sd for those three was understated, and GPM's
apparent 0.207 -> 0.324 improvement between sweeps was noise.

Fixed with a per-mechanism seeded generator, offset by a hash of the mechanism
name so stacked mechanisms do not share a stream. Verified: the same config is
now **bit-exact on CPU** (dAA = 0.00000000).

Re-measured with five seeds after the fix, all three land where the noise had
been hiding them:

| | before (3 seeds, unseeded) | after (5 seeds, seeded) | |
|---|---|---|---|
| gpm | 0.207 then 0.324 across two sweeps | **0.227 ± 0.050** | the 0.324 was noise |
| cbp | −0.081, read as "harmful" | **−0.034 ± 0.037** | no effect, not harmful |
| shrink_perturb | −0.018 | **0.027 ± 0.046** | no effect, unchanged verdict |

The fix also recovered dose-response that the noise had buried. GPM's epsilon
now orders sensibly (0.182 / 0.227 / 0.220 for 0.8 / 0.9 / 0.97) and CBP degrades
monotonically as its replacement rate rises (−0.034 / −0.049 / −0.067). Neither
trend was visible before.

A residual remains on GPU (dAA 0.052 on MPS) and is *not* our code. GPM selects
its basis rank by thresholding the cumulative singular-value spectrum, so float
noise near the threshold flips the retained rank by one — a discrete change to
the constraint that compounds. GPM is inherently higher-variance than the other
mechanisms on any non-deterministic backend and needs more seeds before its
ranking can be trusted.

## Reproducibility

`replay` and `olora` are bit-identical between the v2 and v3 sweeps — same
seeds, same card, unchanged code paths, `Δrho = +0.000` to three decimals. The
harness is deterministic.

Cross-sweep comparison is done in rho rather than raw accuracy, because rho is
normalized against controls re-run on the same hardware. Raw AA is not
comparable across cards; rho is.

---

## What this does not establish

- **5 tasks at 23.7M parameters.** Rankings at this scale are evidence for, not
  proof of, behaviour at 1B+.
- **replay's rho ≈ 1.0 is close to tautological.** At ratio 0.5 half of every
  batch is old data, which approximates joint training by construction. The
  interesting question is what works when a buffer is not allowed — where the
  best result is currently EWC at 0.335, with a plasticity cost.
- **The plasticity family is scored on the wrong axis.** CBP, shrink-perturb and
  kWTA exist to preserve plasticity, not to prevent forgetting. Their negative
  rho here says they do not solve a problem they never claimed to.
- **`si` and `shrink_perturb` have no signature probe**, so for those two there
  is a score without an independent check that the mechanism engaged.


---

# Class-IL: the decisive negative result

227 runs, RTX 3060, one GPU throughout, zero crashes. Twelve tasks, decidable
(disjoint symbol ranges per input shape), no task token at inference.

**Track A** — retention at full capacity (`small`, 23.7M).
`floor 0.0813  ceiling 0.8753  span +0.7940`

| mechanism | rho | +/-sd | LA% | FM | verdict |
|---|---|---|---|---|---|
| ewc[lam=10] | 0.104 | 0.065 | **42%** | 0.179 | marginal |
| gpm[eps=0.8] | 0.073 | 0.039 | 96% | 0.664 | marginal |
| lwf[lam=0.1] | 0.060 | 0.050 | 90% | 0.620 | marginal |
| memory_sparse | 0.051 | 0.064 | 98% | 0.699 | no effect |
| shrink_perturb | 0.042 | 0.048 | 100% | 0.720 | no effect |
| si, memory_layer, sparse_update, kwta | <=0.034 | | ~100% | ~0.75 | no effect |
| cbp, l2p, lora, olora | negative | | 14-101% | | harmful |

**Track B** — plasticity regime (`nano`, where the control loses 0.396 of its
learning ability across the stream).
`floor 0.0198  ceiling 0.7680  span +0.7482`

| mechanism | rho | +/-sd | LA% | FM | verdict |
|---|---|---|---|---|---|
| ewc[lam=10] | 0.144 | 0.036 | **42%** | 0.186 | partial |
| si[c=0.01] | 0.039 | 0.037 | 99% | 0.709 | marginal |
| gpm[eps=0.8] | 0.026 | 0.014 | 82% | 0.587 | marginal |
| kwta, shrink_perturb, cbp, sparse_update | <=0.013 | | 70-105% | | no effect |

## Three things this settles

**Rehearsal-free continual learning does not work here.** The best result
recovers a tenth of the gap between sequential fine-tuning and joint training,
and only EWC in Track B clears `rho - sd > 0.1`. Every mechanism passes its
signature probes, so this is a statement about the methods, not the code.

**EWC "wins" by declining to learn.** Its 42% LA means it retains what it has by
refusing new tasks. That is the stability-plasticity trade taken to its limit,
not a solution to it. AA is essentially LA - FM (correlation 0.987 in Track A,
0.999 in Track B), so any mechanism can trade one for the other and none of them
escape the exchange rate.

**The plasticity family fails in the regime built to favour it.** Track B runs
at a capacity where the control demonstrably loses plasticity (+0.396 decay).
CBP, shrink-perturb, kWTA and sparse-update score 0.003, 0.012, 0.013 and -0.003
there. Several do preserve plasticity — si at 109% of control, cbp at 105% — and
it buys nothing, because plasticity was never the binding constraint. Forgetting
was.

## Forward transfer: none, anywhere

Mean FWT of -0.055 across both tracks, every mechanism, every setting. Nothing
learns a new task faster for having seen earlier ones. These methods can at best
preserve what they had; none of them compound.

## A signature probe earning its keep

GPM failed C2 in Track B at eps 0.9 and 0.97 — 1/5 and 0/5 seeds — while passing
5/5 in Track A at identical settings. The measured value is exactly 0.7500 in
both failing cases, which is `max_bases_frac`: at nano the basis saturates its
cap and freezes 75% of all gradient directions, collapsing LA to 74%. The scores
are reported as untrustworthy rather than ranked, which is the difference
between a benchmark and a leaderboard.
