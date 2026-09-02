# GPM — how it works, and where it breaks

Gradient Projection Memory was selected after measuring all fifteen mechanisms.
It is the only one that **reduces forgetting without sacrificing plasticity**:
FM 0.634 against the control's 0.758, at 91% of the control's learning ability.
Every other mechanism either forgets normally or retains by going rigid — EWC at
lambda=10000 reaches FM 0.000 with 19% plasticity, which is a frozen model, not
a solution.

Its measured rho is 0.073. Replay's is 1.004. This document is about the gap.

---

## 1 · The idea in one sentence

Learn each new task **only in directions that the old tasks never used**, so the
old behaviour is preserved by geometry rather than by penalty.

---

## 2 · Why that works — the guarantee

A linear layer computes `y = Wx`. Update the weights by `dW` and the output
becomes:

```
    (W + dW) x  =  Wx + dW·x
```

The old behaviour survives exactly when `dW·x = 0` for every input `x` the old
tasks actually produced.

Let `M` be an orthonormal basis (shape `in_dim x k`) spanning those old inputs.
Any old input is `x = Mc` for some coefficient vector `c`, so the condition
becomes:

```
    dW · M = 0
```

GPM enforces this by projecting the gradient onto the orthogonal complement of
`M` before every optimizer step — `projection.py`, `before_step`:

```python
    g = module.weight.grad          # (out, in)
    module.weight.grad = g - (g @ M) @ M.T
```

Verify the guarantee holds. Because `M` is orthonormal, `MᵀM = I`:

```
    (g - g M Mᵀ) M  =  gM - g M (MᵀM)  =  gM - gM  =  0
```

So the update is **exactly** orthogonal to the stored subspace. This is the
important difference from EWC: EWC *penalises* movement in important directions
and hopes lambda is tuned right; GPM makes that movement geometrically
impossible. Nothing to tune, no penalty to balance — old-task responses are
mathematically unchanged.

---

## 3 · Where the basis comes from

At each task boundary (`on_task_end`):

**a. Collect.** Run a few batches forward with hooks on every linear layer and
record the *input* activations. `R` has shape `(samples, in_dim)`.

**b. Remove what is already covered.** Only the genuinely new part of the
representation matters:

```
    R  <-  R - (R M_old) M_oldᵀ
```

**c. Find the principal directions.** Take the uncentered covariance and
decompose it. Since `RᵀR` is symmetric positive semi-definite, `U` holds its
eigenvectors and `S` the eigenvalues — the variance along each direction:

```
    U, S, _ = svd(Rᵀ R)
```

**d. Keep enough to explain `eps` of the variance:**

```
    csum = cumsum(S) / sum(S)
    k    = #{ i : csum_i < eps } + 1
```

**e. Append:** `M <- [ M_old | U[:, :k] ]`

---

## 4 · Where it breaks — measured, not theorised

Every task consumes `k_t` directions and **none are ever returned**. After `T`
tasks the free space is `in_dim - sum_t k_t`. When that reaches zero the layer
cannot learn at all.

This is not a hypothetical. Measured over twelve tasks:

| model | eps | basis used | directions frozen | rho | C2 signature |
|---|---|---|---|---|---|
| small | 0.80 | 59.5% of cap | 44.6% | **0.073** | 5/5 |
| small | 0.90 | 97.8% | 73.3% | 0.058 | 5/5 |
| small | 0.97 | 99.2% | 74.4% | **-0.047** | 5/5 |
| nano | 0.80 | 82.0% | 61.5% | **0.026** | 5/5 |
| nano | 0.90 | 99.9% | 74.9% | 0.003 | 1/5 |
| nano | 0.97 | 100.0% | 75.0% | 0.001 | **0/5** |

The relationship is monotonic in both tracks: **the fuller the basis, the worse
GPM performs.** At nano/0.97 the basis is exactly at the `max_bases_frac` cap,
the C2 probe reads exactly 0.7500, and the mechanism fails its own signature in
all five seeds — the harness refusing to score a mechanism operating outside its
valid regime.

Note what the cap does. It does not prevent saturation; it converts "frozen
solid" into "frozen at 75%", and the remaining 25% is not enough. The cap is a
symptom management, not a fix.

---

## 5 · A design choice that accelerates the failure

```python
    eps = min(0.99, eps_base + eps_growth * tasks_seen)   # eps_growth = 0.005
```

The threshold **rises** with task count, so later tasks retain *more* variance
and consume *more* directions — precisely when the basis is most nearly full.
At `eps_base=0.97` and twelve tasks this saturates at the 0.99 ceiling.

The intent was presumably to protect later tasks more carefully. The effect is
to accelerate the collapse. This is the first thing to test reversing.

---

## 6 · Improvement directions, in order of expected value

**1 · An aging basis.** The core problem is monotonic growth. A basis that
evicts directions — by age, or by how little the recent stream uses them —
turns an unbounded consumption into a steady state. GPM that forgets its own
constraints on a schedule is the single most promising change, and nothing in
the registry's papers implements it.

**2 · Soft projection instead of hard.** Currently a direction is binary: free
or frozen. Replace the projector with a weighted one,

```
    g  <-  g ( I - M diag(lambda) Mᵀ ),    lambda_i in [0, 1]
```

with `lambda_i` derived from that direction's importance. No direction is ever
fully frozen, so the free space never collapses to zero — a continuous
relaxation of the mechanism whose discrete cliff we just measured.

**3 · Invert eps_growth.** Shrink the threshold as the basis fills rather than
raising it.

**4 · Per-layer budgets.** One global `eps` for 56 layers of differing width.
Layers whose representations are genuinely low-rank should surrender fewer
directions.

**5 · Boundary-free operation.** GPM declares `requires=("task_boundaries",)`.
A continuous stream has none. Novelty-gated pseudo-boundaries are the plan;
measure how much GPM degrades under *wrong* boundaries before depending on it.

**6 · Memory at scale.** 55-83MB of basis at 23.7M parameters extrapolates to
roughly 16-25GB at 7B. Low-rank or per-block bases are the standard
compressions and nobody has applied them here.

---

## 7 · One property to preserve

Any change must keep the exactness of section 2. The reason GPM is worth
improving is that `dW·M = 0` is a *guarantee*, not a regularisation strength.
Soft projection (idea 2) deliberately trades that guarantee for headroom — which
makes it the most interesting experiment and the one that needs the closest
measurement, because it gives up the property that makes the method principled.
