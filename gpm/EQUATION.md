# The GPM update rule — what it constrains, what it misses, what to try

This is the mathematics, written to be modified. Every claim is tied to a number
we measured, so you can tell which parts are established and which are open.

---

## 1 · Every symbol, once

Work at a single linear layer. All indices `k` refer to layer depth.

**Scalars**

| symbol | meaning | our value |
|---|---|---|
| `L` | the loss | — |
| `eta` | learning rate | 1e-3 |
| `d_in`, `d_out` | layer input / output width | 512 / 512 at `small` |
| `k` | rank of the stored basis | ~0.5 * d_in after 12 tasks |
| `eps` | fraction of activation variance retained | 0.8-0.97 |
| `alpha` | damping (proposed, section 5) | — |
| `sigma_i` | variance along the i-th input direction | spans ~200x |

**Vectors** (columns)

| symbol | meaning | lives in |
|---|---|---|
| `x` | input activation to the layer | R^d_in |
| `x_par` | the part of `x` inside the protected subspace | R^d_in |
| `x_perp` | the residual, `x - x_par` | R^d_in |
| `y = Wx` | layer output | R^d_out |
| `delta_k` | perturbation in layer k's input, old vs new weights | R^d_in |

**Matrices**

| symbol | meaning | shape |
|---|---|---|
| `W` | layer weights | d_out x d_in |
| `dW` | the proposed update (raw gradient, before projection) | d_out x d_in |
| `R` | stacked input activations from old tasks | n_samples x d_in |
| `C = RᵀR` | input covariance (uncentered) | d_in x d_in |
| `M` | orthonormal basis of the protected input subspace | d_in x k |
| `N` | basis of protected *output* directions (proposed, section 6) | d_out x k' |
| `P = I - MMᵀ` | the projector onto the free complement | d_in x d_in |
| `J_k` | Jacobian of everything downstream of layer k | — |

**The three things to keep straight.** `M` acts on the *input* side, so `dW·M`
is a d_out x k matrix — the part of the update that touches protected inputs.
`P` is right-multiplied: `dW·P`. And `M` is orthonormal, so `MᵀM = I` — that
single fact is what makes the guarantee exact.

---

## 2 · The current rule and why it works

```
    dW  <-  dW - (dW M) Mᵀ   =   dW (I - M Mᵀ)   =   dW P
```

The guarantee, in one line:

```
    (dW P) M  =  dW M - dW M (Mᵀ M)  =  dW M - dW M  =  0
```

So for any input `x = Mc` in the protected subspace:

```
    (W + dW P) x  =  Wx + (dW P M) c  =  Wx
```

The layer's response to every stored input is **exactly** unchanged. Not
penalised, not discouraged — algebraically identical. That is the whole method,
and it is why EWC (a penalty) and GPM (a constraint) are different in kind.

**Measured: this part is load-bearing.** Stage 3 replaced the coefficient 1 with
`s < 1` and rho fell monotonically, 0.287 -> 0.108, t ~ 5.6. Weakening the
exactness is strictly worse. Do not spend the next equation on softening it.

---

## 3 · The term that is missing

Decompose any input: `x = x_par + x_perp`, with `x_par = MMᵀx`.

```
    dW P x  =  dW P x_par  +  dW P x_perp
            =      0       +  dW P x_perp
```

**`dW P x_perp` is completely unconstrained.** GPM says nothing about it. And
`x_perp` is not negligible: measured per layer, the stored basis captures

```
    ||x_par||^2 / ||x||^2  ~  0.849       (mean over 28 projected layers)
```

so about **15% of the activation energy at every layer sits outside the
guarantee**. That residual does not decay as training proceeds — we checked,
capture goes 0.812 -> 0.917 over four tasks, it rises.

---

## 4 · Why 15% per layer is fatal — the actual obstacle

A network is a composition, and the perturbation propagates. Let `delta_k` be
the difference in layer k's input between old and new weights. Then

```
    delta_{k+1}  =  W_k delta_k   +   dW_k x_k   +   dW_k delta_k
                     \_______/        \______/        \________/
                    amplification      leak          cross term
```

GPM sets only the middle term to zero, and only for `x_k` in the stored
subspace. Everything else is free:

- **`W_k delta_k`** — the *old* weights amplify whatever error already arrived.
  Nothing in GPM bounds `||W_k||`.
- **`dW_k x_perp`** — the 15% leak, discussed above.
- **`dW_k delta_k`** — the update acting on an incoming perturbation. Second
  order, but nonzero.

And it compounds. At 0.849 capture per layer:

```
    depth  4  ->  0.849^4   =  0.519
    depth  8  ->  0.849^8   =  0.270
    depth 28  ->  0.849^28  =  0.010
```

> **The obstacle, stated plainly.** GPM bounds a *local, per-layer, first-order*
> quantity. Forgetting is a *global, composed* quantity. The equation has no
> term that knows another layer exists. Any new rule that stays inside one
> layer's algebra will hit the same ceiling — which is what Stages 1, 2 and 3
> each demonstrated from a different direction.

---

## 5 · Direction A — replace the truncated basis with the full spectrum

The most promising change, and the cleanest mathematically.

GPM's `M` is the top-`k` eigenvectors of `C`, and `P = I - MMᵀ` is a **binary**
operator: a direction is either fully frozen or fully free, with a cliff at rank
`k`. That cliff is the source of the saturation failure, the reproducibility
problem, and the 15% leak all at once.

Replace it with a **damped** operator that uses every direction:

```
    dW  <-  dW ( I  -  C (C + alpha I)^{-1} )
```

Eigendecompose `C = U diag(sigma) Uᵀ`. Then the operator is diagonal in the same
basis, with eigenvalues

```
    alpha / (sigma_i + alpha)
```

so a direction is suppressed **in proportion to how much the old tasks used it**:

| sigma | suppression factor | behaviour |
|---|---|---|
| >> alpha | ~0 | frozen, as GPM freezes its top-k |
| ~ alpha | ~0.5 | half-free |
| << alpha | ~1 | passes untouched |

Measured on a realistic spectrum (alpha = 1):

```
    sigma   GPM (hard)   damped
    10.00        0.000    0.091
     3.00        0.000    0.250
     0.50        0.000    0.667
     0.05        1.000    0.952
```

**This is not the Stage-3 softening.** Stage 3 scaled the whole projection by a
constant `s`, weakening protection on the *important* directions — which is why
it failed. The damped form does the opposite: important directions get *more*
suppression, unimportant ones less. Stage 3 flattened the profile; this sharpens
it.

Three properties worth having:

1. **No rank, no threshold, no eviction.** Saturation, `eps_growth`,
   `max_bases_frac` and the rank-flip reproducibility bug all disappear —
   they were artefacts of truncation.
2. **No leak.** Every direction is handled, including the 15% tail.
3. **One parameter.** `alpha` is a variance scale: directions below it are
   treated as free.

**An identity worth knowing.** Since `I - C(C + alpha I)^{-1} = alpha (C + alpha I)^{-1}`
(verified numerically to 1e-6), the rule is equivalently

```
    dW  <-  alpha * dW (C + alpha I)^{-1}
```

That is **preconditioning by the damped inverse input covariance** — the same
object as natural gradient and what K-FAC approximates. Your continual-learning
rule and second-order optimization turn out to be the same operator with
different motivations. That is a large body of existing theory and efficient
implementations to draw on, and a strong sign the form is natural rather than
invented.

Caveat: the `alpha` factor shrinks the step, so renormalise (e.g. rescale to
preserve `||dW||`) or the effect is confounded with a smaller learning rate.

---

## 6 · Direction B — constrain the output side as well

GPM constrains only the input side. Nothing says *which output directions* may
move, yet the output is precisely what the next layer receives — the `delta`
that propagates in section 4.

```
    dW  <-  ( I - N Nᵀ ) dW ( I - M Mᵀ )
```

with `N` spanning the output directions old tasks actually used, built the same
way `M` is but from the layer's outputs.

This attacks the compounding term directly: if the update cannot move the output
directions downstream layers depend on, the perturbation entering layer k+1 is
confined to directions those layers were never sensitive to.

Cost: a second basis per layer, roughly doubling GPM's already-large state.

---

## 7 · Direction C — weight by downstream sensitivity

Both A and B still treat every layer identically. But a perturbation introduced
at layer 1 passes through everything after it, while one at the last layer does
not. That is exactly what the compounding table shows.

Build `M` not from activation variance alone but from variance weighted by how
much the loss depends on that direction:

```
    C_weighted  =  Rᵀ diag(w) R,      w_i  ~  || dL / dy_i ||
```

Early layers then surrender more directions, later layers fewer — allocated by
measured influence rather than by a single global `eps` shared across 56 layers
of different widths and depths.

Cheap to test: it changes only how `R` is accumulated, not the update rule.

---

## 8 · Direction D — constrain the function, not the weights

The most ambitious, and the only one that leaves the per-layer frame entirely.

```
    minimise  L_new(W)     subject to   || f_new(x) - f_old(x) ||  <=  epsilon
                                        for x drawn from the old input distribution
```

GPM is a first-order, per-layer surrogate for this. The obstacle is that you
need old `x` — which is replay, and replay already scores rho 1.004.

The interesting middle ground: **sample `x` synthetically from `span(M)`**. You
already store `M`; drawing `x = Mc` for random `c` gives inputs statistically
like the old task's without storing any real data. Then constrain the *composed*
network on those samples. That is a genuinely different object — neither
projection nor rehearsal — and it is the one direction here that could plausibly
break the ceiling rather than raise it.

---

## 9 · What to draw, in order

Nine constructions. The first five build intuition; the last four are where the
open questions live.

| # | draw | what it shows |
|---|---|---|
| 1 | `y = Wx` as a vector transformed by a matrix | the object being protected |
| 2 | `span(M)` as a plane in 3D, `x` split into `x_par` + `x_perp` | what "protected" means |
| 3 | `dW` and `dW P` side by side | the projection removing a component |
| 4 | `(dW P) M = 0` | the guarantee, visibly |
| 5 | `dW P x_perp` — nonzero, pointing anywhere | **the leak** |
| 6 | `delta_{k+1} = W delta_k + dW x + dW delta_k`, three arrows summing | **the obstacle** |
| 7 | suppression `alpha/(sigma+alpha)` vs `sigma`, with GPM's step function overlaid | Direction A |
| 8 | two-sided: `N` on the output, `M` on the input | Direction B |
| 9 | `0.849^depth` decay curve | why per-layer is not enough |

Do 5 and 6 first. They are the two the current equation cannot express, and
everything worth trying is an attempt to add a term to one of them.

---

## 10 · Tools

**GeoGebra 3D Calculator** (geogebra.org/3d) — best for constructions 1-6.
Native vectors, planes, matrices and sliders; you can define `M` as a plane,
drop `x` on it, and watch `x_perp` update as you drag. No setup.

**Desmos** (desmos.com/calculator) — best for 7 and 9. They are ordinary
functions of one variable (`alpha/(sigma+alpha)`, `c^d`) and Desmos handles
sliders and overlaid curves better than anything else.

**Observable** (observablehq.com) — when you outgrow both. Real linear algebra in
JavaScript, arbitrary dimensions, and you can simulate an actual multi-layer
perturbation cascade rather than a picture of one. This is where construction 6
becomes quantitative instead of illustrative.

Start in GeoGebra for 5 and 6; move to Observable when you want to ask *how much*
rather than *what shape*.

---

## 11 · Ranking, by expected value

1. **Direction A (damped spectrum)** — cleanest, removes four existing failure
   modes at once, connects to natural gradient, one parameter. Test first.
2. **Direction C (sensitivity weighting)** — cheap, changes only how `R` is
   accumulated, directly addresses the depth asymmetry.
3. **Direction B (two-sided)** — attacks compounding head-on but doubles the
   state, which is already GPM's worst scaling property.
4. **Direction D (function constraint)** — highest ceiling, hardest, and the only
   one that leaves the per-layer frame. The synthetic-sampling variant is the
   version worth attempting.

A and C are compatible and could be one equation. B and D are separate research
programmes.
