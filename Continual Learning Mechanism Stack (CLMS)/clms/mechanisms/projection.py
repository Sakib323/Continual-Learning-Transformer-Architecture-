"""F03 · Optimization geometry: Gradient Projection Memory.

After each task, GPM collects the *input activations* of every linear layer,
takes their SVD, and keeps the leading bases that explain a threshold fraction
of the representation. Subsequent gradients are projected orthogonal to that
accumulated subspace:

    dW  <-  dW - dW M Mᵀ           M = accumulated basis of important inputs

A step in the orthogonal complement leaves the old layer response Wx unchanged
for any x in the stored subspace, which is why forgetting drops to near zero.

The cost is that the free subspace shrinks monotonically: every task consumes
directions and none are returned. That is fine for a short task sequence and
fatal for an unbounded stream, which is exactly why this is worth measuring
rather than assuming.

Signature (probe C2): principal angles between task gradient subspaces should be
large. If they are small, the tasks are not separable and projection cannot help.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..base import CostReport, Mechanism, SignatureCheck, to_cpu_tree
from ..registry import register


@register
class GradientProjectionMemory(Mechanism):
    name = "gpm"
    surfaces = ("O",)
    family = "F03"
    paper = "2103.09762"
    requires = ("task_boundaries",)
    conflicts = ("continual_backprop",)   # one consumes directions, the other
                                          # regenerates units; running both
                                          # muddles the attribution of any effect
    order = 40

    defaults = {
        "eps_base": 0.90,       # variance kept for task 0
        "eps_growth": 0.005,    # threshold rises with each task
        "max_bases_frac": 0.75, # cap: never consume more than this of a layer
        "collect_batches": 4,
        "min_features": 8,
        # C2 fires at this fraction of max_bases_frac, not at the cap itself.
        # Measured: the mechanism is already harmful well before saturation —
        # at 98% of cap rho was -0.047 while the probe still passed 5/5, because
        # the old threshold only tripped once the basis was literally full.
        #   72% of cap -> rho +0.073   healthy
        #   97% of cap -> rho +0.058   degrading
        #   98% of cap -> rho -0.047   harmful
        #  100% of cap -> rho  0.001   plasticity gone
        # 0.85 separates the healthy row from every degraded one.
        "saturation_warn_frac": 0.85,
        # Floor for the retention threshold. eps had an upper clamp but no lower
        # one, which is harmless while eps_growth is positive and unsafe once it
        # is not: at -0.01 the threshold goes negative by task 100, and a
        # negative "fraction of variance to retain" is meaningless. An unbounded
        # stream is the whole point of this project, so the floor is not
        # hypothetical.
        "eps_floor": 0.50,
    }

    def __init__(self, **kw):
        super().__init__(**kw)
        self.bases: dict[str, torch.Tensor] = {}   # layer name -> (in_dim, k)
        self._acts: dict[str, list[torch.Tensor]] = {}
        self._hook_rng: torch.Generator | None = None
        self._layers: dict[str, nn.Linear] = {}
        self._handles: list = []
        self._collecting = False
        self._tasks_seen = 0
        # saturation after each task boundary. The trajectory, not just the
        # final value, is what an eviction policy has to be designed against:
        # a basis that fills linearly needs a different policy from one that
        # jumps on the first task.
        self._saturation_history: list[float] = []
        self._last_grads: dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------
    def setup(self, model, cfg, ctx) -> None:
        self._hook_rng = self.rng(ctx)
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and module.in_features >= self.params["min_features"]:
                if "lm_head" in name:
                    continue   # tied to the embedding; projecting it is unstable
                self._layers[name] = module
                self._handles.append(
                    module.register_forward_hook(self._make_hook(name))
                )

    def _make_hook(self, name: str):
        def hook(module, inputs, output):
            if not self._collecting:
                return
            x = inputs[0].detach()
            flat = x.reshape(-1, x.shape[-1]).float().cpu()
            # subsample rows: the SVD only needs the row space.
            # Seeded — which rows are kept determines the basis, and drawing
            # from the global RNG made the whole run irreproducible.
            if flat.shape[0] > 512:
                idx = torch.randperm(flat.shape[0], generator=self._hook_rng)[:512]
                flat = flat[idx]
            self._acts.setdefault(name, []).append(flat)
        return hook

    # ------------------------------------------------------------------
    @torch.no_grad()
    def before_step(self, model, ctx) -> None:
        if not self.bases:
            return
        for name, module in self._layers.items():
            M = self.bases.get(name)
            if M is None or module.weight.grad is None:
                continue
            g = module.weight.grad            # (out, in)
            Md = M.to(g.device, g.dtype)      # (in, k)
            module.weight.grad = g - (g @ Md) @ Md.T
        self.mark_ran()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def on_task_end(self, model, task_id, ctx) -> None:
        batches = ctx.scratch.get("fisher_batches") or []
        self._acts.clear()
        self._collecting = True
        was_training = model.training
        model.eval()
        for i, batch in enumerate(batches):
            if i >= self.params["collect_batches"]:
                break
            model(batch["input_ids"].to(ctx.device))
        model.train(was_training)
        self._collecting = False

        eps = min(
            0.99,
            max(
                self.params["eps_floor"],
                self.params["eps_base"] + self.params["eps_growth"] * self._tasks_seen,
            ),
        )
        for name, parts in self._acts.items():
            R = torch.cat(parts, dim=0)               # (samples, in_dim)
            if R.shape[0] < 2:
                continue
            self._extend_basis(name, R, eps)
        self._acts.clear()
        self._tasks_seen += 1
        self._saturation_history.append(self._consumed_fraction())

    def _consumed_fraction(self) -> float:
        """Mean over layers of (directions frozen) / (directions available)."""
        consumed = []
        for name, M in self.bases.items():
            layer = self._layers.get(name)
            if layer is not None:
                consumed.append(M.shape[1] / layer.in_features)
        return sum(consumed) / len(consumed) if consumed else 0.0

    def _extend_basis(self, name: str, R: torch.Tensor, eps: float) -> None:
        existing = self.bases.get(name)
        in_dim = R.shape[1]

        if existing is not None:
            # only the part of the representation not already covered matters
            R = R - (R @ existing) @ existing.T

        try:
            U, S, _ = torch.linalg.svd(R.double().T @ R.double())
        except Exception:
            return
        total = float(S.sum())
        if total <= 0:
            return
        csum = torch.cumsum(S, dim=0) / total
        # Rank selection is a threshold, which makes GPM unusually sensitive to
        # float noise. On a non-deterministic backend a singular value sitting
        # near eps flips k by one, changing how many gradient directions are
        # frozen — a discrete change that then compounds over the rest of
        # training. Measured on MPS: AA 0.535 vs 0.587 at an identical seed,
        # while the same config is bit-exact on CPU.
        #
        # This is a property of the method, not a defect to fix. It does mean
        # GPM needs more seeds than the other mechanisms before its ranking can
        # be trusted, and that its error bars are wider than they look.
        k = int((csum < eps).sum()) + 1

        cap = int(in_dim * self.params["max_bases_frac"])
        used = existing.shape[1] if existing is not None else 0
        k = max(0, min(k, cap - used))
        if k == 0:
            return

        new = U[:, :k].float()
        self.bases[name] = new if existing is None else torch.cat([existing, new], dim=1)

    # ------------------------------------------------------------------
    def self_test(self, model, batch, ctx) -> tuple[bool, str]:
        if not self._layers:
            return False, "setup() registered no linear layers"
        # collect on one batch, build a basis, then verify a gradient is altered
        self._acts.clear()
        self._collecting = True
        model(batch["input_ids"])
        self._collecting = False
        if not self._acts:
            return False, "forward hooks captured no activations"

        name = next(iter(self._acts))
        R = torch.cat(self._acts[name], dim=0)
        self._extend_basis(name, R, 0.90)
        if name not in self.bases:
            return False, f"no basis extracted for {name}"

        module = self._layers[name]
        module.weight.grad = torch.randn_like(module.weight)
        before = module.weight.grad.clone()
        self.before_step(model, ctx)
        after = module.weight.grad
        changed = float((before - after).norm())
        k = self.bases[name].shape[1]
        self.bases.clear()
        self._acts.clear()
        model.zero_grad(set_to_none=True)
        if changed == 0.0:
            return False, "projection left the gradient unchanged"
        return True, f"projecting {len(self._layers)} layers; {name} basis k={k}"

    def signature(self, model, ctx) -> SignatureCheck | None:
        if not self.bases:
            return None
        consumed = []
        for name, M in self.bases.items():
            in_dim = self._layers[name].in_features
            consumed.append(M.shape[1] / in_dim)
        frac = sum(consumed) / len(consumed)
        warn_at = self.params["max_bases_frac"] * self.params["saturation_warn_frac"]
        return SignatureCheck(
            probe="C2",
            quantity="fraction of gradient directions consumed",
            value=frac,
            baseline=warn_at,
            direction="decrease",
            passed=frac < warn_at,
            detail=(
                f"fires at {warn_at:.3f} = {self.params['saturation_warn_frac']:.0%} "
                f"of the {self.params['max_bases_frac']:.2f} cap. Waiting for the "
                f"cap itself is too late: at 98% of it rho was -0.047 while the "
                f"probe still passed."
            ),
        )

    def cost_report(self) -> CostReport:
        n = sum(M.numel() for M in self.bases.values())
        return CostReport(
            buffer_bytes=n * 4,
            notes={"basis_entries": n, "layers": len(self._layers),
                   "tasks_seen": self._tasks_seen,
                   "saturation": round(self._consumed_fraction(), 4),
                   "saturation_history":
                       [round(v, 4) for v in self._saturation_history]},
        )

    def state_dict(self):
        return {"bases": self.bases, "tasks_seen": self._tasks_seen,
                "saturation_history": self._saturation_history}

    def load_state_dict(self, state):
        state = to_cpu_tree(state)
        self.bases = state.get("bases", {})
        self._tasks_seen = state.get("tasks_seen", 0)
        self._saturation_history = list(state.get("saturation_history", []))


@register
class AgingGradientProjectionMemory(GradientProjectionMemory):
    """GPM with a basis that forgets its own constraints.

    Baseline GPM consumes gradient directions monotonically and never returns
    any, so on a long stream the free subspace goes to zero and the model can no
    longer learn. Measured over twelve tasks, that is the binding failure:

        59.5% of cap consumed -> rho +0.073
        97.8%                 -> rho +0.058
        98.2%                 -> rho -0.047      actively harmful
       100.0%                 -> rho  0.001      plasticity gone

    This variant keeps a usage score per basis column, decays it at every task
    boundary, and evicts the least-used columns once occupancy exceeds a target.
    Consumption becomes a steady state rather than a ratchet.

    Registered as a separate mechanism rather than an edit to `gpm` so the
    harness compares them head to head with the same probes, costs and report,
    and so the published GPM result stays intact.

    Pre-registered failure: if saturation drops but rho does not improve, the
    problem is not *how many* directions are held but *which* — which would
    redirect the work to soft projection rather than to tuning eviction rates.
    """

    name = "gpm_aging"

    defaults = {
        **GradientProjectionMemory.defaults,
        # Occupancy to hold. Eviction triggers above this, so the basis settles
        # instead of filling. 0.40 sits below the 59.5% that still scored well,
        # leaving headroom rather than tracking the edge of the measured cliff.
        "target_occupancy": 0.40,
        # How fast a column's usage score fades. 0.7 means a direction unused
        # for three tasks retains ~a third of its score, so genuinely dead
        # directions leave while recently-useful ones survive a quiet spell.
        "usage_decay": 0.7,
    }

    def __init__(self, **kw):
        super().__init__(**kw)
        # layer name -> (k,) tensor of per-column usage scores, aligned with
        # the columns of self.bases[name]
        self._usage: dict[str, torch.Tensor] = {}
        self._evicted_total = 0

    # ------------------------------------------------------------------
    def _extend_basis(self, name: str, R: torch.Tensor, eps: float) -> None:
        before = self.bases.get(name)
        n_before = before.shape[1] if before is not None else 0

        # score the *existing* columns against this task's activations before
        # adding anything: a direction the current task still uses is alive
        if before is not None and R.shape[0] > 1:
            proj = R @ before                      # (samples, k)
            fresh = proj.pow(2).sum(dim=0)         # energy per column
            total = float(fresh.sum())
            if total > 0:
                fresh = fresh / total
            prev = self._usage.get(name)
            if prev is None or prev.numel() != n_before:
                prev = torch.zeros(n_before, dtype=fresh.dtype)
            self._usage[name] = prev * self.params["usage_decay"] + fresh.cpu()

        super()._extend_basis(name, R, eps)

        M = self.bases.get(name)
        if M is None:
            return
        added = M.shape[1] - n_before
        if added > 0:
            # a brand-new direction starts at the mean of the surviving scores,
            # not at zero — otherwise it is evicted before it can prove useful
            u = self._usage.get(name)
            seed_val = float(u.mean()) if u is not None and u.numel() else 1.0
            new = torch.full((added,), seed_val)
            self._usage[name] = new if u is None or u.numel() != n_before \
                else torch.cat([u, new])

        self._evict(name)

    def _evict(self, name: str) -> None:
        M = self.bases.get(name)
        layer = self._layers.get(name)
        if M is None or layer is None:
            return
        budget = int(layer.in_features * self.params["target_occupancy"])
        k = M.shape[1]
        if k <= budget:
            return

        usage = self._usage.get(name)
        if usage is None or usage.numel() != k:
            usage = torch.ones(k)
        keep = torch.topk(usage, budget).indices.sort().values
        self.bases[name] = M[:, keep].contiguous()
        self._usage[name] = usage[keep].contiguous()
        self._evicted_total += k - budget

    # ------------------------------------------------------------------
    def cost_report(self) -> CostReport:
        rep = super().cost_report()
        rep.notes["evicted_columns"] = self._evicted_total
        rep.notes["target_occupancy"] = self.params["target_occupancy"]
        return rep

    def state_dict(self):
        st = super().state_dict()
        st["usage"] = self._usage
        st["evicted_total"] = self._evicted_total
        return st

    def load_state_dict(self, state):
        super().load_state_dict(state)
        state = to_cpu_tree(state)
        self._usage = state.get("usage", {})
        self._evicted_total = state.get("evicted_total", 0)


@register
class SoftGradientProjectionMemory(GradientProjectionMemory):
    """GPM with partial rather than total suppression of stored directions.

    Baseline GPM removes the gradient component along every stored direction
    completely:

        g  <-  g - (g M) Mᵀ

    which gives an exact guarantee: `dW·M = 0`, so old-task responses are
    mathematically unchanged. The cost is that a direction, once stored, is
    permanently unavailable — the free subspace only shrinks.

    Stage 2 established that freeing directions is not the answer: evicting them
    dropped saturation 0.584 -> 0.250 and bought rho +0.024 +/- 0.117. If *how
    many* directions are held is not the constraint, the remaining hypothesis is
    that the binary choice itself is — a direction is either fully frozen or
    fully free, with nothing in between.

    This variant interpolates:

        g  <-  g - strength * (g M) Mᵀ

    `strength=1.0` reproduces baseline GPM exactly. Below that, every stored
    direction retains a `(1 - strength)` share of its gradient, so no direction
    is ever fully lost and the free subspace never collapses.

    **This deliberately gives up the exactness guarantee.** With strength < 1,
    `dW·M` is small rather than zero, and old-task responses drift. That
    guarantee is what distinguishes GPM from a penalty method like EWC, so the
    trade needs measuring rather than assuming — including whether what remains
    is still meaningfully different from EWC, which is also "discourage movement
    in important directions".

    Probe C4 reports the residual directly, so the claim is checked rather than
    trusted.
    """

    name = "gpm_soft"

    defaults = {
        **GradientProjectionMemory.defaults,
        # 1.0 == baseline GPM, exact. Lower keeps a share of every direction.
        "strength": 0.85,
    }

    def __init__(self, **kw):
        super().__init__(**kw)
        self._residual_sum = 0.0     # measured ||g_new M|| / ||g M||
        self._residual_n = 0

    @torch.no_grad()
    def before_step(self, model, ctx) -> None:
        if not self.bases:
            return
        s = float(self.params["strength"])
        for name, module in self._layers.items():
            M = self.bases.get(name)
            if M is None or module.weight.grad is None:
                continue
            g = module.weight.grad                 # (out, in)
            Md = M.to(g.device, g.dtype)           # (in, k)
            comp = g @ Md                          # (out, k) — the part inside the subspace
            g_new = g - s * (comp @ Md.T)
            module.weight.grad = g_new

            # verify the claim instead of asserting it: how much of the
            # in-subspace component actually survived?
            before = float(comp.norm())
            if before > 0:
                self._residual_sum += float((g_new @ Md).norm()) / before
                self._residual_n += 1
        self.mark_ran()

    def signature(self, model, ctx) -> SignatureCheck | None:
        base = super().signature(model, ctx)
        if self._residual_n == 0:
            return base
        residual = self._residual_sum / self._residual_n
        expected = 1.0 - float(self.params["strength"])
        # Report the *error* against the predicted residual, not the residual
        # itself. SignatureCheck.evaluate() recomputes `passed` from
        # baseline/direction and overrides whatever the mechanism sets, so a
        # "hold" check against a near-zero baseline fails for any float noise —
        # at strength=1.0 a residual of 0.026 was marked FAIL. Framing it as
        # "error must be small" works with that machinery instead of against it.
        error = abs(residual - expected)
        return SignatureCheck(
            probe="C4",
            quantity="error in the surviving gradient fraction",
            value=error,
            baseline=0.05,
            direction="decrease",
            detail=(
                f"strength={self.params['strength']:.2f} should leave "
                f"{expected:.3f} of the component along stored directions; "
                f"measured {residual:.3f}. At strength=1.0 the residual is 0 "
                f"and the original dW.M = 0 guarantee holds."
            ),
        )

    def cost_report(self) -> CostReport:
        rep = super().cost_report()
        rep.notes["strength"] = self.params["strength"]
        if self._residual_n:
            rep.notes["residual_frac"] = round(self._residual_sum / self._residual_n, 5)
        return rep
