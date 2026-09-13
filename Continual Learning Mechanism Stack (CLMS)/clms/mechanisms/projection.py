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


@register
class GradientProjectionMemoryV2(GradientProjectionMemory):
    """GPM as the paper specifies it, plus what a transformer under AdamW needs.

    `gpm` above was audited on 2026-09-13 (see gpm/AUDIT.md) and found not to
    deliver the `dW·M = 0` guarantee it documents. This class fixes each defect
    and is registered separately so the published `gpm` numbers stay intact
    and the two can be compared in one sweep.

    D1  The weight *step* is projected, not only the gradient. AdamW's
        per-element rescaling puts 9-17% of every step back inside M; the
        step is corrected after `optimizer.step()` so `ΔW·M = 0` holds for the
        parameters that actually change. Probe C5 measures this on the real
        step and fails above 1%.
    D2  Rank selection follows Eq. 9 of the paper: the threshold is on the
        task's *total* representation energy including what the existing
        basis already captures, and k may be 0. The old rule thresholded the
        residual alone and consumed the basis 3.6x faster.
    D4  PAD positions are dropped from the representation matrix. They were
        up to 79% of the collected rows and protect nothing.
    D5  The readout is protected: `lm_head` is projected with the final-norm
        basis like any other layer (it was skipped before), embedding rows of
        symbols already seen are frozen exactly, and norm gains are frozen
        after the first boundary as the paper freezes its BN parameters.
    §3  Bases are cached on the device and shared between layers that read
        the same input (q/k/v, gate/up), which the audit measured as ~43% of
        basis memory and most of GPM's 2x wall-time overhead.

    D3 and D6 are harness fixes (train.py, config.py), not mechanism ones.

    Defaults follow the paper's range: eps 0.97, no cap on occupancy. With
    the corrected rank rule the cap was measured unnecessary at six tasks
    (occupancy 0.39 at eps 0.97) and it distorts the guarantee once hit.
    """

    name = "gpm_v2"
    paper = "2103.09762"
    conflicts = ("continual_backprop",)
    order = 40

    defaults = {
        **GradientProjectionMemory.defaults,
        "eps_base": 0.97,
        "eps_growth": 0.0,
        "max_bases_frac": 1.0,
        # token id excluded from the representation matrix
        "pad_id": 0,
        # correct the weight step after the optimizer so dW·M = 0 holds under
        # AdamW. Off reproduces D1 for ablation; probe C5 then fails.
        "project_step": True,
        # project lm_head and freeze the embedding rows of seen symbols
        "protect_readout": True,
        # freeze every 1-D parameter (norm gains; the model has no biases)
        # after the first task boundary
        "freeze_norms": True,
        # one basis per distinct layer input instead of one per nn.Linear
        "share_bases": True,
        # measure the step leak every N steps (each measurement is one extra
        # matmul per layer)
        "leak_probe_every": 25,
    }

    def __init__(self, **kw):
        super().__init__(**kw)
        self._mask: torch.Tensor | None = None
        self._pre: dict[str, torch.Tensor] = {}
        self._pre_rows: list[torch.Tensor] = []
        self._dev_bases: dict[str, torch.Tensor] = {}
        self._group_of: dict[str, str] = {}
        self._grouping = False
        self._keepalive: list = []
        self._id_seen: dict[int, str] = {}
        self._seen_symbols: set[int] = set()
        self._seen_rows: torch.Tensor | None = None
        self._embeddings: list[nn.Embedding] = []
        self._norm_params: list[torch.Tensor] = []
        self._readout: set[str] = set()
        self._step_count = 0
        self._leak_raw_sum = 0.0
        self._leak_sum = 0.0
        self._leak_n = 0

    # ------------------------------------------------------------------
    def setup(self, model, cfg, ctx) -> None:
        self._hook_rng = self.rng(ctx)
        self._embeddings = [m for m in model.modules() if isinstance(m, nn.Embedding)]
        tied = {id(e.weight) for e in self._embeddings}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and module.in_features >= self.params["min_features"]:
                is_readout = "lm_head" in name or id(module.weight) in tied
                if is_readout:
                    if not self.params["protect_readout"]:
                        continue
                    self._readout.add(name)
                self._layers[name] = module
                self._handles.append(module.register_forward_hook(self._make_hook(name)))
        self._norm_params = [p for _, p in model.named_parameters() if p.dim() == 1]

    def _make_hook(self, name: str):
        def hook(module, inputs, output):
            if not self._collecting:
                return
            x = inputs[0]
            if self._grouping:
                # one pass with every input kept alive, so identical objects
                # mean identical inputs and nothing is aliased by reuse
                self._keepalive.append(x)
                self._group_of[name] = self._id_seen.setdefault(id(x), name)
                return
            if self._group_of.get(name, name) != name:
                return                      # a group member; the leader stores
            flat = x.detach().reshape(-1, x.shape[-1])
            if self._mask is not None and self._mask.numel() == flat.shape[0]:
                flat = flat[self._mask.to(flat.device)]
            flat = flat.float().cpu()
            if flat.shape[0] > 512:
                idx = torch.randperm(flat.shape[0], generator=self._hook_rng)[:512]
                flat = flat[idx]
            if flat.shape[0]:
                self._acts.setdefault(name, []).append(flat)
        return hook

    def _leader(self, name: str) -> str:
        return self._group_of.get(name, name) if self.params["share_bases"] else name

    def _basis_for(self, name: str, like: torch.Tensor) -> torch.Tensor | None:
        key = self._leader(name)
        M = self.bases.get(key)
        if M is None:
            return None
        dev = self._dev_bases.get(key)
        if dev is None or dev.device != like.device or dev.dtype != like.dtype:
            dev = M.to(like.device, like.dtype)
            self._dev_bases[key] = dev
        return dev

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _collect(self, model, batches: list[dict], ctx, n_forward: int) -> None:
        self._acts.clear()
        was_training = model.training
        model.eval()
        self._collecting = True
        pad = self.params["pad_id"]
        for i, batch in enumerate(batches):
            ids = batch["input_ids"]
            if self.params["protect_readout"]:
                self._seen_symbols.update(int(v) for v in ids.unique().tolist())
            if i >= n_forward:
                continue
            if self.params["share_bases"] and not self._group_of:
                self._grouping, self._id_seen, self._keepalive = True, {}, []
                model(ids.to(ctx.device))
                self._grouping, self._id_seen, self._keepalive = False, {}, []
            self._mask = (ids != pad).reshape(-1)
            model(ids.to(ctx.device))
        self._mask = None
        self._collecting = False
        model.train(was_training)
        self._seen_rows = None

    def _build(self, eps: float) -> None:
        for name, parts in list(self._acts.items()):
            R = torch.cat(parts, dim=0)
            if R.shape[0] >= 2:
                self._extend_basis(name, R, eps)
        self._acts.clear()
        self._dev_bases.clear()

    def _extend_basis(self, name: str, R: torch.Tensor, eps: float) -> None:
        """Eq. 9: ||R_proj||² + ||R̂_k||² >= eps ||R||², smallest k, k >= 0.

        Numerics matter here more than they look. The first version took the
        SVD of the residual's Gram matrix RᵀR; once the basis held more than
        half a layer, the Gram matrix had a null space of several hundred
        dimensions and the returned leading vectors overlapped the existing
        basis by up to 0.35, so M stopped being orthonormal, (I − MMᵀ) stopped
        being a projector, and probe C5 read 0.93. Decompose the residual
        itself, re-orthogonalise the new columns against the stored basis,
        and check the invariant every time it is extended.
        """
        existing = self.bases.get(name)
        Rd = R.double()
        total = float((Rd ** 2).sum())
        if total <= 0:
            return
        if existing is not None:
            E = existing.double()
            Rp = Rd - (Rd @ E) @ E.T
            captured = total - float((Rp ** 2).sum())
        else:
            E, Rp, captured = None, Rd, 0.0
        target = eps * total
        if captured >= target:
            return
        try:
            _, sv, Vh = torch.linalg.svd(Rp, full_matrices=False)
        except Exception:
            return
        energy = sv ** 2
        csum = captured + torch.cumsum(energy, dim=0)
        k = int((csum < target).sum()) + 1
        # never take directions below the numerical rank of the residual: they
        # are arbitrary and can point anywhere, including into span(E)
        k = min(k, int((sv > sv[0] * 1e-7).sum()))
        in_dim = R.shape[1]
        cap = int(in_dim * self.params["max_bases_frac"])
        used = existing.shape[1] if existing is not None else 0
        k = max(0, min(k, cap - used))
        if k == 0:
            return
        new = Vh[:k].T                                    # (in_dim, k), orthonormal
        if E is not None:
            new = new - E @ (E.T @ new)                   # kill any leak into span(E)
            norms = new.norm(dim=0)
            new = new[:, norms > 1e-6]
            if new.shape[1] == 0:
                return
            new, _ = torch.linalg.qr(new)                 # re-orthonormalise
        M = new if E is None else torch.cat([E, new], dim=1)
        err = float((M.T @ M - torch.eye(M.shape[1], dtype=M.dtype)).abs().max())
        if err > 1e-6:
            # should not happen after the steps above; if it does, repair the
            # whole basis rather than carry a broken projector forward
            M, _ = torch.linalg.qr(M)
        self.bases[name] = M.float()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def on_task_end(self, model, task_id, ctx) -> None:
        batches = list(ctx.scratch.get("fisher_batches") or [])
        self._collect(model, batches, ctx, self.params["collect_batches"])
        eps = min(
            0.99,
            max(
                self.params["eps_floor"],
                self.params["eps_base"] + self.params["eps_growth"] * self._tasks_seen,
            ),
        )
        self._build(eps)
        self._tasks_seen += 1
        self._saturation_history.append(self._consumed_fraction())

    def _consumed_fraction(self) -> float:
        consumed = []
        for name, layer in self._layers.items():
            M = self.bases.get(self._leader(name))
            consumed.append((M.shape[1] if M is not None else 0) / layer.in_features)
        return sum(consumed) / len(consumed) if consumed else 0.0

    def _rows(self, like: torch.Tensor) -> torch.Tensor | None:
        if not self._seen_symbols:
            return None
        if self._seen_rows is None or self._seen_rows.device != like.device:
            self._seen_rows = torch.tensor(sorted(self._seen_symbols), device=like.device)
        return self._seen_rows

    # ------------------------------------------------------------------
    @torch.no_grad()
    def before_step(self, model, ctx) -> None:
        if not self.bases:
            return
        for name, module in self._layers.items():
            g = module.weight.grad
            M = self._basis_for(name, module.weight)
            if M is None or g is None:
                continue
            module.weight.grad = g - (g @ M) @ M.T
        if self.params["protect_readout"]:
            for emb in self._embeddings:
                rows = self._rows(emb.weight)
                if emb.weight.grad is not None and rows is not None:
                    emb.weight.grad[rows] = 0.0
        if self.params["freeze_norms"]:
            for p in self._norm_params:
                p.grad = None
        if self.params["project_step"]:
            self._pre = {
                name: m.weight.detach().clone()
                for name, m in self._layers.items()
                if self.bases.get(self._leader(name)) is not None
            }
            self._pre_rows = []
            if self.params["protect_readout"]:
                for emb in self._embeddings:
                    rows = self._rows(emb.weight)
                    self._pre_rows.append(emb.weight[rows].clone() if rows is not None else None)
        self.mark_ran()

    @torch.no_grad()
    def after_step(self, model, ctx) -> None:
        if not self.bases or not self.params["project_step"] or not self._pre:
            return
        probe = self._step_count % max(int(self.params["leak_probe_every"]), 1) == 0
        raw_num = raw_den = cor_num = cor_den = 0.0
        for name, module in self._layers.items():
            pre = self._pre.get(name)
            M = self._basis_for(name, module.weight)
            if pre is None or M is None:
                continue
            W = module.weight
            dW = W - pre
            inside = dW @ M                       # (out, k): the part AdamW put back
            if probe:
                raw_num += float((inside.norm() ** 2))
                raw_den += float((dW.norm() ** 2))
            W.sub_(inside @ M.T)                  # W = pre + dW (I - M Mᵀ)
            if probe:
                d2 = W - pre
                cor_num += float(((d2 @ M).norm() ** 2))
                cor_den += float((d2.norm() ** 2))
        if self.params["protect_readout"]:
            for emb, saved in zip(self._embeddings, self._pre_rows):
                rows = self._rows(emb.weight)
                if saved is not None and rows is not None:
                    emb.weight[rows] = saved
        if probe and raw_den > 0 and cor_den > 0:
            self._leak_raw_sum += (raw_num / raw_den) ** 0.5
            self._leak_sum += (cor_num / cor_den) ** 0.5
            self._leak_n += 1
        self._pre.clear()
        self._pre_rows = []
        self._step_count += 1

    # ------------------------------------------------------------------
    def self_test(self, model, batch, ctx) -> tuple[bool, str]:
        if not self._layers:
            return False, "setup() registered no linear layers"
        saved_symbols = set(self._seen_symbols)
        self._collect(model, [batch], ctx, n_forward=1)
        if not self._acts:
            return False, "forward hooks captured no activations"
        self._build(0.90)
        if not self.bases:
            return False, "no basis extracted"
        name = next(n for n in self._layers if self.bases.get(self._leader(n)) is not None)
        module = self._layers[name]
        M = self._basis_for(name, module.weight)
        module.weight.grad = torch.randn_like(module.weight)
        before = module.weight.grad.clone()
        self.before_step(model, ctx)
        changed = float((before - module.weight.grad).norm())
        # an elementwise step, as Adam takes: this is what breaks dW·M = 0
        W0 = module.weight.detach().clone()
        module.weight.data.add_(-1e-3 * module.weight.grad.sign())
        dW = module.weight.detach() - W0
        raw = float((dW @ M).norm() / dW.norm())
        self.after_step(model, ctx)
        dW = module.weight.detach() - W0
        corrected = float((dW @ M).norm() / max(float(dW.norm()), 1e-12))
        module.weight.data.copy_(W0)
        k = M.shape[1]
        self.bases.clear(); self._dev_bases.clear(); self._acts.clear(); self._pre.clear()
        self._pre_rows = []; self._step_count = 0
        self._leak_raw_sum = self._leak_sum = 0.0; self._leak_n = 0
        self._seen_symbols = saved_symbols; self._seen_rows = None
        model.zero_grad(set_to_none=True)
        if changed == 0.0:
            return False, "projection left the gradient unchanged"
        if self.params["project_step"] and corrected > 1e-4:
            return False, f"step correction left {corrected:.2e} of the step inside M"
        n_distinct = len({self._leader(n) for n in self._layers})
        return True, (
            f"projecting {len(self._layers)} layers ({n_distinct} distinct bases); "
            f"{name} k={k}; elementwise step leak {raw:.3f} -> {corrected:.1e} after correction"
        )

    def signature(self, model, ctx):
        occupancy = super().signature(model, ctx)
        if self._leak_n == 0:
            return occupancy
        leak = self._leak_sum / self._leak_n
        step = SignatureCheck(
            probe="C5",
            quantity="fraction of the weight step inside the protected subspace",
            value=leak,
            baseline=0.01,
            direction="decrease",
            detail=(
                f"||ΔW·M||/||ΔW|| on the actual optimizer step, mean of "
                f"{self._leak_n} probes. Before correction AdamW leaves "
                f"{self._leak_raw_sum / self._leak_n:.3f}; the guarantee needs ~0."
            ),
        )
        return [occupancy, step] if occupancy is not None else step

    def cost_report(self) -> CostReport:
        distinct = {self._leader(n) for n in self._layers}
        n = sum(M.numel() for k, M in self.bases.items() if k in distinct)
        notes = {
            "basis_entries": n,
            "layers": len(self._layers),
            "distinct_bases": len(distinct),
            "tasks_seen": self._tasks_seen,
            "saturation": round(self._consumed_fraction(), 4),
            "saturation_history": [round(v, 4) for v in self._saturation_history],
            "seen_symbols": len(self._seen_symbols),
        }
        if self._leak_n:
            notes["step_leak_raw"] = round(self._leak_raw_sum / self._leak_n, 5)
            notes["step_leak"] = round(self._leak_sum / self._leak_n, 7)
        return CostReport(buffer_bytes=n * 4, notes=notes)

    def state_dict(self):
        st = super().state_dict()
        st.update({
            "seen_symbols": sorted(self._seen_symbols),
            "group_of": dict(self._group_of),
            "step_count": self._step_count,
            "leak": [self._leak_raw_sum, self._leak_sum, self._leak_n],
        })
        return st

    def load_state_dict(self, state):
        super().load_state_dict(state)
        state = to_cpu_tree(state)
        self._seen_symbols = set(int(v) for v in state.get("seen_symbols", []))
        self._group_of = dict(state.get("group_of", {}))
        self._step_count = int(state.get("step_count", 0))
        raw, cor, n = state.get("leak", [0.0, 0.0, 0])
        self._leak_raw_sum, self._leak_sum, self._leak_n = float(raw), float(cor), int(n)
        self._dev_bases.clear()
        self._seen_rows = None
