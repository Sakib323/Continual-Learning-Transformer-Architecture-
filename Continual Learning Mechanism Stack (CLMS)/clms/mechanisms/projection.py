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
            self.params["eps_base"] + self.params["eps_growth"] * self._tasks_seen,
        )
        for name, parts in self._acts.items():
            R = torch.cat(parts, dim=0)               # (samples, in_dim)
            if R.shape[0] < 2:
                continue
            self._extend_basis(name, R, eps)
        self._acts.clear()
        self._tasks_seen += 1

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
        return SignatureCheck(
            probe="C2",
            quantity="fraction of gradient directions consumed",
            value=frac,
            baseline=self.params["max_bases_frac"],
            direction="decrease",
            detail="rises monotonically with tasks; at the cap there is no free "
                   "subspace left and plasticity is gone",
        )

    def cost_report(self) -> CostReport:
        n = sum(M.numel() for M in self.bases.values())
        return CostReport(
            buffer_bytes=n * 4,
            notes={"basis_entries": n, "layers": len(self._layers),
                   "tasks_seen": self._tasks_seen},
        )

    def state_dict(self):
        return {"bases": self.bases, "tasks_seen": self._tasks_seen}

    def load_state_dict(self, state):
        state = to_cpu_tree(state)
        self.bases = state.get("bases", {})
        self._tasks_seen = state.get("tasks_seen", 0)
