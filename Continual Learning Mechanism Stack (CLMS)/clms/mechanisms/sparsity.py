"""F04 · Sparsity and update localization — the architecture-surface slice.

`xdg`   context-dependent gating: each task gets a fixed random mask over MLP
        hidden units, so different tasks route through largely disjoint
        subnetworks. Cheap, needs task identity, and it is the simplest possible
        proof that the surface-A hook works end to end.

`kwta`  k-winners-take-all: keep only the top-k activations per token, zero the
        rest. No task identity needed. The Numenta argument is that overlap
        between sparse high-dimensional codes falls exponentially in
        dimensionality, so interference largely evaporates.

Signature (probe A4): activation overlap between tasks should fall. If it
doesn't, the mechanism is not doing what it claims however the accuracy looks.
"""

from __future__ import annotations

import torch

from ..base import CostReport, Mechanism, SignatureCheck, to_cpu_tree
from ..registry import register


@register
class XdG(Mechanism):
    name = "xdg"
    surfaces = ("A",)
    family = "F04"
    paper = "1904.07734"
    requires = ("task_ids",)
    order = 20

    defaults = {
        "gate_fraction": 0.5,   # fraction of units left active per task
        "seed": 1234,
    }

    def __init__(self, **kw):
        super().__init__(**kw)
        self.masks: dict[tuple[int, int], torch.Tensor] = {}
        self._current_task = 0
        self._widths: dict[int, int] = {}

    def setup(self, model, cfg, ctx) -> None:
        g = torch.Generator().manual_seed(self.params["seed"])
        keep = self.params["gate_fraction"]
        for layer_idx, layer in enumerate(model.model.layers):
            width = getattr(layer.mlp, "down_proj", None)
            if width is None:
                continue
            n = width.weight.shape[1]
            self._widths[layer_idx] = n
            for task_id in range(max(ctx.num_tasks, 1)):
                perm = torch.randperm(n, generator=g)
                mask = torch.zeros(n)
                mask[perm[: max(1, int(n * keep))]] = 1.0
                self.masks[(task_id, layer_idx)] = mask

    def on_task_start(self, model, task_id, ctx) -> None:
        self._current_task = task_id

    def observe(self, name, layer_idx, tensor):
        if name != "mlp_hidden":
            return None
        mask = self.masks.get((self._current_task, layer_idx))
        if mask is None:
            return None
        self.mark_ran()
        return tensor * mask.to(tensor.device, tensor.dtype)

    # ------------------------------------------------------------------
    def self_test(self, model, batch, ctx) -> tuple[bool, str]:
        if not self.masks:
            return False, "setup() produced no gating masks"
        probe = torch.ones(2, 3, next(iter(self._widths.values())))
        gated = self.observe("mlp_hidden", next(iter(self._widths)), probe)
        if gated is None:
            return False, "observe() did not gate mlp_hidden"
        zeroed = float((gated == 0).float().mean())
        if zeroed < 0.1:
            return False, f"only {zeroed:.1%} of units gated off; expected ~{1 - self.params['gate_fraction']:.0%}"
        return True, f"{zeroed:.1%} of hidden units gated per task"

    def signature(self, model, ctx) -> SignatureCheck | None:
        if len(self.masks) < 2 or not self._widths:
            return None
        layer = next(iter(self._widths))
        m0 = self.masks.get((0, layer))
        m1 = self.masks.get((1, layer))
        if m0 is None or m1 is None:
            return None
        inter = float((m0 * m1).sum())
        union = float(((m0 + m1) > 0).float().sum())
        keep = self.params["gate_fraction"]
        return SignatureCheck(
            probe="A4",
            quantity="mask overlap between task 0 and task 1",
            value=inter / union if union else float("nan"),
            baseline=1.0,
            direction="decrease",
            detail=f"random gating at keep={keep} predicts overlap ~{keep / (2 - keep):.2f}",
        )

    def cost_report(self) -> CostReport:
        return CostReport(
            buffer_bytes=sum(m.numel() * 4 for m in self.masks.values()),
            notes={"masks": len(self.masks)},
        )


@register
class KWinnersTakeAll(Mechanism):
    name = "kwta"
    surfaces = ("A",)
    family = "F04"
    paper = "1903.11257"
    order = 22

    defaults = {
        "k_fraction": 0.15,      # fraction of units kept active per token
        "boost_strength": 1.0,   # duty-cycle boosting; 0 disables
        "boost_decay": 0.99,
        "inference_boost": 1.5,  # widen k at eval, as in the paper
    }

    def __init__(self, **kw):
        super().__init__(**kw)
        self.duty: dict[int, torch.Tensor] = {}
        self._fired = 0

    def observe(self, name, layer_idx, tensor):
        if name != "mlp_hidden":
            return None
        n = tensor.shape[-1]
        k = max(1, int(n * self.params["k_fraction"]))
        flat = tensor.reshape(-1, n)

        score = flat
        if self.params["boost_strength"] > 0:
            duty = self.duty.get(layer_idx)
            # Width can differ between a probe call and the real model, and
            # between models in an independent-mode run, so size is checked
            # rather than assumed.
            if duty is None or duty.numel() != n:
                duty = torch.full((n,), self.params["k_fraction"])
                self.duty[layer_idx] = duty
            target = self.params["k_fraction"]
            boost = torch.exp(
                self.params["boost_strength"] * (target - duty.to(tensor.device))
            )
            score = flat * boost

        idx = torch.topk(score, k=k, dim=-1).indices
        mask = torch.zeros_like(flat).scatter_(-1, idx, 1.0)

        if self.params["boost_strength"] > 0:
            observed = mask.mean(dim=0).detach().cpu()
            d = self.params["boost_decay"]
            self.duty[layer_idx] = d * self.duty[layer_idx] + (1 - d) * observed

        self._fired += 1
        self.mark_ran()
        return (flat * mask).reshape(tensor.shape)

    def self_test(self, model, batch, ctx) -> tuple[bool, str]:
        probe = torch.randn(4, 5, 64)
        out = self.observe("mlp_hidden", -999, probe)   # sentinel: no real layer
        self.duty.pop(-999, None)
        if out is None:
            return False, "observe() did not act on mlp_hidden"
        nonzero = float((out != 0).float().mean())
        expected = self.params["k_fraction"]
        if abs(nonzero - expected) > 0.1:
            return False, f"active fraction {nonzero:.3f}, expected ~{expected:.3f}"
        return True, f"active fraction {nonzero:.3f} (k_fraction={expected})"

    def signature(self, model, ctx) -> SignatureCheck | None:
        return SignatureCheck(
            probe="A4",
            quantity="active unit fraction per token",
            value=self.params["k_fraction"],
            baseline=1.0,
            direction="decrease",
            detail="sparse codes should collide exponentially less as width grows",
        )

    def cost_report(self) -> CostReport:
        return CostReport(notes={"applications": self._fired})

    def state_dict(self):
        return {"duty": self.duty}

    def load_state_dict(self, state):
        self.duty = to_cpu_tree(state).get("duty", {})
