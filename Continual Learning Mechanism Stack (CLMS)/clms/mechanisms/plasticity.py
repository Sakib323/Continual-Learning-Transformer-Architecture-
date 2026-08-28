"""F12 · Plasticity maintenance.

`shrink_perturb`     one-line approximation: scale weights toward init, add noise
`continual_backprop` utility-based selective reinitialisation of hidden units

Dohare et al. found Adam to be the worst case tested — plasticity plummets and
effective rank collapses — and that only continual reinjection of diversity held
performance flat over 800 tasks. Gradient descent alone is not enough, which is
the strongest claim in the whole reading list.

Signature (probe A1): effective rank should stay flat instead of collapsing.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..base import CostReport, Mechanism, SignatureCheck, to_cpu_tree
from ..registry import register
from ..eval.probes import effective_rank


@register
class ShrinkAndPerturb(Mechanism):
    name = "shrink_perturb"
    surfaces = ("O",)
    family = "F12"
    paper = "2306.13812"
    order = 80

    defaults = {
        "shrink": 0.999,     # multiplicative pull toward zero
        "noise_std": 0.001,  # gaussian injection, scaled by layer weight magnitude
        "every": 100,        # steps between applications
    }

    def __init__(self, **kw):
        super().__init__(**kw)
        self._applied = 0

    @torch.no_grad()
    def after_step(self, model, ctx) -> None:
        if ctx.step % self.params["every"] != 0:
            return
        for p in model.parameters():
            if p.requires_grad and p.dim() > 1:
                p.mul_(self.params["shrink"])
                noise = torch.randn(p.shape, generator=self.rng(ctx),
                                    dtype=p.dtype).to(p.device)
                p.add_(noise * self.params["noise_std"] * p.abs().mean())
        self._applied += 1
        self.mark_ran()

    def self_test(self, model, batch, ctx) -> tuple[bool, str]:
        before = torch.cat([p.detach().reshape(-1) for p in model.parameters() if p.dim() > 1])
        saved_step = ctx.step
        ctx.step = 0
        self.after_step(model, ctx)
        ctx.step = saved_step
        after = torch.cat([p.detach().reshape(-1) for p in model.parameters() if p.dim() > 1])
        delta = float((after - before).abs().max())
        if delta == 0.0:
            return False, "no parameter changed after an application"
        return True, f"max |delta| = {delta:.3g}"

    def cost_report(self) -> CostReport:
        return CostReport(notes={"applications": self._applied})


@register
class ContinualBackprop(Mechanism):
    """Selective reinitialisation of low-utility MLP hidden units.

    Utility is contribution divided by adaptation cost:

        y = |h - h_mean| * sum_k |w_out[k]|  /  sum_j |w_in[j]|

    tracked as a bias-corrected running average. The lowest-utility fraction rho
    is replaced each step; new units get **zero outgoing weights** so the
    function does not jump, and are protected for `maturity` steps before they
    can be selected again.
    """

    name = "continual_backprop"
    surfaces = ("A", "O")   # A only to observe activations, not to change the graph
    family = "F12"
    paper = "2306.13812"
    # Mechanisms that freeze the backbone leave nothing for CBP to recycle, and
    # GPM consumes gradient directions that CBP is trying to regenerate — the
    # two work against each other and any measured effect is unattributable.
    conflicts = ("lora", "l2p", "olora", "gpm")
    order = 85

    defaults = {
        "rho": 1e-4,        # replacement rate per step, per layer
        "maturity": 100,    # steps a new unit is protected
        "decay": 0.99,      # running-average decay for utility
        "init_std": 0.02,
    }

    def __init__(self, **kw):
        super().__init__(**kw)
        self.util: dict[int, torch.Tensor] = {}
        self.age: dict[int, torch.Tensor] = {}
        self._act: dict[int, torch.Tensor] = {}
        self._steps = 0
        self._replaced = 0
        self._mlps: dict[int, nn.Module] = {}

    # --- capture the widest representation, unchanged ------------------
    def observe(self, name, layer_idx, tensor):
        if name == "mlp_hidden":
            self._act[layer_idx] = tensor.detach().abs().mean(dim=(0, 1)).cpu()
        return None   # never modifies the activation

    def setup(self, model, cfg, ctx) -> None:
        for i, layer in enumerate(model.model.layers):
            mlp = layer.mlp
            if all(hasattr(mlp, a) for a in ("gate_proj", "up_proj", "down_proj")):
                self._mlps[i] = mlp
                n_units = mlp.down_proj.weight.shape[1]
                self.util[i] = torch.zeros(n_units)
                self.age[i] = torch.zeros(n_units, dtype=torch.long)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def after_step(self, model, ctx) -> None:
        self._steps += 1
        decay = self.params["decay"]

        for idx, mlp in self._mlps.items():
            act = self._act.get(idx)
            if act is None:
                continue
            self.age[idx] += 1

            w_out = mlp.down_proj.weight            # (hidden, intermediate)
            w_in = mlp.gate_proj.weight             # (intermediate, hidden)
            contribution = act * w_out.abs().sum(dim=0).cpu()
            adaptation = w_in.abs().sum(dim=1).cpu().clamp_min(1e-8)
            y = contribution / adaptation

            self.util[idx] = decay * self.util[idx] + (1 - decay) * y
            bias_correction = 1.0 - decay ** max(self._steps, 1)
            util = self.util[idx] / bias_correction

            n_units = util.numel()
            draw = float(torch.rand(1, generator=self.rng(ctx)).item())
            n_replace = max(1, int(n_units * self.params["rho"])) if \
                draw < n_units * self.params["rho"] else 0
            if n_replace == 0:
                continue

            eligible = self.age[idx] > self.params["maturity"]
            if not eligible.any():
                continue
            masked = util.clone()
            masked[~eligible] = float("inf")
            victims = torch.topk(masked, k=min(n_replace, int(eligible.sum())),
                                 largest=False).indices

            std = self.params["init_std"]
            for u in victims.tolist():
                mlp.gate_proj.weight[u].normal_(0.0, std)
                mlp.up_proj.weight[u].normal_(0.0, std)
                mlp.down_proj.weight[:, u].zero_()   # no function jump
                self.util[idx][u] = 0.0
                self.age[idx][u] = 0
                self._replaced += 1
            self.mark_ran()

    # ------------------------------------------------------------------
    def self_test(self, model, batch, ctx) -> tuple[bool, str]:
        if not self._mlps:
            return False, "setup() found no MLP layers to track"
        device = ctx.device
        model(batch["input_ids"].to(device))
        if not self._act:
            return False, "no mlp_hidden activations captured — observe() not wired"
        saved_rho, saved_mat = self.params["rho"], self.params["maturity"]
        self.params["rho"], self.params["maturity"] = 1.0, -1
        before = next(iter(self._mlps.values())).down_proj.weight.detach().clone()
        self.after_step(model, ctx)
        after = next(iter(self._mlps.values())).down_proj.weight
        self.params["rho"], self.params["maturity"] = saved_rho, saved_mat
        if torch.equal(before, after):
            return False, "forced replacement changed no weights"
        return True, f"tracking {len(self._mlps)} MLPs, replacement verified"

    def signature(self, model, ctx) -> SignatureCheck | None:
        recorder = ctx.scratch.get("recorder")
        if recorder is None:
            return None
        layers = recorder.layers("mlp_hidden")
        if not layers:
            return None
        ranks = [
            effective_rank(H) for i in layers
            if (H := recorder.stacked("mlp_hidden", i)) is not None
        ]
        ranks = [r for r in ranks if r == r]  # drop nan
        if not ranks:
            return None
        return SignatureCheck(
            probe="A1",
            quantity="mean effective rank of MLP hidden activations",
            value=sum(ranks) / len(ranks),
            direction="hold",
            detail="continual backprop should keep this flat rather than collapsing",
        )

    def cost_report(self) -> CostReport:
        return CostReport(
            buffer_bytes=sum(u.numel() * 4 * 2 for u in self.util.values()),
            notes={"units_replaced": self._replaced, "layers_tracked": len(self._mlps)},
        )

    def state_dict(self):
        return {"util": self.util, "age": self.age, "steps": self._steps}

    def load_state_dict(self, state):
        state = to_cpu_tree(state)
        self.util = state.get("util", {})
        self.age = state.get("age", {})
        self._steps = state.get("steps", 0)
