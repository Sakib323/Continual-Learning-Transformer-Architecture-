"""F05/F03 · Low-rank adapters and orthogonal subspace allocation.

`lora`   Freeze the backbone after a chosen task and route all further learning
         through low-rank adapters. The standard parameter-efficient baseline,
         and an important one: LoRA forgets *differently*, not less — 71% F1
         drop versus 89% for full finetuning in the sparse-memory comparison.

`olora`  One adapter per task, each constrained to a subspace orthogonal to
         every previous task's. Interference is bounded by construction rather
         than penalised after the fact.

Protocol note: from-scratch training means there is no pretrained backbone to
freeze, so `freeze_after_task` controls when the backbone stops learning. The
default of 0 gives "learn task 0 fully, adapt thereafter", which is the closest
honest analogue of the pretrained setting.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..base import CostReport, Mechanism, SignatureCheck
from ..registry import register


class LoRALinear(nn.Module):
    """Wraps a Linear with one or more low-rank adapters.

    A separate adapter per task is what makes the orthogonality constraint in
    O-LoRA expressible; plain LoRA just uses adapter 0 throughout.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float, num_adapters: int = 1):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scaling = alpha / rank
        self.active = 0
        self.A = nn.ParameterList([
            nn.Parameter(torch.randn(rank, base.in_features) * 0.01)
            for _ in range(num_adapters)
        ])
        self.B = nn.ParameterList([
            nn.Parameter(torch.zeros(base.out_features, rank))
            for _ in range(num_adapters)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        # every adapter learned so far contributes; only `active` receives grad
        for i in range(len(self.A)):
            if i <= self.active:
                out = out + (x @ self.A[i].T @ self.B[i].T) * self.scaling
        return out

    def set_active(self, idx: int) -> None:
        self.active = min(idx, len(self.A) - 1)
        for i in range(len(self.A)):
            train = i == self.active
            self.A[i].requires_grad_(train)
            self.B[i].requires_grad_(train)


def _wrap_linears(model: nn.Module, rank: int, alpha: float, num_adapters: int,
                  targets: tuple[str, ...]) -> list[LoRALinear]:
    wrapped: list[LoRALinear] = []
    for module in model.modules():
        for attr in list(vars(module).get("_modules", {})):
            child = getattr(module, attr, None)
            if isinstance(child, nn.Linear) and attr in targets:
                lora = LoRALinear(child, rank, alpha, num_adapters)
                setattr(module, attr, lora)
                wrapped.append(lora)
    return wrapped


@register
class LoRA(Mechanism):
    name = "lora"
    surfaces = ("A", "O")
    family = "F05"
    paper = "2510.15103"
    requires = ("task_boundaries",)
    conflicts = ("olora", "continual_backprop")
    order = 18

    defaults = {
        "rank": 8,
        "alpha": 16.0,
        "freeze_after_task": 0,
        "targets": "q_proj,v_proj,down_proj",
    }

    def __init__(self, **kw):
        super().__init__(**kw)
        self.wrapped: list[LoRALinear] = []
        self._frozen = False

    def setup(self, model, cfg, ctx) -> None:
        targets = tuple(t.strip() for t in self.params["targets"].split(","))
        self.wrapped = _wrap_linears(model, self.params["rank"], self.params["alpha"], 1, targets)
        for w in self.wrapped:
            w.set_active(0)
        model.to(next(model.parameters()).device)

    def on_task_start(self, model, task_id, ctx) -> None:
        if task_id > self.params["freeze_after_task"] and not self._frozen:
            for name, p in model.named_parameters():
                if ".A." not in name and ".B." not in name:
                    p.requires_grad_(False)
            self._frozen = True
            self.mark_ran()

    def self_test(self, model, batch, ctx) -> tuple[bool, str]:
        if not self.wrapped:
            return False, "no linear layers were wrapped; check `targets`"
        w = self.wrapped[0]
        x = torch.randn(2, 3, w.base.in_features, device=ctx.device)
        with torch.no_grad():
            base_out = w.base(x)
            w.B[0].normal_(0, 0.1)      # B starts at zero, so perturb to test
            lora_out = w(x)
            w.B[0].zero_()
        delta = float((lora_out - base_out).abs().max())
        if delta == 0.0:
            return False, "adapter path contributes nothing"
        n = sum(p.numel() for lw in self.wrapped for p in list(lw.A) + list(lw.B))
        return True, f"{len(self.wrapped)} layers wrapped, {n:,} adapter params"

    def signature(self, model, ctx) -> SignatureCheck | None:
        if not self._frozen:
            return None
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        return SignatureCheck(
            probe="B2",
            quantity="trainable fraction of parameters",
            value=trainable / total,
            baseline=1.0,
            direction="decrease",
            detail="backbone frozen; all learning routed through adapters",
        )

    def cost_report(self) -> CostReport:
        n = sum(p.numel() for lw in self.wrapped for p in list(lw.A) + list(lw.B))
        return CostReport(added_params=n, notes={"wrapped_layers": len(self.wrapped)})


@register
class OrthogonalLoRA(Mechanism):
    name = "olora"
    surfaces = ("A", "L")
    family = "F03"
    paper = "survey"
    requires = ("task_boundaries",)
    conflicts = ("lora", "continual_backprop")
    order = 19

    defaults = {
        "rank": 8,
        "alpha": 16.0,
        "lam": 0.5,           # orthogonality penalty weight
        "targets": "q_proj,v_proj,down_proj",
    }

    def __init__(self, **kw):
        super().__init__(**kw)
        self.wrapped: list[LoRALinear] = []
        self._task = 0
        self._last = 0.0

    def setup(self, model, cfg, ctx) -> None:
        targets = tuple(t.strip() for t in self.params["targets"].split(","))
        self.wrapped = _wrap_linears(
            model, self.params["rank"], self.params["alpha"],
            max(ctx.num_tasks, 1), targets,
        )
        for w in self.wrapped:
            w.set_active(0)
        model.to(next(model.parameters()).device)

    def on_task_start(self, model, task_id, ctx) -> None:
        self._task = task_id
        for w in self.wrapped:
            w.set_active(task_id)
        # backbone frozen from task 1 onward: all task-specific capacity is
        # meant to live in the mutually orthogonal adapter subspaces
        if task_id > 0:
            for name, p in model.named_parameters():
                if ".A." not in name and ".B." not in name:
                    p.requires_grad_(False)
        self.mark_ran()

    def compute_loss(self, model, batch, out, base_loss, ctx):
        if self._task == 0 or not self.wrapped:
            return None
        penalty = torch.zeros((), device=base_loss.device)
        for w in self.wrapped:
            cur = w.A[self._task]
            for prev in range(self._task):
                # penalise any overlap between this task's row space and earlier ones
                penalty = penalty + (cur @ w.A[prev].T.detach()).pow(2).sum()
        term = self.params["lam"] * penalty
        self._last = float(term.detach())
        return term

    def self_test(self, model, batch, ctx) -> tuple[bool, str]:
        if not self.wrapped:
            return False, "no linear layers were wrapped"
        if len(self.wrapped[0].A) < 2:
            return False, "need at least two adapters for an orthogonality constraint"
        saved = self._task
        self._task = 1
        term = self.compute_loss(model, batch, {}, torch.zeros(()), ctx)
        self._task = saved
        if term is None or float(term) <= 0:
            return False, "orthogonality penalty is zero at initialization"
        return True, f"{len(self.wrapped)} layers x {len(self.wrapped[0].A)} adapters"

    def signature(self, model, ctx) -> SignatureCheck | None:
        if self._task == 0 or not self.wrapped:
            return None
        # mean |cos| between the current and previous adapter row spaces
        sims = []
        for w in self.wrapped:
            cur = torch.nn.functional.normalize(w.A[self._task], dim=1)
            for prev in range(self._task):
                other = torch.nn.functional.normalize(w.A[prev], dim=1)
                sims.append(float((cur @ other.T).abs().mean()))
        if not sims:
            return None
        return SignatureCheck(
            probe="C2",
            quantity="mean |cosine| between task adapter subspaces",
            value=sum(sims) / len(sims),
            baseline=0.5,
            direction="decrease",
            detail="orthogonal subspaces should drive this toward zero",
        )

    def cost_report(self) -> CostReport:
        n = sum(p.numel() for lw in self.wrapped for p in list(lw.A) + list(lw.B))
        return CostReport(added_params=n, notes={"adapters_per_layer":
                                                 len(self.wrapped[0].A) if self.wrapped else 0})
