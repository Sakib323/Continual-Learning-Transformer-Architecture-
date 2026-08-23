"""F02 · Function- and weight-space regularization.

`lwf`  Learning without Forgetting. Keeps a frozen copy of the model from the
       last boundary and distills its outputs on *current-task* inputs. Needs no
       stored data at all — the new task's own inputs carry the old function.

`si`   Synaptic Intelligence. Importance is a path integral accumulated during
       training rather than a Fisher estimate computed at the boundary, so it
       costs one extra vector per parameter and no extra forward passes.

Signatures: LwF probes D2 (logit drift on a frozen probe set should fall);
SI probes B1, like EWC.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from ..base import CostReport, Mechanism, SignatureCheck, to_cpu_tree
from ..registry import register


@register
class LearningWithoutForgetting(Mechanism):
    name = "lwf"
    surfaces = ("L",)
    family = "F02"
    paper = "1606.09282"
    requires = ("task_boundaries",)
    order = 32

    defaults = {
        "lam": 1.0,          # loss balance weight (lambda_o in the paper)
        "temperature": 2.0,  # T > 1 raises the weight of small logits
    }

    def __init__(self, **kw):
        super().__init__(**kw)
        self.teacher = None
        self._last = 0.0

    # ------------------------------------------------------------------
    def compute_loss(self, model, batch, out, base_loss, ctx):
        if self.teacher is None:
            return None
        with torch.no_grad():
            t_logits = self.teacher(batch["input_ids"])["logits"]
        T = self.params["temperature"]
        # modified cross-entropy of Hinton et al.; T^2 keeps gradient scale
        # comparable to the unsoftened term
        loss = F.kl_div(
            F.log_softmax(out["logits"] / T, dim=-1),
            F.log_softmax(t_logits / T, dim=-1),
            log_target=True,
            reduction="batchmean",
        ) * (T * T)
        term = self.params["lam"] * loss
        self._last = float(term.detach())
        self.mark_ran()
        return term

    def on_task_end(self, model, task_id, ctx) -> None:
        self.teacher = copy.deepcopy(model).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    def self_test(self, model, batch, ctx) -> tuple[bool, str]:
        saved = self.teacher
        self.teacher = copy.deepcopy(model).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        # perturb the student so the two functions genuinely differ
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.01)
            out = model(batch["input_ids"])
        term = self.compute_loss(model, batch, out, torch.zeros(()), ctx)
        self.teacher = saved
        if term is None:
            return False, "no distillation term with a teacher present"
        if float(term) <= 0:
            return False, f"distillation term is {float(term)}, expected > 0"
        return True, f"distillation active ({float(term):.4g})"

    def signature(self, model, ctx) -> SignatureCheck | None:
        if self.teacher is None:
            return None
        return SignatureCheck(
            probe="D2",
            quantity="LwF distillation term",
            value=self._last,
            direction="hold",
            detail="non-zero means the old function is being held in place",
        )

    def cost_report(self) -> CostReport:
        n = sum(p.numel() for p in self.teacher.parameters()) if self.teacher else 0
        return CostReport(added_params=0, buffer_bytes=n * 4,
                          notes={"teacher_params": n})

    def state_dict(self):
        return {"teacher": self.teacher.state_dict() if self.teacher else None}

    def load_state_dict(self, state):
        self._pending_teacher = state.get("teacher")   # applied by setup on resume


@register
class SynapticIntelligence(Mechanism):
    name = "si"
    surfaces = ("L", "O")
    family = "F02"
    paper = "1904.07734"
    requires = ("task_boundaries",)
    order = 31

    defaults = {
        "c": 0.1,        # penalty strength
        "xi": 0.1,       # damping, prevents division by ~0 displacement
    }

    def __init__(self, **kw):
        super().__init__(**kw)
        self.omega: dict[str, torch.Tensor] = {}    # consolidated importance
        self.w: dict[str, torch.Tensor] = {}        # running path integral
        self.anchor: dict[str, torch.Tensor] = {}
        self.task_start: dict[str, torch.Tensor] = {}
        self._prev: dict[str, torch.Tensor] = {}
        self._grad: dict[str, torch.Tensor] = {}

    def setup(self, model, cfg, ctx) -> None:
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.w[n] = torch.zeros_like(p, device="cpu")
                self.task_start[n] = p.detach().clone().cpu()

    # ------------------------------------------------------------------
    def compute_loss(self, model, batch, out, base_loss, ctx):
        if not self.omega:
            return None
        penalty = torch.zeros((), device=base_loss.device)
        for n, p in model.named_parameters():
            if n in self.omega:
                penalty = penalty + (
                    self.omega[n].to(p.device) * (p - self.anchor[n].to(p.device)) ** 2
                ).sum()
        self.mark_ran()
        return self.params["c"] * penalty

    def before_step(self, model, ctx) -> None:
        # snapshot gradient and parameters; the path integral needs both halves
        self._grad = {
            n: p.grad.detach().clone() for n, p in model.named_parameters()
            if p.requires_grad and p.grad is not None
        }
        self._prev = {
            n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad
        }

    @torch.no_grad()
    def after_step(self, model, ctx) -> None:
        # w += -g * delta_theta : the loss decrease this parameter is credited with
        for n, p in model.named_parameters():
            if n in self._grad and n in self._prev:
                delta = p.detach() - self._prev[n]
                self.w[n] += (-self._grad[n] * delta).cpu()
        self.mark_ran()

    @torch.no_grad()
    def on_task_end(self, model, task_id, ctx) -> None:
        xi = self.params["xi"]
        for n, p in model.named_parameters():
            if n not in self.w:
                continue
            total_delta = p.detach().cpu() - self.task_start[n]
            contrib = self.w[n] / (total_delta ** 2 + xi)
            self.omega[n] = self.omega.get(n, torch.zeros_like(contrib)) + contrib.clamp_min(0)
            self.w[n].zero_()
            self.task_start[n] = p.detach().clone().cpu()
        self.anchor = {
            n: p.detach().clone().cpu() for n, p in model.named_parameters() if p.requires_grad
        }

    # ------------------------------------------------------------------
    def self_test(self, model, batch, ctx) -> tuple[bool, str]:
        if not self.w:
            return False, "setup() never ran; path-integral accumulators missing"
        out = model(batch["input_ids"], labels=batch["labels"])
        out["loss"].backward()
        self.before_step(model, ctx)
        if not self._grad:
            return False, "no gradients captured in before_step"
        with torch.no_grad():
            for p in model.parameters():
                if p.requires_grad:
                    p.add_(torch.randn_like(p) * 1e-3)
        self.after_step(model, ctx)
        model.zero_grad(set_to_none=True)
        moved = sum(float(v.abs().sum()) for v in self.w.values())
        if moved == 0.0:
            return False, "path integral did not accumulate"
        return True, f"path integral accumulating (sum|w| = {moved:.4g})"

    def signature(self, model, ctx) -> SignatureCheck | None:
        if not self.omega or not self.anchor:
            return None
        ratios = []
        for n, p in model.named_parameters():
            if n not in self.omega or n not in self.anchor:
                continue
            o = self.omega[n].flatten()
            d = (p.detach().cpu() - self.anchor[n]).abs().flatten()
            if o.numel() < 10:
                continue
            hi, lo = o > torch.quantile(o, 0.9), o < torch.quantile(o, 0.1)
            if hi.any() and lo.any() and d[lo].mean() > 0:
                ratios.append(float(d[hi].mean() / d[lo].mean()))
        if not ratios:
            return None
        return SignatureCheck(
            probe="B1",
            quantity="displacement ratio, high-omega / low-omega params",
            value=sum(ratios) / len(ratios),
            baseline=1.0,
            direction="decrease",
            detail="SI should move parameters it credits with progress less",
        )

    def cost_report(self) -> CostReport:
        n = sum(v.numel() for v in self.w.values())
        return CostReport(buffer_bytes=n * 4 * 3, notes={"tracked_params": n})

    def state_dict(self):
        return {"omega": self.omega, "w": self.w, "anchor": self.anchor,
                "task_start": self.task_start}

    def load_state_dict(self, state):
        state = to_cpu_tree(state)
        self.omega = state.get("omega", {})
        self.w = state.get("w", {})
        self.anchor = state.get("anchor", {})
        self.task_start = state.get("task_start", {})
