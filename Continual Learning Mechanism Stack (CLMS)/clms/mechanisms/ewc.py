"""F02 · Elastic Weight Consolidation and its online variant.

Quadratic penalty pulling each parameter toward its post-task value, weighted by
diagonal Fisher importance:

    L_ewc = (lambda/2) * sum_i F_i * (theta_i - theta*_i)^2

Online EWC keeps one running Fisher decayed by gamma instead of one per task, so
memory stays constant as the stream grows.

Signature (probe B1): displacement on high-Fisher parameters should fall
relative to low-Fisher ones. If it doesn't, EWC is not doing its job whatever
the accuracy says.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..base import CostReport, Mechanism, RunContext, SignatureCheck, to_cpu_tree
from ..registry import register


@register
class EWC(Mechanism):
    name = "ewc"
    surfaces = ("L",)
    family = "F02"
    paper = "1904.07734"
    requires = ("task_boundaries",)
    order = 30

    defaults = {
        "lam": 1000.0,        # penalty strength
        "online": True,       # single decayed Fisher vs one per task
        "gamma": 0.95,        # decay for the online variant
        "fisher_batches": 8,  # batches used to estimate Fisher at a boundary
    }

    def __init__(self, **kw):
        super().__init__(**kw)
        self.fisher: dict[str, torch.Tensor] = {}
        self.anchor: dict[str, torch.Tensor] = {}
        # anchor from the *previous* boundary: the signature probe needs a
        # reference the parameters have actually had a chance to move away from.
        self.prev_anchor: dict[str, torch.Tensor] = {}
        self._penalty_seen = 0.0

    # ------------------------------------------------------------------
    def compute_loss(self, model, batch, out, base_loss, ctx) -> torch.Tensor | None:
        if not self.fisher:
            return None
        penalty = torch.zeros((), device=base_loss.device)
        for n, p in model.named_parameters():
            if n in self.fisher:
                penalty = penalty + (
                    self.fisher[n].to(p.device) * (p - self.anchor[n].to(p.device)) ** 2
                ).sum()
        term = 0.5 * self.params["lam"] * penalty
        self._penalty_seen = float(term.detach())
        return term

    # ------------------------------------------------------------------
    def on_task_end(self, model, task_id, ctx) -> None:
        loader = ctx.scratch.get("fisher_batches")
        if loader is None:
            raise RuntimeError(
                "ewc needs ctx.scratch['fisher_batches'] at the task boundary; "
                "the trainer supplies it"
            )
        new_fisher = self._estimate_fisher(model, loader, ctx.device)
        if self.params["online"] and self.fisher:
            g = self.params["gamma"]
            for n, f in new_fisher.items():
                self.fisher[n] = g * self.fisher.get(n, torch.zeros_like(f)) + f
        else:
            for n, f in new_fisher.items():
                self.fisher[n] = self.fisher.get(n, torch.zeros_like(f)) + f
        self.prev_anchor = self.anchor
        self.anchor = {
            n: p.detach().clone().cpu() for n, p in model.named_parameters() if p.requires_grad
        }

    def _estimate_fisher(self, model, batches, device) -> dict[str, torch.Tensor]:
        fisher = {
            n: torch.zeros_like(p, device="cpu")
            for n, p in model.named_parameters() if p.requires_grad
        }
        n_used = 0
        was_training = model.training
        model.eval()
        for i, batch in enumerate(batches):
            if i >= self.params["fisher_batches"]:
                break
            model.zero_grad(set_to_none=True)
            out = model(batch["input_ids"].to(device), labels=batch["labels"].to(device))
            out["loss"].backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += (p.grad.detach() ** 2).cpu()
            n_used += 1
        model.zero_grad(set_to_none=True)
        if was_training:
            model.train()
        if n_used:
            for n in fisher:
                fisher[n] /= n_used
        return fisher

    # ------------------------------------------------------------------
    def self_test(self, model, batch, ctx) -> tuple[bool, str]:
        """Before any boundary the penalty is legitimately zero, so prove the
        machinery works by planting a synthetic Fisher and checking the loss moves."""
        saved_f, saved_a = self.fisher, self.anchor
        self.fisher = {
            n: torch.ones_like(p).cpu()
            for n, p in list(model.named_parameters())[:2] if p.requires_grad
        }
        self.anchor = {n: torch.zeros_like(f) for n, f in self.fisher.items()}
        dummy = torch.zeros((), requires_grad=True)
        term = self.compute_loss(model, batch, {}, dummy, ctx)
        self.fisher, self.anchor = saved_f, saved_a
        if term is None:
            return False, "compute_loss returned None with a non-empty Fisher"
        if float(term.detach()) <= 0.0:
            return False, f"penalty evaluated to {float(term.detach())}, expected > 0"
        self.mark_ran()
        return True, f"penalty term active ({float(term.detach()):.4g})"

    def signature(self, model, ctx) -> SignatureCheck | None:
        reference = self.prev_anchor or self.anchor
        if not self.fisher or not reference:
            return None
        # displacement of the top-decile Fisher params vs the bottom decile
        ratios = []
        for n, p in model.named_parameters():
            if n not in self.fisher or n not in reference:
                continue
            f = self.fisher[n].flatten()
            d = (p.detach().cpu() - reference[n]).abs().flatten()
            if f.numel() < 10:
                continue
            hi = f > torch.quantile(f, 0.9)
            lo = f < torch.quantile(f, 0.1)
            if hi.any() and lo.any() and d[lo].mean() > 0:
                ratios.append(float(d[hi].mean() / d[lo].mean()))
        if not ratios:
            return None
        return SignatureCheck(
            probe="B1",
            quantity="displacement ratio, high-Fisher / low-Fisher params",
            value=sum(ratios) / len(ratios),
            baseline=1.0,
            direction="decrease",
            detail="EWC should move important parameters less than unimportant ones",
        )

    def cost_report(self) -> CostReport:
        n = sum(f.numel() for f in self.fisher.values())
        return CostReport(
            buffer_bytes=n * 4 * 2,  # fisher + anchor, fp32
            notes={"fisher_entries": n, "last_penalty": self._penalty_seen},
        )

    # ------------------------------------------------------------------
    def state_dict(self):
        return {"fisher": self.fisher, "anchor": self.anchor,
                "prev_anchor": self.prev_anchor}

    def load_state_dict(self, state):
        state = to_cpu_tree(state)
        self.fisher = state.get("fisher", {})
        self.anchor = state.get("anchor", {})
        self.prev_anchor = state.get("prev_anchor", {})
