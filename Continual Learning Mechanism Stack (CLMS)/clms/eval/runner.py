"""Probe runner: holds the reference state every internal probe is defined against.

Most probes are *relative* — drift from a reference representation, KL from
reference logits, gradient conflict against an earlier task. That reference has
to be captured once, on a frozen probe set, and then held constant for the whole
run. If it moves mid-project the numbers stop being comparable across runs,
which is why it lives in one object rather than being recomputed ad hoc.
"""

from __future__ import annotations

import torch

from .probes import (
    ActivationRecorder, activation_overlap, dead_fraction, effective_rank,
    fisher_trace, gradient_interference, linear_cka, logit_drift,
    subspace_principal_angles, update_concentration, weight_magnitude,
)


class ProbeRunner:
    def __init__(self, model, stream, device: str, recorder: ActivationRecorder,
                 n_batches: int = 2, capture: str = "mlp_hidden"):
        self.model = model
        self.stream = stream
        self.device = device
        self.recorder = recorder
        self.n_batches = n_batches
        self.capture = capture

        self.ref_acts: dict[int, torch.Tensor] = {}     # layer -> activations, task 0
        self.ref_logits: list[torch.Tensor] = []
        self.task_acts: dict[int, dict[int, torch.Tensor]] = {}
        self._captured = False

    # ------------------------------------------------------------------
    def _collect(self, task_idx: int) -> dict[int, torch.Tensor]:
        self.recorder.reset()
        self.recorder.enabled = True
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            for batch in self.stream.eval_batches(self.stream.tasks[task_idx], self.n_batches):
                self.model(batch["input_ids"].to(self.device))
        self.model.train(was_training)
        self.recorder.enabled = False
        return {
            i: H for i in self.recorder.layers(self.capture)
            if (H := self.recorder.stacked(self.capture, i)) is not None
        }

    @torch.no_grad()
    def capture_reference(self, task_idx: int = 0) -> None:
        """Freeze the reference: representations and logits on task 0's probe set."""
        self.ref_acts = self._collect(task_idx)
        self.ref_logits = []
        was_training = self.model.training
        self.model.eval()
        for batch in self.stream.eval_batches(self.stream.tasks[task_idx], self.n_batches):
            self.ref_logits.append(
                self.model(batch["input_ids"].to(self.device))["logits"].detach().cpu()
            )
        self.model.train(was_training)
        self._captured = True

    # ------------------------------------------------------------------
    def run(self, task_idx: int, params_before: dict | None = None) -> dict[str, float]:
        out: dict[str, float] = {}
        cur = self._collect(0)          # always evaluated on task 0's probe set

        # A1 / A5 — plasticity
        ranks = [effective_rank(H) for H in cur.values()]
        ranks = [r for r in ranks if r == r]
        if ranks:
            out["A1_effective_rank"] = sum(ranks) / len(ranks)
        deads = [dead_fraction(H) for H in cur.values()]
        deads = [d for d in deads if d == d]
        if deads:
            out["A5_dead_fraction"] = sum(deads) / len(deads)

        # A2 — representation drift from the frozen reference
        if self._captured:
            ckas = [
                linear_cka(self.ref_acts[i], cur[i])
                for i in cur if i in self.ref_acts
                and self.ref_acts[i].shape == cur[i].shape
            ]
            ckas = [c for c in ckas if c == c]
            if ckas:
                out["A2_representation_cka"] = sum(ckas) / len(ckas)

        # A4 — cross-task activation overlap
        self.task_acts[task_idx] = cur
        if 0 in self.task_acts and task_idx > 0:
            overlaps = [
                activation_overlap(self.task_acts[0][i], cur[i])
                for i in cur if i in self.task_acts[0]
                and self.task_acts[0][i].shape[1] == cur[i].shape[1]
            ]
            overlaps = [o for o in overlaps if o == o]
            if overlaps:
                out["A4_activation_overlap"] = sum(overlaps) / len(overlaps)

        # B2 / B3 — parameter movement
        out["B3_weight_magnitude"] = weight_magnitude(self.model)
        if params_before:
            out["B2_update_concentration"] = update_concentration(self.model, params_before)

        # C1 — gradient interference between task 0 and the task just learned
        if task_idx > 0:
            b_old = next(iter(self.stream.eval_batches(self.stream.tasks[0], 1)))
            b_new = next(iter(self.stream.eval_batches(self.stream.tasks[task_idx], 1)))
            out["C1_gradient_interference"] = gradient_interference(
                self.model, b_old, b_new, self.device
            )

        # C3 — sharpness
        out["C3_fisher_trace"] = fisher_trace(
            self.model, self.stream.eval_batches(self.stream.tasks[task_idx], 4),
            self.device, max_batches=4,
        )

        # D2 — logit drift on the frozen probe set
        if self._captured and self.ref_logits:
            out["D2_logit_drift"] = logit_drift(
                self.model,
                self.stream.eval_batches(self.stream.tasks[0], self.n_batches),
                self.ref_logits,
                self.device,
            )

        self.model.zero_grad(set_to_none=True)
        return {f"probe/{k}": v for k, v in out.items()}
