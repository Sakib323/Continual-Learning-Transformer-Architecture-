"""Retention metrics, all derived from the accuracy matrix.

A[i][j] = accuracy on task j after having trained through task i.

Everything else — average accuracy, forgetting, backward and forward transfer,
intransigence — is a projection of that one object, which is why the trainer
evaluates *every task seen so far* after *every* task rather than only the
current one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch


@dataclass
class AccuracyMatrix:
    num_tasks: int
    A: list[list[float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.A:
            self.A = [[float("nan")] * self.num_tasks for _ in range(self.num_tasks)]

    def record(self, after_task: int, on_task: int, acc: float) -> None:
        self.A[after_task][on_task] = acc

    # --- core metrics ---------------------------------------------------
    def average_accuracy(self, after_task: int | None = None) -> float:
        """AA — mean accuracy over tasks seen so far, at one point in the stream."""
        i = self.num_tasks - 1 if after_task is None else after_task
        vals = [self.A[i][j] for j in range(i + 1) if not math.isnan(self.A[i][j])]
        return sum(vals) / len(vals) if vals else float("nan")

    def average_incremental_accuracy(self) -> float:
        """AIA — AA averaged over the whole stream, not just the end."""
        vals = [self.average_accuracy(i) for i in range(self.num_tasks)]
        vals = [v for v in vals if not math.isnan(v)]
        return sum(vals) / len(vals) if vals else float("nan")

    def forgetting(self) -> float:
        """FM — mean over old tasks of (best ever seen − final)."""
        last = self.num_tasks - 1
        per_task = []
        for j in range(last):
            history = [self.A[i][j] for i in range(j, last) if not math.isnan(self.A[i][j])]
            if history and not math.isnan(self.A[last][j]):
                per_task.append(max(history) - self.A[last][j])
        return sum(per_task) / len(per_task) if per_task else 0.0

    def backward_transfer(self) -> float:
        """BWT — mean change in old-task accuracy caused by later learning.

        Negative is forgetting; positive means later tasks *helped* earlier ones.
        """
        last = self.num_tasks - 1
        deltas = [
            self.A[last][j] - self.A[j][j]
            for j in range(last)
            if not math.isnan(self.A[last][j]) and not math.isnan(self.A[j][j])
        ]
        return sum(deltas) / len(deltas) if deltas else 0.0

    def forward_transfer(self, random_baseline: list[float] | None = None) -> float:
        """FWT — accuracy on unseen tasks above chance, before training on them."""
        if random_baseline is None:
            return float("nan")
        vals = [
            self.A[j - 1][j] - random_baseline[j]
            for j in range(1, self.num_tasks)
            if not math.isnan(self.A[j - 1][j])
        ]
        return sum(vals) / len(vals) if vals else float("nan")

    def learning_accuracy(self, mode: str = "sequential") -> float:
        """LA — mean of the diagonal: how well each task was learned at the
        moment it was trained, before any later task could disturb it.

        This separates two failures that AA alone conflates. A low AA can mean
        the model learned and then forgot (LA high, AA low) or that it never
        learned at all (LA low, and there was nothing to forget). Reporting only
        AA and FM makes the second look like partial success.

        Measured: EWC at lambda=100 has LA 0.43 against the sequential
        control's 0.83 — it retains its first two tasks perfectly and then goes
        rigid, never learning tasks 3-5. Its rho of 0.335 is entirely retention
        of what it learned early, not continual learning.

        IM needs a joint baseline to be defined; LA does not, so it is always
        available and always comparable against the sequential control.

        Only defined for a sequential stream. Joint training sees every task at
        once, so "accuracy right after learning task j" names nothing — its
        diagonal is an artefact of when the checkpoints happen to be taken, and
        reading it as plasticity flags the ceiling run as collapsed.
        """
        if mode != "sequential":
            return float("nan")
        vals = [
            self.A[j][j] for j in range(self.num_tasks)
            if not math.isnan(self.A[j][j])
        ]
        return sum(vals) / len(vals) if vals else float("nan")

    def intransigence(self, joint_per_task: list[float] | None = None) -> float:
        """IM — how much worse than joint training the model does on each task
        at the moment it learns it. Isolates lost *plasticity* from lost memory."""
        if joint_per_task is None:
            return float("nan")
        vals = [
            joint_per_task[j] - self.A[j][j]
            for j in range(self.num_tasks)
            if not math.isnan(self.A[j][j])
        ]
        return sum(vals) / len(vals) if vals else float("nan")

    def summary(self, **ctx) -> dict[str, float]:
        return {
            "AA": self.average_accuracy(),
            "AIA": self.average_incremental_accuracy(),
            "LA": self.learning_accuracy(ctx.get("mode", "sequential")),
            "FM": self.forgetting(),
            "BWT": self.backward_transfer(),
            "FWT": self.forward_transfer(ctx.get("random_baseline")),
            "IM": self.intransigence(ctx.get("joint_per_task")),
        }

    def as_dict(self) -> dict:
        return {"num_tasks": self.num_tasks, "A": self.A}


# ---------------------------------------------------------------------------
def recovery_ratio(mechanism_aa: float, floor_aa: float, ceiling_aa: float) -> float:
    """rho — the number that makes mechanisms comparable.

        0  no better than sequential fine-tuning
        1  matches joint training on all tasks at once
       <0  actively harmful, which happens more than you would think

    Without both a floor and a ceiling you can only say a mechanism beats doing
    nothing, which is true of nearly everything and ranks nothing.
    """
    span = ceiling_aa - floor_aa
    if abs(span) < 1e-9:
        return float("nan")
    return (mechanism_aa - floor_aa) / span


def cost_normalised_recovery(rho: float, overhead_fraction: float) -> float:
    """rho per unit of overhead. A mechanism recovering 60% of the gap for 2%
    extra cost beats one recovering 70% for 3x the memory."""
    return rho / (1.0 + max(overhead_fraction, 0.0))


# ---------------------------------------------------------------------------
@torch.no_grad()
def sequence_accuracy(model, batches, device: str = "cpu") -> float:
    """Exact-match accuracy over the answer span.

    Token-level accuracy would flatter every mechanism equally and hide the
    differences that matter; a task counts as solved only if the whole answer
    is right.
    """
    model.eval()
    correct = total = 0
    for batch in batches:
        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        logits = model(ids)["logits"]
        preds = logits[:, :-1].argmax(-1)
        target = labels[:, 1:]
        mask = target != -100
        per_seq_ok = ((preds == target) | ~mask).all(dim=1)
        has_answer = mask.any(dim=1)
        correct += int((per_seq_ok & has_answer).sum())
        total += int(has_answer.sum())
    model.train()
    return correct / total if total else float("nan")
