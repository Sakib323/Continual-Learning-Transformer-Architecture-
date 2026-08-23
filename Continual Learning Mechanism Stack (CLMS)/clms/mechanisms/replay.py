"""F01 · Experience replay and Dark Experience Replay.

`replay`  reservoir buffer of raw (input, label) pairs, interleaved into each
          batch. The baseline everything else has to beat.
`der`     stores the *logits* the network produced at the time, and regresses
          current outputs onto them. Function-space rather than label-space,
          and at buffer 200 it beat plain replay at buffer 500.

Both populate the buffer by reservoir sampling, which needs no task boundaries —
the reason DER works on a boundary-free stream and EWC does not.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..base import CostReport, Mechanism, SignatureCheck
from ..registry import register


class _Reservoir:
    """Uniform sample over an unbounded stream, no boundaries required."""

    def __init__(self, capacity: int, seed: int = 0):
        self.capacity = capacity
        self.data: list[dict] = []
        self.seen = 0
        self._g = torch.Generator().manual_seed(seed)

    def add(self, item: dict) -> None:
        self.seen += 1
        if len(self.data) < self.capacity:
            self.data.append(item)
            return
        j = int(torch.randint(0, self.seen, (1,), generator=self._g))
        if j < self.capacity:
            self.data[j] = item

    def sample(self, n: int) -> list[dict]:
        if not self.data:
            return []
        n = min(n, len(self.data))
        idx = torch.randperm(len(self.data), generator=self._g)[:n]
        return [self.data[int(i)] for i in idx]

    def __len__(self) -> int:
        return len(self.data)

    def nbytes(self) -> int:
        total = 0
        for item in self.data:
            for v in item.values():
                if torch.is_tensor(v):
                    total += v.numel() * v.element_size()
        return total


@register
class ExperienceReplay(Mechanism):
    name = "replay"
    surfaces = ("S", "D")
    family = "F01"
    paper = "2004.07211"
    order = 10   # must run before loss-surface mechanisms see the batch

    defaults = {
        "capacity": 2000,
        "ratio": 0.25,      # replayed fraction of each effective batch
        "store_every": 1,   # subsample the write path for very long streams
    }

    def __init__(self, **kw):
        super().__init__(**kw)
        self.buffer: _Reservoir | None = None

    def setup(self, model, cfg, ctx) -> None:
        self.buffer = _Reservoir(self.params["capacity"], seed=ctx.seed)

    # ------------------------------------------------------------------
    def on_batch(self, batch, step, ctx):
        assert self.buffer is not None
        bs = batch["input_ids"].shape[0]

        # write path: store individual examples from the incoming batch
        if step % self.params["store_every"] == 0:
            for i in range(bs):
                self.buffer.add({
                    "input_ids": batch["input_ids"][i].detach().cpu().clone(),
                    "labels": batch["labels"][i].detach().cpu().clone(),
                })

        # read path: interleave
        n_replay = int(bs * self.params["ratio"])
        if n_replay == 0 or len(self.buffer) == 0:
            return None
        samples = self.buffer.sample(n_replay)
        if not samples:
            return None
        device = batch["input_ids"].device
        merged = dict(batch)
        merged["input_ids"] = torch.cat(
            [batch["input_ids"], torch.stack([s["input_ids"] for s in samples]).to(device)]
        )
        merged["labels"] = torch.cat(
            [batch["labels"], torch.stack([s["labels"] for s in samples]).to(device)]
        )
        if "task_id" in batch:
            merged["task_id"] = torch.cat([
                batch["task_id"],
                torch.full((len(samples),), -1, dtype=torch.long, device=device),
            ])
        self.mark_ran()
        return merged

    # ------------------------------------------------------------------
    def self_test(self, model, batch, ctx) -> tuple[bool, str]:
        if self.buffer is None:
            return False, "setup() never ran; buffer is None"
        before = batch["input_ids"].shape[0]
        out = self.on_batch(batch, 0, ctx)
        if len(self.buffer) == 0:
            return False, "buffer empty after a write"
        if out is None:
            return True, f"buffer filling ({len(self.buffer)} items), no replay on first batch"
        if out["input_ids"].shape[0] <= before:
            return False, "batch did not grow after replay interleave"
        return True, f"batch {before} -> {out['input_ids'].shape[0]}, buffer={len(self.buffer)}"

    def cost_report(self) -> CostReport:
        n = self.buffer.nbytes() if self.buffer else 0
        return CostReport(
            buffer_bytes=n,
            notes={"buffer_items": len(self.buffer) if self.buffer else 0},
        )

    def state_dict(self):
        return {"data": self.buffer.data, "seen": self.buffer.seen} if self.buffer else {}

    def load_state_dict(self, state):
        if self.buffer is not None and state:
            self.buffer.data = state.get("data", [])
            self.buffer.seen = state.get("seen", 0)


@register
class DarkExperienceReplay(Mechanism):
    name = "der"
    surfaces = ("S", "L")
    family = "F01"
    paper = "2004.07211"
    order = 35

    defaults = {
        "capacity": 2000,
        "alpha": 0.5,       # weight on logit matching (DER)
        "beta": 0.5,        # weight on ground-truth CE (the ++ in DER++)
        "batch_size": 16,   # buffer draws per step
    }

    def __init__(self, **kw):
        super().__init__(**kw)
        self.buffer: _Reservoir | None = None
        self._last = 0.0

    def setup(self, model, cfg, ctx) -> None:
        self.buffer = _Reservoir(self.params["capacity"], seed=ctx.seed + 1)

    # ------------------------------------------------------------------
    def compute_loss(self, model, batch, out, base_loss, ctx):
        assert self.buffer is not None
        device = base_loss.device
        term = None

        if len(self.buffer) > 0:
            samples = self.buffer.sample(self.params["batch_size"])
            if samples:
                ids = torch.stack([s["input_ids"] for s in samples]).to(device)
                stored_logits = torch.stack([s["logits"] for s in samples]).to(device)
                labels = torch.stack([s["labels"] for s in samples]).to(device)
                cur = model(ids)["logits"]

                # DER: match the logits recorded throughout the trajectory
                mse = F.mse_loss(cur, stored_logits)
                term = self.params["alpha"] * mse

                # DER++: plus ordinary CE on the same buffer points
                if self.params["beta"] > 0:
                    ce = F.cross_entropy(
                        cur[:, :-1].reshape(-1, cur.shape[-1]),
                        labels[:, 1:].reshape(-1),
                        ignore_index=-100,
                    )
                    term = term + self.params["beta"] * ce
                self._last = float(term.detach())
                self.mark_ran()

        # write path uses the logits produced *now*, mid-trajectory — that is
        # what separates DER from a boundary-snapshot method like FDR
        logits = out.get("logits")
        if logits is not None:
            n = min(4, batch["input_ids"].shape[0])
            for i in range(n):
                self.buffer.add({
                    "input_ids": batch["input_ids"][i].detach().cpu().clone(),
                    "labels": batch["labels"][i].detach().cpu().clone(),
                    "logits": logits[i].detach().cpu().clone(),
                })
        return term

    # ------------------------------------------------------------------
    def self_test(self, model, batch, ctx) -> tuple[bool, str]:
        if self.buffer is None:
            return False, "setup() never ran"
        device = ctx.device
        out = model(batch["input_ids"].to(device), labels=batch["labels"].to(device))
        self.compute_loss(model, batch, out, out["loss"], ctx)   # fills buffer
        if len(self.buffer) == 0:
            return False, "buffer empty after a write — logits missing from model output?"
        term = self.compute_loss(model, batch, out, out["loss"], ctx)
        if term is None:
            return False, "no loss term produced with a non-empty buffer"
        return True, f"logit-matching term active ({float(term.detach()):.4g})"

    def signature(self, model, ctx) -> SignatureCheck | None:
        if self._last == 0.0:
            return None
        return SignatureCheck(
            probe="C1",
            quantity="DER logit-matching term",
            value=self._last,
            direction="hold",
            detail="non-zero means the function-space constraint is active",
        )

    def cost_report(self) -> CostReport:
        n = self.buffer.nbytes() if self.buffer else 0
        return CostReport(buffer_bytes=n, notes={"buffer_items": len(self.buffer) if self.buffer else 0})

    def state_dict(self):
        return {"data": self.buffer.data, "seen": self.buffer.seen} if self.buffer else {}

    def load_state_dict(self, state):
        if self.buffer is not None and state:
            self.buffer.data = state.get("data", [])
            self.buffer.seen = state.get("seen", 0)
