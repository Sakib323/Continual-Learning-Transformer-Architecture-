"""The mechanism contract.

Every continual-learning mechanism in this library, regardless of family,
implements this one interface. A mechanism declares which *surfaces* it touches
and then fills in only the hooks it needs.

Surfaces
--------
A  architecture   changes the model graph; needs the model at build time
L  loss           adds a term to the objective
O  optimizer      modifies gradients or the update rule
S  state          external buffer / datastore; never touches weights
D  data           changes what a batch contains, or when

Two methods are mandatory and exist to catch the project's most dangerous
failure mode: a mechanism that runs but does nothing looks exactly like a
mechanism that legitimately doesn't help.

`self_test`  proves the mechanism *ran* and changed something.
`signature`  proves it did *the thing it claims* — the internal quantity its
             paper says should move. See design/diagnostics-suite.html.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import zlib

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
@dataclass
class SignatureCheck:
    """Result of a mechanism's signature probe."""

    probe: str                  # probe id from the diagnostics suite, e.g. "B2"
    quantity: str               # human-readable name of what was measured
    value: float
    baseline: float | None = None
    direction: str = "decrease"  # "decrease" | "increase" | "hold"
    passed: bool | None = None
    detail: str = ""

    def evaluate(self, tolerance: float = 0.0) -> bool:
        if self.baseline is None:
            self.passed = None
            return True
        if self.direction == "decrease":
            self.passed = self.value < self.baseline * (1.0 - tolerance)
        elif self.direction == "increase":
            self.passed = self.value > self.baseline * (1.0 + tolerance)
        else:
            self.passed = abs(self.value - self.baseline) <= abs(baseline_eps(self.baseline))
        return bool(self.passed)


def baseline_eps(x: float, frac: float = 0.1) -> float:
    return max(abs(x) * frac, 1e-8)


@dataclass
class CostReport:
    """What the mechanism cost. Logged for every run; see build-workflow Risk 04."""

    added_params: int = 0
    buffer_bytes: int = 0
    extra_flops_per_step: int = 0
    wall_ms_per_step: float = 0.0
    peak_memory_bytes: int = 0
    notes: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "added_params": self.added_params,
            "buffer_bytes": self.buffer_bytes,
            "extra_flops_per_step": self.extra_flops_per_step,
            "wall_ms_per_step": self.wall_ms_per_step,
            "peak_memory_bytes": self.peak_memory_bytes,
            **self.notes,
        }


# ---------------------------------------------------------------------------
class Mechanism:
    """Base class. Subclasses override only the hooks they need."""

    # --- identity, declared by every subclass ---------------------------
    name: str = "unnamed"
    surfaces: tuple[str, ...] = ()
    family: str = ""                       # registry family id, e.g. "F02"
    paper: str = ""                        # arXiv id or "external"

    # --- composition metadata -------------------------------------------
    conflicts: tuple[str, ...] = ()        # mechanism names that cannot co-run
    requires: tuple[str, ...] = ()         # stream capabilities, e.g. "task_boundaries"
    order: int = 50                        # lower runs first within a hook

    # --- default hyperparameters; overridden from config ----------------
    defaults: dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        params = dict(self.defaults)
        unknown = set(kwargs) - set(params) - {"enabled"}
        if unknown:
            raise ValueError(
                f"{self.name}: unknown parameters {sorted(unknown)}; "
                f"known: {sorted(params)}"
            )
        params.update({k: v for k, v in kwargs.items() if k != "enabled"})
        self.params = params
        self._ran = False
        self._rng: torch.Generator | None = None

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name} surfaces={'/'.join(self.surfaces)}>"

    # ------------------------------------------------------------------
    # lifecycle hooks — all optional
    # ------------------------------------------------------------------
    def on_model_config(self, cfg: Any, ctx: "RunContext") -> None:
        """Called *before* the model is built, with the resolved model config.

        Surface-A mechanisms that need to know the shape of the network — which
        layer to replace, how wide it is — resolve that here rather than
        guessing during construction.
        """

    def setup(self, model: nn.Module, cfg: Any, ctx: "RunContext") -> None:
        """Called once after the model is built. Allocate buffers here."""

    def build_attention(self, layer_idx: int, default: nn.Module) -> nn.Module | None:
        """Surface A. Return a replacement module, or None to leave it alone."""
        return None

    def build_mlp(self, layer_idx: int, default: nn.Module) -> nn.Module | None:
        """Surface A. Return a replacement module, or None to leave it alone."""
        return None

    def observe(self, name: str, layer_idx: int, tensor: torch.Tensor) -> torch.Tensor | None:
        """Surface A. Transform an intermediate activation, or None to pass through.

        `name` is one of: embed, attn_out, mlp_hidden, resid_mid, resid_out, final.
        """
        return None

    def on_batch(self, batch: dict, step: int, ctx: "RunContext") -> dict | None:
        """Surface D. Return a modified batch, or None to leave it alone."""
        return None

    def compute_loss(
        self, model: nn.Module, batch: dict, out: dict, base_loss: torch.Tensor,
        ctx: "RunContext",
    ) -> torch.Tensor | None:
        """Surface L. Return an *additional* loss term, or None."""
        return None

    def before_step(self, model: nn.Module, ctx: "RunContext") -> None:
        """Surface O. Gradients exist; optimizer has not stepped."""

    def after_step(self, model: nn.Module, ctx: "RunContext") -> None:
        """Surface O. Optimizer has stepped; parameters are updated."""

    def on_task_start(self, model: nn.Module, task_id: int, ctx: "RunContext") -> None:
        """Boundary hook. Reset per-task accumulators here."""

    def on_task_end(self, model: nn.Module, task_id: int, ctx: "RunContext") -> None:
        """Boundary hook. Fisher estimation, SVD bases, buffer resizing."""

    def on_eval(self, model: nn.Module, task_id: int, ctx: "RunContext") -> dict:
        """Extra metrics this mechanism wants recorded."""
        return {}

    # ------------------------------------------------------------------
    # mandatory diagnostics
    # ------------------------------------------------------------------
    def self_test(self, model: nn.Module, batch: dict, ctx: "RunContext") -> tuple[bool, str]:
        """Prove the mechanism actually does something.

        Runs on step 0 of every run. Returning False aborts before any GPU time
        is spent on a configuration that would silently report "no effect".
        """
        raise NotImplementedError(
            f"{self.name} must implement self_test(); see base.Mechanism docstring"
        )

    def signature(self, model: nn.Module, ctx: "RunContext") -> SignatureCheck | None:
        """The internal quantity this mechanism's paper claims should move.

        Returning None is allowed for mechanisms with no distinctive internal
        claim (pure data-surface policy, for example), but prefer a real probe.
        """
        return None

    def cost_report(self) -> CostReport:
        return CostReport()

    # ------------------------------------------------------------------
    # checkpoint/resume — required for interruptible rented instances
    # ------------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        """Mechanism state that must survive a restart.

        A resumed run whose Fisher matrix / replay buffer / SVD bases were lost
        is silently a different experiment. Override whenever the mechanism
        holds state across steps.
        """
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        pass

    # ------------------------------------------------------------------
    def rng(self, ctx: "RunContext") -> torch.Generator:
        """Per-mechanism generator, derived from the run seed.

        Mechanisms that draw randomness during *training* — GPM subsampling
        activations for its basis, shrink-perturb injecting noise, CBP deciding
        which units to reinitialise — must not use the global RNG. Doing so
        makes a run irreproducible at a fixed seed: GPM was measured at
        AA 0.243 vs 0.538 across two runs of the same config, a spread larger
        than most mechanisms' entire effect, and one that seed-to-seed error
        bars cannot see because it lives *within* a seed.

        Offset by a hash of the mechanism name so two mechanisms in one stack do
        not draw the identical sequence.
        """
        if self._rng is None:
            offset = zlib.crc32(self.name.encode()) % 100_000
            self._rng = torch.Generator().manual_seed(ctx.seed + offset)
        return self._rng

    def mark_ran(self) -> None:
        self._ran = True

    @property
    def ran(self) -> bool:
        return self._ran


# ---------------------------------------------------------------------------
@dataclass
class RunContext:
    """Everything a mechanism might need that isn't the model or the batch."""

    step: int = 0
    task_id: int = 0
    num_tasks: int = 1
    epoch: int = 0
    device: str = "cpu"
    seed: int = 0
    stream_capabilities: tuple[str, ...] = ("task_boundaries", "task_ids")
    # scratch space shared between mechanisms and the trainer
    scratch: dict[str, Any] = field(default_factory=dict)

    def has(self, capability: str) -> bool:
        return capability in self.stream_capabilities


# ---------------------------------------------------------------------------
def to_cpu_tree(obj: Any) -> Any:
    """Move every tensor in a nested container to CPU.

    Mechanism buffers (Fisher matrices, path integrals, utility counters) are
    deliberately kept off the accelerator. `torch.load(map_location=device)`
    would drag them onto it and produce a device mismatch on the first use after
    a resume, so state restoration normalises through here.
    """
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: to_cpu_tree(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_cpu_tree(v) for v in obj)
    return obj


def trainable_named_params(model: nn.Module) -> list[tuple[str, torch.Tensor]]:
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def flat_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for _, p in trainable_named_params(model)])


def flat_grads(model: nn.Module) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for _, p in trainable_named_params(model):
        parts.append(
            torch.zeros_like(p).reshape(-1) if p.grad is None else p.grad.detach().reshape(-1)
        )
    return torch.cat(parts)
