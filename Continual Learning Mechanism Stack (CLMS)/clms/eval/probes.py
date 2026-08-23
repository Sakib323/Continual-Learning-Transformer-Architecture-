"""Internal diagnostic probes.

Accuracy tells you *that* forgetting happened. These tell you why, and — more
usefully — whether a mechanism did the specific internal thing its paper claims.
Probe ids match design/diagnostics-suite.html.

    A1  effective rank            A2  representation drift (CKA)
    A3  cross-task overlap        A4  activation overlap (Jaccard)
    A5  dead units
    B1  displacement map          B2  update concentration
    B3  weight magnitude          B4  layer-wise forgetting attribution
    C1  gradient interference     C2  gradient subspace angles
    C3  sharpness / Fisher trace  C4  loss barrier
    D2  logit drift
"""

from __future__ import annotations

import math
from typing import Callable, Iterable

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# activation capture
# ---------------------------------------------------------------------------
class ActivationRecorder:
    """Collects activations flowing through the model's observe() points."""

    def __init__(self, names: tuple[str, ...] = ("mlp_hidden", "resid_out")):
        self.names = names
        self.store: dict[tuple[str, int], list[torch.Tensor]] = {}
        self.enabled = False

    def __call__(self, name: str, layer_idx: int, tensor: torch.Tensor) -> None:
        if not self.enabled or name not in self.names:
            return
        key = (name, layer_idx)
        # flatten batch/time, keep feature dim; detach and move off-device early
        flat = tensor.detach().reshape(-1, tensor.shape[-1]).float().cpu()
        self.store.setdefault(key, []).append(flat)

    def reset(self) -> None:
        self.store.clear()

    def stacked(self, name: str, layer_idx: int, max_rows: int = 4096) -> torch.Tensor | None:
        parts = self.store.get((name, layer_idx))
        if not parts:
            return None
        H = torch.cat(parts, dim=0)
        return H[:max_rows]

    def layers(self, name: str) -> list[int]:
        return sorted(idx for (n, idx) in self.store if n == name)


# ---------------------------------------------------------------------------
# A — representation
# ---------------------------------------------------------------------------
def effective_rank(H: torch.Tensor, eps: float = 1e-12) -> float:
    """A1. Entropy of the singular-value spectrum.

    Collapse here is the clearest signature of plasticity loss, and it moves
    long before accuracy does.
    """
    if H.numel() == 0 or H.shape[0] < 2:
        return float("nan")
    H = H - H.mean(dim=0, keepdim=True)
    try:
        s = torch.linalg.svdvals(H.double())
    except Exception:
        return float("nan")
    s = s[s > eps]
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    return float(torch.exp(-(p * torch.log(p)).sum()))


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """A2/A3. Linear CKA. 1.0 = identical representation, 0 = unrelated."""
    if X.shape[0] != Y.shape[0] or X.shape[0] < 2:
        return float("nan")
    X = (X - X.mean(0, keepdim=True)).double()
    Y = (Y - Y.mean(0, keepdim=True)).double()
    num = torch.linalg.norm(Y.T @ X, ord="fro") ** 2
    den = torch.linalg.norm(X.T @ X, ord="fro") * torch.linalg.norm(Y.T @ Y, ord="fro")
    return float(num / den) if den > 0 else float("nan")


def activation_overlap(Ha: torch.Tensor, Hb: torch.Tensor, quantile: float = 0.9) -> float:
    """A4. Jaccard overlap of the active-unit sets for two tasks.

    The direct test of the sparsity argument: if a sparsity mechanism doesn't
    drive this down, it isn't doing what it claims regardless of accuracy.
    """
    if Ha.numel() == 0 or Hb.numel() == 0:
        return float("nan")
    a = Ha.abs().mean(0)
    b = Hb.abs().mean(0)
    ta = torch.quantile(a, quantile)
    tb = torch.quantile(b, quantile)
    sa, sb = a > ta, b > tb
    union = (sa | sb).sum()
    return float((sa & sb).sum() / union) if union > 0 else float("nan")


def dead_fraction(H: torch.Tensor, threshold: float = 1e-3) -> float:
    """A5. Units whose mean absolute activation is ~zero across the probe set."""
    if H.numel() == 0:
        return float("nan")
    return float((H.abs().mean(0) < threshold).float().mean())


# ---------------------------------------------------------------------------
# B — parameters
# ---------------------------------------------------------------------------
def snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def displacement(model: nn.Module, before: dict[str, torch.Tensor]) -> dict[str, float]:
    """B1. Per-parameter-group L2 movement since the snapshot."""
    out: dict[str, float] = {}
    for n, p in model.named_parameters():
        if n in before:
            out[n] = float((p.detach() - before[n]).norm())
    return out


def update_concentration(
    model: nn.Module, before: dict[str, torch.Tensor], frac: float = 0.9
) -> float:
    """B2. Fraction of parameters accounting for `frac` of total movement.

    The measurable form of "localized updates". Full finetuning lands around
    0.5; sparse mechanisms should reach 1e-3 or below.
    """
    deltas = [
        (p.detach() - before[n]).abs().reshape(-1)
        for n, p in model.named_parameters()
        if n in before
    ]
    if not deltas:
        return float("nan")
    d = torch.cat(deltas)
    total = d.sum()
    if total <= 0:
        return float("nan")
    sorted_d, _ = torch.sort(d, descending=True)
    csum = torch.cumsum(sorted_d, dim=0)
    k = int(torch.searchsorted(csum, total * frac)) + 1
    return k / d.numel()


def weight_magnitude(model: nn.Module) -> float:
    """B3. Mean |theta|. Third of the three plasticity-loss correlates."""
    vals = [p.detach().abs().mean() for p in model.parameters() if p.requires_grad]
    return float(torch.stack(vals).mean()) if vals else float("nan")


@torch.no_grad()
def layerwise_forgetting_attribution(
    model: nn.Module,
    old_params: dict[str, torch.Tensor],
    accuracy_fn: Callable[[], float],
    layer_prefixes: Iterable[str],
) -> dict[str, float]:
    """B4. Which layers actually cause the forgetting?

    Restore one layer at a time to its earlier values and re-measure. The layer
    whose restoration recovers the most is where forgetting lives — which tells
    you where to *put* a mechanism, not just whether one worked.
    """
    base = accuracy_fn()
    results: dict[str, float] = {}
    current = {n: p.detach().clone() for n, p in model.named_parameters()}
    for prefix in layer_prefixes:
        touched = [n for n in old_params if n.startswith(prefix)]
        if not touched:
            continue
        for n, p in model.named_parameters():
            if n in touched:
                p.copy_(old_params[n])
        results[prefix] = accuracy_fn() - base
        for n, p in model.named_parameters():
            if n in touched:
                p.copy_(current[n])
    return results


# ---------------------------------------------------------------------------
# C — gradients and geometry
# ---------------------------------------------------------------------------
def _grad_vector(model: nn.Module, batch: dict, device: str) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    out = model(batch["input_ids"].to(device), labels=batch["labels"].to(device))
    out["loss"].backward()
    g = torch.cat([
        (p.grad.detach().reshape(-1) if p.grad is not None else torch.zeros(p.numel(), device=p.device))
        for p in model.parameters() if p.requires_grad
    ])
    model.zero_grad(set_to_none=True)
    return g


def gradient_interference(model: nn.Module, batch_old: dict, batch_new: dict, device: str = "cpu") -> float:
    """C1. Cosine between old- and new-task gradients.

    Negative means direct conflict: every step on the new task actively damages
    the old one. This is what GEM/A-GEM/MER target, so measuring it says
    whether they are earning their cost.
    """
    g_old = _grad_vector(model, batch_old, device)
    g_new = _grad_vector(model, batch_new, device)
    denom = g_old.norm() * g_new.norm()
    return float(torch.dot(g_old, g_new) / denom) if denom > 0 else float("nan")


def subspace_principal_angles(Ga: torch.Tensor, Gb: torch.Tensor, k: int = 8) -> list[float]:
    """C2. Principal angles (degrees) between two tasks' gradient subspaces.

    GPM's premise made measurable: near-90 degrees means the tasks occupy
    separable directions and projection methods should work well here.
    """
    def basis(G: torch.Tensor) -> torch.Tensor:
        U, _, _ = torch.linalg.svd(G.double(), full_matrices=False)
        return U[:, : min(k, U.shape[1])]

    Ua, Ub = basis(Ga), basis(Gb)
    s = torch.linalg.svdvals(Ua.T @ Ub).clamp(-1.0, 1.0)
    return [float(math.degrees(math.acos(v))) for v in s]


def fisher_trace(model: nn.Module, batches, device: str = "cpu", max_batches: int = 8) -> float:
    """C3. Trace of the empirical Fisher — a cheap sharpness proxy.

    Flatter solutions after task k tend to forget less at task k+1, so this is
    forward-looking in a way accuracy is not.
    """
    total, n = 0.0, 0
    for i, batch in enumerate(batches):
        if i >= max_batches:
            break
        model.zero_grad(set_to_none=True)
        out = model(batch["input_ids"].to(device), labels=batch["labels"].to(device))
        out["loss"].backward()
        total += float(sum(
            (p.grad.detach() ** 2).sum() for p in model.parameters()
            if p.requires_grad and p.grad is not None
        ))
        n += 1
    model.zero_grad(set_to_none=True)
    return total / n if n else float("nan")


@torch.no_grad()
def perturbation_sensitivity(
    model: nn.Module, loss_fn: Callable[[], float], sigma: float = 0.01, trials: int = 3
) -> float:
    """C3b. Loss increase under Gaussian weight noise. Complements Fisher trace."""
    base = loss_fn()
    saved = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    deltas = []
    for _ in range(trials):
        for n, p in model.named_parameters():
            if n in saved:
                p.add_(torch.randn_like(p) * sigma * p.detach().abs().mean())
        deltas.append(loss_fn() - base)
        for n, p in model.named_parameters():
            if n in saved:
                p.copy_(saved[n])
    return float(sum(deltas) / len(deltas))


@torch.no_grad()
def loss_barrier(
    model: nn.Module,
    theta_a: dict[str, torch.Tensor],
    theta_b: dict[str, torch.Tensor],
    loss_fn: Callable[[], float],
    n_points: int = 5,
) -> float:
    """C4. Loss bump on the straight path between two task solutions.

    A low barrier means the solutions share a basin — which predicts that
    merging would have worked, and says something real about whether the tasks
    conflict at all.
    """
    saved = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    def set_to(alpha: float) -> None:
        for n, p in model.named_parameters():
            if n in theta_a and n in theta_b:
                p.copy_(alpha * theta_a[n] + (1 - alpha) * theta_b[n])

    set_to(1.0); la = loss_fn()
    set_to(0.0); lb = loss_fn()
    worst = 0.0
    for i in range(1, n_points):
        a = i / n_points
        set_to(a)
        worst = max(worst, loss_fn() - (a * la + (1 - a) * lb))
    for n, p in model.named_parameters():
        if n in saved:
            p.copy_(saved[n])
    return float(worst)


# ---------------------------------------------------------------------------
# D — behaviour
# ---------------------------------------------------------------------------
@torch.no_grad()
def logit_drift(model: nn.Module, probe_batches, reference_logits, device: str = "cpu") -> float:
    """D2. KL between reference and current outputs on the frozen probe set.

    Catches functional change that accuracy hides: a model can stay correct
    while its confidence structure is destroyed, and that shows up as accuracy
    loss one task later.
    """
    total, n = 0.0, 0
    for batch, ref in zip(probe_batches, reference_logits):
        cur = model(batch["input_ids"].to(device))["logits"]
        p = torch.log_softmax(ref.to(device), dim=-1)
        q = torch.log_softmax(cur, dim=-1)
        total += float(torch.nn.functional.kl_div(q, p, log_target=True, reduction="batchmean"))
        n += 1
    return total / n if n else float("nan")
