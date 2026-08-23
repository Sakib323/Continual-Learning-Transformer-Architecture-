"""Composer: validates a mechanism set, then runs their hooks in a fixed order.

Also implements the `Injector` protocol that `olmo2_cl` expects, so surface-A
mechanisms reach the model without the model importing anything from clms.

Three validations run before a single GPU-second is spent:
  1. conflicts    two enabled mechanisms that cannot co-run
  2. capabilities a mechanism needing something the stream doesn't provide
  3. self-tests   a mechanism that runs but changes nothing
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .base import CostReport, Mechanism, RunContext, SignatureCheck
from . import registry


class ValidationError(RuntimeError):
    pass


class Composer:
    def __init__(self, mechanisms: list[Mechanism], ctx: RunContext):
        self.mechanisms = sorted(mechanisms, key=lambda m: (m.order, m.name))
        self.ctx = ctx
        self._by_surface: dict[str, list[Mechanism]] = {s: [] for s in "ALOSD"}
        for m in self.mechanisms:
            for s in m.surfaces:
                self._by_surface[s].append(m)

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, mech_cfg: dict[str, dict[str, Any]], ctx: RunContext) -> "Composer":
        instances: list[Mechanism] = []
        for name, params in mech_cfg.items():
            if not params.get("enabled", False):
                continue
            klass = registry.get(name)
            instances.append(klass(**{k: v for k, v in params.items() if k != "enabled"}))
        comp = cls(instances, ctx)
        comp.validate()
        return comp

    # ------------------------------------------------------------------
    def validate(self) -> None:
        names = {m.name for m in self.mechanisms}

        # 1. conflicts
        for m in self.mechanisms:
            clash = names & set(m.conflicts)
            if clash:
                raise ValidationError(
                    f"'{m.name}' cannot co-run with {sorted(clash)}. "
                    f"These mechanisms make incompatible demands on the same "
                    f"parameters — enable one or the other, not both."
                )

        # 2. stream capabilities
        for m in self.mechanisms:
            missing = [c for c in m.requires if not self.ctx.has(c)]
            if missing:
                raise ValidationError(
                    f"'{m.name}' requires stream capabilities {missing}, but this "
                    f"stream provides {list(self.ctx.stream_capabilities)}. Results "
                    f"would be meaningless rather than merely worse."
                )

        # 3. ordering ambiguity is a warning, not an error, but make it visible
        for surface in "AO":
            group = self._by_surface[surface]
            orders = [m.order for m in group]
            if len(orders) != len(set(orders)) and len(group) > 1:
                dupes = sorted({o for o in orders if orders.count(o) > 1})
                print(
                    f"[compose] warning: surface {surface} mechanisms share order "
                    f"values {dupes}; hook sequence falls back to name order."
                )

    def run_self_tests(self, model: nn.Module, batch: dict) -> None:
        failures: list[str] = []
        for m in self.mechanisms:
            try:
                ok, detail = m.self_test(model, batch, self.ctx)
            except NotImplementedError as exc:
                failures.append(f"{m.name}: {exc}")
                continue
            except Exception as exc:  # noqa: BLE001 - report, don't mask
                failures.append(f"{m.name}: self_test raised {type(exc).__name__}: {exc}")
                continue
            if not ok:
                failures.append(f"{m.name}: {detail}")
        if failures:
            raise ValidationError(
                "mechanism self-tests failed — these would have run and silently "
                "done nothing:\n  " + "\n  ".join(failures)
            )

    # ------------------------------------------------------------------
    # Injector protocol (consumed by olmo2_cl)
    # ------------------------------------------------------------------
    def build_attention(self, layer_idx: int, default: nn.Module) -> nn.Module:
        module = default
        for m in self._by_surface["A"]:
            replacement = m.build_attention(layer_idx, module)
            if replacement is not None:
                module = replacement
                m.mark_ran()
        return module

    def build_mlp(self, layer_idx: int, default: nn.Module) -> nn.Module:
        module = default
        for m in self._by_surface["A"]:
            replacement = m.build_mlp(layer_idx, module)
            if replacement is not None:
                module = replacement
                m.mark_ran()
        return module

    def observe(self, name: str, layer_idx: int, tensor: torch.Tensor) -> torch.Tensor:
        for m in self._by_surface["A"]:
            transformed = m.observe(name, layer_idx, tensor)
            if transformed is not None:
                tensor = transformed
                m.mark_ran()
        for probe in self.ctx.scratch.get("_observers", []):
            probe(name, layer_idx, tensor)
        return tensor

    # ------------------------------------------------------------------
    # trainer-facing hooks
    # ------------------------------------------------------------------
    def set_model_config(self, cfg: Any) -> None:
        """Give mechanisms the model shape before construction begins."""
        for m in self.mechanisms:
            m.on_model_config(cfg, self.ctx)

    def setup(self, model: nn.Module, cfg: Any) -> None:
        for m in self.mechanisms:
            m.setup(model, cfg, self.ctx)

    def on_batch(self, batch: dict, step: int) -> dict:
        for m in self._by_surface["D"]:
            modified = m.on_batch(batch, step, self.ctx)
            if modified is not None:
                batch = modified
                m.mark_ran()
        return batch

    def compute_loss(
        self, model: nn.Module, batch: dict, out: dict, base_loss: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        total = base_loss
        parts: dict[str, float] = {}
        for m in self._by_surface["L"]:
            extra = m.compute_loss(model, batch, out, base_loss, self.ctx)
            if extra is not None:
                total = total + extra
                parts[f"loss/{m.name}"] = float(extra.detach())
                m.mark_ran()
        return total, parts

    def before_step(self, model: nn.Module) -> None:
        for m in self._by_surface["O"]:
            m.before_step(model, self.ctx)

    def after_step(self, model: nn.Module) -> None:
        for m in self._by_surface["O"]:
            m.after_step(model, self.ctx)

    def on_task_start(self, model: nn.Module, task_id: int) -> None:
        for m in self.mechanisms:
            m.on_task_start(model, task_id, self.ctx)

    def on_task_end(self, model: nn.Module, task_id: int) -> None:
        for m in self.mechanisms:
            m.on_task_end(model, task_id, self.ctx)

    def on_eval(self, model: nn.Module, task_id: int) -> dict[str, float]:
        out: dict[str, float] = {}
        for m in self.mechanisms:
            for k, v in m.on_eval(model, task_id, self.ctx).items():
                out[f"{m.name}/{k}"] = v
        return out

    # ------------------------------------------------------------------
    def signatures(self, model: nn.Module) -> list[SignatureCheck]:
        out: list[SignatureCheck] = []
        for m in self.mechanisms:
            sig = m.signature(model, self.ctx)
            if sig is not None:
                out.append(sig)
        return out

    def costs(self) -> dict[str, dict[str, Any]]:
        return {m.name: m.cost_report().as_dict() for m in self.mechanisms}

    def inert(self) -> list[str]:
        """Mechanisms enabled but never observed to act. Checked after step 1."""
        return [m.name for m in self.mechanisms if not m.ran]

    # ------------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        return {m.name: m.state_dict() for m in self.mechanisms}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for m in self.mechanisms:
            if m.name in state:
                m.load_state_dict(state[m.name])

    @property
    def names(self) -> list[str]:
        return [m.name for m in self.mechanisms]

    def __len__(self) -> int:
        return len(self.mechanisms)

    def __repr__(self) -> str:
        return f"<Composer {len(self.mechanisms)} enabled: {', '.join(self.names) or 'none'}>"
