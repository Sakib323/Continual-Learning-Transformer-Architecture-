"""CLMS — Continual Learning Mechanism Stack.

A library of continual-learning mechanisms that plug into a transformer through
one contract, so a sweep is a config change rather than a rewrite.

    from clms import Composer, RunContext, registry

    ctx  = RunContext(num_tasks=6, device="cuda")
    comp = Composer.from_config(cfg["mechanisms"], ctx)
    model = build_model(model_cfg, injector=comp)

Surfaces: A architecture, L loss, O optimizer, S state, D data.
"""

from .base import Mechanism, RunContext, CostReport, SignatureCheck
from .compose import Composer, ValidationError
from . import registry

__version__ = "0.1.0"
__all__ = [
    "Mechanism", "RunContext", "CostReport", "SignatureCheck",
    "Composer", "ValidationError", "registry",
]
