"""Mechanism registry.

Adding a mechanism means adding one file with an `@register` decorator. The CLI
flags, the config schema and the sweep presets all derive from what is
registered — no argument parser to edit, ever.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Iterator, Type

from .base import Mechanism

_REGISTRY: dict[str, Type[Mechanism]] = {}


def register(cls: Type[Mechanism]) -> Type[Mechanism]:
    if cls.name in _REGISTRY and _REGISTRY[cls.name] is not cls:
        raise ValueError(f"duplicate mechanism name: {cls.name!r}")
    if not cls.surfaces:
        raise ValueError(f"{cls.name!r} declares no surfaces")
    bad = set(cls.surfaces) - set("ALOSD")
    if bad:
        raise ValueError(f"{cls.name!r} declares unknown surfaces {sorted(bad)}")
    _REGISTRY[cls.name] = cls
    return cls


def discover() -> None:
    """Import every module under clms.mechanisms so decorators run."""
    from . import mechanisms

    for mod in pkgutil.iter_modules(mechanisms.__path__):
        if not mod.name.startswith("_"):
            importlib.import_module(f"{mechanisms.__name__}.{mod.name}")


def get(name: str) -> Type[Mechanism]:
    if not _REGISTRY:
        discover()
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown mechanism {name!r}. Registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def all_mechanisms() -> dict[str, Type[Mechanism]]:
    if not _REGISTRY:
        discover()
    return dict(_REGISTRY)


def by_surface(surface: str) -> list[str]:
    return sorted(n for n, c in all_mechanisms().items() if surface in c.surfaces)


def default_config() -> dict[str, dict[str, Any]]:
    """The all-flags-false config: every registered mechanism, disabled.

    This is the control condition, and it is generated rather than written by
    hand so it can never drift out of sync with the registry.
    """
    out: dict[str, dict[str, Any]] = {}
    for name, cls in sorted(all_mechanisms().items()):
        out[name] = {"enabled": False, **cls.defaults}
    return out


def describe() -> Iterator[str]:
    for name, cls in sorted(all_mechanisms().items()):
        flags = "/".join(cls.surfaces)
        req = f" requires={list(cls.requires)}" if cls.requires else ""
        con = f" conflicts={list(cls.conflicts)}" if cls.conflicts else ""
        yield f"{name:<22} [{flags:<5}] {cls.family:<4} {cls.paper:<12}{req}{con}"
