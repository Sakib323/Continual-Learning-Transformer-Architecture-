"""Run configuration: YAML plus dotted CLI overrides.

The mechanism block is *generated* from the registry rather than written by
hand, so it can never drift out of sync with what's implemented. Every run dumps
its fully-resolved config — including defaults nobody typed — because six months
from now that file is the only reliable record of what actually ran.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import registry

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


# ---------------------------------------------------------------------------
BASE_CONFIG: dict[str, Any] = {
    "run": {
        "name": "unnamed",
        "seed": 0,
        "device": "auto",
        "out_dir": "runs",
        "resume": True,
        # A checkpoint exists to survive an interrupted instance. Once a run has
        # written result.json there is nothing left to resume, and keeping the
        # model plus optimizer state for every run costs ~280MB each at 24M
        # params — roughly 40GB across a 141-run tuned sweep, which fills a
        # rented instance mid-sweep.
        "keep_checkpoint": False,
        # Skip a configuration whose result.json already exists. Makes an
        # interrupted sweep resumable at the sweep level, not just per run.
        "skip_completed": True,
    },
    "model": {
        "size_preset": "small",
        "vocab_size": 128,
        "max_position_embeddings": 128,
        "tie_word_embeddings": True,
    },
    "stream": {
        "tasks": ["copy", "reverse", "sort", "modadd23", "induction6"],
        "batch_size": 32,
        "steps_per_task": 400,
        # Optional per-task step budgets, e.g. [400,400,400,400,2000] to model a
        # user who overwhelmingly talks about the last topic. None = uniform.
        "steps_schedule": None,
        "eval_batches": 8,
        "scenario": "task_il",       # task_il | class_il
    },
    "optim": {
        "lr": 3e-4,
        "weight_decay": 0.01,
        "betas": [0.9, 0.95],
        "grad_clip": 1.0,
        "warmup_steps": 50,
        "rewarm_per_task": False,     # F15: deliberate re-warming at each boundary
    },
    "probes": {
        "enabled": True,
        "every_n_steps": 200,
        "capture_layers": "all",
    },
    "mechanisms": {},   # filled from the registry
}


# ---------------------------------------------------------------------------
def default_config() -> dict[str, Any]:
    cfg = json.loads(json.dumps(BASE_CONFIG))   # deep copy of plain data
    cfg["mechanisms"] = registry.default_config()
    return cfg


def load(path: str | Path | None = None, overrides: list[str] | None = None) -> dict[str, Any]:
    cfg = default_config()
    if path is not None:
        cfg = deep_merge(cfg, read_file(path))
    for item in overrides or []:
        apply_override(cfg, item)
    validate(cfg)
    return cfg


def read_file(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    text = p.read_text()
    if p.suffix in (".yaml", ".yml"):
        if yaml is None:
            raise RuntimeError("pyyaml is required to read YAML configs (pip install pyyaml)")
        return yaml.safe_load(text) or {}
    return json.loads(text)


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _coerce(text: str) -> Any:
    low = text.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    if "," in text:
        return [_coerce(p.strip()) for p in text.split(",")]
    return text


def apply_override(cfg: dict, item: str) -> None:
    """Apply one `a.b.c=value` override in place.

    Supports the flag form the plan calls for:
        --set mech.ewc.enabled=true
        --set mech.ewc.lam=5000
    where `mech.` is an alias for `mechanisms.`.
    """
    if "=" not in item:
        raise ValueError(f"override {item!r} must be key=value")
    key, raw = item.split("=", 1)
    key = key.strip()
    if key.startswith("mech."):
        key = "mechanisms." + key[len("mech."):]
    parts = key.split(".")
    node = cfg
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            raise KeyError(
                f"unknown config path {key!r} (failed at {p!r}). "
                f"Available at this level: {sorted(node)}"
            )
        node = node[p]
    leaf = parts[-1]
    if leaf not in node:
        raise KeyError(
            f"unknown config key {key!r}. Available: {sorted(node)}"
        )
    node[leaf] = _coerce(raw)


# ---------------------------------------------------------------------------
PRESETS: dict[str, dict[str, Any]] = {
    # the forgetting floor: sequential fine-tuning, nothing enabled
    "control_sequential": {},
    # both ceilings are handled by the trainer, not by mechanism flags
    "control_joint": {"_trainer_mode": "joint"},
    "control_independent": {"_trainer_mode": "independent"},
    # single-mechanism ablation targets
    "ewc": {"ewc": {"enabled": True}},
    "replay": {"replay": {"enabled": True}},
    "der": {"der": {"enabled": True}},
    "xdg": {"xdg": {"enabled": True}},
    "kwta": {"kwta": {"enabled": True}},
    "cbp": {"continual_backprop": {"enabled": True}},
    "shrink_perturb": {"shrink_perturb": {"enabled": True}},
    "lwf": {"lwf": {"enabled": True}},
    "si": {"si": {"enabled": True}},
    "gpm": {"gpm": {"enabled": True}},
    "lora": {"lora": {"enabled": True}},
    "olora": {"olora": {"enabled": True}},
    "l2p": {"l2p": {"enabled": True}},
    "memory_layer": {"memory_layer": {"enabled": True}},
    "sparse_update": {"sparse_update": {"enabled": True}},
    "memory_sparse": {
        "memory_layer": {"enabled": True},
        "sparse_update": {"enabled": True, "t": 200},
    },
    # a curated stack: replay + sparsity + plasticity
    "stack_rsp": {
        "replay": {"enabled": True, "ratio": 0.25},
        "kwta": {"enabled": True, "k_fraction": 0.15},
        "continual_backprop": {"enabled": True},
    },
}

# The tier-1 ablation: controls plus one mechanism at a time. Deliberately not a
# cross-product — with ~88 mechanisms, independent flags describe 2^88 runs.
ABLATION_SET = [
    "control_sequential",
    "replay", "der", "lwf",           # F01 rehearsal / distillation
    "ewc", "si",                      # F02 regularization
    "gpm", "olora",                   # F03 optimization geometry
    "xdg", "kwta", "memory_sparse",   # F04 sparsity & localization
    "lora", "l2p",                    # F05 frozen backbone
    "cbp", "shrink_perturb",          # F12 plasticity
]


# ---------------------------------------------------------------------------
# Per-mechanism search spaces.
#
# A ranking at one arbitrary hyperparameter is not a fair test — the first nano
# ablation had EWC at lam=1000, where its own signature probe reported the
# penalty was not biting at all. Every mechanism gets a small sweep over the
# parameter that most controls its strength, and is scored at its best setting.
TUNING_GRIDS: dict[str, dict[str, list[Any]]] = {
    "ewc":            {"mech.ewc.lam": [10, 100, 1000, 10000]},
    "si":             {"mech.si.c": [0.01, 0.1, 1.0, 10.0]},
    "lwf":            {"mech.lwf.lam": [0.1, 0.5, 1.0, 4.0]},
    "replay":         {"mech.replay.ratio": [0.1, 0.25, 0.5]},
    "der":            {"mech.der.alpha": [0.1, 0.5, 1.0]},
    "gpm":            {"mech.gpm.eps_base": [0.80, 0.90, 0.97]},
    "kwta":           {"mech.kwta.k_fraction": [0.05, 0.15, 0.35]},
    "xdg":            {"mech.xdg.gate_fraction": [0.3, 0.5, 0.8]},
    "cbp":            {"mech.continual_backprop.rho": [1e-5, 1e-4, 1e-3]},
    "shrink_perturb": {"mech.shrink_perturb.noise_std": [1e-4, 1e-3, 1e-2]},
    "lora":           {"mech.lora.rank": [4, 16, 64]},
    "olora":          {"mech.olora.lam": [0.05, 0.5, 5.0]},
    "l2p":            {"mech.l2p.pool_size": [10, 20, 40]},
    "memory_sparse":  {"mech.sparse_update.slot_frac": [0.01, 0.03, 0.10]},
}


def grid_points(spec: dict[str, list[Any]]) -> list[list[str]]:
    """Expand a grid spec into a list of --set override lists."""
    import itertools
    keys = sorted(spec)
    out = []
    for combo in itertools.product(*(spec[k] for k in keys)):
        out.append([f"{k}={v}" for k, v in zip(keys, combo)])
    return out


def grid_label(overrides: list[str]) -> str:
    """Compact suffix identifying a grid point, e.g. '[lam=1000]'."""
    parts = []
    for o in overrides:
        key, val = o.split("=", 1)
        parts.append(f"{key.rsplit('.', 1)[-1]}={val}")
    return "[" + ",".join(parts) + "]" if parts else ""


def apply_preset(cfg: dict[str, Any], preset: str) -> dict[str, Any]:
    if preset not in PRESETS:
        raise KeyError(f"unknown preset {preset!r}; known: {sorted(PRESETS)}")
    spec = dict(PRESETS[preset])
    mode = spec.pop("_trainer_mode", None)
    if mode:
        cfg["run"]["mode"] = mode
    cfg["mechanisms"] = deep_merge(cfg["mechanisms"], spec)
    cfg["run"]["name"] = preset if cfg["run"]["name"] == "unnamed" else cfg["run"]["name"]
    return cfg


# ---------------------------------------------------------------------------
def validate(cfg: dict[str, Any]) -> None:
    known = registry.all_mechanisms()
    unknown = set(cfg["mechanisms"]) - set(known)
    if unknown:
        raise KeyError(
            f"config references unregistered mechanisms {sorted(unknown)}; "
            f"registered: {sorted(known)}"
        )
    if cfg["stream"]["scenario"] not in ("task_il", "class_il"):
        raise ValueError("stream.scenario must be task_il or class_il")


def enabled_names(cfg: dict[str, Any]) -> list[str]:
    return sorted(n for n, p in cfg["mechanisms"].items() if p.get("enabled"))


def dump(cfg: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix in (".yaml", ".yml") and yaml is not None:
        p.write_text(yaml.safe_dump(cfg, sort_keys=False))
    else:
        p.write_text(json.dumps(cfg, indent=2))
