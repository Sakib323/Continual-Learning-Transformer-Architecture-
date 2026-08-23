#!/usr/bin/env python3
"""Ablation sweep and ranking report.

    # controls + one mechanism at a time, at default hyperparameters
    python scripts/sweep.py --preset ablation --seeds 0,1,2

    # the fair version: each mechanism swept over its own search space,
    # scored at its best setting
    python scripts/sweep.py --preset ablation --tune --seeds 0,1,2

    # one mechanism, explicit grid
    python scripts/sweep.py --preset ewc --grid mech.ewc.lam=10,100,1000,10000

    # rank whatever is already in runs/
    python scripts/sweep.py --report-only

Design note: this deliberately does *not* offer a cross-product over mechanisms.
With ~88 of them, independent on/off flags describe 2^88 configurations. The
useful shape is a ladder — controls, then one at a time (optionally tuned), then
a few curated stacks.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "Continual Learning Mechanism Stack (CLMS)"))

from clms.config import (                                              # noqa: E402
    ABLATION_SET, PRESETS, TUNING_GRIDS, grid_label, grid_points,
)
from clms.eval.metrics import recovery_ratio                           # noqa: E402


BASE_RE = re.compile(r"^(?P<base>[^\[]+)(?P<grid>\[.*\])?$")


def base_name(run_name: str) -> str:
    m = BASE_RE.match(run_name)
    return m.group("base") if m else run_name


# ---------------------------------------------------------------------------
def run_one(preset: str, seed: int, extra: list[str], out_dir: str,
            label: str = "") -> None:
    name = preset + label
    cmd = [
        sys.executable, str(REPO / "train.py"),
        "--preset", preset, "--seed", str(seed),
        "--name", name, "--set", f"run.out_dir={out_dir}",
    ]
    for e in extra:
        cmd += ["--set", e]
    print(f"\n{'=' * 70}\n[sweep] {name} seed={seed}\n{'=' * 70}")
    if subprocess.run(cmd, cwd=REPO).returncode != 0:
        print(f"[sweep] FAILED: {name} seed={seed}")


def plan(presets: list[str], seeds: list[int], tune: bool,
         explicit_grid: dict[str, list] | None) -> list[tuple[str, list[str], str]]:
    """(preset, overrides, label) for every run to perform."""
    jobs: list[tuple[str, list[str], str]] = []
    for preset in presets:
        if explicit_grid:
            points = grid_points(explicit_grid)
        elif tune and preset in TUNING_GRIDS:
            points = grid_points(TUNING_GRIDS[preset])
        else:
            points = [[]]
        for pt in points:
            jobs.append((preset, pt, grid_label(pt)))
    return jobs


# ---------------------------------------------------------------------------
def collect(out_dir: str) -> dict[str, list[dict]]:
    runs: dict[str, list[dict]] = {}
    for path in sorted(Path(out_dir).glob("*/seed*/result.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        runs.setdefault(data["config_name"], []).append(data)
    return runs


def agg(runs: dict[str, list[dict]], name: str, key: str) -> tuple[float, float]:
    vals = [
        r["metrics"][key] for r in runs.get(name, [])
        if r["metrics"].get(key) == r["metrics"].get(key)
    ]
    if not vals:
        return float("nan"), 0.0
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def report(runs: dict[str, list[dict]], best_only: bool = True) -> None:
    if not runs:
        print("no results found")
        return

    floor, _ = agg(runs, "control_sequential", "AA")
    ceiling, _ = agg(runs, "control_joint", "AA")
    has_bounds = floor == floor and ceiling == ceiling

    print(f"\n{'=' * 100}")
    print("RANKING".ljust(30) + "AA (mean±sd)".ljust(20) + "FM".ljust(10)
          + "BWT".ljust(11) + "rho".ljust(9) + "cost".ljust(9) + "n  signatures")
    print("=" * 100)
    if has_bounds:
        print(f"{'floor (sequential)':<30}{floor:.4f}")
        print(f"{'ceiling (joint)':<30}{ceiling:.4f}   span={ceiling - floor:+.4f}")
        if abs(ceiling - floor) < 0.05:
            print("  !! floor and ceiling within 0.05 — the stream is too easy or too "
                  "hard, and every mechanism will look identical")
        print("-" * 100)

    rows = []
    for name in runs:
        if name.startswith("control_"):
            continue
        aa, sd = agg(runs, name, "AA")
        fm, _ = agg(runs, name, "FM")
        bwt, _ = agg(runs, name, "BWT")
        rho = recovery_ratio(aa, floor, ceiling) if has_bounds else float("nan")
        r0 = runs[name][0]
        added = sum(c.get("added_params", 0) for c in r0.get("costs", {}).values())
        buf = sum(c.get("buffer_bytes", 0) for c in r0.get("costs", {}).values())
        overhead = (added / max(r0.get("params", 1), 1)) + (buf / 4 / max(r0.get("params", 1), 1))
        sigs = r0.get("signatures", [])
        sig_str = " ".join(
            f"{s['probe']}:{'ok' if s['passed'] else 'FAIL' if s['passed'] is False else '-'}"
            for s in sigs
        ) or "-"
        rows.append({
            "name": name, "base": base_name(name), "aa": aa, "sd": sd, "fm": fm,
            "bwt": bwt, "rho": rho, "overhead": overhead, "n": len(runs[name]),
            "sig": sig_str,
        })

    if best_only:
        # one line per mechanism, at its best hyperparameter setting
        by_base: dict[str, dict] = {}
        for r in rows:
            cur = by_base.get(r["base"])
            if cur is None or (r["aa"] == r["aa"] and r["aa"] > cur["aa"]):
                by_base[r["base"]] = r
        shown = list(by_base.values())
    else:
        shown = rows

    shown.sort(key=lambda r: (r["rho"] if r["rho"] == r["rho"] else -9e9), reverse=True)
    for r in shown:
        rho_s = f"{r['rho']:+.3f}" if r["rho"] == r["rho"] else "  -  "
        print(f"{r['name']:<30}{r['aa']:.4f}±{r['sd']:.3f}     {r['fm']:<10.3f}"
              f"{r['bwt']:<+11.3f}{rho_s:<9}{r['overhead']:<9.2f}{r['n']}  {r['sig']}")

    print("=" * 100)
    print("rho: 0 = no better than sequential fine-tuning, 1 = matches joint training.")
    print("cost: added params + buffer, as a multiple of model size.")

    # --- the two readings that are easy to get wrong ----------------------
    frozen = [r for r in shown if r["fm"] < 0.05 and r["rho"] < 0.05]
    if frozen:
        print("\nRETAINED BUT DID NOT LEARN — zero forgetting with zero recovery means")
        print("the model froze on task 0. On a forgetting-only metric these look like")
        print("perfect scores; they are the opposite failure (intransigence):")
        for r in frozen:
            print(f"  {r['name']}: FM={r['fm']:.3f} but rho={r['rho']:+.3f}")

    failed = [r for r in shown if "FAIL" in r["sig"]]
    if failed:
        print("\nSIGNATURE FAILURES — the mechanism did not do the internal thing its")
        print("paper claims, so its score reflects a misconfiguration, not the method:")
        for r in failed:
            print(f"  {r['name']}: {r['sig']}")

    overlapping = []
    for i in range(len(shown) - 1):
        a, b = shown[i], shown[i + 1]
        if a["aa"] - a["sd"] <= b["aa"] + b["sd"]:
            overlapping.append(f"{a['name']} ~ {b['name']}")
    if overlapping:
        print(f"\nindistinguishable at this seed count: {', '.join(overlapping)}")
        if all(r["n"] < 3 for r in shown):
            print("  (every configuration has fewer than 3 seeds — treat the whole")
            print("   ordering as provisional; CL rankings routinely flip between seeds)")


# ---------------------------------------------------------------------------
def parse_grid(items: list[str]) -> dict[str, list]:
    spec: dict[str, list] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--grid expects key=v1,v2,...; got {item!r}")
        key, vals = item.split("=", 1)
        spec[key.strip()] = [v.strip() for v in vals.split(",")]
    return spec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="ablation",
                    help="'ablation' for the standard set, or a single preset name")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--set", dest="extra", action="append", default=[],
                    help="forwarded to train.py, e.g. --set model.size_preset=small")
    ap.add_argument("--tune", action="store_true",
                    help="sweep each mechanism over its search space in TUNING_GRIDS")
    ap.add_argument("--grid", action="append", default=[],
                    help="explicit grid, e.g. --grid mech.ewc.lam=10,100,1000")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--all-points", action="store_true",
                    help="report every grid point rather than each mechanism's best")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = ap.parse_args()

    if not args.report_only:
        presets = ABLATION_SET if args.preset == "ablation" else [args.preset]
        unknown = [p for p in presets if p not in PRESETS]
        if unknown:
            raise SystemExit(f"unknown presets {unknown}; known: {sorted(PRESETS)}")
        seeds = [int(s) for s in args.seeds.split(",")]
        explicit = parse_grid(args.grid) if args.grid else None

        jobs = plan(presets, seeds, args.tune, explicit)
        # rho needs both bounds. The floor is as necessary as the ceiling: without
        # it a mechanism can only be said to beat nothing, which ranks nothing.
        for control in ("control_sequential", "control_joint"):
            if control not in presets:
                jobs.append((control, [], ""))
        total = len(jobs) * len(seeds)
        print(f"[sweep] {len(jobs)} configurations x {len(seeds)} seeds = {total} runs")
        if args.dry_run:
            for preset, ov, label in jobs:
                print(f"  {preset + label:<34} {' '.join(ov)}")
            print(f"\nrun scripts/pilot.py first to convert {total} runs into hours")
            return

        for preset, overrides, label in jobs:
            for seed in seeds:
                run_one(preset, seed, args.extra + overrides, args.out_dir, label)

    report(collect(args.out_dir), best_only=not args.all_points)


if __name__ == "__main__":
    main()
