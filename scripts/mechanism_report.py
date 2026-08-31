#!/usr/bin/env python3
"""Per-mechanism status report for a sweep directory.

    python scripts/mechanism_report.py runs_v4
    python scripts/mechanism_report.py runs_v4 --compare runs_v3

Answers, for every mechanism in one table: did it work, did it learn, did it do
what its paper claims, and what did it cost. Reads only `result.json` files, so
it runs anywhere the sweep output has been copied to.

Why these columns:

  rho   recovery ratio. 0 = no better than sequential fine-tuning, 1 = matching
        joint training. Normalised against controls from the *same* sweep, so it
        is comparable across runs on different hardware where raw AA is not.
  LA%   learning accuracy vs the sequential control. Below 100% means the
        mechanism cost plasticity — the model stopped being able to learn new
        tasks. AA alone cannot separate "learned then forgot" from "never
        learned", and the second looks like partial success without this.
  FWT   forward transfer: accuracy on a task before training on it, above that
        task's chance baseline. Positive means earlier tasks helped.
  sig   signature probes passed / attempted. A mechanism can score well without
        having done the thing its paper describes; this is the independent check.
  MB    persistent state the mechanism carries between tasks — replay buffers,
        Fisher matrices, projection bases, adapter weights. Scoring a method
        that stores 56GB against one that stores nothing, and calling the first
        better, compares budgets rather than mechanisms.
  rho/MB  retention bought per megabyte retained. Mechanisms that store nothing
        are reported as "free" rather than as an infinite ratio.
  n     seeds completed. A single-seed row reports sd 0.000, which reads as
        perfect precision and is in fact one unreplicated run — so n is shown
        and any row below the modal seed count is marked.
  cfg   grid points searched. The table reports each mechanism's *best* point,
        so a 4-point grid gets four draws from its own noise and keeps the
        maximum. That is worth roughly half a standard deviation over a 1-point
        grid, which is the same size as the gaps between mechanisms.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path


def load(root: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for cfg in sorted(root.iterdir()):
        if not cfg.is_dir() or cfg.name.startswith("."):
            continue
        runs = [json.loads(p.read_text()) for p in sorted(cfg.glob("seed*/result.json"))]
        if runs:
            out[cfg.name] = runs
    return out


def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def learning_accuracy(A):
    vals = [A[j][j] for j in range(len(A)) if not math.isnan(A[j][j])]
    return mean(vals)


def summarise(runs: dict[str, list[dict]]) -> dict:
    if "control_sequential" not in runs or "control_joint" not in runs:
        raise SystemExit("need control_sequential and control_joint in the sweep")
    floor = mean([r["metrics"]["AA"] for r in runs["control_sequential"]])
    ceil = mean([r["metrics"]["AA"] for r in runs["control_joint"]])
    span = ceil - floor
    ctrl_la = mean([learning_accuracy(r["matrix"]["A"]) for r in runs["control_sequential"]])

    rows = {}
    for name, rs in runs.items():
        if name.startswith("control"):
            continue
        aa = [r["metrics"]["AA"] for r in rs]
        sok = sum(1 for r in rs for s in r.get("signatures", []) if s["passed"])
        stot = sum(1 for r in rs for s in r.get("signatures", []))
        state = mean([
            sum((c.get("buffer_bytes") or 0) + (c.get("added_params") or 0) * 4
                for c in r.get("costs", {}).values())
            for r in rs
        ]) / 1e6
        rows[name] = {
            "state_mb": state,
            "aa": mean(aa),
            "sd": st.stdev(aa) if len(aa) > 1 else 0.0,
            "la": mean([learning_accuracy(r["matrix"]["A"]) for r in rs]) / ctrl_la,
            "fm": mean([r["metrics"]["FM"] for r in rs]),
            "fwt": mean([r["metrics"].get("FWT") for r in rs]),
            "sig": (sok, stot),
            "inert": any(r.get("inert_mechanisms") for r in rs),
            "n": len(rs),
        }
    return {"floor": floor, "ceiling": ceil, "span": span, "rows": rows}


def verdict(rho: float, rsd: float, la: float, sig: tuple[int, int], inert: bool) -> str:
    if inert:
        return "INERT — never fired"
    if sig[1] and sig[0] < sig[1]:
        return f"SIGNATURE FAIL {sig[0]}/{sig[1]} — score not trustworthy"
    if rho > 0.8:
        return "works — matches joint training"
    if rho - rsd > 0.1:
        return "partial" + (f" (plasticity {la:.0%})" if la < 0.9 else "")
    if rho - rsd > 0:
        return "marginal" + (f" (plasticity {la:.0%})" if la < 0.9 else "")
    if abs(rho) <= rsd:
        return "no effect"
    return "harmful" + (f" (plasticity {la:.0%})" if la < 0.9 else "")


def best_per_family(rows: dict) -> dict:
    fam: dict[str, tuple[str, dict]] = {}
    for name, r in rows.items():
        m = name.split("[")[0]
        if m not in fam or r["aa"] > fam[m][1]["aa"]:
            fam[m] = (name, r)
    return fam


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--compare", help="an earlier sweep directory to diff rho against")
    ap.add_argument("--all-points", action="store_true",
                    help="every grid point rather than each mechanism's best")
    args = ap.parse_args()

    cur = summarise(load(Path(args.run_dir)))
    prev = summarise(load(Path(args.compare))) if args.compare else None

    print(f"\n{'=' * 100}")
    print(f"  MECHANISM REPORT — {args.run_dir}")
    print(f"  floor {cur['floor']:.4f}   ceiling {cur['ceiling']:.4f}   "
          f"span {cur['span']:+.4f}")
    if cur["span"] < 0.05:
        print("  !! span below 0.05 — the benchmark cannot separate mechanisms.")
    print(f"{'=' * 100}\n")

    items = (cur["rows"].items() if args.all_points
             else [(n, r) for n, r in best_per_family(cur["rows"]).values()])
    ordered = sorted(items, key=lambda kv: -kv[1]["aa"])

    # the mode, not the max: GPM deliberately draws extra seeds, and taking the
    # max would flag every correctly-run mechanism as under-sampled
    counts = Counter(r["n"] for r in cur["rows"].values())
    modal_n = counts.most_common(1)[0][0] if counts else 0
    hdr = (f"{'mechanism':<26}{'rho':>7}{'±sd':>7}{'n':>3}{'LA%':>6}{'FM':>7}"
           f"{'FWT':>8}{'sig':>7}{'MB':>9}{'rho/MB':>9}")
    if prev:
        hdr += f"{'Δrho':>8}"
    print(hdr + "   verdict")
    print("-" * 100)

    for name, r in ordered:
        rho = (r["aa"] - cur["floor"]) / cur["span"]
        rsd = r["sd"] / cur["span"]
        sig = f"{r['sig'][0]}/{r['sig'][1]}" if r["sig"][1] else "none"
        mb = r["state_mb"]
        per = "free" if mb < 1e-6 else (f"{rho / mb:9.3f}" if rho > 0 else "     -")
        nmark = f"{r['n']}" + ("!" if r["n"] < modal_n else "")
        line = (f"{name:<26}{rho:7.3f}{rsd:7.3f}{nmark:>3}{r['la']:6.0%}"
                f"{r['fm']:7.3f}{r['fwt']:8.3f}{sig:>7}{mb:9.1f}{per:>9}")
        if prev:
            pr = prev["rows"].get(name)
            if pr:
                prho = (pr["aa"] - prev["floor"]) / prev["span"]
                line += f"{rho - prho:+8.3f}"
            else:
                line += f"{'new':>8}"
        print(line + f"   {verdict(rho, rsd, r['la'], r['sig'], r['inert'])}")

    thin = sorted({n.split("[")[0] for n, r in cur["rows"].items() if r["n"] < modal_n})
    if thin:
        print(f"\nFewer than {modal_n} seeds (marked !): {', '.join(thin)}")
        print("  -> sd is not meaningful for these; do not rank them.")

    grid = defaultdict(int)
    for n in cur["rows"]:
        grid[n.split("[")[0]] += 1
    sizes = sorted(set(grid.values()))
    if len(sizes) > 1:
        print(f"\nUneven grid sizes {sizes}: " + ", ".join(
            f"{m}={c}" for m, c in sorted(grid.items(), key=lambda kv: kv[1])))
        print("  -> the table shows each mechanism's BEST point, so a larger grid")
        print("     is a selection advantage, not a better mechanism.")

    missing = [n for n, r in cur["rows"].items() if not r["sig"][1]]
    if missing:
        print(f"\nNo signature probe defined ({len(missing)}): "
              f"{', '.join(sorted({m.split('[')[0] for m in missing}))}")
        print("  -> score with no independent check that the mechanism engaged.")

    free = [(n, (r["aa"] - cur["floor"]) / cur["span"])
            for n, r in cur["rows"].items() if r["state_mb"] < 1e-6]
    if free:
        best_free = max(free, key=lambda kv: kv[1])
        print(f"\nBest mechanism storing NOTHING: {best_free[0]} at rho {best_free[1]:.3f}")
        print("  -> the bar any stateful mechanism has to clear to justify its cost.")

    fwts = [r["fwt"] for r in cur["rows"].values() if not math.isnan(r["fwt"])]
    if fwts:
        print(f"\nForward transfer across all mechanisms: "
              f"min {min(fwts):+.3f}  max {max(fwts):+.3f}  mean {mean(fwts):+.3f}")
        if max(fwts) <= 0.02:
            print("  -> no mechanism learns new tasks faster for having seen earlier ones.")
    print()


if __name__ == "__main__":
    main()
