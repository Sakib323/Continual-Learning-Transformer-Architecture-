#!/usr/bin/env python3
"""Pilot timing: turn a sweep plan into hours and dollars before renting anything.

    python scripts/pilot.py --set model.size_preset=small --tune

Times one short training segment on *this* machine, extrapolates to a full run,
then to the whole sweep. Cheapest de-risking available: a plan that turns out to
be 300 GPU-hours is much better discovered here than on hour three of a rental.

The estimate is only as good as the hardware it runs on — run it on the instance
type you actually intend to rent.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "Continual Learning Mechanism Stack (CLMS)"))

from clms import Composer, RunContext                                  # noqa: E402
from clms import config as cfgmod                                      # noqa: E402
from clms.data import TaskStream, build_task_sequence                  # noqa: E402
from olmo2_cl import Olmo2Config, build_model                          # noqa: E402

# Indicative vast.ai on-demand ranges, USD/hour. Verify before relying on them —
# spot pricing moves and these are a sanity band, not a quote.
GPU_RATES = {"4090": (0.30, 0.55), "A100-40G": (0.80, 1.40), "H100": (1.80, 3.00)}


def measure(cfg: dict, warmup: int = 5, timed: int = 25) -> tuple[float, int, str]:
    rc, mc, sc, oc = cfg["run"], cfg["model"], cfg["stream"], cfg["optim"]
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")

    tasks = build_task_sequence(sc["tasks"])
    stream = TaskStream(tasks, batch_size=sc["batch_size"],
                        steps_per_task=warmup + timed, seed=0,
                        include_task_token=(sc["scenario"] == "task_il"))

    model_cfg = Olmo2Config(
        size_preset=mc["size_preset"], vocab_size=mc["vocab_size"],
        max_position_embeddings=max(mc["max_position_embeddings"], stream.max_len),
    )
    ctx = RunContext(num_tasks=len(tasks), device=device, seed=0)
    comp = Composer.from_config(cfg["mechanisms"], ctx)
    comp.set_model_config(model_cfg)
    model = build_model(model_cfg, injector=comp).to(device)
    comp.setup(model, model_cfg)
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=oc["lr"])
    model.train()

    gen = stream.batches(tasks[0], warmup + timed)
    for i, batch in enumerate(gen):
        if i == warmup:
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
        ctx.step = i
        batch = {k: v.to(device) for k, v in batch.items()}
        batch = comp.on_batch(batch, i)
        out = model(batch["input_ids"], labels=batch["labels"])
        loss, _ = comp.compute_loss(model, batch, out, out["loss"])
        loss.backward()
        comp.before_step(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), oc["grad_clip"])
        opt.step()
        opt.zero_grad(set_to_none=True)
        comp.after_step(model)
    if device == "cuda":
        torch.cuda.synchronize()
    per_step = (time.time() - t0) / timed
    return per_step, model.num_parameters(), device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--preset", default="control_sequential",
                    help="time this configuration; mechanisms add real overhead")
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tune", action="store_true",
                    help="cost the tuned sweep rather than the default-hyperparameter one")
    args = ap.parse_args()

    cfg = cfgmod.load(args.config, args.overrides)
    cfg = cfgmod.apply_preset(cfg, args.preset)

    per_step, params, device = measure(cfg)
    sc = cfg["stream"]
    steps_per_run = sc["steps_per_task"] * len(sc["tasks"])
    run_seconds = per_step * steps_per_run

    n_configs = 0
    for preset in cfgmod.ABLATION_SET:
        grid = cfgmod.TUNING_GRIDS.get(preset) if args.tune else None
        n_configs += len(cfgmod.grid_points(grid)) if grid else 1
    n_configs += 1                      # control_joint
    total_runs = n_configs * args.seeds
    sweep_hours = total_runs * run_seconds / 3600

    print(f"\n{'=' * 62}")
    print(f"  measured on: {device}   model: {params:,} params "
          f"({cfg['model']['size_preset']})")
    print(f"  timing config: {args.preset}")
    print(f"{'=' * 62}")
    print(f"  per step            {per_step * 1000:8.1f} ms")
    print(f"  steps per run       {steps_per_run:8,}  "
          f"({sc['steps_per_task']} x {len(sc['tasks'])} tasks)")
    print(f"  one run             {run_seconds / 60:8.1f} min")
    print(f"{'-' * 62}")
    mode = "tuned" if args.tune else "default hyperparameters"
    print(f"  sweep ({mode})")
    print(f"  configurations      {n_configs:8,}")
    print(f"  x seeds             {args.seeds:8,}")
    print(f"  total runs          {total_runs:8,}")
    print(f"  wall time         {sweep_hours:10.1f} GPU-hours")
    print(f"{'-' * 62}")
    for gpu, (lo, hi) in GPU_RATES.items():
        print(f"  {gpu:<10} ${sweep_hours * lo:8.0f} - ${sweep_hours * hi:.0f}")
    print(f"{'=' * 62}")

    if device != "cuda":
        print("\n  NOTE: measured on a non-CUDA device. Re-run this on the instance")
        print("  type you intend to rent — the ratio is not reliably transferable.")
    if not args.tune:
        print("\n  This costs the sweep at *default* hyperparameters. The first nano")
        print("  ablation showed several mechanisms failing their own signature")
        print("  probes at defaults, so add --tune for the number that matters.")
    if sweep_hours > 24:
        print(f"\n  {sweep_hours:.0f} GPU-hours is more than a day. Options: drop to a")
        print("  smaller size_preset, cut steps_per_task, or trim the grid — the")
        print("  ranking is what you need, and it survives scaling down further")
        print("  than the absolute numbers do.")


if __name__ == "__main__":
    main()
