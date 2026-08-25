#!/usr/bin/env python3
"""CLMS trainer.

    # the forgetting floor — every mechanism off
    python train.py --preset control_sequential

    # the ceiling
    python train.py --preset control_joint

    # one mechanism, with hyperparameters
    python train.py --set mech.ewc.enabled=true --set mech.ewc.lam=5000

    # a curated stack
    python train.py --preset stack_rsp --seed 1

    # what is registered?
    python train.py --list

Three run modes, because "all flags false" is a floor and a floor alone ranks
nothing:
    sequential   task after task            -> the floor (and the test condition)
    joint        all tasks mixed together   -> the ceiling
    independent  a fresh model per task     -> second ceiling, isolates sharing
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "Continual Learning Mechanism Stack (CLMS)"))

from clms import Composer, RunContext, registry            # noqa: E402
from clms import config as cfgmod                          # noqa: E402
from clms.data import TaskStream, build_task_sequence      # noqa: E402
from clms.eval import (                                    # noqa: E402
    AccuracyMatrix, ActivationRecorder, ProbeRunner, sequence_accuracy, snapshot,
)
from olmo2_cl import Olmo2Config, build_model              # noqa: E402


# ---------------------------------------------------------------------------
def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model, oc: dict):
    """Every parameter goes in, including ones currently frozen.

    Mechanisms freeze and unfreeze parameters *per task* — O-LoRA activates a
    different adapter at each boundary, LoRA freezes the backbone. Filtering on
    requires_grad here captures the state at construction time and permanently
    excludes anything a mechanism unfreezes later: it still receives gradients,
    but no optimizer state, so it never moves.

    That silently froze O-LoRA after task 0 across a whole sweep — identical
    results at every lambda, FM exactly 0.0, CKA exactly 1.0. AdamW skips
    parameters whose grad is None, so including everything is safe and the
    requires_grad flags keep doing the actual gating.
    """
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.dim() <= 1 else decay).append(p)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": oc["weight_decay"]},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=oc["lr"],
        betas=tuple(oc["betas"]),
    )


def lr_at(step: int, total: int, oc: dict) -> float:
    warm = oc["warmup_steps"]
    if step < warm:
        return oc["lr"] * (step + 1) / max(warm, 1)
    prog = (step - warm) / max(total - warm, 1)
    return oc["lr"] * 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))


# ---------------------------------------------------------------------------
def evaluate_all(model, stream, upto: int, device: str, matrix: AccuracyMatrix,
                 after_task: int, eval_batches: int) -> dict[str, float]:
    """Evaluate on every task seen so far — the whole accuracy matrix row.

    Evaluating only the current task is the most common way to accidentally
    measure nothing.
    """
    accs = {}
    for j in range(upto + 1):
        task = stream.tasks[j]
        acc = sequence_accuracy(
            model, stream.eval_batches(task, eval_batches), device
        )
        matrix.record(after_task, j, acc)
        accs[f"acc/{task.name}"] = acc
    return accs


# ---------------------------------------------------------------------------
def train(cfg: dict) -> dict:
    rc, mc, sc, oc = cfg["run"], cfg["model"], cfg["stream"], cfg["optim"]
    device = pick_device(rc["device"])
    set_seed(rc["seed"])
    mode = rc.get("mode", "sequential")

    tasks = build_task_sequence(sc["tasks"])
    stream = TaskStream(
        tasks,
        batch_size=sc["batch_size"],
        steps_per_task=sc["steps_per_task"],
        seed=rc["seed"],
        include_task_token=(sc["scenario"] == "task_il"),
    )

    model_cfg = Olmo2Config(
        size_preset=mc["size_preset"],
        vocab_size=mc["vocab_size"],
        max_position_embeddings=max(mc["max_position_embeddings"], stream.max_len),
        tie_word_embeddings=mc["tie_word_embeddings"],
    )

    ctx = RunContext(
        num_tasks=len(tasks),
        device=device,
        seed=rc["seed"],
        stream_capabilities=(
            ("task_boundaries", "task_ids") if sc["scenario"] == "task_il"
            else ("task_boundaries",)
        ),
    )
    composer = Composer.from_config(cfg["mechanisms"], ctx)

    recorder = ActivationRecorder()
    ctx.scratch["_observers"] = [recorder]
    ctx.scratch["recorder"] = recorder

    composer.set_model_config(model_cfg)      # surface-A mechanisms size themselves
    model = build_model(model_cfg, injector=composer).to(device)
    composer.setup(model, model_cfg)
    model.to(device)                          # mechanisms may have added modules

    prober = ProbeRunner(model, stream, device, recorder)

    out_dir = Path(rc["out_dir"]) / rc["name"] / f"seed{rc['seed']}"
    out_dir.mkdir(parents=True, exist_ok=True)

    result_path = out_dir / "result.json"
    if rc.get("skip_completed", True) and result_path.exists():
        print(f"[clms] {rc['name']} seed{rc['seed']} already complete — skipping")
        return json.loads(result_path.read_text())

    cfgmod.dump(cfg, out_dir / "config.yaml")

    print(f"[clms] device={device}  mode={mode}  params={model.num_parameters():,}")
    print(f"[clms] tasks={[t.name for t in tasks]}  scenario={sc['scenario']}")
    print(f"[clms] mechanisms: {composer.names or ['none — control]']}")

    # --- self-tests before any real work -------------------------------
    probe_batch = next(iter(stream.batches(tasks[0], 1)))
    probe_batch = {k: v.to(device) for k, v in probe_batch.items()}
    if len(composer):
        composer.run_self_tests(model, probe_batch)
        print(f"[clms] self-tests passed for {len(composer)} mechanism(s)")

    matrix = AccuracyMatrix(num_tasks=len(tasks))
    optimizer = build_optimizer(model, oc)
    history: list[dict] = []
    t_start = time.time()
    ckpt_path = out_dir / "checkpoint.pt"

    # --- resume ---------------------------------------------------------
    # Rented instances get interrupted. A resumed run whose Fisher matrix or
    # replay buffer was lost is silently a different experiment, so mechanism
    # state is restored alongside the model.
    start_task = 0
    if rc.get("resume", True) and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        composer.load_state_dict(ck["mechanisms"])
        matrix = AccuracyMatrix(num_tasks=len(tasks), A=ck["matrix"]["A"])
        history = ck.get("history", [])
        start_task = ck["task_idx"] + 1
        if ck.get("reference_captured"):
            prober.capture_reference(0)
        print(f"[clms] resumed from {ckpt_path} at task {start_task}")

    # ------------------------------------------------------------------
    if mode == "joint":
        total = sc["steps_per_task"] * len(tasks)
        gens = [stream.batches(t, total) for t in tasks]
        model.train()
        for step in range(total):
            batch = next(gens[step % len(tasks)])
            batch = {k: v.to(device) for k, v in batch.items()}
            for g in optimizer.param_groups:
                g["lr"] = lr_at(step, total, oc)
            out = model(batch["input_ids"], labels=batch["labels"])
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), oc["grad_clip"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if step % 100 == 0:
                print(f"  joint step {step}/{total} loss={float(out['loss'].detach()):.4f}")
        for j in range(len(tasks)):
            matrix.record(len(tasks) - 1, j, sequence_accuracy(
                model, stream.eval_batches(tasks[j], sc["eval_batches"]), device))

    elif mode == "independent":
        for i, task in enumerate(tasks):
            set_seed(rc["seed"])
            m = build_model(model_cfg).to(device)
            opt = build_optimizer(m, oc)
            m.train()
            for step, batch in enumerate(stream.batches(task)):
                batch = {k: v.to(device) for k, v in batch.items()}
                for g in opt.param_groups:
                    g["lr"] = lr_at(step, sc["steps_per_task"], oc)
                out = m(batch["input_ids"], labels=batch["labels"])
                out["loss"].backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), oc["grad_clip"])
                opt.step()
                opt.zero_grad(set_to_none=True)
            acc = sequence_accuracy(m, stream.eval_batches(task, sc["eval_batches"]), device)
            matrix.record(len(tasks) - 1, i, acc)
            print(f"  [independent] {task.name}: {acc:.4f}")

    else:  # sequential — the continual setting
        global_step = start_task * sc["steps_per_task"]
        for task_idx, task in enumerate(tasks):
            if task_idx < start_task:
                continue
            composer.on_task_start(model, task_idx)
            ctx.task_id = task_idx
            params_before = snapshot(model)
            ctx.scratch["params_at_task_start"] = params_before
            if oc["rewarm_per_task"] and task_idx > 0:
                for g in optimizer.param_groups:
                    g["lr"] = oc["lr"]

            model.train()
            for step, batch in enumerate(stream.batches(task)):
                ctx.step = global_step
                batch = {k: v.to(device) for k, v in batch.items()}
                batch = composer.on_batch(batch, global_step)

                for g in optimizer.param_groups:
                    g["lr"] = lr_at(
                        step if oc["rewarm_per_task"] else global_step,
                        sc["steps_per_task"] if oc["rewarm_per_task"]
                        else sc["steps_per_task"] * len(tasks),
                        oc,
                    )

                out = model(batch["input_ids"], labels=batch["labels"])
                loss, parts = composer.compute_loss(model, batch, out, out["loss"])
                loss.backward()

                composer.before_step(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(), oc["grad_clip"])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                composer.after_step(model)

                if step % 100 == 0:
                    extra = " ".join(f"{k}={v:.3g}" for k, v in parts.items())
                    print(f"  [{task.name}] step {step} loss={float(out['loss'].detach()):.4f} {extra}")
                global_step += 1

            # boundary work: Fisher estimation, SVD bases, buffer resizing
            ctx.scratch["fisher_batches"] = list(stream.eval_batches(task, 8))
            composer.on_task_end(model, task_idx)

            row = evaluate_all(model, stream, task_idx, device, matrix,
                               task_idx, sc["eval_batches"])
            entry = {"task": task.name, "task_idx": task_idx, **row}
            if cfg["probes"]["enabled"]:
                # the reference is frozen right after task 0, so every later
                # drift measurement is against the same fixed point
                if task_idx == 0 and not prober._captured:
                    prober.capture_reference(0)
                entry.update(prober.run(task_idx, params_before))
            entry.update(composer.on_eval(model, task_idx))
            history.append(entry)
            print(f"  -> after {task.name}: AA={matrix.average_accuracy(task_idx):.4f}")

            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "mechanisms": composer.state_dict(),
                    "task_idx": task_idx,
                    "matrix": matrix.as_dict(),
                    "history": history,
                    "reference_captured": prober._captured,
                },
                ckpt_path,
            )

    # ------------------------------------------------------------------
    inert = composer.inert()
    if inert:
        print(f"[clms] WARNING: enabled but never acted: {inert}")

    sigs = [
        {
            "probe": s.probe, "quantity": s.quantity, "value": s.value,
            "baseline": s.baseline, "direction": s.direction,
            "passed": s.evaluate(), "detail": s.detail,
        }
        for s in composer.signatures(model)
    ]

    result = {
        "config_name": rc["name"],
        "seed": rc["seed"],
        "mode": mode,
        "mechanisms": composer.names,
        "params": model.num_parameters(),
        "wall_seconds": time.time() - t_start,
        "metrics": matrix.summary(mode=mode),
        "matrix": matrix.as_dict(),
        "history": history,
        "signatures": sigs,
        "costs": composer.costs(),
        "inert_mechanisms": inert,
        "device": device,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    result_path.write_text(json.dumps(result, indent=2, default=str))

    # the checkpoint has served its purpose once the result is on disk
    if not rc.get("keep_checkpoint", False) and ckpt_path.exists():
        ckpt_path.unlink()

    m = result["metrics"]
    print(f"\n[clms] AA={m['AA']:.4f}  FM={m['FM']:.4f}  BWT={m['BWT']:+.4f}")
    for s in sigs:
        flag = {True: "PASS", False: "FAIL", None: "  - "}[s["passed"]]
        print(f"[clms] signature {s['probe']} {flag}  {s['quantity']}={s['value']:.4g}")
    print(f"[clms] results -> {out_dir/'result.json'}")
    return result


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="CLMS trainer")
    ap.add_argument("--config", type=str, default=None, help="YAML/JSON config file")
    ap.add_argument("--preset", type=str, default=None, help=f"one of {sorted(cfgmod.PRESETS)}")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    help="dotted override, e.g. --set mech.ewc.enabled=true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--name", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--mode", type=str, default=None,
                    choices=["sequential", "joint", "independent"])
    ap.add_argument("--list", action="store_true", help="list registered mechanisms and exit")
    ap.add_argument("--dry-run", action="store_true", help="validate config, build model, exit")
    args = ap.parse_args()

    if args.list:
        print(f"{'name':<22} {'surfaces':<8} {'fam':<5} {'paper':<12}")
        print("-" * 64)
        for line in registry.describe():
            print(line)
        print(f"\npresets: {sorted(cfgmod.PRESETS)}")
        return

    cfg = cfgmod.load(args.config, args.overrides)
    if args.preset:
        cfg = cfgmod.apply_preset(cfg, args.preset)
    if args.seed is not None:
        cfg["run"]["seed"] = args.seed
    if args.name:
        cfg["run"]["name"] = args.name
    if args.device:
        cfg["run"]["device"] = args.device
    if args.mode:
        cfg["run"]["mode"] = args.mode

    if args.dry_run:
        ctx = RunContext(num_tasks=len(cfg["stream"]["tasks"]))
        comp = Composer.from_config(cfg["mechanisms"], ctx)
        mcfg = Olmo2Config(size_preset=cfg["model"]["size_preset"],
                           vocab_size=cfg["model"]["vocab_size"])
        print(f"[dry-run] ok. enabled={comp.names or 'none'}")
        print(f"[dry-run] model ~{mcfg.estimated_params():,} params "
              f"(h={mcfg.hidden_size}, L={mcfg.num_hidden_layers})")
        return

    train(cfg)


if __name__ == "__main__":
    main()
