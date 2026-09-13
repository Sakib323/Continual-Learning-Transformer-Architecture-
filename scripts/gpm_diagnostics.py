#!/usr/bin/env python3
"""GPM audit diagnostics — reproduces every number in gpm/AUDIT.md. CPU only.

    python scripts/gpm_diagnostics.py --section A      # eval-set leakage
    python scripts/gpm_diagnostics.py --section B      # PAD dilution
    python scripts/gpm_diagnostics.py --section C      # LR actually seen per task
    python scripts/gpm_diagnostics.py --section D      # AdamW step leak, ~8 min
    python scripts/gpm_diagnostics.py --section E      # rank rule: current vs paper, ~2 min
    python scripts/gpm_diagnostics.py --section F      # protected-subspace accuracy, ~5 min
    python scripts/gpm_diagnostics.py --section G      # current vs corrected GPM, 6 tasks, ~25 min
    python scripts/gpm_diagnostics.py --section all

Sections D-G train a `tiny` (4.5M) model on CPU. Numbers are single-seed
unless stated; they are diagnostics, not results.

`GPMFixed` at the bottom is a prototype of the corrected mechanism (paper rank
rule, PAD rows masked, train-batch collection, post-step projection so the
weight step is orthogonal under AdamW, lm_head projected, seen-symbol embedding
rows and norm gains frozen). It is registered as `gpm_fixed` only while this
script runs; the production version belongs in clms/mechanisms/projection.py.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "Continual Learning Mechanism Stack (CLMS)"))

from clms import Composer, RunContext, registry                       # noqa: E402
from clms.data import TaskStream, build_task_sequence, make_batch     # noqa: E402
from clms.data.synthetic import PAD                                   # noqa: E402
from clms.eval import sequence_accuracy                               # noqa: E402
from clms.mechanisms.projection import GradientProjectionMemory       # noqa: E402
from olmo2_cl import Olmo2Config, build_model                         # noqa: E402
import train as trainmod                                              # noqa: E402

DEV = "cpu"
_STEPS = [300, 200]   # default steps for sections D and E; overridden by --steps
LONG = ["copy", "modadd7", "reverse", "sort", "modadd13", "induction6",
        "copy12", "sortdesc", "modadd23", "reverse12", "induction8", "modadd31"]
OC = {"lr": 3e-4, "weight_decay": 0.01, "betas": [0.9, 0.95], "grad_clip": 1.0, "warmup_steps": 50}


def banner(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78, flush=True)


# ============================================================================
# sections A-C
# ============================================================================
def section_A():
    banner("A · eval-set leakage: fisher/basis batches vs eval batches")
    tasks = build_task_sequence(LONG)
    stream = TaskStream(tasks, batch_size=32, steps_per_task=400, seed=0, include_task_token=False)
    t0 = tasks[0]
    fisher = list(stream.eval_batches(t0, 8))          # what train.py hands to on_task_end
    evalb = list(stream.eval_batches(t0, 2))           # what evaluate_all scores on (eval_batches=2 in runs_A)
    same = all(torch.equal(f["input_ids"], e["input_ids"]) for f, e in zip(fisher, evalb))
    print(f"first {len(evalb)} fisher batches identical to the eval batches: {same}")
    print("=> GPM (collect_batches=4) and EWC build their protection from the exact inputs they are scored on.")

    # ---------------------------------------------------------------------------

def section_B():
    banner("B · PAD dilution of the collected activation rows")
    print(f"stream.max_len = {stream.max_len}")
    for t in tasks:
        b = next(iter(stream.eval_batches(t, 1)))
        ids = b["input_ids"]
        pad_frac = float((ids == PAD).float().mean())
        ans_frac = float((b["labels"] != -100).float().mean())
        print(f"  {t.name:<11} seq_len={ids.shape[1]:>3}  PAD rows={pad_frac:5.1%}   answer rows={ans_frac:5.1%}")

    # ---------------------------------------------------------------------------

def section_C():
    banner("C · learning rate actually used per task (rewarm_per_task=False, global cosine)")
    oc = {"lr": 3e-4, "warmup_steps": 50}
    for n_tasks, spt in [(12, 400), (5, 400)]:
        total = n_tasks * spt
        print(f"  {n_tasks} tasks x {spt} steps:")
        for i in range(n_tasks):
            s0, s1 = i * spt, (i + 1) * spt - 1
            print(f"    task {i:>2}: lr {trainmod.lr_at(s0, total, oc):.2e} -> {trainmod.lr_at(s1, total, oc):.2e}"
                  f"   ({trainmod.lr_at(s0, total, oc)/3e-4:4.0%} -> {trainmod.lr_at(s1, total, oc)/3e-4:4.0%} of peak)")



# ============================================================================
# section D helpers (mini trainer with leak logging)
# ============================================================================
def make_run(preset="nano", tasks_names=("copy", "modadd7"), steps=300, eps_base=0.9, seed=0):
    torch.manual_seed(seed)
    tasks = build_task_sequence(list(tasks_names))
    stream = TaskStream(tasks, batch_size=32, steps_per_task=steps, seed=seed, include_task_token=False)
    mcfg = Olmo2Config(size_preset=preset, vocab_size=128,
                       max_position_embeddings=max(128, stream.max_len), tie_word_embeddings=True)
    ctx = RunContext(num_tasks=len(tasks), device=DEV, seed=seed, stream_capabilities=("task_boundaries",))
    gpm = GradientProjectionMemory(eps_base=eps_base, eps_growth=0.0)
    comp = Composer([gpm], ctx)
    comp.set_model_config(mcfg)
    model = build_model(mcfg, injector=comp).to(DEV)
    comp.setup(model, mcfg)
    return model, stream, tasks, ctx, comp, gpm


def projected_layers(gpm):
    return {n: m for n, m in gpm._layers.items()}


def train_task(model, stream, task, task_idx, ctx, comp, opt, steps, oc,
               rewarm=True, leak_log=None, gpm=None, freeze=()):
    model.train()
    total = steps
    for step, batch in enumerate(stream.batches(task, steps)):
        ctx.step = step
        for g in opt.param_groups:
            g["lr"] = trainmod.lr_at(step, total, oc)
        out = model(batch["input_ids"], labels=batch["labels"])
        loss, _ = comp.compute_loss(model, batch, out, out["loss"])
        loss.backward()
        comp.before_step(model)
        for n, p in model.named_parameters():
            if any(n.startswith(f) or f in n for f in freeze):
                p.grad = None
        torch.nn.utils.clip_grad_norm_(model.parameters(), oc["grad_clip"])
        if leak_log is not None and gpm is not None and gpm.bases:
            before = {n: m.weight.detach().clone() for n, m in gpm._layers.items() if n in gpm.bases}
        opt.step()
        if leak_log is not None and gpm is not None and gpm.bases:
            num = den = 0.0
            for n, m in gpm._layers.items():
                if n not in gpm.bases:
                    continue
                dW = m.weight.detach() - before[n]
                M = gpm.bases[n]
                num += float((dW @ M).norm() ** 2)
                den += float(dW.norm() ** 2)
            leak_log.append(math.sqrt(num / den) if den > 0 else float("nan"))
        opt.zero_grad(set_to_none=True)
        comp.after_step(model)


def acc(model, stream, task, n=4):
    return sequence_accuracy(model, stream.eval_batches(task, n), DEV)


def snapshot_named(model, names):
    return {n: p.detach().clone() for n, p in model.named_parameters() if n in names}


_OC_D = {"lr": 3e-4, "weight_decay": 0.01, "betas": [0.9, 0.95], "grad_clip": 1.0, "warmup_steps": 50}


def run_variant(label, optimizer="adamw", freeze=(), post_project=False, preset="tiny",
                task_names=("copy", "modadd7", "reverse"), steps=None, eps_base=0.9, seed=0):
    steps = steps or _STEPS[0]
    model, stream, tasks, ctx, comp, gpm = make_run(preset, task_names, steps, eps_base, seed)
    if optimizer == "adamw":
        opt = trainmod.build_optimizer(model, OC)
    elif optimizer == "adam_nowd":
        opt = trainmod.build_optimizer(model, {**OC, "weight_decay": 0.0})
    elif optimizer == "sgdm":
        opt = torch.optim.SGD(model.parameters(), lr=OC["lr"], momentum=0.9)
    else:
        raise ValueError(optimizer)

    if post_project:
        # Make the *weight step* orthogonal to M, whatever the optimizer did.
        pre: dict[str, torch.Tensor] = {}
        orig_before, orig_after = gpm.before_step, gpm.after_step

        def before_step(model_, ctx_):
            orig_before(model_, ctx_)
            for n, m in gpm._layers.items():
                if n in gpm.bases:
                    pre[n] = m.weight.detach().clone()

        @torch.no_grad()
        def after_step(model_, ctx_):
            for n, m in gpm._layers.items():
                if n in gpm.bases:
                    M = gpm.bases[n].to(m.weight.dtype)
                    dW = m.weight - pre[n]
                    m.weight.copy_(pre[n] + dW - (dW @ M) @ M.T)
            orig_after(model_, ctx_)

        gpm.before_step, gpm.after_step = before_step, after_step

    oc = dict(OC)
    if optimizer == "sgdm":
        oc["lr"] = 0.05
    A = [[float("nan")] * len(tasks) for _ in tasks]
    leaks_per_task = []
    drift = {}
    for i, task in enumerate(tasks):
        comp.on_task_start(model, i)
        leak_log: list[float] = []
        names_all = [n for n, _ in model.named_parameters()]
        snap = snapshot_named(model, names_all)
        train_task(model, stream, task, i, ctx, comp, opt, steps, oc, leak_log=leak_log, gpm=gpm, freeze=freeze)
        if leak_log:
            leaks_per_task.append((task.name, sum(leak_log) / len(leak_log), max(leak_log)))
        if i > 0:
            # relative movement per parameter group during this task
            groups = {"embed/lm_head": [], "norm gains": [], "projected linears": []}
            for n, p in model.named_parameters():
                rel = float((p.detach() - snap[n]).norm() / (snap[n].norm() + 1e-12))
                if "embed_tokens" in n or "lm_head" in n:
                    groups["embed/lm_head"].append(rel)
                elif "norm" in n:
                    groups["norm gains"].append(rel)
                elif p.dim() == 2:
                    groups["projected linears"].append(rel)
            drift[task.name] = {k: (sum(v) / len(v) if v else float("nan")) for k, v in groups.items()}
        ctx.scratch["fisher_batches"] = list(stream.eval_batches(task, 8))
        comp.on_task_end(model, i)
        for j in range(i + 1):
            A[i][j] = acc(model, stream, tasks[j])
    # summary
    last = len(tasks) - 1
    diag = [A[j][j] for j in range(len(tasks))]
    final = A[last]
    fm = sum(max(A[i][j] for i in range(j, last)) - A[last][j] for j in range(last)) / max(last, 1)
    print(f"\n--- {label}")
    print(f"    LA (diag) = {[round(x, 2) for x in diag]}   final row = {[round(x, 2) for x in final]}   FM = {fm:.3f}")
    for name, mean_l, max_l in leaks_per_task:
        print(f"    during {name:<9} ||dW·M||/||dW|| per step: mean {mean_l:.4f}  max {max_l:.4f}")
    for tname, g in drift.items():
        print(f"    during {tname:<9} rel. movement  " + "  ".join(f"{k}={v:.4f}" for k, v in g.items()))
    sat = gpm._consumed_fraction()
    print(f"    basis occupancy after stream: {sat:.3f}")
    return A, fm


# ---------------------------------------------------------------------------

def section_D():
    banner("D · does the WEIGHT STEP respect dW·M = 0, or only the gradient?  (tiny, 3 tasks, 300 steps)")
    t_start = time.time()
    run_variant("AdamW (as in train.py)                     ", optimizer="adamw")
    run_variant("Adam, weight_decay=0                       ", optimizer="adam_nowd")
    run_variant("SGD+momentum lr=0.05                       ", optimizer="sgdm")
    run_variant("AdamW + step projected AFTER optimizer     ", optimizer="adamw", post_project=True)
    run_variant("AdamW + freeze embed & norms               ", optimizer="adamw", freeze=("embed_tokens", "norm"))
    run_variant("AdamW + post-projection + freeze embed/norm", optimizer="adamw", post_project=True, freeze=("embed_tokens", "norm"))
    print(f"\n[section D took {time.time() - t_start:.0f}s]")



# ============================================================================
# section E
# ============================================================================
def _e_banner():
    banner("E · rank selection: current rule (threshold on residual) vs GPM paper rule (threshold on total)")


class CountingGPM(GradientProjectionMemory):
    name = "gpm_counting"

    def __init__(self, **kw):
        super().__init__(**kw)
        self.log: list[tuple[str, int, int, int]] = []   # (task, layer_idx, k_current, k_paper)

    def _extend_basis(self, name, R, eps):
        existing = self.bases.get(name)
        Rd = R.double()
        total_energy = float((Rd ** 2).sum())
        if existing is not None:
            Rp = Rd - (Rd @ existing.double()) @ existing.double().T
            captured = total_energy - float((Rp ** 2).sum())
        else:
            Rp, captured = Rd, 0.0
        try:
            _, S, _ = torch.linalg.svd(Rp.T @ Rp)
        except Exception:
            return
        # current rule: fraction of the *residual* spectrum
        csum_res = torch.cumsum(S, 0) / max(float(S.sum()), 1e-30)
        k_cur = int((csum_res < eps).sum()) + 1
        # paper rule: captured + sum_{i<=k} sigma_i >= eps * total   (k may be 0)
        target = eps * total_energy
        if captured >= target:
            k_paper = 0
        else:
            csum_tot = captured + torch.cumsum(S, 0)
            k_paper = int((csum_tot < target).sum()) + 1
        in_dim = R.shape[1]
        cap = int(in_dim * self.params["max_bases_frac"])
        used = existing.shape[1] if existing is not None else 0
        self.log.append((name, min(k_cur, cap - used), min(k_paper, cap - used), in_dim))
        super()._extend_basis(name, R, eps)


registry._REGISTRY["gpm_counting"] = CountingGPM


def run_counting(preset="tiny", task_names=tuple(LONG[:6]), steps=None, eps_base=0.9):
    steps = steps or _STEPS[1]
    torch.manual_seed(0)
    tasks = build_task_sequence(list(task_names))
    stream = TaskStream(tasks, batch_size=32, steps_per_task=steps, seed=0, include_task_token=False)
    mcfg = Olmo2Config(size_preset=preset, vocab_size=128, max_position_embeddings=max(128, stream.max_len))
    ctx = RunContext(num_tasks=len(tasks), device=DEV, seed=0, stream_capabilities=("task_boundaries",))
    gpm = CountingGPM(eps_base=eps_base, eps_growth=0.0)
    comp = Composer([gpm], ctx)
    comp.set_model_config(mcfg)
    model = build_model(mcfg, injector=comp)
    comp.setup(model, mcfg)
    opt = trainmod.build_optimizer(model, OC)
    cum_cur = cum_paper = 0
    for i, task in enumerate(tasks):
        comp.on_task_start(model, i)
        n0 = len(gpm.log)
        train_task(model, stream, task, i, ctx, comp, opt, steps, OC)
        ctx.scratch["fisher_batches"] = list(stream.eval_batches(task, 8))
        comp.on_task_end(model, i)
        rows = gpm.log[n0:]
        kc = sum(r[1] for r in rows); kp = sum(r[2] for r in rows); dims = sum(r[3] for r in rows)
        cum_cur += kc; cum_paper += kp
        print(f"  after {task.name:<11} new dirs: current rule {kc:>5} ({kc/dims:5.1%} of widths)   "
              f"paper rule {kp:>5} ({kp/dims:5.1%})   cumulative {cum_cur/dims:5.1%} vs {cum_paper/dims:5.1%}")



def section_E():
    _e_banner()
    t_start = time.time()
    run_counting()
    print(f"\n[section E took {time.time() - t_start:.0f}s]")
    print("\nNote: the basis actually used above follows the CURRENT rule; the paper-rule column is what")
    print("the same activations would have consumed under the reference criterion.")

# ============================================================================
# corrected-GPM prototype and sections F-G
# ============================================================================
class GPMFixed(GradientProjectionMemory):
    """GPM with the audit's fixes applied. Prototype, not the production edit."""
    name = "gpm_fixed"

    def __init__(self, **kw):
        super().__init__(**kw)
        self._mask: torch.Tensor | None = None
        self._pre: dict[str, torch.Tensor] = {}
        self._seen_symbols: set[int] = set()
        self._embed: nn.Parameter | None = None
        self._stream = None
        self._task_of = {}

    def setup(self, model, cfg, ctx):
        self._hook_rng = self.rng(ctx)
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and module.in_features >= self.params["min_features"]:
                self._layers[name] = module            # lm_head included
                self._handles.append(module.register_forward_hook(self._make_hook(name)))
        self._embed = model.model.embed_tokens.weight

    def _make_hook(self, name):
        def hook(module, inputs, output):
            if not self._collecting:
                return
            x = inputs[0].detach()
            flat = x.reshape(-1, x.shape[-1]).float().cpu()
            if self._mask is not None and self._mask.numel() == flat.shape[0]:
                flat = flat[self._mask]                # drop PAD positions
            if flat.shape[0] > 512:
                idx = torch.randperm(flat.shape[0], generator=self._hook_rng)[:512]
                flat = flat[idx]
            self._acts.setdefault(name, []).append(flat)
        return hook

    # --- rank rule from the paper: threshold on TOTAL energy, k may be 0 ----
    def _extend_basis(self, name, R, eps):
        existing = self.bases.get(name)
        Rd = R.double()
        total = float((Rd ** 2).sum())
        if total <= 0:
            return
        if existing is not None:
            E = existing.double()
            Rp = Rd - (Rd @ E) @ E.T
            captured = total - float((Rp ** 2).sum())
        else:
            Rp, captured = Rd, 0.0
        target = eps * total
        if captured >= target:
            return
        try:
            U, S, _ = torch.linalg.svd(Rp.T @ Rp)
        except Exception:
            return
        csum = captured + torch.cumsum(S, 0)
        k = int((csum < target).sum()) + 1
        in_dim = R.shape[1]
        cap = int(in_dim * self.params["max_bases_frac"])
        used = existing.shape[1] if existing is not None else 0
        k = max(0, min(k, cap - used))
        if k == 0:
            return
        new = U[:, :k].float()
        self.bases[name] = new if existing is None else torch.cat([existing, new], dim=1)

    # --- collect from fresh TRAINING-distribution batches, PAD rows masked ---
    @torch.no_grad()
    def on_task_end(self, model, task_id, ctx):
        stream = ctx.scratch["stream"]
        task = stream.tasks[task_id]
        g = torch.Generator().manual_seed(ctx.seed + 7_777 + task_id)
        self._acts.clear()
        self._collecting = True
        was_training = model.training
        model.eval()
        for _ in range(self.params["collect_batches"]):
            b = make_batch(task, stream.batch_size, g, stream.max_len, stream.include_task_token)
            ids = b["input_ids"]
            self._seen_symbols.update(int(v) for v in ids.unique())
            self._mask = (ids != PAD).reshape(-1)
            model(ids.to(ctx.device))
        self._mask = None
        model.train(was_training)
        self._collecting = False
        eps = min(0.99, max(self.params["eps_floor"],
                            self.params["eps_base"] + self.params["eps_growth"] * self._tasks_seen))
        for name, parts in self._acts.items():
            R = torch.cat(parts, dim=0)
            if R.shape[0] >= 2:
                self._extend_basis(name, R, eps)
        self._acts.clear()
        self._tasks_seen += 1
        self._saturation_history.append(self._consumed_fraction())

    # --- gradient projection + old-symbol rows + norm gains frozen -----------
    @torch.no_grad()
    def before_step(self, model, ctx):
        if not self.bases:
            return
        super().before_step(model, ctx)
        if self._embed is not None and self._embed.grad is not None and self._seen_symbols:
            rows = torch.tensor(sorted(self._seen_symbols))
            self._embed.grad[rows] = 0.0
        for n, p in model.named_parameters():
            if "norm" in n:
                p.grad = None
        self._pre = {n: m.weight.detach().clone() for n, m in self._layers.items() if n in self.bases}
        self._pre_embed = self._embed.detach().clone() if self._embed is not None else None

    # --- make the WEIGHT STEP orthogonal, whatever AdamW did -----------------
    @torch.no_grad()
    def after_step(self, model, ctx):
        if not self.bases:
            return
        for n, m in self._layers.items():
            if n in self.bases:
                M = self.bases[n].to(m.weight.dtype)
                dW = m.weight - self._pre[n]
                m.weight.copy_(self._pre[n] + dW - (dW @ M) @ M.T)
        if self._embed is not None and self._seen_symbols:
            rows = torch.tensor(sorted(self._seen_symbols))
            self._embed[rows] = self._pre_embed[rows]
        self.mark_ran()


registry._REGISTRY["gpm_fixed"] = GPMFixed



registry._REGISTRY["gpm_fixed"] = GPMFixed


def build(mech, preset, task_names, steps, seed, **mkw):
    torch.manual_seed(seed)
    tasks = build_task_sequence(list(task_names))
    stream = TaskStream(tasks, batch_size=32, steps_per_task=steps, seed=seed, include_task_token=False)
    mcfg = Olmo2Config(size_preset=preset, vocab_size=128, max_position_embeddings=max(128, stream.max_len))
    ctx = RunContext(num_tasks=len(tasks), device=DEV, seed=seed, stream_capabilities=("task_boundaries",))
    ctx.scratch["stream"] = stream
    mechs = [] if mech is None else [registry.get(mech)(**mkw)]
    comp = Composer(mechs, ctx)
    comp.set_model_config(mcfg)
    model = build_model(mcfg, injector=comp).to(DEV)
    comp.setup(model, mcfg)
    return model, stream, tasks, ctx, comp, (mechs[0] if mechs else None)


def _train_task2(model, stream, task, ctx, comp, opt, steps):
    model.train()
    for step, batch in enumerate(stream.batches(task, steps)):
        ctx.step = step
        for g in opt.param_groups:
            g["lr"] = trainmod.lr_at(step, steps, OC)
        out = model(batch["input_ids"], labels=batch["labels"])
        loss, _ = comp.compute_loss(model, batch, out, out["loss"])
        loss.backward()
        comp.before_step(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), OC["grad_clip"])
        opt.step()
        opt.zero_grad(set_to_none=True)
        comp.after_step(model)


def acc(model, stream, task, n=4):
    return sequence_accuracy(model, stream.eval_batches(task, n), DEV)



def section_F(preset="tiny", steps=300, seed=0):
    banner("F · protected-subspace accuracy after training `copy` (does the 'unprotected' energy carry the answer?)")
    for variant in ("current (eval batches, PAD rows kept)", "fixed collection (train batches, PAD rows dropped)"):
        print(f"\n  --- {variant}")
        for eps in (0.80, 0.90, 0.97, 0.99):
            mech = "gpm" if variant.startswith("current") else "gpm_fixed"
            model, stream, tasks, ctx, comp, gpm = build(mech, preset, ("copy",), steps, seed, eps_base=eps, eps_growth=0.0)
            opt = trainmod.build_optimizer(model, OC)
            _train_task2(model, stream, tasks[0], ctx, comp, opt, steps)
            ctx.scratch["fisher_batches"] = list(stream.eval_batches(tasks[0], 8))
            comp.on_task_end(model, 0)
            base_acc = acc(model, stream, tasks[0])
            # spectrum of one representative layer
            name = "model.layers.0.self_attn.q_proj"
            M = gpm.bases[name]
            # measure top-1 energy share at that layer on eval data
            gpm._acts.clear(); gpm._collecting = True
            if isinstance(gpm, GPMFixed):
                b = next(iter(stream.eval_batches(tasks[0], 1))); gpm._mask = (b["input_ids"] != PAD).reshape(-1)
                model(b["input_ids"]); gpm._mask = None
            else:
                model(next(iter(stream.eval_batches(tasks[0], 1)))["input_ids"])
            gpm._collecting = False
            R = torch.cat(gpm._acts[name]).double()
            S = torch.linalg.svdvals(R)
            e = S ** 2 / (S ** 2).sum()
            Rc = R - R.mean(0, keepdim=True)
            Sc = torch.linalg.svdvals(Rc)
            ec = Sc ** 2 / (Sc ** 2).sum()
            # restrict every projected layer's input to its basis
            handles = []
            for n, m in gpm._layers.items():
                if n in gpm.bases:
                    Mb = gpm.bases[n]
                    def pre_hook(mod, inputs, Mb=Mb):
                        x = inputs[0]
                        return ((x @ Mb) @ Mb.T,)
                    handles.append(m.register_forward_pre_hook(pre_hook))
            proj_acc = acc(model, stream, tasks[0])
            for h in handles:
                h.remove()
            occ = gpm._consumed_fraction()
            print(f"    eps={eps:.2f}  occupancy={occ:5.1%}  k(q_proj L0)={M.shape[1]:>3}/{M.shape[0]}   "
                  f"acc(copy)={base_acc:.2f}  acc with inputs restricted to span(M)={proj_acc:.2f}   "
                  f"top-1 energy share: uncentered {float(e[0]):.2f}, centered {float(ec[0]):.2f}")


def section_G(preset="tiny", steps=300, n_tasks=6, seeds=(0, 1)):
    banner(f"G · current GPM vs corrected GPM, {preset}, first {n_tasks} tasks of the long stream, {steps} steps/task, per-task LR")
    task_names = tuple(LONG[:n_tasks])
    variants = [
        ("control (no mechanism)", None, {}),
        ("gpm current eps=0.8", "gpm", dict(eps_base=0.8, eps_growth=0.0)),
        ("gpm current eps=0.9", "gpm", dict(eps_base=0.9, eps_growth=0.0)),
        ("gpm FIXED   eps=0.9", "gpm_fixed", dict(eps_base=0.9, eps_growth=0.0)),
        ("gpm FIXED   eps=0.97", "gpm_fixed", dict(eps_base=0.97, eps_growth=0.0)),
    ]
    for label, mech, mkw in variants:
        aas, fms, las, occs = [], [], [], []
        for seed in seeds:
            t0 = time.time()
            model, stream, tasks, ctx, comp, gpm = build(mech, preset, task_names, steps, seed, **mkw)
            opt = trainmod.build_optimizer(model, OC)
            A = [[float("nan")] * len(tasks) for _ in tasks]
            for i, task in enumerate(tasks):
                comp.on_task_start(model, i)
                _train_task2(model, stream, task, ctx, comp, opt, steps)
                ctx.scratch["fisher_batches"] = list(stream.eval_batches(task, 8))
                comp.on_task_end(model, i)
                for j in range(i + 1):
                    A[i][j] = acc(model, stream, tasks[j])
            last = len(tasks) - 1
            aa = sum(A[last]) / len(tasks)
            la = sum(A[j][j] for j in range(len(tasks))) / len(tasks)
            fm = sum(max(A[i][j] for i in range(j, last)) - A[last][j] for j in range(last)) / last
            occ = gpm._consumed_fraction() if gpm is not None else 0.0
            aas.append(aa); fms.append(fm); las.append(la); occs.append(occ)
            print(f"    {label:<24} seed{seed}: AA={aa:.3f} LA={la:.3f} FM={fm:.3f} occupancy={occ:.3f}  "
                  f"final row={[round(x, 2) for x in A[last]]}  ({time.time() - t0:.0f}s)", flush=True)
        m = lambda v: sum(v) / len(v)
        print(f"  >> {label:<24} mean over {len(seeds)} seeds: AA={m(aas):.3f}  LA={m(las):.3f}  FM={m(fms):.3f}  occupancy={m(occs):.3f}", flush=True)




if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", default="all", help="A|B|C|D|E|F|G|all")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--preset", default="tiny")
    a = ap.parse_args()
    secs = list("ABCDEFG") if a.section == "all" else [a.section.upper()]
    if a.steps != 300:
        _STEPS[0] = _STEPS[1] = a.steps
    for s in secs:
        if s == "A": section_A()
        elif s == "B": section_B()
        elif s == "C": section_C()
        elif s == "D": section_D()
        elif s == "E": section_E()
        elif s == "F": section_F(a.preset, a.steps)
        elif s == "G": section_G(a.preset, a.steps)
        else: raise SystemExit(f"unknown section {s}")
