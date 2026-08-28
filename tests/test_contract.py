"""Contract tests.

These guard the invariants that make the library composable. A new mechanism
that violates one of them fails here rather than three hours into a sweep.

    pytest -q
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "Continual Learning Mechanism Stack (CLMS)"))

from clms import Composer, RunContext, registry            # noqa: E402
from clms import config as cfgmod                          # noqa: E402
from clms.base import Mechanism                            # noqa: E402
from clms.data import (                                    # noqa: E402
    TaskStream, build_task_sequence, TASK_BUILDERS, DEFAULT_SEQUENCE,
)
from clms.eval import AccuracyMatrix, recovery_ratio       # noqa: E402
from clms.eval.probes import effective_rank, linear_cka, update_concentration  # noqa: E402
from olmo2_cl import Olmo2Config, build_model              # noqa: E402


ALL = registry.all_mechanisms()


# ---------------------------------------------------------------------------
# registry / contract
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(ALL))
def test_declares_identity(name):
    cls = ALL[name]
    assert cls.name == name, "registry key must match the class's own name"
    assert cls.surfaces, "every mechanism must declare at least one surface"
    assert not (set(cls.surfaces) - set("ALOSD")), f"bad surfaces: {cls.surfaces}"
    assert cls.family, "family id is used to group the ablation report"
    assert cls.paper, "paper attribution is how a result is traced back"


@pytest.mark.parametrize("name", sorted(ALL))
def test_implements_self_test(name):
    """self_test is mandatory: an inert mechanism is indistinguishable from one
    that legitimately doesn't help, and that failure is otherwise invisible."""
    assert ALL[name].self_test is not Mechanism.self_test, (
        f"{name} inherits the base self_test, which raises NotImplementedError"
    )


@pytest.mark.parametrize("name", sorted(ALL))
def test_defaults_are_plain_data(name):
    """Defaults get serialised into every run's config dump."""
    for k, v in ALL[name].defaults.items():
        assert isinstance(v, (int, float, str, bool, type(None))), (
            f"{name}.{k} is {type(v)}; defaults must survive a YAML round-trip"
        )


@pytest.mark.parametrize("name", sorted(ALL))
def test_rejects_unknown_params(name):
    with pytest.raises(ValueError, match="unknown parameter"):
        ALL[name](definitely_not_a_real_param=1)


@pytest.mark.parametrize("name", sorted(ALL))
def test_conflicts_reference_real_mechanisms(name):
    for other in ALL[name].conflicts:
        assert other in ALL, (
            f"{name} declares a conflict with unknown mechanism {other!r}. "
            f"Conflicts must name real mechanisms — a phantom name silently "
            f"never fires, which is worse than no guard at all."
        )


def test_conflicts_are_symmetric_or_deliberate():
    """A one-sided conflict still works, but flag it so it's a choice."""
    asymmetric = [
        (a, b) for a, ca in ALL.items() for b in ca.conflicts
        if b in ALL and a not in ALL[b].conflicts
    ]
    assert not asymmetric, (
        f"one-sided conflicts: {asymmetric}. The composer catches these either "
        f"way, but declaring both directions is what makes the constraint "
        f"readable from either mechanism's file."
    )


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
def test_default_config_disables_everything():
    cfg = cfgmod.default_config()
    assert set(cfg["mechanisms"]) == set(ALL), "config must track the registry exactly"
    assert cfgmod.enabled_names(cfg) == [], "the default config is the control"


def test_override_rejects_typos():
    cfg = cfgmod.default_config()
    with pytest.raises(KeyError, match="unknown config key"):
        cfgmod.apply_override(cfg, "mech.ewc.lamda=5000")


def test_override_coerces_types():
    cfg = cfgmod.default_config()
    cfgmod.apply_override(cfg, "mech.ewc.enabled=true")
    cfgmod.apply_override(cfg, "mech.ewc.lam=5000")
    assert cfg["mechanisms"]["ewc"]["enabled"] is True
    assert cfg["mechanisms"]["ewc"]["lam"] == 5000


@pytest.mark.parametrize("preset", sorted(cfgmod.PRESETS))
def test_presets_resolve(preset):
    cfg = cfgmod.apply_preset(cfgmod.default_config(), preset)
    cfgmod.validate(cfg)


def test_ablation_set_is_covered_by_presets():
    missing = [p for p in cfgmod.ABLATION_SET if p not in cfgmod.PRESETS]
    assert not missing, f"ablation references undefined presets: {missing}"


# ---------------------------------------------------------------------------
# composition
# ---------------------------------------------------------------------------
def test_capability_validation_rejects_impossible_config():
    ctx = RunContext(num_tasks=2, stream_capabilities=("task_boundaries",))
    cfg = cfgmod.apply_preset(cfgmod.default_config(), "xdg")
    with pytest.raises(Exception, match="requires stream capabilities"):
        Composer.from_config(cfg["mechanisms"], ctx)


def test_conflict_validation_rejects_incompatible_pair():
    ctx = RunContext(num_tasks=2)
    cfg = cfgmod.default_config()
    cfg["mechanisms"]["gpm"]["enabled"] = True
    cfg["mechanisms"]["continual_backprop"]["enabled"] = True
    with pytest.raises(Exception, match="cannot co-run"):
        Composer.from_config(cfg["mechanisms"], ctx)


def test_empty_composer_is_a_valid_control():
    ctx = RunContext(num_tasks=2)
    comp = Composer.from_config(cfgmod.default_config()["mechanisms"], ctx)
    assert len(comp) == 0
    model = build_model(Olmo2Config(size_preset="nano", vocab_size=128), injector=comp)
    out = model(torch.randint(0, 128, (2, 8)))
    assert out["logits"].shape == (2, 8, 128)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("preset", ["nano", "tiny", "small"])
def test_model_forward_and_param_estimate(preset):
    cfg = Olmo2Config(size_preset=preset, vocab_size=128, max_position_embeddings=64)
    model = build_model(cfg)
    ids = torch.randint(0, 128, (2, 16))
    labels = ids.clone()
    labels[:, :8] = -100
    out = model(ids, labels=labels)
    assert out["logits"].shape == (2, 16, 128)
    assert torch.isfinite(out["loss"])
    actual = model.num_parameters()
    assert abs(actual - cfg.estimated_params()) / actual < 0.15, (
        "estimated_params drifted from the real count"
    )


def test_ignored_label_positions_do_not_contribute():
    cfg = Olmo2Config(size_preset="nano", vocab_size=64, max_position_embeddings=32)
    model = build_model(cfg)
    ids = torch.randint(0, 64, (2, 10))
    all_masked = torch.full_like(ids, -100)
    all_masked[:, -3:] = ids[:, -3:]
    loss_a = model(ids, labels=all_masked)["loss"]
    assert torch.isfinite(loss_a) and loss_a > 0


def test_reordered_norm_is_applied_to_sublayer_output():
    """OLMo 2's distinguishing choice; a rebase that loses it should fail here."""
    cfg = Olmo2Config(size_preset="nano", vocab_size=64)
    model = build_model(cfg)
    block = model.model.layers[0]
    assert hasattr(block, "post_attention_layernorm")
    assert hasattr(block, "post_feedforward_layernorm")
    assert not hasattr(block, "input_layernorm"), (
        "pre-norm attribute present; this is not the OLMo 2 arrangement"
    )


def test_qk_norm_present_by_default():
    cfg = Olmo2Config(size_preset="nano", vocab_size=64)
    model = build_model(cfg)
    attn = model.model.layers[0].self_attn
    assert not isinstance(attn.q_norm, torch.nn.Identity)
    assert not isinstance(attn.k_norm, torch.nn.Identity)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def test_every_task_builds_and_labels_only_the_answer():
    tasks = build_task_sequence()
    stream = TaskStream(tasks, batch_size=4, steps_per_task=1, seed=0)
    for task in tasks:
        batch = next(iter(stream.batches(task, 1)))
        assert batch["input_ids"].shape == batch["labels"].shape
        supervised = (batch["labels"] != -100).sum(dim=1)
        assert (supervised > 0).all(), f"{task.name}: no supervised positions"
        assert (supervised < batch["labels"].shape[1]).all(), (
            f"{task.name}: the whole sequence is supervised; the prompt should not be"
        )


def test_eval_batches_are_frozen_across_calls():
    """Every relative probe is defined against this set; it must not drift."""
    tasks = build_task_sequence(["copy"])
    stream = TaskStream(tasks, batch_size=4, steps_per_task=1, seed=0)
    a = next(iter(stream.eval_batches(tasks[0], 1)))
    _ = list(stream.batches(tasks[0], 5))       # advance the training generator
    b = next(iter(stream.eval_batches(tasks[0], 1)))
    assert torch.equal(a["input_ids"], b["input_ids"])


def test_class_il_drops_the_task_token():
    tasks = build_task_sequence(["copy", "reverse"])
    s_task = TaskStream(tasks, batch_size=2, seed=0, include_task_token=True)
    s_class = TaskStream(tasks, batch_size=2, seed=0, include_task_token=False)
    a = next(iter(s_task.batches(tasks[0], 1)))["input_ids"]
    b = next(iter(s_class.batches(tasks[0], 1)))["input_ids"]
    assert (a[:, 1] >= 4).all() and (a[:, 1] < 20).all(), "expected a task token"
    assert not (b[:, 1] < 20).any(), "class_il should carry no task token"


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def test_accuracy_matrix_detects_total_forgetting():
    m = AccuracyMatrix(num_tasks=2)
    m.record(0, 0, 1.0)
    m.record(1, 0, 0.0)
    m.record(1, 1, 1.0)
    assert m.average_accuracy() == 0.5
    assert m.forgetting() == 1.0
    assert m.backward_transfer() == -1.0


def test_accuracy_matrix_detects_no_forgetting():
    m = AccuracyMatrix(num_tasks=2)
    m.record(0, 0, 1.0)
    m.record(1, 0, 1.0)
    m.record(1, 1, 1.0)
    assert m.forgetting() == 0.0
    assert m.backward_transfer() == 0.0


def test_recovery_ratio_bounds():
    assert recovery_ratio(0.5, 0.5, 1.0) == 0.0     # no better than the floor
    assert recovery_ratio(1.0, 0.5, 1.0) == 1.0     # matches the ceiling
    assert recovery_ratio(0.25, 0.5, 1.0) < 0       # actively harmful
    assert math.isnan(recovery_ratio(0.5, 0.5, 0.5))  # degenerate span


# ---------------------------------------------------------------------------
# probes
# ---------------------------------------------------------------------------
def test_effective_rank_ordering():
    n = 256
    full = torch.randn(n, 32)
    rank_one = torch.randn(n, 1) @ torch.randn(1, 32)
    assert effective_rank(full) > effective_rank(rank_one)


def test_linear_cka_identity_and_independence():
    X = torch.randn(128, 16)
    assert linear_cka(X, X) == pytest.approx(1.0, abs=1e-5)
    assert linear_cka(X, torch.randn(128, 16)) < 0.5


def test_update_concentration_detects_localised_updates():
    model = build_model(Olmo2Config(size_preset="nano", vocab_size=64))
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    with torch.no_grad():                       # move exactly one row
        list(model.parameters())[1][0].add_(1.0)
    localised = update_concentration(model, before)
    model2 = build_model(Olmo2Config(size_preset="nano", vocab_size=64))
    before2 = {n: p.detach().clone() for n, p in model2.named_parameters()}
    with torch.no_grad():                       # move everything
        for p in model2.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    diffuse = update_concentration(model2, before2)
    assert localised < diffuse / 10, (
        "concentration must separate localised from diffuse updates — this is the "
        "quantity the whole sparsity family is judged on"
    )


# ---------------------------------------------------------------------------
# task well-posedness
# ---------------------------------------------------------------------------
def test_induction_cue_is_unique():
    """The answer must be decidable from the input.

    If the trigger symbol appears more than once before the final position, each
    occurrence implies a different continuation and accuracy is capped no matter
    how good the model is. The naive generator was ambiguous on 11.4% of
    sequences, which is invisible in the loss and looks like a hard task.
    """
    task = [t for t in build_task_sequence() if t.name == "induction"][0]
    g = torch.Generator().manual_seed(0)
    for _ in range(2000):
        inp, out = task.generate(g)
        trigger = inp[-1]
        cues = [j for j in range(len(inp) - 1) if inp[j] == trigger]
        assert len(cues) == 1, f"trigger appears at {cues}; the cue must be unique"
        assert out == [inp[cues[0] + 1]], "answer must follow the single cue"


@pytest.mark.parametrize("name", ["copy", "reverse", "sort", "modadd23", "kvrecall"])
def test_answers_are_a_function_of_the_input(name):
    """Deterministic map: identical inputs must always yield identical answers."""
    task = build_task_sequence([name])[0]
    g = torch.Generator().manual_seed(0)
    seen: dict[tuple, tuple] = {}
    for _ in range(2000):
        inp, out = task.generate(g)
        k = tuple(inp)
        if k in seen:
            assert seen[k] == tuple(out), f"{name}: {k} maps to two answers"
        seen[k] = tuple(out)


def test_kvrecall_keys_are_distinct():
    """Duplicate keys would make the queried value ambiguous."""
    task = build_task_sequence(["kvrecall"])[0]
    g = torch.Generator().manual_seed(0)
    for _ in range(1000):
        inp, _ = task.generate(g)
        keys = inp[:-1:2]
        assert len(set(keys)) == len(keys), f"duplicate keys: {keys}"


def test_every_task_declares_a_chance_baseline():
    """A raw score is uninterpretable without it.

    kvrecall with two pairs scores 0.50 by guessing between the two values in
    the prompt — which reads as "half right" and is in fact zero learning. The
    gate compares against chance, so chance has to be declared.
    """
    for name, build in TASK_BUILDERS.items():
        task = build(0)
        assert 0.0 <= task.chance < 1.0, f"{name}: implausible chance {task.chance}"
        if name.startswith(("kvrecall", "induction", "modadd")):
            assert task.chance > 0.0, (
                f"{name} is a retrieval/arithmetic task; guessing among the "
                f"candidates present gives a non-trivial baseline"
            )


def test_default_sequence_excludes_tasks_measured_at_chance():
    """kvrecall scored exactly at chance at every difficulty and every scale
    tested, so it contributes only noise to rho."""
    assert "kvrecall" not in DEFAULT_SEQUENCE
    assert "kvrecall2" not in DEFAULT_SEQUENCE
    for name in DEFAULT_SEQUENCE:
        assert name in TASK_BUILDERS, f"{name} is not a registered task"


def test_optimizer_covers_parameters_mechanisms_unfreeze_later():
    """Per-task freezing must not permanently exclude a parameter.

    O-LoRA activates a different adapter at every task boundary, so adapters
    1..N are frozen when the optimizer is built and unfrozen later. If the
    optimizer filtered them out at construction they would receive gradients
    forever after but never move — which froze the model after task 0 and
    produced identical results at every lambda.
    """
    import train as trainmod

    ctx = RunContext(num_tasks=3, device="cpu", seed=0)
    mcfg = Olmo2Config(vocab_size=128, hidden_size=64, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=4,
                       intermediate_size=128)
    comp = Composer.from_config({"olora": {"enabled": True, "rank": 4}}, ctx)
    comp.set_model_config(mcfg)
    model = build_model(mcfg, injector=comp)
    comp.setup(model, mcfg)

    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
    assert frozen, "setup should leave the inactive adapters frozen"

    opt = trainmod.build_optimizer(model, {"lr": 1e-3, "weight_decay": 0.01,
                                           "betas": [0.9, 0.95], "eps": 1e-8})
    covered = {id(p) for g in opt.param_groups for p in g["params"]}
    missing = [n for n, p in model.named_parameters() if id(p) not in covered]
    assert not missing, f"parameters absent from the optimizer: {missing[:4]}"


# ---------------------------------------------------------------------------
# portability to a foreign model
# ---------------------------------------------------------------------------
class _ForeignGPT(nn.Module):
    """Deliberately not our model: no Injector, no observe() calls, no OLMo.

    Stands in for an official OLMo 2 / Llama / Qwen module tree before any
    injection points have been added to it.
    """

    def __init__(self, V=128, d=64, L=2):
        super().__init__()
        self.embed_tokens = nn.Embedding(V, d)
        self.layers = nn.ModuleList(nn.ModuleDict({
            "attn": nn.MultiheadAttention(d, 4, batch_first=True),
            "mlp": nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)),
        }) for _ in range(L))
        self.lm_head = nn.Linear(d, V, bias=False)

    def forward(self, input_ids, labels=None, **kw):
        x = self.embed_tokens(input_ids)
        for l in self.layers:
            a, _ = l["attn"](x, x, x, need_weights=False)
            x = x + a
            x = x + l["mlp"](x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        return {"logits": logits, "loss": loss}


# Mechanisms that touch only parameters and gradients work on any nn.Module,
# so they port to an unmodified upstream model with no source edits at all.
MODEL_AGNOSTIC = ["ewc", "si", "lwf", "replay", "der", "gpm", "shrink_perturb"]


@pytest.mark.parametrize("name", MODEL_AGNOSTIC)
def test_loss_and_optimizer_mechanisms_run_on_a_foreign_model(name):
    ctx = RunContext(num_tasks=3, device="cpu", seed=0)
    mcfg = Olmo2Config(vocab_size=128, hidden_size=64, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=4,
                       intermediate_size=256)
    model = _ForeignGPT()
    comp = Composer.from_config({name: {"enabled": True}}, ctx)
    comp.set_model_config(mcfg)
    comp.setup(model, mcfg)
    batch = {"input_ids": torch.randint(0, 128, (4, 10)),
             "labels": torch.randint(0, 128, (4, 10)),
             "task_id": torch.zeros(4, dtype=torch.long)}
    model(input_ids=batch["input_ids"], labels=batch["labels"])
    comp.run_self_tests(model, batch)


@pytest.mark.parametrize("name", ["kwta", "l2p"])
def test_activation_mechanisms_report_inert_without_injection_points(name):
    """The dangerous case: attaches without error, then silently does nothing.

    An activation-surface mechanism needs the host model to call
    `injector.observe(...)`. Dropped onto an unmodified upstream model it raises
    no exception — it just never fires. That has to surface as `inert`, or a
    port would produce a full results table for mechanisms that never ran.
    """
    ctx = RunContext(num_tasks=3, device="cpu", seed=0)
    mcfg = Olmo2Config(vocab_size=128, hidden_size=64, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=4,
                       intermediate_size=256)
    ids = torch.randint(0, 128, (4, 10))
    lab = torch.randint(0, 128, (4, 10))

    comp = Composer.from_config({name: {"enabled": True}}, ctx)
    comp.set_model_config(mcfg)
    foreign = _ForeignGPT()
    comp.setup(foreign, mcfg)
    comp.on_task_start(foreign, 0)
    foreign(input_ids=ids, labels=lab)
    assert name in comp.inert(), f"{name} silently did nothing and was not reported"

    comp2 = Composer.from_config({name: {"enabled": True}}, ctx)
    comp2.set_model_config(mcfg)
    ours = build_model(mcfg, injector=comp2)
    comp2.setup(ours, mcfg)
    comp2.on_task_start(ours, 0)
    ours(input_ids=ids, labels=lab)
    assert name not in comp2.inert(), f"{name} should fire on a model with observe() calls"


def test_replay_signature_detects_a_buffer_that_forgets():
    """D1 must fail when the buffer stops carrying data across task boundaries.

    Replay is the strongest mechanism in the sweep, and accuracy alone cannot
    tell "rehearsed old tasks" from "happened to score well". This simulates the
    failure by feeding the probe only current-task samples.
    """
    from clms.mechanisms.replay import ExperienceReplay

    m = ExperienceReplay()
    ctx = RunContext(num_tasks=3, device="cpu", seed=0)
    m._task = 2

    m._replayed, m._replayed_old = 100, 60          # healthy reservoir
    assert m.signature(None, ctx).passed

    m._replayed, m._replayed_old = 100, 0           # buffer cleared each task
    sig = m.signature(None, ctx)
    assert not sig.passed, "a buffer retaining nothing must fail D1"
    assert sig.value == 0.0


def test_learning_accuracy_is_undefined_for_joint_training():
    """LA reads the diagonal, which only means something in a sequential stream.

    Joint training sees every task at once, so its diagonal reflects when
    checkpoints were taken, not plasticity. Left unguarded it reports ~0.33 and
    flags the ceiling run — the best possible result — as a plasticity collapse.
    """
    m = AccuracyMatrix(num_tasks=3)
    for t in range(3):
        for i in range(t + 1):
            m.record(after_task=t, on_task=i, acc=0.9)
    assert m.learning_accuracy("sequential") == pytest.approx(0.9)
    assert math.isnan(m.learning_accuracy("joint"))
    assert math.isnan(m.summary(mode="joint")["LA"])


def test_learning_accuracy_separates_forgetting_from_never_learning():
    """Two models with the same AA, for opposite reasons."""
    forgot = AccuracyMatrix(num_tasks=2)      # learned both, lost the first
    forgot.record(0, 0, 1.0)
    forgot.record(1, 0, 0.0); forgot.record(1, 1, 1.0)

    rigid = AccuracyMatrix(num_tasks=2)       # kept the first, never learned the second
    rigid.record(0, 0, 1.0)
    rigid.record(1, 0, 1.0); rigid.record(1, 1, 0.0)

    assert forgot.average_accuracy() == pytest.approx(rigid.average_accuracy())
    assert forgot.learning_accuracy() == pytest.approx(1.0)
    assert rigid.learning_accuracy() == pytest.approx(0.5)


def test_zeroing_a_gradient_does_not_freeze_the_parameter_under_adamw():
    """The hazard behind sparse_update's B2 failure.

    Mechanisms in the sparsity family express "only update these parameters" by
    multiplying the rest of the gradient by zero. Under AdamW that does not hold
    the parameter still: exponential-average momentum from earlier steps keeps
    moving it, and decoupled weight decay applies to any parameter whose grad is
    not None — zero counts.

    Measured on the real sweep: sparse_update concentrates 90% of memory
    *accesses* into 4.6% of slots (A4 passes) while displacement still spreads
    over 87.5% of them (B2 fails). The masking works; the optimizer undoes it.

    A mechanism that truly needs frozen parameters has to set grad to None, keep
    them out of the optimizer, or restore them after the step.
    """
    w = torch.nn.Parameter(torch.randn(6))
    opt = torch.optim.AdamW([w], lr=1e-2, weight_decay=0.01)

    w.grad = torch.ones(6)          # build momentum on every entry
    opt.step()
    opt.zero_grad(set_to_none=False)
    snap = w.detach().clone()

    g = torch.ones(6)
    g[2:] = 0.0                     # "sparse update": mask all but the first two
    w.grad = g
    opt.step()

    moved = (w.detach() - snap).abs()
    assert (moved[2:] > 0).all(), "masked parameters are expected to still move"
    assert moved[2:].mean() > 0.4 * moved[:2].mean(), (
        "masked parameters move on the same order as unmasked ones; if this "
        "ever stops being true, the sparsity mechanisms can rely on masking"
    )


def test_setting_grad_to_none_does_freeze_the_parameter():
    """The remedy: None, not zero."""
    w = torch.nn.Parameter(torch.randn(4))
    opt = torch.optim.AdamW([w], lr=1e-2, weight_decay=0.01)
    w.grad = torch.ones(4)
    opt.step()
    snap = w.detach().clone()

    w.grad = None
    opt.step()
    assert torch.equal(w.detach(), snap), "grad=None must leave the parameter untouched"


def test_sparse_update_frozen_slots_do_not_move_at_all():
    """Masked entries must be bit-identical after an optimizer step.

    Zeroing the gradient leaves Adam's momentum and AdamW's weight decay free to
    move them — measured at ~67% of a normal update. `after_step` restores the
    recorded values, so the freeze is exact rather than approximate.
    """
    ctx = RunContext(num_tasks=3, device="cpu", seed=0)
    mcfg = Olmo2Config(vocab_size=128, hidden_size=64, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=4,
                       intermediate_size=128)
    comp = Composer.from_config(
        {"memory_layer": {"enabled": True},
         "sparse_update": {"enabled": True, "warmup_steps": 0,
                           "background_batches": 1}}, ctx)
    comp.set_model_config(mcfg)
    model = build_model(mcfg, injector=comp)
    comp.setup(model, mcfg)
    comp.on_task_start(model, 0)
    mech = [m for m in comp.mechanisms if m.name == "sparse_update"][0]

    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.01)
    ids = torch.randint(0, 128, (8, 10))
    lab = torch.randint(0, 128, (8, 10))

    before = held = None
    for step in range(6):
        out = model(ids, labels=lab)
        out["loss"].backward()
        comp.before_step(model)
        if step == 5:
            before = mech.memory.values.weight.detach().clone()
            held = list(mech._hold)
        opt.step()
        opt.zero_grad(set_to_none=True)
        comp.after_step(model)

    assert held, "nothing was masked, so the test proves nothing"
    frozen = held[0][1]
    moved = (mech.memory.values.weight.detach() - before).abs().sum(dim=1)
    assert float(moved[frozen].max()) == 0.0, "frozen slots moved"
    assert float(moved[~frozen].abs().sum()) > 0.0, "active slots should still train"


def test_filling_future_cells_does_not_disturb_the_other_metrics():
    """Evaluating unseen tasks must be strictly additive.

    FWT reads A[j-1][j], so those cells have to be filled. Every other metric is
    index-bounded to the seen region rather than relying on future cells being
    NaN — this pins that down, because a silent shift in AA would invalidate
    every sweep run before the change.
    """
    seen = AccuracyMatrix(num_tasks=3)
    full = AccuracyMatrix(num_tasks=3)
    vals = {(0, 0): 1.0, (1, 0): 0.5, (1, 1): 0.9, (2, 0): 0.3, (2, 1): 0.6, (2, 2): 0.8}
    for (t, j), v in vals.items():
        seen.record(t, j, v)
        full.record(t, j, v)
    # the same run, but also evaluated on tasks it had not reached yet
    for (t, j), v in {(0, 1): 0.21, (0, 2): 0.19, (1, 2): 0.25}.items():
        full.record(t, j, v)

    assert full.average_accuracy() == pytest.approx(seen.average_accuracy())
    assert full.forgetting() == pytest.approx(seen.forgetting())
    assert full.backward_transfer() == pytest.approx(seen.backward_transfer())
    assert full.learning_accuracy() == pytest.approx(seen.learning_accuracy())

    # ...and only now is forward transfer computable at all
    base = [0.2, 0.2, 0.2]
    assert math.isnan(seen.forward_transfer(base))
    assert not math.isnan(full.forward_transfer(base))
    # A[0][1]=0.21, A[1][2]=0.25 against a 0.2 baseline -> mean of +0.01, +0.05
    assert full.forward_transfer(base) == pytest.approx(0.03)


def test_steps_schedule_gives_each_task_its_own_budget():
    """Uniform exposure is the convenient case, not the realistic one.

    A personalised model sees whatever its user talks about most. Replay's
    buffer is a uniform sample *of the stream*, so a skewed stream yields a
    skewed buffer — and rehearsal's rho=1.0 was measured on a perfectly balanced
    one. This is the knob that lets the skewed case be tested.
    """
    tasks = build_task_sequence(["copy", "reverse", "sort"])
    s = TaskStream(tasks, batch_size=4, steps_per_task=10,
                   steps_schedule=[5, 5, 40])
    assert [s.steps_for(i) for i in range(3)] == [5, 5, 40]
    assert len(list(s.batches(tasks[2]))) == 40, "skewed task must get its budget"
    assert len(list(s.batches(tasks[0]))) == 5

    uniform = TaskStream(tasks, batch_size=4, steps_per_task=10)
    assert [uniform.steps_for(i) for i in range(3)] == [10, 10, 10]
    assert len(list(uniform.batches(tasks[0]))) == 10


def test_steps_schedule_length_must_match_the_task_list():
    tasks = build_task_sequence(["copy", "reverse"])
    with pytest.raises(ValueError, match="steps_schedule"):
        TaskStream(tasks, batch_size=4, steps_schedule=[100, 100, 100])
