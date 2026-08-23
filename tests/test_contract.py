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

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "Continual Learning Mechanism Stack (CLMS)"))

from clms import Composer, RunContext, registry            # noqa: E402
from clms import config as cfgmod                          # noqa: E402
from clms.base import Mechanism                            # noqa: E402
from clms.data import TaskStream, build_task_sequence      # noqa: E402
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
