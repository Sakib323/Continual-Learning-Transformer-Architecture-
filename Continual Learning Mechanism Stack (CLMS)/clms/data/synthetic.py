"""Synthetic algorithmic task sequence.

Chosen over TRACE for the ranking sweep because at 20-50M parameters TRACE's
tasks sit near chance on both sides of training, which collapses the
floor-ceiling gap that every mechanism comparison depends on. These give:

  * exact ground truth      -> unambiguous accuracy, no metric arguments
  * tunable task overlap    -> probes A3/A4 become controlled experiments
  * minutes to train        -> the whole reason a 50-run sweep is affordable
  * known internal structure -> a moving probe is usually interpretable

Sequence layout
---------------
    [BOS] [TASK_k] <input> [SEP] <output> [EOS]

Loss and accuracy are computed on the output span only, so a task is scored on
what it was asked to produce rather than on reproducing its own prompt.
Dropping the [TASK_k] token turns Task-IL into Class-IL, which is how the
three-scenario evaluation (probe D4) is implemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

# --- reserved vocabulary ---------------------------------------------------
PAD, BOS, EOS, SEP = 0, 1, 2, 3
TASK_TOKEN_START = 4
MAX_TASKS = 16
DATA_START = TASK_TOKEN_START + MAX_TASKS   # 20
NUM_SYMBOLS = 108                           # data symbols 20..127
VOCAB_SIZE = DATA_START + NUM_SYMBOLS       # 128, matching model.vocab_size


def sym(i: int) -> int:
    return DATA_START + (i % NUM_SYMBOLS)


# ---------------------------------------------------------------------------
@dataclass
class Task:
    name: str
    task_id: int
    generate: Callable[[torch.Generator], tuple[list[int], list[int]]]
    max_len: int
    # Exact-match accuracy a model gets from the best *trivial* strategy — for
    # retrieval tasks, guessing among the candidates actually present in the
    # prompt. Without this a score is uninterpretable: kvrecall with 2 pairs
    # scores 0.50 by guessing, which reads as "half right" and is in fact zero
    # learning. Comparisons in the gate are against this, not against zero.
    chance: float = 0.0

    def sample(self, g: torch.Generator, include_task_token: bool = True) -> tuple[list[int], list[int]]:
        inp, out = self.generate(g)
        seq = [BOS]
        if include_task_token:
            seq.append(TASK_TOKEN_START + self.task_id)
        seq += inp + [SEP] + out + [EOS]
        # labels: -100 everywhere except the answer span (and the closing EOS)
        n_prefix = len(seq) - len(out) - 1
        labels = [-100] * n_prefix + out + [EOS]
        return seq, labels


def _rand_symbols(g: torch.Generator, n: int, hi: int = NUM_SYMBOLS,
                  lo: int = 0) -> list[int]:
    return [sym(int(v)) for v in torch.randint(lo, hi, (n,), generator=g)]


# --- class_il decidability -------------------------------------------------
# Without a task token the model sees only <input> [SEP] and must produce
# <output>. Tasks accepting the same input shape then contradict each other:
# `copy`, `reverse`, `sort`, `sortdesc` and `induction8` all take eight random
# symbols and demand five different answers. Measured, that capped joint
# training at 0.45 against an oracle bound of 0.33 — the benchmark was scoring
# how well the model guessed which task it had been handed.
#
# Giving colliding tasks disjoint symbol ranges makes the task inferable from
# the input, which is both decidable and realistic: different topics genuinely
# use different vocabulary. Ranges only need to be disjoint *within* a group
# that shares an input shape, so they are reused across groups.
SYMBOL_RANGES: dict[str, tuple[int, int]] = {
    # length 2 — the four moduli must not overlap
    "modadd7": (0, 7), "modadd13": (7, 20),
    "modadd23": (20, 43), "modadd31": (43, 74),
    # length 8
    "copy": (0, 20), "reverse": (20, 40), "sort": (40, 60),
    "sortdesc": (60, 80), "induction8": (80, 108),
    # length 12
    "copy12": (0, 54), "reverse12": (54, 108),
    # length 6 — alone in its shape, so it may use everything
    "induction6": (0, 108),
}


# --- task constructors -----------------------------------------------------
def make_copy(task_id: int, length: int = 8, name: str = "copy") -> Task:
    lo, hi = SYMBOL_RANGES.get(name, (0, NUM_SYMBOLS))
    def gen(g):
        s = _rand_symbols(g, length, hi, lo)
        return s, list(s)
    return Task(name, task_id, gen, 2 * length + 5, chance=(hi - lo) ** -length)


def make_reverse(task_id: int, length: int = 8, name: str = "reverse") -> Task:
    lo, hi = SYMBOL_RANGES.get(name, (0, NUM_SYMBOLS))
    def gen(g):
        s = _rand_symbols(g, length, hi, lo)
        return s, list(reversed(s))
    return Task(name, task_id, gen, 2 * length + 5, chance=(hi - lo) ** -length)


def make_sort(task_id: int, length: int = 8, descending: bool = False,
              name: str | None = None) -> Task:
    name = name or ("sortdesc" if descending else "sort")
    lo, hi = SYMBOL_RANGES.get(name, (0, NUM_SYMBOLS))
    def gen(g):
        s = _rand_symbols(g, length, hi, lo)
        return s, sorted(s, reverse=descending)
    return Task(name, task_id, gen, 2 * length + 5, chance=(hi - lo) ** -length)


def make_modadd(task_id: int, modulus: int = 23) -> Task:
    # Every modadd takes two symbols, so in class_il they are indistinguishable
    # unless their operand ranges are disjoint. The offset is what separates
    # "add mod 7" from "add mod 31" when there is no task token to say which.
    name = f"modadd{modulus}"
    lo, _ = SYMBOL_RANGES.get(name, (0, NUM_SYMBOLS))
    def gen(g):
        a = int(torch.randint(0, modulus, (1,), generator=g))
        b = int(torch.randint(0, modulus, (1,), generator=g))
        return [sym(lo + a), sym(lo + b)], [sym(lo + (a + b) % modulus)]
    return Task(name, task_id, gen, 8, chance=1.0 / modulus)


def make_kvrecall(task_id: int, n_pairs: int = 4, name: str = "kvrecall") -> Task:
    def gen(g):
        keys = torch.randperm(NUM_SYMBOLS // 2, generator=g)[:n_pairs]
        vals = torch.randint(NUM_SYMBOLS // 2, NUM_SYMBOLS, (n_pairs,), generator=g)
        seq: list[int] = []
        for k, v in zip(keys, vals):
            seq += [sym(int(k)), sym(int(v))]
        q = int(torch.randint(0, n_pairs, (1,), generator=g))
        seq.append(sym(int(keys[q])))
        return seq, [sym(int(vals[q]))]
    # guessing among the n values present in the prompt
    return Task(name, task_id, gen, 2 * n_pairs + 8, chance=1.0 / n_pairs)


def make_induction(task_id: int, length: int = 10, name: str = "induction") -> Task:
    """Pattern completion: the final token repeats an earlier one; answer with
    whatever followed that earlier occurrence.

    The cue must be *unique*. If the trigger symbol appears more than once
    before the end, each occurrence implies a different continuation and the
    answer is undecidable from the input — measured at 11.4% of sequences in the
    naive version, which caps accuracy no matter how good the model is.
    """
    lo, hi = SYMBOL_RANGES.get(name, (0, NUM_SYMBOLS))
    def gen(g):
        s = _rand_symbols(g, length, hi, lo)
        i = int(torch.randint(0, length - 3, (1,), generator=g))
        trigger = s[i]
        # strip every other occurrence so exactly one cue remains
        for j in range(length - 1):
            while j != i and s[j] == trigger:
                s[j] = sym(int(torch.randint(lo, hi, (1,), generator=g)))
        s[-1] = trigger
        return s, [s[i + 1]]
    # guessing a symbol from the prompt
    return Task(name, task_id, gen, length + 6, chance=1.0 / (length - 1))


TASK_BUILDERS: dict[str, Callable[[int], Task]] = {
    "copy": lambda tid: make_copy(tid),
    "copy4": lambda tid: make_copy(tid, 4, "copy4"),
    "copy12": lambda tid: make_copy(tid, 12, "copy12"),
    "reverse": lambda tid: make_reverse(tid),
    "reverse4": lambda tid: make_reverse(tid, 4, "reverse4"),
    "reverse12": lambda tid: make_reverse(tid, 12, "reverse12"),
    "sort": lambda tid: make_sort(tid),
    "sort4": lambda tid: make_sort(tid, 4, name="sort4"),
    "sortdesc": lambda tid: make_sort(tid, 8, descending=True),
    "sortdesc4": lambda tid: make_sort(tid, 4, descending=True, name="sortdesc4"),
    "modadd7": lambda tid: make_modadd(tid, 7),
    "modadd13": lambda tid: make_modadd(tid, 13),
    "modadd17": lambda tid: make_modadd(tid, 17),
    "modadd23": lambda tid: make_modadd(tid, 23),
    "modadd31": lambda tid: make_modadd(tid, 31),
    "kvrecall": lambda tid: make_kvrecall(tid, 4),
    "kvrecall2": lambda tid: make_kvrecall(tid, 2, "kvrecall2"),
    "induction": lambda tid: make_induction(tid, 10, "induction"),
    "induction6": lambda tid: make_induction(tid, 6, "induction6"),
    "induction8": lambda tid: make_induction(tid, 8, "induction8"),
}

# The default sequence uses the easier retrieval variants.
#
# `kvrecall` (4 pairs) and `induction` (length 10) both require an induction-head
# circuit and were measured unlearned at 24M / 1200 steps — kvrecall at 0.20,
# below the 0.25 you get by guessing among the values present. A task nobody
# learns contributes only noise to every mechanism's score, so the ceiling has to
# be real before it is worth sweeping against.
# kvrecall is excluded at every difficulty. Measured at exactly its chance
# baseline (0.20 with 4 pairs, 0.50 with 2) — the model picks among the values
# present and never learns the key match. Key-value retrieval needs an induction
# head that does not form at this scale, and a task scoring at chance adds only
# noise to every mechanism's rho.
DEFAULT_SEQUENCE = ["copy", "reverse", "sort", "modadd23", "induction6"]

# Tasks with a verified clean ceiling at 24M. Use this if the easier retrieval
# variants still fail the phase-0 gate on your hardware.
CLEAN_SEQUENCE = ["copy", "reverse", "sort", "modadd23"]

# A longer stream, for the mechanisms the five-task benchmark cannot test.
#
# The control shows *no* plasticity loss over five tasks — it learns the fourth
# as well as the first — so kWTA, XdG, CBP, shrink-perturb and sparse-update are
# being scored on a benchmark where the problem they solve never occurs. Their
# rho of ~0 is the correct answer to a question nobody asked.
#
# Ordered to interleave families rather than group them: consecutive tasks from
# the same family transfer, which suppresses the interference the benchmark is
# supposed to measure.
LONG_SEQUENCE = [
    "copy", "modadd7", "reverse", "sort", "modadd13",
    "induction6", "copy12", "sortdesc", "modadd23", "reverse12",
    "induction8", "modadd31",
]


# ---------------------------------------------------------------------------
def build_task_sequence(names: list[str] | None = None) -> list[Task]:
    names = names or DEFAULT_SEQUENCE
    if len(names) > MAX_TASKS:
        raise ValueError(f"at most {MAX_TASKS} tasks; got {len(names)}")
    unknown = [n for n in names if n not in TASK_BUILDERS]
    if unknown:
        raise ValueError(f"unknown tasks {unknown}; known: {sorted(TASK_BUILDERS)}")
    return [TASK_BUILDERS[n](i) for i, n in enumerate(names)]


def make_batch(
    task: Task,
    batch_size: int,
    g: torch.Generator,
    max_len: int,
    include_task_token: bool = True,
) -> dict[str, torch.Tensor]:
    seqs, labs = [], []
    for _ in range(batch_size):
        s, l = task.sample(g, include_task_token)
        pad = max_len - len(s)
        if pad < 0:
            raise ValueError(f"sequence of {len(s)} exceeds max_len {max_len}")
        seqs.append(s + [PAD] * pad)
        labs.append(l + [-100] * pad)
    return {
        "input_ids": torch.tensor(seqs, dtype=torch.long),
        "labels": torch.tensor(labs, dtype=torch.long),
        "task_id": torch.full((batch_size,), task.task_id, dtype=torch.long),
    }


class TaskStream:
    """Iterates a fixed task sequence, one task at a time, with clean boundaries."""

    def __init__(
        self,
        tasks: list[Task],
        batch_size: int = 32,
        steps_per_task: int = 400,
        seed: int = 0,
        include_task_token: bool = True,
        steps_schedule: list[int] | None = None,
    ):
        self.tasks = tasks
        self.batch_size = batch_size
        self.steps_per_task = steps_per_task
        # Per-task step budgets. Uniform exposure is the convenient case, not
        # the realistic one: a personalised model sees whatever its user talks
        # about most, and replay's buffer is a *uniform sample of the stream*,
        # so a skewed stream produces a skewed buffer. Rehearsal was measured at
        # rho=1.0 on a perfectly balanced stream; this is how you find out
        # whether that survives 90% of the traffic being one topic.
        if steps_schedule is not None and len(steps_schedule) != len(tasks):
            raise ValueError(
                f"steps_schedule has {len(steps_schedule)} entries for "
                f"{len(tasks)} tasks"
            )
        self.steps_schedule = steps_schedule
        self.include_task_token = include_task_token
        self.max_len = max(t.max_len for t in tasks)
        self.seed = seed
        self._g = torch.Generator().manual_seed(seed)

    def steps_for(self, task_idx: int) -> int:
        """Training steps allotted to task `task_idx`."""
        if self.steps_schedule is None:
            return self.steps_per_task
        return self.steps_schedule[task_idx]

    def batches(self, task: Task, n: int | None = None):
        if n is None:
            n = self.steps_for(self.tasks.index(task))
        for _ in range(n):
            yield make_batch(
                task, self.batch_size, self._g, self.max_len, self.include_task_token
            )

    def eval_batches(self, task: Task, n_batches: int = 8, seed_offset: int = 10_000):
        """Deterministic evaluation batches — the frozen probe set.

        Derived from a separate generator seeded independently of training, so
        the probe set never changes across runs or configurations.
        """
        g = torch.Generator().manual_seed(self.seed + seed_offset + task.task_id)
        for _ in range(n_batches):
            yield make_batch(task, self.batch_size, g, self.max_len, self.include_task_token)

    @property
    def num_tasks(self) -> int:
        return len(self.tasks)
