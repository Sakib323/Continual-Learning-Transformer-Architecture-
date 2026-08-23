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
NUM_SYMBOLS = 64                            # data symbols 20..83
VOCAB_SIZE = DATA_START + NUM_SYMBOLS       # 84 -> round up in config to 128


def sym(i: int) -> int:
    return DATA_START + (i % NUM_SYMBOLS)


# ---------------------------------------------------------------------------
@dataclass
class Task:
    name: str
    task_id: int
    generate: Callable[[torch.Generator], tuple[list[int], list[int]]]
    max_len: int

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


def _rand_symbols(g: torch.Generator, n: int, hi: int = NUM_SYMBOLS) -> list[int]:
    return [sym(int(v)) for v in torch.randint(0, hi, (n,), generator=g)]


# --- task constructors -----------------------------------------------------
def make_copy(task_id: int, length: int = 8) -> Task:
    def gen(g):
        s = _rand_symbols(g, length)
        return s, list(s)
    return Task("copy", task_id, gen, 2 * length + 5)


def make_reverse(task_id: int, length: int = 8) -> Task:
    def gen(g):
        s = _rand_symbols(g, length)
        return s, list(reversed(s))
    return Task("reverse", task_id, gen, 2 * length + 5)


def make_sort(task_id: int, length: int = 8) -> Task:
    def gen(g):
        s = _rand_symbols(g, length)
        return s, sorted(s)
    return Task("sort", task_id, gen, 2 * length + 5)


def make_modadd(task_id: int, modulus: int = 23) -> Task:
    def gen(g):
        a = int(torch.randint(0, modulus, (1,), generator=g))
        b = int(torch.randint(0, modulus, (1,), generator=g))
        return [sym(a), sym(b)], [sym((a + b) % modulus)]
    return Task(f"modadd{modulus}", task_id, gen, 8)


def make_kvrecall(task_id: int, n_pairs: int = 4) -> Task:
    def gen(g):
        keys = torch.randperm(NUM_SYMBOLS // 2, generator=g)[:n_pairs]
        vals = torch.randint(NUM_SYMBOLS // 2, NUM_SYMBOLS, (n_pairs,), generator=g)
        seq: list[int] = []
        for k, v in zip(keys, vals):
            seq += [sym(int(k)), sym(int(v))]
        q = int(torch.randint(0, n_pairs, (1,), generator=g))
        seq.append(sym(int(keys[q])))
        return seq, [sym(int(vals[q]))]
    return Task("kvrecall", task_id, gen, 2 * n_pairs + 8)


def make_induction(task_id: int, length: int = 10) -> Task:
    def gen(g):
        s = _rand_symbols(g, length)
        # plant a bigram repeat: s[i], s[i+1] ... s[-1] == s[i]
        i = int(torch.randint(0, length - 3, (1,), generator=g))
        s[-1] = s[i]
        return s, [s[i + 1]]
    return Task("induction", task_id, gen, length + 6)


TASK_BUILDERS: dict[str, Callable[[int], Task]] = {
    "copy": lambda tid: make_copy(tid),
    "reverse": lambda tid: make_reverse(tid),
    "sort": lambda tid: make_sort(tid),
    "modadd23": lambda tid: make_modadd(tid, 23),
    "modadd31": lambda tid: make_modadd(tid, 31),
    "kvrecall": lambda tid: make_kvrecall(tid),
    "induction": lambda tid: make_induction(tid),
}

DEFAULT_SEQUENCE = ["copy", "reverse", "sort", "modadd23", "kvrecall", "induction"]


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
    ):
        self.tasks = tasks
        self.batch_size = batch_size
        self.steps_per_task = steps_per_task
        self.include_task_token = include_task_token
        self.max_len = max(t.max_len for t in tasks)
        self.seed = seed
        self._g = torch.Generator().manual_seed(seed)

    def batches(self, task: Task, n: int | None = None):
        for _ in range(n if n is not None else self.steps_per_task):
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
