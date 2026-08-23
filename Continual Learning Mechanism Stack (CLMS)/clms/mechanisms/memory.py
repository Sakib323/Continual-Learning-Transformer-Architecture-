"""F04 · Memory layers and sparse update localization.

`memory_layer`  replaces one mid-stack FFN with a product-key sparse memory:
                a large pool of slots of which each token touches only top-k.

`sparse_update`  the actual continual-learning mechanism. Ranks slots (or MLP
                 neurons, when no memory layer is present) by TF-IDF specificity
                 against a frozen background corpus, and trains only the top-t.

The claim being tested is that interference is a consequence of *sharing*
parameters, so making updates sparse and localized should make forgetting
largely evaporate at equal acquisition. Reported: NaturalQuestions F1 dropped
11% versus 89% for full finetuning and 71% for LoRA.

Signature (probe B2): update concentration — the fraction of parameters
accounting for 90% of total displacement — should fall by orders of magnitude.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import CostReport, Mechanism, SignatureCheck, to_cpu_tree
from ..registry import register


# ---------------------------------------------------------------------------
class ProductKeyMemory(nn.Module):
    """Sparse key-value memory with product-key lookup.

    A flat top-k over N slots costs O(N); factoring the keys into two codebooks
    of sqrt(N) turns it into two O(sqrt(N)) searches, which is what makes a
    million-slot pool affordable.
    """

    def __init__(self, hidden_size: int, num_slots: int, topk: int,
                 key_dim: int, value_dim: int, num_heads: int = 1):
        super().__init__()
        self.n_keys = int(math.isqrt(num_slots))
        self.num_slots = self.n_keys ** 2
        self.topk = topk
        self.num_heads = num_heads
        self.key_dim = key_dim
        half = key_dim // 2

        self.q_proj = nn.Linear(hidden_size, key_dim * num_heads, bias=False)
        self.keys = nn.Parameter(torch.randn(2, num_heads, self.n_keys, half) * 0.02)
        self.values = nn.Embedding(self.num_slots, value_dim)
        nn.init.normal_(self.values.weight, std=0.02)

        # input-dependent gate, as in the reference implementation
        self.gate_proj = nn.Linear(hidden_size, value_dim, bias=False)
        self.out_proj = nn.Linear(value_dim, hidden_size, bias=False)

        # written by sparse_update; None means "train everything"
        self.value_grad_mask: torch.Tensor | None = None
        self.last_indices: torch.Tensor | None = None
        self.access_counts = torch.zeros(self.num_slots)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        h, k, nk = self.num_heads, self.topk, self.n_keys

        q = self.q_proj(x).view(b * t, h, self.key_dim)
        q1, q2 = q.chunk(2, dim=-1)                       # (bt, h, half)

        s1 = torch.einsum("bhd,hnd->bhn", q1, self.keys[0])
        s2 = torch.einsum("bhd,hnd->bhn", q2, self.keys[1])
        v1, i1 = s1.topk(k, dim=-1)                       # (bt, h, k)
        v2, i2 = s2.topk(k, dim=-1)

        # combine the two half-searches into a top-k over the full pool
        scores = (v1.unsqueeze(-1) + v2.unsqueeze(-2)).view(b * t, h, k * k)
        idx = (i1.unsqueeze(-1) * nk + i2.unsqueeze(-2)).view(b * t, h, k * k)
        best, pos = scores.topk(k, dim=-1)
        slots = idx.gather(-1, pos)                       # (bt, h, k)

        weights = F.softmax(best, dim=-1)
        vals = self.values(slots)                         # (bt, h, k, value_dim)
        out = (vals * weights.unsqueeze(-1)).sum(dim=(1, 2))

        self.last_indices = slots.detach()
        with torch.no_grad():
            flat = slots.reshape(-1).cpu()
            self.access_counts.index_add_(0, flat, torch.ones_like(flat, dtype=torch.float))

        out = out.view(b, t, -1)
        return self.out_proj(out * F.silu(self.gate_proj(x)))


# ---------------------------------------------------------------------------
@register
class MemoryLayer(Mechanism):
    name = "memory_layer"
    surfaces = ("A",)
    family = "F04"
    paper = "2510.15103"
    order = 15

    defaults = {
        "auto_scale": True,     # size the pool to the FFN it replaces
        "num_slots": 16384,     # used only when auto_scale is False
        "topk": 16,
        "key_dim": 128,
        "value_dim": 256,
        "num_heads": 2,
        "layer": -1,            # -1 = mid-stack, else an explicit index
    }

    def __init__(self, **kw):
        super().__init__(**kw)
        self.module: ProductKeyMemory | None = None
        self._target_layer: int | None = None
        self._num_layers: int | None = None

    def on_model_config(self, cfg, ctx) -> None:
        self._num_layers = cfg.num_hidden_layers

        if self.params["auto_scale"]:
            # Parameter-match the memory to the FFN it displaces. Without this a
            # fixed 16k-slot pool adds 4.3M parameters to a 0.87M model — five
            # times the network — which makes any accuracy comparison
            # meaningless regardless of how the mechanism performs.
            h = cfg.hidden_size
            self.params["value_dim"] = h
            self.params["key_dim"] = max(32, (h // 2) & ~1)   # even, for the split
            ffn_params = 3 * h * cfg.intermediate_size
            target = max(64, ffn_params // max(h, 1))
            n_keys = max(8, int(math.isqrt(target)))
            self.params["num_slots"] = n_keys ** 2
            self.params["topk"] = min(self.params["topk"], max(4, n_keys // 4))
        want = self.params["layer"]
        # -1 means mid-stack, mirroring the reference layer-12-of-22 placement
        self._target_layer = self._num_layers // 2 if want < 0 else want
        if not 0 <= self._target_layer < self._num_layers:
            raise ValueError(
                f"memory_layer target {self._target_layer} outside "
                f"[0, {self._num_layers})"
            )

    def build_mlp(self, layer_idx: int, default: nn.Module):
        if self._target_layer is None:
            raise RuntimeError(
                "memory_layer: on_model_config() never ran; the trainer must "
                "call composer.set_model_config() before build_model()"
            )
        return self._make(default) if layer_idx == self._target_layer else None

    def _make(self, default: nn.Module) -> ProductKeyMemory:
        hidden = default.down_proj.out_features
        self.module = ProductKeyMemory(
            hidden_size=hidden,
            num_slots=self.params["num_slots"],
            topk=self.params["topk"],
            key_dim=self.params["key_dim"],
            value_dim=self.params["value_dim"],
            num_heads=self.params["num_heads"],
        )
        return self.module

    # ------------------------------------------------------------------
    def self_test(self, model, batch, ctx) -> tuple[bool, str]:
        if self.module is None:
            return False, "no MLP was replaced; check the `layer` parameter"
        x = torch.randn(2, 4, self.module.q_proj.in_features, device=ctx.device)
        self.module.to(ctx.device)
        out = self.module(x)
        if out.shape != x.shape:
            return False, f"shape mismatch: in {tuple(x.shape)}, out {tuple(out.shape)}"
        if self.module.last_indices is None:
            return False, "no slots were selected"
        touched = int(self.module.last_indices.unique().numel())
        frac = touched / self.module.num_slots
        return True, (f"{self.module.num_slots} slots, {touched} touched by a "
                      f"2x4 batch ({frac:.2%})")

    def signature(self, model, ctx) -> SignatureCheck | None:
        if self.module is None:
            return None
        used = int((self.module.access_counts > 0).sum())
        return SignatureCheck(
            probe="A4",
            quantity="fraction of memory slots ever accessed",
            value=used / self.module.num_slots,
            baseline=1.0,
            direction="decrease",
            detail="a dense layer would be 1.0; sparsity here is the whole point",
        )

    def cost_report(self) -> CostReport:
        if self.module is None:
            return CostReport()
        n = sum(p.numel() for p in self.module.parameters())
        return CostReport(added_params=n, notes={"slots": self.module.num_slots})


# ---------------------------------------------------------------------------
@register
class SparseUpdate(Mechanism):
    """Train only the parameters most specific to the current batch.

    With a memory layer present this masks value-slot gradients. Without one it
    falls back to masking MLP neuron gradients by the same criterion, which
    needs no architecture change and tests whether the sparsity principle
    transfers before paying for the real thing.
    """

    name = "sparse_update"
    surfaces = ("O",)
    family = "F04"
    paper = "2510.15103"
    order = 45

    defaults = {
        "auto_scale": True,     # derive t from the pool actually built
        "slot_frac": 0.03,      # trained fraction of slots per step when auto
        "t": 500,               # memory slots per step when auto_scale is False
        "neuron_frac": 0.1,     # fallback path: fraction of MLP neurons per step.
                                # `t` is a slot count and meaningless against a
                                # few hundred neurons, so the two paths are
                                # parameterised separately.
        "ranking": "tfidf",     # tfidf | frequency
        "background_batches": 32,
        "warmup_steps": 50,     # collect background before masking begins
    }

    def __init__(self, **kw):
        super().__init__(**kw)
        self.memory: ProductKeyMemory | None = None
        self.background_df: torch.Tensor | None = None
        self._bg_batches = 0
        self._mlps: list[nn.Module] = []
        self._masked_steps = 0

    def setup(self, model, cfg, ctx) -> None:
        for module in model.modules():
            if isinstance(module, ProductKeyMemory):
                self.memory = module
                self.background_df = torch.zeros(module.num_slots)
                if self.params["auto_scale"]:
                    # t is only meaningful relative to the pool that exists
                    self.params["t"] = max(
                        1, int(module.num_slots * self.params["slot_frac"])
                    )
                break
        if self.memory is None:
            for layer in model.model.layers:
                if hasattr(layer.mlp, "down_proj") and hasattr(layer.mlp, "gate_proj"):
                    self._mlps.append(layer.mlp)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def before_step(self, model, ctx) -> None:
        if self.memory is not None:
            self._mask_memory(ctx)
        elif self._mlps:
            self._mask_neurons(ctx)

    def _mask_memory(self, ctx) -> None:
        mem = self.memory
        assert mem is not None and self.background_df is not None
        if mem.last_indices is None or mem.values.weight.grad is None:
            return

        counts = torch.zeros(mem.num_slots)
        flat = mem.last_indices.reshape(-1).cpu()
        counts.index_add_(0, flat, torch.ones_like(flat, dtype=torch.float))

        # collect the background access distribution before masking anything
        if self._bg_batches < self.params["background_batches"]:
            self.background_df += (counts > 0).float()
            self._bg_batches += 1
            return

        if self.params["ranking"] == "tfidf":
            tf = counts / counts.sum().clamp_min(1.0)
            idf = torch.log(
                (self._bg_batches + 1.0) / (self.background_df + 1.0)
            )
            score = tf * idf
        else:
            score = counts

        t = min(self.params["t"], int((score > 0).sum()))
        if t == 0:
            return
        keep = torch.topk(score, k=t).indices
        mask = torch.zeros(mem.num_slots, 1, device=mem.values.weight.device)
        mask[keep.to(mask.device)] = 1.0
        mem.values.weight.grad.mul_(mask)
        self._masked_steps += 1
        self.mark_ran()

    def _mask_neurons(self, ctx) -> None:
        """Fallback: same specificity logic applied to existing FFN neurons.

        Granularity is thousands of neurons rather than a million slots, so this
        is a weaker instrument — but it needs no upcycling phase.
        """
        for mlp in self._mlps:
            w = mlp.down_proj
            if w.weight.grad is None:
                continue
            # neuron importance for this step = gradient magnitude on its column
            score = w.weight.grad.abs().sum(dim=0)
            t = max(1, int(score.numel() * self.params["neuron_frac"]))
            if t >= score.numel():
                continue
            keep = torch.topk(score, k=t).indices
            mask = torch.zeros_like(score)
            mask[keep] = 1.0
            w.weight.grad.mul_(mask.unsqueeze(0))
            if mlp.gate_proj.weight.grad is not None:
                mlp.gate_proj.weight.grad.mul_(mask.unsqueeze(1))
            if mlp.up_proj.weight.grad is not None:
                mlp.up_proj.weight.grad.mul_(mask.unsqueeze(1))
        self._masked_steps += 1
        self.mark_ran()

    # ------------------------------------------------------------------
    def self_test(self, model, batch, ctx) -> tuple[bool, str]:
        if self.memory is None and not self._mlps:
            return False, "found neither a memory layer nor any MLP to mask"
        out = model(batch["input_ids"], labels=batch["labels"])
        out["loss"].backward()

        if self.memory is not None:
            saved = self._bg_batches
            self._bg_batches = self.params["background_batches"]
            g = self.memory.values.weight.grad
            if g is None:
                self._bg_batches = saved
                model.zero_grad(set_to_none=True)
                return False, "memory values received no gradient"
            nz_before = int((g.abs().sum(dim=1) > 0).sum())
            self.before_step(model, ctx)
            nz_after = int((g.abs().sum(dim=1) > 0).sum())
            self._bg_batches = saved
            model.zero_grad(set_to_none=True)
            if nz_after >= nz_before:
                return False, f"masking did not reduce active slots ({nz_before} -> {nz_after})"
            return True, f"slot gradients {nz_before} -> {nz_after} (t={self.params['t']})"

        mlp = self._mlps[0]
        width = mlp.down_proj.weight.shape[1]
        t = max(1, int(width * self.params["neuron_frac"]))
        if t >= width:
            return False, (
                f"neuron_frac={self.params['neuron_frac']} keeps all {width} "
                f"neurons; nothing would be masked"
            )
        nz_before = int((mlp.down_proj.weight.grad.abs().sum(dim=0) > 0).sum())
        self.before_step(model, ctx)
        nz_after = int((mlp.down_proj.weight.grad.abs().sum(dim=0) > 0).sum())
        model.zero_grad(set_to_none=True)
        if nz_after >= nz_before:
            return False, f"neuron masking had no effect ({nz_before} -> {nz_after})"
        return True, (f"neuron gradients {nz_before} -> {nz_after} of {width} "
                      f"across {len(self._mlps)} MLPs")

    def signature(self, model, ctx) -> SignatureCheck | None:
        before = ctx.scratch.get("params_at_task_start")
        if not before:
            return None
        from ..eval.probes import update_concentration
        return SignatureCheck(
            probe="B2",
            quantity="fraction of params holding 90% of displacement",
            value=update_concentration(model, before),
            baseline=0.5,
            direction="decrease",
            detail="full finetuning sits near 0.5; localization should be orders "
                   "of magnitude below it",
        )

    def cost_report(self) -> CostReport:
        return CostReport(notes={"masked_steps": self._masked_steps,
                                 "background_batches": self._bg_batches})

    def state_dict(self):
        return {"background_df": self.background_df, "bg_batches": self._bg_batches}

    def load_state_dict(self, state):
        state = to_cpu_tree(state)
        self.background_df = state.get("background_df")
        self._bg_batches = state.get("bg_batches", 0)
