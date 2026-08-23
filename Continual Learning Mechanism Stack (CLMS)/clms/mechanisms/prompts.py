"""F05 · Learning to Prompt.

A pool of M prompts, each with a learnable key. Per input, the top-N prompts are
selected by cosine similarity between a query feature and the keys, prepended to
the embedded sequence, and a pull loss draws the selected keys toward the query.
The backbone is frozen, so it cannot forget by construction; all the action is
in a pool that is a fraction of a percent of the parameters.

    x_p = [p_1 ; ... ; p_N ; x_e]
    L   = CE(f(x_p), y) + lam * sum_{i in selected} (1 - cos(q(x), k_i))

Implementation note: prompts are prepended at the embedding hook and stripped at
the final hook, so sequence length is restored before the LM head and label
alignment is unaffected. RoPE positions shift by the prompt length, a small
deviation from the vision-transformer setting the paper describes.

Caveat worth remembering: L2P's strong numbers rest on a *strong frozen
backbone*. From-scratch at 20-50M there is no such backbone, so this is expected
to underperform here — which is itself the result that motivates the size ladder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import CostReport, Mechanism, SignatureCheck
from ..registry import register


@register
class LearningToPrompt(Mechanism):
    name = "l2p"
    surfaces = ("A", "L", "O")
    family = "F05"
    paper = "2112.08654"
    conflicts = ("continual_backprop",)   # frozen backbone; nothing to recycle
    order = 16

    defaults = {
        "pool_size": 20,        # M
        "top_n": 4,             # N selected per input
        "prompt_len": 4,        # Lp tokens per prompt
        "lam": 0.5,             # key-query pull weight
        "freeze_after_task": 0,
        "diversify": True,      # penalise over-used prompts
    }

    def __init__(self, **kw):
        super().__init__(**kw)
        self.pool: nn.Parameter | None = None
        self.keys: nn.Parameter | None = None
        self.hidden = None
        self._prompt_tokens = 0
        self._pull = 0.0
        self._selected_counts: torch.Tensor | None = None
        self._frozen = False

    def on_model_config(self, cfg, ctx) -> None:
        self.hidden = cfg.hidden_size
        self._prompt_tokens = self.params["top_n"] * self.params["prompt_len"]

    def setup(self, model, cfg, ctx) -> None:
        device = next(model.parameters()).device
        M, Lp = self.params["pool_size"], self.params["prompt_len"]
        self.pool = nn.Parameter(torch.randn(M, Lp, self.hidden, device=device) * 0.02)
        self.keys = nn.Parameter(torch.randn(M, self.hidden, device=device) * 0.02)
        # registered on the model so the optimizer and checkpoints pick them up
        model.register_parameter("l2p_pool", self.pool)
        model.register_parameter("l2p_keys", self.keys)
        self._selected_counts = torch.zeros(M)

    # ------------------------------------------------------------------
    def observe(self, name, layer_idx, tensor):
        if self.pool is None:
            return None

        if name == "embed":
            b = tensor.shape[0]
            # query feature: mean-pooled embedding, the cheap stand-in for the
            # paper's frozen-backbone query function
            q = F.normalize(tensor.mean(dim=1), dim=-1)          # (b, h)
            k = F.normalize(self.keys, dim=-1)                   # (M, h)
            sim = q @ k.T                                        # (b, M)

            if self.params["diversify"] and self._selected_counts is not None:
                freq = (self._selected_counts / self._selected_counts.sum().clamp_min(1)).to(sim.device)
                sim = sim - freq.unsqueeze(0)

            top = sim.topk(self.params["top_n"], dim=-1).indices  # (b, N)
            with torch.no_grad():
                self._selected_counts.index_add_(
                    0, top.reshape(-1).cpu(),
                    torch.ones(top.numel(), dtype=torch.float),
                )
            # pull loss: draw selected keys toward the query
            chosen_k = k[top]                                    # (b, N, h)
            self._pull_tensor = (1.0 - (chosen_k * q.unsqueeze(1)).sum(-1)).mean()

            prompts = self.pool[top]                             # (b, N, Lp, h)
            prompts = prompts.reshape(b, -1, self.hidden)        # (b, N*Lp, h)
            self.mark_ran()
            return torch.cat([prompts.to(tensor.dtype), tensor], dim=1)

        if name == "final" and self._prompt_tokens > 0:
            # strip the prompt positions so length matches input_ids again
            return tensor[:, self._prompt_tokens:]

        return None

    def compute_loss(self, model, batch, out, base_loss, ctx):
        pull = getattr(self, "_pull_tensor", None)
        if pull is None:
            return None
        self._pull = float(pull.detach())
        return self.params["lam"] * pull

    def on_task_start(self, model, task_id, ctx) -> None:
        if task_id > self.params["freeze_after_task"] and not self._frozen:
            for name, p in model.named_parameters():
                if not name.startswith("l2p_"):
                    p.requires_grad_(False)
            self._frozen = True

    # ------------------------------------------------------------------
    def self_test(self, model, batch, ctx) -> tuple[bool, str]:
        if self.pool is None:
            return False, "setup() never ran; prompt pool is None"
        b, t = batch["input_ids"].shape
        emb = torch.randn(b, t, self.hidden, device=ctx.device)
        extended = self.observe("embed", -1, emb)
        if extended is None:
            return False, "observe() did not prepend prompts at the embed hook"
        if extended.shape[1] != t + self._prompt_tokens:
            return False, (f"expected length {t + self._prompt_tokens}, "
                           f"got {extended.shape[1]}")
        stripped = self.observe("final", -1, extended)
        if stripped is None or stripped.shape[1] != t:
            return False, "final hook did not restore the original sequence length"
        n = self.pool.numel() + self.keys.numel()
        total = sum(p.numel() for p in model.parameters())
        return True, (f"pool {self.params['pool_size']}x{self.params['prompt_len']}, "
                      f"+{self._prompt_tokens} tokens, {n / total:.3%} of params")

    def signature(self, model, ctx) -> SignatureCheck | None:
        if self._selected_counts is None or self._selected_counts.sum() == 0:
            return None
        p = self._selected_counts / self._selected_counts.sum()
        p = p[p > 0]
        entropy = float(-(p * p.log()).sum())
        max_entropy = float(torch.log(torch.tensor(float(self.params["pool_size"]))))
        return SignatureCheck(
            probe="A4",
            quantity="prompt-selection entropy / max",
            value=entropy / max_entropy if max_entropy > 0 else 0.0,
            baseline=0.3,
            direction="increase",
            detail="a collapsed pool selects the same prompts for everything and "
                   "provides no task separation",
        )

    def cost_report(self) -> CostReport:
        n = (self.pool.numel() + self.keys.numel()) if self.pool is not None else 0
        return CostReport(added_params=n, notes={"prompt_tokens": self._prompt_tokens,
                                                 "last_pull": self._pull})

    def state_dict(self):
        return {
            "pool": self.pool.detach().cpu() if self.pool is not None else None,
            "keys": self.keys.detach().cpu() if self.keys is not None else None,
            "counts": self._selected_counts,
        }

    def load_state_dict(self, state):
        if self.pool is not None and state.get("pool") is not None:
            with torch.no_grad():
                self.pool.copy_(state["pool"].to(self.pool.device))
                self.keys.copy_(state["keys"].to(self.keys.device))
        self._selected_counts = state.get("counts", self._selected_counts)
