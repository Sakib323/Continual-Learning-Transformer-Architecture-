"""OLMo-2 style decoder-only transformer, written for mechanism injection.

Differences from a stock Llama-style block, all of them OLMo 2's:
  * norm on the sublayer *output*, inside the residual (reordered norm)
  * QK-Norm before RoPE
  * no biases

Injection surface
-----------------
Every module that a continual-learning mechanism might want to replace or
observe goes through the `Injector` passed at construction:

    model = Olmo2ForCausalLM(cfg, injector=composer)

`injector.build_mlp(...)` and `build_attention(...)` let an architecture-surface
mechanism swap a submodule (memory layer, MoE, k-WTA) without editing this file.
`injector.observe(...)` gives loss/optimizer-surface mechanisms and the probe
suite a stable place to read activations from.

The default `NullInjector` makes this a plain OLMo 2, which is the
all-flags-false control.
"""

from __future__ import annotations

import math
from typing import Any, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configuration_olmo2 import Olmo2Config


# ---------------------------------------------------------------------------
# injection protocol
# ---------------------------------------------------------------------------
class Injector(Protocol):
    def build_attention(self, layer_idx: int, default: nn.Module) -> nn.Module: ...
    def build_mlp(self, layer_idx: int, default: nn.Module) -> nn.Module: ...
    def observe(self, name: str, layer_idx: int, tensor: torch.Tensor) -> torch.Tensor: ...


class NullInjector:
    """No-op injector. The model is a plain OLMo 2 under this."""

    def build_attention(self, layer_idx: int, default: nn.Module) -> nn.Module:
        return default

    def build_mlp(self, layer_idx: int, default: nn.Module) -> nn.Module:
        return default

    def observe(self, name: str, layer_idx: int, tensor: torch.Tensor) -> torch.Tensor:
        return tensor


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------
class Olmo2RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight.float() * x).to(dtype)


class Olmo2RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_positions: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_positions)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self._cached_len = seq_len

    def forward(self, seq_len: int, device, dtype):
        if seq_len > self._cached_len:
            self._build_cache(seq_len)
        cos = self.cos_cached[:seq_len].to(device=device, dtype=dtype)
        sin = self.sin_cached[:seq_len].to(device=device, dtype=dtype)
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: (b, heads, t, head_dim); cos/sin: (t, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, kv_heads, t, d = x.shape
    return x[:, :, None].expand(b, kv_heads, n_rep, t, d).reshape(b, kv_heads * n_rep, t, d)


# ---------------------------------------------------------------------------
# sublayers
# ---------------------------------------------------------------------------
class Olmo2Attention(nn.Module):
    def __init__(self, cfg: Olmo2Config, layer_idx: int, injector: Injector):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.injector = injector
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.num_key_value_groups

        kv_dim = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, kv_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, kv_dim, bias=False)
        self.o_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)

        # OLMo 2: QK-Norm over the full projection, before head split + RoPE.
        if cfg.use_qk_norm:
            self.q_norm = Olmo2RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
            self.k_norm = Olmo2RMSNorm(kv_dim, cfg.rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        self.rotary = Olmo2RotaryEmbedding(
            self.head_dim, cfg.max_position_embeddings, cfg.rope_theta
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, t, _ = x.shape

        q = self.q_norm(self.q_proj(x))
        k = self.k_norm(self.k_proj(x))
        v = self.v_proj(x)

        q = q.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary(t, x.device, q.dtype)
        q, k = apply_rope(q, k, cos, sin)

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.cfg.attention_dropout if self.training else 0.0,
            is_causal=attn_mask is None,
        )
        out = out.transpose(1, 2).contiguous().view(b, t, -1)
        out = self.injector.observe("attn_out", self.layer_idx, out)
        return self.o_proj(out)


class Olmo2MLP(nn.Module):
    """SwiGLU feed-forward. The default occupant of the surface-A swap point."""

    def __init__(self, cfg: Olmo2Config, layer_idx: int, injector: Injector):
        super().__init__()
        self.layer_idx = layer_idx
        self.injector = injector
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.gate_proj(x)) * self.up_proj(x)
        # The sparsity family (k-WTA, XdG) and the probe suite both hook here:
        # this is the network's widest representation.
        h = self.injector.observe("mlp_hidden", self.layer_idx, h)
        return self.down_proj(h)


class Olmo2DecoderLayer(nn.Module):
    def __init__(self, cfg: Olmo2Config, layer_idx: int, injector: Injector):
        super().__init__()
        self.layer_idx = layer_idx
        self.injector = injector

        self.self_attn = injector.build_attention(
            layer_idx, Olmo2Attention(cfg, layer_idx, injector)
        )
        self.mlp = injector.build_mlp(layer_idx, Olmo2MLP(cfg, layer_idx, injector))

        # Reordered norm: applied to sublayer output, not input.
        self.post_attention_layernorm = Olmo2RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_feedforward_layernorm = Olmo2RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.post_attention_layernorm(self.self_attn(x, attn_mask))
        x = self.injector.observe("resid_mid", self.layer_idx, x)
        x = x + self.post_feedforward_layernorm(self.mlp(x))
        x = self.injector.observe("resid_out", self.layer_idx, x)
        return x


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
class Olmo2Model(nn.Module):
    def __init__(self, cfg: Olmo2Config, injector: Injector | None = None):
        super().__init__()
        self.cfg = cfg
        self.injector = injector or NullInjector()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            Olmo2DecoderLayer(cfg, i, self.injector) for i in range(cfg.num_hidden_layers)
        )
        self.norm = Olmo2RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor | None = None):
        x = self.embed_tokens(input_ids)
        x = self.injector.observe("embed", -1, x)
        for layer in self.layers:
            x = layer(x, attn_mask)
        x = self.norm(x)
        return self.injector.observe("final", -1, x)


class Olmo2ForCausalLM(nn.Module):
    def __init__(self, cfg: Olmo2Config, injector: Injector | None = None):
        super().__init__()
        self.cfg = cfg
        self.model = Olmo2Model(cfg, injector)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        std = self.cfg.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        hidden = self.model(input_ids, attn_mask)
        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            # labels use -100 for positions excluded from the objective, so a
            # task can score only its answer span.
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return {"loss": loss, "logits": logits, "hidden": hidden}

    @torch.no_grad()
    def num_parameters(self, trainable_only: bool = False) -> int:
        return sum(
            p.numel() for p in self.parameters()
            if p.requires_grad or not trainable_only
        )


def build_model(cfg: Olmo2Config, injector: Injector | None = None) -> Olmo2ForCausalLM:
    """Single construction entry point. Mechanisms reach the model only here."""
    return Olmo2ForCausalLM(cfg, injector)
