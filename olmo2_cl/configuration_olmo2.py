"""Configuration for the OLMo-2 style decoder used by CLMS.

Architecture-faithful to OLMo 2 (Groeneveld et al. / AI2):
  * RMSNorm applied to the *output* of each sublayer, inside the residual
    ("reordered norm"), rather than to the sublayer input.
  * QK-Norm: RMSNorm on the query and key projections before RoPE.
  * Rotary position embeddings, SwiGLU feed-forward, no biases anywhere.

Size presets target the 20-50M range used for the mechanism-ranking sweep.
Scaling up is a preset change; nothing else in the stack depends on size.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


# name -> (hidden_size, num_layers, num_heads, num_kv_heads)
SIZE_PRESETS: dict[str, tuple[int, int, int, int]] = {
    "nano": (128, 4, 4, 4),      # ~1M   - unit tests, CPU smoke runs
    "tiny": (256, 6, 8, 4),      # ~6M   - fast CI, laptop debugging
    "small": (512, 8, 8, 4),     # ~25M  - default sweep size
    "base": (640, 10, 10, 5),    # ~50M  - upper end of the sweep ladder
    "mid": (1024, 16, 16, 8),    # ~200M - rank-confirmation rung
}


@dataclass
class Olmo2Config:
    # --- vocabulary / sequence -------------------------------------------
    vocab_size: int = 512
    max_position_embeddings: int = 512

    # --- shape ------------------------------------------------------------
    hidden_size: int = 512
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    intermediate_size: int | None = None      # derived from mlp_ratio if None
    mlp_ratio: float = 8.0 / 3.0
    intermediate_multiple_of: int = 64

    # --- normalisation / position ----------------------------------------
    rms_norm_eps: float = 1e-6
    use_qk_norm: bool = True
    rope_theta: float = 10000.0

    # --- training-time details -------------------------------------------
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True
    attention_dropout: float = 0.0

    # --- bookkeeping ------------------------------------------------------
    size_preset: str | None = None
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    # Mechanisms may stash per-layer decisions here (e.g. which layer index
    # receives a memory layer). Kept in the config so a run is reproducible
    # from its dumped config alone.
    mechanism_notes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.size_preset is not None:
            if self.size_preset not in SIZE_PRESETS:
                raise ValueError(
                    f"unknown size_preset {self.size_preset!r}; "
                    f"choose from {sorted(SIZE_PRESETS)}"
                )
            h, layers, heads, kv = SIZE_PRESETS[self.size_preset]
            self.hidden_size = h
            self.num_hidden_layers = layers
            self.num_attention_heads = heads
            self.num_key_value_heads = kv
            self.intermediate_size = None  # force re-derivation

        if self.intermediate_size is None:
            raw = int(self.hidden_size * self.mlp_ratio)
            m = self.intermediate_multiple_of
            self.intermediate_size = max(m, ((raw + m - 1) // m) * m)

        self._validate()

    # ------------------------------------------------------------------
    def _validate(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be "
                f"divisible by num_key_value_heads ({self.num_key_value_heads})"
            )
        for name in ("vocab_size", "hidden_size", "num_hidden_layers"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    # ------------------------------------------------------------------
    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    def estimated_params(self) -> int:
        """Parameter count, embeddings included. Cheap sanity check."""
        h, i = self.hidden_size, self.intermediate_size
        kv_dim = self.num_key_value_heads * self.head_dim
        per_layer = (
            h * h                       # q_proj
            + h * kv_dim * 2            # k_proj, v_proj
            + h * h                     # o_proj
            + 3 * h * i                 # gate, up, down
            + 4 * h                     # norms (attn, mlp, q, k) approx
        )
        embed = self.vocab_size * h
        head = 0 if self.tie_word_embeddings else self.vocab_size * h
        return per_layer * self.num_hidden_layers + embed + head + h

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Olmo2Config":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})
