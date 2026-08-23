"""OLMo-2 style decoder, vendored and adapted for mechanism injection.

Architecture-faithful reimplementation rather than a byte-copy of AI2's file:
self-contained, no transformers-internals dependency, injection points designed
in. See VENDORED_FROM.md for provenance and what to check when rebasing.
"""

from .configuration_olmo2 import Olmo2Config, SIZE_PRESETS
from .modeling_olmo2 import (
    Olmo2ForCausalLM,
    Olmo2Model,
    Olmo2DecoderLayer,
    Olmo2Attention,
    Olmo2MLP,
    Olmo2RMSNorm,
    Injector,
    NullInjector,
    build_model,
)

__all__ = [
    "Olmo2Config", "SIZE_PRESETS", "Olmo2ForCausalLM", "Olmo2Model",
    "Olmo2DecoderLayer", "Olmo2Attention", "Olmo2MLP", "Olmo2RMSNorm",
    "Injector", "NullInjector", "build_model",
]
