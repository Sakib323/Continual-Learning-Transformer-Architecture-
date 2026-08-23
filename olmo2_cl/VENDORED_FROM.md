# Provenance

This is an **architecture-faithful reimplementation** of OLMo 2, not a copy of
AI2's source file. It was written to be self-contained and hackable: no
dependency on `transformers` internals, and the mechanism injection points are
designed in rather than bolted on.

## Reference sources

| What | Where |
|---|---|
| HF single-file implementation | `huggingface/transformers` → `src/transformers/models/olmo2/modeling_olmo2.py` |
| Diff-against-Llama view | same directory → `modular_olmo2.py` |
| AI2 training stack | `allenai/OLMo-core` → `olmo_core/nn/` |
| Weights & configs | `huggingface.co/allenai` → OLMo-2 model repos |

## OLMo 2 architecture choices reproduced here

- **Reordered norm** — RMSNorm applied to the *output* of attention and the
  feed-forward, inside the residual, instead of to the sublayer input.
- **QK-Norm** — RMSNorm on the query and key projections before RoPE.
- **RoPE**, **SwiGLU**, **no biases**, parametric RMSNorm, GQA.

## If you rebase against upstream

Check these first, they are the parts most likely to drift:
1. Norm placement in `Olmo2DecoderLayer.forward`
2. Whether QK-Norm is applied pre- or post-head-split
3. RoPE scaling (none here; upstream may add it for long context)
4. Initialisation scheme — upstream uses a scaled init we simplify

Nothing in `clms/` depends on this file's internals except through the
`Injector` protocol, so a rebase is contained to this directory.
