# Agent rules (sempyt)

## Project

**sempyt** — named-dimension tensor layer over standard PyTorch.
Package root is `/home/gowrav/Development/sempyt`. This is a **separate
package** from qllm2 and pands. Do not import or edit `qllm2/v11`,
`qllm2/v13`, `qllm2/v7`, or `qllm2/scripts`. The draft in
`qllm2/semantic/` is superseded and left untouched.

Reference tests **transcribe** raw-torch expressions (or import pands'
public `v1.transformer_baseline`) so they stay self-contained.

## Run Python

```bash
cd /home/gowrav/Development/sempyt
# preferred, once the local venv exists:
uv run pytest tests/

# against the qllm2 torch install (no extra download):
PYTHONPATH=src /home/gowrav/Development/qllm2/.venv/bin/python -m pytest tests/
```

`uv sync` after changing `pyproject.toml`. Runtime dep is only `torch>=2.8`.

## Layout

| Path | Purpose |
|---|---|
| `src/sempyt/dim.py` | `Dim`, `ProductDim`, `dims()`, `DimRegistry` |
| `src/sempyt/layout.py` | `LayoutPlan` resolver + plan cache |
| `src/sempyt/tensor.py` | `NamedTensor`, pytree, operators, `.raw()` |
| `src/sempyt/ops.py` | contract / outer / segment / gather / scan |
| `src/sempyt/policies.py` | Real / SplitComplex / NativeComplex |
| `src/sempyt/nn/` | named-axis layers |
| `src/sempyt/zoo/` | GPT, qllm PAM, pands VPS, GLM-style |
| `tests/` | unit + numeric-equivalence proofs |
| `DESIGN.md` | semantics |
| `README.md` | why this is not einops |

## Conventions

- Dim is an **object**. `dims("B T H d")` returns objects. The string form of
  `to()` is an optional escape hatch that parses to the same objects.
- `.shape` is keyed by Dim. Never teach callers `x.shape[0]`.
- Names are frontend-only. Kernel / SDPA / Triton calls go through `.raw(*dims)`.
- No `torch.Tensor` subclass. No `__torch_dispatch__`. No bytecode dim-naming.
- Split-real complex is a **policy**, not a special case in the core.
- Keep changes minimal; match existing style.
- After a verified working change, commit with what / why / test evidence.

## Tests that must stay green

1. Layout algebra + every `ShapeError` path + no-op `to()` returns `self`.
2. GPT zoo == pands `TransformerLM` with loaded weights.
3. v13 `_recur_step_delta` / `_fused_chunk_step` / `_project` transcriptions.
4. pands v1 memory: parallel == chunked == recurrent.
5. pands v2 PhaseLUT + diag_scan.
6. GLM block forward + backward.
7. `torch.compile(fullgraph=True)` on a sempyt Linear+rearrange.
