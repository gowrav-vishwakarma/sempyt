# sempyt

Named-dimension tensors over PyTorch. A tensor is declared with **Dim objects**
(not strings, not integer axis indices). Every rearrange, broadcast, matmul,
and reduction is expressed over those objects. The library lowers eagerly to
plain `permute` / `view` / `expand` / `matmul` / `einsum` — thin wrappers, nothing
new to optimize.

```python
from sempyt import Dim, dims, named, contract, softmax

B, T, H, d = dims("B T H d")          # four Dim *objects*
q = named(q_raw, (B, T, H, d))        # layout: (B, T, H, d)
k = named(k_raw, (B, T, H, d))

Tq, Tk = Dim("Tq"), Dim("Tk")
scores = contract(q.alias(T, Tq), k.alias(T, Tk), over=d)   # (B, Tq, H, Tk)
attn = softmax(scores, over=Tk)
out = contract(attn, v.alias(T, Tk), over=Tk)               # (B, Tq, H, d)

y = x.to(B, H, T, d)   # not transpose(1, 2)
z = state * gamma      # not unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
```

A no-op `to()` whose layout already matches returns the same object and emits
no torch op.

## Why not einops / functorch.dim / named tensors?

| | einops | functorch.dim | PyTorch named tensors | **sempyt** |
|---|---|---|---|---|
| Axis identity | strings, per-call | Dim objects | strings (removed mid-2026) | **Dim objects** |
| Size correlation | kwargs at every unflatten | yes | no | **yes, with file:line provenance** |
| `x.shape[0]` | works (bad) | works | works | **TypeError — `.shape` is keyed by Dim** |
| contract / outer / scan / segment / LUT | no | no | no | **yes** |
| Complex algebra | no | no | no | **pluggable policy** |
| `torch.compile` | clean | hostile (bytecode naming) | gone | **pytree dataclass, fullgraph-clean** |
| FlashAttention / Triton | you already unwrap | subclass issues | n/a | **explicit `.raw(*dims)`** |

einops' string syntax is still accepted as an *optional* front door
(`x.to("B H T d")` / `x.to("B (H d)")`) and parses to the same Dim objects.

## The pain this deletes

v13 `model.py` used to say:

```python
if decay_gamma.dim() == memory_state.dim() - 3:
    decay_factor = decay_gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
else:
    decay_factor = decay_gamma.unsqueeze(-2).unsqueeze(-1)
memory_state = memory_state * decay_factor
# ...
qkv.view(B, T, 3, H, d, 2).transpose(1, 2).contiguous()
retrieval_phase.permute(3, 0, 2, 1)
mass.reshape(K * B * H, C, C)
```

sempyt:

```python
memory = memory * gamma          # named broadcast
q, k, v = project_qkv(qkv, B, T, slot, H, d, px)
phase = phase.to(K, B, H, T)
mass.raw(Kf, C, C)               # Kf = K * B * H  (ProductDim)
```

pands `from_real(x, dim)` needed an out-of-band size because `2d` was anonymous.
Here `d2 = px * d`; `x.to(B, d2)` / `x.to(B, px, d)` are exact inverses.

## Install

```bash
cd /home/gowrav/Development/sempyt
uv sync
# or, against an existing torch:
PYTHONPATH=src python -m pytest tests/
```

Requires Python ≥ 3.10 and `torch>=2.8`. No other runtime dependencies.

## Package

```
src/sempyt/
  dim.py       Dim, ProductDim, dims(), DimRegistry
  layout.py    LayoutPlan (permute / view / squeeze / unsqueeze)
  tensor.py    NamedTensor, .to() / .raw() / named broadcast
  ops.py       contract outer read write reduce softmax
               pack unpack segment gather scan
  policies.py  Real, SplitComplex, NativeComplex
  nn/          Linear RMSNorm LayerNorm Embed RoPE GLU
               SDPAAttention LinearAttention OuterProductMemory PhaseLUT
  zoo/         gpt.py  qllm_pam.py  pands_vps.py  glm.py
```

Reference models in `zoo/` are proven bit-close against:

- pands `v1/transformer_baseline.py` (loaded weights)
- a self-contained transcription of qllm2 `v13/model.py` `_project`,
  `_recur_step_delta`, `_fused_chunk_step` (v13 itself is never imported)
- pands v1 outer-product memory (parallel == chunked == recurrent) and v2
  `PhaseLUT` / `diag_scan`
- a GLM/Kimi-style GQA + MoE block (forward + backward smoke)

## Kernel exit

Names are frontend-only. `NamedTensor` is a pytree-registered dataclass, not a
`torch.Tensor` subclass — subclasses are unwrapped before FlashAttention / SDPA
/ custom Triton. Call `.raw(B, H, T, d)` to get a contiguous plain tensor in
that order; wrap the result back with `named(out, (B, H, T, d))`.
