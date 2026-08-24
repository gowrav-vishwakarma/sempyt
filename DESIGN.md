# sempyt — DESIGN

## 1. Core idea

A tensor is declared with **named Dim objects**. Ops are expressed over those
objects. The framework resolves objects → positional layout work
(`permute` / `view` / `expand` / `matmul` / `einsum`) and executes on standard
PyTorch. Element math is a **policy**.

```python
B, T, H, d = dims("B T H d")
q = named(q_raw, (B, T, H, d))
scores = contract(q.alias(T, Tq), k.alias(T, Tk), over=d)
```

Rank-agnostic: every position of the layout is a named `Dim`. Works for rank 1
through rank 7+.

## 2. Dim is an object

Carried from `qllm2/semantic/DESIGN.md`:

- `Dim(name, size=None, role=None)` — size symbolic until bound.
- Binding: first real tensor that uses a dim binds `size` from that axis.
- **Self-correlation**: the *same object* in two tensors must agree on size.
- Mismatch raises `ShapeError` naming both dims, both sizes, and the
  `file:line` where each size was first bound.
- `role` is documentary (`batch`, `time`, `complex_pair`, …) and never affects math.

`dims("B T H d")` is a batch constructor that returns four Dim objects. There
is no bytecode inspection (the `functorch.dim` trick that breaks on new CPython
releases).

### ProductDim

`HD = H * d`, `Kf = K * B * H`, `d2 = px * d`. Size is *derived*. This is the
only way to name a merged axis (Triton wants `K*B*H` as one batch dim).

**Both split/merge forms are supported:**

1. ProductDim (primary, exact, reversible).
2. DESIGN.md name-repetition + size-factorization: `x.to(B, T, H, d)` splitting
   a bound `D` when at most one factor is unbound.

### String form (optional)

`x.to("B H T d")` / `x.to("B (H d)")` parses against the tensor's own dims
plus an optional `DimRegistry` context. One resolver, two front doors.

## 3. NamedTensor

Wraps `torch.Tensor` + ordered layout + policy. Registered with
`torch.utils._pytree.register_pytree_node` so Dynamo flattens it to
`(tensor, static metadata)` and `torch.compile(fullgraph=True)` sees through it.

**Not** a `torch.Tensor` subclass. PyTorch named tensors died of a per-op
name-propagation table; `__torch_dispatch__` would recreate that, and
subclasses are unwrapped before FlashAttention / custom Triton anyway.

- `.shape` is a `Mapping[Dim, int]`. `x.shape[0]` is a `TypeError`.
- `.to(*dims)` — permute + split + merge + insert + squeeze. Identity → `self`.
- `.raw(*dims)` — contiguous plain tensor in that order. Mandatory kernel exit.
- `.alias(old, new)` — same storage, new identity (attention `Tq` / `Tk`).
- `+` / `*` — named broadcast (missing axes unsqueeze; shared axes size-check).

## 4. Semantic ops

| op | meaning |
|---|---|
| `contract(a, b, over=D)` | named matmul / inner product; SplitComplex → 4-real GEMM |
| `outer(a, b, over=(A,B))` | named outer product |
| `read(S, probe, over=D, conjugate=...)` | inner product, optional conj on probe |
| `write(S, u, k, over=(A,B))` | `S + outer(u, conj(k))` |
| `softmax` / `sum` / `mean` / `normalize` | named reductions |
| `pack` / `unpack` | ProductDim merge / split |
| `segment` / `unsegment` / `expand_legs` | pad→view→crop grids, `S = ceil(T/L)` |
| `gather` / `Index` / `at` | LUT / codebook integer lookup |
| `scan(..., mode=parallel\|chunked\|recurrent)` | `h_t = λ⊙h_{t-1} + u_t` |

## 5. Policies

`Real` (default), `SplitComplex(pair=px)` (v13 split-real `[..., 2]`,
`abs` +1e-8, `norm_vec` is `cnormalize_vec`), `NativeComplex` (torch
`complex64`, `torch.conj` / `abs`).

`px` with `role="complex_pair"` is kept out of `contract(over=d)` automatically.

## 6. Layers and zoo

`sempyt.nn`: Linear, RMSNorm, LayerNorm, Embed, FactorizedEmbed, TiedHead,
RoPE, GLU/SwiGLU, SDPAAttention (`.raw()` → SDPA), LinearAttention,
OuterProductMemory, PhaseLUT.

`sempyt.zoo`: GPT-2 (weight-compatible with pands `TransformerLM`), v13 PAM
primitives, pands VPS band + diag_scan, GLM/Kimi-style GQA+MoE block.

## 7. What is deliberately not here

Custom autograd. A YAML model compiler. DTensor logical-axis sharding (the
`Dim` objects are ready for a later rule table; not in v0.1).
