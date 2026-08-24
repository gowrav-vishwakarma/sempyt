# sempyt

> New to tensors? Start with the beginner walkthrough in [`docs/`](docs/README.md) —
> it teaches matrices and every operation with visual before/after grids, then the
> PyTorch way, then the sempyt one-liner.

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

**Every op that torch spells with `dim=N` is named.** `stack`/`unstack`, `cat`,
`split`, `select`, `take`, `cumsum`/`cumprod`, `argmax`/`argmin`, `topk`,
`repeat_along`, `shift`/`roll`/`flip`/`pad`, `where`, comparisons, `l2_normalize`,
and the unary maps (`silu`/`gelu`/`sigmoid`/`exp`/`log`/`cos`/`sin`/…) all take a
`Dim`, never an integer axis. See [`EXAMPLES.md`](EXAMPLES.md) for the full tour
of every operation category with higher-dimensional (not just row/column)
examples.

## Why we built this

PyTorch had a brief named-tensor experiment, but it was removed in mid-2026.
That leaves researchers with two choices: juggle `dim=0,1,2,3` indices, or use
`einops` strings on every call. Both force you to spend brain cycles on *where* an
axis is instead of *what* it means.

When you are doing R&D, you want to think in terms of the logic:

- "the query is over `Tq`, the key is over `Tk`, contract over `d`"
- "decay `gamma` is per head, so broadcast it over `B` and `T`"
- "split the fused QKV into `slot`, `heads`, `head_dim`"

You do **not** want to mentally map `B=0, T=1, H=2, d=3`, count `.unsqueeze()`
calls, or remember which `.transpose()` you did five lines ago. sempyt brings
named axes back as first-class objects, so the code reads like the idea and
the compiler sorts out the indices.

## Torch vs. others vs. sempyt

### Attention scores: "contract over `d`"

```python
# raw PyTorch: you track positions and write permutations
def attention_scores_torch(q, k):
    # q: (B, Tq, H, d)   k: (B, Tk, H, d)
    q_t = q.transpose(1, 2)              # (B, H, Tq, d)
    k_t = k.transpose(1, 2).transpose(2, 3)  # (B, H, d, Tk)
    scores = torch.matmul(q_t, k_t)      # (B, H, Tq, Tk)
    return scores.permute(0, 2, 1, 3)      # back to (B, Tq, H, Tk)

# einops: clearer, but you re-declare strings on every call
from einops import einsum
def attention_scores_einops(q, k):
    return einsum(q, k, "B Tq H d, B Tk H d -> B Tq H Tk")

# sempyt: the names are objects you define once, the code is the idea
from sempyt import dims, named, contract
B, Tq, Tk, H, d = dims("B Tq Tk H d")
scores = contract(q.alias(T, Tq), k.alias(T, Tk), over=d)   # (B, Tq, H, Tk)
```

### Broadcasting: "multiply `state` by per-head `gamma`"

```python
# raw PyTorch: how many unsqueezes? depends on where H is
gamma = torch.sigmoid(decay_bias)      # (H,)
if gamma.dim() == state.dim() - 3:
    gamma = gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
else:
    gamma = gamma.unsqueeze(-2).unsqueeze(-1)
state = state * gamma

# sempyt: the broadcast is automatic because H is just a named axis
state = state * named(gamma, (H,))
```

### Splitting fused QKV

```python
# raw PyTorch: reshape, permute, contiguous, slice
qkv = qkv.view(B, T, 3, H, d, 2).transpose(1, 2).contiguous()
q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

# sempyt: name the fused axis, then un-name it into three named axes
slot = Dim("slot", 3)
qkv3 = Dim("qkv3", 3 * H.size * d.size)
projected = linear(x)                         # (B, T, qkv3)
split = projected.to(B, T, slot, H, d)        # (B, T, 3, H, d)
q, k, v = split.select(slot, 0), \
          split.select(slot, 1), \
          split.select(slot, 2)
```

### Comparison table

| | raw PyTorch | einops | functorch.dim | PyTorch named tensors | **sempyt** |
|---|---|---|---|---|---|
| Axis identity | integer indices | strings, per-call | Dim objects | strings (removed mid-2026) | **Dim objects** |
| Size correlation | manual | no | yes | no | **yes, with file:line provenance** |
| `x.shape[0]` | works (bad) | works | works | works | **TypeError — `.shape` is keyed by Dim** |
| contract / outer / scan / segment / LUT | manual | no | no | no | **yes** |
| Complex algebra | manual | no | no | no | **pluggable policy** |
| `torch.compile` | clean | clean | hostile (bytecode naming) | gone | **pytree dataclass, fullgraph-clean** |
| FlashAttention / Triton | native | you already unwrap | subclass issues | n/a | **explicit `.raw(*dims)`** |
| Readability | "what axis is index 2?" | "what did I name it this time?" | good | good | **good, and the names are stable objects** |

sempyt is not a replacement for PyTorch — it is a thin, readable front-end that
still lets you drop to raw tensors at kernel boundaries.

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

## Origin

sempyt was built while doing R&D in [`qllm2`](https://github.com/gowrav-vishwakarma/qllm2)
— a language-model research repo. We needed a way to write attention, recurrent
memory, and VSA/phase operations without constantly reverse-engineering which
tensor index meant which logical axis. PyTorch named tensors were gone, and
existing tools were either string-based or subclass-unfriendly, so we made
sempyt: a small, object-first named-tensor wrapper over PyTorch that keeps the
code close to the idea.

## Status

sempyt is **beta** and not yet published on PyPI. The API is still settling as we
drive it from the R&D work in `qllm2`. Expect renames and small breaking changes
until we tag a 1.0 release.

## Install

### From a local clone

```bash
cd /home/gowrav/Development/sempyt
uv sync
# or, against an existing torch:
PYTHONPATH=src python -m pytest tests/
```

### From GitHub (once the repo is public)

Because sempyt is not on PyPI, you can install it directly from a Git URL once it
is pushed to GitHub. Replace the URL with the actual repository path:

```bash
pip install git+https://github.com/YOUR_USERNAME/sempyt.git
```

For editable development from a Git checkout:

```bash
git clone https://github.com/YOUR_USERNAME/sempyt.git
cd sempyt
pip install -e .
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
  structural.py stack unstack cat split select take cumsum argmax topk
               repeat_along shift roll flip pad where ge/gt/le/lt l2_normalize
               + unary maps (silu gelu sigmoid exp log cos sin …)
  policies.py  Real, SplitComplex, NativeComplex
  nn/          Linear RMSNorm LayerNorm Embed RoPE GLU
               SDPAAttention LinearAttention OuterProductMemory PhaseLUT
  zoo/         gpt.py bert.py glm.py mamba.py rwkv.py moe.py vsa.py
               pands_vps.py qllm_pam.py
```

Reference models in `zoo/` are proven bit-close against:

- pands `v1/transformer_baseline.py` (loaded weights)
- a self-contained transcription of qllm2 `v13/model.py` `_project`,
  `_recur_step_delta`, `_fused_chunk_step` (v13 itself is never imported)
- pands v1 outer-product memory (parallel == chunked == recurrent) and v2
  `PhaseLUT` / `diag_scan`
- a GLM/Kimi-style GQA + MoE block (forward + backward smoke)

Further architectures written on the named surface (forward/backward + property
tests): a Mamba-style selective SSM (`mamba.py`), an RWKV-style time/channel mix
(`rwkv.py`), a true top-k Mixture-of-Experts (`moe.py`), a bidirectional BERT
encoder (`bert.py`), and phasor VSA bind/bundle/cleanup (`vsa.py`).

## Kernel exit

Names are frontend-only. `NamedTensor` is a pytree-registered dataclass, not a
`torch.Tensor` subclass — subclasses are unwrapped before FlashAttention / SDPA
/ custom Triton. Call `.raw(B, H, T, d)` to get a contiguous plain tensor in
that order; wrap the result back with `named(out, (B, H, T, d))`.
