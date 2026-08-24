# sempyt by example

> This is the terse reference. If you are new to tensors, start with the
> beginner walkthrough in [`docs/`](docs/README.md) — it teaches each operation
> with visual before/after grids and plain-language factory analogies.

The one rule: **name the axis, never count it.** Anywhere PyTorch would make
you write `dim=1`, `dim=-2`, `.transpose(1, 2)`, `.unsqueeze(-1)`,
`x.reshape(B, H, T, d)`, `torch.stack(xs, dim=2)`, `topk(..., dim=-1)`, or
`repeat_interleave(..., dim=1)`, sempyt lets you pass the `Dim` object instead.

Think in N dimensions, not "rows and columns". A dim is a name with an identity:
`tokens`, `heads`, `state`, `experts`, `keys`, `hyperdim`, `pair` — whatever the
axis *means*. The size rides along and is checked for you.

```python
import torch
import sempyt as sp
from sempyt import Dim, dims, named
```

---

## 1. Dims and layouts

A `Dim` is an object. Two tensors share an axis iff they share the same `Dim`
object (identity, not name string).

```python
# make several bound dims at once
batch, tokens, chan = dims("batch tokens chan", batch=4, tokens=128, chan=512)

# an unbound dim binds to whatever size it first meets, then is checked forever
heads = Dim("heads")                 # size unknown yet
head_dim = Dim("head_dim", 64)       # bound now

x = named(torch.randn(4, 128, 512), (batch, tokens, chan))
x.shape[chan]        # 512   (keyed by Dim, never x.shape[2])
x.size(tokens)       # 128
```

A **ProductDim** is a composite axis — the flattened product of others. It makes
split/merge exact and reversible.

```python
inner = sp.ProductDim(heads, head_dim)   # inner.size == heads.size * head_dim.size
```

---

## 2. `to(...)` — the one reshape/permute/split/merge operation

`to` takes a target layout of `Dim`s and figures out the `permute` + `reshape` +
`squeeze`/`unsqueeze` to get there. You never chain transposes again.

```python
# permute: name the order you want
x.to(chan, batch, tokens)                       # (512, 4, 128) view

# merge two axes into a product axis
qkv = named(torch.randn(4, 128, 8, 64), (batch, tokens, heads, head_dim))
merged = qkv.to(batch, tokens, inner)           # (4, 128, 512)  heads·head_dim fused

# split a product axis back out
back = merged.to(batch, tokens, heads, head_dim)  # (4, 128, 8, 64)

# "reshape to (B, d, H, T)" is just naming the four dims in that order:
qkv.to(batch, head_dim, heads, tokens)
```

Splitting a plain (non-product) axis into freshly named pieces uses `split`
(section 6). Merging arbitrary axes uses a `ProductDim` as above.

---

## 3. `raw(...)` — the explicit exit to a bare tensor

Kernels (SDPA, FlashAttention, Triton, einsum) don't speak names. `raw` hands you
a contiguous `torch.Tensor` in exactly the order you name — the one place numeric
axes reappear, on purpose.

```python
q = named(torch.randn(4, 8, 128, 64), (batch, heads, tokens, head_dim))
y = torch.nn.functional.scaled_dot_product_attention(
    q.raw(batch, heads, tokens, head_dim),      # you name the kernel's layout
    k.raw(batch, heads, tokens, head_dim),
    v.raw(batch, heads, tokens, head_dim),
    is_causal=True,
)
yn = named(y, (batch, heads, tokens, head_dim))  # re-enter the named world
```

---

## 4. Elementwise math and named broadcasting

`+ - * /` broadcast over the **union of named axes** — no manual `unsqueeze` to
line things up. A per-channel gain `(chan,)` multiplies a `(batch, tokens, chan)`
tensor directly.

```python
gain = named(torch.randn(512), (chan,))
y = x * gain                                     # broadcasts over batch, tokens

bias = named(torch.randn(4, 128), (batch, tokens))
z = x + bias                                     # broadcasts over chan
```

Unary maps keep the layout and are all named ops (no `dim`):

```python
sp.silu(x); sp.gelu(x); sp.sigmoid(x); sp.tanh(x); sp.relu(x)
sp.exp(x); sp.log(x); sp.sqrt(x); sp.cos(x); sp.sin(x); sp.softplus(x)
sp.square(x); sp.clamp(x, min=-5, max=5)
sp.apply(x, lambda t: t.pow(3))                  # arbitrary elementwise fn
```

---

## 5. Reductions — name the axis you collapse

```python
sp.sum(x, over=chan)             # (batch, tokens)      — was x.sum(dim=-1)
sp.mean(x, over=tokens)          # (batch, chan)
sp.softmax(logits, over=vocab)   # softmax along the named vocab axis
sp.reduce(x, over=(tokens, chan), how="max")   # reduce several named axes at once
```

---

## 6. Structural ops — build and cut named axes

### stack / unstack — introduce or remove an axis

```python
per_head = [named(torch.randn(4, 128, 64), (batch, tokens, head_dim)) for _ in range(8)]
stacked = sp.stack(per_head, into=heads)             # new 'heads' axis, size 8
stacked = sp.stack(per_head, into=heads, at=tokens)  # insert before 'tokens'
pieces  = sp.unstack(stacked, over=heads)            # back to a list of 8
```

### cat — concatenate along a named axis

```python
a = named(torch.randn(4, 3, 512), (batch, tokens, chan))
b = named(torch.randn(4, 5, 512), (batch, tokens, chan))
ab = sp.cat([a, b], over=tokens)                     # (4, 8, 512)
```

### split — cut one axis into named pieces (the classic qkv split)

```python
q_dim, k_dim, v_dim = dims("q k v", q=512, k=512, v=512)
packed = named(torch.randn(4, 128, 1536), (batch, tokens, Dim("qkv", 1536)))
q, k, v = sp.split(packed, over=packed.layout[-1], sizes=(512, 512, 512),
                   names=(q_dim, k_dim, v_dim))
```

### select / take — pick a position or a range along a named axis

```python
last_tok = sp.select(x, over=tokens, index=-1)       # drop 'tokens', keep last
last_tok = x[tokens, -1]                              # same thing, sugar
window   = sp.take(x, over=tokens, start=2, length=8, new=Dim("win"))  # narrow
```

---

## 7. cumulative / argmax / topk — over a named axis

```python
sp.cumsum(x, over=tokens)         # was x.cumsum(dim=1)
sp.cumprod(gates, over=tokens)

idx = sp.argmax(logits, over=vocab)          # returns an Index over 'vocab'
next_token = idx.values                      # the raw integer tensor

K = Dim("topk")
vals, ids = sp.topk(logits, over=vocab, k=5, kdim=K)   # values on new 'topk' axis
```

`Index` is a named integer tensor that remembers which axis it points into — feed
it straight to `gather`/`at` (section 10).

---

## 8. repeat / shift / roll / flip / pad — over a named axis

```python
# GQA: expand kv-heads up to full head count by naming the axis (not dim=1)
kv_heads = Dim("kv_heads", 2)
kv = named(torch.randn(4, 2, 128, 64), (batch, kv_heads, tokens, head_dim))
q_heads = sp.repeat_along(kv, over=kv_heads, times=4, new=heads)   # -> 8 heads

sp.shift(x, over=tokens, by=1)     # RWKV token-shift: look back one, zero-fill
sp.roll(x, over=tokens, shift=1)   # circular
sp.flip(x, over=tokens)            # reverse a named axis
sp.pad(x, over=tokens, left=1, right=2)   # pad a named axis; fresh sized dim
```

---

## 9. where / comparisons / normalize

```python
mask = sp.gt(x, 0.0)                         # named boolean tensor
y = sp.where(mask, x, sp.zeros_like(x))      # all three broadcast by name
sp.ge(a, b); sp.le(a, b); sp.lt(a, b)
unit = sp.l2_normalize(x, over=chan)         # unit norm along a named axis
```

---

## 10. contract / outer / read / write — named tensor algebra

`contract` is a named matmul: it sums over the shared axis you name, keeps the
rest as batch. No more bookkeeping which of `bhtd,bhsd->bhts` is which.

```python
keys, qs = Dim("keys", 128), Dim("qs", 128)
q = named(torch.randn(4, 8, 128, 64), (batch, heads, qs, head_dim))
k = named(torch.randn(4, 8, 128, 64), (batch, heads, keys, head_dim))
scores = sp.contract(q, k, over=head_dim)    # (batch, heads, qs, keys)

# outer product builds two new axes (associative memory write)
di, dj = Dim("di", 64), Dim("dj", 64)
v = named(torch.randn(4, 8, 64), (batch, heads, di))
kk = named(torch.randn(4, 8, 64), (batch, heads, dj))
S = sp.outer(v, kk, over=(di, dj))           # (batch, heads, di, dj)

# read = inner product (with optional conjugate for complex probes)
y = sp.read(S, q_probe, over=dj)
```

---

## 11. gather / at — integer lookups over a named axis (LUTs, embeddings)

```python
table = named(torch.randn(1000, 64), (Dim("rows", 1000), Dim("feat", 64)))
idx = sp.argmax(scores, over=keys)           # an Index
rows = sp.at(table, idx, over=table.layout[0])   # gather rows by the Index
```

---

## 12. segment / unsegment — chunk a long axis into (blocks, block_len)

```python
blocks, blen = Dim("blocks"), Dim("blen", 64)
chunked, info = sp.segment(x, time=tokens, into=(blocks, blen))  # pads if needed
restored = sp.unsegment(chunked, info)       # exact inverse, crops padding
```

---

## 13. scan — linear recurrence `h_t = λ ⊙ h_{t-1} + u_t` over a named axis

`decay` may omit the time axis (constant λ, LRU/RWKV-style) or include it
(per-step γ_t, Mamba-style). Modes: `recurrent`, `chunked`, `parallel`.

```python
u   = named(torch.randn(4, 128, 512), (batch, tokens, chan))
lam = named(torch.rand(512) * 0.9, (chan,))          # constant per-channel decay
h = sp.scan(u, lam, over=tokens, mode="chunked")     # (4, 128, 512)
```

---

## 14. Element policies — real and complex without leaving names

Complex data is stored split-real on a named `pair` axis `[..., px]`. The policy
makes `*` a complex multiply, `conj()` flip the imaginary part, etc.

```python
from sempyt import SplitComplex
px = Dim("px", 2)
pol = SplitComplex(px)
z = named(torch.randn(4, 128, 64, 2), (batch, tokens, head_dim, px), pol)
w = z * z            # complex multiply on the pair axis
r = sp.select(z, over=px, index=0)   # real part
```

---

## 15. Named `nn` layers

Layers take and return `NamedTensor`s and are declared in terms of `Dim`s.

```python
from sempyt.nn import Linear, RMSNorm, RoPE, SDPAAttention, SwiGLU

proj = Linear(chan, inner, bias=False)       # in-axis 'chan' -> out-axis 'inner'
norm = RMSNorm(chan)
attn = SDPAAttention(chan, heads, head_dim, causal=True)

h = attn(norm(x), time=tokens, batch=batch)  # returns a NamedTensor
```

---

## 16. Where to look next — the zoo

Each model in `sempyt.zoo` is a full architecture written with the surface above,
numeric-axis-free in the model body (kernels stay behind `raw`):

| Model | Highlights (named ops it shows off) |
|-------|-------------------------------------|
| `zoo.gpt.GPTLM` | qkv `split`/`unstack` over a `slot` axis, SDPA at the `raw` edge |
| `zoo.bert.BertEncoder` | same blocks, **bidirectional** (`causal=False`) |
| `zoo.glm.GLMBlock` | GQA `repeat_along` head expand, RoPE complex-mul, soft MoE via `unstack` |
| `zoo.mamba.MambaBlock` | selective `scan` with per-step decay over `time`, `state` axis |
| `zoo.rwkv.RWKVBlock` | `shift` token-mix, WKV as two named `scan`s |
| `zoo.moe.TopKMoE` | true top-k routing via `topk` + `where` + `unstack` over `E` |
| `zoo.pands_vps.OuterProductBand` | `outer`/`contract` memory, `select`+`stack` recurrent read |
| `zoo.vsa` | phasor HDC: `bind`/`unbind`/`bundle`/`cleanup` over a hyperdim axis |
| `zoo.qllm_pam` | v13 complex delta-rule primitives (kernel transcription) |
```
