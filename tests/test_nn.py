"""Shape + grad-flow tests for sempyt.nn layers."""

import torch

from sempyt import Dim, ProductDim, SplitComplex, named
from sempyt.nn import (
    Embed,
    FactorizedEmbed,
    GLU,
    LayerNorm,
    Linear,
    LinearAttention,
    OuterProductMemory,
    PhaseLUT,
    RMSNorm,
    RoPE,
    SDPAAttention,
    SwiGLU,
    TiedHead,
)
from sempyt.testing import assert_close


def test_linear_matches_torch_and_grads():
    inn, out = Dim("in", 8), Dim("out", 5)
    B, T = Dim("B", 2), Dim("T", 3)
    layer = Linear(inn, out)
    ref = torch.nn.Linear(8, 5)
    ref.weight.data.copy_(layer.weight.data)
    ref.bias.data.copy_(layer.bias.data)
    x = named(torch.randn(2, 3, 8, requires_grad=True), (B, T, inn))
    y = layer(x)
    exp = ref(x.data)
    assert_close(y.data, exp)
    y.data.sum().backward()
    assert layer.weight.grad is not None


def test_rmsnorm_layernorm():
    d = Dim("d", 8)
    B = Dim("B", 2)
    x = named(torch.randn(2, 8), (B, d))
    y = RMSNorm(d)(x)
    assert y.layout == (B, d)
    z = LayerNorm(d)(x)
    assert z.layout == (B, d)
    z.data.sum().backward()


def test_embed_and_tied_head():
    V, d = Dim("V", 20), Dim("d", 8)
    B, T = Dim("B", 2), Dim("T", 4)
    emb = Embed(V, d)
    ids = torch.randint(0, 20, (2, 4))
    h = emb(ids, B, T)
    assert h.data.shape == (2, 4, 8)
    logits = TiedHead(emb)(h)
    assert logits.layout[-1] is V
    logits.data.sum().backward()


def test_factorized_embed():
    V, d, r = Dim("V", 20), Dim("d", 8), Dim("r", 3)
    B, T = Dim("B", 2), Dim("T", 4)
    h = FactorizedEmbed(V, d, r)(torch.randint(0, 20, (2, 4)), B, T)
    assert h.data.shape == (2, 4, 8)


def test_rope_cmul():
    Tmax, d, px = Dim("Tmax", 16), Dim("d", 8), Dim("px", 2, role="complex_pair")
    B, H, Tr = Dim("B", 2), Dim("H", 2), Dim("Tr", 6)
    pol = SplitComplex(px)
    rope = RoPE(Tmax, d, px)
    q = named(torch.randn(2, 2, 6, 8, 2), (B, H, Tr, d, px), pol)
    y = q * named(rope.cache[:6], (Tr, d, px), pol)
    assert y.data.shape == q.data.shape


def test_swiglu():
    d, h = Dim("d", 8), Dim("h", 16)
    B = Dim("B", 2)
    x = named(torch.randn(2, 8, requires_grad=True), (B, d))
    y = SwiGLU(d, h)(x)
    assert y.layout[-1] is d
    y.data.sum().backward()


def test_sdpa_attention():
    d, H, dh = Dim("d", 16), Dim("H", 2), Dim("dh", 8)
    B, T = Dim("B", 2), Dim("T", 5)
    attn = SDPAAttention(d, H, dh, dropout=0.0)
    x = named(torch.randn(2, 5, 16, requires_grad=True), (B, T, d))
    y = attn(x, time=T, batch=B)
    assert y.data.shape == (2, 5, 16)
    y.data.sum().backward()
    assert attn.qkv.weight.grad is not None


def test_linear_attention_and_opm():
    d, H, dh = Dim("d", 16), Dim("H", 2), Dim("dh", 8)
    B, T = Dim("B", 2), Dim("T", 6)
    x = named(torch.randn(2, 6, 16, requires_grad=True), (B, T, d))
    y = LinearAttention(d, H, dh)(x, time=T, batch=B)
    assert y.data.shape == (2, 6, 16)
    opm = OuterProductMemory(d, H, dh)
    y2, state = opm.recurrent(x.detach(), time=T, batch=B)
    assert y2.data.shape == (2, 6, 16)
    assert state.data.shape[-2:] == (8, 8)


def test_phase_lut():
    bins, px = Dim("bins", 256), Dim("px", 2, role="complex_pair")
    lut = PhaseLUT(bins, px, phase_mode="exact")
    theta = named(torch.tensor([0.0, 0.5 * 3.14159265]), (Dim("n", 2),))
    z = lut.from_angle(theta, mode="exact")
    assert z.data.shape == (2, 2)
    assert_close(z.data[0, 0], torch.tensor(1.0), atol=1e-5)
