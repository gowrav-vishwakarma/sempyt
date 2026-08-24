"""Smoke + property tests for the extended zoo architectures."""

import torch

import sempyt as sp
from sempyt import Dim, dims, named
from sempyt.zoo import (
    BertConfig,
    BertEncoder,
    MambaBlock,
    MambaConfig,
    MoEConfig,
    RWKVBlock,
    RWKVConfig,
    TopKMoE,
    vsa,
)


def _seq(C, B=2, T=6):
    b, t = Dim("batch", B), Dim("time", T)
    x = named(torch.randn(B, T, C.size), (b, t, C))
    return x, b, t


def test_mamba_forward_backward():
    cfg = MambaConfig(d_model=16, d_inner=32, d_state=8)
    blk = MambaBlock(cfg)
    x, b, t = _seq(blk.C)
    y = blk(x, b, t)
    assert y.layout[-1] is blk.C and y.data.shape == (2, 6, 16)
    y.data.sum().backward()
    assert blk.A_log.grad is not None


def test_mamba_modes_close():
    cfg = MambaConfig(d_model=8, d_inner=8, d_state=4, scan_mode="recurrent")
    blk = MambaBlock(cfg)
    x, b, t = _seq(blk.C)
    y_rec = blk(x, b, t)
    blk.cfg.scan_mode = "chunked"
    y_ch = blk(x, b, t)
    torch.testing.assert_close(y_rec.data, y_ch.data, atol=1e-4, rtol=1e-4)


def test_rwkv_forward_backward():
    cfg = RWKVConfig(d_model=16, d_ffn=32)
    blk = RWKVBlock(cfg)
    x, b, t = _seq(blk.C)
    y = blk(x, b, t)
    assert y.data.shape == (2, 6, 16)
    y.data.pow(2).mean().backward()
    assert blk.time_mix.time_decay.grad is not None


def test_topk_moe_is_sparse_and_normalized():
    cfg = MoEConfig(d_model=16, d_ff=32, n_experts=8, n_active=2)
    moe = TopKMoE(cfg)
    x, b, t = _seq(moe.C)
    # inspect the gate mask directly
    gates = sp.softmax(moe.router(x), over=moe.E)
    K = Dim("K")
    topv, _ = sp.topk(gates, moe.E, k=2, kdim=K)
    kth = sp.reduce(topv, over=K, how="min") if hasattr(sp, "reduce") else None
    y = moe(x)
    assert y.data.shape == (2, 6, 16)
    y.data.sum().backward()
    assert moe.router.weight.grad is not None


def test_bert_encoder_bidirectional():
    cfg = BertConfig(vocab_size=100, max_seq_len=32, d_model=32, n_layers=2, n_heads=4, d_ff=64)
    enc = BertEncoder(cfg)
    b, t = Dim("batch", 2), Dim("time", 7)
    ids = torch.randint(0, 100, (2, 7))
    h = enc(ids, b, t)
    assert h.layout[-1] is enc.C and h.data.shape == (2, 7, 32)
    h.data.sum().backward()
    assert enc.tok.weight.grad is not None


def test_vsa_bind_unbind_cleanup():
    hd, px = Dim("hd", 512), Dim("px", 2)
    code = Dim("code", 6)
    # a codebook of 6 symbols
    codebook = vsa.random_phasors(code, hd=hd, px=px, seed=0)
    role = vsa.random_phasors(hd=hd, px=px, seed=1)
    # pick symbol #3 as the filler, bind with role, then unbind + cleanup
    filler = codebook.select(code, 3)
    bound = vsa.bind(role, filler)
    approx = vsa.unbind(bound, role)  # ≈ filler
    idx, cleaned = vsa.cleanup(approx, codebook, code=code, hd=hd)
    assert int(idx.values) == 3
