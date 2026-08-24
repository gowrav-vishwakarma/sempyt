"""v13 PAM primitives: sempyt zoo == raw-torch transcriptions (no v13 import)."""

import torch

from sempyt import Dim, SplitComplex, named
from sempyt.testing import assert_close
from sempyt.zoo.qllm_pam import decay_matrix, fused_chunk_step, project_qkv, recur_step_delta


def _cmul(a, b):
    return torch.stack(
        [
            a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1],
            a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0],
        ],
        dim=-1,
    )


def test_project_qkv_matches_view_transpose_rope():
    B, T, H, d, px = Dim("B", 2), Dim("T", 5), Dim("H", 3), Dim("d", 4), Dim("px", 2)
    slot = Dim("slot", 3)
    pol = SplitComplex(px)
    qkv = named(torch.randn(2, 5, 3, 3, 4, 2), (B, T, slot, H, d, px), pol)
    rope = named(torch.randn(5, 4, 2), (T, d, px), pol)
    q, k, v = project_qkv(qkv, B, T, slot, H, d, px, rope=rope)
    raw = qkv.data
    qr = raw[:, :, 0].transpose(1, 2).contiguous()
    kr = raw[:, :, 1].transpose(1, 2).contiguous()
    vr = raw[:, :, 2].transpose(1, 2).contiguous()
    qr, kr = _cmul(qr, rope.data), _cmul(kr, rope.data)
    assert_close(q.data, qr)
    assert_close(k.data, kr)
    assert_close(v.data, vr)


def _raw_recur_step_delta(memory, decay, value, key, query, write_beta, erase_beta=None):
    """Transcription of v13/model.py:1373-1416."""
    if erase_beta is None:
        erase_beta = write_beta
    decay_factor = decay.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    memory = memory * decay_factor
    predicted_real = (
        memory[..., 0] * key[..., 0].unsqueeze(-2) - memory[..., 1] * key[..., 1].unsqueeze(-2)
    ).sum(dim=-1)
    predicted_imag = (
        memory[..., 0] * key[..., 1].unsqueeze(-2) + memory[..., 1] * key[..., 0].unsqueeze(-2)
    ).sum(dim=-1)
    write_expanded = write_beta.unsqueeze(-1)
    erase_expanded = erase_beta.unsqueeze(-1)
    update_real = write_expanded * value[..., 0] - erase_expanded * predicted_real
    update_imag = write_expanded * value[..., 1] - erase_expanded * predicted_imag
    update = torch.stack([update_real, update_imag], dim=-1)
    key_conj = torch.stack([key[..., 0], -key[..., 1]], dim=-1)
    outer_real = (
        update[..., 0].unsqueeze(-1) * key_conj[..., 0].unsqueeze(-2)
        - update[..., 1].unsqueeze(-1) * key_conj[..., 1].unsqueeze(-2)
    )
    outer_imag = (
        update[..., 0].unsqueeze(-1) * key_conj[..., 1].unsqueeze(-2)
        + update[..., 1].unsqueeze(-1) * key_conj[..., 0].unsqueeze(-2)
    )
    memory = memory + torch.stack([outer_real, outer_imag], dim=-1)
    state_query_real = (
        memory[..., 0] * query[..., 0].unsqueeze(-2) - memory[..., 1] * query[..., 1].unsqueeze(-2)
    )
    state_query_imag = (
        memory[..., 0] * query[..., 1].unsqueeze(-2) + memory[..., 1] * query[..., 0].unsqueeze(-2)
    )
    output = torch.stack([state_query_real.sum(dim=-1), state_query_imag.sum(dim=-1)], dim=-1)
    return output, memory


def test_recur_step_delta_matches_v13_transcription():
    B, H, di, dj, px = Dim("B", 2), Dim("H", 3), Dim("di", 4), Dim("dj", 4), Dim("px", 2)
    pol = SplitComplex(px)
    mem = named(torch.randn(2, 3, 4, 4, 2), (B, H, di, dj, px), pol)
    dec = named(torch.sigmoid(torch.randn(2, 3)), (B, H))
    val = named(torch.randn(2, 3, 4, 2), (B, H, di, px), pol)
    key = named(torch.randn(2, 3, 4, 2), (B, H, dj, px), pol)
    qry = named(torch.randn(2, 3, 4, 2), (B, H, dj, px), pol)
    wb = named(torch.sigmoid(torch.randn(2, 3)), (B, H))
    eb = named(torch.sigmoid(torch.randn(2, 3)) * 0.9, (B, H))
    out, new_s = recur_step_delta(mem, dec, val, key, qry, wb, eb, di=di, dj=dj, px=px)
    exp_o, exp_s = _raw_recur_step_delta(mem.data, dec.data, val.data, key.data, qry.data, wb.data, eb.data)
    assert_close(out.data, exp_o, rtol=1e-5, atol=1e-6)
    assert_close(new_s.data, exp_s, rtol=1e-5, atol=1e-6)


def _raw_fused_chunk_step(
    queries, keys, values, decay_gamma, retrieval_phase, routing_weights, memory, is_first_chunk
):
    """Transcription of v13/model.py:812-892."""
    K, B, H, C = decay_gamma.shape
    d = queries.shape[-2]
    qr, qi = queries[..., 0], queries[..., 1]
    kr, ki = keys[..., 0], keys[..., 1]
    score_r = qr @ kr.transpose(-1, -2) + qi @ ki.transpose(-1, -2)
    score_i = qi @ kr.transpose(-1, -2) - qr @ ki.transpose(-1, -2)
    D = decay_matrix(decay_gamma.reshape(K * B * H, C), C).reshape(K, B, H, C, C)
    cos_p, sin_p = torch.cos(retrieval_phase), torch.sin(retrieval_phase)
    decay_r = ((routing_weights * cos_p).unsqueeze(-1) * D).sum(0)
    decay_i = ((routing_weights * sin_p).unsqueeze(-1) * D).sum(0)
    wr = score_r * decay_r - score_i * decay_i
    wi = score_r * decay_i + score_i * decay_r
    vr, vi = values[..., 0], values[..., 1]
    out_r = wr @ vr - wi @ vi
    out_i = wr @ vi + wi @ vr
    output = torch.stack([out_r, out_i], dim=-1)
    log_decay = torch.log(decay_gamma + 1e-6)
    cumulative = torch.exp(torch.cumsum(log_decay, dim=-1))
    if not is_first_chunk:
        mr, mi = memory[..., 0], memory[..., 1]
        qrs, qis = qr.unsqueeze(0), qi.unsqueeze(0)
        cr = (mr @ qrs.transpose(-1, -2) - mi @ qis.transpose(-1, -2)).transpose(-1, -2)
        ci = (mr @ qis.transpose(-1, -2) + mi @ qrs.transpose(-1, -2)).transpose(-1, -2)
        comb_r = routing_weights * cos_p * cumulative
        comb_i = routing_weights * sin_p * cumulative
        rr = (cr * comb_r.unsqueeze(-1) - ci * comb_i.unsqueeze(-1)).sum(0)
        ri = (cr * comb_i.unsqueeze(-1) + ci * comb_r.unsqueeze(-1)).sum(0)
        output = output + torch.stack([rr, ri], dim=-1)
    decay_last = D[:, :, :, -1, :]
    vrs, vis = vr.unsqueeze(0), vi.unsqueeze(0)
    wvr = vrs * decay_last.unsqueeze(-1)
    wvi = vis * decay_last.unsqueeze(-1)
    krs, kis = kr.unsqueeze(0), ki.unsqueeze(0)
    st_r = wvr.transpose(-1, -2) @ krs + wvi.transpose(-1, -2) @ kis
    st_i = wvi.transpose(-1, -2) @ krs - wvr.transpose(-1, -2) @ kis
    state_chunk = torch.stack([st_r, st_i], dim=-1)
    total = cumulative[:, :, :, -1]
    memory_new = memory * total[..., None, None, None] + state_chunk
    return output, memory_new


def test_fused_chunk_first_and_nonfirst():
    K, B, H, C, d, px = (
        Dim("K", 2),
        Dim("B", 2),
        Dim("H", 2),
        Dim("C", 4),
        Dim("d", 3),
        Dim("px", 2),
    )
    di, dj = Dim("di", 3), Dim("dj", 3)
    pol = SplitComplex(px)
    q = named(torch.randn(2, 2, 4, 3, 2), (B, H, C, d, px), pol)
    k = named(torch.randn(2, 2, 4, 3, 2), (B, H, C, d, px), pol)
    v = named(torch.randn(2, 2, 4, 3, 2), (B, H, C, d, px), pol)
    g = named(torch.sigmoid(torch.randn(2, 2, 2, 4)), (K, B, H, C))
    ph = named(torch.randn(2, 2, 2, 4), (K, B, H, C))
    rw = torch.softmax(torch.randn(2, 2, 2, 4), dim=0)
    rw = named(rw, (K, B, H, C))
    mem = named(torch.randn(2, 2, 2, 3, 3, 2), (K, B, H, di, dj, px), pol)

    for first in (True, False):
        out, new_s = fused_chunk_step(
            q, k, v, g, ph, rw, mem, first, K=K, B=B, H=H, C=C, d=d, px=px
        )
        exp_o, exp_s = _raw_fused_chunk_step(
            q.data, k.data, v.data, g.data, ph.data, rw.data, mem.data, first
        )
        assert_close(out.data, exp_o, rtol=1e-4, atol=1e-5, msg=f"first={first} out")
        assert_close(new_s.data, exp_s, rtol=1e-4, atol=1e-5, msg=f"first={first} state")
