"""v13 PAM primitives in semantic language. No import of qllm2/v13."""

from __future__ import annotations

from typing import Optional

import torch

from ..dim import Dim
from ..ops import contract, outer
from ..policies import SplitComplex
from ..tensor import NamedTensor, named


def decay_matrix(gamma: torch.Tensor, seq_len: int) -> torch.Tensor:
    """CPU reference of v13 ``_pt_decay_matrix`` (triton_kernels.py:522-528)."""
    log_gamma = torch.log(gamma + 1e-6)
    cum_neg = torch.cumsum(-log_gamma, dim=-1)
    log_decay = (cum_neg.unsqueeze(-1) - cum_neg.unsqueeze(-2)).transpose(-1, -2)
    causal = torch.tril(torch.ones(seq_len, seq_len, device=gamma.device, dtype=gamma.dtype))
    log_decay = log_decay * causal + (1 - causal) * (-1e4)
    return torch.exp(log_decay.clamp(max=0.0))


def project_qkv(
    qkv: NamedTensor,
    B: Dim,
    T: Dim,
    slot: Dim,
    H: Dim,
    d: Dim,
    px: Dim,
    rope: Optional[NamedTensor] = None,
) -> tuple[NamedTensor, NamedTensor, NamedTensor]:
    """``qkv [B,T,3,H,d,2] → Q,K,V [B,H,T,d,2]``, optional RoPE cmul."""
    split = qkv.to(B, T, slot, H, d, px)
    raw = split.to(B, slot, H, T, d, px).data
    q = named(raw[:, 0].contiguous(), (B, H, T, d, px), qkv.policy)
    k = named(raw[:, 1].contiguous(), (B, H, T, d, px), qkv.policy)
    v = named(raw[:, 2].contiguous(), (B, H, T, d, px), qkv.policy)
    if rope is not None:
        q = q * rope
        k = k * rope
    return q, k, v


def recur_step_delta(
    memory: NamedTensor,
    decay: NamedTensor,
    value: NamedTensor,
    key: NamedTensor,
    query: NamedTensor,
    write_beta: NamedTensor,
    erase_beta: Optional[NamedTensor] = None,
    *,
    di: Dim,
    dj: Dim,
    px: Dim,
) -> tuple[NamedTensor, NamedTensor]:
    """One gated delta step. Semantic form of v13 ``_recur_step_delta``.

    memory: (B, H, di, dj, px); decay/betas: (B, H);
    value: (B, H, di, px); key/query: (B, H, dj, px).
    """
    if erase_beta is None:
        erase_beta = write_beta
    memory = memory * decay
    # pred = S @ key  (complex, no conj on the key here — v13 multiplies k not k*)
    pred = contract(memory, key, over=dj)
    u = write_beta * value - erase_beta * pred
    key_conj = key.conj()
    memory = memory + outer(u, key_conj, (di, dj))
    out = contract(memory, query, over=dj)
    return out, memory


def fused_chunk_step(
    queries: NamedTensor,
    keys: NamedTensor,
    values: NamedTensor,
    decay_gamma: NamedTensor,
    retrieval_phase: NamedTensor,
    routing_weights: NamedTensor,
    memory: NamedTensor,
    is_first_chunk: bool,
    *,
    K: Dim,
    B: Dim,
    H: Dim,
    C: Dim,
    d: Dim,
    px: Dim,
    Cs: Optional[Dim] = None,
) -> tuple[NamedTensor, NamedTensor]:
    """Semantic dual-form chunk step (v13 ``_fused_chunk_step``).

    queries/keys/values: (B, H, C, d, px)
    decay_gamma / phase / routing: (K, B, H, C)
    memory: (K, B, H, d, d, px) — we use di, dj = two aliases of d if needed.
    """
    Cs = Cs or Dim("Cs", C.size)
    # scores = Q · K*  (complex inner over d, two time axes)
    Tq, Tk = Dim("Tq", C.size), Dim("Tk", C.size)
    q = queries.alias(C, Tq)
    k = keys.alias(C, Tk)
    v = values.alias(C, Tk)
    # conjugate on k for Q·K*
    scores = contract(q, k.conj(), over=d)  # B H Tq Tk px

    # decay matrix per (K,B,H)
    g = decay_gamma.to(K, B, H, C).data
    Ksz, Bsz, Hsz, Csz = g.shape
    Dmat = decay_matrix(g.reshape(Ksz * Bsz * Hsz, Csz), Csz).reshape(Ksz, Bsz, Hsz, Csz, Csz)
    D = named(Dmat, (K, B, H, Tq, Tk))

    # D̃ = Σ_k α_k e^{iφ_k} D_k
    phase = retrieval_phase.to(K, B, H, C)
    # phase is on C; for broadcast onto Tq we alias
    phase_t = phase.alias(C, Tq)
    route_t = routing_weights.to(K, B, H, C).alias(C, Tq)
    cos_p = named(torch.cos(phase_t.data), phase_t.layout)
    sin_p = named(torch.sin(phase_t.data), phase_t.layout)
    # complex scale α e^{iφ} then * D, sum K
    # D is real; scale is (K,B,H,Tq)
    # D̃[Tq, Tk] = sum_K (α cos) D - i?  decay_real = ((α*cos).unsqueeze Tk) * D
    scale_r = route_t * cos_p
    scale_i = route_t * sin_p
    decay_real = (scale_r * D)
    decay_imag = (scale_i * D)
    # sum over K
    from ..ops import sum as nsum

    Dr = nsum(decay_real, over=K)
    Di = nsum(decay_imag, over=K)
    # weighted = scores * D̃ (complex)
    sr = named(scores.data[..., 0], scores.layout[:-1])
    si = named(scores.data[..., 1], scores.layout[:-1])
    wr = sr * Dr - si * Di
    wi = sr * Di + si * Dr
    # y = weighted @ V  (complex)
    vr = named(v.data[..., 0], v.layout[:-1])
    vi = named(v.data[..., 1], v.layout[:-1])
    # complex (wr+i wi) @ (vr+i vi) = wr@vr - wi@vi + i(wr@vi + wi@vr)
    out_r = contract(wr, vr, over=Tk) - contract(wi, vi, over=Tk)
    out_i = contract(wr, vi, over=Tk) + contract(wi, vr, over=Tk)
    out = named(
        torch.stack([out_r.data, out_i.data], dim=-1),
        out_r.layout + (px,),
        queries.policy,
    )

    log_decay = torch.log(g + 1e-6)
    cumulative = torch.exp(torch.cumsum(log_decay, dim=-1))
    if not is_first_chunk:
        # carried = (S @ Q^T)^T = Q @ S^T  — v13 uses S @ Q.T then transpose
        # memory (K,B,H,di,dj,px), query (B,H,Tq,d,px)
        di, dj = Dim("di", d.size), Dim("dj", d.size)
        # Treat memory last two d's as di, dj — caller should pass that layout.
        # If memory is (K,B,H,d,d,px) the two d's are the SAME object and we cannot
        # distinguish them. Require memory layout (K,B,H,di,dj,px).
        mem = memory
        qq = q  # B H Tq d px  — d must match dj
        # Use raw torch for the carried read to keep the v13 (S @ Q.T).T form
        mr, mi = mem.data[..., 0], mem.data[..., 1]
        qr, qi = queries.data[..., 0], queries.data[..., 1]
        # queries: B H C d; unsqueeze K
        qr_s, qi_s = qr.unsqueeze(0), qi.unsqueeze(0)
        carried_r = (mr @ qr_s.transpose(-1, -2) - mi @ qi_s.transpose(-1, -2)).transpose(-1, -2)
        carried_i = (mr @ qi_s.transpose(-1, -2) + mi @ qr_s.transpose(-1, -2)).transpose(-1, -2)
        comb_r = routing_weights.to(K, B, H, C).data * torch.cos(phase.data) * cumulative
        comb_i = routing_weights.to(K, B, H, C).data * torch.sin(phase.data) * cumulative
        routed_r = (carried_r * comb_r.unsqueeze(-1) - carried_i * comb_i.unsqueeze(-1)).sum(0)
        routed_i = (carried_r * comb_i.unsqueeze(-1) + carried_i * comb_r.unsqueeze(-1)).sum(0)
        out = named(
            out.data + torch.stack([routed_r, routed_i], dim=-1),
            out.layout,
            out.policy,
        )

    # state write: last row of D * V, then V @ K*
    decay_last = Dmat[:, :, :, -1, :]  # K B H Tk
    vr = values.data[..., 0].unsqueeze(0)
    vi = values.data[..., 1].unsqueeze(0)
    wr = vr * decay_last.unsqueeze(-1)
    wi = vi * decay_last.unsqueeze(-1)
    kr = keys.data[..., 0].unsqueeze(0)
    ki = keys.data[..., 1].unsqueeze(0)
    # (decayed V).T @ K  (v13: write.transpose(-1,-2) @ key)
    st_r = wr.transpose(-1, -2) @ kr + wi.transpose(-1, -2) @ ki
    st_i = wi.transpose(-1, -2) @ kr - wr.transpose(-1, -2) @ ki
    state_chunk = torch.stack([st_r, st_i], dim=-1)
    total = cumulative[:, :, :, -1]
    memory_new = memory.data * total[..., None, None, None] + state_chunk
    return out.alias(Tq, C) if out.has(Tq) else out, named(memory_new, memory.layout, memory.policy)
